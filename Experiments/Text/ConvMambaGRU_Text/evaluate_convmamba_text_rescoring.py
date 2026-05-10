import os
import json
import math
import argparse
from collections import defaultdict

import numpy as np
import torch
import editdistance
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

import kenlm

from convmamba_text_trainer import BrainToTextDecoderTrainer
from text_utils import (
    CHAR_VOCAB,
    ctc_greedy_decode,
    ascii_array_to_text,
    char_ids_to_text,
    normalize_text,
)


def build_kenlm(model_path):
    return kenlm.Model(model_path)


def score_hypotheses_with_kenlm(model, hypotheses, normalize_by_length=False):
    scores = []
    for text in hypotheses:
        text = " ".join(text.strip().split())
        if not text:
            scores.append(float("-inf"))
            continue

        score = float(model.score(text, bos=True, eos=True))

        if normalize_by_length:
            n = max(len(text.split()), 1)
            score = score / n

        scores.append(score)

    return scores


def logsumexp(a, b):
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def tokens_to_text(token_ids):
    ids = []
    for idx in token_ids:
        idx = int(idx)
        if idx == 0:
            continue
        ids.append(idx)
    return normalize_text(char_ids_to_text(ids))


def ctc_prefix_beam_search(log_probs, beam_width=10, blank_id=0, topk=10):
    """
    Prefix beam search for CTC.

    Args:
        log_probs: numpy array [T, V] in log-prob space
        beam_width: beam width
        blank_id: blank token id
        topk: number of final hypotheses to return

    Returns:
        list of tuples: [(prefix_token_tuple, score), ...] sorted descending
    """
    T, V = log_probs.shape
    beam = {(): (0.0, -float("inf"))}

    for t in range(T):
        next_beam = defaultdict(lambda: (-float("inf"), -float("inf")))

        for prefix, (p_blank, p_nonblank) in beam.items():
            for c in range(V):
                p = float(log_probs[t, c])

                if c == blank_id:
                    nb_blank, nb_nonblank = next_beam[prefix]
                    nb_blank = logsumexp(nb_blank, p_blank + p)
                    nb_blank = logsumexp(nb_blank, p_nonblank + p)
                    next_beam[prefix] = (nb_blank, nb_nonblank)
                    continue

                end_t = prefix[-1] if len(prefix) > 0 else None
                new_prefix = prefix + (c,)

                if c == end_t:
                    nb_blank, nb_nonblank = next_beam[prefix]
                    nb_nonblank = logsumexp(nb_nonblank, p_nonblank + p)
                    next_beam[prefix] = (nb_blank, nb_nonblank)

                    nb_blank, nb_nonblank = next_beam[new_prefix]
                    nb_nonblank = logsumexp(nb_nonblank, p_blank + p)
                    next_beam[new_prefix] = (nb_blank, nb_nonblank)
                else:
                    nb_blank, nb_nonblank = next_beam[new_prefix]
                    nb_nonblank = logsumexp(nb_nonblank, p_blank + p)
                    nb_nonblank = logsumexp(nb_nonblank, p_nonblank + p)
                    next_beam[new_prefix] = (nb_blank, nb_nonblank)

        beam = dict(
            sorted(
                next_beam.items(),
                key=lambda x: logsumexp(x[1][0], x[1][1]),
                reverse=True,
            )[:beam_width]
        )

    final_beam = []
    for prefix, (p_blank, p_nonblank) in beam.items():
        score = logsumexp(p_blank, p_nonblank)
        final_beam.append((prefix, score))

    final_beam.sort(key=lambda x: x[1], reverse=True)
    return final_beam[:topk]


@torch.inference_mode()
def build_text_lm(model_name="distilgpt2", device="cuda"):
    extra_kwargs = {}
    
    if model_name == "facebook/MobileLLM-Pro":
        extra_kwargs["trust_remote_code"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, **extra_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, **extra_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def score_hypotheses_with_lm(model, tokenizer, hypotheses, device="cuda", length_penalty=0.0):
    if len(hypotheses) == 0:
        return []

    inputs = tokenizer(
        hypotheses,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    scores = []
    for i in range(input_ids.shape[0]):
        n_tokens = int(attention_mask[i].sum().item())
        score = 0.0
        for t in range(1, n_tokens):
            score += log_probs[i, t - 1, input_ids[i, t]].item()
        score -= n_tokens * length_penalty
        scores.append(score)

    return scores


def compute_text_metrics(pred_text, true_text):
    pred_text = normalize_text(pred_text)
    true_text = normalize_text(true_text)

    char_ed = editdistance.eval(list(pred_text), list(true_text))
    char_len = max(len(true_text), 1)

    pred_words = pred_text.split()
    true_words = true_text.split()
    word_ed = editdistance.eval(pred_words, true_words)
    word_len = max(len(true_words), 1)

    return char_ed, char_len, word_ed, word_len


@torch.inference_mode()
def evaluate_rescoring(
    trainer,
    beam_width=10,
    topk=10,
    use_lm=False,
    lm_backend="hf",
    kenlm_model_path=None,
    kenlm_normalize=False,
    lm_model_name="distilgpt2",
    lm_alpha=0.3,
    lm_length_penalty=0.0,
    max_batches=None,
    save_examples_path=None,
):
    trainer.model.eval()

    lm_model = None
    lm_tokenizer = None

    if use_lm:
        if lm_backend == "hf":
            lm_device = trainer.device if trainer.device.type == "cuda" else torch.device("cpu")
            lm_model, lm_tokenizer = build_text_lm(
                model_name=lm_model_name,
                device=lm_device,
            )
        elif lm_backend == "kenlm":
            if kenlm_model_path is None:
                raise ValueError("For lm_backend='kenlm', --kenlm_model_path is required.")
            lm_model = build_kenlm(kenlm_model_path)
        else:
            raise ValueError(f"Unsupported lm_backend: {lm_backend}")

    summary = {
        "greedy": {
            "total_char_edit_distance": 0,
            "total_char_length": 0,
            "total_word_edit_distance": 0,
            "total_word_length": 0,
        },
        "beam": {
            "total_char_edit_distance": 0,
            "total_char_length": 0,
            "total_word_edit_distance": 0,
            "total_word_length": 0,
        },
        "beam_lm": {
            "total_char_edit_distance": 0,
            "total_char_length": 0,
            "total_word_edit_distance": 0,
            "total_word_length": 0,
        },
    }

    examples = []

    total_batches = len(trainer.val_loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    print(f"Starting text evaluation on {total_batches} validation batches...")
    print(f"Beam width: {beam_width} | topk: {topk} | use_lm: {use_lm}")
    if use_lm:
        if lm_backend == "hf":
            print(
                f"LM backend: {lm_backend} | LM model: {lm_model_name} | "
                f"lm_alpha: {lm_alpha} | lm_length_penalty: {lm_length_penalty}"
            )
        else:
            print(
                f"LM backend: {lm_backend} | KenLM model: {kenlm_model_path} | "
                f"lm_alpha: {lm_alpha} | kenlm_normalize: {kenlm_normalize}"
            )

    for batch_idx, batch in enumerate(trainer.val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if batch_idx % 10 == 0:
            print(f"[Progress] Batch {batch_idx + 1}/{total_batches}")

        features = batch["input_features"].to(trainer.device)
        labels = batch["char_seq_ids"].to(trainer.device)
        n_time_steps = batch["n_time_steps"].to(trainer.device)
        char_seq_lens = batch["char_seq_lens"].to(trainer.device)
        day_indicies = batch["day_indicies"].to(trainer.device)

        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                enabled=trainer.args["use_amp"] and trainer.device.type == "cuda",
                dtype=torch.bfloat16,
            ):
                features, n_time_steps = trainer.transform_data(features, n_time_steps, "val")
                adjusted_lens = (
                    (n_time_steps - trainer.args["model"]["patch_size"]) /
                    trainer.args["model"]["patch_stride"] + 1
                ).to(torch.int32)

                valid_mask = (
                    (adjusted_lens > 0) &
                    (char_seq_lens > 0) &
                    (adjusted_lens >= char_seq_lens)
                )

                if not valid_mask.any():
                    continue

                valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

                features = features[valid_mask]
                labels = labels[valid_mask]
                adjusted_lens = adjusted_lens[valid_mask]
                char_seq_lens = char_seq_lens[valid_mask]
                day_indicies = day_indicies[valid_mask]

                logits = trainer.model(features, day_indicies)
                log_probs = logits.log_softmax(dim=2)

        for i in range(logits.shape[0]):
            original_idx = valid_indices[i].item()

            if "raw_text" in batch:
                true_text = normalize_text(batch["raw_text"][original_idx])
            else:
                trans = batch["transcriptions"][original_idx]
                if isinstance(trans, torch.Tensor):
                    trans = trans.cpu().numpy()
                true_text = normalize_text(ascii_array_to_text(trans))

            lp = log_probs[i, :adjusted_lens[i], :].detach().cpu().float().numpy()

            greedy_ids = torch.argmax(
                log_probs[i, :adjusted_lens[i], :], dim=-1
            ).cpu().numpy().tolist()
            greedy_text = normalize_text(ctc_greedy_decode(greedy_ids))

            beam_hyps = ctc_prefix_beam_search(
                lp,
                beam_width=beam_width,
                blank_id=0,
                topk=topk,
            )

            beam_texts = []
            beam_scores = []
            for token_tuple, score in beam_hyps:
                hyp_text = tokens_to_text(token_tuple)
                if len(hyp_text) == 0:
                    continue
                beam_texts.append(hyp_text)
                beam_scores.append(score)

            if len(beam_texts) == 0:
                beam_text = greedy_text
                beam_lm_text = greedy_text
                lm_scores = []
                total_scores = []
            else:
                beam_text = beam_texts[0]

                if use_lm:
                    if lm_backend == "hf":
                        lm_scores = score_hypotheses_with_lm(
                            lm_model,
                            lm_tokenizer,
                            beam_texts,
                            device=lm_model.device,
                            length_penalty=lm_length_penalty,
                        )
                    elif lm_backend == "kenlm":
                        lm_scores = score_hypotheses_with_kenlm(
                            lm_model,
                            beam_texts,
                            normalize_by_length=kenlm_normalize,
                        )
                    else:
                        raise ValueError(f"Unsupported lm_backend: {lm_backend}")

                    total_scores = [
                        ctc_s + lm_alpha * lm_s
                        for ctc_s, lm_s in zip(beam_scores, lm_scores)
                    ]
                    best_idx = int(np.argmax(total_scores))
                    beam_lm_text = beam_texts[best_idx]
                else:
                    lm_scores = []
                    total_scores = []
                    beam_lm_text = beam_text

            for key, pred_text in [
                ("greedy", greedy_text),
                ("beam", beam_text),
                ("beam_lm", beam_lm_text),
            ]:
                char_ed, char_len, word_ed, word_len = compute_text_metrics(pred_text, true_text)
                summary[key]["total_char_edit_distance"] += char_ed
                summary[key]["total_char_length"] += char_len
                summary[key]["total_word_edit_distance"] += word_ed
                summary[key]["total_word_length"] += word_len

            example = {
                "batch_idx": batch_idx,
                "sample_idx_in_batch": int(original_idx),
                "true_text": true_text,
                "greedy_text": greedy_text,
                "beam_text": beam_text,
                "beam_lm_text": beam_lm_text,
            }

            if len(beam_texts) > 0:
                example["nbest"] = []
                for j, txt in enumerate(beam_texts):
                    row = {
                        "text": txt,
                        "ctc_score": float(beam_scores[j]),
                    }
                    if use_lm and j < len(lm_scores):
                        row["lm_score"] = float(lm_scores[j])
                        row["total_score"] = float(total_scores[j])
                    example["nbest"].append(row)

            examples.append(example)

    results = {}
    for key in ["greedy", "beam", "beam_lm"]:
        results[key] = {
            "CER": summary[key]["total_char_edit_distance"] / max(summary[key]["total_char_length"], 1),
            "WER": summary[key]["total_word_edit_distance"] / max(summary[key]["total_word_length"], 1),
            "total_char_length": summary[key]["total_char_length"],
            "total_word_length": summary[key]["total_word_length"],
        }

    if save_examples_path is not None:
        os.makedirs(os.path.dirname(save_examples_path), exist_ok=True)
        with open(save_examples_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "results": results,
                    "examples": examples,
                    "config": {
                        "beam_width": beam_width,
                        "topk": topk,
                        "use_lm": use_lm,
                        "lm_backend": lm_backend,
                        "kenlm_model_path": kenlm_model_path,
                        "kenlm_normalize": kenlm_normalize,
                        "lm_model_name": lm_model_name,
                        "lm_alpha": lm_alpha,
                        "lm_length_penalty": lm_length_penalty,
                        "max_batches": max_batches,
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args", type=str, required=True, help="Path to args yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best checkpoint")
    parser.add_argument("--beam_width", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--use_lm", action="store_true")
    parser.add_argument("--lm_backend", type=str, default="hf", choices=["hf", "kenlm"])
    parser.add_argument("--kenlm_model_path", type=str, default=None)
    parser.add_argument("--kenlm_normalize", action="store_true")
    parser.add_argument("--lm_model_name", type=str, default="distilgpt2")
    parser.add_argument("--lm_alpha", type=float, default=0.3)
    parser.add_argument("--lm_length_penalty", type=float, default=0.0)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output_json", type=str, default="rescoring_results/text_beam_eval.json")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.args)
    cfg["mode"] = "eval"
    cfg["init_from_checkpoint"] = True
    cfg["init_checkpoint_path"] = args.checkpoint

    trainer = BrainToTextDecoderTrainer(cfg)

    results = evaluate_rescoring(
        trainer=trainer,
        beam_width=args.beam_width,
        topk=args.topk,
        use_lm=args.use_lm,
        lm_backend=args.lm_backend,
        kenlm_model_path=args.kenlm_model_path,
        kenlm_normalize=args.kenlm_normalize,
        lm_model_name=args.lm_model_name,
        lm_alpha=args.lm_alpha,
        lm_length_penalty=args.lm_length_penalty,
        max_batches=args.max_batches,
        save_examples_path=args.output_json,
    )

    print("\n=== TEXT DECODING RESULTS ===")
    for key in ["greedy", "beam", "beam_lm"]:
        print(
            f"{key:8s} | "
            f"CER: {results[key]['CER']:.4f} | "
            f"WER: {results[key]['WER']:.4f}"
        )


if __name__ == "__main__":
    main()
