import os
import json
import math
import argparse
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

from convmamba_phoneme_trainer import BrainToPhonemeDecoderTrainer
from phoneme_utils import (
    PHONE_VOCAB,
    ctc_greedy_decode_phoneme_ids,
    calculate_error_rate,
    phoneme_seq_to_string,
)


def logsumexp(a, b):
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def tokens_to_phoneme_seq(token_ids, phone_vocab):
    seq = []
    for idx in token_ids:
        if idx == 0:
            continue
        seq.append(phone_vocab[idx])
    return seq


def phoneme_seq_to_lm_text(phone_seq):
    # Convierte ["DH", "AH", "SIL", "K", "AE", "T"] a "DH AH SIL K AE T"
    return " ".join(phone_seq).strip()


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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

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


def compute_per(pred_phone_seq, true_phone_seq):
    phone_ed = calculate_error_rate(true_phone_seq, pred_phone_seq)
    phone_len = max(len(true_phone_seq), 1)
    return phone_ed, phone_len


@torch.inference_mode()
def evaluate_rescoring(
    trainer,
    beam_width=10,
    topk=10,
    use_lm=False,
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
        lm_device = trainer.device if trainer.device.type == "cuda" else torch.device("cpu")
        lm_model, lm_tokenizer = build_text_lm(
            model_name=lm_model_name,
            device=lm_device,
        )

    summary = {
        "greedy": {
            "total_phone_edit_distance": 0,
            "total_phone_length": 0,
        },
        "beam": {
            "total_phone_edit_distance": 0,
            "total_phone_length": 0,
        },
        "beam_lm": {
            "total_phone_edit_distance": 0,
            "total_phone_length": 0,
        },
    }

    examples = []

    total_batches = len(trainer.val_loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    print(f"Starting phoneme evaluation on {total_batches} validation batches...")
    print(f"Beam width: {beam_width} | topk: {topk} | use_lm: {use_lm}")
    if use_lm:
        print(
            f"LM model: {lm_model_name} | lm_alpha: {lm_alpha} | "
            f"lm_length_penalty: {lm_length_penalty}"
        )

    for batch_idx, batch in enumerate(trainer.val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if batch_idx % 10 == 0:
            print(f"[Progress] Batch {batch_idx + 1}/{total_batches}")

        features = batch["input_features"].to(trainer.device)
        labels = batch["seq_class_ids"].to(trainer.device)
        n_time_steps = batch["n_time_steps"].to(trainer.device)
        phone_seq_lens = batch["phone_seq_lens"].to(trainer.device)
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
                    (phone_seq_lens > 0) &
                    (adjusted_lens >= phone_seq_lens)
                )

                if not valid_mask.any():
                    continue

                valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

                features = features[valid_mask]
                labels = labels[valid_mask]
                adjusted_lens = adjusted_lens[valid_mask]
                phone_seq_lens = phone_seq_lens[valid_mask]
                day_indicies = day_indicies[valid_mask]

                logits = trainer.model(features, day_indicies)
                log_probs = logits.log_softmax(dim=2)

        for i in range(logits.shape[0]):
            original_idx = valid_indices[i].item()

            true_len = int(phone_seq_lens[i].item())
            true_phone_ids = labels[i, :true_len].detach().cpu().numpy().tolist()
            true_phone_seq = [PHONE_VOCAB[int(x)] for x in true_phone_ids if int(x) != 0]

            lp = log_probs[i, :adjusted_lens[i], :].detach().cpu().float().numpy()

            greedy_ids = torch.argmax(
                log_probs[i, :adjusted_lens[i], :], dim=-1
            ).cpu().numpy().tolist()
            greedy_phone_seq = ctc_greedy_decode_phoneme_ids(greedy_ids)

            beam_hyps = ctc_prefix_beam_search(
                lp,
                beam_width=beam_width,
                blank_id=0,
                topk=topk,
            )

            beam_phone_seqs = []
            beam_scores = []
            for token_tuple, score in beam_hyps:
                hyp_phone_seq = tokens_to_phoneme_seq(token_tuple, PHONE_VOCAB)
                if len(hyp_phone_seq) == 0:
                    continue
                beam_phone_seqs.append(hyp_phone_seq)
                beam_scores.append(score)

            if len(beam_phone_seqs) == 0:
                beam_phone_seq = greedy_phone_seq
                beam_lm_phone_seq = greedy_phone_seq
                lm_scores = []
                total_scores = []
            else:
                beam_phone_seq = beam_phone_seqs[0]

                if use_lm:
                    lm_texts = [phoneme_seq_to_lm_text(seq) for seq in beam_phone_seqs]
                    lm_scores = score_hypotheses_with_lm(
                        lm_model,
                        lm_tokenizer,
                        lm_texts,
                        device=lm_model.device,
                        length_penalty=lm_length_penalty,
                    )

                    total_scores = [
                        ctc_s + lm_alpha * lm_s
                        for ctc_s, lm_s in zip(beam_scores, lm_scores)
                    ]
                    best_idx = int(np.argmax(total_scores))
                    beam_lm_phone_seq = beam_phone_seqs[best_idx]
                else:
                    lm_scores = []
                    total_scores = []
                    beam_lm_phone_seq = beam_phone_seq

            for key, pred_phone_seq in [
                ("greedy", greedy_phone_seq),
                ("beam", beam_phone_seq),
                ("beam_lm", beam_lm_phone_seq),
            ]:
                phone_ed, phone_len = compute_per(pred_phone_seq, true_phone_seq)
                summary[key]["total_phone_edit_distance"] += phone_ed
                summary[key]["total_phone_length"] += phone_len

            example = {
                "batch_idx": batch_idx,
                "sample_idx_in_batch": int(original_idx),
                "true_phonemes": phoneme_seq_to_string(true_phone_seq),
                "greedy_phonemes": phoneme_seq_to_string(greedy_phone_seq),
                "beam_phonemes": phoneme_seq_to_string(beam_phone_seq),
                "beam_lm_phonemes": phoneme_seq_to_string(beam_lm_phone_seq),
                "raw_text": batch["raw_text"][original_idx] if "raw_text" in batch else "",
            }

            if len(beam_phone_seqs) > 0:
                example["nbest"] = []
                for j, seq in enumerate(beam_phone_seqs):
                    row = {
                        "phonemes": phoneme_seq_to_string(seq),
                        "lm_text": phoneme_seq_to_lm_text(seq),
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
            "PER": summary[key]["total_phone_edit_distance"] / max(summary[key]["total_phone_length"], 1),
            "total_phone_length": summary[key]["total_phone_length"],
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
    parser.add_argument("--lm_model_name", type=str, default="distilgpt2")
    parser.add_argument("--lm_alpha", type=float, default=0.3)
    parser.add_argument("--lm_length_penalty", type=float, default=0.0)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output_json", type=str, default="rescoring_results/phoneme_beam_eval.json")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.args)
    cfg["mode"] = "eval"
    cfg["init_from_checkpoint"] = True
    cfg["init_checkpoint_path"] = args.checkpoint

    trainer = BrainToPhonemeDecoderTrainer(cfg)

    results = evaluate_rescoring(
        trainer=trainer,
        beam_width=args.beam_width,
        topk=args.topk,
        use_lm=args.use_lm,
        lm_model_name=args.lm_model_name,
        lm_alpha=args.lm_alpha,
        lm_length_penalty=args.lm_length_penalty,
        max_batches=args.max_batches,
        save_examples_path=args.output_json,
    )

    print("\n=== PHONEME DECODING RESULTS ===")
    for key in ["greedy", "beam", "beam_lm"]:
        print(
            f"{key:8s} | "
            f"PER: {results[key]['PER']:.4f}"
        )


if __name__ == "__main__":
    main()