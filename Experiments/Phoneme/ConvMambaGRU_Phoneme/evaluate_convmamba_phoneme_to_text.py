import os
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import editdistance
from omegaconf import OmegaConf

import lm_decoder

from convmamba_phoneme_trainer import BrainToPhonemeDecoderTrainer


def normalize_text(text):
    text = text.lower().strip()
    text = text.replace(">", "")
    text = " ".join(text.split())
    return text


def compute_text_metrics(pred_text, true_text):
    pred_text = normalize_text(pred_text)
    true_text = normalize_text(true_text)

    char_ed = editdistance.eval(list(pred_text), list(true_text))
    char_len = max(len(true_text), 1)

    pred_words = pred_text.split()
    true_words = true_text.split()
    word_ed = editdistance.eval(pred_words, true_words)
    word_len = max(len(true_words), 1)

    return {
        "char_ed": char_ed,
        "char_len": char_len,
        "word_ed": word_ed,
        "word_len": word_len,
    }


def build_lm_decoder(
    model_path,
    max_active=7000,
    min_active=200,
    beam=17.0,
    lattice_beam=8.0,
    acoustic_scale=1.5,
    ctc_blank_skip_threshold=1.0,
    length_penalty=0.0,
    nbest=1,
):
    decode_opts = lm_decoder.DecodeOptions(
        max_active,
        min_active,
        beam,
        lattice_beam,
        acoustic_scale,
        ctc_blank_skip_threshold,
        length_penalty,
        nbest,
    )

    tlg_path = os.path.join(model_path, "TLG.fst")
    words_path = os.path.join(model_path, "words.txt")
    g_path = os.path.join(model_path, "G.fst")
    rescore_g_path = os.path.join(model_path, "G_no_prune.fst")

    if not os.path.exists(rescore_g_path):
        rescore_g_path = ""
        g_path = ""

    if not os.path.exists(tlg_path):
        raise ValueError(f"TLG file not found at {tlg_path}")
    if not os.path.exists(words_path):
        raise ValueError(f"words file not found at {words_path}")

    decode_resource = lm_decoder.DecodeResource(
        tlg_path,
        g_path,
        rescore_g_path,
        words_path,
        "",
    )
    decoder = lm_decoder.BrainSpeechDecoder(decode_resource, decode_opts)
    return decoder


def update_ngram_params(
    ngram_decoder,
    max_active=7000,
    min_active=200,
    beam=17.0,
    lattice_beam=8.0,
    acoustic_scale=1.5,
    ctc_blank_skip_threshold=1.0,
    length_penalty=0.0,
    nbest=100,
):
    decode_opts = lm_decoder.DecodeOptions(
        max_active,
        min_active,
        beam,
        lattice_beam,
        acoustic_scale,
        ctc_blank_skip_threshold,
        length_penalty,
        nbest,
    )
    ngram_decoder.SetOpt(decode_opts)


@torch.inference_mode()
def evaluate_phoneme_to_text(
    trainer,
    lm_path,
    beam=17.0,
    lattice_beam=8.0,
    max_active=7000,
    min_active=200,
    acoustic_scale=0.3,
    ctc_blank_skip_threshold=1.0,
    length_penalty=0.0,
    nbest=1,
    blank_penalty=9.0,
    rescore=False,
    max_batches=None,
    output_json=None,
):
    trainer.model.eval()

    decoder = build_lm_decoder(
        model_path=lm_path,
        max_active=max_active,
        min_active=min_active,
        beam=beam,
        lattice_beam=lattice_beam,
        acoustic_scale=acoustic_scale,
        ctc_blank_skip_threshold=ctc_blank_skip_threshold,
        length_penalty=length_penalty,
        nbest=nbest,
    )

    update_ngram_params(
        decoder,
        max_active=max_active,
        min_active=min_active,
        beam=beam,
        lattice_beam=lattice_beam,
        acoustic_scale=acoustic_scale,
        ctc_blank_skip_threshold=ctc_blank_skip_threshold,
        length_penalty=length_penalty,
        nbest=nbest,
    )

    total_batches = len(trainer.val_loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    print(f"Starting offline phoneme-to-text evaluation on {total_batches} validation batches...")
    print(
        f"Decoder params | beam={beam} lattice_beam={lattice_beam} "
        f"acoustic_scale={acoustic_scale} blank_penalty={blank_penalty} "
        f"nbest={nbest} rescore={rescore}"
    )

    summary = {
        "total_char_ed": 0,
        "total_char_len": 0,
        "total_word_ed": 0,
        "total_word_len": 0,
        "n_examples": 0,
    }

    examples = []
    start_eval = time.time()

    for batch_idx, batch in enumerate(trainer.val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if batch_idx % 10 == 0:
            print(f"[Progress] Batch {batch_idx + 1}/{total_batches}")

        features = batch["input_features"].to(trainer.device)
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
                adjusted_lens = adjusted_lens[valid_mask]
                day_indicies = day_indicies[valid_mask]

                logits = trainer.model(features, day_indicies)
                log_probs = logits.log_softmax(dim=2).detach().cpu().float().numpy()

        valid_indices = valid_indices.tolist()
        if isinstance(valid_indices, int):
            valid_indices = [valid_indices]

        for i in range(log_probs.shape[0]):
            original_idx = valid_indices[i]

            decoder.Reset()
            lp = log_probs[i, :adjusted_lens[i], :]

            lm_decoder.DecodeNumpy(
                decoder,
                lp.astype(np.float32),
                np.zeros_like(lp, dtype=np.float32),
                np.log(blank_penalty),
            )

            decoder.FinishDecoding()

            if rescore:
                try:
                    decoder.Rescore()
                except Exception as e:
                    print(f"Warning: Rescore failed on batch {batch_idx}, sample {i}: {e}")

            results = decoder.result()

            if len(results) > 0:
                pred_text = normalize_text(results[0].sentence)
            else:
                pred_text = ""

            if "raw_text" in batch:
                true_text = normalize_text(batch["raw_text"][original_idx])
            else:
                true_text = ""

            metrics = compute_text_metrics(pred_text, true_text)

            summary["total_char_ed"] += metrics["char_ed"]
            summary["total_char_len"] += metrics["char_len"]
            summary["total_word_ed"] += metrics["word_ed"]
            summary["total_word_len"] += metrics["word_len"]
            summary["n_examples"] += 1

            ex = {
                "batch_idx": batch_idx,
                "sample_idx_in_batch": int(original_idx),
                "true_text": true_text,
                "pred_text": pred_text,
                "char_ed": metrics["char_ed"],
                "char_len": metrics["char_len"],
                "word_ed": metrics["word_ed"],
                "word_len": metrics["word_len"],
            }

            if nbest > 1 and len(results) > 0:
                ex["nbest"] = []
                for hyp in results[:nbest]:
                    row = {
                        "text": hyp.sentence.strip(),
                    }
                    if hasattr(hyp, "ac_score"):
                        row["ac_score"] = float(hyp.ac_score)
                    if hasattr(hyp, "lm_score"):
                        row["lm_score"] = float(hyp.lm_score)
                    ex["nbest"].append(row)

            examples.append(ex)

    elapsed = time.time() - start_eval

    cer = summary["total_char_ed"] / max(summary["total_char_len"], 1)
    wer = summary["total_word_ed"] / max(summary["total_word_len"], 1)

    results = {
        "CER": float(cer),
        "WER": float(wer),
        "n_examples": int(summary["n_examples"]),
        "total_char_len": int(summary["total_char_len"]),
        "total_word_len": int(summary["total_word_len"]),
        "elapsed_sec": float(elapsed),
    }

    if output_json is not None:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "results": results,
                    "examples": examples,
                    "config": {
                        "lm_path": lm_path,
                        "beam": beam,
                        "lattice_beam": lattice_beam,
                        "max_active": max_active,
                        "min_active": min_active,
                        "acoustic_scale": acoustic_scale,
                        "ctc_blank_skip_threshold": ctc_blank_skip_threshold,
                        "length_penalty": length_penalty,
                        "nbest": nbest,
                        "blank_penalty": blank_penalty,
                        "rescore": rescore,
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
    parser.add_argument("--lm_path", type=str, required=True, help="Path to WFST/decoder LM folder")

    parser.add_argument("--beam", type=float, default=17.0)
    parser.add_argument("--lattice_beam", type=float, default=8.0)
    parser.add_argument("--max_active", type=int, default=7000)
    parser.add_argument("--min_active", type=int, default=200)
    parser.add_argument("--acoustic_scale", type=float, default=0.3)
    parser.add_argument("--ctc_blank_skip_threshold", type=float, default=1.0)
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument("--nbest", type=int, default=1)
    parser.add_argument("--blank_penalty", type=float, default=9.0)
    parser.add_argument("--rescore", action="store_true")

    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output_json", type=str, default="rescoring_results/phoneme_to_text_eval.json")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.args)
    cfg["mode"] = "eval"
    cfg["init_from_checkpoint"] = True
    cfg["init_checkpoint_path"] = args.checkpoint

    trainer = BrainToPhonemeDecoderTrainer(cfg)

    results = evaluate_phoneme_to_text(
        trainer=trainer,
        lm_path=args.lm_path,
        beam=args.beam,
        lattice_beam=args.lattice_beam,
        max_active=args.max_active,
        min_active=args.min_active,
        acoustic_scale=args.acoustic_scale,
        ctc_blank_skip_threshold=args.ctc_blank_skip_threshold,
        length_penalty=args.length_penalty,
        nbest=args.nbest,
        blank_penalty=args.blank_penalty,
        rescore=args.rescore,
        max_batches=args.max_batches,
        output_json=args.output_json,
    )

    print("\n=== PHONEME -> TEXT RESULTS ===")
    print(f"WER: {results['WER']:.4f}")
    print(f"Examples: {results['n_examples']}")


if __name__ == "__main__":
    main()