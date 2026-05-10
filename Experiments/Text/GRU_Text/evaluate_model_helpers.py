import re
import numpy as np
import editdistance


def normalize_text(text):
    text = text.lower().strip()
    text = text.replace(">", "")
    text = " ".join(text.split())
    return text


def remove_punctuation(sentence):
    sentence = re.sub(r"[^a-zA-Z\- ']", "", sentence)
    sentence = sentence.replace("- ", " ").lower()
    sentence = sentence.replace("--", "").lower()
    sentence = sentence.replace(" '", "'").lower()
    sentence = sentence.strip()
    sentence = " ".join(sentence.split())
    return sentence


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
        "cer": char_ed / char_len,
        "wer": word_ed / word_len,
    }


def summarize_results(results_dict):
    for key in ["greedy", "beam", "beam_lm"]:
        if key in results_dict:
            print(
                f"{key:8s} | "
                f"CER: {results_dict[key]['CER']:.4f} | "
                f"WER: {results_dict[key]['WER']:.4f}"
            )