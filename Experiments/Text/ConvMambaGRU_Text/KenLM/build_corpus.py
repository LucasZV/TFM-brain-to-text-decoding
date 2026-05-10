import json
from pathlib import Path
from collections import Counter

import h5py

DATASET_DIR = Path("/home/lucas/projects/nejm-brain-to-text/data/hdf5_data_final")
OUTPUT_BASE = Path("/home/lucas/projects/mamba_experiment/language_model_text")

CHAR_VOCAB = [
    "<blank>",
    " ",
    "'",
    ",",
    ".",
    "?",
] + [chr(i) for i in range(ord("a"), ord("z") + 1)]

CHAR_SET = set(CHAR_VOCAB)
CHAR_SET.discard("<blank>")


def ascii_array_to_text(ascii_array) -> str:
    chars = []
    for x in ascii_array:
        x = int(x)
        if x == 0:
            continue
        chars.append(chr(x))
    return "".join(chars)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    filtered = []
    for ch in text:
        if ch in CHAR_SET:
            filtered.append(ch)
    text = "".join(filtered)
    text = " ".join(text.split())
    return text


def collect_sentences_from_file(h5_path: Path):
    sentences = []
    if not h5_path.exists():
        return sentences

    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            g = f[key]
            if "transcription" not in g:
                continue

            raw = ascii_array_to_text(g["transcription"][:])
            norm = normalize_text(raw)
            if norm:
                sentences.append(norm)

    return sentences


def main():
    corpus_dir = OUTPUT_BASE / "corpus"
    data_dir = OUTPUT_BASE / "data"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_sentences = []
    val_sentences = []

    split_counter = {
        "train_files_found": 0,
        "val_files_found": 0,
        "train_sentences": 0,
        "val_sentences": 0,
    }

    sessions = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])

    for session_dir in sessions:
        train_file = session_dir / "data_train.hdf5"
        val_file = session_dir / "data_val.hdf5"

        if train_file.exists():
            split_counter["train_files_found"] += 1
            sents = collect_sentences_from_file(train_file)
            train_sentences.extend(sents)

        if val_file.exists():
            split_counter["val_files_found"] += 1
            sents = collect_sentences_from_file(val_file)
            val_sentences.extend(sents)

    all_sentences = train_sentences + val_sentences

    split_counter["train_sentences"] = len(train_sentences)
    split_counter["val_sentences"] = len(val_sentences)

    (corpus_dir / "train.txt").write_text(
        "\n".join(train_sentences) + ("\n" if train_sentences else ""),
        encoding="utf-8",
    )
    (corpus_dir / "val.txt").write_text(
        "\n".join(val_sentences) + ("\n" if val_sentences else ""),
        encoding="utf-8",
    )
    (corpus_dir / "all.txt").write_text(
        "\n".join(all_sentences) + ("\n" if all_sentences else ""),
        encoding="utf-8",
    )

    char_counter = Counter()
    word_counter = Counter()

    for sent in all_sentences:
        char_counter.update(list(sent))
        word_counter.update(sent.split())

    stats = {
        "dataset_dir": str(DATASET_DIR),
        "train_files_found": split_counter["train_files_found"],
        "val_files_found": split_counter["val_files_found"],
        "train_sentences": len(train_sentences),
        "val_sentences": len(val_sentences),
        "all_sentences": len(all_sentences),
        "unique_words": len(word_counter),
        "unique_chars": len(char_counter),
        "char_vocab_used": sorted(char_counter.keys()),
        "top_50_words": word_counter.most_common(50),
        "top_50_chars": char_counter.most_common(50),
    }

    with open(data_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    with open(data_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "char_vocab": CHAR_VOCAB,
                "char_to_id": {ch: i for i, ch in enumerate(CHAR_VOCAB)},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Corpus built successfully.")
    print(f"Train sentences: {len(train_sentences)}")
    print(f"Val sentences:   {len(val_sentences)}")
    print(f"All sentences:   {len(all_sentences)}")
    print(f"Unique words:    {len(word_counter)}")
    print(f"Unique chars:    {len(char_counter)}")
    print(f"Saved train corpus to: {corpus_dir / 'train.txt'}")
    print(f"Saved val corpus to:   {corpus_dir / 'val.txt'}")
    print(f"Saved all corpus to:   {corpus_dir / 'all.txt'}")
    print(f"Saved stats to:        {data_dir / 'stats.json'}")
    print(f"Saved vocab to:        {data_dir / 'vocab.json'}")


if __name__ == "__main__":
    main()
