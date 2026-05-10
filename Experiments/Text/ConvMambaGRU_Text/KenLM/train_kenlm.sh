#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="$HOME/projects/mamba_experiment/language_model_text"
CORPUS_DIR="$BASE_DIR/corpus"
KENLM_DIR="$BASE_DIR/kenlm"
MODEL_DIR="$KENLM_DIR/models"

mkdir -p "$MODEL_DIR"

TRAIN_TXT="$CORPUS_DIR/train.txt"

echo "Training 3-gram LM..."
lmplz -o 3 < "$TRAIN_TXT" > "$MODEL_DIR/text_3gram.arpa"
build_binary "$MODEL_DIR/text_3gram.arpa" "$MODEL_DIR/text_3gram.binary"

echo "Training 5-gram LM..."
lmplz -o 5 < "$TRAIN_TXT" > "$MODEL_DIR/text_5gram.arpa"
build_binary "$MODEL_DIR/text_5gram.arpa" "$MODEL_DIR/text_5gram.binary"

echo "Done."
echo "Generated files:"
ls -lh "$MODEL_DIR"
