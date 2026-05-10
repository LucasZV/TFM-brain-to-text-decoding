import subprocess
from pathlib import Path

BASE_DIR = Path.home() / "projects/mamba_experiment/language_model_text"
CORPUS_DIR = BASE_DIR / "corpus"
MODEL_DIR = BASE_DIR / "kenlm" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_TXT = CORPUS_DIR / "train.txt"

LMPLZ = Path.home() / "projects/mamba_experiment/language_model_text/kenlm_src/build/bin/lmplz"
BUILD_BINARY = Path.home() / "projects/mamba_experiment/language_model_text/kenlm_src/build/bin/build_binary"

if not TRAIN_TXT.exists():
    raise FileNotFoundError(f"No se encontró el corpus de entrenamiento: {TRAIN_TXT}")

if not LMPLZ.exists():
    raise FileNotFoundError(f"No se encontró lmplz en: {LMPLZ}")

if not BUILD_BINARY.exists():
    raise FileNotFoundError(f"No se encontró build_binary en: {BUILD_BINARY}")

jobs = [
    ("text_3gram", 3),
    ("text_5gram", 5),
]

for model_name, order in jobs:
    arpa_path = MODEL_DIR / f"{model_name}.arpa"
    binary_path = MODEL_DIR / f"{model_name}.binary"

    print(f"\nTraining {order}-gram LM...")
    cmd_arpa = f'"{LMPLZ}" -o {order} < "{TRAIN_TXT}" > "{arpa_path}"'
    subprocess.run(cmd_arpa, shell=True, check=True)

    print(f"Building binary for {model_name}...")
    subprocess.run(
        [str(BUILD_BINARY), str(arpa_path), str(binary_path)],
        check=True,
    )

print("\nDone.")
print("Generated files:")
for p in sorted(MODEL_DIR.glob("*")):
    print(p)
