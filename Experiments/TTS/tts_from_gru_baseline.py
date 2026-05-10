import os
import pandas as pd
from TTS.api import TTS

CSV_PATH = "PATH CSV PREDICCIONES"
OUT_DIR = "PATH DE SALIDA"
MODEL_NAME = "tts_models/en/jenny/jenny"
MAX_EXAMPLES = 20

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

text_col = None
for candidate in ["text", "pred_text", "predicted_text", "sentence", "pred_sentence"]:
    if candidate in df.columns:
        text_col = candidate
        break

if text_col is None:
    raise ValueError(f"No text column found. Columns are: {list(df.columns)}")

tts = TTS(model_name=MODEL_NAME)

for i, row in df.head(MAX_EXAMPLES).iterrows():
    text = str(row[text_col]).strip()
    if not text or text.lower() == "nan":
        continue

    wav_path = os.path.join(OUT_DIR, f"example_{i:03d}.wav")
    txt_path = os.path.join(OUT_DIR, f"example_{i:03d}.txt")

    tts.tts_to_file(text=text, file_path=wav_path)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"Saved {wav_path}")

print(f"\nDone. Outputs in: {OUT_DIR}")
