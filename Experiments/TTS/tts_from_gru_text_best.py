import os
import json
from TTS.api import TTS

JSON_PATH = "PATH JSON PREDICCIONES"
OUT_DIR = "PATH SALIDA"
MODEL_NAME = "tts_models/en/jenny/jenny"
MAX_EXAMPLES = 20

os.makedirs(OUT_DIR, exist_ok=True)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

tts = TTS(model_name=MODEL_NAME)

examples = data["examples"][:MAX_EXAMPLES]

for i, ex in enumerate(examples):
    true_text = ex.get("true_text", "").strip()
    pred_text = ex.get("beam_lm_text", "").strip()

    if not pred_text:
        continue

    wav_path = os.path.join(OUT_DIR, f"example_{i:03d}.wav")
    txt_path = os.path.join(OUT_DIR, f"example_{i:03d}.txt")

    tts.tts_to_file(text=pred_text, file_path=wav_path)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("TRUE TEXT:\n")
        f.write(true_text + "\n\n")
        f.write("PREDICTED TEXT:\n")
        f.write(pred_text + "\n")

    print(f"Saved {wav_path}")

print(f"\nDone. Outputs in: {OUT_DIR}")
