from pathlib import Path
import kenlm

BASE_DIR = Path.home() / "projects/mamba_experiment/language_model_text"
MODEL_DIR = BASE_DIR / "kenlm" / "models"


def load_kenlm_model(model_name="text_5gram.binary"):
    model_path = MODEL_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo KenLM: {model_path}")
    return kenlm.Model(str(model_path))


def score_text(model, text: str, bos: bool = True, eos: bool = True) -> float:
    text = " ".join(text.strip().split())
    if not text:
        return float("-inf")
    return float(model.score(text, bos=bos, eos=eos))


def score_texts(model, texts, bos: bool = True, eos: bool = True):
    return [score_text(model, t, bos=bos, eos=eos) for t in texts]


if __name__ == "__main__":
    model = load_kenlm_model("text_5gram.binary")

    examples = [
        "you can see the code at this point as well.",
        "you can se the cold at this point as well.",
        "how does it keep the cost down?",
        "how dos it keep the cost sin?",
    ]

    print("Testing KenLM scoring:\n")
    for text in examples:
        score = score_text(model, text)
        print(f"{score:12.4f} | {text}")
