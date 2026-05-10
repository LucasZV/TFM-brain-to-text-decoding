#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def safe_get(d: dict[str, Any], *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_result_file(json_path: Path) -> dict[str, Any] | None:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] No se pudo leer {json_path}: {e}")
        return None

    results = data.get("results", {})
    config = data.get("config", {})

    row = {
        "file": str(json_path),
        "folder": str(json_path.parent),
        "filename": json_path.name,
        "parent_folder": json_path.parent.name,
        "beam_width": config.get("beam_width"),
        "topk": config.get("topk"),
        "use_lm": config.get("use_lm"),
        "lm_backend": config.get("lm_backend"),
        "lm_model_name": config.get("lm_model_name"),
        "kenlm_model_path": config.get("kenlm_model_path"),
        "kenlm_normalize": config.get("kenlm_normalize"),
        "lm_alpha": config.get("lm_alpha"),
        "lm_length_penalty": config.get("lm_length_penalty"),
        "max_batches": config.get("max_batches"),
        "greedy_CER": safe_get(results, "greedy", "CER"),
        "greedy_WER": safe_get(results, "greedy", "WER"),
        "beam_CER": safe_get(results, "beam", "CER"),
        "beam_WER": safe_get(results, "beam", "WER"),
        "beam_lm_CER": safe_get(results, "beam_lm", "CER"),
        "beam_lm_WER": safe_get(results, "beam_lm", "WER"),
        "greedy_total_char_length": safe_get(results, "greedy", "total_char_length"),
        "greedy_total_word_length": safe_get(results, "greedy", "total_word_length"),
        "beam_total_char_length": safe_get(results, "beam", "total_char_length"),
        "beam_total_word_length": safe_get(results, "beam", "total_word_length"),
        "beam_lm_total_char_length": safe_get(results, "beam_lm", "total_char_length"),
        "beam_lm_total_word_length": safe_get(results, "beam_lm", "total_word_length"),
    }

    return row


def discover_jsons(root: Path) -> list[Path]:
    return sorted(root.rglob("*.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Resume y ordena resultados de rescoring a partir de archivos JSON."
    )
    parser.add_argument(
        "root",
        type=str,
        help="Ruta raíz donde buscar los JSON."
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="beam_lm_WER",
        help=(
            "Columna por la que ordenar. "
            "Ej: beam_lm_WER, beam_WER, greedy_WER, beam_lm_CER"
        ),
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Orden ascendente. Por defecto se ordena ascendente igual, útil para claridad."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Ruta opcional para guardar CSV."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Número de filas a mostrar por pantalla."
    )

    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"No existe la ruta: {root}")

    json_files = discover_jsons(root)
    if not json_files:
        print("[INFO] No se encontraron archivos JSON.")
        return

    rows = []
    for jp in json_files:
        row = load_result_file(jp)
        if row is not None:
            rows.append(row)

    if not rows:
        print("[INFO] No se pudieron leer resultados válidos.")
        return

    df = pd.DataFrame(rows)

    if args.sort_by not in df.columns:
        print(f"[WARN] La columna '{args.sort_by}' no existe.")
        print("[INFO] Columnas disponibles:")
        for c in df.columns:
            print(f"  - {c}")
        return

    df = df.sort_values(by=args.sort_by, ascending=True, na_position="last").reset_index(drop=True)

    cols_to_show = [
        "parent_folder",
        "filename",
        "use_lm",
        "lm_backend",
        "lm_model_name",
        "lm_alpha",
        "lm_length_penalty",
        "beam_width",
        "topk",
        "greedy_WER",
        "beam_WER",
        "beam_lm_WER",
        "greedy_CER",
        "beam_CER",
        "beam_lm_CER",
    ]
    cols_to_show = [c for c in cols_to_show if c in df.columns]

    print("\n=== Resultados ordenados ===\n")
    print(df[cols_to_show].head(args.top).to_string(index=False))

    if args.csv:
        out_csv = Path(args.csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\n[OK] CSV guardado en: {out_csv}")

    best = df.iloc[0]
    print("\n=== Mejor resultado según la ordenación ===\n")
    for k in cols_to_show:
        print(f"{k}: {best.get(k)}")


if __name__ == "__main__":
    main()
