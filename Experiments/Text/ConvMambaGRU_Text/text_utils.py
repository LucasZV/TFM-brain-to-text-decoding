import string
from typing import List

# Vocabulario base para decodificación directa a texto
# 0 queda reservado para CTC blank
CHAR_VOCAB = [
    "<blank>",
    " ",
    "'",
    ",",
    ".",
    "?",
] + list(string.ascii_lowercase)

CHAR_TO_ID = {ch: i for i, ch in enumerate(CHAR_VOCAB)}
ID_TO_CHAR = {i: ch for ch, i in CHAR_TO_ID.items()}


def ascii_array_to_text(ascii_array) -> str:
    """
    Convierte un array ASCII con padding a string.
    Los ceros se interpretan como padding y se eliminan.
    """
    chars = []
    for x in ascii_array:
        x = int(x)
        if x == 0:
            continue
        chars.append(chr(x))
    return "".join(chars)


def normalize_text(text: str) -> str:
    """
    Normalización simple para la primera versión:
    - minúsculas
    - strip
    - filtra caracteres fuera del vocabulario
    """
    text = text.lower().strip()

    filtered = []
    for ch in text:
        if ch in CHAR_TO_ID and ch != "<blank>":
            filtered.append(ch)
    return "".join(filtered)


def text_to_char_ids(text: str) -> List[int]:
    text = normalize_text(text)
    return [CHAR_TO_ID[ch] for ch in text]


def char_ids_to_text(ids: List[int]) -> str:
    chars = []
    for idx in ids:
        idx = int(idx)
        if idx == 0:
            continue
        if idx in ID_TO_CHAR:
            chars.append(ID_TO_CHAR[idx])
    return "".join(chars)


def ctc_greedy_decode(token_ids: List[int]) -> str:
    """
    Colapsa repetidos y elimina blanks para CTC.
    """
    collapsed = []
    prev = None
    for idx in token_ids:
        idx = int(idx)
        if idx != prev:
            collapsed.append(idx)
        prev = idx

    collapsed = [idx for idx in collapsed if idx != 0]
    return char_ids_to_text(collapsed)
