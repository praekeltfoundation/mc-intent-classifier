import re

from unidecode import unidecode

from src.config.constants import ABBREV_EXPANSIONS


def normalise_text(text: str) -> str:
    """Cleans and standardizes incoming text for model processing."""
    if not text or not isinstance(text, str):
        return ""
    t = text.strip()
    t = unidecode(t).lower()
    for pat, repl in ABBREV_EXPANSIONS.items():
        t = re.sub(pat, repl, t)
    t = re.sub(r"([!?.,])\1{1,}", r"\1", t)  # collapse repeated punctuation
    t = re.sub(r"\s+", " ", t)
    return t.strip()
