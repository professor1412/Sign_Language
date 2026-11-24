"""Keep gloss IDs intact while preparing text for fastText."""
import unicodedata


def is_special(gloss):
    return gloss == "<blank>" or gloss.startswith("__")


def normalize_gloss(gloss, strip_prefixes=False):
    text = unicodedata.normalize("NFC", gloss).lower()
    if strip_prefixes:
        while text.startswith(("loc-", "cl-", "neg-")):
            text = text.split("-", 1)[1]
    if not text or any(c.isspace() for c in text):
        raise ValueError(f"Expected a nonempty single gloss token: {gloss!r}")
    return text
