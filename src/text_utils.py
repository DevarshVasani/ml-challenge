"""Shared Unicode-safe text normalization for retrieval and pair features."""

from __future__ import annotations

import unicodedata
from typing import Any

import pandas as pd


def normalize_text(value: Any) -> str:
    """Return deterministic normalized text while preserving digits and Unicode.

    Missing values become empty strings.  Punctuation and symbols are replaced
    with spaces, rather than deleted, so token boundaries remain meaningful.
    """
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("&", " and ")
    text = "".join(
        " " if unicodedata.category(char).startswith(("P", "S")) else char
        for char in text
    )
    return " ".join(text.split())
