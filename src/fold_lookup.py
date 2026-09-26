"""Explicit positive-component folds with a deterministic unmatched fallback."""

from __future__ import annotations

import hashlib
from typing import Mapping


def fold_for_entity(entity_id: str, explicit_folds: Mapping[str, int], seed: int, n_splits: int) -> int:
    """Use explicit folds first; hash-fold only entities absent from components."""
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    key = str(entity_id)
    if key in explicit_folds:
        fold = int(explicit_folds[key])
        if fold < 0 or fold >= n_splits:
            raise ValueError(f"Explicit fold for {key} is outside [0, {n_splits})")
        return fold
    # Canonical unmatched policy shared with neural_data.stable_unmatched_fold.
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % n_splits

