"""Member D calibrated threshold, ownership and optional exact-under-independence expected-F0.5 decoders."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import evaluate_predictions
from .member_d_contracts import MATCH_PROBABILITY
from .ownership import apply_best_owner


class DecoderA:
    name = "threshold"
    def decode(self, frame: pd.DataFrame, threshold: float, s1_ids: list[str] | None = None) -> dict[str, list[str]]:
        prob = pd.to_numeric(frame[MATCH_PROBABILITY], errors="raise")
        selected = frame[prob >= threshold]
        result = {str(s1): sorted(g.candidate_entity_id.astype(str).tolist()) for s1, g in selected.groupby("source1_entity_id", sort=False)}
        return {s: result.get(s, []) for s in s1_ids} if s1_ids is not None else result


class DecoderB:
    name = "threshold_best_owner"
    def decode(self, frame: pd.DataFrame, threshold: float, ownership_audit: dict[str, Any], s1_ids: list[str] | None = None) -> dict[str, list[str]]:
        selected = apply_best_owner(frame, threshold=threshold, ownership_audit=ownership_audit)
        selected = selected[selected.ownership_selected.astype(bool)] if len(selected) else selected
        result = {str(s1): sorted(g.candidate_entity_id.astype(str).tolist()) for s1, g in selected.groupby("source1_entity_id", sort=False)} if len(selected) else {}
        return {s: result.get(s, []) for s in s1_ids} if s1_ids is not None else result


def _poisson_binomial(probs: np.ndarray) -> np.ndarray:
    dp = np.zeros(len(probs) + 1); dp[0] = 1.0
    for p in probs:
        nxt = np.zeros_like(dp); nxt += dp * (1 - p); nxt[1:] += dp[:-1] * p; dp = nxt
    return dp


def expected_f05_for_set(selected_probs: np.ndarray, unselected_probs: np.ndarray) -> float:
    """Exact E[5*TP/(4*k+M)] under independent Bernoulli candidate labels.

    Empty set uses the contest singleton utility P(M=0).
    """
    k = len(selected_probs)
    if k == 0:
        all_probs = np.concatenate([selected_probs, unselected_probs])
        return float(np.prod(1 - all_probs))
    tp_dist = _poisson_binomial(selected_probs)
    rest_dist = _poisson_binomial(unselected_probs)
    expected = 0.0
    for tp, p_tp in enumerate(tp_dist):
        if p_tp == 0: continue
        for rest, p_rest in enumerate(rest_dist):
            if p_rest == 0: continue
            m = tp + rest
            expected += p_tp * p_rest * (5.0 * tp / (4.0 * k + m))
    return float(expected)


class DecoderC:
    name = "expected_f05"
    MAX_GROUP_SIZE = 100
    def _decode_one(self, group: pd.DataFrame, fallback_threshold: float) -> list[str]:
        if len(group) > self.MAX_GROUP_SIZE:
            return DecoderA().decode(group, fallback_threshold).get(str(group.iloc[0].source1_entity_id), [])
        ordered = group.assign(__p=pd.to_numeric(group[MATCH_PROBABILITY], errors="raise")).sort_values(["__p", "candidate_entity_id"], ascending=[False, True], kind="stable")
        probs = ordered.__p.to_numpy(float); ids = ordered.candidate_entity_id.astype(str).to_numpy()
        best_u = expected_f05_for_set(np.array([], dtype=float), probs); best_k = 0
        for k in range(1, len(probs) + 1):
            u = expected_f05_for_set(probs[:k], probs[k:])
            if u > best_u + 1e-12: best_u, best_k = u, k
        return sorted(ids[:best_k].tolist())
    def decode(self, frame: pd.DataFrame, fallback_threshold: float = .5, s1_ids: list[str] | None = None) -> dict[str, list[str]]:
        result = {str(s1): self._decode_one(g, fallback_threshold) for s1, g in frame.groupby("source1_entity_id", sort=False)}
        return {s: result.get(s, []) for s in s1_ids} if s1_ids is not None else result


def select_decoder(
    frame: pd.DataFrame,
    truth: dict[str, list[str]],
    s1_ids: list[str],
    *,
    thresholds: list[float] | None = None,
    ownership_audit: dict[str, Any] | None = None,
    enable_expected_f05: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or [round(x, 2) for x in np.arange(.05, 1.0, .05)]
    full_truth = {s: truth.get(s, []) for s in s1_ids}; results = []
    for t in thresholds:
        p = DecoderA().decode(frame, t, s1_ids); results.append({"decoder": "threshold", "threshold": t, "macro_f05": evaluate_predictions(full_truth, p)["macro_f05"]})
    if ownership_audit and ownership_audit.get("ownership_enabled"):
        for t in thresholds:
            p = DecoderB().decode(frame, t, ownership_audit, s1_ids); results.append({"decoder": "threshold_best_owner", "threshold": t, "macro_f05": evaluate_predictions(full_truth, p)["macro_f05"]})
    if enable_expected_f05:
        p = DecoderC().decode(frame, s1_ids=s1_ids); results.append({"decoder": "expected_f05", "threshold": None, "macro_f05": evaluate_predictions(full_truth, p)["macro_f05"]})
    order = {"threshold": 0, "threshold_best_owner": 1, "expected_f05": 2}
    best = max(results, key=lambda r: (r["macro_f05"], -order[r["decoder"]]))
    payload = {"decoder_name": best["decoder"], "threshold": best["threshold"], "macro_f05": best["macro_f05"], "all_results": results}
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True); Path(output_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
