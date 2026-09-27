"""Final set decoders for Member D (§7).

Three decoders behind a common interface:

  DecoderA  Calibrated threshold         (always available)
  DecoderB  Threshold + best-owner       (requires ownership audit pass)
  DecoderC  Expected-F0.5 set decoder    (optional; Poisson-binomial DP)

select_decoder  compares A/B/(C) on the selection partition and freezes the
                winning config (ties broken in favour of simpler decoder).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import evaluate_predictions
from .member_d_contracts import (
    PAIR_KEY,
    SCORE_COL_FUSED,
)


# ---------------------------------------------------------------------------
# DecoderA – calibrated threshold
# ---------------------------------------------------------------------------


class DecoderA:
    """Calibrated probability threshold.

    Per pair: match_probability >= threshold -> match.
    Threshold is selected on the *selection* partition only.
    Empty S1 predictions remain possible.
    """

    name = "threshold"

    def decode(
        self,
        frame: pd.DataFrame,
        threshold: float,
        s1_ids: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """
        Parameters
        ----------
        frame : DataFrame with source1_entity_id, candidate_entity_id, match_probability
        threshold : float in [0, 1]
        s1_ids : optional ordered list of all S1 IDs (ensures every query has a row)

        Returns
        -------
        dict {s1_id: [matched_candidate_ids]}
        """
        required = {"source1_entity_id", "candidate_entity_id", SCORE_COL_FUSED}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"DecoderA: frame missing columns {sorted(missing)}")

        prob = pd.to_numeric(frame[SCORE_COL_FUSED], errors="coerce").fillna(-1.0)
        above = frame[prob >= threshold]
        preds: dict[str, list[str]] = {}
        for s1, grp in above.groupby("source1_entity_id"):
            preds[str(s1)] = sorted(str(c) for c in grp["candidate_entity_id"])

        if s1_ids is not None:
            preds = {s1: preds.get(s1, []) for s1 in s1_ids}
        return preds


# ---------------------------------------------------------------------------
# DecoderB – threshold + best-owner
# ---------------------------------------------------------------------------


class DecoderB:
    """Threshold followed by source-record best-owner selection.

    Only enabled when the gold ownership audit passed (violation_count == 0).
    Applies threshold, then calls apply_best_owner, then regroups by S1.
    """

    name = "threshold_best_owner"

    def decode(
        self,
        frame: pd.DataFrame,
        threshold: float,
        ownership_audit: dict[str, Any],
        s1_ids: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """
        Parameters
        ----------
        frame : DataFrame with PAIR_KEY + match_probability + tree_score
        threshold : float in [0, 1]
        ownership_audit : return value of audit_gold_ownership
        s1_ids : optional ordered list of all S1 IDs

        Returns
        -------
        dict {s1_id: [matched_candidate_ids]}
        """
        if not ownership_audit.get("ownership_enabled", False):
            raise ValueError(
                "DecoderB requires ownership audit to have passed (violation_count == 0)."
            )

        from .ownership import apply_best_owner

        # Apply probability threshold first
        prob = pd.to_numeric(frame[SCORE_COL_FUSED], errors="coerce").fillna(-1.0)
        above_threshold = frame[prob >= threshold].copy()

        if above_threshold.empty:
            if s1_ids is not None:
                return {s1: [] for s1 in s1_ids}
            return {}

        # Apply ownership selection
        selected = apply_best_owner(
            above_threshold,
            probability_col=SCORE_COL_FUSED,
            threshold=0.0,  # threshold already applied above
            ownership_audit=ownership_audit,
        )

        # Regroup by S1
        preds: dict[str, list[str]] = {}
        if "ownership_selected" in selected.columns:
            retained = selected[selected["ownership_selected"] == True]
        else:
            retained = selected

        for s1, grp in retained.groupby("source1_entity_id"):
            preds[str(s1)] = sorted(str(c) for c in grp["candidate_entity_id"])

        if s1_ids is not None:
            preds = {s1: preds.get(s1, []) for s1 in s1_ids}
        return preds


# ---------------------------------------------------------------------------
# DecoderC – expected-F0.5 set decoder (optional)
# ---------------------------------------------------------------------------


class DecoderC:
    """Expected-F0.5 set decoder using Poisson-binomial DP.

    F0.5 = 5 * TP / (4 * k + M)

    Requirements from the plan
    --------------------------
    - Evaluates the empty-set case explicitly.
    - Uses Poisson-binomial DP (not ratio-of-expectations).
    - Runtime-bounded: if a source group is too large, falls back to DecoderA.
    - Documents that candidate independence and missing unretrieved positives
      make this approximate.
    - Does NOT apply test-prior correction.

    Approximation notes
    -------------------
    Candidate probabilities are treated as independent Bernoulli variables.
    Unretrieved positives (not in the candidate set) are not modelled.
    The expected-F0.5 is therefore a lower bound on the true expected-F0.5.
    """

    name = "expected_f05"
    MAX_GROUP_SIZE = 200  # fall back to threshold for larger groups

    def _poisson_binomial_dp(self, probs: np.ndarray) -> np.ndarray:
        """Compute P(S=k) for k=0..n using the Poisson-binomial DP.

        Returns array of length n+1 where result[k] = P(sum==k).
        """
        n = len(probs)
        dp = np.zeros(n + 1)
        dp[0] = 1.0
        for p in probs:
            new_dp = np.zeros(n + 1)
            new_dp[1:] += dp[:-1] * p
            new_dp[:] += dp * (1 - p)
            dp = new_dp
        return dp

    def _expected_f05_for_threshold(
        self, probs: np.ndarray, threshold: float
    ) -> tuple[float, list[str]]:
        """Compute expected F0.5 for a given set of candidates above a threshold."""
        above = probs >= threshold
        k = int(above.sum())
        if k == 0:
            return float("nan"), []
        # With k items selected, each with probability p_i of being TP:
        # E[F0.5 | k selected] = integral over possible TP counts
        selected_probs = probs[above]
        tp_dist = self._poisson_binomial_dp(selected_probs)
        # E[F0.5(k, M)] where M is the true positive count (approximated as E[TP])
        M_approx = float(selected_probs.sum())  # E[number of TPs among selected]
        f05_values = np.array([
            (5 * tp) / (4 * k + M_approx) if (4 * k + M_approx) > 0 else 0.0
            for tp in range(k + 1)
        ])
        return float(np.dot(tp_dist, f05_values)), []

    def _decode_one_s1(
        self,
        grp: pd.DataFrame,
        fallback_threshold: float,
    ) -> list[str]:
        """Choose the optimal subset for one S1."""
        if len(grp) == 0:
            return []
        if len(grp) > self.MAX_GROUP_SIZE:
            # Fallback to threshold decoder
            prob = pd.to_numeric(grp[SCORE_COL_FUSED], errors="coerce").fillna(0.0)
            above = grp[prob >= fallback_threshold]
            return sorted(str(c) for c in above["candidate_entity_id"])

        probs = pd.to_numeric(grp[SCORE_COL_FUSED], errors="coerce").fillna(0.0).to_numpy()
        cand_ids = grp["candidate_entity_id"].astype(str).to_numpy()

        # Empty-set utility: P(M=0) -- P that no candidate is a true positive
        p_empty = float(np.prod(1 - probs))
        # F0.5 of empty set (for singletons this should be 1.0; for non-singletons 0.0)
        # We evaluate it by testing threshold=1.0 (never predict) vs smaller thresholds

        best_ef05 = p_empty  # utility of empty prediction ~ P(no TP exists)
        best_selected: list[str] = []

        # Try sorted probability thresholds (each unique prob value as a cut)
        unique_thresholds = sorted(set(probs.tolist()), reverse=True)
        for t in unique_thresholds:
            ef05, _ = self._expected_f05_for_threshold(probs, t)
            if ef05 > best_ef05:
                best_ef05 = ef05
                above_mask = probs >= t
                best_selected = sorted(cand_ids[above_mask].tolist())

        return best_selected

    def decode(
        self,
        frame: pd.DataFrame,
        fallback_threshold: float = 0.5,
        s1_ids: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """
        Parameters
        ----------
        frame : DataFrame with source1_entity_id, candidate_entity_id, match_probability
        fallback_threshold : threshold used when group is too large for DP
        s1_ids : optional ordered list of all S1 IDs

        Returns
        -------
        dict {s1_id: [matched_candidate_ids]}
        """
        required = {"source1_entity_id", "candidate_entity_id", SCORE_COL_FUSED}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"DecoderC: frame missing columns {sorted(missing)}")

        preds: dict[str, list[str]] = {}
        for s1, grp in frame.groupby("source1_entity_id"):
            preds[str(s1)] = self._decode_one_s1(grp, fallback_threshold)

        if s1_ids is not None:
            preds = {s1: preds.get(s1, []) for s1 in s1_ids}
        return preds


# ---------------------------------------------------------------------------
# Decoder selection
# ---------------------------------------------------------------------------


def select_decoder(
    frame: pd.DataFrame,
    truth: dict[str, list[str]],
    s1_ids: list[str],
    thresholds: list[float] | None = None,
    ownership_audit: dict[str, Any] | None = None,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare decoders A/B/(C) on the selection partition and return the winner.

    Parameters
    ----------
    frame : calibrated score DataFrame (selection partition only)
    truth : ground-truth dict for selection S1s
    s1_ids : selection S1 IDs (full list, including singletons)
    thresholds : list of thresholds to grid-search (default: 0.05..0.95 step 0.05)
    ownership_audit : if provided and passed, DecoderB is also evaluated
    output_path : if given, write frozen decoder config JSON here

    Returns
    -------
    dict with keys: decoder_name, threshold, macro_f05, all_results
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]

    full_truth = {s1: truth.get(s1, []) for s1 in s1_ids}
    decoder_a = DecoderA()
    decoder_b = DecoderB()
    decoder_c = DecoderC()

    all_results: list[dict[str, Any]] = []

    # ---- DecoderA grid search ----
    for t in thresholds:
        preds = decoder_a.decode(frame, threshold=t, s1_ids=s1_ids)
        result = evaluate_predictions(full_truth, preds)
        all_results.append({
            "decoder": "threshold",
            "threshold": t,
            "macro_f05": result["macro_f05"],
        })

    # ---- DecoderB (if ownership passed) ----
    if ownership_audit and ownership_audit.get("ownership_enabled", False):
        for t in thresholds:
            try:
                preds = decoder_b.decode(frame, threshold=t, ownership_audit=ownership_audit, s1_ids=s1_ids)
                result = evaluate_predictions(full_truth, preds)
                all_results.append({
                    "decoder": "threshold_best_owner",
                    "threshold": t,
                    "macro_f05": result["macro_f05"],
                })
            except Exception as e:
                all_results.append({
                    "decoder": "threshold_best_owner",
                    "threshold": t,
                    "macro_f05": float("nan"),
                    "error": str(e),
                })

    # ---- DecoderC (try if frame is small enough) ----
    try:
        preds_c = decoder_c.decode(frame, s1_ids=s1_ids)
        result_c = evaluate_predictions(full_truth, preds_c)
        all_results.append({
            "decoder": "expected_f05",
            "threshold": None,
            "macro_f05": result_c["macro_f05"],
        })
    except Exception as e:
        all_results.append({
            "decoder": "expected_f05",
            "threshold": None,
            "macro_f05": float("nan"),
            "error": str(e),
        })

    # ---- Select winner: highest macro_f05; tie -> simplest (A > B > C) ----
    DECODER_ORDER = {"threshold": 0, "threshold_best_owner": 1, "expected_f05": 2}
    valid = [r for r in all_results if not (isinstance(r["macro_f05"], float) and r["macro_f05"] != r["macro_f05"])]
    if not valid:
        raise ValueError("All decoders returned NaN F0.5; cannot select a winner.")
    best = max(
        valid,
        key=lambda r: (r["macro_f05"], -DECODER_ORDER.get(r["decoder"], 99)),
    )

    winner: dict[str, Any] = {
        "decoder_name": best["decoder"],
        "threshold": best.get("threshold"),
        "macro_f05": best["macro_f05"],
        "all_results": all_results,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(winner, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"[decoder] Frozen decoder config -> {output_path}")

    print(
        f"[decoder] Winner: {best['decoder']}  "
        f"threshold={best.get('threshold')}  "
        f"macro_F0.5={best['macro_f05']:.4f}"
    )
    return winner
