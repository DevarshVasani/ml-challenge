"""Validate Member A -> Member D neural-score handoffs.

The validator is intentionally strict for threshold handoff:
- checkpoint card provenance must be verified;
- held-out query exposure must be zero;
- one checkpoint ID only;
- every score shard must have a completion sidecar;
- no missing/non-finite threshold logits;
- canonical 3-part keys must exactly match C's threshold gate;
- optional routed-neural handoff must exactly match route=neural keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .member_d_contracts import (
    FORBIDDEN_PRODUCTION_COLUMNS,
    PAIR_KEY,
    validate_pair_keys,
)
from .neural_contracts import sha256_file

SCORE_COLUMNS = PAIR_KEY + [
    "neural_logit",
    "neural_scored",
    "neural_status",
    "checkpoint_id",
]


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_gate(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    gate = (
        pd.read_parquet(p)
        if p.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(p, sep="\t", dtype=object, keep_default_na=False)
    )
    validate_pair_keys(gate)
    forbidden = sorted(set(gate.columns) & FORBIDDEN_PRODUCTION_COLUMNS)
    if forbidden:
        raise ValueError(
            "Gate contains forbidden/truth-derived fields: "
            f"{forbidden}. Use C's leakage-safe v2+ artifact."
        )
    return gate


def _clean_zone_identifiers(root: str | Path) -> int:
    count = 0
    for path in Path(root).rglob("*:Zone.Identifier"):
        if path.is_file():
            path.unlink()
            count += 1
    return count


def _score_files(score_dir: str | Path) -> list[Path]:
    files = sorted(Path(score_dir).glob("*.neural.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No *.neural.parquet score shards under {score_dir}"
        )
    return files


def _validate_sidecar(score_path: Path) -> dict[str, Any]:
    sidecar = Path(f"{score_path}.complete.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing completion sidecar: {sidecar}")
    payload = _read_json(sidecar)
    state = payload.get("completion", payload.get("status", "complete"))
    if state not in {"complete", "completed", "ok"}:
        raise ValueError(f"Incomplete score sidecar {sidecar}: {state!r}")

    expected_hash = (
        payload.get("output_sha256")
        or payload.get("sha256")
        or payload.get("generated_file_sha256")
    )
    if expected_hash:
        actual_hash = sha256_file(score_path)
        if actual_hash != str(expected_hash):
            raise ValueError(
                f"Output checksum mismatch for {score_path.name}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    return payload


def _load_scores(score_dir: str | Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    sidecars = []
    for path in _score_files(score_dir):
        sidecars.append(_validate_sidecar(path))
        frame = pd.read_parquet(path)
        missing = [c for c in SCORE_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{path.name} missing neural columns: {missing}")
        frames.append(frame[SCORE_COLUMNS].copy())
    scores = pd.concat(frames, ignore_index=True)
    validate_pair_keys(scores)
    return scores, sidecars


def _manifest(score_dir: str | Path) -> dict[str, Any]:
    path = Path(score_dir) / "neural_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing neural manifest: {path}")
    payload = _read_json(path)
    if payload.get("completion") != "complete":
        raise ValueError(
            f"Neural manifest is not complete: {payload.get('completion')!r}"
        )
    if payload.get("pending_inputs"):
        raise ValueError(
            f"Neural manifest still has pending inputs: "
            f"{len(payload['pending_inputs'])}"
        )
    return payload


def _checkpoint_from_manifest(manifest: dict[str, Any]) -> str | None:
    return manifest.get("checkpoint_id") or manifest.get("neural_model_version")


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[PAIR_KEY].astype(str))


def _coverage(expected: pd.DataFrame, actual: pd.DataFrame, label: str) -> dict[str, int]:
    e = _key_index(expected)
    a = _key_index(actual)
    missing = e.difference(a)
    extra = a.difference(e)
    if len(missing) or len(extra):
        raise ValueError(
            f"{label} key coverage mismatch: "
            f"missing={len(missing)} extra={len(extra)}; "
            f"missing_examples={list(missing[:5])}; "
            f"extra_examples={list(extra[:5])}"
        )
    return {"expected": len(e), "actual": len(a), "missing": 0, "extra": 0}


def _validate_scores(
    scores: pd.DataFrame,
    expected_checkpoint: str,
    *,
    expected_rows: int | None,
) -> dict[str, Any]:
    if expected_rows is not None and len(scores) != expected_rows:
        raise ValueError(
            f"Neural row count mismatch: expected {expected_rows}, got {len(scores)}"
        )
    logits = pd.to_numeric(scores["neural_logit"], errors="coerce")
    if logits.isna().any():
        raise ValueError(
            f"Neural scores contain {int(logits.isna().sum())} missing/non-finite logits"
        )

    scored = scores["neural_scored"].astype(bool)
    if not scored.all():
        raise ValueError(
            f"Threshold handoff has {int((~scored).sum())} neural_scored=false rows"
        )

    statuses = scores["neural_status"].astype(str)
    bad_status = ~statuses.isin(["scored", "cached"])
    if bad_status.any():
        raise ValueError(
            "Threshold handoff contains non-success neural_status values: "
            f"{statuses[bad_status].value_counts().to_dict()}"
        )

    checkpoint_ids = sorted(scores["checkpoint_id"].astype(str).unique())
    if checkpoint_ids != [expected_checkpoint]:
        raise ValueError(
            f"Score shards use checkpoint IDs {checkpoint_ids}; "
            f"expected only {expected_checkpoint}"
        )
    return {
        "rows": len(scores),
        "checkpoint_id": expected_checkpoint,
        "status_counts": {
            str(k): int(v) for k, v in statuses.value_counts().to_dict().items()
        },
        "missing_logit_count": 0,
        "duplicate_key_count": 0,
    }


def validate_handoff(
    checkpoint_card_path: str | Path,
    training_exposure_path: str | Path,
    threshold_score_dir: str | Path,
    gate_path: str | Path,
    *,
    routed_score_dir: str | Path | None = None,
    expected_checkpoint: str | None = None,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    card = _read_json(checkpoint_card_path)
    card_checkpoint = str(card.get("checkpoint_id", ""))
    if not card_checkpoint:
        raise ValueError("checkpoint_card.json has no checkpoint_id")
    if expected_checkpoint and card_checkpoint != expected_checkpoint:
        raise ValueError(
            f"Checkpoint card ID {card_checkpoint} != expected {expected_checkpoint}"
        )

    training = card.get("training", {})
    if training.get("inputs_verified") is not True:
        raise ValueError("Neural checkpoint card does not have inputs_verified=true")

    heldout = card.get("exposure", {}).get(
        "heldout_queries_seen_in_training", {}
    )
    leaked = {str(k): int(v) for k, v in heldout.items() if int(v) != 0}
    if leaked:
        raise ValueError(
            f"Held-out queries appear in neural training exposure: {leaked}"
        )

    exposure = pd.read_parquet(training_exposure_path)
    exposure_required = {
        "entity_id",
        "role",
        "candidate_source",
        "fold",
        "seen_as_positive",
    }
    if not exposure_required.issubset(exposure.columns):
        raise ValueError(
            "training_exposure.parquet missing columns: "
            f"{sorted(exposure_required - set(exposure.columns))}"
        )

    manifest = _manifest(threshold_score_dir)
    manifest_checkpoint = _checkpoint_from_manifest(manifest)
    if manifest_checkpoint and str(manifest_checkpoint) != card_checkpoint:
        raise ValueError(
            f"Threshold manifest checkpoint {manifest_checkpoint} "
            f"!= card {card_checkpoint}"
        )

    scores, sidecars = _load_scores(threshold_score_dir)
    score_report = _validate_scores(
        scores,
        card_checkpoint,
        expected_rows=expected_rows,
    )

    gate = _read_gate(gate_path)
    full_coverage = _coverage(gate, scores, "full threshold neural handoff")

    routed_report = None
    if routed_score_dir is not None:
        routed_manifest = _manifest(routed_score_dir)
        routed_checkpoint = _checkpoint_from_manifest(routed_manifest)
        if routed_checkpoint and str(routed_checkpoint) != card_checkpoint:
            raise ValueError(
                f"Routed manifest checkpoint {routed_checkpoint} "
                f"!= card {card_checkpoint}"
            )
        routed, routed_sidecars = _load_scores(routed_score_dir)
        routed_score_report = _validate_scores(
            routed,
            card_checkpoint,
            expected_rows=None,
        )
        route_col = (
            "decision_route" if "decision_route" in gate.columns else "route"
        )
        wanted = gate[
            gate[route_col].astype(str).str.lower().isin(
                ["neural", "neural_fusion"]
            )
        ].copy()
        routed_coverage = _coverage(
            wanted,
            routed,
            "routed neural handoff",
        )
        routed_report = {
            "manifest": str(Path(routed_score_dir) / "neural_manifest.json"),
            "sidecar_count": len(routed_sidecars),
            "scores": routed_score_report,
            "coverage": routed_coverage,
        }

    return {
        "checkpoint": {
            "checkpoint_id": card_checkpoint,
            "inputs_verified": True,
            "heldout_queries_seen_in_training": heldout,
            "training_exposure_rows": len(exposure),
            "checkpoint_card_sha256": sha256_file(checkpoint_card_path),
            "training_exposure_sha256": sha256_file(training_exposure_path),
        },
        "threshold_scores": {
            "manifest": str(Path(threshold_score_dir) / "neural_manifest.json"),
            "sidecar_count": len(sidecars),
            "scores": score_report,
            "coverage": full_coverage,
        },
        "routed_scores": routed_report,
        "gate": {
            "path": str(gate_path),
            "sha256": sha256_file(gate_path),
            "rows": len(gate),
        },
        "handoff_valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-card", required=True)
    parser.add_argument("--training-exposure", required=True)
    parser.add_argument("--threshold-score-dir", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--routed-score-dir")
    parser.add_argument("--expected-checkpoint")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--clean-zone-identifiers",
        action="store_true",
        help="Delete Windows :Zone.Identifier metadata under the handoff root.",
    )
    parser.add_argument(
        "--handoff-root",
        help="Root used only with --clean-zone-identifiers.",
    )
    args = parser.parse_args()

    cleaned = 0
    if args.clean_zone_identifiers:
        if not args.handoff_root:
            raise ValueError(
                "--clean-zone-identifiers requires --handoff-root"
            )
        cleaned = _clean_zone_identifiers(args.handoff_root)

    report = validate_handoff(
        args.checkpoint_card,
        args.training_exposure,
        args.threshold_score_dir,
        args.gate,
        routed_score_dir=args.routed_score_dir,
        expected_checkpoint=args.expected_checkpoint,
        expected_rows=args.expected_rows,
    )
    report["zone_identifier_files_removed"] = cleaned

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
