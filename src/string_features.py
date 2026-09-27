"""RapidFuzz-based name/address similarity features.

Compiled string comparisons (RapidFuzz), not difflib.SequenceMatcher, for
high-volume pair scoring. Keeps house/unit/postcode agreement SEPARATE from
overall address similarity so a matching postcode never masks a mismatched
house number.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

LEGAL_FORM_PATTERN = re.compile(
    r"\b(ltd|limited|llc|llp|inc|incorporated|corp|corporation|co|company|"
    r"pvt|private|pllc|plc|gmbh|sa|sas|srl|bv|nv|ag|kg|pty)\b\.?",
    re.IGNORECASE,
)
HOUSE_NUMBER_PATTERN = re.compile(
    # Anchored near the START of the address. Optionally skips one leading
    # "#469 " / "Unit 5 " style marker so it doesn't grab a suite/unit number
    # instead of the real house number (e.g. "#469 3/1367 Lingapuram..." ->
    # house number is 3/1367, not 469).
    r"^\s*(?:[#]\s*\S+\s+)?(\d{1,6}(?:/\d{1,6})?[a-zA-Z]?)\b"
)
UNIT_PATTERN = re.compile(r"\b(?:apt|apartment|unit|suite|ste|fl|floor|#)\s*[.:#-]?\s*(\w+)", re.IGNORECASE)
POSTCODE_PATTERN = re.compile(r"\b(\d{4,10}(?:-\d{3,4})?)\b")


def _strip_legal_form(name: str) -> str:
    return LEGAL_FORM_PATTERN.sub("", name or "").strip()


def _has_legal_form(name: str) -> bool:
    return bool(LEGAL_FORM_PATTERN.search(name or ""))


def _first_match(pattern: re.Pattern, text: str) -> Optional[str]:
    match = pattern.search(text or "")
    return match.group(1) if match else None


def _normalize_number_token(token: Optional[str]) -> Optional[str]:
    """Normalize a house-number-like token for comparison: strips leading
    zeros from any purely-numeric segment (so '280' == '0280') and
    lowercases any trailing letter suffix, while leaving compound tokens
    like '3/1367' intact (only the digit run is normalized, not the slash).
    """
    if token is None:
        return None
    match = re.match(r"^(\d+)(.*)$", token)
    if not match:
        return token.lower()
    digits, rest = match.group(1), match.group(2)
    return str(int(digits)) + rest.lower()


def _last_match_excluding(pattern: re.Pattern, text: str, exclude: Optional[str]) -> Optional[str]:
    """Take the LAST match (postcodes conventionally trail an address), skipping
    any match identical to `exclude` (the already-extracted house number) so the
    two features never silently duplicate the same digits."""
    matches = [m.group(1) for m in pattern.finditer(text or "")]
    if exclude is not None:
        matches = [m for m in matches if m != exclude]
    return matches[-1] if matches else None


def _tokens(text: str) -> set[str]:
    return set((text or "").lower().split())


def add_string_similarity_features(
    frame: pd.DataFrame,
    name_a_col: str = "name_a",
    name_b_col: str = "name_b",
    address_a_col: str = "address_a",
    address_b_col: str = "address_b",
) -> pd.DataFrame:
    output = frame.copy()

    name_a = output[name_a_col].fillna("").astype(str)
    name_b = output[name_b_col].fillna("").astype(str)
    addr_a = output[address_a_col].fillna("").astype(str)
    addr_b = output[address_b_col].fillna("").astype(str)

    # --- Name: raw and core (legal-form-stripped) similarity, kept separate ---
    output["name_raw_similarity"] = [
        fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(name_a, name_b)
    ]
    core_a = name_a.map(_strip_legal_form)
    core_b = name_b.map(_strip_legal_form)
    output["name_core_similarity"] = [
        fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(core_a, core_b)
    ]
    output["name_partial_similarity"] = [
        fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(core_a, core_b)
    ]

    # --- Legal form: agreement as its own feature, never deleted/hidden ---
    has_legal_a = name_a.map(_has_legal_form)
    has_legal_b = name_b.map(_has_legal_form)
    output["legal_form_present_a"] = has_legal_a.astype("int8")
    output["legal_form_present_b"] = has_legal_b.astype("int8")
    output["legal_form_agreement"] = (has_legal_a == has_legal_b).astype("int8")

    # --- Name: directional missing/extra tokens (order matters: a vs b) ---
    tokens_a = core_a.map(_tokens)
    tokens_b = core_b.map(_tokens)
    output["name_tokens_missing_from_b"] = [
        len(ta - tb) for ta, tb in zip(tokens_a, tokens_b)
    ]
    output["name_tokens_extra_in_b"] = [
        len(tb - ta) for ta, tb in zip(tokens_a, tokens_b)
    ]
    output["name_token_jaccard"] = [
        (len(ta & tb) / len(ta | tb)) if (ta | tb) else 1.0
        for ta, tb in zip(tokens_a, tokens_b)
    ]

    # --- Rare-token disagreement: tokens that are individually uncommon across
    # this batch and appear in one side but not the other are stronger evidence
    # of a real mismatch than common-word disagreement. ---
    all_tokens = pd.concat([tokens_a, tokens_b])
    token_counts: dict[str, int] = {}
    for token_set in all_tokens:
        for token in token_set:
            token_counts[token] = token_counts.get(token, 0) + 1
    rare_threshold = max(2, int(0.01 * len(output)))

    def _rare_disagreement(ta: set[str], tb: set[str]) -> int:
        disagreeing = ta.symmetric_difference(tb)
        return sum(1 for token in disagreeing if token_counts.get(token, 0) <= rare_threshold)

    output["rare_token_disagreement"] = [
        _rare_disagreement(ta, tb) for ta, tb in zip(tokens_a, tokens_b)
    ]

    # --- Initials agreement (handles abbreviated vs spelled-out names) ---
    def _initials(tokens: set[str]) -> str:
        return "".join(sorted(t[0] for t in tokens if t))

    output["name_initials_match"] = [
        (_initials(ta) == _initials(tb)) for ta, tb in zip(tokens_a, tokens_b)
    ]
    output["name_initials_match"] = output["name_initials_match"].astype("int8")

    # --- Length ratio ---
    len_a = core_a.map(len).clip(lower=1)
    len_b = core_b.map(len).clip(lower=1)
    output["name_length_ratio"] = (
        np.minimum(len_a, len_b) / np.maximum(len_a, len_b)
    ).astype("float32")

    # --- Address: overall similarity ---
    # An address of "" scores 0 similarity against anything -- that is a
    # correct string-comparison result but a WRONG modeling signal: a missing
    # address is "no evidence", not "different address". Emit NaN so
    # LightGBM treats it as genuinely missing (and can learn its own split
    # direction) instead of training on a false maximal-dissimilarity signal.
    address_a_missing = addr_a.str.strip().eq("")
    address_b_missing = addr_b.str.strip().eq("")
    any_address_missing = address_a_missing | address_b_missing
    output["address_a_missing"] = address_a_missing.astype("int8")
    output["address_b_missing"] = address_b_missing.astype("int8")

    output["address_raw_similarity"] = [
        (fuzz.token_sort_ratio(a, b) / 100.0) if (a and b) else np.nan
        for a, b in zip(addr_a, addr_b)
    ]

    # --- Address: house number, unit, postcode SEPARATELY. A matching
    # postcode must never mask a mismatched house number. ---
    house_a = addr_a.map(lambda t: _first_match(HOUSE_NUMBER_PATTERN, t))
    house_b = addr_b.map(lambda t: _first_match(HOUSE_NUMBER_PATTERN, t))
    output["house_number_a_present"] = house_a.notna().astype("int8")
    output["house_number_b_present"] = house_b.notna().astype("int8")
    house_a_norm = house_a.map(_normalize_number_token)
    house_b_norm = house_b.map(_normalize_number_token)
    output["house_number_match"] = [
        int(a == b) if (a is not None and b is not None) else -1
        for a, b in zip(house_a_norm, house_b_norm)
    ]
    output["house_number_missing_flag"] = (
        house_a.isna() | house_b.isna()
    ).astype("int8")

    unit_a = addr_a.map(lambda t: _first_match(UNIT_PATTERN, t))
    unit_b = addr_b.map(lambda t: _first_match(UNIT_PATTERN, t))
    output["unit_number_match"] = [
        int(a == b) if (a is not None and b is not None) else -1
        for a, b in zip(unit_a, unit_b)
    ]
    output["unit_missing_flag"] = (unit_a.isna() | unit_b.isna()).astype("int8")

    postcode_a = pd.Series(
        [_last_match_excluding(POSTCODE_PATTERN, t, h) for t, h in zip(addr_a, house_a)],
        index=addr_a.index,
    )
    postcode_b = pd.Series(
        [_last_match_excluding(POSTCODE_PATTERN, t, h) for t, h in zip(addr_b, house_b)],
        index=addr_b.index,
    )
    output["postcode_match"] = [
        int(a == b) if (a is not None and b is not None) else -1
        for a, b in zip(postcode_a, postcode_b)
    ]
    output["postcode_missing_flag"] = (
        postcode_a.isna() | postcode_b.isna()
    ).astype("int8")

    # --- Explicit contradiction flag: postcode matches but house number does
    # not. This is exactly the failure mode the spec warns about. ---
    output["postcode_match_house_mismatch"] = (
        (output["postcode_match"] == 1) & (output["house_number_match"] == 0)
    ).astype("int8")

    # --- Address token overlap (mirrors name token features) ---
    addr_tokens_a = addr_a.map(_tokens)
    addr_tokens_b = addr_b.map(_tokens)
    output["address_token_jaccard"] = [
        np.nan if miss else ((len(ta & tb) / len(ta | tb)) if (ta | tb) else 1.0)
        for ta, tb, miss in zip(addr_tokens_a, addr_tokens_b, any_address_missing)
    ]

    # --- Joint evidence: strong name with contradictory address, and reverse ---
    # Only meaningful when both addresses are actually present -- a missing
    # address is not "contradictory", it is simply unknown.
    strong_name = output["name_core_similarity"] >= 0.85
    weak_address = output["address_raw_similarity"] < 0.5
    weak_name = output["name_core_similarity"] < 0.5
    strong_address = output["address_raw_similarity"] >= 0.85

    output["strong_name_contradictory_address"] = (
        strong_name & weak_address & ~any_address_missing
    ).astype("int8")
    output["weak_name_strong_address"] = (
        weak_name & strong_address & ~any_address_missing
    ).astype("int8")

    # Graded version of the same idea, instead of a hard threshold: how much
    # is the ADDRESS alone trying to carry the decision while the NAME
    # actively disagrees. 0 when name agrees (nothing to override) or when
    # address doesn't actually support a match either. Only meaningful when
    # both addresses are present -- a missing address can't "carry" anything.
    # This lets the tree learn its own cutoff/slope instead of only seeing a
    # single fixed 0.5/0.85 threshold pair.
    name_disagreement = (1.0 - output["name_core_similarity"]).clip(lower=0)
    address_conflict_carry = pd.Series(
        np.minimum(output["address_raw_similarity"].fillna(0), name_disagreement),
        index=output.index,
    )
    output["address_carries_despite_name_conflict"] = np.where(
        any_address_missing, np.nan, address_conflict_carry
    ).astype("float32")

    return output
