#!/usr/bin/env python3
"""
Data Quality & Robustness Audit - Member 4 (Streaming version for large datasets)

Usage:
    python src/audit.py
    python src/audit.py --data-root student_resource/dataset --output-dir audit_output
    python src/audit.py --submission path/to/prototype_submission.tsv
"""
from __future__ import annotations
import argparse, collections, csv, json, math, os, re, subprocess, sys, unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

EXPECTED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
TRAIN_FILES = {"train_source1":"train_source1.tsv","train_source2":"train_source2.tsv","train_source3":"train_source3.tsv"}
TEST_FILES = {"test_source1":"test_source1.tsv","test_source2":"test_source2.tsv","test_source3":"test_source3.tsv"}
GT_FILE = "train_ground_truth.tsv"
SENTINEL_VALUES = frozenset({"NULL","N/A","nan","NaN","NA","-","unknown","none","None","NONE"})
ERROR, WARN, INFO = "ERROR", "WARN", "INFO"
csv.field_size_limit(sys.maxsize)

def has_non_ascii(s): return bool(re.search(r"[^\x00-\x7F]", s)) if s else False
def has_diacritics(s):
    if not s: return False
    return any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", s))
def percentile(data, p):
    if not data: return 0.0
    sd = sorted(data); k = (len(sd)-1)*(p/100.0); f=math.floor(k); c=math.ceil(k)
    return sd[int(k)] if f==c else sd[f]*(c-k)+sd[c]*(k-f)
def jaccard_tokens(a, b):
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta and not tb: return 1.0
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)


class DatasetStats:
    """Streaming stats collector for a single TSV dataset."""
    def __init__(self, name, filepath):
        self.name = name
        self.filepath = filepath
        self.columns = []
        self.row_count = 0
        self.ids = set()
        self.id_dupes = collections.Counter()  # id -> count (only for dups)
        self.country_counts = collections.Counter()
        self.missing_counts = {}  # col -> empty count
        self.blank_counts = {}   # col -> whitespace-only count
        self.sentinel_counts = {}  # col -> Counter of sentinel values
        self.text_lengths = {}    # col -> list (sampled for large datasets)
        self.non_ascii_counts = {}  # col -> count
        self.diacritic_counts = {}  # col -> count

    def process(self):
        """Stream through the file, collecting stats without holding all rows."""
        if not self.filepath.exists():
            return
        with open(self.filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            self.columns = list(reader.fieldnames) if reader.fieldnames else []
            for col in self.columns:
                self.missing_counts[col] = 0
                self.blank_counts[col] = 0
                self.sentinel_counts[col] = collections.Counter()
                self.non_ascii_counts[col] = 0
                self.diacritic_counts[col] = 0
                if col in ("business_name", "business_address"):
                    self.text_lengths[col] = []

            all_ids = collections.Counter()
            for row in reader:
                self.row_count += 1
                eid = row.get("entity_id", "")
                all_ids[eid] += 1
                country = row.get("country", "").strip()
                self.country_counts[country] += 1
                for col in self.columns:
                    val = row.get(col, "")
                    if val == "":
                        self.missing_counts[col] += 1
                    elif val.strip() == "":
                        self.blank_counts[col] += 1
                    else:
                        if val.strip() in SENTINEL_VALUES:
                            self.sentinel_counts[col][val.strip()] += 1
                    if col in self.text_lengths and val.strip():
                        # Sample: keep all for <500k rows, else every 10th
                        if self.row_count <= 500000 or self.row_count % 10 == 0:
                            self.text_lengths[col].append(len(val))
                    if col in ("business_name", "business_address"):
                        if has_non_ascii(val):
                            self.non_ascii_counts[col] += 1

            # Build ID set and duplicate info
            for eid, cnt in all_ids.items():
                self.ids.add(eid)
                if cnt > 1:
                    self.id_dupes[eid] = cnt

    def unique_id_count(self): return len(self.ids)
    def dup_id_count(self): return len(self.id_dupes)
    def dup_row_count(self): return sum(self.id_dupes.values()) - len(self.id_dupes)


def stream_gt(gt_path):
    """Stream through ground truth, returning parsed data."""
    if not gt_path.exists():
        return [], []
    cols = []
    rows_data = []  # list of (s1_id, set_of_matched_ids)
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = list(reader.fieldnames) if reader.fieldnames else []
        for row in reader:
            s1 = row.get("source1_entity_id", "").strip()
            mr = row.get("matched_entity_ids", "").strip()
            matched = set()
            if mr:
                matched = {m.strip() for m in mr.split(",") if m.strip()}
            rows_data.append((s1, matched))
    return cols, rows_data


def audit_all(train_dir, test_dir, output_dir, validator_path, submission_path, repo_root):
    all_findings = []
    print("=== DATA QUALITY AUDIT ===")
    print(f"  Train: {train_dir}")
    print(f"  Test:  {test_dir}")
    print(f"  Output: {output_dir}")
    print()

    # --- Phase 1: Stream through all datasets collecting stats ---
    print("Phase 1: Streaming dataset statistics...")
    datasets = {}  # name -> DatasetStats
    for name, fn in {**TRAIN_FILES, **TEST_FILES}.items():
        d = "train" if name.startswith("train") else "test"
        p = (train_dir if d == "train" else test_dir) / fn
        ds = DatasetStats(name, p)
        print(f"  Streaming {fn}...", end=" ", flush=True)
        ds.process()
        print(f"{ds.row_count} rows, {ds.unique_id_count()} unique IDs")
        datasets[name] = ds
    print()

    # --- Audit 1: Dataset Inventory ---
    print("[1/10] Dataset Inventory...")
    inventory = []
    for name in sorted(datasets.keys()):
        ds = datasets[name]
        if ds.row_count == 0:
            all_findings.append((ERROR, name, f"Dataset {name} has 0 rows."))
            continue
        rec = {
            "dataset": name, "rows": ds.row_count, "columns": len(ds.columns),
            "column_names": ds.columns, "unique_ids": ds.unique_id_count(),
            "duplicate_id_count": ds.dup_id_count(),
            "country_distribution": dict(ds.country_counts.most_common()),
        }
        inventory.append(rec)
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in ds.columns]
        if missing_cols:
            all_findings.append((ERROR, name, f"Missing expected columns: {missing_cols}"))
        dm = f", {ds.dup_id_count()} dup IDs" if ds.dup_id_count() > 0 else ""
        print(f"  [OK] {name}: {ds.row_count} rows, {ds.unique_id_count()} unique IDs{dm}")
    print()

    # --- Audit 2: Missingness ---
    print("[2/10] Missingness...")
    miss_recs = []
    for name in sorted(datasets.keys()):
        ds = datasets[name]
        if ds.row_count == 0: continue
        for col in ds.columns:
            mt = ds.missing_counts.get(col, 0) + ds.blank_counts.get(col, 0)
            pct = (mt / ds.row_count * 100) if ds.row_count > 0 else 0.0
            sc = dict(ds.sentinel_counts.get(col, {}))
            miss_recs.append({
                "dataset": name, "column": col, "total_rows": ds.row_count,
                "empty_count": ds.missing_counts.get(col, 0),
                "blank_only_count": ds.blank_counts.get(col, 0),
                "missing_total": mt, "missing_pct": round(pct, 2),
                "sentinel_values": sc if sc else {},
            })
            if col in ("entity_id", "business_name", "country") and mt > 0:
                all_findings.append((WARN, name, f"Identity field '{col}' has {mt} missing ({pct:.1f}%)."))
            if pct > 50 and col not in ("entity_id", "business_name", "country"):
                all_findings.append((WARN, name, f"Field '{col}' >50% missing ({pct:.1f}%)."))
            if sc:
                all_findings.append((INFO, name, f"Field '{col}' sentinel values: {sc}"))
    mc = sum(1 for r in miss_recs if r["missing_total"] > 0)
    print(f"  [OK] {len(miss_recs)} pairs, {mc} with missing values")
    print()

    # --- Audit 3: Duplicate IDs ---
    print("[3/10] Duplicate IDs...")
    dup_recs = []
    for name in sorted(datasets.keys()):
        ds = datasets[name]
        if ds.dup_id_count() > 0:
            # Note: we can't distinguish exact vs conflicting without re-reading,
            # so we report all as potential duplicates needing investigation
            all_findings.append((WARN, name,
                f"{ds.dup_id_count()} duplicated ID values affecting {ds.dup_id_count() + ds.dup_row_count()} rows."))
            for eid, cnt in sorted(ds.id_dupes.items())[:20]:
                dup_recs.append({"dataset": name, "entity_id": eid, "occurrences": cnt, "category": "needs_review"})
    if dup_recs: print(f"  [WARN] {len(dup_recs)} duplicate ID groups")
    else: print("  [OK] No duplicate IDs")
    print()

    # --- Audit 4: Ground-Truth Integrity ---
    print("[4/10] Ground-truth integrity...")
    gt_path = train_dir / GT_FILE
    gt_summary = {}
    gt_issues = []
    if not gt_path.exists():
        all_findings.append((ERROR, "ground_truth", "GT file not found."))
    else:
        s1_ids = datasets["train_source1"].ids
        s2_ids = datasets["train_source2"].ids
        s3_ids = datasets["train_source3"].ids
        gt_cols, gt_data = stream_gt(gt_path)
        if gt_cols != GT_COLUMNS:
            all_findings.append((ERROR, "ground_truth", f"Unexpected columns: {gt_cols}. Expected: {GT_COLUMNS}"))

        s1_in_gt = set()
        blank_s1 = 0
        orphan_s1, orphan_s2, orphan_s3, invalid_prefix = set(), set(), set(), set()
        dup_gt = 0
        gt_seen = set()
        match_map = {}  # s1 -> set of matched ids

        for s1, matched in gt_data:
            if not s1:
                blank_s1 += 1
                continue
            if s1 not in s1_ids:
                orphan_s1.add(s1)
            rk = f"{s1}|{','.join(sorted(matched))}"
            if rk in gt_seen: dup_gt += 1
            gt_seen.add(rk)
            s1_in_gt.add(s1)
            for mid in matched:
                if mid.startswith("S2-"):
                    if mid not in s2_ids: orphan_s2.add(mid)
                elif mid.startswith("S3-"):
                    if mid not in s3_ids: orphan_s3.add(mid)
                else:
                    invalid_prefix.add(mid)
            # Merge matches for this S1
            if s1 in match_map:
                match_map[s1] |= matched
            else:
                match_map[s1] = set(matched)

        s1_missing = s1_ids - s1_in_gt
        gt_summary = {
            "total_gt_rows": len(gt_data), "unique_s1_in_gt": len(s1_in_gt),
            "total_s1_in_train": len(s1_ids), "s1_missing_from_gt": len(s1_missing),
            "s1_not_in_train": len(orphan_s1), "orphan_s2_refs": len(orphan_s2),
            "orphan_s3_refs": len(orphan_s3), "invalid_prefix_refs": len(invalid_prefix),
            "blank_s1_ids": blank_s1, "duplicate_gt_rows": dup_gt,
        }
        if blank_s1: all_findings.append((ERROR, "ground_truth", f"{blank_s1} blank S1 IDs."))
        if orphan_s1: all_findings.append((ERROR, "ground_truth", f"{len(orphan_s1)} S1 refs not in source1, e.g.: {sorted(orphan_s1)[:5]}"))
        if orphan_s2: all_findings.append((ERROR, "ground_truth", f"{len(orphan_s2)} S2 refs not in source2, e.g.: {sorted(orphan_s2)[:5]}"))
        if orphan_s3: all_findings.append((ERROR, "ground_truth", f"{len(orphan_s3)} S3 refs not in source3, e.g.: {sorted(orphan_s3)[:5]}"))
        if invalid_prefix: all_findings.append((ERROR, "ground_truth", f"{len(invalid_prefix)} invalid prefix IDs, e.g.: {sorted(invalid_prefix)[:5]}"))
        if s1_missing: all_findings.append((WARN, "ground_truth", f"{len(s1_missing)} S1 entities missing from GT."))
        if dup_gt: all_findings.append((WARN, "ground_truth", f"{dup_gt} duplicate GT rows."))

        for oid in sorted(orphan_s1)[:20]:
            gt_issues.append({"issue":"s1_not_in_source","source1_entity_id":oid,"referenced_id":""})
        for oid in sorted(orphan_s2)[:20]:
            gt_issues.append({"issue":"s2_orphan","source1_entity_id":"","referenced_id":oid})
        for oid in sorted(orphan_s3)[:20]:
            gt_issues.append({"issue":"s3_orphan","source1_entity_id":"","referenced_id":oid})

        orph = len(orphan_s1)+len(orphan_s2)+len(orphan_s3)
        if orph: print(f"  [WARN] {orph} orphan GT references")
        else: print(f"  [OK] GT integrity verified ({len(s1_in_gt)} S1)")
        if s1_missing: print(f"  [WARN] {len(s1_missing)} S1 missing from GT")
    print()

    # --- Audit 5: Match Cardinality ---
    print("[5/10] Match cardinality...")
    card_sum = {}
    card_det = []
    if gt_path.exists():
        s1_ids = datasets["train_source1"].ids
        # match_map already built above
        count_dist = collections.Counter()
        for sid in s1_ids:
            n = len(match_map.get(sid, set()))
            count_dist[n] += 1
        zero = count_dist.get(0, 0)
        one = count_dist.get(1, 0)
        multi = sum(v for k,v in count_dist.items() if k > 1)
        card_sum = {
            "s1_with_0_matches": zero, "s1_with_1_match": one,
            "s1_with_multi_matches": multi,
            "detailed_distribution": {str(k):v for k,v in sorted(count_dist.items())},
        }
        for sid in sorted(match_map.keys()):
            n = len(match_map[sid])
            if n > 1:
                card_det.append({"source1_entity_id":sid,"match_count":n,"matched_ids":",".join(sorted(match_map[sid]))})
        all_findings.append((INFO, "match_cardinality", f"0={zero}, 1={one}, >1={multi}"))
        if multi: all_findings.append((WARN, "match_cardinality", f"{multi} S1 entities have multiple matches."))
        print(f"  [OK] 0={zero}, 1={one}, >1={multi}")
    print()

    # --- Audit 6: Train/Test Comparison ---
    print("[6/10] Train vs test comparison...")
    comp_recs = []
    for sn in ("source1", "source2", "source3"):
        tn, en = f"train_{sn}", f"test_{sn}"
        if tn not in datasets or en not in datasets: continue
        td, ed = datasets[tn], datasets[en]
        # Schema comparison
        to = [c for c in td.columns if c not in ed.columns]
        eo = [c for c in ed.columns if c not in td.columns]
        if to: all_findings.append((WARN, sn, f"Train-only columns: {to}"))
        if eo: all_findings.append((WARN, sn, f"Test-only columns: {eo}"))
        # Missingness comparison
        for col in td.columns:
            if col not in ed.columns: continue
            tm = (td.missing_counts.get(col,0)+td.blank_counts.get(col,0))
            em = (ed.missing_counts.get(col,0)+ed.blank_counts.get(col,0))
            tp = (tm/td.row_count*100) if td.row_count else 0
            ep = (em/ed.row_count*100) if ed.row_count else 0
            d = ep - tp
            comp_recs.append({"source":sn,"field":col,"metric":"missing_pct","train_value":round(tp,2),"test_value":round(ep,2),"difference":round(d,2)})
            if abs(d) > 10:
                all_findings.append((WARN, sn, f"'{col}' missingness shift: {tp:.1f}% -> {ep:.1f}%"))
        # Text length comparison
        for col in ("business_name", "business_address"):
            tl = td.text_lengths.get(col, [])
            el = ed.text_lengths.get(col, [])
            for mn, pv in [("median",50),("p90",90),("p95",95),("max",100)]:
                tv = percentile(tl, pv) if tl else 0
                ev = percentile(el, pv) if el else 0
                comp_recs.append({"source":sn,"field":col,"metric":f"len_{mn}","train_value":round(tv,1),"test_value":round(ev,1),"difference":round(ev-tv,1)})
        # Country comparison
        tc, ec = td.country_counts, ed.country_counts
        toc = set(tc.keys()) - set(ec.keys())
        eoc = set(ec.keys()) - set(tc.keys())
        if toc: all_findings.append((INFO, sn, f"Country only in train: {sorted(toc)}"))
        if eoc: all_findings.append((WARN, sn, f"Country only in test: {sorted(eoc)}"))
        for c in sorted(set(tc.keys())|set(ec.keys())):
            comp_recs.append({"source":sn,"field":"country","metric":f"country_{c}","train_value":tc.get(c,0),"test_value":ec.get(c,0),"difference":ec.get(c,0)-tc.get(c,0)})
        # Non-ASCII comparison
        for col in ("business_name","business_address"):
            tna = td.non_ascii_counts.get(col, 0)
            ena = ed.non_ascii_counts.get(col, 0)
            tp2 = (tna/td.row_count*100) if td.row_count else 0
            ep2 = (ena/ed.row_count*100) if ed.row_count else 0
            comp_recs.append({"source":sn,"field":col,"metric":"non_ascii_pct","train_value":round(tp2,2),"test_value":round(ep2,2),"difference":round(ep2-tp2,2)})
            if abs(ep2-tp2) > 5:
                all_findings.append((WARN, sn, f"'{col}' non-ASCII shift: {tp2:.1f}% -> {ep2:.1f}%"))
    print(f"  [OK] {len(comp_recs)} comparison metrics")
    print()

    # --- Audit 7: France-Specific Risks ---
    print("[7/10] France-specific risks...")
    fr_recs = []
    # Collect France stats from datasets
    france_total = 0
    france_label_counts = collections.Counter()
    france_ds_dist = collections.Counter()
    for name, ds in datasets.items():
        for label, cnt in ds.country_counts.items():
            if re.match(r"^(france|fr|fra)$", label, re.IGNORECASE):
                france_total += cnt
                france_label_counts[label] += cnt
                france_ds_dist[name] += cnt

    if france_total == 0:
        all_findings.append((INFO, "france", "No France records found."))
    else:
        all_findings.append((INFO, "france", f"Found {france_total} France records."))
        fr_recs.append({"check":"country_label_variants","details":dict(france_label_counts)})
        if len(france_label_counts) > 1:
            all_findings.append((WARN, "france", f"Multiple France variants: {dict(france_label_counts)}"))
        fr_recs.append({"check":"dataset_distribution","details":dict(france_ds_dist)})

        # Detailed France analysis: stream only France rows from test files
        accent_names = accent_addresses = unicode_issues = 0
        phone_pat = re.compile(r"(?:\+33|0)\s*[1-9](?:\s*\d{2}){4}")
        postal_pat = re.compile(r"\b\d{5}\b")
        phone_fmts, postal_codes = [], []
        abbrev_pats = {
            "Rue/R.": re.compile(r"\b(Rue|R\.)\b", re.IGNORECASE),
            "Avenue/Av.": re.compile(r"\b(Avenue|Av\.?)\b", re.IGNORECASE),
            "Boulevard/Bd": re.compile(r"\b(Boulevard|Bd\.?|Blvd\.?)\b", re.IGNORECASE),
            "Saint/St": re.compile(r"\b(Saint|St\.?)\b", re.IGNORECASE),
            "Sainte/Ste": re.compile(r"\b(Sainte|Ste\.?)\b", re.IGNORECASE),
            "CEDEX": re.compile(r"\bCEDEX\b", re.IGNORECASE),
        }
        abbrev_counts = collections.Counter()
        france_names = collections.defaultdict(list)  # lowered name -> [(eid, addr, dataset)]

        france_pat = re.compile(r"^(france|fr|fra)$", re.IGNORECASE)
        for name, fn in {**TRAIN_FILES, **TEST_FILES}.items():
            d = "train" if name.startswith("train") else "test"
            p = (train_dir if d == "train" else test_dir) / fn
            if not p.exists(): continue
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    c = row.get("country","").strip()
                    if not france_pat.match(c): continue
                    nv = row.get("business_name","")
                    av = row.get("business_address","")
                    if has_diacritics(nv): accent_names += 1
                    if has_diacritics(av): accent_addresses += 1
                    nfc_n = unicodedata.normalize("NFC", nv)
                    nfc_a = unicodedata.normalize("NFC", av)
                    if nv != nfc_n or av != nfc_a: unicode_issues += 1
                    for fld in [nv, av]:
                        phone_fmts.extend(phone_pat.findall(fld))
                    if av: postal_codes.extend(postal_pat.findall(av))
                    for an, ap in abbrev_pats.items():
                        if ap.search(av) or ap.search(nv): abbrev_counts[an] += 1
                    bn = nv.strip().lower()
                    if bn:
                        eid = row.get("entity_id","")
                        france_names[bn].append((eid, av, name))

        fr_recs.append({"check":"diacritics","details":{"accent_in_names":accent_names,"accent_in_addresses":accent_addresses}})
        if accent_names or accent_addresses:
            all_findings.append((INFO, "france", f"Diacritics: {accent_names} names, {accent_addresses} addresses."))
        if unicode_issues:
            all_findings.append((WARN, "france", f"{unicode_issues} NFC mismatches."))
            fr_recs.append({"check":"unicode_normalization","details":{"mismatches":unicode_issues}})
        if phone_fmts:
            uf = sorted(set(phone_fmts))[:10]
            all_findings.append((INFO, "france", f"Phone formats ({len(phone_fmts)}): {uf}"))
            fr_recs.append({"check":"phone_formats","details":{"count":len(phone_fmts),"samples":uf}})
        if postal_codes:
            lz = [p for p in postal_codes if p.startswith("0")]
            fr_recs.append({"check":"postal_codes","details":{"total":len(postal_codes),"unique":len(set(postal_codes)),"leading_zero":len(lz),"samples":sorted(set(postal_codes))[:10]}})
            if lz: all_findings.append((INFO, "france", f"{len(lz)} postal codes with leading zero."))
        if abbrev_counts:
            fr_recs.append({"check":"address_abbreviations","details":dict(abbrev_counts)})
            all_findings.append((INFO, "france", f"Abbreviations: {dict(abbrev_counts)}"))
        # Branch ambiguity
        branch_cands = []
        for bn, entries in sorted(france_names.items()):
            if len(entries) > 1:
                addrs = set(e[1] for e in entries)
                if len(addrs) > 1:
                    branch_cands.append({"business_name":bn,"count":len(entries),"addresses":len(addrs)})
        if branch_cands:
            all_findings.append((WARN, "france", f"{len(branch_cands)} branch ambiguities."))
            fr_recs.append({"check":"branch_ambiguity","details":{"count":len(branch_cands),"samples":branch_cands[:10]}})
    print(f"  [OK] France analysis ({len(fr_recs)} checks)")
    print()

    # --- Audit 8: Difficult Labeled Nonmatches ---
    print("[8/10] Difficult nonmatches...")
    er_recs = []
    if gt_path.exists() and match_map:
        # Strategy: for efficiency with huge datasets, sample S1 entities
        # and compare their names against names of other matched targets
        # Load a sample of S1 entities and their matched targets
        sample_size = min(5000, len(match_map))
        sampled_s1 = sorted(match_map.keys())[:sample_size]

        # We need to load entity data for sampled S1s and their targets
        needed_ids = set(sampled_s1)
        for sid in sampled_s1:
            needed_ids |= match_map.get(sid, set())
        # Also collect all target IDs to compare against
        all_targets = set()
        for targets in match_map.values():
            all_targets |= targets

        # Stream through train files collecting needed entity data
        entity_data = {}  # eid -> {name, address, country}
        for name, fn in TRAIN_FILES.items():
            p = train_dir / fn
            if not p.exists(): continue
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    eid = row.get("entity_id","")
                    if eid in needed_ids or eid in all_targets:
                        entity_data[eid] = {
                            "business_name": row.get("business_name",""),
                            "business_address": row.get("business_address",""),
                            "country": row.get("country",""),
                        }

        # Find difficult non-matches
        candidate_pairs = []
        s1_by_country = collections.defaultdict(list)
        for sid in sampled_s1:
            ed = entity_data.get(sid)
            if ed: s1_by_country[ed.get("country","").strip()].append(sid)

        for country, s1_list in s1_by_country.items():
            ct = set()
            for sid in s1_list: ct |= match_map.get(sid, set())
            for s1_id in s1_list:
                s1e = entity_data.get(s1_id)
                if not s1e: continue
                s1n = s1e.get("business_name","").strip()
                if not s1n: continue
                my_t = match_map.get(s1_id, set())
                for tid in ct:
                    if tid in my_t: continue
                    te = entity_data.get(tid)
                    if not te: continue
                    tn = te.get("business_name","").strip()
                    if not tn: continue
                    ns = jaccard_tokens(s1n, tn)
                    if ns >= 0.5:
                        s1a = s1e.get("business_address","").strip()
                        ta = te.get("business_address","").strip()
                        ads = jaccard_tokens(s1a, ta) if s1a and ta else 0.0
                        reasons = ["similar_name"]
                        if ads >= 0.7: reasons.append("similar_address")
                        candidate_pairs.append((s1_id, tid, ns, ads, "|".join(reasons)))

        candidate_pairs.sort(key=lambda x: -(x[2]*0.7+x[3]*0.3))
        for s1_id, tid, ns, ads, rf in candidate_pairs[:100]:
            s1e = entity_data.get(s1_id, {})
            te = entity_data.get(tid, {})
            er_recs.append({
                "s1_id":s1_id, "other_source":"S2" if tid.startswith("S2-") else "S3",
                "other_id":tid, "label":"nonmatch",
                "s1_name":s1e.get("business_name",""), "other_name":te.get("business_name",""),
                "name_similarity":round(ns,3),
                "s1_address":s1e.get("business_address",""), "other_address":te.get("business_address",""),
                "address_similarity":round(ads,3),
                "s1_country":s1e.get("country",""), "other_country":te.get("country",""),
                "reason_flags":rf,
            })
        if er_recs:
            all_findings.append((INFO, "error_review", f"{len(er_recs)} difficult non-match examples."))
        else:
            all_findings.append((INFO, "error_review", "No high-similarity non-matches found."))
    print(f"  [OK] {len(er_recs)} difficult examples")
    print()

    # --- Audit 9: Submission Writer ---
    print("[9/10] Submission writer review...")
    sw_f = []
    src_dir = repo_root / "src"
    if src_dir.exists():
        for f in src_dir.iterdir():
            if f.suffix == ".py" and f.name != "audit.py":
                try:
                    content = f.read_text(encoding="utf-8", errors="replace")
                    if "to_csv" in content:
                        sw_f.append((INFO, "submission_writer", f"{f.name}: to_csv found - verify sep='\\t' and index=False"))
                    if "source1_entity_id" not in content and "matched_entity_ids" not in content:
                        sw_f.append((WARN, "submission_writer", f"{f.name}: Missing expected output columns."))
                except Exception:
                    sw_f.append((WARN, "submission_writer", f"Could not read {f.name}."))
    if not sw_f:
        sw_f.append((INFO, "submission_writer", "No other src/ files. Member 1's writer not present."))
    all_findings.extend(sw_f)
    print(f"  [OK] {len(sw_f)} observation(s)")
    print()

    # --- Audit 10: Submission Validator ---
    print("[10/10] Submission validation...")
    val_res = {}
    if not validator_path.exists():
        all_findings.append((ERROR, "submission_validation", f"Validator not found: {validator_path}"))
        val_res["status"] = "validator_not_found"
    elif submission_path and submission_path.exists():
        val_res["submission_path"] = str(submission_path)
        cmd = [sys.executable, str(validator_path), "--matching", str(submission_path), "--test-dir", str(test_dir)]
        val_res["command"] = " ".join(cmd)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(repo_root))
            val_res["exit_code"] = proc.returncode
            val_res["stdout"] = proc.stdout
            val_res["stderr"] = proc.stderr
            val_res["status"] = "passed" if proc.returncode == 0 else "failed"
            if proc.returncode == 0:
                all_findings.append((INFO, "submission_validation", "Submission format validation passed."))
            else:
                all_findings.append((ERROR, "submission_validation", f"Validation failed (exit {proc.returncode})."))
        except Exception as e:
            val_res["status"] = "error"
            all_findings.append((ERROR, "submission_validation", f"Error: {e}"))
    else:
        val_res["status"] = "pending"
        val_res["message"] = f"No prototype submission. To run later: python {validator_path} --matching <file> --test-dir {test_dir}"
        all_findings.append((INFO, "submission_validation", "No prototype submission. Validation pending."))
    print(f"  [{val_res.get('status','?').upper()}] {val_res.get('status','')}")
    print()

    # --- Write outputs ---
    print("Writing audit artifacts...")
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_tsv(fp, recs):
        if not recs: return
        fp.parent.mkdir(parents=True, exist_ok=True)
        keys = list(recs[0].keys())
        with open(fp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            for rec in recs:
                row = {}
                for k in keys:
                    v = rec.get(k, "")
                    row[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else (str(v) if v is not None else "")
                w.writerow(row)

    write_tsv(output_dir / "record_counts.tsv", inventory)
    write_tsv(output_dir / "missingness.tsv", miss_recs)
    write_tsv(output_dir / "duplicate_ids.tsv", dup_recs)
    write_tsv(output_dir / "ground_truth_issues.tsv", gt_issues)
    write_tsv(output_dir / "match_cardinality.tsv", card_det)
    write_tsv(output_dir / "train_test_comparison.tsv", comp_recs)
    write_tsv(output_dir / "france_risks.tsv", fr_recs)
    write_tsv(output_dir / "error_review.tsv", er_recs)

    # Markdown report
    errors = [f for f in all_findings if f[0]==ERROR]
    warnings = [f for f in all_findings if f[0]==WARN]
    infos = [f for f in all_findings if f[0]==INFO]
    L = ["# Data Quality & Robustness Audit\n",
         "## Executive Summary\n",
         f"- **Errors**: {len(errors)}", f"- **Warnings**: {len(warnings)}",
         f"- **Info**: {len(infos)}", f"- **Datasets audited**: {len(inventory)}", ""]
    if errors:
        L.append("### Critical Issues\n")
        for s,cx,m in errors: L.append(f"- **[{s}]** [{cx}] {m}")
        L.append("")
    if warnings:
        L.append("### Warnings\n")
        for s,cx,m in warnings: L.append(f"- **[{s}]** [{cx}] {m}")
        L.append("")
    L.append("## Dataset Inventory\n")
    for r in inventory:
        L.append(f"### {r['dataset']}\n")
        L.append(f"- Rows: {r['rows']}")
        L.append(f"- Columns: {r['columns']} - `{r['column_names']}`")
        L.append(f"- Unique IDs: {r['unique_ids']}")
        L.append(f"- Duplicate IDs: {r['duplicate_id_count']}")
        L.append("- Countries:")
        for c2,cnt in r["country_distribution"].items():
            L.append(f"  - {'(empty)' if not c2 else repr(c2)}: {cnt}")
        L.append("")
    L.append("## Missingness\n")
    L.append("| Dataset | Column | Total | Missing | Missing % | Sentinels |")
    L.append("|---------|--------|-------|---------|-----------|-----------|")
    for r in miss_recs:
        st = json.dumps(r["sentinel_values"]) if r["sentinel_values"] else ""
        L.append(f"| {r['dataset']} | {r['column']} | {r['total_rows']} | {r['missing_total']} | {r['missing_pct']}% | {st} |")
    L.append("")
    L.append("## Duplicate IDs\n")
    if dup_recs:
        L.append("| Dataset | Entity ID | Occurrences | Category |")
        L.append("|---------|-----------|-------------|----------|")
        for r in dup_recs[:30]: L.append(f"| {r['dataset']} | {r['entity_id']} | {r['occurrences']} | {r['category']} |")
    else: L.append("No duplicate IDs detected.\n")
    L.append("")
    L.append("## Ground-Truth Integrity\n")
    if gt_summary:
        for k,v in gt_summary.items(): L.append(f"- **{k}**: {v}")
    L.append("")
    if gt_issues:
        L.append("### GT Issues (sample)\n")
        L.append("| Issue | S1 Entity ID | Referenced ID |")
        L.append("|-------|-------------|---------------|")
        for r in gt_issues[:20]: L.append(f"| {r['issue']} | {r.get('source1_entity_id','')} | {r.get('referenced_id','')} |")
    L.append("")
    L.append("## Source 1 Match Cardinality\n")
    if card_sum:
        L.append(f"- 0 matches: {card_sum.get('s1_with_0_matches','N/A')}")
        L.append(f"- 1 match: {card_sum.get('s1_with_1_match','N/A')}")
        L.append(f"- >1 matches: {card_sum.get('s1_with_multi_matches','N/A')}")
        dist = card_sum.get("detailed_distribution",{})
        if dist:
            L.append("\n| Match Count | S1 Entities |")
            L.append("|-------------|-------------|")
            for k,v in sorted(dist.items(), key=lambda x: int(x[0])): L.append(f"| {k} | {v} |")
    L.append("")
    L.append("## Train vs Test Differences\n")
    if comp_recs:
        L.append("| Source | Field | Metric | Train | Test | Diff |")
        L.append("|--------|-------|--------|-------|------|------|")
        for r in comp_recs: L.append(f"| {r['source']} | {r['field']} | {r['metric']} | {r['train_value']} | {r['test_value']} | {r['difference']} |")
    L.append("")
    L.append("## France-Specific Risks\n")
    if fr_recs:
        for r in fr_recs:
            L.append(f"### {r.get('check','')}\n")
            dt = r.get("details",{})
            if isinstance(dt, dict):
                for k,v in dt.items():
                    if isinstance(v,list) and len(v)>5: L.append(f"- **{k}**: {len(v)} items")
                    else: L.append(f"- **{k}**: {v}")
            L.append("")
    else: L.append("No France records.\n")
    L.append("## Difficult Labeled Nonmatches\n")
    if er_recs:
        L.append(f"Top {len(er_recs)} pairs:\n")
        L.append("| S1 ID | Other ID | Name Sim | Addr Sim | Reasons |")
        L.append("|-------|----------|----------|----------|---------|")
        for r in er_recs[:30]: L.append(f"| {r['s1_id']} | {r['other_id']} | {r['name_similarity']} | {r['address_similarity']} | {r['reason_flags']} |")
        if len(er_recs) > 30: L.append(f"\n... and {len(er_recs)-30} more in error_review.tsv")
    else: L.append("No labeled nonmatches for review.\n")
    L.append("")
    L.append("## Submission Format Validation\n")
    L.append(f"- **Status**: {val_res.get('status','unknown')}")
    if "command" in val_res: L.append(f"- **Command**: `{val_res['command']}`")
    if val_res.get("stdout","").strip(): L.append(f"\n```\n{val_res['stdout'].strip()}\n```\n")
    if "message" in val_res: L.append(f"\n{val_res['message']}")
    L.append("")
    L.append("## Confirmed Findings\n")
    for s,cx,m in all_findings:
        if s in (ERROR, WARN): L.append(f"- **[{s}]** [{cx}] {m}")
    L.append("")
    L.append("## Unresolved Risks\n")
    for risk in [
        "GT semantics: Does absence from GT mean 'no match' or 'unknown'?",
        "Country normalization: Raw labels not collapsed.",
        "France (zero-shot): Only in test, no training data.",
        "Branch ambiguity: Same name, different addresses.",
        "Unicode NFC/NFD: May cause comparison failures.",
    ]: L.append(f"- {risk}")
    L.append("")
    (output_dir / "audit_report.md").write_text("\n".join(L), encoding="utf-8")

    # JSON summary
    summary = {
        "status":"completed", "errors":len(errors), "warnings":len(warnings),
        "datasets":{r["dataset"]:{"rows":r["rows"],"unique_ids":r["unique_ids"],"duplicate_ids":r["duplicate_id_count"],"countries":r["country_distribution"]} for r in inventory},
        "ground_truth":gt_summary, "match_cardinality":card_sum,
        "train_test":{"comparison_records":len(comp_recs)},
        "france":{"checks":[r.get("check","") for r in fr_recs]},
        "submission_validation":val_res,
        "findings":[{"severity":s,"context":cx,"message":m} for s,cx,m in all_findings],
    }
    with open(output_dir / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Terminal summary
    print()
    print("=== AUDIT COMPLETE ===")
    for s,cx,m in all_findings:
        if s in (ERROR, WARN): print(f"  [{s}] [{cx}] {m}")
    print()
    print(f"  Errors: {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Report: {output_dir / 'audit_report.md'}")
    print(f"  JSON:   {output_dir / 'audit_summary.json'}")
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description="Data Quality & Robustness Audit (Member 4)")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="audit_output")
    parser.add_argument("--submission", default=None)
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    if args.data_root:
        dr = Path(args.data_root)
        if not dr.is_absolute(): dr = repo_root / dr
    else:
        dr = repo_root / "student_resource" / "dataset"
    td = dr / "train"
    ed = dr / "test"
    od = Path(args.output_dir)
    if not od.is_absolute(): od = repo_root / od
    # Resolve validator: prefer data-root-relative, fall back to student_resource layout
    _vp_candidates = [
        dr.parent / "utils" / "validate_submission.py",
        repo_root / "data" / "utils" / "validate_submission.py",
        repo_root / "student_resource" / "utils" / "validate_submission.py",
    ]
    vp = next((p for p in _vp_candidates if p.exists()), _vp_candidates[2])
    sp = None
    if args.submission:
        sp = Path(args.submission)
        if not sp.is_absolute(): sp = repo_root / sp
    if not td.exists():
        print(f"[ERROR] Train dir not found: {td}")
        return 1
    if not ed.exists():
        print(f"[ERROR] Test dir not found: {ed}")
        return 1
    return audit_all(td, ed, od, vp, sp, repo_root)


if __name__ == "__main__":
    sys.exit(main())
