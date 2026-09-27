"""Full-background phonetic-name retrieval for script-mismatched source names."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import time
from pathlib import Path

from unidecode import unidecode

from .reverse_retrieval import identity, source_path, tokens
from .retrieval_experiments import sha, write_json

VERSION = "phonetic-name-address-v2"
LEGAL = {"private", "limited", "pvt", "ltd", "llc", "inc", "corporation", "company", "corp"}
LEGAL_SKELETONS = {"prvt", "lmtd", "lmt", "pvt", "ltd", "krprtn", "kmpn", "krp"}


def phonetic_tokens(value: str) -> list[str]:
    result = []
    for word in re.findall(r"[a-z]+", unidecode(value or "").lower()):
        if word in LEGAL:
            continue
        word = re.sub(r"[aeiouy]", "", word)
        word = re.sub(r"(.)\1+", r"\1", word)
        word = word.translate(str.maketrans({"c": "k", "q": "k", "z": "s", "w": "v"}))
        if len(word) >= 3:
            result.append(word)
    return list(dict.fromkeys(result))


def query_tokens(value: str) -> list[str]:
    """Discard legal forms whose romanized spellings differ from Latin forms."""
    return [word for word in phonetic_tokens(value) if word not in LEGAL_SKELETONS]


def index_path(root: Path, split: str) -> Path:
    return root / f"{split}_phonetic_address.sqlite"


def build(root: Path, split: str) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    source = source_path(split, "S1")
    path = index_path(root, split)
    meta_path = path.with_suffix(".json")
    config = {"version": VERSION, "source": identity(source)}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("config") != config or not meta.get("complete"):
            raise ValueError("incompatible phonetic index")
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            if db.execute("SELECT count(*) FROM records").fetchone()[0] != meta["rows"]:
                raise ValueError("phonetic index row count differs")
        return meta
    if path.exists():
        raise ValueError(f"unrecognized phonetic index: {path}")
    started = time.time()
    tmp = path.with_suffix(".sqlite.tmp")
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    try:
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("CREATE TABLE records (rid INTEGER PRIMARY KEY, entity_id TEXT NOT NULL UNIQUE, country TEXT NOT NULL, name TEXT NOT NULL, address TEXT NOT NULL)")
        db.execute("CREATE VIRTUAL TABLE search USING fts5(phonetic, address, country, tokenize='unicode61 remove_diacritics 2')")
        count = 0
        with source.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            while batch := list(__import__("itertools").islice(reader, 10_000)):
                rows = [(count + i + 1, r["entity_id"], r["country"], r["business_name"], r["business_address"]) for i, r in enumerate(batch)]
                keys = [(count + i + 1, " ".join(phonetic_tokens(r["business_name"])), r["business_address"], r["country"]) for i, r in enumerate(batch)]
                db.executemany("INSERT INTO records VALUES (?,?,?,?,?)", rows)
                db.executemany("INSERT INTO search(rowid,phonetic,address,country) VALUES (?,?,?,?)", keys)
                db.commit()
                count += len(batch)
        db.execute("INSERT INTO search(search) VALUES ('optimize')")
        db.commit()
    finally:
        db.close()
    os.replace(tmp, path)
    result = {"config": config, "complete": True, "rows": count, "bytes": path.stat().st_size,
              "sha256": sha(path), "seconds": time.time() - started}
    write_json(meta_path, result)
    return result


class PhoneticIndex:
    def __init__(self, root: Path, split: str):
        path = index_path(root, split)
        meta = json.loads(path.with_suffix(".json").read_text())
        if not meta["complete"] or meta["config"] != {"version": VERSION, "source": identity(source_path(split, "S1"))}:
            raise ValueError("incompatible phonetic index")
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def close(self):
        self.db.close()

    def query(self, name: str, address: str, country: str, k: int = 20) -> list[tuple[str, float]]:
        words = sorted(query_tokens(name), key=lambda x: (-len(x), x))
        if not words:
            return []
        # Every probe uses only the source text. Broad postings are discarded.
        address_words = list(dict.fromkeys(t for t in tokens(address) if len(t) >= 3 or any(c.isdigit() for c in t)))
        numbers = [t for t in address_words if any(c.isdigit() for c in t)]
        location = [t for t in address_words if t not in numbers]
        qualifiers = (numbers[:1] + list(reversed(location[-2:])))[:3]
        name_probes = [" AND ".join(f'phonetic:"{word}"' for word in words[:2])]
        name_probes += [f'phonetic:"{word}"' for word in words[:2]]
        probes = [name_probes[0]]
        probes += [f'{name_probes[1]} AND address:"{term}"' for term in qualifiers]
        if len(name_probes) > 2:
            probes += [f'{name_probes[2]} AND address:"{term}"' for term in qualifiers]
        probes += name_probes[1:]
        if country:
            probes = [f'country:"{country}" AND ({query})' for query in probes]
        source_tokens = set(words)
        address_tokens = set(tokens(address))
        found: dict[str, float] = {}
        for query in dict.fromkeys(probes):
            rows = self.db.execute(
                "SELECT r.entity_id,r.country,r.name,r.address FROM search JOIN records r ON r.rid=search.rowid WHERE search MATCH ? LIMIT 501",
                (query,),
            ).fetchall()
            if len(rows) > 500:
                continue
            same_country = [r for r in rows if r[1] == country] if country else rows
            for sid, _, candidate_name, candidate_address in same_country:
                candidate_tokens = set(query_tokens(candidate_name))
                overlap = len(source_tokens & candidate_tokens)
                name_score = overlap / len(source_tokens | candidate_tokens) if candidate_tokens else 0.0
                other_address = set(tokens(candidate_address))
                address_score = len(address_tokens & other_address) / len(address_tokens | other_address) if address_tokens and other_address else 0.0
                source_numbers = {t for t in address_tokens if any(c.isdigit() for c in t)}
                candidate_numbers = {t for t in other_address if any(c.isdigit() for c in t)}
                number_score = len(source_numbers & candidate_numbers) / len(source_numbers | candidate_numbers) if source_numbers and candidate_numbers else 0.0
                score = 0.55 * name_score + 0.25 * address_score + 0.20 * number_score
                found[sid] = max(found.get(sid, -1.0), score)
        return sorted(found.items(), key=lambda x: (-x[1], x[0]))[:k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["build"])
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/phonetic-v2"))
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.split)))


if __name__ == "__main__":
    main()
