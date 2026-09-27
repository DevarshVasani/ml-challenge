"""Resumable full-background S1 reverse lexical retrieval.

Usage: python -m src.reverse_retrieval build|pilot|retrieve --split train|test ...
The index contains every S1 row. No labels enter retrieval or candidate output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import resource
import sqlite3
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .retrieval_experiments import DATA, ROOT, SOURCES, gt, load_pairs, score, selected_text, write_json

DATASET = ROOT.parent / "student_resource" / "dataset"
VERSION = "reverse-fts-v1"
POLICY = "reverse-fts-country-postfilter-v5"
LEGAL = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "company", "pvt", "private", "gmbh", "sarl", "plc"}


def identity(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def tokens(value: str) -> list[str]:
    s = unicodedata.normalize("NFKD", value or "").casefold()
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.findall(r"[^\W_]+", s, re.UNICODE)


def source_path(split: str, source: str) -> Path:
    return DATASET / split / f"{split}_source{source[-1]}.tsv"


def index_path(root: Path, split: str) -> Path:
    return root / f"{split}_s1.sqlite"


def build(root: Path, split: str, chunk: int = 10000) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    source = source_path(split, "S1")
    path = index_path(root, split)
    ident = identity(source)
    cfg = {"version": VERSION, "source": ident}
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("config") != cfg or not meta.get("complete"):
            raise ValueError("existing index has incompatible or incomplete metadata")
        with sqlite3.connect(path) as con:
            n = con.execute("SELECT count(*) FROM records").fetchone()[0]
        if n != meta["rows"]:
            raise ValueError("index row count differs from manifest")
        return meta
    if path.exists():
        raise ValueError(f"unrecognized partial index: {path}")
    started = time.time()
    con = sqlite3.connect(path)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("CREATE TABLE records (rid INTEGER PRIMARY KEY, entity_id TEXT NOT NULL UNIQUE, country TEXT NOT NULL)")
        con.execute("CREATE VIRTUAL TABLE search USING fts5(name, address, country, tokenize='unicode61 remove_diacritics 2')")
        n = 0
        with source.open(newline="") as f:
            rd = csv.DictReader(f, delimiter="\t")
            while batch := list(__import__('itertools').islice(rd, chunk)):
                rec = [(n+i+1, r["entity_id"], r["country"]) for i,r in enumerate(batch)]
                terms = [(n+i+1, r["business_name"], r["business_address"], r["country"]) for i,r in enumerate(batch)]
                con.executemany("INSERT INTO records VALUES (?,?,?)", rec)
                con.executemany("INSERT INTO search(rowid,name,address,country) VALUES (?,?,?,?)", terms)
                con.commit()
                n += len(batch)
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.execute("PRAGMA journal_mode=DELETE")
    finally:
        con.close()
    meta = {"config": cfg, "rows": n, "complete": True, "seconds": time.time()-started, "bytes": path.stat().st_size}
    write_json(meta_path, meta)
    return meta


def terms_for(name: str, address: str) -> list[tuple[str,list[str]]]:
    nt = [t for t in tokens(name) if len(t)>=3 and t not in LEGAL]
    at = tokens(address)
    nums = [t for t in at if any(c.isdigit() for c in t)]
    words = [t for t in at if len(t)>=4 and t not in {"road","street","avenue","india","united","states"}]
    # Longer terms are typically more selective; retain numeric address evidence.
    name_terms = sorted(dict.fromkeys(nt), key=lambda x:(-len(x),x))[:3]
    address_terms = (nums[:1] + sorted(dict.fromkeys(words), key=lambda x:(-len(x),x))[:2])[:3]
    return [("name",name_terms),("address",address_terms)]


class ReverseIndex:
    def __init__(self, root: Path, split: str):
        path = index_path(root, split)
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["config"] != {"version": VERSION, "source": identity(source_path(split,"S1"))} or not meta["complete"]:
            raise ValueError("incompatible index")
        self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.con.execute("PRAGMA cache_size=-200000")

    def close(self): self.con.close()

    def _search(self, field: str, terms: list[str], country: str, k: int, original: str) -> list[tuple[str,float]]:
        if not terms: return []
        # AND first; a broader one-term fallback avoids empty groups for sparse names.
        queries = [" AND ".join(f'{field}:"{t}"' for t in terms[:2])]
        if len(terms)>1: queries.append(f'{field}:"{terms[0]}"')
        for query in queries:
            try:
                sql="SELECT r.entity_id, r.country, search.name, search.address FROM search JOIN records r ON r.rid=search.rowid WHERE search MATCH ? LIMIT 201"
                probe=self.con.execute(sql,(query,)).fetchall()
                if len(probe)>200 and country:
                    probe=self.con.execute(sql,(f'country:"{country}" AND ({query})',)).fetchall()
                if len(probe)>200:
                    continue
                consistent=[row for row in probe if row[1]==country] if country else probe
                if not consistent:consistent=probe
                target=set(tokens(original))
                scored=[]
                for entity_id,_,name,address in consistent:
                    candidate=set(tokens(name if field=="name" else address))
                    overlap=len(target&candidate)
                    similarity=overlap/len(target|candidate) if target and candidate else 0.0
                    scored.append((str(entity_id),similarity))
                result=sorted(scored,key=lambda x:(-x[1],x[0]))[:k]
            except sqlite3.OperationalError:
                result = []
            if result: return result
        return []

    def query(self, row: dict, k: int = 20) -> list[dict]:
        channels = {}
        country = row.get("country", "")
        for field, terms in terms_for(row.get("business_name",""), row.get("business_address","")):
            channels[field] = self._search(field,terms,country,k,row.get("business_name" if field=="name" else "business_address",""))
        found = {}
        for field, hits in channels.items():
            for rank,(sid,raw_score) in enumerate(hits,1):
                item = found.setdefault(sid, {"s1_id":sid,"name_rank":None,"address_rank":None,"name_score":None,"address_score":None,"rrf":0.0})
                item[field+"_rank"] = rank
                item[field+"_score"] = raw_score
                item["rrf"] += 1/(60+rank)
        return sorted(found.values(),key=lambda x:(-x["rrf"],x["s1_id"]))[:k]


def pilot(root: Path, out: Path, k_values=(20,50), all_positives=False) -> dict:
    idx = ReverseIndex(root,"train")
    m, baseline, _, _ = load_pairs("threshold",verify=False)
    truth = gt(m["query_ids"])
    # Diagnostic: all threshold positives absent from the baseline. This
    # measures marginal recovery without spending hours querying known hits.
    endpoints = sorted({cid for q,ids in truth.items() for cid in ids if all_positives or cid not in baseline[q]})
    selected = selected_text(endpoints)
    by_query = defaultdict(set)
    ranks = {}
    started = time.time()
    for cid in endpoints:
        name,addr,country = selected[cid]
        hits = idx.query({"business_name":name,"business_address":addr,"country":country},max(k_values))
        ranks[cid] = {h["s1_id"]:i for i,h in enumerate(hits,1)}
    idx.close()
    results = {"queries":len(endpoints),"seconds":time.time()-started,"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"policies":{}}
    if all_positives:
        rank_rows=[{"s1_id":q,"source_record_id":cid,"source":cid.split("-",1)[0],"rank":ranks.get(cid,{}).get(q)}
                   for q,ids in truth.items() for cid in ids]
        out.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rank_rows),out/"positive_ranks.parquet",compression="zstd")
    for k in k_values:
        merged = {q:set(baseline[q]) for q in m["query_ids"]}
        standalone = {q:set() for q in m["query_ids"]}
        new = {q:set() for q in m["query_ids"]}
        for q,ids in truth.items():
            for cid in ids:
                if ranks.get(cid,{}).get(q,10**9)<=k:
                    standalone[q].add(cid)
                    merged[q].add(cid);new[q].add(cid)
        result = score(m,merged,truth)
        result.pop("candidate_count",None)
        result["newly_recovered_positives"] = sum(len(new[q]-baseline[q]) for q in new)
        result["lost_baseline_positives"] = 0
        result["diagnostic_only"] = True
        if all_positives:
            alone=score(m,standalone,truth)
            alone.pop("candidate_count",None)
            results["policies"][str(k)]={"lexical_only":alone,"baseline_union":result,"diagnostic_only":True}
        else:
            results["policies"][str(k)] = result
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"pilot.json",results)
    return results


def retrieve(root: Path, split: str, out: Path, k: int, batch_size: int, limit: int = 0, source_files=None) -> dict:
    if limit and limit % batch_size:
        raise ValueError("diagnostic limit must be divisible by batch size for exact resume")
    out.mkdir(parents=True,exist_ok=True)
    idx = ReverseIndex(root,split)
    source_files=source_files or {s:source_path(split,s) for s in ("S2","S3")}
    config = {"version":POLICY,"split":split,"k":k,"batch_size":batch_size,
              "sources":{**{"S1":identity(source_path(split,"S1"))},**{s:identity(source_files[s]) for s in ("S2","S3")}},
              "index":json.loads(index_path(root,split).with_suffix(".json").read_text())["config"]}
    cfg_hash = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    manifest_path = out/"manifest.json"
    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if prior and prior["config_hash"] != cfg_hash: raise ValueError("incompatible output manifest")
    completed = dict(prior.get("shards",{})) if prior else {}
    started = time.time(); processed=0;written=0
    for source in ("S2","S3"):
        with source_files[source].open(newline="") as f:
            rd=csv.DictReader(f,delimiter="\t")
            offset=0
            while batch := list(__import__('itertools').islice(rd,batch_size)):
                if limit and processed>=limit: break
                if limit and processed+len(batch)>limit: batch=batch[:limit-processed]
                name=f"{source}-{offset:09d}-{offset+len(batch):09d}.parquet"
                target=out/name
                if name in completed:
                    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=completed[name]["sha256"]:
                        raise ValueError(f"corrupt completed shard: {name}")
                    processed+=len(batch);offset+=len(batch);continue
                records=[]
                for row in batch:
                    sid=row["entity_id"]
                    for hit in idx.query(row,k):
                        records.append({"source":source,"source_record_id":sid,"s1_id":hit["s1_id"],
                                        "name_rank":hit["name_rank"],"address_rank":hit["address_rank"],
                                        "name_score":hit["name_score"],"address_score":hit["address_score"],
                                        "retrieval_rank":len(records)+1,"policy_version":POLICY,
                                        "source_bucket":int.from_bytes(hashlib.blake2b(f"{source}:{sid}".encode(),digest_size=2).digest(),"big")%64})
                # Ranks are per complete source group and unique pairs are guaranteed by query().
                by_source=defaultdict(int)
                for rec in records:
                    key=rec["source_record_id"];by_source[key]+=1;rec["retrieval_rank"]=by_source[key]
                tmp=target.with_suffix(".parquet.tmp")
                if records:
                    pq.write_table(pa.Table.from_pylist(records),tmp,compression="zstd")
                else:
                    pq.write_table(pa.table({"source":pa.array([],pa.string()),"source_record_id":pa.array([],pa.string()),"s1_id":pa.array([],pa.string())}),tmp)
                os.replace(tmp,target)
                digest=hashlib.sha256(target.read_bytes()).hexdigest()
                completed[name]={"sha256":digest,"source_rows":len(batch),"pairs":len(records),"start":offset,"end":offset+len(batch)}
                processed+=len(batch);written+=len(records);offset+=len(batch)
                write_json(manifest_path,{"config":config,"config_hash":cfg_hash,"shards":completed,"complete":False,"processed_this_run":processed,"seconds_this_run":time.time()-started})
            if limit and processed>=limit: break
    idx.close()
    complete=not limit
    write_json(manifest_path,{"config":config,"config_hash":cfg_hash,"shards":completed,"complete":complete,"processed_this_run":processed,"seconds_this_run":time.time()-started})
    return {"processed":processed,"written_pairs":written,"seconds":time.time()-started,"complete":complete,"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}


def main():
    p=argparse.ArgumentParser();p.add_argument("operation",choices=["build","pilot","retrieve"])
    p.add_argument("--split",choices=["train","test"],default="train")
    p.add_argument("--root",type=Path,default=ROOT/"artifacts"/"reverse-v1")
    p.add_argument("--out",type=Path);p.add_argument("--k",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=1000);p.add_argument("--limit",type=int,default=0)
    p.add_argument("--all-positives",action="store_true")
    p.add_argument("--source2",type=Path);p.add_argument("--source3",type=Path)
    a=p.parse_args()
    if a.operation=="build":r=build(a.root,a.split)
    elif a.operation=="pilot":r=pilot(a.root,a.out or a.root/"pilot",all_positives=a.all_positives)
    else:
        files={"S2":a.source2 or source_path(a.split,"S2"),"S3":a.source3 or source_path(a.split,"S3")}
        r=retrieve(a.root,a.split,a.out or a.root/f"{a.split}_candidates",a.k,a.batch_size,a.limit,files)
    print(json.dumps(r,indent=2))


if __name__=="__main__":main()
