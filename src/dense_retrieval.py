"""Pinned Granite embedding cache and bounded GPU exact-search diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoModel, AutoTokenizer

from .reverse_retrieval import DATASET, ROOT, identity, source_path
from .retrieval_experiments import gt, load_pairs, score, selected_text, write_json

MODEL="ibm-granite/granite-embedding-97m-multilingual-r2"
REVISION="835ad14087e140460703cf0fae09f97d469d65c2"
DIM=384


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialize(row):
    return f"name: {row['business_name']}\naddress: {row['business_address']}\ncountry: {row['country']}"


def encoder():
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModel.from_pretrained(MODEL,revision=REVISION).eval().cuda()
    return tok,model


def embed(texts,tok,model,max_length=128):
    x=tok(texts,padding=True,truncation=True,max_length=max_length,return_tensors="pt").to("cuda")
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
        y=model(**x).last_hidden_state[:,0]
        y=torch.nn.functional.normalize(y.float(),dim=1)
    return y.cpu().numpy().astype("float16")


def encode_s1(split,out,batch=256,max_length=128):
    out.mkdir(parents=True,exist_ok=True)
    path=source_path(split,"S1")
    config={"source":identity(path),"model":MODEL,"revision":REVISION,"serializer":"name/address/country-v1","max_length":max_length,"dtype":"float16"}
    meta_path=out/"manifest.json"
    prior=json.loads(meta_path.read_text()) if meta_path.exists() else None
    if prior and prior["config"]!=config:raise ValueError("incompatible embedding cache")
    count=sum(1 for _ in path.open())-1
    offset=prior["offset"] if prior else 0
    vectors=np.memmap(out/"vectors.f16",dtype="float16",mode="r+" if prior else "w+",shape=(count,DIM))
    ids_path=out/"ids.tsv"
    if prior and ids_path.stat().st_size!=prior["id_bytes"]:raise ValueError("ID map size differs")
    tok,model=encoder();started=time.time()
    with path.open(newline="") as f,ids_path.open("a" if prior else "w") as id_out:
        rd=csv.DictReader(f,delimiter="\t")
        from itertools import islice
        for _ in islice(rd,offset):pass
        while rows:=list(islice(rd,batch)):
            arr=embed([serialize(r) for r in rows],tok,model,max_length)
            vectors[offset:offset+len(rows)]=arr
            for r in rows:id_out.write(r["entity_id"]+"\n")
            offset+=len(rows)
            if offset%4096<batch or offset==count:
                vectors.flush();id_out.flush();os.fsync(id_out.fileno())
                write_json(meta_path,{"config":config,"rows":count,"offset":offset,"id_bytes":ids_path.stat().st_size,"complete":offset==count,
                                      "seconds_this_run":time.time()-started})
            if offset%25600<batch:print(json.dumps({"offset":offset,"rate":(offset-(prior["offset"] if prior else 0))/(time.time()-started)}),flush=True)
    vectors.flush()
    final={"config":config,"rows":count,"offset":offset,"id_bytes":ids_path.stat().st_size,"complete":offset==count,
           "seconds_this_run":time.time()-started}
    if offset==count:
        final["checksums"]={"vectors.f16":sha256_file(out/"vectors.f16"),"ids.tsv":sha256_file(ids_path)}
    write_json(meta_path,final)
    return {"offset":offset,"count":count,"seconds":time.time()-started}


def pilot(reference,out,batch=64,ref_chunk=100000):
    meta=json.loads((reference/"manifest.json").read_text())
    if not meta["complete"]:raise ValueError("reference embeddings incomplete")
    refs=np.memmap(reference/"vectors.f16",dtype="float16",mode="r",shape=(meta["rows"],DIM))
    ids=(reference/"ids.tsv").read_text().splitlines()
    if len(ids)!=meta["rows"]:raise ValueError("ID map count differs")
    m,baseline,_,_=load_pairs("threshold",verify=False)
    truth=gt(m["query_ids"])
    endpoints=sorted({cid for v in truth.values() for cid in v})
    text=selected_text(endpoints)
    countries=defaultdict(list)
    with source_path("train","S1").open(newline="") as f:
        for i,row in enumerate(csv.DictReader(f,delimiter="\t")):
            if ids[i]!=row["entity_id"]:raise ValueError("reference ID order differs")
            countries[row["country"]].append(i)
    tok,model=encoder();started=time.time();ranks={}
    for country,indices in countries.items():
        selected=[cid for cid in endpoints if text[cid][2]==country]
        if not selected:continue
        reference_matrix=torch.from_numpy(np.asarray(refs[indices])).cuda()
        for start in range(0,len(selected),batch):
            chunk=selected[start:start+batch]
            q=embed([serialize({"business_name":text[c][0],"business_address":text[c][1],"country":text[c][2]}) for c in chunk],tok,model)
            query=torch.from_numpy(q).cuda()
            values,positions=(query@reference_matrix.T).topk(50,dim=1)
            for cid,indexes in zip(chunk,positions.cpu().tolist()):ranks[cid]={ids[indices[j]]:i for i,j in enumerate(indexes,1)}
            if start and start%(batch*10)==0:print(json.dumps({"country":country,"queries":start+len(chunk),"seconds":time.time()-started}),flush=True)
        del reference_matrix
    results={"queries":len(endpoints),"seconds":time.time()-started,"policies":{}}
    out.mkdir(parents=True,exist_ok=True)
    rank_rows=[{"s1_id":q,"source_record_id":cid,"source":cid.split("-",1)[0],"rank":ranks.get(cid,{}).get(q)}
               for q,ids in truth.items() for cid in ids]
    pq.write_table(pa.Table.from_pylist(rank_rows),out/"positive_ranks.parquet",compression="zstd")
    for k in (20,50):
        pairs={q:set(baseline[q]) for q in m["query_ids"]};dense={q:set() for q in m["query_ids"]};new=0
        for q,positive in truth.items():
            for cid in positive:
                if ranks[cid].get(q,999)>k:continue
                dense[q].add(cid)
                if cid not in pairs[q]:new+=1
                pairs[q].add(cid)
        standalone=score(m,dense,truth)
        union=score(m,pairs,truth)
        standalone.pop("candidate_count",None)
        union.pop("candidate_count",None)
        union["newly_recovered_positives"]=new
        union["lost_baseline_positives"]=0
        results["policies"][str(k)]={"dense_only":standalone,"baseline_union":union,"diagnostic_only":True}
    out.mkdir(parents=True,exist_ok=True);write_json(out/"dense_pilot.json",results)
    return results


def retrieve(split,root,out,k=20,batch_size=1000,encode_batch=256,limit=0,sample_path=None):
    start_all=time.time()
    if limit and limit%batch_size:raise ValueError("limit must be divisible by batch size")
    reference=root/f"{split}_s1";meta=json.loads((reference/"manifest.json").read_text())
    if not meta["complete"]:raise ValueError("reference embedding cache incomplete")
    for name,digest in meta.get("checksums",{}).items():
        if sha256_file(reference/name)!=digest:raise ValueError(f"reference cache checksum differs: {name}")
    path=source_path(split,"S1")
    if meta["config"]["source"]!=identity(path):raise ValueError("reference source changed")
    ids=(reference/"ids.tsv").read_text().splitlines()
    if len(ids)!=meta["rows"]:raise ValueError("reference ID map count differs")
    countries=defaultdict(list)
    with path.open(newline="") as f:
        for i,row in enumerate(csv.DictReader(f,delimiter="\t")):
            if row["entity_id"]!=ids[i]:raise ValueError("reference ID map order differs")
            countries[row["country"]].append(i)
    vectors=np.memmap(reference/"vectors.f16",dtype="float16",mode="r",shape=(meta["rows"],DIM))
    # Country truth was checked on threshold: every labeled edge agreed.
    matrices={country:(torch.from_numpy(np.asarray(vectors[indices])).cuda(),indices) for country,indices in countries.items()}
    full=None
    tok,model=encoder()
    out.mkdir(parents=True,exist_ok=True)
    source_files={s:source_path(split,s) for s in ("S2","S3")}
    if sample_path is not None:
        sample_rows=pq.read_table(sample_path).to_pylist()
        sample_dir=out/"sample_inputs";sample_dir.mkdir(exist_ok=True)
        for source in ("S2","S3"):
            target=sample_dir/f"{source}.tsv"
            if not target.exists():
                tmp=target.with_suffix(".tmp")
                with tmp.open("w",newline="") as f:
                    writer=csv.DictWriter(f,fieldnames=["entity_id","business_name","business_address","country"],delimiter="\t")
                    writer.writeheader()
                    writer.writerows({k:r[k] for k in writer.fieldnames} for r in sample_rows if r["source"]==source)
                os.replace(tmp,target)
            source_files[source]=target
    config={"version":"granite-exact-v1","split":split,"k":k,"batch_size":batch_size,"encode_batch":encode_batch,
            "reference":meta["config"],"sources":{s:identity(source_files[s]) for s in ("S2","S3")},
            "sample":identity(sample_path) if sample_path else None,
            "country_policy":"same-country; full-index fallback on absent country"}
    cfg_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    manifest_path=out/"manifest.json";prior=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if prior and prior["config_hash"]!=cfg_hash:raise ValueError("incompatible output manifest")
    completed=dict(prior.get("shards",{})) if prior else {}
    processed=0;written=0;started=time.time()
    from itertools import islice
    for source in ("S2","S3"):
        with source_files[source].open(newline="") as f:
            rd=csv.DictReader(f,delimiter="\t");offset=0
            while batch:=list(islice(rd,batch_size)):
                if limit and processed>=limit:break
                if limit and processed+len(batch)>limit:batch=batch[:limit-processed]
                name=f"{source}-{offset:09d}-{offset+len(batch):09d}.parquet";target=out/name
                if name in completed:
                    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=completed[name]["sha256"]:
                        raise ValueError(f"corrupt completed shard: {name}")
                    processed+=len(batch);offset+=len(batch);continue
                records=[]
                for pos in range(0,len(batch),encode_batch):
                    chunk=batch[pos:pos+encode_batch]
                    queries=torch.from_numpy(embed([serialize(r) for r in chunk],tok,model)).cuda()
                    groups=defaultdict(list)
                    for i,r in enumerate(chunk):groups[r["country"]].append(i)
                    results={}
                    for country,positions in groups.items():
                        if country in matrices:reference_matrix,original_indices=matrices[country]
                        else:
                            if full is None:full=(torch.from_numpy(np.asarray(vectors)).cuda(),list(range(len(ids))))
                            reference_matrix,original_indices=full
                        scores=queries[positions]@reference_matrix.T
                        values,locations=scores.topk(min(k,scores.shape[1]),dim=1)
                        for local,i in enumerate(positions):
                            results[i]=[(ids[original_indices[j]],float(v)) for j,v in zip(locations[local].cpu().tolist(),values[local].cpu().tolist())]
                    for i,row in enumerate(chunk):
                        sid=row["entity_id"]
                        bucket=int.from_bytes(hashlib.blake2b(f"{source}:{sid}".encode(),digest_size=2).digest(),"big")%64
                        for rank,(s1id,sim) in enumerate(results[i],1):
                            records.append({"source":source,"source_record_id":sid,"s1_id":s1id,
                                            "dense_score":sim,"dense_rank":rank,"source_bucket":bucket,
                                            "policy_version":"granite-exact-v1"})
                tmp=target.with_suffix(".parquet.tmp")
                pq.write_table(pa.Table.from_pylist(records),tmp,compression="zstd")
                os.replace(tmp,target)
                digest=hashlib.sha256(target.read_bytes()).hexdigest()
                completed[name]={"sha256":digest,"source_rows":len(batch),"pairs":len(records),"start":offset,"end":offset+len(batch)}
                processed+=len(batch);written+=len(records);offset+=len(batch)
                write_json(manifest_path,{"config":config,"config_hash":cfg_hash,"shards":completed,"complete":False,
                                          "processed_this_run":processed,"seconds_this_run":time.time()-started})
                if processed%100000<batch_size:print(json.dumps({"processed":processed,"rate":processed/(time.time()-started),"pairs":written}),flush=True)
            if limit and processed>=limit:break
    complete=not limit
    write_json(manifest_path,{"config":config,"config_hash":cfg_hash,"shards":completed,"complete":complete,
                              "processed_this_run":processed,"seconds_this_run":time.time()-started})
    return {"processed":processed,"written_pairs":written,"search_write_seconds":time.time()-started,
            "wall_seconds":time.time()-start_all,"complete":complete,
            "peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            "peak_gpu_bytes":torch.cuda.max_memory_allocated(),
            "output_bytes":sum(p.stat().st_size for p in out.glob("*.parquet"))}


def main():
    p=argparse.ArgumentParser();p.add_argument("operation",choices=["encode-s1","pilot","retrieve"])
    p.add_argument("--split",choices=["train","test"],default="train");p.add_argument("--out",type=Path,default=ROOT/"artifacts"/"dense-v1")
    p.add_argument("--batch-size",type=int,default=256);p.add_argument("--k",type=int,default=20)
    p.add_argument("--limit",type=int,default=0);p.add_argument("--candidate-out",type=Path)
    p.add_argument("--sample",type=Path);a=p.parse_args()
    if a.operation=="encode-s1":result=encode_s1(a.split,a.out/f"{a.split}_s1",a.batch_size)
    elif a.operation=="pilot":result=pilot(a.out/"train_s1",a.out/"pilot")
    else:result=retrieve(a.split,a.out,a.candidate_out or a.out/f"{a.split}_candidates",a.k,a.batch_size,limit=a.limit,sample_path=a.sample)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
