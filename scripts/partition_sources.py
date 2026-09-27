"""Hash-partition S2/S3 TSVs for independent complete-group retrieval workers."""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','test'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--buckets',type=int,default=8)
    p.add_argument('--sample',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    manifest_path=a.output/'manifest.json'
    inputs={}
    if a.sample:
        import pyarrow.parquet as pq
        rows=pq.read_table(a.sample).to_pylist()
        sources={s:[r for r in rows if r['source']==s] for s in ('S2','S3')}
        inputs['sample']={'path':str(a.sample.resolve()),'sha256':sha(a.sample)}
    else:
        root=Path(__file__).resolve().parents[2]/'student_resource'/'dataset'/a.split
        sources={s:root/f'{a.split}_source{s[-1]}.tsv' for s in ('S2','S3')}
        inputs={s:{'path':str(path.resolve()),'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns} for s,path in sources.items()}
    config={'split':a.split,'buckets':a.buckets,'inputs':inputs,'partition':'blake2b(source:id) modulo buckets'}
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['config']!=config:raise ValueError('incompatible partition manifest')
        for rel,info in old['files'].items():
            path=a.output/rel
            if not path.exists() or sha(path)!=info['sha256']:raise ValueError(f'corrupt partition: {path}')
        print(json.dumps({'reused':True,'rows':sum(x['rows'] for x in old['files'].values())}));return
    handles={};writers={};counts=Counter()
    fields=['entity_id','business_name','business_address','country']
    try:
        for s in ('S2','S3'):
            for b in range(a.buckets):
                path=a.output/f'{s}-bucket-{b:02d}.tsv.tmp'
                f=path.open('w',newline='');handles[(s,b)]=f
                w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();writers[(s,b)]=w
        for s,data in sources.items():
            if a.sample:iterator=iter(data)
            else:
                f=data.open(newline='');iterator=csv.DictReader(f,delimiter='\t')
            for row in iterator:
                bucket=int.from_bytes(hashlib.blake2b(f"{s}:{row['entity_id']}".encode(),digest_size=2).digest(),'big')%a.buckets
                writers[(s,bucket)].writerow({k:row[k] for k in fields});counts[(s,bucket)]+=1
            if not a.sample:f.close()
    finally:
        for f in handles.values():f.close()
    files={}
    for (s,b),count in counts.items():
        rel=f'{s}-bucket-{b:02d}.tsv';tmp=a.output/(rel+'.tmp');target=a.output/rel
        os.replace(tmp,target)
        files[rel]={'rows':count,'bytes':target.stat().st_size,'sha256':sha(target)}
    tmp=manifest_path.with_suffix('.tmp');tmp.write_text(json.dumps({'config':config,'files':files,'complete':True},indent=2)+'\n');os.replace(tmp,manifest_path)
    print(json.dumps({'rows':sum(counts.values()),'files':len(files)}))

if __name__=='__main__':main()
