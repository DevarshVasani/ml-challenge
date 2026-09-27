"""Run resumable reverse lexical shards over pre-hashed source partitions."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','test'],required=True)
    p.add_argument('--partitions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--k',type=int,default=20);p.add_argument('--workers',type=int,default=7)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((a.partitions/'manifest.json').read_text())
    if not meta['complete'] or meta['config']['split']!=a.split:raise ValueError('partition input incompatible')
    n=meta['config']['buckets'];pending=list(range(n));active={};started=time.time()
    while pending or active:
        while pending and len(active)<a.workers:
            b=pending.pop(0);dest=a.output/f'bucket-{b:02d}';dest.mkdir(exist_ok=True)
            command=[sys.executable,'-m','src.reverse_retrieval','retrieve','--split',a.split,
                     '--root','artifacts/reverse-v1','--out',str(dest),'--k',str(a.k),'--batch-size','1000',
                     '--source2',str(a.partitions/f'S2-bucket-{b:02d}.tsv'),
                     '--source3',str(a.partitions/f'S3-bucket-{b:02d}.tsv')]
            log=(a.output/f'bucket-{b:02d}.log').open('a')
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            active[b]=(process,log)
            print(json.dumps({'started_bucket':b,'pid':process.pid}),flush=True)
        time.sleep(1)
        for b,(process,log) in list(active.items()):
            if process.poll() is None:continue
            log.close();active.pop(b)
            if process.returncode:raise RuntimeError(f'bucket {b} failed: {a.output/f"bucket-{b:02d}.log"}')
            print(json.dumps({'completed_bucket':b,'elapsed_seconds':time.time()-started}),flush=True)
    files=[json.loads((a.output/f'bucket-{b:02d}'/'manifest.json').read_text()) for b in range(n)]
    result={'complete':all(x['complete'] for x in files),'source_rows':sum(sum(s['source_rows'] for s in x['shards'].values()) for x in files),
            'candidate_pairs':sum(sum(s['pairs'] for s in x['shards'].values()) for x in files),
            'seconds':time.time()-started,'workers':a.workers}
    (a.output/'run_report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
