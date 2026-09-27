"""Bounded Granite throughput and truncation pilot on real source rows."""
import argparse
import csv
import json
import resource
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
REVISION = "835ad14087e140460703cf0fae09f97d469d65c2"


def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--limit",type=int,default=100000)
    p.add_argument("--batch-size",type=int,default=128);p.add_argument("--max-length",type=int,default=128)
    p.add_argument("--output",required=True);a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModel.from_pretrained(MODEL,revision=REVISION).eval().cuda()
    texts=[]
    started=time.time()
    with open(a.input,newline="") as f:
        for r in csv.DictReader(f,delimiter="\t"):
            texts.append(f"name: {r['business_name']}\naddress: {r['business_address']}\ncountry: {r['country']}")
            if len(texts)>=a.limit:break
    read_seconds=time.time()-started
    lengths=[];n=0;tok_seconds=0;gpu_seconds=0
    for i in range(0,len(texts),a.batch_size):
        chunk=texts[i:i+a.batch_size]
        t=time.time();x=tok(chunk,padding=True,truncation=False,return_tensors=None)
        lengths.extend(len(v) for v in x['input_ids']);tok_seconds+=time.time()-t
        t=time.time();x=tok(chunk,padding=True,truncation=True,max_length=a.max_length,return_tensors='pt').to('cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            y=model(**x).last_hidden_state[:,0]
            y=torch.nn.functional.normalize(y.float(),dim=1)
        torch.cuda.synchronize();gpu_seconds+=time.time()-t
        n+=len(chunk)
        if i and i%(a.batch_size*100)==0:
            print(json.dumps({"done":n,"records_per_second":n/(time.time()-started)}),flush=True)
    result={"model":MODEL,"revision":REVISION,"rows":n,"read_seconds":read_seconds,"tokenize_inspect_seconds":tok_seconds,
            "encode_seconds":gpu_seconds,"wall_seconds":time.time()-started,"records_per_second":n/(time.time()-started),
            "max_length":a.max_length,"over_128":sum(v>128 for v in lengths),"over_192":sum(v>192 for v in lengths),
            "over_256":sum(v>256 for v in lengths),"peak_gpu_bytes":torch.cuda.max_memory_allocated(),
            "peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))

if __name__=='__main__':main()
