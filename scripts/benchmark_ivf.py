"""Bounded India IVF-Flat benchmark against GPU exact country search."""
import csv
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch

from src.dense_retrieval import DIM, embed, encoder, serialize
from src.retrieval_experiments import gt, load_pairs, selected_text
from src.reverse_retrieval import source_path, write_json


def main():
    root=Path('artifacts/dense-v1/train_s1')
    meta=json.loads((root/'manifest.json').read_text())
    if not meta['complete']:raise ValueError('incomplete reference cache')
    ids=(root/'ids.tsv').read_text().splitlines()
    vectors=np.memmap(root/'vectors.f16',dtype='float16',mode='r',shape=(len(ids),DIM))
    indices=[]
    with source_path('train','S1').open(newline='') as f:
        for i,r in enumerate(csv.DictReader(f,delimiter='\t')):
            if r['country']=='India':indices.append(i)
    array=np.asarray(vectors[indices],dtype='float32')
    m,_,_,_=load_pairs('threshold',verify=False);truth=gt(m['query_ids'])
    targets={cid:set() for q,cs in truth.items() for cid in cs}
    for q,cs in truth.items():
        for cid in cs:targets[cid].add(q)
    all_text=selected_text(set(targets))
    chosen=[cid for cid in sorted(targets) if all_text[cid][2]=='India'][:200]
    tok,model=encoder()
    query=np.concatenate([embed([serialize({'business_name':all_text[c][0],'business_address':all_text[c][1],'country':all_text[c][2]}) for c in chosen[i:i+128]],tok,model).astype('float32') for i in range(0,len(chosen),128)])
    t=time.time();ref_gpu=torch.from_numpy(array.astype('float16')).cuda();q_gpu=torch.from_numpy(query.astype('float16')).cuda()
    exact=(q_gpu@ref_gpu.T).topk(50,dim=1).indices.cpu().numpy()
    exact_seconds=time.time()-t
    t=time.time();quantizer=faiss.IndexFlatIP(DIM);ivf=faiss.IndexIVFFlat(quantizer,DIM,1024,faiss.METRIC_INNER_PRODUCT)
    faiss.omp_set_num_threads(8)
    rng=np.random.default_rng(42);training=array[rng.choice(len(array),size=50000,replace=False)]
    ivf.train(training);ivf.add(array);build_seconds=time.time()-t
    result={'country':'India','reference_rows':len(array),'queries':len(chosen),'nlist':1024,'exact_seconds':exact_seconds,'ivf_build_seconds':build_seconds,'probes':{}}
    for probe in (16,64):
        ivf.nprobe=probe;t=time.time();_,approx=ivf.search(query,50);elapsed=time.time()-t
        exact20=np.mean([len(set(a[:20])&set(b[:20]))/20 for a,b in zip(exact,approx)])
        exact50=np.mean([len(set(a)&set(b))/50 for a,b in zip(exact,approx)])
        true_exact=true_approx=0
        for cid,exact_row,approx_row in zip(chosen,exact,approx):
            true_exact+=sum(ids[indices[j]] in targets[cid] for j in exact_row[:20])
            true_approx+=sum(ids[indices[j]] in targets[cid] for j in approx_row[:20] if j>=0)
        result['probes'][str(probe)]={'seconds':elapsed,'query_rate':len(chosen)/elapsed,'neighbor_recall_at_20':float(exact20),
                                       'neighbor_recall_at_50':float(exact50),'true_positive_edges_exact20':true_exact,
                                       'true_positive_edges_approx20':true_approx}
    write_json(Path('artifacts/dense-v1/ivf_benchmark.json'),result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
