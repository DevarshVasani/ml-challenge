"""Single aggregate-only final evaluation of frozen reverse policy."""
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.dense_retrieval import DIM,embed,encoder,serialize
from src.reverse_retrieval import ReverseIndex,source_path,write_json
from src.retrieval_experiments import gt,load_pairs,score,selected_text

LEX_K=50
DENSE_K=20


def main():
    started=time.time();m,baseline,_,_=load_pairs('final',verify=False);truth=gt(m['query_ids'])
    endpoints=sorted({cid for ids in truth.values() for cid in ids});text=selected_text(endpoints)
    index=ReverseIndex(Path('artifacts/reverse-v1'),'train')
    lex={}
    for cid in endpoints:
        n,a,c=text[cid]
        lex[cid]={r['s1_id'] for r in index.query({'business_name':n,'business_address':a,'country':c},LEX_K)}
    index.close()
    root=Path('artifacts/dense-v1/train_s1');meta=json.loads((root/'manifest.json').read_text())
    if not meta['complete']:raise ValueError('reference cache incomplete')
    ids=(root/'ids.tsv').read_text().splitlines();vectors=np.memmap(root/'vectors.f16',dtype='float16',mode='r',shape=(len(ids),DIM))
    countries=defaultdict(list)
    with source_path('train','S1').open(newline='') as f:
        for i,r in enumerate(csv.DictReader(f,delimiter='\t')):countries[r['country']].append(i)
    tok,model=encoder();dense={}
    for country,indices in countries.items():
        chosen=[cid for cid in endpoints if text[cid][2]==country]
        if not chosen:continue
        matrix=torch.from_numpy(np.asarray(vectors[indices])).cuda()
        for start in range(0,len(chosen),128):
            chunk=chosen[start:start+128]
            q=torch.from_numpy(embed([serialize({'business_name':text[c][0],'business_address':text[c][1],'country':text[c][2]}) for c in chunk],tok,model)).cuda()
            near=(q@matrix.T).topk(DENSE_K,dim=1).indices.cpu().tolist()
            for cid,row in zip(chunk,near):dense[cid]={ids[indices[j]] for j in row}
        del matrix
    reverse={q:set() for q in m['query_ids']};merged={q:set(baseline[q]) for q in m['query_ids']}
    for q,positives in truth.items():
        for cid in positives:
            if q in lex[cid] or q in dense[cid]:reverse[q].add(cid);merged[q].add(cid)
    reverse_metrics=score(m,reverse,truth);merged_metrics=score(m,merged,truth)
    reverse_metrics.pop('candidate_count',None);merged_metrics.pop('candidate_count',None)
    result={'policy':{'lexical_top_k':LEX_K,'dense_top_k':DENSE_K},'positive_only_diagnostic':True,
            'reverse_only':reverse_metrics,'historical_baseline_union':merged_metrics,
            'newly_recovered_positives':sum(len(merged[q]-baseline[q]) for q in merged),
            'seconds':time.time()-started}
    write_json(Path('artifacts/dense-v1/frozen_final_aggregate.json'),result)
    print(json.dumps({'reverse_oracle':reverse_metrics['oracle_macro_f05'],'baseline_union_oracle':merged_metrics['oracle_macro_f05'],
                      'newly_recovered_positives':result['newly_recovered_positives'],'seconds':result['seconds']}))

if __name__=='__main__':main()
