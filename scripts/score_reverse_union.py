"""Score diagnostic positive-endpoint rank files; never exports these as candidates."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from src.retrieval_experiments import gt,load_pairs,score,write_json


def rank_map(path):
    return {(r['s1_id'],r['source_record_id']):r['rank'] for r in pq.read_table(path).to_pylist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--lexical',type=Path,default=Path('artifacts/reverse-v1/pilot_v3_all/positive_ranks.parquet'))
    p.add_argument('--output',type=Path,default=Path('artifacts/dense-v1/pilot/reverse_union.json'))
    a=p.parse_args()
    m,baseline,_,_=load_pairs('threshold',verify=False)
    truth=gt(m['query_ids'])
    lexical=rank_map(a.lexical)
    dense=rank_map(Path('artifacts/dense-v1/pilot/positive_ranks.parquet'))
    results={}
    for lk,dk in ((20,20),(20,50),(50,20),(50,50)):
        groups={q:set(baseline[q]) for q in m['query_ids']}
        lex_only={q:set() for q in m['query_ids']}
        dense_only={q:set() for q in m['query_ids']}
        both={q:set() for q in m['query_ids']}
        for q,ids in truth.items():
            for cid in ids:
                l=lexical.get((q,cid));d=dense.get((q,cid))
                if l is not None and l<=lk:lex_only[q].add(cid)
                if d is not None and d<=dk:dense_only[q].add(cid)
                if cid in lex_only[q] or cid in dense_only[q]:both[q].add(cid);groups[q].add(cid)
        result={'baseline_union':score(m,groups,truth),'reverse_only':score(m,both,truth),
                'lexical_only':score(m,lex_only,truth),'dense_only':score(m,dense_only,truth),
                'recovered_positives':sum(len(groups[q]-baseline[q]) for q in groups)}
        for name in ('baseline_union','reverse_only','lexical_only','dense_only'):
            result[name].pop('candidate_count',None)
        results[f'lex{lk}_dense{dk}']=result
    write_json(a.output,{'diagnostic_only':True,'results':results})
    print(json.dumps({k:{'baseline_union':v['baseline_union']['oracle_macro_f05'],
                         'reverse_only':v['reverse_only']['oracle_macro_f05'],
                         'recovered_positives':v['recovered_positives']} for k,v in results.items()},indent=2))

if __name__=='__main__':main()
