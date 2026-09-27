"""External, resumable union of complete dense and lexical source groups."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from .retrieval_experiments import write_json


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def merge(dense,lexical,output,buckets=8,memory_gb=12):
    dense_meta=json.loads((dense/'manifest.json').read_text())
    lexical_meta=json.loads((lexical/'run_report.json').read_text())
    if not dense_meta['complete'] or not lexical_meta['complete']:raise ValueError('input channel incomplete')
    config={'version':'reverse-union-v1','dense_config_hash':dense_meta['config_hash'],
            'dense_rows':sum(x['pairs'] for x in dense_meta['shards'].values()),
            'lexical_rows':lexical_meta['candidate_pairs'],'buckets':buckets}
    cfg_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    manifest_path=output/'manifest.json';prior=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if prior and prior['config_hash']!=cfg_hash:raise ValueError('incompatible merge output')
    completed=dict(prior.get('buckets',{})) if prior else {}
    db=duckdb.connect()
    db.execute(f"SET memory_limit='{memory_gb}GB'")
    temp=output/'tmp';temp.mkdir(exist_ok=True)
    db.execute(f"SET temp_directory='{temp}'")
    started=time.time()
    for bucket in range(buckets):
        name=f'bucket-{bucket:02d}.parquet';target=output/name
        if name in completed:
            if not target.exists() or sha(target)!=completed[name]['sha256']:raise ValueError(f'corrupt merged shard: {name}')
            continue
        dense_glob=str(dense/'*.parquet');lex_glob=str(lexical/f'bucket-{bucket:02d}'/'*.parquet')
        tmp=output/(name+'.tmp')
        # Each source group hashes to one bucket. Aggregate full pair keys;
        # channel evidence remains separate and labels never enter the rows.
        sql=f"""
        COPY (
          SELECT source, source_record_id, s1_id,
                 max(dense_score) AS dense_score,
                 min(dense_rank) AS dense_rank,
                 max(name_score) AS name_score,
                 min(name_rank) AS name_rank,
                 max(address_score) AS address_score,
                 min(address_rank) AS address_rank,
                 min(source_bucket) AS source_bucket,
                 'reverse-union-v1' AS policy_version
          FROM (
            SELECT source,source_record_id,s1_id,dense_score,dense_rank,
                   NULL::DOUBLE AS name_score,NULL::BIGINT AS name_rank,
                   NULL::DOUBLE AS address_score,NULL::BIGINT AS address_rank,source_bucket
            FROM read_parquet('{dense_glob}') WHERE source_bucket % {buckets}={bucket}
            UNION ALL
            SELECT source,source_record_id,s1_id,
                   NULL::DOUBLE,NULL::BIGINT,name_score,name_rank,address_score,address_rank,source_bucket
            FROM read_parquet('{lex_glob}')
          ) GROUP BY source,source_record_id,s1_id
        ) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        db.execute(sql)
        os.replace(tmp,target)
        row_count=pq.ParquetFile(target).metadata.num_rows
        groups=db.execute(f"SELECT count(*) FROM (SELECT source,source_record_id FROM read_parquet('{target}') GROUP BY source,source_record_id)").fetchone()[0]
        completed[name]={'pairs':row_count,'source_groups':groups,'bytes':target.stat().st_size,'sha256':sha(target)}
        write_json(manifest_path,{'config':config,'config_hash':cfg_hash,'buckets':completed,'complete':False})
        print(json.dumps({'bucket':bucket,'pairs':row_count,'groups':groups,'seconds':time.time()-started}),flush=True)
    db.close()
    result={'config':config,'config_hash':cfg_hash,'buckets':completed,'complete':True,'seconds_this_run':time.time()-started,
            'pairs':sum(x['pairs'] for x in completed.values()),'source_groups':sum(x['source_groups'] for x in completed.values())}
    write_json(manifest_path,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--dense',type=Path,required=True)
    p.add_argument('--lexical',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--memory-gb',type=int,default=12);a=p.parse_args()
    result=merge(a.dense,a.lexical,a.output,memory_gb=a.memory_gb)
    print(json.dumps({'pairs':result['pairs'],'source_groups':result['source_groups'],'complete':result['complete']}))

if __name__=='__main__':main()
