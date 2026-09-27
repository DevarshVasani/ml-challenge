"""Detached, restartable train/test retrieval and merge stage supervisor."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts'


def state(channel,stage,detail=None):
    target=ART/'reverse-v1'/f'{channel}_queue_state.json'
    tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps({'channel':channel,'stage':stage,'detail':detail,'timestamp':time.time()},indent=2)+'\n')
    os.replace(tmp,target)


def complete(path):
    try:return bool(json.loads(path.read_text())['complete'])
    except (FileNotFoundError,KeyError,json.JSONDecodeError):return False


def run(command,log_path,channel,stage):
    state(channel,stage,{'command':command})
    with log_path.open('a') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{stage} failed with exit {result.returncode}; see {log_path}')


def dense(existing_test_pid):
    for split in ('test','train'):
        output=ART/'dense-v1'/f'{split}_candidates'
        marker=output/'manifest.json'
        if split=='test' and existing_test_pid:
            state('dense','waiting_for_existing_test',{'pid':existing_test_pid})
            while Path(f'/proc/{existing_test_pid}').exists():time.sleep(30)
        if complete(marker):continue
        command=[sys.executable,'-m','src.dense_retrieval','retrieve','--split',split,
                 '--out','artifacts/dense-v1','--candidate-out',str(output),'--batch-size','1000','--k','20']
        run(command,ART/f'dense-v1-{split}-production.log','dense',f'retrieve_{split}')
        if not complete(marker):raise RuntimeError(f'{split} dense manifest is incomplete after command')
    state('dense','complete')


def lexical():
    for split in ('test','train'):
        output=ART/'reverse-v1'/f'{split}_lexical'
        report=output/'run_report.json'
        if complete(report):continue
        command=[sys.executable,'-m','scripts.run_lexical_partitions','--split',split,
                 '--partitions',str(ART/'reverse-v1'/f'{split}_partitions'),
                 '--output',str(output),'--k','50','--workers','7']
        run(command,ART/f'reverse-v1-{split}-production.log','lexical',f'retrieve_{split}')
        if not complete(report):raise RuntimeError(f'{split} lexical report is incomplete after command')
    state('lexical','complete')


def merged():
    for split in ('test','train'):
        dense_manifest=ART/'dense-v1'/f'{split}_candidates'/'manifest.json'
        lexical_report=ART/'reverse-v1'/f'{split}_lexical'/'run_report.json'
        output=ART/'reverse-v1'/f'{split}_union'
        if complete(output/'manifest.json'):continue
        state('merge',f'waiting_{split}')
        while not (complete(dense_manifest) and complete(lexical_report)):
            for channel in ('dense','lexical'):
                failure=ART/'reverse-v1'/f'{channel}_queue_state.json'
                if failure.exists() and json.loads(failure.read_text()).get('stage')=='failed':
                    raise RuntimeError(f'{channel} production queue failed')
            time.sleep(30)
        command=[sys.executable,'-m','src.merge_reverse','--dense',str(dense_manifest.parent),
                 '--lexical',str(lexical_report.parent),'--output',str(output),'--memory-gb','12']
        run(command,ART/f'reverse-v1-{split}-merge.log','merge',f'merge_{split}')
        if not complete(output/'manifest.json'):raise RuntimeError(f'{split} union manifest incomplete after command')
    state('merge','complete')


def main():
    p=argparse.ArgumentParser();p.add_argument('channel',choices=['dense','lexical','merge'])
    p.add_argument('--existing-test-pid',type=int,default=0);a=p.parse_args()
    try:
        {'dense':lambda:dense(a.existing_test_pid),'lexical':lexical,'merge':merged}[a.channel]()
    except BaseException as exc:
        state(a.channel,'failed',{'error':repr(exc)})
        raise

if __name__=='__main__':main()
