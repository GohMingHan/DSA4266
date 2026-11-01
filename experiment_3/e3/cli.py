from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
from e2b.io import load_config as load_e2b_config,canonical_cohort,check_expected,SPLITS
from e2b.prepare import read_export_split
from .common import load_config,read_json,write_json


def check(config):
    c=load_e2b_config(load_config(config)['experiment_2b_config'])
    rows=canonical_cohort(c['cohort']);check_expected(rows,('train','dev'),c.get('expected_counts'))
    report={'experiment_2b_config':c['_path'],'cohort_counts':{s:sum(r['split']==s for r in rows) for s in SPLITS},
            'embedding_caches':{},'ready_to_prepare':True,
            'note':'Index/path check only. prepare reads and verifies all feature arrays. No model selected or trained here.'}
    for m in ('visual','audio'):
        path=Path(c[m+'_embeddings']);state={'path':str(path),'splits':{}}
        try:
            for split in ('train','dev'):
                info,index=read_export_split(path,split,m)
                expected={r['file'] for r in rows if r['split']==split}
                if set(index)!=expected:raise ValueError(f'{split}: cache/cohort video IDs do not match.')
                state['splits'][split]={'videos':len(index),'encoder_id':info['encoder_id']}
            state['status']='ready'
        except (OSError,ValueError,KeyError) as exc:
            state.update(status='not_ready',reason=str(exc));report['ready_to_prepare']=False
        report['embedding_caches'][m]=state
    print(json.dumps(report,indent=2))
    if not report['ready_to_prepare']:raise SystemExit(2)
    return report


def model_args(p):
    p.add_argument('--mode',choices=['cross','self','concat'],default='cross')
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--layers',type=int,default=1)
    p.add_argument('--ff-multiplier',type=int,default=2)
    p.add_argument('--head-width',type=int,default=128)
    p.add_argument('--dropout',type=float,default=.2)
    p.add_argument('--time-encoding',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--attention-backend',choices=['math','flash-math'],default='math')

def runtime_args(p):
    p.add_argument('--device',default='auto')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--cache-videos',type=int,default=4)


def main():
    p=argparse.ArgumentParser(description='Experiment 3: frozen timestamped embeddings -> bidirectional cross-attention -> any-manipulation classifier.')
    subs=p.add_subparsers(dest='command',required=True)
    q=subs.add_parser('check');q.add_argument('--config',default='config.json')
    q=subs.add_parser('extract',help='Reuse the unchanged Experiment 2B encoder bridges/config; no new model choices.')
    q.add_argument('--config',default='config.json');q.add_argument('--modality',choices=['visual','audio','both'],default='both')
    q.add_argument('--splits',nargs='+',choices=SPLITS,default=['train','dev']);q.add_argument('--out')
    q.add_argument('--device',default='cuda:0');q.add_argument('--limit-per-split',type=int,default=0);q.add_argument('--dry-run',action='store_true')
    q=subs.add_parser('prepare');q.add_argument('--config',default='config.json');q.add_argument('--out',required=True)
    q.add_argument('--splits',nargs='+',choices=SPLITS,default=['train','dev'])
    q.add_argument('--e2b-data',help='Optional existing E2B mean-pooled paired data; assert identical pooled inputs too.')
    q=subs.add_parser('train');q.add_argument('--data',required=True);q.add_argument('--out',required=True)
    model_args(q);runtime_args(q)
    q.add_argument('--epochs',type=int,default=50);q.add_argument('--patience',type=int,default=8)
    q.add_argument('--lr',type=float,default=1e-3);q.add_argument('--weight-decay',type=float,default=1e-4)
    q.add_argument('--balance',choices=['class','none'],default='class');q.add_argument('--seed',type=int,default=42)
    q.add_argument('--resume');q.add_argument('--amp',choices=['none'],default='none',help='FP32 only in this initial tested implementation.')
    q=subs.add_parser('smoke');q.add_argument('--data',required=True);model_args(q);runtime_args(q)
    q.add_argument('--steps',type=int,default=2);q.add_argument('--seed',type=int,default=42)
    q.set_defaults(epochs=1,patience=0,lr=1e-3,weight_decay=1e-4,balance='class')
    q=subs.add_parser('evaluate');q.add_argument('--data',required=True);q.add_argument('--checkpoint',required=True)
    q.add_argument('--split',choices=['dev','test'],default='dev');q.add_argument('--out',required=True);runtime_args(q)
    q=subs.add_parser('compare');q.add_argument('--runs',nargs='+',required=True);q.add_argument('--out',required=True)
    q.add_argument('--bootstrap',type=int,default=0);q.add_argument('--seed',type=int,default=42)
    q.add_argument('--allow-training-differences',action='store_true')
    a=p.parse_args()
    for key in ('batch_size','threads','steps','width','heads','layers','head_width','ff_multiplier','epochs'):
        if hasattr(a,key) and getattr(a,key)<1:p.error(f'{key} must be positive.')
    for key in ('workers','cache_videos','patience','limit_per_split','bootstrap'):
        if hasattr(a,key) and getattr(a,key)<0:p.error(f'{key} must not be negative.')
    if hasattr(a,'lr') and (not math.isfinite(a.lr) or a.lr<=0 or not math.isfinite(a.weight_decay) or a.weight_decay<0):p.error('Invalid learning rate/weight decay.')
    if hasattr(a,'dropout') and not 0<=a.dropout<1:p.error('dropout must be in [0,1).')
    if hasattr(a,'width') and (a.width<4 or a.width%2 or a.width%a.heads):p.error('width must be even, >=4, and divisible by heads.')
    if hasattr(a,'splits'):a.splits=list(dict.fromkeys(a.splits))
    if a.command=='check':return check(a.config)
    if a.command=='extract':
        from e2b.cli import extract
        a.config=load_config(a.config)['experiment_2b_config'];return extract(a)
    if a.command=='prepare':
        from .data import prepare
        return prepare(a.config,a.out,a.splits,a.e2b_data)
    if a.command=='compare':
        from .compare import compare
        return compare(a.runs,a.out,a.bootstrap,a.seed,a.allow_training_differences)
    from . import train as trainer
    return getattr(trainer,a.command)(a)
