from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .io import VERSION,SPLITS,load_config,load_slot,canonical_cohort,check_expected,resolve_path


def check(config):
    c=load_config(config);rows=canonical_cohort(c['cohort']);counts={s:sum(r['split']==s for r in rows) for s in SPLITS}
    check_expected(rows,[s for s,n in counts.items() if n],c.get('expected_counts'))
    states={};failed=False
    for m in ('visual','audio'):
        try:
            s=load_slot(c[m+'_slot']);states[m]={'status':'configured','kind':s['kind'],'checkpoint':s['checkpoint']}
            required={'experiment_2a':['project_dir','input_root'],'forensics_adapter':['project_dir','input_manifest','clip_weights'],
                      'ummaformer':['project_dir','input_manifest','upstream'],'preexported':['index'],'custom':['adapter_file']}[s['kind']]
            for key in required:
                if not s.get(key) or not Path(s[key]).exists():raise FileNotFoundError(f'{m} {key}: missing or not configured.')
            states[m]['note']='Path check only. Extraction performs strict checkpoint/source/schema and input checks.'
        except (ValueError,FileNotFoundError,KeyError) as e:
            failed=True;states[m]={'status':'not_ready','reason':str(e)}
    print(json.dumps({'cohort_counts':counts,'models':states,'ready_for_extraction':not failed},indent=2))
    if failed:raise SystemExit(2)


def extract(a):
    c=load_config(a.config)
    for modality in (('visual','audio') if a.modality=='both' else (a.modality,)):
        s=load_slot(c[modality+'_slot']);python=s.get('python') or sys.executable
        if '/' in python or '\\' in python:python=str(resolve_path(python,Path(c[modality+'_slot']).parent))
        if not Path(python).is_file() and shutil.which(python) is None:raise FileNotFoundError(f'Encoder Python executable not found: {python}')
        command=[python,'-m','e2b.extract','--config',c['_path'],'--modality',modality,'--splits',*a.splits,
                 '--device',a.device,'--limit-per-split',str(a.limit_per_split)]
        if a.out:
            out=Path(a.out).expanduser().resolve()
            if a.modality=='both':out=out/modality
            command+=['--out',str(out)]
        print('Launching encoder subprocess:',json.dumps(command),flush=True)
        if a.dry_run:continue
        env=dict(os.environ);root=str(Path(__file__).resolve().parent.parent)
        env['PYTHONPATH']=root+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        subprocess.run(command,cwd=root,env=env,check=True)


def main():
    p=argparse.ArgumentParser(description='Experiment 2B: frozen audio/visual embeddings -> independent pooling -> concatenation classifier.')
    sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('check');c.add_argument('--config',default='config.json')
    e=sub.add_parser('extract');e.add_argument('--config',default='config.json')
    e.add_argument('--modality',choices=['visual','audio','both'],default='both')
    e.add_argument('--splits',nargs='+',choices=SPLITS,default=['train','dev']);e.add_argument('--device',default='cuda:0')
    e.add_argument('--limit-per-split',type=int,default=0);e.add_argument('--out');e.add_argument('--dry-run',action='store_true')
    d=sub.add_parser('prepare');d.add_argument('--config',default='config.json');d.add_argument('--out',required=True)
    d.add_argument('--splits',nargs='+',choices=SPLITS,default=['train','dev']);d.add_argument('--pooling',choices=['mean','mean_max'],default='mean')
    t=sub.add_parser('train');t.add_argument('--data',required=True);t.add_argument('--out',required=True)
    t.add_argument('--mode',choices=['concat','visual','audio'],default='concat')
    t.add_argument('--device',default='auto');t.add_argument('--batch-size',type=int,default=64)
    t.add_argument('--epochs',type=int,default=50);t.add_argument('--patience',type=int,default=8)
    t.add_argument('--lr',type=float,default=1e-3);t.add_argument('--weight-decay',type=float,default=1e-4)
    t.add_argument('--hidden',type=int,default=128);t.add_argument('--dropout',type=float,default=.2)
    t.add_argument('--balance',choices=['class','none'],default='class');t.add_argument('--seed',type=int,default=42)
    t.add_argument('--threads',type=int,default=4);t.add_argument('--resume');t.add_argument('--smoke-test',action='store_true',help='Only allow synthetic test fixtures; never a research run.')
    v=sub.add_parser('evaluate');v.add_argument('--data',required=True);v.add_argument('--checkpoint',required=True);v.add_argument('--out',required=True)
    v.add_argument('--split',choices=['dev','test'],default='dev');v.add_argument('--device',default='auto')
    v.add_argument('--batch-size',type=int,default=128);v.add_argument('--threads',type=int,default=4);v.add_argument('--smoke-test',action='store_true')
    b=sub.add_parser('compare');b.add_argument('--runs',nargs='+',required=True);b.add_argument('--out',required=True)
    b.add_argument('--bootstrap',type=int,default=0);b.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if hasattr(a,'splits'):a.splits=list(dict.fromkeys(a.splits))
    if getattr(a,'batch_size',1)<1 or getattr(a,'threads',1)<1 or getattr(a,'limit_per_split',0)<0 or getattr(a,'bootstrap',0)<0:p.error('Invalid count argument.')
    if a.command=='train':
        import math
        if a.epochs<1 or a.patience<0 or a.hidden<1 or not 0<=a.dropout<1 or not math.isfinite(a.lr) or a.lr<=0 or not math.isfinite(a.weight_decay) or a.weight_decay<0:
            p.error('Invalid model/training settings.')
    if a.command=='check':check(a.config)
    elif a.command=='extract':extract(a)
    elif a.command=='prepare':
        from .prepare import prepare
        prepare(a.config,a.out,a.splits,a.pooling)
    elif a.command=='train':
        from .train import train
        train(a)
    elif a.command=='evaluate':
        from .train import evaluate
        evaluate(a)
    else:
        from .compare import compare
        compare(a.runs,a.out,a.bootstrap,a.seed)
