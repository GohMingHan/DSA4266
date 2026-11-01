#!/usr/bin/env python3
"""Start a NEW TSN-flow UMMAFormer using the saved baseline training settings.

The baseline checkpoint supplies settings and cohort verification ONLY. Its model
weights do not initialize the new 2560-channel model. No original files change.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import torch
from extract_tsn_flow import read_reference
from visual_data import load_manifest
from upstream_adapter import split_identity, source_identity


def build_command(args):
    base=torch.load(args.baseline_checkpoint,map_location='cpu',weights_only=True)
    if base.get('format')!='ummaformer_opencv_visual_v2' or base.get('settings',{}).get('input_mode')!='rgb-flow':
        raise ValueError('Expected your old RGB+pooled-Farneback baseline checkpoint.')
    reference=read_reference(args.reference_manifest)
    now=split_identity(reference)
    for split in ('train','dev'):
        if base['split_identity'].get(split)!=now.get(split):raise ValueError(f'Baseline {split} feature cohort differs from reference manifest.')
    m=load_manifest(args.manifest)
    if m.get('test_only'):raise ValueError('Test-only flow cache cannot be used for research training.')
    keys=('file','split','source_group','modify_video','modify_audio','visual_periods','duration_s')
    refrows={r['file']:r for r in reference['videos'] if r['split'] in ('train','dev')}
    newrows={r['file']:r for r in m['videos'] if r['split'] in ('train','dev')}
    if set(refrows)!=set(newrows):raise ValueError('Training/development IDs differ. Do not train on the extraction pilot as the full study.')
    for f,r in newrows.items():
        if any(r[k]!=refrows[f][k] for k in keys) or r['reference_cache_sha256']!=refrows[f]['cache']['sha256']:
            raise ValueError(f'Cohort/timestamp/RGB provenance mismatch: {f}')
    if source_identity(args.upstream)['source_digest']!=base['upstream']['source_digest']:
        raise ValueError('Official UMMAFormer implementation differs from the baseline. Reuse its unchanged upstream checkout.')
    s=base['settings'].copy()
    batch=args.batch_size if args.batch_size is not None else s['batch_size']
    cmd=[sys.executable,str(Path(__file__).parent/'run_experiment.py'),'train',
         '--manifest',str(args.manifest.resolve()),'--input-mode','rgb-flow',
         '--upstream',str(args.upstream.resolve()),'--out',str(args.out.resolve()),
         '--device',args.device,'--workers',str(args.workers)]
    for key in ('epochs','warmup_epochs','lr','weight_decay','patience','seed','amp','max_consecutive_skips'):
        cmd += ['--'+key.replace('_','-'),str(s[key])]
    cmd += ['--batch-size',str(batch),'--ema' if s['ema'] else '--no-ema']
    if args.resume:cmd+=['--resume',str(args.resume.resolve())]
    report={'baseline_checkpoint':str(args.baseline_checkpoint.resolve()),
            'baseline_best_epoch':base['epoch'],'baseline_dev_auc':base['dev_metrics']['video_visual_roc_auc'],
            'copied_settings':s,'actual_new_batch_size':batch,
            'batch_override_changes_protocol':batch!=s['batch_size'],
            'new_input_channels':2560,'new_model_initialized_from_scratch':True,
            'feature_encoders_frozen_and_cached':True,'baseline_weights_loaded_into_new_model':False,
            'training_videos':sum(r['split']=='train' for r in newrows.values()),
            'development_videos':sum(r['split']=='dev' for r in newrows.values())}
    return cmd,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-checkpoint',type=Path,required=True)
    p.add_argument('--reference-manifest',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--batch-size',type=int,help='Optional explicit memory-driven change; reported as unmatched batching.')
    p.add_argument('--resume',type=Path);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.workers<0 or a.batch_size is not None and (a.batch_size<2 or a.batch_size%2):p.error('Invalid workers/batch size.')
    cmd,report=build_command(a)
    print(json.dumps(report,indent=2),flush=True);print(shlex.join(cmd),flush=True)
    if not a.dry_run:subprocess.run(cmd,check=True)

if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError,subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(2)
