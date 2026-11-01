#!/usr/bin/env python3
"""Measure native cached-feature training for either project, in a fresh process.

No pretrained feature encoder is present. No production checkpoint is loaded or
saved. One real batch is reused to benchmark compute; this is NOT full-epoch I/O
or accuracy benchmarking. Existing project/upstream files are never edited.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--warmup-steps',type=int,default=3)
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.batch_size<2 or a.batch_size%2 or a.warmup_steps<1 or a.steps<2:p.error('Use even batch >=2, positive warmup and >=2 timed steps.')
    if a.out.exists():p.error('Use a new benchmark output file.')
    project=a.project.expanduser().resolve()
    sys.path.insert(0,str(project))
    from run_experiment import get_device,seed_all,scaler_for,train_update
    from visual_data import load_manifest,VisualDataset,MixedBatchSampler
    from upstream_adapter import build_official
    seed_all(42);device=get_device(a.device,'none')
    manifest=load_manifest(a.manifest)
    if manifest.get('test_only'):raise ValueError('Benchmark expects the actual pretrained feature cache.')
    ds=VisualDataset(manifest,'train',input_mode='rgb-flow')
    sampler=MixedBatchSampler(ds.rows,a.batch_size,42)
    batch=[ds[i] for i in next(iter(sampler))]
    model,cfg,utils,identity,counts=build_official(a.upstream,device,'rgb-flow')
    optimizer=utils.make_optimizer(model,cfg['opt']);scaler=scaler_for('none')
    durations=[]
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    for i in range(a.warmup_steps+a.steps):
        if i==a.warmup_steps and device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        sync();start=time.perf_counter()
        _,info=train_update(model,batch,optimizer,scaler,device,'none')
        sync();elapsed=time.perf_counter()-start
        if info['skipped']:raise ValueError('Skipped optimizer update; benchmark not comparable.')
        print(f'{"warmup" if i<a.warmup_steps else "timed"} step {i+1}: {elapsed:.4f} s',flush=True)
        if i>=a.warmup_steps:durations.append(elapsed)
    report={'scope':'temporary native model, repeated real batch; excludes dataset I/O, EMA, dev and checkpoint saving',
            'project':str(project),'manifest':str(a.manifest.resolve()),'input_channels':cfg['model']['input_dim'],
            'batch_size':a.batch_size,'steps':a.steps,'warmup_steps':a.warmup_steps,
            'median_seconds_per_batch':statistics.median(durations),'mean_seconds_per_batch':statistics.mean(durations),
            'samples_per_second_at_median':a.batch_size/statistics.median(durations),
            'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None,
            'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'torch':str(torch.__version__),'amp':'none','parameters':counts,'upstream_commit':identity['git_commit']}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
