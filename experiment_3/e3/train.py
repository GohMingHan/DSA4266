"""Train/evaluate a fusion head only. DEV selects checkpoint and threshold; TEST does neither."""
from __future__ import annotations
from functools import partial
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from e2b.metrics import evaluate_rows, f1_threshold, sigmoid
from .common import (VERSION,TARGET,source_hashes,device_of,seed_all,seed_worker,rng_get,rng_set,
    atomic_save,backend_context,synchronize,read_json,write_json,write_jsonl,require_new,sha256,lock)
from .data import load_index,Sequences,collate,identity_rows
from .model import build_model

SETTING_NAMES=('mode','width','heads','layers','ff_multiplier','head_width','dropout','time_encoding',
               'batch_size','epochs','patience','lr','weight_decay','seed','balance','attention_backend')

def settings_of(a):return {k:getattr(a,k) for k in SETTING_NAMES}

def to_device(batch,device):return {k:v.to(device,non_blocking=True) for k,v in batch.items() if k!='index'}

def make_loader(ds,batch_size,workers,seed,training,balance='class',pin=False):
    g=torch.Generator().manual_seed(seed)
    params=dict(batch_size=batch_size,num_workers=workers,collate_fn=collate,
                worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(seed+19271),pin_memory=pin)
    if training:
        if balance=='class':
            y=np.array([r['label'] for r in ds.rows]);counts=np.bincount(y,minlength=2)
            if (counts==0).any():raise ValueError('Training needs both binary classes.')
            sampler=WeightedRandomSampler(torch.tensor(1./counts[y],dtype=torch.double),len(y),replacement=True,generator=g)
        else:sampler=torch.utils.data.RandomSampler(ds,generator=g)
        return DataLoader(ds,sampler=sampler,**params)
    return DataLoader(ds,shuffle=False,**params)

@torch.no_grad()
def predict(model,ds,device,batch_size,workers=0):
    model.eval();scores=np.full(len(ds),np.nan);seen=np.zeros(len(ds),bool)
    for batch in tqdm(make_loader(ds,batch_size,workers,271828,False,pin=device.type=='cuda'),desc='Evaluate',unit='batch',leave=False):
        ids=batch['index'].numpy()
        if seen[ids].any() or len(np.unique(ids))!=len(ids):raise ValueError('Duplicate prediction indices.')
        z=model(to_device(batch,device))
        if z.shape!=(len(ids),) or not torch.isfinite(z).all():raise FloatingPointError('Nonfinite/invalid model output.')
        scores[ids]=z.detach().double().cpu().numpy();seen[ids]=True
    if not seen.all() or not np.isfinite(scores).all():raise ValueError('Missing predictions.')
    return scores

def save_predictions(path,rows,scores,threshold):
    rows=identity_rows(rows)
    write_jsonl(path,[{**r,'logit':float(z),'score':float(p),'prediction':int(z>=threshold)}
                     for r,z,p in zip(rows,scores,sigmoid(scores))])

def compare_data_checkpoint(c,cfg):
    if c.get('format')!=VERSION or c.get('target')!=TARGET:raise ValueError('Not an Experiment 3 checkpoint.')
    for key in ('encoder_ids','dims'):
        if c[key]!=cfg[key]:raise ValueError(f'Changed {key}; regenerate the comparison with the selected encoder pair.')
    for split in ('train','dev'):
        if cfg['splits'].get(split,{}).get('signature')!=c['split_ids'][split]:
            raise ValueError(f'{split} cohort/sequence embeddings changed after training.')
    if cfg['normalization']['npz_sha256']!=c['normalization_sha256']:raise ValueError('Saved normalization changed.')

def train(a,allow_test_only=False,stop_after_epoch=None):
    device=device_of(a.device);torch.set_num_threads(a.threads);seed_all(a.seed)
    cfg,rows,norm=load_index(a.data,'train',allow_test_only)
    dc,devrows,_=load_index(a.data,'dev',allow_test_only)
    if any(len({r['label'] for r in rs})!=2 for rs in (rows,devrows)):raise ValueError('TRAIN and DEV must include both labels.')
    settings=settings_of(a);model=build_model(cfg['dims'],norm,settings).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    ds=Sequences(cfg,rows,a.cache_videos);dd=Sequences(cfg,devrows,a.cache_videos)
    out=Path(a.out).expanduser().resolve();start=0;best=-float('inf');stale=0;history=[]
    hashes=source_hashes();normalization={m:{k:t.clone() for k,t in st.items()} for m,st in norm.items()}
    if a.resume:
        c=torch.load(a.resume,map_location='cpu',weights_only=True);compare_data_checkpoint(c,cfg)
        if c['settings']!=settings or c['source_hashes']!=hashes:raise ValueError('Resume needs the same settings and source version.')
        if c['run_dir']!=str(out) or not (out/'best.pt').is_file():raise ValueError('Resume inside the original run with best.pt present.')
        model.load_state_dict(c['model'],strict=True);opt.load_state_dict(c['optimizer'])
        start=c['epoch'];best=c['best_auc'];stale=c['stale'];history=c['history'];rng_set(c['rng'])
    else:require_new(out)
    info={'format':VERSION,'target':TARGET,'settings':settings,'encoder_ids':cfg['encoder_ids'],'dims':cfg['dims'],
        'parameters':sum(p.numel() for p in model.parameters()),'train_videos':len(rows),'dev_videos':len(devrows),
        'precision':'float32','device':str(device),'torch':str(torch.__version__),'workers':a.workers,
        'source_hashes':hashes,'test_only':cfg['test_only'],'temporal_policy':cfg['temporal_policy'],
        'time_encoding':'same continuous seconds; fixed sinusoid periods0.5..512s; audio window centers',
        'normalization':cfg['normalization'],'loss':'unweighted BCEWithLogitsLoss',
        'sampling':'binary class-balanced video draws with replacement' if a.balance=='class' else 'shuffle each training video',
        'e2b_comparison_note':'concat mode uses the original E2B MLP and pooled-input normalization; cross/self use all tokens',
        'source_encoder_weights_updated':False}
    if a.patience and stale>=a.patience:print('Already early-stopped; no more epochs.');return history
    if start>=a.epochs:print('Training horizon already reached.');return history
    with lock(out,'.e3.lock'),backend_context(a.attention_backend):
        write_json(out/'run_config.json',info)
        write_json(out/'normalization_train.json',{m:{k:t.tolist() for k,t in st.items()} for m,st in norm.items()})
        try:
            for epoch in range(start,a.epochs):
                synchronize(device);begin=time.perf_counter();model.train()
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                loader=make_loader(ds,a.batch_size,a.workers,a.seed+epoch,True,a.balance,pin=device.type=='cuda')
                loss_sum=0.;seen=0;updates=0;counts=[0,0]
                for batch in tqdm(loader,desc=f'Epoch {epoch+1}/{a.epochs}',unit='batch'):
                    b=to_device(batch,device);opt.zero_grad(set_to_none=True)
                    loss=nn.functional.binary_cross_entropy_with_logits(model(b),b['label'])
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; no invalid update applied.')
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
                    n=len(batch['label']);positives=int(batch['label'].sum())
                    counts[1]+=positives;counts[0]+=n-positives;updates+=1;seen+=n;loss_sum+=float(loss.detach())*n
                synchronize(device);train_seconds=time.perf_counter()-begin
                scores=predict(model,dd,device,a.batch_size,a.workers)
                threshold=f1_threshold([r['label'] for r in devrows],scores)
                metrics=evaluate_rows(devrows,scores,threshold);auc=metrics['video_any_roc_auc']
                improved=auc>best
                if improved:best=auc;stale=0
                else:stale+=1
                synchronize(device)
                row={'epoch':epoch+1,'train_loss':loss_sum/seen,'optimizer_steps':updates,
                     'sampled_videos':seen,'sampled_class_counts':counts,'dev_video_any_roc_auc':auc,
                     'dev_video_any_ap':metrics['video_any_ap'],'stale_evaluations':stale,
                     'train_seconds':train_seconds,'seconds_before_checkpoint':time.perf_counter()-begin,
                     'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None}
                history.append(row);print(json.dumps(row),flush=True)
                state={'format':VERSION,'target':TARGET,'model':{k:t.detach().cpu().clone() for k,t in model.state_dict().items()},
                    'normalization':normalization,'normalization_sha256':cfg['normalization']['npz_sha256'],
                    'settings':settings,'dims':cfg['dims'],'encoder_ids':cfg['encoder_ids'],'encoder_provenance':cfg['encoder_provenance'],
                    'split_ids':{s:cfg['splits'][s]['signature'] for s in ('train','dev')},
                    'e2b_split_ids':{s:cfg['splits'][s]['e2b_mean_signature'] for s in ('train','dev')},
                    'epoch':epoch+1,'threshold_logit':threshold,'dev_metrics':metrics,
                    'test_only':cfg['test_only'],'source_hashes':hashes,'run_dir':str(out),'parameters':info['parameters']}
                if improved:
                    atomic_save(out/'best.pt',state);write_json(out/'best_dev_metrics.json',metrics)
                    save_predictions(out/'best_dev_predictions.jsonl',devrows,scores,threshold)
                state.update(optimizer=opt.state_dict(),rng=rng_get(),best_auc=best,stale=stale,history=history)
                atomic_save(out/'last.pt',state)
                # Commit history after checkpoint, not proof of a saved epoch before checkpoint write.
                write_json(out/'history.json',history)
                if stop_after_epoch is not None and epoch+1>=stop_after_epoch:break  # test-only API; not a CLI research option
                if a.patience and stale>=a.patience:print('Early stopping on DEV video any-manipulation AUC.');break
        except KeyboardInterrupt:
            print('\nInterrupted. Resume the last completed epoch using last.pt; no incomplete-epoch checkpoint saved.')
            raise
    print(f'Best fusion head: {out}/best.pt. Keep both selected encoders for new-video inference.')
    return history


def evaluate(a,allow_test_only=False):
    d=device_of(a.device);torch.set_num_threads(a.threads)
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    cfg,rows,norm=load_index(a.data,a.split,allow_test_only);compare_data_checkpoint(c,cfg)
    if c.get('test_only') and not allow_test_only:raise ValueError('Synthetic checkpoint is not a research result.')
    if c['source_hashes']!=source_hashes():raise ValueError('Source changed since training; preserve the run-specific code.')
    model=build_model(c['dims'],c['normalization'],c['settings']).to(d);model.load_state_dict(c['model'],strict=True)
    begin=time.perf_counter()
    with backend_context(c['settings']['attention_backend']):
        scores=predict(model,Sequences(cfg,rows,a.cache_videos),d,a.batch_size,a.workers)
    metrics=evaluate_rows(rows,scores,c['threshold_logit'])
    metrics.update(format=VERSION,split=a.split,mode=c['settings']['mode'],settings=c['settings'],
        parameters=c['parameters'],checkpoint_sha256=sha256(a.checkpoint),checkpoint_epoch=c['epoch'],
        threshold_source='checkpoint_DEV_only',evaluated_split_id=cfg['splits'][a.split]['signature'],
        train_dev_split_ids=c['split_ids'],e2b_evaluated_split_id=cfg['splits'][a.split]['e2b_mean_signature'],
        e2b_train_dev_split_ids=c['e2b_split_ids'],encoder_ids=c['encoder_ids'],normalization=c['normalization_sha256'],
        temporal_policy=cfg['temporal_policy'],test_only=c['test_only'],seconds=time.perf_counter()-begin)
    if a.split=='dev':
        for key in ('video_any_roc_auc','video_any_ap'):
            if not np.isclose(metrics[key],c['dev_metrics'][key],rtol=0,atol=1e-6):
                raise ValueError(f'DEV parity check failed for {key}. Check backend/hardware/batch behavior; no report published.')
    out=Path(a.out).expanduser().resolve();require_new(out)
    write_json(out/'metrics.json',metrics);save_predictions(out/'video_predictions.jsonl',rows,scores,c['threshold_logit'])
    print(json.dumps(metrics,indent=2));return metrics


def smoke(a,allow_test_only=False):
    d=device_of(a.device);torch.set_num_threads(a.threads);seed_all(a.seed)
    cfg,rows,norm=load_index(a.data,'train',allow_test_only)
    model=build_model(cfg['dims'],norm,settings_of(a)).to(d)
    # Match state names/count against the independent temporal control.
    control=build_model(cfg['dims'],norm,{**settings_of(a),'mode':'self'})
    parameter_match=sum(p.numel() for p in model.parameters())==sum(p.numel() for p in control.parameters()) if a.mode=='cross' else None
    del control
    ids0=[i for i,r in enumerate(rows) if r['label']==0];ids1=[i for i,r in enumerate(rows) if r['label']==1]
    chosen=[]
    for i in range(min(a.batch_size,len(rows))):
        group=ids0 if i%2==0 else ids1
        if not group:raise ValueError('Smoke test requires both classes in the training split.')
        chosen.append(group[(i//2)%len(group)])
    ds=Sequences(cfg,rows,a.cache_videos);batch=collate([ds[i] for i in chosen]);b=to_device(batch,d)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    if d.type=='cuda':torch.cuda.reset_peak_memory_stats(d)
    synchronize(d);start=time.perf_counter()
    with backend_context(a.attention_backend):
        for _ in range(a.steps):
            model.train();opt.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(b),b['label'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite smoke loss.')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
        model.eval()
        with torch.no_grad():
            baseline=model(b)
            changed={k:v.clone() for k,v in b.items()}
            for m in ('visual','audio'):
                changed[m].masked_fill_(b[m+'_padding'].unsqueeze(-1),123456.)
                changed[m+'_time'].masked_fill_(b[m+'_padding'],654321.)
            if not torch.allclose(baseline,model(changed),rtol=1e-5,atol=1e-5):raise ValueError('Padding content affects predictions.')
    synchronize(d)
    report={'status':'smoke_passed','mode':a.mode,'actual_videos_per_step':len(chosen),'steps':a.steps,
        'visual_shape':list(b['visual'].shape),'audio_shape':list(b['audio'].shape),
        'params':sum(p.numel() for p in model.parameters()),'cross_self_parameter_match':parameter_match,
        'seconds':time.perf_counter()-start,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated(d)/1024**2 if d.type=='cuda' else None,
        'checkpoint_saved':False,'development_evaluated':False,'note':'Temporary randomly initialized FUSION HEAD over real frozen-encoder features; no accuracy claim.'}
    print(json.dumps(report,indent=2));return report
