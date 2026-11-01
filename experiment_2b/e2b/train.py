"""Independent pooling -> independent projections -> concatenation -> MLP.

No cross-attention, autoencoder or encoder fine-tuning. Both unimodal controls
are retrained against the SAME any-manipulation target on the SAME paired cohort.
"""
from __future__ import annotations
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from .io import VERSION,read_json,write_json,write_jsonl,require_new,digest,sha256,code_hashes,lock
from .prepare import load_split
from .metrics import evaluate_rows,f1_threshold,sigmoid


class ConcatClassifier(nn.Module):
    def __init__(self,dv,da,mode='concat',hidden=128,dropout=.2):
        super().__init__()
        if mode not in ('concat','visual','audio'):raise ValueError('Unknown classifier mode.')
        self.mode=mode
        def branch(d):return nn.Sequential(nn.Linear(d,hidden),nn.ReLU(),nn.Dropout(dropout))
        self.visual=branch(dv) if mode!='audio' else None
        self.audio=branch(da) if mode!='visual' else None
        self.head=nn.Sequential(nn.Linear(hidden*(2 if mode=='concat' else 1),hidden),
                                nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))
    def forward(self,v,a):
        hs=[]
        if self.visual is not None:hs.append(self.visual(v))
        if self.audio is not None:hs.append(self.audio(a))
        return self.head(torch.cat(hs,dim=-1) if len(hs)>1 else hs[0]).squeeze(-1)


def device_of(value):
    if value=='auto':value='cuda:0' if torch.cuda.is_available() else 'cpu'
    if value=='cuda':value='cuda:0'
    d=torch.device(value)
    if d.type not in ('cpu','cuda'):raise ValueError('Use CPU or a single CUDA device.')
    if d.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable. Explicitly use --device cpu or enable GPU.')
    return d


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False


def normalizer(x):
    mean=x.astype(np.float64).mean(axis=0);std=x.astype(np.float64).std(axis=0)
    std[std<1e-6]=1.
    return {'mean':torch.tensor(mean,dtype=torch.float32),'std':torch.tensor(std,dtype=torch.float32)}


def normalize(v,stats):return (torch.as_tensor(v,dtype=torch.float32)-stats['mean'])/stats['std']


def atomic_save(path,obj):
    p=Path(path); t=p.with_name(p.name+'.tmp');torch.save(obj,t);t.replace(p)


def rng_get():
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def rng_set(x):
    torch.set_rng_state(x['torch'])
    if torch.cuda.is_available() and x['cuda']:torch.cuda.set_rng_state_all(x['cuda'])


@torch.no_grad()
def predict(model,v,a,device,batch_size):
    model.eval();out=[]
    for start in range(0,len(v),batch_size):
        z=model(v[start:start+batch_size].to(device),a[start:start+batch_size].to(device))
        if not torch.isfinite(z).all():raise FloatingPointError('Nonfinite fusion output.')
        out.append(z.detach().double().cpu().numpy())
    return np.concatenate(out)


def settings_of(a):
    return {k:getattr(a,k) for k in ('mode','hidden','dropout','batch_size','epochs','patience','lr','weight_decay','seed','balance')}


def check_dataset(c,cfg):
    if c.get('format')!=VERSION:raise ValueError('Not an Experiment 2B fusion checkpoint.')
    if c['encoder_ids']!=cfg['encoder_ids'] or c['pooling']!=cfg['pooling'] or c['dims']!=cfg['dims']:
        raise ValueError('Encoder/checkpoint/pooling/dimensions changed. Rebuild features and retrain the fusion head; do not reuse old weights.')
    for split in ('train','dev'):
        if cfg['splits'].get(split,{}).get('signature')!=c['split_ids'][split]:
            raise ValueError(f'{split} cohort/features/labels changed after fusion training.')


def train(a):
    device=device_of(a.device);torch.set_num_threads(a.threads);seed_all(a.seed)
    cfg,train_rows,V,A,y=load_split(a.data,'train',a.smoke_test)
    cfg_dev,dev_rows,DV,DA,dy=load_split(a.data,'dev',a.smoke_test)
    if any(len(np.unique(labels))!=2 for labels in (y,dy)):raise ValueError('Train and DEV need both any-fake classes.')
    stats={'visual':normalizer(V),'audio':normalizer(A)}
    v,au=normalize(V,stats['visual']),normalize(A,stats['audio'])
    dv,da=normalize(DV,stats['visual']),normalize(DA,stats['audio'])
    if not all(torch.isfinite(x).all() for x in (v,au,dv,da)):raise ValueError('Nonfinite standardized features.')
    dims=cfg['dims']; model=ConcatClassifier(dims['visual'],dims['audio'],a.mode,a.hidden,a.dropout).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    out=Path(a.out).expanduser().resolve();settings=settings_of(a)
    start=0;best=-float('inf');stale=0;history=[]
    if a.resume:
        c=torch.load(a.resume,map_location='cpu',weights_only=True);check_dataset(c,cfg)
        if c['settings']!=settings or c['code_hashes']!=code_hashes():raise ValueError('Resume requires identical training settings/source code.')
        if c['run_dir']!=str(out):raise ValueError('Resume in the original run folder.')
        if not (out/'best.pt').is_file():raise ValueError('Missing best.pt in resumed run; preserve all checkpoint files.')
        model.load_state_dict(c['model'],strict=True);opt.load_state_dict(c['optimizer'])
        for mod in ('visual','audio'):
            for k in ('mean','std'):
                if not torch.equal(c['normalization'][mod][k],stats[mod][k]):raise ValueError('Training normalization changed.')
        start=c['epoch'];best=c['best_auc'];stale=c['stale'];history=c['history'];rng_set(c['rng'])
    else:require_new(out)
    run_info={'format':VERSION,'settings':settings,'encoder_ids':cfg['encoder_ids'],'dims':dims,
        'pooling':cfg['pooling'],'split_ids':{s:cfg['splits'][s]['signature'] for s in ('train','dev')},
        'normalization':'per-dimension mean/std fitted only on natural TRAIN video vectors',
        'sampling':'inverse binary class frequency videos with replacement' if a.balance=='class' else 'shuffle every training video once',
        'loss':'unweighted BCEWithLogitsLoss; no additional class-specific penalty',
        'train_videos':len(y),'dev_videos':len(dy),'train_class_counts':np.bincount(y,minlength=2).tolist(),
        'parameters':sum(p.numel() for p in model.parameters()),'device':str(device),'torch':str(torch.__version__),
        'test_only':bool(cfg.get('test_only') or a.smoke_test),'target':cfg['target'],'code_hashes':code_hashes()}
    dataset=TensorDataset(v,au,torch.tensor(y,dtype=torch.float32))
    criterion=nn.BCEWithLogitsLoss()
    if a.patience and stale>=a.patience:print('Checkpoint already early-stopped; no more training.');return
    with lock(out):
        write_json(out/'run_config.json',run_info)
        # Only the checkpoint's saved tensors are used for inference.
        write_json(out/'normalization_train.json',{m:{k:t.tolist() for k,t in st.items()} for m,st in stats.items()})
        try:
            for epoch in range(start,a.epochs):
                begin=time.perf_counter();model.train()
                generator=torch.Generator().manual_seed(a.seed+epoch)
                if a.balance=='class':
                    counts=np.bincount(y,minlength=2)
                    sampler=WeightedRandomSampler(torch.tensor(1./counts[y],dtype=torch.double),len(y),replacement=True,generator=generator)
                    loader=DataLoader(dataset,batch_size=a.batch_size,sampler=sampler,num_workers=0)
                else:
                    loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=True,generator=generator,num_workers=0)
                loss_sum=0.;seen=0;sample_counts=[0,0]
                for bv,ba,by in loader:
                    bv,ba,by=bv.to(device),ba.to(device),by.to(device)
                    opt.zero_grad(set_to_none=True);loss=criterion(model(bv,ba),by)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fusion loss.')
                    loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                    opt.step();loss_sum+=float(loss.detach())*len(by);seen+=len(by)
                    sample_counts[1]+=int(by.sum());sample_counts[0]+=len(by)-int(by.sum())
                logits=predict(model,dv,da,device,a.batch_size)
                threshold=f1_threshold(dy,logits);metrics=evaluate_rows(dev_rows,logits,threshold)
                auc=metrics['video_any_roc_auc'];better=auc>best
                if better:best=auc;stale=0
                else:stale+=1
                row={'epoch':epoch+1,'train_loss':loss_sum/seen,'dev_video_any_roc_auc':auc,
                     'dev_video_any_ap':metrics['video_any_ap'],'sampled_class_counts':sample_counts,
                     'seconds':time.perf_counter()-begin,'stale_evaluations':stale}
                history.append(row);write_json(out/'history.json',history);print(json.dumps(row),flush=True)
                state={'format':VERSION,'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    'normalization':stats,'settings':settings,'dims':dims,'pooling':cfg['pooling'],'encoder_ids':cfg['encoder_ids'],
                    'encoder_provenance':cfg['encoder_provenance'],'split_ids':run_info['split_ids'],
                    'epoch':epoch+1,'threshold_logit':threshold,'dev_metrics':metrics,'test_only':run_info['test_only'],
                    'target':cfg['target'],'run_dir':str(out),'code_hashes':code_hashes()}
                if better:
                    atomic_save(out/'best.pt',state);write_json(out/'best_dev_metrics.json',metrics)
                    save_predictions(out/'best_dev_predictions.jsonl',dev_rows,logits,threshold)
                state.update(optimizer=opt.state_dict(),rng=rng_get(),history=history,best_auc=best,stale=stale)
                atomic_save(out/'last.pt',state)
                if a.patience and stale>=a.patience: print('Early stopping on DEV video any-fake ROC-AUC.');break
        except KeyboardInterrupt:
            print('Interrupted. Resume last.pt from the previous COMPLETED epoch; the incomplete epoch was not saved.')
            raise
    print(f'Selected head: {out}/best.pt. Encoders were never updated.')


def save_predictions(path,rows,logits,threshold):
    p=sigmoid(logits)
    write_jsonl(path,[{**r,'logit':float(z),'score':float(s),'prediction':int(z>=threshold)} for r,z,s in zip(rows,logits,p)])


def evaluate(a):
    device=device_of(a.device);torch.set_num_threads(a.threads)
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    cfg,rows,V,A,y=load_split(a.data,a.split,a.smoke_test);check_dataset(c,cfg)
    if c['test_only'] and not a.smoke_test:raise ValueError('Refusing synthetic/demo checkpoint as research result.')
    if c['code_hashes']!=code_hashes():raise ValueError('Fusion source changed since training; keep the run-specific version.')
    s=c['settings'];model=ConcatClassifier(c['dims']['visual'],c['dims']['audio'],s['mode'],s['hidden'],s['dropout']).to(device)
    model.load_state_dict(c['model'],strict=True)
    logits=predict(model,normalize(V,c['normalization']['visual']),normalize(A,c['normalization']['audio']),device,a.batch_size)
    result=evaluate_rows(rows,logits,c['threshold_logit'])
    result.update(split=a.split,mode=s['mode'],checkpoint_sha256=sha256(a.checkpoint),checkpoint_epoch=c['epoch'],
        threshold_source='checkpoint_DEV_only',evaluated_split_id=cfg['splits'][a.split]['signature'],
        train_dev_split_ids=c['split_ids'],encoder_ids=c['encoder_ids'],pooling=c['pooling'],test_only=c['test_only'])
    out=Path(a.out).expanduser().resolve();require_new(out)
    write_json(out/'metrics.json',result);save_predictions(out/'video_predictions.jsonl',rows,logits,c['threshold_logit'])
    print(json.dumps(result,indent=2));return result
