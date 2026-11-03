"""Original ordered clips and fully frozen trained ResNet50 encoder with LSTM head."""

from .config import TemporalConfig, Config
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import json
import torch
from pathlib import Path
from torch import nn
from .data import (
    choose_indices,
    packet_timeline,
    rows_hash,
    write_json,
    training_rows,
    visual_label,
    DevSchedule,
    FixedPoolSampler,
)
from .checkpoints import digest, assert_finite_state
from .models import StagedCNN


TEMPORAL_PROTOCOL='resnet50_lstm_pts10_len16_stride12_one_fake_v1'


def clip_rows(video_id,indices,times,periods,modify_video):
    """Strict stride-12 TRAIN grid; no padding or extra off-stride tail clip."""
    result=[]
    labels=[visual_label(times[i],periods,modify_video) for i in indices]
    for p in range(0,len(indices)-16+1,12):
        ys=labels[p:p+16]
        result.append({'video_id':video_id,'position':p,'fake_count':sum(ys),'label':int(any(ys)),
            'onset':int(any(a<b for a,b in zip(ys,ys[1:]))),'offset':int(any(a>b for a,b in zip(ys,ys[1:])))})
    return result


def build_clip_manifest(data_root,dev_index,output):
    """TRAIN packet timelines only; preserve the complete EXISTING DEV clip schedule."""
    output=Path(output)
    if output.exists():raise FileExistsError('Reuse the immutable clip manifest; existing file preserved')
    rows=training_rows(data_root);videos=[];clips=[]
    dev=DevSchedule(dev_index,data_root)
    try:
        next_vid=max(dev.videos)+1
        for offset,row in enumerate(rows):
            vid=next_vid+offset;path=(Path(data_root)/row['file']).resolve()
            if Path(data_root).resolve() not in path.parents:raise ValueError('TRAIN path escaped root')
            times=packet_timeline(path,row['video_frames']);indices=choose_indices(times)
            periods=row['fake_periods'] if row['modify_video'] else []
            videos.append({'id':vid,'split':'train','source_file':row['file'],
                'source_family':row['original'] or row['file'],'video_label':int(row['modify_video']),
                'periods':periods,'source_bytes':path.stat().st_size,
                'sampled_indices':indices,'sampled_timestamps':list(map(str,(times[i] for i in indices)))})
            clips.extend(clip_rows(vid,indices,times,periods,row['modify_video']))
            if (offset+1)%100==0:print(f'Clip timelines: {offset+1}/{len(rows)} TRAIN videos',flush=True)
        for v in dev.videos.values():
            videos.append({k:v[k] for k in ('id','split','source_file','source_family','video_label','periods','source_bytes','sampled_indices')}|
                {'sampled_timestamps':[str(v['pts'][i]) for i in v['sampled_indices']]})
        clips.extend(dev.candidates('dev','clip'))
    finally:dev.close()
    for cid,c in enumerate(clips):c['id']=cid
    value={'protocol':TEMPORAL_PROTOCOL,'complete':True,'train_metadata_sha256':rows_hash(rows),
        'dev_index_sha256':digest(dev_index),'videos':videos,'clips':clips,
        'policy':{'fps':10,'length':16,'stride':12,'min_fake_frames':1,
            'dev':'immutable existing schedule, including legacy final tail clips; no DEV schedule changes'}}
    output.parent.mkdir(parents=True,exist_ok=True);partial=output.with_suffix('.partial')
    write_json(partial,value);ClipIndex(partial,data_root).close();partial.replace(output)
    return output


class ClipIndex:
    def __init__(self,path,data_root=None):
        self.path=Path(path);self.sha256=digest(path);c=json.loads(self.path.read_text(encoding='utf-8'))
        if c.get('protocol')!=TEMPORAL_PROTOCOL or c.get('complete') is not True:raise ValueError('Invalid temporal manifest')
        self.videos={v['id']:v for v in c['videos']};self.clips={r['id']:r for r in c['clips']}
        if len(self.clips)!=len(c['clips']) or len(self.videos)!=len(c['videos']):raise ValueError('Duplicate manifest IDs')
        seen=set()
        for r in self.clips.values():
            v=self.videos[r['video_id']];p=r['position'];split=v['split'];key=(v['source_file'],p)
            if split not in ('train','dev') or not all(v[k].startswith(split+'/') for k in ('source_file','source_family')):
                raise ValueError('Temporal split/family leakage')
            indices=v['sampled_indices'][p:p+16];times=v['sampled_timestamps'][p:p+16]
            periods=json.loads(v['periods']) if isinstance(v['periods'],str) else v['periods']
            n=sum(visual_label(Fraction(t),periods,bool(v['video_label'])) for t in times)
            if (key in seen or len(indices)!=16 or any(b<=a for a,b in zip(indices,indices[1:])) or
                any(Fraction(b)<=Fraction(a) for a,b in zip(times,times[1:])) or
                (split=='train' and p%12) or r['fake_count']!=n or r['label']!=int(n>0)):
                raise ValueError('Clip chronology, stride, uniqueness or one-fake-frame label mismatch')
            seen.add(key)
        if data_root is not None:
            if rows_hash(training_rows(data_root))!=c['train_metadata_sha256']:raise ValueError('TRAIN metadata changed')
            for v in self.videos.values():
                path=(Path(data_root)/v['source_file']).resolve()
                if Path(data_root).resolve() not in path.parents or path.stat().st_size!=v['source_bytes']:
                    raise ValueError('Clip source changed/escaped root')
        self.report={'protocol':TEMPORAL_PROTOCOL,'test_used':False,'policy':c['policy'],
            'counts':{s:dict(Counter(r['label'] for r in self.candidates(s,'clip'))) for s in ('train','dev')}}
        if 'dev_index_sha256' in c:
            if Counter(v['split'] for v in self.videos.values())!={'train':4000,'dev':500}:
                raise ValueError('Complete TRAIN/DEV clip videos required')
            if self.report['counts']['dev']!={0:3075,1:476}:
                raise ValueError('Full fixed 3,551-clip DEV schedule changed')

    def candidates(self,split,kind):
        if split not in ('train','dev') or kind!='clip':raise ValueError('TRAIN/DEV clips only')
        return [r for r in self.clips.values() if self.videos[r['video_id']]['split']==split]

    def records(self,split,kind,ids=None):
        rows=self.candidates(split,kind) if ids is None else [self.clips[i] for i in ids];result=[]
        for r in rows:
            v=self.videos[r['video_id']];p=r['position']
            if v['split']!=split:raise ValueError('Cross-split clip selection')
            result.append(r|{k:v[k] for k in ('split','source_file','source_family','video_label')}|
                {'kind':'clip','indices':v['sampled_indices'][p:p+16],
                 'timestamps_exact':v['sampled_timestamps'][p:p+16],'metric_eligible':True})
        return result

    def close(self):pass


class ClipPool:
    protocol=TEMPORAL_PROTOCOL
    kind='clip'
    def __init__(self,index):
        self.index=index;self.sha256=index.sha256
        self.frames={r['id']:r for r in index.candidates('train','clip')}
    def records(self,ids):return self.index.records('train','clip',ids)


class ResNetLSTM(nn.Module):
    def __init__(self,encoder,config):
        super().__init__();self.encoder=encoder;self.config=config;self.phase='frozen_encoder'
        self.lstm=nn.LSTM(2048,config.hidden_size,num_layers=config.lstm_layers,batch_first=True,
            dropout=config.dropout if config.lstm_layers>1 else 0.)
        self.dropout=nn.Dropout(config.dropout);self.fc=nn.Linear(config.hidden_size,1)
        for p in self.encoder.parameters():p.requires_grad_(False)
        self.encoder.eval()

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self

    def set_phase(self,finetune):self.phase='frozen_encoder'

    def trainable_report(self):
        return {'phase':self.phase,'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),
            'encoder_frozen':True,'feature_dim':2048,'feature_microbatch':self.config.feature_microbatch}

    def extract_features(self,clips):
        if clips.ndim!=5 or clips.shape[1:]!=(16,3,224,224):raise ValueError('Expected [B,16,3,224,224]')
        flat=clips.flatten(0,1);chunks=[]
        with torch.no_grad():
            for images in flat.split(self.config.feature_microbatch):
                features=self.encoder(images)
                if features.shape!=(len(images),2048):raise ValueError('Encoder must expose pooled 2048 features, before classifier')
                chunks.append(features.float())
        return torch.cat(chunks).reshape(len(clips),16,2048)

    def classify_features(self,features):
        if features.ndim!=3 or features.shape[1:]!=(16,2048):raise ValueError('Expected [B,16,2048]')
        with torch.autocast(features.device.type,enabled=False),torch.backends.cudnn.flags(enabled=False):
            output,_=self.lstm(features.float());return self.fc(self.dropout(output[:,-1]))

    def forward(self,clips):return self.classify_features(self.extract_features(clips))


def trained_resnet(config,checkpoint):
    """Strict selected trained CNN checkpoint. Never substitute ImageNet weights."""
    checkpoint=Path(checkpoint)
    if checkpoint.name!='best.pt' or not checkpoint.is_file():raise FileNotFoundError('Explicit trained ResNet50 best.pt required')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    from .data import PROTOCOL
    if (state.get('protocol')!=PROTOCOL or state.get('config',{}).get('model')!='resnet50' or
        state.get('epoch',-1)<0 or state.get('best_epoch')!=state['epoch']+1 or not state.get('history')):
        raise ValueError('Not a compatible selected trained ResNet50 checkpoint')
    assert_finite_state(state['model'],'CNN model')
    encoder=StagedCNN(Config(model='resnet50',dropout=state['config'].get('dropout',.3)),'.',pretrained=False)
    encoder.load_state_dict(state['model'],strict=True);encoder=encoder.network;encoder.fc=nn.Identity()
    identity={'cnn_checkpoint_sha256':digest(checkpoint),'cnn_best_epoch':state['best_epoch'],
        'cnn_checkpoint_selection':state['config'].get('checkpoint_selection','video_roc_auc'),
        'cnn_config':state['config'],'features':'pooled pre-classifier ResNet50 2048',
        'preprocessing':'existing native 224 RGB /255 ImageNet mean/std','frozen_encoder':True,
        'feature_microbatch':config.feature_microbatch,'feature_dim':2048,'test_used':False}
    return ResNetLSTM(encoder,config),identity


def preflight_temporal(config,clip_manifest,data_root,cnn_best,output):
    """Bounded loader + CUDA forward/backward, zero optimizer updates."""
    from .training import require_cuda,seed_everything,GradientGuard
    from .data import make_loader
    import gc,time
    config.validate();require_cuda();seed_everything(config.seed)
    index=ClipIndex(clip_manifest,data_root);pool=ClipPool(index)
    sampler=FixedPoolSampler(pool,config.clip_per_class,config.seed);ids,report=sampler.select(0)
    # Maximum two clips; smoke backward verifies recurrent parameters, not encoder gradients.
    records=index.records('train','clip',ids[:2]);start=time.perf_counter()
    images,targets,_=next(iter(make_loader(records,data_root,config,training=True)))
    load_seconds=time.perf_counter()-start
    model,identity=trained_resnet(config,cnn_best);model=model.cuda();model.train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config.head_lr)
    scaler=torch.amp.GradScaler('cuda',enabled=config.amp);guard=GradientGuard(config);guard.begin_epoch(0);events=[]
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.float16,enabled=config.amp):
        logits=model(images.cuda()).squeeze(1);loss=nn.functional.binary_cross_entropy_with_logits(logits,targets.cuda())
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Temporal smoke non-finite loss')
    scaler.scale(loss).backward();guard.finish_step(optimizer,scaler,epoch=0,batch=1,emit=events.append,execute_step=False)
    if any(p.grad is not None for p in model.encoder.parameters()):raise AssertionError('Frozen encoder received gradients')
    torch.cuda.synchronize()
    result={'clip_schedule':index.report,'sampling':report,'feature_source':identity,'clips':len(images),
        'load_seconds':load_seconds,'forward_backward_seconds':time.perf_counter()-start,
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'numerics':events[-1],'optimizer_steps':0}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);write_json(output,result)
    del model,optimizer,scaler,images,targets,logits,loss;gc.collect();torch.cuda.empty_cache()
    return result
