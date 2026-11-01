"""Read-only bridges to the user's existing, version-checked research projects.

One backend is loaded per subprocess: legacy projects use colliding module names
such as 'common' and 'experiment'. Never import two of them in one interpreter.
"""
from __future__ import annotations
from collections import defaultdict
from contextlib import nullcontext
import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from .io import digest, sha256, read_json, read_jsonl, under, relative_name, check_sequence


def import_project(project, name):
    project=Path(project).resolve()
    if not (project/(name+'.py')).is_file():
        raise FileNotFoundError(f'{project}/{name}.py is required; use the ORIGINAL training project.')
    sys.path.insert(0,str(project))
    result=importlib.import_module(name)
    if Path(result.__file__).resolve().parent != project:
        raise ValueError('A legacy module from a different project was imported; use separate extraction processes.')
    return result


def freeze(model):
    model.eval()
    model.requires_grad_(False)
    return model


def check_meta(info, row):
    if relative_name(info['file'])!=row['file'] or info['split']!=row['split']:
        raise ValueError('Encoder input and cohort ID/split disagree.')
    for key in ('modify_audio','modify_video'):
        if key in info and int(info[key])!=row[key]:
            raise ValueError(f'{row["file"]}: modality label differs between preprocessing and cohort.')
    aliases={'audio_label':'modify_audio','visual_label':'modify_video',
             'overall_video_label':'label','visual_video_label':'modify_video',
             'audio_video_label':'modify_audio'}
    for key,target in aliases.items():
        if key in info and int(info[key])!=row[target]:
            raise ValueError(f'{row["file"]}: {key} disagrees with cohort.')


class Audio2A:
    def __init__(self, slot, device):
        self.slot=slot; self.runner=r=import_project(slot['project_dir'],'experiment')
        required=('load_checkpoint','make_dataset','predict','loader')
        if any(not hasattr(r,n) for n in required):
            raise ValueError('Expected original Experiment 2A package. A custom Kaggle checkpoint needs its own adapter or preexported embeddings.')
        args=SimpleNamespace(root=Path(slot['input_root']),checkpoint=Path(slot['checkpoint']),
            device=device,aasist_dir=Path(slot['aasist_dir']) if slot.get('aasist_dir') else r.DEFAULT_AASIST_DIR)
        self.model,self.ckpt,records,self.config,self.device=r.load_checkpoint(args)
        if self.ckpt.get('target')!='audio_manipulation' or self.ckpt.get('model_name') not in ('resnet18','aasist'):
            raise ValueError('Not a supported fine-tuned Experiment 2A checkpoint.')
        freeze(self.model)
        self.records={relative_name(v['file']):v for v in records}
        if len(self.records)!=len(records): raise ValueError('Duplicate audio record.')
        self.batch=slot.get('batch_size',16); self.root=Path(slot['input_root'])
        self.provenance={'kind':'experiment_2a','model_name':self.ckpt['model_name'],
            'checkpoint_sha256':sha256(slot['checkpoint']),'preprocessing_id':self.config['config_id'],
            'representation':'forward_features()[0] before acoustic classifier',
            'source':self.model.provenance,'precision':'float32',
            'time_origin':'first_decoded_video_frame','training_split_signatures':self.ckpt['split_signatures']}
    def info(self,row):
        return self.records[row['file']]
    def input_signature(self,row):
        r=self.info(row)
        key='feature_path' if self.ckpt['model_name']=='resnet18' else 'wav_path'
        return digest({'row':r,'input_sha256':sha256(under(self.root,r[key]))})
    @torch.no_grad()
    def encode(self,row):
        r=self.info(row); check_meta(r,row)
        ds=self.runner.make_dataset(self.ckpt['model_name'],self.root,[r],row['split'],self.ckpt['stats'],self.config['audio'])
        _,features=self.runner.predict(self.model,self.runner.loader(ds,self.batch,0),self.device,embeddings=True)
        w=r['windows']
        return {'features':features,'start_s':np.array([v['start_s'] for v in w]),
            'end_s':np.array([v['end_s'] for v in w]),'center_s':np.array([v['center_s'] for v in w]),
            'padding_fraction':np.array([v['padding_fraction'] for v in w])}


class Forensics:
    def __init__(self,slot,device):
        self.slot=slot; self.runner=r=import_project(slot['project_dir'],'experiment')
        for name in ('load_run','FrameDataset','predict'):
            if not hasattr(r,name): raise ValueError('Expected the Forensics Adapter Experiment 1 project.')
        args=SimpleNamespace(manifest=Path(slot['input_manifest']), checkpoint=Path(slot['checkpoint']),
            crops_root=Path(slot['crops_root']) if slot.get('crops_root') else None,
            clip_weights=Path(slot['clip_weights']) if slot.get('clip_weights') else None,
            device=device,verify_files=slot.get('verify_files',True),amp=slot.get('amp',None))
        self.model,self.ckpt,rows,self.config,self.root,self.device,self.amp=r.load_run(args)
        if self.ckpt.get('smoke_test'): raise ValueError('Refusing a random smoke-test visual checkpoint.')
        freeze(self.model); self.records=defaultdict(list)
        for v in rows: self.records[relative_name(v['file'])].append(v)
        for values in self.records.values(): values.sort(key=lambda a:a['timestamp_s'])
        self.provenance={'kind':'forensics_adapter','checkpoint_sha256':sha256(slot['checkpoint']),
            'clip_weights_sha256':self.ckpt.get('clip_weights_sha256'),
            'representation':'adapted features before classification MLP (not plain CLIP or scores)',
            'model':self.ckpt['model'],'objective':self.ckpt['objective'],'config':self.ckpt['config'],
            'training_split_signatures':self.ckpt['split_ids'],
            'precision':self.amp,'attention_backend':slot.get('attention_backend','flash-math'),
            'time_origin':'first_decoded_video_frame'}
    def info(self,row): return self.records[row['file']][0]
    def input_signature(self,row): return digest(self.records[row['file']])
    def attention_context(self):
        if self.slot.get('attention_backend','flash-math')=='default': return nullcontext()
        from torch.nn.attention import sdpa_kernel,SDPBackend
        backends=[SDPBackend.MATH]
        if self.slot.get('attention_backend','flash-math')=='flash-math': backends.insert(0,SDPBackend.FLASH_ATTENTION)
        return sdpa_kernel(backends)
    @torch.no_grad()
    def encode(self,row):
        check_meta(self.info(row),row); frames=self.records[row['file']]
        ds=self.runner.FrameDataset(frames,self.root,row['split'],self.model.config.image_size,False)
        with self.attention_context():
            _,features=self.runner.predict(self.model,ds,self.device,self.slot.get('batch_size',4),0,self.amp,True)
        return {'features':features,'timestamp_s':np.array([r['timestamp_s'] for r in frames],np.float64)}


class PyramidCapture:
    """Observe task-trained temporal features before proposal classification.

    Highest-resolution FPN level only; excludes masked padding. Point coordinates
    are captured from the model, not inferred from array order or fake intervals.
    The scalar internal classifier is NOT used as an embedding.
    """
    def __init__(self,model):
        self.model=model; self.handles=[]; self.clear()
    def clear(self): self.features=None; self.mask=None; self.points=None; self.head_calls=0; self.point_calls=0
    def before_head(self,module,args):
        self.head_calls+=1
        if module.training or len(args)!=2: raise ValueError('Expected eval cls_head(fpn_features,fpn_masks).')
        levels,masks=args
        if not isinstance(levels,(list,tuple)) or not isinstance(masks,(list,tuple)) or not levels:
            raise ValueError('Unknown UMMAFormer FPN input interface.')
        self.features=levels[0].detach().float().cpu().clone()
        self.mask=masks[0].detach().cpu().clone()
    def after_points(self,module,args,out):
        self.point_calls+=1
        if not isinstance(out,(tuple,list)) or not out: raise ValueError('Unknown UMMAFormer point generator output.')
        self.points=out[0].detach().float().cpu().clone()
    def __enter__(self):
        for name in ('cls_head','point_generator'):
            if not isinstance(getattr(self.model,name,None),torch.nn.Module): raise ValueError(f'Missing native {name} module.')
        self.handles=[self.model.cls_head.register_forward_pre_hook(self.before_head),
                      self.model.point_generator.register_forward_hook(self.after_points)]
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
        self.handles=[]
    def arrays(self,item):
        h,m,p=self.features,self.mask,self.points
        if self.head_calls!=1 or self.point_calls!=1 or any(v is None for v in (h,m,p)):
            raise ValueError('Expected exactly one complete FPN/point capture per video.')
        if h.ndim!=3 or h.shape[0]!=1 or m.shape!=(1,1,h.shape[-1]) or p.ndim!=2 or len(p)!=h.shape[-1]:
            raise ValueError('Unsupported captured FPN/point/mask shape.')
        valid=m[0,0].bool().numpy()
        # The installed loader maps time to input positions as t*768/duration-.5.
        # The inverse also matches the native feat_stride/feat_num_frames convention.
        centers=(p[:,0].numpy()*float(item['feat_stride'])+0.5*float(item['feat_num_frames']))/float(item['fps'])
        duration=float(item['duration'])
        if ((centers[valid]<0)|(centers[valid]>duration)).any(): raise ValueError('Captured temporal feature coordinates exceed the decoded video.')
        return {'features':h[0].T.numpy()[valid], 'timestamp_s':centers.astype(np.float64)[valid]}


class UMMA:
    def __init__(self,slot,device):
        self.slot=slot; self.runner=r=import_project(slot['project_dir'],'run_experiment')
        self.ckpt=c=r.checkpoint_load(Path(slot['checkpoint']))
        self.manifest=r.load_manifest(Path(slot['input_manifest']))
        if self.manifest.get('test_only'): raise ValueError('Refusing a random-encoder feature fixture.')
        self.device=r.get_device(device,c['settings']['amp'])
        self.model,cfg,_,identity,_=r.build_official(Path(slot['upstream']),self.device,c['settings']['input_mode'])
        r.verify_checkpoint(c,self.manifest,identity)
        state=c['ema_state'] if c['settings']['ema'] else c['model_state']
        self.model.load_state_dict(state,strict=True); freeze(self.model)
        self.datasets={s:r.VisualDataset(self.manifest,s,input_mode=c['settings']['input_mode'])
            for s in ('train','dev','test') if any(v['split']==s for v in self.manifest['videos'])}
        self.records={}; self.location={}
        for s,ds in self.datasets.items():
            for i,v in enumerate(ds.rows):
                name=relative_name(v['file'])
                if name in self.records: raise ValueError('Duplicate UMMA video.')
                self.records[name]=v; self.location[name]=(s,i)
        self.provenance={'kind':'ummaformer','checkpoint_sha256':sha256(slot['checkpoint']),
            'representation':'highest-resolution FPN sequence at cls_head input; point-generator timestamps; no logits',
            'readout':'fpn_level0','input_mode':c['settings']['input_mode'],
            'local_sources':c['local_sources'],'upstream_source_digest':identity['source_digest'],
            'evaluation_weights':'ema' if c['settings']['ema'] else 'raw',
            'precision':c['settings']['amp'],'feature_kind':self.manifest.get('feature_kind'),
            'time_origin':'first_decoded_video_frame', 'training_split_signatures':{s:c['split_identity'][s] for s in ('train','dev')}}
    def info(self,row):return self.records[row['file']]
    def input_signature(self,row):
        r=self.info(row); p=under(self.manifest['_cache_root'],r['cache']['path'])
        actual=sha256(p)
        if r['cache'].get('sha256') and actual!=r['cache']['sha256']:
            raise ValueError(f'Source feature hash changed: {p}')
        return digest({'row':r,'cache_sha256':actual})
    @torch.no_grad()
    def encode(self,row):
        check_meta(self.info(row),row)
        split,i=self.location[row['file']]; original=self.datasets[split][i]
        # Whitelist inference fields; no fake intervals, modality flags or labels.
        allowed=('video_id','feats','fps','duration','feat_stride','feat_num_frames')
        item={k:original[k] for k in allowed}
        with PyramidCapture(self.model) as capture:
            with self.runner.amp_context(self.device,self.ckpt['settings']['amp']):
                result=self.model([item])
            if not isinstance(result,list) or len(result)!=1 or result[0]['video_id']!=row['file']:
                raise ValueError('Native inference video ID mismatch.')
            return capture.arrays(item)


class Preexported:
    def __init__(self,slot,device):
        self.slot=slot; self.root=Path(slot['index']).parent
        rows=read_jsonl(slot['index']); self.records={relative_name(v['file']):v for v in rows}
        if len(rows)!=len(self.records): raise ValueError('Duplicate preexported video.')
        h=sha256(slot['checkpoint'])
        if slot.get('representation')!='pre_classifier_embeddings':
            raise ValueError('Declare representation=pre_classifier_embeddings; scores/proposals are not embeddings.')
        if slot.get('preprocessing_verified') is not True:
            raise ValueError('Verify the original encoder preprocessing and timeline, then set preprocessing_verified=true. No resampling/normalization is guessed.')
        for r in rows:
            if r.get('checkpoint_sha256')!=h: raise ValueError('Every imported index row must identify the supplied checkpoint SHA256.')
            if r.get('test_only') or r.get('smoke_test'): raise ValueError('Refusing random test-only embeddings.')
        self.provenance={'kind':'preexported','checkpoint_sha256':h,
            'representation':slot['representation'],'description':slot.get('description',''),
            'time_origin':slot['time_origin'],'preprocessing_id':slot.get('preprocessing_id')}
        if (not self.provenance['description'] or not self.provenance['preprocessing_id']
                or 'CHANGE_ME' in str(self.provenance['description']) or 'CHANGE_ME' in str(self.provenance['preprocessing_id'])):
            raise ValueError('Imported embeddings need a readout description and preprocessing_id (a recorded config hash or version).')
    def info(self,row):return self.records[row['file']]
    def input_signature(self,row):
        r=self.info(row); p=under(self.root,r['feature_path'])
        h=sha256(p)
        if r.get('sha256') and r['sha256']!=h: raise ValueError('Imported embedding hash mismatch.')
        return digest({'file':row['file'],'sha256':h})
    def encode(self,row):
        r=self.info(row); check_meta(r,row)
        with np.load(under(self.root,r['feature_path']),allow_pickle=False) as z:
            return {k:z[k] for k in ('features','timestamp_s','start_s','end_s','center_s','padding_fraction') if k in z.files}


class Custom:
    """An explicit local adapter, NOT an automatic loader of unknown .pt files.

    build_encoder(slot, device) must return an engine with records/provenance,
    info(row), input_signature(row), encode_inputs(file, split). Only the last
    method runs inference and it receives NO modality labels or fake intervals.
    """
    def __init__(self,slot,device):
        path=Path(slot['adapter_file'])
        spec=importlib.util.spec_from_file_location('e2b_user_encoder',path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.engine=module.build_encoder(slot,device)
        self.records=self.engine.records
        self.provenance={**self.engine.provenance,'kind':'custom','checkpoint_sha256':sha256(slot['checkpoint']),
                         'adapter_sha256':sha256(path),'time_origin':slot['time_origin']}
        if not self.provenance.get('representation'): raise ValueError('Custom adapter must name the exact embedding readout.')
    def info(self,row):return self.engine.info(row)
    def input_signature(self,row):return self.engine.input_signature(row)
    def encode(self,row):return self.engine.encode_inputs(row['file'],row['split'])


def make_encoder(slot,modality,device):
    kind=slot['kind']
    if modality=='audio' and kind in ('forensics_adapter','ummaformer') or modality=='visual' and kind=='experiment_2a':
        raise ValueError(f'{kind} does not serve modality {modality}.')
    return {'experiment_2a':Audio2A,'forensics_adapter':Forensics,'ummaformer':UMMA,
            'preexported':Preexported,'custom':Custom}[kind](slot,device)
