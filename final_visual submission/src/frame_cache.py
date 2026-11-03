"""Optional lossless, immutable RGB cache, before augmentation/normalization.

Raw RGB is shared by CNN/CLIP. Mask outputs have a separate exact policy key.
Existing entries are never evicted or overwritten. A byte cap stops new writes.
Copy this directory to persistent storage, then attach it read-only next session.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import uuid
import numpy as np
import torch
from PIL import Image
from .data import FrameReader, LazyDataset


class RGBCache:
    def __init__(self, root, identity, *, max_gib=8, readonly=False):
        self.root=Path(root);self.identity=json.loads(json.dumps(identity,sort_keys=True))
        self.limit=int(max_gib*2**30);self.readonly=readonly;self.hits=self.misses=self.writes=0
        self.require_hits=False;self.writes_skipped=0;self.write_disabled=False
        if self.limit<0:raise ValueError('Cache capacity must be nonnegative')
        marker=self.root/'cache_identity.json'
        if not readonly:self.root.mkdir(parents=True,exist_ok=True)
        if marker.exists():
            if json.loads(marker.read_text())!=self.identity:raise ValueError('Cache dataset identity mismatch; choose another directory')
        elif readonly:raise FileNotFoundError(marker)
        else:
            with marker.open('x',encoding='utf-8') as f:json.dump(self.identity,f,sort_keys=True)
        if not readonly:
            with self.connection() as db:
                db.execute('CREATE TABLE IF NOT EXISTS frames (key TEXT PRIMARY KEY, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, metadata TEXT NOT NULL)')
                db.execute('CREATE TABLE IF NOT EXISTS capacity (id INTEGER PRIMARY KEY CHECK(id=1), bytes INTEGER NOT NULL)')
                db.execute('INSERT OR IGNORE INTO capacity SELECT 1,COALESCE(SUM(bytes),0) FROM frames')

    def connection(self):
        # Per-operation connections are spawn-safe; no live SQLite handle is pickled.
        path=(self.root/'manifest.sqlite').resolve()
        return contextlib.closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=60) if self.readonly else sqlite3.connect(path,timeout=60,isolation_level=None))

    def key(self,row,index,stamp,data_root,policy=None):
        if row['split'] not in ('train','dev') or not row['source_file'].startswith(row['split']+'/'):
            raise PermissionError('RGB cache accepts TRAIN/DEV only')
        root=Path(data_root).resolve();source=(root/row['source_file']).resolve()
        if root not in source.parents:raise ValueError('Source escaped root')
        meta={'source_file':row['source_file'],'split':row['split'],'source_index':index,
              'timestamp_exact':stamp,'source_bytes':source.stat().st_size,'policy':policy or 'decoded_rgb24_224_v1'}
        key=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
        return key,meta|{'candidate_id':row['id'],'label':row['label'],'source_family':row.get('source_family')}

    def get(self,key,expected_meta=None):
        return self.get_many({key:expected_meta})[key]

    def get_many(self,requests):
        """One manifest connection per bounded batch; validate metadata and PNGs."""
        stored={}
        with self.connection() as db:
            keys=list(requests)
            for start in range(0,len(keys),256):
                part=keys[start:start+256]
                for key,sha,meta in db.execute('SELECT key,sha256,metadata FROM frames WHERE key IN ('+','.join('?'*len(part))+')',part):
                    stored[key]=(sha,json.loads(meta))
        result={}
        for key,expected in requests.items():
            if key not in stored:self.misses+=1;result[key]=None;continue
            sha,meta=stored[key]
            if expected is not None and any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Cached source metadata differs: '+key)
            raw=(self.root/key[:2]/(key+'.png')).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Corrupt cached PNG: '+key)
            with Image.open(io.BytesIO(raw)) as image:
                if image.mode!='RGB' or image.size!=(224,224):raise ValueError('Cached image must be original RGB224')
                value=np.array(image)
            if meta.get('decoded_rgb_sha256') and hashlib.sha256(value.tobytes()).hexdigest()!=meta['decoded_rgb_sha256']:
                raise ValueError('Cached pixels differ from decoded source hash: '+key)
            self.hits+=1;result[key]=value
        return result

    def put(self,key,meta,pixels):
        if self.readonly or self.limit==0 or self.write_disabled:return False
        if getattr(self,'disk_reserve_bytes',0) and shutil.disk_usage(self.root).free<self.disk_reserve_bytes+256*1024:
            self.writes_skipped+=1;return False
        if pixels.dtype!=np.uint8 or pixels.shape!=(224,224,3):raise ValueError('Cache requires original uint8 RGB224')
        buffer=io.BytesIO();Image.fromarray(pixels).save(buffer,format='PNG',compress_level=1)
        raw=buffer.getvalue();sha=hashlib.sha256(raw).hexdigest()
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                existing=db.execute('SELECT sha256 FROM frames WHERE key=?',(key,)).fetchone()
                if existing:
                    if existing[0]!=sha:raise ValueError('Different pixels for the same cache identity')
                    db.rollback();return True
                used=db.execute('SELECT bytes FROM capacity WHERE id=1').fetchone()[0]
                if used+len(raw)>self.limit:
                    db.rollback();self.write_disabled=True;self.writes_skipped+=1;return False
                path=self.root/key[:2]/(key+'.png');path.parent.mkdir(exist_ok=True)
                if path.exists():
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Unindexed conflicting cache file preserved')
                else:
                    # Transaction serializes writers; atomic rename makes interruption safe.
                    temp=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
                    temp.write_bytes(raw);os.replace(temp,path)
                meta=meta|{'decoded_rgb_sha256':hashlib.sha256(pixels.tobytes()).hexdigest()}
                db.execute('INSERT INTO frames VALUES (?,?,?,?)',(key,sha,len(raw),json.dumps(meta,sort_keys=True)))
                db.execute('UPDATE capacity SET bytes=bytes+? WHERE id=1',(len(raw),));db.commit();self.writes+=1
            except BaseException:db.rollback();raise
        return True

    def report(self):
        with self.connection() as db:n,b=db.execute('SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM frames').fetchone()
        return {'identity':self.identity,'entries':n,'png_bytes':b,'limit_bytes':self.limit,'readonly':self.readonly,
                'process_hits':self.hits,'process_misses':self.misses,'process_writes':self.writes,
                'process_writes_skipped':self.writes_skipped,
                'augmentation_cached':False,'cap_behavior':'stop new writes; preserve existing entries'}


class CachedReader:
    def __init__(self,data_root,cache,mask=None,mask_cache=None):
        self.root=data_root;self.cache=cache;self.reader=FrameReader(data_root,getattr(cache,'ram_cache_mib',0))
        self.mask=mask;self.mask_cache=mask_cache;self.mask_reader=None

    def read(self,records):
        requests={};references={};record_keys=[]
        for row in records:
            keys=[]
            for index,stamp in zip(row['indices'],row['timestamps_exact']):
                key,meta=self.cache.key(row,index,stamp,self.root)
                requests[key]=meta;references[key]=row|{'indices':[index],'timestamps_exact':[stamp]}
                keys.append(key)
            record_keys.append(keys)
        found=self.cache.get_many(requests)
        unique={key:(requests[key],references[key]) for key,value in found.items() if value is None}
        if unique:
            if self.cache.require_hits:raise FileNotFoundError('Prepared RGB cache has missing frames; rerun preparation/verification. Video decoding is disabled during training.')
            secondary=getattr(self.cache,'secondary_cache',None)
            if secondary is not None:
                reused=secondary.get_many({k:{name:value for name,value in requests[k].items() if name!='candidate_id'} for k in unique})
                for key,value in reused.items():
                    if value is not None:found[key]=value
                unique={key:value for key,value in unique.items() if found[key] is None}
        if unique:
            values=self.reader.read([row for meta,row in unique.values()])  # batch by source video in original decoder
            for (key,(meta,row)),images in zip(unique.items(),values):
                found[key]=images[0];self.cache.put(key,meta,images[0])
        result=[]
        for row,keys in zip(records,record_keys):
            images=[]
            for index,stamp,key in zip(row['indices'],row['timestamps_exact'],keys):
                value=found[key]
                if self.mask is not None:
                    mk,meta=self.cache.key(row,index,stamp,self.root,self.mask.identity)
                    masked=self.mask_cache.get(mk) if self.mask_cache else None
                    if masked is None:
                        if self.mask_reader is None:
                            from .face_mask import MaskReader
                            self.mask_reader=MaskReader(self.root,self.mask,0)
                        one=row|{'indices':[index],'timestamps_exact':[stamp]}
                        masked,_,details=self.mask_reader.transform(one,value)
                        if self.mask_cache:self.mask_cache.put(mk,meta|{'mask_details':details},masked)
                    value=masked
                images.append(value)
            result.append(images)
        return result


class CachedPipeline:
    """Original LazyDataset transforms and original mask geometry, cached upstream."""
    def __init__(self,cache,mask=None,mask_cache=None):
        self.cache=cache;self.mask=mask;self.mask_cache=mask_cache
        # Cache is an execution optimization, never a change in model input identity.
        self.identity=mask.identity if mask else None

    def make_loader(self,records,data_root,config,training=False,epoch=0):
        from .face_mask import FacePipeline
        return FacePipeline.make_loader(self,records,data_root,config,training,epoch)

    def dataset(self,records,data_root,config,training=False,epoch=0):
        dataset=LazyDataset(records,data_root,config.seed,epoch,config.flip_probability if training else 0.,0)
        dataset.reader=CachedReader(data_root,self.cache,self.mask,self.mask_cache)
        return dataset

    def snapshot(self,out):
        from .data import write_json
        write_json(Path(out)/'rgb_cache_report.json',self.cache.report())
        if self.mask:self.mask.snapshot(out)
        if self.mask_cache:
            from collections import Counter
            counts=Counter();total=area=0;fallback=clipping=0
            with self.mask_cache.connection() as db:
                for raw, in db.execute('SELECT metadata FROM frames'):
                    meta=json.loads(raw)
                    if meta['policy']!=self.mask.identity:continue
                    details=meta.get('mask_details',{});total+=1
                    counts[details.get('status','unavailable')]+=1
                    area+=details.get('retained_area_fraction',0)
                    fallback+=int(details.get('fallback') is not None)
                    clipping+=int(details.get('facial_regions_clipped') is True)
            write_json(Path(out)/'cached_mask_diagnostics.json',{'cached_frames':total,'status_counts':dict(counts),
                'fallback_count':fallback,'clipping_warning_count':clipping,'mean_retained_area_fraction':area/total if total else None,
                'scope':'all cached TRAIN/DEV masks under this exact policy; per-frame metadata in mask cache manifest'})


def benchmark_cache(records,data_root,cache,count=8):
    """Bounded cold/warm read timings; never samples/rebuilds or updates weights."""
    import time
    reader=CachedReader(data_root,cache);rows=records[:count];timings=[];hashes=[]
    for _ in range(2):
        start=time.perf_counter();decoded=reader.read(rows);timings.append(time.perf_counter()-start)
        hashes.append([hashlib.sha256(image[0].tobytes()).hexdigest() for image in decoded])
    if hashes[0]!=hashes[1]:raise ValueError('Cached and original pixels differ')
    return {'frames':len(rows),'first_read_seconds':timings[0],'second_read_seconds':timings[1],
            'pixel_hashes_match':True,'cache':cache.report(),'note':'First read may already be cached; no epoch runtime extrapolation from this small sample'}


def execute_cached_training(selected,inputs,project,*,frame_cache,mask_cache=None,**kwargs):
    """One CNN only: reuse original preflight/train/resume with a reader adapter.

    No global patching. Original workflow remains the default when cache is None.
    """
    from .config import configuration,RECIPES
    from .workflow import new_id,project_id,preflight,resume_candidate,mask_pipeline
    from .training import train
    from .data import write_json
    from .checkpoints import digest
    from .reporting import export_report
    from .prepared_frames import require_prepared_cache
    require_prepared_cache(frame_cache,inputs)
    if kwargs.get('confirmation')!='TRAIN_SELECTED_MODELS' or len(selected)!=1:raise PermissionError('One explicitly confirmed model required')
    key=selected[0]
    if key=='resnet50_lstm':raise ValueError('LSTM retains its original loader; frame cache is for the frame models')
    cfg=configuration(key,amp=kwargs.get('amp',True),workers=kwargs.get('workers'))
    resume=(kwargs.get('resume_from') or {}).get(key);source=None
    if resume:source,_=resume_candidate(key,inputs,project,source=None if str(resume)=='latest' else resume,amp=cfg.amp,workers=cfg.workers)
    preflight(key,inputs,project,amp=cfg.amp,workers=cfg.workers)
    runid=new_id();parent=Path(project)/'checkpoints'/key/runid;parent.mkdir(parents=True,exist_ok=False)
    r=RECIPES[key]
    write_json(parent/'experiment_identity.json',{'project_id':project_id(project),'model_id':key,'label':r.label,'input':r.input,'run_id':runid,'checkpoint_selection':cfg.checkpoint_selection,'new_run':True,'test_used':False,'resume_source':str(source) if source else None})
    write_json(parent/'source_provenance.json',{'standalone_source_hashes':{'src/'+p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},'reader_adapter':'lossless RGB cache; original model/data policy'})
    write_json(parent/'experiment_plans.json',{cfg.model:{'config':cfg.as_dict(),'input':r.input,'checkpoint_monitor':cfg.checkpoint_selection,'scheduler_monitor':cfg.checkpoint_selection}})
    mask=mask_pipeline(inputs,project,include_geometry_cache=kwargs.get('include_geometry_cache',False)) if r.input=='face_mask' else None
    pipeline=CachedPipeline(frame_cache,mask,mask_cache)
    out=train(cfg,inputs.index,inputs.data,inputs.weights,parent/'run',source,pool_path=inputs.pool,input_pipeline=pipeline)
    export_report(out,project)
    return {key:out}
