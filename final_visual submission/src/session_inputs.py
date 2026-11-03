"""Two explicit session-only preparation actions; no models or TEST execution."""
from collections import Counter
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import shutil
import sqlite3
import time
import torch
from .config import INDEX_SHA256,POOL_SHA256,configuration
from .checkpoints import digest
from .data import (FixedPool,FixedPoolSampler,DevSchedule,FrameReader,LazyDataset,
                   choose_indices,training_rows,rows_hash,visual_label,write_json,memory_mib)
from .frame_cache import RGBCache,CachedPipeline
from .prepared_frames import IDENTITY,PreparationStorageError,prepare_frame_dataset,_load_records
from .temporal import ClipIndex,ClipPool,clip_rows,TEMPORAL_PROTOCOL


def resource_report(cache):
    return {'process_rss_mib':memory_mib(),'disk_free_gib':shutil.disk_usage(cache.root).free/2**30,
            'cache_png_gib':cache.report()['png_bytes']/2**30,'cache_limit_gib':cache.limit/2**30,
            'decoder_ram_cache_mib_per_worker':getattr(cache,'ram_cache_mib',0),
            'disk_cache_writes_skipped':cache.report()['process_writes_skipped']}


def validate_streaming_frames(inputs,cache,*,ram_cache_mib=64,reserve_gib=8):
    """Explicit alternative to full preparation, with the SAME entire frame scope."""
    from .notebook import _TRAINING_LOCK
    if cache.identity!=IDENTITY:raise ValueError('Session cache identity differs')
    if not 0<=ram_cache_mib<=512 or reserve_gib<0:raise ValueError('Invalid bounded streaming resources')
    if not _TRAINING_LOCK.acquire(blocking=False):raise RuntimeError('Wait for the active model before preparing session inputs')
    cache.streaming_verified=False
    try:
        rows=_load_records(inputs)  # Validates both hashes, all labels/counts and source sizes.
        if cache.report()['png_bytes']>cache.limit:raise PreparationStorageError('Existing cache exceeds the selected streaming cap; preserve it and choose a new session directory or a higher cap')
        cache.session_mode='streaming';cache.require_hits=False
        cache.ram_cache_mib=ram_cache_mib;cache.disk_reserve_bytes=int(reserve_gib*2**30)
        # A bounded exact-PTS decode, not inference. No trained checkpoint is needed.
        from .frame_cache import CachedReader
        probes=[next(r for r in rows if r['split']==split and r['label']==y) for split in ('train','dev') for y in (0,1)]
        reader=CachedReader(inputs.data,cache);reader.read(probes)
        report={'mode':'streaming','explicitly_selected':True,'validated':True,'total_frames':len(rows),
                'index_sha256':digest(inputs.index),'pool_sha256':POOL_SHA256,'data_root':str(Path(inputs.data).resolve()),
                'seed':4266,'probe_frames':len(probes),'resources':resource_report(cache),
                'note':'Full fixed scope; every miss decodes through the original exact-PTS reader. No corrupt cache entry is skipped.'}
        cache.streaming_report=report;cache.streaming_verified=True
        print('Validated session streaming: '+json.dumps(report),flush=True)
        return report
    except BaseException:
        cache.streaming_verified=False
        raise
    finally:_TRAINING_LOCK.release()


def require_streaming_frames(cache,inputs,*,announce=True):
    report=getattr(cache,'streaming_report',{})
    if (not getattr(cache,'streaming_verified',False) or report.get('mode')!='streaming' or
        report.get('explicitly_selected') is not True or report.get('total_frames')!=61746 or
        report.get('index_sha256')!=digest(inputs.index) or report.get('pool_sha256')!=digest(inputs.pool) or
        report.get('data_root')!=str(Path(inputs.data).resolve())):
        raise RuntimeError('Run Cell A in explicitly selected streaming mode before training')
    cache.require_hits=False
    if announce:print('Session STREAMING ACTIVE (explicit, validated) | full20,000 TRAIN +41,746 DEV | '+json.dumps(resource_report(cache)),flush=True)
    return report


def prepare_frames(inputs,project,*,mode='prepared',cache_root=None,readonly=False,
                   prepared_gib=12,streaming_gib=2,ram_cache_mib=64,reserve_gib=8):
    """Cell A. Session storage only; storage errors never auto-select streaming."""
    if mode not in ('prepared','streaming'):raise ValueError('FRAME_LOADING_MODE must be prepared or streaming')
    root=Path(cache_root) if cache_root else Path(project)/'.cache/shared_rgb'
    cache=RGBCache(root,IDENTITY,max_gib=prepared_gib if mode=='prepared' else streaming_gib,readonly=readonly)
    if mode=='streaming':report=validate_streaming_frames(inputs,cache,ram_cache_mib=ram_cache_mib,reserve_gib=reserve_gib)
    else:
        cache.session_mode='prepared';cache.ram_cache_mib=0
        try:
            report=prepare_frame_dataset(inputs,cache,report_dir=Path(project)/'results/frame_preparation'/str(time.time_ns()),reserve_gib=reserve_gib)
        except PreparationStorageError as error:
            raise PreparationStorageError(str(error)+' To use bounded on-demand loading explicitly, set FRAME_LOADING_MODE="streaming" and rerun Cell A. No validation or frame selection will be weakened.') from error
    print('Cell A resources: '+json.dumps(resource_report(cache)),flush=True)
    return cache,report


def _clip_value_from_index(inputs):
    """Original manifest schema/order/IDs, using already verified packet timelines.

    Uses original clip_rows and immutable DEV candidates; no video rescans. Raw
    TRAIN annotation periods are retained exactly as in build_clip_manifest.
    """
    if digest(inputs.index)!=INDEX_SHA256:
        from .build_candidates import validate_candidate_index
        validate_candidate_index(inputs.index, inputs.data)
    rows=training_rows(inputs.data);dev=DevSchedule(inputs.index,inputs.data);videos=[];clips=[]
    db=sqlite3.connect(Path(inputs.index).resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    try:
        stored={r['source_file']:dict(r) for r in db.execute("SELECT source_file,pts,sampled_indices,source_bytes FROM videos WHERE split='train'")}
        first=max(dev.videos)+1
        for offset,row in enumerate(rows):
            original=stored[row['file']];times=list(map(Fraction,json.loads(original['pts'])));indices=json.loads(original['sampled_indices'])
            if indices!=choose_indices(times):raise ValueError('Stored10FPS timeline differs from original selection')
            vid=first+offset;periods=row['fake_periods'] if row['modify_video'] else []
            videos.append({'id':vid,'split':'train','source_file':row['file'],'source_family':row['original'] or row['file'],
                'video_label':int(row['modify_video']),'periods':periods,'source_bytes':original['source_bytes'],
                'sampled_indices':indices,'sampled_timestamps':[str(times[i]) for i in indices]})
            clips.extend(clip_rows(vid,indices,times,periods,row['modify_video']))
            if (offset+1)%500==0:print(f'Clip references from existing index: {offset+1}/{len(rows)} TRAIN videos',flush=True)
        for v in dev.videos.values():
            videos.append({k:v[k] for k in ('id','split','source_file','source_family','video_label','periods','source_bytes','sampled_indices')}|
                          {'sampled_timestamps':[str(v['pts'][i]) for i in v['sampled_indices']]})
        clips.extend(dev.candidates('dev','clip'))
    finally:db.close();dev.close()
    for cid,c in enumerate(clips):c['id']=cid
    return {'protocol':TEMPORAL_PROTOCOL,'complete':True,'train_metadata_sha256':rows_hash(rows),
           'dev_index_sha256':INDEX_SHA256,'videos':videos,'clips':clips,
           'policy':{'fps':10,'length':16,'stride':12,'min_fake_frames':1,
                     'dev':'immutable existing schedule, including legacy final tail clips; no DEV schedule changes'}}


class SessionClipReader:
    """Ordered16-frame reads; share base pixels, cache only genuinely absent pixels."""
    def __init__(self,data_root,pipeline):
        self.root=data_root;self.base=pipeline.base;self.extra=pipeline.extra;self.videos=pipeline.videos
        self.reader=FrameReader(data_root,pipeline.ram_cache_mib)
        self.base_hits=self.extra_hits=self.decoded_missing=0

    def read(self,records):
        requests={};keys_by_clip=[];single={}
        for row in records:
            if row['split'] not in ('train','dev') or row['kind']!='clip' or len(row['indices'])!=16:raise PermissionError('Validated TRAIN/DEV16-frame clips only')
            v=self.videos[row['source_file']];periods=json.loads(v['periods']) if isinstance(v['periods'],str) else v['periods']
            keys=[]
            for index,stamp in zip(row['indices'],row['timestamps_exact']):
                y=visual_label(Fraction(stamp),periods,bool(v['video_label']))
                one=row|{'id':index,'kind':'frame','indices':[index],'timestamps_exact':[stamp],'label':y}
                key,meta=self.base.key(one,index,stamp,self.root)
                # Clip IDs/labels describe sequences, not the physical frame. Its
                # true visual frame label is checked separately above.
                requests[key]={k:value for k,value in meta.items() if k!='candidate_id'}
                single[key]=(one,meta);keys.append(key)
            keys_by_clip.append(keys)
        found=self.base.get_many(requests);self.base_hits+=sum(v is not None for v in found.values())
        missing={k:requests[k] for k,v in found.items() if v is None}
        extra=self.extra.get_many(missing) if missing else {}
        for key,value in extra.items():
            if value is not None:found[key]=value;self.extra_hits+=1
        missing=[k for k,v in found.items() if v is None]
        if missing:
            pixels=self.reader.read([single[k][0] for k in missing])
            for key,images in zip(missing,pixels):
                found[key]=images[0];self.extra.put(key,single[key][1],images[0]);self.decoded_missing+=1
        return [[found[key] for key in keys] for keys in keys_by_clip]


class SessionClipPipeline:
    identity=None  # Original full-frame clip preprocessing/checkpoint compatibility.
    def __init__(self,base,extra,index,ram_cache_mib=64):
        self.base=base;self.extra=extra;self.manifest=index.path;self.manifest_sha256=index.sha256
        self.videos={v['source_file']:v for v in index.videos.values()};self.ram_cache_mib=ram_cache_mib

    def dataset(self,records,data_root,config,training=False,epoch=0):
        dataset=LazyDataset(records,data_root,config.seed,epoch,config.flip_probability if training else 0.,0)
        dataset.reader=SessionClipReader(data_root,self)
        return dataset

    def make_loader(self,records,data_root,config,training=False,epoch=0):
        from .face_mask import FacePipeline
        return FacePipeline.make_loader(self,records,data_root,config,training,epoch)

    def snapshot(self,out):
        write_json(Path(out)/'clip_loading_report.json',{'manifest_sha256':self.manifest_sha256,'pixel_storage':'shared RGB + bounded missing-frame overlay',
            'base':resource_report(self.base),'extra':resource_report(self.extra),'preprocessing_unchanged':True})


def prepare_clips(inputs,project,base_cache,*,clip_cache_gib=1,ram_cache_mib=64,reserve_gib=8):
    """Cell B: exact reference manifest + bounded sequence probe, never all clip images."""
    from .notebook import _TRAINING_LOCK
    from .prepared_frames import require_prepared_cache
    require_prepared_cache(base_cache,inputs,announce=False)
    if not 0<=ram_cache_mib<=512 or clip_cache_gib<0:raise ValueError('Invalid clip cache resource limits')
    if not _TRAINING_LOCK.acquire(blocking=False):raise RuntimeError('Wait for the active model before preparing clips')
    try:
        path=Path(inputs.clips) if inputs.clips else Path(project)/'.cache/manifests/clip_manifest.json'
        expected=_clip_value_from_index(inputs)
        if not path.is_file():
            path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_suffix('.partial')
            write_json(partial,expected);ClipIndex(partial,inputs.data).close();partial.replace(path)
        elif json.loads(path.read_text(encoding='utf-8'))!=expected:
            raise ValueError('Existing clip manifest differs from the exact original TRAIN grid/DEV schedule; preserved, not replaced')
        if json.loads(path.read_text(encoding='utf-8'))['dev_index_sha256'] not in (INDEX_SHA256,digest(inputs.index)):raise ValueError('Wrong immutable DEV clip schedule')
        index=ClipIndex(path,inputs.data);cfg=configuration('resnet50_lstm')
        pool=ClipPool(index);ids,sampling=FixedPoolSampler(pool,cfg.clip_per_class,4266).select(0)
        extra=RGBCache(Path(project)/'.cache/clip_rgb',IDENTITY,max_gib=clip_cache_gib)
        if extra.report()['png_bytes']>extra.limit:raise PreparationStorageError('Existing clip cache exceeds selected cap; choose a sufficient cap, preserving existing files')
        extra.disk_reserve_bytes=int(reserve_gib*2**30);extra.ram_cache_mib=ram_cache_mib
        if getattr(base_cache,'session_mode',None)=='streaming':base_cache.secondary_cache=extra
        pipeline=SessionClipPipeline(base_cache,extra,index,ram_cache_mib)
        probes={}
        for split,rows in [('train',index.records('train','clip',ids[:2])),('dev',index.records('dev','clip')[:2])]:
            dataset=pipeline.dataset(rows,inputs.data,cfg,training=split=='train')
            values=dataset.__getitems__(list(range(len(rows))))
            if any(tuple(t.shape)!=(16,3,224,224) or not bool(torch.isfinite(t).all()) for t,_,_ in values):raise ValueError('Invalid prepared clip sequence')
            probes[split]={'clips':len(rows),'shape':[len(rows),16,3,224,224],'shared_frame_hits':dataset.reader.base_hits,
                'overlay_hits':dataset.reader.extra_hits,'newly_decoded_unique_frames':dataset.reader.decoded_missing,
                'decoder_ram_bytes_used':dataset.reader.reader.bytes}
        report={'mode':'ordered references + bounded on-demand missing pixels','manifest':str(path),'manifest_sha256':index.sha256,
                'protocol':index.report,'probe':probes,'epoch_real':cfg.clip_per_class,'epoch_fake':cfg.clip_per_class,
                'seed':4266,'resources':resource_report(extra),'train_sampling':sampling,'trained_models':0}
        out=Path(project)/'results/session_preparation'/str(time.time_ns());out.mkdir(parents=True)
        write_json(out/'clips.json',report)
        print('Cell B clip preparation: '+json.dumps({k:v for k,v in report.items() if k!='train_sampling'}),flush=True)
        return replace(inputs,clips=path),pipeline,report
    finally:_TRAINING_LOCK.release()


def execute_session_lstm(selected,inputs,project,*,clip_pipeline,**kwargs):
    """Original temporal training function with a pixel-reader adapter only."""
    from .workflow import new_id,project_id,preflight,resume_candidate,new_encoder
    from .training import train
    from .reporting import export_report
    if selected!=['resnet50_lstm'] or kwargs.get('confirmation')!='TRAIN_SELECTED_MODELS':raise PermissionError('One confirmed LSTM required')
    if clip_pipeline is None or digest(inputs.clips)!=clip_pipeline.manifest_sha256:raise ValueError('Run Cell B for this clip manifest')
    key=selected[0];cfg=configuration(key,amp=kwargs.get('amp',True),workers=kwargs.get('workers'))
    source=(kwargs.get('resume_from') or {}).get(key)
    if source:source,_=resume_candidate(key,inputs,project,source=None if str(source)=='latest' else source,amp=cfg.amp,workers=cfg.workers)
    encoder=new_encoder(kwargs.get('cnn_best'),project);preflight(key,inputs,project,cnn_best=encoder,amp=cfg.amp,workers=cfg.workers)
    runid=new_id();parent=Path(project)/'checkpoints'/key/runid;parent.mkdir(parents=True,exist_ok=False)
    write_json(parent/'experiment_identity.json',{'project_id':project_id(project),'model_id':key,'label':'ResNet50 LSTM','input':'full_frame_clips',
        'run_id':runid,'checkpoint_selection':cfg.checkpoint_selection,'new_run':True,'test_used':False,'resume_source':str(source) if source else None})
    write_json(parent/'source_provenance.json',{'source_hashes':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},'clip_manifest_sha256':clip_pipeline.manifest_sha256})
    write_json(parent/'experiment_plans.json',{cfg.model:{'config':cfg.as_dict(),'input':'full_frame_clips','checkpoint_monitor':cfg.checkpoint_selection,'scheduler_monitor':cfg.checkpoint_selection}})
    out=train(cfg,inputs.clips,inputs.data,encoder,parent/'run',source,input_pipeline=clip_pipeline)
    export_report(out,project);return {key:out}
