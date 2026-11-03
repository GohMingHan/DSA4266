"""Prepare/verify the exact fixed TRAIN+DEV RGB dataset; no models or TEST.

One video decoder at a time, one decoded image at a time, at most64 cached
images during verification. Metadata stays in memory; image datasets do not.
"""
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import uuid
import zipfile
import numpy as np
from .config import INDEX_SHA256,INDEX_LOGICAL_SHA256,POOL_SHA256
from .checkpoints import digest
from .data import FixedPool,FixedPoolSampler,DevSchedule,FrameReader,write_json,memory_mib

READY='preparation_ready.json'
PNG_BOUND=160*1024  # RGB224 PNG has no auxiliary metadata; conservative incompressible bound.
IDENTITY={'schema':'lossless_indexed_rgb224_v1','index_logical_sha256':INDEX_LOGICAL_SHA256,'pool_sha256':POOL_SHA256}


class PreparationStorageError(OSError):pass


def _atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:write_json(temp,value);temp.replace(path)
    finally:temp.unlink(missing_ok=True)


def _load_records(inputs):
    if digest(inputs.pool)!=POOL_SHA256:
        raise ValueError('Prepare requires the exact original train_pool.json; it will not be rebuilt')
    if digest(inputs.index)!=INDEX_SHA256:
        from .build_candidates import validate_candidate_index
        validate_candidate_index(inputs.index, inputs.data)
    pool=FixedPool(inputs.pool,inputs.data);index=DevSchedule(inputs.index,inputs.data)
    try:records=pool.records()+index.records('dev','frame')
    finally:index.close()
    counts=Counter((r['split'],r['label']) for r in records)
    if counts!={('train',0):10000,('train',1):10000,('dev',0):40115,('dev',1):1631}:
        raise ValueError('The full fixed TRAIN/DEV frame counts differ')
    return records


def stream_video_frames(data_root,records,stats):
    """Same enumeration/origin/PTS/RGB checks as FrameReader, streaming one source."""
    import av
    if not records:return
    source=records[0]['source_file'];root=Path(data_root).resolve();path=(root/source).resolve()
    if root not in path.parents:raise ValueError('Source escaped data root')
    needed={}
    for row in records:
        if row['split'] not in ('train','dev') or not source.startswith(row['split']+'/') or row['source_file']!=source or row['kind']!='frame' or len(row['indices'])!=1:
            raise PermissionError('Only existing TRAIN/DEV frame records from one source may be prepared')
        index=row['indices'][0]
        if index in needed:raise ValueError('Duplicate source frame in preparation scope')
        needed[index]=(Fraction(row['timestamps_exact'][0]),row)
    stats['source_video_opens']+=1
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        origin=Fraction(container.start_time,av.time_base) if container.start_time is not None else Fraction(0)
        for i,frame in enumerate(container.decode(stream)):
            stats['native_frames_decoded']+=1
            if i in needed:
                stamp,row=needed.pop(i)
                if frame.is_corrupt or frame.pts is None or Fraction(frame.pts*frame.time_base)-origin!=stamp:
                    raise ValueError(f'Decoded PTS mismatch: {source}/{i}')
                if (frame.width,frame.height)!=(224,224):raise ValueError('Expected native224 RGB; resizing is prohibited')
                yield row,frame.to_ndarray(format='rgb24')
            if not needed:break
    if needed:raise ValueError(f'Missing requested source frames: {source}/{sorted(needed)}')


def prepare_frame_dataset(inputs,cache,*,report_dir=None,reserve_gib=8,progress_every_videos=25):
    """Full explicit preparation, or verification of an attached read-only dataset.

    Existing entries are verified, not decoded again. Missing entries stream from
    their source video once. Corruption/metadata conflicts fail and are preserved.
    A fresh session must run this verification before using the cache to train.
    """
    from .notebook import _TRAINING_LOCK
    if cache is None:raise ValueError('Initialize the shared RGB cache before preparation')
    if cache.identity!=IDENTITY:raise ValueError('Prepared RGB dataset identity differs')
    if reserve_gib<0 or progress_every_videos<1:raise ValueError('Invalid storage/progress settings')
    if not _TRAINING_LOCK.acquire(blocking=False):raise RuntimeError('Wait for the active model before preparing frames')
    cache.prepared_verified=False;cache.require_hits=False
    started=time.perf_counter();last_print=started
    report_dir=Path(report_dir) if report_dir else (cache.root if not cache.readonly else None)
    stats={'status':'preparing','ready':False,'completed_videos':0,'train_frames':0,'dev_frames':0,
           'new_frames':0,'reused_verified_frames':0,'source_video_opens':0,'native_frames_decoded':0,'failures':[]}
    def progress(force=False):
        nonlocal last_print
        now=time.perf_counter()
        if not force and now-last_print<10:return
        stats['elapsed_seconds']=now-started;stats['cache_png_bytes']=cache.report()['png_bytes']
        stats['process_rss_mib']=memory_mib();stats['disk_free_gib']=shutil.disk_usage(cache.root).free/2**30
        rate=stats['new_frames']/max(stats['elapsed_seconds'],1e-9)
        print(f"Prepare RGB | videos {stats['completed_videos']}/{stats.get('total_videos','?')} | "
              f"TRAIN {stats['train_frames']:,}/20,000 DEV {stats['dev_frames']:,}/41,746 | "
              f"new {stats['new_frames']:,}, reused {stats['reused_verified_frames']:,} | "
              f"decode {rate:.1f} selected frames/s | cache {stats['cache_png_bytes']/2**30:.2f} GiB | "
              f"elapsed {stats['elapsed_seconds']:.1f}s | RSS {stats['process_rss_mib']} MiB | free {stats['disk_free_gib']:.2f} GiB | failures {len(stats['failures'])}",flush=True)
        if report_dir:_atomic_json(report_dir/'preparation_progress.json',stats)
        last_print=now
    try:
        records=_load_records(inputs);start_index_hash=digest(inputs.index);groups=defaultdict(list);requests={};scope=hashlib.sha256()
        for row in records:
            if row['split'] not in ('train','dev') or row['kind']!='frame':raise PermissionError('TRAIN/DEV frames only')
            key,meta=cache.key(row,row['indices'][0],row['timestamps_exact'][0],inputs.data)
            if key in requests:raise ValueError('Duplicate frame in fixed preparation scope')
            requests[key]=(row,meta);groups[row['source_file']].append(key)
            scope.update((json.dumps([key,meta],sort_keys=True)+'\n').encode())
        stats.update(total_videos=len(groups),target_frames=len(records))
        with cache.connection() as db:existing={r[0] for r in db.execute('SELECT key FROM frames')}
        if existing-set(requests):raise ValueError('Cache has entries outside the fixed frame scope; preserved. Choose a separate FRAME_CACHE_ROOT.')
        missing=len(set(requests)-existing);used=cache.report()['png_bytes']
        if missing and cache.readonly:raise PreparationStorageError(f'Read-only cache is incomplete: {missing:,} missing frames; use a writable cache directory')
        if missing and used+missing*PNG_BOUND>cache.limit:
            raise PreparationStorageError(f'Cache capacity insufficient for a guaranteed full preparation: {missing:,} frames remain; need up to {(used+missing*PNG_BOUND)/2**30:.2f} GiB PNG capacity, configured {cache.limit/2**30:.2f} GiB. Use12 GiB.')
        required=missing*(PNG_BOUND+4096)+64*2**20+int(reserve_gib*2**30)
        if missing and shutil.disk_usage(cache.root).free<required:
            raise PreparationStorageError(f'Insufficient disk: {missing:,} frames remain; need up to {required/2**30:.2f} GiB free including {reserve_gib:g} GiB training/recovery reserve. Existing frames are preserved.')
        progress(True)
        for source,keys in sorted(groups.items()):
            absent=[]
            for start in range(0,len(keys),64):
                part=keys[start:start+64];found=cache.get_many({key:requests[key][1] for key in part})
                for key,value in found.items():
                    if value is None:absent.append(requests[key][0])
                    else:
                        stats[requests[key][0]['split']+'_frames']+=1;stats['reused_verified_frames']+=1
                del found
            for row,pixels in stream_video_frames(inputs.data,absent,stats):
                if shutil.disk_usage(cache.root).free<int(reserve_gib*2**30)+PNG_BOUND+64*2**20:
                    raise PreparationStorageError('Disk reserve reached mid-extraction; cache remains incomplete and verified frames can be resumed')
                key,meta=cache.key(row,row['indices'][0],row['timestamps_exact'][0],inputs.data)
                if not cache.put(key,meta,pixels):raise PreparationStorageError('Cache limit reached mid-extraction; no ready marker will be issued')
                saved=cache.get(key,meta)
                if saved is None or not np.array_equal(saved,pixels):raise ValueError('Lossless PNG round-trip differs from the decoded source')
                stats[row['split']+'_frames']+=1;stats['new_frames']+=1
                progress()
            stats['completed_videos']+=1
            if stats['completed_videos']==1 or stats['completed_videos']%progress_every_videos==0:progress(True)
        if stats['train_frames']+stats['dev_frames']!=len(records):raise RuntimeError('Incomplete frame verification; refusing readiness')
        if digest(inputs.index)!=start_index_hash or digest(inputs.pool)!=POOL_SHA256:raise ValueError('Source metadata changed during preparation')
        report={'schema':'prepared_train_dev_rgb_v1','ready':True,'identity':cache.identity,'index_sha256':digest(inputs.index),
                'pool_sha256':POOL_SHA256,'scope_sha256':scope.hexdigest(),'total_frames':len(records),
                'train_frames':stats['train_frames'],'dev_frames':stats['dev_frames'],
                'class_counts':{split:{str(y):sum(r['split']==split and r['label']==y for r in records) for y in (0,1)} for split in ('train','dev')},
                'manifest_sha256':digest(cache.root/'manifest.sqlite'),'png_bytes':cache.report()['png_bytes'],
                'normalization_cached':False,'augmentation_cached':False,'masking_cached':False,'test_used':False}
        if cache.readonly:
            previous=json.loads((cache.root/READY).read_text())
            for key in ['schema','ready','identity','index_sha256','pool_sha256','scope_sha256','total_frames','train_frames','dev_frames','manifest_sha256']:
                if report[key]!=previous[key]:raise ValueError('Attached prepared-dataset verification differs: '+key)
        else:_atomic_json(cache.root/READY,report)
        stats.update(status='ready',ready=True);progress(True)
        if report_dir:_atomic_json(report_dir/'verification_report.json',report)
        cache.require_hits=True;cache.prepared_verified=True;cache.prepared_report=report
        return report
    except BaseException as error:
        stats.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',ready=False)
        stats['failures'].append({'type':type(error).__name__,'message':str(error)})
        try:
            progress(True)
            if report_dir:(report_dir/'failure_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        except Exception:pass  # Preserve the original failure, never declare readiness.
        raise
    finally:_TRAINING_LOCK.release()


def require_prepared_cache(cache,inputs,*,announce=True):
    # Backward-compatible entry point: streaming must be EXPLICITLY prepared and
    # validated by Cell A; an incomplete prepared cache never selects this path.
    if cache is not None and getattr(cache,'session_mode',None)=='streaming':
        from .session_inputs import require_streaming_frames
        return require_streaming_frames(cache,inputs,announce=announce)
    if cache is None or not getattr(cache,'prepared_verified',False):
        raise RuntimeError('Run "Prepare TRAIN/DEV Frame Dataset" before frame-model training; skipped Section5b/preparation cannot fall back to video decoding.')
    report=json.loads((cache.root/READY).read_text())
    if (not report.get('ready') or report.get('identity')!=IDENTITY or
        (report.get('total_frames'),report.get('train_frames'),report.get('dev_frames'))!=(61746,20000,41746) or
        report.get('index_sha256')!=digest(inputs.index) or report.get('pool_sha256')!=digest(inputs.pool) or
        report.get('manifest_sha256')!=digest(cache.root/'manifest.sqlite')):
        raise ValueError('Prepared cache readiness/identity changed; rerun full preparation verification')
    cache.require_hits=True
    if announce:print('Prepared RGB dataset ACTIVE |20,000 TRAIN +41,746 DEV | video fallback DISABLED | '+str(cache.report()),flush=True)
    return report


def benchmark_prepared_frames(inputs,cache,*,count_per_split=8,processor=None):
    """Same indexed frames, uncached video reads vs strict cache + real CLIP loader."""
    from .frame_cache import CachedReader
    from .clip_model import CLIPConfig,make_clip_loader
    from transformers import CLIPImageProcessor
    require_prepared_cache(cache,inputs,announce=False)
    pool=FixedPool(inputs.pool);index=DevSchedule(inputs.index)
    try:
        ids,_=FixedPoolSampler(pool,4000,4266).select(0)  # Disposable: never consumes the training sampler.
        sets={'train':pool.records(ids[:count_per_split]),'dev':index.records('dev','frame')[:count_per_split]}
    finally:index.close()
    result={'test_used':False,'splits':{}}
    for split,rows in sets.items():
        baseline=FrameReader(inputs.data,0);start=time.perf_counter();original=baseline.read(rows);uncached=time.perf_counter()-start
        reader=CachedReader(inputs.data,cache);start=time.perf_counter();cached=reader.read(rows);seconds=time.perf_counter()-start
        if any(not np.array_equal(a[0],b[0]) for a,b in zip(original,cached)):raise ValueError('Benchmark source/cache pixels differ')
        loader=make_clip_loader(rows,inputs.data,processor or CLIPImageProcessor(),CLIPConfig(),cache=cache)
        hits=cache.hits;misses=cache.misses;start=time.perf_counter();images,targets,indices=next(iter(loader));loader_seconds=time.perf_counter()-start
        strict=cache.require_hits
        if strict and (loader.dataset.reader.reader.opens!=0 or loader.dataset.reader.reader.decoded!=0 or cache.misses!=misses or cache.hits-hits!=len(rows)):
            raise RuntimeError('CLIP loader did not obtain every frame from the prepared cache')
        result['splits'][split]={'frames':len(rows),'on_demand_seconds':uncached,'cached_rgb_seconds':seconds,'clip_batch_loading_seconds':loader_seconds,
            'pixel_hashes_match':True,'clip_cache_hits':cache.hits-hits,'clip_cache_misses':cache.misses-misses,
            'clip_video_opens':loader.dataset.reader.reader.opens,'clip_tensor_shape':list(images.shape),
            'loading_mode':getattr(cache,'session_mode','prepared')}
    print('Identical-frame decoding/cache/CLIP benchmark: '+json.dumps(result),flush=True)
    return result


def verify_clip_cache_loader(loader):
    """Use the actual configured dataset/reader before the training loop; no optimizer."""
    from .frame_cache import CachedReader
    reader=loader.dataset.reader
    if not isinstance(reader,CachedReader):raise RuntimeError('CLIP is not using the validated session RGB reader')
    strict=reader.cache.require_hits
    if not strict and not getattr(reader.cache,'streaming_verified',False):raise RuntimeError('Streaming mode has not been explicitly validated')
    before=(reader.cache.hits,reader.cache.misses,reader.reader.opens)
    loader.dataset.__getitems__(list(range(min(8,len(loader.dataset)))))
    hits=reader.cache.hits-before[0];misses=reader.cache.misses-before[1];opens=reader.reader.opens-before[2]
    if strict and (hits!=min(8,len(loader.dataset)) or misses or opens):raise RuntimeError('CLIP cache probe failed')
    report={'cache_hits':hits,'cache_misses':misses,'video_opens':opens,
            'mode':'prepared' if strict else 'validated streaming','normalization':'CLIP processor after validated RGB'}
    print('Actual CLIP reader cache probe: '+json.dumps(report),flush=True)
    return report


def export_prepared_frames(cache,inputs,destination,*,reserve_gib=1):
    """Verified local ZIP for PRIVATE Kaggle dataset upload; never uploads itself."""
    require_prepared_cache(cache,inputs,announce=False)
    if not getattr(cache,'prepared_verified',False):raise ValueError('Optional full-dataset export requires complete preparation, not streaming')
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Existing export preserved; choose a new ZIP name')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with cache.connection() as db:rows=list(db.execute('SELECT key,sha256,metadata FROM frames ORDER BY key'))
    files={name:cache.root/name for name in ['cache_identity.json','manifest.sqlite',READY]}
    expected={name:digest(path) for name,path in files.items()}
    for key,sha,meta in rows:
        name=key[:2]+'/'+key+'.png';files[name]=cache.root/name;expected[name]=sha
    size=sum(p.stat().st_size for p in files.values())
    if shutil.disk_usage(destination.parent).free<size+reserve_gib*2**30:raise PreparationStorageError('Insufficient free space for dataset ZIP; prepared cache preserved')
    partial=destination.with_name('.partial_'+uuid.uuid4().hex+'.zip')
    try:
        with zipfile.ZipFile(partial,'x',zipfile.ZIP_STORED) as z:
            for i,(name,path) in enumerate(files.items(),1):
                if path.is_symlink() or digest(path)!=expected[name]:raise ValueError('Prepared export file changed: '+name)
                z.write(path,name)
                if i==1 or i%5000==0:print(f'PRIVATE dataset export: {i:,}/{len(files):,} files',flush=True)
            z.writestr('PRIVATE_DATASET_README.txt','Contains only the approved20,000 TRAIN and41,746 DEV lossless RGB frames.\nUpload through Kaggle Datasets with visibility PRIVATE. Follow original LAV-DF redistribution terms.\nAttach with FRAME_CACHE_ROOT pointing here and FRAME_CACHE_READONLY=True; run preparation to verify without video re-decoding.\nNo TEST frames, normalized tensors, masks, models or credentials are included.\n')
        with zipfile.ZipFile(partial) as z:
            for name,sha in expected.items():
                h=hashlib.sha256()
                with z.open(name) as member:
                    for block in iter(lambda:member.read(2**20),b''):h.update(block)
                if h.hexdigest()!=sha:raise ValueError('Dataset export member verification failed: '+name)
        partial.replace(destination)
    finally:partial.unlink(missing_ok=True)
    receipt={'archive':str(destination),'sha256':digest(destination),'bytes':destination.stat().st_size,'visibility_required':'PRIVATE','uploaded':False}
    _atomic_json(destination.with_suffix('.receipt.json'),receipt);return receipt
