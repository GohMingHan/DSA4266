#!/usr/bin/env python3
"""Preserve cached ResNet RGB features and replace pooled flow with pretrained TSN.

Original MP4s are decoded again: a 16x16 pooled descriptor cannot recover dense
flow. Only requested pair neighborhoods are computed. No dense-flow image folders
are written. Embeddings and timing are saved once, for frozen-feature training.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import cv2
import torch
from tqdm import tqdm
from common import VERSION, digest, file_stat, json_read, json_write, jsonl_write, sha256, within, require_unchanged
from video_io import probe_video, decoded_rgb, tool_version
from flow_frontend import DenseFlow, analysis_gray, encode_flow_image, stack_pair_indices, pack_stack
from tsn_encoder import TSNFlowEncoder
from visual_data import FEATURE_KIND

REFERENCE_KIND = 'resnet18_imagenet_rgb512_farneback_xy512_v1'


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def read_reference(path):
    path = Path(path).expanduser().resolve()
    data = json_read(path)
    if data.get('manifest_sha256') != digest({k: v for k, v in data.items() if k != 'manifest_sha256'}):
        raise ValueError('Reference manifest integrity check failed.')
    if data.get('feature_kind') != REFERENCE_KIND or data.get('input_dim') != 1024 or data.get('target') != 'modify_video':
        raise ValueError('Use the completed original RGB+Farneback feature manifest, not a frame index or TSN cache.')
    if data.get('test_only'):
        raise ValueError('The reference uses random/test-only RGB weights. Research extraction refuses it.')
    if not data.get('videos'):
        raise ValueError('Empty reference cohort.')
    names, groups = set(), {}
    for row in data['videos']:
        name, split = row['file'], row['split']
        if name in names or split not in ('train', 'dev', 'test'):
            raise ValueError('Duplicate video or invalid split in reference.')
        names.add(name)
        g = row['source_group']
        if g in groups and groups[g] != split:
            raise ValueError('Source group crosses splits; no automatic repair.')
        groups[g] = split
    return data


def choose_rows(reference, splits, limit, seed):
    result = []
    for split in dict.fromkeys(splits):
        pool = sorted([r for r in reference['videos'] if r['split'] == split], key=lambda r:r['file'])
        if not pool:
            raise ValueError(f'No {split} records in reference. Prepare that split using the original RGB cache protocol first.')
        if limit:
            if limit < 4 or limit % 2:
                raise ValueError('Pilot limit-per-split must be an even integer >=4.')
            subset = []
            for label in (False, True):
                eligible = [r for r in pool if bool(r['modify_video']) == label]
                eligible.sort(key=lambda r: digest([seed, split, label, r['file']]))
                if len(eligible) < limit//2:
                    raise ValueError(f'{split} has too few videos for the balanced software pilot.')
                subset.extend(eligible[:limit//2])
            pool = sorted(subset, key=lambda r:r['file'])
        result.extend(pool)
    return sorted(result, key=lambda r:r['file'])


def read_reference_arrays(root, row):
    p = within(root, row['cache']['path'])
    require_unchanged(p, row['cache']['stat'])
    if sha256(p) != row['cache']['sha256']:
        raise ValueError(f'Reference numerical feature hash changed: {p}')
    with np.load(p, allow_pickle=False) as f:
        arrays = {k: f[k].copy() for k in ('rgb','timestamp_s','source_frame_index')}
    n = row['feature_count']
    if arrays['rgb'].shape != (n,512) or arrays['rgb'].dtype != np.float32:
        raise ValueError('Reference RGB must be [T,512] float32.')
    ts, idx = arrays['timestamp_s'], arrays['source_frame_index']
    if ts.shape != (n,) or idx.shape != (n,) or n < 2 or not np.issubdtype(idx.dtype,np.integer):
        raise ValueError('Invalid reference sample-index shape.')
    if ts[0] != 0 or idx[0] != 0 or (np.diff(ts)<=0).any() or (np.diff(idx)<=0).any():
        raise ValueError('Reference timing is not zero-origin and strictly increasing.')
    if not np.isfinite(ts).all() or not np.isfinite(arrays['rgb']).all():
        raise ValueError('Nonfinite reference features.')
    return arrays


def compute_video(source, row, reference_arrays, encoder, dense_flow,
                  batch_size=8, ffmpeg='ffmpeg', ffprobe='ffprobe', timeout=1800):
    timing = probe_video(source, ffprobe, timeout)
    anchors = reference_arrays['source_frame_index']
    pts = timing['pts']
    if anchors[-1] >= len(pts) or not np.allclose(pts[anchors], reference_arrays['timestamp_s'], rtol=0, atol=1e-6):
        raise ValueError('Decoded source timestamps/indices disagree with cached RGB; cannot align streams.')
    if abs(timing['duration_s'] - row['duration_s']) > 1e-5:
        raise ValueError('Decoded duration changed relative to reference.')
    plan = stack_pair_indices(anchors, len(pts))
    needed = set(plan.reshape(-1).tolist())
    last_use = {}
    for j, ids in enumerate(plan):
        for k in set(ids.tolist()):
            last_use[k] = j
    cache, pending, outputs = {}, [], []
    next_window, pairs_done = 0, 0
    before = None
    flow_seconds = encode_seconds = 0.
    begun = time.perf_counter()
    with decoded_rgb(source, timing, ffmpeg, timeout) as decoder:
        for i, frame in decoder:
            gray = analysis_gray(frame)
            if i in needed:
                if before is None:
                    raise ValueError('A requested native pair has no predecessor.')
                t = time.perf_counter()
                cache[i] = encode_flow_image(dense_flow(before, gray))
                flow_seconds += time.perf_counter()-t
                pairs_done += 1
            before = gray
            while next_window < len(plan) and int(plan[next_window].max()) <= i:
                ids = plan[next_window]
                pending.append(pack_stack([cache[int(k)] for k in ids]))
                for k in set(ids.tolist()):
                    if last_use[k] == next_window:
                        del cache[k]
                next_window += 1
                if len(pending) >= batch_size:
                    t = time.perf_counter(); outputs.append(encoder.encode(pending))
                    encode_seconds += time.perf_counter()-t; pending = []
    if pending:
        t = time.perf_counter(); outputs.append(encoder.encode(pending)); encode_seconds += time.perf_counter()-t
    if next_window != len(anchors) or pairs_done != len(needed):
        raise ValueError('Not all scheduled TSN contexts were computed.')
    flow_features = np.concatenate(outputs,axis=0).astype(np.float32,copy=False)
    if flow_features.shape != (len(anchors),2048) or not np.isfinite(flow_features).all():
        raise ValueError('Nonfinite or incorrectly shaped TSN embeddings.')
    arrays = {**reference_arrays, 'flow': flow_features, 'flow_pair_end_indices': plan,
              'flow_context_start_s': pts[plan.min(axis=1)-1],
              'flow_context_end_s': pts[plan.max(axis=1)],
              'flow_pair_dt_s': pts[plan] - pts[plan-1],
              'flow_repeat_fraction': np.array([1-len(np.unique(p))/5 for p in plan],np.float32)}
    if not np.array_equal(arrays['rgb'],reference_arrays['rgb']):
        raise AssertionError('RGB data must not change.')
    stats = {'native_pairs_computed': pairs_done, 'flow_seconds':flow_seconds,
             'encoder_seconds':encode_seconds, 'video_seconds': time.perf_counter()-begun,
             'boundary_repeated_windows': int((arrays['flow_repeat_fraction']>0).sum())}
    return arrays, stats


def validate_cache(path, count):
    with np.load(path,allow_pickle=False) as f:
        if f['rgb'].shape != (count,512) or f['flow'].shape != (count,2048):
            raise ValueError('Invalid TSN cache dimensions.')
        for k in ('rgb','flow','timestamp_s','flow_context_start_s','flow_context_end_s','flow_pair_dt_s'):
            if not np.isfinite(f[k]).all():
                raise ValueError(f'Nonfinite cache array: {k}')
        ts, indices = f['timestamp_s'], f['source_frame_index']
        if ts.shape!=(count,) or indices.shape!=(count,) or count<2 or ts[0]!=0 or indices[0]!=0:
            raise ValueError('Invalid anchor timeline.')
        if (np.diff(ts)<=0).any() or (np.diff(indices)<=0).any():
            raise ValueError('Nonmonotonic feature sequence.')
        if f['flow_pair_end_indices'].shape != (count,5) or f['flow_pair_dt_s'].shape != (count,5):
            raise ValueError('Missing five-pair context.')
        if not (f['flow_pair_dt_s']>0).all():
            raise ValueError('Invalid flow-pair deltas.')
        return {'rgb_sha256':array_hash(f['rgb']), 'timestamp_sha256':array_hash(ts)}


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**arrays)
    os.replace(tmp,path)


@contextmanager
def output_lock(out):
    lock=out/'.extract.lock'
    try:
        with lock.open('x') as f:f.write(str(os.getpid())+'\n')
    except FileExistsError:
        raise ValueError(f'Output locked: {lock}. Remove only a confirmed stale lock; never run two writers.') from None
    try:yield
    finally:lock.unlink(missing_ok=True)


def run(args, encoder=None):
    reference_path=args.reference_manifest.expanduser().resolve()
    root=args.data_root.expanduser().resolve(); out=args.out.expanduser().resolve()
    reference=read_reference(reference_path); cache_root=reference_path.parent
    if not root.is_dir():raise ValueError(f'Dataset root missing: {root}')
    for protected in (root,cache_root):
        if out==protected or out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError('Use a separate output folder outside the original videos and reference cache.')
    rows=choose_rows(reference,args.splits,args.limit_per_split,args.seed)
    for r in rows:
        if not within(root,r['file']).is_file():raise FileNotFoundError(f'Missing source MP4: {r["file"]}')
    cv2.setNumThreads(args.opencv_threads);cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    flow=DenseFlow(args.flow_method)
    encoder=encoder or TSNFlowEncoder(args.flow_weights,args.device)
    config={'version':VERSION,'feature_kind':FEATURE_KIND,'rgb_dim':512,'flow_dim':2048,
            'rgb_source_feature_config':reference['feature_config'],
            'flow_encoder':encoder.identity, 'dense_flow':flow.identity,
            'frontend':{'clip_len':5,'channel_order':'dx1,dy1,...,dx5,dy5',
                        'analysis_size':224,'flow_image_size':256,'bound_pixels':20.,
                        'quantization':'round uint8 to [0,255]; no JPEG round trip',
                        'mean':128.,'std':128.,'stack_policy':'centered native pairs, nearest-pair boundary repeat',
                        'time':'RGB anchors preserved; separate source support bounds retained'},
            'numpy':np.__version__,'opencv':cv2.__version__,
            'ffmpeg':tool_version(args.ffmpeg),'ffprobe':tool_version(args.ffprobe),
            'code_sha256':{n:sha256(Path(__file__).parent/n) for n in
                          ('extract_tsn_flow.py','flow_frontend.py','tsn_encoder.py','video_io.py','common.py')},
            'test_only':bool(encoder.identity.get('test_only',False))}
    config_id=digest(config)
    if out.exists() and any(out.iterdir()) and not (out/'config.json').is_file():
        raise ValueError('Output is not a TSN feature cache. Use a fresh directory.')
    if (out/'config.json').is_file() and json_read(out/'config.json')['config_id']!=config_id:
        raise ValueError('Extractor, weights, estimator or configuration changed. Use a NEW output cache.')
    out.mkdir(parents=True,exist_ok=True)
    with output_lock(out):
        json_write(out/'config.json',{**config,'config_id':config_id})
        (out/'manifest.json').unlink(missing_ok=True)
        good,errors=[],[]; processed=resumed=0; interrupted=False;begin=time.perf_counter()
        try:
            for row in tqdm(rows,desc='Frozen TSN flow; preserved RGB',unit='video'):
                source=within(root,row['file'])
                sid=hashlib.sha256(row['file'].encode()).hexdigest()[:20]
                dest=out/'clips'/row['split']/(sid+'.npz'); rec=out/'records'/(sid+'.json')
                try:
                    # The cached RGB must describe this unchanged input recording.
                    require_unchanged(source,row['source_stat'])
                    before=file_stat(source); source_hash=sha256(source)
                    rgb=read_reference_arrays(cache_root,row)
                    signature=digest({'config_id':config_id,'reference_cache':row['cache']['sha256'],
                                      'source_sha256':source_hash,'file':row['file'],
                                      'visual_periods':row['visual_periods'],'duration_s':row['duration_s']})
                    old=json_read(rec) if rec.is_file() else None
                    if old and old.get('signature')==signature and dest.is_file() and sha256(dest)==old['cache']['sha256']:
                        checks=validate_cache(dest,row['feature_count'])
                        if checks['rgb_sha256']!=array_hash(rgb['rgb']) or checks['timestamp_sha256']!=array_hash(rgb['timestamp_s']):
                            raise ValueError('Resumed cache RGB/timestamps no longer match the reference.')
                        old['cache']['stat']=file_stat(dest);json_write(rec,old)
                        good.append(old);resumed+=1;continue
                    arrays,stats=compute_video(source,row,rgb,encoder,flow,args.encode_batch_size,
                                              args.ffmpeg,args.ffprobe,args.timeout)
                    require_unchanged(source,before)
                    atomic_npz(dest,arrays);checks=validate_cache(dest,row['feature_count'])
                    record={**row,'config_id':config_id,'signature':signature,
                            'source_sha256':source_hash,'source_stat':before,
                            'reference_cache_sha256':row['cache']['sha256'],
                            'preserved_rgb_sha256':checks['rgb_sha256'],
                            'preserved_timestamp_sha256':checks['timestamp_sha256'],
                            'extraction_stats':stats,
                            'cache':{'path':dest.relative_to(out).as_posix(),'sha256':sha256(dest),'stat':file_stat(dest)}}
                    json_write(rec,record);good.append(record);processed+=1
                except (ValueError,OSError,RuntimeError,subprocess.TimeoutExpired) as exc:
                    if isinstance(exc,torch.cuda.OutOfMemoryError) or isinstance(exc,OSError) and exc.errno in (13,28,30):raise
                    errors.append({'file':row['file'],'error':f'{type(exc).__name__}: {exc}'})
                    tqdm.write(f'FAILED {row["file"]}: {exc}')
        except KeyboardInterrupt:interrupted=True
        finally:
            jsonl_write(out/'errors.jsonl',errors)
            summary={'requested_videos':len(rows),'completed_videos':len(good),'processed':processed,'resumed':resumed,
                     'errors':len(errors),'interrupted':interrupted,'counts':dict(Counter(r['split'] for r in good)),
                     'feature_steps':sum(r['feature_count'] for r in good),
                     'cache_gib':sum(r['cache']['stat']['bytes'] for r in good)/1024**3,
                     'seconds':time.perf_counter()-begin,'test_only':config['test_only'],
                     'flow_method':args.flow_method,'rgb_preserved':True,
                     'cached_native_pairs':sum(r.get('extraction_stats',{}).get('native_pairs_computed',0) for r in good),
                     'note':'Timing includes decoding/TSN extraction, not UMMAFormer training.'}
            json_write(out/'summary.json',summary);print(json.dumps(summary,indent=2))
        if interrupted:raise SystemExit('Interrupted. Rerun identical command; completed videos resume. No partial manifest.')
        if errors or len(good)!=len(rows):raise ValueError('Incomplete extraction. See errors.jsonl; no trainable manifest written.')
        manifest={'version':VERSION,'feature_kind':FEATURE_KIND,'target':'modify_video','input_dim':2560,
                  'rgb_dim':512,'flow_dim':2048,'test_only':config['test_only'],'config_id':config_id,
                  'feature_config':config,'reference_manifest':str(reference_path),
                  'reference_manifest_sha256':sha256(reference_path),
                  'splits':list(dict.fromkeys(args.splits)),'pilot_limit_per_split':args.limit_per_split,
                  'time_policy':'unchanged_RGB_anchor_PTS_interpolation_to_768_bin_centers',
                  'videos':sorted(good,key=lambda r:r['file'])}
        manifest['manifest_sha256']=digest(manifest);json_write(out/'manifest.json',manifest)
        print(f'Ready: {out / "manifest.json"}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-manifest',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--flow-weights',type=Path,required=True)
    p.add_argument('--flow-method',choices=['farneback','tvl1'],default='farneback')
    p.add_argument('--splits',nargs='+',choices=['train','dev','test'],default=['train','dev'])
    p.add_argument('--limit-per-split',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--encode-batch-size',type=int,default=8)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--opencv-threads',type=int,default=2)
    p.add_argument('--ffmpeg',default='ffmpeg');p.add_argument('--ffprobe',default='ffprobe')
    p.add_argument('--timeout',type=float,default=1800)
    a=p.parse_args()
    if min(a.encode_batch_size,a.cpu_threads,a.opencv_threads)<1 or a.limit_per_split<0 or not math.isfinite(a.timeout) or a.timeout<=0:
        p.error('Invalid count/thread/timeout setting.')
    run(a)

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,RuntimeError,ImportError,KeyError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(2)
