#!/usr/bin/env python3
"""Cache pretrained RGB + OpenCV Farneback descriptors for an exact LAV-DF cohort.

No CLIP/TSN, pretrained temporal detector, HSV flow encoding, or random frozen
motion CNN. Raw videos are immutable. Failed clips prevent a trainable manifest.
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
import shutil
import sys
import subprocess
import time
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from common import (VERSION, digest, file_stat, json_read, json_write, jsonl_write,
                    sha256, within)
from cohort import select_cohort
from video_io import decoded_rgb, probe_video, sample_indices, tool_version

FEATURE_KIND = 'resnet18_imagenet_rgb512_farneback_xy512_v1'
FLOW_GRID = 16
FLOW_SIZE = 224
FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0)


def flow_descriptor(previous_gray, current_gray, delta_s):
    """16x16 area means of dx/dy in crop widths/heights per second; 512-D.

Temporal displacement is from ADJACENT native decoded frames, not two frames
0.1 seconds apart. Spatial resize is area averaging; velocity units remain
relative to the 224-pixel analysis crop, not relative to the 16x16 storage grid.
"""
    if previous_gray.shape != current_gray.shape or previous_gray.shape != (224, 224):
        raise ValueError('Flow geometry must be matching 224x224 grayscale crops.')
    if not math.isfinite(delta_s) or delta_s <= 0:
        raise ValueError('Positive finite native-frame time delta is required.')
    flow = cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, **FARNEBACK)
    # Fixed physical normalization; NEVER normalize each image/video by its extrema.
    flow[..., 0] /= (current_gray.shape[1] * delta_s)
    flow[..., 1] /= (current_gray.shape[0] * delta_s)
    coarse = cv2.resize(flow, (FLOW_GRID, FLOW_GRID), interpolation=cv2.INTER_AREA)
    descriptor = coarse.transpose(2, 0, 1).reshape(512).astype(np.float32)
    if not np.isfinite(descriptor).all():
        raise ValueError('Nonfinite optical flow descriptor.')
    return descriptor


class RGBEncoder:
    def __init__(self, device, local_weights=None, test_random=False):
        # Import torchvision only for extraction, not for cached-feature training.
        import torchvision
        from torchvision.models import resnet18, ResNet18_Weights
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF
        self.device, self.TF = device, TF
        self.interpolation = InterpolationMode.BILINEAR
        if test_random:
            torch.manual_seed(123)
            self.model = resnet18(weights=None)
            weights_info = {'kind': 'RANDOM_TEST_ONLY', 'test_only': True}
        elif local_weights:
            path = Path(local_weights).expanduser().resolve()
            self.model = resnet18(weights=None)
            self.model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            if not sha256(path).startswith('f37072fd'):
                raise ValueError('Local --rgb-weights must be the official resnet18-f37072fd.pth; '
                                 'refusing unidentified/random weights.')
            weights_info = {'kind': 'torchvision_resnet18_IMAGENET1K_V1', 'sha256': sha256(path),
                            'test_only': False}
        else:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.model = resnet18(weights=weights)  # Official download and hash check; no fallback.
            path = Path(torch.hub.get_dir()) / 'checkpoints' / Path(weights.url).name
            checksum = sha256(path)
            if not checksum.startswith('f37072fd'):
                raise ValueError('Cached RGB weights fail the official filename hash prefix check.')
            weights_info = {'kind': 'torchvision_resnet18_IMAGENET1K_V1',
                            'url': weights.url, 'sha256': checksum, 'test_only': False}
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(device)
        self.model.requires_grad_(False)
        self.mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
        self.identity = {'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                         'weights': weights_info,
                         'geometry': 'PIL RGB bilinear shorter-edge256 -> center224; same crop for flow',
                         'normalization': 'RGB/255 then ImageNet mean/std; no augmentation'}

    def crop(self, native_rgb):
        image = Image.fromarray(native_rgb)
        image = self.TF.resize(image, 256, interpolation=self.interpolation, antialias=True)
        image = self.TF.center_crop(image, [224, 224])
        return np.array(image, dtype=np.uint8, copy=True)

    @torch.inference_mode()
    def encode(self, crops):
        x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to(self.device, dtype=torch.float32) / 255.
        output = self.model((x - self.mean) / self.std).float().cpu().numpy()
        if output.shape != (len(crops), 512) or not np.isfinite(output).all():
            raise ValueError('Invalid RGB features.')
        return output


def extract_one(video, encoder, sample_fps=10., encode_batch_size=32,
                ffmpeg='ffmpeg', ffprobe='ffprobe', timeout=600):
    timing = probe_video(video, ffprobe, timeout)
    indices = sample_indices(timing['pts'], sample_fps)
    selected = set(indices.tolist())
    crops, rgb_parts, flow_parts, flow_dt, valid = [], [], [], [], []
    previous_gray = None
    with decoded_rgb(video, timing, ffmpeg, timeout) as decoder:
        for index, frame in decoder:
            # Geometry is shared by RGB and flow and never depends on labels.
            crop = encoder.crop(frame)
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            if index in selected:
                if index == 0:
                    # No earlier frame exists. Explicit boundary convention, NOT missing-video repair.
                    flow_parts.append(np.zeros(512, np.float32)); flow_dt.append(0.); valid.append(False)
                else:
                    dt = float(timing['pts'][index] - timing['pts'][index-1])
                    flow_parts.append(flow_descriptor(previous_gray, gray, dt))
                    flow_dt.append(dt); valid.append(True)
                crops.append(crop)
                if len(crops) >= encode_batch_size:
                    rgb_parts.append(encoder.encode(crops)); crops = []
            previous_gray = gray
    if crops:
        rgb_parts.append(encoder.encode(crops))
    arrays = {'rgb': np.concatenate(rgb_parts, axis=0), 'flow': np.stack(flow_parts),
              'timestamp_s': timing['pts'][indices], 'source_frame_index': indices.astype(np.int64),
              'flow_dt_s': np.array(flow_dt, np.float64), 'flow_valid': np.array(valid, bool)}
    if len(arrays['rgb']) != len(indices):
        raise ValueError('Feature count mismatch.')
    return arrays, {k: v for k, v in timing.items() if k != 'pts'}


def atomic_npz(path, arrays):
    tmp = path.with_name(path.name + '.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def validate_cache(path, count=None):
    with np.load(path, allow_pickle=False) as z:
        rgb, flow, ts, idx = z['rgb'], z['flow'], z['timestamp_s'], z['source_frame_index']
        if rgb.ndim != 2 or rgb.shape[1] != 512 or flow.shape != rgb.shape:
            raise ValueError(f'Bad feature shape: {path}')
        n = len(rgb)
        if n < 2 or count is not None and n != count:
            raise ValueError(f'Unexpected feature count: {path}')
        if ts.shape != (n,) or idx.shape != (n,) or (np.diff(ts) <= 0).any() or (np.diff(idx) <= 0).any():
            raise ValueError(f'Bad feature timestamps/indices: {path}')
        if not np.isfinite(rgb).all() or not np.isfinite(flow).all() or not np.isfinite(ts).all():
            raise ValueError(f'Nonfinite cache: {path}')
        if ts[0] != 0 or idx[0] != 0 or np.any(flow[0] != 0):
            raise ValueError(f'Invalid initial frame/flow convention: {path}')
        if z['flow_valid'].shape != (n,) or z['flow_valid'][0] or not z['flow_valid'][1:].all():
            raise ValueError(f'Invalid flow-valid flags: {path}')
        if z['flow_dt_s'].shape != (n,) or z['flow_dt_s'][0] != 0 or not (z['flow_dt_s'][1:] > 0).all():
            raise ValueError(f'Invalid adjacent-frame deltas: {path}')
        return n


@contextmanager
def output_lock(root):
    lock = root / '.extract.lock'
    try:
        with lock.open('x') as f:
            f.write(f'{os.getpid()}\n')
    except FileExistsError:
        raise ValueError(f'Extractor lock exists: {lock}. Do not run two writers. '
                         'Remove a stale lock only after verifying no extractor is running.') from None
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def run(args):
    root, out, cohort = [p.expanduser().resolve() for p in (args.data_root, args.out, args.cohort_videos)]
    if not root.is_dir() or not cohort.is_file():
        raise ValueError('Original --data-root and CLIP --cohort-videos must exist.')
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('Keep output separate from source videos and their ancestors.')
    if out == cohort.parent or cohort.is_relative_to(out):
        raise ValueError('Do not overwrite the CLIP manifest directory.')
    meta = args.metadata.expanduser().resolve() if args.metadata else root / 'metadata.min.json'
    if not meta.is_file() and not args.metadata:
        meta = root / 'metadata.json'
    rows = select_cohort(meta, cohort, args.splits, args.limit_per_split, args.seed)
    missing = [r['file'] for r in rows if not within(root, r['file']).is_file()]
    if missing:
        raise ValueError(f'{len(missing)} referenced original MP4s are missing, e.g. {missing[:5]}')
    if out.exists() and any(out.iterdir()) and not (out / 'config.json').is_file():
        raise ValueError('Output is not an RGB-flow cache. Use a new --out.')
    from run_experiment import get_device
    device = get_device(args.device, 'none')
    cv2.setNumThreads(args.opencv_threads)
    cv2.ocl.setUseOpenCL(False)  # Reproducible CPU Farneback; RGB may use CUDA.
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    encoder = RGBEncoder(device, args.rgb_weights, args.test_random_rgb)
    code_hashes = {f: sha256(Path(__file__).parent / f) for f in
                   ('extract_rgb_flow.py', 'video_io.py', 'cohort.py', 'common.py')}
    config = {'version': VERSION, 'feature_kind': FEATURE_KIND, 'rgb_encoder': encoder.identity,
              'sample_fps': args.sample_fps, 'flow_algorithm': FARNEBACK,
              'flow_grid': [16, 16], 'flow_descriptor': 'dx then dy; crop fractions/sec; area means',
              'flow_pair': 'previous native decoded frame -> current sampled frame',
              'initial_flow': 'zeros; flow_valid=false at frame zero',
              'sampling': 'first decoded frame on/after 0,1/fps,...; unique; actual PTS retained',
              'time_origin': 'first_decoded_video_frame', 'numpy': np.__version__, 'opencv': cv2.__version__,
              'ffmpeg': tool_version(args.ffmpeg), 'ffprobe': tool_version(args.ffprobe),
              'device_type': device.type, 'precision': 'fp32_no_tf32', 'code_sha256': code_hashes, 'test_only': args.test_random_rgb}
    config_id = digest(config)
    if (out / 'config.json').exists() and json_read(out / 'config.json')['config_id'] != config_id:
        raise ValueError('Extractor/settings/weights/environment changed. Use a NEW --out cache.')
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out):
        json_write(out / 'config.json', {**config, 'config_id': config_id})
        # Never leave an old manifest eligible when a new/expanded request fails.
        (out / 'manifest.json').unlink(missing_ok=True)
        good, errors, warnings = [], [], []
        processed, resumed = 0, 0
        interrupted = False
        start = time.perf_counter()
        try:
            for row in tqdm(rows, desc='RGB + native-pair Farneback', unit='video'):
                source = within(root, row['file'])
                sid = hashlib.sha256(row['file'].encode()).hexdigest()[:20]
                path = out / 'clips' / row['split'] / (sid + '.npz')
                record = out / 'records' / (sid + '.json')
                before = file_stat(source)
                signature = digest({'config': config_id, 'row': row, 'source_sha256': sha256(source)})
                try:
                    cached = json_read(record) if record.is_file() else None
                    if cached and cached.get('signature') == signature and path.is_file():
                        if sha256(path) == cached['cache']['sha256']:
                            validate_cache(path, cached['feature_count'])
                            cached['cache']['stat'] = file_stat(path)
                            json_write(record, cached)
                            good.append(cached)
                            warnings.extend(cached.get('warnings', []))
                            resumed += 1
                            continue
                    arrays, timing = extract_one(source, encoder, args.sample_fps,
                        args.encode_batch_size, args.ffmpeg, args.ffprobe, args.timeout)
                    if file_stat(source) != before:
                        raise ValueError('Source MP4 changed during extraction.')
                    if any(b > timing['duration_s'] + 1e-5 for a, b in row['visual_periods']):
                        raise ValueError('Visual annotation exceeds decoded duration. No clamping or relabeling.')
                    current_warnings = []
                    if abs(timing['duration_s'] - row['metadata_duration_s']) > .1:
                        current_warnings.append({'file': row['file'], 'type': 'metadata_duration_discrepancy',
                            'metadata_s': row['metadata_duration_s'], 'decoded_s': timing['duration_s']})
                    if 'reference_decoded_end_s' in row and abs(row['reference_decoded_end_s'] - timing['duration_s']) > .02:
                        raise ValueError('Decoded duration differs from the CLIP reference by >20ms; audit timing.')
                    if timing['tail_duration_estimated']:
                        current_warnings.append({'file': row['file'], 'type': 'tail_duration_estimated'})
                    if row['modify_video'] and not any(np.any((arrays['timestamp_s'] >= a) & (arrays['timestamp_s'] < b))
                                                      for a, b in row['visual_periods']):
                        current_warnings.append({'file': row['file'], 'type': 'visual_edit_not_sampled',
                                                'action': 'kept positive; no label-based resampling'})
                    atomic_npz(path, arrays)
                    validate_cache(path, len(arrays['rgb']))
                    info = {**row, **timing, 'signature': signature, 'config_id': config_id,
                            'feature_count': len(arrays['rgb']), 'source_stat': before,
                            'cache': {'path': path.relative_to(out).as_posix(), 'sha256': sha256(path),
                                      'stat': file_stat(path)}, 'warnings': current_warnings}
                    json_write(record, info)
                    good.append(info); warnings.extend(current_warnings); processed += 1
                except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        raise
                    # Disk-full/permissions are global failures, not 5000 per-video retries.
                    if isinstance(exc, OSError) and exc.errno in (13, 28, 30):
                        raise
                    errors.append({'file': row['file'], 'error': f'{type(exc).__name__}: {exc}'})
                    tqdm.write(f'FAILED {row["file"]}: {exc}')
        except KeyboardInterrupt:
            interrupted = True
        jsonl_write(out / 'errors.jsonl', errors)
        jsonl_write(out / 'warnings.jsonl', warnings)
        summary = {'requested_videos': len(rows), 'completed_videos': len(good), 'processed': processed,
                   'resumed': resumed, 'errors': len(errors), 'interrupted': interrupted,
                   'counts': dict(Counter(r['split'] for r in good)),
                   'feature_steps': sum(r['feature_count'] for r in good),
                   'cache_gib': sum(r['cache']['stat']['bytes'] for r in good) / 1024**3,
                   'seconds': time.perf_counter() - start, 'warnings': len(warnings),
                   'pilot_limit_per_split': args.limit_per_split, 'test_only': args.test_random_rgb}
        json_write(out / 'summary.json', summary)
        print(json.dumps(summary, indent=2))
        if interrupted:
            raise SystemExit('Interrupted. Re-run the same extraction command to resume; no trainable manifest written.')
        if errors or len(good) != len(rows):
            raise ValueError(f'Extraction incomplete. Inspect {out / "errors.jsonl"}; no trainable manifest written.')
        for split in args.splits:
            if len({r['modify_video'] for r in good if r['split'] == split}) != 2:
                raise ValueError(f'{split} lacks both visual classes for the planned evaluation.')
        manifest = {'version': VERSION, 'feature_kind': FEATURE_KIND, 'target': 'modify_video',
                    'input_dim': 1024, 'test_only': args.test_random_rgb, 'config_id': config_id,
                    'feature_config': config, 'reference_cohort_path': str(cohort),
                    'reference_cohort_sha256': sha256(cohort), 'source_metadata_sha256': sha256(meta),
                    'splits': list(dict.fromkeys(args.splits)),
                    'time_policy': 'actual_PTS_interpolation_to_768_uniform_bin_centers',
                    'pilot_limit_per_split': args.limit_per_split, 'videos': sorted(good, key=lambda r: r['file'])}
        manifest['manifest_sha256'] = digest(manifest)
        json_write(out / 'manifest.json', manifest)
        print(f'Ready: {out / "manifest.json"}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--cohort-videos', type=Path, required=True, help='Your CLIP videos.jsonl, NOT frames.jsonl.')
    p.add_argument('--metadata', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--splits', nargs='+', choices=['train', 'dev', 'test'], default=['train', 'dev'])
    p.add_argument('--sample-fps', type=float, default=10., help='0: retain every decoded frame.')
    p.add_argument('--device', default='cuda:0', help='RGB encoder device; OpenCV flow is CPU.')
    p.add_argument('--encode-batch-size', type=int, default=32)
    p.add_argument('--opencv-threads', type=int, default=2)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--limit-per-split', type=int, default=0, help='0: exact whole cohort; even >0: balanced software pilot.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--ffprobe', default='ffprobe')
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--rgb-weights', type=Path, help='Optional local official resnet18-f37072fd.pth for offline extraction.')
    p.add_argument('--test-random-rgb', action='store_true', help='SOFTWARE TEST ONLY; resulting cache cannot train/evaluate.')
    args = p.parse_args()
    if not math.isfinite(args.sample_fps) or not 0 <= args.sample_fps <= 240 or min(
            args.encode_batch_size, args.opencv_threads, args.cpu_threads) < 1 or args.limit_per_split < 0:
        p.error('Invalid rate/batch/thread/limit settings.')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error('timeout must be finite and positive.')
    if args.test_random_rgb and args.rgb_weights:
        p.error('Cannot combine random-test mode with pretrained weights.')
    for tool in (args.ffmpeg, args.ffprobe):
        if not shutil.which(tool):
            p.error(f'Executable not found: {tool}')
    run(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
