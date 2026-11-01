"""LAV-DF -> timestamped ORIGINAL frames; no face detection/alignment/cropping.

Exports crop_index.csv compatible with the existing Forensics Adapter project's
build_manifest.py. crop_path is a compatibility field holding a FULL frame path.
No spatial mask or authenticity prediction is generated. Decode acceptance is NOT
human approval of face quality. Inspect audit_sample.html before final training.

Requires Python >=3.10, numpy, opencv-python (or existing conda OpenCV), tqdm,
and the ffmpeg/ffprobe executables. Does not import dlib or PyTorch.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, closing
from datetime import datetime, timezone
import csv
import hashlib
import heapq
import html
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Iterator
from urllib.parse import quote

import cv2
import numpy as np
from tqdm import tqdm

VERSION = 1
INDEX_FIELDS = ['file', 'crop_path', 'timestamp_s', 'source_frame_index',
                'review_status', 'split', 'frame_id', 'record_path',
                'preprocessing', 'width', 'height']

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in values:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    tmp.replace(path)


def relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Expected a nonempty relative path.')
    value = value.replace('\\', '/')
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or ':' in value or '\x00' in value or not p.parts:
        raise ValueError(f'Unsafe relative path: {value!r}')
    return p.as_posix()


def under(root: Path, rel: str) -> Path:
    root = root.resolve()
    target = (root / relative_path(rel)).resolve()
    if not target.is_relative_to(root):
        raise ValueError('Path escapes its root, including through a symlink.')
    return target


def strict_bool(value: Any) -> bool:
    if isinstance(value, bool) or type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError(f'Expected boolean or 0/1, got {value!r}.')


def normalize_metadata(raw: dict) -> dict:
    file = relative_path(raw['file'])
    split = raw['split']
    if split not in ('train', 'dev', 'test'):
        raise ValueError('Unknown split; preserve LAV-DF train/dev/test.')
    duration = float(raw['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid metadata duration.')
    ma, mv = strict_bool(raw['modify_audio']), strict_bool(raw['modify_video'])
    periods = []
    for pair in raw['fake_periods']:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f'Invalid interval: {pair!r}')
        a, b = map(float, pair)
        if not all(map(math.isfinite, (a, b))) or a < 0 or b <= a:
            raise ValueError(f'Invalid interval: {pair!r}')
        # Deliberately do NOT clip/reject at metadata duration.
        periods.append([a, b])
    if bool(ma or mv) != bool(periods):
        raise ValueError('Modification flags and temporal annotations disagree.')
    return {'file': file, 'split': split, 'duration': duration,
            'original': str(raw.get('original') or '').replace('\\', '/') or None,
            'modify_audio': ma, 'modify_video': mv, 'fake_periods': sorted(periods)}


def load_metadata(path: Path) -> tuple[list[dict], list[dict]]:
    raw = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(raw, list):
        raise ValueError('Expected the official LAV-DF metadata JSON list.')
    rows, errors, seen = [], [], set()
    for number, item in enumerate(raw):
        try:
            row = normalize_metadata(item)
        except Exception as exc:
            errors.append({'metadata_row': number, 'file': item.get('file') if isinstance(item, dict) else None,
                           'stage': 'metadata', 'error': str(exc)})
            continue
        if row['file'] in seen:
            raise ValueError(f'Duplicate metadata file: {row["file"]}')
        seen.add(row['file'])
        rows.append(row)
    groups = {}
    for row in rows:
        group = row['original'] or row['file']
        if group in groups and groups[group] != row['split']:
            raise ValueError(f'Source group crosses splits: {group}. No automatic split repair.')
        groups[group] = row['split']
    if not rows:
        raise ValueError('No valid metadata rows.')
    return rows, errors


def run_command(command: list[str], timeout: float = 600) -> str:
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{Path(command[0]).name}: {result.stderr[-3000:]}')
    return result.stdout


def parse_timing(frames: list[dict]) -> dict:
    if not frames:
        raise ValueError('No decoded video frames.')
    times, durations, sizes = [], [], set()
    for f in frames:
        value = f.get('best_effort_timestamp_time', f.get('pts_time'))
        if value is None:
            raise ValueError('Decoded frame is missing a timestamp; no FPS fallback.')
        times.append(float(value))
        durations.append(float(f.get('duration_time', f.get('pkt_duration_time', 0)) or 0))
        sizes.add((int(f['width']), int(f['height'])))
    absolute = np.asarray(times, np.float64)
    if not np.isfinite(absolute).all() or (np.diff(absolute) <= 0).any():
        raise ValueError('Nonfinite, duplicate, or nonmonotonic decoded timestamps.')
    if len(sizes) != 1:
        raise ValueError('Resolution changes within the video; unsupported by the raw-frame reader.')
    w, h = sizes.pop()
    if min(w, h) < 1:
        raise ValueError('Invalid decoded frame size.')
    pts = absolute - absolute[0]
    tail = durations[-1]
    estimated = False
    if not math.isfinite(tail) or tail <= 0:
        if len(pts) < 2:
            raise ValueError('Cannot determine the extent of a single-frame video.')
        tail = float(np.median(np.diff(pts)))
        estimated = True
    return {'pts': pts, 'width': w, 'height': h,
            'first_video_pts_s': float(absolute[0]), 'decoded_video_end_s': float(pts[-1] + tail),
            'tail_duration_estimated': estimated, 'decoded_frame_count': len(pts)}


def probe_video(video: Path, ffprobe: str = 'ffprobe', timeout: float = 600) -> dict:
    # All decoded frames, not packet DTS or an assumed constant FPS.
    result = json.loads(run_command([ffprobe, '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_streams', '-show_entries',
        'frame=best_effort_timestamp_time,pts_time,duration_time,pkt_duration_time,width,height:'
        'stream=sample_aspect_ratio,width,height:stream_tags=rotate:stream_side_data=rotation',
        '-of', 'json', str(video)], timeout))
    timing = parse_timing(result.get('frames', []))
    stream = (result.get('streams') or [{}])[0]
    rotations = [float(s['rotation']) for s in stream.get('side_data_list', []) if 'rotation' in s]
    if 'rotate' in stream.get('tags', {}):
        rotations.append(float(stream['tags']['rotate']))
    if any(abs(x % 360) > 1e-6 and abs(x % 360 - 360) > 1e-6 for x in rotations):
        raise ValueError('Nonzero display rotation: flag for explicit geometric handling; no silent autorotation.')
    sar = stream.get('sample_aspect_ratio', '1:1')
    if sar not in ('1:1', '0:1', 'N/A', None):
        raise ValueError(f'Non-square pixel aspect ratio {sar}; requires explicit preprocessing.')
    return timing


def select_indices(pts: np.ndarray, sample_fps: float) -> list[int]:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 1 or not len(pts) or not np.isfinite(pts).all() or (np.diff(pts) <= 0).any():
        raise ValueError('Expected a nonempty strictly increasing timestamp vector.')
    if not math.isfinite(sample_fps) or sample_fps < 0:
        raise ValueError('Invalid sample FPS.')
    if sample_fps == 0:
        return list(range(len(pts)))
    # First actual frame on/after each 0, 1/fps, ... grid position. Deduplicate;
    # never create synthetic/repeated frames and never substitute the grid time.
    grid = np.arange(math.floor(float(pts[-1]) * sample_fps + 1e-8) + 1) / sample_fps
    indices = np.searchsorted(pts, grid - 1e-8, side='left')
    return np.unique(indices[indices < len(pts)]).tolist()


def _read_exact(stream, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def decode_frames(video: Path, timing: dict, ffmpeg: str = 'ffmpeg', timeout: float = 600) -> Iterator[tuple[int, np.ndarray]]:
    """Pair sequential passthrough-decoded frames with independently probed PTS.

    All frames are consumed and their count verified, even when only some are
    selected. A decode mismatch invalidates the video's COMPLETE training index.
    """
    w, h, count = timing['width'], timing['height'], timing['decoded_frame_count']
    command = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror',
               '-threads', '1', '-noautorotate', '-i', str(video), '-map', '0:v:0',
               '-an', '-sn', '-dn', '-fps_mode', 'passthrough', '-threads', '1',
               '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1']
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err,
                                start_new_session=(os.name != "nt"))
        expired = threading.Event()
        def stop():
            expired.set()
            if proc.poll() is None:
                proc.kill()
        timer = threading.Timer(timeout, stop)
        timer.daemon = True
        timer.start()
        try:
            for index in range(count):
                raw = _read_exact(proc.stdout, w * h * 3)
                if len(raw) != w * h * 3:
                    err.seek(0)
                    detail = err.read().decode('utf-8', 'replace')[-2000:]
                    raise ValueError(f'Incomplete decode at frame {index}/{count}; timeout={expired.is_set()}. {detail}')
                yield index, np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            extra = proc.stdout.read(1)
            if extra:
                raise ValueError('FFmpeg decoded more frames than FFprobe; do not guess index-to-time correspondence.')
            code = proc.wait(timeout=30)
            err.seek(0)
            diagnostic = err.read().decode('utf-8', 'replace')[-2000:]
            if code or expired.is_set() or diagnostic.strip():
                raise ValueError(f'Decoder reported an error/timeout (exit {code}): {diagnostic}')
        finally:
            timer.cancel()
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            if proc.stdout:
                proc.stdout.close()


def source_signature(source: Path, meta: dict, config_id: str) -> str:
    s = source.stat()
    return digest({'file': str(source.resolve()), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns,
                   'metadata': meta, 'config_id': config_id})


def record_path(out: Path, file: str) -> Path:
    return out / 'records' / (hashlib.sha256(file.encode()).hexdigest()[:20] + '.json')


def save_png(path: Path, bgr: np.ndarray, compression: int = 3) -> dict:
    """Lossless PNG storage of the decoded 8-bit BGR image, at its native size."""
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError('Expected a decoded uint8 BGR frame.')
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode('.png', bgr, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not ok:
        raise OSError(f'PNG encoding failed: {path}')
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(data.tobytes())
    tmp.replace(path)
    st = path.stat()
    return {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}


def complete_record_valid(record: dict, out: Path) -> bool:
    if record.get('status') != 'complete' or not record.get('frames'):
        return False
    for f in record['frames']:
        try:
            stat = under(out, f['crop_path']).stat()
        except OSError:
            return False
        if stat.st_size != f['bytes'] or stat.st_mtime_ns != f['mtime_ns']:
            return False
    return True


def process_clip(meta: dict, data_root: Path, out: Path, config: dict, config_id: str,
                 ffmpeg: str = 'ffmpeg', ffprobe: str = 'ffprobe', timeout: float = 600,
                 retry_errors: bool = False) -> str:
    """Commit only complete videos. Resume never indexes partially written frames."""
    path = record_path(out, meta['file'])
    signature = None
    try:
        source = under(data_root, meta['file'])
        signature = source_signature(source, meta, config_id)
        if path.exists():
            old = json.loads(path.read_text(encoding='utf-8'))
            if old.get('source_signature') == signature and old.get('config_id') == config_id:
                if complete_record_valid(old, out):
                    return 'resumed'
                if old.get('status') == 'error' and not retry_errors:
                    return 'cached_error'
        # Invalid/stale/partial work must never remain eligible for the index.
        base = {'file': meta['file'], 'metadata': meta, 'config_id': config_id,
                'source_signature': signature, 'frames': [], 'status': 'in_progress'}
        write_json(path, base)
        timing = probe_video(source, ffprobe, timeout)
        indices = set(select_indices(timing['pts'], config['sample_fps']))
        frames = []
        # closing() also terminates the decoder when Ctrl+C arrives while saving.
        with closing(decode_frames(source, timing, ffmpeg, timeout)) as decoded:
            for index, bgr in decoded:
                if index not in indices:
                    continue
                relative = f"frames/{meta['split']}/{path.stem}/frame_{index:08d}.png"
                stat = save_png(under(out, relative), bgr, config['png_compression'])
                frames.append({'file': meta['file'], 'crop_path': relative,
                    'source_frame_index': index, 'timestamp_s': float(timing['pts'][index]),
                    'absolute_pts_s': float(timing['first_video_pts_s'] + timing['pts'][index]),
                    'split': meta['split'], 'review_status': 'auto_accepted',
                    'preprocessing': 'original_frame_no_crop',
                    'width': timing['width'], 'height': timing['height'],
                    'frame_id': f'{path.stem}:{index:08d}',
                    'record_path': path.relative_to(out).as_posix(), **stat})
        if len(frames) != len(indices):
            raise ValueError('Not all scheduled frames were saved.')
        if source_signature(source, meta, config_id) != signature:
            raise ValueError('Source/metadata changed during extraction.')
        warnings = []
        actual_end = timing['decoded_video_end_s']
        if abs(actual_end - meta['duration']) > 0.1:
            warnings.append({'type': 'metadata_duration_discrepancy',
                'metadata_duration_s': meta['duration'], 'decoded_video_end_s': actual_end})
        last_edit = max((b for a, b in meta['fake_periods']), default=0.0)
        if meta['modify_video'] and last_edit > actual_end + 0.02:
            warnings.append({'type': 'visual_annotation_exceeds_decoded_video',
                'annotation_end_s': last_edit, 'decoded_video_end_s': actual_end,
                'action': 'Manifest builder must reject inconsistent visual labels; no clamping.'})
        if timing['tail_duration_estimated']:
            warnings.append({'type': 'tail_duration_estimated'})
        write_json(path, {**base, 'status': 'complete', 'frames': frames,
            'timing': {k: v for k, v in timing.items() if k != 'pts'}, 'warnings': warnings})
        return 'processed'
    except KeyboardInterrupt:
        # A hard kill leaves in_progress; both states restart this video next time.
        write_json(path, {'file': meta['file'], 'metadata': meta, 'status': 'interrupted',
            'source_signature': signature, 'config_id': config_id, 'frames': []})
        raise
    except OSError as exc:
        # Do not loop through the whole dataset when output storage is unusable.
        if exc.errno in (13, 28, 30):  # EACCES, ENOSPC, EROFS
            raise
        write_json(path, {'file': meta['file'], 'metadata': meta, 'status': 'error',
            'source_signature': signature, 'config_id': config_id,
            'error_type': type(exc).__name__, 'error': str(exc), 'frames': []})
        tqdm.write(f'FLAGGED {meta["file"]}: {exc}')
        return 'error'
    except Exception as exc:
        write_json(path, {'file': meta['file'], 'metadata': meta, 'status': 'error',
            'source_signature': signature, 'config_id': config_id,
            'error_type': type(exc).__name__, 'error': str(exc), 'frames': []})
        tqdm.write(f'FLAGGED {meta["file"]}: {type(exc).__name__}: {exc}')
        return 'error'


@contextmanager
def output_lock(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    lock = out / '.extractor.lock'
    try:
        with lock.open('x', encoding='utf-8') as handle:
            handle.write(f'PID={os.getpid()}\nOnly one writer per extraction folder.\n')
    except FileExistsError:
        raise RuntimeError('Output is locked. After verifying no extractor is running, '
                           'remove only .extractor.lock if it is stale.') from None
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def write_audit(out: Path, sample: list[dict]) -> None:
    parts = ['<!doctype html><meta charset="utf-8"><title>Original-frame audit</title>',
        '<style>body{font:16px sans-serif;max-width:1000px;margin:auto;padding:20px}'
        'main{display:flex;flex-wrap:wrap;gap:16px}article{width:280px;overflow-wrap:anywhere}'
        'img{max-width:100%;height:auto}small{display:block}</style>',
        '<h1>Original-frame audit</h1><p>No face detector, crop, alignment or spatial mask. '
        'All technically decoded frames are retained. This is NOT a face-quality '
        'certification or an authenticity prediction. Inspect examples before training.</p><main>']
    for f in sample:
        link = quote(f['crop_path'], safe='/._-')
        parts.append(f'<article><a href="{link}"><img loading="lazy" src="{link}"></a>'
            f'<small>{html.escape(f["file"])}</small><small>frame {f["source_frame_index"]}, '
            f'{f["timestamp_s"]:.6f} s; {f["width"]} x {f["height"]}</small></article>')
    parts.append('</main>')
    tmp = out / 'audit_sample.html.tmp'
    tmp.write_text('\n'.join(parts), encoding='utf-8')
    tmp.replace(out / 'audit_sample.html')


def rebuild_indexes(out: Path, metas: list[dict], data_root: Path, config_id: str,
                    metadata_error_count: int = 0) -> dict:
    """Rebuild small summaries/CSV from completed per-video records, streaming."""
    meta_map = {m['file']: m for m in metas}
    counts, split_counts, categories = Counter(), Counter(), {}
    heap = []
    temp = out / 'crop_index.csv.tmp'
    with temp.open('w', encoding='utf-8', newline='') as handle, \
         (out / 'video_errors.jsonl.tmp').open('w', encoding='utf-8') as errors, \
         (out / 'warnings.jsonl.tmp').open('w', encoding='utf-8') as warnings:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for path in sorted((out / 'records').glob('*.json')):
            try:
                r = json.loads(path.read_text(encoding='utf-8'))
                if r.get('config_id') != config_id:
                    continue
                if r.get('metadata') != meta_map.get(r.get('file')):
                    raise ValueError('Record metadata changed; rerun extraction for this video.')
                source = under(data_root, r['file'])
                # Missing-source errors are still visible; matching completed work is checked.
                if r['status'] == 'complete':
                    if source_signature(source, r['metadata'], config_id) != r['source_signature']:
                        raise ValueError('Source changed; rerun extraction for this video.')
                    if not complete_record_valid(r, out):
                        raise ValueError('Saved frame missing/changed; rerun extraction for this video.')
                counts['videos_in_cache'] += 1
                if r['status'] in ('in_progress', 'interrupted'):
                    counts['incomplete_videos'] += 1
                    continue
                if r['status'] != 'complete':
                    counts['video_errors'] += 1
                    errors.write(json.dumps(r, allow_nan=False) + '\n')
                    continue
                counts['completed_videos'] += 1
                split_counts[r['metadata']['split'] + '_videos'] += 1
                ma, mv = r['metadata']['modify_audio'], r['metadata']['modify_video']
                category = 'both' if ma and mv else 'audio_only' if ma else 'visual_only' if mv else 'real'
                key = r['metadata']['split'] + '/' + category
                category_count = categories.setdefault(key, {'videos': 0, 'frames': 0})
                category_count['videos'] += 1
                for warning in r.get('warnings', []):
                    warnings.write(json.dumps({'file': r['file'], **warning}, allow_nan=False) + '\n')
                    counts['warnings'] += 1
                for frame in r['frames']:
                    writer.writerow(frame)
                    counts['extracted_frames'] += 1
                    counts['png_bytes'] += frame['bytes']
                    split_counts[frame['split'] + '_frames'] += 1
                    category_count['frames'] += 1
                    entry = (-int(digest(frame['frame_id']), 16), frame['frame_id'], frame)
                    if len(heap) < 50:
                        heapq.heappush(heap, entry)
                    elif entry > heap[0]:
                        heapq.heapreplace(heap, entry)
            except (ValueError, KeyError, OSError) as exc:
                counts['invalid_cache_records'] += 1
                errors.write(json.dumps({'record_path': str(path), 'stage': 'rebuild_index',
                                        'error': str(exc)}, allow_nan=False) + '\n')
    for name in ('crop_index.csv', 'video_errors.jsonl', 'warnings.jsonl'):
        (out / (name + '.tmp')).replace(out / name)
    for key in ('videos_in_cache', 'completed_videos', 'extracted_frames', 'incomplete_videos',
                'video_errors', 'invalid_cache_records', 'warnings', 'png_bytes'):
        counts[key] += 0
    summary = {**dict(counts), 'png_gib': counts['png_bytes'] / 1024**3,
               'counts': dict(split_counts), 'by_split_and_modality': categories,
               'metadata_errors': metadata_error_count, 'metadata_videos': len(metas),
               'index': 'crop_index.csv', 'preprocessing': 'original_frame_no_crop',
               'note': 'auto_accepted means decoded/saved, NOT verified face quality. '
                       'Use build_manifest.py for temporal labels and annotation checks.'}
    write_audit(out, [r[2] for r in sorted(heap, reverse=True)])
    write_json(out / 'summary.json', summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--metadata', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--splits', nargs='+', choices=['train', 'dev', 'test'], default=['train', 'dev'])
    p.add_argument('--sample-fps', type=float, default=10.0, help='0 retains every decoded frame.')
    p.add_argument('--limit-per-split', type=int, default=0, help='0 means all videos in each requested split.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--png-compression', type=int, default=3)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--check-install', action='store_true')
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--ffprobe', default='ffprobe')
    args = p.parse_args()
    try:
        if (not math.isfinite(args.sample_fps) or not 0 <= args.sample_fps <= 240
                or not 0 <= args.png_compression <= 9 or args.limit_per_split < 0
                or not math.isfinite(args.timeout) or args.timeout <= 0):
            raise ValueError('Invalid sample-fps, PNG compression, video limit, or timeout.')
        env = {'numpy': np.__version__, 'opencv': cv2.__version__}
        for name in ('ffmpeg', 'ffprobe'):
            exe = getattr(args, name)
            if shutil.which(exe) is None:
                raise ValueError(f'Missing {name}: install the executable or pass --{name} /full/path.')
            env[name] = run_command([exe, '-version'], 30).splitlines()[0]
        if args.check_install:
            print(json.dumps(env, indent=2))
            print('No dlib, landmark predictor or GPU is required.')
            return
        if args.data_root is None or args.out is None:
            p.error('--data-root and --out are required.')
        data_root, out = args.data_root.expanduser().resolve(), args.out.expanduser().resolve()
        if not data_root.is_dir():
            raise ValueError(f'Dataset folder does not exist: {data_root}')
        if out == data_root or data_root.is_relative_to(out) or out.is_relative_to(data_root):
            raise ValueError('Keep --out separate from the dataset root and its ancestors/subfolders.')
        metadata = args.metadata.expanduser().resolve() if args.metadata else data_root / 'metadata.min.json'
        if args.metadata is None and not metadata.is_file():
            metadata = data_root / 'metadata.json'
        if not metadata.is_file():
            raise ValueError(f'Metadata not found: {metadata}. Supply --metadata /actual/file.json.')
        metas, metadata_errors = load_metadata(metadata)
        config = {'sample_fps': args.sample_fps, 'png_compression': args.png_compression}
        provenance = {'version': VERSION, 'preprocessing': 'original_frame_no_crop',
            'config': config, 'environment': env, 'source_root': str(data_root),
            'metadata_path': str(metadata), 'script_sha256': sha256_file(Path(__file__)),
            'time_origin': 'first_decoded_video_frame', 'output_pixels': 'native_size_bgr24_png',
            'sampling': 'first_decoded_frame_on_or_after_regular_grid_no_duplicates',
            'rotation_policy': 'reject_nonzero_display_rotation'}
        config_id = digest(provenance)
        if out.exists() and any(out.iterdir()) and not (out / 'config.json').exists():
            raise ValueError('Output is nonempty but is not an extractor cache; choose a new --out.')
        if (out / 'config.json').is_file():
            old = json.loads((out / 'config.json').read_text(encoding='utf-8'))
            if old.get('config_id') != config_id:
                raise ValueError('Code/settings/environment/source changed, or this is a face-crop '
                                 'cache. Use a NEW --out; do not mix preprocessing pipelines.')
        interrupted = False
        with output_lock(out):
            write_json(out / 'config.json', {**provenance, 'config_id': config_id})
            write_jsonl(out / 'metadata_errors.jsonl', metadata_errors)
            selected = []
            for split in dict.fromkeys(args.splits):
                pool = sorted([m for m in metas if m['split'] == split], key=lambda m: m['file'])
                random.Random(args.seed).shuffle(pool)
                selected.extend(pool[:args.limit_per_split] if args.limit_per_split else pool)
            outcomes = Counter()
            try:
                for meta in tqdm(selected, desc='Original frames (no crop)', unit='video'):
                    outcome = process_clip(meta, data_root, out, config, config_id,
                        args.ffmpeg, args.ffprobe, args.timeout, args.retry_errors)
                    outcomes[outcome] += 1
            except KeyboardInterrupt:
                interrupted = True
                print('\nInterrupted. Saving indexes for completed videos; '
                      'the unfinished video restarts next time.', flush=True)
            finally:
                summary = rebuild_indexes(out, metas, data_root, config_id, len(metadata_errors))
                write_json(out / 'last_run.json', {'utc': datetime.now(timezone.utc).isoformat(),
                    'selected_this_run': len(selected), 'outcomes': dict(outcomes),
                    'interrupted': interrupted})
                print(json.dumps(summary, indent=2))
                print(f"Index: {out / 'crop_index.csv'}")
                print(f"Audit: {out / 'audit_sample.html'}")
        if interrupted:
            raise SystemExit(130)
        if summary['completed_videos'] == 0:
            raise ValueError('No completed videos; inspect video_errors.jsonl and metadata_errors.jsonl.')
    except (ValueError, RuntimeError, OSError) as exc:
        p.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
