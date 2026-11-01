"""Exact decoded PTS and sequential RGB decoding; no assumed FPS or silent autorotation."""
from __future__ import annotations
from contextlib import contextmanager
import json
import math
from pathlib import Path
import subprocess
import tempfile
import threading
import numpy as np


def tool_version(name):
    r = subprocess.run([name, '-version'], capture_output=True, text=True, timeout=30, check=True)
    return r.stdout.splitlines()[0]


def probe_video(path: Path, ffprobe='ffprobe', timeout=600):
    # Read complete stream metadata, including side_data_list. No version-specific
    # stream_side_data= selector, and no escaped colon literals in subprocess argv.
    r = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_streams', '-of', 'json', str(path)],
        capture_output=True, text=True, timeout=timeout, encoding='utf-8', errors='replace')
    if r.returncode or r.stderr.strip():
        raise RuntimeError(f'FFprobe failed: {r.stderr[-2000:]}')
    result = json.loads(r.stdout)
    frames = result.get('frames', [])
    streams = result.get('streams', [])
    if not frames or not streams:
        raise ValueError('No decoded video frames/stream.')
    stream = streams[0]
    rotations = [float(s['rotation']) for s in stream.get('side_data_list', []) if 'rotation' in s]
    if 'rotate' in stream.get('tags', {}):
        rotations.append(float(stream['tags']['rotate']))
    if any(not math.isfinite(v) or abs((v + 180) % 360 - 180) > 1e-6 for v in rotations):
        raise ValueError('Nonzero display rotation; handle geometry explicitly, do not silently rotate.')
    if stream.get('sample_aspect_ratio') not in (None, '1:1', '0:1', 'N/A'):
        raise ValueError('Non-square pixel aspect ratio requires explicit geometry handling.')
    absolute, sizes = [], set()
    for f in frames:
        value = f.get('best_effort_timestamp_time', f.get('pts_time'))
        if value is None:
            raise ValueError('Missing decoded timestamp; no frame-index/FPS fallback.')
        absolute.append(float(value))
        sizes.add((int(f['width']), int(f['height'])))
    times = np.array(absolute, dtype=np.float64)
    if len(times) < 2 or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('Require >=2 frames with finite strictly increasing timestamps.')
    if len(sizes) != 1:
        raise ValueError('Changing frame geometry is unsupported.')
    width, height = sizes.pop()
    if min(width, height) < 1:
        raise ValueError('Invalid decoded frame size.')
    tail = float(frames[-1].get('duration_time', frames[-1].get('pkt_duration_time', 0)) or 0)
    estimated = not math.isfinite(tail) or tail <= 0
    if estimated:
        tail = float(np.median(np.diff(times)))
    pts = times - times[0]
    return dict(pts=pts, first_video_pts_s=float(times[0]), duration_s=float(pts[-1] + tail),
                tail_duration_estimated=estimated, width=width, height=height,
                decoded_frame_count=len(times))


def sample_indices(pts, rate):
    pts = np.asarray(pts, np.float64)
    if rate == 0:
        return np.arange(len(pts), dtype=np.int64)
    grid = np.arange(int(np.floor(pts[-1] * rate + 1e-8)) + 1, dtype=np.float64) / rate
    idx = np.searchsorted(pts, grid - 1e-8, side='left')
    return np.unique(idx[idx < len(pts)])


def read_exact(handle, size):
    buf = bytearray()
    while len(buf) < size:
        part = handle.read(size - len(buf))
        if not part:
            break
        buf.extend(part)
    return bytes(buf)


@contextmanager
def decoded_rgb(path, timing, ffmpeg='ffmpeg', timeout=600):
    """Yield an iterator; normal exhaustion validates count, return code and stderr."""
    cmd = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror',
           '-threads', '1', '-noautorotate', '-i', str(path), '-map', '0:v:0',
           '-an', '-sn', '-dn', '-vsync', '0', '-threads', '1',
           '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
    with tempfile.TemporaryFile() as errors:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors)
        expired = threading.Event()
        def kill():
            expired.set()
            if p.poll() is None:
                p.kill()
        timer = threading.Timer(timeout, kill)
        timer.daemon = True
        timer.start()
        def iterate():
            nbytes = timing['width'] * timing['height'] * 3
            for i in range(timing['decoded_frame_count']):
                raw = read_exact(p.stdout, nbytes)
                if len(raw) != nbytes:
                    errors.seek(0)
                    raise RuntimeError(f'Incomplete decode at {i}; timeout={expired.is_set()}; '
                                       + errors.read().decode(errors='replace')[-1500:])
                yield i, np.frombuffer(raw, np.uint8).reshape(timing['height'], timing['width'], 3)
            if p.stdout.read(1):
                raise ValueError('Decoder produced more frames than FFprobe; refusing guessed timing.')
            code = p.wait(timeout=30)
            errors.seek(0)
            message = errors.read().decode(errors='replace')
            if code or expired.is_set() or message.strip():
                raise RuntimeError(f'FFmpeg decode error ({code}): {message[-1500:]}')
        try:
            yield iterate()
        finally:
            timer.cancel()
            if p.poll() is None:
                p.kill()
            p.wait()
            if p.stdout:
                p.stdout.close()
