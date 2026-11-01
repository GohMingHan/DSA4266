"""Small, strict I/O helpers. Existing datasets, checkpoints and sources are read-only."""
from __future__ import annotations
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any
import numpy as np

VERSION = 'experiment_2b_v1'
SPLITS = ('train', 'dev', 'test')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                      separators=(',', ':')).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def write_jsonl(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def write_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def resolve_path(value, base):
    if not isinstance(value, str) or not value.strip() or 'CHANGE_ME' in value:
        raise ValueError(f'Configure a real path, not {value!r}.')
    p = Path(os.path.expandvars(value)).expanduser()
    return (p if p.is_absolute() else Path(base) / p).resolve()


def relative_name(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Expected a nonempty metadata file identifier.')
    v = value.replace('\\', '/')
    p = PurePosixPath(v)
    if p.is_absolute() or '..' in p.parts or ':' in v or '\x00' in v or not p.parts:
        raise ValueError(f'Unsafe relative identifier: {value!r}')
    return p.as_posix()


def under(root, name):
    root = Path(root).resolve(); p = (root / relative_name(name)).resolve()
    if not p.is_relative_to(root):
        raise ValueError('A feature path escapes its root (including symlinks).')
    return p


def file_key(name):
    return hashlib.sha256(relative_name(name).encode()).hexdigest()[:24]


def flag(value):
    if isinstance(value, bool) or type(value) is int and value in (0, 1):
        return int(value)
    raise ValueError(f'Expected boolean or integer 0/1, not {value!r}')


def require_new(path):
    p = Path(path)
    if p.exists() and (not p.is_dir() or any(p.iterdir())):
        raise ValueError(f'Output is not empty: {p}. Use a new folder; nothing was overwritten.')
    p.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def lock(folder, name='.e2b.lock'):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    try:
        with path.open('x', encoding='utf-8') as f:
            json.dump({'pid': os.getpid()}, f)
    except FileExistsError:
        raise ValueError(f'{path} exists. Do not run concurrent writers. Remove only a confirmed stale lock.') from None
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def code_hashes():
    base = Path(__file__).resolve().parent
    return {p.name: sha256(p) for p in sorted(base.glob('*.py'))}


def canonical_cohort(path):
    """Accept official metadata JSON list or the earlier visual videos.jsonl.

    Never infer modality flags from a class folder, model score or global label.
    Preserve original split assignment. Union linked original/source families to
    check leakage; this is not an independent identity reconstruction.
    """
    path = Path(path)
    raw = read_jsonl(path) if path.suffix.lower() == '.jsonl' else read_json(path)
    if isinstance(raw, dict) and isinstance(raw.get('videos'), list):
        raw = raw['videos']
    if not isinstance(raw, list) or not raw:
        raise ValueError('Cohort must be nonempty JSON records or videos.jsonl.')
    result = []; seen = set(); parent = {}; splits_by_group = {}
    def find(s):
        parent.setdefault(s, s)
        while parent[s] != s:
            parent[s] = parent[parent[s]]; s = parent[s]
        return s
    def join(a,b):
        ra,rb=find(a),find(b)
        if ra != rb: parent[max(ra,rb)] = min(ra,rb)
    for r in raw:
        name = relative_name(r['file'])
        if name in seen: raise ValueError(f'Duplicate cohort video: {name}')
        seen.add(name)
        split = r['split']
        if split not in SPLITS: raise ValueError(f'Unexpected split {split!r}')
        ma,mv = flag(r['modify_audio']),flag(r['modify_video'])
        original = relative_name(r['original']) if r.get('original') else name
        # Prefer explicit source_group only for grouping; do not use it as pair ID.
        group = relative_name(r['source_group']) if r.get('source_group') else original
        join(name,original); join(name,group)
        result.append({'file':name,'split':split,'modify_audio':ma,'modify_video':mv,
                       'label':int(ma or mv),'source_group':group,
                       'category':'both' if ma and mv else 'audio_only' if ma else 'visual_only' if mv else 'real'})
    for r in result:
        root = find(r['file']); r['source_family'] = root
        if root in splits_by_group and splits_by_group[root] != r['split']:
            raise ValueError(f'Source family crosses splits: {root}. Do not automatically resplit frames.')
        splits_by_group[root]=r['split']
    return sorted(result,key=lambda r:(SPLITS.index(r['split']),r['file']))


def check_expected(rows, splits, expected):
    for split in splits:
        n=sum(r['split']==split for r in rows)
        if not n: raise ValueError(f'No cohort videos in requested split {split}.')
        if expected and split in expected and expected[split] is not None and n != expected[split]:
            raise ValueError(f'{split}: expected {expected[split]} videos, cohort contains {n}. Audit the cohort or explicitly change expected_counts.')


def load_config(path):
    path = Path(path).expanduser().resolve(); c=read_json(path)
    if c.get('format') != VERSION: raise ValueError('Unexpected Experiment 2B config format.')
    c['_path']=str(path); c['_base']=str(path.parent)
    for field in ('cohort','visual_slot','audio_slot','visual_embeddings','audio_embeddings'):
        c[field]=str(resolve_path(c[field],path.parent))
    return c


def load_slot(path):
    path=Path(path).expanduser().resolve(); s=read_json(path)
    kind=s.get('kind','unconfigured')
    if kind == 'unconfigured':
        raise ValueError(f'{path}: choose a backend from examples/ and place the selected trained checkpoint here. No random encoder will be used.')
    if kind not in ('experiment_2a','forensics_adapter','ummaformer','preexported','custom'):
        raise ValueError(f'Unsupported kind {kind!r}. Use a known adapter or documented preexported embeddings.')
    s['_path']=str(path); s['_base']=str(path.parent)
    s['checkpoint']=str(resolve_path(s['checkpoint'],path.parent))
    if not Path(s['checkpoint']).is_file():
        raise FileNotFoundError(f'Place the selected trained checkpoint at {s["checkpoint"]}.')
    for k in ('project_dir','input_root','input_manifest','upstream','clip_weights','crops_root','aasist_dir','index','adapter_file'):
        if s.get(k) is not None: s[k]=str(resolve_path(s[k],path.parent))
    if s.get('time_origin') != 'first_decoded_video_frame':
        raise ValueError('Explicitly declare time_origin=first_decoded_video_frame after verifying your exporter timeline. No guessed audio offset.')
    if s.get('batch_size',4)<1: raise ValueError('Encoder batch_size must be positive.')
    return s


def check_sequence(arrays, modality):
    """Return numeric embeddings and their source time, never diagnostic scores."""
    h=np.asarray(arrays['features'])
    if h.ndim!=2 or h.shape[0]<1 or h.shape[1]<3 or h.dtype.kind not in 'fi' or not np.isfinite(h).all():
        raise ValueError('features must be a finite numeric [T,D] array, T>=1 and D>=3. Do not provide probabilities/class logits as embeddings.')
    h=np.asarray(h,dtype=np.float32)
    if not np.isfinite(h).all(): raise ValueError('Features overflow float32.')
    n=len(h)
    out={'features':h}
    if modality=='audio':
        for k in ('start_s','end_s','center_s'):
            t=np.asarray(arrays[k],np.float64)
            if t.shape!=(n,) or not np.isfinite(t).all(): raise ValueError(f'Invalid {k}')
            out[k]=t
        if (out['start_s'] < -1e-6).any() or (out['end_s']<=out['start_s']).any():
            raise ValueError('Audio windows must have valid original video-relative intervals.')
        if ((out['center_s']<out['start_s']) | (out['center_s']>out['end_s'])).any():
            raise ValueError('Audio centers lie outside source windows.')
        times=out['center_s']
        if 'padding_fraction' in arrays:
            pad=np.asarray(arrays['padding_fraction'],np.float32)
            if pad.shape!=(n,) or not np.isfinite(pad).all() or ((pad<0)|(pad>=1)).any():
                raise ValueError('Invalid audio repetition/padding fraction.')
            out['padding_fraction']=pad
    else:
        times=np.asarray(arrays['timestamp_s'],np.float64)
        if times.shape!=(n,) or not np.isfinite(times).all() or (times < -1e-6).any():
            raise ValueError('Invalid visual timestamps.')
        out['timestamp_s']=times
    if (np.diff(times)<=0).any(): raise ValueError('Time coordinates must be strictly increasing, not re-numbered after failures.')
    return out
