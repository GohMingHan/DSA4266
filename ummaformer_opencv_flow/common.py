"""Small, dependency-light helpers for the LAV-DF UMMAFormer experiment."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

VERSION = '2.0.0-opencv-flow'
SPLITS = ('train', 'dev', 'test')


def json_read(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def json_write(path: Path, obj: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def jsonl_write(path: Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as out:
        for row in rows:
            out.write(json.dumps(row, allow_nan=False) + '\n')
    os.replace(tmp, path)


def digest(obj: Any):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256(path: Path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative_file(s: str):
    if not isinstance(s, str) or not s.strip():
        raise ValueError('Expected a nonempty relative file name.')
    p = PurePosixPath(s.replace('\\', '/'))
    if p.is_absolute() or '..' in p.parts or ':' in str(p) or '\x00' in str(p):
        raise ValueError(f'Unsafe relative path: {s!r}')
    return p.as_posix()


def within(root: Path, rel: str):
    root = root.resolve()
    p = (root / relative_file(rel)).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f'Path escapes root: {rel}')
    return p


def file_stat(path: Path):
    s = path.stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def require_unchanged(path: Path, expected: dict):
    if file_stat(path) != expected:
        raise ValueError(f'File changed since preparation: {path}. Prepare a NEW experiment directory.')


def strict_bool(v):
    if isinstance(v, bool) or type(v) is int and v in (0, 1):
        return bool(v)
    raise ValueError(f'Expected bool or 0/1, got {v!r}')
