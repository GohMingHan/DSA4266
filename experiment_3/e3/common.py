from __future__ import annotations
import contextlib
import os
import random
from pathlib import Path
import numpy as np
import torch
from e2b.io import (read_json, read_jsonl, write_json, write_jsonl, write_npz,
                    digest, sha256, under, require_new, lock, resolve_path)
from . import VERSION

TARGET = 'modify_audio OR modify_video'

def source_hashes():
    base = Path(__file__).resolve().parents[1]
    paths = sorted((base/'e3').glob('*.py')) + sorted((base/'e2b').glob('*.py'))
    paths += [base/'pipeline.py']
    return {p.relative_to(base).as_posix(): sha256(p) for p in paths}

def load_config(path):
    p = Path(path).expanduser().resolve()
    cfg = read_json(p)
    if cfg.get('format') != VERSION:
        raise ValueError('Use the Experiment 3 config.json, with format=experiment_3_v1.')
    return {**cfg, 'experiment_2b_config': str(resolve_path(cfg['experiment_2b_config'], p.parent))}

def device_of(value):
    if value == 'auto': value = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if value == 'cuda': value = 'cuda:0'
    d = torch.device(value)
    if d.type not in ('cuda','cpu'): raise ValueError('Use cpu or cuda:0.')
    if d.type == 'cuda':
        if not torch.cuda.is_available(): raise ValueError('CUDA unavailable. Enable a GPU or explicitly use --device cpu.')
        if d.index is not None and d.index >= torch.cuda.device_count(): raise ValueError('CUDA index unavailable.')
    return d

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    value = torch.initial_seed() % (2**32)
    random.seed(value); np.random.seed(value)

def rng_get():
    return {'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def rng_set(state):
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])

def atomic_save(path, state):
    path = Path(path); tmp = path.with_name(path.name+'.tmp')
    torch.save(state, tmp); tmp.replace(path)

def backend_context(name):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    choices = {'math': [SDPBackend.MATH],
               'flash-math': [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]}
    if name not in choices: raise ValueError('Unsupported attention policy.')
    # Active through forward AND backward. cuDNN/memory-efficient are not enabled.
    return sdpa_kernel(choices[name])

def synchronize(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)

def assert_separate_output(out, protected):
    out = Path(out).expanduser().resolve()
    for value in protected:
        p = Path(value).expanduser().resolve()
        if p.is_file(): p = p.parent
        if out == p or out.is_relative_to(p) or p.is_relative_to(out):
            raise ValueError(f'Output {out} must be separate from input/cache {p}.')
    return out
