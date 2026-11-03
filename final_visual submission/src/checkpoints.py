"""Atomic model state, strict resume validation and deterministic state restoration."""

import hashlib
import math
import random
import shutil
import numpy as np
import torch
from pathlib import Path
from .data import PROTOCOL


RUNTIME_MUTABLE = {'epochs', 'workers', 'prefetch_factor', 'cache_mib', 'loader_timeout_seconds',
                   'train_progress_every', 'dev_progress_every', 'heartbeat_seconds',
                   'max_consecutive_gradient_skips', 'max_gradient_skips_per_epoch'}


def saved_config_value(saved, key):
    # Checkpoints predating the optional selector used video ROC-AUC exclusively.
    return saved.get(key, {'checkpoint_selection':'video_roc_auc', 'training_pool':'legacy_full_index'}.get(key))


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (state[0], state[1].tolist(), state[2], state[3], state[4]),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    s = state['numpy']; np.random.set_state((s[0], np.array(s[1], dtype=np.uint32), s[2], s[3], s[4]))
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def assert_finite_state(value, name):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f'Checkpoint contains non-finite tensor: {name}; not safe to resume')
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite_state(item, f'{name}.{key}')
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            assert_finite_state(item, f'{name}[{i}]')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'Checkpoint contains non-finite value: {name}; not safe to resume')


def load_resume_bundle(source, config, index_hash, expected_completed_epoch=None, *, pool_hash=None, protocol=PROTOCOL):
    """Validate only the explicitly supplied checkpoints, never search old runs."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f'Attach the saved run and set RESUME_FROM to its last.pt: {source}')
    state = torch.load(source, map_location='cpu', weights_only=True)
    required = {'protocol', 'config', 'index_sha256', 'epoch', 'history', 'model', 'optimizer',
                'scheduler', 'scaler', 'rng', 'sampler', 'best', 'stale', 'best_epoch', 'best_balanced_dev_loss'}
    if not required <= state.keys() or state['protocol'] != protocol or state['index_sha256'] != index_hash:
        raise ValueError('Resume requires a complete v2 checkpoint and the exact saved candidate index')
    if state.get('pool_sha256') != pool_hash:
        raise ValueError('Resume requires the exact immutable TRAIN pool manifest')
    saved = state['config']
    for key, value in config.as_dict().items():
        if key not in RUNTIME_MUTABLE and key != 'amp' and saved_config_value(saved, key) != value:
            raise ValueError(f'Resume configuration mismatch: {key}: saved={saved.get(key)} requested={value}')
    if not saved['amp'] and config.amp:
        raise ValueError('This recovery path allows unchanged precision or AMP -> FP32, not FP32 -> AMP')
    completed = state['epoch']+1
    if expected_completed_epoch is not None and completed != expected_completed_epoch:
        raise ValueError(f'Expected completed Epoch {expected_completed_epoch}, found Epoch {completed}; refusing wrong checkpoint')
    if completed < 1 or [row['epoch'] for row in state['history']] != list(range(1, completed+1)):
        raise ValueError('Saved history does not match checkpoint epoch')
    if state['sampler']['epoch'] != state['epoch']:
        raise ValueError('Sampler coverage state does not match checkpoint epoch')
    if completed >= config.epochs or state['stale'] >= config.patience:
        raise ValueError('Checkpoint has reached epoch cap or early stopping; no automatic restart')
    if not math.isfinite(state['best']) or not 1 <= state['best_epoch'] <= completed:
        raise ValueError('Invalid best-checkpoint selection state')
    assert_finite_state(state['model'], 'last.model')
    assert_finite_state(state['optimizer'], 'last.optimizer')
    if saved['amp']:
        scale = state['scaler'].get('scale')
        if scale is None or not math.isfinite(scale) or scale <= 0:
            raise ValueError('AMP source checkpoint has no valid saved GradScaler scale')
    best_source = source.parent/'best.pt'
    if not best_source.is_file():
        raise FileNotFoundError(f'Keep the matching best.pt beside last.pt: {best_source}')
    prior_best = torch.load(best_source, map_location='cpu', weights_only=True)
    if (prior_best.get('protocol'), prior_best.get('index_sha256'), prior_best.get('config', {}).get('model'),
        prior_best.get('best'), prior_best.get('epoch', -1)+1) != (
        protocol, index_hash, config.model, state['best'], state['best_epoch']):
        raise ValueError('Companion best.pt does not match saved selection state')
    if prior_best.get('pool_sha256') != pool_hash:
        raise ValueError('Companion best.pt pool identity differs')
    assert_finite_state(prior_best['model'], 'best.model')
    for key, value in config.as_dict().items():
        if key not in RUNTIME_MUTABLE and key != 'amp' and saved_config_value(prior_best['config'], key) != value:
            raise ValueError(f'Companion best.pt configuration mismatch: {key}')
    precision_changed = bool(saved['amp']) != bool(config.amp)
    report = {'source_last': str(source), 'source_best': str(best_source), 'model': config.model,
        'completed_epoch': completed, 'next_epoch': completed+1, 'best_epoch': state['best_epoch'],
        'best_dev_video_roc_auc': state['history'][state['best_epoch']-1]['dev_video']['roc_auc']
            if 'dev_video' in state['history'][state['best_epoch']-1] else (state['best'] if config.checkpoint_selection == 'video_roc_auc' else None),
        'checkpoint_selection': config.checkpoint_selection, 'best_selection_score': state['best'], 'stale_epochs': state['stale'],
        'saved_amp': saved['amp'], 'requested_amp': config.amp, 'precision_changed': precision_changed,
        'saved_amp_scale': state['scaler'].get('scale'),
        'scaler_action': 'restored' if config.amp else 'disabled; saved AMP state retained as provenance',
        'saved_last_dev_frame_roc_auc': state['history'][-1]['dev']['roc_auc'],
        'numerics_counters_in_source': 'numerics' in state,
        'source_checkpoints_read_only': True}
    return {'state': state, 'best_state': prior_best, 'report': report}


def restore_training_state(bundle, model, optimizer, scheduler, scaler, sampler):
    state = bundle['state']
    if scaler.is_enabled() != bundle['report']['requested_amp']:
        raise ValueError('Active scaler precision differs from resume preflight')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    if scaler.is_enabled():
        scaler.load_state_dict(state['scaler'])
    sampler.load_state_dict(state['sampler'])
    restore_rng(state['rng'])


def copy_resume_boundary(bundle, output):
    """Create independent boundary copies; never write to source/existing files."""
    output = Path(output)
    if output.resolve() == Path(bundle['report']['source_last']).parent.resolve():
        raise FileExistsError('Resume output must differ from checkpoint source directory')
    if (output/'best.pt').exists() or (output/'last.pt').exists():
        raise FileExistsError('Resume output already contains checkpoints; choose a new directory')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle['report']['source_best'], output/'best.pt')
    shutil.copy2(bundle['report']['source_last'], output/'last.pt')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_checkpoint(path, payload):
    path = Path(path); temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary); temporary.replace(path)


def bind_resume_identity(output):
    """Only rebind metadata in NEW boundary copies; never touch original weights/files."""
    import json
    output=Path(output)
    identity=json.loads((output/'experiment_identity.json').read_text())
    for name in ('best.pt','last.pt'):
        path=output/name
        state=torch.load(path,map_location='cpu',weights_only=True)
        prior=state.get('experiment_identity',{})
        if prior.get('project_id')!=identity['project_id'] or prior.get('model_id')!=identity['model_id']:
            raise ValueError('Cannot bind a foreign checkpoint to this resume segment')
        state['experiment_identity']=identity
        atomic_checkpoint(path,state)
