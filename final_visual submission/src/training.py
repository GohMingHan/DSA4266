"""Training/evaluation loop, live progress, AMP safeguards and epoch persistence."""

import contextlib
import gc
import json
import math
import os
import random
import shutil
import threading
import time
import traceback
import numpy as np
import torch
from datetime import datetime, timezone
from pathlib import Path
from torch import nn
from .data import make_loader, PROTOCOL, write_json, memory_mib, cnn_inputs, cnn_records
from .models import StagedCNN, make_optimizer
from .config import checkpoint_score, selection_label
from .backup import package_model_run
from .evaluation import (
    binary_metrics,
    class_loss_metrics,
    evaluate_predictions,
    write_history_metrics,
    plot_history,
    prediction_artifacts,
    write_final_numerics,
    dev_threshold_analysis,
)
from .checkpoints import (
    load_resume_bundle,
    restore_training_state,
    copy_resume_boundary,
    rng_state,
    restore_rng,
    digest,
    atomic_checkpoint,
)


LOGGING_VERSION = 'cnn_v2_amp_safe_resume_v1'


def model_label(name):
    return {'resnet50': 'ResNet50', 'efficientnet_b0': 'EfficientNet-B0'}.get(name, name)


def should_log_batch(batch, total, interval):
    return batch == 1 or batch == total or batch % interval == 0


class RunProgress:
    """Only the parent process writes run logs; heartbeat never touches CUDA."""
    def __init__(self, output, model, heartbeat_seconds=30):
        self.output = Path(output)
        self.model = model_label(model)
        self.heartbeat_seconds = heartbeat_seconds
        self.state = {'model': model, 'logging_version': LOGGING_VERSION, 'stage': 'starting'}
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.changed_at = time.monotonic()

    def _persist(self):
        temporary = self.output/'run_status.tmp'
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        temporary.replace(self.output/'run_status.json')

    def note(self, message):
        with self.lock:
            print(message, flush=True)
            event = {'time_utc': datetime.now(timezone.utc).isoformat(), **self.state, 'message': message}
            with (self.output/'progress.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(event, allow_nan=False)+'\n'); handle.flush()

    def numerics(self, event):
        with self.lock:
            with (self.output/'numerics.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'time_utc': datetime.now(timezone.utc).isoformat(), **event}, allow_nan=False)+'\n')
                handle.flush()

    def stage(self, name, *, announce=False, persist=True, **details):
        with self.lock:
            self.state.update(stage=name, **details)
            self.state['updated_utc'] = datetime.now(timezone.utc).isoformat()
            self.changed_at = time.monotonic()
            if persist:
                self._persist()
            if announce:
                visible = {k: v for k, v in details.items() if k not in ('candidate_ids', 'sources')}
                if 'candidate_ids' in details:
                    visible['candidate_count'] = len(details['candidate_ids'])
                if 'sources' in details:
                    visible['video_count'] = len(details['sources'])
                self.note(f'{self.model} | {name} | ' + json.dumps(visible))

    def _heartbeat(self):
        while not self.stop.wait(self.heartbeat_seconds):
            with self.lock:
                self._persist()
                elapsed = time.monotonic()-self.changed_at
                self.note(f"{self.model} | heartbeat | stage={self.state['stage']} | "
                          f"epoch={self.state.get('epoch', '?')} batch={self.state.get('batch', '?')} | "
                          f'{elapsed:.0f}s in current operation')

    def __enter__(self):
        self.stage('starting', announce=True)
        if self.heartbeat_seconds > 0:
            self.thread = threading.Thread(target=self._heartbeat, daemon=True, name='cnn-progress')
            self.thread.start()
        return self

    def __exit__(self, error_type, error, tb):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        if error is not None:
            # Never replace or suppress the original error if diagnostic I/O fails.
            text = ''.join(traceback.format_exception(error_type, error, tb))
            failed_at = dict(self.state)
            print(f"\n{self.model} FAILED at {failed_at.get('stage')}: {error_type.__name__}: {error}", flush=True)
            print(text, flush=True)
            try:
                (self.output/'failure_traceback.txt').write_text(text, encoding='utf-8')
                self.stage('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                           failed_at=failed_at, error_type=error_type.__name__, error=str(error))
            except Exception as diagnostic_error:
                print(f'Could not persist failure diagnostics: {diagnostic_error}', flush=True)
        else:
            self.stage('completed', announce=True)
        return False


def print_epoch_summary(progress, entry, config, best_epoch):
    train, dev, video = entry['train'], entry['dev'], entry['dev_video']
    unit='clip' if config.model=='resnet50_lstm' else 'frame'
    def number(value):
        return 'undefined' if value is None else f'{value:.6f}'
    lines = [f"\n{progress.model} | Epoch {entry['epoch']}/{config.epochs} completed (TRAIN + DEV)",
        f"  TRAIN loss {number(train['loss'])} | ROC-AUC {number(train['roc_auc'])} | samples {train['n']}",
        f"  TRAIN PR-AUC {number(train.get('pr_auc'))} | precision/recall/F1/F2 {number(train['precision'])}/{number(train['recall'])}/{number(train['f1'])}/{number(train.get('f2'))}",
        f"  DEV loss {number(dev['loss'])} | balanced {number(dev['balanced_loss'])} | "
        f"real {number(dev['real_class_loss'])} | fake {number(dev['fake_class_loss'])}",
        f"  DEV {unit} PR-AUC {number(dev.get('pr_auc'))} | {unit} ROC-AUC {number(dev['roc_auc'])} | video ROC-AUC {number(video['roc_auc'])} | video PR-AUC {number(video.get('pr_auc'))}",
        f"  DEV {unit} precision/recall/F1/F2 {number(dev['precision'])}/{number(dev['recall'])}/{number(dev['f1'])}/{number(dev.get('f2'))}",
        f"  DEV video precision/recall/F1 {number(video['precision'])}/{number(video['recall'])}/{number(video['f1'])}",
        f"  Threshold {config.threshold} | learning rates {entry['learning_rates']} | phase {entry['phase']}",
        f"  Duration {entry['train_seconds']+entry['dev_seconds']:.1f}s (TRAIN {entry['train_seconds']:.1f}s; DEV {entry['dev_seconds']:.1f}s)",
        f"  Best checkpoint {'IMPROVED' if entry['is_best'] else 'unchanged'} (best epoch {best_epoch}; {selection_label(config.checkpoint_selection)}) | "
        f"early stopping {entry['stale_epochs']}/{config.patience}",
        '  Saving history and checkpoints now; confirmed paths follow.']
    if 'numerics' in train:
        n = train['numerics']
        lines.insert(-1, f"  Optimizer updates {n['optimizer_steps']} | skipped {n['skipped_steps']} "
                     '(AMP scales and gradient norms saved in numerics.jsonl / history)')
    progress.note('\n'.join(lines))


NUMERICS_VERSION = 'amp_safe_resume_v1'


class NumericalInstabilityError(FloatingPointError):
    pass


class GradientGuard:
    def __init__(self, config, state=None):
        self.config = config
        self.state = dict(state) if state is not None else {
            'optimizer_steps': 0, 'skipped_steps': 0, 'consecutive_skips': 0,
            'max_consecutive_skips': 0, 'epoch': -1, 'epoch_optimizer_steps': 0,
            'epoch_skipped_steps': 0}

    def state_dict(self):
        return dict(self.state)

    def begin_epoch(self, epoch):
        if epoch != self.state['epoch']:
            self.state.update(epoch=epoch, epoch_optimizer_steps=0, epoch_skipped_steps=0)

    def finish_step(self, optimizer, scaler, *, epoch, batch, emit, execute_step=True):
        """Call once after scaled backward. Never clip non-finite gradients.

        unscale_ records inf/NaN for GradScaler BEFORE any inspection/clipping.
        On AMP overflow, step() must still run: GradScaler skips optimizer.step
        and update() backs off its scale. A pre-hook makes an unexpected actual
        optimizer call in that branch fail before touching weights or moments.
        execute_step=False supports disposable diagnostics without weight updates.
        """
        if scaler.is_enabled() != bool(self.config.amp):
            raise ValueError('GradScaler enabled flag differs from config.amp')
        scale_before = float(scaler.get_scale())
        if not math.isfinite(scale_before) or scale_before <= 0:
            raise NumericalInstabilityError(f'Invalid AMP scale {scale_before}')
        scaler.unscale_(optimizer)
        named = [(f"{group.get('name', gi)}.parameter{pi}", p) for gi, group in enumerate(optimizer.param_groups)
                 for pi, p in enumerate(group['params']) if p.grad is not None]
        if not named:
            raise NumericalInstabilityError('No optimizer gradients after backward')
        if any(p.grad.is_sparse or p.grad.is_complex() for _, p in named):
            raise NumericalInstabilityError('CNN guard expects real dense gradients')
        # CNN parameters/gradients remain FP32 under autocast. Their finite
        # elements can still produce an overflowing FP32 L2 norm. Accumulate
        # these reductions in FP64 rather than treating that as gradient NaN/Inf.
        norms = torch.stack([torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float64) for _, p in named])
        total = torch.linalg.vector_norm(norms)
        finite = bool(torch.isfinite(total))
        norm = float(total) if finite else None
        event = {'version': NUMERICS_VERSION, 'epoch': epoch+1, 'batch': batch,
                 'amp': bool(self.config.amp), 'amp_scale_before': scale_before,
                 'gradient_norm': norm, 'gradient_norm_status': 'finite' if finite else ('nan' if bool(torch.isnan(total)) else 'inf'),
                 'gradient_norm_accumulation': 'float64',
                 'clip_max_norm': 5., 'optimizer_step_applied': False, 'skipped_step': False}
        if not finite:
            bad = [name for name, p in named if not bool(torch.isfinite(p.grad).all())]
            if not bad:
                event.update(reason='nonfinite_norm_with_finite_component_norms', amp_scale_after=scale_before)
                emit(event)
                optimizer.zero_grad(set_to_none=True)
                raise NumericalInstabilityError('Non-finite FP64 gradient norm without bad component norms; no optimizer update')
            self.state['skipped_steps'] += 1
            self.state['epoch_skipped_steps'] += 1
            self.state['consecutive_skips'] += 1
            self.state['max_consecutive_skips'] = max(self.state['max_consecutive_skips'], self.state['consecutive_skips'])
            event.update(skipped_step=True, reason='nonfinite_gradient_elements', bad_parameters=bad)
            if scaler.is_enabled():
                def forbid_update(*args, **kwargs):
                    raise NumericalInstabilityError('GradScaler attempted an optimizer update with non-finite gradients')
                # GradScaler found_inf from unscale_ must skip this call. The
                # hook is a defensive invariant, not an alternative skip path.
                hook = optimizer.register_step_pre_hook(forbid_update)
                try:
                    if execute_step:
                        scaler.step(optimizer)
                    scaler.update()
                finally:
                    hook.remove()
            optimizer.zero_grad(set_to_none=True)
            event.update(amp_scale_after=float(scaler.get_scale()), **{k: v for k, v in self.state.items() if k != 'epoch'})
            emit(event)
            if not scaler.is_enabled():
                raise NumericalInstabilityError('Non-finite FP32 gradients: skipped update and stopped; investigate inputs/operations, not loss scaling')
            if event['amp_scale_after'] >= scale_before or event['amp_scale_after'] <= 0:
                raise NumericalInstabilityError('AMP overflow did not produce a valid lower scale; no optimizer update')
            if (self.state['consecutive_skips'] >= self.config.max_consecutive_gradient_skips or
                self.state['epoch_skipped_steps'] >= self.config.max_gradient_skips_per_epoch):
                raise NumericalInstabilityError('Repeated AMP instability: updates were skipped safely. Resume the last completed epoch with amp=False or investigate the recorded batches.')
            return event
        coefficient = min(1., 5. / (norm + 1e-6))
        with torch.no_grad():
            for _, p in named:
                p.grad.mul_(coefficient)
        # GradScaler checked before clipping. Independently guard the actual
        # post-clip tensors so no new non-finite value can reach optimizer.step.
        if not bool(torch.stack([torch.isfinite(p.grad).all() for _, p in named]).all()):
            event.update(reason='nonfinite_after_clipping', amp_scale_after=scale_before)
            emit(event); optimizer.zero_grad(set_to_none=True)
            raise NumericalInstabilityError('Clipping produced non-finite gradients; no optimizer update')
        if execute_step:
            scaler.step(optimizer)
            self.state['optimizer_steps'] += 1
            self.state['epoch_optimizer_steps'] += 1
        scaler.update()
        self.state['consecutive_skips'] = 0
        event.update(amp_scale_after=float(scaler.get_scale()), clip_coefficient=coefficient,
                     optimizer_step_applied=execute_step, reason='finite', **{k: v for k, v in self.state.items() if k != 'epoch'})
        emit(event)
        return event


def verify_cuda_overflow_skip():
    """Two synthetic CUDA values, injected overflow, zero actual optimizer steps."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA overflow smoke check requires the Kaggle GPU')
    from .config import Config
    parameter = torch.nn.Parameter(torch.ones(2, device='cuda'))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', init_scale=8.)
    guard = GradientGuard(Config()); guard.begin_epoch(0)
    before = parameter.detach().clone()
    scaler.scale(parameter.sum()).backward()
    parameter.grad.fill_(float('inf'))
    events = []
    guard.finish_step(optimizer, scaler, epoch=0, batch=1, emit=events.append)
    if not torch.equal(parameter, before) or optimizer.state or scaler.get_scale() != 4.:
        raise NumericalInstabilityError('CUDA smoke check failed: overflow changed weights/state or did not back off scale')
    return {'version': NUMERICS_VERSION, 'cuda_overflow_skip_verified': True,
            'actual_optimizer_steps': 0, 'event': events[-1]}


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CNN v2 training requires CUDA. Enable a Kaggle GPU; CPU training is disabled.')


def run_epoch(model, loader, config, optimizer=None, scaler=None, *, progress=None, epoch=0, gradient_guard=None):
    training = optimizer is not None
    split = 'train' if training else 'dev'
    label = 'Training' if training else 'DEV evaluation'
    total = len(loader)
    interval = config.train_progress_every if training else config.dev_progress_every
    current_batch = 0
    def stage(operation, **details):
        if current_batch == 1:
            details.update(persist=True, announce=True)
        if progress is not None:
            progress.stage(f'{split}_{operation}', epoch=epoch+1, split=split, **details)
    def note(message):
        if progress is not None:
            progress.note(message)
        else:
            print(message, flush=True)
    finite_norms = []
    last_numerics = None
    if training:
        gradient_guard = gradient_guard or GradientGuard(config)
        gradient_guard.begin_epoch(epoch)
        scale_start = float(scaler.get_scale())
    def numerical_event(event):
        nonlocal last_numerics
        event.update(candidate_ids=[r['id'] for r in expected_records],
                     sources=sorted({r['source_file'] for r in expected_records}), finite_forward=True)
        last_numerics = event
        if event['gradient_norm'] is not None:
            finite_norms.append(event['gradient_norm'])
        if progress is not None:
            progress.numerics(event)
        if event['skipped_step'] or event['reason'] != 'finite':
            note(f"{model_label(config.model)} | Epoch {epoch+1} Batch {event['batch']} | "
                 f"NUMERICAL EVENT: {event['reason']} | norm={event['gradient_norm_status']} | "
                 f"AMP scale {event['amp_scale_before']} -> {event['amp_scale_after']} | "
                 f"skipped={event['skipped_step']} consecutive={event.get('consecutive_skips', 0)} | "
                 f"bad parameters={event.get('bad_parameters', [])}")
    stage('start', announce=True, total_batches=total, workers=loader.num_workers)
    model.train(training)
    labels, scores, predictions = [], [], []
    sums = {0: 0., 1: 0.}; counts = {0: 0, 1: 0}
    started = time.perf_counter()
    stage('iterator_start', batch=0)
    iterator = iter(loader)
    with contextlib.nullcontext() if training else torch.no_grad():
        for batch in range(1, total+1):
            current_batch = batch
            expected_records = loader.dataset.records[(batch-1)*config.batch_size:batch*config.batch_size]
            stage('fetch_batch', batch=batch, candidate_ids=[r['id'] for r in expected_records],
                  sources=sorted({r['source_file'] for r in expected_records}))
            batch_started = time.perf_counter()
            images, targets, indices = next(iterator)
            load_seconds = time.perf_counter()-batch_started
            stage('transfer_to_cuda', persist=False, candidate_ids=[loader.dataset.records[i]['id'] for i in indices.tolist()])
            images = images.to('cuda', non_blocking=True)
            targets = targets.to('cuda', non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            stage('forward_loss', persist=False)
            with torch.autocast('cuda', dtype=torch.float16, enabled=config.amp):
                logits = model(images).squeeze(1)
                losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(losses).all()):
                raise FloatingPointError('Nonfinite forward logits/BCE; no optimizer update')
            if training:
                stage('backward', persist=False)
                scaler.scale(losses.mean()).backward()
                stage('gradient_unscale_check_clip_step', persist=False)
                gradient_guard.finish_step(optimizer, scaler, epoch=epoch, batch=batch, emit=numerical_event)
            stage('collect_metrics', persist=False)
            ys = targets.detach().int().cpu().tolist()
            ps = torch.sigmoid(logits.detach().float()).cpu().tolist()
            ls = losses.detach().float().cpu().tolist()
            for i, y, p, loss in zip(indices.tolist(), ys, ps, ls):
                labels.append(y); scores.append(p); sums[y] += loss; counts[y] += 1
                if not training:
                    predictions.append({'record_index': i, 'target': y, 'probability': p})
            stage('batch_completed', samples_seen=len(labels))
            if should_log_batch(batch, total, interval):
                numerical_text = (f" | Skipped steps {gradient_guard.state['epoch_skipped_steps']}" if training else '')
                note(f'{model_label(config.model)} | Epoch {epoch+1}/{config.epochs} | '
                     f'{label}: Batch {batch}/{total} | Loss: {sum(ls)/len(ls):.4f} | '
                     f'Mean loss: {sum(sums.values())/sum(counts.values()):.4f} | '
                     f'Load {load_seconds:.2f}s | Batch {time.perf_counter()-batch_started:.2f}s' + numerical_text +
                     f' | CUDA allocated/reserved {torch.cuda.memory_allocated()/2**20:.0f}/'
                     f'{torch.cuda.memory_reserved()/2**20:.0f} MiB | parent RSS {memory_mib()} MiB')
    stage('iterator_shutdown')
    del iterator
    stage('compute_epoch_metrics', announce=True)
    if len(labels) != len(loader.dataset):
        raise RuntimeError(f'{split}: saw {len(labels)} of {len(loader.dataset)} expected samples')
    metrics = binary_metrics(labels, scores, config.threshold)
    metrics.update(class_loss_metrics(sums, counts))
    metrics['loss'] = sum(sums.values()) / sum(counts.values())
    if training:
        metrics['numerics'] = {'optimizer_steps': gradient_guard.state['epoch_optimizer_steps'],
            'skipped_steps': gradient_guard.state['epoch_skipped_steps'],
            'total_skipped_steps_since_tracking': gradient_guard.state['skipped_steps'],
            'max_consecutive_skips_since_tracking': gradient_guard.state['max_consecutive_skips'],
            'gradient_norm_mean_finite': sum(finite_norms)/len(finite_norms) if finite_norms else None,
            'gradient_norm_max_finite': max(finite_norms) if finite_norms else None,
            'amp_scale_start': scale_start, 'amp_scale_end': float(scaler.get_scale()),
            'precision': 'amp_fp16' if config.amp else 'fp32'}
    return metrics, predictions, time.perf_counter() - started


def commit_epoch(out, history, entry, payload, improved, records, predictions, report, config, progress, persistent_backup=None, input_pipeline=None):
    """Commit locally, create a downloadable ZIP; remote backup is opt-in. No plots."""
    print_epoch_summary(progress, entry, config, payload['best_epoch'])
    progress.stage('write_history_metrics', announce=True)
    write_history_metrics(out, history)
    progress.note(f'History saved: {out/"history.csv"} | {out/"history.jsonl"}')
    if improved:
        progress.stage('save_best_checkpoint', announce=True)
        atomic_checkpoint(out/'best.pt', payload)
    progress.stage('save_last_checkpoint', announce=True)
    atomic_checkpoint(out/'last.pt', payload)
    progress.note(f'Checkpoint saved: {out/"last.pt"}\n'
                  f'Best checkpoint {"updated" if improved else "retained"}: {out/"best.pt"}')
    if improved:
        progress.stage('write_best_prediction_artifacts', announce=True)
        prediction_artifacts(out, records, predictions, report, render_plots=False,unit_label='Clip' if config.model=='resnet50_lstm' else 'Frame')
    if input_pipeline is not None:
        progress.stage('snapshot_input_pipeline_metadata', announce=True)
        input_pipeline.snapshot(out)
    progress.stage('create_epoch_download_zip', announce=True)
    receipt = package_model_run(out, config.model, entry['epoch'], snapshot_lock=progress.lock)
    progress.note(f'Download ZIP (completed Epoch {entry["epoch"]}): {receipt["archive"]}\n'
                  f'SHA256 {receipt["sha256"]} | {receipt["bytes"]/2**20:.1f} MiB. Download to Windows before expiry.')
    if persistent_backup is not None:
        progress.stage('verify_private_epoch_backup', announce=True)
        backup_rng = rng_state()
        try:
            persistent_backup.save_run(out, config, entry['epoch'], snapshot_lock=progress.lock)
        finally:
            restore_rng(backup_rng)
    progress.stage('epoch_artifacts_completed', announce=True)


def train(config, index_path, data_root, weights_root, output, resume_from=None, *, persistent_backup=None, pool_path=None, input_pipeline=None):
    config.validate()
    if persistent_backup is not None:
        persistent_backup.require_ready(config, digest(index_path))
    require_cuda()
    out = Path(output)
    if out.exists():
        raise FileExistsError(f'Existing run preserved: {out}. Choose a new versioned run; use explicit resume_from.')
    out.mkdir(parents=True)
    with RunProgress(out, config.model, config.heartbeat_seconds) as progress:
        return _train(config, index_path, data_root, weights_root, out, resume_from, progress, persistent_backup, pool_path, input_pipeline)


def checkpoint_dev_report(state, records, policy, kind):
    """Reuse predictions stored with the selected weights; never approximate BCE.

    Older checkpoints without saved predictions require one final inference pass.
    Invalid saved predictions fail explicitly rather than silently being reused.
    """
    predictions = state.get('dev_predictions')
    if predictions is None:
        return None
    if (state['epoch']+1 != state['best_epoch'] or len(predictions) != len(records) or
            any(p['record_index'] != i or p['target'] != records[i]['label']
                for i, p in enumerate(predictions))):
        raise ValueError('Selected checkpoint DEV predictions do not match the fixed schedule')
    report = evaluate_predictions(records, predictions, policy, kind)
    score = checkpoint_score(report, state['config']['checkpoint_selection'])
    if not math.isclose(score, state['best'], rel_tol=0., abs_tol=1e-12):
        raise ValueError('Selected checkpoint DEV predictions disagree with its selection score')
    entry = state['history'][state['best_epoch']-1]
    if entry['epoch'] != state['best_epoch']:
        raise ValueError('Selected checkpoint DEV loss history has a different epoch')
    report['losses'] = {k: entry['dev'][k] for k in ('loss', 'real_class_loss', 'fake_class_loss', 'balanced_loss')}
    if any(v is None or not math.isfinite(v) for v in report['losses'].values()):
        raise ValueError('Selected checkpoint DEV losses are missing or non-finite')
    return predictions, report


def _train(config, index_path, data_root, weights_root, out, resume_from, progress, persistent_backup, pool_path=None, input_pipeline=None):
    loader_factory = make_loader if input_pipeline is None else input_pipeline.make_loader
    for name in ('experiment_identity.json','source_provenance.json'):
        if (out.parent/name).is_file(): shutil.copy2(out.parent/name,out/name)
    if input_pipeline is not None:
        write_json(out/'input_pipeline.json', input_pipeline.identity)
    progress.stage('seed_and_environment', announce=True)
    seed_everything(config.seed)
    environment = {'logging_version': LOGGING_VERSION, 'pid': os.getpid(), 'torch': str(torch.__version__),
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'workers': config.workers,
        'worker_context': 'spawn' if config.workers else 'main_process',
        'worker_timeout_seconds': config.loader_timeout_seconds if config.workers else 0,
        'process_rss_mib': memory_mib(), 'output_disk_free_mib': shutil.disk_usage(out).free/2**20,
        'shared_memory_free_mib': shutil.disk_usage('/dev/shm').free/2**20 if Path('/dev/shm').exists() else None}
    write_json(out/'runtime_environment.json', environment)
    progress.stage('load_candidate_index', announce=True)
    temporal=config.model=='resnet50_lstm'
    protocol=PROTOCOL;kind='frame';feature_source=None
    if temporal:
        from .temporal import ClipIndex, ClipPool, trained_resnet, TEMPORAL_PROTOCOL
        from .data import FixedPoolSampler
        index=ClipIndex(index_path,data_root);pool=None;protocol=TEMPORAL_PROTOCOL;kind='clip'
        sampler=FixedPoolSampler(ClipPool(index),config.clip_per_class,config.seed)
    else:
        index, sampler, pool = cnn_inputs(config, index_path, data_root, pool_path)
    index_hash = digest(index_path)
    pool_hash = pool.sha256 if pool is not None else None
    if pool is not None:
        shutil.copy2(pool.path, out/'train_pool.json')
    if persistent_backup is not None:
        persistent_backup.require_ready(config, index_hash)
    plan_path = out.parent/'experiment_plans.json'
    if plan_path.is_file():
        write_json(out/'experiment_plan.json', json.loads(plan_path.read_text())[config.model])
    if temporal:
        shutil.copy2(index_path,out/'clip_manifest.json')
    dev_records = index.records('dev', kind)
    progress.stage('initialize_dev_loader', announce=True)
    dev_loader = loader_factory(dev_records, data_root, config)
    progress.stage('initialize_cuda_model_optimizer', announce=True)
    if temporal:
        model,feature_source=trained_resnet(config,weights_root)
        model=model.cuda();write_json(out/'feature_source.json',feature_source)
        optimizer=torch.optim.AdamW([{'name':'temporal_head','params':[p for p in model.parameters() if p.requires_grad],
            'lr':config.head_lr}],weight_decay=config.weight_decay)
    else:
        model = StagedCNN(config, weights_root, pretrained=resume_from is None).cuda()
        optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=config.scheduler_patience)
    scaler = torch.amp.GradScaler('cuda', enabled=config.amp)
    history = []; start = 0; best = -1.; stale = 0; best_epoch = 0; best_balanced = None
    status = 'trained'
    gradient_guard = GradientGuard(config)
    inactive_amp_scaler_state = {}
    if resume_from is not None:
        progress.stage('validate_and_restore_checkpoint', announce=True)
        bundle = load_resume_bundle(resume_from, config, index_hash, pool_hash=pool_hash,protocol=protocol)
        state = bundle['state']
        expected_pipeline = None if input_pipeline is None else input_pipeline.identity
        if any(saved.get('input_pipeline') != expected_pipeline for saved in (state, bundle['best_state'])):
            raise ValueError('Resume input preprocessing differs; face/full-frame checkpoints cannot be interchanged')
        if temporal and state.get('feature_source')!=feature_source:
            raise ValueError('Temporal resume requires the identical selected trained CNN and feature configuration')
        start = state['epoch'] + 1; best = state['best']; stale = state['stale']
        history = state['history']; best_epoch = state['best_epoch']; best_balanced = state['best_balanced_dev_loss']
        restore_training_state(bundle, model, optimizer, scheduler, scaler, sampler)
        gradient_guard = GradientGuard(config, state.get('numerics'))
        inactive_amp_scaler_state = (state['scaler'] if state['config']['amp'] and not config.amp else
                                     state.get('inactive_amp_scaler_state', {}))
        write_json(out/'resume_state.json', bundle['report'])
        progress.note('Resume validated: ' + json.dumps(bundle['report']))
        # Copy into this NEW run only. Originals remain unmodified. A second
        # interrupted Epoch 4 can still resume this boundary checkpoint.
        copy_resume_boundary(bundle, out)
        from .checkpoints import bind_resume_identity
        bind_resume_identity(out)
        write_history_metrics(out, history)
        progress.note(f'Completed Epoch {start} restored; next Epoch {start+1}. '
                      f'Boundary checkpoints copied to {out}; source checkpoints preserved.')
        status = 'resumed'
        del state, bundle
    write_json(out / 'config.json', config.as_dict() | {'protocol': protocol, 'index_sha256': index_hash, 'pool_sha256':pool_hash,
        'selection_metric': selection_label(config.checkpoint_selection), 'threshold_policy': f'epoch/full-DEV metrics at {config.threshold}; separate final DEV family-calibrated F2 analysis',
        'scheduler_monitor': selection_label(config.checkpoint_selection),
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0),
        'start_mode': 'resumed' if resume_from else 'fresh',
        'resume_from': str(resume_from) if resume_from else None, 'logging_version': LOGGING_VERSION,
        'numerics_version': NUMERICS_VERSION, 'precision': 'amp_fp16' if config.amp else 'fp32'})
    progress.note(f'{config.model}: starting {"resumed" if resume_from else "fresh"} training; '
                  f'epochs {start+1} onward; output {out}; logging {LOGGING_VERSION}')
    policy = {'aggregation': 'topk', 'topk_fraction': config.topk_fraction,
              'prediction_threshold': config.threshold, 'video_threshold': config.threshold}
    for epoch in range(start, config.epochs):
        torch.cuda.reset_peak_memory_stats()
        progress.stage('epoch_setup', epoch=epoch+1, batch=0)
        model.set_phase(epoch >= config.warmup_epochs)
        phase = model.trainable_report()
        write_json(out / f"trainable_{model.phase}.json", phase)
        ids, diagnostic = sampler.select(epoch)
        write_json(out / f'sampling_epoch_{epoch+1:03}.json', diagnostic | {'candidate_ids': ids})
        phase_label = 'Frozen ResNet50 + temporal head' if temporal else ('Head warm-up' if epoch < config.warmup_epochs else ('Final-stage fine-tuning' if model.phase == 'final_stage' else 'Head-only training'))
        progress.note(f'\n{progress.model} | Epoch {epoch+1}/{config.epochs} | {phase_label}')
        progress.stage('materialize_selected_records', announce=True)
        records = index.records('train','clip',ids) if temporal else cnn_records(index, pool, ids)
        progress.stage('initialize_train_loader', announce=True)
        loader = loader_factory(records, data_root, config, training=True, epoch=epoch)
        lrs = {g['name']: g['lr'] for g in optimizer.param_groups}
        train_metrics, _, train_seconds = run_epoch(model, loader, config, optimizer, scaler, progress=progress,
                                                  epoch=epoch, gradient_guard=gradient_guard)
        del loader, records
        dev_metrics, predictions, dev_seconds = run_epoch(model, dev_loader, config, progress=progress, epoch=epoch)
        progress.stage('aggregate_dev_video_metrics', announce=True)
        report = evaluate_predictions(dev_records, predictions, policy, kind)
        report['losses'] = {k: dev_metrics[k] for k in ('loss', 'real_class_loss', 'fake_class_loss', 'balanced_loss')}
        score = checkpoint_score(report, config.checkpoint_selection)
        improved = score > best
        if improved:
            best = score; best_epoch = epoch+1; stale = 0; best_balanced = dev_metrics['balanced_loss']
        elif epoch >= config.warmup_epochs:
            stale += 1
        if epoch >= config.warmup_epochs:
            scheduler.step(score)
        entry = {'epoch': epoch+1, 'train': train_metrics, 'dev': dev_metrics, 'dev_video': report['video'],
            'sampling': diagnostic, 'learning_rates': lrs, 'train_seconds': train_seconds, 'dev_seconds': dev_seconds,
            'phase': model.phase, 'trainable_parameters': phase['trainable_parameters'],
            'checkpoint_selection': config.checkpoint_selection, 'selection_score': score,
            'batchnorm_running_stats_frozen': True, 'is_best': improved, 'stale_epochs': stale,
            'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20}
        history.append(entry)
        progress.stage('build_checkpoint_payload', announce=True)
        payload = {'protocol': protocol, 'config': config.as_dict(), 'index_sha256': index_hash, 'pool_sha256':pool_hash,
            'feature_source':feature_source,
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(), 'rng': rng_state(), 'sampler': sampler.state_dict(), 'epoch': epoch,
            'history': history, 'best': best, 'stale': stale, 'best_epoch': best_epoch,
            'dev_predictions': predictions,
            'best_balanced_dev_loss': best_balanced, 'numerics': gradient_guard.state_dict(),
            'inactive_amp_scaler_state': inactive_amp_scaler_state}
        if (out/'experiment_identity.json').is_file():
            payload['experiment_identity'] = json.loads((out/'experiment_identity.json').read_text())
        if input_pipeline is not None:
            payload['input_pipeline'] = input_pipeline.identity
        commit_epoch(out, history, entry, payload, improved, dev_records, predictions, report, config, progress, persistent_backup, input_pipeline)
        if stale >= config.patience:
            print(f'Early stopping: best {selection_label(config.checkpoint_selection)} {best:.6f} at epoch {best_epoch}', flush=True)
            break
    # Regenerate final best artifacts even if an earlier session stopped during
    # plotting, or the resumed segment did not improve the prior best.
    progress.stage('load_best_for_final_dev', announce=True)
    best_state = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    saved = checkpoint_dev_report(best_state, dev_records, policy, kind)
    if saved is not None:
        predictions, report = saved
        progress.note('Reusing selected-checkpoint DEV predictions and exact saved BCE losses; no additional DEV inference.')
    else:
        model.load_state_dict(best_state['model'])
        progress.note('Legacy checkpoint lacks saved DEV predictions: one final full DEV evaluation (no optimizer steps).')
        best_dev_metrics, predictions, _ = run_epoch(model, dev_loader, config, progress=progress, epoch=best_epoch-1)
        report = evaluate_predictions(dev_records, predictions, policy, kind)
        report['losses'] = {k: best_dev_metrics[k] for k in ('loss', 'real_class_loss', 'fake_class_loss', 'balanced_loss')}
    progress.stage('final_prediction_artifacts', announce=True)
    prediction_artifacts(out, dev_records, predictions, report,unit_label=kind.title())
    progress.stage('final_plots_and_dev_threshold_analysis', announce=True)
    if config.checkpoint_selection in ('frame_roc_auc', 'clip_roc_auc'):
        plot_history(out, history, unit_label=kind.title(), best_epoch=best_epoch,
                     selection_metric=config.checkpoint_selection)
    else:
        plot_history(out, history,unit_label=kind.title())
    write_final_numerics(out, history)
    from .evaluation import save_calibration
    calibration=save_calibration(out,dev_records,predictions,config.seed)
    operating=calibration['reporting']
    progress.note(f"DEV reporting families at calibration F2 threshold {calibration['threshold']:.6f}: "
        f"precision {operating['precision']:.6f}, recall {operating['recall']:.6f}, F1 {operating['f1']:.6f}, F2 {operating['f2']:.6f}")
    threshold_analysis=dev_threshold_analysis(
        [p['target'] for p in predictions], [p['probability'] for p in predictions], config.threshold)
    threshold_analysis['scope']=f'DEV-only descriptive {kind} analysis; exploratory, optimistic on full DEV'
    write_json(out/'dev_threshold_analysis.json',threshold_analysis)
    assessment = {'status': 'Historical reference unavailable; inspect actual DEV curves.',
                  'training_auc_is_online_augmented': True,
                  'best_to_final_dev_video_auc_change': history[-1]['dev_video']['roc_auc'] - history[best_epoch-1]['dev_video']['roc_auc'],
                  'note': 'A smaller training gap alone does not demonstrate improved generalization.'}
    assessment.update(status='New standalone ROC-AUC-selected run; no historical metrics imported.',
                      checkpoint_selection=config.checkpoint_selection,
                      comparison_metric='DEV video PR-AUC (trapezoidal)')
    progress.stage('write_final_summary', announce=True)
    write_json(out/'generalization_assessment.json', assessment)
    write_json(out / 'training_summary.json', {'status': status, 'model': config.model, 'epochs_run': len(history),
        'best_epoch': best_epoch, 'best_dev_video_roc_auc': report['video']['roc_auc'],
        'best_dev_video_precision': report['video']['precision'], 'best_dev_video_recall': report['video']['recall'],
        'best_dev_video_f1': report['video']['f1'], 'best_dev_video_f2': report['video']['f2'],
        'checkpoint_selection': config.checkpoint_selection, 'best_selection_score': best, 'best_balanced_dev_loss': best_balanced,
        'threshold': config.threshold, 'topk_fraction': config.topk_fraction, 'test_used': False,
        'overfitting_assessment': assessment})
    if config.checkpoint_selection in ('frame_roc_auc', 'clip_roc_auc'):
        from .reporting import export_epoch_report
        progress.stage('save_cumulative_epoch_table', announce=True)
        export_epoch_report(out, out, kind=kind,
                            checkpoint_selection=config.checkpoint_selection, in_run=True)
    progress.stage('create_final_model_zip', announce=True)
    if input_pipeline is not None:
        input_pipeline.snapshot(out)
    final_zip = package_model_run(out, config.model, len(history), final=True, snapshot_lock=progress.lock)
    progress.note(f'Final {progress.model} download ZIP: {final_zip["archive"]}\nSHA256 {final_zip["sha256"]}')
    if persistent_backup is not None:
        progress.stage('verify_private_final_backup', announce=True)
        persistent_backup.save_run(out, config, len(history), kind='final_artifacts', snapshot_lock=progress.lock)
    index.close(); del model, optimizer, scheduler, scaler, dev_loader, best_state, payload
    gc.collect(); torch.cuda.empty_cache()
    return out
