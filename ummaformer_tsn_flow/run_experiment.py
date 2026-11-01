#!/usr/bin/env python3
"""Train/evaluate the official UMMAFormer visual model on an audited LAV-DF cohort.

The model, losses, optimizer grouping, scheduler implementation and temporal NMS
come from the author's checkout. Our loader, balanced-video sampler, dev AUC
selection, finite-gradient guard and all-cohort evaluation are adaptations.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import nullcontext
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from common import VERSION, json_write, jsonl_write, sha256
from experiment_metrics import choose_threshold, video_metrics, temporal_metrics
from visual_data import VisualDataset, MixedBatchSampler, list_collate, load_manifest
from upstream_adapter import (build_official, local_sources, split_identity,
                              ExponentialAverage)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # Exact bitwise GPU reproducibility is not guaranteed across software/hardware.


def get_device(name, amp):
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable. No silent CPU fallback.')
        if device.index is None:
            device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    elif device.type != 'cpu':
        raise ValueError('This integration supports CPU or a single CUDA device.')
    if device.type != 'cuda' and amp != 'none':
        raise ValueError('Use --amp none on CPU.')
    if amp == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not report BF16 support.')
    return device


def amp_context(device, amp):
    return nullcontext() if amp == 'none' else torch.autocast('cuda',
        dtype=torch.float16 if amp == 'fp16' else torch.bfloat16)


def scaler_for(amp):
    return torch.amp.GradScaler('cuda', enabled=(amp == 'fp16'), init_scale=128.0)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_checkpoint(path, content):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(content, tmp)
    os.replace(tmp, path)


def rng_state():
    ns = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [ns[0], ns[1].tolist(), int(ns[2]), int(ns[3]), float(ns[4])],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def checkpoint_load(path):
    # This runner only loads its own tensor/primitive checkpoints, not arbitrary author pickles.
    c = torch.load(path, map_location='cpu', weights_only=True)
    if c.get('format') != 'ummaformer_tsn_flow_visual_v1':
        raise ValueError('Not a checkpoint produced by this runner. Do not substitute a full-LAV-DF pretrained detector.')
    return c


def verify_checkpoint(c, manifest, identity):
    if c['local_sources'] != local_sources() or c['upstream']['source_digest'] != identity['source_digest']:
        raise ValueError('Training source changed. Use the run-specific original code and upstream checkout.')
    current = split_identity(manifest)
    for split in ('train', 'dev'):
        if c['split_identity'].get(split) != current.get(split):
            raise ValueError(f'{split} cohort/labels/features/timing changed since training.')


def train_settings(args):
    return {k: getattr(args, k) for k in ('epochs', 'warmup_epochs', 'batch_size', 'lr',
        'weight_decay', 'seed', 'amp', 'patience', 'ema', 'max_consecutive_skips', 'input_mode')}


def train_update(model, batch, optimizer, scaler, device, amp):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with amp_context(device, amp):
        losses = model(batch)
    if 'final_loss' not in losses or any(not torch.isfinite(v).all() for v in losses.values()):
        raise FloatingPointError('Nonfinite/unsupported training loss. Stopping before updating weights.')
    total = losses['final_loss']
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    bad = [name for name, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    scale_before = float(scaler.get_scale())
    if bad and amp != 'fp16':
        raise FloatingPointError(f'Nonfinite gradients in {len(bad)} tensors, e.g. {bad[:5]}')
    if not bad:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    else:
        norm = None
    scaler.step(optimizer)  # GradScaler skips the update after a detected FP16 overflow.
    scaler.update()
    values = {k: float(v.detach()) for k, v in losses.items()}
    info = {'skipped': bool(bad), 'nonfinite_gradient_tensors': len(bad),
            'grad_norm': float(norm) if norm is not None else None,
            'scale_before': scale_before, 'scale_after': float(scaler.get_scale())}
    return values, info


@torch.no_grad()
def predict(model, dataset, device, amp):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=list_collate)
    predictions, scores = {}, []
    for i, batch in enumerate(tqdm(loader, desc='Evaluate videos', leave=False)):
        row = dataset.rows[i]
        # Do not provide ground truth, modality labels or intervals to inference.
        item = {k: v for k, v in batch[0].items() if k not in ('segments', 'labels', 'av_labels')}
        with amp_context(device, amp):
            result = model([item])
        if not isinstance(result, list) or len(result) != 1 or result[0]['video_id'] != row['file']:
            raise ValueError('Unexpected official model inference interface.')
        result = result[0]
        segments = result['segments'].detach().float().cpu().numpy()
        confidence = result['scores'].detach().float().cpu().numpy()
        classes = result['labels'].detach().cpu().numpy()
        if not np.isfinite(segments).all() or not np.isfinite(confidence).all():
            raise FloatingPointError(f'Nonfinite predictions: {row["file"]}')
        if not np.isin(classes, [0]).all():
            raise ValueError('Unexpected class index; this is a single visual-forgery class experiment.')
        if (confidence < 0).any() or (confidence > 1.00001).any():
            raise ValueError('Proposal scores outside [0, 1].')
        proposals = []
        for segment, score in zip(segments, confidence):
            a, b = map(float, segment)
            # Official postprocessing clamps proposals to the video; remove only
            # zero-length predictions produced by that clipping, not ground truth.
            if a < -1e-4 or b > row['duration_s'] + 1e-4:
                raise ValueError('Predictions are outside video extent; feature-time conversion is inconsistent.')
            if b <= a:
                continue
            proposals.append({'segment': [a, b], 'score': float(score), 'label': 'visual_fake'})
        proposals.sort(key=lambda p: -p['score'])
        predictions[row['file']] = proposals
        scores.append({'file': row['file'], 'split': row['split'],
                       'visual_label': int(row['modify_video']),
                       'modify_audio': int(row['modify_audio']),
                       'source_group': row['source_group'],
                       'score': max((p['score'] for p in proposals), default=0.0),
                       'proposal_count': len(proposals)})
    return scores, predictions


def evaluate_results(rows, scores, proposals, threshold=None):
    if threshold is None:
        threshold = choose_threshold([r['visual_label'] for r in scores], [r['score'] for r in scores])
    result = video_metrics(scores, threshold)
    result.update(temporal_metrics(rows, proposals))
    return result


def save_predictions(folder, scores, proposals, metrics):
    folder.mkdir(parents=True, exist_ok=True)
    jsonl_write(folder / 'video_scores.jsonl', scores)
    jsonl_write(folder / 'proposals.jsonl', ({'file': f, 'proposals': p} for f, p in proposals.items()))
    json_write(folder / 'metrics.json', metrics)


def train(args):
    manifest = load_manifest(args.manifest)
    if manifest.get('test_only'):
        raise ValueError('Random-encoder diagnostic cache cannot be used for training.')
    device = get_device(args.device, args.amp)
    seed_all(args.seed)
    model, cfg, utilities, identity, counts = build_official(args.upstream, device, args.input_mode)
    print(json.dumps({'model': cfg['model_name'], 'input_mode': args.input_mode, 'input_channels': cfg['model']['input_dim'], **counts, 'device': str(device), 'amp': args.amp}, indent=2))
    train_ds, dev_ds = VisualDataset(manifest, 'train', input_mode=args.input_mode), VisualDataset(manifest, 'dev', input_mode=args.input_mode)
    sampler = MixedBatchSampler(train_ds.rows, args.batch_size, args.seed)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.workers,
                        collate_fn=list_collate, persistent_workers=(args.workers > 0))
    opt_cfg = copy.deepcopy(cfg['opt'])
    opt_cfg.update(learning_rate=args.lr, weight_decay=args.weight_decay,
                   epochs=args.epochs - args.warmup_epochs,
                   warmup=(args.warmup_epochs > 0), warmup_epochs=args.warmup_epochs)
    optimizer = utilities.make_optimizer(model, opt_cfg)
    scheduler = utilities.make_scheduler(optimizer, opt_cfg, len(loader))
    scaler = scaler_for(args.amp)
    ema = ExponentialAverage(model) if args.ema else None
    out = args.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError('Output is not empty. Use a NEW --out or resume its last.pt.')
    out.mkdir(parents=True, exist_ok=True)
    settings = train_settings(args)
    start, history, best_auc, best_ap, stale = 0, [], -1., -1., 0
    if args.resume:
        c = checkpoint_load(args.resume)
        verify_checkpoint(c, manifest, identity)
        if c['settings'] != settings or Path(c['out']).resolve() != out:
            raise ValueError('Resume requires the original settings and output directory.')
        model.load_state_dict(c['model_state'], strict=True)
        model.loss_normalizer = c['loss_normalizer']
        optimizer.load_state_dict(c['optimizer'])
        scheduler.load_state_dict(c['scheduler'])
        scaler.load_state_dict(c['scaler'])
        if ema:
            ema.module.load_state_dict(c['ema_state'], strict=True)
        start, history = c['epoch'], c['history']
        best_auc, best_ap, stale = c['best_auc'], c['best_ap50'], c['stale_epochs']
        restore_rng(c['rng'])
        del c
    json_write(out / 'run_config.json', {'version': VERSION, 'settings': settings,
        'model_config': cfg, 'upstream': identity, 'local_sources': local_sources(),
        'split_identity': split_identity(manifest), 'parameter_counts': counts,
        'feature_kind': manifest['feature_kind'], 'feature_config': manifest.get('feature_config'),
        'training_videos': len(train_ds), 'development_videos': len(dev_ds),
        'sampled_videos_per_epoch': len(sampler) * args.batch_size,
        'selection': 'dev_video_visual_roc_auc_max_proposal',
        'evaluation_weights': 'ema' if ema else 'raw',
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'notes': ['Preserved ResNet18 RGB512 plus frozen pretrained TSN flow2048; no audio; dense-flow method in feature manifest.', 'Mixed-class video sampler with minority oversampling.',
                  'Actual feature PTS interpolated to 768 bin centers; no label-based time sampling.',
                  'All-cohort temporal AP includes negative videos.', 'Not a full-dataset paper reproduction.']})
    for epoch in range(start, args.epochs):
        sampler.set_epoch(epoch)
        begin = time.perf_counter()
        terms, updates, skipped, streak = defaultdict(float), 0, 0, 0
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for batch in tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}', unit='batch'):
            values, info = train_update(model, batch, optimizer, scaler, device, args.amp)
            for name, value in values.items():
                terms[name] += value
            if info['skipped']:
                skipped += 1; streak += 1
                print(json.dumps({'event': 'fp16_update_skipped', 'epoch': epoch+1, **info}))
                if streak >= args.max_consecutive_skips:
                    raise FloatingPointError('Too many consecutive FP16 skips; stop and investigate or use BF16/FP32.')
            else:
                streak = 0; updates += 1
                scheduler.step()
                if ema:
                    ema.update(model)
        if updates == 0:
            raise FloatingPointError('No optimizer updates in this epoch.')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - begin
        selected = ema.module if ema else model
        scores, proposals = predict(selected, dev_ds, device, args.amp)
        metrics = evaluate_results(dev_ds.rows, scores, proposals)
        auc = metrics['video_visual_roc_auc']
        ap = metrics['all_cohort_visual_temporal_ap']['AP@0.50']
        better_auc, better_ap = auc > best_auc, ap > best_ap
        stale = 0 if better_auc else stale + 1
        best_auc, best_ap = max(best_auc, auc), max(best_ap, ap)
        record = {'epoch': epoch+1, 'loss': terms['final_loss'] / len(loader),
            'loss_terms': {k: v / len(loader) for k, v in terms.items()},
            'optimizer_steps': updates, 'skipped_optimizer_steps': skipped,
            'final_grad_scale': float(scaler.get_scale()), 'train_seconds': train_seconds,
            'dev_video_visual_roc_auc': auc,
            'dev_temporal_ap': metrics['all_cohort_visual_temporal_ap'],
            'epoch_seconds_before_checkpoint': time.perf_counter() - begin,
            'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated(device)/1024**2 if device.type == 'cuda' else None}
        history.append(record)
        json_write(out / 'history.json', history)
        print(json.dumps(record))
        save_predictions(out / 'last_dev', scores, proposals, metrics)
        if better_auc:
            save_predictions(out / 'best_dev', scores, proposals, metrics)
        state = {'format': 'ummaformer_tsn_flow_visual_v1', 'epoch': epoch+1,
            'out': str(out), 'settings': settings, 'history': history,
            'model_state': cpu_state(model), 'ema_state': cpu_state(ema.module) if ema else None,
            'loss_normalizer': float(model.loss_normalizer),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(), 'rng': rng_state(), 'best_auc': best_auc,
            'best_ap50': best_ap, 'stale_epochs': stale, 'dev_metrics': metrics,
            'local_sources': local_sources(), 'upstream': identity,
            'split_identity': split_identity(manifest), 'parameter_counts': counts}
        save_checkpoint(out / 'last.pt', state)
        if better_auc:
            save_checkpoint(out / 'best_video_auc.pt', state)
        if better_ap:
            save_checkpoint(out / 'best_temporal_ap50.pt', state)
        del state
        if args.patience and stale >= args.patience and epoch + 1 > args.warmup_epochs:
            print(f'Early stopping: no dev-video-AUC improvement for {stale} epochs.')
            break
    print(f'Video-comparison checkpoint: {out / "best_video_auc.pt"}')


def evaluate(args):
    c = checkpoint_load(args.checkpoint)
    amp = c['settings']['amp']
    device = get_device(args.device, amp)
    manifest = load_manifest(args.manifest)
    if manifest.get('test_only'):
        raise ValueError('Random-encoder diagnostic cache cannot be used for evaluation.')
    model, cfg, utilities, identity, counts = build_official(args.upstream, device, c['settings']['input_mode'])
    verify_checkpoint(c, manifest, identity)
    state = c['ema_state'] if c['settings']['ema'] else c['model_state']
    model.load_state_dict(state, strict=True)
    ds = VisualDataset(manifest, args.split, input_mode=c['settings']['input_mode'])
    out = args.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty evaluation output directory.')
    scores, proposals = predict(model, ds, device, amp)
    # Never choose a threshold on test. Reuse the development threshold stored at this checkpoint.
    metrics = evaluate_results(ds.rows, scores, proposals, threshold=c['dev_metrics']['threshold'])
    metrics.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
                   checkpoint_epoch=c['epoch'], split=args.split,
                   threshold_source='checkpoint_development_set',
                   evaluated_split_identity=split_identity(manifest)[args.split],
                   input_mode=c['settings']['input_mode'], feature_kind=manifest['feature_kind'],
                   evaluation_weights='ema' if c['settings']['ema'] else 'raw',
                   settings=c['settings'], parameter_counts=counts)
    save_predictions(out, scores, proposals, metrics)
    print(json.dumps(metrics, indent=2))


def smoke(args):
    device = get_device(args.device, args.amp)
    seed_all(args.seed)
    model, cfg, utilities, identity, counts = build_official(args.upstream, device, args.input_mode)
    if args.manifest:
        manifest = load_manifest(args.manifest)
        ds = VisualDataset(manifest, 'train', input_mode=args.input_mode)
        sampler = MixedBatchSampler(ds.rows, args.batch_size, args.seed)
        batch = [ds[i] for i in next(iter(sampler))]
        data_kind = 'real audited feature batch; temporary model, no production checkpoint'
    else:
        batch = []
        for i in range(args.batch_size):
            positive = bool(i % 2)
            batch.append({'video_id': f'synthetic_{i}', 'feats': torch.randn(cfg['model']['input_dim'], 768),
                'segments': torch.tensor([[150., 250.]]) if positive else None,
                'labels': torch.zeros(1, dtype=torch.long) if positive else None,
                'fps': 1., 'duration': 8., 'feat_stride': 8./768,
                'feat_num_frames': 8./768})
        data_kind = 'synthetic random features; SOFTWARE TEST ONLY'
    optimizer = utilities.make_optimizer(model, cfg['opt'])
    scaler = scaler_for(args.amp)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    begin = time.perf_counter()
    for i in range(args.steps):
        values, info = train_update(model, batch, optimizer, scaler, device, args.amp)
        print(json.dumps({'step': i+1, 'losses': values, **info}))
        if info['skipped']:
            raise FloatingPointError('Smoke test had a skipped update. Resolve numerical issues before long training.')
    model.eval()
    with torch.no_grad():
        item = {k: v for k, v in batch[0].items() if k not in ('segments', 'labels', 'av_labels')}
        result = model([item])
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    print(json.dumps({'smoke_passed': True, 'data': data_kind, 'input_channels': cfg['model']['input_dim'], 'input_mode': args.input_mode, **counts,
        'seconds_including_first_run_overhead': time.perf_counter() - begin,
        'inference_proposals': int(result[0]['scores'].numel()),
        'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated(device)/1024**2 if device.type == 'cuda' else None,
        'upstream_commit': identity['git_commit']}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    def shared(q):
        q.add_argument('--upstream', type=Path, default=Path(__file__).resolve().parent / 'upstream')
        q.add_argument('--device', default='cuda:0')
    t = subs.add_parser('train'); shared(t)
    t.add_argument('--input-mode', choices=['rgb-flow', 'rgb'], default='rgb-flow')
    t.add_argument('--manifest', type=Path, required=True)
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--epochs', type=int, default=15, help='Total, including warmup (release recipe: 5+10).')
    t.add_argument('--warmup-epochs', type=int, default=5)
    t.add_argument('--batch-size', type=int, default=4)
    t.add_argument('--workers', type=int, default=2)
    t.add_argument('--lr', type=float, default=.001)
    t.add_argument('--weight-decay', type=float, default=.05)
    t.add_argument('--patience', type=int, default=8, help='Dev-video-AUC patience; 0 disables.')
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--amp', choices=['none', 'bf16', 'fp16'], default='none')
    t.add_argument('--ema', action=argparse.BooleanOptionalAction, default=True)
    t.add_argument('--resume', type=Path)
    t.add_argument('--max-consecutive-skips', type=int, default=4)
    e = subs.add_parser('evaluate'); shared(e)
    e.add_argument('--manifest', type=Path, required=True)
    e.add_argument('--checkpoint', type=Path, required=True)
    e.add_argument('--split', choices=['dev', 'test'], default='dev')
    e.add_argument('--out', type=Path, required=True)
    s = subs.add_parser('smoke'); shared(s)
    s.add_argument('--input-mode', choices=['rgb-flow', 'rgb'], default='rgb-flow')
    s.add_argument('--manifest', type=Path)
    s.add_argument('--batch-size', type=int, default=2)
    s.add_argument('--steps', type=int, default=2)
    s.add_argument('--seed', type=int, default=42)
    s.add_argument('--amp', choices=['none', 'bf16', 'fp16'], default='none')
    a = p.parse_args()
    if a.command in ('train', 'smoke') and (a.batch_size < 2 or a.batch_size % 2):
        p.error('batch-size must be even and at least 2.')
    if a.command == 'train':
        if a.epochs <= a.warmup_epochs or a.warmup_epochs < 0 or a.workers < 0 or a.patience < 0:
            p.error('Invalid epochs/warmup/workers/patience.')
        if not math.isfinite(a.lr) or a.lr <= 0 or not math.isfinite(a.weight_decay) or a.weight_decay < 0:
            p.error('Invalid learning rate/weight decay.')
        if a.max_consecutive_skips < 1:
            p.error('max-consecutive-skips must be positive.')
        train(a)
    elif a.command == 'evaluate':
        evaluate(a)
    else:
        if a.steps < 1:
            p.error('steps must be positive.')
        smoke(a)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, ImportError, FloatingPointError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
