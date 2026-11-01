"""Experiment 2A: train/evaluate/export ResNet-18 and AASIST on matched LAV-DF audio."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from audio_data import make_dataset
from common import (aggregate, best_f1_threshold, device_from_arg, evaluate_predictions,
                    file_sha256, loader, load_records, seed_everything, split_signature,
                    training_statistics)
from prepare_audio import write_json, write_jsonl
from models import build_model, DEFAULT_AASIST_DIR


@torch.inference_mode()
def predict(model, batches, device, embeddings=False):
    model.eval()
    probabilities = np.empty(len(batches.dataset), dtype=np.float32)
    features = np.empty((len(batches.dataset), model.embedding_dim), dtype=np.float32) if embeddings else None
    for x, _, indices in tqdm(batches, desc='Inference', leave=False):
        h, logits = model.forward_features(x.to(device, non_blocking=True))
        if not torch.isfinite(logits).all() or not torch.isfinite(h).all():
            raise RuntimeError('Nonfinite model output; FP32 is required for this baseline.')
        idx = indices.numpy()
        probabilities[idx] = logits.sigmoid().cpu().numpy()
        if features is not None:
            features[idx] = h.cpu().numpy()
    return probabilities, features


def torch_save_atomic(value, path):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def source_compatible(saved, current):
    # Initial pretrained weight hash is provenance, not an inference dependency.
    return {k: v for k, v in saved.items() if k != 'initial_weights_sha256'} == {
        k: v for k, v in current.items() if k != 'initial_weights_sha256'}


def get_stats(root, records, config):
    path = root / 'normalization_train.json'
    signature = split_signature(records, 'train')
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['train_signature'] == signature and saved['config_id'] == config['config_id']:
            return saved['stats']
    stats = training_statistics(root, records, config['audio'])
    write_json(path, {'train_signature': signature, 'config_id': config['config_id'], 'stats': stats})
    return stats


def run_protocol(args, config, signatures, samples_per_epoch):
    return {'target': 'audio_manipulation', 'config_id': config['config_id'],
            'train_dev_signatures': signatures, 'initialization_track': args.init,
            'seed': args.seed, 'pooling': args.pooling,
            'sampling': 'inverse_class_frequency_windows_with_replacement',
            'samples_per_epoch': samples_per_epoch,
            'micro_batch_size': args.batch_size, 'accumulate': args.accumulate,
            'max_epochs': args.epochs, 'patience': args.patience,
            'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
            'optimizer': 'AdamW', 'scheduler': 'cosine',
            'freeze_epochs': args.freeze_epochs, 'threshold_rule': 'development_F1',
            'precision': 'float32', 'frequency_augmentation': False}


def train(args):
    root, out = args.root.resolve(), args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Training output is not empty. Use a new --out; checkpoints are never silently overwritten.')
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = device_from_arg(args.device)
    records, config = load_records(root)
    stats = get_stats(root, records, config) if args.model == 'resnet18' else None
    train_ds = make_dataset(args.model, root, records, 'train', stats, config['audio'])
    dev_ds = make_dataset(args.model, root, records, 'dev', stats, config['audio'])
    if any(len(np.unique(ds.labels)) != 2 for ds in (train_ds, dev_ds)):
        raise ValueError('Both train and development must contain both AUDIO window classes. Increase the pilot size.')
    if len({r['audio_video_label'] for r in dev_ds.records}) != 2:
        raise ValueError('Development VIDEOS must contain both audio classes.')
    model = build_model(args.model, args.init, args.aasist_weights, args.aasist_dir).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                           eta_min=args.learning_rate * 0.05)
    counts = np.bincount(train_ds.labels, minlength=2)
    sample_count = args.samples_per_epoch or len(train_ds)
    # Separate RNG from model construction: both backbones see the same sampled
    # window indices for a fixed seed/epoch/budget, despite different architectures.
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(torch.as_tensor(1. / counts[train_ds.labels], dtype=torch.double),
                                    sample_count, replacement=True, generator=generator)
    train_loader = loader(train_ds, args.batch_size, args.workers, sampler=sampler)
    dev_loader = loader(dev_ds, args.batch_size, args.workers)
    signatures = {s: split_signature(records, s) for s in ('train', 'dev')}
    protocol = run_protocol(args, config, signatures, sample_count)
    write_json(out / 'run_config.json', {'protocol': protocol, 'model': args.model,
               'model_provenance': model.provenance, 'embedding_dim': model.embedding_dim,
               'parameters': sum(p.numel() for p in model.parameters()),
               'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                               'platform': platform.platform(), 'device': str(device)},
               'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    if stats is not None:
        write_json(out / 'normalization.json', stats)
    best_auc, stale, history = -math.inf, 0, []
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    for epoch in range(args.epochs):
        start_time = time.perf_counter()
        frozen = args.init == 'pretrained' and epoch < args.freeze_epochs
        model.train()
        model.freeze_encoder(frozen)
        optimizer.zero_grad(set_to_none=True)
        running, seen, group_samples, steps = 0., 0, 0, 0
        for batch_i, (x, y, _) in enumerate(tqdm(train_loader, desc=f'{args.model}: epoch {epoch + 1}/{args.epochs}')):
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss.')
            loss.backward()
            running += loss.item()
            seen += len(y)
            group_samples += len(y)
            # Normalize over the actual accumulated sample count, including a
            # partial final accumulation group. BatchNorm still uses microbatches.
            if (batch_i + 1) % args.accumulate == 0 or batch_i + 1 == len(train_loader):
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(group_samples)
                nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                group_samples = 0
                steps += 1
        p, _ = predict(model, dev_loader, device)
        videos = aggregate(dev_ds, p, args.pooling)
        vy, vp = np.array([r['audio_label'] for r in videos]), np.array([r['score'] for r in videos])
        auc = float(roc_auc_score(vy, vp))
        row = {'epoch': epoch + 1, 'loss': running / seen, 'dev_video_audio_roc_auc': auc,
               'encoder_frozen': frozen, 'samples_seen': seen, 'optimizer_steps': steps,
               'learning_rate': optimizer.param_groups[0]['lr'],
               'elapsed_seconds': time.perf_counter() - start_time}
        scheduler.step()
        history.append(row)
        write_json(out / 'history.json', history)
        print(json.dumps(row))
        if auc > best_auc:
            best_auc, stale = auc, 0
            checkpoint = {'format_version': 1, 'model': model.state_dict(), 'stats': stats,
                          'audio': config['audio'], 'config_id': config['config_id'],
                          'split_signatures': signatures, 'protocol': protocol,
                          'pooling': args.pooling, 'epoch': epoch + 1,
                          'window_threshold': best_f1_threshold(dev_ds.labels, p),
                          'video_threshold': best_f1_threshold(vy, vp),
                          'dev_video_audio_roc_auc': auc, 'target': 'audio_manipulation',
                          'seed': args.seed, 'model_name': args.model,
                          'embedding_dim': model.embedding_dim,
                          'model_provenance': model.provenance, 'initialization': args.init}
            torch_save_atomic(checkpoint, out / 'best.pt')
            report, _ = evaluate_predictions(dev_ds, p, args.pooling,
                                             checkpoint['window_threshold'], checkpoint['video_threshold'])
            write_json(out / 'best_dev_metrics.json', report)
        else:
            stale += 1
        if stale >= args.patience:
            break
    print(f'Saved {out / "best.pt"}. Test data were not used for model or threshold selection.')


def load_checkpoint(args):
    device = device_from_arg(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    records, config = load_records(args.root.resolve())
    if config['config_id'] != checkpoint['config_id']:
        raise ValueError('Preprocessing configuration differs from training.')
    for split, signature in checkpoint['split_signatures'].items():
        if split_signature(records, split) != signature:
            raise ValueError(f'{split} data changed after training. Restore the frozen cohort or retrain.')
    # No original pretrained weights are needed to load a complete trained checkpoint.
    model = build_model(checkpoint['model_name'], 'scratch', aasist_dir=args.aasist_dir)
    if not source_compatible(checkpoint['model_provenance'], model.provenance):
        raise ValueError('Model implementation/configuration changed since this checkpoint was trained.')
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device), checkpoint, records, config, device


def evaluate(args):
    model, ckpt, records, config, device = load_checkpoint(args)
    ds = make_dataset(ckpt['model_name'], args.root.resolve(), records, args.split, ckpt['stats'], config['audio'])
    p, _ = predict(model, loader(ds, args.batch_size, args.workers), device)
    report, videos = evaluate_predictions(ds, p, ckpt['pooling'], ckpt['window_threshold'], ckpt['video_threshold'])
    report.update({'split': args.split, 'model': ckpt['model_name'], 'initialization': ckpt['initialization'],
                   'seed': ckpt['seed'], 'checkpoint_sha256': file_sha256(args.checkpoint),
                   'config_id': ckpt['config_id'], 'protocol': ckpt['protocol'],
                   'evaluated_cohort_signature': split_signature(records, args.split)})
    # Acoustic false positive rates in real/visual-only videos versus recall in
    # audio-only/joint videos are named explicitly, not conflated as one accuracy.
    for name, stats in report['by_category'].items():
        key = 'acoustic_recall' if name in ('audio_only', 'both') else 'acoustic_false_positive_rate'
        stats[key] = stats.pop('predicted_fake_fraction')
    for record in videos:
        record['predicted_audio_label'] = int(record['score'] >= ckpt['video_threshold'])
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'metrics.json', report)
    write_jsonl(args.out / 'video_predictions.jsonl', videos)
    windows = []
    for k, (i, j) in enumerate(ds.index):
        r = ds.records[i]
        windows.append({'clip_id': r['clip_id'], 'file': r['file'], **r['windows'][j], 'score': float(p[k])})
    write_jsonl(args.out / 'window_predictions.jsonl', windows)
    print(json.dumps(report, indent=2))


def export(args):
    model, ckpt, records, config, device = load_checkpoint(args)
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_hash = file_sha256(args.checkpoint)
    provenance_path = args.out / 'export_config.json'
    provenance = {'checkpoint_sha256': ckpt_hash, 'config_id': config['config_id'],
                  'model': ckpt['model_name'], 'embedding_dim': model.embedding_dim}
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError('Export directory contains features from a different checkpoint. Use a new directory.')
    write_json(provenance_path, provenance)
    # Rebuild a complete index for the requested splits. Existing unrequested
    # splits remain on disk but are not silently included in the new index.
    outputs = []
    for split in args.splits:
        selected = [r for r in records if r['split'] == split]
        if not selected:
            raise ValueError(f'No cached videos for requested split {split}.')
        for record in tqdm(selected, desc=f'Export {split}'):
            ds = make_dataset(ckpt['model_name'], args.root.resolve(), [record], split, ckpt['stats'], config['audio'])
            p, features = predict(model, loader(ds, args.batch_size, 0), device, embeddings=True)
            target = args.out / split / f'{record["clip_id"]}.npz'
            target.parent.mkdir(parents=True, exist_ok=True)
            w = record['windows']
            np.savez_compressed(target, features=features, mean_feature=features.mean(axis=0),
                                start_s=np.array([x['start_s'] for x in w]),
                                end_s=np.array([x['end_s'] for x in w]),
                                center_s=np.array([x['center_s'] for x in w]),
                                padding_fraction=np.array([x['padding_fraction'] for x in w]),
                                audio_probability=p)
            outputs.append({'clip_id': record['clip_id'], 'file': record['file'], 'split': split,
                            'feature_path': target.relative_to(args.out).as_posix(),
                            'feature_dim': model.embedding_dim, 'n_windows': len(w),
                            'source_group': record['source_group'], 'checkpoint_sha256': ckpt_hash,
                            'audio_label': record['audio_video_label'],
                            'overall_video_label': record['overall_video_label']})
    write_jsonl(args.out / 'index.jsonl', outputs)
    print(f'Exported {len(outputs)} videos. Use video identifiers and timestamps to join modalities.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    tr, ev, ex = (sub.add_parser(s) for s in ('train', 'evaluate', 'export'))
    for p in (tr, ev, ex):
        p.add_argument('--root', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--device', default='auto')
        p.add_argument('--batch-size', type=int, default=4)
        p.add_argument('--workers', type=int, default=0)
        p.add_argument('--threads', type=int, default=4)
        p.add_argument('--aasist-dir', type=Path, default=DEFAULT_AASIST_DIR)
    tr.add_argument('--model', choices=['resnet18', 'aasist'], required=True)
    tr.add_argument('--init', choices=['scratch', 'pretrained'], default='scratch')
    tr.add_argument('--aasist-weights', type=Path)
    tr.add_argument('--epochs', type=int, default=30)
    tr.add_argument('--patience', type=int, default=5)
    tr.add_argument('--learning-rate', type=float, default=1e-4)
    tr.add_argument('--weight-decay', type=float, default=1e-4)
    tr.add_argument('--freeze-epochs', type=int, default=0)
    tr.add_argument('--accumulate', type=int, default=1)
    tr.add_argument('--samples-per-epoch', type=int, default=0)
    tr.add_argument('--seed', type=int, default=42)
    tr.add_argument('--pooling', choices=['max', 'mean'], default='max')
    for p in (ev, ex):
        p.add_argument('--checkpoint', type=Path, required=True)
    ev.add_argument('--split', choices=['dev', 'test'], default='test')
    ex.add_argument('--splits', nargs='+', choices=['train', 'dev', 'test'], default=['train', 'dev', 'test'])
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.threads < 1:
        parser.error('batch-size and threads must be positive; workers must be nonnegative.')
    if args.command == 'train':
        if min(args.epochs, args.patience, args.accumulate) < 1 or args.samples_per_epoch < 0 or args.freeze_epochs < 0:
            parser.error('Invalid training counts.')
        if args.learning_rate <= 0 or args.weight_decay < 0:
            parser.error('learning-rate must be positive; weight-decay must be nonnegative.')
    torch.set_num_threads(args.threads)
    {'train': train, 'evaluate': evaluate, 'export': export}[args.command](args)


if __name__ == '__main__':
    main()
