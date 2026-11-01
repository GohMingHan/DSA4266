"""Timestamp-only temporal shift sensitivity for Experiment 3 frozen fusion heads.

Run: python -m xai.temporal_shift_sensitivity_v2 --data ... --checkpoint ...
Changes time coordinates only; never moves embeddings or padding masks.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from e3.common import backend_context, device_of, source_hashes
from e3.data import Sequences, collate, load_index
from e3.model import build_model
from e3.train import compare_data_checkpoint, make_loader, to_device

SHIFTS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
CATEGORIES = ('real', 'audio_only', 'visual_only', 'both')


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def read_reference(path):
    result = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r['file'] in result:
                    raise ValueError('Duplicate reference prediction')
                result[r['file']] = float(r['logit'])
    return result


def key(mode, shift):
    return 'original' if mode == 'original' else f'{mode}_{shift:+g}s'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--reference-predictions', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--split', choices=('dev', 'test'), default='test')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        p.error('batch-size and threads must be positive')
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite: {out}')
    device = device_of(args.device)
    torch.set_num_threads(args.threads)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg, rows, _ = load_index(args.data, args.split)
    compare_data_checkpoint(ckpt, cfg)
    if ckpt['source_hashes'] != source_hashes():
        raise ValueError('Source changed since training')
    if ckpt.get('test_only') or cfg.get('test_only'):
        raise ValueError('Synthetic fixtures not allowed')
    model = build_model(ckpt['dims'], ckpt['normalization'], ckpt['settings']).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    threshold = float(ckpt['threshold_logit'])
    reference = read_reference(args.reference_predictions)
    if set(reference) != {r['file'] for r in rows}:
        raise ValueError('Reference IDs do not match dataset')
    conditions = [('original', 0.0)] + [(mode, shift) for mode in ('audio_shift', 'joint_shift') for shift in SHIFTS]
    scores = {key(mode, shift): np.full(len(rows), np.nan, dtype=np.float64) for mode, shift in conditions}
    dataset = Sequences(cfg, rows, cache_videos=8)
    loader = make_loader(dataset, args.batch_size, 0, 271828, False, pin=device.type == 'cuda')
    with torch.inference_mode(), backend_context(ckpt['settings']['attention_backend']):
        for batch in loader:
            indices = batch['index'].numpy()
            original = to_device(batch, device)
            for mode, shift in conditions:
                # Make a shallow copy; clone only tensors whose time values change.
                b = dict(original)
                if mode in ('audio_shift', 'joint_shift'):
                    b['audio_time'] = original['audio_time'].clone()
                    valid = ~original['audio_padding']
                    b['audio_time'][valid] += shift
                if mode == 'joint_shift':
                    b['visual_time'] = original['visual_time'].clone()
                    valid = ~original['visual_padding']
                    b['visual_time'][valid] += shift
                z = model(b)
                if z.shape != (len(indices),) or not torch.isfinite(z).all():
                    raise ValueError('Invalid output')
                scores[key(mode, shift)][indices] = z.detach().cpu().double().numpy()
    for k, v in scores.items():
        if not np.isfinite(v).all():
            raise ValueError(f'Missing scores for {k}')
    err = max(abs(scores['original'][i] - reference[r['file']]) for i, r in enumerate(rows))
    print('Full-input maximum absolute logit error:', err)
    if err > 1e-5:
        raise ValueError('Full-input reference parity failed')
    if ckpt['settings']['mode'] == 'concat':
        max_change = max(float(np.max(np.abs(v - scores['original']))) for k, v in scores.items())
        if max_change > 1e-6:
            raise ValueError(f'Concat unexpectedly sensitive to timestamps: {max_change}')
        print('PASS: Concat timestamp invariance:', max_change)
    y = np.array([r['label'] for r in rows], dtype=np.int64)
    groups = np.array([r['category'] for r in rows])
    summary = {'model': ckpt['settings']['mode'], 'split': args.split,
               'intervention': 'timestamp values only; embeddings and padding unchanged',
               'threshold_logit': threshold, 'reference_max_abs_error': float(err),
               'conditions': {}}
    for mode, shift in conditions:
        name = key(mode, shift)
        z = scores[name]
        pred = z >= threshold
        delta = z - scores['original']
        entry = {
            'mode': mode, 'shift_seconds': shift,
            'roc_auc': float(roc_auc_score(y, z)),
            'median_abs_delta_logit': float(np.median(np.abs(delta))),
            'mean_abs_delta_logit': float(np.mean(np.abs(delta))),
            'flip_rate': float(np.mean(pred != (scores['original'] >= threshold))),
            'categories': {}
        }
        for category in CATEGORIES:
            mask = groups == category
            if not mask.any():
                continue
            entry['categories'][category] = {
                'n': int(mask.sum()),
                'predicted_fake_rate': float(pred[mask].mean()),
                'median_abs_delta_logit': float(np.median(np.abs(delta[mask]))),
                'median_signed_delta_logit': float(np.median(delta[mask])),
                'flip_rate': float(np.mean(pred[mask] != (scores['original'][mask] >= threshold)))
            }
        summary['conditions'][name] = entry
        print(f'{name:18s} AUC={entry["roc_auc"]:.4f} '
              f'median|dlogit|={entry["median_abs_delta_logit"]:.4f} '
              f'flip={entry["flip_rate"]:.3f}')
    out.mkdir(parents=True)
    save_json(out / 'summary.json', summary)
    with (out / 'predictions.jsonl').open('w', encoding='utf-8') as fh:
        for i, r in enumerate(rows):
            record = {'file': r['file'], 'source_family': r['source_family'],
                      'category': r['category'], 'label': int(r['label'])}
            for mode, shift in conditions:
                name = key(mode, shift)
                record[name + '_logit'] = float(scores[name][i])
                record[name + '_prediction'] = int(scores[name][i] >= threshold)
            fh.write(json.dumps(record, allow_nan=False) + '\n')
    print('Saved:', out)

if __name__ == '__main__':
    main()
