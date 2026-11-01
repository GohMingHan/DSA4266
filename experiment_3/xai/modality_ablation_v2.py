"""Frozen-feature modality-content ablation for Experiment 3 V2."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from e2b.metrics import evaluate_rows, sigmoid
from e3.common import backend_context, require_new, sha256, source_hashes
from e3.data import load_index, Sequences, collate, identity_rows
from e3.model import build_model
from e3.train import compare_data_checkpoint, to_device

CONDITIONS = ('full', 'no_audio', 'no_visual', 'neither')


def intervene(batch, model, condition):
    result = dict(batch)
    for modality in ('visual', 'audio'):
        if condition not in ('neither', 'no_' + modality):
            continue
        mean = getattr(model, modality + '_mean')
        seq = result[modality].clone()
        valid = ~result[modality + '_padding']
        seq[valid] = mean.expand_as(seq)[valid]
        result[modality] = seq
        result[modality + '_mean'] = mean.unsqueeze(0).expand_as(
            result[modality + '_mean']
        ).clone()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--out', required=True)
    parser.add_argument('--reference-predictions', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg, rows, _ = load_index(args.data, args.split)
    compare_data_checkpoint(checkpoint, cfg)
    if checkpoint['source_hashes'] != source_hashes():
        raise RuntimeError('Training source hashes differ from checkpoint.')
    if checkpoint.get('test_only'):
        raise RuntimeError('Synthetic checkpoint is not a research result.')

    model = build_model(checkpoint['dims'], checkpoint['normalization'],
                        checkpoint['settings']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    threshold = float(checkpoint['threshold_logit'])

    reference = {}
    with open(args.reference_predictions, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r['file'] in reference:
                    raise ValueError('Duplicate reference file')
                reference[r['file']] = float(r['logit'])
    if set(reference) != {r['file'] for r in rows}:
        raise ValueError('Reference predictions do not match evaluation split.')

    dataset = Sequences(cfg, rows, cache_videos=4)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate)
    scores = {condition: np.full(len(rows), np.nan, dtype=np.float64)
              for condition in CONDITIONS}
    seen = np.zeros(len(rows), dtype=bool)
    with torch.inference_mode(), backend_context(checkpoint['settings']['attention_backend']):
        for batch in loader:
            ids = batch['index'].numpy()
            if seen[ids].any():
                raise ValueError('Duplicate batch indices')
            seen[ids] = True
            batch = to_device(batch, device)
            for condition in CONDITIONS:
                modified = intervene(batch, model, condition)
                output = model(modified).detach().double().cpu().numpy()
                if output.shape != (len(ids),) or not np.isfinite(output).all():
                    raise RuntimeError('Invalid predictions')
                scores[condition][ids] = output
    if not seen.all():
        raise RuntimeError('Missing predictions')

    expected = np.array([reference[r['file']] for r in rows])
    max_error = float(np.max(np.abs(scores['full'] - expected)))
    print('Full-input maximum absolute logit error:', max_error)
    if max_error > 1e-5:
        raise RuntimeError('Full-input predictions differ from existing evaluation; refusing results.')

    metrics = {}
    for condition in CONDITIONS:
        metrics[condition] = evaluate_rows(rows, scores[condition], threshold)
        print(condition, 'AUC:', metrics[condition]['video_any_roc_auc'],
              'F1:', metrics[condition]['f1'],
              'FPR:', metrics[condition]['false_positive_rate'])

    output = Path(args.out)
    require_new(output)
    predictions = []
    for i, row in enumerate(identity_rows(rows)):
        record = dict(row)
        for condition in CONDITIONS:
            z = float(scores[condition][i])
            record[condition + '_logit'] = z
            record[condition + '_score'] = float(sigmoid(np.array([z]))[0])
            record[condition + '_prediction'] = int(z >= threshold)
        record['audio_content_delta_logit'] = float(scores['full'][i] - scores['no_audio'][i])
        record['visual_content_delta_logit'] = float(scores['full'][i] - scores['no_visual'][i])
        predictions.append(record)

    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as f:
        for record in predictions:
            f.write(json.dumps(record) + '\n')
    summary = {
        'split': args.split,
        'mode': checkpoint['settings']['mode'],
        'checkpoint_sha256': sha256(args.checkpoint),
        'encoder_ids': cfg['encoder_ids'],
        'normalization_sha256': cfg['normalization']['npz_sha256'],
        'threshold_logit': threshold,
        'full_parity_max_abs_logit_error': max_error,
        'intervention': 'Replace valid embedding tokens and pooled vectors with TRAIN mean; retain timestamps, padding, model biases.',
        'metrics': metrics,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Saved:', output)


if __name__ == '__main__':
    main()
