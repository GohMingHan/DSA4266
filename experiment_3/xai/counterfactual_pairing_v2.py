"""Matched-source-family audio/visual counterfactuals for Experiment 3 V2.

Run as: python -m xai.counterfactual_pairing_v2 ...
Counterfactual pairings have no asserted ground-truth labels.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from e3.common import backend_context, device_of, source_hashes
from e3.data import Sequences, collate, load_index
from e3.model import build_model
from e3.train import compare_data_checkpoint, to_device

CATS = ('real', 'audio_only', 'visual_only', 'both')


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--reference-predictions', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        p.error('batch-size and threads must be positive')
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    device = device_of(args.device)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg, rows, _ = load_index(args.data, 'test')
    compare_data_checkpoint(checkpoint, cfg)
    if checkpoint['source_hashes'] != source_hashes():
        raise ValueError('Model source differs from training version')
    if checkpoint.get('test_only') or cfg.get('test_only'):
        raise ValueError('Synthetic fixture not allowed')
    model = build_model(checkpoint['dims'], checkpoint['normalization'], checkpoint['settings']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    threshold = float(checkpoint['threshold_logit'])
    dataset = Sequences(cfg, rows, cache_videos=8)

    families = defaultdict(dict)
    for i, r in enumerate(rows):
        family = families[r['source_family']]
        if r['category'] in family:
            raise ValueError(f'Duplicate category in family {r["source_family"]}')
        family[r['category']] = i
    if len(rows) != 500 or len(families) != 125 or any(set(f) != set(CATS) for f in families.values()):
        raise ValueError('Expected 125 complete four-category TEST families')

    reference = {}
    with open(args.reference_predictions) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r['file'] in reference:
                    raise ValueError('Duplicate reference file')
                reference[r['file']] = float(r['logit'])
    if set(reference) != {r['file'] for r in rows}:
        raise ValueError('Reference prediction file IDs differ from TEST dataset')

    pairings = []
    for family_id, members in sorted(families.items()):
        for visual_cat in CATS:
            for audio_cat in CATS:
                vi, ai = members[visual_cat], members[audio_cat]
                vr, ar = rows[vi], rows[ai]
                pairings.append((family_id, visual_cat, audio_cat, vi, ai, vr, ar))

    results = []
    max_control_error = 0.0
    with torch.inference_mode(), backend_context(checkpoint['settings']['attention_backend']):
        for start in range(0, len(pairings), args.batch_size):
            chunk = pairings[start:start + args.batch_size]
            items = []
            for _, _, _, vi, ai, _, _ in chunk:
                visual = dataset[vi]
                audio = dataset[ai]
                # Original visual content and time, donor audio content and time.
                # collate() builds fresh padding masks for each mixed pair.
                items.append({
                    'index': vi,
                    'label': visual['label'],
                    'visual': visual['visual'],
                    'visual_time': visual['visual_time'],
                    'visual_mean': visual['visual_mean'],
                    'audio': audio['audio'],
                    'audio_time': audio['audio_time'],
                    'audio_mean': audio['audio_mean'],
                })
            batch = to_device(collate(items), device)
            logits = model(batch).detach().cpu().numpy().reshape(-1)
            if len(logits) != len(chunk) or not np.isfinite(logits).all():
                raise ValueError('Invalid model output')
            for entry, z in zip(chunk, logits):
                family_id, vcat, acat, vi, ai, vr, ar = entry
                z = float(z)
                if vi == ai:
                    err = abs(z - reference[vr['file']])
                    max_control_error = max(max_control_error, err)
                results.append({
                    'source_family': family_id,
                    'visual_category': vcat,
                    'audio_category': acat,
                    'visual_file': vr['file'],
                    'audio_file': ar['file'],
                    'unchanged': vi == ai,
                    'visual_steps': int(vr['visual']['shape'][0]),
                    'audio_windows': int(ar['audio']['shape'][0]),
                    'logit': z,
                    'prediction': int(z >= threshold),
                })
    if max_control_error > 1e-5:
        raise ValueError(f'Full-input parity failed: maximum error {max_control_error}')

    lookup = {(r['source_family'], r['visual_category'], r['audio_category']): r for r in results}
    for r in results:
        base = lookup[(r['source_family'], r['visual_category'], r['visual_category'])]
        r['baseline_logit'] = base['logit']
        r['delta_logit'] = r['logit'] - base['logit']
        r['prediction_flipped'] = int(r['prediction'] != base['prediction'])

    summary = []
    for vcat in CATS:
        for acat in CATS:
            group = [r for r in results if r['visual_category'] == vcat and r['audio_category'] == acat]
            assert len(group) == 125
            summary.append({
                'visual_category': vcat,
                'audio_category': acat,
                'n': len(group),
                'median_delta_logit': float(np.median([r['delta_logit'] for r in group])),
                'mean_delta_logit': float(np.mean([r['delta_logit'] for r in group])),
                'positive_delta_fraction': float(np.mean([r['delta_logit'] > 0 for r in group])),
                'prediction_flip_rate': float(np.mean([r['prediction_flipped'] for r in group])),
                'predicted_fake_rate': float(np.mean([r['prediction'] for r in group])),
            })
    out.mkdir(parents=True)
    with (out / 'pair_predictions.jsonl').open('w') as fh:
        for r in results:
            fh.write(json.dumps(r, allow_nan=False) + '\n')
    write_json(out / 'summary.json', {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'dataset': str(Path(args.data).resolve()),
        'model': checkpoint['settings']['mode'],
        'threshold_logit': threshold,
        'source_families': len(families),
        'pairings': len(results),
        'unchanged_max_abs_logit_error': max_control_error,
        'interpretation': 'Counterfactual embeddings; no ground-truth labels assigned to swapped pairs',
        'groups': summary,
    })
    print('PASS: unchanged-input max absolute logit error:', max_control_error)
    print('Counterfactual pairs:', len(results))
    print('Visual target / Audio donor / Median delta logit / Flip rate')
    for r in summary:
        print(f"{r['visual_category']:12s} {r['audio_category']:12s} {r['median_delta_logit']:+9.3f} {r['prediction_flip_rate']:.3f}")
    print('Saved:', out)


if __name__ == '__main__':
    main()
