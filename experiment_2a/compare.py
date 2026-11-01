"""Compare matched evaluated runs; optional paired SOURCE-GROUP bootstrap.

No threshold or architecture selection is performed on test data.
The confidence interval conditions on trained checkpoints: it does not quantify
training-seed variability. Run independent seeds separately.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from prepare_audio import read_jsonl, write_json


def paired_bootstrap(a, b, repetitions, seed):
    if len(a) != len(b) or [r['clip_id'] for r in a] != [r['clip_id'] for r in b]:
        raise ValueError('Bootstrap requires exactly paired videos in matching order.')
    y = np.array([r['audio_label'] for r in a])
    sa, sb = np.array([r['score'] for r in a]), np.array([r['score'] for r in b])
    pa, pb = np.array([r['predicted_audio_label'] for r in a]), np.array([r['predicted_audio_label'] for r in b])
    groups = {}
    for i, row in enumerate(a):
        groups.setdefault(row['source_group'], []).append(i)
    indices = [np.array(v) for _, v in sorted(groups.items())]
    rng = np.random.default_rng(seed)
    auc, balanced = [], []
    for _ in range(repetitions):
        selected = rng.integers(0, len(indices), len(indices))
        ix = np.concatenate([indices[k] for k in selected])
        if len(np.unique(y[ix])) < 2:
            continue
        auc.append(float(roc_auc_score(y[ix], sb[ix]) - roc_auc_score(y[ix], sa[ix])))
        balanced.append(float(balanced_accuracy_score(y[ix], pb[ix]) - balanced_accuracy_score(y[ix], pa[ix])))
    result = {'direction': 'run_B_minus_run_A', 'unit': 'source_group',
              'source_groups': len(indices), 'requested_replicates': repetitions,
              'valid_replicates': len(auc), 'seed': seed}
    if auc:
        result['roc_auc_difference_95pct_percentile_interval'] = np.quantile(auc, [.025, .975]).tolist()
        result['balanced_accuracy_difference_95pct_percentile_interval'] = np.quantile(balanced, [.025, .975]).tolist()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True, help='Evaluation directories, not training directories.')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=0, help='For exactly two runs; e.g. 1000.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--allow-training-differences', action='store_true', help='Explicitly mark an unmatched-training comparison.')
    args = p.parse_args()
    if len(args.runs) < 2 or args.bootstrap < 0:
        p.error('Require at least two runs and a nonnegative bootstrap count.')
    reports = [json.loads((path / 'metrics.json').read_text()) for path in args.runs]
    predictions = [sorted(read_jsonl(path / 'video_predictions.jsonl'), key=lambda r: r['clip_id']) for path in args.runs]
    first = reports[0]
    for report, rows in zip(reports[1:], predictions[1:]):
        for key in ('target', 'split', 'config_id', 'evaluated_cohort_signature', 'pooling'):
            if report[key] != first[key]:
                raise ValueError(f'Unmatched {key}; cannot present these as a matched evaluation.')
        if report['protocol'] != first['protocol'] and not args.allow_training_differences:
            raise ValueError('Training protocols/seeds differ. Use matched runs, or explicitly pass --allow-training-differences.')
        if [(r['clip_id'], r['audio_label'], r['source_group']) for r in rows] != [
                (r['clip_id'], r['audio_label'], r['source_group']) for r in predictions[0]]:
            raise ValueError('Video IDs, target labels or source groups differ.')
    output = []
    for path, report in zip(args.runs, reports):
        output.append({'run': str(path), 'model': report['model'], 'initialization': report['initialization'],
                       'seed': report['seed'], 'split': report['split'], **{
                           k: report['video_audio'][k] for k in ('n', 'positive_prevalence', 'roc_auc',
                           'average_precision', 'eer_interpolated', 'balanced_accuracy', 'precision', 'recall', 'f1', 'threshold')}})
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    result = {'rows': output, 'matched_training_protocol': all(r['protocol'] == first['protocol'] for r in reports),
              'metric_unit': 'video', 'target': 'audio_manipulation'}
    if args.bootstrap:
        if len(args.runs) != 2:
            raise ValueError('Paired bootstrap requires exactly two runs.')
        result['paired_bootstrap'] = paired_bootstrap(predictions[0], predictions[1], args.bootstrap, args.seed)
    write_json(args.out / 'comparison.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
