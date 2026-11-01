"""Paired source-family interaction analysis of Experiment 3 V2 TEST swaps.

Run from experiment_3:
  python xai/analyze_crossmodal_interactions_v2.py
No inference or model loading is performed.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

CATS = ('real', 'audio_only', 'visual_only', 'both')
# First contrast changes only visual manipulation label; second changes audio manipulation label.
CONTRASTS = (
    ('visual_only', 'audio_only'),  # primary: orthogonal single-modality manipulations
    ('visual_only', 'both'),
    ('audio_only', 'audio_only'),
    ('audio_only', 'both'),
    ('both', 'audio_only'),
    ('both', 'both'),
)


def read_pairs(path):
    records = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    if len(records) != 2000:
        raise ValueError(f'{path}: expected 2000 rows, got {len(records)}')
    lookup = {}
    for r in records:
        key = (r['source_family'], r['visual_category'], r['audio_category'])
        if key in lookup:
            raise ValueError(f'Duplicate pairing: {key}')
        if not np.isfinite(r['logit']):
            raise ValueError(f'Nonfinite logit: {key}')
        lookup[key] = float(r['logit'])
    families = sorted({k[0] for k in lookup})
    if len(families) != 125:
        raise ValueError(f'Expected 125 families, got {len(families)}')
    for family in families:
        if any((family, v, a) not in lookup for v in CATS for a in CATS):
            raise ValueError(f'Incomplete family: {family}')
    return lookup, families


def interaction(L, family, vcat, acat):
    # Difference-in-differences of raw logits.
    return (L[family, vcat, acat] - L[family, vcat, 'real']
            - L[family, 'real', acat] + L[family, 'real', 'real'])


def ci_bootstrap(x, stat, rng, n_boot):
    # Each array element is a complete, independent resampling unit (source family).
    n = len(x)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        estimates[i] = stat(x[rng.integers(0, n, size=n)])
    return [float(v) for v in np.percentile(estimates, [2.5, 97.5])]


def summarize(x):
    return {
        'median_signed': float(np.median(x)),
        'median_absolute': float(np.median(np.abs(x))),
        'mean_signed': float(np.mean(x)),
        'positive_fraction': float(np.mean(x > 0)),
        'negative_fraction': float(np.mean(x < 0)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--concat', type=Path, default=Path('xai/results/concat_v2_test_counterfactual/pair_predictions.jsonl'))
    p.add_argument('--cross', type=Path, default=Path('xai/results/cross_v2_test_counterfactual/pair_predictions.jsonl'))
    p.add_argument('--out', type=Path, default=Path('xai/results/crossmodal_interactions_v2_test'))
    p.add_argument('--bootstrap', type=int, default=5000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.bootstrap < 100:
        p.error('--bootstrap must be at least 100')
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite {args.out}')

    concat, f1 = read_pairs(args.concat)
    cross, f2 = read_pairs(args.cross)
    if f1 != f2:
        raise ValueError('Models were evaluated on different source families')
    family_rows = []
    summary = []
    for vcat, acat in CONTRASTS:
        values = {}
        for name, lookup in (('concat', concat), ('cross', cross)):
            values[name] = np.array([interaction(lookup, f, vcat, acat) for f in f1], dtype=float)
        diff_abs = np.abs(values['cross']) - np.abs(values['concat'])
        diff_signed = values['cross'] - values['concat']
        for idx, family in enumerate(f1):
            family_rows.append({
                'source_family': family, 'visual_contrast': vcat, 'audio_contrast': acat,
                'concat_interaction': float(values['concat'][idx]),
                'cross_interaction': float(values['cross'][idx]),
                'difference_signed_cross_minus_concat': float(diff_signed[idx]),
                'difference_absolute_cross_minus_concat': float(diff_abs[idx]),
            })
        for name in ('concat', 'cross'):
            x = values[name]
            # Independent fixed seeds for reproducibility, not for statistical independence.
            rng = np.random.default_rng(args.seed + CONTRASTS.index((vcat, acat)) * 10 + (0 if name == 'concat' else 1))
            summary.append({
                'visual_contrast': vcat, 'audio_contrast': acat, 'model': name,
                'n_families': len(f1), **summarize(x),
                'median_signed_ci95': ci_bootstrap(x, np.median, rng, args.bootstrap),
                'median_absolute_ci95': ci_bootstrap(x, lambda y: np.median(np.abs(y)), rng, args.bootstrap),
            })
        rng = np.random.default_rng(args.seed + CONTRASTS.index((vcat, acat)) * 10 + 2)
        summary.append({
            'visual_contrast': vcat, 'audio_contrast': acat,
            'model': 'cross_minus_concat_paired', 'n_families': len(f1),
            'median_signed_difference': float(np.median(diff_signed)),
            'median_absolute_difference': float(np.median(diff_abs)),
            'median_signed_difference_ci95': ci_bootstrap(diff_signed, np.median, rng, args.bootstrap),
            'median_absolute_difference_ci95': ci_bootstrap(diff_abs, np.median, rng, args.bootstrap),
            'fraction_cross_larger_absolute': float(np.mean(diff_abs > 0)),
        })
    args.out.mkdir(parents=True)
    with (args.out / 'family_interactions.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(family_rows[0]))
        writer.writeheader()
        writer.writerows(family_rows)
    (args.out / 'summary.json').write_text(json.dumps({
        'description': 'Logit difference-in-differences, same-family 2x2 counterfactual contrasts',
        'formula': 'f(V1,A1)-f(V1,A0)-f(V0,A1)+f(V0,A0)',
        'baseline_visual': 'real', 'baseline_audio': 'real',
        'n_families': len(f1), 'bootstrap_resamples': args.bootstrap,
        'bootstrap_unit': 'source_family', 'seed': args.seed,
        'interpretation_limit': 'Non-additivity is not proof of audiovisual synchronization or causal interaction.',
        'summary': summary,
    }, indent=2) + '\n')
    print('Contrast (visual, audio) | Concat median | Cross median | Concat median | Cross median | paired median abs difference [95% CI]')
    print('                          signed I        signed I        |I|             |I|')
    for i, (vcat, acat) in enumerate(CONTRASTS):
        c, x, d = summary[3*i:3*i+3]
        lo, hi = d['median_absolute_difference_ci95']
        print(f'{vcat:12s}, {acat:12s} | {c["median_signed"]:+8.3f} | {x["median_signed"]:+8.3f} | '
              f'{c["median_absolute"]:8.3f} | {x["median_absolute"]:8.3f} | '
              f'{d["median_absolute_difference"]:+8.3f} [{lo:+.3f}, {hi:+.3f}]')
    print('Saved:', args.out)


if __name__ == '__main__':
    main()
