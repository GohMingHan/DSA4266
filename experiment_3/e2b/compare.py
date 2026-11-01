"""Compare cached video predictions from matched Experiment 2B controls."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from .io import read_json,read_jsonl,write_json,require_new


def compare(paths,out,bootstrap=0,seed=42):
    roots=[Path(p).resolve() for p in paths]
    if len(roots)<2:raise ValueError('Provide at least two evaluation directories.')
    reports=[read_json(r/'metrics.json') for r in roots]
    names=[str(r) for r in roots]
    for key in ('target','split','evaluated_split_id','train_dev_split_ids','encoder_ids','pooling'):
        if any(r.get(key)!=reports[0].get(key) for r in reports):raise ValueError(f'Cannot compare unmatched {key}.')
    preds=[]
    for r in roots:
        rows=read_jsonl(r/'video_predictions.jsonl'); mapping={x['file']:x for x in rows}
        if len(mapping)!=len(rows):raise ValueError('Duplicate prediction IDs.')
        preds.append(mapping)
    ids=sorted(preds[0])
    for pred in preds:
        if sorted(pred)!=ids:raise ValueError('Different video cohorts.')
        for k in ids:
            if any(pred[k][field]!=preds[0][k][field] for field in ('label','source_family','split')):
                raise ValueError('Paired prediction targets or source groups differ.')
    output=Path(out).resolve();require_new(output)
    fields=('video_any_roc_auc','video_any_ap','accuracy','balanced_accuracy','f1','precision','recall','false_positive_rate')
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['metric']+[r['mode']+' : '+p.name for r,p in zip(reports,roots)])
        for key in fields:w.writerow([key]+[r[key] for r in reports])
    summary={'runs':names,'metrics':reports,'bootstrap':None,
             'note':'Controls predict any manipulation, not the original audio-native or visual-native labels.'}
    if bootstrap:
        if len(roots)!=2:raise ValueError('Paired AUC bootstrap currently expects exactly two runs.')
        families={}
        for i,k in enumerate(ids):families.setdefault(preds[0][k]['source_family'],[]).append(i)
        groups=list(families.values());y=np.array([preds[0][k]['label'] for k in ids])
        scores=[np.array([p[k]['logit'] for k in ids]) for p in preds]
        rng=np.random.default_rng(seed); diffs=[];skips=0
        for _ in range(bootstrap):
            indices=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
            if len(np.unique(y[indices]))!=2:skips+=1;continue
            diffs.append(roc_auc_score(y[indices],scores[1][indices])-roc_auc_score(y[indices],scores[0][indices]))
        summary['bootstrap']={'direction':'second minus first','source_families':len(groups),'valid_replicates':len(diffs),
            'skipped_one_class_replicates':skips,'auc_difference_95_percentile_interval':np.quantile(diffs,[.025,.975]).tolist() if diffs else None,
            'note':'Conditional on fitted checkpoints; does not account for model selection or training-seed variability.'}
    write_json(output/'comparison.json',summary)
    print(f'Comparison: {output}/comparison.csv');return summary
