"""Paired comparison of one-head alternatives. Never ensembles predictions."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from e2b.metrics import evaluate_rows
from .common import VERSION,TARGET,read_json,read_jsonl,write_json,require_new

FIELDS=('video_any_roc_auc','video_any_ap','accuracy','balanced_accuracy','f1','precision','recall','false_positive_rate')

def comparable_ids(report):
    if report.get('format')==VERSION:
        return report['e2b_evaluated_split_id'],report['e2b_train_dev_split_ids']
    if report.get('pooling')!='mean':
        raise ValueError('External baseline must be the audited Experiment 2B mean-pooling baseline, not a native audio/visual score.')
    return report['evaluated_split_id'],report['train_dev_split_ids']

def compare(paths,out,bootstrap=0,seed=42,allow_training_differences=False):
    if len(paths)<2 or len(set(map(str,paths)))!=len(paths):raise ValueError('Supply >=2 distinct evaluation folders.')
    roots=[Path(p).expanduser().resolve() for p in paths]
    reports=[read_json(p/'metrics.json') for p in roots]
    if any(r.get('target')!=TARGET for r in reports):raise ValueError('Cannot compare native audio/visual scores with any-manipulation scores.')
    for key in ('split','encoder_ids','test_only'):
        if any(r.get(key)!=reports[0].get(key) for r in reports):raise ValueError(f'Mismatch in {key}.')
    if any(comparable_ids(r)!=comparable_ids(reports[0]) for r in reports):
        raise ValueError('Pooled inputs, train/dev identity or evaluated cohort differ.')
    if all(r.get('format')==VERSION for r in reports):
        for key in ('evaluated_split_id','train_dev_split_ids','normalization','temporal_policy'):
            if any(r.get(key)!=reports[0].get(key) for r in reports):raise ValueError(f'Different sequences or preparation: {key}.')
    settings=[r.get('settings') for r in reports]
    differences={}
    if all(s is not None for s in settings):
        fields=('batch_size','epochs','patience','lr','weight_decay','seed','balance','dropout','head_width')
        for field in fields:
            vals=[s.get(field) for s in settings]
            if any(v!=vals[0] for v in vals):differences[field]=vals
        attention=[i for i,r in enumerate(reports) if r['mode'] in ('cross','self')]
        if len(attention)>1:
            for field in ('width','heads','layers','ff_multiplier','time_encoding','attention_backend'):
                vals=[settings[i][field] for i in attention]
                if any(v!=vals[0] for v in vals):differences[field]=vals
            counts=[reports[i]['parameters'] for i in attention]
            if any(v!=counts[0] for v in counts):differences['attention_parameters']=counts
    else:
        differences['external_training_settings']='External Experiment 2B evaluation does not include all training settings; inspect its run_config.json.'
    if differences and not allow_training_differences:
        raise ValueError(f'Nonmatched/unverified training settings: {differences}. For a disclosed cross-experiment comparison only, use --allow-training-differences.')
    predictions=[]
    for root,report in zip(roots,reports):
        rows=read_jsonl(root/'video_predictions.jsonl');mapping={r['file']:r for r in rows}
        if len(rows)!=len(mapping):raise ValueError('Duplicate prediction IDs.')
        for r in rows:
            if r['label']!=int(r['modify_audio'] or r['modify_video']):raise ValueError('Wrong any-fake label.')
        recomputed=evaluate_rows(rows,[r['logit'] for r in rows],report['threshold_logit'])
        for key in FIELDS:
            if recomputed[key] is None or report[key] is None:
                if recomputed[key]!=report[key]:raise ValueError('Metrics disagree with prediction records.')
            elif not np.isclose(recomputed[key],report[key],rtol=0,atol=1e-6):
                raise ValueError(f'{root}: {key} disagrees with per-video scores.')
        predictions.append(mapping)
    ids=sorted(predictions[0])
    for pred in predictions:
        if sorted(pred)!=ids:raise ValueError('Different video cohorts.')
        for name in ids:
            for field in ('label','split','source_family','category','modify_audio','modify_video'):
                if pred[name][field]!=predictions[0][name][field]:raise ValueError('Different per-video labels/source groups.')
    summary={'format':'experiment_3_comparison_v1','runs':[str(p) for p in roots],
             'metrics':reports,'training_setting_differences':differences,'bootstrap':None,
             'note':'DEV selection evidence, not guaranteed generalization. No ensemble, score inversion or threshold retuning.'}
    if bootstrap:
        if len(roots)!=2:raise ValueError('Paired bootstrap needs exactly two runs.')
        family={}
        for i,k in enumerate(ids):family.setdefault(predictions[0][k]['source_family'],[]).append(i)
        groups=list(family.values());y=np.array([predictions[0][k]['label'] for k in ids]);rng=np.random.default_rng(seed)
        z=[np.array([p[k]['logit'] for k in ids]) for p in predictions];values=[];skips=0
        for _ in range(bootstrap):
            ind=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
            if len(np.unique(y[ind]))<2:skips+=1;continue
            values.append([roc_auc_score(y[ind],z[1][ind])-roc_auc_score(y[ind],z[0][ind]),
                           average_precision_score(y[ind],z[1][ind])-average_precision_score(y[ind],z[0][ind])])
        summary['bootstrap']={'direction':'second minus first','source_families':len(groups),
            'valid_replicates':len(values),'skipped_one_class_replicates':skips,
            'auc_difference_95_percentile_interval':np.quantile(np.array(values)[:,0],[.025,.975]).tolist() if values else None,
            'ap_difference_95_percentile_interval':np.quantile(np.array(values)[:,1],[.025,.975]).tolist() if values else None,
            'note':'Paired source-family resampling; conditional on fitted selected checkpoints. Not corrected for DEV search or seed uncertainty.'}
    output=Path(out).expanduser().resolve();require_new(output)
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['metric']+[r['mode']+' : '+p.name for r,p in zip(reports,roots)]+(['second_minus_first'] if len(roots)==2 else []))
        for field in FIELDS:
            vals=[r[field] for r in reports]
            w.writerow([field]+vals+([vals[1]-vals[0] if all(v is not None for v in vals) else None] if len(vals)==2 else []))
    write_json(output/'comparison.json',summary)
    print(f'Comparison saved: {output}/comparison.csv');return summary
