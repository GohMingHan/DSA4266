#!/usr/bin/env python3
"""Compare completed DEV evaluations from the old and TSN-flow pipelines.

Feature identities necessarily differ. Check semantic cohort identities and
per-video score rows instead; never compare on unmatched videos. No ensemble.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def read_rows(path):
    with Path(path).open(encoding='utf-8') as f:rows=[json.loads(s) for s in f if s.strip()]
    result={r['file']:r for r in rows}
    if len(result)!=len(rows):raise ValueError(f'Duplicate file in {path}')
    return result

def cohort(manifest,split):
    m=read_json(manifest)
    keys=('file','split','source_group','modify_video','modify_audio','visual_periods','duration_s')
    return {r['file']:{k:r[k] for k in keys} for r in m['videos'] if r['split']==split}


def compare(baseline,candidate,baseline_manifest,candidate_manifest):
    bm,cm=[read_json(Path(x)/'metrics.json') for x in (baseline,candidate)]
    if bm.get('split')!='dev' or cm.get('split')!='dev':raise ValueError('This selection helper is DEV-only.')
    if bm.get('score_rule')!='maximum_final_proposal_score_or_zero' or cm.get('score_rule')!=bm['score_rule']:
        raise ValueError('Scoring rules differ.')
    bc,cc=cohort(baseline_manifest,'dev'),cohort(candidate_manifest,'dev')
    if bc!=cc:raise ValueError('DEV videos/labels/groups/timing differ. No subset intersection or cherry-picking.')
    if cohort(baseline_manifest,'train')!=cohort(candidate_manifest,'train'):
        raise ValueError('Training cohorts differ.')
    # Verify copied RGB numerically by its recorded original cache provenance.
    bfull=read_json(baseline_manifest);cfull=read_json(candidate_manifest)
    bref={r['file']:r for r in bfull['videos']}
    for r in cfull['videos']:
        if r['split'] in ('train','dev') and r.get('reference_cache_sha256')!=bref[r['file']]['cache']['sha256']:
            raise ValueError(f'Candidate was not built from the matching reference RGB cache: {r["file"]}')
    br,cr=[read_rows(Path(x)/'video_scores.jsonl') for x in (baseline,candidate)]
    if set(br)!=set(cr) or set(br)!=set(bc):raise ValueError('Evaluation IDs do not match full DEV cohort.')
    files=sorted(br);y=[];x=[];z=[]
    for f in files:
        a,b=br[f],cr[f]
        for k in ('visual_label','modify_audio','source_group','split'):
            if a[k]!=b[k]:raise ValueError(f'Mismatched {k} in {f}')
        if a['visual_label']!=int(bc[f]['modify_video']):raise ValueError('Prediction label differs from metadata.')
        if not all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in (a['score'],b['score'])):
            raise ValueError('Proposal probability score must be finite and in [0,1].')
        y.append(a['visual_label']);x.append(a['score']);z.append(b['score'])
    if set(y)!={0,1}:raise ValueError('The full comparison cohort must contain both visual classes.')
    for m,p in ((bm,x),(cm,z)):
        if m.get('videos')!=len(y):raise ValueError('Metric video count does not match the score file.')
        if abs(roc_auc_score(y,p)-m['video_visual_roc_auc'])>1e-8 or abs(average_precision_score(y,p)-m['video_visual_ap'])>1e-8:
            raise ValueError('Saved metrics do not match per-video score files.')
    metrics=('video_visual_roc_auc','video_visual_ap','f1','precision','recall','balanced_accuracy')
    table=[(k,float(bm[k]),float(cm[k]),float(cm[k])-float(bm[k])) for k in metrics]
    for k in ('AP@0.50','AP@0.75','AP@0.95'):
        a=bm['all_cohort_visual_temporal_ap'][k];b=cm['all_cohort_visual_temporal_ap'][k]
        table.append((k,a,b,b-a))
    report={'target':'visual_manipulation','split':'dev','videos':len(files),'training_videos':len(cohort(baseline_manifest,'train')),
            'baseline_checkpoint':bm.get('checkpoint'),'candidate_checkpoint':cm.get('checkpoint'),
            'baseline_feature_kind':bfull['feature_kind'],'candidate_feature_kind':cfull['feature_kind'],
            'candidate_flow_method':cfull['feature_config']['dense_flow']['method'],
            'rgb_reference_cache_match':True,'single_model_comparison_not_ensemble':True,
            'metrics':{k:{'baseline':a,'tsn_flow':b,'tsn_minus_baseline':d} for k,a,b,d in table},
            'limitations':['This helper checks cohort/readout compatibility, NOT independently matched training budgets.',
                          'TSN changes flow encoding, temporal support, input channels and parameter count; not architecture-only causality.',
                          'With TV-L1, the dense-flow estimator also changes. Development selection is not held-out test performance.']}
    return table,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--baseline-manifest',type=Path,required=True);p.add_argument('--candidate-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):p.error('Use a new/empty output directory.')
    table,report=compare(a.baseline,a.candidate,a.baseline_manifest,a.candidate_manifest)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    with (a.out/'comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['metric','baseline','tsn_flow','tsn_minus_baseline']);w.writerows(table)
    print((a.out/'comparison.csv').read_text())

if __name__=='__main__':main()
