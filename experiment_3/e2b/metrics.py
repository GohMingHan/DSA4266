"""Video-level any-manipulation metrics. AUC uses continuous logits."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score


def sigmoid(x):
    x=np.asarray(x,np.float64)
    return np.exp(-np.logaddexp(0.,-x))


def f1_threshold(y,scores):
    y=np.asarray(y,np.int64); s=np.asarray(scores,np.float64)
    if len(np.unique(y))!=2 or not np.isfinite(s).all():raise ValueError('DEV needs two classes and finite scores.')
    order=np.argsort(-s,kind='stable'); y=y[order]; s=s[order]
    ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
    tp=np.cumsum(y)[ends]; predicted=ends+1; total=int(y.sum())
    f=2*tp/(predicted+total)
    # Descending thresholds: choose highest threshold among exact maximum ties.
    return float(s[ends[np.flatnonzero(f==f.max())[0]]])


def binary_metrics(y,logits,threshold):
    y=np.asarray(y,np.int64); z=np.asarray(logits,np.float64)
    if y.shape!=z.shape or not len(y) or not np.isfinite(z).all() or not np.isfinite(threshold):
        raise ValueError('Invalid video predictions.')
    p=z>=threshold; tp=int(np.sum(p&(y==1))); tn=int(np.sum(~p&(y==0)))
    fp=int(np.sum(p&(y==0))); fn=int(np.sum(~p&(y==1)))
    precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.
    fpr=fp/(fp+tn) if fp+tn else None
    both=len(np.unique(y))==2
    return {'video_any_roc_auc':float(roc_auc_score(y,z)) if both else None,
            'video_any_ap':float(average_precision_score(y,z)) if both else None,
            'accuracy':(tp+tn)/len(y),'precision':precision,'recall':recall,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'balanced_accuracy':.5*(recall+tn/(tn+fp)) if both else None,
            'false_positive_rate':fpr,'confusion_matrix_0_1':[[tn,fp],[fn,tp]],
            'videos':len(y),'positives':int(y.sum()),'negatives':int((y==0).sum()),
            'positive_prevalence':float(y.mean()),'threshold_logit':float(threshold),
            'threshold_probability':float(sigmoid(threshold)),
            'score_rule':'fusion_logit (higher = any manipulation); sigmoid is uncalibrated',
            'target':'modify_audio OR modify_video'}


def evaluate_rows(rows,logits,threshold):
    result=binary_metrics([r['label'] for r in rows],logits,threshold); sub={}
    for cat in ('real','audio_only','visual_only','both'):
        ids=[i for i,r in enumerate(rows) if r['category']==cat]
        rate=float(np.mean(np.asarray(logits)[ids]>=threshold)) if ids else None
        sub[cat]={'videos':len(ids), 'false_positive_rate' if cat=='real' else 'recall':rate}
    result['by_category']=sub
    return result
