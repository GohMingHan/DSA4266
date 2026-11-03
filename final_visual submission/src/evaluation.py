"""Exact loss/AUC/F2/video metrics, plots and separate DEV threshold analysis."""

import csv
import hashlib
import json
import math
import numpy as np
from pathlib import Path
from .data import write_json, stable_seed
from collections import defaultdict


def class_loss_metrics(loss_sums, loss_counts):
    """Return whole-epoch class means; an absent class stays explicitly undefined."""
    if set(loss_sums)!={0,1} or set(loss_counts)!={0,1}:
        raise ValueError('Loss sums and counts must contain real=0 and fake=1')
    if any(not isinstance(loss_counts[label],int) or loss_counts[label]<0 for label in (0,1)):
        raise ValueError('Class loss counts must be nonnegative integers')
    if any(not math.isfinite(loss_sums[label]) or loss_sums[label]<0 for label in (0,1)):
        raise ValueError('Class loss sums must be finite and nonnegative')
    means={label:(loss_sums[label]/loss_counts[label] if loss_counts[label] else None)
           for label in (0,1)}
    balanced=(means[0]+means[1])/2 if all(means[label] is not None for label in (0,1)) else None
    return {'real_class_loss':means[0],'fake_class_loss':means[1],
            'balanced_loss':balanced,
            'real_class_loss_samples':loss_counts[0],
            'fake_class_loss_samples':loss_counts[1]}


def binary_curves(labels, scores):
    """Tie-aware ROC/PR points and areas from saved probabilities."""
    if not labels or len(labels)!=len(scores) or any(y not in (0,1) for y in labels):
        raise ValueError('Nonempty, matching binary labels/scores required')
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError('Finite probabilities in [0,1] required')
    positives=sum(labels); negatives=len(labels)-positives
    ordered=sorted(zip(scores,labels),reverse=True)
    tp=fp=0
    roc_fpr=[0.0]; roc_tpr=[0.0]
    pr_recall=[0.0]; pr_precision=[1.0]
    thresholds=[]; average_precision=0.0; pr_auc=0.0; roc_auc=0.0
    for i in range(len(ordered)):
        score,label=ordered[i]
        tp+=label; fp+=1-label
        if i+1<len(ordered) and ordered[i+1][0]==score:
            continue
        thresholds.append(score)
        recall=tp/positives if positives else 0.0
        precision=tp/(tp+fp)
        if positives:
            average_precision+=(recall-pr_recall[-1])*precision
            pr_auc+=(recall-pr_recall[-1])*(precision+pr_precision[-1])/2
        pr_recall.append(recall); pr_precision.append(precision)
        if positives and negatives:
            fpr=fp/negatives; tpr=recall
            roc_auc+=(fpr-roc_fpr[-1])*(tpr+roc_tpr[-1])/2
            roc_fpr.append(fpr); roc_tpr.append(tpr)
    return {'roc_auc':roc_auc if positives and negatives else None,
            'average_precision':average_precision if positives else None,
            'pr_auc':pr_auc if positives else None,
            'roc':{'fpr':roc_fpr,'tpr':roc_tpr,'thresholds':thresholds} if positives and negatives else None,
            'pr':{'recall':pr_recall,'precision':pr_precision,'thresholds':thresholds} if positives else None}


def binary_metrics(labels,scores,threshold=0.5):
    if not labels or len(labels)!=len(scores) or any(y not in (0,1) for y in labels):
        raise ValueError('Nonempty, matching binary labels/scores required')
    if not 0 <= threshold <= 1 or any(not math.isfinite(p) or not 0<=p<=1 for p in scores):
        raise ValueError('Finite probabilities and threshold in [0,1] required')
    predicted=[int(p>=threshold) for p in scores]
    tp=sum(y==1 and p==1 for y,p in zip(labels,predicted))
    tn=sum(y==0 and p==0 for y,p in zip(labels,predicted))
    fp=sum(y==0 and p==1 for y,p in zip(labels,predicted))
    fn=sum(y==1 and p==0 for y,p in zip(labels,predicted))
    positives=sum(labels); negatives=len(labels)-positives
    curves=binary_curves(labels,scores)
    sensitivity=tp/(tp+fn) if tp+fn else 0.0
    specificity=tn/(tn+fp) if tn+fp else 0.0
    return {'roc_auc':curves['roc_auc'],'average_precision':curves['average_precision'],
            'pr_auc':curves['pr_auc'],
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
            'f2':5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0.0,
            'precision':tp/(tp+fp) if tp+fp else 0.0,'recall':tp/(tp+fn) if tp+fn else 0.0,
            'sensitivity':sensitivity,'specificity':specificity,
            'balanced_accuracy':(sensitivity+specificity)/2,
            'false_positive_rate':fp/negatives if negatives else 0.0,
            'false_negative_rate':fn/positives if positives else 0.0,
            'accuracy':(tp+tn)/len(labels),'threshold':threshold,'n':len(labels),
            'positive':positives,'negative':negatives,'tp':tp,'tn':tn,'fp':fp,'fn':fn,
            'confusion_matrix':[[tn,fp],[fn,tp]]}


def metrics_with_f2(metrics):
    """Non-mutating legacy enrichment; use only this metric set's actual counts/P/R."""
    result = dict(metrics or {})
    if result.get('f2') is not None:
        return result
    if all(result.get(k) is not None for k in ('tp', 'fp', 'fn')):
        tp, fp, fn = (result[k] for k in ('tp', 'fp', 'fn'))
        denominator = 5*tp+fp+4*fn
        result['f2'] = 5*tp/denominator if denominator else 0.
    elif result.get('confusion_matrix') is not None:
        (_, fp), (fn, tp) = result['confusion_matrix']
        denominator = 5*tp+fp+4*fn
        result['f2'] = 5*tp/denominator if denominator else 0.
    elif all(result.get(k) is not None for k in ('precision', 'recall')):
        precision, recall = result['precision'], result['recall']
        denominator = 4*precision+recall
        result['f2'] = 5*precision*recall/denominator if denominator else 0.
    return result


def video_predictions(records, predictions, policy):
    """Aggregate all scheduled predictions, including boundary/ambiguous samples."""
    from collections import defaultdict
    groups=defaultdict(list)
    for prediction in predictions:
        row=records[prediction['record_index']]
        groups[row['source_file']].append((row,prediction['probability']))
    result=[]
    for source,items in sorted(groups.items()):
        probabilities=sorted((p for _,p in items),reverse=True)
        if policy['aggregation']=='topk':
            count=max(1,math.ceil(len(probabilities)*policy['topk_fraction']))
            score=sum(probabilities[:count])/count
        elif policy['aggregation']=='proportion':
            score=sum(p>=policy['prediction_threshold'] for p in probabilities)/len(probabilities)
        else:
            raise ValueError('Unknown aggregation')
        row=items[0][0]
        if len({r['video_label'] for r,_ in items})!=1:
            raise ValueError('Inconsistent video labels')
        result.append({'source_file':source,'source_family':row['source_family'],
                       'target':row['video_label'],'probability':score,'samples':len(items)})
    return result


def evaluate_predictions(records,predictions,policy,kind='frame'):
    if kind not in ('frame','clip'):
        raise ValueError('Expected frame or clip evaluation')
    indices=[p['record_index'] for p in predictions]
    if sorted(indices)!=list(range(len(records))):
        raise ValueError('Evaluation needs exactly one prediction for every scheduled input')
    guarded=[p for p in predictions if records[p['record_index']]['metric_eligible']]
    scored=predictions if kind=='frame' else guarded
    if not scored:
        raise ValueError('No unambiguous samples for unit metrics')
    unit=binary_metrics([records[p['record_index']]['label'] for p in scored],
                        [p['probability'] for p in scored],policy['prediction_threshold'])
    guarded_metrics=(binary_metrics([records[p['record_index']]['label'] for p in guarded],
                        [p['probability'] for p in guarded],policy['prediction_threshold']) if guarded else None)
    videos=video_predictions(records,predictions,policy)
    video=binary_metrics([v['target'] for v in videos],[v['probability'] for v in videos],policy['video_threshold'])
    return {'unit':unit,'unit_scope':'all scheduled frame timestamps' if kind=='frame' else 'eligible clips only',
            'boundary_guarded_unit':guarded_metrics,'video':video,'videos':videos,
            'scheduled_predictions':len(predictions),'metric_eligible_predictions':len(scored),
            'metric_ineligible_predictions':len(predictions)-len(scored),
            'ignored_for_unit_metrics':len(predictions)-len(scored)}


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = prefix + str(key)
        if isinstance(item, dict):
            result.update(flatten(item, name + '.'))
        else:
            result[name] = json.dumps(item) if isinstance(item, list) else item
    return result


def write_history_metrics(out, history):
    """Persist the logically appended history before any figure rendering."""
    out = Path(out)
    # Enrich only the newly written copy, never mutate restored checkpoint history.
    history = [dict(row, **{key: metrics_with_f2(row[key]) for key in ('train', 'dev', 'dev_video') if key in row}) for row in history]
    json_tmp = out/'history.jsonl.tmp'
    json_tmp.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in history), encoding='utf-8')
    json_tmp.replace(out/'history.jsonl')
    rows = [flatten(r) for r in history]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    csv_tmp = out/'history.csv.tmp'
    with csv_tmp.open('w', newline='', encoding='utf-8') as h:
        writer = csv.DictWriter(h, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    csv_tmp.replace(out/'history.csv')


def plot_history(out, history,unit_label='Frame', *, best_epoch=None, selection_metric=None):
    out = Path(out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    epochs = [r['epoch'] for r in history]
    for key, label in (('loss', 'TRAIN (balanced, augmented)'),):
        axes[0, 0].plot(epochs, [r['train'][key] for r in history], label=label)
    axes[0, 0].plot(epochs, [r['dev']['loss'] for r in history], label='DEV natural')
    for key, label in (('balanced_loss', 'Balanced DEV'), ('real_class_loss', 'Real DEV'), ('fake_class_loss', 'Fake DEV')):
        axes[0, 1].plot(epochs, [r['dev'][key] for r in history], label=label)
    axes[1, 0].plot(epochs, [r['train']['roc_auc'] for r in history], label='TRAIN online')
    axes[1, 0].plot(epochs, [r['dev']['roc_auc'] for r in history], label='DEV '+unit_label.lower())
    axes[1, 1].plot(epochs, [r['dev_video']['roc_auc'] for r in history], label='DEV video Top-5%')
    axes[0, 2].plot(epochs, [r['dev'].get('pr_auc') for r in history], label='DEV '+unit_label.lower()+' PR-AUC')
    axes[1, 2].plot(epochs, [r['dev'].get('precision') for r in history], label='DEV precision at 0.5')
    axes[1, 2].plot(epochs, [r['dev'].get('recall') for r in history], label='DEV recall at 0.5')
    if selection_metric in ('frame_roc_auc', 'clip_roc_auc'):
        if best_epoch not in epochs:
            raise ValueError('ROC-selected graphs require the saved selected epoch in the completed history')
        # D/E keep the original six-panel layout and curves; add the requested
        # video PR-AUC and classification lines without changing legacy plots.
        axes[0, 2].plot(epochs, [r['dev_video'].get('pr_auc') for r in history], label='DEV video PR-AUC')
        for level, prefix in (('dev', unit_label), ('dev_video', 'Video')):
            for key in ('precision', 'recall', 'f1', 'f2'):
                if level == 'dev' and key in ('precision', 'recall'): continue
                axes[1, 2].plot(epochs, [metrics_with_f2(r[level]).get(key) for r in history],
                                label=f'DEV {prefix} {key.upper()} at 0.5')
        for ax in axes.flat:
            ax.axvline(best_epoch, color='#d4a000', linestyle='--', label=f'Selected epoch {best_epoch}')
        level = 'clip' if selection_metric == 'clip_roc_auc' else 'frame'
        fig.suptitle(f'Selected by DEV {level} ROC-AUC | best epoch {best_epoch}')
    for i, (ax, title) in enumerate(zip(axes.flat, ('BCE loss', 'Class-aware DEV BCE', 'Frame PR-AUC',
                                                   'Frame ROC-AUC', 'Video ROC-AUC', 'Frame precision / recall'))):
        ax.set_title(title.replace('Frame',unit_label)); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.2)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        if i >= 2:
            ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out / 'training_curves.png', dpi=150); plt.close(fig)


def prediction_artifacts(out, records, predictions, report, render_plots=True,unit_label='Frame'):
    out = Path(out)
    with (out / 'best_dev_predictions.csv').open('w', newline='', encoding='utf-8') as h:
        fields = ['candidate_id', 'source_file', 'source_family', 'timestamp_exact', 'label', 'probability']
        writer = csv.DictWriter(h, fieldnames=fields); writer.writeheader()
        for p in predictions:
            row = records[p['record_index']]
            writer.writerow(dict(zip(fields, (row['id'], row['source_file'], row['source_family'],
                                              row['timestamps_exact'][0], row['label'], p['probability']))))
    write_json(out / 'best_dev_metrics.json', {k: v for k, v in report.items() if k != 'videos'})
    write_json(out / 'best_dev_video_predictions.json', report['videos'])
    curves = {}
    for name, labels, scores in (
        (unit_label.lower(), [p['target'] for p in predictions], [p['probability'] for p in predictions]),
        ('video', [p['target'] for p in report['videos']], [p['probability'] for p in report['videos']])):
        curves[name] = binary_curves(labels, scores)
    write_json(out / 'best_dev_curves.json', curves)
    if not render_plots:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for level, curve in curves.items():
        axes[0].plot(curve['roc']['fpr'], curve['roc']['tpr'], label=level)
        axes[1].plot(curve['pr']['recall'], curve['pr']['precision'], label=level)
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='DEV ROC')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='DEV precision-recall')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out / 'best_dev_roc_pr.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, level in zip(axes, ('unit', 'video')):
        cm = report[level]['confusion_matrix']; ax.imshow(cm, cmap='Blues', vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha='center', va='center')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
               xlabel='Predicted', ylabel='True', title=(unit_label if level == 'unit' else 'Video') + f" threshold {report[level]['threshold']}")
    fig.tight_layout(); fig.savefig(out / 'best_dev_confusion.png', dpi=150); plt.close(fig)


def write_final_numerics(out, history):
    """One final debugging figure; detailed batch records remain in numerics.jsonl."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for key, title, ax in zip(('gradient_norm_max_finite', 'amp_scale_end', 'skipped_steps'),
            ('Max finite pre-clip gradient norm', 'AMP scale', 'Skipped optimizer steps'), axes):
        ax.plot([r['epoch'] for r in history], [r['train'].get('numerics', {}).get(key) for r in history], marker='o')
        ax.set(title=title, xlabel='Epoch', xlim=(0, None), ylim=(0, None)); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(Path(out)/'numerical_curves.png', dpi=120); plt.close(fig)


def dev_threshold_analysis(labels, scores, reference_threshold=.5):
    """O(N log N) DEV-only sweep with ties grouped; no policy or weights changed."""
    y = np.asarray(labels, dtype=int); p = np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Threshold analysis requires finite binary DEV probabilities')
    positives, negatives = int(y.sum()), int(len(y)-y.sum())
    if not positives or not negatives:
        raise ValueError('DEV threshold analysis requires both classes')
    order = np.argsort(-p, kind='stable'); yp, pp = y[order], p[order]
    ends = np.r_[np.flatnonzero(pp[:-1] != pp[1:]), len(p)-1]
    tp_all = np.cumsum(yp); rows = []
    def row(threshold, tp, fp):
        fn, tn = positives-tp, negatives-fp
        precision, recall = tp/(tp+fp) if tp+fp else 0., tp/positives
        return {'threshold': float(threshold), 'precision': precision, 'recall': recall,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'f2': 5*tp/(5*tp+fp+4*fn) if 5*tp+fp+4*fn else 0.,
            'false_positive_rate': fp/negatives, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
    # nextafter includes the valid 'predict none' operating point, even if p=1.
    rows.append(row(np.nextafter(float(pp[0]), float('inf')), 0, 0))
    for end in ends:
        tp = int(tp_all[end]); rows.append(row(pp[end], tp, int(end+1-tp)))
    selected = p >= reference_threshold
    reference = row(reference_threshold, int(y[selected].sum()), int(selected.sum()-y[selected].sum()))
    return {'scope': 'DEV-only descriptive frame threshold analysis; exploratory, optimistic on DEV',
        'does_not_change': ['ROC-AUC', 'PR-AUC', 'BCE loss', 'configured threshold', 'TEST policy'],
        'reference': reference, 'max_f1': max(rows, key=lambda r: r['f1']),
        'max_f2': max(rows, key=lambda r: r['f2']),
        'fpr_tradeoffs': [{'max_fpr': limit, 'operating_point': max(
            (r for r in rows if r['false_positive_rate'] <= limit), key=lambda r: (r['recall'], r['precision']))}
            for limit in (0., .001, .005, .01, .05)],
        'thresholds_evaluated': len(rows), 'n': len(y)}


def family_partition(records,seed=4266):
    groups=defaultdict(list)
    for i,r in enumerate(records):
        if r['split']!='dev' or not r['source_file'].startswith('dev/') or not r['source_family'].startswith('dev/'):
            raise ValueError('Threshold calibration is DEV only')
        groups[r['source_family']].append(i)
    # Stratify by whether a family contains any visually fake scheduled input.
    strata=defaultdict(list)
    for family,ids in groups.items():strata[int(any(records[i]['label'] for i in ids))].append(family)
    calibration=[];reporting=[]
    for families in strata.values():
        families.sort(key=lambda f:stable_seed(seed,'dev_threshold_partition',f))
        for j,family in enumerate(families):
            (calibration if j%2==0 else reporting).extend(groups[family])
    calibration.sort();reporting.sort()
    if any(len({records[i]['label'] for i in ids})!=2 for ids in (calibration,reporting)):
        raise ValueError('Not enough separate DEV families with both classes; no fallback to in-sample threshold reporting')
    return calibration,reporting


def calibrated_f2(records,predictions,seed=4266):
    if sorted(p['record_index'] for p in predictions)!=list(range(len(records))):raise ValueError('Complete DEV predictions required')
    by_index={p['record_index']:p for p in predictions};cal,rep=family_partition(records,seed)
    def values(ids):return [records[i]['label'] for i in ids],[by_index[i]['probability'] for i in ids]
    sweep=dev_threshold_analysis(*values(cal));threshold=sweep['max_f2']['threshold']
    return {'scope':'DEV family-separated calibration/reporting; full DEV was used for checkpoint selection, so reporting is not an untouched final holdout',
        'selection':'maximum calibration F2; equal F2 chooses highest threshold (descending sweep)',
        'seed':seed,'threshold':threshold,'calibration':binary_metrics(*values(cal),threshold),
        'reporting':binary_metrics(*values(rep),threshold),'reporting_at_0_5':binary_metrics(*values(rep),.5),
        'calibration_sweep':sweep,'calibration_families':sorted({records[i]['source_family'] for i in cal}),
        'reporting_families':sorted({records[i]['source_family'] for i in rep}),
        'calibration_ids':[records[i]['id'] for i in cal],'reporting_ids':[records[i]['id'] for i in rep],
        'full_dev_ranking_metrics_unchanged':True,'test_used':False}


def save_calibration(out,records,predictions,seed=4266):
    report=calibrated_f2(records,predictions,seed);out=Path(out)
    write_json(out/'dev_f2_calibration.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(5,4));cm=report['reporting']['confusion_matrix'];ax.imshow(cm,cmap='Blues',vmin=0)
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i][j]),ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Real','Fake'],yticklabels=['Real','Fake'],
        xlabel='Predicted',ylabel='True',title=f"DEV reporting families; F2 threshold {report['threshold']:.4f}")
    fig.tight_layout();fig.savefig(out/'dev_f2_reporting_confusion.png',dpi=150);plt.close(fig)
    return report
