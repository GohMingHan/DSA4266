"""Visual video classification and single-class temporal AP on ALL evaluated videos.

Independent evaluation code. Not a claim of bit-for-bit parity with the paper's
benchmark evaluator. Predictions on visually negative videos count as false positives.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score)


def choose_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    if not len(thresholds):
        raise ValueError('No development-set threshold candidates.')
    best = np.flatnonzero(np.isclose(f1, f1.max(), atol=1e-12, rtol=0))[-1]
    return float(thresholds[best])


def video_metrics(rows, threshold):
    y = np.asarray([int(r['visual_label']) for r in rows])
    scores = np.asarray([r['score'] for r in rows], dtype=np.float64)
    if not len(y) or not np.isfinite(scores).all() or set(y.tolist()) != {0, 1}:
        raise ValueError('Video metrics need finite scores and both visual classes.')
    prediction = (scores >= threshold).astype(int)
    return {'video_visual_roc_auc': float(roc_auc_score(y, scores)),
            'video_visual_ap': float(average_precision_score(y, scores)),
            'threshold': float(threshold), 'threshold_rule': 'dev_max_f1_highest_tie',
            'f1': float(f1_score(y, prediction, zero_division=0)),
            'precision': float(precision_score(y, prediction, zero_division=0)),
            'recall': float(recall_score(y, prediction, zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
            'confusion_matrix_0_1': confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
            'videos': len(rows), 'visual_positives': int(y.sum()),
            'visual_negatives': int((y == 0).sum()), 'score_rule': 'maximum_final_proposal_score_or_zero'}


def temporal_iou(segment, ground_truth):
    g = np.asarray(ground_truth, dtype=np.float64).reshape(-1, 2)
    if not len(g):
        return np.empty(0)
    a, b = map(float, segment)
    inter = np.maximum(0, np.minimum(b, g[:, 1]) - np.maximum(a, g[:, 0]))
    return inter / np.maximum((b - a) + (g[:, 1] - g[:, 0]) - inter, 1e-12)


def interpolated_ap(tp, fp, n_gt):
    if n_gt <= 0:
        return None
    if not len(tp):
        return 0.0
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.r_[0., recall, 1.]
    precision = np.r_[0., precision, 0.]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changing = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changing + 1] - recall[changing]) * precision[changing + 1]))


def temporal_metrics(rows, predictions, thresholds=(.5, .75, .95)):
    gt = {r['file']: r['visual_periods'] for r in rows}
    if set(gt) != set(predictions):
        raise ValueError('Every evaluated video needs an explicit prediction list, including empty ones.')
    n_gt = sum(map(len, gt.values()))
    ranked = []
    for file in sorted(predictions):
        for order, p in enumerate(predictions[file]):
            s, e = p['segment']
            if not np.isfinite([s, e, p['score']]).all() or e <= s:
                raise ValueError(f'Invalid proposal for {file}: {p}')
            ranked.append((float(p['score']), file, order, [s, e]))
    ranked.sort(key=lambda x: (-x[0], x[1], x[2]))
    aps = {}
    for threshold in thresholds:
        assigned = {f: np.zeros(len(segments), bool) for f, segments in gt.items()}
        tp, fp = [], []
        for score, file, order, segment in ranked:
            overlap = temporal_iou(segment, gt[file])
            eligible = np.where((overlap >= threshold) & ~assigned[file])[0]
            matched = bool(len(eligible))
            if matched:
                chosen = eligible[np.argmax(overlap[eligible])]
                assigned[file][chosen] = True
            tp.append(int(matched)); fp.append(int(not matched))
        aps[f'AP@{threshold:.2f}'] = interpolated_ap(tp, fp, n_gt)
    negatives = [f for f, segments in gt.items() if not segments]
    return {'all_cohort_visual_temporal_ap': aps, 'ground_truth_segments': n_gt,
            'proposals': len(ranked), 'negative_videos': len(negatives),
            'proposals_on_negative_videos': sum(len(predictions[f]) for f in negatives),
            'evaluation_scope': 'all videos; audio-only and genuine videos have empty visual ground truth',
            'interpolation': 'continuous precision envelope; one-to-one matching at each tIoU'}
