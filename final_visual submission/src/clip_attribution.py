"""DEV-only representative CLIP explanations; original four-example API is unchanged.

Positive Fake-logit Grad-CAM is reproduced exactly for comparison. The alternative
is signed layer gradient x activation for the Fake-minus-Real logit (log odds),
summed independently within each patch token. Neither is a face segmentation or
a causal/completeness guarantee. No hooks, parameter gradients or weights persist.
"""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import base64
import csv
import hashlib
import html
import io
import json
import math
import numpy as np
import torch
from torch.nn import functional as F
from .clip_model import CLIP_SEED, MODEL_ID, checkpoint_components, fake_probability
from .clip_workflow import validate_clip_checkpoint
from .checkpoints import digest
from .data import DevSchedule, FrameReader, stable_seed, write_json
from .workflow import guard_dev, new_id

CATEGORIES = ('tp', 'tn', 'fp', 'fn')
BANDS = ('high', 'medium', 'borderline')
BASELINE = 'fake_logit_gradcam'
SIGNED = 'margin_patch_gradient_x_activation'


def select_examples(records, predictions, per_category=10, seed=CLIP_SEED):
    """Round-robin confidence strata, preferring unused videos within each stratum.

    Confidence=max(p,1-p): high>=.90, medium .65-.90, borderline .50-.65.
    Repeated-video frames prefer wider timestamp separation. Selection never sees
    attributions. Saved threshold .5 and visual FRAME labels define categories.
    """
    if type(per_category) is not int or not 1 <= per_category <= 10:
        raise ValueError('per_category must be an integer from1 to10')
    if seed != CLIP_SEED:raise ValueError('CLIP example selection requires seed4266')
    if sorted(p['record_index'] for p in predictions) != list(range(len(records))):
        raise ValueError('Complete, unique saved DEV frame predictions required')
    groups = {c: {b: [] for b in BANDS} for c in CATEGORIES}
    for p in predictions:
        row = records[p['record_index']];prob = float(p['probability'])
        if row['split'] != 'dev' or row['kind'] != 'frame':raise PermissionError('DEV frames only')
        if row['label'] not in (0, 1) or p['target'] != row['label']:raise ValueError('Saved/frame target mismatch')
        if not math.isfinite(prob) or not 0 <= prob <= 1:raise ValueError('Invalid saved probability')
        category = ('tp' if prob >= .5 else 'fn') if row['label'] else ('fp' if prob >= .5 else 'tn')
        confidence = max(prob, 1-prob)
        band = 'high' if confidence >= .9 else 'medium' if confidence >= .65 else 'borderline'
        groups[category][band].append(dict(p, category=category, confidence=confidence, confidence_band=band))
    selected = [];counts = {c: {b: len(groups[c][b]) for b in BANDS} for c in CATEGORIES}
    for category in CATEGORIES:
        used = Counter();times = {};chosen = []
        while len(chosen) < per_category and any(groups[category].values()):
            for band in BANDS:
                items = groups[category][band]
                if not items or len(chosen) >= per_category:continue
                def rank(p):
                    r = records[p['record_index']];video = r['source_file'];t = Fraction(r['timestamps_exact'][0])
                    gap = min((abs(t-v) for v in times.get(video, [])), default=Fraction(0))
                    confidence_rank = -p['confidence'] if band == 'high' else abs(p['confidence']-(.775 if band == 'medium' else .5))
                    return used[video], -gap, confidence_rank, stable_seed(seed, category, r['id'])
                p = min(items, key=rank);items.remove(p);chosen.append(p)
                row = records[p['record_index']];video = row['source_file'];used[video] += 1
                times.setdefault(video, []).append(Fraction(row['timestamps_exact'][0]))
        selected.extend(chosen)
    return selected, counts


def patch_attributions(activation, fake_gradient, margin_gradient):
    """Pure token formulas, including exact baseline scaling and honest zero maps."""
    activation, fake_gradient, margin_gradient = (t.detach() for t in (activation, fake_gradient, margin_gradient))
    if activation.ndim != 3 or activation.shape != fake_gradient.shape or activation.shape != margin_gradient.shape:
        raise ValueError('Expected matching [batch,CLS+patches,channels] tensors')
    if not all(bool(torch.isfinite(t).all()) for t in (activation, fake_gradient, margin_gradient)):
        raise FloatingPointError('Non-finite CLIP attribution')
    a, g, m = activation[:, 1:, :], fake_gradient[:, 1:, :], margin_gradient[:, 1:, :]
    grid = math.isqrt(a.shape[1])
    if grid < 1 or grid*grid != a.shape[1]:raise ValueError('Patch tokens do not form a square')
    raw = (a*g.mean(dim=1, keepdim=True)).sum(dim=-1)
    signed_raw = (a*m).sum(dim=-1)  # No spatial averaging or ReLU.
    if not all(bool(torch.isfinite(t).all()) for t in (raw, signed_raw)):
        raise FloatingPointError('Non-finite attribution products; map was not replaced')
    def upsample(values):
        return F.interpolate(values.reshape(-1, 1, grid, grid), size=(224, 224), mode='bilinear', align_corners=False)[:, 0]
    positive = upsample(raw.relu());maxima = positive.flatten(1).amax(1).view(-1, 1, 1)
    baseline = torch.where(maxima > 0, positive/maxima.clamp_min(1e-8), torch.zeros_like(positive))
    signed_pixels = upsample(signed_raw)
    # Scale by ORIGINAL patch extrema, retaining signs and unmodified raw arrays.
    scales = signed_raw.abs().amax(1).view(-1, 1, 1)
    signed = torch.where(scales > 0, signed_pixels/torch.where(scales > 0, scales, torch.ones_like(scales)), torch.zeros_like(signed_pixels))
    diagnostics = []
    for i in range(len(a)):
        zero = bool((positive[i] == 0).all());gradient_zero = bool((g[i] == 0).all())
        reason = 'not_zero'
        if zero:
            reason = ('zero_patch_gradients' if gradient_zero else 'zero_patch_activations' if bool((a[i] == 0).all())
                      else 'nonpositive_scores_removed_by_relu' if bool((raw[i] < 0).any()) else 'weighted_sum_exactly_zero')
        per_patch_fake = (a[i]*g[i]).sum(-1)
        diagnostics.append({'baseline_zero': zero, 'baseline_zero_reason': reason,
            'cls_fake_gradient_l1': float(fake_gradient[i, 0].abs().sum()),
            'cls_margin_gradient_l1': float(margin_gradient[i, 0].abs().sum()),
            'fake_patch_gradient_l1': float(g[i].abs().sum()), 'mean_fake_gradient_l1': float(g[i].mean(0).abs().sum()),
            'mean_gradient_cancellation_ratio': float(g[i].mean(0).abs().sum()/g[i].abs().mean(0).sum()) if not gradient_zero else 0.,
            'activation_l1': float(a[i].abs().sum()), 'baseline_raw_min': float(raw[i].min()), 'baseline_raw_max': float(raw[i].max()),
            'baseline_positive_patches': int((raw[i] > 0).sum()), 'baseline_negative_patches': int((raw[i] < 0).sum()),
            'fake_per_patch_positive_count': int((per_patch_fake > 0).sum()),
            'signed_zero': bool((signed_raw[i] == 0).all()), 'signed_raw_min': float(signed_raw[i].min()),
            'signed_raw_max': float(signed_raw[i].max()), 'signed_absmax': float(scales[i].item()),
            'signed_positive_patches': int((signed_raw[i] > 0).sum()), 'signed_negative_patches': int((signed_raw[i] < 0).sum()),
            'weak_baseline_signal': bool(maxima[i].item() < 1e-8), 'patch_grid': grid})
    return {'baseline': baseline.detach().cpu().numpy(), 'signed': signed.detach().cpu().numpy(),
            'baseline_raw_patches': raw.detach().cpu().numpy().reshape(-1, grid, grid),
            'signed_raw_patches': signed_raw.detach().cpu().numpy().reshape(-1, grid, grid), 'diagnostics': diagnostics}


class CLIPPatchAttribution:
    def __init__(self, model):
        self.model = model
        self.target_layer = model.encoder.vision_model.encoder.layers[-2].layer_norm1

    def generate(self, pixel_values):
        captured = [];states = [(m, m.training) for m in self.model.modules()]
        handle = self.target_layer.register_forward_hook(lambda m, i, o: captured.append(o))
        try:
            self.model.eval()
            with torch.inference_mode(False), torch.enable_grad(), torch.autocast(pixel_values.device.type, enabled=False):
                logits = self.model(pixel_values.detach().clone().float().requires_grad_(True))
                if len(captured) != 1 or logits.shape != (len(pixel_values), 2):raise ValueError('Unexpected CLIP activation/logits')
                a = captured[0]
                fake_g, = torch.autograd.grad(logits[:, 0].sum(), a, retain_graph=True)
                margin_g, = torch.autograd.grad((logits[:, 0]-logits[:, 1]).sum(), a)
                if not bool(torch.isfinite(logits).all()):raise FloatingPointError('Non-finite CLIP logits')
                result = patch_attributions(a, fake_g, margin_g)
                result.update(probabilities=fake_probability(logits).detach().cpu().numpy(), logits=logits.detach().cpu().numpy())
                return result
        finally:
            handle.remove();captured.clear()
            for module, state in states:module.training = state


def _weight_hash(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode());value.update(str(tensor.dtype).encode());value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def _save_figure(path, rgb, result, metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    heat, signed = result['baseline'][0], result['signed'][0]
    positive_color = plt.get_cmap('magma')(heat)[..., :3]
    signed_color = plt.get_cmap('RdBu_r')((signed+1)/2)[..., :3]
    def overlay(color, alpha):return np.clip((1-alpha[..., None])*rgb/255.+alpha[..., None]*color, 0, 1)
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(rgb);axes[0].set_title('Original indexed DEV frame')
    im = axes[1].imshow(heat, cmap='magma', vmin=0, vmax=1);axes[1].set_title('Original Fake-logit Grad-CAM\npositive only; ReLU')
    fig.colorbar(im, ax=axes[1], fraction=.046, label='Relative positive attribution [0,1]')
    axes[2].imshow(overlay(positive_color, .45*heat));axes[2].set_title('Grad-CAM overlay\nzero = original pixels')
    im = axes[3].imshow(signed, cmap='RdBu_r', vmin=-1, vmax=1);axes[3].set_title('Signed patch gradient x activation\nFake minus Real logit')
    fig.colorbar(im, ax=axes[3], fraction=.046, label='Blue: negative / red: positive [-1,1]')
    axes[4].imshow(overlay(signed_color, .45*np.abs(signed)));axes[4].set_title('Signed overlay\nzero = original pixels')
    for ax in axes:ax.axis('off')
    d = result['diagnostics'][0]
    fig.suptitle(f"{metadata['category'].upper()} / {metadata['confidence_band']} | GT={metadata['ground_truth']} | predicted={metadata['predicted_label']} | saved Fake p={metadata['saved_probability']:.6f} / FP32={metadata['fake_probability']:.6f}\n"
        f"video={metadata['video_id']} | {metadata['source_file']} | exact t={metadata['timestamp_exact']} s | frame index={metadata['source_index']}\n"
        f"{d['patch_grid']}x{d['patch_grid']} coarse patches | zero Grad-CAM: {d['baseline_zero_reason']} | raw Grad-CAM max={d['baseline_raw_max']:.3g}; signed absmax={d['signed_absmax']:.3g}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .81));fig.savefig(path, dpi=120);plt.close(fig)


def _gallery(destination, examples):
    from PIL import Image
    chunks = ['<!doctype html><html><head><meta charset="utf-8"><title>CLIP DEV attribution gallery</title>',
        '<style>body{font:16px sans-serif;margin:20px}article{border:1px solid #aaa;padding:10px;margin:12px 0}img{max-width:100%;height:auto}.scroll{max-height:760px;overflow:auto}nav a{margin-right:20px}</style></head><body>',
        '<h1>CLIP DEV representative attribution gallery</h1><p>Seed4266; saved frame predictions at threshold0.5. Selection precedes attribution. All selected examples, including zero maps, are retained.</p>',
        '<p>Grad-CAM: positive Fake-logit attribution only. Signed patch attribution: red positive / blue negative for Fake-minus-Real logit; white zero. Scales are relative per example, not comparable causal effect sizes. 7x7 patches are not face boundaries.</p>',
        '<nav>'+''.join(f'<a href="#{c}">{c.upper()}</a>' for c in CATEGORIES)+'</nav>']
    for category in CATEGORIES:
        items = [r for r in examples if r['category'] == category]
        chunks.append(f'<h2 id="{category}">{category.upper()} ({len(items)})</h2><div class="scroll">')
        if not items:chunks.append('<p>No saved DEV examples in this category. No replacements invented.</p>')
        for row in items:
            with Image.open(destination/row['figure']) as image:
                image.thumbnail((1200, 360));buffer = io.BytesIO();image.save(buffer, format='PNG')
            data = base64.b64encode(buffer.getvalue()).decode()
            caption = f"{row['example_id']} | {row['confidence_band']} | GT={row['ground_truth']} predicted={row['predicted_label']} Fake p={row['saved_probability']:.6f} | video={row['video_id']} t={row['timestamp_exact']} | zero={row['baseline_zero_reason']}"
            chunks.append('<article><p>'+html.escape(caption)+'</p><a href="'+html.escape(row['figure'], quote=True)+'"><img loading="lazy" alt="'+html.escape(caption, quote=True)+'" src="data:image/png;base64,'+data+'"></a></article>')
        chunks.append('</div>')
    chunks.append('</body></html>');(destination/'gallery.html').write_text('\n'.join(chunks), encoding='utf-8')


def explain_clip_dev_gallery(run, inputs, project, weights=None, *, split='dev', per_category=10, seed=CLIP_SEED, cache=None, device=None):
    """Only selected frames are inferred; best checkpoint/full saved DEV predictions required."""
    guard_dev(split)  # Before checkpoint, index or source reads.
    if type(per_category) is not int or not 1 <= per_category <= 10:raise ValueError('per_category must be1..10')
    if seed != CLIP_SEED:raise ValueError('Use seed4266')
    run = Path(run);project = Path(project);checkpoint = run/'best.pt'
    summary = json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
    checkpoint_hash = digest(checkpoint)
    state = validate_clip_checkpoint(checkpoint, project, index_hash=digest(inputs.index), pool_hash=digest(inputs.pool))
    if (state['epoch']+1 != state['best_epoch'] or state['best_epoch'] != summary['best_epoch'] or
        state['config']['threshold'] != .5 or state['config']['checkpoint_selection'] != 'frame_roc_auc'):
        raise ValueError('Expected the actual selected best checkpoint, fixed threshold0.5 and frame ROC-AUC selection')
    index = DevSchedule(inputs.index, inputs.data)
    try:records = index.records('dev', 'frame')
    finally:index.close()
    selected, available = select_examples(records, state['dev_predictions'], per_category, seed)
    destination = project/'results'/MODEL_ID/'attribution_dev'/new_id();destination.mkdir(parents=True, exist_ok=False)
    manifest = {'seed': seed, 'per_category': per_category, 'threshold': .5, 'available_by_category_band': available,
        'selected': selected, 'checkpoint_sha256': checkpoint_hash, 'best_epoch': state['best_epoch'],
        'checkpoint_selection': 'frame_roc_auc', 'index_sha256': digest(inputs.index), 'pool_sha256': digest(inputs.pool),
        'target_layer': 'encoder.vision_model.encoder.layers[-2].layer_norm1; CLS excluded',
        'methods': {BASELINE: 'ReLU(sum_channels(activation * mean_patch(Fake-logit gradient))); original scaling',
                    SIGNED: 'sum_channels(activation * per-patch gradient(Fake logit - Real logit)); no ReLU'},
        'selection': 'Round-robin high>=.90 / medium>=.65 / borderline<.65 predicted-class confidence; video diversity and temporal spacing; no attribution-based filtering',
        'status': 'started', 'test_used': False}
    write_json(destination/'provenance.json', manifest)  # Auditable selection saved BEFORE rendering.
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.random.fork_rng(devices=[]):model, processor = checkpoint_components(state, weights)
    model = model.to(device).eval();before = _weight_hash(model);engine = CLIPPatchAttribution(model)
    if cache is not None:
        from .frame_cache import CachedReader
        reader = CachedReader(inputs.data, cache)
    else:reader = FrameReader(inputs.data, 0)
    examples = [];csv_rows = []
    try:
        from PIL import Image
        for number, prediction in enumerate(selected, 1):
            row = records[prediction['record_index']];rgb = reader.read([row])[0][0]
            pixels = processor(images=[Image.fromarray(rgb)], return_tensors='pt')['pixel_values'].to(device)
            result = engine.generate(pixels);probability = float(result['probabilities'][0]);difference = abs(probability-prediction['probability'])
            if difference > .01:raise ValueError('Attribution inference differs from saved best DEV probability by more than0.01')
            example_id = prediction['category']+'_'+str(row['id']);figure = example_id+'.png'
            metadata = {'example_id': example_id, 'category': prediction['category'], 'confidence_band': prediction['confidence_band'],
                'confidence': prediction['confidence'], 'candidate_id': row['id'], 'record_index': prediction['record_index'],
                'video_id': row.get('video_id', row['source_file']), 'source_file': row['source_file'], 'source_family': row.get('source_family', ''),
                'source_index': row['indices'][0], 'timestamp_exact': row['timestamps_exact'][0], 'label': row['label'],
                'ground_truth': 'Fake' if row['label'] else 'Real', 'predicted_label': 'Fake' if prediction['probability'] >= .5 else 'Real',
                'threshold': .5, 'saved_probability': prediction['probability'], 'fake_probability': probability, 'probability_difference': difference,
                'fp32_decision_agrees': (probability >= .5) == (prediction['probability'] >= .5),
                'fake_logit': float(result['logits'][0, 0]), 'real_logit': float(result['logits'][0, 1]),
                'checkpoint': 'best.pt', 'checkpoint_sha256': checkpoint_hash, 'best_epoch': state['best_epoch'],
                'run_id': state['experiment_identity']['run_id'], 'figure': figure, 'arrays': example_id+'.npz', **result['diagnostics'][0]}
            _save_figure(destination/figure, rgb, result, metadata)
            np.savez_compressed(destination/metadata['arrays'], baseline=result['baseline'][0], signed=result['signed'][0],
                                baseline_raw_patches=result['baseline_raw_patches'][0], signed_raw_patches=result['signed_raw_patches'][0])
            examples.append(metadata)
            for method, target, zero, scale in [(BASELINE, 'Fake logit0', metadata['baseline_zero'], 'positive [0,1]; original normalization'),
                                                (SIGNED, 'Fake logit0 - Real logit1', metadata['signed_zero'], 'signed [-1,1]; raw patch absmax')]:
                csv_rows.append(metadata | {'method': method, 'target': target, 'zero_map': zero, 'display_scale': scale})
            print(f"CLIP DEV attribution {number}/{len(selected)} | {example_id} {prediction['confidence_band']} | zero CAM={metadata['baseline_zero_reason']}", flush=True)
        with (destination/'examples.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]) if csv_rows else ['example_id', 'method']);writer.writeheader();writer.writerows(csv_rows)
        _gallery(destination, examples)
        manifest.update(status='complete', examples=len(examples), method_rows=len(csv_rows),
            selected_video_counts={c: len({r['source_file'] for r in examples if r['category'] == c}) for c in CATEGORIES},
            zero_counts={m: sum(r['method'] == m and r['zero_map'] for r in csv_rows) for m in (BASELINE, SIGNED)})
    except BaseException as error:
        manifest.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        unchanged = _weight_hash(model) == before and digest(checkpoint) == checkpoint_hash
        manifest['weights_and_checkpoint_unchanged'] = unchanged
        write_json(destination/'provenance.json', manifest)
        del model
        from .clip_reporting import gc_collect
        gc_collect()
        if not unchanged:raise RuntimeError('Attribution changed model/checkpoint state')
    return destination
