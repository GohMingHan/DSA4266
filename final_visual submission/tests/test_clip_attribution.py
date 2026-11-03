"""Offline mathematical, selection and export checks; no trained data/downloads."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src import clip_attribution as explain
from src.clip_model import CLIPViTGradCAM
from src.workflow import Inputs
from test_clip import tiny_model


def examples(count=12):
    rows = [];predictions = []
    for category, y in [('tp', 1), ('tn', 0), ('fp', 0), ('fn', 1)]:
        for j in range(count):
            confidence = [.98, .78, .52][j % 3]
            prob = confidence if category in ('tp', 'fp') else 1-confidence
            i = len(rows)
            rows.append({'id': i, 'video_id': i, 'kind': 'frame', 'split': 'dev', 'label': y,
                'source_file': f'dev/{category}_{j}.mp4', 'source_family': f'dev/family_{j}',
                'indices': [j], 'timestamps_exact': [str(j)+'/10']})
            predictions.append({'record_index': i, 'target': y, 'probability': prob})
    return rows, predictions


class RepresentativeSelection(unittest.TestCase):
    def test_ten_per_category_strata_reproducible_unique_and_video_diverse(self):
        rows, predictions = examples()
        selected, available = explain.select_examples(rows, predictions)
        again, _ = explain.select_examples(rows, list(reversed(predictions)))
        self.assertEqual(selected, again);self.assertEqual(len(selected), 40)
        self.assertEqual(len({p['record_index'] for p in selected}), 40)
        for category in explain.CATEGORIES:
            items = [p for p in selected if p['category'] == category]
            self.assertEqual(len(items), 10)
            self.assertEqual({p['confidence_band'] for p in items}, set(explain.BANDS))
            self.assertEqual(len({rows[p['record_index']]['source_file'] for p in items}), 10)
            self.assertEqual(sum(available[category].values()), 12)

    def test_prefers_different_videos_and_spaced_frames_without_hiding_zeros(self):
        rows, predictions = examples()
        # All same-band candidate priorities prefer an unused video to another
        # adjacent high-confidence frame of an already selected video.
        for i in range(0, 12, 3):rows[i]['source_file'] = 'dev/repeated.mp4'
        selected, _ = explain.select_examples(rows, predictions, per_category=6)
        tp = [p for p in selected if p['category'] == 'tp']
        self.assertEqual(len(tp), 6)
        self.assertEqual([p['confidence_band'] for p in tp], list(explain.BANDS)*2)
        self.assertEqual(sum(rows[p['record_index']]['source_file'] == 'dev/repeated.mp4' for p in tp), 2)
        small, _ = explain.select_examples(*examples(1), per_category=10)
        self.assertEqual(len(small), 4)  # No duplicates to fill requested quotas.

    def test_invalid_predictions_seed_limits_and_test_records_fail(self):
        rows, predictions = examples(1)
        for number in [0, 11, True, 2.5]:
            with self.assertRaises(ValueError):explain.select_examples(rows, predictions, number)
        with self.assertRaises(ValueError):explain.select_examples(rows, predictions, seed=42)
        with self.assertRaises(ValueError):explain.select_examples(rows, predictions[:-1])
        with self.assertRaises(ValueError):explain.select_examples(rows, predictions[:-1]+[predictions[0]])
        wrong = [dict(p) for p in predictions];wrong[0]['target'] = 0
        with self.assertRaises(ValueError):explain.select_examples(rows, wrong)
        wrong = [dict(p) for p in predictions];wrong[0]['probability'] = float('nan')
        with self.assertRaises(ValueError):explain.select_examples(rows, wrong)
        with self.assertRaises(PermissionError):explain.select_examples([dict(r, split='test') for r in rows], predictions)


class NumericalAttribution(unittest.TestCase):
    def test_signed_formula_and_zero_relu_have_distinct_truthful_diagnostics(self):
        a = torch.ones(1, 5, 2);g = -torch.ones_like(a)
        m = torch.tensor([[[0., 0.], [2., 1.], [-2., -1.], [1., -1.], [0., 0.]]])
        result = explain.patch_attributions(a, g, m);d = result['diagnostics'][0]
        self.assertTrue(d['baseline_zero']);self.assertFalse(d['signed_zero'])
        self.assertEqual(d['baseline_zero_reason'], 'nonpositive_scores_removed_by_relu')
        np.testing.assert_array_equal(result['baseline'], 0)
        np.testing.assert_array_equal(result['signed_raw_patches'], [[[3., -3.], [0., 0.]]])
        self.assertGreater(result['signed'].max(), 0);self.assertLess(result['signed'].min(), 0)
        zero = explain.patch_attributions(a, torch.zeros_like(a), torch.zeros_like(a))
        self.assertEqual(zero['diagnostics'][0]['baseline_zero_reason'], 'zero_patch_gradients')
        np.testing.assert_array_equal(zero['signed'], 0)
        # Spatial cancellation is retained, not repaired with absolute gradients.
        g[:, 1:] = torch.tensor([[[1., 1.], [-1., -1.], [1., 1.], [-1., -1.]]])
        cancelled = explain.patch_attributions(a, g, g)
        self.assertEqual(cancelled['diagnostics'][0]['baseline_zero_reason'], 'weighted_sum_exactly_zero')
        self.assertEqual(cancelled['diagnostics'][0]['mean_gradient_cancellation_ratio'], 0.)
        self.assertTrue(cancelled['diagnostics'][0]['baseline_zero'])
        self.assertFalse(cancelled['diagnostics'][0]['signed_zero'])

    def test_original_engine_equivalence_hooks_gradients_modes_and_finite_difference(self):
        torch.manual_seed(4266);model = tiny_model().eval();pixels = torch.randn(1, 3, 224, 224)
        # Preserve preexisting parameter gradients too; no zero_grad/backward calls.
        parameter = next(model.parameters());parameter.grad = torch.ones_like(parameter)
        before = explain._weight_hash(model);grad = parameter.grad.clone()
        original, original_p = CLIPViTGradCAM(model).generate(pixels)
        engine = explain.CLIPPatchAttribution(model);hooks = len(engine.target_layer._forward_hooks)
        result = engine.generate(pixels)
        np.testing.assert_array_equal(original, result['baseline'])
        np.testing.assert_array_equal(original_p, result['probabilities'])
        self.assertEqual(len(engine.target_layer._forward_hooks), hooks)

        self.assertEqual(explain._weight_hash(model), before)
        torch.testing.assert_close(parameter.grad, grad, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in list(model.parameters())[1:]))
        logits = torch.from_numpy(result['logits'])
        torch.testing.assert_close(torch.sigmoid(logits[:, 0]-logits[:, 1]), torch.from_numpy(result['probabilities']))
        # Direct finite-difference multiplicative perturbation of ONE hidden patch
        # checks sum_channels(activation * dmargin/dactivation) independently.
        expected = float(result['signed_raw_patches'][0, 0, 0]);values = []
        for change in [-.01, .01]:
            def alter(module, inputs, output):
                modified = output.clone();modified[:, 1, :] *= 1+change;return modified
            handle = engine.target_layer.register_forward_hook(alter)
            try:
                with torch.no_grad():output = model(pixels);values.append(float(output[0, 0]-output[0, 1]))
            finally:handle.remove()
        self.assertAlmostEqual((values[1]-values[0])/.02, expected, delta=2e-5)
        model.train();model.encoder.eval();states = [m.training for m in model.modules()]
        engine.generate(pixels)
        self.assertEqual(states, [m.training for m in model.modules()])
        with patch.object(model, 'forward', side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):engine.generate(pixels)
        self.assertEqual(len(engine.target_layer._forward_hooks), hooks)
        with self.assertRaises(FloatingPointError):engine.generate(pixels*float('nan'))
        self.assertEqual(len(engine.target_layer._forward_hooks), hooks)

    def test_actual_zero_fake_head_is_not_replaced_by_positive_attribution(self):
        torch.manual_seed(4266);model = tiny_model().eval()
        with torch.no_grad():model.classifier.weight[0].zero_()
        engine = explain.CLIPPatchAttribution(model);pixels = torch.randn(1, 3, 224, 224)
        result = engine.generate(pixels)
        np.testing.assert_array_equal(result['baseline'], 0)
        self.assertEqual(result['diagnostics'][0]['baseline_zero_reason'], 'zero_patch_gradients')
        self.assertGreater(result['diagnostics'][0]['signed_absmax'], 0)
        original, _ = CLIPViTGradCAM(model).generate(pixels)
        np.testing.assert_array_equal(result['baseline'], original)


class ExportAndSafety(unittest.TestCase):
    def test_last_checkpoint_or_wrong_selection_is_rejected_before_dev_reads(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t);run = root/'run';run.mkdir();(run/'training_summary.json').write_text('{"best_epoch":2}')
            for path in [run/'best.pt', root/'index', root/'pool']:path.write_bytes(b'fixture')
            inputs = Inputs(root, root/'index', root/'pool', root/'weights')
            states = [
                {'epoch': 2, 'best_epoch': 2, 'config': {'threshold': .5, 'checkpoint_selection': 'frame_roc_auc'}},
                {'epoch': 1, 'best_epoch': 2, 'config': {'threshold': .6, 'checkpoint_selection': 'frame_roc_auc'}},
                {'epoch': 1, 'best_epoch': 2, 'config': {'threshold': .5, 'checkpoint_selection': 'frame_pr_auc'}}]
            for state in states:
                with patch.object(explain, 'validate_clip_checkpoint', return_value=state),patch.object(explain, 'DevSchedule') as index:
                    with self.assertRaises(ValueError):explain.explain_clip_dev_gallery(run, inputs, root)
                    index.assert_not_called()

    def test_dev_gallery_png_csv_arrays_provenance_and_missing_category(self):
        rows, predictions = examples(3)
        predictions = [p for p in predictions if p['record_index'] < 9];rows = rows[:9]  # No FN.
        model = tiny_model().eval();before = explain._weight_hash(model)
        with tempfile.TemporaryDirectory() as t, redirect_stdout(io.StringIO()):
            root = Path(t);run = root/'run';run.mkdir();(run/'training_summary.json').write_text('{"best_epoch":2}')
            for p in [run/'best.pt', root/'index', root/'pool']:p.write_bytes(b'synthetic fixture')
            inputs = Inputs(root, root/'index', root/'pool', root/'weights')
            state = {'epoch': 1, 'best_epoch': 2, 'config': {'threshold': .5, 'checkpoint_selection': 'frame_roc_auc'},
                'dev_predictions': predictions, 'experiment_identity': {'run_id': 'fixture'}}
            index = SimpleNamespace(records=lambda *args: rows, close=lambda: None)
            reader = SimpleNamespace(read=lambda records: [[np.full((224, 224, 3), r['id'], np.uint8)] for r in records])
            processor = lambda images, return_tensors: {'pixel_values': torch.full((1, 3, 224, 224), int(np.array(images[0])[0, 0, 0]), dtype=torch.float32)}
            def generate(pixels):
                i = int(pixels[0, 0, 0, 0]);a = torch.ones(1, 50, 2)
                output = explain.patch_attributions(a, -a, -a)
                output.update(probabilities=np.array([predictions[i]['probability']]), logits=np.array([[1., 0.]]))
                return output
            with patch.object(explain, 'validate_clip_checkpoint', return_value=state),patch.object(explain, 'DevSchedule', return_value=index),patch.object(explain, 'FrameReader', return_value=reader),patch.object(explain, 'checkpoint_components', return_value=(model, processor)),patch.object(explain.CLIPPatchAttribution, 'generate', side_effect=generate):
                out = explain.explain_clip_dev_gallery(run, inputs, root, per_category=2, device='cpu')
            with (out/'examples.csv').open(encoding='utf-8') as f:exported = list(csv.DictReader(f))
            self.assertEqual(len(exported), 12);self.assertEqual(len(list(out.glob('*.png'))), 6)
            self.assertEqual({r['method'] for r in exported}, {explain.BASELINE, explain.SIGNED})
            self.assertTrue(all(r['zero_map'] == 'True' for r in exported if r['method'] == explain.BASELINE))
            for row in exported:
                with np.load(out/row['arrays'], allow_pickle=False) as arrays:
                    np.testing.assert_array_equal(arrays['baseline'], 0)
                    self.assertTrue((arrays['signed'] < 0).all())
                self.assertEqual(row['checkpoint_sha256'], explain.digest(run/'best.pt'))
            provenance = json.loads((out/'provenance.json').read_text())
            self.assertEqual(provenance['status'], 'complete');self.assertTrue(provenance['weights_and_checkpoint_unchanged'])
            self.assertEqual(provenance['examples'], 6);self.assertEqual(len(provenance['selected']), 6)
            self.assertFalse(provenance['test_used']);self.assertEqual(explain._weight_hash(model), before)
            gallery = (out/'gallery.html').read_text(encoding='utf-8')
            self.assertIn('data:image/png;base64,', gallery);self.assertIn('FN (0)', gallery)

    def test_test_gate_precedes_every_read(self):
        with patch.object(explain, 'validate_clip_checkpoint') as checkpoint, patch.object(explain, 'DevSchedule') as index:
            with self.assertRaises(PermissionError):explain.explain_clip_dev_gallery('/absent', None, None, split='test')
            checkpoint.assert_not_called();index.assert_not_called()


if __name__ == '__main__':unittest.main()
