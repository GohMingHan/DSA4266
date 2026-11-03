"""Terminal entry point: dispatch/IO tests, never real model training."""
from contextlib import redirect_stdout, redirect_stderr, closing
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from src import cli
from src.config import configuration
from src.workflow import Inputs


def settings(root, models=None):
    args = cli.parser().parse_args(['--output', str(root/'output'), '--artifact-dir', str(root/'artifacts')])
    value = cli.load_settings(args)
    value['models'] = models or ['resnet50_baseline'];return value


def run(root, key, complete=True):
    path = root/'checkpoints'/key/'unique'/ 'run';path.mkdir(parents=True)
    identity = json.loads((cli.ROOT/'configs/project.json').read_text()) | {'model_id': key, 'run_id': 'unique'}
    (path/'experiment_identity.json').write_text(json.dumps(identity))
    if complete:
        (path/'training_summary.json').write_text('{"best_epoch":1}')
        (path/'best.pt').write_bytes(b'fixture, never deserialized')
    return path


class EntryPoint(unittest.TestCase):
    def test_saved_evaluation_never_implicitly_starts_full_dev_inference(self):
        with tempfile.TemporaryDirectory() as t, redirect_stdout(io.StringIO()):
            value = settings(Path(t));root = cli.workspace(value);path = run(root,'resnet50_baseline')
            inputs = Inputs(Path(t),Path(t)/'index',Path(t)/'pool',Path(t)/'weights')
            index = SimpleNamespace(records=lambda *args: ['fixed DEV records'],close=lambda:None)
            state = {'epoch':0,'best_epoch':1};report = {'unit':{},'video':{},'losses':{}}
            with patch.object(cli,'selected_runs',return_value={'resnet50_baseline':path}),\
                 patch('src.workflow.checkpoint_identity',return_value=({},state)),patch('src.data.DevSchedule',return_value=index),\
                 patch('src.training.checkpoint_dev_report',return_value=([],report)) as reuse,patch('src.reporting.evaluate_dev') as inference:
                cli.evaluate_models(value,root,inputs,fresh=False)
                reuse.assert_called_once();inference.assert_not_called()
                reuse.return_value=None
                with self.assertRaisesRegex(ValueError,'No automatic inference'):cli.evaluate_models(value,root,inputs,fresh=False)
                inference.assert_not_called()

    def test_small_learning_rates_remain_visible_without_changing_metric_values(self):
        self.assertEqual(cli.format_metric('Backbone LR',3e-6),'3.00e-06')
        self.assertEqual(cli.format_metric('Classifier/Head LR',1e-4),'1.00e-04')
        self.assertEqual(cli.format_metric('Video F2',.866412),'0.8664')
        self.assertEqual(cli.format_metric('Video F2',None),'N/A')

    def test_help_standalone_and_settings_validate_modes_paths_and_model_ids(self):
        process = subprocess.run([sys.executable, '-B', str(cli.ROOT/'run.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0);self.assertIn('--mode', process.stdout)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t);value = settings(root)
            self.assertEqual(value['mode'], 'results');self.assertEqual(value['models'], ['resnet50_baseline'])
            for argv in [['--models','unknown'],['--models','resnet50_baseline','resnet50_baseline'],
                         ['--mode','train','--build-metadata'],['--output',str(cli.ROOT/'results')],
                         ['--mode','results','--fresh-evaluation']]:
                with self.subTest(argv=argv), self.assertRaises(ValueError):cli.load_settings(cli.parser().parse_args(argv))

    def test_no_cuda_blocks_training_and_fresh_evaluation_before_data_or_outputs(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for argv in [['--mode','train'],['--mode','evaluate','--fresh-evaluation']]:
                with patch('torch.cuda.is_available',return_value=False),patch.object(cli,'prepare_metadata') as metadata,patch.object(cli,'train_models') as train,redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()) as error:
                    code = cli.main(argv+['--output',str(root/'out')])
                    self.assertEqual(code, 2);self.assertIn('CPU training/inference substitution is disabled', error.getvalue())
                    metadata.assert_not_called();train.assert_not_called();self.assertFalse((root/'out').exists())

    def test_dependency_order_adds_baseline_only_if_not_supplied_or_completed(self):
        with tempfile.TemporaryDirectory() as t:
            value = settings(Path(t), ['resnet50_lstm','efficientnet_b0_baseline']);root = cli.workspace(value)
            models, encoder = cli.ordered_models(value, root)
            self.assertEqual(models, ['resnet50_baseline','resnet50_lstm','efficientnet_b0_baseline']);self.assertIsNone(encoder)
            baseline = run(root, 'resnet50_baseline')
            models, encoder = cli.ordered_models(value, root)
            self.assertEqual(models, ['resnet50_lstm','efficientnet_b0_baseline']);self.assertEqual(encoder, baseline/'best.pt')
            value['resnet_checkpoint'] = Path(t)/'explicit/best.pt'
            self.assertEqual(cli.ordered_models(value, root)[1], value['resnet_checkpoint'])

    def test_all_six_dispatch_sequential_independent_and_lstm_receives_new_baseline(self):
        with tempfile.TemporaryDirectory() as t, redirect_stdout(io.StringIO()):
            value = settings(Path(t), list(cli.all_models()));root = cli.workspace(value)
            inputs = Inputs(Path(t),Path(t)/'index',Path(t)/'pool',Path(t)/'weights',Path(t)/'clips')
            called = [];paths = {};pipeline = object()
            def cnn(selected, inputs, project, **kwargs):
                key = selected[0];called.append(key);paths[key] = run(root,key)
                cfg = configuration(key,amp=value['amp'],workers=value['workers'])
                self.assertEqual((cfg.seed,cfg.batch_size,cfg.train_per_class),(4266,32,4000))
                return {key: paths[key]}
            def temporal(selected, inputs, project, **kwargs):
                key = selected[0];self.assertEqual(kwargs['cnn_best'],paths['resnet50_baseline']/'best.pt')
                self.assertIs(kwargs['clip_pipeline'],pipeline);called.append(key);paths[key] = run(root,key);return {key:paths[key]}
            def clip(inputs, project, weights, **kwargs):
                self.assertEqual(kwargs['config'].seed,4266);self.assertEqual(kwargs['config'].backbone_lr,3e-6)
                called.append(cli.CLIP_ID);paths[cli.CLIP_ID] = run(root,cli.CLIP_ID);return paths[cli.CLIP_ID]
            with patch.object(cli,'assets',return_value=inputs),patch.object(cli,'prepare_rgb',return_value=object()),\
                 patch('src.frame_cache.execute_cached_training',side_effect=cnn),patch('src.session_inputs.execute_session_lstm',side_effect=temporal),\
                 patch('src.session_inputs.prepare_clips',return_value=(inputs,pipeline,{})),patch('src.workflow.new_encoder',side_effect=lambda path,project: path),\
                 patch('src.clip_workflow.preflight_clip'),patch('src.clip_workflow.train_clip',side_effect=clip):
                result = cli.train_models(value, root, inputs, {}, SimpleNamespace(testsRun=1,wasSuccessful=lambda:True))
            self.assertEqual(called,list(cli.all_models()));self.assertEqual(len(set(result.values())),6)
            self.assertFalse((root/'training.lock').exists())

    def test_partial_and_completed_runs_never_silently_restart(self):
        with tempfile.TemporaryDirectory() as t, redirect_stdout(io.StringIO()):
            value = settings(Path(t));root = cli.workspace(value);path = run(root,'resnet50_baseline',complete=False)
            with patch.object(cli,'prepare_rgb') as pixels:
                with self.assertRaisesRegex(ValueError,'Partial'):cli.train_models(value,root,None,None,None)
                pixels.assert_not_called()
                (path/'training_summary.json').write_text('{"best_epoch":1}')
                self.assertEqual(cli.train_models(value,root,None,None,None),{'resnet50_baseline':path})
                pixels.assert_not_called()

    def test_metadata_reuse_is_read_only_and_build_requires_explicit_action(self):
        with tempfile.TemporaryDirectory() as t, redirect_stdout(io.StringIO()):
            value = settings(Path(t));index = value['index_path'];pool = value['train_pool']
            with patch('src.build_candidates.build_candidates') as ib,patch('src.data.build_pool') as pb:
                with self.assertRaises(FileNotFoundError):cli.prepare_metadata(value)
                ib.assert_not_called();pb.assert_not_called()
                index.parent.mkdir(parents=True);pool.parent.mkdir(parents=True)
                index.write_bytes(b'index fixture');pool.write_bytes(b'pool fixture')
                from src.config import POOL_SHA256
                report = {'index_sha256':'fixture','index_logical_sha256':'fixture','pool_sha256':POOL_SHA256,'dev_frames':41746,'dev_videos':500}
                before = [index.read_bytes(),pool.read_bytes()]
                with patch('src.build_candidates.validate_candidate_index'),patch('src.checkpoints.digest',return_value=POOL_SHA256),patch('src.workflow.validate_inputs',return_value=report):
                    cli.prepare_metadata(value)
                self.assertEqual(before,[index.read_bytes(),pool.read_bytes()]);ib.assert_not_called();pb.assert_not_called()

    def test_foreign_output_and_existing_training_lock_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            value = settings(Path(t));value['output_dir'].mkdir();(value['output_dir']/'precious.txt').write_text('preserve')
            with self.assertRaises(FileExistsError):cli.workspace(value)
            self.assertEqual((value['output_dir']/'precious.txt').read_text(),'preserve')
            value['output_dir'] = Path(t)/'new';root = cli.workspace(value)
            with cli.training_lease(root):
                with self.assertRaises(RuntimeError):
                    with cli.training_lease(root):pass
                self.assertTrue((root/'training.lock').exists())
            self.assertFalse((root/'training.lock').exists())

    def test_copied_experimental_sources_and_golden_fixture_are_unchanged(self):
        manifest = json.loads((cli.ROOT/'configs/source_manifest.json').read_text())
        for row in manifest['files']:
            actual = hashlib.sha256((cli.ROOT/row['path']).read_bytes()).hexdigest()
            self.assertEqual(actual,row['submission_sha256'],row['path'])
            if row['adaptation'] == 'Copied unchanged':self.assertEqual(actual,row['source_sha256'],row['path'])
        names = {Path(r['path']).name for r in manifest['files'] if r['adaptation'] == 'Copied unchanged'}
        self.assertTrue({'config.py','data.py','models.py','training.py','temporal.py','evaluation.py','checkpoints.py','face_mask.py','clip_model.py','clip_attribution.py'} <= names)
        self.assertFalse(list(cli.ROOT.glob('*.ipynb')));self.assertFalse((cli.ROOT/'metadata').exists())

    @unittest.skipUnless(os.environ.get('DSA4266_INDEX') and os.environ.get('DSA4266_POOL'),'External reference metadata not provided')
    def test_rebuilt_sqlite_layout_accepted_by_full_frame_scope_but_label_changes_rejected(self):
        from src.prepared_frames import _load_records
        from src.config import INDEX_SHA256
        from src.build_candidates import validate_candidate_index
        with tempfile.TemporaryDirectory() as t:
            index = Path(t)/'candidates.sqlite';shutil.copyfile(os.environ['DSA4266_INDEX'],index)
            with closing(sqlite3.connect(index)) as db:db.execute('PRAGMA user_version=19')
            self.assertNotEqual(hashlib.sha256(index.read_bytes()).hexdigest(),INDEX_SHA256)
            records = _load_records(SimpleNamespace(index=index,pool=Path(os.environ['DSA4266_POOL']),data=None))
            self.assertEqual(len(records),61746)
            with closing(sqlite3.connect(index)) as db:
                db.execute('UPDATE frames SET timestamp_s=timestamp_s+.01 WHERE id=1');db.commit()
            with self.assertRaises(ValueError):validate_candidate_index(index)


if __name__ == '__main__':unittest.main()
