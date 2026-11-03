"""CPU-only notebook orchestration tests. Training is always mocked."""
import ast
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from src import notebook as nb
from src.config import ORDER,RECIPES,configuration,POOL_SHA256,INDEX_LOGICAL_SHA256
from src.workflow import Inputs
from src.evaluation import binary_metrics


def test_result(ok=True):return SimpleNamespace(testsRun=58,wasSuccessful=lambda:ok)


def data_report():return {'pool_sha256':POOL_SHA256,'index_logical_sha256':INDEX_LOGICAL_SHA256,'dev_frames':41746,'dev_videos':500}


def project(root):
    (root/'configs').mkdir();(root/'configs/project.json').write_text('{"project_id":"fixture"}')
    return Inputs(root/'videos',root/'index',root/'pool',root/'imagenet',root/'clips',root/'yunet',root/'landmarker')


def run_fixture(root,key,*,name='one',complete=True,checkpoint=True,video_pr=.5,video_roc=.7):
    run=root/'checkpoints'/key/name/'run';run.mkdir(parents=True)
    identity={'project_id':'fixture','model_id':key,'run_id':name,'label':RECIPES[key].label}
    (run/'experiment_identity.json').write_text(json.dumps(identity))
    cfg=configuration(key,amp=False,workers=0)
    state={'config':cfg.as_dict(),'experiment_identity':identity,'model':{'fixture':torch.ones(1)},
           'epoch':0,'best_epoch':1,'input_pipeline':None}
    if checkpoint:torch.save(state,run/'best.pt')
    if complete:
        unit=binary_metrics([0,0,1,1],[.1,.2,.8,.9],.5)
        video={**binary_metrics([0,0,1,1],[.7,.2,.8,.1],.5),'pr_auc':video_pr,'roc_auc':video_roc}
        metrics={'unit':unit,'video':video,'losses':{'loss':.4,'balanced_loss':.5}}
        (run/'config.json').write_text(json.dumps(cfg.as_dict()))
        (run/'training_summary.json').write_text('{"best_epoch":1}')
        (run/'best_dev_metrics.json').write_text(json.dumps(metrics))
        (run/'history.jsonl').write_text(json.dumps({'epoch':1,'train':{'loss':.2},'dev':unit,'dev_video':video})+'\n')
    return run


class ModelCells(unittest.TestCase):
    def setUp(self):
        # These legacy tests isolate orchestration; real preparation gates have
        # dedicated tests in test_prepared_frames and never permit GPU training.
        gate=patch('src.prepared_frames.require_prepared_cache',return_value={})
        gate.start();self.addCleanup(gate.stop)

    def test_confirmation_tests_validation_and_mode_gates(self):
        with tempfile.TemporaryDirectory() as t,redirect_stdout(io.StringIO()):
            root=Path(t);inputs=project(root);key='resnet50_baseline'
            with patch.object(nb,'execute_training') as train:
                nb.run_model_cell(key,inputs,root,confirmation='TRAIN_'+key,execution_mode='inspect')
                with self.assertRaises(PermissionError):nb.run_model_cell(key,inputs,root,confirmation='TRAIN_resnet50_face_mask',execution_mode='train')
                for result,report in [(None,data_report()),(test_result(False),data_report()),(test_result(),None)]:
                    with self.assertRaises(RuntimeError):
                        nb.run_model_cell(key,inputs,root,confirmation='TRAIN_'+key,execution_mode='train',test_result=result,data_report=report)
                train.assert_not_called()

    def test_each_call_trains_one_model_and_lstm_uses_new_baseline(self):
        with tempfile.TemporaryDirectory() as t,redirect_stdout(io.StringIO()):
            root=Path(t);inputs=project(root);runs={};calls=[]
            def train(selected,supplied_inputs,supplied_project,**kwargs):
                key=selected[0];self.assertEqual(selected,[key]);self.assertEqual(supplied_inputs.weights,inputs.weights)
                self.assertEqual(kwargs['confirmation'],'TRAIN_SELECTED_MODELS')
                if key=='resnet50_lstm':self.assertEqual(kwargs['cnn_best'],runs['resnet50_baseline']/'best.pt')
                else:self.assertIsNone(kwargs['cnn_best'])
                calls.append(key);return {key:run_fixture(root,key)}
            with patch.object(nb,'execute_training',side_effect=train),patch('src.session_inputs.execute_session_lstm',side_effect=train),patch.object(nb,'_show_model',side_effect=lambda k,r,p:{k:r}):
                for key in ORDER:
                    runs.update(nb.run_model_cell(key,inputs,root,confirmation='TRAIN_'+key,execution_mode='train',
                        test_result=test_result(),data_report=data_report(),runs=runs,amp=False,workers=0,
                        clip_pipeline=object() if key=='resnet50_lstm' else None))
            self.assertEqual(calls,list(ORDER));self.assertEqual(len({str(p) for p in runs.values()}),5)

    def test_completed_run_is_discovered_and_replayed_after_restart(self):
        with tempfile.TemporaryDirectory() as t,redirect_stdout(io.StringIO()):
            root=Path(t);inputs=project(root);key='resnet50_baseline';run=run_fixture(root,key)
            with patch.object(nb,'execute_training') as train,patch.object(nb,'show_run') as show:
                result=nb.run_model_cell(key,inputs,root,confirmation='TRAIN_'+key,execution_mode='train',
                                         test_result=test_result(),data_report=data_report())
                self.assertEqual(result,{key:run});show.assert_called_once_with(run,root);train.assert_not_called()
            self.assertEqual(nb.collect_runs(root),{key:run})

    def test_partial_run_requires_resume_and_forwards_only_this_model(self):
        with tempfile.TemporaryDirectory() as t,redirect_stdout(io.StringIO()):
            root=Path(t);inputs=project(root);key='resnet50_baseline';run_fixture(root,key,complete=False)
            # Two old boundaries must not prevent an explicitly selected resume.
            run_fixture(root,key,name='two',complete=False)
            kwargs=dict(confirmation='TRAIN_'+key,execution_mode='train',test_result=test_result(),data_report=data_report())
            with patch.object(nb,'execute_training') as train:
                with self.assertRaises(ValueError):nb.run_model_cell(key,inputs,root,**kwargs)
                train.assert_not_called()
            with patch.object(nb,'execute_training',return_value={key:root/'new/run'}) as train,patch.object(nb,'_show_model',return_value={}):
                nb.run_model_cell(key,inputs,root,resume_from={key:'latest','efficientnet_b0_baseline':'other'},**kwargs)
                self.assertEqual(train.call_args.kwargs['resume_from'],{key:'latest'})

    def test_parallel_model_start_is_blocked_and_error_releases_lock(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);inputs=project(root);key='resnet50_baseline'
            kwargs=dict(confirmation='TRAIN_'+key,execution_mode='train',test_result=test_result(),data_report=data_report())
            nb._TRAINING_LOCK.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError,'Another notebook model'):nb.run_model_cell(key,inputs,root,**kwargs)
            finally:nb._TRAINING_LOCK.release()
            with patch.object(nb,'execute_training',side_effect=RuntimeError('synthetic failure')):
                with self.assertRaisesRegex(RuntimeError,'synthetic failure'):nb.run_model_cell(key,inputs,root,**kwargs)
            self.assertTrue(nb._TRAINING_LOCK.acquire(blocking=False));nb._TRAINING_LOCK.release()

    def test_lstm_rejects_missing_manifest_partial_or_historical_baseline(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);inputs=project(root);key='resnet50_lstm'
            kwargs=dict(confirmation='TRAIN_'+key,execution_mode='train',test_result=test_result(),data_report=data_report(),clip_pipeline=object())
            with patch.object(nb,'execute_training') as train:
                with self.assertRaises(FileNotFoundError):nb.run_model_cell(key,replace(inputs,clips=None),root,**kwargs)
                with self.assertRaises(FileNotFoundError):nb.run_model_cell(key,inputs,root,**kwargs)
                baseline=run_fixture(root,'resnet50_baseline',complete=False)
                with self.assertRaises(ValueError):nb.run_model_cell(key,inputs,root,**kwargs)
                (baseline/'training_summary.json').write_text('{"best_epoch":1}')
                state=torch.load(baseline/'best.pt',weights_only=True);state['experiment_identity']['project_id']='historical';torch.save(state,baseline/'best.pt')
                with self.assertRaises(ValueError):nb.run_model_cell(key,inputs,root,**kwargs)
                train.assert_not_called()

    def test_other_model_ambiguity_does_not_block_independent_cell(self):
        with tempfile.TemporaryDirectory() as t,redirect_stdout(io.StringIO()):
            root=Path(t);inputs=project(root)
            run_fixture(root,'efficientnet_b0_baseline',name='one');run_fixture(root,'efficientnet_b0_baseline',name='two')
            self.assertEqual(nb.run_model_cell('resnet50_baseline',inputs,root),{})
            with self.assertRaises(ValueError):nb.collect_runs(root)


class SharedSetupAndComparison(unittest.TestCase):
    def test_shared_reuse_keeps_optional_resources_explicit(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);inputs=replace(project(root),clips=None,detector=None,landmarker=None)
            with patch.object(nb,'prepare_manifests',return_value=inputs) as prepare,patch.object(nb,'validate_inputs',return_value=data_report()),patch.object(nb,'setup_assets') as assets,patch.object(nb,'execute_training') as train:
                result,report=nb.prepare_notebook(inputs,root,test_result=test_result())
                self.assertEqual(result,inputs);self.assertEqual(prepare.call_args.kwargs,{'artifact_mode':'REUSE','need_clips':False})
                self.assertEqual(report['notebook_ready'],{'face_assets':False,'lstm_manifest':False})
                assets.assert_not_called();train.assert_not_called()
                nb.prepare_notebook(inputs,root,test_result=test_result(),artifact_mode='BUILD')
                self.assertTrue(prepare.call_args.kwargs['need_clips'])
                with self.assertRaises(RuntimeError):nb.prepare_notebook(inputs,root,test_result=test_result(False))

    def test_completed_pending_partial_missing_and_tie_break_keep_actual_metrics(self):
        with tempfile.TemporaryDirectory() as t,redirect_stdout(io.StringIO()):
            root=Path(t);project(root)
            run_fixture(root,'resnet50_baseline',video_pr=.5,video_roc=.7)
            run_fixture(root,'resnet50_lstm',video_pr=.5,video_roc=.8)
            run_fixture(root,'resnet50_face_mask',complete=False)
            run_fixture(root,'efficientnet_b0_baseline',checkpoint=False)
            with patch('IPython.display.display') as display:
                rows,ranking,out=nb.show_comparison(root,nb.collect_runs(root))
            self.assertEqual(len(rows),5);self.assertEqual(ranking[0]['model'],'ResNet50 LSTM')
            statuses={row['Model']:row['Status'] for row in rows}
            self.assertEqual(statuses['ResNet50 Face mask'],'Partial')
            self.assertEqual(statuses['EfficientNet-B0 baseline'],'Missing checkpoint')
            self.assertEqual(statuses['EfficientNet-B0 Face mask'],'Pending')
            pending=next(row for row in rows if row['Status']=='Pending');self.assertIsNone(pending['Video PR-AUC'])
            temporal=next(row for row in rows if row['Model']=='ResNet50 LSTM');self.assertEqual(temporal['Metric level'],'Clip')
            self.assertNotEqual(temporal['Frame/Clip F2'],temporal['Video F2'])
            self.assertGreater((out/'metrics_comparison.png').stat().st_size,1000)
            self.assertGreater((out/'video_comparison.png').stat().st_size,1000)
            self.assertGreaterEqual(display.call_count,3)


if __name__=='__main__':unittest.main()
