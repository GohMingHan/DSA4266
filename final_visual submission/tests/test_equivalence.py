"""Before/after regression fixtures; no original package or dataset is needed."""
import ast
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from ast_fingerprints import definition_fingerprint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
torch.set_num_threads(2)
from src.config import Config, ORDER, configuration, declared_configurations
from src.data import FixedPoolSampler
from src.models import StagedCNN

GOLDEN = json.loads((ROOT/'tests/fixtures/equivalence.json').read_text(encoding='utf-8'))


class RefactorEquivalence(unittest.TestCase):
    def test_all_preserved_definitions(self):
        """Preserved bodies match; explicitly documented artifact adapters are separate."""
        trees = {}
        for row in GOLDEN['definitions']:
            with self.subTest(module=row['module'], definition=row['name']):
                tree = trees.setdefault(row['module'], ast.parse((ROOT/'src'/f"{row['module']}.py").read_text(encoding='utf-8')))
                node = next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==row['name'])
                digest = definition_fingerprint(node)
                expected=row['sha256']
                if 'artifact_adaptation' in row:
                    self.assertTrue(row['artifact_adaptation']['reason'])
                    expected=row['artifact_adaptation']['sha256']
                self.assertEqual(digest,expected,'Behavior differs from the recorded source')

    def test_effective_configurations_unchanged(self):
        self.assertEqual(declared_configurations(),GOLDEN['configurations'])
        self.assertEqual(json.loads((ROOT/'configs/models.json').read_text()),GOLDEN['configurations'])

    def test_seeded_cnn_names_outputs_and_checkpoint_loading(self):
        for key in ORDER:
            if key=='resnet50_lstm':continue
            with self.subTest(model=key):
                cfg=configuration(key,amp=False)
                torch.manual_seed(4266);model=StagedCNN(cfg,'.',pretrained=False).eval()
                expected=GOLDEN['models'][cfg.model]
                self.assertEqual([name for name,_ in model.named_parameters()],expected['parameter_names'])
                with torch.no_grad():output=model(torch.zeros(1,3,224,224))
                torch.testing.assert_close(output,torch.tensor(expected['output']),rtol=1e-6,atol=1e-7)
                other=StagedCNN(cfg,'.',pretrained=False).eval()
                other.load_state_dict(model.state_dict(),strict=True)
                with torch.no_grad():torch.testing.assert_close(other(torch.zeros(1,3,224,224)),output,rtol=0,atol=0)
                self.assertIsNot(next(model.parameters()),next(other.parameters()))

    def test_rotating_sampling_matches_pre_refactor(self):
        class Pool:
            sha256='synthetic'
            def __init__(self):
                self.frames={i:{'id':i,'label':i%2,'source_file':f'train/{i//2}','source_family':f'train/family{i//4}'} for i in range(24)}
            def records(self,ids):return [self.frames[i] for i in ids]
        sampler=FixedPoolSampler(Pool(),4,4266)
        self.assertEqual([sampler.select(e)[0] for e in range(5)],GOLDEN['sampling'])

    def test_real_resnet_encoder_checkpoint_is_loaded_and_frozen(self):
        from src.temporal import trained_resnet
        from src.workflow import new_encoder
        from src.data import PROTOCOL
        cfg=configuration('resnet50_baseline',amp=False)
        torch.manual_seed(4266);cnn=StagedCNN(cfg,'.',pretrained=False)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'configs').mkdir();(root/'configs/project.json').write_text('{"project_id":"fixture"}')
            state={'protocol':PROTOCOL,'config':cfg.as_dict(),'epoch':0,'best_epoch':1,'history':[{'epoch':1}],
                   'model':cnn.state_dict(),'experiment_identity':{'project_id':'fixture','model_id':'resnet50_baseline'},'input_pipeline':None}
            path=root/'best.pt';torch.save(state,path)
            self.assertEqual(new_encoder(path,root),path)
            temporal,identity=trained_resnet(configuration('resnet50_lstm',amp=False),path)
            torch.testing.assert_close(temporal.encoder.conv1.weight,cnn.network.conv1.weight,rtol=0,atol=0)
            temporal.train()
            self.assertFalse(temporal.encoder.training)
            self.assertTrue(all(not p.requires_grad for p in temporal.encoder.parameters()))
            self.assertEqual((temporal.lstm.input_size,temporal.lstm.hidden_size,temporal.lstm.num_layers),(2048,128,1))
            self.assertTrue(identity['frozen_encoder'])

    def test_flat_imports_and_notebook_execution_contract(self):
        for p in (ROOT/'src').glob('*.py'):
            importlib.import_module('src' if p.stem=='__init__' else 'src.'+p.stem)
            tree=ast.parse(p.read_text(encoding='utf-8'))
            for n in ast.walk(tree):
                if isinstance(n,ast.ImportFrom):
                    self.assertFalse((n.module or '').startswith(('cnn_experiment','cnn_face_experiment','visual_project')))
                elif isinstance(n,ast.Import):
                    self.assertFalse(any(a.name.startswith(('cnn_experiment','cnn_face_experiment','visual_project')) for a in n.names))
        self.assertFalse(any(p.is_dir() for p in (ROOT/'src').iterdir() if p.name!='__pycache__'))
        notebook=ROOT/'notebooks/final_visual.ipynb'
        if not notebook.exists():return  # Notebook is intentionally excluded from Kaggle source ZIP.
        nb=json.loads(notebook.read_text(encoding='utf-8'))
        cells={c['id']:c for c in nb['cells']}
        self.assertEqual(len(cells),len(nb['cells']))
        for cell in nb['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
        options=ast.parse(''.join(cells['fv_010']['source']))
        values={n.targets[0].id:ast.literal_eval(n.value) for n in options.body if isinstance(n,ast.Assign)}
        self.assertEqual((values['EXECUTION_MODE'],values['RESUME_FROM'],values['ARTIFACT_MODE']),('inspect',{},'REUSE'))
        self.assertNotIn('SELECTED_MODELS',values)
        self.assertNotIn('TRAIN_CONFIRMATION',values)
        ids=[c['id'] for c in nb['cells']]
        self.assertLess(ids.index('fv_012'),ids.index('fv_028'))
        self.assertLess(ids.index('fv_028'),ids.index('fv_train_resnet50_baseline'))
        self.assertLess(ids.index('fv_train_resnet50_baseline'),ids.index('fv_018'))
        training_ids=['fv_train_resnet50_baseline','fv_train_resnet50_face_mask','fv_train_resnet50_lstm',
                      'fv_train_efficientnet_b0_baseline','fv_train_efficientnet_b0_face_mask']
        self.assertEqual([identifier for identifier in ids if identifier in training_ids],training_ids)
        self.assertNotIn('execute_training',''.join(cells['fv_014']['source']))
        self.assertNotIn('run_model_cell',''.join(cells['fv_014']['source']))

    def test_cumulative_tables_plots_and_distinct_video_f2(self):
        from src.evaluation import binary_metrics,write_history_metrics,plot_history
        from src.reporting import epoch_table,export_epoch_report
        labels=[0,0,1,1]
        unit=binary_metrics(labels,[.1,.2,.8,.9],.5)
        video=binary_metrics(labels,[.7,.2,.8,.1],.5)
        self.assertNotEqual(unit['f2'],video['f2'])
        for kind in ['frame','clip']:
            with self.subTest(level=kind),tempfile.TemporaryDirectory() as t:
                root=Path(t);cfg=configuration('resnet50_lstm' if kind=='clip' else 'resnet50_baseline')
                history=[]
                for epoch in [1,2]:
                    dev={**unit,'roc_auc':.7 if epoch==1 else 1.,'loss':.4,'balanced_loss':.5,'real_class_loss':.3,'fake_class_loss':.7}
                    history.append({'epoch':epoch,'train':{**unit,'loss':.2},'dev':dev,'dev_video':video,
                                    'learning_rates':{'head':1e-4,'final_stage':1.5e-5},'is_best':True})
                (root/'config.json').write_text(json.dumps(cfg.as_dict()))
                (root/'training_summary.json').write_text('{"best_epoch":2}')
                write_history_metrics(root,history)
                rows,best,level=epoch_table(root,kind=kind)
                self.assertEqual((len(rows),len(rows[0]),best,level),(2,19,2,kind))
                self.assertEqual(rows[1]['Best Checkpoint'],'Selected best')
                self.assertEqual(rows[1]['Video F2'],video['f2'])
                self.assertEqual(rows[1][kind.title()+' F2'],unit['f2'])
                export_epoch_report(root,root,kind=kind,checkpoint_selection=cfg.checkpoint_selection,in_run=True)
                self.assertIn('#fff0b3',(root/'epoch_metrics.html').read_text())
                plot_history(root,history,unit_label=kind.title(),best_epoch=2,selection_metric=cfg.checkpoint_selection)
                self.assertGreater((root/'training_curves.png').stat().st_size,1000)

    def test_source_zip_is_flat_and_excludes_generated_material(self):
        from src.backup import build_source_archive
        import shutil,zipfile
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for name in ['src','tests','configs']:
                shutil.copytree(ROOT/name,root/name,ignore=shutil.ignore_patterns('__pycache__'))
            for name in ['README.md','requirements.txt','run.py','settings.json','.gitignore','.gitattributes','THIRD_PARTY_NOTICES.txt']:shutil.copy2(ROOT/name,root/name)
            (root/'src/not_for_upload.pt').write_bytes(b'dummy generated state')
            (root/'configs/candidates.sqlite').write_bytes(b'dummy database')
            (root/'configs/development_log.json').write_text('{}')
            archive=root/'source.zip';receipt=build_source_archive(root,archive)
            with zipfile.ZipFile(archive) as z:
                names=z.namelist()
                self.assertIn('src/workflow.py',names)
                self.assertIn('tests/fixtures/equivalence.json',names)
                self.assertIn('run.py',names)
                self.assertIn('settings.json',names)
                self.assertNotIn('src/not_for_upload.pt',names)
                self.assertNotIn('configs/candidates.sqlite',names)
                self.assertNotIn('configs/development_log.json',names)
                self.assertFalse(any(n.endswith('.ipynb') for n in names))
                self.assertFalse(any(n.startswith(('src/cnn_experiment/','src/cnn_face_experiment/','src/visual_project/')) for n in names))
                self.assertEqual(receipt['files'],len(names)-1)


if __name__=='__main__':unittest.main()
