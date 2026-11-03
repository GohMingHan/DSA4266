"""Small offline CLIP and lossless-cache checks; no pretrained download/training."""
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.clip_model import CLIPConfig,CLIPBinaryClassifier,CLIPDataset,CLIPViTGradCAM,classification_loss,fake_probability
from src.frame_cache import RGBCache,CachedReader,CachedPipeline


def tiny_model():
    from transformers import CLIPVisionConfig,CLIPVisionModelWithProjection
    config=CLIPVisionConfig(hidden_size=32,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,
                            image_size=224,patch_size=32,projection_dim=512)
    encoder=CLIPVisionModelWithProjection(config)
    encoder.set_attn_implementation('eager')
    return CLIPBinaryClassifier(encoder=encoder)


def record(label=1):
    return {'id':7,'source_file':'train/source.mp4','source_family':'train/family','split':'train','kind':'frame',
            'indices':[2],'timestamps_exact':['1/5'],'label':label,'video_label':1,'metric_eligible':True}


class CLIPIntegration(unittest.TestCase):
    def test_seed4266_repeats_rngs_head_and_preserves_encoder(self):
        import copy,random
        from src.clip_workflow import clip_seed,clip_seed_policy
        encoder=tiny_model().encoder
        before={name:value.clone() for name,value in encoder.state_dict().items()}
        samples=[];heads=[]
        for _ in range(2):
            clip_seed()
            samples.append((random.random(),np.random.random(),torch.rand(4)))
            heads.append(CLIPBinaryClassifier(encoder=copy.deepcopy(encoder)))
        self.assertEqual(samples[0][:2],samples[1][:2]);torch.testing.assert_close(samples[0][2],samples[1][2],rtol=0,atol=0)
        for name,value in heads[0].classifier.state_dict().items():torch.testing.assert_close(value,heads[1].classifier.state_dict()[name],rtol=0,atol=0)
        for model in heads:
            for name,value in model.encoder.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.manual_seed_all') as cuda_seed:
            clip_seed()
            self.assertTrue(cuda_seed.called);self.assertTrue(all(call.args==(4266,) for call in cuda_seed.call_args_list))
        policy=clip_seed_policy(CLIPConfig())
        self.assertTrue(all(value==4266 for key,value in policy.items() if not key.endswith('_derivation')))

    def test_dataloader_generator_and_batches_repeat_from_seed4266(self):
        from transformers import CLIPImageProcessor
        from src.clip_model import make_clip_loader
        from src.data import stable_seed
        rows=[record(0),record(1)|{'id':8}];processor=CLIPImageProcessor();cfg=CLIPConfig()
        loaders=[make_clip_loader(rows,'.',processor,cfg,epoch=2,training=True) for _ in range(2)]
        expected=stable_seed(4266,2,True)%2**63
        self.assertEqual([loader.generator.initial_seed() for loader in loaders],[expected,expected])
        pixels=np.arange(224*224*3,dtype=np.uint8).reshape(224,224,3)
        with patch('src.data.FrameReader.read',return_value=[[pixels],[pixels]]):
            first=next(iter(loaders[0]));second=next(iter(loaders[1]))
        for a,b in zip(first,second):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertIsInstance(loaders[0].sampler,torch.utils.data.SequentialSampler)

    def test_exact_recipe_and_class_mapping(self):
        cfg=CLIPConfig().validate()
        self.assertEqual((cfg.seed,cfg.sampling_seed,cfg.epochs,cfg.patience,cfg.batch_size,cfg.workers),(4266,4266,15,5,32,0))
        self.assertEqual((cfg.backbone_lr,cfg.head_lr,cfg.weight_decay,cfg.gradient_clip),(3e-6,1e-4,1e-4,1.))
        logits=torch.tensor([[3.,-2.],[-2.,3.]],requires_grad=True);targets=torch.tensor([1.,0.])
        loss=classification_loss(logits,targets)
        torch.testing.assert_close(loss,torch.nn.functional.binary_cross_entropy_with_logits(logits[:,0]-logits[:,1],targets,reduction='none'))
        self.assertGreater(fake_probability(logits)[0],.99);self.assertLess(fake_probability(logits)[1],.01)
        with self.assertRaises(ValueError):replace(cfg,backbone_lr=1e-4).validate()

    def test_architecture_normalized_embedding_and_patch_gradcam(self):
        model=tiny_model().eval();pixels=torch.randn(1,3,224,224)
        original={k:v.clone() for k,v in model.state_dict().items()};seen=[]
        handle=model.classifier.register_forward_pre_hook(lambda m,args:seen.append(args[0].detach()))
        with torch.no_grad():logits=model(pixels)
        handle.remove();self.assertEqual(tuple(logits.shape),(1,2));torch.testing.assert_close(seen[0].norm(dim=1),torch.ones(1))
        cam=CLIPViTGradCAM(model);hooks=len(cam.target_layer._forward_hooks)
        maps,prob=cam.generate(pixels)
        self.assertEqual(maps.shape,(1,224,224));self.assertTrue(np.isfinite(maps).all());self.assertEqual(len(cam.target_layer._forward_hooks),hooks)
        self.assertFalse(model.training);self.assertTrue(all(p.grad is None for p in model.parameters()))
        for key,value in model.state_dict().items():torch.testing.assert_close(value,original[key],rtol=0,atol=0)
        with patch.object(model,'forward',side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):cam.generate(pixels)
        self.assertEqual(len(cam.target_layer._forward_hooks),hooks)

    def test_original_processor_and_shared_flip(self):
        from transformers import CLIPImageProcessor
        from src.data import stable_seed
        processor=CLIPImageProcessor();cfg=CLIPConfig();row=record()
        pixels=np.zeros((224,224,3),dtype=np.uint8);pixels[:,:112]=255
        dataset=CLIPDataset([row],'.',processor,cfg,epoch=1,training=True)
        with patch.object(dataset.reader,'read',return_value=[[pixels]]):value,target,index=dataset[0]
        flip=stable_seed(4266,1,'frame',7,'flip')/2**64<.5
        expected=processor(images=pixels[:,::-1,:].copy() if flip else pixels,return_tensors='pt')['pixel_values'][0]
        torch.testing.assert_close(value,expected,rtol=0,atol=0);self.assertEqual((target.item(),index),(1.,0))
        dev=row|{'split':'dev','source_file':'dev/source.mp4'}
        self.assertEqual(CLIPDataset([dev],'.',processor,cfg,training=False).flip_probability,0)
        self.assertEqual(tuple(value.shape),(3,224,224))

    def test_optimizer_no_freezing_and_checkpoint_strict_load(self):
        from src.clip_workflow import optimizer_for
        model=tiny_model();optimizer=optimizer_for(model,CLIPConfig())
        self.assertEqual([g['lr'] for g in optimizer.param_groups],[3e-6,1e-4])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        other=tiny_model();other.load_state_dict(model.state_dict(),strict=True)
        self.assertEqual(list(model.state_dict()),list(other.state_dict()))

    def test_full_finetuned_components_restore_and_gradcam_without_pretrained_assets(self):
        from transformers import CLIPImageProcessor
        from src.clip_model import checkpoint_components
        model=tiny_model().eval();pristine=model.encoder.visual_projection.weight.detach().clone()
        with torch.no_grad():
            model.encoder.visual_projection.weight.add_(.01)
            model.classifier.bias.add_(.2)
        processor=CLIPImageProcessor()
        state={'model':model.state_dict(),'encoder_config':json.loads(json.dumps(model.encoder.config.to_dict())),
               'processor_config':json.loads(json.dumps(processor.to_dict()))}
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'best.pt';torch.save(state,path)
            saved=torch.load(path,map_location='cpu',weights_only=True)
        with patch('transformers.CLIPVisionModelWithProjection.from_pretrained',side_effect=AssertionError('must not fetch pretrained encoder')):
            restored,restored_processor=checkpoint_components(saved)
        self.assertFalse(torch.equal(restored.encoder.visual_projection.weight,pristine))
        for name,value in model.state_dict().items():torch.testing.assert_close(restored.state_dict()[name],value,rtol=0,atol=0)
        pixels=torch.zeros(1,3,224,224)
        original_maps,original_scores=CLIPViTGradCAM(model).generate(pixels)
        maps,scores=CLIPViTGradCAM(restored.eval()).generate(pixels)
        np.testing.assert_array_equal(scores,original_scores);np.testing.assert_array_equal(maps,original_maps)
        image=np.full((224,224,3),128,dtype=np.uint8)
        torch.testing.assert_close(processor(images=image,return_tensors='pt')['pixel_values'],restored_processor(images=image,return_tensors='pt')['pixel_values'],rtol=0,atol=0)

    def test_pinned_public_asset_setup_is_explicit_and_reuses_verified_snapshot(self):
        from src.clip_model import prepare_clip_assets,MODEL_NAME,REVISION
        from transformers import CLIPImageProcessor
        encoder=tiny_model().encoder
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'assets'
            with self.assertRaises(FileNotFoundError):prepare_clip_assets(path)
            with patch('transformers.CLIPVisionModelWithProjection.from_pretrained',return_value=encoder) as load,patch('transformers.CLIPImageProcessor.from_pretrained',return_value=CLIPImageProcessor()):
                self.assertEqual(prepare_clip_assets(path,download=True),path)
                self.assertEqual(load.call_args.args,(MODEL_NAME,));self.assertEqual(load.call_args.kwargs['revision'],REVISION)
                self.assertIs(load.call_args.kwargs['weights_only'],True);self.assertIs(load.call_args.kwargs['use_safetensors'],False)
            with patch('transformers.CLIPVisionModelWithProjection.from_pretrained',side_effect=AssertionError('must reuse')):prepare_clip_assets(path)
            (path/'config.json').write_text('changed')
            with self.assertRaises(ValueError):prepare_clip_assets(path)


    def test_fp32_nonfinite_never_updates_and_clip_norm_one(self):
        from src.clip_workflow import CLIPGradientGuard
        p=torch.nn.Parameter(torch.tensor([2.]));optimizer=torch.optim.AdamW([p],lr=.1);scaler=torch.amp.GradScaler('cpu',enabled=False)
        guard=CLIPGradientGuard(CLIPConfig(amp=False));events=[]
        p.grad=torch.tensor([100.]);guard.step(optimizer,scaler,emit=events.append,epoch=0,batch=1)
        self.assertLessEqual(float(p.grad.norm()),1.);before=p.detach().clone();state=pickle.dumps(optimizer.state_dict())
        p.grad=torch.tensor([float('inf')])
        with self.assertRaises(FloatingPointError):guard.step(optimizer,scaler,emit=events.append,epoch=0,batch=2)
        torch.testing.assert_close(p,before);self.assertEqual(pickle.dumps(optimizer.state_dict()),state);self.assertTrue(events[-1]['skipped_step'])

    def test_cpu_amp_overflow_backoff_and_recovery(self):
        from src.clip_workflow import CLIPGradientGuard
        p=torch.nn.Parameter(torch.tensor([2.]));optimizer=torch.optim.AdamW([p],lr=.1);scaler=torch.amp.GradScaler('cpu',init_scale=8.)
        guard=CLIPGradientGuard(CLIPConfig());events=[];before=p.detach().clone()
        scaler.scale(p.sum()).backward();p.grad.fill_(float('inf'))
        guard.step(optimizer,scaler,emit=events.append,epoch=0,batch=1)
        torch.testing.assert_close(p,before);self.assertFalse(optimizer.state);self.assertEqual(scaler.get_scale(),4.)
        scaler.scale(p.sum()).backward();guard.step(optimizer,scaler,emit=events.append,epoch=0,batch=2)
        self.assertFalse(torch.equal(p,before));self.assertEqual(guard.state['optimizer_steps'],1)

    def test_fake_positive_epoch_metrics_without_updates(self):
        from src.clip_workflow import clip_epoch
        class Model(torch.nn.Module):
            def forward(self,x):return torch.stack((x[:,0,0,0],-x[:,0,0,0]),dim=1)
        dataset=torch.utils.data.TensorDataset(torch.tensor([1.,-1.]).view(2,1,1,1).expand(2,3,224,224),torch.tensor([1.,0.]),torch.tensor([0,1]))
        loader=torch.utils.data.DataLoader(dataset,batch_size=2)
        metrics,predictions,_=clip_epoch(Model(),loader,CLIPConfig(amp=False),device='cpu')
        self.assertEqual((metrics['roc_auc'],metrics['f2']),(1.,1.));self.assertEqual([r['target'] for r in predictions],[1,0])
        self.assertAlmostEqual(metrics['balanced_loss'],metrics['loss'])

    def test_test_gate_and_blank_confirmation_no_training(self):
        from src.clip_reporting import explain_clip_dev,run_clip_cell
        with patch('src.clip_reporting.validate_clip_checkpoint',side_effect=AssertionError('must not load')):
            with self.assertRaises(PermissionError):explain_clip_dev(None,None,None,None,split='test')
        with tempfile.TemporaryDirectory() as t,patch('src.clip_reporting.train_clip',side_effect=AssertionError('must not train')),redirect_stdout(io.StringIO()):
            self.assertIsNone(run_clip_cell(None,Path(t),None))
        with tempfile.TemporaryDirectory() as t,self.assertRaises(PermissionError):run_clip_cell(None,Path(t),None,enabled=True,execution_mode='train')

    def test_completed_cell_replays_without_assets_or_preflight_and_new_training_checks_preflight(self):
        from src import clip_reporting as reporting
        from src.clip_model import MODEL_ID
        from src.checkpoints import digest
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run=root/'run';run.mkdir();(run/'training_summary.json').write_text('{}')
            with patch.object(reporting,'find_clip_run',return_value=run),patch.object(reporting,'show_clip') as show,patch.object(reporting,'train_clip') as train:
                self.assertEqual(reporting.run_clip_cell(None,root,None,enabled=True,confirmation='TRAIN_'+MODEL_ID,execution_mode='train'),run)
                train.assert_not_called();show.assert_called_once_with(run)
            for name in ['index','pool']:(root/name).write_bytes(name.encode())
            inputs=SimpleNamespace(index=root/'index',pool=root/'pool')
            with patch.object(reporting,'find_clip_run',return_value=None),patch.object(reporting,'show_clip'),patch.object(reporting,'train_clip',return_value=run) as train:
                kwargs={'enabled':True,'confirmation':'TRAIN_'+MODEL_ID,'execution_mode':'train'}
                with self.assertRaisesRegex(RuntimeError,'GPU preflight'):reporting.run_clip_cell(inputs,root,None,**kwargs)
                train.assert_not_called()
                smoke={'optimizer_steps':0,'model_seed':4266,'amp':True,'index_sha256':digest(inputs.index),'pool_sha256':digest(inputs.pool),
                       'splits':{'train':{'finite':True},'dev':{'finite':True}}}
                reporting.run_clip_cell(inputs,root,None,preflight_report=smoke,**kwargs)
                self.assertEqual(train.call_args.kwargs['config'].seed,4266)

    def test_dev_png_csv_export_all_categories_without_trained_weights(self):
        """Synthetic IO test; the actual token-gradient engine is tested separately."""
        from src import clip_reporting as reporting
        from src.workflow import Inputs
        labels=[1,0,0,1];probabilities=[.9,.1,.8,.2]
        rows=[record(y)|{'id':i,'split':'dev','source_file':f'dev/v{i}.mp4'} for i,y in enumerate(labels)]
        predictions=[{'record_index':i,'target':y,'probability':p} for i,(y,p) in enumerate(zip(labels,probabilities))]
        model=tiny_model().eval();original={k:v.clone() for k,v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run=root/'run';run.mkdir();(run/'training_summary.json').write_text('{}');(run/'best.pt').write_bytes(b'synthetic checkpoint fixture')
            for name in ['index','pool']:(root/name).write_bytes(b'fixture')
            weights=root/'weights';weights.mkdir();(weights/'clip_pretrained_manifest.json').write_text('{"fixture":true}')
            inputs=Inputs(root,root/'index',root/'pool',weights)
            state={'model':model.state_dict(),'dev_predictions':predictions,'pretrained_manifest':{'fixture':True}}
            index=SimpleNamespace(records=lambda *args:rows,close=lambda:None)
            reader=SimpleNamespace(read=lambda records:[[np.full((224,224,3),r['id']*40,dtype=np.uint8)] for r in records])
            cam=SimpleNamespace(generate=lambda pixels:(np.ones((1,224,224),dtype=np.float32),np.array([probabilities[int(pixels[0,0,0,0])]])))
            processor=lambda images,return_tensors:{'pixel_values':torch.full((1,3,224,224),int(np.array(images[0])[0,0,0])//40,dtype=torch.float32)}
            with patch.object(reporting,'validate_clip_checkpoint',return_value=state),patch.object(reporting,'DevSchedule',return_value=index),patch.object(reporting,'FrameReader',return_value=reader),patch.object(reporting,'checkpoint_components',return_value=(model,processor)),patch.object(reporting,'CLIPViTGradCAM',return_value=cam):
                out=reporting.explain_clip_dev(run,inputs,root,weights,device='cpu')
            self.assertEqual(sorted(p.stem for p in out.glob('*.png')),['fn','fp','tn','tp'])
            import csv
            with (out/'gradcam_examples.csv').open() as f:export=list(csv.DictReader(f))
            self.assertEqual({r['category'] for r in export},{'tp','tn','fp','fn'})
            self.assertTrue(all(r['source_index']=='2' and r['timestamp_exact']=='1/5' for r in export))
            self.assertEqual(next(r for r in export if r['category']=='fp')['label'],'0')
            for name,value in model.state_dict().items():torch.testing.assert_close(value,original[name],rtol=0,atol=0)


class LosslessCache(unittest.TestCase):
    def setup_cache(self,root):
        (root/'videos/train').mkdir(parents=True);(root/'videos/train/source.mp4').write_bytes(b'fake video stub')
        return RGBCache(root/'cache',{'index':'fixture','pool':'fixture'}),root/'videos'

    def test_lossless_reuse_readonly_integrity_identity_and_capacity(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);cache,videos=self.setup_cache(root);row=record();key,meta=cache.key(row,2,'1/5',videos)
            pixels=np.random.default_rng(4266).integers(0,256,(224,224,3),dtype=np.uint8);cache.put(key,meta,pixels)
            np.testing.assert_array_equal(cache.get(key),pixels)
            readonly=RGBCache(root/'cache',cache.identity,readonly=True);np.testing.assert_array_equal(readonly.get(key),pixels)
            with self.assertRaises(ValueError):RGBCache(root/'cache',{'wrong':'identity'})
            small=RGBCache(root/'small',cache.identity,max_gib=0);small.put(key,meta,pixels);self.assertIsNone(small.get(key))
            path=root/'cache'/key[:2]/(key+'.png');path.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):readonly.get(key)
            with self.assertRaises(PermissionError):cache.key(row|{'split':'test','source_file':'test/a'},2,'1/5',videos)

    def test_reader_decode_once_and_preserve_records_and_transform(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);cache,videos=self.setup_cache(root);pixels=np.full((224,224,3),123,dtype=np.uint8);row=record()
            reader=CachedReader(videos,cache)
            with patch.object(reader.reader,'read',return_value=[[pixels]]) as decode:
                np.testing.assert_array_equal(reader.read([row,row])[0][0],pixels)
                np.testing.assert_array_equal(reader.read([row])[0][0],pixels);self.assertEqual(decode.call_count,1)
            from src.config import Config
            from src.data import LazyDataset
            cfg=Config();pipeline=CachedPipeline(cache);dataset=pipeline.dataset([row],videos,cfg,training=True,epoch=2)
            original=LazyDataset([row],videos,4266,2,.5)
            with patch.object(original.reader,'read',return_value=[[pixels]]):expected=original[0][0]
            torch.testing.assert_close(dataset[0][0],expected,rtol=0,atol=0)
            self.assertIsNone(pipeline.identity);pickle.loads(pickle.dumps(dataset))

    def test_mask_cache_policy_namespace_preserves_original_pixels(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);cache,videos=self.setup_cache(root);row=record();pixels=np.full((224,224,3),123,dtype=np.uint8)
            class Mask:
                identity={'mask':'exact-policy'}
            masked=RGBCache(root/'masked',cache.identity);reader=CachedReader(videos,cache,Mask(),masked)
            mask_reader=SimpleNamespace(transform=lambda r,p:(p.copy(),None,{}));reader.mask_reader=mask_reader
            with patch.object(reader.reader,'read',return_value=[[pixels]]):first=reader.read([row])[0][0]
            reader.mask_reader=SimpleNamespace(transform=lambda *a:(_ for _ in ()).throw(AssertionError('cached mask must be reused')))
            np.testing.assert_array_equal(reader.read([row])[0][0],first)
            self.assertEqual(cache.report()['entries'],1);self.assertEqual(masked.report()['entries'],1)

    def test_cached_reader_is_spawn_loader_compatible(self):
        from src.config import Config
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);cache,videos=self.setup_cache(root);row=record()
            key,meta=cache.key(row,2,'1/5',videos);cache.put(key,meta,np.full((224,224,3),123,dtype=np.uint8))
            loader=CachedPipeline(cache).make_loader([row],videos,Config(workers=1),training=True)
            images,targets,indices=next(iter(loader))
            self.assertEqual(tuple(images.shape),(1,3,224,224));self.assertEqual((targets.tolist(),indices.tolist()),([1.],[0]))


class CLIPRecovery(unittest.TestCase):
    def test_mocked_lifecycle_backup_restore_resume_and_comparison(self):
        """Exercise real persistence/orchestration with mocked epochs, zero model updates."""
        from src import clip_workflow as workflow
        from src.clip_model import MODEL_ID
        from src.clip_reporting import clip_comparison_row,show_six_comparison
        from src.config import POOL_SHA256,INDEX_LOGICAL_SHA256
        from src.evaluation import binary_metrics,class_loss_metrics
        from src.workflow import Inputs
        from src.checkpoints import digest,atomic_checkpoint
        from IPython.utils.capture import capture_output
        from transformers import CLIPImageProcessor
        labels=[0,1,0,1]
        metrics=binary_metrics(labels,[.1,.8,.2,.9]);metrics.update(class_loss_metrics({0:.4,1:1.2},{0:2,1:2}));metrics['loss']=.4
        predictions=[{'record_index':i,'target':label,'probability':p} for i,(label,p) in enumerate(zip(labels,[.1,.8,.2,.9]))]
        rows=[record(y)|{'id':i,'source_file':f'dev/v{i}.mp4','source_family':f'dev/f{i}','split':'dev','video_label':y} for i,y in enumerate(labels)]
        class Index:
            def records(self,*args):return rows
            def close(self):pass
        class Pool:
            sha256=POOL_SHA256
            def records(self,*args):return [record(0),record(1)]
        class Sampler:
            def __init__(self):self.epoch=-1
            def select(self,epoch):
                assert epoch==self.epoch+1;self.epoch=epoch
                return [0,1],{'epoch':epoch+1}
            def state_dict(self):return {'epoch':self.epoch}
            def load_state_dict(self,state):self.epoch=state['epoch']
        def inputs_factory(*args):return Index(),Sampler(),Pool()
        model=tiny_model();model.cuda=lambda:model
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'configs').mkdir();(root/'configs/project.json').write_text('{"project_id":"clip-test"}')
            (root/'src').mkdir();(root/'index').write_bytes(b'index');(root/'pool').write_bytes(b'pool')
            weights=root/'weights';weights.mkdir();(weights/'clip_pretrained_manifest.json').write_text('{"fixture":"no downloaded weights"}')
            inputs=Inputs(root/'videos',root/'index',root/'pool',weights)
            result=SimpleNamespace(testsRun=1,wasSuccessful=lambda:True)
            report={'pool_sha256':POOL_SHA256,'index_logical_sha256':INDEX_LOGICAL_SHA256,'dev_frames':41746,'dev_videos':500}
            calls=[]
            def epoch(model,loader,cfg,**kwargs):
                i=kwargs['epoch'];calls.append((i,kwargs.get('optimizer') is not None))
                probabilities=[.1,.8,.2,.9] if i==1 else [.7,.8,.2,.1]
                value={**metrics,**binary_metrics(labels,probabilities)}
                current=[{'record_index':j,'target':y,'probability':p} for j,(y,p) in enumerate(zip(labels,probabilities))]
                if kwargs.get('optimizer') is not None:
                    optimizer=kwargs['optimizer'];p=optimizer.param_groups[0]['params'][0]
                    if not optimizer.state:
                        optimizer.state[p]={'step':torch.tensor(3.),'exp_avg':torch.full_like(p,.125),'exp_avg_sq':torch.full_like(p,.25)}
                    else:self.assertEqual(float(optimizer.state[p]['step']),3.)
                return value,current,.01
            with patch.object(workflow,'require_prepared_cache'),patch.object(workflow,'verify_clip_cache_loader'),patch.object(workflow,'require_cuda'),patch('torch.nn.Module.cuda',lambda self:self),patch.object(workflow,'prepare_clip_assets',return_value=weights),patch.object(workflow,'cnn_inputs',side_effect=inputs_factory),patch.object(workflow,'CLIPBinaryClassifier',return_value=model),patch.object(workflow,'load_processor',return_value=CLIPImageProcessor()),patch.object(workflow,'make_clip_loader',return_value=[]),patch.object(workflow,'clip_epoch',side_effect=epoch),patch('src.evaluation.save_calibration'),capture_output():
                run=workflow.train_clip(inputs,root,weights,confirmation='TRAIN_'+MODEL_ID,config=CLIPConfig(amp=False),test_result=result,data_report=report)
                self.assertEqual([e for e,training in calls if training],list(range(7)))
                self.assertEqual(len(json.loads((run/'epoch_metrics.json').read_text())['rows'][0]),19)
                receipt=workflow.clip_backup(run,final=True);restored=workflow.restore_clip(receipt['archive'],root)
                self.assertNotEqual(restored,run)
                self.assertEqual(digest(restored/'last.pt'),digest(run/'last.pt'))
                state=workflow.validate_clip_checkpoint(run/'last.pt',root)
                self.assertEqual((state['config']['seed'],state['config']['sampling_seed']),(4266,4266))
                self.assertEqual(state['seed_policy']['head_initialization'],4266)
                self.assertIn('encoder_config',state);self.assertIn('processor_config',state)
                self.assertTrue(any(name.startswith('encoder.') for name in state['model']))
                self.assertTrue(all(name in state['model'] for name in ['classifier.weight','classifier.bias']))
                self.assertEqual(json.loads((run/'config.json').read_text())['seed_policy']['torch_cuda'],4266)
                invalid=dict(state);invalid['config']=dict(state['config'],seed=42)
                atomic_checkpoint(run/'legacy_seed.pt',invalid)
                with self.assertRaisesRegex(ValueError,'recipe differs'):workflow.validate_clip_checkpoint(run/'legacy_seed.pt',root)
                self.assertEqual((state['best_epoch'],state['stale'],state['sampler']['epoch']),(2,5,6))
                original=digest(run/'last.pt');partial=root/'partial';partial.mkdir()
                state.update(epoch=2,history=state['history'][:3],stale=1,sampler={'epoch':2})
                atomic_checkpoint(partial/'last.pt',state)
                import shutil
                shutil.copy2(run/'best.pt',partial/'best.pt')
                calls.clear()
                resumed=workflow.train_clip(inputs,root,weights,confirmation='TRAIN_'+MODEL_ID,config=CLIPConfig(amp=False),test_result=result,data_report=report,resume_from=partial/'last.pt')
                self.assertEqual([e for e,training in calls if training],[3,4,5,6])
                self.assertEqual(digest(run/'last.pt'),original);self.assertNotEqual(run,resumed)
                row=clip_comparison_row(root,run);self.assertEqual(row['Status'],'Completed, verified checkpoint')
                self.assertEqual(row['Video F2'],metrics['f2'])
                self.assertEqual(clip_comparison_row(root)['Status'],'Pending')
                table,ranking,out=show_six_comparison(root,clip_run=run)
                self.assertEqual(len(table),6);self.assertEqual(len(ranking),1)
                (run/'best.pt').rename(run/'moved.pt')
                self.assertEqual(clip_comparison_row(root,run)['Status'],'Missing checkpoint')
                (run/'training_summary.json').unlink()
                self.assertEqual(clip_comparison_row(root,run)['Status'],'Partial')


if __name__=='__main__':unittest.main()
