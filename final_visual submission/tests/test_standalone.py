import unittest,tempfile,json,hashlib,copy,ast,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fractions import Fraction
from unittest.mock import patch
import numpy as np
import torch
torch.set_num_threads(2)
from src.config import ORDER,configuration,declared_configurations
from src.workflow import guard_dev,execute_training,Inputs
from src.models import StagedCNN
from src.temporal import ResNetLSTM,clip_rows
from src.evaluation import binary_metrics,video_predictions
from src.training import GradientGuard,NumericalInstabilityError
from src.data import make_loader
from src.face_mask import MaskConfig,masked_frame


class DummyEncoder(torch.nn.Module):
    def __init__(self):super().__init__();self.fc=torch.nn.Linear(3,2048)
    def forward(self,x):return self.fc(x.mean((2,3)))

class ConfigAndModels(unittest.TestCase):
    def test_scheduler_matches_checkpoint_metric(self):
        # Exercise the production constructor and step expression without training.
        from src.config import checkpoint_score,selection_label
        import src.training as training
        tree=ast.parse(Path(training.__file__).read_text(encoding='utf-8'))
        loop=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_train')
        constructor=next(n for n in ast.walk(loop) if isinstance(n,ast.Assign)
                         and any(isinstance(t,ast.Name) and t.id=='scheduler' for t in n.targets))
        score_node=next(n for n in ast.walk(loop) if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='score' for t in n.targets))
        step=next(n for n in ast.walk(loop) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                  and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
                  and n.value.func.value.id=='scheduler' and n.value.func.attr=='step')
        self.assertEqual(ast.dump(step.value.args[0]),ast.dump(ast.Name(id='score',ctx=ast.Load())))
        init_code=compile(ast.Module(body=[constructor],type_ignores=[]),'<production scheduler>','exec')
        step_code=compile(ast.Module(body=[score_node,step],type_ignores=[]),'<production scheduler step>','exec')
        declared=declared_configurations()
        for key in ORDER:
            with self.subTest(model=key):
                config=configuration(key,amp=False)
                self.assertEqual(declared[key]['scheduler_monitor'],selection_label(config.checkpoint_selection))
                self.assertEqual(declared[key]['scheduler_monitor'],declared[key]['checkpoint_monitor'])
                model=torch.nn.Linear(1,1)
                optimizer=torch.optim.AdamW(model.parameters(),lr=config.head_lr)
                env={'torch':torch,'config':config,'optimizer':optimizer,'checkpoint_score':checkpoint_score}
                exec(init_code,env);scheduler=env['scheduler']
                self.assertEqual((scheduler.mode,scheduler.factor,scheduler.patience,scheduler.min_lrs),('max',.5,1,[0]))
                # Unit ROC improves while both old monitor metrics decline: no LR drop.
                for roc,other in [(.6,.9),(.7,.8),(.8,.7)]:
                    env['report']={'unit':{'roc_auc':roc,'pr_auc':other},'video':{'roc_auc':other}}
                    exec(step_code,env)
                    self.assertEqual(env['score'],roc)
                    self.assertEqual(optimizer.param_groups[0]['lr'],config.head_lr)
                # Unit ROC plateaus while old monitors improve: reduce after patience.
                for other in [.8,.9]:
                    env['report']={'unit':{'roc_auc':.8,'pr_auc':other},'video':{'roc_auc':other}}
                    exec(step_code,env)
                self.assertEqual(optimizer.param_groups[0]['lr'],config.head_lr*.5)

    def test_five_configs(self):
        self.assertEqual(len(ORDER),5)
        for key in ORDER:
            c=configuration(key);self.assertEqual((c.head_lr,c.epochs,c.patience,c.seed,c.dropout),(1e-4,10,3,4266,.3))
            self.assertEqual(c.checkpoint_selection,'clip_roc_auc' if key=='resnet50_lstm' else 'frame_roc_auc')
            if key!='resnet50_lstm':self.assertEqual((c.backbone_lr,c.batch_size,c.warmup_epochs),(1.5e-5,32,2))
            else:self.assertEqual((c.batch_size,c.clip_per_class,c.hidden_size,c.lstm_layers),(8,1500,128,1))

    def test_both_cnn_shapes_and_bn(self):
        for key in ['resnet50_baseline','efficientnet_b0_baseline']:
            c=configuration(key,amp=False);torch.manual_seed(4266);m=StagedCNN(c,'.',pretrained=False)
            for fine in [False,True]:
                m.set_phase(fine);m.train();report=m.trainable_report();self.assertTrue(report['batchnorm_running_stats_frozen'])
                with torch.no_grad():self.assertEqual(tuple(m(torch.zeros(1,3,224,224)).shape),(1,1))
                self.assertFalse(any(x.training for x in m.modules() if isinstance(x,torch.nn.BatchNorm2d)))
            del m

    def test_temporal_encoder_frozen(self):
        c=configuration('resnet50_lstm',amp=False);m=ResNetLSTM(DummyEncoder(),c);m.train()
        logits=m(torch.zeros(1,16,3,224,224));self.assertEqual(tuple(logits.shape),(1,1));logits.sum().backward()
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in m.encoder.parameters()))
        self.assertIsNotNone(m.lstm.weight_ih_l0.grad);self.assertEqual(tuple(m.lstm.weight_ih_l0.shape),(512,2048))

    def test_imagenet_load_request_independent(self):
        # No pretrained downloads and no use of any historical trained checkpoint.
        for key in ['resnet50_baseline','resnet50_face_mask','efficientnet_b0_baseline','efficientnet_b0_face_mask']:
            with patch('src.models.build_model',return_value=torch.nn.Identity()) as build:
                with patch.object(StagedCNN,'set_phase'):
                    StagedCNN(configuration(key),'/new_imagenet_weights',pretrained=True)
                self.assertTrue(build.call_args.kwargs['pretrained'])

class DataAndMetrics(unittest.TestCase):
    def test_best_checkpoint_predictions_reused_without_inference(self):
        from src.training import checkpoint_dev_report
        from src.evaluation import evaluate_predictions
        records=[{'label':y,'metric_eligible':True,'source_file':f'dev/{i}','source_family':f'dev/{i}','video_label':y}
                 for i,y in enumerate([0,1,0,1])]
        predictions=[{'record_index':i,'target':r['label'],'probability':p} for i,(r,p) in enumerate(zip(records,[.1,.9,.3,.8]))]
        policy={'aggregation':'topk','topk_fraction':.05,'prediction_threshold':.5,'video_threshold':.5}
        losses={'loss':.23456789,'real_class_loss':.1,'fake_class_loss':.4,'balanced_loss':.25}
        expected=evaluate_predictions(records,predictions,policy)
        for key in ORDER:
            cfg=configuration(key);kind='clip' if key=='resnet50_lstm' else 'frame'
            state={'epoch':1,'best_epoch':2,'best':expected['unit']['roc_auc'],'config':cfg.as_dict(),
                   'dev_predictions':predictions,'history':[{'epoch':1},{'epoch':2,'dev':losses}]}
            with patch('src.training.run_epoch',side_effect=AssertionError('No inference allowed')):
                saved,report=checkpoint_dev_report(state,records,policy,kind)
            self.assertEqual(saved,predictions);self.assertEqual(report['losses'],losses)
            self.assertEqual(report['unit'],expected['unit']);self.assertEqual(report['video'],expected['video'])
            self.assertIsNone(checkpoint_dev_report({k:v for k,v in state.items() if k!='dev_predictions'},records,policy,kind))
            bad=copy.deepcopy(state);bad['dev_predictions'][0]['target']=1
            with self.assertRaises(ValueError):checkpoint_dev_report(bad,records,policy,kind)
            bad=copy.deepcopy(state);bad['best']=.5
            with self.assertRaises(ValueError):checkpoint_dev_report(bad,records,policy,kind)

    def test_one_fake_clip_and_stride(self):
        ts=[Fraction(i,10) for i in range(40)];rows=clip_rows(1,list(range(40)),ts,[[Fraction(1,2),Fraction(3,5)]],True)
        self.assertEqual([r['position'] for r in rows],[0,12,24]);self.assertEqual(rows[0]['label'],1)

    def test_video_top5_is_mean_and_f2(self):
        records=[{'source_file':'dev/a','source_family':'dev/a','video_label':1} for _ in range(40)]
        predictions=[{'record_index':i,'probability':p} for i,p in enumerate([.9,.7]+[.1]*38)]
        self.assertAlmostEqual(video_predictions(records,predictions,{'aggregation':'topk','topk_fraction':.05})[0]['probability'],.8)
        m=binary_metrics([1,1,0,0],[.9,.1,.8,.1],.5);self.assertAlmostEqual(m['f2'],5/(5+4+1))

    def test_mask_fallback_and_label_independence(self):
        image=np.full((224,224,3),90,np.uint8);meta={'status':'fallback_no_face','confidence':None,'crop_xyxy':[0,0,224,224]}
        result,alpha,r=masked_frame(image,None,meta,MaskConfig(),failure='no_face')
        np.testing.assert_array_equal(result,image);self.assertEqual(r['status'],'fallback_no_face');self.assertTrue((alpha==1).all())

    def test_mask_interior_pixels_and_geometry(self):
        # Synthetic dense oval; no images or labels from the dataset.
        a=np.linspace(0,2*np.pi,478,endpoint=False);points=np.stack([112+45*np.cos(a),110+65*np.sin(a)],1).astype(np.float32)
        image=np.random.default_rng(1).integers(0,255,(224,224,3),dtype=np.uint8)
        out,alpha,r=masked_frame(image,points,{'status':'detected','confidence':.9,'crop_xyxy':[20,20,204,204]})
        self.assertEqual(r['status'],'masked');np.testing.assert_array_equal(out[alpha==1],image[alpha==1]);self.assertTrue((out[alpha==0]==128).all())

    def test_spawn_loader(self):
        import av
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'train').mkdir();video=root/'train/x.mkv'
            with av.open(str(video),'w') as c:
                s=c.add_stream('ffv1',rate=25);s.width=s.height=224;s.pix_fmt='bgr0'
                f=av.VideoFrame.from_ndarray(np.full((224,224,3),90,np.uint8),format='rgb24')
                for packet in s.encode(f):c.mux(packet)
                for packet in s.encode():c.mux(packet)
            row={'id':0,'kind':'frame','split':'train','source_file':'train/x.mkv','indices':[0],'timestamps_exact':['0'],'label':0}
            c=configuration('resnet50_baseline',amp=False,workers=0);a=next(iter(make_loader([row],root,c)))[0]
            c.workers=1;b=next(iter(make_loader([row],root,c)))[0];torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_same_mask_loader_train_dev_with_cached_geometry(self):
        import av
        from types import SimpleNamespace
        from src.face_mask import BoxCache,CropConfig
        from src.face_mask import FaceMaskPipeline
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);identity={'kind':'face_mask','fixture':True};rows=[]
            angles=np.linspace(0,2*np.pi,478,endpoint=False);points=np.stack([112+45*np.cos(angles),110+65*np.sin(angles)],1).astype(np.float32)
            image=np.full((224,224,3),90,np.uint8)
            _,_,metadata=masked_frame(image,points,{'status':'detected','confidence':.9,'crop_xyxy':[20,20,204,204]})
            for split in ['train','dev']:
                (root/split).mkdir()
                with av.open(str(root/split/'x.mkv'),'w') as c:
                    s=c.add_stream('ffv1',rate=25);s.width=s.height=224;s.pix_fmt='bgr0'
                    for packet in s.encode(av.VideoFrame.from_ndarray(image,format='rgb24')):c.mux(packet)
                    for packet in s.encode():c.mux(packet)
                rows.append({'id':len(rows),'kind':'frame','split':split,'source_file':split+'/x.mkv','indices':[0],'timestamps_exact':['0'],'label':len(rows)})
            cache=BoxCache(root/'geometry.sqlite',identity)
            for row in rows:cache.put(row,metadata)
            cache.close()
            pipeline=SimpleNamespace(weights=root/'not-needed.onnx',landmark_weights=root/'not-needed.task',cache_path=root/'geometry.sqlite',identity=identity,config=CropConfig(),mask_config=MaskConfig())
            cfg=configuration('resnet50_baseline',amp=False,workers=0);cfg.flip_probability=0
            train=FaceMaskPipeline.dataset(pipeline,[rows[0]],root,cfg,training=True)[0][0]
            dev=FaceMaskPipeline.dataset(pipeline,[rows[1]],root,cfg,training=False)[0][0]
            torch.testing.assert_close(train,dev,rtol=0,atol=0)

class SafetyAndResume(unittest.TestCase):
    def test_bounded_preflight_actual_batch_and_zero_updates(self):
        from contextlib import ExitStack
        from src.workflow import preflight
        class TinyCNN(torch.nn.Module):
            def __init__(self,*args,**kwargs):
                super().__init__();self.final=torch.nn.Linear(3,3);self.classifier=torch.nn.Linear(3,1)
            def head(self):return self.classifier
            def final_parameters(self):return list(self.final.parameters())
            def set_phase(self,fine):
                self.phase='final_stage' if fine else 'head'
                for p in self.final.parameters():p.requires_grad_(fine)
            def forward(self,x):return self.classifier(self.final(x.mean((2,3))))
        rows=[{'id':i,'label':i%2} for i in range(100)]
        class Pool:
            def __init__(self,*a):pass
            def records(self,ids):return [rows[i] for i in ids]
        class Sampler:
            def __init__(self,*a):pass
            def select(self,e):return list(range(100)),{'samples':8000,'real':4000,'fake':4000,'duplicate_rate':0.}
        class Dev:
            def __init__(self,*a):pass
            def records(self,*a):return rows
            def close(self):pass
        seen=[]
        def loader(records,*args,**kwargs):
            seen.append((len(records),kwargs.get('training',False)))
            return [(torch.ones(len(records),3,224,224),torch.tensor([r['label'] for r in records],dtype=torch.float32),torch.arange(len(records)))]
        model=TinyCNN();before=copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as t,ExitStack() as stack:
            for name,value in [('require_cuda',lambda:None),('validate_inputs',lambda x:None),('FixedPool',Pool),
                               ('FixedPoolSampler',Sampler),('DevSchedule',Dev),('make_loader',loader),('StagedCNN',lambda *a,**k:model)]:
                stack.enter_context(patch('src.workflow.'+name,value))
            stack.enter_context(patch.object(torch.nn.Module,'cuda',lambda x:x));stack.enter_context(patch.object(torch.Tensor,'cuda',lambda x:x))
            for name in ['synchronize','reset_peak_memory_stats']:stack.enter_context(patch('torch.cuda.'+name,return_value=None))
            for name in ['max_memory_allocated','max_memory_reserved']:stack.enter_context(patch('torch.cuda.'+name,return_value=2**20))
            stack.enter_context(patch('torch.cuda.mem_get_info',return_value=(10*2**30,15*2**30)))
            stack.enter_context(patch('torch.cuda.get_device_name',return_value='CPU fixture'))
            p=Path(t);report=preflight('resnet50_baseline',Inputs(p,p,p,p),p,amp=False,workers=0)
        self.assertEqual(seen,[(32,True),(32,False)])
        self.assertEqual(report['optimizer_steps'],0);self.assertFalse(report['full_dev_inference_performed'])
        self.assertEqual([p['phase'] for p in report['phase_measurements']],['head','final_stage'])
        for k,v in before.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)

    def test_training_and_test_gate(self):
        with self.assertRaises(PermissionError):execute_training(ORDER,None,ROOT)
        with self.assertRaises(PermissionError):guard_dev('test')
        from src.reporting import explain_dev,evaluate_dev
        for fn in [explain_dev,evaluate_dev]:
            with self.assertRaises(PermissionError):fn('absent',None,ROOT,split='test')

    def test_nonfinite_fp32_does_not_step(self):
        c=configuration('resnet50_baseline',amp=False);model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters());before=copy.deepcopy(model.state_dict())
        for p in model.parameters():p.grad=torch.full_like(p,float('inf'))
        g=GradientGuard(c);g.begin_epoch(0);scaler=torch.amp.GradScaler('cpu',enabled=False)
        with self.assertRaises(NumericalInstabilityError):g.finish_step(opt,scaler,epoch=0,batch=1,emit=lambda x:None)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_amp_overflow_skips_and_updates_scale_on_cpu(self):
        c=configuration('resnet50_baseline');m=torch.nn.Linear(1,1);opt=torch.optim.AdamW(m.parameters());before=copy.deepcopy(m.state_dict())
        scaler=torch.amp.GradScaler('cpu',enabled=True);scaler.scale(m(torch.ones(1,1)).sum()).backward()
        for p in m.parameters():p.grad.fill_(float('inf'))
        scale=scaler.get_scale();g=GradientGuard(c);g.begin_epoch(0)
        g.finish_step(opt,scaler,epoch=0,batch=1,emit=lambda x:None)
        self.assertLess(scaler.get_scale(),scale)
        for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_gradcam_hook_cleanup_and_unchanged_weights(self):
        from src.gradcam import GradCAM
        class Tiny(torch.nn.Module):
            def __init__(self):super().__init__();self.conv=torch.nn.Conv2d(3,2,1);self.head=torch.nn.Linear(2,1)
            def forward(self,x):return self.head(self.conv(x).mean((2,3)))
        m=Tiny().eval();before=copy.deepcopy(m.state_dict());hooks=len(m.conv._forward_hooks)
        r=GradCAM(m,m.conv)(torch.ones(1,3,8,8));self.assertEqual(r['heatmaps'].shape,(1,8,8));self.assertEqual(len(m.conv._forward_hooks),hooks)
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_exact_resume_state(self):
        from src.checkpoints import rng_state,load_resume_bundle,restore_training_state,atomic_checkpoint
        from src.data import PROTOCOL
        c=configuration('resnet50_baseline',amp=False,workers=0);model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters(),lr=c.head_lr);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
        (model(torch.ones(1,2)).sum()).backward();opt.step();scheduler.step(.7)
        state={'protocol':PROTOCOL,'config':c.as_dict(),'index_sha256':'fixture-index','pool_sha256':'fixture-pool','epoch':0,'history':[{'epoch':1,'dev':{'roc_auc':.7},'dev_video':{'roc_auc':.6}}],'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':{},'rng':rng_state(),'sampler':{'epoch':0,'coverage':[1,2]},'best':.7,'stale':0,'best_epoch':1,'best_balanced_dev_loss':.5}
        class Sampler:
            def load_state_dict(self,s):self.state=s
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);atomic_checkpoint(p/'best.pt',state);atomic_checkpoint(p/'last.pt',state)
            b=load_resume_bundle(p/'last.pt',c,'fixture-index',pool_hash='fixture-pool')
            new=torch.nn.Linear(2,1);o=torch.optim.AdamW(new.parameters());s=torch.optim.lr_scheduler.ReduceLROnPlateau(o);sampler=Sampler()
            restore_training_state(b,new,o,s,torch.amp.GradScaler('cpu',enabled=False),sampler)
            self.assertEqual(b['report']['next_epoch'],2);self.assertEqual(sampler.state,state['sampler']);self.assertEqual(o.state_dict()['param_groups'],opt.state_dict()['param_groups'])
            for k,v in new.state_dict().items():torch.testing.assert_close(v,state['model'][k],rtol=0,atol=0)

    def test_new_session_zip_resumes_next_epoch_with_all_state(self):
        from src.checkpoints import rng_state,restore_rng,restore_training_state
        from src.backup import package_model_run
        from src.backup import restore_run
        from src.workflow import resume_candidate
        from src.data import PROTOCOL
        from src.data import FixedPoolSampler
        from src.config import POOL_SHA256
        import random
        class FixturePool:
            sha256=POOL_SHA256
            def __init__(self):
                self.frames={i:{'id':i,'label':i%2,'source_file':f'train/{i//2}',
                                'source_family':f'train/family{i//4}'} for i in range(24)}
            def records(self,ids):return [self.frames[i] for i in ids]
        cfg=configuration('resnet50_baseline',amp=True,workers=0)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old=root/'session1';new=root/'session2'
            for p in [old,new]:
                (p/'configs').mkdir(parents=True);(p/'configs/project.json').write_text('{"project_id":"fixture"}')
            index=root/'index';index.write_text('synthetic index; no dataset')
            run=old/'checkpoints/resnet50_baseline/segment/run';run.mkdir(parents=True)
            identity={'project_id':'fixture','model_id':'resnet50_baseline','run_id':'segment'}
            (run/'experiment_identity.json').write_text(json.dumps(identity))
            model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters(),lr=cfg.head_lr)
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=1)
            scaler=torch.amp.GradScaler('cpu',enabled=True)
            scaler.scale(model(torch.ones(1,2)).sum()).backward();scaler.step(opt);scaler.update();scheduler.step(.7)
            sampler=FixedPoolSampler(FixturePool(),2,cfg.seed);sampler.select(0)
            history=[{'epoch':1,'dev':{'roc_auc':.7},'dev_video':{'roc_auc':.6}}]
            state={'protocol':PROTOCOL,'config':cfg.as_dict(),'experiment_identity':identity,
                   'index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),'pool_sha256':POOL_SHA256,
                   'epoch':0,'history':history,'model':model.state_dict(),'optimizer':opt.state_dict(),
                   'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(),'sampler':sampler.state_dict(),
                   'best':.7,'best_epoch':1,'stale':0,'best_balanced_dev_loss':.5}
            for name in ['best.pt','last.pt']:torch.save(state,run/name)
            for name in ['config.json','history.csv','history.jsonl']:(run/name).write_text('{}')
            before={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ['best.pt','last.pt']}
            archive=package_model_run(run,'resnet50',1)['archive'];restored=restore_run(archive,new)
            inputs=Inputs(root/'videos',index,root/'pool',root/'weights')
            source,report=resume_candidate('resnet50_baseline',inputs,new,source=restored/'last.pt',amp=True,workers=0)
            self.assertEqual(report['next_epoch'],2)
            from src.checkpoints import load_resume_bundle
            b=load_resume_bundle(source,cfg,state['index_sha256'],pool_hash=state['pool_sha256'])
            new_model=torch.nn.Linear(2,1);new_opt=torch.optim.AdamW(new_model.parameters())
            new_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(new_opt,mode='max')
            new_sampler=FixedPoolSampler(FixturePool(),2,cfg.seed)
            restore_rng(state['rng']);expected_rng=(random.random(),np.random.random(),torch.rand(1))
            new_scaler=torch.amp.GradScaler('cpu',enabled=True)
            restore_training_state(b,new_model,new_opt,new_scheduler,new_scaler,new_sampler)
            actual_rng=(random.random(),np.random.random(),torch.rand(1))
            self.assertEqual(actual_rng[:2],expected_rng[:2]);torch.testing.assert_close(actual_rng[2],expected_rng[2])
            self.assertEqual(new_scheduler.state_dict(),scheduler.state_dict())
            self.assertEqual(new_scaler.state_dict(),scaler.state_dict())
            self.assertEqual(new_sampler.select(1),sampler.select(1))
            self.assertEqual(new_opt.state_dict()['param_groups'],opt.state_dict()['param_groups'])
            for key,values in new_opt.state_dict()['state'].items():
                for name,value in values.items():torch.testing.assert_close(value,opt.state_dict()['state'][key][name])
            for name,h in before.items():self.assertEqual(hashlib.sha256((run/name).read_bytes()).hexdigest(),h)
            with self.assertRaises(ValueError):resume_candidate('resnet50_baseline',inputs,new,source=new/'missing.pt',amp=False)

    def test_verified_zip_roundtrip_and_tamper(self):
        from src.backup import package_model_run,restore_local_zip,REQUIRED
        import zipfile
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);run=root/'version/run';run.mkdir(parents=True)
            for n in REQUIRED:(run/n).write_text('synthetic fixture')
            receipt=package_model_run(run,'resnet50',1);out=restore_local_zip(receipt['archive'],root/'restored')
            self.assertIn('version',Path(receipt['archive']).name)
            self.assertEqual((out/'best.pt').read_text(),'synthetic fixture')
            with zipfile.ZipFile(receipt['archive'],'a') as z:z.writestr('unlisted.txt','tamper')
            with self.assertRaises(ValueError):restore_local_zip(receipt['archive'],root/'bad')

    def test_pending_comparison_not_fabricated(self):
        from src.reporting import comparison
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'configs').mkdir();(p/'configs/project.json').write_text('{"project_id":"fixture"}')
            rows,rank,out=comparison(p);self.assertEqual(len(rows),5);self.assertFalse(rank);self.assertTrue(all(r['Best epoch'] is None for r in rows))
            self.assertTrue(all(r['Threshold']==.5 and r['Checkpoint selection'].endswith('ROC-AUC') for r in rows))

    def test_no_reference_imports(self):
        for p in (ROOT/'src').rglob('*.py'):
            t=p.read_text(encoding='utf-8');self.assertNotIn('sys.path.insert',t);self.assertNotIn('from linked',t);self.assertNotIn('import linked',t)

    def test_resume_identity_rebind_only_new_copies(self):
        import shutil
        from src.checkpoints import bind_resume_identity
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old=root/'old';new=root/'new';old.mkdir();new.mkdir()
            identity={'project_id':'fixture','model_id':'resnet50_baseline','run_id':'old'}
            for name in ['best.pt','last.pt']:torch.save({'model':{'x':torch.ones(2)},'experiment_identity':identity},old/name);shutil.copy2(old/name,new/name)
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in old.iterdir()}
            (new/'experiment_identity.json').write_text(json.dumps({**identity,'run_id':'new'}));bind_resume_identity(new)
            for name in before:
                self.assertEqual(hashlib.sha256((old/name).read_bytes()).hexdigest(),before[name])
                state=torch.load(new/name,weights_only=True);self.assertEqual(state['experiment_identity']['run_id'],'new');torch.testing.assert_close(state['model']['x'],torch.ones(2))

    def test_historical_encoder_is_refused(self):
        from src.workflow import new_encoder
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'configs').mkdir();(root/'configs/project.json').write_text('{"project_id":"new"}')
            torch.save({'config':{'model':'resnet50'}},root/'best.pt')
            with self.assertRaises(ValueError):new_encoder(root/'best.pt',root)

    def test_all_five_sequential_and_new_encoder_wiring(self):
        from src.workflow import execute_training
        from src.data import PROTOCOL
        called=[]
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'configs').mkdir();(p/'configs/project.json').write_text('{"project_id":"fixture"}');(p/'docs').mkdir();(p/'docs/source_provenance.json').write_text('{}')
            inputs=Inputs(p/'videos',p/'index',p/'pool',p/'imagenet',p/'clips')
            def fake_train(cfg,index,data,weights,out,*args,**kwargs):
                out=Path(out);out.mkdir();identity=json.loads((out.parent/'experiment_identity.json').read_text())
                called.append((identity['model_id'],Path(weights),out))
                state={'protocol':PROTOCOL,'config':cfg.as_dict(),'experiment_identity':identity,'model':{'fixture':torch.ones(1)},'input_pipeline':None,'epoch':0,'best_epoch':1}
                torch.save(state,out/'best.pt');return out
            with patch('src.workflow.preflight',return_value={'optimizer_steps':0}),patch('src.workflow.train',side_effect=fake_train),patch('src.workflow.mask_pipeline',return_value=object()),patch('src.reporting.export_report'):
                result=execute_training(ORDER,inputs,p,confirmation='TRAIN_SELECTED_MODELS')
            self.assertEqual([x[0] for x in called],list(ORDER));self.assertEqual(len({x[2] for x in called}),5)
            self.assertEqual(called[2][1],result['resnet50_baseline']/'best.pt')
            for i in [0,1,3,4]:self.assertEqual(called[i][1],inputs.weights)

    def test_new_project_report_backup_restore_paths(self):
        from src.reporting import export_report
        from src.backup import backup_run,restore_run
        from src.data import PROTOCOL
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'configs').mkdir();(p/'configs/project.json').write_text('{"project_id":"fixture"}')
            run=p/'checkpoints/resnet50_baseline/segment/run';run.mkdir(parents=True)
            identity={'project_id':'fixture','model_id':'resnet50_baseline','run_id':'segment','label':'ResNet50 baseline','input':'full_frame'}
            (run/'experiment_identity.json').write_text(json.dumps(identity));(run/'training_summary.json').write_text('{"best_epoch":1}')
            (run/'history.csv').write_text('epoch\n1\n');(run/'history.jsonl').write_text('{"epoch":1}\n')
            cfg=configuration('resnet50_baseline',amp=False).as_dict();(run/'config.json').write_text(json.dumps(cfg))
            state={'protocol':PROTOCOL,'config':cfg,'experiment_identity':identity,'model':{'fixture':torch.ones(1)},'epoch':0,'best_epoch':1,'best':.7,'history':[{'epoch':1}],'index_sha256':'fixture','pool_sha256':'fixture'}
            for name in ['best.pt','last.pt']:torch.save(state,run/name)
            out=export_report(run,p);self.assertEqual(out,p/'results/resnet50_baseline/segment');self.assertFalse(list(out.glob('*.pt')))
            with self.assertRaises(FileExistsError):export_report(run,p)
            receipt=backup_run(run,p,final=True);restored=restore_run(receipt['archive'],p)
            self.assertIn('segment',Path(receipt['archive']).name)
            self.assertTrue((restored/'last.pt').is_file());self.assertNotEqual(restored,run)
            foreign=p/'foreign';(foreign/'configs').mkdir(parents=True);(foreign/'configs/project.json').write_text('{"project_id":"different"}')
            with self.assertRaises(ValueError):restore_run(receipt['archive'],foreign)

if __name__=='__main__':unittest.main()
