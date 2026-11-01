"""Software tests only: synthetic media/arrays and explicitly labelled model doubles."""
from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));torch.set_num_threads(2)
from common import digest,file_stat,json_write,sha256
from flow_frontend import DenseFlow,encode_flow_image,stack_pair_indices,pack_stack
from tsn_encoder import make_backbone,map_key,convert_backbone_state,load_checkpoint
from extract_tsn_flow import (read_reference,choose_rows,read_reference_arrays,compute_video,
                             atomic_npz,validate_cache,array_hash,run,REFERENCE_KIND)
from video_io import probe_video
from visual_data import FEATURE_KIND,VisualDataset,load_manifest,resample_at_bin_centers,MixedBatchSampler


@pytest.mark.parametrize('source,dest',[
 ('conv1.conv.weight','conv1.weight'),('conv1.bn.weight','bn1.weight'),
 ('conv1.bn.running_var','bn1.running_var'),
 ('layer1.0.conv1.conv.weight','layer1.0.conv1.weight'),
 ('layer3.4.conv2.bn.running_mean','layer3.4.bn2.running_mean'),
 ('layer4.0.conv3.bn.num_batches_tracked','layer4.0.bn3.num_batches_tracked'),
 ('layer2.0.downsample.conv.weight','layer2.0.downsample.0.weight'),
 ('layer2.0.downsample.bn.bias','layer2.0.downsample.1.bias')])
def test_key_mapping(source,dest):assert map_key(source)==dest


def mm_key(key):
    if key=='conv1.weight':return 'backbone.conv1.conv.weight'
    if key.startswith('bn1.'):return 'backbone.conv1.bn.'+key[4:]
    import re
    m=re.fullmatch(r'(layer[1-4]\.\d+)\.conv([123])\.(.+)',key)
    if m:return f'backbone.{m[1]}.conv{m[2]}.conv.{m[3]}'
    m=re.fullmatch(r'(layer[1-4]\.\d+)\.bn([123])\.(.+)',key)
    if m:return f'backbone.{m[1]}.conv{m[2]}.bn.{m[3]}'
    m=re.fullmatch(r'(layer[1-4]\.\d+)\.downsample\.([01])\.(.+)',key)
    if m:return f'backbone.{m[1]}.downsample.{"conv" if m[2]=="0" else "bn"}.{m[3]}'
    raise AssertionError(key)


@pytest.fixture(scope='module')
def model_and_state():
    torch.manual_seed(12);model=make_backbone().eval()
    state={mm_key(k):v for k,v in model.state_dict().items()}
    state['cls_head.fc_cls.weight']=torch.zeros(200,2048)
    state['cls_head.fc_cls.bias']=torch.zeros(200)
    return model,state


def test_mapping_complete_and_random_backbone_forward(model_and_state):
    model,state=model_and_state
    mapped,excluded=convert_backbone_state(state,model)
    assert len(excluded)==2 and set(mapped)==set(model.state_dict())
    for k,v in model.state_dict().items():assert torch.equal(v,mapped[k])
    model.load_state_dict(mapped,strict=True)
    with torch.inference_mode():features=model(torch.zeros(1,10,256,256))
    assert features.shape==(1,2048) and torch.isfinite(features).all()


def test_missing_learned_weight_fails(model_and_state):
    model,state=model_and_state;s=dict(state);s.pop('backbone.layer1.0.conv1.conv.weight')
    with pytest.raises(ValueError,match='Incomplete'):convert_backbone_state(s,model)


def test_rgb_checkpoint_shape_fails(model_and_state):
    model,state=model_and_state;s=dict(state);s['backbone.conv1.conv.weight']=torch.zeros(64,3,7,7)
    with pytest.raises(ValueError,match='Shape'):convert_backbone_state(s,model)


def test_unknown_backbone_key_fails():
    with pytest.raises(ValueError):map_key('imaginary.weight')


def test_nonfinite_pretrained_tensor_fails(model_and_state):
    model,state=model_and_state;s=dict(state);s['backbone.conv1.bn.weight']=torch.full((64,),float('nan'))
    with pytest.raises(ValueError,match='Nonfinite'):convert_backbone_state(s,model)


def test_absent_bn_counter_only_allowed(model_and_state):
    model,state=model_and_state;s={k:v for k,v in state.items() if not k.endswith('num_batches_tracked')}
    m,_=convert_backbone_state(s,model)
    assert all(int(v)==0 for k,v in m.items() if k.endswith('num_batches_tracked'))


def test_wrong_checkpoint_hash_fails_before_unpickling(tmp_path):
    p=tmp_path/'wrong.pth';p.write_bytes(b'not an official checkpoint')
    with pytest.raises(ValueError,match='SHA256'):load_checkpoint(p)


def test_flow_quantization_channels_and_zero():
    z=encode_flow_image(np.zeros((224,224,2),np.float32))
    assert z.shape==(2,256,256) and np.all(z==128)
    assert np.all(pack_stack([z]*5)==0)
    f=np.zeros((224,224,2),np.float32);f[...,0]=100;f[...,1]=-100
    e=encode_flow_image(f);s=pack_stack([e]*5)
    assert np.all(s[0::2]==127/128) and np.all(s[1::2]==-1)


def test_stack_chronological_interleaving():
    pairs=[np.stack([np.full((256,256),100+i,np.uint8),np.full((256,256),150+i,np.uint8)]) for i in range(5)]
    s=pack_stack(pairs)
    np.testing.assert_allclose(s[:,0,0],(np.array([100,150,101,151,102,152,103,153,104,154])-128)/128)


def test_native_context_edge_repetition():
    p=stack_pair_indices(np.array([0,3,7]),8)
    np.testing.assert_array_equal(p,[[1,1,1,1,2],[1,2,3,4,5],[5,6,7,7,7]])
    np.testing.assert_array_equal(stack_pair_indices(np.array([0,1]),2),np.ones((2,5)))


def test_dense_flow_actual_translation():
    import cv2
    x=np.random.default_rng(12).integers(0,256,(224,224),dtype=np.uint8)
    y=cv2.warpAffine(x,np.float32([[1,0,2],[0,1,0]]),(224,224))
    f=DenseFlow('farneback')(x,y)
    assert np.median(f[30:-30,30:-30,0])==pytest.approx(2.,rel=.15)
    assert np.median(np.abs(f[30:-30,30:-30,1]))<.1


def test_tvl1_has_no_silent_fallback(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2,'optflow',None,raising=False)
    with pytest.raises(ImportError,match='contrib'):DenseFlow('tvl1')


class ExplicitFakeEncoder:
    identity={'kind':'EXPLICIT_TEST_DOUBLE_NOT_PRETRAINED','test_only':True}
    def encode(self,stacks):
        x=np.asarray(stacks);assert x.shape[1:]==(10,256,256)
        return np.repeat(x.mean(axis=(1,2,3))[:,None],2048,axis=1).astype(np.float32)


def video(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x64:rate=20:duration=0.5',
                    '-c:v','mpeg4','-threads','1','-y',str(path)],check=True,timeout=20)


def reference_fixture(tmp_path,per_split=2):
    source=tmp_path/'source';cache=tmp_path/'reference';rows=[]
    for split in ('train','dev'):
        for i in range(per_split):
            name=f'{split}/{i}.mp4';v=source/name;video(v)
            ts=np.arange(5,dtype=np.float64)/10;idx=np.arange(0,10,2,dtype=np.int64)
            rgb=np.repeat((ts+i)[:,None],512,axis=1).astype(np.float32)
            p=cache/'clips'/split/f'{i}.npz'
            atomic_npz(p,{'rgb':rgb,'flow':np.zeros_like(rgb),'timestamp_s':ts,'source_frame_index':idx})
            rows.append({'file':name,'split':split,'source_group':name,'modify_video':bool(i%2),'modify_audio':False,
                         'visual_periods':[[.1,.3]] if i%2 else [],'duration_s':.5,'feature_count':5,
                         'source_stat':file_stat(v),'config_id':'REFERENCE_TEST','warnings':[],
                         'cache':{'path':p.relative_to(cache).as_posix(),'sha256':sha256(p),'stat':file_stat(p)}})
    m={'target':'modify_video','feature_kind':REFERENCE_KIND,'input_dim':1024,'test_only':False,
       'feature_config':{'kind':'SYNTHETIC_REFERENCE_TEST'},'videos':rows}
    m['manifest_sha256']=digest(m);json_write(cache/'manifest.json',m)
    return source,cache,m


def args_for(source,cache,out):
    return SimpleNamespace(data_root=source,reference_manifest=cache/'manifest.json',out=out,splits=['train','dev'],
                           limit_per_split=0,seed=42,flow_method='farneback',flow_weights=Path('unused.pth'),device='cpu',
                           encode_batch_size=2,cpu_threads=2,opencv_threads=1,ffmpeg='ffmpeg',ffprobe='ffprobe',timeout=60)


def test_full_extract_resume_and_preserved_rgb(tmp_path):
    source,cache,m=reference_fixture(tmp_path)
    out=tmp_path/'new';a=args_for(source,cache,out)
    run(a,encoder=ExplicitFakeEncoder())
    result=json.loads((out/'manifest.json').read_text())
    assert result['test_only'] is True and result['input_dim']==2560 and len(result['videos'])==4
    for old,new in zip(sorted(m['videos'],key=lambda r:r['file']),result['videos']):
        with np.load(cache/old['cache']['path']) as x,np.load(out/new['cache']['path']) as y:
            np.testing.assert_array_equal(x['rgb'],y['rgb']);np.testing.assert_array_equal(x['timestamp_s'],y['timestamp_s'])
            assert y['flow'].shape==(5,2048)
            np.testing.assert_allclose(y['flow_pair_dt_s'],.05)
    run(a,encoder=ExplicitFakeEncoder())
    assert json.loads((out/'summary.json').read_text())['resumed']==4
    # Even test-double extraction exercises real FFmpeg, real Farneback, and stack geometry.


def test_source_timestamp_mismatch_fails(tmp_path):
    source,cache,m=reference_fixture(tmp_path);row=m['videos'][0]
    x=read_reference_arrays(cache,row);x['timestamp_s'][1]+=.03
    with pytest.raises(ValueError,match='timestamps'):
        compute_video(source/row['file'],row,x,ExplicitFakeEncoder(),DenseFlow())


def test_reference_tampering_fails(tmp_path):
    source,cache,m=reference_fixture(tmp_path);m['target']='any_fake';json_write(cache/'manifest.json',m)
    with pytest.raises(ValueError,match='integrity'):read_reference(cache/'manifest.json')


def test_random_reference_rejected(tmp_path):
    source,cache,m=reference_fixture(tmp_path);m['test_only']=True;m['manifest_sha256']=digest({k:v for k,v in m.items() if k!='manifest_sha256'})
    json_write(cache/'manifest.json',m)
    with pytest.raises(ValueError,match='random'):read_reference(cache/'manifest.json')


def test_interruption_keeps_only_complete_records_then_resumes(tmp_path,monkeypatch):
    import extract_tsn_flow as ex
    source,cache,m=reference_fixture(tmp_path);a=args_for(source,cache,tmp_path/'out')
    real=ex.compute_video;calls=[0]
    def interrupt(*args,**kwargs):
        calls[0]+=1
        if calls[0]==2:raise KeyboardInterrupt
        return real(*args,**kwargs)
    monkeypatch.setattr(ex,'compute_video',interrupt)
    with pytest.raises(SystemExit,match='Interrupted'):ex.run(a,encoder=ExplicitFakeEncoder())
    assert not (a.out/'manifest.json').exists() and not (a.out/'.extract.lock').exists()
    assert json.loads((a.out/'summary.json').read_text())['completed_videos']==1
    monkeypatch.setattr(ex,'compute_video',real);ex.run(a,encoder=ExplicitFakeEncoder())
    assert json.loads((a.out/'summary.json').read_text())['resumed']==1


def numerical_manifest(root):
    rows=[]
    for split in ('train','dev'):
        for i in (0,1):
            file=f'{split}/{i}.mp4';p=root/'clips'/split/f'{i}.npz'
            ts=np.arange(5,dtype=np.float64)/10
            atomic_npz(p,{'rgb':np.full((5,512),i*.1,np.float32),'flow':np.full((5,2048),i*.1,np.float32),
                          'timestamp_s':ts,'source_frame_index':np.arange(5)})
            rows.append({'file':file,'split':split,'source_group':file,'modify_video':bool(i),'modify_audio':False,
                         'visual_periods':[[.1,.3]] if i else [],'duration_s':.5,'feature_count':5,'config_id':'UNIT_TEST',
                         'cache':{'path':p.relative_to(root).as_posix(),'sha256':sha256(p),'stat':file_stat(p)}})
    m={'videos':rows,'feature_kind':FEATURE_KIND,'input_dim':2560,'rgb_dim':512,'flow_dim':2048,'target':'modify_video',
       'test_only':False,'feature_config':{'kind':'NUMERICAL_UNIT_TEST'}}
    m['manifest_sha256']=digest(m);json_write(root/'manifest.json',m)
    return load_manifest(root/'manifest.json')


def test_loader2560_and_temporal_inverse(tmp_path):
    m=numerical_manifest(tmp_path);ds=VisualDataset(m,'train')
    assert ds[0]['feats'].shape==(2560,768) and ds[0]['segments'] is None
    x=ds[1];sec=x['segments']*x['feat_stride']+.5*x['feat_num_frames']
    torch.testing.assert_close(sec,torch.tensor([[.1,.3]]))
    assert VisualDataset(m,'train',input_mode='rgb')[0]['feats'].shape==(512,768)


def test_loader_changed_cache_fails(tmp_path):
    m=numerical_manifest(tmp_path);p=tmp_path/m['videos'][0]['cache']['path'];p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):VisualDataset(m,'train')[0]


def test_loader_rejects_old_kind(tmp_path):
    m=numerical_manifest(tmp_path);m.pop('_cache_root');m['feature_kind']=REFERENCE_KIND
    m['manifest_sha256']=digest({k:v for k,v in m.items() if k!='manifest_sha256'});json_write(tmp_path/'manifest.json',m)
    with pytest.raises(ValueError,match='extract_tsn_flow'):load_manifest(tmp_path/'manifest.json')


def test_official_factory_2560_no_padding(monkeypatch):
    import upstream_adapter as a
    def fake(repo):
        return {'model_name':a.EXPECTED_MODEL,'dataset':{},'model':{}},lambda name,**kw:torch.nn.Conv1d(kw['input_dim'],2,1),None,{'source_digest':'TEST'}
    monkeypatch.setattr(a,'load_official',fake)
    model,cfg,_,_,counts=a.build_official('unused',torch.device('cpu'))
    assert model.in_channels==2560 and cfg['dataset']['input_dim']==2560


def test_sampler_is_video_balanced_and_complete():
    rows=[{'modify_video':i>=7} for i in range(12)]
    b=list(MixedBatchSampler(rows,4,42));assert set(sum(b,[]))==set(range(12))
    assert all(sum(rows[i]['modify_video'] for i in x)==2 for x in b)


def test_random_tsn_cache_training_refused(tmp_path,monkeypatch):
    import run_experiment as r
    monkeypatch.setattr(r,'load_manifest',lambda p:{'test_only':True})
    with pytest.raises(ValueError,match='diagnostic'):r.train(SimpleNamespace(manifest='unused'))


def test_new_trainer_checkpoint_evaluate_resume_with_explicit_double(tmp_path,monkeypatch):
    import run_experiment as r
    numerical_manifest(tmp_path)
    class Toy(torch.nn.Module):
        def __init__(self):super().__init__();self.layer=torch.nn.Conv1d(2560,1,1);self.loss_normalizer=1.
        def forward(self,batch):
            if self.training:return {'final_loss':self.layer(torch.stack([b['feats'] for b in batch])).square().mean()}
            output=[]
            for b in batch:
                assert not {'segments','labels','av_labels'}&set(b)
                output.append({'video_id':b['video_id'],'scores':self.layer(b['feats'][None]).mean().sigmoid().reshape(1),
                               'segments':torch.tensor([[.1,.3]]),'labels':torch.zeros(1,dtype=torch.long)})
            return output
    class Utils:
        @staticmethod
        def make_optimizer(m,c):return torch.optim.AdamW(m.parameters(),lr=c.get('learning_rate',.001))
        @staticmethod
        def make_scheduler(o,c,n):return torch.optim.lr_scheduler.LambdaLR(o,lambda _:1)
    def build(repo,device,mode):
        m=Toy().to(device);return m,{'model_name':'EXPLICIT_TEST_DOUBLE','model':{'input_dim':2560},'opt':{}},Utils,{'source_digest':'TEST','git_commit':None},{'total_parameters':2561,'trainable_parameters':2561}
    monkeypatch.setattr(r,'build_official',build)
    a=SimpleNamespace(manifest=tmp_path/'manifest.json',device='cpu',amp='none',seed=42,upstream=tmp_path/'unused',
                      input_mode='rgb-flow',batch_size=2,workers=0,lr=.0003,weight_decay=.05,epochs=2,warmup_epochs=0,
                      ema=True,out=tmp_path/'run',resume=None,patience=8,max_consecutive_skips=4)
    r.train(a);c=r.checkpoint_load(a.out/'last.pt')
    assert c['format']=='ummaformer_tsn_flow_visual_v1' and len(c['history'])==2
    r.evaluate(SimpleNamespace(manifest=a.manifest,checkpoint=a.out/'best_video_auc.pt',device='cpu',
                               upstream=a.upstream,split='dev',out=tmp_path/'eval'))
    result=json.loads((tmp_path/'eval/metrics.json').read_text())
    assert result['videos']==2 and result['feature_kind']==FEATURE_KIND and result['evaluation_weights']=='ema'
    a.resume=a.out/'last.pt';r.train(a)
    assert len(r.checkpoint_load(a.out/'last.pt')['history'])==2


def test_old_checkpoint_is_not_resumed(tmp_path):
    import run_experiment as r
    p=tmp_path/'old.pt';torch.save({'format':'ummaformer_opencv_visual_v2'},p)
    with pytest.raises(ValueError,match='Not a checkpoint'):r.checkpoint_load(p)


def test_train_like_baseline_copies_settings_not_weights(tmp_path,monkeypatch):
    import train_like_baseline as helper
    from upstream_adapter import split_identity
    source,cache,reference=reference_fixture(tmp_path)
    candidate_root=tmp_path/'candidate';candidate=numerical_manifest(candidate_root)
    candidate.pop('_cache_root')
    originals={r['file']:r for r in reference['videos']}
    for row in candidate['videos']:row['reference_cache_sha256']=originals[row['file']]['cache']['sha256']
    candidate['manifest_sha256']=digest({k:v for k,v in candidate.items() if k!='manifest_sha256'})
    json_write(candidate_root/'manifest.json',candidate)
    settings=dict(epochs=60,warmup_epochs=5,batch_size=16,lr=.0003,weight_decay=.05,
                  patience=8,seed=42,amp='none',ema=True,max_consecutive_skips=4,input_mode='rgb-flow')
    cp=tmp_path/'baseline.pt';torch.save({'format':'ummaformer_opencv_visual_v2','settings':settings,
        'split_identity':split_identity(reference),'upstream':{'source_digest':'SAME'},
        'epoch':11,'dev_metrics':{'video_visual_roc_auc':.8284}},cp)
    monkeypatch.setattr(helper,'source_identity',lambda p:{'source_digest':'SAME'})
    a=SimpleNamespace(baseline_checkpoint=cp,reference_manifest=cache/'manifest.json',manifest=candidate_root/'manifest.json',
                      upstream=tmp_path/'upstream',out=tmp_path/'train',device='cpu',workers=0,batch_size=None,resume=None)
    cmd,r=helper.build_command(a)
    assert cmd[cmd.index('--epochs')+1]=='60' and cmd[cmd.index('--batch-size')+1]=='16'
    assert '--resume' not in cmd and r['baseline_weights_loaded_into_new_model'] is False
    a.batch_size=8;cmd,r=helper.build_command(a)
    assert r['batch_override_changes_protocol'] is True


def test_compare_reads_matched_single_model_scores(tmp_path):
    spec=importlib.util.spec_from_file_location('tsn_compare_test',ROOT/'tools/compare_readouts.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    from experiment_metrics import video_metrics,choose_threshold
    source,cache,reference=reference_fixture(tmp_path)
    candidate=dict(reference);candidate['feature_kind']=FEATURE_KIND;candidate['feature_config']={'dense_flow':{'method':'farneback'}}
    candidate['videos']=[{**r,'reference_cache_sha256':r['cache']['sha256']} for r in reference['videos']]
    json_write(tmp_path/'candidate.json',candidate)
    for name,values in [('base',[.2,.7]),('candidate',[.1,.9])]:
        folder=tmp_path/name;folder.mkdir(exist_ok=True)
        rows=[{'file':f'dev/{i}.mp4','split':'dev','source_group':f'dev/{i}.mp4','visual_label':i,'modify_audio':0,'score':v} for i,v in enumerate(values)]
        (folder/'video_scores.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        metrics=video_metrics(rows,choose_threshold([0,1],values))
        metrics.update(split='dev',all_cohort_visual_temporal_ap={'AP@0.50':.5,'AP@0.75':.4,'AP@0.95':.1})
        json_write(folder/'metrics.json',metrics)
    table,report=helper.compare(tmp_path/'base',tmp_path/'candidate',cache/'manifest.json',tmp_path/'candidate.json')
    assert report['videos']==2 and report['rgb_reference_cache_match']
    assert report['metrics']['video_visual_roc_auc']['tsn_minus_baseline']==0
    bad=json.loads((tmp_path/'candidate.json').read_text());bad['videos'][0]['source_group']='another'
    json_write(tmp_path/'candidate.json',bad)
    with pytest.raises(ValueError,match='Training cohorts'):helper.compare(tmp_path/'base',tmp_path/'candidate',cache/'manifest.json',tmp_path/'candidate.json')
