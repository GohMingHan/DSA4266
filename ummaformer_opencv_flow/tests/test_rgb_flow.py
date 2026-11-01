"""Local software tests; no claims of upstream-model, GPU or LAV-DF accuracy validation."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)
from common import digest, file_stat, json_write, sha256
from cohort import select_cohort, normalize
from video_io import sample_indices, probe_video, decoded_rgb
from extract_rgb_flow import (FEATURE_KIND, RGBEncoder, extract_one, flow_descriptor, validate_cache, atomic_npz)
from visual_data import (VisualDataset, load_manifest, resample_at_bin_centers, MixedBatchSampler)
from experiment_metrics import temporal_metrics, video_metrics, choose_threshold


def cache_row(root, file, positive, seconds=1.):
    import hashlib
    sid = hashlib.sha256(file.encode()).hexdigest()[:20]
    path = root / 'clips' / (sid + '.npz')
    ts = np.arange(10) / 10
    x = np.repeat(ts[:, None], 512, axis=1).astype(np.float32)
    atomic_npz(path, dict(rgb=x, flow=x, timestamp_s=ts, source_frame_index=np.arange(10)*2,
                         flow_valid=np.arange(10)>0, flow_dt_s=np.r_[0., np.full(9, .05)]))
    return dict(file=file, split=file.split('/')[0], modify_video=positive, modify_audio=False,
                source_group=file, visual_periods=[[.2,.6]] if positive else [],
                duration_s=seconds, feature_count=10, config_id='test',
                cache=dict(path=path.relative_to(root).as_posix(), sha256=sha256(path), stat=file_stat(path)))


def make_manifest(root):
    rows = [cache_row(root, f'{s}/{i}.mp4', bool(i)) for s in ('train','dev') for i in (0,1)]
    m = dict(videos=rows, target='modify_video', feature_kind=FEATURE_KIND, test_only=False)
    m['manifest_sha256'] = digest(m)
    json_write(root/'manifest.json',m)
    return load_manifest(root/'manifest.json')


def test_audio_only_is_visual_negative():
    r=normalize(dict(file='train/a.mp4',split='train',duration=1,modify_audio=True,modify_video=False,
                     fake_periods=[[.1,.5]]))
    assert r['visual_periods']==[] and r['fake_periods']==[[.1,.5]]


def test_cohort_jsonl_exact_join_and_splits(tmp_path):
    metadata=[dict(file=f'{s}/{i}.mp4',split=s,duration=1,modify_audio=False,modify_video=bool(i%2),
                   fake_periods=[[.2,.6]] if i%2 else []) for s in ('train','dev','test') for i in range(6)]
    json_write(tmp_path/'metadata.json',metadata)
    selected=[r for r in metadata if r['file'] in ('train/0.mp4','train/1.mp4','dev/2.mp4','dev/3.mp4','test/0.mp4')]
    (tmp_path/'videos.jsonl').write_text('\n'.join(json.dumps(r) for r in selected))
    result=select_cohort(tmp_path/'metadata.json',tmp_path/'videos.jsonl',['train','dev'])
    assert {r['file'] for r in result}=={'train/0.mp4','train/1.mp4','dev/2.mp4','dev/3.mp4'}
    selected[0]['modify_video']=True
    (tmp_path/'videos.jsonl').write_text('\n'.join(json.dumps(r) for r in selected))
    with pytest.raises(ValueError,match='disagreement'):
        select_cohort(tmp_path/'metadata.json',tmp_path/'videos.jsonl',['train','dev'])


def test_duplicate_cohort_fails(tmp_path):
    json_write(tmp_path/'meta.json',[])
    line=json.dumps({'file':'train/a.mp4'})
    (tmp_path/'cohort.jsonl').write_text(line+'\n'+line)
    with pytest.raises(ValueError,match='Duplicate'):
        select_cohort(tmp_path/'meta.json',tmp_path/'cohort.jsonl',['train'])


def test_missing_id_fails(tmp_path):
    json_write(tmp_path/'meta.json',[])
    (tmp_path/'cohort.jsonl').write_text(json.dumps({'file':'train/a.mp4'}))
    with pytest.raises(ValueError,match='absent'):
        select_cohort(tmp_path/'meta.json',tmp_path/'cohort.jsonl',['train'])


def test_sampling_native_indices():
    pts=np.arange(12)/30
    idx=sample_indices(pts,10)
    np.testing.assert_array_equal(idx,[0,3,6,9])
    np.testing.assert_array_equal(sample_indices(pts,0),np.arange(12))


def test_flow_static_and_known_translation():
    import cv2
    rng=np.random.default_rng(0)
    image=rng.integers(0,256,(224,224),dtype=np.uint8)
    still=flow_descriptor(image,image,.04)
    assert still.shape==(512,) and np.max(np.abs(still))<.003
    translated=cv2.warpAffine(image,np.float32([[1,0,2],[0,1,0]]),(224,224))
    descriptor=flow_descriptor(image,translated,.04).reshape(2,16,16)
    expected=2/(224*.04)
    assert np.median(descriptor[0,2:-2,2:-2])==pytest.approx(expected,rel=.1)
    assert np.median(np.abs(descriptor[1,2:-2,2:-2]))<.02


def test_flow_rejects_bad_dt():
    with pytest.raises(ValueError):
        flow_descriptor(np.zeros((224,224),np.uint8),np.zeros((224,224),np.uint8),0.)


def test_timestamp_interpolation_not_row_number():
    ts=np.array([0., .1, .7])
    f=np.repeat(ts[:,None],2,axis=1).astype(np.float32)
    out=resample_at_bin_centers(f,ts,1.,10)
    np.testing.assert_allclose(out[:,0],np.minimum((np.arange(10)+.5)/10,.7),atol=1e-6)


def test_loader_shape_and_inverse_time(tmp_path):
    m=make_manifest(tmp_path)
    ds=VisualDataset(m,'train')
    assert ds[0]['feats'].shape==(1024,768)
    assert ds[0]['segments'] is None
    p=ds[1]
    inverse=(p['segments']*p['feat_stride']+.5*p['feat_num_frames'])/p['fps']
    torch.testing.assert_close(inverse,torch.tensor([[.2,.6]]))
    assert VisualDataset(m,'train',input_mode='rgb')[1]['feats'].shape==(512,768)


def test_changed_cache_fails(tmp_path):
    m=make_manifest(tmp_path)
    p=tmp_path/m['videos'][0]['cache']['path']
    with p.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='changed'):
        VisualDataset(m,'train')[0]


def test_manifest_integrity(tmp_path):
    m=make_manifest(tmp_path)
    raw=json.loads((tmp_path/'manifest.json').read_text())
    raw['target']='any_fake'
    json_write(tmp_path/'manifest.json',raw)
    with pytest.raises(ValueError,match='integrity'):
        load_manifest(tmp_path/'manifest.json')


def test_cache_validates(tmp_path):
    r=cache_row(tmp_path,'train/1.mp4',True)
    assert validate_cache(tmp_path/r['cache']['path'])==10


def test_mixed_sampler_covers_both_classes():
    rows=[dict(modify_video=i>7) for i in range(12)]
    sampler=MixedBatchSampler(rows,4)
    batches=list(sampler)
    assert set(sum(batches,[]))==set(range(12))
    assert all(sum(rows[i]['modify_video'] for i in b)==2 and len(set(b))==4 for b in batches)


def test_negative_video_false_proposal_counts():
    rows=[dict(file='p',visual_periods=[[1,2]]),dict(file='n',visual_periods=[])]
    pred={'p':[dict(segment=[1,2],score=.8)],'n':[dict(segment=[1,2],score=.9)]}
    assert temporal_metrics(rows,pred)['all_cohort_visual_temporal_ap']['AP@0.50']==pytest.approx(.5)
    rows2=[dict(visual_label=0,score=.1),dict(visual_label=1,score=.9)]
    assert video_metrics(rows2,choose_threshold([0,1],[.1,.9]))['video_visual_roc_auc']==1.


def test_official_factory_dimensions_not_shape_padding(monkeypatch):
    import upstream_adapter as adapter
    def fake_load(repo):
        def factory(name,**kw):
            return torch.nn.Conv1d(kw['input_dim'],4,1)
        return ({'model_name':adapter.EXPECTED_MODEL,'dataset':{'input_dim':4096},
                 'model':{'input_dim':4096,'audio_input_dim':0}},factory,None,{'source_digest':'test'})
    monkeypatch.setattr(adapter,'load_official',fake_load)
    model,cfg,_,_,_=adapter.build_official('unused',torch.device('cpu'),'rgb-flow')
    assert model.in_channels==1024 and cfg['dataset']['input_dim']==1024
    model,*_=adapter.build_official('unused',torch.device('cpu'),'rgb')
    assert model.in_channels==512


def test_gradient_guard():
    from run_experiment import train_update,scaler_for
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__();self.layer=torch.nn.Linear(4,1)
        def forward(self,x):return {'final_loss':self.layer(x).square().mean()}
    model=M();opt=torch.optim.AdamW(model.parameters(),lr=.001)
    losses,info=train_update(model,torch.randn(2,4),opt,scaler_for('none'),torch.device('cpu'),'none')
    assert math.isfinite(losses['final_loss']) and not info['skipped']


def test_real_decode_and_random_resnet_extraction(tmp_path):
    import shutil
    if not shutil.which('ffmpeg'):pytest.skip('FFmpeg unavailable')
    path=tmp_path/'synthetic.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x64:rate=20:duration=1',
                    '-c:v','mpeg4','-threads','1','-y',str(path)],check=True,timeout=30)
    timing=probe_video(path)
    assert timing['decoded_frame_count']==20 and timing['duration_s']==pytest.approx(1.)
    enc=RGBEncoder(torch.device('cpu'),test_random=True)
    arrays,timing=extract_one(path,enc,sample_fps=10,encode_batch_size=4)
    assert arrays['rgb'].shape==(10,512) and arrays['flow'].shape==(10,512)
    np.testing.assert_array_equal(arrays['source_frame_index'],np.arange(0,20,2))
    # Each retained flow uses 0.05-second adjacent native frames, NOT 0.1-second sampled gaps.
    np.testing.assert_allclose(arrays['flow_dt_s'][1:],.05)
    assert not arrays['flow_valid'][0] and arrays['flow_valid'][1:].all()
    assert all(not p.requires_grad for p in enc.model.parameters())


def test_random_cache_training_refused(tmp_path,monkeypatch):
    import run_experiment as runner
    m=make_manifest(tmp_path)
    monkeypatch.setattr(runner,'load_manifest',lambda p: {**m,'test_only':True})
    with pytest.raises(ValueError,match='diagnostic'):
        runner.train(argparse.Namespace(manifest=tmp_path/'manifest.json'))


def test_train_checkpoint_evaluate_orchestration_with_explicit_test_double(tmp_path,monkeypatch):
    """Exercise runner IO, not the official architecture or its losses."""
    import run_experiment as runner
    from types import SimpleNamespace
    make_manifest(tmp_path)
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.layer=torch.nn.Conv1d(1024,1,1); self.loss_normalizer=1.
        def forward(self,batch):
            if self.training:
                x=torch.stack([b['feats'] for b in batch])
                y=self.layer(x).square().mean()
                return {'final_loss':y}
            result=[]
            for b in batch:
                assert all(k not in b for k in ('labels','segments','av_labels'))
                score=self.layer(b['feats'][None]).mean().sigmoid().reshape(1)
                result.append({'video_id':b['video_id'],'scores':score,
                               'segments':torch.tensor([[.2,.6]]),'labels':torch.zeros(1,dtype=torch.long)})
            return result
    class Utilities:
        @staticmethod
        def make_optimizer(model,cfg):return torch.optim.AdamW(model.parameters(),lr=cfg.get('learning_rate',.001))
        @staticmethod
        def make_scheduler(opt,cfg,n):return torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    def factory(repo,device,mode):
        model=Toy().to(device)
        cfg={'model_name':'EXPLICIT_TEST_DOUBLE_NOT_UMMAFORMER','model':{'input_dim':1024},'opt':{}}
        ident={'source_digest':'test_double','git_commit':None}
        counts={'total_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':1025}
        return model,cfg,Utilities,ident,counts
    monkeypatch.setattr(runner,'build_official',factory)
    a=SimpleNamespace(manifest=tmp_path/'manifest.json', device='cpu',amp='none',seed=42,
                      upstream=tmp_path/'unused',input_mode='rgb-flow',batch_size=2,workers=0,
                      lr=.001,weight_decay=.05,epochs=2,warmup_epochs=0,ema=True,
                      out=tmp_path/'run',resume=None,patience=8,max_consecutive_skips=4)
    runner.train(a)
    assert (a.out/'best_video_auc.pt').is_file()
    state=runner.checkpoint_load(a.out/'last.pt')
    assert state['settings']['input_mode']=='rgb-flow'
    assert len(state['history'])==2
    runner.evaluate(SimpleNamespace(checkpoint=a.out/'best_video_auc.pt',device='cpu',
                    manifest=a.manifest,upstream=a.upstream,split='dev',out=tmp_path/'eval'))
    metrics=json.loads((tmp_path/'eval'/'metrics.json').read_text())
    assert metrics['videos']==2 and metrics['threshold_source']=='checkpoint_development_set'
    a.resume=a.out/'last.pt'
    runner.train(a)  # no extra epochs; verify unchanged run resumes cleanly
