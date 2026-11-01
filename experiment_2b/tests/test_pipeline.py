import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from e2b.io import (canonical_cohort,check_sequence,load_slot,relative_name,write_json,read_json,sha256,digest)
from e2b.prepare import prepare,pool,load_split
from e2b.metrics import f1_threshold,binary_metrics
from e2b.train import ConcatClassifier,train,evaluate,normalizer,normalize
from e2b.compare import compare
from helpers import fixture,paired,train_args,eval_args


def test_or_target_and_family(tmp_path):
    cfg=fixture(tmp_path)
    rows=canonical_cohort(tmp_path/'cohort.json')
    assert len(rows)==64
    assert sum(r['label'] for r in rows)==48
    assert len({r['source_family'] for r in rows})==16
    assert {r['category'] for r in rows}=={'real','audio_only','visual_only','both'}


def test_cross_split_family_rejected(tmp_path):
    fixture(tmp_path);rows=read_json(tmp_path/'cohort.json');rows[-1]['original']='train/g0_0.mp4'
    write_json(tmp_path/'bad.json',rows)
    with pytest.raises(ValueError,match='crosses splits'):canonical_cohort(tmp_path/'bad.json')


@pytest.mark.parametrize('name',['/absolute.mp4','../file.mp4','C:/file.mp4','a/../../b',''])
def test_unsafe_id(name):
    with pytest.raises(ValueError):relative_name(name)


def test_unconfigured_slot_never_falls_back(tmp_path):
    write_json(tmp_path/'encoder.json',{'kind':'unconfigured'})
    with pytest.raises(ValueError,match='No random encoder'):load_slot(tmp_path/'encoder.json')


@pytest.mark.parametrize('d',[1,2])
def test_probabilities_not_embeddings(d):
    with pytest.raises(ValueError,match='D>=3'):check_sequence({'features':np.ones((3,d)),'timestamp_s':np.arange(3)},'visual')


def test_invalid_timeline():
    with pytest.raises(ValueError,match='strictly increasing'):
        check_sequence({'features':np.ones((3,5)),'timestamp_s':np.array([0,.1,.1])},'visual')


def test_bad_audio_window():
    with pytest.raises(ValueError):check_sequence({'features':np.ones((2,5)),'start_s':[0,1],'end_s':[1,.9],'center_s':[.5,1.2]},'audio')


def test_normalizer_train_only():
    train_x=np.array([[1.,2],[3,2]],np.float32);val_x=np.array([[101,25]],np.float32)
    s=normalizer(train_x)
    assert np.allclose(s['mean'].numpy(),[2,2]);assert np.allclose(s['std'].numpy(),[1,1])
    assert np.allclose(normalize(val_x,s).numpy(),[[99,23]])


def test_pooling_is_independent():
    x=np.array([[1,2,3],[3,6,1]],np.float32)
    assert np.array_equal(pool(x,'mean'),[2,4,2])
    assert np.array_equal(pool(x,'mean_max'),[2,4,2,3,6,3])


def test_feature_hash_validation(tmp_path):
    data=paired(tmp_path)
    with (data/'train.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='changed'):load_split(data,'train',True)


def test_no_silent_intersection(tmp_path):
    c=fixture(tmp_path);p=tmp_path/'audio/splits/dev.json';m=read_json(p);m['videos'].pop();m['signature']=digest({k:v for k,v in m.items() if k!='signature'});write_json(p,m)
    with pytest.raises(ValueError,match='exactly match'):prepare(c,tmp_path/'pair',allow_test_only=True)
    assert (tmp_path/'pair/coverage_error.json').is_file()
    assert not (tmp_path/'pair/dataset.json').exists()


def test_pilot_blocked(tmp_path):
    c=fixture(tmp_path);p=tmp_path/'visual/splits/train.json';m=read_json(p);m['partial_pilot']=True;m['signature']=digest({k:v for k,v in m.items() if k!='signature'});write_json(p,m)
    with pytest.raises(ValueError,match='Pilot'):prepare(c,tmp_path/'pair',allow_test_only=True)


def test_synthetic_cannot_be_research(tmp_path):
    c=fixture(tmp_path)
    with pytest.raises(ValueError,match='test embeddings'):prepare(c,tmp_path/'pair')


def test_prepare_preserves_train_when_test_added(tmp_path):
    c=fixture(tmp_path/'input');a=prepare(c,tmp_path/'train_dev',allow_test_only=True)
    b=prepare(c,tmp_path/'all',('train','dev','test'),allow_test_only=True)
    assert a['splits']['train']['signature']==b['splits']['train']['signature']
    assert a['splits']['dev']['signature']==b['splits']['dev']['signature']


@pytest.mark.parametrize('mode',['visual','audio','concat'])
def test_modes_end_to_end(tmp_path,mode):
    data=paired(tmp_path)
    args=train_args(data,tmp_path/'run',mode=mode)
    train(args)
    c=torch.load(tmp_path/'run/best.pt',weights_only=True,map_location='cpu')
    report=evaluate(eval_args(data,tmp_path/'run/best.pt',tmp_path/'eval'))
    assert report['target']=='modify_audio OR modify_video'
    assert report['videos']==16 and report['positives']==12
    assert report['threshold_logit']==c['threshold_logit']
    assert report['threshold_source']=='checkpoint_DEV_only'
    assert set(report['by_category'])=={'real','audio_only','visual_only','both'}
    expected=torch.tensor(load_split(data,'train',True)[2].astype(np.float64).mean(0),dtype=torch.float32)
    assert torch.equal(c['normalization']['visual']['mean'],expected)


def test_threshold_does_not_change_auc():
    y=[0,0,1,1];z=[.1,.2,.3,.4]
    assert binary_metrics(y,z,.15)['video_any_roc_auc']==binary_metrics(y,z,.35)['video_any_roc_auc']==1
    assert f1_threshold(y,z)==.3


def test_classifier_batch_one_and_ignored_stream():
    v=torch.randn(1,7);a=torch.randn(1,5)
    for mode in ('concat','visual','audio'):
        m=ConcatClassifier(7,5,mode,8,0).eval()
        assert m(v,a).shape==(1,)
        if mode=='visual':assert torch.equal(m(v,a),m(v,a*100))
        if mode=='audio':assert torch.equal(m(v,a),m(v*100,a))


def test_resume_exact_completed_epoch(tmp_path,monkeypatch):
    import e2b.train as module
    data=paired(tmp_path)
    train(train_args(data,tmp_path/'continuous'))
    original=module.predict;calls=[0]
    def interrupted(*a,**k):
        calls[0]+=1
        if calls[0]==3:raise KeyboardInterrupt()
        return original(*a,**k)
    monkeypatch.setattr(module,'predict',interrupted)
    with pytest.raises(KeyboardInterrupt):train(train_args(data,tmp_path/'resumed'))
    assert torch.load(tmp_path/'resumed/last.pt',weights_only=True)['epoch']==2
    monkeypatch.setattr(module,'predict',original)
    train(train_args(data,tmp_path/'resumed',resume=str(tmp_path/'resumed/last.pt')))
    x=torch.load(tmp_path/'continuous/last.pt',weights_only=True);y=torch.load(tmp_path/'resumed/last.pt',weights_only=True)
    for key in x['model']:assert torch.equal(x['model'][key],y['model'][key])


def test_early_stopping(tmp_path):
    data=paired(tmp_path);train(train_args(data,tmp_path/'run',epochs=20,patience=2,lr=1e-30,dropout=0.))
    h=read_json(tmp_path/'run/history.json');assert len(h)==3


def test_compare_matched_controls(tmp_path):
    data=paired(tmp_path)
    for mode in ('visual','concat'):
        train(train_args(data,tmp_path/mode,mode=mode,epochs=2))
        evaluate(eval_args(data,tmp_path/mode/'best.pt',tmp_path/('eval_'+mode)))
    s=compare([tmp_path/'eval_visual',tmp_path/'eval_concat'],tmp_path/'comparison',20)
    assert s['bootstrap']['valid_replicates']>0


def test_exported_score_columns_not_input(tmp_path):
    data=paired(tmp_path)
    cfg,rows,V,A,y=load_split(data,'train',True)
    assert V.shape==(32,7) and A.shape==(32,5)
    assert not any('score' in k or 'probability' in k for k in rows[0])
