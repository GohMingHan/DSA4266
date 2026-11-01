from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from helpers import setup,args,evaluate_args
from e3.train import train,evaluate,smoke,settings_of,make_loader
from e3.data import load_index,Sequences
from e3.compare import compare
from e3.common import read_json,write_json

@pytest.mark.parametrize('mode',['concat','self','cross'])
def test_end_to_end(mode,tmp_path):
    _,data,_=setup(tmp_path)
    run=tmp_path/mode;h=train(args(data,run,mode=mode),allow_test_only=True)
    assert len(h)==3
    report=evaluate(evaluate_args(data,run/'best.pt',tmp_path/'eval',split='test'),allow_test_only=True)
    assert report['videos']==16 and report['positives']==12 and report['threshold_source']=='checkpoint_DEV_only'
    assert (run/'normalization_train.json').exists()


def test_resume_matches_uninterrupted(tmp_path):
    _,data,_=setup(tmp_path)
    train(args(data,tmp_path/'full'),allow_test_only=True)
    train(args(data,tmp_path/'resume'),allow_test_only=True,stop_after_epoch=1)
    train(args(data,tmp_path/'resume',resume=str(tmp_path/'resume'/'last.pt')),allow_test_only=True)
    a=torch.load(tmp_path/'full'/'last.pt',weights_only=True);b=torch.load(tmp_path/'resume'/'last.pt',weights_only=True)
    for key in a['model']:assert torch.equal(a['model'][key],b['model'][key]),key
    assert a['epoch']==b['epoch']==3


def test_resume_changed_settings_rejected(tmp_path):
    _,data,_=setup(tmp_path);run=tmp_path/'run'
    train(args(data,run),allow_test_only=True,stop_after_epoch=1)
    with pytest.raises(ValueError,match='same settings'):
        train(args(data,run,resume=str(run/'last.pt'),lr=.003),allow_test_only=True)


def test_existing_run_protected(tmp_path):
    _,data,_=setup(tmp_path);run=tmp_path/'run';train(args(data,run),allow_test_only=True,stop_after_epoch=1)
    with pytest.raises(ValueError,match='not empty'):train(args(data,run),allow_test_only=True)


def test_real_smoke_api_no_checkpoint(tmp_path):
    _,data,_=setup(tmp_path);report=smoke(args(data,tmp_path/'unused'),allow_test_only=True)
    assert report['cross_self_parameter_match'] and not report['checkpoint_saved']
    assert not (tmp_path/'unused').exists()


def test_class_sampler_sequence_order_same_across_models(tmp_path):
    _,data,_=setup(tmp_path);cfg,rows,_=load_index(data,'train',True);ds=Sequences(cfg,rows)
    a=make_loader(ds,8,0,42,True);torch.randn(1000);b=make_loader(ds,8,0,42,True)
    ai=[v for x in a for v in x['index'].tolist()];bi=[v for x in b for v in x['index'].tolist()]
    assert ai==bi and len(ai)==len(ds)


def test_matched_compare_and_bootstrap(tmp_path):
    _,data,_=setup(tmp_path)
    for m in ('self','cross'):
        train(args(data,tmp_path/m,mode=m),allow_test_only=True)
        evaluate(evaluate_args(data,tmp_path/m/'best.pt',tmp_path/(m+'_eval')),allow_test_only=True)
    result=compare([tmp_path/'self_eval',tmp_path/'cross_eval'],tmp_path/'comparison',30)
    assert not result['training_setting_differences'] and result['bootstrap']['valid_replicates']==30
    assert (tmp_path/'comparison'/'comparison.csv').exists()


def test_external_e2b_comparison(tmp_path):
    from e2b.train import train as old_train,evaluate as old_eval
    from e2b_fixtures import train_args as oldargs,eval_args as oldeval
    _,data,old=setup(tmp_path,verify_e2b=True)
    old_train(oldargs(old,tmp_path/'oldrun',epochs=3,hidden=12,lr=.001))
    old_eval(oldeval(old,tmp_path/'oldrun'/'best.pt',tmp_path/'oldeval',split='dev'))
    train(args(data,tmp_path/'newrun'),allow_test_only=True)
    evaluate(evaluate_args(data,tmp_path/'newrun'/'best.pt',tmp_path/'neweval'),allow_test_only=True)
    with pytest.raises(ValueError,match='Nonmatched'):compare([tmp_path/'oldeval',tmp_path/'neweval'],tmp_path/'bad')
    report=compare([tmp_path/'oldeval',tmp_path/'neweval'],tmp_path/'compare',10,allow_training_differences=True)
    assert report['bootstrap']['valid_replicates']==10


def test_early_stopping_real_counter(tmp_path,monkeypatch):
    import e3.train as t
    _,data,_=setup(tmp_path)
    monkeypatch.setattr(t,'predict',lambda model,ds,*a,**k:np.zeros(len(ds)))
    h=train(args(data,tmp_path/'run',epochs=8,patience=2),allow_test_only=True)
    assert len(h)==3 and h[-1]['stale_evaluations']==2


def test_checkpoint_rejects_changed_encoder(tmp_path):
    from e3.train import compare_data_checkpoint
    _,data,_=setup(tmp_path);run=tmp_path/'run';train(args(data,run,epochs=1),allow_test_only=True)
    c=torch.load(run/'best.pt',weights_only=True);cfg=read_json(data/'dataset.json');cfg['encoder_ids']['audio']='other'
    with pytest.raises(ValueError,match='encoder_ids'):compare_data_checkpoint(c,cfg)
