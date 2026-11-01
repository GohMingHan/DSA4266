import json
from pathlib import Path
import numpy as np
import pytest
import torch
from helpers import setup
from e2b.io import read_json,write_json,digest,sha256,write_npz
from e3.data import prepare,load_index,Sequences,collate
from e2b.train import normalizer
from e2b.prepare import load_split


def test_e2b_mean_compatibility_and_no_copy(tmp_path):
    conf,data,old=setup(tmp_path,verify_e2b=True)
    cfg,rows,norm=load_index(data,'train',True)
    oc,ors,V,A,y=load_split(old,'train',True)
    for m,x in [('visual',V),('audio',A)]:
        for key,val in normalizer(x).items():assert torch.equal(val,norm[m][key])
    assert cfg['splits']['train']['e2b_mean_signature']==oc['splits']['train']['signature']
    ds=Sequences(cfg,rows);entry=ds[0]
    assert np.array_equal(entry['visual_mean'].numpy(),V[0])
    assert not (data/'clips').exists()
    assert cfg['counts']['train']['labels']=={'0':8,'1':24}


def test_all_tokens_timestamps_preserved(tmp_path):
    _,data,_=setup(tmp_path)
    cfg,rows,_=load_index(data,'train',True);ds=Sequences(cfg,rows)
    r=rows[0]
    for m in ('visual','audio'):
        with np.load(Path(cfg['embedding_roots'][m])/r[m]['feature_path']) as raw:
            assert np.array_equal(ds[0][m].numpy(),raw['features'])
            key='timestamp_s' if m=='visual' else 'center_s'
            np.testing.assert_allclose(ds[0][m+'_time'].numpy(),raw[key],atol=1e-7)


def test_test_fixture_blocked(tmp_path):
    _,data,_=setup(tmp_path)
    with pytest.raises(ValueError,match='Synthetic'):load_index(data,'train')


def test_prepare_requires_train_dev(tmp_path):
    conf,_,_=setup(tmp_path)
    with pytest.raises(ValueError,match='train\+dev'):prepare(conf,tmp_path/'new',('test',),allow_test_only=True)


def test_missing_video_not_silently_removed(tmp_path):
    conf,data,_=setup(tmp_path)
    c=read_json(conf);e2b=read_json(c['experiment_2b_config']);root=Path(c['experiment_2b_config']).parent
    path=root/'audio'/'splits'/'train.json';s=read_json(path);s['videos'].pop()
    s['signature']=digest({k:v for k,v in s.items() if k!='signature'});write_json(path,s)
    with pytest.raises(ValueError,match='IDs differ'):prepare(conf,tmp_path/'missing',allow_test_only=True)
    assert not (tmp_path/'missing'/'dataset.json').exists()


def test_corrupt_embedding_blocked(tmp_path):
    _,data,_=setup(tmp_path);cfg,rows,_=load_index(data,'train',True)
    p=Path(cfg['embedding_roots']['visual'])/rows[0]['visual']['feature_path'];p.write_bytes(b'bad')
    with pytest.raises(ValueError,match='corrupted'):Sequences(cfg,rows)[0]


def test_original_files_unchanged(tmp_path):
    conf,data,_=setup(tmp_path)
    root=tmp_path/'source';before={str(p):sha256(p) for p in root.rglob('*') if p.is_file()}
    prepare(conf,tmp_path/'again',allow_test_only=True)
    after={str(p):sha256(p) for p in root.rglob('*') if p.is_file()};assert before==after


def test_normalization_no_dev_or_test_fit(tmp_path):
    conf,data,_=setup(tmp_path);cfg,_,norm=load_index(data,'train',True)
    for split in ('dev','test'):
        base=tmp_path/'source'/'audio';path=base/'splits'/f'{split}.json';s=read_json(path)
        for r in s['videos']:
            p=base/r['feature_path']
            with np.load(p) as z:arr={k:z[k].copy() for k in z.files}
            arr['features']+=1000;write_npz(p,**arr);r['sha256']=sha256(p)
        s['signature']=digest({k:v for k,v in s.items() if k!='signature'});write_json(path,s)
    prepare(conf,tmp_path/'new',('train','dev','test'),allow_test_only=True)
    _,_,other=load_index(tmp_path/'new','train',True)
    for m in norm:
        for k in norm[m]:assert torch.equal(norm[m][k],other[m][k])


def test_train_signatures_same_when_test_added(tmp_path):
    conf,data,_=setup(tmp_path,with_test=True)
    cfg=prepare(conf,tmp_path/'td',('train','dev'),allow_test_only=True);orig=read_json(data/'dataset.json')
    assert cfg['splits']['train']==orig['splits']['train']
    assert cfg['splits']['dev']==orig['splits']['dev']
    assert cfg['normalization']==orig['normalization']


def test_pooled_cache_not_sequence_cache(tmp_path):
    conf,data,old=setup(tmp_path,verify_e2b=True)
    e2bp=Path(read_json(conf)['experiment_2b_config']);c=read_json(e2bp);c['visual_embeddings']=str(old);write_json(e2bp,c)
    with pytest.raises(FileNotFoundError):prepare(conf,tmp_path/'fail',allow_test_only=True)


def test_wrong_label_detected(tmp_path):
    conf,data,_=setup(tmp_path)
    p=tmp_path/'source'/'audio'/'splits'/'train.json';s=read_json(p)
    s['videos'][0]['label']=1;s['signature']=digest({k:v for k,v in s.items() if k!='signature'});write_json(p,s)
    with pytest.raises(ValueError,match='Conflicting'):prepare(conf,tmp_path/'bad',allow_test_only=True)
