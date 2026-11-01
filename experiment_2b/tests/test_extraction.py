from pathlib import Path
import numpy as np
import pytest
from e2b.io import VERSION,write_json,write_jsonl,write_npz,sha256,read_json,digest
from e2b.extract import extract
from helpers import fixture


def setup_import(root,mod):
    config=fixture(root/'source')
    c=read_json(config);folder=config.parent
    external=folder/('external_'+mod);external.mkdir()
    weights=folder/'models'/mod/'best.pt';weights.parent.mkdir(parents=True);weights.write_bytes(b'test-identification-only')
    all_rows=[]
    for split in ('train','dev','test'):
        m=read_json(folder/mod/'splits'/f'{split}.json')
        for r in m['videos']:
            source=folder/mod/r['feature_path'];target=external/(Path(r['feature_path']).stem+'.npz')
            target.write_bytes(source.read_bytes())
            all_rows.append({'file':r['file'],'split':split,'feature_path':target.name,'checkpoint_sha256':sha256(weights)})
    write_jsonl(external/'index.jsonl',all_rows)
    write_json(folder/'models'/mod/'encoder.json',{'kind':'preexported','checkpoint':'best.pt','index':str(external/'index.jsonl'),
       'representation':'pre_classifier_embeddings','description':'synthetic preexported interoperability fixture',
       'preprocessing_id':'software_fixture_v1','preprocessing_verified':True,'time_origin':'first_decoded_video_frame'})
    return config


@pytest.mark.parametrize('mod',['audio','visual'])
def test_extraction_resume_and_later_test(tmp_path,mod):
    config=setup_import(tmp_path,mod);out=tmp_path/'export'
    first=extract(config,mod,['train','dev'],out,device='cpu')
    assert first['completed_videos']==48 and first['errors']==0
    second=extract(config,mod,['train','dev'],out,device='cpu')
    assert second['resumed']==48
    before=read_json(out/'splits/train.json')
    extract(config,mod,['test'],out,device='cpu')
    assert read_json(out/'splits/train.json')==before
    assert read_json(out/'splits/test.json')['partial_pilot'] is False


def test_import_malformed_features_not_published(tmp_path):
    config=setup_import(tmp_path,'visual');out=tmp_path/'export'
    folder=config.parent/'external_visual';first=next(folder.glob('*.npz'))
    # Wrong first feature may not be first sorted; all invalid ensures circuit breaker.
    for f in folder.glob('*.npz'):write_npz(f,features=np.ones((3,1)),timestamp_s=[0,.5,1.])
    with pytest.raises(RuntimeError,match='First three'):extract(config,'visual',['train'],out,device='cpu')
    assert not (out/'splits/train.json').exists()
    assert not (out/'.e2b.lock').exists()


def test_pilot_export_marked(tmp_path):
    config=setup_import(tmp_path,'audio');out=tmp_path/'export'
    r=extract(config,'audio',['train','dev'],out,limit=4,device='cpu')
    assert r['completed_videos']==8 and r['partial_pilot']
    assert read_json(out/'splits/dev.json')['partial_pilot']
