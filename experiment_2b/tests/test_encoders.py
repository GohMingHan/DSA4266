import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from e2b.encoders import PyramidCapture,Preexported,check_meta,freeze
from e2b.io import write_json,write_jsonl,write_npz,sha256,check_sequence


class PointMaker(torch.nn.Module):
    def forward(self,fs):
        return [torch.stack([torch.arange(f.shape[-1]).float(),torch.zeros(f.shape[-1]),torch.ones(f.shape[-1]),torch.ones(f.shape[-1])],1) for f in fs]
class Head(torch.nn.Module):
    def forward(self,fs,ms):return [f[:,:1]*m for f,m in zip(fs,ms)]
class TinyTemporal(torch.nn.Module):
    """Explicit test double, NOT UMMAFormer or a performance benchmark."""
    def __init__(self):
        super().__init__();self.neck=torch.nn.Conv1d(3,7,1);self.point_generator=PointMaker();self.cls_head=Head()
    def forward(self,items):
        x=items[0]['feats'][None];f=self.neck(x);m=torch.ones((1,1,x.shape[-1]),dtype=torch.bool);m[:,:,-1]=False
        pts=self.point_generator((f,));self.cls_head((f,),(m,))
        return [{'video_id':items[0]['video_id']}]


def test_umma_capture_features_not_scores():
    m=freeze(TinyTemporal());item={'video_id':'dev/a.mp4','feats':torch.randn(3,5),
                                 'duration':2.,'fps':1.,'feat_stride':.4,'feat_num_frames':.4}
    before={k:v.clone() for k,v in m.state_dict().items()}
    with torch.no_grad(),PyramidCapture(m) as cap:
        m([item]);out=cap.arrays(item)
    assert out['features'].shape==(4,7)
    assert np.allclose(out['timestamp_s'],[.2,.6,1,1.4])
    assert not m.cls_head._forward_pre_hooks and not m.point_generator._forward_hooks
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())


def test_capture_cleanup_on_error():
    m=freeze(TinyTemporal())
    with pytest.raises(RuntimeError):
        with PyramidCapture(m):raise RuntimeError('test')
    assert not m.cls_head._forward_pre_hooks


def test_capture_refuses_training():
    m=TinyTemporal()
    with pytest.raises(ValueError,match='eval'):
        with PyramidCapture(m):m([{'video_id':'a','feats':torch.randn(3,5)}])


def test_meta_disagreement():
    info={'file':'train/a.mp4','split':'train','modify_video':0}
    with pytest.raises(ValueError):check_meta(info,{'file':'train/a.mp4','split':'train','modify_video':1,'modify_audio':0,'label':1})


def test_import_preexported_and_hash(tmp_path):
    (tmp_path/'weights.pt').write_bytes(b'software-test-identity-only')
    write_npz(tmp_path/'a.npz',features=np.ones((3,7),np.float32),timestamp_s=np.array([0,.5,1.]))
    ck=sha256(tmp_path/'weights.pt')
    write_jsonl(tmp_path/'index.jsonl',[{'file':'train/a.mp4','split':'train','feature_path':'a.npz','checkpoint_sha256':ck}])
    slot={'checkpoint':str(tmp_path/'weights.pt'),'index':str(tmp_path/'index.jsonl'),
          'representation':'pre_classifier_embeddings','preprocessing_verified':True,'description':'fixture','preprocessing_id':'fixture-v1','time_origin':'first_decoded_video_frame'}
    engine=Preexported(slot,'cpu');row={'file':'train/a.mp4','split':'train','label':0,'modify_audio':0,'modify_video':0}
    assert check_sequence(engine.encode(row),'visual')['features'].shape==(3,7)
    original=engine.input_signature(row)
    write_npz(tmp_path/'a.npz',features=np.zeros((3,7),np.float32),timestamp_s=np.array([0,.5,1.]))
    assert engine.input_signature(row)!=original
    slot['preprocessing_verified']=False
    with pytest.raises(ValueError,match='preprocessing'):Preexported(slot,'cpu')
