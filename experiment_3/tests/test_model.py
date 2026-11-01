import copy
import numpy as np
import pytest
import torch
from e3.model import FusionModel, ContinuousTimeEncoding
from e3.data import collate
from e3.common import backend_context


def sample(length_v=7,length_a=3):
    return {'index':0,'label':1,'visual':torch.randn(length_v,7),'audio':torch.randn(length_a,5),
            'visual_time':torch.linspace(0,3,length_v),'audio_time':torch.linspace(.4,2.9,length_a),
            'visual_mean':torch.randn(7),'audio_mean':torch.randn(5)}

def norm():return {m:{'mean':torch.zeros(d),'std':torch.ones(d)} for m,d in [('visual',7),('audio',5)]}

def model(mode='cross',**kw):
    return FusionModel({'visual':7,'audio':5},norm(),mode=mode,width=16,heads=2,layers=1,head_width=12,dropout=0.,**kw)

@pytest.mark.parametrize('mode',['cross','self','concat'])
def test_forward_backward(mode):
    torch.manual_seed(42);m=model(mode);b=collate([sample(),sample(4,5)])
    with backend_context('math'):
        z=m(b);assert z.shape==(2,);z.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)

@pytest.mark.parametrize('mode',['cross','self'])
def test_padding_invariance(mode):
    torch.manual_seed(7);m=model(mode).eval();b=collate([sample(),sample(3,1)])
    with torch.no_grad(),backend_context('math'):
        p=m(b);c=copy.deepcopy(b)
        for mod in ('visual','audio'):
            c[mod].masked_fill_(c[mod+'_padding'].unsqueeze(-1),float('nan'))
            c[mod+'_time'].masked_fill_(c[mod+'_padding'],float('nan'))
        torch.testing.assert_close(p,m(c),atol=1e-6,rtol=1e-6)

@pytest.mark.parametrize('mode',['cross','self'])
def test_extra_batch_padding_same_result(mode):
    torch.manual_seed(20);m=model(mode).eval();one=sample(4,2)
    with torch.no_grad(),backend_context('math'):
        a=m(collate([one]));b=m(collate([one,sample(17,9)]))[:1]
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)


def test_parameter_and_initial_state_match():
    torch.manual_seed(42);a=model('cross');torch.manual_seed(42);b=model('self')
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    assert a.state_dict().keys()==b.state_dict().keys()
    for key,val in a.state_dict().items():assert torch.equal(val,b.state_dict()[key])

@pytest.mark.parametrize('mode',['cross','self'])
def test_empty_modality_rejected(mode):
    b=collate([sample()]);b['audio_padding'][:]=True
    with pytest.raises(ValueError,match='no valid tokens'):model(mode)(b)


def test_time_encoding_uses_seconds_not_position():
    pe=ContinuousTimeEncoding(16);mask=torch.zeros(1,3,dtype=torch.bool)
    x=pe(torch.tensor([[.0,1.,3.]]),mask);y=pe(torch.tensor([[.0,2.,3.]]),mask)
    assert torch.equal(x[:,0],y[:,0]) and torch.equal(x[:,2],y[:,2])
    assert not torch.allclose(x[:,1],y[:,1])


def test_self_streams_independent_until_readout():
    torch.manual_seed(4);b=collate([sample(),sample(4,5)])
    for mode in ('self','cross'):
        m=model(mode).eval();observed=[]
        handle=m.visual_blocks[0].register_forward_hook(lambda module,inputs,output:observed.append(output.detach().clone()))
        with torch.no_grad(),backend_context('math'):
            m(b);c=copy.deepcopy(b);c['audio']=torch.randn_like(c['audio'])*5;m(c)
        handle.remove()
        assert torch.allclose(observed[0],observed[1]) == (mode=='self')


def test_labels_not_model_inputs():
    torch.manual_seed(3);m=model().eval();b=collate([sample(),sample(3,2)])
    with torch.no_grad(),backend_context('math'):
        p=m(b);b['label']=1-b['label'];b['index']+=123;torch.testing.assert_close(p,m(b))


def test_concat_exact_original_head():
    from e2b.train import ConcatClassifier
    torch.manual_seed(42);m=model('concat');torch.manual_seed(42);original=ConcatClassifier(7,5,'concat',12,0.)
    b=collate([sample(),sample(3,2)])
    torch.testing.assert_close(m(b),original(b['visual_mean'],b['audio_mean']),rtol=0,atol=0)

@pytest.mark.parametrize('width,heads',[(15,3),(16,3),(2,1)])
def test_bad_width(width,heads):
    with pytest.raises(ValueError):FusionModel({'visual':7,'audio':5},norm(),width=width,heads=heads)
