#!/usr/bin/env python3
"""Check extraction dependencies and optionally one real pretrained TSN forward."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import cv2
import torch
import torchvision
from tsn_encoder import TSNFlowEncoder, load_checkpoint
from flow_frontend import DenseFlow


def parity_with_mmaction(weights, encoder):
    """Optional independent backbone calculation using an ALREADY installed MMAction2 0.x.

This never installs/downgrades anything. An absent/incompatible installation is a
clear failure, not a claim of verified parity. Run in a separate environment.
"""
    from mmaction.models import build_model
    state,_=load_checkpoint(weights)
    reference=build_model(dict(type='Recognizer2D',
        backbone=dict(type='ResNet',depth=50,in_channels=10,norm_eval=False),
        cls_head=None,test_cfg=dict(average_clips=None,feature_extraction=True)))
    if set(reference.state_dict()) - set(state):
        raise ValueError('Official MMAction model has missing checkpoint keys.')
    reference.load_state_dict({k:state[k] for k in reference.state_dict()},strict=True)
    reference.eval().to(encoder.device)
    torch.manual_seed(71)
    x=torch.rand(2,10,256,256,device=encoder.device)*2-1
    with torch.inference_mode():
        a=encoder.model(x)
        b=reference.backbone(x)
        if isinstance(b,tuple):
            if len(b)!=1:raise ValueError('Unexpected MMAction output stages.')
            b=b[0]
        b=b.mean(dim=(-2,-1))
    torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)
    return {'status':'passed','max_absolute_difference':float((a-b).abs().max()),'samples':2}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flow-weights',type=Path)
    p.add_argument('--flow-method',choices=['farneback','tvl1'],default='farneback')
    p.add_argument('--device',default='cpu')
    p.add_argument('--parity-mmaction',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(4)
    info={'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
          'numpy':np.__version__,'opencv':cv2.__version__,'cuda_available':torch.cuda.is_available(),
          'ffmpeg':shutil.which('ffmpeg'),'ffprobe':shutil.which('ffprobe')}
    if not info['ffmpeg'] or not info['ffprobe']:raise ValueError('FFmpeg AND ffprobe must be installed.')
    info['flow_estimator']=DenseFlow(a.flow_method).identity
    if a.flow_weights:
        encoder=TSNFlowEncoder(a.flow_weights,a.device)
        features=encoder.encode(np.zeros((1,10,256,256),np.float32))
        info.update(real_pretrained_forward='passed',shape=list(features.shape),weights=encoder.identity)
        if a.parity_mmaction:info['mmaction_parity']=parity_with_mmaction(a.flow_weights,encoder)
    elif a.parity_mmaction:raise ValueError('--parity-mmaction requires --flow-weights.')
    print(json.dumps(info,indent=2))

if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError,ImportError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(2)
