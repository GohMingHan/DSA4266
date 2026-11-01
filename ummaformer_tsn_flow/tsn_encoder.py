"""Frozen TSN-R50 flow backbone, loaded from the checkpoint linked by UMMAFormer.

This is a strict MMAction2 -> torchvision ResNet-50 backbone mapping, not an
ImageNet RGB model with invented flow weights. No MMAction/MMCV runtime is needed.
The removed TSN action head/segment consensus is not used for local embeddings.
"""
from __future__ import annotations
from pathlib import Path
import re
import torch
from torch import nn
from common import sha256

CHECKPOINT_NAME = 'tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804-8622cf38.pth'
CHECKPOINT_URL = ('https://download.openmmlab.com/mmaction/recognition/tsn/'
                  'tsn_r50_320p_1x1x8_150e_activitynet_clip_flow/' + CHECKPOINT_NAME)
CHECKPOINT_PREFIX = '8622cf38'
FEATURE_DIM = 2048
INPUT_CHANNELS = 10  # five (dx,dy) pairs, interleaved in chronological order


def make_backbone() -> nn.Module:
    from torchvision.models import resnet50
    model = resnet50(weights=None)  # All backbone tensors MUST be replaced below.
    model.conv1 = nn.Conv2d(10, 64, 7, stride=2, padding=3, bias=False)
    model.fc = nn.Identity()
    return model


def map_key(key: str) -> str:
    """Translate the MMAction ConvModule keys into torchvision names."""
    if key == 'conv1.conv.weight':
        return 'conv1.weight'
    if key.startswith('conv1.bn.'):
        return 'bn1.' + key[len('conv1.bn.'):]
    m = re.fullmatch(r'(layer[1-4]\.\d+)\.conv([123])\.conv\.(weight)', key)
    if m:
        return f'{m[1]}.conv{m[2]}.{m[3]}'
    m = re.fullmatch(r'(layer[1-4]\.\d+)\.conv([123])\.bn\.(.+)', key)
    if m:
        return f'{m[1]}.bn{m[2]}.{m[3]}'
    m = re.fullmatch(r'(layer[1-4]\.\d+)\.downsample\.(conv|bn)\.(.+)', key)
    if m:
        return f'{m[1]}.downsample.{0 if m[2] == "conv" else 1}.{m[3]}'
    raise ValueError(f'Unrecognized official backbone key: {key!r}. No partial load.')


def convert_backbone_state(state: dict, model: nn.Module) -> tuple[dict, list[str]]:
    """Require every learned tensor and running statistic. BN counters alone may be absent."""
    if not isinstance(state, dict):
        raise ValueError('Checkpoint state_dict must be a dictionary.')
    converted, excluded = {}, []
    for raw_key, value in state.items():
        key = raw_key[7:] if raw_key.startswith('module.') else raw_key
        if key.startswith('cls_head.'):
            excluded.append(raw_key)
            continue
        if not key.startswith('backbone.'):
            raise ValueError(f'Unexpected checkpoint tensor outside backbone/action head: {raw_key}')
        name = map_key(key[len('backbone.'):])
        if name in converted or not isinstance(value, torch.Tensor):
            raise ValueError(f'Duplicate/non-tensor backbone value: {raw_key}')
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite pretrained tensor: {raw_key}')
        converted[name] = value.detach().cpu()
    expected = model.state_dict()
    extras = sorted(set(converted) - set(expected))
    missing = sorted(set(expected) - set(converted))
    substantive = [k for k in missing if not k.endswith('num_batches_tracked')]
    if extras or substantive:
        raise ValueError(f'Incomplete/incompatible TSN backbone. Missing={substantive[:10]}, extra={extras[:10]}')
    for k in missing:
        converted[k] = torch.zeros_like(expected[k])  # Evaluation-only BN integer counter.
    for k, reference in expected.items():
        if converted[k].shape != reference.shape:
            raise ValueError(f'Shape mismatch {k}: {tuple(converted[k].shape)} != {tuple(reference.shape)}')
    if converted['conv1.weight'].shape != (64, 10, 7, 7):
        raise ValueError('This is not the ten-channel TSN flow checkpoint.')
    return converted, excluded


def load_checkpoint(path: Path) -> tuple[dict, str]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'TSN weights missing: {path}. Run fetch_tsn_flow.py first.')
    checksum = sha256(path)
    if not checksum.startswith(CHECKPOINT_PREFIX):
        raise ValueError(f'TSN checkpoint SHA256 {checksum} does not match the official filename prefix '
                         f'{CHECKPOINT_PREFIX}. Do not substitute RGB/random weights.')
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise RuntimeError('Unable to load the official checkpoint safely. Do not disable weights_only '
                           'or bypass the hash check; retain the original error for diagnosis.') from exc
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise ValueError('Expected the official MMAction checkpoint containing state_dict.')
    return checkpoint['state_dict'], checksum


class TSNFlowEncoder:
    def __init__(self, weights: Path, device: str | torch.device = 'cpu'):
        import torchvision
        self.device = torch.device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable. No silent CPU fallback.')
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Use cpu or cuda:<index>.')
        state, checksum = load_checkpoint(weights)
        model = make_backbone()
        mapped, excluded = convert_backbone_state(state, model)
        model.load_state_dict(mapped, strict=True)
        self.model = model.eval().requires_grad_(False).to(self.device)
        self.identity = {
            'kind': 'TSN_R50_ActivityNet_clip_flow_150e',
            'checkpoint_name': CHECKPOINT_NAME, 'checkpoint_url': CHECKPOINT_URL,
            'checkpoint_sha256': checksum, 'input_channels': 10, 'output_dim': 2048,
            'backend': 'strict_mmaction_backbone_to_torchvision_resnet50',
            'stride_style': 'pytorch', 'feature_layer': 'global_average_pool_after_layer4',
            'action_head_excluded_keys': excluded, 'trained_here': False,
            'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
            'precision': 'float32_no_tf32', 'numerical_mmaction_parity_verified_here': False,
        }
        del state, mapped

    @torch.inference_mode()
    def encode(self, stacks):
        import numpy as np
        a = np.asarray(stacks, dtype=np.float32)
        if a.ndim != 4 or a.shape[1:] != (10, 256, 256) or not len(a):
            raise ValueError(f'Expected [B,10,256,256] normalized flow; got {a.shape}.')
        if not np.isfinite(a).all() or a.min() < -1.001 or a.max() > 1.001:
            raise ValueError('Invalid flow tensor: use the explicit uint8/bound20/128 normalization.')
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('The extraction encoder must stay frozen in eval mode.')
        features = self.model(torch.from_numpy(np.ascontiguousarray(a)).to(self.device))
        features = features.float().cpu().numpy()
        if features.shape != (len(a), 2048) or not np.isfinite(features).all():
            raise ValueError('Invalid TSN flow feature output.')
        return features
