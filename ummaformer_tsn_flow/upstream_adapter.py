"""Import and configure the actual UMMAFormer release. No replacement architecture."""
from __future__ import annotations
from pathlib import Path
import copy
import importlib
import json
import subprocess
import sys
import torch
from common import sha256, digest


EXPECTED_MODEL = 'AVLocPointTransformerRecoveryNoNorm'


def source_identity(repo):
    repo = Path(repo).resolve()
    required = [repo / 'configs/UMMAFormer/lavdf_tsn.yaml',
                repo / 'libs/modeling/av_recoverynonorm_meta_arch.py']
    if not all(p.is_file() for p in required):
        raise ValueError(f'Not the expected official UMMAFormer checkout: {repo}')
    paths = sorted([p for p in (repo / 'libs').rglob('*') if p.is_file() and p.suffix in ('.py', '.cpp', '.h', '.cu')] + [required[0]])
    hashes = {str(p.relative_to(repo)): sha256(p) for p in paths}
    try:
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    lock = repo.parent / 'upstream_revision.txt'
    if lock.is_file() and lock.read_text().strip() != commit:
        raise ValueError('UMMAFormer commit differs from the recorded bootstrap checkout.')
    return {'git_commit': commit, 'files_sha256': hashes, 'source_digest': digest(hashes)}


def load_official(repo):
    repo = Path(repo).resolve()
    identity = source_identity(repo)
    sys.path.insert(0, str(repo / 'libs/utils'))  # in-place NMS build is also supported
    sys.path.insert(0, str(repo))
    try:
        core = importlib.import_module('libs.core')
        modeling = importlib.import_module('libs.modeling')
        utils = importlib.import_module('libs.utils')
    except ImportError as exc:
        raise ImportError('Could not import the official implementation. Run bootstrap.sh in this '
                          'environment, including the C++ NMS build. Original error: ' + str(exc)) from exc
    if not Path(core.__file__).resolve().is_relative_to(repo):
        raise ValueError('A different package named libs was imported; start a fresh Python process.')
    cfg = core.load_config(str(repo / 'configs/UMMAFormer/lavdf_tsn.yaml'))
    if cfg['model_name'] != EXPECTED_MODEL or cfg['dataset']['input_dim'] != 4096:
        raise ValueError('Unexpected upstream visual-only configuration; review this new source version first.')
    if cfg['dataset'].get('audio_input_dim', 0) != 0 or cfg['dataset']['file_prefix'] != ['rgb', 'flow']:
        raise ValueError('Expected visual RGB+flow only, with no audio input.')
    # Library configuration is retained. The separate loader implements visual labels
    # and complete-clip feature resampling; native model/loss/NMS implementations remain.
    return cfg, modeling.make_meta_arch, utils, identity


def build_official(repo, device, input_mode='rgb-flow'):
    cfg, factory, utils, identity = load_official(repo)
    if input_mode not in ('rgb-flow', 'rgb'):
        raise ValueError('Unknown input mode.')
    # Deliberate input-representation adaptation, not shape-compatible TSN imitation.
    cfg = copy.deepcopy(cfg)
    channels = 2560 if input_mode == 'rgb-flow' else 512
    cfg['dataset']['input_dim'] = channels
    cfg['dataset']['audio_input_dim'] = 0
    cfg['model']['input_dim'] = channels
    cfg['model']['audio_input_dim'] = 0
    cfg['custom_input'] = {'mode': input_mode, 'rgb': 'preserved frozen ImageNet ResNet18 avgpool512 from reference cache',
                           'flow': 'frozen pretrained TSN R50 ActivityNet flow avgpool2048; see manifest for estimator/frontend',
                           'time': 'actual PTS interpolated to 768 uniform bin centers'}
    model = factory(cfg['model_name'], **cfg['model']).to(device)
    counts = {'total_parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return model, cfg, utils, identity, counts


def local_sources():
    root = Path(__file__).resolve().parent
    return {p.name: sha256(p) for p in sorted(root.glob('*.py'))}


def split_identity(manifest):
    result = {}
    for split in ('train', 'dev', 'test'):
        rows = [r for r in manifest['videos'] if r['split'] == split]
        if not rows:
            continue
        content = []
        for r in sorted(rows, key=lambda r: r['file']):
            content.append({k: r[k] for k in ('file', 'split', 'source_group', 'modify_video',
                 'modify_audio', 'visual_periods', 'duration_s', 'config_id')} | {
                 'cache_sha256': r['cache']['sha256']})
        result[split] = digest(content)
    return result


class ExponentialAverage:
    def __init__(self, model, decay=.999):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        for name, averaged in self.module.state_dict().items():
            current = source[name].detach()
            if torch.is_floating_point(averaged):
                averaged.mul_(self.decay).add_(current, alpha=1 - self.decay)
            else:
                averaged.copy_(current)
