"""Common model interface: (embedding, logit with positive meaning manipulated)."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

BASE = Path(__file__).resolve().parent
DEFAULT_AASIST_DIR = BASE / 'third_party' / 'aasist'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class AudioResNet18(nn.Module):
    embedding_dim = 512
    input_kind = 'spectrogram'

    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        old = self.backbone.conv1
        conv = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                conv.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
        self.backbone.conv1 = conv
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 1))
        self.provenance = {'implementation': 'torchvision_resnet18_single_channel',
                           'wrapper_sha256': sha256(Path(__file__))}

    def encode(self, x):
        return self.backbone(x)

    def forward_features(self, x):
        h = self.encode(x)
        return h, self.head(h).squeeze(-1)

    def forward(self, x):
        return self.forward_features(x)[1]

    def freeze_encoder(self, frozen):
        for p in self.backbone.parameters():
            p.requires_grad_(not frozen)
        if frozen:
            self.backbone.eval()


class AudioAASIST(nn.Module):
    input_kind = 'waveform'

    def __init__(self, weights: Path | None = None, source_dir: Path = DEFAULT_AASIST_DIR):
        super().__init__()
        source = source_dir / 'AASIST.py'
        configuration = source_dir / 'AASIST.conf'
        for path in [source, configuration]:
            if not path.is_file():
                raise FileNotFoundError(f'Missing AASIST implementation file: {path}')
        cfg = json.loads(configuration.read_text(encoding='utf-8'))['model_config']
        if cfg['architecture'] != 'AASIST' or cfg['nb_samp'] != 64600:
            raise ValueError('This adapter expects the standard full AASIST configuration.')
        spec = importlib.util.spec_from_file_location('experiment_2a_aasist_impl', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.network = module.Model(cfg)
        self.embedding_dim = 5 * cfg['gat_dims'][1]
        self.provenance = {'implementation': ('aasist_official_architecture_port' if source_dir.resolve() == DEFAULT_AASIST_DIR.resolve() else 'aasist_external_source'),
                           'model_sha256': sha256(source), 'config_sha256': sha256(configuration),
                           'wrapper_sha256': sha256(Path(__file__)), 'model_config': cfg,
                           'source_url': 'https://github.com/clovaai/aasist/blob/main/models/AASIST.py'}
        if weights is not None:
            # No unsafe pickle fallback. Expected: authors' plain state dictionary.
            state = torch.load(weights, map_location='cpu', weights_only=True)
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
                raise ValueError('Expected a tensor-only official AASIST state dictionary.')
            state = {k.removeprefix('module.'): v for k, v in state.items()}
            self.network.load_state_dict(state, strict=True)
            self.provenance['initial_weights_sha256'] = sha256(weights)

    @staticmethod
    def spoof_logit(two_class_logits):
        # Official labels: 0=spoof, 1=bonafide. Dataset labels: 1=manipulated.
        # sigmoid(logit_0-logit_1) is exactly softmax(logits)[:, 0].
        return two_class_logits[:, 0] - two_class_logits[:, 1]

    def forward_features(self, x):
        if x.ndim != 2 or x.shape[1] != 64600:
            raise ValueError('AASIST expects [batch, 64600] raw 16-kHz waveform samples.')
        h, logits = self.network(x, Freq_aug=False)
        return h, self.spoof_logit(logits)

    def forward(self, x):
        return self.forward_features(x)[1]

    def freeze_encoder(self, frozen):
        for name, p in self.network.named_parameters():
            p.requires_grad_(not frozen or name.startswith('out_layer.'))
        if frozen:
            self.network.eval()
            self.network.out_layer.train()


def build_model(name, initialization='scratch', aasist_weights=None, aasist_dir=DEFAULT_AASIST_DIR):
    if name == 'resnet18':
        return AudioResNet18(pretrained=initialization == 'pretrained')
    if name == 'aasist':
        if initialization == 'pretrained' and aasist_weights is None:
            raise ValueError('Pretrained AASIST requires --aasist-weights; run download_aasist.py first.')
        return AudioAASIST(weights=aasist_weights if initialization == 'pretrained' else None,
                           source_dir=aasist_dir)
    raise ValueError(f'Unknown model: {name}')
