"""Independent ImageNet CNNs, staged fine-tuning and frozen BN policy."""

import hashlib
import torch
from pathlib import Path
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights

MODELS = ('resnet50', 'efficientnet_b0')


def weight_enum(model):
    if model not in MODELS:
        raise ValueError(f'Unsupported model: {model}')
    return ResNet50_Weights.IMAGENET1K_V2 if model == 'resnet50' else EfficientNet_B0_Weights.IMAGENET1K_V1


def weights_path(cache, model='resnet50'):
    return Path(cache) / weight_enum(model).url.rsplit('/', 1)[-1]


def download_weights(cache, model='resnet50'):
    """Explicit network command; training itself never downloads implicitly."""
    weights = weight_enum(model)
    Path(cache).mkdir(parents=True, exist_ok=True)
    torch.hub.load_state_dict_from_url(weights.url, model_dir=str(cache), check_hash=True, map_location='cpu')
    return str(weights_path(cache, model))


def pretrained_backbone(name, cache, pretrained):
    backbone = resnet50(weights=None) if name == 'resnet50' else efficientnet_b0(weights=None)
    if pretrained:
        path = weights_path(cache, name)
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}. Run download-weights --model {name} --cache {cache}')
        expected = path.stem.rsplit('-', 1)[-1]
        if not hashlib.sha256(path.read_bytes()).hexdigest().startswith(expected):
            raise ValueError('Pretrained weight SHA256 mismatch')
        backbone.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return backbone


def build_model(model='resnet50', cache='.cache/weights', pretrained=True, hidden_size=128, dropout=0.3):
    if model not in MODELS:
        raise ValueError(f'Unsupported model: {model}')
    name = 'resnet50' if model == 'resnet50' else 'efficientnet_b0'
    backbone = pretrained_backbone(name, cache, pretrained)
    if model == 'resnet50':
        backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(2048, 1))
        return backbone
    if model == 'efficientnet_b0':
        backbone.classifier[0] = nn.Dropout(dropout)
        backbone.classifier[1] = nn.Linear(1280, 1)
        return backbone
    raise ValueError("Only the two finalized CNN architectures are supported here")


class StagedCNN(nn.Module):
    def __init__(self, config, weights_root, pretrained=True):
        super().__init__()
        self.network = build_model(config.model, weights_root, pretrained=pretrained, dropout=config.dropout)
        self.name = config.model
        self.phase = 'head'
        self.allow_finetune = config.finetune_final_stage
        self.set_phase(False)

    def head(self):
        return self.network.fc if self.name == 'resnet50' else self.network.classifier

    def final_modules(self):
        return [self.network.layer4] if self.name == 'resnet50' else list(self.network.features.children())[-2:]

    def final_parameters(self):
        return [p for module in self.final_modules() for p in module.parameters()]

    def set_phase(self, finetune):
        active = finetune and self.allow_finetune
        for p in self.network.parameters():
            p.requires_grad_(False)
        for p in self.head().parameters():
            p.requires_grad_(True)
        for p in self.final_parameters():
            p.requires_grad_(active)
        self.phase = 'final_stage' if active else 'head'
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if not hasattr(self, 'name'):
            return self
        self.network.eval()  # frozen stochastic depth/dropout and BN remain fixed
        self.head().train(mode)
        if self.phase == 'final_stage':
            for module in self.final_modules():
                module.train(mode)
        # Conservatively keep all ImageNet BN running statistics fixed, including
        # final-stage BN. Trainable final-stage affine weights may still update.
        for module in self.network.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, images):
        return self.network(images)

    def trainable_report(self):
        names = [name for name, p in self.named_parameters() if p.requires_grad]
        expected = {id(p) for p in self.head().parameters()}
        if self.phase == 'final_stage':
            expected.update(id(p) for p in self.final_parameters())
        actual = {id(p) for p in self.parameters() if p.requires_grad}
        if actual != expected:
            raise AssertionError('Unexpected trainable backbone parameters')
        if any(m.training for m in self.network.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)):
            raise AssertionError('BN running statistics must be frozen')
        return {'phase': self.phase, 'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'trainable_names': names, 'batchnorm_running_stats_frozen': True}


def make_optimizer(model, config):
    return torch.optim.AdamW([
        {'params': list(model.head().parameters()), 'lr': config.head_lr, 'name': 'head'},
        {'params': model.final_parameters(), 'lr': config.backbone_lr, 'name': 'final_stage'},
    ], weight_decay=config.weight_decay)
