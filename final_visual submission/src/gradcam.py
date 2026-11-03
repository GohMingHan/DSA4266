"""Reusable Fake-logit gradient engine and hook cleanup; DEV dispatch is in reporting."""

import numpy as np
import torch
from torch.nn import functional as F
from .models import StagedCNN


def target_layer(model, name):
    """Last spatial feature output (7x7 for native 224x224 inputs)."""
    network = model.network if isinstance(model, StagedCNN) else model
    if name == 'resnet50':
        return network.layer4[-1], 'network.layer4[-1]'
    if name == 'efficientnet_b0':
        return network.features[-1], 'network.features[-1]'
    raise ValueError('Grad-CAM supports ResNet50 and EfficientNet-B0 CNNs only')


class GradCAM:
    """Fake-logit Grad-CAM; autograd.grad never accumulates parameter gradients.

    Fresh input gradients make frozen backbones work. Hooks exist only during a
    call and are removed even on failure. FP32 avoids half-precision overflow.
    """
    def __init__(self, model, layer):
        self.model, self.layer = model, layer

    def __call__(self, images):
        states = [(module, module.training) for module in self.model.modules()]
        captured = []
        handle = self.layer.register_forward_hook(lambda module, inputs, output: captured.append(output))
        try:
            self.model.eval()
            with torch.inference_mode(False), torch.enable_grad(), torch.autocast(images.device.type, enabled=False):
                inputs = images.detach().clone().float().requires_grad_(True)
                logits = self.model(inputs).reshape(-1)
                if len(captured) != 1 or captured[0].ndim != 4 or logits.numel() != len(inputs):
                    raise ValueError('Expected one spatial activation and one binary Fake logit per image')
                activation = captured[0]
                gradient, = torch.autograd.grad(logits.sum(), activation)
                if not all(bool(torch.isfinite(t).all()) for t in (logits, activation, gradient)):
                    raise FloatingPointError('Non-finite Grad-CAM logit, activation or gradient')
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                raw = (weights * activation).sum(dim=1, keepdim=True).relu()
                maps = F.interpolate(raw, size=inputs.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
                maxima = maps.flatten(1).amax(dim=1).view(-1, 1, 1)
                maps = torch.where(maxima > 0, maps / maxima.clamp_min(torch.finfo(maps.dtype).tiny), torch.zeros_like(maps))
                return {'heatmaps': maps.detach().cpu().numpy(),
                        'probabilities': logits.detach().sigmoid().cpu().numpy(),
                        'logits': logits.detach().cpu().numpy(),
                        'zero_maps': (maxima.reshape(-1) == 0).cpu().tolist(),
                        'activation_shape': list(activation.shape)}
        finally:
            handle.remove()
            captured.clear()
            # Direct flags preserve mixed BN/dropout states without invoking custom train().
            for module, training in states:
                module.training = training
