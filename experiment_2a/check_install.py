"""CPU/GPU numerical smoke test of both models; not a deepfake accuracy test."""
import argparse
import json
import time
import torch
from common import device_from_arg
from models import build_model, DEFAULT_AASIST_DIR
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cpu')
    p.add_argument('--aasist-dir', type=Path, default=DEFAULT_AASIST_DIR)
    args = p.parse_args()
    torch.set_num_threads(2)
    device = device_from_arg(args.device)
    for name, shape in [('resnet18', (2, 1, 128, 404)), ('aasist', (2, 64600))]:
        torch.manual_seed(42)
        model = build_model(name, aasist_dir=args.aasist_dir).to(device)
        model.train()
        start = time.perf_counter()
        features, logits = model.forward_features(torch.randn(shape, device=device) * .03)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.], device=device))
        if not torch.isfinite(loss):
            raise RuntimeError(f'{name}: nonfinite loss')
        loss.backward()
        if not any(p.grad is not None for p in model.parameters()):
            raise RuntimeError(f'{name}: no gradients')
        if device.type == 'cuda':
            torch.cuda.synchronize()
        print(json.dumps({'model': name, 'features': list(features.shape), 'logits': list(logits.shape),
                          'parameters': sum(p.numel() for p in model.parameters()),
                          'forward_backward_passed': True, 'seconds': time.perf_counter() - start}))
        del model


if __name__ == '__main__':
    main()
