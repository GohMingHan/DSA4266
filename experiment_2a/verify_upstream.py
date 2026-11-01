"""Compare the bundled AASIST port to a separately downloaded official source.

Run download_aasist.py first. This check is stronger than a shape/parameter-count
test: it loads identical random parameters and compares eval-mode features/logits.
It does not verify published ASVspoof metrics or real-world detection accuracy.
"""
import argparse
from pathlib import Path
import torch
from models import AudioAASIST


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, default=Path('upstream_aasist'))
    p.add_argument('--atol', type=float, default=1e-5)
    p.add_argument('--rtol', type=float, default=1e-4)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    local = AudioAASIST().eval()
    official = AudioAASIST(source_dir=args.upstream).eval()
    official.network.load_state_dict(local.network.state_dict(), strict=True)
    with torch.inference_mode():
        for amplitude in (0.0, 0.01, 0.1):
            x = amplitude * torch.randn(2, 64600)
            lh, ll = local.forward_features(x)
            oh, ol = official.forward_features(x)
            torch.testing.assert_close(lh, oh, atol=args.atol, rtol=args.rtol)
            torch.testing.assert_close(ll, ol, atol=args.atol, rtol=args.rtol)
    print('PASS: parameter keys/shapes and eval outputs match within the specified tolerances.')


if __name__ == '__main__':
    main()
