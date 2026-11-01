#!/usr/bin/env python3
"""Download/check the exact pretrained TSN flow checkpoint linked by UMMAFormer."""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import sys
import urllib.request
from common import json_write, sha256
from tsn_encoder import CHECKPOINT_NAME, CHECKPOINT_URL, CHECKPOINT_PREFIX, TSNFlowEncoder


def fetch(out: Path):
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        temporary = out.with_name(out.name + '.download')
        request = urllib.request.Request(CHECKPOINT_URL, headers={'User-Agent': 'LAVDF-TSN-flow-research/1.0'})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as f:
                done = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk); done += len(chunk)
                    print(f'\rDownloaded {done / 1024**2:.1f} MiB', end='', flush=True)
            print()
            checksum = sha256(temporary)
            if not checksum.startswith(CHECKPOINT_PREFIX):
                raise ValueError('Downloaded bytes do not match the official checkpoint hash prefix. '
                                 'No output checkpoint has been published.')
            os.replace(temporary, out)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    encoder = TSNFlowEncoder(out, 'cpu')  # Full strict tensor/shape load, no inference.
    json_write(out.with_suffix(out.suffix + '.json'), {**encoder.identity, 'bytes': out.stat().st_size})
    print(f'Pretrained TSN flow weights verified and loadable: {out}')
    print('No random fallback and no new LAV-DF training. Use check_install.py for a forward test.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('weights') / CHECKPOINT_NAME)
    a = p.parse_args()
    fetch(a.out)

if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)
