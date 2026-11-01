"""Optional: retrieve the authors' original AASIST source and pretrained weights.

The offline scratch runs do not need this. Network access is required here.
Downloads are pinned to one resolved Git revision and hashes are recorded.
Never loads code or model files during download.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
import urllib.error

REPOSITORY = 'clovaai/aasist'
FILES = {'AASIST.py': 'models/AASIST.py', 'AASIST.conf': 'config/AASIST.conf',
         'LICENSE': 'LICENSE', 'data_utils_reference.py': 'data_utils.py',
         'README_upstream.md': 'README.md'}


def retrieve(url, limit=30_000_000):
    request = urllib.request.Request(url, headers={'User-Agent': 'LAVDF-Experiment2A/1.0'})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f'Download exceeds size limit: {url}')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'upstream_aasist')
    p.add_argument('--revision', default='main', help='Git ref or full commit SHA; resolved and locked on first run.')
    p.add_argument('--weights', action='store_true', help='Also download the official ASVspoof 2019 LA pretrained state dictionary.')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lock_path = args.out / 'download_manifest.json'
    previous = json.loads(lock_path.read_text()) if lock_path.exists() else None
    try:
        if previous:
            revision = previous['commit']
            if args.revision != 'main' and args.revision not in (revision, previous['requested_revision']):
                raise ValueError('This destination is locked to another revision. Use a new --out.')
        elif re.fullmatch(r'[0-9a-fA-F]{40}', args.revision):
            revision = args.revision.lower()
        else:
            url = f'https://api.github.com/repos/{REPOSITORY}/commits/{urllib.parse.quote(args.revision, safe="")}'
            revision = json.loads(retrieve(url))['sha']
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Could not resolve a full commit SHA.')
        wanted = dict(FILES)
        if args.weights:
            wanted['AASIST.pth'] = 'models/weights/AASIST.pth'
        files = dict(previous.get('files', {})) if previous else {}
        for filename, remote in wanted.items():
            target = args.out / filename
            if target.exists() and filename in files:
                checksum = hashlib.sha256(target.read_bytes()).hexdigest()
                if checksum == files[filename]['sha256']:
                    print(f'Verified existing {filename}')
                    continue
                raise ValueError(f'Existing {target} differs from its recorded hash. Use a fresh directory.')
            url = f'https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{remote}'
            data = retrieve(url)
            if filename.endswith('.pth') and data.startswith(b'version https://git-lfs'):
                raise ValueError('Received a Git LFS pointer, not a checkpoint. Retrieve the real official weight file.')
            temporary = target.with_name(target.name + '.tmp')
            temporary.write_bytes(data)
            temporary.replace(target)
            files[filename] = {'url': url, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
            print(f'Downloaded {filename}: {len(data):,} bytes')
        lock = {'repository': f'https://github.com/{REPOSITORY}', 'requested_revision': args.revision,
                'commit': revision, 'files': files}
        lock_path.write_text(json.dumps(lock, indent=2), encoding='utf-8')
        print(f'Locked source revision: {revision}')
        print(f'Official implementation: use --aasist-dir "{args.out}"')
        if args.weights:
            print(f'Pretrained weights: use --init pretrained --aasist-weights "{args.out / "AASIST.pth"}"')
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(f'AASIST download failed: {exc}\nNo training was started. Check internet access; the bundled scratch model does not require a download.') from exc


if __name__ == '__main__':
    main()
