"""Exercise the complete CLI with non-speech synthetic videos and artificial labels.

Run from any directory: python tests/run_smoke.py --out <NEW_DIRECTORY>
Requires FFmpeg, FFprobe and the normal project dependencies. No network needed.
Do NOT use the synthetic scores as deepfake-detection results.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new smoke-test output directory.')
    output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[1]
    steps = []

    def run(name, *command):
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, *map(str, command)], cwd=project,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        steps.append(name)
        print(f'PASS {name}', flush=True)

    raw, cache = output / 'raw', output / 'cache'
    run('make_data', 'tests/make_smoke_data.py', '--out', raw)
    command = ['prepare_audio.py', '--data-root', raw, '--out', cache, '--splits', 'train', 'dev', 'test']
    run('prepare', *command)
    run('resume_prepare', *command)
    summary = json.loads((cache / 'summary.json').read_text())
    assert summary['resumed_this_run'] == 12 and summary['failures_this_run'] == 0
    for model in ['resnet18', 'aasist']:
        training = output / model
        run(f'train_{model}', 'experiment.py', 'train', '--root', cache, '--model', model,
            '--init', 'scratch', '--out', training, '--device', 'cpu', '--batch-size', '2',
            '--threads', '2', '--epochs', '2', '--accumulate', '3', '--samples-per-epoch', '4')
        run(f'evaluate_{model}', 'experiment.py', 'evaluate', '--root', cache,
            '--checkpoint', training / 'best.pt', '--split', 'test', '--out', output / f'eval_{model}',
            '--device', 'cpu', '--batch-size', '2', '--threads', '2')
        run(f'export_{model}', 'experiment.py', 'export', '--root', cache,
            '--checkpoint', training / 'best.pt', '--out', output / f'features_{model}',
            '--device', 'cpu', '--batch-size', '2', '--threads', '2')
    run('compare', 'compare.py', '--runs', output / 'eval_resnet18', output / 'eval_aasist',
        '--out', output / 'comparison', '--bootstrap', '100')
    (output / 'software_validation.json').write_text(json.dumps({
        'passed_steps': steps, 'synthetic_videos': 12, 'research_accuracy_claimed': False}, indent=2))
    print('All CLI steps passed. This is software validation, not a benchmark result.')


if __name__ == '__main__':
    main()
