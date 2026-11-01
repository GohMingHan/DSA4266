#!/usr/bin/env python3
"""File-driven Optuna/TPE search wrapper for an existing UMMAFormer run_experiment.py.

Each trial uses the existing train CLI and reads DEV metrics from best_video_auc.pt.
Completely bypasses SQLite databases to avoid permission/readonly crashes, saving 
all records and results directly as standalone JSON files.
"""
from __future__ import annotations
import argparse, json, math, shlex, shutil, subprocess, sys, time, traceback
from pathlib import Path
import optuna
import torch


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True, help='Folder containing run_experiment.py')
    p.add_argument('--manifest', type=Path, required=True, help='Existing train/dev feature manifest')
    p.add_argument('--upstream', type=Path, required=True, help='Existing UMMAFormer upstream source directory')
    p.add_argument('--out-root', type=Path, required=True, help='New output folder, outside project source')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--amp', choices=['none', 'fp16', 'bf16'], default='none')
    p.add_argument('--trials', type=int, default=15)
    p.add_argument('--timeout-hours', type=float)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def metric(metrics, keys):
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    raise KeyError(f'No finite metric among {keys}; checkpoint keys: {sorted(metrics)}')


def main():
    a = get_args()
    project = a.project.expanduser().resolve()
    manifest = a.manifest.expanduser().resolve()
    upstream = a.upstream.expanduser().resolve()
    root = a.out_root.expanduser().resolve()
    
    if not (project / 'run_experiment.py').is_file():
        raise SystemExit(f'Missing {project / "run_experiment.py"}')
    if not manifest.is_file() or not upstream.is_dir():
        raise SystemExit('Manifest file or upstream directory does not exist.')
    if a.trials < 1 or a.epochs < 2 or a.batch_size < 1 or a.patience < 1:
        raise SystemExit('trials/batch-size/patience must be positive and epochs >= 2.')
    if root == project or project in root.parents or root in project.parents:
        raise SystemExit('Use a dedicated output folder outside the project source tree.')
        
    root.mkdir(parents=True, exist_ok=True)
    
    # Create an in-memory Optuna study with TPE sampler (no SQLite files needed!)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=a.seed),
    )

    def command(out, lr, wd, warmup):
        return [sys.executable, str(project / 'run_experiment.py'), 'train',
                '--upstream', str(upstream), '--manifest', str(manifest),
                '--input-mode', 'rgb-flow', '--out', str(out), '--device', a.device,
                '--batch-size', str(a.batch_size), '--workers', str(a.workers),
                '--epochs', str(a.epochs), '--warmup-epochs', str(warmup),
                '--lr', str(lr), '--weight-decay', str(wd),
                '--patience', str(a.patience), '--seed', str(a.seed), '--amp', a.amp]

    if a.dry_run:
        cmd = command(root / 'trial_dry_run', 0.0003, 0.05, min(5, a.epochs - 1))
        print('Dry run only; no training started.\n' + shlex.join(cmd))
        return

    start_time = time.time()
    timeout_seconds = a.timeout_hours * 3600 if a.timeout_hours else None

    for trial_idx in range(a.trials):
        if timeout_seconds and (time.time() - start_time) > timeout_seconds:
            print('\nTimeout reached. Stopping optimization loop.')
            break

        trial = study.ask()
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        wd = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
        warmup = trial.suggest_int('warmup_epochs', 0, min(8, a.epochs - 1))
        
        out = root / f'trial_{trial_idx:04d}'
        ckpt = out / 'best_video_auc.pt'
        
        # If a valid checkpoint already exists, resume/reuse it seamlessly
        if ckpt.is_file():
            try:
                data = torch.load(ckpt, map_location='cpu', weights_only=True)
                dev = data.get('dev_metrics')
                if isinstance(dev, dict):
                    ap = metric(dev, ('video_visual_ap', 'video_ap', 'video_average_precision'))
                    auc = metric(dev, ('video_visual_roc_auc', 'video_roc_auc', 'video_auc'))
                    study.tell(trial, ap)
                    print(f'\n[trial {trial_idx}] Resuming from existing checkpoint. DEV AP={ap:.6f}; DEV ROC-AUC={auc:.6f}', flush=True)
                    continue
            except Exception:
                pass

        if out.exists():
            shutil.rmtree(out)
            
        cmd = command(out, lr, wd, warmup)
        print(f'\n[trial {trial_idx}] {shlex.join(cmd)}', flush=True)
        
        record = {
            'trial': trial_idx,
            'params': {'lr': lr, 'weight_decay': wd, 'warmup_epochs': warmup, 
                       'batch_size': a.batch_size, 'epochs': a.epochs, 
                       'patience': a.patience, 'seed': a.seed, 'amp': a.amp},
            'command': cmd,
            'started_unix': time.time(),
            'status': 'running'
        }
        
        try:
            subprocess.run(cmd, cwd=project, check=True)
            if not ckpt.is_file():
                raise FileNotFoundError(f'Missing expected checkpoint: {ckpt}')
                
            data = torch.load(ckpt, map_location='cpu', weights_only=True)
            dev = data.get('dev_metrics')
            if not isinstance(dev, dict):
                raise ValueError('Checkpoint has no dev_metrics dictionary; inspect runner checkpoint format.')
                
            ap = metric(dev, ('video_visual_ap', 'video_ap', 'video_average_precision'))
            auc = metric(dev, ('video_visual_roc_auc', 'video_roc_auc', 'video_auc'))
            
            record.update(
                status='complete',
                finished_unix=time.time(),
                objective_video_visual_ap=ap,
                secondary_video_visual_roc_auc=auc,
                checkpoint=str(ckpt),
                dev_metrics=dev
            )
            
            record_path = out / 'optuna_trial.json'
            record_path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            
            study.tell(trial, ap)
            print(f'[trial {trial_idx}] DEV AP={ap:.6f}; DEV ROC-AUC={auc:.6f}', flush=True)
            
        except Exception as e:
            out.mkdir(parents=True, exist_ok=True)
            record.update(
                status='failed',
                finished_unix=time.time(),
                error=repr(e),
                traceback=traceback.format_exc()
            )
            record_path = out / 'optuna_trial.json'
            record_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            print(f'[trial {trial_idx}] Failed with error: {e}', flush=True)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best_trial = study.best_trial if len(completed_trials) > 0 else None
    best = {
        'objective': 'development video_visual_ap',
        'best_value': best_trial.value if best_trial else None,
        'best_trial_number': best_trial.number if best_trial else None,
        'best_params': best_trial.params if best_trial else None,
        'warning': 'Existing trainer chooses best_video_auc.pt by DEV ROC-AUC; AP is measured on that checkpoint.'
    }
    (root / 'best_result.json').write_text(json.dumps(best, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('\nOptimization finished. Best result:\n' + json.dumps(best, indent=2))

if __name__ == '__main__':
    main()