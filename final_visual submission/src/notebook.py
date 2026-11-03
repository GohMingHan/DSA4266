"""Shared notebook setup, one-model execution and saved-result presentation.

Training algorithms remain in workflow/training. Importing this module never
starts training. Each notebook model cell has its own explicit confirmation.
"""
from dataclasses import replace
import json
from pathlib import Path
import threading

from .config import RECIPES, INDEX_LOGICAL_SHA256, POOL_SHA256
from .workflow import (prepare_manifests, validate_inputs, setup_assets, execute_training,
                       discover_runs, choose_runs, new_encoder, project_id)
from .reporting import show_run, comparison, display_completed_run

_TRAINING_LOCK=threading.Lock()


def _require_tests(result):
    if result is None or result.testsRun<1 or not result.wasSuccessful():
        raise RuntimeError('Automatic preflight tests must pass before training. Run section1 in this kernel.')


def prepare_notebook(inputs,project,*,test_result,artifact_mode='REUSE',download_assets=False):
    """One shared setup for all five cells; original validators/builders are reused.

    REUSE never constructs metadata. Explicit BUILD also prepares the missing
    LSTM clip manifest. Optional mask assets do not block full-frame inspection.
    """
    _require_tests(test_result)
    project=Path(project)
    cached=project/'.cache/manifests/clip_manifest.json'
    if inputs.clips is None and cached.is_file():inputs=replace(inputs,clips=cached)
    inputs=prepare_manifests(inputs,project,artifact_mode=artifact_mode,
                            need_clips=inputs.clips is not None or artifact_mode=='BUILD')
    assets=project/'.cache/assets'
    detector=Path(inputs.detector) if inputs.detector else assets/'face_detection_yunet_2023mar.onnx'
    landmarker=Path(inputs.landmarker) if inputs.landmarker else assets/'face_landmarker.task'
    if download_assets or inputs.detector is not None or inputs.landmarker is not None or detector.is_file() and landmarker.is_file():
        inputs=setup_assets(inputs,project,download=download_assets)
    report=validate_inputs(inputs)
    report['notebook_ready']={'face_assets':inputs.detector is not None and inputs.landmarker is not None,
                              'lstm_manifest':inputs.clips is not None}
    return inputs,report


def collect_runs(project,*,runs=None,saved_runs=None):
    """Discover restored runs; retain the original explicit-choice ambiguity rule."""
    return choose_runs(project,{**(saved_runs or {}),**(runs or {})})


def _model_run(model_id,project,runs,saved_runs):
    explicit={**(saved_runs or {}),**(runs or {})}
    if model_id in explicit:
        run=Path(explicit[model_id])
    else:
        paths=discover_runs(project)[model_id]
        if len(paths)>1:raise ValueError('Multiple runs for '+model_id+'; choose SAVED_RUNS[model_id] explicitly.')
        if not paths:return None
        run=paths[0]
    identity_file=run/'experiment_identity.json'
    if not identity_file.is_file():raise FileNotFoundError('Run identity missing: '+str(run))
    identity=json.loads(identity_file.read_text(encoding='utf-8'))
    if identity.get('project_id')!=project_id(project) or identity.get('model_id')!=model_id:
        raise ValueError('Foreign or wrong-model run refused: '+str(run))
    return run


def _show_model(model_id,run,project):
    if run is None:
        print(RECIPES[model_id].label+': Pending. No training requested.',flush=True)
        return {}
    if (run/'training_summary.json').is_file():
        if (run/'best.pt').is_file():
            show_run(run,project)  # Original complete table, yellow best epoch, original graph order.
        else:
            print('Saved completed metrics; best checkpoint is missing and cannot be ranked as verified.',flush=True)
            display_completed_run(run,name=RECIPES[model_id].label,kind='clip' if model_id=='resnet50_lstm' else 'frame')
        summary=json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
        print('Best selected epoch:',summary.get('best_epoch'),'| run:',run,flush=True)
    else:
        print(RECIPES[model_id].label+': Partial; completed final results are not claimed. Use RESUME_FROM to recover.',flush=True)
    print('Best checkpoint:',run/'best.pt','| Last checkpoint:',run/'last.pt',flush=True)
    from IPython.display import FileLink,display
    for archive in sorted((run.parent/'downloads').glob('*.zip')):
        print('Download recovery/final ZIP:',archive,flush=True);display(FileLink(str(archive)))
    return {model_id:run}


def _baseline_encoder(project,runs,saved_runs,cnn_best):
    if cnn_best is None:
        baseline=_model_run('resnet50_baseline',project,runs,saved_runs)
        if baseline is None:raise FileNotFoundError('Complete section6A or restore this project baseline before section6C.')
        cnn_best=baseline/'best.pt'
    cnn_best=Path(cnn_best)
    if not (cnn_best.parent/'training_summary.json').is_file():
        raise ValueError('LSTM requires a completed new ResNet50 baseline, including its training_summary.json.')
    return new_encoder(cnn_best,project)  # Original project/model/phase/selected-best checks.


def run_model_cell(model_id,inputs,project,*,confirmation='',execution_mode='inspect',
                   test_result=None,data_report=None,runs=None,saved_runs=None,resume_from=None,
                   cnn_best=None,amp=True,workers=None,include_geometry_cache=False,start_new=False,
                   frame_cache=None,mask_cache=None,clip_pipeline=None):
    """Train exactly one confirmed model, or replay its independently saved result.

    Completed runs are replayed by default. Partial runs require an explicit
    RESUME_FROM entry. start_new=True is an advanced explicit independent rerun.
    A nonblocking lock prevents overlapping GPU jobs from these notebook cells.
    """
    if model_id not in RECIPES:raise ValueError('Unknown finalized model ID')
    expected='TRAIN_'+model_id
    if confirmation not in ('',expected):raise PermissionError('Use this model confirmation exactly: '+expected)
    if execution_mode!='train' or not confirmation:
        if confirmation:print('Training disabled by EXECUTION_MODE='+str(execution_mode)+'.',flush=True)
        return _show_model(model_id,_model_run(model_id,project,runs,saved_runs),project)
    _require_tests(test_result)
    if (not data_report or data_report.get('pool_sha256')!=POOL_SHA256 or
        data_report.get('index_logical_sha256')!=INDEX_LOGICAL_SHA256 or
        data_report.get('dev_frames')!=41746 or data_report.get('dev_videos')!=500):
        raise RuntimeError('Run shared section5 validation before training any model.')
    if not _TRAINING_LOCK.acquire(blocking=False):raise RuntimeError('Another notebook model is training; wait for that cell to finish.')
    try:
        resume={model_id:resume_from[model_id]} if resume_from and model_id in resume_from else {}
        existing=None if resume or start_new else _model_run(model_id,project,runs,saved_runs)
        if existing is not None and not resume and not start_new:
            if (existing/'training_summary.json').is_file():
                print('Reusing completed model. No retraining requested.',flush=True)
                return _show_model(model_id,existing,project)
            raise ValueError('Partial run exists. Set RESUME_FROM['+repr(model_id)+'] to its last.pt (or latest). A fresh restart must be explicit.')
        if RECIPES[model_id].input=='face_mask' and (inputs.detector is None or inputs.landmarker is None):
            raise FileNotFoundError('Face assets are not ready. Set YUNET/LANDMARKER or DOWNLOAD_PUBLIC_ASSETS in setup and run section5.')
        encoder=None
        if model_id=='resnet50_lstm':
            if inputs.clips is None:raise FileNotFoundError('LSTM clip manifest missing. Attach CLIP_MANIFEST, or explicitly use ARTIFACT_MODE=BUILD in shared section5 once.')
            if clip_pipeline is None:raise ValueError('Run Cell B — Prepare Clips before starting/resuming LSTM training.')
            encoder=_baseline_encoder(project,runs,saved_runs,cnn_best)
        else:
            from .prepared_frames import require_prepared_cache
            require_prepared_cache(frame_cache,inputs)
        executor=execute_training
        cache_options={}
        if frame_cache is not None and model_id!='resnet50_lstm':
            from .frame_cache import execute_cached_training
            executor=execute_cached_training
            cache_options={'frame_cache':frame_cache,'mask_cache':mask_cache}
        if model_id=='resnet50_lstm' and clip_pipeline is not None:
            from .session_inputs import execute_session_lstm
            executor=execute_session_lstm;cache_options={'clip_pipeline':clip_pipeline}
        result=executor([model_id],inputs,project,confirmation='TRAIN_SELECTED_MODELS',
            resume_from=resume,cnn_best=encoder,amp=amp,workers=workers,
            include_geometry_cache=include_geometry_cache,**cache_options)
        return _show_model(model_id,result[model_id],project)
    finally:
        _TRAINING_LOCK.release()


def show_comparison(project,runs=None):
    """Original table/ranking plus visible comparisons of verified saved metrics."""
    import matplotlib
    matplotlib.use('Agg')  # Same saved-PNG backend as the original training plots.
    rows,ranking,out=comparison(project,runs)
    from IPython.display import Image,display
    eligible=[row for row in rows if row['Status']=='Completed, verified checkpoint']
    if eligible:
        import matplotlib.pyplot as plt
        panels=[('Video AUC',[('Video PR-AUC','PR-AUC'),('Video ROC-AUC','ROC-AUC')]),
                ('Video classification at threshold 0.5',[('Video F1','F1'),('Video F2','F2')]),
                ('Full DEV BCE',[('DEV Loss','Natural DEV'),('Balanced DEV Loss','Balanced DEV')])]
        fig,axes=plt.subplots(1,3,figsize=(18,5))
        positions={row['Model']:i for i,row in enumerate(eligible)}
        for ax,(title,metrics) in zip(axes,panels):
            for offset,(field,label) in enumerate(metrics):
                points=[row for row in eligible if row.get(field) is not None]
                ax.bar([positions[row['Model']]+(offset-.5)*.35 for row in points],
                       [row[field] for row in points],width=.35,label=label)
            ax.set_xticks(list(positions.values()),list(positions),rotation=35,ha='right')
            ax.set_title(title);ax.set_ylim(bottom=0)
            if title!='Full DEV BCE':ax.set_ylim(top=1)
            ax.legend()
        fig.tight_layout();fig.savefig(out/'metrics_comparison.png',dpi=150);plt.close(fig)
        for name in ['video_comparison.png','metrics_comparison.png']:
            if (out/name).is_file():display(Image(filename=str(out/name)))
    return rows,ranking,out
