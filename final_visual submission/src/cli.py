"""Portable terminal orchestration over the preserved experimental implementation."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLIP_ID = 'clip_vit_b32_shared'
MODES = ('check', 'prepare', 'train', 'evaluate', 'results', 'gradcam', 'restore')
PATH_KEYS = ('data_root', 'weights_root', 'artifact_dir', 'output_dir', 'index_path',
             'train_pool', 'clip_manifest', 'resnet_checkpoint', 'clip_weights', 'yunet', 'landmarker')


def parser():
    p = argparse.ArgumentParser(description='DSA4266 visual experiments; no notebook or GitHub authentication required.')
    p.add_argument('--config', type=Path, default=ROOT/'settings.json')
    p.add_argument('--mode', choices=MODES)
    p.add_argument('--models', nargs='+', help='Model IDs separated by spaces, or all')
    for option, dest in [('data-root', 'data_root'), ('weights-root', 'weights_root'),
                         ('artifact-dir', 'artifact_dir'), ('output', 'output_dir'),
                         ('index-path', 'index_path'), ('train-pool', 'train_pool'),
                         ('clip-manifest', 'clip_manifest'), ('resnet-checkpoint', 'resnet_checkpoint'),
                         ('clip-weights', 'clip_weights'), ('yunet', 'yunet'), ('landmarker', 'landmarker')]:
        p.add_argument('--'+option, dest=dest, type=Path)
    p.add_argument('--run', action='append', default=[], metavar='MODEL=RUN_DIRECTORY')
    p.add_argument('--resume', action='append', default=[], metavar='MODEL=LAST_PT')
    p.add_argument('--build-metadata', action='store_true', help='Prepare mode only; explicitly build missing artifacts outside this folder')
    p.add_argument('--download-assets', action='store_true', help='Prepare mode only; explicit pinned public weight/asset downloads')
    p.add_argument('--prepare-frames', action='store_true', help='Prepare mode only; optional session RGB preparation')
    p.add_argument('--prepare-clips', action='store_true', help='Prepare mode only; exact original ordered clip references')
    p.add_argument('--gpu-smoke', action='store_true', help='Check mode: bounded real model/data GPU test, zero optimizer updates')
    p.add_argument('--fresh-evaluation', action='store_true', help='Evaluate mode: one full DEV GPU inference instead of saved predictions')
    p.add_argument('--restore-zip', type=Path, help='Restore mode: verified recovery ZIP; originals are never overwritten')
    p.add_argument('--list-models', action='store_true')
    return p


def all_models():
    from .config import ORDER
    return (*ORDER, CLIP_ID)


def load_settings(args):
    defaults = json.loads((ROOT/'settings.json').read_text(encoding='utf-8'))
    path = args.config.expanduser().resolve()
    supplied = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(supplied, dict) or set(supplied)-set(defaults):raise ValueError('Unknown settings or invalid settings object')
    settings = defaults | supplied
    for key in PATH_KEYS:
        value = getattr(args, key, None)
        if value is not None:settings[key] = value.expanduser().resolve()
        elif settings[key] is not None:
            value = Path(settings[key]).expanduser()
            settings[key] = (value if value.is_absolute() else path.parent/value).resolve()
    for key in ('saved_runs', 'resume_from'):
        if not isinstance(settings[key], dict):raise ValueError(key+' must map model IDs to paths')
        settings[key] = {model: (Path(value) if Path(value).is_absolute() else path.parent/value).resolve()
                         for model, value in settings[key].items()}
    for arguments, key in [(args.run, 'saved_runs'), (args.resume, 'resume_from')]:
        for item in arguments:
            model, sep, value = item.partition('=')
            if not sep or not value:raise ValueError('Use MODEL=PATH for --run/--resume')
            settings[key][model] = Path(value).expanduser().resolve()
    settings['mode'] = args.mode or settings['mode']
    models = args.models or settings['models']
    if models == ['all']:models = list(all_models())
    if not isinstance(models, list) or not models or len(set(models)) != len(models) or set(models)-set(all_models()):
        raise ValueError('Choose unique known model IDs; use --list-models')
    unknown = (set(settings['saved_runs']) | set(settings['resume_from']))-set(all_models())
    if unknown:raise ValueError('Unknown run/resume model IDs: '+str(sorted(unknown)))
    settings['models'] = [key for key in all_models() if key in models]
    if settings['mode'] not in MODES:raise ValueError('Unknown execution mode')
    if type(settings['amp']) is not bool or settings['workers'] is not None and (type(settings['workers']) is not int or settings['workers'] < 0):
        raise ValueError('Invalid AMP/worker setting')
    if settings['frame_loading_mode'] not in ('prepared', 'streaming'):raise ValueError('Choose prepared or explicitly configured streaming')
    if any(not isinstance(settings[k], (int, float)) or settings[k] < 0 for k in
           ('prepared_gib', 'streaming_gib', 'ram_cache_mib', 'disk_reserve_gib', 'clip_pixel_cache_gib')):
        raise ValueError('Resource bounds must be nonnegative')
    for key in ('artifact_dir', 'output_dir'):
        value = settings[key]
        if value == ROOT or ROOT in value.parents:raise ValueError(key+' must be OUTSIDE the submission folder')
    if settings['artifact_dir'] == settings['output_dir']:raise ValueError('Use separate artifact and output directories')
    settings['index_path'] = settings['index_path'] or settings['artifact_dir']/'index/candidates.sqlite'
    settings['train_pool'] = settings['train_pool'] or settings['artifact_dir']/'pool/train_pool.json'
    settings['clip_manifest'] = settings['clip_manifest'] or settings['artifact_dir']/'clip_manifest.json'
    settings['clip_weights'] = settings['clip_weights'] or settings['weights_root']/'clip_pretrained'
    if args.build_metadata or args.download_assets or args.prepare_frames or args.prepare_clips:
        if settings['mode'] != 'prepare':raise ValueError('Build/download/preparation flags are restricted to --mode prepare')
    if args.fresh_evaluation and settings['mode'] != 'evaluate':raise ValueError('--fresh-evaluation requires --mode evaluate')
    if args.gpu_smoke and settings['mode'] != 'check':raise ValueError('--gpu-smoke requires --mode check')
    return settings


def device_report(require_gpu=False):
    import torch
    available = torch.cuda.is_available()
    result = {'torch': str(torch.__version__), 'cuda_build': torch.version.cuda, 'cuda_available': available}
    if available:
        result.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                      total_memory_gib=torch.cuda.get_device_properties(0).total_memory/2**30)
    print('Device:', json.dumps(result), flush=True)
    if require_gpu and not available:
        raise RuntimeError('This operation requires an NVIDIA GPU, a compatible driver and a CUDA-enabled matching PyTorch/torchvision pair. CPU training/inference substitution is disabled. Use --mode results or --mode check for CPU-only work.')
    return result


def _outside(path):
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:raise ValueError('Generated files must remain outside the submission folder: '+str(path))
    return path


def workspace(settings):
    root = _outside(settings['output_dir']);marker = root/'submission_workspace.json'
    identity = json.loads((ROOT/'configs/project.json').read_text(encoding='utf-8'))
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise FileExistsError('Output directory is not a submission workspace; existing files preserved. Choose a new output directory.')
    root.mkdir(parents=True, exist_ok=True)
    if marker.is_file() and json.loads(marker.read_text())['project_id'] != identity['project_id']:
        raise ValueError('Foreign output workspace')
    (root/'configs').mkdir(exist_ok=True);(root/'docs').mkdir(exist_ok=True)
    config = root/'configs/project.json'
    if config.exists():
        if json.loads(config.read_text()) != identity:raise ValueError('Output project identity differs')
    else:shutil.copyfile(ROOT/'configs/project.json', config)
    provenance = root/'docs/source_provenance.json'
    if not provenance.exists():shutil.copyfile(ROOT/'configs/source_manifest.json', provenance)
    if not marker.exists():marker.write_text(json.dumps({'project_id': identity['project_id'], 'schema': 1})+'\n', encoding='utf-8')
    return root


def input_paths(settings):
    from .workflow import Inputs
    return Inputs(settings['data_root'], settings['index_path'], settings['train_pool'], settings['weights_root'],
                  settings['clip_manifest'] if settings['clip_manifest'].is_file() else None,
                  settings['yunet'], settings['landmarker'])


def prepare_metadata(settings, build=False):
    from .build_candidates import build_candidates, validate_candidate_index
    from .data import build_pool, PoolConfig
    from .config import POOL_SHA256
    from .checkpoints import digest
    index, pool = settings['index_path'], settings['train_pool']
    if not index.is_file():
        if not build:raise FileNotFoundError('External candidates.sqlite missing. Set index_path or explicitly use --mode prepare --build-metadata.')
        _outside(index)
        if index.name != 'candidates.sqlite':raise ValueError('Index filename must be candidates.sqlite')
        build_candidates(settings['data_root'], index.parent)
    validate_candidate_index(index, settings['data_root'])
    if not pool.is_file():
        if not build:raise FileNotFoundError('External train_pool.json missing; supply it or explicitly use --mode prepare --build-metadata.')
        _outside(pool)
        if pool.name != 'train_pool.json':raise ValueError('Pool filename must be train_pool.json')
        built = build_pool(settings['data_root'], pool.parent, PoolConfig())
        if Path(built).resolve() != pool.resolve():raise ValueError('Unexpected generated pool path')
    if digest(pool) != POOL_SHA256:raise ValueError('Fixed TRAIN pool hash differs; no selection rule was relaxed')
    from .workflow import validate_inputs
    inputs = input_paths(settings);report = validate_inputs(inputs)
    print('Verified artifacts:', json.dumps({k:report[k] for k in ('index_sha256','index_logical_sha256','pool_sha256','dev_frames','dev_videos')}), flush=True)
    return inputs, report


def run_tests(settings):
    # External reference tests retain their original exact hash/schema/record assertions.
    for variable, key in [('DSA4266_INDEX', 'index_path'), ('DSA4266_POOL', 'train_pool')]:
        if settings[key].is_file():os.environ[variable] = str(settings[key])
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise RuntimeError('Regression tests failed; model execution blocked')
    return result


def selected_runs(settings, root):
    from .workflow import discover_runs, project_id
    from .clip_reporting import find_clip_run
    found = discover_runs(root);result = {}
    for key in settings['models']:
        explicit = settings['saved_runs'].get(key)
        if key in settings['resume_from']:explicit = settings['resume_from'][key].parent
        if key == CLIP_ID:
            run = find_clip_run(root, explicit)
        else:
            paths = [explicit] if explicit is not None else found[key]
            if len(paths) > 1:raise ValueError('Multiple '+key+' runs; choose --run MODEL=RUN_DIRECTORY')
            run = Path(paths[0]) if paths else None
            if run is not None:
                identity = json.loads((run/'experiment_identity.json').read_text())
                if identity.get('project_id') != project_id(root) or identity.get('model_id') != key:
                    raise ValueError('Wrong project/model run: '+key)
        if run is not None:result[key] = run
    return result


def ordered_models(settings, root):
    """Use an explicit/discovered selected baseline, or add the baseline dependency."""
    models = list(settings['models']);encoder = settings['resnet_checkpoint']
    if 'resnet50_lstm' in models and 'resnet50_baseline' not in models and encoder is None:
        probe = dict(settings, models=['resnet50_baseline'], resume_from={})
        baseline = selected_runs(probe, root).get('resnet50_baseline')
        if baseline is not None and (baseline/'training_summary.json').is_file() and (baseline/'best.pt').is_file():
            encoder = baseline/'best.pt'
        else:models.insert(0, 'resnet50_baseline')
    return [k for k in all_models() if k in models], encoder


def prepare_rgb(settings, inputs, root):
    from .session_inputs import prepare_frames
    return prepare_frames(inputs, root, mode=settings['frame_loading_mode'], prepared_gib=settings['prepared_gib'],
        streaming_gib=settings['streaming_gib'], ram_cache_mib=settings['ram_cache_mib'], reserve_gib=settings['disk_reserve_gib'])[0]


def assets(settings, inputs, root, models, download=False):
    from .workflow import setup_assets
    from .models import download_weights, weights_path
    if any(k.endswith('_face_mask') for k in models):
        from .face_mask import ensure_landmark_runtime
        ensure_landmark_runtime(install=False)
        asset_root = settings['weights_root']/'face_assets'
        inputs = replace(inputs, detector=inputs.detector or asset_root/'face_detection_yunet_2023mar.onnx',
                         landmarker=inputs.landmarker or asset_root/'face_landmarker.task')
        inputs = setup_assets(inputs, root, download=download)
    if download:
        for model in ('resnet50', 'efficientnet_b0'):
            if any(k.startswith(model) for k in models) and not weights_path(settings['weights_root'], model).is_file():
                _outside(settings['weights_root']);download_weights(settings['weights_root'], model)
        if CLIP_ID in models:
            from .clip_model import prepare_clip_assets
            _outside(settings['clip_weights']);prepare_clip_assets(settings['clip_weights'], download=True)
    return inputs


@contextmanager
def training_lease(root):
    """One sequential training command per output workspace; never steal a lock."""
    lock = root/'training.lock'
    try:
        with lock.open('x', encoding='utf-8') as f:json.dump({'pid': os.getpid()}, f)
    except FileExistsError:raise RuntimeError('Training lock exists: '+str(lock)+'. Check its PID before manually removing a stale lock.') from None
    try:yield
    finally:lock.unlink(missing_ok=True)


def train_models(settings, root, inputs, data_report, tests):
    from .frame_cache import execute_cached_training
    from .session_inputs import prepare_clips, execute_session_lstm
    from .workflow import new_encoder
    from .clip_model import CLIPConfig
    from .clip_workflow import preflight_clip, train_clip
    models, encoder = ordered_models(settings, root)
    settings = dict(settings, models=models);runs = selected_runs(settings, root)
    pending = [k for k in models if k in settings['resume_from'] or k not in runs or not (runs[k]/'training_summary.json').is_file()]
    for key in pending:
        if key in runs and key not in settings['resume_from']:
            raise ValueError('Partial '+key+' run exists. Set resume_from or --resume explicitly; no restart was attempted.')
    print('Sequential model order:', ', '.join(models), flush=True)
    if not pending:return runs
    inputs = assets(settings, inputs, root, pending)
    with training_lease(root):
        cache = prepare_rgb(settings, inputs, root);clips = None
        for key in models:
            if key not in pending:
                print(key+': completed run reused: '+str(runs[key]), flush=True)
                if key == 'resnet50_baseline':encoder = runs[key]/'best.pt'
                continue
            common = dict(confirmation='TRAIN_SELECTED_MODELS', resume_from=settings['resume_from'], amp=settings['amp'], workers=settings['workers'])
            if key == 'resnet50_lstm':
                encoder = new_encoder(encoder, root)
                if clips is None:
                    if inputs.clips is None:inputs = replace(inputs, clips=settings['clip_manifest'])
                    inputs, clips, _ = prepare_clips(inputs, root, cache, clip_cache_gib=settings['clip_pixel_cache_gib'],
                        ram_cache_mib=settings['ram_cache_mib'], reserve_gib=settings['disk_reserve_gib'])
                result = execute_session_lstm([key], inputs, root, clip_pipeline=clips, cnn_best=encoder, **common)
                runs.update(result)
            elif key == CLIP_ID:
                cfg = CLIPConfig(amp=settings['amp']);source = settings['resume_from'].get(key)
                preflight_clip(inputs, root, settings['clip_weights'], config=cfg, cache=cache, resume_from=source)
                runs[key] = train_clip(inputs, root, settings['clip_weights'], confirmation='TRAIN_'+CLIP_ID,
                    config=cfg, test_result=tests, data_report=data_report, cache=cache, resume_from=source)
            else:
                runs.update(execute_cached_training([key], inputs, root, frame_cache=cache, **common))
                if key == 'resnet50_baseline':encoder = runs[key]/'best.pt'
    return runs


def display_results(settings, root, runs=None):
    """Terminal metrics plus the existing complete CSV/HTML/PNG reports."""
    from IPython.utils.capture import capture_output
    from .clip_reporting import show_six_comparison
    from .reporting import epoch_table
    runs = selected_runs(settings, root) if runs is None else runs
    for key in settings['models']:
        run = runs.get(key)
        if run is None:print(key+': Pending');continue
        print('\n'+key+' | '+str(run), flush=True)
        rows, best, _ = epoch_table(run, kind='clip' if key == 'resnet50_lstm' else 'frame')
        print('Selected epoch:', best)
        if rows:
            print('\t'.join(rows[0]))
            for row in rows:print('\t'.join(format_metric(k,v) for k,v in row.items()))
        print('Best:', run/'best.pt', '| Last:', run/'last.pt')
    # The reporting functions also work in ordinary Python. Suppress only their
    # notebook display objects, never training progress or errors.
    with capture_output():
        rows, ranking, out = show_six_comparison(root, {k:v for k,v in runs.items() if k != CLIP_ID}, clip_run=runs.get(CLIP_ID))
    print('\nModel | Level | Video PR-AUC | Video ROC-AUC | Video F2 | Status')
    for row in rows:
        print(' | '.join('N/A' if row.get(k) is None else f'{row[k]:.4f}' if isinstance(row[k], float) else str(row[k])
              for k in ['Model','Metric level','Video PR-AUC','Video ROC-AUC','Video F2','Status']))
    print('Ranking (video trapezoidal PR-AUC, then ROC-AUC):', json.dumps(ranking), flush=True)
    print('Comparison CSV/HTML/plots:', out, flush=True)
    return rows, ranking, out


def format_metric(column, value):
    if value is None:return 'N/A'
    if isinstance(value, float):
        return f'{value:.2e}' if column in ('Backbone LR','Classifier/Head LR') else f'{value:.4f}'
    return str(value)


def evaluate_models(settings, root, inputs, fresh=False):
    from .reporting import evaluate_dev
    from .clip_workflow import validate_clip_checkpoint, clip_epoch, POLICY
    from .clip_model import checkpoint_components, CLIPConfig, make_clip_loader
    from .data import DevSchedule, write_json
    from .evaluation import evaluate_predictions, prediction_artifacts
    from .checkpoints import digest
    from .workflow import new_id
    runs = selected_runs(settings, root)
    if fresh:inputs = assets(settings, inputs, root, list(runs))
    for key, run in runs.items():
        if not (run/'training_summary.json').is_file():print(key+': Partial; final evaluation skipped');continue
        if key != CLIP_ID:
            if fresh:report = evaluate_dev(run, inputs, root, force_inference=True)
            else:
                from .workflow import checkpoint_identity
                from .training import checkpoint_dev_report
                from .temporal import ClipIndex
                _, state = checkpoint_identity(run/'best.pt', root, key)
                if state['epoch']+1 != state['best_epoch']:raise ValueError('Use the selected best CNN/LSTM checkpoint')
                kind = 'clip' if key == 'resnet50_lstm' else 'frame'
                index = ClipIndex(run/'clip_manifest.json', inputs.data) if kind == 'clip' else DevSchedule(inputs.index, inputs.data)
                try:records = index.records('dev', kind)
                finally:index.close()
                saved = checkpoint_dev_report(state, records, POLICY, kind)
                if saved is None:
                    metrics = run/'best_dev_metrics.json'
                    if not metrics.is_file():raise ValueError('Saved DEV predictions/metrics are unavailable; explicitly request --fresh-evaluation with CUDA. No automatic inference was started.')
                    report = json.loads(metrics.read_text(encoding='utf-8'))
                else:_, report = saved
                del state
                print(key+': reused selected-best DEV predictions/metrics; zero inference.')
            print(key+': '+(str(report) if isinstance(report, Path) else json.dumps({k:report[k] for k in ('unit','video','losses')})))
            continue
        state = validate_clip_checkpoint(run/'best.pt', root, index_hash=digest(inputs.index), pool_hash=digest(inputs.pool))
        if state['epoch']+1 != state['best_epoch']:raise ValueError('Use the selected best CLIP checkpoint')
        if not fresh:
            print(key+': reused saved best-checkpoint DEV metrics; zero inference.', json.dumps({k:state['report'][k] for k in ('unit','video','losses')}))
            continue
        import torch
        index = DevSchedule(inputs.index, inputs.data)
        try:records = index.records('dev', 'frame')
        finally:index.close()
        model, processor = checkpoint_components(state);model.cuda().eval()
        cfg = CLIPConfig(amp=state['config']['amp'])
        loader = make_clip_loader(records, inputs.data, processor, cfg)
        metrics, predictions, _ = clip_epoch(model, loader, cfg)
        report = evaluate_predictions(records, predictions, POLICY, 'frame')
        report['losses'] = {k:metrics[k] for k in ('loss','balanced_loss','real_class_loss','fake_class_loss')}
        destination = root/'results'/key/state['experiment_identity']['run_id']/('dev_evaluation_'+new_id());destination.mkdir(parents=True)
        prediction_artifacts(destination, records, predictions, report)
        write_json(destination/'evaluation_provenance.json', {'checkpoint_sha256': digest(run/'best.pt'), 'split': 'dev', 'fresh_inference': True})
        del model, loader, state;torch.cuda.empty_cache();print('Full CLIP DEV evaluation:', destination)
    return runs


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        settings = load_settings(args)
        if args.list_models:
            from .config import declared_configurations
            from .clip_model import CLIPConfig
            print(json.dumps(declared_configurations() | {CLIP_ID: CLIPConfig().as_dict()}, indent=2));return 0
        mode = settings['mode']
        device_report(require_gpu=mode == 'train' or args.fresh_evaluation or args.gpu_smoke)
        if mode == 'check':
            run_tests(settings)
            if settings['index_path'].is_file() and settings['train_pool'].is_file():
                inputs, _ = prepare_metadata(settings)
            else:
                inputs = None;print('Code tests passed. External artifacts/data were NOT validated; provide paths or run explicit metadata preparation.')
            if args.gpu_smoke:
                if inputs is None:raise FileNotFoundError('Real GPU smoke needs external metadata, videos and weights')
                root = workspace(settings);models, encoder = ordered_models(settings, root)
                inputs = assets(settings, inputs, root, models);cache = prepare_rgb(settings, inputs, root)
                from .workflow import preflight
                from .clip_workflow import preflight_clip
                for key in models:
                    if key == CLIP_ID:
                        from .clip_model import CLIPConfig
                        preflight_clip(inputs, root, settings['clip_weights'], cache=cache, config=CLIPConfig(amp=settings['amp']))
                    else:preflight(key, inputs, root, cnn_best=encoder, amp=settings['amp'], workers=settings['workers'])
                print('GPU smoke completed; zero optimizer updates. Full training was not run.')
            return 0
        root = workspace(settings)
        if mode == 'results':display_results(settings, root);return 0
        if mode == 'restore':
            if args.restore_zip is None:raise ValueError('--mode restore requires --restore-zip')
            from .backup import restore_run
            from .clip_workflow import restore_clip
            if len(settings['models']) != 1:raise ValueError('Select exactly one model when restoring')
            key = settings['models'][0];run = (restore_clip if key == CLIP_ID else restore_run)(args.restore_zip, root)
            identity = json.loads((run/'experiment_identity.json').read_text())
            if identity['model_id'] != key:raise ValueError('Restored archive model differs from --models; original archive preserved')
            print('Restored NEW run:', run);return 0
        inputs, report = prepare_metadata(settings, build=args.build_metadata)
        if mode == 'prepare':
            inputs = assets(settings, inputs, root, settings['models'], download=args.download_assets)
            if args.prepare_clips:
                from .session_inputs import _clip_value_from_index
                from .temporal import ClipIndex
                target = settings['clip_manifest'];expected = _clip_value_from_index(inputs)
                if target.exists():
                    if json.loads(target.read_text()) != expected:raise ValueError('Existing clip manifest differs; preserved')
                else:
                    _outside(target);target.parent.mkdir(parents=True, exist_ok=True)
                    from .data import write_json
                    partial = target.with_suffix('.partial');write_json(partial, expected);ClipIndex(partial, inputs.data).close();partial.replace(target)
                print('Original clip manifest:', target)
            if args.prepare_frames:prepare_rgb(settings, inputs, root)
            print('Preparation completed; no training.');return 0
        if mode == 'train':
            tests = run_tests(settings);runs = train_models(settings, root, inputs, report, tests)
            display_results(settings, root, runs);return 0
        if mode == 'evaluate':evaluate_models(settings, root, inputs, args.fresh_evaluation);return 0
        if mode == 'gradcam':
            from .clip_attribution import explain_clip_dev_gallery
            from .clip_reporting import explain_clip_dev
            from .reporting import explain_dev
            inputs = assets(settings, inputs, root, settings['models'])
            for key, run in selected_runs(settings, root).items():
                if key == CLIP_ID:
                    print('Original four DEV examples:', explain_clip_dev(run, inputs, root))
                    print('Representative DEV gallery:', explain_clip_dev_gallery(run, inputs, root, per_category=settings['gradcam_per_category']))
                elif key == 'resnet50_lstm':print('LSTM Grad-CAM is not supported; no substitute model selected.')
                else:
                    for category in ('tp','tn','fp','fn'):
                        print(explain_dev(run, inputs, root, category=category, count=settings['gradcam_per_category']))
            return 0
        raise ValueError('Unhandled mode')
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print('ERROR: '+str(error), file=sys.stderr, flush=True);return 2
