"""Verified source packaging, rolling/final recovery ZIPs and new-session restoration."""

import contextlib
import hashlib
import json
import shutil
import uuid
import zipfile
import secrets
from pathlib import Path, PurePosixPath
from .checkpoints import digest
from .config import RECIPES


REQUIRED = ('best.pt', 'last.pt', 'history.csv', 'history.jsonl', 'config.json')
OPTIONAL = ('experiment_identity.json', 'source_provenance.json', 'numerics.jsonl', 'progress.jsonl', 'run_status.json', 'runtime_environment.json',
            'resume_state.json', 'trainable_head.json', 'trainable_final_stage.json',
            'experiment_plan.json', 'training_summary.json', 'generalization_assessment.json',
            'best_dev_metrics.json', 'best_dev_predictions.csv', 'best_dev_curves.json',
            'best_dev_video_predictions.json', 'dev_threshold_analysis.json',
            'training_curves.png', 'numerical_curves.png', 'best_dev_confusion.png', 'best_dev_roc_pr.png',
            'train_pool.json','clip_manifest.json','feature_source.json','dev_f2_calibration.json','dev_f2_reporting_confusion.png',
            'input_pipeline.json','face_boxes.sqlite','face_detection_report.json','face_detector.onnx','face_detector_LICENSE.txt',
            'face_landmarker.task','video_dev_f2_calibration.json',
            'epoch_metrics.csv','epoch_metrics.json','epoch_metrics.html')


def create_download_test(session):
    """Tiny ZIP with a code to read AFTER downloading/extracting on the laptop."""
    folder = Path(session)/'downloads'; folder.mkdir(parents=True, exist_ok=True)
    archive = folder/('download_test_'+uuid.uuid4().hex[:8]+'.zip')
    code = secrets.token_hex(4).upper()
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        z.writestr('READ_ME.txt', 'DSA4266 download test\n\nDownload this ZIP to your Windows laptop.\n'
            'Right-click it, select Extract All, and open this file locally.\n'
            f'Enter this verification code in notebook section 11: {code}\n'
            'This is a dummy text file, not model weights.\n')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise OSError('Download-test ZIP failed CRC verification')
    return {'archive': str(archive), 'sha256': sha256(archive), 'bytes': archive.stat().st_size,
            'verification_sha256': hashlib.sha256(code.encode()).hexdigest()}


def confirm_download(session, report, code):
    """User attestation that the downloaded file opened; never infer a download."""
    if Path(report['archive']).resolve().parent != (Path(session)/'downloads').resolve():
        raise ValueError('Download confirmation belongs to a different session; create/download this session\'s test ZIP.')
    actual = hashlib.sha256(str(code).strip().upper().encode()).hexdigest()
    if not secrets.compare_digest(actual, report['verification_sha256']):
        raise ValueError('Code does not match. Download/extract the test ZIP on Windows and read READ_ME.txt.')
    if sha256(report['archive']) != report['sha256']:
        raise ValueError('The test ZIP changed; create and download a new test.')
    proof = {'confirmed': True, 'method': 'user-entered code from downloaded/extracted ZIP',
             'test_archive': report['archive'], 'sha256': report['sha256']}
    path = Path(session)/'manual_download_verified.json'
    path.write_text(json.dumps(proof, indent=2)+'\n', encoding='utf-8')
    return proof


def restore_local_zip(archive,destination):
    """Verify every hash and extract flat run files into a NEW directory only."""
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Restore destination exists; originals preserved')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or any('/' in name or '\\' in name or name in ('.','..') for name in names):
            raise ValueError('Expected a flat manual checkpoint archive')
        manifest=json.loads(z.read('backup_manifest.json'))
        if manifest.get('schema')!='dsa4266-manual-zip-v1' or set(names)!=set(manifest['files'])|{'backup_manifest.json'}:
            raise ValueError('Invalid manual backup manifest')
        if not set(REQUIRED)<=set(manifest['files']):raise ValueError('Incomplete checkpoint archive')
        for name,expected in manifest['files'].items():
            h=hashlib.sha256()
            with z.open(name) as member:
                for chunk in iter(lambda:member.read(2**20),b''):h.update(chunk)
            if h.hexdigest()!=expected:raise ValueError('Checkpoint archive hash mismatch')
        if z.testzip() is not None:raise ValueError('Checkpoint archive CRC mismatch')
        destination.mkdir(parents=True,exist_ok=False)
        for name in names:
            with z.open(name) as source,(destination/name).open('xb') as target:shutil.copyfileobj(source,target)
    return destination


def package_model_run(out, model, epoch, *, final=False, snapshot_lock=None):
    """Bounded rolling epoch ZIP; immutable final ZIP. No remote authentication.

    Flat members make best.pt/last.pt directly reusable after extraction. Only
    generated artifacts are eligible. Existing unrelated archives are preserved.
    """
    out = Path(out)
    if out.is_symlink() or not out.is_dir() or model not in ('resnet50', 'efficientnet_b0','resnet50_lstm'):
        raise ValueError('Expected a real CNN output directory')
    for name in REQUIRED:
        if not (out/name).is_file():
            raise FileNotFoundError(f'Cannot package incomplete run: {name}')
    files = {name: out/name for name in REQUIRED+OPTIONAL if (out/name).is_file()}
    files.update({p.name: p for p in out.glob('sampling_epoch_[0-9][0-9][0-9].json')})
    for name in ('experiment_plans.json', 'manual_download_verified.json'):
        if (out.parent/name).is_file():
            files['session_'+name] = out.parent/name
    if any(p.is_symlink() for p in files.values()):
        raise ValueError('Symlink artifacts cannot be packaged')
    destination = out.parent/'downloads'; destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < sum(p.stat().st_size for p in files.values())+64*2**20:
        raise OSError('Not enough free space to create a verified model ZIP; checkpoints remain untouched.')
    identity_path = out/'experiment_identity.json'
    run_id = json.loads(identity_path.read_text())['run_id'] if identity_path.is_file() else out.parent.name
    if not run_id or any(not (c.isalnum() or c in '_-') for c in run_id):
        raise ValueError('Unsafe run identifier for backup filename')
    prefix = model+'_'+run_id
    filename = (prefix+'_final_'+uuid.uuid4().hex[:8]+'.zip' if final else prefix+'_latest_completed_epoch.zip')
    archive = destination/filename; receipt_path = archive.with_suffix('.manifest.json')
    if archive.exists():
        if final or not receipt_path.is_file():
            raise FileExistsError('Existing archive preserved; it is not this run\'s rolling ZIP.')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('source_run') != str(out.resolve()) or receipt.get('model') != model:
            raise FileExistsError('Existing archive belongs to another run; preserved.')
    partial = destination/('.partial_'+uuid.uuid4().hex+'.zip')
    try:
        with (snapshot_lock or contextlib.nullcontext()), zipfile.ZipFile(partial, 'x', zipfile.ZIP_STORED) as z:
            for name, path in sorted(files.items()):
                z.write(path, name)
        hashes = {}
        with zipfile.ZipFile(partial) as z:
            for name in z.namelist():
                h = hashlib.sha256()
                with z.open(name) as member:
                    for block in iter(lambda: member.read(2**20), b''):
                        h.update(block)
                hashes[name] = h.hexdigest()
        manifest = {'schema': 'dsa4266-manual-zip-v1', 'model': model, 'completed_epoch': epoch,
                    'kind': 'final' if final else 'epoch', 'files': hashes}
        with zipfile.ZipFile(partial, 'a') as z:
            z.writestr('backup_manifest.json', json.dumps(manifest, indent=2)+'\n')
        with zipfile.ZipFile(partial) as z:
            if z.testzip() is not None:
                raise OSError('Model ZIP CRC verification failed; previous rolling ZIP retained.')
        receipt = {'archive': str(archive), 'bytes': partial.stat().st_size, 'sha256': sha256(partial),
            'source_run': str(out.resolve()), 'model': model, 'completed_epoch': epoch,
            'kind': manifest['kind'], 'external_download_required': True}
        partial.replace(archive)  # Only the known rolling artifact can be replaced.
        temp_receipt = receipt_path.with_suffix('.json.tmp')
        temp_receipt.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
        temp_receipt.replace(receipt_path)
    finally:
        partial.unlink(missing_ok=True)
    return receipt


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def _boundary(run,project):
    from .workflow import checkpoint_identity
    run=Path(run);identity,last=checkpoint_identity(run/'last.pt',project)
    _,best=checkpoint_identity(run/'best.pt',project,identity['model_id'])
    if (best['epoch']+1,best['best'],best.get('index_sha256'),best.get('pool_sha256'))!=(last['best_epoch'],last['best'],last.get('index_sha256'),last.get('pool_sha256')):
        raise ValueError('Best/last checkpoint boundary differs; do not back up or resume a mixed pair')
    if [r['epoch'] for r in last['history']]!=list(range(1,last['epoch']+2)):
        raise ValueError('Checkpoint/history completed epochs differ')
    return identity,last

def backup_run(run,project,*,final=False):
    run=Path(run);identity,last=_boundary(run,project)
    if final and not (run/'training_summary.json').is_file():raise ValueError('Partial run: use final=False for rolling recovery')
    return package_model_run(run,last['config']['model'],last['epoch']+1,final=final)

def restore_run(archive,project):
    """New project archives only; never imports historical results/checkpoints."""
    from .workflow import project_id, new_id
    with zipfile.ZipFile(archive) as z:
        if 'experiment_identity.json' not in z.namelist():raise ValueError('Not a final_visual archive')
        identity=json.loads(z.read('experiment_identity.json'))
        if identity.get('project_id')!=project_id(project) or identity.get('model_id') not in RECIPES:raise ValueError('Foreign/historical archive refused')
    destination=Path(project)/'checkpoints'/identity['model_id']/('restored_'+new_id())/'run'
    restore_local_zip(archive,destination)
    _boundary(destination,project)
    return destination

def build_source_archive(project,archive):
    from .workflow import project_id
    project=Path(project);archive=Path(archive)
    if archive.exists():raise FileExistsError('Existing source ZIP preserved')
    files=[]
    for base in ['src','tests']:
        files.extend(p for p in (project/base).rglob('*.py') if '__pycache__' not in p.parts)
    files.extend(project/name for name in ['src/YUNET_LICENSE.txt','configs/models.json','configs/project.json',
                 'tests/fixtures/equivalence.json','requirements.txt','README.md',
                 'run.py','settings.json','.gitignore','.gitattributes','THIRD_PARTY_NOTICES.txt','configs/source_manifest.json'])
    if any(not p.is_file() or p.is_symlink() for p in files):
        raise ValueError('Source package requires regular source/configuration files only')
    hashes={p.relative_to(project).as_posix():digest(p) for p in files}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(project).as_posix())
        z.writestr('source_bundle_manifest.json',json.dumps({'project_id':project_id(project),'files':hashes},indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise OSError('Source ZIP failed CRC')
        for name,h in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=h:raise OSError('Source ZIP hash mismatch')
    return {'archive':str(archive),'sha256':digest(archive),'files':len(hashes),'bytes':archive.stat().st_size}
