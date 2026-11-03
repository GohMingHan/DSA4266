"""Inputs, preflight, run identities and explicitly requested sequential orchestration."""

from dataclasses import dataclass, replace
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import gc
import json
import shutil
import uuid
import hashlib
import time
import torch
from .checkpoints import digest, load_resume_bundle
from .config import selection_label, RECIPES, ORDER, INDEX_SHA256, POOL_SHA256, configuration
from .data import (
    write_json,
    make_loader,
    LazyDataset,
    FixedPool,
    FixedPoolSampler,
    DevSchedule,
    PoolConfig,
    build_pool,
)
from .models import StagedCNN
from .build_candidates import validate_candidate_index, build_candidates
from .temporal import ClipIndex, build_clip_manifest, trained_resnet, preflight_temporal
from .training import train, GradientGuard, seed_everything, require_cuda
from .face_mask import FaceMaskPipeline, ensure_landmarker, ensure_landmark_runtime, ensure_detector


def new_id():return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC_')+uuid.uuid4().hex[:8]

@dataclass(frozen=True)
class Inputs:
    data: Path
    index: Path
    pool: Path | None
    weights: Path
    clips: Path | None=None
    detector: Path | None=None
    landmarker: Path | None=None

def project_id(project):return json.loads((Path(project)/'configs/project.json').read_text())['project_id']

def guard_dev(split):
    if split!='dev':raise PermissionError('Only DEV execution is available. TEST is locked and must not be inspected.')

def validate_inputs(inputs, *, source_check=True):
    """Read existing TRAIN/DEV metadata only. Never decode videos here."""
    identity=validate_candidate_index(inputs.index,inputs.data if source_check else None)
    dev=DevSchedule(inputs.index,inputs.data if source_check else None)
    try:report=dict(dev.report)
    finally:dev.close()
    if inputs.pool is None:raise FileNotFoundError('Attach original train_pool.json or explicitly prepare manifests')
    if digest(inputs.pool)!=POOL_SHA256:raise ValueError('Wrong original fixed20k TRAIN pool hash')
    pool=FixedPool(inputs.pool,inputs.data if source_check else None)
    report['train_pool']=pool.report();report.update(identity);report['pool_sha256']=POOL_SHA256
    a=FixedPoolSampler(pool,4000,4266);b=FixedPoolSampler(pool,4000,4266);sampling=[]
    for e in range(10):
        ids,r=a.select(e);other,_=b.select(e)
        if ids!=other or len(set(ids))!=8000 or r['real']!=4000 or r['fake']!=4000:raise AssertionError('Sampling changed')
        sampling.append({k:r[k] for k in ['epoch','unique_samples','real','fake','duplicate_rate','cumulative_coverage']})
    report['disposable_sampling_checks']=sampling
    return report

def prepare_manifests(inputs, project, *, artifact_mode='REUSE', build_missing=None, need_clips=False):
    """REUSE never builds; explicit BUILD constructs missing artifacts only.

    To regenerate an index choose a new, absent INDEX_PATH in working storage.
    Existing invalid or partial artifacts are refused rather than overwritten.
    build_missing remains a compatibility alias for earlier local callers.
    """
    if artifact_mode not in ('REUSE','BUILD'):raise ValueError('ARTIFACT_MODE must be REUSE or BUILD')
    if build_missing is not None:
        if type(build_missing) is not bool:raise ValueError('build_missing must be boolean')
        if artifact_mode!='REUSE':raise ValueError('Use artifact_mode or build_missing, not both')
        artifact_mode='BUILD' if build_missing else 'REUSE'
    build=artifact_mode=='BUILD'
    index=Path(inputs.index)
    if not index.is_file():
        if not build:raise FileNotFoundError('Candidate index missing. Use the repository metadata/candidates.sqlite, or explicitly select ARTIFACT_MODE=BUILD and a new working-storage index path.')
        if index.name!='candidates.sqlite':raise ValueError('BUILD index filename must be candidates.sqlite')
        index=build_candidates(inputs.data,index.parent)
    validate_candidate_index(index,inputs.data)
    inputs=replace(inputs,index=index)
    cache=Path(project)/'.cache/manifests';pool=inputs.pool;clips=inputs.clips
    if pool is None or not Path(pool).is_file():
        existing=cache/'pool/train_pool.json'
        if existing.is_file():pool=existing
        elif build:pool=build_pool(inputs.data,existing.parent,PoolConfig())
        else:raise FileNotFoundError('Set TRAIN_POOL to the original manifest, or explicitly select ARTIFACT_MODE=BUILD to reproduce the fixed20k pool once from TRAIN videos. REUSE never builds.')
    result=replace(inputs,pool=Path(pool))
    validate_inputs(result)
    if need_clips and (clips is None or not Path(clips).is_file()):
        existing=cache/'clip_manifest.json'
        if existing.is_file():clips=existing
        elif build:clips=build_clip_manifest(inputs.data,inputs.index,existing)
        else:raise FileNotFoundError('Set CLIP_MANIFEST or explicitly build the unchanged clip manifest')
    if clips is not None:
        # A rebuilt database may have different page layout, but every schema/row
        # was already matched above. Accept the original manifest in that case.
        if json.loads(Path(clips).read_text())['dev_index_sha256'] not in (INDEX_SHA256,digest(index)):raise ValueError('Wrong clip DEV index')
        clip=ClipIndex(clips,inputs.data);clip.close()
    return replace(result,clips=None if clips is None else Path(clips))

def mask_pipeline(inputs,project, *, include_geometry_cache=False):
    ensure_landmark_runtime(install=False)
    if inputs.detector is None or inputs.landmarker is None:raise FileNotFoundError('Set pinned YuNet and MediaPipe asset paths in setup')
    p=FaceMaskPipeline(inputs.detector,inputs.landmarker,Path(project)/'.cache/face_mask/geometry.sqlite',INDEX_SHA256,POOL_SHA256)
    p.include_geometry_cache=include_geometry_cache
    return p

def setup_assets(inputs,project, *, download=False):
    """Explicit setup only. Training never fetches code or weights."""
    root=Path(project)/'.cache/assets'
    detector=Path(inputs.detector) if inputs.detector else root/'face_detection_yunet_2023mar.onnx'
    landmarker=Path(inputs.landmarker) if inputs.landmarker else root/'face_landmarker.task'
    ensure_detector(detector,download=download);ensure_landmarker(landmarker,download=download)
    return replace(inputs,detector=detector,landmarker=landmarker)

def checkpoint_identity(path,project,model_id=None):
    """Reject all historical checkpoints and unrelated/newer experimental branches."""
    state=torch.load(path,map_location='cpu',weights_only=True)
    identity=state.get('experiment_identity',{})
    if identity.get('project_id')!=project_id(project) or identity.get('model_id') not in RECIPES:
        raise ValueError('Checkpoint was not produced by this final_visual project; historical weights refused')
    if model_id is not None and identity['model_id']!=model_id:raise ValueError('Wrong model variant checkpoint')
    expected=configuration(identity['model_id'],amp=state['config']['amp'],workers=state['config']['workers'])
    if state['config']['checkpoint_selection']!=expected.checkpoint_selection:raise ValueError('Checkpoint selector differs')
    for name in ['model','head_lr','backbone_lr','dropout','warmup_epochs','weight_decay','training_pool','batch_size','seed']:
        if state['config'][name]!=getattr(expected,name):raise ValueError('Checkpoint recipe differs: '+name)
    from .checkpoints import assert_finite_state
    assert_finite_state(state['model'],'selected model')
    return identity,state

def new_encoder(path,project):
    if path is None:raise FileNotFoundError('Train NEW resnet50_baseline first or set NEW_RESNET_BEST to its best.pt')
    identity,state=checkpoint_identity(path,project,'resnet50_baseline')
    if Path(path).name!='best.pt' or state.get('input_pipeline') is not None or state['epoch']+1!=state['best_epoch']:
        raise ValueError('LSTM requires this project selected full-frame baseline best.pt')
    del state
    return Path(path)

def discover_runs(project):
    result={k:[] for k in ORDER}
    for k in ORDER:
        for p in sorted((Path(project)/'checkpoints'/k).glob('*/run')):
            if (p/'experiment_identity.json').is_file():result[k].append(p)
    return result

def choose_runs(project, explicit=None):
    result={};explicit=explicit or {}
    for k,paths in discover_runs(project).items():
        if k in explicit:result[k]=Path(explicit[k])
        elif len(paths)==1:result[k]=paths[0]
        elif len(paths)>1:raise ValueError('Multiple runs for '+k+'; choose a run explicitly instead of silently selecting one')
    return result

def preflight(model_id,inputs,project,*,cnn_best=None,amp=True,workers=None):
    """Bounded actual decoding/gradient checks; ZERO optimizer updates."""
    cfg=configuration(model_id,amp=amp,workers=workers);require_cuda();validate_inputs(inputs)
    folder=Path(project)/'results/diagnostics'/model_id/new_id();folder.mkdir(parents=True)
    try:
        if model_id=='resnet50_lstm':
            if inputs.clips is None:raise FileNotFoundError('Clip manifest required')
            return preflight_temporal(cfg,inputs.clips,inputs.data,new_encoder(cnn_best,project),folder/'smoke.json')
        pool=FixedPool(inputs.pool,inputs.data);ids,diag=FixedPoolSampler(pool,4000,cfg.seed).select(0)
        records=pool.records(ids[:cfg.batch_size]);pipeline=mask_pipeline(inputs,project) if RECIPES[model_id].input=='face_mask' else None
        print(f'{model_id}: GPU preflight loading {len(records)} TRAIN frames; zero optimizer updates.',flush=True)
        started=time.perf_counter()
        loader=(pipeline.make_loader if pipeline else make_loader)(records,inputs.data,cfg,training=True,epoch=0)
        images,targets,_=next(iter(loader));load_seconds=time.perf_counter()-started
        seed_everything(cfg.seed);model=StagedCNN(cfg,inputs.weights,pretrained=True).cuda();events=[];phases=[]
        from .models import make_optimizer
        optimizer=make_optimizer(model,cfg)
        for fine in [False,True]:
            model.set_phase(fine);model.zero_grad(set_to_none=True)
            scaler=torch.amp.GradScaler('cuda',enabled=cfg.amp);guard=GradientGuard(cfg);guard.begin_epoch(0)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            for attempt in range(3):
                model.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16,enabled=cfg.amp):
                    logits=model(images.cuda()).reshape(-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets.cuda())
                if not bool(torch.isfinite(logits).all() and torch.isfinite(loss)):raise FloatingPointError('Non-finite preflight logits/loss')
                scaler.scale(loss).backward();event=guard.finish_step(optimizer,scaler,epoch=0,batch=attempt+1,emit=events.append,execute_step=False)
                if not event['skipped_step']:break
                print(f'Preflight {model.phase}: AMP overflow safely skipped; reduced scale to {scaler.get_scale()}.',flush=True)
            else:raise FloatingPointError('Repeated preflight gradient overflow; inspect smoke diagnostics before training')
            torch.cuda.synchronize()
            phases.append({'phase':model.phase,'loss':float(loss.detach()),'forward_backward_seconds':time.perf_counter()-started,
                           'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                           'additional_adam_state_mib_after_real_updates':sum(p.numel()*8 for p in model.parameters() if p.requires_grad)/2**20})
        # DEV remains full in training; smoke checks only one bounded batch, no AUC claim.
        index=DevSchedule(inputs.index,inputs.data)
        try:dev_records=index.records('dev','frame')[:cfg.batch_size]
        finally:index.close()
        started=time.perf_counter()
        dev_loader=(pipeline.make_loader if pipeline else make_loader)(dev_records,inputs.data,cfg)
        dev_images,dev_targets,_=next(iter(dev_loader));dev_load_seconds=time.perf_counter()-started
        model.eval();model.zero_grad(set_to_none=True)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=cfg.amp):
            dev_logits=model(dev_images.cuda()).reshape(-1)
            dev_loss=torch.nn.functional.binary_cross_entropy_with_logits(dev_logits,dev_targets.cuda())
        if not bool(torch.isfinite(dev_logits).all() and torch.isfinite(dev_loss)):raise FloatingPointError('Non-finite DEV smoke output')
        if optimizer.state:raise AssertionError('Preflight must not initialize optimizer state or perform updates')
        free,total=torch.cuda.mem_get_info()
        report={'model_id':model_id,'optimizer_steps':0,'frames':len(images),'phases':['head','final_stage'],
                'phase_measurements':phases,'train_load_seconds':load_seconds,'dev_frames':len(dev_images),
                'dev_load_seconds':dev_load_seconds,'dev_batch_loss':float(dev_loss),'full_dev_inference_performed':False,
                'gpu':torch.cuda.get_device_name(0),'gpu_total_mib':total/2**20,'gpu_free_mib_after_smoke':free/2**20,
                'memory_note':'Measured without optimizer updates; Adam moment buffers are estimated separately, not allocated.',
                'numerics':events,'workers':cfg.workers,'sampling':{k:diag[k] for k in ['samples','real','fake','duplicate_rate']}}
        write_json(folder/'smoke.json',report)
        print('GPU preflight:',json.dumps({k:report[k] for k in ['model_id','optimizer_steps','frames','dev_frames','phase_measurements','train_load_seconds','dev_load_seconds']}),flush=True)
        del model,optimizer,scaler,loader,dev_loader,images,targets,loss,logits,dev_images,dev_targets,dev_logits,dev_loss
        gc.collect();torch.cuda.empty_cache()
        return report
    except Exception as exc:
        write_json(folder/'failure.json',{'type':type(exc).__name__,'message':str(exc),
            'model_id':model_id,'batch_size':cfg.batch_size,'amp':cfg.amp,'optimizer_steps':0,
            'numerics':locals().get('events',[]),'phase_measurements':locals().get('phases',[])});raise

def resume_candidate(model_id,inputs,project,*,source=None,amp=True,workers=None):
    cfg=configuration(model_id,amp=amp,workers=workers)
    paths=[Path(source)] if source else sorted((Path(project)/'checkpoints'/model_id).glob('*/run/last.pt'),key=lambda p:p.stat().st_mtime,reverse=True)
    errors=[]
    for p in paths:
        try:
            checkpoint_identity(p,project,model_id)
            from .temporal import TEMPORAL_PROTOCOL
            from .data import PROTOCOL
            temporal=model_id=='resnet50_lstm'
            index=inputs.clips if temporal else inputs.index
            b=load_resume_bundle(p,cfg,digest(index),pool_hash=None if temporal else POOL_SHA256,protocol=TEMPORAL_PROTOCOL if temporal else PROTOCOL)
            checkpoint_identity(p.parent/'best.pt',project,model_id)
            return p,b['report']
        except (ValueError,FileNotFoundError,RuntimeError) as e:errors.append(str(p)+': '+str(e))
    raise ValueError('No valid resumable completed boundary. Completed/early-stopped runs are evaluation-only. '+str(errors))

def execute_training(selected,inputs,project,*,confirmation='',resume_from=None,cnn_best=None,amp=True,workers=None,include_geometry_cache=False):
    if confirmation!='TRAIN_SELECTED_MODELS':raise PermissionError('Explicit TRAIN_SELECTED_MODELS confirmation required; setup never trains')
    selected=list(selected)
    if not selected or len(selected)!=len(set(selected)) or any(k not in RECIPES for k in selected):raise ValueError('Select unique known model IDs')
    # Explicit list is sequential; for all-five ensure the new encoder is trained first.
    if 'resnet50_lstm' in selected and 'resnet50_baseline' in selected and selected.index('resnet50_lstm')<selected.index('resnet50_baseline'):
        raise ValueError('New ResNet50 baseline must precede its LSTM')
    outputs={};resume_from=resume_from or {};project=Path(project)
    for key in selected:
        cfg=configuration(key,amp=amp,workers=workers);r=RECIPES[key]
        source=None
        if key in resume_from:source,_=resume_candidate(key,inputs,project,source=None if str(resume_from[key])=='latest' else resume_from[key],amp=amp,workers=workers)
        encoder=new_encoder(cnn_best,project) if key=='resnet50_lstm' else None
        preflight(key,inputs,project,cnn_best=encoder,amp=amp,workers=workers)
        runid=new_id();parent=project/'checkpoints'/key/runid;parent.mkdir(parents=True,exist_ok=False)
        identity={'project_id':project_id(project),'model_id':key,'label':r.label,'input':r.input,'run_id':runid,'checkpoint_selection':cfg.checkpoint_selection,'new_run':True,'test_used':False,'resume_source':str(source) if source else None}
        write_json(parent/'experiment_identity.json',identity)
        provenance=json.loads((project/'docs/source_provenance.json').read_text())
        provenance['standalone_source_hashes']={str(p.relative_to(project)):digest(p) for p in sorted((project/'src').rglob('*.py'))}
        write_json(parent/'source_provenance.json',provenance)
        plan={'config':cfg.as_dict(),'input':r.input,'checkpoint_monitor':cfg.checkpoint_selection,'scheduler_monitor':selection_label(cfg.checkpoint_selection),'cross_model_selection':'DEV video PR-AUC','historical_results_imported':False}
        write_json(parent/'experiment_plans.json',{cfg.model:plan})
        pipeline=mask_pipeline(inputs,project,include_geometry_cache=include_geometry_cache) if r.input=='face_mask' else None
        out=train(cfg,inputs.clips if key=='resnet50_lstm' else inputs.index,inputs.data,encoder if encoder else inputs.weights,parent/'run',source,pool_path=None if encoder else inputs.pool,input_pipeline=pipeline)
        outputs[key]=out
        from .reporting import export_report
        export_report(out,project)
        if key=='resnet50_baseline':cnn_best=out/'best.pt'
    return outputs
