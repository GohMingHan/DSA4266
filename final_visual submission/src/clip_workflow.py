"""Independent sixth-model lifecycle using the original CNN manifests/metrics.

No scheduler/warm-up: the groupmate's lower-encoder-LR recipe has neither.
Only this module trains CLIP; existing five-model execution stays unchanged.
"""
from dataclasses import replace
from pathlib import Path
import gc
import json
import math
import shutil
import time
import uuid
import zipfile
import numpy as np
import torch
from .clip_model import (MODEL_ID,CLIP_SEED,CLIPConfig,CLIPBinaryClassifier,classification_loss,
                         fake_probability,load_processor,prepare_clip_assets,make_clip_loader,checkpoint_components)
from .data import cnn_inputs,write_json,FixedPoolSampler
from .config import INDEX_LOGICAL_SHA256,POOL_SHA256
from .checkpoints import atomic_checkpoint,digest,rng_state,restore_rng,assert_finite_state
from .evaluation import (binary_metrics,class_loss_metrics,evaluate_predictions,write_history_metrics,
                         prediction_artifacts,plot_history,write_final_numerics)
from .training import RunProgress,require_cuda,should_log_batch,NumericalInstabilityError
from .workflow import new_id,project_id
from .prepared_frames import require_prepared_cache,verify_clip_cache_loader

PROTOCOL='clip_shared_fixed20k_pts10_v1'
POLICY={'aggregation':'topk','topk_fraction':.05,'prediction_threshold':.5,'video_threshold':.5}


def clip_seed(seed=CLIP_SEED):
    # One seed for CLIP initialization/RNGs and the unchanged shared data pipeline.
    import random
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def clip_seed_policy(config):
    return {'head_initialization':config.seed,'python':config.seed,'numpy':config.seed,
            'torch_cpu':config.seed,'torch_cuda':config.seed,'sampling':config.sampling_seed,
            'dataloader_base':config.sampling_seed,'augmentation_base':config.sampling_seed,
            'dataloader_derivation':'stable_seed(base, epoch, training); PyTorch derives worker seeds',
            'augmentation_derivation':'stable_seed(base, epoch, kind, candidate_id, flip)'}


def optimizer_for(model,config):
    return torch.optim.AdamW([{'name':'final_stage','params':model.encoder.parameters(),'lr':config.backbone_lr},
                             {'name':'head','params':model.classifier.parameters(),'lr':config.head_lr}],weight_decay=config.weight_decay)


def preflight_clip(inputs,project,weights,*,config=CLIPConfig(),cache=None,resume_from=None):
    """Bounded8 TRAIN+8 DEV forward-only GPU test, using exact existing records."""
    from .notebook import _TRAINING_LOCK
    from .config import Config
    config.validate();require_prepared_cache(cache,inputs);require_cuda()
    if not _TRAINING_LOCK.acquire(blocking=False):raise RuntimeError('Wait for the active model before a GPU preflight')
    index=None
    try:
        clip_seed(config.seed);index,sampler,pool=cnn_inputs(Config(),inputs.index,inputs.data,inputs.pool)
        ids,report=sampler.select(0)
        if resume_from:
            state=validate_clip_checkpoint(resume_from,project,config=config,index_hash=digest(inputs.index),pool_hash=pool.sha256)
            model,processor=checkpoint_components(state,weights)
        else:
            weights=prepare_clip_assets(weights);model=CLIPBinaryClassifier(weights);processor=load_processor(weights)
        model=model.cuda().eval()
        result={'optimizer_steps':0,'fresh_pretrained':resume_from is None,'model_seed':config.seed,
                'sampling_seed':config.sampling_seed,'seed_policy':clip_seed_policy(config),
                'amp':config.amp,'index_sha256':digest(inputs.index),'pool_sha256':pool.sha256,'splits':{}}
        for split,rows in [('train',pool.records(ids[:8])),('dev',index.records('dev','frame')[:8])]:
            loader=make_clip_loader(rows,inputs.data,processor,config,training=split=='train',cache=cache)
            verify_clip_cache_loader(loader)
            images,targets,_=next(iter(loader))
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=config.amp):
                logits=model(images.cuda());losses=classification_loss(logits,targets.cuda())
            if not bool(torch.isfinite(losses).all()):raise FloatingPointError('CLIP preflight loss non-finite')
            result['splits'][split]={'frames':len(rows),'input_shape':list(images.shape),'ce_loss':float(losses.float().mean()),'finite':True}
        result['cuda_allocated_mib']=torch.cuda.memory_allocated()/2**20
        out=Path(project)/'results'/MODEL_ID/'preflight'/new_id();out.mkdir(parents=True);write_json(out/'report.json',result)
        del model;gc.collect();torch.cuda.empty_cache()
        return result
    finally:
        if index:index.close()
        _TRAINING_LOCK.release()


class CLIPGradientGuard:
    def __init__(self,config,state=None):
        self.config=config;self.state=dict(state or {'optimizer_steps':0,'skipped_steps':0,'consecutive_skips':0})

    def step(self,optimizer,scaler,*,emit,epoch,batch):
        before=float(scaler.get_scale());scaler.unscale_(optimizer)
        parameters=[p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        if not parameters:raise NumericalInstabilityError('No CLIP gradients')
        norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach(),dtype=torch.float64) for p in parameters]))
        finite=bool(torch.isfinite(norm));event={'epoch':epoch+1,'batch':batch,'gradient_norm':float(norm) if finite else None,
            'amp_scale_before':before,'clip_max_norm':1.,'skipped_step':not finite}
        if not finite:
            self.state['skipped_steps']+=1;self.state['consecutive_skips']+=1
            def forbid(*args,**kwargs):raise NumericalInstabilityError('Non-finite CLIP optimizer update blocked')
            hook=optimizer.register_step_pre_hook(forbid)
            try:
                if scaler.is_enabled():scaler.step(optimizer);scaler.update()
            finally:hook.remove()
            optimizer.zero_grad(set_to_none=True)
        else:
            coefficient=min(1.,self.config.gradient_clip/(float(norm)+1e-6))
            for p in parameters:p.grad.mul_(coefficient)
            if not all(bool(torch.isfinite(p.grad).all()) for p in parameters):raise NumericalInstabilityError('Non-finite clipped CLIP gradients')
            scaler.step(optimizer);scaler.update();self.state['optimizer_steps']+=1;self.state['consecutive_skips']=0
        event.update(amp_scale_after=float(scaler.get_scale()),**self.state);emit(event)
        if not finite:
            if not scaler.is_enabled() or not 0<event['amp_scale_after']<before:raise NumericalInstabilityError('CLIP non-finite FP32 gradients or invalid AMP backoff; no update')
            if self.state['consecutive_skips']>=self.config.max_consecutive_gradient_skips:raise NumericalInstabilityError('Repeated CLIP AMP instability; resume completed epoch with amp=False')
        return event


def clip_epoch(model,loader,config,*,optimizer=None,scaler=None,guard=None,progress=None,epoch=0,device='cuda'):
    training=optimizer is not None;model.train(training);started=time.perf_counter();labels=[];scores=[];predictions=[]
    sums={0:0.,1:0.};counts={0:0,1:0};norms=[];events=[];load_total=compute_total=0.
    start_steps=dict(guard.state) if training else {}
    with torch.enable_grad() if training else torch.no_grad():
        if progress:progress.stage('clip_iterator_start',epoch=epoch+1,batch=0,total_batches=len(loader))
        iterator=iter(loader)
        for batch in range(1,len(loader)+1):
            if progress:progress.stage('clip_train_fetch_batch' if training else 'clip_dev_fetch_batch',epoch=epoch+1,batch=batch,total_batches=len(loader),completed_batches=batch-1)
            fetch_start=time.perf_counter();images,targets,indices=next(iterator)
            load=time.perf_counter()-fetch_start;load_total+=load;compute_start=time.perf_counter()
            if progress:progress.stage('clip_train_compute' if training else 'clip_dev_compute',persist=False,batch=batch,load_seconds=load)
            images=images.to(device,non_blocking=True);targets=targets.to(device,non_blocking=True)
            if training:optimizer.zero_grad(set_to_none=True)
            with torch.autocast(torch.device(device).type,dtype=torch.float16,enabled=config.amp):
                logits=model(images);losses=classification_loss(logits,targets)
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(losses).all()):raise FloatingPointError('CLIP non-finite forward/CE; no optimizer update')
            if training:
                scaler.scale(losses.mean()).backward()
                def emit(event):
                    events.append(event)
                    if event['gradient_norm'] is not None:norms.append(event['gradient_norm'])
                    if progress:progress.numerics(event)
                event=guard.step(optimizer,scaler,emit=emit,epoch=epoch,batch=batch)
                if guard.state['skipped_steps']-start_steps['skipped_steps']>=config.max_gradient_skips_per_epoch:raise NumericalInstabilityError('Too many CLIP AMP overflows this epoch')
                if event['skipped_step'] and progress:progress.note('CLIP AMP overflow safely skipped: '+json.dumps(event))
            ys=targets.detach().int().cpu().tolist();ps=fake_probability(logits.detach()).cpu().tolist();ls=losses.detach().float().cpu().tolist()
            for i,y,p,loss in zip(indices.tolist(),ys,ps,ls):
                labels.append(y);scores.append(p);sums[y]+=loss;counts[y]+=1
                if not training:predictions.append({'record_index':i,'target':y,'probability':p})
            if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
            compute=time.perf_counter()-compute_start;compute_total+=compute
            timing={'epoch':epoch+1,'split':'train' if training else 'dev','batch':batch,'total_batches':len(loader),
                    'loading_seconds':load,'computation_seconds':compute,'batch_seconds':load+compute}
            if progress:
                progress.stage('clip_batch_completed',persist=False,batch=batch,completed_batches=batch,load_seconds=load,compute_seconds=compute)
                if hasattr(progress,'output'):
                    with (progress.output/'batch_timings.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(timing)+'\n')
            interval=config.train_progress_every if training else config.dev_progress_every
            if should_log_batch(batch,len(loader),interval):
                text=f'CLIP | Epoch {epoch+1}/{config.epochs} | {"TRAIN" if training else "DEV"} Batch {batch}/{len(loader)} | CE {sum(ls)/len(ls):.4f} | Load {load:.2f}s | Compute {compute:.2f}s | Batch {load+compute:.2f}s'
                if training:text+=f' | AMP {scaler.get_scale():g} | skipped {guard.state["skipped_steps"]}'
                if torch.device(device).type=='cuda':text+=f' | GPU allocated/reserved {torch.cuda.memory_allocated()/2**20:.0f}/{torch.cuda.memory_reserved()/2**20:.0f} MiB'
                if progress:progress.note(text)
        del iterator
    if len(labels)!=len(loader.dataset):raise RuntimeError('Incomplete CLIP epoch; checkpoints not advanced')
    result=binary_metrics(labels,scores,.5);result.update(class_loss_metrics(sums,counts));result['loss']=sum(sums.values())/len(labels)
    result['loading_seconds']=load_total;result['computation_seconds']=compute_total
    if training:result['numerics']={'optimizer_steps':guard.state['optimizer_steps']-start_steps['optimizer_steps'],'skipped_steps':guard.state['skipped_steps']-start_steps['skipped_steps'],
        'gradient_norm_mean_finite':sum(norms)/len(norms) if norms else None,'gradient_norm_max_finite':max(norms) if norms else None,
        'amp_scale_start':events[0]['amp_scale_before'],'amp_scale_end':events[-1]['amp_scale_after']}
    return result,predictions,time.perf_counter()-started


def validate_clip_checkpoint(path,project,*,config=None,index_hash=None,pool_hash=None,assets=None):
    state=torch.load(path,map_location='cpu',weights_only=True);identity=state.get('experiment_identity',{})
    if state.get('protocol')!=PROTOCOL or identity.get('project_id')!=project_id(project) or identity.get('model_id')!=MODEL_ID:raise ValueError('Foreign/historical CLIP checkpoint refused')
    saved=dict(state['config']);saved.pop('amp',None);saved.pop('workers',None)
    expected=(config or CLIPConfig()).as_dict();expected.pop('amp');expected.pop('workers')
    if saved!=expected:raise ValueError('CLIP resume recipe differs')
    if state.get('seed_policy',clip_seed_policy(config or CLIPConfig()))!=clip_seed_policy(config or CLIPConfig()):raise ValueError('CLIP checkpoint seed metadata differs')
    if index_hash and state['index_sha256']!=index_hash:raise ValueError('CLIP index changed')
    if pool_hash and state['pool_sha256']!=pool_hash:raise ValueError('CLIP pool changed')
    if assets and state['pretrained_manifest']!=assets:raise ValueError('CLIP pretrained assets changed')
    if [row['epoch'] for row in state['history']]!=list(range(1,state['epoch']+2)):raise ValueError('CLIP history boundary corrupt')
    if state['sampler']['epoch']!=state['epoch']:raise ValueError('CLIP sampler epoch differs from recovery boundary')
    selected=max(state['history'],key=lambda row:row['selection_score'])
    if (selected['epoch'],selected['selection_score'])!=(state['best_epoch'],state['best']):raise ValueError('CLIP best epoch/score differs from history')
    if state['stale']!=state['epoch']+1-state['best_epoch']:raise ValueError('CLIP early-stop state differs from history')
    predictions=state['dev_predictions']
    if sorted(p['record_index'] for p in predictions)!=list(range(state['report']['unit']['n'])):raise ValueError('CLIP saved DEV predictions incomplete')
    if state['scheduler'] is not None:raise ValueError('Original CLIP recipe has no scheduler')
    for name in ['model','optimizer','scaler']:assert_finite_state(state[name],'CLIP '+name)
    return state


def clip_backup(run,*,final=False):
    from .backup import REQUIRED,OPTIONAL,sha256
    run=Path(run);identity=json.loads((run/'experiment_identity.json').read_text())
    history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines() if line]
    epoch=history[-1]['epoch']
    files={name:run/name for name in REQUIRED+OPTIONAL+('clip_pretrained_manifest.json','rgb_cache_report.json','batch_timings.jsonl') if (run/name).is_file()}
    if any(name not in files for name in REQUIRED):raise FileNotFoundError('Incomplete CLIP recovery boundary')
    files.update({p.name:p for p in run.glob('sampling_epoch_*.json')})
    if any(p.is_symlink() for p in files.values()):raise ValueError('Symlink backup refused')
    folder=run.parent/'downloads';folder.mkdir(exist_ok=True)
    target=folder/(MODEL_ID+'_'+identity['run_id']+('_final_'+uuid.uuid4().hex[:8] if final else '_latest_completed_epoch')+'.zip')
    if target.exists():
        if final:raise FileExistsError('Existing final archive preserved')
        with zipfile.ZipFile(target) as z:
            previous=json.loads(z.read('experiment_identity.json'))
        if previous!=identity:raise FileExistsError('Existing rolling archive has another run identity; preserved')
    if shutil.disk_usage(folder).free<sum(p.stat().st_size for p in files.values())+64*2**20:raise OSError('Insufficient space for CLIP ZIP; checkpoints preserved')
    temporary=folder/('.partial_'+uuid.uuid4().hex+'.zip')
    try:
        manifest={'schema':'dsa4266-manual-zip-v1','model':MODEL_ID,'completed_epoch':epoch,'files':{name:digest(path) for name,path in files.items()}}
        with zipfile.ZipFile(temporary,'x',zipfile.ZIP_STORED) as z:
            for name,path in files.items():z.write(path,name)
            z.writestr('backup_manifest.json',json.dumps(manifest))
        with zipfile.ZipFile(temporary) as z:
            if z.testzip():raise OSError('CLIP ZIP CRC failed')
            import hashlib
            for name,expected in manifest['files'].items():
                h=hashlib.sha256()
                with z.open(name) as member:
                    for block in iter(lambda:member.read(2**20),b''):h.update(block)
                if h.hexdigest()!=expected:raise OSError('CLIP ZIP member hash failed; previous archive retained')
        temporary.replace(target)
    finally:temporary.unlink(missing_ok=True)
    return {'archive':str(target),'sha256':sha256(target),'completed_epoch':epoch}


def restore_clip(archive,project):
    from .backup import restore_local_zip
    with zipfile.ZipFile(archive) as z:
        identity=json.loads(z.read('experiment_identity.json'))
        if identity.get('model_id')!=MODEL_ID or identity.get('project_id')!=project_id(project):raise ValueError('Foreign CLIP archive')
    destination=Path(project)/'checkpoints'/MODEL_ID/('restored_'+new_id())/'run'
    restore_local_zip(archive,destination)
    last=validate_clip_checkpoint(destination/'last.pt',project);best=validate_clip_checkpoint(destination/'best.pt',project)
    if (best['epoch']+1,best['best'])!=(last['best_epoch'],last['best']):raise ValueError('Mixed CLIP checkpoint pair')
    return destination


def train_clip(inputs,project,weights,*,confirmation='',config=CLIPConfig(),test_result=None,data_report=None,cache=None,resume_from=None):
    from .notebook import _require_tests,_TRAINING_LOCK
    if confirmation!='TRAIN_'+MODEL_ID:raise PermissionError('Explicit TRAIN_clip_vit_b32_shared confirmation required')
    _require_tests(test_result)
    if not data_report or (data_report.get('pool_sha256'),data_report.get('index_logical_sha256'),data_report.get('dev_frames'),data_report.get('dev_videos'))!=(POOL_SHA256,INDEX_LOGICAL_SHA256,41746,500):raise RuntimeError('Shared full DEV/pool validation required')
    config.validate();require_prepared_cache(cache,inputs);require_cuda()
    if not _TRAINING_LOCK.acquire(blocking=False):raise RuntimeError('Another model cell is training')
    index=None
    try:
        project=Path(project);clip_seed(config.seed)
        # Keep indexing/pool validation and original sampler EXACT; all CLIP RNGs use4266.
        from .config import Config
        index,sampler,pool=cnn_inputs(Config(),inputs.index,inputs.data,inputs.pool)
        dev=index.records('dev','frame');index_hash=digest(inputs.index)
        state=best_state=None
        if resume_from:
            state=validate_clip_checkpoint(resume_from,project,config=config,index_hash=index_hash,pool_hash=pool.sha256)
            assets=state['pretrained_manifest']
            best_state=validate_clip_checkpoint(Path(resume_from).parent/'best.pt',project,config=config,index_hash=index_hash,pool_hash=pool.sha256,assets=assets)
            if (best_state['epoch']+1,best_state['best'])!=(state['best_epoch'],state['best']):raise ValueError('Mixed CLIP best/last boundary')
            model,processor=checkpoint_components(state,weights)
        else:
            weights=prepare_clip_assets(weights);assets=json.loads((weights/'clip_pretrained_manifest.json').read_text())
            model=CLIPBinaryClassifier(weights);processor=load_processor(weights)
        model=model.cuda();optimizer=optimizer_for(model,config);scaler=torch.amp.GradScaler('cuda',enabled=config.amp)
        guard=CLIPGradientGuard(config)
        encoder_config=json.loads(json.dumps(model.encoder.config.to_dict()))
        processor_config=json.loads(json.dumps(processor.to_dict()))
        history=[];start=0;best=-1.;stale=0;best_epoch=0
        if state:
            model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
            if config.amp and state['config']['amp']:scaler.load_state_dict(state['scaler'])
            sampler.load_state_dict(state['sampler']);restore_rng(state['rng']);guard=CLIPGradientGuard(config,state['numerics'])
            history=state['history'];start=state['epoch']+1;best=state['best'];stale=state['stale'];best_epoch=state['best_epoch']
        runid=new_id();out=project/'checkpoints'/MODEL_ID/runid/'run';out.mkdir(parents=True,exist_ok=False)
        identity={'project_id':project_id(project),'model_id':MODEL_ID,'label':'CLIP ViT-B/32 (Shared CNN Data Preprocessing)','input':'full_frame','run_id':runid,'checkpoint_selection':'frame_roc_auc','test_used':False}
        write_json(out/'experiment_identity.json',identity);write_json(out/'config.json',config.as_dict()|{'seed_policy':clip_seed_policy(config),'loss':'two-class cross entropy; Fake output0','scheduler_monitor':None,'index_sha256':index_hash,'pool_sha256':pool.sha256})
        write_json(out/'clip_pretrained_manifest.json',assets)
        write_json(out/'source_provenance.json',{'source_hashes':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},'seed_policy':clip_seed_policy(config),'pretrained':assets,'comparison':'video trapezoidal PR-AUC'})
        if state:
            # Copy originals into NEW run; retain new identity and all recovery state.
            for name,saved in [('last.pt',state),('best.pt',best_state)]:
                saved=dict(saved);saved['experiment_identity']=identity;atomic_checkpoint(out/name,saved)
            write_history_metrics(out,history)
        dev_loader=make_clip_loader(dev,inputs.data,processor,config,cache=cache)
        verify_clip_cache_loader(dev_loader)
        with RunProgress(out,'CLIP ViT-B/32') as progress:
            for epoch in range(start,config.epochs):
                if stale>=config.patience:break
                progress.stage('clip_epoch_setup',epoch=epoch+1,batch=0)
                ids,diagnostic=sampler.select(epoch);write_json(out/f'sampling_epoch_{epoch+1:03}.json',diagnostic|{'candidate_ids':ids})
                loader=make_clip_loader(pool.records(ids),inputs.data,processor,config,epoch=epoch,training=True,cache=cache)
                train,_,ts=clip_epoch(model,loader,config,optimizer=optimizer,scaler=scaler,guard=guard,progress=progress,epoch=epoch)
                del loader
                metrics,predictions,ds=clip_epoch(model,dev_loader,config,progress=progress,epoch=epoch)
                progress.stage('clip_aggregate_dev_metrics',batch=None)
                report=evaluate_predictions(dev,predictions,POLICY,'frame');report['losses']={k:metrics[k] for k in ['loss','balanced_loss','real_class_loss','fake_class_loss']}
                score=metrics['roc_auc']
                if score is None or not math.isfinite(score):raise ValueError('Undefined CLIP DEV ROC-AUC')
                improved=score>best
                if improved:best=score;best_epoch=epoch+1;stale=0
                else:stale+=1
                entry={'epoch':epoch+1,'train':train,'dev':metrics,'dev_video':report['video'],'train_seconds':ts,'dev_seconds':ds,'learning_rates':{g['name']:g['lr'] for g in optimizer.param_groups},'phase':'all_encoder_and_head','is_best':improved,'selection_score':score,'checkpoint_selection':'frame_roc_auc','stale_epochs':stale,'sampling':diagnostic}
                history.append(entry)
                payload={'protocol':PROTOCOL,'config':config.as_dict(),'seed_policy':clip_seed_policy(config),'encoder_config':encoder_config,'processor_config':processor_config,'experiment_identity':identity,'pretrained_manifest':assets,'index_sha256':index_hash,'pool_sha256':pool.sha256,
                         'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'scheduler':None,'rng':rng_state(),'sampler':sampler.state_dict(),
                         'epoch':epoch,'history':history,'best':best,'best_epoch':best_epoch,'stale':stale,'numerics':guard.state,'dev_predictions':predictions,'report':report}
                progress.stage('clip_save_history_checkpoints',batch=None)
                write_history_metrics(out,history)
                if improved:atomic_checkpoint(out/'best.pt',payload);prediction_artifacts(out,dev,predictions,report,render_plots=False)
                atomic_checkpoint(out/'last.pt',payload)
                progress.note(f'CLIP Epoch {epoch+1} complete | TRAIN CE {train["loss"]:.4f} | DEV CE {metrics["loss"]:.4f} | balanced {metrics["balanced_loss"]:.4f}\nFrame ROC/PR {score:.4f}/{metrics["pr_auc"]:.4f} | video ROC/PR {report["video"]["roc_auc"]:.4f}/{report["video"]["pr_auc"]:.4f}\nFrame P/R/F1/F2 '+str({k:metrics[k] for k in ['precision','recall','f1','f2']})+' | Video '+str({k:report['video'][k] for k in ['precision','recall','f1','f2']})+f'\nBest epoch {best_epoch}; stale {stale}/{config.patience}; LRs {entry["learning_rates"]}; duration {ts+ds:.1f}s\nCheckpoints: {out/"best.pt"}, {out/"last.pt"}')
                if cache:write_json(out/'rgb_cache_report.json',cache.report())
                progress.stage('clip_create_epoch_recovery_zip',batch=None)
                progress.note('Download completed-epoch ZIP: '+clip_backup(out)['archive'])
            progress.stage('clip_final_saved_prediction_reports',batch=None)
            selected=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
            prediction_artifacts(out,dev,selected['dev_predictions'],selected['report'])
            plot_history(out,history,best_epoch=best_epoch,selection_metric='frame_roc_auc');write_final_numerics(out,history)
            from .evaluation import save_calibration
            save_calibration(out,dev,selected['dev_predictions'],4266)
            write_json(out/'training_summary.json',{'status':'resumed' if resume_from else 'trained','model':MODEL_ID,'seed':config.seed,'sampling_seed':config.sampling_seed,'epochs_run':len(history),'best_epoch':best_epoch,'checkpoint_selection':'frame_roc_auc','best_selection_score':best,'threshold':.5,'test_used':False})
            from .reporting import export_epoch_report
            export_epoch_report(out,out,kind='frame',checkpoint_selection='frame_roc_auc',in_run=True)
            progress.stage('clip_create_final_recovery_zip',batch=None)
            progress.note('Final CLIP ZIP: '+clip_backup(out,final=True)['archive'])
        del model,optimizer,scaler,dev_loader
        gc.collect();torch.cuda.empty_cache()
        return out
    finally:
        if index:index.close()
        _TRAINING_LOCK.release()
