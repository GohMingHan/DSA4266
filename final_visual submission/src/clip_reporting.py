"""Saved sixth-model replay, unified comparison and DEV-only token Grad-CAM."""
from pathlib import Path
import csv
import json
import random
import numpy as np
import torch
from .clip_model import MODEL_ID,CLIP_SEED,CLIPConfig,CLIPViTGradCAM,checkpoint_components
from .clip_workflow import validate_clip_checkpoint,train_clip
from .data import DevSchedule,FrameReader,write_json
from .workflow import project_id,new_id,guard_dev
from .checkpoints import digest
from .reporting import display_completed_run,table_html,REPORT_COLUMNS


def find_clip_run(project,explicit=None):
    paths=[Path(explicit)] if explicit is not None else list((Path(project)/'checkpoints'/MODEL_ID).glob('*/run'))
    if explicit is not None and not (Path(explicit)/'experiment_identity.json').is_file():raise FileNotFoundError('CLIP run identity missing')
    paths=[p for p in paths if (p/'experiment_identity.json').is_file()]
    for path in paths:
        identity=json.loads((path/'experiment_identity.json').read_text())
        if identity.get('project_id')!=project_id(project) or identity.get('model_id')!=MODEL_ID:raise ValueError('Foreign CLIP run refused')
    if len(paths)>1:raise ValueError('Multiple CLIP runs; set CLIP_SAVED_RUN explicitly')
    return paths[0] if paths else None


def show_clip(run):
    if run is None:print('CLIP ViT-B/32: Pending');return
    if not (Path(run)/'training_summary.json').is_file():print('CLIP: Partial; use CLIP_RESUME_FROM explicitly');return
    display_completed_run(run,name='CLIP ViT-B/32 — shared CNN data',kind='frame')
    print('Checkpoint paths:',Path(run)/'best.pt',Path(run)/'last.pt')
    from IPython.display import display,FileLink
    for path in sorted((Path(run).parent/'downloads').glob('*.zip')):display(FileLink(str(path)))


def run_clip_cell(inputs,project,weights,*,enabled=False,confirmation='',execution_mode='inspect',
                  test_result=None,data_report=None,cache=None,resume_from=None,saved_run=None,amp=True,
                  preflight_report=None):
    run=find_clip_run(project,Path(resume_from).parent if resume_from else saved_run)
    if not enabled or execution_mode!='train':show_clip(run);return run
    if confirmation!='TRAIN_'+MODEL_ID:raise PermissionError('Explicit CLIP model confirmation required')
    if run is not None and resume_from is None:
        if (run/'training_summary.json').is_file():show_clip(run);return run
        raise ValueError('Partial CLIP run exists; set CLIP_RESUME_FROM to its last.pt')
    if (not preflight_report or preflight_report.get('optimizer_steps')!=0 or
        preflight_report.get('model_seed')!=CLIP_SEED or preflight_report.get('amp')!=amp or
        preflight_report.get('index_sha256')!=digest(inputs.index) or
        preflight_report.get('pool_sha256')!=digest(inputs.pool) or
        any(not preflight_report.get('splits',{}).get(split,{}).get('finite') for split in ('train','dev'))):
        raise RuntimeError('Run the bounded Model6 GPU preflight with these inputs/AMP settings before training.')
    run=train_clip(inputs,project,weights,confirmation=confirmation,config=CLIPConfig(amp=amp),test_result=test_result,data_report=data_report,cache=cache,resume_from=resume_from)
    show_clip(run);return run


def clip_comparison_row(project,run=None):
    cfg=CLIPConfig();row={key:None for key in REPORT_COLUMNS}
    row.update({'Model':'CLIP ViT-B/32 — shared CNN data','Input preprocessing':'full_frame; CLIP normalization',
        'Head/temporal LR':cfg.head_lr,'Backbone LR':cfg.backbone_lr,'Metric level':'Frame','Threshold':.5,
        'Checkpoint selection':'DEV frame ROC-AUC','Status':'Pending'})
    if run is None:return row
    run=Path(run);identity=json.loads((run/'experiment_identity.json').read_text())
    if identity.get('model_id')!=MODEL_ID or identity.get('project_id')!=project_id(project):raise ValueError('Foreign CLIP run refused')
    row.update({'Run ID':identity['run_id'],'Status':'Partial'})
    if not (run/'training_summary.json').is_file():return row
    summary=json.loads((run/'training_summary.json').read_text());metrics=json.loads((run/'best_dev_metrics.json').read_text())
    history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines() if line]
    entry=next(r for r in history if r['epoch']==summary['best_epoch'])
    row.update({'Best epoch':summary['best_epoch'],'TRAIN Loss':entry['train']['loss'],'DEV Loss':metrics['losses']['loss'],'Balanced DEV Loss':metrics['losses']['balanced_loss'],'Status':'Missing checkpoint'})
    for prefix,level in [('Frame/Clip','unit'),('Video','video')]:
        if metrics[level]['threshold']!=.5:raise ValueError('Comparison requires threshold0.5')
        for name,key in [('PR-AUC','pr_auc'),('ROC-AUC','roc_auc'),('Precision','precision'),('Recall','recall'),('F1','f1'),('F2','f2')]:row[prefix+' '+name]=metrics[level].get(key)
    if (run/'best.pt').is_file():
        state=validate_clip_checkpoint(run/'best.pt',project)
        if (state['epoch']+1,state['best_epoch'])!=(summary['best_epoch'],summary['best_epoch']):raise ValueError('CLIP selected epoch mismatch')
        expected={k:v for k,v in state['report'].items() if k!='videos'}
        if expected!=metrics:raise ValueError('Saved CLIP metrics do not match selected checkpoint')
        row['Status']='Completed, verified checkpoint'
    return row


def show_six_comparison(project,runs=None,*,clip_run=None):
    from IPython.utils.capture import capture_output
    from IPython.display import HTML,Image,display
    from .reporting import comparison
    # Reuse existing five-model verification; show a single final six-row table.
    with capture_output():rows,_,out=comparison(project,runs)
    rows.append(clip_comparison_row(project,find_clip_run(project,clip_run)))
    eligible=[r for r in rows if r['Status']=='Completed, verified checkpoint']
    eligible.sort(key=lambda r:(-r['Video PR-AUC'],-r['Video ROC-AUC'],r['Model']))
    ranking=[{'rank':i+1,'model':r['Model'],'run_id':r['Run ID'],'video_pr_auc':r['Video PR-AUC'],'video_roc_auc':r['Video ROC-AUC']} for i,r in enumerate(eligible)]
    with (out/'six_model_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out/'six_model_comparison.json',{'rows':rows,'ranking':ranking,'primary_metric':'DEV video trapezoidal PR-AUC','tie_breaker':'DEV video ROC-AUC, then model name','threshold':.5,'test_used':False})
    html=table_html(rows,title='Six visual models — saved selected checkpoints, threshold0.5')
    (out/'six_model_comparison.html').write_text(html,encoding='utf-8');display(HTML(html))
    if eligible:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(19,5))
        for ax,fields,title in zip(axes,[['Video PR-AUC','Video ROC-AUC'],['Video F1','Video F2'],['DEV Loss','Balanced DEV Loss']],['Video AUC','Video at threshold0.5','Full DEV loss']):
            for offset,field in enumerate(fields):
                valid=[(i,r[field]) for i,r in enumerate(eligible) if r.get(field) is not None]
                ax.bar([i+(offset-.5)*.35 for i,v in valid],[v for i,v in valid],width=.35,label=field)
            ax.set_xticks(range(len(eligible)),[r['Model'] for r in eligible],rotation=35,ha='right');ax.set_ylim(bottom=0);ax.set_title(title);ax.legend()
            if title!='Full DEV loss':ax.set_ylim(top=1)
        fig.tight_layout();fig.savefig(out/'six_model_comparison.png',dpi=150);plt.close(fig);display(Image(filename=str(out/'six_model_comparison.png')))
    print('Ranked completed verified checkpoints:',ranking or 'Pending')
    return rows,ranking,out


def explain_clip_dev(run,inputs,project,weights=None,*,split='dev',seed=CLIP_SEED,cache=None,device=None):
    guard_dev(split)  # Gate before reading checkpoint/index or decoding pixels.
    run=Path(run)
    if not (run/'training_summary.json').is_file():raise ValueError('Complete new shared-data CLIP training first')
    state=validate_clip_checkpoint(run/'best.pt',project,index_hash=digest(inputs.index),pool_hash=digest(inputs.pool))
    index=DevSchedule(inputs.index,inputs.data)
    try:records=index.records('dev','frame')
    finally:index.close()
    predictions=state['dev_predictions']
    if sorted(p['record_index'] for p in predictions)!=list(range(len(records))):raise ValueError('Incomplete selected DEV predictions')
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model,processor=checkpoint_components(state,weights)
    model=model.to(device).eval();cam=CLIPViTGradCAM(model)
    if cache:
        from .frame_cache import CachedReader
        reader=CachedReader(inputs.data,cache)
    else:reader=FrameReader(inputs.data,0)
    groups={k:[] for k in ['tp','tn','fp','fn']}
    for p in predictions:
        row=records[p['record_index']]
        if p['target']!=row['label']:raise ValueError('Saved DEV target differs from indexed label')
        category=('tp' if p['probability']>=.5 else 'fn') if row['label'] else ('fp' if p['probability']>=.5 else 'tn')
        groups[category].append(p)
    destination=Path(project)/'results'/MODEL_ID/'gradcam_dev'/new_id();destination.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    saved=[]
    for category,items in groups.items():
        if not items:print('No DEV '+category.upper()+' examples at threshold0.5');continue
        p=random.Random(seed).choice(sorted(items,key=lambda p:records[p['record_index']]['id']))
        row=records[p['record_index']];rgb=reader.read([row])[0][0]
        pixels=processor(images=[Image.fromarray(rgb)],return_tensors='pt')['pixel_values'].to(device)
        maps,probabilities=cam.generate(pixels);probability=float(probabilities[0]);difference=abs(probability-p['probability'])
        if difference>.01:raise ValueError('Grad-CAM inference differs from selected DEV probability by more than0.01')
        heat=maps[0];color=plt.get_cmap('jet')(heat)[...,:3];overlay=np.clip(.6*rgb/255.+.4*color,0,1)
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,image,title in zip(axes,[rgb,heat,overlay],['Original indexed frame','Fake-logit patch Grad-CAM','Overlay']):
            ax.imshow(image,cmap='jet' if title.startswith('Fake') else None);ax.set_title(title);ax.axis('off')
        fig.suptitle(f'CLIP {category.upper()} | saved Fake p={p["probability"]:.4f}, CAM FP32 p={probability:.4f} | GT={"Fake" if row["label"] else "Real"}\n{row["source_file"]} @ {row["timestamps_exact"][0]}')
        path=destination/(category+'.png');fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
        saved.append({'category':category,'candidate_id':row['id'],'source_file':row['source_file'],'source_index':row['indices'][0],'timestamp_exact':row['timestamps_exact'][0],'label':row['label'],'threshold':.5,'fake_probability':probability,'saved_probability':p['probability'],'difference':difference,'zero_heatmap':bool(np.max(heat)==0),'figure':path.name})
    with (destination/'gradcam_examples.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(saved[0]) if saved else ['category']);w.writeheader();w.writerows(saved)
    write_json(destination/'provenance.json',{'checkpoint_sha256':digest(run/'best.pt'),'index_sha256':digest(inputs.index),'seed':seed,'target':'penultimate layer_norm1; Fake logit0; patch tokens only','threshold':.5,'test_used':False})
    del model;gc_collect()
    return destination


def gc_collect():
    import gc
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
