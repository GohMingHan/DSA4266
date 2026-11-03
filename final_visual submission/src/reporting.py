"""Cumulative epoch tables, five-model reports, saved DEV replay and explanations."""

import csv
import io
import json
import math
import zipfile
import shutil
import gc
import torch
from html import escape
from pathlib import Path, PurePosixPath
from .data import write_json, make_loader, LazyDataset
from .evaluation import evaluate_predictions, prediction_artifacts, plot_history
from .models import StagedCNN
from .temporal import ResNetLSTM
from .checkpoints import digest
from .gradcam import GradCAM, target_layer
from .config import ORDER, RECIPES, configuration
from .workflow import checkpoint_identity, project_id, guard_dev, mask_pipeline, new_id, choose_runs


GRAPH_ORDER = ('training_curves.png', 'best_dev_confusion.png', 'best_dev_roc_pr.png',
               'dev_f2_reporting_confusion.png')


class SavedRun:
    """Read only allowlisted DEV/history artifacts, including flat/nested run ZIPs."""
    def __init__(self, source):
        self.source = Path(source)
        if any(p.lower() in ('test', 'final_test') for p in self.source.parts):
            raise PermissionError('This reporting interface accepts saved TRAIN/DEV artifacts only')
        self.archive = self.source.suffix.lower() == '.zip'
        self.single_history = self.source.is_file() and self.source.suffix.lower() in ('.csv', '.json', '.jsonl')
        if self.archive:
            with zipfile.ZipFile(self.source) as archive:
                names = archive.namelist()
            roots = {str(PurePosixPath(n).parent) for n in names if PurePosixPath(n).name in ('history.jsonl', 'history.json', 'history.csv')
                     and not {'test', 'final_test', '..'} & set(PurePosixPath(n).parts)}
            if len(roots) != 1:
                raise ValueError('Choose a single completed model ZIP, not a mixed-session archive')
            prefix = next(iter(roots))
            self.prefix = '' if prefix == '.' else prefix+'/'

    def read(self, name):
        allowed = set(GRAPH_ORDER) | {'history.jsonl', 'history.json', 'history.csv', 'config.json',
            'training_summary.json', 'input_pipeline.json', 'best_dev_metrics.json', 'best_dev_video_predictions.json',
            'best_dev_predictions.csv', 'dev_f2_calibration.json', 'video_dev_f2_calibration.json'}
        if name not in allowed:
            raise PermissionError('Not an allowlisted DEV reporting artifact')
        if self.single_history:
            expected = 'history'+self.source.suffix.lower()
            return self.source.read_bytes() if name == expected else None
        if self.archive:
            with zipfile.ZipFile(self.source) as archive:
                try:
                    return archive.read(self.prefix+name)
                except KeyError:
                    return None
        path = self.source/name
        return path.read_bytes() if path.is_file() else None

    def json(self, name):
        data = self.read(name)
        return json.loads(data.decode('utf-8-sig')) if data is not None else None


def _scalar(value):
    if value is None or value == '' or isinstance(value, str) and value.lower() in ('nan', 'n/a', 'none', 'null'):
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        if value in ('True', 'False'):
            return value == 'True'
        return value


def _unflatten(row):
    result = {}
    for field, value in row.items():
        target = result; parts = field.split('.')
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _scalar(value)
    return result


def load_history(source):
    run = source if isinstance(source, SavedRun) else SavedRun(source)
    data = run.read('history.jsonl')
    if data is not None:
        rows = [json.loads(line) for line in data.decode('utf-8-sig').splitlines() if line.strip()]
    elif run.read('history.json') is not None:
        value = run.json('history.json'); rows = value.get('history', []) if isinstance(value, dict) else value
    else:
        data = run.read('history.csv')
        rows = [_unflatten(row) for row in csv.DictReader(io.StringIO(data.decode('utf-8-sig')))] if data is not None else []
    epochs = [int(row['epoch']) for row in rows]
    if epochs != sorted(set(epochs)):
        raise ValueError('Expected unique completed epochs in increasing order')
    rows = [dict(row, epoch=epoch) for row, epoch in zip(rows, epochs)]
    return rows


def _selected_epoch(history, summary):
    if summary.get('best_epoch') is not None:
        return int(summary['best_epoch'])
    improved = [int(r['epoch']) for r in history if _scalar(r.get('is_best')) is True]
    return improved[-1] if improved else None  # no AUC maximization or re-selection


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def at_half(metrics, fallback_threshold=None):
    """Keep AUC/loss; hide classification values if their saved threshold is unknown/non-0.5."""
    from .evaluation import metrics_with_f2
    metrics = metrics_with_f2(metrics)
    threshold = metrics.get('threshold', fallback_threshold)
    if not _numeric(threshold) or not math.isclose(threshold, .5, abs_tol=1e-12):
        metrics = {key: value for key, value in metrics.items() if key not in ('precision', 'recall', 'f1', 'f2')}
    return metrics


def _best_metrics(run, config, kind):
    """Re-threshold saved best probabilities only; never reuse them for other epochs."""
    from .evaluation import binary_metrics
    saved = run.json('best_dev_metrics.json') or {}
    unit = at_half(saved.get('unit', {}), config.get('threshold'))
    video = at_half(saved.get('video', {}), config.get('threshold'))
    videos = run.json('best_dev_video_predictions.json')
    if videos and any(video.get(key) is None for key in ('f2', 'pr_auc', 'roc_auc', 'precision', 'recall', 'f1')):
        computed = binary_metrics([v['target'] for v in videos], [v['probability'] for v in videos], .5)
        video = {**computed, **{k: v for k, v in video.items() if v is not None}}
    raw = run.read('best_dev_predictions.csv') if any(unit.get(key) is None for key in ('f2', 'pr_auc', 'roc_auc', 'precision', 'recall', 'f1')) else None
    if raw is not None:
        rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        # Legacy clip CSVs do not record metric eligibility: never assume all clips qualify.
        if rows and (kind == 'frame' or saved.get('unit', {}).get('n') == len(rows)):
            computed = binary_metrics([int(r['label']) for r in rows], [float(r['probability']) for r in rows], .5)
            unit = {**computed, **{k: v for k, v in unit.items() if v is not None}}
    return unit, video, saved.get('losses', {})


def epoch_table(source, *, kind=None):
    """One cumulative row per completed epoch; missing facts remain None/N/A."""
    run = source if isinstance(source, SavedRun) else SavedRun(source)
    config, summary = run.json('config.json') or {}, run.json('training_summary.json') or {}
    history = load_history(run)
    kind = kind or ('clip' if 'lstm' in config.get('model', '').lower() else 'frame')
    if kind not in ('frame', 'clip'):
        raise ValueError('Choose explicit frame or clip reporting')
    label = kind.title(); best_epoch = _selected_epoch(history, summary)
    saved_unit, saved_video, saved_losses = _best_metrics(run, config, kind)
    rows = []
    for entry in history:
        selected = entry['epoch'] == best_epoch
        unit = at_half(entry.get('dev', {}), config.get('threshold'))
        video = at_half(entry.get('dev_video', {}), config.get('threshold'))
        if selected:
            unit = {**saved_unit, **{k: v for k, v in unit.items() if v is not None}}
            video = {**saved_video, **{k: v for k, v in video.items() if v is not None}}
        lr = entry.get('learning_rates', {})
        rows.append({'Epoch': entry['epoch'], 'TRAIN Loss': entry.get('train', {}).get('loss'),
            'DEV Loss': unit.get('loss') if unit.get('loss') is not None else saved_losses.get('loss') if selected else None,
            'Balanced DEV Loss': unit.get('balanced_loss') if unit.get('balanced_loss') is not None else saved_losses.get('balanced_loss') if selected else None,
            f'{label} PR-AUC': unit.get('pr_auc'), f'{label} ROC-AUC': unit.get('roc_auc'),
            'Video ROC-AUC': video.get('roc_auc'), 'Video PR-AUC': video.get('pr_auc'),
            f'{label} Precision': unit.get('precision'), f'{label} Recall': unit.get('recall'),
            f'{label} F1': unit.get('f1'), f'{label} F2': unit.get('f2'),
            'Video Precision': video.get('precision'), 'Video Recall': video.get('recall'),
            'Video F1': video.get('f1'), 'Video F2': video.get('f2'),
            'Backbone LR': lr.get('final_stage'), 'Classifier/Head LR': lr.get('head', lr.get('temporal_head')),
            'Best Checkpoint': 'Selected best' if selected else ('—' if best_epoch is not None else 'N/A')})
    rows = [{key: None if isinstance(value, float) and not math.isfinite(value) else value for key, value in row.items()} for row in rows]
    return rows, best_epoch, kind


def table_html(rows, *, title='Completed epoch DEV metrics (threshold 0.5)'):
    """Four-decimal metrics, scientific LR values, sticky header and horizontal scrolling."""
    if not rows:
        return '<p>'+escape(title)+': N/A (no saved history)</p>'
    fields = list(dict.fromkeys(key for row in rows for key in row))
    result = ['<div style="max-width:100%;overflow-x:auto;max-height:650px;overflow-y:auto;">',
              '<table style="border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums;">',
              '<caption style="text-align:left;">'+escape(title)+'</caption><thead><tr>']
    for field in fields:
        result.append('<th style="position:sticky;top:0;background:#e9eef5;padding:7px;border:1px solid #aaa;">'+escape(field)+'</th>')
    result.append('</tr></thead><tbody>')
    for row in rows:
        selected = row.get('Best Checkpoint') == 'Selected best'
        result.append('<tr'+(' style="background:#fff0b3;font-weight:bold;"' if selected else '')+'>')
        for field in fields:
            value = row.get(field)
            if value is None or isinstance(value, float) and not math.isfinite(value):
                rendered = 'N/A'
            elif field.endswith('LR') and _numeric(value):
                rendered = f'{value:.4e}'  # 1.5e-5 must not misleadingly display as 0.0000
            elif _numeric(value) and field not in ('Epoch', 'Best Epoch'):
                rendered = f'{value:.4f}'
            else:
                rendered = str(value)
            result.append('<td style="padding:6px;border:1px solid #ccc;">'+escape(rendered)+'</td>')
        result.append('</tr>')
    return ''.join(result)+ '</tbody></table></div>'


def display_completed_run(source, *, name=None, kind=None, graph_names=GRAPH_ORDER):
    """Call AFTER train() returns, or independently to replay saved results. Read-only."""
    from IPython.display import HTML, Image, display
    run = SavedRun(source)
    rows, best_epoch, kind = epoch_table(run, kind=kind)
    print(name or str(source), '| completed history; classification metrics at threshold 0.5')
    display(HTML(table_html(rows)))
    for filename in graph_names:  # original images in their original order, no regeneration
        image = run.read(filename)
        if image is not None:
            display(Image(data=image))
        else:
            print(filename+': N/A (saved figure unavailable)')
    selected = next((row for row in rows if row['Epoch'] == best_epoch), {})
    print('Selected best epoch:', best_epoch if best_epoch is not None else 'N/A',
          '| Video F2 @0.5:', f"{selected['Video F2']:.4f}" if _numeric(selected.get('Video F2')) else 'N/A')
    calibration = run.json('dev_f2_calibration.json')
    if calibration:
        op = calibration.get('reporting', {})
        formatted = {k: f'{v:.4f}' if _numeric(v) else 'N/A' for k, v in
            {'threshold': calibration.get('threshold'), **{k: op.get(k) for k in ('precision', 'recall', 'f1', 'f2')}}.items()}
        print('DEV family-calibrated F2 reporting:', formatted)
        print('Checkpoint selection used full DEV: reporting is not an untouched final holdout.')
        print('Threshold changes cannot improve ROC/PR-AUC or BCE. TEST remains isolated.')
    video_calibration = run.json('video_dev_f2_calibration.json')
    if video_calibration:
        reporting = video_calibration.get('reporting', {})
        from .evaluation import metrics_with_f2
        f2 = metrics_with_f2(reporting).get('f2')
        print('DEV-calibrated video F2:', f'{f2:.4f}' if _numeric(f2) else 'N/A',
              '| saved threshold:', video_calibration.get('threshold', 'N/A'))
    return rows


def export_epoch_report(source, destination, *, kind=None, checkpoint_selection=None, in_run=False):
    """Export a NEW report, optionally as new flat files in its completed run."""
    destination = Path(destination)
    if in_run:
        if destination.resolve() != Path(source).resolve() or not destination.is_dir():
            raise ValueError('Flat run reports must stay inside their own existing source run')
        if any((destination/name).exists() for name in ('epoch_metrics.csv','epoch_metrics.json','epoch_metrics.html')):
            raise FileExistsError('Existing cumulative reports preserved')
    else:
        destination.mkdir(parents=True, exist_ok=False)
    rows, selected, native_kind = epoch_table(source, kind=kind)
    with (destination/'epoch_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['Epoch']); writer.writeheader(); writer.writerows(rows)
    metadata = {'rows': rows, 'selected_epoch': selected, 'level': native_kind,
                'threshold': .5, 'missing_values': 'null / N/A; never fabricated'}
    title = 'Completed epoch DEV metrics (threshold 0.5)'
    if checkpoint_selection is not None:
        from .config import selection_label
        metadata['checkpoint_selection'] = checkpoint_selection
        metadata['selection_label'] = selection_label(checkpoint_selection)
        title += ' | selected by '+selection_label(checkpoint_selection)
    (destination/'epoch_metrics.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    (destination/'epoch_metrics.html').write_text(table_html(rows, title=title), encoding='utf-8')
    return destination


REPORT_COLUMNS=['Model','Run ID','Input preprocessing','Head/temporal LR','Backbone LR','Best epoch','TRAIN Loss','DEV Loss','Balanced DEV Loss','Frame/Clip PR-AUC','Frame/Clip ROC-AUC','Video ROC-AUC','Video PR-AUC','Frame/Clip Precision','Frame/Clip Recall','Frame/Clip F1','Frame/Clip F2','Video Precision','Video Recall','Video F1','Video F2','Checkpoint selection']

def export_report(run,project):
    run=Path(run);identity,state=checkpoint_identity(run/'best.pt',project)
    if not (run/'training_summary.json').is_file():raise ValueError('Run is partial; report only completed models')
    out=Path(project)/'results'/identity['model_id']/identity['run_id']
    if out.exists():raise FileExistsError('Saved report preserved: '+str(out))
    out.mkdir(parents=True)
    allowed={'.json','.jsonl','.csv','.png','.html'}
    for p in run.iterdir():
        if p.is_file() and p.suffix in allowed and not p.name.startswith('sampling_epoch_') and p.name not in ['numerics.jsonl','progress.jsonl','run_status.json']:
            shutil.copy2(p,out/p.name)
    checkpoints={}
    for name in ['best.pt','last.pt']:
        p=run/name
        if p.is_file():checkpoints[name]={'path':str(p.resolve()),'sha256':digest(p),'bytes':p.stat().st_size}
    write_json(out/'checkpoint_manifest.json',{'experiment_identity':identity,'checkpoints':checkpoints,'weights_copied':False})
    (out/'README.md').write_text('# '+identity['label']+'\n\nNew ROC-AUC-selected run. Full-DEV threshold 0.5; Top-5% mean video scores. '\
        'Epoch tables precede saved plots. Calibrated thresholds are separate DEV-only analyses. '\
        'Checkpoint paths/hashes are in checkpoint_manifest.json. No historical results imported.\n')
    return out

def show_run(run,project):
    identity,_=checkpoint_identity(Path(run)/'best.pt',project)
    return display_completed_run(run,name=identity['label'],kind='clip' if identity['model_id']=='resnet50_lstm' else 'frame')

def comparison(project,runs=None):
    """Five rows; select explicit run IDs if multiple exist. Rank completed checkpoints only."""
    runs=choose_runs(project,runs);rows=[];eligible=[]
    for key in ORDER:
        r=RECIPES[key];cfg=configuration(key)
        from .config import selection_label
        row={k:None for k in REPORT_COLUMNS};row.update({'Model':r.label,'Input preprocessing':r.input,'Head/temporal LR':cfg.head_lr,'Backbone LR':'Frozen' if key=='resnet50_lstm' else cfg.backbone_lr,'Status':'Pending','Metric level':'Clip' if key=='resnet50_lstm' else 'Frame','Threshold':.5,'Checkpoint selection':selection_label(cfg.checkpoint_selection)})
        run=Path(runs[key]) if key in runs else None
        if run and run.is_dir():
            identity_path=run/'experiment_identity.json'
            if identity_path.is_file():
                identity=json.loads(identity_path.read_text())
                if identity.get('project_id')!=project_id(project) or identity.get('model_id')!=key:raise ValueError('Foreign experiment results refused')
                row['Run ID']=identity['run_id']
                summary=run/'training_summary.json';metrics=run/'best_dev_metrics.json'
                row['Status']='Partial'
                if summary.is_file() and metrics.is_file():
                    s=json.loads(summary.read_text());m=json.loads(metrics.read_text());h=[json.loads(x) for x in (run/'history.jsonl').read_text().splitlines() if x]
                    entry=next(x for x in h if x['epoch']==s['best_epoch'])
                    row.update({'Best epoch':s['best_epoch'],'TRAIN Loss':entry['train']['loss'],'DEV Loss':m['losses']['loss'],'Balanced DEV Loss':m['losses']['balanced_loss']})
                    for prefix,level in [('Frame/Clip','unit'),('Video','video')]:
                        for label,field in [('PR-AUC','pr_auc'),('ROC-AUC','roc_auc'),('Precision','precision'),('Recall','recall'),('F1','f1'),('F2','f2')]:row[prefix+' '+label]=m[level].get(field)
                    if m['unit']['threshold']!=.5 or m['video']['threshold']!=.5:raise ValueError('Comparison requires threshold 0.5')
                    row['Status']='Missing checkpoint'
                    if (run/'best.pt').is_file():
                        _,state=checkpoint_identity(run/'best.pt',project,key)
                        if state['best_epoch']!=s['best_epoch'] or state['epoch']+1!=s['best_epoch']:raise ValueError('Best metrics/checkpoint mismatch')
                        row['Status']='Completed, verified checkpoint';eligible.append(row)
        rows.append(row)
    # Trapezoidal video PR-AUC is primary; ROC-AUC only breaks exact ties.
    eligible.sort(key=lambda r:(-r['Video PR-AUC'],-r['Video ROC-AUC'],r['Model']))
    ranking=[{'rank':i+1,'model':r['Model'],'run_id':r['Run ID'],'video_pr_auc':r['Video PR-AUC'],'video_roc_auc':r['Video ROC-AUC']} for i,r in enumerate(eligible)]
    out=Path(project)/'results/comparisons'/new_id();out.mkdir(parents=True)
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(out/'comparison.json',{'rows':rows,'ranking':ranking,'primary_metric':'DEV video trapezoidal PR-AUC','threshold':.5,'test_used':False})
    (out/'comparison.html').write_text(table_html(rows,title='New ROC-selected models; classification threshold 0.5'),encoding='utf-8')
    if eligible:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,5));ys=list(range(len(eligible)))
        ax.barh(ys,[r['Video PR-AUC'] for r in eligible],label='Video PR-AUC');ax.set(yticks=ys,yticklabels=[r['Model'] for r in eligible],xlim=(0,1),xlabel='Full DEV trapezoidal PR-AUC',title='Completed verified checkpoints only');ax.invert_yaxis();fig.tight_layout();fig.savefig(out/'video_comparison.png',dpi=150);plt.close(fig)
    from IPython.display import HTML,display
    display(HTML((out/'comparison.html').read_text()))
    print('Ranking by DEV video PR-AUC:',ranking or 'Pending; no verified completed checkpoints',flush=True)
    return rows,ranking,out

def evaluate_dev(run,inputs,project,*,force_inference=False,split='dev'):
    guard_dev(split);run=Path(run);identity,state=checkpoint_identity(run/'best.pt',project)
    if not force_inference and (run/'best_dev_metrics.json').is_file():
        print('Reusing saved selected-best DEV metrics; no inference.',flush=True)
        return json.loads((run/'best_dev_metrics.json').read_text())
    from .training import run_epoch, require_cuda
    from .data import DevSchedule
    from .temporal import ClipIndex
    require_cuda();key=identity['model_id'];cfg=configuration(key,amp=state['config']['amp']);kind='clip' if key=='resnet50_lstm' else 'frame'
    if kind=='clip':
        index=ClipIndex(run/'clip_manifest.json',inputs.data)
        encoder=StagedCNN(configuration('resnet50_baseline'),'.',pretrained=False).network;encoder.fc=torch.nn.Identity();model=ResNetLSTM(encoder,cfg)
    else:index=DevSchedule(inputs.index,inputs.data);model=StagedCNN(cfg,'.',pretrained=False)
    pipeline=mask_pipeline(inputs,project) if RECIPES[key].input=='face_mask' else None
    if (pipeline.identity if pipeline else None)!=state.get('input_pipeline'):raise ValueError('Saved preprocessing differs')
    records=index.records('dev',kind);model.load_state_dict(state['model'],strict=True);model.cuda().eval()
    loader=(pipeline.make_loader if pipeline else make_loader)(records,inputs.data,cfg)
    metrics,predictions,_=run_epoch(model,loader,cfg)
    report=evaluate_predictions(records,predictions,{'aggregation':'topk','topk_fraction':.05,'prediction_threshold':.5,'video_threshold':.5},kind)
    report['losses']={k:metrics[k] for k in ['loss','real_class_loss','fake_class_loss','balanced_loss']}
    out=Path(project)/'results'/key/identity['run_id']/('dev_evaluation_'+new_id());out.mkdir(parents=True)
    prediction_artifacts(out,records,predictions,report,unit_label=kind.title());write_json(out/'evaluation_provenance.json',{'checkpoint_sha256':digest(run/'best.pt'),'experiment_identity':identity,'split':'dev'})
    index.close();del model,loader,state;gc.collect();torch.cuda.empty_cache()
    return out

def explain_dev(run,inputs,project,*,category='tp',count=3,split='dev'):
    """Four CNNs only. Saved DEV predictions select examples; no threshold tuning."""
    guard_dev(split);run=Path(run);identity,state=checkpoint_identity(run/'best.pt',project);key=identity['model_id']
    if key=='resnet50_lstm':raise NotImplementedError('CNN Grad-CAM is supported for four frame models; LSTM attribution is intentionally not claimed')
    cfg=configuration(key,amp=False,workers=0);model=StagedCNN(cfg,'.',pretrained=False);model.load_state_dict(state['model'],strict=True);model.eval()
    with (run/'best_dev_predictions.csv').open() as f:pred=list(csv.DictReader(f))
    rule={'tp':lambda y,p:y==1 and p>=.5,'tn':lambda y,p:y==0 and p<.5,'fp':lambda y,p:y==0 and p>=.5,'fn':lambda y,p:y==1 and p<.5}
    if category not in rule:raise ValueError('Use tp/tn/fp/fn')
    pred=sorted([r for r in pred if rule[category](int(r['label']),float(r['probability']))],key=lambda r:(-abs(float(r['probability'])-.5),int(r['candidate_id'])))[:count]
    if not pred:raise ValueError('No DEV examples in requested category')
    from .data import DevSchedule
    index=DevSchedule(inputs.index,inputs.data);records=index.records('dev','frame',[int(r['candidate_id']) for r in pred]);index.close()
    pipeline=mask_pipeline(inputs,project) if RECIPES[key].input=='face_mask' else None
    if (pipeline.identity if pipeline else None)!=state.get('input_pipeline'):raise ValueError('Exact saved mask policy required')
    dataset=pipeline.dataset(records,inputs.data,cfg) if pipeline else LazyDataset(records,inputs.data)
    layer,_=target_layer(model,cfg.model);out=Path(project)/'results'/key/identity['run_id']/('gradcam_'+new_id());out.mkdir(parents=True);rows=[]
    import numpy as np
    import matplotlib.pyplot as plt
    original_hash=digest(run/'best.pt')
    for i,record in enumerate(records):
        tensor,_,_=dataset[i];cam=GradCAM(model,layer)(tensor[None]);rgb=dataset.reader.read([record])[0][0];heat=cam['heatmaps'][0]
        overlay=np.clip(.6*rgb/255+.4*plt.get_cmap('jet')(heat)[...,:3],0,1)
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,img,title in zip(axes,[rgb,heat,overlay],['Actual model input RGB','Fake-logit Grad-CAM','Overlay']):ax.imshow(img,cmap='jet' if title.startswith('Fake') else None);ax.set_title(title);ax.axis('off')
        p=float(cam['probabilities'][0]);fig.suptitle(f"{identity['label']} | {record['source_file']} | {record['timestamps_exact'][0]}s | GT={record['label']} | Fake p={p:.4f}",fontsize=9)
        fig.tight_layout();fig.savefig(out/f'frame_{i:03}.png',dpi=150);plt.close(fig)
        rows.append({'figure':f'frame_{i:03}.png','candidate_id':record['id'],'video':record['source_file'],'timestamp_exact':record['timestamps_exact'][0],'label':record['label'],'fake_probability':p,'category':category})
    with (out/'frames.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if digest(run/'best.pt')!=original_hash:raise AssertionError('Checkpoint modified')
    write_json(out/'provenance.json',{'checkpoint_sha256':original_hash,'split':'dev','experiment_identity':identity,'target':'Fake logit','input_pipeline':state.get('input_pipeline'),'weights_modified':False})
    return out
