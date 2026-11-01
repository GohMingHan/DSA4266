"""Join by exact metadata file ID; independently pool each frozen feature stream."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np
from .io import (VERSION,SPLITS,read_json,write_json,write_jsonl,write_npz,digest,sha256,
                 under,require_new,load_config,canonical_cohort,check_expected,check_sequence)


def read_export_split(root,split,modality,allow_test_only=False):
    root=Path(root); info=read_json(root/'export.json'); s=read_json(root/'splits'/f'{split}.json')
    if info.get('format')!=VERSION or s.get('format')!=VERSION or s.get('modality')!=modality:
        raise ValueError('Wrong embedding schema or modality.')
    if info['encoder_id']!=s['encoder_id']: raise ValueError('Mixed encoders in embedding cache.')
    if info.get('time_origin')!='first_decoded_video_frame': raise ValueError('Timeline origin is not verified.')
    if s.get('partial_pilot'): raise ValueError('Pilot export cannot enter a reported training cohort. Run full extraction into its own cache.')
    if (s.get('test_only') or info.get('test_only')) and not allow_test_only:
        raise ValueError('Random/synthetic test embeddings are blocked from research runs.')
    if s.get('signature')!=digest({k:v for k,v in s.items() if k!='signature'}):
        raise ValueError('Embedding index integrity check failed.')
    rows={r['file']:r for r in s['videos']}
    if len(rows)!=len(s['videos']): raise ValueError('Duplicate embedding video ID.')
    return info,rows


def read_sequence(root,record,modality):
    p=under(root,record['feature_path'])
    if sha256(p)!=record['sha256']: raise ValueError(f'Embedding changed or corrupted: {p}')
    with np.load(p,allow_pickle=False) as z: data=check_sequence(z,modality)
    if list(data['features'].shape)!=record['shape']: raise ValueError('Embedding shape disagrees with index.')
    return data


def pool(features,mode):
    # Pool each stream on its own; NO pairing of individual audio windows to frames.
    h=np.asarray(features,np.float64)
    mean=h.mean(axis=0)
    if mode=='mean':return mean.astype(np.float32)
    if mode=='mean_max':return np.concatenate([mean,h.max(axis=0)]).astype(np.float32)
    raise ValueError('Unknown independent pooling mode.')


def split_content_id(rows,visual,audio,encoder_ids,pooling):
    return digest({'rows':rows,'visual_sha256':__import__('hashlib').sha256(np.ascontiguousarray(visual).tobytes()).hexdigest(),
                   'audio_sha256':__import__('hashlib').sha256(np.ascontiguousarray(audio).tobytes()).hexdigest(),
                   'shapes':[list(visual.shape),list(audio.shape)],'encoder_ids':encoder_ids,'pooling':pooling,'target':'modify_audio OR modify_video'})


def prepare(config_path,out,splits=('train','dev'),pooling='mean',allow_test_only=False):
    c=load_config(config_path); rows=canonical_cohort(c['cohort']); check_expected(rows,splits,c.get('expected_counts'))
    output=Path(out).expanduser().resolve()
    for k in ('visual_embeddings','audio_embeddings'):
        p=Path(c[k])
        if output==p or output.is_relative_to(p) or p.is_relative_to(output): raise ValueError('Paired output must be separate from embedding caches.')
    require_new(output)
    prepared={}; encoder_ids=None; dims=None; counts={}; timing_warnings=[]; test_only=False
    for split in splits:
        selected=[r for r in rows if r['split']==split]; ids={r['file'] for r in selected}
        vi,vr=read_export_split(c['visual_embeddings'],split,'visual',allow_test_only)
        ai,ar=read_export_split(c['audio_embeddings'],split,'audio',allow_test_only)
        this_ids={'visual':vi['encoder_id'],'audio':ai['encoder_id']}
        test_only=test_only or vi.get('test_only',False) or ai.get('test_only',False)
        if encoder_ids is not None and this_ids!=encoder_ids: raise ValueError('Different encoders used between splits.')
        encoder_ids=this_ids
        errors={m:{'missing':sorted(ids-set(index)),'extra':sorted(set(index)-ids)} for m,index in [('visual',vr),('audio',ar)]}
        if any(v['missing'] or v['extra'] for v in errors.values()):
            write_json(output/'coverage_error.json',{'split':split,**errors})
            raise ValueError('Audio/visual IDs do not exactly match the requested cohort. See coverage_error.json. No automatic intersection or relabelling.')
        vv=[];aa=[];items=[]
        for r in selected:
            name=r['file']
            for m,index in [('visual',vr),('audio',ar)]:
                if index[name]['split']!=split or index[name]['encoder_id']!=encoder_ids[m]:
                    raise ValueError('Embedding split or encoder identity mismatch.')
            v=read_sequence(c['visual_embeddings'],vr[name],'visual')
            a=read_sequence(c['audio_embeddings'],ar[name],'audio')
            # Check overlapping timelines, without stretching/cropping them to match.
            if a['start_s'].min()>v['timestamp_s'].max()+1e-6 or a['end_s'].max()<v['timestamp_s'].min()-1e-6:
                raise ValueError(f'No shared audio/visual source time: {name}. Verify the audio offset.')
            if abs(float(a['start_s'].min()-v['timestamp_s'].min()))>.5 or abs(float(a['end_s'].max()-v['timestamp_s'].max()))>.5:
                timing_warnings.append({'file':name,'type':'audio_visual_extent_difference',
                    'visual_anchor_range':[float(v['timestamp_s'].min()),float(v['timestamp_s'].max())],
                    'audio_support_range':[float(a['start_s'].min()),float(a['end_s'].max())],
                    'note':'Pooled classification does not require equal sequence lengths; inspect before fine alignment. No rescaling applied.'})
            vv.append(pool(v['features'],pooling)); aa.append(pool(a['features'],pooling))
            items.append({**r,'visual_steps':len(v['features']),'audio_windows':len(a['features'])})
        V,A=np.stack(vv),np.stack(aa)
        if not np.isfinite(V).all() or not np.isfinite(A).all():raise ValueError('Nonfinite pooled embeddings.')
        this_dims={'visual':int(V.shape[1]),'audio':int(A.shape[1])}
        if dims is not None and dims!=this_dims:raise ValueError('Feature dimensions change between splits.')
        dims=this_dims
        signature=split_content_id(items,V,A,encoder_ids,pooling)
        write_npz(output/f'{split}.npz',visual=V,audio=A,labels=np.array([r['label'] for r in items],np.int64))
        write_jsonl(output/f'{split}.jsonl',items)
        prepared[split]={'signature':signature,'npz_sha256':sha256(output/f'{split}.npz'),
                         'rows_sha256':sha256(output/f'{split}.jsonl'),'videos':len(items)}
        counts[split]={'videos':len(items),'label_counts':dict(Counter(str(r['label']) for r in items)),
                       'categories':dict(Counter(r['category'] for r in items)),
                       'source_families':len({r['source_family'] for r in items})}
    result={'format':VERSION,'target':'modify_audio OR modify_video','pooling':pooling,'dims':dims,
            'encoder_ids':encoder_ids,'encoder_provenance':{'visual':vi,'audio':ai},
            'splits':prepared,'counts':counts,'test_only':test_only,
            'time_origin':'first_decoded_video_frame','timing_warnings':len(timing_warnings),
            'note':'One pair of independently pooled embeddings per video. Labels/timestamps are NOT classifier inputs. Original feature sequences remain in embedding caches.'}
    write_jsonl(output/'timing_warnings.jsonl',timing_warnings)
    write_json(output/'dataset.json',result)
    print(f'Paired dataset: {output}/dataset.json\n{counts}')
    return result


def load_split(root,split,allow_test_only=False):
    from .io import read_jsonl
    root=Path(root); cfg=read_json(root/'dataset.json')
    if cfg.get('format')!=VERSION or cfg.get('target')!='modify_audio OR modify_video': raise ValueError('Wrong paired dataset format/target.')
    if cfg.get('test_only') and not allow_test_only:raise ValueError('Synthetic smoke data is not research data.')
    if split not in cfg['splits']:raise ValueError(f'{split} has not been prepared.')
    rec=cfg['splits'][split]
    for ext,key in [('npz','npz_sha256'),('jsonl','rows_sha256')]:
        if sha256(root/f'{split}.{ext}')!=rec[key]:raise ValueError(f'{split} data file changed; rebuild in a new output.')
    rows=read_jsonl(root/f'{split}.jsonl')
    with np.load(root/f'{split}.npz',allow_pickle=False) as z:
        V,A,y=z['visual'].copy(),z['audio'].copy(),z['labels'].copy()
    if len(rows)!=len(V) or len(V)!=len(A) or y.shape!=(len(V),):raise ValueError('Paired array lengths disagree.')
    if V.ndim!=2 or A.ndim!=2 or V.shape[1]!=cfg['dims']['visual'] or A.shape[1]!=cfg['dims']['audio']:raise ValueError('Paired dimensions disagree.')
    if not np.array_equal(y,np.array([r['label'] for r in rows])) or not np.isin(y,[0,1]).all():raise ValueError('Target corruption.')
    if not np.isfinite(V).all() or not np.isfinite(A).all():raise ValueError('Nonfinite pooled features.')
    if split_content_id(rows,V,A,cfg['encoder_ids'],cfg['pooling'])!=rec['signature']:raise ValueError('Paired dataset integrity failed.')
    return cfg,rows,V,A,y
