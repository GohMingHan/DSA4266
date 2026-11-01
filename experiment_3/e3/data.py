"""Read the UNPOOLED canonical Experiment 2B caches without copying or altering them.

The trained encoders are not loaded by this module. Normalization uses the exact
natural TRAIN video-mean statistics used by the Experiment 2B mean-pooling head.
Every observed sequence entry is retained. Times use a shared video-relative origin.
"""
from __future__ import annotations
from collections import Counter, OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from e2b.io import (load_config as load_e2b_config, canonical_cohort, check_expected,
                    check_sequence, relative_name, SPLITS)
from e2b.prepare import (read_export_split, read_sequence, pool, split_content_id,
                         load_split as load_e2b_paired)
from e2b.train import normalizer
from .common import (VERSION,TARGET,load_config,read_json,read_jsonl,write_json,
    write_jsonl,write_npz,digest,sha256,under,require_new,lock,assert_separate_output)

CORE = ('file','split','modify_audio','modify_video','label','source_group','source_family','category')

def identity_rows(items):
    return [{**{k:r[k] for k in CORE}, 'visual_steps':r['visual']['shape'][0],
             'audio_windows':r['audio']['shape'][0]} for r in items]

def split_id(items,encoder_ids):
    return digest({'format':VERSION,'rows':items,'encoder_ids':encoder_ids,'target':TARGET,
                   'temporal_policy':'all_original_observations; no interpolation/subsampling'})

def prepare(config_path,out,splits=('train','dev'),e2b_data=None,allow_test_only=False):
    splits = [s for s in SPLITS if s in splits]
    if not {'train','dev'} <= set(splits):
        raise ValueError('Prepare train+dev together; add test later in a new folder with unchanged train/dev.')
    c=load_e2b_config(load_config(config_path)['experiment_2b_config'])
    cohort=canonical_cohort(c['cohort']);check_expected(cohort,splits,c.get('expected_counts'))
    roots={m:Path(c[m+'_embeddings']).resolve() for m in ('visual','audio')}
    output=assert_separate_output(out,list(roots.values()) + ([e2b_data] if e2b_data else []))
    require_new(output)
    ids=None;dims=None;prepared={};counts={};warnings=[];test_only=False;provenance={};norm=None
    with lock(output,'.e3.lock'):
        for split in splits:
            selected=[r for r in cohort if r['split']==split];expected={r['file'] for r in selected}
            indices={}
            for m in ('visual','audio'):
                info,index=read_export_split(roots[m],split,m,allow_test_only)
                raw_index=read_json(roots[m]/'splits'/f'{split}.json')
                if info.get('modality')!=m or raw_index.get('split')!=split:
                    raise ValueError('Incorrect modality/split in canonical export.')
                test_only |= bool(info.get('test_only') or raw_index.get('test_only'))
                provenance[m]=info;indices[m]=index
                missing=sorted(expected-set(index));extra=sorted(set(index)-expected)
                if missing or extra:
                    write_json(output/'coverage_error.json',{'split':split,'modality':m,'missing':missing,'extra':extra})
                    raise ValueError('Embedding/cohort IDs differ. See coverage_error.json; no intersection/zero filling performed.')
            current_ids={m:provenance[m]['encoder_id'] for m in roots}
            if ids is not None and ids!=current_ids:raise ValueError('Encoder identity differs across splits.')
            ids=current_ids;items=[];V=[];A=[];lengths={'visual':[],'audio':[]}
            for r in tqdm(selected,desc=f'Validate {split} sequences',unit='video'):
                name=r['file'];entry={**r}
                arrays={}
                for m in roots:
                    rec=indices[m][name]
                    if rec['split']!=split or rec['encoder_id']!=ids[m]:raise ValueError('Record split/encoder mismatch.')
                    if 'status' in rec and rec['status']!='complete':raise ValueError('Incomplete embedding record.')
                    if relative_name(rec['file'])!=name:raise ValueError('Unsafe/noncanonical record ID.')
                    for key in ('modify_audio','modify_video','label','source_family','category'):
                        if key in rec and rec[key]!=r[key]:raise ValueError(f'Conflicting {key} for {name}.')
                    arr=read_sequence(roots[m],rec,m);arrays[m]=arr
                    # Only numeric encoder representations and input-time coordinates enter the loader.
                    entry[m]={k:rec[k] for k in ('feature_path','sha256','shape')}
                    lengths[m].append(len(arr['features']))
                v,a=arrays['visual'],arrays['audio']
                if a['start_s'].min()>v['timestamp_s'].max()+1e-6 or a['end_s'].max()<v['timestamp_s'].min()-1e-6:
                    raise ValueError(f'No overlapping audio/visual time: {name}. Verify preprocessing offsets.')
                delta=max(abs(float(a['start_s'].min()-v['timestamp_s'].min())),
                          abs(float(a['end_s'].max()-v['timestamp_s'].max())))
                if delta>.5:
                    warnings.append({'file':name,'type':'audio_visual_extent_difference','seconds':delta,
                       'visual_anchor_range':[float(v['timestamp_s'].min()),float(v['timestamp_s'].max())],
                       'audio_support_range':[float(a['start_s'].min()),float(a['end_s'].max())],
                       'note':'Not corrected automatically. Review before interpreting temporal alignment.'})
                V.append(pool(v['features'],'mean'));A.append(pool(a['features'],'mean'));items.append(entry)
            vmean,amean=np.stack(V),np.stack(A)
            d={'visual':int(vmean.shape[1]),'audio':int(amean.shape[1])}
            if dims is not None and d!=dims:raise ValueError('Changing feature dimensions across splits.')
            dims=d
            canonical=identity_rows(items)
            e2b_sig=split_content_id(canonical,vmean,amean,ids,'mean')
            if e2b_data is not None:
                # The existing mean-concatenation dataset is optional; when supplied verify EXACT inputs.
                old=read_json(Path(e2b_data)/'dataset.json')
                if split in old['splits']:
                    oc,ors,ov,oa,oy=load_e2b_paired(e2b_data,split,allow_test_only)
                    if oc['pooling']!='mean' or oc['encoder_ids']!=ids or ors!=canonical:
                        raise ValueError('Experiment 2B paired dataset is not the same mean-pooled cohort/encoder pair.')
                    if not np.array_equal(ov,vmean) or not np.array_equal(oa,amean):
                        raise ValueError('Sequence means differ from Experiment 2B inputs.')
                    if oc['splits'][split]['signature']!=e2b_sig:raise ValueError('Experiment 2B fingerprint mismatch.')
            if split=='train':norm={m:normalizer(x) for m,x in [('visual',vmean),('audio',amean)]}
            write_jsonl(output/f'{split}.jsonl',items)
            prepared[split]={'signature':split_id(items,ids),'rows_sha256':sha256(output/f'{split}.jsonl'),
                             'videos':len(items),'e2b_mean_signature':e2b_sig}
            counts[split]={'videos':len(items),'labels':dict(Counter(str(r['label']) for r in items)),
                          'categories':dict(Counter(r['category'] for r in items)),
                          'source_families':len({r['source_family'] for r in items}),
                          'sequence_lengths':{m:{'min':int(min(x)),'max':int(max(x)),'mean':float(np.mean(x))}
                                              for m,x in lengths.items()}}
        write_npz(output/'normalization.npz',**{f'{m}_{key}':val.numpy() for m,st in norm.items() for key,val in st.items()})
        result={'format':VERSION,'target':TARGET,'dims':dims,'encoder_ids':ids,'encoder_provenance':provenance,
          'embedding_roots':{m:str(p) for m,p in roots.items()},'splits':prepared,'counts':counts,
          'normalization':{'method':'natural_TRAIN_video_mean_statistics; shared with E2B mean baseline',
              'npz_sha256':sha256(output/'normalization.npz')},
          'time_origin':'first_decoded_video_frame','temporal_policy':'all_original_observations',
          'test_only':test_only,'timing_warnings':len(warnings),'e2b_paired_reference':str(e2b_data) if e2b_data else None}
        write_jsonl(output/'timing_warnings.jsonl',warnings);write_json(output/'dataset.json',result)
    print(f'Sequence index ready: {output}/dataset.json\n{counts}')
    return result


def load_index(root,split,allow_test_only=False):
    root=Path(root).expanduser().resolve();cfg=read_json(root/'dataset.json')
    if cfg.get('format')!=VERSION or cfg.get('target')!=TARGET:raise ValueError('Not an Experiment 3 sequence dataset.')
    if cfg.get('test_only') and not allow_test_only:raise ValueError('Synthetic test fixtures cannot be used as research data.')
    if split not in cfg['splits']:raise ValueError(f'{split} has not been prepared.')
    item=cfg['splits'][split];path=root/f'{split}.jsonl'
    if sha256(path)!=item['rows_sha256']:raise ValueError('Sequence index changed; prepare into a new output.')
    rows=read_jsonl(path)
    if len(rows)!=item['videos'] or split_id(rows,cfg['encoder_ids'])!=item['signature']:
        raise ValueError('Sequence-index content integrity failure.')
    if len({r['file'] for r in rows})!=len(rows):raise ValueError('Duplicate video rows.')
    for r in rows:
        if r['split']!=split or r['label']!=int(r['modify_audio'] or r['modify_video']):raise ValueError('Invalid split/target.')
    for m,p in cfg['embedding_roots'].items():
        info,index=read_export_split(p,split,m,allow_test_only)
        if info['encoder_id']!=cfg['encoder_ids'][m]:raise ValueError('Encoder cache changed after preparation.')
        if set(index)!={r['file'] for r in rows}:raise ValueError('Sequence cache coverage changed.')
        for r in rows:
            if any(index[r['file']][k]!=r[m][k] for k in ('feature_path','sha256','shape')):
                raise ValueError('Source embedding record changed. Rebuild in a new prepared folder.')
    path=root/'normalization.npz'
    if sha256(path)!=cfg['normalization']['npz_sha256']:raise ValueError('Saved normalization changed.')
    with np.load(path,allow_pickle=False) as f:
        norm={m:{k:torch.from_numpy(f[f'{m}_{k}'].copy()) for k in ('mean','std')} for m in ('visual','audio')}
    for m in norm:
        for key in ('mean','std'):
            x=norm[m][key]
            if x.shape!=(cfg['dims'][m],) or not torch.isfinite(x).all():raise ValueError('Invalid normalization.')
        if (norm[m]['std']<=0).any():raise ValueError('Normalization std must be positive.')
    return cfg,rows,norm


class Sequences(Dataset):
    def __init__(self,cfg,rows,cache_videos=4):
        self.cfg=cfg;self.rows=rows;self.limit=max(0,int(cache_videos));self.cache=OrderedDict()
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        if index in self.cache:
            item=self.cache.pop(index);self.cache[index]=item;return item
        r=self.rows[index];result={'index':index,'label':r['label']}
        for m in ('visual','audio'):
            arrays=read_sequence(self.cfg['embedding_roots'][m],r[m],m)
            result[m]=torch.from_numpy(arrays['features'].copy())
            key='timestamp_s' if m=='visual' else 'center_s'
            result[m+'_time']=torch.tensor(arrays[key],dtype=torch.float32)
            # Compute E2B's mean BEFORE standardization, retaining exact original pooled vectors.
            result[m+'_mean']=torch.from_numpy(pool(arrays['features'],'mean'))
        if self.limit:
            self.cache[index]=result
            while len(self.cache)>self.limit:self.cache.popitem(last=False)
        return result


def collate(items):
    if not items:raise ValueError('Empty batch.')
    batch={'label':torch.tensor([x['label'] for x in items],dtype=torch.float32),
           'index':torch.tensor([x['index'] for x in items],dtype=torch.long)}
    for m in ('visual','audio'):
        n=max(len(x[m]) for x in items);d=items[0][m].shape[1]
        x=torch.zeros(len(items),n,d,dtype=torch.float32);t=torch.zeros(len(items),n)
        pad=torch.ones(len(items),n,dtype=torch.bool)
        for i,item in enumerate(items):
            k=len(item[m]);x[i,:k]=item[m];t[i,:k]=item[m+'_time'];pad[i,:k]=False
        batch[m]=x;batch[m+'_time']=t;batch[m+'_padding']=pad
        batch[m+'_mean']=torch.stack([x[m+'_mean'] for x in items])
    return batch
