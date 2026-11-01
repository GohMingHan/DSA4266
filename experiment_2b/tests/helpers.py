from pathlib import Path
from types import SimpleNamespace
import numpy as np
from e2b.io import VERSION,digest,write_json,write_jsonl,write_npz,sha256,file_key
from e2b.prepare import prepare


def fixture(root,with_test=True):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(21);rows=[]
    for split,n in [('train',8),('dev',4),('test',4)]:
        for j in range(n):
            orig=f'{split}/g{j}_0.mp4'
            for k,(ma,mv) in enumerate([(0,0),(1,0),(0,1),(1,1)]):
                rows.append({'file':f'{split}/g{j}_{k}.mp4','original':orig,'split':split,
                             'modify_audio':ma,'modify_video':mv,'duration':2.,'fake_periods':[[.5,1.0]] if ma or mv else []})
    write_json(root/'cohort.json',rows)
    for mod,d in [('visual',7),('audio',5)]:
        base=root/mod;provenance={'format':VERSION,'modality':mod,'time_origin':'first_decoded_video_frame',
            'engine':{'checkpoint_sha256':'synthetic-'+mod,'representation':'synthetic software features'},'test_only':True}
        provenance['encoder_id']=digest(provenance);write_json(base/'export.json',provenance)
        for split in ('train','dev','test'):
            if split=='test' and not with_test:continue
            entries=[]
            for r in rows:
                if r['split']!=split:continue
                t=3+(int(r['file'].split('_')[-1].split('.')[0])%3)
                values=rng.normal(size=(t,d)).astype('float32')*.05
                values[:,0]+=3*r['modify_video'] if mod=='visual' else 3*r['modify_audio']
                path=base/'clips'/split/(file_key(r['file'])+'.npz')
                arrays={'features':values}
                if mod=='visual':arrays['timestamp_s']=np.linspace(0,1.8,t)
                else:
                    arrays['start_s']=np.linspace(0,1,t);arrays['end_s']=np.linspace(1,2,t);arrays['center_s']=np.linspace(.5,1.5,t)
                write_npz(path,**arrays)
                entries.append({'file':r['file'],'split':split,'encoder_id':provenance['encoder_id'],
                    'feature_path':path.relative_to(base).as_posix(),'shape':[t,d],'sha256':sha256(path)})
            index={'format':VERSION,'encoder_id':provenance['encoder_id'],'modality':mod,'split':split,
                   'partial_pilot':False,'test_only':True,'videos':entries};index['signature']=digest(index)
            write_json(base/'splits'/f'{split}.json',index)
    c={'format':VERSION,'cohort':'cohort.json','visual_slot':'models/visual/encoder.json','audio_slot':'models/audio/encoder.json',
       'visual_embeddings':'visual','audio_embeddings':'audio','expected_counts':{'train':32,'dev':16,'test':16},'threads':2}
    write_json(root/'config.json',c)
    return root/'config.json'


def paired(root,with_test=True):
    config=fixture(Path(root)/'inputs',with_test)
    out=Path(root)/'paired'
    prepare(config,out,('train','dev','test') if with_test else ('train','dev'),allow_test_only=True)
    return out


def train_args(data,out,**kw):
    d=dict(data=str(data),out=str(out),mode='concat',hidden=12,dropout=.1,batch_size=8,
           epochs=5,patience=0,lr=.01,weight_decay=.0001,seed=42,balance='class',
           threads=2,device='cpu',resume=None,smoke_test=True)
    d.update(kw);return SimpleNamespace(**d)


def eval_args(data,ckpt,out,**kw):
    d=dict(data=str(data),checkpoint=str(ckpt),out=str(out),split='test',device='cpu',
           batch_size=8,threads=2,smoke_test=True);d.update(kw);return SimpleNamespace(**d)
