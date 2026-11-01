"""One-modality frozen extraction worker, launched in the selected model environment."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from .io import (VERSION,SPLITS,read_json,write_json,write_jsonl,write_npz,digest,sha256,
                 file_key,under,lock,load_config,load_slot,canonical_cohort,check_expected,
                 check_sequence,code_hashes)
from .encoders import make_encoder,check_meta


def extract(config_path,modality,splits,out=None,limit=0,device='cuda:0'):
    c=load_config(config_path); slot=load_slot(c[modality+'_slot'])
    cohort=canonical_cohort(c['cohort']); check_expected(cohort,splits,c.get('expected_counts'))
    output=Path(out or c[modality+'_embeddings']).expanduser().resolve()
    protected=[Path(slot['checkpoint']).parent]
    for key in ('project_dir','input_root','input_manifest','index'):
        if slot.get(key):
            p=Path(slot[key]); protected.append(p if p.is_dir() else p.parent)
    for p in protected:
        if output==p or output.is_relative_to(p) or p.is_relative_to(output):
            raise ValueError(f'Put embeddings outside protected model/source/input directory {p}.')
    if output.exists() and any(output.iterdir()) and not (output/'export.json').exists():
        raise ValueError('Output is nonempty but is not an Experiment 2B embedding cache.')
    torch.set_num_threads(int(c.get('threads',4)))
    start=time.perf_counter()
    # Instantiate only the selected encoder; no new pretrained weights are fetched.
    engine=make_encoder(slot,modality,device)
    provenance={'format':VERSION,'modality':modality,'engine':engine.provenance,
                'e2b_source_hashes':code_hashes(),'numpy':np.__version__,'torch':str(torch.__version__),
                'time_origin':'first_decoded_video_frame','test_only':False}
    encoder_id=digest(provenance); provenance['encoder_id']=encoder_id
    if (output/'export.json').is_file() and read_json(output/'export.json')!=provenance:
        raise ValueError('Encoder/checkpoint/preprocessing/source changed. Use a new embedding folder; do not mix features.')
    selected=[]
    for split in splits:
        rows=[r for r in cohort if r['split']==split]
        # Stable label-independent pilot selection (not sampling only fake regions).
        rows.sort(key=lambda r:(digest([42,r['file']]),r['file']))
        selected.extend(rows[:limit] if limit else rows)
    missing=[r['file'] for r in selected if r['file'] not in engine.records]
    if missing:
        raise ValueError(f'{len(missing)} cohort videos absent from the selected encoder input. Examples: {missing[:5]}. Finish/audit preprocessing; do not relabel or silently drop them.')
    for row in selected: check_meta(engine.info(row),row)
    counts=Counter(); failed=[]; interrupted=False; complete=[]
    with lock(output):
        write_json(output/'export.json',provenance)
        # A failed retry must not leave a published split looking complete.
        for split in splits:
            (output/'splits'/f'{split}.json').unlink(missing_ok=True)
        from tqdm import tqdm
        try:
            for row in tqdm(selected,desc=f'{modality}: frozen embeddings',unit='video'):
                name=row['file']; key=file_key(name)
                record_path=output/'records'/f'{key}.json'
                target=output/'clips'/row['split']/f'{key}.npz'
                try:
                    source_id=engine.input_signature(row)
                    identity={'file':name,'split':row['split'],'encoder_id':encoder_id,
                              'source_signature':source_id}
                    old=read_json(record_path) if record_path.is_file() else None
                    if old and old.get('status')=='complete':
                        if any(old.get(k)!=v for k,v in identity.items()):
                            raise ValueError('Cached source identity differs; use a new embedding folder.')
                        if target.is_file() and sha256(target)==old['sha256']:
                            complete.append(old); counts['resumed']+=1; continue
                    arrays=check_sequence(engine.encode(row),modality)
                    write_npz(target,**arrays)
                    record={**identity,'status':'complete','feature_path':target.relative_to(output).as_posix(),
                            'sha256':sha256(target),'shape':list(arrays['features'].shape),
                            'checkpoint_sha256':provenance['engine']['checkpoint_sha256'],
                            'time_origin':provenance['time_origin']}
                    write_json(record_path,record)
                    complete.append(record); counts['processed']+=1
                except (OSError,ValueError,RuntimeError,KeyError,FloatingPointError) as exc:
                    failed.append({'file':name,'split':row['split'],'type':type(exc).__name__,'error':str(exc)})
                    print(f'FLAGGED {name}: {exc}',file=sys.stderr)
                    # Stop a systemic compatibility error before repeatedly failing thousands of videos.
                    if len(failed)>=3 and not complete:
                        raise RuntimeError('First three videos failed. Fix the reported input/adapter issue before expanding extraction.') from exc
        except KeyboardInterrupt:
            interrupted=True
            print('\nInterrupted. Completed video embeddings are safe; rerun the same command.')
        finally:
            write_jsonl(output/'errors_latest.jsonl',failed)
            for split in splits:
                expected=[r for r in selected if r['split']==split]
                found=sorted([r for r in complete if r['split']==split],key=lambda r:r['file'])
                if len(found)==len(expected) and not interrupted:
                    dims={r['shape'][1] for r in found}
                    if len(dims)!=1: raise ValueError('Embedding dimension differs within a split.')
                    body={'format':VERSION,'encoder_id':encoder_id,'modality':modality,
                          'split':split,'partial_pilot':bool(limit),'test_only':False,'videos':found}
                    body['signature']=digest(body)
                    write_json(output/'splits'/f'{split}.json',body)
            report={'requested_videos':len(selected),'completed_videos':len(complete),**dict(counts),
                    'errors':len(failed),'interrupted':interrupted,'partial_pilot':bool(limit),
                    'modality':modality,'encoder_id':encoder_id,
                    'feature_steps':sum(r['shape'][0] for r in complete),'seconds':time.perf_counter()-start,
                    'note':'Embeddings only; no encoder fine-tuning, classification evaluation or accuracy claim.'}
            write_json(output/'summary.json',report); print(report)
    if interrupted: raise SystemExit(130)
    if failed or len(complete)!=len(selected): raise ValueError('Incomplete embedding extraction. No complete split manifest for failed splits.')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--modality',choices=['visual','audio'],required=True)
    p.add_argument('--splits',nargs='+',choices=SPLITS,default=['train','dev'])
    p.add_argument('--out');p.add_argument('--limit-per-split',type=int,default=0);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.limit_per_split<0: p.error('limit must be nonnegative')
    extract(a.config,a.modality,list(dict.fromkeys(a.splits)),a.out,a.limit_per_split,a.device)


if __name__=='__main__':
    try: main()
    except (ValueError,KeyError,ImportError,OSError,RuntimeError,FloatingPointError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        print('Keep strict loading checks. Unknown notebook checkpoint formats need an adapter; a filename alone is insufficient.',file=sys.stderr)
        raise SystemExit(2)
