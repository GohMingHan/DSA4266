"""Explicit original artifact construction and SQLite logical-identity validation.

Importing this module never builds or changes an artifact. No linked imports.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3

from . import config
from .data import build_index, build_pool, PoolConfig, train_dev_rows, rows_hash, validate_index, write_json


def _file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(2**20),b''):h.update(block)
    return h.hexdigest()


def _readonly(path):
    path=Path(path).resolve()
    if not path.is_file():raise FileNotFoundError('Candidate database is missing: '+str(path))
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    db.execute('PRAGMA query_only=ON')
    return db


def logical_index_report(path):
    """Stream all schema/rows in primary-key order; ignore SQLite page layout.

    JSON fields are canonicalized, including exact PTS strings and interval arrays.
    Every ID, timestamp, label, source reference and source byte count is hashed.
    """
    db=_readonly(path)
    try:
        if {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}!={'info','videos','frames','clips'}:
            raise ValueError('Unexpected candidate schema tables')
        schema=[{'type':kind,'name':name,'table':table,'sql':' '.join(sql.split()) if sql else None}
                for kind,name,table,sql in db.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name")]
        tables={}
        for table,order in [('info','key'),('videos','id'),('frames','id'),('clips','id')]:
            cursor=db.execute(f'SELECT * FROM {table} ORDER BY {order}')
            columns=[c[0] for c in cursor.description]
            h=hashlib.sha256();count=0;first=None;middle=[];last=None
            for raw in cursor:
                row=list(raw)
                for i,name in enumerate(columns):
                    if table=='videos' and name in ('periods','pts','sampled_indices'):
                        row[i]=json.loads(row[i])
                if table=='info' and row[0]=='policy':row[1]=json.loads(row[1])
                packed=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                h.update(packed+b'\n');count+=1
                sample={'id':row[0],'sha256':hashlib.sha256(packed).hexdigest()}
                if first is None:first=sample
                if count%10000==0:middle=[sample]
                last=sample
            samples=[]
            for item in [first,*middle,last]:
                if item is not None and item not in samples:samples.append(item)
            tables[table]={'columns':columns,'rows':count,'sha256':h.hexdigest(),'sample_records':samples}
        value={'schema':schema,'tables':{k:{field:v[field] for field in ('columns','rows','sha256')} for k,v in tables.items()}}
        identity=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return {'logical_sha256':identity,'schema':schema,'tables':tables,'test_used':False}
    finally:db.close()


@lru_cache(maxsize=8)
def _checked_identity(path,size,mtime_ns,file_hash):
    # Cache only immutable file identities; a changed file has a different key.
    db=_readonly(path)
    try:
        info=dict(db.execute('SELECT key,value FROM info'))
        if info.get('protocol')!='cnn_v2_pts10_clip16_stride12_one_fake' or info.get('complete')!='true':
            raise ValueError('Candidate database is incomplete or has a different protocol')
        counts=validate_index(db)
    finally:db.close()
    expected=config.INDEX_LOGICAL_SHA256
    if file_hash==config.INDEX_SHA256:
        logical=expected
    else:
        logical=logical_index_report(path)['logical_sha256']
        if logical!=expected:
            raise ValueError('Candidate schema, records, deterministic IDs or labels differ from the approved original index')
    return {'index_sha256':file_hash,'index_logical_sha256':logical,
            'reference_index_sha256':config.INDEX_SHA256,
            'train_dev_metadata_sha256':info['train_dev_metadata_sha256'],**counts}


def validate_candidate_index(path,data_root=None):
    """Accept the approved bytes or a fully logically equivalent rebuilt index."""
    path=Path(path).resolve()
    if not path.is_file():raise FileNotFoundError('Candidate database is missing: '+str(path))
    stat=path.stat();file_hash=_file_sha256(path)
    result=dict(_checked_identity(str(path),stat.st_size,stat.st_mtime_ns,file_hash))
    if data_root is not None and rows_hash(train_dev_rows(data_root))!=result['train_dev_metadata_sha256']:
        raise ValueError('Original TRAIN/DEV annotations or split definitions differ')
    return result


def compare_candidate_indexes(candidate,reference):
    """Compare complete schema and every row, plus deterministic sample hashes."""
    a=logical_index_report(candidate);b=logical_index_report(reference)
    differences=[]
    if a['schema']!=b['schema']:differences.append('schema')
    for table in ('info','videos','frames','clips'):
        if a['tables'][table]!=b['tables'][table]:differences.append(table)
    return {'equivalent':not differences,'differences':differences,
            'candidate_sha256':_file_sha256(candidate),'reference_sha256':_file_sha256(reference),
            'candidate':a,'reference':b,'test_used':False}


def build_candidates(data_root,output,*,reference=None):
    """Explicit build into a NEW directory with the exact original build_index."""
    path=build_index(data_root,output)
    report=validate_candidate_index(path,data_root)
    if reference is not None:
        comparison=compare_candidate_indexes(path,reference)
        if not comparison['equivalent']:raise ValueError('Built index differs from the supplied reference')
        report['equivalence']=comparison
    else:
        report['logical_validation']='All rows/schema match the approved original logical fingerprint'
    write_json(Path(output)/'compatibility_report.json',report)
    return path


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Explicit TRAIN/DEV artifact build; never trains or reads TEST videos.')
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New output directory; existing directories are refused')
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--build-pool',action='store_true',help='Also build the unchanged fixed20k TRAIN pool')
    args=parser.parse_args()
    path=build_candidates(args.data_root,args.output,reference=args.reference)
    print('Candidate index:',path,flush=True)
    if args.build_pool:
        pool=build_pool(args.data_root,args.output/'pool',PoolConfig())
        if _file_sha256(pool)!=config.POOL_SHA256:raise ValueError('Fixed20k pool differs from the approved original hash')
        print('Fixed TRAIN pool:',pool,flush=True)


if __name__=='__main__':main()
