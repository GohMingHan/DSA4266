"""Artifact regression checks: synthetic builds; approved index is read-only."""
from contextlib import redirect_stdout, closing
from fractions import Fraction
import hashlib
import os
import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
INDEX=Path(os.environ.get('DSA4266_INDEX',str(ROOT/'metadata/candidates.sqlite')))
POOL=Path(os.environ.get('DSA4266_POOL',str(ROOT/'metadata/train_pool.json')))
sys.path.insert(0,str(ROOT))
from src import config,data
from src.build_candidates import validate_candidate_index,compare_candidate_indexes,logical_index_report
from src.workflow import Inputs,prepare_manifests

TIMES=[Fraction(i,25) for i in range(76)]
EXPECTED={('train','frame'):(93,85,8),('dev','frame'):(62,54,8),
          ('train','clip'):(9,8,1),('dev','clip'):(6,5,1)}


def fixture_rows(root):
    rows=[]
    for split in ('dev','train'):
        for name,visual,audio in [('a',False,False),('b',True,False),*([('c',False,True)] if split=='train' else [])]:
            relative=split+'/'+name+'.mp4'
            path=root/relative;path.parent.mkdir(exist_ok=True);path.write_bytes(b'synthetic-placeholder')
            rows.append({'file':relative,'split':split,'original':None if name=='a' else split+'/a.mp4',
                         'modify_video':visual,'modify_audio':audio,'fake_periods':[[.4,1.2]] if name!='a' else [],
                         'video_frames':len(TIMES)})
    return rows


def synthetic_index(root,output):
    rows=fixture_rows(root)
    with patch.object(data,'train_dev_rows',return_value=rows),patch.object(data,'packet_timeline',return_value=TIMES),patch.object(data,'EXPECTED',EXPECTED),redirect_stdout(io.StringIO()):
        return data.build_index(root,output)


class SyntheticArtifacts(unittest.TestCase):
    def test_original_generator_labels_ids_timestamps_and_loader_records(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);path=synthetic_index(root,root/'index')
            with patch.object(data,'EXPECTED',EXPECTED):
                index=data.Index(path)
                try:
                    self.assertEqual(index.report['counts']['dev_frame'],{'total':62,'real':54,'fake':8})
                    rows=index.records('dev','frame');clips=index.records('dev','clip')
                    self.assertEqual(len(rows),62);self.assertEqual(len(clips),6)
                    self.assertEqual({r['split'] for r in rows},{'dev'})
                    for row in rows:
                        expected=int(row['source_file']=='dev/b.mp4' and Fraction('0.4')<=Fraction(row['timestamps_exact'][0])<Fraction('1.2'))
                        self.assertEqual(row['label'],expected)
                        self.assertEqual(Fraction(row['timestamps_exact'][0]),TIMES[row['indices'][0]])
                    self.assertEqual([r['id'] for r in rows],list(range(1,63)))
                    self.assertEqual([r['position'] for r in clips[:3]],[0,12,15])
                    audio=[r for r in index.records('train','frame') if r['source_file']=='train/c.mp4']
                    self.assertTrue(audio);self.assertTrue(all(r['label']==0 for r in audio))
                    dataset=data.LazyDataset(rows,root,4266,0,0.,0)
                    self.assertEqual(len(dataset),62)
                finally:index.close()

    def test_repeat_build_is_logically_and_deterministically_identical(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a=synthetic_index(root,root/'first');b=synthetic_index(root,root/'second')
            result=compare_candidate_indexes(a,b)
            self.assertTrue(result['equivalent']);self.assertFalse(result['differences'])
            before=a.read_bytes()
            with self.assertRaises(FileExistsError):data.build_index(root,a.parent)
            self.assertEqual(a.read_bytes(),before)

    def test_original_fixed_pool_builder_deterministic_and_loader_compatible(self):
        class Reader:
            def read(self,records):
                for row in records:
                    if row['split']!='train':raise AssertionError('TRAIN only')
                    if Fraction(row['timestamps_exact'][0])!=TIMES[row['indices'][0]]:raise AssertionError('Wrong PTS')
                return []
        with tempfile.TemporaryDirectory() as t,redirect_stdout(io.StringIO()):
            root=Path(t);rows=[r for r in fixture_rows(root) if r['split']=='train']
            paths=[data.build_pool(root,root/name,data.PoolConfig(per_class=4),rows=rows,
                                   timeline_fn=lambda *_:TIMES,reader=Reader()) for name in ('pool_a','pool_b')]
            self.assertEqual(paths[0].read_bytes(),paths[1].read_bytes())
            pool=data.FixedPool(paths[0],expected_per_class=4)
            self.assertEqual([sum(r['label']==y for r in pool.records()) for y in (0,1)],[4,4])
            sampler=data.FixedPoolSampler(pool,2,4266)
            first,_=sampler.select(0);second,_=sampler.select(1)
            self.assertFalse(set(first)&set(second));self.assertEqual(len(set(first+second)),8)
            self.assertEqual(len(data.LazyDataset(pool.records(first),root,4266,0,0.,0)),4)

    def test_reuse_never_builds_and_missing_artifact_fails_clearly(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);index=root/'metadata/candidates.sqlite';index.parent.mkdir();index.write_bytes(b'fixture')
            pool=root/'train_pool.json';pool.write_text('{}')
            inputs=Inputs(root,index,pool,root)
            with patch('src.workflow.validate_candidate_index'),patch('src.workflow.validate_inputs'),patch('src.workflow.build_candidates') as index_build,patch('src.workflow.build_pool') as pool_build:
                result=prepare_manifests(inputs,root)
                self.assertEqual(result.index,index);index_build.assert_not_called();pool_build.assert_not_called()
                with self.assertRaisesRegex(FileNotFoundError,'ARTIFACT_MODE=BUILD'):
                    prepare_manifests(Inputs(root,index,None,root),root)
                index_build.assert_not_called();pool_build.assert_not_called()
                with self.assertRaisesRegex(FileNotFoundError,'index missing'):
                    prepare_manifests(Inputs(root,root/'absent/candidates.sqlite',pool,root),root)
                with self.assertRaises(ValueError):prepare_manifests(inputs,root,artifact_mode='automatic')

    def test_explicit_build_only_missing_then_reuses(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);index=root/'working_index/candidates.sqlite'
            inputs=Inputs(root,index,None,root)
            def index_build(_,directory):
                directory.mkdir();index.write_bytes(b'fixture');return index
            def pool_build(_,directory,policy):
                self.assertEqual((policy.per_class,policy.seed),(10000,4266))
                directory.mkdir(parents=True);path=directory/'train_pool.json';path.write_text('{}');return path
            with patch('src.workflow.validate_candidate_index'),patch('src.workflow.validate_inputs'),patch('src.workflow.build_candidates',side_effect=index_build) as ib,patch('src.workflow.build_pool',side_effect=pool_build) as pb:
                prepare_manifests(inputs,root,artifact_mode='BUILD')
                prepare_manifests(inputs,root,artifact_mode='BUILD')
                self.assertEqual((ib.call_count,pb.call_count),(1,1))

    def test_invalid_existing_index_is_not_replaced_or_used_to_build_pool(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);index=root/'candidates.sqlite';index.write_bytes(b'invalid original must stay')
            with patch('src.workflow.build_candidates') as ib,patch('src.workflow.build_pool') as pb:
                with self.assertRaises(sqlite3.DatabaseError):
                    prepare_manifests(Inputs(root,index,None,root),root,artifact_mode='BUILD')
                ib.assert_not_called();pb.assert_not_called()
            self.assertEqual(index.read_bytes(),b'invalid original must stay')




@unittest.skipUnless((INDEX).is_file(),'Approved index intentionally absent from offline source ZIP')
class ApprovedIndex(unittest.TestCase):
    def test_copied_hash_complete_schema_splits_and_training_loaders(self):
        path=INDEX
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),config.INDEX_SHA256)
        report=validate_candidate_index(path)
        self.assertEqual(report['index_logical_sha256'],config.INDEX_LOGICAL_SHA256)
        logical=logical_index_report(path)
        self.assertEqual(logical['logical_sha256'],config.INDEX_LOGICAL_SHA256)
        self.assertEqual({k:v['rows'] for k,v in logical['tables'].items()},dict(info=4,videos=4500,frames=356804,clips=30368))
        index=data.DevSchedule(path)
        try:
            rows=index.records('dev','frame');clips=index.records('dev','clip')
            self.assertEqual((len(rows),sum(r['label']==0 for r in rows),sum(r['label'] for r in rows)),(41746,40115,1631))
            self.assertEqual(len({r['source_file'] for r in rows}),500);self.assertEqual(len(clips),3551)
            self.assertTrue(all(len(r['indices'])==16 for r in clips))
            videos=dict(index.db.execute('SELECT split,COUNT(*) FROM videos GROUP BY split'))
            self.assertEqual(videos,{'train':4000,'dev':500})
            self.assertEqual(index.db.execute('PRAGMA quick_check').fetchone()[0],'ok')
        finally:index.close()

    def test_equivalent_layout_is_accepted_but_changed_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'candidates.sqlite';shutil.copy2(INDEX,path)
            with closing(sqlite3.connect(path)) as db:
                db.execute('PRAGMA user_version=17');db.commit()
            self.assertNotEqual(hashlib.sha256(path.read_bytes()).hexdigest(),config.INDEX_SHA256)
            self.assertEqual(validate_candidate_index(path)['index_logical_sha256'],config.INDEX_LOGICAL_SHA256)
            self.assertTrue(compare_candidate_indexes(path,INDEX)['equivalent'])
            with closing(sqlite3.connect(path)) as db:
                real=db.execute('SELECT id FROM frames WHERE label=0 LIMIT 1').fetchone()[0]
                fake=db.execute('SELECT id FROM frames WHERE label=1 LIMIT 1').fetchone()[0]
                db.execute('UPDATE frames SET label=1-label WHERE id IN (?,?)',(real,fake))
                db.commit()
            # Counts remain identical: row-level identity must still reject this.
            with self.assertRaisesRegex(ValueError,'records.*labels differ'):validate_candidate_index(path)


@unittest.skipUnless((POOL).is_file(),'Approved TRAIN pool intentionally absent from offline source ZIP')
class ApprovedTrainPool(unittest.TestCase):
    def test_exact_hash_policy_labels_and_unique_records(self):
        path=POOL
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),config.POOL_SHA256)
        pool=data.FixedPool(path)
        self.assertEqual(pool.content['policy'],data.PoolConfig().__dict__)
        self.assertEqual((pool.content['policy']['seed'],len(pool.frames)),(4266,20000))
        records=pool.records()
        self.assertEqual([sum(r['label']==y for r in records) for y in (0,1)],[10000,10000])
        self.assertEqual(len({(r['source_file'],r['source_index']) for r in records}),20000)
        self.assertTrue(all(r['split']=='train' and r['source_file'].startswith('train/') and r['source_family'].startswith('train/') for r in records))

    def test_seed4266_rotation_is_reproducible_balanced_and_covers_pool(self):
        pool=data.FixedPool(POOL)
        a=data.FixedPoolSampler(pool,4000,4266);b=data.FixedPoolSampler(pool,4000,4266)
        covered=set()
        for epoch in range(3):
            ids,report=a.select(epoch);other,_=b.select(epoch)
            self.assertEqual(ids,other);self.assertEqual(len(set(ids)),8000)
            self.assertEqual((report['real'],report['fake']),(4000,4000))
            covered.update(ids)
        self.assertEqual(len(covered),20000)

    def test_explicit_repository_pool_reuses_without_any_builder(self):
        inputs=Inputs(ROOT,INDEX,POOL,ROOT)
        with patch('src.workflow.validate_candidate_index'),patch('src.workflow.validate_inputs'),patch('src.workflow.build_candidates') as ib,patch('src.workflow.build_pool') as pb:
            result=prepare_manifests(inputs,ROOT,artifact_mode='REUSE')
            self.assertEqual(result.pool,POOL)
            ib.assert_not_called();pb.assert_not_called()


if __name__=='__main__':unittest.main()
