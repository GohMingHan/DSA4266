"""Small real-video preparation/strict-cache tests; never train a model or read TEST."""
import ast
from contextlib import redirect_stdout,ExitStack
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import prepared_frames as prepared
from src.frame_cache import RGBCache,CachedReader
from src.data import FrameReader
from src.checkpoints import digest


def fixture(root):
    import av
    rows=[]
    for split in ['train','dev']:
        (root/split).mkdir()
        with av.open(str(root/split/'tiny.mkv'),'w') as container:
            stream=container.add_stream('ffv1',rate=10);stream.width=stream.height=224;stream.pix_fmt='bgr0'
            for i in range(4):
                image=np.full((224,224,3),i*40+10,dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image,format='rgb24')):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        for i in [0,2]:
            rows.append({'id':len(rows),'kind':'frame','split':split,'source_file':split+'/tiny.mkv','source_family':split+'/family',
                         'indices':[i],'timestamps_exact':['0' if i==0 else '1/5'],'label':int(i==2),'video_label':1})
    (root/'index').write_bytes(b'synthetic index');(root/'pool').write_bytes(b'synthetic pool')
    return SimpleNamespace(data=root,index=root/'index',pool=root/'pool'),rows


def synthetic_scope(inputs,rows):
    stack=ExitStack();stack.enter_context(patch.object(prepared,'_load_records',return_value=rows))
    stack.enter_context(patch.object(prepared,'INDEX_SHA256',digest(inputs.index)))
    stack.enter_context(patch.object(prepared,'POOL_SHA256',digest(inputs.pool)))
    return stack


class PreparedRGB(unittest.TestCase):
    def test_production_scope_is_exact_existing_61746_records(self):
        index=Path(os.environ.get('DSA4266_INDEX',str(ROOT/'metadata/candidates.sqlite')))
        pool=Path(os.environ.get('DSA4266_POOL',str(ROOT/'metadata/train_pool.json')))
        if not index.is_file() or not pool.is_file():self.skipTest('External reference metadata not provided')
        rows=prepared._load_records(SimpleNamespace(index=index,pool=pool,data=None))
        self.assertEqual(len(rows),61746)
        self.assertEqual(sum(r['split']=='train' for r in rows),20000)
        self.assertEqual(sum(r['split']=='dev' for r in rows),41746)
        self.assertTrue(all(r['kind']=='frame' and r['split'] in ('train','dev') for r in rows))

    def test_stream_exact_pixels_pts_and_one_open_per_source(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);inputs,rows=fixture(root);cache=RGBCache(root/'cache',prepared.IDENTITY,max_gib=.01)
            expected=FrameReader(root).read(rows)
            with synthetic_scope(inputs,rows),redirect_stdout(io.StringIO()):report=prepared.prepare_frame_dataset(inputs,cache,reserve_gib=0)
            progress=json.loads((cache.root/'preparation_progress.json').read_text())
            self.assertTrue(report['ready']);self.assertEqual(progress['source_video_opens'],2)
            self.assertEqual((report['train_frames'],report['dev_frames']),(2,2))
            reader=CachedReader(root,cache)
            with patch.object(reader.reader,'read',side_effect=AssertionError('must not decode')):actual=reader.read(rows)
            for a,b in zip(expected,actual):np.testing.assert_array_equal(a[0],b[0])
            self.assertTrue(cache.require_hits)
            with synthetic_scope(inputs,rows),patch('av.open',side_effect=AssertionError('complete cache must not open videos')),redirect_stdout(io.StringIO()):prepared.prepare_frame_dataset(inputs,cache,reserve_gib=0)
            self.assertEqual(json.loads((cache.root/'preparation_progress.json').read_text())['source_video_opens'],0)
            # A synthetic four-record fixture cannot satisfy the production train gate.
            with self.assertRaises(ValueError):prepared.require_prepared_cache(cache,inputs)

    def test_interruption_and_resume_preserve_verified_images(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);inputs,rows=fixture(root);cache=RGBCache(root/'cache',prepared.IDENTITY,max_gib=.01)
            original=cache.put;calls=0
            def interrupted(*args):
                nonlocal calls
                calls+=1
                if calls==2:raise KeyboardInterrupt('synthetic interruption')
                return original(*args)
            with synthetic_scope(inputs,rows),patch.object(cache,'put',side_effect=interrupted),redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):prepared.prepare_frame_dataset(inputs,cache,reserve_gib=0)
            self.assertFalse(getattr(cache,'prepared_verified',False));self.assertFalse((cache.root/prepared.READY).exists())
            png=next(cache.root.rglob('*.png'));before=(digest(png),png.stat().st_mtime_ns)
            with synthetic_scope(inputs,rows),redirect_stdout(io.StringIO()):prepared.prepare_frame_dataset(inputs,cache,reserve_gib=0)
            progress=json.loads((cache.root/'preparation_progress.json').read_text())
            self.assertEqual((progress['new_frames'],progress['reused_verified_frames']),(3,1))
            self.assertEqual((digest(png),png.stat().st_mtime_ns),before)

    def test_corrupt_metadata_missing_png_and_disk_fail_do_not_mark_ready(self):
        for failure in ['disk','capacity','pts','corrupt','missing','metadata']:
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as t:
                root=Path(t);inputs,rows=fixture(root);cache=RGBCache(root/'cache',prepared.IDENTITY,max_gib=.01)
                with synthetic_scope(inputs,rows),redirect_stdout(io.StringIO()):
                    if failure in ['corrupt','missing','metadata']:
                        prepared.prepare_frame_dataset(inputs,cache,reserve_gib=0)
                        if failure=='corrupt':next(cache.root.rglob('*.png')).write_bytes(b'bad')
                        elif failure=='missing':next(cache.root.rglob('*.png')).unlink()
                        else:
                            with cache.connection() as db:
                                key,raw=db.execute('SELECT key,metadata FROM frames LIMIT 1').fetchone()
                                meta=json.loads(raw);meta['label']=1-meta['label'];db.execute('UPDATE frames SET metadata=? WHERE key=?',(json.dumps(meta),key))
                    if failure=='capacity':cache.limit=1
                    if failure=='pts':rows[0]['timestamps_exact']=['1/100']
                    with ExitStack() as stack:
                        if failure=='disk':stack.enter_context(patch.object(prepared.shutil,'disk_usage',return_value=SimpleNamespace(free=0)))
                        with self.assertRaises((ValueError,OSError)):prepared.prepare_frame_dataset(inputs,cache,reserve_gib=0)
                    self.assertFalse(cache.prepared_verified)
                    self.assertFalse(json.loads((cache.root/'preparation_progress.json').read_text())['ready'])

    def test_readonly_attach_export_and_clip_reader_no_video_decode(self):
        from src.clip_model import make_clip_loader,CLIPConfig
        from transformers import CLIPImageProcessor
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);inputs,rows=fixture(root);cache=RGBCache(root/'cache',prepared.IDENTITY,max_gib=.01)
            with synthetic_scope(inputs,rows),redirect_stdout(io.StringIO()):prepared.prepare_frame_dataset(inputs,cache,reserve_gib=0)
            # Export operates generically on verified files; production count gate is separately tested.
            with patch.object(prepared,'require_prepared_cache'),redirect_stdout(io.StringIO()):receipt=prepared.export_prepared_frames(cache,inputs,root/'export.zip',reserve_gib=0)
            import zipfile
            attached=root/'attached'
            with zipfile.ZipFile(receipt['archive']) as z:z.extractall(attached)
            readonly=RGBCache(attached,prepared.IDENTITY,readonly=True)
            with synthetic_scope(inputs,rows),patch('av.open',side_effect=AssertionError('must not decode attached frames')),redirect_stdout(io.StringIO()):
                prepared.prepare_frame_dataset(inputs,readonly,report_dir=root/'verify',reserve_gib=0)
                loader=make_clip_loader(rows[:2],root,CLIPImageProcessor(),CLIPConfig(),cache=readonly)
                probe=prepared.verify_clip_cache_loader(loader)
            self.assertEqual(probe['cache_hits'],2);self.assertEqual(probe['video_opens'],0)
            self.assertFalse(receipt['uploaded']);self.assertEqual(receipt['visibility_required'],'PRIVATE')
            # Strict runtime cache never silently decodes an uncached frame.
            other=rows[0]|{'indices':[1],'timestamps_exact':['1/10']}
            with self.assertRaises(FileNotFoundError):CachedReader(root,readonly).read([other])

    def test_no_cache_blocks_training_without_cuda_or_model_initialization(self):
        from src.clip_workflow import train_clip
        from src.config import POOL_SHA256,INDEX_LOGICAL_SHA256
        result=SimpleNamespace(testsRun=1,wasSuccessful=lambda:True)
        report={'pool_sha256':POOL_SHA256,'index_logical_sha256':INDEX_LOGICAL_SHA256,'dev_frames':41746,'dev_videos':500}
        with patch('src.clip_workflow.require_cuda',side_effect=AssertionError('must not reach CUDA')):
            with self.assertRaisesRegex(RuntimeError,'Prepare TRAIN/DEV'):train_clip(None,None,None,confirmation='TRAIN_clip_vit_b32_shared',test_result=result,data_report=report)

    def test_progress_records_current_batch_before_fetch_and_saves_timings(self):
        from src.clip_workflow import clip_epoch
        from src.clip_model import CLIPConfig
        from src.training import RunProgress
        class Model(torch.nn.Module):
            def forward(self,x):return torch.stack((x[:,0,0,0],-x[:,0,0,0]),dim=1)
        with tempfile.TemporaryDirectory() as t,redirect_stdout(io.StringIO()):
            with RunProgress(Path(t),'CLIP',heartbeat_seconds=0) as progress:
                class Loader:
                    dataset=range(6)
                    def __len__(self):return 3
                    def __iter__(self):
                        for batch in range(1,4):
                            self_test.assertEqual(progress.state['batch'],batch)
                            self_test.assertEqual(progress.state['stage'],'clip_dev_fetch_batch')
                            yield torch.tensor([-1.,1.]).view(2,1,1,1),torch.tensor([0.,1.]),torch.tensor([2*(batch-1),2*(batch-1)+1])
                self_test=self
                metrics,_,_=clip_epoch(Model(),Loader(),CLIPConfig(amp=False),progress=progress,device='cpu')
            timings=[json.loads(line) for line in (Path(t)/'batch_timings.jsonl').read_text().splitlines()]
            self.assertEqual([r['batch'] for r in timings],[1,2,3]);self.assertTrue(all(r['loading_seconds']>=0 and r['computation_seconds']>=0 for r in timings))
            self.assertEqual(metrics['roc_auc'],1.)



if __name__=='__main__':unittest.main()
