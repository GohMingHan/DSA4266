"""Session modes/ordered clip pixel reuse, with real tiny videos and no training."""
from contextlib import redirect_stdout,ExitStack
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import session_inputs as session
from src.frame_cache import RGBCache,CachedReader
from src.prepared_frames import IDENTITY,require_prepared_cache
from src.clip_model import CLIPConfig,make_clip_loader
from src.data import FrameReader,LazyDataset
from src.config import configuration


def fixture(root):
    import av
    rows=[];videos={};clips=[]
    for split in ['train','dev']:
        (root/split).mkdir()
        path=root/split/'source.mkv'
        with av.open(str(path),'w') as container:
            stream=container.add_stream('ffv1',rate=10);stream.width=stream.height=224;stream.pix_fmt='bgr0'
            for i in range(30):
                pixels=np.zeros((224,224,3),dtype=np.uint8);pixels[:,:,0]=i*7;pixels[:,112:,1]=100
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels,format='rgb24')):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        v={'source_file':split+'/source.mkv','source_family':split+'/family','periods':[['1','11/10']], 'video_label':1}
        videos[v['source_file']]=v
        for i in range(30):
            rows.append({'id':len(rows),'split':split,'kind':'frame','source_file':v['source_file'],'source_family':v['source_family'],
                         'video_label':1,'label':int(i==10),'indices':[i],'timestamps_exact':[str(session.Fraction(i,10))]})
        for start in [0,12]:
            indices=list(range(start,start+16))
            clips.append({'id':len(clips),'split':split,'kind':'clip','source_file':v['source_file'],'source_family':v['source_family'],
                          'video_label':1,'label':int(10 in indices),'indices':indices,'timestamps_exact':[str(session.Fraction(i,10)) for i in indices]})
    (root/'index').write_bytes(b'index');(root/'pool').write_bytes(b'pool')
    return SimpleNamespace(data=root,index=root/'index',pool=root/'pool'),rows,videos,clips


class SessionModes(unittest.TestCase):
    def test_explicit_streaming_validates_scope_and_allows_original_clip_loader(self):
        from transformers import CLIPImageProcessor
        from src.prepared_frames import verify_clip_cache_loader
        with tempfile.TemporaryDirectory() as t,redirect_stdout(io.StringIO()):
            root=Path(t);inputs,rows,_,_=fixture(root);cache=RGBCache(root/'cache',IDENTITY,max_gib=.01)
            with patch.object(session,'_load_records',return_value=rows):
                report=session.validate_streaming_frames(inputs,cache,ram_cache_mib=1,reserve_gib=0)
            report.update(total_frames=61746,index_sha256=session.digest(inputs.index),pool_sha256=session.digest(inputs.pool))
            self.assertEqual(require_prepared_cache(cache,inputs,announce=False)['mode'],'streaming')
            loader=make_clip_loader(rows[:8],root,CLIPImageProcessor(),CLIPConfig(),cache=cache)
            probe=verify_clip_cache_loader(loader)
            self.assertEqual(probe['mode'],'validated streaming');self.assertLessEqual(loader.dataset.reader.reader.bytes,2**20)
            self.assertFalse(cache.require_hits)
            cache.streaming_verified=False
            with self.assertRaises(RuntimeError):require_prepared_cache(cache,inputs)

    def test_no_automatic_fallback_or_corruption_skip(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            with patch.object(session,'prepare_frame_dataset',side_effect=session.PreparationStorageError('no space')),patch.object(session,'validate_streaming_frames') as streaming:
                with self.assertRaisesRegex(session.PreparationStorageError,'explicitly'):session.prepare_frames(None,root,reserve_gib=0)
                streaming.assert_not_called()
            inputs,rows,_,_=fixture(root);cache=RGBCache(root/'stream',IDENTITY,max_gib=.01)
            with patch.object(session,'_load_records',return_value=rows),redirect_stdout(io.StringIO()):session.validate_streaming_frames(inputs,cache,reserve_gib=0)
            key,_=cache.key(rows[0],0,'0',root)
            (cache.root/key[:2]/(key+'.png')).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):CachedReader(root,cache).read([rows[0]])

    def test_disk_reserve_stops_writes_but_not_valid_streaming(self):
        with tempfile.TemporaryDirectory() as t,redirect_stdout(io.StringIO()):
            root=Path(t);inputs,rows,_,_=fixture(root);cache=RGBCache(root/'cache',IDENTITY,max_gib=.01)
            cache.disk_reserve_bytes=10**20
            pixels=FrameReader(root).read([rows[0]])[0][0]
            key,meta=cache.key(rows[0],0,'0',root)
            self.assertFalse(cache.put(key,meta,pixels));self.assertFalse(list(cache.root.rglob('*.png')))
            reader=CachedReader(root,cache)
            actual=reader.read([rows[0]])[0][0];np.testing.assert_array_equal(actual,pixels)
            self.assertGreater(cache.writes_skipped,0)

    def test_clip_order_labels_normalization_flip_and_shared_pixels_without_copies(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);inputs,rows,videos,clips=fixture(root)
            base=RGBCache(root/'base',IDENTITY,max_gib=.01);extra=RGBCache(root/'extra',IDENTITY,max_gib=.01)
            decoded=FrameReader(root).read(rows)
            for row,images in zip(rows,decoded):
                key,meta=base.key(row,row['indices'][0],row['timestamps_exact'][0],root);base.put(key,meta,images[0])
            index=SimpleNamespace(path=root/'manifest',sha256='fixture',videos={i:v for i,v in enumerate(videos.values())})
            pipeline=session.SessionClipPipeline(base,extra,index,ram_cache_mib=1)
            cfg=configuration('resnet50_lstm',amp=False)
            for training,selected in [(True,clips[:2]),(False,clips[2:])]:
                dataset=pipeline.dataset(selected,root,cfg,training=training,epoch=2)
                original=LazyDataset(selected,root,4266,2,.5 if training else 0.,0)
                with patch.object(dataset.reader.reader,'read',side_effect=AssertionError('all pixels shared')):
                    actual=dataset.__getitems__([0,1])
                expected=original.__getitems__([0,1])
                for a,b in zip(actual,expected):
                    torch.testing.assert_close(a[0],b[0],rtol=0,atol=0);self.assertEqual(a[1].item(),b[1].item())
                self.assertEqual(tuple(actual[0][0].shape),(16,3,224,224))
            self.assertEqual(extra.report()['entries'],0);self.assertIsNone(pipeline.identity)
            pickle.loads(pickle.dumps(pipeline))

    def test_overlapping_clips_decode_unique_pixels_and_reuse_overlay_across_readers(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);_,rows,videos,clips=fixture(root);base=RGBCache(root/'base',IDENTITY,max_gib=.01);extra=RGBCache(root/'extra',IDENTITY,max_gib=.01)
            pipeline=session.SessionClipPipeline(base,extra,SimpleNamespace(path=root/'manifest',sha256='fixture',videos={i:v for i,v in enumerate(videos.values())}),ram_cache_mib=1)
            reader=session.SessionClipReader(root,pipeline);result=reader.read(clips[:2])
            self.assertEqual(reader.decoded_missing,28);self.assertEqual(extra.report()['entries'],28)
            other=session.SessionClipReader(root,pipeline)
            with patch.object(other.reader,'read',side_effect=AssertionError('must reuse disk pixels')):again=other.read(clips[:2])
            for a,b in zip(result,again):
                for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
            self.assertEqual(other.extra_hits,28)
            base.secondary_cache=extra;base.require_hits=False
            # The corresponding CNN frame reuses the same physical overlay PNG,
            # without copying it to the base cache or using a clip-level label.
            frame_reader=CachedReader(root,base)
            with patch.object(frame_reader.reader,'read',side_effect=AssertionError('share overlay')):frame_reader.read([rows[0]])
            self.assertEqual(base.report()['entries'],0)


    def test_temporal_training_adapter_keeps_config_and_checkpoint_preprocessing_identity(self):
        from src.workflow import Inputs
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'configs').mkdir();(root/'configs/project.json').write_text('{"project_id":"fixture"}')
            manifest=root/'clips.json';manifest.write_text('{}')
            inputs=Inputs(root,root/'index',root/'pool',root/'weights',manifest)
            pipeline=SimpleNamespace(manifest_sha256=session.digest(manifest),identity=None)
            def fake_train(cfg,index,data,encoder,out,resume,**kwargs):
                self.assertEqual(cfg.as_dict(),configuration('resnet50_lstm',amp=False,workers=0).as_dict())
                self.assertIs(kwargs['input_pipeline'],pipeline);self.assertIsNone(kwargs['input_pipeline'].identity)
                self.assertEqual(index,manifest);self.assertEqual(encoder,root/'best.pt');self.assertIsNone(resume)
                Path(out).mkdir();return Path(out)
            with patch('src.workflow.new_encoder',return_value=root/'best.pt'),patch('src.workflow.preflight'),patch('src.training.train',side_effect=fake_train),patch('src.reporting.export_report'):
                result=session.execute_session_lstm(['resnet50_lstm'],inputs,root,clip_pipeline=pipeline,
                    confirmation='TRAIN_SELECTED_MODELS',cnn_best=root/'best.pt',amp=False,workers=0)
            self.assertIn('resnet50_lstm',result)


if __name__=='__main__':unittest.main()
