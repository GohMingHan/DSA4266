"""Timestamp-aware preserved-RGB / pretrained-TSN-flow cache loader and explicit balanced-video sampler."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from common import digest, json_read, require_unchanged, within

FEATURE_KIND = 'preserved_resnet18_rgb512_pretrained_tsn_flow2048_v1'


def load_manifest(path):
    path = Path(path).expanduser().resolve()
    m = json_read(path)
    expected = m.get('manifest_sha256')
    if expected != digest({k: v for k, v in m.items() if k != 'manifest_sha256'}):
        raise ValueError('Manifest integrity check failed.')
    if m.get('target') != 'modify_video' or m.get('feature_kind') != FEATURE_KIND:
        raise ValueError('Expected an extract_tsn_flow.py cache with RGB512 + TSN-flow2048. Do not pass the old Farneback cache.')
    if m.get('input_dim') != 2560 or m.get('rgb_dim') != 512 or m.get('flow_dim') != 2048:
        raise ValueError('Unexpected TSN feature dimensions.')
    if not m.get('videos'):
        raise ValueError('Empty feature manifest.')
    m['_cache_root'] = str(path.parent)
    return m


def list_collate(batch):
    return batch


def resample_at_bin_centers(features, timestamps, duration, length=768):
    """Interpolate actual PTS; hold endpoint values outside observed feature centers.

    Ground truth is NEVER used here. Upsampling does not create new motion evidence.
    """
    ts = np.asarray(timestamps, np.float64)
    features = np.asarray(features, np.float32)
    if (features.ndim != 2 or len(features) != len(ts) or len(ts) < 2 or ts.ndim != 1
            or not np.isfinite(ts).all() or not np.isfinite(features).all()
            or (np.diff(ts) <= 0).any() or not math.isfinite(duration)
            or duration <= ts[-1] or ts[0] != 0 or length <= 0):
        raise ValueError('Invalid feature timeline.')
    centers = (np.arange(length, dtype=np.float64) + .5) * duration / length
    right = np.clip(np.searchsorted(ts, centers, side='right'), 1, len(ts)-1)
    left = right - 1
    mix = np.clip((centers - ts[left]) / (ts[right] - ts[left]), 0., 1.).astype(np.float32)
    return features[left] * (1 - mix[:, None]) + features[right] * mix[:, None]


class VisualDataset(Dataset):
    def __init__(self, manifest, split, length=768, input_mode='rgb-flow'):
        if input_mode not in ('rgb-flow', 'rgb'):
            raise ValueError('input_mode must be rgb-flow or rgb.')
        self.rows = [r for r in manifest['videos'] if r['split'] == split]
        self.length, self.input_mode = int(length), input_mode
        self.root = Path(manifest['_cache_root'])
        if not self.rows:
            raise ValueError(f'No {split} videos in manifest.')
        if length != 768:
            raise ValueError("This integration keeps the authors' 768-step temporal configuration.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = within(self.root, row['cache']['path'])
        require_unchanged(path, row['cache']['stat'])
        with np.load(path, allow_pickle=False) as z:
            rgb, flow, ts = z['rgb'], z['flow'], z['timestamp_s']
            if rgb.shape != (row['feature_count'], 512) or flow.shape != (row['feature_count'], 2048):
                raise ValueError(f'Feature shape changed: {path}')
            raw = np.concatenate([rgb, flow], axis=1) if self.input_mode == 'rgb-flow' else rgb
            features = resample_at_bin_centers(raw, ts, row['duration_s'], self.length)
        features = torch.from_numpy(np.ascontiguousarray(features.T))
        duration = row['duration_s']
        segments, labels = None, None
        if row['modify_video']:
            segments = torch.tensor(row['visual_periods'], dtype=torch.float32)
            segments = segments * (self.length / duration) - .5
            labels = torch.zeros(len(segments), dtype=torch.long)
        return {'video_id': row['file'], 'feats': features,
                'segments': segments, 'labels': labels,
                'av_labels': torch.tensor([int(row['modify_video']), 0], dtype=torch.long),
                'fps': 1.0, 'duration': duration,
                'feat_stride': duration / self.length, 'feat_num_frames': duration / self.length}


class MixedBatchSampler(Sampler):
    """Every batch has both visual classes, required by the released loss branches.

Both classes are shuffled. The smaller pool cycles; every video appears at least
once per epoch. This is not the authors' unconstrained shuffled sampler.
"""
    def __init__(self, rows, batch_size=4, seed=42):
        if batch_size < 2 or batch_size % 2:
            raise ValueError('Use an even batch size >= 2.')
        self.pools = [np.asarray([i for i, r in enumerate(rows) if int(r['modify_video']) == label])
                      for label in (0, 1)]
        self.half, self.seed, self.epoch = batch_size // 2, seed, 0
        if min(map(len, self.pools)) < self.half:
            raise ValueError('Each class needs at least batch_size/2 videos; lower batch size.')
        self.steps = math.ceil(max(map(len, self.pools)) / self.half)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        # Padding the final cycle may duplicate a video across batches, never in one batch.
        perm = [list(rng.permutation(pool)) for pool in self.pools]
        cursor = [0, 0]
        for _ in range(self.steps):
            batch = []
            for c in (0, 1):
                chosen = []
                while len(chosen) < self.half:
                    if cursor[c] >= len(perm[c]):
                        perm[c] = list(rng.permutation(self.pools[c]))
                        cursor[c] = 0
                    value = int(perm[c][cursor[c]])
                    cursor[c] += 1
                    if value not in chosen:
                        chosen.append(value)
                batch.extend(chosen)
            rng.shuffle(batch)
            yield batch
