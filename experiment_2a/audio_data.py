"""Matched datasets: same clip/window index, two numerical representations."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from common import SpectrogramDataset
from prepare_audio import repeat_pad


class WaveformDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], split: str, audio: dict):
        self.root, self.audio = root, audio
        self.length = round(audio['window_seconds'] * audio['sample_rate'])
        self.records = [r for r in records if r['split'] == split]
        self.index = [(i, j) for i, r in enumerate(self.records) for j in range(len(r['windows']))]
        self.labels = np.array([self.records[i]['windows'][j]['audio_label'] for i, j in self.index], dtype=np.int64)
        if not self.index:
            raise ValueError(f'No windows in split {split!r}.')

    @lru_cache(maxsize=4)
    def _read_clip(self, path: str) -> np.ndarray:
        x, sr = sf.read(path, dtype='float32', always_2d=False)
        if sr != self.audio['sample_rate'] or x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError(f'Invalid waveform: {path}')
        return x

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i, j = self.index[index]
        row, w = self.records[i], self.records[i]['windows'][j]
        x = self._read_clip(str(self.root / row['wav_path']))
        start, end = w['start_sample'], w['end_sample']
        if not 0 <= start < end <= len(x) or end - start != w['valid_samples']:
            raise ValueError(f'Invalid window offsets for {row["file"]}.')
        padded = repeat_pad(x[start:end], self.length)
        return torch.from_numpy(padded), torch.tensor(float(w['audio_label'])), index


def make_dataset(model_name, root, records, split, stats, audio):
    if model_name == 'resnet18':
        if stats is None:
            raise ValueError('ResNet requires training-only spectrogram normalization statistics.')
        return SpectrogramDataset(root, records, split, stats, audio)
    if model_name == 'aasist':
        return WaveformDataset(root, records, split, audio)
    raise ValueError(f'Unknown model: {model_name}')
