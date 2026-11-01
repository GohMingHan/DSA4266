"""Shared metadata, normalization, sampling and evaluation for Experiment 2A.

Targets: acoustic manipulation, not 'any kind of fake video'. The 512-D embeddings
can subsequently be joined to visual features using video IDs and timestamps.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from prepare_audio import read_jsonl, write_json, write_jsonl


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_signature(records: list[dict], split: str) -> str:
    # Include labels and window intervals, not just source path metadata.
    data = sorted([r for r in records if r["split"] == split], key=lambda r: r["clip_id"])
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()



def load_records(root: Path) -> tuple[list[dict], dict]:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("pipeline_version") != 2 or config["audio"].get("padding") != "repeat":
        raise ValueError("Use a NEW cache prepared by this Experiment 2A package.")
    records = read_jsonl(root / "manifest.jsonl")
    groups, ids = {}, set()
    for r in records:
        if r["clip_id"] in ids:
            raise ValueError("Duplicate clip in manifest.")
        ids.add(r["clip_id"])
        if r["config_id"] != config["config_id"]:
            raise ValueError("Mixed preprocessing configurations.")
        group = r["source_group"]
        if group in groups and groups[group] != r["split"]:
            raise ValueError("Source group crosses split boundaries.")
        groups[group] = r["split"]
        if not r.get("wav_path") or not (root / r["wav_path"]).is_file():
            raise FileNotFoundError(f"Missing waveform for {r['file']}; rerun prepare_audio.py.")
        if not (root / r["feature_path"]).is_file():
            raise FileNotFoundError(root / r["feature_path"])
    if not records:
        raise ValueError("Empty manifest.")
    return records, config


def valid_columns(window: dict, ncols: int, audio: dict) -> np.ndarray:
    if audio.get("padding") == "repeat":
        return np.ones(ncols, dtype=bool)
    return np.arange(ncols) * audio["hop_length"] < window["valid_samples"]


def training_statistics(root: Path, records: list[dict], audio: dict) -> dict:
    """Per-mel-band moments from TRAIN only. Exclude fully padded columns."""
    total = np.zeros(audio["n_mels"], dtype=np.float64)
    squares = total.copy()
    count = 0
    train = [r for r in records if r["split"] == "train"]
    if not train:
        raise ValueError("No training records.")
    for r in tqdm(train, desc="Train-only normalization", unit="video"):
        with np.load(root / r["feature_path"], allow_pickle=False) as f:
            x = f["log_mel"].astype(np.float64)
        if x.shape != tuple(r["shape"]) or not np.isfinite(x).all():
            raise ValueError(f"Corrupt feature array: {r['file']}")
        for j, window in enumerate(r["windows"]):
            z = x[j][:, valid_columns(window, x.shape[-1], audio)]
            total += z.sum(axis=1)
            squares += np.square(z).sum(axis=1)
            count += z.shape[1]
    mean = total / count
    variance = np.maximum(squares / count - mean**2, 0)
    std = np.maximum(np.sqrt(variance), 1.0)  # do not amplify near-constant bands
    return {"mean": mean.tolist(), "std": std.tolist(), "columns": count,
            "computed_from": "train_only", "minimum_std_db": 1.0}


class SpectrogramDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], split: str, stats: dict, audio: dict):
        self.root, self.audio = root, audio
        self.records = [r for r in records if r["split"] == split]
        self.index = [(i, j) for i, r in enumerate(self.records) for j in range(len(r["windows"]))]
        self.mean = np.asarray(stats["mean"], np.float32)[:, None]
        self.std = np.asarray(stats["std"], np.float32)[:, None]
        self.labels = np.array([self.records[i]["windows"][j]["audio_label"] for i, j in self.index], dtype=np.int64)
        if not self.index:
            raise ValueError(f"No windows in split {split!r}.")

    @lru_cache(maxsize=4)
    def _read_clip(self, path: str) -> np.ndarray:
        with np.load(path, allow_pickle=False) as f:
            x = f["log_mel"].astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite spectrogram in {path}")
        return x

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        i, j = self.index[index]
        row, window = self.records[i], self.records[i]["windows"][j]
        x = self._read_clip(str(self.root / row["feature_path"]))[j]
        z = (x - self.mean) / self.std
        # Pad with standardized zero; do not let arbitrary padded dB values dominate.
        z[:, ~valid_columns(window, x.shape[-1], self.audio)] = 0
        return torch.from_numpy(np.ascontiguousarray(z[None], dtype=np.float32)), torch.tensor(float(window["audio_label"])), index


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.isdigit():
        value = f"cuda:{value}"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Use --device cpu or install a compatible PyTorch build.")
    return torch.device(value)


def loader(dataset: Dataset, batch_size: int, workers: int, shuffle=False, sampler=None) -> DataLoader:
    # workers=0 is the portable default, including Windows.
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                      shuffle=shuffle if sampler is None else False, sampler=sampler,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def aggregate(dataset: SpectrogramDataset, p: np.ndarray, pooling: str) -> list[dict]:
    by_video = defaultdict(list)
    for k, (i, _) in enumerate(dataset.index):
        by_video[i].append(float(p[k]))
    result = []
    for i, scores in by_video.items():
        r = dataset.records[i]
        result.append({"clip_id": r["clip_id"], "file": r["file"], "source_group": r["source_group"],
                       "audio_label": r["audio_video_label"], "overall_video_label": r["overall_video_label"],
                       "modify_video": r["modify_video"], "modify_audio": r["modify_audio"],
                       "score": max(scores) if pooling == "max" else float(np.mean(scores)),
                       "n_windows": len(scores)})
    return result


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both classes in development data.")
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    # Highest threshold on an exact tie, for deterministic behavior.
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def equal_error_rate(y: np.ndarray, p: np.ndarray) -> float | None:
    """Linear interpolation of the empirical ROC at FPR=FNR; a diagnostic only."""
    if len(np.unique(y)) != 2:
        return None
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
    fnr = 1 - tpr
    d = fpr - fnr
    j = int(np.flatnonzero(d >= 0)[0])
    if j == 0 or d[j] == 0:
        return float(fpr[j])
    a = -d[j - 1] / (d[j] - d[j - 1])
    return float(fpr[j - 1] + a * (fpr[j] - fpr[j - 1]))


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Empty, mismatched or nonfinite predictions.")
    predicted = (p >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    return {"n": int(len(y)), "positive_prevalence": float(np.mean(y)),
            "eer_interpolated": equal_error_rate(y, p),
            "threshold": threshold, "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if both else None,
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "average_precision": float(average_precision_score(y, p)) if both else None,
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "confusion_matrix_labels_0_1": confusion_matrix(y, predicted, labels=[0, 1]).tolist()}


def evaluate_predictions(ds: SpectrogramDataset, p: np.ndarray, pooling: str,
                         window_threshold: float, video_threshold: float) -> tuple[dict, list[dict]]:
    videos = aggregate(ds, p, pooling)
    vy = np.array([r["audio_label"] for r in videos])
    vp = np.array([r["score"] for r in videos])
    by_category = {}
    for name, ma, mv in [("real", False, False), ("visual_only", False, True),
                         ("audio_only", True, False), ("both", True, True)]:
        selected = [r for r in videos if r["modify_audio"] == ma and r["modify_video"] == mv]
        by_category[name] = {"n": len(selected), "predicted_fake_fraction":
            float(np.mean([r["score"] >= video_threshold for r in selected])) if selected else None}
    return {"target": "audio_manipulation", "pooling": pooling,
            "window": metrics(ds.labels, p, window_threshold),
            "video_audio": metrics(vy, vp, video_threshold), "by_category": by_category}, videos

