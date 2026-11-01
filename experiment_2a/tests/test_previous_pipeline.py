from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_audio import (AudioConfig, extract_wav, load_metadata, log_mel, make_windows,
                            merge_periods, normalize_metadata, overlap_seconds,
                            probe_timing, window_starts)
from models import AudioResNet18
from common import best_f1_threshold, metrics, valid_columns


def metadata(**changes):
    value = {"file": "videos/a.mp4", "split": "train", "duration": 3.0,
             "original": None, "modify_audio": True, "modify_video": False,
             "fake_periods": [[1.0, 1.4]]}
    return normalize_metadata({**value, **changes})


def test_invalid_boolean_not_silently_true():
    with pytest.raises(ValueError):
        metadata(modify_audio="false")


def test_missing_interval_not_silently_negative():
    with pytest.raises(ValueError):
        metadata(fake_periods=[])


def test_source_groups_cannot_cross_splits(tmp_path):
    a, b = metadata(), metadata(file="videos/b.mp4", split="test", original="videos/a.mp4")
    path = tmp_path / "meta.json"
    path.write_text(json.dumps([a, b]))
    with pytest.raises(ValueError, match="crosses splits"):
        load_metadata(path)


def test_tail_window_and_short_clip():
    assert window_starts(500, 200, 100) == [0, 100, 200, 300]
    assert window_starts(510, 200, 100) == [0, 100, 200, 300, 310]
    assert window_starts(50, 200, 100) == [0]


def test_adjacent_boundaries_are_not_overlap():
    assert overlap_seconds(0, 1, [[1, 1.5]]) == 0
    assert overlap_seconds(1.5, 2, [[1, 1.5]]) == 0
    assert overlap_seconds(1.2, 1.4, [[1, 1.5]]) == pytest.approx(0.2)


def test_union_prevents_double_counting():
    p = merge_periods([[1, 1.8], [1.2, 2]], 3)
    assert p == [[1.0, 2.0]]
    assert overlap_seconds(0, 3, p) == 1


def test_log_mel_silence_finite_shape():
    c = AudioConfig()
    result = log_mel(np.zeros(c.window_samples, np.float32), c)
    assert result.shape == (128, 404)
    assert np.isfinite(result).all()
    assert np.allclose(result, -100)


def test_loudness_not_erased_by_per_window_peak_reference():
    c = AudioConfig()
    t = np.arange(c.window_samples) / c.sample_rate
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    loud, quiet = log_mel(tone, c), log_mel(tone * 0.1, c)
    mask = loud > -50
    assert np.median(loud[mask] - quiet[mask]) == pytest.approx(20, abs=0.01)


def test_visual_only_is_genuine_audio():
    c = AudioConfig()
    m = metadata(modify_audio=False, modify_video=True)
    _, windows = make_windows(np.zeros(48000, np.float32), m, {"audio_offset_s": 0}, c)
    assert all(w["audio_label"] == 0 for w in windows)


def test_local_audio_labels_and_offsets():
    c = AudioConfig(window_seconds=1.0, stride_seconds=0.5)
    m = metadata(fake_periods=[[1.0, 1.25]])
    _, windows = make_windows(np.zeros(44800, np.float32), m, {"audio_offset_s": 0.2}, c)
    assert windows[0]["start_s"] == pytest.approx(0.2)
    assert windows[0]["audio_label"] == 1
    assert windows[-1]["end_s"] == pytest.approx(3.0)
    assert windows[-1]["audio_label"] == 0


def test_short_clip_padding_recorded_and_maskable():
    c = AudioConfig()
    m = metadata(duration=0.4, modify_audio=False, fake_periods=[])
    x, windows = make_windows(np.zeros(6400, np.float32), m, {"audio_offset_s": 0}, c)
    assert x.shape == (1, 128, 404)
    assert windows[0]["padding_fraction"] == pytest.approx(1 - 6400 / 64600)
    assert valid_columns(windows[0], 201, {"hop_length": 160}).sum() == 40


def test_missing_annotated_audio_is_flagged():
    c = AudioConfig()
    with pytest.raises(ValueError, match="outside available"):
        make_windows(np.zeros(16000, np.float32), metadata(), {"audio_offset_s": 0}, c)


def test_resnet_forward_and_backward():
    torch.set_num_threads(2)
    model = AudioResNet18(pretrained=False)
    x = torch.randn(2, 1, 128, 201)
    h = model.encode(x)
    assert h.shape == (2, 512)
    y = model(x)
    assert y.shape == (2,)
    nn_loss = torch.nn.BCEWithLogitsLoss()(y, torch.tensor([0., 1.]))
    nn_loss.backward()
    assert model.backbone.conv1.weight.grad is not None


def test_threshold_rule_and_single_class_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert best_f1_threshold(y, p) == pytest.approx(0.8)
    assert metrics(y, p, 0.5)["f1"] == 1
    assert metrics(np.zeros(2), np.array([0.2, 0.3]), 0.5)["roc_auc"] is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="FFmpeg/ffprobe not installed")
@pytest.mark.parametrize("offset", [0.0, 0.2])
def test_real_ffmpeg_extraction_and_pts(tmp_path, offset):
    video = tmp_path / "clip.mp4"
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
        "color=size=64x64:rate=25:duration=3", "-itsoffset", str(offset), "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=16000:duration=2.5", "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "mpeg4", "-c:a", "aac", str(video)]
    subprocess.run(command, check=True, capture_output=True)
    timing = probe_timing(video, "ffprobe")
    y = extract_wav(video, tmp_path / "audio.wav", 16000)
    assert len(y) > 39000
    assert abs(len(y) / 16000 - timing["decoded_audio_duration_s"]) < 0.03
    # AAC encoder delay can shift the first presented audio sample relative to
    # the requested offset; assert that extraction reports decoded PTS, not a
    # silently imposed zero offset.
    if offset:
        assert timing["audio_offset_s"] > 0.05
    else:
        assert abs(timing["audio_offset_s"]) < 0.01


def test_normalization_uses_train_only_and_excludes_padded_columns(tmp_path):
    from common import training_statistics
    x = np.full((1, 128, 404), 10, dtype=np.float32)
    # Valid sample count covers first 100 STFT columns; the remainder are padded.
    x[:, :, 100:] = -90
    np.savez_compressed(tmp_path / "train.npz", log_mel=x)
    np.savez_compressed(tmp_path / "dev.npz", log_mel=np.full_like(x, 90))
    records = [
        {"split": "train", "feature_path": "train.npz", "shape": list(x.shape),
         "file": "train.mp4", "windows": [{"valid_samples": 16000}]},
        {"split": "dev", "feature_path": "dev.npz", "shape": list(x.shape),
         "file": "dev.mp4", "windows": [{"valid_samples": 32000}]},
    ]
    stats = training_statistics(tmp_path, records, {"n_mels": 128, "hop_length": 160})
    assert np.allclose(stats["mean"], 10)
    assert np.allclose(stats["std"], 1)
    assert stats["columns"] == 100
