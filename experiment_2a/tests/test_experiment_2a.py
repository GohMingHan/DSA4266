from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import soundfile as sf
import torch
from audio_data import make_dataset
from common import equal_error_rate, split_signature, valid_columns
from compare import paired_bootstrap
from models import AudioAASIST
from prepare_audio import AudioConfig, log_mel, make_windows, normalize_metadata, repeat_pad


def make_record(tmp_path, duration=6.0, modify_audio=True, modify_video=False):
    c = AudioConfig()
    path = tmp_path / 'example.wav'
    y = (.1 * np.sin(2 * np.pi * 432 * np.arange(round(duration * 16000)) / 16000)).astype(np.float32)
    sf.write(path, y, 16000, subtype='PCM_16')
    y, _ = sf.read(path, dtype='float32')
    meta = normalize_metadata({'file': 'videos/example.mp4', 'original': None, 'split': 'train',
           'duration': duration, 'modify_audio': modify_audio, 'modify_video': modify_video,
           'fake_periods': [[duration * .7, duration * .8]] if modify_audio or modify_video else []})
    spec, windows = make_windows(y, meta, {'audio_offset_s': 0}, c)
    np.savez_compressed(tmp_path / 'example.npz', log_mel=spec)
    row = {**meta, 'clip_id': 'example', 'source_group': 'example',
           'feature_path': 'example.npz', 'wav_path': 'example.wav',
           'shape': list(spec.shape), 'windows': windows}
    return row, c


def test_repeat_padding_matches_tile_and_truncation():
    x = np.array([1., 2., 3.], dtype=np.float32)
    assert repeat_pad(x, 8).tolist() == [1., 2., 3., 1., 2., 3., 1., 2.]
    assert repeat_pad(x, 2).tolist() == [1., 2.]
    with pytest.raises(ValueError):
        repeat_pad(np.array([]), 8)


@pytest.mark.parametrize('duration', [1.0, 6.0])
def test_same_waveform_and_spectrogram_windows(tmp_path, duration):
    row, c = make_record(tmp_path, duration=duration)
    stats = {'mean': [0.] * 128, 'std': [1.] * 128}
    raw = make_dataset('aasist', tmp_path, [row], 'train', None, asdict(c))
    spec = make_dataset('resnet18', tmp_path, [row], 'train', stats, asdict(c))
    assert raw.index == spec.index
    assert np.array_equal(raw.labels, spec.labels)
    for k in (0, len(raw) - 1):
        waveform, a, i = raw[k]
        mel, b, j = spec[k]
        assert waveform.shape == (64600,)
        assert mel.shape == (1, 128, 404)
        assert a == b and i == j
        np.testing.assert_allclose(log_mel(waveform.numpy(), c), mel.numpy()[0], rtol=1e-5, atol=1e-5)


def test_repeated_context_not_assigned_extra_timestamps(tmp_path):
    row, c = make_record(tmp_path, duration=1.)
    w = row['windows'][0]
    assert w['start_s'] == 0 and w['end_s'] == 1
    assert w['input_samples'] == 64600 and w['valid_samples'] == 16000
    assert w['padding_fraction'] == pytest.approx(1 - 16000 / 64600)
    assert valid_columns(w, 404, asdict(c)).all()


def test_visual_only_negative_for_both_branches(tmp_path):
    row, c = make_record(tmp_path, modify_audio=False, modify_video=True)
    for name in ('aasist', 'resnet18'):
        ds = make_dataset(name, tmp_path, [row], 'train', {'mean':[0.] * 128, 'std':[1.] * 128}, asdict(c))
        assert np.all(ds.labels == 0)


def test_fake_logit_orientation_and_ce_equivalence():
    logits = torch.tensor([[3., -1.], [-1., 3.]], requires_grad=True)
    labels = torch.tensor([1., 0.])
    margin = AudioAASIST.spoof_logit(logits)
    torch.testing.assert_close(margin.sigmoid(), logits.softmax(-1)[:, 0])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(margin, labels)
    ce = torch.nn.functional.cross_entropy(logits, (1 - labels).long())
    torch.testing.assert_close(bce, ce)
    assert margin[0] > 0 and margin[1] < 0


def test_aasist_full_architecture_forward_backward():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = AudioAASIST()
    assert sum(p.numel() for p in model.parameters()) == 297866
    h, logits = model.forward_features(torch.randn(2, 64600) * .01)
    assert h.shape == (2, 160) and logits.shape == (2,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.network.out_layer.weight.grad is not None
    assert model.network.GAT_layer_S.att_proj.weight.grad is not None
    with pytest.raises(ValueError, match='64600'):
        model(torch.zeros(2, 32000))


def test_tensor_only_state_dict_roundtrip(tmp_path):
    torch.set_num_threads(2)
    a = AudioAASIST().eval()
    path = tmp_path / 'random_state_fixture.pth'
    torch.save(a.network.state_dict(), path)
    b = AudioAASIST(weights=path).eval()
    with torch.inference_mode():
        x = torch.randn(1, 64600) * .01
        torch.testing.assert_close(a(x), b(x))
    # A random fixture is NOT an official pretrained checkpoint validation.


def test_window_label_change_changes_signature(tmp_path):
    row, _ = make_record(tmp_path)
    copied = deepcopy(row)
    copied['windows'][0]['audio_label'] = 1 - copied['windows'][0]['audio_label']
    assert split_signature([row], 'train') != split_signature([copied], 'train')


@pytest.mark.parametrize('scores,expected', [([.1,.9],0.), ([.9,.1],1.), ([.5,.5],.5)])
def test_eer_endpoints(scores, expected):
    assert equal_error_rate(np.array([0,1]),np.array(scores)) == pytest.approx(expected)


def test_paired_bootstrap_identical_predictions():
    rows = [{'clip_id': str(i), 'audio_label': i % 2, 'score': .2 + .6 * (i % 2),
             'source_group': str(i // 2), 'predicted_audio_label': i % 2} for i in range(8)]
    result = paired_bootstrap(rows, deepcopy(rows), 25, 42)
    assert result['valid_replicates'] == 25
    assert result['roc_auc_difference_95pct_percentile_interval'] == [0., 0.]


def test_waveform_dataset_refuses_out_of_bounds_window(tmp_path):
    row, c = make_record(tmp_path)
    row['windows'][0]['end_sample'] = 99999999
    ds = make_dataset('aasist', tmp_path, [row], 'train', None, asdict(c))
    with pytest.raises(ValueError, match='offsets'):
        ds[0]
