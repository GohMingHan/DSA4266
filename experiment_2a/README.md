# Experiment 2A: LAV-DF audio — log-mel ResNet-18 versus waveform AASIST

Two end-to-end audio detectors use the **same decoded audio, timestamped windows,
window labels, official dataset splits, sampling budget, and evaluation rules**.
ResNet-18 receives numerical log-mel spectrograms. AASIST receives waveforms.
Both export timestamped embeddings for the later visual–audio experiments.

This package does not require DeepfakeBench. Extract audio from the original
LAV-DF videos, not silent face-crop exports. The earlier two-second spectrogram
cache is **not compatible**; use a new output folder.

**Method attribution:** the ResNet/log-mel combination is our specified baseline,
not a reproduction of Pham et al. or another paper's complete recipe. AASIST uses
the authors' published architecture/configuration; the bundled implementation is
a documented MIT-licensed port, not a byte-identical upstream checkout. The
original source can also be downloaded and used directly. The LAV-DF windowing,
labels, shared optimizer and evaluation are adaptations, not an ASVspoof result
reproduction. See [METHODS.md](METHODS.md).

## 1. Installation

Python 3.11 is a suggested environment. The validation environment was Python
3.13.5 on Linux/CPU; Windows and CUDA were not executed here.

```powershell
conda create -n lavdf-exp2a python=3.11 -y
conda activate lavdf-exp2a
python -m pip install -r requirements.txt
```

Install mutually compatible `torch` and `torchvision` builds for your CPU/CUDA
using https://pytorch.org/get-started/locally/ . For CPU-only installation:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Install the FFmpeg executables, **both ffmpeg and ffprobe**, and add them to PATH.
A Python package named ffmpeg is not a substitute. Official download information:
https://ffmpeg.org/download.html

```powershell
ffmpeg -version
ffprobe -version
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available())"
python check_install.py --device cpu
python -m pytest -q
```

`check_install.py` runs forward/backward passes for both full models without any
pretrained download. Use `--device cuda` to check your GPU. The test loss is not a
measure of real deepfake detection performance.

## 2. Dataset and extraction

The metadata must be the official LAV-DF JSON list with `file`, `split`,
`duration`, `modify_audio`, `modify_video`, `fake_periods`, and optional `original`.
The existing train/dev/test assignments are retained. Video paths are resolved
relative to `--data-root`; there is no assumed video subdirectory layout.
The default lookup is `metadata.min.json`, then `metadata.json`. Override with
`--metadata "D:\path\metadata.json"` when necessary.

Start with a deterministic, shuffled pilot:

```powershell
$Data = "D:\datasets\LAV-DF"
$Audio = "D:\lavdf_exp2a_audio"

python prepare_audio.py `
  --data-root "$Data" `
  --out "$Audio" `
  --splits train dev `
  --limit-per-split 100
```

This selects up to 100 videos **per requested split**, not 100 total. The pilot
is not guaranteed class-balanced. Training refuses to run when train or dev
lacks either audio class. Increase the pilot size rather than fabricating labels.

Inspect `summary.json`, `errors_latest_run.jsonl`, some WAVs and a few
spectrograms. The summary reports both numerical-feature and WAV disk usage.
Repeat without `--limit-per-split` to process the complete train/dev partitions:

```powershell
python prepare_audio.py --data-root "$Data" --out "$Audio" --splits train dev
```

Completed matching clips are resumed. Do not change preprocessing settings or
library versions inside the same cache folder. The configuration fingerprint
makes mismatches explicit. Errors are recorded and excluded from **both** model
cohorts; missing audio is never labelled genuine.

The full intended train/dev cohort must be finished before reported training.
Adding more train/dev clips after training invalidates its cohort fingerprint.
Pilot training should use disposable run directories and must be repeated after
expanding the cohort. Test can be extracted later without changing train/dev.

### Representation and labels

| Item | Setting |
|---|---|
| Waveform | First audio stream, mono, 16,000 Hz, PCM16 WAV |
| Window samples | **64,600**, exactly 4.0375 seconds |
| Window stride | 0.5 seconds, plus a final tail-aligned window |
| Short clips | Repeat the available waveform to 64,600 samples for BOTH branches |
| Mel bands | 128, Slaney scale/normalization, 0–8 kHz |
| FFT / Hann window / hop | 512 / 400 / 160 samples |
| Log conversion | `10*log10(power)`, fixed reference 1, floor -100 dB |
| Spectrogram | Float32 `[128, 404]`, not a PNG screenshot |
| ResNet batch | `[B, 1, 128, 404]` |
| AASIST batch | `[B, 64600]` |

Do not resize spectrograms to 224x224 or use plot axes/colour bars as input. The
same sample window used for AASIST is the source of the ResNet spectrogram.
Neither branch applies silence trimming, denoising, enhancement, time stretching,
or per-window peak normalization. Audio is necessarily decoded, downmixed,
resampled and quantized to PCM16. Original video files are not changed.

AASIST's official code uses 64,600 samples and repeats short utterances. The
4.0375-second window is therefore deliberately different from the earlier
2-second baseline. It is a matched starting protocol, not a claim that long
contexts are optimal for LAV-DF's localized edits. `prepare_audio.py` enforces
16 kHz and 64,600 samples for this experiment.

Window selection does not use ground-truth fake intervals. Labels do:

```python
window_audio_label = int(modify_audio and overlap_seconds > 0)
video_audio_label = int(modify_audio)
overall_video_label = int(modify_audio or modify_video)
```

Visual-only fakes are negative for this audio-native task. A positive window
contains **some** annotated manipulated audio; it is not necessarily entirely
synthetic. Tiny overlaps remain difficult positive examples and are recorded as
`audio_fake_seconds`. Fake intervals are not fed into the CNN as input features.

The waveform/audio start offset is measured against the first decoded video
frame. Discontinuous audio, missing streams, unavailable annotated intervals and
inconsistent durations are flagged. Windows retain actual source-time bounds.
For short clips, artificial repetitions do **not** extend `end_s`; the actual
source interval, input length and `padding_fraction` remain separate. Repetition
can create an artificial boundary, so examine short-clip performance separately.

### Storage

```text
lavdf_exp2a_audio/
  config.json
  manifest.jsonl
  summary.json
  errors_latest_run.jsonl
  normalization_train.json        # created by the ResNet training path
  clips/<prefix>/
    <clip_id>.wav                 # retained automatically for AASIST
    <clip_id>.npz                 # log_mel: [N_windows, 128, 404]
    <clip_id>.json                # labels, timing, provenance, window list
```

Overlapping spectrograms duplicate information; measure disk usage on the pilot.
The implementation loads the JSON manifest into RAM and caches a few decoded
clips per worker; it is not a sharded/distributed production data system. Use a
separate cache for a storage-precision ablation. Test throughput before launching
the full dataset. Source/metadata signatures detect declared input changes;
this is not a cryptographic audit of every cached numerical array on every epoch.

Optional inspection (the output image is not a training input):

```powershell
python -m pip install matplotlib
python preview.py "D:\lavdf_exp2a_audio\clips\ab\example.npz" --window 0 --png preview.png
```

## 3. Train the primary matched pair from scratch

Use the same settings for both models. Run each command separately from the
project folder:

```powershell
python experiment.py train `
  --root "$Audio" `
  --model resnet18 `
  --init scratch `
  --out "D:\lavdf_runs\E2A_resnet18_s42" `
  --device cuda `
  --batch-size 4 `
  --epochs 30 `
  --seed 42

python experiment.py train `
  --root "$Audio" `
  --model aasist `
  --init scratch `
  --out "D:\lavdf_runs\E2A_aasist_s42" `
  --device cuda `
  --batch-size 4 `
  --epochs 30 `
  --seed 42
```

Use `--device cpu` without CUDA. Both examples train the full encoder, not only
a cached-feature classifier. Batch size 4 is a conservative starting setting,
not a GPU-memory guarantee. Reduce it on **both** runs when needed. `--accumulate
6` accumulates six microbatches before updating; BatchNorm still sees each
microbatch, so it is not numerically equivalent to batch size 24.

Both default to FP32, AdamW at 1e-4, weight decay 1e-4, cosine scheduling,
inverse-frequency class-balanced TRAIN window sampling, and BCEWithLogitsLoss.
There is no second inverse-frequency loss weighting. Their sampler has a
separate seeded RNG, so matching seed/budget gives matching sampled window
indices regardless of model initialization. This balances classes, **not videos
or identities**; long clips still provide more windows.

Best checkpoints are selected by **dev video-level acoustic ROC-AUC**. Early
stopping has patience 5. Window and video operating thresholds maximize dev F1.
The default video aggregator is maximum window probability; `--pooling mean` is
an explicit matched ablation. Max pooling can create duration-dependent false
alarms; mean pooling can dilute short edits. Select policies with development
data only. Test EER is a curve diagnostic, not a selected deployment threshold.

The ResNet frontend uses training-only per-mel mean/std; repeated context is
included because it is actually supplied to both models. AASIST consumes the
PCM-scaled waveform without spectrogram normalization. There are no natural-image
flips, crops or colour augmentations and no AASIST frequency augmentation in the
matched baseline. Both systems' predictions are **not calibrated probabilities**.

Each run creates `best.pt`, `history.json`, `run_config.json`, and
`best_dev_metrics.json`; ResNet also writes `normalization.json`. The checkpoint
contains the model, normalization, preprocessing identity, train/dev signatures,
model-source hashes, pooling rule and dev thresholds. It uses tensor-safe loading.
Training resume is not implemented; a fresh training run requires an empty output
directory. Preprocessing resume is supported.

A small code-path pilot can add `--epochs 2 --samples-per-epoch 200 --batch-size 2`
to both runs, but those are not final research results. Repeat the reported
configurations with seeds **42, 43, 44**, each in a new run directory.

## 4. Original upstream source and pretrained experiments (optional)

The default AASIST file is an attributed local port with the full architecture,
297,866 parameters and the upstream state-dictionary key layout. To run the
original authors' source directly instead:

```powershell
python download_aasist.py --out ".\upstream_aasist"
python verify_upstream.py --upstream ".\upstream_aasist"
```

The downloader resolves `main` to a commit, fetches all files from that exact
commit and records their SHA256 hashes. On reruns it retains the locked revision.
Use `--revision <40-character-commit>` to request a specific commit. It does not
execute downloaded code. Preserve `download_manifest.json` with your report.

The verification script loads identical random parameters in the bundled and
upstream networks and compares features/logits in evaluation mode. This optional
network-dependent verification was **not executed in the validation environment**.
A failure must be investigated, not ignored. Source download/official checkpoint
access could not be tested here.

To use upstream in training, evaluation, export, or `check_install.py`, append:

```powershell
--aasist-dir ".\upstream_aasist"
```

Use the same source directory throughout a run: its source and config hashes are
checked at inference. Full ASVspoof research reproduction still requires the
original data/protocol/trainer, not this LAV-DF wrapper.

For an additional **transfer-learning** track:

```powershell
python download_aasist.py --out ".\upstream_aasist" --weights

python experiment.py train --root "$Audio" --model resnet18 --init pretrained --out "D:\lavdf_runs\E2A_resnet18_transfer_s42" --device cuda --batch-size 4 --epochs 30 --seed 42

python experiment.py train --root "$Audio" --model aasist --init pretrained --aasist-dir ".\upstream_aasist" --aasist-weights ".\upstream_aasist\AASIST.pth" --out "D:\lavdf_runs\E2A_aasist_transfer_s42" --device cuda --batch-size 4 --epochs 30 --seed 42
```

ResNet initializes from ImageNet; AASIST initializes from the authors' ASVspoof
2019 LA checkpoint. This comparison includes **different pretraining data**, not
only architecture. Both still need LAV-DF training. `--freeze-epochs 1` optionally
warms up only the classification head before unfreezing; otherwise all learned
parameters update immediately. No pretrained weights are included in this zip.

Official AASIST has output class 0=spoof, 1=bonafide. The wrapper uses
`logit_spoof - logit_bonafide`, so `sigmoid` is exactly the original softmax spoof
score. Package targets and exported scores consistently use **1=manipulated**.
The code never assumes the second official output means fake.

## 5. Locked test evaluation and paired comparison

After freezing preprocessing, model selection and thresholds:

```powershell
python prepare_audio.py --data-root "$Data" --out "$Audio" --splits test

python experiment.py evaluate --root "$Audio" --checkpoint "D:\lavdf_runs\E2A_resnet18_s42\best.pt" --split test --out "D:\lavdf_results\E2A_resnet18_s42" --device cuda

python experiment.py evaluate --root "$Audio" --checkpoint "D:\lavdf_runs\E2A_aasist_s42\best.pt" --split test --out "D:\lavdf_results\E2A_aasist_s42" --device cuda

python compare.py --runs "D:\lavdf_results\E2A_resnet18_s42" "D:\lavdf_results\E2A_aasist_s42" --out "D:\lavdf_results\E2A_comparison_s42" --bootstrap 1000
```

Add the corresponding `--aasist-dir` on AASIST evaluation when the original
upstream file was used for training. No original pretrained checkpoint is needed
to load your complete trained checkpoint.

Evaluation outputs:

- `metrics.json`: window/video ROC-AUC, AP, interpolated EER, F1, precision,
  recall, balanced accuracy, confusion matrices and class prevalence.
- `video_predictions.jsonl`: video IDs, source groups, labels, scores, decisions.
- `window_predictions.jsonl`: per-window scores and source-time annotations.

The four modality categories are reported separately. Real and visual-only
videos contribute **acoustic false positives**; audio-only and jointly manipulated
videos contribute **acoustic recall**. These are not any-fake-video metrics.
No min-tDCF is computed because this task does not include an ASV system's scores.

`compare.py` refuses mismatched data/configuration/targets/pooling or, by default,
mismatched training budgets/seeds. Its CSV/JSON comparison is video-level, not a
comparison of correlated windows treated as independent examples. With exactly
two runs, `--bootstrap 1000` gives paired source-group percentile intervals for
AUC and balanced-accuracy differences (run B minus run A). Source groups are not
necessarily identities; this does not replace seed variability or identity-level
uncertainty analysis. Single-class bootstrap replicates are counted and skipped.
Use `--allow-training-differences` only to explicitly label an unmatched-training
comparison; it never permits mismatched evaluation cohorts.

Report mean/std across independently trained seeds. Do not choose a winner from
test results and then tune that winner on the same test partition.

## 6. Export features for Experiments 2B and 3

```powershell
python experiment.py export --root "$Audio" --checkpoint "D:\lavdf_runs\E2A_resnet18_s42\best.pt" --out "D:\lavdf_features\audio_resnet18" --device cuda

python experiment.py export --root "$Audio" --checkpoint "D:\lavdf_runs\E2A_aasist_s42\best.pt" --out "D:\lavdf_features\audio_aasist" --device cuda
```

Again, append the original `--aasist-dir` when appropriate. Export defaults to
all three splits; explicitly request `--splits train dev` before test extraction.

```text
features           [N_windows, 512] for ResNet, [N_windows, 160] for AASIST
mean_feature       [512] or [160]
start_s            [N_windows]
end_s              [N_windows]
center_s           [N_windows]
padding_fraction   [N_windows] (diagnostic; not a label or extra recorded time)
audio_probability  [N_windows] (uncalibrated model scores)
```

Use `mean_feature` as the simple pooled-vector baseline and the full sequence for
cross-modal interaction. Match with visual features using the metadata `file`
identifier and verified common video-relative timestamps, not DataLoader order.
All label fields are bookkeeping/targets, not fusion inputs. Regenerate features
after changing the encoder. The NPZ/JSONL output is a documented standalone
interface, not automatically plug-compatible with every earlier project script.

The approximately four-second feature describes its complete context, not an
instantaneous frame or guaranteed lip-sync measurement. Artificial repeats for
short clips are not real additional events. For a detailed synchronization study,
shorter-context training should be a separately validated future protocol.

For the final multimodal task, freeze the selected audio encoder and train your
matched video-level audio/visual/fusion heads on `modify_audio OR modify_video`.
The Experiment 2A audio-native score is not automatically that trained control.

## 7. Validation and sources

See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for exact executed tests and
[METHODS.md](METHODS.md) for paper attribution and implementation deviations.
Tests use non-speech generated signals and artificial labels only. No real LAV-DF
detection accuracy is asserted. Windows, CUDA, full-dataset scaling, ImageNet
weight downloads and official AASIST pretrained downloads remain unvalidated here.

Dataset/source-model licenses still apply. The AASIST port retains its upstream
MIT notice in `third_party/aasist/LICENSE`; no LAV-DF data or pretrained weights
are redistributed.
