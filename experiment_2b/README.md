# Experiment 2B — frozen visual/audio embeddings and concatenation

**This is the simple fusion experiment, not cross-attention or an autoencoder.**
It uses exactly one selected visual encoder, one selected audio encoder, and one
new video classifier. All encoder weights stay frozen. Classification target:

```python
label = int(modify_audio or modify_video)  # 1 means ANY manipulation
```

A visual-only forgery is positive here, even though it was an audio-negative in
Experiment 2A. In the planned four-category-balanced 4,000/500/500 cohort, this
means 3,000/375/375 positive videos and 1,000/125/125 genuinely negative videos.
The code derives and reports actual counts instead of assuming those numbers.

## What is included

```
experiment_2b/
  pipeline.py
  config.json
  requirements.txt
  e2b/                       implementation
  models/
    visual/
      encoder.json           select a loader AFTER choosing the model
      best.pt                YOU put your selected trained checkpoint here
    audio/
      encoder.json
      best.pt                YOU put your selected trained checkpoint here
  examples/                  configuration templates and custom-adapter contract
  tests/                     synthetic software tests; no dataset or real weights
```

No `best.pt` is shipped; there is no random-model fallback. The models directories
already exist, containing explanatory files and unconfigured slot files. A model
name/file extension does not identify an arbitrary saved model's architecture,
normalization or label convention. **The slots require configuration as well as
weights.** Do not rename a custom notebook checkpoint and assume that changes its
format.

### Supported checkpoint bridges

| Slot backend (`kind`) | What it loads | What it exports |
|---|---|---|
| `experiment_2a` | Your original Experiment 2A ResNet-18 or AASIST checkpoint and its original code/preprocessed audio | `forward_features()[0]`: 512-D or 160-D audio-window embeddings |
| `forensics_adapter` | Your original Forensics Adapter/CLIP-probe checkpoint, original project, visual manifest and matching frozen CLIP weights | The trained adapted visual embeddings before its classifier; dimension comes from the model |
| `ummaformer` | Your RGB-only, RGB+Farneback, or RGB+TSN-flow checkpoint, its original project/upstream and matching input cache | Finest-resolution learned temporal FPN features entering the proposal classifier, with masks removed and model point coordinates mapped to seconds |
| `preexported` | Any previously exported **embeddings** satisfying the format below | Validated/canonicalized sequences; original checkpoint hash checked |
| `custom` | An explicitly implemented local loader in `model_adapter.py` | Whatever learned intermediate representation that adapter explicitly defines |

The UMMAFormer bridge does **not** use the weak scalar internal video-classifier
score, and does **not** mistake its raw ResNet/TSN input cache for the trained
UMMAFormer output. It observes the highest-resolution FPN input to `cls_head`
using read-only hooks, leaving native source/checkpoints untouched. This readout
is a specified adaptation, not a claim that it is the best fusion representation.
Only that finest level is exported. Features are temporally contextualized, not
instantaneous observations, and the 768-bin interpolation creates no new detail.

**Your Kaggle `evaluate_test_models.py` and its underlying training/model code have
not been inspected.** A checkpoint produced there is supported by `experiment_2a`
only when it is actually the original package format with its original model
implementation and preprocessing. Different architecture keys, three-channel
spectrograms, different windowing/normalization etc. need a custom adapter or a
preexported bridge. CNN/YOLO formats not supplied so far are not guessed either.

## 1. Install separately from the encoder projects

In WSL, after saving the ZIP in Windows Downloads:

```bash
cd "$HOME/dsa4266"
unzip "/mnt/c/Users/wuche/Downloads/experiment_2b.zip"
cd "$HOME/dsa4266/experiment_2b"
```

Use an environment with a working PyTorch installation. You can reuse
`ummaformer` for the small fusion head:

```bash
conda activate ummaformer
python -m pip install -r requirements.txt
python pipeline.py --help
```

Do not reinstall/upgrade a working CUDA/PyTorch installation just for this add-on.
The head uses ordinary float32 PyTorch and has no `torchsort`, temporal NMS, dlib,
FFmpeg or torchvision requirement. **Individual encoder bridges still require
their own original dependencies.** The worker for each encoder can use a separate
Python executable from its existing environment, configured with the slot's
`python` field. `null` means use the interpreter running `pipeline.py`.

Examples are WSL paths, not Colab/Kaggle paths. On a notebook VM set the actual
VM paths and normally `python: null`. It cannot see your `/home/evan` or `/mnt/c`
folders without copying/mounting the necessary files. All output paths must be
writable; put the project under `/kaggle/working` or `/content`, not a read-only
input mount. Preserve outputs/checkpoints outside ephemeral notebook storage.

Keep this project **outside** the original encoder project directories. Some
original checkpoint validators hash all their top-level Python files. Do not
install new loaders into those project roots or disable their source checks.

## 2. Fill the model slots after selection

Select using development performance and an appropriate resource benchmark, not
repeated tuning against the already-viewed audio test scores. No encoder is
selected automatically here.

Example checkpoint copies (edit the source paths to the files you actually select):

```bash
cp "/path/to/selected_visual_checkpoint.pt" models/visual/best.pt
cp "/path/to/selected_audio_checkpoint.pt" models/audio/best.pt
```

These are copies, not modifications of the old runs. Auxiliary assets such as
frozen CLIP weights may be left in place and referenced by an absolute path.

### Example: UMMAFormer + original Experiment 2A audio model

```bash
cp examples/visual_ummaformer.json models/visual/encoder.json
cp examples/audio_experiment2a.json models/audio/encoder.json
```

Open the two copied JSON files and set the actual paths. Relative `checkpoint`
paths are resolved from **that encoder.json's directory**. Thus `best.pt` points
to the checkpoint immediately next to `encoder.json`.

For TSN-flow UMMAFormer, use `visual_ummaformer_tsn.json` and its new matching
feature manifest/project. RGB-only is inferred from its own checkpoint; do not
try to load RGB+flow weights into an RGB-only architecture.

For Forensics Adapter, copy `visual_forensics_adapter.json` instead. Set its
`clip_weights` to the exact frozen backbone file used in that run (whether your
local project uses L/14 or a verified Base adaptation). Keep the original
architecture code. The bridge uses the stored configuration and does not silently
substitute another CLIP variant. `amp:null` preserves the checkpoint's inference
precision. `attention_backend:flash-math` keeps memory-efficient/cuDNN SDPA off,
as in your stable launcher. This is local to the extraction process.

For AASIST, set `aasist_dir` to the actual external source directory used during
training, or leave `null` for the original package's bundled source. The full
trained checkpoint includes learned weights; the original pretrained .pth is
not downloaded again. ResNet also loads its trained checkpoint, not fresh
ImageNet weights. Saved audio normalization is used; none is fitted on DEV/TEST.

### Configure the authoritative cohort

In top-level `config.json`, set `cohort` to either:

* The reduced LAV-DF-5000 `metadata.min.json`/`metadata.json`, or
* Your existing `lavdf5000_visual_manifest/videos.jsonl`, which must contain
  `file,split,modify_audio,modify_video` (and source identifiers).

Use the actual reduced cohort, **not the full 136k metadata**. No videos are
randomly resplit. You can start with train/dev only; when preparing test later,
use a cohort containing the fixed test IDs and unchanged train/dev records.
Expected counts apply only to requested/available splits. Do not lower them to
hide failures without auditing and documenting a changed cohort.

Then:

```bash
python pipeline.py check --config config.json
```

Before you have selected/configured the encoders, `not_ready` is intentional.
This is a path/schema preflight, not proof that real checkpoint inference works.

## 3. Extract frozen embeddings once

First use small, separate extraction-pilot folders:

```bash
python pipeline.py extract --config config.json --modality visual \
  --splits train dev --device cuda:0 --limit-per-split 4 --out work/pilot_visual

python pipeline.py extract --config config.json --modality audio \
  --splits train dev --device cuda:0 --limit-per-split 4 --out work/pilot_audio
```

Each pilot processes four videos per split, uses real trained checkpoints, and
checks actual output shape/timing. It does not train or report accuracy. Pilot
exports are explicitly marked partial and cannot silently become the main cohort.

After both succeed, omit the pilot limit and `--out` to use the full caches from
`config.json`:

```bash
python pipeline.py extract --config config.json --modality visual \
  --splits train dev --device cuda:0

python pipeline.py extract --config config.json --modality audio \
  --splits train dev --device cuda:0
```

Or `--modality both` runs them sequentially, each in its own Python process. This
avoids importing two legacy packages both named `common`, `models`, `experiment`.
It also avoids holding two large encoders on the GPU simultaneously.

The selected encoder is set to evaluation mode and its weights are frozen.
Inference does not use labels as network inputs. Native loaders may inspect
annotations for bookkeeping/cohort verification, never to choose fake-only test
windows or provide them as input features. Full source sequences/windows from
those audited loaders are retained. For UMMAFormer a whitelist removes all target
fields before native inference. Frozen feature extraction is not joint fine-tuning.

### Cache and resume

```
work/embeddings/visual/             work/embeddings/audio/
  export.json                        export.json
  splits/train.json                  splits/train.json
  splits/dev.json                    splits/dev.json
  records/<video_id>.json            records/<video_id>.json
  clips/<split>/<video_id>.npz        clips/<split>/<video_id>.npz
  summary.json                       summary.json
```

NPZ contains numeric `features[T,D]` and timestamps; it contains no labels or
probability fields. Encoder identity includes checkpoint/source/preprocessing
provenance. Completed videos are checked by source/content hashes and resumed
when rerunning the identical command. Ctrl+C restarts the interrupted video.
Failed records never become zero vectors or genuine labels. Split manifests are
published only when all requested videos succeeded. After a hard kill, remove
only a confirmed stale `.e2b.lock`, with no other writer active.

Changing an encoder, preprocessing, source version, precision or extraction
software environment requires a new embedding folder and a new fusion run. Do
not overwrite a cache used by an existing experiment. Adding the test split later
using the same frozen configuration is supported; existing train/dev indices are
not replaced by a test-only index.

## 4. Pair and pool

```bash
python pipeline.py prepare --config config.json --splits train dev \
  --pooling mean --out work/paired_train_dev
```

The default is **strict pairing**: the exact requested video IDs must exist in
both exports. There is no silent intersection or relabelling. A missing/extra-ID
report is written when they differ. Inspect exclusions in prior preprocessing
and explicitly freeze a documented common cohort before trying again.

For each video:

```
visual sequence -> mean over its own valid entries -> v [Dv]
audio sequence  -> mean over its own real windows  -> a [Da]
```

Sequence lengths need not match. There is no frame-index to audio-window-index
pairing, time warping, or cross-attention. Timestamps are retained in the original
exports for Experiment 3. Disjoint modality time ranges are rejected. Extent
differences greater than half a second are reported for review, not automatically
corrected. Shared file IDs/time-origin checks cannot prove semantic synchronization
if a custom export supplied wrong timestamps; audit those when selecting a model.

Mean pooling is an explicitly simple baseline. It can dilute short manipulated
segments; it is not claimed optimal or equivalent to the old maximum-proposal
UMMAFormer score. `--pooling mean_max` adds per-dimension maximums as a separate
DEV-selected ablation and needs a new paired folder and new training run. No
scores or ground-truth intervals are used to select features for either pool.

Output contains one visual vector, one audio vector, and one overall target per
video; `dataset.json` reports exact dimensions, counts and fingerprints. Large
encoders are not loaded during subsequent classifier training.

## 5. Train Experiment 2B

```bash
python pipeline.py train \
  --data work/paired_train_dev \
  --mode concat \
  --out runs/concat_s42 \
  --device cuda:0 \
  --batch-size 64 \
  --epochs 50 \
  --patience 8 \
  --seed 42
```

`--device cpu` also works; this head is much smaller than the image encoders.
No GPU speed/memory guarantee is made. Proposed defaults, not paper optima:

```
TRAIN-only per-dimension standardization (independent modalities)
v -> Linear(Dv,128) -> ReLU -> dropout 0.2 -----+
                                               +-> concatenate [256]
a -> Linear(Da,128) -> ReLU -> dropout 0.2 -----+       |
                                             Linear(256,128)
                                             ReLU + dropout 0.2
                                             Linear(128,1)
```

AdamW at 1e-3, weight decay 1e-4, float32, gradient-norm limit 5. This is one pair
per video (4,000 samples), **not** 315k frame samples. Only the projection/head
parameters update. Code does not optimize the two encoders together.

Default `--balance class` uses inverse binary-class-frequency **video sampling**
with replacement for 4,000 draws per epoch, with unweighted BCEWithLogitsLoss.
With 75% positives this oversamples the real class, unlike the earlier rare-fake-
frame task. It does not use a 22.5 positive weight. There is no guarantee every
video appears per resampled epoch, and sources/identities are not equally weighted.
`--balance none` is a separate ordinary-shuffle alternative. DEV/TEST are never
resampled. Normalization is fitted on the natural TRAIN vectors, not DEV/TEST or
on rebalanced draws.

Best checkpoint and early stopping use **DEV video any-manipulation ROC-AUC**.
The operating threshold is selected from that checkpoint's DEV logits using
maximum F1 (highest threshold on ties). AUC/AP use continuous logits. Raw logits
avoid sigmoid saturation ties; saved sigmoid scores are not called calibrated
probabilities. Threshold is saved in logit and probability form.

```
runs/concat_s42/
  best.pt
  last.pt
  history.json
  run_config.json
  normalization_train.json
  best_dev_metrics.json
  best_dev_predictions.jsonl
```

Ctrl+C does not erase saved epochs. Resume with the same settings/output and:

```bash
python pipeline.py train --data work/paired_train_dev --mode concat \
  --out runs/concat_s42 --device cuda:0 --batch-size 64 --epochs 50 --patience 8 \
  --seed 42 --resume runs/concat_s42/last.pt
```

An unfinished epoch restarts from the previous saved `last.pt`; there is no
mid-batch training checkpoint. Saved source/settings/data identities are checked.
Repeat final configurations with seeds 42/43/44 instead of choosing a lucky seed.

## 6. Train matched controls

The original audio/visual-native scores do NOT answer the same label question.
Train controls with the **same paired data, target and training settings**:

```bash
python pipeline.py train --data work/paired_train_dev --mode visual \
  --out runs/visual_only_s42 --device cuda:0 --seed 42

python pipeline.py train --data work/paired_train_dev --mode audio \
  --out runs/audio_only_s42 --device cuda:0 --seed 42
```

Those commands match the defaults used above. The audio-only control must treat
visual-only forgeries as positive for the final task, even when acoustically
indistinguishable from genuine recordings. This is a modality limitation, not
permission to remove those videos. All controls use exactly the same cohort.
Input sizes/parameter counts differ for controls and are reported.

## 7. Development evaluation and comparison

```bash
python pipeline.py evaluate --data work/paired_train_dev \
  --checkpoint runs/concat_s42/best.pt --split dev \
  --out evaluations/concat_dev --device cuda:0

python pipeline.py evaluate --data work/paired_train_dev \
  --checkpoint runs/audio_only_s42/best.pt --split dev \
  --out evaluations/audio_dev --device cuda:0

python pipeline.py compare --runs evaluations/audio_dev evaluations/concat_dev \
  --out evaluations/audio_vs_concat --bootstrap 1000
```

Outputs: video any-fake ROC-AUC, AP, accuracy, balanced accuracy, F1, precision,
recall, genuine-video FPR, confusion matrix and audio-only/visual-only/joint-fake
recall. `video_predictions.jsonl` contains IDs, targets, logits, scores and decisions.
`compare` does not ensemble predictions. Optional paired source-family bootstrap
requires exactly two runs; intervals are conditional on selected models, not an
adjustment for repeated DEV selection or independent training randomness.

## 8. Locked test evaluation

Finish model/pooling/head selection using DEV first. Your previous audio test
results were already viewed: describe that honestly rather than retroactively
claiming perfectly blinded encoder selection. Do not search this test set for new
thresholds or feature choices.

When test preprocessing exists in BOTH original encoders' inputs and the cohort
contains the fixed test IDs:

```bash
python pipeline.py extract --config config.json --modality both --splits test --device cuda:0

python pipeline.py prepare --config config.json --splits train dev test \
  --pooling mean --out work/paired_with_test

python pipeline.py evaluate --data work/paired_with_test \
  --checkpoint runs/concat_s42/best.pt --split test \
  --out evaluations/concat_test --device cuda:0
```

The new paired dataset must have identical train/dev fingerprints and encoder
identities; the existing head is not retrained and no threshold is fit on test.
For imported embeddings, update the external `index.jsonl` to include test records
without changing any train/dev files/checkpoint/preprocessing. Native loaders
also require their original train/dev cohort unchanged when test is added.

## 9. A different CNN, YOLO or notebook checkpoint

Two choices, without changing the fusion model code:

**Preferred minimal integration:** export learned embeddings from its original
working inference environment, then use `examples/preexported.json`. Each row of
`index.jsonl` must contain:

```json
{"file":"train/000123.mp4","split":"train","feature_path":"train/000123.npz","checkpoint_sha256":"ACTUAL_SHA256_OF_THE_SELECTED_CHECKPOINT"}
```

`file` is the EXACT original metadata path, not a basename or a new crop ID.
`feature_path` is relative to the index's directory. Index rows must be unique.
Optional label fields are cross-checked against the authoritative cohort. Add
`sha256` for the NPZ too when available. Use `sha256sum models/audio/best.pt` to
obtain a local checkpoint hash. A hash is an identity check, not an authenticity
or training-quality guarantee.

Visual NPZ: `features[T,D]`, `timestamp_s[T]`.
Audio NPZ: `features[W,D]`, `start_s[W]`, `end_s[W]`, `center_s[W]`; optional
`padding_fraction[W]` for repeated short windows. All time is relative to the FIRST
DECODED VIDEO frame. D must be at least 3; scalar/2-class scores are rejected.
Shapes alone cannot prove that a wider array is an embedding: confirm the exact
pre-classifier layer. No 1000-class probabilities, prediction scores, ground-truth
fake intervals or modality labels may be fed as features.

Supply the model's training-time channel conversion, resampling, spectrogram
frontend, normalization, face/input resize etc. unchanged. Set
`preprocessing_verified:true` only after checking them. No new default log-mel or
ImageNet preprocessing is invented for an unknown checkpoint. A preexported
backend does not load the checkpoint itself, so it works when Kaggle/WSL model
environments differ; it checks exported rows against the checkpoint identity.

**Full custom loader:** implement `examples/model_adapter.py` for your actual
model, copy it into the model slot, and use `examples/custom.json`. The template
raises NotImplementedError until implemented; it is not a fake universal loader.
Only run adapter Python/checkpoints you trust. All bundled checkpoint loading uses
`weights_only=True`; there is no unsafe pickle fallback or automatic download.

## 10. Experiment 3 and future inference

The separate sequence caches survive pooling. Experiment 3 can reuse these exact
encoder checkpoints, features and source-time coordinates, introducing interaction
BEFORE pooling. Experiment 2B intentionally discards within-stream order only at
its pooling step. Changing encoders later requires new features and retraining
the fusion head even if the dimensions happen to match.

`best.pt` contains the trained **fusion head and normalization**, not a new
self-contained replacement for both encoders. To score a NEW video end-to-end,
you still need each selected encoder and its original preprocessing, then the
same independent pooling and head. Cached-feature evaluation skips repeated
encoder execution; deployment to unseen media cannot skip it. A raw-MP4 serving
API/GUI is not supplied because your final encoders and their notebook formats
are not yet fixed.

## Validation and attribution

See VALIDATION_REPORT.md for executed tests and limits. Tests are synthetic, not
real deepfake accuracy. This is a specified practical baseline rather than a
claimed reproduction of one fusion paper. Existing model/dataset licenses still
apply. No weights, media, or original project source are redistributed.
