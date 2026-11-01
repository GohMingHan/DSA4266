# UMMAFormer with frozen pretrained TSN flow features

A separate experiment for your existing **RGB + pooled Farneback** project.
This package preserves the existing RGB512 arrays and timestamps, recomputes
native dense flow, uses a pretrained TSN ResNet-50 flow encoder, and trains a new
UMMAFormer on **RGB512 + TSN-flow2048 = 2560 channels**.

**This is not an ensemble and not end-to-end TSN fine-tuning.** TSN is pretrained
and frozen. It runs during extraction only. Your previous code, videos, features,
upstream checkout and trained checkpoints are not edited.

The primary configuration retains Farneback as the *flow estimator*. The learned
TSN encoder replaces the old 16x16 averaging/flattening representation. It does
not replace Farneback with a neural flow estimator. The new five-pair context,
input encoding and feature width also differ. This is a documented representation
adaptation, **not an exact reproduction of the paper's complete TSN pipeline**.
See [METHODS.md](METHODS.md).

## 1. Install beside, not inside, the old project

Download the ZIP into your Windows Downloads directory. In your WSL terminal:

```bash
conda activate ummaformer
cd "$HOME/dsa4266"
unzip "/mnt/c/Users/wuche/Downloads/ummaformer_tsn_flow.zip"
cd "$HOME/dsa4266/ummaformer_tsn_flow"

Project="$PWD"
OldProject="$HOME/dsa4266/ummaformer_opencv_flow"
Reference="$OldProject/features/rgb_flow_train_dev/manifest.json"
Upstream="$OldProject/upstream"
Data="/mnt/c/Users/wuche/Downloads/LAV-DF"
Baseline="$OldProject/runs/umma_rgb_flow_lr0p0003_s42/best_video_auc.pt"
FlowWeights="$Project/weights/tsn_flow_activitynet.pth"

ls -lh "$Reference" "$Baseline"
test -d "$Upstream/libs" && echo "Existing UMMAFormer checkout found"
python check_install.py
```

Use a new project folder; do not unpack over the old project or a currently
running experiment. If the ZIP was saved elsewhere, change the download path.
`Data` must be the original-video root used for your existing feature cache.
Cached source-file size/mtime and decoded timestamps are checked. Do not edit
cache signatures to make a different source appear identical.

The default workflow reuses your working Python environment: PyTorch,
torchvision, NumPy, OpenCV, Pillow, scikit-learn, tqdm, FFmpeg and ffprobe, plus
UMMAFormer's already-built torchsort/NMS. **Do not reinstall or downgrade a working
PyTorch environment.** MMAction2/MMCV are not required for the default path. The
checkpoint backbone is loaded by strict tensor-name/shape mapping to torchvision
ResNet-50. `requirements.txt` documents the small Python dependencies; it is not
an instruction to upgrade every package before trying the existing environment.

No pretrained weights or upstream UMMAFormer source are included in the ZIP.
Use the same unchanged official checkout as your successful baseline, via the
explicit `--upstream` argument in every runner command below.

## 2. Download and validate the flow checkpoint

```bash
python fetch_tsn_flow.py --out "$FlowWeights"

python check_install.py \
  --flow-weights "$FlowWeights" \
  --device cuda:0
```

The helper downloads the **specific ActivityNet-clip flow checkpoint linked by
UMMAFormer**:

```
tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804-8622cf38.pth
```

It verifies the published filename's SHA256 prefix `8622cf38`, records the full
SHA256, and strictly requires every learned backbone tensor. The action-classifier
head is excluded. A ResNet ImageNet RGB checkpoint is NOT a substitute. No
partial/random fallback is provided.

The check should report `real_pretrained_forward: passed` with shape `[1, 2048]`.
This is a loading/forward check, not an accuracy test or independent numerical
parity verification. Internet is required for the first weight download, not for
subsequent local extraction/training once all dependencies and files are present.
An interrupted download restarts; it does not corrupt a previously valid file.

The downloaded checkpoint could not be fetched in the generation environment.
If loading fails, keep the traceback; do not bypass strict/hash/safe-loading checks.

### Optional independent MMAction numerical check

An independently installed, compatible **MMAction2 0.x** environment can run:

```bash
python check_install.py \
  --flow-weights "$FlowWeights" \
  --device cpu \
  --parity-mmaction
```

This compares the same real checkpoint in the torchvision mapping and MMAction
backbone. It does not install MMAction. Use a separate environment instead of
changing the working UMMAFormer installation. Real pretrained numerical parity
was not executed here; source mapping and shape tests are not a substitute.

## 3. Extract an eight-video pilot

```bash
Pilot="$Project/features/tsn_flow_pilot"

python extract_tsn_flow.py \
  --reference-manifest "$Reference" \
  --data-root "$Data" \
  --flow-weights "$FlowWeights" \
  --out "$Pilot" \
  --splits train dev \
  --flow-method farneback \
  --limit-per-split 4 \
  --device cuda:0 \
  --encode-batch-size 8

cat "$Pilot/summary.json"
```

The pilot chooses two visual negatives and two visual positives per split,
without using manipulated-interval positions to sample features. This is a
software pilot, not a research result. It checks every selected RGB cache file,
copies its RGB vectors and timestamp/index arrays exactly, and reads original
MP4s to construct the new TSN contexts. The previous pooled flow cannot be expanded
back into dense flow maps.

Native consecutive-frame flow pairs are computed before forming five-pair
contexts. The first/last real pair is repeated only where a context reaches an
edge. Actual support intervals and repetition fractions are recorded. Context is
centered/offline, not causal. The original RGB anchor timeline remains unchanged.

Expected successful pilot: 8 selected/completed videos, zero errors,
`test_only=false`, and a final `Ready: .../manifest.json` message. Check the actual summary rather than
assuming success. Failed videos are not silently removed to change the cohort.

Smoke-test the **real native temporal model** on this cache:

```bash
python run_experiment.py smoke \
  --upstream "$Upstream" \
  --manifest "$Pilot/manifest.json" \
  --input-mode rgb-flow \
  --device cuda:0 \
  --batch-size 2 \
  --steps 2 \
  --amp none
```

Use batch 2 for this tiny pilot; it has only two examples of each visual class
per split. The pretrained TSN encoder is not involved in this smoke test. The
newly initialized temporal model, native loss and backward pass are tested; no
research checkpoint is saved.

## 4. Extract the complete matching train/dev cohort

Use a NEW full cache and remove `--limit-per-split`:

```bash
Features="$Project/features/tsn_flow_train_dev"

python extract_tsn_flow.py \
  --reference-manifest "$Reference" \
  --data-root "$Data" \
  --flow-weights "$FlowWeights" \
  --out "$Features" \
  --splits train dev \
  --flow-method farneback \
  --device cuda:0 \
  --encode-batch-size 8

cat "$Features/summary.json"
TSNManifest="$Features/manifest.json"
```

This uses all train/dev videos from the reference manifest, expected to be your
4,000/500 cohort. Counts are checked, not assumed. No extra face crops, original
PNG cache, new RGB encoder pass or full TSN feature archive is needed.

`--encode-batch-size` controls ONLY the frozen TSN extraction batches. It does
not control the UMMAFormer training batch. Lower it if extraction runs out of
GPU memory. The default extractor has bounded streaming memory, CPU thread limits,
per-video timing logs, and a per-video timeout (`--timeout`, default 1800 seconds).

### Resume and cache contents

Ctrl+C saves completed video records; rerun the identical extraction command.
Completed verified videos are reused and the incomplete one restarts. Keep the
same output/configuration. A settings/weights/code/environment change requires a
new cache. After a hard kill, remove `.extract.lock` only after confirming there
is no other writer. No original dataset files are deleted.

```
features/tsn_flow_train_dev/
  config.json
  summary.json
  errors.jsonl
  manifest.json                 # published only when every requested video succeeds
  clips/<split>/<video_id>.npz
  records/<video_id>.json
```

Each NPZ contains:

```
rgb                    [T, 512]     bit-identical to reference
flow                   [T, 2048]    pretrained TSN embedding, not action logits
timestamp_s            [T]          unchanged RGB anchor times
source_frame_index     [T]          unchanged native decoded indices
flow_pair_end_indices  [T, 5]
flow_context_start_s   [T]
flow_context_end_s     [T]
flow_pair_dt_s         [T, 5]
flow_repeat_fraction  [T]
```

The model loader still resamples by real timestamps to 768 bins. Interpolation
does not create new observed temporal detail. No authenticity labels are used as
features or by the flow encoder.

## 5. Check your intended training batch and benchmark cost

The wider input also changes trainable input/reconstruction layers. Do NOT assume
that batch 16 fits just because it fitted the 1024-channel baseline.

```bash
python run_experiment.py smoke \
  --upstream "$Upstream" \
  --manifest "$TSNManifest" \
  --input-mode rgb-flow \
  --device cuda:0 \
  --batch-size 16 \
  --steps 2 \
  --amp none
```

This uses a temporary model and no baseline weights. After successful smoke,
benchmark both projects in separate processes, with identical batch/precision:

```bash
python tools/benchmark_training.py \
  --project "$OldProject" \
  --manifest "$Reference" \
  --upstream "$Upstream" \
  --device cuda:0 --batch-size 16 --warmup-steps 3 --steps 10 \
  --out "$Project/benchmarks/old_pooled_flow.json"

python tools/benchmark_training.py \
  --project "$Project" \
  --manifest "$TSNManifest" \
  --upstream "$Upstream" \
  --device cuda:0 --batch-size 16 --warmup-steps 3 --steps 10 \
  --out "$Project/benchmarks/pretrained_tsn_flow.json"
```

Compare `median_seconds_per_batch`, `samples_per_second_at_median` and
`peak_cuda_allocated_mib`. Each benchmark reuses one actual feature batch, warms
up, synchronizes CUDA timing, and trains a temporary randomly initialized
UMMAFormer. It loads/saves no production checkpoint and excludes dataset I/O,
EMA, full-dev evaluation and checkpoint writing. It is a **compute benchmark**, not
an end-to-end epoch estimate or an accuracy result. Real training history also
records timing. No measured user-GPU speed or memory fit is claimed in advance.

Do not run benchmarks alongside an active training job on the same GPU.
If batch 16 is too large, test an even batch such as 8 and document the change.
For a strictly matched new comparison, also retrain the old representation with
that smaller training batch rather than claiming unchanged batching.

## 6. Train a new UMMAFormer with the baseline's saved settings

```bash
Run="$Project/runs/rgb_tsn_flow_s42"

python train_like_baseline.py \
  --baseline-checkpoint "$Baseline" \
  --reference-manifest "$Reference" \
  --manifest "$TSNManifest" \
  --upstream "$Upstream" \
  --out "$Run" \
  --device cuda:0 \
  --workers 2 \
  --dry-run
```

Inspect the printed settings/command, then repeat without `--dry-run` to train:

```bash
python train_like_baseline.py \
  --baseline-checkpoint "$Baseline" \
  --reference-manifest "$Reference" \
  --manifest "$TSNManifest" \
  --upstream "$Upstream" \
  --out "$Run" \
  --device cuda:0 \
  --workers 2
```

The helper verifies old checkpoint/reference cohort identity, unchanged official
upstream source, exact semantic train/dev membership, and RGB parent-cache
hashes. It copies the baseline's **maximum training horizon**, warmup, LR, weight
decay, patience, seed, AMP mode, EMA choice and batch size. For your tuned run this
is expected to be maximum60/warmup5/lr0.0003/wd0.05/patience8/seed42/FP32/EMA/batch16.
The selected baseline epoch11 is NOT used as the new maximum training duration.

**The old trained model weights are NOT copied.** Its checkpoint supplies settings
and reference identity only. The new 2560-channel model starts from its own new
initialization on frozen, pretrained cached inputs. Original loss semantics and
maximum-proposal video readout are retained; no internal-head or negative-loss
modification is bundled into this experiment.

An explicit `--batch-size 8` override is available to the helper when memory
requires it; the changed protocol is reported. It is not silently auto-reduced.
A larger input also changes parameter count, so this is a representation/system
ablation rather than a pure same-parameter causal test.

Early stopping and best-checkpoint selection use **dev video-level visual
ROC-AUC**. Output includes `history.json`, `run_config.json`, `last.pt`,
`best_video_auc.pt`, `best_temporal_ap50.pt`, and `best_dev/`.
Use the same full command plus `--resume "$Run/last.pt"` after interruption. Only
this new run's checkpoint is accepted; old RGB+flow/TSN archives/internal-head
outputs cannot initialize or resume it. Epoch checkpoints are not per-batch
checkpoints.

## 7. Evaluate and compare maximum-proposal scores on DEV

Evaluate the new model using THIS new project's runner:

```bash
NewEval="$Project/evaluations/tsn_flow_dev"

python run_experiment.py evaluate \
  --upstream "$Upstream" \
  --manifest "$TSNManifest" \
  --checkpoint "$Run/best_video_auc.pt" \
  --split dev \
  --out "$NewEval" \
  --device cuda:0
```

The input mode, precision and EMA selection are loaded from the checkpoint. TSN
weights are not required for this cached-feature evaluation. Do not evaluate the
new checkpoint with the old 1024-channel project.

To obtain a fresh baseline report without changing the old source:

```bash
BaseEval="$Project/evaluations/old_flow_reference_dev"
(
  cd "$OldProject"
  python run_experiment.py evaluate \
    --upstream "$Upstream" \
    --manifest "$Reference" \
    --checkpoint "$Baseline" \
    --split dev \
    --out "$BaseEval" \
    --device cuda:0
)
```

The baseline should reproduce your existing proposal AUC0.8284. Never substitute
the internal-classifier experiment's nested `metrics.json` here. An existing
normal baseline evaluation directory can be used instead of reevaluating.

```bash
python tools/compare_readouts.py \
  --baseline "$BaseEval" \
  --candidate "$NewEval" \
  --baseline-manifest "$Reference" \
  --candidate-manifest "$TSNManifest" \
  --out "$Project/evaluations/tsn_vs_pooled_flow"
```

The helper checks exact train/dev semantic cohorts and RGB-parent provenance,
per-video IDs/labels/groups, continuous score validity, video counts, and AUC/AP
consistency with the saved score files. It writes `comparison.csv` and
`comparison.json`, with **TSN minus baseline** differences for video metrics and
temporal AP. It does not average the models or independently authenticate every
training setting; retain both run configs and the training-helper record.

A higher dev AUC is a selection result, not independent test performance.
A worse result remains a useful ablation. Compare repeated seeds before strong
claims, and retain localization metrics even when selecting on video AUC.

## 8. Optional TV-L1 front end (a separate experiment)

An OpenCV-contrib build exposing `cv2.optflow.DualTVL1OpticalFlow_create` or its
alternative factory can use `--flow-method tvl1`. Test availability first:

```bash
python check_install.py --flow-method tvl1
```

If missing, this option stops explicitly. Use a **separate extraction environment**
for any OpenCV changes; do not install overlapping opencv-python/contrib/headless
packages or modify the baseline environment during a run. The default Farneback
experiment does not require contrib. TV-L1 extraction could be substantially
more expensive on CPU and was not executed here.

TV-L1 is closer to conventional TSN optical-flow preparation, but this code still
retains the fixed earlier RGB geometry/timing and avoids intermediate JPEGs. It
is not asserted to reproduce the authors' denseflow files. TV-L1 changes both the
flow estimator and its encoded representation relative to the baseline. Give it
a new cache and a separate run. The TSN checkpoint is the same.

## 9. Test later, after selection

Do not tune on the held-out500 videos. Once the representation, training and
readout are frozen, the reference cache must include the same train/dev rows plus
the planned test rows. Extract with `--splits train dev test` and the unchanged
TSN configuration, then use the new model's evaluator with `--split test` and its
stored dev threshold. Matching train/dev records are reused. Split fingerprints
are checked. The comparison helper above is deliberately DEV-only.

## 10. Timing and storage summary

- **Extraction:** more work than simple pooling, because dense flow over five-pair
  contexts and TSN inference are needed once. Original RGB is reused. Store
  embeddings, not millions of flow images.
- **Training:** no ResNet18 or TSN forward/backward, only cached-feature UMMAFormer.
  Width changes1024->2560; trainable layers, activation memory and compute can
  increase. The complete training-time factor must be measured.
- **Storage:** combined numeric feature width is2.5x. Compressed cache size is not
  guaranteed to scale identically; record pilot bytes. Original cache is retained.
- **Inference on new videos:** both RGB/flow extraction and the temporal model are
  needed. Caching removes repeated *training* work, not the front-end cost of a
  genuinely new video. This is one detector pipeline, not a two-checkpoint ensemble.

## Validation and references

See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for exact tests executed and
[METHODS.md](METHODS.md) for primary source links and deviations. Your real LAV-DF
files, actual pretrained checkpoint download/parity, native UMMAFormer execution,
CUDA, WSL and final accuracy remain unverified here. The mandatory weight and
real-feature smoke checks above establish them locally before a reported run.
