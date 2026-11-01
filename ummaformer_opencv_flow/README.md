# UMMAFormer + RGB + OpenCV Farneback, for your existing LAV-DF cohort

**New standalone pipeline. No TSN feature archive, CLIP checkpoint, or 10-FPS PNG
folder is required. Do not overwrite your old CLIP or UMMAFormer projects.**

This project supplies the previously missing feature-extraction stage and an
updated UMMAFormer training/evaluation integration. It reads the exact cohort
from your existing `videos.jsonl` and accesses MP4s in the full LAV-DF directory.
It does not assume that a physical `LAV-DF-5000` video folder exists.

## Architecture and method status

```
Original MP4s selected by the existing CLIP videos.jsonl
  |
  +-- RGB -> frozen pretrained ResNet-18 -> 512 values per sampled frame
  |
  +-- adjacent native decoded frames -> OpenCV Farneback dx/dy
       -> 16 x 16 area-pooled velocity grids -> 512 values
  |
  +-- concatenate -> 1024 values at each sampled timestamp
       -> timestamp-aware resampling to 768 temporal bins
       -> actual official UMMAFormer temporal model
       -> forgery proposals + video visual score
```

The temporal model, native losses, optimizer grouping, scheduler and NMS are
imported from your `ymhzyj/UMMAFormer` checkout. The input channel count is changed
from 4096 to **1024**, including the reconstruction input/output dimensions that
are constructed from the model configuration. We do not pad custom features to
4096 and call them TSN. The original temporal architecture settings are otherwise
retained. The optional RGB-only run uses 512 input channels.

**This is a custom input-representation adaptation, NOT reproduction of the TSN
features or published UMMAFormer AUC.** ResNet-18 is ImageNet-pretrained, not a
pretrained face-forgery detector. Motion is a deterministic coarse flow descriptor,
not a pretrained TSN flow embedding. This version deliberately does NOT freeze a
randomly initialized motion CNN and pretend those are learned motion features.
UMMAFormer's trainable input projections and temporal network learn how to use
the fixed RGB and motion inputs.

The ResNet is evaluated once during extraction, then absent from the training
process. Farneback runs on CPU. The RGB encoder can run on CUDA. No audio is used.

## Validation status

19 local CPU tests passed, including actual Farneback on translated textured
images, actual MP4 decode, actual ResNet-18 architecture extraction with explicitly
random **test-only** weights, timestamp interpolation, modality labels, cache
checks, native-model factory argument checks, and checkpoint/evaluation/resume
orchestration with an explicitly labelled test double. A separate extraction CLI
test processed four synthetic videos and then resumed all four without re-extraction.

**The real upstream UMMAFormer forward/backward pass, native NMS, downloaded RGB
pretrained weights, your LAV-DF data, CUDA memory, runtime and accuracy were NOT
executed here.** The upstream download was unavailable in this environment. Use
the real-model and real-feature smoke tests below before starting the full run.
Read `VALIDATION_REPORT.md` for the exact scope. Software tests do not establish
scientific performance or pretrained numerical parity.

## 1. Install alongside your existing project

Extract this folder to:

```
/home/evan/dsa4266/ummaformer_opencv_flow/
```

Use your Linux/WSL shell, not PowerShell:

```bash
conda activate ummaformer
cd /home/evan/dsa4266/ummaformer_opencv_flow
bash setup.sh --reuse-upstream /home/evan/dsa4266/ummaformer_lavdf5000/upstream
```

This reuses the official checkout from the earlier package via a symlink. It does
not edit its Python/model source or your old experiment files. Do not update that
checkout during a run. Its commit and source hashes are recorded in new checkpoints.

The setup checks existing `torchsort` and temporal NMS. The previous CUDA-toolkit
fix may still be required if those extensions were never successfully installed.
Farneback itself introduces no CUDA compilation requirement.

The RGB encoder additionally requires **torchvision matched to torch**. For your
reported `torch 2.7.1+cu128`, the matching version is `torchvision 0.22.1`. If the
setup explicitly reports that torchvision is missing/broken, and torch is STILL
2.7.1+cu128, run:

```bash
python -m pip install --no-deps torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
bash setup.sh --reuse-upstream /home/evan/dsa4266/ummaformer_lavdf5000/upstream
```

Do not reinstall/downgrade working PyTorch, CUDA or the graphics driver. A working
OpenCV import is reused; otherwise setup installs an OpenCV version compatible
with NumPy 1.x. Do not install multiple OpenCV wheel variants in the same environment.

If no old checkout exists, `bash setup.sh` clones the official repository. This
requires internet access and working extension build tools.

## 2. Smoke-test the real, newly configured temporal model

```bash
python run_experiment.py smoke \
  --input-mode rgb-flow \
  --device cuda:0 \
  --batch-size 2 \
  --steps 2 \
  --amp none
```

This uses synthetic feature tensors, but the actual upstream model. It checks the
1024-channel model, native losses, backward pass and proposal inference, then
prints measured parameter counts and peak allocated CUDA memory. No checkpoint
is saved. Do not assume the parameter count is still 49M after changing the input
representation. Resolve any error before continuing.

## 3. Extract a small REAL-video pilot

Your known paths are already filled in:

```bash
python extract_rgb_flow.py \
  --data-root /mnt/c/Users/wuche/Downloads/LAV-DF \
  --cohort-videos /home/evan/lavdf5000_visual_manifest/videos.jsonl \
  --out features/rgb_flow_pilot \
  --splits train dev \
  --sample-fps 10 \
  --limit-per-split 4 \
  --device cuda:0 \
  --encode-batch-size 32
```

The first real extraction downloads torchvision's ImageNet ResNet-18 weights
(about 45 MB, not the huge TSN archive) into the standard torch model cache.
If already cached, they are reused. There is no automatic random-weight fallback.
For an offline machine, add `--rgb-weights /actual/path/resnet18-f37072fd.pth` using
the official torchvision checkpoint; its filename hash prefix and tensor shapes
are checked and its full SHA256 is recorded.

The pilot selects two visually negative and two visually positive videos per
requested split, using seed 42. This is explicitly a software pilot, not your final
cohort. A normal full run with no limit uses EVERY train/dev video listed in the
reference, without creating another random subset. Counts are printed; they are
not assumed to be 4000/500 if your audited CLIP manifest contains different counts.

Read the summary:

```bash
cat features/rgb_flow_pilot/summary.json
```

Expected: 8 completed videos, 0 errors, not interrupted, `test_only=false`.
Warnings such as metadata-duration discrepancies remain in `warnings.jsonl`.

Now test one batch of these real cached features:

```bash
python run_experiment.py smoke \
  --manifest features/rgb_flow_pilot/manifest.json \
  --input-mode rgb-flow \
  --device cuda:0 --batch-size 2 --steps 2 --amp none
```

**Do not use `--test-random-rgb` in your experiment.** That switch exists only for
offline software tests; train/evaluate refuse its cache. Do not remove that guard.

## 4. Extract the complete train/dev cohort

Use a NEW full output directory and remove the pilot limit:

```bash
python extract_rgb_flow.py \
  --data-root /mnt/c/Users/wuche/Downloads/LAV-DF \
  --cohort-videos /home/evan/lavdf5000_visual_manifest/videos.jsonl \
  --out features/rgb_flow_train_dev \
  --splits train dev \
  --sample-fps 10 \
  --device cuda:0 \
  --encode-batch-size 32
```

Only `train` and `dev` are extracted. Your original MP4s, metadata, CLIP manifest
and PNGs are never modified. The final cache looks like:

```
features/rgb_flow_train_dev/
  config.json
  summary.json
  errors.jsonl
  warnings.jsonl
  manifest.json             # written only after all requested videos succeed
  clips/train/<video-hash>.npz
  clips/dev/<video-hash>.npz
  records/<video-hash>.json
```

**No old `prepare_experiment.py` command is needed.** The extractor creates the
new manifest directly. Do not pass this cache to the old TSN-only loader.

If interrupted, rerun the IDENTICAL command. Matching completed videos are checked
and resumed; incomplete/failed clips are retried. No `--retry-errors` flag is
needed. If settings, weights, source code or the encoding environment change, use
a new output directory. Do not write to a cache currently used by training.

A file lock prevents two extractors writing to the same cache. After a hard kill,
remove `.extract.lock` only after checking that no extractor is still running.
Nothing automatically deletes original data or partially reuses invalid features.

An extraction failure prevents `manifest.json` from being written. Do not edit
error records into successes, silently drop failed videos, or train on a partial
cohort just to obtain an AUC.

## 5. Full-cache smoke test and training

```bash
python run_experiment.py smoke \
  --manifest features/rgb_flow_train_dev/manifest.json \
  --input-mode rgb-flow \
  --device cuda:0 --batch-size 2 --steps 2 --amp none

python run_experiment.py train \
  --manifest features/rgb_flow_train_dev/manifest.json \
  --input-mode rgb-flow \
  --out runs/umma_rgb_flow_s42 \
  --device cuda:0 \
  --batch-size 2 \
  --workers 2 \
  --epochs 15 \
  --warmup-epochs 5 \
  --lr 0.001 \
  --weight-decay 0.05 \
  --patience 8 \
  --seed 42 \
  --amp none
```

These are starting settings carried over from the earlier integration, NOT tuned
hyperparameters or a claimed reproduction recipe for these custom features. The
upstream YAML has 10 main epochs and defaults add 5 warmup epochs; we use 15 total.
Batch 2 differs from the authors' batch 32. Smaller learning rates can be selected
on dev as a separately recorded run. Start with FP32 rather than inheriting your
previous CLIP FP16 overflow problems.

Every batch contains genuine/visually negative and visually manipulated videos,
which avoids empty branches in the native reconstruction/detection loss. The
smaller class is oversampled; every training video appears at least once per
epoch. `--batch-size` counts VIDEOS, not images. This is not the same unit of work
as 16,000 CLIP frames per epoch.

Best-model selection uses dev video visual ROC-AUC, with EMA weights by default.
Temporal AP selects a separate diagnostic checkpoint. Test data never select the
checkpoint, learning rate or decision threshold. No full-LAV-DF-trained UMMAFormer
checkpoint is loaded: the temporal model starts newly initialized.

Outputs:

```
runs/umma_rgb_flow_s42/
  run_config.json
  history.json
  last.pt
  best_video_auc.pt
  best_temporal_ap50.pt
  best_dev/{metrics.json,video_scores.jsonl,proposals.jsonl}
  last_dev/{metrics.json,video_scores.jsonl,proposals.jsonl}
```

Resume with the same command/settings and append:

```bash
--resume runs/umma_rgb_flow_s42/last.pt
```

Do not resume TSN, CLIP or RGB-only checkpoints in RGB+flow runs. The input shapes,
checkpoint format, source hashes and feature fingerprints are checked. Preserve a
run-specific source copy. Upstream checkpoints are not interchangeable with these.

## 6. RGB-only ablation using the same extracted cache

No extraction is repeated. Use a different run directory:

```bash
python run_experiment.py train \
  --manifest features/rgb_flow_train_dev/manifest.json \
  --input-mode rgb \
  --out runs/umma_rgb_only_s42 \
  --device cuda:0 --batch-size 2 --workers 2 \
  --epochs 15 --warmup-epochs 5 --lr 0.001 --weight-decay 0.05 \
  --patience 8 --seed 42 --amp none
```

This changes the input channels to 512 and omits flow. It provides an explicit
ablation; input-layer parameter counts also differ and should be reported.

## 7. Evaluate

```bash
python run_experiment.py evaluate \
  --manifest features/rgb_flow_train_dev/manifest.json \
  --checkpoint runs/umma_rgb_flow_s42/best_video_auc.pt \
  --split dev \
  --out evaluations/umma_rgb_flow_s42_dev \
  --device cuda:0
```

Input mode is read from the checkpoint. Video score is maximum final forgery
proposal confidence, or zero when no proposal exists. Every evaluated video is
retained. Report video ROC-AUC, average precision, dev-threshold F1/precision/recall,
balanced accuracy, and temporal AP at tIoU 0.50/0.75/0.95. False proposals on genuine
and audio-only videos count as false positives.

The temporal AP here is labelled **all-cohort visual temporal AP**. Do not equate
it automatically with paper results using a different cohort or annotation target.

After locking decisions, test requires a reference videos.jsonl containing the
chosen test IDs and unchanged train/dev rows. Run extraction with `--splits train
dev test`; matching train/dev cache entries can be resumed. The new manifest must
retain the original train/dev feature fingerprints. Then evaluate with `--split
test` and a fresh output directory; the stored dev threshold is reused.

## 8. Optional sequential pipeline

After setup and both pilot checks succeed:

```bash
bash run_pipeline.sh
```

The provided defaults use your exact source/cohort paths. This performs complete
train/dev extraction, real-cache smoke and training with `set -euo pipefail`.
Override `FEATURES`, `RUN`, `DEVICE` or `BATCH` as environment variables when needed.
A failed extraction or smoke stops the pipeline. Training never overwrites a
nonempty run directory unless explicitly resuming.

## Temporal and spatial conventions

- All original frames are decoded sequentially and verified against FFprobe frame
  count/PTS. The code never invents timestamps as frame_index/FPS.
- RGB geometry uses bilinear shorter-edge 256, center crop 224, matching the
  documented ResNet-18 inference geometry. Flow uses the SAME unnormalized crop.
  This is an ordinary model crop, not new face detection/alignment.
- Retain the first actual frame at/after each 10-Hz grid point, deduplicated, and
  store its actual timestamp and source index. `--sample-fps 0` keeps every frame.
- Flow at retained index i is calculated between native frames i-1 and i. The
  decoder retains the previous native frame even when no feature was saved there.
  Computing only retained native pairs is equivalent to computing every pair and
  then selecting those pairs. It does not use adjacent 10-FPS PNGs.
- The first frame has no predecessor: its flow is explicitly zero and
  `flow_valid=false`. All subsequent retained flows are valid; this convention is
  not used to hide missing videos or decoding errors.
- Flow dx/224/dt and dy/224/dt are crop widths/heights per second. Each component
  is area-averaged to 16x16; flatten dx first, then dy, producing 512 values. No HSV,
  per-video min/max normalization, clipping, random flow weights, or annotations
  enter this descriptor.
- NPZ contains `rgb [T,512]`, `flow [T,512]`, `timestamp_s [T]`,
  `source_frame_index [T]`, `flow_dt_s [T]`, and `flow_valid [T]`.
- Loader interpolates features by actual timestamps onto 768 bin CENTERS spanning
  the decoded duration; endpoints are held. Mapping is `index=t*768/duration-0.5`.
  Inference uses its exact inverse. Upsampling does NOT add missing temporal detail.
- Metadata duration mismatch is logged; decoded extent controls timing. Nonzero
  display rotation/non-square pixels and intervals beyond decoded extent fail
  explicitly rather than silently corrupt geometry or labels.
- `modify_video` determines the visual target. Audio-only fakes are visual negatives.
  No interval is inferred from frame appearance; no label-dependent feature sampling.

## Scientific limitations

This is a lightweight practical baseline, not a claim that hand-engineered flow is
as discriminative as a trained TSN flow encoder. Coarse pooling and 10-Hz sampling
can lose subtle short artifacts. Optical flow also responds to normal head/camera
motion, so it is not itself evidence of forgery. The pretrained RGB encoder is
fixed; performance depends on how informative its representations are.

Keep the same audited videos, splits and visual labels as CLIP. Report different
encoders, input preprocessing, temporal sampling, objectives, video pooling and
parameter counts. A higher AUC would compare complete pipelines, not isolate a
single architectural cause. Keep the CLIP outputs and test split untouched.

## Sources and licenses

Official references checked for this implementation:

- UMMAFormer code and license: https://github.com/ymhzyj/UMMAFormer
- Visual configuration: https://github.com/ymhzyj/UMMAFormer/blob/main/configs/UMMAFormer/lavdf_tsn.yaml
- Native model/loss/inference: https://github.com/ymhzyj/UMMAFormer/blob/main/libs/modeling/av_recoverynonorm_meta_arch.py
- ResNet-18 and preprocessing: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- OpenCV Farneback declaration/documentation: https://github.com/opencv/opencv/blob/4.x/modules/video/include/opencv2/video/tracking.hpp
- Matching torch/torchvision versions: https://pytorch.org/get-started/previous-versions/

Only integration/extraction code is included. No UMMAFormer source, pretrained
weights, LAV-DF videos, or authors' features are redistributed. Respect each
upstream license and dataset/model terms separately. See LICENSE for this glue code.
