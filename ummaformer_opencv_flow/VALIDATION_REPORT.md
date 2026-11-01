# Validation report — OpenCV-flow integration

## Executed locally

Environment: Python 3.13.5, torch 2.10.0+cpu, torchvision
0.25.0+cpu, OpenCV 4.13.0, NumPy 2.3.5; CPU only.
FFmpeg and FFprobe were available. CUDA available: False.

`python -m pytest -q tests`: **19 passed** (last run: 4.97 s).
`python -m compileall -q .`: passed.
`bash -n setup.sh run_pipeline.sh`: passed.
Extractor and train CLI `--help`: passed.

Tests cover:
- Exact videos.jsonl joining, selected splits, duplicated/missing IDs and metadata disagreement.
- Visual-only labels: audio-only manipulation is a visual negative.
- Native-frame index selection; actual FFprobe PTS and FFmpeg frame count.
- Actual OpenCV Farneback for static and known translated textured images; sign/velocity units.
- Actual torchvision ResNet-18 architecture extraction using explicitly RANDOM diagnostic weights.
- Cache shapes, finite values, first-flow validity, tampering/integrity checks.
- Timestamp-aware resampling and feature-bin/seconds inverse mapping.
- Mixed-class batches and complete source coverage.
- Temporal matching, negative-video false proposals, classification metrics.
- 1024/512-channel factory arguments using an explicit test double, not a native model.
- Finite-gradient guard; 2-epoch checkpoint/evaluation/resume orchestration using an explicit test double.
- Training refusal for diagnostic random-encoder caches.

A separate CLI integration test generated four synthetic 1-second MP4s at 20 FPS
(two train and two dev with artificial test labels). The first run extracted 40
feature steps total at 10 FPS, generated a complete manifest, and reported zero
errors. The second identical command resumed all four caches without re-extraction.
Attempted training on that RANDOM-test cache failed intentionally with exit 2.
These synthetic clips and random features are NOT a deepfake performance benchmark.

## Not executed / not claimed

- Real downloaded ImageNet pretrained weights or pretrained numerical parity.
- The official upstream UMMAFormer forward/backward pass and C++ NMS.
- The user's actual LAV-DF files, cohort contents, feature distribution or timestamps.
- CUDA training, 8-GB memory fit, speedup, convergence, ROC-AUC or temporal AP.
- pip/setup against the user's current environment.

The upstream checkout could not be downloaded inside this runtime. Its official
configuration and model interfaces were inspected through primary-source web pages;
source is not reconstructed or silently replaced in production. The package imports
the actual checkout on the user's machine. Mandatory model and real-feature smoke
commands are included in README.md. Any downstream smoke error must be resolved
before a long extraction/training run.

## Interpretation

Local tests establish the implemented extraction/data/metric/orchestration behavior
on the exercised cases. They do not prove full-model compatibility or scientific
performance. Model input channels, feature semantics and preprocessing differ from
the published TSN experiment. Record the printed runtime parameter counts rather
than repeating an unverified 49M figure.
