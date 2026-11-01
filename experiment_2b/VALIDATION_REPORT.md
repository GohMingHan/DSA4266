# Experiment 2B — validation report

## Execution environment

Executed on 2026-10-04 in Linux with Python 3.13.5, PyTorch 2.10.0+cpu,
torchvision 0.25.0+cpu, NumPy 2.3.5 and scikit-learn 1.8.0.
No NVIDIA GPU was available. The user's CUDA/WSL/Kaggle environment was not used.

## Automated tests

Command, from this package directory:

```bash
python -m pytest -q
```

**37 tests passed** in the final run. They cover:

- Overall labels (`modify_audio OR modify_video`), modality categories and
  connected source-family separation across partitions.
- Unsafe paths, missing/unconfigured model slots, input/output schema checks,
  nonfinite embeddings and invalid source timestamps.
- Rejection of scalar/two-class scores as feature sequences; shapes alone cannot
  authenticate a wider embedding, so the readout must still be verified.
- Mean/mean-max pooling, independent feature dimensions, strict video-ID pairing,
  checksum changes and failure rather than silent cohort intersection.
- Training-only normalization and unchanged train/dev fingerprints when the
  test partition is subsequently added.
- Concatenation, audio-only and visual-only head training/evaluation on explicit
  synthetic data; controls ignore the unused modality.
- Development-only threshold selection, early stopping, strict state loading and
  checkpoint provenance.
- CPU checkpoint resume reproducing uninterrupted final parameters in the tested
  deterministic configuration (not a cross-device reproducibility promise).
- Paired prediction matching and source-group bootstrap reporting.
- Read-only UMMAFormer-style FPN/point hooks, masks, coordinate conversion,
  frozen parameters and cleanup with a deliberately small model test double.
- Imported-embedding/checkpoint identity requirements, real-video export cache
  schema, per-video resume, appended test splits and partial-pilot markers.

Python compilation and `python pipeline.py --help` also passed.

## Additional original audio-model bridge checks

Outside the 37-test suite, the **actual original Experiment 2A model classes**
were used in separate Python processes:

| Model | Input | Export shape | Result |
|---|---|---|---|
| Single-channel ResNet-18 | Synthetic log-mel, 128 x 404 | [1, 512] | Passed |
| Original bundled AASIST architecture | Synthetic 64,600-sample waveform | [1, 160] | Passed |

These checks exercised the original checkpoint loader, dataset/normalization,
model construction and `forward_features` inference through the new bridge.
The model states were **randomly initialized, synthetic fixture checkpoints**,
not downloaded pretrained weights or the user's trained models. Model eval mode
and frozen parameters were checked. No detection accuracy is inferred.

The separate checks are not packaged with legacy source: they used the previous
Experiment 2A archive mounted in the working environment.

## Additional command-line integration check

The public CLI trained the concatenation head for three epochs on synthetic
paired vectors, wrote best/last checkpoints and then evaluated a separate
synthetic test partition. It used 32 training, 16 development and 16 test videos
and the explicit `--smoke-test` permission. Loading, prediction output, subgroup
reporting and the saved development threshold worked. Synthetic accuracy values
are software fixtures, not research results and are not presented as evidence
of model quality.

## Not validated here

- The user's real LAV-DF recordings, feature manifests or trained checkpoints.
- Native UMMAFormer with its actual compiled torchsort/NMS forward pass.
- The real Forensics Adapter checkpoint/backbone forward pass.
- CUDA, mixed precision, GPU memory, Windows/WSL and Kaggle execution or timing.
- A custom Kaggle checkpoint from the currently unavailable
  `evaluate_test_models.py` or its training code.
- Numerical parity of unknown external model implementations or preprocessing.
- Deepfake accuracy, generalization, improvement from fusion, or calibration.

The native UMMAFormer and Forensics bridges were implemented against inspected
original local interfaces and the documented/read-only upstream feature path.
Their original strict source/checkpoint checks remain active. The short actual
checkpoint extraction pilot is still required locally.

## Supported scope

The package is a working frozen-embedding fusion implementation with explicit
bridges for known prior project formats. It is **not a universal loader** for any
file named `.pt`/`.pth`. Unconfigured slots stop instead of falling back to random
models. Unknown CNN/YOLO/notebook checkpoints require their correct exporter or
an implemented custom adapter. The example custom adapter intentionally raises
`NotImplementedError` until its model-specific implementation is supplied.

There are no model weights, real dataset files or original project sources in
this ZIP. Model slots contain configuration templates only. Input projects,
media, feature caches and selected checkpoints are never overwritten.
