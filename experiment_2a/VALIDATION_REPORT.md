# Validation report — Experiment 2A

Validation date: 2026-09-19.

## Executed automated tests

`python -m pytest -q` completed with **31 passed**. The test suite covers:

- Strict modality flags, localized annotations, interval union/boundaries, and source-group split checks.
- Fixed-stride plus tail coverage, measured audio/video offsets, missing annotated audio, and actual FFmpeg extraction.
- Finite log-mel values, expected `[128,404]` input shape, fixed-reference power behavior, and training-only normalization.
- Exact shared waveform/spectrogram windows, including repeat padding of short clips; no invented additional source-time bounds.
- Visual-only manipulations remaining acoustic negatives in both model branches.
- Full ResNet-18 forward/backward and 512-D embeddings.
- Full AASIST port forward/backward, 297,866 parameters and 160-D embeddings.
- AASIST spoof/bonafide output orientation and equivalence of the binary-logit loss to the correctly mapped two-class loss.
- Tensor-only checkpoint loading using a random state-dictionary fixture, changed-label cohort fingerprints, and invalid waveform offsets.
- EER endpoint/tie behavior and paired source-group bootstrap behavior with identical predictions.

The state-dictionary test uses locally generated random parameters, NOT downloaded official pretrained weights.

## Executed end-to-end CLI checks

Generated **12 non-speech synthetic videos**, four in each of train/dev/test,
using tones and artificial modality/interval labels. These are software fixtures,
not genuine/forged speech examples. Completed:

1. FFmpeg/ffprobe extraction, window labelling, WAV/spectrogram writing: 12 successful clips, zero failures.
2. Repeated preparation: all 12 matching clips resumed.
3. Both ResNet-18 and the bundled AASIST port trained for two epochs on CPU, including an incomplete gradient-accumulation group.
4. Safe loading of both best checkpoints and locked-threshold test evaluation.
5. Timestamped embedding export for all 12 clips from each model; feature dimensions and finite values checked.
6. Matched comparison and 100-replicate paired source-group bootstrap path.

The orchestration tool reached its wall-clock limit after AASIST training; the
remaining evaluation/export/comparison CLI steps were run separately and passed.
No code failure is being represented as a successful uninterrupted script run.
The earlier one-epoch end-to-end workflow also completed for both models.

No synthetic accuracy/AUC is presented as evidence of deepfake-detection quality.

## Not executed / not established

Actual LAV-DF media were not provided or processed here. No trained research
checkpoint, LAV-DF detection accuracy, temporal-localization accuracy or superiority
of either model is claimed.

The live download of official AASIST source/weights could not be performed from
the execution container. The source was inspected through web retrieval, and the
bundled file is a documented local port, not a byte-identical downloaded checkout.
The optional `verify_upstream.py` numerical comparison against a separately
downloaded upstream implementation remains unexecuted. Strict compatibility with
the actual official pretrained checkpoint has not been established by the random
state-dictionary fixture. The downloader and optional upstream/transfer commands
are included for execution in an internet-connected environment.

ImageNet pretrained-weight downloads, GPU/CUDA execution, Windows runtime,
full-dataset memory/storage/throughput and multi-worker performance were not tested.
No calibrated probabilities, ASVspoof reproduction or deployment safety guarantee
is established. Three-seed research training and held-out evaluation remain user
experiments, not results supplied by this package.

## Environment

Python: 3.13.5
Platform: Linux-6.18.44-x86_64-with-glibc2.41
Device: CPU, FP32.

- numpy: 2.3.5
- librosa: 0.11.0
- soundfile: 0.13.1
- scipy: 1.17.0
- scikit-learn: 1.8.0
- tqdm: 4.67.3
- pytest: 9.0.2
- torch: 2.10.0+cpu
- torchvision: 0.25.0+cpu
