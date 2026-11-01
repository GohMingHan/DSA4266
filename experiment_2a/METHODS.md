# Experiment 2A: method definition and research attribution

## Research question

Under shared LAV-DF audio supervision, how does a log-mel/ResNet-18 system compare
with a raw-waveform AASIST system? This is a **two-system comparison**, not a
representation-only causal ablation: backbone structure and parameter count also
change. An initialization-control study trains both from scratch. A second track
can use their respective pretrained weights, explicitly noting different sources.

## What is followed versus adapted

| Component | Source | What this implementation does |
|---|---|---|
| AASIST architecture | Jung et al., AASIST; official clovaai/aasist model | Preserves sinc bank, six residual blocks, spectral/temporal graph attention, two heterogeneous branches, pooling/readout and two-class head. |
| AASIST configuration | Official AASIST.conf | Uses 64,600 samples, 70 sinc bands, official filter widths, graph dimensions, pool ratios and temperatures. |
| Short-utterance handling | Official data_utils.py | Repeats short signals, and applies that same signal to the spectrogram branch. |
| Long-clip handling | LAV-DF adaptation | Replaces original random-train/first-eval crops with deterministic sliding windows plus tail coverage; labels each source interval. |
| ResNet | He et al.; torchvision implementation | Converts the input convolution to one channel and replaces the image head with a binary head. |
| Log-mel frontend | Explicit experimental specification using librosa | 128 Slaney mel bands, FFT 512, Hann 400, hop 160, fixed-reference log power; no claim of reproducing a specific spoofing paper's frontend. |
| Targets | Official LAV-DF modify_audio/fake_periods | Uses interval-overlap window targets and acoustic video targets; visual-only edits are negative for audio. |
| Optimization | Shared experimental protocol | AdamW, cosine schedule, balanced window sampling, BCE, dev AUC selection and dev F1 thresholds; not the original full ASVspoof training recipe. |
| Evaluation | Shared LAV-DF protocol | Window/video discrimination and subtype errors; no reproduction of ASVspoof EER/t-DCF results or LAV-DF localization AP. |

## AASIST source provenance

The bundled file is **a local port of inspected MIT-licensed source**, not a
byte-identical checkout. It retains the official parameter key names and shapes.
The port condenses formatting, explicitly converts sinc-filter arguments to
NumPy arrays for NumPy 2.x compatibility, and omits two unused local expansions.
It intentionally preserves the official residual block's use of the original
input in its first convolution. It does not invent a smaller fallback network.
The configured model contains 297,866 parameters and yields a 160-D embedding.

Source inspected on 2026-09-19:
https://github.com/clovaai/aasist/blob/main/models/AASIST.py
https://raw.githubusercontent.com/clovaai/aasist/main/config/AASIST.conf
https://raw.githubusercontent.com/clovaai/aasist/main/data_utils.py

Model/source hashes are saved with trained checkpoints. They identify this
specific local implementation; they are not represented as an upstream commit.
`download_aasist.py` optionally obtains the original source from a resolved,
locked Git commit, with a download/hash manifest. `verify_upstream.py` compares
both networks under identical parameters; the network-dependent comparison was
not executed here. `--aasist-dir` supports directly running the upstream file.

## Suggested report wording

"We compared a single-channel ResNet-18 using log-mel spectrograms with AASIST,
following the latter's published architecture and adapting its input protocol to
localized LAV-DF audio manipulation. Both models used the same 16-kHz audio,
64,600-sample windows, interval-overlap labels and dataset partitions. We selected
checkpoints by development-set video-level acoustic ROC-AUC and fixed decision
thresholds using development data. The spectrogram frontend and shared training
protocol are specified experimental choices, rather than a reproduction of one
previous paper's complete pipeline."

Describe whether the bundled port or original upstream file was used; record
source hashes/revision. Report scratch and transfer-learning tracks separately.
Include seeds, selected epoch, training-window budget, normalization, padding,
aggregation, exclusions, class prevalence and per-modality-category performance.

## Limitations to report

A window overlapping a tiny edit is weakly labelled positive, not entirely fake.
Repeated short recordings introduce artificial boundaries. The 4.04-second
context may dilute short edits; maximum video pooling may amplify isolated false
alarms and depend on video duration. Balanced window sampling does not balance
source recordings. Source-group checks do not independently reconstruct identity
splits. Fixed source artifacts, silence, recording quality, and generation cues
can be shortcuts. High in-dataset scores do not establish cross-dataset validity.
Preserve test protocols and assess those risks before claiming generalization.

## Primary references

1. Jung et al. **AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph
   Attention Networks.** ICASSP 2022; arXiv preprint 2021.
   https://arxiv.org/abs/2110.01200
   Official source: https://github.com/clovaai/aasist
2. He et al. **Deep Residual Learning for Image Recognition.** CVPR 2016.
   https://arxiv.org/abs/1512.03385
   Torchvision: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
3. Cai et al. **Glitch in the Matrix: A Large Scale Benchmark for Content Driven
   Audio-Visual Forgery Detection and Localization.** CVIU 2023.
   https://arxiv.org/abs/2305.01979
   Official metadata/loader: https://github.com/ControlNet/LAV-DF/blob/master/dataset/lavdf.py
4. librosa frontend documentation:
   https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html
   https://librosa.org/doc/0.11.0/generated/librosa.power_to_db.html
5. FFmpeg/ffprobe documentation:
   https://ffmpeg.org/ffmpeg.html
   https://ffmpeg.org/ffprobe.html

No claim is made that the custom log-mel ResNet baseline reproduces the Pham et al.
method mentioned earlier in the conversation. The distinction is intentional.
