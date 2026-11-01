# Validation report

## Executed environment

Python 3.13.5 on Linux; PyTorch 2.10.0+cpu; torchvision 0.25.0+cpu;
NumPy 2.3.5; OpenCV 4.13.0; FFmpeg and ffprobe available.
No CUDA device and no OpenCV-contrib DualTVL1 implementation were available.

## Executed checks

`python -m pytest -q --disable-warnings --maxfail=1`: **35 passed**.
All project Python modules compiled. The extraction/downloader/training-helper/
benchmark/comparison CLI help paths ran. `python check_install.py` ran using
actual installed dependencies and the Farneback front end.

The automated checks cover:

- Strict MMAction ConvModule-to-torchvision key mapping across stem,
  bottlenecks and downsampling. Missing learned tensors, nonfinite tensors,
  unexpected keys and RGB-vs-flow input shapes are rejected.
- Full ResNet-50 architecture with ten input channels: a random, explicitly
  test-only initialization performed a real `[1,10,256,256] -> [1,2048]` CPU
  forward pass. This was NOT a pretrained checkpoint test.
- Five native-pair ordering, edge repetition, support intervals, fixed-bound
  flow quantization, interleaved dx/dy channels and normalization.
- Actual Farneback on translated synthetic texture, with recovered horizontal
  displacement checked against the known translation.
- Actual FFmpeg synthetic MP4 decoding and timestamp alignment.
- End-to-end extraction orchestration on synthetic videos using a deliberately
  small fake TSN encoder: exact RGB preservation, cache/output shape, complete
  cohort publication, resume and interrupted-video recovery. Such test caches are
  marked test_only and refused by research training.
- New 2560-channel loader, 768-bin timestamp interpolation and visual labels.
- Native-model factory argument dimensions checked through an explicit mock,
  not an actual upstream UMMAFormer forward pass.
- Training, development evaluation, checkpoint saving/loading and resume paths
  using the adapted runner and an explicitly labelled small trainable model test
  double, not the native UMMAFormer architecture.
- Baseline-settings transfer, matched cohort checks, new-model initialization
  policy and explicit batch override reporting.
- Completed-evaluation comparison, per-video matching, metric consistency and
  rejection of mismatched source groups.

The previous project ZIP used for compatibility inspection was
`ummaformer_opencv_flow.zip`. The recovered top-level source-file hashes matched
the hashes in the user's posted internal-head evaluation report. The new project
uses a distinct checkpoint/feature format and does not overwrite that project.

## Not executed / not claimed

- Download or deserialize the actual ActivityNet pretrained TSN checkpoint.
  Its URL/name and input interface were verified from the official UMMAFormer
  extraction scripts. The generation environment could not download its bytes.
- Numerical equivalence of real pretrained outputs with MMAction2. An optional
  independent parity check is supplied, but was not executed here.
- Native upstream UMMAFormer forward/backward, compiled temporal NMS, torchsort,
  CUDA memory fit or training throughput for the 2560-channel input.
- Any real LAV-DF media, cache, checkpoint or detection result.
- WSL, Windows-mounted disk behavior or interruption of CUDA kernels.
- OpenCV TV-L1 extraction.
- Any improvement over the user's 0.8284 development ROC-AUC, or a specific
  factor of training-time increase/decrease.

The synthetic tests validate software paths, not deepfake accuracy or the
scientific value of the motion representation. The real pretrained forward,
eight-video extraction, native-model smoke test and matched hardware benchmarks
in README.md are the required local checks before the full experiment.
