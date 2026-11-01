# Validation report — Experiment 3

## Executed locally

Date: 5 October 2026. Environment: Python3.13.5, PyTorch2.10.0+cpu,
NumPy2.3.5, scikit-learn1.8.0. No CUDA device was available.

`python -m pytest -q --disable-warnings`: **43 passed in 7.64 seconds**.
These are software tests using explicitly marked synthetic canonical feature
caches, not LAV-DF experiments or trained-encoder accuracy measurements.

### Model tests

- Actual implemented cross, independent-self and concat heads: forward/backward
  execution and finite gradients with unequal visual/audio sequence lengths.
- Padding content and extra batch-padding invariance, including NaNs inserted
  only into excluded positions. All-masked modalities rejected.
- Shared continuous-time encoding depends on input seconds, not token numbering.
- Cross/self have identical parameter names, counts and initial tensor values
  under the same seed. Self visual hidden output is independent of audio before
  pooled classification; cross visual output can depend on audio.
- Changing target labels/record indices does not affect predictions.
- Concat output exactly matches the original E2B ConcatClassifier when given the
  same initialized weights and input means.

### Data/provenance tests

- Sequence means, normalization and E2B comparison fingerprints match the original
  E2B preparation code. No sequence arrays are copied into the prepared folder.
- All numeric feature entries/time associations remain available; original source
  cache files are unchanged by preparation.
- Missing video, corruption, conflicting labels, pooled-vector-as-sequence input,
  and marked synthetic fixtures are rejected in the production research path.
- Dev/test values do not influence training normalization.
- Adding a test index leaves train/dev identities and normalization unchanged.
- Explicitly marked fixtures require a private test API; no CLI train bypass exists.

### Training/evaluation/CLI tests

- Actual three-mode optimization, DEV selection, checkpoint writing and fixed-
  threshold TEST evaluation on synthetic sequences.
- Interrupted-at-epoch-boundary resume matches uninterrupted CPU state tensors
  exactly with the same settings. Changed settings and nonempty new runs rejected.
- Early-stopping patience exercised with controlled equal DEV scores.
- Real-feature smoke API writes no checkpoint/evaluation result.
- Training sampler order is independent of other model random draws.
- Matched per-video comparisons and paired source-family AUC/AP bootstrap.
- Comparison with an actual original E2B trainer/evaluator run on the same
  synthetic inputs; external training-setting limitations are labelled explicitly.
- Actual CLI parsing/training/evaluation/comparison through a test harness that
  admits MARKED fixtures, without making the production CLI accept them.
- Unsupported FP16 and research use of synthetic caches fail explicitly.

## Additional full-width check

The actual default head (width256, four heads, one block per direction, FFN x2,
classifier hidden128) completed CPU forward/backward with finite gradients for
both cross and self modes. Input shapes were `[2,768,256]` visual and `[2,19,512]`
audio, with unequal true lengths and padding. Both had **1,319,169 parameters**
for these example encoder dimensions. Actual parameter counts vary with the
selected encoders; the script reports them.

These checks used generated tensors, not real trained LAV-DF embeddings. They
are not hardware-throughput benchmarks, and CPU timing is not a GPU prediction.

All **nine** vendored `e2b/*.py` modules were verified byte-identical to the
user's supplied `experiment_2b.zip`. Python syntax compilation completed.

## Not executed / not claimed

- Real LAV-DF cross-attention/concat/self training or detection performance.
- User's selected encoder checkpoints, new native UMMAFormer/Forensics Adapter
  extraction, pretrained weight downloads, or new-media end-to-end inference.
- NVIDIA GPU execution, GPU memory fit, WSL, Kaggle, Colab, or PyTorch2.7.1 runtime.
- Guaranteed numerical equivalence across devices/torch versions.
- Improved AUC, lip synchronization detection, causal attention explanations,
  temporal/pixel localization, or published MulT benchmark reproduction.

The intended PyTorch API target includes the user's2.7.1 installation, but only
2.10.0 CPU was executed here. Start with the actual-feature GPU smoke tests before
full training. Keep an untouched copy of the run's source/config/caches/checkpoints.
