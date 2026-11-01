# Method definition and attribution

## Experimental question

Do independently pooled embeddings from one frozen visual detector and one
frozen acoustic detector improve whole-video any-manipulation classification
compared with matched single-modality heads?

This is feature-level fusion, not independent prediction averaging, not
cross-attention, and not an autoencoder. Only the small head and its independent
linear projections are trained. Both original model heads may still execute
within their original inference functions, but their prediction outputs are
ignored; extraction uses explicit intermediate learned representations.

## Primary implementation

For embeddings V in R^(Nv x Dv), A in R^(Na x Da):

v = mean_t(V), a = mean_t(A)
v' = (v - mean_train_visual) / std_train_visual
a' = (a - mean_train_audio) / std_train_audio
p = sigmoid(MLP(concat(projection_visual(v'), projection_audio(a'))))

y = modify_video OR modify_audio

Mean/max concatenation is an optional separately selected pooling ablation.
Coordinates are retained but are not numeric classifier input columns. The fusion
stage never receives fake-period boundaries, modality labels, source IDs, video
paths, category strings, or per-video authenticity predictions as features.

The head is deliberately small (128 units per branch, 128-unit hidden classifier,
dropout .2, AdamW 1e-3/weight_decay1e-4). These are proposed baseline settings, not
established optimal hyperparameters. Fifty epochs is a ceiling with patience8 on
DEV any-fake AUC. AUC uses raw logits; a DEV-F1 threshold is saved and reused on
TEST. Scores are not claimed calibrated probabilities.

Class-aware sampling balances overall binary video labels in expectation during
training. In the proposed equal-four-category cohort, any-fake positives are 75%,
not the ~4% positive rate of individual visual frames. There is no double
correction via additional inverse-frequency loss weights. DEV/TEST preserve the
fixed cohort. Matched visual/audio heads also target any manipulation.

## Encoder adapters and scope

Audio and Forensics Adapter bridges reuse the original projects' strict loading,
training normalization, input construction and feature-readout definitions. They
are not a universal loader of arbitrary notebook weights. See README for the
explicit preexported/custom bridge for unknown formats.

UMMAFormer exports the highest-resolution FPN sequence fed into its temporal
classification head. Native FPN masks remove padded entries. Native point-
generator coordinates are converted to seconds via the loader's exact
feat_stride/feat_num_frames/fps convention. The same interpolation/feature input
used in the trained checkpoint is retained. No labels are passed to inference.
Readout layer choice is a new specified fusion adaptation; it is not justified
by the earlier internal-classifier AUC, and it is not the scalar internal score.
Temporal context in UMMAFormer and the ~4.04-second audio windows means timestamps
are anchors/support intervals, not instantaneous lip-synchronization measurements.

## Public primary references checked while building the add-on

- UMMAFormer native forward sequence (backbone, neck, point generation, temporal
  classification head):
  https://github.com/ymhzyj/UMMAFormer/blob/main/libs/modeling/av_recoverynonorm_meta_arch.py
- PyTorch evaluation mode versus no-grad/inference mode:
  https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html
- PyTorch BCEWithLogitsLoss:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
- PyTorch restricted checkpoint loading:
  https://docs.pytorch.org/docs/2.7/notes/serialization.html
- Train-only preprocessing and avoiding selection leakage:
  https://scikit-learn.org/stable/common_pitfalls.html

These references explain components, not validate the custom pipeline's deepfake
accuracy. The earlier supplied Experiment 1/2A and UMMAFormer packages are the
source of the specific local checkpoint/feature interfaces. Preserve their
model/dataset licenses and source versions.

## Limits

Near-perfect acoustic test scores do not guarantee any-fake classification or
multimodal improvement. Average pooling can lose short-event evidence. Cached
encoder features may overfit the supervised encoder training cohort; the fusion
study is conditional on those frozen checkpoints and their selection process.
Source-family checks do not reconstruct unknown subject identities. Selected
cohort split assignments must already be valid. Source/hash checks cannot prove
that an unknown pretrained checkpoint never saw a held-out identity/dataset.
Development selection over many trials biases development estimates; bootstrap
intervals do not remove this bias. Previously viewed test results must be reported
honestly. No new test-set tuning should be performed.

All extraction functions are read-only with respect to original source projects,
checkpoints, and input caches. They write new feature caches only. End-to-end
inference still needs the original preprocessing plus both encoders and this
head. Only cached fusion-head training/evaluation avoid the encoder cost.
