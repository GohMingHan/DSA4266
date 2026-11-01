# Experiment 3 method specification

## Question and scope

Does interaction between timestamped frozen visual and audio representations,
before pooling, improve overall-video manipulation detection over independent
pooling and over equally sized independent temporal attention?

This carries forward the project's Experiment3 design: two directional global
cross-attention operations, shared-time encodings, padding masks, pooling and an
MLP; no autoencoder. It is a custom implementation, not a reproduction of the MulT
benchmark or a published LAV-DF fusion result.

## Inputs and targets

Use one chosen visual checkpoint and one chosen audio checkpoint, selected without
further tuning on test. Exact common file IDs and original split assignments are
retained. E2B canonical numeric sequence arrays supply V[Nv,Dv] with timestamps and
A[Na,Da] with source window start/end/center. Every observed token is retained;
no label-driven crop, probability selection, sequence-length cap or new temporal
interpolation is applied. Variable lengths are padded only inside a batch.

Audio and UMMAFormer representations summarize context, not instantaneous sensory
measurements. Token centers need not denote millisecond-aligned lip/audio events.
All times share the first decoded video-frame origin; no separate per-modality
normalization of time is used. Feature arrays must be learned embeddings, not
scalar probabilities or classification-head logits. Raw TSN/RGB inputs are not a
substitute for the selected trained temporal model's exported features.

The target is y=modify_video OR modify_audio. The initial four-category-balanced
cohort is therefore 75% positive at video level. Dev/test remain naturally
distributed within that fixed cohort. Class-aware video resampling balances binary
classes in expectation for training only. It is not source/identity balancing.
The model sees neither labels nor paths, categories or fake-period annotations as
feature channels.

## Normalization and controls

For every TRAIN video, compute the per-modality mean embedding, then estimate
per-channel means and standard deviations across those video means. This is the
EXACT original E2B mean-pooling normalization. Use these same frozen affine
statistics on individual tokens for cross/self attention, and on means for concat.
It does not enforce unit variance of temporal tokens. Near-constant pooled
channels use unit scale, as in E2B. Dev/test do not fit statistics.

The concat control instantiates the original E2B ConcatClassifier: independent
Linear(D,128)/ReLU/dropout branches and Linear256->128/ReLU/dropout/Linear128->1.
It uses the original full-sequence pooled arrays, not time-encoded token means.
Self and cross use identical parameter shapes and initial tensor values for a
matched seed, with attention width256, four heads, one directional block per
modality, FFN multiplier2 and final MLP512->128->1. Only context selection differs.
There is no cross-stream attention in the self control, though the final MLP can
combine its two independently pooled outputs. Parameter matching is NOT a promise
of matching FLOPs or memory, because Nv and Na differ.

## Attention

Each stream projects standardized tokens to width256. A shared fixed sinusoidal
encoding of real seconds, with log-spaced periods0.5..512s, is added. Each
bidirectional layer reads both OLD input streams:

V_next = Block_V(query=V, context=A)
A_next = Block_A(query=A, context=V)

Block: LayerNorms on query/context, multi-head scaled dot-product attention,
residual+dropout, then pre-LayerNorm two-linear-layer GELU FFN with residual.
Key padding uses True=excluded for nn.MultiheadAttention. Padded query positions
are zeroed after projection/residual updates and excluded from pooling. The
implementation rejects empty modalities. The attention mask is global/noncausal;
there is no tight time-band or learned forced alignment.

After final stream LayerNorm, masked means are concatenated and classified.
Output is one logit per video; sigmoid is a monotone score transformation, not a
probability calibration guarantee.

## Optimization/evaluation

Default: AdamW1e-3, weight_decay1e-4, dropout0.2, batch8, unweighted BCEWithLogitsLoss,
FP32, gradient norm limit5, max50 epochs/patience8. These follow the recent E2B
head budget and are proposed starting settings, not tuned optima. There is no
scheduler in this initial head comparison. Sampler randomness is separate from
model initialization; matching seed and epoch gives matching sampled video IDs.

Use the same settings across cross/self/concat except architecture mode. A smaller
matched batch may be required on a particular GPU. DataLoaderworkers0 is the
conservative initial setting. Attention backends default to math-only; optional
flash+math still excludes cuDNN and memory-efficient attention. The context is
active during backward too. AMP/FP16 is deliberately excluded here.

Best checkpoint and patience use DEV overall-video ROC-AUC. DEV maximum-F1 cutoff
is fixed with a highest-cutoff exact-tie convention and reused on test. AUC/AP use
continuous logits. Evaluate every requested video and retain per-video predictions.
Re-evaluated DEV AUC/AP must match the saved checkpoint within1e-6. Paired
source-family bootstrap intervals compare two matched alternatives; they do not
remove selection bias or substitute for seed replication. No TEST selection is
performed. Previously inspected audio test scores remain a disclosed limitation.

## Engineering and reproducibility

E2B compatibility modules are copied unmodified. Encoder extraction uses the same
configured Python processes, schemas, provenance and read-only checkpoint loaders.
E3 requires full canonical exports, not pilot subsets. Preparation verifies hashes
and coverage and writes references, not duplicate large feature archives. Source
hashes and train/dev sequence/normalization fingerprints are saved in each head
checkpoint. Exact means additionally link E3 outputs to E2B mean-pooling fingerprints.
Resume restores complete-epoch state; the encoders do not enter the optimizer.

Files/caches must remain immutable while used by a run. Hashes identify inputs;
they do not prove valid semantic synchronization, independent identities, external
pretraining provenance, or absence of dataset shortcuts. New-media inference still
requires both encoders and their original preprocessing plus the chosen fusion head.

## Primary references

1. Tsai et al. (2019), Multimodal Transformer for Unaligned Multimodal Language
   Sequences (MulT). Directional crossmodal attention is relevant conceptual
   background; this implementation is not the full MulT architecture/protocol.
   https://arxiv.org/abs/1906.00295
2. PyTorch2.7 MultiheadAttention API, Q/K/V and padding semantics:
   https://docs.pytorch.org/docs/2.7/generated/torch.nn.MultiheadAttention.html
3. PyTorch2.7 attention backend context:
   https://docs.pytorch.org/docs/2.7/generated/torch.nn.attention.sdpa_kernel.html
4. Scikit-learn, common pitfalls and train-only transformations:
   https://scikit-learn.org/stable/common_pitfalls.html
5. The user's supplied Experiment2B package (experiment_2b_v1), README/METHODS,
   canonical sequence export, normalizer, classifier and metric implementations.
6. The project's earlier EXPERIMENT_PLAN.md: frozen sequences, directional global
   cross-attention, parameter-matched self-attention control, and explicit masks.

These are method/software sources, not evidence of achieved deepfake accuracy.
