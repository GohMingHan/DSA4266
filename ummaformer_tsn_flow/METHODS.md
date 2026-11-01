# Experiment definition: frozen pretrained TSN flow features

## Exact question
Does a pretrained flow CNN representation improve the existing visual-only
UMMAFormer pipeline over pooled Farneback descriptors, on the same audited
LAV-DF videos? This is a single-model comparison, not a prediction ensemble.

## Held fixed
- Every RGB512 vector is copied bit-for-bit from the reference cache.
- Same metadata `file`, official split, source family, visual label, visual
  intervals, decoded duration, source indices, and anchor timestamps.
- Original MP4s, old numerical caches, old code, official upstream checkout and
  trained baseline checkpoints are never edited.
- The downstream model/loss/optimizer/scheduler/NMS come from the same official
  UMMAFormer checkout. Existing negative-video loss semantics are retained.
- `train_like_baseline.py` reads the baseline's complete training settings,
  including maximum horizon, warmup, LR, batch, EMA, patience and seed. It does
  not initialize the new temporal model with old trained weights.
- Main score remains maximum final proposal confidence (zero if none), not the
  weak internal classifier tested earlier. Selection is dev visual video AUC.

## What changes
Old: RGB512 + 16x16 area-pooled dx/dy velocity512 = 1024 channels.
New: RGB512 + pretrained TSN R50 global-average-pooled flow2048 = 2560 channels.

The TSN source checkpoint is the exact ActivityNet-clip flow checkpoint linked by
UMMAFormer (`...20200804-8622cf38.pth`), not an ImageNet RGB checkpoint. Its action
classification head is omitted; only its backbone representation is extracted.
The backbone uses five chronological (dx,dy) pairs, ten channels, ResNet-50 with
PyTorch-style strides, and global spatial averaging after layer4.

The code strictly maps MMAction ConvModule convolution and BN tensor names into
an equivalent torchvision ResNet50 backbone. It requires all learned tensors and
running statistics. Missing BN integer counters alone are initialized to zero;
these do not affect eval-mode forward output. Unknown/mismatched weights fail.
Numerical comparison with an independently installed MMAction2 is provided as an
optional check, not claimed to have been executed here with real pretrained data.

### Primary dense-flow method: Farneback
To avoid changing both the estimator and the encoder at once, the default keeps
Farneback and the previous 224x224 PIL analysis geometry. It replaces 16x16 pooling
with a pretrained flow CNN and a five-pair context. This is NOT an exact paper
feature reproduction: the pretrained flow CNN was not trained for this custom
LAV-DF/Farneback representation. Transfer may help or hurt.

Five native (not adjacent 10-FPS) pairs surround each cached RGB anchor. Pair k
means frame[k-1] -> frame[k]; clamp at the first/last real pair for edge contexts.
The actual context support and repetition fraction are saved separately. The
anchor timestamp remains the RGB timestamp. This is offline centered context,
not a causal real-time streaming model. No fake intervals choose input samples.

Dense displacement remains in pixels per frame pair. Do NOT feed the old
crop-fractions/second descriptor into TSN. Fixed bound +/-20 pixels maps to uint8
[0,255] with rounding; encoded fields are resized to 256x256 and normalized as
(x-128)/128. Channels are dx1,dy1,...,dx5,dy5. No HSV colors, per-video min-max
normalization, or ImageNet RGB normalization. Unlike disk-based original flow
extraction, there is no intermediate JPEG round trip.

The author's extractor uses 5-pair, 256-square, mean128/std128 inputs and local
2048-D outputs. We preserve that interface but retain our exact RGB anchors and
analysis geometry. This is an implementation/representation adaptation, not
pixel-identical replication of the authors' original feature archive.

### Optional TV-L1
`--flow-method tvl1` uses OpenCV CPU DualTVL1 with explicit parameters. It requires
an OpenCV contrib build and a NEW output cache. It is closer to conventional
TSN flow-data preparation, but OpenCV CPU behavior/geometry/compression need not
match the authors' denseflow setup. It changes the estimator as well as feature
encoding relative to the old baseline. There is no silent fallback.

## Frozen versus trainable
TSN has learned weights from external pretraining, but remains frozen in this
experiment. Both RGB and TSN networks run only during feature preparation (RGB is
reused here), and are absent from UMMAFormer training. The new temporal model's
input projections/reconstruction dimensions are configured to 2560, with no
zero-padding to 4096, no random dimensionality reduction, and no TSN logits used
as features. End-to-end TSN fine-tuning is NOT implemented.

## Interpretation and cost
Changing input width and reconstruction dimensions changes parameter count,
memory and compute, not merely the information presented to fixed parameters.
Cache numeric feature width is 2.5x the old combined width; compressed file size
and whole-model speed do not necessarily scale by 2.5. One-time extraction adds
TSN inference and five-pair context construction; epoch training reuses arrays.
Report extraction, training, dev evaluation, checkpoint I/O, and end-to-end
inference costs separately. Batch-size overrides are explicitly reported by the
helper; re-run a matched-batch baseline for a stricter controlled comparison.

Test labels are never used for training or parameter/readout selection. Keep the
same successful cohort; errors stop manifest publication rather than quietly
changing examples. RGB-copy checks do not independently prove that every original
RGB feature was correct; they prove consistency with the earlier audited cache.

## Primary source references (reviewed for this implementation)
- UMMAFormer extraction/checkpoint links:
  https://github.com/ymhzyj/UMMAFormer
  https://github.com/ymhzyj/UMMAFormer/blob/main/tools/gen_tsn_features_lavdf.sh
  https://github.com/ymhzyj/UMMAFormer/blob/main/tools/tsn_feature_extraction.py
- Official checkpoint:
  https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804-8622cf38.pth
- MMAction ResNet structure:
  https://mmaction2.readthedocs.io/en/stable/_modules/mmaction/models/backbones/resnet.html
- MMAction flow channel formatting:
  https://github.com/open-mmlab/mmaction2/blob/main/mmaction/datasets/transforms/formatting.py
- TSN paper:
  https://arxiv.org/abs/1608.00859
- Torchvision ResNet:
  https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html

No dataset, upstream UMMAFormer source, or pretrained weights are redistributed.
