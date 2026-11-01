# Experiment 3 — timestamped audio–visual cross-attention

Use **one selected visual encoder + one selected audio encoder**, frozen, and train
one new overall-video detector. This is the next stage of your LAV-DF study, not an
ensemble, not an autoencoder and not joint fine-tuning of the encoders.

The project reuses your **Experiment 2B model slots and unpooled sequence caches**.
The already-created `work/paired_train_dev/*.npz` vectors alone are NOT enough:
attention needs the sequences in `work/embeddings/visual` and `work/embeddings/audio`.
Do not use UMMAFormer's raw RGB/Farneback/TSN input cache as if it were an export
from your trained visual detector. The Experiment 2B bridge handles that distinction.

## What is included

```
experiment_3/
    pipeline.py
    config.json
    requirements.txt
    e3/                 new sequence preparation, cross/self models, training and comparison
    e2b/                exact unmodified compatibility copy from your supplied E2B package
    tests/              synthetic software tests, never real accuracy results
    README.md
    METHODS.md
    VALIDATION_REPORT.md
```

No model weights or dataset files are distributed. No original encoder project,
checkpoint, native UMMAFormer source, or Experiment 2B file is overwritten.

### Three alternatives

| `--mode` | Representation before pooling | Purpose |
|---|---|---|
| `concat` | Independent original mean pooling; original E2B MLP | Retrain an E2B-style baseline at this run's matched budget |
| `self` | Visual self-attention + audio self-attention, independently | Parameter-matched temporal/capacity control |
| `cross` | Visual queries audio and audio queries visual | Experiment 3 |

For cross/self, the projections, time encodings, attention blocks, feed-forward
sizes, pooling and classifier are the same. Only which stream provides keys and
values differs. Parameter counts and initial tensor values match for the same
seed. Compute/memory need not match: self-attention on a long visual sequence can
cost more than cross-attention to a shorter audio sequence. The concat head is
intentionally smaller; comparisons with it alone do not isolate cross-modal effects.

## 1. Install next to Experiment 2B

After saving the ZIP in Windows Downloads, in WSL:

```bash
conda activate ummaformer
cd "$HOME/dsa4266"
unzip "/mnt/c/Users/wuche/Downloads/experiment_3.zip"
cd "$HOME/dsa4266/experiment_3"
python -m pip install -r requirements.txt
python pipeline.py --help
```

Reuse the working PyTorch/CUDA installation (your reported 2.7.1+cu128 is the
intended API target). Do not reinstall torch, torchvision, torchsort, or the GPU
driver for this fusion head. Its runtime dependencies are torch, NumPy,
scikit-learn and tqdm. Encoder extraction still needs each encoder's original
working environment and input preprocessing.

On Kaggle/Colab, extract the project in a writable directory and edit all paths to
those on the VM. `/home/evan` and `/mnt/c` are not automatically available there.
Persistent storage is required for checkpoints; local runtime storage can disappear.

## 2. Use the same encoder choices as Experiment 2B

Your `experiment_3/config.json` is simply:

```json
{
  "format": "experiment_3_v1",
  "experiment_2b_config": "../experiment_2b/config.json"
}
```

Change that path only when your E2B project lives elsewhere. It resolves relative
to this config file, not to the current shell directory. All E2B input/cache/model
paths continue to resolve relative to their own original config/slot files.

Select and configure your two models **once**, in:

```
../experiment_2b/models/visual/best.pt
../experiment_2b/models/visual/encoder.json
../experiment_2b/models/audio/best.pt
../experiment_2b/models/audio/encoder.json
```

Do NOT select different encoders for E2B and E3. Supported extraction is exactly
what the prior E2B package supports: original audio ResNet18/AASIST, Forensics
Adapter/CLIP, the UMMAFormer variants, or an explicitly verified custom/preexported
adapter. An arbitrary `.pt` dropped into a folder is not a complete loader.

### Embeddings already exported

```bash
python pipeline.py check --config config.json
```

`ready_to_prepare: true` means canonical full train/dev indices exist and their
video IDs match the authoritative cohort. This is a preflight, not model accuracy
or validation of every numerical array. The next `prepare` command reads all arrays.
A `not_ready` report is expected while model selection/extraction is unfinished.

### Embeddings not yet exported

Use your existing E2B export commands, OR this equivalent wrapper:

```bash
# Optional separate interface pilot; never used as the research training cohort.
python pipeline.py extract --config config.json --modality both \
  --splits train dev --device cuda:0 --limit-per-split 4 --out work/encoder_pilot

# Full exports to the SAME folders specified in the E2B config.
python pipeline.py extract --config config.json --modality both \
  --splits train dev --device cuda:0
```

This calls an **unchanged copy of E2B's extraction implementation** in each selected
model's configured Python environment. It does not create a second architecture
choice. Checkpoint/source/preprocessing validation remains active. Do not run
concurrent exporters into the same cache. A changed encoder or numerical
extraction environment needs a new cache and new fusion runs.

## 3. Prepare sequence indices (not new image/audio preprocessing)

```bash
python pipeline.py prepare \
  --config config.json \
  --splits train dev \
  --out work/sequences_train_dev
```

When you have already prepared the E2B mean-pooled dataset, use this stronger check:

```bash
python pipeline.py prepare \
  --config config.json \
  --splits train dev \
  --e2b-data ../experiment_2b/work/paired_train_dev \
  --out work/sequences_train_dev
```

Run ONE of those commands into a new directory. The second verifies that pooling
the source sequences reproduces E2B's exact video vectors and label/cohort records.
It accepts the `mean` E2B baseline, not a `mean_max` alternative.

The stage validates exact video IDs, hashes, shapes, finite values, original split
membership, modality labels, linked source-family separation, and common time
origin. Missing/extra videos cause an explicit error, not an intersection or zero
fill. Source-family checks do not independently reconstruct unknown identities.
Disjoint modality time ranges fail. Extent differences above 0.5 s are reported
in `timing_warnings.jsonl`; inspect them instead of silently time-warping inputs.

**All sequence entries and their timestamp associations are retained.** There is
no cap, interpolation, label-based window selection or additional FPS sampling.
Original caches are referenced rather than copied. Do not remove or change them.

The preparation creates:

```
work/sequences_train_dev/
    dataset.json
    train.jsonl
    dev.jsonl
    normalization.npz
    timing_warnings.jsonl
```

Inspect `dataset.json`: expected cohort is train=4000/dev=500, with approximately
3000/375 any-fake positives under your four-category-balanced design. This is not
the 4% manipulated-frame distribution and not the 50% acoustic-positive target.

Normalization uses per-dimension mean/std across **natural TRAIN video means**,
separately for each modality, exactly as E2B mean pooling. These same affine
statistics are applied to every valid token for attention. Thus the normalization
policy does not add a train/dev or comparison confound. No stats are fitted on dev,
test, padded values or rebalanced training draws. This policy is not a claim of
unit variance across individual temporal tokens. It may be studied separately later.

## 4. Run actual-feature forward/backward checks

```bash
SeqData="$PWD/work/sequences_train_dev"

python pipeline.py smoke --data "$SeqData" --mode cross \
  --device cuda:0 --batch-size 8 --steps 2 --attention-backend math

python pipeline.py smoke --data "$SeqData" --mode self \
  --device cuda:0 --batch-size 8 --steps 2 --attention-backend math
```

Both tests use a temporary random **fusion head**, not random encoder embeddings.
They check finite gradients, real feature dimensions, padding behavior and peak
memory. No checkpoint, AUC, or development selection is produced. Cross/self
parameter-count equality is checked in the cross test.

If memory is insufficient for either head, use a smaller batch (e.g. 4) for both
checks and all the matched research runs. Batch 8 is a starting value, not a memory
guarantee; full sequence length affects attention cost. The self control may need
more attention-matrix memory than cross even with identical parameter count.

Default precision is **float32** and default attention is **math only**. This uses
CUDA when `--device cuda:0` is supplied; it does not force CPU. It avoids the
memory-efficient/cuDNN SDPA paths involved in the earlier LSE issue. The context
covers forward and backward. `--attention-backend flash-math` is an optional
recorded performance experiment. FP16 is intentionally not implemented in this
first package; do not use `run_safe_attention.py` from the CLIP project here.

## 5. Train the cross-attention model

```bash
python pipeline.py train \
  --data "$SeqData" \
  --mode cross \
  --out runs/cross_s42 \
  --device cuda:0 \
  --batch-size 8 \
  --workers 0 \
  --width 256 --heads 4 --layers 1 --head-width 128 \
  --lr 0.001 --weight-decay 0.0001 --dropout 0.2 \
  --epochs 50 --patience 8 --seed 42 \
  --attention-backend math --amp none
```

These are proposed baseline hyperparameters, not known optima. Width256/4 heads
follow the earlier E3 design. The 50-epoch ceiling/patience8 and AdamW settings
match the more recent E2B baseline rather than the older generic 30/5 proposal.
The two encoders are absent from this optimizer and never updated.

At each layer:

```
V0 = linear(standardized_visual) + time_encoding(visual_seconds)
A0 = linear(standardized_audio)  + time_encoding(audio_window_center_seconds)
V1 = residual_attention_and_feedforward(query=V0, keys/values=A0)
A1 = residual_attention_and_feedforward(query=A0, keys/values=V0)
# Both directions read the OLD states, not one already-updated branch.
y_logit = MLP(concat(masked_mean(V1), masked_mean(A1)))
```

Each attention direction has residual connections, pre-normalization,
feed-forward layers and dropout. Padding keys are excluded and padded queries are
zeroed after transformations; final pooling excludes padding. Empty modalities
are rejected instead of creating all-masked attention and NaNs.

Time encoding uses fixed sine/cosine functions of shared video-relative seconds
(periods 0.5 to 512 s), NOT a separate reset/index scale for each modality.
Visual frame counts and audio-window counts can differ. Audio window centers are
anchors for approximately four-second contextual embeddings; the model is not
claimed to verify precise lip synchronization. The attention is global/offline,
not causal streaming or local time-band attention. No timestamps are derived from
`fake_periods`, and no modality flags, paths, source IDs or probabilities enter the
network as feature columns.

Loss: unweighted BCEWithLogitsLoss, gradient norm limit5, finite-loss/gradient
checks. Default class-aware video sampling draws N_train examples per epoch with
replacement, balancing binary labels **in expectation**, with no additional
inverse-frequency loss weight. In your planned cohort this oversamples genuine
videos, not rare manipulated frames. Each source recording is not weighted equally.
A sampler generator independent from model initialization matches video draws
across head choices for the same seed/epoch. `--balance none` is a separate
ordinary-shuffle alternative.

With 4000 training videos and batch8, each epoch contains500 batches. Sequence
length and attention operations make these batches more expensive than E2B's
pooled MLP batches. This still avoids rerunning both encoders every epoch. It is
not guaranteed to take a particular number of seconds on your GPU.

Development is evaluated after every epoch. DEV **any-manipulation video AUC**
selects `best.pt` and early stopping; continuous logits are used for AUC/AP.
The selected checkpoint's DEV maximum-F1 threshold (highest cutoff on exact ties)
is saved and reused unchanged for test. Sigmoid output is not claimed calibrated.

Outputs:

```
runs/cross_s42/
    best.pt
    last.pt
    history.json
    run_config.json
    normalization_train.json
    best_dev_metrics.json
    best_dev_predictions.jsonl
```

`best.pt` is only the fusion head plus normalization/provenance. It does not embed
the original visual/audio checkpoints. New-video inference still needs both
selected encoders, their preprocessing and this head. Cached sequence evaluation
is included; an arbitrary-raw-MP4 service/UI is not supplied.

## 6. Train controls using the same settings

Use the same full command as Step5, changing ONLY mode and output directory.
Because the defaults above are the same, these shorter commands are equivalent:

```bash
python pipeline.py train --data "$SeqData" --mode self \
  --out runs/self_s42 --device cuda:0 --batch-size 8 \
  --epochs 50 --patience 8 --seed 42

python pipeline.py train --data "$SeqData" --mode concat \
  --out runs/concat_s42 --device cuda:0 --batch-size 8 \
  --epochs 50 --patience 8 --seed 42
```

Self and cross are parameter-matched; concat instantiates the original E2B
`ConcatClassifier` at head width128 and uses the EXACT original mean vectors.
This repeats the inexpensive concat head only, not encoder extraction/training.
It is useful when earlier E2B batch size or budget differed. Same seed does not
make the differently sized concat and attention models parameter-identical.

Keep existing E2B visual-only and audio-only **any-manipulation** controls in the
report. Their native Experiment1/2A metrics are not substitutes for those controls.

## 7. Evaluate and compare on DEV

```bash
python pipeline.py evaluate --data "$SeqData" --checkpoint runs/cross_s42/best.pt \
  --split dev --out evaluations/cross_dev --device cuda:0 --batch-size 8

python pipeline.py evaluate --data "$SeqData" --checkpoint runs/self_s42/best.pt \
  --split dev --out evaluations/self_dev --device cuda:0 --batch-size 8

python pipeline.py evaluate --data "$SeqData" --checkpoint runs/concat_s42/best.pt \
  --split dev --out evaluations/concat_dev --device cuda:0 --batch-size 8

python pipeline.py compare --runs evaluations/self_dev evaluations/cross_dev \
  --out evaluations/self_vs_cross --bootstrap 1000

python pipeline.py compare --runs evaluations/concat_dev evaluations/cross_dev \
  --out evaluations/concat_vs_cross --bootstrap 1000
```

Every evaluation publishes `metrics.json` and `video_predictions.jsonl`. Metrics
include overall-video AUC, AP, F1, precision, recall, accuracy, balanced accuracy,
confusion matrix, genuine-video false-positive rate and recall by fake category.
DEV re-evaluation must reproduce its checkpoint AUC/AP within1e-6; a mismatch
stops publication for investigation. No prediction subset is silently retained.

Comparison verifies exact paired video/source-group labels, encoder identities,
train/dev fingerprints, sequence identity and normalization (for E3 results).
It checks per-video scores against report metrics before comparing. It never
ensembles the predictions. With two runs, bootstrap intervals are paired
source-family intervals for AUC/AP differences **second minus first**. They are
conditional on the selected checkpoints, not corrected for DEV model selection
or training-seed uncertainty. Repeat final matched settings with seeds42/43/44.

To compare an EXISTING E2B mean-pooling evaluation:

```bash
python pipeline.py compare \
  --runs ../experiment_2b/evaluations/concat_dev evaluations/cross_dev \
  --out evaluations/existing_e2b_vs_cross --bootstrap 1000 \
  --allow-training-differences
```

External E2B reports do not include every training setting, so that flag explicitly
labels an unverified/different training-protocol comparison. It does NOT relax
cohort, target, encoder, mean-vector or per-video checks. For a better-matched
comparison use the retrained concat control above. Do not change an external
report's metadata to make it pass.

## 8. Resume training

Repeat the SAME original training settings and original output, adding:

```bash
--resume runs/cross_s42/last.pt
```

For example (defaults identical to Step5):

```bash
python pipeline.py train --data "$SeqData" --mode cross --out runs/cross_s42 \
  --device cuda:0 --batch-size 8 --epochs 50 --patience 8 --seed 42 \
  --resume runs/cross_s42/last.pt
```

The last completed epoch, optimizer, random state, best metric and patience are
restored. Ctrl+C does not produce a half-epoch checkpoint. Nonempty runs are never
overwritten. Source/settings/data mismatches stop resume. A previous early stop
is not silently reset. There is no automatic guarantee of bit-identical results
across different torch versions, GPU models or algorithms.

## 9. Test only after locking the protocol

The audio test results have already been viewed in the project. Report that
honestly; do not claim retrospectively blinded encoder selection. Use DEV for
additional encoder/fusion/readout choices and do not tune against further TEST
results.

When the canonical exports and authoritative cohort include the intended test IDs:

```bash
# Only needed if test embeddings are not yet exported; original input manifests
# must contain the fixed test IDs with train/dev unchanged.
python pipeline.py extract --config config.json --modality both --splits test --device cuda:0

python pipeline.py prepare --config config.json --splits train dev test \
  --out work/sequences_with_test

python pipeline.py evaluate --data work/sequences_with_test \
  --checkpoint runs/cross_s42/best.pt --split test \
  --out evaluations/cross_test --device cuda:0 --batch-size 8
```

Prepare into a NEW folder. Training and development sequence fingerprints,
encoder identities and normalization must remain unchanged. No new threshold or
normalization is fitted from test. Do not run training on this data with different
cohorts to "repair" a mismatch. The trainer explicitly uses only train and dev.

## Scope and limits

This is a specified custom frozen-feature experiment, not reproduction of a
published deepfake-fusion score. There is no autoencoder/reconstruction loss,
contrastive synchronization loss, local time-band mask, end-to-end encoder tuning,
missing-modality imputation, or predicted-pixel/temporal localization loss here.
The output is an overall-video authenticity score. Attention weights would not,
by themselves, establish causal explanations or trustworthy localized evidence.
Improvement over E2B is an open empirical outcome of this experiment.

See METHODS.md for definitions/references and VALIDATION_REPORT.md for exactly
what was tested. Do not equate synthetic software tests with LAV-DF accuracy.
