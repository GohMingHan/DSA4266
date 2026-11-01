# Audio-Visual Deepfake Detection with CLIP, AASIST, and Multimodal Fusion

Research code and evaluation artifacts for video-level deepfake detection using frozen **CLIP ViT-B/32** visual representations and **AASIST** audio representations. We compare a mean-pooled **concatenation + MLP** fusion model with a sequence-based **bidirectional cross-attention** model, and investigate their behavior using explainability (XAI) experiments.

> **Reproducibility status:** Existing TRAIN/DEV/TEST dataset identities, model checkpoint hashes, and final TEST metrics have been checked. A clean-clone, end-to-end regeneration of the full pipeline has **not yet been verified**. Large model weights, raw videos, and extracted embeddings are not committed to Git.

## Overview

| Component | Description |
| --- | --- |
| Dataset | Local 5,000-video LAV-DF subset (`LAV-DF-5000-NEW`) |
| Split | 4,000 TRAIN / 500 DEV / 500 TEST videos |
| Categories | Real, audio-only manipulation, visual-only manipulation, audiovisual manipulation |
| Visual features | CLIP ViT-B/32 V2, frozen; 512-D frame embeddings at 10 FPS |
| Audio features | AASIST, frozen; 160-D audio-window embeddings |
| Classification target | `modify_audio OR modify_video` for fusion models |
| Fusion models | Mean-pooling + concatenation MLP; bidirectional cross-attention |
| Training seed | 42 |

The final fusion models share the same frozen encoder outputs. Concat mean-pools each modality before classification; Cross-Attention retains the temporal sequences for cross-modal interactions, then aggregates features for video-level prediction. The visual-only CLIP evaluation uses the **`modify_video`** target; its overall metrics should not be interpreted as a directly controlled any-manipulation baseline.

## Repository layout

```text
.
├── experiment_2a/               # AASIST pipeline and feature export
├── experiment_2b/               # Visual-only CLIP evaluation and E2B configuration
├── experiment_3/
│   ├── e2b/                     # Canonical embedding / paired-data utilities
│   ├── e3/                      # Fusion architectures, preparation, training, evaluation
│   ├── adapters/                # Canonical V2 TRAIN/DEV/TEST manifest builders
│   ├── evaluations/             # Small saved metrics, predictions, comparison scripts
│   ├── xai/                     # Ablation, counterfactual, interaction and timing analysis
│   └── pipeline.py
├── lavdf_new5000/               # CLIP V2 export scripts and dataset utilities
└── reproducibility/
    ├── environment.yml
    ├── checkpoint_sha256.txt
    └── dataset_identity_v2.json
```

Some generated directories (for example, `experiment_3/work/`, model `runs/`, raw video data and embedding exports) are intentionally ignored by Git. Exact external checkpoint download URLs are **not yet supplied**; see [Checkpoints](#checkpoints).

## Environment

The experiments were run in the `ummaformer` Conda environment. Start with the exported environment:

```bash
conda env create -f reproducibility/environment.yml
conda activate ummaformer
```

The exported environment is a record of the working setup, not a guarantee of cross-platform solvability. Verify CUDA, PyTorch, `transformers`, `av`, and the upstream AASIST implementation on your machine. The repository also contains `experiment_2a/upstream_aasist`; retain its exact compatible source version when reproducing embeddings.

Set dataset and checkpoint paths for your local machine:

```bash
export LAVDF_METADATA="/path/to/LAV-DF-5000-NEW/metadata.new5000.json"
export CLIP_CHECKPOINT="/path/to/clip_v2_best.pt"
export AASIST_CHECKPOINT="/path/to/aasist_best.pt"
```

The metadata path must refer to the same dataset and split membership as the original experiment. The dataset videos are expected to be accessible relative to the metadata location as used by the export scripts. The audio export also expects the preprocessing outputs under `experiment_2a_new5000/` and the original audio configuration in the repository root `config.json`; see the corresponding scripts for the exact layout.

## Checkpoints

**Do not substitute different encoder weights** if your goal is to reproduce the reported results. Verify downloaded files with `sha256sum`.

| Model | Original location | SHA256 |
| --- | --- | --- |
| CLIP V2 | `best.pt` (repository root) | `9f846830fa093ceaed0ad272c5cb5ab9255ec301670eb542fbd2fb2825acada3` |
| AASIST | `experiment_2b/models/audio/best.pt` | `29b87952d352c4d69c0316d34effdba4babc0e8c156c18fd1993b39061153efc` |
| Concat V2 | `experiment_3/runs/concat_clip_v2_new5000_s42/best.pt` | `cd4d987400c2119373c42cb19f136a6dccc7d6419cdc253816816016836d6a8f` |
| Cross V2 | `experiment_3/runs/cross_clip_v2_new5000_s42/best.pt` | `da1c9baeda66fef33e72206a79ac8433895e7b21910b1464b1d8f97474e9b08d` |

Hashes are also stored in `reproducibility/checkpoint_sha256.txt`.

**Checkpoint availability:** [TODO: add authorized, persistent download links or an access-request procedure]. The checkpoint files are not part of the normal Git repository.

## Data and feature extraction

The dataset contains 1,000 source families with four variants each across the full 5,000-video cohort. Each DEV and TEST split has 125 complete source families, with one real, audio-only, visual-only, and audiovisual variant per family. Preserve source-family grouping and split membership.

### 1. Extract CLIP V2 features

Run from the repository root after setting `LAVDF_METADATA` and `CLIP_CHECKPOINT`:

```bash
python lavdf_new5000/clip_export_v2_train.py
python lavdf_new5000/clip_export_v2_dev.py
python lavdf_new5000/clip_export_v2_test.py
```

These scripts use the trained CLIP V2 encoder and sample frames at 10 FPS. Expected export locations:

```text
experiment_2b_outputs/new5000_v2_train/
experiment_2b_outputs/new5000_v2_dev/
experiment_2b_outputs/new5000_v2_test/
```

The canonical visual export folder includes the identifier `vit_b32_shared_fixed20k_10fps_9f846830fa09_0576c2b0`. The metadata and checkpoint hashes determine the export naming. Confirm the generated `export_config.json` and `index.jsonl` before continuing.

### 2. Extract AASIST features

The AASIST export scripts require the separately prepared `experiment_2a_new5000` audio dataset and matching preprocessing configurations. Run from the repository root:

```bash
python experiment_2a/transfer/export_aasist_new5000.py
python experiment_2a/transfer/export_aasist_new5000_test.py
```

Expected export locations:

```text
experiment_2a_new5000_aasist_embeddings/
experiment_2a_new5000_aasist_test_embeddings/
```

The scripts verify the AASIST checkpoint hash and audio preprocessing configuration. The preprocessing stage that creates `experiment_2a_new5000/` must also be reproduced; its full clean-clone command sequence is **not yet verified**.

### 3. Build canonical V2 caches and TRAIN/DEV dataset

Run from `experiment_3/`:

```bash
cd experiment_3
python -m adapters.build_new5000_v2_manifests
python pipeline.py prepare \
  --config config_clip_v2_new5000.json \
  --out work/sequences_clip_v2_new5000_train_dev_fixed \
  --splits train dev
```

These commands target **new output directories**; preparation scripts intentionally refuse to overwrite existing outputs. `experiment_3/config_clip_v2_new5000.json` points to `../experiment_2b/config_clip_v2_new5000.json`, whose cohort path uses `${LAVDF_METADATA}`. The V2 canonical embedding roots must match those declared in the E2B config.

**Important:** An earlier directory named `sequences_clip_v2_new5000_train_dev` was generated from *different* encoder identities and normalization. The reported final results use **`sequences_clip_v2_new5000_train_dev_fixed`** only. A fresh-run byte-for-byte match of the `_fixed` directory has not yet been established.

### 4. Add TEST without recomputing TRAIN normalization

```bash
python -m adapters.add_new5000_v2_test
python -m adapters.build_new5000_v2_test_dataset
```

The final dataset is `work/sequences_clip_v2_new5000_with_test/`. These scripts preserve the TRAIN/DEV identity and normalization while appending TEST. Confirm that all 500 TEST videos are included.

### Dataset identity checks

Compare the regenerated `dataset.json` with `reproducibility/dataset_identity_v2.json`. Expected values:

| Split | Videos | Split signature |
| --- | ---: | --- |
| TRAIN | 4,000 | `954dabd0a448fa64a11709a5e14174157cd8004018218bf4d4e51f3df28e5fae` |
| DEV | 500 | `378d990cc52c0219b467db46ca1afd6d7ec0a7001ec4978341a894a6668e82b7` |
| TEST | 500 | `6eb0dec5e5456a06cbba77194f1319c1ea40f5b3a803911a5d648b7659bb1e90` |

Expected encoder IDs:

```text
visual: dc0e7b441834066fa0212acc6ea27a96f73a214b11fb3d6b12028ed9d81c2309
audio:  314453656435b250a368d1adad0b7a7fb30dce5ef72576abe6aa599c1d406b90
```

Expected TRAIN normalization NPZ SHA256: `5684a8581961b84333ba306553776b959618a2b1e6a1f79b4f0c767fd45b015d`.

The generated dataset metadata contains absolute embedding paths; these must point to the regenerated local caches. The dataset identity record intentionally omits machine-specific roots.

## Train fusion models

Run from `experiment_3/`. Both models were trained with seed 42, batch size 8, maximum 50 epochs, patience 8, learning rate 0.001, weight decay 0.0001, dropout 0.2, width 256, 4 attention heads, one layer, feed-forward multiplier 2, head width 128, class balancing, time encoding enabled, and math attention backend. Concat does not use temporal attention, even though shared CLI settings are recorded.

**Concat V2:**

```bash
python pipeline.py train \
  --data work/sequences_clip_v2_new5000_train_dev_fixed \
  --out runs/concat_clip_v2_new5000_s42 \
  --mode concat --width 256 --heads 4 --layers 1 \
  --ff-multiplier 2 --head-width 128 --dropout 0.2 \
  --time-encoding --attention-backend math \
  --batch-size 8 --epochs 50 --patience 8 \
  --lr 0.001 --weight-decay 0.0001 \
  --balance class --seed 42 --device cuda:0
```

**Cross-Attention V2:**

```bash
python pipeline.py train \
  --data work/sequences_clip_v2_new5000_train_dev_fixed \
  --out runs/cross_clip_v2_new5000_s42 \
  --mode cross --width 256 --heads 4 --layers 1 \
  --ff-multiplier 2 --head-width 128 --dropout 0.2 \
  --time-encoding --attention-backend math \
  --batch-size 8 --epochs 50 --patience 8 \
  --lr 0.001 --weight-decay 0.0001 \
  --balance class --seed 42 --device cuda:0
```

Use new output directories when retraining. Exact checkpoint hashes may differ across hardware or software versions even when evaluation behavior is similar.

## Evaluate on TEST

The saved checkpoints use thresholds selected on DEV; **do not tune thresholds on TEST**.

```bash
python pipeline.py evaluate \
  --data work/sequences_clip_v2_new5000_with_test \
  --checkpoint runs/concat_clip_v2_new5000_s42/best.pt \
  --split test --out evaluations/concat_clip_v2_new5000_test \
  --device cuda:0 --batch-size 8

python pipeline.py evaluate \
  --data work/sequences_clip_v2_new5000_with_test \
  --checkpoint runs/cross_clip_v2_new5000_s42/best.pt \
  --split test --out evaluations/cross_clip_v2_new5000_test \
  --device cuda:0 --batch-size 8
```

Reference TEST results (500 videos, 375 manipulated and 125 genuine):

| Metric | Concat V2 | Cross V2 |
| --- | ---: | ---: |
| Video ROC-AUC | 0.994197 | 0.997867 |
| Average precision | 0.997958 | 0.999293 |
| Accuracy | 0.9420 | 0.9760 |
| Precision | 0.9915 | 0.9892 |
| Recall | 0.9307 | 0.9787 |
| F1 | 0.9601 | 0.9839 |
| False-positive rate | 0.0240 | 0.0320 |

### Category-specific comparison against visual-only CLIP

The visual-only CLIP V2 evaluation must first be available at `experiment_2b/evaluations/visual_modify_video_v2_new5000_test/`. It predicts **visual manipulation**, while the fusion models predict **any manipulation**. The category comparisons below are restricted to categories where those targets agree.

From `experiment_3/`:

```bash
python evaluations/category_comparison_v2_new5000_test/compare.py
```

This script reads the three saved prediction files and refuses to overwrite an existing `comparison.json`. Run in a clean output location or preserve the already committed results. Reference TEST category comparisons:

| Comparison | Model | ROC-AUC | F1 | Accuracy | Recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Real vs visual-only | CLIP V2 | 0.9519 | 0.7793 | 0.8120 | 0.6640 | 0.0400 |
| Real vs visual-only | Concat V2 | 0.9851 | 0.8918 | 0.9000 | 0.8240 | 0.0240 |
| Real vs visual-only | Cross V2 | 0.9939 | 0.9512 | 0.9520 | 0.9360 | 0.0320 |
| Real vs audiovisual | CLIP V2 | 0.9831 | 0.8983 | 0.9040 | 0.8480 | 0.0400 |
| Real vs audiovisual | Concat V2 | 0.9994 | 0.9841 | 0.9840 | 0.9920 | 0.0240 |
| Real vs audiovisual | Cross V2 | 1.0000 | 0.9843 | 0.9840 | 1.0000 | 0.0320 |

## Explainability (XAI)

The `experiment_3/xai/` directory contains the analysis code and small saved results:

| Analysis | Main script | Saved results |
| --- | --- | --- |
| Modality ablation | `modality_ablation_v2.py`, `analyze_modality_ablation.py` | `xai/results/{concat,cross}_v2_dev/` |
| Same-family audio/visual counterfactual pairing | `counterfactual_pairing_v2.py` | `xai/results/{concat,cross}_v2_test_counterfactual/` |
| Factorial interaction effects | `analyze_crossmodal_interactions_v2.py` | `xai/results/crossmodal_interactions_v2_test/` |
| Timestamp-shift sensitivity | `temporal_shift_sensitivity_v2.py` | `xai/results/{concat,cross}_v2_test_temporal_shift/` |

Counterfactual pairing evaluated 16 visual/audio combinations within each of 125 TEST source families (2,000 combinations per model). Interaction analysis used paired family-level contrasts and bootstrap uncertainty. Temporal-shift tests modified timestamp inputs, **not the underlying audio waveform or embedding order**.

Selected findings: both models respond strongly to manipulated-audio substitutions; Cross shows larger raw-logit non-additivity than Concat across the evaluated contrasts; Concat is invariant to timestamp-only shifts, while Cross shows modest positional-encoding sensitivity. These observations **do not by themselves establish causal audiovisual synchronization**, and the models' raw-logit scales differ.

The XAI scripts may require existing evaluation prediction files and specific checkpoint paths. Inspect their `--help` output and source defaults before execution. Exact XAI command sequences should be validated in a clean environment before publication as one-click reproduction instructions.

## Reproducibility and limitations

- **Data access:** Raw LAV-DF videos are not included. Obtain the dataset under its applicable access and redistribution terms. The local 5,000-video subset and its metadata/split membership must match the experiment.
- **Weights:** Four checkpoints are required for exact frozen-model inference. Download links are pending; hashes are supplied above.
- **Encoder provenance:** CLIP uses the specific V2 checkpoint, not the older lower-learning-rate CLIP checkpoint. AASIST uses the original frozen checkpoint.
- **No TEST tuning:** DEV was used for checkpoint/threshold selection. TEST was later used for evaluation and post-hoc XAI investigation; it is not a completely untouched holdout for all subsequent analysis choices.
- **Targets differ:** Visual-only CLIP's `modify_video` target differs from fusion's `modify_audio OR modify_video` target; interpret category comparisons accordingly.
- **Artifacts:** Model weights, videos, embedding caches, and prepared datasets are not tracked in ordinary Git. Small evaluation results and dataset identity records are included.
- **Portability:** Paths were made configurable for the main V2 export/preparation scripts, but historical scripts may still contain machine-specific paths. Some generated metadata records absolute cache paths.
- **Reproduction verification:** Existing checkpoint hashes and dataset signatures were checked; a complete clean-checkout regeneration and exact output-hash comparison remain outstanding.


