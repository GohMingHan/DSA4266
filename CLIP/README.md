# CLIP ViT-B/32 — Preliminary Deepfake Detection Experiments

**DSA4266 Project | CLIP Experiments**

This branch documents preliminary visual deepfake detection experiments on LAV-DF using a pretrained CLIP ViT-B/32 vision encoder with a linear binary classification head. It covers lightweight frame preprocessing, initial fine-tuning, overfitting experiments, standardized DEV evaluation, and qualitative Grad-CAM analysis.

> **Scope:** These three notebooks document **preliminary** CLIP experiments. The final CLIP model used in the group's visual-model comparison was **retrained separately under the shared visual training protocol**. Neither the preliminary Lower LR checkpoint nor its standardized DEV results should be presented as the final retrained CLIP results. TEST evaluation is outside the scope of these notebooks.

## 1. Project objectives

- Explore CLIP ViT-B/32 fine-tuning for visual deepfake detection.
- Investigate overfitting through a frozen-encoder configuration and a lower encoder learning rate.
- Compare preliminary checkpoints on DEV without using TEST for model selection.
- Evaluate the selected preliminary checkpoint under the shared DEV protocol.
- Inspect selected DEV predictions with ViT-compatible Grad-CAM.

## 2. Notebook structure

| Notebook | Purpose |
|---|---|
| `01-Preliminary_Frame-Preprocessing.ipynb` | Extract frames for the preliminary TRAIN/VAL experiments. |
| `02_Initial_CLIP_FineTuning.ipynb` | Fine-tune the initial CLIP model and record its training and DEV results. |
| `03_CLIP_Overfitting_Experiments.ipynb` | Compare frozen-encoder and lower-LR configurations; run the preliminary checkpoint's standardized DEV evaluation and DEV Grad-CAM analysis. |

The notebooks are organized in the order above. Some later evaluation cells restore saved checkpoints and outputs rather than repeating training.

## 3. Data preparation and evaluation

### Preliminary TRAIN/VAL preprocessing

To reduce extraction and training cost, up to **three frames per video** were used in the preliminary experiments. Real frames were uniformly sampled; fake frames were sampled from annotated manipulation intervals. The prepared dataset contained **12,000 TRAIN frames** and **1,500 VAL frames**. This frame dataset was **not** the training input for the final retrained CLIP model.

### Preliminary whole-video DEV comparison

The initial CLIP, frozen encoder, and lower-LR configurations were compared on DEV using **16 uniformly sampled frames per video and max pooling**. This protocol is distinct from the later standardized DEV evaluation.

### Standardized DEV evaluation of the preliminary checkpoint

The selected preliminary **Lower LR, Epoch 7** checkpoint was evaluated **without further training** using the shared **10 FPS PTS-based frame sampling** and **Top-5% mean video aggregation** protocol. The DEV evaluation comprised **500 videos** and **41,746 sampled frames** (40,115 Real; 1,631 Fake). Frame and video classification metrics below use threshold **0.5**; AUC metrics use continuous scores.

For each video, CLIP assigns a fake score to each sampled frame. The highest-scoring 5% of frames are averaged to produce one video-level score. This emphasizes potentially manipulated intervals, but the selected high-scoring frames are not guaranteed to coincide with annotated manipulation intervals.

## 4. Preliminary results

### Original whole-video DEV comparison

| Configuration | Video DEV ROC-AUC |
|---|---:|
| Initial CLIP | 0.8488 |
| Frozen encoder | 0.6824 |
| **Lower LR** | **0.8975** |

The initial model showed signs of overfitting. Freezing the encoder limited adaptation; reducing the encoder learning rate from **1e-5** to **3e-6** gave the strongest preliminary whole-video DEV ROC-AUC, although it did not eliminate overfitting.

### Standardized DEV: preliminary Lower LR checkpoint

| Metric | Frame DEV | Video DEV |
|---|---:|---:|
| Accuracy | 0.9199 | 0.8200 |
| Precision | 0.2992 | 0.7857 |
| Recall | 0.7830 | 0.8800 |
| F1-score | 0.4330 | 0.8302 |
| F2-score | 0.5916 | 0.8594 |
| ROC-AUC | 0.9369 | 0.8946 |
| PR-AUC (trapezoidal) | 0.5873 | 0.8853 |

Frame-level Fake prevalence is approximately 3.9%, whereas the video DEV split is balanced. Frame and video PR-AUC should therefore not be interpreted as directly comparable measures of task difficulty. These results describe an **intermediate preliminary checkpoint**, not the final retrained model.

## 5. Interpretability: Grad-CAM

ViT-compatible Grad-CAM was applied to selected **DEV** frames for qualitative inspection. Gradients of the Fake-class logit with respect to intermediate transformer patch features yield a coarse **7 × 7** spatial attribution map, upsampled to the **224 × 224** input. These maps illustrate which regions contributed to the model's prediction; they are **not validated manipulation masks** or evidence of precise forgery localization. Selected examples were used for illustration, not checkpoint selection.

## 6. Kaggle resources

The following datasets are published under the Kaggle account `yanhanr`. They provide the original data, preliminary training frames, saved checkpoints, standardized evaluation resources, and prediction outputs used in this workflow.


| Resource | Role | Kaggle link / identifier |
|---|---|---|
| LAV-DF-5000 | Original videos and metadata | [LAV-DF-5000](https://www.kaggle.com/datasets/yanhanr/lav-df-5000) |
| LAV-DF-frame | Prepared preliminary TRAIN/VAL frames | [LAV-DF-frame](https://www.kaggle.com/datasets/yanhanr/lav-df-frame) |
| CNN standardized DEV protocol | Shared DEV sampling index and evaluation utilities | [Standardized DEV Protocol](https://www.kaggle.com/datasets/yanhanr/cnn-standardized-dev-protocol) |
| CLIP baseline + LR | Preliminary initial and Lower LR checkpoints (including `clip_lower_lr_best.pt`) | [CLIP Baseline + LR](https://www.kaggle.com/datasets/yanhanr/clip-baseline-lr) |
| CLIP ViT-B/32 best | Initial checkpoint (`clip_vit_b32_best.pt`) | [CLIP ViT-B/32 Best](https://www.kaggle.com/datasets/yanhanr/clip-vit-b23-best) |
| Standardized DEV frame predictions | Published intermediate frame scores | [Frame Predictions](https://www.kaggle.com/datasets/yanhanr/clip-standardized-dev-frame-predictions-csv) |
| Standardized DEV video predictions | Published intermediate video scores | [Video Predictions](https://www.kaggle.com/datasets/yanhanr/clip-standardized-dev-video-predictions) |


**Naming note:** The checkpoint dataset's display title has been corrected to `clip_vit_b32_best`, but its existing Kaggle URL slug remains `clip-vit-b23-best`. This is a dataset naming typo, not a different model architecture. The actual checkpoint file is named `clip_vit_b32_best.pt`.

### Paths used in Kaggle notebooks

```text
/kaggle/input/datasets/yanhanr/lav-df-5000/LAV-DF-5000
/kaggle/input/datasets/yanhanr/lav-df-frame
/kaggle/input/datasets/yanhanr/cnn-standardized-dev-protocol
```

Notebook 03 also searches attached inputs for `metadata.min.json`, `clip_lower_lr_best.pt`, and saved standardized DEV prediction/configuration files. The checkpoint and output datasets must be attached where required. Kaggle mount paths can vary between accounts and dataset versions; inspect `/kaggle/input` and update `DATA_ROOT` or other path variables as necessary.

## 7. Downloading datasets

Install and authenticate the Kaggle CLI according to Kaggle's instructions:

```bash
pip install kaggle
```

Download the three core datasets:

```bash
kaggle datasets download -d yanhanr/lav-df-5000 --unzip
kaggle datasets download -d yanhanr/lav-df-frame --unzip
kaggle datasets download -d yanhanr/cnn-standardized-dev-protocol --unzip
```

Download the initial CLIP checkpoint dataset (the legacy `b23` URL is intentional):

```bash
kaggle datasets download -d yanhanr/clip-vit-b23-best --unzip
```

For the baseline/Lower LR checkpoint dataset and published DEV prediction datasets, open their Kaggle pages and copy the **exact** `owner/slug` into `kaggle datasets download -d owner/slug --unzip`. Do not derive CLI identifiers from display titles. Download into separate directories to avoid filename collisions.

On Kaggle, attach the relevant datasets using **Add Input** before running a notebook. Original video data may be subject to LAV-DF licensing and redistribution conditions.

## 8. Reproducibility and scope

The preliminary workflow is: **frame preprocessing → initial CLIP fine-tuning → frozen/lower-LR experiments → preliminary DEV comparison → standardized DEV evaluation of the selected preliminary checkpoint → DEV Grad-CAM**.

Reproducing training from scratch requires the source data, frame dataset, dependencies, and sufficient compute. Restoring saved checkpoints allows later evaluation stages to run without repeating training. The notebooks preserve Kaggle-specific path assumptions, so execution in another account or locally may require path adjustments. Saved prediction datasets are supplementary outputs, not replacements for the required input data and checkpoints.

### Final Standardized CLIP Model

Following the preliminary experiments documented here, CLIP ViT-B/32 was retrained using the shared visual training protocol adopted for the CNN models, including standardized 10 FPS sampling and training-frame selection. The final checkpoint was selected at Epoch 3 based on DEV frame-level ROC-AUC.

The complete retraining procedure, final DEV evaluation results, and Grad-CAM visualizations are documented separately in the **`CNN-Experiments`** branch.

## 9. Limitations

- Preliminary 3-frame training and standardized final-model training are different experimental stages.
- Preliminary 16-frame max-pooling DEV results and 10 FPS Top-5% standardized DEV results are not interchangeable.
- DEV-guided model and threshold choices should not be presented as independent TEST performance.
- Class imbalance affects frame-level precision and PR-AUC interpretation.
- Grad-CAM provides qualitative attribution, not verified manipulation localization.
