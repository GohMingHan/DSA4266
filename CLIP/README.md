# CLIP ViT-B/32: Preliminary Fine-Tuning and Standardized DEV Experiments

## Overview

These notebooks document the preliminary development of a CLIP ViT-B/32 visual deepfake detector on LAV-DF-5000, including initial fine-tuning, overfitting investigation, and standardized DEV evaluation.

The experiments compared an initial fully fine-tuned CLIP model, a frozen-encoder configuration, and a lower-learning-rate configuration. The Lower LR checkpoint (epoch 7) achieved the strongest preliminary DEV performance and was subsequently evaluated using the shared 10 FPS sampling protocol.

These notebooks do not contain the final CLIP model. Following the preliminary experiments, CLIP was retrained separately using the shared visual training protocol adopted for the CNN experiments. The final checkpoint was selected at epoch 3, achieving video-level DEV ROC-AUC of 0.9107 and PR-AUC of 0.9012. The final training results and Grad-CAM visualizations are maintained separately.

## Notebook 01: Preliminary Frame Preprocessing

Due to the computational cost of extracting and processing video frames, a lightweight preprocessing strategy was initially adopted for preliminary CLIP experiments. Up to three frames were sampled per video: visually real frames were sampled uniformly across the video, while visually fake frames were sampled within annotated manipulation intervals. The original TRAIN/DEV splits were preserved.

The resulting frame dataset was used for the initial CLIP fine-tuning and overfitting experiments in Notebooks 02 and 03. It was not used to train the final CLIP model, which was subsequently retrained using the shared 10 FPS visual sampling and training-frame selection protocol adopted for the CNN experiments.

## Notebook 02: Initial CLIP Fine-Tuning

### Purpose and training

Notebook 02 establishes a fully fine-tuned CLIP ViT-B/32 baseline using the preprocessed `LAV-DF-frame` TRAIN/VAL folders. It fixes random seed 42, applies CLIP-specific preprocessing, trains a two-class head together with the vision encoder, and selects the best checkpoint using **frame-level validation ROC-AUC**. The recorded best checkpoint was from **epoch 3**, with validation ROC-AUC **0.9361**. The training curves indicate overfitting: training loss continued decreasing while validation loss increased after its earlier minimum. The baseline motivates the frozen-encoder and lower-encoder-learning-rate experiments in Notebook 03.

### Original whole-video DEV protocol

The baseline whole-video DEV evaluation uses **500 DEV videos**, **16 uniformly sampled frames per video**, **max pooling** of frame fake probabilities, and a **fixed threshold of 0.5**. Video frames are sampled without using ground-truth manipulation intervals. Four aggregations were compared on the same saved DEV predictions: max, mean, top-3 mean and top-5 mean. The notebook's detailed DEV aggregation summary reports:

| DEV aggregation | ROC-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Max | 0.8543 | 0.7735 | 0.6852 | 0.8880 |
| Top-3 mean | 0.7817 | 0.7066 | 0.7308 | 0.6840 |
| Top-5 mean | 0.7637 | 0.5768 | 0.7052 | 0.4880 |
| Mean | 0.7570 | 0.2027 | 0.6522 | 0.1200 |

Two slightly different preliminary baseline DEV results were recorded: ROC-AUC 0.8543 (accuracy = 0.7400, F1 = 0.7735) and ROC-AUC 0.8488 (accuracy = 0.7460, F1 = 0.7720). The three-model comparison in Notebook 03 uses the latter results. This minor discrepancy may reflect differences between saved evaluation runs and has not been independently resolved.

## Data and task

The classifier predicts whether an individual RGB frame is visually manipulated. The preliminary training setup uses a balanced `LAV-DF-frame` collection of 12,000 training frames and 1,500 validation frames. The original LAV-DF-5000 videos are used for whole-video evaluation. The positive class is **Fake**: the ImageFolder mapping is `fake = 0`, `real = 1`, so the fake probability is obtained from `softmax(logits, dim=1)[:, 0]`.

The standardized DEV protocol uses a naturally imbalanced set of sampled frames and labels a frame Fake only when the video is visually manipulated and the sampled timestamp falls inside a visual-manipulation interval. Audio-only manipulation is not treated as visual manipulation. Video labels reflect whether the video contains visual manipulation.

## Model and training design

The architecture comprises the pretrained `openai/clip-vit-base-patch32` vision encoder with projection, normalized image embeddings, and a two-output linear classification head. Each experiment starts from the pretrained CLIP weights rather than continuing from the previously trained baseline checkpoint.

| Configuration | Encoder treatment | Encoder learning rate | Classifier learning rate | Best epoch | Best frame DEV ROC-AUC |
|---|---|---:|---:|---:|---:|
| Original baseline | Fully fine-tuned | 1e-5 | 1e-4 | 3 | 0.9361 |
| Experiment A: Frozen encoder | Frozen; train classification head | — | 1e-4 | 15 | 0.8190 |
| Experiment B: Lower LR | Fully fine-tuned | 3e-6 | 1e-4 | 7 | 0.9705 |

The experiments use AdamW with weight decay `1e-4` and seed 42. The lower-LR experiment uses CUDA automatic mixed precision with GradScaler and gradient clipping (maximum norm 1.0). The frozen experiment trains only the classification head, leaving the encoder in evaluation mode. Checkpoints within each training run are selected by **frame-level validation ROC-AUC**. The lower-LR run stopped after 12 epochs, retaining its best checkpoint from epoch 7.

### Original DEV comparison (16 frames, max pooling)

The initial whole-video comparison used 16 uniformly sampled frames per video, max pooling of frame fake probabilities, and a fixed video threshold of 0.5. This was the protocol for comparing the baseline and the two overfitting-mitigation experiments, not the final standardized evaluation.

| Model | Video accuracy | Video F1 | Video ROC-AUC |
|---|---:|---:|---:|
| Baseline | 0.7460 | 0.7720 | 0.8488 |
| Frozen encoder | 0.6120 | 0.6969 | 0.6824 |
| Lower encoder LR | 0.8080 | 0.8182 | 0.8975 |

The lower-LR configuration was selected for subsequent standardized evaluation because it performed best under the original DEV comparison. Freezing the encoder reduced the apparent training/validation divergence but substantially weakened detection performance. Lowering the learning rate did not eliminate later-epoch overfitting, making DEV-based checkpoint selection important.

## Why a standardized protocol was introduced

The original CLIP evaluation used 16 uniform frames and max pooling, whereas the CNN experiments used denser 10 FPS sampling and Top-5% mean aggregation. To make cross-model DEV comparisons more meaningful, the selected lower-LR CLIP checkpoint was re-evaluated using the CNN group's shared sampling and aggregation protocol. **The model weights were not changed or retrained.** CLIP-specific image preprocessing was retained, since different model architectures require different input normalization.

The shared DEV SQLite index contains exact source frame indices, PTS timestamps, and frame labels. The sampling policy starts at the first PTS, advances on a 10 FPS timeline, selects the nearest source frame (earlier frame on ties), and removes duplicate selections. PyAV decodes source frames and checks exact PTS alignment.

For each video with `N` sampled frames, sort fake probabilities in descending order, set `k = max(1, ceil(0.05 * N))`, and average the highest `k` probabilities. This is the **Top-5% mean video score**. ROC-AUC and PR-AUC are computed from continuous scores. Classification metrics use a specified threshold.

## Standardized DEV evaluation (Preliminary Lower LR Checkpoint)

The preliminary Lower LR checkpoint (epoch 7) was evaluated using the shared 10 FPS sampling and Top-5% mean aggregation protocol without additional training. These results represent an intermediate evaluation and should not be confused with the final CLIP model, which was subsequently retrained under the shared visual training protocol.

The standardized DEV set contains 500 videos and 41,746 sampled frames (40,115 Real; 1,631 Fake). Precision, recall, F1, F2 and accuracy were calculated at a fixed threshold of 0.5, while ROC-AUC and PR-AUC were computed from continuous prediction scores.

| Metric | Frame DEV (threshold 0.5) | Video DEV (threshold 0.5) |
|---|---:|---:|
| Accuracy | 0.9199 | 0.8200 |
| Precision | 0.2992 | 0.7857 |
| Recall | 0.7830 | 0.8800 |
| F1 | 0.4330 | 0.8302 |
| F2 | 0.5916 | 0.8594 |
| ROC-AUC | 0.9369 | 0.8946 |
| PR-AUC (trapezoidal) | 0.5873 | 0.8853 |

The frame-level class distribution is heavily imbalanced (approximately 3.9% Fake), making accuracy alone an inadequate performance measure. Frame-level and video-level PR-AUC values are not directly comparable because their prediction units and class prevalences differ. Video-level metrics were prioritized when assessing the preliminary model under the shared evaluation protocol.

## DEV threshold analysis

Thresholds from 0.05 to 0.95 in increments of 0.01 were evaluated using saved standardized **video-level DEV** scores. The selection rule maximized DEV video F1; ties were resolved by choosing the threshold closest to 0.5, then the lower threshold. F2 was reported as a recall-oriented supplementary measure, not used to select the final threshold.

| DEV video metric | Threshold 0.50 | Selected threshold 0.35 |
|---|---:|---:|
| Accuracy | 0.8200 | 0.8280 |
| Precision | 0.7857 | 0.7733 |
| Recall | 0.8800 | 0.9280 |
| F1 | 0.8302 | 0.8436 |
| F2 | 0.8594 | 0.8923 |
| False positives | 60 | 68 |
| False negatives | 30 | 18 |

A lower threshold of 0.16 achieved the highest observed F2 (0.9128) but increased false alarms. Threshold 0.35 was retained because **F1 was the chosen primary objective**. ROC-AUC and PR-AUC do not change when only the classification threshold changes.

The DEV-selected configuration for the preliminary Lower LR checkpoint was: epoch 7, 10 FPS PTS-based sampling, PyAV decoding, CLIP-specific image preprocessing, Top-5% mean aggregation, video threshold 0.35, and frame-level reporting threshold 0.5. This configuration was used for the preliminary standardized DEV analysis and was not adopted as the final team's CLIP configuration.

## Grad-CAM explainability demonstration

A small subset of DEV frames was selected for qualitative ViT-compatible Grad-CAM analysis using the **preliminary Lower LR CLIP checkpoint (epoch 7)**. The selected examples included true positives, true negatives, false positives, false negatives, and high-scoring frames contributing to the Top-5% mean aggregation of correctly detected fake videos.

Grad-CAM was generated using gradients of the fake-class logit with respect to intermediate Transformer patch-token representations. The resulting 7×7 attribution maps were upsampled to the 224×224 input resolution.

These heatmaps provide qualitative explanations rather than precise manipulation localization. Due to the coarse spatial resolution, highlighted facial or background regions cannot be interpreted as confirmed forgery artifacts. All examples were selected from DEV for illustration and were not used for model selection.

**These visualizations correspond to the preliminary checkpoint, not the final CLIP model retrained under the shared visual training protocol.**

## Reproducing the evaluation

The preliminary CLIP workflow consists of three notebooks, organized in the following order:

Notebook 01 — Frame Preprocessing: Extracts frames for the initial training and validation datasets.

Notebook 02 — Initial CLIP Fine-Tuning: Trains the initial CLIP model and evaluates its performance on DEV.

Notebook 03 — Overfitting Experiments: Investigates frozen-encoder and lower-learning-rate configurations, followed by standardized DEV evaluation and Grad-CAM analysis.

The notebooks expect Kaggle-mounted datasets containing the original LAV-DF-5000 videos and metadata, the preprocessed LAV-DF-frame dataset, relevant checkpoints, and shared protocol files (dev_evaluation_candidates.sqlite and data.py). Dataset paths may need to be updated when running in a different environment.

For a fresh run, prepare the required datasets and dependencies, and recreate the preliminary training checkpoints if necessary. Verify the shared DEV sampling index and decoding procedure, evaluate the selected Lower LR checkpoint on standardized DEV, and perform DEV-only threshold analysis. Grad-CAM visualizations can then be generated using the saved DEV predictions.

Preserve the preliminary checkpoints, DEV prediction CSVs, and selected configuration JSON to support reproducibility without rerunning the full evaluation. The notebooks document preliminary CLIP experiments only; the final retrained CLIP model and subsequent audio-visual fusion evaluation are maintained separately.

### Environment and Dataset Paths

The notebooks were developed and executed in Kaggle. Dataset paths are configured for the original Kaggle environment and may need to be updated when running under a different account or locally.

To reproduce the experiments, attach the required datasets and update the relevant paths (`DATA_ROOT`, `ORIGINAL_ROOT`, and `PROTOCOL_ROOT`) to match their locations. The original training and evaluation code is preserved to maintain consistency with the reported results.

## Interpretation and limitations

The preliminary experiments showed that freezing the CLIP encoder reduced apparent overfitting but weakened DEV performance, while reducing the encoder learning rate produced the strongest preliminary results. However, signs of overfitting remained, highlighting the importance of DEV-based checkpoint selection.

The standardized DEV evaluation enabled comparison under the shared sampling and aggregation protocol, but differences in model architecture, pretraining, and training data selection were not fully controlled. The heavily imbalanced frame-level labels and substantial video-level false positives also remain important limitations.

The Grad-CAM examples provide qualitative attribution insights rather than validated manipulation localization. The preliminary results should not be interpreted as the final CLIP model's performance, since the final model was subsequently retrained using the shared visual training protocol.