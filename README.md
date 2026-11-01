# Audio Deepfake Detection Using ResNet18 and AASIST

**DSA4266 Project | Audio Experiment**

This repository documents an audio-based deepfake detection experiment comparing **ResNet18** and **AASIST** on the **LAV-DF** dataset. The study evaluates classification performance and investigates model behaviour using **Integrated Gradients (IG)** and complementary acoustic-feature visualisations.

> **Scope:** This branch focuses on audio-based detection. The results below relate to the evaluated LAV-DF test split and should not be interpreted as guaranteed performance on unseen datasets or real-world recordings.

## 1. Project objectives

- Distinguish authentic from audio-manipulated samples using two different neural network architectures.
- Compare ResNet18's time-frequency representation with AASIST's raw-waveform processing.
- Evaluate classification using accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices.
- Examine the temporal evidence used by the models through Integrated Gradients, alongside signal-level acoustic features.
- Identify limitations and opportunities for further evaluation.

## 2. Dataset and preprocessing

The experiments use **LAV-DF (Localized Audio Visual DeepFake)**. The prepared experiment data include **4,000 training**, **500 development** and **500 test videos**. The test split contains **250 authentic** and **250 manipulated** samples.

## Dataset and Experimental Resources

The datasets, experimental resources, and explainability outputs used in this project are hosted on Kaggle. Due to the size of the datasets, they are maintained separately from the GitHub repository.

### Kaggle Resources

| Resource | Description | Link |
|---|---|---|
| Experiment 2A (Version 10) | Published Experiment 2A dataset, with Version 10 referenced for reproducibility. | [Kaggle Dataset](https://www.kaggle.com/datasets/gohminghan/experiment-2a/versions/10) |
| LAV-DF Experiment 2A Audio | Audio dataset associated with the deepfake detection experiments. | [Audio Dataset](https://www.kaggle.com/datasets/gohminghan/lav-df-experiment-2a-audio) |
| Deepfake Audio XAI Results | Published explainability results associated with ResNet18 and AASIST. | [XAI Results](https://www.kaggle.com/datasets/gohminghan/deepfake-audio-xai-results) |
| Audio Extraction | Kaggle notebook documenting the audio extraction workflow. | [Audio Extraction Notebook](https://www.kaggle.com/code/gohminghan/audio-extraction) |

### Downloading the Datasets

The datasets can be downloaded through the Kaggle website or using the Kaggle API.

Install the Kaggle CLI:

```bash
pip install kaggle
```

Download the Experiment 2A dataset:

```bash
kaggle datasets download -d gohminghan/experiment-2a -v 10 --unzip
```

Download the LAV-DF audio dataset:

```bash
kaggle datasets download -d gohminghan/lav-df-experiment-2a-audio --unzip
```

Download the explainability results:

```bash
kaggle datasets download -d gohminghan/deepfake-audio-xai-results --unzip
```

Run these commands in separate directories to avoid unintended filename collisions.

### Reproducibility

The project evaluates two deep learning architectures, ResNet18 and AASIST, for audio deepfake detection.

The published resources support the following stages of the experimental workflow:

1. Audio extraction and preparation.
2. Model training and evaluation.
3. Classification performance analysis.
4. Explainable AI analysis using Integrated Gradients.
5. Acoustic feature analysis and interpretation.

The GitHub repository contains the experiment implementation and documentation, while Kaggle provides access to the associated datasets and published outputs.

**Note:** Access to the original LAV-DF material and any redistributed derivatives remains subject to the applicable dataset licence and usage conditions.

Audio is processed at **16 kHz** and segmented into windows of **64,600 samples (approximately 4.04 seconds)**, using a **0.5-second stride**. Each model receives a different representation of these windows:

| Component | ResNet18 | AASIST |
|---|---|---|
| Input | Single-channel log-Mel spectrogram | Raw audio waveform |
| Window length | 64,600 waveform samples before feature extraction | 64,600 waveform samples |
| Representation | 128 Mel-frequency bins; example feature shape 128 × 404 | One-dimensional waveform |
| Prediction task | Binary authentic/manipulated classification | Binary authentic/manipulated classification |

The audio preprocessing is shared where appropriate; model-specific transformations are applied afterward. Video-level predictions in the reported evaluation use **maximum pooling** over window-level scores.

## 3. Model architectures

### ResNet18

ResNet18 is adapted from a residual convolutional image classifier to operate on log-Mel spectrograms. Its first convolution accepts a single channel instead of RGB input. The backbone generates a **512-dimensional embedding**, followed by a binary classification head comprising **dropout (0.2)** and a linear layer producing one logit.

### AASIST

AASIST (**Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks**) operates on raw waveforms and learns spectro-temporal representations using its anti-spoofing architecture and graph-attention components. It provides an architectural contrast to ResNet18's spectrogram-based convolutional processing.

## 4. Training configuration

The experiments used the following reported training settings:

| Setting | Value |
|---|---|
| Maximum epochs | 100 |
| Optimiser | AdamW |
| Loss | BCEWithLogitsLoss |
| Learning-rate scheduler | Cosine annealing |
| Batch size | 4 |
| Initial learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Sampling | Weighted sampling |
| Gradient clipping | Maximum norm 5.0 |
| Early stopping patience | 5 epochs |

The maximum of **100 epochs** is a training cap, not a claim that either model trained for all 100 epochs. Previously reported selected checkpoints were from **epoch 2 for ResNet18** and **epoch 1 for AASIST**. Checkpoint selection and full configuration should be cross-checked against the saved run artifacts when reproducing the experiment.

## 5. Test results

### 5.1 Classification performance

Both models were evaluated on the **same 500-video test set**.

| Metric | ResNet18 | AASIST |
|---|---:|---:|
| Accuracy | 0.998000 (99.80%) | 0.998000 (99.80%) |
| Precision | 0.996016 | 0.996016 |
| Recall | 1.000000 | 1.000000 |
| F1-score | 0.998004 | 0.998004 |
| ROC-AUC | 0.999936 | 1.000000 |
| Selected video-level threshold | 0.998295 | 0.949567 |

The selected decision thresholds are **model-specific** and should not be compared as if they measure relative confidence or calibration.

### 5.2 Confusion matrices

Rows are ground-truth labels; columns are predicted labels. **Manipulated** is treated as the positive class.

**ResNet18**

| Actual / Predicted | Authentic | Manipulated |
|---|---:|---:|
| Authentic | 249 (TN) | 1 (FP) |
| Manipulated | 0 (FN) | 250 (TP) |

**AASIST**

| Actual / Predicted | Authentic | Manipulated |
|---|---:|---:|
| Authentic | 249 (TN) | 1 (FP) |
| Manipulated | 0 (FN) | 250 (TP) |

Both models correctly identified all 250 manipulated test samples, with one false positive each. Their identical confusion-matrix totals do **not** mean they necessarily made errors on the same recording.

### 5.3 Interpretation

ResNet18 and AASIST achieve the same test-set accuracy, precision, recall and F1-score. AASIST's ROC-AUC of 1.000000 is slightly higher than ResNet18's 0.999936; this difference is small and does not establish statistically significant superiority or generalisation to other datasets.

## 6. Explainable AI and acoustic analysis

The interpretability workflow examines **Integrated Gradients (IG)** attributions and signal-level acoustic features. The verified visualisation set comprises:

| Model | IG figures | Acoustic-feature figures |
|---|---:|---:|
| ResNet18 | 10 | 10 |
| AASIST | 10 | 10 |
| **Total** | **20** | **20** |

**Integrated Gradients** attributes model output to portions of the input relative to a baseline. ResNet18's explanations are associated with its time-frequency spectrogram input, whereas AASIST's explanations are associated with its waveform-based input. These explanations describe model sensitivity; they do not independently prove that a highlighted region is manipulated.

**Acoustic-feature visualisations** provide complementary context through:

- Root mean square (RMS) energy
- Spectral flux
- Spectral flatness
- Fundamental frequency (F0), where measurable

These features characterise properties of the audio signal rather than directly explaining a model's internal decision. Ground-truth manipulated intervals can be overlaid for visual comparison. Since both models can be analysed on the same underlying waveform, the acoustic curves themselves are not separate model performance measurements.

### 6.1 Findings and limitations

The visualisations support a qualitative examination of whether attribution patterns coincide with annotated manipulated intervals and observable acoustic changes. Interpretation should account for window boundaries, alignment, the choice of IG baseline, and the fact that signal changes can occur in authentic speech as well.

The verified set of 40 figures is **a collection of visualisations**, not 40 independent model accuracy tests. No occlusion-sensitivity experiment is claimed here.

## 7. Error analysis

Although each model produced only one false positive on the 500-video test set, the identity of those errors differed in the previously reported analysis. Example contrasting scores were:

| Test video | ResNet18 score | AASIST score |
|---|---:|---:|
| `test/046416.mp4` | 0.999746 | 0.014089 |
| `test/130104.mp4` | 0.002433 | 0.967995 |

These examples illustrate how models with similar aggregate metrics can behave differently on individual recordings. The scores should be interpreted against each model's own decision threshold.

## 8. Reproducibility and repository contents

The cleaned experiment notebook is intended to document model training, evaluation, Integrated Gradients and acoustic analysis. Actual file availability and paths depend on the contents committed to this branch.

- `deepfake_audio_github_clean_v2.ipynb` — cleaned experiment notebook, if committed under this filename.
- `README.md` — project overview and reported findings.

**To reproduce:** obtain the LAV-DF dataset subject to its access and redistribution terms; prepare the audio manifest and derived features; configure dataset/checkpoint paths in the notebook; then execute preprocessing, model training or checkpoint loading, evaluation and XAI sections as applicable. Paths originally used in Kaggle may require adaptation to another environment. Do not assume that the notebook will execute unchanged without the original datasets, model definitions and saved artifacts.

**Dataset and checkpoints:** Large media files and trained checkpoints are not presumed to be included in this repository. Check the dataset licence and permissions before publishing any derived media or dataset samples.

## 9. Limitations and future work

- Evaluate on external datasets, unseen synthesis methods and more challenging recording conditions.
- Report confidence intervals and paired statistical comparisons before drawing strong conclusions from small differences in test metrics.
- Investigate false positives and whether errors are associated with recording quality or other non-manipulation cues.
- Assess attribution robustness and temporal localisation more systematically.
- Verify reproducibility using the final notebook, environment dependencies, saved configuration and model checkpoints.

## 10. Conclusion

**ResNet18 and AASIST both achieved 99.8% accuracy on the 500-video LAV-DF test split**, with perfect recall for manipulated samples and one false positive per model. Their near-identical aggregate performance makes the comparison of learned evidence especially important. Integrated Gradients and acoustic-feature visualisations provide complementary views of model behaviour, while differences in individual predictions highlight the need for case-level error analysis. These findings are promising within the evaluated split, but external validation remains necessary.
