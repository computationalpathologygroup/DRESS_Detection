# Automated Detection of DRESS Syndrome  
**by Sirada Kittipaisarnkul**  
**Radboud University Medical Center (RadboudUMC)**  

---

## Overview

This project presents a **weakly supervised deep learning pipeline** for the automated classification of **Drug Reaction with Eosinophilia and Systemic Symptoms (DRESS)** versus **Morbilliform Drug Eruption (MDE)** from whole slide images (WSIs). It incorporates modern *Multiple Instance Learning (MIL)* techniques and multi-resolution feature extraction to assist clinical diagnosis in dermatopathology.

---

## Dataset

> **Note**: This dataset is **private** and cannot be redistributed.

- **231 WSIs** from:
  - Massachusetts General Hospital (MGH)  
  - Brigham and Women's Hospital (BWH)  
  - Ohio State University Wexner Medical Center (OSU)  
- Each WSI is labeled weakly at slide level (DRESS or MDE)

**Examples**  
<table>
<tr>
  <th>DRESS WSI</th>
  <th>MDE WSI</th>
</tr>
<tr>
  <td><img src="https://github.com/user-attachments/assets/94f15547-63e5-457a-a790-6acf167c5f6f" width="250"/></td>
  <td><img src="https://github.com/user-attachments/assets/899deef4-ad88-451f-a0c0-3e9a2c627869" width="250"/></td>
</tr>
<tr>
  <td><img src="https://github.com/user-attachments/assets/a6de7473-dd9c-4c92-a392-0be2fe3d38db" width="250"/></td>
  <td><img src="https://github.com/user-attachments/assets/84f0ccb5-c2e1-4714-a943-922d6d95c072" width="250"/></td>
</tr>
</table>

---

## Methodology

### 1. Patch Extraction & Feature Encoding (TRIDENT)

- WSIs are tiled at **10× and 20× magnification**.
- **TRIDENT** extracts patch-level features using multiple encoders:
  - [UNI (2024)](https://arxiv.org/abs/2402.11833)
  - [Gigapath (2024)](https://arxiv.org/abs/2402.09856)
  - Hoptim1 (internal encoder)

*Please refer to TRIDENT repository

---

### 2. Multiple Instance Learning (MIL)

We compare three MIL-based pipelines:

#### A. **ABMIL (Attention-based MIL)**  
> Global attention pooling on patch features.
> Use the same command as CLAM but change this flag to --model_type abmil

#### B. **CLAM (Clustering-constrained Attention MIL)**  
> Selects top-k most informative patches using attention.  
> Trained per encoder, then ensembled via late fusion (product of probabilities).
*Please refer to CLAM repository

#### C. **Top-k ZoomMIL Refinement**  
> Uses top-k coordinates from 10× CLAM to zoom into 20× regions.  
> Aggregates both magnifications using average or sum fusion.

```bash
python zoom.py \
  --checkpoint_path results/CLAM_UNI_10x.pt \
  --csv_path dataset_csv/dataset_split.csv \
  --features_dir_10x Data/Features/UNI_10x \
  --features_dir_20x Data/Features/UNI_20x \
  --output_csv results/zoom_predictions.csv \
  --fusion avg
```
---

### 3. Inference and Ensemble

#### Ensemble Inference Script
Late-fusion ensemble of CLAM models from multiple encoders:

```bash
python ensemble.py \
  --model_paths results/CLAM_UNI_20x.pt results/CLAM_Gigapath_20x.pt results/CLAM_Hoptim1_20x.pt \
  --feature_dirs Data/Features/UNI_20x Data/Features/Gigapath_20x Data/Features/Hoptim1_20x \
  --dataset_csv dataset_csv/dataset_split.csv \
  --output_csv results/ensemble_predictions.csv
```
---
### 4. Attention Heatmap Generation
We use CLAM's attention scores to visualize diagnostically relevant regions in each WSI. These heatmaps help interpret model focus and support clinical validation.
1. Run the CLAM model in inference mode.
2. Extract attention scores from the attention pooling layer.
3. Overlay the scores back onto WSI coordinates to generate a heatmap.
4. Save heatmaps as image files (PNG/JPG) and raw attention scores if needed


> **Note:**  
> Modify the configuration in [`heatmap/configs/config_template.yaml`](heatmap/configs/config_template.yaml) to match your data paths and settings.
