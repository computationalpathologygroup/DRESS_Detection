# Automated Detection of DRESS Syndrome  
**by Sirada Kittipaisarnkul**  
**Radboud University Medical Center (RadboudUMC)**  

---

## Overview

This project presents a **weakly supervised deep learning pipeline** for the automated classification of **Drug Reaction with Eosinophilia and Systemic Symptoms (DRESS)** versus **Morbilliform Drug Eruption (MDE)** from whole slide images (WSIs). It incorporates modern *Multiple Instance Learning (MIL)* techniques and multi-resolution feature extraction to assist clinical diagnosis in dermatopathology.

---

## Dataset

> **Note**: This dataset is **private** and cannot be redistributed.

- **233 WSIs** from:
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

---

### 2. Multiple Instance Learning (MIL)

We compare three MIL-based pipelines:

#### A. **ABMIL (Attention-based MIL)**  
> Global attention pooling on patch features.

#### B. **CLAM (Clustering-constrained Attention MIL)**  
> Selects top-k most informative patches using attention.  
> Trained per encoder, then ensembled via late fusion (product of probabilities).

#### C. **Top-k ZoomMIL Refinement**  
> Uses top-k coordinates from 10× CLAM to zoom into 20× regions.  
> Aggregates both magnifications using average or sum fusion.

---

## Uploading the code.........80% > > >
