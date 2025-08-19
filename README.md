# Fraud Detection with Generative Augmentation (CTGAN → F-CTAB-GAN)


This repository explores **fraud detection using generative augmentation techniques**.
Starting with **CTGAN**, we balance highly imbalanced credit card fraud datasets, then extend into a novel architecture **F-CTAB-GAN**, designed to capture fraud-specific patterns with cost-sensitive learning and privacy-preserving training.

---

## Motivation

* Real-world fraud datasets are **extremely imbalanced** (fraud = <1%).
* Traditional models often have **high precision but poor recall** → they **miss many frauds**.
* Generative models like **CTGAN** can synthesize realistic fraud transactions to improve training.
* Our next step: **F-CTAB-GAN**, a fraud-aware GAN that learns fraud patterns explicitly while addressing **privacy, evolving frauds, and imbalanced data**.

---

## Completed Work: CTGAN Baseline

We first applied **CTGAN (Conditional GAN for Tabular Data)** on the **Kaggle Credit Card Fraud dataset**.

### Key Results

* **Without augmentation (baseline):**

  * Random Forest: High precision (0.93) but recall = 0.77 (missed frauds).
  * Logistic Regression: Recall only 0.65.
  * XGBoost: Balanced but weak overall.

* **With CTGAN (balanced data):**

  * **XGBoost** improved drastically (Precision = 0.88, Recall = 0.76, AUC ≈ 0.98).
  * Logistic Regression had higher recall (0.81) but too many false positives.
  * Random Forest balanced precision & recall better than baseline.

**Conclusion:** CTGAN **helped models learn fraud patterns** better, especially boosting **recall** (catching more frauds).

---

## Next Stage: F-CTAB-GAN

### Objective

Build a **fraud-aware, cost-sensitive GAN** (F-CTAB-GAN) to generate fraud-specific synthetic data and boost classifier performance beyond CTGAN.

### Innovations

* **Fraud-Pattern-Aware Conditional Vector**

  * Uses an **autoencoder** to embed transaction sequences (e.g., Time + Amount), inspired by HMM-like behavior.

* **Ensemble Generator**

  * **G1 (General MLP)** + **G2 (Fraud-Focused MLP)** → blended outputs to adapt to evolving frauds.

* **Cost-Sensitive Hybrid Loss**

  * Penalizes **false negatives** more heavily (since missing fraud is costly).

* **Fraud-Risk Imputation**

  * Missing values handled via autoencoder scores for realistic fraud patterns.

* **Privacy-Preserving Training**

  * Discriminator trained with **Differential Privacy (DP-SGD)**.

---

## Planned Architecture

```
+-------------------+
| Input Data        |
| Credit Card (Time,|
| Amount, Class)    |
+-------------------+
         ↓
+-------------------+
| Preprocessing     |
| - Fraud Autoenc.  |
| - VGM/Min-Max     |
| - Fraud-Risk Imp. |
| - SelectKBest     |
+-------------------+
         ↓
+-------------------+
| Cond. Vector      |
| Class + Fraud     |
| Pattern Embeds    |
+-------------------+
         ↓
+-------------------+
| Ensemble Gen.     |
| - G1 (General)    |
| - G2 (Fraud)      |
| - Blend (70/30)   |
| Loss: WGAN+GP +   |
| Info + Cost-Sens. |
+-------------------+
         ↓
| Discriminator +   |
| DP-SGD            |
+-------------------+
         ↓
| Aux Classifier    |
| Fraud Detection   |
+-------------------+
         ↓
| Output: Fraud-Aware|
| Synthetic Samples  |
+-------------------+
```

---

## Evaluation

We will compare **CTGAN vs F-CTAB-GAN** on:

* **Fraud Detection Performance:** Recall, Precision, F1, AUC, G-Mean
* **Data Utility:** Statistical similarity (KL-divergence, JS-divergence)
* **Privacy:** ε-privacy budget, resistance to inference attacks
* **Ablation Studies:** G1-only, G2-only, without fraud embeddings

---

## Roadmap

*  **CTGAN Implementation & Benchmarking**
*  **F-CTAB-GAN Development (in progress, 4–6 weeks in PyTorch)**
*  Ablation experiments + privacy evaluation
*  Paper-style writeup + PPT for final presentation

---

##  Usage


### Train CTGAN

```bash
python train_ctgan.py --dataset creditcard
```

### Train F-CTAB-GAN (upcoming)

```bash
python train_fctabgan.py --dataset creditcard --epochs 100
```

## Acknowledgments

* **CTGAN:** Xu et al. *"Modeling Tabular Data using Conditional GAN"* (2019)
* **Opacus:** Facebook AI’s library for differential privacy
* Kaggle for providing the **Credit Card Fraud dataset**

---

With CTGAN we showed augmentation **significantly improves fraud detection**.
Next, with **F-CTAB-GAN**, we aim to make fraud detection **fraud-aware, cost-sensitive, and privacy-preserving**.
