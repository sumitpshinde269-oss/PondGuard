---
title: PondGuard - AI Aquaculture Water-Quality Alerts
emoji: 🐟
colorFrom: cyan
colorTo: blue
sdk: streamlit
sdk_version: 1.21.0
app_file: app.py
pinned: false
---

# 🐟 PondGuard: Early Water-Quality Alerts for Aquaculture Farms

**PondGuard** is an intelligent early-warning decision support system designed to help freshwater aquaculture farmers prevent fish mortality, reduce disease outbreaks, and maintain optimal pond conditions.

By integrating **Machine Learning pattern classification** (trained on 4,300 peer-reviewed aquaculture water samples) with a **deterministic threshold screening engine** and **fish-behavior symptom detection**, PondGuard provides actionable, multi-layered risk insights in seconds.

---

## 🚀 Key Features

* **AI Water Quality Classification:** Powered by an end-to-end Gradient Boosting Pipeline predicting **Excellent Quality**, **Good Quality**, and **Poor Quality** with full class probability distributions.
* **Dual-Layer Safety Architecture:**
  * *Layer 1 (AI Pattern Recognition):* Evaluates multi-variate parameter relationships across Temperature, Dissolved Oxygen, pH, Ammonia, and Nitrite.
  * *Layer 2 (Deterministic Thresholds):* Checks critical physiological stress thresholds (e.g., severe hypoxia $DO < 3.0$ mg/L, elevated toxic ammonia $> 0.5$ mg/L, nitrite poisoning $> 0.5$ mg/L).
* **Fish Health Observation Analysis:** Natural language screening flags distress symptoms (e.g., *gasping at surface, lethargy, lesions, unusual swimming*).
* **Interactive Demo Presets:** 1-click test scenarios (Optimal Pond, Routine Pond, Critical Hypoxia/Toxicity).
* **Historical Logging & CSV Export:** Download individual pond readings or complete multi-pond session logs for auditing and veterinary follow-up.

---

## 📊 Dataset & Model Benchmarks

* **Dataset Source:** Mendeley Data ([DOI: 10.17632/y78ty2g293.2](https://data.mendeley.com/datasets/y78ty2g293/2))
* **Study:** *Aquaculture - Water Quality Dataset (WQD)* by Veeramsetty, Arabelli, and Bernatin.
* **Sample Count:** 4,300 real aquaculture records.
* **Model Pipeline:** `SimpleImputer(strategy='median')` + `GradientBoostingClassifier(n_estimators=150, max_depth=4)`.
* **Evaluation Results (Stratified 20% Test Split - 860 Unseen Samples):**
  * **Test Accuracy:** **88.26%**
  * **Macro Precision:** 0.8878 | **Weighted Precision:** 0.8890
  * **Macro Recall:** 0.8867 | **Weighted Recall:** 0.8826
  * **Macro F1-Score:** 0.8797 | **Weighted F1-Score:** 0.8780

---

## 🛠️ Quickstart Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Retrain or Verify the Aquaculture Model (Optional)
```bash
python train_aquaculture_model.py
```

### 3. Launch the PondGuard Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

