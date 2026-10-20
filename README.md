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

**PondGuard** is an intelligent early-warning decision support system designed to help freshwater aquaculture farmers protect fish welfare, prevent hypoxia mortality, and maintain optimal pond conditions.

By integrating **Machine Learning pattern classification** (trained on 4,300 benchmark aquaculture water samples) with a **deterministic threshold screening engine**, **fish-behavior symptom detection**, and **farmer-friendly plain-language action guidance**, PondGuard provides actionable, multi-layered risk insights in seconds.

---

## ⚠️ Scientific Advisory & Simulation Notice

> **Important Advisory:** PondGuard is designed as an educational decision-support tool.
>
> 1. **Simulated Benchmark Dataset:** The underlying Mendeley dataset (*Aquaculture - Water Quality Dataset (WQD)* by Veeramsetty et al., DOI: 10.17632/y78ty2g293.2) consists of **synthetically simulated parameter distributions** (with benchmark extremes up to 84°C and pH 0.0039–14.85).
> 2. **Unvalidated on Live Farms:** The machine learning pipeline **has not yet been validated on live, physical aquaculture farm telemetry**. Real pond conditions experience diurnal photosynthetic dissolved-oxygen fluctuations and complex biological cycles.
> 3. **Probabilistic Guidance, Not Guarantees:** Model classifications represent probabilistic statistical pattern matches, never guaranteed biological outcomes.
> 4. **Independent Deterministic Guardrails:** Deterministic physiological threshold alerts (hypoxia $DO < 3.0$ mg/L, toxic ammonia $> 0.5$ mg/L, nitrite $> 0.5$ mg/L) run independently of the ML pipeline to guarantee immediate safety screening. Always confirm readings with calibrated physical test kits or probes.

---

## 🚀 Key Features

* **Farmer-Friendly Visual Dashboard:** Clear Green/Yellow/Red status badges for all parameters and an overall pond health indicator. No chemistry expertise required.
* **Suggested Farmer Action Plan:** Safe, practical operator steps for **Aeration**, **Feeding adjustments**, **Water exchange**, and **Fish health monitoring**.
* **AI Water Quality Classification:** Gradient Boosting Pipeline predicting **Excellent Quality**, **Good Quality**, and **Poor Quality** with full class probability meters.
* **Dual-Layer Safety Architecture:**
  * *Layer 1 (AI Pattern Recognition):* Evaluates multi-variate parameter relationships across Temperature, Dissolved Oxygen, pH, Ammonia, and Nitrite.
  * *Layer 2 (Deterministic Thresholds):* Checks critical physiological stress thresholds (e.g., severe hypoxia $DO < 3.0$ mg/L, elevated toxic ammonia $> 0.5$ mg/L, nitrite poisoning $> 0.5$ mg/L).
* **Fish Health Observation Screening:** Keyword screening flags visible symptoms (e.g., *gasping at surface, lethargy, lesions, unusual swimming*).
* **Interactive Demo Presets:** 1-click test scenarios (Optimal Pond, Routine Pond, Critical Hypoxia/Toxicity).
* **Historical Logging & CSV Export:** Download individual pond readings or complete multi-pond session logs for veterinary and farm auditing.

---

## 📊 Dataset & Model Benchmarks

* **Dataset:** Mendeley Data ([DOI: 10.17632/y78ty2g293.2](https://data.mendeley.com/datasets/y78ty2g293/2)), Veeramsetty, Arabelli, and Bernatin.
* **Sample Count:** 4,300 simulated aquaculture samples (balanced 3-class distribution).
* **Model Pipeline:** `Pipeline([('imputer', SimpleImputer(strategy='median')), ('classifier', GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42))])`.
* **Reproducible 3-Way Split:** 70% Train (3,010) / 15% Validation (645) / 15% Unseen Test (645) stratified split.
* **Evaluation Results (Holdout Test Set - 645 Unseen Samples):**
  * **Test Accuracy:** **87.60%** (vs 34.88% Most-Frequent Baseline, 81.09% Decision Tree depth=3)
  * **Macro F1-Score:** **0.8730** | **Macro Precision:** 0.8805 | **Macro Recall:** 0.8801
  * **Poor-Class (Class 2) Recall:** **70.22%** | **Precision:** 92.40% | **F1-Score:** 0.7980
  * **Probability Calibration:** Multi-class Log-Loss = 0.3842; Brier Score (Poor Class) = 0.1076.
  * *False Negative Analysis:* 25.7% of Poor samples in the 14-parameter dataset have stress confined exclusively to hidden variables ($\text{CO}_2$, BOD, Turbidity). On samples where stress is captured by the 5 measured parameters, model recall is **96.82%**.

---

## 🛠️ Quickstart Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Retrain or Verify the Aquaculture Model
```bash
python train_aquaculture_model.py
```

### 3. Launch the PondGuard Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.


