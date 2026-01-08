# Rainfall Prediction Classifier (Melbourne)

**Author:** Gautam Govind

This is a machine learning classification project that predicts whether it will rain **today** in the Melbourne metro area using historical weather observations.

The work focuses on:
- cleaning a real-world dataset (missing values, categorical directions, mixed feature types),
- preventing leakage by shifting the target to a “predict today using yesterday’s available info” setup,
- training and comparing two classifiers (Random Forest vs Logistic Regression),
- evaluating beyond accuracy (precision, recall, confusion matrix),
- understanding what features drive predictions (feature importance).

---

## Dataset

Source: `weatherAUS-2.csv` (Australian daily weather observations, 2008–2017).

In this notebook, I filter to the Melbourne metropolitan region:
- Melbourne
- MelbourneAirport
- Watsonia

---

## Project Structure

- `FinalProject_AUSWeather.ipynb` - complete notebook with preprocessing, training, and evaluation
- `README.md` - this file

---

## Setup

### Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn seaborn
```

### Run the notebook

Open the `.ipynb` file in:
- Jupyter Notebook / JupyterLab, or
- Google Colab (upload the notebook)

Then run cells from top to bottom.

---

## Problem Framing (Leakage Fix)

The original dataset has:
- `RainToday` (today’s rain label)
- `RainTomorrow` (tomorrow’s rain label)

To make the prediction practical, I reframe it as:

- `RainYesterday` = original `RainToday`
- `RainToday` = original `RainTomorrow`

This way, the model predicts today’s rainfall using features available up to yesterday.

---

## Approach

### 1) Data cleaning
- Drop rows with missing values (`dropna()`).
- Keep only Melbourne-region locations.
- Convert `Date` into a seasonal feature (Summer/Autumn/Winter/Spring).
- Use a pipeline so preprocessing is applied consistently to train and test.

### 2) Preprocessing
- Numeric features: `StandardScaler`
- Categorical features: `OneHotEncoder(handle_unknown="ignore")`
- Combined using `ColumnTransformer`

### 3) Models trained
- Random Forest (with `GridSearchCV` hyperparameter tuning)
- Logistic Regression (with regularization + optional class balancing)

---

## Results (Melbourne Region)

Target distribution is imbalanced:
- ~76% “No rain”
- ~24% “Rain”

So accuracy alone is not enough. The important part is how well the model catches rainy days.

### Random Forest (best model in this notebook)
- **Accuracy:** ~84%
- **Precision (Rain):** ~75%
- **Recall (Rain):** ~51%
- **F1 (Rain):** ~0.61

Interpretation:
- When the model predicts rain, it is correct most of the time (good precision).
- It still misses about half of actual rainy days (recall needs improvement).

### Logistic Regression
- **Accuracy:** ~82%
- **Precision (Rain):** ~67%
- **Recall (Rain):** ~46%
- **F1 (Rain):** ~0.55

Random Forest performs better overall for this setup.

---

## What Features Matter Most

From Random Forest feature importance, the strongest predictors include:
- Humidity at 3pm
- Pressure at 3pm
- Cloud cover (9am and 3pm)
- Sunshine
- RainYesterday

These align with real-world intuition: moisture, pressure changes, cloudiness, and sunlight strongly relate to rainfall likelihood.

---

## Notes on Limitations

This project deliberately keeps the pipeline simple and explainable, but there are two real issues:

1) **Class imbalance**
   - Rain is the minority class.
   - High accuracy can happen even with weak rain detection.

2) **Recall is the bottleneck**
   - If the goal is “don’t miss rainy days”, then recall matters more than accuracy.
   - In real deployment (events, logistics, planning), missing rain can be costly.

---

## Improvements I’d Try Next

- Handle imbalance more directly:
  - `class_weight="balanced"` for tree models
  - SMOTE or other resampling
  - threshold tuning (optimize for recall or business cost)

- Stronger models:
  - Gradient Boosting (XGBoost / LightGBM / CatBoost)
  - calibrated probabilities + threshold selection

- Better temporal features:
  - rolling averages (humidity/pressure trends)
  - deltas (pressure3pm - pressure9am, temp3pm - temp9am)
  - lag features

- Evaluation aligned to use-case:
  - ROC-AUC / PR-AUC
  - cost-sensitive scoring (false negatives vs false positives)

---

## How to Use This in an Interview

If someone asks “What’s the main value here?” I keep it simple:
- I fixed leakage so the task is realistic.
- I used a clean preprocessing pipeline with mixed feature types.
- I tuned models properly with cross-validation.
- I didn’t stop at accuracy and showed why recall matters for rainfall.

---

**Project by Gautam Govind**
