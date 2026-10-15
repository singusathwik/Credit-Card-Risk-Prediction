## Credit Card Score Risk Insights

An end‑to‑end machine learning project that predicts **credit card default risk for the next month** and exposes it through a **professional web interface**.

The system:
- Trains advanced ML models on the famous **“Default of Credit Card Clients”** dataset.
- Engineers 25+ financial and behavioral features.
- Selects and saves the **best model** (ensemble‑based) with about **81% test accuracy**.
- Serves real‑time risk scores through a **Flask API**.
- Provides a modern, single‑page **Tailwind CSS** UI for business users.

---

## 1. Project Overview

- **Goal**: Estimate the probability that a credit card customer will **default next month**.
- **Target variable**: `IsDefaulter` (1 = will default, 0 = will not default).
- **Main outputs**:
  - Default probability (0–1 and %)
  - Risk level (Low / Medium / High)
  - Model accuracy summary (train & test)

### High‑Level Architecture

- **Training script**: `train_model.py`
  - Loads and cleans the raw Excel dataset.
  - Builds a rich feature matrix with engineered features.
  - Trains multiple models and advanced ensembles.
  - Selects the best model and saves:
    - `model.pkl` (best model + preprocessing)
    - `feature_columns.json` (feature order)
    - `model_accuracy.json` (metrics).

- **Backend API**: `server.py`
  - Flask app that loads the saved model artifacts.
  - Endpoints:
    - `GET /` – serves the frontend (`index.html`).
    - `GET /accuracy` and `GET /api/accuracy` – returns accuracy metrics.
    - `POST /predict` and `POST /api/predict` – returns risk predictions.

- **Frontend UI**: `index.html`
  - Tailwind‑styled single page.
  - Uses vanilla JavaScript to:
    - Call the Flask API.
    - Display model accuracy, probability, and risk levels.

- **Testing utility**: `test_model.py`
  - Simple CLI script to quickly verify the saved model and accuracy.

---

## 2. How the Model Works

### 2.1 Dataset

File: `default of credit card clients.xls`

Key raw columns used:
- **Demographics**: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
- **Repayment status**: `PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6`
- **Bill amounts**: `BILL_AMT1` … `BILL_AMT6`
- **Payment amounts**: `PAY_AMT1` … `PAY_AMT6`

The column `default payment next month` is renamed to `IsDefaulter`.

### 2.2 Feature Engineering (in `build_dataset`)

Starting from the raw features, many new features are created, including:

- **Bill statistics**
  - `AVG_BILL_AMT`, `MAX_BILL_AMT`, `MIN_BILL_AMT`, `BILL_VOLATILITY`
- **Payment statistics**
  - `AVG_PAY_AMT`, `TOTAL_PAY_AMT`, `PAY_CONSISTENCY`
- **Ratios & utilization**
  - `PAY_BILL_RATIO` – payment vs bill
  - `UTILIZATION_RATIO` – average bill vs credit limit
  - `LIMIT_UTILIZATION` – max bill vs limit
  - `PAY_EFFICIENCY` – payment effectiveness vs bills
  - `PAY_COVERAGE_AVG` – average coverage across months
- **Delays & behavior**
  - `AVG_PAY_DELAY`, `MAX_PAY_DELAY`, `MIN_PAY_DELAY`
  - `DELAY_COUNT`, `SEVERE_DELAY_COUNT`
  - `PAY_RELIABILITY` – inverse of delay severity
  - `PAY_PATTERN_CONSISTENCY` – stability of payments
- **Trends & momentum**
  - `BILL_TREND`, `PAY_TREND`
  - `PAY_MOMENTUM`, `BILL_MOMENTUM`
  - `DEBT_GROWTH_RATE`
  - `UTIL_TREND_AVG`, `UTIL_TREND_STD`
- **Groups & interactions**
  - `AGE_GROUP`, `LIMIT_GROUP`
  - `OUTSTANDING_BAL1`, `OUTSTANDING_BAL_AVG`
  - `LIMIT_AGE_INTERACTION`
  - `DELAY_UTIL_INTERACTION`
  - `HIGH_RISK_INDICATOR` (combined flag from high utilization, delays, low efficiency)

All NaN / infinite values are cleaned using:
- `fillna(0)`
- replacing `np.inf` / `-np.inf` with 0
- `np.nan_to_num` to ensure the final feature matrix `X` is safe.

### 2.3 Training Pipeline (in `train_and_save`)

1. **Train/Test Split**
   - `train_test_split` with:
     - `test_size=0.2` (80% train, 20% test)
     - `stratify=y` (preserve class balance)
     - `random_state=42`

2. **Feature Selection**
   - `SelectKBest(f_classif, k='all')` currently used mainly as a scoring step.
   - Returns a consistent feature ordering and can be tuned later.

3. **Class Imbalance Handling**
   - If available: `SMOTE` (`imblearn.over_sampling.SMOTE`)
     - Oversamples minority class on the training set.
     - Helps the model learn defaulters better.

4. **Scaling**
   - `StandardScaler` on selected features for models that need it (especially boosting models).

5. **Models Trained**

   The script trains multiple strong models:

   - **Random Forest (RF)** – highly optimized:
     - `n_estimators=800`, `max_depth=35`
     - `min_samples_split=3`, `min_samples_leaf=1`
     - `max_features='sqrt'`
     - `class_weight='balanced_subsample'`

   - **Gradient Boosting (GB)** – with scaling in a `Pipeline`:
     - `n_estimators=500`, `learning_rate=0.05`
     - `max_depth=9`
     - `subsample=0.85`, `max_features='sqrt'`

   - **XGBoost (optional)** – if `xgboost` is installed:
     - `n_estimators=600`, `learning_rate=0.05`
     - `max_depth=8`, `min_child_weight=3`
     - `subsample=0.85`, `colsample_bytree=0.85`

   - **Second Random Forest (RF2)** – alternative hyperparameters for diversity.

   - **Stacking Ensemble**:
     - Base models: RF, GB (with its own scaler pipeline), RF2, and optionally XGBoost.
     - Final estimator: another GradientBoosting classifier.

   - **Voting Ensemble**:
     - Soft voting over RF, RF2, and optionally XGBoost.

6. **Model Selection**

   - Each model’s test accuracy is calculated on `X_test_selected`.
   - All accuracies are stored in a dictionary.
   - **Best model is chosen by highest test accuracy**.
   - Typical results on this dataset (after feature engineering & tuning):
     - RF / GB / XGB / Ensembles around **0.81 test accuracy (~81%)**.

7. **Metrics Saved**

   After selecting the best model:
   - Training accuracy on **SMOTE‑resampled** data.
   - Training accuracy on **original** training data.
   - Test accuracy on the held‑out test set.
   - Full classification report is printed.

   These are saved in `model_accuracy.json`:
   ```json
   {
     "accuracy": <test_accuracy>,
     "accuracy_percent": <test_accuracy * 100>,
     "train_accuracy": <train_acc_on_smote>,
     "train_accuracy_percent": ...,
     "train_accuracy_original": <train_acc_on_original>,
     "train_accuracy_original_percent": ...
   }
   ```

8. **Artifacts Saved**
   - `model.pkl` – best model plus selector (and scaler if needed).
   - `feature_columns.json` – base feature names used for API input.
   - `model_accuracy.json` – metrics for frontend and logging.

---

## 3. Backend API (`server.py`)

### 3.1 Libraries Used

- **Flask** – lightweight web framework.
- **Flask‑CORS** *(optional)* – CORS support (the code also has a manual fallback).
- **joblib** – load the trained model.
- **NumPy** – numeric arrays.

### 3.2 Model Loading

On startup:
- Loads `model.pkl` (either a bare model or a dict with `model`, `selector`, `model_type`).
- Loads `feature_columns.json`.
- Optionally logs the accuracy from `model_accuracy.json`.

### 3.3 Endpoints

- **`GET /`**
  - Serves `index.html`, the main UI.

- **`GET /accuracy` and `GET /api/accuracy`**
  - Reads `model_accuracy.json`.
  - Returns metrics as JSON.

- **`POST /predict` and `POST /api/predict`**
  - Expects JSON with all base input fields:
    - `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`,
    - `PAY_0` ... `PAY_6` (subset),
    - `BILL_AMT1` ... `BILL_AMT6`,
    - `PAY_AMT1` ... `PAY_AMT6`.
  - Builds an input row in the exact column order from `feature_columns.json`.
  - Applies the stored `selector` to match training features.
  - Calls `model.predict_proba(...)` if available.
  - Returns:
    ```json
    {
      "default_probability": 0.23,
      "will_default": false,
      "model_accuracy": 0.81
    }
    ```

### 3.4 CORS Handling

- Tries to use `Flask-CORS` if installed: `pip install flask-cors`.
- If not installed, a manual `@app.after_request` hook adds:
  - `Access-Control-Allow-Origin: *`
  - so that UIs served from other origins could still call the API.

---

## 4. Frontend (`index.html`)

The frontend is a **single HTML file** with:

- **Tailwind CSS via CDN** for styling.
- **Lucide icons via CDN** for professional icons.
- **Vanilla JavaScript** for:
  - Loading model accuracy on page load.
  - Submitting the form to `/api/predict`.
  - Updating the results view.

### Key UI Features

- Hero section with project name: **“Credit Card Score Risk Insights”**.
- Professional layout with:
  - **Customer Profile** form:
    - Basic info (limit, gender, education, marriage, age).
    - Repayment status for last 6 months.
    - 6 months of bill amounts and payment amounts.
  - **Risk Assessment card**:
    - Risk level badge (Low / Medium / High).
    - Progress bar showing default probability.
    - Text interpretation of risk.
    - Model accuracy display (from `model_accuracy.json`).
  - **Info cards** explaining models, real‑time analysis, and risk mitigation.

### Frontend Logic

- On load:
  - `loadAccuracy()` calls `GET /api/accuracy`.
  - Displays accuracy % in the stats panel and in the guidelines.

- On form submit:
  - Gathers all form fields into a JSON payload.
  - Sends `POST http://127.0.0.1:5000/api/predict`.
  - Shows a loading spinner on the button.
  - On success:
    - Updates risk level, color scheme, and bar width.
    - Updates default probability text.
    - Shows interpretation and accuracy.
  - On error:
    - Displays an error message banner.

---

## 5. Training & Testing Scripts

### 5.1 Training – `train_model.py`

Run:
```bash
python train_model.py
```

This will:
- Load Excel dataset.
- Engineer advanced features.
- Split into train/test.
- Optionally apply SMOTE (if `imbalanced-learn` is installed).
- Train multiple models:
  - 2x Random Forest, Gradient Boosting, optional XGBoost, Stacking, Voting.
- Compare and print accuracies.
- Select and save the best model.
- Save feature column list and accuracy metrics.

### 5.2 Quick Testing – `test_model.py`

Run:
```bash
python test_model.py
```

This will:
- Load `model.pkl` and `feature_columns.json`.
- Load `model_accuracy.json` (if present) and print accuracy.
- Build a fixed sample input.
- Print the predicted default probability and decision.

---

## 6. Libraries & Dependencies

### Python (core)

- **pandas** – reading and manipulating the Excel dataset.
- **numpy** – numeric arrays and safe feature computations.
- **scikit-learn**:
  - `RandomForestClassifier`, `GradientBoostingClassifier`
  - `VotingClassifier`, `StackingClassifier`
  - `train_test_split`, `SelectKBest`, `StandardScaler`
  - `classification_report`, `accuracy_score`
- **joblib** – saving and loading the trained model.
- **Flask** – backend web framework.

### Python (optional but recommended)

- **xgboost** – gradient boosting model with strong performance.
- **imbalanced-learn (SMOTE)** – handle imbalanced classes.
- **flask-cors** – nicer CORS handling (optional; fallback provided).

### Frontend

- **Tailwind CSS (CDN)** – modern utility‑first styling.
- **Lucide Icons (CDN)** – clean SVG icons.
- **Vanilla JavaScript** – fetch API, DOM manipulation.

---

## 7. How to Run the Full Project

1. **Install Python dependencies** (minimal):
   ```bash
   pip install flask scikit-learn pandas numpy joblib openpyxl
   ```

   Optional (for maximum accuracy and robustness):
   ```bash
   pip install xgboost imbalanced-learn flask-cors
   ```

2. **Train the model**:
   ```bash
   python train_model.py
   ```

3. **Start the Flask server**:
   ```bash
   python server.py
   ```

4. **Open the app**:
   - Navigate in your browser to:
     - `http://127.0.0.1:5000`

5. **Use the UI**:
   - Fill in customer profile, bills, and payments.
   - Click **“Run Risk Assessment”**.
   - Review:
     - Default probability.
     - Risk level (Low/Medium/High).
     - Model accuracy.

---

## 8. Summary

This project delivers:

- A **production‑style ML pipeline**:
  - Robust feature engineering.
  - Handling of class imbalance.
  - Model comparison and selection.
  - Persisted artifacts and metrics.

- A **clean API layer** via Flask:
  - Simple, well‑defined endpoints for predictions and metrics.

- A **professional UI**:
  - Tailwind‑based single page.
  - Real‑time risk feedback and visualization.

You can extend it by:
- Adjusting feature engineering.
- Tuning model hyperparameters.
- Enabling XGBoost and SMOTE for higher performance.
- Integrating authentication or logging for real deployments.

# Credit Card Score Risk Insights

A professional, AI-powered credit card default risk assessment platform built with React, Tailwind CSS, and advanced machine learning models.

## Features

- 🎯 **High Accuracy**: Ensemble ML models achieving 81%+ accuracy
- ⚡ **Real-Time Analysis**: Instant risk assessments
- 🎨 **Modern UI**: Beautiful, responsive design with Tailwind CSS
- 📊 **Detailed Insights**: Comprehensive risk scoring and probability analysis
- 🔒 **Professional**: Enterprise-grade risk assessment tool

## Tech Stack

### Frontend
- **React 18** - Modern UI framework
- **Vite** - Fast build tool
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Beautiful icons

### Backend
- **Flask** - Python web framework
- **Scikit-learn** - Machine learning library
- **XGBoost** (optional) - Advanced boosting algorithm
- **SMOTE** (optional) - Class imbalance handling

## Installation

### 1. Install Python Dependencies

```bash
pip install flask scikit-learn pandas numpy joblib openpyxl
```

Optional (for better accuracy):
```bash
pip install xgboost imbalanced-learn
```

### 2. Install Node.js Dependencies

```bash
npm install
```

## Usage

### 1. Train the Model

First, train the machine learning model:

```bash
python train_model.py
```

This will:
- Load and preprocess the credit card data
- Create advanced features
- Train multiple models (RF, GB, XGBoost, Ensemble)
- Save the best model to `model.pkl`
- Display model accuracy

### 2. Start the Backend Server

```bash
python server.py
```

The Flask server will start on `http://127.0.0.1:5000`

### 3. Start the Frontend Development Server

In a new terminal:

```bash
npm run dev
```

The React app will start on `http://localhost:3000`

## Project Structure

```
Credit Card/
├── src/
│   ├── components/
│   │   ├── Header.jsx          # Navigation header
│   │   ├── RiskForm.jsx        # Input form component
│   │   ├── RiskResults.jsx    # Results display component
│   │   └── StatsPanel.jsx      # Model accuracy panel
│   ├── App.jsx                 # Main application component
│   ├── main.jsx                # React entry point
│   └── index.css               # Global styles
├── train_model.py              # Model training script
├── server.py                   # Flask backend API
├── test_model.py              # Model testing script
├── package.json               # Node.js dependencies
├── vite.config.js             # Vite configuration
├── tailwind.config.js         # Tailwind CSS configuration
└── README.md                   # This file
```

## API Endpoints

### `POST /api/predict`
Submit customer data for risk assessment.

**Request Body:**
```json
{
  "LIMIT_BAL": 200000,
  "SEX": 1,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 35,
  "PAY_0": 0,
  "PAY_2": 0,
  ...
}
```

**Response:**
```json
{
  "default_probability": 0.23,
  "will_default": false,
  "model_accuracy": 0.81
}
```

### `GET /api/accuracy`
Get the model's accuracy metrics.

**Response:**
```json
{
  "accuracy": 0.81,
  "accuracy_percent": 81.0,
  "train_accuracy": 0.98,
  "train_accuracy_percent": 98.28
}
```

## Model Information

The system uses an ensemble of advanced machine learning models:

- **Random Forest**: Multiple trees for robust predictions
- **Gradient Boosting**: Sequential learning for accuracy
- **XGBoost**: Optimized gradient boosting (if available)
- **Stacking Ensemble**: Combines all models for best results

Features include:
- 25+ engineered features
- Feature selection
- SMOTE for handling class imbalance
- Hyperparameter optimization

## Development

### Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## License

This project is for educational and analytical purposes only.

## Disclaimer

This tool is for analytical support only and does not constitute a lending decision on its own. Always use this score alongside internal policies and manual review.

