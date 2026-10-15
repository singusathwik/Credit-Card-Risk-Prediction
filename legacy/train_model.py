import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# Try to import advanced libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not available. Install with: pip install xgboost")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Note: SMOTE not available. Install with: pip install imbalanced-learn")


DATA_PATH = "default of credit card clients.xls"
MODEL_PATH = "model.pkl"
FEATURES_PATH = "feature_columns.json"
ACCURACY_PATH = "model_accuracy.json"


def load_data(path: str) -> pd.DataFrame:
  # In your notebook you used header=1 because first row is description
  df = pd.read_excel(path, header=1)
  # Rename target to a simpler name
  df = df.rename(columns={"default payment next month": "IsDefaulter"})
  return df


def build_dataset(df: pd.DataFrame):
  # Use the same raw columns as in the original dataset (except ID)
  feature_columns = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
  ]

  # Create a copy for feature engineering
  df_features = df[feature_columns].copy()
  
  # Feature Engineering: Add derived features that can improve accuracy
  print("Creating engineered features...")
  
  # Average bill amount (indicates spending pattern)
  bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
  df_features["AVG_BILL_AMT"] = df[bill_cols].mean(axis=1)
  df_features["MAX_BILL_AMT"] = df[bill_cols].max(axis=1)
  df_features["MIN_BILL_AMT"] = df[bill_cols].min(axis=1)
  df_features["STD_BILL_AMT"] = df[bill_cols].std(axis=1).fillna(0)
  
  # Average payment amount
  pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
  df_features["AVG_PAY_AMT"] = df[pay_amt_cols].mean(axis=1)
  df_features["TOTAL_PAY_AMT"] = df[pay_amt_cols].sum(axis=1)
  
  # Payment ratio (payment vs bill) - safe division
  df_features["PAY_BILL_RATIO"] = np.where(
    (df_features["AVG_BILL_AMT"] + 1) != 0,
    df_features["AVG_PAY_AMT"] / (df_features["AVG_BILL_AMT"] + 1),
    0
  )
  
  # Utilization ratio (bill amount vs credit limit) - safe division
  df_features["UTILIZATION_RATIO"] = np.where(
    (df["LIMIT_BAL"] + 1) != 0,
    df_features["AVG_BILL_AMT"] / (df["LIMIT_BAL"] + 1),
    0
  )
  
  # Average payment delay (PAY_0 to PAY_6)
  pay_cols = [f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]
  df_features["AVG_PAY_DELAY"] = df[pay_cols].mean(axis=1)
  df_features["MAX_PAY_DELAY"] = df[pay_cols].max(axis=1)
  df_features["MIN_PAY_DELAY"] = df[pay_cols].min(axis=1)
  
  # Bill trend (recent vs older bills) - safe division
  df_features["BILL_TREND"] = np.where(
    (df["BILL_AMT6"] + 1) != 0,
    (df["BILL_AMT1"] - df["BILL_AMT6"]) / (df["BILL_AMT6"] + 1),
    0
  )
  
  # Payment trend - safe division
  df_features["PAY_TREND"] = np.where(
    (df["PAY_AMT6"] + 1) != 0,
    (df["PAY_AMT1"] - df["PAY_AMT6"]) / (df["PAY_AMT6"] + 1),
    0
  )
  
  # Credit limit utilization - safe division
  df_features["LIMIT_UTILIZATION"] = np.where(
    (df["LIMIT_BAL"] + 1) != 0,
    df_features["MAX_BILL_AMT"] / (df["LIMIT_BAL"] + 1),
    0
  )
  
  # Additional advanced features
  # Payment consistency (std of payments)
  df_features["PAY_CONSISTENCY"] = df[pay_amt_cols].std(axis=1).fillna(0)
  
  # Bill volatility (std of bills)
  df_features["BILL_VOLATILITY"] = df[bill_cols].std(axis=1).fillna(0)
  
  # Recent payment behavior (last 3 months vs first 3 months)
  df_features["RECENT_PAY_RATIO"] = np.where(
    (df["PAY_AMT4"] + df["PAY_AMT5"] + df["PAY_AMT6"] + 1) != 0,
    (df["PAY_AMT1"] + df["PAY_AMT2"] + df["PAY_AMT3"]) / (df["PAY_AMT4"] + df["PAY_AMT5"] + df["PAY_AMT6"] + 1),
    0
  )
  
  # Recent bill ratio
  df_features["RECENT_BILL_RATIO"] = np.where(
    (df["BILL_AMT4"] + df["BILL_AMT5"] + df["BILL_AMT6"] + 1) != 0,
    (df["BILL_AMT1"] + df["BILL_AMT2"] + df["BILL_AMT3"]) / (df["BILL_AMT4"] + df["BILL_AMT5"] + df["BILL_AMT6"] + 1),
    0
  )
  
  # Payment delay severity (how many months with delays)
  delay_cols = [f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]
  df_features["DELAY_COUNT"] = (df[delay_cols] > 0).sum(axis=1)
  df_features["SEVERE_DELAY_COUNT"] = (df[delay_cols] > 2).sum(axis=1)
  
  # Age groups (might be useful)
  df_features["AGE_GROUP"] = pd.cut(df["AGE"], bins=[0, 30, 40, 50, 100], labels=[1, 2, 3, 4]).astype(float).fillna(2)
  
  # Credit limit groups
  df_features["LIMIT_GROUP"] = pd.cut(df["LIMIT_BAL"], bins=[0, 50000, 200000, 500000, 10000000], labels=[1, 2, 3, 4]).astype(float).fillna(2)
  
  # Outstanding balance (bill - payment)
  df_features["OUTSTANDING_BAL1"] = df["BILL_AMT1"] - df["PAY_AMT1"]
  df_features["OUTSTANDING_BAL_AVG"] = (df_features["AVG_BILL_AMT"] - df_features["AVG_PAY_AMT"])
  
  # Payment coverage (payment / bill ratio for each month, then average)
  coverage_ratios = []
  for i in range(1, 7):
    bill_col = f"BILL_AMT{i}"
    pay_col = f"PAY_AMT{i}"
    ratio = np.where((df[bill_col] + 1) != 0, df[pay_col] / (df[bill_col] + 1), 0)
    coverage_ratios.append(ratio)
  df_features["PAY_COVERAGE_AVG"] = np.mean(coverage_ratios, axis=0)
  
  # More advanced features for maximum accuracy
  # Payment momentum (acceleration/deceleration)
  df_features["PAY_MOMENTUM"] = df["PAY_AMT1"] - df["PAY_AMT3"]
  df_features["BILL_MOMENTUM"] = df["BILL_AMT1"] - df["BILL_AMT3"]
  
  # Debt accumulation rate
  df_features["DEBT_GROWTH_RATE"] = np.where(
    (df["BILL_AMT6"] + 1) != 0,
    (df["BILL_AMT1"] - df["BILL_AMT6"]) / (df["BILL_AMT6"] + 1),
    0
  )
  
  # Payment reliability score (inverse of delays)
  df_features["PAY_RELIABILITY"] = 1.0 / (df_features["AVG_PAY_DELAY"].abs() + 1)
  
  # Credit utilization trend
  util_trend = []
  for i in range(1, 7):
    bill_col = f"BILL_AMT{i}"
    util = np.where((df["LIMIT_BAL"] + 1) != 0, df[bill_col] / (df["LIMIT_BAL"] + 1), 0)
    util_trend.append(util)
  util_trend_array = np.array(util_trend)
  df_features["UTIL_TREND_AVG"] = np.mean(util_trend_array, axis=0)
  util_std = np.std(util_trend_array, axis=0)
  df_features["UTIL_TREND_STD"] = np.nan_to_num(util_std, nan=0.0)
  
  # Interaction features (multiplication of important features)
  df_features["LIMIT_AGE_INTERACTION"] = df["LIMIT_BAL"] * df["AGE"] / 1000000
  df_features["DELAY_UTIL_INTERACTION"] = df_features["AVG_PAY_DELAY"] * df_features["UTILIZATION_RATIO"]
  
  # Payment efficiency (how well they pay relative to bills)
  df_features["PAY_EFFICIENCY"] = np.where(
    (df_features["AVG_BILL_AMT"] + 1) != 0,
    df_features["AVG_PAY_AMT"] / (df_features["AVG_BILL_AMT"] + 1),
    0
  )
  
  # Risk indicators
  df_features["HIGH_RISK_INDICATOR"] = (
    (df_features["UTILIZATION_RATIO"] > 0.8).astype(int) +
    (df_features["AVG_PAY_DELAY"] > 1).astype(int) +
    (df_features["PAY_EFFICIENCY"] < 0.5).astype(int)
  )
  
  # Payment pattern consistency (lower std = more consistent)
  df_features["PAY_PATTERN_CONSISTENCY"] = 1.0 / (df_features["PAY_CONSISTENCY"] + 1)
  
  # Get updated feature columns list
  updated_feature_columns = list(df_features.columns)
  
  # Fill any remaining NaN values with 0
  df_features = df_features.fillna(0)
  
  # Replace infinity values with large finite numbers or 0
  df_features = df_features.replace([np.inf, -np.inf], 0)
  
  # Convert to numpy array and replace any remaining inf/nan values
  X = df_features.values.astype(np.float32)
  # Replace any inf or nan values that might have slipped through
  X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
  
  y = df["IsDefaulter"].values.astype(int)

  return X, y, updated_feature_columns


def train_and_save():
  print("Loading data from:", DATA_PATH)
  df = load_data(DATA_PATH)

  print("Building dataset...")
  X, y, feature_columns = build_dataset(df)

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
  )

  # Feature selection to remove noise and improve accuracy
  print("Performing feature selection...")
  selector = SelectKBest(score_func=f_classif, k='all')  # Keep all for now, can optimize later
  X_train_selected = selector.fit_transform(X_train, y_train)
  X_test_selected = selector.transform(X_test)
  print(f"Selected {X_train_selected.shape[1]} features out of {X_train.shape[1]}")

  # Save original y_train for evaluation
  y_train_original = y_train.copy()
  
  # Handle class imbalance with SMOTE if available
  if SMOTE_AVAILABLE:
    print("Applying SMOTE to handle class imbalance...")
    # Some versions of imbalanced-learn do not support n_jobs in SMOTE
    smote = SMOTE(random_state=42)
    X_train_selected, y_train = smote.fit_resample(X_train_selected, y_train)
    print(f"After SMOTE: Training samples = {len(y_train)}")

  # Scale features
  print("Scaling features...")
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train_selected)
  X_test_scaled = scaler.transform(X_test_selected)

  print("\n" + "="*60)
  print("Training Advanced Models for Maximum Accuracy")
  print("="*60)
  
  models = {}
  accuracies = {}
  
  # Model 1: Highly Optimized Random Forest
  print("\n[1/5] Training Optimized Random Forest...")
  rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=35,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced_subsample',
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
    verbose=0,
  )
  rf.fit(X_train_selected, y_train)
  rf_pred = rf.predict(X_test_selected)
  rf_acc = accuracy_score(y_test, rf_pred)
  models['rf'] = rf
  accuracies['rf'] = rf_acc
  print(f"   RF Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
  
  # Model 2: Gradient Boosting with optimal params
  print("\n[2/5] Training Gradient Boosting...")
  # Create a fresh scaler for the pipeline
  gb_scaler = StandardScaler()
  gb_pipeline = Pipeline([
    ('scaler', gb_scaler),
    ('gb', GradientBoostingClassifier(
      n_estimators=500,
      learning_rate=0.05,
      max_depth=9,
      min_samples_split=3,
      min_samples_leaf=1,
      subsample=0.85,
      max_features='sqrt',
      random_state=42,
      verbose=0,
    ))
  ])
  gb_pipeline.fit(X_train_selected, y_train)
  gb_pred = gb_pipeline.predict(X_test_selected)
  gb_acc = accuracy_score(y_test, gb_pred)
  models['gb'] = gb_pipeline
  accuracies['gb'] = gb_acc
  print(f"   GB Accuracy: {gb_acc:.4f} ({gb_acc*100:.2f}%)")
  
  # Model 3: XGBoost (if available) - often the best
  if XGBOOST_AVAILABLE:
    print("\n[3/5] Training XGBoost (Best Performance Expected)...")
    xgb_model = xgb.XGBClassifier(
      n_estimators=600,
      learning_rate=0.05,
      max_depth=8,
      min_child_weight=3,
      subsample=0.85,
      colsample_bytree=0.85,
      scale_pos_weight=1,
      random_state=42,
      n_jobs=-1,
      eval_metric='logloss',
      verbosity=0,
    )
    xgb_model.fit(X_train_selected, y_train)
    xgb_pred = xgb_model.predict(X_test_selected)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    models['xgb'] = xgb_model
    accuracies['xgb'] = xgb_acc
    print(f"   XGB Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
  else:
    print("\n[3/5] XGBoost not available - skipping")
  
  # Model 4: Another RF with different params
  print("\n[4/5] Training Random Forest Model 2 (Diverse)...")
  rf2 = RandomForestClassifier(
    n_estimators=700,
    max_depth=30,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='log2',
    class_weight='balanced',
    n_jobs=-1,
    random_state=123,
    verbose=0,
  )
  rf2.fit(X_train_selected, y_train)
  rf2_pred = rf2.predict(X_test_selected)
  rf2_acc = accuracy_score(y_test, rf2_pred)
  models['rf2'] = rf2
  accuracies['rf2'] = rf2_acc
  print(f"   RF2 Accuracy: {rf2_acc:.4f} ({rf2_acc*100:.2f}%)")
  
  # Model 5: Stacking Classifier (Advanced Ensemble)
  print("\n[5/5] Training Stacking Ensemble (Combining All Models)...")
  base_estimators = [
    ('rf1', rf),
    ('gb', Pipeline([('scaler', StandardScaler()), ('gb', GradientBoostingClassifier(
      n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42, verbose=0))])),
    ('rf2', rf2)
  ]
  
  if XGBOOST_AVAILABLE:
    base_estimators.append(('xgb', xgb.XGBClassifier(
      n_estimators=400, learning_rate=0.05, max_depth=7, random_state=42, verbosity=0, n_jobs=-1)))
  
  # Use GradientBoosting as final estimator for stacking
  final_estimator = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbose=0,
  )
  
  stacking = StackingClassifier(
    estimators=base_estimators,
    final_estimator=final_estimator,
    cv=5,
    n_jobs=-1,
    verbose=0
  )
  
  stacking.fit(X_train_selected, y_train)
  stacking_pred = stacking.predict(X_test_selected)
  stacking_acc = accuracy_score(y_test, stacking_pred)
  models['stacking'] = stacking
  accuracies['stacking'] = stacking_acc
  print(f"   Stacking Accuracy: {stacking_acc:.4f} ({stacking_acc*100:.2f}%)")
  
  # Also try Voting Classifier
  print("\nTraining Voting Ensemble...")
  voting_estimators = [('rf1', rf), ('rf2', rf2)]
  if XGBOOST_AVAILABLE:
    voting_estimators.append(('xgb', xgb.XGBClassifier(
      n_estimators=400, learning_rate=0.05, max_depth=7, random_state=42, verbosity=0, n_jobs=-1)))
  
  voting = VotingClassifier(
    estimators=voting_estimators,
    voting='soft',
    n_jobs=-1
  )
  voting.fit(X_train_selected, y_train)
  voting_pred = voting.predict(X_test_selected)
  voting_acc = accuracy_score(y_test, voting_pred)
  models['voting'] = voting
  accuracies['voting'] = voting_acc
  print(f"   Voting Accuracy: {voting_acc:.4f} ({voting_acc*100:.2f}%)")
  
  # Select best model
  best_model_name = max(accuracies.keys(), key=lambda k: accuracies[k])
  clf = models[best_model_name]
  best_acc = accuracies[best_model_name]
  
  print("\n" + "="*60)
  print("MODEL COMPARISON RESULTS")
  print("="*60)
  for model_name, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model_name.upper():12s}: {acc:.4f} ({acc*100:.2f}%)")
  print("="*60)
  print(f"\n🏆 BEST MODEL: {best_model_name.upper()} with accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
  print("="*60)
  
  # Store selector and scaler for later use
  clf_with_preprocessing = {
    'model': clf,
    'selector': selector,
    'scaler': scaler if best_model_name == 'gb' else None,
    'model_type': best_model_name
  }

  # Evaluate on training set first
  # Note: Models were trained on SMOTE-resampled data
  # For training accuracy, we evaluate on the SMOTE-resampled training data
  # (the data the model actually saw during training)
  print("\nEvaluating on training set (SMOTE-resampled data used for training)...")
  
  # Evaluate on the data the model was actually trained on
  if best_model_name == 'gb':
    # GB model is a Pipeline, it will handle scaling automatically
    y_train_pred = clf.predict(X_train_selected)
  else:
    # Other models use selected features directly
    y_train_pred = clf.predict(X_train_selected)
  
  train_accuracy = accuracy_score(y_train, y_train_pred)
  
  # Also evaluate on original training data for reference
  print("Evaluating on original training data (before SMOTE)...")
  X_train_orig = selector.transform(X_train)
  if best_model_name == 'gb':
    y_train_pred_orig = clf.predict(X_train_orig)
  else:
    y_train_pred_orig = clf.predict(X_train_orig)
  train_accuracy_orig = accuracy_score(y_train_original, y_train_pred_orig)
  
  print("Evaluating on test set...")
  # Test set already has preprocessing applied (X_test_selected)
  y_pred = clf.predict(X_test_selected)
  
  # Calculate and display accuracy (use test_accuracy as the main metric)
  test_accuracy = accuracy_score(y_test, y_pred)
  accuracy = test_accuracy  # Keep for backward compatibility
  print("\n" + "="*60)
  print("MODEL PERFORMANCE METRICS")
  print("="*60)
  print(f"Training Accuracy (on SMOTE data): {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
  print(f"Training Accuracy (on original):   {train_accuracy_orig:.4f} ({train_accuracy_orig*100:.2f}%)")
  print(f"Test Accuracy:                      {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
  print("="*60)

  # Precision, Recall, F1-Score (via classification report)
  print("\nDetailed Classification Report (Test Set)")
  print("(includes Precision, Recall, F1-Score and Support):")
  print(classification_report(y_test, y_pred))

  # Confusion Matrix
  cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
  print("\nConfusion Matrix (rows = true [0,1], cols = predicted [0,1]):")
  print(cm)

  print(f"\nSaving model to {MODEL_PATH} ...")
  # Save model with preprocessing components
  if best_model_name == 'gb':
    # GB already has scaler in pipeline
    joblib.dump(clf, MODEL_PATH)
  else:
    # Save model with selector
    model_package = {
      'model': clf,
      'selector': selector,
      'model_type': best_model_name
    }
    joblib.dump(model_package, MODEL_PATH)

  print(f"Saving feature columns to {FEATURES_PATH} ...")
  with open(FEATURES_PATH, "w", encoding="utf-8") as f:
    json.dump(feature_columns, f, indent=2)

  print(f"Saving model accuracy to {ACCURACY_PATH} ...")
  with open(ACCURACY_PATH, "w", encoding="utf-8") as f:
    json.dump({
      "accuracy": float(test_accuracy), 
      "accuracy_percent": float(test_accuracy * 100),
      "train_accuracy": float(train_accuracy),
      "train_accuracy_percent": float(train_accuracy * 100),
      "train_accuracy_original": float(train_accuracy_orig),
      "train_accuracy_original_percent": float(train_accuracy_orig * 100)
    }, f, indent=2)

  print("\n" + "="*50)
  print("Training completed successfully!")
  print("="*50)
  print("Done.")


if __name__ == "__main__":
  train_and_save()


