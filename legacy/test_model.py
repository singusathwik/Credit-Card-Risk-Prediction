import json

import joblib
import numpy as np


MODEL_PATH = "model.pkl"
FEATURES_PATH = "feature_columns.json"
ACCURACY_PATH = "model_accuracy.json"


def load_model_and_features():
  model = joblib.load(MODEL_PATH)
  with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    feature_columns = json.load(f)
  return model, feature_columns


def build_sample(feature_columns):
  """
  Build a reasonable example input.
  You can edit these values freely to test different scenarios.
  """
  sample = {
    "LIMIT_BAL": 200000,
    "SEX": 1,  # 1 = male, 2 = female
    "EDUCATION": 2,  # 1=grad,2=univ,3=high school,4=others
    "MARRIAGE": 1,  # 1=married,2=single,3=others
    "AGE": 35,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 50000,
    "BILL_AMT2": 48000,
    "BILL_AMT3": 47000,
    "BILL_AMT4": 46000,
    "BILL_AMT5": 45000,
    "BILL_AMT6": 44000,
    "PAY_AMT1": 50000,
    "PAY_AMT2": 50000,
    "PAY_AMT3": 50000,
    "PAY_AMT4": 50000,
    "PAY_AMT5": 50000,
    "PAY_AMT6": 50000,
  }

  x = np.array(
    [float(sample.get(col, 0.0)) for col in feature_columns],
    dtype=np.float32,
  ).reshape(1, -1)

  return x, sample


def main():
  model, feature_columns = load_model_and_features()
  
  # Load and display model accuracy if available
  try:
    with open(ACCURACY_PATH, "r", encoding="utf-8") as f:
      accuracy_data = json.load(f)
      accuracy = accuracy_data.get("accuracy", 0)
      print("="*50)
      print(f"MODEL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
      print("="*50)
      print()
  except FileNotFoundError:
    print("Note: Model accuracy not found. Run train_model.py first to see accuracy.\n")
  
  x, sample = build_sample(feature_columns)

  prob_default = float(model.predict_proba(x)[0, 1])
  will_default = prob_default >= 0.5

  print("Input sample:")
  for k, v in sample.items():
    print(f"  {k}: {v}")

  print("\nModel output:")
  print(f"  Default probability: {prob_default:.4f}")
  print(f"  Will default (threshold 0.5): {will_default}")


if __name__ == "__main__":
  main()


