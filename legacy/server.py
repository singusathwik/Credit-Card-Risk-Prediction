import json
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, send_from_directory, request, jsonify

# Try to import CORS, make it optional
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("Note: flask-cors not installed. Install with: pip install flask-cors")

MODEL_PATH = Path("model.pkl")
FEATURES_PATH = Path("feature_columns.json")
ACCURACY_PATH = Path("model_accuracy.json")

app = Flask(__name__, static_folder=".", static_url_path="")

# Enable CORS if available
if CORS_AVAILABLE:
    CORS(app)
else:
    # Manual CORS headers as fallback
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response


def load_model_and_features():
  if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    return None, None, None

  model_data = joblib.load(MODEL_PATH)
  
  # Handle both old format (just model) and new format (dict with model and preprocessing)
  if isinstance(model_data, dict):
    model = model_data.get('model')
    selector = model_data.get('selector')
    model_type = model_data.get('model_type', 'unknown')
  else:
    model = model_data
    selector = None
    model_type = 'legacy'
  
  with FEATURES_PATH.open("r", encoding="utf-8") as f:
    feature_columns = json.load(f)
  
  return model, feature_columns, selector


model, feature_columns, selector = load_model_and_features()

# Load and display model accuracy if available
if model is not None:
  try:
    with ACCURACY_PATH.open("r", encoding="utf-8") as f:
      accuracy_data = json.load(f)
      accuracy = accuracy_data.get("accuracy", 0)
      print("="*50)
      print(f"MODEL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
      print("="*50)
  except FileNotFoundError:
    print("Note: Model accuracy file not found.")


@app.route("/")
def index():
  # Serve the main frontend page
  return send_from_directory(".", "index.html")


@app.route("/accuracy", methods=["GET"])
@app.route("/api/accuracy", methods=["GET"])  # Also support /api/ prefix
def get_accuracy():
  """
  Get the model's accuracy score.
  """
  try:
    if ACCURACY_PATH.exists():
      with ACCURACY_PATH.open("r", encoding="utf-8") as f:
        accuracy_data = json.load(f)
      return jsonify(accuracy_data)
    else:
      return jsonify({"error": "Accuracy file not found. Run train_model.py first."}), 404
  except Exception as e:
    return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])  # Also support /api/ prefix
def predict():
  """
  Real prediction endpoint using trained model (if available).
  """
  global model, feature_columns, selector

  if model is None or feature_columns is None:
    return jsonify({"error": "Model not trained yet. Run train_model.py first."}), 500

  data = request.get_json(force=True) or {}

  # Build feature vector in the exact column order used during training
  features = []
  for col in feature_columns:
    val = data.get(col, 0)
    try:
      val = float(val)
    except (TypeError, ValueError):
      val = 0.0
    features.append(val)

  X = np.array(features, dtype=np.float32).reshape(1, -1)
  
  # Apply feature selection if available
  if selector is not None:
    X = selector.transform(X)

  # Predict probability of default (class 1)
  # Handle both regular models and pipelines
  if hasattr(model, 'predict_proba'):
    probs = model.predict_proba(X)[0]
  else:
    # Fallback for models without predict_proba
    pred = model.predict(X)[0]
    probs = [1 - pred, pred]  # Approximate probabilities
    
  prob_default = float(probs[1])
  will_default = prob_default >= 0.5

  # Include accuracy in response if available
  response = {
      "default_probability": prob_default,
      "will_default": will_default,
    }
  
  # Try to include accuracy
  try:
    if ACCURACY_PATH.exists():
      with ACCURACY_PATH.open("r", encoding="utf-8") as f:
        accuracy_data = json.load(f)
        response["model_accuracy"] = accuracy_data.get("accuracy", None)
  except:
    pass  # Don't fail if accuracy can't be loaded
  
  return jsonify(response)


if __name__ == "__main__":
  app.run(host="127.0.0.1", port=5000, debug=True)



