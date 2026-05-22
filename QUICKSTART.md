# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

**Python:**
```bash
pip install flask flask-cors scikit-learn pandas numpy joblib openpyxl
```

**Node.js:**
```bash
npm install
```

### Step 2: Train the Model

```bash
python train_model.py
```

Wait for training to complete. This may take a few minutes.

### Step 3: Run the Application

**Terminal 1 - Start Backend:**
```bash
python server.py
```

**Terminal 2 - Start Frontend:**
```bash
npm run dev
```

## 🎉 You're Done!

- Frontend: http://localhost:3000
- Backend API: http://127.0.0.1:5000

## 📝 Optional: For Better Accuracy

Install advanced libraries:
```bash
pip install xgboost imbalanced-learn
```

Then retrain the model for potentially higher accuracy.

## 🐛 Troubleshooting

**Port already in use?**
- Change port in `vite.config.js` (frontend)
- Change port in `server.py` (backend)

**Model not found?**
- Make sure you ran `python train_model.py` first
- Check that `model.pkl` exists in the project directory

**CORS errors?**
- Make sure `flask-cors` is installed: `pip install flask-cors`

