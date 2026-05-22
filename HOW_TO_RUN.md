# How to Run the Server

## ✅ Server Status

The server IS running when you see this output:
```
==================================================
MODEL ACCURACY: 0.8123 (81.23%)
==================================================
 * Serving Flask app 'server'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

**This means the server is working!** It's waiting for requests.

## 🚀 How to Use

### Option 1: Keep Server Running (Recommended)

1. **Start the server** (in Terminal 1):
   ```bash
   python server.py
   ```
   - Leave this terminal open
   - Don't close it or press Ctrl+C
   - The server will keep running

2. **Start the React frontend** (in Terminal 2):
   ```bash
   npm run dev
   ```
   - This will open http://localhost:3000
   - The React app will connect to the Flask server automatically

### Option 2: Test the Server Directly

Open your browser and visit:
- **Main page**: http://127.0.0.1:5000/
- **Accuracy API**: http://127.0.0.1:5000/api/accuracy
- **Predict API**: http://127.0.0.1:5000/api/predict (needs POST request)

### Option 3: Test with Browser Developer Tools

1. Open http://localhost:3000 (after starting React)
2. Open Browser DevTools (F12)
3. Go to Network tab
4. Fill out the form and submit
5. You'll see the API calls to `/api/predict`

## 🔍 Verify Server is Running

The server is running if:
- ✅ You see "Running on http://127.0.0.1:5000"
- ✅ You see the model accuracy displayed
- ✅ No error messages appear
- ✅ The terminal is waiting (not showing errors)

## ⚠️ Common Issues

**"Server not responding"**
- Make sure the server terminal is still open
- Check that you see "Running on http://127.0.0.1:5000"
- Try refreshing your browser

**"Connection refused"**
- The server might have stopped
- Restart it with: `python server.py`

**"404 Not Found"**
- Make sure you're using the correct URL
- Check that routes are `/api/accuracy` or `/api/predict`

## 📝 Next Steps

1. ✅ Server is running (you see the output above)
2. ⏭️ Start React frontend: `npm run dev`
3. ⏭️ Open browser: http://localhost:3000
4. ⏭️ Test the risk assessment form!

