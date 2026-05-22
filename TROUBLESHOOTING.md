# Troubleshooting Guide

## If you see a white screen or "not working":

### Step 1: Check Both Servers Are Running

**Terminal 1 - Flask Backend:**
```bash
python server.py
```
Should show:
```
* Running on http://127.0.0.1:5000
```

**Terminal 2 - React Frontend:**
```bash
npm run dev
```
Should show:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:3000/
```

### Step 2: Open the Correct URL

✅ **CORRECT:** http://localhost:3000
❌ **WRONG:** http://127.0.0.1:5000 (this is just the backend)

### Step 3: Check Browser Console

1. Open http://localhost:3000
2. Press F12 to open Developer Tools
3. Go to "Console" tab
4. Look for any red error messages
5. Share the errors you see

### Step 4: Common Issues

**Issue: "Failed to fetch" or CORS error**
- Solution: Make sure Flask server is running
- Check: http://127.0.0.1:5000/api/accuracy in browser

**Issue: "Cannot find module"**
- Solution: Run `npm install` again

**Issue: White screen with no errors**
- Solution: Check if React components are loading
- Open Network tab in DevTools and check for failed requests

**Issue: Port 3000 already in use**
- Solution: Kill the process or change port in `vite.config.js`

### Step 5: Test API Directly

Open these URLs in your browser:
- http://127.0.0.1:5000/api/accuracy (should show JSON)
- http://localhost:3000 (should show the React app)

### Quick Fix Commands

```bash
# Stop all servers (Ctrl+C in each terminal)

# Restart Flask
python server.py

# Restart React (in new terminal)
npm run dev

# If React doesn't work, reinstall
npm install
npm run dev
```

## Still Not Working?

Share:
1. What URL you're accessing
2. What you see (white screen, error message, etc.)
3. Any errors from browser console (F12)
4. Any errors from the terminal where servers are running

