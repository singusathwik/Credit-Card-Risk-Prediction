"""
Quick test script to verify the Flask server is working
"""
import requests
import json

def test_server():
    base_url = "http://127.0.0.1:5000"
    
    print("Testing Flask Server...")
    print("=" * 50)
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"✓ Server is running! Status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("✗ Server is NOT running. Please start it with: python server.py")
        return
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Test 2: Check accuracy endpoint
    try:
        response = requests.get(f"{base_url}/api/accuracy", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Accuracy endpoint works!")
            print(f"  Model Accuracy: {data.get('accuracy', 'N/A')}")
        else:
            print(f"⚠ Accuracy endpoint returned: {response.status_code}")
    except Exception as e:
        print(f"⚠ Accuracy endpoint error: {e}")
    
    # Test 3: Test predict endpoint with sample data
    try:
        sample_data = {
            "LIMIT_BAL": 200000,
            "SEX": 1,
            "EDUCATION": 2,
            "MARRIAGE": 1,
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
        
        response = requests.post(
            f"{base_url}/api/predict",
            json=sample_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Predict endpoint works!")
            print(f"  Default Probability: {data.get('default_probability', 0):.2%}")
            print(f"  Will Default: {data.get('will_default', False)}")
        else:
            print(f"⚠ Predict endpoint returned: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"⚠ Predict endpoint error: {e}")
    
    print("=" * 50)
    print("Server test complete!")

if __name__ == "__main__":
    test_server()

