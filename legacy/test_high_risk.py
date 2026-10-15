"""
Test script to demonstrate high-risk credit card input
"""
import requests
import json

def test_high_risk_input():
    """
    Example of high-risk input that should result in high default probability.
    
    High-risk characteristics:
    1. Very high credit utilization (bills near credit limit)
    2. Multiple months of late payments (2-3 months overdue)
    3. Low payment amounts relative to bills
    4. Increasing bill amounts over time
    5. High outstanding debt
    """
    
    base_url = "http://127.0.0.1:5000"
    
    # High-risk input example
    high_risk_data = {
        "LIMIT_BAL": 200000,      # Credit limit: 200,000
        "SEX": 1,                 # 1 = Male, 2 = Female
        "EDUCATION": 2,           # Education level
        "MARRIAGE": 1,            # 1 = Married, 2 = Single, 3 = Others
        "AGE": 28,                # Age
        "PAY_0": 2,               # 2 months overdue (current month) - SEVERE
        "PAY_2": 3,               # 3 months overdue - VERY SEVERE
        "PAY_3": 2,               # 2 months overdue
        "PAY_4": 1,               # 1 month overdue
        "PAY_5": 0,               # Paid on time
        "PAY_6": 0,               # Paid on time
        "BILL_AMT1": 195000,      # Recent bill: 195k (97.5% of limit!)
        "BILL_AMT2": 190000,      # 190k (95% of limit)
        "BILL_AMT3": 185000,      # 185k (92.5% of limit)
        "BILL_AMT4": 180000,      # 180k (90% of limit)
        "BILL_AMT5": 175000,      # 175k (87.5% of limit)
        "BILL_AMT6": 170000,      # 170k (85% of limit)
        "PAY_AMT1": 5000,         # Only paying 5k (vs 195k bill!)
        "PAY_AMT2": 3000,         # Only paying 3k (vs 190k bill!)
        "PAY_AMT3": 2000,         # Only paying 2k (vs 185k bill!)
        "PAY_AMT4": 1000,         # Only paying 1k (vs 180k bill!)
        "PAY_AMT5": 5000,         # Paying 5k (vs 175k bill)
        "PAY_AMT6": 5000,         # Paying 5k (vs 170k bill)
    }
    
    print("=" * 60)
    print("HIGH RISK CREDIT CARD INPUT TEST")
    print("=" * 60)
    print("\nRisk Indicators:")
    print("  • Credit Utilization: 85-97% (VERY HIGH)")
    print("  • Payment Delays: 2-3 months overdue (SEVERE)")
    print("  • Payment Coverage: Only 1-3% of bills being paid")
    print("  • Debt Trend: Increasing bills over time")
    print("  • Outstanding Balance: ~190k+ (near credit limit)")
    print("\n" + "-" * 60)
    
    try:
        response = requests.post(
            f"{base_url}/api/predict",
            json=high_risk_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            default_prob = result.get("default_probability", 0)
            will_default = result.get("will_default", False)
            accuracy = result.get("model_accuracy", "N/A")
            
            print("\nPREDICTION RESULTS:")
            print(f"  Default Probability: {default_prob:.2%}")
            print(f"  Will Default: {will_default}")
            print(f"  Model Accuracy: {accuracy}")
            print("\n" + "-" * 60)
            
            if default_prob >= 0.7:
                print("✓ HIGH RISK CONFIRMED - Default probability >= 70%")
            elif default_prob >= 0.5:
                print("⚠ MEDIUM-HIGH RISK - Default probability >= 50%")
            else:
                print("⚠ Lower risk than expected - may need to adjust input")
                
        else:
            print(f"✗ Error: Server returned status {response.status_code}")
            print(f"  Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Server is not running!")
        print("  Please start the server with: python server.py")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("=" * 60)


if __name__ == "__main__":
    test_high_risk_input()

