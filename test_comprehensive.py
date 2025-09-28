import requests
import json
import numpy as np
from datetime import datetime, timedelta

API_BASE = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("=== TESTING HEALTH ENDPOINT ===")
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_prediction_valid_data():
    """Test prediction with valid data"""
    print("\n=== TESTING PREDICTION WITH VALID DATA ===")

    # Create realistic test features (matching training data structure)
    test_data = {
        "temperature": 72.5,
        "humidity": 65.0,
        "wind_speed": 8.2,
        "pressure": 1013.2,
        "hour": 14,
        "day_of_week": 2,
        "month": 6,
        "is_weekend": False,
        "is_holiday": False
    }

    print(f"Input data: {json.dumps(test_data, indent=2)}")

    response = requests.post(f"{API_BASE}/predict", json=test_data)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")

        # Validate response structure
        required_keys = ["quantiles", "pred", "num_features"]
        for key in required_keys:
            if key not in result:
                print(f"ERROR: Missing key '{key}' in response")
                return False

        # Validate predictions are reasonable (typical load: 40-80 GW)
        predictions = result["pred"]
        if len(predictions) != 3:
            print(f"ERROR: Expected 3 quantile predictions, got {len(predictions)}")
            return False

        p10, p50, p90 = predictions
        print(f"Predictions - P10: {p10:.1f} MW, P50: {p50:.1f} MW, P90: {p90:.1f} MW")

        # Check if predictions are ordered correctly
        if not (p10 <= p50 <= p90):
            print(f"ERROR: Quantiles not properly ordered: P10={p10:.1f}, P50={p50:.1f}, P90={p90:.1f}")
            return False

        # Check if predictions are in reasonable range (30-100 GW)
        if not (30000 <= p50 <= 100000):
            print(f"WARNING: P50 prediction {p50:.1f} MW seems unrealistic")

        print("SUCCESS: Valid prediction received")
        return True
    else:
        print(f"ERROR: Request failed with status {response.status_code}")
        print(f"Error: {response.text}")
        return False

def test_prediction_edge_cases():
    """Test prediction with edge case data"""
    print("\n=== TESTING EDGE CASES ===")

    test_cases = [
        {
            "name": "Extreme cold",
            "data": {
                "temperature": -20.0,
                "humidity": 90.0,
                "wind_speed": 25.0,
                "pressure": 950.0,
                "hour": 18,
                "day_of_week": 0,
                "month": 1,
                "is_weekend": False,
                "is_holiday": False
            }
        },
        {
            "name": "Extreme heat",
            "data": {
                "temperature": 110.0,
                "humidity": 20.0,
                "wind_speed": 2.0,
                "pressure": 1050.0,
                "hour": 15,
                "day_of_week": 5,
                "month": 7,
                "is_weekend": True,
                "is_holiday": False
            }
        },
        {
            "name": "Holiday weekend",
            "data": {
                "temperature": 75.0,
                "humidity": 55.0,
                "wind_speed": 10.0,
                "pressure": 1015.0,
                "hour": 12,
                "day_of_week": 6,
                "month": 12,
                "is_weekend": True,
                "is_holiday": True
            }
        }
    ]

    all_passed = True
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        response = requests.post(f"{API_BASE}/predict", json=test_case['data'])

        if response.status_code == 200:
            result = response.json()
            predictions = result["pred"]
            p10, p50, p90 = predictions
            print(f"  Result - P10: {p10:.1f}, P50: {p50:.1f}, P90: {p90:.1f} MW")

            if not (p10 <= p50 <= p90):
                print(f"  ERROR: Quantiles not ordered properly")
                all_passed = False
            else:
                print(f"  SUCCESS: Valid prediction")
        else:
            print(f"  ERROR: Request failed with status {response.status_code}")
            all_passed = False

    return all_passed

def test_invalid_data():
    """Test API error handling"""
    print("\n=== TESTING ERROR HANDLING ===")

    invalid_cases = [
        {
            "name": "Missing required field",
            "data": {
                "temperature": 70.0,
                "humidity": 60.0
                # Missing other required fields
            }
        },
        {
            "name": "Invalid data types",
            "data": {
                "temperature": "not_a_number",
                "humidity": 60.0,
                "wind_speed": 8.0,
                "pressure": 1013.0,
                "hour": 14,
                "day_of_week": 2,
                "month": 6,
                "is_weekend": False,
                "is_holiday": False
            }
        },
        {
            "name": "Empty request",
            "data": {}
        }
    ]

    error_handling_works = True
    for test_case in invalid_cases:
        print(f"\nTesting: {test_case['name']}")
        response = requests.post(f"{API_BASE}/predict", json=test_case['data'])

        if response.status_code != 200:
            print(f"  SUCCESS: Properly rejected with status {response.status_code}")
        else:
            print(f"  ERROR: Should have rejected invalid data")
            error_handling_works = False

    return error_handling_works

def main():
    print("PowerScope API Comprehensive Testing")
    print("=" * 50)

    tests = [
        ("Health Check", test_health),
        ("Valid Predictions", test_prediction_valid_data),
        ("Edge Cases", test_prediction_edge_cases),
        ("Error Handling", test_invalid_data)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()