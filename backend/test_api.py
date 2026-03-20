"""
Test script for ExoHabitAI API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:5000/api"

# Sample exoplanet data
sample_data = {
    "mass_earth": 1.0,
    "semimajor_axis": 1.0,
    "star_temp_k": 5778,
    "star_luminosity": 0.0,
    "star_metallicity": 0.0,
    "log_stellar_flux": 0.0,
    "log_surface_gravity": 0.69,
    "bulk_density_gcc": 5.51
}

def test_health():
    """Test health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_binary_prediction():
    """Test binary classification"""
    print("\n=== Testing Binary Prediction ===")
    response = requests.post(
        f"{BASE_URL}/predict/binary",
        json=sample_data
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_multiclass_prediction():
    """Test multi-class classification"""
    print("\n=== Testing Multi-Class Prediction ===")
    response = requests.post(
        f"{BASE_URL}/predict/multiclass",
        json=sample_data
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_both_predictions():
    """Test combined prediction"""
    print("\n=== Testing Both Predictions ===")
    response = requests.post(
        f"{BASE_URL}/predict/both",
        json=sample_data
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_features():
    """Test features endpoint"""
    print("\n=== Testing Features Endpoint ===")
    response = requests.get(f"{BASE_URL}/features")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("ExoHabitAI API Test Suite")
    print("=" * 50)
    
    try:
        test_health()
        test_features()
        test_binary_prediction()
        test_multiclass_prediction()
        test_both_predictions()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
    
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API")
        print("Make sure the Flask server is running:")
        print("  python backend/app.py")
    except Exception as e:
        print(f"\nERROR: {e}")
