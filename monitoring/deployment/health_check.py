#!/usr/bin/env python3
"""
Health Check Script for Model Server
"""

import requests
import time
import sys

def check_health(url="http://localhost:5002/health", timeout=5):
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ Model server is healthy")
            return True
        else:
            print(f"⚠️  Model server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to model server at {url}")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Health check timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def check_prediction(url="http://localhost:5002/invocations"):
    """Test prediction endpoint"""
    sample_data = {
        "dataframe_split": {
            "columns": [
                "area_sqft", "bedrooms", "bathrooms", "parking", "age_years",
                "floor", "has_gym", "has_pool", "price_per_sqft", "room_bath_ratio",
                "total_rooms", "amenities_score", "is_new", "is_spacious"
            ],
            "data": [[1500.0, 2.0, 1.5, 1.0, 5.0, 3.0, 1.0, 0.0, 1.2, 1.33, 3.5, 2.0, 1.0, 1.0]]
        }
    }
    
    try:
        response = requests.post(
            url,
            json=sample_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            prediction = response.json()
            print(f"✅ Prediction successful: {prediction}")
            return True
        else:
            print(f"⚠️  Prediction failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Model Server Health Check\n")
    
    # Check health endpoint
    health_ok = check_health()
    
    if health_ok:
        # Check prediction endpoint
        print("\nTesting prediction endpoint...")
        pred_ok = check_prediction()
        
        if pred_ok:
            print("\n✅ All checks passed!")
            sys.exit(0)
        else:
            print("\n⚠️  Prediction check failed")
            sys.exit(1)
    else:
        print("\n❌ Health check failed")
        sys.exit(1)
