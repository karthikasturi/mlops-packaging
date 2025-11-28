"""
Model drift detection for breast cancer prediction model
"""
import pickle
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

def load_model(model_path='../model/model.pkl'):
    """Load the trained model"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def check_data_drift(X_train, X_new):
    """
    Simple data drift detection using statistical measures
    Compares feature distributions between training and new data
    """
    drift_detected = False
    drift_features = []
    
    for i in range(X_train.shape[1]):
        # Calculate mean and std for each feature
        train_mean = np.mean(X_train[:, i])
        train_std = np.std(X_train[:, i])
        
        new_mean = np.mean(X_new[:, i])
        new_std = np.std(X_new[:, i])
        
        # Check if new data mean is outside 2 standard deviations
        if abs(new_mean - train_mean) > 2 * train_std:
            drift_detected = True
            drift_features.append({
                'feature_index': i,
                'train_mean': train_mean,
                'new_mean': new_mean,
                'difference': abs(new_mean - train_mean)
            })
    
    return drift_detected, drift_features

def check_model_performance_drift(model, X_test, y_test, baseline_accuracy, threshold=0.05):
    """
    Check if model performance has degraded
    threshold: acceptable drop in accuracy (default 5%)
    """
    current_accuracy = accuracy_score(y_test, model.predict(X_test))
    accuracy_drop = baseline_accuracy - current_accuracy
    
    performance_drift = accuracy_drop > threshold
    
    return performance_drift, current_accuracy, accuracy_drop

def simulate_drift_detection():
    """
    Simulate drift detection on new data
    """
    print("="*60)
    print("MODEL DRIFT DETECTION SIMULATION")
    print("="*60)
    
    # Load model
    print("\n1. Loading trained model...")
    model = load_model()
    
    # Load original dataset (simulate training data)
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Calculate baseline accuracy
    baseline_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"   Baseline Accuracy: {baseline_accuracy:.4f}")
    
    # Simulate new data with slight drift (add noise to features)
    print("\n2. Simulating new data with potential drift...")
    X_new = X_test.copy()
    # Add random noise to simulate drift
    noise_level = 0.1
    X_new = X_new + np.random.normal(0, noise_level * X_new.std(axis=0), X_new.shape)
    
    # Check for data drift
    print("\n3. Checking for data drift...")
    data_drift, drift_features = check_data_drift(X_train, X_new)
    
    if data_drift:
        print(f"   ⚠️  DATA DRIFT DETECTED!")
        print(f"   Number of features with drift: {len(drift_features)}")
        print(f"   Features with significant drift (top 3):")
        for feat in drift_features[:3]:
            print(f"      - Feature {feat['feature_index']}: "
                  f"train_mean={feat['train_mean']:.2f}, "
                  f"new_mean={feat['new_mean']:.2f}")
    else:
        print("   ✓ No significant data drift detected")
    
    # Check for performance drift
    print("\n4. Checking for performance drift...")
    perf_drift, current_accuracy, accuracy_drop = check_model_performance_drift(
        model, X_new, y_test, baseline_accuracy
    )
    
    print(f"   Current Accuracy: {current_accuracy:.4f}")
    print(f"   Accuracy Drop: {accuracy_drop:.4f} ({accuracy_drop*100:.2f}%)")
    
    if perf_drift:
        print(f"   ⚠️  PERFORMANCE DRIFT DETECTED!")
        print(f"   Model performance has degraded by {accuracy_drop*100:.2f}%")
    else:
        print("   ✓ Model performance is stable")
    
    # Decision
    print("\n5. Retraining Decision:")
    if data_drift or perf_drift:
        print("   🔄 RETRAINING RECOMMENDED")
        print("   Reason:", end=" ")
        if data_drift and perf_drift:
            print("Both data drift and performance drift detected")
        elif data_drift:
            print("Data drift detected")
        else:
            print("Performance drift detected")
        return True
    else:
        print("   ✓ Model is performing well, no retraining needed")
        return False

def save_drift_report(drift_detected, timestamp=None):
    """Save drift detection report"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        'timestamp': timestamp,
        'drift_detected': drift_detected,
        'recommendation': 'Retrain model' if drift_detected else 'Continue monitoring'
    }
    
    filename = f'drift_report_{timestamp}.txt'
    with open(filename, 'w') as f:
        f.write(f"Drift Detection Report - {timestamp}\n")
        f.write("="*60 + "\n")
        f.write(f"Drift Detected: {drift_detected}\n")
        f.write(f"Recommendation: {report['recommendation']}\n")
    
    print(f"\n   Report saved to: {filename}")
    return filename

if __name__ == "__main__":
    retrain_needed = simulate_drift_detection()
    save_drift_report(retrain_needed)
    
    print("\n" + "="*60)
    if retrain_needed:
        print("Next steps: Run 'python continuous_training.py' to retrain")
    print("="*60)
