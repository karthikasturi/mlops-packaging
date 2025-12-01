"""
Data Drift Detection with MLflow Tracking

This module demonstrates:
1. Comparing new data against baseline
2. Detecting distribution shifts
3. Logging drift metrics to MLflow
4. Alerting when retraining is needed
"""

import pandas as pd
import numpy as np
import mlflow
from scipy import stats
from datetime import datetime
import json


class DataDriftDetector:
    """
    Detects data drift and logs results to MLflow
    """
    
    def __init__(self, experiment_name="house-rental-drift-detection"):
        """
        Initialize drift detection
        
        Args:
            experiment_name: MLflow experiment name
        """
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri("http://localhost:5001")
        mlflow.set_experiment(experiment_name)
        
    def calculate_distribution_drift(self, baseline_data, current_data, feature):
        """
        Calculate drift using Kolmogorov-Smirnov test
        
        Args:
            baseline_data: Historical data
            current_data: Current data
            feature: Feature name to test
            
        Returns:
            Dictionary with drift metrics
        """
        baseline_values = baseline_data[feature].dropna()
        current_values = current_data[feature].dropna()
        
        # KS test
        ks_statistic, ks_pvalue = stats.ks_2samp(baseline_values, current_values)
        
        # Mean and std comparison
        baseline_mean = baseline_values.mean()
        current_mean = current_values.mean()
        mean_shift = ((current_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0
        
        baseline_std = baseline_values.std()
        current_std = current_values.std()
        std_shift = ((current_std - baseline_std) / baseline_std) * 100 if baseline_std != 0 else 0
        
        # Drift detected if p-value < 0.05
        drift_detected = ks_pvalue < 0.05
        
        return {
            'feature': feature,
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'drift_detected': drift_detected,
            'baseline_mean': baseline_mean,
            'current_mean': current_mean,
            'mean_shift_percent': mean_shift,
            'baseline_std': baseline_std,
            'current_std': current_std,
            'std_shift_percent': std_shift
        }
    
    def detect_drift(self, baseline_data, current_data, numerical_features):
        """
        Detect drift across multiple features
        
        Args:
            baseline_data: Historical DataFrame
            current_data: Current DataFrame
            numerical_features: List of numerical features to check
            
        Returns:
            Dictionary with drift results
        """
        drift_results = {}
        features_with_drift = []
        
        for feature in numerical_features:
            if feature in baseline_data.columns and feature in current_data.columns:
                result = self.calculate_distribution_drift(
                    baseline_data, 
                    current_data, 
                    feature
                )
                drift_results[feature] = result
                
                if result['drift_detected']:
                    features_with_drift.append(feature)
        
        return {
            'drift_results': drift_results,
            'features_with_drift': features_with_drift,
            'drift_percentage': (len(features_with_drift) / len(numerical_features)) * 100,
            'total_features_checked': len(numerical_features)
        }
    
    def log_drift_to_mlflow(self, drift_analysis, baseline_version, current_version):
        """
        Log drift detection results to MLflow
        
        Args:
            drift_analysis: Dictionary with drift results
            baseline_version: Baseline data version
            current_version: Current data version
        """
        with mlflow.start_run(run_name=f"drift_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log versions being compared
            mlflow.log_param("baseline_version", baseline_version)
            mlflow.log_param("current_version", current_version)
            mlflow.log_param("detection_timestamp", datetime.now().isoformat())
            
            # Log overall drift metrics
            mlflow.log_metric("drift_percentage", drift_analysis['drift_percentage'])
            mlflow.log_metric("total_features_checked", drift_analysis['total_features_checked'])
            mlflow.log_metric("features_with_drift_count", len(drift_analysis['features_with_drift']))
            
            # Log individual feature drift metrics
            for feature, result in drift_analysis['drift_results'].items():
                mlflow.log_metric(f"{feature}_ks_statistic", result['ks_statistic'])
                mlflow.log_metric(f"{feature}_ks_pvalue", result['ks_pvalue'])
                mlflow.log_metric(f"{feature}_mean_shift_pct", result['mean_shift_percent'])
                mlflow.log_metric(f"{feature}_std_shift_pct", result['std_shift_percent'])
                mlflow.log_param(f"{feature}_drift_detected", str(result['drift_detected']))
            
            # Set tags
            mlflow.set_tag("drift_status", "detected" if drift_analysis['features_with_drift'] else "no_drift")
            mlflow.set_tag("retraining_needed", "yes" if drift_analysis['drift_percentage'] > 20 else "no")
            
            # Log drift summary as artifact
            drift_summary = {
                'timestamp': datetime.now().isoformat(),
                'baseline_version': baseline_version,
                'current_version': current_version,
                'drift_percentage': drift_analysis['drift_percentage'],
                'features_with_drift': drift_analysis['features_with_drift'],
                'detailed_results': {
                    k: {
                        'drift_detected': v['drift_detected'],
                        'ks_pvalue': v['ks_pvalue'],
                        'mean_shift_percent': v['mean_shift_percent']
                    }
                    for k, v in drift_analysis['drift_results'].items()
                }
            }
            
            drift_summary_path = 'monitoring/drift_summary.json'
            with open(drift_summary_path, 'w') as f:
                json.dump(drift_summary, f, indent=2)
            
            mlflow.log_artifact(drift_summary_path, artifact_path="drift_reports")
            
            print(f"✓ Drift analysis logged to MLflow")
            print(f"  Run ID: {mlflow.active_run().info.run_id}")


def demonstrate_drift_detection():
    """
    Demonstrate drift detection with synthetic data
    """
    print("\n" + "=" * 70)
    print("DATA DRIFT DETECTION WITH MLFLOW")
    print("=" * 70)
    
    # Generate baseline data (original distribution)
    print("\n📊 Generating baseline dataset...")
    np.random.seed(42)
    n_samples = 1000
    
    baseline_data = pd.DataFrame({
        'area_sqft': np.random.normal(1500, 500, n_samples),
        'bedrooms': np.random.poisson(2.5, n_samples),
        'bathrooms': np.random.poisson(2, n_samples),
        'age_years': np.random.exponential(10, n_samples),
        'rent': np.random.normal(2000, 600, n_samples)
    })
    
    print(f"✓ Baseline data shape: {baseline_data.shape}")
    print(f"  Average rent: ${baseline_data['rent'].mean():.2f}")
    print(f"  Average area: {baseline_data['area_sqft'].mean():.2f} sqft")
    
    # Generate current data with drift
    print("\n📊 Generating current dataset (with drift)...")
    np.random.seed(123)
    
    current_data = pd.DataFrame({
        'area_sqft': np.random.normal(1600, 550, n_samples),  # Slight shift
        'bedrooms': np.random.poisson(2.7, n_samples),  # Slight increase
        'bathrooms': np.random.poisson(2.1, n_samples),  # Slight increase
        'age_years': np.random.exponential(12, n_samples),  # Properties getting older
        'rent': np.random.normal(2300, 650, n_samples)  # Rent increased significantly
    })
    
    print(f"✓ Current data shape: {current_data.shape}")
    print(f"  Average rent: ${current_data['rent'].mean():.2f}")
    print(f"  Average area: {current_data['area_sqft'].mean():.2f} sqft")
    
    # Detect drift
    print("\n" + "=" * 70)
    print("🔍 DETECTING DATA DRIFT")
    print("=" * 70)
    
    detector = DataDriftDetector()
    numerical_features = ['area_sqft', 'bedrooms', 'bathrooms', 'age_years', 'rent']
    
    drift_analysis = detector.detect_drift(
        baseline_data,
        current_data,
        numerical_features
    )
    
    # Display results
    print(f"\n📈 DRIFT ANALYSIS RESULTS:")
    print(f"  Features checked: {drift_analysis['total_features_checked']}")
    print(f"  Features with drift: {len(drift_analysis['features_with_drift'])}")
    print(f"  Drift percentage: {drift_analysis['drift_percentage']:.1f}%")
    
    print("\n📊 DETAILED FEATURE ANALYSIS:")
    print("-" * 70)
    
    for feature, result in drift_analysis['drift_results'].items():
        status = "⚠️  DRIFT DETECTED" if result['drift_detected'] else "✓ No drift"
        print(f"\n{feature.upper()}")
        print(f"  Status: {status}")
        print(f"  KS Statistic: {result['ks_statistic']:.4f}")
        print(f"  P-value: {result['ks_pvalue']:.4f}")
        print(f"  Mean shift: {result['mean_shift_percent']:+.2f}%")
        print(f"  Std shift: {result['std_shift_percent']:+.2f}%")
        print(f"  Baseline mean: {result['baseline_mean']:.2f}")
        print(f"  Current mean: {result['current_mean']:.2f}")
    
    # Log to MLflow
    print("\n" + "=" * 70)
    print("📝 LOGGING TO MLFLOW")
    print("=" * 70)
    
    detector.log_drift_to_mlflow(
        drift_analysis,
        baseline_version="v20251201_baseline",
        current_version="v20251201_current"
    )
    
    # Recommendations
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    
    if drift_analysis['drift_percentage'] > 30:
        print("⚠️  CRITICAL: Significant drift detected (>30%)")
        print("   → Model retraining is HIGHLY RECOMMENDED")
        print("   → Review data collection process")
        print("   → Investigate root causes of drift")
    elif drift_analysis['drift_percentage'] > 20:
        print("⚠️  WARNING: Moderate drift detected (>20%)")
        print("   → Consider model retraining")
        print("   → Monitor model performance closely")
    elif drift_analysis['drift_percentage'] > 0:
        print("ℹ️  INFO: Minor drift detected")
        print("   → Continue monitoring")
        print("   → Retraining may not be necessary yet")
    else:
        print("✅ No significant drift detected")
        print("   → Current model should perform well")
    
    print("\n" + "=" * 70)
    print("✅ DRIFT DETECTION COMPLETE")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("  1. Review drift details in MLflow UI: mlflow ui")
    print("  2. If drift is significant, retrain the model")
    print("  3. Set up automated drift monitoring")
    print("  4. Update baseline data periodically")


if __name__ == "__main__":
    demonstrate_drift_detection()
