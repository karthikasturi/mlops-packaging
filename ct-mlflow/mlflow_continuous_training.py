"""
Continuous Training with MLflow Tracking
Demonstrates drift detection and retraining with experiment tracking
"""
import pickle
import os
import mlflow
import mlflow.sklearn
import numpy as np
from datetime import datetime
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# MLflow Configuration
MLFLOW_TRACKING_URI = "http://localhost:5001"
EXPERIMENT_NAME = "breast_cancer_continuous_training"

class MLflowContinuousTraining:
    def __init__(self, model_dir='../model'):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, 'model.pkl')
        
        # Configure MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
        print(f"Experiment: {EXPERIMENT_NAME}")
    
    def load_current_model(self):
        """Load the current production model"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def detect_data_drift(self, X_train, X_new, threshold=2.0):
        """
        Detect data drift using statistical comparison
        Returns drift score and whether drift is detected
        """
        drift_scores = []
        
        for i in range(X_train.shape[1]):
            train_mean = np.mean(X_train[:, i])
            train_std = np.std(X_train[:, i])
            new_mean = np.mean(X_new[:, i])
            
            if train_std > 0:
                drift_score = abs(new_mean - train_mean) / train_std
                drift_scores.append(drift_score)
        
        avg_drift_score = np.mean(drift_scores)
        drift_detected = avg_drift_score > threshold
        
        return drift_detected, avg_drift_score, drift_scores
    
    def detect_performance_drift(self, model, X_test, y_test, baseline_accuracy, threshold=0.05):
        """
        Detect performance drift
        Returns whether drift is detected and current accuracy
        """
        if model is None:
            return True, 0.0  # No model exists, need to train
        
        current_accuracy = accuracy_score(y_test, model.predict(X_test))
        accuracy_drop = baseline_accuracy - current_accuracy
        
        performance_drift = accuracy_drop > threshold
        
        return performance_drift, current_accuracy
    
    def train_model(self, X_train, y_train, n_estimators=100, max_depth=10):
        """Train a new Random Forest model"""
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model and return metrics"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics
    
    def run_continuous_training(self):
        """
        Main continuous training pipeline with MLflow tracking
        """
        print("="*70)
        print("CONTINUOUS TRAINING WITH MLFLOW TRACKING")
        print("="*70)
        
        with mlflow.start_run(run_name=f"ct_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            mlflow.log_param("tracking_uri", MLFLOW_TRACKING_URI)
            mlflow.log_param("timestamp", datetime.now().isoformat())
            
            # Step 1: Load data
            print("\n[STEP 1] Loading breast cancer dataset...")
            data = load_breast_cancer()
            X, y = data.data, data.target
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            mlflow.log_param("n_samples_train", X_train.shape[0])
            mlflow.log_param("n_samples_test", X_test.shape[0])
            mlflow.log_param("n_features", X_train.shape[1])
            
            print(f"   Training samples: {X_train.shape[0]}")
            print(f"   Test samples: {X_test.shape[0]}")
            
            # Step 2: Load current model
            print("\n[STEP 2] Loading current production model...")
            current_model = self.load_current_model()
            
            if current_model is None:
                print("   No existing model found")
                baseline_accuracy = 0.0
                mlflow.log_param("baseline_exists", False)
            else:
                print("   Current model loaded successfully")
                baseline_accuracy = accuracy_score(y_test, current_model.predict(X_test))
                print(f"   Baseline accuracy: {baseline_accuracy:.4f}")
                mlflow.log_param("baseline_exists", True)
                mlflow.log_metric("baseline_accuracy", baseline_accuracy)
            
            # Step 3: Detect data drift
            print("\n[STEP 3] Detecting data drift...")
            # Simulate new data with slight noise
            X_new = X_test + np.random.normal(0, 0.05 * X_test.std(axis=0), X_test.shape)
            
            data_drift, drift_score, drift_scores = self.detect_data_drift(X_train, X_new)
            
            print(f"   Average drift score: {drift_score:.4f}")
            mlflow.log_metric("data_drift_score", drift_score)
            mlflow.log_param("data_drift_detected", data_drift)
            
            if data_drift:
                print("   ⚠️  DATA DRIFT DETECTED!")
            else:
                print("   ✓ No significant data drift")
            
            # Step 4: Detect performance drift
            print("\n[STEP 4] Detecting performance drift...")
            perf_drift, current_accuracy = self.detect_performance_drift(
                current_model, X_test, y_test, baseline_accuracy
            )
            
            if current_model:
                print(f"   Current accuracy: {current_accuracy:.4f}")
                mlflow.log_metric("current_accuracy", current_accuracy)
            
            mlflow.log_param("performance_drift_detected", perf_drift)
            
            if perf_drift:
                print("   ⚠️  PERFORMANCE DRIFT DETECTED!")
            else:
                print("   ✓ Model performance is stable")
            
            # Step 5: Decision to retrain
            retrain_needed = data_drift or perf_drift or (current_model is None)
            mlflow.log_param("retrain_triggered", retrain_needed)
            
            if not retrain_needed:
                print("\n[DECISION] No retraining needed")
                mlflow.log_param("action_taken", "no_retraining")
                return False
            
            print("\n[DECISION] 🔄 RETRAINING INITIATED")
            mlflow.log_param("action_taken", "retraining")
            
            # Step 6: Train new model
            print("\n[STEP 5] Training new model...")
            n_estimators = 100
            max_depth = 10
            
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            
            new_model = self.train_model(X_train, y_train, n_estimators, max_depth)
            print("   ✓ Model training completed")
            
            # Step 7: Evaluate new model
            print("\n[STEP 6] Evaluating new model...")
            metrics = self.evaluate_model(new_model, X_test, y_test)
            
            print(f"   Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall:    {metrics['recall']:.4f}")
            print(f"   F1 Score:  {metrics['f1_score']:.4f}")
            
            # Log all metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Step 8: Compare and deploy
            print("\n[STEP 7] Model comparison and deployment...")
            
            if current_model is None or metrics['accuracy'] >= baseline_accuracy:
                print("   ✓ New model performs better or equal")
                
                # Save model locally
                with open(self.model_path, 'wb') as f:
                    pickle.dump(new_model, f)
                print(f"   ✓ Model saved to {self.model_path}")
                
                # Log model to MLflow
                mlflow.sklearn.log_model(new_model, "model")
                print("   ✓ Model logged to MLflow")
                
                # Register model
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                model_details = mlflow.register_model(model_uri, "breast_cancer_model")
                print(f"   ✓ Model registered: {model_details.name} v{model_details.version}")
                
                mlflow.log_param("deployment_status", "deployed")
                
                improvement = metrics['accuracy'] - baseline_accuracy
                mlflow.log_metric("accuracy_improvement", improvement)
                
                print(f"\n   Accuracy improvement: {improvement*100:.2f}%")
                
                print("\n" + "="*70)
                print("✓ CONTINUOUS TRAINING COMPLETED - NEW MODEL DEPLOYED")
                print("="*70)
                
                return True
            else:
                print("   ⚠️  New model performs worse, keeping current model")
                mlflow.log_param("deployment_status", "rejected")
                
                print("\n" + "="*70)
                print("⚠️  TRAINING COMPLETED - OLD MODEL RETAINED")
                print("="*70)
                
                return False

def main():
    """Main function"""
    print("Starting Continuous Training with MLflow...")
    print(f"Make sure MLflow server is running on {MLFLOW_TRACKING_URI}")
    print("\nTo start MLflow server, run:")
    print("  mlflow server --host 0.0.0.0 --port 5001\n")
    
    try:
        pipeline = MLflowContinuousTraining()
        success = pipeline.run_continuous_training()
        
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print(f"1. View results in MLflow UI: {MLFLOW_TRACKING_URI}")
        print("2. Compare experiments and model versions")
        print("3. Deploy the best model to production")
        
        if success:
            print("\n✓ New model is ready for deployment!")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure MLflow server is running:")
        print("  mlflow server --host 0.0.0.0 --port 5001")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
