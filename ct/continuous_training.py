"""
Continuous training pipeline for breast cancer prediction model
Automatically retrains model when drift is detected
"""
import pickle
import os
from datetime import datetime
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class ContinuousTrainingPipeline:
    def __init__(self, model_path='../model/model.pkl', backup_dir='../model/model_backups'):
        self.model_path = model_path
        self.backup_dir = backup_dir
        self.training_history = []
        
        # Create backup directory if it doesn't exist
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
    
    def backup_current_model(self):
        """Backup the current model before retraining"""
        if os.path.exists(self.model_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f'model_backup_{timestamp}.pkl')
            
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(backup_path, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"✓ Current model backed up to: {backup_path}")
            return backup_path
        else:
            print("No existing model to backup")
            return None
    
    def load_and_prepare_data(self, add_noise=False):
        """Load and prepare training data"""
        print("\nLoading dataset...")
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        # Simulate new data with potential drift
        if add_noise:
            print("Adding noise to simulate data drift...")
            noise_level = 0.05
            X = X + np.random.normal(0, noise_level * X.std(axis=0), X.shape)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def train_new_model(self, X_train, y_train, n_estimators=100):
        """Train a new model"""
        print("\nTraining new model...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print("✓ Model training completed")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def compare_models(self, old_model, new_model, X_test, y_test):
        """Compare old and new model performance"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        old_accuracy = accuracy_score(y_test, old_model.predict(X_test))
        new_accuracy = accuracy_score(y_test, new_model.predict(X_test))
        
        print(f"Old Model Accuracy: {old_accuracy:.4f}")
        print(f"New Model Accuracy: {new_accuracy:.4f}")
        print(f"Improvement: {(new_accuracy - old_accuracy)*100:.2f}%")
        
        # Decide whether to deploy new model
        if new_accuracy >= old_accuracy:
            print("\n✓ New model performs better or equal, deploying...")
            return True
        else:
            print("\n⚠️  New model performs worse, keeping old model")
            return False
    
    def save_model(self, model, version=None):
        """Save the trained model"""
        if version:
            model_path = f'model_v{version}.pkl'
        else:
            model_path = self.model_path
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✓ Model saved to: {model_path}")
        return model_path
    
    def run_continuous_training(self, force_retrain=False):
        """
        Run the continuous training pipeline
        """
        print("="*60)
        print("CONTINUOUS TRAINING PIPELINE")
        print("="*60)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Started at: {timestamp}\n")
        
        # Step 1: Backup current model
        print("Step 1: Backing up current model...")
        backup_path = self.backup_current_model()
        
        # Step 2: Load data
        print("\nStep 2: Loading and preparing data...")
        X_train, X_test, y_train, y_test = self.load_and_prepare_data(add_noise=False)
        
        # Step 3: Train new model
        print("\nStep 3: Training new model...")
        new_model = self.train_new_model(X_train, y_train)
        new_accuracy = self.evaluate_model(new_model, X_test, y_test)
        
        # Step 4: Compare with old model if exists
        if os.path.exists(self.model_path) and not force_retrain:
            print("\nStep 4: Comparing with existing model...")
            with open(self.model_path, 'rb') as f:
                old_model = pickle.load(f)
            
            deploy_new = self.compare_models(old_model, new_model, X_test, y_test)
        else:
            print("\nStep 4: No existing model found, deploying new model...")
            deploy_new = True
        
        # Step 5: Deploy new model if better
        if deploy_new:
            print("\nStep 5: Deploying new model...")
            self.save_model(new_model)
            
            # Save training history
            self.training_history.append({
                'timestamp': timestamp,
                'accuracy': new_accuracy,
                'deployed': True
            })
            
            print("\n" + "="*60)
            print("✓ CONTINUOUS TRAINING COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"New model deployed with accuracy: {new_accuracy:.4f}")
        else:
            print("\n" + "="*60)
            print("⚠️  TRAINING COMPLETED - OLD MODEL RETAINED")
            print("="*60)
        
        return deploy_new

def main():
    """Main function to run continuous training"""
    import sys
    
    force_retrain = '--force' in sys.argv
    
    pipeline = ContinuousTrainingPipeline()
    success = pipeline.run_continuous_training(force_retrain=force_retrain)
    
    if success:
        print("\nNext steps:")
        print("1. Test the new model: python app.py")
        print("2. Rebuild Docker image: docker build -t mlops-app:v2 .")
        print("3. Deploy to production")
    
    return success

if __name__ == "__main__":
    main()
