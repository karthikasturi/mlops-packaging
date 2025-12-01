"""
Model Training with MLflow Tracking & Model Registry

This module demonstrates:
1. Experiment tracking with MLflow
2. Hyperparameter logging and tuning
3. Metrics tracking (RMSE, MAE, R², etc.)
4. Model versioning with MLflow Model Registry
5. Model artifact export (pickle file for deployment)
6. Comparing multiple experiment runs
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from datetime import datetime
import pickle
import os
import json


class HouseRentalModelTrainer:
    """
    Handles model training with comprehensive MLflow tracking
    """
    
    def __init__(self, experiment_name="house-rental-model-training"):
        """
        Initialize the model trainer
        
        Args:
            experiment_name: MLflow experiment name
        """
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri("http://localhost:5001")
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        
    def load_data(self, data_path='monitoring/data'):
        """
        Load preprocessed training and test data
        
        Args:
            data_path: Path to the data directory
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\n" + "=" * 70)
        print("LOADING PREPROCESSED DATA")
        print("=" * 70)
        
        train_df = pd.read_csv(os.path.join(data_path, 'house_rental_train_processed.csv'))
        test_df = pd.read_csv(os.path.join(data_path, 'house_rental_test_processed.csv'))
        
        # Exclude categorical columns and target
        feature_cols = [col for col in train_df.columns 
                       if col != 'rent' and 'location' not in col and 'furnishing' not in col]
        
        X_train = train_df[feature_cols]
        y_train = train_df['rent']
        X_test = test_df[feature_cols]
        y_test = test_df['rent']
        
        print(f"✓ Train samples: {len(X_train)}")
        print(f"✓ Test samples: {len(X_test)}")
        print(f"✓ Features: {len(feature_cols)}")
        print(f"✓ Feature names: {', '.join(feature_cols[:5])}...")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        # Additional metrics
        mse = mean_squared_error(y_true, y_pred)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'mse': mse
        }
    
    def train_random_forest(self, X_train, X_test, y_train, y_test, 
                           n_estimators=100, max_depth=10, min_samples_split=2):
        """
        Train Random Forest model with MLflow tracking
        
        Args:
            X_train, X_test, y_train, y_test: Train and test data
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            
        Returns:
            Trained model and metrics
        """
        with mlflow.start_run(run_name=f"rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            print("\n" + "=" * 70)
            print("TRAINING: Random Forest Regressor")
            print("=" * 70)
            
            # Log hyperparameters
            hyperparameters = {
                'model_type': 'RandomForestRegressor',
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'random_state': 42,
                'n_jobs': -1
            }
            
            mlflow.log_params(hyperparameters)
            print("\n📋 Hyperparameters:")
            for key, value in hyperparameters.items():
                print(f"   {key}: {value}")
            
            # Train model
            print("\n🚀 Training model...")
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(y_train, y_pred_train)
            test_metrics = self.calculate_metrics(y_test, y_pred_test)
            
            # Log metrics
            mlflow.log_metrics({
                'train_rmse': train_metrics['rmse'],
                'train_mae': train_metrics['mae'],
                'train_r2': train_metrics['r2'],
                'train_mape': train_metrics['mape'],
                'test_rmse': test_metrics['rmse'],
                'test_mae': test_metrics['mae'],
                'test_r2': test_metrics['r2'],
                'test_mape': test_metrics['mape']
            })
            
            # Log model with input example and signature
            from mlflow.models import infer_signature
            input_example = X_train.iloc[:5]
            # Create signature to avoid integer schema warnings
            signature = infer_signature(X_train.astype('float64'), y_train)
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name="house_rental_rf",
                input_example=input_example,
                signature=signature
            )
            
            # Set tags
            mlflow.set_tag("model_family", "ensemble")
            mlflow.set_tag("algorithm", "random_forest")
            mlflow.set_tag("framework", "scikit-learn")
            
            # Print metrics
            print("\n📊 Training Metrics:")
            print(f"   RMSE: ${train_metrics['rmse']:.2f}")
            print(f"   MAE:  ${train_metrics['mae']:.2f}")
            print(f"   R²:   {train_metrics['r2']:.4f}")
            print(f"   MAPE: {train_metrics['mape']:.2f}%")
            
            print("\n📊 Test Metrics:")
            print(f"   RMSE: ${test_metrics['rmse']:.2f}")
            print(f"   MAE:  ${test_metrics['mae']:.2f}")
            print(f"   R²:   {test_metrics['r2']:.4f}")
            print(f"   MAPE: {test_metrics['mape']:.2f}%")
            
            run_id = mlflow.active_run().info.run_id
            print(f"\n✅ Model logged to MLflow (Run ID: {run_id[:8]}...)")
            
            return model, test_metrics, run_id
    
    def train_gradient_boosting(self, X_train, X_test, y_train, y_test,
                               n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Train Gradient Boosting model with MLflow tracking
        """
        with mlflow.start_run(run_name=f"gb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            print("\n" + "=" * 70)
            print("TRAINING: Gradient Boosting Regressor")
            print("=" * 70)
            
            # Log hyperparameters
            hyperparameters = {
                'model_type': 'GradientBoostingRegressor',
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'random_state': 42
            }
            
            mlflow.log_params(hyperparameters)
            print("\n📋 Hyperparameters:")
            for key, value in hyperparameters.items():
                print(f"   {key}: {value}")
            
            # Train model
            print("\n🚀 Training model...")
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(y_train, y_pred_train)
            test_metrics = self.calculate_metrics(y_test, y_pred_test)
            
            # Log metrics
            mlflow.log_metrics({
                'train_rmse': train_metrics['rmse'],
                'train_mae': train_metrics['mae'],
                'train_r2': train_metrics['r2'],
                'train_mape': train_metrics['mape'],
                'test_rmse': test_metrics['rmse'],
                'test_mae': test_metrics['mae'],
                'test_r2': test_metrics['r2'],
                'test_mape': test_metrics['mape']
            })
            
            # Log model with input example and signature
            from mlflow.models import infer_signature
            input_example = X_train.iloc[:5]
            # Create signature to avoid integer schema warnings
            signature = infer_signature(X_train.astype('float64'), y_train)
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name="house_rental_gb",
                input_example=input_example,
                signature=signature
            )
            
            # Set tags
            mlflow.set_tag("model_family", "ensemble")
            mlflow.set_tag("algorithm", "gradient_boosting")
            mlflow.set_tag("framework", "scikit-learn")
            
            # Print metrics
            print("\n📊 Training Metrics:")
            print(f"   RMSE: ${train_metrics['rmse']:.2f}")
            print(f"   MAE:  ${train_metrics['mae']:.2f}")
            print(f"   R²:   {train_metrics['r2']:.4f}")
            
            print("\n📊 Test Metrics:")
            print(f"   RMSE: ${test_metrics['rmse']:.2f}")
            print(f"   MAE:  ${test_metrics['mae']:.2f}")
            print(f"   R²:   {test_metrics['r2']:.4f}")
            
            run_id = mlflow.active_run().info.run_id
            print(f"\n✅ Model logged to MLflow (Run ID: {run_id[:8]}...)")
            
            return model, test_metrics, run_id
    
    def train_ridge_regression(self, X_train, X_test, y_train, y_test, alpha=1.0):
        """
        Train Ridge Regression model with MLflow tracking
        """
        with mlflow.start_run(run_name=f"ridge_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            print("\n" + "=" * 70)
            print("TRAINING: Ridge Regression")
            print("=" * 70)
            
            # Log hyperparameters
            hyperparameters = {
                'model_type': 'Ridge',
                'alpha': alpha,
                'random_state': 42
            }
            
            mlflow.log_params(hyperparameters)
            print("\n📋 Hyperparameters:")
            for key, value in hyperparameters.items():
                print(f"   {key}: {value}")
            
            # Train model
            print("\n🚀 Training model...")
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(y_train, y_pred_train)
            test_metrics = self.calculate_metrics(y_test, y_pred_test)
            
            # Log metrics
            mlflow.log_metrics({
                'train_rmse': train_metrics['rmse'],
                'train_mae': train_metrics['mae'],
                'train_r2': train_metrics['r2'],
                'test_rmse': test_metrics['rmse'],
                'test_mae': test_metrics['mae'],
                'test_r2': test_metrics['r2']
            })
            
            # Log model with input example and signature
            from mlflow.models import infer_signature
            input_example = X_train.iloc[:5]
            # Create signature to avoid integer schema warnings
            signature = infer_signature(X_train.astype('float64'), y_train)
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name="house_rental_ridge",
                input_example=input_example,
                signature=signature
            )
            
            # Set tags
            mlflow.set_tag("model_family", "linear")
            mlflow.set_tag("algorithm", "ridge_regression")
            
            print("\n📊 Test Metrics:")
            print(f"   RMSE: ${test_metrics['rmse']:.2f}")
            print(f"   R²:   {test_metrics['r2']:.4f}")
            
            run_id = mlflow.active_run().info.run_id
            print(f"\n✅ Model logged to MLflow (Run ID: {run_id[:8]}...)")
            
            return model, test_metrics, run_id
    
    def save_model_pickle(self, model, feature_names, model_name='best_model', 
                         save_path='monitoring/models'):
        """
        Save model as pickle file for deployment
        
        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name for the saved model
            save_path: Directory to save the model
            
        Returns:
            Path to saved pickle file
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Create model package with metadata
        model_package = {
            'model': model,
            'feature_names': feature_names,
            'model_type': type(model).__name__,
            'training_date': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # Save as pickle
        pickle_path = os.path.join(save_path, f'{model_name}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"\n💾 Model saved as pickle: {pickle_path}")
        print(f"   Size: {os.path.getsize(pickle_path) / 1024:.2f} KB")
        
        # Also save metadata as JSON
        metadata = {
            'model_type': model_package['model_type'],
            'training_date': model_package['training_date'],
            'version': model_package['version'],
            'feature_names': feature_names,
            'n_features': len(feature_names)
        }
        
        metadata_path = os.path.join(save_path, f'{model_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   Metadata: {metadata_path}")
        
        return pickle_path
    
    def compare_models(self):
        """
        Compare all models in the experiment
        
        Returns:
            DataFrame with model comparison
        """
        print("\n" + "=" * 70)
        print("COMPARING ALL MODELS")
        print("=" * 70)
        
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            print("❌ No experiment found!")
            return None
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_r2 DESC"]
        )
        
        if not runs:
            print("❌ No runs found!")
            return None
        
        comparison_data = []
        for run in runs:
            data = {
                'run_id': run.info.run_id[:8],
                'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                'model_type': run.data.params.get('model_type', 'N/A'),
                'test_r2': run.data.metrics.get('test_r2', 0),
                'test_rmse': run.data.metrics.get('test_rmse', 0),
                'test_mae': run.data.metrics.get('test_mae', 0),
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S')
            }
            comparison_data.append(data)
        
        df = pd.DataFrame(comparison_data)
        
        print("\n📊 Model Comparison (sorted by Test R²):\n")
        print(df.to_string(index=False))
        
        # Find best model
        best_run = comparison_data[0]
        print(f"\n🏆 Best Model: {best_run['model_type']}")
        print(f"   Run ID: {best_run['run_id']}")
        print(f"   Test R²: {best_run['test_r2']:.4f}")
        print(f"   Test RMSE: ${best_run['test_rmse']:.2f}")
        
        return df
    
    def register_best_model(self, model_name="house_rental_best_model"):
        """
        Register the best performing model to MLflow Model Registry
        
        Args:
            model_name: Name for registered model
            
        Returns:
            Model version info
        """
        print("\n" + "=" * 70)
        print("REGISTERING BEST MODEL TO MODEL REGISTRY")
        print("=" * 70)
        
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        
        # Get best run based on test_r2
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_r2 DESC"],
            max_results=1
        )
        
        if not runs:
            print("❌ No runs found!")
            return None
        
        best_run = runs[0]
        run_id = best_run.info.run_id
        
        print(f"\n✓ Best run identified: {run_id[:8]}...")
        print(f"  Model type: {best_run.data.params.get('model_type', 'N/A')}")
        print(f"  Test R²: {best_run.data.metrics.get('test_r2', 0):.4f}")
        
        # Register model
        model_uri = f"runs:/{run_id}/model"
        
        try:
            model_version = mlflow.register_model(model_uri, model_name)
            
            print(f"\n✅ Model registered successfully!")
            print(f"   Name: {model_name}")
            print(f"   Version: {model_version.version}")
            print(f"   Status: {model_version.status}")
            
            # Add description
            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=f"Best house rental prediction model trained on {datetime.now().strftime('%Y-%m-%d')}. "
                           f"Test R²: {best_run.data.metrics.get('test_r2', 0):.4f}"
            )
            
            return model_version
            
        except Exception as e:
            print(f"⚠️  Error registering model: {str(e)}")
            return None
    
    def transition_model_stage(self, model_name, version, stage="Production"):
        """
        Transition model to a specific stage in Model Registry
        
        Args:
            model_name: Name of registered model
            version: Model version number
            stage: Target stage (Staging, Production, Archived)
        """
        print(f"\n🔄 Transitioning model to {stage}...")
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True
            )
            
            print(f"✅ Model {model_name} v{version} transitioned to {stage}")
            
        except Exception as e:
            print(f"❌ Error transitioning model: {str(e)}")


def run_training_pipeline():
    """
    Run complete model training pipeline with multiple models
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "MODEL TRAINING PIPELINE WITH MLFLOW")
    print("=" * 80)
    
    trainer = HouseRentalModelTrainer()
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = trainer.load_data()
    
    # Train multiple models with different hyperparameters
    models_results = []
    
    # 1. Random Forest - Configuration 1
    print("\n\n" + "█" * 80)
    print("EXPERIMENT 1: Random Forest (100 trees, depth=10)")
    print("█" * 80)
    model_rf1, metrics_rf1, run_id_rf1 = trainer.train_random_forest(
        X_train, X_test, y_train, y_test,
        n_estimators=100, max_depth=10
    )
    models_results.append(('RandomForest_100_10', model_rf1, metrics_rf1))
    
    # 2. Random Forest - Configuration 2
    print("\n\n" + "█" * 80)
    print("EXPERIMENT 2: Random Forest (200 trees, depth=15)")
    print("█" * 80)
    model_rf2, metrics_rf2, run_id_rf2 = trainer.train_random_forest(
        X_train, X_test, y_train, y_test,
        n_estimators=200, max_depth=15
    )
    models_results.append(('RandomForest_200_15', model_rf2, metrics_rf2))
    
    # 3. Gradient Boosting - Configuration 1
    print("\n\n" + "█" * 80)
    print("EXPERIMENT 3: Gradient Boosting (100 trees, lr=0.1)")
    print("█" * 80)
    model_gb1, metrics_gb1, run_id_gb1 = trainer.train_gradient_boosting(
        X_train, X_test, y_train, y_test,
        n_estimators=100, learning_rate=0.1, max_depth=3
    )
    models_results.append(('GradientBoosting_100_0.1', model_gb1, metrics_gb1))
    
    # 4. Gradient Boosting - Configuration 2
    print("\n\n" + "█" * 80)
    print("EXPERIMENT 4: Gradient Boosting (150 trees, lr=0.05)")
    print("█" * 80)
    model_gb2, metrics_gb2, run_id_gb2 = trainer.train_gradient_boosting(
        X_train, X_test, y_train, y_test,
        n_estimators=150, learning_rate=0.05, max_depth=4
    )
    models_results.append(('GradientBoosting_150_0.05', model_gb2, metrics_gb2))
    
    # 5. Ridge Regression
    print("\n\n" + "█" * 80)
    print("EXPERIMENT 5: Ridge Regression (alpha=1.0)")
    print("█" * 80)
    model_ridge, metrics_ridge, run_id_ridge = trainer.train_ridge_regression(
        X_train, X_test, y_train, y_test,
        alpha=1.0
    )
    models_results.append(('Ridge_1.0', model_ridge, metrics_ridge))
    
    # Compare all models
    print("\n\n" + "█" * 80)
    print("MODEL COMPARISON")
    print("█" * 80)
    comparison_df = trainer.compare_models()
    
    # Find best model
    best_model_name, best_model, best_metrics = max(
        models_results, 
        key=lambda x: x[2]['r2']
    )
    
    print("\n\n" + "█" * 80)
    print("SAVING BEST MODEL AS PICKLE")
    print("█" * 80)
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test R²: {best_metrics['r2']:.4f}")
    print(f"   Test RMSE: ${best_metrics['rmse']:.2f}")
    
    # Save best model as pickle
    pickle_path = trainer.save_model_pickle(
        best_model, 
        feature_names,
        model_name='house_rental_best_model',
        save_path='monitoring/models'
    )
    
    # Register best model
    print("\n\n" + "█" * 80)
    print("MODEL REGISTRY OPERATIONS")
    print("█" * 80)
    
    model_version = trainer.register_best_model("HouseRentalPredictor")
    
    if model_version:
        # Transition to Production
        trainer.transition_model_stage(
            "HouseRentalPredictor",
            model_version.version,
            stage="Production"
        )
    
    # Summary
    print("\n\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    print("\n✅ Experiments Completed:")
    print(f"   - Total models trained: {len(models_results)}")
    print(f"   - Best model: {best_model_name}")
    print(f"   - Best Test R²: {best_metrics['r2']:.4f}")
    
    print("\n✅ Artifacts Created:")
    print(f"   - Pickle file: {pickle_path}")
    print(f"   - MLflow runs: {len(models_results)}")
    print(f"   - Registered model: HouseRentalPredictor")
    
    print("\n📊 View Results:")
    print("   MLflow UI: http://localhost:5001")
    print("   Navigate to: Experiments → house-rental-model-training")
    print("   Navigate to: Models → HouseRentalPredictor")
    
    print("\n📦 Deployment Ready:")
    print(f"   Model file: {pickle_path}")
    print("   Ready for packaging and deployment!")
    
    print("\n" + "=" * 80)
    
    return best_model, pickle_path, comparison_df


def load_model_from_pickle(pickle_path):
    """
    Load model from pickle file for inference
    
    Args:
        pickle_path: Path to pickle file
        
    Returns:
        Model package
    """
    print(f"\n📥 Loading model from: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        model_package = pickle.load(f)
    
    print(f"✓ Model type: {model_package['model_type']}")
    print(f"✓ Features: {len(model_package['feature_names'])}")
    print(f"✓ Training date: {model_package['training_date']}")
    print(f"✓ Version: {model_package['version']}")
    
    return model_package


def demonstrate_model_inference():
    """
    Demonstrate model inference using the saved pickle file
    """
    print("\n" + "=" * 70)
    print("MODEL INFERENCE DEMONSTRATION")
    print("=" * 70)
    
    # Load model
    model_package = load_model_from_pickle('monitoring/models/house_rental_best_model.pkl')
    model = model_package['model']
    feature_names = model_package['feature_names']
    
    # Load test data
    test_df = pd.read_csv('monitoring/data/house_rental_test_processed.csv')
    X_test = test_df[feature_names]
    y_test = test_df['rent']
    
    # Make predictions
    print("\n🔮 Making predictions on test data...")
    predictions = model.predict(X_test)
    
    # Show sample predictions
    print("\n📊 Sample Predictions:\n")
    sample_df = pd.DataFrame({
        'Actual Rent': y_test.head(10).values,
        'Predicted Rent': predictions[:10],
        'Difference': y_test.head(10).values - predictions[:10]
    })
    
    print(sample_df.to_string(index=False))
    
    print("\n✅ Model inference successful!")
    print("   Ready for production deployment!")


def main():
    """
    Main entry point
    """
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "inference":
        # Demonstrate inference
        demonstrate_model_inference()
    else:
        # Run training pipeline
        best_model, pickle_path, comparison_df = run_training_pipeline()
        
        print("\n\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("1. View experiments in MLflow UI: http://localhost:5001")
        print("2. Compare model metrics and hyperparameters")
        print("3. Download the pickle file for deployment")
        print("4. Test inference: python train_model.py inference")
        print("5. Package the model for production deployment")
        print("=" * 80)


if __name__ == "__main__":
    main()
