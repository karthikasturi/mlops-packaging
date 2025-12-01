"""
Complete MLflow Data Management Example

This script demonstrates the full lifecycle:
1. Data ingestion and versioning
2. Querying and comparing versions
3. Drift detection
4. Model training with versioned data
"""

import sys
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def train_model_with_versioned_data():
    """
    Train a model using versioned data from MLflow
    """
    print("\n" + "=" * 70)
    print("MODEL TRAINING WITH VERSIONED DATA")
    print("=" * 70)
    
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("house-rental-model-training")
    
    # Load the versioned data
    print("\n📥 Loading versioned training data...")
    
    try:
        train_df = pd.read_csv('monitoring/data/house_rental_train_processed.csv')
        test_df = pd.read_csv('monitoring/data/house_rental_test_processed.csv')
        
        print(f"✓ Train data: {train_df.shape}")
        print(f"✓ Test data: {test_df.shape}")
        
    except FileNotFoundError:
        print("❌ Processed data not found!")
        print("💡 Run 'python data_ingestion.py' first")
        return None
    
    # Prepare features and target
    feature_cols = [col for col in train_df.columns if col != 'rent' and 'location' not in col and 'furnishing' not in col]
    
    X_train = train_df[feature_cols]
    y_train = train_df['rent']
    X_test = test_df[feature_cols]
    y_test = test_df['rent']
    
    print(f"\n📊 Features used: {len(feature_cols)}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"rf_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log data version info
        mlflow.log_param("data_source", "house_rental_versioned")
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", len(feature_cols))
        
        # Train model
        print("\n🚀 Training Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Log model parameters
        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        })
        
        # Predict and evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Log metrics
        mlflow.log_metrics({
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2
        })
        
        # Log model with input example and signature
        from mlflow.models import infer_signature
        input_example = X_train.iloc[:5]
        # Create signature to avoid integer schema warnings
        signature = infer_signature(X_train.astype('float64'), y_train)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            signature=signature
        )
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv('monitoring/feature_importance.csv', index=False)
        mlflow.log_artifact('monitoring/feature_importance.csv', artifact_path="model_analysis")
        
        # Set tags
        mlflow.set_tag("model_type", "regression")
        mlflow.set_tag("framework", "scikit-learn")
        mlflow.set_tag("data_versioned", "yes")
        
        print("\n" + "=" * 70)
        print("📊 MODEL PERFORMANCE")
        print("=" * 70)
        print(f"Train RMSE: ${train_rmse:.2f}")
        print(f"Test RMSE:  ${test_rmse:.2f}")
        print(f"Train MAE:  ${train_mae:.2f}")
        print(f"Test MAE:   ${test_mae:.2f}")
        print(f"Train R²:   {train_r2:.4f}")
        print(f"Test R²:    {test_r2:.4f}")
        
        print("\n🎯 Top 5 Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']:20s}: {row['importance']:.4f}")
        
        print("\n✅ Model training completed and logged to MLflow")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")
        
        return model, test_r2


def run_complete_pipeline():
    """
    Run the complete data management pipeline
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "MLFLOW DATA MANAGEMENT PIPELINE")
    print("=" * 80)
    print("\nThis demonstration will:")
    print("  1. Ingest and version house rental data")
    print("  2. Track preprocessing parameters")
    print("  3. Train a model with versioned data")
    print("  4. Log all artifacts to MLflow")
    print("\n" + "=" * 80)
    
    # Step 1: Data Ingestion
    print("\n\n")
    print("█" * 80)
    print("STEP 1: DATA INGESTION & PREPARATION")
    print("█" * 80)
    
    try:
        from data_ingestion import HouseRentalDataIngestion
        
        ingestion = HouseRentalDataIngestion()
        train_data, test_data, train_meta, test_meta = ingestion.run_data_ingestion_pipeline(
            n_samples=1000,
            save_path='monitoring/data'
        )
        
    except Exception as e:
        print(f"\n❌ Error in data ingestion: {str(e)}")
        print("💡 Make sure all dependencies are installed")
        return
    
    # Step 2: Model Training
    print("\n\n")
    print("█" * 80)
    print("STEP 2: MODEL TRAINING WITH VERSIONED DATA")
    print("█" * 80)
    
    try:
        model, test_r2 = train_model_with_versioned_data()
        
    except Exception as e:
        print(f"\n❌ Error in model training: {str(e)}")
        return
    
    # Step 3: Summary
    print("\n\n")
    print("█" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("█" * 80)
    print("\n✅ Data Ingestion:")
    print(f"   - Generated {len(train_data) + len(test_data)} samples")
    print(f"   - Dataset version: {train_meta['dataset_hash']}")
    print(f"   - Preprocessing logged: ✓")
    
    print("\n✅ Model Training:")
    print(f"   - Model type: Random Forest Regressor")
    print(f"   - Test R² Score: {test_r2:.4f}")
    print(f"   - Model logged: ✓")
    
    print("\n✅ MLflow Tracking:")
    print("   - All parameters logged: ✓")
    print("   - All metrics logged: ✓")
    print("   - All artifacts saved: ✓")
    
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    print("\n📊 View Results:")
    print("   Run: mlflow ui")
    print("   Open: http://localhost:5001")
    
    print("\n📁 Generated Files:")
    print("   - monitoring/data/house_rental_raw.csv")
    print("   - monitoring/data/house_rental_train_processed.csv")
    print("   - monitoring/data/house_rental_test_processed.csv")
    print("   - monitoring/data/preprocessing_params.json")
    print("   - monitoring/data/train_metadata.json")
    print("   - monitoring/data/test_metadata.json")
    print("   - monitoring/feature_importance.csv")
    
    print("\n📚 Next Steps:")
    print("   1. Explore the MLflow UI to see all logged experiments")
    print("   2. Run 'python query_data_versions.py' to query data versions")
    print("   3. Run 'python detect_data_drift.py' to detect data drift")
    print("   4. Compare different preprocessing strategies")
    print("   5. Track model performance across data versions")
    
    print("\n" + "=" * 80)


def show_menu():
    """
    Display interactive menu
    """
    print("\n" + "=" * 70)
    print("MLFLOW DATA INGESTION & PREPARATION - MENU")
    print("=" * 70)
    print("\nChoose an option:")
    print("  1. Run complete pipeline (ingestion + training)")
    print("  2. Data ingestion only")
    print("  3. Query data versions")
    print("  4. Detect data drift")
    print("  5. Train model with existing data")
    print("  6. Exit")
    print("\n" + "=" * 70)
    
    choice = input("\nEnter your choice (1-6): ").strip()
    return choice


def main():
    """
    Main entry point
    """
    if len(sys.argv) > 1:
        # Command line argument provided
        command = sys.argv[1]
        
        if command == "pipeline":
            run_complete_pipeline()
        elif command == "ingest":
            from data_ingestion import main as ingest_main
            ingest_main()
        elif command == "query":
            from query_data_versions import demonstrate_version_management
            demonstrate_version_management()
        elif command == "drift":
            from detect_data_drift import demonstrate_drift_detection
            demonstrate_drift_detection()
        elif command == "train":
            train_model_with_versioned_data()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: pipeline, ingest, query, drift, train")
    else:
        # Interactive menu
        while True:
            choice = show_menu()
            
            if choice == "1":
                run_complete_pipeline()
                break
            elif choice == "2":
                from data_ingestion import main as ingest_main
                ingest_main()
                break
            elif choice == "3":
                from query_data_versions import demonstrate_version_management
                demonstrate_version_management()
                break
            elif choice == "4":
                from detect_data_drift import demonstrate_drift_detection
                demonstrate_drift_detection()
                break
            elif choice == "5":
                train_model_with_versioned_data()
                break
            elif choice == "6":
                print("\n👋 Goodbye!")
                break
            else:
                print("\n❌ Invalid choice! Please enter 1-6.")
                continue


if __name__ == "__main__":
    main()
