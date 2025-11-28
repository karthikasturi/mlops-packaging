"""
Model comparison script using MLflow
Compares different model versions and their performance
"""
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://localhost:5001"
EXPERIMENT_NAME = "breast_cancer_continuous_training"

def compare_models():
    """Compare all models in the experiment"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # Get experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"Experiment '{EXPERIMENT_NAME}' not found!")
        return
    
    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=10
    )
    
    print("="*80)
    print(f"MODEL COMPARISON - {EXPERIMENT_NAME}")
    print("="*80)
    print(f"\nTotal runs: {len(runs)}\n")
    
    # Display runs in a table format
    print(f"{'Run ID':<35} {'Accuracy':<10} {'Deployed':<10} {'Date':<20}")
    print("-"*80)
    
    for run in runs:
        run_id = run.info.run_id[:32]
        accuracy = run.data.metrics.get('accuracy', 0.0)
        deployed = run.data.params.get('deployment_status', 'N/A')
        timestamp = run.info.start_time / 1000  # Convert to seconds
        
        from datetime import datetime
        date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"{run_id:<35} {accuracy:<10.4f} {deployed:<10} {date_str:<20}")
    
    print("\n" + "="*80)
    
    # Get best model
    best_run = max(runs, key=lambda r: r.data.metrics.get('accuracy', 0.0))
    best_accuracy = best_run.data.metrics.get('accuracy', 0.0)
    
    print(f"\n🏆 BEST MODEL")
    print(f"Run ID: {best_run.info.run_id}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"Precision: {best_run.data.metrics.get('precision', 0.0):.4f}")
    print(f"Recall: {best_run.data.metrics.get('recall', 0.0):.4f}")
    print(f"F1 Score: {best_run.data.metrics.get('f1_score', 0.0):.4f}")
    
    print("\n" + "="*80)

def list_model_versions():
    """List all registered model versions"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    try:
        versions = client.search_model_versions("name='breast_cancer_model'")
        
        print("="*80)
        print("REGISTERED MODEL VERSIONS")
        print("="*80)
        print(f"\nTotal versions: {len(versions)}\n")
        
        print(f"{'Version':<10} {'Stage':<15} {'Run ID':<35} {'Created':<20}")
        print("-"*80)
        
        for version in versions:
            version_num = version.version
            stage = version.current_stage
            run_id = version.run_id[:32]
            created = version.creation_timestamp / 1000
            
            from datetime import datetime
            date_str = datetime.fromtimestamp(created).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"{version_num:<10} {stage:<15} {run_id:<35} {date_str:<20}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"No model versions found or error: {e}")

if __name__ == "__main__":
    print("Connecting to MLflow...")
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}\n")
    
    compare_models()
    print("\n")
    list_model_versions()
