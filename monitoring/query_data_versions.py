"""
Query and Compare Data Versions in MLflow

This script demonstrates how to:
1. Query different data ingestion runs from MLflow
2. Compare preprocessing strategies
3. Analyze data statistics across versions
4. Load specific data versions for model training
"""

import mlflow
import json
import pandas as pd
from datetime import datetime


class DataVersionManager:
    """
    Manages and compares different versions of datasets tracked in MLflow
    """
    
    def __init__(self, experiment_name="house-rental-data-ingestion"):
        """
        Initialize the data version manager
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri("http://localhost:5001")
        mlflow.set_experiment(experiment_name)
        self.client = mlflow.tracking.MlflowClient()
        
    def list_all_data_versions(self):
        """
        List all data ingestion runs with their versions and metadata
        
        Returns:
            DataFrame with run information
        """
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            print(f"❌ Experiment '{self.experiment_name}' not found!")
            return None
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        if not runs:
            print("❌ No runs found in this experiment!")
            return None
        
        run_data = []
        for run in runs:
            run_info = {
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_version': run.data.params.get('dataset_version', 'N/A'),
                'dataset_hash': run.data.params.get('raw_data_hash', 'N/A'),
                'train_samples': run.data.params.get('train_samples', 'N/A'),
                'test_samples': run.data.params.get('test_samples', 'N/A'),
                'status': run.info.status
            }
            run_data.append(run_info)
        
        df = pd.DataFrame(run_data)
        return df
    
    def compare_data_versions(self, run_ids):
        """
        Compare multiple data versions side by side
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            
            data = {
                'run_id': run_id[:8],
                'version': run.data.params.get('dataset_version', 'N/A'),
                'samples': run.data.params.get('raw_data_samples', 'N/A'),
                'train_size': run.data.params.get('train_samples', 'N/A'),
                'test_size': run.data.params.get('test_samples', 'N/A'),
                'avg_rent': run.data.metrics.get('train_avg_rent', 'N/A'),
                'std_rent': run.data.metrics.get('train_std_rent', 'N/A'),
                'avg_area': run.data.metrics.get('train_avg_area', 'N/A')
            }
            comparison_data.append(data)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def get_preprocessing_params(self, run_id):
        """
        Retrieve preprocessing parameters for a specific run
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Dictionary of preprocessing parameters
        """
        run = self.client.get_run(run_id)
        
        # Get all parameters that start with 'preprocessing_'
        preprocessing_params = {
            key.replace('preprocessing_', ''): value
            for key, value in run.data.params.items()
            if key.startswith('preprocessing_')
        }
        
        return preprocessing_params
    
    def download_data_version(self, run_id, local_path="downloaded_data"):
        """
        Download processed data from a specific version
        
        Args:
            run_id: MLflow run ID
            local_path: Local directory to save artifacts
            
        Returns:
            Path to downloaded artifacts
        """
        import os
        
        os.makedirs(local_path, exist_ok=True)
        
        # Download processed data artifacts
        artifact_path = self.client.download_artifacts(
            run_id, 
            "processed_data", 
            local_path
        )
        
        print(f"✓ Downloaded data version from run {run_id[:8]}...")
        print(f"✓ Location: {artifact_path}")
        
        return artifact_path
    
    def load_specific_version(self, run_id):
        """
        Load a specific data version for model training
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            train_df, test_df, metadata
        """
        import os
        import tempfile
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Download artifacts
        self.client.download_artifacts(run_id, "processed_data", temp_dir)
        self.client.download_artifacts(run_id, "metadata", temp_dir)
        
        # Load data
        train_df = pd.read_csv(os.path.join(temp_dir, "processed_data", "house_rental_train_processed.csv"))
        test_df = pd.read_csv(os.path.join(temp_dir, "processed_data", "house_rental_test_processed.csv"))
        
        # Load metadata
        with open(os.path.join(temp_dir, "metadata", "train_metadata.json"), 'r') as f:
            train_metadata = json.load(f)
        
        with open(os.path.join(temp_dir, "metadata", "test_metadata.json"), 'r') as f:
            test_metadata = json.load(f)
        
        metadata = {
            'train': train_metadata,
            'test': test_metadata
        }
        
        return train_df, test_df, metadata
    
    def get_data_statistics(self, run_id):
        """
        Get detailed statistics for a data version
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Dictionary of statistics
        """
        run = self.client.get_run(run_id)
        
        stats = {
            'Dataset Version': run.data.params.get('dataset_version', 'N/A'),
            'Total Samples': run.data.params.get('raw_data_samples', 'N/A'),
            'Train Samples': run.data.params.get('train_samples', 'N/A'),
            'Test Samples': run.data.params.get('test_samples', 'N/A'),
            'Average Rent': f"${run.data.metrics.get('train_avg_rent', 0):.2f}",
            'Rent Std Dev': f"${run.data.metrics.get('train_std_rent', 0):.2f}",
            'Average Area': f"{run.data.metrics.get('train_avg_area', 0):.2f} sqft",
            'Dataset Hash': run.data.params.get('raw_data_hash', 'N/A'),
            'Feature Extraction': run.data.params.get('feature_extraction_method', 'N/A')
        }
        
        return stats


def demonstrate_version_management():
    """
    Demonstrate data version management capabilities
    """
    print("\n" + "=" * 70)
    print("DATA VERSION MANAGEMENT WITH MLFLOW")
    print("=" * 70)
    
    manager = DataVersionManager()
    
    # List all versions
    print("\n📋 LISTING ALL DATA VERSIONS\n")
    versions_df = manager.list_all_data_versions()
    
    if versions_df is not None and not versions_df.empty:
        print(versions_df.to_string(index=False))
        print(f"\n✓ Found {len(versions_df)} data versions")
        
        # Get the latest run
        latest_run_id = versions_df.iloc[0]['run_id']
        
        print("\n" + "=" * 70)
        print("📊 LATEST VERSION STATISTICS")
        print("=" * 70)
        
        stats = manager.get_data_statistics(latest_run_id)
        for key, value in stats.items():
            print(f"{key:20s}: {value}")
        
        print("\n" + "=" * 70)
        print("⚙️  PREPROCESSING PARAMETERS")
        print("=" * 70)
        
        preprocessing = manager.get_preprocessing_params(latest_run_id)
        for key, value in preprocessing.items():
            value_str = str(value)
            if len(value_str) > 60:
                value_str = value_str[:60] + "..."
            print(f"{key:30s}: {value_str}")
        
        # Compare versions if more than one exists
        if len(versions_df) > 1:
            print("\n" + "=" * 70)
            print("🔍 COMPARING DATA VERSIONS")
            print("=" * 70)
            
            run_ids = versions_df['run_id'].head(min(3, len(versions_df))).tolist()
            comparison_df = manager.compare_data_versions(run_ids)
            print("\n", comparison_df.to_string(index=False))
        
        # Example: Load specific version
        print("\n" + "=" * 70)
        print("💾 LOADING SPECIFIC DATA VERSION")
        print("=" * 70)
        print(f"Loading data from run: {latest_run_id[:8]}...")
        
        try:
            train_df, test_df, metadata = manager.load_specific_version(latest_run_id)
            print(f"✓ Train data shape: {train_df.shape}")
            print(f"✓ Test data shape: {test_df.shape}")
            print(f"✓ Train timestamp: {metadata['train']['timestamp']}")
            print(f"✓ Dataset hash: {metadata['train']['dataset_hash']}")
            
            print("\n📊 Sample of loaded train data:")
            print(train_df.head(3).to_string())
            
        except Exception as e:
            print(f"⚠️  Could not load data: {str(e)}")
        
    else:
        print("❌ No data versions found!")
        print("\n💡 TIP: Run 'python data_ingestion.py' first to create data versions")
    
    print("\n" + "=" * 70)
    print("✅ VERSION MANAGEMENT DEMONSTRATION COMPLETE")
    print("=" * 70)


def example_train_with_version():
    """
    Example: How to use a specific data version in model training
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: TRAINING WITH SPECIFIC DATA VERSION")
    print("=" * 70)
    
    manager = DataVersionManager()
    versions_df = manager.list_all_data_versions()
    
    if versions_df is not None and not versions_df.empty:
        # Get the latest version
        latest_run_id = versions_df.iloc[0]['run_id']
        dataset_version = versions_df.iloc[0]['dataset_version']
        
        print(f"\n🎯 Selected data version: {dataset_version}")
        print(f"🔗 Run ID: {latest_run_id[:8]}")
        
        # Load the data
        train_df, test_df, metadata = manager.load_specific_version(latest_run_id)
        
        print(f"\n✓ Loaded train data: {train_df.shape}")
        print(f"✓ Loaded test data: {test_df.shape}")
        
        # Start a new training run that references this data version
        mlflow.set_tracking_uri("http://localhost:5001")
        mlflow.set_experiment("house-rental-model-training")
        
        with mlflow.start_run(run_name=f"training_with_{dataset_version}"):
            # Log the data version being used
            mlflow.log_param("data_version", dataset_version)
            mlflow.log_param("data_run_id", latest_run_id)
            mlflow.log_param("data_hash", metadata['train']['dataset_hash'])
            
            # Your model training code would go here
            print("\n🚀 Starting model training with versioned data...")
            print("   (This is where you'd train your model)")
            
            # Log metrics
            mlflow.log_metric("data_samples_train", len(train_df))
            mlflow.log_metric("data_samples_test", len(test_df))
            
            print("\n✅ Training run logged with data version reference")
    else:
        print("\n❌ No data versions available for training")
        print("💡 TIP: Run 'python data_ingestion.py' first")


if __name__ == "__main__":
    # Demonstrate version management
    demonstrate_version_management()
    
    print("\n\n")
    
    # Show example of training with specific version
    example_train_with_version()
    
    print("\n" + "=" * 70)
    print("USAGE TIPS")
    print("=" * 70)
    print("1. Use list_all_data_versions() to see available versions")
    print("2. Use compare_data_versions() to compare preprocessing strategies")
    print("3. Use load_specific_version() to load data for training")
    print("4. Always log the data version in your training runs")
    print("5. Use dataset hashes to ensure exact reproducibility")
    print("=" * 70)
