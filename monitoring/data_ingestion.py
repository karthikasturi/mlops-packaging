"""
Data Ingestion & Preparation with MLflow Tracking
House Rental Dataset Example

This module demonstrates:
1. Data ingestion and versioning
2. Feature extraction and preprocessing
3. Dataset versioning with MLflow
4. Tracking preprocessing parameters
5. Logging metadata for reproducibility
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import hashlib
import json
import os


class HouseRentalDataIngestion:
    """
    Handles data ingestion, preprocessing, and versioning for house rental data
    """
    
    def __init__(self, experiment_name="house-rental-data-ingestion"):
        """
        Initialize the data ingestion pipeline
        
        Args:
            experiment_name: MLflow experiment name
        """
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri("http://localhost:5001")
        mlflow.set_experiment(experiment_name)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic house rental data
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with house rental data
        """
        np.random.seed(42)
        
        data = {
            'area_sqft': np.random.randint(500, 3000, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'location': np.random.choice(['Downtown', 'Suburbs', 'Rural', 'Uptown'], n_samples),
            'furnishing': np.random.choice(['Furnished', 'Semi-Furnished', 'Unfurnished'], n_samples),
            'parking': np.random.randint(0, 4, n_samples),
            'age_years': np.random.randint(0, 50, n_samples),
            'floor': np.random.randint(0, 20, n_samples),
            'has_gym': np.random.choice([0, 1], n_samples),
            'has_pool': np.random.choice([0, 1], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Generate rent based on features (with some noise)
        base_rent = (
            df['area_sqft'] * 0.5 +
            df['bedrooms'] * 200 +
            df['bathrooms'] * 150 +
            (df['location'] == 'Downtown').astype(int) * 500 +
            (df['location'] == 'Uptown').astype(int) * 300 +
            (df['furnishing'] == 'Furnished').astype(int) * 300 +
            df['parking'] * 100 +
            df['has_gym'] * 150 +
            df['has_pool'] * 200 -
            df['age_years'] * 10
        )
        
        df['rent'] = base_rent + np.random.normal(0, 200, n_samples)
        df['rent'] = df['rent'].clip(lower=500)  # Minimum rent
        
        return df
    
    def calculate_dataset_hash(self, df):
        """
        Calculate hash of the dataset for versioning
        
        Args:
            df: Input DataFrame
            
        Returns:
            SHA256 hash of the dataset
        """
        # Convert DataFrame to string and calculate hash
        df_str = df.to_json(orient='records')
        return hashlib.sha256(df_str.encode()).hexdigest()[:16]
    
    def extract_features(self, df):
        """
        Extract additional features from the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Feature engineering
        df['price_per_sqft'] = df['rent'] / df['area_sqft']
        df['room_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 0.1)
        df['total_rooms'] = df['bedrooms'] + df['bathrooms']
        df['amenities_score'] = df['has_gym'] + df['has_pool'] + (df['parking'] > 0).astype(int)
        df['is_new'] = (df['age_years'] < 5).astype(int)
        df['is_spacious'] = (df['area_sqft'] > df['area_sqft'].median()).astype(int)
        
        return df
    
    def preprocess_data(self, df, fit=True):
        """
        Preprocess the data: encoding, scaling, and transformations
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the transformers (True for training data)
            
        Returns:
            Preprocessed DataFrame and preprocessing parameters
        """
        df = df.copy()
        preprocessing_params = {}
        
        # Encode categorical variables
        categorical_columns = ['location', 'furnishing']
        
        for col in categorical_columns:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
                preprocessing_params[f'{col}_mapping'] = dict(
                    zip(self.label_encoders[col].classes_, 
                        self.label_encoders[col].transform(self.label_encoders[col].classes_))
                )
            else:
                df[col + '_encoded'] = self.label_encoders[col].transform(df[col])
        
        # Select numerical features for scaling
        numerical_features = [
            'area_sqft', 'bedrooms', 'bathrooms', 'parking', 
            'age_years', 'floor', 'price_per_sqft', 'room_bath_ratio',
            'total_rooms', 'amenities_score'
        ]
        
        if fit:
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
            preprocessing_params['scaler_mean'] = self.scaler.mean_.tolist()
            preprocessing_params['scaler_std'] = self.scaler.scale_.tolist()
            preprocessing_params['numerical_features'] = numerical_features
        else:
            df[numerical_features] = self.scaler.transform(df[numerical_features])
        
        return df, preprocessing_params
    
    def log_dataset_metadata(self, df, dataset_name, preprocessing_params):
        """
        Log dataset metadata to MLflow
        
        Args:
            df: DataFrame
            dataset_name: Name/identifier for the dataset
            preprocessing_params: Dictionary of preprocessing parameters
        """
        metadata = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'n_samples': int(len(df)),
            'n_features': int(len(df.columns)),
            'columns': list(df.columns),
            'dataset_hash': self.calculate_dataset_hash(df),
            'target_column': 'rent',
            'missing_values': {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
            'data_statistics': {
                'rent_mean': float(df['rent'].mean()),
                'rent_std': float(df['rent'].std()),
                'rent_min': float(df['rent'].min()),
                'rent_max': float(df['rent'].max()),
                'area_mean': float(df['area_sqft'].mean()),
                'area_std': float(df['area_sqft'].std())
            }
        }
        
        return metadata
    
    def run_data_ingestion_pipeline(self, n_samples=1000, save_path='data'):
        """
        Execute the complete data ingestion and preparation pipeline
        
        Args:
            n_samples: Number of samples to generate
            save_path: Path to save the processed data
            
        Returns:
            Processed train and test datasets
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"data_ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            print("=" * 70)
            print("STEP 1: Data Generation")
            print("=" * 70)
            
            # Generate synthetic data
            df_raw = self.generate_synthetic_data(n_samples)
            raw_data_path = os.path.join(save_path, 'house_rental_raw.csv')
            df_raw.to_csv(raw_data_path, index=False)
            
            print(f"✓ Generated {len(df_raw)} samples")
            print(f"✓ Saved raw data to: {raw_data_path}")
            
            # Log raw data version
            mlflow.log_param("raw_data_samples", len(df_raw))
            mlflow.log_param("raw_data_path", raw_data_path)
            mlflow.log_param("raw_data_hash", self.calculate_dataset_hash(df_raw))
            mlflow.log_artifact(raw_data_path, artifact_path="raw_data")
            
            print("\n" + "=" * 70)
            print("STEP 2: Feature Extraction")
            print("=" * 70)
            
            # Feature extraction
            df_featured = self.extract_features(df_raw)
            
            # Log feature extraction parameters
            new_features = [col for col in df_featured.columns if col not in df_raw.columns]
            mlflow.log_param("extracted_features", json.dumps(new_features))
            mlflow.log_param("feature_extraction_method", "manual_engineering")
            
            print(f"✓ Extracted {len(new_features)} new features:")
            for feat in new_features:
                print(f"  - {feat}")
            
            print("\n" + "=" * 70)
            print("STEP 3: Data Splitting")
            print("=" * 70)
            
            # Split data
            train_df, test_df = train_test_split(
                df_featured, 
                test_size=0.2, 
                random_state=42
            )
            
            print(f"✓ Train set: {len(train_df)} samples ({len(train_df)/len(df_featured)*100:.1f}%)")
            print(f"✓ Test set: {len(test_df)} samples ({len(test_df)/len(df_featured)*100:.1f}%)")
            
            # Log split parameters
            mlflow.log_param("train_test_split_ratio", "80:20")
            mlflow.log_param("train_samples", len(train_df))
            mlflow.log_param("test_samples", len(test_df))
            mlflow.log_param("random_state", 42)
            
            print("\n" + "=" * 70)
            print("STEP 4: Data Preprocessing")
            print("=" * 70)
            
            # Preprocess training data
            train_df_processed, preprocessing_params = self.preprocess_data(train_df, fit=True)
            
            # Preprocess test data (using fitted transformers)
            test_df_processed, _ = self.preprocess_data(test_df, fit=False)
            
            print("✓ Applied preprocessing transformations:")
            print("  - Label encoding for categorical variables")
            print("  - Standard scaling for numerical features")
            print(f"  - Total features after preprocessing: {len(train_df_processed.columns)}")
            
            # Save preprocessing parameters
            preprocessing_params_path = os.path.join(save_path, 'preprocessing_params.json')
            # Convert numpy types to Python types for JSON serialization
            preprocessing_params_serializable = {}
            for key, value in preprocessing_params.items():
                if isinstance(value, (list, tuple)):
                    preprocessing_params_serializable[key] = [float(v) if hasattr(v, 'item') else v for v in value]
                elif isinstance(value, dict):
                    preprocessing_params_serializable[key] = {k: (int(v) if hasattr(v, 'item') else v) for k, v in value.items()}
                else:
                    preprocessing_params_serializable[key] = value
            
            with open(preprocessing_params_path, 'w') as f:
                json.dump(preprocessing_params_serializable, f, indent=2)
            
            # Log preprocessing parameters
            mlflow.log_params({
                f"preprocessing_{key}": str(value)[:250]  # MLflow param limit
                for key, value in preprocessing_params.items()
                if key not in ['scaler_mean', 'scaler_std']
            })
            mlflow.log_artifact(preprocessing_params_path, artifact_path="preprocessing")
            
            print("\n" + "=" * 70)
            print("STEP 5: Dataset Versioning & Metadata Logging")
            print("=" * 70)
            
            # Log dataset metadata for training data
            train_metadata = self.log_dataset_metadata(
                train_df_processed, 
                "house_rental_train", 
                preprocessing_params
            )
            
            # Log dataset metadata for test data
            test_metadata = self.log_dataset_metadata(
                test_df_processed,
                "house_rental_test",
                preprocessing_params
            )
            
            # Save metadata
            train_metadata_path = os.path.join(save_path, 'train_metadata.json')
            test_metadata_path = os.path.join(save_path, 'test_metadata.json')
            
            with open(train_metadata_path, 'w') as f:
                json.dump(train_metadata, f, indent=2)
            
            with open(test_metadata_path, 'w') as f:
                json.dump(test_metadata, f, indent=2)
            
            # Log metadata to MLflow
            mlflow.log_artifact(train_metadata_path, artifact_path="metadata")
            mlflow.log_artifact(test_metadata_path, artifact_path="metadata")
            
            # Log key statistics
            mlflow.log_metrics({
                "train_avg_rent": train_metadata['data_statistics']['rent_mean'],
                "train_std_rent": train_metadata['data_statistics']['rent_std'],
                "train_avg_area": train_metadata['data_statistics']['area_mean'],
                "test_avg_rent": test_metadata['data_statistics']['rent_mean'],
                "test_std_rent": test_metadata['data_statistics']['rent_std'],
            })
            
            print(f"✓ Logged train dataset metadata (hash: {train_metadata['dataset_hash']})")
            print(f"✓ Logged test dataset metadata (hash: {test_metadata['dataset_hash']})")
            
            # Save processed datasets
            train_processed_path = os.path.join(save_path, 'house_rental_train_processed.csv')
            test_processed_path = os.path.join(save_path, 'house_rental_test_processed.csv')
            
            train_df_processed.to_csv(train_processed_path, index=False)
            test_df_processed.to_csv(test_processed_path, index=False)
            
            mlflow.log_artifact(train_processed_path, artifact_path="processed_data")
            mlflow.log_artifact(test_processed_path, artifact_path="processed_data")
            
            print(f"✓ Saved processed train data to: {train_processed_path}")
            print(f"✓ Saved processed test data to: {test_processed_path}")
            
            # Log dataset version
            dataset_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow.log_param("dataset_version", dataset_version)
            mlflow.set_tag("pipeline_stage", "data_ingestion")
            mlflow.set_tag("data_quality", "validated")
            
            print("\n" + "=" * 70)
            print("STEP 6: Data Quality Summary")
            print("=" * 70)
            print(f"✓ Dataset Version: {dataset_version}")
            print(f"✓ Total samples processed: {len(df_featured)}")
            print(f"✓ Features after engineering: {len(df_featured.columns)}")
            print(f"✓ Missing values: {df_featured.isnull().sum().sum()}")
            print(f"✓ Rent range: ${train_metadata['data_statistics']['rent_min']:.2f} - ${train_metadata['data_statistics']['rent_max']:.2f}")
            print(f"✓ All artifacts logged to MLflow")
            
            print("\n" + "=" * 70)
            print("✅ DATA INGESTION PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(f"\nMLflow UI: Run 'mlflow ui' to view tracked experiments")
            print(f"Data location: {save_path}/")
            
            return train_df_processed, test_df_processed, train_metadata, test_metadata


def main():
    """
    Main function to run the data ingestion pipeline
    """
    print("\n" + "=" * 70)
    print("HOUSE RENTAL DATA INGESTION & PREPARATION WITH MLFLOW")
    print("=" * 70)
    print()
    
    # Initialize the data ingestion pipeline
    ingestion = HouseRentalDataIngestion(
        experiment_name="house-rental-data-ingestion"
    )
    
    # Run the pipeline
    train_data, test_data, train_meta, test_meta = ingestion.run_data_ingestion_pipeline(
        n_samples=1000,
        save_path='monitoring/data'
    )
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Review logged data in MLflow UI: mlflow ui")
    print("2. Use the processed datasets for model training")
    print("3. Track model performance with the logged data versions")
    print("4. Compare different data preprocessing strategies")
    print("=" * 70)


if __name__ == "__main__":
    main()
