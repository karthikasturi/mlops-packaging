# Data Ingestion & Preparation with MLflow Tracking

This example demonstrates how to use MLflow for tracking data ingestion, preprocessing, and versioning in an MLOps pipeline using house rental data.

## 📋 Overview

This module showcases best practices for:

1. **Data Ingestion & Versioning** - Track dataset versions with hashes and timestamps
2. **Feature Extraction** - Engineer new features and log transformation steps
3. **Preprocessing Pipeline** - Apply scaling, encoding, and log all parameters
4. **Metadata Logging** - Record comprehensive dataset statistics and characteristics
5. **Reproducibility** - Ensure all data transformations are tracked and reproducible

## 🏗️ Architecture

```
Data Ingestion Pipeline
├── Raw Data Generation
├── Feature Engineering
├── Train/Test Split
├── Preprocessing (Encoding, Scaling)
├── Metadata Logging
└── MLflow Tracking
```

## 🎯 Use Case: House Rental Price Prediction

The example uses synthetic house rental data with the following features:
- Area (square feet)
- Number of bedrooms/bathrooms
- Location (Downtown, Suburbs, Rural, Uptown)
- Furnishing status
- Parking spaces
- Age of property
- Floor number
- Amenities (gym, pool)

## 🚀 Getting Started

### Prerequisites

Ensure you have MLflow installed and the virtual environment activated:

```bash
source ../mlflow-env/bin/activate
pip install mlflow scikit-learn pandas numpy
```

### Running the Example

```bash
cd monitoring
python data_ingestion.py
```

### View Results in MLflow UI

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

## 📊 What Gets Tracked

### Parameters Logged
- Raw data samples count
- Data file paths and hashes
- Feature extraction methods
- Train/test split ratio
- Preprocessing parameters (encoding mappings, scaling factors)
- Dataset version identifier

### Metrics Logged
- Average rent (train/test)
- Rent standard deviation
- Average area statistics

### Artifacts Logged
- **raw_data/** - Original dataset CSV
- **processed_data/** - Preprocessed train/test datasets
- **preprocessing/** - Preprocessing parameters JSON
- **metadata/** - Comprehensive dataset metadata

### Tags
- `pipeline_stage`: data_ingestion
- `data_quality`: validated

## 🔍 Key Features

### 1. Dataset Versioning
```python
dataset_hash = calculate_dataset_hash(df)
mlflow.log_param("dataset_hash", dataset_hash)
mlflow.log_param("dataset_version", "v20251201_143022")
```

### 2. Feature Engineering Tracking
```python
# Extracted features are logged
mlflow.log_param("extracted_features", [
    "price_per_sqft",
    "room_bath_ratio",
    "total_rooms",
    "amenities_score"
])
```

### 3. Preprocessing Parameters
```python
# All transformations are recorded
mlflow.log_params({
    "location_mapping": {"Downtown": 0, "Suburbs": 1, ...},
    "scaler_mean": [1200.5, 2.8, ...],
    "scaler_std": [650.2, 1.1, ...]
})
```

### 4. Metadata Logging
```python
metadata = {
    "timestamp": "2025-12-01T14:30:22",
    "n_samples": 1000,
    "dataset_hash": "a3f5c8d2e1b4f7a9",
    "data_statistics": {...}
}
```

## 📈 Benefits

1. **Reproducibility** - Every data transformation is tracked
2. **Version Control** - Dataset versions are hashed and logged
3. **Traceability** - Complete audit trail of data preparation
4. **Comparison** - Easy to compare different preprocessing strategies
5. **Debugging** - Quick identification of data-related issues

## 🔧 Customization

### Using Your Own Data

Replace the `generate_synthetic_data()` method with your data loading logic:

```python
def load_your_data(self, file_path):
    df = pd.read_csv(file_path)
    # Your data loading logic
    return df
```

### Adding Custom Features

Extend the `extract_features()` method:

```python
def extract_features(self, df):
    df = df.copy()
    # Add your custom feature engineering
    df['custom_feature'] = df['col1'] * df['col2']
    return df
```

### Custom Preprocessing

Modify the `preprocess_data()` method to add your transformations:

```python
def preprocess_data(self, df, fit=True):
    # Add custom preprocessing steps
    # All parameters will be automatically logged
    return df, preprocessing_params
```

## 📁 Output Structure

After running the pipeline, you'll find:

```
monitoring/
├── data/
│   ├── house_rental_raw.csv
│   ├── house_rental_train_processed.csv
│   ├── house_rental_test_processed.csv
│   ├── preprocessing_params.json
│   ├── train_metadata.json
│   └── test_metadata.json
└── data_ingestion.py
```

## 🔄 Integration with Training Pipeline

Use the logged data version in your training pipeline:

```python
# In your training script
with mlflow.start_run():
    # Load data with tracked version
    data_version = "v20251201_143022"
    mlflow.log_param("data_version", data_version)
    
    # Train model with versioned data
    train_model(data_version)
```

## 📚 Additional Resources

- [MLflow Tracking Documentation](https://mlflow.org/docs/latest/tracking.html)
- [Data Versioning Best Practices](https://mlflow.org/docs/latest/tracking.html#dataset-tracking)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

## 🎓 Learning Outcomes

After working with this example, you'll understand:
- How to version datasets using MLflow
- Best practices for tracking preprocessing parameters
- How to log comprehensive metadata for reproducibility
- Integration of data preparation with ML pipelines
- Dataset lineage and traceability

## 🤝 Next Steps

1. Run the example and explore the MLflow UI
2. Modify the feature engineering logic
3. Try different preprocessing strategies and compare results
4. Integrate with the model training pipeline
5. Set up automated data quality checks

---

**Note**: This is a demonstration example using synthetic data. Adapt it to your specific use case and data requirements.
