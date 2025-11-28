"""
Flask API for breast cancer prediction
"""
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model
MODEL_PATH = '../model/model.pkl'
model = None

def load_model():
    """Load the trained model from pickle file"""
    global model
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
        model = None

# Load model when app starts
load_model()

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Breast Cancer Prediction API',
        'endpoints': {
            '/predict': 'POST - Make a prediction',
            '/health': 'GET - Check API health'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    model_status = 'loaded' if model is not None else 'not loaded'
    return jsonify({
        'status': 'healthy',
        'model_status': model_status
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict breast cancer diagnosis
    
    Expects JSON with 'features' key containing 30 feature values
    Example:
    {
        "features": [17.99, 10.38, 122.8, 1001, 0.1184, ...]
    }
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Get features from request
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({
                'error': 'Missing "features" in request body'
            }), 400
        
        features = data['features']
        
        # Validate number of features
        if len(features) != 30:
            return jsonify({
                'error': f'Expected 30 features, got {len(features)}'
            }), 400
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        # Map prediction to label
        result = {
            'prediction': int(prediction),
            'diagnosis': 'malignant' if prediction == 0 else 'benign',
            'probability': {
                'malignant': float(probability[0]),
                'benign': float(probability[1])
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
