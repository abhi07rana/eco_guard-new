from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from flask_cors import CORS
import io
import os  # Added import for os to handle environment variable

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load models
air_quality_model = tf.keras.models.load_model('EcoGuard_Air_Quality.keras')
water_quality_model = tf.keras.models.load_model('EcoGuard_Water_Quality.keras')
pollution_model = tf.keras.models.load_model('EcoGuard_Pollution.keras')

# Load pre-processing tools
scaler = joblib.load('Health_Incidents_Scaler.pkl')  # Load the scaler
label_encoder = joblib.load('Health_Incidents_LabelEncoder.pkl')  # Load the label encoder

# Define a route to upload the CSV file and make predictions
@app.route('/upload', methods=['POST'])
def upload_file():
    # Ensure a file is uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']

    # Check if the file is CSV
    if file.filename == '' or not file.filename.endswith('.csv'):
        return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400

    try:
        # Read the uploaded CSV file into a DataFrame
        data = pd.read_csv(io.StringIO(file.read().decode('utf-8')))

        # Ensure necessary columns are present
        if 'Health Incidents' not in data.columns:
            return jsonify({"error": "'Health Incidents' column is missing in the CSV."}), 400

        # Preprocess the data
        X = data.drop(columns=['Health Incidents', 'Location'], errors='ignore')  # Drop unnecessary columns
        X_scaled = scaler.transform(X)  # Apply the same scaling as during training

        # Make predictions using the models
        air_quality_predictions = air_quality_model.predict(X_scaled)
        water_quality_predictions = water_quality_model.predict(X_scaled)
        pollution_predictions = pollution_model.predict(X_scaled)

        # Decode the predictions to the actual labels
        air_quality_class = np.argmax(air_quality_predictions, axis=1)
        water_quality_class = np.argmax(water_quality_predictions, axis=1)
        pollution_class = np.argmax(pollution_predictions, axis=1)

        air_quality_labels = label_encoder.inverse_transform(air_quality_class)
        water_quality_labels = label_encoder.inverse_transform(water_quality_class)
        pollution_labels = label_encoder.inverse_transform(pollution_class)

        # Add predicted labels to the DataFrame
        data['Predicted Air Quality'] = air_quality_labels
        data['Predicted Water Quality'] = water_quality_labels
        data['Predicted Pollution'] = pollution_labels

        # Return the results as JSON
        result = data.to_dict(orient='records')  # Convert the DataFrame to a list of dictionaries
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"An error occurred while processing the file: {str(e)}"}), 500

# Default route
@app.route('/')
def home():
    return "EcoGuard Model API is running!"

# Run the Flask app (only for local testing, Render will handle the deployment)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

