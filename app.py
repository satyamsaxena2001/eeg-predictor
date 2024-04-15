import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the saved model
loaded_model = load_model('nueral_network.h5')

# Specify the directory to save uploaded Excel files
UPLOAD_FOLDER = 'uploaded_files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def preprocess_eeg_data(data):
    # Assuming 'data' is a DataFrame containing EEG data (without labels)
    scaler = StandardScaler()
    processed_data = scaler.fit_transform(data)
    return processed_data


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # If the user does not select a file, the browser may send an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Save the uploaded Excel file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Read the uploaded Excel file
        data = pd.read_excel(file_path)  # Adjust this based on the file type you expect (e.g., CSV, Excel)

        # Preprocess the EEG data
        processed_data = preprocess_eeg_data(data)

        # Make prediction using the pre-trained model
        predictions = loaded_model.predict(processed_data)

        # Process the predictions to determine the emotion labels
        predicted_labels = np.argmax(predictions, axis=1)
        emotion_labels = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}  # Define your label mapping here

        predicted_emotions = [emotion_labels[label] for label in predicted_labels]

        return jsonify({'predictions': predicted_emotions})

    except Exception as e:
        print(e)
        return jsonify({'error': 'Error processing uploaded file'})


if __name__ == "__main__":
    app.run(debug=True)
