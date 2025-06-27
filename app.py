from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, url_for
import os
import io
from huggingface_hub import hf_hub_download # Import hf_hub_download

# Available backend options are: "jax", "torch", "tensorflow".
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

app = Flask(__name__)

# Define the Hugging Face model repository ID and the specific file to download
# HF_MODEL_REPO_ID = "amulmm/brain_tumor_training"
# HF_MODEL_FILE = "saved_model.keras"

# Download the model file
# model_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=HF_MODEL_FILE)

# Load the model directly from the downloaded .keras file
model = keras.models.load_model("saved_model.keras")

# Define image dimensions
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Define class names (assuming 4 classes as per cobacoba.py)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary', 'non_brain']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction='No selected file')
        if file:
            # Save the uploaded file temporarily
            filepath = os.path.join('uploads', file.filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(filepath)

            # Preprocess the image
            img = image.load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Create a batch
            img_array /= 255.0 # Normalize pixel values

            # Make prediction using the loaded model
            predictions = model.predict(img_array)
            
            # Proses hasil prediksi
            predicted_class_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_index] * 100

            # Set a confidence threshold
            CONFIDENCE_THRESHOLD = 80.0 # Adjust as needed

            if confidence < CONFIDENCE_THRESHOLD:
                predicted_class_name = "Unknown (Not a brain tumor image or low confidence)"
            else:
                predicted_class_name = CLASS_NAMES[predicted_class_index]

            # Clean up temporary file
            os.remove(filepath)

            return render_template('index.html', prediction=f'Prediction: {predicted_class_name} (Confidence: {confidence:.2f}%)')
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)