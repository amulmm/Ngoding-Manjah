from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, url_for
import os
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Define model path and Hugging Face Hub details
MODEL_PATH = "/tmp/brain_tumor_model_baru.h5"
HF_REPO_ID = "amulmm/brain_tumor_training" # Ganti dengan repo_id Anda yang sebenarnya
HF_FILENAME = "brain_tumor_model_baru.h5"

# Muat model hanya sekali saat fungsi diinisialisasi (cold start)
if not os.path.exists(MODEL_PATH):
    print("Mengunduh model dari Hugging Face Hub...")
    try:
        hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, local_dir="/tmp/")
        print("Model berhasil diunduh.")
    except Exception as e:
        print(f"Terjadi kesalahan saat mengunduh model: {e}")
        # Handle error, maybe exit or raise an exception
        exit() # Keluar jika model tidak dapat diunduh

# Load the trained model
model = load_model(MODEL_PATH)

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

            # Make prediction
            predictions = model.predict(img_array)
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