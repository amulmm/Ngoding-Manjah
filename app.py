from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Path to the trained model (assuming it's saved in the same directory)
MODEL_PATH = 'brain_tumor_model.h5'

# Load the trained model
model = load_model(MODEL_PATH)

# Define image dimensions
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Define class names (assuming 4 classes as per cobacoba.py)
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

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
            CONFIDENCE_THRESHOLD = 95.0 # Adjust as needed

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