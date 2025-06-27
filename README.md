# Brain Tumor Detection Model

This repository contains a deep learning model for brain tumor detection from MRI images. The model is built using TensorFlow/Keras and is designed to classify MRI scans into different categories: glioma, meningioma, notumor, pituitary, and non_brain.

## Model Details
- **Model Type**: Keras H5 model
- **Model File**: `brain_tumor_model_baru.h5`
- **Input Image Size**: 150x150 pixels
- **Classes**: `['glioma', 'meningioma', 'notumor', 'pituitary', 'non_brain']`

## How to Use (Local Inference)

1.  **Clone the repository**:
    ```bash
    git clone https://huggingface.co/amulmm/brain_tumor_training # Replace with your actual repo URL
    cd brain_tumor_training
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Flask application** (if applicable, based on `app.py`):
    ```bash
    python app.py
    ```
    Then navigate to `http://127.0.0.1:5000/` in your web browser.

## Hugging Face Inference API

You can use the Hugging Face Inference API to get predictions from this model. Here's an example Python script:

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/amulmm/brain_tumor_training" # Replace with your actual repo ID
headers = {"Authorization": "Bearer YOUR_HUGGING_FACE_API_TOKEN"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

# Example usage:
# output = query("path/to/your/image.jpg")
# print(output)
```

Replace `YOUR_HUGGING_FACE_API_TOKEN` with your actual Hugging Face API token and `path/to/your/image.jpg` with the path to your MRI image file.

## Training Data
The model was trained on a dataset of brain MRI images. (Add more details about your dataset if available, e.g., source, size, distribution of classes).

## License
(Specify your model's license here, e.g., MIT, Apache 2.0, etc.)