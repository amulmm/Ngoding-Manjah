import tensorflow as tf
import tensorflow as tf
from transformers import AutoConfig, TFAutoModelForImageClassification, AutoImageProcessor
from collections import Counter
import os
import logging

# Konfigurasi logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Asumsikan Anda sudah melatih model dan menyimpannya di 'brain_tumor_model_baru.h5'
# Ganti dengan path model yang sudah Anda latih
model_path = './brain_tumor_model_baru.h5' 

# Muat model Keras yang sudah dilatih
loaded_keras_model = tf.keras.models.load_model(model_path)

# Definisikan CLASS_NAMES dan ID ke Label (penting untuk konfigurasi Hugging Face)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary', 'non_brain']
id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
label2id = {name: i for i, name in enumerate(CLASS_NAMES)}

# Buat instance image processor dasar atau yang sesuai dengan preprocessing Anda
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", size={"height": 150, "width": 150}) # Menggunakan model pre-trained sebagai referensi untuk AutoImageProcessor

# Buat direktori sementara untuk menyimpan model Hugging Face
hf_model_dir = "brain_tumor_hf_model"
os.makedirs(hf_model_dir, exist_ok=True)

# Simpan image processor
image_processor.save_pretrained(hf_model_dir)

# Simpan model Keras dalam format SavedModel (direkomendasikan TensorFlow)
keras_saved_model_path = os.path.join(hf_model_dir, "saved_model.keras")
loaded_keras_model.save(keras_saved_model_path)
logger.info(f"Model Keras SavedModel disimpan di: {keras_saved_model_path}")

# Unggah juga file .h5 asli ke repositori Hugging Face
import shutil
shutil.copy(model_path, hf_model_dir)
logger.info(f"File model .h5 asli disalin ke: {hf_model_dir}")
logger.info(f"Model Keras SavedModel disimpan di: {keras_saved_model_path}")

# Unggah ke Hugging Face Hub
from huggingface_hub import HfApi, create_repo

# Pastikan Anda sudah login via `huggingface-cli login` di terminal Anda
# Atau lewat kode: from huggingface_hub import login; login(token="hf_YOUR_TOKEN")

repo_id = "amulmm/brain_tumor_training" # ID repositori Anda di Hugging Face

api = HfApi()

# Unggah folder `hf_model_dir` (yang berisi `preprocessor_config.json` dan `saved_model`)
api.upload_folder(
    folder_path=hf_model_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Add TensorFlow SavedModel and image processor config"
)
logger.info(f"Model dan preprocessor berhasil diunggah ke Hugging Face Hub: {repo_id}")