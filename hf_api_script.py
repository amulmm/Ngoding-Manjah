from huggingface_hub import HfApi
import os

# Inisialisasi HfApi
api = HfApi()

# Path ke model yang ingin Anda unggah
model_path = "brain_tumor_model_baru.h5"

# Ganti 'your-username' dengan nama pengguna Hugging Face Anda yang sebenarnya
# Ganti 'your-model-name' dengan nama repositori yang Anda inginkan
repo_id = "amulmm/brain_tumor_training"

# Pastikan repositori ada. Buat jika belum ada.
try:
    api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
    print(f"Repositori {repo_id} sudah ada atau berhasil dibuat.")
except Exception as e:
    print(f"Terjadi kesalahan saat membuat atau memeriksa repositori: {e}")
    exit() # Keluar jika repositori tidak dapat dibuat atau diakses

print(f"Mengunggah model dari {model_path} ke {repo_id}...")

try:
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="brain_tumor_model_baru.h5",
        repo_id=repo_id,
        repo_type="model",
    )
    print("Model berhasil diunggah!")
except Exception as e:
    print(f"Terjadi kesalahan saat mengunggah model: {e}")