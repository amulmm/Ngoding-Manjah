# Import library yang diperlukan
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

# Definisikan konstanta untuk ukuran gambar dan batch size

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_CHANNELS = 3 # RGB
BATCH_SIZE = 32
# NUM_CLASSES akan ditentukan secara dinamis setelah mengumpulkan class_names

# Fungsi load_and_prepare_data tidak lagi diperlukan karena logika spesifik dataset ada di main.

def create_model(input_shape, num_classes):
    """Fungsi untuk membuat model CNN menggunakan transfer learning dengan VGG16.
    
    Args:
        input_shape (tuple): Bentuk data input gambar (height, width, channels)
        num_classes (int): Jumlah kelas output
        
    Returns:
        tf.keras.Model: Model TensorFlow yang sudah dikompilasi
    """
    # Memuat model VGG16 yang sudah dilatih sebelumnya di ImageNet
    # include_top=False: tidak menyertakan lapisan klasifikasi teratas
    # weights='imagenet': menggunakan bobot yang dilatih di dataset ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Membekukan lapisan dasar VGG16 agar tidak dilatih ulang
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze beberapa lapisan teratas dari base_model untuk fine-tuning
    # Misalnya, unfreeze 4 lapisan terakhir
    for layer in base_model.layers[-4:]:
        layer.trainable = True
        
    # Menambahkan lapisan kustom di atas model dasar
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Membuat model akhir
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Mengkompilasi model
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, train_generator, validation_generator, epochs=20):
    """Fungsi untuk melatih dan mengevaluasi model menggunakan generator data.
    
    Args:
        model (tf.keras.Model): Model TensorFlow yang sudah dikompilasi
        train_generator: Generator data untuk training
        validation_generator: Generator data untuk validasi
        epochs (int): Jumlah epoch untuk training
        
    Returns:
        tf.keras.Model: Model yang sudah dilatih
    """
    # Melatih model menggunakan data dari generator
    # steps_per_epoch: Jumlah batch yang diambil dari generator per epoch
    # validation_steps: Jumlah batch yang diambil dari generator validasi per epoch
    print("Mulai training model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        verbose=1
    ) # <mcreference link="https://keras.io/api/models/model_training_apis/#fit-method" index="8">8</mcreference>
    print("Training selesai.")
    
    # Mengevaluasi model pada data validasi
    # Ini memberikan gambaran seberapa baik model akan bekerja pada data baru yang belum pernah dilihat
    print("Mengevaluasi model...")
    loss, accuracy = model.evaluate(validation_generator, steps=max(1, validation_generator.samples // BATCH_SIZE), verbose=0) # <mcreference link="https://keras.io/api/models/model_training_apis/#evaluate-method" index="9">9</mcreference>
    print(f"Loss pada data validasi: {loss:.4f}")
    print(f"Akurasi pada data validasi: {accuracy:.4f}")
    
    return model

def main():
    """Fungsi utama untuk menjalankan alur kerja AI deteksi tumor otak.
    """
    try:
        base_path = os.path.dirname(__file__)
    except NameError:
        base_path = os.getcwd()

    data_dir = os.path.join(base_path, 'Training') # Menggunakan Training sebagai sumber data utama
    test_dir = os.path.join(base_path, 'Testing') # Data testing terpisah untuk evaluasi akhir

    # Pastikan direktori data ada
    if not os.path.isdir(data_dir):
        print(f"Error: Direktori data training tidak ditemukan di {data_dir}. Pastikan dataset telah diekstrak dengan benar.")
        return
    if not os.path.isdir(test_dir):
        print(f"Error: Direktori data testing tidak ditemukan di {test_dir}. Pastikan dataset telah diekstrak dengan benar.")
        return

    # Mengumpulkan semua path gambar dan label dari direktori training dan testing
    all_image_paths = []
    all_image_labels = []
    class_names = sorted(os.listdir(data_dir)) # Asumsi nama folder adalah nama kelas
    # Tambahkan 'non_brain' secara eksplisit jika belum ada, atau pastikan direktori 'non_brain' ada di Training
    if 'non_brain' not in class_names:
        class_names.append('non_brain')
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # Pastikan direktori 'non_brain' ada di Training dan Testing
    for class_name in ['non_brain']:
        train_class_path = os.path.join(data_dir, class_name)
        test_class_path = os.path.join(test_dir, class_name)
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)
        print(f"Pastikan Anda telah menambahkan gambar non-otak ke direktori: {train_class_path} dan {test_class_path}")

    NUM_CLASSES = len(class_names) # Perbarui NUM_CLASSES setelah semua kelas terkumpul
    print(f"Jumlah kelas yang terdeteksi: {NUM_CLASSES} ({class_names})")

    print("Mengumpulkan path gambar dan label dari direktori Training...")
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_image_paths.append(os.path.join(class_path, img_name))
                    all_image_labels.append(class_to_idx[class_name])
    
    print("Mengumpulkan path gambar dan label dari direktori Testing...")
    test_image_paths = []
    test_image_labels = []
    for class_name in class_names:
        class_path = os.path.join(test_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    test_image_paths.append(os.path.join(class_path, img_name))
                    test_image_labels.append(class_to_idx[class_name])

    # Mengacak data
    all_image_paths, all_image_labels = shuffle(all_image_paths, all_image_labels, random_state=42)

    # Konversi label ke one-hot encoding
    # Konversi label ke one-hot encoding, pastikan num_classes sesuai dengan jumlah kelas yang sebenarnya
    all_image_labels = tf.keras.utils.to_categorical(all_image_labels, num_classes=len(class_names))
    test_image_labels = tf.keras.utils.to_categorical(test_image_labels, num_classes=len(class_names))

    # Inisialisasi K-Fold Cross-Validation
    n_splits = 5 # Jumlah lipatan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_losses = []
    best_model = None
    best_accuracy = 0.0

    print(f"Memulai {n_splits}-Fold Cross-Validation...")
    for fold, (train_index, val_index) in enumerate(kf.split(all_image_paths)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        train_fold_paths = np.array(all_image_paths)[train_index]
        train_fold_labels = all_image_labels[train_index]
        val_fold_paths = np.array(all_image_paths)[val_index]
        val_fold_labels = all_image_labels[val_index]

        # Membuat ImageDataGenerator untuk training (dengan augmentasi)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Membuat ImageDataGenerator untuk validasi (hanya rescale)
        validation_datagen = ImageDataGenerator(rescale=1./255)

        # Menggunakan flow_from_dataframe untuk memuat gambar dari path yang sudah dikumpulkan
        train_generator = train_datagen.flow_from_dataframe(
            pd.DataFrame({'filepaths': train_fold_paths, 'labels': [class_names[np.argmax(l)] for l in train_fold_labels]}),
            x_col='filepaths',
            y_col='labels',
            target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            classes=class_names # Pastikan generator mengetahui semua kelas yang mungkin
        )

        validation_generator = validation_datagen.flow_from_dataframe(
            pd.DataFrame({'filepaths': val_fold_paths, 'labels': [class_names[np.argmax(l)] for l in val_fold_labels]}),
            x_col='filepaths',
            y_col='labels',
            target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False,
            classes=class_names # Pastikan generator mengetahui semua kelas yang mungkin
        )

        # Membuat model baru untuk setiap fold
        model = create_model((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS), NUM_CLASSES)
        model.summary()

        # Melatih dan mengevaluasi model untuk fold saat ini
        trained_model_fold = train_and_evaluate_model(model, train_generator, validation_generator, epochs=20)
        
        # Evaluasi pada data validasi fold
        loss, accuracy = trained_model_fold.evaluate(validation_generator, verbose=0)
        print(f"Fold {fold+1} - Loss Validasi: {loss:.4f}, Akurasi Validasi: {accuracy:.4f}")
        fold_losses.append(loss)
        fold_accuracies.append(accuracy)

        # Simpan model terbaik berdasarkan akurasi validasi
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = trained_model_fold

    print("\n--- Hasil Cross-Validation ---")
    print(f"Rata-rata Akurasi Validasi: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    print(f"Rata-rata Loss Validasi: {np.mean(fold_losses):.4f} (+/- {np.std(fold_losses):.4f})")

    # Simpan model terbaik secara keseluruhan
    if best_model:
        model_save_path = os.path.join(base_path, 'brain_tumor_model_best_cv.h5')
        print(f"Menyimpan model terbaik ke: {model_save_path}")
        best_model.save(model_save_path)
        print("Model terbaik berhasil disimpan.")
    else:
        print("Tidak ada model terbaik yang ditemukan.")

    # Evaluasi akhir pada data testing yang terpisah (jika ada)
    if test_image_paths:
        print("\n--- Evaluasi Akhir pada Data Testing Terpisah ---")
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_dataframe(
            pd.DataFrame({'filepaths': test_image_paths, 'labels': [class_names[np.argmax(l)] for l in test_image_labels]}),
            x_col='filepaths',
            y_col='labels',
            target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        if best_model:
            test_loss, test_accuracy = best_model.evaluate(test_generator, verbose=0)
            print(f"Loss pada data testing: {test_loss:.4f}")
            print(f"Akurasi pada data testing: {test_accuracy:.4f}")
        else:
            print("Tidak ada model terbaik untuk dievaluasi pada data testing.")

if __name__ == '__main__':
    main()