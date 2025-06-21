# Import library yang diperlukan
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Definisikan konstanta untuk ukuran gambar dan batch size

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_CHANNELS = 3 # RGB
BATCH_SIZE = 32
NUM_CLASSES = 4 # glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor

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
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE), # Pastikan minimal 1
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // BATCH_SIZE), # Pastikan minimal 1
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

    training_dir = os.path.join(base_path, 'Training')
    testing_dir = os.path.join(base_path, 'Testing')

    # 1. Mempersiapkan generator data
    print(f"Mempersiapkan generator data training dari: {training_dir}")
    print(f"Mempersiapkan generator data validasi dari: {testing_dir}")

        # Pastikan direktori training dan testing ada
    if not os.path.isdir(training_dir):
            print(f"Error: Direktori training tidak ditemukan di {training_dir}. Pastikan dataset telah diekstrak dengan benar.")
            return
    if not os.path.isdir(testing_dir):
            print(f"Error: Direktori testing tidak ditemukan di {testing_dir}. Pastikan dataset telah diekstrak dengan benar.")
            return
    
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

    # Menggunakan flow_from_directory untuk memuat gambar dari direktori
    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        testing_dir,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # 2. Membuat model
    print("Membuat model...")
    model = create_model((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS), NUM_CLASSES)
    model.summary()

    # 3. Melatih dan mengevaluasi model
    print("Melatih dan mengevaluasi model...")
    trained_model = train_and_evaluate_model(model, train_generator, validation_generator, epochs=20)

    # Path untuk menyimpan model
    model_save_path = os.path.join(base_path, 'brain_tumor_model.h5')

    # Menyimpan model yang telah dilatih
    print(f"Menyimpan model ke: {model_save_path}")
    trained_model.save(model_save_path)
    print("Model berhasil disimpan.")

if __name__ == '__main__':
    main()