import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import logging
import colorlog
import cv2
import tempfile
import shutil
from tensorflow.keras.datasets import cifar10

# Konfigurasi logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s',
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={
        'message':  {
            'ERROR':    'red',
            'CRITICAL': 'red'
        }
    }
))

logger = colorlog.getLogger()
logger.addHandler(handler);
logger.setLevel(logging.INFO)

# Definisikan konstanta untuk ukuran gambar dan batch size

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_CHANNELS = 3 # RGB
BATCH_SIZE = 20

def create_model(input_shape, num_classes):
    """Fungsi untuk membuat model CNN menggunakan transfer learning dengan VGG16.
    
    Args:
        input_shape (tuple): Bentuk data input gambar (height, width, channels)
        num_classes (int): Jumlah kelas output
        
    Returns:
        tf.keras.Model: Model TensorFlow yang sudah dikompilasi
    """
    logger.info("Membuat model VGG16...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-4:]:
        layer.trainable = True
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.00005), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    logger.info("Model VGG16 berhasil dibuat.")
    return model

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_and_evaluate_model(model, train_generator, validation_generator, epochs=10):
    """Fungsi untuk melatih dan mengevaluasi model menggunakan generator data.
    
    Args:
        model (tf.keras.Model): Model TensorFlow yang sudah dikompilasi
        train_generator: Generator data untuk training
        validation_generator: Generator data untuk validasi
        epochs (int): Jumlah epoch untuk training
        
    Returns:
        tf.keras.Model: Model yang sudah dilatih
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    ]

    logger.info("Mulai training model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        verbose=1,
        callbacks=callbacks
    )
    logger.info("Training selesai.")
    
    print("Mengevaluasi model...")
    loss, accuracy = model.evaluate(validation_generator, steps=max(1, validation_generator.samples // BATCH_SIZE), verbose=0)
    print(f"Loss pada data validasi: {loss:.4f}")
    print(f"Akurasi pada data validasi: {accuracy:.4f}")
    
    return model

def main():
    """Fungsi utama untuk menjalankan alur kerja AI deteksi tumor otak.
    """
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    base_path = os.getcwd()

    all_image_paths = []
    all_image_labels = []
    
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary', 'non_brain']
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    NUM_CLASSES = len(class_names)
    print(f"Jumlah kelas yang terdeteksi: {NUM_CLASSES} ({class_names})")

    print("Mengumpulkan path gambar dan label dari direktori utama...")
    for class_name in class_names:
        if class_name == 'non_brain':
            continue

        class_path_training = os.path.join(base_path, 'Training', class_name)
        class_path_testing = os.path.join(base_path, 'Testing', class_name)

        if os.path.isdir(class_path_training):
            for img_name in os.listdir(class_path_training):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_image_paths.append(os.path.join(class_path_training, img_name))
                    all_image_labels.append(class_to_idx[class_name])
        if os.path.isdir(class_path_testing):
            for img_name in os.listdir(class_path_testing):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_image_paths.append(os.path.join(class_path_testing, img_name))
                    all_image_labels.append(class_to_idx[class_name])
        
    print(f"Total gambar yang dikumpulkan: {len(all_image_paths)}")
    from collections import Counter
    label_counts = Counter([class_names[label] for label in all_image_labels])
    print(f"Distribusi kelas: {label_counts}")

    unique_labels_collected = sorted(list(set(all_image_labels)))
    print(f"Label unik yang dikumpulkan (indeks): {unique_labels_collected}")
    if len(unique_labels_collected) != NUM_CLASSES - 1: # -1 karena non_brain belum masuk
        print(f"Peringatan: Jumlah label unik yang dikumpulkan ({len(unique_labels_collected)}) tidak sesuai dengan jumlah kelas yang diharapkan ({NUM_CLASSES - 1}).")

    # Tambahkan dataset gambar umum lainnya untuk kelas 'non_brain' dari folder Training
    general_images_training_path = os.path.join(base_path, 'Training', 'non_brain')
    if os.path.isdir(general_images_training_path):
        print(f"Mengumpulkan gambar umum dari: {general_images_training_path}")
        for img_name in os.listdir(general_images_training_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(general_images_training_path, img_name))
                all_image_labels.append(class_to_idx['non_brain'])
    else:
        print(f"Peringatan: Direktori gambar umum tidak ditemukan di {general_images_training_path}.")

    # Tambahkan dataset gambar umum lainnya untuk kelas 'non_brain' dari folder Testing
    general_images_testing_path = os.path.join(base_path, 'Testing', 'non_brain')
    if os.path.isdir(general_images_testing_path):
        print(f"Mengumpulkan gambar umum dari: {general_images_testing_path}")
        for img_name in os.listdir(general_images_testing_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(general_images_testing_path, img_name))
                all_image_labels.append(class_to_idx['non_brain'])
    else:
        print(f"Peringatan: Direktori gambar umum tidak ditemukan di {general_images_testing_path}.")

    data_df = pd.DataFrame({'path': all_image_paths, 'label': all_image_labels})
    data_df['label'] = data_df['label'].astype(str)

    train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42, stratify=data_df['label'])

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col='label',
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=[str(i) for i in range(NUM_CLASSES)],
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='path',
        y_col='label',
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=[str(i) for i in range(NUM_CLASSES)],
        shuffle=False
    )

    model = create_model((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), NUM_CLASSES)
    trained_model = train_and_evaluate_model(model, train_generator, validation_generator, epochs=10)

    model_save_path = os.path.join(base_path, 'brain_tumor_model_baru.h5')
    trained_model.save(model_save_path)
    logger.info(f"Model berhasil disimpan di: {model_save_path}")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        logger.info(f"Direktori sementara {temp_dir} berhasil dihapus.")

    print("Proses pelatihan selesai.")

if __name__ == '__main__':
    main()