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
from sklearn.metrics import classification_report, confusion_matrix
import logging
import colorlog

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
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Definisikan konstanta untuk ukuran gambar dan batch size

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_CHANNELS = 3 # RGB
BATCH_SIZE = 20
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
    logger.info("Membuat model VGG16...")
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
    predictions = Dense(5, activation='softmax')(x)
    
    # Membuat model akhir
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Mengkompilasi model
    model.compile(optimizer=Adam(learning_rate=0.00005), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    logger.info("Model VGG16 berhasil dibuat.")
    return model

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
    # Menambahkan callbacks
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
    ) # <mcreference link="https://keras.io/api/models/model_training_apis/#fit-method" index="8">8</mcreference>
    logger.info("Training selesai.")
    
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
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    base_path = os.getcwd()

    # Mengumpulkan semua path gambar dan label dari direktori utama
    all_image_paths = []
    all_image_labels = []
    
    # Definisikan nama-nama kelas secara eksplisit
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary', 'non_brain']
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    NUM_CLASSES = 5
    print(f"Jumlah kelas yang terdeteksi: {NUM_CLASSES} ({class_names})")

    print("Mengumpulkan path gambar dan label dari direktori utama...")
    # Iterate through class_names, but exclude 'non_brain' for local file system scan
    # 'non_brain' will be populated by CIFAR-10 dataset later
    for class_name in [name for name in class_names if name != 'non_brain']:
        class_path_training = os.path.join(base_path, 'Training', class_name)
        class_path_testing = os.path.join(base_path, 'Testing', class_name)

        # Kumpulkan gambar dari direktori Training
        if os.path.isdir(class_path_training):
            for img_name in os.listdir(class_path_training):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_image_paths.append(os.path.join(class_path_training, img_name))
                    all_image_labels.append(class_to_idx[class_name])
        # Kumpulkan gambar dari direktori Testing
        if os.path.isdir(class_path_testing):
            for img_name in os.listdir(class_path_testing):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_image_paths.append(os.path.join(class_path_testing, img_name))
                    all_image_labels.append(class_to_idx[class_name])
        


    print(f"Total gambar yang dikumpulkan: {len(all_image_paths)}")
    # Verifikasi distribusi kelas
    from collections import Counter
    label_counts = Counter([class_names[label] for label in all_image_labels])
    print(f"Distribusi kelas: {label_counts}")

    # Verifikasi bahwa semua kelas yang diharapkan ada di all_image_labels
    unique_labels_collected = sorted(list(set(all_image_labels)))
    print(f"Label unik yang dikumpulkan (indeks): {unique_labels_collected}")
    if len(unique_labels_collected) != NUM_CLASSES:
        print(f"Peringatan: Jumlah label unik yang dikumpulkan ({len(unique_labels_collected)}) tidak sesuai dengan jumlah kelas yang diharapkan ({NUM_CLASSES}).")

    # Pisahkan data menjadi training dan validasi
    import cv2
    import tempfile
    from sklearn.model_selection import train_test_split
    # Add CIFAR-10 dataset
    from tensorflow.keras.datasets import cifar10
    (cifar_images, _), _ = cifar10.load_data()
    cifar_images = [cv2.resize(img, (150,150)) for img in cifar_images[:2000]]
    assert len(cifar_images) > 0, 'Tidak ada gambar CIFAR yang ditemukan'
    cifar_paths = ['cifar_'+str(i) for i in range(len(cifar_images))]
    cifar_labels = [4] * len(cifar_images)

    # Save CIFAR-10 images to temp directory
    cifar_dir = tempfile.mkdtemp()
    for i in range(len(cifar_images)):
        cv2.imwrite(
            os.path.join(cifar_dir, f'cifar_{i}.png'),
            cv2.cvtColor(cifar_images[i], cv2.COLOR_RGB2BGR)
        )

    # Get CIFAR image paths
    cifar_paths = [
        os.path.join(cifar_dir, f) 
        for f in os.listdir(cifar_dir)
        if f.endswith('.png')
    ]
    cifar_labels = [4] * len(cifar_paths)

    # Combine medical and CIFAR datasets
    combined_paths = all_image_paths + cifar_paths
    combined_labels = all_image_labels + cifar_labels

    # Split combined dataset
    train_paths, val_paths, train_labels_idx, val_labels_idx = train_test_split(
        combined_paths, combined_labels, test_size=0.2, random_state=42, stratify=combined_labels
    )

    # Mengacak data training
    train_paths, train_labels_idx = shuffle(train_paths, train_labels_idx, random_state=42)

    # Konversi label ke one-hot encoding
    train_labels = tf.keras.utils.to_categorical(train_labels_idx, num_classes=len(class_names))
    val_labels = tf.keras.utils.to_categorical(val_labels_idx, num_classes=len(class_names))

    # Membuat DataFrame untuk generator
    train_df = pd.DataFrame({'path': train_paths, 'label': [class_names[idx] for idx in train_labels_idx]})
    val_df = pd.DataFrame({'path': val_paths, 'label': [class_names[idx] for idx in val_labels_idx]})
    logger.info("Train DataFrame head:\n%s" % train_df.head())
    logger.info("Validation DataFrame head:\n%s" % val_df.head())

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
        dataframe=train_df,
        x_col='path',
        y_col='label',
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        classes=class_names,
        validate_filenames=True
    )

    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='path',
        y_col='label',
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        classes=class_names
    )

    # Membuat dan melatih model
    model = create_model((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS), NUM_CLASSES)
    trained_model = train_and_evaluate_model(model, train_generator, validation_generator, epochs=20)

    # Menyimpan model yang sudah dilatih
    model.save('brain_tumor_model.h5')
    print("Model berhasil disimpan sebagai brain_tumor_model.h5")

if __name__ == "__main__":
    main()