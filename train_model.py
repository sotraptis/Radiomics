
import os
import numpy as np
import pydicom
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Φόρτωση και προεπεξεργασία δεδομένων
def load_dicom_image(file_path, img_size=(128, 128)):
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array
    image = image / np.max(image)  # Κανονικοποίηση
    image = tf.image.resize(image, img_size)
    return image

def prepare_data(directory, img_size=(128, 128)):
    images = []
    labels = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".dcm"):
                img_path = os.path.join(subdir, file)
                image = load_dicom_image(img_path, img_size)
                images.append(image)
                # Υποθέτουμε ότι ο φάκελος ονομάζεται "cancer" ή "healthy"
                label = 1 if 'cancer' in subdir else 0
                labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Δημιουργία του μοντέλου CNN
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Φόρτωση δεδομένων
train_images, train_labels = prepare_data('datasets/train')
val_images, val_labels = prepare_data('datasets/validation')

# Δημιουργία και εκπαίδευση του μοντέλου
input_shape = (128, 128, 1)
model = create_cnn_model(input_shape)

# Εκπαίδευση του μοντέλου
model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Αποθήκευση του εκπαιδευμένου μοντέλου
model.save('models/lung_cancer_cnn_model.h5')
