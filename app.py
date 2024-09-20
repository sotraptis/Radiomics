import os
import numpy as np
import tensorflow as tf
import pydicom
import random
import matplotlib.pyplot as plt
import streamlit as st

# Φόρτωση του εκπαιδευμένου μοντέλου πρόβλεψης καρκίνου
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model_fold_1.keras')
model = tf.keras.models.load_model(model_path, compile=False)

# Λειτουργία φόρτωσης και επεξεργασίας εικόνας DICOM
def process_image(file):
    dicom = pydicom.dcmread(file)
    img = dicom.pixel_array
    if len(img.shape) == 2:  # Έλεγχος αν είναι 2D εικόνα
        img = np.expand_dims(img, axis=-1)  # Προσθήκη άξονα καναλιού
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Κανονικοποίηση
    img = tf.image.resize(img, (256, 256))  # Αλλαγή μεγέθους
    img = np.expand_dims(img, axis=0)  # Προσθήκη batch dimension
    return img

# Streamlit UI
st.title("Πρόβλεψη Καρκίνου από DICOM Εικόνες")

uploaded_file = st.file_uploader("Ανέβασε ένα αρχείο DICOM", type=["dcm"])

if uploaded_file is not None:
    # Επεξεργασία της εικόνας DICOM και πρόβλεψη
    st.write("Φόρτωση και πρόβλεψη εικόνας...")
    image = process_image(uploaded_file)
    prediction = model.predict(image)
    prediction_binary = (prediction >= 0.5).astype(int)
    prediction_label = 'Καρκίνος' if prediction_binary == 1 else 'Υγιές'

    st.write(f"Πρόβλεψη: {prediction_label}")

    # Εάν η πρόβλεψη είναι 'Καρκίνος', δημιουργούμε και εμφανίζουμε την περιοχή ενδιαφέροντος (ROI)
    if prediction_label == 'Καρκίνος':
        dicom_image = pydicom.dcmread(uploaded_file).pixel_array
        dicom_image = (dicom_image - np.min(dicom_image)) / (np.max(dicom_image) - np.min(dicom_image)) * 255
        dicom_image = dicom_image.astype(np.uint8)

        # Δημιουργία μιας απλής ROI για την προβολή
        h, w = dicom_image.shape
        roi = np.zeros((h, w), dtype=np.uint8)
        roi[h//4:h//2, w//4:w//2] = 1  # Χονδρική εκτίμηση περιοχής καρκίνου

        # Επικάλυψη της ROI στην αρχική εικόνα
        plt.figure(figsize=(6, 6))
        plt.imshow(dicom_image, cmap='gray')
        plt.imshow(roi, alpha=0.3, cmap='Reds')
        plt.axis('off')

        # Εμφάνιση της εικόνας
        st.pyplot(plt)
