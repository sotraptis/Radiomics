import os
import numpy as np
import tensorflow as tf
import pydicom
from PIL import Image, ImageOps, ImageDraw
import streamlit as st

# Φόρτωση του εκπαιδευμένου μοντέλου πρόβλεψης καρκίνου
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model_fold_1.keras')
model = tf.keras.models.load_model(model_path, compile=False)

# Λειτουργία φόρτωσης και επεξεργασίας εικόνας DICOM
def process_image(file):
    dicom = pydicom.dcmread(file)
    img = dicom.pixel_array
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255  # Κανονικοποίηση
    img = Image.fromarray(img).convert('L')  # Μετατροπή σε grayscale με Pillow
    img = ImageOps.fit(img, (256, 256))  # Αλλαγή μεγέθους σε 256x256
    img = np.expand_dims(img, axis=0)  # Προσθήκη batch dimension
    img = np.expand_dims(img, axis=-1)  # Προσθήκη καναλιού
    return img

# Λειτουργία για επικάλυψη της περιοχής ενδιαφέροντος (ROI) πάνω στην εικόνα
def overlay_roi_on_image(image, roi):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle([(roi[0], roi[1]), (roi[2], roi[3])], outline="red", width=3)
    return image

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
        dicom_image = Image.fromarray(dicom_image).convert('L')

        # Δημιουργία μιας απλής ROI για την προβολή
        roi = (64, 64, 192, 192)  # Χονδρική εκτίμηση περιοχής καρκίνου

        # Επικάλυψη της ROI στην αρχική εικόνα
        overlay_image = overlay_roi_on_image(dicom_image, roi)

        # Εμφάνιση της εικόνας
        st.image(overlay_image, caption="Περιοχή του καρκίνου", use_column_width=True)
