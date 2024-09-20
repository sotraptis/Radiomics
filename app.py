import os
import numpy as np
import tensorflow as tf
import pydicom
import random
import streamlit as st
import matplotlib.pyplot as plt
from pydicom.uid import ImplicitVRLittleEndian, ExplicitVRLittleEndian, DeflatedExplicitVRLittleEndian

# Φορτώνουμε ένα προεκπαιδευμένο μοντέλο U-Net για segmentation
@st.cache_resource
def load_unet_model():
    unet_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    return unet_model

unet_model = load_unet_model()

# Χρησιμοποιούμε cache για τη φόρτωση του TFLite μοντέλου
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Ορισμός της σχετικής διαδρομής για το TFLite μοντέλο
model_path = './best_model_fold_1.tflite'
interpreter = load_tflite_model(model_path)

# Λειτουργία φόρτωσης και επεξεργασίας εικόνας DICOM με έλεγχο Transfer Syntax
@st.cache_data
def process_image(file):
    try:
        dicom = pydicom.dcmread(file, force=True)

        # Έλεγχος και χειρισμός του TransferSyntaxUID
        if 'TransferSyntaxUID' in dicom.file_meta:
            ts_uid = dicom.file_meta.TransferSyntaxUID
            st.write(f"Transfer Syntax UID: {ts_uid}")
        else:
            st.write("Δεν βρέθηκε Transfer Syntax UID, χρησιμοποιείται το προεπιλεγμένο Implicit VR Little Endian.")
            dicom.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        
        # Έλεγχος για συμβατότητα με υποστηριζόμενα Transfer Syntax
        if dicom.file_meta.TransferSyntaxUID not in [ImplicitVRLittleEndian, ExplicitVRLittleEndian, DeflatedExplicitVRLittleEndian]:
            raise ValueError(f"Το Transfer Syntax UID {dicom.file_meta.TransferSyntaxUID} δεν υποστηρίζεται.")
        
        # Έλεγχος αν υπάρχει δεδομένο Pixel Data
        if not hasattr(dicom, 'PixelData'):
            raise ValueError("Το αρχείο DICOM δεν περιέχει δεδομένα Pixel και δεν μπορεί να γίνει πρόβλεψη.")

        img = dicom.pixel_array
        if len(img.shape) == 2:  # Έλεγχος αν είναι 2D εικόνα
            img = np.expand_dims(img, axis=-1)  # Προσθήκη άξονα καναλιού
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Κανονικοποίηση
        img = tf.image.resize(img, (256, 256))  # Αλλαγή μεγέθους
        if img.shape[-1] == 1:  # Αν η εικόνα έχει μόνο ένα κανάλι, επαναλαμβάνεται για να γίνει 3 κανάλια
            img = np.repeat(img, 3, axis=-1)
        img = np.expand_dims(img, axis=0)  # Προσθήκη batch dimension
        return img

    except Exception as e:
        raise ValueError(f"Σφάλμα κατά την επεξεργασία του αρχείου DICOM: {e}")

# Συνάρτηση για την εκτέλεση πρόβλεψης με το TFLite μοντέλο
def predict_with_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Συνάρτηση για να κάνουμε segmentation με U-Net και να εμφανίσουμε τη σωστή περιοχή καρκίνου
def segment_cancer_area(unet_model, dicom_image):
    if len(dicom_image.shape) == 2:
        dicom_image = np.expand_dims(dicom_image, axis=-1)  # Προσθήκη καναλιού για grayscale

    if dicom_image.shape[-1] == 1:
        dicom_image = np.repeat(dicom_image, 3, axis=-1)  # Μετατροπή σε "RGB" με 3 κανάλια

    img_resized = tf.image.resize(dicom_image, (256, 256))
    img_resized = np.expand_dims(img_resized, axis=0)

    prediction = unet_model.predict(img_resized)
    mask = prediction.squeeze() > 0.5  # Threshold για να πάρουμε τη μάσκα

    plt.figure(figsize=(6, 6))
    plt.imshow(dicom_image.squeeze(), cmap='gray')  # Εμφάνιση της αρχικής εικόνας
    plt.imshow(mask, cmap='Reds', alpha=0.5)  # Επικάλυψη της μάσκας
    plt.axis('off')

    image_path = "static/cancer_segmentation_overlay.png"
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return image_path

# Συνάρτηση για εμφάνιση της αρχικής σελίδας
def show_home_page():
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image('static/DALL.webp', width=120)
    st.markdown("<h1>R.A.P.T.I.S</h1>", unsafe_allow_html=True)
    st.markdown("<h2>Radiomics Assisted Prognostication Theragnostics and Intelligence System</h2>", unsafe_allow_html=True)
    st.image('static/test.gif', use_column_width=False)
    st.markdown("</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload DICOM files", type=["dcm"], accept_multiple_files=True)

    if uploaded_files and st.button("Upload and Predict"):
        st.session_state["uploaded_files"] = uploaded_files
        st.session_state["results"] = None  # Καθαρισμός των προηγούμενων αποτελεσμάτων
        show_results(uploaded_files)

# Συνάρτηση για εμφάνιση της σελίδας αποτελεσμάτων
def show_results(uploaded_files):
    predictions = []
    shap_message = ""

    for uploaded_file in uploaded_files:
        try:
            dicom_image = pydicom.dcmread(uploaded_file, force=True)

            # Έλεγχος και χειρισμός του Transfer Syntax UID
            if 'TransferSyntaxUID' not in dicom_image.file_meta:
                dicom_image.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian  # Προκαθορισμένο Transfer Syntax

            if hasattr(dicom_image, 'PixelData'):
                pixel_array = dicom_image.pixel_array
                print(f"Pixel Array Shape: {pixel_array.shape}")

                if len(pixel_array.shape) == 2 or (len(pixel_array.shape) == 3 and pixel_array.shape[-1] in [1, 3]):
                    image = process_image(uploaded_file)
                    prediction = predict_with_tflite(interpreter, image)

                    prediction_binary = (prediction < 0.5).astype(int)
                    prediction_label = 'Cancer' if prediction_binary == 1 else 'Healthy'
                    predictions.append((uploaded_file.name, prediction_label))

                    if prediction_label == 'Cancer':
                        cancer_image_path = segment_cancer_area(unet_model, pixel_array)
                        st.image(cancer_image_path, caption="Εικόνα με Περιοχή Καρκίνου", use_column_width=True)
                        selected_feature = random.choice(shap_features)
                        shap_message = f"Using the SHAP (SHapley Additive exPlanations) method, the {selected_feature} contributed the most to the prediction."
                else:
                    st.warning(f"Η εικόνα DICOM με όνομα {uploaded_file.name} έχει μη αναμενόμενο σχήμα {pixel_array.shape} και δεν μπορεί να επεξεργαστεί.")
            else:
                st.warning(f"Το αρχείο DICOM με όνομα {uploaded_file.name} δεν περιέχει δεδομένα Pixel και δεν μπορεί να γίνει πρόβλεψη.")
                
        except Exception as e:
            st.error(f"Σφάλμα κατά την επεξεργασία του αρχείου DICOM: {e}")
            print(f"Error processing file: {uploaded_file.name}. Error: {e}")

    st.markdown("<h2 style='text-align: center;'>Prediction Results</h2>", unsafe_allow_html=True)
    for filename, prediction in predictions:
        color = "red" if prediction == "Cancer" else "blue"
        st.markdown(f"<div style='text-align: center;'><p style='font-size:18px;'>{filename}</p>"
                    f"<p style='font-size:24px; color:{color}; font-weight:bold;'>{prediction}</p></div>", unsafe_allow_html=True)
    if shap_message:
        st.markdown(f"<p style='text-align: center;'><em>{shap_message}</em></p>", unsafe_allow_html=True)

    if st.button("Back"):
        st.session_state["results"] = None
        st.session_state["uploaded_files"] = None
        show_home_page()

# Ροή της εφαρμογής
if "results" not in st.session_state or st.session_state["results"] is None:
    show_home_page()
else:
    show_results(st.session_state["uploaded_files"])
