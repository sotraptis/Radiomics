import os
import numpy as np
import tensorflow as tf
import pydicom
import streamlit as st

# Φόρτωση του εκπαιδευμένου μοντέλου TensorFlow Lite
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model_fold_1.tflite')

# Δημιουργία του TensorFlow Lite Interpreter
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Λήψη εισόδων και εξόδων από το TFLite μοντέλο
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Λειτουργία φόρτωσης και επεξεργασίας εικόνας DICOM
def process_image(file):
    dicom = pydicom.dcmread(file)
    img = dicom.pixel_array
    if len(img.shape) == 2:  # Έλεγχος αν είναι 2D εικόνα
        img = np.expand_dims(img, axis=-1)  # Προσθήκη άξονα καναλιού
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Κανονικοποίηση
    img = tf.image.resize(img, (256, 256))  # Αλλαγή μεγέθους
    img = np.expand_dims(img, axis=0)  # Προσθήκη batch dimension
    img = img.astype(np.float32)  # Μετατροπή σε float32 για το μοντέλο
    return img

# Λειτουργία για πρόβλεψη με το TFLite μοντέλο
def predict_cancer(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()  # Εκτέλεση της πρόβλεψης
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Streamlit UI
st.title("Πρόβλεψη Καρκίνου από DICOM Εικόνες")

uploaded_file = st.file_uploader("Ανέβασε ένα αρχείο DICOM", type=["dcm"])

if uploaded_file is not None:
    # Επεξεργασία της εικόνας DICOM και πρόβλεψη
    st.write("Φόρτωση και πρόβλεψη εικόνας...")
    image = process_image(uploaded_file)
    prediction = predict_cancer(image)
    prediction_binary = (prediction >= 0.5).astype(int)
    prediction_label = 'Καρκίνος' if prediction_binary == 1 else 'Υγιές'

    st.write(f"Πρόβλεψη: {prediction_label}")

    # Εάν η πρόβλεψη είναι 'Καρκίνος', εμφανίζουμε την εικόνα DICOM
    if prediction_label == 'Καρκίνος':
        dicom_image = pydicom.dcmread(uploaded_file).pixel_array
        dicom_image = (dicom_image - np.min(dicom_image)) / (np.max(dicom_image) - np.min(dicom_image)) * 255
        dicom_image = dicom_image.astype(np.uint8)

        # Εμφάνιση της εικόνας
        st.image(dicom_image, caption="Εικόνα με Καρκίνο", use_column_width=True)
