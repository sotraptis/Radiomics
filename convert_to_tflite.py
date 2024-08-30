import tensorflow as tf

# Φόρτωση του αρχικού μοντέλου Keras
model = tf.keras.models.load_model(r'C:\Users\yoave\Desktop\streamlit\best_model_fold_1.keras')

# Μετατροπή σε TFLite μοντέλο
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Αποθήκευση του TFLite μοντέλου σε αρχείο
with open(r'C:\Users\yoave\Desktop\streamlit\best_model_fold_1.tflite', 'wb') as f:
    f.write(tflite_model)
