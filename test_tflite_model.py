import tensorflow as tf

def load_tflite_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

model_path = r'C:\Users\yoave\Desktop\streamlit - Αντιγραφή\best_model_fold_1.tflite'
load_tflite_model(model_path)
