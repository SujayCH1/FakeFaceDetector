import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

def load_anti_spoofing_model(model_path):
    try:
        model = load_model(model_path)
        print("Full model loaded successfully.")
    except:
        print("Failed to load full model. Attempting to load model with custom objects...")
        try:
            model = load_model(model_path, compile=False)
            print("Model loaded successfully with compile=False.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return model

def preprocess_frame(frame, target_size):
    frame = cv2.resize(frame, target_size)
    frame = frame.astype("float32") / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def draw_result(frame, prediction, face_coords=None):
    is_real = prediction > 0.08
    label = "Real" if is_real else "Fake"
    
    real_color = (0, 255, 0)  
    fake_color = (0, 0, 255)  
    text_color = (255, 255, 255)
    
    color = real_color if is_real else fake_color
    
    if face_coords is not None:
        x, y, w, h = face_coords
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label_position = (x, y - 10)
    else:
        label_position = (10, 30)
    
    cv2.rectangle(frame, (label_position[0], label_position[1] - 20), 
                  (label_position[0] + 180, label_position[1] + 20), (0, 0, 0), -1)
    
    cv2.putText(frame, f"{label}: {prediction:.2f}", label_position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    return frame

def real_time_anti_spoofing(model, img_height, img_width):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            processed_face = preprocess_frame(face, (img_width, img_height))

            prediction = model.predict(processed_face)[0][0]

            frame = draw_result(frame, prediction, (x, y, w, h))

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        cv2.imshow("Anti-Spoofing Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_height, img_width = 224, 224
    model_path = 'C:\\Users\\sujun\\Documents\\Projects\\Python\\PythonCV\\AntiSpoofing\\DATA\\MODEL\\anti_spoofing_model.h5'

    model = load_anti_spoofing_model(model_path)
    if model is not None:
        real_time_anti_spoofing(model, img_height, img_width)
    else:
        print("Failed to load the model. Please check the model path and ensure it's a valid Keras model.")