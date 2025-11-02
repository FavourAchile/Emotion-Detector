import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Path to model
MODEL_PATH = r"models/emotion_cnn.h5"

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def build_model():
    """Define CNN structure compatible with FER-like datasets"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(EMOTIONS), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Try loading the model
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Loaded model: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Could not load model ({e}). Creating a new untrained model instead.")
    model = build_model()

def predict_emotion(img):
    """Preprocess and predict emotion from a BGR OpenCV image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face / 255.0
    face = np.expand_dims(face, axis=(0, -1))  # Shape: (1, 48, 48, 1)
    
    preds = model.predict(face)
    emotion = EMOTIONS[np.argmax(preds)]
    return {'emotion': emotion, 'probabilities': preds.tolist()}
