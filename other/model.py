import tensorflow as tf
from tensorflow.keras import layers, models

# Define emotion labels (these must match the model)
EMOTION_CLASSES = ['angry','disgust','fear','happy','sad','surprise','neutral']

def build_emotion_cnn(input_shape=(48,48,1), num_classes=len(EMOTION_CLASSES)):
    model = models.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_trained_model(path='models/emotion_cnn.h5'):
    try:
        model = tf.keras.models.load_model(path)
        print(model.summary())
        print(f"✅ Loaded trained model from {path}")
        return model
    except Exception as e:
        print(f"⚠️ Could not load model from {path}: {e}")
        print("➡️ Creating a new untrained model instead (predictions will be random).")
        return build_emotion_cnn()
