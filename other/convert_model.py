import tensorflow as tf
from tensorflow.keras.models import load_model

old_model_path = r"C:\Users\SHEDRACK ACHILE\Desktop\Achile.23CG034016\models\emotion_cnn.h5"
new_model_path = r"C:\Users\SHEDRACK ACHILE\Desktop\Achile.23CG034016\models\emotion_cnn_converted.keras"

# Try to load with legacy format handling
model = tf.keras.models.load_model(old_model_path, compile=False)
model.save(new_model_path)
print("✅ Model converted successfully to:", new_model_path)
