# create_toy_model.py
from model import build_emotion_cnn
import os

os.makedirs('models', exist_ok=True)
model = build_emotion_cnn()
# Save untrained model weights (this allows the web app to load a model file)
model.save('models/emotion_cnn.h5')
print("Saved toy model to models/emotion_cnn.h5")
