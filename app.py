from flask import Flask, request, render_template, jsonify
#from fer import FER
import cv2
import numpy as np
import os

app = Flask(__name__)

# Initialize the pretrained FER detector
detector = FER(mtcnn=True)  # mtcnn=True for more accurate face detection

# Home page
@app.route("/")
def index():
    return render_template("index.html")  # keep your existing HTML form

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Convert uploaded file to OpenCV image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Detect emotions
    results = detector.detect_emotions(img)
    if not results:
        return jsonify({"emotion": "No face detected"})
    
    # Get the first face detected
    emotions = results[0]["emotions"]
    dominant_emotion = max(emotions, key=emotions.get)
    
    return jsonify({"emotion": dominant_emotion})

if __name__ == "__main__":
    app.run(debug=True)
