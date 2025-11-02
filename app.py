from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

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
    
    # TODO: Add your own emotion detection logic here
    # Currently, just returns a placeholder
    dominant_emotion = "Not implemented"

    return jsonify({"emotion": dominant_emotion})

if __name__ == "__main__":
    # Use PORT environment variable for Render deployment
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
