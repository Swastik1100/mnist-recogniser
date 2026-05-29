import base64
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

app = Flask(__name__, static_folder=".")
model = joblib.load(MODEL_PATH)


def preprocess_image(image_data: str) -> np.ndarray:
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    raw = base64.b64decode(image_data)
    image = Image.open(BytesIO(raw)).convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.reshape(1, 784)


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "Missing image"}), 400

    try:
        features = preprocess_image(image_data)
        prediction = int(model.predict(features)[0])
    except Exception as exc:
        return jsonify({"error": f"Invalid image data: {exc}"}), 400

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
