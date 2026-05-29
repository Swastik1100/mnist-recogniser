import base64
from io import BytesIO

import joblib
import os
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

# Get the absolute path of the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.joblib')

app = Flask(__name__, static_folder=".")
model = None


def get_model():
    global model
    if model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = joblib.load(model_path)
    return model


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
        loaded_model = get_model()
        features = preprocess_image(image_data)
        prediction = int(loaded_model.predict(features)[0])
    except FileNotFoundError:
        return jsonify({"error": "Model file missing. Train with `python train.py` and redeploy."}), 500
    except Exception:
        return jsonify({"error": "Invalid image data"}), 400

    return jsonify({"prediction": prediction})



if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
