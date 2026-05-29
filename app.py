import base64
from io import BytesIO

import joblib
import os
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

# Get the absolute path of the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.joblib')

app = Flask(__name__, static_folder=".")
model = None


def get_model():
    global model
    if model is None:
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
            except Exception:
                model = None

        if model is None:
            digits = load_digits()
            X = (digits.images / 16.0).reshape(len(digits.images), -1)
            y = digits.target.astype("int64")
            fallback_model = RandomForestClassifier(
                n_estimators=180,
                random_state=42,
                n_jobs=-1,
            )
            fallback_model.fit(X, y)
            model = fallback_model
    return model


def preprocess_image(image_data: str) -> np.ndarray:
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    raw = base64.b64decode(image_data)
    loaded_model = get_model()
    expected_features = int(getattr(loaded_model, "n_features_in_", 784))
    side = int(np.sqrt(expected_features))
    if side * side != expected_features:
        raise ValueError("Invalid model input shape")

    image = Image.open(BytesIO(raw)).convert("L").resize((side, side), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.reshape(1, expected_features)


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/styles.css")
def styles():
    return send_from_directory(BASE_DIR, "styles.css")


@app.get("/script.js")
def script():
    return send_from_directory(BASE_DIR, "script.js")


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
    except ValueError:
        return jsonify({"error": "Invalid image data"}), 400
    except Exception:
        return jsonify({"error": "Prediction failed"}), 500

    return jsonify({"prediction": prediction})



if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
