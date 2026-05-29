# mnist-recogniser

A full-stack MNIST digit recognizer with a drawing canvas UI, Flask backend, and a classic scikit-learn model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Train model

```bash
python train.py
```

This creates `model.joblib` in the project root.

If `model.joblib` is missing in deployment, the backend now auto-trains a small fallback model using scikit-learn's built-in digits dataset so predictions still work.

## Run app

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser, draw one digit (0-9), and click **Predict**.
