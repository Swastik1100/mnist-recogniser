from pathlib import Path

import joblib
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


def main() -> None:
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype("float32") / 255.0
    y = mnist.target.astype("int64")

    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
