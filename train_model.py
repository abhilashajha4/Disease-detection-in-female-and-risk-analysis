from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "breast_cancer.csv"
MODEL_PATH = BASE_DIR / "models" / "breast_cancer_model.joblib"


def train_and_save_model() -> dict:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in ["id", "Unnamed: 32"] if c in df.columns])

    target_col = "diagnosis"
    features = [col for col in df.columns if col != target_col]

    X = df[features]
    y = df[target_col].map({"M": 1, "B": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    feature_importance = (
        pd.Series(pipeline.named_steps["model"].feature_importances_, index=features)
        .sort_values(ascending=False)
        .to_dict()
    )

    artifact = {
        "model": pipeline,
        "features": features,
        "medians": X.median(numeric_only=True).to_dict(),
        "ranges": {
            col: {
                "min": float(X[col].min()),
                "max": float(X[col].max()),
                "mean": float(X[col].mean()),
            }
            for col in features
        },
        "feature_importance": feature_importance,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions)),
            "recall": float(recall_score(y_test, predictions)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        },
        "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    return artifact


if __name__ == "__main__":
    saved = train_and_save_model()
    print("Model trained and saved.")
    print(saved["metrics"])
