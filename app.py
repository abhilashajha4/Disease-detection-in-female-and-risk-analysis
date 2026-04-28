from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

from train_model import MODEL_PATH, train_and_save_model


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)


DISPLAY_NAMES = {
    "se": "Standard Error",
    "mean": "Mean Measurements",
    "worst": "Worst Measurements",
}


def load_artifact() -> dict:
    if not MODEL_PATH.exists():
        return train_and_save_model()
    return joblib.load(MODEL_PATH)


ARTIFACT = load_artifact()


def group_features(features: list[str]) -> dict[str, list[str]]:
    groups = {"mean": [], "se": [], "worst": []}
    for feature in features:
        if feature.endswith("_mean"):
            groups["mean"].append(feature)
        elif feature.endswith("_se"):
            groups["se"].append(feature)
        elif feature.endswith("_worst"):
            groups["worst"].append(feature)
    return groups


def make_label(feature_name: str) -> str:
    cleaned = feature_name.replace("_", " ")
    cleaned = cleaned.replace("se", "SE")
    return cleaned.title().replace("Concave Points", "Concave Points")


def to_float(form_value: str | None, fallback: float) -> float:
    try:
        return float(form_value) if form_value not in (None, "") else float(fallback)
    except (TypeError, ValueError):
        return float(fallback)


def build_feature_cards() -> list[dict]:
    return [
        {
            "name": feature,
            "label": make_label(feature),
            "median": round(float(ARTIFACT["medians"][feature]), 6),
            "min": round(float(ARTIFACT["ranges"][feature]["min"]), 6),
            "max": round(float(ARTIFACT["ranges"][feature]["max"]), 6),
            "mean": round(float(ARTIFACT["ranges"][feature]["mean"]), 6),
            "importance": round(float(ARTIFACT["feature_importance"][feature]) * 100, 2),
        }
        for feature in ARTIFACT["features"]
    ]


@app.route("/", methods=["GET"])
def index():
    cards = build_feature_cards()
    grouped = group_features(ARTIFACT["features"])
    defaults = {feature: ARTIFACT["medians"][feature] for feature in ARTIFACT["features"]}
    return render_template(
        "index.html",
        feature_groups=grouped,
        display_names=DISPLAY_NAMES,
        cards=cards,
        defaults=defaults,
        metrics=ARTIFACT["metrics"],
        dataset_shape=ARTIFACT["dataset_shape"],
        top_features=list(ARTIFACT["feature_importance"].items())[:6],
        result=None,
    )


@app.route("/awareness", methods=["GET"])
def awareness():
    return render_template("awareness.html")


@app.route("/predict", methods=["POST"])
def predict():
    values = {
        feature: to_float(request.form.get(feature), ARTIFACT["medians"][feature])
        for feature in ARTIFACT["features"]
    }

    input_df = pd.DataFrame([values])
    probability = float(ARTIFACT["model"].predict_proba(input_df)[0][1])
    prediction = int(ARTIFACT["model"].predict(input_df)[0])

    risk_percent = round(probability * 100, 2)
    diagnosis = "Malignant" if prediction == 1 else "Benign"
    tone = "danger" if prediction == 1 else "safe"
    guidance = (
        "High-risk pattern detected. Please use this as a screening aid and confirm with a clinician."
        if prediction == 1
        else "Low-risk pattern detected. This is reassuring, but it should not replace medical advice."
    )

    cards = build_feature_cards()
    grouped = group_features(ARTIFACT["features"])

    return render_template(
        "index.html",
        feature_groups=grouped,
        display_names=DISPLAY_NAMES,
        cards=cards,
        defaults=values,
        metrics=ARTIFACT["metrics"],
        dataset_shape=ARTIFACT["dataset_shape"],
        top_features=list(ARTIFACT["feature_importance"].items())[:6],
        result={
            "diagnosis": diagnosis,
            "confidence": risk_percent,
            "tone": tone,
            "guidance": guidance,
            "benign_percent": round(100 - risk_percent, 2),
        },
    )


if __name__ == "__main__":
    app.run(debug=True)
