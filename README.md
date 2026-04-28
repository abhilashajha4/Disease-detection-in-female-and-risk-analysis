# Breast Cancer Prediction Project

This project trains a breast cancer classification model from the provided dataset and serves an animated Flask web app for prediction.

## Run

```bash
python train_model.py
python app.py
```

Then open `http://127.0.0.1:5000`.

## Features

- Random Forest classifier trained on the provided CSV
- Animated modern UI with grouped clinical inputs
- Confidence-based benign vs malignant prediction
- Dataset-backed default values and model metrics

## Dataset

The project uses `data/breast_cancer.csv`, copied from your provided dataset.
