"""
Training & Evaluation Pipeline for Fake News Detection
Trains the TF-IDF + Logistic Regression pipeline on True.csv & Fake.csv,
computes classification metrics, and exports reports.
"""

import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

try:
    from .preprocess import clean_text
except ImportError:
    from preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def evaluate_existing_or_train(sample_size=4000):
    """
    Evaluates the model on test samples from True.csv and Fake.csv,
    generates a confusion matrix plot, and saves metrics.json.
    """
    true_path = os.path.join(DATA_DIR, "True.csv")
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    model_path = os.path.join(MODELS_DIR, "fake_news_model.pkl")

    print("[Evaluation] Loading data samples...")
    # Load sample to evaluate quickly and thoroughly
    n_per_class = sample_size // 2
    df_true = pd.read_csv(true_path, nrows=n_per_class)
    df_fake = pd.read_csv(fake_path, nrows=n_per_class)

    df_true["label"] = 1  # Real
    df_fake["label"] = 0  # Fake

    df = pd.concat([df_true, df_fake], ignore_index=True)
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = (df["title"] + " " + df["text"]).apply(clean_text)

    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    if os.path.exists(model_path):
        print(f"[Evaluation] Loading model from: {model_path}")
        pipeline = joblib.load(model_path)
    else:
        print("[Training] Building and training Pipeline...")
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50000, min_df=2, ngram_range=(1, 2), stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, model_path)
        print(f"[Training] Model saved to {model_path}")

    print("[Evaluation] Generating predictions on test set...")
    preds = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))

    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1-Score : {f1 * 100:.2f}%")

    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake (0)", "Real (1)"],
        yticklabels=["Fake (0)", "Real (1)"]
    )
    plt.title("Confusion Matrix - Test Evaluation")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[Reports] Confusion matrix saved: {cm_path}")

    # Save metrics JSON
    metrics_summary = {
        "model_name": "Logistic Regression + TF-IDF (ISOT Benchmark)",
        "test_samples": len(y_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "classification_report": classification_report(
            y_test, preds, target_names=["Fake", "Real"], output_dict=True
        )
    }

    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[Reports] Metrics saved: {metrics_path}")


if __name__ == "__main__":
    evaluate_existing_or_train()
