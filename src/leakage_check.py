"""
Leakage / robustness check for the ISOT Fake News dataset.

Why: in ISOT, real articles usually start with "CITY (Reuters) -" and the data
contains duplicate rows, so a model can score ~99% by learning source artifacts
instead of anything about truthfulness.

This script uses the SAME setup as src/train.py (same text cleaning, same
TF-IDF + Logistic Regression settings, same first-N-rows-per-file sampling,
same 75/25 stratified split, labels Real=1 / Fake=0) and trains it three times:

  level 0  baseline         raw text
  level 1  no Reuters tag   removes the "(Reuters)" dateline and the word "reuters"
  level 2  no source tags   level 1 + other publisher artifacts
                            (featured image, getty images, pic.twitter.com, ...)

Usage (from the project root, with data/True.csv and data/Fake.csv present):
    python src/leakage_check.py                    # same 4000-article sample as train.py
    python src/leakage_check.py --sample-size 0    # use all articles

Optional test on a different dataset (CSV with a text column and a label column):
    python src/leakage_check.py --external data/other.csv --text-col text \
        --label-col label --real-value 1

Results are written to reports/leakage_check_results.json
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from preprocess import clean_text  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

REUTERS_DATELINE = re.compile(r"^\s*[^()\n]{0,60}\(reuters\)\s*[-–—]*\s*", re.I)
REUTERS_WORD = re.compile(r"\breuters\b", re.I)
OTHER_ARTIFACTS = re.compile(
    r"featured image|getty images|image via|pic\.twitter\.com/\S+|"
    r"21st century wire|via twitter|via youtube|via facebook",
    re.I,
)

LEVELS = {
    0: "Baseline (raw text)",
    1: "Reuters tag removed",
    2: "Reuters + other source tags removed",
}


def strip_artifacts(raw: str, level: int) -> str:
    """Remove source artifacts from RAW text (before clean_text lowercases it)."""
    raw = str(raw)
    if level >= 1:
        raw = REUTERS_DATELINE.sub("", raw)
        raw = REUTERS_WORD.sub(" ", raw)
    if level >= 2:
        raw = OTHER_ARTIFACTS.sub(" ", raw)
    return raw


def prepare(raw: str, level: int) -> str:
    return clean_text(strip_artifacts(raw, level))


def make_pipeline() -> Pipeline:
    # identical to src/train.py
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, min_df=2, ngram_range=(1, 2), stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def top_words(pipe: Pipeline, k: int = 15) -> dict:
    names = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coef = pipe.named_steps["classifier"].coef_[0]  # positive -> Real (1), negative -> Fake (0)
    order = np.argsort(coef)
    return {"toward_fake": names[order[:k]].tolist(), "toward_real": names[order[::-1][:k]].tolist()}


def score(y_true, y_pred) -> dict:
    # metrics for the Fake class (label 0), the class we want to catch
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0], zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "fake_precision": round(float(p[0]), 4),
        "fake_recall": round(float(r[0]), 4),
        "fake_f1": round(float(f[0]), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),  # rows/cols: Fake, Real
    }


def load_isot(sample_size: int) -> pd.DataFrame:
    true_path = os.path.join(DATA_DIR, "True.csv")
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    for p in (true_path, fake_path):
        if not os.path.exists(p):
            sys.exit(f"Missing {p}. Download the ISOT dataset and put True.csv and Fake.csv in data/ (see data/README.md).")
    n = sample_size // 2 if sample_size > 0 else None
    real = pd.read_csv(true_path, nrows=n)
    fake = pd.read_csv(fake_path, nrows=n)
    real["label"], fake["label"] = 1, 0
    df = pd.concat([real, fake], ignore_index=True)
    df["raw"] = df["title"].fillna("") + " " + df["text"].fillna("")
    return df[["raw", "label"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-size", type=int, default=4000, help="total articles (half per file); 0 = all")
    ap.add_argument("--external", default=None, help="CSV from a different dataset for an out-of-dataset test")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--real-value", default="1", help="value in the label column that means REAL")
    args = ap.parse_args()

    df = load_isot(args.sample_size)
    # drop duplicates on the cleaned baseline text, as train.py does
    df["key"] = df["raw"].map(lambda t: prepare(t, 0))
    n0 = len(df)
    df = df[df["key"].str.len() > 0].drop_duplicates(subset="key").reset_index(drop=True)
    print(f"Articles loaded: {n0} | after removing empty/duplicates: {len(df)}")
    print(f"Real (1): {(df.label == 1).sum()} | Fake (0): {(df.label == 0).sum()}")

    tr, te = train_test_split(df.index, test_size=0.25, random_state=42, stratify=df["label"])
    y_tr, y_te = df.loc[tr, "label"], df.loc[te, "label"]
    print(f"Train: {len(tr)} | Test: {len(te)}")

    ext = None
    if args.external:
        e = pd.read_csv(args.external)
        ext = pd.DataFrame({
            "raw": e[args.text_col].fillna("").astype(str),
            "label": (e[args.label_col].astype(str) == str(args.real_value)).astype(int),
        }).drop_duplicates(subset="raw")
        print(f"External set: {len(ext)} articles")

    results, rows = [], []
    for level, name in LEVELS.items():
        print(f"\n--- Level {level}: {name} ---")
        X_tr = df.loc[tr, "raw"].map(lambda t: prepare(t, level))
        X_te = df.loc[te, "raw"].map(lambda t: prepare(t, level))
        pipe = make_pipeline().fit(X_tr, y_tr)
        m = score(y_te, pipe.predict(X_te))
        words = top_words(pipe)
        print(f"Accuracy {m['accuracy']:.4f} | Fake precision {m['fake_precision']:.4f} | "
              f"Fake recall {m['fake_recall']:.4f} | Fake F1 {m['fake_f1']:.4f}")
        print(f"Confusion matrix [[Fake->Fake, Fake->Real],[Real->Fake, Real->Real]]: {m['confusion_matrix']}")
        print("Top words toward FAKE:", ", ".join(words["toward_fake"]))
        print("Top words toward REAL:", ", ".join(words["toward_real"]))
        entry = {"level": level, "variant": name, **m, "top_words": words}
        rows.append((name, m))
        if ext is not None:
            em = score(ext["label"], pipe.predict(ext["raw"].map(lambda t: prepare(t, level))))
            print(f"External accuracy {em['accuracy']:.4f} | Fake F1 {em['fake_f1']:.4f}")
            entry["external"] = em
            rows.append((f"  -> external: {name}", em))
        results.append(entry)

    print("\n" + "=" * 84)
    print(f"{'Variant':44s}{'Acc':>9s}{'F-Prec':>10s}{'F-Rec':>10s}{'F-F1':>10s}")
    print("-" * 84)
    for name, m in rows:
        print(f"{name:44s}{m['accuracy']:9.4f}{m['fake_precision']:10.4f}{m['fake_recall']:10.4f}{m['fake_f1']:10.4f}")
    print("=" * 84)
    print("(F- = Fake class)")

    drop = (results[0]["accuracy"] - results[-1]["accuracy"]) * 100
    print(f"\nAccuracy change, baseline -> fully cleaned: -{drop:.2f} percentage points")
    print("A large drop means the baseline score partly relied on source artifacts." if drop > 2
          else "A small drop means the score does not depend much on the removed artifacts.")
    print("ISOT can still contain other shortcuts; an external dataset is the stronger test.")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = os.path.join(REPORTS_DIR, "leakage_check_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"test_size": len(te), "results": results}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
