#!/usr/bin/env python3
"""
Basic Fake News Detector (single-file, no web UI)

USAGE:
    python fake_news.py
        - Auto-load dataset, train, evaluate, then enter interactive predictions.

OPTIONS:
    python fake_news.py --save
        - Also saves: model.pkl and vectorizer.pkl

DATASET AUTO-DETECT (any one works):
    1) dataset.csv   -> columns: text,label  (label in {FAKE, REAL} or {0,1})
       (optionally may include 'title'; if present, title+text are concatenated)
    2) True.csv + Fake.csv -> Kaggle format; will be merged and labeled automatically.
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import shuffle

try:
    from joblib import dump
except Exception:
    dump = None  # saving is optional

# ----------------------------
# Data Loading & Preparation
# ----------------------------
def load_dataset():
    """
    Returns:
        df: DataFrame with columns ['text','label'] where label in {'FAKE','REAL'}
    Auto-detects:
      - dataset.csv  (expects at least 'text' and 'label'; optional 'title')
      - True.csv + Fake.csv (Kaggle)
    """
    if os.path.exists("dataset.csv"):
        df = pd.read_csv("dataset.csv")
        # Normalize columns
        cols = {c.lower(): c for c in df.columns}
        text_col = None
        label_col = None

        # If both title and text exist, concatenate
        has_title = "title" in cols
        if "text" in cols:
            text_col = cols["text"]
        if "label" in cols:
            label_col = cols["label"]

        if text_col is None or label_col is None:
            raise ValueError(
                "dataset.csv must contain at least 'text' and 'label' columns."
            )

        if has_title:
            df["__merged_text"] = (
                df[cols["title"]].fillna("").astype(str) + " " + df[text_col].fillna("").astype(str)
            ).str.strip()
            text_series = df["__merged_text"]
        else:
            text_series = df[text_col].fillna("").astype(str)

        labels = df[label_col].astype(str).str.upper().replace({"1": "REAL", "0": "FAKE"})
        # Keep only FAKE/REAL
        labels = labels.replace({"FALSE": "FAKE", "TRUE": "REAL"})

        df_out = pd.DataFrame({"text": text_series, "label": labels})

    elif os.path.exists("True.csv") and os.path.exists("Fake.csv"):
        df_true = pd.read_csv("True.csv")
        df_fake = pd.read_csv("Fake.csv")

        # Prefer 'text' if present, else combine title+text-like fields
        def extract_text(d):
            cols_lower = {c.lower(): c for c in d.columns}
            if "text" in cols_lower:
                return d[cols_lower["text"]].fillna("").astype(str)
            # fallbacks commonly seen in Kaggle versions
            title = d[cols_lower["title"]].fillna("").astype(str) if "title" in cols_lower else ""
            content = d[cols_lower.get("subject", "")].fillna("").astype(str) if "subject" in cols_lower else ""
            date = d[cols_lower.get("date", "")].fillna("").astype(str) if "date" in cols_lower else ""
            merged = (title + " " + content + " " + date).str.strip()
            return merged

        df_true = pd.DataFrame({"text": extract_text(df_true), "label": "REAL"})
        df_fake = pd.DataFrame({"text": extract_text(df_fake), "label": "FAKE"})
        df_out = pd.concat([df_true, df_fake], axis=0, ignore_index=True)
    else:
        raise FileNotFoundError(
            "No dataset found. Provide either:\n"
            "  - dataset.csv (with columns: text,label [,title])\n"
            "  - OR True.csv and Fake.csv (Kaggle format) in the current folder."
        )

    # Clean, dedupe, shuffle
    df_out["text"] = df_out["text"].fillna("").astype(str)
    df_out["label"] = df_out["label"].astype(str).str.upper().map(lambda x: "REAL" if x == "REAL" else "FAKE")
    df_out = df_out.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df_out = df_out[df_out["text"].str.strip() != ""]
    df_out = shuffle(df_out, random_state=42).reset_index(drop=True)

    if len(df_out) < 50:
        raise ValueError(f"Dataset too small after cleaning: {len(df_out)} rows.")

    return df_out


# ----------------------------
# Modeling
# ----------------------------
def train_and_evaluate(df, test_size=0.2, random_state=42):
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        ngram_range=(1, 2),
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = PassiveAggressiveClassifier(max_iter=100, random_state=random_state)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
    report = classification_report(y_test, y_pred, digits=3)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "metrics": {
            "accuracy": acc,
            "confusion_matrix": cm,
            "labels_order": ["FAKE", "REAL"],
            "report": report
        }
    }


def interactive_predict(model, vectorizer):
    print("\n--- Interactive Prediction ---")
    print("Type/paste a news snippet and press Enter.")
    print("Press just Enter on an empty line to quit.\n")
    while True:
        try:
            text = input("News> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if text == "":
            print("Goodbye!")
            break
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        print(f"Prediction: {'✅ REAL' if pred == 'REAL' else '❌ FAKE'}\n")


def main():
    parser = argparse.ArgumentParser(description="Basic Fake News Detector (single-file, no web UI)")
    parser.add_argument("--save", action="store_true", help="Save model.pkl and vectorizer.pkl")
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_dataset()
    print(f"Loaded {len(df)} rows. Label counts:\n{df['label'].value_counts()}\n")

    print("Training & evaluating...")
    result = train_and_evaluate(df)
    model = result["model"]
    vectorizer = result["vectorizer"]
    metrics = result["metrics"]

    # Print metrics
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print("Confusion Matrix (rows=true, cols=pred) [FAKE, REAL]:")
    print(metrics["confusion_matrix"])
    print("\nClassification Report:")
    print(metrics["report"])

    if args.save:
        if dump is None:
            print("\njoblib not available; cannot save artifacts.")
        else:
            dump(model, "model.pkl")
            dump(vectorizer, "vectorizer.pkl")
            print("\nSaved model.pkl and vectorizer.pkl")

    # Interactive predictions
    interactive_predict(model, vectorizer)


if __name__ == "__main__":
    pd.set_option("display.max_colwidth", 200)
    np.set_printoptions(linewidth=120)
    main()
