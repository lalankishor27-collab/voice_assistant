# train.py
"""
Models:
 - Logistic Regression
 - Linear SVM (Calibrated)
 - Multinomial Naive Bayes
 - KNN

Outputs:
 - models/vectorizer.joblib
 - models/best_model.joblib   <-- this is the one your assistant should load
 - models/results_summary.csv
 - models/<model>.joblib (each individual model)
"""

import os
import sys
import re
import time
import joblib
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# --------------------------
# CONFIG
# --------------------------
INTENTS_CSV = "intents.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

VECT_FILE = os.path.join(MODELS_DIR, "vectorizer.joblib")
BEST_MODEL_FILE = os.path.join(MODELS_DIR, "best_model.joblib")
RESULTS_CSV = os.path.join(MODELS_DIR, "results_summary.csv")

# TF-IDF settings
MAX_FEATURES = 12000
NGRAM = (1, 2)
MIN_DF = 1

AUG_THRESHOLD = 150
AUG_SUFFIXES = ["", " please", " now", " kindly", " quickly"]

RANDOM_STATE = 42
CV_FOLDS = 5

SELECTION_METRIC = "accuracy"   # you can change to f1_macro if needed

# --------------------------
# HELPERS
# --------------------------
def normalize(s):
    if pd.isna(s): return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def augment(df):
    rows = []
    for _, r in df.iterrows():
        base = r["text"]
        intent = r["intent"]
        for suf in AUG_SUFFIXES:
            t = (base + suf).strip()
            rows.append({"text": t, "intent": intent})
            rows.append({"text": t.capitalize(), "intent": intent})
    return pd.DataFrame(rows)

# --------------------------
# LOAD DATA
# --------------------------
if not os.path.exists(INTENTS_CSV):
    print("[FATAL] intents.csv missing!")
    sys.exit()

df = pd.read_csv(INTENTS_CSV)
df["text"] = df["text"].astype(str).map(normalize)
df["intent"] = df["intent"].astype(str).str.strip()
df = df[df["text"].str.len() > 0].reset_index()

print("[INFO] Loaded:", len(df), "samples")

# --------------------------
# AUGMENT IF SMALL
# --------------------------
if len(df) < AUG_THRESHOLD:
    print("[INFO] Applying light augmentation...")
    df2 = augment(df)
    df = pd.concat([df, df2], ignore_index=True)
    df.drop_duplicates(subset=["text", "intent"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("[INFO] After augmentation:", len(df))

# --------------------------
# VECTORIZER
# --------------------------
vectorizer = TfidfVectorizer(ngram_range=NGRAM, max_features=MAX_FEATURES, min_df=MIN_DF)
X_all = vectorizer.fit_transform(df["text"])
y_all = df["intent"].values

joblib.dump(vectorizer, VECT_FILE)
print("[INFO] Vectorizer saved.")

# --------------------------
# MODELS
# --------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "LinearSVM": CalibratedClassifierCV(LinearSVC(max_iter=4000, class_weight="balanced")),
    "MultinomialNB": MultinomialNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []

# --------------------------
# CROSS-VALIDATION EVALUATION
# --------------------------
print("\n[INFO] Running evaluation with CV =", CV_FOLDS)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    print("\n====================================================")
    print("[TRAIN] Model:", name)
    start = time.time()

    # Cross-validated predictions
    y_pred = cross_val_predict(model, X_all, y_all, cv=cv, method="predict")

    # Fit on full dataset for saving
    model.fit(X_all, y_all)
    train_time = time.time() - start

    # Metrics
    acc = accuracy_score(y_all, y_pred)
    f1_macro = f1_score(y_all, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_all, y_pred, average="weighted", zero_division=0)

    print(f"[RESULT] Accuracy={acc:.4f}, Macro-F1={f1_macro:.4f}")

    # Save model
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))

    results.append({
        "model": name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "train_time": train_time
    })

# --------------------------
# SAVE RESULTS TABLE
# --------------------------
res_df = pd.DataFrame(results).sort_values(by=SELECTION_METRIC, ascending=False)
res_df.to_csv(RESULTS_CSV, index=False)
print("\n[INFO] Results saved to:", RESULTS_CSV)
print(res_df)

# --------------------------
# SELECT BEST MODEL
# --------------------------
best_name = res_df.iloc[0]["model"]
best_path = os.path.join(MODELS_DIR, f"{best_name}.joblib")

joblib.dump(joblib.load(best_path), BEST_MODEL_FILE)

print("\n[INFO] BEST MODEL SELECTED:", best_name)
print("[INFO] Saved as:", BEST_MODEL_FILE)
print("\n[DONE]")
