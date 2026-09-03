# train.py
"""
Offline Voice Assistant Intent Classification Pipeline
- Clean 80/20 Train / Held-out Test Stratified Split
- Leakage-Free 5-Fold Stratified Cross-Validation on Training Data via sklearn Pipeline
- Primary Model Selection: CV Macro F1 (Secondary: CV Accuracy)
- Comprehensive Metrics: Accuracy, Macro F1, Weighted F1, Train Time, Inference Latency, Model Size
- Final Unbiased Evaluation on Held-out Test Set (Confusion Matrix, Error Analysis)
- Visual Artifacts via Matplotlib (Confusion Matrix, Model Comparison)
"""

import os
import sys
import re
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
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
OUTPUTS_DIR = "model_compare_outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

BEST_MODEL_FILE = os.path.join(MODELS_DIR, "best_model.joblib")
VECT_FILE = os.path.join(MODELS_DIR, "vectorizer.joblib")
RESULTS_CSV = os.path.join(MODELS_DIR, "results_summary.csv")
TEST_METRICS_CSV = os.path.join(MODELS_DIR, "test_metrics.csv")
TEST_ERROR_CSV = os.path.join(MODELS_DIR, "test_error_analysis.csv")
CONF_MATRIX_CSV = os.path.join(MODELS_DIR, "confusion_matrix_test.csv")
OOF_PROBS_FILE = os.path.join(MODELS_DIR, "oof_train_predictions.joblib")

MAX_FEATURES = 12000
NGRAM = (1, 2)
MIN_DF = 1
RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.20

# --------------------------
# TEXT PREPROCESSING
# --------------------------
def normalize(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --------------------------
# LOAD & CLEAN DATA
# --------------------------
if not os.path.exists(INTENTS_CSV):
    print(f"[FATAL] {INTENTS_CSV} missing!")
    sys.exit(1)

df = pd.read_csv(INTENTS_CSV)
df["text"] = df["text"].astype(str).map(normalize)
df["intent"] = df["intent"].astype(str).str.strip()
df = df[df["text"].str.len() > 0].reset_index(drop=True)

print(f"[INFO] Loaded {len(df)} samples across {df['intent'].nunique()} unique intents.")
print(f"[INFO] Class distribution summary:\n{df['intent'].value_counts().to_string()}\n")

# --------------------------
# 80/20 STRATIFIED TRAIN / TEST SPLIT
# --------------------------
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["intent"],
    random_state=RANDOM_STATE
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"[INFO] Dataset split: Train={len(train_df)} ({(1-TEST_SIZE)*100:.0f}%), Held-out Test={len(test_df)} ({TEST_SIZE*100:.0f}%)")

X_train, y_train = train_df["text"].values, train_df["intent"].values
X_test, y_test = test_df["text"].values, test_df["intent"].values

# --------------------------
# MODEL DEFINITIONS (PIPELINES TO PREVENT DATA LEAKAGE)
# --------------------------
def build_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=NGRAM, max_features=MAX_FEATURES, min_df=MIN_DF)),
        ("classifier", clf)
    ])

models = {
    "LinearSVM": CalibratedClassifierCV(LinearSVC(max_iter=4000, class_weight="balanced", random_state=RANDOM_STATE)),
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
    "MultinomialNB": MultinomialNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []
fitted_pipelines = {}
oof_predictions_dict = {}

# --------------------------
# 5-FOLD CV ON TRAINING SET ONLY
# --------------------------
print(f"\n[INFO] Starting {CV_FOLDS}-Fold Stratified Cross-Validation strictly on Training Data...")
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for name, clf in models.items():
    print(f"\n----------------------------------------------------")
    print(f"[TRAIN] Evaluating Candidate: {name}")
    pipe = build_pipeline(clf)

    # 1. Out-of-fold CV predictions (no data leakage: vectorizer fits inside folds)
    t0 = time.time()
    y_oof_pred = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict")
    cv_time = time.time() - t0

    # Also collect out-of-fold probabilities for calibration / threshold tuning
    try:
        y_oof_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")
    except Exception:
        y_oof_proba = None

    oof_predictions_dict[name] = {
        "y_true": y_train,
        "y_pred": y_oof_pred,
        "y_proba": y_oof_proba,
        "classes": np.unique(y_train)
    }

    # 2. Compute CV Metrics
    acc = accuracy_score(y_train, y_oof_pred)
    f1_macro = f1_score(y_train, y_oof_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_train, y_oof_pred, average="weighted", zero_division=0)

    # 3. Fit pipeline on full 80% train set
    t_fit_start = time.time()
    pipe.fit(X_train, y_train)
    fit_duration = time.time() - t_fit_start

    # 4. Measure inference latency per sample (batch of 50 repeated calls)
    test_sample = [X_train[0]]
    inf_times = []
    for _ in range(50):
        t_inf_start = time.perf_counter()
        _ = pipe.predict(test_sample)
        inf_times.append((time.perf_counter() - t_inf_start) * 1000)
    inf_latency_ms = float(np.median(inf_times))

    # 5. Save candidate pipeline and measure disk size
    candidate_path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(pipe, candidate_path)
    model_size_kb = os.path.getsize(candidate_path) / 1024.0

    fitted_pipelines[name] = pipe

    print(f"[RESULT] {name}: CV Accuracy = {acc:.4f} | CV Macro-F1 = {f1_macro:.4f} | CV Weighted-F1 = {f1_weighted:.4f}")
    print(f"         Fit Time: {fit_duration:.3f}s | Latency: {inf_latency_ms:.2f}ms/sample | Size: {model_size_kb:.1f} KB")

    results.append({
        "model": name,
        "cv_accuracy": round(acc, 4),
        "cv_f1_macro": round(f1_macro, 4),
        "cv_f1_weighted": round(f1_weighted, 4),
        "train_time_s": round(fit_duration, 4),
        "inference_latency_ms": round(inf_latency_ms, 3),
        "model_size_kb": round(model_size_kb, 1)
    })

# Save OOF predictions for threshold tuning
joblib.dump(oof_predictions_dict, OOF_PROBS_FILE)
print(f"\n[INFO] Out-of-fold training probabilities saved to: {OOF_PROBS_FILE}")

# --------------------------
# MODEL SELECTION
# --------------------------
# Primary selection: cv_f1_macro. Secondary: cv_accuracy
res_df = pd.DataFrame(results).sort_values(
    by=["cv_f1_macro", "cv_accuracy"], ascending=[False, False]
).reset_index(drop=True)
res_df.to_csv(RESULTS_CSV, index=False)
print("\n====================================================")
print("[INFO] Cross-Validation Comparison Summary:")
print(res_df.to_string(index=False))
print("====================================================")

best_name = res_df.iloc[0]["model"]
best_pipe = fitted_pipelines[best_name]

# Save best model pipeline
joblib.dump(best_pipe, BEST_MODEL_FILE)
# Also extract vectorizer for backward compatibility
joblib.dump(best_pipe.named_steps["tfidf"], VECT_FILE)
print(f"\n[INFO] BEST MODEL SELECTED: {best_name} (Primary: CV Macro-F1 = {res_df.iloc[0]['cv_f1_macro']:.4f})")
print(f"[INFO] Best pipeline saved to: {BEST_MODEL_FILE}")
print(f"[INFO] Vectorizer extracted and saved to: {VECT_FILE}")

# --------------------------
# UNBIASED HELD-OUT TEST EVALUATION
# --------------------------
print("\n====================================================")
print(f"[INFO] Running final unbiased evaluation of '{best_name}' on held-out test set ({len(X_test)} samples)...")
y_test_pred = best_pipe.predict(X_test)
try:
    y_test_proba = best_pipe.predict_proba(X_test)
    max_probs = np.max(y_test_proba, axis=1)
except Exception:
    max_probs = np.ones(len(y_test))

test_acc = accuracy_score(y_test, y_test_pred)
test_f1_macro = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
test_f1_weighted = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

print(f"[TEST RESULT] Test Accuracy    = {test_acc:.4f}")
print(f"[TEST RESULT] Test Macro-F1    = {test_f1_macro:.4f}")
print(f"[TEST RESULT] Test Weighted-F1 = {test_f1_weighted:.4f}")

# Save test metrics
test_metrics_df = pd.DataFrame([{
    "best_model": best_name,
    "test_accuracy": round(test_acc, 4),
    "test_f1_macro": round(test_f1_macro, 4),
    "test_f1_weighted": round(test_f1_weighted, 4),
    "test_samples": len(X_test)
}])
test_metrics_df.to_csv(TEST_METRICS_CSV, index=False)

# Classification Report
labels = sorted(list(np.unique(y_test)))
clf_report = classification_report(y_test, y_test_pred, labels=labels, zero_division=0)
print("\n[INFO] Detailed Test Classification Report:\n")
print(clf_report)

# --------------------------
# TEST CONFUSION MATRIX & ERROR ANALYSIS
# --------------------------
cm = confusion_matrix(y_test, y_test_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv(CONF_MATRIX_CSV)
print(f"[INFO] Confusion matrix saved to: {CONF_MATRIX_CSV}")

# Save row-level test error analysis
errors = []
for i in range(len(X_test)):
    errors.append({
        "text": X_test[i],
        "actual_intent": y_test[i],
        "predicted_intent": y_test_pred[i],
        "confidence": round(float(max_probs[i]), 4),
        "is_correct": bool(y_test[i] == y_test_pred[i])
    })
error_df = pd.DataFrame(errors)
error_df.to_csv(TEST_ERROR_CSV, index=False)
misclassifications = error_df[~error_df["is_correct"]]
print(f"[INFO] Test error analysis saved to: {TEST_ERROR_CSV}")
print(f"[INFO] Misclassified test samples: {len(misclassifications)} / {len(X_test)}")
if not misclassifications.empty:
    print(misclassifications[["text", "actual_intent", "predicted_intent", "confidence"]].to_string(index=False))

# --------------------------
# VISUALIZATIONS VIA MATPLOTLIB
# --------------------------
print("\n[INFO] Generating performance plots via Matplotlib...")

# 1. Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(14, 11))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=60)
ax.set_title(f"Confusion Matrix on Held-Out Test Set ({best_name})\nAccuracy: {test_acc:.3f} | Macro-F1: {test_f1_macro:.3f}", fontsize=12)
plt.tight_layout()
cm_plot_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
plt.savefig(cm_plot_path, dpi=200)
plt.close()
print(f"[INFO] Saved confusion matrix plot to: {cm_plot_path}")

# 2. Model Comparison Bar Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
model_names = res_df["model"].tolist()
accs = res_df["cv_accuracy"].tolist()
f1s = res_df["cv_f1_macro"].tolist()

x = np.arange(len(model_names))
width = 0.35

ax1.bar(x - width/2, accs, width, label="CV Accuracy", color="#3498db")
ax1.bar(x + width/2, f1s, width, label="CV Macro-F1", color="#2ecc71")
ax1.set_ylabel("Score")
ax1.set_title("5-Fold Cross-Validation: Accuracy vs Macro-F1")
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=15)
ax1.set_ylim([0.7, 1.0])
ax1.legend(loc="lower right")
ax1.grid(axis="y", linestyle="--", alpha=0.7)

latencies = res_df["inference_latency_ms"].tolist()
ax2.bar(model_names, latencies, color="#e67e22", width=0.4)
ax2.set_ylabel("Latency (ms / sample)")
ax2.set_title("Inference Latency per Sample (Lower is Better)")
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, rotation=15)
ax2.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
compare_plot_path = os.path.join(OUTPUTS_DIR, "model_comparison.png")
plt.savefig(compare_plot_path, dpi=200)
plt.close()
print(f"[INFO] Saved model comparison plot to: {compare_plot_path}")

print("\n[DONE] Leakage-free training and evaluation complete!")
