# tune_threshold.py
"""
Threshold Tuning on Training Data Cross-Validation Probabilities (Leakage-Free)
- Strictly utilizes out-of-fold predictions from 5-Fold CV on the 80% train set
- Sweeps confidence threshold from 0.20 to 0.80
- Formally computes:
    * Accepted: confidence >= threshold
    * Correct Acceptance (True Acceptance): predicted == actual and confidence >= threshold
    * False Acceptance: predicted != actual and confidence >= threshold
    * Fallback: confidence < threshold
    * Precision among accepted: correct_accepted / accepted
- Selects evidence-based optimal threshold balancing high Correct Acceptance vs low False Acceptance
- Generates tradeoff plot: model_compare_outputs/threshold_tradeoff.png
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS_DIR = "models"
OUTPUTS_DIR = "model_compare_outputs"
OOF_PROBS_FILE = os.path.join(MODELS_DIR, "oof_train_predictions.joblib")
RESULTS_CSV = os.path.join(MODELS_DIR, "results_summary.csv")
THRESHOLD_CSV = os.path.join(MODELS_DIR, "threshold_analysis.csv")

if not os.path.exists(OOF_PROBS_FILE):
    print(f"[ERROR] {OOF_PROBS_FILE} not found. Please run 'python train.py' first.")
    sys.exit(1)

oof_dict = joblib.load(OOF_PROBS_FILE)

# Identify best model from results summary
best_model_name = "LinearSVM"
if os.path.exists(RESULTS_CSV):
    res_df = pd.read_csv(RESULTS_CSV)
    best_model_name = res_df.iloc[0]["model"]

if best_model_name not in oof_dict:
    best_model_name = list(oof_dict.keys())[0]

print(f"[INFO] Running threshold analysis for Best Model: {best_model_name}")
data = oof_dict[best_model_name]
y_true = np.array(data["y_true"])
y_proba = data["y_proba"]
classes = np.array(data["classes"])

if y_proba is None:
    print(f"[ERROR] {best_model_name} does not provide predict_proba outputs.")
    sys.exit(1)

max_probs = np.max(y_proba, axis=1)
pred_classes = classes[np.argmax(y_proba, axis=1)]

total_samples = len(y_true)
thresholds = np.arange(0.20, 0.85, 0.05)

analysis = []
for t in thresholds:
    t = round(float(t), 2)
    accepted_mask = max_probs >= t
    fallback_mask = ~accepted_mask

    accepted_count = int(np.sum(accepted_mask))
    fallback_count = int(np.sum(fallback_mask))

    correct_accepted = int(np.sum((pred_classes == y_true) & accepted_mask))
    false_accepted = int(np.sum((pred_classes != y_true) & accepted_mask))

    acceptance_rate = accepted_count / total_samples
    correct_acc_rate = correct_accepted / total_samples
    false_acc_rate = false_accepted / total_samples
    fallback_rate = fallback_count / total_samples
    precision_accepted = (correct_accepted / accepted_count) if accepted_count > 0 else 1.0

    analysis.append({
        "threshold": t,
        "acceptance_rate": round(acceptance_rate, 4),
        "correct_acceptance_rate": round(correct_acc_rate, 4),
        "false_acceptance_rate": round(false_acc_rate, 4),
        "fallback_rate": round(fallback_rate, 4),
        "precision_when_accepted": round(precision_accepted, 4),
        "correct_count": correct_accepted,
        "false_count": false_accepted,
        "fallback_count": fallback_count
    })

thresh_df = pd.DataFrame(analysis)
thresh_df.to_csv(THRESHOLD_CSV, index=False)
print(f"[INFO] Threshold analysis saved to: {THRESHOLD_CSV}\n")
print(thresh_df.to_string(index=False))

# Objective for optimal threshold:
# High correct acceptance, minimize false acceptance (penalty: false_acceptance * 3)
# Score = correct_acceptance_rate - 2.5 * false_acc_rate - 0.2 * fallback_rate
scores = (
    thresh_df["correct_acceptance_rate"]
    - 2.5 * thresh_df["false_acceptance_rate"]
    - 0.15 * thresh_df["fallback_rate"]
)
best_idx = scores.idxmax()
optimal_threshold = thresh_df.iloc[best_idx]["threshold"]
opt_row = thresh_df.iloc[best_idx]

print("\n====================================================")
print(f"[INFO] EVIDENCE-BASED OPTIMAL THRESHOLD: {optimal_threshold:.2f}")
print(f"       Correct Acceptance Rate : {opt_row['correct_acceptance_rate']*100:.1f}%")
print(f"       False Acceptance Rate   : {opt_row['false_acceptance_rate']*100:.1f}%")
print(f"       Fallback Rate           : {opt_row['fallback_rate']*100:.1f}%")
print(f"       Precision When Accepted : {opt_row['precision_when_accepted']*100:.1f}%")
print("====================================================")

# Save optimal threshold config
with open(os.path.join(MODELS_DIR, "tuned_threshold.txt"), "w") as f:
    f.write(f"{optimal_threshold:.2f}\n")

# Matplotlib Tradeoff Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresh_df["threshold"], thresh_df["correct_acceptance_rate"] * 100, label="Correct Acceptance (%)", marker="o", color="#2ecc71", linewidth=2)
ax.plot(thresh_df["threshold"], thresh_df["false_acceptance_rate"] * 100, label="False Acceptance / Erroneous Exec (%)", marker="s", color="#e74c3c", linewidth=2)
ax.plot(thresh_df["threshold"], thresh_df["fallback_rate"] * 100, label="Fallback / Clarification Prompt (%)", marker="^", color="#f39c12", linewidth=2)
ax.axvline(optimal_threshold, color="#34495e", linestyle="--", linewidth=1.8, label=f"Selected Threshold ({optimal_threshold:.2f})")

ax.set_xlabel("Confidence Threshold", fontsize=11)
ax.set_ylabel("Percentage of Utterances (%)", fontsize=11)
ax.set_title(f"Confidence Threshold Tuning on Out-Of-Fold Train CV ({best_model_name})\nBalancing True Acceptance vs False Acceptance Risk", fontsize=12)
ax.legend(loc="best", frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)
ax.set_xlim([0.18, 0.82])
ax.set_ylim([-2, 102])

plt.tight_layout()
plot_path = os.path.join(OUTPUTS_DIR, "threshold_tradeoff.png")
plt.savefig(plot_path, dpi=200)
plt.close()
print(f"[INFO] Saved threshold trade-off curve to: {plot_path}")
