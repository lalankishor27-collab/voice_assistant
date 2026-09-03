# assistant_text.py
"""
Text-based Interactive Assistant & Pipeline Benchmark
- Loads the leakage-free Pipeline (best_model.joblib)
- Applies train-tuned confidence threshold
- Two-Tier OOD Handling: Explicit 'unknown' class + Confidence Gating
- Dispatches through ActionRouter
- Displays component latency breakdown (preprocess, inference, action)
"""

import os
import sys
import re
import time
import joblib
import numpy as np

from actions import ActionRouter

MODELS_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
TUNED_THRESH_PATH = os.path.join(MODELS_DIR, "tuned_threshold.txt")

# 1. Load Confidence Threshold
CONF_THRESHOLD = 0.35
if os.path.exists(TUNED_THRESH_PATH):
    try:
        with open(TUNED_THRESH_PATH, "r") as f:
            CONF_THRESHOLD = float(f.read().strip())
    except Exception:
        pass

# 2. Load Pipeline
if not os.path.exists(BEST_MODEL_PATH):
    print(f"[FATAL] Model not found at {BEST_MODEL_PATH}. Please run 'python train.py' first.")
    sys.exit(1)

model = joblib.load(BEST_MODEL_PATH)
router = ActionRouter(safe_mode=True)

# 3. Text Normalizer
def normalize_text(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def predict_intent(user_input: str):
    t_start = time.perf_counter()

    # Preprocessing
    t_prep_start = time.perf_counter()
    clean_text = normalize_text(user_input)
    prep_ms = (time.perf_counter() - t_prep_start) * 1000

    # Rule-based overrides (fast deterministic path for high-frequency primitives)
    rule_intent = None
    if any(w in clean_text.split() for w in ("hello", "hi", "hey", "goodmorning", "goodevening")):
        rule_intent = "greeting"
    elif "what time" in clean_text or clean_text == "time":
        rule_intent = "get_time"

    # ML Inference
    t_inf_start = time.perf_counter()
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([clean_text])[0]
        classes = model.classes_
        top_idx = np.argsort(probs)[::-1]
        predicted_intent = classes[top_idx[0]]
        confidence = float(probs[top_idx[0]])
        top_k = [(classes[i], float(probs[i])) for i in top_idx[:3]]
    else:
        predicted_intent = model.predict([clean_text])[0]
        confidence = 1.0
        top_k = [(predicted_intent, 1.0)]
    inf_ms = (time.perf_counter() - t_inf_start) * 1000

    # Apply rule override if present
    if rule_intent:
        final_intent = rule_intent
        confidence = 1.0
        is_override = True
    else:
        final_intent = predicted_intent
        is_override = False

    # Two-Tier Decision & Action Dispatch
    t_act_start = time.perf_counter()
    if confidence < CONF_THRESHOLD:
        # Tier 2: Low Confidence Fallback
        top3_options = ", ".join([intent for intent, _ in top_k])
        action_res = {
            "success": False,
            "action": "clarification_fallback",
            "reply": f"I am not confident in what you meant (Confidence: {confidence:.2f} < {CONF_THRESHOLD:.2f}). Did you mean: {top3_options}?",
            "metadata": {"fallback_reason": "low_confidence", "top_k": top_k}
        }
    elif final_intent == "unknown":
        # Tier 1: Explicit Out-of-Domain Detection
        action_res = router.dispatch("unknown", {"text": clean_text})
    else:
        # High-confidence in-domain intent
        action_res = router.dispatch(final_intent, {"text": clean_text})
    act_ms = (time.perf_counter() - t_act_start) * 1000

    total_proc_ms = (time.perf_counter() - t_start) * 1000

    return {
        "raw_text": user_input,
        "clean_text": clean_text,
        "final_intent": final_intent,
        "confidence": confidence,
        "is_override": is_override,
        "top_k": top_k,
        "action_result": action_res,
        "latency": {
            "preprocess_ms": round(prep_ms, 3),
            "inference_ms": round(inf_ms, 3),
            "action_ms": round(act_ms, 3),
            "total_processing_ms": round(total_proc_ms, 3)
        }
    }

def main():
    print("==========================================================")
    print(" Voice Assistant (Text Diagnostic Mode)")
    print(f" Loaded Model: {type(model).__name__}")
    print(f" Tuned Confidence Threshold: {CONF_THRESHOLD:.2f}")
    print(" Type 'exit' or press Ctrl+C to quit.")
    print("==========================================================\n")

    while True:
        try:
            query = input("You > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Exiting...")
                break

            result = predict_intent(query)
            act = result["action_result"]
            lat = result["latency"]

            print(f"  [Intent]     : {result['final_intent']} (Conf: {result['confidence']:.2f}{' - Rule Override' if result['is_override'] else ''})")
            print(f"  [Action]     : {act['action']} (Success: {act['success']})")
            print(f"  [Reply]      : \"{act['reply']}\"")
            print(f"  [Latency]    : NLU Prep: {lat['preprocess_ms']:.2f}ms | Inference: {lat['inference_ms']:.2f}ms | Action: {lat['action_ms']:.2f}ms | Total: {lat['total_processing_ms']:.2f}ms")
            if not result['is_override']:
                top_str = " | ".join([f"{name}: {prob:.2f}" for name, prob in result['top_k']])
                print(f"  [Candidates] : {top_str}")
            print("-" * 58)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
