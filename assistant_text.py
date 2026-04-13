# assistant_text.py
"""
Simple text-based interface to your NLU + action pipeline.
Usage:
    (.venv) > python assistant_text.py
Then just type commands like: hello, what time is it, play music
Type "exit" or Ctrl+C to quit.
"""

import os
import joblib
import re
import time
import sys
import numpy as np
import webbrowser
import pyttsx3

MODELS_DIR = "models"
VECT_PATH = os.path.join(MODELS_DIR, "vectorizer.joblib")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
FALLBACK_MODEL_PATH = os.path.join(MODELS_DIR, "intent_clf.joblib")

# Load vectorizer
if not os.path.exists(VECT_PATH):
    print(f"[FATAL] Vectorizer not found at {VECT_PATH}. Please run the training script first.")
    sys.exit(1)
vect = joblib.load(VECT_PATH)

# Load classifier: prefer best_model.joblib, fallback to intent_clf.joblib
clf = None
if os.path.exists(BEST_MODEL_PATH):
    try:
        clf = joblib.load(BEST_MODEL_PATH)
        print(f"[INFO] Loaded classifier from {BEST_MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load {BEST_MODEL_PATH}: {e}")
        clf = None

if clf is None and os.path.exists(FALLBACK_MODEL_PATH):
    try:
        clf = joblib.load(FALLBACK_MODEL_PATH)
        print(f"[INFO] Loaded fallback classifier from {FALLBACK_MODEL_PATH}")
    except Exception as e:
        print(f"[FATAL] Failed to load fallback classifier {FALLBACK_MODEL_PATH}: {e}")
        sys.exit(1)

if clf is None:
    print("[FATAL] No classifier found. Train a model first.")
    sys.exit(1)

# TTS : set to False if you don't want audio
USE_TTS = True
engine = pyttsx3.init() if USE_TTS else None

def normalize_text(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def rule_override(clean_text):
    if any(w in clean_text.split() for w in ("hello","hi","hey","goodmorning","goodevening")):
        return "greeting", 1.0
    if "time" in clean_text or "what time" in clean_text:
        return "get_time", 1.0
    if "date" in clean_text or "today" in clean_text:
        return "get_date", 1.0
    if any(x in clean_text for x in ("youtube", "you tube")):
        return "open_youtube", 1.0
    if any(x in clean_text for x in ("volume up","increase volume","louder")):
        return "increase_volume", 1.0
    if any(x in clean_text for x in ("volume down","decrease volume","quieter")):
        return "decrease_volume", 1.0
    return None, None

def map_intent_to_reply(intent):
    if intent == "get_time":
        return time.strftime("The time is %H:%M")
    if intent == "get_date":
        return time.strftime("Today is %Y-%m-%d")
    if intent == "greeting":
        return "Hello! How can I help?"
    if intent == "turn_on_light":
        return "Okay, turning on the light."
    if intent == "turn_off_light":
        return "Okay, turning off the light."
    if intent == "play_music":
        return "Playing music."
    if intent == "stop_music":
        return "Stopping music."
    if intent == "increase_volume":
        return "Increasing volume."
    if intent == "decrease_volume":
        return "Decreasing volume."
    if intent == "open_youtube":
        webbrowser.open("https://www.youtube.com")
        return "Opening YouTube."
    if intent == "open_google":
        webbrowser.open("https://www.google.com")
        return "Opening google."
    return f"Recognized intent: {intent}"

def speak(text):
    if engine:
        engine.say(text)
        engine.runAndWait()

def get_prob_for_prediction(clf, X):
    """
    Return top probability if classifier supports predict_proba.
    If not supported, returns None.
    """
    try:
        probs = clf.predict_proba(X)
        if probs is not None:
            # return max probability for predicted class
            return float(np.max(probs, axis=1)[0])
    except Exception:
        return None
    return None

def main():
    print("Text assistant (type 'exit' to quit).")
    while True:
        try:
            txt = input("You: ").strip()
            if not txt:
                continue
            if txt.lower() in ("exit","quit"):
                break
            clean = normalize_text(txt)

            intent, prob = rule_override(clean)
            probs = None
            if intent is None:
                X = vect.transform([clean])
                intent = clf.predict(X)[0]
                prob = get_prob_for_prediction(clf, X)

            reply = map_intent_to_reply(intent)
            print("Agent:", reply, f"(intent={intent}, prob={prob})")
            speak(reply)
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
