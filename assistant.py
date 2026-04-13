import os
import sys
import queue
import json
import time
import threading
import datetime
import csv
import traceback
import re

import sounddevice as sd
import numpy as np
import webrtcvad
from vosk import Model, KaldiRecognizer
import joblib
import pyttsx3

# -------------------------
# Config
# -------------------------
FS = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(FS * FRAME_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * 2
VAD_AGGRESSIVENESS = 2
SILENCE_FRAMES_AFTER_SPEECH = int(300 / FRAME_MS)
CONF_THRESHOLD = 0.45
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "utterances.csv")
VOSK_MODEL_DIR = "vosk-model"
MODELS_DIR = "models"
STREAM_BLOCKSIZE = FRAME_SAMPLES

os.makedirs(LOG_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "asr_text", "clean_text", "pred_intent", "prob", "top_k", "model_used"])

# -------------------------
# Load VOSK model
# -------------------------
abs_vosk_path = os.path.abspath(VOSK_MODEL_DIR)
if not os.path.exists(abs_vosk_path):
    raise SystemExit(f"VOSK model folder not found at: {abs_vosk_path}")

try:
    vosk_model = Model(abs_vosk_path)
    print(f"[INFO] Loaded VOSK model from {abs_vosk_path}")
except Exception:
    traceback.print_exc()
    raise

# -------------------------
# Load vectorizer and classifier (prefer best_model)
# -------------------------
vect_path = os.path.join(MODELS_DIR, "vectorizer.joblib")
best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
clf_fallback_path = os.path.join(MODELS_DIR, "intent_clf.joblib")

if not os.path.exists(vect_path):
    raise SystemExit(f"Vectorizer not found at {vect_path}. Run training first.")

vect = joblib.load(vect_path)
print(f"[INFO] Loaded vectorizer from {vect_path}")

clf = None
model_name_loaded = None
if os.path.exists(best_model_path):
    try:
        clf = joblib.load(best_model_path)
        model_name_loaded = os.path.basename(best_model_path)
        print(f"[INFO] Loaded classifier from {best_model_path}")
    except Exception as e:
        print(f"[WARN] Failed to load {best_model_path}: {e}")

if clf is None and os.path.exists(clf_fallback_path):
    try:
        clf = joblib.load(clf_fallback_path)
        model_name_loaded = os.path.basename(clf_fallback_path)
        print(f"[INFO] Loaded classifier from fallback {clf_fallback_path}")
    except Exception as e:
        print(f"[FATAL] Failed to load fallback classifier {clf_fallback_path}: {e}")
        sys.exit(1)

if clf is None:
    print("[FATAL] No classifier found. Train a model first.")
    sys.exit(1)

# -------------------------
# TTS and stream control
# -------------------------
engine = pyttsx3.init()
tts_lock = threading.Lock()
stream_obj = None

def speak(text: str):
    global stream_obj
    with tts_lock:
        try:
            if stream_obj is not None and stream_obj.active:
                stream_obj.stop()
        except Exception:
            pass
        engine.say(text)
        engine.runAndWait()
        try:
            if stream_obj is not None:
                stream_obj.start()
        except Exception:
            pass

# -------------------------
# Logging
# -------------------------
def log_turn(asr_text: str, clean_text: str, pred_intent: str, prob: float, top_k: str):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.now().isoformat(), asr_text, clean_text, pred_intent, float(prob) if prob is not None else "", top_k, model_name_loaded])

# -------------------------
# Audio capture
# -------------------------
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("[AUDIO STATUS]", status, file=sys.stderr)
    try:
        audio_q.put_nowait(indata.copy().tobytes())
    except queue.Full:
        pass

# -------------------------
# VAD collect
# -------------------------
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

def vad_collect_speech(timeout=None):
    frames = []
    triggered = False
    silent_frames = 0
    start_time = time.time()

    while True:
        try:
            chunk = audio_q.get(timeout=timeout)
        except queue.Empty:
            return None

        if len(chunk) < FRAME_BYTES:
            chunk = chunk.ljust(FRAME_BYTES, b'\x00')
        elif len(chunk) > FRAME_BYTES:
            chunk = chunk[:FRAME_BYTES]

        is_speech = False
        try:
            is_speech = vad.is_speech(chunk, FS)
        except Exception:
            is_speech = True

        if is_speech:
            frames.append(chunk)
            triggered = True
            silent_frames = 0
        else:
            if triggered:
                silent_frames += 1
                frames.append(chunk)
                if silent_frames > SILENCE_FRAMES_AFTER_SPEECH:
                    break
            else:
                pass

        if timeout is not None and (time.time() - start_time) > timeout:
            if triggered and frames:
                break
            return None

    if not frames:
        return None
    return b"".join(frames)

# -------------------------
# NLU helpers
# -------------------------
def normalize_text(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def get_top_k_from_probs(probs, k=3):
    idx = np.argsort(probs)[::-1][:k]
    return [(clf.classes_[i], float(probs[i])) for i in idx]

def safe_predict_proba(clf, X):
    """Return prob array if available, else None."""
    try:
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X)[0]
    except Exception:
        return None
    return None

# -------------------------
# Process segment
# -------------------------
def process_segment_and_respond(segment_bytes: bytes):
    rec = KaldiRecognizer(vosk_model, FS)
    rec.AcceptWaveform(segment_bytes)
    res = rec.Result()
    j = json.loads(res)
    asr_text = j.get("text", "").strip()
    print(f"[ASR] -> {asr_text!r}")

    if not asr_text:
        speak("I didn't catch that. Could you repeat?")
        return

    clean_text = normalize_text(asr_text)

    # RULE-BASED OVERRIDES (fast deterministic checks)
    intent = None
    top_prob = None
    probs = None
    # greetings
    if any(w in clean_text.split() for w in ("hello","hi","hey","goodmorning","goodevening")):
        intent = "greeting"; top_prob = 1.0
    # time
    elif "time" in clean_text or "what time" in clean_text:
        intent = "get_time"; top_prob = 1.0
    # volume up
    elif any(x in clean_text for x in ("volume up","increase volume","raise the volume","make it louder","louder")):
        intent = "increase_volume"; top_prob = 1.0
    # volume down
    elif any(x in clean_text for x in ("volume down","decrease volume","lower the volume","make it quieter","quieter")):
        intent = "decrease_volume"; top_prob = 1.0
    else:
        # fallback to classifier
        try:
            X = vect.transform([clean_text])
            intent = clf.predict(X)[0]
            probs = safe_predict_proba(clf, X)
            top_prob = float(np.max(probs)) if probs is not None else None
        except Exception as e:
            print("[ERROR] NLU processing failed:", e)
            intent = "unknown"
            top_prob = 0.0

    # compute top-k for logging / clarification (if available)
    top_k_str = ""
    if probs is not None:
        top_k = get_top_k_from_probs(probs, k=3)
        top_k_str = ";".join([f"{p}:{prob:.3f}" for p,prob in top_k])
    else:
        top_k = [(intent, top_prob)]

    print("[NLU] intent:", intent, "prob:", top_prob, "top_k:", top_k)

    # log
    try:
        log_turn(asr_text, clean_text, intent, top_prob, top_k_str)
    except Exception as e:
        print("[WARN] Failed to log turn:", e)

    # low confidence handling
    if top_prob is None or top_prob < CONF_THRESHOLD:
        if probs is not None:
            top3 = get_top_k_from_probs(probs, k=3)
            options = ", or ".join([t for t,_ in top3])
            speak(f"I am not sure what you meant. Did you mean {options}? Please say yes or repeat.")
            print("[NLU] Asked for clarification. top3:", top3)
            return
        else:
            speak("I am not sure what you meant. Could you please repeat or clarify?")
            return

    # map intents -> replies / actions
    if intent == "get_time":
        reply = time.strftime("The time is %H:%M")
    elif intent == "greeting":
        reply = "Hello! How can I help?"
    elif intent == "turn_on_light":
        reply = "Okay, turning on the light."
    elif intent == "turn_off_light":
        reply = "Okay, turning off the light."
    elif intent == "play_music":
        reply = "Playing music."
    elif intent == "stop_music":
        reply = "Stopping music."
    elif intent == "increase_volume":
        reply = "Increasing volume."
    elif intent == "decrease_volume":
        reply = "Decreasing volume."
    else:
        reply = f"Recognized intent {intent}."

    speak(reply)

# -------------------------
# Main loop & stream
# -------------------------
def main_loop():
    print("[INFO] Listening... Speak after the prompt. Press Ctrl+C to exit.")
    try:
        while True:
            segment = vad_collect_speech(timeout=10.0)
            if segment is None:
                continue
            process_segment_and_respond(segment)
            time.sleep(0.12)
    except KeyboardInterrupt:
        print("\n[INFO] Exiting on user interrupt.")
    except Exception:
        print("[ERROR] Unexpected exception in main loop:")
        traceback.print_exc()

def start_stream_and_run():
    global stream_obj
    try:
        stream_obj = sd.InputStream(
            samplerate=FS,
            channels=1,
            dtype='int16',
            blocksize=STREAM_BLOCKSIZE,
            callback=audio_callback
        )
        stream_obj.start()
        main_loop()
    finally:
        try:
            if stream_obj is not None:
                stream_obj.stop()
                stream_obj.close()
        except Exception:
            pass

if __name__ == "__main__":
    print("---- Voice Assistant (patched) ----")
    start_stream_and_run()
