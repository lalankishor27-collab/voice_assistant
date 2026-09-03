# assistant.py
"""
Offline Voice Assistant: Real-Time Audio Capture, Vosk ASR, ML Pipeline & ActionRouter
- Real-time VAD voice segment collection
- Vosk offline acoustic decoding
- Leakage-free trained Pipeline inference (TF-IDF + Calibrated Model)
- Two-Tier OOD & Clarification (Tuned Threshold + 'unknown' class)
- Decoupled ActionRouter execution with safe simulation for destructive commands
- Detailed latency telemetry: Interaction Latency vs Algorithmic Processing Latency
"""

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

from actions import ActionRouter

# -------------------------
# Config
# -------------------------
FS = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(FS * FRAME_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * 2
VAD_AGGRESSIVENESS = 2
SILENCE_FRAMES_AFTER_SPEECH = int(300 / FRAME_MS)
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "utterances.csv")
VOSK_MODEL_DIR = "vosk-model"
MODELS_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
TUNED_THRESH_PATH = os.path.join(MODELS_DIR, "tuned_threshold.txt")
STREAM_BLOCKSIZE = FRAME_SAMPLES

# 1. Load Tuned Confidence Threshold
CONF_THRESHOLD = 0.35
if os.path.exists(TUNED_THRESH_PATH):
    try:
        with open(TUNED_THRESH_PATH, "r") as f:
            CONF_THRESHOLD = float(f.read().strip())
    except Exception:
        pass
print(f"[INFO] Using confidence threshold: {CONF_THRESHOLD:.2f}")

os.makedirs(LOG_DIR, exist_ok=True)
LOG_HEADER = [
    "timestamp", "asr_text", "clean_text", "pred_intent", "confidence", "top_k",
    "fallback_triggered", "vad_speech_duration_ms", "asr_latency_ms",
    "inference_latency_ms", "action_latency_ms", "tts_latency_ms",
    "total_processing_latency_ms", "total_interaction_latency_ms", "model_used"
]
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(LOG_HEADER)

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
# Load Pipeline & ActionRouter
# -------------------------
if not os.path.exists(BEST_MODEL_PATH):
    raise SystemExit(f"Model not found at {BEST_MODEL_PATH}. Run training first.")

pipeline = joblib.load(BEST_MODEL_PATH)
model_name_loaded = os.path.basename(BEST_MODEL_PATH)
print(f"[INFO] Loaded pipeline from {BEST_MODEL_PATH}")

router = ActionRouter(safe_mode=True)

# -------------------------
# TTS and stream control
# -------------------------
engine = pyttsx3.init()
tts_lock = threading.Lock()
stream_obj = None

def speak(text: str) -> float:
    """Speak text through TTS and return elapsed milliseconds."""
    global stream_obj
    t0 = time.perf_counter()
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
    return (time.perf_counter() - t0) * 1000

# -------------------------
# Production Telemetry Logging
# -------------------------
def log_turn(
    asr_text: str,
    clean_text: str,
    pred_intent: str,
    confidence: float,
    top_k: str,
    fallback_triggered: bool,
    vad_ms: float,
    asr_ms: float,
    inf_ms: float,
    act_ms: float,
    tts_ms: float,
    total_proc_ms: float,
    total_interact_ms: float
):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.datetime.now().isoformat(),
            asr_text,
            clean_text,
            pred_intent,
            round(float(confidence), 4) if confidence is not None else "",
            top_k,
            fallback_triggered,
            round(vad_ms, 2),
            round(asr_ms, 2),
            round(inf_ms, 2),
            round(act_ms, 2),
            round(tts_ms, 2),
            round(total_proc_ms, 2),
            round(total_interact_ms, 2),
            model_name_loaded
        ])

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
    start_time = time.perf_counter()

    while True:
        try:
            chunk = audio_q.get(timeout=timeout)
        except queue.Empty:
            return None, 0.0

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

        if timeout is not None and (time.perf_counter() - start_time) > timeout:
            if triggered and frames:
                break
            return None, 0.0

    if not frames:
        return None, 0.0

    vad_duration_ms = (time.perf_counter() - start_time) * 1000
    return b"".join(frames), vad_duration_ms

# -------------------------
# NLU helpers
# -------------------------
def normalize_text(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# -------------------------
# Process segment and execute action
# -------------------------
def process_segment_and_respond(segment_bytes: bytes, vad_duration_ms: float):
    t_proc_start = time.perf_counter()

    # 1. ASR Decoding
    t_asr_start = time.perf_counter()
    rec = KaldiRecognizer(vosk_model, FS)
    rec.AcceptWaveform(segment_bytes)
    res = rec.Result()
    j = json.loads(res)
    asr_text = j.get("text", "").strip()
    asr_ms = (time.perf_counter() - t_asr_start) * 1000
    print(f"\n[ASR] -> \"{asr_text}\" (Decoded in {asr_ms:.1f} ms)")

    if not asr_text:
        tts_ms = speak("I didn't catch that. Could you repeat?")
        total_proc_ms = (time.perf_counter() - t_proc_start) * 1000
        log_turn("", "", "empty_asr", 0.0, "", True, vad_duration_ms, asr_ms, 0.0, 0.0, tts_ms, total_proc_ms, vad_duration_ms + total_proc_ms)
        return

    clean_text = normalize_text(asr_text)

    # 2. Rule-based Fast-Path Overrides
    rule_intent = None
    if any(w in clean_text.split() for w in ("hello", "hi", "hey", "goodmorning", "goodevening")):
        rule_intent = "greeting"
    elif "what time" in clean_text or clean_text == "time":
        rule_intent = "get_time"

    # 3. Pipeline Inference
    t_inf_start = time.perf_counter()
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([clean_text])[0]
        classes = pipeline.classes_
        top_idx = np.argsort(probs)[::-1]
        predicted_intent = classes[top_idx[0]]
        confidence = float(probs[top_idx[0]])
        top_k = [(classes[i], float(probs[i])) for i in top_idx[:3]]
    else:
        predicted_intent = pipeline.predict([clean_text])[0]
        confidence = 1.0
        top_k = [(predicted_intent, 1.0)]
    inf_ms = (time.perf_counter() - t_inf_start) * 1000

    if rule_intent:
        final_intent = rule_intent
        confidence = 1.0
        is_override = True
    else:
        final_intent = predicted_intent
        is_override = False

    top_k_str = "; ".join([f"{name}: {prob:.2f}" for name, prob in top_k])
    print(f"[NLU] Intent: {final_intent} (Conf: {confidence:.2f}{' [Rule Override]' if is_override else ''})")
    print(f"[NLU] Candidates: {top_k_str}")

    # 4. Action Dispatch (Two-Tier OOD / Low Confidence Handling)
    t_act_start = time.perf_counter()
    fallback_triggered = False

    if confidence < CONF_THRESHOLD:
        # Tier 2: Low Confidence Fallback -> Ask for clarification
        fallback_triggered = True
        options = ", or ".join([name for name, _ in top_k[:3]])
        reply = f"I am not sure what you meant. Did you mean {options}?"
        act_ms = (time.perf_counter() - t_act_start) * 1000
    elif final_intent == "unknown":
        # Tier 1: Explicit Out-of-Domain Intent
        fallback_triggered = True
        action_res = router.dispatch("unknown", {"text": clean_text})
        reply = action_res["reply"]
        act_ms = (time.perf_counter() - t_act_start) * 1000
    else:
        # In-domain confident intent
        action_res = router.dispatch(final_intent, {"text": clean_text})
        reply = action_res["reply"]
        act_ms = (time.perf_counter() - t_act_start) * 1000

    print(f"[ACTION] Reply: \"{reply}\"")

    # 5. TTS Response
    tts_ms = speak(reply)

    # 6. Latency Accounting
    total_proc_ms = (time.perf_counter() - t_proc_start) * 1000
    total_interaction_ms = vad_duration_ms + total_proc_ms

    print(f"[LATENCY] Speech Duration: {vad_duration_ms:.0f}ms | Processing: {total_proc_ms:.1f}ms "
          f"(ASR: {asr_ms:.1f}ms, Inf: {inf_ms:.2f}ms, Act: {act_ms:.2f}ms, TTS: {tts_ms:.1f}ms) | Total: {total_interaction_ms:.0f}ms")

    # 7. Production Telemetry Logging
    try:
        log_turn(
            asr_text, clean_text, final_intent, confidence, top_k_str,
            fallback_triggered, vad_duration_ms, asr_ms, inf_ms, act_ms,
            tts_ms, total_proc_ms, total_interaction_ms
        )
    except Exception as e:
        print(f"[WARN] Telemetry logging failed: {e}")

# -------------------------
# Main loop & stream
# -------------------------
def main_loop():
    print(f"\n[INFO] Listening... Speak clearly into your microphone. (Ctrl+C to exit)")
    try:
        while True:
            segment, vad_duration_ms = vad_collect_speech(timeout=10.0)
            if segment is None:
                continue
            process_segment_and_respond(segment, vad_duration_ms)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Exiting on user interrupt.")
    except Exception:
        print("[ERROR] Exception in main loop:")
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
    print("==========================================================")
    print("🎙️ Offline Voice Assistant (MathCo Architecture)")
    print("==========================================================")
    start_stream_and_run()
