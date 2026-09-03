# 🎙️ Production-Grade Offline Voice Assistant with Leakage-Free Intent Classification

An end-to-end, privacy-preserving, offline voice assistant built with real-time speech recognition (Vosk), robust intent classification (scikit-learn Pipelines), decoupled action dispatching (`ActionRouter`), and local text-to-speech (`pyttsx3`).

This project is engineered with statistical rigor: eliminating TF-IDF data leakage, using stratified 80/20 train/test evaluation with 5-fold cross-validation on training data, data-driven confidence threshold tuning, two-tier out-of-domain rejection, and granular latency profiling.

---

## 🏗️ System Architecture

```text
                    USER SPEECH
                         │
                         ▼
                ┌─────────────────┐
                │ Microphone      │  (sounddevice @ 16kHz, 16-bit PCM)
                │ sounddevice     │
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ WebRTC VAD      │  (Aggressiveness: 2, 300ms silence hangover)
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ Vosk ASR        │  (Offline Kaldi acoustic & language model)
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ Text Normalize  │  (Regex cleaning, case folding)
                └────────┬────────┘
                         ▼
          ┌─────────────────────────────┐
          │ TF-IDF + ML Pipeline        │  (sklearn.pipeline.Pipeline)
          │  - Vectorizer: n-gram (1,2) │
          │  - Classifier: Linear SVM   │  (CalibratedClassifierCV)
          └──────────────┬──────────────┘
                         ▼
                 Confidence Check
                    /          \
        ≥ 0.35 (HIGH)          < 0.35 (LOW)
             │                       │
             ▼                       ▼
    Intent == "unknown"?       Clarification Prompt
        /         \            ("Did you mean X, Y, or Z?")
      YES          NO
       │            │
       ▼            ▼
   OOD Handler  In-Domain Intent
       │            │
       └─────┬──────┘
             ▼
      ┌──────────────┐
      │ ActionRouter │  (Decoupled execution & safe mock mode)
      └──────┬───────┘
             ▼
          Response
             │
             ▼
        pyttsx3 TTS     (Local speech synthesis)
             │
             ▼
        USER HEARS
```

Alongside live execution runs a continuous telemetry pipeline:
```text
Production Telemetry: Interaction Duration | Processing Latency (ASR + NLU + Action + TTS) | Fallback Logging
Offline Evaluation:   Held-out Confusion Matrix | Precision/Recall/F1 per intent | Misclassification Profiling
```

---

## 📁 Project Structure

```bash
voice_assistant/
├── assistant.py               # Main voice-activated assistant with VAD & live ASR
├── assistant_text.py          # Interactive text interface & pipeline benchmark
├── actions.py                 # Modular ActionRouter with safe mock execution
├── train.py                   # Leakage-free 80/20 split & 5-fold CV training pipeline
├── tune_threshold.py          # Empirical confidence threshold tuning on train folds
├── intents.csv                # Balanced dataset (532 samples, 19 intents, 28/intent)
├── requirements.txt           # Pinned project dependencies
├── models/
│   ├── best_model.joblib      # Winning pipeline (TfidfVectorizer + Calibrated LinearSVM)
│   ├── vectorizer.joblib      # Extracted TF-IDF vectorizer (backward compatibility)
│   ├── results_summary.csv    # Cross-validation comparison across all 4 candidate models
│   ├── test_metrics.csv       # Final evaluation metrics on held-out test set
│   ├── test_error_analysis.csv# Row-level misclassification breakdown on test set
│   ├── confusion_matrix_test.csv # Raw confusion matrix counts on test set
│   ├── threshold_analysis.csv # Threshold trade-off data (0.20 to 0.80)
│   └── tuned_threshold.txt    # Persisted optimal confidence cutoff (0.35)
├── model_compare_outputs/
│   ├── confusion_matrix.png   # Matplotlib heatmap on held-out test set
│   ├── model_comparison.png   # Accuracy, F1, and inference latency comparison
│   └── threshold_tradeoff.png # Correct Acceptance vs False Acceptance vs Fallback curve
├── tests/
│   └── test_pipeline.py       # Automated unit tests (preprocessing, routing, safety, OOD)
├── logs/
│   └── utterances.csv         # Telemetry log with component latency breakdowns
├── vosk-model/                # Offline Kaldi speech recognition model directory
└── README.md                  # System documentation & technical defense guide
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites & Virtual Environment
Requires **Python 3.8 to 3.12** on Windows, Linux, or macOS.

```bash
git clone https://github.com/lalankishor27-collab/voice_assistant.git
cd voice_assistant

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Speech Recognition Model
Download the compact offline Vosk English model (~40 MB) from [AlphaCephei](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip), extract it, and place it in the `vosk-model/` directory.

### 4. Train the Leakage-Free Pipeline
```bash
python train.py
```

### 5. Tune the Confidence Threshold
```bash
python tune_threshold.py
```

### 6. Run Automated Unit Tests
```bash
python -m unittest discover tests
```

### 7. Run the Assistant
- **Interactive Text Mode (No Microphone Required):**
  ```bash
  python assistant_text.py
  ```
- **Live Voice-Activated Mode:**
  ```bash
  python assistant.py
  ```

---

## 📊 Rigorous Machine Learning Evaluation

### 1. Elimination of TF-IDF Data Leakage
In naive implementations, TF-IDF vectorization is often fit over the entire dataset before cross-validation. This leaks the vocabulary, document frequencies, and inverse document frequency (IDF) weights from validation folds into training folds.

**Our Solution:** We encapsulate `TfidfVectorizer` and each candidate classifier inside an `sklearn.pipeline.Pipeline`. Preprocessing and vocabulary extraction are executed strictly on the training fold during each CV iteration:

```python
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=12000, min_df=1)),
    ("classifier", CalibratedClassifierCV(LinearSVC(class_weight="balanced")))
])
```

### 2. Evaluation Protocol
- **Dataset:** 532 curated, balanced utterances across 19 intents (28 samples per class).
- **Split:** Stratified 80% Training (425 samples) / 20% Held-out Test (107 samples).
- **Cross-Validation:** 5-fold Stratified CV executed **strictly on the 80% training set**.
- **Model Selection Metric:** Primary: **CV Macro-F1** (protects against class imbalance / tail intents); Secondary: **CV Accuracy**.
- **Final Evaluation:** The selected best model is evaluated **once** on the untouched 20% held-out test set.

### 3. Empirical Model Comparison (5-Fold CV on 80% Train Set)

| Model | CV Accuracy | CV Macro-F1 | CV Weighted-F1 | Fit Time | Latency / Sample | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Calibrated)** | **91.06%** | **91.16%** | **91.20%** | **0.31s** | **3.88 ms** | **793.3 KB** |
| Logistic Regression | 88.71% | 88.57% | 88.61% | 0.05s | 0.37 ms | 191.0 KB |
| Multinomial Naive Bayes | 86.59% | 86.13% | 86.16% | 0.004s | 0.32 ms | 338.8 KB |
| K-Nearest Neighbors (k=5) | 77.18% | 76.69% | 76.73% | 0.004s | 0.77 ms | 76.0 KB |

> **Best Model Selected:** `LinearSVM` achieved the highest Macro-F1 (**0.9116**) and Accuracy (**0.9106**).

### 4. Unbiased Evaluation on Held-Out Test Set (107 Samples)

| Metric | Empirical Result |
| :--- | :---: |
| **Test Accuracy** | **90.65%** (97 / 107 correct) |
| **Test Macro-F1** | **90.53%** |
| **Test Weighted-F1** | **90.66%** |

#### Confusion Matrix & Error Analysis Insights
The confusion matrix heatmap (`model_compare_outputs/confusion_matrix.png`) identifies specific semantic overlaps:
1. `turn_on_light` ↔ `turn_off_light`: Utterances like *"light the room"* or *"activate the lights"* share high lexical overlap with "light" and "lights".
2. `decrease_volume` ↔ `increase_volume`: *"volume needs to go down"* predicted `increase_volume` due to the strong unigram "volume", but with a low confidence score of **0.36**.
3. **Low-Confidence Safeguard:** Over 50% of test misclassifications produced confidence scores below the **0.40** mark, meaning that in production, the confidence filter intercepts them and requests clarification rather than executing an erroneous command.

---

## 🎯 Evidence-Based Confidence Threshold Tuning

Rather than guessing an arbitrary threshold like `0.45`, we run a sweep on **out-of-fold cross-validation probabilities on the training data**:

- **Accepted:** Prediction has $\text{confidence} \ge \tau$
- **Correct Acceptance (True Acceptance):** $\hat{y} = y$ and $\text{confidence} \ge \tau$
- **False Acceptance (Erroneous Execution):** $\hat{y} \ne y$ and $\text{confidence} \ge \tau$
- **Fallback Rate:** $\text{confidence} < \tau$ (routes to clarification)

| Threshold ($\tau$) | Acceptance Rate | Correct Acceptance | False Acceptance | Fallback Rate | Precision When Accepted |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.20 | 99.1% | 91.1% | 8.0% | 0.9% | 91.9% |
| 0.25 | 97.2% | 90.4% | 6.8% | 2.8% | 93.0% |
| 0.30 | 95.5% | 88.9% | 6.6% | 4.5% | 93.1% |
| **0.35 (Optimal)** | **93.2%** | **87.8%** | **5.4%** | **6.8%** | **94.2%** |
| 0.40 | 88.7% | 84.5% | 4.2% | 11.3% | 95.2% |
| 0.45 | 85.9% | 82.6% | 3.3% | 14.1% | 96.2% |
| 0.50 | 80.0% | 76.9% | 3.1% | 20.0% | 96.2% |

> **Conclusion:** Setting the threshold at **0.35** yields an optimal balance: **87.8% Correct Acceptance**, only **5.4% False Acceptance**, while maintaining **94.2% Precision** upon acceptance and keeping the Fallback Rate under **7%**.

---

## ⏱️ Latency Profiling: Where is the Real Bottleneck?

Every turn is instrumented with sub-millisecond timers. We explicitly distinguish between **Interaction Latency** and **Processing Latency**:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TOTAL LATENCY BREAKDOWN                          │
├──────────────────────────┬──────────────────────────────────────────────────┤
│ Stage                    │ Typical Latency                                  │
├──────────────────────────┼──────────────────────────────────────────────────┤
│ 1. User Speech Duration  │ 1200 ms – 2500 ms (depends on utterance length)  │
│ 2. VAD Silence Hangover  │ 300 ms (guarantees sentence completion)          │
│ ──────────────────────── │ ──────────────────────────────────────────────── │
│ Interaction Subtotal     │ ~1500 ms – 2800 ms (Physical speech duration)   │
├──────────────────────────┼──────────────────────────────────────────────────┤
│ 3. Vosk ASR Decoding     │ 250 ms – 450 ms (Algorithmic Bottleneck!)        │
│ 4. Text Preprocessing    │ 0.05 ms – 0.2 ms                                 │
│ 5. Pipeline Inference    │ 2.0 ms – 4.0 ms (Linear SVM)                     │
│ 6. ActionRouter Dispatch │ 0.01 ms – 0.05 ms                                │
│ 7. pyttsx3 TTS Synthesis │ 150 ms – 300 ms                                  │
├──────────────────────────┼──────────────────────────────────────────────────┤
│ Processing Latency       │ 400 ms – 750 ms (Total system computation time)  │
└──────────────────────────┴──────────────────────────────────────────────────┘
```

> **Interview Insight:** Machine learning inference is **not** the latency bottleneck—it takes only **~3.8 ms** (<1% of processing time). The dominant computational bottleneck is acoustic speech recognition (**Vosk ASR @ ~300 ms**), followed by audio speech synthesis (**TTS @ ~200 ms**).

---

## 🛡️ Two-Tier Out-of-Domain (OOD) Strategy

Standard multi-class classifiers suffer from the closed-world assumption: forced to pick one in-domain intent even when given completely alien queries. We implement a **two-tier defense**:

1. **Tier 1 (Explicit OOD Intent):** The model is trained on an explicit `unknown` class comprising out-of-domain queries (e.g. *"what is quantum computing"*, *"recipe for cookies"*, *"book a flight to London"*).
2. **Tier 2 (Confidence Threshold Gating):** If any query produces a maximum posterior probability below $\tau = 0.35$, the system refuses blind execution and triggers a clarification prompt listing the top candidate intents.

---

## ⚙️ Modular `ActionRouter` & Safety Demonstration

The NLU classifier is completely decoupled from execution logic via `ActionRouter`:

```python
class ActionRouter:
    def dispatch(self, intent: str, context: dict = None) -> dict:
        ...
```

### Safety Guard for Destructive Commands
For interview demos and safe evaluations, destructive commands (`shutdown_device`, `restart_device`) execute in **safe mock simulation mode**:
```python
def _handle_shutdown(self, ctx):
    return {
        "success": True,
        "action": "shutdown_device",
        "reply": "Shutdown command recognized. System shutdown prevented in demonstration mode.",
        "metadata": {"simulated": True}
    }
```
This demonstrates scalable architectural design without risking unintentional laptop shutdowns during a live interview demo.

---

## 💡 MathCo Technical Interview Talking Points

| Question / Attack Angle | Defensible Technical Answer |
| :--- | :--- |
| **"Did you have data leakage in cross-validation?"** | *"No. In earlier iterations, TF-IDF was fit on the full dataset, which leaks vocabulary and IDF weights across folds. I refactored the pipeline using `sklearn.pipeline.Pipeline`, ensuring vectorization and classifier training are strictly isolated inside each training fold."* |
| **"Why prioritize Macro-F1 over Accuracy?"** | *"In intent classification, rare or safety-critical commands (e.g., alarms or device controls) can be starved if evaluated purely on accuracy. Macro-F1 weights every class equally regardless of sample frequency, exposing poor performance on minority intents."* |
| **"Why not use an LLM or Sentence Transformers?"** | *"For an offline, real-time voice assistant running on edge hardware, local latency and compute footprint are critical. TF-IDF + Calibrated Linear SVM runs in under 4ms with a model size under 1MB, requiring no GPU and zero external API dependencies. An LLM or heavy transformer would introduce 500ms–2000ms latency and high memory requirements for an already bounded 19-intent classification task."* |
| **"How did you choose your confidence threshold?"** | *"I did not pick an arbitrary number. I swept thresholds from 0.20 to 0.80 on out-of-fold cross-validation probabilities on the training data. At 0.35, the system achieved 87.8% true acceptance and 94.2% precision upon acceptance, while suppressing false executions to 5.4%."* |
| **"Where is the system bottleneck?"** | *"Instrumentation reveals that ML inference takes ~3.8 ms (<1% of processing time). The dominant delay is Vosk acoustic ASR decoding (~350 ms) and TTS synthesis (~200 ms). Total interaction latency is primarily determined by physical human speech duration."* |

---

## 📄 License & Author

- **Author:** Lalan Kishor (MCA - AI & IoT, NIT Patna)
- **GitHub:** [@lalankishor27-collab](https://github.com/lalankishor27-collab)
- **License:** MIT
