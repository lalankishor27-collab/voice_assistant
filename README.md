# 🎙️ Voice Assistant with Intent Classification

A Python-based offline voice assistant that performs real‑time speech recognition, intent detection, and spoken responses. It supports both **voice input** (using Vosk for offline ASR) and **text input** for debugging.

## ✨ Features

- 🎤 **Real‑time voice capture** with Voice Activity Detection (VAD)
- 🧠 **Intent classification** using scikit‑learn (Logistic Regression, SVM, Naive Bayes, KNN)
- 🗣️ **Offline speech recognition** with Vosk (no internet required)
- 🔊 **Text‑to‑speech** responses via `pyttsx3`
- 📝 **Rule‑based overrides** for high‑confidence commands (greetings, time, volume)
- 📊 **Model evaluation** with cross‑validation and automatic best‑model selection
- 📁 **Logging** of all utterances and predictions to CSV
- 💻 **Text‑based interface** for testing without a microphone

## 📁 Project Structure
```bash
voice_assistant/
├── assistant.py 
├── assistant_text.py 
├── train.py 
├── intents.csv
├── requirements.txt
├── models/ 
│ ├── vectorizer.joblib
│ ├── best_model.joblib
│ ├── LogisticRegression.joblib
│ ├── LinearSVM.joblib
│ ├── MultinomialNB.joblib
│ ├── KNN.joblib
│ └── results_summary.csv
├── logs/
│ └── utterances.csv
├── vosk-model/
└── README.md
```


## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/lalankishor27-collab/voice_assistant.git
cd voice_assistant
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
```
# Windows
```bash
.venv\Scripts\activate
```
# Linux/macOS
```bash
source .venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Download the Vosk speech recognition model
- The assistant requires a Vosk model for offline ASR. Download a small English model:

- [vosk-model-small-en-us-0.15 (40 MB)](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip)

- Extract the contents into a folder named vosk-model in the project root.

- Your folder should look like:
```
vosk-model/
├── am/
├── conf/
├── graph/
├── ivector/
└── README
```

### 5. Train the intent classifier
```
python train.py
```
## 🎮 Usage
## Voice‑Activated Mode
```
python assistant.py
```
## Speak after the prompt. The assistant will:

-Detect speech using VAD

-Transcribe with Vosk

-Predict intent

-Respond via TTS

-Press Ctrl+C to exit.

## Text‑Based Mode (No Microphone)
```
python assistant_text.py
```

## Supported Intents (examples)

## Command Examples

| Command Examples                     | Intent            |
|------------------------------------|------------------|
| "hello", "hi", "good morning"      | greeting         |
| "what time is it?"                 | get_time         |
| "what is the date?"                | get_date         |
| "turn on the light"                | turn_on_light    |
| "play music"                       | play_music       |
| "volume up", "make it louder"      | increase_volume  |
| "open YouTube"                     | open_youtube     |
| "set alarm for 7 am"               | set_alarm        |

...and many more (see intents.csv)
## 🧪 Model Performance

- After training, check models/results_summary.csv for cross‑validation metrics.

## Example output:

## Model Performance

| Model               | Accuracy | F1 Macro | Train Time |
|--------------------|----------|----------|------------|
| LogisticRegression | 0.987    | 0.986    | 0.45s      |
| LinearSVM          | 0.991    | 0.990    | 0.62s      |
| MultinomialNB      | 0.964    | 0.962    | 0.03s      |
| KNN                | 0.951    | 0.949    | 0.02s      |

The best model is automatically saved as best_model.joblib.

## 📝 Adding New Intents

1. Add new rows to intents.csv with the format:
```
your new command phrase,new_intent_name
```
2.  the rule‑based overrides in assistant.py and assistant_text.py if desired.

3. Add a response handler in the map_intent_to_reply function.

4. Retrain the model:
```
python train.py
```

## 📦 Dependencies

- Python 3.8+

- sounddevice – audio capture

- vosk – offline speech recognition

- webrtcvad – voice activity detection

- pyttsx3 – text‑to‑speech

- scikit‑learn – machine learning

- numpy, pandas, joblib – data processing and model persistence

- See requirements.txt for exact versions.

## 📄 License

This project is for educational purposes. Feel free to modify and use it as a base for your own voice assistant projects.

## 🙋 Author

Lalan Kishor

MCA (AI & IoT) – NIT Patna

GitHub: @lalankishor27-collab

