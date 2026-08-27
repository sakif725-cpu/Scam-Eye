---
title: AI Fraud Detection System
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# 🛡️ AI Fraud Call and Message Detector

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Faster-Whisper](https://img.shields.io/badge/Faster--Whisper-CTranslate2-green.svg)](https://github.com/SYSTRAN/faster-whisper)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-DistilBERT-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An intelligent, multi-modal, privacy-preserving AI detection system that identifies fraudulent SMS, WhatsApp communications, and scam voice calls using a unified NLP and machine learning pipeline.

---

## 1. Project Overview & Problem Statement

Telecom and mobile messaging fraud causes billions of dollars in losses worldwide every year. Fraudulent schemes have evolved significantly:
- **Coercive KYC Scams**: Threatening immediate bank account or SIM suspension.
- **Credential & OTP Harvesting**: Deceiving users into sharing one-time passwords or MPINs.
- **Reverse UPI / QR Scams**: Tricking victims into entering PINs to "receive" funds.
- **Impersonation Vishing Calls**: Scammers pretending to be police officers, customs officials, or bank executives.

Traditional spam filters rely primarily on static word blacklists or caller ID heuristics, making them brittle against emerging phrasing, Hinglish/multilingual scams, and voice calls.

### The Solution
The **AI Fraud Call & Message Detector** provides an end-to-end protective layer:
1. **Text Messages**: Analyzed directly through safe feature normalization and NLP classification.
2. **Voice Audio & Calls**: Transcribed on-device via **Faster-Whisper** and routed into the identical NLP intelligence pipeline.
3. **Multi-Factor Risk Engine**: Combines machine learning probability with behavioral red-flag severity to generate a 4-tier Risk Level (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
4. **Explainable AI (XAI)**: Generates human-understandable evidence, highlighted triggers, and tailored cybersecurity safety advice.

---

## 2. Proposed System Architecture & Workflow

```
                               ┌─────────────────────────┐
                               │   Text Message Input    │
                               └────────────┬────────────┘
                                            │
                                            ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│ Call Audio (WAV/MP3/M4A)│ ──► │ Speech-to-Text (Whisper)│
└─────────────────────────┘     └────────────┬────────────┘
                                             │ (Transcript)
                                             ▼
                                ┌─────────────────────────┐
                                │ Safe Text Preprocessing │
                                └────────────┬────────────┘
                                             │
                                             ▼
                                ┌─────────────────────────┐
                                │    Active AI Model      │
                                │ (TF-IDF+LR / DistilBERT)│
                                └────────────┬────────────┘
                                             │ Probability & Prediction
                                             ▼
                                ┌─────────────────────────┐
                                │  Explainability Engine  │
                                │  (Pattern & Red Flags)  │
                                └────────────┬────────────┘
                                             │ Indicators
                                             ▼
                                ┌─────────────────────────┐
                                │ Real-Time Risk Engine   │
                                │ (LOW/MED/HIGH/CRITICAL) │
                                └────────────┬────────────┘
                                             │
                                             ▼
                                ┌─────────────────────────┐
                                │ Streamlit UI Dashboard  │
                                └─────────────────────────┘
```

---

## 3. Dataset Pipeline & Multilingual Handling

### Pipeline Stages
1. **Automated Fetching**: Attempts downloading public datasets (e.g. UCI SMS Spam Collection / Kaggle mirrors).
2. **Fallback Dataset**: If network access is restricted or download fails, the pipeline seamlessly loads `data/fallback_sample_data.csv` (a balanced set of 90+ realistic fraud and genuine scenarios in English and Hinglish).
3. **Safe Text Preprocessing**: Normalizes URLs (`http_url_token`), phone numbers (`phone_num_token`), currency values (`currency_amount_token`), and OTP tokens while preserving Hindi/Hinglish vocabulary (e.g., *khata*, *band*, *turant*, *inaam*, *paise*).
4. **Deduplication & Validation**: Eliminates duplicates and empty records.
5. **Stratified Splitting**: 70% Training, 15% Validation, 15% Held-out Test.

### Current Limitations
- The built-in fallback dataset is designed for structural verification and hackathon demonstration; production systems require hundreds of thousands of domain-specific samples.
- Hinglish transliteration variants (e.g., *turant* vs *turunt*) are captured via n-grams, but full colloquial semantic capture requires multilingual transformer fine-tuning (`xlm-roberta` / `indic-bert`).

---

## 4. AI Models & Architecture

### Baseline Models
- **TF-IDF + Logistic Regression**: Uses word and character n-grams `(1, 2)` with balanced class weighting. Delivers ultra-low latency (<5ms) and high interpretability.
- **TF-IDF + Multinomial Naive Bayes**: Lightweight probabilistic classifier for comparative evaluation.

### Deep Learning Transformer Model
- **DistilBERT (`distilbert-base-uncased`)**: Fine-tuned for binary sequence classification with PyTorch and Hugging Face Transformers. Leverages contextual attention to capture subtle grammatical coercion and deceptive context.

---

## 5. Evaluation Metrics Explained

| Metric | Formula | Meaning in Fraud Detection |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall correctness across all messages. |
| **Precision** | $\frac{TP}{TP + FP}$ | Out of all messages flagged as Fraud, how many were truly fraudulent? High precision avoids annoying false alarms. |
| **Recall (Crucial)** | $\frac{TP}{TP + FN}$ | Out of all actual fraudulent messages, how many did the AI catch? **This is the most critical metric**, because missing a fraud message (False Negative) can result in stolen money or compromised identity. |
| **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | The harmonic mean of precision and recall. |

---

## 6. How Faster-Whisper Works

[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) is a reimplementation of OpenAI's Whisper model using **CTranslate2**, a fast inference engine for Transformer models.
- **Speed & Memory**: Up to **4x faster** than vanilla PyTorch Whisper with **half the RAM consumption** using `int8` quantization on CPU and `float16` on GPU.
- **Voice Activity Detection (VAD)**: Built-in Silero VAD filters out background noise and silence before transcription.
- **100% Local**: Transcribes audio on-device without cloud API latency or subscription fees.

---

## 7. Installation & Setup

### Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, 3.13, or 3.14
- `virtualenv` or `venv`

### Step 1: Clone Repository & Create Virtual Environment
```bash
# Clone the repository
git clone <repository-url>
cd "message detection.py"

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 8. Dataset Preparation & Model Training

Run the complete pipeline from terminal:

```bash
# 1. Preprocess dataset (creates train, val, test splits)
python3 src/data_preprocessor.py

# 2. Train Baseline TF-IDF Models (Logistic Regression & Naive Bayes)
python3 src/train_baseline.py

# 3. Fine-Tune DistilBERT Transformer Model
python3 src/train_transformer.py

# 4. Evaluate Benchmark and Select Champion Model
python3 src/evaluate_models.py
python3 src/model_selector.py
```

---

## 9. Running the Application & Unit Tests

### Launch the Streamlit Web UI
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Run Automated Unit Tests
```bash
pytest tests/ -v
```

---

## 10. Voice Call / Audio Analysis

1. Navigate to the **📞 Call / Audio Analysis** tab in the Streamlit UI.
2. Upload a voice recording (`.wav`, `.mp3`, `.m4a`, `.ogg`) or use your microphone.
3. The system executes:
   $$\text{Audio Input} \longrightarrow \text{Faster-Whisper} \longrightarrow \text{Transcript} \longrightarrow \text{AI Classifier} \longrightarrow \text{Risk Engine} \longrightarrow \text{Report}$$
4. The transcript is displayed alongside detected red flags, confidence scores, and action advice.

---

## 11. Privacy & Security Considerations

- **Local Execution**: All inference (ML, DistilBERT, Faster-Whisper) executes locally on your hardware.
- **Zero Third-Party Data Transmission**: No message contents or voice audio are transmitted to paid commercial AI APIs.
- **Transient Audio Cleanup**: Uploaded audio files are processed in memory or ephemeral temporary buffers and deleted immediately after transcription.
- **No Permanent PII Storage**: The system operates statelessly without retaining user messages in database tables unless explicitly opted in.

---

## 12. Modular Extensibility

### Integrating FastAPI (Microservice Backend)
The core AI logic in `src/` has zero dependencies on Streamlit. You can add a FastAPI service in minutes:
```python
# api.py (Example future extension)
from fastapi import FastAPI
from pydantic import BaseModel
from src.predictor import predict_message

app = FastAPI(title="AI Fraud Detector API")

class MessagePayload(BaseModel):
    text: str

@app.post("/api/v1/analyze")
def analyze(payload: MessagePayload):
    return predict_message(payload.text)
```

### Future Real-Time Mobile / Android Integration
- **Android Call Screening Service**: Integrate via `CallScreeningService` or `InCallService` to capture caller metadata.
- **On-Device OnnxRuntime / TFLite**: Export the baseline TF-IDF model or quantized DistilBERT to ONNX/TFLite for ultra-low battery consumption directly inside an Android APK.

---

## 13. Disclaimer

> [!IMPORTANT]
> This application is a **hackathon and educational prototype**. Machine learning predictions and risk assessments are probabilistic and should **not** be used as the sole basis for critical financial, legal, or security decisions. Real-world deployment requires continuous training on large-scale, representative fraud datasets.

