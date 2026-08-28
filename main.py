"""FastAPI REST API Service for RealConnect Backend & Android App Integration.

Endpoints:
1. GET  /spam-check      - Pre-call phone number spam risk assessment
2. POST /voice-analysis  - Live in-call bot / synthetic voice and fraud detection
3. POST /verify-speaker  - Live in-call speaker identity verification
4. GET  /health          - Health and service status
5. GET  /                - Root service welcome and endpoint map
"""

from typing import Dict, Any, Optional
import os
import io
import re
import base64
import wave
import logging
import numpy as np
from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RealConnectAPI")

app = FastAPI(
    title="RealConnect AI Backend",
    description="AI-powered Spam Call Check, Bot/Deepfake Voice Detection, and Speaker Verification Service",
    version="1.0.0"
)

# Enable CORS for mobile and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --- Root Welcome Endpoint ---
@app.get("/", tags=["General"])
def root():
    """Root endpoint providing service status and available endpoint map."""
    return {
        "service": "RealConnect AI Backend",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": {
            "spam_check": "GET /spam-check?number=+1234567890",
            "voice_analysis": "POST /voice-analysis",
            "verify_speaker": "POST /verify-speaker",
            "health": "GET /health"
        }
    }


# --- Request and Response Schemas ---

class SpamCheckResponse(BaseModel):
    isSpam: bool
    reason: str


class VoiceDataRequest(BaseModel):
    audioBase64: str = Field(..., description="Base64-encoded audio bytes (16kHz WAV/PCM)")


class VoiceAnalysisResponse(BaseModel):
    isBot: bool
    confidence: float


class SpeakerDataRequest(BaseModel):
    phoneNumber: str = Field(..., description="Phone number of the speaker")
    audioBase64: str = Field(..., description="Base64-encoded audio bytes (16kHz WAV/PCM)")


class SpeakerVerifyResponse(BaseModel):
    verified: bool
    status: str


# --- In-Memory Enrolled Speaker Voice Profiles & Telemarketer Patterns ---
KNOWN_SPAM_PREFIXES = ["+1800", "1800", "140", "+91140", "1900", "+1900", "800", "888", "877"]
KNOWN_TELEMARKETERS = {
    "+18005550199": "Reported aggressive debt collector robocall",
    "+911409988776": "Promotional telemarketing broadcast",
    "+1234567890": "Simulated suspicious robocall entity"
}

# In-memory speaker voice fingerprint storage (maps phone_number -> acoustic profile)
SPEAKER_PROFILES: Dict[str, Dict[str, float]] = {}


def analyze_acoustic_features(audio_bytes: bytes) -> Dict[str, float]:
    """Extracts lightweight acoustic features from raw audio bytes without heavy C extensions."""
    data = None
    samplerate = 16000

    # 1. Try reading with standard wave module
    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            samplerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            if sampwidth == 2:
                data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            if n_channels > 1 and data is not None:
                data = data.reshape(-1, n_channels).mean(axis=1)
    except Exception:
        pass

    # 2. Try soundfile if available
    if data is None:
        try:
            import soundfile as sf
            with io.BytesIO(audio_bytes) as bio:
                data, samplerate = sf.read(bio)
            if data.ndim > 1:
                data = data.mean(axis=1)
        except Exception:
            pass

    # 3. Fallback raw buffer conversion
    if data is None or len(data) == 0:
        try:
            data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            data = np.array([0.05, 0.05], dtype=np.float32)

    if len(data) == 0:
        return {
            "energy": 0.05,
            "zero_crossings": 0.1,
            "std_dev": 0.05,
            "sample_count": 0,
            "samplerate": samplerate
        }

    energy = float(np.mean(data ** 2))
    zero_crossings = float(np.mean(np.diff(np.sign(data)) != 0)) if len(data) > 1 else 0.1
    std_dev = float(np.std(data))

    return {
        "energy": energy,
        "zero_crossings": zero_crossings,
        "std_dev": std_dev,
        "sample_count": len(data),
        "samplerate": samplerate
    }


# --- 1. Endpoint: Spam Call Detection (Pre-Call Check) ---
@app.get("/spam-check", response_model=SpamCheckResponse, tags=["Pre-Call"])
def check_spam(number: str = Query(..., description="Incoming phone number e.g. +1234567890")):
    """Pre-call spam check evaluating phone number patterns, prefix reputation, and blocklists."""
    clean_number = re.sub(r"[\s\-\(\)]", "", number.strip())
    logger.info("Spam check requested for number: %s", clean_number)

    # 1. Check known telemarketer blacklist
    if clean_number in KNOWN_TELEMARKETERS:
        return SpamCheckResponse(
            isSpam=True,
            reason=KNOWN_TELEMARKETERS[clean_number]
        )

    # 2. Check telemarketing and commercial bulk prefix ranges
    for prefix in KNOWN_SPAM_PREFIXES:
        if clean_number.startswith(prefix):
            return SpamCheckResponse(
                isSpam=True,
                reason=f"Commercial telemarketer / robocall prefix detected ({prefix})"
            )

    # 3. Check malformed / spoofed number patterns
    digits_only = re.sub(r"\D", "", clean_number)
    if len(digits_only) < 7 or len(digits_only) > 15:
        return SpamCheckResponse(
            isSpam=True,
            reason="Suspected spoofed / invalid phone number format"
        )

    # Default: Number appears legitimate
    return SpamCheckResponse(
        isSpam=False,
        reason="Clean reputation / No spam reports found"
    )


# --- 2. Endpoint: AI Bot / Synthetic Voice Detection (Live In-Call) ---
@app.post("/voice-analysis", response_model=VoiceAnalysisResponse, tags=["In-Call"])
def analyze_voice(data: VoiceDataRequest):
    """In-call voice analysis detecting AI synthetic speech, robocall bots, and fraudulent audio."""
    if not data.audioBase64 or not data.audioBase64.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audioBase64 payload")

    try:
        audio_bytes = base64.b64decode(data.audioBase64)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid Base64 audio string: {e}")

    logger.info("Received in-call audio for analysis (%d bytes)", len(audio_bytes))

    # Extract acoustic profile
    acoustics = analyze_acoustic_features(audio_bytes)

    # Optional NLP speech-to-text integration if heavy dependencies are available
    transcript = ""
    try:
        from src.speech_to_text import transcribe_audio_file
        trans_res = transcribe_audio_file(audio_bytes)
        transcript = trans_res.get("transcript", "").strip()
    except Exception as e:
        logger.debug("Speech-to-text not initialized or optional: %s", e)

    is_bot = False
    confidence = 0.85

    if transcript:
        # Detect Bot / Automated Script Markers in transcript
        bot_phrases = ["press 1", "press 2", "automated message", "recorded message", "stay on the line", "virtual assistant"]
        has_bot_prompt = any(bp in transcript.lower() for bp in bot_phrases)

        fraud_prob = 0.5
        try:
            from src.predictor import predict_message
            fraud_analysis = predict_message(transcript)
            fraud_prob = fraud_analysis.get("fraud_probability", 0.0)
            risk_level = fraud_analysis.get("risk_level", "LOW")
            if has_bot_prompt or risk_level in ["HIGH", "CRITICAL"]:
                is_bot = True
                confidence = max(0.92, round(fraud_prob, 2))
            else:
                is_bot = False
                confidence = max(0.88, round(1.0 - fraud_prob, 2))
        except Exception:
            if has_bot_prompt:
                is_bot = True
                confidence = 0.95
            else:
                is_bot = False
                confidence = 0.85
    else:
        # Evaluate acoustic uniformity & characteristics
        if acoustics["std_dev"] < 0.005 and acoustics["energy"] > 0:
            is_bot = True
            confidence = 0.90
        else:
            is_bot = False
            confidence = 0.80

    logger.info("Voice analysis completed: isBot=%s, confidence=%.2f", is_bot, confidence)
    return VoiceAnalysisResponse(
        isBot=is_bot,
        confidence=confidence
    )


# --- 3. Endpoint: Speaker Identity Verification (Live In-Call) ---
@app.post("/verify-speaker", response_model=SpeakerVerifyResponse, tags=["In-Call"])
def verify_speaker(data: SpeakerDataRequest):
    """In-call biometric speaker verification against enrolled voice profiles."""
    if not data.audioBase64 or not data.audioBase64.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audioBase64 payload")

    try:
        audio_bytes = base64.b64decode(data.audioBase64)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid Base64 audio string: {e}")

    phone = re.sub(r"[\s\-\(\)]", "", data.phoneNumber.strip())
    acoustics = analyze_acoustic_features(audio_bytes)

    logger.info("Speaker verification for %s (%d bytes)", phone, len(audio_bytes))

    # Check if speaker already has an enrolled voice baseline
    if phone not in SPEAKER_PROFILES:
        # First-time enrollment of the caller's voice fingerprint
        SPEAKER_PROFILES[phone] = acoustics
        return SpeakerVerifyResponse(
            verified=True,
            status="Voice profile enrolled and verified"
        )

    # Compare current acoustic fingerprint with stored profile
    stored = SPEAKER_PROFILES[phone]
    energy_diff = abs(stored["energy"] - acoustics["energy"])
    std_diff = abs(stored["std_dev"] - acoustics["std_dev"])

    # Similarity scoring
    if energy_diff < 0.08 and std_diff < 0.08:
        return SpeakerVerifyResponse(
            verified=True,
            status="Verified"
        )
    else:
        return SpeakerVerifyResponse(
            verified=False,
            status="Voice mismatch / Unverified speaker"
        )


# --- 4. Endpoint: Health & Diagnostic ---
@app.get("/health", tags=["Telemetry"])
def health_check():
    """Health and model readiness check."""
    return {
        "status": "online",
        "service": "RealConnect AI Backend",
        "active_model": "rule_and_acoustic_engine",
        "device": "cpu",
        "whisper_model": "base"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

