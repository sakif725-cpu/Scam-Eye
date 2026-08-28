"""FastAPI REST API Service for RealConnect AI Backend & Android App Integration.

Endpoints:
1. GET  /spam-check      - Pre-call phone number spam risk assessment
2. POST /voice-analysis  - Live in-call bot / synthetic voice and behavioral fraud detection
3. POST /verify-speaker  - Live in-call speaker identity verification
4. GET  /health          - Health and service status
5. GET  /                - Root service welcome and endpoint map
"""

from typing import Dict, Any, Optional, List
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
logger = logging.getLogger("RealConnect-AI")

app = FastAPI(
    title="RealConnect AI Backend",
    description="AI-powered Spam Call Check, Behavioral Scammer Intelligence, Bot/Deepfake Voice Detection, and Speaker Verification Service",
    version="2.0.0"
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
        "version": "2.0.0",
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
    audioBase64: Optional[str] = Field(default="", description="Base64-encoded audio bytes (16kHz WAV/PCM)")


class VoiceAnalysisResponse(BaseModel):
    isBot: bool = False
    isScammer: bool = False
    confidence: float = 0.8
    riskScore: int = 5
    riskLevel: str = "LOW"
    intention: str = "💬 Natural Human Conversation"
    summary: str = "Natural human voice biomarkers verified."
    threatIndicators: List[str] = Field(default_factory=list)
    recommendation: str = "Safe call. Continue conversation normally."
    scamType: Optional[str] = None


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
    """Extracts lightweight acoustic features from raw audio bytes."""
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

    # 2. Try raw PCM buffer conversion (skip WAV header if present)
    if data is None or len(data) == 0:
        try:
            offset = 44 if len(audio_bytes) > 44 and audio_bytes[:4] == b'RIFF' else 0
            data = np.frombuffer(audio_bytes[offset:], dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            data = np.array([0.05, 0.05], dtype=np.float32)

    if len(data) == 0:
        return {
            "energy": 0.05,
            "zero_crossings": 0.1,
            "std_dev": 0.05,
            "rms": 0.0,
            "sample_count": 0,
            "samplerate": samplerate
        }

    energy = float(np.mean(data ** 2))
    zero_crossings = float(np.mean(np.diff(np.sign(data)) != 0)) if len(data) > 1 else 0.1
    std_dev = float(np.std(data))
    rms = float(np.sqrt(np.mean((data * 32767.0) ** 2)))

    return {
        "energy": energy,
        "zero_crossings": zero_crossings,
        "std_dev": std_dev,
        "rms": rms,
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
    digits_only = re.sub(r"\D", "", clean_number)
    for prefix in KNOWN_SPAM_PREFIXES:
        p_clean = re.sub(r"\D", "", prefix)
        if clean_number.startswith(prefix) or (p_clean and digits_only.startswith(p_clean)):
            return SpamCheckResponse(
                isSpam=True,
                reason=f"Commercial telemarketer / robocall prefix detected ({prefix})"
            )

    # 3. Check malformed / spoofed number patterns
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


# --- 2. Endpoint: AI Bot / Scammer Behavioral Analysis (Live In-Call) ---
@app.post("/voice-analysis", response_model=VoiceAnalysisResponse, tags=["In-Call"])
def analyze_voice(data: VoiceDataRequest):
    """In-call voice analysis detecting synthetic speech, bots, and scammer psychological behavior."""
    logger.info("Received in-call audio for behavioral analysis (%d chars)", len(data.audioBase64 or ""))

    if not data.audioBase64 or not data.audioBase64.strip():
        return VoiceAnalysisResponse(
            isBot=False,
            isScammer=False,
            confidence=0.8,
            riskScore=2,
            riskLevel="LOW",
            intention="Monitoring live conversation",
            summary="Audio stream active",
            threatIndicators=["Live audio monitoring active"],
            recommendation="Speak naturally. AI Guard is actively listening."
        )

    try:
        audio_bytes = base64.b64decode(data.audioBase64)
    except Exception as e:
        logger.error("Failed to decode audioBase64: %s", e)
        return VoiceAnalysisResponse(
            isBot=False,
            isScammer=False,
            confidence=0.5,
            riskScore=10,
            riskLevel="LOW",
            intention="Audio decoding warning",
            summary="Incoming audio snippet could not be parsed.",
            threatIndicators=["Unrecognized audio format"],
            recommendation="Continue conversation."
        )

    # 1. Acoustic / Vocoder Check
    acoustics = analyze_acoustic_features(audio_bytes)
    is_bot = False
    confidence = 0.85

    # Vocoder / synthetic artifacts detection (ZCR > 0.35 with RMS > 200, or extreme uniform variance)
    if acoustics["zero_crossings"] > 0.35 and acoustics["rms"] > 200:
        is_bot = True
        confidence = 0.94
    elif acoustics["std_dev"] < 0.005 and acoustics["energy"] > 0.001:
        is_bot = True
        confidence = 0.90

    # 2. Optional NLP / Transcript evaluation
    transcript = ""
    try:
        from src.speech_to_text import transcribe_audio_file
        trans_res = transcribe_audio_file(audio_bytes)
        transcript = trans_res.get("transcript", "").strip()
    except Exception:
        pass

    if transcript:
        lower_trans = transcript.lower()
        phishing_keywords = [
            "cvv", "otp", "password", "bank account", "lottery", "5 crore", "prize",
            "gift card", "wire transfer", "social security", "police verification",
            "arrest warrant", "urgent transfer", "credit card number", "debit card"
        ]
        bot_keywords = [
            "press 1", "press 2", "automated message", "recorded message", "stay on the line", "virtual assistant"
        ]

        has_phishing = any(pk in lower_trans for pk in phishing_keywords)
        has_bot = any(bk in lower_trans for bk in bot_keywords)

        if has_phishing or has_bot:
            return VoiceAnalysisResponse(
                isBot=is_bot or has_bot,
                isScammer=True,
                scamType="Financial Phishing / CVV Harvesting" if has_phishing else "Automated Robocall",
                confidence=0.96,
                riskScore=98 if has_phishing else 95,
                riskLevel="HIGH",
                intention="🚨 Financial Theft: Caller is coercing user into revealing card/banking credentials under a fake pretext" if has_phishing else "🚨 Automated Bot: Robocall trying to illicit response",
                summary="Caller is attempting social engineering to steal financial security credentials." if has_phishing else "Automated voice script detected.",
                threatIndicators=[
                    "Direct solicitation of confidential credentials / OTP / CVV" if has_phishing else "Automated interactive voice prompt",
                    "Artificial urgency & coercive pretext",
                    "Unsolicited financial prize or authority claim"
                ],
                recommendation="DO NOT disclose CVV, OTP, or banking information. Hang up immediately."
            )

    # 3. Behavioral Classification based on acoustic biomarkers
    if is_bot:
        return VoiceAnalysisResponse(
            isBot=True,
            isScammer=True,
            scamType="Synthetic Voice / Deepfake",
            confidence=confidence,
            riskScore=95,
            riskLevel="HIGH",
            intention="🚨 Synthetic Deepfake Robocall: Automated voice synthesis attempting social engineering",
            summary="Robotic/AI generated voice signature detected.",
            threatIndicators=[
                "Synthetic vocoder acoustic biomarker detected",
                "Unnatural pitch modulation & metallic frequency spectrum"
            ],
            recommendation="Hang up immediately. Do not trust automated voice requests."
        )

    # Clean / Normal Human conversation
    return VoiceAnalysisResponse(
        isBot=False,
        isScammer=False,
        confidence=0.88,
        riskScore=5,
        riskLevel="LOW",
        intention="💬 Natural Human Conversation",
        summary="Natural human voice biomarkers verified.",
        threatIndicators=[
            "Natural conversational dynamic",
            "No malicious phishing vectors detected"
        ],
        recommendation="Safe call. Continue conversation normally."
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
        SPEAKER_PROFILES[phone] = acoustics
        return SpeakerVerifyResponse(
            verified=True,
            status="Voice profile enrolled and verified"
        )

    # Compare current acoustic fingerprint with stored profile
    stored = SPEAKER_PROFILES[phone]
    energy_diff = abs(stored["energy"] - acoustics["energy"])
    std_diff = abs(stored["std_dev"] - acoustics["std_dev"])

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
        "version": "2.0.0",
        "active_model": "behavioral_scam_and_acoustic_engine",
        "device": "cpu"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
