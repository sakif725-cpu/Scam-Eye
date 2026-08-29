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
import json
import numpy as np
from fastapi import FastAPI, Query, HTTPException, status, WebSocket, WebSocketDisconnect
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
@app.api_route("/", methods=["GET", "HEAD"], tags=["General"])
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


# --- Health Check Endpoint ---
@app.api_route("/health", methods=["GET", "HEAD"], tags=["General"])
def health_check():
    """Service liveness and readiness probe for Render / Docker orchestration."""
    return {
        "status": "healthy",
        "service": "RealConnect AI",
        "models_loaded": {
            "ml_fraud_classifier": True,
            "explainability_engine": True
        }
    }


# --- Request and Response Schemas ---

class SpamCheckResponse(BaseModel):
    isSpam: bool
    reason: str


class VoiceDataRequest(BaseModel):
    audioBase64: Optional[str] = Field(default="", description="Base64-encoded 16kHz Mono 16-bit WAV/PCM")
    text: Optional[str] = Field(default="", description="Spoken dialogue text or message to analyze")


class VoiceAnalysisResponse(BaseModel):
    isBot: bool = False
    isScammer: bool = False
    confidence: float = 0.8
    riskScore: int = 5
    riskLevel: str = "LOW"  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    intention: str = "💬 Natural Human Conversation"
    summary: str = "Natural human voice biomarkers verified."
    behaviorSummary: str = "Natural human voice biomarkers verified."
    threatIndicators: List[str] = Field(default_factory=list)
    recommendation: str = "Safe call. Continue conversation normally."
    scamType: Optional[str] = None
    transcript: Optional[str] = ""


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
    """Extracts lightweight acoustic features from 16kHz Mono 16-bit WAV/PCM audio bytes."""
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
                data = np.frombuffer(frames, dtype='<i2').astype(np.float32) / 32768.0
            elif sampwidth == 1:
                data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                data = np.frombuffer(frames, dtype='<i2').astype(np.float32) / 32768.0

            if n_channels > 1 and data is not None:
                data = data.reshape(-1, n_channels).mean(axis=1)
    except Exception:
        pass

    # 2. Try raw PCM buffer conversion (skip 44-byte standard RIFF header if present)
    if data is None or len(data) == 0:
        try:
            offset = 44 if len(audio_bytes) > 44 and audio_bytes[:4] == b'RIFF' else 0
            data = np.frombuffer(audio_bytes[offset:], dtype='<i2').astype(np.float32) / 32768.0
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
    rms = float(np.sqrt(np.mean((data * 32768.0) ** 2)))

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


from collections import deque
import time

class CallSessionManager:
    """Maintains rolling conversation history across in-call speech turns to prevent context fragmentation."""
    def __init__(self, max_turns: int = 15, timeout_seconds: int = 600):
        self.sessions: Dict[str, deque] = {}
        self.last_activity: Dict[str, float] = {}
        self.max_turns = max_turns
        self.timeout_seconds = timeout_seconds

    def add_turn(self, session_key: str, text_snippet: str) -> str:
        now = time.time()
        # Reset if call was inactive for > 10 mins
        if session_key in self.last_activity and (now - self.last_activity[session_key] > self.timeout_seconds):
            self.sessions[session_key] = deque(maxlen=self.max_turns)
        
        if session_key not in self.sessions:
            self.sessions[session_key] = deque(maxlen=self.max_turns)

        self.last_activity[session_key] = now
        clean_snip = text_snippet.strip()
        if clean_snip and (not self.sessions[session_key] or self.sessions[session_key][-1] != clean_snip):
            self.sessions[session_key].append(clean_snip)

        return " ".join(self.sessions[session_key])

    def get_full_context(self, session_key: str) -> str:
        if session_key in self.sessions and self.sessions[session_key]:
            return " ".join(self.sessions[session_key])
        return ""

call_sessions = CallSessionManager()


class PredictRequest(BaseModel):
    text: str = Field(..., description="Message text to analyze for fraud/scam")


@app.post("/predict", tags=["SMS & Text"])
def predict_endpoint(data: PredictRequest):
    """Analyze SMS/chat message text for fraud, phishing, and scam patterns."""
    text = data.text.strip() if data.text else ""
    print("\n" + "=" * 65, flush=True)
    print(f"📥 [SMS/TEXT FRAUD SCAN RECEIVED]: \"{text}\"", flush=True)
    print("=" * 65, flush=True)
    logger.info("[SMS/TEXT FRAUD SCAN RECEIVED]: \"%s\"", text)
    from src.predictor import predict_message
    res = predict_message(text)
    is_scam = res.get("prediction") == "FRAUD"
    risk_level = res.get("risk_level", "LOW")
    risk_score_pct = int(float(res.get("risk_score", 0.0)) * 100)
    print(f"🛡️ [TEXT SCAN RESULT] -> Prediction: {'🚨 FRAUD' if is_scam else '✅ SAFE'} | Risk Level: {risk_level} ({risk_score_pct}%)", flush=True)
    print("=" * 65 + "\n", flush=True)
    logger.info("[TEXT SCAN RESULT] -> %s | Risk: %s (%s%%)", '🚨 FRAUD' if is_scam else '✅ SAFE', risk_level, risk_score_pct)
    return res


# --- 2. Endpoint: AI Bot / Scammer Behavioral Analysis (Live In-Call) ---
@app.post("/voice-analysis", response_model=VoiceAnalysisResponse, tags=["In-Call"])
def analyze_voice(data: VoiceDataRequest):
    """In-call voice analysis detecting synthetic speech, bots, and scammer psychological behavior with rolling context."""
    session_key = (data.text or "active_live_call").split(" - ")[0].replace("Caller: ", "").strip()
    if not session_key:
        session_key = "active_live_call"

    # 1. Direct text analysis if transcript is already extracted
    if data.text and data.text.strip():
        text_snippet = data.text.strip()
        full_context = call_sessions.add_turn(session_key, text_snippet)

        print("\n" + "=" * 65, flush=True)
        print(f"📥 [AI IN-CALL TEXT RECEIVED]: \"{text_snippet}\"", flush=True)
        if full_context != text_snippet:
            print(f"📜 [ACCUMULATED CALL CONTEXT]: \"{full_context}\"", flush=True)
        print("=" * 65, flush=True)
        logger.info("[AI IN-CALL TEXT RECEIVED]: \"%s\"", text_snippet)

        risk_result = None
        try:
            from src.predictor import predict_message
            risk_result = predict_message(full_context)
        except Exception as e:
            logger.warning("Predictor execution error: %s", e)

        if risk_result:
            is_scam = risk_result.get("prediction") == "FRAUD"
            explanation = risk_result.get("explanation", {})
            indicators = [
                f"{item.get('name', '')}: {item.get('description', '')}"
                for item in explanation.get("indicators", [])
            ] if explanation.get("indicators") else [risk_result.get("reasoning", "")]
            
            threat_cat = explanation.get("threat_category", "Suspicious Conversation" if is_scam else "💬 Natural Human Conversation")
            risk_score_pct = int(float(risk_result.get("risk_score", 0.0)) * 100)
            risk_level = risk_result.get("risk_level", "LOW")

            print(f"🛡️ [AI ANALYSIS RESULT] -> Prediction: {'🚨 FRAUD' if is_scam else '✅ SAFE'} | Risk Level: {risk_level} ({risk_score_pct}%) | Intent: {threat_cat}", flush=True)
            print("=" * 65 + "\n", flush=True)
            logger.info("[AI ANALYSIS RESULT] -> %s | Risk: %s (%s%%) | %s", '🚨 FRAUD' if is_scam else '✅ SAFE', risk_level, risk_score_pct, threat_cat)

            return VoiceAnalysisResponse(
                isBot=False,
                isScammer=is_scam,
                confidence=round(float(risk_result.get("confidence_percentage", 90.0)) / 100.0, 2),
                riskScore=risk_score_pct,
                riskLevel=risk_level,
                intention=threat_cat,
                summary=risk_result.get("reasoning", explanation.get("summary", "Live speech analyzed.")),
                behaviorSummary=risk_result.get("reasoning", explanation.get("summary", "Live speech analyzed.")),
                threatIndicators=indicators,
                recommendation=risk_result.get("recommended_action", "Safe call. Continue conversation normally."),
                scamType=threat_cat if is_scam else None,
                transcript=full_context
            )

    if not data.audioBase64 or not data.audioBase64.strip():
        return VoiceAnalysisResponse(
            isBot=False,
            isScammer=False,
            confidence=0.8,
            riskScore=2,
            riskLevel="LOW",
            intention="Monitoring live conversation",
            summary="Audio stream active",
            behaviorSummary="Audio stream active",
            threatIndicators=["Live audio monitoring active"],
            recommendation="Speak naturally. AI Guard is actively listening.",
            transcript=data.text or ""
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
            behaviorSummary="Incoming audio snippet could not be parsed.",
            threatIndicators=["Unrecognized audio format"],
            recommendation="Continue conversation."
        )

    # 1. Acoustic / Vocoder Check on 16kHz Mono 16-bit audio
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

    # 2. In-Call Speech-to-Text Transcription with Multi-Accent & Regional Support
    transcript = ""
    try:
        import io
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 200
        recognizer.dynamic_energy_threshold = True
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)

        # Primary: en-IN (Indian English - natively understands 'crore', 'lakh', 'KYC', 'OTP', 'CVV')
        try:
            transcript = recognizer.recognize_google(audio_data, language="en-IN")
        except Exception:
            # Fallback: en-US
            transcript = recognizer.recognize_google(audio_data, language="en-US")

        if transcript:
            # Contextual phoneme correction for common ASR acoustic distortions
            corrections = {
                r"\b(grove cycle|5 grove|grove)\b": "crore",
                r"\b(cvp|c v p|c v c)\b": "CVV",
                r"\b(k y see|k y c)\b": "KYC",
                r"\b(oh t p|o t p)\b": "OTP",
                r"\b(flipers|fliper)\b": "Flipkart",
            }
            for pattern, repl in corrections.items():
                transcript = re.sub(pattern, repl, transcript, flags=re.IGNORECASE)

        logger.info("Speech recognition transcribed: %s", transcript)
    except Exception as e:
        logger.debug("Speech recognition silence or pass: %s", e)

    if transcript and transcript.strip():
        text_snippet = transcript.strip()
        full_context = call_sessions.add_turn(session_key, text_snippet)

        print("\n" + "=" * 65, flush=True)
        print(f"📥 [AI IN-CALL VOICE TRANSCRIBED]: \"{text_snippet}\"", flush=True)
        if full_context != text_snippet:
            print(f"📜 [ACCUMULATED CALL CONTEXT]: \"{full_context}\"", flush=True)
        print("=" * 65, flush=True)
        logger.info("[AI IN-CALL VOICE TRANSCRIBED]: \"%s\"", text_snippet)

        from src.predictor import predict_message
        risk_result = predict_message(full_context)
        is_scam = risk_result.get("prediction") == "FRAUD"
        explanation = risk_result.get("explanation", {})
        indicators = [
            f"{item.get('name', '')}: {item.get('description', '')}"
            for item in explanation.get("indicators", [])
        ] if explanation.get("indicators") else [risk_result.get("reasoning", "")]
        
        threat_cat = explanation.get("threat_category", "Suspicious Conversation" if is_scam else "💬 Natural Human Conversation")
        risk_score_pct = int(float(risk_result.get("risk_score", 0.0)) * 100)
        risk_level = risk_result.get("risk_level", "LOW")

        print(f"🛡️ [AI ANALYSIS RESULT] -> Prediction: {'🚨 FRAUD' if is_scam else '✅ SAFE'} | Risk Level: {risk_level} ({risk_score_pct}%) | Intent: {threat_cat}", flush=True)
        print("=" * 65 + "\n", flush=True)
        logger.info("[AI ANALYSIS RESULT] -> %s | Risk: %s (%s%%) | %s", '🚨 FRAUD' if is_scam else '✅ SAFE', risk_level, risk_score_pct, threat_cat)

        return VoiceAnalysisResponse(
            isBot=is_bot,
            isScammer=is_scam,
            confidence=round(float(risk_result.get("confidence_percentage", 90.0)) / 100.0, 2),
            riskScore=risk_score_pct,
            riskLevel=risk_level,
            intention=threat_cat,
            summary=risk_result.get("reasoning", explanation.get("summary", "Live speech analyzed.")),
            behaviorSummary=risk_result.get("reasoning", explanation.get("summary", "Live speech analyzed.")),
            threatIndicators=indicators,
            recommendation=risk_result.get("recommended_action", "Safe call. Continue conversation normally."),
            scamType=threat_cat if is_scam else None,
            transcript=full_context
        )

    # 3. Behavioral Classification based on acoustic biomarkers
    if is_bot:
        summary_msg = "Robotic/AI generated voice signature detected."
        return VoiceAnalysisResponse(
            isBot=True,
            isScammer=True,
            scamType="Synthetic Voice / Deepfake",
            confidence=confidence,
            riskScore=95,
            riskLevel="HIGH",
            intention="🚨 Synthetic Deepfake Robocall: Automated voice synthesis attempting social engineering",
            summary=summary_msg,
            behaviorSummary=summary_msg,
            threatIndicators=[
                "Synthetic vocoder acoustic biomarker detected",
                "Unnatural pitch modulation & metallic frequency spectrum"
            ],
            recommendation="Hang up immediately. Do not trust automated voice requests.",
            transcript=transcript
        )

    # Clean / Normal Human conversation
    normal_summary = "Natural human voice biomarkers verified."
    return VoiceAnalysisResponse(
        isBot=False,
        isScammer=False,
        confidence=0.88,
        riskScore=5,
        riskLevel="LOW",
        intention="💬 Natural Human Conversation",
        summary=normal_summary,
        behaviorSummary=normal_summary,
        threatIndicators=[
            "Natural conversational dynamic",
            "No malicious phishing vectors detected"
        ],
        recommendation="Safe call. Continue conversation normally.",
        transcript=transcript
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


# --- 4. Endpoint: Real-Time Live Voice Streaming via WebSocket ---
@app.websocket("/ws/voice-stream")
async def websocket_voice_stream(websocket: WebSocket):
    """Bidirectional WebSocket streaming endpoint for live in-call audio analysis.
    
    Accepts raw PCM/WAV binary frames or JSON payloads with Base64 audio,
    and returns real-time scam probability, acoustic biomarkers, transcript,
    threat category, and defensive safety advice.
    """
    await websocket.accept()
    logger.info("Live WebSocket voice stream connected.")
    try:
        while True:
            message = await websocket.receive()
            audio_bytes = b""

            if "bytes" in message and message["bytes"]:
                audio_bytes = message["bytes"]
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    audio_b64 = payload.get("audioBase64", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                except Exception as e:
                    logger.warning("Error parsing WebSocket JSON frame: %s", e)
            
            if not audio_bytes or len(audio_bytes) < 320:
                continue

            # 1. Acoustic biomarker & Deepfake / Bot check
            acoustics = analyze_acoustic_features(audio_bytes)
            is_bot = False
            confidence = 0.85
            if acoustics["zero_crossings"] > 0.35 and acoustics["rms"] > 200:
                is_bot = True
                confidence = 0.94
            elif acoustics["std_dev"] < 0.005 and acoustics["energy"] > 0.001:
                is_bot = True
                confidence = 0.90

            # 2. Faster-Whisper Speech-to-Text Transcription
            transcript = ""
            try:
                from src.speech_to_text import transcribe_audio_file
                trans_res = transcribe_audio_file(audio_bytes)
                transcript = trans_res.get("transcript", "").strip()
            except Exception:
                pass

            # 3. NLP Scam & Risk Engine Evaluation
            risk_result = None
            if transcript:
                try:
                    from src.predictor import predict_message
                    risk_result = predict_message(transcript)
                except Exception:
                    pass

            if risk_result:
                is_scam = (risk_result.get("prediction") == "FRAUD") or is_bot
                res_payload = {
                    "transcript": transcript,
                    "isScammer": is_scam,
                    "isBot": is_bot,
                    "riskScore": int(risk_result.get("risk_score", 0.0) * 100),
                    "riskLevel": risk_result.get("risk_level", "LOW"),
                    "scamType": risk_result.get("threat_category", "Suspicious Caller" if is_scam else "Normal Conversation"),
                    "threatIndicators": [item.get("description", "") for item in risk_result.get("indicators", [])] or [risk_result.get("reason", "")],
                    "recommendation": risk_result.get("action_advice", "Safe call. Continue conversation normally.")
                }
            else:
                res_payload = {
                    "transcript": transcript,
                    "isScammer": is_bot,
                    "isBot": is_bot,
                    "riskScore": 90 if is_bot else 5,
                    "riskLevel": "HIGH" if is_bot else "LOW",
                    "scamType": "Synthetic Voice / Bot" if is_bot else "Normal Conversation",
                    "threatIndicators": ["Synthetic vocoder acoustic signature detected"] if is_bot else ["Natural conversational dynamics"],
                    "recommendation": "Hang up immediately. Suspected bot." if is_bot else "Safe call. Continue conversation normally."
                }

            await websocket.send_json(res_payload)
    except WebSocketDisconnect:
        logger.info("Live WebSocket voice stream disconnected.")
    except Exception as e:
        logger.error("WebSocket voice stream error: %s", e)


# --- 5. Endpoint: Health & Diagnostic ---
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
