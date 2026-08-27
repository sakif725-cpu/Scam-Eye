"""Unit tests for FastAPI REST API endpoints matching RealConnect Android contract."""

import base64
import numpy as np
import io
import soundfile as sf
import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def generate_dummy_wav_base64() -> str:
    """Generates a short synthetic 16kHz sine wave audio in Base64 encoding."""
    samplerate = 16000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    # 440 Hz standard tone
    audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, samplerate, format="WAV")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "RealConnect" in data["service"]


def test_spam_check_clean_number():
    res = client.get("/spam-check?number=+14155552671")
    assert res.status_code == 200
    data = res.json()
    assert "isSpam" in data
    assert "reason" in data
    assert isinstance(data["isSpam"], bool)
    assert isinstance(data["reason"], str)


def test_spam_check_known_telemarketer():
    res = client.get("/spam-check?number=+18005550199")
    assert res.status_code == 200
    data = res.json()
    assert data["isSpam"] is True
    assert "robocall" in data["reason"].lower() or "telemarketer" in data["reason"].lower()


def test_voice_analysis_endpoint():
    b64_audio = generate_dummy_wav_base64()
    payload = {"audioBase64": b64_audio}

    res = client.post("/voice-analysis", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "isBot" in data
    assert "confidence" in data
    assert isinstance(data["isBot"], bool)
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0


def test_verify_speaker_endpoint():
    b64_audio = generate_dummy_wav_base64()
    payload = {
        "phoneNumber": "+1234567890",
        "audioBase64": b64_audio
    }

    # 1. Enrollment & First Verification
    res = client.post("/verify-speaker", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "verified" in data
    assert "status" in data
    assert isinstance(data["verified"], bool)
    assert isinstance(data["status"], str)

    # 2. Subsequent Verification with matching audio
    res2 = client.post("/verify-speaker", json=payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["verified"] is True
    assert data2["status"] == "Verified"
