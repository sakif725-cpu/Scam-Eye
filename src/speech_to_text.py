"""Speech-to-Text Transcription Engine using Faster-Whisper.

Transcribes call audio files (WAV, MP3, M4A, OGG) and microphone recordings locally.
Optimized using CTranslate2 for CPU/GPU execution without cloud API keys or external costs.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import os
import io
import tempfile
import logging
try:
    import torch
except ImportError:
    torch = None

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SpeechTranscriber:
    """Wrapper around Faster-Whisper for local, privacy-preserving audio transcription."""

    def __init__(self, model_size: str = settings.WHISPER_DEFAULT_MODEL_SIZE):
        """Initializes the Faster-Whisper ASR model.

        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small').
        """
        if model_size not in settings.WHISPER_ALLOWED_SIZES:
            logger.warning("Model size '%s' not in allowed list %s. Defaulting to 'base'.",
                           model_size, settings.WHISPER_ALLOWED_SIZES)
            model_size = settings.WHISPER_DEFAULT_MODEL_SIZE

        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # CPU on Mac/Linux uses 'int8' or 'float32' for optimal performance
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = None

    def _ensure_model_loaded(self) -> None:
        """Lazy loader for the Faster-Whisper model to conserve memory."""
        if self.model is None:
            from faster_whisper import WhisperModel
            logger.info("Loading Faster-Whisper model '%s' (device=%s, compute_type=%s)...",
                        self.model_size, self.device, self.compute_type)
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(settings.BASE_DIR / ".cache" / "whisper")
                )
                logger.info("Faster-Whisper model '%s' loaded successfully.", self.model_size)
            except Exception as e:
                logger.warning("Failed to load compute_type '%s'. Retrying with 'float32' on CPU: %s",
                               self.compute_type, e)
                self.device = "cpu"
                self.compute_type = "float32"
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="float32",
                    download_root=str(settings.BASE_DIR / ".cache" / "whisper")
                )

    def transcribe(
        self,
        audio_source: Union[str, Path, bytes, io.BytesIO],
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transcribes audio from a file path, raw bytes, or BytesIO buffer.

        Args:
            audio_source: File path or raw audio binary data.
            language: Optional language code (e.g. 'en', 'hi') or None for auto-detection.

        Returns:
            Dictionary containing:
            - transcript: Full transcribed text string.
            - language: Detected or specified language.
            - language_probability: Confidence of detected language.
            - duration: Total audio duration in seconds.
            - segments: List of timed transcript segments.
        """
        temp_file_created = False
        temp_path = None

        try:
            self._ensure_model_loaded()

            # Handle binary input (BytesIO or raw bytes)
            if isinstance(audio_source, (bytes, io.BytesIO)):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_path = temp_file.name
                temp_file_created = True

                if isinstance(audio_source, io.BytesIO):
                    temp_file.write(audio_source.getvalue())
                else:
                    temp_file.write(audio_source)
                temp_file.close()
                audio_input = temp_path
            elif isinstance(audio_source, (str, Path)):
                audio_input = str(audio_source)
                if not os.path.exists(audio_input):
                    raise FileNotFoundError(f"Audio file not found: {audio_input}")
            else:
                raise TypeError(f"Unsupported audio source type: {type(audio_source)}")

            # Perform transcription
            segments, info = self.model.transcribe(
                audio_input,
                beam_size=5,
                language=language,
                vad_filter=True,  # Voice activity detection to remove silence
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Aggregate transcript segments
            segment_list = []
            full_text_parts = []
            for seg in segments:
                text_clean = seg.text.strip()
                if text_clean:
                    full_text_parts.append(text_clean)
                    segment_list.append({
                        "start": round(seg.start, 2),
                        "end": round(seg.end, 2),
                        "text": text_clean
                    })

            full_transcript = " ".join(full_text_parts).strip()

            return {
                "transcript": full_transcript,
                "language": info.language if info else "unknown",
                "language_probability": round(float(info.language_probability), 4) if info else 1.0,
                "duration": round(float(info.duration), 2) if info else 0.0,
                "segments": segment_list,
                "model_size": self.model_size,
                "device": self.device
            }

        except Exception as err:
            logger.error("Audio transcription failed: %s", err)
            return {
                "transcript": "",
                "error": str(err),
                "language": "unknown",
                "language_probability": 0.0,
                "duration": 0.0,
                "segments": [],
                "model_size": self.model_size,
                "device": self.device
            }

        finally:
            # Privacy & Cleanup: Remove temporary audio files immediately
            if temp_file_created and temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.debug("Failed to remove temp file %s: %s", temp_path, e)


# Global singleton instance
_TRANSCRIBER_INSTANCE: Optional[SpeechTranscriber] = None


def get_transcriber(model_size: str = settings.WHISPER_DEFAULT_MODEL_SIZE) -> SpeechTranscriber:
    """Returns a singleton instance of SpeechTranscriber."""
    global _TRANSCRIBER_INSTANCE
    if _TRANSCRIBER_INSTANCE is None or _TRANSCRIBER_INSTANCE.model_size != model_size:
        _TRANSCRIBER_INSTANCE = SpeechTranscriber(model_size=model_size)
    return _TRANSCRIBER_INSTANCE


def transcribe_audio_file(file_path: Union[str, Path, bytes, io.BytesIO]) -> Dict[str, Any]:
    """Convenience function to transcribe audio using Faster-Whisper."""
    transcriber = get_transcriber()
    return transcriber.transcribe(file_path)

