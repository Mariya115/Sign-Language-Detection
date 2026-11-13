"""
Text Conversion and Multi-lingual Text-to-Speech (TTS) Service
Supports English and Kannada audio synthesis via gTTS and pyttsx3.
"""

import os
import io
import time
import base64
from typing import Optional, Dict, Any, List

# Try imports
try:
    import pyttsx3
    HAS_PYTTSX3 = True
except Exception:
    HAS_PYTTSX3 = False

try:
    from gtts import gTTS
    HAS_GTTS = True
except Exception:
    HAS_GTTS = False


class SentenceBuilder:
    """
    Constructs coherent text/sentences from continuous recognized ISL characters or words.
    Distinguishes character finger-spelling vs. word-level signs.
    """

    def __init__(self, char_repeat_threshold: int = 4, word_pause_timeout: float = 2.0):
        self.char_repeat_threshold = char_repeat_threshold
        self.word_pause_timeout = word_pause_timeout
        self.current_word: List[str] = []
        self.words: List[str] = []
        self.history: List[Dict[str, Any]] = []
        self.last_char_seen: Optional[str] = None
        self.last_char_count: int = 0
        self.last_timestamp: float = time.time()

    def add_prediction(self, sign_label: str, confidence: float, is_word: bool = False) -> Dict[str, Any]:
        """
        Incorporate a new stable prediction into the text sequence.
        """
        now = time.time()
        sign_label = sign_label.strip()

        if not sign_label or sign_label.lower() in ["sign not recognized", "no hand detected"]:
            return self.get_state()

        if is_word:
            # Word-level sign: directly append as full word token
            self._commit_current_word()
            self.words.append(sign_label.upper())
            self.history.append({
                "type": "word",
                "value": sign_label.upper(),
                "confidence": round(confidence, 3),
                "timestamp": now,
            })
            self.last_char_seen = None
            self.last_char_count = 0
        else:
            # Character-level sign: require persistent detection across frames
            if sign_label == self.last_char_seen:
                self.last_char_count += 1
            else:
                self.last_char_seen = sign_label
                self.last_char_count = 1

            if self.last_char_count == self.char_repeat_threshold:
                char_to_add = sign_label.upper()
                self.current_word.append(char_to_add)
                self.history.append({
                    "type": "char",
                    "value": char_to_add,
                    "confidence": round(confidence, 3),
                    "timestamp": now,
                })

        self.last_timestamp = now
        return self.get_state()

    def add_space(self) -> Dict[str, Any]:
        """Insert word boundary (space)."""
        self._commit_current_word()
        return self.get_state()

    def backspace(self) -> Dict[str, Any]:
        """Remove last character or word."""
        if self.current_word:
            self.current_word.pop()
        elif self.words:
            self.words.pop()
        return self.get_state()

    def clear(self) -> Dict[str, Any]:
        """Reset entire sentence buffer."""
        self.current_word.clear()
        self.words.clear()
        self.last_char_seen = None
        self.last_char_count = 0
        return self.get_state()

    def _commit_current_word(self):
        if self.current_word:
            self.words.append("".join(self.current_word))
            self.current_word.clear()

    def get_sentence(self) -> str:
        parts = list(self.words)
        if self.current_word:
            parts.append("".join(self.current_word))
        return " ".join(parts)

    def get_state(self) -> Dict[str, Any]:
        return {
            "sentence": self.get_sentence(),
            "words": list(self.words),
            "current_word": "".join(self.current_word),
            "history_count": len(self.history),
        }


class TextToSpeechEngine:
    """
    Multi-engine Text-To-Speech provider.
    Supports English (via offline pyttsx3 or gTTS) and Kannada (via gTTS 'kn').
    """

    def __init__(self):
        self.offline_engine = None
        if HAS_PYTTSX3:
            try:
                self.offline_engine = pyttsx3.init()
                self.offline_engine.setProperty("rate", 160)
            except Exception as e:
                print(f"[TTS] pyttsx3 initialization warning: {e}")

    def synthesize_audio_base64(self, text: str, lang: str = "en") -> Optional[str]:
        """
        Synthesize speech from text and return as base64-encoded MP3/WAV string for frontend playback.
        Supported languages: 'en' (English), 'kn' (Kannada).
        """
        text = text.strip()
        if not text:
            return None

        # 1. Primary: gTTS for high quality multi-lingual (English and Kannada)
        if HAS_GTTS:
            try:
                # Map to gTTS language code
                target_lang = "kn" if lang.lower() in ["kn", "kannada"] else "en"
                tts = gTTS(text=text, lang=target_lang, slow=False)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                audio_b64 = base64.b64encode(fp.read()).decode("utf-8")
                return f"data:audio/mp3;base64,{audio_b64}"
            except Exception as e:
                print(f"[TTS] gTTS error (falling back): {e}")

        # 2. Fallback: offline pyttsx3 for English
        if HAS_PYTTSX3 and self.offline_engine and lang.lower() in ["en", "english"]:
            try:
                temp_wav = "results/temp_tts.wav"
                os.makedirs("results", exist_ok=True)
                self.offline_engine.save_to_file(text, temp_wav)
                self.offline_engine.runAndWait()
                if os.path.isfile(temp_wav):
                    with open(temp_wav, "rb") as f:
                        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                    try:
                        os.remove(temp_wav)
                    except Exception:
                        pass
                    return f"data:audio/wav;base64,{audio_b64}"
            except Exception as e:
                print(f"[TTS] pyttsx3 error: {e}")

        return None
