"""Audio output helpers — WAV writing with no third-party dependency.

We write 16-bit PCM WAV ourselves so the package has zero required dependencies on
the audio side.  ``soundfile`` is used when it is present (it is what the Kaggle
notebook installs anyway), but nothing here requires it.
"""

from __future__ import annotations

import io
import struct
import wave

import numpy as np


def to_pcm16(x: np.ndarray) -> bytes:
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


def wav_bytes(x: np.ndarray, sr: int) -> bytes:
    """Return a complete RIFF/WAVE file as bytes."""
    x = np.asarray(x)
    if x.ndim == 2:                       # (n, channels) -> interleaved
        channels = x.shape[1]
        pcm = np.clip(x, -1, 1).reshape(-1)
    else:
        channels = 1
        pcm = np.clip(x, -1, 1)
    data = to_pcm16(pcm)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(data)
    return buf.getvalue()


def write_wav(path: str, x: np.ndarray, sr: int) -> None:
    with open(path, "wb") as fh:
        fh.write(wav_bytes(x, sr))


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch)
    return x, sr


def duration(x: np.ndarray, sr: int) -> float:
    return len(x) / float(sr)
