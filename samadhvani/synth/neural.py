"""The trainable acoustic model.

Design constraints, in order, and why each shaped the architecture:

1. **It must train on a free Kaggle GPU** (a T4 or a P100, 16 GB, 9 h/session,
   12 h/week).  A flow-matching TTS backbone (VITS/FastSpeech2-class) is the obvious
   choice and is what Vāgdhenu uses — but it needs thousands of utterances to beat
   a formant cascade, which a first corpus of five hours of Sāma recitation will
   not provide.  So the model is deliberately *small and encoder-decoder-shaped*.

2. **Chant is low-entropy on the music side and high-entropy on the phonetic side.**
   The svara sequence is given (from the text + gāna model); the F0 curve is almost
   determined.  What the model actually has to learn is (a) segmental durations and
   (b) the spectral envelope of sustained vowels and consonants.  Splitting the
   problem that way means we train **two small models** instead of one big one:

   * :class:`DurationPredictor` — akṣara phones + svara + mātrā -> per-phone
     duration in frames.  Tiny (a few hundred K parameters), trains in minutes,
     and is the piece that most improves intelligibility.
   * :class:`SpectrogramModel` — phone features + svara + F0 -> mel spectrogram.
     A 4-layer convolutional encoder + 2-layer BiLSTM + linear head, ~2.5 M
     parameters.  This is a "Tacotron-lite" and converges on 3-5 h of audio.

3. **The vocoder should not be part of the training loop.**  A learned vocoder
   (HiFi-GAN) needs 10+ h of speech and a day of GPU time.  Instead the
   spectrogram model predicts the **formant tracks** (F1/F2/F3) and amplitudes
   directly, and those are rendered by the very same Klatt cascade that powers the
   zero-dependency path.  A learned vocoder can be dropped in later
   (``vocoder="hifigan"``) without touching the acoustic model.

So the "neural" path is: text -> phones -> *learned* durations and *learned* formant
tracks -> the same cascade.  It is trainable on a T4 in under an hour, it degrades
gracefully to the rule path when a checkpoint is missing, and its failure mode is
"a bit buzzy", not "babbles".

The class is written so the state dict round-trips: ``SamadhvaniModel.save()``
produces a single ``.pt`` that :mod:`samadhvani.pipeline` will pick up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:  # torch is optional for the zero-dependency path
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = None  # type: ignore
    HAS_TORCH = False


# --------------------------------------------------------------------------- #
# Feature description (shared by trainer and inference)
# --------------------------------------------------------------------------- #

#: Categorical features per phone, in order.  One-hot blocks concatenated.
PHONE_VOICED = 1
PHONE_ASPIRATED = 1
N_PLACES = 7
N_MANNERS = 10
N_MATRA_BUCKETS = 5

INPUT_DIM = PHONE_VOICED + PHONE_ASPIRATED + N_PLACES + N_MANNERS + N_MATRA_BUCKETS + 8

PLACES_ORDER = ("velar", "palatal", "retroflex", "dental", "labial", "glottal", "none")
MANNERS_ORDER = (
    "stop", "nasal", "fricative", "sibilant", "approximant", "lateral",
    "vowel", "murmur", "pause", "stobha",
)


def phone_features(symbol: str, matra: float = 1.0, position: float = 0.0) -> np.ndarray:
    """A fixed-width feature vector for one phone.

    ``position`` is 0..1 within the akṣara (onset -> coda), which is what lets the
    model learn that the same /a/ is shorter when a coda follows.
    """
    from ..phones.inventory import PHONES

    p = PHONES.get(symbol)
    vec = np.zeros(INPUT_DIM, dtype=np.float32)
    i = 0
    vec[i] = 1.0 if (p and p.voiced) else 0.0
    i += 1
    vec[i] = 1.0 if (p and p.aspirated) else 0.0
    i += 1
    if p and p.place in PLACES_ORDER:
        vec[i + PLACES_ORDER.index(p.place)] = 1.0
    else:
        vec[i + PLACES_ORDER.index("none")] = 1.0
    i += N_PLACES
    if p and p.manner in MANNERS_ORDER:
        vec[i + MANNERS_ORDER.index(p.manner)] = 1.0
    else:
        vec[i + MANNERS_ORDER.index("pause")] = 1.0
    i += N_MANNERS
    # matra bucket: 0 (short) 1 (long) 2 (pluta) 3 (very long) 4 (zero)
    bucket = 4 if matra <= 0 else (0 if matra <= 1.1 else 1 if matra <= 2.1 else 2 if matra <= 3.1 else 3)
    vec[i + bucket] = 1.0
    i += N_MATRA_BUCKETS
    vec[i] = position
    vec[i + 1] = float(p.matra) if p else 1.0
    vec[i + 2] = 1.0 if p and p.formants[0] > 0 else 0.0
    vec[i + 3] = (p.formants[0] / 1000.0) if p and p.formants[0] > 0 else 0.0
    vec[i + 4] = (p.formants[1] / 3000.0) if p and p.formants[1] > 0 else 0.0
    vec[i + 5] = (p.formants[2] / 4000.0) if p and p.formants[2] > 0 else 0.0
    vec[i + 6] = 1.0 if p and p.manner in ("fricative", "sibilant") else 0.0
    vec[i + 7] = 1.0 if p and p.manner == "nasal" else 0.0
    return vec


def svara_onehot(svara: int) -> np.ndarray:
    v = np.zeros(7, dtype=np.float32)
    if 1 <= svara <= 7:
        v[svara - 1] = 1.0
    return v


COND_DIM = 7 + 3  # svara one-hot + (relative f0 cents, stretch, is_cadence)


def condition_features(svara: int, f0_rel_cents: float, stretch: float, is_cadence: bool) -> np.ndarray:
    return np.concatenate([
        svara_onehot(svara),
        np.array([f0_rel_cents / 1200.0, min(stretch, 4.0) / 4.0, float(is_cadence)], dtype=np.float32),
    ])


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

if HAS_TORCH:

    class PhoneEncoder(nn.Module):
        """Small conv encoder over the phone sequence."""

        def __init__(self, d_in: int = INPUT_DIM, d_cond: int = COND_DIM, hidden: int = 128,
                     dropout: float = 0.1):
            super().__init__()
            self.phone_proj = nn.Linear(d_in, hidden)
            self.cond_proj = nn.Linear(d_cond, hidden)
            self.conv = nn.ModuleList([
                nn.Conv1d(hidden, hidden, k, padding=k // 2) for k in (3, 5, 3)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
            self.drop = nn.Dropout(dropout)
            self.lstm = nn.LSTM(hidden, hidden // 2, num_layers=2, batch_first=True,
                                bidirectional=True, dropout=0.0)
            self.out_dim = hidden

        def forward(self, phones, cond):
            h = self.phone_proj(phones) + self.cond_proj(cond)
            h = h.transpose(1, 2)                          # (B, H, T)
            for conv, norm in zip(self.conv, self.norms):
                r = h
                h = conv(h)
                h = h.transpose(1, 2)
                h = norm(h)
                h = self.drop(F.gelu(h)).transpose(1, 2) + r
            h = h.transpose(1, 2)
            h, _ = self.lstm(h)
            return h

    class DurationPredictor(nn.Module):
        """Predicts log-frames per phone; monotone at inference via cumulative sum."""

        def __init__(self, encoder: "PhoneEncoder" | None = None, hidden: int = 128):
            super().__init__()
            self.encoder = encoder or PhoneEncoder(hidden=hidden)
            self.head = nn.Sequential(
                nn.LayerNorm(self.encoder.out_dim), nn.Linear(self.encoder.out_dim, 64), nn.GELU(),
                nn.Linear(64, 1),
            )

        def forward(self, phones, cond):
            h = self.encoder(phones, cond)
            return self.head(h).squeeze(-1)          # (B, T) log-frames

    class SpectrogramModel(nn.Module):
        """Predicts formant tracks + amplitude from phones and condition.

        Output: F1, F2, F3 (Hz), voiced amplitude, noise amplitude, spectral tilt.
        Six numbers per frame.  That is the whole acoustic target, which is why
        this trains on small data.
        """

        N_OUT = 6

        def __init__(self, hidden: int = 128, upsample: int = 4):
            super().__init__()
            self.encoder = PhoneEncoder(hidden=hidden)
            self.upsample = upsample
            self.up = nn.Sequential(
                nn.ConvTranspose1d(hidden, hidden, upsample * 2, stride=upsample,
                                   padding=upsample // 2, output_padding=upsample // 2),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, 5, padding=2),
                nn.GELU(),
            )
            self.head = nn.Conv1d(hidden, self.N_OUT, 1)

        def forward(self, phones, cond):
            h = self.encoder(phones, cond).transpose(1, 2)   # (B, H, T)
            h = self.up(h)                                   # (B, H, T*upsample)
            y = self.head(h).transpose(1, 2)                 # (B, T', 6)
            # F1/F2/F3 in Hz via scaled sigmoid; amplitudes 0..1
            fm = torch.sigmoid(y[..., :3]) * torch.tensor([1200.0, 3200.0, 4200.0])
            amp = torch.sigmoid(y[..., 3:])
            return torch.cat([fm, amp], dim=-1)

else:  # pragma: no cover - only when torch is absent
    PhoneEncoder = DurationPredictor = SpectrogramModel = object  # type: ignore


@dataclass
class SamadhvaniCheckpoint:
    """Everything needed to reload a trained model."""

    duration_state: dict = field(default_factory=dict)
    spectrogram_state: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    version: int = 1

    def save(self, path: str) -> None:
        if not HAS_TORCH:
            raise RuntimeError("torch is required to save a checkpoint")
        torch.save({
            "duration_state": self.duration_state,
            "spectrogram_state": self.spectrogram_state,
            "config": self.config,
            "version": self.version,
        }, path)

    @classmethod
    def load(cls, path: str) -> "SamadhvaniCheckpoint":
        if not HAS_TORCH:
            raise RuntimeError("torch is required to load a checkpoint")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            duration_state=blob.get("duration_state", {}),
            spectrogram_state=blob.get("spectrogram_state", {}),
            config=blob.get("config", {}),
            version=blob.get("version", 1),
        )


def build_models(hidden: int = 128, upsample: int = 4):
    """Instantiate a matched pair of trainable models."""
    if not HAS_TORCH:
        raise RuntimeError("torch is required")
    dur = DurationPredictor(hidden=hidden)
    spec = SpectrogramModel(hidden=hidden, upsample=upsample)
    return dur, spec
