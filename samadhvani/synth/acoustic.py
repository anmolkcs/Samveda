"""The acoustic model that learns from recordings.

The previous model predicted **six numbers per frame** — F1/F2/F3 and three
amplitudes — copied out of the phone inventory.  It could not learn a singer,
because there was nothing in the target a recording could teach it.  This one
predicts a **40-band log-mel frame**, which is what a recording actually is.

Why this is still small enough for a free GPU
---------------------------------------------
Because the melody is not predicted.  F0 comes from the svara notation via
:func:`samadhvani.audio.mel.f0_track_from_notes`, and it is fed back in as a
conditioning signal.  So the network never has to learn pitch, timing, or the
scale — the three things that force a general TTS model to be large.  What is left
is the *envelope*: which harmonics the singer's throat emphasises, how the vowel
colours the sound.  That is a much smaller function, and 3-4 M parameters is
enough for a few hours of one voice.

Two heads, one encoder:

``mel``   (B, T', 40) log-mel, trained with L1 on masked frames.
``aux``   (B, T') voicing/energy, a cheap auxiliary signal that helps the encoder
          commit to onsets instead of smearing them.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:                                    # pragma: no cover
    torch = None
    nn = None
    F = None
    HAS_TORCH = False

from ..audio.mel import HOP, N_MELS, SR

#: Log-mel frames per second, so predictions can be resynthesised at the right speed.
FRAMES_PER_SECOND = SR / HOP


def phone_features(sym: str, matra: float = 1.0, position: float = 0.0,
                   n_symbols: int = 512) -> np.ndarray:
    """A hashed one-hot for a phone symbol.

    Deliberately not a linguistic feature vector: the phone inventory's articulatory
    labels are useful for the rule engine, but for a learned model a stable, wide
    identity code is stronger and does not bake in a theory.  Hashing keeps the input
    dimension fixed as the inventory grows.
    """
    v = np.zeros(n_symbols, dtype=np.float32)
    h = (hash(sym) & 0x7FFFFFFF) % n_symbols
    v[h] = 1.0
    v[0] = min(matra, 4.0) / 4.0
    v[1] = position
    return v


def mel_condition(svara: int, f0_hz: float, tonic_hz: float, stretch: float,
                  is_cadence: bool = False) -> np.ndarray:
    """Conditioning: the *given* melody, as numbers.

    Seven svara one-hots, log-F0 relative to the tonic, a stretch value (how much
    longer than nominal this note is held) and a cadence flag.  Everything here is
    known before synthesis, which is the whole reason the model can be small.
    """
    v = np.zeros(12, dtype=np.float32)
    v[max(0, min(6, int(svara) - 1))] = 1.0
    v[7] = float(np.log2(max(f0_hz, 1.0) / max(tonic_hz, 1.0)) * 12.0) / 12.0
    v[8] = float(np.clip(stretch, 0.0, 4.0)) / 4.0
    v[9] = 1.0 if is_cadence else 0.0
    v[10] = float(np.log2(max(f0_hz, 1.0) / 100.0)) / 2.0
    v[11] = float(np.clip(tonic_hz / 146.83, 0.5, 1.6)) - 1.0
    return v


PHONE_DIM = 512
COND_DIM = 12


if HAS_TORCH:

    class AcousticModel(nn.Module):
        """Mel-spectrogram predictor: phones + given melody → log-mel.

        Inputs are at the *phone* rate; the decoder upsamples to mel-frame rate, so
        one input row becomes ``upsample`` output frames.  That is why the model can
        be causal-ish and cheap while still producing 86 frames per second.
        """

        def __init__(self, hidden: int = 192, n_layers: int = 4,
                     upsample: int = 4, n_mels: int = N_MELS, dropout: float = 0.1):
            super().__init__()
            self.upsample = upsample
            self.n_mels = n_mels
            self.inp = nn.Linear(PHONE_DIM + COND_DIM, hidden)
            self.conv_in = nn.Conv1d(hidden, hidden, 5, padding=2)
            self.blocks = nn.ModuleList([
                nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2 ** (i % 3))
                for i in range(n_layers)
            ])
            self.norms = nn.ModuleList([nn.GroupNorm(8, hidden) for _ in range(n_layers)])
            self.drop = nn.Dropout(dropout)
            # frame-rate decoder
            self.up = nn.Sequential(
                nn.ConvTranspose1d(hidden, hidden, upsample * 2, stride=upsample,
                                   padding=upsample // 2, output_padding=upsample // 2),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, 5, padding=2),
                nn.GELU(),
            )
            self.mel_head = nn.Conv1d(hidden, n_mels, 1)
            self.aux_head = nn.Conv1d(hidden, 1, 1)

        def forward(self, phones, cond, target_len: int | None = None):
            h = torch.cat([phones, cond], dim=-1)
            h = self.drop(F.gelu(self.inp(h))).transpose(1, 2)      # (B,H,T)
            h = F.gelu(self.conv_in(h))
            for blk, nrm in zip(self.blocks, self.norms):
                h = h + F.gelu(nrm(blk(h)))                          # residual
                h = self.drop(h)
            h = self.up(h)                                           # (B,H,T*up)
            mel = self.mel_head(h).transpose(1, 2)                   # (B,T',n_mels)
            aux = self.aux_head(h).transpose(1, 2)                   # (B,T',1)
            if target_len is not None and mel.shape[1] != target_len:
                mel = F.interpolate(mel.transpose(1, 2), size=target_len,
                                    mode="linear", align_corners=False).transpose(1, 2)
                aux = F.interpolate(aux.transpose(1, 2), size=target_len,
                                    mode="linear", align_corners=False).transpose(1, 2)
            return mel, aux[..., 0]

        def n_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

else:                                                # pragma: no cover
    AcousticModel = object                           # type: ignore


def mel_loss(pred, target, mask, aux_pred=None, aux_target=None) -> "torch.Tensor":
    """Masked L1 on the mel, plus a small auxiliary term.

    Masking matters: without it the padding dominates the loss, the model learns to
    predict silence, and every clip comes out the same length and flat.
    """
    m = mask.unsqueeze(-1)
    l1 = (torch.abs(pred - target) * m).sum() / m.sum().clamp(min=1.0) / pred.shape[-1]
    total = l1
    if aux_pred is not None and aux_target is not None:
        a = (torch.abs(aux_pred - aux_target) * mask).sum() / mask.sum().clamp(min=1.0)
        total = total + 0.2 * a
    return total
