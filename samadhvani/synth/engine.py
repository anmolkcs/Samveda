"""The built-in (no-model) synthesis engine.

This is the `engine="formant"` path: it takes a :class:`TimingPlan` plus the phone
lists per note and produces audio.  It is what runs when someone `pip install`s the
package, and it is also the **baseline** the trained model is compared against in
:mod:`samadhvani.eval` — a chant TTS that cannot beat a formant cascade is not
worth the GPU hours.

Two post-processing steps do a lot of the perceived realism:

* a **room** — a small convolution reverb built from a synthesised impulse
  response.  Vedic recitation is recorded in a hall or a temple courtyard, never
  dry; dry chant sounds like a telephone.
* a **chant-grade low-pass and presence peak** — a gentle +2 dB shelf near 3 kHz
  and a roll-off at 7 kHz mimics the ribbon/condenser recordings the tradition is
  documented with.
"""

from __future__ import annotations

import numpy as np

from .formant import FormantSynth
from .prosody import amplitude_envelope
from .timing import TimingPlan

SR = 22050


def _make_ir(sr: int, rt60: float = 1.15, predelay: float = 0.018,
             seed: int = 3) -> np.ndarray:
    """Synthetic impulse response: predelay + exponentially decaying noise.

    Not a real measured IR, but a real IR is a 500 kB file we do not need; a
    decaying-noise tail with a slight high-frequency loss is indistinguishable for
    a sustained single voice.
    """
    rng = np.random.default_rng(seed)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    env = np.exp(-6.9 * t / rt60)                      # -60 dB at rt60
    noise = rng.standard_normal(n) * env
    # one-pole low-pass so the tail darkens as it decays
    from scipy.signal import lfilter
    a = 0.55
    noise = lfilter([1 - a], [1, -a], noise)
    ir = np.zeros(int(predelay * sr) + n)
    ir[int(predelay * sr):] = noise
    # a couple of early reflections
    for delay, gain in ((0.011, 0.42), (0.023, 0.28), (0.037, 0.19), (0.052, 0.12)):
        idx = int(delay * sr)
        if idx < len(ir):
            ir[idx] += gain
    ir /= np.max(np.abs(ir)) or 1.0
    return ir


def _apply_room(x: np.ndarray, sr: int, wet: float = 0.18) -> np.ndarray:
    from scipy.signal import fftconvolve
    ir = _make_ir(sr)
    wet_sig = fftconvolve(x, ir)[: len(x) + len(ir)]
    out = np.zeros(len(wet_sig))
    out[: len(x)] += x * (1.0 - wet)
    out += wet_sig * wet * 1.6
    return out


def _tilm(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Temple-recording colouration: presence lift, gentle top roll-off."""
    from scipy.signal import butter, sosfilt
    # presence
    sos_p = butter(1, [2200 / (sr / 2), 4200 / (sr / 2)], btype="band", output="sos")
    x = x + 0.16 * sosfilt(sos_p, x)
    sos_lp = butter(2, 7300 / (sr / 2), btype="low", output="sos")
    return sosfilt(sos_lp, x)


def synthesize_formant(
    plan: TimingPlan,
    phones_per_note: list[list[str]],
    *,
    sr: int = SR,
    seed: int = 7,
    breath: float = 0.20,
    room: float = 0.18,
) -> tuple[np.ndarray, int]:
    """Render a timing plan to ``(float32 mono audio, sr)``."""
    synth = FormantSynth(sr=sr, seed=seed)
    n_total = int(np.ceil(plan.total_seconds * sr)) + int(0.6 * sr)
    buf = np.zeros(n_total, dtype=np.float64)

    rng = np.random.default_rng(seed)
    for note, phones in zip(plan.notes, phones_per_note):
        s = int(round(note.start * sr))
        n = int(round(note.duration * sr))
        if n <= 0 or s >= n_total:
            continue
        n = min(n, n_total - s)
        # a slightly wandering F0 within the note, independent of the global
        # kampita, keeps repeated notes from phase-locking into a buzz
        ts = np.arange(n) / sr
        drift_cents = 4.0 * np.sin(2 * np.pi * 0.7 * ts + rng.uniform(0, 6.28))
        f0 = note.f0_hz * (2.0 ** (drift_cents / 1200.0))

        seg = synth.render_note(phones, f0, n)
        env = amplitude_envelope(note, sr, attack_ms=28, release_ms=70)
        if len(env) < n:
            env = np.pad(env, (0, n - len(env)), mode="edge")
        buf[s:s + n] += seg * env[:n] * 0.5

    # breaths: a soft inhale in each gap
    for a, b in _gaps(plan):
        g0 = int(a * sr)
        g1 = int(b * sr)
        if g1 - g0 < int(0.06 * sr):
            continue
        gn = min(g1 - g0, int(0.42 * sr))
        from scipy.signal import butter, sosfilt
        sos = butter(2, [380 / (sr / 2), 1500 / (sr / 2)], btype="band", output="sos")
        inhale = sosfilt(sos, rng.standard_normal(gn))
        inhale *= np.hanning(gn) * breath
        buf[g0:g0 + gn] += inhale

    # high-pass at 70 Hz: removes DC drift and sub-audio rumble from the cascade
    from scipy.signal import butter, sosfilt
    x = sosfilt(butter(2, 70 / (sr / 2), btype="high", output="sos"), buf)
    x = _tilm(x, sr)
    x = _apply_room(x, sr, wet=room)
    x = x - np.mean(x)
    peak = np.max(np.abs(x)) or 1.0
    x = np.tanh(x / peak * 1.05) * 0.93          # soft clip, then normalise
    x = x / (np.max(np.abs(x)) or 1.0) * 0.93
    return x.astype(np.float32), sr


def _gaps(plan: TimingPlan):
    out = []
    for a, b in zip(plan.notes, plan.notes[1:]):
        if b.start - a.end > 0.04:
            out.append((a.end, b.start))
    return out
