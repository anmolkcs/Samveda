"""Prosody: the F0 contour and the articulation of one note.

Two things make chant sound like chant and not like sung speech:

**Held, stepped pitch.**  A sāman moves in discrete svara steps and then *holds*.
There is no continuous intonation curve of the kind a speech TTS predicts; F0 sits
on a note, glides to the next at syllable boundaries, and stays flat in between.
The glide is where the music lives — a plain step without a transition sounds like
a robot switching frequencies.

**Kampita and meṇḍ.**  The ornamented notes (the ones the tradition marks with a
shake, and the line-final cadence) carry a narrow pitch oscillation of 4–7 Hz at
about a quarter-tone depth.  It is one line of code and it is the single biggest
perceptual difference between this and a generic sine-wave chant.

This module produces a *sample-rate* F0 track plus a per-note voicing and amplitude
envelope, which both the neural vocoder and the built-in formant synthesiser know
how to consume.
"""

from __future__ import annotations

import numpy as np

from .timing import Note, TimingPlan


def _hann(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def f0_track(
    plan: TimingPlan,
    sr: int,
    *,
    glide_ms: float = 45.0,
    kampita_hz: float = 5.2,
    kampita_cents: float = 32.0,
    vibrato_ramp: float = 0.25,
) -> np.ndarray:
    """Sample-accurate F0 in Hz, with legato glides and kampita on held notes.

    ``kampita_cents`` is the peak deviation; 32 cents (~a sixth of a semitone) is
    about right for a restrained Kauthuma shake.  Increase on the cadence.
    """
    n_total = int(np.ceil(plan.total_seconds * sr)) + 1
    f0 = np.zeros(n_total, dtype=np.float64)
    if not plan.notes:
        return f0

    glide_n = max(1, int(glide_ms / 1000.0 * sr))
    glide = _hann(glide_n)

    for i, note in enumerate(plan.notes):
        s = int(round(note.start * sr))
        e = min(int(round(note.end * sr)), n_total)
        if e <= s:
            continue
        seg_len = e - s
        seg = np.full(seg_len, note.f0_hz, dtype=np.float64)

        # Legato glide in from the previous note (only if the previous note ended
        # where this one starts; across a breath we step, we do not slide).
        prev = plan.notes[i - 1] if i > 0 else None
        if prev is not None and abs(prev.end - note.start) < 1e-6 and abs(prev.f0_hz - note.f0_hz) > 1e-3:
            k = min(glide_n, seg_len)
            glide = _hann(k)
            frm = prev.f0_hz
            to = note.f0_hz
            # exponential (log-frequency) interpolation: musically correct and
            # it keeps the cents interval constant across octaves
            ratio = np.exp(np.linspace(np.log(frm), np.log(to), k))
            seg[:k] = ratio * glide[:k] + seg[:k] * (1.0 - glide[:k])
        else:
            # onsets also get a small "touch" from below (a reciter reaches the note)
            k = min(int(glide_n * 1.4), seg_len)
            if k > 1 and note.is_line_start:
                glide = _hann(k)
                ratio = np.exp(np.linspace(np.log(note.f0_hz * 0.985), np.log(note.f0_hz), k))
                seg[:k] = ratio * glide[:k] + seg[:k] * (1.0 - glide[:k])

        # Kampita on long or cadential notes.
        depth = 0.0
        if note.duration > 0.55:
            depth = kampita_cents * (1.35 if note.is_cadence else 1.0)
        if depth > 0:
            ts = np.arange(seg_len) / sr
            env = np.clip(ts / max(vibrato_ramp, 1e-3), 0, 1)
            env *= np.clip((note.duration - ts) / max(vibrato_ramp, 1e-3), 0, 1)
            cents = np.sin(2 * np.pi * kampita_hz * ts + 0.4) * depth * env
            seg *= 2.0 ** (cents / 1200.0)

        f0[s:e] = seg

    return f0


def voicing_mask(plan: TimingPlan, sr: int, *, fade_ms: float = 12.0) -> np.ndarray:
    """1.0 while phonating, 0.0 in the breath gaps, with smooth edges."""
    n_total = int(np.ceil(plan.total_seconds * sr)) + 1
    mask = np.zeros(n_total, dtype=np.float64)
    for note in plan.notes:
        s = int(round(note.start * sr))
        e = min(int(round(note.end * sr)), n_total)
        if e > s:
            mask[s:e] = 1.0
    fade = max(1, int(fade_ms / 1000.0 * sr))
    if fade > 1:
        kernel = _hann(fade * 2)
        kernel /= kernel.sum()
        mask = np.convolve(mask, kernel, mode="same")
    return mask


def breath_positions(plan: TimingPlan) -> list[tuple[float, float]]:
    """Silent spans where a singer would breathe: (start, end) seconds."""
    out: list[tuple[float, float]] = []
    for a, b in zip(plan.notes, plan.notes[1:]):
        if b.start - a.end > 0.05:
            out.append((a.end, b.start))
    return out


def amplitude_envelope(note: Note, sr: int, *, attack_ms: float = 35.0,
                       release_ms: float = 55.0) -> np.ndarray:
    """Per-note loudness: a soft attack, a swell, and a gentle release.

    Chant does not have the consonant-driven amplitude spikes of speech; the
    envelope is closer to an organ.  A small swell in the middle keeps long notes
    from sounding synthetic.
    """
    n = max(1, int(round(note.duration * sr)))
    a = min(int(attack_ms / 1000.0 * sr), n // 2)
    r = min(int(release_ms / 1000.0 * sr), n // 2)
    env = np.ones(n, dtype=np.float64)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a) ** 0.7
    if r > 0:
        env[-r:] = np.linspace(1.0, 0.0, r) ** 0.8
    t = np.linspace(0, 1, n)
    env *= 0.9 + 0.1 * np.sin(np.pi * t)
    if note.is_cadence:
        env *= 0.85 + 0.15 * np.exp(-3 * t)     # settle in the cadence
    return env
