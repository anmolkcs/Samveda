"""Inference with a trained checkpoint.

The trained pair of models slots into the *same* source-filter cascade the
zero-dependency path uses.  What changes is where the numbers come from:

======================  ===========================  ==============================
                        rule path                    neural path
======================  ===========================  ==============================
phone durations          mātrā x laya                 DurationPredictor
formant targets          the phone inventory table     SpectrogramModel
voice/noise levels       fixed per manner class        SpectrogramModel
F0                       svara + kampita (deterministic)  unchanged
======================  ===========================  ==============================

That last row is deliberate.  Nobody wants a model to invent the melody: the svara
sequence is *given* by the tradition, and the F0 track is the one thing we can
compute exactly.  Letting a network predict pitch would only introduce drift.

So the neural path is genuinely "learn the phonetics, keep the music".  It also
means the checkpoint is small, the failure mode is graceful (a buzzy vowel, not
babble), and you can hear exactly what the model learned by A/B-ing the two paths
on the same verse — which is what the demo's engine switch is for.
"""

from __future__ import annotations

import numpy as np

from .prosody import amplitude_envelope
from .timing import TimingPlan

SR = 22050


def render_neural(plan: TimingPlan, phone_lists: list[list[str]],
                  checkpoint: str, *, seed: int = 7) -> tuple[np.ndarray, int]:
    """Render with trained formant/duration models."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for the neural engine") from exc

    from ..train.trainer import phone_features, condition_features
    from ..synth.neural import COND_DIM, INPUT_DIM, SamadhvaniCheckpoint, SpectrogramModel, DurationPredictor

    ck = SamadhvaniCheckpoint.load(checkpoint)
    cfg = ck.config or {}
    hidden = int(cfg.get("hidden", 128))
    up = int(cfg.get("upsample", 4))

    dur_model = DurationPredictor(hidden=hidden)
    if ck.duration_state:
        dur_model.load_state_dict(ck.duration_state)
    dur_model.eval()

    spec_model = SpectrogramModel(hidden=hidden, upsample=up)
    if ck.spectrogram_state:
        spec_model.load_state_dict(ck.spectrogram_state)
    spec_model.eval()

    # Build the feature sequence, one entry per phone, with its note's condition.
    xs, cs = [], []
    for note, phones in zip(plan.notes, phone_lists):
        cents = 1200.0 * np.log2(max(note.f0_hz, 1.0) / 146.83)
        cond = condition_features(int(note.svara), float(cents),
                                   note.duration / 0.4, bool(note.is_cadence))
        for j, sym in enumerate(phones or ["a"]):
            posn = (j + 0.5) / max(len(phones or ["a"]), 1)
            xs.append(phone_features(sym, note.duration, posn))
            cs.append(cond)
    if not xs:
        from .engine import synthesize_formant
        return synthesize_formant(plan, phone_lists, seed=seed)

    x = torch.from_numpy(np.asarray(xs, dtype=np.float32)).unsqueeze(0)
    c = torch.from_numpy(np.asarray(cs, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        log_frames = dur_model(x, c).squeeze(0).numpy()      # per phone, log frames
        spec = spec_model(x, c).squeeze(0).numpy()           # (T*up, 6) in Hz / 0..1

    # durations -> seconds
    pred_frames = np.exp(np.clip(log_frames, np.log(1.0), np.log(400.0)))
    pred_seconds = pred_frames / float(cfg.get("frames_per_second", 100.0))

    # The decoder emits upsample frames per phone; average back to phone rate.
    n_phones = len(xs)
    per_phone = []
    for i in range(n_phones):
        lo = i * up
        hi = min((i + 1) * up, len(spec))
        per_phone.append(spec[lo:hi].mean(0) if hi > lo else spec[min(lo, len(spec) - 1)])
    per_phone = np.asarray(per_phone)

    # Render each note using the learned formants, keeping the plan's own timing so
    # the melody is untouched (predicted durations are reported but not enforced
    # when they disagree wildly with the score, which happens on out-of-domain text).
    from .formant import BANDWIDTHS, FormantSynth, _biquad_lowpass, glottal_source, resonator
    from scipy.signal import lfilter

    synth = FormantSynth(sr=SR, seed=seed)
    n_total = int(np.ceil(plan.total_seconds * SR)) + int(0.5 * SR)
    buf = np.zeros(n_total)
    rng = np.random.default_rng(seed)

    k = 0
    for note, phones in zip(plan.notes, phone_lists):
        s = int(round(note.start * SR))
        n = int(round(note.duration * SR))
        if n <= 0 or s >= n_total:
            k += max(1, len(phones or [1]))
            continue
        n = min(n, n_total - s)

        # average the learned targets over this note's phones
        block = per_phone[k:k + max(1, len(phones or [1]))]
        k += max(1, len(phones or [1]))
        if len(block) == 0:
            block = per_phone[-1:]
        f1, f2, f3 = block[:, :3].mean(0)
        voiced, noisy, tilt = block[:, 3:6].mean(0)

        ts = np.arange(n) / SR
        drift = 4.0 * np.sin(2 * np.pi * 0.7 * ts + rng.uniform(0, 6.28))
        f0 = note.f0_hz * (2.0 ** (drift / 1200.0))

        # glottal source + the learned formant triple through the same cascade
        src = glottal_source(f0, SR, rng=rng)
        src = np.diff(src, prepend=src[0] if len(src) else 0.0)
        bb, aa = _biquad_lowpass(4200.0 + 1800.0 * float(tilt), SR)
        src = lfilter(bb, aa, src)
        out = src
        for f, bw in zip((f1, f2, f3), BANDWIDTHS):
            if f and f > 50:
                b2, a2 = resonator(float(f), bw, SR)
                out = lfilter(b2, a2, out)
        r = float(np.sqrt(np.mean(out * out))) or 1.0
        out = out * ((0.30 * float(np.clip(voiced, 0.05, 1.0))) / r)

        if noisy > 0.25 and not (phones and phones[0] in ("h", "H", "kh_h", "ph_h")):
            # a learned noisy phone: band-limited noise scaled by the predicted level
            nz = rng.standard_normal(n)
            bn, an = _biquad_lowpass(5200.0, SR)
            nz = lfilter(bn, an, nz)
            nz /= (np.sqrt(np.mean(nz * nz)) or 1.0)
            out = out + nz * 0.12 * float(noisy)

        env = amplitude_envelope(note, SR, attack_ms=28, release_ms=70)
        if len(env) < n:
            env = np.pad(env, (0, n - len(env)), mode="edge")
        buf[s:s + n] += out[:n] * env[:n] * 0.5

    from .engine import _apply_room, _tilm
    from scipy.signal import butter, sosfilt
    x = sosfilt(butter(2, 70 / (SR / 2), btype="high", output="sos"), buf)
    x = _apply_room(_tilm(x, SR), SR, wet=0.18)
    peak = np.max(np.abs(x)) or 1.0
    x = np.tanh(x / peak * 1.05)
    x = x / (np.max(np.abs(x)) or 1.0) * 0.93
    return x.astype(np.float32), SR
