"""Render with the trained voice: rule-engine melody, learned timbre.

The rule engine already produces a correct note plan and correct timing — that part
was never the problem, and it comes from the notation.  What it cannot do is sound
like a particular singer, because its spectral envelope is a formant table from a
textbook.

So this renderer keeps the plan and replaces the envelope:

1. render the plan with the formant engine (fast, correct length, correct pitch);
2. analyse it to log-mel;
3. ask the trained model what this singer's envelope would be, given the *same* F0,
   voicing and energy;
4. add that correction, and resynthesise with harmonic excitation so the pitch
   survives.

The result is not a neural vocoder.  It is a trained spectral correction on top of a
deterministic chant, and that is what the caveat in the demo says.  Being explicit
about that is the difference between a tool a reciter will trust and one they will
not.
"""

from __future__ import annotations

import numpy as np

from ..audio.mel import HOP, SR, _mel_centres, harmonic_vocode, logmel


def _band_centres() -> np.ndarray:
    """Centre frequency of each mel band, for frequency-dependent weighting."""
    return _mel_centres()


def _align_onset(x: np.ndarray, ref: np.ndarray, hop: int = HOP,
                 sr: int = SR) -> np.ndarray:
    """Shift ``x`` so its first voiced sample lines up with ``ref``'s.

    Measured, not assumed: if the offset is not there, this is a no-op.
    """
    def first_voiced(y: np.ndarray) -> int:
        e = np.abs(np.asarray(y, dtype=np.float32))
        if not len(e):
            return 0
        thr = e.max() * 0.02
        idx = np.flatnonzero(e > thr)
        return int(idx[0]) if len(idx) else 0

    lag = first_voiced(x) - first_voiced(ref)
    if lag <= 0:
        return np.asarray(x, dtype=np.float32)
    # Advance by ``lag`` and pad the tail by the same amount: the content now starts
    # where the plan says it does, *and* the total length is unchanged, so the note
    # timings reported alongside the audio stay true.
    out = x[lag:]
    out = np.pad(out, (0, min(lag, len(x))))
    return out[:len(x)].astype(np.float32)


def _condition_track(base: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (f0, voiced) for the *rule-engine* render.

    Measured from the rendered audio, not from the plan, and that distinction is
    load-bearing: :func:`synthesize_formant` inserts breath gaps between phrases, so
    the audio is longer than the sum of the note durations.  Driving the vocoder from
    plan timestamps therefore misaligns pitch against the mel by whatever those gaps
    add, which lands harmonics on the wrong frames and pulls the decoded melody an
    octave off.  Measuring F0 from the same signal being corrected cannot drift.
    """
    from ..train.voice_trainer import f0_from_audio
    f0 = f0_from_audio(base, SR, HOP)[:n_frames]
    if len(f0) < n_frames:
        f0 = np.pad(f0, (0, n_frames - len(f0)), mode="edge")
    return f0.astype(np.float32), (f0 > 0).astype(np.float32)


def render_voice(plan, phone_lists, checkpoint: str, *, base_audio=None,
                 seed: int = 7, room: float = 0.18, strength: float = 0.5,
                 max_correction_db: float = 10.0,
                 protect_hz: float = 420.0) -> tuple[np.ndarray, int]:
    """Apply the trained voice colour to an already-rendered chant.

    ``base_audio`` is the rule-engine render that the caller produced.  It is passed
    in rather than re-synthesised here on purpose: re-rendering with even slightly
    different options changes the length, and then the note timings that describe the
    audio are simply wrong.  One render, one timeline.
    """
    import torch

    from .engine import synthesize_formant
    from ..train.voice_trainer import _F0_CENTRE, load_voice

    model, mean_spec, cfg = load_voice(checkpoint)

    # 1. the deterministic chant, which fixes length, onset and pitch
    base = (np.asarray(base_audio, dtype=np.float32) if base_audio is not None
            else synthesize_formant(plan, phone_lists, seed=seed, breath=0.2, room=0.0)[0])
    if not len(base):
        return base, SR
    M = logmel(base, SR)
    n_frames = M.shape[0]
    f0, voiced = _condition_track(base, n_frames)

    # 2. what this singer does to those frames
    energy = (M.mean(axis=1) + 80.0) / 80.0
    feat = np.stack([
        np.where(f0 > 0, np.log2(np.maximum(f0, 1.0)) - _F0_CENTRE, 0.0),
        voiced,
        energy,
    ], axis=1).astype(np.float32)

    with torch.no_grad():
        pred = model(torch.from_numpy(feat)[None]).numpy()[0]

    # 3. the learned envelope, bounded.
    #    A correction must never be able to make the render worse than the rule
    #    engine it is correcting: on a small or mismatched corpus the residual is
    #    noise, and an unbounded add turns a correct chant into a wrong one.  So the
    #    residual is clamped and scaled, and unvoiced frames keep their own energy
    #    (otherwise sibilants vanish into the singer's vowel colour).
    #    The low bands are where F0 and the first formants live.  A 40-band mel has
    #    ~1 band between a 147 Hz fundamental and its second harmonic, so letting the
    #    correction move energy there can flip the perceived octave — which is exactly
    #    the failure the melody metric catches.  So the correction is tapered to zero
    #    below ``protect_hz`` and the fundamental region is left to the rule engine.
    pred = np.clip(np.asarray(pred, dtype=np.float32), -max_correction_db,
                   max_correction_db)
    freqs = _band_centres()
    taper = np.clip((freqs - protect_hz) / max(1.0, protect_hz), 0.0, 1.0)
    #    Third guard: the correction is trusted only where there is actually sound.
    #    The model was trained on trimmed, energetic crops, so on near-silent frames
    #    it extrapolates, and the first notes of a phrase are the quietest frames in
    #    the render.  An energy gate keeps the correction where it has evidence.
    db = energy * 80.0 - 80.0
    loud = np.clip((db - (db.max() - 30.0)) / 10.0, 0.0, 1.0)
    gate = (voiced[:, None] > 0).astype(np.float32) * loud[:, None]
    blend = M + strength * pred * gate * taper[None, :]

    # 4. resynthesise with the measured F0 so the melody cannot drift
    x = harmonic_vocode(blend, f0, SR)

    # 5. restore the onset.  Half-window analysis and the harmonic phase reset put the
    #    first voiced sample a few frames late, and every note boundary inherits the
    #    lag, which the melody metric sees as notes landing late.  Drop the dead air in
    #    front and match the base envelope's head so phrase timing stays honest.
    x = _align_onset(x, base)
    # 5b. The caller has *already* applied the room, and it is applied to the base
    #     signal before we analyse it.  Applying it again here double-wets the audio
    #     and, worse, the reverb tail smears each note onset into its neighbour, which
    #     is what makes a melody tracker read the wrong svara on short notes.  The
    #     ``room`` argument is kept for the standalone-render path only.
    if room > 0 and base_audio is None:
        try:
            from .engine import _apply_room
            x = _apply_room(np.asarray(x, dtype=np.float64), SR, wet=room).astype(np.float32)
        except Exception:
            pass
    # 6. same length as the base, so the reported note timings remain true.
    #    (``_apply_room`` pads its convolution tail; trimming it is better than
    #    letting the audio and the metadata disagree.)
    if len(x) > len(base):
        x = x[:len(base)]
    elif len(x) < len(base):
        x = np.pad(x, (0, len(base) - len(x)))
    peak = float(np.max(np.abs(x))) or 1.0
    return (x / peak * 0.89).astype(np.float32), SR
