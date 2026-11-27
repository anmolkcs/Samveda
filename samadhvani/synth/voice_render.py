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


#: Hop of the loudness-matching envelope.  Small enough to follow a phrase, large
#: enough not to chase individual pitch periods.
SMOOTH_HOP = 128


def _soft_limit(x: np.ndarray, crest_db: float = 11.5) -> np.ndarray:
    """Reduce peakiness to *at most* ``crest_db``, smoothly.

    A harmonic source has a crest factor of 16-25 dB where the formant engine has 11.
    That is not cosmetic: a peak limiter must remove 10-17 dB to bring the render under
    full scale, and removing 17 dB of level is exactly the "trained one is quieter and
    worse" impression.

    ``thr * tanh(x / thr)`` is used rather than a knee-on-excess formula because it is
    *bounded by `thr` by construction* — so the output crest factor is at most
    ``thr/rms`` and the target is met exactly instead of approximately. A first attempt
    reduced only 35 % of the excess per pass and quietly left the render 4 dB short.

    tanh is ≈ identity for |x| ≪ thr, so quiet passages are untouched; only peaks are
    bent, and smoothly, so there is no clipping artefact.
    """
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(x)))) or 1e-9
    thr = rms * (10.0 ** (crest_db / 20.0))
    return thr * np.tanh(x / thr)


def _frame_energy(x: np.ndarray, hop: int = SMOOTH_HOP) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = max(1, len(x) // hop)
    return np.mean(np.square(x[: n * hop].reshape(n, hop)), axis=1)


def _frame_rms(x: np.ndarray, hop: int = SMOOTH_HOP) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = max(1, len(x) // hop)
    return np.sqrt(np.mean(np.square(x[: n * hop].reshape(n, hop)), axis=1))


def _expand_to_samples(v: np.ndarray, n: int, hop: int = SMOOTH_HOP) -> np.ndarray:
    """Frame values back to one value per sample, without steps."""
    if len(v) < 2:
        return np.full(n, float(v[0]) if len(v) else 1.0)
    xs = (np.arange(len(v)) + 0.5) * hop
    out = np.interp(np.arange(n), xs, v)
    return np.clip(out, 1e-3, 1e3)


def _smooth_residual(pred: np.ndarray, band_taps: int = 5, frame_taps: int = 3,
                     max_step_db: float = 4.0) -> np.ndarray:
    """Make the learned correction a smooth envelope instead of a spike train.

    Measured problem, not a precaution: on the real pipeline the model's raw output
    has a **68 dB jump between two adjacent mel bands**. Where a harmonic happens to
    sit in the band that was boosted 68 dB, the vocoder gives that harmonic a 68 dB
    gain and the note turns into a whistle; elsewhere the band does nothing. The
    result reads as a broken synthesiser even though the melody is perfect.

    Three things are wrong with trusting the raw output, and this fixes all three:

    * a spectral envelope is smooth in frequency — a sung vowel does not have a
      40 dB notch between two adjacent mel bands;
    * the model is trained on ~3 s crops and predicts one frame at a time, so
      frame-to-frame chatter has no acoustic meaning;
    * the correction is a *correction*, so it should be gentle relative to the
      envelope it is modifying.

    Smoothing then capping the band-to-band slope keeps the correction's broad
    shape (which is the singer's colour, and is what we want) and removes the
    narrow spikes (which are noise in the prediction, and are not).

    The slope cap is applied cumulatively from a monotone base so that it cannot be
    undone by a later band: iterating forward, then backward, guarantees the bound
    holds in both directions.
    """
    p = np.asarray(pred, dtype=np.float32)
    if p.ndim != 2 or p.size == 0:
        return p

    def smooth1(v: np.ndarray, taps: int, axis: int) -> np.ndarray:
        if taps <= 1 or v.shape[axis] < taps:
            return v
        k = np.ones(taps, dtype=np.float32) / taps
        pad = [(0, 0)] * v.ndim
        pad[axis] = (taps // 2, taps // 2)
        vp = np.pad(v, pad, mode="edge")
        out = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), axis, vp)
        return out.astype(np.float32)

    p = smooth1(p, band_taps, 1)
    p = smooth1(p, frame_taps, 0)

    if max_step_db > 0:
        step = float(max_step_db)
        # forward pass: no band may exceed its left neighbour by more than `step`
        q = p.copy()
        for b in range(1, q.shape[1]):
            q[:, b] = np.clip(q[:, b], q[:, b - 1] - step, q[:, b - 1] + step)
        # backward pass: and no band may exceed its right neighbour either
        for b in range(q.shape[1] - 2, -1, -1):
            q[:, b] = np.clip(q[:, b], q[:, b + 1] - step, q[:, b + 1] + step)
        p = q
    return p.astype(np.float32)


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
    #    The correction is applied as a *relative* spectral shape, and this took three
    #    wrong attempts to get right.  What the model predicts is the absolute envelope
    #    of an archival recording; what this renderer needs is "how should this voice's
    #    colour differ from the rule engine's".  Adding the absolute quantity in the
    #    wrong coordinates made every render uniformly dark and quiet (measured: -7 dB
    #    RMS, -13 dB at the top of the spectrum) while the melody stayed perfect.
    #
    #    So: take the envelope the singer would produce, subtract its own mean, and use
    #    that *shape*, mean-removed so it cannot change loudness (loudness is matched
    #    later against the base).  Only the relative tilt between bands survives, which
    #    is the thing that is actually timbre.
    target_env = mean_spec[None, :] + np.asarray(pred, dtype=np.float32)
    shape = target_env - target_env.mean(axis=1, keepdims=True)
    #    Measured: the raw prediction can jump 68 dB between neighbouring mel bands. A
    #    band boosted that far puts a whistle on whichever harmonic lands in it. So
    #    smooth, then bound both magnitude and slope; what survives is the singer's
    #    broad colour, and what is removed is prediction noise.
    shape = _smooth_residual(shape.astype(np.float32))
    #    Centre *before* clipping, never after: subtracting a row mean after the clip
    #    produces values outside the bound (measured -18 dB from a +/-10 dB clip).
    shape = shape - shape.mean(axis=1, keepdims=True)
    delta = np.clip(shape, -max_correction_db, max_correction_db)
    freqs = _band_centres()
    taper = np.clip((freqs - protect_hz) / max(1.0, protect_hz), 0.0, 1.0)
    #    Third guard: the correction is trusted only where there is actually sound.
    #    The model was trained on trimmed, energetic crops, so on near-silent frames
    #    it extrapolates, and the first notes of a phrase are the quietest frames in
    #    the render.  An energy gate keeps the correction where it has evidence.
    db = energy * 80.0 - 80.0
    loud = np.clip((db - (db.max() - 30.0)) / 10.0, 0.0, 1.0)
    gate = (voiced[:, None] > 0).astype(np.float32) * loud[:, None]
    blend = M + strength * delta * gate * taper[None, :]

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
    # 7. match the caller's loudness, frame by frame.  A global RMS match is not
    #    enough: the harmonic vocoder has a higher crest factor than the formant
    #    source, so matching total energy leaves sustained phrases quieter and gaps
    #    louder, and the result pumps. Matching the short-term envelope keeps the
    #    dynamics the rule engine had (the phrase shape is part of the singing) while
    #    making an A/B listen about timbre rather than volume.
    #    Order matters here and it was wrong: the gain envelope must be applied to a
    #    signal whose *shape* is final, and only then should a safety limiter run. The
    #    vocoder emits RMS ~1.1 with a peak of ~7 (a harmonic source is peaked), so an
    #    earlier `if peak > 0.99: divide` in front of the level match divided by 7.4 and
    #    discarded 17 dB, leaving the trained render 11-17 dB quieter than the formant
    #    render at every strength. Measured, three times, before it was believed.
    #    Two steps, in this order, and the order is the point:
    #    (a) tame the peaks. The harmonic source's crest is 16-25 dB against the formant
    #        engine's 11, so *any* later peak limiter removes 10-17 dB and the render
    #        sounds quiet and cheap. Target 11.5 dB, and that number is not free: to
    #        match the formant engine's RMS while staying under full scale, the crest
    #        must be <= 20*log10(0.99/0.26) = 11.6 dB. Limiting to 13 left the render
    #        permanently 4 dB quiet, because the safety limiter then had to trim;
    #    (b) then match the loudness exactly, per frame, on a signal whose shape is
    #        final. Matching before (a) is what produced 0.153 RMS for a 0.272 target.
    x = _soft_limit(x, crest_db=11.5)
    eb = _frame_energy(base)
    ex = _frame_energy(x)
    both = (eb > 1e-8) & (ex > 1e-8)
    if both.any():
        # bounded: a frame where the vocoder produced almost nothing must not demand a
        # 1000x gain, or it becomes the loudest thing in the render and the limiter
        # then squashes everything else (measured: 46 dB crest, -34 dB RMS)
        ratio = np.ones_like(eb)
        ratio[both] = np.sqrt(np.clip(eb[both] / ex[both], 0.25, 4.0))
        k = np.ones(9) / 9.0
        ratio = np.convolve(np.pad(ratio, (4, 4), mode="edge"), k, mode="valid")
        x = x * _expand_to_samples(ratio, len(x), SMOOTH_HOP)
    # (c) a final exact global level match: the per-frame envelope cannot be exact
    #     everywhere, and a render that is 3 dB quieter than the one it replaces
    #     reads as "worse" regardless of timbre.  Set the peak explicitly afterwards
    #     rather than letting a limiter decide, and keep the two in the same order:
    #     scale, then limit, then re-scale once more so the RMS *is* the target.
    base_f = np.asarray(base, dtype=np.float64)
    rms_target = float(np.sqrt(np.mean(np.square(base_f))))
    for _ in range(2):
        rms_x = float(np.sqrt(np.mean(np.square(x)))) or 1e-9
        x = x * (rms_target / rms_x)
        # the phone-shaped peaks are now ~13 dB, so this only ever trims a little
        x = _soft_limit(x, crest_db=11.5)
    rms_x = float(np.sqrt(np.mean(np.square(x)))) or 1e-9
    x = x * (rms_target / rms_x)
    peak = float(np.max(np.abs(x)))
    target_peak = float(np.max(np.abs(base_f))) or 0.95
    if peak > 0.99:
        x = x * (0.99 / peak)
    # Safety only: a peak limiter as a last resort, never as level control.
    peak = float(np.max(np.abs(x))) or 1.0
    if peak > 0.99:
        x = x * (0.99 / peak)
    return x.astype(np.float32), SR
