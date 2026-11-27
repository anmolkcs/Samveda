"""Audio front end: decode anything, analyse to log-mel, resynthesise from log-mel.

This is the module that was missing.  Until it existed, the trainer had nothing to
learn from a recording: it only ever saw formant targets copied out of the phone
inventory.  With it, a WAV on disk becomes a target tensor, and a predicted tensor
becomes a WAV.

Everything here is **numpy/scipy only** — no librosa, no soundfile requirement for
the maths.  ``soundfile`` is used only to *decode* (it handles MP3/OGG/FLAC and,
importantly, byte-range seeking inside MP3), and degrades to the stdlib ``wave``
module for plain WAV.

Two deliberate choices:

* **log-mel, 40 bands**, 22.05 kHz, 256-sample hop (86.1 fps).  Small enough to
  train on a free GPU in an afternoon, large enough to carry vowel identity.
* **Griffin-Lim resynthesis**, because it needs no trained vocoder.  It is not as
  clean as a neural vocoder: expect a slight metallic edge on sustained notes.
  Phase reconstruction is the honest cost of not having a vocoder checkpoint, and
  it is stated in the demo rather than hidden.
"""

from __future__ import annotations

import math

import numpy as np

SR = 22050
N_FFT = 1024
HOP = 256
N_MELS = 40
FMIN = 60.0
FMAX = 7600.0


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #

def read_audio(path: str, sr: int = SR) -> np.ndarray:
    """Decode any audio file to mono float32 at ``sr``.

    Uses ``soundfile`` when available (MP3/OGG/FLAC/WAV), else the stdlib ``wave``
    reader for PCM WAV.  Resampling is linear interpolation, which is adequate for
    mel features and avoids a scipy build dependency.
    """
    try:
        import soundfile as sf
        x, file_sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:
        from ..synth.io import read_wav
        x, file_sr = read_wav(path)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float32)
    if file_sr != sr and len(x):
        n = int(round(len(x) * sr / float(file_sr)))
        x = np.interp(np.linspace(0.0, len(x) - 1, n, dtype=np.float64),
                      np.arange(len(x), dtype=np.float64), x).astype(np.float32)
    return x


def read_audio_window(path: str, start_s: float, end_s: float | None = None,
                      sr: int = SR, pad_s: float = 1.0) -> np.ndarray:
    """Decode just one time window of a file.

    For MP3 this seeks rather than decoding the whole file, which is the difference
    between a two-minute and a four-hour job on a 44-minute clip.  ``pad_s`` is
    taken from both sides so that a timestamp that is off by a second does not clip
    the first syllable; the caller trims silence afterwards.
    """
    start = max(0.0, start_s - pad_s)
    dur = None if end_s is None else max(0.05, (end_s - start_s) + 2 * pad_s)
    try:
        import soundfile as sf
        with sf.SoundFile(path) as f:
            f.seek(int(start * f.samplerate))
            n = -1 if dur is None else int(dur * f.samplerate)
            x = f.read(n, dtype="float32", always_2d=True).mean(axis=1)
            file_sr = f.samplerate
    except Exception:
        x = read_audio(path, sr=sr)[int(start * sr):]
        file_sr = sr
    x = np.asarray(x, dtype=np.float32)
    if file_sr != sr and len(x):
        n = int(round(len(x) * sr / float(file_sr)))
        x = np.interp(np.linspace(0.0, len(x) - 1, n, dtype=np.float64),
                      np.arange(len(x), dtype=np.float64), x).astype(np.float32)
    return x


# --------------------------------------------------------------------------- #
# Silence / breath-group detection
# --------------------------------------------------------------------------- #

def frame_energy(x: np.ndarray, hop: int = HOP) -> np.ndarray:
    n = max(1, len(x) // hop)
    e = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = x[i * hop:(i + 1) * hop]
        e[i] = float(np.sqrt(np.mean(seg * seg))) if len(seg) else 0.0
    return e


def trim_silence(x: np.ndarray, sr: int = SR, hop: int = HOP,
                 rel: float = 0.06, min_ms: float = 60.0) -> np.ndarray:
    """Trim leading/trailing low-energy frames.

    ``rel`` is relative to the clip's own loud percentile rather than an absolute
    dBFS threshold, because archive.org rips of one performance can differ by 12 dB
    between volumes and an absolute gate silently discards whole clips.
    """
    if not len(x):
        return x
    e = frame_energy(x, hop)
    if not len(e):
        return x
    thr = max(np.percentile(e, 95) * rel, 1e-5)
    idx = np.flatnonzero(e >= thr)
    if not len(idx):
        return x
    pad = int(0.05 * sr)
    a = max(0, idx[0] * hop - pad)
    b = min(len(x), (idx[-1] + 1) * hop + pad)
    out = x[a:b]
    if len(out) < min_ms / 1000.0 * sr:      # too short: keep the original
        return x
    return out


def breath_groups(x: np.ndarray, sr: int = SR, hop: int = HOP,
                  rel: float = 0.12, min_gap_ms: float = 180.0,
                  min_group_ms: float = 350.0) -> list[tuple[float, float]]:
    """Split audio into sung phrases, separated by the reciter's breaths.

    Sāma chanting is delivered in breath groups, and the gāna text marks them with
    daṇḍas.  Detecting them from energy is not as good as having the timestamps, but
    it means a clip whose transcription has no timings still yields utterances.
    """
    if not len(x):
        return []
    e = frame_energy(x, hop)
    if not len(e):
        return []
    voiced = e >= max(np.percentile(e, 95) * rel, 1e-5)
    min_gap = max(1, int(min_gap_ms / 1000.0 * sr / hop))
    min_len = max(1, int(min_group_ms / 1000.0 * sr / hop))

    groups: list[tuple[float, float]] = []
    i, n = 0, len(voiced)
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        gap = 0
        last = i
        while j < n:
            if voiced[j]:
                last = j
                gap = 0
            else:
                gap += 1
                if gap >= min_gap:
                    break
            j += 1
        if last - i + 1 >= min_len:
            groups.append((i * hop / sr, (last + 1) * hop / sr))
        i = j
    return groups


# --------------------------------------------------------------------------- #
# log-mel analysis
# --------------------------------------------------------------------------- #

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int = SR, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """Triangular mel filters as a (n_mels, n_fft//2+1) matrix."""
    n_bins = n_fft // 2 + 1
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(min(fmax, sr / 2 - 100)),
                          n_mels + 2)
    hz = _mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(1, n_mels + 1):
        lo, ce, hi = bins[m - 1], bins[m], bins[m + 1]
        if ce == lo:
            ce = min(lo + 1, n_bins - 1)
        if hi <= ce:
            hi = min(ce + 1, n_bins - 1)
        for k in range(lo, ce):
            fb[m - 1, k] = (k - lo) / max(1, ce - lo)
        for k in range(ce, hi):
            fb[m - 1, k] = (hi - k) / max(1, hi - ce)
        # Area-normalise.  Without this, a band's reading depends on how many FFT
        # bins it spans, so a *flat* input spectrum does not produce a flat mel
        # spectrogram: wide high-frequency triangles collect more energy than narrow
        # low ones, and the tilt is invisible until you compare two signals whose
        # levels differ.  It then shows up as a constant offset — measured at 2.6 dB
        # between the corpus and the renderer — which a model happily memorises as if
        # it were the singer's timbre.
        tot = fb[m - 1].sum()
        if tot > 0:
            fb[m - 1] /= tot
    return fb


_FB_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _filterbank_cached(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """`mel_filterbank` memoised — it is identical for every call in a process."""
    key = (sr, n_fft, n_mels)
    fb = _FB_CACHE.get(key)
    if fb is None:
        fb = mel_filterbank(sr, n_fft, n_mels)
        _FB_CACHE[key] = fb
    return fb


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def _frames_view(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    return np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, n_fft),
        strides=(x.strides[0] * hop, x.strides[0]), writeable=False)


def stft(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Complex STFT, (frames, n_fft//2+1)."""
    frames = _frames_view(np.asarray(x, dtype=np.float64), n_fft, hop)
    return np.fft.rfft(frames * _hann(n_fft), axis=1)


def stft_mag(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    return np.abs(stft(x, n_fft, hop)).astype(np.float32)


def istft(spec: np.ndarray, n_fft: int = N_FFT, hop: int = HOP,
          length: int | None = None) -> np.ndarray:
    """Weighted overlap-add inverse STFT (Hann analysis + synthesis).

    Vectorised: because ``hop`` divides ``n_fft``, the overlap-add is ``n_fft/hop``
    whole-array adds rather than one add per frame.  That is the difference between
    Griffin-Lim taking 2 s and 40 ms per render.
    """
    assert n_fft % hop == 0, "n_fft must be a multiple of hop"
    r = n_fft // hop
    win2 = (_hann(n_fft) ** 2).reshape(r, hop)
    frames = np.fft.irfft(spec, n=n_fft, axis=1) * _hann(n_fft)
    n_frames = frames.shape[0]
    blocks = frames.reshape(n_frames, r, hop)

    out = np.zeros((n_frames + r, hop), dtype=np.float64)
    norm = np.zeros((n_frames + r, hop), dtype=np.float64)
    for k in range(r):
        out[k:k + n_frames] += blocks[:, k, :]
        norm[k:k + n_frames] += win2[k]
    # Normalise only where the window actually covers the sample.  Dividing by a
    # near-zero norm at the very first and last samples multiplies rounding noise by
    # thousands, which is an audible click at the start of every render.
    covered = norm > 1e-3
    out = np.where(covered, out / np.where(covered, norm, 1.0), 0.0)
    flat = out.ravel()
    if length is None:
        return flat.astype(np.float32)
    if len(flat) >= length:
        return flat[:length].astype(np.float32)
    return np.pad(flat, (0, length - len(flat))).astype(np.float32)


def griffin_lim(mag: np.ndarray, n_fft: int = N_FFT, hop: int = HOP,
                iters: int = 64, momentum: float = 0.99,
                seed: int = 0) -> np.ndarray:
    """Estimate a waveform whose STFT magnitude matches ``mag``.

    Fast Griffin-Lim (Perraudin et al. 2013): project onto the magnitude constraint,
    then accelerate along the direction of successive projected estimates.  ~64
    iterations reaches the floor set by 40-band mel inversion, not by phase.
    """
    mag = np.asarray(mag, dtype=np.float64)
    n_frames = mag.shape[0]
    length = n_frames * hop + n_fft
    rng = np.random.default_rng(seed)
    c = mag * np.exp(2j * np.pi * rng.random(mag.shape))
    c_prev = c.copy()
    for i in range(iters):
        s = stft(istft(c, n_fft, hop, length), n_fft, hop)[:n_frames]
        c = mag * np.exp(1j * np.angle(s))          # projected estimate
        accelerated = c + momentum * (c - c_prev)   # c_prev = previous *projected*
        c_prev = c
        c = accelerated
    return istft(c, n_fft, hop, length)


def logmel(x: np.ndarray, sr: int = SR, n_mels: int = N_MELS,
           n_fft: int = N_FFT, hop: int = HOP,
           fb: np.ndarray | None = None) -> np.ndarray:
    """(frames, n_mels) log-magnitude mel spectrogram, dB-scale, floored at -80 dB."""
    fb = mel_filterbank(sr, n_fft, n_mels) if fb is None else fb
    mel = stft_mag(x, n_fft, hop) @ fb.T.astype(np.float32)
    out = 20.0 * np.log10(np.maximum(mel, 1e-8))
    # Absolute floor, not peak-relative: a peak-relative floor makes the top of the
    # range depend on the loudest frame, so the same signal normalised differently
    # trains a model that has to re-learn loudness every clip.  -100 dB is below any
    # real recording and above log(0).
    return np.maximum(out, -100.0).astype(np.float32)


def mel_to_mag(logmel_db: np.ndarray, sr: int = SR, n_mels: int = N_MELS,
               n_fft: int = N_FFT) -> np.ndarray:
    """Invert log-mel back to a linear magnitude spectrogram (least squares)."""
    fb = mel_filterbank(sr, n_fft, n_mels)
    mel = 10.0 ** (logmel_db / 20.0)
    inv = np.linalg.pinv(fb)
    return np.maximum(mel @ inv.T, 0.0).astype(np.float32)


def mel_to_wav(logmel_db: np.ndarray, sr: int = SR, iters: int = 48) -> np.ndarray:
    """log-mel → waveform, normalised to -1 dBFS peak."""
    mag = mel_to_mag(logmel_db, sr)
    x = griffin_lim(mag, iters=iters)
    peak = float(np.max(np.abs(x))) or 1.0
    return (x / peak * 0.89).astype(np.float32)


# --------------------------------------------------------------------------- #
# Harmonic resynthesis — the vocoder that knows the melody
# --------------------------------------------------------------------------- #

def f0_track_from_notes(notes, n_frames: int, sr: int = SR, hop: int = HOP) -> np.ndarray:
    """Per-mel-frame F0 from a note plan (``start``/``duration``/``f0_hz``).

    A chant is *sustained and harmonic*, and its pitch is not something to be
    estimated from a spectrogram: the svara notation already gives it.  Handing the
    known F0 to the vocoder is what makes resynthesis land in tune instead of an
    octave out, which is what phase-guessing from 40 mel bands does.
    """
    f0 = np.zeros(n_frames, dtype=np.float32)
    for nt in notes:
        get = (lambda k: nt[k]) if isinstance(nt, dict) else (lambda k: getattr(nt, k))
        a = int(get("start") * sr / hop)
        b = int((get("start") + get("duration")) * sr / hop)
        a, b = max(0, a), min(n_frames, max(a + 1, b))
        if b > a:
            f0[a:b] = float(get("f0_hz"))
    return f0


def _mel_centres(sr: int = SR, n_fft: int = N_FFT, n_mels: int = N_MELS,
                 fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """Centre frequency of each mel band, in Hz.  The mel axis is the envelope axis."""
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(min(fmax, sr / 2 - 100)), n_mels)
    return _mel_to_hz(mel_pts)


def harmonic_vocode(logmel_db: np.ndarray, f0_track: np.ndarray, sr: int = SR,
                    n_fft: int = N_FFT, hop: int = HOP, n_mels: int = N_MELS,
                    seed: int = 0, rolloff: float = 0.9,
                    phase_spread: float = 0.45,
                    breath: float = 0.03) -> np.ndarray:
    """Resynthesise from log-mel using a known F0 contour (source-filter).

    A chant is sustained and harmonic and **its pitch is not something to estimate
    from a spectrogram** — the svara notation already gives it. So build a harmonic
    series at exactly ``k*f0``, with amplitudes read off the mel envelope.

    This used to build an STFT frame per analysis hop and invert it. That cannot
    work cleanly, and measuring proved it rather than listening:

    * a harmonic placed on the nearest FFT *bin* is up to ``sr/(2*n_fft)`` = 21.5 Hz
      from its true frequency, which at a 147 Hz fundamental is a 2.6 % tuning error,
      different for every harmonic;
    * so the waveform inside each frame runs at the bin frequency while the phase
      advance between frames follows *f0*. The two disagree by up to 62 degrees
      across one window, and overlap-add of Hann windows with that drift partially
      cancels: a perfectly periodic 146.83 Hz input came out with periodicity 0.92
      and 46 % of its energy outside the harmonic frequencies. That is the buzzy,
      metallic, "obviously synthetic" quality.

    Now the harmonics are summed directly in the time domain with a phase
    accumulator, ``sum_k A_k(t) * cos(k*phi(t) + k*pi)``, where
    ``phi = 2*pi * integral(f0)``. Pitch is exact by construction, the waveform is
    periodic by construction (every harmonic is an integer multiple of the same
    instantaneous frequency, so there is no beating), and the glottal pulse shape
    comes from the ``k*pi`` phases.

    Unvoiced frames keep the frame-based path, because there the envelope *is* the
    signal: noise shaped by the mel curve, which is what a consonant is.
    """
    logmel_db = np.asarray(logmel_db, dtype=np.float64)
    n_frames = logmel_db.shape[0]
    n_bins = n_fft // 2 + 1
    centres = _mel_centres(sr, n_fft, n_mels)
    freqs = np.arange(n_bins) * sr / n_fft
    rng = np.random.default_rng(seed)
    nyq = sr / 2.0
    n = (n_frames - 1) * hop + n_fft

    # The mel curve, extended across the spectrum so interpolation never extrapolates.
    xs = np.concatenate(([0.0], centres, [nyq]))
    db = np.concatenate((logmel_db[:, :1], logmel_db, logmel_db[:, -1:]), axis=1)
    env = 10.0 ** (np.stack([np.interp(freqs, xs, row) for row in db]) / 20.0)

    f0_track = np.asarray(f0_track, dtype=np.float64)
    if len(f0_track) < n_frames:
        f0_track = np.pad(f0_track, (0, n_frames - len(f0_track)), mode="edge")
    f0_track = f0_track[:n_frames]
    voiced = f0_track > 0

    # Frame centres in seconds, for interpolating the frame-rate tracks to samples.
    tc = (np.arange(n_frames) * hop + n_fft / 2.0) / sr
    ts = np.arange(n) / sr

    # ---- per-sample pitch and phase ------------------------------------------ #
    f0_s = np.interp(ts, tc, f0_track) if voiced.any() else np.zeros(n)
    f0_s = np.where(f0_s > 0, f0_s, 0.0)
    # exact phase accumulator: phi(t) = 2*pi * integral of f0
    phi = 2.0 * np.pi * np.cumsum(f0_s) / sr

    # ---- harmonic amplitudes, frame rate -> sample rate ---------------------- #
    f_lo = 55.0
    f_max = float(np.max(f0_track[voiced])) if voiced.any() else 0.0
    k_max = int(nyq / f_lo) if f_max <= 0 else int(min(nyq / max(f_max, 1e-9), 80))
    y = np.zeros(n)
    if k_max >= 1:
        # Amplitude of harmonic k in each frame, from the envelope at k*f0 — then
        # scaled per frame so the *power in each mel band* matches the mel frame.
        #
        # The previous version matched the frame's single loudest band, which is a
        # heuristic that silently loses energy: measured, it produced 0.155 RMS from a
        # source whose own mel gave 0.261 — a 4.5 dB error, applied to every frame,
        # which then looked like "the trained engine is quieter" in an A/B listen.
        #
        # The exact statement is linear in power: `logmel` computes `fb @ |X|^2`, so if
        # the comb's per-band power is `p0` and the target is `t = 10**(mel/10)`, the
        # gain satisfying it in least squares is
        #     g^2 = (p0 . t) / (p0 . p0)
        # — one dot product per frame, and it makes the vocoder's output have the
        # loudness the mel frame describes by construction.
        fb = _filterbank_cached(sr, n_fft, n_mels)
        kk = np.arange(1, k_max + 1)
        A = np.zeros((n_frames, k_max))
        line = np.zeros(n_bins)
        for f_i in np.flatnonzero(voiced):
            f0 = float(f0_track[f_i])
            line[:] = 0.0
            # amplitude of each harmonic from the envelope at its own frequency
            for k in kk:
                fk = k * f0
                if fk >= nyq or fk < f_lo:
                    continue
                j = int(round(fk * n_fft / sr))
                if 0 < j < n_bins:
                    a = float(np.interp(fk, freqs, env[f_i])) * (k ** -rolloff)
                    A[f_i, k - 1] = a
                    line[j] = a
            # per-frame power match, so the comb has the loudness the mel frame says
            p0 = fb @ (line ** 2)
            tgt = 10.0 ** (logmel_db[f_i] / 10.0)
            num = float(p0 @ tgt)
            den = float(p0 @ p0)
            g = math.sqrt(max(num, 0.0) / den) if den > 0 else 0.0
            A[f_i] *= min(max(g, 0.0), 40.0)
        # Harmonic phases are spread over the cycle instead of all aligning at one
        # instant.  With every harmonic in phase the sum is a delta-like click: measured
        # crest factor 17-25 dB against the formant engine's 11 dB, which a peak limiter
        # then squashes, costing the render ~13 dB of level.  A glottal pulse is broad,
        # not a click, so the phases follow `psi_k = k*phase_spread`, which is a
        # time-shift of the pulse (no audible change in pitch or timbre, since the
        # magnitude spectrum is identical) while lowering the crest factor.
        for k in range(k_max):
            a = np.interp(ts, tc, A[:, k])
            if not np.any(a):
                continue
            y += a * np.cos((k + 1) * phi + (k + 1) * phase_spread)

    # ---- unvoiced and breath: the frame-based path, where it is correct ------ #
    noise_frames = (~voiced)
    if breath > 0 or noise_frames.any():
        spec = np.zeros((n_frames, n_bins), dtype=np.complex128)
        # aspiration lives above the fundamental so it cannot read as a lower note
        lo = int(min(n_bins - 1, max(f_lo, 2.5 * f_max) * n_fft / sr)) if f_max else 0
        for i in np.flatnonzero(noise_frames | (voiced & (breath > 0))):
            e = env[i].copy()
            e[:lo] *= 0.03
            mag = e * rng.standard_normal(n_bins)
            if noise_frames[i]:
                spec[i] = mag * np.exp(2j * np.pi * rng.random(n_bins))
            else:
                spec[i] = breath * mag * np.exp(2j * np.pi * rng.random(n_bins))
        nz = istft(spec, n_fft, hop, length=n)
        y = y + nz[:n]

    # Level: match the mel's own level, rather than peak-normalising.  A glottal pulse
    # train is peaked, so dividing by the peak throws away several dB — and because the
    # rule engine does not peak-normalise the same way, the trained render came out
    # quieter than the render it was meant to colour.
    #
    # Note on computing this: use the *same* `logmel` the target was made with, and
    # scale by the difference of mean levels.  An earlier attempt built the mel with a
    # positional call (`logmel(y, sr, n_mels, n_fft, hop)`) while the target came from
    # `logmel(x, sr)`; the two agreed in shape but 16 dB apart in level, which is how a
    # function that looks like a pure rescale turns into a quiet render.
    #    Match *power*, not the mean of the dB values.  Averaging dB across 40 bands
    #    weights a silent band as heavily as a loud one, and a harmonic comb cannot fill
    #    bands the mel has at -55 dB, so the mean reads 8 dB low even when the loud bands
    #    are perfect — measured: +8.2 dB too loud on a steady tone.  Summing linear power
    #    is what "same loudness" actually means.
    got_db = logmel(y.astype(np.float32), sr)
    ref = np.asarray(logmel_db)
    p_got = float(np.sum(10.0 ** (got_db / 10.0)))
    p_ref = float(np.sum(10.0 ** (ref / 10.0)))
    if p_got > 0.0 and p_ref > 0.0 and np.isfinite(p_got) and np.isfinite(p_ref):
        y = y * math.sqrt(p_ref / p_got)
    # No peak limiter here.  A harmonic source has a 16 dB crest factor against the
    # formant engine's 11 dB, so hard-limiting to full scale silently removed ~13 dB of
    # level — and the caller then had nothing left to match against, which is why the
    # trained render was consistently quieter than the render it coloured.  Level and
    # limiting belong to exactly one place: the caller, which knows what the target is.
    return y.astype(np.float32)

    peak = float(np.max(np.abs(x))) or 1.0
    return (x / peak * 0.89).astype(np.float32)
