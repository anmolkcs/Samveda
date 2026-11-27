"""A source-filter formant synthesiser for sāmagāna.

Why ship a formant synthesiser as the default voice?

* It is **deterministic and cheap** — the demo must run on a machine with no GPU,
  no checkpoint, no model download, and still sound like a person chanting rather
  than a scanner beep.
* Chant is the one speech style where a source-filter model is genuinely
  competitive: one sustained voice, a narrow pitch range, long vowels, and almost
  none of the coarticulation storms that make formant speech sound robotic.
* Vedic chant is mostly vowel, and the intelligibility limit of formant synthesis
  is precisely consonants.

The engine is a Klatt-style cascade with the details that make chant read as human:

* a **Rosenberg glottal pulse** source, with per-period **jitter** and **shimmer**
  (a perfectly periodic source sounds synthetic within about two seconds);
* **aspiration noise**, increasing for the aspirated series, so kʰ/gʰ/bʰ are
  acoustically different from k/g/b and not merely longer;
* **spectral shaping per phone**: the sibilants ś/ṣ/s sit in three distinct noise
  bands, and the jihvāmūlīya / upadhmānīya visarga allophones keep their places;
* **formant transitions** between phones — consonant place is carried by the F2
  movement in the neighbouring vowel, which is exactly how humans hear it;
* a **nasal antiformant** for the anusvāra and the nasal series;
* **retroflex** F3 lowering, which is what makes ṭ/ḍ/ṇ read as retroflex.

Output is mono float32 at 22.05 kHz.  The piece that a neural model replaces is the
*source and the formant tracks* (see :mod:`samadhvani.synth.neural`); the cascade
stays, because a trained model that only predicts F0 and formant tracks needs far
less data than one that predicts waveform, and "trainable on a free Kaggle GPU"
is a hard requirement here.

Efficiency note: every phone's filters are constant, so each phone is filtered
once with ``scipy.signal.lfilter`` (a C loop) rather than per sample in Python.
Phones are then joined with 12 ms equal-power crossfades, which is the standard
diphone-synthesis trick and is audibly indistinguishable from a continuously
varying formant track at these transition rates.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from ..phones.inventory import CONSONANT_LOCUS, NOISE_BANDS, PHONES

SR = 22050

#: F1/F2/F3 half-power bandwidths, in Hz, for a male voice.  Narrower than speech
#: default: chant is sustained, so the formants are better defined.
BANDWIDTHS = (78.0, 95.0, 140.0, 190.0)


# --------------------------------------------------------------------------- #
# Glottal source
# --------------------------------------------------------------------------- #


def glottal_source(f0: np.ndarray, sr: int, *, jitter: float = 0.0035,
                   shimmer: float = 0.035, rng: np.random.Generator | None = None,
                   open_quotient: float = 0.62) -> np.ndarray:
    """Band-limited glottal pulse train with jitter and shimmer.

    Each period is a Rosenberg pulse: a raised-cosine opening phase and a fast
    exponential closing phase.  That asymmetry is what gives a real glottal flow its
    spectral tilt (about -12 dB/octave above the first formant) and it is why this
    sounds like a voice instead of a sawtooth.
    """
    rng = rng or np.random.default_rng(7)
    n = len(f0)
    out = np.zeros(n, dtype=np.float64)
    pos = 0
    while pos < n:
        f = f0[pos]
        if not np.isfinite(f) or f <= 0:
            pos += 32
            continue
        period = sr / f
        period *= 1.0 + jitter * rng.standard_normal()
        npulse = int(round(np.clip(period, 8.0, 4096.0)))
        if npulse < 6:
            pos += max(1, npulse)
            continue
        open_n = max(2, int(npulse * open_quotient))
        t = np.arange(open_n) / open_n
        pulse = 0.5 - 0.5 * np.cos(np.pi * t)              # smooth opening
        pulse *= np.exp(-2.2 * t)                          # fast closing
        close_n = npulse - open_n
        if close_n > 0:
            t2 = np.arange(close_n) / close_n
            pulse = np.concatenate([pulse, 0.30 * 0.5 * (1 + np.cos(np.pi * t2))])
        amp = 1.0 + shimmer * rng.standard_normal()
        seg = pulse * amp
        end = min(pos + len(seg), n)
        out[pos:end] += seg[: end - pos]
        pos += npulse
    return out


def _biquad_lowpass(fc: float, sr: int, q: float = 0.7071) -> tuple[np.ndarray, np.ndarray]:
    w0 = 2 * np.pi * min(fc, sr * 0.49) / sr
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return b / a[0], a / a[0]


def resonator(f: float, bw: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Two-pole resonator as an IIR biquad (the Klatt formant element)."""
    r = float(np.exp(-np.pi * bw / sr))
    theta = 2 * np.pi * f / sr
    a1 = 2 * r * np.cos(theta)
    a2 = -(r * r)
    gain = (1 - r) * np.sqrt(1 - 2 * r * np.cos(2 * theta) + r * r)
    b = np.array([gain])
    a = np.array([1.0, -a1, -a2])
    return b, a


def _band_noise(n: int, band: tuple[float, float], sr: int,
                rng: np.random.Generator) -> np.ndarray:
    """Noise shaped into a band with -12 dB/octave skirts, via the FFT."""
    if n <= 1:
        return np.zeros(max(n, 0))
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    lo, hi = band
    shape = np.ones_like(freqs)
    below = freqs < lo
    above = freqs > hi
    shape[below] = (np.maximum(freqs[below], 1.0) / lo) ** 2
    shape[above] = (hi / np.maximum(freqs[above], 1.0)) ** 2
    y = np.fft.irfft(spec * shape, n)
    rms = float(np.sqrt(np.mean(y * y)))
    return y / rms if rms > 1e-9 else y


# --------------------------------------------------------------------------- #
# Per-phone articulation
# --------------------------------------------------------------------------- #

#: Noise band for each phone that is noisy at all.
_PHONE_NOISE: dict[str, tuple[float, float]] = {
    "s": NOISE_BANDS["s"],
    "sh": NOISE_BANDS["sh"],
    "ss": NOISE_BANDS["ss"],
    "h": NOISE_BANDS["h"],
    "H": NOISE_BANDS["h"],
    "kh_h": NOISE_BANDS["kh_h"],
    "ph_h": NOISE_BANDS["ph_h"],
    "murmur": (320.0, 1400.0),
}

#: Which phones are oral (a formant structure worth rendering).
_VOWEL_SYMS = {"a", "aa", "i", "ii", "u", "uu", "R", "RR", "L", "e", "ai", "o", "au"}
_NASAL_SYMS = {"ng", "ny", "nn", "n", "m", "murmur"}
_APPROX_SYMS = {"y", "r", "l", "v", "zh"}
_STOP_SYMS = {
    "k", "kh", "g", "gh", "c", "ch", "j", "jh", "tt", "tth", "dd", "ddh",
    "t", "th", "d", "dh", "p", "ph", "b", "bh",
}
_FRIC_SYMS = {"s", "sh", "ss", "h", "H", "kh_h", "ph_h"}


def vowel_targets(sym: str) -> tuple[float, float, float]:
    """Formant triple for a vowel symbol, handling diphthongs as a glide."""
    if sym in DIPHTHONG_STARTS:
        a, b = DIPHTHONG_STARTS[sym]
        fa = PHONES[a].formants
        fb = PHONES[b].formants
        return fa, fb  # type: ignore[return-value]
    p = PHONES.get(sym)
    return p.formants if p else (500, 1500, 2500)  # type: ignore[return-value]


#: Sanskrit e and o are always diphthongs; ai/au likewise.
DIPHTHONG_STARTS: dict[str, tuple[str, str]] = {
    "e": ("a", "i"), "ai": ("a", "i"), "o": ("a", "u"), "au": ("a", "u"),
}


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Guarantee exactly ``n`` samples (renderers may return fewer)."""
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


class FormantSynth:
    """Render a note plan to audio with a Klatt-style cascade."""

    def __init__(self, sr: int = SR, seed: int = 7):
        self.sr = sr
        self.seed = seed

    # -- one segment with static formants ---------------------------------- #
    def _voice_segment(self, n: int, formants: tuple[float, float, float],
                       f0: np.ndarray, *, aspiration: float = 0.03,
                       tilt_hz: float = 5200.0, nasal: bool = False,
                       noise: tuple[float, float] | None = None,
                       noise_gain: float = 0.0,
                       breathy: bool = False,
                       voice_rms: float = 0.30) -> np.ndarray:
        """Render ``n`` samples of a sustained voiced (and optionally noisy) phone.

        Voice and noise levels are normalised **independently**.  If they are not,
        the noise path — which is white-ish and therefore has far more energy per
        sample — drowns the voice completely, and the result is a whisper with
        occasional pitch.  Every level here is an absolute target rms, so the mix
        of a vowel and a sibilant is set by design rather than by accident.
        """
        sr = self.sr
        rng = np.random.default_rng(self.seed + n % 997)
        if n <= 0:
            return np.zeros(0)

        voice = np.zeros(n)
        if voice_rms > 0:
            src = glottal_source(f0, sr, rng=rng)
            # Lip radiation is a differentiator (-6 dB/oct, +90 deg).  Omitting it
            # is the commonest formant-synthesis bug: it leaves a large DC term and
            # a boomy spectrum that sounds like a foghorn rather than a voice.
            src = np.diff(src, prepend=src[0] if len(src) else 0.0)
            b, a = _biquad_lowpass(tilt_hz, sr)
            src = lfilter(b, a, src)
            src_rms = float(np.sqrt(np.mean(src * src))) or 1.0

            # Aspiration is injected at the glottis, before the vocal-tract filter,
            # which is physically correct: breath noise is shaped by the same tube.
            # It must be scaled *relative to the glottal source*, not to 1.0 —
            # otherwise the noise overwhelms the pulse train and the result is a
            # whisper with a faint pitch under it.
            if aspiration > 0:
                hnr = aspiration * (2.2 if breathy else 1.0)
                src = src + _band_noise(n, (250.0, 4200.0), sr, rng) * (hnr * src_rms)

            out = src
            for f, bw in zip(formants, BANDWIDTHS):
                if f <= 0:
                    continue
                bb, aa = resonator(f, bw, sr)
                out = lfilter(bb, aa, out)
            if nasal:
                bz, az = resonator(1050.0, 130.0, sr)
                out = out - 0.45 * lfilter(bz, az, src)

            # the cascade has an arbitrary overall scale; pin it by rms
            r = float(np.sqrt(np.mean(out * out)))
            voice = out * (voice_rms / r) if r > 1e-9 else out

        if noise is not None and noise_gain > 0:
            nz = _band_noise(n, noise, sr, rng)
            bn, an = _biquad_lowpass(min(noise[1] * 1.15, sr * 0.48), sr)
            nz = lfilter(bn, an, nz)
            r = float(np.sqrt(np.mean(nz * nz)))
            nz = nz * (noise_gain / r) if r > 1e-9 else nz
            voice = voice + nz
        return voice

    # -- one phone ---------------------------------------------------------- #
    def render_phone(self, sym: str, n: int, f0: np.ndarray) -> np.ndarray:
        """Render a phone of ``n`` samples with the given F0 track."""
        if n <= 0:
            return np.zeros(0)
        if len(f0) != n:
            f0 = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(f0)), f0)

        if sym in _VOWEL_SYMS:
            tgt = vowel_targets(sym)
            if isinstance(tgt[0], tuple):
                fa, fb = tgt
                # diphthong: two halves with a formant glide between them
                half = n // 2
                first = self._voice_segment(half, fa, f0[:half])  # type: ignore[arg-type]
                second = self._voice_segment(n - half, fb, f0[half:])  # type: ignore[arg-type]
                return self._crossfade_join(first, second, min(int(0.045 * self.sr), half or 1))
            return self._voice_segment(n, tgt, f0, aspiration=0.035, voice_rms=0.32)  # type: ignore[arg-type]

        if sym in _NASAL_SYMS:
            p = PHONES.get(sym)
            fm = p.formants if p and p.formants[0] > 0 else (300.0, 1100.0, 2400.0)
            # nasals: low F1, damped higher formants
            fm = (fm[0] if fm[0] else 300.0, fm[1] if fm[1] else 1100.0, 2300.0)
            return self._voice_segment(n, fm, f0, nasal=True, voice_rms=0.24)  # type: ignore[arg-type]

        if sym in _APPROX_SYMS:
            place = PHONES[sym].place if sym in PHONES else "dental"
            locus = CONSONANT_LOCUS.get(place, (1500.0, 2400.0))
            fm = (340.0, locus[0], locus[1])
            return self._voice_segment(n, fm, f0, aspiration=0.02, voice_rms=0.26)  # type: ignore[arg-type]

        if sym in _STOP_SYMS:
            p = PHONES[sym]
            place = p.place
            locus = CONSONANT_LOCUS.get(place, (1500.0, 2400.0))
            fm = (320.0, locus[0], locus[1])
            voiced = p.voiced
            aspirated = p.aspirated
            # closure is silent except for voice bar in voiced stops
            if n < 6:
                return np.zeros(n)
            close_n = int(n * 0.62)
            out = np.zeros(n)
            if voiced:
                out[:close_n] = self._voice_segment(
                    close_n, (180.0, 900.0, 2200.0), f0[:close_n], aspiration=0.0,
                    voice_rms=0.10)
            # release burst
            burst_n = max(2, int(0.010 * self.sr))
            burst_n = min(burst_n, n - close_n)
            b, a = _biquad_lowpass(min(locus[1] * 1.6, self.sr * 0.48), self.sr)
            rng = np.random.default_rng(self.seed + hash(sym) % 1000)
            burst = lfilter(b, a, rng.standard_normal(burst_n))
            br = float(np.sqrt(np.mean(burst * burst))) or 1.0
            burst *= (0.16 if not aspirated else 0.20) / br
            out[close_n:close_n + burst_n] += burst
            # aspiration after the burst for the aspirated series
            tail = n - close_n - burst_n
            if tail > 0:
                if aspirated:
                    gap = _band_noise(tail, (400.0, 3600.0), self.sr, rng)
                    gr = float(np.sqrt(np.mean(gap * gap))) or 1.0
                    out[close_n + burst_n:] += gap * 0.10 / gr
                if voiced:
                    voiced_tail = self._voice_segment(
                        tail, fm, f0[close_n + burst_n:], aspiration=0.02, voice_rms=0.12)
                    out[close_n + burst_n:] += voiced_tail[:tail]
            return out

        if sym in _FRIC_SYMS or sym in _PHONE_NOISE:
            band = _PHONE_NOISE.get(sym, (2000.0, 6000.0))
            if sym in ("h", "H", "kh_h", "ph_h"):
                # these are breathy and partly voiced
                return self._voice_segment(
                    n, (500.0, 1500.0, 2500.0), f0,
                    aspiration=0.10, noise=band, noise_gain=0.09, breathy=True,
                    voice_rms=0.16)
            return self._voice_segment(
                n, (400.0, 1800.0, 3000.0), f0,
                aspiration=0.0, noise=band, noise_gain=0.13, voice_rms=0.0)

        # pause / unknown
        return np.zeros(n)

    # -- glue --------------------------------------------------------------- #
    @staticmethod
    def _crossfade_join(a: np.ndarray, b: np.ndarray, n_fade: int) -> np.ndarray:
        if n_fade <= 0 or len(a) == 0 or len(b) == 0:
            return np.concatenate([a, b])
        n_fade = min(n_fade, len(a), len(b))
        t = np.linspace(0, np.pi / 2, n_fade)
        win_a = np.cos(t)[:, None] if a.ndim > 1 else np.cos(t)
        win_b = np.sin(t)[:, None] if b.ndim > 1 else np.sin(t)
        out = np.concatenate([a[:-n_fade], a[-n_fade:] * win_a + b[:n_fade] * win_b, b[n_fade:]])
        return out

    def render_note(self, phones: list[str], f0: np.ndarray, n: int) -> np.ndarray:
        """Render one note, distributing its samples across its phones."""
        if n <= 0:
            return np.zeros(0)
        if not phones:
            phones = ["a"]
        # Weights: the nucleus is held, onsets are brief, codas are shorter.
        weights = []
        for sym in phones:
            if sym in _VOWEL_SYMS:
                weights.append(2.8)
            elif sym in _PHONE_NOISE or sym in _FRIC_SYMS:
                weights.append(1.5)
            elif sym in _STOP_SYMS:
                weights.append(0.75)
            elif sym in _NASAL_SYMS:
                weights.append(0.95)
            else:
                weights.append(1.0)
        total_w = sum(weights)
        alloc = [max(1, int(n * w / total_w)) for w in weights]
        # fix rounding so the sum is exactly n
        while sum(alloc) > n and len(alloc) > 1:
            alloc[-1] -= 1
        alloc[-1] += n - sum(alloc)

        cursor = 0
        out = np.zeros(n)
        for sym, ln in zip(phones, alloc):
            ln = max(0, min(ln, n - cursor))
            if ln == 0:
                continue
            seg = self.render_phone(sym, ln, f0[cursor:cursor + ln])
            seg = _fit_length(seg, ln)
            # overlap-add with a short crossfade to its predecessor
            if cursor > 0:
                fade = min(int(0.012 * self.sr), ln // 2)
                if fade > 0:
                    t = np.linspace(0, np.pi / 2, fade)
                    out[cursor:cursor + fade] = out[cursor:cursor + fade] * np.cos(t) + seg[:fade] * np.sin(t)
                    out[cursor + fade:cursor + ln] += seg[fade:]
                else:
                    out[cursor:cursor + ln] += seg
            else:
                out[cursor:cursor + ln] += seg
            cursor += ln
        return out
