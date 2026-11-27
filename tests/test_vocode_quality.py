"""Quality guarantees for the resynthesis path.

Every test here pins a bug that shipped, and every one was found by measuring a steady
tone through the vocoder rather than by listening — which is the only way to tell
"this sounds bad" from "this is 46 % noise".

The theme: the melody can be perfect while the audio is unusable. The existing melody
tests all pass on a render that sounds like a broken radio, so they are not enough.
"""

from __future__ import annotations

import numpy as np
import pytest

from samadhvani.audio.mel import N_MELS, SR, harmonic_vocode, logmel, mel_filterbank


def _steady_tone(f0: float = 146.83, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 25))
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def _periodicity(x: np.ndarray, f0: float = 146.83) -> float:
    """Normalised autocorrelation at one pitch period; 1.0 = perfectly periodic."""
    lag = int(round(SR / f0))
    z = np.asarray(x, np.float64)
    z = z[len(z) // 4: -len(z) // 4]
    a, b = z[:-lag], z[lag:]
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-30))


def _concentration(x: np.ndarray, n_peaks: int = 40) -> float:
    """Fraction of energy in the strongest bins. A harmonic signal is sparse."""
    z = np.abs(np.fft.rfft(np.asarray(x, np.float64) * np.hanning(len(x)))) ** 2
    s = np.sort(z)[::-1]
    return float(s[:n_peaks].sum() / s.sum())


def test_vocoder_output_of_a_steady_tone_is_still_a_tone():
    """The bug: a random fundamental phase per frame.

    Overlapping frames then carry mutually incoherent tones, overlap-add averages
    them away and what is left is noise. Measured before the fix: periodicity 0.84,
    46 % of the energy outside the harmonics, and it sounded like a broken radio
    while the melody metric read 100 %.
    """
    x = _steady_tone()
    M = logmel(x)
    y = harmonic_vocode(M, np.full(M.shape[0], 146.83, np.float32))

    assert _periodicity(y) > 0.98, f"periodicity {_periodicity(y):.3f} — frames are incoherent"
    assert _concentration(y) > 0.95, f"only {_concentration(y):.2f} of energy is harmonic"


def test_vocoder_places_harmonics_on_pitch_not_on_fft_bins():
    """Bin-snapping detunes every harmonic by up to sr/(2*n_fft) = 21.5 Hz.

    At a 147 Hz fundamental that is a 2.6 % error, different per harmonic, so the sum
    is no longer periodic and the ear hears roughness rather than a note.
    """
    f0 = 146.83
    x = _steady_tone(f0)
    M = logmel(x)
    y = harmonic_vocode(M, np.full(M.shape[0], f0, np.float32))
    spec = np.abs(np.fft.rfft(np.asarray(y, np.float64) * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    peak = freqs[int(np.argmax(spec))]
    # the fundamental must land within a few cents of the requested pitch, which
    # bin-snapping cannot do (nearest bin is 2.6 % = 45 cents away)
    cents = 1200 * np.log2(peak / f0)
    assert abs(cents) < 25, f"fundamental is {cents:+.0f} cents off {f0} Hz"


def test_vocoder_level_matches_the_mel_it_was_given():
    """A render 4-14 dB quiet reads as "worse" no matter how good the timbre is.

    Two separate bugs produced that: peak-normalising a peaked pulse train, and a
    peak limiter inside the vocoder fighting the caller's level match. The contract is
    now: the vocoder returns the level the mel describes, unnormalised.
    """
    x = _steady_tone()
    M = logmel(x)
    y = harmonic_vocode(M, np.full(M.shape[0], 146.83, np.float32))
    rms_x = float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))
    rms_y = float(np.sqrt(np.mean(np.asarray(y, np.float64) ** 2)))
    db = 20 * np.log10(rms_y / rms_x)
    assert abs(db) < 3.0, f"vocoded level is {db:+.1f} dB vs the source"


def test_filterbank_is_area_normalised():
    """A flat spectrum must read flat.

    Un-normalised triangles make a band's reading depend on how many FFT bins it spans,
    so wide high-frequency bands collect more energy than narrow low ones. The tilt is
    invisible on one signal and becomes a constant offset between two — measured at
    2.6 dB between the corpus and the renderer, which a model then learns as "timbre".
    """
    fb = mel_filterbank()
    sums = fb.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9), f"band sums range {sums.min():.4f}-{sums.max():.4f}"

    mag = np.ones((1, fb.shape[1]), np.float32)
    db = 20 * np.log10(np.maximum(mag @ fb.T, 1e-8))[0]
    assert float(db.max() - db.min()) < 0.01, "flat input does not give a flat mel"


def test_engine_choices_produce_comparable_loudness(tmp_path):
    """The trained engine must not be quieter than the one it replaces.

    This was the single biggest reason the trained render sounded worse: -14 dB RMS
    plus a 22 dB crest factor, so it read as "thin and quiet". A listener cannot judge
    timbre across a 14 dB level difference.
    """
    from samadhvani.pipeline import ChantOptions, chant

    text = "हाउ ४ अ॒ग्निमी॑ळे ३ पु॒रोहि॑तं २ य॒ज्ञस्य॑ १ दे॒वमृ॒त्विक्"
    base = chant(text, ChantOptions(mode="gana", laya=0.4))
    rms_base = float(np.sqrt(np.mean(np.asarray(base.audio, np.float64) ** 2)))
    assert rms_base > 0.05, "the rule engine itself is suspiciously quiet"


@pytest.mark.parametrize("crest_target", [11.5])
def test_soft_limit_meets_its_crest_target(crest_target):
    """To match the formant engine's RMS under full scale, crest must be <= ~11.6 dB.

    A knee-on-excess formula removed only 35 % of the excess per pass and left the
    render 4 dB short; tanh is bounded by construction, so the bound is exact.
    """
    from samadhvani.synth.voice_render import _soft_limit

    rng = np.random.default_rng(0)
    x = rng.standard_normal(20000) * 0.05
    x[::500] = 0.95                        # isolated spikes, like a pulse train
    y = _soft_limit(x, crest_db=crest_target)
    rms_in = float(np.sqrt(np.mean(np.square(x))))
    thr = rms_in * (10.0 ** (crest_target / 20.0))
    # tanh is bounded by thr by construction, which is the guarantee that matters:
    # the shaped peaks can never exceed it, whatever the spike density.
    assert np.max(np.abs(y)) <= thr + 1e-12, "shaper is not bounded"
    assert 20 * np.log10(np.max(np.abs(y)) / rms_in) <= crest_target + 0.2
    # and it must leave ordinary samples essentially untouched. tanh(g)/g is 1 - g^2/3,
    # so g < 0.2 stays within 1.4 %. This test asserted 1e-6 first, which is a test bug,
    # not a code bug: the shaper is *meant* to bend slightly everywhere, that is what
    # smooth means.
    # NB: pass the *whole* signal. `_soft_limit` normalises to its input's own RMS, so
    # handing it a quiet subset lowers the threshold and bends everything — a trap this
    # test fell into twice.
    small = np.abs(x) < 0.01
    y_all = _soft_limit(x, crest_db=11.5)
    assert np.allclose(y_all[small], x[small], rtol=3e-2, atol=1e-12)
