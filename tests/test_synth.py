"""Tests for the acoustic layer: timing, prosody, and the formant engine.

These are mostly *property* tests rather than golden-file comparisons, because the
synthesiser is stochastic (jitter, shimmer) and a golden WAV would fail on any
unrelated change.  What we pin down instead is the set of properties that, if they
broke, would make the output stop being chant:

* the pitch track follows the svara sequence, at the right absolute frequencies;
* held notes are long, and cadences are longer than the notes before them;
* the signal is periodic enough to read as a voice;
* voiced phones are not drowned by the noise path (a bug this suite caught live);
* every phone in the inventory renders without raising and with bounded output.
"""

import numpy as np
import pytest

from samadhvani.phones.g2p import phones_for_view
from samadhvani.phones.inventory import PHONES
from samadhvani.synth.engine import synthesize_formant
from samadhvani.synth.formant import FormantSynth
from samadhvani.synth.io import read_wav, wav_bytes, write_wav
from samadhvani.synth.prosody import f0_track, voicing_mask
from samadhvani.synth.timing import TimingOptions, plan_timing
from samadhvani.text.svaras import SAMA_SVARAS, annotate_svaras, svara_to_semitone
from samadhvani.text.syllabify import syllabify

VERSE = "हाउ अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म्"
SR = 22050


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

def _plan(text=VERSE, **kw):
    view = syllabify(text)
    svaras = [t.svara for t in annotate_svaras(view)]
    phones = [p.phones for p in phones_for_view(view)]
    return plan_timing(view.aksaras, svaras, groups=view.groups,
                       options=TimingOptions(**kw), phone_lists=phones)


def test_plan_covers_every_syllable():
    view = syllabify(VERSE)
    plan = _plan()
    assert len(plan.notes) == len(view.aksaras)


def test_notes_are_contiguous_and_ordered():
    plan = _plan()
    for a, b in zip(plan.notes, plan.notes[1:]):
        assert b.start >= a.start
        assert a.duration > 0
        assert b.start >= a.end - 1e-9


def test_laya_scales_duration():
    slow = _plan(matra_seconds=0.55)
    fast = _plan(matra_seconds=0.28)
    assert slow.total_seconds > fast.total_seconds * 1.5


def test_cadence_is_longer_than_median_note():
    plan = _plan()
    cad = [n.duration for n in plan.notes if n.is_cadence]
    non = [n.duration for n in plan.notes if not n.is_cadence]
    if cad and non:
        assert np.mean(cad) > np.median(non)


def test_svara_to_frequency_matches_scale():
    tonic = 146.83
    for svara in range(1, 8):
        want = tonic * 2 ** (SAMA_SVARAS[svara]["semitone"] / 12)
        assert svara_to_semitone(svara, tonic_hz=tonic) == pytest.approx(want)


def test_register_shift_lowers_pitch():
    low = _plan(register_shift_semitones=-5)
    base = _plan()
    assert low.notes[0].f0_hz < base.notes[0].f0_hz


# --------------------------------------------------------------------------- #
# Prosody
# --------------------------------------------------------------------------- #

def test_f0_track_follows_svaras():
    plan = _plan(matra_seconds=0.4)
    f0 = f0_track(plan, SR)
    assert len(f0) > 0
    voiced = f0[f0 > 0]
    assert len(voiced) > 0
    # every scheduled note frequency should appear in the track
    for note in plan.notes[:6]:
        window = f0[int(note.start * SR): int(note.end * SR)]
        assert np.median(window) == pytest.approx(note.f0_hz, rel=0.06)


def test_f0_is_musically_bounded():
    plan = _plan()
    f0 = f0_track(plan, SR)
    voiced = f0[f0 > 0]
    assert voiced.min() > 60
    assert voiced.max() < 500


def test_voicing_mask_is_binary_in_the_middle():
    plan = _plan()
    mask = voicing_mask(plan, SR)
    assert mask.max() <= 1.0 + 1e-9
    assert mask.min() >= -1e-9
    assert mask.mean() > 0.5


# --------------------------------------------------------------------------- #
# The formant engine
# --------------------------------------------------------------------------- #

def _periodicity(x, sr=SR, lo=70, hi=400):
    from scipy.signal import butter, sosfilt
    x = sosfilt(butter(4, 1200 / (sr / 2), btype="low", output="sos"), x)
    frame = 2048
    vals = []
    lo_lag, hi_lag = int(sr / hi), int(sr / lo)
    for i in range(0, max(1, len(x) - frame), 1024):
        fr = x[i:i + frame]
        fr = fr - fr.mean()
        d = float(np.dot(fr, fr))
        if d < 1e-9:
            continue
        ac = np.correlate(fr, fr, "full")[len(fr) - 1:] / d
        seg = ac[lo_lag:hi_lag]
        if len(seg):
            vals.append(float(seg.max()))
    return float(np.mean(vals)) if vals else 0.0


@pytest.mark.parametrize("sym", ["a", "aa", "i", "u", "R", "m", "y"])
def test_voiced_phones_are_periodic(sym):
    """A voiced phone must read as a voice, not as noise.

    This is the regression test for the bug where the noise path (normalised to
    peak) completely masked the glottal source (attenuated by the cascade): voices
    measured ~0.1 periodicity and the output was a whisper.
    """
    synth = FormantSynth(seed=3)
    x = synth.render_phone(sym, int(0.35 * SR), np.full(int(0.35 * SR), 146.83))
    assert _periodicity(x) > 0.45, f"{sym} is not periodic enough to be a voice"


def test_voiced_and_noise_phones_have_comparable_levels():
    synth = FormantSynth(seed=3)
    n = int(0.25 * SR)
    f0 = np.full(n, 146.83)
    vowel = np.sqrt(np.mean(synth.render_phone("aa", n, f0) ** 2))
    sib = np.sqrt(np.mean(synth.render_phone("ss", n, f0) ** 2))
    # neither should bury the other by more than ~15 dB
    ratio_db = abs(20 * np.log10((vowel + 1e-9) / (sib + 1e-9)))
    assert ratio_db < 15


def test_aspirated_stop_has_more_noise_than_plain():
    synth = FormantSynth(seed=5)
    n = int(0.12 * SR)
    f0 = np.full(n, 146.83)
    plain = synth.render_phone("k", n, f0)
    aspirated = synth.render_phone("kh", n, f0)

    def flatness(x):
        p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 + 1e-12
        return float(np.exp(np.mean(np.log(p))) / np.mean(p))

    assert flatness(aspirated) >= flatness(plain)


def test_retroflex_and_dental_differ():
    synth = FormantSynth(seed=7)
    n = int(0.10 * SR)
    f0 = np.full(n, 146.83)
    a = synth.render_phone("t", n, f0)
    b = synth.render_phone("tt", n, f0)
    assert not np.allclose(a, b)


def test_every_inventory_phone_renders():
    """No phone may crash the engine, and every output must be finite and bounded."""
    synth = FormantSynth(seed=1)
    n = int(0.08 * SR)
    f0 = np.full(n, 146.83)
    for sym in PHONES:
        x = synth.render_phone(sym, n, f0)
        assert np.all(np.isfinite(x)), f"{sym} produced non-finite samples"
        assert np.abs(x).max() < 50, f"{sym} is wildly unbounded"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_full_render_is_stable(seed):
    plan = _plan(matra_seconds=0.3)
    view = syllabify(VERSE)
    phones = [p.phones for p in phones_for_view(view)]
    x, sr = synthesize_formant(plan, phones, seed=seed)
    assert sr == SR
    assert np.all(np.isfinite(x))
    assert np.abs(x).max() <= 1.0
    assert np.sqrt(np.mean(x ** 2)) > 0.02          # not silence
    assert _periodicity(x) > 0.4                     # still a voice after post-fx


def test_render_is_deterministic_given_seed():
    plan = _plan(matra_seconds=0.3)
    view = syllabify(VERSE)
    phones = [p.phones for p in phones_for_view(view)]
    a, _ = synthesize_formant(plan, phones, seed=11)
    b, _ = synthesize_formant(plan, phones, seed=11)
    assert np.allclose(a, b)


def test_different_seeds_differ():
    plan = _plan(matra_seconds=0.3)
    view = syllabify(VERSE)
    phones = [p.phones for p in phones_for_view(view)]
    a, _ = synthesize_formant(plan, phones, seed=1)
    b, _ = synthesize_formant(plan, phones, seed=2)
    assert not np.allclose(a, b)


def test_dc_offset_is_removed():
    plan = _plan(matra_seconds=0.3)
    view = syllabify(VERSE)
    phones = [p.phones for p in phones_for_view(view)]
    x, _ = synthesize_formant(plan, phones)
    assert abs(float(np.mean(x))) < 0.01


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #

def test_wav_roundtrip(tmp_path):
    x = (0.5 * np.sin(2 * np.pi * 220 * np.arange(4410) / SR)).astype(np.float32)
    path = str(tmp_path / "t.wav")
    write_wav(path, x, SR)
    y, sr = read_wav(path)
    assert sr == SR
    assert len(y) == len(x)
    assert np.max(np.abs(y - x)) < 0.01


def test_wav_bytes_is_a_riff_file():
    data = wav_bytes(np.zeros(100, dtype=np.float32), SR)
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
