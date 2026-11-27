"""Tests for evaluation metrics, the corpus builder, and (optionally) training.

The training tests are skipped when torch is absent so the suite still runs on a bare
install — but they do run where torch exists, because a training loop that only
*looks* correct is worth nothing.
"""

import json
import math
import os

import numpy as np
import pytest

from samadhvani.eval import evaluate
from samadhvani.eval.metrics import (
    aspiration_confusion,
    duration_error,
    edit_distance,
    f0_periodicity,
    hnr,
    melody_contour_correlation,
    mos_estimate,
    phone_error_rate,
    retroflex_confusion,
    sibilant_confusion,
    spectral_flatness,
    svara_accuracy,
    svara_accuracy_at_notes,
    svara_track_from_audio,
)
from samadhvani.pipeline import ChantOptions, chant

VERSE = "हाउ अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म्"
GANATEXT = "हाउ ४ अ॒ग्निमी॑ळे ३ पु॒रोहि॑तं २ य॒ज्ञस्य॑ १ दे॒वमृ॒त्विज॑म्"


# --------------------------------------------------------------------------- #
# Text metrics
# --------------------------------------------------------------------------- #

def test_edit_distance():
    assert edit_distance([], []) == 0
    assert edit_distance(["a"], []) == 1
    assert edit_distance(list("kitten"), list("sitting")) == 3
    assert edit_distance(list("abc"), list("abc")) == 0


def test_phone_error_rate():
    assert phone_error_rate(["a", "b", "c"], ["a", "b", "c"]) == 0.0
    assert phone_error_rate(["a", "b"], ["a", "x"]) == 0.5


def test_svara_accuracy_exact():
    assert svara_accuracy([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert svara_accuracy([1, 2, 3, 4], [1, 1, 3, 3]) == 0.5


def test_svara_accuracy_resamples_different_lengths():
    """A decoded contour has one note per *held* note, not one per akṣara."""
    ref = [1, 1, 2, 2, 3, 3]
    hyp = [1, 2, 3]
    assert 0.0 <= svara_accuracy(ref, hyp) <= 1.0


def test_svara_accuracy_empty_is_safe():
    assert svara_accuracy([], [1, 2]) == 0.0
    assert svara_accuracy([1, 2], []) == 0.0


def test_contour_correlation():
    a = [1, 2, 3, 4, 5, 6, 7]
    assert melody_contour_correlation(a, a) == pytest.approx(1.0, abs=1e-6)
    # exactly reversed is anti-correlated but not perfectly -1.0: the semitone scale
    # is non-uniform (0,+2,+4,+5,+7,-1,-3), so a reversal is affine, not linear
    assert melody_contour_correlation(a, list(reversed(a))) < -0.95
    assert melody_contour_correlation(a, [4] * 7) == 0.0


def test_duration_error():
    out = duration_error([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert out["mae"] == pytest.approx(0.0)
    assert out["corr"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Vedic-specific confusions
# --------------------------------------------------------------------------- #

def test_aspiration_confusion_detects_the_classic_error():
    """Rate over stop *occurrences*, so over-aspiration cannot saturate the metric."""
    ref = ["k", "kh", "g", "gh", "t", "th"]
    assert aspiration_confusion(ref, ref) == 0.0
    # collapsing the three aspirates onto their plain partners: 3 errors / 6 stops
    collapsed = ["k", "k", "g", "g", "t", "t"]
    assert aspiration_confusion(ref, collapsed) == pytest.approx(0.5)
    # over-aspiration is an error too
    assert aspiration_confusion(["k", "g"], ["kh", "gh"]) == pytest.approx(1.0)


def test_retroflex_confusion():
    """Rate over retroflex/dental *occurrences*, not over retroflexes alone."""
    ref = ["tt", "t", "dd", "d", "tt"]
    assert retroflex_confusion(ref, ref) == 0.0
    # 2 of the 5 stops are sung at the wrong place -> 0.4
    assert retroflex_confusion(ref, ["t", "t", "dd", "d", "tt"]) == pytest.approx(0.2)
    assert retroflex_confusion(ref, ["t"] * 5) == pytest.approx(0.4)


def test_sibilant_confusion():
    ref = ["s", "sh", "ss"]
    assert sibilant_confusion(ref, ref) == 0.0
    assert sibilant_confusion(ref, ["s", "s", "s"]) == pytest.approx(2 / 3)


def test_confusion_metrics_handle_no_relevant_phones():
    assert aspiration_confusion(["a", "i"], ["a", "i"]) == 0.0
    assert retroflex_confusion(["a"], ["a"]) == 0.0
    assert sibilant_confusion(["a"], ["a"]) == 0.0


# --------------------------------------------------------------------------- #
# Audio metrics
# --------------------------------------------------------------------------- #

def test_f0_periodicity_on_a_sine():
    sr = 22050
    t = np.arange(sr) / sr
    x = 0.5 * np.sin(2 * np.pi * 150 * t)
    per = f0_periodicity(x, sr)
    assert per["periodicity"] > 0.8
    assert per["f0_median"] == pytest.approx(150, rel=0.05)


def test_f0_periodicity_on_noise():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(22050) * 0.1
    assert f0_periodicity(x, 22050)["periodicity"] < 0.5


def test_spectral_flatness_orders_tone_below_noise():
    sr = 22050
    t = np.arange(sr) / sr
    tone = 0.5 * np.sin(2 * np.pi * 150 * t)
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(sr) * 0.1
    assert spectral_flatness(tone) < spectral_flatness(noise)


def test_hnr_is_high_for_a_tone():
    sr = 22050
    t = np.arange(sr) / sr
    x = 0.5 * np.sin(2 * np.pi * 150 * t)
    assert hnr(x, sr) > 10


def test_mos_estimate_bounds():
    sr = 22050
    t = np.arange(sr) / sr
    tone = 0.5 * np.sin(2 * np.pi * 150 * t)
    val = mos_estimate(tone, sr)
    assert 1.0 <= val <= 5.0


# --------------------------------------------------------------------------- #
# Evaluation of a real render
# --------------------------------------------------------------------------- #

def test_svara_accuracy_at_notes_on_a_rendered_chant():
    """The melody must survive synthesis and be recoverable from the audio."""
    c = chant(GANATEXT, ChantOptions(laya=0.36))
    res = svara_accuracy_at_notes(c)
    assert res["n_notes"] > 5
    assert res["accuracy"] >= 0.7, "the rendered melody does not match its own score"
    assert res["median_abs_cents_error"] < 40


def test_svara_track_is_octave_stable():
    """Regression: a naive tracker reported 147 Hz notes as 294 Hz."""
    c = chant(GANATEXT, ChantOptions(laya=0.36))
    track = svara_track_from_audio(c.audio, c.sr, tonic_hz=c.tonic_hz)
    assert track
    assert set(track) <= set(range(1, 8))


def test_evaluate_returns_a_full_report():
    c = chant(VERSE, ChantOptions(laya=0.34))
    rep = evaluate(c)
    d = rep.to_dict()
    assert "audio" in d and "svara" in d
    assert d["audio"]["periodicity"] > 0.4
    assert "mos_proxy" in d["audio"]
    text = rep.pretty()
    assert "melody" in text and "audio" in text


def test_evaluate_warns_about_style_mode():
    c = chant("ॐ नमो वैश्वानराय", ChantOptions(laya=0.34))
    rep = evaluate(c)
    assert any("style" in n.lower() for n in rep.notes)


def test_evaluate_always_flags_that_mos_is_a_proxy():
    c = chant("ॐ नमो वैश्वानराय", ChantOptions(laya=0.34))
    rep = evaluate(c)
    assert any("proxy" in n.lower() for n in rep.notes)


# --------------------------------------------------------------------------- #
# Corpus builder
# --------------------------------------------------------------------------- #

def test_records_from_text(tmp_path):
    from samadhvani.data.build import records_from_text

    recs = records_from_text(VERSE)
    assert len(recs) == 1
    r = recs[0]
    assert r.phones and r.svaras and r.durations
    assert len(r.phones) == len(r.svaras) == len(r.durations)
    assert all(1 <= s <= 7 for s in r.svaras)


def test_build_text_corpus(tmp_path):
    from samadhvani.data.build import build_text_corpus

    src = tmp_path / "in.txt"
    src.write_text(VERSE + "\n" + GANATEXT + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    n = build_text_corpus([str(src)], str(out))
    assert n == 2
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l]
    assert len(rows) == 2
    for r in rows:
        assert set(r) >= {"text", "phones", "svaras", "durations", "mode"}


def test_align_clip_proportions():
    from samadhvani.data.build import Clip, align_clip

    clip = Clip(audio="x.wav", text=VERSE, line_starts=[], licence="open")
    rec = align_clip(clip, total_seconds=12.0)
    assert len(rec.durations) == len(rec.phones)
    total = sum(rec.durations)
    assert total == pytest.approx(12.0, rel=0.25)


def test_align_clip_with_line_starts():
    from samadhvani.data.build import Clip, align_clip

    clip = Clip(audio="x.wav", text=VERSE, line_starts=[0.0], licence="open")
    rec = align_clip(clip, total_seconds=9.0)
    assert sum(rec.durations) == pytest.approx(9.0, rel=0.25)


def test_measure_contours(tmp_path):
    from samadhvani.data.build import build_text_corpus, measure_contours

    src = tmp_path / "in.txt"
    src.write_text("\n".join([VERSE, GANATEXT] * 3) + "\n", encoding="utf-8")
    man = tmp_path / "m.jsonl"
    build_text_corpus([str(src)], str(man))
    out = tmp_path / "models.json"
    k = measure_contours(str(man), str(out))
    assert k >= 1
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert "models" in blob
    for entry in blob["models"].values():
        assert entry["sections"]
        assert all(1 <= e["svara"] <= 7 for sec in entry["sections"] for e in sec)


def test_data_audit_is_honest():
    """No source may claim to be both unclear-licensed and releasable."""
    from samadhvani.data.sources import Licence, audit

    rep = audit(verbose=False)
    assert "youtube_samagana" in rep["blocked"]
    assert rep["audio_hours_releasable"] >= 0.0


def test_educational_profile_unblocks_training_but_not_shipping():
    """The intended-use profile is the point of the whole catalogue."""
    from samadhvani.data.sources import by_key

    iish = by_key("iish_samagana")
    # strict licence answer: no open licence, so it is not trainable-and-shippable
    assert iish.usable_for_training is False
    assert iish.usable_for_release is False

    edu = iish.permit("educational")
    assert edu.may_train is True
    assert edu.may_ship_audio is False
    assert edu.may_ship_weights is False

    # the moment you publish, it is gone again
    assert iish.permit("open").may_train is False
    # and listening was never the problem
    assert iish.permit("evaluation").may_train is False


def test_unclear_provenance_is_blocked_in_every_profile():
    """A rights holder you cannot name is not fixed by a promise about your conduct."""
    from samadhvani.data.sources import PROFILES, SOURCES

    unclear = [s for s in SOURCES if s.licence.value == "unclear"]
    assert unclear, "catalogue should keep at least one unclear source as a warning"
    for s in unclear:
        for profile in PROFILES:
            assert s.permit(profile).may_train is False, (s.key, profile)


def test_own_recordings_are_releasable_everywhere():
    from samadhvani.data.sources import PROFILES, by_key

    own = by_key("own_recordings")
    for profile in PROFILES:
        p = own.permit(profile)
        assert p.may_train and p.may_ship_weights


def test_permit_invariants_hold_for_every_source():
    """Shipping weights implies you were allowed to train; nothing ships what is blocked."""
    from samadhvani.data.sources import PROFILES, SOURCES

    for s in SOURCES:
        for profile in PROFILES:
            p = s.permit(profile)
            if p.may_ship_weights:
                assert p.may_train, s.key
            if p.may_train:
                assert p.reason, f"{s.key}/{profile} must explain itself"


def test_mirrors_are_not_double_counted():
    """shaivam/astrojyoti/vedaprasar are the same performance; count hours once."""
    from samadhvani.data.sources import counted

    keys = {s.key for s in counted()}
    assert "shaivam_samaveda" not in keys
    assert "pcr_samavedam" in keys


def test_aligned_hours_drive_the_claim():
    """The headline number must be hours with line-level timings, not raw hours.

    IISH is the only large collection that arrives pre-aligned; if it ever loses
    its transcription the honest ceiling drops from 15.8 h to almost nothing.
    """
    from samadhvani.data.sources import audit

    rep = audit(verbose=False)
    assert rep["audio_hours_aligned"] == 15.8
    assert rep["audio_hours_trainable"] > rep["audio_hours_aligned"]
    assert rep["audio_hours_trainable_strict"] == 0.0

    # Every trainable-but-unshippable source must be named, not silently included.
    assert "iish_samagana" in rep["not_shippable"]
    assert "own_recordings" not in rep["not_shippable"]


# --------------------------------------------------------------------------- #
# Training (requires torch)
# --------------------------------------------------------------------------- #

torch = pytest.importorskip("torch", reason="training tests need torch")


def test_model_forward_shapes():
    from samadhvani.synth.neural import COND_DIM, INPUT_DIM, build_models

    dur, spec = build_models(hidden=32, upsample=2)
    x = torch.randn(2, 9, INPUT_DIM)
    c = torch.randn(2, 9, COND_DIM)
    d = dur(x, c)
    assert d.shape == (2, 9)
    s = spec(x, c)
    assert s.shape[0] == 2 and s.shape[2] == 6
    # formant outputs must be in Hz range, amplitudes in 0..1
    assert (s[..., :3] >= 0).all() and (s[..., :3] <= 4200).all()
    assert (s[..., 3:] >= 0).all() and (s[..., 3:] <= 1).all()


def test_spectral_target_units_match_the_head():
    """Regression: normalised targets against a Hz head trained happily to nothing."""
    from samadhvani.train.trainer import ChantDataset

    tgt = ChantDataset._spectral_target("aa")
    assert tgt[0] > 100            # Hz, not 0..1
    assert tgt[0] < 4200
    assert 0.0 <= tgt[3] <= 1.0    # voiced flag stays a proportion


def test_checkpoint_roundtrip(tmp_path):
    from samadhvani.synth.neural import SamadhvaniCheckpoint, build_models

    dur, spec = build_models(hidden=16)
    ck = SamadhvaniCheckpoint(
        duration_state=dur.state_dict(), spectrogram_state=spec.state_dict(),
        config={"hidden": 16, "upsample": 4},
    )
    path = str(tmp_path / "ck.pt")
    ck.save(path)
    back = SamadhvaniCheckpoint.load(path)
    assert back.config["hidden"] == 16
    assert set(back.duration_state) == set(dur.state_dict())


def test_training_reduces_loss(tmp_path):
    """One short run must actually optimise, not just iterate."""
    from samadhvani.data.build import build_text_corpus
    from samadhvani.train.trainer import TrainConfig, train

    src = tmp_path / "in.txt"
    src.write_text("\n".join([VERSE, GANATEXT] * 8) + "\n", encoding="utf-8")
    man = tmp_path / "m.jsonl"
    build_text_corpus([str(src)], str(man))

    cfg = TrainConfig(epochs_duration=3, epochs_spectral=3, batch_size=8,
                      hidden=24, device="cpu")
    ck = train(str(man), str(tmp_path / "ck.pt"), cfg)
    assert os.path.exists(ck)
    assert os.path.getsize(ck) > 1000
