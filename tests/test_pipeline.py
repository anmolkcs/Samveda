"""End-to-end tests: mode selection, honesty of the output, and the API contract."""

import numpy as np
import pytest

from samadhvani.pipeline import Chant, ChantOptions, chant
from samadhvani.text.vikara import SCHOOLS

GANATEXT = "हाउ ४ अ॒ग्निमी॑ळे ३ पु॒रोहि॑तं २ य॒ज्ञस्य॑ १ दे॒वमृ॒त्विज॑म्"
SAMHITA = "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म्"
PLAIN = "ॐ नमो वैश्वानराय"


# --------------------------------------------------------------------------- #
# Mode selection — the central honesty guarantee
# --------------------------------------------------------------------------- #

def test_gana_text_selects_gana_mode():
    c = chant(GANATEXT)
    assert c.mode == "gana"


def test_accented_text_selects_samhita_mode():
    c = chant(SAMHITA)
    assert c.mode == "samhita"


def test_plain_text_selects_style_mode():
    c = chant(PLAIN)
    assert c.mode == "style"


def test_style_mode_always_warns_that_it_is_not_the_real_saman():
    """The most important message the system has: styled is not authoritative."""
    c = chant(PLAIN)
    joined = " ".join(c.caveats).lower()
    assert "styled" in joined or "style" in joined
    assert "not the s" in joined or "not the sāman" in joined


def test_samhita_mode_says_it_is_recitation_not_gana():
    c = chant(SAMHITA)
    joined = " ".join(c.caveats).lower()
    assert "recitation" in joined


def test_gana_mode_does_not_claim_purity_beyond_what_it_reads():
    c = chant(GANATEXT)
    assert any("nothing was invented" in x.lower() for x in c.caveats)


def test_explicit_mode_overrides_detection():
    c = chant(PLAIN, ChantOptions(mode="style"))
    assert c.mode == "style"


def test_unknown_mode_still_renders():
    # 'auto' and 'avoid' both fall through to detection
    c = chant(PLAIN, ChantOptions(mode="avoid"))
    assert c.mode in ("style", "samhita", "gana")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def test_chant_returns_playable_audio():
    c = chant(PLAIN, ChantOptions(laya=0.34))
    assert isinstance(c, Chant)
    assert c.sr == 22050
    assert len(c.audio) > c.sr * 2
    assert np.abs(c.audio).max() <= 1.0
    assert np.all(np.isfinite(c.audio))


def test_metadata_is_complete():
    c = chant(SAMHITA)
    d = c.to_dict()
    assert len(d["syllables"]) == len(d["svaras"])
    assert len(d["notes"]) == len(d["syllables"])
    assert all(1 <= s <= 7 for s in d["svaras"])
    assert d["duration"] > 0
    assert d["school"] == "kauthuma"


def test_tonic_is_exposed_and_correct():
    c = chant(PLAIN, ChantOptions(tonic_hz=196.0))
    assert c.tonic_hz == pytest.approx(196.0, rel=0.01)
    # the note on svara 4 must actually be the tonic
    for n in c.notes:
        if n.svara == 4:
            assert n.f0_hz == pytest.approx(c.tonic_hz, rel=0.02)
            break


def test_transpose_shifts_every_note():
    a = chant(PLAIN, ChantOptions(transpose=0.0))
    b = chant(PLAIN, ChantOptions(transpose=5.0))
    ratio = b.notes[0].f0_hz / a.notes[0].f0_hz
    assert ratio == pytest.approx(2 ** (5 / 12), rel=0.01)


def test_laya_changes_duration():
    fast = chant(PLAIN, ChantOptions(laya=0.24))
    slow = chant(PLAIN, ChantOptions(laya=0.62))
    assert slow.duration > fast.duration * 1.6


@pytest.mark.parametrize("script_text", [
    "ಅಗ್ನಿಮೀಳೇ ಪುರೋಹಿತಂ",             # Kannada
    "అగ్నిమీళే పురోహితం",              # Telugu
    "agnim īḷe purohitaṃ",             # IAST
])
def test_all_input_scripts_render(script_text):
    c = chant(script_text)
    assert len(c.syllables) > 3
    assert c.duration > 1.0


def test_script_conversion_is_reported():
    c = chant("ಅಗ್ನಿಮೀಳೇ")
    assert any("kannada" in x.lower() for x in c.caveats)


def test_school_changes_register_or_stobhas():
    a = chant(PLAIN, ChantOptions(school="kauthuma", mode="style"))
    b = chant(PLAIN, ChantOptions(school="jaiminiya", mode="style"))
    changed = (a.audio.shape != b.audio.shape) or not np.allclose(
        a.audio[: min(len(a.audio), len(b.audio))],
        b.audio[: min(len(a.audio), len(b.audio))], atol=1e-3)
    assert changed


def test_model_changes_register():
    a = chant(PLAIN, ChantOptions(model="gramageya"))
    b = chant(PLAIN, ChantOptions(model="aranyaka"))
    assert b.notes[0].f0_hz < a.notes[0].f0_hz


def test_empty_input_raises():
    with pytest.raises(ValueError):
        chant("   ")


def test_non_sanskrit_input_is_rejected_not_babble():
    with pytest.raises(ValueError):
        chant("hello world this is english")


def test_unknown_school_raises():
    with pytest.raises(ValueError):
        chant(PLAIN, ChantOptions(school="nope"))


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        chant(PLAIN, ChantOptions(model="nope"))


def test_render_is_fast_enough_for_a_demo():
    import time
    t0 = time.time()
    chant(PLAIN, ChantOptions(laya=0.3))
    elapsed = time.time() - t0
    assert elapsed < 10.0, f"render took {elapsed:.1f}s; the demo would feel broken"


def test_neural_without_checkpoint_falls_back_gracefully():
    c = chant(PLAIN, ChantOptions(engine="neural", checkpoint=None, laya=0.3))
    assert len(c.audio) > 0
    assert any("fell back" in x.lower() for x in c.caveats)


# --------------------------------------------------------------------------- #
# School metadata
# --------------------------------------------------------------------------- #

def test_all_schools_have_metadata():
    for key, school in SCHOOLS.items():
        assert school.name
        assert school.iast
        assert school.notes


def test_jaiminiya_and_kauthuma_stobhas_differ():
    assert set(SCHOOLS["kauthuma"].stobhas) != set(SCHOOLS["jaiminiya"].stobhas)
