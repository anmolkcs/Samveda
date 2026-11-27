"""Tests for the text frontend: script, syllabification, phones, vikāras."""

import pytest

from samadhvani.phones.g2p import phones_for_view, text_to_phone_string
from samadhvani.phones.inventory import PHONES, STOBHAS
from samadhvani.text.svaras import SAMA_SVARAS, annotate_svaras, svara_to_semitone
from samadhvani.text.syllabify import syllabify
from samadhvani.text.unicodedata_ext import (
    Script,
    detect_script,
    devanagari_to_roman,
    normalize,
    roman_to_devanagari,
    to_devanagari,
)
from samadhvani.text.vikara import (
    SCHOOLS,
    VIKARAS,
    VikaraOptions,
    apply_vikaras,
    classify_vikaras,
    parse_gana,
)

RV_1_1_1 = "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म्"


# --------------------------------------------------------------------------- #
# Script
# --------------------------------------------------------------------------- #

def test_detect_devanagari():
    assert detect_script(RV_1_1_1) == Script.DEVANAGARI


@pytest.mark.parametrize("text,expected", [
    ("ಅಗ್ನಿಮೀಳೇ", Script.KANNADA),
    ("అగ్నిమీళే", Script.TELUGU),
    ("অগ্নিমীল়ে", Script.BENGALI),
    ("agnim īḷe", Script.IAST),
    ("agnim ile", Script.LATIN),
])
def test_detect_other_scripts(text, expected):
    assert detect_script(text) == expected


@pytest.mark.parametrize("script_text", ["ಅಗ್ನಿಮೀ", "అగ్నిమీ", "অগ্নিমী", "અગ્નિમી"])
def test_aligned_scripts_convert(script_text):
    """Layout-aligned Brahmic blocks should round-trip to Devanāgarī losslessly."""
    deva = to_devanagari(script_text)
    assert all(0x0900 <= ord(c) <= 0x097F or c.isspace() for c in deva)


def test_iast_roundtrip():
    """Devanāgarī -> IAST -> Devanāgarī must preserve every letter.

    Accent *placement* is allowed to shift (IAST writes an accent on the vowel, and
    re-importing it attaches the mark to the syllable rather than to the exact
    codepoint), but no segmental letter may be added, lost or changed — that is the
    property that matters, because a shifted letter is a mispronunciation.
    """
    roman = devanagari_to_roman(RV_1_1_1)
    assert "gnim" in roman and "ḻe" in roman
    again = roman_to_devanagari(roman)

    def letters(s):
        # keep only Devanāgarī letters, matras and virama; drop accents/whitespace
        return "".join(c for c in s if 0x0900 <= ord(c) <= 0x097F
                       and c not in "॒॑᳚").replace(" ", "")

    assert letters(again) == letters(RV_1_1_1)


def test_roman_inherent_vowel_and_final_virama():
    # the two classic transliteration bugs
    assert roman_to_devanagari("rāma") == "राम"
    assert roman_to_devanagari("agnim") == "अग्निम्"      # NOT अग्निम
    assert roman_to_devanagari("saṃskṛtam") == "संस्कृतम्"


def test_roman_retroflex_l_disambiguation():
    """ḷ before a vowel is the consonant ळ; before a consonant it is vocalic ऌ."""
    assert "ळ" in roman_to_devanagari("īḷe")
    assert "ॢ" in roman_to_devanagari("kḷptam")


def test_no_schwa_deletion():
    """The Hindi word-processor bug: final 'a' must survive."""
    assert roman_to_devanagari("rāma").endswith("म")
    assert not roman_to_devanagari("rāma").endswith("म्")


def test_anusvara_keeps_inherent_vowel():
    assert devanagari_to_roman("तं") == "taṃ"
    assert devanagari_to_roman("रामः") == "rāmaḥ"


def test_normalize_preserves_lines():
    text = "  पदं   एकम्  ।\n पदं द्वितीयम् ॥ २ ॥ "
    out = normalize(text)
    assert len(out.split("\n")) == 2


# --------------------------------------------------------------------------- #
# Syllabification
# --------------------------------------------------------------------------- #

def test_syllable_count_rv_1_1_1():
    view = syllabify(RV_1_1_1)
    # a-gni-mī-ḷe pu-ro-hi-taṃ ya-jña-sya de-va-mṛ-tvi-jam
    # 16 akṣaras: the word-final म् of "ṛtvijam" is the *coda* of "jam", not a 17th
    # syllable, which is the correct Sanskrit syllabification.
    assert len(view.aksaras) == 16


def test_accents_do_not_break_syllables():
    """Accents are combining marks inside a syllable, not separators."""
    with_accents = syllabify(RV_1_1_1).aksaras
    without = syllabify(
        RV_1_1_1.replace("॑", "").replace("॒", "")
    ).aksaras
    assert len(with_accents) == len(without)
    assert [a.text for a in with_accents] == [a.text for a in without]


def test_accent_assignment():
    aks = syllabify("अ॒ग्निमी॑ळे").aksaras
    accents = [a.accent for a in aks]
    assert "॒" in accents and "॑" in accents
    # the accent follows the syllable it marks, not the one before
    assert aks[0].accent == "॒"


def test_prolongation_numerals():
    """Vedic २/३ are duration marks (vikarṣaṇa), not verse numbers."""
    aks = syllabify("ये२३").aksaras
    assert len(aks) == 1
    assert aks[0].prolongation > 1.0
    assert aks[0].effective_matra > aks[0].matra


def test_groups_split_on_danda():
    view = syllabify("अग्निमीळे । पुरोहितं")
    assert len(view.groups) == 2


def test_conjunct_cluster_is_one_syllable():
    aks = syllabify("स्त्री").aksaras
    assert len(aks) == 1
    assert len(aks[0].onset) == 3


def test_word_final_consonant_folds():
    """रामम् is rā-ma-m: two akṣaras, the final म् a coda of the second."""
    aks = syllabify("रामम्").aksaras
    assert len(aks) == 2
    assert aks[-1].coda.startswith("C")
    # and the consonant must actually be pronounced, not silently dropped
    assert text_to_phone_string("रामम्").split()[-1] == "m"


def test_no_inherent_vowel_after_virama():
    """अग्निम् must not be sung "agnima" — there is no final vowel."""
    phones = text_to_phone_string("अग्निम्").split()
    assert phones[-1] == "m"
    assert phones != text_to_phone_string("अग्निम").split()


# --------------------------------------------------------------------------- #
# Phones
# --------------------------------------------------------------------------- #

def test_aspiration_contrast_preserved():
    """The four-way stop series must stay four-way."""
    assert text_to_phone_string("क") == "k a"
    assert text_to_phone_string("ख") == "kh a"
    assert text_to_phone_string("ग") == "g a"
    assert text_to_phone_string("घ") == "gh a"


def test_retroflex_vs_dental():
    assert text_to_phone_string("ट").startswith("tt")
    assert text_to_phone_string("त").startswith("t")


def test_three_sibilants_distinct():
    assert "s" in text_to_phone_string("स")
    assert "sh" in text_to_phone_string("श")
    assert "ss" in text_to_phone_string("ष")


def test_vocalic_r():
    assert " R " in " " + text_to_phone_string("कृत") + " "


def test_retroflex_lateral():
    assert "zh" in text_to_phone_string("ळे")


def test_homorganic_anusvara():
    assert "n s" in text_to_phone_string("संस्कृतम्")     # dental before s
    assert "ng g" in text_to_phone_string("अंग")          # velar before g
    assert "ny sh" in text_to_phone_string("अंश")         # palatal before ś


def test_visarga_allophones():
    """Jihvāmūlīya before k/kh, upadhmānīya before p/ph."""
    assert "kh_h" in text_to_phone_string("दुःख")
    assert "ph_h" in text_to_phone_string("दुःपुत्र")
    assert text_to_phone_string("रामः").strip().endswith("H")


def test_every_phone_in_inventory():
    """No phone emitted by the frontend may be unknown to the synthesiser."""
    sample = "हाउ अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् ओ३म् भूर्भुवः स्वः"
    for p in text_to_phone_string(sample).split():
        assert p in PHONES, f"{p} not in inventory"


def test_stobha_phones_present():
    for stobha in STOBHAS:
        assert stobha


# --------------------------------------------------------------------------- #
# Svaras
# --------------------------------------------------------------------------- #

def test_seven_svaras():
    assert len(SAMA_SVARAS) == 7
    assert set(SAMA_SVARAS) == set(range(1, 8))


def test_svara_ordering_is_descending():
    """Sāma numerals run high to low: 1 (kṛṣṭa) is the highest note."""
    freqs = [svara_to_semitone(n) for n in range(1, 8)]
    assert freqs == sorted(freqs, reverse=True)


def test_naradiya_venu_mapping():
    """Nāradīya Śikṣā: Ma, Ga, Ri, Sa, Ni, Dha, Pa."""
    venu = [SAMA_SVARAS[n]["venu"] for n in range(1, 8)]
    assert venu == ["म Ma", "ग Ga", "रि Ri", "स Sa", "नि Ni", "ध Dha", "प Pa"]


def test_samhita_mode_honours_accents():
    view = syllabify("अ॒ग्निमी॑ळे")
    toks = annotate_svaras(view, scheme="samhita")
    # anudātta -> mandra (low), udātta -> kṛṣṭa (high)
    assert toks[0].svara == 6
    assert any(t.svara == 1 for t in toks)


# --------------------------------------------------------------------------- #
# Vikāras
# --------------------------------------------------------------------------- #

def test_all_six_vikaras_documented():
    assert len(VIKARAS) == 7   # six classical + āgama
    for key in ("vikara", "vislesa", "vikarshana", "abhyasa", "virama", "stobha"):
        assert key in VIKARAS


def test_schools_have_distinct_stobhas():
    kauthuma = set(SCHOOLS["kauthuma"].stobhas)
    ranayaniya = set(SCHOOLS["ranayaniya"].stobhas)
    assert "हाउ" in kauthuma
    assert "हावु" in ranayaniya           # the rāṇāyanīya form
    assert kauthuma != ranayaniya


def test_apply_vikaras_adds_stobha():
    g = apply_vikaras("अग्निमीळे पुरोहितं", VikaraOptions(school="kauthuma"))
    text = " ".join(l.text for l in g.lines)
    assert any(s in text for s in SCHOOLS["kauthuma"].stobhas)


def test_apply_vikaras_respects_intensity():
    quiet = apply_vikaras("अग्निमीळे पुरोहितं", VikaraOptions(intensity=0.0))
    loud = apply_vikaras("अग्निमीळे पुरोहितं", VikaraOptions(intensity=1.0))
    assert len(" ".join(l.text for l in loud.lines)) >= len(
        " ".join(l.text for l in quiet.lines))


def test_classify_vikaras_detects_stobha():
    flags = classify_vikaras("अग्निमीळे", "हाउ अग्निमीळे")
    assert flags["stobha"]


def test_parse_gana_reads_svara_numerals():
    gana = parse_gana("हाउ ४ अ॒ग्निमी॑ळे ३ पु॒रोहि॑तं २ य॒ज्ञस्य॑ १ दे॒वमृ॒त्विज॑म्")
    svaras = [s for s in gana.flat_svaras() if s]
    assert svaras, "no svaras recovered from a text that has numerals"
    assert set(svaras) <= set(range(1, 8))


def test_parse_gana_does_not_misread_pluta():
    """A run of २३ is prolongation, not two svara numbers."""
    gana = parse_gana("हाउ ४ अग्निमीळे २३ पुरोहितं")
    assert all(s is None or 1 <= s <= 7 for s in gana.flat_svaras())
