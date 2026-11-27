"""Devanāgarī -> phone symbols.

The mapping is deliberately *lossless in the places that matter*: aspiration,
retroflexion, sibilant identity and vocalic liquids all survive.  Two context
rules do real work:

**Homorganic anusvāra** (Pāṇini 8.4.58–59).  ``ṃ`` assimilates to the class of the
following consonant: ``saṃskṛta`` -> ``s a n s ...``, ``saṃgha`` -> ``s a ng gh a``.
Before a sibilant or ``h`` it stays a true nasal murmur, and before ``r/l/v/ś/ṣ/s``
the tradition varies.  We pick the nasal of the *manner place* of the next
consonant, which is what a careful reciter does.

**Visarga allophones** (Pāṇini 8.3.37–41).  Before ``k/kh/p/ph`` the visarga is not
a plain breath: ``k``/``kh`` trigger the *jihvāmūlīya* (velar fricative ``x``), and
``p``/``ph`` the *upadhmānīya* (bilabial fricative ``ɸ``).  These are preserved in
Sāma and Ṛg recitation and are among the details a general Sanskrit TTS flattens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..text.syllabify import Aksara, syllabify
from ..text.unicodedata_ext import normalize, to_devanagari
from .inventory import PHONES, STOBHAS, Phone, phone

# --------------------------------------------------------------------------- #
# Devanāgarī consonant -> phone symbol
# --------------------------------------------------------------------------- #

CONSONANT_PHONE: dict[str, str] = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "tt", "ठ": "tth", "ड": "dd", "ढ": "ddh", "ण": "nn",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "ळ": "zh", "व": "v",
    "श": "sh", "ष": "ss", "स": "s", "ह": "h",
}

#: Roman vowel symbol (as produced by the syllabifier) -> phone symbol.
VOWEL_PHONE: dict[str, str] = {
    "a": "a", "ā": "aa", "i": "i", "ī": "ii", "u": "u", "ū": "uu",
    "ṛ": "R", "ṝ": "RR", "ḷ": "L", "ḹ": "L", "e": "e", "ai": "ai",
    "o": "o", "au": "au", "ê": "e", "ô": "o", "": "",           # bare consonant
}

#: Place of articulation of each Devanāgarī consonant, for anusvāra assimilation.
_CONSONANT_CLASS: dict[str, str] = {}
for _row, _place in zip(
    ("कखगघङ", "चछजझञ", "टठडढण", "तथदधन", "पफबभम"),
    ("velar", "palatal", "retroflex", "dental", "labial"),
):
    for _c in _row:
        _CONSONANT_CLASS[_c] = _place
for _c in "य":
    _CONSONANT_CLASS[_c] = "palatal"
for _c in "रष":
    _CONSONANT_CLASS[_c] = "retroflex"
for _c in "लस":
    _CONSONANT_CLASS[_c] = "dental"
for _c in "व":
    _CONSONANT_CLASS[_c] = "labial"
for _c in "श":
    _CONSONANT_CLASS[_c] = "palatal"
for _c in "ह":
    _CONSONANT_CLASS[_c] = "glottal"

_CLASS_NASAL = {
    "velar": "ng", "palatal": "ny", "retroflex": "nn",
    "dental": "n", "labial": "m", "glottal": "murmur",
}


def homorganic_anusvara(next_consonant: str, *, strict: bool = True) -> str:
    """Nasal that replaces ``ṃ`` before ``next_consonant`` (a Devanāgarī letter)."""
    place = _CONSONANT_CLASS.get(next_consonant)
    if place is None or place == "glottal":
        return "murmur"
    if not strict:
        return "m"
    return _CLASS_NASAL.get(place, "murmur")


def resolve_visarga(next_consonant: str) -> str:
    """The visarga allophone before ``next_consonant``.

    ``ḥ`` + k/kh -> jihvāmūlīya, + p/ph -> upadhmānīya, else plain visarga.
    """
    if next_consonant in ("क", "ख"):
        return "kh_h"
    if next_consonant in ("प", "फ"):
        return "ph_h"
    return "H"


# --------------------------------------------------------------------------- #
# Akṣara -> phones
# --------------------------------------------------------------------------- #

@dataclass
class AksaraPhones:
    """The phone sequence of one akṣara, with its svara/mātrā metadata."""

    phones: list[str] = field(default_factory=list)
    nucleus: str = ""
    text: str = ""
    matra: float = 1.0
    svara: int | None = None
    stretch: float = 1.0
    is_stobha: bool = False

    def __len__(self) -> int:
        return len(self.phones)


def aksara_to_phones(
    ak: Aksara,
    *,
    next_char: str = "",
    prev_ak: Aksara | None = None,
    strict_anusvara: bool = True,
) -> AksaraPhones:
    """Convert one :class:`Aksara` to phone symbols."""
    symbols: list[str] = []

    # Stobha syllables are looked up whole: "हाउ" is not /h/+/ā/+/u/ run together,
    # it is a single melismatic unit that the model treats as one nucleus.
    if ak.source in STOBHAS:
        return AksaraPhones(
            phones=STOBHAS[ak.source].split(), nucleus="stobha", text=ak.source,
            matra=ak.effective_matra, svara=ak.svara, stretch=ak.prolongation,
            is_stobha=True,
        )

    for c in ak.onset:
        if c in "़":
            continue
        # ऴ (Vedic zh) is not in the consonant table; map conservatively.
        sym = CONSONANT_PHONE.get(c)
        if sym:
            symbols.append(sym)

    nuc = ak.nucleus
    if nuc:
        sym = VOWEL_PHONE.get(nuc, "")
        if sym:
            symbols.append(sym)

    # A folded virama-final consonant ("रामम्" -> रा + मम्) is a CodaConsonant and is
    # emitted as a real (unreleased) consonant, not dropped: the difference between
    # virāma and an inherent 'a' is phonemic in Vedic recitation.
    if ak.coda.startswith("C"):
        cons = ak.coda[1:]
        if cons in CONSONANT_PHONE:
            symbols.append(CONSONANT_PHONE[cons])

    # Coda: anusvāra assimilates, visarga takes its allophone.
    if "ṃ" in ak.coda:
        symbols.append(homorganic_anusvara(next_char, strict=strict_anusvara))
    if "ḥ" in ak.coda:
        symbols.append(resolve_visarga(next_char))

    # A syllable that is only a consonant (virama, word-final) still needs a phone.
    if not symbols and ak.onset:
        symbols = [CONSONANT_PHONE.get(c, "") for c in ak.onset if c in CONSONANT_PHONE]

    return AksaraPhones(
        phones=[s for s in symbols if s],
        nucleus=nuc, text=ak.text, matra=ak.effective_matra,
        svara=ak.svara, stretch=max(1.0, ak.prolongation),
    )


def _next_consonant_of(aksaras: list[Aksara], i: int) -> str:
    """First consonant *after* the coda of akṣara ``i``.

    A coda (ṃ/ḥ) assimilates to whatever consonant actually follows it, which may
    be at the start of the next akṣara's onset cluster (``saṃskṛta`` needs ``s``,
    the *last* member of the cluster's first position) or, for a word-final coda
    at the end of a line, nothing at all.
    """
    ak = aksaras[i]
    if not ("ṃ" in ak.coda or "ḥ" in ak.coda):
        return ""
    if i + 1 >= len(aksaras):
        return ""
    nxt = aksaras[i + 1]
    return nxt.onset[0] if nxt.onset else ""


def text_to_phone_string(text: str, *, school: str | None = None, roman: bool = False) -> str:
    """Full pipeline as a flat phone string, e.g. for TTS checkpoints or corpora."""
    dev = to_devanagari(text)
    view = syllabify(dev)
    out: list[str] = []
    for i, ak in enumerate(view.aksaras):
        nxt = _next_consonant_of(view.aksaras, i)
        ap = aksara_to_phones(ak, next_char=nxt)
        out.extend(ap.phones)
        if roman:
            out.append("|")
    if roman:
        return " ".join(p for p in out if p != "|")
    return " ".join(out)


def phones_for_view(view, *, strict_anusvara: bool = True) -> list[AksaraPhones]:
    """Convert a whole :class:`SyllableView`."""
    out: list[AksaraPhones] = []
    for i, ak in enumerate(view.aksaras):
        nxt = _next_consonant_of(view.aksaras, i)
        out.append(aksara_to_phones(ak, next_char=nxt, strict_anusvara=strict_anusvara))
    return out
