"""The phone inventory.

Samadhvani models an *articulatory* state rather than an abstract token, so the
same table can drive two very different acoustic engines:

* the neural model in :mod:`samadhvani.models` — it takes features + svara targets
  as conditioning and predicts mel spectrograms;
* the dependency-free NumPy synthesiser in :mod:`samadhvani.synth` — a source/filter
  cascade whose vowels are built from the classical F1–F3 targets below.

That is why every phone carries ``place``, ``manner``, ``voiced``, ``aspirated`` and
formant targets instead of being an opaque integer.  It also means the fallback
engine is a real formant synthesiser and not a beep machine, so a `pip install`-only
user still hears something chant-like.

Sanskrit's contrasts that Hindi-trained frontends lose, and that we keep:

* the full four-way stop series (voiceless, voiceless aspirate, voiced, voiced aspirate)
  at five places of articulation;
* three sibilants — dental ``s``, retroflex ``ṣ``, palatal ``ś`` — kept acoustically
  distinct rather than collapsed;
* the retroflex series ṭ ṭh ḍ ḍh ṇ, including the retroflex aspirates;
* vocalic ``ṛ`` and ``ṝ``;
* ``ḻ`` (ळ), which occurs in Vedic and is the bane of most Indic TTS;
* visarga with its two sandhi allophones, jihvāmūlīya and upadhmānīya
  (Pāṇini 8.3.37–41), and homorganic anusvāra.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Feature vocabulary
# --------------------------------------------------------------------------- #

PLACES = ("velar", "palatal", "retroflex", "dental", "labial", "glottal", "none")
MANNERS = (
    "stop", "nasal", "fricative", "sibilant", "approximant", "lateral",
    "vowel", "murmur", "pause", "stobha",
)

#: Default male chant-singer formants, F1/F2/F3 in Hz.
#: (Peterson & Barney-style values for a male vocal tract, ~17.5 cm.)
VOWEL_FORMANTS: dict[str, tuple[int, int, int]] = {
    "a": (730, 1090, 2440),
    "aa": (700, 1050, 2400),
    "i": (270, 2290, 3010),
    "ii": (260, 2250, 2990),
    "u": (300, 870, 2240),
    "uu": (290, 850, 2220),
    "R": (490, 1350, 1690),   # syllabic ṛ — retroflex-tinted, lowered F3
    "RR": (480, 1330, 1670),
    "L": (380, 1500, 2400),   # syllabic ḷ
    "e": (530, 1840, 2480),
    "ai": (530, 1840, 2480),
    "o": (570, 840, 2410),
    "au": (570, 840, 2410),
}

#: Diphthong targets: (start vowel, end vowel)
DIPHTHONGS: dict[str, tuple[str, str]] = {
    "ai": ("a", "i"),
    "au": ("a", "u"),
    "e": ("a", "i"),   # Sanskrit e/o are *always* diphthongs (ai/au contracted)
    "o": ("a", "u"),
}

#: F2 locus of each consonant, used to make formant transitions audible.
CONSONANT_LOCUS: dict[str, tuple[int, int]] = {
    #            F2-locus, F3-locus
    "velar": (1900, 2300),
    "palatal": (2200, 2800),
    "retroflex": (1700, 2100),
    "dental": (1800, 2500),
    "labial": (900, 2200),
    "glottal": (1500, 2400),
    "none": (1500, 2400),
}

#: Noise band (low, high) Hz for fricatives and sibilants.
NOISE_BANDS: dict[str, tuple[int, int]] = {
    "s": (4000, 8000),     # dental, high
    "sh": (2200, 5200),    # palatal ś
    "ss": (1700, 4800),    # retroflex ṣ
    "h": (700, 2600),      # glottal
    "kh_h": (1200, 3200),  # velar fricative (jihvāmūlīya)
    "ph_h": (800, 2400),   # bilabial fricative (upadhmānīya)
}


@dataclass(frozen=True)
class Phone:
    """One phone, with both symbolic identity and articulatory detail."""

    symbol: str            # canonical ASCII symbol used in feature matrices
    ipa: str
    kind: str              # vowel | vowel_long | stop | nasal | fricative | ...
    matra: float           # relative duration, 1.0 = a short vowel
    voiced: bool
    aspirated: bool
    place: str
    manner: str
    formants: tuple[int, int, int] = (500, 1500, 2500)
    f0_octave: int = 0     # octave shift relative to Sa (sāma notes go low)
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_vowel(self) -> bool:
        return self.manner == "vowel"

    @property
    def is_obstruent(self) -> bool:
        return self.manner in ("stop", "fricative", "sibilant")


def _v(sym: str, ipa: str, matra: float, *, long: bool = False) -> Phone:
    base = sym.rstrip("0123456789")
    fmt = VOWEL_FORMANTS.get(base[:2]) or VOWEL_FORMANTS.get(base[0]) or (500, 1500, 2500)
    return Phone(
        symbol=sym, ipa=ipa, kind="vowel_long" if long else "vowel", matra=matra,
        voiced=True, aspirated=False, place="none", manner="vowel",
        formants=fmt, tags=("vowel",),
    )


def _c(sym, ipa, place, manner, voiced, aspirated, matra=0.55, **kw) -> Phone:
    return Phone(
        symbol=sym, ipa=ipa, kind=manner, matra=matra, voiced=voiced,
        aspirated=aspirated, place=place, manner=manner,
        formants=(0,) * 3 if manner != "approximant" else VOWEL_FORMANTS["a"],
        **kw,
    )


# --------------------------------------------------------------------------- #
# The inventory
# --------------------------------------------------------------------------- #

PHONES: dict[str, Phone] = {}


def _register(*phones: Phone) -> None:
    for p in phones:
        if p.symbol in PHONES:
            raise ValueError(f"duplicate phone {p.symbol}")
        PHONES[p.symbol] = p


# -- vowels ----------------------------------------------------------------- #
_register(
    _v("a", "ɐ", 1.0),
    _v("aa", "ɑː", 2.0, long=True),
    _v("i", "i", 1.0),
    _v("ii", "iː", 2.0, long=True),
    _v("u", "u", 1.0),
    _v("uu", "uː", 2.0, long=True),
    _v("R", "r̩", 1.0),
    _v("RR", "r̩ː", 2.0, long=True),
    _v("L", "l̩", 1.0),
    _v("e", "eː", 2.0, long=True),  # realisations in DIPHTHONGS
    _v("ai", "ai", 2.0, long=True),
    _v("o", "oː", 2.0, long=True),
    _v("au", "au", 2.0, long=True),
)

# -- stops: five places x four laryngeal settings ---------------------------- #
_STOPS = [
    # place,      T,      Th,     D,      Dh
    ("velar",    ("k", "kʰ"), ("kh", "kʰ"), ("g", "ɡ"), ("gh", "ɡʱ")),
    ("palatal",  ("c", "t͡ɕ"), ("ch", "t͡ɕʰ"), ("j", "d͡ʑ"), ("jh", "d͡ʑʱ")),
    ("retroflex", ("tt", "ʈ"), ("tth", "ʈʰ"), ("dd", "ɖ"), ("ddh", "ɖʱ")),
    ("dental",   ("t", "t̪"), ("th", "t̪ʰ"), ("d", "d̪"), ("dh", "d̪ʱ")),
    ("labial",   ("p", "p"), ("ph", "pʰ"), ("b", "b"), ("bh", "bʱ")),
]
for place, t, th, d, dh in _STOPS:
    _register(
        _c(t[0], t[1], place, "stop", voiced=False, aspirated=False),
        _c(th[0], th[1], place, "stop", voiced=False, aspirated=True),
        _c(d[0], d[1], place, "stop", voiced=True, aspirated=False),
        _c(dh[0], dh[1], place, "stop", voiced=True, aspirated=True),
    )

# -- nasals ------------------------------------------------------------------ #
_register(
    _c("ng", "ŋ", "velar", "nasal", True, False, 0.9),
    _c("ny", "ɲ", "palatal", "nasal", True, False, 0.9),
    _c("nn", "ɳ", "retroflex", "nasal", True, False, 0.9),
    _c("n", "n̪", "dental", "nasal", True, False, 0.9),
    _c("m", "m", "labial", "nasal", True, False, 0.9),
)

# -- sibilants and fricatives ------------------------------------------------ #
_register(
    _c("s", "s̪", "dental", "sibilant", False, False, 0.9),
    _c("sh", "ɕ", "palatal", "sibilant", False, False, 0.9),
    _c("ss", "ʂ", "retroflex", "sibilant", False, False, 0.9),
    _c("h", "ɦ", "glottal", "fricative", True, False, 0.7),
)
# Visarga and its two sandhi allophones (Pāṇini 8.3.37–41).
_register(
    _c("H", "h", "glottal", "fricative", False, False, 0.7, tags=("visarga",)),
    _c("kh_h", "x", "velar", "fricative", False, False, 0.6, tags=("jihvamuliya",)),
    _c("ph_h", "ɸ", "labial", "fricative", False, False, 0.6, tags=("upadhmaniya",)),
    _c("murmur", "ŋ", "none", "murmur", True, False, 0.6, tags=("anusvara",)),
)

# -- approximants and liquids ----------------------------------------------- #
_register(
    _c("y", "j", "palatal", "approximant", True, False, 0.6),
    _c("r", "ɾ", "retroflex", "approximant", True, False, 0.55),
    _c("l", "l", "dental", "approximant", True, False, 0.6),
    _c("v", "ʋ", "labial", "approximant", True, False, 0.6),
    _c("zh", "ɻ", "retroflex", "approximant", True, False, 0.6),
)

# -- prosodic placeholders --------------------------------------------------- #
_register(
    Phone("sil", "", "pause", 0.0, False, False, "none", "pause"),
    Phone("br", "", "pause", 1.5, False, False, "none", "pause", tags=("breath",)),
    Phone("hang", "", "stobha", 1.0, True, False, "none", "stobha", tags=("stobha",)),
)

#: Multi-character sāma stobha syllables, written as one token.  Stobha are not
#: words: they are the meaningless "hau", "ho", "i", "hā'u" syllables that turn a
#: ṛc into a sāman.  Their exact form changes between śākhās, which is precisely
#: what the "recitation school" setting in the UI switches between.
STOBHAS: dict[str, str] = {
    # canonical form -> phonetic realisation (space-separated phone symbols)
    "हाउ": "h aa u",
    "हाउहाउ": "h aa u h aa u",
    "हो": "h o",
    "होवा": "h o v aa",
    "हा": "h aa",
    "हाऊ": "h aa uu",
    "ओ": "o",
    "ओवा": "o v aa",
    "औ": "au",
    "औहोवा": "au h o v aa",
    "उ": "u",
    "इ": "i",
    "ए": "e",
    "यि": "y i",
    "वा": "v aa",
    "ना": "n aa",
    "ता": "t aa",
    "राइ": "r aa i",
    "रायि": "r aa y i",
    "रायी": "r aa y ii",
    "ऊ": "uu",
}


def phone(symbol: str) -> Phone:
    if symbol == "ee":  # alias used by stobha table
        symbol = "e"
    return PHONES[symbol]


def all_phones() -> list[Phone]:
    return list(PHONES.values())


def duration_class(symbol: str) -> float:
    return phone(symbol).matra
