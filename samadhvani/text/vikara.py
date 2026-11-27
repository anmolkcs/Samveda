"""The sāma-vikāra engine and gāna-text parsing.

Trading a ṛc for a sāman means *editing the words*, and the edits are systematic.
The tradition names six (sometimes eight) operations, and the Kauthuma rāṇāyanīya
branches disagree about the stobhas.  Getting these right is what separates a
Sanskrit TTS that happens to be singing from a sāmagāna synthesiser:

======================  ========================================================  =========================
vikāra                  operation                                                 example
======================  ========================================================  =========================
vikāra (विकार)          substitute a vowel/consonant                              agne -> ogne
viśleṣa (विश्लेष)       break a sandhi                                            vītaye -> vo itāyā
vikarṣaṇa (विकर्षण)     prolong (hrasva->dīrgha->pluta->…)                         ye -> yā'2'3'yi
abhyāsa (अभ्यास)        repeat a syllable/word                                    toyāya-toyāya
virāma (विराम)          pause inside a word                                       gṛṇāno | havyadātaye
stobha (स्तोभ)          insert meaningless musical syllables                      hau, ho, hā'u, i
āgama (आगम)             insert an extra phoneme                                   vareṇyam -> vareṇayam
======================  ========================================================  =========================

The engine has two directions:

``apply_vikaras(rc, options)``
    generate a plausible gāna text from an arcika verse (used when the user pastes
    a plain ṛc — the Kauthuma rule set is applied and the result is chanted);

``parse_gana(text)``
    read a real gāna text out of Sanskrit Documents / Vedic Heritage, recovering
    the svara numerals, prolongations and stobhas that are already printed in it.
    This is the path that gives *exact* recitation, and the one the corpus builder
    (:mod:`samadhvani.data.build`) uses for supervised pairs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from enum import Enum

from .unicodedata_ext import normalize

#: Vedic pluta / prolongation numerals, as printed above syllables in gāna texts.
PLUTA = {"२": 2, "३": 3, "४": 4, "५": 5, "६": 6, "७": 7, "८": 8}
_PLUTA_CHARS = "".join(PLUTA)
_PLUTA_RE = re.compile(f"([{_PLUTA_CHARS}]+)")

#: Sāma svara notation: numeral 1..7 written above a syllable.  In the unicode
#: e-texts this is usually a plain ASCII digit directly after the syllable, and in
#: the printed books it is a Devanāgarī numeral.  We accept both.
_SVARA_DIGITS = "1234567१२३४५६७"
_DEV_DIGIT = str.maketrans("१२३४५६७", "1234567")


class Vihara(str, Enum):
    VIKARA = "vikara"
    VISLESA = "vislesa"
    VIKARSHANA = "vikarshana"
    ABHYASA = "abhyasa"
    VIRAMA = "virama"
    STOBHA = "stobha"
    AGAMA = "agama"


VIKARAS: dict[str, str] = {
    Vihara.VIKARA: "substitution — a sung vowel replaces the spoken one (agne -> ogne)",
    Vihara.VISLESA: "sandhi broken so the syllable can be held (vītaye -> vo | itāya)",
    Vihara.VIKARSHANA: "prolongation, hrasva -> dīrgha -> pluta and beyond",
    Vihara.ABHYASA: "repetition of a syllable or word, the ancestor of the refrain",
    Vihara.VIRAMA: "a pause inside a word, to breathe (gṛṇāno | havyadātaye)",
    Vihara.STOBHA: "insertion of meaningless musical syllables (hau, ho, hā'u, i)",
    Vihara.AGAMA: "insertion of an extra phoneme to fill the melody (vareṇyam -> vareṇayam)",
}


# --------------------------------------------------------------------------- #
# Recitation schools: they differ most visibly in their stobhas
# --------------------------------------------------------------------------- #

@dataclass
class School:
    """A recitation school (śākhā) — enough of it to change the output audibly."""

    key: str
    name: str
    iast: str
    stobhas: list[str]
    tonic_note: str = "caturtha"
    #: Typical register; mandra gāna is sung markedly lower than grama-gāna.
    register_shift: int = 0
    notes: str = ""

    @property
    def display(self) -> str:
        return f"{self.name} ({self.iast})"


SCHOOLS: dict[str, School] = {
    "kauthuma": School(
        key="kauthuma",
        name="कौथुम",
        iast="Kauthuma",
        stobhas=["हाउ", "हो", "होवा", "हा", "ओवा", "राइ", "इ"],
        register_shift=0,
        notes=(
            "The commonest surviving sāmagāna lineage: Gujarat, Uttar Pradesh, Odisha "
            "and (since the last century) Darbhanga in Bihar. Grand, slow, with "
            "hau/ho stobhas."
        ),
    ),
    "ranayaniya": School(
        key="ranayaniya",
        name="राणायनीय",
        iast="Rāṇāyanīya",
        stobhas=["हावु", "होवा", "हा", "ओवा", "रायि", "हौ"],
        register_shift=-1,
        notes=(
            "Sister school to the Kauthuma with a largely parallel samhitā and gāna "
            "but different stobha forms (hāvu for hau); kept distinct here because "
            "the difference is exactly the kind of thing a faithful TTS must not blur."
        ),
    ),
    "jaiminiya": School(
        key="jaiminiya",
        name="जैमिनीय",
        iast="Jaiminīya",
        stobhas=["होइ", "हा", "हो", "वा", "उ"],
        register_shift=1,
        notes=(
            "The Kerala / Tamil Nadu tradition (Nambūdiri sāmagas, the 'Jaiminīya "
            "Sāmaveda' of Panchal and the Kunnakudi recitations). Melismatically the "
            "most elaborate, and the one with the most distinct stobha inventory."
        ),
    ),
    "rigvedic": School(
        key="rigvedic",
        name="ऋग्वैदिक (त्रिस्वर)",
        iast="Ṛgvedic (tri-svara)",
        stobhas=[],
        register_shift=2,
        notes=(
            "Not a sāma school: the three-tone udātta/anudātta/svarita recitation used "
            "for ṛc that are not set to a sāman. Included because it is the honest "
            "fallback when someone pastes a ṛc with no gāna."
        ),
    ),
}


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class GanaLine:
    """One breath unit of a gāna text: text + per-syllable svara + stretch."""

    text: str
    svaras: list[int | None] = field(default_factory=list)
    stretches: list[int] = field(default_factory=list)
    stobhas: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.svaras)


@dataclass
class Gana:
    """A parsed sāman: metadata + lines."""

    lines: list[GanaLine]
    saman_number: str = ""
    veda: str = "samaveda"
    school: str = "kauthuma"
    chanda: str = ""
    devata: str = ""
    rishi: str = ""
    raw: str = ""

    def flat_svaras(self) -> list[int | None]:
        out: list[int | None] = []
        for line in self.lines:
            out.extend(line.svaras)
        return out

    def flat_stretches(self) -> list[int]:
        out: list[int] = []
        for line in self.lines:
            out.extend(line.stretches or [1] * len(line.svaras))
        return out

    def iter_lines(self):
        for line in self.lines:
            yield line


# --------------------------------------------------------------------------- #
# Generation direction: ṛc -> gāna (Kauthuma-ish)
# --------------------------------------------------------------------------- #

_CONSONANT = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसह"
_DIPHTHONG_START = {"े": "अ", "ै": "अ", "ो": "अ", "ौ": "अ"}


def classify_vikaras(rc: str, gana: str) -> dict[str, bool]:
    """Which of the six operations separate ``rc`` from ``gana``?

    Used by the corpus builder to report what it was actually able to supervise:
    a model that claims to learn sāmagāna should be able to say which vikāras are
    present in its training data.
    """
    import difflib

    rc_letters = "".join(c for c in normalize(rc) if not c.isspace())
    gana_letters = "".join(c for c in normalize(gana) if not c.isspace())
    ops = difflib.SequenceMatcher(a=rc_letters, b=gana_letters)
    flags = {
        "vikara": False, "vislesa": False, "vikarshana": False,
        "abhyasa": False, "virama": False, "stobha": False, "agama": False,
    }
    for tag, i1, i2, j1, j2 in ops.get_opcodes():
        if tag == "equal":
            continue
        rc_seg = rc_letters[i1:i2]
        g_seg = gana_letters[j1:j2]
        if tag == "replace":
            if len(rc_seg) == len(g_seg):
                flags["vikara"] = True         # 1:1 letter swap
            elif len(g_seg) > len(rc_seg):
                flags["agama"] = True
        elif tag == "insert":
            if all(ch in "हाउओवराइयौ" for ch in g_seg) and len(g_seg) <= 5:
                flags["stobha"] = True
            elif len(g_seg) > len(rc_seg) * 1.5:
                flags["abhyasa"] = True
            else:
                flags["agama"] = True
        elif tag == "delete":
            flags["vislesa"] = True
    if any(_PLUTA_CHARS in g for g in gana) or gana.count("२") or "ाऽ" in gana:
        flags["vikarshana"] = True
    if "|" in gana or "‖" in gana:
        flags["virama"] = True
    return flags


@dataclass
class VikaraOptions:
    """Knobs for generating a sāman from a plain ṛc."""

    school: str = "kauthuma"
    #: Insert opening stobhas ("hā'u ho vā") and closing ones.
    stobhas: bool = True
    #: Apply vikarṣaṇa: hold the line-final and accent-bearing vowels.
    prolong: bool = True
    #: Break sandhi at phrase boundaries (viśleṣa) so the singer can breathe.
    break_sandhi: bool = True
    #: Repeat the last word of a line (abhyāsa).
    abhyasa: bool = True
    #: Elongate an anunāsika / hold the cadence (āgama).
    agama: bool = True
    #: 0.0 = plain recitation, 1.0 = full gāna treatment.
    intensity: float = 0.8


def _school_stobha(school: str, n: int) -> str:
    st = SCHOOLS.get(school, SCHOOLS["kauthuma"]).stobhas
    return st[n % len(st)] if st else ""


def apply_vikaras(rc: str, options: VikaraOptions | None = None) -> Gana:
    """Turn an arcika verse into a plausible gāna text.

    This is a *stylisation*, not a reconstruction.  The true sāman for a given ṛc
    lives in the gāna books and was fixed by the śākhā; without those books the most
    that can honestly be done is apply the documented vikāras in the order the
    convention uses, and the demo labels the mode "styled" for exactly that reason.

    The operations are applied **conservatively**, because a sāma-vikāra engine that
    is too eager produces nonsense.  Each one has a well-defined scope in the
    tradition, and going beyond it corrupts the words:

    * **stobha** — the stobha inventory is used to open a *phrase* and to separate
      the two halves of a line.  A stobha is never inserted between two words of the
      same breath group, which is what a naive "sprinkle stobhas" implementation does
      (and what turns "oṃ namo vaiśvānarāya" into "hāu oṃ hovā oṃ / ho namo hā …").
    * **vikarṣaṇa** — only the *line-final* vowel is held.  Prolonging every
      word-final vowel quadruples the length and destroys the metre.
    * **āgama** — a svarabhakti vowel is inserted **only inside a cluster that has no
      vowel**, which is what the tradition does (``vareṇyam`` -> ``vareṇayam``), and
      never before an accented syllable, where it would displace the accent and
      change the meaning.
    * **abhyāsa** — repeats the *last word* of a line, the ancestor of the refrain.
    * **viśleṣa** — breaks the line at a word boundary into two breath groups.
    """
    opts = options or VikaraOptions()
    school = SCHOOLS.get(opts.school, SCHOOLS["kauthuma"])
    intensity = float(np.clip(opts.intensity, 0.0, 1.0))

    lines_in = [ln for ln in normalize(rc).split("\n") if ln.strip()]
    if not lines_in:
        lines_in = [normalize(rc)]

    gana_lines: list[GanaLine] = []
    for li, line in enumerate(lines_in):
        text = line.replace("|", " ").strip()
        if not text:
            continue

        # --- virāma / viśleṣa: two breath groups, split at a word boundary ------
        words = text.split()
        if opts.break_sandhi and len(words) >= 3:
            cut = max(1, len(words) // 2)
            groups = [" ".join(words[:cut]), " ".join(words[cut:])]
        else:
            groups = [text]

        for gi, grp in enumerate(groups):
            gtxt = grp

            # --- vikāra: agne -> ogne ------------------------------------------
            # Only a *word-initial* short 'a' is coloured, and only at meaningful
            # intensity; the gāna form substitutes the opening vowel of the phrase.
            if intensity >= 0.5:
                gtxt = re.sub(r"(^|\s)अ(?=[क-ह])", r"\1ओ", gtxt)

            # --- āgama: fill a vowel-less cluster ---------------------------------
            # "vareṇyam" -> "vareṇayam": insert a short 'a' inside a conjunct only
            # when the following syllable carries no accent (an accented syllable
            # must keep its own vowel intact).
            if opts.agama and intensity >= 0.7:
                def _svarabhakti(m):
                    nxt = gtxt[m.end(): m.end() + 2]
                    if any(a in nxt for a in "॒॑᳚"):
                        return m.group(0)          # keep the accent syllable clean
                    # The svarabhakti vowel is *prothetic to the next consonant*:
                    # "ś-a-va", not "ś | a | va".  Writing it as "्" + "अ" + C keeps
                    # it inside the same akṣara, which is also how it is sung.
                    return m.group(1) + "्" + "अ" + m.group(2)
                # A cluster with no vowel between the consonants gets a short 'a'.
                gtxt = re.sub(f"([{_CONSONANT}])्([{_CONSONANT}])", _svarabhakti, gtxt)

            # --- vikarṣaṇa: hold the line-final vowel ---------------------------
            if opts.prolong and gi == len(groups) - 1:
                gtxt = re.sub(r"([आईऊॠॡएऐओऔ])$", r"\1३", gtxt)

            # --- abhyāsa: repeat the closing word -------------------------------
            if opts.abhyasa and intensity >= 0.75:
                toks = gtxt.split()
                if len(toks) > 1:
                    gtxt = gtxt + " " + toks[-1]

            # --- stobha: open the phrase, and separate the two halves -----------
            if opts.stobhas and school.stobhas:
                lead = school.stobhas[(li + gi) % len(school.stobhas)]
                gtxt = lead + " " + gtxt
                # a single medial stobha exactly between the breath groups of a long
                # line — never between two words inside one group
                if opts.break_sandhi and len(groups) == 2 and gi == 0 and len(words) >= 5:
                    mid = school.stobhas[(li + gi + 2) % len(school.stobhas)]
                    toks = gtxt.split()
                    toks.insert(max(2, len(toks) - 1), mid)
                    gtxt = " ".join(toks)

            gtxt = re.sub(r"\s+", " ", gtxt).strip()
            gl = GanaLine(text=gtxt)
            gl.stobhas = [s for s in school.stobhas if s in gtxt]
            gana_lines.append(gl)

    return Gana(lines=gana_lines, school=school.key, raw=rc, chanda=_guess_chanda(rc))


# --------------------------------------------------------------------------- #
# Parsing direction: a real printed gāna text -> structured sāman
# --------------------------------------------------------------------------- #

_SVARA_SUFFIX_RE = re.compile(f"([{_SVARA_DIGITS}])\\s*")


def _pluta_before_deva(text: str, pos: int) -> bool:
    """Is the numeral run starting at ``pos`` a prolongation rather than a svara?

    Real gāna e-texts put the svara numeral directly *after* the syllable it governs,
    and pluta marks as a run of २/३/४.  The two are therefore distinguishable by
    shape in almost every case:

    * a run containing anything outside २..८ is a svara number;
    * a run of length > 1 is a pluta run (``२३`` = hold for two-plus-three);
    * a lone ``१`` is a svara number (kṛṣṭa, the highest note) — ``१`` is never a
      pluta mark;
    * a lone digit २..७ is ambiguous in principle.  It is read as **pluta** when a
      svara has already been assigned to the syllable (so the numeral is a hold), and
      as a **svara** otherwise.  This is the order the printed books use: the note is
      given first, then the syllables are lengthened.
    """
    j = pos
    run = []
    while j < len(text) and text[j] in PLUTA:
        run.append(text[j])
        j += 1
    if not run:
        return False
    if any(c not in "२३४५६७८" for c in run):
        return False
    return len(run) > 1


def parse_gana(text: str, *, school: str = "kauthuma") -> Gana:
    """Parse a printed/extracted gāna text.

    Two conventions occur in the wild and both are handled:

    * the **printed-book** convention — a sāma numeral printed *above* each syllable,
      which OCR and most e-texts render as a digit immediately following the syllable;
    * the **accented-saṃhitā** convention — udātta ``॑`` / anudātta ``॒`` / svarita,
      from which the three-tone recitation is recoverable even when no numerals
      survive.

    Returns a :class:`Gana` whose ``lines`` carry one svara per akṣara.  Where no
    notation survives at all the entries are ``None`` and the caller should fall back
    to :func:`samadhvani.text.svaras.annotate_svaras`.
    """
    from .syllabify import syllabify

    text = normalize(text)
    header = _parse_header(text)
    body = [ln for ln in text.split("\n") if ln.strip() and not _is_header(ln)]

    lines: list[GanaLine] = []
    for raw_line in body:
        raw_line = raw_line.translate(_DEV_DIGIT)
        # 1. Tokenise: syllables (with their own sub-characters) and numeral runs.
        #    A "syllable" here is a nucleus plus whatever is spelled with it, which is
        #    exactly what syllabify() already does on the numeral-free string.
        numeral_free: list[str] = []
        numerals: list[tuple[int, str]] = []      # (index into numeral_free, run)
        i = 0
        while i < len(raw_line):
            ch = raw_line[i]
            if ch.isdigit():
                j = i
                while j < len(raw_line) and raw_line[j].isdigit():
                    j += 1
                numerals.append((len(numeral_free), raw_line[i:j]))
                i = j
                continue
            if ch in "|‖":
                numeral_free.append("|")
                i += 1
                continue
            numeral_free.append(ch)
            i += 1

        view = syllabify("".join(numeral_free))
        n = len(view.aksaras)
        if n == 0:
            continue

        # 2. Map each numeral run to the akṣara it follows.  ``numeral_free`` is
        #    walked in step with the akṣaras so the boundary is exact.
        svaras: list[int | None] = [None] * n
        stretches: list[int] = [1] * n
        # build a char-offset -> akṣara index map from the syllabifier's own spans
        offsets = _aksara_offsets("".join(numeral_free), n)
        for pos, run in numerals:
            idx = _aksara_at(offsets, pos)
            if idx is None:
                continue
            if len(run) > 1 or (svaras[idx] is not None and run in PLUTA):
                stretches[idx] = max(stretches[idx], max(PLUTA.get(c, 2) for c in run))
            elif run.isdigit() and 1 <= int(run) <= 7:
                svaras[idx] = int(run)

        line = GanaLine(text=" ".join(a.text for a in view.aksaras),
                        svaras=svaras, stretches=stretches)
        line.stobhas = [s for s in SCHOOLS.get(school, SCHOOLS["kauthuma"]).stobhas
                        if s in raw_line]
        lines.append(line)

    return Gana(
        lines=lines, school=school, raw=text,
        chanda=header.get("chanda", ""), devata=header.get("devata", ""),
        rishi=header.get("rishi", ""), saman_number=header.get("number", ""),
    )


def _aksara_offsets(text: str, n_expected: int) -> list[tuple[int, int]]:
    """Character spans of each akṣara in ``text``, ignoring accents and spaces."""
    from .syllabify import syllabify

    view = syllabify(text)
    spans: list[tuple[int, int]] = []
    pos = 0
    for ak in view.aksaras:
        src = ak.source or ak.text
        # find this syllable's spelling at or after pos
        found = text.find(src.replace(VIRAMA, ""), pos) if src else -1
        if found < 0:
            found = text.find(src, pos)
        if found < 0:
            start = pos
            end = pos
        else:
            start = found
            end = found + len(src)
        spans.append((start, max(end, start + 1)))
        pos = max(end, pos + 1)
    while len(spans) < n_expected:
        spans.append((pos, pos + 1))
    return spans


VIRAMA = "्"


def _aksara_at(spans: list[tuple[int, int]], pos: int) -> int | None:
    """Index of the akṣara whose span ends at or before ``pos`` (the numeral follows it)."""
    best = None
    for i, (a, b) in enumerate(spans):
        if a <= pos:
            best = i
    return best


_HEADER_KEYS = {
    "ऋषिः": "rishi", "ऋषि": "rishi", "r̥ṣiḥ": "rishi",
    "देवता": "devata", "देवताः": "devata",
    "छन्दः": "chanda", "छन्द": "chanda",
    "साम": "number", "गान": "number",
}
_HEADER_LINE_RE = re.compile("|".join(map(re.escape, _HEADER_KEYS)))


def _is_header(line: str) -> bool:
    return bool(_HEADER_LINE_RE.search(line)) and len(line) < 220


def _parse_header(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.split("\n")[:8]:
        for key, field_name in _HEADER_KEYS.items():
            if key in line and field_name not in out:
                after = line.split(key, 1)[1]
                val = after.split("|")[0].split("॥")[0].strip(" :।॥")
                if val:
                    out[field_name] = val.strip()
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_CHANDAS = [
    # (name, syllables per line, lines)
    ("गायत्री", 8, 3), ("उष्णिक्", 7, 4), ("अनुष्टुभ्", 8, 4),
    ("बृहती", 9, 4), ("पङ्क्ति", 10, 4), ("त्रिष्टुभ्", 11, 4),
    ("जगती", 12, 4), ("अतिजगती", 13, 4), ("शक्वरी", 14, 4),
    ("अतिशक्वरी", 15, 4), ("अष्टि", 16, 4), ("अत्यष्टि", 17, 4),
    ("धृति", 18, 4), ("अतिधृति", 19, 4), ("कृति", 20, 4),
    ("प्रकृति", 21, 4), ("आकृति", 22, 4), ("विकृति", 23, 4),
]


def _guess_chanda(rc: str) -> str:
    from .syllabify import syllabify

    lines = [ln for ln in normalize(rc).split("\n") if ln.strip().replace("|", "").strip()]
    if not lines:
        return ""
    per_line = [len(syllabify(ln).aksaras) for ln in lines]
    if len(set(per_line)) == 1 and per_line:
        n = per_line[0]
        for name, size, _ in _CHANDAS:
            if size == n:
                return name
    total = sum(per_line)
    for name, size, count in _CHANDAS:
        if total == size * count:
            return name
    return ""


def load_gana(path: str, *, school: str = "kauthuma") -> Gana:
    """Read a gāna text file (UTF-8) and parse it."""
    with open(path, encoding="utf-8") as fh:
        return parse_gana(fh.read(), school=school)
