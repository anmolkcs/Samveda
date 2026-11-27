"""Script detection and lossless-where-possible transliteration.

Everything in Samadhvani is normalised to Devanāgarī internally, then mapped to
phones.  Devanāgarī is the *lingua franca* of the Vedic e-text world (the National
Mission for Manuscripts, Vedic Heritage Portal and GRETIL all ship it), and every
other Indic script shares a near-identical codepoint layout with it, so script
conversion is mostly an offset.

Two things are deliberately *not* done here, and both matter for Sanskrit:

* No schwa deletion.  Hindi word-processors silently drop word-final ``a``; that
  turns ``rāma`` into ``rām`` and ``gṛṇāno havyadātaye`` into nonsense.  Our phone
  mapper has an explicit ``schwa_deletion`` policy that defaults to Sanskrit (none).
  This is the same failure Vāgdhenu works around by routing through Kannada
  orthography; because we own the frontend we can simply not make the mistake.

* No sandhi *resolution*.  Sandhi is left alone for the acoustic model; the
  syllabifier only needs to know where the nuclei are.
"""

from __future__ import annotations

import re
import unicodedata
from enum import Enum

# --------------------------------------------------------------------------- #
# Scripts
# --------------------------------------------------------------------------- #


class Script(str, Enum):
    DEVANAGARI = "devanagari"
    BENGALI = "bengali"
    GURMUKHI = "gurmukhi"
    GUJARATI = "gujarati"
    ORIYA = "oriya"
    TAMIL = "tamil"
    TELUGU = "telugu"
    KANNADA = "kannada"
    MALAYALAM = "malayalam"
    IAST = "iast"
    LATIN = "latin"
    UNKNOWN = "unknown"


#: Unicode blocks that share Devanāgarī's internal layout, with the block start.
#: (Tamil is *not* here: it is lossy — no aspirates, no voiced/voiceless contrast.)
_ALIGNED_BLOCKS: dict[Script, int] = {
    Script.BENGALI: 0x0980,
    Script.GURMUKHI: 0x0A00,
    Script.GUJARATI: 0x0A80,
    Script.ORIYA: 0x0B00,
    Script.TELUGU: 0x0C00,
    Script.KANNADA: 0x0C80,
    Script.MALAYALAM: 0x0D00,
}

_DEVANAGARI_START = 0x0900
VIRAMA_DEV = "्"
_CONSONANTS_DEVA = set("कखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसह")
_MATRA_DEVA = set("ािीुूृॄॢॣेैोौॅॉ")
_VOWEL_SIGNS_DEVA = _MATRA_DEVA
_BLOCK_SPAN = 0x80

#: Kannada has a pre-composed ڂ.. no: it has *extra* letters we must fold onto
#: Devanāgarī (e.g. Kannada ೞ ḻa, which has no Devanāgarī counterpart).
_SCRIPT_DROPS: dict[Script, str] = {
    Script.GURMUKHI: "ਖਘਛਝਠਢਥਧਫਭਸ਼",
    Script.TAMIL: "ஃ",
}


def detect_script(text: str) -> Script:
    """Return the dominant script of ``text``.

    Counts codepoints per block rather than looking at the first character, so a
    verse quoted in Devanāgarī with an English gloss is still Devanāgarī.
    """
    text = unicodedata.normalize("NFC", text)
    counts: dict[Script, int] = {}
    latin = 0
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            counts[Script.DEVANAGARI] = counts.get(Script.DEVANAGARI, 0) + 1
            continue
        for script, start in _ALIGNED_BLOCKS.items():
            if start <= cp < start + _BLOCK_SPAN:
                counts[script] = counts.get(script, 0) + 1
                break
        else:
            if 0x0B80 <= cp <= 0x0BFF:
                counts[Script.TAMIL] = counts.get(Script.TAMIL, 0) + 1
            elif cp < 0x0250 and ch.isalpha():
                latin += 1
    if not counts and not latin:
        return Script.UNKNOWN
    if not counts:
        # Latin: IAST if it carries Indic diacritics, else plain Latin.
        iast_chars = set("āīūṛṝḷḹēōṃḥṅñṭḍṇśṣḻ")
        return Script.IAST if any(ch in iast_chars for ch in text) else Script.LATIN
    return max(counts, key=lambda s: counts[s])


# --------------------------------------------------------------------------- #
# Devanāgarī <-> IAST
# --------------------------------------------------------------------------- #

#: Pre-composed Devanāgarī syllables are unnecessary: we work letter-by-letter.
_DEV_TO_IAST: dict[str, str] = {
    # independent vowels
    "अ": "a", "आ": "ā", "इ": "i", "ई": "ī", "उ": "u", "ऊ": "ū",
    "ऋ": "ṛ", "ॠ": "ṝ", "ऌ": "ḷ", "ॡ": "ḹ", "ए": "e", "ऐ": "ai",
    "ओ": "o", "औ": "au", "ऍ": "ê", "ऑ": "ô",
    # consonants
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ṅ",
    "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ñ",
    "ट": "ṭ", "ठ": "ṭh", "ड": "ḍ", "ढ": "ḍh", "ण": "ṇ",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "ळ": "ḻ", "व": "v",
    "श": "ś", "ष": "ṣ", "स": "s", "ह": "h",
    # matras
    "ा": "ā", "ि": "i", "ी": "ī", "ु": "u", "ू": "ū",
    "ृ": "ṛ", "ॄ": "ṝ", "ॢ": "ḷ", "ॣ": "ḹ", "े": "e", "ै": "ai",
    "ो": "o", "ौ": "au", "ॅ": "ê", "ॉ": "ô",
    # signs
    "ं": "ṃ", "ँ": "m̐", "ः": "ḥ", "्": "", "ऽ": "’",
    "ॐ": "oṃ", "।": "|", "॥": "‖",
    # Vedic accents
    "॑": "\u0301", "॒": "\u0331", "᳚": "\u0301", "॓": "\u0300", "॔": "\u0301",
}

_IAST_TO_DEV: dict[str, str] = {}
for _d, _i in _DEV_TO_IAST.items():
    _IAST_TO_DEV.setdefault(_i, _d)
# Reverse matras must win for vowel letters used *after* a consonant.
_MATRA_FROM_IAST: dict[str, str] = {
    "ā": "ा", "i": "ि", "ī": "ी", "u": "ु", "ū": "ू",
    "ṛ": "ृ", "ṝ": "ॄ", "ḷ": "ॢ", "ḹ": "ॣ", "e": "े", "ai": "ै",
    "o": "ो", "au": "ौ", "ê": "ॅ", "ô": "ॉ",
}
_VOWEL_FROM_IAST: dict[str, str] = {
    "a": "अ", "ā": "आ", "i": "इ", "ī": "ई", "u": "उ", "ū": "ऊ",
    "ṛ": "ऋ", "ṝ": "ॠ", "ḷ": "ऌ", "ḹ": "ॡ", "e": "ए", "ai": "ऐ",
    "o": "ओ", "au": "औ", "ê": "ऍ", "ô": "ऑ",
}
_CONS_FROM_IAST: dict[str, str] = {
    v: k for k, v in _DEV_TO_IAST.items() if k not in _VOWEL_FROM_IAST.values() and k not in "ांँः्ऽॐ।॥॒॑᳚॓॔" and k not in "ािीुूृॄॢॣेैोौॅॉ"
}
_CONS_FROM_IAST.update({
    "k": "क", "kh": "ख", "g": "ग", "gh": "घ", "ṅ": "ङ",
    "c": "च", "ch": "छ", "j": "ज", "jh": "झ", "ñ": "ञ",
    "ṭ": "ट", "ṭh": "ठ", "ḍ": "ड", "ḍh": "ढ", "ṇ": "ण",
    "t": "त", "th": "थ", "d": "द", "dh": "ध", "n": "न",
    "p": "प", "ph": "फ", "b": "ब", "bh": "भ", "m": "म",
    "y": "य", "r": "र", "l": "ल", "ḻ": "ळ", "v": "व",
    "ś": "श", "ṣ": "ष", "s": "स", "h": "ह", "ḻ": "ळ",
})

#: Vedic om, which survives IAST as plain "oṃ" and must not stay Latin.
_SPECIAL_IAST = {"oṃ": "ॐ", "om": "ॐ"}

# Longest-first matching keeps ``kh`` from being read as ``k`` + ``h``.
_IAST_KEYS = sorted(
    set(list(_CONS_FROM_IAST) + list(_VOWEL_FROM_IAST) + list(_MATRA_FROM_IAST)
        + ["ṃ", "m̐", "ḥ", "'", "’", "|", "‖"] + list(_SPECIAL_IAST)),
    key=len,
    reverse=True,
)


def devanagari_to_roman(text: str, scheme: str = "iast") -> str:
    """Transliterate Devanāgarī to IAST (or plain ASCII via ``scheme='ascii'``)."""
    out: list[str] = []
    text = unicodedata.normalize("NFC", text)
    for idx, ch in enumerate(text):
        out.append(_DEV_TO_IAST.get(ch, ch))
        # Devanāgarī writes the inherent vowel only *implicitly*: क is "ka".
        # Emit it unless a matra, virama, or another consonant follows.
        if ch in _CONSONANTS_DEVA:
            nxt = text[idx + 1: idx + 2]
            # Only a matra, the virama, or another combining mark cancels the
            # inherent vowel.  Anusvāra and visarga *follow* a full syllable, so
            # "तं" is "taṃ", not "tṃ".
            if nxt not in _VOWEL_SIGNS_DEVA and nxt != "्":
                out.append("a")
    s = "".join(out)
    s = s.replace("’", "'")
    if scheme == "ascii":
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = (s.replace("L", "l").replace("R", "r"))
    return s


#: Combining marks, as they appear after NFD.  IAST's diacritics *are* combining
#: marks, so a transliterator that works on precomposed characters breaks the moment
#: it receives decomposed input — and "yá" decomposes to "y" + U+0301, which is
#: indistinguishable from the Vedic udātta accent until you look at the base letter.
_MACRON = "\u0304"
_ACUTE = "\u0301"
_TILDE = "\u0303"
_DOT_ABOVE = "\u0307"
_DOT_BELOW = "\u0323"
_RING_BELOW = "\u0325"
_CIRCUMFLEX = "\u0302"
_MACRON_BELOW = "\u0331"
_GRAVE = "\u0300"

#: base + combining marks -> the IAST letter they spell
_CLUSTER_TO_IAST: dict[str, str] = {
    "a" + _MACRON: "ā", "i" + _MACRON: "ī", "u" + _MACRON: "ū",
    "r" + _DOT_BELOW: "ṛ", "r" + _DOT_BELOW + _MACRON: "ṝ",
    "l" + _DOT_BELOW: "ḷ", "l" + _DOT_BELOW + _MACRON: "ḹ",
    # The retroflex lateral ḻ (ऴ/ळ) is canonical-decomposed as l + LINE BELOW
    # (U+0331), not l + dot-below — which is why "īḷe" and "īḻe" are different
    # letters and only the second is the Vedic ळ.
    "l" + _MACRON_BELOW: "ḻ",
    "l" + _DOT_BELOW + _MACRON_BELOW: "ḻ",
    "l" + _MACRON_BELOW + _DOT_BELOW: "ḻ",
    "m" + _DOT_BELOW: "ṃ", "h" + _DOT_BELOW: "ḥ",
    "t" + _DOT_BELOW: "ṭ", "d" + _DOT_BELOW: "ḍ", "n" + _DOT_BELOW: "ṇ",
    "s" + _DOT_BELOW: "ṣ", "s" + _ACUTE: "ś",     # ś is s + acute, not an accent
    "n" + _TILDE: "ñ", "n" + _DOT_ABOVE: "ṅ",
    "a" + _CIRCUMFLEX: "ê", "e" + _CIRCUMFLEX: "ê", "o" + _CIRCUMFLEX: "ô",
    "e" + _MACRON: "e", "o" + _MACRON: "o",
    "r" + _RING_BELOW: "r̥", "l" + _RING_BELOW: "l̥",
    "m" + _TILDE: "m̐",
}

#: Marks that remain after folding and are prosodic, not segmental.
_ACCENT_MARKS = {_ACUTE: "॑", _GRAVE: "॑", _MACRON_BELOW: "॒", "\u0341": "᳚"}

_VOWEL_BASES = set("aāiīuūṛṝḷḹeoêô")


def _fold_iast_clusters(text: str) -> str:
    """Rewrite NFD IAST into precomposed letters plus Vedic accent markers.

    ``yá-jña-sya`` decomposes to ``y + a + U+0301``, and ``ś`` to ``s + U+0301``.
    One is an accent and the other is a letter; the difference is the base
    character.  Folding clusters first resolves that unambiguously, and the accent
    markers that survive are carried through to Devanāgarī as ``॑``/``॒``.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        base = text[i]
        j = i + 1
        marks: list[str] = []
        while j < n and unicodedata.combining(text[j]):
            marks.append(text[j])
            j += 1
        if not marks:
            out.append(base)
            i = j
            continue
        key = base + "".join(marks)
        if key in _CLUSTER_TO_IAST:
            out.append(_CLUSTER_TO_IAST[key])
            i = j
            continue
        # Not a segmental cluster: split letters from accents.
        letter_marks = [m for m in marks if m not in _ACCENT_MARKS]
        accent_marks = [m for m in marks if m in _ACCENT_MARKS]
        sub = base + "".join(letter_marks)
        out.append(_CLUSTER_TO_IAST.get(sub, sub))
        out.extend(accent_marks)
        i = j
    return "".join(out)


def roman_to_devanagari(text: str) -> str:
    """Transliterate IAST (or plain ASCII Sanskrit) to Devanāgarī.

    Two Sanskrit-specific decisions that a naive transliterator gets wrong:

    * A **word-final consonant has no vowel**.  ``agnim`` is अग्निम्, not अग्निम.
      So the implicit ``a`` is dropped when a consonant ends a word.
    * ``ḷ`` is genuinely ambiguous in IAST: ISO 15919 reserves it for vocalic ऌ,
      but the Vedic scholarly tradition overwhelmingly uses it for the retroflex
      lateral ळ (as in ``agnim īḷe``).  We resolve by context: before a vowel it is
      the consonant ळ, before a consonant the vocalic ऌ.  ``ḻ`` is always ळ.
    """
    text = _fold_iast_clusters(unicodedata.normalize("NFD", text).strip())
    out: list[str] = []
    i = 0
    n = len(text)
    prev_was_cons = False

    def next_nonspace(k: int) -> str:
        while k < n and text[k].isspace():
            k += 1
        return text[k: k + 1]

    while i < n:
        ch = text[i]
        if ch.isspace():
            if prev_was_cons:          # word-final consonant: needs virama
                out.append(VIRAMA_DEV)
                prev_was_cons = False
            out.append(ch)
            i += 1
            continue
        if ch in ",.;:!?()[]\"'":
            if prev_was_cons:
                out.append(VIRAMA_DEV)
                prev_was_cons = False
            out.append(ch)
            i += 1
            continue

        matched = None
        for key in _IAST_KEYS:
            if text.startswith(key, i):
                matched = key
                break
        if matched is None:
            out.append(ch)
            prev_was_cons = False
            i += 1
            continue

        # ḷ disambiguation (see docstring)
        if matched == "ḷ":
            nxt = next_nonspace(i + 1)
            if nxt and nxt in (_VOWEL_FROM_IAST or {}) or nxt in ("a", "ā", "i", "ī", "u", "ū", "e", "o"):
                matched = "ḻ"          # consonant ळ
            else:
                matched = "ḷ"          # vocalic ऌ

        if matched in _CONS_FROM_IAST:
            if prev_was_cons:          # previous consonant had no vowel -> virama
                out.append(VIRAMA_DEV)
            out.append(_CONS_FROM_IAST[matched])
            prev_was_cons = True
        elif matched in _VOWEL_FROM_IAST:
            if prev_was_cons:
                if matched != "a":
                    out.append(_MATRA_FROM_IAST[matched])
            else:
                out.append(_VOWEL_FROM_IAST[matched])
            prev_was_cons = False
        elif matched in _MATRA_FROM_IAST:
            # A matra is only right after a consonant.  At a word boundary the same
            # vowel is an independent letter: the accented short vowel that IAST
            # writes as "á"/"í" (yá-jña-sya) must become अ-..., not a stray matra,
            # which would silently shift every following letter by one.
            if prev_was_cons:
                out.append(_MATRA_FROM_IAST[matched])
            else:
                out.append(_VOWEL_FROM_IAST[matched])
            prev_was_cons = False
        elif matched in _SPECIAL_IAST:
            out.append(_SPECIAL_IAST[matched])
            prev_was_cons = False
        elif matched in ("ṃ", "m̐"):
            out.append("ं")
            prev_was_cons = False
        elif matched == "ḥ":
            out.append("ः")
            prev_was_cons = False
        elif matched in ("'", "’"):
            out.append("ऽ")
            prev_was_cons = False
        elif matched in ("|", "‖"):
            out.append(matched)
            prev_was_cons = False
        else:
            out.append(_DEV_TO_IAST.get(matched, matched))
            prev_was_cons = False
        i += len(matched)

    if prev_was_cons:
        out.append(VIRAMA_DEV)
    return "".join(out)


# --------------------------------------------------------------------------- #
# Indic script offsets
# --------------------------------------------------------------------------- #


def _block_convert(text: str, src: Script, dst: Script = Script.DEVANAGARI) -> str:
    """Convert between two layout-aligned Brahmic blocks by codepoint offset."""
    if src not in _ALIGNED_BLOCKS or dst != Script.DEVANAGARI:
        raise ValueError(f"{src} is not layout-aligned with Devanāgarī")
    start = _ALIGNED_BLOCKS[src]
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if start <= cp < start + _BLOCK_SPAN:
            out.append(chr(cp - start + _DEVANAGARI_START))
        else:
            out.append(ch)
    return "".join(out)


def to_devanagari(text: str, *, script: Script | None = None) -> str:
    """Normalise any supported script to Devanāgarī."""
    text = normalize(text)
    script = script or detect_script(text)
    if script == Script.DEVANAGARI:
        return text
    if script in (Script.IAST, Script.LATIN):
        return roman_to_devanagari(text)
    if script in _ALIGNED_BLOCKS:
        dev = _block_convert(text, script)
        # Kannada/Telugu/Malayalam have ḻa (ಳ/ళ/ള) which Devanāgarī writes as ळ;
        # the offset gives us the right codepoint already, no special case needed.
        return dev
    if script == Script.TAMIL:
        return _tamil_to_devanagari(text)
    raise ValueError(f"unsupported script: {script}")


#: Tamil is lossy: it merges voiced/voiceless and has no aspirates.  We map the
#: *unaspirated* series (which is what Tamil actually writes) and let the caller
#: override with a Devanāgarī original when fidelity matters.
_TAMIL_MAP = {
    "அ": "अ", "ஆ": "आ", "இ": "इ", "ஈ": "ई", "உ": "उ", "ஊ": "ऊ",
    "எ": "ए", "ஏ": "ए", "ஐ": "ऐ", "ஒ": "ओ", "ஓ": "ओ", "ஔ": "औ",
    "க": "क", "ங": "ङ", "ச": "च", "ஞ": "ञ", "ட": "ट", "ண": "ण",
    "த": "त", "ந": "न", "ப": "प", "ம": "म", "ய": "य", "ர": "र",
    "ல": "ल", "ள": "ळ", "வ": "व", "ழ": "ऴ", "ற": "र", "ன": "न",
    "ஶ": "श", "ஜ": "ज", "ஷ": "ष", "ஸ": "स", "ஹ": "ह",
    "ா": "ा", "ி": "ि", "ீ": "ी", "ு": "ु", "ூ": "ू", "ெ": "े",
    "ே": "े", "ை": "ै", "ொ": "ो", "ோ": "ो", "ௌ": "ौ",
    "்": "्", "ஂ": "ं", "ஃ": "ः",
}


def _tamil_to_devanagari(text: str) -> str:
    return "".join(_TAMIL_MAP.get(ch, ch) for ch in text)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
_SAMA_NUMERALS = dict(zip("१२३४५६७", "1234567"))


def strip_symbols(text: str) -> str:
    """Remove punctuation, daṇḍas, verse numbers and editorial marks."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_ZERO_WIDTH)
    text = re.sub(r"[|‖।.!?;:\"“”„(),\[\]{}<>*#/\\—–-]+", " ", text)
    text = re.sub(r"[0-9०-९]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize(text: str) -> str:
    """Canonicalise a verse for processing.

    * NFC, zero-width joiners dropped
    * Devanāgarī punctuation and Western punctuation folded to a space
    * repeated whitespace collapsed, but *line structure preserved* — in gāna
      texts a line is a breath unit (a daṇḍa-delimited arcika), so it is
      musically meaningful and must survive normalisation.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_ZERO_WIDTH)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[|‖।]+", " | ", line)
        line = re.sub(r"\s*\|\s*", " | ", line)
        line = line.strip(" |")
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def is_devanagari_char(ch: str) -> bool:
    return 0x0900 <= ord(ch) <= 0x097F
