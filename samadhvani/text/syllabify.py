"""Devanāgarī akṣara segmentation.

An *akṣara* is a syllable: an optional onset cluster, one nucleus (short, long, or
syllabic liquid), and an optional coda (anusvāra / visarga).  Sanskrit orthography
makes this almost trivial — the nucleus is exactly one of an independent vowel or a
consonant-with-matra — which is a large part of why akṣara-level force alignment is
the right granularity for chant corpora (see :mod:`samadhvani.data`).

Two things are *not* trivial and are handled explicitly:

* **Vedic accents** (udātta ``॑``, anudātta ``॒``) are combining marks that sit
  *inside* a syllable.  They are stripped before segmentation and re-attached, so
  that ``अ॒ग्निमी॑ळे`` does not split into seven pieces.

* **Sāma prolongation**.  A gāna text writes a sung long vowel three, four or five
  times over (vikarṣaṇa): ``ये`` becomes ``याऽ२३यि``, ``हा३उ`` and so on.  Those
  numerals are *duration*, not verse numbers, so the forward text normaliser must
  not strip them (it doesn't — ``strip_symbols`` is opt-in) and the syllabifier
  folds them into the preceding nucleus as a lengthening factor.

The output is a list of :class:`Aksara`, each carrying its onset/nucleus/coda, its
mātrā length, and (if a svara map was supplied) the sāma note it is sung on.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .unicodedata_ext import normalize

VIRAMA = "्"
NUKTA = "़"

#: Consonants that can stack into an onset cluster.
_CONSONANTS = set("कखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसह")
_CONSONANT_RE = re.compile(f"[{''.join(_CONSONANTS)}]")

#: Independent vowel -> (vowel symbol, length in mātrās)
_INDEPENDENT = {
    "अ": ("a", 1.0), "आ": ("ā", 2.0), "इ": ("i", 1.0), "ई": ("ī", 2.0),
    "उ": ("u", 1.0), "ऊ": ("ū", 2.0), "ऋ": ("ṛ", 1.0), "ॠ": ("ṝ", 2.0),
    "ऌ": ("ḷ", 1.0), "ॡ": ("ḹ", 2.0), "ए": ("e", 2.0), "ऐ": ("ai", 2.0),
    "ओ": ("o", 2.0), "औ": ("au", 2.0), "ऍ": ("ê", 1.0), "ऑ": ("ô", 2.0),
}

#: Matra -> (vowel symbol, length in mātrās).  The implicit ``a`` is separate.
_MATRA = {
    "ा": ("ā", 2.0), "ि": ("i", 1.0), "ी": ("ī", 2.0), "ु": ("u", 1.0),
    "ू": ("ū", 2.0), "ृ": ("ṛ", 1.0), "ॄ": ("ṝ", 2.0), "ॢ": ("ḷ", 1.0),
    "ॣ": ("ḹ", 2.0), "े": ("e", 2.0), "ै": ("ai", 2.0), "ो": ("o", 2.0),
    "ौ": ("au", 2.0), "ॅ": ("ê", 1.0), "ॉ": ("ô", 2.0),
}

_ACCENTS = "॒॑॓॔᳚᳡"
#: Sāma musical notation that is not part of the word.
_NOTATION_RE = re.compile("[॒॑॓॔᳚᳡२३४५६७८]")
_ACCENT_RE = re.compile(f"[{_ACCENTS}]")

#: Sāma lengthening marks.  Vedic ``२``/``३``/``४`` are *pluta* (prolongation)
#: marks, not digits; they multiply the nucleus.
_PROLONG_RE = re.compile("[२३४५६]")
_PROLONG_FACTOR = {"२": 2.0, "३": 3.0, "४": 4.0, "५": 5.0, "६": 6.0}

#: Characters that end a sung line / breath unit without being a syllable.
_BREAKS = set("|‖।। \n\t.,;:!?-–—")


@dataclass
class Aksara:
    """One syllable of a chant text."""

    onset: list[str] = field(default_factory=list)   # Devanāgarī consonants
    nucleus: str = "a"                              # roman vowel symbol
    nucleus_deva: str = "अ"
    coda: str = ""                                  # "" | "ṃ" | "ḥ"
    matra: float = 1.0
    accent: str = ""                                # udātta / anudātta mark, if any
    prolongation: float = 1.0
    svara: int | None = None                        # 1..7 sāma note
    source: str = ""                                # the raw Devanāgarī slice

    # -- conveniences ------------------------------------------------------ #
    @property
    def text(self) -> str:
        """Devanāgarī spelling of the syllable, for display.

        The source string is used because it preserves the spelling the text
        actually had (including a folded coda consonant), but the *notation* is
        stripped: the prolongation numerals and the accents are musical directions,
        not letters, and showing "मो२" beside "मो" in a syllable table reads as a
        text error rather than as a held note.
        """
        raw = self.source or (
            self.nucleus_deva if not self.onset
            else "".join(self.onset) + self.nucleus_deva
        )
        return _NOTATION_RE.sub("", raw)

    @property
    def notation(self) -> str:
        """The raw spelling including prolongation numerals and accents."""
        return self.source

    @property
    def is_stobha(self) -> bool:
        return any("\u0950" in o for o in self.onset) or self.source in ("ओ", "औ")

    @property
    def effective_matra(self) -> float:
        """Length after vikarṣaṇa (prolongation) is applied."""
        return self.matra * self.prolongation

    def __repr__(self) -> str:
        return f"Aksara({self.text!r}, {self.nucleus!r}, m={self.matra:g}, s={self.svara})"


@dataclass
class SyllableView:
    """A whole verse, segmented into akṣaras and breath groups."""

    aksaras: list[Aksara]
    groups: list[list[int]]               # indices into ``aksaras`` per musical line
    raw: str = ""

    def __len__(self) -> int:
        return len(self.aksaras)

    def syllables(self) -> list[str]:
        return [a.text for a in self.aksaras]

    def total_matra(self) -> float:
        return sum(a.effective_matra for a in self.aksaras)


def _clean_line(line: str) -> str:
    line = unicodedata.normalize("NFC", line)
    return _ACCENT_RE.sub("", line)


def _flush(aksaras: list[Aksara], group: list[int], nxt: int) -> None:
    if group:
        aksaras.append  # (documentation only)


def syllabify(text: str, *, keep_accents: bool = True, svara_map: list[int] | None = None) -> SyllableView:
    """Segment ``text`` into akṣaras.

    ``svara_map`` is an optional flat list of sāma note numbers (1–7) with one
    entry per akṣara, as produced by :func:`samadhvani.text.svaras.annotate_svaras`.
    """
    text = normalize(text)

    # Accents are combining marks sitting between a consonant and its matra
    # (``मी॑`` = म + ी + ॑).  If left in place the matra lookup misses and the
    # inherent vowel is silently lost ("agnimīḷe" -> "agnimḷe"), so they are
    # removed from the working string and mapped back onto spanning akṣaras.
    accents: dict[int, str] = {}
    clean_chars: list[str] = []
    clean_to_orig: list[int] = []
    for orig_i, ch in enumerate(text):
        if ch in _ACCENTS:
            accents[orig_i] = ch
            continue
        clean_chars.append(ch)
        clean_to_orig.append(orig_i)
    clean = "".join(clean_chars)

    # orig offset -> clean offset (for accent re-attachment)
    def to_clean(orig_i: int) -> int:
        import bisect
        return bisect.bisect_left(clean_to_orig, orig_i)

    spans: list[tuple[int, int]] = []
    aksaras: list[Aksara] = []
    groups: list[list[int]] = []
    current_group: list[int] = []

    i = 0
    n = len(clean)
    while i < n:
        ch = clean[i]

        if ch in ("\n", "|"):
            if current_group:
                groups.append(current_group)
                current_group = []
            i += 1
            continue
        if ch in _PROLONG_FACTOR:
            if aksaras:
                f = _PROLONG_FACTOR[ch]
                aksaras[-1].prolongation *= f
                aksaras[-1].source += ch
                spans[-1] = (spans[-1][0], i + 1)
            i += 1
            continue
        if ch.isspace() or ch in ".,;:!?-–—" or ch.isdigit():
            i += 1
            continue
        if ch == "ऽ":   # avagraha: the elided initial 'a', carries no syllable
            i += 1
            continue

        # ---- consonant-initial akṣara ----
        if ch in _CONSONANTS:
            start = i
            onset: list[str] = []
            source = ""
            # Reset per-syllable state.  Without this, `nucleus` keeps the value from
            # the previous akṣara and a word-final consonant looks like it already
            # has a vowel — which is how "रामम्" silently grew a third syllable.
            nucleus, matra, nucleus_deva = "a", 1.0, "अ"
            coda = ""
            while i < n and clean[i] in _CONSONANTS:
                onset.append(clean[i])
                source += clean[i]
                i += 1
                if i < n and clean[i] == NUKTA:
                    source += clean[i]
                    i += 1
                if i < n and clean[i] == VIRAMA:
                    source += clean[i]
                    i += 1
                    continue
                break
            # The onset loop consumes the virama of *cluster-internal* consonants
            # (रा-म-म्: the first म is followed by a virama, but the second म ends the
            # cluster).  A virama left here therefore means the whole syllable has no
            # vowel and belongs to the previous akṣara.
            syllable_closed = (i > start and clean[i - 1] == VIRAMA)
            if syllable_closed:
                nucleus, matra, nucleus_deva = "", 0.0, VIRAMA
            elif i < n and clean[i] in _MATRA:
                nucleus, matra = _MATRA[clean[i]]
                nucleus_deva = clean[i]
                source += clean[i]
                i += 1
            elif i < n and clean[i] == VIRAMA:
                nucleus, matra, nucleus_deva = "", 0.0, VIRAMA
                source += clean[i]
                i += 1
            else:
                nucleus, matra, nucleus_deva = "a", 1.0, "अ"
            coda = ""
            while i < n and clean[i] in ("ं", "ँ", "ः"):
                coda += "ṃ" if clean[i] in "ंँ" else "ḥ"
                source += clean[i]
                i += 1
            if not nucleus and not coda:
                # A virama-final consonant is a *coda* of the previous syllable, not a
                # syllable of its own: रामम् is rā-ma-m, two akṣaras, not three.  It
                # belongs to the preceding nucleus even where it is spelled as part of
                # a new onset cluster (which is why the fold happens here and not in
                # the onset loop).
                if aksaras:
                    # `source` here ends with the consonant but not its virama (the
                    # onset loop only keeps the virama of cluster-internal members),
                    # so the check above already sees nucleus == "" — append both.
                    # A trailing folded consonant: keep the whole consonant so the
                    # phone layer can emit it (the fold can happen several letters
                    # back, e.g. after a cluster).
                    folded = onset[-1] if onset else ""
                    aksaras[-1].coda += ("C" + folded) if folded else "C"
                    aksaras[-1].source += source
                    spans[-1] = (spans[-1][0], i)
                continue
            aksaras.append(Aksara(onset=onset, nucleus=nucleus, nucleus_deva=nucleus_deva,
                                  coda=coda, matra=matra, source=source))
            spans.append((start, i))
            current_group.append(len(aksaras) - 1)
            continue

        # ---- vowel-initial akṣara ----
        if ch in _INDEPENDENT:
            start = i
            nucleus, matra = _INDEPENDENT[ch]
            ak = Aksara(onset=[], nucleus=nucleus, nucleus_deva=ch, matra=matra, source=ch)
            i += 1
            while i < n and clean[i] in ("ं", "ँ", "ः"):
                ak.coda += "ṃ" if clean[i] in "ंँ" else "ḥ"
                ak.source += clean[i]
                i += 1
            aksaras.append(ak)
            spans.append((start, i))
            current_group.append(len(aksaras) - 1)
            continue

        if ch == "ॐ":
            start = i
            aksaras.append(Aksara(nucleus="o", nucleus_deva="ओ", matra=2.0, coda="ṃ", source="ॐ"))
            spans.append((start, i + 1))
            current_group.append(len(aksaras) - 1)
            i += 1
            continue

        i += 1  # unknown symbol: skip

    if current_group:
        groups.append(current_group)
    if not groups and aksaras:
        groups = [list(range(len(aksaras)))]

    # Re-attach Vedic accents by span (an accent follows the syllable it marks).
    if keep_accents and accents:
        import bisect
        starts = [sp[0] for sp in spans]
        for orig_pos, acc in accents.items():
            cpos = to_clean(orig_pos)
            idx = bisect.bisect_left(starts, cpos) - 1
            if 0 <= idx < len(aksaras) and not aksaras[idx].accent:
                aksaras[idx].accent = acc

    if svara_map:
        for idx, ak in enumerate(aksaras):
            if idx < len(svara_map) and svara_map[idx]:
                ak.svara = svara_map[idx]

    return SyllableView(aksaras=aksaras, groups=groups, raw=text)


def text_to_aksaras(text: str, **kw) -> list[Aksara]:
    return syllabify(text, **kw).aksaras


# --------------------------------------------------------------------------- #
# Light sandhi splitting
# --------------------------------------------------------------------------- #

#: Sandhi is not *undone* — the acoustic model should hear what is written.  But
#: for alignment and for displaying "words" in the UI it helps to know plausible
#: boundaries.  These are the high-precision cases only.
_SANDHI_CUES = [
    (re.compile(r"([ािीुूृेैोौ])्(?=[कखगघचछजझटठडढतथदधनपफबभमयरलवशषसह])"), r"\1 | "),
    (re.compile(r"ो(?=[अआइईउऊएऐओऔ])"), "o' | "),
    (re.compile(r"े(?=[अआइईउऊएऐओऔ])"), "e' | "),
]


def split_sandhi(text: str) -> list[str]:
    """Return plausible word boundaries, *without* altering the surface form.

    This is a hint, not a segmentation: Sanskrit sandhi is context-dependent and a
    regex cannot resolve it.  Anything that consumes this for supervision should
    treat the output as soft labels.
    """
    text = normalize(text)
    spaced = text
    for pat, rep in _SANDHI_CUES:
        spaced = pat.sub(rep, spaced)
    return [w for w in spaced.replace("|", " ").split() if w]
