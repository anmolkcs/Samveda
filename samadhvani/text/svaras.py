"""The Sāmavedic seven-note system and its arithmetic.

Sāma notation is older than, and reads in the opposite direction to, the modern
saptasvara.  The seven sāma svaras descend from the highest:

    क्रुष्ट kṛṣṭa   प्रथम prathama   द्वितीय dvitīya   तृतीय tṛtīya
    चतुर्थ caturtha   मन्द्र mandra   अतिस्वार्य atisvārya

Nāradīya Śikṣā 1.5.1–2 relates them to the flute (veṇu) as Ma, Ga, Ri, Sa, Ni,
Dha, Pa — which is why the sāma scale is *not* a permutation of the modern one but
a rotation whose tonic is where the phrases actually cadence.  We model it as an
ordered scale in semitones above "Sa" (the śaḍja), with Sa placed at sāma-4
(caturtha) for the Kauthuma tradition.  Practically, what matters for synthesis is:

1. the **relative** interval of each note,
2. that the *pair* udātta/anudātta (RV-style three-tone recitation) maps into the
   same seven-note space as the full gāna melodies, so one model covers both.

The numbers 1..7 that later texts print above the syllables are *sāma-vikāra
notation* (they also mark prolongation); the kauthuma gāna texts encode the same
information with Vedic accents plus numerals.  We accept both.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from ..phones.inventory import VOWEL_FORMANTS


class Svara(IntEnum):
    """The seven sāma notes, high to low (values are the traditional numbers)."""

    KRUSHTA = 1
    PRATHAMA = 2
    DVITIYA = 3
    TRITIYA = 4
    CHATURTHA = 5
    MANDRA = 6
    ATISVARYA = 7


SAMA_SVARAS: dict[int, dict] = {
    1: {"name": "क्रुष्ट", "iast": "kṛṣṭa", "venu": "म Ma", "western": "Ma", "semitone": 5},
    2: {"name": "प्रथम", "iast": "prathama", "venu": "ग Ga", "western": "Ga", "semitone": 4},
    3: {"name": "द्वितीय", "iast": "dvitīya", "venu": "रि Ri", "western": "Ri", "semitone": 2},
    4: {"name": "तृतीय", "iast": "tṛtīya", "venu": "स Sa", "western": "Sa", "semitone": 0},
    5: {"name": "चतुर्थ", "iast": "caturtha", "venu": "नि Ni", "western": "Ni", "semitone": -1},
    6: {"name": "मन्द्र", "iast": "mandra", "venu": "ध Dha", "western": "Dha", "semitone": -3},
    7: {"name": "अतिस्वार्य", "iast": "atisvārya", "venu": "प Pa", "western": "Pa", "semitone": -5},
}

#: Nāradīya-śikṣā's veṇu mapping, exposed explicitly because people look for it.
NARADIYA_VENU_MAP = {k: v["venu"] for k, v in SAMA_SVARAS.items()}

#: Sāma numbers (1..7) and their Vedic-accent / common-text spellings.
#: Note 4 (tṛtīya, Sa) is the reciting baseline; mandra drops an octave region in
#: slow gāna, which matters for register planning.
_MANDRA_OCTAVE = 6

#: How far each note sits from the tonic, in semitones.  The venu mapping gives
#: sāma-4 = Sa.  Notes below 4 are flattened by the traditional scale's shape;
#: these intervals follow the ×2/×3 (grama) tetrachord construction.
def svara_to_semitone(svara: int, *, tonic_hz: float = 146.83) -> float:
    """Absolute frequency of a sāma svara, given the tonic (Sa, tṛtīya).

    Default tonic 146.83 Hz (D3) — a comfortable male chant register that leaves
    room for the mandra notes above the vocal floor and for kṛṣṭa without
    straining.  The reference recordings of Kauthuma gāna at low pitch sit close
    to this.
    """
    if svara not in SAMA_SVARAS:
        raise ValueError(f"svara must be 1..7, got {svara}")
    semi = SAMA_SVARAS[svara]["semitone"]
    hz = tonic_hz * (2.0 ** (semi / 12.0))
    return hz


@dataclass
class SvaraToken:
    """A svara assignment for one akṣara, with its provenance."""

    svara: int
    duration_matra: float = 1.0
    glide_from: int | None = None      # kampita / intonation glide
    stretch: float = 1.0               # vikarṣaṇa / karshana multiplier
    source: str = "rule"               # rule | markup | model | accent

    @property
    def name_iast(self) -> str:
        return SAMA_SVARAS[self.svara]["iast"]


def annotate_svaras(aksaras, *, scheme: str = "tradition") -> list[SvaraToken]:
    """Assign a sāma svara to every akṣara of a syllabified verse.

    This is the last-resort "no notation in the source text" path.  It is a
    *musically informed heuristic*, not a transcription model: it reproduces the
    shape of gāna — a phrase starts on tṛtīya/Sa, lifts for the accent-bearing
    syllable, cadences down to mandra or atisvārya at the breath boundary — while
    remaining honest about being an approximation.

    Where a real gāna text (with accents and numerals) is available, use
    :func:`samadhvani.text.vikara.parse_gana` instead; that path is exact.
    """
    from ..text.syllabify import SyllableView

    tokens: list[SvaraToken] = []
    if isinstance(aksaras, SyllableView):
        groups = aksaras.groups
        items = aksaras.aksaras
    else:
        items = list(aksaras)
        groups = [list(range(len(items)))]

    for group in groups:
        n = len(group)
        if n == 0:
            continue
        for pos, idx in enumerate(group):
            ak = items[idx]
            frac = pos / max(n - 1, 1)

            if scheme == "tradition":
                # A gāna phrase rises to a peak in its middle and settles at the end.
                if pos == 0:
                    s = 4                       # tṛtīya — the reciting note
                elif pos == n - 1:
                    s = 7 if n > 6 else 6       # cadence: atisvārya / mandra
                elif 0.45 <= frac <= 0.7:
                    s = 2 if pos % 2 else 1     # prathama / kṛṣṭa at the peak
                elif frac > 0.7:
                    s = 5
                else:
                    s = 3
            else:
                # Ṛgvedic recitation honours the accents directly: udātta (॑) is the
                # raised note, anudātta (॒) the lowered one, and an unmarked
                # syllable is svarita.  Mapped into the sāma scale so the acoustic
                # model sees one note space regardless of mode.
                if ak.accent == "॑":
                    s = 1        # udātta  -> kṛṣṭa, the top of the recitation
                elif ak.accent == "॒":
                    s = 6        # anudātta -> mandra, the bottom
                else:
                    s = 4        # svarita -> tṛtīya, the middle note

            # The avagraha / pluta of the text already lives on the akṣara.
            stretch = max(1.0, ak.prolongation)
            tokens.append(
                SvaraToken(
                    svara=int(ak.svara or s),
                    duration_matra=ak.effective_matra,
                    stretch=stretch,
                    source="markup" if ak.svara else "rule",
                )
            )
    return tokens
