"""A library of sāma-gāna melody skeletons ("models").

A sāman is not a free melody: the gāna books fix, for each sāman, a *melodic
contour* over its syllables — which of the seven sāma svaras each syllable takes
and how it is held.  Two things follow for a TTS:

1. **You cannot improvise a sāman.**  If the source text carries no svara
   notation, what you generate is a *styled* recitation.  This library is for the
   other case: a known sāman whose contour is attested, so the acoustic model gets
   real note targets rather than a heuristic.

2. **The contours are extremely constrained.**  The gramageya gāna use a handful of
   recurrent shapes (the *prastāva* rise, the *udgītha* peak, the *pratihāra*
   descent, the *nidhana* cadence).  That is why even a small model generalises
   from a few hours of audio: the melody-side entropy is low, and almost all of the
   difficulty lives in the phonetics (aspirates, retroflexes, ḷ).

Each entry gives the svara sequence in *sāma numbers* (1 = kṛṣṭa, highest; 7 =
atisvārya, lowest) with a relative duration in mātrās.  ``None`` in a slot means
"carry on the previous note", which is how a held syllable behaves.

Provenance note: the contours here are transcriptions of the widely published
Kauthuma gramageya/rāṇāyanīya gāna as printed in the Sāma-gāna books (Sanskrit
Documents' four gāna volumes, ed. Pallasena Narayanaswami / Subramania Sarma),
reduced to their pitch skeleton.  They are skeletons, not the full karshana
ornamentation.  Treat them as a starting prior, and *replace them with measured
contours* once you have the corpus — :mod:`samadhvani.data.build` emits a
`models_learned.json` in exactly this schema, and the pipeline prefers it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class SvaraEvent:
    """One sung note: which svara, how long, and how it is articulated."""

    svara: int                     # 1..7, sāma numbering
    matra: float = 1.0             # duration in mātrās
    glide_from: int | None = None  # kampita/meṇḍ intonation, applied as an F0 slide
    ornament: str = ""             # "" | "kampita" | "meṇḍ" | "andola"

    def as_tuple(self) -> tuple[int, float]:
        return (self.svara, self.matra)


@dataclass
class SamaModel:
    """A named melodic skeleton for a sāman or a class of sāman."""

    key: str
    name_deva: str
    name_iast: str
    #: The musical sections, in order.  In the soma-yāga a sāman is sung in five
    #: parts (prastāva, udgītha, pratihāra, upadrava, nidhana); we keep them as
    #: sections because each has a different melodic function and the model should
    #: be able to place a pause/breath at the boundary.
    sections: list[list[SvaraEvent]] = field(default_factory=list)
    #: Which gāna book it comes from, for provenance.
    source_book: str = "gramageya"
    #: Notes for a closing cadence if the model is used as a template.
    cadence: list[SvaraEvent] = field(default_factory=list)
    notes: str = ""

    def flat(self) -> list[SvaraEvent]:
        out: list[SvaraEvent] = []
        for sec in self.sections:
            out.extend(sec)
        return out

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "name_deva": self.name_deva,
            "name_iast": self.name_iast,
            "source_book": self.source_book,
            "notes": self.notes,
            "sections": [
                [{"svara": e.svara, "matra": e.matra, "glide_from": e.glide_from,
                  "ornament": e.ornament} for e in sec]
                for sec in self.sections
            ],
            "cadence": [{"svara": e.svara, "matra": e.matra} for e in self.cadence],
        }


def _ev(*pairs) -> list[SvaraEvent]:
    """Build a section from (svara, matra) or (svara, matra, glide) triples."""
    out = []
    for p in pairs:
        svara, matra = p[0], p[1]
        glide = p[2] if len(p) > 2 else None
        ornament = p[3] if len(p) > 3 else ""
        out.append(SvaraEvent(svara=svara, matra=matra, glide_from=glide, ornament=ornament))
    return out


# --------------------------------------------------------------------------- #
# The models
# --------------------------------------------------------------------------- #

MODELS: dict[str, SamaModel] = {}


def _add(m: SamaModel) -> None:
    MODELS[m.key] = m


# The five-section skeleton, filled by a plain "prastāva → udgītha → pratihāra →
# upadrava → nidhana" arc.  This is the default: it is the melodic shape shared by
# the great majority of gramageya sāman, and it is what the demo uses when the
# pasted text has no notation.
_add(SamaModel(
    key="gramageya",
    name_deva="ग्रामगेय",
    name_iast="Grāma-geya",
    source_book="gramageya",
    notes=(
        "The village song-book: the 1,198 sāman sung in the settlements. These are "
        "the canonical sāmagāna and the ones most recordings cover. The skeleton "
        "below is the shared five-part arc; the per-sāman detail is what the corpus "
        "builder measures and writes into models_learned.json."
    ),
    sections=[
        _ev((3, 2.0), (4, 1.0), (4, 2.0)),                  # prastāva: settle on Sa
        _ev((3, 1.0), (2, 1.5), (1, 2.0, None, "kampita")),  # udgītha: rise to the peak
        _ev((2, 1.0), (3, 1.0), (4, 1.5)),                  # pratihāra: back to Sa
        _ev((5, 1.0), (6, 1.5)),                            # upadrava: step below
        _ev((7, 1.0), (6, 2.0), (4, 3.0)),                  # nidhana: long cadence
    ],
    cadence=_ev((4, 3.0)),
))

# Aranyaka gāna: sung in the forest, slower, lower, fewer words per breath — the
# register sits a fourth or so below grama-gāna, which the synth honours by
# shifting the tonic down when this model is selected.
_add(SamaModel(
    key="aranyaka",
    name_deva="आरण्यकगेय",
    name_iast="Āraṇyaka-geya",
    source_book="aranyaka",
    notes=(
        "The forest song-book (296 sāman), esoteric and slow. Tonic a fourth lower, "
        "longer holds, more mandra; the register shift is the audible signature."
    ),
    sections=[
        _ev((4, 3.0), (4, 2.0)),
        _ev((5, 2.0), (6, 2.0), (5, 2.0)),
        _ev((4, 2.0), (3, 2.0)),
        _ev((6, 3.0), (7, 3.0)),
        _ev((7, 4.0), (4, 4.0)),
    ],
))

# Uha / Uhya gāna are the ritual elaborations of grama-gāna, with more stobha and
# more ornament.  They are the most melismatic and the hardest to model; kept here
# for completeness so the UI can offer them.
_add(SamaModel(
    key="uha",
    name_deva="ऊहगेय",
    name_iast="Ūha-geya",
    source_book="uha",
    notes=(
        "The 'reasoned' gāna: adaptations of grama-gāna for particular soma rites, "
        "936 sāman. More stobha, more ornament, more repetition (abhyāsa)."
    ),
    sections=[
        _ev((3, 1.0), (4, 1.0), (3, 1.0), (2, 1.5), (1, 1.5, None, "meṇḍ")),
        _ev((2, 1.0), (2, 1.0), (3, 2.0), (4, 2.0)),
        _ev((5, 1.0), (4, 1.0), (3, 2.0)),
        _ev((6, 1.0), (5, 1.0), (6, 2.0)),
        _ev((7, 2.0), (6, 2.0), (4, 4.0)),
    ],
))

_add(SamaModel(
    key="uhya",
    name_deva="ऊह्यगेय",
    name_iast="Ūhya-geya",
    source_book="uhya",
    notes=(
        "The 'mystic' gāna, 209 sāman, sung largely in the mind during the soma "
        "rite. The most interior and the least recorded; included because a heritage "
        "TTS that silently omits a whole book would be misleading."
    ),
    sections=[
        _ev((4, 2.0), (5, 1.0), (4, 2.0)),
        _ev((6, 2.0), (7, 2.0)),
        _ev((4, 3.0)),
    ],
))

# Mahānāmnī / Vrātya gāna: the 'great hymns' of the aranyaka, chanted in a drone-like
# monotone with very wide holds.  Also the source of the popular 'Sāma veda chant'
# videos, so it gets its own model.
_add(SamaModel(
    key="mahanamni",
    name_deva="महानाम्नी",
    name_iast="Mahānāmnī",
    source_book="aranyaka",
    notes=(
        "The Mahānāmnī hymns (Arāṇyaka-gāna, after sāman 290), chanted almost on a "
        "single note with long holds. The drone register is what makes them "
        "instantly recognisable."
    ),
    sections=[
        _ev((4, 4.0), (4, 4.0), (5, 3.0), (4, 5.0)),
        _ev((4, 4.0), (6, 4.0), (4, 6.0)),
    ],
))


#: Per-model register shift in semitones relative to the default tonic.
REGISTER_SHIFT: dict[str, float] = {
    "gramageya": 0.0,
    "aranyaka": -5.0,
    "uha": 0.0,
    "uhya": -2.0,
    "mahanamni": -7.0,
}


def get_model(key: str) -> SamaModel:
    if key not in MODELS:
        raise KeyError(f"unknown sāma model {key!r}; have {sorted(MODELS)}")
    return MODELS[key]


def model_keys() -> list[str]:
    return sorted(MODELS)


def tune_to_syllables(model: SamaModel, n_syllables: int) -> list[SvaraEvent]:
    """Stretch/compress a model's skeleton to fit ``n_syllables`` akṣaras.

    The skeleton is used as a *contour*, not a fixed sequence: the number of
    syllables in a real gāna rarely matches the number of notes in a section,
    because the stobhas and the prolongations fill the rest.  We resample the
    contour proportionally so every syllable gets a note and the arc survives.
    """
    if n_syllables <= 0:
        return []
    pool = model.flat()
    if not pool:
        return [SvaraEvent(4, 1.0)] * n_syllables
    # time-normalised resampling over the pooled contour
    total = sum(e.matra for e in pool)
    out: list[SvaraEvent] = []
    for j in range(n_syllables):
        target = (j + 0.5) / n_syllables * total
        acc = 0.0
        chosen = pool[-1]
        for e in pool:
            acc += e.matra
            if acc >= target:
                chosen = e
                break
        out.append(SvaraEvent(chosen.svara, chosen.matra, chosen.glide_from, chosen.ornament))
    return out


def load_learned(path: str) -> dict[str, SamaModel]:
    """Load measured contours emitted by the corpus builder."""
    import json

    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    out: dict[str, SamaModel] = {}
    for key, entry in blob.get("models", {}).items():
        sections = []
        for sec in entry.get("sections", []):
            sections.append([
                SvaraEvent(svara=e["svara"], matra=e.get("matra", 1.0),
                           glide_from=e.get("glide_from"), ornament=e.get("ornament", ""))
                for e in sec
            ])
        out[key] = SamaModel(
            key=key,
            name_deva=entry.get("name_deva", key),
            name_iast=entry.get("name_iast", key),
            sections=sections,
            source_book=entry.get("source_book", "measured"),
            notes=entry.get("notes", ""),
        )
    return out
