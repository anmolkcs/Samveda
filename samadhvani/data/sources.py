"""Where the data comes from, and what each source can legally be used for.

A chant TTS needs two different kinds of data and they are *not* interchangeable:

* **text** — the gāna books.  Abundant and freely available.  Hundreds of thousands
  of akṣaras with svara notation.
* **audio with text** — rare, and encumbered.  This is the bottleneck, and the
  single most useful thing this file can do is be blunt about it.

The catalogue below is the honest version: what exists, where it lives, what its
licence appears to be, and how many hours of *usable* audio it actually has.  Every
entry has a ``redistribute`` flag; anything not explicitly open must not be baked
into a released checkpoint.

Read :func:`audit` — it prints the total usable hours.  If you are starting today,
that number is small, and the sensible first move is the recording protocol in
``docs/DATA.md``, not a bigger model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Licence(str, Enum):
    OPEN = "open"                  # CC-BY / CC0 / public domain: safe to train and ship
    OPEN_TEXT_ONLY = "text-open"   # the text is open, the audio is not
    RESEARCH_ONLY = "research"     # may be used to train, may not be redistributed
    UNCLEAR = "unclear"            # provenance unknown: do not train on it
    PERMISSION = "permission"      # needs a written licence from the holder


@dataclass
class Source:
    key: str
    name: str
    url: str
    kind: str                       # text | audio | both
    licence: Licence
    veda: str = "samaveda"
    school: str = ""
    est_hours: float = 0.0
    est_verses: int = 0
    has_svara_notation: bool = False
    notes: str = ""
    local_path: str | None = None
    redistribute: bool = False

    @property
    def usable_for_training(self) -> bool:
        return self.licence in (Licence.OPEN, Licence.RESEARCH_ONLY, Licence.PERMISSION)

    @property
    def usable_for_release(self) -> bool:
        return self.licence is Licence.OPEN


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #

SOURCES: list[Source] = [
    # ---- text ------------------------------------------------------------ #
    Source(
        key="vhp_kauthuma",
        name="Vedic Heritage Portal — Kauthuma Saṃhitā (Pūrvārcika / Uttarārcika)",
        url="https://vedicheritage.gov.in/samhitas/samaveda-samhitas/kauthuma-samhita/",
        kind="text",
        licence=Licence.OPEN_TEXT_ONLY,
        school="kauthuma",
        est_verses=1875,
        has_svara_notation=True,
        notes=(
            "The Government of India's portal. The Kauthuma saṃhitā with Vedic accents. "
            "Primary reference text; the audio on the same portal is streamed for "
            "listening and is NOT published under an open licence — see the note on "
            "the portal audio below."
        ),
    ),
    Source(
        key="vhf_audio",
        name="Vedic Heritage Portal — recitation audio (Sāmaveda Uḍgātā recordings)",
        url="https://vedicheritage.gov.in/samaveda/",
        kind="audio",
        licence=Licence.PERMISSION,
        school="kauthuma",
        est_hours=6.0,
        notes=(
            "Recordings of Kauthuma sāmagāna by named uḍgātṛs. Excellent quality and the "
            "closest thing to a canonical reference. They are published for listening, "
            "not for training; the honest path is to write for permission (the portal "
            "names the institution and the artist). Until then treat them as an "
            "evaluation set at most — and even that needs a look at the terms."
        ),
        redistribute=False,
    ),
    Source(
        key="sd_gramageya",
        name="Sanskrit Documents — Sāma-gāna, Book I (Grāma-geya)",
        url="https://sanskritdocuments.org/sanskrit/veda/",
        kind="text",
        licence=Licence.OPEN,
        school="kauthuma",
        est_verses=1198,
        has_svara_notation=True,
        notes=(
            "1,198 gāna, 579 pages, typed by Pallasena Narayanaswami with the Vedic "
            "accents and sāma numerals carefully preserved. This is the best free gāna "
            "e-text that exists and the backbone of the text side."
        ),
        redistribute=True,
    ),
    Source(
        key="sd_aranyaka",
        name="Sanskrit Documents — Sāma-gāna, Book II (Āraṇyaka)",
        url="https://sanskritdocuments.org/doc_veda/aranyakaganam.pdf",
        kind="text",
        licence=Licence.OPEN,
        school="kauthuma",
        est_verses=296,
        has_svara_notation=True,
        notes="296 gāna. PDF; needs text extraction, the accents survive extraction.",
        redistribute=True,
    ),
    Source(
        key="sd_uha",
        name="Sanskrit Documents — Sāma-gāna, Book III (Ūha)",
        url="https://sanskritdocuments.org/sanskrit/veda/",
        kind="text",
        licence=Licence.OPEN,
        school="kauthuma",
        est_verses=936,
        has_svara_notation=True,
        notes="936 gāna, 515 pages.",
        redistribute=True,
    ),
    Source(
        key="sd_uhya",
        name="Sanskrit Documents — Sāma-gāna, Book IV (Ūhya)",
        url="https://sanskritdocuments.org/sanskrit/veda/",
        kind="text",
        licence=Licence.OPEN,
        school="kauthuma",
        est_verses=209,
        has_svara_notation=True,
        notes="209 gāna.",
        redistribute=True,
    ),
    Source(
        key="jaiminiya_text",
        name="Jaiminīya Sāmaveda (text, with Gāna)",
        url="https://sanskritdocuments.org/",
        kind="text",
        licence=Licence.OPEN,
        school="jaiminiya",
        est_verses=3000,
        notes=(
            "The Kerala tradition's recension. Text is available; the audio is scattered "
            "across YouTube uploads of Nambūdiri recitations with unclear rights."
        ),
        redistribute=True,
    ),
    # ---- audio ----------------------------------------------------------- #
    Source(
        key="youtube_samagana",
        name="YouTube — Kauthuma / Rāṇāyanīya sāmagāna uploads",
        url="https://www.youtube.com/",
        kind="audio",
        licence=Licence.UNCLEAR,
        school="kauthuma",
        est_hours=20.0,
        notes=(
            "There is a fair amount here: temple recordings, Vedic pathshala uploads, "
            "individual uḍgātṛ channels. The copyright status is per-video and usually "
            "not stated. **Do not train a released model on it without permission.** "
            "It is useful for listening and for building an evaluation set you never "
            "publish."
        ),
        redistribute=False,
    ),
    Source(
        key="archive_samaganam",
        name="Internet Archive — sAmagAnam (PR Iyer transcriptions)",
        url="https://archive.org/details/sAmagAnam",
        kind="text",
        licence=Licence.OPEN,
        has_svara_notation=True,
        notes=(
            "Scanned gāna texts with page-numbered audio timings for ~11 hours of "
            "gramageya and aranyaka gāna. The scans are public domain; the audio the "
            "timings refer to is not included."
        ),
        redistribute=True,
    ),
    Source(
        key="own_recordings",
        name="Your own recordings (the recommended path)",
        url="",
        kind="audio",
        licence=Licence.OPEN,
        est_hours=0.0,
        notes=(
            "Five hours of carefully recorded, consented, single-singer Sāma recitation "
            "with aligned text beats every public source on this list — that is the "
            "whole lesson of Vāgdhenu, whose ~5 h purpose-recorded corpus is what made "
            "the system work. Protocol in docs/DATA.md."
        ),
        redistribute=True,
    ),
]


def by_key(key: str) -> Source:
    for s in SOURCES:
        if s.key == key:
            return s
    raise KeyError(key)


def audit(verbose: bool = True) -> dict:
    """Summarise what is actually trainable, and how much of it there is."""
    text_sources = [s for s in SOURCES if s.kind in ("text", "both")]
    audio_sources = [s for s in SOURCES if s.kind in ("audio", "both")]
    trainable_audio = [s for s in audio_sources if s.usable_for_training]
    releasable_audio = [s for s in audio_sources if s.usable_for_release]
    total_train_hours = sum(s.est_hours for s in trainable_audio)
    total_release_hours = sum(s.est_hours for s in releasable_audio)

    report = {
        "text_sources": len(text_sources),
        "audio_sources": len(audio_sources),
        "text_with_svara": [s.key for s in text_sources if s.has_svara_notation],
        "total_ganas_in_text": sum(s.est_verses for s in text_sources),
        "audio_hours_trainable": total_train_hours,
        "audio_hours_releasable": total_release_hours,
        "blocked": [s.key for s in SOURCES if s.licence in (Licence.UNCLEAR,)],
        "needs_permission": [s.key for s in SOURCES if s.licence is Licence.PERMISSION],
    }
    if verbose:
        print(f"{'source':22s} {'kind':6s} {'licence':11s} {'hrs':>6s}  train? ship?")
        print("-" * 72)
        for s in SOURCES:
            print(f"{s.key:22s} {s.kind:6s} {s.licence.value:11s} {s.est_hours:6.1f}"
                  f"   {'Y' if s.usable_for_training else 'n':^6s} {'Y' if s.usable_for_release else 'n':^5s}")
        print("-" * 72)
        print(f"trainable audio: {total_train_hours:.1f} h   "
              f"redistributable audio: {total_release_hours:.1f} h")
        print(f"text: {report['total_ganas_in_text']} gāna across "
              f"{len(report['text_with_svara'])} sources with svara notation")
        if report["blocked"]:
            print(f"BLOCKED (do not train a released model): {', '.join(report['blocked'])}")
        if report["needs_permission"]:
            print(f"needs written permission: {', '.join(report['needs_permission'])}")
    return report


if __name__ == "__main__":
    audit()
