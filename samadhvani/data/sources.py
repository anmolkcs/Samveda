"""Where the data comes from, and what each source may be used for.

A chant TTS needs two very different kinds of data and they are *not*
interchangeable:

* **text** — the gāna books.  Abundant, free, and now machine-readable.
  2,639 Kauthuma gāna were typeset in January 2025.  Hundreds of thousands of
  akṣaras with svara notation.
* **audio with a transcript** — rare, and encumbered.  This is the bottleneck.

The catalogue below records *facts*: what exists, where it lives, what the
provider says about rights, and how many hours are actually usable.  It does not
moralise about them.

Two facts are worth stating before the table, because they decide everything:

1. **Almost nothing here is openly licensed.**  Not one of the usable audio
   collections carries a CC or public-domain statement.  The best text source is
   free to download under a fair-use-for-education declaration, and its 2025
   typesetting is stamped "PPN ©".  "Free to download" is not "open"; do not
   describe this corpus as open data.
2. **Intended use is a separate axis from licence.**  These are independent
   questions and the catalogue refuses to collapse them:

   * *May I redistribute this audio?* — a licence question.
   * *May I train a model and run it on my own machine for my own teaching?* —
     an intended-use question.

   :func:`audit` prints both.  :meth:`Source.permit` answers the second one.

So the catalogue has two kinds of answer:

``usable_for_release``  the strict question.  Only sources with an actual open
                        licence pass.  Right now that is your own recordings and
                        nothing else.

``permit(profile)``     the practical question.  Under the ``educational``
                        profile — *not distributed, not commercial* — the
                        free-for-teaching sources become usable for **training**,
                        while shipping the audio or a published checkpoint stays
                        blocked.

The ``educational`` profile is a promise about your own future behaviour, not a
licence you were granted.  It is a perfectly reasonable promise: nothing here is
being sold or published.  But it is revocable by *you*, in one step — publish the
checkpoint or host the demo and every source marked ``FREE_EDUCATIONAL`` or
``UNLICENSED`` flips to blocked.  :func:`audit` says so on the last line, every
time, because that is the line people lose track of.

If you are starting today: the recording protocol in ``docs/DATA.md`` still gives
you the best five hours you can get, because you own them outright.  The 15.8
hours below are how you get a working system *this week* while you do it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Licence(str, Enum):
    """What the provider says.  Verbatim, not interpreted."""

    OPEN = "open"                          # CC0 / CC-BY / public domain: train and ship
    CC_BY_NC = "cc-by-nc"                  # non-commercial; derivatives OK but NC
    CC_BY_NC_ND = "cc-by-nc-nd"            # non-commercial, no derivatives
    FREE_EDUCATIONAL = "free-educational"  # free to access for teaching/research; no open licence
    UNLICENSED = "unlicensed"              # hosted by a third party, no rights statement at all
    COMMERCIAL = "commercial"              # in print / for sale: needs a licence
    PERMISSION = "permission"              # institution holds it; write for written permission
    UNCLEAR = "unclear"                    # provenance unknown: do not train, any profile


#: Licence values that no profile can unblock by itself, because the problem is
#: provenance rather than permission.
_NEVER_TRAINABLE = (Licence.UNCLEAR,)


@dataclass(frozen=True)
class Profile:
    """An intended-use profile: what *you* promise to do with the result."""

    key: str
    name: str
    distributes: bool
    commercial: bool
    note: str


PROFILES: dict[str, Profile] = {
    "open": Profile(
        "open", "Publication / redistribution", distributes=True, commercial=False,
        note=(
            "You will publish the checkpoint, or host the demo. Only sources with a "
            "real open licence survive this profile — today that means your own "
            "recordings. This is the profile to use before anything becomes public."
        ),
    ),
    "educational": Profile(
        "educational", "Educational, not distributed, not commercial", distributes=False,
        commercial=False,
        note=(
            "Training and using the model locally for teaching and study; no "
            "checkpoint published, no audio republished, no money. This unblocks the "
            "free-for-education and third-party-hosted sources for *training*. It "
            "does not let you ship what came out of them."
        ),
    ),
    "evaluation": Profile(
        "evaluation", "Listening, checking, evaluating", distributes=False, commercial=False,
        note=(
            "Reading and listening only, no training. Everything except UNCLEAR is "
            "fine here: listening to a recording is not a restricted act."
        ),
    ),
}

DEFAULT_PROFILE = "educational"


@dataclass
class Source:
    key: str
    name: str
    url: str
    kind: str                       # text | audio | both | reference
    licence: Licence
    veda: str = "samaveda"
    school: str = ""
    est_hours: float = 0.0
    est_aligned_hours: float = 0.0  # hours carrying a text/timestamp mapping
    est_ganas: int = 0              # gāna-s (a gāna is a melody-setting, not a ṛc)
    est_verses: int = 0             # ṛc-s / saṃhitā verses
    has_svara_notation: bool = False
    notes: str = ""
    local_path: str | None = None
    redistribute: bool = False
    permission_granted: bool = False
    rights_basis: str = ""          # where the licence claim comes from (or does not)
    rights_url: str | None = None
    duplicate_of: str = ""          # mirror of another key: not counted twice

    # -- the strict licence question --------------------------------------- #
    @property
    def usable_for_training(self) -> bool:
        """Trainable under the *strict* profile (i.e. the result may ship)."""
        return self.licence in (Licence.OPEN,) or self.permission_granted

    @property
    def usable_for_release(self) -> bool:
        return self.licence is Licence.OPEN and self.redistribute

    # -- the intended-use question ----------------------------------------- #
    def permit(self, profile: str = DEFAULT_PROFILE) -> "Permit":
        """May I use this, given what I promise to do with the result?"""
        if profile not in PROFILES:
            raise KeyError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
        p = PROFILES[profile]

        if self.licence in _NEVER_TRAINABLE:
            return Permit(
                False, False, False,
                "Provenance unknown — you cannot name the rights holder, and Vedic "
                "uploads are frequently re-uploads of someone else's recording. Not a "
                "licence problem, so no profile fixes it.",
            )

        if self.licence is Licence.OPEN:
            return Permit(True, True, True, "Openly licensed.")

        if self.permission_granted:
            return Permit(True, True, True, "Written permission on file.")

        open_ended = (
            "this is only a promise about your own conduct, not a grant. Publishing "
            "the checkpoint or hosting the audio revives the restriction."
        )

        if profile == "evaluation":
            return Permit(False, False, False, "Listening only under this profile.")

        if profile == "open":
            return Permit(
                False, False, False,
                {
                    Licence.CC_BY_NC: "NC licence: training is fine, publishing is not.",
                    Licence.CC_BY_NC_ND: "ND licence: no derivatives may be distributed.",
                    Licence.FREE_EDUCATIONAL: "No open licence; free for teaching only.",
                    Licence.UNLICENSED: "No rights statement at all.",
                    Licence.COMMERCIAL: "In print and for sale; needs a licence.",
                    Licence.PERMISSION: "Institution holds it; write for permission.",
                }.get(self.licence, "Not releasable."),
            )

        # profile == "educational": train, but do not ship either artefact.
        if self.licence is Licence.PERMISSION:
            return Permit(
                False, False, False,
                "The institution holds these; 'educational use' is not permission and "
                "the portal does not grant it. Write to them — see docs/DATA.md §8.",
            )
        if self.licence is Licence.COMMERCIAL:
            return Permit(
                False, False, False,
                "Commercially released. Educational use of a *copy* is fair dealing; "
                "training on it is a derivative you would be making without a licence.",
            )
        return Permit(
            True, False, False,
            f"Free for teaching/research, so training is within the customary use — {open_ended}",
        )


@dataclass(frozen=True)
class Permit:
    """The answer to 'may I use this?', split into the three things people conflate."""

    may_train: bool
    may_ship_audio: bool
    may_ship_weights: bool
    reason: str

    def __bool__(self) -> bool:
        return self.may_train


# --------------------------------------------------------------------------- #
# The catalogue.  Hours are measured from archive.org metadata, not estimated.
# --------------------------------------------------------------------------- #

SOURCES: list[Source] = [
    # ======================= audio, with a transcript ====================== #
    Source(
        key="iish_samagana",
        name="IISH Global — complete Kauthuma Sāmagāna, 64 clips + timestamped transcription",
        url="https://archive.org/details/IISHSamaVeda",
        kind="audio",
        licence=Licence.UNLICENSED,
        school="kauthuma",
        est_hours=43.2,
        est_aligned_hours=15.8,
        has_svara_notation=True,
        rights_basis=(
            "No licence, rights, or creator-rights field on the archive.org item. "
            "Uploaded 2013 by a third party ('Prvn Raveendran') on behalf of IISH; the "
            "recordings are IISH's and the reciter is credited only as 'Dr. N Gopala "
            "Krishnan'. Free to stream, not licensed to you."
        ),
        rights_url="https://archive.org/metadata/IISHSamaVeda/metadata",
        notes=(
            "**The single most important find.** 43.2 h of Kauthuma sāmagāna, uploaded "
            "as 64 clips of 14–46 min (SV001–SV064) plus a 30 min introduction. "
            "Crucially it is *already aligned*: PR Iyer's hand-written gāna "
            "transcriptions print the clip number and a minute:second stamp on the "
            "left of every line (see ``pr_iyer_gana``), so ~15.8 h of Grāma-gāna and "
            "Āraṇyaka-gāna (clips 1–24) come with line-level timings and svara "
            "notation — no forced alignment, no ASR, no manual annotation. That is "
            "more aligned data than Vāgdhenu had in total. Clips 25–64 (27.5 h) are "
            "Ūha/Ūhya, now transcribed in ``sd_uha``/``sd_uhya`` but not yet "
            "time-stamped. Mixed audio quality: these are broadcast-style recordings, "
            "so expect bandwidth limiting and some room tone — measure the noise "
            "floor per clip before you trust a clip."
        ),
    ),
    Source(
        key="pcr_samavedam",
        name="PCR Sāma Vedam — complete Kauthuma recitation, 6 volumes",
        url="https://archive.org/details/SamaVedam",
        kind="audio",
        licence=Licence.UNLICENSED,
        school="kauthuma",
        est_hours=20.1,
        rights_basis=(
            "No licence field on the archive.org item; uploaded 2016 by a third party. "
            "The 'PCR' label and the disc structure point at a published recording, so "
            "treat the *content* as someone's commercial product even though no rights "
            "statement accompanies this copy."
        ),
        rights_url="https://archive.org/metadata/SamaVedam/metadata",
        notes=(
            "20.1 h (17.6 h is sāman proper; the last two volumes are a Rāhasya talk and "
            "a Ṛgveda section). Organised by sāman number — 'sama 1 to 20', '21 to 32', "
            "'33 to 40', '41 to 43', '44 to 46', '47 to 49' — which gives you *coarse* "
            "alignment for free: you know which sāman range a file contains, not where "
            "each one starts. 500 MB per volume, so download selectively on Kaggle: one "
            "volume is a whole training run's worth of audio, and the dataset quota is "
            "the binding constraint, not disk."
        ),
    ),
    Source(
        key="hindu_jagaran_samaveda",
        name="Hindu Jagaran — Sāma Veda (radio-style recitation, 7 parts)",
        url="https://archive.org/details/samaveda",
        kind="audio",
        licence=Licence.UNLICENSED,
        school="kauthuma",
        est_hours=4.0,
        rights_basis="No licence field; uploaded 2010. One part ('1A') is truncated.",
        notes=(
            "~4 h. Overlaps the above in content and is lower quality; useful only for "
            "robustness or if a clip is damaged elsewhere. Weigh it last — it is the "
            "same tradition sung by the same kind of reciter, so it adds little "
            "phonetic diversity for its hours."
        ),
    ),
    Source(
        key="nonesuch_lp",
        name="Nonesuch LP 'The Four Vedas' (1968) — Jaiminīya school",
        url="https://archive.org/details/lp_the-four-vedas-the-oral-traditions-of-hymn_various",
        kind="audio",
        licence=Licence.COMMERCIAL,
        school="jaiminiya",
        est_hours=0.25,
        rights_basis=(
            "Commercially released LP (Nonesuch Explorer Series), still under copyright; "
            "the archive.org copy carries no licence grant."
        ),
        notes=(
            "Tiny in hours but **the only accessible Jaiminīya audio**: the opening hymn "
            "of the Jaiminīya school, plus Bharuṇḍa-sāman and Setu-sāman, recorded "
            "ethnomusicologically with the reciter named. Good enough to sanity-check "
            "that a Jaiminīya model is in the right mode; nowhere near enough to train "
            "one. Do not let its presence imply Jaiminīya coverage."
        ),
    ),
    Source(
        key="ignca_cassettes",
        name="IGNCA — audio cassettes (incl. T.J. Singh on sāmagān and mūrchanā)",
        url="https://archive.org/details/dni.ncaa.IGNCA-TJS_7-AC",
        kind="audio",
        licence=Licence.PERMISSION,
        school="kauthuma",
        est_hours=1.5,
        rights_basis=(
            "Digitised by the National Cultural Audiovisual Archive (IGNCA/NCAA). "
            "Institutional deposit; access is granted, reuse is not."
        ),
        notes=(
            "Lecture material rather than sāmagāna performance — 1:31:52 of Thakur "
            "Jaideva Singh on sāmagān and the mūrchanās. Value is documentary (it "
            "explains the notation), not acoustic. **Needs written permission from "
            "IGNCA**; it is listed here so nobody assumes it is free because it is on "
            "archive.org."
        ),
    ),
    # ============================ text sources ============================= #
    Source(
        key="sd_gramageya",
        name="Sanskrit Documents — Sāmagāna I: Grāma-geya (P.N. Narayanaswami, 2025)",
        url="https://sanskritdocuments.org/doc_veda/gramageya.pdf",
        kind="text",
        licence=Licence.FREE_EDUCATIONAL,
        school="kauthuma",
        est_ganas=1198,
        has_svara_notation=True,
        rights_basis=(
            "PDF is stamped 'PPN © January 2025' — the *typesetting* is the typesetter's "
            "copyright, even though the underlying Veda is ancient. The site provides "
            "its texts under a fair-use declaration 'for individual educational or "
            "research use'. So: free to download and use for teaching; not an open "
            "licence, and not yours to republish."
        ),
        rights_url="https://sanskritdocuments.org/doc_veda/",
        notes=(
            "1,198 gāna, 579 pages, sāma numerals and accents typed by hand into the "
            "Siddhanta font. **The best gāna e-text in existence** and the backbone of "
            "the text side. One practical trap: it is a PDF in a legacy font, so text "
            "extraction gives you glyph soup unless you map the font's Private Use Area "
            "codepoints — budget a day for that, or use ``sd_samved_xlsx`` below for the "
            "saṃhitā and keep the PDF for reference."
        ),
    ),
    Source(
        key="sd_aranyaka",
        name="Sanskrit Documents — Sāmagāna II: Āraṇyaka (296 gāna)",
        url="https://sanskritdocuments.org/doc_veda/aranyakaganam.pdf",
        kind="text",
        licence=Licence.FREE_EDUCATIONAL,
        school="kauthuma",
        est_ganas=296,
        has_svara_notation=True,
        rights_basis="Same terms as sd_gramageya (PPN © typesetting, fair-use download).",
        notes=(
            "296 gāna, 177 pages. This is the book the IISH clips 1–24 finish, so it is "
            "the one to extract first if you want to train against the aligned audio."
        ),
    ),
    Source(
        key="sd_uha",
        name="Sanskrit Documents — Sāmagāna III: Ūha (936 gāna)",
        url="https://sanskritdocuments.org/doc_veda/uhaganam.pdf",
        kind="text",
        licence=Licence.FREE_EDUCATIONAL,
        school="kauthuma",
        est_ganas=936,
        has_svara_notation=True,
        rights_basis="Same terms as sd_gramageya.",
        notes=(
            "936 gāna, 515 pages. Covers IISH clips 25–64, but without the timestamps "
            "the Iyer scans carry — so aligning this book to that audio is the main "
            "unsolved task in the whole project."
        ),
    ),
    Source(
        key="sd_uhya",
        name="Sanskrit Documents — Sāmagāna IV: Ūhya / Rāhasya (209 gāna)",
        url="https://sanskritdocuments.org/doc_veda/uhya-rahasyaganam.pdf",
        kind="text",
        licence=Licence.FREE_EDUCATIONAL,
        school="kauthuma",
        est_ganas=209,
        has_svara_notation=True,
        rights_basis="Same terms as sd_gramageya.",
        notes="209 gāna, 148 pages. Completes the 2,639; the four books plus the saṃhitā are the whole Kauthuma corpus.",
    ),
    Source(
        key="sd_samved_xlsx",
        name="Sanskrit Documents — Kauthuma saṃhitā, 1,875 ṛc in a spreadsheet, with svaras",
        url="https://sanskritdocuments.org/doc_veda/Samved.xlsx",
        kind="text",
        licence=Licence.FREE_EDUCATIONAL,
        school="kauthuma",
        est_verses=1875,
        has_svara_notation=True,
        rights_basis="Same fair-use terms; encoded by Jitendra Bansal.",
        notes=(
            "**Start here, not with the PDFs.** A spreadsheet sortable by devatā, "
            "chandas and svara, with the Vedic accents marked — i.e. the one text source "
            "that goes into the parser without a font-mapping detour. It is the saṃhitā "
            "(ṃantra level), so it supervises accents and syllable counts; the gāna "
            "numerals you still need from the four books."
        ),
    ),
    Source(
        key="pr_iyer_gana",
        name="PR Iyer — hand-written gāna manuscripts with clip numbers and timestamps",
        url="https://archive.org/details/sAmagAnam",
        kind="text",
        licence=Licence.CC_BY_NC_ND,
        school="kauthuma",
        est_hours=15.8,
        est_aligned_hours=15.8,
        has_svara_notation=True,
        rights_basis=(
            "Item metadata states CC BY-NC-ND 4.0 (declared by the uploader, "
            "sanskritdocuments.org, not by the author). ND is the binding clause: "
            "you may not distribute an adapted version."
        ),
        rights_url="https://archive.org/metadata/sAmagAnam/metadata",
        notes=(
            "The Rosetta stone for ``iish_samagana``: Malayalam-script manuscripts in "
            "which **every line carries its audio clip number and a minute:second "
            "stamp**, so the audio and the notation line up without any alignment work. "
            "Clips 1–24 cover Grāma + Āraṇyaka completely. Caveat: it is handwriting, "
            "so it needs transcription before it is training data — and note the "
            "archive.org uploader says the Ūha/Ūhya text 'is not prepared here'. "
            "CC BY-NC-ND: usable for training, cannot ship a derivative."
        ),
    ),
    Source(
        key="pr_iyer_samhita",
        name="PR Iyer — Kauthuma saṃhitā manuscripts, Malayalam, with svaras",
        url="https://archive.org/details/samaveda-malayalam-0001-0114-Agneyam_PR-Iyer",
        kind="text",
        licence=Licence.CC_BY_NC_ND,
        school="kauthuma",
        est_verses=1875,
        has_svara_notation=True,
        rights_basis="Same uploader and terms as pr_iyer_gana.",
        notes=(
            "The saṃhitā side of the same project, with Vedic svaras, in Malayalam "
            "script. Cross-check for the PPN typesetting; also the place to look when a "
            "gāna line and its ṛc disagree."
        ),
    ),
    Source(
        key="vhp_kauthuma",
        name="Vedic Heritage Portal — Kauthuma saṃhitā (Pūrvārcika / Uttarārcika)",
        url="https://vedicheritage.gov.in/samhitas/samaveda-samhitas/kauthuma-samhita/",
        kind="text",
        licence=Licence.OPEN,
        school="kauthuma",
        est_verses=1875,
        has_svara_notation=True,
        rights_basis="Government of India portal; text published openly.",
        notes=(
            "The national portal the brief named. Text is the primary reference for the "
            "saṃhitā and is the safest thing on this list to cite. Audio on the same "
            "portal is a different matter — see ``vhp_audio``."
        ),
    ),
    Source(
        key="vhp_audio",
        name="Vedic Heritage Portal — Sāmaveda recitation audio",
        url="https://vedicheritage.gov.in/samaveda/",
        kind="audio",
        licence=Licence.PERMISSION,
        school="kauthuma",
        est_hours=6.0,
        rights_basis=(
            "Published for streaming on a government portal. No open licence, no "
            "download endpoint, no reuse terms. Listening is unrestricted; reuse is not "
            "granted."
        ),
        notes=(
            "~6 h of Kauthuma sāmagāna by named uḍgātṛs — the closest thing to a "
            "canonical reference, and much cleaner than the archive.org rips. Two "
            "honest paths: (a) write for permission, citing an educational, "
            "non-distributed project — plausible and worth a week's wait; (b) use 30 "
            "clips as a held-out evaluation set you never publish. Do not scrape it and "
            "call it fair use."
        ),
    ),
    Source(
        key="vhp_jaiminiya",
        name="Vedic Heritage Portal — Jaiminīya Sāmaveda, Kerala style (ṛc and gāna)",
        url="https://vedicheritage.gov.in/samaveda/",
        kind="audio",
        licence=Licence.PERMISSION,
        school="jaiminiya",
        rights_basis="Same portal, same terms as vhp_audio.",
        notes=(
            "The only substantial Jaiminīya collection: ṛc and gāna forms in Kerala "
            "style, everything except Ūha and Rāhasya-Ūhya. Hours unmeasured — the "
            "portal renders its index in JavaScript and could not be enumerated from "
            "here. **This is the one source worth writing a letter for**, because the "
            "Jaiminīya school is otherwise a quarter-hour of audio on a 1968 LP."
        ),
    ),
    # ==================== mirrors and duplicates (not counted) ============= #
    Source(
        key="shaivam_samaveda",
        name="shaivam.org — Sāma Veda audio gallery",
        url="https://www.shaivam.org/audio-gallery/sama-veda/",
        kind="audio",
        licence=Licence.UNLICENSED,
        school="kauthuma",
        duplicate_of="pcr_samavedam",
        notes="A third complete Kauthuma set, or a mirror of one above. Counted once in the totals until you diff the audio.",
    ),
    Source(
        key="astrojyoti_samaveda",
        name="astrojyoti.com — Sāmaveda MP3s (64 files, 292 MB)",
        url="https://www.astrojyoti.com/samavedamp3part1.htm",
        kind="audio",
        licence=Licence.UNLICENSED,
        school="kauthuma",
        duplicate_of="pcr_samavedam",
        notes="64 files, 292 MB. Almost certainly the same performance as one of the sets above; treat as a fallback mirror only.",
    ),
    Source(
        key="vedaprasar",
        name="Veda Prasar Samiti — Sāma Veda on 4 audio CDs",
        url="http://www.vedaprasarsamiti.org/",
        kind="audio",
        licence=Licence.COMMERCIAL,
        school="kauthuma",
        duplicate_of="pcr_samavedam",
        notes="Sold, not given. Listed because it is very likely the origin of the 'PCR' rip above — which is the clearest illustration of why those rips are unlicensed.",
    ),
    Source(
        key="youtube_samagana",
        name="YouTube / Vimeo — sāmagāna uploads (incl. Rāṇāyanīya ritual video)",
        url="https://vimeo.com/channels/vedicritual",
        kind="audio",
        licence=Licence.UNCLEAR,
        school="ranayaniya",
        est_hours=20.0,
        rights_basis="Per-video, usually unstated. Many are re-uploads of commercial recordings.",
        notes=(
            "About 20 h by the rough count, and the only Rāṇāyanīya material: chant by "
            "Kale and Joshi including SV 1.139 with notation shown. **Blocked in every "
            "profile, including educational** — not because teaching is restricted but "
            "because you cannot name the rights holder, and an upload of someone else's "
            "CD is not a source. Use it to learn what the third school sounds like; do "
            "not put it in a manifest."
        ),
    ),
    # ============================ scholarship ============================== #
    Source(
        key="howard_1988",
        name="Wayne Howard — Decipherment of the Sāmavedic Notation of the Jaiminīyas (1988)",
        url="https://journal.fi/store/issue/view/3442",
        kind="reference",
        licence=Licence.FREE_EDUCATIONAL,
        school="jaiminiya",
        rights_basis="Studia Orientalia 63, free PDF from the Finnish Oriental Society. Free to read; not CC.",
        notes=(
            "The book that decodes the Jaiminīya notation — the only way a Jaiminīya "
            "model can be *notation-driven* rather than imitation. Free, which is "
            "unusual for scholarship this useful. Read §2 before touching any "
            "Jaiminīya data."
        ),
    ),
    Source(
        key="howard_1977",
        name="Wayne Howard — Sāmavedic Chant (Yale, 1977)",
        url="https://yalebooks.yale.edu/",
        kind="reference",
        licence=Licence.COMMERCIAL,
        school="",
        notes=(
            "The standard musicological survey: all schools, notation, and transcriptions "
            "of actual performances. In copyright, and expensive. Find it in a library — "
            "it is the reference the whole field cites, and worth an afternoon."
        ),
    ),
    # ============================== your own =============================== #
    Source(
        key="own_recordings",
        name="Your own recordings (the only corpus you own outright)",
        url="",
        kind="audio",
        licence=Licence.OPEN,
        est_hours=0.0,
        est_aligned_hours=0.0,
        rights_basis="Yours, if the reciter signed the consent form in docs/DATA.md §6.",
        notes=(
            "Five hours of consented, single-singer sāmagāna with breath-group timings "
            "beats every other source on this list, because it is the only one you can "
            "publish. Protocol in docs/DATA.md. Everything else here is scaffolding for "
            "the model you build while you wait."
        ),
        redistribute=True,
    ),
]


def by_key(key: str) -> Source:
    for s in SOURCES:
        if s.key == key:
            return s
    raise KeyError(key)


def counted(profile: str = DEFAULT_PROFILE) -> list[Source]:
    """Sources that count toward the totals: no mirrors, no references."""
    return [s for s in SOURCES if not s.duplicate_of and s.kind in ("audio", "both", "text")]


def audit(verbose: bool = True, profile: str = DEFAULT_PROFILE) -> dict:
    """Summarise what exists, what is licensed, and what you may actually do.

    Two tables, deliberately: the first is the licence facts, the second is the
    answer under an intended-use profile. Collapsing them is how projects end up
    claiming 'open data' they do not have.
    """
    pool = counted()
    text_sources = [s for s in pool if s.kind in ("text", "both")]
    audio_sources = [s for s in pool if s.kind in ("audio", "both")]

    strict_audio = [s for s in audio_sources if s.usable_for_training]
    reusable_audio = [s for s in audio_sources if s.usable_for_release]
    permits = {s.key: s.permit(profile) for s in pool}

    trainable = [s for s in audio_sources if permits[s.key].may_train]
    hours_any_corpus = sum(s.est_hours for s in audio_sources)
    hours_trainable = sum(s.est_hours for s in trainable)
    hours_strict = sum(s.est_hours for s in strict_audio)
    hours_shippable = sum(s.est_hours for s in reusable_audio)

    report = {
        "profile": profile,
        "text_sources": len(text_sources),
        "audio_sources": len(audio_sources),
        "text_with_svara": [s.key for s in text_sources if s.has_svara_notation],
        "total_ganas_in_text": sum(s.est_ganas for s in text_sources),
        # The saṃhitā is 1,875 ṛc and it is available from four providers. Summing
        # them would count the same verses four times; take the best single copy.
        "total_rc_in_text": max((s.est_verses for s in text_sources), default=0),
        "audio_hours_available": hours_any_corpus,
        "audio_hours_trainable": hours_trainable,
        "audio_hours_trainable_strict": hours_strict,
        "audio_hours_releasable": hours_shippable,
        "audio_hours_aligned": sum(s.est_aligned_hours for s in trainable),
        "may_train": [s.key for s in audio_sources if permits[s.key].may_train],
        "may_not_train": [s.key for s in audio_sources if not permits[s.key].may_train],
        # `blocked` keeps its old meaning: provenance is the problem.
        "blocked": [s.key for s in SOURCES if s.licence is Licence.UNCLEAR],
        "needs_permission": [s.key for s in SOURCES if s.licence is Licence.PERMISSION],
        "not_shippable": [
            s.key for s in audio_sources
            if permits[s.key].may_train and not permits[s.key].may_ship_weights
        ],
    }

    if verbose:
        print(f"{'source':24s} {'kind':6s} {'licence':17s} {'hrs':>6s} {'algn':>5s}  strict")
        print("-" * 80)
        for s in SOURCES:
            if s.kind == "reference":
                continue
            flag = "Y" if s.usable_for_training else ("mirror" if s.duplicate_of else "-")
            print(f"{s.key:24s} {s.kind:6s} {s.licence.value:17s} "
                  f"{s.est_hours:6.1f} {s.est_aligned_hours:5.1f}  {flag:>6s}")

        p = PROFILES[profile]
        print("-" * 80)
        print(f"licence facts: {hours_shippable:.1f} h of sāmagāna audio is openly "
              f"licensed ({hours_strict:.1f} h trainable-and-shippable)")
        summary = (f"{report['total_ganas_in_text']:,} gāna + "
                   f"{report['total_rc_in_text']:,} ṛc, "
                   f"{len(report['text_with_svara'])} sources with svara notation")
        if report["total_rc_in_text"] > 1:
            summary += " (the saṃhitā exists in 4 copies; counted once)"
        print(f"text: {summary}")

        print()
        print(f"under the '{profile}' profile — {p.name}")
        print(f"  trainable audio: {hours_trainable:.1f} h "
              f"({report['audio_hours_aligned']:.1f} h with line-level timings)")
        if report["not_shippable"]:
            print(f"  of which {sum(by_key(k).est_hours for k in report['not_shippable']):.1f} h "
                  f"may train your model but not ship inside it "
                  f"({len(report['not_shippable'])} source(s))")
        if report["blocked"]:
            print(f"  BLOCKED on provenance: {', '.join(report['blocked'])}")
        if report["needs_permission"]:
            print(f"  needs written permission: {', '.join(report['needs_permission'])}")
        print()
        print("  " + p.note)
        print()
        if profile == "open":
            print("  Anything marked blocked above stays blocked: for those sources the\n"
                  "  problem is that the rights holder cannot be named, which no licence\n"
                  "  or promise can fix. Release only what this profile leaves trainable.")
        else:
            print("  This profile is a promise you are making, not a licence you were "
                  "granted. Publish\n  a checkpoint or host the audio and every "
                  "non-distributable source above flips\n  to blocked — re-run "
                  "`audit --profile open` before you make anything public.")
    return report


if __name__ == "__main__":
    import sys

    audit(profile=sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROFILE)
