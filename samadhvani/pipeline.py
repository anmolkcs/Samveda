"""The pipeline: verse in, chant out.

    verse ──▶ script normalisation ──▶ akṣara segmentation ──▶ svara plan
          ──▶ sāma-vikāra generation ──▶ phone sequence ──▶ timing ──▶ audio

Almost every design decision in this file is about *honesty*.  A sāmagāna TTS can
operate in three increasingly faithful modes, and it should always be obvious which
one produced what you are hearing:

``mode="style"``
    The user pasted a plain verse.  We apply the documented vikāras and lay the
    result on a gāna model's melodic skeleton.  This is a *stylised* recitation with
    the right phonetics and the right musical grammar — **not** the sāman that the
    tradition would actually sing for that ṛc.  The output says so.

``mode="samhita"``
    The text carries Vedic accents (``॒`` ``॑``).  The three-tone accent map is
    honoured directly, which is what a ṛc recitation actually does.  Faithful for
    recitation, not for gāna.

``mode="gana"``
    The text is an actual gāna text with sāma notation.  Nothing is invented: the
    svara numerals, prolongations and stobhas are read out of the text.

The distinction matters because the alternative — cheerfully claiming to render
"the" Sāmaveda chant for any Sanskrit verse — is the failure mode of a lot of
heritage-tech.  The metadata on :class:`Chant` records the mode, the vikāras applied,
and a plain-language caveat.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .models.sama_models import (
    MODELS,
    REGISTER_SHIFT,
    get_model,
    model_keys,
    tune_to_syllables,
)
from .phones.g2p import phones_for_view
from .synth.engine import synthesize_formant
from .synth.io import write_wav
from .synth.prosody import f0_track
from .synth.timing import TimingOptions, plan_timing
from .text.svaras import SAMA_SVARAS, annotate_svaras
from .text.syllabify import syllabify
from .text.unicodedata_ext import Script, detect_script, normalize, to_devanagari
from .text.vikara import SCHOOLS, VikaraOptions, apply_vikaras, classify_vikaras, parse_gana

SR = 22050


@dataclass
class ChantOptions:
    """Everything the demo's controls map onto."""

    #: "avoid" | "style" | "samhita" | "gana"; "avoid" picks the most faithful
    #: available mode for the text automatically.
    mode: str = "auto"
    #: sāma school — changes stobhas and register.
    school: str = "kauthuma"
    #: gāna model to lay the melody on (gramageya, aranyaka, uha, uhya, mahanamni).
    model: str = "gramageya"
    #: Seconds per mātrā: 0.30 brisk, 0.42 ceremonial, 0.55 very slow.
    laya: float = 0.42
    #: Tonic / Sa in Hz. 146.83 = D3, a comfortable male chant register.
    tonic_hz: float = 146.83
    #: Semitone transposition applied on top of the tonic.
    transpose: float = 0.0
    #: Strength of the stylisation when mode == "style".
    intensity: float = 0.8
    #: Room reverb amount 0..0.5.
    room: float = 0.18
    #: Extra breath length in seconds.
    breath: float = 0.20
    #: Which engine: "formant" (built in), "voice" (trained voice colour, needs a
    #: checkpoint from ``samadhvani.train.voice_trainer``), or "neural" (legacy v1).
    engine: str = "formant"
    #: Path to a trained checkpoint for engine="neural".
    checkpoint: str | None = None
    seed: int = 7


@dataclass
class NoteInfo:
    text: str
    phones: list[str]
    svara: int
    svara_name: str
    f0_hz: float
    start: float
    duration: float


@dataclass
class Chant:
    """The result: audio plus a complete, inspectable account of how it was made."""

    audio: np.ndarray
    sr: int
    text_devanagari: str
    mode: str
    school: str
    model: str
    chanda: str = ""
    svaras: list[int] = field(default_factory=list)
    syllables: list[str] = field(default_factory=list)
    phones: list[list[str]] = field(default_factory=list)
    notes: list[NoteInfo] = field(default_factory=list)
    vikaras: dict[str, bool] = field(default_factory=dict)
    #: The absolute frequency of Sa (sāma svara 4) actually used.  This must be
    #: carried explicitly: the first note is *not* necessarily the tonic, and an
    #: evaluator that assumes it is produces a constant cents offset equal to the
    #: interval between the opening note and Sa.
    tonic_hz: float = 146.83
    caveats: list[str] = field(default_factory=list)
    render_ms: float = 0.0

    @property
    def duration(self) -> float:
        return len(self.audio) / float(self.sr)

    def save(self, path: str) -> str:
        write_wav(path, self.audio, self.sr)
        return path

    def to_dict(self) -> dict:
        return {
            "text": self.text_devanagari,
            "mode": self.mode,
            "school": self.school,
            "model": self.model,
            "chanda": self.chanda,
            "duration": round(self.duration, 3),
            "sr": self.sr,
            "render_ms": round(self.render_ms, 1),
            "syllables": self.syllables,
            "svaras": self.svaras,
            "vikaras": self.vikaras,
            "caveats": self.caveats,
            "notes": [
                {
                    "text": n.text, "phones": n.phones, "svara": n.svara,
                    "svara_name": n.svara_name, "f0_hz": round(n.f0_hz, 2),
                    "start": round(n.start, 3), "duration": round(n.duration, 3),
                }
                for n in self.notes
            ],
        }


# --------------------------------------------------------------------------- #
# Mode detection
# --------------------------------------------------------------------------- #

def _looks_like_gana(text: str) -> bool:
    """Is this an actual gāna text — i.e. does it carry sāma *notation*?

    The only reliable signal is the notation itself: the svara numerals printed
    against the syllables.  A text can carry stobhas ("hāu", "hovā") and still be a
    saṃhitā or a plain verse with a stobha pasted in front, so stobha presence is
    **not** sufficient.  Getting this wrong matters: claiming gāna mode for a text
    with no numerals would mean claiming the notes were read from the text when
    they were invented, which is exactly the dishonesty the modes exist to prevent.
    """
    import re

    # A Devanāgarī svara numeral next to a syllable, or a run of pluta marks.
    if re.search(r"[क-ह][ािीुूृेैोौंँः॒॑]?\s*[१-७]", text):
        return True
    if re.search(r"[१-७]\s*[क-ह]", text):
        return True
    if len(re.findall(r"[२३४५६]", text)) >= 2:
        return True
    return False


def _has_vedic_accents(text: str) -> bool:
    return any(ch in text for ch in "॒॑᳚")


def _resolve_mode(text: str, requested: str) -> tuple[str, list[str]]:
    caveats: list[str] = []
    if requested != "auto" and requested != "avoid":
        return requested, caveats
    if _looks_like_gana(text):
        return "gana", caveats
    if _has_vedic_accents(text):
        return "samhita", caveats
    return "style", caveats


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def chant(text: str, options: ChantOptions | None = None, **overrides) -> Chant:
    """Render a Vedic verse as a chant.

    ``overrides`` are convenience keyword arguments that update ``options``.
    """
    t0 = time.time()
    opts = options or ChantOptions()
    if overrides:
        opts = ChantOptions(**{**opts.__dict__, **overrides})
    if opts.school not in SCHOOLS:
        raise ValueError(f"unknown school {opts.school!r}; have {sorted(SCHOOLS)}")
    if opts.model not in MODELS:
        raise ValueError(f"unknown model {opts.model!r}; have {model_keys()}")

    caveats: list[str] = []

    # 1. normalise the script
    script = detect_script(text)
    deva = to_devanagari(text)
    deva = normalize(deva)
    if script.value not in ("devanagari", "unknown"):
        caveats.append(f"Input script {script.value} was transliterated to Devanāgarī.")

    # 2. decide how faithful we can be
    mode, mode_caveats = _resolve_mode(text, opts.mode)
    caveats.extend(mode_caveats)

    if mode == "gana":
        gana = parse_gana(deva, school=opts.school)
        flat = gana.flat_svaras()
        view = syllabify(deva, svara_map=[s for s in flat])
        svaras = [s or 4 for s in flat]
        if len(svaras) < len(view.aksaras):
            svaras += [4] * (len(view.aksaras) - len(svaras))
        svaras = svaras[: len(view.aksaras)]
        chanda = gana.chanda
        caveats.append(
            "Gāna mode: svaras were read from the notation in the text; nothing was invented."
        )
    elif mode == "samhita":
        view = syllabify(deva)
        tokens = annotate_svaras(view, scheme="samhita")
        svaras = [t.svara for t in tokens]
        chanda = ""
        caveats.append(
            "Saṃhitā mode: the three-tone Vedic accents in the text were mapped onto the "
            "sāma scale. This is recitation (pāṭha), not gāna."
        )
    else:  # style
        view = syllabify(deva)
        gen = apply_vikaras(
            deva,
            VikaraOptions(school=opts.school, intensity=opts.intensity),
        )
        chanda = gen.chanda
        styled = " ".join(line.text for line in gen.lines)
        styled_view = syllabify(styled)
        if len(styled_view.aksaras) >= 3:
            view = styled_view
            deka = styled
        # Notes come from the gāna model's melodic skeleton, resampled to syllable count.
        svara_events = tune_to_syllables(get_model(opts.model), len(view.aksaras))
        svaras = [e.svara for e in svara_events]
        caveats.append(
            "Styled mode: the verse was dressed in the documented sāma-vikāras and laid on "
            f"the {get_model(opts.model).name_iast} melodic skeleton. This is a faithful "
            "*style* — phonetics and musical grammar — but it is not the sāman that this "
            "ṛc's own śākhā would sing. Paste a gāna text (with svara numerals) for exact "
            "recitation."
        )

    if not view.aksaras:
        raise ValueError("no syllables found — the input may not be Sanskrit text")

    # Guard against a Latin sentence being handed in as if it were Sanskrit.
    # Transliteration makes this impossible to detect from the *output* — "hello"
    # transliterates happily into हेल्लो — so it has to be decided from the input's
    # alphabet.  Sanskrit's Latin inventory is small and closed: it has no w, x, f,
    # z or q (a "v" carries that role), and no doubled consonants beyond a genuine
    # cluster.  Anything outside it is not Sanskrit.
    if script in (Script.LATIN, Script.IAST):
        stray = sorted({c.lower() for c in text
                        if c.isalpha() and c.lower() not in _SANSKRIT_LATIN})
        if stray:
            raise ValueError(
                "this does not look like Sanskrit: the input contains "
                f"{', '.join(repr(c) for c in stray[:5])}, which are outside the "
                "Sanskrit transliteration alphabet. Supply Devanāgarī, IAST with "
                "diacritics, or another Indic script."
            )

    # 3. phones
    ph = phones_for_view(view)
    phone_lists = [p.phones for p in ph]

    # 4. timing
    register = REGISTER_SHIFT.get(opts.model, 0.0) + SCHOOLS[opts.school].register_shift
    timing_opts = TimingOptions(
        matra_seconds=opts.laya,
        tonic_hz=opts.tonic_hz,
        register_shift_semitones=register + opts.transpose,
        breath=opts.breath,
    )
    plan = plan_timing(view.aksaras, svaras, groups=view.groups,
                       options=timing_opts, phone_lists=phone_lists)

    # 5. render
    if opts.engine == "voice" and not opts.checkpoint:
        caveats.append("No checkpoint given for the voice engine; fell back to formant. "
                       "Train one with samadhvani.train.voice_trainer.")
    if opts.engine == "voice" and opts.checkpoint:
        # The trained path: render the rule-engine chant first, then correct its
        # spectral envelope with the learned voice.  Rendering twice would give two
        # different timelines and the reported note times would stop being true.
        from .synth.voice_render import render_voice

        base, sr = synthesize_formant(plan, phone_lists, seed=opts.seed,
                                      breath=opts.breath, room=0.0)
        audio, sr = render_voice(plan, phone_lists, opts.checkpoint,
                                 base_audio=base, seed=opts.seed, room=opts.room)
        caveats.append(
            "Voice engine: melody from the svara notation, timbre from the trained "
            "corpus, phase from harmonic resynthesis. Not a neural vocoder — expect "
            "a slight edge on sustained notes.")
    elif opts.engine == "neural" and opts.checkpoint:
        from .synth.neural_render import render_neural

        audio, sr = render_neural(plan, phone_lists, opts.checkpoint, seed=opts.seed)
    else:
        if opts.engine == "neural" and not opts.checkpoint:
            caveats.append("No checkpoint given for the neural engine; fell back to formant.")
        audio, sr = synthesize_formant(plan, phone_lists, seed=opts.seed,
                                       breath=opts.breath, room=opts.room)

    tonic_used = timing_opts.tonic_hz * 2.0 ** (
        (timing_opts.register_shift_semitones) / 12.0)

    notes = [
        NoteInfo(
            text=n.text, phones=n.phones, svara=n.svara,
            svara_name=SAMA_SVARA_NAME.get(n.svara, str(n.svara)),
            f0_hz=n.f0_hz, start=n.start, duration=n.duration,
        )
        for n in plan.notes
    ]

    vikaras = classify_vikaras(deva, " ".join(a.text for a in view.aksaras))

    return Chant(
        audio=audio, sr=sr, text_devanagari=deva, mode=mode, school=opts.school,
        model=opts.model, chanda=chanda, svaras=svaras,
        syllables=[a.text for a in view.aksaras], phones=phone_lists,
        notes=notes, vikaras=vikaras, caveats=caveats,
        tonic_hz=tonic_used,
        render_ms=(time.time() - t0) * 1000.0,
    )


SAMA_SVARA_NAME = {k: v["iast"] for k, v in SAMA_SVARAS.items()}

#: The closed Latin inventory of Sanskrit transliteration (IAST/ASCII/ISO-15919).
_SANSKRIT_LATIN = set("aāiīuūṛṝḷḹeoêôṃḥṁŋñṅṭḍṇśṣḻkvgyrcjtdnpbmlsh")


def chant_to_file(text: str, path: str, **kw) -> Chant:
    c = chant(text, **kw)
    c.save(path)
    return c
