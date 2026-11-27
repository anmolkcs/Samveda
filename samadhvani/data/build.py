"""Build the training corpus from gāna text plus whatever audio you have.

Two jobs:

1. **Text-side corpus** (always possible, no audio needed).  Parse the gāna books
   into aligned (text, phone, svara) records.  This is enough to train the duration
   model and the melody side, and it is a large, clean, fully-licensed dataset.

2. **Audio-side manifest** (needs you to supply recordings).  For each recording we
   need a transcript and a rough alignment.  Because Sāma recitation is syllable-
   timed and the svara notation tells us how long each syllable is held, we can do
   **akṣara-level forced alignment without a frame-level aligner**: the timing plan
   from the text predicts each akṣara's duration proportionally, and we only need
   to find where each line starts.  A single manual line-start timestamp per breath
   group is a 30-second job for a 10-minute recording, versus hours for phone-level
   forced alignment.

The output is plain JSONL so it can be inspected, diffed, and shipped:

``manifest.jsonl``
    one row per utterance: ``{audio, text, phones, svaras, durations, mode, source, licence}``
``models_learned.json``
    measured svara contours per sāman, in the schema
    :func:`samadhvani.models.sama_models.load_learned` reads.
``text_corpus.jsonl``
    the text-only corpus, usable with no audio at all.

Everything is deterministic given the input files, so re-running does not churn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..models.sama_models import MODELS, REGISTER_SHIFT
from ..phones.g2p import phones_for_view
from ..pipeline import ChantOptions
from ..text.svaras import SAMA_SVARAS
from ..text.syllabify import syllabify
from ..text.unicodedata_ext import detect_script, normalize, to_devanagari
from ..text.vikara import SCHOOLS, apply_vikaras, parse_gana
from ..synth.timing import TimingOptions, plan_timing


@dataclass
class Record:
    """One supervised example."""

    audio: str = ""
    text: str = ""
    phones: list[list[str]] = field(default_factory=list)
    svaras: list[int] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    f0_hz: list[float] = field(default_factory=list)
    mode: str = "text"
    source: str = ""
    licence: str = "open"
    school: str = "kauthuma"
    model: str = "gramageya"
    chanda: str = ""
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = self.__dict__.copy()
        return json.dumps(d, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Text corpus
# --------------------------------------------------------------------------- #

def records_from_text(text: str, *, school: str = "kauthuma", model: str = "gramageya",
                      source: str = "unknown", licence: str = "open") -> list[Record]:
    """Turn a gāna text (or a plain verse) into phone+svara records."""
    deva = normalize(to_devanagari(text))
    mode = "gana"
    svaras: list[int] = []
    chanda = ""
    if any(ch in deva for ch in "१२३४५६७२३४५६"):
        gana = parse_gana(deva, school=school)
        chanda = gana.chanda
        svaras = [s or 0 for s in gana.flat_svaras()]
        view = syllabify(deva, svara_map=svaras)
    else:
        mode = "samhita" if any(c in deva for c in "॒॑᳚") else "plain"
        view = syllabify(deva)
    if not view.aksaras:
        return []
    # Fill any unparsed syllable from the accent/the default skeleton, so a corpus
    # record never carries a 0 note.  A 0 here is contagious: it survives into the
    # measured contours and from there into a trained model.
    if len(svaras) < len(view.aksaras):
        from ..text.svaras import annotate_svaras
        scheme = "samhita" if mode == "samhita" else "tradition"
        toks = annotate_svaras(view, scheme=scheme)
        fallback = [t.svara for t in toks]
        svaras = [
            (svaras[i] if i < len(svaras) and svaras[i] else fallback[i])
            for i in range(len(view.aksaras))
        ]
    svaras = [int(s) if s and 1 <= int(s) <= 7 else 4 for s in svaras[: len(view.aksaras)]]

    ph = phones_for_view(view)
    plan = plan_timing(view.aksaras, svaras, groups=view.groups,
                       options=TimingOptions(), phone_lists=[p.phones for p in ph])
    return [Record(
        text=deva, phones=[p.phones for p in ph], svaras=svaras,
        durations=[round(n.duration, 4) for n in plan.notes],
        f0_hz=[round(n.f0_hz, 2) for n in plan.notes],
        mode=mode, source=source, licence=licence, school=school, model=model,
        chanda=chanda,
        meta={"syllables": len(view.aksaras), "groups": len(view.groups)},
    )]


def build_text_corpus(paths: Iterable[str], out: str, *, school: str = "kauthuma",
                      model: str = "gramageya") -> int:
    """Parse ``paths`` (UTF-8 text files, one verse/gāna per line) into JSONL."""
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for p in paths:
            src = os.path.basename(p)
            with open(p, encoding="utf-8") as inp:
                for line in inp:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    for rec in records_from_text(line, school=school, model=model,
                                                 source=src):
                        fh.write(rec.to_json() + "\n")
                        n += 1
    return n


# --------------------------------------------------------------------------- #
# Audio manifest with akṣara-proportional alignment
# --------------------------------------------------------------------------- #

@dataclass
class Clip:
    audio: str
    text: str
    line_starts: list[float] = field(default_factory=list)
    school: str = "kauthuma"
    model: str = "gramageya"
    licence: str = "open"
    source: str = ""


def _load_clip_file(path: str) -> Clip:
    """A clip is a tiny JSON sidecar next to the audio.

    ``{"text": "…", "line_starts": [0.0, 4.2, 9.8], "school": "kauthuma",
       "licence": "open", "source": "…"}``
    """
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    return Clip(
        audio=blob.get("audio", path.replace(".json", ".wav")),
        text=blob["text"],
        line_starts=blob.get("line_starts", []),
        school=blob.get("school", "kauthuma"),
        model=blob.get("model", "gramageya"),
        licence=blob.get("licence", "open"),
        source=blob.get("source", ""),
    )


def align_clip(clip: Clip, *, total_seconds: float | None = None) -> Record:
    """Express a recording as per-akṣara durations.

    If ``line_starts`` is empty the clip is assumed to be one breath group starting
    at 0.  Otherwise each line's span is divided among its akṣaras in proportion to
    the mātrās the text specifies — which, because Sāma timing really is mātrā-based,
    lands within a few percent of what a frame aligner would give.
    """
    deva = normalize(to_devanagari(clip.text))
    view = syllabify(deva)
    if not view.aksaras:
        raise ValueError("clip has no syllables")
    ph = phones_for_view(view)
    ph_list = [p.phones for p in ph]

    # svara from notation if present, else the school's default skeleton
    flat: list[int] = []
    if any(ch in deva for ch in "१२३४५६७२३४५६"):
        g = parse_gana(deva, school=clip.school)
        flat = [s or 4 for s in g.flat_svaras()]
    if len(flat) < len(view.aksaras):
        from ..text.svaras import annotate_svaras
        toks = annotate_svaras(view, scheme="tradition")
        flat = [t.svara for t in toks]
    flat = flat[: len(view.aksaras)]

    # ideal (mātrā-proportional) durations
    ideal = plan_timing(view.aksaras, flat, groups=view.groups,
                        options=TimingOptions(), phone_lists=ph_list)

    # group the ideal notes by breath group so we can rescale per line
    durations: list[float] = []
    groups = view.groups or [list(range(len(view.aksaras)))]
    starts = list(clip.line_starts) or [0.0]
    if total_seconds is not None and len(starts) < len(groups):
        # no line timings: distribute total time by the ideal proportions
        total_ideal = ideal.total_seconds or 1.0
        scale = total_seconds / total_ideal
        durations = [round(n.duration * scale, 4) for n in ideal.notes]
    else:
        note_i = 0
        for gi, group in enumerate(groups):
            if note_i >= len(ideal.notes):
                break
            seg_notes = ideal.notes[note_i: note_i + len(group)]
            note_i += len(group)
            if gi + 1 < len(starts) and len(starts) > gi + 1:
                span = starts[gi + 1] - starts[gi]
            elif total_seconds is not None:
                span = total_seconds - starts[gi]
            else:
                span = sum(n.duration for n in seg_notes)
            ideal_span = sum(n.duration for n in seg_notes) or 1.0
            scale = span / ideal_span
            durations.extend(round(n.duration * scale, 4) for n in seg_notes)
        if len(durations) < len(view.aksaras):
            durations.extend([0.0] * (len(view.aksaras) - len(durations)))

    return Record(
        audio=clip.audio, text=deva, phones=ph_list, svaras=flat,
        durations=durations,
        f0_hz=[round(n.f0_hz, 2) for n in ideal.notes],
        mode="audio", source=clip.source, licence=clip.licence,
        school=clip.school, model=clip.model,
        meta={"line_starts": starts, "n_groups": len(groups)},
    )


def build_audio_manifest(clip_dir: str, out: str) -> int:
    """Read every ``*.json`` sidecar in ``clip_dir`` into a manifest."""
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for p in sorted(Path(clip_dir).glob("*.json")):
            clip = _load_clip_file(str(p))
            try:
                rec = align_clip(clip)
            except Exception as exc:           # keep going: one bad clip is fine
                print(f"  ! {p.name}: {exc}")
                continue
            fh.write(rec.to_json() + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Learned melody contours
# --------------------------------------------------------------------------- #

def measure_contours(manifest: str, out: str) -> int:
    """Average the svara sequences seen per (school, model) into model contours.

    This is how the hand-written skeletons in :mod:`samadhvani.models.sama_models`
    get replaced by measured ones.  The schema matches what
    :func:`load_learned` expects, so the pipeline picks the file up by path.
    """
    from collections import defaultdict

    buckets: dict[str, list[list[int]]] = defaultdict(list)
    dropped = 0
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            # Keep only valid sāma numbers.  A 0 (an unparsed syllable) must never
            # reach the contour average, or it propagates into a trained model as a
            # "note below the scale" that nothing can reach.
            seq = [int(s) for s in (rec.get("svaras") or []) if s and 1 <= int(s) <= 7]
            if not seq:
                dropped += 1
                continue
            key = f"{rec.get('school', 'kauthuma')}_{rec.get('model', 'gramageya')}"
            buckets[key].append(seq)
    if dropped:
        print(f"  ! skipped {dropped} records with no valid svara")

    models: dict[str, dict] = {}
    for key, seqs in buckets.items():
        # crude but effective: resample every sequence to a fixed length and take
        # the per-slot median svara.  Medians are robust to a mis-parsed numeral.
        import statistics

        L = 24
        grid = [[] for _ in range(L)]
        for seq in seqs:
            if not seq:
                continue
            for j in range(L):
                grid[j].append(seq[min(len(seq) - 1, int(j / L * len(seq)))])
        med = [min(7, max(1, int(round(statistics.median(g))))) if g else 4 for g in grid]
        # split into the five traditional sections
        sec_len = max(1, L // 5)
        sections = []
        for s in range(0, L, sec_len):
            chunk = med[s: s + sec_len]
            if chunk:
                sections.append([{"svara": v, "matra": 1.5 if i == len(chunk) - 1 else 1.0}
                                 for i, v in enumerate(chunk)])
        models[key] = {
            "name_deva": key, "name_iast": key, "source_book": "measured",
            "notes": f"measured from {len(seqs)} utterances", "sections": sections,
        }

    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "models": models}, fh, ensure_ascii=False, indent=1)
    return len(models)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="samadhvani.data.build",
                                 description="Build Sāma chant training corpora.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("text", help="parse gāna/verse text files into a corpus")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--out", default="manifest.jsonl")
    p.add_argument("--school", default="kauthuma")

    p = sub.add_parser("audio", help="align clip sidecars into a manifest")
    p.add_argument("clip_dir")
    p.add_argument("-o", "--out", default="manifest.jsonl")

    p = sub.add_parser("contours", help="measure learned melodic contours")
    p.add_argument("manifest")
    p.add_argument("-o", "--out", default="models_learned.json")

    p = sub.add_parser("audit", help="show what data exists and what may be used")
    p.add_argument("--profile", default=None,
                   help="intended-use profile: educational (default), open, evaluation")
    args = ap.parse_args(argv)

    if args.cmd == "text":
        n = build_text_corpus(args.inputs, args.out, school=args.school)
        print(f"wrote {n} records to {args.out}")
    elif args.cmd == "audio":
        n = build_audio_manifest(args.clip_dir, args.out)
        print(f"wrote {n} records to {args.out}")
    elif args.cmd == "contours":
        n = measure_contours(args.manifest, args.out)
        print(f"wrote {n} learned models to {args.out}")
    elif args.cmd == "audit":
        from .sources import DEFAULT_PROFILE, audit
        audit(profile=args.profile or DEFAULT_PROFILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
