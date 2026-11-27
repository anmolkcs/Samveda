"""The command line.

    samadhvani chant "हाउ अ॒ग्निमी॑ळे …" -o chant.wav
    samadhvani chant verse.txt -o out.wav --mode style --school ranayaniya
    samadhvani analyze "ॐ नमो वैश्वानराय"          # phones, svaras, vikāras
    samadhvani eval chant.wav --text "…"             # score a rendered file
    samadhvani audit                                 # what data may be used
    samadhvani serve --port 8000                     # the demo

Everything is also available as a library call (``samadhvani.chant``); the CLI exists
so the tool is usable without opening Python.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__


def _read_text_arg(value: str) -> str:
    """Accept either literal text or a path (a path is a very common slip)."""
    if os.path.exists(value) and os.path.isfile(value):
        with open(value, encoding="utf-8") as fh:
            return fh.read()
    return value


def cmd_chant(args) -> int:
    from .pipeline import ChantOptions, chant

    text = _read_text_arg(args.text)
    opts = ChantOptions(
        mode=args.mode, school=args.school, model=args.model, laya=args.laya,
        tonic_hz=args.tonic, transpose=args.transpose, intensity=args.intensity,
        room=args.room, breath=args.breath, engine=args.engine,
        checkpoint=args.checkpoint, seed=args.seed,
    )
    c = chant(text, opts)
    c.save(args.out)
    print(f"wrote {args.out}  ({c.duration:.1f}s, {len(c.notes)} notes, mode={c.mode})")
    if args.verbose:
        print(json.dumps({k: v for k, v in c.to_dict().items() if k != "notes"},
                         ensure_ascii=False, indent=1))
    for caveat in c.caveats:
        print(f"  · {caveat}", file=sys.stderr)
    return 0


def cmd_analyze(args) -> int:
    from .pipeline import chant

    c = chant(_read_text_arg(args.text), laya=args.laya)
    d = c.to_dict()
    d.pop("notes", None)
    print(json.dumps(d, ensure_ascii=False, indent=1))
    return 0


def cmd_eval(args) -> int:
    from .eval.metrics import evaluate
    from .pipeline import ChantOptions, chant
    from .synth.io import read_wav

    if args.wav:
        audio, sr = read_wav(args.wav)
        # Re-derive the score the file was rendered with, then score the audio.
        c = chant(_read_text_arg(args.text), ChantOptions(laya=args.laya))
        c.audio = audio
        c.sr = sr
    else:
        c = chant(_read_text_arg(args.text), ChantOptions(laya=args.laya))
    rep = evaluate(c)
    print(rep.pretty())
    return 0


def cmd_audit(args) -> int:
    from .data.sources import DEFAULT_PROFILE, PROFILES, audit

    profile = getattr(args, "profile", None) or DEFAULT_PROFILE
    if profile not in PROFILES:
        print(f"unknown profile {profile!r}; choose from {', '.join(PROFILES)}",
              file=sys.stderr)
        return 2
    audit(profile=profile)
    return 0


def cmd_serve(args) -> int:
    from .app.serve import run

    return run(host=args.host, port=args.port, reload=args.reload)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="samadhvani",
        description="Samadhvani — Sāmaveda chant synthesis.",
    )
    ap.add_argument("--version", action="version", version=f"samadhvani {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("chant", help="render a verse to a WAV file")
    p.add_argument("text", help="the verse, or a path to a UTF-8 file containing it")
    p.add_argument("-o", "--out", default="chant.wav")
    p.add_argument("--mode", default="auto", choices=["auto", "style", "samhita", "gana"])
    p.add_argument("--school", default="kauthuma",
                   choices=["kauthuma", "ranayaniya", "jaiminiya", "rigvedic"])
    p.add_argument("--model", default="gramageya",
                   choices=["gramageya", "aranyaka", "uha", "uhya", "mahanamni"])
    p.add_argument("--laya", type=float, default=0.42, help="seconds per mātrā")
    p.add_argument("--tonic", type=float, default=146.83, help="Sa in Hz")
    p.add_argument("--transpose", type=float, default=0.0, help="semitones")
    p.add_argument("--intensity", type=float, default=0.8)
    p.add_argument("--room", type=float, default=0.18)
    p.add_argument("--breath", type=float, default=0.20)
    p.add_argument("--engine", default="formant",
                   choices=["formant", "voice", "neural"],
                   help="formant (built in), voice (trained voice colour), "
                        "neural (legacy v1)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_chant)

    p = sub.add_parser("analyze", help="show syllables, phones, svaras, vikāras")
    p.add_argument("text")
    p.add_argument("--laya", type=float, default=0.42)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("eval", help="score a render")
    p.add_argument("text")
    p.add_argument("--wav", default=None, help="score this audio instead of re-rendering")
    p.add_argument("--laya", type=float, default=0.42)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("audit", help="what data exists and what may be trained on")
    p.add_argument("--profile", default=None,
                   help="intended-use profile: educational (default), open, evaluation")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("serve", help="run the web demo")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
