"""The demo server.

    uvicorn samadhvani.app.server:app --host 0.0.0.0 --port 8000

Endpoints
---------
``GET  /``                 the single-page demo
``GET  /api/samples``      the sample verses + UI options
``POST /api/chant``        render a chant, return WAV (base64) + full metadata
``POST /api/analyze``      text analysis only: syllables, phones, svaras, vikāras
``GET  /api/health``       version + engine availability
``GET  /api/data``         the data audit, so the page can show where training data
                           actually comes from and what may be used

The server deliberately returns *all* the intermediate structure along with the
audio.  A chant TTS that only hands back a WAV teaches nothing; showing the svara
per syllable, the phones, and which vikāras were applied is both more useful and
more honest about what the system did.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..data.sources import DEFAULT_PROFILE, PROFILES, SOURCES, audit
from ..eval.metrics import evaluate
from ..models.sama_models import MODELS, model_keys
from ..pipeline import ChantOptions, chant
from ..synth.io import wav_bytes
from ..text.svaras import SAMA_SVARAS
from ..text.unicodedata_ext import detect_script
from ..text.vikara import SCHOOLS, VIKARAS

HERE = Path(__file__).parent
STATIC = HERE / "static"

#: Checkpoint locations tried in order for the trained engines.
CHECKPOINT_CANDIDATES = [
    os.environ.get("SAMADHVANI_CKPT", ""),
    str(HERE.parent.parent / "checkpoints" / "voice.pt"),
    str(Path.home() / ".cache" / "samadhvani" / "voice.pt"),
    str(HERE.parent.parent / "checkpoints" / "samadhvani.pt"),
    str(Path.home() / ".cache" / "samadhvani" / "samadhvani.pt"),
]


def _checkpoint_kind(path: str | None) -> str | None:
    """Which engine a checkpoint serves: ``"voice"`` or ``"neural"``.

    The two are not interchangeable.  ``voice-color`` v2 checkpoints were trained
    before alignment and predict a spectral residual, so they must go through the
    formant renderer; older post-alignment checkpoints drive the neural engine.
    """
    if not path:
        return None
    try:
        import torch

        # weights_only=False is required and safe here: the checkpoint carries numpy
        # arrays (corpus mean spectrum), and torch >= 2.6 defaults weights_only to
        # True, which raises on those and would make the engine silently disappear.
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if isinstance(blob, dict) and blob.get("kind") == "voice-color":
        return "voice"
    return "neural"


def _voice_provenance(path: str | None) -> dict | None:
    """What the loaded voice model was trained on, in words a listener can read.

    A demo that plays a trained voice owes the listener one fact: whose voice is
    this, and was it trained on the tradition or on something synthetic?  Two
    checkpoints can report identical metrics and mean completely different things.
    """
    if not path:
        return None
    try:
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    prov = (blob or {}).get("provenance") or {}
    sources = [os.path.basename(str(x)) for x in prov.get("sources", [])]
    archive = [x for x in sources if x.upper().startswith("SV")]
    if archive:
        summary = (f"trained on {prov.get('hours', 0):.2f} h of IISH sāmagāna "
                   f"({len(archive)} archive.org clips)")
    elif sources:
        summary = (f"trained on {prov.get('hours', 0):.2f} h of {len(sources)} local "
                   "audio files — not archival sāmagāna")
    else:
        summary = "no provenance recorded (checkpoint predates provenance tracking)"
    return {
        "summary": summary,
        "hours": prov.get("hours"),
        "n_clips": len(archive) or len(sources),
        "is_archival": bool(archive),
        "metrics": (blob or {}).get("metrics", {}),
    }


def _find_checkpoint(kind: str | None = None) -> str | None:
    """First checkpoint on disk, optionally restricted to one engine."""
    for c in CHECKPOINT_CANDIDATES:
        if not (c and Path(c).exists()):
            continue
        if kind is None or _checkpoint_kind(c) == kind:
            return c
    return None


app = FastAPI(title="Samadhvani — Sāmaveda chant synthesis", version=__version__)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class ChantRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    mode: str = "auto"
    school: str = "kauthuma"
    model: str = "gramageya"
    laya: float = Field(0.42, ge=0.22, le=0.70)
    tonic_hz: float = Field(146.83, ge=90.0, le=260.0)
    transpose: float = Field(0.0, ge=-12.0, le=12.0)
    intensity: float = Field(0.8, ge=0.0, le=1.0)
    room: float = Field(0.18, ge=0.0, le=0.6)
    breath: float = Field(0.20, ge=0.0, le=0.8)
    engine: str = "formant"
    evaluate: bool = True
    seed: int = 7


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    school: str = "kauthuma"
    model: str = "gramageya"


# --------------------------------------------------------------------------- #
# Static page
# --------------------------------------------------------------------------- #

@app.get("/")
def index():
    p = STATIC / "index.html"
    if not p.exists():
        return JSONResponse({"error": "UI not built"}, status_code=500)
    return FileResponse(str(p))


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health():
    try:
        import torch  # noqa: F401
        neural = True
    except Exception:
        neural = False
    ck = _find_checkpoint()
    voice_ck = _find_checkpoint("voice")
    neural_ck = _find_checkpoint("neural")
    engines = ["formant"]
    if voice_ck:
        engines.append("voice")
    if neural_ck:
        engines.append("neural")
    return {
        "ok": True,
        "version": __version__,
        "torch_available": neural,
        "checkpoint": ck,
        "neural_ready": bool(neural and neural_ck),
        "voice_ready": bool(voice_ck),
        "voice_provenance": _voice_provenance(voice_ck),
        "engines": engines,
        "modes": ["auto", "style", "samhita", "gana"],
        "models": model_keys(),
    }


@app.get("/api/samples")
def samples():
    from .samples import SAMPLE_VERSES

    return {
        "samples": SAMPLE_VERSES,
        "schools": [
            {"key": k, "name": v.name, "iast": v.iast, "display": v.display,
             "notes": v.notes, "stobhas": v.stobhas}
            for k, v in SCHOOLS.items()
        ],
        "models": [
            {"key": k, "name_deva": m.name_deva, "name_iast": m.name_iast,
             "notes": m.notes, "source_book": m.source_book}
            for k, m in MODELS.items()
        ],
        "svaras": [
            {"number": k, "name": v["name"], "iast": v["iast"], "venu": v["venu"],
             "semitone": v["semitone"]}
            for k, v in SAMA_SVARAS.items()
        ],
        "vikaras": {k.value: v for k, v in VIKARAS.items()},
    }


@app.get("/api/data")
def data_audit(profile: str = DEFAULT_PROFILE):
    """The data audit.  ``profile`` is an intended-use profile, not a licence filter.

    Two booleans are returned per source on purpose: ``train``/``ship`` are the
    strict licence answer, ``may_train``/``may_ship_weights`` are the answer under
    the requested profile.  A source can be ``train: false, may_train: true`` and
    that is the whole point — usable for teaching, not licensed for release.
    """
    if profile not in PROFILES:
        raise HTTPException(400, f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    rep = audit(verbose=False, profile=profile)
    return {
        "report": rep,
        "profiles": [
            {"key": p.key, "name": p.name, "distributes": p.distributes,
             "commercial": p.commercial, "note": p.note}
            for p in PROFILES.values()
        ],
        "sources": [
            {"key": s.key, "name": s.name, "url": s.url, "kind": s.kind,
             "licence": s.licence.value, "school": s.school, "hours": s.est_hours,
             "verses": s.est_verses, "ganas": s.est_ganas,
             "aligned_hours": s.est_aligned_hours,
             "svara_notation": s.has_svara_notation,
             "notes": s.notes, "rights_basis": s.rights_basis,
             "duplicate_of": s.duplicate_of,
             "train": s.usable_for_training, "ship": s.usable_for_release,
             "may_train": s.permit(profile).may_train,
             "may_ship_weights": s.permit(profile).may_ship_weights,
             "permit_reason": s.permit(profile).reason}
            for s in SOURCES
        ],
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if req.school not in SCHOOLS:
        raise HTTPException(400, f"unknown school {req.school}")
    if req.model not in MODELS:
        raise HTTPException(400, f"unknown model {req.model}")
    try:
        c = chant(req.text, ChantOptions(mode=req.mode if False else "auto",
                                         school=req.school, model=req.model,
                                         laya=0.42))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    d = c.to_dict()
    d.pop("notes", None)
    return d


@app.post("/api/chant")
def chant_endpoint(req: ChantRequest):
    if req.school not in SCHOOLS:
        raise HTTPException(400, f"unknown school {req.school}; have {sorted(SCHOOLS)}")
    if req.model not in MODELS:
        raise HTTPException(400, f"unknown model {req.model}; have {model_keys()}")
    if req.engine not in ("formant", "voice", "neural"):
        raise HTTPException(400, "engine must be formant, voice or neural")

    ck = _find_checkpoint(req.engine) if req.engine != "formant" else None
    if req.engine == "voice" and not ck:
        raise HTTPException(
            400,
            "No voice checkpoint found. Train one with "
            "`python -m samadhvani.data.build corpus` then "
            "`python -m samadhvani.data.build f0` and "
            "`python -m samadhvani.train.voice_trainer`, or use the formant engine.",
        )
    if req.engine == "neural" and not ck:
        raise HTTPException(
            400,
            "No trained checkpoint found. Train one with "
            "`python -m samadhvani.train.trainer`, or use the formant engine.",
        )

    opts = ChantOptions(
        mode=req.mode, school=req.school, model=req.model, laya=req.laya,
        tonic_hz=req.tonic_hz, transpose=req.transpose, intensity=req.intensity,
        room=req.room, breath=req.breath, engine=req.engine, checkpoint=ck,
        seed=req.seed,
    )
    try:
        c = chant(req.text, opts)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"render failed: {exc}") from exc

    payload = c.to_dict()
    payload["audio_b64"] = base64.b64encode(wav_bytes(c.audio, c.sr)).decode()
    payload["audio_mime"] = "audio/wav"
    payload["tonic_hz"] = round(c.tonic_hz, 2)
    payload["script_detected"] = detect_script(req.text).value

    if req.evaluate:
        try:
            rep = evaluate(c)
            sm = rep.to_dict()["svara"]
            sm.pop("per_note", None)
            payload["evaluation"] = {
                "svara": sm,
                "audio": rep.to_dict()["audio"],
            }
        except Exception as exc:  # evaluation must never break a render
            payload["evaluation"] = {"error": str(exc)}

    return payload


@app.post("/api/chant.wav")
def chant_wav(req: ChantRequest):
    """Same as /api/chant but streams a WAV directly, for <audio src> use."""
    # fall back silently: this endpoint is used directly in an <audio src>
    engine = "formant" if req.engine not in ("voice", "neural") else req.engine
    ck = _find_checkpoint(engine)
    if not ck:
        engine = "formant"
    opts = ChantOptions(
        mode=req.mode, school=req.school, model=req.model, laya=req.laya,
        tonic_hz=req.tonic_hz, transpose=req.transpose, intensity=req.intensity,
        room=req.room, breath=req.breath,
        engine=engine, checkpoint=ck, seed=req.seed,
    )
    try:
        c = chant(req.text, opts)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return Response(content=wav_bytes(c.audio, c.sr), media_type="audio/wav",
                    headers={"Content-Disposition": 'inline; filename="chant.wav"'})
