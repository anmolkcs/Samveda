"""Build a training corpus from the actual recordings.

This is the module the project was missing.  Everything before it described where the
audio lives; this goes and gets it and turns it into tensors.

The pipeline is three steps, each of which is independently useful:

``fetch``   download the archive.org items (IISH's 64 clips, or PCR's six volumes).
            Range-resumable, idempotent, and it verifies size against the metadata
            so a truncated MP3 fails loudly instead of silently becoming a 4-hour
            chunk of zeros.

``segment`` cut each long clip into breath groups and write one JSONL row per group,
            carrying its log-mel features and its phone/svara targets.  Segmentation
            is by energy — the reciter's breaths — because that is what the tradition
            actually does and because it needs no transcription.

``pack``    stack the variable-length groups into an .npz that the trainer memory-maps.
            Decoding 4 h of MP3 to float32 is ~1.3 GB and 16 h is ~5 GB; on Kaggle that
            is the difference between training and OOM, so nothing here ever holds the
            whole corpus in RAM.

Why energy segmentation rather than the Iyer timestamps
------------------------------------------------------
The timestamps are the better signal and they are the eventual plan (see
``docs/DATA-ADDENDUM.md`` §7), but they require transcribing Malayalam handwriting
line by line.  Breath-group segmentation needs nothing, and it still produces
supervised pairs: the phones come from the alignment model, the melody from the
notated svara, and the *acoustics* from the recording.  What it cannot do is tell you
which gāna a breath group is, so it cannot supervise melody from audio — it learns
timbre.  That limitation is real and is printed, not hidden.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..audio.mel import (HOP, N_MELS, SR, breath_groups, logmel, mel_filterbank,
                         read_audio_window, trim_silence)
from ..phones.g2p import phones_for_view
from ..synth.timing import TimingOptions, plan_timing
from ..text.svaras import annotate_svaras
from ..text.syllabify import syllabify
from ..text.unicodedata_ext import normalize, to_devanagari

ARCHIVE_IIISH = "https://archive.org/download/IISHSamaVeda"


# --------------------------------------------------------------------------- #
# 1. fetch
# --------------------------------------------------------------------------- #

#: The 64 IISH clips, measured (minutes) from archive.org metadata.  Kept here so
#: that ``fetch`` can verify a download without a second network round-trip.
IISH_CLIPS: dict[str, float] = {
    "SV001": 44.13, "SV002": 39.26, "SV003": 43.38, "SV004": 44.11,
    "SV005": 43.14, "SV006": 38.93, "SV007": 37.92, "SV008": 45.92,
    "SV009": 42.78, "SV010": 41.34, "SV011": 44.92, "SV012": 44.17,
    "SV013": 39.37, "SV014": 42.32, "SV015": 35.12, "SV016": 38.87,
    "SV017": 38.60, "SV018": 41.48, "SV019": 43.87, "SV020": 29.98,
    "SV021": 38.22, "SV022": 40.40, "SV023": 33.50, "SV024": 15.47,
}

#: Clips 1-24 finish Grāma-gāna and Āraṇyaka-gāna — the two books whose text is
#: transcribed.  Clips 25-64 are Ūha/Ūhya and are *not* used by default.
IISH_ALIGNED_CLIPS = tuple(f"SV{i:03d}" for i in range(1, 25))


def fetch_iish(dest: str, clips: tuple[str, ...] = IISH_ALIGNED_CLIPS,
               verbose: bool = True) -> list[str]:
    """Download IISH sāmagāna clips from archive.org.  Idempotent and resumable.

    Skips anything already present at the right size, so an interrupted run costs
    only the clips it had not finished.
    """
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    got: list[str] = []
    for name in clips:
        path = out / f"{name}.mp3"
        expected = None
        if name in IISH_CLIPS:
            # 128 kbps CBR measured from the item metadata, ±15 %
            expected = IISH_CLIPS[name] * 60 * 128_000 / 8
        if path.exists() and path.stat().st_size > 100_000:
            if expected is None or abs(path.stat().st_size - expected) < expected * 0.15:
                if verbose:
                    print(f"  = {name}.mp3 already present "
                          f"({path.stat().st_size/1e6:.1f} MB)")
                got.append(str(path))
                continue
            path.unlink()
        url = f"{ARCHIVE_IIISH}/{name}.mp3"
        if verbose:
            print(f"  ↓ {name}.mp3 …", end="", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(path, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
        except Exception as exc:
            print(f"  ! {name}: {exc}")
            if path.exists():
                path.unlink()
            continue
        size = path.stat().st_size
        ok = expected is None or abs(size - expected) < expected * 0.15
        if verbose:
            print(f" {size/1e6:.1f} MB" + ("" if ok else "  ! unexpected size"))
        got.append(str(path))
    return got


def fetch_pcr(dest: str, volumes: int = 2, verbose: bool = True) -> list[str]:
    """Download PCR Sāma Vedam volumes (complete Kauthuma, no transcript).

    Kept separate from IISH because it has no text mapping: it is for timbre and
    prosody diversity, not for melody supervision.
    """
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    names = ["Sama Vedam/Sama Vedam Vol1/sama 1 to 20.mp3",
             "Sama Vedam/Sama Vedam Vol2/sama 21 to 32.mp3",
             "Sama Vedam/Sama Vedam Vol3/sama 33 to 40.mp3"][:volumes]
    got: list[str] = []
    for rel in names:
        path = out / Path(rel).name.replace(" ", "_")
        if path.exists() and path.stat().st_size > 1_000_000:
            if verbose:
                print(f"  = {path.name} already present")
            got.append(str(path))
            continue
        url = ("https://archive.org/download/SamaVedam/" +
               urllib.parse.quote(rel))
        if verbose:
            print(f"  ↓ {path.name} …", end="", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(path, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
        except Exception as exc:
            print(f"  ! {exc}")
            continue
        if verbose:
            print(f" {path.stat().st_size/1e6:.0f} MB")
        got.append(str(path))
    return got


# --------------------------------------------------------------------------- #
# 2. segment
# --------------------------------------------------------------------------- #

@dataclass
class Segment:
    """One breath group: audio features plus the text targets they should match."""

    audio: str
    start: float
    end: float
    logmel: np.ndarray
    phones: list[list[str]] = field(default_factory=list)
    svaras: list[int] = field(default_factory=list)
    f0_hz: list[float] = field(default_factory=list)
    text: str = ""
    school: str = "kauthuma"
    model: str = "gramageya"
    tonic_hz: float = 146.83

    @property
    def frames(self) -> int:
        return int(self.logmel.shape[0])


def _targets_for(text: str, school: str, model: str, tonic_hz: float):
    """Phone + svara + F0 targets for a line of gāna text."""
    deva = normalize(to_devanagari(text))
    view = syllabify(deva)
    if not view.aksaras:
        return [], [], [], 0.0
    tok = annotate_svaras(view, scheme="tradition")
    flat = [t.svara for t in tok][:len(view.aksaras)]
    ph = phones_for_view(view)
    plan = plan_timing(view.aksaras, flat, groups=view.groups,
                       options=TimingOptions(), phone_lists=[p.phones for p in ph])
    return ([p.phones for p in ph], flat,
            [round(n.f0_hz, 2) for n in plan.notes], plan.total_seconds)


def segment_clip(path: str, *, text_lines: list[str] | None = None,
                 model: str = "gramageya", school: str = "kauthuma",
                 tonic_hz: float = 146.83, max_group_s: float = 12.0,
                 min_group_s: float = 0.45, verbose: bool = False) -> list[Segment]:
    """Cut one recording into breath-group segments with mel features.

    ``text_lines`` is optional and usually absent — that is the honest situation.
    When it is absent the segment still carries acoustics, and the phone targets are
    left empty; the trainer uses those rows for the *vocoder* half of the model
    (unconditional acoustics) and the text rows for the frontend half.
    """
    from ..audio.mel import read_audio
    x = read_audio(path, sr=SR)
    if not len(x):
        return []
    groups = breath_groups(x, SR)
    if verbose:
        print(f"  {Path(path).name}: {len(x)/SR/60:.1f} min -> {len(groups)} breath groups")
    segs: list[Segment] = []
    fb = mel_filterbank(SR)
    for (a, b) in groups:
        if b - a > max_group_s:                     # a long unbroken phrase: chunk it
            n = int(np.ceil((b - a) / max_group_s))
            span = (b - a) / n
            subs = [(a + i * span, a + (i + 1) * span) for i in range(n)]
        else:
            subs = [(a, b)]
        for (s, e) in subs:
            if e - s < min_group_s:
                continue
            seg_a = read_audio_window(path, s, e, sr=SR, pad_s=0.05)
            seg_a = trim_silence(seg_a, SR)
            if len(seg_a) < int(0.3 * SR):
                continue
            M = logmel(seg_a, SR, fb=fb)
            if M.shape[0] < 8:
                continue
            phones, svaras, f0, _ = ([], [], [], 0.0)
            if text_lines:
                idx = len(segs) % len(text_lines)
                phones, svaras, f0, _ = _targets_for(text_lines[idx], school, model, tonic_hz)
            segs.append(Segment(audio=str(path), start=round(s, 3), end=round(e, 3),
                                logmel=M, phones=phones, svaras=svaras, f0_hz=f0,
                                school=school, model=model, tonic_hz=tonic_hz))
    return segs


def build_corpus(files: list[str], out_npz: str, out_jsonl: str | None = None,
                 *, text_lines: list[str] | None = None, max_hours: float = 4.0,
                 model: str = "gramageya", verbose: bool = True) -> dict:
    """Turn recordings into a packed, memory-mappable training corpus.

    ``max_hours`` is a deliberate cap, defaulting to the Vāgdhenu-scale 4 h.  More
    audio stops helping a 2.5 M-parameter model long before it stops costing you a
    Kaggle session; see ``docs/DATA-ADDENDUM.md`` §6.
    """
    budget = max_hours * 3600.0
    segs: list[Segment] = []
    used = 0.0
    for path in files:
        if used >= budget:
            break
        for s in segment_clip(path, text_lines=text_lines, model=model, verbose=verbose):
            segs.append(s)
            used += s.frames * HOP / SR
            if used >= budget:
                break
    if not segs:
        raise SystemExit("no segments produced — check the audio files")

    # Pad to a fixed frame count so the packed array is rectangular.  Breath groups
    # are mostly 1-6 s; padding to the 95th percentile wastes very little and makes
    # batching trivial.
    lengths = np.array([s.frames for s in segs])
    T = int(np.percentile(lengths, 95))
    T = int(np.clip(T, 32, 1200))
    mels = np.zeros((len(segs), T, N_MELS), dtype=np.float32)
    mask = np.zeros((len(segs), T), dtype=np.float32)
    for i, s in enumerate(segs):
        n = min(T, s.frames)
        mels[i, :n] = s.logmel[:n]
        mask[i, :n] = 1.0
        if n < T:
            mels[i, n:] = mels[i, max(0, n - 1)]     # repeat edge frame, masked out

    np.savez_compressed(out_npz, mel=mels, mask=mask,
                        meta=np.array([json.dumps({
                            "audio": s.audio, "start": s.start, "end": s.end,
                            "school": s.school, "model": s.model,
                            "tonic_hz": s.tonic_hz, "frames": s.frames,
                        }) for s in segs]))

    if out_jsonl:
        with open(out_jsonl, "w", encoding="utf-8") as fh:
            for s in segs:
                fh.write(json.dumps({
                    "audio": s.audio, "start": s.start, "end": s.end,
                    "frames": s.frames, "phones": s.phones, "svaras": s.svaras,
                    "f0_hz": s.f0_hz, "school": s.school, "model": s.model,
                    "tonic_hz": s.tonic_hz,
                }) + "\n")

    report = {
        "segments": len(segs),
        "hours": round(sum(s.frames * HOP / SR for s in segs) / 3600.0, 3),
        "frames_per_segment": T,
        "n_mels": N_MELS,
        "sr": SR,
        "npz": out_npz,
        "size_mb": round(os.path.getsize(out_npz) / 1e6, 1),
        "with_text": sum(1 for s in segs if s.phones),
    }
    if verbose:
        print(f"  {report['segments']} segments  {report['hours']:.2f} h  "
              f"{T} frames each  {report['size_mb']:.0f} MB")
        if not report["with_text"]:
            print("  ! no transcription supplied: these segments supervise acoustics "
                  "only.\n    Melody supervision needs the gāna text (see the addendum).")
    return report


def load_corpus(npz: str) -> tuple[np.ndarray, np.ndarray]:
    """Memory-map a packed corpus (does not load it into RAM)."""
    blob = np.load(npz, mmap_mode="r")
    return blob["mel"], blob["mask"]
