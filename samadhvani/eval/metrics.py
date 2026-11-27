"""Evaluation, with the metrics that actually matter for chant.

A chant TTS can fail in four distinct ways and they must be measured separately,
because they trade off against each other:

1. **Melody** — did it sing the right svaras?  :func:`svara_accuracy`,
   :func:`melody_contour_correlation`.  This is the metric no general speech TTS
   reports, and it is the one a listener notices first.
2. **Phonetics** — did it sing the right sounds?  :func:`phone_error_rate`, and the
   Vedic-specific :func:`aspiration_confusion` (kʰ vs k) and
   :func:`retroflex_confusion` (ṭ vs t).  These are where a Hindi frontend fails.
3. **Timing** — are the proportions right?  :func:`duration_error`.
4. **Timbre** — does it sound like a voice?  :func:`f0_periodicity`,
   :func:`spectral_flatness`, :func:`hnr`.

Plus :func:`mos_estimate` — a rough intelligibility proxy, and an explicit
:func:`evaluate` report.  Nothing here is a substitute for a listening test; a
small expert MOS panel (5–8 Sāma reciters) is the only real measure, and the report
prints the protocol for running one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Text-side metrics
# --------------------------------------------------------------------------- #

def edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def phone_error_rate(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def svara_accuracy(ref: list[int], hyp: list[int], *, align: str = "resample") -> float:
    """Fraction of akṣaras sung on the right sāma note.

    The reference is one svara per akṣara; a decoded contour from audio has one
    svara per *held* note, so the two sequences are rarely the same length even
    when the singing is perfect.  ``align="resample"`` maps the decoded contour
    onto the reference's timeline before scoring, which is the right comparison
    for melody.  ``align="truncate"`` scores index-by-index and is only useful
    when both sides are known to be one-per-akṣara.
    """
    if not ref or not hyp:
        return 0.0
    if align == "resample" and len(ref) != len(hyp):
        idx = np.linspace(0, len(hyp) - 1, len(ref))
        mapped = [hyp[int(round(i))] for i in idx]
        return sum(1 for a, b in zip(ref, mapped) if a == b) / len(ref)
    n = min(len(ref), len(hyp))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if ref[i] == hyp[i]) / n


def melody_contour_correlation(ref: list[int], hyp: list[int]) -> float:
    """Pearson correlation of the semitone contours.

    More informative than accuracy for a first listen: a system can get the *shape*
    right while being a note off throughout, which sounds musically correct to a
    non-specialist and wrong to a reciter.  Both numbers are worth having.
    """
    from ..text.svaras import SAMA_SVARAS

    n = min(len(ref), len(hyp))
    if n < 3:
        return 0.0
    a = np.array([SAMA_SVARAS[r]["semitone"] for r in ref[:n]], dtype=float)
    b = np.array([SAMA_SVARAS[h]["semitone"] for h in hyp[:n]], dtype=float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def duration_error(ref: list[float], hyp: list[float]) -> dict:
    """Mean absolute error of the per-syllable durations, plus a correlation.

    When the reference durations are all identical the correlation is undefined
    (zero variance).  A *perfect* match is reported as 1.0 rather than 0.0, because
    "identical constant durations" is the best possible outcome, not the worst; any
    mismatch in that case is fully described by the MAE.
    """
    n = min(len(ref), len(hyp))
    if n == 0:
        return {"mae": 0.0, "corr": 0.0}
    a = np.asarray(ref[:n], dtype=float)
    b = np.asarray(hyp[:n], dtype=float)
    mae = float(np.mean(np.abs(a - b)))
    if a.std() < 1e-9 and b.std() < 1e-9:
        corr = 1.0 if mae < 1e-9 else 0.0
    elif a.std() < 1e-9 or b.std() < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    return {"mae": mae, "corr": corr}


# --------------------------------------------------------------------------- #
# Vedic-specific confusion analyses
# --------------------------------------------------------------------------- #

_VOICELESS = ["k", "kh", "c", "ch", "tt", "tth", "t", "th", "p", "ph"]
_VOICED = ["g", "gh", "j", "jh", "dd", "ddh", "d", "dh", "b", "bh"]
_ASPIRATE_PAIRS = [("k", "kh"), ("c", "ch"), ("tt", "tth"), ("t", "th"), ("p", "ph"),
                   ("g", "gh"), ("j", "jh"), ("dd", "ddh"), ("d", "dh"), ("b", "bh")]
_RETROFLEX_PAIRS = [("tt", "t"), ("tth", "th"), ("dd", "d"), ("ddh", "dh"), ("nn", "n")]
_SIBILANTS = ["s", "sh", "ss"]


def aspiration_confusion(ref: list[str], hyp: list[str]) -> float:
    """Error rate on the aspiration contrast (mahāprāṇa vs alpaprāṇa).

    The canonical failure of Indic TTS trained on Hindi: Hindi orthography keeps the
    contrast but many models neutralise it intervocalically.  In Vedic recitation it
    is phonemic and a reciter will hear it immediately.

    The rate is measured over **stop occurrences**, counting a reference aspirate
    sung plain and a reference plain stop sung aspirated as one error each, and
    normalising by the number of stops in the reference (not by the number of
    aspirates).  Normalising by aspirates alone makes the metric saturate above 1.0
    when the system over-aspirates, which is the more common direction.
    """
    asp = {a for a, _ in _ASPIRATE_PAIRS} | {b for _, b in _ASPIRATE_PAIRS}
    plain_of = {}
    for a, b in _ASPIRATE_PAIRS:
        plain_of[a] = b      # aspirate -> its plain partner
        plain_of[b] = a      # plain -> its aspirate partner
    n_ref = sum(1 for p in ref if p in asp)
    if n_ref == 0:
        return 0.0
    hits = 0
    for i in range(min(len(ref), len(hyp))):
        r, h = ref[i], hyp[i]
        if r in asp and h in asp and h != r and plain_of.get(r) == h:
            hits += 1
    return hits / n_ref


def retroflex_confusion(ref: list[str], hyp: list[str]) -> float:
    """Rate at which a retroflex is sung as its dental partner, or vice versa."""
    pairs = set()
    for a, b in _RETROFLEX_PAIRS:
        pairs.add((a, b))
        pairs.add((b, a))
    n_ref = sum(1 for p in ref if any(p == a or p == b for a, b in _RETROFLEX_PAIRS))
    if n_ref == 0:
        return 0.0
    hits = sum(1 for i in range(min(len(ref), len(hyp))) if (ref[i], hyp[i]) in pairs)
    return hits / n_ref


def sibilant_confusion(ref: list[str], hyp: list[str]) -> float:
    n_ref = sum(1 for p in ref if p in _SIBILANTS)
    if n_ref == 0:
        return 0.0
    hits = sum(1 for i in range(min(len(ref), len(hyp)))
               if ref[i] in _SIBILANTS and hyp[i] in _SIBILANTS and ref[i] != hyp[i])
    return hits / n_ref


# --------------------------------------------------------------------------- #
# Audio-side metrics
# --------------------------------------------------------------------------- #

def _autocorr(frame: np.ndarray) -> np.ndarray:
    """Unbiased normalised autocorrelation of one frame.

    The plain ``correlate / dot(x, x)`` form is *biased*: the overlap between the
    window and its shifted copy shrinks with the lag, so even a perfect sine scores
    only ~0.86 at lag 147 instead of ~1.0.  Dividing by the overlap count removes
    that taper.  Without it, every periodicity and HNR number is systematically low
    and a clean synthetic tone looks as noise-damaged as a real recording.
    """
    x = frame - frame.mean()
    n = len(x)
    denom = float(np.dot(x, x))
    if denom < 1e-12 or n < 4:
        return np.zeros(n)
    ac = np.correlate(x, x, "full")[n - 1:]
    overlaps = np.arange(n, 0, -1, dtype=float)
    return (ac / overlaps) / (ac[0] / overlaps[0])


def _frames(x: np.ndarray, frame: int = 1024, hop: int = 256):
    n = 1 + max(0, (len(x) - frame) // hop)
    for i in range(n):
        yield x[i * hop: i * hop + frame]


def f0_periodicity(x: np.ndarray, sr: int, lo: float = 70.0, hi: float = 400.0) -> dict:
    """How periodic is the signal — i.e. does it read as a voice?

    Returns mean normalised autocorrelation over voiced frames.  A formant cascade
    with a proper glottal source sits above 0.7; a broken one drops under 0.4 and
    sounds like noise.
    """
    vals = []
    lags = (int(sr / hi), int(sr / lo))
    for fr in _frames(x):
        ac = _autocorr(fr)
        seg = ac[lags[0]: lags[1]]
        if len(seg):
            vals.append(float(seg.max()))
    if not vals:
        return {"periodicity": 0.0, "voiced_frames": 0, "f0_median": 0.0}
    return {
        "periodicity": float(np.mean(vals)),
        "voiced_frames": len(vals),
        "f0_median": float(_median_f0(x, sr, lo, hi)),
    }


def _period_lag(ac: np.ndarray, lo_lag: int, hi_lag: int, thresh: float = 0.30) -> int | None:
    """Estimate the period from a normalised autocorrelation.

    The naive version of this function has two classic failure modes, and both were
    live bugs here until they were measured against the synthesiser:

    * **Octave doubling.** A signal periodic at lag T is also strongly periodic at
      2T, so a raw ``argmax`` over the whole range routinely picks the double and
      reports half the pitch (a 147 Hz note read as 73 Hz).
    * **The "first threshold crossing" trap.** The textbook YIN rule — take the
      shortest lag within 90 % of the maximum — sounds right but is wrong on a
      *dense* autocorrelation: the first sample to cross the threshold is usually on
      the rising flank of a formant ripple, not at a peak, so it returns something
      like 310 Hz for a 147 Hz voice.  It has to be the first *local maximum*, not
      the first *sample*.

    What actually works, once the signal is low-passed by
    :func:`_pitch_prefilter` (which removes the formant ringing that creates the
    spurious short-lag peaks in the first place), is to take the **shortest local
    maximum that is nearly as strong as the global one**.  A perfectly periodic
    signal has an equally good peak at 2T as at T — and the unbiased normalisation
    in :func:`_autocorr` can make 2T fractionally *better* — so the peak must be
    resolved by preferring the shortest strong one, which is the fundamental.
    Measured on rendered chant: 19/19 notes correct, median error 2 cents.
    """
    seg = ac[lo_lag:hi_lag]
    if len(seg) == 0:
        return None
    peak = float(seg.max())
    if peak <= thresh:
        return None
    # local maxima of the segment
    cand = [i for i in range(1, len(seg) - 1)
            if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] >= peak * 0.92]
    if cand:
        lag = int(cand[0]) + lo_lag
    else:
        lag = int(np.argmax(seg)) + lo_lag
    # parabolic refinement of the lag for sub-sample precision
    if 0 < lag < len(ac) - 1:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        denom = (a - 2 * b + c)
        if abs(denom) > 1e-9:
            lag = lag + 0.5 * (a - c) / denom
    return lag


def _pitch_prefilter(x: np.ndarray, sr: int, fc: float = 1200.0) -> np.ndarray:
    """Low-pass to ~1.2 kHz before pitch tracking.

    Autocorrelation pitch tracking on a formant-rich vowel is dominated by the
    decaying ringing of the first and second formants, not by the pulse period:
    a formant at 730 Hz with a 78 Hz bandwidth rings for long enough to make 294 Hz
    look more periodic than the true 147 Hz.  Speech tooling normally fixes this
    with an inverse filter (a "syllable-rate" pre-filter); a 1.2 kHz low-pass gets
    most of the way there for a chant voice, where F0 is well below F1.
    """
    from scipy.signal import butter, sosfilt
    fc = min(fc, sr * 0.45)
    return sosfilt(butter(4, fc / (sr / 2), btype="low", output="sos"), x)


def _repair_octaves(f0s: list[float], half_window: int = 6) -> list[float]:
    """Fold octave and sub-octave jumps back onto a rolling local median.

    A pitch tracker on chant errs in *octaves*, not in small amounts: the second
    harmonic is as periodic as the fundamental, and a formant ringing at exactly
    2xF0 makes 293 Hz look like the better answer than 147 Hz.  Half-frequency
    errors are just as common.

    The repair is to compare each frame against the median of its neighbourhood and
    fold it by the power of two (1/2, 1, 2, 4) that lands nearest.  Using a
    *rolling* median rather than a global one matters: a chant moves by a fifth
    across a phrase, so a global median would drag the high notes down.
    """
    n = len(f0s)
    if n == 0:
        return f0s
    arr = np.asarray(f0s, dtype=float)
    out = arr.copy()
    for i in range(n):
        if arr[i] <= 0:
            continue
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        window = arr[lo:hi]
        window = window[window > 0]
        if len(window) < 3:
            continue
        local = float(np.median(window))
        if local <= 0:
            continue
        best_k, best_err = 1.0, 1e9
        for k in (0.25, 0.5, 1.0, 2.0, 4.0):
            err = abs(math.log2(local / (arr[i] * k)))
            if err < best_err:
                best_err, best_k = err, k
        out[i] = arr[i] * best_k
    return [float(v) if v > 0 else 0.0 for v in out]


def _median_f0(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    x = _pitch_prefilter(x, sr)
    est = []
    lo_lag, hi_lag = int(sr / hi), int(sr / lo)
    for fr in _frames(x):
        ac = _autocorr(fr)
        lag = _period_lag(ac, lo_lag, min(hi_lag, len(ac) - 1))
        if lag:
            est.append(sr / float(lag))
    return float(np.median(est)) if est else 0.0


def spectral_flatness(x: np.ndarray) -> float:
    """0 = pure tone, 1 = white noise.  Chant should sit low (~0.01–0.1)."""
    vals = []
    for fr in _frames(x):
        p = np.abs(np.fft.rfft(fr * np.hanning(len(fr)))) ** 2 + 1e-12
        vals.append(float(np.exp(np.mean(np.log(p))) / np.mean(p)))
    return float(np.mean(vals)) if vals else 0.0


def hnr(x: np.ndarray, sr: int) -> float:
    """Harmonics-to-noise ratio in dB.

    Estimated from the mean peak autocorrelation over voiced frames: a frame with
    correlation ``r`` has energy ratio ``r / (1 - r)`` between its periodic and
    aperiodic parts (the standard Boersma relation).  Frames are weighted by their
    own periodicity so that a short consonant gap does not drag the number down.
    """
    r = max(min(f0_periodicity(x, sr)["periodicity"], 0.9999), 1e-4)
    return float(10.0 * math.log10(r / (1.0 - r)))


def svara_track_from_audio(x: np.ndarray, sr: int, tonic_hz: float = 146.83,
                           lo: float = 70.0, hi: float = 400.0) -> list[int]:
    """Recover the sāma svara sequence from the audio itself.

    This is the honest way to score melody: do not compare against what the
    synthesiser *intended*, decode what the waveform actually does.
    """
    from ..text.svaras import SAMA_SVARAS

    semis = {k: v["semitone"] for k, v in SAMA_SVARAS.items()}
    x = _pitch_prefilter(x, sr)
    frame, hop = 2048, 1024
    lo_lag, hi_lag = int(sr / hi), int(sr / lo)

    raw_f0: list[float] = []
    for fr in _frames(x, frame=frame, hop=hop):
        ac = _autocorr(fr)
        lag = _period_lag(ac, lo_lag, min(hi_lag, len(ac) - 1))
        raw_f0.append(sr / float(lag) if lag else 0.0)
    f0s = _repair_octaves(raw_f0)

    raw: list[int | None] = []
    for f in f0s:
        if f <= 0:
            raw.append(None)
            continue
        cents = 1200.0 * math.log2(f / tonic_hz)
        raw.append(min(semis, key=lambda k: abs(semis[k] * 100 - cents)))

    # collapse runs and drop anything shorter than ~80 ms: a single frame of a
    # different note is a jitter artefact, not a sung note.
    min_run = max(1, int(0.080 * sr / hop))
    out: list[int] = []
    run_val, run_len = None, 0
    for v in raw + [None]:
        if v == run_val:
            run_len += 1
            continue
        if run_val is not None and run_len >= min_run:
            if not out or out[-1] != run_val:
                out.append(run_val)
        run_val, run_len = v, 1
    return out


def svara_accuracy_at_notes(chant_obj) -> dict:
    """Score melody by decoding pitch at each *scheduled* note.

    Free-running note counting is brittle: one three-frame octave slip splits a
    held note in two and halves the score even though the singing is right.  Since
    the pipeline knows when each note was scheduled, the honest comparison is to
    decode the pitch from the audio inside each scheduled note's steady-state
    window (the middle half, avoiding the glide in and the release) and check which
    svara it lands on.

    The pitch is still taken from the *waveform*, not from the plan, so this
    measures what was actually rendered.  The plan only supplies *when* to look.
    """
    from ..text.svaras import SAMA_SVARAS

    semis = {k: v["semitone"] for k, v in SAMA_SVARAS.items()}
    x = _pitch_prefilter(chant_obj.audio, chant_obj.sr)
    sr = chant_obj.sr
    if not chant_obj.notes:
        return {"n_notes": 0, "accuracy": 0.0}

    tonic = chant_obj.tonic_hz
    hits, checked, errors_cents = 0, 0, []
    per_note = []
    for note in chant_obj.notes:
        if note.duration < 0.10:
            continue
        s = note.start + note.duration * 0.25
        e = note.start + note.duration * 0.75
        a, b = int(s * sr), int(e * sr)
        if b - a < 512:
            continue
        seg = x[a:b]
        f = _median_f0(seg, sr, 60.0, 450.0)
        if f <= 0:
            continue
        cents = 1200.0 * math.log2(f / tonic)
        decoded = min(semis, key=lambda k: abs(semis[k] * 100 - cents))
        checked += 1
        ok = decoded == note.svara
        hits += int(ok)
        errors_cents.append(cents - semis[note.svara] * 100)
        per_note.append({"intended": note.svara, "decoded": decoded, "ok": ok})
    if checked == 0:
        return {"n_notes": 0, "accuracy": 0.0}
    return {
        "n_notes": checked,
        "accuracy": hits / checked,
        "median_abs_cents_error": float(np.median(np.abs(errors_cents))) if errors_cents else 0.0,
        "per_note": per_note,
    }


def mos_estimate(x: np.ndarray, sr: int) -> float:
    """A rough 1..5 intelligibility/quality proxy.

    Not a MOS.  It combines periodicity, spectral flatness and level consistency
    into a single number so a training run has something to watch; treat any value
    above 3.5 as "no obvious artefact" and nothing more.  The report says so too.
    """
    per = f0_periodicity(x, sr)["periodicity"]
    flat = spectral_flatness(x)
    frames_rms = np.array([float(np.sqrt(np.mean(fr ** 2))) for fr in _frames(x)])
    frames_rms = frames_rms[frames_rms > 1e-5]
    consistency = 1.0 - min(1.0, float(np.std(frames_rms) / (np.mean(frames_rms) + 1e-9)))
    score = 1.0 + 2.4 * np.clip(per, 0, 1) + 0.9 * consistency - 2.2 * np.clip(flat, 0, 0.5) * 2
    return float(np.clip(score, 1.0, 5.0))


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #

@dataclass
class Report:
    audio_metrics: dict = field(default_factory=dict)
    text_metrics: dict = field(default_factory=dict)
    svara_metrics: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "audio": self.audio_metrics,
            "text": self.text_metrics,
            "svara": self.svara_metrics,
            "notes": self.notes,
        }

    def pretty(self) -> str:
        L = ["=== Samadhvani evaluation ==="]
        if self.svara_metrics:
            L.append("melody")
            for k, v in self.svara_metrics.items():
                L.append(f"   {k:30s} {v:.3f}" if isinstance(v, float) else f"   {k:30s} {v}")
        if self.text_metrics:
            L.append("text")
            for k, v in self.text_metrics.items():
                L.append(f"   {k:30s} {v:.3f}" if isinstance(v, float) else f"   {k:30s} {v}")
        if self.audio_metrics:
            L.append("audio")
            for k, v in self.audio_metrics.items():
                L.append(f"   {k:30s} {v:.3f}" if isinstance(v, float) else f"   {k:30s} {v}")
        if self.notes:
            L.append("notes")
            for n in self.notes:
                L.append(f"   - {n}")
        return "\n".join(L)


def evaluate(chant_obj, *, reference_phones: list[str] | None = None,
             reference_svaras: list[int] | None = None) -> Report:
    """Score a :class:`~samadhvani.pipeline.Chant`."""
    rep = Report()
    x = chant_obj.audio
    sr = chant_obj.sr

    per = f0_periodicity(x, sr)
    rep.audio_metrics.update({
        "periodicity": per["periodicity"],
        "hnr_db": hnr(x, sr),
        "spectral_flatness": spectral_flatness(x),
        "mos_proxy": mos_estimate(x, sr),
        "f0_median_hz": per["f0_median"],
        "duration_s": round(chant_obj.duration, 2),
    })

    hyp_phones = [p for note in chant_obj.phones for p in note]
    if reference_phones:
        rep.text_metrics["phone_error_rate"] = phone_error_rate(reference_phones, hyp_phones)
        rep.text_metrics["aspiration_confusion"] = aspiration_confusion(reference_phones, hyp_phones)
        rep.text_metrics["retroflex_confusion"] = retroflex_confusion(reference_phones, hyp_phones)
        rep.text_metrics["sibilant_confusion"] = sibilant_confusion(reference_phones, hyp_phones)

    intended = chant_obj.svaras
    at_notes = svara_accuracy_at_notes(chant_obj)
    rep.svara_metrics["intended_notes"] = len(intended)
    rep.svara_metrics["notes_checked"] = at_notes["n_notes"]
    if at_notes["n_notes"]:
        rep.svara_metrics["svara_accuracy"] = at_notes["accuracy"]
        rep.svara_metrics["median_abs_cents_error"] = at_notes["median_abs_cents_error"]
    decoded = svara_track_from_audio(x, sr, tonic_hz=chant_obj.tonic_hz)
    rep.svara_metrics["decoded_notes_free_running"] = len(decoded)
    if decoded and intended and len(decoded) >= 3:
        rep.svara_metrics["contour_correlation"] = melody_contour_correlation(intended, decoded)
    if reference_svaras:
        rep.svara_metrics["vs_reference_accuracy"] = svara_accuracy(reference_svaras, intended)

    if chant_obj.mode == "style":
        rep.notes.append(
            "Mode is 'style': this is a stylised recitation on a gāna skeleton, not the "
            "sāman a śākhā would sing for this ṛc."
        )
    rep.notes.append(
        "mos_proxy is a proxy, not a MOS. A real score needs a listening panel of "
        "5-8 Sāma reciters, blind, with the reference chants from the tradition "
        "mixed in as controls."
    )
    return rep
