"""Timing: turn a svara sequence into a syllable-by-syllable rhythm.

Sāmagāna has no fixed tempo — a Kauthuma udgātṛ stretches a line to twice the
length of the same line sung by a Jaiminīya singer, and the whole point of
vikarṣaṇa is that a single syllable can occupy four or five beats.  So timing is
expressed in *mātrās* and the absolute value of a mātrā is a stop-time setting:

* ``laya`` — a mātrā is roughly 0.20–0.45 s.  Slow (vilambita) is the tradition's
  default and is also the setting that makes the chant legible in a demo.
* held syllables (line-final, pluta, or on the cadence note) get an extra hold,
  which is what makes it sound chanted rather than spoken.

Everything downstream (F0 contour, phone durations, legato) reads this plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..text.svaras import SAMA_SVARAS, svara_to_semitone


@dataclass
class Note:
    """A single scheduled note."""

    ak_index: int
    svara: int
    start: float            # seconds
    duration: float         # seconds
    f0_hz: float
    text: str = ""
    phones: list[str] = field(default_factory=list)
    is_cadence: bool = False
    is_line_start: bool = False
    is_line_end: bool = False
    group: int = 0

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def semitone(self) -> int:
        return SAMA_SVARAS[self.svara]["semitone"]


@dataclass
class TimingPlan:
    notes: list[Note]
    matra_seconds: float
    total_seconds: float
    groups: list[tuple[int, int]] = field(default_factory=list)  # (start_note, end_note)

    def __len__(self) -> int:
        return len(self.notes)


@dataclass
class TimingOptions:
    #: Seconds per mātrā. 0.42 is a slow, ceremonial laya.
    matra_seconds: float = 0.42
    #: How much to lengthen a line-final note (nidhana breath).
    cadence_hold: float = 1.9
    #: Extra length on the first note of a line.
    line_start_hold: float = 1.15
    #: Silence inserted at each breath (seconds).
    breath: float = 0.18
    #: Tonic in Hz (Sa = tṛtīya, sāma note 4).
    tonic_hz: float = 146.83
    #: Applied from the chosen gāna model / school.
    register_shift_semitones: float = 0.0
    #: Free per-note stretch multiplier, e.g. to slow the whole chant.
    tempo: float = 1.0
    #: Micro-pause before a consonant cluster, to let the onset be audible.
    cluster_gap: float = 0.02


def plan_timing(
    aksaras,
    svaras: list[int],
    *,
    groups: list[list[int]] | None = None,
    options: TimingOptions | None = None,
    phone_lists: list[list[str]] | None = None,
) -> TimingPlan:
    """Lay out notes in time.

    ``aksaras`` are the syllabified akṣaras (for lengths), ``svaras`` the sāma note
    per akṣara, and ``groups`` the breath units (indices into ``aksaras``).
    """
    opts = options or TimingOptions()
    n = len(aksaras)
    if groups is None:
        groups = [list(range(n))]

    # Apply register shift by moving the tonic, not the notes: the sāma scale is
    # fixed relative to its śaḍja, so shifting the tonic shifts every note.
    tonic = opts.tonic_hz * (2.0 ** (opts.register_shift_semitones / 12.0))

    notes: list[Note] = []
    t = 0.0
    group_spans: list[tuple[int, int]] = []
    for gi, group in enumerate(groups):
        if not group:
            continue
        first_note = len(notes)
        for pos, idx in enumerate(group):
            ak = aksaras[idx]
            svara = int(svaras[idx]) if idx < len(svaras) and svaras[idx] else 4
            matra = max(ak.effective_matra if hasattr(ak, "effective_matra") else 1.0, 0.5)
            dur = matra * opts.matra_seconds
            is_first = pos == 0
            is_last = pos == len(group) - 1
            if is_last and len(group) > 1:
                dur *= opts.cadence_hold
            elif is_first:
                dur *= opts.line_start_hold
            dur *= opts.tempo
            dur = min(matra * opts.matra_seconds * opts.cadence_hold, max(dur, opts.matra_seconds * 0.55))

            f0 = tonic * 2.0 ** (SAMA_SVARAS[svara]["semitone"] / 12.0)
            phones = phone_lists[idx] if phone_lists and idx < len(phone_lists) else []
            notes.append(Note(
                ak_index=idx, svara=svara, start=t, duration=dur, f0_hz=f0,
                text=getattr(ak, "text", ""), phones=list(phones),
                is_cadence=is_last and len(group) > 1, is_line_start=is_first,
                is_line_end=is_last, group=gi,
            ))
            t += dur
        if gi != len(groups) - 1:
            t += opts.breath
            group_spans.append((first_note, len(notes) - 1))

    if len(groups) == 1:
        group_spans = [(0, len(notes) - 1)] if notes else []

    return TimingPlan(notes=notes, matra_seconds=opts.matra_seconds,
                      total_seconds=notes[-1].end if notes else 0.0,
                      groups=group_spans)


def note_frequencies(plan: TimingPlan) -> list[float]:
    return [n.f0_hz for n in plan.notes]
