"""Acoustic layer: timing, prosody, and the two synthesis engines."""

from .timing import Note, TimingOptions, TimingPlan, plan_timing
from .prosody import amplitude_envelope, breath_positions, f0_track, voicing_mask
from .engine import synthesize_formant
from .io import read_wav, wav_bytes, write_wav

__all__ = [
    "Note",
    "TimingOptions",
    "TimingPlan",
    "plan_timing",
    "f0_track",
    "voicing_mask",
    "breath_positions",
    "amplitude_envelope",
    "synthesize_formant",
    "wav_bytes",
    "write_wav",
    "read_wav",
]
