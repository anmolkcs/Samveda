"""Text frontend: normalisation, syllabification, and the sāma-vikāra engine."""

from .unicodedata_ext import (
    normalize,
    to_devanagari,
    detect_script,
    devanagari_to_roman,
    roman_to_devanagari,
    strip_symbols,
    Script,
)
from .syllabify import (
    Aksara,
    SyllableView,
    syllabify,
    split_sandhi,
    text_to_aksaras,
)
from .vikara import (
    VIKARAS,
    Gana,
    GanaLine,
    apply_vikaras,
    classify_vikaras,
    parse_gana,
    load_gana,
)
from .svaras import (
    Svara,
    SvaraToken,
    SAMA_SVARAS,
    NARADIYA_VENU_MAP,
    svara_to_semitone,
    annotate_svaras,
)

__all__ = [
    "normalize",
    "to_devanagari",
    "detect_script",
    "devanagari_to_roman",
    "roman_to_devanagari",
    "strip_symbols",
    "Script",
    "Aksara",
    "SyllableView",
    "syllabify",
    "split_sandhi",
    "text_to_aksaras",
    "VIKARAS",
    "Gana",
    "GanaLine",
    "apply_vikaras",
    "classify_vikaras",
    "parse_gana",
    "load_gana",
    "Svara",
    "SvaraToken",
    "SAMA_SVARAS",
    "NARADIYA_VENU_MAP",
    "svara_to_semitone",
    "annotate_svaras",
]
