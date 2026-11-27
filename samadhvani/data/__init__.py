"""Corpus building and the data-source audit.

Note on imports: :mod:`samadhvani.data.build` runs as a module
(``python -m samadhvani.data.build``), so this ``__init__`` deliberately does *not*
eagerly import it — doing so makes the module appear in ``sys.modules`` before the
runtime executes it and produces a confusing RuntimeWarning.  Use the submodules
directly, or the re-exports below which import lazily.
"""

from __future__ import annotations

from .sources import SOURCES, Licence, Source, audit

__all__ = [
    "SOURCES", "Licence", "Source", "audit",
    "Record", "align_clip", "build_audio_manifest", "build_text_corpus",
    "measure_contours",
]


def __getattr__(name):
    if name in ("Record", "align_clip", "build_audio_manifest",
                "build_text_corpus", "measure_contours"):
        from . import build
        return getattr(build, name)
    raise AttributeError(name)
