"""Samadhvani — a Sāmaveda sāmagāna TTS.

A small, self-contained chant synthesis stack:

    text  ->  phone  ->  svara  ->  timing  ->  audio

* ``samadhvani.text``   – script/Sanskrit normalisation, syllabification, sāma-vikāra engine
* ``samadhvani.phones`` – the phone inventory and Devanāgarī -> phone mapping
* ``samadhvani.synth``  – the acoustic model (a tiny voiced-source + formant network the
                          Kaggle trainer can learn) and a NumPy fallback synthesiser
* ``samadhvani.models`` – the Kāṇva/Kauthuma/rāṇāyanīya sāma-gāna "model" library
"""

__version__ = "0.1.0"

__all__ = ["text", "phones", "synth", "models", "train", "data", "eval"]

def __getattr__(name):
    if name in ("chant", "Chant", "ChantOptions"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)
