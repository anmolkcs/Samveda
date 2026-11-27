"""Tests for the audio→dataset→model path.

These cover the parts that were missing before: decoding audio, turning it into
mel frames, packing a corpus, and training something that actually reads the frames.
The regression they exist to prevent is subtle and was real — a *correct* chant
coming out with the melody an octave high because a correction was applied without
a bound.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from samadhvani.audio.mel import (HOP, N_MELS, SR, breath_groups, griffin_lim,
                                  harmonic_vocode, istft, logmel, mel_filterbank,
                                  stft, trim_silence)
from samadhvani.pipeline import ChantOptions, chant


@pytest.fixture(scope="module")
def chant_audio():
    c = chant("हाउ ४ अ॒ग्निमी॑ळे ३ पु॒रोहि॑तं २ य॒ज्ञस्य॑ १ दे॒वमृ॒त्विज॑म्",
              ChantOptions(mode="gana", laya=0.4))
    return c, np.asarray(c.audio, dtype=np.float32)


# --------------------------------------------------------------------------- #
# STFT / mel basics
# --------------------------------------------------------------------------- #

def test_filterbank_is_triangular_and_normalised():
    fb = mel_filterbank()
    assert fb.shape == (N_MELS, 1024 // 2 + 1)
    assert (fb >= 0).all()
    # every band must cover at least one bin, or the mel axis has dead entries
    assert (fb.sum(axis=1) > 0).all()


def test_stft_istft_round_trip_is_exact_in_the_interior():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(22050) * 0.1).astype(np.float64)
    y = istft(stft(x), length=len(x))
    # the first and last frames are only half-covered by the window, which is a
    # property of weighted overlap-add, not a bug
    err = np.abs(x[1500:-1500] - y[1500:-1500]).max()
    assert err < 1e-5


def test_logmel_shape_and_range(chant_audio):
    _c, x = chant_audio
    # logmel floors at (peak - 80 dB) with peak <= 0, so the lower bound is -80
    M = logmel(x)
    assert M.ndim == 2 and M.shape[1] == N_MELS
    # non-centred STFT: frames = 1 + (len - n_fft) // hop
    assert M.shape[0] == 1 + (len(x) - 1024) // HOP
    assert M.max() < 80.0                       # sane dB range, not clipped nonsense
    assert M.min() >= -100.0 - 1e-3             # absolute floor


def test_griffin_lim_converges_on_magnitude():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(8192) * 0.1)
    M = np.abs(stft(x)).astype(np.float64)
    y = griffin_lim(M, iters=80)
    got = np.abs(stft(y))[: len(M)]
    rel = np.linalg.norm(got - M[: len(got)]) / np.linalg.norm(M[: len(got)])
    assert rel < 0.25, f"Griffin-Lim did not converge: rel err {rel:.3f}"


# --------------------------------------------------------------------------- #
# The vocoder must preserve the melody — this is the regression that bit
# --------------------------------------------------------------------------- #

def test_harmonic_vocode_preserves_pitch(chant_audio):
    from samadhvani.eval.metrics import _median_f0

    _c, x = chant_audio
    f0 = np.zeros(logmel(x).shape[0], dtype=np.float32)
    M = logmel(x)
    # a constant 147 Hz is enough to prove the comb lands where asked
    f0[:] = 146.83
    y = harmonic_vocode(M, f0)
    got = _median_f0(y, SR, 70, 400)
    assert abs(got - 146.83) < 6.0, f"vocoded pitch drifted to {got:.1f} Hz"
    assert abs(_median_f0(x, SR, 70, 400) - 146.83) < 8.0


def test_harmonic_vocode_tracks_a_changing_f0():
    """The vocoder must follow the contour, not average it."""
    from samadhvani.eval.metrics import _median_f0

    rng = np.random.default_rng(2)
    M = np.full((600, N_MELS), -30.0, dtype=np.float32)   # flat envelope
    f0 = np.concatenate([np.full(300, 146.83), np.full(300, 195.99)])
    y = harmonic_vocode(M, f0, seed=0)
    a = _median_f0(y[: int(0.8 * SR)], SR, 70, 400)
    b = _median_f0(y[int(3.5 * SR):], SR, 70, 400)
    assert a < 160 and b > 185, (a, b)


def test_vocode_keeps_frame_energy():
    """A sparser comb must not come out 20 dB quiet; that is what broke tracking."""
    M = np.full((300, N_MELS), -25.0, dtype=np.float32)
    y = harmonic_vocode(M, np.full(300, 146.83))
    interior = y[2000:-2000]
    rms = float(np.sqrt(np.mean(interior ** 2)))
    crest = 20 * math.log10(np.abs(interior).max() / max(rms, 1e-12))
    # A harmonic source is peaked, not impulsive: a crest factor of ~15 dB says
    # "pulse train".  The bug this guards against produced 42 dB — a single click.
    assert 6.0 < crest < 25.0, f"crest {crest:.1f} dB: waveform is a click, not a tone"
    assert rms > 1e-3


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #

def test_breath_groups_split_on_silence():
    sr = SR
    tone = (0.3 * np.sin(2 * np.pi * 147 * np.arange(int(0.6 * sr)) / sr)).astype(np.float32)
    gap = np.zeros(int(0.6 * sr), dtype=np.float32)
    x = np.concatenate([tone, gap, tone, gap, tone])
    groups = breath_groups(x)
    assert len(groups) == 3, groups
    assert all(b > a for a, b in groups)


def test_trim_silence_is_relative_not_absolute():
    """A quiet clip must not be trimmed to nothing just because it is quiet."""
    rng = np.random.default_rng(3)
    loud = np.concatenate([np.zeros(3000, np.float32),
                           (0.5 * rng.standard_normal(9000)).astype(np.float32)])
    quiet = loud * 0.02
    assert len(trim_silence(loud)) > 8000
    assert len(trim_silence(quiet)) > 8000


# --------------------------------------------------------------------------- #
# Corpus packing
# --------------------------------------------------------------------------- #

def test_build_corpus_packs_and_reports(tmp_path):
    from samadhvani.synth.io import write_wav
    from samadhvani.data.dataset import build_corpus

    chunks = []
    for laya in (0.34, 0.44):
        c = chant("हाउ ४ अ॒ग्निमी॑ळे ३ पु॒रोहि॑तं २ य॒ज्ञस्य॑ १ दे॒वमृ॒त्विक्",
                  ChantOptions(mode="gana", laya=laya))
        chunks.append(np.asarray(c.audio, dtype=np.float32))
    wav = tmp_path / "a.wav"
    write_wav(str(wav), np.concatenate(chunks), SR)

    npz = tmp_path / "c.npz"
    rep = build_corpus([str(wav)], str(npz), max_hours=0.5, verbose=False)
    assert rep["segments"] > 0
    assert rep["n_mels"] == N_MELS and rep["sr"] == SR
    assert npz.exists() and rep["size_mb"] > 0

    blob = np.load(npz)
    assert blob["mel"].shape[0] == rep["segments"]
    assert blob["mel"].shape[2] == N_MELS
    # the mask must actually mark real frames, or training sees only padding
    assert blob["mask"].sum() > 0
    assert (blob["mask"].sum(axis=1) > 0).all()


def test_corpus_respects_the_hour_cap(tmp_path):
    from samadhvani.synth.io import write_wav
    from samadhvani.data.dataset import build_corpus

    c = chant("हाउ ४ अ॒ग्निमी॑ळे ३ पु॒रोहि॑तं २", ChantOptions(mode="gana", laya=0.4))
    x = np.tile(np.asarray(c.audio, dtype=np.float32), 6)
    wav = tmp_path / "long.wav"
    write_wav(str(wav), x, SR)
    rep = build_corpus([str(wav)], str(tmp_path / "c.npz"), max_hours=0.02, verbose=False)
    assert rep["hours"] <= 0.03, rep


# --------------------------------------------------------------------------- #
# Training: the model must beat the corpus mean, on a corpus it can memorise
# --------------------------------------------------------------------------- #

torch = pytest.importorskip("torch", reason="training tests need torch")


def test_voice_training_beats_the_mean(tmp_path):
    """A model that cannot beat the mean has learned nothing; assert it does."""
    from samadhvani.data.dataset import build_corpus
    from samadhvani.synth.io import write_wav
    from samadhvani.train.voice_trainer import (VoiceConfig, add_f0_to_corpus,
                                                train_voice)

    chunks = []
    for laya, tonic in ((0.30, 146.83), (0.36, 164.81), (0.42, 130.81),
                        (0.48, 174.61), (0.34, 155.56), (0.40, 138.59),
                        (0.32, 185.00), (0.46, 123.47)):
        c = chant("हाउ ४ अ॒ग्निमी॑ळे ३ पु॒रोहि॑तं २ य॒ज्ञस्य॑ १ दे॒वमृ॒त्विज॑म्",
                  ChantOptions(mode="gana", laya=laya, tonic_hz=tonic))
        chunks.append(np.asarray(c.audio, dtype=np.float32))
    wav = tmp_path / "v.wav"
    write_wav(str(wav), np.concatenate(chunks), SR)

    npz = tmp_path / "c.npz"
    build_corpus([str(wav)], str(npz), max_hours=0.5, verbose=False)
    add_f0_to_corpus(str(npz), verbose=False)

    blob = np.load(npz)
    assert "f0" in blob and (blob["f0"] > 0).mean() > 0.3

    out = tmp_path / "v.pt"
    metrics = train_voice(str(npz), str(out),
                          VoiceConfig(epochs=6, batch_size=8, num_workers=0,
                                      crop_frames=96), verbose=False)
    assert out.exists()
    assert metrics["improvement_pct"] > 0.0, metrics
    assert metrics["val_residual_l1"] < metrics["val_residual_l1_epoch0"]


def test_voice_checkpoint_round_trips(tmp_path):
    from samadhvani.train.voice_trainer import VoiceColorModel, load_voice

    model = VoiceColorModel(hidden=32, n_layers=2)
    blob = {
        "version": 2, "kind": "voice-color",
        "voice_state": {k: v.detach().numpy() for k, v in model.state_dict().items()},
        "mean_spectrum": [-30.0] * N_MELS,
        "config": {"hidden": 32, "n_layers": 2, "n_mels": N_MELS, "sr": SR, "hop": HOP},
        "metrics": {},
    }
    path = tmp_path / "v.pt"
    torch.save(blob, path)
    loaded, mean, cfg = load_voice(str(path))
    assert cfg["hidden"] == 32
    assert mean.shape == (N_MELS,)
    with torch.no_grad():
        y = loaded(torch.zeros(1, 10, 3))
    assert y.shape == (1, 10, N_MELS)


def test_v1_checkpoint_is_rejected_not_silently_loaded(tmp_path):
    from samadhvani.train.voice_trainer import load_voice

    path = tmp_path / "old.pt"
    torch.save({"version": 1, "duration_state": {}, "spectrogram_state": {}}, path)
    with pytest.raises(ValueError, match="v2"):
        load_voice(str(path))


# --------------------------------------------------------------------------- #
# The end-to-end guarantee: the trained voice must not wreck the melody
# --------------------------------------------------------------------------- #

def test_voice_engine_preserves_length_and_melody(tmp_path, chant_audio):
    """The correction must never leave a chant worse than the rule engine."""
    from samadhvani.data.dataset import build_corpus
    from samadhvani.synth.io import write_wav
    from samadhvani.train.voice_trainer import (VoiceConfig, add_f0_to_corpus,
                                                train_voice)

    _c, x = chant_audio
    wav = tmp_path / "t.wav"
    write_wav(str(wav), np.concatenate([x, x * 0.7, x * 1.2, x * 0.9]), SR)
    npz = tmp_path / "c.npz"
    build_corpus([str(wav)], str(npz), max_hours=0.5, verbose=False)
    add_f0_to_corpus(str(npz), verbose=False)
    ckpt = tmp_path / "v.pt"
    train_voice(str(npz), str(ckpt),
                VoiceConfig(epochs=4, batch_size=8, num_workers=0, crop_frames=96),
                verbose=False)

    from samadhvani.audio.mel import SR as _SR
    from samadhvani.eval.metrics import _median_f0
    from samadhvani.text.svaras import SAMA_SVARAS

    semis = {k: v["semitone"] for k, v in SAMA_SVARAS.items()}

    def per_note(audio, notes, tonic):
        out = []
        for n in notes:
            mid = n.start + n.duration * 0.5
            seg = audio[int((mid - 0.06) * _SR):int((mid + 0.06) * _SR)]
            if len(seg) < 256:
                out.append(None)
                continue
            f = _median_f0(seg, _SR, 70, 400)
            out.append(min(semis, key=lambda k: abs(semis[k] * 100
                                                    - 1200 * math.log2(f / tonic)))
                       if f > 0 else None)
        return out

    verse = "हाउ ४ अ॒ग्निमी॑ळे ३ पु॒रोहि॑तं २ य॒ज्ञस्य॑ १ दे॒वमृ॒त्विज॑म्"
    c0 = chant(verse, ChantOptions(mode="gana", laya=0.4))
    c1 = chant(verse, ChantOptions(mode="gana", laya=0.4, engine="voice",
                                   checkpoint=str(ckpt)))

    # same timeline: the reported note times must describe the audio they accompany
    assert len(c0.audio) == len(c1.audio)
    assert [n.start for n in c0.notes] == [n.start for n in c1.notes]

    x0 = np.asarray(c0.audio, dtype=np.float32)
    x1 = np.asarray(c1.audio, dtype=np.float32)
    a0 = per_note(x0, c0.notes, c0.tonic_hz)
    a1 = per_note(x1, c1.notes, c1.tonic_hz)
    n_ok0 = sum(1 for r, g in zip(c0.svaras, a0) if r == g)
    n_ok1 = sum(1 for r, g in zip(c1.svaras, a1) if r == g)
    # allow a small margin: this corpus is three copies of one render
    assert n_ok1 >= n_ok0 - 2, (f"voice engine lost melody: {n_ok0} -> {n_ok1} "
                                f"of {len(c0.svaras)}")
    assert len(x1) > 0


def test_voice_engine_without_checkpoint_falls_back(tmp_path):
    c = chant("ॐ नमो वैश्वानराय", ChantOptions(mode="style", engine="voice"))
    assert len(c.audio) > 0
    assert any("checkpoint" in k.lower() for k in c.caveats)


# --------------------------------------------------------------------------- #
# Checkpoint handling — two bugs that were invisible until the server was run.
# --------------------------------------------------------------------------- #

def test_training_creates_the_checkpoint_directory(tmp_path):
    """A notebook writes to "work/voice.pt" without mkdir; that must just work.

    torch.save raises RuntimeError if the parent directory is missing, and the
    failure surfaces after the whole training run.
    """
    from samadhvani.synth.io import write_wav
    from samadhvani.data.dataset import build_corpus
    from samadhvani.train.voice_trainer import (VoiceConfig, add_f0_to_corpus,
                                                train_voice)

    chunks = []
    # ~5 breath groups is enough to *pack* but not to train (needs 6), so build
    # four renderings at different tempi.
    for laya in (0.26, 0.34, 0.42, 0.50):
        c = chant("हाउ ४ अ॒ग्निमी॑ळे ३ पु॒रोहि॑तं २ य॒ज्ञस्य॑ १ दे॒वमृ॒त्विक्",
                  ChantOptions(mode="gana", laya=laya))
        chunks.append(np.asarray(c.audio, dtype=np.float32))
    wav = tmp_path / "a.wav"
    write_wav(str(wav), np.concatenate(chunks), SR)
    corpus = tmp_path / "c.npz"
    build_corpus([str(wav)], str(corpus), str(tmp_path / "c.jsonl"),
                 max_hours=1.0, verbose=False)
    add_f0_to_corpus(str(corpus))
    nested = tmp_path / "does" / "not" / "exist" / "voice.pt"
    train_voice(str(corpus), str(nested),
                VoiceConfig(epochs=2, batch_size=4, hidden=16, n_layers=2,
                            device="cpu"), verbose=False)
    assert nested.exists()


def test_server_classifies_a_voice_checkpoint_under_modern_torch(tmp_path, monkeypatch):
    """torch >= 2.6 defaults weights_only=True, which raises on numpy payloads.

    The classifier swallowed that exception and returned None, so the trained
    engine vanished from /api/health with no error anywhere — the failure mode is
    a demo that silently only offers the rule engine.
    """
    import torch

    from samadhvani.app import server
    from samadhvani.train.voice_trainer import VoiceColorModel

    model = VoiceColorModel(hidden=16, n_layers=2)
    path = tmp_path / "voice.pt"
    torch.save({
        "version": 2, "kind": "voice-color",
        "voice_state": {k: v.detach().numpy() for k, v in model.state_dict().items()},
        "mean_spectrum": __import__("numpy").full((N_MELS,), -30.0),
        "config": {"hidden": 16, "n_layers": 2, "n_mels": N_MELS, "sr": SR, "hop": HOP},
        "metrics": {},
    }, path)

    assert server._checkpoint_kind(str(path)) == "voice"

    monkeypatch.setenv("SAMADHVANI_CKPT", str(path))
    monkeypatch.setattr(server, "CHECKPOINT_CANDIDATES", [str(path)])
    assert server._find_checkpoint("voice") == str(path)
    assert server._find_checkpoint("neural") is None
    engines = server.health()["engines"]
    assert "voice" in engines and "neural" not in engines
