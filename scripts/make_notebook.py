import io
import json

Q = '"' * 3


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s.strip("\n").splitlines(True)}


cells = []

cells.append(md("""
# समध्वनि · Samadhvani — train a Sāmaveda chant voice on a free Kaggle GPU

**This notebook needs no manual steps.** Run all cells and it will:

1. install the repo and its dependencies (a few minutes);
2. **download real Sāmagāna audio** from archive.org — IISH's complete Kauthuma
   Āraṇyaka/Grāma-gāna set;
3. segment it into breath groups and pack a training corpus;
4. extract per-frame pitch (this is the slow CPU step, done once);
5. **train the voice model**, and print whether it beat its baseline;
6. render the same verse twice — rule engine vs trained voice — so you can hear the
   difference, with a melody score for each.

**Turn on the GPU first:** *Settings → Accelerator → GPU T4 x2* (or P100).
The notebook works on CPU too, just slower.

**What is trained, and what is not.** The melody is *not* learned: svaras come from
the notation and F0 is computed exactly. What the model learns is the singer's
spectral envelope — the part that makes one uḍgātṛ sound unlike another.

| stage | what | time (measured, mostly CPU) |
|---|---|---|
| download 24 clips | ~0.9 GB of MP3 | 2–5 min |
| segment + pack | 4 h → ~4 000 breath groups | ~1 min |
| pitch track | per-frame F0 | ~6 min |
| train | 0.25 M params, CPU-timed | ~1 min/epoch on CPU, seconds on a T4 |

So a full 4-hour run is roughly **20 minutes of compute**; nearly all of it is the
download and the pitch track, not the training.

Set `HOURS` below to control how much audio you use. **4 hours is the Vāgdhenu
scale** and is the default; 15.8 is available and does not need it. Start with 4.
"""))

cells.append(md("""
## 1 · Setup

Clones the repo (once) and checks that the environment can import it. It installs
**nothing that Kaggle already ships**, which is a deliberate rule explained in the
cell — the short version is that pip-managing numpy next to the image's compiled
scipy is how you get an unsolvable import error twenty minutes in.
"""))

cells.append(code("""
# --------------------------------------------------------------------------- #
# Setup.  Installs as little as possible, on purpose.
#
# An earlier version of this cell ran `pip install "numpy<2" scipy soundfile` to
# match this repo's local venv. On Kaggle that replaces the numpy the image's own
# scipy was *compiled* against, and the failure is not a clean "incompatible
# numpy" message: scipy dies partway through `import numpy as np` with
#     ModuleNotFoundError: No module named 'numpy.char'
# (numpy 2 removed numpy.char as a submodule; a 1.26 leftover reappeared).
# numpy reports 2.1.3, scipy is broken, and it needs a *new session* to recover.
#
# So: this cell never pip-installs numpy, scipy or torch. Everything the project
# uses -- numpy, scipy.signal.{butter,sosfilt,lfilter,fftconvolve}, torch -- is
# already in the image and already consistent with itself.
# --------------------------------------------------------------------------- #
import os, sys, subprocess

REPO = os.environ.get("SAMADHVANI_REPO", "")
if not REPO:
    if os.path.isdir("samveda/samadhvani"):
        REPO = "samveda"
    elif os.path.isdir("Samveda/samadhvani"):
        REPO = "Samveda"
    elif os.path.isdir("samadhvani"):
        REPO = "."
    else:
        r = subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/anmolkcs/Samveda.git", "samveda"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                "Could not clone the repository. Turn internet on "
                "(Settings -> Internet -> On) and re-run this cell. " + r.stderr[-400:])
        REPO = "samveda"
os.chdir(REPO)
sys.path.insert(0, os.getcwd())
print("repo:", os.getcwd())

# soundfile is the one genuinely missing piece: it decodes and, crucially, *seeks
# inside* MP3, so reading one phrase from a 44-minute clip costs milliseconds.
# --no-deps because its dependencies (cffi, numpy) ship with the image.
try:
    import soundfile as sf
    assert "MP3" in sf.available_formats(), "libsndfile built without MP3"
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "soundfile"], check=False)
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "soundfile is needed to decode the MP3s and could not be installed. "
            "Turn internet on (Settings -> Internet -> On) and re-run this cell."
        ) from exc

import numpy as np, torch
print("numpy", np.__version__, "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("  no GPU attached — Settings -> Accelerator -> GPU T4 x2. "
          "Training still works on CPU, just slower.")
"""))

cells.append(md("""
## 2 · Check the environment before spending 20 minutes on it

A download is slow and a broken import is instant, so prove the imports work
*first*. This cell exists because the previous version of this notebook failed at
the fetch step with a scipy/numpy mismatch that could have been detected here in
one second.
"""))

cells.append(code("""
# Fail fast: if these import, the rest of the notebook will run.
import scipy, importlib
from scipy.signal import butter, sosfilt, lfilter, fftconvolve

for mod in ("samadhvani.audio.mel", "samadhvani.data.dataset",
            "samadhvani.train.voice_trainer", "samadhvani.synth.voice_render",
            "samadhvani.pipeline"):
    importlib.import_module(mod)

from samadhvani.audio.mel import SR, HOP, N_MELS
print(f"imports OK — numpy {np.__version__}, scipy {scipy.__version__}, "
      f"{SR} Hz, hop {HOP}, {N_MELS} mel bands")
print(f"cuda available: {torch.cuda.is_available()}")
"""))

cells.append(md("""
## 3 · Settings

`HOURS` is the corpus size. 4 h matches the scale of the published reference system
(Vāgdhenu, ~5 h); more audio stops helping a model this small long before it stops
costing you session time. `CLIPS` is how many of IISH's 24 Āraṇyaka/Grāma-gāna
recordings to pull — 24 is all of them, and they are the ones whose text is
transcribed.
"""))

cells.append(code("""
HOURS   = 4.0      # corpus cap. 4 = Vāgdhenu scale. Raise to 15.8 for everything.
CLIPS   = 24       # IISH clips to download (1-24 are the transcribed books)
EPOCHS  = 60       # voice model epochs; early stopping usually fires before this
BATCH   = 16
WORK    = "work"
os.makedirs(WORK, exist_ok=True)
print(f"target: {HOURS} h from {CLIPS} clips, {EPOCHS} epochs max")
"""))

cells.append(md("""
## 4 · Get the audio

IISH Global's complete Kauthuma sāmagāna has been on archive.org since 2013 as
`IISHSamaVeda`. Clips 1–24 finish Grāma-gāna and Āraṇyaka-gāna, which are the two
books whose text is transcribed, and they total 15.8 hours.

The download is resumable: re-running skips clips already present at the right size,
so an interrupted session costs only what it had not finished.
"""))

cells.append(code("""
from samadhvani.data.dataset import fetch_iish, IISH_ALIGNED_CLIPS

clips = IISH_ALIGNED_CLIPS[:CLIPS]
files = fetch_iish(WORK + "/clips", clips=clips)
total_mb = sum(os.path.getsize(f) for f in files) / 1e6
print(f"\\n{len(files)}/{len(clips)} clips, {total_mb:.0f} MB")
if not files:
    print("No clips downloaded. Check that internet is enabled for this notebook "
          "(Settings -> Internet -> On).")
"""))

cells.append(md("""
## 5 · Build the corpus

Cut each recording into breath groups — the phrases a reciter actually sings in one
breath — and turn each into a log-mel frame stack.

Segmentation is by energy, not by the transcriber's timestamps. That is a deliberate
trade-off and worth understanding:

* **what it gives you:** supervised acoustics with no annotation work at all;
* **what it cannot give you:** which gāna each phrase is, so it learns *timbre*, not
  *melody*. Melody supervision needs the text aligned to audio, which is the next
  milestone (see `docs/DATA-ADDENDUM.md`).
"""))

cells.append(code("""
from samadhvani.data.dataset import build_corpus

rep = build_corpus(files, WORK + "/corpus.npz", WORK + "/corpus.jsonl",
                   max_hours=HOURS)
print(f"\\n{rep['segments']} segments, {rep['hours']:.2f} h, "
      f"{rep['frames_per_segment']} frames each, {rep['size_mb']:.0f} MB packed")
"""))

cells.append(md("""
## 6 · Pitch, once

F0 is recovered from the audio by autocorrelation — the melody-in-the-recording, not
the melody-in-the-notation, which is what the voice model conditions on. This is the
slowest CPU step; doing it before training means the GPU session is not spent on it.
"""))

cells.append(code("""
from samadhvani.train.voice_trainer import add_f0_to_corpus
add_f0_to_corpus(WORK + "/corpus.npz")
"""))

cells.append(md("""
## 7 · Train

Watch the **validation** number, not the training one. The model predicts a log-mel
*residual* against the corpus mean, so the first epoch is already at "average frame";
what matters is how far below that it gets.

The printed `improvement %` is the honest headline: if it is near zero, the model has
learned nothing about this voice and the render will sound like the rule engine.
"""))

cells.append(code("""
from samadhvani.train.voice_trainer import VoiceConfig, train_voice

metrics = train_voice(
    WORK + "/corpus.npz",
    WORK + "/voice.pt",
    VoiceConfig(epochs=EPOCHS, batch_size=BATCH,
                device="cuda" if torch.cuda.is_available() else "cpu"),
)
metrics
"""))

cells.append(md("""
## 8 · Listen — rule engine vs trained voice

Same verse, same melody, same timing. The only difference is the spectral envelope.
Both are scored by decoding pitch from the waveform at each scheduled note, so the
number reflects what came out, not what went in.

**The one guarantee that matters:** the trained voice must not lose melody. The
correction is bounded and frequency-tapered precisely so it cannot.
"""))

cells.append(code("""
from samadhvani.pipeline import chant, ChantOptions
from samadhvani.audio.mel import SR
from samadhvani.eval.metrics import _median_f0
from samadhvani.text.svaras import SAMA_SVARAS

semis = {k: v["semitone"] for k, v in SAMA_SVARAS.items()}

def per_note(audio, notes, tonic):
    out = []
    for n in notes:
        mid = n.start + n.duration * 0.5
        seg = audio[int((mid - 0.06) * SR):int((mid + 0.06) * SR)]
        if len(seg) < 256:
            out.append(None); continue
        f = _median_f0(seg, SR, 70, 400)
        out.append(min(semis, key=lambda k: abs(semis[k] * 100
                                                - 1200 * np.log2(f / tonic)))
                   if f > 0 else None)
    return out

def score(c):
    got = per_note(np.asarray(c.audio, dtype=np.float32), c.notes, c.tonic_hz)
    ok = sum(1 for r, g in zip(c.svaras, got) if r == g)
    return 100.0 * ok / max(len(c.svaras), 1)

verse = "हाउ ४ अ॒ग्निमी॑ळे ३ पु॒रोहि॑तं २ य॒ज्ञस्य॑ १ दे॒वमृ॒त्विज॑म्"
ckpt = WORK + "/voice.pt"

a = chant(verse, ChantOptions(mode="gana", laya=0.4))
b = chant(verse, ChantOptions(mode="gana", laya=0.4, engine="voice", checkpoint=ckpt))
a.save("01_rule_engine.wav")
b.save("02_trained_voice.wav")
print(f"rule engine  : {a.duration:5.1f} s, {len(a.notes):2d} notes, melody {score(a):5.1f}%")
print(f"trained voice: {b.duration:5.1f} s, {len(b.notes):2d} notes, melody {score(b):5.1f}%")
print("\\nfiles: 01_rule_engine.wav  02_trained_voice.wav")

try:
    from IPython.display import Audio, display
    display(Audio("01_rule_engine.wav"))
    display(Audio("02_trained_voice.wav"))
except Exception:
    pass
"""))

cells.append(md("""
## 9 · What you just built, and what it is not

**It is** a voice model trained on real Sāmagāna, and a pipeline that goes from
archive.org to audio with no manual annotation. The melody is notation-driven and
exact; the timbre is measured from the recordings.

**It is not** a full neural TTS. Two honest gaps:

1. **Phase is reconstructed, not generated.** The vocoder is a harmonic synthesiser
   driven by the pitch contour, so sustained notes have a slight synthetic edge. A
   neural vocoder is the fix, and it needs the same corpus you just packed.
2. **Melody is not learned yet.** It comes from the notation, which for a gāna text
   is exactly right. For *un-notated* text, nothing here invents a melody — that is
   why the demo refuses to guess and says so.

## Where the next real gain is

Text-to-audio alignment. The transcriber's manuscripts print a clip number and a
minute:second stamp beside every line, so the mapping exists — it needs transcribing.
With it, the model conditions on phonemes rather than just pitch, and then it can
sing text it has never seen in this voice.
"""))

cells.append(md("""
## 10 · Try your own verse

Paste any gāna text with its svara numerals (१–७) and it will be read from the
notation. Without numerals it will use the school's default melodic skeleton — and
say so, rather than pretending to know the sāman.
"""))

cells.append(code("""
mine = "ओ३म् भूर्भुव॒: स्व॒: तत्स॑वि॒तुर्वरेᣚण्यं॒ भर्गो॑ दे॒वस्य॑ धीमहि"
c = chant(mine, ChantOptions(mode="auto", laya=0.4, engine="voice", checkpoint=ckpt))
c.save("03_mine.wav")
print("mode:", c.mode, "| notes:", len(c.notes), f"| {c.duration:.1f} s")
for note in c.caveats:
    print(" ·", note)
try:
    from IPython.display import Audio, display
    display(Audio("03_mine.wav"))
except Exception:
    pass
"""))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
                   "dataSources": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with io.open("notebooks/kaggle_samadhvani.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", len(cells), "cells")
