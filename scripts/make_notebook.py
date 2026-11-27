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
# समध्वनि · Samadhvani — train a Sāmaveda chant TTS on a free Kaggle GPU

This notebook takes you from an empty Kaggle session to a chant synthesiser you can
listen to, well inside a single T4 session.

**What gets trained.** Not a full waveform model. Two small ones:

| model | learns | params | why |
|---|---|---|---|
| `DurationPredictor` | how long each phone is held | ~0.4 M | the biggest intelligibility win; trains in minutes |
| `SpectrogramModel` | F1/F2/F3 + voice/noise per frame | ~2.5 M | small enough to converge on 3–5 h of audio |

**What does *not* get trained.** The melody. F0 comes from the svara sequence and the
sāma scale, computed exactly. A network that predicted pitch could only add drift.

**Before you run it.** Read section 3. Training data for Sāma chant is the hard part
and the licences are not uniform — the audit below prints exactly what you may and may
not use. Do not quietly train on a YouTube upload and ship the weights.

**Turn on the GPU:** *Settings → Accelerator → GPU T4 x2* (or P100).
"""))

cells.append(md("""
## 1 · Setup

The package is a normal Python project (`samadhvani/`). This installs what is missing
and pulls the repo into the session.

> If the notebook is running from the repo itself (recommended), set `REPO = "."`.
"""))

cells.append(code("""
import os, sys, subprocess

REPO = os.environ.get("SAMADHVANI_REPO", ".")
if not os.path.isdir(os.path.join(REPO, "samadhvani")):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/anmolkcs/Samveda.git", "samveda"], check=True)
    REPO = "samveda"

os.chdir(REPO)
sys.path.insert(0, os.getcwd())

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "numpy", "scipy", "soundfile"], check=False)

import torch, numpy as np
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("numpy", np.__version__)
"""))

cells.append(md("""
## 2 · Check the pipeline works before spending GPU time

A 20-second sanity pass. If the frontend is broken, training on it wastes a GPU-hour,
so verify text → phones → svara → audio first.
"""))

cells.append(code("""
from samadhvani.pipeline import chant, ChantOptions

verse = "हाउ अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म्"

c = chant(verse, ChantOptions(mode="auto"))
print(f"mode      : {c.mode}")
print(f"aksaras   : {len(c.syllables)}")
print(f"svaras    : {c.svaras}")
print(f"phones    : {c.phones[:5]} ...")
print(f"duration  : {c.duration:.2f} s")
c.save("sanity.wav")

try:
    from IPython.display import Audio, display
    display(Audio("sanity.wav"))
except Exception:
    print("(no notebook audio here; open sanity.wav)")
"""))

cells.append(md("""
## 3 · The data question (read this)

Two very different resources, and only one of them is the bottleneck:

* **Text — complete and free.** The four Sāma-gāna books (Grāma-geya 1,198 gāna,
  Āraṇyaka 296, Ūha 936, Ūhya 209) were typeset with Vedic accents and sāma numerals in
  January 2025. That is the whole Kauthuma corpus: **2,639 gāna**. "Free" is not "open":
  the typesetting is `PPN ©` and the download terms are fair-use-for-education.
* **Audio — 67 h, none of it openly licensed.** The find is IISH's complete Kauthuma
  sāmagāna (43.2 h, archive.org since 2013), whose transcription carries clip numbers
  and minute:second stamps — so **15.8 h arrive pre-aligned**, three times what the
  Vāgdhenu paper used, with no annotation work.

**Two axes, kept apart.** *May I redistribute?* is a licence question; *may I train on
it and run it for teaching?* is an intended-use question. Under the default
`educational` profile (not distributed, not commercial) ~67 h are trainable. None of it
is shippable.

Run the audit and see both numbers.
"""))

cells.append(code("""
from samadhvani.data.sources import audit
report = audit()                 # educational profile, the default
report_open = audit(verbose=False, profile="open")   # what survives publication
print("trainable:", report["audio_hours_trainable"], "h")
print("releasable:", report_open["audio_hours_trainable"], "h  <- publish this and your corpus vanishes")
"""))

cells.append(md("""
### 3a · Two honest ways forward

**A — text-only (works today, no audio).** Build the corpus from the gāna books. This
supervises durations and melody, but not timbre: it will not learn a particular
singer's voice. Still worth doing — it produces the correct svara/phone alignments that
a later audio run fine-tunes from.

**B — five hours of your own recording (the real answer).** Consented, single singer,
one microphone, aligned at akṣara level. That beats every public source on the list.
Protocol in `docs/DATA.md`. The alignment needs only **one timestamp per breath group**,
because Sāma timing really is mātrā-proportional — 30 seconds of annotation for a
10-minute recording.
"""))

cells.append(code("""
import os
os.makedirs("work", exist_ok=True)

# One verse per line. Replace with text extracted from the gāna books — or paste
# your own gāna text with its svara numerals, which the parser reads directly.
sample = "\\n".join([
    "हाउ अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म्",
    "होता॑रं रत्न॒धात॑मम्",
    "इ॒षे त्वो॒र्जे त्वा॑ वा॒यव॑ स्थ दे॒वो व॑: सवि॒ता प्रार्प॑यतु",
    "अग्न॒ आयूं॑षि प॒वसे॒ आ सु॒वीर्यं॒ यथा॑ देवा॒ना न म॑र्त्यः",
    "हाउ होवा अ॒ग्न आयूं॑षि प॒वसे॑ हाउ होवा",
    "ओ३म् भूर्भुव॒: स्व॒: तत्स॑वि॒तुर्वरे᳚ण्यं॒ भर्गो॑ दे॒वस्य॑ धीमहि",
]) + "\\n"
open("work/sample.txt", "w", encoding="utf-8").write(sample * 24)   # stand in for a real corpus

from samadhvani.data.build import build_text_corpus
n = build_text_corpus(["work/sample.txt"], "work/text_corpus.jsonl")
print(f"text corpus: {n} records")

# With real recordings, this is the shape (see docs/DATA.md):
#   from samadhvani.data.build import build_audio_manifest
#   n = build_audio_manifest("work/clips", "work/manifest.jsonl")
#
# sidecar format, one per clip:
#   {"audio": "clip01.wav", "text": "...gana text...",
#    "line_starts": [0.0, 4.2, 9.8], "school": "kauthuma", "licence": "open"}
"""))

cells.append(md("""
## 4 · Train

Two stages. The duration model converges in minutes even on CPU; the spectral model
wants the GPU. Watch the *validation* curve, not the training curve.
"""))

cells.append(code("""
from samadhvani.train.trainer import TrainConfig, train

cfg = TrainConfig(
    epochs_duration=40,
    epochs_spectral=60,
    batch_size=16,
    lr=3e-4,
    hidden=128,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

ckpt = train("work/text_corpus.jsonl", "work/samadhvani.pt", cfg)
print("checkpoint:", ckpt, round(os.path.getsize(ckpt) / 1e6, 1), "MB")
"""))

cells.append(md("""
## 5 · Listen: rule engine vs trained engine

The same verse through both paths. The formant engine is the baseline — if the trained
model does not beat it, the checkpoint is not ready, and you can hear where it fails
(usually the vowels smear, which means the formant loss needs more weight or the data
more variety).

Note what is *identical* in both: the melody. The svara sequence is given by the
tradition; only the phonetics come from the model.
"""))

cells.append(code("""
from samadhvani.pipeline import chant, ChantOptions
from samadhvani.eval import evaluate

for label, opts in [("formant", ChantOptions(engine="formant")),
                    ("neural", ChantOptions(engine="neural",
                                            checkpoint="work/samadhvani.pt"))]:
    c = chant(verse, opts)
    fn = label + ".wav"
    c.save(fn)
    rep = evaluate(c)
    sm = rep.svara_metrics
    print(f"--- {label} ---")
    print(f"  svara accuracy {sm.get('svara_accuracy', 0):.0%}"
          f" | pitch err {sm.get('median_abs_cents_error', 0):.1f} cents"
          f" | periodicity {rep.audio_metrics['periodicity']:.3f}")
    try:
        from IPython.display import Audio, display
        display(Audio(fn))
    except Exception:
        print("  (open", fn, ")")
"""))

cells.append(md("""
## 6 · Score the melody from the audio, not from the plan

`svara_accuracy_at_notes` decodes pitch inside each scheduled note and checks which
sāma svara it lands on. The evaluator is only trustworthy because it is
octave-corrected: a naive autocorrelation tracker reports a 147 Hz note as 294 Hz and
halves the score of perfect singing.
"""))

cells.append(code("""
from samadhvani.eval.metrics import svara_accuracy_at_notes

c = chant(verse, ChantOptions(engine="formant"))
res = svara_accuracy_at_notes(c)
print(f"checked {res['n_notes']} notes | accuracy {res['accuracy']:.0%}"
      f" | median |cents| error {res['median_abs_cents_error']:.1f}")
print("per note:", res["per_note"][:8], "...")
"""))

cells.append(md("""
## 7 · Measure real melody contours (once you have audio)

The skeletons in `samadhvani/models/sama_models.py` are prior knowledge. With
recordings you can replace them with *measured* contours: the builder takes the median
svara per position across utterances and writes JSON in the same schema, which the
pipeline then prefers.
"""))

cells.append(code("""
from samadhvani.data.build import measure_contours
k = measure_contours("work/text_corpus.jsonl", "work/models_learned.json")
print(f"wrote {k} measured models")

blob = json.load(open("work/models_learned.json")) if "json" in dir() else None
if blob is None:
    import json
    blob = json.load(open("work/models_learned.json"))
for key, entry in blob["models"].items():
    print(key, "->", [s["svara"] for s in entry["sections"][0]][:12], "...")
"""))

cells.append(md("""
## 8 · Save your work

Kaggle sessions are ephemeral. Save the checkpoint to *Output* so you can download it
or attach it to a later notebook as a dataset. A checkpoint this small is easy to keep.
"""))

cells.append(code("""
import shutil
shutil.copy("work/samadhvani.pt", "samadhvani.pt")
print(f"samadhvani.pt: {os.path.getsize('samadhvani.pt') / 1e6:.1f} MB")
print("also written: sanity.wav, formant.wav, neural.wav, work/models_learned.json")
"""))

cells.append(md("""
## Where to go next

1. **Get audio.** Everything above is a demonstration on synthetic targets. The moment
   you have a few hours of real recitation, the spectral model starts learning timbre
   rather than reproducing the inventory table, and the difference is large.
   See `docs/DATA.md`.
2. **Then a vocoder.** With 10+ hours you can replace the formant cascade with HiFi-GAN.
   The acoustic model needs no change, because it predicts formant tracks and levels,
   not waveforms.
3. **Widen the schools.** Kauthuma, Rāṇāyanīya and Jaiminīya differ most in stobhas and
   register. The `school` setting already switches stobhas and shifts the tonic; a
   corpus per school is the next real gain.
4. **Do a listening test.** A blind MOS panel of 5–8 Sāma reciters, with traditional
   recordings as controls. A proxy metric is not a MOS.
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
