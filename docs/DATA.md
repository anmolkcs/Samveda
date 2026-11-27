# Getting the data — the only part that is actually hard

Text for Sāma chant is abundant, open, and already machine-readable. Audio with a
transcript is rare, and much of what exists is not licensed for training. Everything
in this document is about closing that gap without doing something you would have to
apologise for later.

Run `python -m samadhvani.data.build audit` for the live version of this table.

---

## 1 · What exists

### Text — open, large, ready

| source | what | size | notes |
|---|---|---|---|
| Sanskrit Documents, Sāma-gāna Book I | Grāma-geya | 1,198 gāna | accented, with sāma numerals; the best open gāna e-text |
| Book II | Āraṇyaka | 296 gāna | PDF |
| Book III | Ūha | 936 gāna | PDF |
| Book IV | Ūhya | 209 gāna | PDF |
| Vedic Heritage Portal | Kauthuma saṃhitā | 1,875 ṛc | government of India; text is openly published |
| Internet Archive | PR Iyer transcriptions | — | page-timed gāna, public domain scans |

That is roughly **7,500 gāna of training text**, and it is enough to supervise
durations and melody.

### Audio — small, and mostly not yours to train on

| source | hours | licence | verdict |
|---|---|---|---|
| Vedic Heritage Portal recitations | ~6 h | published for listening | **write for permission**; do not assume |
| YouTube sāmagāna uploads | ~20 h | per-video, usually unstated | **do not train a released model** |
| Your own recordings | 0 h until you record | yours | **the recommended path** |

The uncomfortable fact: there is no large open Sāma recitation corpus. Any project
that claims one either has a private arrangement or has not checked.

---

## 2 · Why five hours is enough

The reference point is Vāgdhenu (Prathosh A P, IISc), the ṛgvedic śloka-to-chant
system this project is modelled on. Its voice came from a purpose-recorded
single-speaker corpus of about **five hours**, plus a voice-steering retrain, and it
reaches an expert MOS near 4.6. It did not need a hundred hours, because of a
property of the domain:

> **Chant has low musical entropy and high phonetic entropy.**

The melody is given: the svara sequence is in the text and the scale is fixed. What a
model has to learn is the *phonetics* — how this singer produces a retroflex aspirate,
how long a pluta is held, how much breath sits on a visarga. That is a much smaller
problem than general TTS, where the model must invent prosody, and it is why a small
model on five hours can work.

Five hours also happens to be a *feasible* ask: about 30 sessions of 10 minutes, or
two full days of a single reciter with proper breaks.

---

## 3 · The recording protocol

The point of a protocol is that the resulting corpus is trainable without a week of
cleanup afterwards.

### Before

* **One reciter, one microphone, one room, for the whole corpus.** A single-speaker
  corpus is the single biggest quality lever. If you must have several reciters, keep
  them as separate "voices" — do not mix them.
* **A large-diaphragm condenser or a decent dynamic mic, 20–30 cm, off-axis.** A USB
  mic at 48 kHz is fine. Phone recordings are usable but cost you ~15 % of final
  quality.
* **A quiet room with soft furnishings.** Aim for a noise floor below −55 dBFS. Record
  a 10-second silence at the start of each session so you can measure it and denoise
  later if needed.
* **48 kHz, 24-bit, mono, WAV.** Never MP3 for the master.
* **Consent, in writing, before you start.** A one-page form: what is being recorded,
  that it will be used to train a synthesis model, that the model and derived data may
  be released, and that the reciter may withdraw a clip. Store it. See §6.

### While recording

* **Record the gāna, not the saṃhitā**, when you can — gāna is what makes a chant TTS
  interesting. If a gāna is unfamiliar to the reciter, record the saṃhitā too.
* **One gāna per take, with a spoken slate** before it: "*gāna 14, Grāma-geya*". The
  slate is a free, robust index that costs two seconds.
* **Do not stop and restart mid-line.** A mistake means redo the whole breath group;
  spliced lines are a nightmare to align and they teach the model nothing good.
* **Leave the natural breath in.** Breath is part of the style; do not edit it out.
* **Record more than you think you need.** Expect to discard 10–15 % as unusable.

### After

* **Do not normalise per clip.** Apply one gain across the whole corpus so relative
  dynamics survive; normalise once at the end.
* **Keep the raw and the processed audio separately.** Never overwrite a master.
* **Write the sidecar as you go**, not later. The annotation is the part people
  postpone and then lose.

---

## 4 · Annotation: one timestamp per breath group

This is the part where most small corpus efforts die, because they assume phone-level
forced alignment. For Sāma chant you do not need it.

Sāma timing is **mātrā-based**: the text says how long each syllable is held, and the
reciter follows it. So an akṣara-level alignment is recoverable from *one timestamp per
breath group* — the daṇḍa-delimited unit that is sung in a single breath. Everything
inside is divided in proportion to the mātrās the score specifies.

That is a 30-second job for a 10-minute recording, and
`samadhvani/data/build.py:align_clip` does the division.

### The sidecar format

One JSON file per clip, next to the WAV:

```json
{
  "audio":     "grāmageya_014.wav",
  "text":      "हाउ ४ अ॒ग्निमी॑ळे ३ पु॒रोहि॑तं २ य॒ज्ञस्य॑ १ दे॒वमृ॒त्विज॑म्",
  "line_starts": [0.0, 4.2, 9.8],
  "school":    "kauthuma",
  "model":     "gramageya",
  "licence":   "open",
  "source":    "own-recording-2026-03",
  "reciter":   "anonymous-01"
}
```

* `text` — the gāna text **with its svara numerals**. If you paste a text with
  numerals, the parser reads the melody from it and nothing is invented. That is worth
  more than any amount of audio.
* `line_starts` — the start of each breath group, in seconds. If you omit it, the clip
  is treated as one group and `align_clip` will proportion the timing across the whole
  clip if you pass `total_seconds`.
* `licence` — `open` for your own consented recordings.

Then:

```bash
python -m samadhvani.data.build audio clips/ -o manifest.jsonl
python -m samadhvani.data.build contours manifest.jsonl -o models_learned.json
python -m samadhvani.train.trainer --manifest manifest.jsonl --out samadhvani.pt
```

The `contours` step is what replaces the hand-written melodic skeletons in
`samadhvani/models/sama_models.py` with contours measured from the actual singing.
Do not skip it — it is the difference between "a chant in the right style" and "this
reciter's chant".

---

## 5 · Quality tiering

Not every clip is worth training on. Grade automatically, then spot-check:

| tier | criteria | use |
|---|---|---|
| `gold` | clean, in tune, no speaking, text matches | train the acoustic model |
| `silver` | small pitch slip or one repeated word | train with a lower weight |
| `review` | audible deviation or noise | validate by hand before including |
| `reject` | wrong text, cut off, heavy noise | keep for robustness work only |

A useful automatic first pass: decode the pitch inside each scheduled note with
`samadhvani.eval.metrics.svara_accuracy_at_notes`. A clip whose decoded melody matches
its own score is almost always a clean take — the reciter sang what the notation said.
That is exactly the alignment-free grader Vāgbodhinī uses for its practice tools, and
it transfers here for free.

---

## 6 · Consent and licensing, concretely

If you release a model, you are releasing something derived from a person's voice.
Handle it explicitly:

* **Written consent** naming: recording, training, model release, derived-data
  release, withdrawal. One page, signed, stored with the audio.
* **Attribute reciters** where they wish to be named, and never where they do not.
  Offer both options; a reciter may consider their gāna a religious act rather than a
  performance.
* **Do not train on third-party audio**, even for "just a quick experiment", if the
  resulting weights might ship. Keep an `unclear` bucket that is never mixed in.
* **Publish the audit.** `samadhvani/data/sources.py` exists so anyone can see exactly
  what went into a checkpoint. Keep it accurate.
* **Respect the tradition's own view of transmission.** Sāma recitation is lineage-bound:
  the stobhas differ between Kauthuma, Rāṇāyanīya and Jaiminīya, and a reciter will hear
  a merged tradition as an error, not a feature. Record and label by śākhā.

---

## 7 · A realistic first corpus

If you want a target that is both ambitious and achievable:

| item | target | why |
|---|---|---|
| reciters | 1 | single-speaker training is what makes a voice |
| śākhā | Kauthuma (or Rāṇāyanīya) | choose one; do not merge |
| gāna | 200–300 from Grāma-geya | enough phonetic coverage; Grāma-geya is the canonical book |
| audio | 4–6 h | the Vāgdhenu-scale number |
| annotation | breath-group starts + gāna number | a day's work, not a month's |
| held out | 30 clips, never trained on | the only honest evaluation |

200 gāna × ~60 s ≈ 3.3 h. Double-record a subset to reach five.

That corpus plus the code here will produce a system that sings a real gāna the way a
real reciter sings it — which is the only claim worth making.
