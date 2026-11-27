# Getting the data — the only part that is actually hard

Text for Sāma chant is abundant, free, and since January 2025 fully machine-readable.
Audio with a transcript is rare — but less rare than it was, and a full survey with
measured hours is in **`docs/DATA-ADDENDUM.md`**. Read that for the source list; this
document is about what you do with it.

Run `samadhvani audit` for the live version of the tables. Add `--profile open` to see
what survives publishing, which is a much shorter list.

!!! note "The two-axis rule"

    *May I redistribute this?* is a **licence** question. *May I train a model and run
    it for teaching?* is an **intended-use** question. They are independent, and this
    project keeps both answers rather than merging them into one cheerful "usable".
    Nothing in the public corpus below is openly licensed. Under the default
    `educational` profile (not distributed, not commercial) about 67 hours are
    trainable. None of it is shippable, and `--profile open` returns 0.0 h.

---

## 1 · What exists

### Text — free, large, ready

| source | what | size | terms |
|---|---|---|---|
| `sd_gramageya` | Grāma-geya, typeset 2025 | 1,198 gāna | PPN © typesetting; fair-use download |
| `sd_aranyaka` | Āraṇyaka | 296 gāna | same |
| `sd_uha` | Ūha | 936 gāna | same |
| `sd_uhya` | Ūhya / Rāhasya | 209 gāna | same |
| `sd_samved_xlsx` | Kauthuma saṃhitā, spreadsheet | 1,875 ṛc | same — **start here** |
| `pr_iyer_gana` | hand-written gāna **with timestamps** | 15.8 h of audio | CC BY-NC-ND 4.0 |
| `vhp_kauthuma` | Kauthuma saṃhitā | 1,875 ṛc | **open** |

That is the **complete** Kauthuma corpus — **2,639 gāna** — and it is enough to
supervise durations and melody. Note the count carefully: the old version of this
document said "7,500 gāna", which was wrong, because it added the same 1,875 ṛc four
times (four providers carry them) and mixed ṛc into a gāna count. The units are
different things.

### Audio — 67 h usable, 15.8 h aligned, 0 h shippable

| source | hours | aligned | licence |
|---|---|---|---|
| `iish_samagana` | 43.2 | **15.8** | none stated |
| `pcr_samavedam` | 20.1 | range only | none stated |
| `vhp_audio` | ~6 | no | permission required |
| `hindu_jagaran_samaveda` | 4.0 | no | none stated |
| `nonesuch_lp` (Jaiminīya) | 0.2 | no | commercial |
| `ignca_cassettes` | 1.5 | no | institutional |
| Your own recordings | 0 h until you record | — | **yours** |

The uncomfortable fact stands, in a sharper form than before: **there is no openly
licensed Sāma recitation corpus at all.** Any project that claims one either has a
private arrangement or has not checked.

But the *useful* fact is much better than the first version of this document
suggested. `iish_samagana` is 43 hours of complete Kauthuma sāmagāna whose
transcription already carries clip numbers and minute:second stamps — so ~15.8
hours arrive aligned, which is three times what Vāgdhenu needed, with no annotation
work. That is the corpus. See `DATA-ADDENDUM.md` §2.

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

And there is now a shortcut worth knowing about: 15.8 hours of Grāma-gāna and
Āraṇyaka-gāna are already sang, already transcribed, and already timestamped
(`iish_samagana` + `pr_iyer_gana`). That is three times Vāgdhenu's budget, and it
means the model can be *working* before anybody picks up a microphone. Record your
own five hours anyway — for the release path, described in §7.

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
  be released, and that the reciter may withdraw a clip. Store it. See §6, and §7 for what the profile permits.

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

## 7 · Intended use, and the promise you are making

`samadhvani audit` reports two things per source, and they are not the same:

* `train` / `ship` — the **licence** answer. Almost everything is `-`.
* `may_train` / `may_ship_weights` — the answer under an **intended-use profile**.

The default profile is `educational`: *not distributed, not commercial*. Under it,
about 67 hours of sāmagāna become trainable, including recordings that carry no
licence at all. That is a defensible position for a teaching project, and it is the
position this repository documents.

Be clear about what it is not:

* **It is not a licence.** It is a promise about your own future conduct, and it is
  revocable by you in one step.
* **It does not extend to outsiders.** Someone who forks this repo and publishes the
  result has not made your promise. That is why the promise must travel with the
  checkpoint — record the source keys and the profile in the checkpoint directory.
* **It does not cover provenance-unknown material.** `UNCLEAR` sources stay blocked in
  every profile, including this one (§5 of the addendum explains why).
* **It ends at publication.** Publish a checkpoint, or host the demo on a public URL,
  and every source marked `FREE_EDUCATIONAL`, `CC_BY_NC*` or `UNLICENSED` flips to
  blocked. Re-run `samadhvani audit --profile open` before you do it. It will print
  `0.0 h`.

The permitted path is well-defined and large. The *releasable* path is one thing
only: your own consented recordings. Plan both, separately, on purpose.

---

## 8 · Writing for permission

Two holders are worth a letter, because they gate the two things that cannot be
bought elsewhere:

| ask | what it unlocks | why it is plausible |
|---|---|---|
| Vedic Heritage Portal — Jaiminīya gāna (Kerala style) | the only substantial Jaiminīya audio | educational use, non-distributed, no money |
| Vedic Heritage Portal — Kauthuma recitation | ~6 h of clean reference audio | same, plus a named held-out set |

Write to the institution named on the portal, naming the reciter, and say exactly
what you want:

1. permission to **train** a synthesis model on the recordings;
2. acknowledge that **no audio and no model** will be redistributed while permission
   is educational-only;
3. offering to **credit** the reciter and the institution by name, or to omit them if
   they prefer (a reciter may consider their gāna a religious act, not a performance);
4. offering a **withdrawal** mechanism — a clip removed on request within 30 days.

Set `permission_granted=True` on the source in `sources.py` when a written grant
arrives, and put the letter in the repository. `permit()` then returns `may_train`
and `may_ship_weights` for that source, and the audit totals move. That is the only
mechanism by which those numbers are allowed to change.

IGNCA holds the cassettes (`ignca_cassettes`); Nonesuch and Veda Prasar hold the
commercially released material. None of those are worth the letter yet — they are
either too small or too clearly commercial.

---

## 9 · A realistic first corpus

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
