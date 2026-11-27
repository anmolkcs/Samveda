# समध्वनि · Samadhvani

**A Sāmaveda sāmagāna text-to-speech system — svara-aware, śākhā-aware, and trainable
on a free Kaggle GPU.**

Paste a Vedic verse. Get a chant. Any Indic script, IAST, or Devanāgarī with Vedic
accents. Seven sāma svaras, the six sāma-vikāras, three recitation schools, five
gāna books — and a small model you can train yourself on a T4 in under an hour.

```bash
pip install -e '.[app]'
samadhvani serve            # → http://localhost:8000
```

```bash
samadhvani chant "हाउ अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म्" -o chant.wav
```

---

## The problem this solves

The reference point is [Vāgdhenu](https://prathosh.in/vagdhenu/), which maps a metrical
Sanskrit verse to its *pārāyaṇa* recitation. It is excellent, and it solves the ṛgvedic
case. The Sāmaveda is a harder and different problem, and a general Sanskrit TTS cannot
address it:

| | ṛgvedic śloka recitation | **sāmagāna** |
|---|---|---|
| the text | the ṛc as written | **edited** — vikāra, viśleṣa, vikarṣaṇa, abhyāsa, virāma, stobha |
| the melody | three accents, or a vṛtta tune | **seven sāma svaras**, fixed per sāman in the gāna books |
| the voice | one reciter, one register | **five gāna books**, three schools, registers a fourth apart |
| "the right tune" | determined by the metre | **determined by the śākhā's gāna text** |

So a faithful Sāma system has to: keep Sanskrit's phonetic contrasts intact, apply the
vikāras correctly, honour a seven-note scale rather than speech prosody, and — crucially
— **know when it does not know the answer**.

## Three modes, and why the distinction matters

A sāman is not improvised. The gāna books fix the melody, the śākhā fixes the stobhas,
and no system can know the right tune for an arbitrary Sanskrit verse. So Samadhvani
never pretends to:

| mode | when | what it does | honest? |
|---|---|---|---|
| **gāna** | the text has sāma numerals | reads the notes out of the text | nothing invented |
| **saṃhitā** | the text has Vedic accents | honours the three-tone accent map | recitation, not gāna |
| **style** | neither | applies the documented vikāras on a gāna skeleton | *labelled* as styled |

Every render returns the mode it used, the vikāras it applied, and a plain-language
caveat. That is the design centre of the project: a tool that claimed to know the right
tune for arbitrary Sanskrit would be lying, and one that says which mode it is in lets
you rely on it where it is reliable.

See [`docs/pipeline.txt`](docs/pipeline.txt) for the full data flow.

## What it hears

```
Devanāgarī / IAST / Kannada / Telugu / Bengali / …
        │  script detection, transliteration, no schwa deletion
        ▼
   akṣara segmentation ─────────────────────────── accents, pluta, stobhas
        │
        ▼
   sāma-vikāra engine ──────────────────────────── vikāra, viśleṣa, vikarṣaṇa,
        │                                          abhyāsa, virāma, stobha, āgama
        ▼
   svara plan ──────────────────────────────────── seven sāma svaras → semitones
        │                                          (Nāradīya Śikṣā venu mapping)
        ▼
   phones ──────────────────────────────────────── aspiration, retroflexion,
        │                                          three sibilants, vocalic ṛ,
        │                                          ḷ, jihvāmūlīya, upadhmānīya
        ▼
   timing → F0 (exact) + durations/formants (learned) → audio
```

The acoustic side splits deliberately:

* **F0 is computed, not predicted.** The svara sequence is given by the tradition, so
  the melody is exact. A network would only add drift.
* **A network learns the phonetics** — per-phone durations and formant tracks
  (F1/F2/F3). That is a small target, which is why it trains on 3–5 hours of audio and
  a free T4.

Two engines render it:

| engine | needs | what it is |
|---|---|---|
| `formant` | NumPy + SciPy | Klatt-style source–filter cascade: jittered Rosenberg glottal source, per-phone spectral shaping, formant transitions, nasal antiformant, hall reverb |
| `neural` | a trained checkpoint | the same cascade, with durations and formant tracks from the trained models |

The formant engine exists so the demo runs anywhere with no checkpoint and no GPU — and
so there is an honest *baseline*. A trained model that cannot beat it is not worth the
GPU hours.

## Measured, not asserted

```
$ samadhvani eval "हाउ ४ अ॒ग्निमी॑ळे ३ पु॒रोहि॑तं २ य॒ज्ञस्य॑ १ दे॒वमृ॒त्विज॑म्"

=== Samadhvani evaluation ===
melody
   intended_notes                 19
   notes_checked                  19
   svara_accuracy                 0.947
   median_abs_cents_error         2.4
   decoded_notes_free_running     19
audio
   periodicity                    0.817
   hnr_db                         6.3
   spectral_flatness              0.046
   mos_proxy                      3.35
```

The melody numbers are decoded **from the waveform**, not from the plan: pitch is
tracked inside each scheduled note and snapped to the sāma scale. That matters, because
the first version of this evaluator reported 79 % for perfectly-sung chant — a naive
autocorrelation tracker reads a 147 Hz note as 294 Hz. The tracker is now octave-corrected
and low-passes to 1.2 kHz before correlating (`samadhvani/eval/metrics.py` explains why
both fixes were needed).

`mos_proxy` is a proxy, not a MOS, and the report says so. A real score needs a blind
panel of 5–8 Sāma reciters.

## Train it on a free Kaggle GPU

Open [`notebooks/kaggle_samadhvani.ipynb`](notebooks/kaggle_samadhvani.ipynb), turn on
a T4, **Run All**. It downloads the audio itself, so there is nothing to upload and
nothing to click. Roughly 20 minutes of compute for a 4-hour corpus.

The same thing from a shell:

```bash
python -m samadhvani.data.build fetch --clips aligned     # 24 clips, 0.9 GB, resumable
python -m samadhvani.data.build corpus --hours 4          # -> corpus.npz
python -m samadhvani.data.build f0                        # pitch track, once
python -m samadhvani.train.voice_trainer --npz corpus.npz --out voice.pt
samadhvani chant "…" --engine voice --checkpoint voice.pt
```

What the trained model actually learns, and what it does not:

| | learned from audio? | where it comes from | params |
|---|---|---|---|
| spectral envelope (timbre) | **yes** | 4 h of IISH sāmagāna | 0.25 M |
| F0, duration, svaras, tonic | no | computed exactly from the notation | — |

That split is the design, not a shortcut. Because pitch and timing are solved rather
than learned, a quarter-million parameters are enough for one voice, training finishes
in minutes, and the melody cannot degrade — the render is verified note by note.

**How much audio?** 4 h, which is the Vāgdhenu scale. The reasoning, the measured
runtimes, and a 20-minute experiment that settles the question on your own data are in
[`docs/HOW-MUCH-AUDIO.md`](docs/HOW-MUCH-AUDIO.md).

## The data question (read this before training)

Text is the easy half. Audio is the bottleneck. `samadhvani audit` prints two tables,
and the second one is the interesting one:

```
licence facts: 0.0 h of sāmagāna audio is openly licensed (0.0 h trainable-and-shippable)
text: 2,639 gāna + 1,875 ṛc, 8 sources with svara notation (the saṃhitā exists in 4 copies; counted once)

under the 'educational' profile — Educational, not distributed, not commercial
  trainable audio: 67.3 h (15.8 h with line-level timings)
  BLOCKED on provenance: youtube_samagana
  needs written permission: ignca_cassettes, vhp_audio, vhp_jaiminiya
```

* **Text — complete, and free since 2025.** All **2,639** Kauthuma gāna were typeset
  with sāma numerals and accents in January 2025. (The previous version of this file
  said "~7,500 gāna" — that was wrong: it added the same 1,875 ṛc four times, once per
  provider, and mixed ṛc into a gāna count.)
* **Audio — 67 hours, none of it openly licensed.** The find is `iish_samagana`: 43.2 h
  of complete Kauthuma sāmagāna on archive.org since 2013, with PR Iyer's
  transcriptions carrying **clip numbers and minute:second stamps per line**, so
  **15.8 h arrive pre-aligned** — three times Vāgdhenu's budget, with no annotation work.

**One axis, not two.** *May I redistribute this?* is a licence question; *may I train on
it and run it for teaching?* is an intended-use question. They are independent, and this
repo keeps both answers:

* Under the default `educational` profile (not distributed, not commercial), ~67 h are
  trainable.
* **None of it is shippable.** `samadhvani audit --profile open` returns `0.0 h`. A
  downloadable open-licensed checkpoint of this voice cannot currently be built from
  public data.

That profile is a **promise about your conduct, not a licence you were granted**, and
publishing a checkpoint or hosting the demo revives every restriction at once. The
release path is still what it always was: **five hours of your own consented,
single-singer recording**. Protocol, breath-group annotation (one timestamp per breath
group is enough — Sāma timing really is mātrā-proportional) and consent terms are in
[`docs/DATA.md`](docs/DATA.md); the full source survey with measured hours is in
[`docs/DATA-ADDENDUM.md`](docs/DATA-ADDENDUM.md).

## Sanskrit details this gets right

These are the things a Hindi-trained frontend silently breaks:

* **No schwa deletion.** `rāma` stays राम, not राम्. Hindi word processors delete the
  final `a`; the phone mapper has an explicit policy and it defaults to Sanskrit.
* **The full four-way stop series** — k kʰ g gʱ — at five places of articulation, kept
  distinct. Many Indic TTS models neutralise the aspiration contrast intervocalically.
* **Three sibilants** s / ś / ṣ in three distinct noise bands, not merged.
* **Retroflexes**, including the aspirates ṭʰ ḍʱ, with F3 lowering and formant
  transitions.
* **Vocalic ṛ and ḷ**, and the retroflex lateral ḷ (ळ) — the bane of most Indic TTS.
* **Jihvāmūlīya and upadhmānīya** (Pāṇini 8.3.37–41): the visarga is not one phone but
  a velar or bilabial fricative before k/kh and p/ph.
* **Homorganic anusvāra** (8.4.58–59): `saṃskṛta` → `s a n s …`, `aṃśa` → `a ny sh a`.
* **Vedic accents as combining marks**, not punctuation, so `अ॒ग्निमी॑ळे` stays one word
  and the accent maps to a svara.
* **Pluta numerals** (`२٣४`) are duration, not verse numbers — a normaliser that strips
  digits destroys the metre.

## Layout

```
samadhvani/
  text/        script, syllabification, sāma-vikāras, the seven svaras
  phones/      articulatory inventory, Devanāgarī → phone mapping
  synth/       timing, prosody, the formant engine, the neural engine
  models/      sāma-gāna melodic skeletons (and measured contours)
  train/       duration + spectral training loops (Kaggle-sized)
  data/        corpus builder, breath-group alignment, source audit
  eval/        svara accuracy, aspiration/retroflex confusion, HNR
  app/         FastAPI server + the demo page
notebooks/     kaggle_samadhvani.ipynb (executed end to end)
docs/          DATA.md (recording protocol), pipeline.txt
tests/         145 tests
```

```bash
pytest -q          # 145 passed
```

## Honest limitations

* **`style` mode is stylistic.** It applies the documented vikāras and a gāna skeleton.
  It is not the sāman a śākhā would sing for that ṛc. This is stated on every render.
* **No checkpoint ships.** The neural engine is a training harness with a working,
  baseline-beating default — not a pre-trained voice. The formant engine is what you
  hear out of the box, and it sounds like a chant, not a singer.
* **The melodic skeletons are prior knowledge**, reduced from published gāna contours to
  their pitch skeleton. `samadhvani.data.build contours` can replace them with measured
  ones — but it averages the **notated svara sequence**, which is already in the text.
  It is not a pitch tracker, and no audio is read.
* **The model learns timbre, not melody.** Segmentation is by breath, so a training
  segment is "this singer, singing something, at this pitch". Nothing tells the model
  *which* gāna it is hearing, so it cannot learn melody from audio. Melody comes from
  the notation, which is exact but only exists for notated text. Closing this needs
  the Iyer timestamps transcribed — a language job, not a GPU job; see
  `docs/DATA-ADDENDUM.md` §7.
* **Phase is resynthesised.** The vocoder is a harmonic synthesiser driven by the
  measured F0, so sustained notes have a slight synthetic edge compared with a neural
  vocoder. That is the next model to train, and 15.8 h is the right corpus for it.
* **The reference-system comparison is not like for like.** Vāgdhenu is an end-to-end
  flow-matching TTS with tens of millions of parameters and a fine-tuned neural vocoder;
  this is a rule engine plus a 0.25 M envelope model. The audio budget is comparable,
  the models are not, and `docs/HOW-MUCH-AUDIO.md` says so rather than borrowing
  credibility from the comparison.
* **The Jaiminīya and Rāṇāyanīya traditions** are represented by their stobha inventories
  and register, not by measured contours. Merging schools would be a factual error, so
  they stay separate even where the data is thin. Concretely: all 15.8 aligned hours are
  Kauthuma, the Jaiminīya school exists in this survey as **14 minutes on a 1968 LP**,
  and Rāṇāyanīya exists only as provenance-unknown video. Neither can be trained.
* **No pretrained voice, and no releasable one.** The 67 trainable hours are all
  non-distributable. The corpus is real, the code is real, and a model trained on it
  may be run for teaching — but it may not be published.
* **Not a ritual authority.** This is a research and heritage-access tool. A recording of
  an uḍgātṛ is the authority; this is software that tries not to misrepresent itself.

## License

MIT for the code. Text and audio carry the terms of their sources — **none of the
public sāmagāna audio is openly licensed**, and the 2025 gāna typesetting is
`PPN ©` while the Iyer manuscripts are CC BY-NC-ND. Run `samadhvani audit --profile
open` before publishing a checkpoint; read [`docs/DATA-ADDENDUM.md`](docs/DATA-ADDENDUM.md)
§4 before describing this project's data as open.
