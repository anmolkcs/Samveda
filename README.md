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

Open [`notebooks/kaggle_samadhvani.ipynb`](notebooks/kaggle_samadhvani.ipynb) — every
cell of it has been executed end to end. Or:

```bash
python -m samadhvani.data.build audit                     # what you may train on
python -m samadhvani.data.build text ganam/*.txt -o train.jsonl
python -m samadhvani.train.trainer --manifest train.jsonl --out samadhvani.pt
samadhvani chant "…" --engine neural --checkpoint samadhvani.pt
```

What trains, and what does not:

| | learns | params | time on a T4 |
|---|---|---|---|
| `DurationPredictor` | how long each phone is held | ~0.4 M | minutes |
| `SpectrogramModel` | F1/F2/F3 + voice/noise per frame | ~2.5 M | well under an hour |
| F0 | *nothing — computed from the svaras* | — | — |

## The data question (read this before training)

Text is the easy half. Audio is the bottleneck, and much of what exists is not licensed
for training. `samadhvani audit` prints the honest table:

```
trainable audio: 6.0 h   redistributable audio: 0.0 h
text: 7514 gāna across 6 sources with svara notation
BLOCKED (do not train a released model): youtube_samagana
needs written permission: vhf_audio
```

* **Text — open and large.** ~7,500 gāna with Vedic accents and sāma numerals across the
  four gāna books and the Kauthuma saṃhitā.
* **Audio — small and encumbered.** The Vedic Heritage Portal recordings are excellent
  and are published for listening, not for training. YouTube uploads are per-video and
  usually unstated.

The realistic path is **five hours of your own consented, single-singer recording** —
which is the scale that comparable systems use. The recording protocol, the
breath-group annotation scheme (one timestamp per breath group is enough, because Sāma
timing really is mātrā-proportional) and the consent requirements are in
[`docs/DATA.md`](docs/DATA.md).

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
  their pitch skeleton. With recordings, `samadhvani.data.build contours` replaces them
  with measured ones.
* **The Jaiminīya and Rāṇāyanīya traditions** are represented by their stobha inventories
  and register, not by measured contours. Merging schools would be a factual error, so
  they stay separate even where the data is thin.
* **Not a ritual authority.** This is a research and heritage-access tool. A recording of
  an uḍgātṛ is the authority; this is software that tries not to misrepresent itself.

## License

MIT for the code. Text and audio carry the licences of their sources — run
`samadhvani audit`, and read `docs/DATA.md` before publishing a checkpoint.
