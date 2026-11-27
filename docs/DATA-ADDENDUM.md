# The data survey — every source, and what it is actually worth

This is the full survey behind `samadhvani/data/sources.py`. It exists because the
brief said *"see all possible sources"*, and because the honest answer to "can I
train this on Kaggle?" turned out to be much better than the first version of
`docs/DATA.md` claimed — but only under a specific, revocable assumption about
what you will do with the result.

Read this alongside `audit`:

```bash
samadhvani audit                      # educational profile (default)
samadhvani audit --profile open       # what survives publication
samadhvani audit --profile evaluation # listening only
```

All hours below are **measured from archive.org metadata** (file `length` fields),
not estimated. Where a number is an estimate it says so.

---

## 1 · The headline

| | hours |
|---|---|
| sāmagāna audio that exists and is reachable | **68.8 h** |
| of that, trainable under the educational profile | **67.3 h** |
| of that, carrying line-level text/timestamp alignment | **15.8 h** |
| openly licensed (CC0 / CC-BY / public domain) | **0.0 h** |
| publishes the Veda text under an open licence | 1 source (**Vedic Heritage Portal**) |

For scale: Vāgdhenu, the system this project is modelled on, was built on about
**5 h**. The 15.8 h of *aligned* audio alone is 3× that, and it needed no
annotation work — the timings were printed by the transcriber forty years before
anybody thought about TTS.

The catch is in the last two rows, and it is the whole subject of this document.

---

## 2 · The one find that changes the project

**IISH Global's complete Kauthuma Sāmagāna, 64 clips, 43.2 hours, already
timestamped.**

It has been sitting on archive.org since 2013 as `IISHSamaVeda`, uploaded by a
third party. Nobody in the TTS world appears to have noticed, because it is filed
under `audio_sermons` with the genre tag "Blues" and no metadata worth reading.

Why it matters:

* **It is complete.** All four gāna books — Grāma-geya, Āraṇyaka, Ūha, Ūhya.
* **It is pre-aligned.** PR Iyer's handwritten transcriptions
  (`archive.org/details/sAmagAnam`) print, on the left of every line, *the audio
  clip number and a minute:second stamp*. So the mapping from audio to notation is
  in the manuscript. You do not need a forced aligner, an ASR model, or a week of
  hand annotation — the three things that kill small corpus projects.
* **The alignment is complete exactly where the text is complete.** Clips 1–24
  (15.8 h) finish Grāma-gāna and Āraṇyaka-gāna, which is precisely what Iyer
  transcribed. Clips 25–64 (27.5 h) are Ūha/Ūhya, which are now typeset
  (`sd_uha`, `sd_uhya`) but *not* timestamped. That gap is the main unsolved task
  in the project, and it is a bounded one.

Per-clip lengths, so you can plan a Kaggle run without downloading 43 GB:

| clips | content | hours | avg clip |
|---|---|---|---|
| intro | spoken introduction | 0.5 | 30 min |
| 1–24 | Grāma-gāna + Āraṇyaka-gāna | **15.8** | 39 min |
| 25–64 | Ūha-gāna + Ūhya-gāna | 27.5 | 41 min |

One volume of `pcr_samavedam` (below) is ~500 MB and roughly an hour of audio;
that ratio is the number that matters when you are fitting inside Kaggle's
dataset limit, not total disk.

---

## 3 · Everything else, ranked by usefulness per hour

### Audio

| source | hours | aligned | licence | verdict |
|---|---|---|---|---|
| `iish_samagana` | 43.2 | 15.8 | none stated | **the corpus**; train, do not ship |
| `pcr_samavedam` | 20.1 | ~0 (~range) | none stated, commercial origin | use for prosody diversity |
| `hindu_jagaran_samaveda` | 4.0 | no | none stated | lowest priority; same tradition |
| `nonesuch_lp` | 0.2 | no | commercial LP (1968) | the *only* Jaiminīya audio |
| `ignca_cassettes` | 1.5 | no | institutional deposit | lecture, not performance; needs permission |
| `vhp_audio` | ~6 | no | permission | clean reference; write for it |
| `vhp_jaiminiya` | unmeasured | no | permission | **the source worth a letter** |
| `shaivam`, `astrojyoti` | mirrors | — | none stated | same performance; not counted |
| `vedaprasar` | commercial CDs | — | sold | almost certainly the origin of the rips |
| `youtube_samagana` | ~20 | no | per-video, unstated | blocked in *every* profile |

`pcr_samavedam` is organised by sāman number — "sama 1 to 20", "21 to 32",
"33 to 40", "41 to 43", "44 to 46", "47 to 49" — which is *coarse* alignment: you
know the range, not the offset. That is enough to bootstrap a segmenter, not
enough to supervise melody. Use it for phonetic coverage, not for melody.

**On the Jaiminīya school.** It is the great hole in this corpus. A quarter hour
on a 1968 Nonesuch LP, plus a JavaScript-rendered index on the national portal
that could not be enumerated from here. Howard's *Decipherment of the Sāmagavedic
Notation of the Jaiminīyas* (1988) is **free** at `journal.fi/store/issue/view/3442`
— so the notation is decodable even where the audio is not. If you write one
letter, write it to the Vedic Heritage Portal about Jaiminīya gāna.

### Text

| source | what | size | licence |
|---|---|---|---|
| `sd_gramageya` | Grāma-geya, typeset 2025 | 1,198 gāna / 579 pp | PPN ©, free-for-education |
| `sd_aranyaka` | Āraṇyaka | 296 gāna / 177 pp | same |
| `sd_uha` | Ūha | 936 gāna / 515 pp | same |
| `sd_uhya` | Ūhya / Rāhasya | 209 gāna / 148 pp | same |
| `sd_samved_xlsx` | Kauthuma saṃhitā, spreadsheet | 1,875 ṛc | same |
| `pr_iyer_gana` | hand-written gāna + **timestamps** | 15.8 h worth | CC BY-NC-ND 4.0 |
| `pr_iyer_samhita` | saṃhitā, Malayalam | 1,875 ṛc | same |
| `vhp_kauthuma` | Kauthuma saṃhitā | 1,875 ṛc | **open** |

**2,639 gāna** — the complete Kauthuma corpus — typeset by Pallasena
Narayanaswami in January 2025, with the sāma numerals and accents typed by hand
into the Siddhanta font. Before 2025 this project would have needed the print
editions. Now it does not.

Practical order of attack:

1. **`sd_samved_xlsx` first.** It is a spreadsheet, so it goes into the parser
   without a font-mapping detour. It handles accents and syllable counts.
2. **`pr_iyer_gana` second**, because the timestamps are what make the audio
   usable. It is handwriting, so it needs transcription — but only of the lines
   you are training on, and clip 1–24 covers the aligned hours.
3. **The four PDFs last.** They are in a legacy font: text extraction yields
   glyph soup unless you map the Private Use Area codepoints. Budget a day. Do it
   when you want the Ūha/Ūhya text, not before.

---

## 4 · The licence reality

This is the part to get right, because it is the part people get wrong in public.

**Nothing in this corpus is openly licensed.** Not one usable audio collection
carries a CC or public-domain statement. The archive.org items that matter most —
`IISHSamaVeda`, `SamaVedam`, `samaveda` — have *no licence field at all*. They are
third-party uploads of performances that someone else made.

**"Free to download" is not "open".** sanskritdocuments.org, which is the best
text source in the world for this, provides its material under a declaration of
*"fair use for individual educational or research use"*, and the 2025 gāna
typesetting is stamped **"PPN © January 2025"** — the typesetter's copyright on
the typesetting, even though the Veda itself is ancient. The portal (`vhp_*`)
publishes text openly, but its audio is stream-based with no reuse terms.

So the truth about a distributed sāmagāna TTS, stated plainly:

> A downloadable open-licensed checkpoint of this voice **cannot currently be
> built from public data.** You can build the model. You cannot ship it, if it
> was trained on any of the audio above.

### Why that is not a blocker for this project

Because of the two-axis distinction the catalogue is built around:

* *May I redistribute this?* — a **licence** question. Answer: no.
* *May I train a model and run it locally for teaching?* — an **intended-use**
  question. Answer: yes, under the educational profile.

Those are independent. Treating them as one axis is what produces either
needless paralysis ("nothing is open, abandon project") or dishonest marketing
("open data!"). The catalogue keeps both numbers and refuses to merge them.

### What the educational profile does and does not buy

| | train | ship audio | ship weights |
|---|---|---|---|
| Vedic Heritage Portal, own recordings | yes | — | **yes** |
| PR Iyer manuscripts (CC BY-NC-ND) | yes | no | no |
| IISH, PCR, Hindu Jagaran (no licence) | yes | no | no |
| YouTube / Vimeo (provenance unknown) | **no** | no | no |
| Nonesuch LP, Veda Prasar CDs | no | no | no |
| IGNCA cassettes, portal audio (institutional) | no | no | no |

Note the third row. An unlicensed-but-freely-hosted recording is *trainable under
this profile* but not *unblocked*: the difference between them is that the first
is a statement about your conduct and the second would be a claim about someone
else's grant. The catalogue never makes the second claim — `permit_reason` for
those sources says, in as many words, that this is only a promise about your own
conduct, not a grant.

**YouTube stays blocked even here.** Not because teaching is restricted, but
because you cannot name the rights holder, and an upload of someone else's CD is
not a source. `UNCLEAR` is the one licence value no profile can lift. That is
deliberate: it is the failure mode that would actually get you in trouble, and it
is the one people are most tempted by, because the audio is the easiest to find.

---

## 5 · How to stay honest as the project grows

The profile is revocable *by you*, in one step. Three things follow.

1. **Before you publish anything, re-run the audit.**
   ```bash
   samadhvani audit --profile open
   ```
   If it says `0.0 h` — and it will — then a released checkpoint needs a corpus
   that does not include any of the audio in §3. In practice that means recording.

2. **Keep a checkpoint provenance record.** Every checkpoint directory should
   carry the list of source keys that went into it, plus the profile it was
   trained under. `audit` is the natural format for this. A checkpoint whose
   provenance you cannot reconstruct is unshippable by default.

3. **Do not let the demo drift into a release.** A hosted demo that synthesises
   audio from third-party training data is a redistribution event in most people's
   reading, and it is exactly the step the profile assumed you would not take.
   Running it on your own machine for a class is within the promise; putting it on
   a public URL is not.

The point of writing this down is not to be gloomy. It is that the *permitted*
path is well-defined and large: 67 hours of real sāmagāna, 15.8 of it pre-aligned
and notation-complete, against a reference system that worked on 5. That is
enough to build something that genuinely sings the Sāmaveda. It is not enough to
give it away, and a project that says so in its first paragraph is more credible
than one that quietly does not.

---

## 6 · Open questions, honestly listed

* **Hours of the portal's Jaiminīya collection: unknown.** The index is
  JavaScript-rendered and could not be enumerated here. Somebody should just load
  it in a browser and count.
* **Is `shaivam` / `astrojyoti` the same performance as `pcr_samavedam`?** Almost
  certainly, possibly at different bitrates. Diff a few minutes before assuming.
* **Does the Iyer timestamp scheme survive into clips 25–64?** The uploader says
  the Ūha/Ūhya text "is not prepared here", which implies no — but the clips exist
  and the printed editions do, so a mapping may be reconstructible from sāman
  counts per clip.
* **Audio quality of `iish_samagana` is unmeasured.** These are broadcast-style
  recordings. Measure the noise floor per clip before trusting it; expect to drop
  10–15 %.
* **Tonic drift across 43 hours of one reciter.** Do not assume a single D3 for
  the whole corpus. `Chant.tonic_hz` estimates it per clip; use it.
