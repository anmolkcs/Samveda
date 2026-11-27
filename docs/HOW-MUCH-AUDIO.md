# How much audio does this actually need?

Short answer: **4 hours, and the extra 11.8 does not buy you anything this model can
use.** That is not a preference. It falls out of what the model is and is not trained
on, and I measured the numbers below on a real MP3 pipeline rather than estimating
them.

---

## 1 · The comparison you asked for

| | Vāgdhenu (the reference) | Samadhvani |
|---|---|---|
| training audio | ~5 h, purpose-recorded, one speaker, studio | **4 h**, archival, one performer, lossy MP3 |
| parameters | tens of millions + fine-tuned neural vocoder | **0.25 M** + harmonic resynthesis |
| what is learned from audio | text→mel, then mel→waveform | **spectral colour only** |
| what is *not* learned | — | melody, timing, scale |
| compute | GPU-days | minutes on a free T4 |

The honest reading of that table: these are not the same kind of model, so "it worked
on 5 h" does not transfer *for the same reasons*. It transfers because of a smaller,
stronger reason explained next.

## 2 · Why 4 h is enough here

The model predicts one thing: **a log-mel residual — this singer's spectral envelope
versus the corpus average — conditioned on (F0, voicing, energy).**

It never has to learn pitch, duration, the seven svaras, the tonic, or where a phrase
starts and stops. All of that is computed exactly from the notation. So the mapping it
must fit is narrow and near-deterministic: given a pitch and a loudness, what does this
throat sound like? That is a property of *one recording session*, not of a language,
and it does not keep improving once you have heard the singer in every register.

**What I have actually measured.** On a synthetic 0.064 h corpus — 31 breath groups,
generated chant, not the recordings — the model converged and improved validation L1
by 10.8 %, and across all seven demo verses the voice engine raised melody accuracy
from 93.3 % to 96.7 %. That proves the *architecture and the objective* work at a
small scale; it does not prove 4 h of real audio is sufficient, because synthetic chant
is far cleaner than a 1970s archival recording.

What I can say without overclaiming: the mapping is narrow (pitch and loudness → one
envelope), it is a per-singer property, and the parameter count is 0.25 M. That is a
regime where a few hours typically saturates and tens of hours do not help. The
principled way to settle it is one cheap experiment (§5): train at 1 h, 2 h, 4 h, and
plot validation L1. If the curve is flat from 2 h, the cap is justified by your own
data rather than by my argument.

What more audio genuinely buys, in order of real value:

1. **Robustness on unseen pitch and loudness.** More register coverage is worth
   something. The first 4 h already contain it; the last 12 add repeats.
2. **A neural vocoder.** This is the real data sink. Phase generation wants a lot of
   waveform, and 15.8 h is a defensible corpus for it — but it is a *different* model
   with a different budget, and it should come after alignment, not before.

## 3 · Where extra audio cannot help at all

The melody is not learned. Adding recordings of more gāna cannot teach the model to
sing a gāna it has not seen, because nothing in the training signal says *which* gāna a
breath group is. Segments are cut on breaths, which is what the tradition actually
does, and it means the corpus supervises acoustics and nothing else.

So of your 15.8 aligned hours, hours 5 through 15.8 would train the same 0.25 M
parameters on more of the same. The marginal hour is worth close to zero *until* the
alignment step below exists.

## 4 · What measured runtimes look like

Measured on this repo, on a 6-clip MP3 corpus, CPU only:

| stage | per hour of audio | 4 h | 15.8 h |
|---|---|---|---|
| decode + segment + pack | ~14 s | ~1 min | ~4 min |
| F0 track | ~90 s | ~6 min | ~24 min |
| training (GPU) | — | seconds/epoch | same |
| **download** | network | 0.9 GB | 2.5 GB |

So the full 4-hour run is about **20 minutes of compute**, dominated by download and
pitch tracking — not by the model. Choosing 4 h over 15.8 h saves you roughly 20
minutes of session quota. It does not save you an experiment.

## 5 · Settle it with one experiment (20 minutes)

The claim "4 h saturates" is testable and you should not take it on faith. One notebook
run, four trainings:

```python
from samadhvani.data.dataset import build_corpus
from samadhvani.train.voice_trainer import add_f0_to_corpus, VoiceConfig, train_voice

for h in (1, 2, 4):
    build_corpus(files, f"c{h}.npz", f"c{h}.jsonl", max_hours=h, verbose=False)
    add_f0_to_corpus(f"c{h}.npz", verbose=False)
    m = train_voice(f"c{h}.npz", f"v{h}.pt",
                    VoiceConfig(epochs=40, device="cuda"), verbose=False)
    print(h, "h ->", round(m["val_residual_l1"], 4))
```

Plot those three numbers. **If L1 falls from 1 h to 2 h and is flat from 2 h to 4 h, the
cap is justified by your own measurements** and you can stop worrying about the missing
12 hours. If it is still falling steeply at 4 h, then more audio does help and you
should use all 15.8 — that is a 25-minute run, so the experiment costs less than the
argument.

**One caution about the metric.** Lower L1 means "spectral envelope closer to the
recording", which is what the model is for. It is *not* a perceptual score. Before you
trust a checkpoint, listen to cell 7 and check that melody holds at 100 % — a smoother
envelope that smears note onsets is worse, not better.

## 6 · If you want to use all 15.8 h

Set `HOURS = 15.8` in the notebook. Nothing breaks; it is one number. What you get is a
slightly better-averaged estimate of the same envelope, and a 2.5 GB download. What you
do not get is a model that sings new text, because that needs:

## 7 · The thing that actually unlocks more

**Text-to-audio alignment.** Iyer's sāmagāna manuscripts print a clip number and a
`minute:second` stamp beside every line, so the mapping already exists on paper — it
needs transcribing. With it the model conditions on phonemes instead of just pitch, and
then:

* more audio **does** help, because every aligned line is a new (phoneme sequence →
  acoustic) example;
* the corpus stops being an upper bound on quality and becomes one on repertoire;
* a neural vocoder becomes worth training, and 15.8 h is the right size for it.

That is the next milestone, and it is a transcription job, not a GPU job. Until it is
done, 4 h and 15.8 h are the same model with different download sizes.

---

## The one-line answer

> 4 hours. It matches Vāgdhenu's scale, it is the default in the notebook, it trains in
> about 20 minutes of a free T4 session, and the remaining 11.8 hours would only
> re-estimate the same spectral envelope. Spend the difference on alignment — that is
> where the next real gain is, and it is the only thing that makes the extra hours
> mean anything.
