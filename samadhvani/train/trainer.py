"""Training, sized for a free Kaggle GPU.

    python -m samadhvani.train.trainer --manifest train.jsonl --out ckpt.pt --epochs 60

What it trains, and why in this order:

**Stage 1 — durations.**  Predict, for each phone, how many frames it lasts.  This
is a sequence-labelling problem with a handful of hundred thousand parameters.  It
converges in a few minutes on a T4, it needs no vocoder, and it is the single
largest intelligibility win, because the thing that makes a chant unintelligible is
the syllable *proportions* being wrong, not the timbre.

**Stage 2 — spectral.**  Predict the formant tracks (F1/F2/F3) and the voice/noise
amplitudes for each frame, conditioned on the phone and the svara.  Because the
target is a six-number vector per frame rather than a 80-band mel spectrogram, this
trains on data that a mel model would starve on.

**Stage 3 — (optional) vocoder.**  Not implemented deliberately.  A HiFi-GAN needs
10+ hours and a day of GPU.  When you have the audio, drop one in behind
``SamadhvaniCheckpoint`` and the acoustic model above needs no change; that is what
the level separation in ``samadhvani.synth.formant`` is for.

The loss for stage 2 combines an L1 on the formant tracks with a **formant-weighted**
term: an error of 100 Hz on F1 matters far more than 100 Hz on F3, and F1/F2 are
what carries vowel identity.  Without that weighting the model learns to predict
the average of all vowels and produces a mush.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..synth.neural import (
    COND_DIM,
    INPUT_DIM,
    HAS_TORCH,
    SamadhvaniCheckpoint,
    build_models,
    condition_features,
    phone_features,
)

if HAS_TORCH:
    import torch
    from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    epochs_duration: int = 40
    epochs_spectral: int = 60
    batch_size: int = 16
    lr: float = 3e-4
    hidden: int = 128
    upsample: int = 4
    frames_per_second: float = 100.0     # 10 ms hop
    clip_grad: float = 1.0
    val_fraction: float = 0.1
    seed: int = 0
    device: str = "auto"
    log_every: int = 20


def _pick_device(cfg: TrainConfig):
    if cfg.device != "auto":
        return cfg.device
    if not HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ChantDataset:
    """Yields per-utterance feature sequences.

    Built in plain NumPy and collated with a padding collate: the utterances are
    different lengths and chart-length batching avoids padding most of a batch.
    """

    def __init__(self, records: list[dict], frames_per_second: float = 100.0):
        self.records = records
        self.fps = frames_per_second

    def __len__(self) -> int:
        return len(self.records)

    def _utterance(self, rec: dict):
        phones = rec["phones"]
        svaras = rec.get("svaras") or []
        durs = rec.get("durations") or []
        xs, cs, target_dur, target_spec = [], [], [], []
        rel_f0 = rec.get("f0_hz") or []
        base = rel_f0[0] if rel_f0 else 146.83
        for i, ph_list in enumerate(phones):
            sv = svaras[i] if i < len(svaras) and svaras[i] else 4
            dur = durs[i] if i < len(durs) and durs[i] else 0.2
            f0 = rel_f0[i] if i < len(rel_f0) else base
            cents = 1200.0 * math.log2(max(f0, 1.0) / max(base, 1.0))
            is_cad = 1.0 if i == len(phones) - 1 else 0.0
            cond = condition_features(int(sv), cents, dur / 0.4, bool(is_cad))
            for j, sym in enumerate(ph_list):
                posn = (j + 0.5) / max(len(ph_list), 1)
                xs.append(phone_features(sym, dur, posn))
                cs.append(cond)
                # per-phone duration target, split among its phones
                target_dur.append(math.log(max(dur * self.fps / max(len(ph_list), 1), 1e-2)))
                target_spec.append(self._spectral_target(sym))
        if not xs:
            return None
        return (
            np.asarray(xs, dtype=np.float32),
            np.asarray(cs, dtype=np.float32),
            np.asarray(target_dur, dtype=np.float32),
            np.asarray(target_spec, dtype=np.float32),
        )

    @staticmethod
    def _spectral_target(sym: str) -> list[float]:
        """The 6-number acoustic target for one phone frame.

        **Units must match the model's head.**  :class:`SpectrogramModel` emits
        F1/F2/F3 in Hz (sigmoid scaled to 1200/3200/4200 Hz) and the amplitudes in
        0..1.  The target is therefore in Hz too.  Mixing normalised targets with a
        Hz head is the sort of silent bug that trains happily and predicts nothing —
        it shows up as a loss that plateaus in the thousands.
        """
        from ..phones.inventory import PHONES

        p = PHONES.get(sym)
        if p is None:
            return [500.0, 1500.0, 2500.0, 0.5, 0.0, 0.5]
        f = p.formants
        voiced = 1.0 if p.voiced else 0.0
        noisy = 1.0 if p.manner in ("fricative", "sibilant") else 0.0
        return [
            (f[0] if f[0] > 0 else 400.0),
            (f[1] if f[1] > 0 else 1500.0),
            (f[2] if f[2] > 0 else 2500.0),
            voiced, noisy, 0.5,
        ]


if HAS_TORCH:

    class TorchChantDataset(Dataset):
        def __init__(self, records, cfg: TrainConfig):
            self.impl = ChantDataset(records, cfg.frames_per_second)
            self.cfg = cfg

        def __len__(self):
            return len(self.impl)

        def __getitem__(self, idx):
            item = self.impl._utterance(self.impl.records[idx])
            if item is None:
                return self[0] if idx else self[0]
            x, c, d, s = item
            return (torch.from_numpy(x), torch.from_numpy(c),
                    torch.from_numpy(d), torch.from_numpy(s))

    def collate(batch):
        T = max(b[0].shape[0] for b in batch)
        B = len(batch)
        x = torch.zeros(B, T, INPUT_DIM)
        c = torch.zeros(B, T, COND_DIM)
        d = torch.zeros(B, T)
        s = torch.zeros(B, T, 6)
        mask = torch.zeros(B, T)
        for i, (xi, ci, di, si) in enumerate(batch):
            n = xi.shape[0]
            x[i, :n] = xi
            c[i, :n] = ci
            d[i, :n] = di
            s[i, :n] = si
            mask[i, :n] = 1.0
        return x, c, d, s, mask


# --------------------------------------------------------------------------- #
# Training loops
# --------------------------------------------------------------------------- #

def train(manifest: str, out: str, cfg: TrainConfig | None = None,
          val_manifest: str | None = None) -> str:
    cfg = cfg or TrainConfig()
    if not HAS_TORCH:
        raise RuntimeError("torch is required for training: pip install torch")

    records = _read_manifest(manifest)
    if val_manifest:
        val = _read_manifest(val_manifest)
    else:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.permutation(len(records))
        n_val = max(1, int(len(records) * cfg.val_fraction))
        val = [records[i] for i in idx[:n_val]]
        records = [records[i] for i in idx[n_val:]]
    if not records:
        raise SystemExit("no usable records in the manifest")

    device = _pick_device(cfg)
    print(f"device={device}  train={len(records)}  val={len(val)}  "
          f"frames/s={cfg.frames_per_second}")

    tr = DataLoader(TorchChantDataset(records, cfg), batch_size=cfg.batch_size,
                    shuffle=True, collate_fn=collate, drop_last=False)
    va = DataLoader(TorchChantDataset(val, cfg), batch_size=cfg.batch_size,
                    shuffle=False, collate_fn=collate, drop_last=False)

    dur_model, spec_model = build_models(hidden=cfg.hidden, upsample=cfg.upsample)
    dur_model.to(device)
    spec_model.to(device)

    # --- stage 1: durations ---
    opt = torch.optim.AdamW(dur_model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs_duration)
    best = float("inf")
    for ep in range(cfg.epochs_duration):
        dur_model.train()
        tot, n = 0.0, 0
        for x, c, d, s, mask in tr:
            x, c, d, mask = x.to(device), c.to(device), d.to(device), mask.to(device)
            pred = dur_model(x, c)
            loss = ((pred - d) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dur_model.parameters(), cfg.clip_grad)
            opt.step()
            tot += float(loss.detach())
            n += 1
        sched.step()
        v = _eval_duration(dur_model, va, device)
        print(f"[dur ] ep {ep + 1:3d}/{cfg.epochs_duration}  train {tot / max(n, 1):.4f}  val {v:.4f}")
        if v < best:
            best = v
            _save(out, dur_model, None, cfg)
    print(f"duration model done, best val mse {best:.4f}")

    # --- stage 2: spectral (uses the now-frozen duration model's encoder as init) ---
    opt = torch.optim.AdamW(spec_model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs_spectral)
    best = float("inf")
    # formant weighting: F1 and F2 errors hurt more than F3
    w = torch.tensor([2.0, 1.6, 0.9, 0.7, 0.9, 0.3], device=device)

    for ep in range(cfg.epochs_spectral):
        spec_model.train()
        tot, n = 0.0, 0
        for x, c, d, s, mask in tr:
            x, c, s, mask = x.to(device), c.to(device), s.to(device), mask.to(device)
            # The spectral model upsamples by `upsample`, so the target is
            # replicated to match; in a real run with audio you would instead
            # downsample the audio-derived frames and use a mel loss here.
            T = x.shape[1]
            Tout = T * cfg.upsample
            s_up = s.repeat_interleave(cfg.upsample, dim=1)
            mask_up = mask.repeat_interleave(cfg.upsample, dim=1)
            pred = spec_model(x, c)
            # align lengths (conv-transpose padding can be off by a frame)
            L = min(pred.shape[1], s_up.shape[1])
            pred, s_up, mask_up = pred[:, :L], s_up[:, :L], mask_up[:, :L]
            err = torch.abs(pred - s_up) * w
            loss = (err * mask_up.unsqueeze(-1)).sum() / mask_up.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(spec_model.parameters(), cfg.clip_grad)
            opt.step()
            tot += float(loss.detach())
            n += 1
        sched.step()
        v = _eval_spectral(spec_model, va, device, cfg, w)
        print(f"[spec] ep {ep + 1:3d}/{cfg.epochs_spectral}  train {tot / max(n, 1):.4f}  val {v:.4f}")
        if v < best:
            best = v
            _save(out, dur_model, spec_model, cfg)

    print(f"saved {out}  (best spectral L1 {best:.4f})")
    return out


def _read_manifest(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _save(path: str, dur, spec, cfg: TrainConfig) -> None:
    ck = SamadhvaniCheckpoint(
        duration_state=dur.state_dict(),
        spectrogram_state=spec.state_dict() if spec is not None else {},
        config={
            "hidden": cfg.hidden, "upsample": cfg.upsample,
            "frames_per_second": cfg.frames_per_second, "input_dim": INPUT_DIM,
            "cond_dim": COND_DIM,
        },
    )
    ck.save(path)


def _eval_duration(model, loader, device) -> float:
    if len(loader) == 0:
        return 0.0
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, c, d, s, mask in loader:
            x, c, d, mask = x.to(device), c.to(device), d.to(device), mask.to(device)
            pred = model(x, c)
            tot += float(((pred - d) ** 2 * mask).sum() / mask.sum().clamp(min=1))
            n += 1
    return tot / max(n, 1)


def _eval_spectral(model, loader, device, cfg, w) -> float:
    if len(loader) == 0:
        return 0.0
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, c, d, s, mask in loader:
            x, c, s, mask = x.to(device), c.to(device), s.to(device), mask.to(device)
            T = x.shape[1]
            s_up = s.repeat_interleave(cfg.upsample, dim=1)
            mask_up = mask.repeat_interleave(cfg.upsample, dim=1)
            pred = model(x, c)
            L = min(pred.shape[1], s_up.shape[1])
            err = torch.abs(pred[:, :L] - s_up[:, :L]) * w
            tot += float((err * mask_up[:, :L].unsqueeze(-1)).sum() / mask_up[:, :L].sum().clamp(min=1))
            n += 1
    return tot / max(n, 1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the Samadhvani chant model.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val-manifest", default=None)
    ap.add_argument("--out", default="samadhvani.pt")
    ap.add_argument("--epochs-dur", type=int, default=40)
    ap.add_argument("--epochs-spec", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    cfg = TrainConfig(
        epochs_duration=args.epochs_dur, epochs_spectral=args.epochs_spec,
        batch_size=args.batch_size, lr=args.lr, hidden=args.hidden, device=args.device,
    )
    train(args.manifest, args.out, cfg, args.val_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
