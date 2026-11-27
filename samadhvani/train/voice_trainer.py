"""Train the voice-colour model on a packed corpus.  Fits in one free-GPU session.

What this trains, and why it is this and not a full TTS stack
------------------------------------------------------------
A text-conditioned acoustic model needs phone-level alignment: this frame of audio
belongs to this akṣara.  We do not have that yet — it needs Iyer's Malayalam
timestamps transcribed (``docs/DATA-ADDENDUM.md`` §7).  Training on synthetic
alignment would just teach the network to imitate the rule engine, which is
pointless.

What the recordings *can* teach without any alignment is the voice itself: the
singer's spectral envelope as a function of pitch.  That is a real, useful function —
it is most of what makes one uḍgātṛ sound unlike another — and it is learnable
because F0 is recoverable from the audio by autocorrelation and needs no transcript.

So this model is:

    (F0, energy, coarse time context)  ->  log-mel residual vs. the corpus mean

and at synthesis time the residual is added to the rule engine's mel before
resynthesis.  The rule engine supplies *what phoneme is being sung*; this supplies
*what this singer's throat does to it*.

Checkpoint is ``version: 2`` with ``voice_state``.  A v1 six-number checkpoint is
rejected loudly rather than loaded into the wrong shape.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from ..audio.mel import HOP, N_MELS, SR

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except Exception:                                    # pragma: no cover
    torch = None
    Dataset = object
    HAS_TORCH = False

#: log2 of F0, centred so the values are O(1) for the network.
_F0_CENTRE = math.log2(146.83)


@dataclass
class VoiceConfig:
    epochs: int = 60
    batch_size: int = 16
    lr: float = 1e-3
    hidden: int = 128
    n_layers: int = 3
    dropout: float = 0.05
    val_fraction: float = 0.1
    seed: int = 7
    device: str = "auto"
    crop_frames: int = 256
    patience: int = 8
    num_workers: int = 2


def _pick_device(pref: str) -> str:
    if pref != "auto":
        return pref
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------- #
# Features that need no transcript
# --------------------------------------------------------------------------- #

def f0_from_audio(x: np.ndarray, sr: int = SR, hop: int = HOP,
                  lo: float = 70.0, hi: float = 400.0) -> np.ndarray:
    """Per-mel-frame F0 (0 where unvoiced), using the project's own estimator.

    Reusing :mod:`samadhvani.eval.metrics` keeps one pitch algorithm in the codebase,
    so what the trainer sees and what the evaluator measures cannot drift apart.
    """
    from ..eval.metrics import _autocorr, _frames, _period_lag, _repair_octaves, _pitch_prefilter
    x = _pitch_prefilter(np.asarray(x, dtype=np.float32), sr)
    frame = 1024
    raw = []
    for fr in _frames(x, frame=frame, hop=hop):
        ac = _autocorr(fr)
        lag = _period_lag(ac, int(sr / hi), min(int(sr / lo), len(ac) - 1))
        raw.append(sr / float(lag) if lag else 0.0)
    f0 = _repair_octaves(raw)
    n = int(np.ceil(len(x) / hop))
    out = np.zeros(n, dtype=np.float32)
    m = min(n, len(f0))
    out[:m] = np.asarray(f0[:m], dtype=np.float32)
    return out


def _frame_energy(mel: np.ndarray) -> np.ndarray:
    """Energy proxy straight from the mel (mean dB), no extra windowing pass."""
    return mel.mean(axis=1).astype(np.float32)


if HAS_TORCH:

    class VoiceDataset(Dataset):
        """(f0, energy) → mel residual, on fixed-length crops."""

        def __init__(self, mel_path: str, indices: np.ndarray, cfg: VoiceConfig,
                     mean_spec: np.ndarray):
            self.blob = np.load(mel_path, mmap_mode="r")
            self.mel = self.blob["mel"]
            self.mask = self.blob["mask"]
            self.f0 = self.blob["f0"] if "f0" in self.blob else None
            self.idx = np.asarray(indices)
            self.cfg = cfg
            self.mean_spec = mean_spec.astype(np.float32)

        def __len__(self) -> int:
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            m = np.asarray(self.mel[j], dtype=np.float32)
            k = np.asarray(self.mask[j], dtype=np.float32)
            f = (np.asarray(self.f0[j], dtype=np.float32) if self.f0 is not None
                 else np.zeros(m.shape[0], dtype=np.float32))
            valid = int(k.sum()) or m.shape[0]
            T = min(self.cfg.crop_frames, valid)
            a = int(np.random.randint(0, max(1, valid - T + 1)))
            m = m[a:a + T]
            f = f[a:a + T]
            if m.shape[0] < T:
                pad = T - m.shape[0]
                m = np.pad(m, ((0, pad), (0, 0)), mode="edge")
                f = np.pad(f, (0, pad), mode="edge")
            resid = m - self.mean_spec[None, :]
            feat = np.stack([
                np.where(f > 0, np.log2(np.maximum(f, 1.0)) - _F0_CENTRE, 0.0),
                (f > 0).astype(np.float32),
                (_frame_energy(m) + 80.0) / 80.0,
            ], axis=1).astype(np.float32)
            return feat, resid, np.ones(T, dtype=np.float32)

    class VoiceColorModel(torch.nn.Module):
        """Predicts a mel residual from pitch, voicing and energy alone.

        Deliberately tiny (~0.3 M): the function it learns is smooth — a singer's
        formant envelope varies with pitch in a slowly changing way — and a big
        network on four hours of one voice would memorise clips instead.
        """

        def __init__(self, hidden: int = 128, n_layers: int = 3, n_mels: int = N_MELS,
                     dropout: float = 0.05):
            super().__init__()
            self.inp = torch.nn.Conv1d(3, hidden, 5, padding=2)
            self.blocks = torch.nn.ModuleList([
                torch.nn.Conv1d(hidden, hidden, 5, padding=2 * 2 ** (i % 3),
                                dilation=2 ** (i % 3)) for i in range(n_layers)])
            self.norms = torch.nn.ModuleList([
                torch.nn.GroupNorm(8, hidden) for _ in range(n_layers)])
            self.drop = torch.nn.Dropout(dropout)
            self.head = torch.nn.Conv1d(hidden, n_mels, 1)

        def forward(self, feat):
            h = torch.nn.functional.gelu(self.inp(feat.transpose(1, 2)))
            for blk, nrm in zip(self.blocks, self.norms):
                h = h + torch.nn.functional.gelu(nrm(blk(h)))
                h = self.drop(h)
            return self.head(h).transpose(1, 2)

        def n_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

else:                                                # pragma: no cover
    VoiceColorModel = object                         # type: ignore


def corpus_mean_spectrum(npz: str) -> np.ndarray:
    """The baseline the model predicts a *residual* against.

    Using the corpus mean (rather than zero) makes the target small and centred, so
    the first epoch is already at the level of "sounds like an average frame" instead
    of "sounds like noise".
    """
    blob = np.load(npz, mmap_mode="r")
    mel, mask = blob["mel"], blob["mask"]
    tot = np.zeros(N_MELS, dtype=np.float64)
    w = 0.0
    for i in range(0, mel.shape[0], max(1, mel.shape[0] // 200)):
        m = np.asarray(mel[i], dtype=np.float64)
        k = np.asarray(mask[i], dtype=np.float64)
        tot += (m * k[:, None]).sum(axis=0)
        w += k.sum()
    return (tot / max(w, 1.0)).astype(np.float32)


def add_f0_to_corpus(npz: str, verbose: bool = True) -> str:
    """Compute and store a per-frame F0 track for every segment.

    Written back into the same ``.npz`` under the ``f0`` key.  Doing it once at build
    time rather than in the training loop is worth the extra pass: pitch tracking is
    the most expensive non-GPU step, and a Kaggle GPU session should not spend its
    quota on it.
    """
    blob = np.load(npz)
    if "f0" in blob:
        if verbose:
            print("  f0 already present")
        return npz
    from ..audio.mel import read_audio_window
    mel, mask = blob["mel"], blob["mask"]
    meta = [json.loads(s) for s in blob["meta"]]
    f0 = np.zeros(mel.shape[:2], dtype=np.float32)
    for i, md in enumerate(meta):
        n = int(mask[i].sum())
        try:
            seg = read_audio_window(md["audio"], md["start"], md["end"], pad_s=0.0)
        except Exception:
            continue
        if not len(seg):
            continue
        tr = f0_from_audio(seg)
        m = min(n, len(tr))
        f0[i, :m] = tr[:m]
    np.savez_compressed(npz, mel=mel, mask=mask, meta=blob["meta"], f0=f0)
    if verbose:
        voiced = float((f0 > 0).mean()) * 100
        print(f"  f0 written for {len(meta)} segments ({voiced:.0f} % frames voiced)")
    return npz


def train_voice(npz: str, out: str, cfg: VoiceConfig | None = None,
                verbose: bool = True) -> dict:
    """Train the voice-colour model and write a checkpoint."""
    if not HAS_TORCH:
        raise RuntimeError("torch is required: pip install torch")
    cfg = cfg or VoiceConfig()
    blob = np.load(npz, mmap_mode="r")
    if "f0" not in blob:
        raise ValueError("corpus has no f0 track — call add_f0_to_corpus() first")
    n = blob["mel"].shape[0]
    if n < 6:
        raise ValueError(f"corpus has only {n} segments — need at least 6 "
                         "(and more is much better; see docs/DATA-ADDENDUM.md §6)")

    mean_spec = corpus_mean_spectrum(npz)
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(n)
    # Never let validation be a single segment: on a tiny corpus one held-out clip
    # makes the val curve jump by 20 % between epochs and early stopping fires on
    # noise.  Two is the minimum that gives a stable mean.
    n_val = min(max(2, int(n * cfg.val_fraction)), max(2, n - 2))
    val_idx, tr_idx = order[:n_val], order[n_val:]

    device = _pick_device(cfg.device)
    model = VoiceColorModel(hidden=cfg.hidden, n_layers=cfg.n_layers,
                            dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=max(1, cfg.patience // 2))

    def collate(batch):
        T = max(b[0].shape[0] for b in batch)
        B = len(batch)
        feat = np.zeros((B, T, 3), dtype=np.float32)
        resid = np.zeros((B, T, N_MELS), dtype=np.float32)
        mask = np.zeros((B, T), dtype=np.float32)
        for i, (f, r, k) in enumerate(batch):
            L = f.shape[0]
            feat[i, :L] = f
            resid[i, :L] = r
            mask[i, :L] = k
        return (torch.from_numpy(feat), torch.from_numpy(resid), torch.from_numpy(mask))

    tr = DataLoader(VoiceDataset(npz, tr_idx, cfg, mean_spec), batch_size=cfg.batch_size,
                    shuffle=True, collate_fn=collate, num_workers=cfg.num_workers)
    va = DataLoader(VoiceDataset(npz, val_idx, cfg, mean_spec), batch_size=cfg.batch_size,
                    shuffle=False, collate_fn=collate, num_workers=cfg.num_workers)

    if verbose:
        print(f"device={device}  params={model.n_params()/1e6:.2f} M  "
              f"train={len(tr_idx)}  val={len(val_idx)}  "
              f"crop={cfg.crop_frames} frames (~{cfg.crop_frames/HOP*HOP/SR*HOP:.1f} s)")

    def epoch(loader, train: bool):
        model.train(train)
        tot, nb = 0.0, 0
        with torch.set_grad_enabled(train):
            for feat, resid, mask in loader:
                feat, resid, mask = feat.to(device), resid.to(device), mask.to(device)
                pred = model(feat)
                m = mask.unsqueeze(-1)
                loss = (torch.abs(pred - resid) * m).sum() / m.sum().clamp(min=1.0)
                loss = loss / resid.shape[-1]
                if train:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                tot += float(loss.detach())
                nb += 1
        return tot / max(nb, 1)

    t0 = time.time()
    best, best_state, best_ep = math.inf, None, -1
    stale = 0
    base_val = None
    for ep in range(1, cfg.epochs + 1):
        tl = epoch(tr, True)
        vl = epoch(va, False)
        sched.step(vl)
        if base_val is None:
            base_val = vl
        if vl < best - 1e-6:
            best, best_ep, stale = vl, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if verbose and (ep == 1 or ep % max(1, cfg.epochs // 12) == 0 or ep == cfg.epochs):
            print(f"  ep {ep:3d}/{cfg.epochs}  train {tl:.4f}  val {vl:.4f}"
                  + ("  *" if ep == best_ep else ""))
        if stale >= cfg.patience:
            if verbose:
                print(f"  early stop at epoch {ep} (no improvement for {stale})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    ckpt = {
        "version": 2,
        "kind": "voice-color",
        "voice_state": {k: v.cpu().numpy() for k, v in model.state_dict().items()},
        "mean_spectrum": mean_spec.tolist(),
        "config": {
            "hidden": cfg.hidden, "n_layers": cfg.n_layers, "n_mels": N_MELS,
            "sr": SR, "hop": HOP, "crop_frames": cfg.crop_frames,
            "f0_centre": _F0_CENTRE,
        },
        "metrics": {
            "val_residual_l1": round(best, 5),
            "val_residual_l1_epoch0": round(base_val, 5),
            "improvement_pct": round(100.0 * (1 - best / max(base_val, 1e-9)), 1),
            "segments": int(n),
            "train_seconds": round(time.time() - t0, 1),
            "best_epoch": best_ep,
        },
    }
    if not HAS_TORCH:
        raise RuntimeError("torch required")
    # A one-line notebook should not have to mkdir; the parent may be "work/",
    # "checkpoints/", or an absolute path that does not exist yet.
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    if verbose:
        print(f"  saved {out}  ({ckpt['metrics']['val_residual_l1']:.4f} L1, "
              f"{ckpt['metrics']['improvement_pct']:+.1f} % vs epoch 1, "
              f"{ckpt['metrics']['train_seconds']:.0f} s)")
        if ckpt["metrics"]["improvement_pct"] < 1.0:
            print("  ! the model barely beat the corpus mean — this voice is not "
                  "distinctive\n    enough for the data given, or the corpus is too small.")
    return ckpt["metrics"]


def load_voice(path: str):
    """Load a v2 voice checkpoint, refusing anything else rather than guessing."""
    if not HAS_TORCH:
        raise RuntimeError("torch is required")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or blob.get("version") != 2:
        raise ValueError(
            f"{path} is not a v2 voice checkpoint (got version "
            f"{blob.get('version') if isinstance(blob, dict) else '?'}). "
            "Re-train with samadhvani.train.voice_trainer.")
    cfg = blob["config"]
    model = VoiceColorModel(hidden=cfg["hidden"], n_layers=cfg["n_layers"])
    model.load_state_dict({k: torch.tensor(v) for k, v in blob["voice_state"].items()})
    model.eval()
    return model, np.asarray(blob["mean_spectrum"], dtype=np.float32), cfg


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m samadhvani.train.voice_trainer --npz corpus.npz``."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="samadhvani.train.voice_trainer",
        description="Train the voice-colour model on a packed, pitch-annotated corpus.")
    ap.add_argument("--npz", required=True, help="packed corpus from `data.build corpus`")
    ap.add_argument("--out", default="checkpoints/voice.pt", help="checkpoint path")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--f0-only", action="store_true",
                    help="just add the pitch track to the corpus and stop")
    args = ap.parse_args(argv)

    if not Path(args.npz).exists():
        print(f"no such corpus: {args.npz}\n"
              "build one first: python -m samadhvani.data.build corpus --hours 4")
        return 2
    if args.f0_only:
        add_f0_to_corpus(args.npz)
        return 0
    add_f0_to_corpus(args.npz)
    m = train_voice(args.npz, args.out,
                    VoiceConfig(epochs=args.epochs, batch_size=args.batch_size,
                                lr=args.lr, hidden=args.hidden,
                                n_layers=args.n_layers, device=args.device))
    print(f"\nval residual L1 {m['val_residual_l1']:.4f} "
          f"({m['improvement_pct']:+.1f} % vs the corpus mean)")
    print(f"next: python -m samadhvani chant \"<gāna text>\" --engine voice "
          f"--checkpoint {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
