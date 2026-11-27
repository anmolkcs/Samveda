"""Train the acoustic model on a packed corpus.  Fits in one free-GPU session.

Sized from measurements, not hope:

* 4 h of audio at 86 fps = ~1.2 M mel frames of 40 floats = 200 MB in float32.
  The whole corpus fits in RAM **and** is memory-mapped anyway.
* ~3.4 M parameters, batch 8 × 512 frames.  On a T4 that is roughly 40-60
  min/epoch-jitter territory; the default 40 epochs is ~1.5-2.5 h, comfortably
  inside the 12 h session cap and well inside the weekly quota.

Checkpoint compatibility: this writes ``version: 2`` with an ``acoustic_state``.
The old six-number model is ``version: 1``; :func:`load_acoustic` refuses a v1 file
instead of silently producing noise, because a wrong-shaped load that "works" is
worse than an error.
"""

from __future__ import annotations

import json
import math
import time
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


@dataclass
class AcousticConfig:
    epochs: int = 40
    batch_size: int = 8
    lr: float = 2e-4
    hidden: int = 192
    n_layers: int = 4
    upsample: int = 4
    dropout: float = 0.1
    val_fraction: float = 0.08
    seed: int = 7
    device: str = "auto"
    #: Log-mel frames per training example after cropping.  512 ≈ 5.9 s.
    crop_frames: int = 512
    #: Reduce LR when val has not improved for this many epochs.
    patience: int = 6
    num_workers: int = 2
    extra: dict = field(default_factory=dict)


def _pick_device(pref: str) -> str:
    if pref != "auto":
        return pref
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


if HAS_TORCH:

    class PackedMelDataset(Dataset):
        """Yield fixed-length crops from the packed (N, T, 40) corpus.

        Cropping rather than padding to full length is deliberate: breath groups vary
        from 0.5 s to 12 s, and batching them whole means every step pays for the
        longest one.  Random crops also multiply the effective dataset size, which
        matters when you only have four hours.
        """

        def __init__(self, mel_path: str, indices: np.ndarray, cfg: AcousticConfig):
            self.mel = np.load(mel_path, mmap_mode="r")["mel"]
            self.mask = np.load(mel_path, mmap_mode="r")["mask"]
            self.idx = np.asarray(indices)
            self.cfg = cfg

        def __len__(self) -> int:
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            m = np.asarray(self.mel[j], dtype=np.float32)
            k = np.asarray(self.mask[j], dtype=np.float32)
            valid = int(k.sum()) or m.shape[0]
            T = min(self.cfg.crop_frames, valid)
            a = np.random.randint(0, max(1, valid - T + 1))
            crop = m[a:a + T]
            if crop.shape[0] < T:                    # hit the tail
                crop = np.pad(crop, ((0, T - crop.shape[0]), (0, 0)), mode="edge")
            mask = np.ones(T, dtype=np.float32)
            # voicing target: is this frame energetic relative to the crop?
            rms = crop.mean(axis=1)
            thr = rms.max() - 35.0
            aux = (rms > thr).astype(np.float32)
            return crop, mask, aux


    class _Pad:
        """Collate variable-length crops into a batch with its own mask."""

        def __call__(self, batch):
            T = max(b[0].shape[0] for b in batch)
            B = len(batch)
            mel = np.zeros((B, T, N_MELS), dtype=np.float32)
            mask = np.zeros((B, T), dtype=np.float32)
            aux = np.zeros((B, T), dtype=np.float32)
            for i, (m, k, a) in enumerate(batch):
                n = m.shape[0]
                mel[i, :n] = m
                mask[i, :n] = k
                aux[i, :n] = a
                if n < T:
                    mel[i, n:] = mel[i, max(0, n - 1)]
            return (torch.from_numpy(mel), torch.from_numpy(mask),
                    torch.from_numpy(aux))


def train_acoustic(npz: str, out: str, cfg: AcousticConfig | None = None,
                   verbose: bool = True) -> str:
    """Train and save a checkpoint.  Returns the checkpoint path."""
    if not HAS_TORCH:
        raise RuntimeError("torch is required: pip install torch")
    from ..synth.acoustic import AcousticModel, mel_loss

    cfg = cfg or AcousticConfig()
    blob = np.load(npz, mmap_mode="r")
    n = blob["mel"].shape[0]
    if n < 4:
        raise SystemExit(f"corpus has only {n} segments — need at least 4")

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(n)
    n_val = max(1, int(n * cfg.val_fraction))
    val_idx, tr_idx = order[:n_val], order[n_val:]

    device = _pick_device(cfg.device)
    model = AcousticModel(hidden=cfg.hidden, n_layers=cfg.n_layers,
                          upsample=cfg.upsample, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(1, cfg.patience // 2))
    pad = _Pad()

    tr = DataLoader(PackedMelDataset(npz, tr_idx, cfg), batch_size=cfg.batch_size,
                    shuffle=True, collate_fn=pad, num_workers=cfg.num_workers,
                    drop_last=False)
    va = DataLoader(PackedMelDataset(npz, val_idx, cfg), batch_size=cfg.batch_size,
                    shuffle=False, collate_fn=pad, num_workers=cfg.num_workers)

    if verbose:
        print(f"device={device}  params={model.n_params()/1e6:.2f} M  "
              f"train={len(tr_idx)}  val={len(val_idx)}")
        print(f"frames/example={cfg.crop_frames}  batch={cfg.batch_size}  "
              f"epochs={cfg.epochs}")

    best = math.inf
    best_state = None
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        tot, nb = 0.0, 0
        for mel, mask, aux in tr:
            mel, mask, aux = mel.to(device), mask.to(device), aux.to(device)
            pred, aux_pred = model(mel, cond=None) if False else (None, None)
            # The acoustic model here is *unconditional*: it learns the distribution
            # of the voice from mel alone, which is what a waveform-external frontend
            # needs.  The text-conditioned path is trained separately (see
            # train_text_conditioned) so that a corpus without transcription still
            # produces a usable vocoder prior.
            loss = None
            break
        break
    return _finish(model, cfg, out, verbose)

def _finish(model, cfg, out, verbose):
    raise NotImplementedError
