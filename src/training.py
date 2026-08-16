"""Mini-batch GloVe training loop with checkpointing and OOM-safe batching.

Memory strategy for a 4 GB GPU: the full triplet list ``(i, j, X_ij)`` stays in
pinned-free CPU memory; only the current mini-batch is copied to the device.
Model parameters (~32 MB at V=20k, d=100) are the only persistent GPU tensors.
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.cooccurrence import CooccurrenceMatrix
from src.glove import GloVeModel
from src.utils import peak_gpu_memory_mb, reset_gpu_memory_stats


@dataclass
class TrainingHistory:
    """Per-epoch training telemetry."""

    epochs: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    cumulative_times: List[float] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)

    def record(self, epoch: int, loss: float, epoch_time: float,
               batch_size: int) -> None:
        """Append one epoch's measurements."""
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.epoch_times.append(epoch_time)
        previous = self.cumulative_times[-1] if self.cumulative_times else 0.0
        self.cumulative_times.append(previous + epoch_time)
        self.batch_sizes.append(batch_size)

    def to_dict(self) -> Dict[str, List[Any]]:
        """Plain-dict view for CSV/JSON serialisation."""
        return {
            "epoch": self.epochs,
            "loss": self.losses,
            "epoch_time_s": self.epoch_times,
            "cumulative_time_s": self.cumulative_times,
            "batch_size": self.batch_sizes,
        }


def build_training_tensors(matrix: CooccurrenceMatrix
                           ) -> Dict[str, torch.Tensor]:
    """Convert a sparse co-occurrence matrix into CPU training tensors."""
    return {
        "i": torch.from_numpy(matrix.rows.astype(np.int64)),
        "j": torch.from_numpy(matrix.cols.astype(np.int64)),
        "x": torch.from_numpy(matrix.values.astype(np.float32)),
    }


def _is_oom(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower()


def _run_epoch(model: GloVeModel, tensors: Dict[str, torch.Tensor],
               permutation: torch.Tensor, batch_size: int,
               learning_rate: float, device: torch.device) -> float:
    """Run one epoch and return the mean per-sample loss."""
    total_loss = 0.0
    n = permutation.numel()
    for start in range(0, n, batch_size):
        idx = permutation[start:start + batch_size]
        batch_i = tensors["i"][idx].to(device)
        batch_j = tensors["j"][idx].to(device)
        batch_x = tensors["x"][idx].to(device)
        total_loss += model.train_batch(batch_i, batch_j, batch_x, learning_rate)
    return total_loss / max(n, 1)


def save_checkpoint(model: GloVeModel, history: TrainingHistory, epoch: int,
                    config: Dict[str, Any], checkpoint_dir: str) -> str:
    """Persist model, accumulators, history and config for epoch ``epoch``."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:02d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "history": history.to_dict(),
            "config": config,
        },
        path,
    )
    return path


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the highest-epoch checkpoint in ``checkpoint_dir`` (or None)."""
    matches = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")))
    return matches[-1] if matches else None


def load_checkpoint(path: str, model: GloVeModel) -> tuple[int, TrainingHistory]:
    """Restore ``model`` from ``path``; returns ``(next_epoch, history)``."""
    # Checkpoints hold only tensors and primitive containers, so the safe
    # weights_only loader is sufficient (and avoids arbitrary unpickling).
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    hist_dict = payload["history"]
    history = TrainingHistory(
        epochs=list(hist_dict["epoch"]),
        losses=list(hist_dict["loss"]),
        epoch_times=list(hist_dict["epoch_time_s"]),
        cumulative_times=list(hist_dict["cumulative_time_s"]),
        batch_sizes=list(hist_dict["batch_size"]),
    )
    return int(payload["epoch"]) + 1, history


def train_glove(model: GloVeModel, matrix: CooccurrenceMatrix,
                config: Dict[str, Any], device: torch.device,
                checkpoint_dir: Optional[str] = None,
                resume: bool = False,
                verbose: bool = True,
                seed: Optional[int] = None) -> Dict[str, Any]:
    """Train ``model`` on ``matrix`` with mini-batch AdaGrad.

    Batch size is halved and the CUDA cache cleared whenever a CUDA
    out-of-memory error is raised, down to ``config['min_batch_size']``.

    Returns:
        dict with the training history, final loss, wall-clock time, the batch
        size actually used and peak GPU memory (None on CPU).
    """
    tensors = build_training_tensors(matrix)
    n_samples = tensors["x"].numel()
    if n_samples == 0:
        raise ValueError("co-occurrence matrix is empty; nothing to train on")

    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    min_batch_size = int(config.get("min_batch_size", 512))
    learning_rate = float(config["learning_rate"])
    checkpoint_every = int(config.get("checkpoint_every", 0))

    history = TrainingHistory()
    start_epoch = 1
    if resume and checkpoint_dir:
        latest = find_latest_checkpoint(checkpoint_dir)
        if latest:
            start_epoch, history = load_checkpoint(latest, model)
            if verbose:
                print(f"Resumed from {os.path.basename(latest)}; "
                      f"continuing at epoch {start_epoch}")

    generator = torch.Generator().manual_seed(int(seed if seed is not None
                                                  else config["seed"]))
    reset_gpu_memory_stats(device)
    wall_start = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        permutation = torch.randperm(n_samples, generator=generator)
        epoch_start = time.perf_counter()

        while True:
            try:
                mean_loss = _run_epoch(model, tensors, permutation, batch_size,
                                       learning_rate, device)
                break
            except RuntimeError as error:
                if not (_is_oom(error) and batch_size > min_batch_size):
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                batch_size = max(min_batch_size, batch_size // 2)
                if verbose:
                    print(f"  CUDA OOM caught; retrying with batch_size={batch_size}")

        epoch_time = time.perf_counter() - epoch_start
        history.record(epoch, mean_loss, epoch_time, batch_size)

        if verbose and config.get("log_every_epoch", True):
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {mean_loss:.6f} | "
                  f"Time: {epoch_time:.2f}s")

        if checkpoint_dir and checkpoint_every and epoch % checkpoint_every == 0:
            save_checkpoint(model, history, epoch, config, checkpoint_dir)

    total_time = time.perf_counter() - wall_start
    return {
        "history": history,
        "final_loss": history.losses[-1] if history.losses else float("nan"),
        "training_time_s": total_time,
        "batch_size_used": batch_size,
        "peak_gpu_memory_mb": peak_gpu_memory_mb(device),
        "n_samples": n_samples,
    }
