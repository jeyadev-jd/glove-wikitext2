"""Experiment orchestration: one reusable pipeline, ablations, CPU/GPU benchmark."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import config as project_config
from src.cooccurrence import CooccurrenceMatrix, build_cooccurrence
from src.evaluation import (EmbeddingIndex, evaluate_analogies, oov_analysis)
from src.glove import GloVeModel
from src.preprocessing import Vocabulary, build_vocabulary, tokenize
from src.training import train_glove
from src.utils import reset_gpu_memory_stats, set_seed


@dataclass
class PipelineResult:
    """Everything produced by a single end-to-end training run."""

    config: Dict[str, Any]
    vocabulary: Vocabulary
    matrix: CooccurrenceMatrix
    model: GloVeModel
    training: Dict[str, Any]
    index: EmbeddingIndex


class CorpusCache:
    """Tokenises the corpus once and reuses it across experiments."""

    def __init__(self, text: str) -> None:
        self.tokens: List[str] = tokenize(text)

    def __len__(self) -> int:
        return len(self.tokens)


def run_pipeline(tokens: Sequence[str], cfg: Dict[str, Any],
                 device: torch.device, name: str = "custom-glove",
                 checkpoint_dir: Optional[str] = None,
                 resume: bool = False, verbose: bool = True) -> PipelineResult:
    """Vocabulary -> co-occurrence -> training -> embedding index for one config."""
    set_seed(int(cfg["seed"]))

    vocabulary = build_vocabulary(
        tokens,
        min_frequency=int(cfg["min_frequency"]),
        max_vocab_size=int(cfg["max_vocab_size"]),
        verbose=verbose,
    )
    indexed = vocabulary.encode(tokens)
    matrix = build_cooccurrence(indexed, len(vocabulary),
                                window_size=int(cfg["window_size"]),
                                verbose=verbose)

    reset_gpu_memory_stats(device)
    model = GloVeModel(
        vocab_size=len(vocabulary),
        embedding_dim=int(cfg["embedding_dim"]),
        device=device,
        seed=int(cfg["seed"]),
        init_scale=float(cfg["init_scale"]),
        x_max=float(cfg["x_max"]),
        alpha=float(cfg["alpha"]),
        epsilon=float(cfg["epsilon"]),
    )
    training = train_glove(model, matrix, cfg, device,
                           checkpoint_dir=checkpoint_dir, resume=resume,
                           verbose=verbose)

    index = EmbeddingIndex(model.embeddings().numpy(), vocabulary.idx_to_word,
                           name=name)
    return PipelineResult(cfg, vocabulary, matrix, model, training, index)


def _summarise_run(result: PipelineResult,
                   analogies: Sequence[Tuple[str, str, str, str]]
                   ) -> Dict[str, Any]:
    _, analogy_summary = evaluate_analogies(result.index, analogies)
    coverage = oov_analysis(result.index, analogies)
    return {
        "vocab_size": len(result.vocabulary),
        "nonzero_pairs": result.matrix.nnz,
        "embedding_dim": result.model.embedding_dim,
        "final_loss": result.training["final_loss"],
        "analogy_accuracy": analogy_summary["accuracy"],
        "analogy_evaluated": analogy_summary["evaluated"],
        "analogy_oov": analogy_summary["oov_analogies"],
        "OOV_rate": coverage["oov_rate_percent"],
        "training_time": result.training["training_time_s"],
        "peak_gpu_memory_mb": result.training["peak_gpu_memory_mb"],
    }


def run_ablations(tokens: Sequence[str], base_config: Dict[str, Any],
                  device: torch.device,
                  grid: Optional[Dict[str, List[Any]]] = None,
                  analogies: Optional[Sequence[Tuple[str, str, str, str]]] = None,
                  verbose: bool = False) -> List[Dict[str, Any]]:
    """Run one-parameter-at-a-time ablations and return tabular result rows."""
    grid = grid or project_config.ABLATION_GRID
    analogies = analogies or project_config.ANALOGIES
    rows: List[Dict[str, Any]] = []

    for parameter, values in grid.items():
        for value in values:
            cfg = dict(base_config)
            cfg[parameter] = value
            cfg["checkpoint_every"] = 0        # ablations do not checkpoint
            label = f"{parameter}={value}"
            print(f"\n[ablation] {label}")
            result = run_pipeline(tokens, cfg, device,
                                  name=f"ablation-{label}", verbose=verbose)
            row = {"experiment": parameter, "parameter": parameter, "value": value}
            row.update(_summarise_run(result, analogies))
            row["window_size"] = cfg["window_size"]
            row["min_frequency"] = cfg["min_frequency"]
            row["epochs"] = cfg["epochs"]
            rows.append(row)
            print(f"[ablation] {label} -> loss={row['final_loss']:.4f} "
                  f"acc={row['analogy_accuracy']:.3f} "
                  f"time={row['training_time']:.1f}s")
            del result
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def run_cpu_gpu_benchmark(tokens: Sequence[str], base_config: Dict[str, Any],
                          epochs: int = 3,
                          verbose: bool = False) -> List[Dict[str, Any]]:
    """Time the identical configuration on CPU and CUDA.

    ``epochs`` is deliberately small: the point is a per-epoch throughput
    comparison, not a second full training run.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; benchmark needs both devices")

    cfg = dict(base_config)
    cfg["epochs"] = int(epochs)
    cfg["checkpoint_every"] = 0
    rows: List[Dict[str, Any]] = []

    for device_name in ("cpu", "cuda"):
        device = torch.device(device_name)
        print(f"\n[benchmark] {device_name} ({epochs} epochs)")
        result = run_pipeline(tokens, cfg, device,
                              name=f"bench-{device_name}", verbose=verbose)
        history = result.training["history"]
        rows.append({
            "device": device_name,
            "gpu_name": (torch.cuda.get_device_properties(0).name
                         if device_name == "cuda" else ""),
            "epochs": epochs,
            "training_time_s": result.training["training_time_s"],
            "mean_epoch_time_s": (sum(history.epoch_times) / len(history.epoch_times)
                                  if history.epoch_times else float("nan")),
            "final_loss": result.training["final_loss"],
            "batch_size": result.training["batch_size_used"],
            "peak_gpu_memory_mb": result.training["peak_gpu_memory_mb"],
        })
        del result
        torch.cuda.empty_cache()

    cpu_time = rows[0]["training_time_s"]
    gpu_time = rows[1]["training_time_s"]
    speedup = cpu_time / gpu_time if gpu_time else float("nan")
    for row in rows:
        row["gpu_speedup"] = round(speedup, 3)
    print(f"[benchmark] GPU speedup: {speedup:.2f}x")
    return rows


def export_vectors(result: PipelineResult, vectors_dir: str) -> Dict[str, str]:
    """Write ``word_vectors.txt``, ``word_vectors.npy`` and ``vocab.json``."""
    import json

    import numpy as np

    os.makedirs(vectors_dir, exist_ok=True)
    vectors = result.index.vectors
    words = result.index.idx_to_word

    npy_path = os.path.join(vectors_dir, "word_vectors.npy")
    np.save(npy_path, vectors)

    txt_path = os.path.join(vectors_dir, "word_vectors.txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        for word, row in zip(words, vectors):
            handle.write(word + " " + " ".join(f"{v:.6f}" for v in row) + "\n")

    vocab_path = os.path.join(vectors_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "idx_to_word": words,
                "word_to_idx": result.vocabulary.word_to_idx,
                "word_counts": result.vocabulary.word_counts,
                "stats": result.vocabulary.stats,
            },
            handle, ensure_ascii=False,
        )
    return {"npy": npy_path, "txt": txt_path, "vocab": vocab_path}
