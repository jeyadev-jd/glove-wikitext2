"""Sparse word-context co-occurrence construction.

A dense ``V x V`` float32 matrix at ``V = 20,000`` would need 400M cells
(1.49 GiB), which neither fits comfortably in 4 GB of VRAM nor is remotely
necessary: real corpora fill well under 1% of it. This module therefore builds
the matrix as ``(row, col, value)`` triplets and never materialises the dense
form.

Windowing follows the original GloVe ``cooccur.c``: out-of-vocabulary tokens
are dropped from the stream first, and the symmetric window is applied over the
retained token sequence. For a target at position ``p`` and a context at
position ``q`` inside the window, the increment is ``1 / |p - q|``, so distant
context words contribute less. Both directions ``(i, j)`` and ``(j, i)`` are
accumulated, making ``X`` symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm


@dataclass
class CooccurrenceMatrix:
    """Sparse symmetric co-occurrence data in coordinate (COO) form."""

    rows: np.ndarray          # int32, shape (nnz,)
    cols: np.ndarray          # int32, shape (nnz,)
    values: np.ndarray        # float32, shape (nnz,)
    vocab_size: int

    @property
    def nnz(self) -> int:
        """Number of stored non-zero entries."""
        return int(self.values.shape[0])

    def to_dict(self) -> Dict[Tuple[int, int], float]:
        """Dense-free dictionary view, mainly for tests and inspection."""
        return {
            (int(r), int(c)): float(v)
            for r, c, v in zip(self.rows, self.cols, self.values)
        }

    def get(self, i: int, j: int) -> float:
        """Look up ``X[i, j]`` (0.0 if the pair never co-occurred)."""
        mask = (self.rows == i) & (self.cols == j)
        if not mask.any():
            return 0.0
        return float(self.values[mask][0])

    def memory_stats(self) -> Dict[str, float]:
        """Dense vs sparse footprint and sparsity of the matrix."""
        vocab = self.vocab_size
        possible = float(vocab) * float(vocab)
        dense_bytes = possible * 4.0                       # float32
        sparse_bytes = float(self.nnz) * (4 + 4 + 4)       # int32 + int32 + float32
        return {
            "vocab_size": vocab,
            "possible_dense_entries": possible,
            "nonzero_entries": self.nnz,
            "sparsity_percent": 100.0 * (1.0 - self.nnz / possible) if possible else 0.0,
            "density_percent": 100.0 * self.nnz / possible if possible else 0.0,
            "dense_memory_bytes": dense_bytes,
            "sparse_memory_bytes": sparse_bytes,
            "compression_ratio": dense_bytes / sparse_bytes if sparse_bytes else 0.0,
        }


def build_cooccurrence(indexed_tokens: Sequence[int], vocab_size: int,
                       window_size: int = 1,
                       verbose: bool = True,
                       chunk_size: int = 5_000_000) -> CooccurrenceMatrix:
    """Accumulate distance-weighted co-occurrence counts over a token stream.

    Args:
        indexed_tokens: corpus as vocabulary indices (OOV already removed).
        vocab_size: number of vocabulary entries.
        window_size: symmetric context radius (>= 1).
        verbose: print progress and summary statistics.
        chunk_size: tokens processed per vectorised pass (memory/speed knob).

    Returns:
        A :class:`CooccurrenceMatrix` with duplicate pairs summed.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if vocab_size < 1:
        raise ValueError("vocab_size must be >= 1")

    tokens = np.asarray(indexed_tokens, dtype=np.int64)
    n = tokens.shape[0]
    if n < 2:
        empty_i = np.empty(0, dtype=np.int32)
        return CooccurrenceMatrix(empty_i, empty_i.copy(),
                                  np.empty(0, dtype=np.float32), vocab_size)

    # Emit (pair_key, weight) events chunk-by-chunk, then aggregate once with a
    # sort-based group-by. Fully vectorised: no Python loop over token pairs.
    key_blocks: List[np.ndarray] = []
    weight_blocks: List[np.ndarray] = []
    total_steps = window_size * max(1, (n + chunk_size - 1) // chunk_size)
    bar = tqdm(total=total_steps, desc="co-occurrence", disable=not verbose)

    for distance in range(1, window_size + 1):
        weight = 1.0 / distance
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            left = tokens[start:stop]
            right = tokens[start + distance: stop + distance]
            span = min(left.shape[0], right.shape[0])
            if span > 0:
                left, right = left[:span], right[:span]
                # Both directions -> symmetric matrix.
                keys = np.concatenate([left * vocab_size + right,
                                       right * vocab_size + left])
                unique_keys, counts = np.unique(keys, return_counts=True)
                key_blocks.append(unique_keys)
                weight_blocks.append(counts.astype(np.float64) * weight)
            bar.update(1)
    bar.close()

    if not key_blocks:
        empty_i = np.empty(0, dtype=np.int32)
        return CooccurrenceMatrix(empty_i, empty_i.copy(),
                                  np.empty(0, dtype=np.float32), vocab_size)

    all_keys = np.concatenate(key_blocks)
    all_weights = np.concatenate(weight_blocks)
    keys, inverse = np.unique(all_keys, return_inverse=True)
    values = np.bincount(inverse, weights=all_weights, minlength=keys.shape[0])

    rows = (keys // vocab_size).astype(np.int32)
    cols = (keys % vocab_size).astype(np.int32)
    matrix = CooccurrenceMatrix(rows, cols, values.astype(np.float32), vocab_size)

    if verbose:
        print_cooccurrence_stats(matrix)
    return matrix


def print_cooccurrence_stats(matrix: CooccurrenceMatrix) -> Dict[str, float]:
    """Print (and return) the sparsity/memory summary of a co-occurrence matrix."""
    stats = matrix.memory_stats()
    dense_gb = stats["dense_memory_bytes"] / 1024 ** 3
    sparse_mb = stats["sparse_memory_bytes"] / 1024 ** 2
    print(f"Vocabulary size:            {stats['vocab_size']:,}")
    print(f"Non-zero co-occurrences:    {stats['nonzero_entries']:,}")
    print(f"Possible dense entries:     {stats['possible_dense_entries']:,.0f}")
    print(f"Sparsity:                   {stats['sparsity_percent']:.4f}%")
    print(f"Estimated dense memory:     {dense_gb:.3f} GB (float32)")
    print(f"Estimated sparse memory:    {sparse_mb:.2f} MB (i32+i32+f32)")
    print(f"Compression ratio:          {stats['compression_ratio']:.1f}x")
    return stats


def build_from_text(tokens: Iterable[str], word_to_idx: Dict[str, int],
                    window_size: int = 1, verbose: bool = False,
                    vocab_size: Optional[int] = None) -> CooccurrenceMatrix:
    """Convenience wrapper: string tokens -> indices -> co-occurrence matrix."""
    indexed: List[int] = [word_to_idx[t] for t in tokens if t in word_to_idx]
    size = vocab_size if vocab_size is not None else len(word_to_idx)
    return build_cooccurrence(indexed, size, window_size=window_size, verbose=verbose)
