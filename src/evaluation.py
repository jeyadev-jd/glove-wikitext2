"""Evaluation: cosine similarity, nearest neighbours, analogies, OOV analysis.

Cosine similarity is implemented from scratch (L2 normalisation + dot product);
no similarity utility from scikit-learn or gensim is used.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def l2_normalize(vectors: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Return ``vectors`` with unit L2 norm along the last axis."""
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.sqrt((array * array).sum(axis=-1, keepdims=True))
    return array / np.maximum(norms, epsilon)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between ``a`` (d,) or (n,d) and ``b`` (m,d)."""
    a2 = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b2 = np.atleast_2d(np.asarray(b, dtype=np.float32))
    if a2.shape[-1] != b2.shape[-1]:
        raise ValueError(f"dimension mismatch: {a2.shape[-1]} vs {b2.shape[-1]}")
    sims = l2_normalize(a2) @ l2_normalize(b2).T
    return sims[0] if np.ndim(a) == 1 else sims


class EmbeddingIndex:
    """Word vectors plus lookup helpers shared by custom and pretrained models."""

    def __init__(self, vectors: np.ndarray, idx_to_word: Sequence[str],
                 name: str = "model") -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[0] != len(idx_to_word):
            raise ValueError("vectors and vocabulary have different lengths")
        self.name = name
        self.vectors = vectors
        self.idx_to_word = list(idx_to_word)
        self.word_to_idx = {w: i for i, w in enumerate(self.idx_to_word)}
        self._unit = l2_normalize(vectors)

    def __contains__(self, word: str) -> bool:
        return word in self.word_to_idx

    def __len__(self) -> int:
        return len(self.idx_to_word)

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return int(self.vectors.shape[1])

    def vector(self, word: str) -> np.ndarray:
        """Raw vector for ``word``.

        Raises:
            KeyError: if the word is out of vocabulary.
        """
        if word not in self.word_to_idx:
            raise KeyError(f"'{word}' is out of vocabulary for {self.name}")
        return self.vectors[self.word_to_idx[word]]

    def _rank(self, query_vector: np.ndarray, top_k: int,
              exclude: Iterable[str]) -> List[Tuple[str, float]]:
        query_unit = l2_normalize(np.asarray(query_vector, dtype=np.float32))
        scores = self._unit @ query_unit
        blocked = {self.word_to_idx[w] for w in exclude if w in self.word_to_idx}
        if blocked:
            scores = scores.copy()
            scores[list(blocked)] = -np.inf
        k = min(top_k, len(self.idx_to_word))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.idx_to_word[i], float(scores[i])) for i in top]

    def nearest_neighbors(self, word: str, top_k: int = 10
                          ) -> List[Tuple[str, float]]:
        """Top-``k`` most cosine-similar words, excluding ``word`` itself."""
        return self._rank(self.vector(word), top_k, exclude=[word])

    def analogy(self, a: str, b: str, c: str, top_k: int = 1
                ) -> List[Tuple[str, float]]:
        """Solve ``a - b + c`` and return the best candidates.

        The three source words are excluded from the candidate set.
        """
        query = self.vector(a) - self.vector(b) + self.vector(c)
        return self._rank(query, top_k, exclude=[a, b, c])


def select_query_words(index: EmbeddingIndex, preferred: Sequence[str],
                       fallbacks: Sequence[str], count: int = 5
                       ) -> Tuple[List[str], List[str]]:
    """Pick ``count`` in-vocabulary query words; returns ``(chosen, missing)``.

    Preferred words that are out of vocabulary are reported and replaced from
    ``fallbacks``, then from the most frequent remaining vocabulary entries.
    """
    chosen = [w for w in preferred if w in index]
    missing = [w for w in preferred if w not in index]
    for candidate in list(fallbacks) + index.idx_to_word:
        if len(chosen) >= count:
            break
        if candidate in index and candidate not in chosen:
            chosen.append(candidate)
    return chosen[:count], missing


def evaluate_nearest_neighbors(index: EmbeddingIndex, queries: Sequence[str],
                               top_k: int = 10) -> List[Dict[str, object]]:
    """Nearest-neighbour rows: query, rank, neighbor, cosine_similarity."""
    rows: List[Dict[str, object]] = []
    for query in queries:
        if query not in index:
            continue
        for rank, (neighbor, score) in enumerate(
                index.nearest_neighbors(query, top_k), start=1):
            rows.append({
                "model": index.name,
                "query": query,
                "rank": rank,
                "neighbor": neighbor,
                "cosine_similarity": round(score, 6),
            })
    return rows


def evaluate_analogies(index: EmbeddingIndex,
                       analogies: Sequence[Tuple[str, str, str, str]]
                       ) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """Evaluate ``a - b + c ~= expected`` over an analogy set.

    Quadruples containing an out-of-vocabulary word are marked ``OOV`` and are
    excluded from the accuracy denominator (reported separately).

    Returns:
        ``(rows, summary)`` where summary holds accuracy and OOV statistics.
    """
    rows: List[Dict[str, object]] = []
    correct = evaluated = oov = 0

    for a, b, c, expected in analogies:
        missing = [w for w in (a, b, c, expected) if w not in index]
        if missing:
            oov += 1
            rows.append({
                "model": index.name, "a": a, "b": b, "c": c,
                "expected": expected, "predicted": "OOV", "correct": "OOV",
                "similarity": "", "missing_words": " ".join(missing),
            })
            continue
        (predicted, score), = index.analogy(a, b, c, top_k=1)
        is_correct = predicted == expected
        evaluated += 1
        correct += int(is_correct)
        rows.append({
            "model": index.name, "a": a, "b": b, "c": c,
            "expected": expected, "predicted": predicted,
            "correct": bool(is_correct), "similarity": round(score, 6),
            "missing_words": "",
        })

    summary = {
        "model": index.name,
        "total_analogies": len(analogies),
        "evaluated": evaluated,
        "oov_analogies": oov,
        "correct": correct,
        "accuracy": (correct / evaluated) if evaluated else 0.0,
    }
    return rows, summary


def oov_analysis(index: EmbeddingIndex,
                 analogies: Sequence[Tuple[str, str, str, str]],
                 extra_words: Optional[Sequence[str]] = None
                 ) -> Dict[str, float]:
    """Vocabulary coverage of the evaluation words for one embedding model."""
    words: List[str] = [w for quad in analogies for w in quad]
    words.extend(extra_words or [])
    unique = sorted(set(words))
    found = [w for w in unique if w in index]
    missing = [w for w in unique if w not in index]
    return {
        "model": index.name,
        "vocab_size": len(index),
        "total_evaluation_words": len(unique),
        "words_found": len(found),
        "words_missing": len(missing),
        "oov_count": len(missing),
        "oov_rate_percent": 100.0 * len(missing) / len(unique) if unique else 0.0,
        "missing_words": " ".join(missing),
    }
