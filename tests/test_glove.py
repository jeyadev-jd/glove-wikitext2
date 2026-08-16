"""Core correctness tests for the GloVe-from-scratch implementation.

Run with:  python -m pytest tests -q      (or)      python tests/test_glove.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cooccurrence import build_cooccurrence, build_from_text  # noqa: E402
from src.evaluation import EmbeddingIndex, cosine_similarity  # noqa: E402
from src.glove import (GloVeModel, weighting_function,  # noqa: E402
                       weighting_function_scalar)
from src.preprocessing import build_vocabulary, tokenize  # noqa: E402

TOY_CORPUS = "the cat sat on the mat"


# ------------------------------------------------------------------ tokenizer

def test_tokenize_basic():
    assert tokenize("The Cat SAT on the mat.") == \
        ["the", "cat", "sat", "on", "the", "mat"]


def test_tokenize_keeps_apostrophes_and_numbers():
    assert tokenize("Don't stop -- 42 times!") == ["don't", "stop", "42", "times"]


# ----------------------------------------------------------------- vocabulary

def test_vocabulary_min_frequency_filter():
    tokens = ["a"] * 6 + ["b"] * 5 + ["c"] * 4
    vocab = build_vocabulary(tokens, min_frequency=5, max_vocab_size=100,
                             verbose=False)
    assert set(vocab.word_to_idx) == {"a", "b"}
    assert "c" not in vocab
    assert vocab.stats["removed_rare_types"] == 1


def test_vocabulary_size_cap_and_ordering():
    tokens = ["a"] * 10 + ["b"] * 9 + ["c"] * 8
    vocab = build_vocabulary(tokens, min_frequency=1, max_vocab_size=2,
                             verbose=False)
    assert vocab.idx_to_word == ["a", "b"]
    assert vocab.word_to_idx["a"] == 0


def test_vocabulary_encode_drops_oov():
    vocab = build_vocabulary(["a", "a", "b", "b"], min_frequency=1,
                             max_vocab_size=10, verbose=False)
    assert vocab.encode(["a", "zzz", "b"]) == [vocab.index("a"), vocab.index("b")]


# -------------------------------------------------------------- co-occurrence

def test_toy_cooccurrence_window_one():
    """'the cat sat on the mat' with window=1 -> each adjacent pair counts 1.0."""
    tokens = tokenize(TOY_CORPUS)
    vocab = build_vocabulary(tokens, min_frequency=1, max_vocab_size=50,
                             verbose=False)
    matrix = build_from_text(tokens, vocab.word_to_idx, window_size=1)
    x = matrix.to_dict()
    idx = vocab.word_to_idx

    # Symmetry.
    for (i, j), value in x.items():
        assert math.isclose(x[(j, i)], value, rel_tol=1e-6)
    # Adjacent pairs present with weight 1.0.
    assert math.isclose(x[(idx["the"], idx["cat"])], 1.0, rel_tol=1e-6)
    assert math.isclose(x[(idx["cat"], idx["sat"])], 1.0, rel_tol=1e-6)
    # 'the' is adjacent to 'mat' once and to 'on' once.
    assert math.isclose(x[(idx["the"], idx["mat"])], 1.0, rel_tol=1e-6)
    # Non-adjacent pair absent.
    assert (idx["cat"], idx["mat"]) not in x


def test_toy_cooccurrence_distance_weighting():
    """window=2 -> distance-2 pairs accumulate 0.5, distance-1 pairs 1.0."""
    tokens = tokenize(TOY_CORPUS)
    vocab = build_vocabulary(tokens, min_frequency=1, max_vocab_size=50,
                             verbose=False)
    matrix = build_from_text(tokens, vocab.word_to_idx, window_size=2)
    x = matrix.to_dict()
    idx = vocab.word_to_idx

    # 'the' occurs at positions 0 and 4; 'sat' at 2, so it is at distance 2 from
    # both occurrences: 0.5 + 0.5 = 1.0.
    assert math.isclose(x[(idx["the"], idx["sat"])], 1.0, rel_tol=1e-6)
    # 'cat'(1) <-> 'on'(3): distance 2, once.
    assert math.isclose(x[(idx["cat"], idx["on"])], 0.5, rel_tol=1e-6)
    # 'the'(0) <-> 'cat'(1): distance 1, once.
    assert math.isclose(x[(idx["the"], idx["cat"])], 1.0, rel_tol=1e-6)
    # 'the'(4) <-> 'mat'(5): distance 1 -> 1.0.
    assert math.isclose(x[(idx["the"], idx["mat"])], 1.0, rel_tol=1e-6)


def test_cooccurrence_accumulates_repeats():
    """Repeated bigram accumulates: 'a b a b' has a<->b at distance 1 three times."""
    matrix = build_cooccurrence([0, 1, 0, 1], vocab_size=2, window_size=1,
                                verbose=False)
    x = matrix.to_dict()
    assert math.isclose(x[(0, 1)], 3.0, rel_tol=1e-6)
    assert math.isclose(x[(1, 0)], 3.0, rel_tol=1e-6)


def test_memory_stats_sparsity():
    matrix = build_cooccurrence([0, 1, 0, 1], vocab_size=100, window_size=1,
                                verbose=False)
    stats = matrix.memory_stats()
    assert stats["possible_dense_entries"] == 100 * 100
    assert stats["nonzero_entries"] == 2
    assert stats["sparsity_percent"] > 99.0


# --------------------------------------------------------- weighting function

def test_weighting_function_regions():
    assert math.isclose(weighting_function_scalar(50.0, 100.0, 0.75),
                        (0.5) ** 0.75, rel_tol=1e-9)
    assert weighting_function_scalar(100.0, 100.0, 0.75) == 1.0
    assert weighting_function_scalar(250.0, 100.0, 0.75) == 1.0


def test_weighting_function_vectorized_matches_scalar():
    xs = [1.0, 25.0, 99.9, 100.0, 500.0]
    vector = weighting_function(torch.tensor(xs), 100.0, 0.75).tolist()
    for value, x in zip(vector, xs):
        assert math.isclose(value, weighting_function_scalar(x), rel_tol=1e-6)


# --------------------------------------------------------------- glove maths

def test_loss_matches_hand_computation():
    """Manually compute J for a one-sample batch and compare with forward()."""
    model = GloVeModel(vocab_size=3, embedding_dim=2, seed=0)
    model.W = torch.tensor([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
    model.W_context = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    model.b = torch.tensor([0.5, 0.0, 0.0])
    model.b_context = torch.tensor([0.0, -0.25, 0.0])

    i = torch.tensor([0]); j = torch.tensor([1]); x = torch.tensor([10.0])
    error, weight, loss = model.forward(i, j, x)

    expected_error = 1 * 3 + 2 * 4 + 0.5 - 0.25 - math.log(10.0)
    expected_weight = (10.0 / 100.0) ** 0.75
    assert math.isclose(float(error), expected_error, rel_tol=1e-5)
    assert math.isclose(float(weight), expected_weight, rel_tol=1e-6)
    assert math.isclose(float(loss), expected_weight * expected_error ** 2,
                        rel_tol=1e-5)


def test_analytic_gradients_match_autograd():
    """Hand-written gradients must agree with autograd on the same objective."""
    torch.manual_seed(0)
    vocab, dim = 5, 4
    model = GloVeModel(vocab_size=vocab, embedding_dim=dim, seed=1)
    i = torch.tensor([0, 2, 4]); j = torch.tensor([1, 3, 0])
    x = torch.tensor([3.0, 120.0, 47.0])

    error, weight, _ = model.forward(i, j, x)
    grads = model.backward(i, j, error, weight)

    w = model.W.clone().requires_grad_(True)
    wc = model.W_context.clone().requires_grad_(True)
    b = model.b.clone().requires_grad_(True)
    bc = model.b_context.clone().requires_grad_(True)
    err = (w[i] * wc[j]).sum(1) + b[i] + bc[j] - torch.log(x)
    (weight * err * err).sum().backward()

    ref_w = torch.zeros_like(model.W).index_add_(0, i, grads.d_w)
    ref_wc = torch.zeros_like(model.W_context).index_add_(0, j, grads.d_w_context)
    ref_b = torch.zeros_like(model.b).index_add_(0, i, grads.d_b)
    ref_bc = torch.zeros_like(model.b_context).index_add_(0, j, grads.d_b_context)

    assert torch.allclose(ref_w, w.grad, atol=1e-5)
    assert torch.allclose(ref_wc, wc.grad, atol=1e-5)
    assert torch.allclose(ref_b, b.grad, atol=1e-5)
    assert torch.allclose(ref_bc, bc.grad, atol=1e-5)


def test_adagrad_updates_parameters_and_accumulators():
    model = GloVeModel(vocab_size=4, embedding_dim=3, seed=2)
    before_w = model.W.clone()
    before_acc = model.grad_sq_W.clone()
    i = torch.tensor([0, 1]); j = torch.tensor([2, 3]); x = torch.tensor([5.0, 9.0])
    model.train_batch(i, j, x, learning_rate=0.05)
    assert not torch.allclose(before_w[i], model.W[i])
    assert (model.grad_sq_W[i] > before_acc[i]).any()
    assert torch.allclose(before_w[3], model.W[3])  # untouched row unchanged


def test_tiny_synthetic_training_is_finite_and_decreases():
    """Phase-5 sanity: loss finite, decreasing, no NaN on a tiny corpus."""
    tokens = tokenize("king man queen woman " * 4)
    vocab = build_vocabulary(tokens, min_frequency=1, max_vocab_size=10,
                             verbose=False)
    matrix = build_from_text(tokens, vocab.word_to_idx, window_size=2)
    model = GloVeModel(vocab_size=len(vocab), embedding_dim=8, seed=42)

    i = torch.from_numpy(matrix.rows.astype(np.int64))
    j = torch.from_numpy(matrix.cols.astype(np.int64))
    x = torch.from_numpy(matrix.values.astype(np.float32))

    losses = []
    for _ in range(25):
        losses.append(model.train_batch(i, j, x, 0.05) / len(x))

    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < losses[0]
    assert torch.isfinite(model.embeddings()).all()


def test_checkpoint_roundtrip():
    model = GloVeModel(vocab_size=6, embedding_dim=4, seed=3)
    model.train_batch(torch.tensor([0]), torch.tensor([1]),
                      torch.tensor([7.0]), 0.05)
    state = model.state_dict()
    restored = GloVeModel(vocab_size=6, embedding_dim=4, seed=99)
    restored.load_state_dict(state)
    assert torch.allclose(model.W, restored.W)
    assert torch.allclose(model.grad_sq_W, restored.grad_sq_W)


# ---------------------------------------------------------------- evaluation

def test_cosine_similarity_known_values():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    sims = cosine_similarity(a, b)
    assert math.isclose(float(sims[0]), 1.0, abs_tol=1e-6)
    assert math.isclose(float(sims[1]), 0.0, abs_tol=1e-6)
    assert math.isclose(float(sims[2]), -1.0, abs_tol=1e-6)


def test_nearest_neighbors_excludes_query():
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    index = EmbeddingIndex(vectors, ["a", "b", "c"], name="toy")
    neighbors = index.nearest_neighbors("a", top_k=2)
    assert [w for w, _ in neighbors] == ["b", "c"]


def test_analogy_returns_valid_word_and_excludes_sources():
    vectors = np.array([
        [1.0, 1.0],   # king
        [1.0, 0.0],   # man
        [0.0, 0.0],   # woman
        [0.0, 1.0],   # queen
    ], dtype=np.float32)
    index = EmbeddingIndex(vectors, ["king", "man", "woman", "queen"], name="toy")
    (predicted, score), = index.analogy("king", "man", "woman", top_k=1)
    assert predicted in {"queen"}
    assert predicted not in {"king", "man", "woman"}
    assert -1.0 <= score <= 1.0


if __name__ == "__main__":
    failures = 0
    for name, func in sorted(list(globals().items())):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            except Exception as exc:  # pragma: no cover - surfacing real errors
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
