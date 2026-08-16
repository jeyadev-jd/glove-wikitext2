"""GloVe model: parameters, weighting function, loss, analytic gradients, AdaGrad.

Objective (Pennington et al., 2014, eq. 8)::

    J = sum_{i,j : X_ij > 0} f(X_ij) * (w_i . w~_j + b_i + b~_j - log X_ij)^2

with the weighting function::

    f(x) = (x / x_max)^alpha   if x < x_max
    f(x) = 1                   otherwise

Gradients are derived and implemented by hand (see :meth:`GloVeModel.backward`);
autograd is deliberately not used, and neither is ``torch.optim``. PyTorch is
used only as a vectorised tensor/CUDA library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


def weighting_function(x: torch.Tensor, x_max: float = 100.0,
                       alpha: float = 0.75) -> torch.Tensor:
    """Vectorised GloVe weighting ``f(x)``, clamped to 1.0 at ``x >= x_max``."""
    return torch.clamp(x / x_max, max=1.0).pow(alpha)


def weighting_function_scalar(x: float, x_max: float = 100.0,
                              alpha: float = 0.75) -> float:
    """Scalar reference implementation of ``f(x)`` (used by the tests)."""
    if x < x_max:
        return (x / x_max) ** alpha
    return 1.0


@dataclass
class GloVeGradients:
    """Per-sample gradients for one mini-batch."""

    d_w: torch.Tensor            # (B, d)
    d_w_context: torch.Tensor    # (B, d)
    d_b: torch.Tensor            # (B,)
    d_b_context: torch.Tensor    # (B,)


class GloVeModel:
    """GloVe parameters with hand-written forward, backward and AdaGrad steps.

    Parameters:
        ``W`` (V, d), ``W_context`` (V, d), ``b`` (V,), ``b_context`` (V,).

    All tensors live on ``device``. For ``V = 20,000`` and ``d = 100`` the
    parameters plus AdaGrad accumulators occupy roughly 32 MB in float32, which
    is trivial for a 4 GB GPU; only mini-batches of co-occurrence triplets are
    streamed in.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 100,
                 device: Optional[torch.device] = None,
                 seed: int = 42, init_scale: float = 0.5,
                 x_max: float = 100.0, alpha: float = 0.75,
                 epsilon: float = 1e-8) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = device or torch.device("cpu")
        self.x_max = x_max
        self.alpha = alpha
        self.epsilon = epsilon

        generator = torch.Generator(device="cpu").manual_seed(seed)
        bound = init_scale / embedding_dim

        def _uniform(*shape: int) -> torch.Tensor:
            tensor = (torch.rand(*shape, generator=generator) * 2 - 1) * bound
            return tensor.to(device=self.device, dtype=torch.float32)

        # Trainable parameters, randomly initialised (never from pretrained).
        self.W = _uniform(vocab_size, embedding_dim)
        self.W_context = _uniform(vocab_size, embedding_dim)
        self.b = _uniform(vocab_size)
        self.b_context = _uniform(vocab_size)

        # AdaGrad squared-gradient accumulators, initialised to 1.0 as in the
        # reference GloVe implementation (avoids a huge first step).
        self.grad_sq_W = torch.ones_like(self.W)
        self.grad_sq_W_context = torch.ones_like(self.W_context)
        self.grad_sq_b = torch.ones_like(self.b)
        self.grad_sq_b_context = torch.ones_like(self.b_context)

        # Reusable per-batch gradient buffers (avoids reallocating V x d
        # tensors on every mini-batch).
        self._buf_w = torch.zeros_like(self.W)
        self._buf_wc = torch.zeros_like(self.W_context)
        self._buf_b = torch.zeros_like(self.b)
        self._buf_bc = torch.zeros_like(self.b_context)

    # ------------------------------------------------------------------ math

    def forward(self, i: torch.Tensor, j: torch.Tensor,
                x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-sample error, weight and loss for a batch.

        Args:
            i: (B,) int64 target-word indices.
            j: (B,) int64 context-word indices.
            x: (B,) float32 co-occurrence counts (all > 0).

        Returns:
            ``(error, weight, loss)`` where
            ``error = w_i . w~_j + b_i + b~_j - log(X_ij)``,
            ``weight = f(X_ij)`` and ``loss = weight * error^2`` (per sample).
        """
        w_i = self.W[i]                     # (B, d)
        w_j = self.W_context[j]             # (B, d)
        dot = (w_i * w_j).sum(dim=1)        # (B,)
        error = dot + self.b[i] + self.b_context[j] - torch.log(x)
        weight = weighting_function(x, self.x_max, self.alpha)
        loss = weight * error * error
        return error, weight, loss

    def backward(self, i: torch.Tensor, j: torch.Tensor,
                 error: torch.Tensor, weight: torch.Tensor) -> GloVeGradients:
        """Analytic gradients of ``weight * error^2`` w.r.t. the parameters.

        d/dw_i      = 2 * f(X_ij) * e_ij * w~_j
        d/dw~_j     = 2 * f(X_ij) * e_ij * w_i
        d/db_i      = 2 * f(X_ij) * e_ij
        d/db~_j     = 2 * f(X_ij) * e_ij
        """
        scale = (2.0 * weight * error).unsqueeze(1)   # (B, 1)
        d_w = scale * self.W_context[j]
        d_w_context = scale * self.W[i]
        d_bias = scale.squeeze(1)
        return GloVeGradients(d_w, d_w_context, d_bias, d_bias)

    def adagrad_step(self, i: torch.Tensor, j: torch.Tensor,
                     grads: GloVeGradients, learning_rate: float) -> None:
        """Apply one manual AdaGrad update per parameter row touched by the batch.

        Per-sample gradients are first summed into a per-row batch gradient
        (a frequent word such as "the" appears in many samples of the same
        batch), then a single AdaGrad update is applied::

            accumulator_row += g_row^2
            param_row       -= lr * g_row / sqrt(accumulator_row + eps)

        Summing before updating matters: applying one raw AdaGrad step per
        occurrence lets a high-frequency row move hundreds of times inside a
        single batch, which was observed to diverge to NaN in float32.
        """
        grad_w = self._buf_w.zero_().index_add_(0, i, grads.d_w)
        grad_wc = self._buf_wc.zero_().index_add_(0, j, grads.d_w_context)
        grad_b = self._buf_b.zero_().index_add_(0, i, grads.d_b)
        grad_bc = self._buf_bc.zero_().index_add_(0, j, grads.d_b_context)

        for param, accumulator, gradient in (
            (self.W, self.grad_sq_W, grad_w),
            (self.W_context, self.grad_sq_W_context, grad_wc),
            (self.b, self.grad_sq_b, grad_b),
            (self.b_context, self.grad_sq_b_context, grad_bc),
        ):
            accumulator.add_(gradient * gradient)
            param.sub_(learning_rate * gradient /
                       torch.sqrt(accumulator + self.epsilon))

    def train_batch(self, i: torch.Tensor, j: torch.Tensor, x: torch.Tensor,
                    learning_rate: float) -> float:
        """Full forward/backward/update cycle for one batch; returns summed loss."""
        error, weight, loss = self.forward(i, j, x)
        grads = self.backward(i, j, error, weight)
        self.adagrad_step(i, j, grads, learning_rate)
        return float(loss.sum().item())

    # ------------------------------------------------------------- interface

    def embeddings(self) -> torch.Tensor:
        """Final word vectors ``W + W_context`` as prescribed by the paper."""
        return (self.W + self.W_context).detach().cpu()

    def to(self, device: torch.device) -> "GloVeModel":
        """Move all parameters and accumulators to ``device`` in place."""
        self.device = device
        for name in ("W", "W_context", "b", "b_context",
                     "grad_sq_W", "grad_sq_W_context",
                     "grad_sq_b", "grad_sq_b_context",
                     "_buf_w", "_buf_wc", "_buf_b", "_buf_bc"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def state_dict(self) -> Dict[str, Any]:
        """Serialisable snapshot of parameters, accumulators and settings."""
        return {
            "W": self.W.detach().cpu(),
            "W_context": self.W_context.detach().cpu(),
            "b": self.b.detach().cpu(),
            "b_context": self.b_context.detach().cpu(),
            "grad_sq_W": self.grad_sq_W.detach().cpu(),
            "grad_sq_W_context": self.grad_sq_W_context.detach().cpu(),
            "grad_sq_b": self.grad_sq_b.detach().cpu(),
            "grad_sq_b_context": self.grad_sq_b_context.detach().cpu(),
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "x_max": self.x_max,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "GloVeModel":
        """Restore a snapshot produced by :meth:`state_dict`."""
        if state["vocab_size"] != self.vocab_size:
            raise ValueError("checkpoint vocab_size does not match model")
        if state["embedding_dim"] != self.embedding_dim:
            raise ValueError("checkpoint embedding_dim does not match model")
        for name in ("W", "W_context", "b", "b_context",
                     "grad_sq_W", "grad_sq_W_context",
                     "grad_sq_b", "grad_sq_b_context"):
            setattr(self, name, state[name].to(self.device))
        self.x_max = state["x_max"]
        self.alpha = state["alpha"]
        self.epsilon = state["epsilon"]
        return self
