"""Central configuration for the GloVe-from-scratch project.

Every hyperparameter and path used by the training/evaluation pipeline lives
here so that experiments can vary a single knob without touching the
implementation modules.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PATHS: Dict[str, str] = {
    "root": PROJECT_ROOT,
    "data": os.path.join(PROJECT_ROOT, "data"),
    "results": os.path.join(PROJECT_ROOT, "results"),
    "checkpoints": os.path.join(PROJECT_ROOT, "checkpoints"),
    "vectors": os.path.join(PROJECT_ROOT, "vectors"),
    "report": os.path.join(PROJECT_ROOT, "report"),
}

# Baseline configuration (assignment defaults).
CONFIG: Dict[str, Any] = {
    "seed": 42,
    "max_vocab_size": 20000,
    "min_frequency": 5,
    "embedding_dim": 100,
    "window_size": 1,
    "x_max": 100.0,
    "alpha": 0.75,
    "learning_rate": 0.05,
    "epochs": 30,
    "batch_size": 4096,
    "epsilon": 1e-8,
    # Engineering knobs.
    "device": "auto",             # "auto" | "cuda" | "cpu"
    "checkpoint_every": 5,        # epochs; 0 disables checkpointing
    "init_scale": 0.5,            # uniform init in [-scale/dim, +scale/dim]
    "log_every_epoch": True,
    "min_batch_size": 512,        # OOM fallback floor
}

# Pipeline-level switches (expensive stages default to off where required).
RUN_ABLATIONS = True
RUN_CPU_BENCHMARK = False

# Ablation grids: one parameter changes at a time, everything else = CONFIG.
ABLATION_GRID: Dict[str, list] = {
    "embedding_dim": [50, 100, 200],
    "window_size": [1, 2, 5],
    "min_frequency": [1, 5, 10],
    "epochs": [10, 20, 30],
}

# Evaluation assets.
NEAREST_NEIGHBOR_QUERIES = ["king", "computer", "city", "government", "music"]

NEIGHBOR_FALLBACKS = [
    "war", "world", "film", "school", "state", "river", "album", "season",
    "team", "church", "party", "song", "german", "british", "american",
]

# Analogy quadruples are stored as (a, b, c, expected) and solved as
# ``a - b + c ~= expected`` (the classic "king - man + woman = queen" form).
ANALOGIES = [
    ("king", "man", "woman", "queen"),
    ("queen", "woman", "man", "king"),
    ("france", "paris", "london", "england"),
    ("italy", "rome", "madrid", "spain"),
    ("germany", "berlin", "paris", "france"),
    ("london", "england", "france", "paris"),
    ("bigger", "big", "small", "smaller"),
    ("better", "good", "bad", "worse"),
    ("walking", "walk", "run", "running"),
    ("french", "france", "germany", "german"),
    ("daughter", "son", "father", "mother"),
    ("sister", "brother", "boy", "girl"),
    ("days", "day", "year", "years"),
    ("his", "he", "she", "her"),
    ("played", "play", "win", "won"),
    ("two", "one", "three", "four"),
]

# Remote assets. Corpus and pretrained vectors are downloaded, never vendored.
WIKITEXT2_URLS = {
    "train": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
             "wikitext-2-raw-v1/train-00000-of-00001.parquet",
    "validation": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
                  "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
            "wikitext-2-raw-v1/test-00000-of-00001.parquet",
}

GLOVE_6B_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_6B_MEMBER = "glove.6B.100d.txt"


def get_config(**overrides: Any) -> Dict[str, Any]:
    """Return a copy of the baseline config with ``overrides`` applied.

    Raises:
        KeyError: if an override key is not a known configuration key.
    """
    cfg = copy.deepcopy(CONFIG)
    for key, value in overrides.items():
        if key not in cfg:
            raise KeyError(f"Unknown configuration key: {key!r}")
        cfg[key] = value
    return cfg


def ensure_dirs() -> None:
    """Create all project output directories if they do not exist."""
    for key in ("data", "results", "checkpoints", "vectors", "report"):
        os.makedirs(PATHS[key], exist_ok=True)
