"""Official GloVe (glove.6B.100d) loader — used for comparison only.

These vectors are downloaded after our own model has finished training and are
never used to initialise or fine-tune it.
"""

from __future__ import annotations

import os
import zipfile
from typing import List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

import config as project_config
from src.download import download_file
from src.evaluation import EmbeddingIndex


def download_glove_6b(data_dir: Optional[str] = None,
                      member: Optional[str] = None) -> str:
    """Download and extract ``glove.6B.100d.txt``; returns the extracted path.

    The 862 MB archive is cached in ``data/`` and excluded from the submission
    archive.
    """
    data_dir = data_dir or project_config.PATHS["data"]
    member = member or project_config.GLOVE_6B_MEMBER
    os.makedirs(data_dir, exist_ok=True)

    extracted = os.path.join(data_dir, member)
    if os.path.exists(extracted):
        return extracted

    archive = os.path.join(data_dir, "glove.6B.zip")
    download_file(project_config.GLOVE_6B_URL, archive, description="glove.6B.zip")
    with zipfile.ZipFile(archive) as zf:
        if member not in zf.namelist():
            raise FileNotFoundError(
                f"{member} not found in archive; members: {zf.namelist()}")
        zf.extract(member, path=data_dir)
    return extracted


def load_glove_txt(path: str, max_words: Optional[int] = None,
                   verbose: bool = True) -> Tuple[np.ndarray, List[str]]:
    """Parse a whitespace-delimited GloVe text file into (vectors, vocabulary)."""
    words: List[str] = []
    rows: List[np.ndarray] = []
    dim: Optional[int] = None

    with open(path, "r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="load glove.6B", disable=not verbose):
            parts = line.rstrip().split(" ")
            if len(parts) < 3:
                continue
            word, values = parts[0], parts[1:]
            if dim is None:
                dim = len(values)
            elif len(values) != dim:
                continue  # skip malformed lines (words containing spaces)
            words.append(word)
            rows.append(np.asarray(values, dtype=np.float32))
            if max_words is not None and len(words) >= max_words:
                break

    if not rows:
        raise ValueError(f"no vectors parsed from {path}")
    return np.vstack(rows), words


def load_pretrained_index(data_dir: Optional[str] = None,
                          max_words: Optional[int] = None,
                          verbose: bool = True) -> EmbeddingIndex:
    """Download (if needed) and wrap official GloVe vectors in an EmbeddingIndex."""
    path = download_glove_6b(data_dir=data_dir)
    vectors, words = load_glove_txt(path, max_words=max_words, verbose=verbose)
    return EmbeddingIndex(vectors, words, name="glove.6B.100d")
