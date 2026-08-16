"""WikiText-2 acquisition.

The corpus is downloaded programmatically from the Hugging Face mirror of the
Salesforce/wikitext dataset (raw parquet shards) and cached under ``data/``.
It is never vendored into the repository or the submission archive.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import pandas as pd

import config as project_config
from src.download import download_file


def _cache_path(split: str, data_dir: str) -> str:
    return os.path.join(data_dir, f"wikitext2_{split}.parquet")


def download_wikitext2(split: str = "train",
                       data_dir: Optional[str] = None,
                       force: bool = False) -> str:
    """Download one WikiText-2 (raw) split and return the local parquet path."""
    if split not in project_config.WIKITEXT2_URLS:
        raise ValueError(
            f"Unknown split {split!r}; expected one of "
            f"{sorted(project_config.WIKITEXT2_URLS)}"
        )
    data_dir = data_dir or project_config.PATHS["data"]
    os.makedirs(data_dir, exist_ok=True)
    destination = _cache_path(split, data_dir)
    return download_file(
        project_config.WIKITEXT2_URLS[split], destination, force=force,
        description=f"wikitext-2 {split}",
    )


def load_wikitext2_text(split: str = "train",
                        data_dir: Optional[str] = None,
                        force_download: bool = False) -> str:
    """Return one WikiText-2 split as a single raw text string."""
    path = download_wikitext2(split, data_dir=data_dir, force=force_download)
    frame = pd.read_parquet(path)
    if "text" not in frame.columns:
        raise ValueError(f"Unexpected parquet schema in {path}: {list(frame.columns)}")
    return "".join(frame["text"].astype(str).tolist())


def load_all_splits(data_dir: Optional[str] = None) -> Dict[str, str]:
    """Return {split: raw text} for train/validation/test."""
    return {
        split: load_wikitext2_text(split, data_dir=data_dir)
        for split in project_config.WIKITEXT2_URLS
    }
