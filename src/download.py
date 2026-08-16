"""Small streaming HTTP downloader with on-disk caching."""

from __future__ import annotations

import os
import shutil

import requests
from tqdm.auto import tqdm

_CHUNK = 1 << 20  # 1 MiB


def download_file(url: str, destination: str, force: bool = False,
                  description: str = "") -> str:
    """Stream ``url`` to ``destination`` unless a cached copy already exists.

    The download is written to a ``.part`` file and moved into place only on
    success, so an interrupted run never leaves a truncated cache entry.
    """
    if os.path.exists(destination) and not force:
        return destination

    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    partial = destination + ".part"
    label = description or os.path.basename(destination)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0)) or None
        with open(partial, "wb") as handle, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"download {label}"
        ) as bar:
            for chunk in response.iter_content(chunk_size=_CHUNK):
                if not chunk:
                    continue
                handle.write(chunk)
                bar.update(len(chunk))

    shutil.move(partial, destination)
    return destination
