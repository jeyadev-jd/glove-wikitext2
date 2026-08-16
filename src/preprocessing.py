"""Tokenisation and vocabulary construction.

Tokeniser choice
----------------
A regex tokeniser (``[a-z]+(?:'[a-z]+)?|[0-9]+(?:[.,][0-9]+)*``) is used rather
than a heavyweight NLP pipeline because:

* WikiText-2 raw text is already sentence-segmented and lightly normalised, so
  the extra machinery of spaCy/NLTK buys nothing for a co-occurrence model;
* it has no external model download, keeping the project reproducible offline
  after the corpus is cached;
* it drops standalone punctuation (which carries no distributional meaning for
  GloVe) while keeping intra-word apostrophes ("don't" -> "don't") and numbers
  as single tokens.

Punctuation is therefore removed rather than kept as tokens; a comma between
two words does not consume a context slot, which matters for the default
``window_size = 1``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+(?:[.,][0-9]+)*")

# WikiText's rare-word placeholder and the section markers are not real words.
_CORPUS_ARTEFACTS = {"unk"}


def tokenize(text: str) -> List[str]:
    """Lowercase ``text`` and split it into word/number tokens."""
    if not isinstance(text, str):
        raise TypeError(f"tokenize expects str, got {type(text).__name__}")
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [tok for tok in tokens if tok not in _CORPUS_ARTEFACTS]


@dataclass
class Vocabulary:
    """Word/index mappings plus corpus statistics."""

    word_to_idx: Dict[str, int]
    idx_to_word: List[str]
    word_counts: Dict[str, int]
    stats: Dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.idx_to_word)

    def __contains__(self, word: str) -> bool:
        return word in self.word_to_idx

    def index(self, word: str) -> int:
        """Index of ``word``.

        Raises:
            KeyError: if the word is out of vocabulary.
        """
        return self.word_to_idx[word]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        """Map tokens to indices, dropping out-of-vocabulary tokens.

        Rare tokens are *removed* (per the assignment), not mapped to <UNK>.
        """
        lookup = self.word_to_idx
        return [lookup[tok] for tok in tokens if tok in lookup]


def build_vocabulary(tokens: Sequence[str], min_frequency: int = 5,
                     max_vocab_size: int = 20000,
                     verbose: bool = True) -> Vocabulary:
    """Build a frequency-filtered, size-capped vocabulary from ``tokens``.

    Words with a corpus frequency below ``min_frequency`` are removed. From the
    survivors the ``max_vocab_size`` most frequent words are kept. Ties are
    broken alphabetically so the vocabulary is deterministic.
    """
    if min_frequency < 1:
        raise ValueError("min_frequency must be >= 1")
    if max_vocab_size < 1:
        raise ValueError("max_vocab_size must be >= 1")

    raw_counts = Counter(tokens)
    frequent = {w: c for w, c in raw_counts.items() if c >= min_frequency}
    ordered = sorted(frequent.items(), key=lambda kv: (-kv[1], kv[0]))[:max_vocab_size]

    idx_to_word = [word for word, _ in ordered]
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}
    word_counts = {word: raw_counts[word] for word in idx_to_word}

    raw_token_count = len(tokens)
    kept_token_count = sum(word_counts.values())
    stats = {
        "raw_token_count": raw_token_count,
        "unique_raw_tokens": len(raw_counts),
        "vocab_size": len(idx_to_word),
        "filtered_token_count": kept_token_count,
        "removed_rare_types": len(raw_counts) - len(idx_to_word),
        "removed_token_count": raw_token_count - kept_token_count,
        "percent_tokens_retained": (
            100.0 * kept_token_count / raw_token_count if raw_token_count else 0.0
        ),
        "min_frequency": min_frequency,
        "max_vocab_size": max_vocab_size,
    }

    if verbose:
        print(f"Raw token count:            {stats['raw_token_count']:,}")
        print(f"Filtered token count:       {stats['filtered_token_count']:,}")
        print(f"Unique raw tokens:          {stats['unique_raw_tokens']:,}")
        print(f"Vocabulary size:            {stats['vocab_size']:,}")
        print(f"Removed rare token types:   {stats['removed_rare_types']:,}")
        print(f"Percentage tokens retained: {stats['percent_tokens_retained']:.2f}%")

    return Vocabulary(word_to_idx, idx_to_word, word_counts, stats)
