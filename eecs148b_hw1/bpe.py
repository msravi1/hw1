"""Byte-level BPE tokenizer — training, encoding, and decoding."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import regex

# GPT-2 style pre-tokenisation regex (from tiktoken)
# Uses the `regex` module for Unicode property (\p{L}, \p{N}) support.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Returns
    -------
    vocab : dict[int, bytes]
        Mapping from token-id to byte-string.
    merges : list[tuple[bytes, bytes]]
        Ordered list of merge operations performed during training.
    """
    if special_tokens is None:
        special_tokens = []

    # --- 1. read corpus ------------------------------------------------
    text = Path(input_path).read_text(encoding="utf-8")

    # --- 2. strip special tokens and pre-tokenise ----------------------
    # Split on special tokens so that no merge crosses their boundary.
    if special_tokens:
        split_pat = "|".join(re.escape(t) for t in special_tokens)
        chunks = re.split(split_pat, text)
    else:
        chunks = [text]

    # Build word→count mapping (word = tuple of single-byte bytes)
    word_counts: dict[tuple[bytes, ...], int] = defaultdict(int)
    for chunk in chunks:
        for m in regex.finditer(PAT, chunk):
            word = tuple(bytes([b]) for b in m.group().encode("utf-8"))
            word_counts[word] += 1

    # --- 3. initial vocabulary -----------------------------------------
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    # Reserve IDs for special tokens (placed right after byte vocab)
    special_token_ids: dict[str, int] = {}
    for st in special_tokens:
        special_token_ids[st] = next_id
        vocab[next_id] = st.encode("utf-8")
        next_id += 1

    num_merges = vocab_size - next_id  # how many BPE merges to learn

    # --- 4. pair counting helpers --------------------------------------
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    # pair → set of words that contain the pair (for fast update)
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    def _count_pairs_in_word(word: tuple[bytes, ...], count: int):
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += count
            pair_to_words[pair].add(word)

    for word, count in word_counts.items():
        _count_pairs_in_word(word, count)

    merges: list[tuple[bytes, bytes]] = []

    # --- 5. merge loop -------------------------------------------------
    for _ in range(num_merges):
        if not pair_counts:
            break

        # Find most-frequent pair (break ties lexicographically greatest)
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        if pair_counts[best_pair] < 1:
            break

        a, b = best_pair
        merged = a + b
        merges.append(best_pair)
        vocab[next_id] = merged
        next_id += 1

        # Update words that contain the best pair
        affected_words = list(pair_to_words.get(best_pair, []))
        for word in affected_words:
            if word not in word_counts:
                continue
            count = word_counts[word]
            if count == 0:
                continue

            # Build new word by merging every occurrence of the pair
            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_t = tuple(new_word)

            if new_word_t == word:
                continue

            # Remove old pair counts
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
                pair_to_words[pair].discard(word)

            # Remove old word
            del word_counts[word]

            # Add new word
            word_counts[new_word_t] = word_counts.get(new_word_t, 0) + count

            # Add new pair counts
            for i in range(len(new_word_t) - 1):
                pair = (new_word_t[i], new_word_t[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count
                pair_to_words[pair].add(new_word_t)

        # Clean up best_pair from pair_counts (should already be 0 or gone)
        pair_counts.pop(best_pair, None)
        pair_to_words.pop(best_pair, None)

    return vocab, merges


# ---------------------------------------------------------------------------
# Tokenizer class
# ---------------------------------------------------------------------------

class Tokenizer:
    """Byte-level BPE tokenizer (encode / decode)."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)  # id → bytes
        self.merges = list(merges)

        # Add special tokens if not present
        self.special_tokens: list[str] = []
        if special_tokens:
            self.special_tokens = list(special_tokens)
            existing_bytes = set(self.vocab.values())
            next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
            for st in special_tokens:
                st_bytes = st.encode("utf-8")
                if st_bytes not in existing_bytes:
                    self.vocab[next_id] = st_bytes
                    next_id += 1

        # Reverse mapping bytes → id
        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # Build ordered merge dict for fast lookup: (a,b) → rank
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

    # ---- serialisation helpers ----------------------------------------

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        # vocab: JSON  {id_str: hex_or_list_of_ints}
        with open(vocab_filepath, "r") as f:
            raw_vocab = json.load(f)
        vocab: dict[int, bytes] = {}
        for k, v in raw_vocab.items():
            vocab[int(k)] = bytes(v)

        with open(merges_filepath, "r") as f:
            raw_merges = json.load(f)
        merges = [(bytes(a), bytes(b)) for a, b in raw_merges]

        return cls(vocab, merges, special_tokens)

    # ---- encoding -----------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        ids: list[int] = []

        # Handle special tokens: split text on special tokens, encode
        # each chunk independently, insert special-token IDs in order.
        if self.special_tokens:
            split_pat = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
            parts = re.split(f"({split_pat})", text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                ids.append(self.bytes_to_id[part.encode("utf-8")])
                continue
            if not part:
                continue
            # Pre-tokenise
            for m in regex.finditer(PAT, part):
                word = m.group().encode("utf-8")
                token_list = [bytes([b]) for b in word]
                token_list = self._apply_merges(token_list)
                for tok in token_list:
                    ids.append(self.bytes_to_id[tok])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily yield token IDs from an iterable of strings."""
        for text in iterable:
            yield from self.encode(text)

    def _apply_merges(self, tokens: list[bytes]) -> list[bytes]:
        """Apply BPE merges in priority order."""
        while len(tokens) >= 2:
            # Find the pair with the lowest merge rank
            best_pair = None
            best_rank = len(self.merges)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            # Merge all occurrences of best_pair
            a, b = best_pair
            merged = a + b
            new_tokens: list[bytes] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    # ---- decoding -----------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string."""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")
