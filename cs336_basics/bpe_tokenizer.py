import regex as re
from typing import Iterable, Iterator, Self, TypeAlias

from cs336_basics import common_types
from cs336_basics import token_utils
from cs336_basics import utils


PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:

    Pair: TypeAlias = tuple[bytes, bytes]
    Word: TypeAlias = tuple[bytes, ...]

    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None

    def __init__(
        self,
        vocab: common_types.Vocab,
        merges: common_types.MergeList,
        special_tokens=None,
    ):
        self.vocab = dict(vocab.items())
        self.merges = list(merges)
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Self:
        vocab, merges = token_utils.load_vocab_and_merges(
            vocab_filepath, merges_filepath
        )
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            for e in self.encode(s):
                yield e

    def decode(self, ids: list[int]) -> str:
        # bytes_list = list(map(self.params.vocab.get, indices))
        # return b"".join(bytes_list).decode(encoding='utf-8', errors='replace')
        def iterable_source(ids: Iterable[int]) -> Iterator[int]:
            for e in ids:
                for b in self.vocab[e]:
                    yield b

        bs = bytes(iterable_source(ids))
        return bs.decode(encoding="utf-8", errors="replace")

    @classmethod
    def pre_tokenize(
        cls, text: str, special_tokens: list[str]
    ) -> Iterator[tuple[bytes, ...]]:
        split_pattern = "|".join([re.escape(st) for st in special_tokens])
        parts = re.split(split_pattern, text)
        for p in parts:
            for m in re.finditer(PAT, p):
                yield tuple([bytes([b]) for b in m.group(0).encode("utf-8")])

    @classmethod
    def get_word_counts(cls, words: Iterable[Word]) -> dict[Word, int]:
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.setdefault(w, 0) + 1
        return word_counts

    @classmethod
    def get_pair_counts(
        cls, word_counts: dict[Word, int]
    ) -> tuple[dict[Pair, int], dict[Pair, set[Word]]]:
        pair_counts: dict[Pair, int] = {}
        pair_to_words: dict[Pair, set[Word]] = {}
        for w, cnt in word_counts.items():
            for pair in zip(w[:-1], w[1:]):
                pair_counts[pair] = pair_counts.setdefault(pair, 0) + cnt
                pair_to_words.setdefault(pair, set()).add(w)
        return pair_counts, pair_to_words

    @classmethod
    def get_max_pair(cls, pair_counts: dict[Pair, int]) -> Pair:
        max_pair = None
        max_count = 0
        for p, c in pair_counts.items():
            if c > max_count:
                max_pair = p
                max_count = c
            elif c == max_count and (max_pair is None or p > max_pair):
                max_pair = p
        return max_pair

    @classmethod
    def get_merged_word(cls, word: Word, pair: Pair, joined_pair: bytes) -> Word:
        current_w = []
        idx = 0
        while idx < len(word) - 1:
            if (word[idx], word[idx + 1]) == pair:
                current_w.append(joined_pair)
                idx += 2
            else:
                current_w.append(word[idx])
                idx += 1
        if idx < len(word):
            current_w.append(word[idx])
        return tuple(current_w)

    @classmethod
    def update_pair_counts_for_word(
        cls,
        old_word: Word,
        word_count: int,
        pair: Pair,
        new_word: Word,
        pair_counts: dict[Pair, int],
        pair_to_words: dict[Pair, set[Word]],
    ) -> None:
        for orig_pair in zip(old_word[:-1], old_word[1:]):
            if orig_pair == pair:
                continue
            pair_counts[orig_pair] = pair_counts[orig_pair] - word_count
            if pair_counts[orig_pair] == 0:
                del pair_counts[orig_pair]
            if old_word in pair_to_words.get(orig_pair, set()):
                pair_to_words[orig_pair].remove(old_word)
                if not pair_to_words[orig_pair]:
                    del pair_to_words[orig_pair]
        for new_pair in zip(new_word[:-1], new_word[1:]):
            pair_counts[new_pair] = pair_counts.setdefault(new_pair, 0) + word_count
            pair_to_words.setdefault(new_pair, set()).add(new_word)

    @classmethod
    def update_all_for_pair(
        cls,
        pair: Pair,
        joined_pair: bytes,
        word_counts: dict[Word, int],
        pair_counts: dict[Pair, int],
        pair_to_words: dict[Pair, set[Word]],
    ) -> None:
        del pair_counts[pair]
        for w in pair_to_words[pair]:
            word_count = word_counts[w]
            del word_counts[w]
            current_w = cls.get_merged_word(w, pair, joined_pair)
            word_counts[current_w] = word_count
            cls.update_pair_counts_for_word(
                w, word_count, pair, current_w, pair_counts, pair_to_words
            )
        del pair_to_words[pair]

    @classmethod
    def train(cls, path: str, vocab_size: int, special_tokens: list[str]) -> Self:
        vocab = {idx: st.encode("utf-8") for idx, st in enumerate(special_tokens)}
        vocab.update({idx + len(special_tokens): bytes([idx]) for idx in range(256)})
        merges = []

        with open(path, "rb") as f:
            text = f.read().decode("utf-8", errors="ignore")

        word_counts = cls.get_word_counts(cls.pre_tokenize(text, special_tokens))

        pair_counts, pair_to_words = cls.get_pair_counts(word_counts)

        while len(vocab) < vocab_size:
            max_pair = cls.get_max_pair(pair_counts)
            merges.append(max_pair)
            joined_pair = b"".join(max_pair)
            vocab[len(vocab)] = joined_pair
            cls.update_all_for_pair(
                max_pair, joined_pair, word_counts, pair_counts, pair_to_words
            )

        return cls(vocab, merges, special_tokens)
