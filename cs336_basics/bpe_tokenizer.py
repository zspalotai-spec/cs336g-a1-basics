from collections import Counter
from datetime import datetime
import multiprocessing as mp
import regex as re
from typing import Iterable, Iterator, Self, TypeAlias

from cs336_basics import common_types
from cs336_basics import pretokenization_example
from cs336_basics import token_utils


PAT = re.compile(
    rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:

    Pair: TypeAlias = tuple[bytes, bytes]
    Word: TypeAlias = tuple[bytes, ...]

    vocab: dict[int, bytes]
    merges: list[Pair]
    merges_dict = dict[Pair, int]
    merge2joined = dict[Pair, bytes]
    special_tokens: list[str] | None
    special_tokens_bytes: set[bytes]
    word2token: dict[bytes, int]

    def __init__(
        self,
        vocab: common_types.Vocab,
        merges: common_types.MergeList,
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab.items())
        self.merges = list(merges)
        self.merges_dict = {p: idx for idx, p in enumerate(self.merges)}
        self.merge2joined = {p: b"".join(p) for p in self.merges}
        self.special_tokens = special_tokens
        if special_tokens:
            self.special_tokens_bytes = {sp.encode("utf-8") for sp in special_tokens}
        else:
            self.special_tokens_bytes = set()
        self.word2token = {}
        for token, word in self.vocab.items():
            self.word2token[word] = token

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
        text = text.encode("utf-8")
        parts = self.split_by_special_tokens(
            text, self.special_tokens, keep_specials=True
        )
        tokens = []
        for part in parts:
            if not part:
                continue
            token_maybe = self.word2token.get(part,None)
            if token_maybe is not None:
                tokens.append(token_maybe)
                continue
            for w in self.pre_tokenize_encode(part):
                #print("w ", w)
                if not w:
                    continue
                token_maybe = self.word2token.get(w,None)
                if token_maybe is not None:
                    tokens.append(token_maybe)
                    continue
                ww = tuple([bytes([b]) for b in w])
                merge_idxs: dict[int, int] = {}
                for p in zip(ww[:-1], ww[1:]):
                    merge_idx = self.merges_dict.get(p, None)
                    if merge_idx is not None:
                        merge_idxs[merge_idx] = merge_idxs.get(merge_idx, 0) + 1

                while merge_idxs:
                    #print("ww ", ww)
                    #print("merge_idxs ", merge_idxs)
                    smallest_index = min(merge_idxs.keys())
                    pair = self.merges[smallest_index]
                    joined_pair = self.merge2joined[pair]
                    del merge_idxs[smallest_index]
                    next_ww, merged_indices_old, merged_indices_new = self.get_merged_word(ww, pair, joined_pair)
                    affected_old_first_indices = set()
                    old_word_len = len(ww)
                    for idx in merged_indices_old:
                        if idx > 0:
                            affected_old_first_indices.add(idx - 1)
                        if idx < old_word_len - 2:
                            affected_old_first_indices.add(idx + 1)
                    for old_first_index in affected_old_first_indices:
                        orig_pair = (ww[old_first_index], ww[old_first_index + 1])
                        if orig_pair == pair:
                            continue
                        old_merge_idx = self.merges_dict.get(orig_pair,None)
                        if old_merge_idx is None:
                            continue
                        orig_cnt = merge_idxs[old_merge_idx]
                        if orig_cnt == 1:
                            del merge_idxs[old_merge_idx]
                        else:
                            merge_idxs[old_merge_idx] = orig_cnt-1
                    affected_new_first_indices = set()
                    new_word_len = len(next_ww)
                    for idx in merged_indices_new:
                        if idx > 0:
                            affected_new_first_indices.add(idx - 1)
                        if idx < new_word_len - 1:
                            affected_new_first_indices.add(idx)
                    for new_first_index in affected_new_first_indices:
                        new_pair = (next_ww[new_first_index], next_ww[new_first_index + 1])
                        new_merge_idx = self.merges_dict.get(new_pair,None)
                        if new_merge_idx is None:
                            continue
                        merge_idxs[new_merge_idx] = merge_idxs.get(new_merge_idx,0)+1
                    ww = next_ww
                #print("ww ", ww)
                tokens.extend(map(self.word2token.get, ww))
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            for e in self.encode(s):
                yield e

    def decode(self, ids: list[int]) -> str:
        # bytes_list = list(map(self.vocab.get, ids))
        # return b"".join(bytes_list).decode(encoding='utf-8', errors='replace')
        def iterable_source(ids: Iterable[int]) -> Iterator[int]:
            for e in ids:
                for b in self.vocab[e]:
                    yield b

        bs = bytes(iterable_source(ids))
        return bs.decode(encoding="utf-8", errors="replace")

    @classmethod
    def split_by_special_tokens(
        cls, text: bytes, special_tokens: list[str], keep_specials=False
    ) -> list[bytes]:
        if not special_tokens:
            return [text]
        special_tokens = sorted(special_tokens, reverse=True)
        if keep_specials:
            split_pattern = (
                b"("
                + b"|".join([re.escape(st.encode("utf-8")) for st in special_tokens])
                + b")"
            )
        else:
            split_pattern = b"|".join(
                [re.escape(st.encode("utf-8")) for st in special_tokens]
            )
        return re.split(split_pattern, text)

    @classmethod
    def pre_tokenize_encode(cls, text: bytes) -> Iterator[bytes]:
        for m in re.finditer(PAT, text):
            yield m.group(0)

    @classmethod
    def pre_tokenize(cls, texts: Iterable[bytes]) -> Iterator[tuple[bytes, ...]]:
        for t in texts:
            for m in re.finditer(PAT, t):
                yield tuple([bytes([b]) for b in m.group(0)])

    @classmethod
    def get_pair_counts(
        cls, word_counts: dict[int, int], idx2word: dict[int, Word]
    ) -> tuple[dict[Pair, int], dict[Pair, dict[int, int]]]:
        pair_counts: dict[Pair, int] = {}
        pair_to_words: dict[Pair, dict[int, int]] = {}
        for w_idx, cnt in word_counts.items():
            w = idx2word[w_idx]
            for pair in zip(w[:-1], w[1:]):
                old_cnt = pair_counts.get(pair, 0)
                pair_counts[pair] = old_cnt + cnt
                if old_cnt == 0:
                    pair_to_words[pair] = {w_idx: 1}
                else:
                    pair_to_words[pair][w_idx] = pair_to_words[pair].get(w_idx, 0) + 1
        return pair_counts, pair_to_words

    @classmethod
    def get_max_pair(cls, count_pairs: dict[int, set[Pair]]) -> Pair:
        return max(count_pairs[max(count_pairs.keys())])

    @classmethod
    def get_count_pairs(cls, pair_counts: dict[Pair, int]) -> dict[int, set[Pair]]:
        ret = {}
        for k, v in pair_counts.items():
            ret.setdefault(v, set()).add(k)
        return ret

    @classmethod
    def remove_from_count_pairs(
        cls, cnt: int, pair: Pair, count_pairs: dict[int, set[Pair]]
    ) -> None:
        pairs = count_pairs[cnt]
        pairs.remove(pair)
        if not pairs:
            del count_pairs[cnt]

    @classmethod
    def get_merged_word(
        cls, word: Word, pair: Pair, joined_pair: bytes
    ) -> tuple[Word, list[int], list[int]]:
        idx = 0
        current_idx = 0
        word_len = len(word)
        current_w = [None] * word_len
        merged_indices_old = []
        merged_indices_new = []
        while idx < word_len - 1:
            if (word[idx], word[idx + 1]) == pair:
                current_w[current_idx] = joined_pair
                merged_indices_old.append(idx)
                merged_indices_new.append(current_idx)
                idx += 2
            else:
                current_w[current_idx] = word[idx]
                idx += 1
            current_idx += 1
        if idx < word_len:
            current_w[current_idx] = word[idx]
            current_idx += 1
        return tuple(current_w[:current_idx]), merged_indices_old, merged_indices_new

    @classmethod
    def update_pair_counts_for_word(
        cls,
        word_idx: int,
        old_word: Word,
        word_count: int,
        merged_indices_old: list[int],
        pair: Pair,
        new_word: Word,
        merged_indices_new: list[int],
        pair_counts: dict[Pair, int],
        pair_to_words: dict[Pair, dict[int, int]],
        updated_pairs: dict[Pair, int],
    ) -> None:
        affected_old_first_indices = set()
        old_word_len = len(old_word)
        for idx in merged_indices_old:
            if idx > 0:
                affected_old_first_indices.add(idx - 1)
            if idx < old_word_len - 2:
                affected_old_first_indices.add(idx + 1)
        for old_first_index in affected_old_first_indices:
            orig_pair = (old_word[old_first_index], old_word[old_first_index + 1])
            if orig_pair == pair:
                continue
            orig_cnt = pair_counts[orig_pair]
            new_cnt = orig_cnt - word_count
            if orig_pair not in updated_pairs:
                updated_pairs[orig_pair] = orig_cnt
            if new_cnt == 0:
                del pair_counts[orig_pair]
                del pair_to_words[orig_pair]
            else:
                pair_counts[orig_pair] = new_cnt
                pair_to_words[orig_pair][word_idx] -= 1
        affected_new_first_indices = set()
        new_word_len = len(new_word)
        for idx in merged_indices_new:
            if idx > 0:
                affected_new_first_indices.add(idx - 1)
            if idx < new_word_len - 1:
                affected_new_first_indices.add(idx)
        for new_first_index in affected_new_first_indices:
            new_pair = (new_word[new_first_index], new_word[new_first_index + 1])
            orig_cnt = pair_counts.get(new_pair, 0)
            pair_counts[new_pair] = orig_cnt + word_count
            if orig_cnt == 0:
                pair_to_words[new_pair] = {word_idx: 1}
            else:
                pair_to_words[new_pair][word_idx] = (
                    pair_to_words[new_pair].get(word_idx, 0) + 1
                )
            if new_pair not in updated_pairs:
                updated_pairs[new_pair] = orig_cnt

    @classmethod
    def update_all_for_pair(
        cls,
        pair: Pair,
        joined_pair: bytes,
        word_counts: dict[int, int],
        idx2word: dict[int, Word],
        pair_counts: dict[Pair, int],
        pair_to_words: dict[Pair, dict[int, int]],
        count_pairs: dict[int, set[Pair]],
    ) -> None:
        cnt = pair_counts[pair]
        del pair_counts[pair]
        cls.remove_from_count_pairs(cnt, pair, count_pairs)
        updated_pairs = {}
        for w_idx in pair_to_words[pair]:
            w = idx2word[w_idx]
            current_w, merged_indices_old, merged_indices_new = cls.get_merged_word(
                w, pair, joined_pair
            )
            idx2word[w_idx] = current_w

            word_count = word_counts[w_idx]
            cls.update_pair_counts_for_word(
                w_idx,
                w,
                word_count,
                merged_indices_old,
                pair,
                current_w,
                merged_indices_new,
                pair_counts,
                pair_to_words,
                updated_pairs,
            )
        del pair_to_words[pair]
        for updated_pair, orig_cnt in updated_pairs.items():
            if orig_cnt > 0:
                cls.remove_from_count_pairs(orig_cnt, updated_pair, count_pairs)
            new_cnt = pair_counts.get(updated_pair, 0)
            if new_cnt > 0:
                count_pairs.setdefault(new_cnt, set()).add(updated_pair)

    @classmethod
    def get_indexed_word_counts(
        cls, old_word_counts: dict[Word, int]
    ) -> tuple[dict[int, int], dict[int, Word]]:
        idx2word: dict[int, Word] = {}
        word_counts: dict[int, int] = {}
        for idx, (w, cnt) in enumerate(old_word_counts.items()):
            idx2word[idx] = w
            word_counts[idx] = cnt
        return word_counts, idx2word

    @classmethod
    def get_word_counts(cls, words: Iterable[Word]) -> dict[Word, int]:
        word_counts = dict(Counter(words))
        return word_counts

    @classmethod
    def partial_word_counts(cls, path, start, end, special_tokens):
        with open(path, "rb") as f:
            f.seek(start)
            text = f.read(end - start)
        parts = cls.split_by_special_tokens(text, special_tokens, keep_specials=False)
        words = cls.pre_tokenize(parts)
        word_counts = cls.get_word_counts(words)
        return word_counts

    @classmethod
    def train_loop(
        cls,
        vocab_size: int,
        word_counts: dict[int, int],
        vocab: dict[int, bytes],
        idx2word: dict[int, Word],
    ) -> list[Pair]:
        pair_counts, pair_to_words = cls.get_pair_counts(word_counts, idx2word)
        count_pairs = cls.get_count_pairs(pair_counts)
        vocab_len = len(vocab)
        merges: list[Pair] = [None] * (vocab_size - vocab_len)
        merge_idx = 0
        while vocab_len < vocab_size:
            max_pair = cls.get_max_pair(count_pairs)
            merges[merge_idx] = max_pair
            merge_idx += 1
            joined_pair = b"".join(max_pair)
            vocab[vocab_len] = joined_pair
            vocab_len += 1
            cls.update_all_for_pair(
                max_pair,
                joined_pair,
                word_counts,
                idx2word,
                pair_counts,
                pair_to_words,
                count_pairs,
            )
        return merges

    @classmethod
    def train(
        cls,
        path: str,
        vocab_size: int,
        special_tokens: list[str],
        num_processes=10,
        num_chunks=10,
    ) -> Self:
        times = []
        times.append(datetime.now())
        vocab = {idx: st.encode("utf-8") for idx, st in enumerate(special_tokens)}
        vocab.update({idx + len(special_tokens): bytes([idx]) for idx in range(256)})

        times.append(datetime.now())
        with open(path, "rb") as f:
            boundaries = pretokenization_example.find_chunk_boundaries(
                f, num_chunks, b"<|endoftext|>"
            )
        with mp.Pool(num_processes) as p:
            word_counts_list = p.starmap(
                cls.partial_word_counts,
                [
                    (path, start, end, special_tokens)
                    for start, end in zip(boundaries[:-1], boundaries[1:])
                ],
            )
        times.append(datetime.now())
        old_word_counts = word_counts_list[0]
        for wc in word_counts_list[1:]:
            for k, v in wc.items():
                orig_cnt = old_word_counts.get(k, 0)
                old_word_counts[k] = orig_cnt + v
        word_counts, idx2word = cls.get_indexed_word_counts(old_word_counts)
        times.append(datetime.now())
        merges = cls.train_loop(vocab_size, word_counts, vocab, idx2word)
        times.append(datetime.now())
        return cls(vocab, merges, special_tokens), times
