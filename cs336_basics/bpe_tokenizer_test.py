import unittest

from cs336_basics import bpe_tokenizer


class TestBPETokenizer(unittest.TestCase):
    def test_split_by_special_tokens(self):
        res = list(
            bpe_tokenizer.BPETokenizer.split_by_special_tokens(
                b"test<|<|special|>ok<|special2|>", ["<|special|>", "<|special2|>"]
            )
        )
        assert res == [b"test<|", b"ok", b""]

    def test_split_by_special_tokens_keep_specials(self):
        res = list(
            bpe_tokenizer.BPETokenizer.split_by_special_tokens(
                b"test<|<|special|>ok<|special2|>",
                ["<|special|>", "<|special2|>"],
                keep_specials=True,
            )
        )
        assert res == [b"test<|", b"<|special|>", b"ok", b"<|special2|>", b""]

    def test_pre_tokenize(self):
        res = list(bpe_tokenizer.BPETokenizer.pre_tokenize([b"test<|", b"ok", b""]))
        assert res == [(b"t", b"e", b"s", b"t"), (b"<", b"|"), (b"o", b"k")]

    def test_get_word_counts(self):
        res = bpe_tokenizer.BPETokenizer.get_word_counts(
            [(b"t", b"e", b"s", b"t"), (b"o", b"k"), (b"t", b"e", b"s", b"t")]
        )
        assert res == {(b"t", b"e", b"s", b"t"): 2, (b"o", b"k"): 1}

    def test_get_indexed_word_counts(self):
        word_counts, idx2word = bpe_tokenizer.BPETokenizer.get_indexed_word_counts(
            {(b"t", b"e", b"s", b"t"): 2, (b"o", b"k"): 1}
        )
        assert word_counts == {0: 2, 1: 1}
        assert idx2word == {0: (b"t", b"e", b"s", b"t"), 1: (b"o", b"k")}

    def test_get_pair_counts(self):
        word_counts = {0: 2, 1: 1, 2: 3}
        idx2word = {
            0: (b"t", b"e", b"s", b"t"),
            1: (b"w", b"e", b"s", b"t"),
            2: (b"e", b"e", b"e", b"t"),
        }
        pair_counts, pair2words = bpe_tokenizer.BPETokenizer.get_pair_counts(
            word_counts, idx2word
        )
        assert pair_counts == {
            (b"t", b"e"): 2,
            (b"e", b"s"): 3,
            (b"s", b"t"): 3,
            (b"w", b"e"): 1,
            (b"e", b"e"): 6,
            (b"e", b"t"): 3,
        }
        assert pair2words == {
            (b"t", b"e"): {0: 1},
            (b"e", b"s"): {0: 1, 1: 1},
            (b"s", b"t"): {0: 1, 1: 1},
            (b"w", b"e"): {1: 1},
            (b"e", b"e"): {2: 2},
            (b"e", b"t"): {2: 1},
        }

    def test_get_max_pair(self):
        pair_counts = {
            (b"t", b"e"): 2,
            (b"e", b"s"): 3,
            (b"s", b"t"): 3,
            (b"w", b"e"): 1,
            (b"e", b"t"): 3,
        }
        count_pairs = bpe_tokenizer.BPETokenizer.get_count_pairs(pair_counts)
        max_pair = bpe_tokenizer.BPETokenizer.get_max_pair(count_pairs)
        assert max_pair == (b"s", b"t")

    def test_get_merged_word_beginning(self):
        new_word, affected_indices_old, affected_indices_new = (
            bpe_tokenizer.BPETokenizer.get_merged_word(
                (b"e", b"e", b"e", b"t"), (b"e", b"e"), b"ee"
            )
        )
        assert new_word == (b"ee", b"e", b"t")
        assert affected_indices_old == [1]
        assert affected_indices_new == [0]

    def test_get_merged_word_middle(self):
        new_word, affected_indices_old, affected_indices_new = (
            bpe_tokenizer.BPETokenizer.get_merged_word(
                (b"w", b"e", b"e", b"e", b"e", b"t"), (b"e", b"e"), b"ee"
            )
        )
        assert new_word == (b"w", b"ee", b"ee", b"t")
        assert affected_indices_old == [0,2,4]
        assert affected_indices_new == [0,1,2]

    def test_get_merged_word_end(self):
        new_word, affected_indices_old, affected_indices_new = (
            bpe_tokenizer.BPETokenizer.get_merged_word(
                (b"e", b"e", b"e", b"t"), (b"e", b"t"), b"et"
            )
        )
        assert new_word == (b"e", b"e", b"et")
        assert affected_indices_old == [1]
        assert affected_indices_new == [1]

    def test_update_pair_counts_for_word(self):
        old_word = (b"e", b"e", b"e", b"t")
        pair = (b"e", b"e")
        new_word, affected_indices_old, affected_indices_new = (
            bpe_tokenizer.BPETokenizer.get_merged_word(
                (b"e", b"e", b"e", b"t"), (b"e", b"e"), b"ee"
            )
        )
        pair_counts = {
            (b"t", b"e"): 2,
            (b"e", b"s"): 3,
            (b"s", b"t"): 3,
            (b"w", b"e"): 1,
            (b"e", b"e"): 6,
            (b"e", b"t"): 3,
        }
        pair2words = {
            (b"t", b"e"): {0: 1},
            (b"e", b"s"): {0: 1, 1: 1},
            (b"s", b"t"): {0: 1, 1: 1},
            (b"w", b"e"): {1: 1},
            (b"e", b"e"): {2: 2},
            (b"e", b"t"): {2: 1},
        }
        updated_pairs = {}
        bpe_tokenizer.BPETokenizer.update_pair_counts_for_word(
            word_idx=2,
            old_word=old_word,
            word_count=3,
            affected_indices_old=affected_indices_old,
            pair=pair,
            new_word=new_word,
            affected_indices_new=affected_indices_new,
            pair_counts=pair_counts,
            pair_to_words=pair2words,
            updated_pairs=updated_pairs,
        )
        assert pair_counts == {
            (b"t", b"e"): 2,
            (b"e", b"s"): 3,
            (b"s", b"t"): 3,
            (b"w", b"e"): 1,
            (b"e", b"e"): 6,
            (b"ee", b"e"): 3,
            (b"e", b"t"): 3,
        }
        assert pair2words == {
            (b"t", b"e"): {0: 1},
            (b"e", b"s"): {0: 1, 1: 1},
            (b"s", b"t"): {0: 1, 1: 1},
            (b"w", b"e"): {1: 1},
            (b"e", b"e"): {2: 2},
            (b"ee", b"e"): {2: 1},
            (b"e", b"t"): {2: 1},
        }

    def test_update_all_for_pair(self):
        pair = (b"e", b"s")
        joined_pair = b"es"
        word_counts = {0: 2, 1: 1, 2: 3}
        idx2word = {
            0: (b"t", b"e", b"s", b"t"),
            1: (b"w", b"e", b"s", b"t"),
            2: (b"e", b"e", b"e", b"t"),
        }
        pair_counts = {
            (b"t", b"e"): 2,
            (b"e", b"s"): 3,
            (b"s", b"t"): 3,
            (b"w", b"e"): 1,
            (b"e", b"e"): 6,
            (b"e", b"t"): 3,
        }
        pair2words = {
            (b"t", b"e"): {0: 1},
            (b"e", b"s"): {0: 1, 1: 1},
            (b"s", b"t"): {0: 1, 1: 1},
            (b"w", b"e"): {1: 1},
            (b"e", b"e"): {2: 2},
            (b"e", b"t"): {2: 1},
        }
        count_pairs = bpe_tokenizer.BPETokenizer.get_count_pairs(pair_counts)
        bpe_tokenizer.BPETokenizer.update_all_for_pair(
            pair,
            joined_pair,
            word_counts,
            idx2word,
            pair_counts,
            pair2words,
            count_pairs,
        )
        assert word_counts == {0: 2, 1: 1, 2: 3}
        assert idx2word == {
            0: (b"t", b"es", b"t"),
            1: (b"w", b"es", b"t"),
            2: (b"e", b"e", b"e", b"t"),
        }
        assert pair_counts == {
            (b"t", b"es"): 2,
            (b"es", b"t"): 3,
            (b"w", b"es"): 1,
            (b"e", b"e"): 6,
            (b"e", b"t"): 3,
        }
        assert pair2words == {
            (b"t", b"es"): {0: 1},
            (b"es", b"t"): {0: 1, 1: 1},
            (b"w", b"es"): {1: 1},
            (b"e", b"e"): {2: 2},
            (b"e", b"t"): {2: 1},
        }
        assert count_pairs == {
            1: set([(b"w", b"es")]),
            2: set([(b"t", b"es")]),
            3: set([(b"es", b"t"), (b"e", b"t")]),
            6: set([(b"e", b"e")]),
        }

    def test_train_loop(self):
        vocab_size = 10
        vocab: dict[int, bytes] = {}
        word1 = (b"i", b"n", b"d", b"e", b"p", b"e", b"n", b"d", b"e", b"n", b"t")
        word2 = (b"i", b"n", b"d", b"e", b"p", b"e", b"n", b"d", b"e", b"n", b"c", b"e")
        word3 = (b"i", b"n")
        word4 = (b"e", b"n")
        word5 = (b"a", b"n", b"d")
        word_counts, idx2word = bpe_tokenizer.BPETokenizer.get_indexed_word_counts(
            {
                word1: 1,
                word2: 2,
                word3: 100,
                word4: 20,
                word5: 50,
            }
        )
        merges = bpe_tokenizer.BPETokenizer.train_loop(
            vocab_size, word_counts, vocab, idx2word
        )
        assert merges


if __name__ == "__main__":
    unittest.main()
