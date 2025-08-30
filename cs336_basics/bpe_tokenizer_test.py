from cs336_basics import bpe_tokenizer


def test_pre_tokenize():
    res = list(
        bpe_tokenizer.BPETokenizer.pre_tokenize(
            "test<|special|>ok<|special2|>", ["<|special|>", "<|special2|>"]
        )
    )
    assert res == [(b"t", b"e", b"s", b"t"), (b"o", b"k")]


def test_get_word_counts():
    res = bpe_tokenizer.BPETokenizer.get_word_counts(
        [(b"t", b"e", b"s", b"t"), (b"o", b"k"), (b"t", b"e", b"s", b"t")]
    )
    assert res == {(b"t", b"e", b"s", b"t"): 2, (b"o", b"k"): 1}


def test_get_pair_counts():
    pair_counts, pair2words = bpe_tokenizer.BPETokenizer.get_pair_counts(
        {
            (b"t", b"e", b"s", b"t"): 2,
            (b"w", b"e", b"s", b"t"): 1,
            (b"e", b"e", b"e", b"t"): 3,
        }
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
        (b"t", b"e"): set([(b"t", b"e", b"s", b"t")]),
        (b"e", b"s"): set([(b"t", b"e", b"s", b"t"), (b"w", b"e", b"s", b"t")]),
        (b"s", b"t"): set([(b"t", b"e", b"s", b"t"), (b"w", b"e", b"s", b"t")]),
        (b"w", b"e"): set([(b"w", b"e", b"s", b"t")]),
        (b"e", b"e"): set([(b"e", b"e", b"e", b"t")]),
        (b"e", b"t"): set([(b"e", b"e", b"e", b"t")]),
    }


def test_get_max_pair():
    pair_counts = {
        (b"t", b"e"): 2,
        (b"e", b"s"): 3,
        (b"s", b"t"): 3,
        (b"w", b"e"): 1,
        (b"e", b"t"): 3,
    }
    max_pair = bpe_tokenizer.BPETokenizer.get_max_pair(pair_counts)
    assert max_pair == (b"s", b"t")


def test_get_merged_word():
    new_word = bpe_tokenizer.BPETokenizer.get_merged_word(
        (b"e", b"e", b"e", b"t"), (b"e", b"e"), b"ee"
    )
    assert new_word == (b"ee", b"e", b"t")


def test_update_pair_counts_for_word():
    old_word = (b"e", b"e", b"e", b"t")
    pair = (b"e", b"e")
    new_word = (b"ee", b"e", b"t")
    pair_counts = {
        (b"t", b"e"): 2,
        (b"e", b"s"): 3,
        (b"s", b"t"): 3,
        (b"w", b"e"): 1,
        (b"e", b"e"): 6,
        (b"e", b"t"): 3,
    }
    pair2words = {
        (b"t", b"e"): set([(b"t", b"e", b"s", b"t")]),
        (b"e", b"s"): set([(b"t", b"e", b"s", b"t"), (b"w", b"e", b"s", b"t")]),
        (b"s", b"t"): set([(b"t", b"e", b"s", b"t"), (b"w", b"e", b"s", b"t")]),
        (b"w", b"e"): set([(b"w", b"e", b"s", b"t")]),
        (b"e", b"e"): set([(b"e", b"e", b"e", b"t")]),
        (b"e", b"t"): set([(b"e", b"e", b"e", b"t")]),
    }
    bpe_tokenizer.BPETokenizer.update_pair_counts_for_word(
        old_word, 3, pair, new_word, pair_counts, pair2words
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
        (b"t", b"e"): set([(b"t", b"e", b"s", b"t")]),
        (b"e", b"s"): set([(b"t", b"e", b"s", b"t"), (b"w", b"e", b"s", b"t")]),
        (b"s", b"t"): set([(b"t", b"e", b"s", b"t"), (b"w", b"e", b"s", b"t")]),
        (b"w", b"e"): set([(b"w", b"e", b"s", b"t")]),
        (b"e", b"e"): set([(b"e", b"e", b"e", b"t")]),
        (b"ee", b"e"): set([(b"ee", b"e", b"t")]),
        (b"e", b"t"): set([(b"ee", b"e", b"t")]),
    }


def test_update_all_for_pair():
    pair = (b"e", b"s")
    joined_pair = b"es"
    word_counts = {
        (b"t", b"e", b"s", b"t"): 2,
        (b"w", b"e", b"s", b"t"): 1,
        (b"e", b"e", b"e", b"t"): 3,
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
        (b"t", b"e"): set([(b"t", b"e", b"s", b"t")]),
        (b"e", b"s"): set([(b"t", b"e", b"s", b"t"), (b"w", b"e", b"s", b"t")]),
        (b"s", b"t"): set([(b"t", b"e", b"s", b"t"), (b"w", b"e", b"s", b"t")]),
        (b"w", b"e"): set([(b"w", b"e", b"s", b"t")]),
        (b"e", b"e"): set([(b"e", b"e", b"e", b"t")]),
        (b"e", b"t"): set([(b"e", b"e", b"e", b"t")]),
    }
    bpe_tokenizer.BPETokenizer.update_all_for_pair(
        pair, joined_pair, word_counts, pair_counts, pair2words
    )
    assert word_counts == {
        (b"t", b"es", b"t"): 2,
        (b"w", b"es", b"t"): 1,
        (b"e", b"e", b"e", b"t"): 3,
    }
    assert pair_counts == {
        (b"t", b"es"): 2,
        (b"es", b"t"): 3,
        (b"w", b"es"): 1,
        (b"e", b"e"): 6,
        (b"e", b"t"): 3,
    }
    assert pair2words == {
        (b"t", b"es"): set([(b"t", b"es", b"t")]),
        (b"es", b"t"): set([(b"t", b"es", b"t"), (b"w", b"es", b"t")]),
        (b"w", b"es"): set([(b"w", b"es", b"t")]),
        (b"e", b"e"): set([(b"e", b"e", b"e", b"t")]),
        (b"e", b"t"): set([(b"e", b"e", b"e", b"t")]),
    }
