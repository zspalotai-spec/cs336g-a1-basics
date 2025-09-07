NORMALIZER = 1e9

def get_embedding_flops(vocab_size: int, d_model: int) -> int:
    return 0


def get_linear_flops(d_in: int, d_out: int, total_batch: int) -> int:
    return 2 * total_batch * d_in * d_out / NORMALIZER


def get_rms_norm_flops(d_model: int, total_batch: int) -> int:
    return 3 * total_batch * d_model / NORMALIZER


def get_swi_glu_flops(d_model: int, d_ff: int, total_batch: int) -> int:
    return (
        2 * get_linear_flops(d_model, d_ff, total_batch)
        + 2 * total_batch * d_ff
        + get_linear_flops(d_ff, d_model, total_batch)
    ) / NORMALIZER


def get_attention_flops(d_model: int, num_heads: int, seq_len: int, batch: int) -> int:
    d_k = d_model // num_heads
    total_batch = num_heads * batch
    return total_batch * (
        2 * seq_len * d_k * seq_len + seq_len * seq_len + 2 * seq_len * seq_len * d_k
    ) / NORMALIZER


def get_transformer_block_flops(
    d_model: int, num_heads: int, d_ff: int, seq_len: int, batch: int
) -> dict:
    total_batch = seq_len * batch
    return {
        "transformer_rms": 2 * get_rms_norm_flops(d_model, total_batch),
        "transformer_attention": get_attention_flops(
            d_model, num_heads, seq_len, batch
        ),
        "transformer_ff": get_swi_glu_flops(d_model, d_ff, total_batch),
    }


def get_transform_lm_flops(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    batch: int,
) -> dict:
    transformer = get_transformer_block_flops(d_model, num_heads, d_ff, context_length, batch)
    ret = {k: num_layers * v for k, v in transformer.items()}
    total_batch = context_length * batch
    ret["final_rms_norm"] = get_rms_norm_flops(d_model, total_batch)
    ret["final_linear"] = get_linear_flops(d_model, vocab_size, total_batch)
    ret["total"] = sum(ret.values())
    return ret


if __name__ == "__main__":
    print(get_transform_lm_flops(50257, 1024, 12, 768, 12, 4 * 768, 1))
    print(get_transform_lm_flops(50257, 1024, 36, 1280, 20, 4 * 1280, 1))
    print(get_transform_lm_flops(50257, 1024, 48, 1600, 25, 6400, 1))
    print(get_transform_lm_flops(50257, 16 * 1024, 48, 1600, 25, 6400, 1))
