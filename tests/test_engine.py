"""
Test Engine class. Example run:

python -m pytest tests/test_engine.py -v
"""

import torch
from nanochat.engine import KVCache


def test_kv_cache_resize():
    """
    The KV cache was not resized correctly, more information here:
    https://github.com/karpathy/nanochat/pull/186
    This test reproduces the issue and will be merged alongside the fix.
    """

    batch_size = 2
    num_heads = 3
    seq_len = 4
    kv_compressed = 5
    rotated_dim = 3
    num_layers = 6

    kv_cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        kv_compressed=kv_compressed,
        rotated_dim=rotated_dim,
        num_layers=num_layers,
    )

    # Insert a single token with a distinct fill value to all layers
    def insert_token(token_idx):
        for layer_idx in range(num_layers):
            ckv = torch.full(
                (batch_size, num_heads, 1, kv_compressed),
                fill_value=float(token_idx),
                dtype=torch.float32,
            )
            kr = torch.full(
                (batch_size, num_heads, 1, rotated_dim),
                fill_value=float(token_idx * 100),
                dtype=torch.float32,
            )
            kv_cache.insert_kv(layer_idx, ckv, kr)

    # Insert 4 tokens (fills the initial seq_len=4)
    for i in range(4):
        insert_token(i)

    # Record the original state of the cache
    original_cache_ckv = kv_cache.ckv_cache.clone()
    original_cache_kr = kv_cache.kr_cache.clone()
    original_seq_len = original_cache_ckv.shape[3]

    # Insert the 5th token, which will trigger a resize
    insert_token(4)
    # Verify that the cache actually resized
    new_seq_len = kv_cache.ckv_cache.shape[3]
    assert new_seq_len > original_seq_len, (
        f"Cache did not resize: original seq_len={original_seq_len}, new seq_len={new_seq_len}"
    )

    # Verify that the original 4 tokens are still intact after resize
    for layer_idx in range(num_layers):
        for token_idx in range(4):
            # Check that resized cache matches expected values
            expected_ckv = float(token_idx)
            expected_kr = float(token_idx * 100)
            actual_ckv = kv_cache.ckv_cache[layer_idx, :, :, token_idx, :]
            actual_kr = kv_cache.kr_cache[layer_idx, :, :, token_idx, :]
            assert (actual_ckv == expected_ckv).all(), (
                f"Layer {layer_idx}, token {token_idx}: compressed kv corrupted, expected {expected_ckv}"
            )
            assert (actual_kr == expected_kr).all(), (
                f"Layer {layer_idx}, token {token_idx}: value corrupted, expected {expected_kr}"
            )
            # And that the original cache matches resized cache
            original_ckv = original_cache_ckv[layer_idx, :, :, token_idx, :]
            original_kr = original_cache_kr[layer_idx, :, :, token_idx, :]

            assert (actual_ckv == original_ckv).all(), (
                f"Layer {layer_idx}, token {token_idx}: compressed kv doesn't match original"
            )
            assert (actual_kr == original_kr).all(), (
                f"Layer {layer_idx}, token {token_idx}: value doesn't match original"
            )
