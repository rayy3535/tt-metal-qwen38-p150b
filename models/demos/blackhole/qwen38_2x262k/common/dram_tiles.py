# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""shuffle_tensor_tiles, extracted from DeepSeek-V3-B1/tests/unit_tests/test_dram_streaming_matmul.py (code
verbatim; see ../PROVENANCE.md)."""

import torch


def shuffle_tensor_tiles(tensor, tile_size, num_banks):
    """
    Vectorized tile shuffle for WIDTH_SHARDED DRAM layout.

    Reorders tiles within each bank's shard from row-major to column-major.
    This is required because TTNN stores tiles row-major, but we want to
    stream K tiles contiguously for each N column.

    For shape [K, N] with WIDTH_SHARDED, TTNN tilizes with row-major tiles:
        (0,0), (0,1), ..., (0,pN-1), (1,0), (1,1), ...
    But we want column-major within each shard for streaming K tiles at a time:
        (0,0), (1,0), ..., (K - 1,0), (0,1), (1,1), ...

    Args:
        tensor: [*, K, N] tensor (supports batch dimensions)
        tile_size: tile dimension (assumes square tiles)
        num_banks: number of DRAM banks (shards)

    Returns:
        [*, K, N] tensor with tiles rearranged per-shard (same shape as input)
    """
    orig_shape = tensor.shape
    K = orig_shape[-2]
    N = orig_shape[-1]

    # Pad N to align with num_banks if needed
    lcm = tile_size * num_banks
    n_padded = ((N + lcm - 1) // lcm) * lcm
    needs_padding = n_padded != N

    # Flatten batch dimensions
    tensor = tensor.reshape(-1, K, N)
    batch_size = tensor.shape[0]

    if needs_padding:
        tensor = torch.nn.functional.pad(tensor, (0, n_padded - N))

    K_tiles = K // tile_size
    per_N = n_padded // num_banks
    per_N_tiles = per_N // tile_size
    num_tiles_per_shard = K_tiles * per_N_tiles

    # Split into shards: [batch, K, N] -> [batch, K, num_banks, per_N]
    tensor = tensor.reshape(batch_size, K, num_banks, per_N)
    # -> [batch, num_banks, K, per_N]
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    # -> [batch * num_banks, K, per_N]
    shards = tensor.reshape(-1, K, per_N)

    # Reshape to tiles: [shards, K_tiles, tile_h, per_N_tiles, tile_w]
    tiles = shards.reshape(-1, K_tiles, tile_size, per_N_tiles, tile_size)
    # -> [shards, K_tiles, per_N_tiles, tile_h, tile_w]
    tiles = tiles.permute(0, 1, 3, 2, 4).contiguous()
    # Flatten tile grid: [shards, num_tiles, tile_h, tile_w]
    tiles = tiles.reshape(-1, num_tiles_per_shard, tile_size, tile_size)

    # Compute source indices for shuffle
    # For shuffled position i, source = (i % K_tiles) * per_N_tiles + (i // K_tiles)
    i = torch.arange(num_tiles_per_shard, device=tensor.device)
    source_idx = (i % K_tiles) * per_N_tiles + (i // K_tiles)

    # Gather tiles in shuffled order
    shuffled_tiles = tiles[:, source_idx, :, :]

    # Reshape back to shards
    shuffled_tiles = shuffled_tiles.reshape(-1, K_tiles, per_N_tiles, tile_size, tile_size)
    shuffled_tiles = shuffled_tiles.permute(0, 1, 3, 2, 4).contiguous()
    shuffled_shards = shuffled_tiles.reshape(-1, K, per_N)

    # Reshape back to full tensor
    shuffled = shuffled_shards.reshape(batch_size, num_banks, K, per_N)
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    shuffled = shuffled.reshape(batch_size, K, n_padded)

    # Slice back to original N if padded
    if needs_padding:
        shuffled = shuffled[:, :, :N]

    # Restore original batch dimensions
    shuffled = shuffled.reshape(*orig_shape)

    return shuffled
