// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gdn_step_wide reader: one core per (head, column c) of a 4-core group; per (lane, head) item it reads only the
// q/k tile c of qk head h/3, the v and z tiles c of head h and the state column c (tiles 4i + c).
// Base per-item design (gdn_step):
// Per (lane, head) item: the projection's q/k/v/z/a/b tiles, the three older conv ring slots' q/k/v tiles, the four
// taps' q/k/v tiles (tile-major: cb_s holds tile t's three slots at 3t.., cb_t its four taps at 4t.., pushed per
// tile), the head's dt_bias / neg_exp_A tiles, the lane's state tiles, and the lane's row mask; writes
// the projection's q/k/v tiles into the newest ring slot.  Tile ids: q 4(h/3)+c, k 16+4(h/3)+c, v 32+4h+c,
// z 80+4h+c, a 128, b 129 (one tile row of the [1,1,rows,4160] projection); state ((lane*12+h)*16 + t).
// Compile-time args: 0 rows; then TensorAccessorArgs of projected, slot0, slot1, slot2, tap0..tap3, dtna, norm,
// state, newest.  Runtime args: 0 projected, 1-3 slots, 4-7 taps, 8 dtna, 9 norm, 10 state, 11 newest addresses,
// 12 items, then (lane, head) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_p = 0, cb_s = 1, cb_t = 2, cb_z = 3, cb_ab = 4, cb_dtna = 5, cb_w = 6, cb_state = 7;
constexpr uint32_t cb_mask = 8, cb_scaler = 9;
constexpr uint32_t value_heads = 12, head_tiles = 4, qkv_tiles = 3, head_state_tiles = 16, col_state_tiles = 4;
constexpr uint32_t k_tile0 = 16, v_tile0 = 32, z_tile0 = 80, a_tile = 128, b_tile = 129;
constexpr uint32_t bf16_tile_bytes = 2048, fp32_tile_bytes = 4096, face_elements = 256;

constexpr uint32_t face_element(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * face_elements + (row & 15) * 16 + (col & 15);
}

// Reserves `count` pages of `cb` and reads page page_ids[t] of `tensor` into page t; the caller barriers and pushes.
template <typename Accessor>
void read_pages(uint32_t cb_id, const Accessor& tensor, const uint32_t* page_ids, uint32_t count, uint32_t page_bytes) {
    const Noc noc;
    CircularBuffer cb(cb_id);
    cb.reserve_back(count);
    for (uint32_t t = 0; t < count; ++t) {
        noc.async_read(
            tensor, cb, tensor.get_aligned_page_size(), {.page_id = page_ids[t]}, {.offset_bytes = t * page_bytes});
    }
}

void fill_words(uint32_t l1_addr, uint32_t words, uint32_t value) {
    CoreLocalMem<volatile uint32_t> dst(l1_addr);
    for (uint32_t k = 0; k < words; ++k) {
        dst[k] = value;
    }
}

// The L1 bases of an item's conv inputs: P, S and T reserved for the whole item.
struct ConvInputs {
    uint32_t projected;  // cb_p: the 12 projection q/k/v tiles
    uint32_t slots;      // cb_s: three older ring slots per tile, tile-major
    uint32_t taps;       // cb_t: four taps per tile, tile-major
};

// Issues (without a barrier) tile `tile`'s conv input group: page `page` of the projection, of the three older ring
// slots and of the four taps.
template <
    typename Projected,
    typename Slot0,
    typename Slot1,
    typename Slot2,
    typename Tap0,
    typename Tap1,
    typename Tap2,
    typename Tap3>
void read_conv_group(
    uint32_t page,
    uint32_t tile,
    const ConvInputs& conv_l1,
    const Projected& projected,
    const Slot0& slot0,
    const Slot1& slot1,
    const Slot2& slot2,
    const Tap0& tap0,
    const Tap1& tap1,
    const Tap2& tap2,
    const Tap3& tap3) {
    const Noc noc;
    const CoreLocalMem<uint32_t> projected_l1(conv_l1.projected), slots_l1(conv_l1.slots), taps_l1(conv_l1.taps);
    noc.async_read(
        projected,
        projected_l1,
        projected.get_aligned_page_size(),
        {.page_id = page},
        {.offset_bytes = tile * bf16_tile_bytes});
    noc.async_read(
        slot0,
        slots_l1,
        slot0.get_aligned_page_size(),
        {.page_id = page},
        {.offset_bytes = (3 * tile + 0) * bf16_tile_bytes});
    noc.async_read(
        slot1,
        slots_l1,
        slot1.get_aligned_page_size(),
        {.page_id = page},
        {.offset_bytes = (3 * tile + 1) * bf16_tile_bytes});
    noc.async_read(
        slot2,
        slots_l1,
        slot2.get_aligned_page_size(),
        {.page_id = page},
        {.offset_bytes = (3 * tile + 2) * bf16_tile_bytes});
    noc.async_read(
        tap0,
        taps_l1,
        tap0.get_aligned_page_size(),
        {.page_id = page},
        {.offset_bytes = (4 * tile + 0) * bf16_tile_bytes});
    noc.async_read(
        tap1,
        taps_l1,
        tap1.get_aligned_page_size(),
        {.page_id = page},
        {.offset_bytes = (4 * tile + 1) * bf16_tile_bytes});
    noc.async_read(
        tap2,
        taps_l1,
        tap2.get_aligned_page_size(),
        {.page_id = page},
        {.offset_bytes = (4 * tile + 2) * bf16_tile_bytes});
    noc.async_read(
        tap3,
        taps_l1,
        tap3.get_aligned_page_size(),
        {.page_id = page},
        {.offset_bytes = (4 * tile + 3) * bf16_tile_bytes});
}
}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t projected_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t slot0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t slot1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t slot2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap3_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dtna_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t norm_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t newest_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t col = get_arg_val<uint32_t>(arg++);
    const uint32_t items = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr auto projected_args = TensorAccessorArgs<1>();
    constexpr auto slot0_args = TensorAccessorArgs<projected_args.next_compile_time_args_offset()>();
    constexpr auto slot1_args = TensorAccessorArgs<slot0_args.next_compile_time_args_offset()>();
    constexpr auto slot2_args = TensorAccessorArgs<slot1_args.next_compile_time_args_offset()>();
    constexpr auto tap0_args = TensorAccessorArgs<slot2_args.next_compile_time_args_offset()>();
    constexpr auto tap1_args = TensorAccessorArgs<tap0_args.next_compile_time_args_offset()>();
    constexpr auto tap2_args = TensorAccessorArgs<tap1_args.next_compile_time_args_offset()>();
    constexpr auto tap3_args = TensorAccessorArgs<tap2_args.next_compile_time_args_offset()>();
    constexpr auto dtna_args = TensorAccessorArgs<tap3_args.next_compile_time_args_offset()>();
    constexpr auto norm_args = TensorAccessorArgs<dtna_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<norm_args.next_compile_time_args_offset()>();
    constexpr auto newest_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();

    const auto projected = TensorAccessor(projected_args, projected_addr);
    const auto slot0 = TensorAccessor(slot0_args, slot0_addr);
    const auto slot1 = TensorAccessor(slot1_args, slot1_addr);
    const auto slot2 = TensorAccessor(slot2_args, slot2_addr);
    const auto tap0 = TensorAccessor(tap0_args, tap0_addr);
    const auto tap1 = TensorAccessor(tap1_args, tap1_addr);
    const auto tap2 = TensorAccessor(tap2_args, tap2_addr);
    const auto tap3 = TensorAccessor(tap3_args, tap3_addr);
    const auto dtna = TensorAccessor(dtna_args, dtna_addr);
    const auto norm = TensorAccessor(norm_args, norm_addr);
    const auto state = TensorAccessor(state_args, state_addr);
    const auto newest = TensorAccessor(newest_args, newest_addr);

    const Noc noc;

    {
        const uint32_t norm_pages[1] = {col};
        // no barrier here: the norm tile lands with the first item's conv reads and is pushed after the first
        // item's state barrier (the compute reads it only in the gated norm)
        read_pages(cb_w, norm, norm_pages, 1, bf16_tile_bytes);
        // reduce scalers (fp32, whole tiles): 1.0 for sums, 1/128 for means
        CircularBuffer(cb_scaler).reserve_back(2);
        const uint32_t scaler_l1 = CircularBuffer(cb_scaler).get_write_ptr();
        fill_words(scaler_l1, fp32_tile_bytes / 4, 0x3F800000u);
        fill_words(scaler_l1 + fp32_tile_bytes, fp32_tile_bytes / 4, 0x3C000000u);
        CircularBuffer(cb_scaler).push_back(2);
    }

    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t head = get_arg_val<uint32_t>(arg++);
        const uint32_t lane0 = get_arg_val<uint32_t>(arg++);
        const uint32_t lanes = get_arg_val<uint32_t>(arg++);
        const uint32_t qk_head = head / 3;

        const uint32_t qkv_pages[qkv_tiles] = {
            qk_head * head_tiles + col, k_tile0 + qk_head * head_tiles + col, v_tile0 + head * head_tiles + col};
        noc.async_write_barrier();  // the previous item's newest-slot writes have left the cb_p slot
        // The conv inputs go out per output tile t (P[t], the three slots' tile t at cb_s 3t.., the four taps' tile t
        // at cb_t 4t..) with one group of reads in flight ahead of each barrier, so the conv starts after one group.
        // The L1 bases are taken once: the per-tile pushes below move the CBs' write pointers.
        CircularBuffer(cb_p).reserve_back(qkv_tiles);
        CircularBuffer(cb_s).reserve_back(3 * qkv_tiles);
        CircularBuffer(cb_t).reserve_back(4 * qkv_tiles);
        const ConvInputs conv_l1 = {
            .projected = CircularBuffer(cb_p).get_write_ptr(),
            .slots = CircularBuffer(cb_s).get_write_ptr(),
            .taps = CircularBuffer(cb_t).get_write_ptr()};
        read_conv_group(qkv_pages[0], 0, conv_l1, projected, slot0, slot1, slot2, tap0, tap1, tap2, tap3);
        for (uint32_t t = 0; t < qkv_tiles; ++t) {
            if (t + 1 < qkv_tiles) {
                read_conv_group(
                    qkv_pages[t + 1], t + 1, conv_l1, projected, slot0, slot1, slot2, tap0, tap1, tap2, tap3);
            }
            noc.async_read_barrier();  // groups <= t + 1 landed; group t goes to the compute, P[t] to the newest slot
            noc.async_write(
                CoreLocalMem<uint32_t>(conv_l1.projected),
                newest,
                newest.get_aligned_page_size(),
                {.offset_bytes = t * bf16_tile_bytes},
                {.page_id = qkv_pages[t]});
            CircularBuffer(cb_p).push_back(1);
            CircularBuffer(cb_s).push_back(3);
            CircularBuffer(cb_t).push_back(4);
        }

        const uint32_t z_pages[1] = {z_tile0 + head * head_tiles + col};
        read_pages(cb_z, projected, z_pages, 1, bf16_tile_bytes);
        // per user u (row u of every row tile): its a/b tiles (element (u, head) moved to (0, 0)), dt/A, its state
        // column and its row mask, all prefetched behind one barrier
        CircularBuffer(cb_ab).reserve_back(2 * lanes);
        CircularBuffer(cb_dtna).reserve_back(2 * lanes);
        CircularBuffer(cb_state).reserve_back(col_state_tiles * lanes);
        {
            const uint32_t ab_l1 = CircularBuffer(cb_ab).get_write_ptr();
            const uint32_t dtna_l1 = CircularBuffer(cb_dtna).get_write_ptr();
            const uint32_t state_l1 = CircularBuffer(cb_state).get_write_ptr();
            const CoreLocalMem<uint32_t> ab_mem(ab_l1), dtna_mem(dtna_l1), state_mem(state_l1);
            for (uint32_t j = 0; j < lanes; ++j) {
                const uint32_t u = lane0 + j;
                noc.async_read(
                    projected,
                    ab_mem,
                    projected.get_aligned_page_size(),
                    {.page_id = a_tile},
                    {.offset_bytes = (2 * j) * bf16_tile_bytes});
                noc.async_read(
                    projected,
                    ab_mem,
                    projected.get_aligned_page_size(),
                    {.page_id = b_tile},
                    {.offset_bytes = (2 * j + 1) * bf16_tile_bytes});
                noc.async_read(
                    dtna,
                    dtna_mem,
                    dtna.get_aligned_page_size(),
                    {.page_id = head},
                    {.offset_bytes = (2 * j) * fp32_tile_bytes});
                noc.async_read(
                    dtna,
                    dtna_mem,
                    dtna.get_aligned_page_size(),
                    {.page_id = value_heads + head},
                    {.offset_bytes = (2 * j + 1) * fp32_tile_bytes});
                for (uint32_t i = 0; i < col_state_tiles; ++i) {
                    noc.async_read(
                        state,
                        state_mem,
                        state.get_aligned_page_size(),
                        {.page_id = (u * value_heads + head) * head_state_tiles + i * head_tiles + col},
                        {.offset_bytes = (j * col_state_tiles + i) * fp32_tile_bytes});
                }
            }
            noc.async_read_barrier();
            for (uint32_t j = 0; j < lanes; ++j) {
                const uint32_t u = lane0 + j;
                CoreLocalMem<volatile uint16_t> a_elements(ab_l1 + (2 * j) * bf16_tile_bytes);
                CoreLocalMem<volatile uint16_t> b_elements(ab_l1 + (2 * j + 1) * bf16_tile_bytes);
                const uint32_t lane_element = face_element(u, head);
                a_elements[0] = a_elements[lane_element];
                b_elements[0] = b_elements[lane_element];
            }
        }
        CircularBuffer(cb_z).push_back(1);
        CircularBuffer(cb_ab).push_back(2 * lanes);
        CircularBuffer(cb_dtna).push_back(2 * lanes);
        CircularBuffer(cb_state).push_back(col_state_tiles * lanes);
        if (item == 0) {
            // the norm weight lands as row 0 of its tile; the gated norm multiplies it elementwise, so copy row 0 to
            // every user row (both 16-wide halves: faces 0 and 1)
            CoreLocalMem<volatile uint16_t> w_elements(CircularBuffer(cb_w).get_write_ptr());
            for (uint32_t u = 1; u < rows; ++u) {
                for (uint32_t c = 0; c < 16; ++c) {
                    w_elements[face_element(u, c)] = w_elements[face_element(0, c)];
                    w_elements[face_element(u, 16 + c)] = w_elements[face_element(0, 16 + c)];
                }
            }
            CircularBuffer(cb_w).push_back(1);
        }

        CircularBuffer(cb_mask).reserve_back(lanes);
        for (uint32_t j = 0; j < lanes; ++j) {
            const uint32_t u = lane0 + j;
            const uint32_t mask_l1 = CircularBuffer(cb_mask).get_write_ptr() + j * bf16_tile_bytes;
            fill_words(mask_l1, bf16_tile_bytes / 4, 0);
            CoreLocalMem<volatile uint16_t> elements(mask_l1);
            const uint32_t left = face_element(u, 0);
            for (uint32_t c = 0; c < 16; ++c) {
                elements[left + c] = 0x3F80;
                elements[left + face_elements + c] = 0x3F80;
            }
        }
        CircularBuffer(cb_mask).push_back(lanes);
    }
    noc.async_write_barrier();
}
