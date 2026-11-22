// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gdn_step_wide writer, one core per (value head, column c) of a 4-core group.  Per (lane, head) item:
//  1. exchange 1: this core's conv'd q_c / k_c tiles (cb_mine) into every group member's cb_qkx at tiles c and 4 + c;
//  2. exchange 2: this core's read-out tile o_c (cb_omine) into every member's cb_ox at tile c;
//  3. the 4 new state tiles of column c (state tile (lane*12+h)*16 + 4i + c) and the gated tile c to the output.
// Each exchange: every member reserves its receive CB, then raises READY on all members (itself included); a sender
// waits READY == 4, clears it, writes its tiles into the members' CB slots (the CBs sit at the same address on every
// core and advance in lockstep), raises DATA on all members; a receiver waits DATA == 4, clears it and pushes the CB.
// A member reaches the next item's exchange only after every member's data of this one landed, so the clears never
// race a later raise; the semaphores end every launch at 0.
// Compile-time args: 0 rows; then TensorAccessorArgs of state and out.  Runtime args: 0 state, 1 out addresses, 2
// column c, 3-18 the 4 members' NoC (x0, y0, x1, y1) (NoC 0 / NoC 1 coordinates), 19 items, then (lane, head) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_outs = 25, cb_outg = 26, cb_qkx = 28, cb_ox = 29, cb_mine = 30, cb_omine = 31;
constexpr uint32_t value_heads = 12, head_tiles = 4, head_state_tiles = 16, members = 4;
constexpr uint32_t bf16_tile_bytes = 2048, fp32_tile_bytes = 4096, face_bytes = 512, half_row_bytes = 32;
constexpr uint32_t sem_ready1 = 0, sem_data1 = 1, sem_ready2 = 2, sem_data2 = 3;

struct Members {
    uint32_t x[members];
    uint32_t y[members];
};

FORCE_INLINE void raise_all(const Members& m, uint32_t sem_addr) {
    for (uint32_t i = 0; i < members; ++i) {
        noc_semaphore_inc(get_noc_addr(m.x[i], m.y[i], sem_addr), 1);
    }
}

FORCE_INLINE void wait_clear(uint32_t sem_addr) {
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait(sem, members);
    noc_semaphore_set(sem, 0);
}

// Send `count` tiles of `src_cb` to every member's `dst_cb` write slot, tile k at dst tile offsets[k].
template <uint32_t count>
FORCE_INLINE void exchange(
    const Members& m,
    uint32_t src_cb,
    uint32_t dst_cb,
    uint32_t dst_tiles,
    const uint32_t (&offsets)[count],
    uint32_t ready_addr,
    uint32_t data_addr) {
    CircularBuffer dst(dst_cb);
    CircularBuffer src(src_cb);
    dst.reserve_back(dst_tiles);
    const uint32_t dst_addr = dst.get_write_ptr();
    raise_all(m, ready_addr);
    src.wait_front(count);
    const uint32_t src_addr = src.get_read_ptr();
    wait_clear(ready_addr);
    for (uint32_t i = 0; i < members; ++i) {
        for (uint32_t k = 0; k < count; ++k) {
            noc_async_write(
                src_addr + k * bf16_tile_bytes,
                get_noc_addr(m.x[i], m.y[i], dst_addr + offsets[k] * bf16_tile_bytes),
                bf16_tile_bytes);
        }
    }
    noc_async_write_barrier();
    raise_all(m, data_addr);
    src.pop_front(count);
    wait_clear(data_addr);
    dst.push_back(dst_tiles);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr auto state_args = TensorAccessorArgs<1>();
    constexpr auto out_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    uint32_t arg = 0;
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t col = get_arg_val<uint32_t>(arg++);
    Members m;
    for (uint32_t i = 0; i < members; ++i) {
        const uint32_t x0 = get_arg_val<uint32_t>(arg++);
        const uint32_t y0 = get_arg_val<uint32_t>(arg++);
        const uint32_t x1 = get_arg_val<uint32_t>(arg++);
        const uint32_t y1 = get_arg_val<uint32_t>(arg++);
        m.x[i] = noc_index == 0 ? x0 : x1;
        m.y[i] = noc_index == 0 ? y0 : y1;
    }
    const uint32_t items = get_arg_val<uint32_t>(arg++);
    const auto state = TensorAccessor(state_args, state_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    const Noc noc;
    const uint32_t ready1 = get_semaphore(sem_ready1), data1 = get_semaphore(sem_data1);
    const uint32_t ready2 = get_semaphore(sem_ready2), data2 = get_semaphore(sem_data2);
    const uint32_t qk_offsets[2] = {col, head_tiles + col};
    const uint32_t o_offsets[1] = {col};

    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t head = get_arg_val<uint32_t>(arg++);
        const uint32_t lane0 = get_arg_val<uint32_t>(arg++);
        const uint32_t lanes = get_arg_val<uint32_t>(arg++);

        exchange<2>(m, cb_mine, cb_qkx, 2 * head_tiles, qk_offsets, ready1, data1);

        // each user's new state column, in user order as the compute produces them
        for (uint32_t u = lane0; u < lane0 + lanes; ++u) {
            CircularBuffer(cb_outs).wait_front(head_tiles);
            for (uint32_t i = 0; i < head_tiles; ++i) {
                noc.async_write(
                    CircularBuffer(cb_outs),
                    state,
                    state.get_aligned_page_size(),
                    {.offset_bytes = i * fp32_tile_bytes},
                    {.page_id = (u * value_heads + head) * head_state_tiles + i * head_tiles + col});
            }
            noc.async_write_barrier();
            CircularBuffer(cb_outs).pop_front(head_tiles);
        }

        exchange<1>(m, cb_omine, cb_ox, head_tiles, o_offsets, ready2, data2);

        // the gated tile holds every user's row (rows 0..rows-1): one whole-page write
        CircularBuffer(cb_outg).wait_front(1);
        if (lanes == rows) {
            noc.async_write(
                CircularBuffer(cb_outg),
                out,
                out.get_aligned_page_size(),
                {.offset_bytes = 0},
                {.page_id = head * head_tiles + col});
        } else {
            // split users: this group's rows only (both 16-wide halves of each row)
            for (uint32_t u = lane0; u < lane0 + lanes; ++u) {
                const uint32_t row_offset = ((u >> 4) * 2) * face_bytes + (u & 15) * half_row_bytes;
                for (uint32_t h = 0; h < 2; ++h) {
                    noc.async_write(
                        CircularBuffer(cb_outg),
                        out,
                        half_row_bytes,
                        {.offset_bytes = row_offset + h * face_bytes},
                        {.page_id = head * head_tiles + col, .offset_bytes = row_offset + h * face_bytes});
                }
            }
        }
        noc.async_write_barrier();
        CircularBuffer(cb_outg).pop_front(1);
    }
    noc_async_atomic_barrier();
}
