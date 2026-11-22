// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_index IX_C4=1 selector (BRISC, NoC 1): waits for the npad gathered pages (SEM_G, one increment per sending core
// after its write barrier), multicasts the 64 * npad bytes of its CB_ARR to the rectangle's CB_ARR, write barrier
// (acks from all rectangle cores: the tile has landed everywhere), then +1 SEM_D on the rectangle.
// Trace -> output row 64: [8] start, [9] all pages in, [10] multicast + barrier done, [11] increment done.
// Compile-time: TensorAccessorArgs pos, out.  Runtime: 0 pos, 1 out, 2 u, 3..6 rectangle (NoC 1), 7 rectangle cores.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ix_common.hpp"

using namespace ix;

FORCE_INLINE uint32_t wall_lo() {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0];
}

void kernel_main() {
    const uint32_t pos_addr = get_arg_val<uint32_t>(0), out_addr = get_arg_val<uint32_t>(1);
    const uint32_t u = get_arg_val<uint32_t>(2);
    const uint32_t rx0 = get_arg_val<uint32_t>(3), ry0 = get_arg_val<uint32_t>(4);
    const uint32_t rx1 = get_arg_val<uint32_t>(5), ry1 = get_arg_val<uint32_t>(6);
    const uint32_t nr = get_arg_val<uint32_t>(7);
    constexpr auto a_pos = TensorAccessorArgs<0>();
    constexpr auto a_out = TensorAccessorArgs<a_pos.next_compile_time_args_offset()>();
    const auto posb = TensorAccessor(a_pos, pos_addr, 64);
    const auto outb = TensorAccessor(a_out, out_addr, OUT_WORDS * 4);
    const uint32_t ts0 = wall_lo();
    const uint32_t ws = IX_WS_ADDR();
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws);
    noc_async_read(posb.get_noc_addr(0), ws, 64);
    noc_async_read_barrier();
    const uint32_t pos = w[u];
    volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws + 1024);
    for (uint32_t j = 0; j < OUT_WORDS; ++j) {
        tr[j] = 0;
    }
    tr[8] = ts0;
    if (pos != PARKED && pos + 1 > C4_MAX_L) {
        tr[63] = C4_OVER;  // routed to C4 with > 1,024 blocks: no selection (the host asserts on this word)
    }
    if (IX_SPARSE(pos)) {
        const uint32_t complete = (pos + 1) / 4, npt = (complete + 15) / 16, npad = (npt + 3) & ~3u;
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_G)), npad);
        tr[9] = wall_lo();
#ifdef IX_OPG
        {  // IX_OPG (qsa_layer QS_OPGATE_C4): every page is in (all scored), release the out cores' o_proj weight reads
            volatile tt_l1_ptr uint32_t* fl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(IX_DEC_ADDR() + IX_OPG);
            IX_OPG_GATE();  // (the out cores have cleared their copies: ix_common.hpp)
            fl[0] = 1;
            fl[1] = fl[2] = fl[3] = 0;
            noc_async_write_multicast(
                IX_DEC_ADDR() + IX_OPG, get_noc_multicast_addr(rx0, ry0, rx1, ry1, IX_DEC_ADDR() + IX_OPG), 16, nr);
            noc_async_writes_flushed();
        }
#endif
        const uint32_t arr = get_write_ptr(CB_ARR);
#ifdef IX_C4LINK
        // IX_C4LINK: the gathered tile multicast linked with a semaphore SET multicast (value 1): same path, ordered,
        // no write barrier in between (the scoring cores wait for >= 1 once)
        noc_async_write_multicast(arr, get_noc_multicast_addr(rx0, ry0, rx1, ry1, arr), 64 * npad, nr, true);
        tr[10] = wall_lo();
        w[252] = 1;  // 16 B aligned source (the semaphore destination is 16 B aligned)
        noc_semaphore_set_multicast(ws + 4 * 252, get_noc_multicast_addr(rx0, ry0, rx1, ry1, get_semaphore(SEM_D)), nr);
        noc_async_write_barrier();
#else
        noc_async_write_multicast(arr, get_noc_multicast_addr(rx0, ry0, rx1, ry1, arr), 64 * npad, nr);
        noc_async_write_barrier();
        tr[10] = wall_lo();
        noc_semaphore_inc_multicast(get_noc_multicast_addr(rx0, ry0, rx1, ry1, get_semaphore(SEM_D)), 1, nr);
        noc_async_atomic_barrier();
#endif
        tr[11] = wall_lo();
        tr[0] = npad;
    }
    noc_async_write(ws + 1024, outb.get_noc_addr(64), OUT_WORDS * 4);
    noc_async_write_barrier();
}
