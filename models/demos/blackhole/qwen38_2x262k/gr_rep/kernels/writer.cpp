// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gr_rep writer (BRISC).  All CBs sit at the same L1 address on every core, so a sender writes straight into the
// receivers' CB memory and raises their semaphore.
// Role 0 (down core c, member m of stream group g):
//   zero this core's cb_stats tile, raise READY_STAT on the 16 members, wait READY_STAT == 16 (all zeroed);
//   sum of squares (cb_stat_out [0, 0], face 0 row 0) -> row m of every member's cb_stats, raise SEM_STAT;
//   r_g (cb_r [0, 0]) -> tile g of cb_rin on up cores u with u % 16 == m, raise SEM_UP;
//   partial tile j (cb_part, fp32 rows 0 of faces 0 / 1) -> row c of reducer j's cb_red (2 tiles, 64 rows), SEM_PART;
//   reducer j < 10: row 0 of low_j (cb_lowout, bf16) -> tile j of every up core's cb_low, raise SEM_UP;
//   reducer 10: the 4 scalar gate tiles (cb_gates) -> the gates tensor.
// Role 1 (up core): its 2 mixed tiles (cb_out) -> the mixed tensor.
// Compile-time: 0 role; TensorAccessorArgs of the output (gates / mixed).
// Runtime (down): 0 gates, 1 c, 2 member, 3 group, 4 reducer, 5..20 members' NoC xy, 21 n_ups, then n_ups up NoC xy,
// then reducers' NoC xy (11), then all up cores' NoC xy (40).  (up): 0 mixed, 1 first hidden tile.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

namespace {
FORCE_INLINE uint64_t at(uint32_t xy, uint32_t addr) { return get_noc_addr(xy >> 16, xy & 0xFFFF, addr); }
}  // namespace

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t cb_stat_out = get_named_compile_time_arg_val("cb_stat_out");
    constexpr uint32_t cb_stats = get_named_compile_time_arg_val("cb_stats");
    constexpr uint32_t cb_r = get_named_compile_time_arg_val("cb_r");
    constexpr uint32_t cb_rin = get_named_compile_time_arg_val("cb_rin");
    constexpr uint32_t cb_part = get_named_compile_time_arg_val("cb_part");
    constexpr uint32_t cb_red = get_named_compile_time_arg_val("cb_red");
    constexpr uint32_t cb_lowout = get_named_compile_time_arg_val("cb_lowout");
    constexpr uint32_t cb_low = get_named_compile_time_arg_val("cb_low");
    constexpr uint32_t cb_gates = get_named_compile_time_arg_val("cb_gates");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t group = get_named_compile_time_arg_val("group");
    constexpr uint32_t down_nt = get_named_compile_time_arg_val("down_nt");
    constexpr uint32_t up_ht = get_named_compile_time_arg_val("up_ht");
    constexpr uint32_t streams = get_named_compile_time_arg_val("streams");
    constexpr uint32_t reducers = get_named_compile_time_arg_val("reducers");
    constexpr uint32_t up_cores = get_named_compile_time_arg_val("up_cores");
    constexpr uint32_t sem_ready = 0, sem_stat = 1, sem_part = 2, sem_up = 3, sem_go = 4;
    constexpr auto out_args = TensorAccessorArgs<1>();

    if constexpr (role == 1) {
        const auto mixed = TensorAccessor(out_args, get_arg_val<uint32_t>(0));
        const uint32_t first = get_arg_val<uint32_t>(1);
        CircularBuffer out(cb_out);
        out.wait_front(up_ht);
        for (uint32_t hl = 0; hl < up_ht; ++hl) {
            noc_async_write(out.get_read_ptr() + hl * 2048, mixed.get_noc_addr(first + hl), 2048);
        }
        noc_async_write_barrier();
        out.pop_front(up_ht);
        return;
    }

    const auto gates = TensorAccessor(out_args, get_arg_val<uint32_t>(0));
    const uint32_t c = get_arg_val<uint32_t>(1);
    const uint32_t member = get_arg_val<uint32_t>(2);
    const uint32_t grp = get_arg_val<uint32_t>(3);
    const uint32_t reducer = get_arg_val<uint32_t>(4);
    // NoC tables in the common runtime args (one copy per kernel, not per core): the reducers', the up cores', then
    // the down cores' packed (x << 16 | y).
    constexpr uint32_t reducers_arg = 0;
    constexpr uint32_t all_ups_arg = reducers;
    constexpr uint32_t downs_arg = reducers + up_cores;
    const uint32_t n_ups = member < up_cores ? (up_cores - member + group - 1) / group : 0;
    const uint32_t ready = get_semaphore(sem_ready), stat = get_semaphore(sem_stat), part_sem = get_semaphore(sem_part),
                   up_sem = get_semaphore(sem_up);

    // 1. zero cb_stats (rows 16..31 must read 0), tell the members
    const uint32_t stats_l1 = CircularBuffer(cb_stats).get_write_ptr();
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_l1);
        for (uint32_t k = 0; k < 1024; ++k) {
            z[k] = 0;
        }
    }
    for (uint32_t m = 0; m < group; ++m) {
        noc_semaphore_inc(at(get_common_arg_val<uint32_t>(downs_arg + grp * group + m), ready), 1);
    }
    {
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready);
        noc_semaphore_wait(s, group);
        noc_semaphore_set(s, 0);
    }

    // 2. sum of squares -> row `member` of every member's cb_stats
    {
        CircularBuffer so(cb_stat_out);
        so.wait_front(1);  // follows this core's reads: let up core c % 40 start its weight reads
        if (c < up_cores) {
            noc_semaphore_inc(at(get_common_arg_val<uint32_t>(all_ups_arg + c), get_semaphore(sem_go)), 1);
        }

        const uint32_t src = so.get_read_ptr();
        for (uint32_t m = 0; m < group; ++m) {
            noc_async_write(
                src, at(get_common_arg_val<uint32_t>(downs_arg + grp * group + m), stats_l1 + member * 64), 64);
        }
        noc_async_write_barrier();
        for (uint32_t m = 0; m < group; ++m) {
            noc_semaphore_inc(at(get_common_arg_val<uint32_t>(downs_arg + grp * group + m), stat), 1);
        }
        so.pop_front(1);
    }

    // 3. r_g -> tile g of cb_rin on this member's up cores
    {
        CircularBuffer r(cb_r);
        r.wait_front(1);
        const uint32_t src = r.get_read_ptr();
        const uint32_t dst = CircularBuffer(cb_rin).get_write_ptr() + grp * 4096;
        for (uint32_t i = 0; i < n_ups; ++i) {
            noc_async_write(src, at(get_common_arg_val<uint32_t>(all_ups_arg + member + i * group), dst), 64);
        }
        noc_async_write_barrier();
        for (uint32_t i = 0; i < n_ups; ++i) {
            noc_semaphore_inc(at(get_common_arg_val<uint32_t>(all_ups_arg + member + i * group), up_sem), 1);
        }
        r.pop_front(1);
    }

    // 4. partial row c of each of the 11 N tiles -> reducer j's cb_red
    {
        CircularBuffer pt(cb_part);
        pt.wait_front(down_nt);
        const uint32_t src = pt.get_read_ptr();
        const uint32_t red_l1 = CircularBuffer(cb_red).get_write_ptr();
        const uint32_t t = c / 32, row = c % 32;
        const uint32_t dst0 = red_l1 + t * 4096 + (row / 16) * 2 * 1024 + (row % 16) * 64;
        for (uint32_t j = 0; j < down_nt; ++j) {
            const uint32_t xy = get_common_arg_val<uint32_t>(reducers_arg + j);
            noc_async_write(src + j * 4096, at(xy, dst0), 64);
            noc_async_write(src + j * 4096 + 1024, at(xy, dst0 + 1024), 64);
        }
        noc_async_write_barrier();
        for (uint32_t j = 0; j < down_nt; ++j) {
            noc_semaphore_inc(at(get_common_arg_val<uint32_t>(reducers_arg + j), part_sem), 1);
        }
        pt.pop_front(down_nt);
    }

    // 5. reducers.  Every exit flushes the semaphore atomics: a kernel that ends with un-acked non-posted atomics
    // leaves the NoC counters dirty for the next launch (hung traced replay; watcher: missing atomics barrier).
    if (reducer == 255) {
        noc_async_atomic_barrier();
        return;
    }
    if (reducer < reducers - 1) {
        CircularBuffer lo(cb_lowout);
        lo.wait_front(1);
        const uint32_t src = lo.get_read_ptr();
        const uint32_t dst = CircularBuffer(cb_low).get_write_ptr() + reducer * 2048;
        for (uint32_t u = 0; u < up_cores; ++u) {
            const uint32_t xy = get_common_arg_val<uint32_t>(all_ups_arg + u);
            noc_async_write(src, at(xy, dst), 32);
            noc_async_write(src + 512, at(xy, dst + 512), 32);
        }
        noc_async_write_barrier();
        for (uint32_t u = 0; u < up_cores; ++u) {
            noc_semaphore_inc(at(get_common_arg_val<uint32_t>(all_ups_arg + u), up_sem), 1);
        }
        lo.pop_front(1);
    } else {
        CircularBuffer g(cb_gates);
        g.wait_front(1);
        noc_async_write(g.get_read_ptr(), gates.get_noc_addr(0), 64);  // row 0 cols 0..15 (inj_b at col b)
        noc_async_write_barrier();
        g.pop_front(1);
    }
    noc_async_atomic_barrier();
}
