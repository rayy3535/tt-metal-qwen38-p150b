// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SEL1 on the SFPU (QIX_SFSEL=1), reader (NCRISC, NoC 1).  Per job (row i,
// type): N = NBK row i word 0; params page -> compute (cb 2: N, NT, type); N > 512 and type FULL / SLO: row i of LO
// gathered into cb 0 (sf1_gather.hpp; HI is gathered by the writer), one slot of NTS pages (fixed-size slots: no CB
// wrap).  SLO: then wait for the partner's Hi (semaphore SEM; the word lands in cb 3's buffer) and push cb 3.
// Types: 0 FULL (Lo + Hi + codes here), 1 SLO (Lo here, Hi from the partner core, codes here), 2 SHI (Hi only, sent).
// Runtime: 0 LO, 1 (unused), 2 NBK, 3 NC, 4 NTS, 5 njobs, 6 wfirst (sf1_s1wr.cpp), then (row, type) per job.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "sf1_gather.hpp"

void kernel_main() {
    constexpr auto la = TensorAccessorArgs<0>();
    constexpr auto na = TensorAccessorArgs<la.next_compile_time_args_offset()>();
    constexpr uint32_t SEM = get_compile_time_arg_val(na.next_compile_time_args_offset());
    constexpr uint32_t cb_lo = 0, cb_par = 2, cb_hin = 3, cb_rs = 4;
    const auto LO = TensorAccessor(la, get_arg_val<uint32_t>(0), 4096);
    const auto NBK = TensorAccessor(na, get_arg_val<uint32_t>(2), 64);
    const uint32_t NC = get_arg_val<uint32_t>(3), NTS = get_arg_val<uint32_t>(4), njobs = get_arg_val<uint32_t>(5);
    const uint32_t wfirst = get_arg_val<uint32_t>(6);
    volatile tt_l1_ptr uint32_t* sem1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM + 1));
    uint32_t nlo = 0;
    cb_reserve_back(cb_rs, 1);
    volatile tt_l1_ptr uint32_t* nb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_rs));
    for (uint32_t q = 0; q < njobs; ++q) {
        const uint32_t i = get_arg_val<uint32_t>(7 + 2 * q), type = get_arg_val<uint32_t>(8 + 2 * q);
        noc_async_read(NBK.get_noc_addr(i), reinterpret_cast<uint32_t>(nb), 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        const uint32_t N = nb[0];
        const uint32_t NT = N > 512 ? (N + 1023) >> 10 : 0;
        cb_reserve_back(cb_par, 1);
        volatile tt_l1_ptr uint32_t* par = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_par));
        par[0] = N;
        par[1] = NT;
        par[2] = type;
        cb_push_back(cb_par, 1);
        if (NT == 0) {
            continue;
        }
        if (type != 2) {
            cb_reserve_back(cb_lo, NTS);
            if (nlo++ == 0 && wfirst) {  // the writer's first HI gather goes alone (sf1_s1wr.cpp)
                noc_semaphore_wait(sem1, 1);
                noc_semaphore_set(sem1, 0);
            }
            sf1g::gather(LO, i, N, NT, NC, get_write_ptr(cb_lo));
            if (nlo == 1 && !wfirst) {
                *sem1 = 1;  // the writer may start its HI gathers
            }
            cb_push_back(cb_lo, NTS);
        }
        if (type == 1) {  // SLO: the partner's Hi word lands in cb 3's buffer (offset 0), then the semaphore
            cb_reserve_back(cb_hin, 1);
            DeviceZoneScopedN("S1-HIW");
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM));
            noc_semaphore_wait(sem, 1);
            noc_semaphore_set(sem, 0);
            invalidate_l1_cache();
            cb_push_back(cb_hin, 1);
        }
    }
    noc_async_read_barrier();
}
