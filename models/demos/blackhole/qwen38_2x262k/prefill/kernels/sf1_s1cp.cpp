// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SEL1 on the SFPU (QIX_SFSEL=1), compute.  Per job (params page cb 2: N, NT, type; NT = 0 -> nothing):
// FULL: Lo = 512th largest key of the lo tiles (cb 0), Hi = of the hi tiles (cb 1), 32-step bisections (sf1_sfpu.hpp,
// tiles re-unpacked to DEST 0..5 every step), keys -> transport tile (DEST 7) -> cb 5 (own use) -> per tile: lo -> DEST
// 0, hi -> DEST 1, keys -> DEST 2, codes (in 2.0 / band 1.0 / out 0) -> cb 16 (bf16) for the writer. SLO: as FULL with
// Hi = the partner's word (cb 3, element 0).  SHI: Hi only -> transport tile -> cb 6 (writer sends it). Full-sync fp32
// DEST (8 tiles); DEST 6 = the bisection state (SCR in sf1_sfpu.hpp).  Runtime: 0 njobs, 1 NTS (slot pages).
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "sf1_sfpu.hpp"
#include "api/debug/kernel_profiler.hpp"  // zones only in profiler builds

using namespace ckernel;

// one 32-step bisection (k = 512) of the NT tiles of cb into state unit UP (0 = Lo, 1 = Hi)
template <int UP>
inline void bisect(const uint32_t cb, const uint32_t NT) {
    DeviceZoneScopedN("S1-BIS");
    MATH((sf1::clear_p<UP>()));
#ifdef SF1_NOBIS  // timing-only probe (QIX_SF1_DIAG=nobis): no bisection steps (outputs wrong)
    return;
#endif
    for (int b = 31; b >= 0; --b) {
        MATH((sf1::step_begin<UP>(b)));
        for (uint32_t t = 0; t < NT; ++t) {
            copy_tile(cb, t, t % 6);
            MATH((sf1::count_tile(t % 6)));
        }
        MATH((sf1::step_end<UP>(512)));
    }
}

inline void pack_keys(const uint32_t cb) {
    MATH((sf1::put_keys(7)));
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(7, cb);
    cb_push_back(cb, 1);
    tile_regs_release();
}

void kernel_main() {
    const uint32_t njobs = get_arg_val<uint32_t>(0), NTS = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_lo = 0, cb_hi = 1, cb_par = 2, cb_hin = 3, cb_key = 5, cb_kout = 6, cb_code = 16;
    compute_kernel_hw_startup(cb_lo, cb_lo, cb_code);
    for (uint32_t q = 0; q < njobs; ++q) {
        cb_wait_front(cb_par, 1);
        const uint32_t NT = read_tile_value(cb_par, 0, 1), type = read_tile_value(cb_par, 0, 2);
        cb_pop_front(cb_par, 1);
        if (NT == 0) {
            continue;
        }
        tile_regs_acquire();
        reconfig_data_format_srca(cb_lo);
        copy_init(cb_lo);
        mul_binary_tile_init();
        if (type != 2) {
            {
                DeviceZoneScopedN("S1-CWAIT");
                cb_wait_front(cb_lo, NTS);
            }
            bisect<0>(cb_lo, NT);
        }
        if (type == 0 || type == 2) {
            {
                DeviceZoneScopedN("S1-CWAIT");
                cb_wait_front(cb_hi, NTS);
            }
            bisect<1>(cb_hi, NT);
        } else {  // SLO: Hi from the partner
            {
                DeviceZoneScopedN("S1-HIN");
                cb_wait_front(cb_hin, 1);
            }
            const uint32_t hv = read_tile_value(cb_hin, 0, 0);
            cb_pop_front(cb_hin, 1);
            MATH((sf1::set_p<1>(hv)));
            cb_wait_front(cb_hi, NTS);
        }
        if (type == 2) {
            pack_keys(cb_kout);
            cb_pop_front(cb_hi, NTS);
            continue;
        }
        pack_keys(cb_key);
        cb_wait_front(cb_key, 1);
        DeviceZoneScopedN("S1-CODE");
        for (uint32_t t = 0; t < NT; ++t) {
            tile_regs_acquire();
            copy_init(cb_lo);
            copy_tile(cb_lo, t, 0);
            copy_tile(cb_hi, t, 1);
            copy_init(cb_key);
            copy_tile(cb_key, 0, 2);
            mul_binary_tile_init();
            MATH((sf1::code_tile(0, 2)));
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_code, 1);
            pack_reconfig_data_format(cb_code);
            pack_tile(0, cb_code);
            cb_push_back(cb_code, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_key, 1);
        cb_pop_front(cb_lo, NTS);
        cb_pop_front(cb_hi, NTS);
    }
}
