// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SEL1 on the SFPU (QIX_SFSEL=1), writer (BRISC).  Per job (row i, type), N = NBK row i word 0:
// SHI: N > 512 -> the Hi key from the transport tile (cb 6; elements (4, 0) / (4, 1): 0x3F800000 | half << 7) -> the
//      partner core's cb 3 buffer (same L1 address on every core) + its semaphore SEM.
// FULL / SLO: ix3_sel1.cpp's outputs and list code: N <= 512 -> all in; else the codes (cb 16, bf16 tiles,
//      tile t element (r, c) = key 1,024 t + 32 r + c: 0x4000 in, 0x3F80 band, 0 out) scanned in ascending key order
//      replace the in-place classification (in: kl > Hi; band: !(kh < Lo)), then INB / II / JJ / CNT (+ JO spill
//      staging, OVC) exactly as ix3_sel1.cpp IX3_FAST (+ B4_PFL / IX3_SPILL / IX3_OVC).
// Every job with N > 512 first gathers row i of HI into cb 1 (sf1_gather.hpp, NoC 0).
// Runtime: 0 INB, 1 II, 2 JJ, 3 CNT, 4 NBK, [5 JO (IX3_SPILL)], [next OVC (IX3_OVC)], then HI, NC, NTS, partner x,
// partner y, njobs, S (HI slots), wfirst (1: the core's first gather is the writer's), then (row, type) per job.  CT:
// the accessors in that order (HI after OVC), then SEM (SEM + 1 = the local first-gather semaphore).  Scratch: cb 7.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "sf1_gather.hpp"

void kernel_main() {
    constexpr auto ba = TensorAccessorArgs<0>();
    constexpr auto ia = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto ja = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ca = TensorAccessorArgs<ja.next_compile_time_args_offset()>();
    constexpr auto na = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    const auto INB = TensorAccessor(ba, get_arg_val<uint32_t>(0), 1024);
    const auto II = TensorAccessor(ia, get_arg_val<uint32_t>(1), ix3::CAP * 4);
    const auto JJ = TensorAccessor(ja, get_arg_val<uint32_t>(2), ix3::CAP * 4);
    const auto CNT = TensorAccessor(ca, get_arg_val<uint32_t>(3), 64);
    const auto NBK = TensorAccessor(na, get_arg_val<uint32_t>(4), 64);
#ifdef IX3_SPILL
    constexpr auto joa = TensorAccessorArgs<na.next_compile_time_args_offset()>();
    const auto JO = TensorAccessor(joa, get_arg_val<uint32_t>(5), IX3_JOW * 4);
    constexpr uint32_t OVOFF = joa.next_compile_time_args_offset(), OVR = 6;
#else
    constexpr uint32_t OVOFF = na.next_compile_time_args_offset(), OVR = 5;
#endif
#ifdef IX3_OVC
    constexpr auto ova = TensorAccessorArgs<OVOFF>();
    const auto OVC = TensorAccessor(ova, get_arg_val<uint32_t>(OVR), 64);
    constexpr uint32_t RB0 = OVR + 1, HIOFF = ova.next_compile_time_args_offset();
#else
    constexpr uint32_t RB0 = OVR, HIOFF = OVOFF;
#endif
    constexpr auto hia = TensorAccessorArgs<HIOFF>();
    constexpr uint32_t SEM = get_compile_time_arg_val(hia.next_compile_time_args_offset());
    const auto HI = TensorAccessor(hia, get_arg_val<uint32_t>(RB0), 4096);
    const uint32_t NC = get_arg_val<uint32_t>(RB0 + 1), NTS = get_arg_val<uint32_t>(RB0 + 2);
    constexpr uint32_t RB = RB0 + 3;
    constexpr uint32_t cb_hi = 1, cb_hin = 3, cb_kout = 6, cb_scr = 7, cb_code = 16;
    const uint32_t px = get_arg_val<uint32_t>(RB), py = get_arg_val<uint32_t>(RB + 1),
                   njobs = get_arg_val<uint32_t>(RB + 2);
    const uint32_t S = get_arg_val<uint32_t>(RB + 3), wfirst = get_arg_val<uint32_t>(RB + 4);
    cb_reserve_back(cb_scr, 1);
    const uint32_t base = get_write_ptr(cb_scr);
    uint32_t* ii = reinterpret_cast<uint32_t*>(base);
    uint32_t* jj = ii + ix3::CAP;
    uint32_t* inb = jj + ix3::CAP;
    uint32_t* cnt = inb + 256;
    uint32_t* hw = cnt + 16;  // the Hi word sent to the partner (SHI)
    // N of every job (<= 2 per core) first; HI rows are gathered up to S jobs ahead (S = the CB slots), so a job's hi
    // tiles are in L1 before its codes are needed.  The core's first gather: the writer starts at once when wfirst,
    // else it waits for the reader's first LO gather (local semaphore SEM + 1): one tensor per core in flight at the
    // start halves each NoC's load there (measured: the first gather is on the critical path).
    uint32_t nj[4];
    for (uint32_t q = 0; q < njobs; ++q) {
        noc_async_read(NBK.get_noc_addr(get_arg_val<uint32_t>(RB + 5 + 2 * q)), reinterpret_cast<uint32_t>(cnt), 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        nj[q] = cnt[0];
    }
    volatile tt_l1_ptr uint32_t* sem1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM + 1));
    uint32_t g = 0, ng = 0;
    for (uint32_t q = 0; q < njobs; ++q) {
        const uint32_t i = get_arg_val<uint32_t>(RB + 5 + 2 * q), type = get_arg_val<uint32_t>(RB + 6 + 2 * q);
        for (; g < njobs && g < q + S; ++g) {  // HI rows of jobs g < q + S -> cb 1 (this RISC's NoC 0)
            const uint32_t Ng = nj[g];
            if (Ng <= 512) {
                continue;
            }
            cb_reserve_back(cb_hi, NTS);
            if (ng++ == 0 && !wfirst) {
                noc_semaphore_wait(sem1, 1);
                noc_semaphore_set(sem1, 0);
            }
            sf1g::gather(HI, get_arg_val<uint32_t>(RB + 5 + 2 * g), Ng, (Ng + 1023) >> 10, NC, get_write_ptr(cb_hi));
            if (ng == 1 && wfirst) {
                *sem1 = 1;  // the reader may start its LO gathers
            }
            cb_push_back(cb_hi, NTS);
        }
        const uint32_t N = nj[q];
        if (type == 2) {
            if (N > 512) {
                cb_wait_front(cb_kout, 1);
                volatile tt_l1_ptr uint32_t* k = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_kout));
                const uint32_t w0 = k[ix3::toff(4, 0)], w1 = k[ix3::toff(4, 1)];
                hw[0] = (((w0 >> 7) & 0xFFFFu) << 16) | ((w1 >> 7) & 0xFFFFu);
                cb_pop_front(cb_kout, 1);
                asm volatile("" ::: "memory");
                noc_async_write(reinterpret_cast<uint32_t>(hw), get_noc_addr(px, py, get_write_ptr(cb_hin)), 4);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(SEM)), 1);
                noc_async_atomic_barrier();
            }
            continue;
        }
        for (uint32_t w = 0; w < 256; ++w) {
            inb[w] = 0;
        }
        uint32_t nin = 0, nband = 0, flag = 0;
#ifdef B4_PFL
        uint16_t* inl = reinterpret_cast<uint16_t*>(inb);
        for (uint32_t w = 0; w < 256; ++w) {
            inb[w] = 0xFFFFFFFFu;
        }
        if (N > 0 && N <= 512) {
            for (uint32_t j = 0; j < N; ++j) {
                inl[j] = static_cast<uint16_t>(j);
            }
            nin = N;
#else
        if (N > 0 && N <= 512) {
            for (uint32_t j = 0; j < N; ++j) {
                inb[j >> 5] |= 1u << (j & 31);
            }
            nin = N;
#endif
        } else if (N > 512) {
            const uint32_t NT = (N + 1023) >> 10;
            DeviceZoneScopedN("S1-SCAN");
#ifndef SF1_NOCACHE
            // the L1 data cache (off by default on BH RISCs: firmware_common.h) on for the scan only: plain loads of
            // the code words; it is a read cache, invalidated after each code tile lands (the packer writes L1 behind
            // it)
            set_l1_data_cache<true>();
#endif
            for (uint32_t t = 0; t < NT; ++t) {
                cb_wait_front(cb_code, 1);
                invalidate_l1_cache();  // the codes were packed after any cached copy: plain (cached) loads below
                const uint32_t cbase = get_read_ptr(cb_code);
#ifdef SF1_NOSCAN  // timing-only probe (QIX_SF1_DIAG=noscan): codes not scanned (outputs wrong)
                for (uint32_t r = 0; r < 0; ++r) {
#elif defined(SF1_LOADONLY)  // timing-only probe (QIX_SF1_DIAG=loadonly): every code word loaded + OR-ed, nothing else
                {
                    const uint32_t* cw = reinterpret_cast<const uint32_t*>(cbase);
                    uint32_t o = 0;
                    for (uint32_t k = 0; k < 512; ++k) {
                        o |= cw[k];
                    }
                    hw[1] = o;
                }
                for (uint32_t r = 0; r < 0; ++r) {
#else
            for (uint32_t r = 0; r < 32; ++r) {
#endif
                    for (uint32_t h = 0; h < 2; ++h) {
                        const uint32_t j0 = 1024 * t + 32 * r + 16 * h;
                        if (j0 >= N) {
                            break;
                        }
                        // the 16 codes of keys j0 .. j0 + 15 = 8 words (two bf16 codes each), each word loaded once
                        const uint32_t* cw = reinterpret_cast<const uint32_t*>(cbase + 2 * ix3::toff(r, 16 * h));
                        // nonzero words only; low half = key j0 + 2 k, high half = j0 + 2 k + 1 (ascending order kept)
                        for (uint32_t k = 0; k < 8; ++k) {
                            const uint32_t w2 = cw[k];
                            if (w2 == 0) {
                                continue;
                            }
                            for (uint32_t hf = 0; hf < 2; ++hf) {
                                const uint32_t code = hf ? (w2 >> 16) : (w2 & 0xFFFFu);
                                const uint32_t j = j0 + 2 * k + hf;
                                if (code == 0 || j >= N) {
                                    continue;
                                }
#ifdef SF1_NOSTORE  // timing-only probe (QIX_SF1_DIAG=nostore): codes counted, no list stores (outputs wrong)
                                nin += code == 0x4000u;
                                nband += code != 0x4000u;
                                continue;
#endif
                                if (code == 0x4000u) {
#ifdef B4_PFL
                                    inl[nin] = static_cast<uint16_t>(j);
#else
                                    inb[j >> 5] |= 1u << (j & 31);
#endif
                                    ++nin;
                                } else {
                                    if (nband < ix3::CAP) {
                                        jj[nband] = j;
                                    } else {
                                        flag = 1;
#ifdef IX3_SPILL
                                        const uint32_t e = nband - ix3::CAP;
                                        ii[e & (ix3::CAP - 1)] = j;
                                        if ((e & (ix3::CAP - 1)) == ix3::CAP - 1) {
                                            asm volatile("" ::: "memory");
                                            noc_async_write(
                                                reinterpret_cast<uint32_t>(ii),
                                                JO.get_noc_addr(i, 4 * (e + 1 - ix3::CAP)),
                                                ix3::CAP * 4);
                                            noc_async_write_barrier();
                                        }
#endif
                                    }
                                    ++nband;
                                }
                            }
                        }
                    }
                }
                cb_pop_front(cb_code, 1);
            }
#ifndef SF1_NOCACHE
            set_l1_data_cache<false>();
#endif
#ifdef IX3_SPILL
            if (nband > ix3::CAP) {
                const uint32_t ex = nband - ix3::CAP, r = ex & (ix3::CAP - 1);
                if (r) {
                    asm volatile("" ::: "memory");
                    noc_async_write(
                        reinterpret_cast<uint32_t>(ii), JO.get_noc_addr(i, 4 * (ex - r)), (4 * r + 63) & ~63u);
                    noc_async_write_barrier();
                }
            }
#endif
        }
        DeviceZoneScopedN("S1-TAIL");
        for (uint32_t q2 = 0; q2 < ix3::CAP; ++q2) {
            ii[q2] = i;
            if (q2 >= nband) {
                jj[q2] = 0;
            }
        }
#ifdef IX3_OVC
        if (flag) {
            noc_async_read(OVC.get_noc_addr(i), reinterpret_cast<uint32_t>(cnt), 64);
            noc_async_read_barrier();
            invalidate_l1_cache();
            cnt[0] += 1;
            cnt[1] = cnt[1] > nband ? cnt[1] : nband;
            cnt[2] += nband - ix3::CAP;
            asm volatile("" ::: "memory");
            noc_async_write(reinterpret_cast<uint32_t>(cnt), OVC.get_noc_addr(i), 64);
            noc_async_write_barrier();
        }
#endif
        for (uint32_t w = 0; w < 16; ++w) {
            cnt[w] = 0;
        }
        cnt[0] = nin;
        cnt[1] = nband;
        cnt[2] = flag;
        cnt[3] = N;
        noc_async_write(reinterpret_cast<uint32_t>(inb), INB.get_noc_addr(i), 1024);
        noc_async_write(reinterpret_cast<uint32_t>(ii), II.get_noc_addr(i), ix3::CAP * 4);
        noc_async_write(reinterpret_cast<uint32_t>(jj), JJ.get_noc_addr(i), ix3::CAP * 4);
        noc_async_write(reinterpret_cast<uint32_t>(cnt), CNT.get_noc_addr(i), 64);
        noc_async_write_barrier();
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
