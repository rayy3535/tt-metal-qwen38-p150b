# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""qsa_probe: card micro-benchmarks that size the QSA program.

HeadLoad (head window): N MB of bank-local, ordered DRAM reads issued by the 8 bank streamers of a GR-head program
(moe_layer.MoeLayer(hw=...)) at kernel start / after inj (AG1 done) / after x, to measure what DRAM streaming does to
the AG1 / AG2 / x-multicast windows.  Each streamer reads its share as ONE ordered stream of `pkt`-byte packets from
its own bank (the streamer index b is the bank id: get_optimal_dram_bank_to_logical_worker_assignment order), `infl`
packets in flight, into a discarded ring (kernels/hw_bulk.hpp).  Every knob is a runtime argument (one compile, one
program hash for all variants).  Source data: the first bytes of the layer's gate/up expert tensor in that bank.
"""

from __future__ import annotations

import os

CLOCK_MHZ = 1350


class HeadLoad:
    CB = 13  # free in the MoE program
    RING = 262144  # landing ring bytes (infl * pkt must fit; 16 KB x 15 in flight = full-rate DSM-like streams)

    def __init__(
        self, mb=0.0, gate="t0", noc=0, banks=0xFF, pkt=8192, infl=8, delay_us=0.0, pace=0, cpb=1, odelay_us=0.0
    ):
        self.mb, self.gate, self.noc, self.banks = float(mb), gate, int(noc), int(banks)
        self.pkt, self.infl, self.delay_us, self.pace = int(pkt), int(infl), float(delay_us), int(pace)
        # cpb readers per bank = the QSA compute-core placement (bank_cores: the streamer + cpb - 1 nearest free
        # non-up out cores); the out cores stream on their BRISC (NoC 0) right after grr writer_out (+ odelay_us)
        self.cpb, self.odelay_us = int(cpb), float(odelay_us)
        assert self.gate in ("t0", "inj", "x"), self.gate
        assert 1 <= self.infl <= 15 and self.pkt <= 16384 and self.pkt % 64 == 0
        assert self.infl * self.pkt <= self.RING
        assert 0 < self.banks <= 0xFF
        assert 1 <= self.cpb <= 8 and (self.cpb == 1 or self.noc == 0)

    @classmethod
    def from_env(cls, prefix="QSA_PRE_"):
        e = lambda k, d: os.environ.get(prefix + k, d)  # noqa: E731
        return cls(
            mb=e("MB", "0"),
            gate=e("AT", "t0"),
            noc=e("NOC", "0"),
            banks=int(e("BANKS", "0xff"), 0),
            pkt=e("PKT", "8192"),
            infl=e("INFL", "8"),
            delay_us=e("DELAY", "0"),
            pace=e("PACE", "0"),
            cpb=e("CPB", "1"),
            odelay_us=e("ODELAY", "0"),
        )

    def bytes_per_bank(self):
        n = bin(self.banks).count("1")
        per = int(self.mb * 1e6 / n)
        return (per + self.pkt - 1) // self.pkt * self.pkt

    def per_core(self):
        """Bytes per reader (cpb readers per bank; = bytes_per_bank() for cpb = 1)."""

        return (self.bytes_per_bank() // self.cpb + self.pkt - 1) // self.pkt * self.pkt

    def out_ring(self):
        return self.infl * self.pkt

    def args(self, b, noc, src_addr, vc):
        """9 runtime args for streamer b's RISC on NoC ``noc`` (bytes 0 unless this RISC streams)."""

        on = self.mb > 0 and noc == self.noc and (self.banks >> b) & 1
        return [
            self.per_core() if on else 0,
            {"t0": 0, "inj": 1, "x": 2}[self.gate],
            src_addr,
            b,
            self.pkt,
            self.infl,
            int(self.delay_us * CLOCK_MHZ),
            self.pace,
            vc,
        ]

    def args_out(self, b, k, src_addr, vc):
        """9 runtime args for reader k >= 1 of bank b (an out core's BRISC, NoC 0, after grr writer_out): gate 0 (the
        call site is the gate; SEM 10 belongs to the GR head there), its own slice at src + k * per_core()."""

        on = self.mb > 0 and k < self.cpb and (self.banks >> b) & 1
        per = self.per_core()
        return [per if on else 0, 0, src_addr + k * per, b, self.pkt, self.infl, int(self.odelay_us * CLOCK_MHZ)] + [
            self.pace,
            vc,
        ]

    def __repr__(self):
        return (
            f"HeadLoad(mb={self.mb}, at={self.gate}, noc={self.noc}, banks={self.banks:#04x}, pkt={self.pkt}, "
            f"infl={self.infl}, delay_us={self.delay_us}, pace={self.pace}, per_bank={self.bytes_per_bank()}, "
            f"cpb={self.cpb}, per_core={self.per_core()}, odelay_us={self.odelay_us})"
        )


# ---- one-card DRAM read probe (kernels/rd_probe.cpp)
RD_SRC = "models/demos/blackhole/qwen38_2x262k/qsa_probe/kernels/rd_probe.cpp"


def bank_cores(device, per_bank, card=0, extra=0, up=None, metric=None):
    """QSA compute-core placement: per bank b the bank worker W_b (NoC-0 assignment) + the
    nearest free non-up GR out cores of card ``card`` (logical Manhattan, round-robin over the banks); ``extra`` more
    cores go to banks 0, 1, ...  up: the up cores to avoid (default: card ``card``'s, moe_layer.roles).  Returns [[cores
    of bank b] for b in 0..7].  metric: manhattan (logical, default) | noc0 (NoC-0 response ring distance)."""

    from models.demos.blackhole.qwen38_2x262k import moe_layer as ml

    rl = ml.roles(device, card)
    w = rl.streamers
    up_ = rl.up if up is None else up
    free = [c for c in ml.gf.stream_layout()[3] if c not in up_ and c not in w]
    dist = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])  # noqa: E731
    metric = metric or os.environ.get("QA_PLACE", "manhattan")  # qsa_layer passes QS_PLACE
    assert metric in ("manhattan", "noc0", "row", "row2") or metric.startswith("row-"), metric
    if metric.startswith("row"):
        # QS_PLACE=row: rows first - the endpoint's own row, then the rows below it along NoC 0's +y
        # ring, then the noc0 ring distance.  Measured (QA_KVZ profiling): cores in their bank's DRAM-endpoint row stream
        # their KV in a median 6.9 us vs 12.0 for the other rows (their responses turn into shared columns).
        ph = lambda c: ((c[0] + 1) if c[0] <= 5 else (c[0] + 5), c[1] + 2)  # noqa: E731
        ring = lambda e, c: (ph(c)[0] - (ph(e)[0] - 1)) % 17 + (ph(c)[1] - ph(e)[1]) % 12  # noqa: E731
        # row2: banks 0 / 4 (their endpoints share row 11) keep the noc0 order; row-<banks>: the listed banks do
        keep = {0, 4} if metric == "row2" else ({int(ch) for ch in metric[4:]} if metric.startswith("row-") else set())
        rowb = lambda b: b not in keep  # noqa: E731
        key = lambda c, b: (  # noqa: E731
            ((ph(c)[1] - ph(w[b])[1]) % 12 if rowb(b) else 0, ring(w[b], c), dist(c, w[b]), c)
        )
    elif metric == "noc0":
        # noc0: NoC-0 response distance from the bank's DRAM endpoint (one hop -x of its NoC-0 bank worker) to the
        # core along NoC 0's +x / +y ring (wraps count the whole ring), in physical coordinates (logical x <= 5 -> x + 1,
        # x >= 6 -> x + 5, y + 2: measured from the profiler's core coordinates; 17 x 12 NoC grid)
        ph = lambda c: ((c[0] + 1) if c[0] <= 5 else (c[0] + 5), c[1] + 2)  # noqa: E731
        ring = lambda e, c: (ph(c)[0] - (ph(e)[0] - 1)) % 17 + (ph(c)[1] - ph(e)[1]) % 12  # noqa: E731
        key = lambda c, b: (ring(w[b], c), dist(c, w[b]), c)  # noqa: E731
    else:
        key = lambda c, b: (dist(c, w[b]), c)  # noqa: E731
    out = [[w[b]] for b in range(8)]
    for k in range(per_bank - 1 + (1 if extra else 0)):
        for b in range(8):
            if k == per_bank - 1 and b >= extra:
                break
            c = min(free, key=lambda c: key(c, b))
            free.remove(c)
            out[b].append(c)
    return out


def rd_program(device, groups, buf_addr, mode, pkt, count, infl=8, region=0, sorted_=0, token=0, noc=0, seed=1):
    """groups[b] = cores reading bank b; core k of bank b reads `count` packets (mode 0: the contiguous slice at
    buf_addr + k * count * pkt of that bank; mode 1: random pages of the bank's first `region` pages).  token = 1:
    the cores of a bank issue one after the other (one ordered stream per bank)."""

    import ttnn
    from models.demos.blackhole.qwen38_2x262k.common import program as fp

    cores = [c for g in groups for c in g]
    cs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])
    rt = []
    for b, g in enumerate(groups):
        for k, c in enumerate(g):
            nxt = g[k + 1] if token and k + 1 < len(g) else None
            nv = device.worker_core_from_logical_core(ttnn.CoreCoord(*nxt)) if nxt else None
            addr = buf_addr + (k * count * pkt if mode == 0 else 0)
            rt.append(
                (
                    ttnn.CoreCoord(*c),
                    [mode, b, addr, pkt, count, infl, region, seed + 7919 * (8 * k + b), sorted_]
                    + [int(token and k > 0), nv.x if nv else 0, nv.y if nv else 0, 0, b & 3],
                )
            )
    ring = infl * pkt
    cb = fp.cb_descriptor(0, ttnn.uint32, ring, 1, cs)
    k = fp.reader_kernel(RD_SRC, cs, [], rt)
    k.config = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0 if noc == 0 else ttnn.NOC.NOC_1
    )
    desc = fp.program_descriptor([k], [cb], [fp.semaphore_descriptor(0, cs, 0)])
    return desc, cores
