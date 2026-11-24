# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""(MTP verify layout) The GDN conv history ``conv8`` of one layer: the only description of its byte layout outside the
GDN verify kernel (gdn_fused/kernels/rest_reader.cpp: the L0 text, or the coefficients of kernel_defines()).

A record is the 64 B of one (user, history slot, qkv tile column c): channels 32 c .. 32 c + 31 in channel order, i.e.
[left 16 channels (32 B), right 16 channels (32 B)] of tile column c.  Users of this module: the host reset
(StepModel.vfy_zero_user), the host seed (StepModel.vfy_seed_conv8), the device commit seed (prefill/kernels/cmv_rd.cpp,
runtime args from rec()), the device reset (prefill/kernels/rz_wr.cpp, items from user_regions()), the allocation
(StepModel: C8_SHAPE) and the GDN verify build (gdn_fused: kernel_defines()).

Layouts (switch VF2_C8, read at the first use in the process - the model load - then fixed; default "0"; the server's MTP
mode sets "b" through serve/mtp_serve.py MTP_ENV, an explicit value wins):
  "0"      L0 [1, 1, 32, 2560]: one 5,120 B page per (user, slot); page 8 u + slot, record at byte 64 c.  A DRAM-interleaved
           page i sits in bank i % 8, so every record of a (user, slot) is in bank slot (= position % 8).
  "a"/"b"  L4 [1, 1, 320, 256]: one 512 B page per (user, tile column), slot-minor; page 80 u + c, record at byte 64 slot, so
           a record's bank is c % 8 for every user and slot.  "a": the verify reader reads 64 B records (addresses only);
           "b": it reads one 512 B page per (user, column) and takes the lag records from it in L1.
Every layout holds the same record bytes; only their addresses differ (the GDN outputs are bitwise the same).

Constraints any other layout must keep (else this interface changes): conv8 is a DRAM-interleaved ROW_MAJOR tensor; a record
is 64 contiguous bytes (two 32 B halves at + 0 / + 32); a record's address is affine in the tile column c; a user's records
are whole contiguous pages (user_regions)."""

from __future__ import annotations

import os
from dataclasses import dataclass

H = 2560  # hidden (qkv) channels
NCOL = H // 32  # qkv tile columns: 80
SLOTS = 8  # history slots per user (position % 8)
USERS = 4
REC = 64  # bytes per record


@dataclass(frozen=True)
class Layout:
    name: str  # the VF2_C8 value
    shape: tuple  # bf16 ROW_MAJOR, DRAM-interleaved
    pu: int  # a record's page = pu u + ps_ slot + pc c (pu = pages per user: users are contiguous)
    ps_: int
    pc: int
    ou: int  # its byte offset in that page = ou u + os_ slot + oc c
    os_: int
    oc: int
    pageread: bool  # verify reader: one page read per (user, column) instead of one record read per (row, lag)

    @property
    def page(self):
        return self.shape[-1] * 2


LAYOUTS = {
    "0": Layout("0", (1, 1, USERS * SLOTS, H), SLOTS, 1, 0, 0, 0, REC, False),
    "a": Layout("a", (1, 1, USERS * NCOL, SLOTS * REC // 2), NCOL, 0, 1, 0, REC, 0, False),
    "b": Layout("b", (1, 1, USERS * NCOL, SLOTS * REC // 2), NCOL, 0, 1, 0, REC, 0, True),
}
_ACTIVE = {}


def layout() -> Layout:
    """The process's conv8 layout: VF2_C8 read at the first call (the model load), then fixed.  A later different VF2_C8 in
    the environment is an error: the device tensors and the compiled verify kernels already use the first value."""
    v = os.environ.get("VF2_C8", "0")
    if v not in LAYOUTS:
        raise ValueError(f"VF2_C8={v!r}: only {sorted(LAYOUTS)}")
    if "L" not in _ACTIVE:
        _ACTIVE["L"] = LAYOUTS[v]
    elif not _ACTIVE.get("pinned") and _ACTIVE["L"].name != v:
        raise RuntimeError(f"VF2_C8 changed from {_ACTIVE['L'].name!r} to {v!r} after the conv8 layout was fixed")
    return _ACTIVE["L"]


def use(name: str):
    """(tests) Fix the layout explicitly; VF2_C8 is ignored from then on."""
    _ACTIVE.clear()
    _ACTIVE["L"], _ACTIVE["pinned"] = LAYOUTS[name], True


def __getattr__(name):  # C8_SHAPE / PAGE of the active layout (module attributes, resolved at the first use)
    if name == "C8_SHAPE":
        return layout().shape
    if name == "PAGE":
        return layout().page
    raise AttributeError(name)


def kernel_defines() -> dict:
    """The GDN verify reader's (rest_reader.cpp) conv8 defines for the active layout: none for L0 (its kernel code is the
    original L0 text, #ifndef C8_L4); L4: page = C8_PU u + c, byte = C8_OS slot (+ C8_PAGEREAD for read mode b)."""
    L = layout()
    if L.name == "0":
        return {}
    assert (L.ps_, L.pc, L.ou, L.oc) == (0, 1, 0, 0), L
    d = {"C8_L4": "1", "C8_PU": str(L.pu), "C8_OS": str(L.os_)}
    if L.pageread:
        d["C8_PAGEREAD"] = "1"
    return d


def rec(u: int, slot: int):
    """-> (pb, ps, ob, os): the record of tile column c of (user u, history slot ``slot``) is at page pb + ps * c, byte offset
    ob + os * c within that page."""
    assert 0 <= u < USERS and 0 <= slot < SLOTS, (u, slot)
    L = layout()
    return L.pu * u + L.ps_ * slot, L.pc, L.ou * u + L.os_ * slot, L.oc


def user_regions(u: int):
    """-> [(first page, pages, offset in page, bytes per page)]: every byte of user u's records (and nothing else)."""
    assert 0 <= u < USERS, u
    L = layout()
    return [(L.pu * u, L.pu, 0, L.page)]


def region_bytes(regions, page=None):
    """[(first page, pages, offset, bytes)] -> the sorted list of linear byte ranges [lo, hi) they cover (pages in order)."""
    page = layout().page if page is None else page
    out = []
    for p0, n, off, nb in regions:
        assert 0 <= off and off + nb <= page, (off, nb, page)
        if off == 0 and nb == page:
            out.append((p0 * page, (p0 + n) * page))
        else:
            out += [((p0 + i) * page + off, (p0 + i) * page + off + nb) for i in range(n)]
    return merge(out)


def merge(ranges):
    """Sorted, touching / overlapping ranges merged."""
    out = []
    for lo, hi in sorted(ranges):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def rec_addr(u: int, slot: int, c: int) -> int:
    """Linear byte address of the record (u, slot, c) in the tensor's bytes (pages in order)."""
    pb, ps, ob, os_ = rec(u, slot)
    return (pb + ps * c) * layout().page + ob + os_ * c


def put_records(words16, u: int, slot: int, row16):
    """Host form of the seed: the records of (user u, slot) <- ``row16`` (a [2560] int16 row in channel order), written into
    ``words16`` = one card's conv8 bytes viewed as int16 [pages * PAGE / 2] (flat; a torch tensor or a list).  Equal to the
    plain row write ``v[8 u + slot] = row`` for L0; for L4 one strided write of a torch tensor (else per record)."""
    L = layout()
    if L.name != "0" and hasattr(words16, "view") and hasattr(row16, "reshape"):
        words16.view(USERS * NCOL, SLOTS, REC // 2)[L.pu * u : L.pu * u + NCOL, slot] = row16.reshape(NCOL, REC // 2)
        return
    for c in range(NCOL):
        a = rec_addr(u, slot, c) // 2
        words16[a : a + REC // 2] = row16[32 * c : 32 * c + 32]


def seed_args(u: int, p: int):
    """(P1K_SEED) cmv_rd's record args for the commit before the step feeding position p: lag i (i = 0..2) = position
    q = p - 3 + i, written iff q >= 0, to (user u, slot q % 8).  -> (lag mask, (pb_0..2), (ob_0..2), ps, os)."""
    mask, pb, ob = 0, [0, 0, 0], [0, 0, 0]
    _, ps, _, os_ = rec(u, 0)
    for i in range(3):
        q = p - 3 + i
        if q < 0:
            continue
        b, s_, o, t_ = rec(u, q % SLOTS)
        assert (s_, t_) == (ps, os_), "record strides must not depend on the slot"
        mask |= 1 << i
        pb[i], ob[i] = b, o
    return mask, tuple(pb), tuple(ob), ps, os_
