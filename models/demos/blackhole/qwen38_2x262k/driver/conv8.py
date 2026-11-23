# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""(MTP verify layout) The GDN conv history ``conv8`` of one layer: the only description of its byte layout outside the
GDN verify kernel (gdn_fused/kernels/rest_reader.cpp reads / writes it as ``usr * 8 + slot`` pages, record offset
``64 * qkv_page``).

A record is the 64 B of one (user, history slot, qkv tile column c): channels 32 c .. 32 c + 31 in channel order, i.e.
[left 16 channels (32 B), right 16 channels (32 B)] of tile column c.  Users of this module: the host reset
(StepModel.vfy_zero_user), the host seed (StepModel.vfy_seed_conv8), the device commit seed (prefill/kernels/cmv_rd.cpp,
runtime args from rec()) and the device reset (prefill/kernels/rz_wr.cpp, items from user_regions()).

Constraints any other layout must keep (else this interface changes): conv8 is a DRAM-interleaved ROW_MAJOR tensor; a record
is 64 contiguous bytes (two 32 B halves at + 0 / + 32); a record's address is affine in the tile column c."""

from __future__ import annotations

H = 2560  # hidden (qkv) channels
NCOL = H // 32  # qkv tile columns: 80
SLOTS = 8  # history slots per user (position % 8)
USERS = 4
REC = 64  # bytes per record
C8_SHAPE = (1, 1, USERS * SLOTS, H)  # bf16 ROW_MAJOR: one 5,120 B page per (user, slot)
PAGE = C8_SHAPE[-1] * 2  # bytes per page


def rec(u: int, slot: int):
    """-> (pb, ps, ob, os): the record of tile column c of (user u, history slot ``slot``) is at page pb + ps * c, byte offset
    ob + os * c within that page."""
    assert 0 <= u < USERS and 0 <= slot < SLOTS, (u, slot)
    return SLOTS * u + slot, 0, 0, REC


def user_regions(u: int):
    """-> [(first page, pages, offset in page, bytes per page)]: every byte of user u's records (and nothing else)."""
    assert 0 <= u < USERS, u
    return [(SLOTS * u, SLOTS, 0, PAGE)]


def region_bytes(regions, page=PAGE):
    """[(first page, pages, offset, bytes)] -> the sorted list of linear byte ranges [lo, hi) they cover (pages in order)."""
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


def put_records(words16, u: int, slot: int, row16):
    """Host form of the seed: the records of (user u, slot) <- ``row16`` (a [2560] int16 row in channel order), written into
    ``words16`` = one card's conv8 bytes viewed as int16 [pages * PAGE / 2] (flat).  Equal to the plain row write
    ``v[8 u + slot] = row`` for the current layout."""
    pb, ps, ob, os_ = rec(u, slot)
    hw = PAGE // 2
    for c in range(NCOL):
        a = (pb + ps * c) * hw + (ob + os_ * c) // 2
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
