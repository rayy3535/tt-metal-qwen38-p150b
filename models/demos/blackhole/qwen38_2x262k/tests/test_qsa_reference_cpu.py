# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of qsa/cache.py and qsa/reference.py (no device, no checkpoint: random weights).

  pytest models/demos/blackhole/qwen38_2x262k/tests/test_qsa_reference_cpu.py
"""

from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_2x262k.qsa import cache as C
from models.demos.blackhole.qwen38_2x262k.qsa import reference as R
from models.demos.blackhole.qwen38_2x262k.ref.tt.qsa import Qwen38QSA, Qwen38QSADimensions, Qwen38QSAWeights

CFG = SimpleNamespace(qsa_rope_dim=64, rope_theta=10_000_000, max_position_embeddings=262_144)


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=0.02: (torch.randn(*s, generator=g) * sc).bfloat16()  # noqa: E731
    dims = Qwen38QSADimensions(2560, 24, 2, 256, 64, 4, 1, 128, 2048, 4)
    # index query heads = the key head + noise, so that (with x sharing a common component) block scores are positive:
    # purely random weights put ~1/16 of the blocks at relu-sum 0, i.e. exact ties at the top-512 boundary where the
    # oracle's torch.topk order is arbitrary (the reference's tie rule is tested separately)
    ik = r(128, 2560)
    index_qk = torch.cat([(ik.float() + 0.3 * r(128, 2560).float()).bfloat16() for _ in range(4)] + [ik])
    return Qwen38QSAWeights(
        layer_idx=3,
        dimensions=dims,
        rms_norm_eps=1e-6,
        qg=r(12288, 2560),
        k=r(512, 2560),
        v=r(512, 2560),
        out=r(2560, 6144),
        q_norm=r(256, sc=0.1),
        k_norm=r(256, sc=0.1),
        index_qk=index_qk,
        index_q_norm=r(128, sc=0.1),
        index_k_norm=r(128, sc=0.1),
    )


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (a.norm() * b.norm()))


def test_bf8_dequant_edges():
    w = torch.tensor([[1.9921875, -1.9921875, 1.0, 0.0] * 4, [3.984375, 1.0, 2.0**-126, 0.0078125] * 4])
    q = R.bf8_dequant(w)
    assert q[0, 0] == 1.984375 and q[0, 1] == -1.984375  # mantissa overflow clamps (7 bits incl. the leading one)
    assert q[1, 0] == 3.96875 and q[1, 2] == 0.0  # tiny value flushed under a large shared exponent


def test_tie_rule():
    s = torch.tensor([[3.0, 1.0, 2.0, 1.0, 1.0, 0.0]])
    assert R.select_blocks(s, 3).tolist() == [[0, 1, 2]]  # equal scores -> lower block id first
    assert R.select_blocks(s, 4).tolist() == [[0, 1, 2, 3]]


@pytest.mark.parametrize("layout", ["tiles", "rows"])
def test_cache_layout(layout):
    g = torch.Generator().manual_seed(1)
    lengths = [130, 7, None, 300]
    k = [torch.randn(2, 400, 256, generator=g).bfloat16() for _ in lengths]
    v = [torch.randn(2, 400, 256, generator=g).bfloat16() for _ in lengths]
    raw = [torch.randn(400, 128, generator=g).bfloat16() for _ in lengths]
    pk = [torch.randn(100, 128, generator=g).bfloat16() for _ in lengths]
    c = C.build_caches(k, v, raw, pk, lengths, capacity_tokens=512, scramble_seed=3, index_layout=layout)
    assert int(c.positions[2]) == C.PARKED and not bool(c.kv_pools[0][: C.PAGES_PER_PT_BLOCK].any())  # null block
    for u, n in enumerate(lengths):
        if n is None:
            continue
        t = torch.arange(n)
        page, row = c.page_table.page_and_row(u, t)
        assert torch.equal(page % C.NUM_BANKS, (t // 4) % C.NUM_BANKS)  # block j in bank j % 8
        for pt in c.page_table.table[u][c.page_table.table[u] > 0].tolist():  # a bank's 2 pages per pt block: 8 KB
            p = torch.arange(16) + 16 * pt
            assert torch.equal(p[8:] // C.NUM_BANKS, p[:8] // C.NUM_BANKS + 1) and torch.equal(p[8:] % 8, p[:8] % 8)
        for card in range(4):
            kk, vv = c.read_kv(card, u, t)
            assert torch.equal(kk, k[u][card // 2, :n]) and torch.equal(vv, v[u][card // 2, :n])
        assert torch.equal(c.read_pooled_keys(u, n // 4), pk[u][: n // 4])
        assert torch.equal(c.raw_ring[u, : n % 4], raw[u][4 * (n // 4) : n])


@pytest.fixture(scope="module")
def oracle_run():
    w = _weights()
    T = 2060
    g = torch.Generator().manual_seed(2)
    x = (0.5 * torch.randn(1, T, 2560, generator=g) + torch.randn(1, 1, 2560, generator=g)).bfloat16()
    cos, sin = C.rope_table(CFG, T + 64)
    mask = torch.where(torch.arange(T)[None, :] <= torch.arange(T)[:, None], 0.0, torch.finfo(torch.float32).min)
    out, st, sel = Qwen38QSA(w).forward(x, (cos[None, :T], sin[None, :T]), mask.view(1, 1, T, T))
    return w, x, out, st, sel, cos, sin


def test_oracle_roundtrip_and_device_chain(oracle_run):
    """ORACLE precision through the device-layout caches reproduces Qwen38QSA (prefill row L - 1, the same math as a
    decode step): exact selection, out within bf16 ulps; DEVICE precision: batched == alone, chain == fresh build."""
    w, x, out, st, sel, cos, sin = oracle_run
    pwo = R.QSAPreparedWeights.from_oracle(w, R.ORACLE)
    pwd = R.QSAPreparedWeights.from_oracle(w, R.DEVICE)
    Ls = (2056, 1, 2052, 5)
    xs = torch.stack([x[0, L - 1] for L in Ls])
    c = torch.arange(2060 // 4) * 4  # pooled keys of the oracle's raw keys at ORACLE precision
    pk = R.pooled_key(
        st.raw_index_keys[0, : 4 * (2060 // 4)].view(-1, 4, 128), pwo.gamma["ik"], pwo.eps, cos[c], sin[c], "bf16"
    )
    caches = C.build_caches(
        [st.keys[0]] * 4,
        [st.values[0]] * 4,
        [st.raw_index_keys[0]] * 4,
        [pk] * 4,
        [L - 1 for L in Ls],
        capacity_tokens=2124,
        scramble_seed=5,
    )
    res = R.decode_step(xs, caches, pwo, cos, sin)
    for u, L in enumerate(Ls):
        assert _pcc(res[u]["out"], out[0, L - 1]) > 0.99999
        if L > R.DENSE_MAX_LEN:
            complete = L // 4
            want = torch.nonzero(sel[0, 0, L - 1, : 4 * complete].view(complete, 4)[:, 0]).flatten()
            assert torch.equal(res[u]["sel_blocks"], want)

    # DEVICE: caches from the device chain's own prefix; batched == alone; 6 steps on self-updated caches == fresh
    def dev_caches(lens, seed):
        cont = [R.prefix_cache_content(x[0, : max(n, 0)], pwd, cos, sin) for n in lens]
        return C.build_caches(
            [q["k_hat"] for q in cont],
            [q["v"] for q in cont],
            [q["raw"] for q in cont],
            [q["pooled"] for q in cont],
            list(lens),
            capacity_tokens=2124,
            scramble_seed=seed,
        )

    L0 = [2047, 3, 2050, 10]
    cd = dev_caches(L0, 7)
    for step in range(6):
        xs = torch.stack([x[0, n + step] for n in L0])
        rb = R.decode_step(xs, cd, pwd, cos, sin)
        if step == 0:
            for u in range(4):
                ca = dev_caches([n if i == u else 0 for i, n in enumerate(L0)], 9)
                ca.positions = torch.tensor([L0[u] if i == u else C.PARKED for i in range(4)])
                ra = R.decode_step(xs, ca, pwd, cos, sin)
                assert torch.equal(ra[u]["out"], rb[u]["out"]) and all(ra[i] is None for i in range(4) if i != u)
    fresh = dev_caches([n + 6 for n in L0], 11)
    for u, n in enumerate(L0):
        t = torch.arange(n + 6)
        assert all(torch.equal(a, b) for a, b in zip(cd.read_kv(1, u, t), fresh.read_kv(1, u, t)))
        assert torch.equal(cd.read_pooled_keys(u, (n + 6) // 4), fresh.read_pooled_keys(u, (n + 6) // 4))
    assert torch.equal(cd.positions, fresh.positions)


@pytest.mark.parametrize("fields", ["f8", "f6"])
def test_radix_select_equals_select_blocks(fields):
    """The device top-512 algorithm (radix rounds on the fp32 bit pattern, exact final bin, (T, id_cut) rule)
    selects exactly reference.select_blocks' set, incl. ties, zeros, -0.0, masked blocks and n = 512 / 513."""
    from models.demos.blackhole.qwen38_2x262k.qsa import indexer as I

    f = I.FIELDS_8 if fields == "f8" else I.FIELDS_6
    g = torch.Generator().manual_seed(11)
    rows = [
        torch.full((2, 8192), 3.25),
        torch.zeros(2, 4096),
        torch.randint(0, 5, (4, 8192), generator=g).float(),
        torch.rand(8, 513, generator=g),
        torch.rand(2, 512, generator=g),
        torch.exp(torch.randn(4, 8192, generator=g) * 20).clamp(max=3e38),
        torch.cat([torch.full((2, 600), 2.0), torch.full((2, 400), 1.0), torch.full((2, 3000), 0.5)], 1),
    ]
    neg0 = torch.rand(2, 2000, generator=g)
    neg0[:, ::7] = -0.0
    masked = torch.rand(2, 4096, generator=g)
    masked[:, 3000:] = -float("inf")
    for s in rows + [neg0, masked]:
        eq, n, _ = I.check_equal(s, fields=f)
        assert eq == n
        if fields == "f8":  # the device's exact scheme (octave window, range shifts, id rounds)
            sel, _ = I.radix_select_device(s)
            ref = torch.zeros_like(sel)
            s2 = s.float().clone()
            s2[~torch.isfinite(s2)] = -1.0
            ref.scatter_(1, R.select_blocks(s2, min(512, s.shape[1])), True)
            ref = torch.where((torch.isfinite(s).sum(-1) <= 512)[:, None], torch.isfinite(s), ref)
            assert bool((sel == ref).all())
