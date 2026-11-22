# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""KV-only MTP prefill rows (MTP_PF=1, with the MTP head on).

After layer 47's MoE of every prefill chunk the chain holds the layer-47 residual of the chunk's rows in RA, hidden-split (card
d = columns 640 d .. 640 d + 639 of each of the 4 branches; prefill/gr_tp.py).  Row p of the chunk (position p, p <= L - 2)
gets the MTP layer's cache-state writes for the input fuse(emb(t_(p+1)), h_p) - the token-path alignment row's state writes
without its attention / MoE / tail (the MTP layer's prefill output is never read; row L - 1 stays the shadow alignment row
after the forced decode step):

  N   per card: sq = [h_loc | e_loc]^2 (fp32) . IND -> s [512, 128] fp32 (col 0 = sum over the 2,560 local h columns, col 1 =
      over the 640 local e columns; e = the embedding rows of t_(p+1), written by the host hidden-split like layer 0's input)
  S   REX "ss" all-gather of s -> [512, 512] (the chain's Rex "ss": its own start fences, no ttnn CCL)
  N2  r_h = rsqrt(((s0 + s1) + s2) + s3) / 10,240 + eps), r_e likewise over 2,560 (fp32, card order); hn_b / en = (x r) (1 + w)
      (fp32, then bf16: zero_centered_rms_norm's order) -> 5 bf16 [512, 640] operands U
  X   REX "x" all-gather of each U -> [512, 2,560] (hn_b of the 4 branches, en: the full K on every card)
  F   card d's 640 output columns: x_b = bf16(bf16(hn_b . Wh_d) + bf16(en . We_d)) (MTP_FUSE_RND=2 structure; BF8 weights,
      HiFi4, fp32 accumulation) -> the MTP residual chunk, hidden-split, in the chain's RB (free after layer 47's MoE)
  G   GrRead2 with the MTP attn hyper-connection weights (RB -> x on every card)
  P   QsaRing.state_chunk(x) with the MTP QSA weights (kv_only): QP + QN ring form into the MTP head's own KV pool / mirror /
      card-u ipool / raw ring through decode's page table, the other cards' writes into this object's key copy KC (16 MiB)

No QA / QO / MoE: so the MTP QSA's shift-floor deviation (MTP gammas 7.3; attention only) does not arise here.  The fusion glue is
ttnn ops (prefill sublayers are host-dispatched like the chain's QSA sublayers); every buffer a cached program is keyed on (the
REX inputs / outputs, RB) is preallocated and written through optional output tensors.  Long mode only."""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.prefill import gr_tp
from models.demos.blackhole.qwen38_2x262k.prefill import qsa_ring as QR
from models.demos.blackhole.qwen38_2x262k.prefill.gr import gr_own as GO

MTP_PF = os.environ.get("MTP_PF", "0")
# MTP_PF_FUSE: "2" (default) = fuse_v2 (less DRAM) - h / e squared separately in place (no [512, 3200] fp32 cat /
# square temporaries), U computed per branch into one buffer, one XO; "1" = fuse_v1 (5 U, 2 XO, full IND), kept as
# the bitwise reference.  MTP_PF_CHECK=1: both forms per chunk, RB compared bitwise (host read; check runs only).
MTP_PF_FUSE = os.environ.get("MTP_PF_FUSE", "2")
MTP_PF_CHECK = os.environ.get("MTP_PF_CHECK", "0")
BF = torch.bfloat16
H, HL, CARDS, C = 2560, 640, 4, 512
DR = ttnn.DRAM_MEMORY_CONFIG


class _MtpCk:
    """The checkpoint seen through a prefix map: backbone layer 47 (a full_attention layer: shape / type checks) -> the MTP
    layer's attention tensors (mtp.layers.0.self_attn.*)."""

    def __init__(self, ck):
        self.ck, self.config = ck, ck.config

    def tensor(self, name):
        return self.ck.tensor(name.replace("model.language_model.layers.47.self_attn.", "mtp.layers.0.self_attn."))


def mtp_qsa_weights(ck):
    from models.demos.blackhole.qwen38_2x262k.ref.tt.qsa import Qwen38QSAWeights

    W = Qwen38QSAWeights.from_checkpoint(_MtpCk(ck), 47)
    assert torch.equal(W.k, ck.tensor("mtp.layers.0.self_attn.k_proj.weight"))
    return W


class MtpPrefill:
    def __init__(self, mesh, ck, chain, head, rex_x, eps):
        """chain = the driver's PrefillChain (decode's page table, the L1 fence sets, RA / RB); head = the MtpHead (its
        kv / mir / ipool / ring); rex_x = a Rex(mesh, "x") created before any capture (its start-fence semaphores are L1).
        """
        from models.demos.blackhole.qwen38_2x262k.ref.config import Qwen38Placement
        from models.demos.blackhole.qwen38_2x262k.ref.tt.gr import Qwen38GatedResidualWeights

        self.mesh, self.chain, self.eps = mesh, chain, eps
        cfg = ck.config
        q47 = chain.L[47]
        assert q47.kind == "Q"
        dst = chain.dm.L[47].t
        st = {"kv": head.t["kv"], "mir": head.t["mir"], "ip": head.t["ipool"], "ring": head.t["ring"]}
        if (
            "ring8" in head.t
        ):  # (MTP3_VFY) the verify family's 8-slot ring, written on every card with MTP3_IPALL
            st["ring8"] = head.t["ring8"]
        for k, dk in (("kv", "kv"), ("mir", "mir"), ("ip", "ipool"), ("ring", "ring")):
            assert tuple(st[k].shape) == tuple(dst[dk].shape) and st[k].dtype == dst[dk].dtype, (
                k,
                st[k].shape,
                dst[dk].shape,
            )
        pl0 = Qwen38Placement(cfg, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))
        grw = Qwen38GatedResidualWeights.from_mtp_checkpoint(ck, pl0, block="attn")
        sets = chain.sets
        self.A = QR.QsaRing(mesh, mtp_qsa_weights(ck), grw, eps=eps, cap=q47.A.cap, pt=chain.pt, cos=q47.A.cos,
                            sin=q47.A.sin, u=chain.u, share={"r": sets["r"], "w": sets["w"], "rex": sets["rex"],
                                                             "rout": chain.RB}, state=st, kv_only=True)  # fmt: skip
        self.rex_ss, self.rex_x = sets["rex"], rex_x
        # DRAM reduction:
        # (a) QsaQPS reads only its packed stream S after prepare(): the BF8 Wqp it was packed from is dead (11.49 MB / card)
        ttnn.deallocate(self.A.Wqp)
        self.A.Wqp = None
        # (b) the GrOwn behind GrRead2 / WriteRS: only w1b / E / one / w_dn / w_up / F are read (gr/g2.py:319, gr/wr.py); its
        #     g1 scratch and the fp32 w1 are never read here (13.8 MB / card)
        g1 = self.A.gr2.g1
        for n_ in ("ssp", "xn", "lop", "low", "inj", "xl", "rout", "ssg", "log", "x", "w1"):
            if isinstance(getattr(g1, n_), GO.NoScratch):  # not allocated (GR_SCRATCH=0)
                continue
            ttnn.deallocate(getattr(g1, n_))
            setattr(g1, n_, None)
        # (c) the per-chunk QSA scratch (QP out P, QN outs QH / KH / IQ, RoPE tiles CS) aliased onto layer 47's QsaRing: written
        #     before every read in both; the MTP rows run after layer 47's MoE, layer 47 next writes them in the next chunk
        a47 = q47.A
        for n_ in ("P", "QH", "KH", "IQ", "CS"):
            mine, theirs = getattr(self.A, n_), getattr(a47, n_)
            assert tuple(mine.shape) == tuple(theirs.shape) and mine.dtype == theirs.dtype, n_
            ttnn.deallocate(mine)
            setattr(self.A, n_, theirs)
        rep, shard = ttnn.ReplicateTensorToMesh(mesh), ttnn.ShardTensorToMesh(mesh, dim=0)

        def th(v, dt=ttnn.bfloat16, m=rep):
            return ttnn.from_torch(v, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DR, mesh_mapper=m)

        # fusion weights, card d = its 640 output columns: [K = 2,560 in, 640 out] BF8 (the decode fusion's format)
        fh, fe = ck.tensor("mtp.fc_hidden.weight").t(), ck.tensor("mtp.fc_embedding.weight").t()  # [in, out]
        cols = lambda w: torch.stack(
            [w[:, HL * d : HL * (d + 1)].contiguous().view(1, H, HL) for d in range(CARDS)]
        )  # noqa: E731
        self.Wh, self.We = th(cols(fh), ttnn.bfloat8_b, shard), th(cols(fe), ttnn.bfloat8_b, shard)
        gh = 1.0 + ck.tensor("mtp.pre_fc_norm_hidden.weight").float()  # [10240]
        ge = 1.0 + ck.tensor("mtp.pre_fc_norm_embedding.weight").float()  # [2560]
        assert gh.shape == (4 * H,) and ge.shape == (H,)
        # (1 + w) of the 5 local units of card d: branch b's columns 2560 b + 640 d .., then e's 640 d ..
        g = [
            [gh[H * b + HL * d : H * b + HL * (d + 1)] for b in range(4)] + [ge[HL * d : HL * (d + 1)]]
            for d in range(CARDS)
        ]
        self.G = [
            th(torch.stack([g[d][i].view(1, 1, HL) for d in range(CARDS)]), ttnn.float32, shard) for i in range(5)
        ]
        v1 = MTP_PF_FUSE == "1" or MTP_PF_CHECK == "1"
        if v1:
            ind = torch.zeros(4 * HL + HL, 128)
            ind[: 4 * HL, 0], ind[4 * HL :, 1] = 1.0, 1.0
            self.IND = th(ind.view(1, 1, -1, 128), ttnn.float32)
        # fuse_v2: col 0 = the h sum (IND_h), col 1 = the e sum (IND_e)
        ih, ie = torch.zeros(4 * HL, 128), torch.zeros(HL, 128)
        ih[:, 0], ie[:, 1] = 1.0, 1.0
        self.IND_h, self.IND_e = th(ih.view(1, 1, -1, 128), ttnn.float32), th(ie.view(1, 1, -1, 128), ttnn.float32)
        sc = torch.zeros(1, 1, 1, 32)
        sc[..., 0], sc[..., 1] = 1.0 / (4 * H), 1.0 / H
        self.SC = th(sc, ttnn.float32)
        self.E = th(torch.zeros(CARDS, 1, C, HL, dtype=BF), m=shard)
        self.SB = th(torch.zeros(1, 1, C, 128), ttnn.float32)
        self.AGO = th(torch.zeros(1, 1, C, 512), ttnn.float32)
        if v1:
            self.U = [th(torch.zeros(1, 1, C, HL, dtype=BF)) for _ in range(5)]
            self.XO = [
                th(torch.zeros(1, 1, C, H, dtype=BF)) for _ in range(2)
            ]  # en (kept for the 4 branches), hn_b (reused)
        self.U1 = th(torch.zeros(1, 1, C, HL, dtype=BF))  # fuse_v2: one U, written per branch right before its gather
        self.XO1 = th(torch.zeros(1, 1, C, H, dtype=BF))  # fuse_v2: one XO (en, then each hn_b)
        self.chk = dict(chunks=0, equal=0, first_bad=None)
        self.hi4 = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.emb = None  # host [vocab, 2560] bf16 (set by the driver: the PLE host's table, the MTP's own embedding)
        self.n_chunks = 0

    def put_e(self, ids_next: torch.Tensor):
        """host: the chunk's e rows = emb(t_(p+1)) for its V rows (ids_next [V]), pad rows 0, hidden-split -> E."""
        V = int(ids_next.shape[0])
        e = torch.zeros(C, H, dtype=BF)
        e[:V] = self.emb[ids_next.long()]
        h = torch.stack([e[:, HL * d : HL * (d + 1)].contiguous().view(1, C, HL) for d in range(CARDS)])
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                                        mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0)), self.E)  # fmt: skip

    def fuse(self, RA, out):
        if MTP_PF_FUSE == "1":
            return self.fuse_v1(RA, out)
        return self.fuse_v2(RA, out)

    def fuse_v2(self, RA, out):
        """fuse_v1's arithmetic with ~5 MB of transients instead of ~13 MB and 5.2 MB fewer persistent buffers:
        N as two matmuls (h [512, 2560] . IND_h, e [512, 640] . IND_e; the K = 3,200 one adds exact zeros for the other part,
        split tile-aligned at 2,560) + an exact add, each part typecast (exact) and squared in place; U_b from the RA / E slice
        (typecast exact: the values of fuse_v1's cf slices), one U and one XO reused (the REX start fence orders a gather into
        a buffer after the receiver's previous reader, as fuse_v1's XO[1] reuse)."""
        sl = lambda t, c0, c1: ttnn.slice(t, [0, 0, 0, c0], [1, 1, C, c1])  # noqa: E731
        sh = []
        for src, ind in ((RA, self.IND_h), (self.E, self.IND_e)):
            f = ttnn.typecast(src, ttnn.float32)
            ttnn.multiply(f, f, output_tensor=f)
            sh.append(ttnn.matmul(f, ind, compute_kernel_config=self.hi4, dtype=ttnn.float32))
            ttnn.deallocate(f)
        ttnn.add(sh[0], sh[1], output_tensor=self.SB)
        for t in sh:
            ttnn.deallocate(t)
        self.rex_ss.run(self.SB, self.AGO)
        tot = sl(self.AGO, 0, 32)
        for c in range(1, CARDS):  # ((s0 + s1) + s2) + s3, fp32 (cols 0 = h, 1 = e of each card's 128-column slot)
            tot = ttnn.add(tot, sl(self.AGO, 128 * c, 128 * c + 32))
        r = ttnn.rsqrt(ttnn.add(ttnn.multiply(tot, self.SC), self.eps))
        rh, re = sl(r, 0, 1), sl(r, 1, 2)

        def unit(i):
            src = sl(RA, HL * i, HL * (i + 1)) if i < 4 else self.E
            xi = ttnn.multiply(ttnn.typecast(src, ttnn.float32), rh if i < 4 else re)
            ttnn.multiply(xi, self.G[i], dtype=ttnn.bfloat16, output_tensor=self.U1)
            ttnn.deallocate(xi)

        unit(4)
        self.rex_x.run(self.U1, self.XO1)
        ep = ttnn.matmul(self.XO1, self.We, compute_kernel_config=self.hi4, dtype=ttnn.bfloat16)
        xs = []
        for b in range(4):
            unit(b)
            self.rex_x.run(self.U1, self.XO1)
            hp = ttnn.matmul(self.XO1, self.Wh, compute_kernel_config=self.hi4, dtype=ttnn.bfloat16)
            xs.append(ttnn.add(hp, ep, dtype=ttnn.bfloat16))
            ttnn.deallocate(hp)
        cx = ttnn.concat(xs, dim=3)
        ttnn.copy(cx, out)
        ttnn.deallocate(cx)
        for t in xs + [ep]:
            ttnn.deallocate(t)
        return out

    def fuse_v1(self, RA, out):
        """N, S, N2, X, F: the MTP residual chunk (hidden-split) into ``out`` (the chain's RB)."""
        cat = ttnn.concat(
            [RA, self.E], dim=3
        )  # [512, 3200] bf16: 4 branches x 640 local h columns, 640 local e columns
        cf = ttnn.typecast(cat, ttnn.float32)
        ttnn.deallocate(cat)
        sq = ttnn.multiply(cf, cf)
        ttnn.matmul(sq, self.IND, compute_kernel_config=self.hi4, dtype=ttnn.float32, optional_output_tensor=self.SB)
        ttnn.deallocate(sq)
        self.rex_ss.run(self.SB, self.AGO)
        sl = lambda t, c0, c1: ttnn.slice(t, [0, 0, 0, c0], [1, 1, C, c1])  # noqa: E731
        tot = sl(self.AGO, 0, 32)
        for c in range(1, CARDS):  # ((s0 + s1) + s2) + s3, fp32 (cols 0 = h, 1 = e of each card's 128-column slot)
            tot = ttnn.add(tot, sl(self.AGO, 128 * c, 128 * c + 32))
        r = ttnn.rsqrt(ttnn.add(ttnn.multiply(tot, self.SC), self.eps))
        rh, re = sl(r, 0, 1), sl(r, 1, 2)
        for i in range(5):
            xi = ttnn.multiply(sl(cf, HL * i, HL * (i + 1)), rh if i < 4 else re)
            ttnn.multiply(xi, self.G[i], dtype=ttnn.bfloat16, output_tensor=self.U[i])
        ttnn.deallocate(cf)
        self.rex_x.run(self.U[4], self.XO[0])
        ep = ttnn.matmul(self.XO[0], self.We, compute_kernel_config=self.hi4, dtype=ttnn.bfloat16)
        xs = []
        for b in range(4):  # XO[1] reused per branch (one command queue: the matmul has read it before the next gather)
            self.rex_x.run(self.U[b], self.XO[1])
            hp = ttnn.matmul(self.XO[1], self.Wh, compute_kernel_config=self.hi4, dtype=ttnn.bfloat16)
            xs.append(ttnn.add(hp, ep, dtype=ttnn.bfloat16))
            ttnn.deallocate(hp)
        # ttnn.concat takes no output tensor (TT_FATAL concat.cpp:538): concat, then copy into the chain's RB
        # (untraced prefill sublayer: the temporary is allowed; +1 [512, 2560] bf16 copy per chunk, estimated ~10 us)
        cx = ttnn.concat(xs, dim=3)
        ttnn.copy(cx, out)
        ttnn.deallocate(cx)
        for t in xs + [ep]:
            ttnn.deallocate(t)
        return out

    def run(self, k: int, ids_next: torch.Tensor):
        """chunk k (p0 = 512 k), ids_next = t_(p+1) of its V rows: the MTP state writes of rows p0 .. p0 + V - 1 (after layer
        47's MoE of the chunk, in queue order)."""
        V = int(ids_next.shape[0])
        A = self.A
        A.u = self.chain.u
        A.set_chunk(C * k, V, state_only=True)
        self.put_e(ids_next)
        if MTP_PF_CHECK == "1":  # fuse_v1 vs fuse_v2 on this chunk, bitwise (host reads; check runs only)
            rd = lambda: [ttnn.to_torch(t_).clone() for t_ in ttnn.get_device_tensors(self.chain.RB)]  # noqa: E731
            self.fuse_v1(self.chain.RA, self.chain.RB)
            a_ = rd()
            self.fuse_v2(self.chain.RA, self.chain.RB)
            b_ = rd()
            eq = all(torch.equal(x_.view(torch.int16), y_.view(torch.int16)) for x_, y_ in zip(a_, b_))
            self.chk["chunks"] += 1
            self.chk["equal"] += int(eq)
            if not eq and self.chk["first_bad"] is None:
                self.chk["first_bad"] = (k, sum(int((x_ != y_).sum()) for x_, y_ in zip(a_, b_)))
            rm = self.chain.RB
        else:
            rm = self.fuse(self.chain.RA, self.chain.RB)
        x, _ = A.gr2.read(rm)
        A.state_chunk(x)
        self.n_chunks += 1

    def residual(self):
        """(checks) the last chunk's MTP residual, host [512, 10240] bf16."""
        return gr_tp.local_to_residual([ttnn.to_torch(x) for x in ttnn.get_device_tensors(self.chain.RB)]).bfloat16()
