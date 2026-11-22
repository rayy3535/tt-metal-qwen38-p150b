# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP4 routed-expert cache for moe_tp4, built from the HF checkpoint with qwen38_2x262k's own code only.

The vendored checkpoint loader (ref/) reads each backbone MoE layer's 512 routed experts in BF16, moe_tp4/layout.py
re-lays them into the per-card bank blocks, and ttnn packs them to BFLOAT4_B on the host (no device).  One tensorbin
per (layer, card, gu|d) plus manifest.json (bytes + sha256), the format of bf4-experts-tp4-v1.

  python -m models.demos.blackhole.qwen38_2x262k.tools.convert_tp4 --out DIR [--layers 0-47]
      convert (resumable: layers already in DIR/manifest.json are skipped)
  python -m models.demos.blackhole.qwen38_2x262k.tools.convert_tp4 --out DIR --layers 0,23,47 --verify REF_DIR
      convert those layers into DIR, then compare every file with REF_DIR: sha256 against REF_DIR/manifest.json,
      and on a mismatch the decoded tiles (ttnn.load_tensor -> to_torch, bit view) to tell a container difference
      from a value difference.  Never writes into REF_DIR.
  python -m models.demos.blackhole.qwen38_2x262k.tools.convert_tp4 --out DIR --mtp
      the MTP layer's routed experts (mtp.layers.0.mlp) instead of backbone layers: DIR/mtp-00/card{c}_{gu|d}
      .tensorbin, manifest key "mtp:0"; same layout / format as a backbone layer (moe_layer.mesh_experts reads it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.moe_tp4 import layout as L
from models.demos.blackhole.qwen38_2x262k.ref.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_2x262k.ref.config import Qwen38Placement
from models.demos.blackhole.qwen38_2x262k.ref.tt.moe import Qwen38MoEWeights

FORMAT = "qwen38_2x262k moe_tp4 v1"
DEFAULT_CHECKPOINT = os.path.expanduser("~/qwen38/weights/Qwen3.8-Flash-Next")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_layers(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def new_manifest(checkpoint_root: str) -> dict:
    return {
        "format": FORMAT,
        "layout": "moe_tp4/layout.py: gu K-quarter blocks [k20][t5] per bank, down [j5][n10] per bank, expert-major",
        "dtype": "BFLOAT4_B",
        "host_shape": {"gu": [32 * 512 * L.GU_BLOCK_TILES, 256], "d": [32 * 512 * L.D_BLOCK_TILES, 256]},
        "checkpoint_root": checkpoint_root,
        "converter": "qwen38_2x262k/tools/convert_tp4.py",
        "layers": {},
    }


def layer_dirname(layer: int, mtp: bool = False) -> str:
    return f"mtp-{layer:02d}" if mtp else f"layer-{layer:02d}"


def convert_layer(ck, pl, layer: int, out_dir: str, mtp: bool = False) -> dict:
    w = Qwen38MoEWeights(ck, pl, mtp_layer_index=layer) if mtp else Qwen38MoEWeights(ck, pl, layer_index=layer)
    shards = [w.routed_device_shard(d) for d in range(4)]
    gu = torch.cat([s.gate_up[0] for s in shards])  # [512, 2560, 1280]
    dn = torch.cat([s.down[0] for s in shards])  # [512, 640, 2560]
    del shards
    entry = {}
    ldir = f"{out_dir}/{layer_dirname(layer, mtp)}"
    os.makedirs(ldir, exist_ok=True)
    for card in range(L.CARDS):
        for name, blocks in (("gu", L.gu_bank_blocks(gu, card)), ("d", L.d_bank_blocks(dn, card))):
            host = L.bank_tensor(blocks)[None, None]
            del blocks
            t = ttnn.from_torch(host, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
            del host
            path = f"{ldir}/card{card}_{name}.tensorbin"
            tmp = path[: -len(".tensorbin")] + ".tmp.tensorbin"
            ttnn.dump_tensor(tmp, t)
            os.replace(tmp, path)
            entry[f"card{card}_{name}"] = {"bytes": os.path.getsize(path), "sha256": sha256(path)}
            del t
    return entry


def verify_layer(layer: int, entry: dict, out_dir: str, ref_dir: str, ref_man: dict) -> bool:
    ref_entry = ref_man["layers"].get(f"backbone:{layer}")
    if ref_entry is None:
        print(f"VERIFY layer {layer}: not in {ref_dir}/manifest.json", flush=True)
        return False
    ok = True
    for key, got in entry.items():
        want = ref_entry[key]
        if got["sha256"] == want["sha256"] and got["bytes"] == want["bytes"]:
            print(f"VERIFY layer {layer} {key}: sha256 identical ({got['bytes']} bytes)", flush=True)
            continue
        a = ttnn.to_torch(ttnn.load_tensor(f"{out_dir}/layer-{layer:02d}/{key}.tensorbin"))
        b = ttnn.to_torch(ttnn.load_tensor(f"{ref_dir}/layer-{layer:02d}/{key}.tensorbin"))
        same = a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))
        diff = "n/a" if a.shape != b.shape else int((a.view(torch.int16) != b.view(torch.int16)).sum())
        print(
            f"VERIFY layer {layer} {key}: sha256 DIFFERS ({got['bytes']} vs {want['bytes']} bytes); decoded values "
            f"{'identical (container only)' if same else 'DIFFER'}: {diff} elements",
            flush=True,
        )
        ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--verify", default=None, help="reference cache directory (read only)")
    ap.add_argument("--mtp", action="store_true", help="convert the MTP layer's experts (mtp.layers.0.mlp)")
    args = ap.parse_args()
    if args.mtp and args.verify:
        raise SystemExit("--mtp has no reference cache to --verify against")
    out_dir = os.path.abspath(os.path.expanduser(args.out))
    ref_dir = os.path.abspath(os.path.expanduser(args.verify)) if args.verify else None
    if ref_dir is not None and ref_dir == out_dir:
        raise SystemExit("--out must differ from --verify (the reference cache is never written)")
    os.makedirs(out_dir, exist_ok=True)
    man_path = f"{out_dir}/manifest.json"
    man = json.load(open(man_path)) if os.path.exists(man_path) else new_manifest(args.checkpoint)
    ref_man = json.load(open(f"{ref_dir}/manifest.json")) if ref_dir else None
    ck = Qwen38Checkpoint(os.path.expanduser(args.checkpoint))
    pl = Qwen38Placement(ck.config, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))
    all_ok = True
    for layer in [0] if args.mtp else parse_layers(args.layers):
        key = f"mtp:{layer}" if args.mtp else f"backbone:{layer}"
        if key in man["layers"]:
            print(f"layer {layer}: done, skip", flush=True)
        else:
            t0 = time.time()
            man["layers"][key] = convert_layer(ck, pl, layer, out_dir, mtp=args.mtp)
            with open(man_path + ".tmp", "w") as f:
                json.dump(man, f, indent=1)
            os.replace(man_path + ".tmp", man_path)
            print(f"layer {layer}: converted in {time.time() - t0:.0f} s", flush=True)
        if ref_man is not None:
            all_ok &= verify_layer(layer, man["layers"][key], out_dir, ref_dir, ref_man)
    if ref_man is not None:
        print("VERIFY_OK" if all_ok else "VERIFY_FAIL", flush=True)
    print("CONVERT_DONE", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
