# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Open the 4-card ring mesh outside pytest, exactly as a pytest test gets it from the
``bh_2d_mesh_device`` fixture: tt-metal's conftest.py ``bh_2d_mesh_device_context`` (the TT_MESH_GRAPH_DESC_PATH
branch) + ``set_fabric`` / ``reset_fabric`` + tests/scripts/common.py ``get_updated_device_params``, extracted (only
the branches this mesh takes: Blackhole, fabric_config set, no fabric_tensix_config / reliability_mode /
fabric_manager, no dispatch_core_axis / type), then ``create_submesh(MeshShape(4, 1))`` + the program cache as the
test does.  Provenance: PROVENANCE.md "tt-metal core (conftest)".

Usage (env: TT_MESH_GRAPH_DESC_PATH = a 4x1 ring descriptor from qwen38_2x262k/mgd/, plus the B4_* switches; see
scripts/run_server.sh):

    with open_ring(model.device_params()) as mesh:
        d = StepDriver(mesh, ck).start()
"""

from __future__ import annotations

import contextlib
import os

import ttnn


def _updated_device_params(device_params):
    """tests/scripts/common.py get_updated_device_params (dispatch_core_axis / type None, no tensix config)."""

    p = dict(device_params)
    axis = p.pop("dispatch_core_axis", None)
    kind = p.pop("dispatch_core_type", None)
    tensix = p.get("fabric_tensix_config", None)
    assert ttnn.device.is_blackhole(), "the 4-card ring is Blackhole only"
    if not (p.get("fabric_config", None) and tensix) and axis == ttnn.DispatchCoreAxis.ROW:
        axis = ttnn.DispatchCoreAxis.COL
    p["dispatch_core_config"] = ttnn.DispatchCoreConfig(kind, axis, tensix)
    return p


def _set_fabric(fabric_config, fabric_router_config):
    """conftest.py set_fabric with reliability_mode / fabric_tensix_config / fabric_manager at their defaults."""

    if not fabric_config:
        return
    args = (
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    if fabric_router_config is not None:
        ttnn.set_fabric_config(*args, fabric_router_config)
    else:
        ttnn.set_fabric_config(*args)


@contextlib.contextmanager
def open_ring(device_params, submesh=(4, 1), program_cache=True):
    """The full mesh from the mesh graph descriptor, then the (4, 1) submesh the driver runs on."""

    assert os.environ.get("TT_MESH_GRAPH_DESC_PATH"), "set TT_MESH_GRAPH_DESC_PATH (qwen38_2x262k/mgd/p150_4x1_ring.textproto)"
    assert ttnn.get_num_devices() in (1, 2, 4, 8, 32), f"{ttnn.get_num_devices()} devices"
    p = _updated_device_params(device_params)
    fabric_config = p.pop("fabric_config", None)
    for k in ("fabric_tensix_config", "reliability_mode", "fabric_manager"):
        assert p.pop(k, None) is None, f"{k} is not handled here"
    _set_fabric(fabric_config, p.pop("fabric_router_config", None))
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn._ttnn.multi_device.SystemMeshDescriptor().shape(), **p)
    try:
        mesh = mesh_device.create_submesh(ttnn.MeshShape(submesh))
        if program_cache:
            mesh.enable_program_cache()
        yield mesh
    finally:
        for sm in mesh_device.get_submeshes():
            ttnn.close_mesh_device(sm)
        ttnn.close_mesh_device(mesh_device)
        if fabric_config:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
