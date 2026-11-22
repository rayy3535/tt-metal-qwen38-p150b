# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""create_fabric_router_config, extracted verbatim from DeepSeek-V3-B1/tests/unit_tests/ccl_test_utils.py
(see ../PROVENANCE.md)."""

import ttnn


def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config
