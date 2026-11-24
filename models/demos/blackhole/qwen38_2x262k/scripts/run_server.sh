#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Start the 2-user, full-context (2 x 262K) Qwen3.8-Flash-Next server on 4 x Blackhole p150b with the settings the published numbers were
# measured with: the long-context mode (2 slots x 262,144 tokens), device prefill, MTP speculative decode (--mtp auto),
# an OpenAI-compatible HTTP API on 127.0.0.1:8020.
#
# usage: models/demos/blackhole/qwen38_2x262k/scripts/run_server.sh [--print-env] [extra server args ...]
#   --print-env   print the environment and the command the server would get, then exit (starts nothing)
#   extra args    passed to the server after the defaults (a repeated flag wins), e.g. --port 8000
#
# Machine-specific settings (environment, with defaults):
#   TT_METAL_HOME            this tt-metal checkout (default: derived from this script's location); must be built
#   PYTHON                   default $TT_METAL_HOME/python_env/bin/python (from ./create_venv.sh)
#   QWEN38_CHECKPOINT        Hugging Face checkpoint directory (default ~/qwen38/weights/Qwen3.8-Flash-Next)
#   TT_METAL_CACHE           JIT kernel cache directory (default ~/.cache/qwen38_2x262k/jit-cache)
#   TT_MESH_GRAPH_DESC_PATH  mesh graph descriptor (default mgd/p150_4x1_ring_pinned.textproto next to this script's
#                            directory; its card pinning is for one specific host, see the file header;
#                            mgd/p150_4x1_ring.textproto is the unpinned ring)
#   TT_VISIBLE_DEVICES       default 0,1,2,3
#   QWEN38_REQUEST_LOG            optional JSONL path: one record per answered request (--ledger)
#   QWEN38_HOLD_ON_HANG=1    optional: on a watchdog fire keep the process alive for live debugging (--hold-on-hang)
#                            instead of exiting with code 124; for use under a supervisor that captures and resets
# The routed-expert cache (tools/convert_tp4.py) must exist at ~/qwen38/cache/bf4-experts-tp4-v1 and the MTP expert
# cache at ~/qwen38/cache/bf4-experts-tp4-mtp-v1/mtp-00 (or set MTP_TP4); see the repository README.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG=$(dirname "$HERE")
export TT_METAL_HOME=${TT_METAL_HOME:-$(cd "$PKG/../../../.." && pwd)}
export PYTHONPATH=$TT_METAL_HOME
PYTHON=${PYTHON:-$TT_METAL_HOME/python_env/bin/python}
CHECKPOINT=${QWEN38_CHECKPOINT:-$HOME/qwen38/weights/Qwen3.8-Flash-Next}
export TT_METAL_CACHE=${TT_METAL_CACHE:-$HOME/.cache/qwen38_2x262k/jit-cache}
export TT_MESH_GRAPH_DESC_PATH=${TT_MESH_GRAPH_DESC_PATH:-$PKG/mgd/p150_4x1_ring_pinned.textproto}
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0,1,2,3}
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost

# Serving mode: 2 slots x 262,144-token context, device prefill in the server, live MoE expert path.
export QWEN38_CTX=262144 PF_SERVE=1 MOE_LIVE=1
# Seconds to retry a busy port at start-up (a restarted server waits for the previous one to release it).
export B4_SERVE_BIND_RETRY_S=120

# Decode kernel configuration (GDN layer, all-gathers, QSA attention).  These are the measured defaults of this
# implementation; the variables are read by the kernels' host code at import time.
export FID=2 XS0=1
export GDN_AG3_MCAST=1 GDN_DONE_AFTER_SEND=1 GDN_GATE_CORE=1 GDN_GRR=1 GDN_GRR_FID=4 GDN_GRR_LOCAL=1 GDN_GRR_LOWFIRST=1
export GDN_GRR_PRE=1 GDN_GRR_ROWS=1 GDN_HELPER_BUFS=5 GDN_HELPER_TINY=1 GDN_ROLE_SPLIT=1 GDN_SEND_FLUSH=1
export GDN_SEND_NCRISC=1 GDN_SEND_ROWS=1 GDN_STATE_EARLY= GDN_STATE_NCRISC=1 GDN_STREAMERS=1 GDN_STREAM_BUFS=40
export GDN_STREAM_CT=4 GDN_STREAM_HELPERS=1 GDN_STREAM_RESCOPE=1 GDN_STREAM_TINY=1 GDN_WOUT_EARLY= GDN_WOUT_GO=1
export GDN_WOUT_LOCAL=1 GDN_WOUT_PRE= GDN_WOUT_STREAM=1 GDN_XO_TINY=1 GDN_X_MCAST=1
export B4_AG_IDS=1 B4_AG_PLE=1
export QC_IX=1 QC_O80=0 QC_PT512=1
# L1 bytes per worker core reserved for the decode programs.
export WORKER_L1=1300000

# Device prefill configuration (the same values serve/server.py applies by default from prefill.PROD_ENV; exported here
# so the environment is explicit).
export QSA_RING=1 QIX_DM=1 QIX_BS=2 QIX_SFAST=1 QAS_QGB=1 QAS_MFOLD=1 QAS_PF=1 QAS_MF=1 QSA_QO=1 QIX_BUCKET=1024
export QSA_DEKEY=1 PF_PLE=1 PF_PARTIAL=1 IXI_F=1 IXI_POOL=1 IXI_INC=1

ARGS=(--host 127.0.0.1 --port 8020 --checkpoint "$CHECKPOINT")
[ -n "${QWEN38_REQUEST_LOG:-}" ] && ARGS+=(--ledger "$QWEN38_REQUEST_LOG")
[ "${QWEN38_HOLD_ON_HANG:-0}" = 1 ] && ARGS+=(--hold-on-hang)

PRINT=0
if [ "${1:-}" = --print-env ]; then
  PRINT=1
  shift
fi
CMD=("$PYTHON" -m models.demos.blackhole.qwen38_2x262k.serve.server "${ARGS[@]}" "$@")
cd "$TT_METAL_HOME"
if [ $PRINT = 1 ]; then
  env | grep -v '^\(PWD\|OLDPWD\|SHLVL\|_\)=' | LC_ALL=C sort
  echo "COMMAND: ${CMD[*]}"
  exit 0
fi
mkdir -p "$TT_METAL_CACHE"
exec "${CMD[@]}"
