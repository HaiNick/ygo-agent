#!/usr/bin/env bash
# Start the ygoinf FastAPI inference server on 127.0.0.1:3000 (CPU).
#
# Usage:
#   tools/start_inf_server.sh [CHECKPOINT]
# Env overrides:
#   CHECKPOINT      path to .flax_model (default: scripts/checkpoints/ft_r4_100M_main15017.flax_model)
#   CODE_LIST       path to code_list.txt (default: ~/ygo/ygo-agent/scripts/code_list.txt)
#   NUM_EMBEDDINGS  embedding table size (default: 15017)
#   HOST            bind host (default: 127.0.0.1)
#   PORT            bind port (default: 3000)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="$HOME/ygo/runs/ft_0546_pool52_r4/checkpoints"

if [[ -n "${1:-}" ]]; then
  CKPT="$(realpath "$1")"
elif [[ -n "${CHECKPOINT:-}" ]]; then
  CKPT="$(realpath "$CHECKPOINT")"
else
  # the promoted fine-tuned model, else the newest r4 checkpoint, else the baseline
  CKPT="$REPO/scripts/checkpoints/ft_r4_100M_main15017.flax_model"
  if [[ ! -f "$CKPT" ]]; then
    CKPT="$(ls -t "$CKPT_DIR"/*.flax_model 2>/dev/null | head -n 1 || true)"
  fi
  if [[ -z "$CKPT" ]]; then
    CKPT="$REPO/scripts/checkpoints/0546_22750M_main15017_filled.flax_model"
  fi
fi

[[ -f "$CKPT" ]] || { echo "Checkpoint not found: $CKPT" >&2; exit 1; }

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CHECKPOINT="$CKPT"
export CODE_LIST="${CODE_LIST:-$REPO/scripts/code_list.txt}"
export NUM_EMBEDDINGS="${NUM_EMBEDDINGS:-15017}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-3000}"

source "$HOME/ygo/.venv/bin/activate"
export PYTHONPATH="$REPO/ygoinf:$REPO:${PYTHONPATH:-}"

echo "Starting ygoinf server on http://${HOST}:${PORT}"
echo "  JAX_PLATFORMS=$JAX_PLATFORMS"
echo "  CHECKPOINT=$CHECKPOINT"
echo "  CODE_LIST=$CODE_LIST"
echo "  NUM_EMBEDDINGS=$NUM_EMBEDDINGS"

cd "$REPO/ygoinf"
exec uvicorn ygoinf.server:app --host "$HOST" --port "$PORT"
