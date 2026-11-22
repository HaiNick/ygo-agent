#!/usr/bin/env bash
# Start the ygoinf FastAPI inference server on 127.0.0.1:3000 (CPU).
#
# Usage:
#   tools/start_inf_server.sh [CHECKPOINT]
# Env overrides:
#   CHECKPOINT      path to .flax_model (default: scripts/checkpoints/ft_v2_50M_main15017.flax_model,
#                   the promoted model, docs/model_ft_v2_50M.md)
#   CODE_LIST       path to code_list.txt (default: ~/ygo/ygo-agent/scripts/code_list.txt)
#   NUM_EMBEDDINGS  embedding table size (default: 15017)
#   CARDS_DB        cards.cdb for card-name declarations (default: assets/locale/en/cards.cdb)
#   ANNOUNCE_POOL   deck files or folders (':'-separated) that rank declaration candidates,
#                   the decks the env loads in training (default: ~/ygo/decks/train)
#   OBS_VERSION     observation version the checkpoint was trained on, 1 or 2 (default: 2 for the
#                   promoted model, 1 for a CHECKPOINT given by hand and for the fallbacks)
#   HOST            bind host (default: 127.0.0.1)
#   PORT            bind port (default: 3000)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="$HOME/ygo/runs/ft_0546_pool52_r4/checkpoints"

DEFAULT_OBS_VERSION=1
if [[ -n "${1:-}" ]]; then
  CKPT="$(realpath "$1")"
elif [[ -n "${CHECKPOINT:-}" ]]; then
  CKPT="$(realpath "$CHECKPOINT")"
else
  # the promoted model (observation version 2), else the version 1 models: the previous
  # promoted model, the newest r4 checkpoint, the baseline
  CKPT="$REPO/scripts/checkpoints/ft_v2_50M_main15017.flax_model"
  DEFAULT_OBS_VERSION=2
  if [[ ! -f "$CKPT" ]]; then
    DEFAULT_OBS_VERSION=1
    CKPT="$REPO/scripts/checkpoints/ft_r4_100M_main15017.flax_model"
  fi
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
export CARDS_DB="${CARDS_DB:-$REPO/assets/locale/en/cards.cdb}"
export ANNOUNCE_POOL="${ANNOUNCE_POOL:-$HOME/ygo/decks/train}"
export OBS_VERSION="${OBS_VERSION:-$DEFAULT_OBS_VERSION}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-3000}"

source "$HOME/ygo/.venv/bin/activate"
export PYTHONPATH="$REPO/ygoinf:$REPO:${PYTHONPATH:-}"

echo "Starting ygoinf server on http://${HOST}:${PORT}"
echo "  JAX_PLATFORMS=$JAX_PLATFORMS"
echo "  CHECKPOINT=$CHECKPOINT"
echo "  CODE_LIST=$CODE_LIST"
echo "  NUM_EMBEDDINGS=$NUM_EMBEDDINGS"
echo "  CARDS_DB=$CARDS_DB"
echo "  ANNOUNCE_POOL=$ANNOUNCE_POOL"
echo "  OBS_VERSION=$OBS_VERSION"

cd "$REPO/ygoinf"
exec uvicorn ygoinf.server:app --host "$HOST" --port "$PORT"
