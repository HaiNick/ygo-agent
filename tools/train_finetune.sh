#!/usr/bin/env bash
# Start fine-tuning with scripts/cleanba.py in a detached tmux session.
#
# README single-GPU settings (16 envs x 2 actor threads, 8 minibatches, lr 1e-4,
# vloss_clip 1.0, eval vs the built-in bot every 50 iters), plus:
#   --checkpoint INIT, --num_embeddings 15017, --freeze_id True (embeddings frozen
#   as in the original training), --deck DECK_DIR, --save_interval 50,
#   --n_saved 0 (keep every checkpoint), TensorBoard logs on.
#
# usage: tools/train_finetune.sh [RUN_NAME]
#   env: INIT (filled 0546 checkpoint), DECK_DIR (~/ygo/decks/train), SEED (0),
#        NUM_EMBEDDINGS (15017), EXTRA (extra cleanba.py args)
#
# Layout: ~/ygo/runs/<RUN_NAME>/{checkpoints/,tb/,train.log,run_info.txt}
# Watch:  tail -f ~/ygo/runs/<RUN_NAME>/train.log   |   tmux attach -t <RUN_NAME>
# Stop:   touch ~/ygo/runs/<RUN_NAME>/checkpoints/STOP   (or Ctrl-C in the tmux session)
#         -> finishes the iteration, saves a checkpoint, exits
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:-ft_$(date +%Y%m%d_%H%M)}"
INIT="$(realpath "${INIT:-$REPO/scripts/checkpoints/0546_22750M_main15017_filled.flax_model}")"
DECK_DIR="$(realpath "${DECK_DIR:-$HOME/ygo/decks/train}")"
SEED="${SEED:-0}"
NEMB="${NUM_EMBEDDINGS:-15017}"
EXTRA="${EXTRA:-}"
DIR="$HOME/ygo/runs/$RUN"

if tmux has-session -t "$RUN" 2>/dev/null; then
  echo "tmux session $RUN already exists" >&2; exit 1
fi
[[ -e "$DIR" ]] && { echo "$DIR already exists" >&2; exit 1; }
mkdir -p "$DIR/checkpoints" "$DIR/tb"

CMD=(python -u cleanba.py
  --actor-device-ids 0 --learner-device-ids 0
  --deck "$DECK_DIR"
  --local-num_envs 16 --num-minibatches 8 --learning-rate 1e-4 --vloss_clip 1.0
  --save_interval 50 --local_eval_episodes 32 --eval_interval 50 --seed "$SEED"
  --checkpoint "$INIT" --num_embeddings "$NEMB" --freeze_id True
  --ckpt_dir "$DIR/checkpoints" --n_saved 0 --tb_dir "$DIR/tb")
# shellcheck disable=SC2206
CMD+=($EXTRA)

{
  echo "run: $RUN"
  echo "started: $(date -Is)"
  echo "commit: $(cd "$REPO" && git rev-parse --short HEAD)$(cd "$REPO" && git diff --quiet || echo ' (dirty)')"
  echo "init: $INIT"
  echo "decks: $DECK_DIR ($(ls "$DECK_DIR"/*.ydk | wc -l) files)"
  echo "cmd: cd $REPO/scripts && ${CMD[*]}"
} > "$DIR/run_info.txt"

printf -v QCMD '%q ' "${CMD[@]}"
tmux new-session -d -s "$RUN" \
  "cd '$REPO/scripts' && source '$HOME/ygo/.venv/bin/activate' && exec $QCMD >> '$DIR/train.log' 2>&1"
echo "started tmux session '$RUN'"
cat "$DIR/run_info.txt"
