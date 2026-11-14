#!/usr/bin/env bash
# Delete old checkpoints of a training run, keeping:
#   - the newest KEEP_LAST files (default 5)
#   - the newest file of every MILESTONE megasteps (default 10: 0-9M, 10-19M, ...)
#   - every checkpoint named in an eval report under ~/ygo/evals/*.md
#   - the run's init checkpoint if it lives in this folder
#
# usage: tools/prune_checkpoints.sh RUN_DIR [--delete]
#   Without --delete it only prints what it would remove (dry run).
#   env: KEEP_LAST (5), MILESTONE (10), EVALS (~/ygo/evals)
# Refuses to run while a cleanba.py process uses the run folder.
set -euo pipefail
DIR="$(realpath "$1")/checkpoints"
DO="${2:-}"
KEEP_LAST="${KEEP_LAST:-5}"
MILESTONE="${MILESTONE:-10}"
EVALS="${EVALS:-$HOME/ygo/evals}"
[[ -d "$DIR" ]] || { echo "no checkpoints folder: $DIR" >&2; exit 1; }
if pgrep -f "cleanba.py.*$(dirname "$DIR")" > /dev/null; then
  echo "a training process is using $(dirname "$DIR"); not pruning" >&2; exit 1
fi
INIT=""
[[ -f "$(dirname "$DIR")/init_checkpoint" ]] && INIT="$(basename "$(cat "$(dirname "$DIR")/init_checkpoint")")"
mapfile -t files < <(ls -t "$DIR"/*.flax_model 2>/dev/null)
declare -A seen_ms
kept=0; removed=0; freed=0
for i in "${!files[@]}"; do
  f="${files[$i]}"; b="$(basename "$f")"; keep=""
  (( i < KEEP_LAST )) && keep="newest"
  [[ "$b" == "$INIT" ]] && keep="init"
  if [[ -z "$keep" && "$b" =~ _([0-9]+)M ]]; then
    # saves rarely land exactly on a multiple, so keep the newest file of
    # every MILESTONE-wide bucket (files are listed newest first)
    bucket=$(( BASH_REMATCH[1] / MILESTONE ))
    if [[ -z "${seen_ms[$bucket]:-}" ]]; then keep="milestone bucket $((bucket * MILESTONE))M"; fi
    seen_ms[$bucket]=1
  fi
  if [[ -z "$keep" && -d "$EVALS" ]] && grep -rqs --include='*.md' -F "${b%.flax_model}" "$EVALS"; then keep="in an eval report"; fi
  if [[ -n "$keep" ]]; then
    kept=$((kept + 1)); echo "keep   $b ($keep)"
  else
    removed=$((removed + 1)); freed=$((freed + $(stat -c %s "$f")))
    if [[ "$DO" == "--delete" ]]; then rm -f -- "$f"; echo "delete $b"; else echo "would delete $b"; fi
  fi
done
echo "kept $kept, $([[ "$DO" == "--delete" ]] && echo deleted || echo "would delete") $removed files ($((freed / 1048576)) MiB)"
