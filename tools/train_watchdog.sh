#!/usr/bin/env bash
# Run scripts/cleanba.py for a run folder made by tools/train_finetune.sh and
# resume after abnormal exits.
#
#   tools/train_watchdog.sh RUN_DIR
#
# RUN_DIR holds cmd_args (cleanba.py arguments without --checkpoint and
# --tb_offset, one per line), init_checkpoint and tb_offset.
# - attempt 1 starts from init_checkpoint; later attempts from the newest
#   checkpoint in RUN_DIR/checkpoints, with --tb_offset = last logged global_step
# - exit code 0 (normal end, STOP file or Ctrl-C clean stop) ends the watchdog
# - an abnormal exit is logged to RUN_DIR/watchdog.log; after 30 s training
#   resumes, unless RUN_DIR/checkpoints/STOP exists or this was the 3rd crash
#   within one hour
# - each attempt runs with YGO_CRASH_FILE=RUN_DIR/crash_attemptN.txt: the env
#   writes every env's deck pair, seeds and turn there on a fatal signal
set -u

DIR="$(realpath "$1")"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$DIR/train.log"
WLOG="$DIR/watchdog.log"
mapfile -t ARGS < "$DIR/cmd_args"
INIT="$(cat "$DIR/init_checkpoint")"
TB_OFFSET="$(cat "$DIR/tb_offset")"
MAX_CRASHES=3
WINDOW=3600
WAIT=30

wlog() { echo "$(date -Is) $*" | tee -a "$WLOG"; }

cd "$REPO/scripts" || exit 1
# shellcheck disable=SC1091
source "$HOME/ygo/.venv/bin/activate"

crashes=()
attempt=0
while true; do
  attempt=$((attempt + 1))
  ckpt="$INIT"
  off="$TB_OFFSET"
  if (( attempt > 1 )); then
    newest="$(ls -t "$DIR"/checkpoints/*.flax_model 2>/dev/null | head -1)"
    [[ -n "$newest" ]] && ckpt="$newest"
    last="$(grep -o 'global_step=[0-9]*' "$LOG" 2>/dev/null | tail -1 | cut -d= -f2)"
    [[ -n "$last" ]] && off="$last"
  fi
  crash_file="$DIR/crash_attempt$attempt.txt"
  wlog "attempt $attempt: start from $ckpt (tb_offset $off)"
  YGO_CRASH_FILE="$crash_file" python -u cleanba.py "${ARGS[@]}" \
    --checkpoint "$ckpt" --tb_offset "$off" >> "$LOG" 2>&1
  rc=$?
  if (( rc == 0 )); then
    wlog "attempt $attempt: exited normally"
    exit 0
  fi
  sig=""
  (( rc > 128 )) && sig=" (signal $((rc - 128)))"
  info="no crash info written"
  [[ -s "$crash_file" ]] && info="crash info: $crash_file"
  if [[ -e "$DIR/checkpoints/STOP" ]]; then
    wlog "attempt $attempt: exit code $rc$sig with STOP file present; not resuming ($info)"
    exit "$rc"
  fi
  now="$(date +%s)"
  crashes+=("$now")
  recent=0
  for t in "${crashes[@]}"; do
    (( now - t < WINDOW )) && recent=$((recent + 1))
  done
  wlog "attempt $attempt: ABNORMAL EXIT, exit code $rc$sig; $info; $recent crash(es) in the last hour"
  if (( recent >= MAX_CRASHES )); then
    wlog "giving up: $MAX_CRASHES crashes within one hour"
    exit 1
  fi
  wlog "resuming in $WAIT s"
  sleep "$WAIT"
done
