#!/usr/bin/env bash
# One entry point for training runs: start, stop, resume, status, record, eval.
#
#   tools/ygo.sh start  [RUN]          new run folder ~/ygo/runs/RUN (default: ft_<date>)
#   tools/ygo.sh stop   [RUN] [--wait] clean stop: finish the iteration, save, exit
#   tools/ygo.sh resume [RUN]          continue RUN in the same folder from its newest checkpoint
#   tools/ygo.sh status [RUN]          state, steps, SPS, last checkpoint, last eval
#   tools/ygo.sh list                  all run folders with their state
#   tools/ygo.sh record [RUN] DECK [GAMES] [OPPONENT]
#                                      newest checkpoint of RUN vs OPPONENT (default: the
#                                      promoted model), recorded as replays; works while
#                                      training runs
#   tools/ygo.sh eval   [RUN] [full]   newest checkpoint vs the promoted model per deck
#                                      (quick 4-deck subset unless "full")
#   tools/ygo.sh tb                    TensorBoard on http://localhost:6006
#   tools/ygo.sh tui                   dashboard (tools/ygo_tui.py)
#
# RUN defaults to the run used last (~/ygo/runs/.current). DECK is a file name in
# ~/ygo/decks/mine without .ydk, or a path.
# start passes these environment variables to tools/train_finetune.sh: INIT
# (default: the promoted model), TB_OFFSET (default: 104906752 with that model),
# SAVE_INTERVAL (default 500), XLA_MEM_FRACTION (default 0.65), NUM_ENVS,
# ACTOR_THREADS, NUM_STEPS, DECK_DIR, SEED, EXTRA.
#
# A stop saves the weights, not the optimizer state: every resume starts with
# fresh Adam moments.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YGO="${YGO_HOME:-$HOME/ygo}"
RUNS="$YGO/runs"
MODEL="${MODEL:-$REPO/scripts/checkpoints/ft_r4_100M_main15017.flax_model}"
MODEL_STEPS=104906752
MDPRO_REPLAY="${MDPRO_REPLAY:-/mnt/d/Games/MDPro4/Replay}"

die() { echo "$*" >&2; exit 1; }
current() { [[ -f "$RUNS/.current" ]] && cat "$RUNS/.current" || true; }
pick_run() {  # pick_run [NAME] -> sets RUN, DIR
  RUN="${1:-$(current)}"
  [[ -n "$RUN" ]] || die "no run given and no current run (start one with: tools/ygo.sh start NAME)"
  DIR="$RUNS/$RUN"
  [[ -d "$DIR" ]] || die "run folder not found: $DIR"
}
newest_ckpt() { ls -t "$1"/checkpoints/*.flax_model 2>/dev/null | head -1 || true; }
state() {  # state DIR -> running | stopping | stopped | crashed | new
  local d="$1" r; r="$(basename "$d")"
  if tmux has-session -t "=$r" 2>/dev/null; then
    [[ -e "$d/checkpoints/STOP" ]] && echo stopping || echo running
  elif [[ ! -s "$d/watchdog.log" ]]; then
    # runs from before the watchdog have a train.log but no watchdog.log
    [[ -s "$d/train.log" ]] && echo stopped || echo new
  elif tail -1 "$d/watchdog.log" | grep -q "exited normally"; then echo stopped
  else echo crashed; fi
}

cmd="${1:-}"; shift || true
case "$cmd" in
start)
  RUN="${1:-ft_$(date +%Y%m%d_%H%M)}"
  if [[ -z "${INIT:-}" ]]; then export INIT="$MODEL"; export TB_OFFSET="${TB_OFFSET:-$MODEL_STEPS}"; fi
  export SAVE_INTERVAL="${SAVE_INTERVAL:-500}" XLA_MEM_FRACTION="${XLA_MEM_FRACTION:-0.65}"
  "$REPO/tools/train_finetune.sh" "$RUN"
  echo "$RUN" > "$RUNS/.current"
  ;;
stop)
  wait=0; args=(); for a in "$@"; do [[ "$a" == --wait ]] && wait=1 || args+=("$a"); done
  pick_run "${args[0]:-}"
  [[ "$(state "$DIR")" == running || "$(state "$DIR")" == stopping ]] || die "$RUN is not running ($(state "$DIR"))"
  touch "$DIR/checkpoints/STOP"
  echo "stop requested for $RUN: it finishes the current iteration, saves a checkpoint and exits"
  if (( wait )); then
    while tmux has-session -t "=$RUN" 2>/dev/null; do sleep 2; done
    echo "$RUN stopped; newest checkpoint: $(basename "$(newest_ckpt "$DIR")")"
  fi
  ;;
resume)
  pick_run "${1:-}"
  s="$(state "$DIR")"
  [[ "$s" == stopped || "$s" == crashed ]] || die "$RUN is $s; only a stopped or crashed run can be resumed"
  ck="$(newest_ckpt "$DIR")"
  [[ -n "$ck" ]] || die "$RUN has no checkpoint to resume from"
  rm -f "$DIR/checkpoints/STOP"
  ENVSET=""
  mem="$(grep -a '^XLA_PYTHON_CLIENT_MEM_FRACTION:' "$DIR/run_info.txt" 2>/dev/null | awk '{print $2}')"
  mem="${XLA_MEM_FRACTION:-$mem}"
  [[ -n "$mem" && "$mem" != default ]] && ENVSET="export XLA_PYTHON_CLIENT_MEM_FRACTION=$mem && "
  echo "$(date -Is) resume from $(basename "$ck")" >> "$DIR/watchdog.log"
  tmux new-session -d -s "$RUN" "${ENVSET}exec '$REPO/tools/train_watchdog.sh' '$DIR'"
  echo "$RUN" > "$RUNS/.current"
  echo "resumed $RUN from $(basename "$ck") (same folder; log continues in $DIR/train.log)"
  ;;
status)
  pick_run "${1:-}"
  log="$DIR/train.log"; ck="$(newest_ckpt "$DIR")"
  echo "run:        $RUN ($(state "$DIR"))"
  echo "step:       $(grep -a -o 'global_step=[0-9]*' "$log" 2>/dev/null | tail -1 | cut -d= -f2)"
  echo "sps:        $(grep -a -o 'SPS: [0-9]*' "$log" 2>/dev/null | tail -1 | cut -d' ' -f2)"
  echo "last eval:  $(grep -a -o 'eval_win_rate=[0-9.]*' "$log" 2>/dev/null | tail -1 | cut -d= -f2) (win rate vs the built-in bot)"
  echo "checkpoint: ${ck:+$(basename "$ck") ($(date -r "$ck" '+%Y-%m-%d %H:%M'))}"
  echo "checkpoints: $(ls "$DIR"/checkpoints/*.flax_model 2>/dev/null | wc -l) files, $(du -sh "$DIR/checkpoints" 2>/dev/null | cut -f1)"
  echo "non-finite: $(grep -a -c 'WARNING: non-finite' "$log" 2>/dev/null || true) iterations with dropped updates"
  echo "watchdog:   $(tail -1 "$DIR/watchdog.log" 2>/dev/null)"
  ;;
list)
  cur="$(current)"
  for d in "$RUNS"/*/; do
    r="$(basename "$d")"
    printf '%s %-28s %-9s %s\n' "$([[ "$r" == "$cur" ]] && echo '*' || echo ' ')" "$r" "$(state "${d%/}")" \
      "$(basename "$(newest_ckpt "${d%/}")" 2>/dev/null)"
  done
  ;;
record)
  # record [RUN] DECK [GAMES] [OPPONENT]; RUN may be omitted
  if [[ -n "${1:-}" && -d "$RUNS/$1" ]]; then pick_run "$1"; shift; else pick_run ""; fi
  deck="${1:-}"; games="${2:-3}"; opp="${3:-$MODEL}"
  [[ -n "$deck" ]] || die "usage: tools/ygo.sh record [RUN] DECK [GAMES] [OPPONENT]"
  [[ -f "$deck" ]] || deck="$YGO/decks/mine/$deck.ydk"
  [[ -f "$deck" ]] || die "deck not found: $deck"
  ck="$(newest_ckpt "$DIR")"; [[ -n "$ck" ]] || ck="$(cat "$DIR/init_checkpoint")"
  out="$YGO/evals/records/${RUN}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$out" && cd "$out"
  ln -s "$REPO/scripts/script" script; ln -s "$REPO/scripts/code_list.txt" code_list.txt
  # shellcheck disable=SC1091
  source "$YGO/.venv/bin/activate"
  echo "recording $games games: $(basename "$ck") vs $(basename "$opp"), deck $(basename "$deck" .ydk)"
  XLA_PYTHON_CLIENT_PREALLOCATE=false python "$REPO/scripts/battle.py" --deck "$deck" --checkpoint1 "$ck" \
    --checkpoint2 "$opp" --num_embeddings 15017 --num_episodes "$games" --record > match.log 2>&1 \
    || { tail -5 match.log; die "recording failed, see $out/match.log"; }
  grep -a "win_rate=" match.log | tail -1
  name="$(basename "$deck" .ydk)_$(basename "$ck" .flax_model)"
  i=0
  while IFS= read -r f; do
    i=$((i + 1)); mv "$f" "$out/${name}_game$i.yrp"
    [[ -d "$MDPRO_REPLAY" ]] && cp "$out/${name}_game$i.yrp" "$MDPRO_REPLAY/"
  done < <(ls -tr "$out"/replay/*.yrp 2>/dev/null)
  echo "$i replays in $out$([[ -d "$MDPRO_REPLAY" ]] && echo " (copied to $MDPRO_REPLAY)")"
  ;;
eval)
  if [[ -n "${1:-}" && -d "$RUNS/$1" ]]; then pick_run "$1"; shift; else pick_run ""; fi
  ck="$(newest_ckpt "$DIR")"; [[ -n "$ck" ]] || die "$RUN has no checkpoint yet"
  [[ "${1:-}" == full ]] && q=0 || q=1
  QUICK=$q XLA_PYTHON_CLIENT_PREALLOCATE=false "$REPO/tools/eval_vs_baseline.sh" "$ck" "$MODEL"
  ;;
tb)
  tmux has-session -t "=ygo_tb" 2>/dev/null && die "TensorBoard already runs: http://localhost:6006 (stop: tmux kill-session -t ygo_tb)"
  tmux new-session -d -s ygo_tb "source '$YGO/.venv-embed/bin/activate' && tensorboard --logdir '$RUNS' --port 6006"
  echo "TensorBoard: http://localhost:6006 (stop: tmux kill-session -t ygo_tb)"
  ;;
tui)
  # shellcheck disable=SC1091
  source "$YGO/.venv/bin/activate"
  exec python "$REPO/tools/ygo_tui.py" "$@"
  ;;
*)
  sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  ;;
esac
