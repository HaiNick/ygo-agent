#!/usr/bin/env bash
# Play a checkpoint against the baseline checkpoint with scripts/battle.py, per deck.
#
# Each deck is a mirror deck (both agents play it). battle.py runs N games with
# half of them seating the candidate first and half second (accurate mode), and
# the candidate is agent1.
#
# usage: tools/eval_vs_baseline.sh CKPT [BASELINE] [NUM_EMBEDDINGS] [DECK_DIR] [OUT_MD]
#   defaults: BASELINE = scripts/checkpoints/0546_22750M_main15017_filled.flax_model
#             NUM_EMBEDDINGS = 15017, DECK_DIR = ~/ygo/decks/mine,
#             OUT_MD = ~/ygo/evals/<ckpt>_vs_<baseline>.md
#   env: N_GAMES (128), SEED (0)
#        QUICK=1  only the quick-check subset: the base deck plus three new
#                 archetypes (Artmage_PowerPatron, Blitzclique_A, Elfnote_KewlTune)
#        ONLY="DeckA DeckB"  only these decks (file names without .ydk)
#   For use while training runs: XLA_PYTHON_CLIENT_PREALLOCATE=false
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$(realpath "$1")"
BASE="$(realpath "${2:-$REPO/scripts/checkpoints/0546_22750M_main15017_filled.flax_model}")"
NEMB="${3:-15017}"
DECKS="${4:-$HOME/ygo/decks/mine}"
cn="$(basename "$CKPT" .flax_model)"; bn="$(basename "$BASE" .flax_model)"
OUT="$(realpath -m "${5:-$HOME/ygo/evals/${cn}_vs_${bn}.md}")"
N_GAMES="${N_GAMES:-128}"
SEED="${SEED:-0}"
QUICK_DECKS="00_Base_Branded-Despia Artmage_PowerPatron Blitzclique_A Elfnote_KewlTune"
ONLY="${ONLY:-}"
[[ "${QUICK:-0}" == 1 && -z "$ONLY" ]] && ONLY="$QUICK_DECKS"
LOGDIR="$(dirname "$OUT")/logs_${cn}_vs_${bn}"
mkdir -p "$LOGDIR"

source "$HOME/ygo/.venv/bin/activate"
cd "$REPO/scripts"

shopt -s nullglob
if [[ -n "$ONLY" ]]; then
  decks=()
  for n in $ONLY; do
    [[ -f "$DECKS/$n.ydk" ]] || { echo "deck not found: $DECKS/$n.ydk" >&2; exit 1; }
    decks+=("$DECKS/$n.ydk")
  done
else
  decks=("$DECKS"/*.ydk)
fi
(( ${#decks[@]} )) || { echo "no .ydk files in $DECKS" >&2; exit 1; }

rows=()
for f in "${decks[@]}"; do
  name="$(basename "$f" .ydk)"
  log="$LOGDIR/$name.log"
  python battle.py --deck "$f" --checkpoint1 "$CKPT" --checkpoint2 "$BASE" --num_embeddings "$NEMB" \
    --num_episodes "$N_GAMES" --seed "$SEED" > "$log" 2>&1 || true
  row=$(python3 - "$log" "$name" <<'EOF'
import re, sys
s, name = open(sys.argv[1]).read(), sys.argv[2]
if "Traceback" in s or "what()" in s or "win_rate=" not in s:
    print(f"| {name} | error | | | | see {sys.argv[1]} |"); sys.exit()
wr = float(re.findall(r"win_rate=([0-9.]+)", s)[-1])
ln = float(re.findall(r"len=([0-9.]+)", s)[-1])
m = re.search(r"0/1 matrix.*?\n.*?\n0\s+([0-9.]+)\s+[0-9.]+\s*\n1\s+([0-9.]+)", s, re.S)
first, second = (float(m.group(1)), float(m.group(2))) if m else (float("nan"), float("nan"))
print(f"| {name} | {wr:.3f} | {first:.3f} | {second:.3f} | {ln:.1f} | |")
EOF
)
  echo "$row"
  rows+=("$row")
done

mean=$(printf '%s\n' "${rows[@]}" | awk -F'|' '$3 ~ /[0-9]/ {s += $3; n++} END {if (n) printf "%.3f", s / n; else print "n/a"}')
mkdir -p "$(dirname "$OUT")"
{
  echo "# ${cn} vs ${bn}"
  echo
  echo "- Candidate (agent1): \`$CKPT\`"
  echo "- Baseline (agent2): \`$BASE\`"
  echo "- \`--num_embeddings $NEMB\`, $N_GAMES games per deck (half with each seat), seed $SEED, greedy actions"
  [[ -n "$ONLY" ]] && echo "- Deck subset: $ONLY"
  echo "- Engine/env commit: \`$(cd "$REPO" && git rev-parse --short HEAD)\`, date $(date -u +%Y-%m-%d)"
  echo
  echo "| Deck | Candidate win rate | as first player | as second player | Avg length | Note |"
  echo "|---|---|---|---|---|---|"
  printf '%s\n' "${rows[@]}"
  echo "| **Mean over decks** | **$mean** | | | | |"
  echo
  echo "Logs: \`$LOGDIR\`"
} > "$OUT"
echo "wrote $OUT (mean win rate $mean)"
