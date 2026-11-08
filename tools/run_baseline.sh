#!/usr/bin/env bash
# Baseline table for a checkpoint on a set of decks.
#
#   per deck: win rate vs the random bot (scripts/eval.py, N_RANDOM games, random seat)
#             mirror match vs itself    (scripts/battle.py, N_MIRROR games, both seats)
#
# In the mirror the same checkpoint plays both sides, so its win rate is ~50% by
# construction; the table reports the first-player win rate and game length.
#
# usage: tools/run_baseline.sh CKPT NUM_EMBEDDINGS [DECK_DIR] [OUT_MD]
#   env: N_RANDOM (256), N_MIRROR (128)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$(realpath "$1")"
NEMB="$2"
DECKS="${3:-$HOME/ygo/decks/mine}"
OUT="$(realpath -m "${4:-$REPO/docs/baseline.md}")"
N_RANDOM="${N_RANDOM:-256}"
N_MIRROR="${N_MIRROR:-128}"
LOGDIR="$HOME/ygo/selfplay_logs/baseline_$(basename "$CKPT" .flax_model)"
mkdir -p "$LOGDIR"

source "$HOME/ygo/.venv/bin/activate"
cd "$REPO/scripts"

shopt -s nullglob
decks=("$DECKS"/*.ydk)
(( ${#decks[@]} )) || { echo "no .ydk files in $DECKS" >&2; exit 1; }

rows=()
for f in "${decks[@]}"; do
  name="$(basename "$f" .ydk)"
  python eval.py --deck "$f" --checkpoint "$CKPT" --num_embeddings "$NEMB" \
    --bot_type random --num_episodes "$N_RANDOM" --num_envs 64 > "$LOGDIR/${name}_random.log" 2>&1
  python battle.py --deck "$f" --checkpoint1 "$CKPT" --checkpoint2 "$CKPT" --num_embeddings "$NEMB" \
    --num_episodes "$N_MIRROR" > "$LOGDIR/${name}_mirror.log" 2>&1
  row=$(python3 - "$LOGDIR/${name}_random.log" "$LOGDIR/${name}_mirror.log" "$name" <<'EOF'
import re, sys
r, m, name = open(sys.argv[1]).read(), open(sys.argv[2]).read(), sys.argv[3]
def last(pat, s, default="n/a"):
    x = re.findall(pat, s)
    return x[-1] if x else default
wr = last(r"win_rate=([0-9.]+)", r)
rl = last(r"len=([0-9.]+)", r)
ml = last(r"len=([0-9.]+)", m)
mw = last(r"win_rate=([0-9.]+)", m)
p0 = re.search(r"\n0\s+([0-9.]+)\s+([0-9.]+)", m)
first = f"{float(p0.group(1)) + float(p0.group(2)):.3f}" if p0 else "n/a"
err = " (error)" if "what()" in r + m or "Traceback" in r + m else ""
fmt = lambda x: f"{float(x):.3f}" if x != "n/a" else x
print(f"| {name}{err} | {fmt(wr)} | {float(rl):.1f} | {fmt(mw)} | {first} | {float(ml):.1f} |"
      if rl != "n/a" and ml != "n/a" else f"| {name}{err} | {fmt(wr)} | n/a | {fmt(mw)} | {first} | n/a |")
EOF
)
  echo "$row"
  rows+=("$row")
done

mkdir -p "$(dirname "$OUT")"
{
  echo "# Baseline before fine-tuning"
  echo
  echo "- Checkpoint: \`$(basename "$CKPT")\`, \`--num_embeddings $NEMB\`"
  echo "- Engine: ygopro-core 0.0.3, env at \`$(cd "$REPO" && git rev-parse --short HEAD)\`"
  echo "- Date: $(date -u +%Y-%m-%d)"
  echo "- vs random: \`scripts/eval.py --bot_type random\`, $N_RANDOM games, agent seat random, greedy (argmax) actions"
  echo "- Mirror: \`scripts/battle.py\`, checkpoint vs itself, $N_MIRROR games, half of the games with each seat"
  echo "- In the mirror both sides are the same model, so its win rate is ~0.5 by construction;"
  echo "  the first-player win rate and game length are the informative columns."
  echo
  echo "| Deck | Win rate vs random | Avg length vs random | Mirror win rate | Mirror first-player win rate | Mirror avg length |"
  echo "|---|---|---|---|---|---|"
  printf '%s\n' "${rows[@]}"
  echo
  echo "Logs: \`$LOGDIR\`"
} > "$OUT"
echo "wrote $OUT"
