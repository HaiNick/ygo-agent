#!/usr/bin/env bash
# One command for the acceptance checks. Run it from any checkout or worktree;
# it builds and tests THAT checkout and never touches another one.
#
# usage: tools/run_checks.sh [quick|full]      (default: quick)
#   env: CKPT       model for the model checks (default: filled 0546)
#        DECKS      deck folder for the per-deck checks (default: ~/ygo/decks/mine)
#        POOL       training pool (default: ~/ygo/decks/train)
#        REF        reference checkout for the determinism check (optional): a
#                   folder with a built ygoenv/ tree, for example the main
#                   checkout. Without it the digests are only printed.
#        GAMES      games for tripwire and self-play in full mode (default 256)
#   While training runs: XLA_PYTHON_CLIENT_PREALLOCATE=false is set here.
#
# quick: build, duel registry stress, embedding rows, record + yrp_verify (2 decks),
#        determinism digests, finite model outputs, ygoinf encoding, deck lint
# full:  quick + record all decks, tripwire (vs random and vs the option-0 bot),
#        random self-play per deck (0 Lua errors, 0 crashes)
set -uo pipefail
MODE="${1:-quick}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${CKPT:-$REPO/scripts/checkpoints/0546_22750M_main15017_filled.flax_model}"
DECKS="${DECKS:-$HOME/ygo/decks/mine}"
POOL="${POOL:-$HOME/ygo/decks/train}"
GAMES="${GAMES:-256}"
REF="${REF:-}"
LOG="$(mktemp -d)"
# shellcheck disable=SC1091
source "$HOME/ygo/.venv/bin/activate"
export CMAKE_POLICY_VERSION_MINIMUM=3.5 PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO/ygoenv:$REPO" XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$REPO"
pass=0; fail=0
step() {  # step NAME CMD...
  local name="$1"; shift
  local log="$LOG/$(echo "$name" | tr ' /' '__').log"
  if "$@" > "$log" 2>&1; then pass=$((pass + 1)); echo "PASS  $name  $(tail -1 "$log" | cut -c1-110)"
  else fail=$((fail + 1)); echo "FAIL  $name  (log: $log)"; tail -4 "$log" | cut -c1-200 | sed 's/^/      /'; fi
}
for f in scripts/script/constant.lua assets/locale/en/cards.cdb "$CKPT"; do
  [[ -e "$f" ]] || { echo "missing $f (link scripts/script, assets/locale and scripts/checkpoints into this checkout)"; exit 2; }
done

step "build" bash -c 'xmake f -m release -y >/dev/null && xmake b -y ygopro_ygoenv && xmake b -y yrp_verify && xmake b -y duel_registry_stress'
STRESS="$(find build -name duel_registry_stress -type f | head -1)"
step "duel registry stress" "$STRESS" locked 16 20000
step "embedding rows" python tools/test_embedding_rows.py "$CKPT" --num-embeddings 15017
mapfile -t all_decks < <(ls "$DECKS"/*.ydk)
if [[ "$MODE" == full ]]; then rec=("${all_decks[@]}"); else rec=("${all_decks[@]:0:2}"); fi
for d in "${rec[@]}"; do step "record+verify $(basename "$d" .ydk)" tools/record_and_verify.sh "$d" 2; done
digests() { for d in "$DECKS" "$POOL"; do python tools/dump_trajectory.py "$d" --games 32 --clamp-level 13 2>/dev/null | tail -1; done; }
determinism() {
  local new ref; new="$(digests)"; echo "$new"
  if [[ -n "$REF" ]]; then
    ref="$(PYTHONPATH="$REF/ygoenv:$REPO" digests)"
    [[ "$new" == "$ref" ]] && echo "identical to $REF" || { echo "DIFFERENT from $REF:"; echo "$ref"; return 1; }
  else
    echo "no REF given: digests printed, not compared"
  fi
}
step "determinism (32 games per deck set)" determinism
step "finite model outputs" python tools/check_finite.py "$CKPT" --deck "$POOL" --steps 300 --envs 64
step "ygoinf encoding" python tools/test_ygoinf_encoding.py --deck "$DECKS"
step "deck lint (strict)" python tools/lint_decks.py "$DECKS" "$POOL" --strict

if [[ "$MODE" == full ]]; then
  # Reference (filled 0546, decks/mine, 256 games, seed 0, builds 15c7726 and
  # audit-fixes give the same numbers): 1.000 vs random, 0.953 vs the option-0
  # bot (0.973 on the training pool). A broken observation drops both far lower.
  tripwire() {  # tripwire BOT_TYPE MIN_WIN_RATE
    (cd scripts && python eval.py --deck "$DECKS" --checkpoint "$CKPT" --num_embeddings 15017 --bot_type "$1" \
      --num_episodes "$GAMES" --num_envs 64 --seed 0 2>&1 | grep -a -E "win_rate=|Traceback|what\(\)" | tail -1) | tee "$LOG/tw.txt"
    python - "$LOG/tw.txt" "$2" <<'EOF'
import re, sys
m = re.search(r"win_rate=([0-9.]+)", open(sys.argv[1]).read())
sys.exit(0 if m and float(m.group(1)) >= float(sys.argv[2]) else 1)
EOF
  }
  step "tripwire vs random ($GAMES games, min ${MIN_RANDOM:-0.98})" tripwire random "${MIN_RANDOM:-0.98}"
  step "tripwire vs option-0 bot ($GAMES games, min ${MIN_BOT:-0.93})" tripwire greedy "${MIN_BOT:-0.93}"
  selfplay() {  # selfplay DECK.ydk: GAMES random self-play games, no Lua errors, clean exit
    local out="$LOG/sp_$(basename "$1").txt"
    python tools/dump_trajectory.py "$1" --games "$GAMES" --envs 16 > "$out" 2>&1 || { tail -3 "$out"; return 1; }
    local n; n="$(grep -c "lua-error" "$out" || true)"
    echo "$(tail -1 "$out" | cut -d' ' -f3-) lua_errors $n"
    [[ "$n" == 0 ]]
  }
  for d in "${all_decks[@]}"; do step "self-play $(basename "$d" .ydk)" selfplay "$d"; done
fi
echo "---- $pass passed, $fail failed (logs in $LOG)"
exit $(( fail > 0 ))
