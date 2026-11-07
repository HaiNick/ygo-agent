#!/usr/bin/env bash
# Build the training deck pool.
#
#   train/ = stock assets/deck/*.ydk + all of mine/*.ydk + N extra copies of the base deck
#
# The env samples decks uniformly per .ydk file (see ygopro.h load_deck: "random"
# picks uniformly from deck_names_, keyed by file stem, independently for each
# seat). Identical files therefore count separately, so N copies of the base deck
# give it weight (1 + N) / total.
#
# Usage: tools/build_train_pool.sh [-n COPIES] [-b BASE_DECK] [-o OUT_DIR]
#   -n  extra copies of the base deck (default 10)
#   -b  base deck file (default ~/ygo/decks/mine/00_Base_Branded-Despia.ydk)
#   -o  output directory (default ~/ygo/decks/train); existing *.ydk in it are replaced
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STOCK="$REPO/assets/deck"
MINE="${MINE:-$HOME/ygo/decks/mine}"
COPIES=10
BASE="$MINE/00_Base_Branded-Despia.ydk"
OUT="$HOME/ygo/decks/train"

while getopts "n:b:o:h" opt; do
  case $opt in
    n) COPIES="$OPTARG" ;;
    b) BASE="$OPTARG" ;;
    o) OUT="$OPTARG" ;;
    h) sed -n '2,16p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done

[[ "$COPIES" =~ ^[0-9]+$ ]] || { echo "copies must be a non-negative integer" >&2; exit 2; }
[[ -f "$BASE" ]] || { echo "base deck not found: $BASE" >&2; exit 1; }

mkdir -p "$OUT"
find "$OUT" -maxdepth 1 -name '*.ydk' -delete

# Deck names are file stems; a duplicate stem would silently overwrite a deck in
# init_ygopro's dict, so refuse collisions.
declare -A seen
add() {
  local src="$1" dst_name="$2"
  if [[ -n "${seen[$dst_name]:-}" ]]; then
    echo "deck name collision: $dst_name ($src vs ${seen[$dst_name]})" >&2
    exit 1
  fi
  seen[$dst_name]="$src"
  cp "$src" "$OUT/$dst_name.ydk"
}

n_stock=0; n_mine=0
for f in "$STOCK"/*.ydk; do add "$f" "$(basename "$f" .ydk)"; n_stock=$((n_stock + 1)); done
for f in "$MINE"/*.ydk;  do add "$f" "$(basename "$f" .ydk)"; n_mine=$((n_mine + 1)); done
base_stem="$(basename "$BASE" .ydk)"
for i in $(seq 1 "$COPIES"); do add "$BASE" "$(printf '%s_copy%02d' "$base_stem" "$i")"; done

total=$((n_stock + n_mine + COPIES))
echo "train pool: $OUT"
echo "  stock: $n_stock  mine: $n_mine  base copies: $COPIES  total: $total"
awk -v c="$COPIES" -v t="$total" 'BEGIN { printf "  base deck weight per seat: %d/%d = %.1f%%\n", c + 1, t, 100 * (c + 1) / t }'
