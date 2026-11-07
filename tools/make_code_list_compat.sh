#!/bin/bash
# v0.1-compatible code list for main's env: the 864 v0.1 cards first, in
# original order (card IDs 1..864 match the checkpoint), then every other
# card from main's code_list.txt so on-demand lookups (tokens, etc.) succeed.
set -euo pipefail
cd ~/ygo/ygo-agent
out=~/ygo/code_list_v01compat.txt
cp ~/ygo/code_list_864.txt "$out"
awk 'NR==FNR { seen[$1] = 1; next } !($1 in seen)' ~/ygo/code_list_864.txt scripts/code_list.txt >> "$out"
wc -l "$out"
sed -n '864,866p' "$out"
