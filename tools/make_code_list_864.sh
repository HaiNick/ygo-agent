#!/bin/bash
# Rebuild the v0.1-era 864-card code list (from dbf5142^) in main's
# "code has_script" format, taking has_script from main's code_list.txt.
set -euo pipefail
cd ~/ygo/ygo-agent
git show dbf5142^:scripts/code_list.txt \
  | awk 'NR==FNR { s[$1] = $2; next } { print $1, (($1 in s) ? s[$1] : 1) }' scripts/code_list.txt - \
  > ~/ygo/code_list_864.txt
wc -l ~/ygo/code_list_864.txt
head -3 ~/ygo/code_list_864.txt
echo "missing from main list: $(git show dbf5142^:scripts/code_list.txt | grep -vxFf <(cut -d' ' -f1 scripts/code_list.txt) | wc -l)"
echo "has_script=0 count: $(awk '$2 == 0' ~/ygo/code_list_864.txt | wc -l)"
