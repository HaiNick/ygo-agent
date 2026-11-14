#!/usr/bin/env bash
set -euo pipefail

for sess in neos_web ws_proxy ygoinf ygopro_room; do
    if tmux has-session -t "$sess" 2>/dev/null; then
        tmux kill-session -t "$sess"
        echo "Stopped tmux session: $sess"
    fi
done

echo "=== Remaining tmux sessions ==="
tmux ls || true
