#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YGO="${YGO_HOME:-$HOME/ygo}"

export PATH="$HOME/.local/node/bin:$HOME/.local/bin:$PATH"

# 1. Ensure ygoinf is running on 127.0.0.1:3000 (CPU)
if ! curl -fsS http://127.0.0.1:3000/ >/dev/null 2>&1; then
    tmux kill-session -t ygoinf 2>/dev/null || true
    tmux new-session -d -s ygoinf "bash $REPO/tools/start_inf_server.sh > $YGO/ygoinf.log 2>&1"
fi

# 2. Start ygopro-server loop on 127.0.0.1:7911
tmux kill-session -t ygopro_room 2>/dev/null || true
tmux new-session -d -s ygopro_room "cd $YGO/local_server && while true; do ./ygopro-server 7911 -1 5 0 F T F 8000 5 1 0 0 >> $YGO/local_server/room.log 2>&1 || true; sleep 0.5; done"

# 3. Start YGOPro packet-aware WebSocket <-> TCP proxy (ws://127.0.0.1:7912 -> 127.0.0.1:7911)
tmux kill-session -t ws_proxy 2>/dev/null || true
tmux new-session -d -s ws_proxy "$YGO/.venv/bin/python -u $REPO/tools/ws_tcp_proxy.py --ws-host 127.0.0.1 --ws-port 7912 --tcp-host 127.0.0.1 --tcp-port 7911 > $YGO/local_server/ws_proxy.log 2>&1"

# 4. Start local neos-ts web client on http://127.0.0.1:5173
tmux kill-session -t neos_web 2>/dev/null || true
tmux new-session -d -s neos_web "export PATH=$HOME/.local/node/bin:\$PATH && cd $YGO/src/neos-ts && npx vite --host 127.0.0.1 --port 5173 > $YGO/local_server/neos_web.log 2>&1"

# 5. Add Local_Offline (127.0.0.1:7911) to MDPro4 hosts2.conf if present
for hfile in /mnt/d/Games/MDPro4/Data/hosts2.conf /mnt/d/Games/MDPro4/Data/hosts.conf; do
    if [ -f "$hfile" ] && ! grep -q "127.0.0.1" "$hfile"; then
        sed -i '1i\[Local_Offline]\nHost = 127.0.0.1\nPort = 7911\n' "$hfile" 2>/dev/null || true
    fi
done

echo "Waiting for services to come up..."
sleep 3
echo "=== Service Status ==="
echo -n "1) ygoinf (http://127.0.0.1:3000/): "
curl -fsS http://127.0.0.1:3000/ && echo ""
echo -n "2) neos-ts (http://127.0.0.1:5173/): "
curl -fsSI http://127.0.0.1:5173/ | head -n 1
echo "3) Listening TCP/WS ports (3000, 5173, 7911, 7912):"
ss -tlnp | grep -E ":(3000|5173|7911|7912)\b" || true
