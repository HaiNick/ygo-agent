#!/usr/bin/env bash
# Record random-play games with the env's replay writer and check them:
#   1. tests/yrp_verify replays each file through the core (must reach MSG_WIN
#      with every response used)
#   2. both 40-byte player name fields must contain a NUL terminator
#
# usage: tools/record_and_verify.sh DECK.ydk [GAMES=2]
#   Run with the env build on PYTHONPATH (default: this checkout). Needs
#   `xmake b yrp_verify` once. Replays go to a temporary folder.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DECK="$(realpath "$1")"
GAMES="${2:-2}"
V="$(find "$REPO/build" -name yrp_verify -type f | head -1)"
[[ -x "$V" ]] || { echo "build tests first: xmake b yrp_verify" >&2; exit 1; }
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/replay"
ln -s "$REPO/scripts/script" "$TMP/script"
cp "$REPO/scripts/code_list.txt" "$TMP/"
export PYTHONPATH="${PYTHONPATH:-$REPO/ygoenv:$REPO}"
cd "$TMP"
python - "$DECK" "$GAMES" "$REPO" > "$TMP/play.log" 2>&1 <<'EOF'
import sys
import numpy as np
import ygoenv
from pathlib import Path
deck_path, games, repo = sys.argv[1], int(sys.argv[2]), sys.argv[3]
from ygoenv.ygopro import init_module
name = Path(deck_path).stem
init_module(str(Path(repo, "assets/locale/en/cards.cdb")), "code_list.txt", {name: deck_path})
envs = ygoenv.make(task_id="YGOPro-v1", env_type="gymnasium", num_envs=1, num_threads=1, seed=11,
                   deck1=name, deck2=name, max_options=24, n_history_actions=32, play_mode="self",
                   async_reset=False, verbose=True, record=True)
rng = np.random.default_rng(3)
obs, info = envs.reset()
done = 0
while done < games:
    a = (rng.integers(0, 1 << 30, size=1) % np.maximum(info["num_options"], 1)).astype(np.int32)
    obs, r, term, trunc, info = envs.step(a)
    done += int((term | trunc)[0])
EOF
n=$(ls "$TMP/replay"/*.yrp 2>/dev/null | wc -l)
[[ "$n" -ge 1 ]] || { echo "FAIL: no replay written"; tail -5 "$TMP/play.log"; exit 1; }
"$V" "$REPO/assets/locale/en/cards.cdb" "$REPO/scripts/script" "$TMP/replay"/*.yrp | grep -E "PASS|FAIL|replays passed"
python - "$TMP/replay" <<'EOF'
import glob, lzma, struct, sys
bad = 0
for p in sorted(glob.glob(sys.argv[1] + "/*.yrp")):
    b = open(p, "rb").read()
    hid, ver, flag, seed, datasize = struct.unpack_from("<5I", b, 0)
    off = 32 if hid == 0x31707279 else 80
    body = b[off:]
    if flag & 1:
        props = b[24:29]
        filt = [lzma._decode_filter_properties(lzma.FILTER_LZMA1, props)]
        body = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=filt).decompress(body, max_length=datasize)
    names = [body[i * 40:(i + 1) * 40].decode("utf-16-le", "replace") for i in range(2)]
    ok = all("\x00" in n for n in names)
    bad += not ok
    print("names", "ok  " if ok else "FAIL", [n.split("\x00")[0] for n in names])
sys.exit(1 if bad else 0)
EOF
