"""Walk every core message buffer of recorded games with the message length table.

The env's observation version 2 processes every message of a buffer (docs/OBS_V2.md, F1). It
finds the end of a message with a length table (ygopro.h msg_payload_len), a port of the table
in tools/ygoinf_parity.py (msg_len). A wrong length would desync the parser, so this check
walks every buffer that the recording build (tests/ygopro_parity.so) produces and counts:

  misparse   the table has no length for a message id, a length runs past the buffer, or the
             walk doesn't end exactly at the end of the buffer
  decision not last   a message that asks for a decision is followed by another message

The games are random self-play with random decks from the folder, one env.

usage: python tools/msg_len_check.py [--deck DIR] [--games 40] [--seed 0] [--obs-version 1]
exit code 0 = no misparse.
"""
import argparse, os, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ygoinf_parity as yp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--deck", default="~/ygo/decks/mine")
ap.add_argument("--games", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--obs-version", type=int, default=1)
a = ap.parse_args()
deck_dir = os.path.abspath(os.path.expanduser(a.deck))

import numpy as np  # noqa: E402

P, penv, _, db, decks = yp.make_envs(deck_dir, "random", "random", a.seed, False, a.obs_version)
rng = np.random.default_rng(a.seed)
n_buf = n_msg = n_bytes = misparse = not_last = 0
types = Counter()
examples = []


def check(buf):
    """Number of messages, or raise ValueError."""
    global not_last
    p, n = 0, 0
    while p < len(buf):
        m = buf[p]; p += 1
        try:
            ln = yp.msg_len(m, buf, p)
        except IndexError:
            raise ValueError(f"length of message {m} at {p - 1} reads past the buffer")
        if p + ln > len(buf):
            raise ValueError(f"message {m} at {p - 1} with length {ln} runs past the buffer ({len(buf)} bytes)")
        types[yp.MSG_NAME.get(m, str(m))] += 1
        p += ln; n += 1
        if m in yp.DECISIONS and p < len(buf):
            not_last += 1
    return n


obs, info = penv.reset()
done = steps = 0
while done < a.games:
    for kind, data in P.drain():
        if kind != 1:
            continue
        n_buf += 1; n_bytes += len(data)
        try:
            n_msg += check(data)
        except ValueError as e:
            misparse += 1
            if len(examples) < 5:
                examples.append(f"{e}: {data[:64].hex()}")
    act = int(rng.integers(0, 1 << 30) % max(int(info["num_options"][0]), 1))
    obs, rew, term, trunc, info = penv.step(np.array([act], dtype=np.int32))
    steps += 1
    if term[0] or trunc[0]:
        done += 1
print(f"obs_version {a.obs_version}: {done} games, {steps} env steps, {n_buf} buffers, {n_bytes} bytes, "
      f"{n_msg} messages of {len(types)} types, misparses {misparse}, decision not last in its buffer {not_last}")
for e in examples:
    print("  " + e)
print("messages by type: " + ", ".join(f"{k} {v}" for k, v in types.most_common()))
sys.exit(1 if misparse else 0)
