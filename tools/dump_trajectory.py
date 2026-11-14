"""Determinism check: play fixed-seed games with a seeded random policy and hash everything.

Two env builds are behavior-identical if this prints the same digest for both. Run it once per
build (select the build with PYTHONPATH=<checkout>/ygoenv:<checkout>) and compare the output.

usage: python tools/dump_trajectory.py DECK_DIR [--games 32] [--envs 8] [--seed 7] [--out FILE]
       [--timeout 600]
The last line is "digest <sha256> games <n> steps <n>". --out writes one hash per step.
"""
import argparse, hashlib, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("deck")
p.add_argument("--games", type=int, default=32)
p.add_argument("--envs", type=int, default=8)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--timeout", type=int, default=600)
p.add_argument("--out")
p.add_argument("--clamp-level", type=int, default=None,
               help="clamp the card level feature before hashing (to compare with builds before the env clamped it)")
args = p.parse_args()
deck_path = os.path.abspath(os.path.expanduser(args.deck))
os.chdir(ROOT / "scripts")

import numpy as np
import ygoenv
from ygoai.utils import init_ygopro

deck = init_ygopro("YGOPro-v1", "english", deck_path, "code_list.txt")
n = args.envs
envs = ygoenv.make(
    task_id="YGOPro-v1", env_type="gymnasium", num_envs=n, num_threads=n, seed=args.seed,
    deck1=deck, deck2=deck, max_options=24, n_history_actions=32, play_mode="self",
    async_reset=False, greedy_reward=False, timeout=args.timeout, oppo_info=False)
rng = np.random.default_rng(args.seed)
KEYS = ("cards_", "global_", "actions_", "h_actions_")
total = hashlib.sha256()
out = open(args.out, "w") if args.out else None


def record(step, obs, extra):
    h = hashlib.sha256()
    for k in KEYS:
        a = obs[k]
        if k == "cards_" and args.clamp_level is not None:
            a = a.copy(); a[:, :, 9] = np.minimum(a[:, :, 9], args.clamp_level)
        h.update(np.ascontiguousarray(a).tobytes())
    for x in extra:
        h.update(np.ascontiguousarray(x).tobytes())
    total.update(h.digest())
    if out:
        out.write(f"{step} {h.hexdigest()}\n")

obs, info = envs.reset()
record(0, obs, (info["num_options"], info["to_play"]))
# each env contributes its first games // envs episodes, so the set of games is fixed
quota = np.full(n, args.games // n); quota[: args.games % n] += 1
done_games = np.zeros(n, dtype=np.int64)
step = 0
while (done_games < quota).any():
    step += 1
    nopt = np.maximum(info["num_options"], 1)
    action = (rng.integers(0, 1 << 30, size=n) % nopt).astype(np.int32)
    obs, rew, term, trunc, info = envs.step(action)
    record(step, obs, (action, rew, term, trunc, info["num_options"], info["to_play"]))
    done_games += (term | trunc)
if out:
    out.close()
print(f"digest {total.hexdigest()} games {args.games} steps {step * n}")
