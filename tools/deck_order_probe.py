"""Measure how much a model depends on the order of its own deck rows (docs/OBS_V2.md, issue A).

The env writes one row per card of the player's own deck into cards_, in core sequence order
(the last row is the top card). A real client doesn't know that order. This script changes the
observation of ONE seat in Python, after the env and before the model, and plays the model
against itself: seat A sees the changed observation, seat B the env's. The env isn't changed.

Own-deck rows are the rows of cards_ with column 2 (location) == 1 (deck) and column 4
(owner) == 0 (me). They are the first rows of the observation. Inside that block only columns
5 and 7 to 40 can differ between rows (position, attribute, race, level, counter, negated,
atk, def, types); columns 0, 1 (card id) are 0 for every row (K1), column 3 (sequence) is 0.

Variants (--variant):
  none      no change (control: shows the seat effect and the noise).
  sorted    the block's rows in a fixed order (sorted by their feature bytes), like a client
            that sends the remaining deck sorted.
  shuffled  the block's rows in a new random order at every decision.
  blank     the block's rows keep only location, owner and position (columns 7 to 40 zeroed):
            the model no longer sees which cards are left in its deck. Not a client view; a
            reference for how much the deck CONTENT matters.

  ids       not about the deck order (issue B, K1): seat A gets the real card id in columns 0, 1
            of every card row whose features are visible, instead of 0. The ids come from the
            env's oppo_info observation (all cards with ids, plus a visibility mask), from which
            the normal observation is rebuilt. --check-rebuild steps a normal env next to it and
            counts the steps where the rebuilt observation differs (expected 0).

Action index (--remap, default on): actions_ column 0 is 1 + the cards_ row of the action's
card. With --remap the index follows the card to its new row, so a deck search still points
at the right card (a client knows the code of every candidate from the message). With
--no-remap the index stays, so it points at whatever card now sits in that row: that is what a
client does that numbers its sorted deck 0..n-1 and then refers to cards by sequence.

--probe also runs seat A's decisions through the model with the unchanged observation and the
same recurrent state, and counts how often the greedy action differs and how far the logits move.

usage: python tools/deck_order_probe.py --deck DECK.ydk --checkpoint CKPT [--variant sorted]
         [--no-remap] [--probe] [--games 128] [--seed 0] [--env-threads 8] [--json FILE]
Set JAX_PLATFORMS=cpu and CUDA_VISIBLE_DEVICES="" to keep it off the GPU.
"""
import argparse, json, math, os, sys, time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--deck", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--num-embeddings", type=int, default=15017)
p.add_argument("--variant", default="none", choices=["none", "sorted", "shuffled", "blank", "ids"])
p.add_argument("--check-rebuild", action="store_true")
p.add_argument("--remap", dest="remap", action="store_true", default=True)
p.add_argument("--no-remap", dest="remap", action="store_false")
p.add_argument("--probe", action="store_true")
p.add_argument("--games", type=int, default=128)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--env-threads", type=int, default=8)
p.add_argument("--max-steps", type=int, default=20000, help="safety stop (batched steps)")
p.add_argument("--json")
p.add_argument("--obs-version", type=int, default=1, help="observation version of the env (docs/OBS_V2.md)")
args = p.parse_args()
deck_path = os.path.abspath(os.path.expanduser(args.deck))
ckpt_path = os.path.abspath(os.path.expanduser(args.checkpoint))
json_path = os.path.abspath(os.path.expanduser(args.json)) if args.json else None
os.chdir(ROOT / "scripts")
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
import flax
import ygoenv
from ygoai.utils import init_ygopro
from ygoai.rl.jax.agent import RNNAgent, ModelArgs

LOC_DECK = 1  # location_to_id(LOCATION_DECK)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def honest(cards, actions, rows_mask, variant, remap, rng):
    """Return changed copies of cards (N, R, 41) and actions (N, A, 12) for the envs in rows_mask."""
    cards = cards.copy(); actions = actions.copy()
    n_points = 0
    for i in np.nonzero(rows_mask)[0]:
        c = cards[i]
        deck = (c[:, 2] == LOC_DECK) & (c[:, 4] == 0)
        n = int(deck.sum())
        if n == 0:
            continue
        # own deck rows come first, so the block is rows 0..n-1
        assert deck[:n].all(), "own deck rows are not the first rows"
        idx = actions[i, :, 0].astype(np.int64)
        in_deck = (idx >= 1) & (idx <= n)
        n_points += int(in_deck.any())
        if variant == "blank":
            cards[i, :n, 7:] = 0
            continue
        block = c[:n]
        if variant == "sorted":
            order = np.lexsort(block[:, :4:-1].T)  # by columns 5..40, column 5 most significant
        else:
            order = rng.permutation(n)
        cards[i, :n] = block[order]  # new row j holds old row order[j]
        if remap and in_deck.any():
            inv = np.empty(n, dtype=np.int64); inv[order] = np.arange(n)
            actions[i, in_deck, 0] = (inv[idx[in_deck] - 1] + 1).astype(actions.dtype)
    return cards, actions, n_points


deck = init_ygopro("YGOPro-v1", "english", deck_path, "code_list.txt")
N = args.games
WITH_IDS = args.variant == "ids"
cfg = dict(task_id="YGOPro-v1", env_type="gymnasium", num_envs=N, num_threads=min(args.env_threads, N),
           seed=args.seed, deck1=deck, deck2=deck, player=-1, max_options=24, n_history_actions=32,
           play_mode="self", async_reset=False, verbose=False, record=False, obs_version=args.obs_version)
envs = ygoenv.make(oppo_info=WITH_IDS, **cfg)
check_env = ygoenv.make(oppo_info=False, **cfg) if (WITH_IDS and args.check_rebuild) else None
KEYS = ("cards_", "global_", "actions_", "h_actions_")
# mask_ column -> cards_ columns (ygopro.h _set_obs_mask_)
MASK_COLS = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
             (12, 14), (14, 16), (16, 41)]
rebuild_diff = 0
ids_obs = None


def pick(o):
    """The model's observation. For --variant ids: rebuild the normal observation from the
    oppo_info one and keep a copy of cards_ with the real ids in the visible rows."""
    global ids_obs
    if not WITH_IDS:
        return {k: o[k] for k in KEYS}
    g, mask = o["cards_"], o["mask_"]
    cards = np.zeros_like(g)
    for j, (lo, hi) in enumerate(MASK_COLS):
        cards[:, :, lo:hi] = g[:, :, lo:hi] * mask[:, :, j:j + 1]
    visible = mask[:, :, 6:7] == 1
    ids_obs = cards.copy()
    ids_obs[:, :, 0:2] = np.where(visible, g[:, :, 0:2], 0)
    cards[:, :, 0:2] = 0
    return dict(cards_=cards, global_=o["global_"], actions_=o["actions_"], h_actions_=o["h_actions_"])


agent = RNNAgent(**asdict(ModelArgs()), embedding_shape=args.num_embeddings)
obs, info = envs.reset()
obs = pick(obs)
if check_env is not None:
    cobs, _ = check_env.reset()
    rebuild_diff += int(not all(np.array_equal(obs[k], cobs[k]) for k in KEYS))
params = jax.jit(agent.init)(jax.random.PRNGKey(0), jax.tree.map(lambda x: jnp.asarray(x[:1]), obs),
                             agent.init_rnn_state(1))
with open(ckpt_path, "rb") as f:
    params = flax.serialization.from_bytes(params, f.read())
params = jax.device_put(params)
platform = str(jax.tree.leaves(params)[0].devices()) if hasattr(jax.tree.leaves(params)[0], "devices") else "?"


@jax.jit
def forward(params, rs_a, rs_b, obs, main, done):
    """One model call per step: every env's acting seat uses its own recurrent state."""
    rs = jax.tree.map(lambda a, b: jnp.where(main[:, None], a, b), rs_a, rs_b)
    nxt, logits = agent.apply(params, obs, rs)[:2]
    rs_a = jax.tree.map(lambda n, a: jnp.where(main[:, None], n, a), nxt, rs_a)
    rs_b = jax.tree.map(lambda n, b: jnp.where(main[:, None], b, n), nxt, rs_b)
    # like battle.py: zero both states after the blank terminal observation
    rs_a, rs_b = jax.tree.map(lambda x: jnp.where(done[:, None], 0, x), (rs_a, rs_b))
    return rs_a, rs_b, logits


@jax.jit
def logits_only(params, rs_a, rs_b, obs, main):
    rs = jax.tree.map(lambda a, b: jnp.where(main[:, None], a, b), rs_a, rs_b)
    return agent.apply(params, obs, rs)[1]


rng = np.random.default_rng(args.seed + 12345)
rs_a = agent.init_rnn_state(N)
rs_b = agent.init_rnn_state(N)
# seat A (the changed observation) is player 0 in the first half of the envs, player 1 in the rest
seat_a = np.concatenate([np.zeros(N // 2, dtype=np.int64), np.ones(N - N // 2, dtype=np.int64)])
to_play = info["to_play"]
done = np.zeros(N, dtype=bool)
collected = np.zeros(N, dtype=bool)
ep_ret = np.zeros(N, dtype=np.float64)
ep_len = np.zeros(N, dtype=np.int64)
res = []  # (env, seat of A, A won, player 0 won, length)
st = dict(decisions_a=0, decisions_a_multi=0, with_deck_rows=0, deck_pointing=0,
          probe_n=0, probe_flip=0, probe_flip_deck_pointing=0, probe_deck_pointing=0,
          probe_max_abs=0.0, probe_sum_abs=0.0)
step = 0
t0 = time.time()
while not collected.all() and step < args.max_steps:
    main = to_play == seat_a
    live = ~collected & (obs["global_"][:, 22] == 0)
    nopt = info["num_options"]
    obs_in = obs
    if WITH_IDS:
        sel_rows = (main & live)[:, None, None]
        obs_in = dict(obs, cards_=np.where(sel_rows, ids_obs, obs["cards_"]))
    elif args.variant != "none":
        cards, actions, _ = honest(obs["cards_"], obs["actions_"], main & live, args.variant, args.remap, rng)
        obs_in = dict(obs, cards_=cards, actions_=actions)
    # bookkeeping on seat A's real decisions
    sel = main & live
    st["decisions_a"] += int(sel.sum())
    multi = sel & (nopt > 1)
    st["decisions_a_multi"] += int(multi.sum())
    n_deck = ((obs["cards_"][:, :, 2] == LOC_DECK) & (obs["cards_"][:, :, 4] == 0)).sum(axis=1)
    st["with_deck_rows"] += int((sel & (n_deck > 1)).sum())
    a_idx = obs["actions_"][:, :, 0].astype(np.int64)
    points = ((a_idx >= 1) & (a_idx <= n_deck[:, None])).any(axis=1)
    st["deck_pointing"] += int((sel & points).sum())

    rs_a_prev, rs_b_prev = rs_a, rs_b
    rs_a, rs_b, logits = forward(params, rs_a, rs_b, obs_in, main, done)
    logits = np.asarray(logits)
    action = logits.argmax(axis=1)
    if args.probe and args.variant != "none" and multi.any():
        ref = np.asarray(logits_only(params, rs_a_prev, rs_b_prev, obs, main))
        ref_action = ref.argmax(axis=1)
        valid = np.arange(logits.shape[1])[None, :] < nopt[:, None]
        diff = np.where(valid, np.abs(logits - ref), 0.0).max(axis=1)
        flip = (ref_action != action) & multi
        st["probe_n"] += int(multi.sum())
        st["probe_flip"] += int(flip.sum())
        st["probe_deck_pointing"] += int((multi & points).sum())
        st["probe_flip_deck_pointing"] += int((flip & points).sum())
        st["probe_max_abs"] = max(st["probe_max_abs"], float(diff[multi].max()))
        st["probe_sum_abs"] += float(diff[multi].sum())

    prev_to_play = to_play
    obs, rew, term, trunc, info = envs.step(action.astype(np.int32))
    obs = pick(obs)
    if check_env is not None:
        cobs = check_env.step(action.astype(np.int32))[0]
        rebuild_diff += int(not all(np.array_equal(obs[k], cobs[k]) for k in KEYS))
    to_play = info["to_play"]
    done = term | trunc
    step += 1
    ep_ret += rew
    ep_len += 1
    for i in np.nonzero(done)[0]:
        r, ln = ep_ret[i], ep_len[i]
        ep_ret[i] = 0; ep_len[i] = 0
        if collected[i]:
            continue
        collected[i] = True
        # the reward is from the view of the player who made the last move
        a_won = (r > 0) == bool(main[i])
        p0_won = (r > 0) == (prev_to_play[i] == 0)
        res.append((int(i), int(seat_a[i]), int(a_won), int(p0_won), int(ln)))
    if step % 200 == 0:
        print(f"step {step} games {int(collected.sum())}/{N} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

res = np.array(res, dtype=np.int64).reshape(-1, 5)
n = len(res)
k = int(res[:, 2].sum())
lo, hi = wilson(k, n)
first = res[res[:, 1] == 0]; second = res[res[:, 1] == 1]
out = dict(
    deck=Path(deck_path).stem, checkpoint=Path(ckpt_path).name, variant=args.variant,
    remap=bool(args.remap), seed=args.seed, games=n, a_wins=k, a_win_rate=k / max(n, 1),
    ci95=[lo, hi], a_first_games=len(first), a_first_wins=int(first[:, 2].sum()),
    a_second_games=len(second), a_second_wins=int(second[:, 2].sum()),
    player0_wins=int(res[:, 3].sum()), mean_len=float(res[:, 4].mean()) if n else 0.0,
    rebuild_checked=check_env is not None, rebuild_diff_steps=rebuild_diff,
    batched_steps=step, seconds=round(time.time() - t0, 1), device=platform, **st)
if st["probe_n"]:
    out["probe_flip_rate"] = st["probe_flip"] / st["probe_n"]
    out["probe_mean_abs"] = st["probe_sum_abs"] / st["probe_n"]
print(json.dumps(out))
if json_path:
    with open(json_path, "w") as f:
        json.dump(out, f, indent=1)
