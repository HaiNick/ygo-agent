"""Run sampled self-play through the model and fail if any logits or value are non-finite.

Catches observation features that index outside an embedding table (Flax returns NaN for those),
as a card level above 13 did before the env clamped it.

usage: python tools/check_finite.py CKPT [--deck DIR] [--steps 400] [--envs 64] [--num-embeddings 15017]
exit code 0 = every decision had finite logits (over the legal options) and a finite value.
"""
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--deck", default="~/ygo/decks/train")
p.add_argument("--steps", type=int, default=400)
p.add_argument("--envs", type=int, default=64)
p.add_argument("--num-embeddings", type=int, default=15017)
p.add_argument("--seed", type=int, default=77)
args = p.parse_args()
ckpt = os.path.abspath(os.path.expanduser(args.checkpoint))
deck_path = os.path.abspath(os.path.expanduser(args.deck))
os.chdir(ROOT / "scripts")

import numpy as np
import ygoenv
import jax, jax.numpy as jnp, flax
from ygoai.utils import init_ygopro
from ygoai.rl.jax.agent import RNNAgent

n = args.envs
deck = init_ygopro("YGOPro-v1", "english", deck_path, "code_list.txt")
envs = ygoenv.make(task_id="YGOPro-v1", env_type="gymnasium", num_envs=n, num_threads=min(n, 12), seed=args.seed,
                   deck1=deck, deck2=deck, max_options=24, n_history_actions=32, play_mode="self",
                   async_reset=False, greedy_reward=False, oppo_info=False)
agent = RNNAgent(embedding_shape=args.num_embeddings)
sample_obs = jax.tree.map(lambda x: jnp.array([x]), envs.observation_space.sample())
params = jax.jit(agent.init)(jax.random.PRNGKey(0), sample_obs, agent.init_rnn_state(1))
with open(ckpt, "rb") as f:
    params = flax.serialization.from_bytes(params, f.read())


@jax.jit
def fwd(params, obs, rstate, key):
    rstate, logits, value = agent.apply(params, obs, rstate)[:3]
    u = jax.random.uniform(key, shape=logits.shape)
    return rstate, logits, value[:, 0], jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=-1)

obs, info = envs.reset()
rstate = agent.init_rnn_state(n)
key = jax.random.PRNGKey(args.seed)
done = np.zeros(n, bool)
bad = 0; total = 0; max_level = 0
for _ in range(args.steps):
    key, sub = jax.random.split(key)
    o = dict(obs); o["mask_"] = None
    rstate, logits, value, action = fwd(params, o, rstate, sub)
    logits, value, action = np.asarray(logits), np.asarray(value), np.asarray(action)
    nopt = np.maximum(info["num_options"], 1)
    ok = np.isfinite(value) & np.array([np.isfinite(logits[i, :nopt[i]]).all() for i in range(n)])
    bad += int((~ok).sum()); total += n
    max_level = max(max_level, int(obs["cards_"][:, :, 9].max()))
    rstate = tuple(np.where(np.isfinite(np.asarray(x)), np.asarray(x), 0) for x in rstate)
    rstate = tuple(np.where(done[:, None], 0, x) for x in rstate)
    action = np.minimum(np.where(ok, action, 0), nopt - 1).astype(np.int32)
    obs, _, term, trunc, info = envs.step(action)
    done = term | trunc
print(f"decisions={total} non_finite={bad} max_level_feature={max_level}")
sys.exit(1 if bad else 0)
