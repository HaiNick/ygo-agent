"""Compare the outputs of two checkpoints on the same observations.

Plays sampled self-play with checkpoint A (both seats) and runs every decision through
checkpoint B too, with the same observation and the same recurrent state. Reports the largest
absolute difference of the logits (over the legal options), of the value and of the next
recurrent state, and the number of decisions whose greedy action differs.

Use: after tools/obs_v2_ckpt.py, the new checkpoint must give the same outputs as its source on
observation version 1 wherever the counter feature is 0 or 15. Decisions with a card row whose
counter feature is 1 to 14 are reported separately ("mid"): the script changed those rows of the
table. On version 1 there are none in the runs so far. On version 2 they are the cards with 1 to
14 counters, and there the two checkpoints differ on purpose.

Run it on the CPU (JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""). On the GPU the two forward passes
of one compiled function don't give bit-equal results: a checkpoint compared with itself showed
logit differences up to 0.05 and 56 different greedy actions in 25,600 decisions.

usage: python tools/compare_ckpt_outputs.py CKPT_A CKPT_B [--deck DIR] [--steps 300] [--envs 64]
         [--obs-version 1] [--num-embeddings 15017] [--seed 0] [--tol 0]
exit code 0 = every difference outside the "mid" decisions is at most --tol.
"""
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("ckpt_a")
p.add_argument("ckpt_b")
p.add_argument("--deck", default="~/ygo/decks/train")
p.add_argument("--steps", type=int, default=300)
p.add_argument("--envs", type=int, default=64)
p.add_argument("--obs-version", type=int, default=1)
p.add_argument("--num-embeddings", type=int, default=15017)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tol", type=float, default=0.0)
args = p.parse_args()
ck_a = os.path.abspath(os.path.expanduser(args.ckpt_a))
ck_b = os.path.abspath(os.path.expanduser(args.ckpt_b))
deck_path = os.path.abspath(os.path.expanduser(args.deck))
os.chdir(ROOT / "scripts")
sys.path.insert(0, str(ROOT))

import numpy as np
import ygoenv
import jax, jax.numpy as jnp, flax
from ygoai.utils import init_ygopro
from ygoai.rl.jax.agent import RNNAgent

n = args.envs
deck = init_ygopro("YGOPro-v1", "english", deck_path, "code_list.txt")
envs = ygoenv.make(task_id="YGOPro-v1", env_type="gymnasium", num_envs=n, num_threads=min(n, 8), seed=args.seed,
                   deck1=deck, deck2=deck, max_options=24, n_history_actions=32, play_mode="self",
                   async_reset=False, greedy_reward=False, oppo_info=False, obs_version=args.obs_version)
agent = RNNAgent(embedding_shape=args.num_embeddings)
sample_obs = jax.tree.map(lambda x: jnp.array([x]), envs.observation_space.sample())
init = jax.jit(agent.init)(jax.random.PRNGKey(0), sample_obs, agent.init_rnn_state(1))


def load(path):
    with open(path, "rb") as f:
        return flax.serialization.from_bytes(init, f.read())


pa, pb = load(ck_a), load(ck_b)


@jax.jit
def fwd(pa, pb, obs, rstate, key):
    ra, la, va = agent.apply(pa, obs, rstate)[:3]
    rb, lb, vb = agent.apply(pb, obs, rstate)[:3]
    u = jax.random.uniform(key, shape=la.shape)
    action = jnp.argmax(la - jnp.log(-jnp.log(u)), axis=-1)
    d_state = jnp.max(jnp.stack([jnp.max(jnp.abs(x - y)) for x, y in zip(ra, rb)]))
    return ra, la, lb, va[:, 0], vb[:, 0], action, d_state


obs, info = envs.reset()
rstate = agent.init_rnn_state(n)
key = jax.random.PRNGKey(args.seed)
done = np.zeros(n, bool)
total = greedy_diff = 0
d_logit = d_value = d_state = 0.0
counter_rows = 0
mid_rows = mid_decisions = mid_greedy = 0
m_logit = m_value = 0.0
for _ in range(args.steps):
    key, sub = jax.random.split(key)
    o = dict(obs); o["mask_"] = None
    rstate, la, lb, va, vb, action, ds = fwd(pa, pb, o, rstate, sub)
    la, lb, va, vb, action = (np.asarray(x) for x in (la, lb, va, vb, action))
    nopt = np.maximum(info["num_options"], 1)
    cnt = obs["cards_"][:, :, 10]
    mid = ((cnt >= 1) & (cnt <= 14)).any(axis=1)
    mid_rows += int(((cnt >= 1) & (cnt <= 14)).sum()); mid_decisions += int(mid.sum())
    for i in range(n):
        k = int(nopt[i])
        dl = float(np.abs(la[i, :k] - lb[i, :k]).max())
        dg = int(la[i, :k].argmax() != lb[i, :k].argmax())
        dv = float(abs(va[i] - vb[i]))
        if mid[i]:
            m_logit = max(m_logit, dl); m_value = max(m_value, dv); mid_greedy += dg
        else:
            d_logit = max(d_logit, dl); d_value = max(d_value, dv); greedy_diff += dg
    if not mid.any():
        d_state = max(d_state, float(ds))
    total += n
    counter_rows += int((obs["cards_"][:, :, 10] != 0).sum())
    rstate = tuple(np.where(done[:, None], 0, np.asarray(x)) for x in rstate)
    action = np.minimum(action, nopt - 1).astype(np.int32)
    obs, _, term, trunc, info = envs.step(action)
    done = term | trunc
print(f"obs_version={args.obs_version} decisions={total} card_rows_with_counters={counter_rows} "
      f"max_abs_logit_diff={d_logit:.3g} max_abs_value_diff={d_value:.3g} max_abs_state_diff={d_state:.3g} "
      f"greedy_action_differs={greedy_diff}")
print(f"mid: decisions with a counter feature of 1 to 14: {mid_decisions} ({mid_rows} rows), there "
      f"max_abs_logit_diff={m_logit:.3g} max_abs_value_diff={m_value:.3g} greedy_action_differs={mid_greedy}")
ok = max(d_logit, d_value, d_state) <= args.tol
sys.exit(0 if ok else 1)
