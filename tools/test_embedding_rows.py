"""Check which embedding a card id actually gets through the real model code.

Builds RNNAgent with --num-embeddings, restores the checkpoint (as eval.py and
battle.py do), and for the last code in code_list.txt checks:
  1. restored table rows == num_embeddings + 1 (agent.py adds row 0 for unknown)
  2. the Embed lookup inside the model (Encoder_0/Embed_0) for that id:
     finite, equal to the stored voyage-2 vector, not equal to row 0
  3. a forward pass on a real BlueEyes observation with one card's id set to
     that code: finite logits/value, and the output changes when the row changes

usage: python tools/test_embedding_rows.py CKPT --num-embeddings N [--vectors PKL]
exit code 0 = the last code gets its real vector and the forward pass is finite.
"""
import argparse, os, pickle, sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIG_CWD = os.getcwd()
os.chdir(ROOT / "scripts")

import numpy as np
import jax, jax.numpy as jnp
import flax
import ygoenv
from ygoai.utils import init_ygopro
from ygoai.rl.jax.agent import RNNAgent, ModelArgs

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--num-embeddings", type=int, required=True)
ap.add_argument("--vectors", default=os.path.expanduser("~/ygo/embeddings/voyage2.pkl"))
args = ap.parse_args()
args.ckpt = os.path.abspath(os.path.join(ORIG_CWD, args.ckpt))
args.vectors = os.path.abspath(os.path.join(ORIG_CWD, args.vectors))

codes = [int(l.split()[0]) for l in open("code_list.txt") if l.strip()]
last_code, last_id = codes[-1], len(codes)
vectors = pickle.load(open(args.vectors, "rb"))

deck = init_ygopro("YGOPro-v1", "english", "../assets/deck/BlueEyes.ydk", "code_list.txt")
envs = ygoenv.make(task_id="YGOPro-v1", env_type="gymnasium", num_envs=1, num_threads=1, seed=0,
                   deck1=deck, deck2=deck, player=-1, max_options=24, n_history_actions=32,
                   play_mode="random", async_reset=False)
obs, _ = envs.reset()
obs = {k: v for k, v in obs.items() if k != "mask_"}
obs["mask_"] = None  # as EnvPreprocess(skip_mask=True)

agent = RNNAgent(**asdict(ModelArgs()), embedding_shape=args.num_embeddings)
key = jax.random.PRNGKey(0)
sample = jax.tree.map(lambda x: jnp.array([x]), {k: v[0] for k, v in obs.items() if v is not None})
sample["mask_"] = None
rstate = agent.init_rnn_state(1)
params = jax.jit(agent.init)(key, sample, rstate)
with open(args.ckpt, "rb") as f:
    params = flax.serialization.from_bytes(params, f.read())
table = np.asarray(params["params"]["Encoder_0"]["Embed_0"]["embedding"])

ok = True
def report(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")

print(f"checkpoint {args.ckpt}")
print(f"num_embeddings={args.num_embeddings}  restored table rows={table.shape[0]}  "
      f"last code {last_code} -> id {last_id}")
report("restored rows == num_embeddings + 1 (agent.py adds row 0)", table.shape[0] == args.num_embeddings + 1)
report("table has a row for the last id", table.shape[0] > last_id)

# 2. the model's own Embed module on that id
emb = flax.linen.Embed(num_embeddings=args.num_embeddings + 1, features=table.shape[1])  # agent.py: n_embed = 1 + num_embeddings
out = np.asarray(emb.apply({"params": {"embedding": jnp.asarray(table)}}, jnp.array([last_id])))[0]
report("Embed(id) is finite", np.isfinite(out).all(), f"(NaN count {int(np.isnan(out).sum())})")
if last_code in vectors:
    v = np.asarray(vectors[last_code], np.float32)
    report("Embed(id) == voyage-2 vector", np.allclose(out, v, atol=1e-6),
           f"(max diff {np.nanmax(np.abs(out - v)):.2e})")
report("Embed(id) != row 0 placeholder", not np.allclose(out, table[0]))

# 3. forward pass with one real card slot set to the last id
cards = obs["cards_"].copy()
slot = int(np.nonzero(cards[0, :, 2])[0][0])  # first slot with a location
cards[0, slot, 0], cards[0, slot, 1] = last_id >> 8, last_id & 0xff
o = dict(obs, cards_=cards)
batch = jax.tree.map(lambda x: jnp.asarray(x), {k: v for k, v in o.items() if v is not None})
batch["mask_"] = None
apply = jax.jit(agent.apply)
_, logits, value = apply(params, batch, agent.init_rnn_state(1))[:3]
logits, value = np.asarray(logits), np.asarray(value)
n = int(obs["actions_"][0, :, 3].astype(bool).sum()) or 1
report("forward pass finite", np.isfinite(logits[0, :n]).all() and np.isfinite(value).all(),
       f"(value {value.ravel()[0]:.4f})")
if table.shape[0] > last_id:
    p2 = jax.tree.map(lambda x: x, params)
    t2 = table.copy(); t2[last_id] = table[0]
    p2["params"]["Encoder_0"]["Embed_0"]["embedding"] = jnp.asarray(t2)
    _, logits2, value2 = apply(p2, batch, agent.init_rnn_state(1))[:3]
    report("output depends on that row", not np.allclose(np.asarray(value2), value),
           f"(value with row 0 instead: {np.asarray(value2).ravel()[0]:.4f})")
sys.exit(0 if ok else 1)
