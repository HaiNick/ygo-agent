"""Check that ygoinf encodes observations the way the training env does.

Static and unit checks (no game state is replayed through both encoders yet):
  1. system-string effect ids equal the env's (sorted keys of system_strings in ygopro.h, from 16)
  2. card ids from code_list.txt equal the env's (line number; last line = number of lines)
  3. card rows: id columns 0, counters 0 or 15, level clamped to 13
  4. cancel actions are not added to the action history
  5. live: every system-string effect id the env emits during random play is one ygoinf can produce

usage: python tools/test_ygoinf_encoding.py [--deck ~/ygo/decks/mine] [--steps 300]
exit code 0 = all checks pass.
"""
import argparse, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ygoinf"))
p = argparse.ArgumentParser()
p.add_argument("--deck", default="~/ygo/decks/mine")
p.add_argument("--steps", type=int, default=300)
args = p.parse_args()
deck_path = os.path.abspath(os.path.expanduser(args.deck))
os.chdir(ROOT / "scripts")

import numpy as np
from ygoinf import features as F
from ygoinf.models import Card, Location, Controller, Position, Attribute, Race, Type, Phase

fails = []
def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        fails.append(name)

# 1. system strings
src = (ROOT / "ygoenv/ygoenv/ygopro/ygopro.h").read_text()
block = src[src.index("system_strings = {"):src.index("static std::string get_system_string")]
keys = sorted(set(int(k) for k in re.findall(r"\{(\d+),\s*\"", block)))
env_ids = {k: i + 16 for i, k in enumerate(keys)}
check("system-string ids match the env", F.system_string_to_id == env_ids,
      f"{len(env_ids)} strings, 221 -> {F.system_string_to_id.get(221)}, env {env_ids[221]}")

# 2. code list
F.code_to_id.clear()
F.init_code_list("code_list.txt")
lines = open("code_list.txt").read().splitlines()
last = int(lines[-1].split()[0])
check("card ids follow code_list.txt line numbers", F.code_to_id[last] == len(lines) and F.code_to_id[int(lines[0].split()[0])] == 1,
      f"{len(lines)} lines, last code -> {F.code_to_id[last]}")

# 3. card rows
def card(**kw):
    base = dict(code=last, location=Location.mzone, sequence=0, controller=Controller.me, position=Position.faceup_attack,
                overlay_sequence=-1, attribute=Attribute.dark, race=Race.spellcaster, level=4, counter=0, negated=False,
                attack=2000, defense=1700, types=[Type.monster, Type.effect])
    base.update(kw)
    return Card(**base)
x = F.encode_card(card())
check("card id columns are 0", x[0] == 0 and x[1] == 0)
check("counter is 0 or 15", F.encode_card(card(counter=0))[10] == 0 and F.encode_card(card(counter=1))[10] == 15
      and F.encode_card(card(counter=3))[10] == 15)
check("level is clamped to 13", F.encode_card(card(level=15))[9] == 13)

# 4. history
st = F.PredictState()
row = np.zeros(F.N_ACTION_FEATURES, dtype=np.uint8); row[3] = 2
cancel = row.copy(); cancel[4] = F.action_act_to_id[F.ActionAct.cancel]
act = row.copy(); act[4] = F.action_act_to_id[F.ActionAct.activate]
class Msg:  # minimal stand-in for ActionMsg in transform_select_idx
    class data: msg_type = "select_chain"
for chosen, expect in ((1, 0), (0, 1)):
    st._probs, st._actions, st._action_msg, st._turn, st._phase = [0.5, 0.5], np.stack([act, cancel]), Msg, 1, Phase.main1
    st.update_history_actions(chosen)
    n = int((st.history_actions.h_actions[:, 3] != 0).sum())
    check(f"history after choosing {'cancel' if chosen == 1 else 'activate'} has {expect} entries", n == expect, f"got {n}")

# 5. live effect ids from the env
import ygoenv
from ygoai.utils import init_ygopro
deck = init_ygopro("YGOPro-v1", "english", deck_path, "code_list.txt")
envs = ygoenv.make(task_id="YGOPro-v1", env_type="gymnasium", num_envs=32, num_threads=8, seed=5, deck1=deck, deck2=deck,
                   max_options=24, n_history_actions=32, play_mode="self", async_reset=False)
rng = np.random.default_rng(0)
obs, info = envs.reset()
seen = set(); max_level = 0; id_cols = 0
for _ in range(args.steps):
    a = (rng.integers(0, 1 << 30, size=32) % np.maximum(info["num_options"], 1)).astype(np.int32)
    obs, r, term, trunc, info = envs.step(a)
    seen |= set(np.unique(obs["actions_"][:, :, 6]).tolist())
    max_level = max(max_level, int(obs["cards_"][:, :, 9].max())); id_cols += int(obs["cards_"][:, :, :2].sum())
sys_seen = {e for e in seen if e >= 16}
check("every system-string effect id from the env exists in ygoinf", sys_seen <= set(F.system_string_to_id.values()),
      f"env emitted {sorted(sys_seen)}")
check("env card id columns are 0 and level <= 13 (what ygoinf mirrors)", id_cols == 0 and max_level <= 13)
print("FAILED: " + ", ".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
