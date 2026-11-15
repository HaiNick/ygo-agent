"""Check that ygoinf encodes observations the way the training env does.

Static and unit checks:
  1. system-string effect ids equal the env's (sorted keys of system_strings in ygopro.h, from 16)
  2. card ids from code_list.txt equal the env's (line number; last line = number of lines)
  3. card rows: id columns 0, counters 0 or 15, level clamped to 13
  4. cancel actions are not added to the action history
  5. live: every system-string effect id the env emits during random play is one ygoinf can produce
  6. card-name declarations: candidate order and is_declarable on fixed cases
State-level parity (tools/ygoinf_parity.py, each in its own process; needs `xmake b ygopro_parity`):
  7. decks/mine, random decks: every observation array ygoinf builds equals the env's, and the
     recording build gives the same observations as ygopro_ygoenv
  8. Artmage_DarkMagician mirror: the same, and at least one card-name declaration compared for each
     of The Black Goat Laughs, Crossout Designator and Artmage Academic Arcane Arts Acropolis

usage: python tools/test_ygoinf_encoding.py [--deck ~/ygo/decks/mine] [--steps 300]
         [--parity-games 10] [--announce-games 20]
exit code 0 = all checks pass.
"""
import argparse, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ygoinf"))
p = argparse.ArgumentParser()
p.add_argument("--deck", default="~/ygo/decks/mine")
p.add_argument("--steps", type=int, default=300)
p.add_argument("--parity-games", type=int, default=10)
p.add_argument("--announce-games", type=int, default=20)
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
# 6. card-name declarations
from ygoinf import announce as A
db = str(ROOT / "assets/locale/en/cards.cdb")
ctx = F.init_announce(db, [deck_path])
cd = ctx.card_data
BLACK_GOAT, CROSSOUT, ACROPOLIS = 49299410, 65681983, 74733322
# "is a monster card": every candidate is declarable, tokens and rule-code cards never are
ops_monster = [0x1, A.OPCODE_ISTYPE]
cands = A.announce_candidates(ctx, ops_monster, [], [], [], [], 24)
check("declaration candidates are declarable, at most 24", 0 < len(cands) <= 24 and
      all(A.is_declarable(cd[c], ops_monster) for c in cands), f"{len(cands)} candidates")
tokens = [c for c, d in cd.items() if d.type & A.TYPE_TOKEN][:50]
check("tokens are not declarable", tokens and not any(A.is_declarable(cd[c], ops_monster) for c in tokens))
# order: named code, opponent's seen (newest first), own seen, own deck by pool rank, pool
mine = {n: A.read_ydk(f) for n, f in ((f.stem, f) for f in Path(deck_path).glob("*.ydk"))}
own_main, own_extra = mine["Artmage_DarkMagician"]
named = [CROSSOUT, A.OPCODE_ISCODE, 0x1, A.OPCODE_ISTYPE, A.OPCODE_OR]  # this code, or any monster
seen_op, seen_me = [own_main[0], BLACK_GOAT], [own_main[1]]
c2 = A.announce_candidates(ctx, named, seen_op, seen_me, own_main, own_extra, 24)
decl = lambda c: A.is_declarable(cd[c], named)
expect = list(dict.fromkeys([CROSSOUT] + [c for c in [BLACK_GOAT, own_main[0], own_main[1]] if decl(c)]))
check("declaration order: named, opponent seen newest first, own seen", c2[:len(expect)] == expect,
      f"got {c2[:5]}")
rank = ctx.ranked_for(own_main, own_extra)
own_order = [c for c in rank if c in set(own_main) | set(own_extra) and decl(c) and c not in c2[:len(expect)]]
check("declaration order: then own deck in pool order", c2[len(expect):len(expect) + len(own_order)] == own_order[:24 - len(expect)])
check("pool rank: deck count desc, then code", rank == sorted(rank, key=lambda c: (-sum(c in set(m) | set(e) for m, e in mine.values()), c)))

# 7, 8. state-level parity in separate processes
import json, subprocess, tempfile
def parity(name, extra):
    out = Path(tempfile.mkdtemp()) / "parity.json"
    cmd = [sys.executable, str(ROOT / "tools/ygoinf_parity.py"), "--deck", deck_path, "--json", str(out)] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if not out.exists():
        check(name, False, (r.stdout + r.stderr).strip().splitlines()[-1:] or "no output")
        return None
    return json.loads(out.read_text())
r = parity("parity decks/mine", ["--games", str(args.parity_games), "--seed", "0"])
if r is not None:
    check("parity decks/mine: recording build equals ygopro_ygoenv", r["harness_diff"] == 0, f"{r['steps']} steps")
    check("parity decks/mine: all observation arrays equal", r["compared"] > 0 and not r["col_diff"] and not r["errors"],
          f"{r['compared']} decisions, differences {r['col_diff'] or 'none'}, errors {r['errors'] or 'none'}")
r = parity("parity Artmage_DarkMagician", ["--games", str(args.announce_games), "--seed", "3",
                                           "--deck1", "Artmage_DarkMagician", "--deck2", "Artmage_DarkMagician"])
if r is not None:
    check("parity Artmage mirror: all observation arrays equal", r["harness_diff"] == 0 and not r["col_diff"] and not r["errors"],
          f"{r['compared']} decisions, differences {r['col_diff'] or 'none'}, errors {r['errors'] or 'none'}")
    got = {int(k): (n, r["announce_ok"].get(k, 0)) for k, n in r["announce"].items()}
    for code, nm in ((BLACK_GOAT, "The Black Goat Laughs"), (CROSSOUT, "Crossout Designator"),
                     (ACROPOLIS, "Artmage Academic Arcane Arts Acropolis")):
        n, ok = got.get(code, (0, 0))
        check(f"declaration by {nm} compared and equal", n > 0 and ok == n, f"{ok} of {n}")

print("FAILED: " + ", ".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
