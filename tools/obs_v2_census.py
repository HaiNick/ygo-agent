"""Count how often the open observation issues occur (docs/OBS_V2.md, issues C to G).

Plays games in the recording build of the unchanged env (tests/ygopro_parity.so, see
tools/ygoinf_parity.py) with one env, reads the core's message stream and the field snapshots,
and counts per decision, per game and per deck:

  C9   opponent deck/hand/extra rows whose features the env exposes although this seat never
       got a confirmation for them ("leak"), and cards this seat was shown that the env hides
       ("miss"). Windows are the env's: a confirmation counts until the next MSG_CHAIN_SOLVED.
  C10  decisions whose message has more than 24 options (idle, battle, select card, unselect
       card, tribute, chain), and idle/battle commands that lose their phase change that way.
  C11  decisions with a card that has counters (feature 15), with the real counts.
  C12  messages dropped because they follow a skipped message in the same buffer, and decisions
       where that changes LP, turn, turn player, phase or the seen codes the env tracks.
  G    decisions the env answers without the model: End Phase discards (random), sort_card
       (default order), select_counter (first card first), and phase changes the env doesn't
       offer (End Phase when Battle Phase or Main Phase 2 is possible).

A decision is attributed to the deck of the player who decides. Decks are drawn at random from
the folder for both seats.

usage: python tools/obs_v2_census.py --deck DIR [--games 40] [--seed 0] [--policy model|random]
         [--checkpoint CKPT] [--num-embeddings 15017] [--json FILE]
policy model: greedy actions of the checkpoint (batch 1, CPU is fine). policy random: the
action stream of tools/ygoinf_parity.py, so the step count can be checked against its report.
"""
import argparse, json, os, struct, sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ygoinf_parity as yp  # noqa: E402
from ygoinf_parity import M, DECISIONS, ENV_EXACT, walk, u32, DuelState  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--deck", default="~/ygo/decks/mine")
ap.add_argument("--games", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--policy", default="model", choices=["model", "random"])
ap.add_argument("--checkpoint")
ap.add_argument("--num-embeddings", type=int, default=15017)
ap.add_argument("--json")
a = ap.parse_args()
deck_dir = os.path.abspath(os.path.expanduser(a.deck))
ckpt = os.path.abspath(os.path.expanduser(a.checkpoint)) if a.checkpoint else None
json_path = os.path.abspath(os.path.expanduser(a.json)) if a.json else None

import numpy as np  # noqa: E402

P, penv, _, db, decks = yp.make_envs(deck_dir, "random", "random", a.seed, False)  # chdir scripts/
sys.path.insert(0, str(ROOT))
KEYS = ("cards_", "global_", "actions_", "h_actions_")
HIDDEN = (yp.LOCATION_DECK, yp.LOCATION_HAND, yp.LOCATION_EXTRA)
LOCID = {1: "deck", 2: "hand", 7: "extra"}


def read_main(path):
    out, on = [], False
    for line in open(path):
        s = line.strip()
        if s.startswith("#main"): on = True; continue
        if s.startswith("#extra") or s.startswith("!side"): on = False; continue
        if on and s.isdigit(): out.append(int(s))
    return tuple(sorted(out))


deck_by_main = {read_main(pth): name for name, pth in decks.items()}

if a.policy == "model":
    import jax, jax.numpy as jnp, flax
    from ygoai.rl.jax.agent import RNNAgent, ModelArgs
    agent = RNNAgent(**asdict(ModelArgs()), embedding_shape=a.num_embeddings)

    @jax.jit
    def fwd(params, rs, obs):
        nxt, logits = agent.apply(params, obs, rs)[:2]
        return nxt, logits


def raw_options(m, b):
    """(option count before the cut to 24, has a phase change) for the messages that list
    options directly. None for the others."""
    if m == M["SELECT_IDLECMD"]:
        o, n = 1, 0
        for _ in range(5):
            n += b[o]; o += 1 + b[o] * 7
        n += b[o]; o += 1 + b[o] * 11
        to_bp, to_ep = b[o], b[o + 1]
        phases = int(bool(to_bp)) + int(bool(to_ep) and not to_bp)
        return n + phases, phases > 0
    if m == M["SELECT_BATTLECMD"]:
        n1 = b[1]; o = 2 + n1 * 11; n2 = b[o]; o += 1 + n2 * 8
        to_m2, to_ep = b[o], b[o + 1]
        phases = int(bool(to_m2)) + int(bool(to_ep) and not to_m2)
        return n1 + n2 + phases, phases > 0
    if m in (M["SELECT_CARD"], M["SELECT_TRIBUTE"]):
        return b[4], False
    if m == M["SELECT_UNSELECT_CARD"]:
        return b[5] + int(bool(b[1])), False
    if m == M["SELECT_CHAIN"]:
        n = b[1]
        forced = any(b[11 + 14 * i + 1] for i in range(n))
        return n + (0 if forced else 1), False
    return None


tot = Counter()                       # run totals
by_deck = defaultdict(Counter)        # deck -> counters (decisions of that deck's player)
game_flags = set()                    # (flag, player) seen in the current game
games_with = Counter()                # flag -> games
deck_games_with = defaultdict(Counter)
deck_games = Counter()
dropped_types = Counter()
counter_hist = Counter()
by_msg = Counter()
trunc_by_msg = Counter()


def flag(name, player, n=1, decision=True):
    tot[name] += n
    by_deck[cur_decks[player]][name] += n
    game_flags.add((name, player))


def end_game():
    names = {n for n, _ in game_flags}
    for n in names:
        games_with[n] += 1
    for n, pl in game_flags:
        deck_games_with[cur_decks[pl]][n] += 1
    for pl in range(2):
        deck_games[cur_decks[pl]] += 1
    game_flags.clear()


env_st = DuelState()                  # the env's tracking, with its message drops
full_st = DuelState()                 # the same without drops
seat_st = [DuelState(), DuelState()]  # what each seat receives (confirmations of deck cards only to the viewer)
snapshot = None
cur_decks = ["?", "?"]
need_decks = True
pending = None                        # last decision message since the previous step: (m, payload)


def seat_receives(seat, m, b):
    if m == M["CONFIRM_CARDS"] and b[2] > 0 and b[3 + 5] == yp.LOCATION_DECK:
        return b[0] == seat
    return True


def feed(records):
    global snapshot, need_decks, pending
    for kind, data in records:
        if kind == 0:
            for s in (env_st, full_st, *seat_st):
                s.new_duel()
            need_decks = True
        elif kind == 2:
            snapshot = yp.parse_snapshot(data)
            if need_decks:
                need_decks = False
                for p in range(2):
                    main = tuple(sorted(c.code for loc in (yp.LOCATION_DECK, yp.LOCATION_HAND)
                                        for c in snapshot[(p, loc)]))
                    cur_decks[p] = deck_by_main.get(main, "?")
        else:
            dropped = False
            for m, b in walk(data):
                if m not in DECISIONS:
                    if not dropped:
                        env_st.apply(m, b, lambda mm, bb: True)
                    else:
                        tot["dropped_messages"] += 1
                        dropped_types[yp.MSG_NAME.get(m, str(m))] += 1
                    full_st.apply(m, b, lambda mm, bb: True)
                    for seat in range(2):
                        seat_st[seat].apply(m, b, lambda mm, bb, seat=seat: seat_receives(seat, mm, bb))
                else:
                    pl = b[1] if m == M["SELECT_SUM"] else b[0]
                    if dropped:
                        tot["decision_after_dropped_message"] += 1
                    # decisions the env answers by itself
                    if m == M["SORT_CARD"]:
                        flag("G_sort_card", pl)
                        if b[1] > 1: flag("G_sort_card_2plus", pl)
                    elif m == M["SELECT_COUNTER"]:
                        flag("G_select_counter", pl)
                        if b[5] > 1: flag("G_select_counter_2cards", pl)
                    elif m == M["SELECT_CARD"]:
                        if env_st.discard_hand and env_st.phase == yp.PHASE_END:
                            flag("G_end_phase_discard", pl)
                            if b[4] > b[2]: flag("G_end_phase_discard_choice", pl)
                        for s in (env_st, full_st, *seat_st):
                            s.discard_hand = False
                    elif m == M["SELECT_IDLECMD"]:
                        o = 1
                        for _ in range(5):
                            o += 1 + b[o] * 7
                        o += 1 + b[o] * 11
                        if b[o] and b[o + 1]: flag("G_idle_end_phase_withheld", pl)
                    elif m == M["SELECT_BATTLECMD"]:
                        o = 2 + b[1] * 11; o += 1 + b[o] * 8
                        if b[o] and b[o + 1]: flag("G_battle_end_phase_withheld", pl)
                    pending = (m, b)
                if m not in ENV_EXACT and m not in DECISIONS:
                    dropped = True


def census(obs, info):
    """Everything counted at a decision the model sees."""
    global pending
    p = int(info["to_play"][0])
    cards = obs["cards_"][0]
    tot["decisions"] += 1
    by_deck[cur_decks[p]]["decisions"] += 1
    # ---- C10
    if pending is not None:
        m, b = pending
        by_msg[yp.MSG_NAME[m]] += 1
        ro = raw_options(m, b)
        if ro is not None and ro[0] > 24:
            flag("C10_truncated", p)
            trunc_by_msg[yp.MSG_NAME[m]] += 1
            tot["C10_options_lost"] += ro[0] - 24
            if ro[1]:
                flag("C10_phase_change_cut", p)
    else:
        by_msg["(next step of a multi-card selection)"] += 1
        if int(info["num_options"][0]) >= 24:
            flag("C10_at_cap_multi_select_step", p)
    if pending is not None and pending[0] == M["ANNOUNCE_CARD"]:
        flag("announce_card_decisions", p)
        if env_st.seen != full_st.seen:
            flag("C12_seen_codes_differ_at_announce_card", p)
    pending = None
    # ---- C11
    crow = cards[:, 10] != 0
    if crow.any():
        flag("C11_counter_card_visible", p)
        real = []
        for (pl, loc), cs in snapshot.items():
            for c in cs:
                for w in c.counters:
                    real.append(w >> 16)
                if len(c.counters) > 1:
                    flag("C11_two_counter_types_on_a_card", p)
        for r in real:
            counter_hist[min(r, 20)] += 1
        if any(r != 1 for r in real):
            flag("C11_count_not_1", p)
        if len(set(real)) > 1:
            flag("C11_different_counts_visible", p)
    # ---- C9 leak: opponent hidden-zone rows with features
    opp = cards[:, 4] == 1
    hid = opp & np.isin(cards[:, 2], (1, 2, 7))
    if (hid & (cards[:, 5] != 0)).any():
        flag("C9_revealed_set_nonempty", p)
    exposed = hid & (cards[:, 7:] != 0).any(axis=1)
    if exposed.any():
        o = 1 - p
        known = seat_st[p].known
        leak = 0
        for locid, loc in ((1, yp.LOCATION_DECK), (2, yp.LOCATION_HAND), (7, yp.LOCATION_EXTRA)):
            rows = np.nonzero(opp & (cards[:, 2] == locid))[0]
            cs = snapshot[(o, loc)]
            for j, r in enumerate(rows):
                if not exposed[r] or j >= len(cs):
                    continue
                c = cs[j]
                public = loc == yp.LOCATION_EXTRA and (c.pos & yp.POS_FACEUP)
                if public or known.get((c.con, c.loc, c.seq)) == c.code:
                    tot[f"C9_exposed_ok_rows_{LOCID[locid]}"] += 1
                else:
                    leak += 1
                    tot[f"C9_leak_rows_{LOCID[locid]}"] += 1
        if leak:
            flag("C9_leak", p)
    # ---- C9 miss: cards this seat was shown that the env hides
    o = 1 - p
    miss = 0
    for (con, loc, seq), code in seat_st[p].known.items():
        if con != o:
            continue
        for c in snapshot.get((o, loc), ()):
            if c.seq == seq and c.code == code:
                hidden_zone = loc in HIDDEN
                facedown = bool(c.pos & yp.POS_FACEDOWN)
                if not (hidden_zone or facedown):
                    break
                if hidden_zone and loc == yp.LOCATION_EXTRA and not facedown and env_st.revealed:
                    break  # face-up extra deck card, shown while revealed_ is non-empty
                spec = yp.ls_to_spec(loc, seq, 0, True)
                if hidden_zone and not env_st.revealed:
                    miss += 1          # count-only row
                elif facedown and spec not in env_st.revealed:
                    miss += 1          # face-down row without features
                elif hidden_zone and not facedown and loc != yp.LOCATION_EXTRA and spec not in env_st.revealed:
                    pass               # shown anyway because the whole zone is unhidden
                break
    if miss:
        flag("C9_miss", p)
        tot["C9_miss_cards"] += miss
    # ---- C12
    if (env_st.lp != full_st.lp or env_st.turn != full_st.turn or env_st.tp != full_st.tp
            or env_st.phase != full_st.phase):
        flag("C12_global_differs", p)
        if env_st.phase != full_st.phase: tot["C12_phase_differs"] += 1
        if env_st.lp != full_st.lp: tot["C12_lp_differs"] += 1
        if env_st.turn != full_st.turn or env_st.tp != full_st.tp: tot["C12_turn_differs"] += 1
    if env_st.seen != full_st.seen:
        flag("C12_seen_codes_differ", p)
    if {tuple(sorted(v)) for v in env_st.seen} != {tuple(sorted(v)) for v in full_st.seen}:
        tot["C12_seen_sets_differ"] += 1


rng = np.random.default_rng(a.seed)
obs, info = penv.reset()
if a.policy == "model":
    obs1 = {k: jnp.asarray(obs[k]) for k in KEYS}
    params = jax.jit(agent.init)(jax.random.PRNGKey(0), obs1, agent.init_rnn_state(1))
    with open(ckpt, "rb") as f:
        params = flax.serialization.from_bytes(params, f.read())
    rstates = [agent.init_rnn_state(1), agent.init_rnn_state(1)]
done_games = steps = 0
after_terminal = False
while done_games < a.games:
    feed(P.drain())
    nopt = int(info["num_options"][0])
    if not after_terminal:
        census(obs, info)
    if a.policy == "random":
        act = int(rng.integers(0, 1 << 30) % max(nopt, 1))
    elif after_terminal:
        act = 0
        rstates = [agent.init_rnn_state(1), agent.init_rnn_state(1)]
    else:
        pl = int(info["to_play"][0])
        rstates[pl], logits = fwd(params, rstates[pl], {k: obs[k] for k in KEYS})
        act = int(np.asarray(logits)[0].argmax())
    obs, rew, term, trunc, info = penv.step(np.array([act], dtype=np.int32))
    steps += 1
    after_terminal = bool(term[0] or trunc[0])
    if after_terminal:
        done_games += 1
        feed(P.drain())
        pending = None
        end_game()
        if done_games % 10 == 0:
            print(f"games {done_games} steps {steps}", file=sys.stderr, flush=True)

out = dict(deck_dir=deck_dir, policy=a.policy, checkpoint=Path(ckpt).name if ckpt else None, seed=a.seed,
           games=done_games, env_steps=steps, totals=dict(sorted(tot.items())),
           games_with=dict(sorted(games_with.items())), by_msg=dict(by_msg), truncated_by_msg=dict(trunc_by_msg),
           dropped_message_types=dict(dropped_types.most_common()),
           real_counter_values=dict(sorted(counter_hist.items())),
           deck_games=dict(sorted(deck_games.items())),
           by_deck={d: dict(sorted(c.items())) for d, c in sorted(by_deck.items())},
           deck_games_with={d: dict(sorted(c.items())) for d, c in sorted(deck_games_with.items())})
print(json.dumps(out["totals"], indent=1))
print(json.dumps(out["games_with"], indent=1))
if json_path:
    with open(json_path, "w") as f:
        json.dump(out, f, indent=1)
