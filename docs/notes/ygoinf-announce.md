# ygoinf: env parity test and card-name declarations

Branch `ygoinf-announce`, based on `new-cards` (`29430d6`). Session of 2026-11-15, CPU only.

## Summary

- A state-level parity test now replays the same game states through the env's encoder and
  ygoinf's (`tools/ygoinf_parity.py`). With the client given the env's own state, ygoinf matches
  the env in every observation array at every decision: 0 differences in 36,737 decisions over
  three runs, including 144 card-name declarations.
- Before the fixes, about 84% of the compared decisions differed and every game desynced.
  Twelve causes were found and fixed in ygoinf. One was a real-play bug: the finish option of
  multi-card selections was missing from ygoinf's response.
- ygoinf supports `MSG_ANNOUNCE_CARD` with the env's candidate list (commit `1ada8e8`), in the same
  order and with the same features. The request schema has a new `announce_card` message, and
  the existing messages have optional fields. Old clients keep working.
- A real client can't send one input the way the env has it: the order of its own deck. With
  that order the remaining gap is one decision in 14,691 (C9). Without it, almost every decision
  differs in the own-deck card rows. This needs a decision, see "What a client can't match".

## Parity test

### How it works

`tests/ygoinf_parity_env.cpp` builds the unchanged `ygoenv/ygopro/ygopro.h` as a second Python
module, `ygopro_parity` (`xmake b ygopro_parity`, copied to `tests/`). The preprocessor
interposes two ygopro-core calls:

- `get_message`: record the message buffer, then query every location of both players with the
  env's query flags (the field state at the end of the buffer).
- `start_duel`: record a duel boundary.

The env's namespace is renamed in this build so that both modules load in one process (pybind11
registers types by C++ type).

`tools/ygoinf_parity.py` then:

1. Runs the recording env and the real `ygopro_ygoenv` side by side with the same seed and the
   same random actions, and counts the steps where their observations differ. This proves that
   the recording doesn't change the games.
2. Walks the recorded message stream with a reference client. The message lengths are a port of
   `tests/yrp_verify.cpp`. The client tracks what the env tracks from messages (LP, turn, phase,
   turn player, revealed specs, seen codes, confirmed cards, the discard hint) and builds one
   ygoinf request per decision. That includes the decisions the env answers by itself, so
   ygoinf's action history gets the same entries.
3. Calls `ygoinf.features.predict` with a stub model that captures the encoded input, and compares
   `cards_`, `global_`, `actions_` and `h_actions_` with the env's arrays at every decision the env
   shows the model. It counts the decisions with a difference per feature column, and the option
   count separately.

Views (`--view`):

| View | What the client gets | Purpose |
|---|---|---|
| `env` | Everything the env's own state contains: the true order of the own deck, the env's handling of messages after a skipped message (C12, emulated), cards the opponent looked at privately that the env exposes (C9) | Verdict on ygoinf's encoder |
| `client-deck` | Only what the seat receives: every message, only confirmations sent to this seat, but the true deck order | Gaps other than the deck order |
| `client` | Same, and the own deck sorted by code | Everything a real client can't know |

### Hidden cards

The env writes an opponent card in the deck, hand or extra deck as a count-only row (location
and owner, position 0) unless its `revealed_` set is non-empty (C9). The reference client sends
such a card with code 0, empty fields and position `none`. ygoinf then writes the same row and
gives it no spec, so no action can point at it, as in the env. When `revealed_` is non-empty, the
env writes these cards with position face-down, and the client sends them that way. An opponent
face-down card on the field or banished face-down is sent with code 0, empty fields, its zone,
sequence and real position.

Places are sent in reverse order on purpose, to check that ygoinf sorts them.

### Results

Seed 0 unless noted. `decks/mine` and `decks/train` with random decks per game. "Artmage mirror"
is `Artmage_DarkMagician` against itself (all three declaration cards), seed 3.

| Run | Games | Env steps | Recording vs real env | Decisions compared | With a difference |
|---|---|---|---|---|---|
| decks/mine, env view | 40 | 14,730 | 0 steps differ | 14,691 | 0 |
| decks/train, env view | 40 | 12,305 | 0 (checked on 12 games, seed 2) | 12,266 | 0 |
| Artmage mirror, env view | 20 | 9,799 | 0 steps differ | 9,780 | 0 |
| decks/mine, client-deck view | 40 | 14,730 | | 14,691 | 1 |
| decks/mine, client view | 40 | 14,730 | | 14,691 | 14,570 |

Card-name declarations compared, grouped by the card that started the chain. All were equal in
the env view:

| Run | The Black Goat Laughs | Crossout Designator | Artmage Academic Arcane Arts Acropolis |
|---|---|---|---|
| decks/mine | 0 | 7 | 20 |
| decks/train | 2 | 1 | 5 |
| Artmage mirror | 31 | 17 | 61 |

Per message type (decks/mine, env view, compared / with a difference): idle command 4,288 / 0,
chain 2,653 / 0, place 2,541 / 0, select card 1,929 / 0, battle command 860 / 0, position 798 / 0,
effect yes/no 585 / 0, unselect card 428 / 0, yes/no 290 / 0, tribute 142 / 0, option 92 / 0,
announce attribute 45 / 0, announce card 27 / 0, select sum 13 / 0. Announce number appears in the
training pool (8 / 0).

Before the fixes (ygoinf as on `new-cards`, env view, 40 games). Comparison stops at the first
error of a game until the next game:

| | decks/mine | decks/train |
|---|---|---|
| Decisions compared | 1,333 | 1,308 |
| With a difference | 1,115 | 1,092 |
| Games stopped by an error | 40 of 40 | 40 of 40 |
| Stopped on the End Phase option the env doesn't offer | 37 | 39 |
| Stopped on `announce_card` (unsupported) | 3 | 1 |

Features with a difference before (decks/mine, decisions): `h_actions_.place` 986,
`h_actions_.id` 436, `actions_.place` 312, `info.num_options` 193, `actions_.msg` and
`actions_.phase` 191, `actions_.id` 58, `h_actions_.msg` 12, `cards_.position` 8,
`h_actions_.finish` 5.

After the fixes, without card-name declarations (intermediate commit): 0 of 11,587 (decks/mine) and
0 of 11,586 (decks/train) decisions differ. 10 and 2 games stop at `announce_card`.

Client view, decks/mine, features with a difference (decisions of 14,691): every card feature of
the own-deck rows (`cards_` attribute, race, level, atk, def, types: 14,400 to 14,540),
`h_actions_.id` 10,025 and `actions_.id` 672 (cards picked from the deck point at the wrong row).
In the client-deck view only one decision differs: a card the opponent looked at privately (C9)
that the env shows. The reference client counted 3 such card rows.

## What was fixed in ygoinf

| # | Mismatch | Cause | Fix |
|---|---|---|---|
| 1 | Option count, `actions_.phase`, desync | ygoinf offered End Phase when Battle Phase or Main Phase 2 was possible; the env drops it | Skip it; the response lists it with probability -1 |
| 2 | `actions_.place` | Places in the client's order; the env uses flag order | Sort by place id. `select_disfield` answered -1, now the index |
| 3 | `actions_.id` | `select_position` had no card id | Card id from the message code |
| 4 | `actions_.msg` | `select_sum` actions labeled `select_tribute` (4 instead of 12) | Correct label, candidates by card index like the env |
| 5 | `cards_.position` | Own deck, hand, extra deck cards that aren't face-down kept their position | Write 0 like the env |
| 6 | `actions_.card_index` | Count-only opponent rows got a spec | No spec for them |
| 7 | Missing predictions | `can_finish` not extended for selected cards, `zip()` dropped the finish option | Keep the lists aligned; insert selected cards in index order |
| 8 | `h_actions_.id` | The env records its own single-option answers with `spec_to_card_id`: real id of a face-down card, the xyz monster for a material, 0 for a count-only card | Same rule (`auto_card_id`), using the message code |
| 9 | Desync, `h_actions_` | A card selection that continues after a model pick is shown to the model even with one option (`step()` calls `handle_multi_select` without the single-option check) | Call the model in that case |
| 10 | `h_actions_`, recurrent state | End Phase discards (hint 501) are random in the env, without the model or a history entry | Same, with the new `select_hint` field |
| 11 | `actions_.id` | A confirmed deck card in `select_card` takes its code as card id in the env | New `confirmed` field |
| 12 | Crash | `announce_number` compared an object with an int (`TypeError` on every request) | Compare the number |
| 13 | `announce_card` unsupported | | See next section |

## Card-name declarations

`ygoinf/ygoinf/announce.py` is a port of the env's code from `1ada8e8`:

- `is_declarable`: `is_declarable_env` (ygopro-core 0.0.3 `is_declarable`), with int32 arithmetic
  and C++ division.
- `load_card_data`: the env's card data (rule code split, alias inheritance for alternative
  artworks, extra set codes, `second_code`).
- `read_ydk`, `rank_pool`: the env's deck parser and pool ranking (deck files containing the code,
  descending, then code; names starting with `_` don't count).
- `announce_candidates`: the candidate order (named codes, opponent's seen codes newest first, own
  seen codes, own deck in pool order, the pool, then `code_list.txt`), at most 24 (`MAX_ACTIONS`).

Each candidate becomes an action with message id 16 and the card id of the code. The response is
the code. The history entry is the same as in the env (message id and card id).

The pool comes from the server: `CARDS_DB` (the `cards.cdb` the env loads) and `ANNOUNCE_POOL`
(deck files or folders, `:`-separated). `tools/start_inf_server.sh` defaults them to
`assets/locale/en/cards.cdb` and `~/ygo/decks/train`. In training the playing deck is always one
of the loaded decks. If the client's deck (as a set of codes) isn't one of the pool decks, ygoinf
counts it as one more deck file. That is the env with the pool and that deck loaded.

### Schema change (all backward compatible)

New message `announce_card` in `action_msg.data`:

```json
{
  "msg_type": "announce_card",
  "opcodes": [65681983, 1073742080],
  "seen": {"me": [93983867, 46986414], "opponent": [14558127]},
  "deck_main": [46986414, 46986414, "... 40 or more codes"],
  "deck_extra": [1561110, "..."]
}
```

- `opcodes`: the opcodes of `MSG_ANNOUNCE_CARD`, unchanged (uint32).
- `seen.me`, `seen.opponent`: codes revealed this duel by cards that player controls, oldest
  first, a repeated code moves to the end. The client adds a code when it receives
  `MSG_SUMMONING`, `MSG_SPSUMMONING`, `MSG_FLIPSUMMONING` or `MSG_CHAINING` (player: controller byte
  of the location), `MSG_MOVE` to the graveyard, or face-up to the field, banishment or the extra
  deck (player: new controller), `MSG_POS_CHANGE` to a face-up position, and each card of
  `MSG_CONFIRM_CARDS`, `MSG_CONFIRM_DECKTOP`, `MSG_CONFIRM_EXTRATOP` (player: card controller). Reset
  per duel.
- `deck_main`, `deck_extra`: the player's deck list.

New optional fields on existing messages:

| Message | Field | Meaning |
|---|---|---|
| `select_card` cards | `code` | Card code from the message |
| `select_card` cards | `confirmed` | A deck or extra deck card whose code was confirmed for that controller and location since the last `MSG_CHAIN_SOLVED` |
| `select_card` | `select_hint` | Last `HINT_SELECTMSG` value received before the message, 0 if none |
| `select_tribute`, `select_sum`, `select_unselect_card` cards | `code` | Card code from the message |

Without the new fields ygoinf behaves as before for those inputs, which means the history of rare
single-option decisions and confirmed cards can differ from training.

### Client contract

What the client must send so that ygoinf builds what training saw (the reference client in
`tools/ygoinf_parity.py` is the executable version):

- Cards in env order: own cards then the opponent's; per player deck, hand, monster zones, spell
  zones, graveyard, banished, extra deck; per location in core sequence order; xyz materials right
  before their monster, with `overlay_sequence`.
- Own cards: all fields. `attribute`, `race`, `types` from the card database (printed values).
  `level`, `attack`, `defense` current values from the query (level: rank or link rating if
  non-zero; a link monster's defense is its link marker). Materials: printed values, link
  defense 0.
- Opponent cards in deck, hand, extra deck: code 0, empty fields, position `none`, unless a
  `MSG_CONFIRM_CARDS` arrived since the last `MSG_CHAIN_SOLVED` (the env's `revealed_`, which the
  env also keeps across duels). Then position face-down for hidden cards and all fields for
  face-up ones.
- Opponent face-down cards elsewhere: code 0, empty fields, real zone, sequence and position.
- `counter`: any non-zero value (encoded as 15). `negated`: disabled or forbidden status.
- `global`: LP, turn count and phase as tracked from `MSG_NEW_TURN`, `MSG_NEW_PHASE` and the LP
  messages, `is_first` = player 0.
- One request per decision, including single-option ones and each step of a multi-card selection.
- `select_place`/`select_disfield`: any order.

## What a client can't match

These stop at a description, as the rules require. None of them is approximated in ygoinf.

1. **Own deck order (every decision).** The env queries the own deck with codes and writes one
   row per card in core sequence order, so the model sees the deck order (which card is on top).
   A client doesn't know its shuffled deck order. Measured: 14,570 of 14,691 decisions differ in
   the client view, 1 in the client-deck view. A client can send the remaining deck in some
   order, but every choice differs from training. Options: accept it, or change the env to hide
   the own deck order and fine-tune (an observation change, needs your decision).
2. **Cards the opponent looked at privately (C9).** The env adds every `MSG_CONFIRM_CARDS` to
   `revealed_` and its seen codes, whoever sees the cards. The ygopro server sends a deck
   confirmation only to the viewer. 1 decision in 14,691 (client-deck view), 3 card rows.
3. **Messages after a skipped message (C12).** The env drops them; a client receives them. In the
   runs above this changed nothing a client sends (no difference in LP, turn, phase, seen codes,
   or revealed specs in the client-deck view, and no decision ever followed a dropped message in
   the same buffer). It can still matter in rare states.
4. **Single-option selection of a material of a face-down xyz monster.** The env records the
   monster's real code; the client knows only the material's code. Not seen in the runs.
5. **Own cards outside the pool.** The env offers them last in shuffled deck order. ygoinf counts
   the client's deck as a pool deck, so this case doesn't arise.

Still unsupported in ygoinf, as in the env (the process or request fails): declaring a race,
rock-paper-scissors, more than one attribute or number, numbers above 12, select card with
minimum 0, tributes with different minimum and maximum or weights, sums with overflow or more than
two must-select cards.

## Neos: steps for one game

Neos's agent bridge (`~/ygo/src/neos-ts/src/api/ygoAgent/transaction.ts`) has no card-name
declaration: it throws `Unsupported announce type` and the human declares. It also doesn't send
the contract above (it uses printed attack, defense and level, its own card order, and adds the
remaining own deck from the deck list). So this game checks the fixed ygoinf through the HTTP
layer and the parts of the game Neos supports. The model's declaration is checked by the parity
test, not here. Neos isn't changed on this branch.

1. In WSL, check that nothing else serves port 3000 and that no eval needs the CPU:
   `curl -s http://127.0.0.1:3000/ ; free -h`. If an old ygoinf answers, stop it:
   `tmux kill-session -t ygoinf`.
2. Start everything from this worktree, so that its ygoinf runs:
   `cd ~/ygo/wt-ygoinf-announce && tools/start_local_play.sh`
   This overwrites `~/ygo/ygoinf.log` and the logs in `~/ygo/local_server`.
3. Check the log: `grep -E "announce_card|loaded checkpoint" ~/ygo/ygoinf.log`. Expect
   `announce_card: 52 pool decks, ... pool codes` and the checkpoint
   `ft_r4_100M_main15017.flax_model`.
4. Open `http://127.0.0.1:5173/` and join the local room the way you did in earlier local games
   (the proxy is `ws://127.0.0.1:7912`, the server `127.0.0.1:7911`). Use the deck
   `Artmage_DarkMagician` from `~/ygo/decks/mine`: it has The Black Goat Laughs, Crossout Designator
   and Acropolis. `DarkMagician_B` has Crossout Designator.
5. Enable Neos's ygo-agent for your side, as in earlier local games, and play one game against
   a second client (for example a second browser window or MDPro4 on `Local_Offline`).
6. When you activate Crossout Designator or The Black Goat Laughs, Neos asks you for the card name
   (the agent can't). Declare by hand. Note the turn.
7. After the game: `grep -nE "error|Traceback|predict time" ~/ygo/ygoinf.log | tail -50`. Expect no
   `Traceback`. Report any `error` line with the message type.
8. Stop the services with `tools/stop_local_play.sh`.

What to report: the deck, whether the agent made every other decision without errors, and the
error lines from step 7.

## Checks run

- `tools/test_ygoinf_encoding.py` (via `run_checks.sh`): 20 of 20 checks pass, among them the
  parity on decks/mine (10 games, 3,254 decisions, recording build equal on 3,263 steps) and on
  the Artmage mirror (20 games, 9,780 decisions, 31, 17 and 61 declarations by the three cards,
  all equal).
- `tools/run_checks.sh quick`: 9 passed, 0 failed (build now includes `ygopro_parity`).

## Determinism digests

`tools/dump_trajectory.py DECK --games 32`, unchanged env, this worktree's build:

- decks/mine: `2b0c259a5526dc607556d957de7fc90ddc11443f1549e0f2564e2f7751e24adb` (12,232 steps)
- decks/train: `85a2ab9c25429f416299d73578b0484b314136017f36cbb686a956bc4688f0b9` (11,088 steps)

Both equal `new-cards`. `ygoenv/` and the training code are unchanged.

## Not tested

- A real client. Neos isn't changed and wasn't run (see the steps above). No client sends the new
  fields yet.
- The model's outputs. The test compares the model's inputs; it uses a stub model.
- The HTTP transport. `httpx` isn't installed, so instead the server's endpoint functions were
  called inside its lifespan with the promoted checkpoint: the settings loaded 52 pool decks
  (695 codes), an `announce_card` request returned 24 candidates (opponent's seen code first), and
  an old-style `select_yesno` request without new fields still worked.
- Deck folders other than `decks/mine` and `decks/train`, and `play_mode` other than self-play.
- Races, rock-paper-scissors and the other unsupported decisions listed above (not in these decks).
- Two parity runs started at the same moment: one failed with `database is locked` while opening
  `cards.cdb`. Sequential runs and runs started a little apart worked.
