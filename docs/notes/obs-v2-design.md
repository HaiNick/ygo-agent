# Observation v2: measurements

Branch `obs-v2-design`, based on `new-cards` (`56474f1`). Session of 2026-11-15, CPU only
(a training run used the GPU the whole time). The design and the recommendation are in
`docs/OBS_V2.md`. This file has the commands and the raw numbers.

Nothing under `ygoenv/`, `ygoinf/`, `scripts/cleanba.py` or the model code was changed. New
files: `tools/deck_order_probe.py`, `tools/obs_v2_census.py`, and the two documents.

## Setup

```bash
cd ~/ygo/wt-obs-v2
ln -s ../../ygopro-scripts scripts/script                    # as in ~/ygo/ygo-agent, not committed
ln -s /home/user/ygo/ygo-agent/assets/locale assets/locale   # not committed
source ~/ygo/.venv/bin/activate
export CMAKE_POLICY_VERSION_MINIMUM=3.5 PATH="$HOME/.local/bin:$PATH"
xmake f -m release -y && xmake b -y -j4 ygopro_ygoenv && xmake b -y -j4 ygopro_parity
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$PWD/ygoenv:$PWD
CK=~/ygo/ygo-agent/scripts/checkpoints/ft_r4_100M_main15017.flax_model
```

Every measurement process ran with `nice -n 10 taskset -c 4-11` (games) or `taskset -c 0-3`
(census), 8 env threads at most, and reported `CpuDevice(id=0)`. Free memory before each run
was 9.5 to 10.9 GB.

## Determinism digests

`python tools/dump_trajectory.py DECKS --games 32 --clamp-level 13`, this worktree's build:

- decks/mine: `2b0c259a5526dc607556d957de7fc90ddc11443f1549e0f2564e2f7751e24adb` (12,232 steps)
- decks/train: `85a2ab9c25429f416299d73578b0484b314136017f36cbb686a956bc4688f0b9` (11,088 steps)

Both equal `new-cards`.

## Task 1: own deck order (issue A)

### What the env writes and what the client view sends

- `_set_obs_cards` (`ygopro.h:2766`) writes the deciding player's cards first, and the deck
  first among them: one row per deck card from `get_cards_in_location(player, LOCATION_DECK)`,
  in core sequence order. The last deck row is the top card.
- For these rows the columns are: 0, 1 card id = 0 (K1); 2 location = 1 (deck); 3 sequence = 0
  (only written for monster zone, spell zone and graveyard); 4 owner = 0; 5 position = face-down
  id when the card is face-down, else 0; 6 overlay = 0; 7 to 40 attribute, race, level, counter,
  negated, atk, def, types. So two deck rows can differ only in columns 5 and 7 to 40, and the
  deck order is carried only by the row order.
- `actions_` column 0 is 1 + the `cards_` row of the action's card (0 = none). A deck search
  (`select_card` with deck candidates) points at deck rows.
- The `client` view of `tools/ygoinf_parity.py` (`RefClient.cards`, line 509) sorts the own-deck
  rows by card code and renumbers their sequence 0..n-1. Messages still name deck cards by their
  real sequence, so ygoinf resolves an action's card through the renumbered rows and finds a
  different card. That is the source of the `actions_.id` (672) and `h_actions_.id` (10,025)
  differences in the client view, next to the reordered rows themselves (14,400 to 14,540).

### The model can't see the row order

`ygoai/rl/jax/agent.py`: the card rows go through `CardEncoder` (per-row embeddings, no row
index) and then through `get_encoder_layer_cls(...)`, which builds
`LlamaEncoderLayer(..., rope=False)` (line 22). No positional embedding is added anywhere. The
only position-like input is column 3, which is 0 for deck rows. The legal actions use the same
kind of layer in `FiLMActor` (line 423), also without positions. Only the action history uses
`rope=True` (line 320).

So the card encoder is equivariant to a permutation of the rows, and the only consumer of row
numbers is the gather `f_cards[B, spec_index]` for the legal actions (line 340). If the action
index follows the card to its new row, the model's output must be the same up to float rounding.
The measurement confirms it.

### Honest variants

Defined in `tools/deck_order_probe.py` (docstring). Rows: `cards_[:, r, :]` with column 2 == 1
and column 4 == 0, which are the first n rows. All 41 columns of a row move together.

| Name | Deck rows of seat A | `actions_` column 0 |
|---|---|---|
| control | unchanged | unchanged |
| sorted_remap | sorted by their bytes in columns 5 to 40 | follows the card |
| shuffled_remap | new random order at every decision | follows the card |
| sorted_noremap | sorted | unchanged (points at whatever card is now in that row) |
| shuffled_noremap | new random order at every decision | unchanged |
| blank | columns 7 to 40 zeroed (deck content hidden, not a client view) | unchanged |
| ids | not a deck-order variant, see "Issue B" below | unchanged |

`sorted` sorts by feature bytes, not by card code, because the observation has no code (K1).
For the model this is the same thing: a fixed order that doesn't depend on the deck order.

The `noremap` variants are close to what the reference client's `client` view does. They differ
from it in one point: here the action keeps its true card id in columns 1, 2, in the client view
ygoinf also takes the id from the wrong row.

### Commands

```bash
for deck in 00_Base_Branded-Despia Artmage_PowerPatron Blitzclique_A Elfnote_KewlTune; do
  d=~/ygo/decks/mine/$deck.ydk
  python tools/deck_order_probe.py --deck $d --checkpoint $CK --variant none --games 128
  python tools/deck_order_probe.py --deck $d --checkpoint $CK --variant sorted --probe --games 128
  python tools/deck_order_probe.py --deck $d --checkpoint $CK --variant sorted --no-remap --probe --games 128
  python tools/deck_order_probe.py --deck $d --checkpoint $CK --variant shuffled --probe --games 128
  python tools/deck_order_probe.py --deck $d --checkpoint $CK --variant shuffled --no-remap --probe --games 128
  python tools/deck_order_probe.py --deck $d --checkpoint $CK --variant blank --probe --games 128
  python tools/deck_order_probe.py --deck $d --checkpoint $CK --variant ids --probe --games 128
done
```

Promoted model against itself, mirror deck, greedy actions, seed 0, 128 envs and the first game
of each env. Seat A (changed observation) is player 0 (moves first) in 64 games and player 1 in
64. `--probe` also runs every decision of seat A through the model with the unchanged
observation and the same recurrent state. One run took 80 to 430 s.

### Results per deck

At 128 games the 95% interval is about plus or minus 0.085.

| Variant | Deck | Games | A wins | A win rate | 95% interval | A first | A second | Player 0 wins | Mean length |
|---|---|---|---|---|---|---|---|---|---|
| control | 00_Base_Branded-Despia | 128 | 64 | 0.500 | 0.415 to 0.585 | 29/64 | 35/64 | 58 | 242 |
| control | Artmage_PowerPatron | 128 | 57 | 0.445 | 0.362 to 0.532 | 26/64 | 31/64 | 59 | 201 |
| control | Blitzclique_A | 128 | 65 | 0.508 | 0.422 to 0.593 | 23/64 | 42/64 | 45 | 166 |
| control | Elfnote_KewlTune | 128 | 62 | 0.484 | 0.400 to 0.570 | 33/64 | 29/64 | 68 | 166 |
| sorted_remap | 00_Base_Branded-Despia | 128 | 64 | 0.500 | 0.415 to 0.585 | 29/64 | 35/64 | 58 | 242 |
| sorted_remap | Artmage_PowerPatron | 128 | 57 | 0.445 | 0.362 to 0.532 | 26/64 | 31/64 | 59 | 201 |
| sorted_remap | Blitzclique_A | 128 | 65 | 0.508 | 0.422 to 0.593 | 23/64 | 42/64 | 45 | 166 |
| sorted_remap | Elfnote_KewlTune | 128 | 62 | 0.484 | 0.400 to 0.570 | 33/64 | 29/64 | 68 | 166 |
| shuffled_remap | 00_Base_Branded-Despia | 128 | 64 | 0.500 | 0.415 to 0.585 | 29/64 | 35/64 | 58 | 242 |
| shuffled_remap | Artmage_PowerPatron | 128 | 57 | 0.445 | 0.362 to 0.532 | 26/64 | 31/64 | 59 | 201 |
| shuffled_remap | Blitzclique_A | 128 | 65 | 0.508 | 0.422 to 0.593 | 23/64 | 42/64 | 45 | 166 |
| shuffled_remap | Elfnote_KewlTune | 128 | 62 | 0.484 | 0.400 to 0.570 | 33/64 | 29/64 | 68 | 166 |
| sorted_noremap | 00_Base_Branded-Despia | 128 | 69 | 0.539 | 0.453 to 0.623 | 34/64 | 35/64 | 63 | 219 |
| sorted_noremap | Artmage_PowerPatron | 128 | 62 | 0.484 | 0.400 to 0.570 | 30/64 | 32/64 | 62 | 215 |
| sorted_noremap | Blitzclique_A | 128 | 65 | 0.508 | 0.422 to 0.593 | 28/64 | 37/64 | 55 | 181 |
| sorted_noremap | Elfnote_KewlTune | 128 | 61 | 0.477 | 0.392 to 0.563 | 35/64 | 26/64 | 73 | 166 |
| shuffled_noremap | 00_Base_Branded-Despia | 128 | 71 | 0.555 | 0.468 to 0.638 | 34/64 | 37/64 | 61 | 220 |
| shuffled_noremap | Artmage_PowerPatron | 128 | 56 | 0.438 | 0.355 to 0.524 | 28/64 | 28/64 | 64 | 201 |
| shuffled_noremap | Blitzclique_A | 128 | 65 | 0.508 | 0.422 to 0.593 | 23/64 | 42/64 | 45 | 168 |
| shuffled_noremap | Elfnote_KewlTune | 128 | 58 | 0.453 | 0.370 to 0.539 | 30/64 | 28/64 | 66 | 160 |
| blank | 00_Base_Branded-Despia | 128 | 72 | 0.562 | 0.476 to 0.645 | 36/64 | 36/64 | 64 | 230 |
| blank | Artmage_PowerPatron | 128 | 59 | 0.461 | 0.377 to 0.547 | 24/64 | 35/64 | 53 | 212 |
| blank | Blitzclique_A | 128 | 68 | 0.531 | 0.445 to 0.616 | 28/64 | 40/64 | 52 | 156 |
| blank | Elfnote_KewlTune | 128 | 68 | 0.531 | 0.445 to 0.616 | 36/64 | 32/64 | 68 | 171 |
| ids | 00_Base_Branded-Despia | 128 | 72 | 0.562 | 0.476 to 0.645 | 31/64 | 41/64 | 54 | 223 |
| ids | Artmage_PowerPatron | 128 | 60 | 0.469 | 0.384 to 0.555 | 27/64 | 33/64 | 58 | 211 |
| ids | Blitzclique_A | 128 | 73 | 0.570 | 0.484 to 0.653 | 29/64 | 44/64 | 49 | 169 |
| ids | Elfnote_KewlTune | 128 | 66 | 0.516 | 0.430 to 0.600 | 33/64 | 33/64 | 64 | 165 |

### Results over the four decks

512 games per variant: the 95% interval of one rate is plus or minus 0.043, of the difference
between two variants about plus or minus 0.06.

| Variant | Games | A wins | A win rate | 95% interval | A first | A second | Player 0 win rate | Decisions probed | Greedy action differs | of those: deck-pointing | Max abs logit diff | Mean of max abs logit diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | 512 | 248 | 0.484 | 0.441 to 0.528 | 111/256 | 137/256 | 0.449 | n/a | n/a | n/a | n/a | n/a |
| sorted_remap | 512 | 248 | 0.484 | 0.441 to 0.528 | 111/256 | 137/256 | 0.449 | 49159 | 0 (0.00%) | 0 of 2852 | 7.8e-05 | 4.5e-06 |
| shuffled_remap | 512 | 248 | 0.484 | 0.441 to 0.528 | 111/256 | 137/256 | 0.449 | 49159 | 0 (0.00%) | 0 of 2852 | 5.9e-05 | 4.5e-06 |
| sorted_noremap | 512 | 257 | 0.502 | 0.459 to 0.545 | 127/256 | 130/256 | 0.494 | 50654 | 1154 (2.28%) | 1154 of 2975 | 7.6 | 0.033 |
| shuffled_noremap | 512 | 250 | 0.488 | 0.445 to 0.532 | 115/256 | 135/256 | 0.461 | 47382 | 1253 (2.64%) | 1253 of 2821 | 4.6 | 0.032 |
| blank | 512 | 267 | 0.521 | 0.478 to 0.564 | 124/256 | 143/256 | 0.463 | 49205 | 3005 (6.11%) | 997 of 2919 | 21 | 1.4 |
| ids | 512 | 271 | 0.529 | 0.486 to 0.572 | 120/256 | 151/256 | 0.439 | 49406 | 1123 (2.27%) | 164 of 2890 | 11 | 0.57 |

"Deck-pointing": decisions with at least one legal action whose card is in the own deck (5.8%
of seat A's decisions). "Max abs logit diff" is over the legal options of one decision.

Reading:

- **Order with the index following the card (`sorted_remap`, `shuffled_remap`): no effect at
  all.** 0 of 49,159 decisions change, the logits move by at most 7.8e-05 (float rounding), and
  all 512 games end exactly like the control games (same winner, same length, per deck).
- **Order with a stale index (`noremap`):** 2.3% and 2.6% of the decisions change, all of them
  deck searches (39% and 44% of the deck-pointing decisions). The win rates (0.502, 0.488) are
  inside the noise of the control (0.484, interval 0.441 to 0.528).
- **Deck content hidden (`blank`):** 6.1% of the decisions change, win rate 0.521, inside the
  noise. The model's play doesn't measurably depend on the own-deck rows at all.
- **Seat effect (control):** seat A wins 0.484, as expected for identical agents (0.5 is inside
  the interval). The player who moves first wins 0.449 (230 of 512, interval 0.407 to 0.492).

## Issue B: card id 0 (K1)

### How many cards look the same to the model

Printed attribute, race, level, atk, def and types from `cards.cdb`, per deck (main and extra),
script in the session scratchpad:

| Deck | Distinct codes | Distinct feature rows | Codes sharing a row with another code | Main-deck copies sharing a row |
|---|---|---|---|---|
| 00_Base_Branded-Despia | 45 | 33 | 19 | 22 of 47 |
| Artmage_DarkMagician | 45 | 35 | 14 | 28 of 60 |
| Artmage_PowerPatron | 38 | 32 | 9 | 16 of 40 |
| Blitzclique_A | 27 | 24 | 6 | 12 of 40 |
| Blitzclique_B | 33 | 28 | 8 | 14 of 40 |
| Branded-Despia_B | 39 | 28 | 17 | 22 of 40 |
| DarkMagician_A | 35 | 28 | 12 | 20 of 40 |
| DarkMagician_B | 36 | 27 | 15 | 23 of 40 |
| Elfnote_KewlTune | 40 | 39 | 2 | 2 of 40 |
| Elfnote_PowerPatron | 33 | 32 | 2 | 5 of 40 |
| SkyStriker_RadiantTyphoon | 35 | 24 | 13 | 22 of 40 |
| Mean, decks/mine | 36.9 | 30.0 | 10.6 | 16.9 of 42.5 (39%) |
| Mean, decks/train (52) | 37.2 | 29.5 | 11.7 | 17.3 of 42.8 (40%) |

All Normal Spells have the same row, all Quick-Play Spells, all Normal Traps, and so on. The
issue applies to every decision (100%), because every card row has id 0.

### The id path in the checkpoint

A card row is `MLP(features) * swish(E[id] @ W)`, followed by a layer norm (`agent.py:125,
142`). `E` is `Encoder_0/Embed_0/embedding` (15,018 x 1,024, frozen, shared with the action and
history ids) and `W` is `Encoder_0/CardEncoder_0/MLP_1/Dense_0/kernel` (1,024 x 128, no bias).
In training `id` was always 0, so `W` only ever saw row 0.

Multiplier `g_i = swish(E[i] @ W)` for the card ids of the deck folders against `g_0`:

| Cards | `|g_i - g_0| / |g_0|` mean (min to max) | Cosine to `g_0` mean (min) | `|g_i| / |g_0|` mean |
|---|---|---|---|
| decks/mine, 200 ids | 0.16 (0.10 to 0.20) | 1.00 (0.99) | 1.12 |
| decks/train, 695 ids | 0.16 (0.09 to 0.20) | 1.00 (0.99) | 1.12 |

Row 0 has norm 0.755, the card rows have norm 1.0. Real ids move the multiplier by 16% on
average and keep its direction.

### Real ids without any fine-tune (`--variant ids`)

The env's `oppo_info=True` observation has every card with its id and a visibility mask.
`deck_order_probe.py` rebuilds the normal observation from it and gives seat A the ids of the
rows whose features are visible. Check of the rebuild
(`--variant ids --check-rebuild --games 16`, Artmage_PowerPatron): 0 of 538 batched steps
differ from a normal env stepped next to it.

Result (tables above): 0.529 over 512 games (271 wins), control 0.484. The difference of
+0.045 is inside the noise (plus or minus 0.06). 2.27% of the greedy decisions change, the
mean largest logit change is 0.57. So the promoted model doesn't break when real ids appear.

## Issues C to G: frequencies (`tools/obs_v2_census.py`)

```bash
python tools/obs_v2_census.py --deck ~/ygo/decks/mine  --policy model  --checkpoint $CK --games 220 --json mine_model.json
python tools/obs_v2_census.py --deck ~/ygo/decks/train --policy model  --checkpoint $CK --games 208 --json train_model.json
python tools/obs_v2_census.py --deck ~/ygo/decks/mine  --policy random --games 40
python tools/obs_v2_census.py --deck ~/ygo/decks/train --policy random --games 40
```

One env of the recording build (`tests/ygopro_parity.so`), random decks for both seats, seed 0.
`model` is the promoted model's greedy play for both seats. The random runs repeat the games of
`docs/notes/ygoinf-announce.md`: 14,730 and 12,305 env steps and 14,691 and 12,266 decisions,
the same as there, and the same single C9 leak decision with 3 deck rows on decks/mine.

Runs were started one after another (`cards.cdb` lock).

### Totals

Decisions (share of all decisions) and games with at least one case.

| Count | mine, model (220 games, 43,163 decisions) | train, model (208 games, 39,653 decisions) | mine, random (40 games, 14,691) | train, random (40 games, 12,266) |
|---|---|---|---|---|
| C9 leak: opponent deck rows shown without a confirmation to this seat | 15 (0.035%), 12 games, 41 rows | 1 (0.003%), 1 game, 3 rows | 1 (0.007%), 1 game | 1 (0.008%), 1 game |
| C9 leak in hand or extra deck rows | 0 | 0 | 0 | 0 |
| C9 miss: a card this seat was shown stays hidden | 91 (0.211%), 31 games, 125 cards | 94 (0.237%), 24 games, 290 cards | 5 (0.034%), 4 games | 21 (0.171%), 9 games |
| `revealed_` non-empty | 1,138 (2.64%), 181 games | 917 (2.31%), 134 games | 318 (2.16%) | 252 (2.05%) |
| C10: message has more than 24 options | 22 (0.051%), 21 games | 36 (0.091%), 27 games | 9 (0.061%) | 10 (0.082%) |
| C10: by message | 19 select card, 3 idle | 25 select card, 11 idle | 7 select card, 2 idle | 7 select card, 3 unselect card |
| C10: options lost | 138 | 219 | 80 | 98 |
| C10: idle command that loses its phase change | 3 (0.007%), 3 games | 11 (0.028%), 6 games | 2, 1 game | 0 |
| C10: later step of a multi-card selection at 24 options | 17, 16 games | 16, 12 games | 1 | 2 |
| C11: a card with counters is visible | 149 (0.345%), 3 games | 118 (0.298%), 6 games | 41 (0.279%), 5 games | 174 (1.42%), 4 games |
| C11: with a real count other than 1 | 102 (0.236%), 2 games | 105 (0.265%), 5 games | 26, 4 games | 69, 3 games |
| C11: two counter types on one card | 0 | 0 | 0 | 0 |
| C12: messages dropped | 3,101 | 2,130 | 926 | 584 |
| C12: LP, turn, turn player or phase differs from full processing | 0 | 0 | 0 | 0 |
| C12: seen codes differ from full processing | 1,572 (3.64%), 32 games | 2,373 (5.98%), 24 games | 249 (1.70%), 9 games | 1,038 (8.46%), 9 games |
| C12: seen codes differ at a card-name declaration | 7 of 123 declarations, 5 games | 2 of 41, 2 games | 0 of 27 | 0 of 8 |
| C12: a decision follows a dropped message in the same buffer | 0 | 0 | 0 | 0 |
| G: End Phase discard (random), all | 6, 5 games | 8, 6 games | 1 | 3 |
| G: End Phase discard with a real choice | 4 (0.009%), 4 games | 8 (0.020%), 6 games | 1 | 3 |
| G: sort_card (default order) | 31 (0.072%), 29 games | 65 (0.164%), 30 games | 6, 6 games | 13, 8 games |
| G: sort_card with 2 or more cards | 31 | 58 | 6 | 9 |
| G: select_counter (first card first) | 1, 1 game | 3, 3 games | 2 | 1 |
| G: select_counter with 2 cards | 0 | 0 | 0 | 0 |
| G: idle command, End Phase possible but not offered | 4,035 (9.35%), 220 games | 3,978 (10.0%), 208 games | 3,211 (21.9%) | 2,706 (22.1%) |
| G: battle command, End Phase possible but not offered | 2,267 (5.25%), 220 games | 2,222 (5.60%), 208 games | 1,520 (10.3%) | 1,276 (10.4%) |
| G: multi-select fallback (`MS: idx` lines on stdout) | 0 | 0 | 0 | 0 |

The G rows are counted per message, not per model decision, because the model isn't asked.

Real counter values behind feature 15 (card-decisions): mine, model: 1: 47, 2: 76, 3: 7, 4: 8,
6: 10, 9: 1. train, model: 1: 13, 2: 50, 3: 9, 4: 16, 5: 24, 6: 1, 7: 4, 10: 1.

Dropped message types (mine, model): SHUFFLE_DECK 1,739, CARD_HINT 667, BECOME_TARGET 325,
HINT 227, MOVE 36, SHUFFLE_HAND 36, CONFIRM_DECKTOP 34, PLAYER_HINT 20, CHAIN_NEGATED 9,
RANDOM_SELECTED 8. Train, model: SHUFFLE_DECK 1,222, HINT 310, BECOME_TARGET 242, MOVE 186,
CARD_HINT 127, POS_CHANGE 12, SHUFFLE_HAND 12, RANDOM_SELECTED 6, CONFIRM_DECKTOP 5,
PLAYER_HINT 4, CHAIN_NEGATED 3, SHUFFLE_EXTRA 1.

### Own decks (model policy, 220 games)

Decisions of the deck's player, in brackets the deck-games (one game counts for both decks in it).

| Deck | Decisions | Deck-games | C9 leak | C9 miss | C10 cut | C10 phase cut | C11 counters | C12 seen codes | Discard choice | sort_card | select_counter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 00_Base_Branded-Despia | 4,814 | 49 | 0 | 3 (1) | 8 (8) | 0 | 0 | 98 (4) | 1 (1) | 0 | 0 |
| Artmage_DarkMagician | 4,243 | 42 | 0 | 9 (2) | 5 (5) | 2 (2) | 98 (2) | 278 (7) | 0 | 0 | 1 (1) |
| Artmage_PowerPatron | 3,085 | 33 | 0 | 11 (4) | 1 (1) | 1 (1) | 50 (2) | 89 (3) | 0 | 0 | 0 |
| Blitzclique_A | 3,965 | 47 | 0 | 1 (1) | 0 | 0 | 0 | 59 (2) | 0 | 0 | 0 |
| Blitzclique_B | 3,264 | 36 | 0 | 2 (2) | 0 | 0 | 0 | 95 (2) | 0 | 0 | 0 |
| Branded-Despia_B | 3,972 | 38 | 0 | 1 (1) | 7 (7) | 0 | 0 | 11 (1) | 2 (2) | 0 | 0 |
| DarkMagician_A | 2,141 | 29 | 4 (3) | 2 (1) | 0 | 0 | 0 | 319 (13) | 0 | 15 (14) | 0 |
| DarkMagician_B | 2,918 | 35 | 11 (9) | 2 (1) | 0 | 0 | 1 (1) | 536 (20) | 0 | 16 (16) | 0 |
| Elfnote_KewlTune | 2,623 | 33 | 0 | 33 (11) | 0 | 0 | 0 | 10 (1) | 0 | 0 | 0 |
| Elfnote_PowerPatron | 5,238 | 51 | 0 | 12 (3) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SkyStriker_RadiantTyphoon | 6,900 | 47 | 0 | 15 (4) | 1 (1) | 0 | 0 | 77 (3) | 1 (1) | 0 | 0 |

Hardest hit:

- C9 leak and sort_card: only DarkMagician_A and DarkMagician_B (Dark Magical Circle looks at
  the top 3 cards). sort_card appears in 30 of their 64 deck-games.
- C9 miss: Elfnote_KewlTune (11 of 33 deck-games), then SkyStriker and the PowerPatron decks.
- C10: the two Branded-Despia decks (15 of the 22 cut decisions, all card selections) and
  Artmage_DarkMagician (the idle commands that lose a phase change).
- C11: Artmage_DarkMagician and Artmage_PowerPatron only.
- C12 seen codes: the DarkMagician decks (33 of 64 deck-games).

Training pool (model policy, 208 games, 4 to 16 deck-games per deck, so only the leaders):

- C9 miss: CenturIon 24 of 617 decisions, Pachycephalo 13 of 473, Labrynth 15 of 1,317.
- C10 phase change cut: BestowedDragonF 4, SinfulSnakeKash 3, SnakeEyeTear 2, SnakeEyeAlter 1.
- C11: SinfulSnake2 51 of 649, SnakeEyeFire2 35 of 1,164.
- C12 seen codes: NatRunick 604 of 2,167, SnakeEyeTear 343 of 882, Branded60 281 of 851.
- sort_card: NatRunick 40 (9 of 11 deck-games), Floowandereeze 6 (4 of 6), SnakeEyeTear 6
  (4 of 8), DarkMagician_B 4 (4 of 5).
- End Phase discard: Voiceless 3, Branded-Despia_B, Labrynth, Branded 1 each.

Static: Dark Magical Circle (47222536) is in DarkMagician_A and DarkMagician_B. Pot of Prosperity
(84211599) is in CenturIon2, Floowandereeze, Floowandereeze2, Labrynth, Pachycephalo and
SnakeEyeAlter, and in no own deck.

### Reused and re-measured

| Number | Source | Status |
|---|---|---|
| Client view 14,570 of 14,691, client-deck view 1 of 14,691 | `ygoinf-announce.md` | Reused. The 1 leak decision with 3 rows is reproduced by the census (random policy) |
| `revealed_` non-empty in 2.4% to 2.6% of decisions | AUDIT C9 | Re-measured: 2.1% to 2.6% |
| Opponent deck or extra rows exposed in 0.04% to 0.09% | AUDIT C9 | Re-measured with a stricter definition (only rows this seat got no confirmation for, face-up extra deck cards excluded): 0.003% to 0.035% |
| Opponent hand rows never exposed | AUDIT C9 | Re-measured: 0 |
| 0.10% of decisions cut to 24 (97 card selections, 26 idle) | AUDIT C10 | Re-measured with the promoted model: 0.051% (mine), 0.091% (train) |
| LP and turn never differ, phase 0.2% to 0.8% | AUDIT C12 | Re-measured against full processing of the same stream: 0 differences in LP, turn, turn player and phase. The AUDIT phase number compared with the core's own phase, which is a different reference |
| No difference in seen codes in the parity runs | `ygoinf-announce.md` | Refined: the tracked lists differ in 1.7% to 8.5% of decisions, but only 9 of 164 card-name declarations (model policy) see a different list |
| Counter feature, End Phase discards, sort_card, select_counter, phase options | none | New |

## What `obs_v2_census.py` counts exactly

- Leak: an opponent row in the deck, hand or extra deck with a non-zero feature in columns 7 to
  40, where this seat received no `MSG_CONFIRM_CARDS` for that card since the last
  `MSG_CHAIN_SOLVED` and the card isn't a face-up extra deck card.
- Miss: a card of the opponent that this seat received a confirmation for since the last
  `MSG_CHAIN_SOLVED`, still in that place with that code, hidden or face-down, that the env
  writes without features. The window is the env's own. A client that remembers a revealed hand
  card for longer would know more. That wasn't measured.
- C10: the option count is rebuilt from the message for idle, battle, select card, unselect
  card, tribute and chain. Select sum and the later steps of a multi-card selection have no
  count in the message. For those only "24 options shown" is counted.
- C12: three trackers (`DuelState` from `tools/ygoinf_parity.py`) read the same stream: one with
  the env's drops, one with every message, one per seat.

## Not tested

See the list at the end of `docs/OBS_V2.md`.
