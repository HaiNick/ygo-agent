# Observation version 2: implementation notes

Branch `obs-v2-impl` (base `new-cards`). Design and decisions: `docs/OBS_V2.md` and
`docs/notes/obs-v2-design.md`. This note lists what each milestone changed, the numbers measured,
what needs a decision from Nicklas, the fine-tune command, the gate plan, and what was not tested.

All tests ran on WSL2 with the RTX 4070 SUPER. Builds, digests, census, and parity ran on the CPU.
Matches between checkpoints ran on the GPU. `--num_embeddings 15017` everywhere.

## Summary

- Observation version 1 is the default and is bit-identical to `new-cards`. The
  `tools/dump_trajectory.py` digests were checked after every commit:
  decks/mine `2b0c259a5526dc607556d957de7fc90ddc11443f1549e0f2564e2f7751e24adb`,
  decks/train `85a2ab9c25429f416299d73578b0484b314136017f36cbb686a956bc4688f0b9`.
- Observation version 2 digests at the head of the branch:
  decks/mine `f6fdf049d84021534604933307c7fd0e4ed856bd80569e070be1c56d2548ecd5`,
  decks/train `16d0383ca9b8ebbb2aa35beb8c05b15d8fb43c662c0510b33c2cd5d2b51c720b`.
- No array shape changed, in the observation or in the checkpoint.
- No card was removed from a deck.
- No training run was started.
- Version 2 start checkpoint: `~/ygo/obs_v2/ft_r4_100M_main15017_v2init.flax_model`.

## Two points that need an OK from Nicklas

Both are behind `obs_version 2` only. Version 1 is untouched.

1. **Deck picks point at the wrong card on version 1 (new finding, commit 8f05855).** For the
   selecting player's own deck, the core numbers the candidates of select_card,
   select_unselect_card, select_tribute, and select_sum as 0, 1, 2, ... (`card::get_select_sequence`).
   The env reads that number as a deck position. A deck search then points at an unrelated deck row
   and takes that row's id. In 40 random games, 671 of 674 deck picks on decks/mine and 585 of 587
   on decks/train point at the wrong card. Version 2 resolves the row by the card code in the
   message: the actions in order, each takes the first own-deck row with its code that no earlier
   action took. Result on version 2: 0 wrong of 776 and 0 of 623. This was not in the item list. It
   is in because item B1 (real ids) makes the wrong id visible to the model, and because the
   client view can't reach parity without it.
2. **C1 is wider than the decision text.** On version 2, face-up cards in the opponent's extra
   deck (Pendulum monsters) and face-up cards in the opponent's hand are always shown. Both are
   public in the game. The revealed set (cards shown by an effect) works as decided: per seat, in
   the env's window, until the next MSG_CHAIN_SOLVED.

## Milestones

| Milestone | State | Commits |
| --- | --- | --- |
| M1 ygoinf lookup by code, contract, canonical comparison | Done, with a change (see M1) | 8e1b444, 34a72c5 |
| M2 env option `obs_version` | Done | b6e55bf |
| M3 items B1, C1, E1, D1, F1, G1 | Done | 720cc37, 8f05855, 69e983e, 46276bd, 1659597, e3ad0dd, c27886f, 5ca0770 |
| M4 checkpoint for E1 | Done | bf72b31 |
| M5 ygoinf and reference client on version 2 | Done | 5ada558 |
| M6 version per seat | Done | b8181bc |
| M7 cold-start check | Done, passed | (numbers in b8181bc) |
| M8 G2 sort_card | Done | 461bbc3 |
| M9 this note | Done | this commit |

### M1

- `tools/ygoinf_parity.py` compares the own deck in a canonical order in both views, and moves
  `actions_` column 0 with it.
- ygoinf looks up an own-deck action by card code (`resolve_own_deck`).
- **Change against the plan:** the target "client view, 1 difference on version 1" can't be
  reached. The reason is the deck-pick finding above: on version 1 the env's own index is wrong,
  so a correct client disagrees with it. The lookup by code is active from version 2 on. On
  version 1, 673 decisions stay outside the comparison (1 C9 leak and 672 deck picks). The env
  view on version 1 has 0 differences.

### M2

- Env option `obs_version` (default 1). `--obs_version` in `scripts/cleanba.py`, `battle.py`, and
  `eval.py`. `--obs-version` in `tools/dump_trajectory.py`, `check_finite.py`,
  `test_embedding_rows.py`, `deck_order_probe.py`, `obs_v2_census.py`, `ygoinf_parity.py`, and
  `msg_len_check.py`.

### M3

One commit per item. Census: random play, 40 games, decks/mine and decks/train
(`tools/obs_v2_census.py`).

| Counter | Version 1, mine | Version 1, train | Version 2, mine | Version 2, train |
| --- | --- | --- | --- | --- |
| Deck pick points at the wrong card | 671 | 585 | 0 | 0 |
| B1: rows with features and card id 0 | 14,691 | 12,266 | 0 | 0 |
| C1 (C9): hidden card shown (leak) | 1 | 1 | 0 | 0 |
| C1 (C9): revealed card not shown (miss) | 5 | 21 | 0 | 0 |
| E1 (C11): counter feature differs from the count | 37 | 174 | 0 | 0 |
| D1 (C10): phase change cut off at 24 actions | 2 | 0 | 0 | 0 |
| G1: End Phase discard chosen by the env | 1 | 3 | 0 | 0 |
| G2: sort_card with 2 or more cards answered by the env | 6 | 9 | 0 | 0 |

- **F1** (every message is processed, with a length table): the census can't count it, so a probe
  measured it. Artmage mirror, 150 games: 15 of 779 card-name declarations differ between the
  full and the dropped message list. On version 2 the env shows the full list in 15 of 15. On
  version 1 it shows the dropped list in 15 of 15. `tools/msg_len_check.py` walks every buffer
  with the length table: 0 misparses in 612,615 buffers on version 2 at the head of the branch.
- Core 0.0.3 has no MSG_REFRESH_DECK, MSG_CARD_SELECTED, or MSG_UNEQUIP. The table leaves them
  out.
- `tools/run_checks.sh quick`: 9 of 9 at the end of M3.

### M4

- `tools/obs_v2_ckpt.py` copies row 15 of the counter embedding to rows 1 to 14. Output:
  `~/ygo/obs_v2/ft_r4_100M_main15017_v2init.flax_model` (74,795,617 bytes).
- Only rows 1 to 14 of the counter table differ from the promoted model
  (`tools/compare_ckpt_outputs.py`). On version 1 observations the two checkpoints give the same
  output: 32,000 decisions on the CPU, difference 0.
- The compare tool is not exact on the GPU (the same checkpoint against itself differs by up to
  0.05 in a logit). Use `JAX_PLATFORMS=cpu` for exactness checks.

### M5

- ygoinf setting `obs_version` (`OBS_VERSION` in `tools/start_inf_server.sh`). The reference
  client in `tools/ygoinf_parity.py` has no emulation of C9 and C12 on version 2.
- Parity on version 2, at the head of the branch (after M8). Differences are 0 in all runs.

| Run | Games | Decisions | Env view | Client view |
| --- | --- | --- | --- | --- |
| decks/mine | 40 | 16,908 | 0 | 0 |
| decks/train | 40 | 13,322 | 0 | 0 |
| Artmage mirror | 20 | 9,780 | 0 | 0 |

- Version 1, env view, decks/mine: 0 differences in 14,691 decisions.
- `tools/run_checks.sh quick`: 9 of 9 at the end of M5.

### M6

- Env options `obs_version2`, `obs_version_swap_from`, `obs_version_alternate`. `battle.py` has
  `--obs_version1` and `--obs_version2`. `tools/eval_vs_baseline.sh` reads `OBS_VERSION1` and
  `OBS_VERSION2` (default 1 and 1). `tools/train_finetune.sh` and `tools/ygo.sh` read
  `OBS_VERSION`. `tools/ygo.sh eval` and `record` read a run's version from its `cmd_args` and
  play the opponent on `MODEL_OBS_VERSION` (default 1).
- Seat check: 5 seatings, 22,501 to 22,733 decisions each, 0 observations of the wrong version.
- In a duel with one seat on each version the env processes every message (F1 is per duel).

### M7 and M8: cold-start check

`battle.py`, GPU, four quick decks, 128 games each, seed 0. The v2init checkpoint plays on
version 2, the promoted model on version 1. Acceptance: 0.44 or better.

| Run | Wins | Win rate | 95% interval | Per deck |
| --- | --- | --- | --- | --- |
| M7 (before G2) | 289 of 512 | 0.564 | 0.521 to 0.607 | 0.609, 0.500, 0.609, 0.539 |
| M8 (after G2) | 289 of 512 | 0.564 | 0.521 to 0.607 | 0.609, 0.500, 0.609, 0.539 |
| Control: promoted on version 1 against itself | 263 of 512 | 0.514 | 0.470 to 0.557 | |

Deck order: Branded-Despia, Artmage_PowerPatron, Blitzclique_A, Elfnote_KewlTune.

The M8 run is bit-identical to the M7 run (same game lengths and rewards). The four quick decks
never send a sort_card with 2 or more cards, so this repeat does not test G2. For that reason
there is an extra run on the two decks that do (the design note counts sort_card in 30 of their
64 deck-games), 128 games each, after G2:

| Deck | v2init on version 2 against promoted on version 1 | Control: promoted against itself |
| --- | --- | --- |
| DarkMagician_A | 59 of 128 = 0.461 | 65 of 128 = 0.508 |
| DarkMagician_B | 64 of 128 = 0.500 | 67 of 128 = 0.523 |
| Both | 123 of 256 = 0.480 (0.420 to 0.542) | 132 of 256 = 0.516 (0.455 to 0.576) |

Both decks are above 0.44. The rate is 0.036 below the control, which is inside the noise of 256
games. A small cost is plausible all the same: the sort decisions use message id 17, a row the
model has never trained. There is no run of these two decks from before G2, so the cost of G2
alone is not measured.

Other checks with the v2init checkpoint on version 2 (repeated after M8 with the same result):

- `tools/check_finite.py`: 0 non-finite values in 19,200 decisions on decks/train and on
  decks/mine.
- `tools/test_embedding_rows.py`: all PASS.
- `tools/deck_order_probe.py --variant shuffled --probe`, on the CPU, 128 games per deck:
  0 flips in 13,331, 13,840, 11,192, and 10,618 decisions (maximum logit difference 7.1e-05).
  On the GPU the probe shows 7 to 17 flips per deck with a maximum logit difference of 0.03 to
  0.05. That is GPU noise, not an effect of the deck order.

### M8

- sort_card on version 2: the model orders the cards as repeated picks ("which card goes on top
  next"), message id 17 in the actions. n cards give n - 1 decisions. Row 17 of the message
  embedding is untrained in every checkpoint so far, so the model learns this in the fine-tune.
- Sort decisions compared in parity: 14 on decks/mine, 23 on decks/train.
- `tools/run_checks.sh quick`: 9 of 9 at the end of M8. Its deck lint still prints the version 1
  text for SortDecktop cards ("sort_card is answered with the default order"). That is an info
  line and stays true for version 1.

## Client contract on version 2

A client that talks to ygoinf with `obs_version 2` must do the following. Version 1 clients need
no change.

- **Own deck actions:** send the card code from the message with every card of select_card,
  select_unselect_card, select_tribute, select_sum, and sort_card. ygoinf finds the deck row by
  code. The order of the own deck in the request does not matter.
- **Revealed cards:** set `confirmed` on a card in a deck or extra deck whose code was shown to
  this player since the last MSG_CHAIN_SOLVED. Show a revealed opponent card (hand, deck, extra
  deck) with its code for the same window.
- **Counters:** send the real counter count. ygoinf clamps it to 15.
- **End Phase discard:** send it as a normal select_card. The model decides.
- **sort_card:** one request per pick with `msg_type: sort_card`, all cards, and the picks so far
  in `selected`, until one card is left. Answer the core with one byte per card: its place, 0 = on
  top. The k-th pick gets place k (from 0), the card that is left gets the last place. A message
  with one card needs no request.
- **Card-name declarations:** as in `docs/notes/ygoinf-announce.md`. The client must process every
  message, also the ones that follow a decision in the same buffer.

## Fine-tune command

Run it from this worktree (`~/ygo/wt-obs-v2-impl`). The other worktrees don't have `obs_version`.

```bash
cd ~/ygo/wt-obs-v2-impl
INIT=~/ygo/obs_v2/ft_r4_100M_main15017_v2init.flax_model \
TB_OFFSET=104906752 \
OBS_VERSION=2 \
tools/ygo.sh start ft_obs_v2_r1
```

- `TB_OFFSET=104906752` is the step count of the promoted model, so that curves and checkpoint
  names continue. `tools/ygo.sh` sets it by itself only when `INIT` is not given.
- `OBS_VERSION=2` adds `--obs_version 2` to the run's `cmd_args`. The watchdog, `resume`, `eval`,
  and `record` read it from there.
- All other settings are the defaults of `tools/ygo.sh start` (save interval 500 iterations,
  about 2.05M steps).
- `tools/ygo.sh` and `tools/train_finetune.sh` needed no further change.

## Gate plan

The opponent is the promoted model (`scripts/checkpoints/ft_r4_100M_main15017.flax_model`) on
version 1. The run's checkpoint plays on version 2.

| Gate | When | Command | Pass |
| --- | --- | --- | --- |
| 1 | First checkpoint at or after 10M steps of this run (about step 114.9M in the name) | `tools/ygo.sh eval ft_obs_v2_r1` | Mean of the four quick decks 0.50 or better |
| 2 | First checkpoint at or after 50M steps of this run (about step 154.9M) | `tools/ygo.sh eval ft_obs_v2_r1 full` | Mean of all 11 decks above 0.55 and no deck below 0.40: promote |

- `tools/ygo.sh eval` takes the newest checkpoint of the run. For a specific checkpoint:

  ```bash
  QUICK=1 OBS_VERSION1=2 OBS_VERSION2=1 tools/eval_vs_baseline.sh \
    ~/ygo/runs/ft_obs_v2_r1/checkpoints/CHECKPOINT.flax_model \
    scripts/checkpoints/ft_r4_100M_main15017.flax_model
  ```

- Stop the run before an eval, or accept two GPU processes at once. The tests of this branch
  never ran two.
- If gate 1 fails, compare with the cold-start number first (0.564 before training). A drop below
  it after 10M steps points at the new inputs (real ids, real counter counts, sort_card), not at
  noise: the interval of 512 games is about plus or minus 0.043.
- After a promotion, the promoted model's version is 2. Set `MODEL_OBS_VERSION=2` for
  `tools/ygo.sh eval` and `record` from then on, and `OBS_VERSION=2` for the ygoinf server.

## Not tested

- `tools/ygo.sh start`, `resume`, `eval`, and `record`, and `tools/train_finetune.sh`: no training
  run was started. Tested instead: `bash -n`, the version lookup from `cmd_args`, and
  `tools/eval_vs_baseline.sh` with `OBS_VERSION1=2 OBS_VERSION2=1` on 1 deck and 16 games.
- `scripts/cleanba.py --obs_version 2`: not run. The env option behind it is covered by the tests
  above.
- `battle.py --record` and the non-accurate mode with two versions.
- A real client (MDPro) against ygoinf on version 2. Only the reference client in
  `tools/ygoinf_parity.py` was used.
- sort_card: that the picked order is the order on the deck afterwards (the direction is read
  from the core source), a sort of the opponent's deck top, and a replay file with a sort played
  back in a client.
- Parity with a trained policy. All parity and census runs use random play.
- The seat swap after an env step timeout (the replacement env starts with its initial seating).
- The deferred items D2, D3, C2, G3, G4, and G5 to G7 are not implemented.
