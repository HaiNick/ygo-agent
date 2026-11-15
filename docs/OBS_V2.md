# Observation v2: design and recommendation

Date: 2026-11-15. Branch `obs-v2-design`, based on `new-cards` (`56474f1`). This is a
measurement and design document. Nothing in the env, ygoinf, the trainer or the model was
changed. Commands and raw numbers: `docs/notes/obs-v2-design.md`.

Model: `ft_r4_100M_main15017.flax_model` (promoted). "Decisions" are decisions shown to the
model. Frequencies are from the promoted model's greedy self-play with random decks: 220 games
on `decks/mine` (43,163 decisions) and 208 games on `decks/train` (39,653 decisions).

## Summary

1. **Own deck order (A) needs no observation change and no fine-tune.** The model has no
   positional encoding on the card rows, so it can't see the order of its deck rows. Measured:
   with the deck rows sorted or reshuffled at every decision, and the action's card index
   following the card, 0 of 49,159 decisions change, the logits move by at most 7.8e-05, and
   all 512 games end exactly like the control games. The 14,570 of 14,691 differences in the
   parity client view are real differences in the input arrays, but the model's output doesn't
   depend on them. The fix is in the client contract and in how the parity test compares.
2. **A client that points deck searches at the wrong row changes 2.3% to 2.6% of the
   decisions** (39% to 44% of the deck searches). The win rate stays inside the noise (0.502
   and 0.488 against 0.484 for the control, 512 games each, plus or minus 0.06 on a
   difference). This is a client bug to avoid, not something to train for.
3. **Real card ids (B) can be switched on without breaking the promoted model.** Without any
   fine-tune, a seat that gets real ids in its visible card rows wins 0.529 of 512 mirror games
   (control 0.484, inside the noise) and 2.27% of its decisions change. Today 39% of a deck's
   main-deck copies share their feature row with a different card.
4. **The other issues are rare:** each affects 0.003% to 0.35% of decisions, except the seen
   codes of C12 (3.6% to 6.0% of decisions, but they only matter at card-name declarations,
   where 9 of 164 differ).
5. **Recommended bundle:** A now (contract and test only). Observation v2 with one fine-tune:
   real card ids, C9, real counter counts, phase changes kept under the 24 cut, full message
   processing, End Phase discards by the model, and sort_card as the last, droppable item.
   Deferred: more than 24 options, select_counter, offering End Phase directly.
6. **Training on the current observation should continue.** No fix changes the shape of an
   array in the checkpoint, so v2 can start from whichever v1 checkpoint is best at that time.

## Task 1 result: dependence on the own deck order

Promoted model against itself, mirror deck, greedy actions, 128 games per deck (64 per seat),
decks 00_Base_Branded-Despia, Artmage_PowerPatron, Blitzclique_A, Elfnote_KewlTune. Seat A
gets the changed observation. Tool: `tools/deck_order_probe.py`.

Honest variant, precisely: the rows of `cards_` with column 2 == 1 (deck) and column 4 == 0
(own) are the first n rows. They are reordered as whole rows (only columns 5 and 7 to 40 can
differ between them). `actions_` column 0 (1 + row of the action's card) either follows the
card ("index follows") or stays ("index stale").

| Variant | Games | Seat A win rate | 95% interval | Decisions where the greedy action changes | Largest logit change |
|---|---|---|---|---|---|
| Control (no change) | 512 | 0.484 | 0.441 to 0.528 | | |
| Deck rows sorted, index follows | 512 | 0.484 | 0.441 to 0.528 | 0 of 49,159 | 7.8e-05 |
| Deck rows reshuffled per decision, index follows | 512 | 0.484 | 0.441 to 0.528 | 0 of 49,159 | 5.9e-05 |
| Deck rows sorted, index stale | 512 | 0.502 | 0.459 to 0.545 | 1,154 of 50,654 (2.28%) | 7.6 |
| Deck rows reshuffled, index stale | 512 | 0.488 | 0.445 to 0.532 | 1,253 of 47,382 (2.64%) | 4.6 |
| Deck rows blanked (content hidden) | 512 | 0.521 | 0.478 to 0.564 | 3,005 of 49,205 (6.11%) | 21 |

Per deck (128 games, interval about plus or minus 0.085) every variant is inside the noise of
its control. The table per deck is in the notes.

What this says:

- **The model doesn't use the deck order, because it can't.** The two "index follows" variants
  reproduce the control games exactly, in all four decks. The cause is the architecture
  (`LlamaEncoderLayer(rope=False)` for cards and for legal actions, `agent.py:22`), so it holds
  for every checkpoint of this model, not only for the promoted one.
- **The deck order isn't a training leak either.** What isn't in the model's function can't
  have shaped its play. The value estimate has the same blind spot.
- **Even the deck content matters little:** with the deck rows blanked, 6.1% of the decisions
  change but the win rate (0.521) stays inside the noise. With 512 games an effect below about
  0.06 can't be seen.
- **Seat effect:** in the control the player who moves first wins 0.449 (230 of 512, interval
  0.407 to 0.492). Seat A, which is first in half of the games, is at 0.484 as expected.

## Issues, frequencies and fix options

Frequencies: decisions with the promoted model on `decks/mine` and `decks/train`. "Loads"
means the promoted checkpoint loads without a change to any array shape.

### A. Own deck order

- **Frequency:** every decision has the rows in core order (14,570 of 14,691 differ in the
  parity client view, reused). Effect on the model: none (measured above).
- **Hardest hit:** none.
- **Option A1 (recommended): fix the contract and the test, not the env.**
  - Client contract: send the remaining own deck in any order. When a message names a deck card
    (a search, a cost), refer to a row with that card's code. Among equal cards any copy is
    right, because their rows are identical.
  - ygoinf: resolve an own-deck card by its code when the client can't know the sequence. The
    `code` field on `select_card`, `select_tribute`, `select_sum` and `select_unselect_card`
    cards exists since `ygoinf-announce`.
  - Parity test: before comparing, bring the own-deck rows of both sides into one order
    (sorted by row bytes) and move `actions_` column 0 with them. Then the client view can
    reach 0 differences.
  - Observation shape: unchanged. Checkpoint: unchanged. Fine-tune: none.
  - Client inside Master Duel: it knows its deck list and what has left the deck, so it can
    send the remaining deck. It sees the cards it may pick in a search.
  - Risk: low. A later model with positional encoding on the card rows would bring the leak
    back. `deck_order_probe.py --variant shuffled --probe` is the guard (0 changed decisions).
- **Option A2: reorder the deck rows in the env** (sorted or shuffled). It would make the two
  inputs equal array by array, but it changes the determinism digests and gives the model
  nothing. Not recommended.
- **Option A3: accept the stale index.** A client that numbers its sorted deck and keeps the
  core's sequence numbers changes 2.3% of the decisions. No measurable loss in 512 games, but
  it's avoidable, and in the real reference client the action's card id is wrong too (not
  measured). Not recommended.

### B. Card id 0 in the card rows (K1)

- **Frequency:** 100% of decisions. On average 16.9 of 42.5 main-deck copies (39%) of an own
  deck share their feature row with a different card of the same deck (train pool: 40%).
- **Hardest hit:** Artmage_DarkMagician (28 of 60), DarkMagician_B (23 of 40), the Branded-Despia
  decks and SkyStriker (22 each). Least: Elfnote_KewlTune (2 of 40).
- **What happens with real ids on a model trained with id 0:** a card row is
  `MLP(features) * swish(E[id] @ W)`. `W` (`CardEncoder_0/MLP_1/Dense_0/kernel`, 1,024 x 128)
  only ever saw `E[0]`. For the real ids of the decks the multiplier differs from the id-0
  multiplier by 16% in norm on average (10% to 20%), with cosine 0.99 to 1.00, and a layer norm
  follows. So real ids are a mild, card-specific perturbation of every visible row. Measured
  without any fine-tune: 2.27% of the greedy decisions change, the win rate is 0.529 over 512
  games against 0.484 for the control (inside the noise, no sign of damage).
- **Option B1 (recommended): write the id in the env** for every row whose features are
  written (`_set_obs_cards` already computes `card_id` and doesn't pass it, `ygopro.h:2816`).
  - Shape: unchanged (columns 0 and 1 exist). Loads: yes. New meaning: `MLP_1/Dense_0/kernel`
    now gets all embedding rows as input. `Embed_0` stays frozen (`--freeze_id True`), so the
    row checks of `tools/test_embedding_rows.py` still hold.
  - Warm-up: none needed (measured). Start the fine-tune directly.
  - ygoinf: it has the code of every visible card and wrote the id before the K1 alignment.
    Client: it knows the code of every card it can see.
  - Risk: low for the start. Open: how much strength the ids add. Hidden cards must keep id 0,
    which the existing `hide` flag does. Together with C9 this decides which opponent cards
    carry an id, so B and C9 belong in the same change.
- **Option B2: exact warm start.** Keep id 0 in the multiplier and add the id through a new,
  zero-initialized projection (`card row + E[id] @ W_new`). The model's outputs are bit-equal at
  the start. It needs a model change and one new array (1,024 x 128). Only worth it if B1 had
  shown damage, and it didn't.
- **Option B3: leave it.** Keeps the largest known blind spot.

### C. Revealed cards (C9)

- **Frequency:** `revealed_` is non-empty in 2.64% (mine) and 2.31% (train) of decisions.
  - Miss (a card this seat was shown stays hidden): 91 decisions (0.211%) in 31 of 220 games on
    mine, 94 (0.237%) in 24 of 208 games on train.
  - Leak (opponent deck rows shown that this seat never saw): 15 decisions (0.035%) in 12 games
    on mine, 1 (0.003%) on train. No hand or extra deck rows.
- **Hardest hit:** miss: Elfnote_KewlTune (11 of 33 deck-games), SkyStriker, the PowerPatron
  decks. Leak: only DarkMagician_A and DarkMagician_B (Dark Magical Circle).
- **Option C1 (recommended): correct the viewer, per seat, per duel.**
  - `MSG_CONFIRM_CARDS`: record the card for the player who sees it (byte 0), not by the
    inverted prefix (`ygopro.h:4347`). Keep one set per seat. Unhide only the confirmed cards,
    not the whole zone. Clear at `MSG_CHAIN_SOLVED` as today, and in `reset()`.
  - Shape: unchanged. Loads: yes. New meaning: opponent hand and deck rows with features and
    (with B) an id. The model has never seen a filled opponent hand row (0 in all runs), so
    this input is new and is learned in the fine-tune.
  - ygoinf: the reference client already tracks exactly this per seat (`known`). Client in
    Master Duel: the game shows revealed cards to the player, the bridge must capture them.
  - Risk: low in the env. Medium in the bridge: a missed reveal gives the model less than
    training did. That direction is harmless compared with today's leak.
- **Option C2: C1 plus memory.** Keep a revealed card known until it moves or its zone is
  shuffled. More information, but the client must track shuffles. Not measured. Defer.
- **Option C3: only stop the leak** (clear in `reset()`, drop the entries of private looks).
  Smallest change, keeps the miss.

### D. More than 24 legal actions (C10)

- **Frequency:** 22 decisions (0.051%) in 21 games on mine, 36 (0.091%) in 27 games on train.
  By message: 44 card selections, 14 idle commands. An idle command loses its phase change in 3
  (0.007%) and 11 (0.028%) decisions. Later steps of a multi-card selection sit at 24 options
  in 17 and 16 more decisions.
- **Hardest hit:** card selections in the two Branded-Despia decks (15 of the 22 on mine). Phase
  change lost: Artmage_DarkMagician and Artmage_PowerPatron on mine, BestowedDragonF,
  SinfulSnakeKash and SnakeEyeTear in the pool.
- **Option D1 (recommended): keep the phase changes.** When an idle or battle command has more
  than 24 actions, keep the first 24 minus the number of phase actions and append the phase
  actions.
  - Shape: unchanged. Loads: yes. No new meaning. Changes 14 of 82,816 decisions.
  - ygoinf: apply the same rule to the client's action list. Client: sends all actions anyway.
  - Risk: low. The turn can always be ended.
- **Option D2: raise `max_options`** to 32 or 48. `actions_` becomes (N, 12). The action
  encoder has no positional parameters, so the checkpoint loads. The actor and the rollout
  buffers grow with N. Largest count seen: 50 (AUDIT, model play), 80 (random). Defer: it buys
  0.05% to 0.09% of decisions for a shape change everywhere.
- **Option D3: paging for card selections.** New action semantics. Defer.

### E. Counter feature (C11)

- **Frequency:** a card with counters is visible in 149 decisions (0.345%) in 3 of 220 games on
  mine, 118 (0.298%) in 6 games on train. In 102 and 105 of them a real count isn't 1. Real
  counts go up to 9 and 10. No card had two counter types.
- **Hardest hit:** Artmage_DarkMagician and Artmage_PowerPatron (all of mine). Pool:
  SinfulSnake2, SnakeEyeFire2.
- **Option E1 (recommended): write the count** (`value >> 16`, clamped to 15, of the first
  counter type).
  - Shape: unchanged. Loads: yes, the table `CardEncoder_0/Embed_6/embedding` already has 16
    rows. Rows 1 to 14 were never trained. Copy row 15 into rows 1 to 14 before the fine-tune
    (a script like `tools/remap_ckpt.py`), then every card with counters looks exactly as
    before at the start.
  - ygoinf: send the real count again (it did before the K4c alignment). Client: Master Duel
    shows the count on the card.
  - Risk: low.
- **Option E2: leave it** and keep ygoinf sending 15.

### F. Messages after a skipped message (C12)

- **Frequency:** 3,101 (mine) and 2,130 (train) messages dropped, mostly SHUFFLE_DECK,
  CARD_HINT, BECOME_TARGET and HINT. LP, turn, turn player and phase: 0 differences against
  full processing. The seen codes differ in 1,572 (3.64%) and 2,373 (5.98%) decisions, which
  only matters for the candidate order of card-name declarations: 7 of 123 (mine) and 2 of 41
  (train) declarations see a different list. No decision followed a dropped message in the
  same buffer.
- **Hardest hit:** DarkMagician_A and DarkMagician_B (33 of 64 deck-games). Pool: NatRunick,
  SnakeEyeTear, Branded60.
- **Option F1 (recommended): process every message.** Give the env the message length table
  (`tests/yrp_verify.cpp`, ported in `tools/ygoinf_parity.py:msg_len`), and skip exactly one
  message where a handler now jumps to the end of the buffer.
  - Shape: unchanged. Loads: yes. No new meaning. Observation effect: the candidate order of
    card-name declarations, and any state a dropped MOVE, POS_CHANGE or CONFIRM_DECKTOP feeds.
  - ygoinf and client: a real client receives every message, so this removes an emulation.
  - Risk: medium. About 40 handler exits change, and a wrong length desyncs the parser. The
    recorded streams of the parity build give a direct test: walk every buffer with the table.
- **Option F2: leave it** and keep the emulation in the client contract. The client must then
  drop the same messages as the env to match training.

### G. Decisions the env makes without the model

Found in `ygopro.h` (line numbers of this branch):

| # | Decision | What the env does | Frequency (mine, train) |
|---|---|---|---|
| G1 | End Phase discard for hand size (`:4980`) | Picks at random | 4 and 8 with a real choice (0.009%, 0.020%), in 4 and 6 games |
| G2 | sort_card (`:4375`) | Default order | 31 and 65 messages (0.072%, 0.164%), in 29 and 30 games |
| G3 | select_counter (`:5673`) | Takes from the first listed card, then the second | 1 and 3 messages, none with two cards |
| G4 | End Phase in an idle command while Battle Phase is possible (`:5571`), and in a battle command while Main Phase 2 is possible (`:4837`) | Not offered. The model must go through the Battle Phase and Main Phase 2 | 4,035 and 3,978 idle commands (9.4%, 10.0%), 2,267 and 2,222 battle commands (5.3%, 5.6%), every game |
| G5 | select_sum (`:5207`, `:2399`) | The selection ends at the first complete sum, a longer combination with the same start can't be chosen | 19 and 21 select_sum decisions, the case itself not counted |
| G6 | Unselecting a card, cancelling a selection (`:4904`) | Not offered | Not counted |
| G7 | Multi-card selection, card not found (`:2440`) | Answers with the first cards | 0 in all runs |
| G8 | Who moves first | Player 0 always | Every game. In Master Duel the coin toss winner chooses |
| G9 | Card-name declaration (`announce.py`) | At most 24 candidates | By design |

Hardest hit: G2 is in 30 of the 64 deck-games of DarkMagician_A and DarkMagician_B and in no
other own deck. In the pool: NatRunick (9 of 11 deck-games), Floowandereeze, SnakeEyeTear.
G1 on mine: the Branded-Despia decks and SkyStriker, once or twice each.

- **G1, option (recommended): let the model choose.** Remove the special case, the message
  then takes the normal card selection path.
  - Shape: unchanged. Loads: yes. No new array meaning, but a situation the model never
    decided. ygoinf: remove the mirrored random choice (fix 10 of `ygoinf-announce`). Client:
    Master Duel asks the player, so the bridge needs an answer anyway.
  - Risk: low.
- **G2, option (recommended, droppable): sort_card as repeated picks.** "Which card goes on
  top next", once per card, as a card selection with its own message id.
  - Shape: unchanged. Loads: yes. New meaning: message id 17 in
    `ActionEncoderV1_0/Embed_0/embedding` (30 rows, ids 1 to 16 used) is untrained.
  - ygoinf: new message type in the schema. Client: Master Duel asks for the order.
  - Risk: medium (new decision code in the env, ygoinf and the client). It is the only issue
    that hits one archetype in about half of its games.
- **G3:** defer. 4 messages in 428 games, none with a real choice.
- **G4:** defer. No information is lost and the turn can always be ended, but offering End
  Phase directly changes the action list in about 15% of all decisions. The cost today: ending
  a turn from Main Phase 1 passes through the Battle Phase and Main Phase 2. The bridge must
  click that path.
- **G5, G6, G7, G9:** leave. Document G5 as a known limit.
- **G8:** not an observation issue. The bridge needs a rule, see the questions.

## Proposed observation v2 bundle

### Goes in

| Step | Item | Changes the observation | Needs the fine-tune |
|---|---|---|---|
| 0 | A: client contract, ygoinf lookup by code, parity comparison with own-deck rows in one order | No | No |
| 1 | B1: real card ids in visible rows | Yes, every decision | Yes |
| 2 | C1: revealed cards per seat | Yes, about 0.25% of decisions | Yes |
| 3 | E1: real counter count, row copy in the checkpoint | Yes, about 0.3% | Exact at the start |
| 4 | D1: keep phase changes under the cut | Yes, 0.007% to 0.028% | No |
| 5 | F1: process every message | Yes, declarations only | No |
| 6 | G1: End Phase discards by the model | Adds decisions, 0.009% to 0.020% | Learned in the run |
| 7 | G2: sort_card as repeated picks (drop it if step 6 is late) | Adds decisions, 0.07% to 0.16% | Learned in the run |

### Deferred

D2 and D3 (more than 24 options), C2 (memory for revealed cards), G3 (select_counter), G4 (End
Phase offered directly), G5 to G7. Reason in each case: under 0.1% of decisions, or a large
change of the action distribution for no new information.

### Order of implementation

1. Step 0 on its own branch, because it needs no env change: ygoinf and
   `tools/ygoinf_parity.py`. Target: client view at 1 difference in 14,691 (the C9 leak) with
   the unchanged env.
2. An env option `obs_version` (default 1). Version 1 must keep both determinism digests. Each
   of steps 1 to 7 is one commit behind that option, each with its digest for version 2 and the
   census count it removes.
3. The same switch in ygoinf, and the reference client without the C9 and C12 emulation for
   version 2.
4. A per-seat observation version in the env or in `battle.py`, so that a v2 model can play a
   v1 model. Without it the fine-tuned model can't be compared with the promoted one.
5. Checkpoint script for E1 (copy counter row 15 to rows 1 to 14).
6. Fine-tune.

### Acceptance checks

| Check | Target |
|---|---|
| `tools/dump_trajectory.py` with `obs_version` 1 | Both digests equal `new-cards` |
| `tools/ygoinf_parity.py`, version 2, env view | 0 differences (decks/mine 40 games, decks/train 40, Artmage mirror 20) |
| The same, client view | 0 differences |
| `tools/obs_v2_census.py` on version 2 | 0 for C9 leak, C9 miss, C10 phase change cut, C12 seen codes |
| `tools/deck_order_probe.py --variant shuffled --probe`, every new checkpoint | 0 changed decisions |
| Promoted checkpoint (with the E1 row copy) on version 2 against itself on version 1, 512 games | Not below 0.44 (the lower end of the control interval) |
| `tools/check_finite.py`, `tools/test_embedding_rows.py` on version 2 | Pass |
| `tools/run_checks.sh quick` | Pass |
| After the fine-tune: v2 model against the promoted model on version 1, mirror decks | See the plan |

### Fine-tune plan

At about 4,500 steps per second: 10M steps = 37 minutes, 25M = 1.5 hours, 50M = 3.1 hours,
100M = 6.2 hours.

- Start: the best v1 checkpoint at that time, with the E1 row copy. Same settings as r5
  (`tools/train_finetune.sh`, 64 envs x 2 threads x 32 steps, `--freeze_id True`).
- Stage 1, 10M steps (37 minutes): against its start checkpoint (on version 1) on the four
  quick decks, 128 games each. Gate: 0.50 or better over the 512 games (interval plus or minus
  0.043). The cold start is already at 0.529, so a drop below 0.44 means a bug, not a slow
  start.
- Stage 2, to 50M steps (3.1 hours in total): all 11 own decks, 128 games each. Promote if the
  mean is above 0.55 and no deck is below 0.40.
- Stage 3, to 100M steps (6.2 hours in total), only if stage 2 still improved over stage 1.
- For scale: r4 needed about 105M steps (12 hours at its speed) for 0.833 against its start.
  How much the ids are worth is unknown. The plan measures it at 10M and 50M.

### Training in the meantime

Continue on the current observation. No fix changes an array shape, the only checkpoint edit is
the E1 row copy, and the cold start with real ids showed no loss. A v1 checkpoint therefore
carries over. Stop the v1 run at a milestone once the acceptance checks pass. Steps trained on
v1 after that point are worth less than steps on v2.

## Questions for Nicklas

| # | Question | Recommendation | Number behind it |
|---|---|---|---|
| 1 | A: leave the env as it is and fix the contract and the parity comparison? | Yes | 0 of 49,159 decisions change when the deck rows are reordered and the index follows. 512 of 512 games equal to the control |
| 2 | B: real card ids in v2, written directly (B1), or with the exact warm start (B2)? | B1 | 39% of main-deck copies share a row with another card. Cold start 0.529 against 0.484 (512 games each, noise 0.06), 2.27% of decisions change |
| 3 | C9: fix per seat with the env's window (C1), or also remember revealed cards (C2)? | C1 now, C2 later | Miss in 0.21% to 0.24% of decisions and 31 of 220 games. Leak in 0.035% of decisions, only the DarkMagician decks |
| 4 | C10: keep the phase changes (D1) or raise the limit (D2)? | D1 | 14 idle commands in 82,816 decisions lose their phase change. 58 decisions (0.07%) are cut at all |
| 5 | C11: real counter counts? | Yes | 0.30% to 0.35% of decisions, real counts up to 10, on mine only the two Artmage decks |
| 6 | C12: process every message in v2, or keep the emulation in the client? | Process | 9 of 164 card-name declarations see a different candidate order. Nothing else differs |
| 7 | G1: should the model choose End Phase discards? | Yes | 12 real choices in 82,816 decisions. Cheap, and it removes a special case from ygoinf |
| 8 | G2: sort_card in v2? | Yes, as the last item | 30 of 64 DarkMagician deck-games on mine. 0.07% to 0.16% of decisions |
| 9 | G4: offer End Phase directly? | No, defer | It would change the action list in about 15% of decisions and adds no information |
| 10 | Fine-tune budget? | 50M steps (3.1 hours) with a gate at 10M (37 minutes), 100M only if still improving | 4,500 steps per second |
| 11 | Continue the current run until v2 is ready? | Yes | No array shape changes. Cold start on real ids without loss |
| 12 | G8: which seat should the bridge choose after a won coin toss? | Decide per deck after a larger run. From these mirrors: second | First player wins 0.449 (230 of 512, interval 0.407 to 0.492), four decks only |
| 13 | Is the per-seat observation version for evals (implementation step 4) in scope of the v2 work? | Yes | Without it there is no v2 against v1 number |

## Not tested

- No fix was implemented. The statements on shapes, loading and exact starts come from reading
  the code and the checkpoint's arrays, not from a run. The E1 row copy wasn't run.
- No fine-tune was run. The strength gained from real ids or any other fix is unknown.
- Task 1 used 4 of the 11 own decks, mirror matches, greedy actions, one seed and one
  checkpoint. Sampled actions (as in training) weren't run. The invariance to the row order
  follows from the architecture, so it doesn't depend on these choices. The win rates of the
  stale-index, blank and ids variants do.
- That the model can't see the order of the legal actions is read from the code (`rope=False`
  in the actor). No run permuted the actions.
- The stale-index variants keep the action's true card id. The reference client's `client` view
  also changes that id. That exact input wasn't played.
- The ids variant gives ids to the rows the env shows today. With C1 a few opponent rows more
  would carry ids.
- ygoinf, the parity tool and the reference client weren't changed. The proposed comparison
  with reordered deck rows isn't implemented, so "client view at 0 differences" is a target,
  not a result.
- No real client, no Neos, no Master Duel bridge. What the bridge can read (reveals, counter
  counts, the order prompt) is an assumption from how the game shows it.
- The census ran one seed per folder (220 and 208 games). Per-deck counts rest on 29 to 51
  deck-games for the own decks and 4 to 16 for pool decks. Rare events (select_counter, phase
  change cut) have counts below 15.
- The census counts C9 with the env's window (until `MSG_CHAIN_SOLVED`). What a client with
  memory would know wasn't measured. G5 (select_sum ends early) and G6 weren't counted.
- The GPU wasn't used. All timings are CPU timings and say nothing about training speed.
