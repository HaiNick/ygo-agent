# Code audit: ygo-agent on `new-cards`

- Date: 2026-11-13
- Audited commit: `15c7726` (`new-cards`), in the worktree `~/ygo/ygo-agent-audit` on branch `audit-fixes`
- Core: ygopro-core 0.0.3 (`1f7769c`), scripts `b83db1c`
- Scope: Phase 1, read-only. Nothing in this document is fixed yet. The only repository change is this file.

## Summary

The most important results, in order:

1. **A monster with a level above 13 turns the model's output into NaN (C1).** The level embedding has
   14 rows and the env writes the level unclamped. From that decision on, both players' LSTM states
   are NaN, so both seats play option 0 until the game ends. In run `ft_0546_pool52_r4`, 2.0% of the
   logged minibatches had non-finite losses. The cause is reproduced and a one-line clamp removes all
   30 reproduced cases.
2. **The NaN is invisible in training (C2).** `cleanba.py` replaces a NaN loss with 0 before the abort
   check, so the check cannot fire, and checkpoints are saved without a finiteness check.
3. **ygoinf feeds the model inputs that training never produced (K1, K2, K3).** The env always writes
   card ID 0 in the card list, ygoinf writes real IDs. System-string effect IDs use a different order.
   The action history includes "cancel" actions in ygoinf and not in the env. This matters for Neos
   play and for the Master Duel bridge.
4. **Training throughput is limited by the actor loop, not by the simulator alone (P1).** In r4 the
   actor spends 59% of a rollout in model inference at batch size 16, and the learner idles 38% of the
   time. A different actor layout gives more SPS without touching the env; see the measured table.
5. **The MDPro4 failure of replay D has a concrete lead (O4).** D is the only tested replay whose
   player names fill all 20 characters without a terminator. The replay writer also writes two
   bytes past the name buffer (C3). A copy of D with terminated names is in the MDPro4 replay folder
   for a one-minute test.

Everything that the game engine reports to the env is parsed with the correct byte layout: all 16
decision messages and all state messages that the env reads were compared with the core's writers
and match.

## Step 0: state of the machine

| Item | Finding |
|---|---|
| Training | Not running. No tmux server, no Python processes. WSL booted 8 minutes before the audit started. |
| Newest run | `~/ygo/runs/ft_0546_pool52_r4`: started 2026-11-08 12:37, clean stop 2026-11-09 00:42 at iteration 22588. Last checkpoint `1794137851_100M_iter22588.flax_model`. SPS 2,213. Final eval vs the built-in bot 0.97. |
| Watchdog | r3 and r4: one attempt each, "exited normally". No crash files, no env timeouts, zero `lua-error` lines in all four run logs. |
| `new-cards` | Clean history, head `15c7726`, in sync with `origin-fork/new-cards`. |
| Uncommitted in the main checkout | `ygoinf/ygoinf/features.py` (parse two-column `code_list.txt`), `ygoinf/ygoinf/jax_inf.py` (`NUM_EMBEDDINGS` env var, default 15017). Untracked: `tools/start_inf_server.sh`, `tools/start_local_play.sh`, `tools/stop_local_play.sh`, `tools/ws_tcp_proxy.py`, `tools/auto_stop_at_100m.sh`, `assets/lflist.conf`, `scripts/replay/`. |
| Worktree `~/ygo/ygo-agent-replay` | Branch `replay-yrp2` at `15c7726` plus uncommitted work: a build option `xmake f --replay_format=yrp2\|yrp2_nolzma\|yrp1` (64 added lines in `xmake.lua`, `ygopro.h`, `duel_registry.h`, `tests/yrp_verify.cpp`), and test replays in `scripts/replay/` and `scripts/replay_verify_set/`. This is the code that produced the yrp1 test files C, D, and E. |
| Local play stack | Already built and used on 2026-11-08: `~/ygo/local_server/ygopro-server` (core `1f7769c`), `~/ygo/src/{srvpro,neos-ts,ygopro-server}`. `~/ygo/ygoinf.log` shows 210 predictions and zero errors with checkpoint `43M`. |
| `KNOWN_ISSUES.md` | Doesn't exist anywhere under `~/ygo`. `docs/` has five short files (action, baseline, feature engineering, network design, support). |
| Build | The worktree build is byte-identical to the main checkout's `.so` (same SHA-256), so builds are reproducible. This makes the determinism gate cheap. |

**Note:** the prompt for this audit contains two unfilled placeholders (`<DESCRIBE THE NEOS ISSUE>`,
`<YES/NO>`). The audit doesn't use that information.

## How the measurements were made

All runs used the worktree, separate processes, and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

- **Env-only benchmark**: random legal actions, self-play, train pool (52 decks), `max_options=24`.
- **Instrumented build**: a copy of the env with counters and timers behind `#ifdef YGO_AUDIT`,
  built outside the repository (`~/ygo/audit_instr/`, patch in `audit_instrumentation.patch`). It
  compares the env's LP, turn, and phase with the core's own values at every decision, counts
  truncated decisions, buffer high-water marks, and exposure of hidden zones.
- **Model self-play**: `battle.py`, filled 0546 vs itself, 128 parallel games, train pool.
- **NaN hunt**: 4,000 steps × 128 envs (512,000 decisions), actions sampled from the 0546 policy.
- **Training data**: TensorBoard scalars of run r4.
- **Training layouts**: `cleanba.py --debug` for 200 s per layout, same checkpoint, same pool.

`py-spy` and `perf` aren't installed, so the time split comes from the instrumented build.

## 1. Correctness

Severity scale: critical (wrong results in normal use), high, medium, low. Effort: S (under an
hour), M (half a day), L (more). Line numbers refer to `15c7726`.

### C1. Level above 13 produces NaN for the whole observation (critical, S)

- **Where:** `ygoenv/ygoenv/ygopro/ygopro.h:2878` (`f_cards(offset, 9) = c.level_`),
  `ygoai/rl/jax/agent.py:115` (`embed(14, ...)` for the level).
- **What's wrong:** the env writes the current level without a limit. Flax `nn.Embed` returns NaN for
  an index outside the table. One NaN card row makes the logits and the value NaN.
- **Failure scenario:** a level-raising effect puts a monster at level 14 or 15 (reproduced with a
  DARK Spellcaster 2000/1700 effect monster from the Dark Magician decks). Every decision that sees
  the card is NaN. The LSTM state becomes NaN and stays NaN until the game ends, also after the
  card leaves. `argmax` over NaN returns 0, so the agent plays option 0. In self-play both seats
  are affected. In training, the minibatch is non-finite and the update is dropped.
- **Evidence:** 30 NaN observations in 512,000 sampled decisions (with the state sanitized after
  each one, so the real count per game is much higher). Clamping the level to 13 fixes 30 of 30.
  Run r4: 44 to 46 of 2,258 logged minibatches non-finite (2.0%). Run r3: 5 of 195 (2.6%).
- **Fix:** clamp in the env: `std::min<uint32_t>(c.level_, 13)`. This matches ygoinf
  (`features.py:484`). Games without a level above 13 stay bit-identical.
- **Also needed:** a Python-side check (C2), because any future out-of-range feature fails the same way.

### C2. Training hides NaN losses and saves without a finiteness check (high, S)

- **Where:** `scripts/cleanba.py:938` and `:967` (`loss = jnp.where(isnan | isinf, 0.0, loss)`),
  `:781` (`optax.apply_if_finite(tx, max_consecutive_errors=10)`), `:1181` (`if np.isnan(loss)`),
  `:1215` (checkpoint save).
- **What's wrong:** the loss is set to 0 before it's returned, so the abort check at `:1181` never
  fires. The gradients of a NaN loss are still NaN, so `apply_if_finite` drops the update. After 10
  dropped updates in a row it applies the NaN update anyway. Checkpoints are written without
  checking the parameters.
- **Failure scenario:** a burst of NaN minibatches (C1 during a long game on several envs) reaches
  10 in a row. The parameters become NaN, every later checkpoint is NaN, and the watchdog restarts
  from the newest (NaN) file after a crash. With `--n_saved` above 0 the good checkpoints rotate out.
- **Fix:** keep the per-minibatch finite flag as a metric, log the count of dropped updates
  (`opt_state.notfinite_count` and `total_notfinite`), print a warning when it's above zero, and
  refuse to save a checkpoint with non-finite parameters. Abort when the parameters are non-finite.

### C3. Replay name writer: out-of-bounds write and unterminated names (high for `--record`, S)

- **Where:** `ygopro.h:3131-3138` (`str_to_uint16`), `:2192-2201` (caller).
- **What's wrong:** `dest[strlen(src) + 1] = 0` writes one element past the string end. For a 19-char
  name it writes `name[20]`, for a 20-char name `name[21]`. The buffer is `uint16_t name[20]`, so
  this is a stack write out of bounds. A 20-char name also has no terminator inside the 40-byte field.
- **Failure scenario:** every deck whose `"Agent <deck>"` or `"Bob <deck>"` name has 19 or more
  characters, for example `00_Base_Branded-Despia`, `Artmage_PowerPatron`, `Elfnote_KewlTune`.
  Readers that search the 40 bytes for a terminator fail. See O4.
- **Fix:** copy at most 19 characters into a zero-filled buffer.

### C4. `Reset()` swallows exceptions (medium, S)

- **Where:** `ygopro.h:5840-5845`. The task runs through `submit_task`, and the code only calls
  `fut.wait_for`, never `fut.get()`.
- **Failure scenario:** `reset()` runs the duel up to the first agent decision. In eval against the
  built-in bot, that can be a whole opponent turn. An exception there (unsupported message, C5) is
  stored in the future and discarded. The env then continues with a half-reset state and stale
  legal actions.
- **Fix:** call `fut.get()` after the wait so the exception propagates like in `Step()`.

### C5. Unsupported engine messages terminate the process (medium, M)

- **Where:** `ygopro.h:5729-5736` (unknown message) and the explicit `throw` sites: `:4820`
  (select card with min 0), `:4952` (tribute min ≠ max), `:4957` (weighted tribute), `:4989-4996`
  (select sum mode 1 or more than 2 forced cards), `:5503` (select place count above 1), `:5562`
  (more than 2 counter cards), `:5605` (announced number above 12), `:5638` (more than 1 attribute).
- **What's wrong:** these messages have no handler: `MSG_TOSS_COIN`, `MSG_TOSS_DICE`,
  `MSG_ANNOUNCE_RACE`, `MSG_CONFIRM_EXTRATOP`, `MSG_SWAP_GRAVE_DECK`, `MSG_REVERSE_DECK`,
  `MSG_DECK_TOP`, `MSG_FIELD_DISABLED`, `MSG_CANCEL_TARGET`, `MSG_ROCK_PAPER_SCISSORS`,
  `MSG_HAND_RES`, `MSG_MATCH_KILL`. An exception in `Step()` is uncaught in the env thread, so the
  process aborts. The watchdog restarts it and gives up after three crashes in an hour.
- **Evidence:** no crash in runs r1 to r4. A static scan of all deck scripts finds no coin, dice, or
  race declarations. It does find cards that can reach a `throw`:
  - Archnemeses Protos (1 train deck) declares two attributes, which throws at `:5638`.
  - Kashtira Shangri-Ira (1 train deck) disables zones (`MSG_FIELD_DISABLED`, `MSG_SELECT_DISFIELD`).
  - Wake Up Centur-Ion!, Puzzlomino, Way Where There's a Will, Runick Smiting Storm (train decks)
    announce numbers.
  - Eight train-pool cards remove counters (Borreload Savage Dragon in 4 decks, Selene in 3).
- **Fix:** add a table with every message's length so informational messages are skipped exactly
  instead of throwing, and add `tools/lint_decks.py` that flags cards whose scripts call functions
  the env can't answer. Supporting the missing decisions is a behavior change and needs your OK per case.

### C6. Message and query buffers are smaller than the core's (medium, S)

- **Where:** `ygopro.h:1949` (`byte data_[4096]`), `:1953` (`byte query_buf_[4096]`). The core's
  `SIZE_MESSAGE_BUFFER` is `0x2000` (`common.h:29`), and `query_field_card` has no size limit.
- **Evidence:** largest message buffer seen: 646 bytes. Largest query: 3,300 bytes, which is 55 deck
  cards × 60 bytes. The limit is reached at 69 cards in one location (4,140 bytes).
- **Failure scenario:** a 60-card deck game where one graveyard or banished pile passes 68 cards, or a
  message above 4,096 bytes. The copy overwrites the members after the buffer.
- **Fix:** `data_[SIZE_MESSAGE_BUFFER]`, `query_buf_[0x4000]`, and a check of the returned lengths.

### C7. `step()` doesn't check the action index (medium, S)

- **Where:** `ygopro.h:2480-2483`. `callback_(idx)` and `legal_actions_[idx]` use the Python-supplied
  index without a range check.
- **Failure scenario:** any caller that sends an index at or above `num_options` reads past the
  vector. The instrumented runs saw zero such calls in 331,000 steps, so this is a guard, not an
  active bug.
- **Fix:** throw a clear error (or clamp and count) when the index is out of range.

### C8. Card rows aren't bounded (low, S)

- **Where:** `ygopro.h:2700-2753` (`_set_obs_cards`). The comment "check offset exceeds max_cards"
  has no code. The buffer holds 160 rows.
- **Evidence:** maximum seen is 141 rows with the 60-card decks in the pool.
- **Failure scenario:** two 60-card decks (150 cards) plus more than 10 tokens. The write runs into
  the next env's slice of the batch.
- **Fix:** stop at 160 rows and count the event.

### C9. Hidden-information handling is inverted and not reset (medium, behavior change, M)

- **Where:** `ygopro.h:4242` (`revealed_.insert(ls_to_spec(loc, seq, 0, c == player))`), `:2715`
  and `:2795` (any revealed card unhides whole zones for the query), `:2113-2118` (`reset()` doesn't
  clear `revealed_`).
- **What's wrong:** in `MSG_CONFIRM_CARDS` the first byte is the player who sees the cards. The
  opponent prefix `o` is added when the card's controller *is* that viewer, which is the wrong way
  around. Cards revealed to the opponent never become visible in the observation. Cards a player
  looks at privately (own deck top) can become visible to the opponent.
- **Evidence:** `revealed_` is non-empty at 2.4% to 2.6% of decisions. Opponent hand cards were never
  exposed (0 rows in 331,000 decisions). Opponent deck or extra-deck rows were exposed in 0.04% to
  0.09% of decisions (up to 42 deck rows and 116 extra-deck rows per run; face-up extra-deck cards
  are public anyway).
- **Fix:** correct the prefix, clear `revealed_` in `reset()`. This changes observations in about
  2.5% of decisions, so it needs your approval and a short fine-tune check. The `confirmed_spec_codes_`
  hint added on this branch already covers the action side.

### C10. Legal actions are cut to 24 without priority (medium, behavior change, S)

- **Where:** `ygopro.h:2640-2642`.
- **Evidence:** 123 of 119,620 decisions (0.10%) in model self-play were truncated: 97 card
  selections and 26 idle commands. Largest option count: 50 (80 in random play).
- **Failure scenario:** in an idle command with more than 24 options, the last entries are the
  phase changes, so the agent can't enter the Battle Phase or end the turn until the list shrinks. In
  a card selection the agent can't pick candidates after the first 24.
- **Fix (needs approval):** keep the phase actions when truncating. The number of options stays 24.

### C11. Counter feature is always 15 when a card has counters (low, behavior change, S)

- **Where:** `ygopro.h:3534-3541`, `:2879`. The query returns `type | count << 16` per counter type.
  The env stores the raw word and clamps it to 15.
- **Effect:** the model sees "15" for any card with counters. ygoinf sends the real count
  (`features.py:485`), which is one more train/serve difference.
- **Fix (needs approval):** use `value >> 16`. Or make ygoinf send 15 to match training.

### C12. Messages after a skipped message are dropped (low, M)

- **Where:** every `dp_ = dl_; return;` in `handle_message` (about 40 sites, for example `:3803`,
  `:3844`, `:4416`).
- **What's wrong:** one buffer can hold several messages. Skipping to the end drops the rest.
- **Evidence:** over 331,000 decisions the env's LP and turn counter never differed from the core's.
  The phase differed in 0.2% to 0.8% of decisions outside the battle sub-phases (cause not yet
  identified) and in 8% to 11% inside them, which is inherent to the protocol. The visible effect is
  in `track_seen_codes` (`:3672`): when several cards move in one step, only the first is recorded
  as seen, which affects the candidate order for card-name declarations.
- **Fix:** the message length table from C5.

### C13. Smaller correctness items (low)

| ID | Where | What | Fix |
|---|---|---|---|
| C13a | `ygopro.h:5821-5835` | A step timeout returns a fabricated win (reward 1.0 for player 1) into training, and the stuck thread and duel leak. Zero timeouts in 100M steps. | Mark the transition as truncated, count it in the log. |
| C13b | `ygopro.h:2498-2503` | The `max_steps` cutoff declares a winner by LP (tie: player 1) with reason "LP reached 0". Not seen in the measurements. | Report it as truncation. |
| C13c | `ygopro.h:526` | `read_decks` silently skips a line with trailing spaces or a byte-order mark, so a card disappears from the deck without a message. | Trim, and warn about any skipped non-comment line. |
| C13d | `ygopro.h:1565-1693` | Calling `init_module` twice duplicates `deck_names_` (biasing random deck choice) and leaks script buffers. | Clear the globals or refuse a second call. |
| C13e | `ygopro.h:2610` | `deck_names_ids_[...]` uses `operator[]` on a shared map from every env thread. A missing name inserts and races. | Use `find`. |
| C13f | `ygopro.h:2105-2111`, `:2158` | `ai_player_` is uninitialized in self-play and read for the nickname. | Initialize to 0. |
| C13g | `ygopro.h:1226-1229` | `link_marker` is assigned after `defense` is zeroed, so it's always 0 (unused today). | Swap the lines. |
| C13h | `ygopro.h:3184-3201` | `save_replay` ignores write errors and throws if `./replay` is missing, which aborts the process. | Check `fwrite`, create the folder. |
| C13i | `scripts/eval.py:215-230` | Counts the first N finished episodes across 64 envs, which over-represents short games. Without `--checkpoint` it silently plays option 0 (`:200-204`). | Per-env quota as in `battle.py --accurate`. Print which policy plays. |
| C13n | `scripts/cleanba.py:1223` | In `--debug` mode a clean stop calls `writer.close()` on the dummy writer, which raises. `os._exit` is never reached and the actor threads keep the process alive forever. Seen during the benchmark runs. | Give the dummy writer a `close`. |
| C13j | `ygoai/utils.py:60-75` | `load_embeddings` can't read the two-column `code_list.txt` and pads to 999, so `--embedding_file` can't work with 15,017 codes. It fails loudly. | Parse the first column, drop the fixed pad. |
| C13k | `tools/train_watchdog.sh:45` | After a crash the offset comes from the last log line, not from the checkpoint, so the step counter jumps ahead by up to `save_interval × 4096`. | Store the step next to the checkpoint. |
| C13l | `scripts/cleanba.py:616` | Log timestamps are UTC+8, seven hours ahead of the machine. | Use local time. |
| C13m | `scripts/cleanba.py:1212` | Checkpoint "M" means 2^20 steps, so `100M` is 104.86 million steps. | Document it or switch to 10^6. |

### Checked and found correct

- Byte layouts of all decision messages (`select_idlecmd`, `battlecmd`, `chain`, `card`,
  `unselect_card`, `tribute`, `sum`, `position`, `place`, `disfield`, `counter`, `effectyn`, `yesno`,
  `option`, `announce_attrib`, `announce_number`, `announce_card`, `sort_card`) against
  `playerop.cpp` at `1f7769c`.
- Query layout in `get_card` and `get_cards_in_location` against `card::get_infos`.
- `MSG_CONFIRM_CARDS` (with `skip_panel`), `MSG_HINT`, `MSG_NEW_PHASE`, LP messages, `MSG_WIN`.
- Recurrent state reset: `cleanba.py`, `eval.py`, `battle.py`, and the model's own step all zero the
  state after the blank terminal observation, so training and evaluation agree.
- Seat assignment and win-rate sign in `cleanba.py`, `ygoai/rl/jax/eval.py`, and `battle.py`
  (parallel mode).
- State buffers are zero-initialized (`core/array.h:79`), which the observation code relies on.
- The duel registry lock covers every create and end call.
- Atomic writes: `cleanba.py` checkpoints, `voyage_embed.py`, and `embed_new_cards.py` write to a
  temporary file and rename. `tools/remap_ckpt.py:40` and `tools/extend_embeddings.py:63` write the
  destination directly (low, S: use the same pattern).
- `VOYAGE_API_KEY` is read from the environment and never printed.

## 2. Consistency between cleanba, eval, battle, and ygoinf

`cleanba.py`, `eval.py`, and `battle.py` build the same `RNNAgent` and feed it the same way. The
differences are all between the C++ env and ygoinf's Python re-implementation of the observation.

### K1. Card IDs in the card list: the env writes 0, ygoinf writes the real ID (high for ygoinf, S to align)

- **Where:** `ygopro.h:2744` calls `_set_obs_card_(f_cards, offset, c, hide)` without the `card_id`
  argument, so columns 0 and 1 of `cards_` are always 0. This is upstream code since the first
  commit. `ygoinf/ygoinf/features.py:465` writes the real ID.
- **Effect in training:** the model has never seen which card is in which zone. Card identity only
  arrives through the legal actions and the action history. The card features are multiplied by the
  embedding of row 0, which is why row 0 must never change.
- **Effect in ygoinf:** every visible card is multiplied by a different embedding row. The model
  runs on inputs outside its training distribution at every decision.
- **Fix:** write 0 in ygoinf to match training. Verify with the differential test in section 5.
- **Larger option (your decision):** pass the real ID in the env so the model can see the cards.
  That's a change of the observation and needs a planned fine-tune, not a bug fix.

### K2. System-string effect IDs are numbered differently (high for ygoinf, S)

- **Where:** `ygopro.h:327-408` and `:635-637`: `system_strings` is a `std::map`, so IDs follow the
  sorted keys (1 → 16, 30 → 17, 31 → 18, ..., 1050 → 38). `features.py:73-87` numbers the same
  strings in source order (1050 → 16, ..., 1 → 67, 30 → 68).
- **Effect:** for every yes/no prompt, option, and chain entry with a system description (for
  example 30 "continue this attack?", 31 "attack directly?", 221 "activate trigger effect?"), ygoinf
  sends a different effect ID than the env.
- **Fix:** build the ygoinf table from the sorted list.

### K3. Action history: ygoinf records cancels, the env doesn't (medium, S)

- **Where:** `ygopro.h:2417` returns early for `ActionAct::Cancel`. `features.py:1047-1078` records
  every action.
- **Effect:** declined chains are the most common decision, so the 32-entry history in ygoinf is
  mostly filled with entries that training never contained.
- **Fix:** skip cancels in ygoinf.

### K4. Other ygoinf differences

| ID | Where | What | Severity |
|---|---|---|---|
| K4a | `features.py:229-245` | `MSG_ANNOUNCE_CARD` isn't in `msg_to_id`, so card-name declarations fail in ygoinf. Three cards in your decks use them: Crossout Designator, Artmage Academic Arcane Arts Acropolis, The Black Goat Laughs. | medium |
| K4b | `ygoinf/ygoinf/jax_inf.py:12-22`, `features.py:99` | The committed code builds a 1,000-row table (loading the 15,018-row checkpoint raises a shape error) and can't parse the two-column code list. The fixes exist only as uncommitted edits in the main checkout. | medium |
| K4c | `features.py:484-485` | Level is clamped to 13 (the env isn't, C1). Counter is the real count (the env sends 15, C11). | see C1, C11 |
| K4d | `features.py:472-480`, `ygopro.h:2864-2873` | Position of cards in deck, hand, and extra: the env writes 0 unless face-down. ygoinf passes the client's value through. Opponent hidden cards: the env writes only location and owner. What ygoinf writes depends on what the client sends. | needs the differential test |
| K4e | `tools/start_local_play.sh` | Untracked, with `/home/user` paths and an edit of MDPro4's `hosts2.conf`. | low |

### K5. battle.py single-env path (see O1)

`scripts/battle.py:189` applies `jnp.where` to the LSTM state tuple, which fails when `num_envs` is 1.

## 3. Performance

### Where the time goes

Run r4 (TensorBoard, mean of the second half of the run), per update of 4,096 steps:

| Part | Time | Share |
|---|---|---|
| Rollout (one actor thread, 128 step calls × 16 envs) | 1.62 s | |
| of which model inference | 0.95 s (7.4 ms per call) | 59% |
| of which env stepping | 0.68 s (5.3 ms per call) | 41% |
| Learner update | 1.21 s | |
| Learner waiting for data | 0.75 s | 38% of the learner's time |

The actor is the bottleneck, and inside the actor the model call is the larger part. Inference at
batch 16 costs the same as at batch 64:

| Batch | ms per call (GPU idle) | Samples per second |
|---|---|---|
| 16 | 3.2 | 4,970 |
| 32 | 3.7 | 8,770 |
| 64 | 3.6 | 17,800 |
| 128 | 5.8 | 22,000 |
| 256 | 10.4 | 24,700 |

Env-only throughput (random play, Ryzen 5 7600X, 6 cores, 12 threads):

| Envs | Threads | Steps per second |
|---|---|---|
| 1 | 1 | 2,265 |
| 16 | 16 | 6,757 |
| 16 | 6 | 6,670 |
| 32 | 12 | 6,028 |
| 64 | 12 | 4,300 |
| 128 | 12 | 7,595 |
| 256 | 12 | 6,722 |
| 2 pools × 16 | 16 each | 10,700 total |

Twelve hardware threads give about 3 times the single-thread rate.

Per env step, from the instrumented build:

| Part | 1 thread, random | 16 threads, random | Model self-play |
|---|---|---|---|
| Core `process()` | 116 µs | 211 µs | 270 µs |
| Observation (`WriteState`) | 125 µs | 168 µs | 259 µs |
| of which core queries | 42 µs | 70 µs | 141 µs |
| Thread handoff and wait inside `Step()` | 57 µs | 105 µs | n/a |
| Reset (per episode) | 6.9 ms | 7.4 ms | 36 ms (128 threads) |

Observation encoding costs as much as the game engine itself.

### Training layouts (measured)

`cleanba.py --debug`, 200 s each (E: 240 s), filled 0546 checkpoint, train pool, eval off, 8
minibatches unless noted. SPS is the script's own figure after warm-up.

| Layout (envs × actor threads × `num_steps`) | Steps per update | SPS | vs today | Learner time per update | Learner wait |
|---|---|---|---|---|---|
| A: 16 × 2 × 128 (today) | 4,096 | 2,690 | | 1.04 s | 0.5 to 1.1 s |
| B: 64 × 1 × 64 | 4,096 | 3,170 | +18% | 0.67 s | 0.5 to 0.8 s |
| C: 128 × 1 × 32 | 4,096 | 3,855 | +43% | 0.65 s | 0.2 to 0.9 s |
| D: 64 × 2 × 32 | 4,096 | 4,260 | +58% | 0.63 s | 0.1 to 0.4 s |
| E: 64 × 2 × 128, 32 minibatches | 16,384 | 3,980 | +48% | 3.8 s | 0 s (learner-bound) |

Layout A reads 2,690 here and 2,170 to 2,210 in r4 because the benchmark has no periodic eval and
no TensorBoard writer. Compare layouts with each other, not with r4. A shorter `num_steps` also
makes the learner update cheaper (1.04 s → 0.64 s), because the LSTM unrolls over fewer steps.

### P1. Actor layout (high gain, S, hyperparameter change)

- **What:** 16 envs per actor wastes the GPU: a call for 64 envs costs the same as for 16.
- **Proposal:** more envs per actor thread, with `num_steps` reduced so the update size stays 4,096.
  The table above shows the measured SPS per layout.
- **Trade-off:** `num_steps` is the backpropagation length of the LSTM and the horizon of the
  advantage estimate. `--collect_steps` can keep a longer horizon. This is your decision.
- **Measured:** layout D (64 envs × 2 threads × 32 steps) reaches 4,260 SPS, +58%, with the same
  4,096 steps per update. Layout E keeps `num_steps` at 128 and gets +48%, but with updates four
  times as large and four times as rare.
- **Limit:** in layout E the learner never waits, so beyond about 4,000 SPS the learner is the next
  bottleneck and the env optimizations (P2 to P4) stop paying off until it's faster.

### P2. Stop deep-copying card data in the observation (medium gain, M, behavior-preserving)

- **Where:** `ygopro.h:3495` and `:3525` (`Card c = c_get_card(code)` in `get_cards_in_location`),
  `:3450` (`get_card`).
- **What:** each visible card is copied with its name, full card text, and 16 effect strings. That's
  about 100 cards and roughly 1,800 string allocations per decision. Queries are only 42 of the
  125 µs in `WriteState`.
- **Expected gain:** 40 to 80 µs per step single-threaded (the 83 µs outside the queries also cover
  the spec maps), which is 10% to 20% of env CPU. Likely more under threads because the allocator is
  shared. Measure before and after.
- **Fix:** a small struct with the dynamic fields and a pointer to the static card.

### P3. Remove the per-step thread handoff (medium gain, M, behavior-preserving)

- **Where:** `ygopro.h:5854-5880`. Every step is handed to a per-env thread pool and the worker waits.
- **What:** 57 µs per step single-threaded and 105 µs at 16 threads, which is 18% to 21% of the
  step. Its only purpose is the 600 s timeout, which never fired in 100M steps.
- **Fix:** run the step inline when `timeout` is 0, and keep today's path as the default.

### P4. Cache compiled Lua scripts (small to medium gain, M, behavior-preserving)

- **What:** each reset creates a Lua state and parses `constant.lua`, `utility.lua`, `procedure.lua`,
  and about 110 card scripts: 6.9 ms per episode, about 9% of env CPU at 190 steps per episode.
- **Fix:** compile each script once in `init_module` and hand the core the bytecode (keeping debug
  info so error messages keep their line numbers). The gain needs to be measured before committing.

### P5. Thread counts (small gain, S)

- `local_env_threads` defaults to the env count, the eval pool adds 8 more threads, and each env has
  5 pool threads of its own. With 12 hardware threads, cap the env threads at 12 in total and
  measure. Cheap to try together with P1.

### P6. Not worth doing

- Python overhead per step call is small next to inference.
- Skipping attack and defense queries for cards in the deck would save time but can change
  observations, so it isn't behavior-preserving.

## 4. Known open issues

### O1. `battle.py --record` (medium, S)

- **Where:** `scripts/battle.py:103-108` sets `num_envs = 1`, then `:126-129` (`accurate`) sets it
  back to `num_episodes`.
- **Effects:**
  - Games run in parallel while recording, although the C++ side expects one env.
  - With one episode, `main_player` is `[1]` (`:233-238`), so checkpoint1 never moves first.
  - With one env, `get_probs` applies `jnp.where` to a tuple (`:189`) and crashes.
  - In verbose mode, `main_player = 1 - main_player` (`:300`) flips every env's seat when any game
    ends, which corrupts results when more than one env runs.
- **Fix:** in record mode force one env, play the episodes one after another, alternate the seat per
  game starting with checkpoint1 first, and reset the state with `jax.tree.map`.

### O2. Legacy `ygopro0` target (low, S)

- `xmake.lua:13-28` builds `ygoenv/ygoenv/ygopro0` (4,800 lines) against the same `ygopro-core`
  package, which no longer matches the old API. References: `ygoenv/ygoenv/entry.py:22` (guarded by
  `try`), `ygoai/utils.py:47` (`YGOPro-v0`).
- **Recommendation:** remove the target, the folder, and the `YGOPro-v0` branch. Fixing it means
  porting 4,800 lines that nothing uses. `edopro` is already optional and can stay.

### O3. Disk growth (low, S)

| Item | Size now | Growth |
|---|---|---|
| Checkpoints | 3.3 GB for r4 (47 files × 71 MB), 3.8 GB total | With `--n_saved 0` and `SAVE_INTERVAL=500`: about 3.5 GB per 100M steps. With the script default of 50: about 35 GB per 100M steps. |
| TensorBoard, logs | Under 10 MB per run | Negligible. |
| Replays, evals, self-play logs | Under 25 MB together | Negligible. |
| Free space | 928 GB on the WSL disk, 159 GB free on `C:` (where the WSL disk file lives) | |

**Proposed policy:** `tools/prune_checkpoints.sh RUN_DIR` that keeps the newest 5 checkpoints, every
10M-step milestone, the init checkpoint, and every file an eval report in `~/ygo/evals` refers to.
Run it from the watchdog after each clean exit. Don't use `--n_saved` for this, because it would
also delete milestones (and see C2).

### O4. MDPro4 can't play Branded replays (timeboxed investigation)

**a) `yrp_verify` on D with MDPro4's scripts**

| Script set | A | B | C | D | E |
|---|---|---|---|---|---|
| Ours (`b83db1c`) | pass | pass | pass | pass | pass |
| MDPro4 `Data/script.zip` | pass | pass | pass | pass | pass |
| `script.zip` plus the super-pre pack on top (410 files changed, 150 added) | pass | pass | pass | pass | pass |

D uses all 336 responses and reaches `MSG_WIN` in every configuration. The pre-release scripts
don't change the outcome of this game.

**b) MDPro4's core**

`MDPro4_Data/Plugins/x86_64/ocgcore.dll` (1.69 MB) isn't a plain build of `1f7769c`. It contains
all 592 Lua API names of `1f7769c`, including the newest ones (`RevealSelectDeckSequence`,
`IsCardType`), exports `create_duel` and `create_duel_v2`, and uses Lua 5.4. It also has API that
our core lacks (`GetRegistryValue`, `SetRegistryValue`, `AnnounceCardFilter`, `GetChainMetaValue`,
`SetCardData`, `SetEntityCode`, `IsXyzSummonableByRose`, and others) and extra exports
(`preload_script`, `set_registry_value`). It's a superset fork at least as new as our core. Nothing
in the fingerprint points to different fusion or message handling.

**c) Other data MDPro4 loads**

- All 45 card codes of D exist in MDPro4's `Data/locales/en-US/cards.cdb` with identical data
  columns (type, stats, level, race, attribute, set codes, alias).
- `test-update.cdb` in the pre-release pack changes only the `ot` column (3 → 11) for 7 of D's
  codes. The core doesn't read `ot`.
- `Data/script.zip` has a script for every code in D. The pack replaces 8 of them.
- `Player.log` has no exception from a replay load, but it only covers the last session.

**New lead: the player names**

| Replay | Names in the file | Plays in MDPro4 (your report) |
|---|---|---|
| B, C | `Agent BlueEyes` / `Bob BlueEyes` | yes |
| E | `Agent Blitzclique_A` / `Bob Blitzclique_A` (17 and 19 chars) | yes |
| D | `Agent 00_Base_Brande` / `Bob 00_Base_Branded-` (20 chars, no terminator) | no |

D is the only one of the five whose name fields have no terminator (C3). The usual C# replay
reader decodes the 40 bytes and cuts at the first NUL. Without a NUL that call fails, and a failed
load is a plausible source of "not fully recorded". This also explains why removing the pre-release
pack changed nothing.

**Test (one minute):**

1. Open `D2_yrp1_Base-Branded_name19.yrp` in MDPro4. It's D with only the two terminator bytes
   changed, and it passes `yrp_verify`.
2. As a cross-check, the files `Artmage_PowerPatron_*` and `Elfnote_KewlTune_*` in the same folder
   also have unterminated names and should fail. The three `Blitzclique_A_*` files should play.

If D2 plays, the fix is C3 and nothing else is needed for MDPro4.

**d) Playing through a standard YGOPro server with ygoinf**

- **Feasible, and already done once on this PC:** `tools/start_local_play.sh` starts ygoinf (CPU,
  port 3000), `ygopro-server` (port 7911, same core `1f7769c`), a WebSocket proxy, and Neos. The log
  of 2026-11-08 shows a full session with the 43M checkpoint. MDPro4's replay folder contains
  `.yrp3d` files from that session.
- **What's needed for bot-vs-bot recordings:** a headless client that speaks the YGOPro protocol and
  asks ygoinf for each decision. Neos does this in the browser today. A headless version is new
  work (L). The server then writes standard replays.
- **ygoinf with 15,017 embeddings:** works with the uncommitted edits (K4b), but it currently feeds
  the model different inputs than training (K1, K2, K3). Until those are fixed, games played through
  ygoinf don't show the model's real strength.
- **Phase 3:** the Master Duel bridge needs ygoinf anyway, so K1 to K4 and the differential test are
  on the path either way.

**Recommendation:** test D2 first. If it plays, fix C3, keep the built-in replay writer for
self-play recordings (it's the only way to record env games exactly), and treat the server route as
part of Phase 3, not as a replay fix. Before either, fix the ygoinf input differences.

## 5. Tests and safety net

**Untested today:** the observation encoding (no golden files), ygoinf against the env, `battle.py`
and `eval.py` paths (`--record`, single env), NaN handling in training, the watchdog resume path,
deck files with unusual formatting, every message outside the 52-deck pool, and the replay writer
with long names.

**Proposal: `tools/run_checks.sh`** in the worktree, with a quick and a full mode:

| Step | Check | Pass condition | Mode |
|---|---|---|---|
| 1 | Build `ygopro_ygoenv`, `yrp_verify`, `duel_registry_stress` | Build succeeds | quick |
| 2 | `duel_registry_stress locked 16 20000` | Exit code 0 | quick |
| 3 | `tools/test_embedding_rows.py` on the filled 0546 checkpoint | Exit code 0 | quick |
| 4 | Record 4 games per deck in `~/ygo/decks/mine` (one env, long deck names included), then `yrp_verify` | All pass, all names terminated | quick: 2 decks, full: 11 |
| 5 | Determinism: 32 games, fixed seeds, action sequences and observation hashes compared with a stored reference build | Identical | quick |
| 6 | Finite outputs: 20,000 sampled decisions through the model | Zero non-finite logits or values | quick |
| 7 | Tripwire: filled 0546 vs random and vs the option-0 bot, 256 games each, per-env quota | Both at least 0.98 | full |
| 8 | Random self-play, 256 games per deck in `~/ygo/decks/mine` | 0 `lua-error` lines, 0 crashes, 0 timeouts | full |
| 9 | Differential test: the same game states through the env's encoder and ygoinf's `features.py` | Tensors equal | full, after K1 to K4 |
| 10 | `tools/lint_decks.py` on `mine` and `train` | No card that reaches an unsupported message | quick |

Step 5 needs a small tool that dumps actions and observation hashes per step. The byte-identical
rebuild found in Step 0 means the reference can be regenerated from any commit.

## 6. Code quality (brief)

- Dead or unused code: `ygopro0/` (O2), `scripts/cleanba_{l,g,nnx,rnd}.py`, `scripts/impala.py`,
  `scripts/torch_ppo/`, `ygoai/rl/agent.py`, `agent2.py`, `ppo.py`, `buffer.py`, `mcts/`,
  `spec_to_card_id` (`ygopro.h:3089-3122`, half of it has no effect), the commented timing code in
  `step()`, the unreachable part of the `MSG_HINT` handler (`:4036-4050`).
- Duplicated logic: `_set_obs_cards` and `_set_obs_mask` (`:2700-2833`), the verbose and non-verbose
  parsers of each select message, and the whole observation encoding in C++ and in ygoinf.
- Confusing names: `--num_embeddings 15017` means 15,018 rows. `ckpt_maneger`, `global_learner_decices`.
  `greedy_reward` is on in eval and off in training. `_msgs` IDs are positional.
- `_set_obs_action` reads the member `msg_` instead of the argument for `MSG_SELECT_DISFIELD` (`:3076`).
- `tools/` scripts assume `~/ygo` (44 references). Fine for one machine, worth one config file.
- Missing docs: `KNOWN_ISSUES.md`, a description of the observation as actually produced (card IDs
  are 0, counters saturate), and the replay format.
- `ygoinf/setup.py` pins `numpy==1.26.4` and `tflite-runtime`. Don't install it into `~/ygo/.venv`
  without checking against `~/ygo/constraints.txt`.

## Prioritized plan: top 10

| # | Item | What you get | Effort | Risk |
|---|---|---|---|---|
| 1 | C1: clamp the level to 13 | No more NaN games, no more dropped updates (2% of minibatches) | S | None for games without levels above 13, which stay identical |
| 2 | C2: NaN visibility and finite check before saving | A NaN can't silently poison checkpoints again | S | None (Python only, no change to the math) |
| 3 | Test suite `tools/run_checks.sh` with the determinism tool (section 5, steps 1 to 8, 10) | The acceptance gates as one command | M | None |
| 4 | C3: replay names (after you test D2 in MDPro4) | Valid replays for all deck names, likely fixes MDPro4 | S | None for game logic |
| 5 | C4, C6, C7, C8: `fut.get()`, buffer sizes, index and row bounds | Removes four memory-safety and silent-failure paths | S | None, behavior-preserving |
| 6 | P1: actor layout | +43% to +58% SPS measured, with no code change | S | Changes training hyperparameters (`num_steps`), your decision |
| 7 | K1 to K4 and the differential test (step 9) | ygoinf sees what training saw. Required for Neos and Master Duel | M | Changes ygoinf's behavior on purpose |
| 8 | P2 and P3: no card copies, inline step | About 20% plus 15% to 20% of env time | M | Low, guarded by the determinism gate |
| 9 | O1 and O2: `battle.py --record`, remove `ygopro0` | Correct recordings with seat alternation, less dead code | S | Low |
| 10 | C5 and C12: message length table and deck lint. O3: checkpoint pruning | No process aborts from informational messages, safer pool changes, bounded disk | M | Low |

Not in the top 10 because they change what the model sees or does and need a decision first: C9
(revealed cards), C10 (action truncation order), C11 (counters), and passing real card IDs in the
env (K1, larger option).

## Phase 2 results (2026-11-14, branch `audit-fixes`)

All ten plan items were approved. One commit per item, each with its test or measurement in the
commit message. `KNOWN_ISSUES.md` lists what's fixed and what's open.

| Plan item | Commit subject | Result |
|---|---|---|
| 1 (C1) | ygopro env: clamp the card level feature to 13 | Non-finite decisions in 512,000: 11 → 0 |
| 2 (C2) | cleanba: make non-finite losses visible, never save NaN checkpoints | Warning, counter, exit code 3 instead of a NaN save. `--debug` stop no longer hangs |
| 3 | Add tools/run_checks.sh | Quick and full mode |
| 4 (C3) | Replays: terminate player names, fix write past the name buffer | Names terminated for all decks, replays pass `yrp_verify` |
| 5 (C4, C6, C7, C8) | ygopro env: bounds and error-propagation fixes | Digests identical |
| 6 (P1) | train_finetune: 64 envs x 2 threads x 32 steps | See the SPS table below |
| 7 (K1 to K4) | ygoinf: encode observations like the training env | 9 of 9 encoding checks pass (2 failed before) |
| 8 (P2, P3) | don't copy card text; optional inline stepping | Digests identical. Env-only +0% to +10% and up to +30% single-threaded |
| 9 (O1, O2) | battle.py: fix --record; Remove the legacy ygopro0 env | 1 episode → 1 replay, 4 → 4 |
| 10 (C5, O3) | don't abort on display-only messages; deck lint; prune_checkpoints.sh | Coin and dice test deck: abort → 172 games |

Training SPS, `cleanba.py --debug`, 200 s, same checkpoint and pool:

| Layout | Build `15c7726` | Build `audit-fixes` |
|---|---|---|
| 16 × 2 × 128 (old default) | 2,690 | 2,850 (+6%) |
| 64 × 2 × 32 (new default), `--timeout 600` | 4,260 | 4,590 (+71% over the old default on the old build) |
| 64 × 2 × 32, `--timeout 0` | | 4,598 (no gain in training, so the default stays 600) |

Corrections to Phase 1 estimates:

- P2 (card text copies) gave about 4% on average, not the estimated 10% to 20%.
- P3 (inline stepping) gives 20% to 30% for a single env thread, 5% at 64 envs on 12 threads, and
  nothing in training, where the CPU is already saturated.

Deliberately not done:

- C12 (processing messages that follow a skipped one) changes observations and needs a verified
  length for every message. Only the abort on unknown display-only messages was removed.
- ygoinf support for card-name declarations and a state-level differential test (step 9 of the test
  plan). `tools/test_ygoinf_encoding.py` covers the encoding tables and unit behavior.
- C9, C10, C11, and real card ids in the card list (K1, larger option): these change what the model
  sees and stay open for a decision.

How the determinism gate was run: the baseline build is `15c7726`. Its digests are compared with the
level column clamped to 13 (`dump_trajectory.py --clamp-level 13`), because item 1 changes exactly
that column on purpose. With the clamp applied, every commit on this branch produces the same
digests as the baseline for `decks/mine` and `decks/train`.
