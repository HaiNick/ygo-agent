# Known issues

Status of `new-cards` at `9023f0d` (2026-11-22): `audit-fixes`, `train-r5`, `ygoinf-announce`,
`obs-v2-design` and `obs-v2-impl` are merged. IDs refer to `docs/AUDIT.md` (C, K, O, P) and to
`docs/OBS_V2.md` (items A to G). The notes in `docs/notes/` have the measurements.

## Fixed

| ID | Issue | Check that covers it |
|---|---|---|
| C1 | A card level above 13 made the model output NaN for the rest of the game | `tools/check_finite.py` |
| C2 | Training hid non-finite losses and could save NaN checkpoints | warning in `train.log`, `stats/nonfinite_updates`, exit code 3 instead of a save |
| C3 | Replay player names of 19 or more characters: write past the buffer, no terminator. Games cut off by the step limit were saved as incomplete replays | `tools/record_and_verify.sh` |
| C4, C6, C7, C8 | `Reset()` dropped exceptions; message and query buffers too small; no check of the action index; card rows unbounded | `tools/dump_trajectory.py` (no behavior change) |
| C5 (part) | Display-only messages (coin, dice, deck top, disabled zones, ...) and unknown system strings aborted the process | coin and dice test deck, see the commit message |
| C13n | Clean stop in `cleanba.py --debug` hung | |
| K1 to K4 | ygoinf encoded card ids, system-string ids, history, counters, places, positions and selections differently from training, and could not load 15,017-row checkpoints. Twelve causes fixed on `ygoinf-announce`, one of them a real-play bug (missing finish option of multi-card selections) | `tools/ygoinf_parity.py`: 0 differences in 36,737 decisions (env view), `tools/test_ygoinf_encoding.py` |
| K (announce) | ygoinf didn't support card-name declarations (`MSG_ANNOUNCE_CARD`: Crossout Designator, Artmage Academic Arcane Arts Acropolis, The Black Goat Laughs). Now with the env's candidate list, order and features | parity test: 144 declarations, all equal |
| Memory | The trainer's RSS grew until the OOM kill (r5 died at 15.0 GB after 2 h 44 min). Not a leak: glibc kept memory freed by the env threads. `cleanba.py` now calls `malloc_trim(0)` every 10 s (`--malloc_trim_interval`, 0 turns it off) | `docs/notes/train-r5.md`: 3.60 to 3.76 GB over 12 minutes with the fix, 4.48 to 5.67 GB without, same SPS |
| A | The own deck order: the model can't see it (no positional encoding on card rows), so no env change. ygoinf resolves own-deck actions by card code, the parity test compares the own deck in a canonical order | `tools/deck_order_probe.py --variant shuffled --probe`: 0 changed decisions |
| O1 | `battle.py --record`: parallel games, checkpoint1 never first, crash with one episode | |
| O2 | Legacy `ygopro0` target removed | |
| O3 | Checkpoint retention | `tools/prune_checkpoints.sh` (dry run by default) |
| P1 to P3 | Actor layout (`tools/train_finetune.sh`), no card text copies, optional inline stepping (`--timeout 0`) | SPS numbers in the commit messages |
| M6 | A version 2 model couldn't play a version 1 model | per-seat version in the env, `battle.py --obs_version1/--obs_version2`, `OBS_VERSION1/2` in `tools/eval_vs_baseline.sh`, `tools/ygo.sh eval` and `record` |

### Fixed on observation version 2 only

The env option `obs_version` (default 1) selects the observation. Version 1 is bit-identical to
the env before `obs-v2-impl` (both `tools/dump_trajectory.py` digests), so the version 1 models
keep the inputs they were trained on. The promoted model (`docs/model_ft_v2_50M.md`) uses
version 2. Census: `tools/obs_v2_census.py`, random play, 40 games each on `decks/mine` and
`decks/train`, version 1 and version 2.

| ID | Issue | Version 1 (mine, train) | Version 2 |
|---|---|---|---|
| Deck picks | The core numbers the candidates of a search in the own deck 0, 1, 2, ...; the env read that as a deck position, so a deck pick pointed at an unrelated row and took its card id. New finding of `obs-v2-impl`. Version 2 resolves the row by card code | 671 of 674, 585 of 587 picks wrong | 0 of 776, 0 of 623 |
| K1, item B1 | Card id 0 in every card row; identity only through actions and history | 14,691 and 12,266 rows | 0 |
| C9, item C1 | Revealed cards not shown (miss), private looks shown to the opponent (leak) | miss 5 and 21, leak 1 and 1 | 0. Per seat, until the next `MSG_CHAIN_SOLVED`. Face-up opponent hand and extra deck cards are always shown |
| C10, item D1 | Phase changes cut off when an idle or battle command has more than 24 actions | 2 and 0 | 0 |
| C11, item E1 | Counter feature 15 for any card with counters | 37 and 174 | 0, real count clamped to 15 |
| C12, item F1 | Messages after a skipped message were dropped | 15 of 15 differing declarations show the dropped list (Artmage mirror, 150 games) | full list in 15 of 15. `tools/msg_len_check.py`: 0 misparses in 612,615 buffers |
| G1 | End Phase discard for hand size picked at random by the env | 1 and 3 | 0, the model decides |
| G2 | sort_card answered with the default order | 6 and 9 | 0, repeated picks with message id 17 |

ygoinf (`OBS_VERSION` in `tools/start_inf_server.sh`) and the reference client of
`tools/ygoinf_parity.py` follow the same switch. Parity on version 2: 0 differences in the env
view and the client view (decks/mine 16,908 decisions, decks/train 13,322, Artmage mirror 9,780).

## Notes

- **Current best model:** `docs/model_ft_v2_50M.md` (version 2, 0.649 against the previous
  model). The best version 1 model is `docs/model_ft_r4_100M.md`. `tools/ygo.sh` and
  `tools/start_inf_server.sh` default to the version 2 model.
- **Training starts from the main checkout.** Start runs with `~/ygo/ygo-agent/tools/ygo.sh` on
  a merged `new-cards`. `tools/train_finetune.sh` and the watchdog run `cleanba.py` from the
  checkout that started the run (`run_info.txt` records it), every crash resume and
  `tools/ygo.sh resume` runs it from there again, and `scripts/checkpoints/` (git-ignored) only
  exists in the main checkout. A worktree that is later changed, rebuilt or removed would change
  or break a running run. Develop and test in worktrees, merge, then start. `ft_obs_v2_r1` was
  started this way, although `docs/notes/obs-v2-impl.md` still says to start it from the
  worktree.
- **CPU next to a run:** a new tmux session gets the tmux server's environment, not the caller's.
  Set `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=` inside the session command and check
  `/proc/PID/environ` (incident in `docs/notes/train-r5.md`). The checkpoint compare tools are
  only exact on the CPU (on the GPU the same checkpoint differs by up to 0.05 in a logit).
- Replay format: the default build writes yrp2 with LZMA, which MDPro4 plays. `xmake f
  --replay_format=yrp2_nolzma` or `--replay_format=yrp1` builds the other formats. A `yrp1` build
  creates duels with the single-seed RNG, so the same seed gives different games. Use it for
  recording only.

Run the checks with `tools/run_checks.sh quick` or `tools/run_checks.sh full`.

## Open

### Version 1

Every item of the table "Fixed on observation version 2 only" is still present on version 1, by
design. It matters for the version 1 models (`ft_r4_100M_main15017` and older) and for evals
against them. The deck-pick bug is in the training data of every version 1 checkpoint.

### Deferred on version 2

From `docs/OBS_V2.md`, not implemented:

- **More than 24 legal actions (C10, D2 and D3).** The list is still cut after 24 entries (0.05%
  to 0.09% of decisions); only the phase changes are kept (D1).
- **Memory for revealed cards (C2).** A revealed card is shown only until the next
  `MSG_CHAIN_SOLVED`.
- **select_counter (G3):** the env takes from the first listed card, then the second.
- **End Phase not offered directly (G4)** while Battle Phase or Main Phase 2 is possible (about
  15% of decisions). The bridge must click through those phases.
- **select_sum (G5)** ends at the first complete sum. **Unselect and cancel (G6)** aren't
  offered. **Multi-card selection with a card not found (G7)** answers with the first cards.
- **Who moves first (G8):** always player 0. The bridge needs a rule after a won coin toss.
- **Card-name declarations (G9):** at most 24 candidates.

### Not supported, the process aborts if it happens

`tools/lint_decks.py` lists the cards that can reach these. Most of them are only in
training-pool decks (`~/ygo/decks/train`), so an abort ends a training run (the watchdog
resumes), not an eval on `decks/mine`.

- Declaring more than one attribute. Pool only: Archnemeses Protos (TenyiSword). Also reachable
  in own decks: Hyperinvoked Aeon, Mudragon of the Swamp, Mixousia the Confounder.
- Declaring a number above 12. Pool only: Wake Up Centur-Ion!, Puzzlomino, Way Where There's a
  Will, Runick Smiting Storm.
- Selecting more than one zone to disable. Pool only: Sprind the Irondash Dragon, Kashtira
  Shangri-Ira.
- Counters on more than two cards. Eight cards in the pool (Black-Winged Dragon, Borreload Savage
  Dragon, two Kaiju, ...). One of them, Selene, Queen of the Master Magicians, is also in the own
  deck Artmage_DarkMagician.
- Race declarations and rock-paper-scissors (no card in either folder).
- Select card with a minimum of 0, tribute with different minimum and maximum, weighted tributes.

ygoinf fails the request in the same cases.

### ygoinf and clients

- No real client was tested against ygoinf, on either version. Only the reference client of
  `tools/ygoinf_parity.py` follows the contract.
- Neos's agent bridge has no card-name declaration (the human declares) and doesn't send the
  contract (printed values, its own card order). It isn't ready for version 2.
- `tools/start_local_play.sh` starts ygoinf, a local `ygopro-server`, a WebSocket proxy, and Neos. It
  overwrites `~/ygo/ygoinf.log` and the logs in `~/ygo/local_server`, and adds a `Local_Offline`
  host entry to MDPro4's `hosts2.conf` if that file has no local entry.
- Two parity runs started at the same moment: one failed with `database is locked` on
  `cards.cdb`.

### Untested paths on version 2

Exercised since `obs-v2-impl`: `scripts/cleanba.py --obs_version 2` and `tools/ygo.sh start`,
`stop`, `resume` and `eval` (run `ft_obs_v2_r1`, 50.9M steps, eval reports in `~/ygo/evals`), and
one `tools/ygo.sh record` of 3 games (2026-11-15). Still untested:

- `tools/ygo.sh start` with the version 2 defaults (resolved values checked, no run started).
- `battle.py` in non-accurate mode with two versions.
- sort_card: that the picked order is the order on the deck afterwards (read from the core
  source), a sort of the opponent's deck top, a replay with a sort played back in a client.
- Parity and the census with a trained policy (all runs used random play).
- The seat swap after an env step timeout (the replacement env starts with its initial seating).
- `tools/check_finite.py`, `tools/test_embedding_rows.py` and `tools/deck_order_probe.py` on a
  trained version 2 checkpoint (they ran on the start checkpoint).
- A real client on version 2.

### Smaller items

- A step timeout feeds a fabricated win into training (C13a). It never happened in 100M steps.
- `scripts/eval.py` counts the first N finished games across all envs, which favors short games
  (C13i). `battle.py` in accurate mode doesn't have this problem.
- `--embedding_file` in `cleanba.py` can't read the two-column `code_list.txt` (C13j).
- After a stop or a crash the watchdog resumes with the last logged step as `--tb_offset`, not the
  saved one, so the step counter falls behind the checkpoint by up to `log_frequency x 4096`
  steps per resume, and after a crash the checkpoint can be up to `save_interval x 4096` steps
  older than the offset (C13k). Seen in `ft_obs_v2_r1`: 32,768 steps behind at the first resume.
- `train.log` timestamps are UTC+8 (C13l). Checkpoint names count in 2^20 steps (C13m).
- `tools/remap_ckpt.py` and `tools/extend_embeddings.py` write their output without a temporary file.
- `tools/lint_decks.py` still prints "announce_card works in the env, not in ygoinf" and
  "sort_card is answered with the default order". The first is outdated, the second only true on
  version 1.
- MDPro4 and Branded replays: the unterminated names are fixed (C3). Whether that was the whole
  cause is confirmed once `D2_yrp1_Base-Branded_name19.yrp` plays in MDPro4.
