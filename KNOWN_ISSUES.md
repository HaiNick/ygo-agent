# Known issues

Status after the `audit-fixes` branch (2026-11-14). IDs refer to `docs/AUDIT.md`, which has the
details, measurements, and line numbers.

## Fixed on `audit-fixes`

| ID | Issue | Check that covers it |
|---|---|---|
| C1 | A card level above 13 made the model output NaN for the rest of the game | `tools/check_finite.py` |
| C2 | Training hid non-finite losses and could save NaN checkpoints | warning in `train.log`, `stats/nonfinite_updates`, exit code 3 instead of a save |
| C3 | Replay player names of 19 or more characters: write past the buffer, no terminator. Games cut off by the step limit were saved as incomplete replays | `tools/record_and_verify.sh` |
| C4, C6, C7, C8 | `Reset()` dropped exceptions; message and query buffers too small; no check of the action index; card rows unbounded | `tools/dump_trajectory.py` (no behavior change) |
| C5 (part) | Display-only messages (coin, dice, deck top, disabled zones, ...) and unknown system strings aborted the process | coin and dice test deck, see the commit message |
| C13n | Clean stop in `cleanba.py --debug` hung | |
| K1 to K4 (part) | ygoinf encoded card ids, system-string ids, history, and counters differently from training, and could not load 15,017-row checkpoints | `tools/test_ygoinf_encoding.py` |
| O1 | `battle.py --record`: parallel games, checkpoint1 never first, crash with one episode | |
| O2 | Legacy `ygopro0` target removed | |
| O3 | Checkpoint retention | `tools/prune_checkpoints.sh` (dry run by default) |
| P1 to P3 | Actor layout (`tools/train_finetune.sh`), no card text copies, optional inline stepping (`--timeout 0`) | SPS numbers in the commit messages |

Run everything with `tools/run_checks.sh quick` or `tools/run_checks.sh full`.

## Notes

- The current best model is described in `docs/model_ft_r4_100M.md`.
- Replay format: the default build writes yrp2 with LZMA, which MDPro4 plays. `xmake f
  --replay_format=yrp2_nolzma` or `--replay_format=yrp1` builds the other formats. A `yrp1` build
  creates duels with the single-seed RNG, so the same seed gives different games. Use it for
  recording only.

## Open

### Needs a decision because it changes what the model sees or does

- **The model never sees card ids in the card list (K1).** The env writes id 0 for every card in
  `cards_`. Card identity only reaches the model through legal actions and the action history. This
  is how every checkpoint was trained. Passing the real id is a change of the observation that needs
  a planned fine-tune.
- **Revealed cards (C9).** Cards shown to the opponent don't become visible in the observation, and
  cards a player looks at privately can become visible to the opponent in about 0.05% of decisions.
- **More than 24 legal actions (C10).** The list is cut after 24 entries (0.1% of decisions). In an
  idle command the phase changes are at the end, so they can be cut off.
- **Counters (C11).** The counter feature is 15 for any card with counters.
- **Messages after a skipped message (C12).** They're dropped. LP and turn tracking are unaffected
  (measured), but only the first card of a multi-card move is recorded as "seen".

### Not supported, the process aborts if it happens

`tools/lint_decks.py` lists the cards that can reach these.

- Declaring more than one attribute (Archnemeses Protos, in one training-pool deck).
- Declaring a number above 12.
- Selecting more than one zone to disable, or counters on more than two cards.
- Race declarations and rock-paper-scissors.
- Select card with a minimum of 0, tribute with different minimum and maximum, weighted tributes.

### ygoinf

- Card-name declarations (`MSG_ANNOUNCE_CARD`) aren't supported. Crossout Designator, Artmage
  Academic Arcane Arts Acropolis, and The Black Goat Laughs use them.
- No test replays the same game state through the env's encoder and ygoinf's. The position of
  hidden cards depends on what the client sends.
- `tools/start_local_play.sh` starts ygoinf, a local `ygopro-server`, a WebSocket proxy, and Neos. It
  overwrites `~/ygo/ygoinf.log` and the logs in `~/ygo/local_server`, and adds a `Local_Offline`
  host entry to MDPro4's `hosts2.conf` if that file has no local entry.

### Smaller items

- A step timeout feeds a fabricated win into training (C13a). It never happened in 100M steps.
- `scripts/eval.py` counts the first N finished games across all envs, which favors short games
  (C13i). `battle.py` in accurate mode doesn't have this problem.
- `--embedding_file` in `cleanba.py` can't read the two-column `code_list.txt` (C13j).
- After a crash the watchdog's step offset can run ahead of the checkpoint by up to
  `save_interval × 4096` steps (C13k).
- `train.log` timestamps are UTC+8 (C13l). Checkpoint names count in 2^20 steps (C13m).
- `tools/remap_ckpt.py` and `tools/extend_embeddings.py` write their output without a temporary file.
- MDPro4 and Branded replays: the unterminated names are fixed (C3). Whether that was the whole
  cause is confirmed once `D2_yrp1_Base-Branded_name19.yrp` plays in MDPro4.
