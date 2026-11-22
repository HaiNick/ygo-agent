# Model: `ft_v2_50M_main15017.flax_model`

The current best model, the first one trained on observation version 2 (`docs/OBS_V2.md`). It's
the checkpoint of run `ft_obs_v2_r1` after 50.9 million steps, copied to `scripts/checkpoints/`
(the folder is git-ignored, so the file itself isn't in the repository). It replaces
`ft_r4_100M_main15017.flax_model` (`docs/model_ft_r4_100M.md`).

| Item | Value |
|---|---|
| Source | `~/ygo/runs/ft_obs_v2_r1/checkpoints/1794766131_148M_iter9711.flax_model` (byte-identical) |
| SHA-256 | `df38671278b73ec4d288ac32adb81ec1c9763119a3637d91d64a0d41f416df76` |
| Size | 74,795,617 bytes, 18.7 million parameters (18,697,674) in 128 arrays |
| Start | `~/ygo/obs_v2/ft_r4_100M_main15017_v2init.flax_model`, SHA-256 `b0b19eb4448ba5fcdab3f8d1fb0677ca151dfab05f879730d428cba15163a1d9`: `ft_r4_100M_main15017` with rows 1 to 14 of the counter embedding set to row 15 (`tools/obs_v2_ckpt.py`, item E1) |
| Observation | Version 2. It must be served and evaluated with `obs_version 2` |
| Training | 12,429 updates of 4,096 steps = 50,909,184 steps of self-play on the 52-deck pool (`~/ygo/decks/train`), env at `9023f0d` |
| Step count | 155,783,168 on the run's step counter, the `TB_OFFSET` to continue from it (`MODEL_STEPS` in `tools/ygo.sh`) |
| Use | `--num_embeddings 15017` everywhere (the table has 15,018 rows) |

## Training

Run `ft_obs_v2_r1`, started with `tools/ygo.sh start` from the main checkout
(`~/ygo/ygo-agent`, `new-cards` at `9023f0d`) with `OBS_VERSION=2`:

- `tools/train_finetune.sh` defaults: 64 envs x 2 actor threads x 32 steps, 8 minibatches,
  learning rate 1e-4, `--vloss_clip 1.0`, `--freeze_id True`, seed 0, save interval 500
  iterations, XLA memory fraction 0.65, `--obs_version 2`.
- Segment 1, 2026-11-15 18:17 to 19:06: 2,718 iterations from the start checkpoint
  (`TB_OFFSET` 104,906,752), stopped for gate 1.
- Segment 2, 2026-11-15 19:08 to 22:05: 9,711 iterations, resumed from `1794763057_110M_iter2718`.
- 0 iterations with non-finite updates in `train.log`.
- The step counter is 32,768 below 104,906,752 + 50,909,184 = 155,815,936, because the resume
  took the last logged step (iteration 2,710) as its offset, not the saved one (2,718). That is
  issue C13k in `KNOWN_ISSUES.md`. The names of the checkpoints and the curves in TensorBoard use
  the counter, so `MODEL_STEPS` uses it too.

## Checks (2026-11-22, CPU)

- Compared with the start checkpoint, 127 of 128 arrays changed. The card embedding table
  (`Encoder_0/Embed_0`, 15,018 x 1,024) is unchanged (`--freeze_id True`).
- All parameters are finite.
- Not run on this checkpoint: `tools/check_finite.py`, `tools/test_embedding_rows.py`,
  `tools/deck_order_probe.py --probe`. They passed on the start checkpoint on version 2
  (`docs/notes/obs-v2-impl.md`, M7 and M8).

## Against the previous best model

`tools/eval_vs_baseline.sh`, mirror decks, 128 games per deck, half with each seat, greedy
actions, seed 0. This model plays on version 2, `ft_r4_100M_main15017` on version 1. Full report:
`~/ygo/evals/1794766131_148M_iter9711_vs_ft_r4_100M_main15017.md` (the checkpoint under its run
name).

| Deck | Win rate | As first player | As second player |
|---|---|---|---|
| 00_Base_Branded-Despia | 0.672 | 0.688 | 0.656 |
| Artmage_DarkMagician | 0.531 | 0.578 | 0.484 |
| Artmage_PowerPatron | 0.586 | 0.484 | 0.688 |
| Blitzclique_A | 0.656 | 0.656 | 0.656 |
| Blitzclique_B | 0.586 | 0.500 | 0.672 |
| Branded-Despia_B | 0.773 | 0.750 | 0.797 |
| DarkMagician_A | 0.695 | 0.594 | 0.797 |
| DarkMagician_B | 0.797 | 0.688 | 0.906 |
| Elfnote_KewlTune | 0.594 | 0.547 | 0.641 |
| Elfnote_PowerPatron | 0.656 | 0.656 | 0.656 |
| SkyStriker_RadiantTyphoon | 0.594 | 0.547 | 0.641 |
| **Mean** | **0.649** | | |

Per deck the 95% interval is about plus or minus 0.085. Gate 2 of the plan in
`docs/notes/obs-v2-impl.md` was a mean above 0.55 and no deck below 0.40: the mean is 0.649, the
lowest deck 0.531.

Earlier points on the same comparison:

| Checkpoint | Steps in the run | Decks | Mean | Report |
|---|---|---|---|---|
| Start checkpoint (cold start, `battle.py`) | 0 | quick 4 | 0.564 | `docs/notes/obs-v2-impl.md`, M7 |
| `1794763057_110M_iter2718` (gate 1, pass at 0.50) | 11.1M | quick 4 | 0.598 | `~/ygo/evals/1794763057_110M_iter2718_vs_ft_r4_100M_main15017.md` |
| This model | 50.9M | all 11 | 0.649 | above |

Gate 1 per deck: 00_Base_Branded-Despia 0.750, Artmage_PowerPatron 0.555, Blitzclique_A 0.523,
Elfnote_KewlTune 0.562.

The report `~/ygo/evals/1794766131_148M_iter9711_vs_ft_v2_50M_main15017.md` (mean 0.505) plays
this model against itself, both on version 2: the two files are byte-identical. It's a control
for the seat and deck balance of the eval, not a comparison.

## How to use it

`tools/ygo.sh` and `tools/start_inf_server.sh` use it by default. Run them from the main checkout
(`~/ygo/ygo-agent`), where `scripts/checkpoints/` holds the file.

```bash
# continue training from it (INIT, TB_OFFSET=155783168 and OBS_VERSION=2 are the defaults)
tools/ygo.sh start RUN_NAME

# the same, spelled out for tools/train_finetune.sh
INIT=scripts/checkpoints/ft_v2_50M_main15017.flax_model TB_OFFSET=155783168 OBS_VERSION=2 \
  tools/train_finetune.sh RUN_NAME

# compare a new version 2 checkpoint against it
OBS_VERSION1=2 OBS_VERSION2=2 tools/eval_vs_baseline.sh NEW_CHECKPOINT \
  scripts/checkpoints/ft_v2_50M_main15017.flax_model

# serve it (OBS_VERSION=2 is the default with this checkpoint)
tools/start_inf_server.sh
```

Replace `RUN_NAME` with the new run's folder name and `NEW_CHECKPOINT` with the checkpoint path.
`tools/ygo.sh eval` and `record` play a run's newest checkpoint against this model on version 2.
To compare against the previous model instead:
`MODEL=scripts/checkpoints/ft_r4_100M_main15017.flax_model MODEL_STEPS=104906752 MODEL_OBS_VERSION=1 tools/ygo.sh eval RUN`.

The baseline of `tools/eval_vs_baseline.sh` and `tools/run_checks.sh` stays
`0546_22750M_main15017_filled`, so older reports remain comparable.

## Known limits

- **Version 2 only.** A client or env on version 1 gives it inputs it wasn't trained on (card id
  0 in every row, the version 1 deck picks, no revealed cards). A ygoinf client must follow the
  version 2 contract in `docs/notes/obs-v2-impl.md`: card codes for own-deck actions,
  `confirmed`, real counter counts, End Phase discards and sort_card as decisions, every message
  processed.
- **Evals are mirror matches against one opponent**, greedy, seed 0, 128 games per deck. No eval
  against other decks, sampled actions or the built-in bot was run for this report.
- **sort_card** (message id 17) was untrained at the start. The two decks that use it most,
  DarkMagician_A and DarkMagician_B, score 0.695 and 0.797 here, but no eval isolates the sort
  decisions.
- The deferred version 2 items (more than 24 options, memory for revealed cards, select_counter,
  End Phase offered directly, select_sum and unselect limits) are open. See `KNOWN_ISSUES.md`.
- Not tested with a real client (MDPro, Neos, a Master Duel bridge).
