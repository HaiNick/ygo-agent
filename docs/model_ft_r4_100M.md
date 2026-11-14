# Model: `ft_r4_100M_main15017.flax_model`

The current best model. It's the last checkpoint of run `ft_0546_pool52_r4`, copied to
`scripts/checkpoints/` (the folder is git-ignored, so the file itself isn't in the repository).

| Item | Value |
|---|---|
| Source | `~/ygo/runs/ft_0546_pool52_r4/checkpoints/1794137851_100M_iter22588.flax_model` |
| SHA-256 (first 20 hex digits) | `1382150012b668b2c1df` |
| Size | 74,795,617 bytes, 18.7 million parameters in 128 arrays |
| Start | `0546_22750M_main15017_filled.flax_model` (upstream 0546 with the table extended to 15,017 cards) |
| Training | About 104.9 million steps of self-play on the 52-deck pool (`~/ygo/decks/train`), runs r1 to r4, env at `15c7726` |
| Use | `--num_embeddings 15017` everywhere (the table has 15,018 rows) |

## Checks (2026-11-14, env at `76cda4a`)

- The embedding table is identical to the start checkpoint: row 0 and all card rows are unchanged
  (`--freeze_id True`). The other 127 arrays changed.
- All parameters are finite. `tools/test_embedding_rows.py` passes.
- `tools/check_finite.py`: 19,200 sampled decisions, 0 non-finite.
- `scripts/eval.py`, `~/ygo/decks/mine`, 256 games, seed 0:

| Opponent | This model | Start checkpoint |
|---|---|---|
| Random bot | 1.000 | 1.000 |
| Option-0 bot | 0.988 | 0.953 |

## Against the start checkpoint

`tools/eval_vs_baseline.sh`, mirror decks, 128 games per deck, half with each seat, greedy actions.
Full report: `~/ygo/evals/ft_r4_100M_main15017_vs_0546_22750M_main15017_filled.md`.

| Deck | Win rate | As first player | As second player |
|---|---|---|---|
| 00_Base_Branded-Despia | 0.789 | 0.734 | 0.844 |
| Artmage_DarkMagician | 0.859 | 0.891 | 0.828 |
| Artmage_PowerPatron | 0.914 | 0.906 | 0.922 |
| Blitzclique_A | 0.898 | 0.859 | 0.938 |
| Blitzclique_B | 0.914 | 0.859 | 0.969 |
| Branded-Despia_B | 0.766 | 0.688 | 0.844 |
| DarkMagician_A | 0.820 | 0.828 | 0.812 |
| DarkMagician_B | 0.766 | 0.719 | 0.812 |
| Elfnote_KewlTune | 0.789 | 0.719 | 0.859 |
| Elfnote_PowerPatron | 0.891 | 0.859 | 0.922 |
| SkyStriker_RadiantTyphoon | 0.758 | 0.734 | 0.781 |
| **Mean** | **0.833** | | |

Earlier checkpoints on the four-deck quick subset scored 0.789 (11M) and 0.795 (35M) against the
same start checkpoint.

## How to use it

```bash
# continue training from it
INIT=scripts/checkpoints/ft_r4_100M_main15017.flax_model TB_OFFSET=104857600 tools/train_finetune.sh RUN_NAME

# compare a new checkpoint against it
tools/eval_vs_baseline.sh NEW_CHECKPOINT scripts/checkpoints/ft_r4_100M_main15017.flax_model

# serve it (tools/start_inf_server.sh uses it by default)
tools/start_inf_server.sh
```

Replace `RUN_NAME` with the new run's folder name and `NEW_CHECKPOINT` with the checkpoint path.
The start checkpoint stays the default baseline of `tools/eval_vs_baseline.sh` and
`tools/run_checks.sh`, so older reports remain comparable.

## Known limits

Trained before the level clamp (AUDIT C1), so about 2% of its updates were dropped and games with a
level-14 monster were played badly by both seats. See `KNOWN_ISSUES.md` for the open observation
issues.
