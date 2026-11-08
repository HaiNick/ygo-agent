# Baseline before fine-tuning

- Checkpoint: `0546_22750M_main15017_filled.flax_model`, `--num_embeddings 15017`
- Engine: ygopro-core 0.0.3, env at `42a16b9`
- Date: 2026-11-08
- vs random: `scripts/eval.py --bot_type random`, 256 games, agent seat random, greedy (argmax) actions
- Mirror: `scripts/battle.py`, checkpoint vs itself, 128 games, half of the games with each seat
- In the mirror both sides are the same model, so its win rate is ~0.5 by construction;
  the first-player win rate and game length are the informative columns.

| Deck | Win rate vs random | Avg length vs random | Mirror win rate | Mirror first-player win rate | Mirror avg length |
|---|---|---|---|---|---|
| 00_Base_Branded-Despia | 1.000 | 97.7 | 0.492 | 0.383 | 232.8 |
| Artmage_DarkMagician | 1.000 | 105.2 | 0.555 | 0.445 | 291.1 |
| Artmage_PowerPatron | 1.000 | 113.7 | 0.547 | 0.500 | 282.0 |
| Blitzclique_A | 0.988 | 126.2 | 0.508 | 0.414 | 232.1 |
| Blitzclique_B | 0.988 | 124.1 | 0.398 | 0.398 | 228.1 |
| Branded-Despia_B | 0.996 | 99.3 | 0.539 | 0.367 | 246.8 |
| DarkMagician_A | 1.000 | 97.7 | 0.492 | 0.367 | 230.0 |
| DarkMagician_B | 1.000 | 93.6 | 0.445 | 0.398 | 207.5 |
| Elfnote_KewlTune | 0.996 | 92.7 | 0.453 | 0.422 | 177.0 |
| Elfnote_PowerPatron | 1.000 | 94.9 | 0.500 | 0.469 | 203.5 |
| SkyStriker_RadiantTyphoon | 0.996 | 137.0 | 0.492 | 0.477 | 310.1 |

Logs: `/home/user/ygo/selfplay_logs/baseline_0546_22750M_main15017_filled`
