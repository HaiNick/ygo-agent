# promote-v2: promotion of the first version 2 model

Branch `promote-v2`, based on `new-cards` (`9023f0d`). Session of 2026-11-22. Run
`ft_obs_v2_r1` was training the whole time, so everything ran on the CPU. `ft_obs_v2_r1` was
only read; the approved cleanup deleted from the other run folders. No env build, no change to `ygoenv/`, `ygoinf/` or the training code.

## Summary

- **Promoted: `scripts/checkpoints/ft_v2_50M_main15017.flax_model`** (no new file: it already
  existed). Model card: `docs/model_ft_v2_50M.md`.
- `tools/ygo.sh` and `tools/start_inf_server.sh` default to it on observation version 2.
  `tools/ygo.sh start RUN` now begins a version 2 run from it with no extra variables.
- `tools/ygo.sh record` runs on the CPU, `eval` on the CPU while training runs (idea from
  `~/ygo/ygo-agent-cpu`).
- `KNOWN_ISSUES.md` describes `new-cards` at `9023f0d`.
- The step count of the r4 model is 104,906,752. `docs/model_ft_r4_100M.md` is fixed.
- Cleanup: a list with recommendations below. After Nicklas's OK about 6.87 GB were deleted, see
  "Cleanup done".

## Task 1: which model

| Candidate | File | SHA-256 |
|---|---|---|
| 50M checkpoint | `~/ygo/ygo-agent/scripts/checkpoints/ft_v2_50M_main15017.flax_model` | `df38671278b73ec4d288ac32adb81ec1c9763119a3637d91d64a0d41f416df76` |
| Newest checkpoint of the run at the time of the reports | `~/ygo/runs/ft_obs_v2_r1/checkpoints/1794766131_148M_iter9711.flax_model` | `df38671278b73ec4d288ac32adb81ec1c9763119a3637d91d64a0d41f416df76` |

The rule was: promote the newest checkpoint if the newest report `*_vs_ft_v2_50M_main15017.md`
has a mean of 0.53 or higher, else the 50M checkpoint. That report,
`~/ygo/evals/1794766131_148M_iter9711_vs_ft_v2_50M_main15017.md`, has a mean of 0.505, so the
rule picks the 50M checkpoint. The two files are byte-identical (same SHA-256), so the report is
a self-match and the choice changes nothing about the weights. The run was stopped at iteration
9,711 on 2026-11-15 and only resumed today at 15:35 from that same checkpoint. Since then it
saved `1795358152_150M` (15:45) and `1795358152_152M` (15:53). No eval report exists for them,
so they aren't candidates.

Numbers behind the promotion (this model on version 2 against `ft_r4_100M_main15017` on
version 1, mirror decks, 128 games per deck, greedy, seed 0):

| Check | Result | Target |
|---|---|---|
| Gate 2, all 11 decks (`1794766131_148M_iter9711_vs_ft_r4_100M_main15017.md`) | mean 0.649, lowest deck 0.531 (Artmage_DarkMagician) | mean above 0.55, no deck below 0.40 |
| Gate 1, quick 4 decks, 11.1M steps (`1794763057_110M_iter2718_vs_ft_r4_100M_main15017.md`) | 0.598 | 0.50 |
| Cold start, quick 4 decks (`docs/notes/obs-v2-impl.md`) | 0.564 | 0.44 |
| Self-match (the 0.505 report) | 0.505 | control only |

Steps: 12,429 updates of 4,096 = 50,909,184 steps in the run (2,718 + 9,711). The task text
said 50.7M; the logs give 50.9M. The run's step counter at the checkpoint is 155,783,168, which
is `MODEL_STEPS`.

## Task 3: what changed in the scripts

### `tools/ygo.sh`

| Variable | Before | Now |
|---|---|---|
| `MODEL` | `scripts/checkpoints/ft_r4_100M_main15017.flax_model` | `scripts/checkpoints/ft_v2_50M_main15017.flax_model` |
| `MODEL_STEPS` | 104906752, fixed | 155783168, can be overridden |
| `MODEL_OBS_VERSION` | 1 | 2 |
| `OBS_VERSION` in `start` without `INIT` | 1 (from `train_finetune.sh`) | `MODEL_OBS_VERSION` |

- With `INIT` given, `start` behaves as before: `TB_OFFSET` and `OBS_VERSION` default to 0 and 1
  in `tools/train_finetune.sh`.
- `eval` and `record` play a run's checkpoint on the run's version against `MODEL` on
  `MODEL_OBS_VERSION` (now 2). A version 1 run is compared as 1 against 2.
- To compare against r4: `MODEL=scripts/checkpoints/ft_r4_100M_main15017.flax_model
  MODEL_STEPS=104906752 MODEL_OBS_VERSION=1`.
- The help text is printed up to the end of the header comment, not a fixed line range.

### `tools/start_inf_server.sh`

- The default checkpoint is `ft_v2_50M_main15017.flax_model` with `OBS_VERSION` 2.
- A checkpoint given as an argument or with `CHECKPOINT`, and the fallbacks (`ft_r4_100M`, the
  newest r4 checkpoint, the baseline), default to `OBS_VERSION` 1 as before. `OBS_VERSION` set by
  hand always wins.

### The change in `~/ygo/ygo-agent-cpu`

Branch `record-on-cpu` (at `9307f77`, merged into `new-cards`, no commits of its own) had an
uncommitted change to `tools/ygo.sh`: record on the CPU, eval on the CPU while a `cleanba.py`
process runs, `DEVICE=cpu|gpu` to override, and a header printer. It fits cleanly, so it's in, as
its own commit, ported onto the per-seat version arguments. One addition: the CPU case also sets
`CUDA_VISIBLE_DEVICES=` (empty), as the notes recommend. The uncommitted diff in
`~/ygo/ygo-agent-cpu` is still there, untouched.

### How it was checked

`bash -n` on both scripts, then copies of both in a scratch folder with stubs for
`train_finetune.sh`, `eval_vs_baseline.sh`, `python` and `uvicorn`, so nothing started:

| Case | Resolved |
|---|---|
| `ygo.sh start r1` | `INIT=.../ft_v2_50M_main15017.flax_model TB_OFFSET=155783168 OBS_VERSION=2 SAVE_INTERVAL=500 XLA_MEM_FRACTION=0.65` |
| `TB_OFFSET=5` | `TB_OFFSET=5`, rest as above |
| `OBS_VERSION=1` | `OBS_VERSION=1` |
| `MODEL=r4 MODEL_STEPS=104906752 MODEL_OBS_VERSION=1` | `INIT=r4 TB_OFFSET=104906752 OBS_VERSION=1` |
| `INIT=x` | `TB_OFFSET` and `OBS_VERSION` unset (0 and 1 in `train_finetune.sh`), as before |
| `INIT=x OBS_VERSION=2 TB_OFFSET=7` | passed through |
| `eval` of a version 1 run, of a version 2 run | `OBS_VERSION1=1 OBS_VERSION2=2`, `2 2`; with `MODEL_OBS_VERSION=1`: `1 1`, `2 1` |
| `record` of a version 1 run, of a version 2 run | `--obs_version1 1 --obs_version2 2`, `2 2` |
| `eval` and `record`, training running, no `DEVICE` | `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=` |
| `DEVICE=gpu` | `JAX_PLATFORMS=cuda`, `CUDA_VISIBLE_DEVICES` unset |
| `start_inf_server.sh` | `CHECKPOINT=ft_v2_50M OBS_VERSION=2` |
| with `OBS_VERSION=1`, with an argument, with `CHECKPOINT=x OBS_VERSION=2` | `ft_v2_50M 1`, `x 1`, `x 2` |
| without the v2 file | `ft_r4_100M OBS_VERSION=1` |

## Task 5: r4 step count

`~/ygo/runs/ft_0546_pool52_r4`: `tb_offset` 12,386,304, stop at iteration 22,588, 4,096 steps
per update (16 envs x 2 actor threads x 128 steps). 12,386,304 + 22,588 x 4,096 = 104,906,752.
The last logged step, 104,873,984 at `actor_update=22580`, fits the same formula. 104,857,600 is
100 x 2^20, the rounded checkpoint name. `tools/ygo.sh` was right; the model card's example is
fixed.

## Task 6: cleanup list

The state before the cleanup; what was deleted is in "Cleanup done". Sizes from `du -sh`.
"References" means a grep over the three checkouts,
`~/ygo/runs/*/run_info.txt`, `~/ygo/evals/*.md`, `~/.bashrc` and `~/.profile`.

### Git worktrees and branches

`git worktree list`, `git branch --merged new-cards`, `git branch -r --merged new-cards`:

| Item | Size | What it is | References | Recommendation |
|---|---|---|---|---|
| Worktree `~/ygo/ygo-agent-cpu`, branch `record-on-cpu` | 6.3 MB | At `9307f77`, merged, no own commits. Uncommitted `tools/ygo.sh` (ported here), untracked `assets/locale` symlink | none | Remove the worktree and the branch once `promote-v2` is merged (`git worktree remove --force`, the diff is lost but ported) |
| Worktree `~/ygo/wt-promote`, branch `promote-v2` | 2.6 MB | This branch | none | Remove after the merge |
| Local branch `main` | | Merged | | Keep (default branch) |
| `origin-fork/audit-fixes`, `core-bump`, `obs-v2-design`, `obs-v2-impl`, `train-r5`, `wsl-build`, `ygoinf-announce` | | Merged into `new-cards` | the notes name the branches, not the refs | Delete on `origin-fork` |
| `origin-fork/main`, `origin-fork/new-cards` | | | | Keep |
| `origin/*` | | Upstream, read-only | | Leave |

### Untracked files in `~/ygo/ygo-agent`

| Item | Size | What it is | References | Recommendation |
|---|---|---|---|---|
| `assets/lflist.conf` | 6 B | Ban list file from 2026-11-08 | `docs/AUDIT.md` lists it as untracked | Keep until it's clear whether local play needs it, then commit or delete |
| `assets/locale/en/cards.cdb-shm`, `cards.cdb-wal` | | SQLite side files of an open `cards.cdb` | | Leave: they belong to a running process (training) |
| `scripts/replay/*.yrp` (2 files), `scripts/replay/old/*.yrp` (10) | 56 KB | Replays from 2026-11-07 and 08 | `docs/AUDIT.md` (as untracked) | Delete, or move to `~/ygo/evals` |
| `tools/auto_stop_at_100m.sh` | 1.7 KB | One-off stop of r4 at 100M | `docs/AUDIT.md` (as untracked) | Delete. `tools/ygo.sh stop` replaces it |

### Run folders

`tools/prune_checkpoints.sh RUN_DIR` without `--delete` (keeps the newest 5, the newest file per
10M, every checkpoint named in an eval report, the run's init):

| Run | Size | Checkpoints | Dry run | Recommendation |
|---|---|---|---|---|
| `ft_0546_pool52` | 72 MB | 1 | keeps 1 | Delete the folder: first try, superseded |
| `ft_0546_pool52_r2` | 143 MB | 2 | keeps 2 | Delete the folder. Its 4M is the init of r3, lineage only |
| `ft_0546_pool52_r3` | 286 MB | 4 | keeps 4 | Keep `1794127019_11M_iter1954` (init of r4, in eval reports), delete the other 3 (about 214 MB) |
| `ft_0546_pool52_r4` | 3.3 GB | 46 | deletes 30 (2,139 MiB), keeps 16 | Run with `--delete`. The final checkpoint is `ft_r4_100M` (same SHA-256) |
| `ft_0546_pool52_r5` | 4.9 GB | 70 | deletes 52 (3,709 MiB), keeps 18 | Run with `--delete`. The version 1 line ended here, so `KEEP_LAST=1 MILESTONE=50` would also do |
| `ft_obs_v2_r1` | 1.9 GB | 26 at the time, growing | refused: training uses the folder | Dry run again after the run stops. Keep `148M_iter9711` (the promoted model) and `110M_iter2718` (gate 1) |

With the two `--delete` runs about 5.7 GB of 11 GB are freed.

### Old folders and files under `~/ygo`

| Item | Size | What it is | References | Recommendation |
|---|---|---|---|---|
| `audit_base/` | 5.0 MB | Env build and digests of the audit base (2026-11-14) | none. The digests are in the notes | Delete |
| `audit_instr/` | 296 MB | Instrumented audit build and its data: `mdpro4/` 147 MB, `t2/` 143 MB, `audit_instrumentation.patch` 9.8 KB, `nan_hunt.npz` 228 KB | `docs/AUDIT.md` names the folder and the patch | Keep the patch (move it to `~/ygo/backup`), delete the rest |
| `replay_builds/`, `replay_builds.log` | 17 MB, 3.5 KB | yrp1, yrp2 and yrp2_nolzma builds and `yrp_verify` of the replay format test (2026-11-08) | none. The format is an `xmake` option now | Delete |
| `obs_v2/work/` | 8.1 MB | 207 files of the observation v2 sessions: census JSON, parity, digests, patches, build logs | none. The numbers are in `docs/notes/obs-v2-*.md` | Delete, or keep (small) |
| `obs_v2/ft_r4_100M_main15017_v2init.flax_model` | 71 MB | Start checkpoint of `ft_obs_v2_r1` | `ft_obs_v2_r1/init_checkpoint`, model cards | Keep |
| `ygopro_ygoenv.v0.1-prebuilt.so.bak` | 3.7 MB | Upstream prebuilt env from 2024-07-08 | none | Delete |
| `constraints.txt.bak` | 85 B | Older package pins (2026-11-07) | none (`constraints.txt` itself is in `docs/AUDIT.md`) | Delete, keep `constraints.txt` |
| `ab_build.log`, `ab_build2.log`, `smoke.log`, `train_be.log`, `train_flaxpin.log`, `replay_battle.log` | 0.5 KB, 0.5 KB, 2.8 KB, 19 KB, 19 KB, 495 KB | Setup logs from 2026-11-07 (first builds, smoke test, first training tries) | none | Delete |
| `selfplay_logs/` files from 2026-11-07: 11 per-deck logs, `artmage_dm_verbose.log`, `lua/` | about 1 MB | First self-play tests | none for these files. The folder and `baseline_0546_22750M_main15017_filled/` are used by `tools/run_baseline.sh` and `docs/baseline.md` | Delete these files, keep the folder |

Also seen, not on the list: `0546_trim13472.flax_model` (66 MB), `audit_clean.so` (4.1 MB),
`code_list_v01.txt` (empty), `code_list_v01compat.txt`, `code_list_864.txt`,
`.main_so_sha_before_replay_work`. No references were checked for these.

## Cleanup done (2026-11-22)

Done with Nicklas's OK, after the list above. Sizes from `du -sb` (MB = 10^6 bytes).

| Item | Size |
|---|---|
| `runs/ft_0546_pool52` | 74.8 MB |
| `runs/ft_0546_pool52_r2` | 149.8 MB |
| r3: `1794127019_6M`, `_8M`, `_10M` (only `1794127019_11M_iter1954` is left) | 224.4 MB |
| r4: 30 checkpoints by `tools/prune_checkpoints.sh --delete`, 16 kept, including the final `1794137851_100M_iter22588` | 2,243.9 MB |
| r5: 52 checkpoints by `tools/prune_checkpoints.sh --delete`, 18 kept, including the final `1794698802_232M_iter18908` | 3,889.4 MB |
| `audit_instr/` except `audit_instrumentation.patch` (`docs/AUDIT.md` points at the patch) | 254.0 MB |
| `audit_base/` | 5.0 MB |
| `replay_builds/` and `replay_builds.log` | 16.8 MB |
| `obs_v2/work/` (`obs_v2/ft_r4_100M_main15017_v2init.flax_model` is kept) | 7.9 MB |
| `ygopro_ygoenv.v0.1-prebuilt.so.bak`, `constraints.txt.bak` | 3.8 MB |
| Setup logs from 2026-11-07: `ab_build.log`, `ab_build2.log`, `smoke.log`, `train_be.log`, `train_flaxpin.log`, `replay_battle.log` | 0.5 MB |
| **Total** | **about 6.87 GB** |

- r4 and r5 were pruned only after a fresh dry run listed the same 30 and 52 files as above. The
  only run checkpoint a doc names (r4's final) is kept.
- Deleted on `origin-fork`, each checked to be in `new-cards` first: `audit-fixes`, `core-bump`,
  `obs-v2-design`, `obs-v2-impl`, `train-r5`, `wsl-build`, `ygoinf-announce`. Left: `main`,
  `new-cards`, `promote-v2`.

| Disk usage | Before (16:12) | After |
|---|---|---|
| `~/ygo` | 26 GB | 19 GB |
| `~/ygo/runs` | 11 GB | 4.6 GB |
| Used on the WSL disk | 38 GB | 32 GB |

The drop is a little smaller than the deletions because `ft_obs_v2_r1` kept saving a 75 MB
checkpoint about every 8 minutes.

### Pending

- After `promote-v2` is merged: remove the worktree `~/ygo/ygo-agent-cpu` and the branch
  `record-on-cpu` (its uncommitted `tools/ygo.sh` diff is ported in `f4ba63e`), and this worktree
  `~/ygo/wt-promote`.
- After `ft_obs_v2_r1` stops: dry run of `tools/prune_checkpoints.sh ~/ygo/runs/ft_obs_v2_r1`,
  then `--delete` with an OK. Keep `1794766131_148M_iter9711` (the promoted model) and
  `1794763057_110M_iter2718` (gate 1).
- Not decided: the `selfplay_logs/` files from 2026-11-07 (about 1 MB) and the untracked files in
  `~/ygo/ygo-agent` (see the list above).
- `runs/ft_0546_pool52_r3/init_checkpoint` points at the deleted r2 checkpoint. It's only read
  when r3 is started fresh, which won't happen.

## Not tested

- No training run, eval, recording or ygoinf server was started. `tools/ygo.sh start`, `eval`,
  `record` and `tools/start_inf_server.sh` were checked with `bash -n` and stubs only.
- How long an eval takes on the CPU, and how much a CPU eval slows the env threads of a running
  training.
- `DEVICE` auto with no training running (the run was active the whole session).
- `tools/check_finite.py`, `tools/test_embedding_rows.py`, `tools/deck_order_probe.py` and
  `tools/run_checks.sh` on the promoted checkpoint. They need the env, which isn't built in this
  worktree, and CPU time the training run uses.
- `tools/prune_checkpoints.sh` with `--delete`, and its dry run on `ft_obs_v2_r1`.
- Which process holds `cards.cdb-wal` open.
- A real client with the version 2 model.
