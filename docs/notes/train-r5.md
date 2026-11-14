# train-r5: host memory growth in training

Branch `train-r5`, based on `new-cards` (`9307f77`). Date: 2026-11-14.

## Summary

- **Cause:** nothing is leaked. Live memory stays flat. The trainer's RSS grows because
  glibc malloc keeps memory freed by the env threads in its per-thread arenas. The
  allocator returns memory to the OS only from the top of a heap, and the free space
  sits in the middle of the heaps. The 400 to 1,200 env threads create and destroy
  duels (Lua states, card objects) all the time, so the free, still-resident part of
  the arenas grows without bound. That killed the r5 trainer (pid 86897) after 2 h 44 min
  at 15.0 GB anon RSS.
- **Fix:** `scripts/cleanba.py` starts a daemon thread that calls `malloc_trim(0)` every
  10 s (`--malloc_trim_interval`, 0 turns it off). That call returns the free pages of
  all arenas to the OS. It only changes RSS. It doesn't change what the program
  computes.
- **Result:** on r5's layout on the GPU over 12 minutes, RSS without the fix went from
  4.48 GB to 5.67 GB between 131 s and 729 s. With the fix it stayed between 3.60 and
  3.76 GB over the same interval. SPS was 4,251 without the fix and 4,254 with it.
- **Determinism digests:** identical before and after (the change doesn't touch the env).

## Evidence

### 1. Shape of r5's rss.log

The restarted trainer (pid 115498) grew 17 to 335 MB per minute with no pattern. Some
minutes added 1.6 GB per million steps, others 0.08 GB. Twice, at 12:18:58 and
12:34:16, RSS fell by 1.12 and 1.23 GB with no restart and no swap use (`VmSwap: 0 kB`).

The steps don't line up with the checkpoint saves (12:11:45, 12:21:16, 12:31:03) or with
the eval interval (about every 45 s). That fits an allocator that sometimes frees a heap
top. It doesn't fit Python objects that pile up per iteration, eval, or checkpoint.

### 2. Env only, no model (64 envs, random actions, 10 minutes)

Measured with `mallinfo2()` from inside the process. "In use" is memory the program
holds. "Free in arenas" is memory glibc keeps for reuse.

| Time | RSS | In use | Free in arenas | Steps | Games |
|---|---|---|---|---|---|
| 10 s | 617 MB | 312 MB | 269 MB | 52,288 | 137 |
| 308 s | 2,551 MB | 392 MB | 2,123 MB | 1,484,544 | 4,609 |
| 607 s | 3,695 MB | 314 MB | 3,345 MB | 2,810,112 | 8,715 |

At the end, one `malloc_trim(0)` call brought RSS from 3,695,296 kB down to 736,576 kB.

`malloc_info` (150 s run) showed 96 arenas, the glibc cap of 8 x 12 cores. Single arenas
were 95 MB, with 93 MB of it free. 705 MB of the free space was in 415 chunks between
0.5 and 1.7 MB.

Variants, all 64 envs:

| Variant | RSS at about 110 s | RSS at about 180 s |
|---|---|---|
| Default | 1,652 MB | 1,975 MB |
| `MALLOC_MMAP_THRESHOLD_=131072` (no dynamic threshold) | 1,553 MB | 1,745 MB |
| `MALLOC_ARENA_MAX=1024` (one arena per thread) | 1,940 MB | 2,498 MB |
| `malloc_trim(0)` every 10 s | 510 MB | 523 MB (607 MB at 307 s) |

Large blocks aren't the cause (the fixed mmap threshold makes no difference). Arenas
shared between threads aren't the cause either: one arena per thread is worse. With trim
every 10 s, steps per second were the same as without it. Each trim took 94 ms on
average (131 ms at most), on its own thread.

### 3. Trainer (`cleanba.py --debug`)

**On the CPU** (32 envs x 2 actor threads x 8 steps, 8 minibatches, learner on the CPU,
514 s, about 120 SPS): after the JIT warm-up, in use stayed flat (1.27 to 1.28 GB from
200 s to 514 s). Free in arenas grew from 0.33 to 0.66 GB. The CPU learner is too slow
for a long curve. The first try at 32 x 2 x 32 stopped itself after 121 s because its XLA
buffers took 4.5 GB.

**On the GPU, r5's layout** (with r5 stopped: 64 envs x 2 actor threads x 32 steps,
`--timeout 600`, eval every 50 iterations with 32 episodes, XLA memory fraction 0.65,
promoted model, `--debug`, 1,178 threads, 12 minutes). The first run used
`--malloc_trim_interval 0`, which runs the code as it was before the fix.

| Time | RSS before | In use before | Free before | RSS after | In use after |
|---|---|---|---|---|---|
| 10 s | 1,378 MB | 365 MB | 9 MB | 1,323 MB | 320 MB |
| 131 s | 4,478 MB | 1,677 MB | 1,424 MB | 3,628 MB | 1,649 MB |
| 314 s | 5,131 MB | 1,650 MB | 2,104 MB | 3,758 MB | 1,665 MB |
| 557 s | 5,474 MB | 1,649 MB | 2,449 MB | 3,603 MB | 1,615 MB |
| 729 s | 5,665 MB | 1,666 MB | 2,623 MB | 3,674 MB | 1,688 MB |

- Steps: 2,744,320 before, 2,703,360 after (the last logged global_step of each run).
- SPS: 4,251 before, 4,254 after.
- Without the fix, RSS grew 1.19 GB from 131 s to 729 s (about 2.3M steps). With the
  fix, it grew 0.05 GB.
- In use is the same in both runs. With the fix, "free in arenas" still grows in the
  `mallinfo2` numbers, because trimmed pages are still counted as free. They're just no
  longer resident.

### 4. r4 compared with r5

r4 (env at `15c7726`, 16 envs x 2 actor threads x 128 steps) ran 12 hours and about
92M steps on the same 15 GiB host. Its boot's kernel log (`journalctl -b -2 -k`) has no
OOM kill. r4 had 32 active duel threads and r5 has 128, plus twice the SPS (2,200 against
4,300). So r4 either grew more slowly or leveled off below the limit. Its logs have no
RSS samples, so I can't tell which.

The audit commits between `15c7726` and `9307f77` don't change how memory is held: the
env-only test shows that in-use memory doesn't grow. The layout is the likely
difference: more threads churning duels at twice the rate.

## Suspects ruled out

- **Env per-game state:** `revealed_` isn't cleared in `reset()` (AUDIT C9), but it's a
  set of position strings with a few dozen entries at most. The replay buffer is only
  filled with `--record`, and it's cleared at game end and in `reset()`. `crash_info_` is
  overwritten every reset, and the crash slots are a fixed 4096 x 384 byte table. Duels
  are freed through `ygo_registry::end_duel` in `reset()`. The Lua message handler
  allocates nothing that persists. All of this is consistent with in-use memory staying
  at 280 to 390 MB over 8,715 games.
- **Python side:** no list grows per iteration. The deques are bounded, `init_rstates`
  gets one append and one pop per iteration, and the storage is rebuilt each iteration.
  The TensorBoard writer is a dummy in `--debug`, so it wasn't measured. In-use memory in
  the trainer is flat, so if the writer grows at all, it's small. The non-finite counter
  is an int.

## Determinism digests

`tools/dump_trajectory.py --games 32 --clamp-level 13`:

| Build | decks/mine | decks/train |
|---|---|---|
| `new-cards` (`~/ygo/ygo-agent`, `9307f77`) | `2b0c259a5526dc607556d957de7fc90ddc11443f1549e0f2564e2f7751e24adb` | `85a2ab9c25429f416299d73578b0484b314136017f36cbb686a956bc4688f0b9` |
| `train-r5` (this worktree's build) | `2b0c259a5526dc607556d957de7fc90ddc11443f1549e0f2564e2f7751e24adb` | `85a2ab9c25429f416299d73578b0484b314136017f36cbb686a956bc4688f0b9` |

The fix is in `cleanba.py`, which `dump_trajectory.py` doesn't import, so the digests
can't change. The table confirms that this worktree's env build matches `new-cards`.

## Not tested

- A full-length run with the fix. The longest test was 12 minutes (2.7M steps). Over that
  time, trimmed RSS stayed between 3.60 and 3.76 GB. Whether it stays flat over hours,
  for example because pages that are only partly free can't be returned, needs r5's
  `rss.log` after the resume with this commit.
- The TensorBoard writer and checkpoint saving under the fix. `--debug` turns both off.
- `--timeout 0` (inline stepping), eval with `--eval_checkpoint`, and multi-device or
  distributed setups.
- Other allocators (jemalloc, tcmalloc through `LD_PRELOAD`). They would need a package
  install, which this session isn't allowed to do.
- The exact allocation pattern inside ygopro-core and Lua that leaves the free chunks in
  the middle of the heaps.
- The env-only tests imported the `~/ygo/ygo-agent` env build instead of this worktree's
  build (the tmux environment problem below). Both builds are from `9307f77`, and their
  digests are identical.

## Incident during the session

Between 12:44 and 13:00, two trainer tests that were supposed to run on the CPU ran on
the GPU next to r5. Cause: a new tmux session gets the tmux server's environment, so the
exported `JAX_PLATFORMS=cpu` and `CUDA_VISIBLE_DEVICES=` never reached the process. While
the tests ran, r5's train_time went from about 0.65 s to about 7 s per iteration. Its
logged SPS fell from about 4,200 to about 3,590. r5 kept running (same pid) and recovered
when the tests stopped. Later tests set the variables inside the session command and
checked `/proc/<pid>/environ` before going on.

## Resuming r5 with the fix

The trim thread is on by default, so a resume from this branch needs no new flag. To get
the old behavior, pass `--malloc_trim_interval 0`.
