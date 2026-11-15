"""Prepare a checkpoint trained on observation version 1 for observation version 2.

One change (docs/OBS_V2.md, E1): the counter feature of a card row. Version 1 wrote 15 for any
card with counters, so of the 16 rows of the counter embedding table
(params/Encoder_0/CardEncoder_0/Embed_6/embedding) only rows 0 and 15 were ever trained.
Version 2 writes the real count, clamped to 15. This script copies row 15 into rows 1 to 14.
A card with counters then gives the model exactly what it gave before, whatever the count,
and the fine-tune can pull the rows apart.

No array changes its shape. Nothing else in the checkpoint is touched, so the outputs on
version 1 observations (counter feature 0 or 15) are unchanged: check with
tools/compare_ckpt_outputs.py.

The output is written to a temporary file next to the destination and renamed. An existing
destination is not overwritten unless --force is given.

usage: python tools/obs_v2_ckpt.py IN_CKPT OUT_CKPT [--force]
"""
import argparse, os, sys, tempfile

import numpy as np
from flax import serialization

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--force", action="store_true", help="overwrite an existing OUT_CKPT")
a = ap.parse_args()
src = os.path.abspath(os.path.expanduser(a.src))
dst = os.path.abspath(os.path.expanduser(a.dst))
if src == dst:
    sys.exit("IN_CKPT and OUT_CKPT are the same file")
if os.path.exists(dst) and not a.force:
    sys.exit(f"{dst} exists (use --force to overwrite)")

with open(src, "rb") as f:
    state = serialization.msgpack_restore(f.read())

parent = state["params"]["Encoder_0"]["CardEncoder_0"]["Embed_6"]
table = np.asarray(parent["embedding"])
if table.ndim != 2 or table.shape[0] != 16:
    sys.exit(f"unexpected counter table shape {table.shape}, expected (16, n)")
new = table.copy()
new[1:15] = table[15]
before = np.abs(table[1:15] - table[15]).max()
parent["embedding"] = new.astype(table.dtype)
print(f"counter table {table.shape} {table.dtype}: rows 1 to 14 set to row 15 "
      f"(largest difference to row 15 before: {before:.4g}), rows 0 and 15 unchanged: "
      f"{bool((new[0] == table[0]).all() and (new[15] == table[15]).all())}")

data = serialization.msgpack_serialize(state)
fd, tmp = tempfile.mkstemp(prefix=os.path.basename(dst) + ".", suffix=".tmp", dir=os.path.dirname(dst))
try:
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, os.stat(src).st_mode & 0o777)  # mkstemp creates the file with mode 600
    os.replace(tmp, dst)
except BaseException:
    if os.path.exists(tmp):
        os.unlink(tmp)
    raise
print(f"wrote {dst} ({len(data)} bytes, input {os.path.getsize(src)} bytes)")
