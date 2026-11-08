"""Extend a checkpoint's card-ID embedding table to cover every code in code_list.txt.

Card id = 1-based line number in code_list.txt; row 0 is the "unknown" vector.
The model looks rows up with jnp.take (flax nn.Embed); ids past the end are NaN. The
table needs len(code_list) + 1 rows, i.e. --num_embeddings len(code_list).

Existing rows are copied unchanged (asserted). Each new row j gets the voyage-2
vector of code_list[j - 1]; codes without a vector get a copy of row 0 and are
reported. Rows that already exist are never modified, including placeholder
rows (copies of row 0) for codes that now have a vector, unless
--fill-placeholders is given.

usage: python tools/extend_embeddings.py IN_CKPT OUT_CKPT
         [--code-list scripts/code_list.txt] [--vectors ~/ygo/embeddings/voyage2.pkl]
"""
import argparse, os, pickle
from pathlib import Path

import numpy as np
from flax import serialization

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--code-list", default=str(ROOT / "scripts/code_list.txt"))
ap.add_argument("--vectors", default=os.path.expanduser("~/ygo/embeddings/voyage2.pkl"))
ap.add_argument("--fill-placeholders", action="store_true",
                help="also overwrite existing placeholder rows (breaks the 'existing rows unchanged' rule)")
args = ap.parse_args()

codes = [int(l.split()[0]) for l in open(args.code_list) if l.strip()]
vectors = pickle.load(open(args.vectors, "rb"))
with open(args.src, "rb") as f:
    state = serialization.msgpack_restore(f.read())
parent = state["params"]["Encoder_0"]["Embed_0"]
old = np.asarray(parent["embedding"])
n_rows = len(codes) + 1
assert old.shape[0] <= n_rows, f"table already has {old.shape[0]} rows"

new = np.empty((n_rows, old.shape[1]), dtype=old.dtype)
new[: old.shape[0]] = old
missing = []
for j in range(old.shape[0], n_rows):
    v = vectors.get(codes[j - 1])
    if v is None:
        new[j] = old[0]
        missing.append(codes[j - 1])
    else:
        new[j] = np.asarray(v, dtype=old.dtype)

filled = 0
if args.fill_placeholders:
    for j in range(1, old.shape[0]):
        if j - 1 < len(codes) and np.array_equal(old[j], old[0]) and codes[j - 1] in vectors:
            new[j] = np.asarray(vectors[codes[j - 1]], dtype=old.dtype)
            filled += 1
else:
    assert np.array_equal(new[: old.shape[0]], old), "existing rows changed"

parent["embedding"] = new
with open(args.dst, "wb") as f:
    f.write(serialization.msgpack_serialize(state))
print(f"rows: {old.shape[0]} -> {n_rows}; new rows with a vector: {n_rows - old.shape[0] - len(missing)}; "
      f"placeholder (no vector): {len(missing)} {missing[:10]}")
if filled:
    print(f"--fill-placeholders: overwrote {filled} existing placeholder rows")
print(f"wrote {args.dst}; use --num_embeddings {n_rows - 1} (agent.py adds 1 for row 0)")
