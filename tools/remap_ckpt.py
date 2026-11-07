"""Re-index a v0.1 checkpoint's card-ID embedding table to main's code_list.txt.

v0.1 card IDs = 1-based line numbers in the 864-card list (dbf5142^:scripts/code_list.txt).
main card IDs = 1-based line numbers in main's 13472-card list.
Row 0 is the "unknown" embedding and is used for cards the checkpoint never saw.

usage: python remap_ckpt.py OLD_LIST NEW_LIST IN_CKPT OUT_CKPT
"""
import sys
import numpy as np
from flax import serialization

old_list, new_list, src, dst = sys.argv[1:5]

def codes(path):
    with open(path) as f:
        return [int(line.split()[0]) for line in f if line.strip()]

old_codes, new_codes = codes(old_list), codes(new_list)
old_id = {c: i + 1 for i, c in enumerate(old_codes)}

with open(src, "rb") as f:
    state = serialization.msgpack_restore(f.read())

emb_parent = state["params"]["Encoder_0"]["Embed_0"]
old_emb = np.asarray(emb_parent["embedding"])
print("old embedding:", old_emb.shape, "old list:", len(old_codes), "new list:", len(new_codes))
assert old_emb.shape[0] >= len(old_codes) + 1

new_emb = np.repeat(old_emb[0:1], len(new_codes) + 1, axis=0)
hits = 0
for j, c in enumerate(new_codes, start=1):
    i = old_id.get(c)
    if i is not None:
        new_emb[j] = old_emb[i]
        hits += 1
emb_parent["embedding"] = new_emb.astype(old_emb.dtype)
print(f"new embedding: {new_emb.shape}, cards carried over: {hits}/{len(old_codes)}")

with open(dst, "wb") as f:
    f.write(serialization.msgpack_serialize(state))
print("wrote", dst)
