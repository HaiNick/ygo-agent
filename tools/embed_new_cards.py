"""Embed new card codes locally (no API) and append them through the fitted map.

Runs in the GPU venv (~/ygo/.venv-embed). Uses the local model and ridge map
written by tools/local_embed_fit.py to produce voyage-2-compatible vectors.

Steps:
  1. --add-decks DIR...  append deck codes missing from code_list.txt
     (append only; "code has_script" lines; existing lines never change)
  2. every code in code_list.txt without a voyage-2 vector and without a mapped
     vector is embedded locally: text = ygoai.embed Card.format() (same template
     as scripts/card/embedding.py), vector = normalise(local @ W + b)
  3. mapped vectors are stored separately (default ~/ygo/embeddings/local_mapped.pkl)
     so the voyage-2 file stays pure; a merged file (voyage preferred) is written
     for checkpoint extension
  4. --ckpt IN --out OUT  extend the checkpoint table with the merged vectors via
     tools/extend_embeddings.py in the training venv (existing rows unchanged)

usage:
  ~/ygo/.venv-embed/bin/python tools/embed_new_cards.py --map ~/ygo/embeddings/local_map_<model>.npz \\
      [--add-decks ~/ygo/decks/incoming] [--ckpt IN.flax_model --out OUT.flax_model]
"""
import argparse, glob, json, os, pickle, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from ygoai.embed import read_cards  # noqa: E402

EMB = Path(os.path.expanduser("~/ygo/embeddings"))


def atomic_pickle(obj, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="npz from tools/local_embed_fit.py")
    ap.add_argument("--code-list", default=str(ROOT / "scripts/code_list.txt"))
    ap.add_argument("--cdb", default=str(ROOT / "assets/locale/en/cards.cdb"))
    ap.add_argument("--voyage", default=str(EMB / "voyage2.pkl"))
    ap.add_argument("--mapped", default=str(EMB / "local_mapped.pkl"))
    ap.add_argument("--merged", default=str(EMB / "all_vectors.pkl"))
    ap.add_argument("--add-decks", nargs="*", default=[])
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--train-python", default=os.path.expanduser("~/ygo/.venv/bin/python"))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    _, cards = read_cards(args.cdb)
    c2c = {c.code: c for c in cards}
    code_list = [int(l.split()[0]) for l in open(args.code_list) if l.strip()]
    known = set(code_list)

    # 1. append new deck codes (append only)
    if args.add_decks:
        new = []
        for d in args.add_decks:
            for p in sorted(glob.glob(os.path.join(d, "*.ydk"))):
                for l in open(p):
                    l = l.strip()
                    if l.isdigit() and int(l) not in known:
                        known.add(int(l))
                        new.append(int(l))
        bad = [c for c in new if c not in c2c]
        if bad:
            sys.exit(f"codes not in the card database, nothing appended: {bad}")
        if new:
            raw = open(args.code_list, "rb").read()
            with open(args.code_list, "a") as f:
                if raw and not raw.endswith(b"\n"):
                    f.write("\n")
                for c in new:
                    has_script = int((ROOT / "scripts/script" / f"c{c}.lua").exists())
                    f.write(f"{c} {has_script}\n")
            code_list += new
        print(f"appended {len(new)} new codes to {args.code_list}")

    # 2. embed codes that have no vector yet
    voyage = pickle.load(open(args.voyage, "rb")) if os.path.exists(args.voyage) else {}
    mapped = pickle.load(open(args.mapped, "rb")) if os.path.exists(args.mapped) else {}
    todo = [c for c in code_list if c not in voyage and c not in mapped]
    missing_text = [c for c in todo if c not in c2c]
    todo = [c for c in todo if c in c2c]
    z = np.load(args.map, allow_pickle=False)
    model_name = str(z["model"])
    W, b = z["W"], z["b"]
    print(f"map: {model_name}, held-out metrics {z['metrics']}")
    print(f"to embed locally: {len(todo)}; no card text (skipped): {missing_text}")
    if todo:
        import torch
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name, device="cuda", trust_remote_code=True,
                                model_kwargs={"torch_dtype": torch.float16})
        X = m.encode([c2c[c].format() for c in todo], batch_size=args.batch_size,
                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        assert X.shape[1] == W.shape[0], "map was fitted for a different model dimension"
        P = X @ W + b
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        mapped.update({c: P[i] for i, c in enumerate(todo)})
        atomic_pickle(mapped, args.mapped)
        json.dump({"model": model_name, "map": os.path.abspath(args.map), "n": len(mapped)},
                  open(str(args.mapped) + ".meta.json", "w"), indent=1)

    # 3. merged vectors, voyage preferred
    merged = dict(mapped)
    merged.update(voyage)
    atomic_pickle(merged, args.merged)
    cover = sum(1 for c in code_list if c in merged)
    print(f"merged vectors: {len(merged)} -> {args.merged}; code_list coverage {cover}/{len(code_list)}")

    # 4. extend the checkpoint in the training venv
    if args.ckpt:
        if not args.out:
            sys.exit("--out is required with --ckpt")
        subprocess.run([args.train_python, str(ROOT / "tools/extend_embeddings.py"), args.ckpt, args.out,
                        "--code-list", args.code_list, "--vectors", args.merged], check=True)


if __name__ == "__main__":
    main()
