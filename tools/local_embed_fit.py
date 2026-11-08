"""Fit a linear map from a local embedding model to voyage-2 card vectors.

Runs in the GPU venv (~/ygo/.venv-embed). Card text is exactly
scripts/card/embedding.py's (ygoai.embed read_cards + Card.format).

Pairs: every code with a voyage-2 vector in --vectors, excluding --exclude
(default embed864.pkl: 556 of those 864 vectors were made from older card text
than the current database, so their text no longer matches the vector).

Split 80/20 (fixed seed). Ridge regression local -> voyage with an intercept;
lambda chosen on a validation split of the 80%. Reports held-out cosine
(prediction normalised, as voyage vectors are unit length) and top-1 retrieval
accuracy among the held-out set.

  python tools/local_embed_fit.py --model BAAI/bge-large-en-v1.5 [--model ...]
  writes ~/ygo/embeddings/local_map_<model>.npz (W, b, lambda, metrics)
"""
import argparse, json, os, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from ygoai.embed import read_cards  # noqa: E402

EMB = Path(os.path.expanduser("~/ygo/embeddings"))


def card_texts(cdb, codes):
    _, cards = read_cards(cdb)
    c2c = {c.code: c for c in cards}
    return {c: c2c[c].format() for c in codes if c in c2c}


def encode(model_name, texts, batch_size, cache):
    cache_file = EMB / f"local_{model_name.replace('/', '__')}.npz"
    have = {}
    if cache and cache_file.exists():
        z = np.load(cache_file)
        have = dict(zip(z["codes"].tolist(), z["vecs"]))
    todo = [c for c in texts if c not in have]
    if todo:
        import torch
        from sentence_transformers import SentenceTransformer
        kw = {"trust_remote_code": True}
        m = SentenceTransformer(model_name, device="cuda", model_kwargs={"torch_dtype": torch.float16}, **kw)
        t0 = time.time()
        vecs = m.encode([texts[c] for c in todo], batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)
        print(f"  encoded {len(todo)} cards in {time.time() - t0:.0f}s, dim {vecs.shape[1]}, "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        have.update(zip(todo, vecs.astype(np.float32)))
        codes = np.array(list(have))
        np.savez(cache_file, codes=codes, vecs=np.stack([have[c] for c in codes]))
        del m
        torch.cuda.empty_cache()
    return {c: have[c] for c in texts}


def ridge(X, Y, lam):
    Xa = np.hstack([X, np.ones((len(X), 1), X.dtype)])
    reg = lam * np.eye(Xa.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0  # no penalty on the intercept
    Wb = np.linalg.solve(Xa.T.astype(np.float64) @ Xa + reg, Xa.T.astype(np.float64) @ Y)
    return Wb[:-1].astype(np.float32), Wb[-1].astype(np.float32)


def predict(X, W, b):
    P = X @ W + b
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def metrics(P, Y):
    Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    cos = (P * Yn).sum(1)
    top1 = float((np.argmax(P @ Yn.T, axis=1) == np.arange(len(P))).mean())
    return {"cos_mean": float(cos.mean()), "cos_median": float(np.median(cos)),
            "cos_p5": float(np.percentile(cos, 5)), "top1": top1}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--cdb", default=str(ROOT / "assets/locale/en/cards.cdb"))
    ap.add_argument("--vectors", default=str(EMB / "voyage2.pkl"))
    ap.add_argument("--exclude", default=str(ROOT / "scripts/checkpoints/embed864.pkl"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    voy = pickle.load(open(args.vectors, "rb"))
    excl = set(pickle.load(open(args.exclude, "rb"))) if args.exclude else set()
    codes = sorted(c for c in voy if c not in excl)
    texts = card_texts(args.cdb, codes)
    codes = [c for c in codes if c in texts]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(codes))
    n_test = len(codes) // 5
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    n_val = len(train_idx) // 8
    val_idx, fit_idx = train_idx[:n_val], train_idx[n_val:]
    Y = np.stack([np.asarray(voy[c], np.float32) for c in codes])
    print(f"pairs: {len(codes)} (train {len(train_idx)}, held-out {n_test}); excluded {len(excl)}")

    results = {}
    for name in args.model:
        print(f"== {name}")
        loc = encode(name, {c: texts[c] for c in codes}, args.batch_size, cache=True)
        X = np.stack([loc[c] for c in codes])
        best = None
        for lam in [0.01, 0.1, 1.0, 3.0, 10.0, 30.0]:
            W, b = ridge(X[fit_idx], Y[fit_idx], lam)
            m = metrics(predict(X[val_idx], W, b), Y[val_idx])
            if best is None or m["cos_mean"] > best[1]["cos_mean"]:
                best = (lam, m)
        lam = best[0]
        W, b = ridge(X[train_idx], Y[train_idx], lam)
        m = metrics(predict(X[test_idx], W, b), Y[test_idx])
        print(f"  lambda={lam}  held-out cosine mean {m['cos_mean']:.4f} median {m['cos_median']:.4f} "
              f"p5 {m['cos_p5']:.4f}  top-1 retrieval {m['top1']:.3f}")
        out = EMB / f"local_map_{name.replace('/', '__')}.npz"
        np.savez(out, W=W, b=b, lam=lam, model=name, metrics=json.dumps(m),
                 n_train=len(train_idx), n_test=n_test, dim=X.shape[1])
        results[name] = dict(m, lam=lam, dim=int(X.shape[1]))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
