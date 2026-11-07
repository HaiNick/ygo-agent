"""Resumable voyage-2 card embedding over the Voyage REST API (no SDK needed).

Text per card is exactly scripts/card/embedding.py's: ygoai.embed.read_cards(cdb)
and Card.format(). The request matches voyageai.Client().embed(texts,
model="voyage-2", truncation=False). Unlike scripts/card/embedding.py this tool
never writes code_list.txt.

Store: a pickle dict {code: float32[1024]} (same format as embed864.pkl), written
atomically after every batch, plus <store>.meta.json with token usage.

  python tools/voyage_embed.py test CODE [CODE ...]      cosine vs --ref vectors
  python tools/voyage_embed.py estimate                  codes in code_list without a vector
  python tools/voyage_embed.py run                       embed them (resumable)

Rate limits default to 3 requests/min and 10,000 tokens/min; batches are capped
at ~3,000 estimated tokens. HTTP 429 and 5xx are retried with backoff. The API
key is read from VOYAGE_API_KEY and is never printed.
"""
import argparse, collections, json, os, pickle, sys, time, urllib.error, urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ygoai.embed import read_cards  # noqa: E402

API = "https://api.voyageai.com/v1/embeddings"


def load_texts(cdb, codes):
    _, cards = read_cards(cdb)
    c2c = {c.code: c for c in cards}
    texts, missing = {}, []
    for code in codes:
        if code in c2c:
            texts[code] = c2c[code].format()
        else:
            missing.append(code)
    return texts, missing


def read_code_list(path):
    return [int(l.split()[0]) for l in open(path) if l.strip()]


class Store:
    def __init__(self, path, seed=None):
        self.path = Path(path)
        self.meta_path = self.path.with_suffix(self.path.suffix + ".meta.json")
        if self.path.exists():
            self.vectors = pickle.load(open(self.path, "rb"))
        else:
            self.vectors = {}
            if seed:
                self.vectors.update({int(k): np.asarray(v, np.float32) for k, v in pickle.load(open(seed, "rb")).items()})
        self.meta = json.load(open(self.meta_path)) if self.meta_path.exists() else {
            "model": "voyage-2", "chars": 0, "tokens": 0, "requests": 0}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.vectors, f)
        os.replace(tmp, self.path)
        tmpm = self.meta_path.with_suffix(".tmp")
        json.dump(self.meta, open(tmpm, "w"), indent=1)
        os.replace(tmpm, self.meta_path)

    def chars_per_token(self, default=3.5):
        return self.meta["chars"] / self.meta["tokens"] if self.meta["tokens"] else default


class Limiter:
    def __init__(self, rpm, tpm):
        self.rpm, self.tpm = rpm, tpm
        self.log = collections.deque()  # (time, tokens)

    def _trim(self):
        while self.log and time.time() - self.log[0][0] >= 60:
            self.log.popleft()

    def wait(self, est_tokens):
        while True:
            self._trim()
            used = sum(t for _, t in self.log)
            if len(self.log) < self.rpm and used + est_tokens <= self.tpm:
                return
            time.sleep(max(0.5, 60 - (time.time() - self.log[0][0]) + 0.5))

    def record(self, tokens):
        self.log.append((time.time(), tokens))


def embed_request(texts, key, max_retries=8):
    body = json.dumps({"input": texts, "model": "voyage-2", "truncation": False}).encode()
    delay = 20.0
    for attempt in range(max_retries):
        req = urllib.request.Request(API, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.load(r)
            data = sorted(out["data"], key=lambda d: d["index"])
            return [d["embedding"] for d in data], int(out["usage"]["total_tokens"])
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                ra = e.headers.get("Retry-After")
                wait = float(ra) if ra and ra.replace(".", "").isdigit() else delay
                print(f"  HTTP {e.code}, retry {attempt + 1}/{max_retries} in {wait:.0f}s", flush=True)
                time.sleep(wait)
                delay = min(delay * 2, 300)
                continue
            msg = e.read().decode(errors="replace")[:300]
            raise SystemExit(f"HTTP {e.code}: {msg}")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  network error ({e}), retry in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300)
    raise SystemExit("too many retries")


def batches(codes, texts, cpt, max_tokens, max_items=128):
    cur, cur_tok = [], 0
    for c in codes:
        t = len(texts[c]) / cpt + 4
        if cur and (cur_tok + t > max_tokens or len(cur) >= max_items):
            yield cur, cur_tok
            cur, cur_tok = [], 0
        cur.append(c)
        cur_tok += t
    if cur:
        yield cur, cur_tok


def get_key():
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        raise SystemExit("VOYAGE_API_KEY is not set in this shell")
    return key


def embed_codes(store, codes, texts, args):
    key = get_key()
    lim = Limiter(args.rpm, args.tpm)
    todo = [c for c in codes if c not in store.vectors]
    done0, t0 = len(codes) - len(todo), time.time()
    n = done0
    for batch, est in batches(todo, texts, store.chars_per_token(), args.max_batch_tokens):
        lim.wait(est)
        vecs, tokens = embed_request([texts[c] for c in batch], key)
        lim.record(tokens)
        for c, v in zip(batch, vecs):
            store.vectors[c] = np.asarray(v, np.float32)
        store.meta["chars"] += sum(len(texts[c]) for c in batch)
        store.meta["tokens"] += tokens
        store.meta["requests"] += 1
        store.save()
        n += len(batch)
        el = time.time() - t0
        rate = (n - done0) / el if el > 0 else 0
        eta = (len(codes) - n) / rate / 60 if rate else float("nan")
        print(f"  {n}/{len(codes)} cards, batch {len(batch)} cards / {tokens} tokens, "
              f"total {store.meta['tokens']} tokens, ETA {eta:.0f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["test", "estimate", "run"])
    ap.add_argument("codes", nargs="*", type=int)
    ap.add_argument("--cdb", default=str(ROOT / "assets/locale/en/cards.cdb"))
    ap.add_argument("--code-list", default=str(ROOT / "scripts/code_list.txt"))
    ap.add_argument("--ref", default=str(ROOT / "scripts/checkpoints/embed864.pkl"),
                    help="existing voyage-2 vectors; seeds the store and is the reference for test")
    ap.add_argument("--store", default=os.path.expanduser("~/ygo/embeddings/voyage2.pkl"))
    ap.add_argument("--rpm", type=int, default=3)
    ap.add_argument("--tpm", type=int, default=10000)
    ap.add_argument("--max-batch-tokens", type=int, default=3000)
    ap.add_argument("--calib", type=int, default=0,
                    help="test mode: also embed N missing codes in the same request to measure chars/token")
    args = ap.parse_args()

    store = Store(args.store, seed=args.ref)
    ref = pickle.load(open(args.ref, "rb"))

    if args.mode == "test":
        code_list = read_code_list(args.code_list)
        calib = [c for c in code_list if c not in store.vectors][: args.calib]
        texts, missing = load_texts(args.cdb, args.codes + calib)
        if missing:
            raise SystemExit(f"codes not in cdb: {missing}")
        vecs, tokens = embed_request([texts[c] for c in args.codes + calib], get_key())
        for c, v in zip(args.codes, vecs):
            v = np.asarray(v, np.float32)
            r = np.asarray(ref[c], np.float32)
            cos = float(v @ r / np.linalg.norm(v) / np.linalg.norm(r))
            print(f"{c}: cosine={cos:.6f}  |new|={np.linalg.norm(v):.4f}  text={texts[c][:70]!r}")
        for c, v in zip(calib, vecs[len(args.codes):]):
            store.vectors[c] = np.asarray(v, np.float32)
        store.meta["chars"] += sum(len(texts[c]) for c in args.codes + calib)
        store.meta["tokens"] += tokens
        store.meta["requests"] += 1
        store.save()
        print(f"request tokens: {tokens}; measured chars/token so far: {store.chars_per_token():.3f}")
        return

    code_list = read_code_list(args.code_list)
    todo = [c for c in code_list if c not in store.vectors]
    texts, missing = load_texts(args.cdb, todo)
    todo = [c for c in todo if c in texts]
    if args.mode == "estimate":
        cpt = store.chars_per_token()
        chars = sum(len(texts[c]) for c in todo)
        tokens = chars / cpt + 4 * len(todo)
        nb = sum(1 for _ in batches(todo, texts, cpt, args.max_batch_tokens))
        minutes = max(nb / args.rpm, tokens / (args.tpm * 0.95))
        print(f"code_list: {len(code_list)}  have vectors: {len(code_list) - len(todo) - len(missing)}")
        print(f"to embed: {len(todo)}  not in cdb (skipped): {len(missing)} {missing[:10]}")
        print(f"chars: {chars}  chars/token: {cpt:.3f} (from {store.meta['tokens']} measured tokens)")
        print(f"estimated tokens: {tokens:,.0f}  requests: {nb}  duration: {minutes:.0f} min ({minutes / 60:.1f} h)")
        return

    print(f"embedding {len(todo)} cards into {args.store}")
    embed_codes(store, todo, texts, args)
    print("done:", len(store.vectors), "vectors in store")


if __name__ == "__main__":
    main()
