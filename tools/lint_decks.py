"""Static check of deck files against what the env can handle.

Scans the Lua script of every card in the given decks for core functions whose
messages the env cannot answer (it throws, which ends the process) or that
ygoinf does not support. A hit means "this card can reach that code path", not
that it always will.

usage: python tools/lint_decks.py DECK_DIR_OR_YDK [...] [--strict]
exit code: 0, or 1 with --strict if any card is in the "throws" class.
"""
import argparse, re, sqlite3, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# function -> (class, what happens)
RULES = {
    "AnnounceRace": ("throws", "MSG_ANNOUNCE_RACE has no handler"),
    "RockPaperScissors": ("throws", "MSG_ROCK_PAPER_SCISSORS has no handler"),
    "AnnounceLevel": ("may throw", "announce_number throws for values above 12"),
    "AnnounceNumber": ("may throw", "announce_number throws for values above 12"),
    "AnnounceAttribute": ("may throw", "announce_attrib throws when more than 1 attribute is declared"),
    "SelectDisableField": ("may throw", "select_disfield throws when more than 1 zone is selected"),
    "RemoveCounter": ("may throw", "select_counter throws for more than 2 candidate cards"),
    "AnnounceCard": ("ygoinf", "announce_card works in the env, not in ygoinf"),
    "SortDecktop": ("info", "sort_card is answered with the default order"),
}
p = argparse.ArgumentParser()
p.add_argument("decks", nargs="+")
p.add_argument("--strict", action="store_true")
p.add_argument("--scripts", default=str(ROOT / "scripts" / "script"))
p.add_argument("--cdb", default=str(ROOT / "assets" / "locale" / "en" / "cards.cdb"))
args = p.parse_args()

files = []
for d in args.decks:
    d = Path(d).expanduser()
    files += sorted(d.glob("*.ydk")) if d.is_dir() else [d]
codes = {}
for f in files:
    for line in f.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.isdigit():
            codes.setdefault(int(line), set()).add(f.stem)
        elif line and not line.startswith(("#", "!")):
            print(f"warning: {f.name}: line {line!r} is not a card code (the env skips it)")
con = sqlite3.connect(f"file:{args.cdb}?mode=ro", uri=True)
name = lambda c: (con.execute("select name from texts where id=?", (c,)).fetchone() or ["?"])[0]
hits = {}
missing = []
for c in sorted(codes):
    sp = Path(args.scripts) / f"c{c}.lua"
    if not sp.exists():
        missing.append(c)
        continue
    text = sp.read_text(encoding="utf-8", errors="replace")
    for fn in RULES:
        if re.search(r"\b" + fn + r"\b", text):
            hits.setdefault(fn, []).append(c)
print(f"{len(files)} deck files, {len(codes)} unique codes, {len(missing)} without a script file (normal monsters)")
bad = 0
for fn, (cls, why) in RULES.items():
    for c in hits.get(fn, []):
        bad += cls == "throws"
        decks = sorted(codes[c])
        print(f"[{cls:9s}] {fn:18s} {c:>9d} {name(c)} ({len(decks)} decks: {', '.join(decks[:3])}{'...' if len(decks) > 3 else ''}) - {why}")
sys.exit(1 if (args.strict and bad) else 0)
