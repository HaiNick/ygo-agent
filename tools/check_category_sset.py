"""Check that the core keeps 64-bit effect categories (CATEGORY_SSET = 0x200000000).

Core 0.0.2 stored categories in 32 bits and dropped CATEGORY_SSET; core 0.0.3
stores them in 64 bits. This runs Lua inside a real core duel via the
core_lua_probe binary (xmake b core_lua_probe):

  1. generic: Effect.GlobalEffect() + SetCategory(CATEGORY_SSET), read back
     with GetCategory() and IsHasCategory()
  2. card check: for every card in the given decks whose script calls
     SetCategory(...) with CATEGORY_SSET, evaluate that exact expression and
     check the bit survives

usage: python tools/check_category_sset.py [DECK_DIR ...]   (default ~/ygo/decks/mine)
exit code 0 if every check passes.
"""
import glob, os, re, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "script"
PROBE = ROOT / "build" / "linux" / "x86_64" / "release" / "core_lua_probe"

deck_dirs = sys.argv[1:] or [os.path.expanduser("~/ygo/decks/mine")]
codes = set()
for d in deck_dirs:
    for p in glob.glob(os.path.join(d, "*.ydk")):
        codes |= {int(l) for l in map(str.strip, open(p)) if l.isdigit()}

lua = [
    "local e=Effect.GlobalEffect()",
    "e:SetCategory(CATEGORY_SSET)",
    "Debug.Message('generic get '..tostring(e:GetCategory()))",
    "Debug.Message('generic has '..tostring(e:IsHasCategory(CATEGORY_SSET)))",
]
cards = []
for code in sorted(codes):
    p = SCRIPTS / f"c{code}.lua"
    if not p.exists():
        continue
    txt = re.sub(r"--[^\n]*", "", p.read_text(errors="ignore"))
    for i, expr in enumerate(re.findall(r"SetCategory\(([^()]*CATEGORY_SSET[^()]*)\)", txt)):
        tag = f"{code}#{i}"
        cards.append((tag, expr.strip()))
        lua += [
            "do local e=Effect.GlobalEffect()",
            f"  e:SetCategory({expr})",
            f"  Debug.Message('card {tag} '..tostring(e:IsHasCategory(CATEGORY_SSET)))",
            "end",
        ]

with tempfile.NamedTemporaryFile("w", suffix=".lua", delete=False) as f:
    f.write("\n".join(lua) + "\n")
    lua_path = f.name

if not PROBE.exists():
    sys.exit(f"probe binary not found: {PROBE} (run: xmake b core_lua_probe)")
r = subprocess.run([str(PROBE), str(SCRIPTS), lua_path], capture_output=True, text=True)
os.unlink(lua_path)
out = dict(line.rsplit(" ", 1) for line in r.stdout.splitlines() if " " in line)
if r.stderr.strip():
    print(r.stderr.strip())

ok = True
def check(name, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name}: {got} (want {want})")

check("GlobalEffect GetCategory", out.get("generic get"), str(0x200000000))
check("GlobalEffect IsHasCategory", out.get("generic has"), "true")
for tag, expr in cards:
    check(f"card {tag} SetCategory({expr})", out.get(f"card {tag}"), "true")
print(f"{len(cards)} card expressions from {len({t.split('#')[0] for t, _ in cards})} cards checked")
sys.exit(0 if ok and r.returncode == 0 else 1)
