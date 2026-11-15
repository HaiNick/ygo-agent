"""Card-name declarations (MSG_ANNOUNCE_CARD), ported from the training env.

The env (ygoenv/ygopro/ygopro.h, commit 1ada8e8) offers the model a candidate
list instead of all 15,017 codes. This module reproduces that list exactly:

  1. codes named in the opcodes (`code OPCODE_ISCODE`), in opcode order
  2. codes the opponent revealed this duel, most recent first
  3. codes this player revealed this duel, most recent first
  4. this player's deck list (main and extra), in pool order
  5. every code of the loaded decks, by the number of deck files that
     contain it (descending), then by code
  6. only if nothing else qualifies: code_list.txt order

Only codes the core accepts (is_declarable, a port of ygopro-core 0.0.3
playerop.cpp) and that have a card id are offered, at most `cap` of them.

The env-side references are `is_declarable_env`, `init_module` (pool ranking,
card data) and `announce_card_candidates` in ygopro.h. Keep them in sync.
"""
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

OPCODE_ADD = 0x40000000
OPCODE_SUB = 0x40000001
OPCODE_MUL = 0x40000002
OPCODE_DIV = 0x40000003
OPCODE_AND = 0x40000004
OPCODE_OR = 0x40000005
OPCODE_NEG = 0x40000006
OPCODE_NOT = 0x40000007
OPCODE_ISCODE = 0x40000100
OPCODE_ISSETCARD = 0x40000101
OPCODE_ISTYPE = 0x40000102
OPCODE_ISRACE = 0x40000103
OPCODE_ISATTRIBUTE = 0x40000104

TYPE_TOKEN = 0x4000
SIZE_SETCODE = 16

# ygopro-core card_data.h second_code (cards with two names)
SECOND_CODE = {78734254, 13857930, 1784686, 11082056, 46232525}
# ygopro.h db_query_card_data: cards with more set codes than the column holds
EXTRA_SETCODE = {
    8512558: [0x8f, 0x54, 0x59, 0x82, 0x13a],
    55088578: [0x8f, 0x54, 0x59, 0x82, 0x13a],
}


@dataclass
class CardData:
    code: int
    alias: int = 0
    setcode: List[int] = field(default_factory=list)
    type: int = 0
    level: int = 0
    attack: int = 0
    defense: int = 0
    race: int = 0
    attribute: int = 0
    rule_code: int = 0

    def is_setcode(self, value: int) -> bool:
        for x in self.setcode:
            if not x:
                return False
            if check_setcode(x, value):
                return True
        return False


def check_setcode(setcode: int, value: int) -> bool:
    settype = value & 0x0FFF
    setsubtype = value & 0xF000
    return bool(setcode) and (setcode & 0x0FFF) == settype and (setcode & setsubtype) == setsubtype


def split_setcode(value: int) -> List[int]:
    # ygopro-core write_setcode: non-zero 16-bit groups, low to high
    out = []
    value &= (1 << 64) - 1
    while value:
        if value & 0xFFFF:
            out.append(value & 0xFFFF)
        value >>= 16
    return out[:SIZE_SETCODE]


def _is_alternative(code: int, alias: int) -> bool:
    return bool(alias) and alias < code + 20 and code < alias + 20


def load_card_data(db_path: str, codes: Iterable[int]) -> Dict[int, CardData]:
    """Card data for `codes` the way the env's init_module builds it."""
    codes = list(codes)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = {}
    cur = con.cursor()
    for i in range(0, len(codes), 500):
        chunk = codes[i:i + 500]
        q = ("SELECT id, alias, setcode, type, level, atk, def, race, attribute FROM datas "
             f"WHERE id IN ({','.join('?' * len(chunk))})")
        for r in cur.execute(q, chunk):
            rows[r[0]] = r
    con.close()
    out = {}
    for code in codes:
        if code not in rows:
            raise KeyError(f"card {code} from the code list is not in {db_path}")
        _, alias, setcode, type_, level, atk, def_, race, attribute = rows[code]
        setcode = int(setcode) & ((1 << 64) - 1)
        cd = CardData(code=code, alias=alias, setcode=split_setcode(setcode), type=type_,
                      level=level & 0xFF, attack=atk, defense=def_, race=race, attribute=attribute)
        if code == 5405695 or (cd.alias and not (cd.type & TYPE_TOKEN) and not _is_alternative(code, cd.alias)):
            cd.rule_code = cd.alias
            cd.alias = 0
        if code in EXTRA_SETCODE:
            cd.setcode = list(EXTRA_SETCODE[code])
        out[code] = cd
    # alternative artworks inherit the rule code of their base card
    for cd in out.values():
        if cd.rule_code or not cd.alias or (cd.type & TYPE_TOKEN):
            continue
        base = out.get(cd.alias)
        if base is not None:
            cd.rule_code = base.rule_code
    return out


def _i32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x & 0x80000000 else x


def _cdiv(a: int, b: int) -> int:
    # C++ integer division truncates toward zero
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def is_declarable(cd: CardData, opcodes: Sequence[int]) -> bool:
    """Port of ygopro.h is_declarable_env (ygopro-core 0.0.3 is_declarable)."""
    if cd.alias:
        return False
    stack: List[int] = []
    for it in opcodes:
        it &= 0xFFFFFFFF
        if it in (OPCODE_ADD, OPCODE_SUB, OPCODE_MUL, OPCODE_DIV, OPCODE_AND, OPCODE_OR):
            if len(stack) >= 2:
                rhs = stack.pop()
                lhs = stack.pop()
                if it == OPCODE_ADD:
                    r = lhs + rhs
                elif it == OPCODE_SUB:
                    r = lhs - rhs
                elif it == OPCODE_MUL:
                    r = lhs * rhs
                elif it == OPCODE_DIV:
                    r = _cdiv(lhs, rhs) if rhs != 0 else 0
                elif it == OPCODE_AND:
                    r = int(bool(lhs) and bool(rhs))
                else:
                    r = int(bool(lhs) or bool(rhs))
                stack.append(_i32(r))
        elif it == OPCODE_NEG:
            if stack:
                stack.append(_i32(-stack.pop()))
        elif it == OPCODE_NOT:
            if stack:
                stack.append(int(not stack.pop()))
        elif it == OPCODE_ISCODE:
            if stack:
                code = stack.pop()
                stack.append(int(cd.code == (code & 0xFFFFFFFF)))
        elif it == OPCODE_ISSETCARD:
            if stack:
                set_code = stack.pop()
                stack.append(int(cd.is_setcode(set_code & 0xFFFFFFFF)))
        elif it == OPCODE_ISTYPE:
            if stack:
                stack.append(_i32(cd.type & (stack.pop() & 0xFFFFFFFF)))
        elif it == OPCODE_ISRACE:
            if stack:
                stack.append(_i32(cd.race & (stack.pop() & 0xFFFFFFFF)))
        elif it == OPCODE_ISATTRIBUTE:
            if stack:
                stack.append(_i32(cd.attribute & (stack.pop() & 0xFFFFFFFF)))
        else:
            stack.append(_i32(it))
    if len(stack) != 1 or stack[-1] == 0:
        return False
    if cd.code not in SECOND_CODE and (cd.rule_code or (cd.type & TYPE_TOKEN)):
        return False
    return True


def read_ydk(path) -> Tuple[List[int], List[int]]:
    """Main and extra deck of a .ydk file, parsed like the env's read_decks."""
    main, extra = [], []
    target = main
    with open(path, newline="") as f:
        for line in f.read().split("\n"):
            if "side" in line:
                break
            if "extra" in line:
                target = extra
                continue
            if line.endswith("\r"):
                line = line[:-1]
            if line and line.isdigit() and line.isascii():
                target.append(int(line))
    return main, extra


def deck_key(main: Iterable[int], extra: Iterable[int]) -> frozenset:
    return frozenset(main) | frozenset(extra)


def rank_pool(decks: Dict[str, Tuple[List[int], List[int]]]) -> List[int]:
    """Codes of the loaded decks by number of deck files containing them
    (descending), then by code. Names starting with '_' don't count."""
    count: Dict[int, int] = {}
    for name, (main, extra) in decks.items():
        if not name or name[0] == "_":
            continue
        for c in set(main) | set(extra):
            count[c] = count.get(c, 0) + 1
    return sorted(count, key=lambda c: (-count[c], c))


def load_pool(paths: Iterable[str]) -> Dict[str, Tuple[List[int], List[int]]]:
    """Deck files the way init_ygopro passes them to the env: a folder means
    every *.ydk in it, keyed by file stem."""
    decks = {}
    for p in paths:
        p = Path(p).expanduser()
        files = sorted(p.glob("*.ydk")) if p.is_dir() else [p]
        for f in files:
            decks[f.stem] = read_ydk(f)
    return decks


class AnnounceContext:
    """Everything the candidate list needs that doesn't change during a duel."""

    def __init__(self, card_data: Dict[int, CardData], code_to_id: Dict[int, int],
                 code_list_order: List[int], pool_decks: Dict[str, Tuple[List[int], List[int]]]):
        self.card_data = card_data
        self.code_to_id = code_to_id
        self.code_list_order = code_list_order
        self.pool_decks = pool_decks
        self.pool_keys = {deck_key(m, e) for m, e in pool_decks.values()}
        self.pool_ranked = rank_pool(pool_decks)
        self._ranked_with: Dict[frozenset, List[int]] = {}

    def ranked_for(self, own_main: Sequence[int], own_extra: Sequence[int]) -> List[int]:
        """Pool ranking of an env that loaded the pool decks and this player's
        deck. In training the playing deck is always one of the loaded decks;
        if it isn't in the pool folder, it counts as one more deck file."""
        key = deck_key(own_main, own_extra)
        if not key or key in self.pool_keys:
            return self.pool_ranked
        if key not in self._ranked_with:
            decks = dict(self.pool_decks)
            decks["\x01own"] = (list(own_main), list(own_extra))
            self._ranked_with[key] = rank_pool(decks)
        return self._ranked_with[key]


def announce_candidates(ctx: AnnounceContext, opcodes: Sequence[int], oppo_seen: Sequence[int],
                        own_seen: Sequence[int], own_main: Sequence[int], own_extra: Sequence[int],
                        cap: int) -> List[int]:
    """Port of ygopro.h announce_card_candidates. `*_seen` are in reveal order
    (oldest first, a repeated code moves to the end), as the env keeps them."""
    out: List[int] = []
    added = set()
    pool_ranked = ctx.ranked_for(own_main, own_extra)

    def try_add(code: int):
        if len(out) >= cap or code == 0 or code in added:
            return
        if code not in ctx.code_to_id:
            return
        cd = ctx.card_data.get(code)
        if cd is None or not is_declarable(cd, opcodes):
            return
        added.add(code)
        out.append(code)

    for i in range(1, len(opcodes)):
        if opcodes[i] == OPCODE_ISCODE and opcodes[i - 1] < OPCODE_ADD:
            try_add(opcodes[i - 1])
    for c in reversed(oppo_seen):
        try_add(c)
    for c in reversed(own_seen):
        try_add(c)
    if len(out) < cap:
        own = list(dict.fromkeys(list(own_main) + list(own_extra)))
        own_set = set(own)
        for c in pool_ranked:
            if c in own_set:
                try_add(c)
        for c in own:  # own cards outside the pool (not expected)
            try_add(c)
    for c in pool_ranked:
        if len(out) >= cap:
            break
        try_add(c)
    if not out:
        for c in ctx.code_list_order:
            if len(out) >= cap:
                break
            try_add(c)
    if not out:
        raise ValueError(f"no declarable card for opcodes {list(opcodes)}")
    return out
