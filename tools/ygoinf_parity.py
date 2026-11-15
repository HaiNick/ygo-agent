"""Parity test: the same game states through the env's encoder and ygoinf's.

The env runs a seeded random policy (self-play). A test build of the env
(tests/ygoinf_parity_env.cpp, module ygopro_parity) records the raw
information a game client has: every message buffer from the core and the
state of every card. A reference client below turns that into ygoinf requests
the way a real client would (one request per decision, including decisions the
env answers by itself because they have a single option), and ygoinf's encoder
builds the four observation arrays. They are compared with the env's arrays at
every decision the env shows the model.

Before anything else the test checks that the recording build gives the same
observations as the real ygopro_ygoenv module for the same seed.

Two views of the game (--view):

  env     The client is given everything the env's own state contains, so a
          difference is a bug in ygoinf's encoder. This includes the order of
          the player's own deck, the env's handling of messages after a skipped
          message (C12: the env drops them), and cards the opponent looked at
          privately that the env exposes (C9). The verdict of the test.
  client  The client uses only what it receives: all messages (none dropped),
          only confirmations sent to its own seat, and its own deck sorted by
          code because the deck order is unknown. A difference here is
          information a real client can't send. It's reported, not judged.
  client-deck  Like client, but with the true order of the own deck, to separate the
          deck order from the other gaps.

Hidden cards: the env writes an opponent card in the deck, hand or extra deck
as a count-only row (location and owner, position 0) unless the env's
revealed_ set is non-empty (C9). The client sends such cards with code 0 and
position none, and ygoinf then writes the same row and gives it no spec. An
opponent face-down card elsewhere is sent with code 0 and its real position.

usage: python tools/ygoinf_parity.py [--deck DIR] [--deck1 NAME --deck2 NAME]
         [--games N] [--seed S] [--view env|client|client-deck] [--ygoinf DIR] [--json FILE] [--no-ref]
Build the recording module first: xmake b ygopro_parity (copies it to tests/).
PARITY_DEBUG=N prints the rows of the first N decisions with a difference.
exit code 0 = no mismatch in the env view (and the recording build is faithful).
"""
import argparse
import json
import os
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- core constants
LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE = 0x01, 0x02, 0x04, 0x08
LOCATION_GRAVE, LOCATION_REMOVED, LOCATION_EXTRA, LOCATION_OVERLAY = 0x10, 0x20, 0x40, 0x80
LOCATION_ONFIELD = LOCATION_MZONE | LOCATION_SZONE
POS_FACEUP, POS_FACEDOWN = 0x5, 0xA
STATUS_DISABLED, STATUS_FORBIDDEN = 0x0001, 0x4000000
HINT_SELECTMSG = 3
PHASE_END = 0x200
LOC_ORDER = [LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE,
             LOCATION_GRAVE, LOCATION_REMOVED, LOCATION_EXTRA]
HIDDEN_FOR_OPPONENT = {LOCATION_DECK, LOCATION_HAND, LOCATION_EXTRA}

M = dict(RETRY=1, HINT=2, WIN=5, SELECT_BATTLECMD=10, SELECT_IDLECMD=11, SELECT_EFFECTYN=12,
         SELECT_YESNO=13, SELECT_OPTION=14, SELECT_CARD=15, SELECT_CHAIN=16, SELECT_PLACE=18,
         SELECT_POSITION=19, SELECT_TRIBUTE=20, SELECT_COUNTER=22, SELECT_SUM=23,
         SELECT_DISFIELD=24, SORT_CARD=25, SELECT_UNSELECT_CARD=26, CONFIRM_DECKTOP=30,
         CONFIRM_CARDS=31, SHUFFLE_DECK=32, SHUFFLE_HAND=33, REFRESH_DECK=34, SWAP_GRAVE_DECK=35,
         SHUFFLE_SET_CARD=36, REVERSE_DECK=37, DECK_TOP=38, SHUFFLE_EXTRA=39, NEW_TURN=40,
         NEW_PHASE=41, CONFIRM_EXTRATOP=42, MOVE=50, POS_CHANGE=53, SET=54, SWAP=55,
         FIELD_DISABLED=56, SUMMONING=60, SUMMONED=61, SPSUMMONING=62, SPSUMMONED=63,
         FLIPSUMMONING=64, FLIPSUMMONED=65, CHAINING=70, CHAINED=71, CHAIN_SOLVING=72,
         CHAIN_SOLVED=73, CHAIN_END=74, CHAIN_NEGATED=75, CHAIN_DISABLED=76, CARD_SELECTED=80,
         RANDOM_SELECTED=81, BECOME_TARGET=83, DRAW=90, DAMAGE=91, RECOVER=92, EQUIP=93,
         LPUPDATE=94, UNEQUIP=95, CARD_TARGET=96, CANCEL_TARGET=97, PAY_LPCOST=100,
         ADD_COUNTER=101, REMOVE_COUNTER=102, ATTACK=110, BATTLE=111, ATTACK_DISABLED=112,
         DAMAGE_STEP_START=113, DAMAGE_STEP_END=114, MISSED_EFFECT=120, TOSS_COIN=130,
         TOSS_DICE=131, ROCK_PAPER_SCISSORS=132, HAND_RES=133, ANNOUNCE_RACE=140,
         ANNOUNCE_ATTRIB=141, ANNOUNCE_CARD=142, ANNOUNCE_NUMBER=143, CARD_HINT=160,
         TAG_SWAP=161, RELOAD_FIELD=162, AI_NAME=163, SHOW_HINT=164, PLAYER_HINT=165,
         MATCH_KILL=170, CUSTOM_MSG=180)
MSG_NAME = {v: k for k, v in M.items()}
DECISIONS = {M[k] for k in ("SELECT_BATTLECMD", "SELECT_IDLECMD", "SELECT_EFFECTYN", "SELECT_YESNO",
                            "SELECT_OPTION", "SELECT_CARD", "SELECT_CHAIN", "SELECT_PLACE",
                            "SELECT_POSITION", "SELECT_TRIBUTE", "SELECT_COUNTER", "SELECT_SUM",
                            "SELECT_DISFIELD", "SORT_CARD", "SELECT_UNSELECT_CARD",
                            "ROCK_PAPER_SCISSORS", "ANNOUNCE_RACE", "ANNOUNCE_ATTRIB",
                            "ANNOUNCE_CARD", "ANNOUNCE_NUMBER")}
# Env handlers (ygopro.h handle_message) that consume exactly their own bytes.
# Every other non-decision handler ends with dp_ = dl_ and drops the rest of
# the buffer (C12). SORT_CARD drops too, but it's always last.
ENV_EXACT = {M[k] for k in ("NEW_TURN", "NEW_PHASE", "HINT", "CONFIRM_CARDS", "DAMAGE", "RECOVER",
                            "LPUPDATE", "PAY_LPCOST", "DAMAGE_STEP_START", "DAMAGE_STEP_END", "WIN")}


def msg_len(m, b, p):
    """Payload length of message m at b[p:] (port of tests/yrp_verify.cpp,
    which follows the reference client's ReplayAnalyze)."""
    u8 = lambda o: b[p + o]
    fixed = {M["HINT"]: 6, M["WIN"]: 2, M["SELECT_EFFECTYN"]: 13, M["SELECT_YESNO"]: 5,
             M["SELECT_PLACE"]: 6, M["SELECT_DISFIELD"]: 6, M["SELECT_POSITION"]: 6,
             M["SHUFFLE_DECK"]: 1, M["REFRESH_DECK"]: 1, M["SWAP_GRAVE_DECK"]: 1, M["REVERSE_DECK"]: 0,
             M["DECK_TOP"]: 6, M["NEW_TURN"]: 1, M["NEW_PHASE"]: 2, M["MOVE"]: 16, M["POS_CHANGE"]: 9,
             M["SET"]: 8, M["SWAP"]: 16, M["FIELD_DISABLED"]: 4, M["SUMMONING"]: 8, M["SPSUMMONING"]: 8,
             M["FLIPSUMMONING"]: 8, M["SUMMONED"]: 0, M["SPSUMMONED"]: 0, M["FLIPSUMMONED"]: 0,
             M["CHAINING"]: 16, M["CHAINED"]: 1, M["CHAIN_SOLVING"]: 1, M["CHAIN_SOLVED"]: 1,
             M["CHAIN_NEGATED"]: 1, M["CHAIN_DISABLED"]: 1, M["CHAIN_END"]: 0, M["DAMAGE"]: 5,
             M["RECOVER"]: 5, M["LPUPDATE"]: 5, M["PAY_LPCOST"]: 5, M["EQUIP"]: 8, M["UNEQUIP"]: 4,
             M["CARD_TARGET"]: 8, M["CANCEL_TARGET"]: 8, M["ADD_COUNTER"]: 7, M["REMOVE_COUNTER"]: 7,
             M["ATTACK"]: 8, M["BATTLE"]: 26, M["ATTACK_DISABLED"]: 0, M["DAMAGE_STEP_START"]: 0,
             M["DAMAGE_STEP_END"]: 0, M["MISSED_EFFECT"]: 8, M["ROCK_PAPER_SCISSORS"]: 1,
             M["HAND_RES"]: 1, M["ANNOUNCE_RACE"]: 6, M["ANNOUNCE_ATTRIB"]: 6, M["CARD_HINT"]: 9,
             M["PLAYER_HINT"]: 6, M["MATCH_KILL"]: 4}
    if m in fixed:
        return fixed[m]
    if m == M["SELECT_BATTLECMD"]:
        n1 = u8(1); o = 2 + n1 * 11; n2 = u8(o); return o + 1 + n2 * 8 + 2
    if m == M["SELECT_IDLECMD"]:
        o = 1
        for _ in range(5):
            o += 1 + u8(o) * 7
        return o + 1 + u8(o) * 11 + 3
    if m == M["SELECT_OPTION"]:
        return 2 + u8(1) * 4
    if m in (M["SELECT_CARD"], M["SELECT_TRIBUTE"]):
        return 5 + u8(4) * 8
    if m == M["SELECT_UNSELECT_CARD"]:
        o = 6 + u8(5) * 8; return o + 1 + u8(o) * 8
    if m == M["SELECT_CHAIN"]:
        return 11 + u8(1) * 14
    if m == M["SELECT_COUNTER"]:
        return 6 + u8(5) * 9
    if m == M["SELECT_SUM"]:
        o = 8 + u8(8) * 11 + 1; return o + 1 + u8(o) * 11
    if m == M["SORT_CARD"]:
        return 2 + u8(1) * 7
    if m in (M["CONFIRM_DECKTOP"], M["CONFIRM_EXTRATOP"]):
        return 2 + u8(1) * 7
    if m == M["CONFIRM_CARDS"]:
        return 3 + u8(2) * 7
    if m in (M["SHUFFLE_HAND"], M["SHUFFLE_EXTRA"]):
        return 2 + u8(1) * 4
    if m == M["SHUFFLE_SET_CARD"]:
        return 2 + u8(1) * 8
    if m in (M["CARD_SELECTED"], M["RANDOM_SELECTED"]):
        return 2 + u8(1) * 4
    if m == M["BECOME_TARGET"]:
        return 1 + u8(0) * 4
    if m == M["DRAW"]:
        return 2 + u8(1) * 4
    if m in (M["TOSS_COIN"], M["TOSS_DICE"]):
        return 2 + u8(1)
    if m in (M["ANNOUNCE_CARD"], M["ANNOUNCE_NUMBER"]):
        return 2 + u8(1) * 4
    if m == M["TAG_SWAP"]:
        return u8(2) * 4 + u8(4) * 4 + 9
    if m in (M["AI_NAME"], M["SHOW_HINT"]):
        return 2 + struct.unpack_from("<H", b, p)[0] + 1
    if m == M["RELOAD_FIELD"]:
        o = 1
        for _ in range(2):
            o += 4
            for _ in range(7):
                o += 1 + (2 if u8(o) else 0)
            for _ in range(8):
                o += 1 + (1 if u8(o) else 0)
            o += 6
        return o + 1
    raise ValueError(f"unknown message {m}")


def walk(buf):
    """Messages of one core buffer: list of (msg, payload bytes)."""
    out, p = [], 0
    while p < len(buf):
        m = buf[p]; p += 1
        n = msg_len(m, buf, p)
        out.append((m, buf[p:p + n]))
        p += n
    return out


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def i32(b, o):
    return struct.unpack_from("<i", b, o)[0]


# ---------------------------------------------------------------- snapshot
class QCard:
    __slots__ = ("code", "con", "loc", "seq", "pos", "level", "rank", "atk", "def_", "xyz",
                 "counters", "status", "link", "link_marker")


def parse_snapshot(blob):
    """{(player, location): [QCard]} from a kind-2 record."""
    out, p = {}, 0
    while p < len(blob):
        pl, loc = blob[p], blob[p + 1]
        n = u32(blob, p + 2); p += 6
        q = blob[p:p + n]; p += n
        cards, o = [], 0
        while o < len(q):
            ln = u32(q, o)
            if ln == 4:  # LEN_EMPTY
                o += 4
                continue
            r = o + 4
            flags = u32(q, r); r += 4
            c = QCard()
            c.code = u32(q, r); r += 4
            c.con, c.loc, c.seq, c.pos = q[r], q[r + 1], q[r + 2], q[r + 3]; r += 4
            c.level = u32(q, r); c.rank = u32(q, r + 4); r += 8
            c.atk = i32(q, r); c.def_ = i32(q, r + 4); r += 8
            if flags & 0x4000:
                r += 4
            nx = u32(q, r); r += 4
            c.xyz = [u32(q, r + 4 * i) for i in range(nx)]; r += 4 * nx
            nc = u32(q, r); r += 4
            c.counters = [u32(q, r + 4 * i) for i in range(nc)]; r += 4 * nc
            c.status = u32(q, r); r += 4
            r += 8  # lscale, rscale
            c.link = u32(q, r); c.link_marker = u32(q, r + 4)
            cards.append(c)
            o += ln
        out[(pl, loc)] = cards
    return out


# ---------------------------------------------------------------- card data
class StaticCard:
    __slots__ = ("type", "level", "attack", "defense", "race", "attribute")


def load_card_data(db_path, codes):
    """Printed card data (cards.cdb datas) the way the env's Card has it
    (db_query_card): level & 0xff, a link monster's defense is 0."""
    import sqlite3
    codes = list(codes)
    out = {}
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    for i in range(0, len(codes), 500):
        chunk = codes[i:i + 500]
        q = f"SELECT id, type, level, atk, def, race, attribute FROM datas WHERE id IN ({','.join('?' * len(chunk))})"
        for code, type_, level, atk, df, race, attr in con.execute(q, chunk):
            c = StaticCard()
            c.type, c.level, c.attack, c.race, c.attribute = type_, level & 0xFF, atk, race, attr
            c.defense = 0 if type_ & 0x4000000 else df
            out[code] = c
    con.close()
    return out


# ---------------------------------------------------------------- client
def ls_to_spec(loc, seq, pos, opponent):
    s = ""
    if loc & LOCATION_HAND: s = "h"
    elif loc & LOCATION_MZONE: s = "m"
    elif loc & LOCATION_SZONE: s = "s"
    elif loc & LOCATION_GRAVE: s = "g"
    elif loc & LOCATION_REMOVED: s = "r"
    elif loc & LOCATION_EXTRA: s = "x"
    s += str(seq + 1)
    if loc & LOCATION_OVERLAY:
        s += chr(ord("a") + pos)
    return ("o" if opponent else "") + s


class DuelState:
    """What the env tracks from messages (ygopro.h), seen through one seat."""

    def __init__(self):
        self.lp = [8000, 8000]
        self.turn = 0
        self.tp = 0
        self.phase = 0x01
        self.revealed = set()       # env revealed_ (C9 semantics), never reset per duel
        self.seen = [[], []]        # env seen_codes_
        self.confirmed = defaultdict(list)  # env confirmed_locs_
        self.known = {}             # (con, loc, seq) -> code from confirmations this seat received
        self.discard_hand = False
        self.last_hint = [0, 0]

    def new_duel(self):
        self.lp = [8000, 8000]
        self.turn = 0
        self.seen = [[], []]
        self.confirmed.clear()
        self.known.clear()
        self.discard_hand = False
        self.last_hint = [0, 0]

    def note_seen(self, player, code):
        if player > 1 or code == 0:
            return
        v = self.seen[player]
        if code in v:
            v.remove(code)
        v.append(code)

    def apply(self, m, b, receives):
        """Update from one handled message. receives(m, b) says whether this
        seat gets the message."""
        if m in (M["SUMMONING"], M["SPSUMMONING"], M["FLIPSUMMONING"], M["CHAINING"]):
            self.note_seen(u32(b, 4) & 0xFF, u32(b, 0))
        elif m == M["MOVE"]:
            newloc = u32(b, 8)
            loc, pos = (newloc >> 8) & 0xFF, (newloc >> 24) & 0xFF
            if loc == LOCATION_GRAVE or ((loc & (LOCATION_ONFIELD | LOCATION_REMOVED | LOCATION_EXTRA)) and (pos & POS_FACEUP)):
                self.note_seen(newloc & 0xFF, u32(b, 0))
        elif m == M["POS_CHANGE"]:
            if b[8] & POS_FACEUP:
                self.note_seen(u32(b, 4) & 0xFF, u32(b, 0))
        elif m in (M["CONFIRM_CARDS"], M["CONFIRM_DECKTOP"], M["CONFIRM_EXTRATOP"]):
            if receives(m, b):
                off = 2 if m == M["CONFIRM_CARDS"] else 1
                n = b[off]; off += 1
                for i in range(n):
                    code = u32(b, off); c, l, s = b[off + 4], b[off + 5], b[off + 6]
                    self.note_seen(c, code)
                    self.confirmed[(c << 8) | l].append(code)
                    self.known[(c, l, s)] = code
                    off += 7
            if m == M["CONFIRM_CARDS"]:
                # the env inserts these specs whoever sees the cards
                player, n, off = b[0], b[2], 3
                for i in range(n):
                    c, l, s = b[off + 4], b[off + 5], b[off + 6]
                    self.revealed.add(ls_to_spec(l, s, 0, c == player))
                    off += 7
        if m == M["NEW_TURN"]:
            self.tp = b[0]; self.turn += 1
        elif m == M["NEW_PHASE"]:
            self.phase = struct.unpack_from("<H", b, 0)[0]
        elif m == M["HINT"]:
            if b[0] == HINT_SELECTMSG:
                if u32(b, 2) == 501:
                    self.discard_hand = True
        elif m in (M["DAMAGE"], M["PAY_LPCOST"]):
            self.lp[b[0]] -= u32(b, 1)
        elif m == M["RECOVER"]:
            self.lp[b[0]] += u32(b, 1)
        elif m == M["LPUPDATE"]:
            self.lp[b[0]] = u32(b, 1)
        elif m == M["CHAIN_SOLVED"]:
            self.revealed.clear()
            self.confirmed.clear()
            self.known.clear()


class RefClient:
    """Builds ygoinf requests for both seats from the recorded stream."""

    def __init__(self, F, Mo, view, card_data):
        self.F, self.Mo, self.view = F, Mo, view
        self.sort_deck = view == "client"
        self.cd = card_data
        # env view: one state equal to the env's. client view: one per seat.
        self.states = [DuelState()] if view == "env" else [DuelState(), DuelState()]
        self.snapshot = None
        self.decks = [None, None]
        self.queue = []
        self.cur = None
        self.pstates = [None, None]
        self.prev_idx = [0, 0]
        self.stats = Counter()
        self.debug_last = None
        self.new_duel()

    def st(self, seat):
        return self.states[0] if self.view == "env" else self.states[seat]

    def new_duel(self):
        for s in self.states:
            s.new_duel()
        self.queue, self.cur = [], None
        self.pstates = [self.F.PredictState(), self.F.PredictState()]
        self.prev_idx = [0, 0]
        self.decks = [None, None]
        self.need_decks = True

    # ---- stream
    def feed(self, records):
        for kind, data in records:
            if kind == 0:
                if self.cur is not None or self.queue:
                    self.stats["decisions_left_at_duel_end"] += 1
                self.new_duel()
            elif kind == 1:
                self._buffer(data)
            else:
                self.snapshot = parse_snapshot(data)
                if self.need_decks:
                    self.need_decks = False
                    for p in range(2):
                        main = [c.code for loc in (LOCATION_DECK, LOCATION_HAND)
                                for c in self.snapshot[(p, loc)]]
                        extra = [c.code for c in self.snapshot[(p, LOCATION_EXTRA)]]
                        self.decks[p] = (main, extra)
                if self.queue and self.queue[-1].get("snap") is None:
                    self.queue[-1]["snap"] = self.snapshot

    def _buffer(self, buf):
        msgs = walk(buf)
        dropped = False
        for k, (m, b) in enumerate(msgs):
            env_handles = not dropped
            for seat, s in enumerate(self.states):
                handles = env_handles if self.view == "env" else True
                if handles and m not in DECISIONS:
                    s.apply(m, b, (lambda mm, bb, seat=seat: self._receives(seat, mm, bb)))
            if m in DECISIONS:
                if k != len(msgs) - 1:
                    self.stats["decision_not_last_in_buffer"] += 1
                if dropped:
                    self.stats["decision_after_dropped_message"] += 1
                    if self.view == "env":
                        continue
                self.queue.append(dict(m=m, b=b, snap=None,
                                       st=[self._copy_state(s) for s in self.states]))
                if m == M["SELECT_CARD"]:
                    # env discard_hand_: set by any 501 hint, cleared by the next select_card
                    for s in self.states:
                        s.discard_hand = False
            if m not in ENV_EXACT and m not in DECISIONS:
                dropped = True

    def _receives(self, seat, m, b):
        if self.view == "env":
            return True
        if m == M["CONFIRM_CARDS"] and b[2] > 0 and b[3 + 5] == LOCATION_DECK:
            # the server sends deck confirmations to the viewing player only
            return b[0] == seat
        return True

    @staticmethod
    def _copy_state(s):
        c = DuelState.__new__(DuelState)
        c.lp = list(s.lp); c.turn = s.turn; c.tp = s.tp; c.phase = s.phase
        c.revealed = set(s.revealed); c.seen = [list(s.seen[0]), list(s.seen[1])]
        c.confirmed = defaultdict(list, {k: list(v) for k, v in s.confirmed.items()})
        c.known = dict(s.known); c.discard_hand = s.discard_hand; c.last_hint = list(s.last_hint)
        return c

    # ---- request building
    def _cardloc(self, p, c, l, s, pos):
        Mo = self.Mo
        overlay = bool(l & LOCATION_OVERLAY)
        return dict(controller="me" if c == p else "opponent", location=LOCNAME[l & 0x7F],
                    sequence=s, overlay_sequence=pos if overlay else -1)

    def _cardinfo(self, p, code, c, l, s):
        return dict(code=code, controller="me" if c == p else "opponent", location=LOCNAME[l & 0x7F], sequence=s)

    def _static_card(self, code):
        cd = self.cd[code]
        return cd, cd.attack, cd.defense

    def _card_dict(self, p, code, con, loc, seq, pos, overlay_seq, level, atk, df, counter, negated, blank=False):
        if blank:
            return dict(code=0, location=LOCNAME[loc], sequence=seq, controller="me" if con == p else "opponent",
                        position=POSNAME.get(pos, "none"), overlay_sequence=-1, attribute="none", race="none",
                        level=0, counter=0, negated=False, attack=0, defense=0, types=[])
        cd = self.cd[code]
        return dict(code=code, location=LOCNAME[loc], sequence=seq, controller="me" if con == p else "opponent",
                    position=POSNAME.get(pos, "none") if overlay_seq < 0 else "faceup",
                    overlay_sequence=overlay_seq, attribute=ATTRNAME[cd.attribute], race=RACENAME[cd.race],
                    level=level, counter=counter, negated=negated, attack=atk, defense=df,
                    types=[TYPENAMES[i] for i, bit in enumerate(TYPEBITS) if cd.type & bit])

    def cards(self, p, st, snap):
        out = []
        for pi in range(2):
            pl = (p + pi) % 2
            opp = pi == 1
            for loc in LOC_ORDER:
                qcards = snap[(pl, loc)]
                if opp and loc in HIDDEN_FOR_OPPONENT and not st.revealed:
                    for c in qcards:
                        out.append(self._card_dict(p, 0, pl, loc, 0, 0, -1, 0, 0, 0, 0, False, blank=True))
                    continue
                rows = []
                for c in qcards:
                    for i, xc in enumerate(c.xyz):
                        cd, atk, df = self._static_card(xc)
                        rows.append(self._card_dict(p, xc, c.con, c.loc, c.seq, 0, i, cd.level, atk, df, 0, False))
                    spec = ls_to_spec(c.loc, c.seq, c.pos, opp)
                    hide = opp and bool(c.pos & POS_FACEDOWN) and spec not in st.revealed
                    if not hide and opp and (c.pos & POS_FACEDOWN) and self.view != "env":
                        if st.known.get((c.con, c.loc, c.seq)) != c.code:
                            self.stats["client_cannot_know_revealed_card"] += 1
                            hide = True
                    if hide:
                        pos = c.pos
                        if loc in HIDDEN_FOR_OPPONENT:
                            pos = POS_FACEDOWN
                        rows.append(self._card_dict(p, 0, c.con, c.loc, c.seq, pos, -1, 0, 0, 0, 0, False, blank=True))
                        continue
                    cd = self.cd[c.code]
                    level = cd.level
                    if c.level & 0xFF: level = c.level & 0xFF
                    if c.rank & 0xFF: level = c.rank & 0xFF
                    if c.link & 0xFF: level = c.link & 0xFF
                    df = c.link_marker if c.link_marker > 0 else c.def_
                    counter = c.counters[0] if c.counters else 0
                    neg = bool(c.status & (STATUS_DISABLED | STATUS_FORBIDDEN))
                    rows.append(self._card_dict(p, c.code, c.con, c.loc, c.seq, c.pos, -1, level, c.atk, df, counter, neg))
                if self.sort_deck and not opp and loc == LOCATION_DECK:
                    # a client doesn't know the order of its own deck
                    rows.sort(key=lambda r: r["code"])
                    for i, r in enumerate(rows):
                        r["sequence"] = i
                out.extend(rows)
        return out

    def global_(self, p, st):
        return dict(my_lp=st.lp[p], op_lp=st.lp[1 - p], turn=st.turn, phase=PHASENAME[st.phase],
                    is_first=p == 0, is_my_turn=p == st.tp)

    def action_msg(self, p, d, st):
        m, b = d["m"], d["b"]
        L = lambda c, l, s, pos: self._cardloc(p, c, l, s, pos)
        if m == M["SELECT_IDLECMD"]:
            cmds, o = [], 1
            for t in ("summon", "sp_summon", "reposition", "mset", "set"):
                n = b[o]; o += 1
                for _ in range(n):
                    code = u32(b, o); c, l, s = b[o + 4], b[o + 5], b[o + 6]; o += 7
                    cmds.append(dict(cmd_type=t, data=dict(card_info=self._cardinfo(p, code, c, l, s),
                                                           effect_description=0, response=len(cmds))))
            n = b[o]; o += 1
            for _ in range(n):
                code = u32(b, o); c, l, s = b[o + 4], b[o + 5], b[o + 6]; desc = u32(b, o + 7); o += 11
                cmds.append(dict(cmd_type="activate", data=dict(card_info=self._cardinfo(p, code, c, l, s),
                                                                effect_description=desc, response=len(cmds))))
            if b[o]: cmds.append(dict(cmd_type="to_bp"))
            if b[o + 1]: cmds.append(dict(cmd_type="to_ep"))
            return dict(msg_type="select_idlecmd", idle_cmds=cmds)
        if m == M["SELECT_BATTLECMD"]:
            cmds, o = [], 1
            n = b[o]; o += 1
            for _ in range(n):
                code = u32(b, o); c, l, s = b[o + 4], b[o + 5], b[o + 6]; desc = u32(b, o + 7); o += 11
                cmds.append(dict(cmd_type="activate", data=dict(card_info=self._cardinfo(p, code, c, l, s),
                                 effect_description=desc, direct_attackable=False, response=len(cmds))))
            n = b[o]; o += 1
            for _ in range(n):
                code = u32(b, o); c, l, s = b[o + 4], b[o + 5], b[o + 6]; da = b[o + 7]; o += 8
                cmds.append(dict(cmd_type="attack", data=dict(card_info=self._cardinfo(p, code, c, l, s),
                                 effect_description=0, direct_attackable=bool(da & 1), response=len(cmds))))
            if b[o]: cmds.append(dict(cmd_type="to_m2"))
            if b[o + 1]: cmds.append(dict(cmd_type="to_ep"))
            return dict(msg_type="select_battlecmd", battle_cmds=cmds)
        if m == M["SELECT_EFFECTYN"]:
            return dict(msg_type="select_effectyn", code=u32(b, 1), location=L(b[5], b[6], b[7], b[8]),
                        effect_description=u32(b, 9))
        if m == M["SELECT_YESNO"]:
            return dict(msg_type="select_yesno", effect_description=u32(b, 1))
        if m == M["SELECT_OPTION"]:
            return dict(msg_type="select_option", options=[dict(code=u32(b, 2 + 4 * i), response=i) for i in range(b[1])])
        if m in (M["SELECT_CARD"], M["SELECT_TRIBUTE"]):
            n, o, cards = b[4], 5, []
            for i in range(n):
                code = u32(b, o); c, l, s, x = b[o + 4], b[o + 5], b[o + 6], b[o + 7]; o += 8
                if m == M["SELECT_CARD"]:
                    conf = l in (LOCATION_DECK, LOCATION_EXTRA) and code in st.confirmed.get((c << 8) | l, [])
                    cards.append(dict(location=L(c, l, s, x), response=i, code=code, confirmed=conf))
                else:
                    cards.append(dict(location=L(c, l, s, 0), level=x, response=i, code=code))
            d_ = dict(msg_type="select_card" if m == M["SELECT_CARD"] else "select_tribute",
                      cancelable=bool(b[1]), min=b[2], max=b[3], cards=cards, selected=list(d["selected"]))
            if m == M["SELECT_CARD"]:
                d_["select_hint"] = 501 if d["discard_hint"] else 0
            return d_
        if m == M["SELECT_UNSELECT_CARD"]:
            n, o, cards = b[5], 6, []
            for i in range(n):
                code = u32(b, o); c, l, s, x = b[o + 4], b[o + 5], b[o + 6], b[o + 7]; o += 8
                cards.append(dict(location=L(c, l, s, x), response=i, code=code))
            return dict(msg_type="select_unselect_card", finishable=bool(b[1]), cancelable=bool(b[2]),
                        min=b[3], max=b[4], selected_cards=[], selectable_cards=cards)
        if m == M["SELECT_CHAIN"]:
            n, o, chains = b[1], 11, []
            forced = False
            for i in range(n):
                forced = forced or b[o + 1] != 0
                code = u32(b, o + 2); c, l, s, x = b[o + 6], b[o + 7], b[o + 8], b[o + 9]; desc = u32(b, o + 10)
                chains.append(dict(code=code, location=L(c, l, s, x), effect_description=desc, response=i))
                o += 14
            return dict(msg_type="select_chain", forced=forced, chains=chains)
        if m in (M["SELECT_PLACE"], M["SELECT_DISFIELD"]):
            flag = u32(b, 2)
            places = []
            for j, (ctl, loc, nmax) in enumerate((("me", "mzone", 7), ("me", "szone", 8),
                                                  ("opponent", "mzone", 7), ("opponent", "szone", 8))):
                v = (flag >> (8 * j)) & 0xFF
                for i in range(nmax):
                    if not (v & (1 << i)):
                        places.append(dict(controller=ctl, location=loc, sequence=i))
            # a client can list places in any order; send them reversed to check the sorting
            return dict(msg_type="select_place" if m == M["SELECT_PLACE"] else "select_disfield",
                        count=b[1], places=places[::-1])
        if m == M["SELECT_POSITION"]:
            pos = b[5]
            return dict(msg_type="select_position", code=u32(b, 1),
                        positions=[POSNAME[x] for x in (1, 2, 4, 8) if pos & x])
        if m == M["ANNOUNCE_ATTRIB"]:
            flag = u32(b, 2)
            return dict(msg_type="announce_attrib", count=b[1],
                        attributes=[dict(attribute=ATTRNAME[1 << i], response=1 << i) for i in range(7) if flag & (1 << i)])
        if m == M["ANNOUNCE_NUMBER"]:
            return dict(msg_type="announce_number", count=1,
                        numbers=[dict(number=u32(b, 2 + 4 * i), response=i) for i in range(b[1])])
        if m == M["ANNOUNCE_CARD"]:
            main, extra = self.decks[p]
            if self.sort_deck:
                main, extra = sorted(main), sorted(extra)
            return dict(msg_type="announce_card", opcodes=[u32(b, 2 + 4 * i) for i in range(b[1])],
                        seen=dict(me=list(st.seen[p]), opponent=list(st.seen[1 - p])),
                        deck_main=main, deck_extra=extra)
        if m == M["SELECT_SUM"]:
            o = 8
            def rd(o):
                c, l, s = b[o + 4], b[o + 5], b[o + 6]; prm = u32(b, o + 7)
                return dict(location=L(c, l, s, 0), level1=prm & 0xFF, level2=prm >> 16, response=0, code=u32(b, o))
            must = []
            for i in range(b[o]):
                must.append(rd(o + 1 + 11 * i))
            o = o + 1 + 11 * b[o]
            cards = []
            for i in range(b[o]):
                r = rd(o + 1 + 11 * i); r["response"] = i; cards.append(r)
            return dict(msg_type="select_sum", overflow=b[0] != 0, level_sum=u32(b, 2) & 0xFFFF,
                        min=b[6], max=b[7], cards=cards, must_cards=must, selected=list(d["selected"]))
        raise NotImplementedError(MSG_NAME[m])

    def player_of(self, d):
        m, b = d["m"], d["b"]
        return b[1] if m == M["SELECT_SUM"] else b[0]

    def env_auto(self, d):
        """Decisions the env answers without any legal action."""
        m, b = d["m"], d["b"]
        if m in (M["SELECT_COUNTER"], M["SORT_CARD"]):
            return True
        if m == M["SELECT_CHAIN"] and b[1] == 0 and b[2] == 0:
            return True
        return False

    def request(self, d):
        """ygoinf Input for the current step of decision d."""
        p = self.player_of(d)
        st = d["st"][0] if self.view == "env" else d["st"][p]
        data = self.action_msg(p, d, st)
        inp = {"global": self.global_(p, st), "cards": self.cards(p, st, d["snap"]),
               "action_msg": {"data": data}}
        return p, self.Mo.Input(**inp)

    def call(self, d, captured):
        """One ygoinf call for decision d. Returns (player, response)."""
        p, inp = self.request(d)
        st_ = self.pstates[p]
        def model_fn(rstate, model_input):
            captured.append(model_input)
            n = int((model_input["actions_"][:, 3] != 0).sum())
            return rstate, [1.0 / max(n, 1)] * max(n, 1), 0.0
        res = self.F.predict(model_fn, inp, self.prev_idx[p], st_)
        return p, inp, res


LOCNAME = {LOCATION_DECK: "deck", LOCATION_HAND: "hand", LOCATION_MZONE: "mzone", LOCATION_SZONE: "szone",
           LOCATION_GRAVE: "grave", LOCATION_REMOVED: "removed", LOCATION_EXTRA: "extra"}
POSNAME = {0: "none", 1: "faceup_attack", 2: "facedown_attack", 3: "attack", 4: "faceup_defense",
           5: "faceup", 8: "facedown_defense", 10: "facedown", 12: "defense"}
ATTRNAME = {0: "none", 1: "earth", 2: "water", 4: "fire", 8: "wind", 16: "light", 32: "dark", 64: "divine"}
RACES = ["warrior", "spellcaster", "fairy", "fiend", "zombie", "machine", "aqua", "pyro", "rock", "windbeast",
         "plant", "insect", "thunder", "dragon", "beast", "beast_warrior", "dinosaur", "fish", "sea_serpent",
         "reptile", "psycho", "devine", "creator_god", "wyrm", "cyberse", "illusion"]
RACENAME = {0: "none", **{1 << i: r for i, r in enumerate(RACES)}}
TYPEBITS = [0x1, 0x2, 0x4, 0x10, 0x20, 0x40, 0x80, 0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000,
            0x10000, 0x20000, 0x40000, 0x80000, 0x100000, 0x200000, 0x400000, 0x800000, 0x1000000,
            0x2000000, 0x4000000]
TYPENAMES = ["monster", "spell", "trap", "normal", "effect", "fusion", "ritual", "trap_monster", "spirit",
             "union", "dual", "tuner", "synchro", "token", "quick_play", "continuous", "equip", "field",
             "counter", "flip", "toon", "xyz", "pendulum", "special", "link"]
PHASENAME = {0x1: "draw", 0x2: "standby", 0x4: "main1", 0x8: "battle_start", 0x10: "battle_step",
             0x20: "damage", 0x40: "damage_calculation", 0x80: "battle", 0x100: "main2", 0x200: "end"}

FEATURES = {
    "cards_": ["id_hi", "id_lo", "location", "sequence", "controller", "position", "overlay", "attribute",
               "race", "level", "counter", "negated", "atk_hi", "atk_lo", "def_hi", "def_lo"]
              + ["type_" + t for t in TYPENAMES],
    "global_": ["my_lp_hi", "my_lp_lo", "op_lp_hi", "op_lp_lo", "turn", "phase", "is_first", "is_my_turn"]
               + [f"n_{w}_{l}" for w in ("my", "op") for l in ("deck", "hand", "mzone", "szone", "grave", "removed", "extra")]
               + ["terminal"],
    "actions_": ["card_index", "id_hi", "id_lo", "msg", "act", "finish", "effect", "phase", "position",
                 "number", "place", "attribute"],
    "h_actions_": ["card_index", "id_hi", "id_lo", "msg", "act", "finish", "effect", "phase", "position",
                   "number", "place", "attribute", "turn_diff", "phase_at"],
}


# ---------------------------------------------------------------- driver
def make_envs(deck_dir, deck1, deck2, seed, with_ref):
    os.chdir(ROOT / "scripts")
    sys.path.insert(0, str(ROOT / "tests"))
    import numpy as np  # noqa: F401
    import ygopro_parity as P
    from ygoenv.python.api import py_env
    import ygoenv
    from ygoai.utils import init_ygopro
    db = str(ROOT / "assets" / "locale" / "en" / "cards.cdb")
    decks = {f.stem: str(f) for f in Path(deck_dir).glob("*.ydk")}
    cfg = dict(num_envs=1, num_threads=1, seed=seed, deck1=deck1, deck2=deck2, max_options=24,
               n_history_actions=32, play_mode="self", async_reset=False, greedy_reward=False,
               timeout=600, oppo_info=False)
    P.init_module(db, "code_list.txt", decks)
    Spec, _, _, Gymnasium = py_env(P._YGOProEnvSpec, P._YGOProEnvPool)
    penv = Gymnasium(Spec(Spec.gen_config(gym_reset_return_info=True, **cfg)))
    renv = None
    if with_ref:
        init_ygopro("YGOPro-v1", "english", deck_dir, "code_list.txt")
        renv = ygoenv.make(task_id="YGOPro-v1", env_type="gymnasium", **cfg)
    return P, penv, renv, db, decks


def run(deck_dir="~/ygo/decks/mine", deck1="random", deck2="random", games=20, seed=0, view="env",
        ygoinf_dir=None, with_ref=True, max_examples=5, log=print):
    import numpy as np
    deck_dir = os.path.abspath(os.path.expanduser(deck_dir))
    sys.path.insert(0, str(Path(ygoinf_dir).resolve() if ygoinf_dir else ROOT / "ygoinf"))
    P, penv, renv, db, decks = make_envs(deck_dir, deck1, deck2, seed, with_ref)
    from ygoinf import features as F
    from ygoinf import models as Mo
    F.code_to_id.clear()
    F.init_code_list("code_list.txt")
    if hasattr(F, "init_announce"):
        F.init_announce(db, [deck_dir])
    card_data = load_card_data(db, [c for c in F.code_to_id if c != 0])

    client = RefClient(F, Mo, view, card_data)
    rng = np.random.default_rng(seed)
    KEYS = ("cards_", "global_", "actions_", "h_actions_")
    col_diff = Counter()       # (key, feature) -> decisions with a difference
    examples = defaultdict(list)
    by_msg = Counter()
    by_msg_bad = Counter()
    harness_diff = 0
    compared = 0
    errors = Counter()
    announce = Counter()       # code of the last chaining card -> announce decisions compared
    announce_ok = Counter()
    last_chaining = [0]
    desync_skip = False

    def ref_check(obs, info, robs, rinfo):
        nonlocal harness_diff
        if renv is None:
            return
        same = all(np.array_equal(obs[k], robs[k]) for k in KEYS) and \
            np.array_equal(info["num_options"], rinfo["num_options"]) and np.array_equal(info["to_play"], rinfo["to_play"])
        if not same:
            harness_diff += 1

    obs, info = penv.reset()
    if renv is not None:
        robs, rinfo = renv.reset()
        ref_check(obs, info, robs, rinfo)
    done_games = 0
    steps = 0
    after_terminal = False
    while done_games < games:
        recs = P.drain()
        for kind, data in recs:
            if kind == 1:
                for m, b in walk(data):
                    if m == M["CHAINING"]:
                        last_chaining[0] = u32(b, 0)
        # the env's discard_hand_ flag at the time of each select_card is
        # computed from the env-view state when the decision is queued
        client.feed(recs)
        nopt = int(info["num_options"][0])
        a = int(rng.integers(0, 1 << 30) % max(nopt, 1))
        if not desync_skip and not after_terminal:
            res = resolve(client, F, obs, info, KEYS, col_diff, examples, by_msg, by_msg_bad, errors,
                          announce, announce_ok, last_chaining[0], max_examples, a)
            if res == "desync":
                desync_skip = True
            elif res == "compared":
                compared += 1
        obs, rew, term, trunc, info = penv.step(np.array([a], dtype=np.int32))
        if renv is not None:
            robs, rrew, rterm, rtrunc, rinfo = renv.step(np.array([a], dtype=np.int32))
            ref_check(obs, info, robs, rinfo)
        steps += 1
        after_terminal = bool(term[0] or trunc[0])
        if after_terminal:
            done_games += 1
            desync_skip = False
    return dict(view=view, games=games, steps=steps, compared=compared, harness_diff=harness_diff,
                col_diff={f"{k}.{f}": n for (k, f), n in sorted(col_diff.items())},
                by_msg=dict(by_msg), by_msg_bad=dict(by_msg_bad), errors=dict(errors),
                stats=dict(client.stats), announce=dict(announce), announce_ok=dict(announce_ok),
                examples={f"{k}.{f}": v for (k, f), v in examples.items()})


def resolve(client, F, obs, info, KEYS, col_diff, examples, by_msg, by_msg_bad, errors,
            announce, announce_ok, chaining_code, max_examples, env_action):
    """Walk the client's decisions until the one the env is waiting for,
    compare it, and apply the env's chosen action."""
    import numpy as np
    to_play = int(info["to_play"][0])
    while True:
        if client.cur is None:
            if not client.queue:
                errors["client_has_no_pending_decision"] += 1
                if os.environ.get("PARITY_DEBUG"):
                    print("DEBUG no pending: env msg ids", obs["actions_"][0][:, 3].tolist()[:8],
                          "nopt", int(info["num_options"][0]), "to_play", to_play,
                          "last", client.debug_last)
                return "desync"
            d = client.queue.pop(0)
            d["selected"] = []
            d["discard_hint"] = d["m"] == M["SELECT_CARD"] and d["st"][0].discard_hand
            client.cur = d
        d = client.cur
        client.debug_last = (MSG_NAME[d["m"]], list(d["selected"]))
        if client.env_auto(d):
            client.cur = None
            continue
        last = not client.queue
        name = MSG_NAME[d["m"]]
        captured = []
        try:
            p, inp, res = client.call(d, captured)
        except Exception as e:  # unsupported in this ygoinf
            errors[f"{name}: {type(e).__name__}: {str(e)[:80]}"] += 1
            return "desync"
        preds = res.action_preds
        valid = [j for j, pr in enumerate(preds) if pr.prob != -1]
        discard = d.get("discard_hint") and d["st"][0].phase == PHASE_END and d["m"] == M["SELECT_CARD"]
        if discard:
            # env picks at random: no observation, no history. Pick like the env
            # can't be reproduced; the field state follows from the snapshots.
            client.prev_idx[p] = valid[0]
            client.cur = None
            continue
        n = len(valid)
        if not captured or (not last):
            if captured:
                errors[f"{name}: client shows {n} options where the env decided alone"] += 1
                return "desync"
            client.prev_idx[p] = valid[0]
            if advance(client, d, inp, preds, valid[0]):
                client.cur = None
            continue
        # the decision the env shows the model
        if p != to_play:
            errors["to_play differs"] += 1
            return "desync"
        x = captured[0] if captured else None
        by_msg[name] += 1
        bad = False
        if x is None:
            errors[f"{name}: ygoinf did not call the model"] += 1
            if os.environ.get("PARITY_DEBUG"):
                print("DEBUG no model call", name, "selected", d["selected"], "n", n, "ncards", len(inp.action_msg.data.cards),
                      [(pr.prob, pr.response) for pr in preds])
            return "desync"
        if min(n, 24) != int(info["num_options"][0]):
            col_diff[("info", "num_options")] += 1
            bad = True
        for k in KEYS:
            a, b = np.asarray(x[k]), obs[k][0]
            diff = a != b
            if diff.any():
                cols = np.nonzero(diff.reshape(-1, diff.shape[-1]).any(0))[0] if diff.ndim > 1 else np.nonzero(diff)[0]
                for c in cols:
                    f = FEATURES[k][c]
                    col_diff[(k, f)] += 1
                    if len(examples[(k, f)]) < max_examples:
                        rows = np.nonzero(diff[:, c])[0].tolist() if diff.ndim > 1 else []
                        examples[(k, f)].append(dict(msg=name, rows=rows[:6],
                                                     ygoinf=(a[:, c] if diff.ndim > 1 else a)[rows[:6]].tolist() if rows else int(a[c]),
                                                     env=(b[:, c] if diff.ndim > 1 else b)[rows[:6]].tolist() if rows else int(b[c])))
                bad = True
        if bad:
            by_msg_bad[name] += 1
            if os.environ.get("PARITY_DEBUG") and client.stats["debug_prints"] < int(os.environ.get("PARITY_DEBUG")):
                client.stats["debug_prints"] += 1
                print("DEBUG mismatch", name, "selected", d["selected"], "nopt", int(info["num_options"][0]), "n", n)
                for k in ("actions_", "h_actions_"):
                    a_, b_ = np.asarray(x[k]), obs[k][0]
                    rows = np.nonzero((a_ != b_).any(1))[0][:4]
                    for r in rows:
                        print(f"   {k}[{r}] ygoinf {a_[r].tolist()}")
                        print(f"   {k}[{r}] env    {b_[r].tolist()}")
                if d["m"] == M["ANNOUNCE_CARD"]:
                    print("   seen", inp.action_msg.data.seen, "opcodes", inp.action_msg.data.opcodes)
        if d["m"] == M["ANNOUNCE_CARD"]:
            announce[chaining_code] += 1
            if not bad:
                announce_ok[chaining_code] += 1
        j = valid[env_action] if env_action < len(valid) else valid[0]
        client.prev_idx[p] = j
        if advance(client, d, inp, preds, j):
            client.cur = None
        return "compared"


def advance(client, d, inp, preds, j):
    """Apply the choice of pred j to decision d. True when d is finished."""
    m = d["m"]
    if m in (M["SELECT_CARD"], M["SELECT_TRIBUTE"]):
        r = preds[j].response
        if r == -1:
            return True
        d["selected"].append(r)
        return len(d["selected"]) >= inp.action_msg.data.max
    if m == M["SELECT_SUM"]:
        if preds[j].can_finish:
            return True
        d["selected"].append(preds[j].response)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="~/ygo/decks/mine")
    ap.add_argument("--deck1", default="random")
    ap.add_argument("--deck2", default="random")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", default="env", choices=["env", "client", "client-deck"])
    ap.add_argument("--ygoinf", default=None, help="folder that contains the ygoinf package (default: this checkout)")
    ap.add_argument("--no-ref", action="store_true", help="skip the comparison with ygopro_ygoenv")
    ap.add_argument("--json")
    a = ap.parse_args()
    r = run(a.deck, a.deck1, a.deck2, a.games, a.seed, a.view, a.ygoinf, not a.no_ref)
    report(r)
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))
    ok = r["harness_diff"] == 0 and not r["col_diff"] and not r["errors"]
    sys.exit(0 if ok or a.view != "env" else 1)


def report(r):
    print(f"view {r['view']}: {r['games']} games, {r['steps']} env steps, {r['compared']} decisions compared")
    print(f"recording build vs ygopro_ygoenv: {r['harness_diff']} steps differ")
    print("decisions per message (compared / with a difference):")
    for k in sorted(r["by_msg"]):
        print(f"  {k:22s} {r['by_msg'][k]:6d} {r['by_msg_bad'].get(k, 0):6d}")
    print("features with a difference (decisions):" + ("" if r["col_diff"] else " none"))
    for k, n in r["col_diff"].items():
        print(f"  {k:28s} {n:6d}  e.g. {r['examples'].get(k, [''])[0]}")
    if r["errors"]:
        print("errors (each stops the comparison until the next game):")
        for k, n in r["errors"].items():
            print(f"  {n:5d}  {k}")
    if r["stats"]:
        print("stream statistics:", r["stats"])
    if r["announce"]:
        print("card-name declarations (by the card that started the chain): compared / equal")
        for k, n in r["announce"].items():
            print(f"  {k}: {n} / {r['announce_ok'].get(k, 0)}")


if __name__ == "__main__":
    main()
