// Replay a YGOPro .yrp through ygopro-core the way the reference client does
// (Fluorohydride/ygopro gframe/replay.cpp OpenReplay/ReadInfo and
// replay_mode.cpp StartDuel/ReplayAnalyze) and check that it is complete.
//
//   yrp_verify <cards.cdb> <script_dir> <file.yrp> [...]
//
// PASS: the duel reaches MSG_WIN, every recorded response has been used at
// that point, and the core never answered MSG_RETRY (a rejected response).
// Card data is built like the env's loader (rule_code split, alias
// inheritance, extra set codes). Exit code 0 only if every file passes.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <lzma.h>
#include <sqlite3.h>

#include "ygopro-core/card_data.h"
#include "ygopro-core/common.h"
#include "ygopro-core/ocgapi.h"

// client-only message ids, commented out in the core's common.h (never emitted)
#ifndef MSG_REFRESH_DECK
#define MSG_REFRESH_DECK 34
#endif
#ifndef MSG_CARD_SELECTED
#define MSG_CARD_SELECTED 80
#endif
#ifndef MSG_UNEQUIP
#define MSG_UNEQUIP 95
#endif

#define REPLAY_COMPRESSED 0x1
#define REPLAY_TAG 0x2
#define REPLAY_SINGLE_MODE 0x8
#define REPLAY_UNIFORM 0x10
#define REPLAY_ID_YRP1 0x31707279
#define REPLAY_ID_YRP2 0x32707279

struct ReplayHeader {
  uint32_t id{}, version{}, flag{}, seed{}, datasize{}, start_time{};
  uint8_t props[8]{};
};
struct ExtendedReplayHeader {
  ReplayHeader base;
  uint32_t seed_sequence[SEED_COUNT]{};
  uint8_t header_version{1};
  uint8_t value[15]{};
};
constexpr size_t MAX_REPLAY_SIZE = 0x80000;
constexpr size_t MAX_COMP_SIZE = UINT16_MAX - 1 - sizeof(ExtendedReplayHeader);

// ---------------------------------------------------------------- card data
static sqlite3 *g_db = nullptr;
static std::string g_script_dir;
static std::vector<char> g_script_buf;

static bool query_card(uint32_t code, card_data *d) {
  sqlite3_stmt *st = nullptr;
  sqlite3_prepare_v2(g_db, "SELECT alias, setcode, type, level, atk, def, race, attribute FROM datas WHERE id=?",
                     -1, &st, nullptr);
  sqlite3_bind_int64(st, 1, code);
  bool ok = sqlite3_step(st) == SQLITE_ROW;
  if (ok) {
    d->clear();
    d->code = code;
    d->alias = (uint32_t)sqlite3_column_int64(st, 0);
    write_setcode(d->setcode, (uint64_t)sqlite3_column_int64(st, 1));
    d->type = (uint32_t)sqlite3_column_int64(st, 2);
    auto is_alternative = [](uint32_t c, uint32_t a) {
      return a && (a < c + 20) && (c < a + 20);
    };
    if (code == 5405695 || (d->alias && !(d->type & TYPE_TOKEN) && !is_alternative(code, d->alias))) {
      d->rule_code = d->alias;
      d->alias = 0;
    }
    static const std::map<uint32_t, std::vector<uint16_t>> extra_setcode = {
        {8512558u, {0x8f, 0x54, 0x59, 0x82, 0x13a}},
        {55088578u, {0x8f, 0x54, 0x59, 0x82, 0x13a}},
    };
    auto es = extra_setcode.find(code);
    if (es != extra_setcode.end()) std::copy(es->second.begin(), es->second.end(), d->setcode);
    uint32_t level = (uint32_t)sqlite3_column_int64(st, 3);
    d->level = level & 0xff;
    d->lscale = (level >> 24) & 0xff;
    d->rscale = (level >> 16) & 0xff;
    d->attack = sqlite3_column_int(st, 4);
    d->defense = sqlite3_column_int(st, 5);
    if (d->type & TYPE_LINK) {
      d->link_marker = d->defense;
      d->defense = 0;
    }
    d->race = (uint32_t)sqlite3_column_int64(st, 6);
    d->attribute = (uint32_t)sqlite3_column_int64(st, 7);
  }
  sqlite3_finalize(st);
  return ok;
}

static uint32_t card_reader_cb(uint32_t code, card_data *d) {
  if (!query_card(code, d)) {
    d->clear();
    return 0;
  }
  // alternative artworks inherit the base card's rule code (env init_module)
  if (!d->rule_code && d->alias && !(d->type & TYPE_TOKEN)) {
    card_data base;
    if (query_card(d->alias, &base)) d->rule_code = base.rule_code;
  }
  return 0;
}

static uint8_t *script_reader_cb(const char *name, int *len) {
  std::string n(name);
  const std::string prefix = "./script/";
  std::string rel = n.rfind(prefix, 0) == 0 ? n.substr(prefix.size()) : n;
  std::ifstream f(g_script_dir + "/" + rel, std::ios::binary);
  if (!f) return nullptr;
  g_script_buf.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  *len = (int)g_script_buf.size();
  return reinterpret_cast<uint8_t *>(g_script_buf.data());
}

static int g_lua_errors = 0;
static uint32_t message_handler_cb(intptr_t pduel, uint32_t type) {
  char buf[1024] = {0};
  get_log_message(pduel, buf);
  if (type == 1) {
    ++g_lua_errors;
    std::fprintf(stderr, "    [lua-error] %s\n", buf);
  }
  return 0;
}

// ---------------------------------------------------------------- replay file
struct Replay {
  ExtendedReplayHeader h;
  std::vector<uint8_t> data;
  size_t pos = 0, info_end = 0;
  std::vector<std::vector<uint32_t>> main, extra;
  int32_t start_lp = 0, start_hand = 0, draw_count = 0;
  uint32_t duel_flag = 0;
  bool read(void *dst, size_t n) {
    if (n > data.size() - pos) return false;
    std::memcpy(dst, data.data() + pos, n);
    pos += n;
    return true;
  }
  template <class T> T get() { T v{}; read(&v, sizeof v); return v; }
};

static std::string open_replay(const char *path, Replay &r) {
  FILE *fp = std::fopen(path, "rb");
  if (!fp) return "cannot open";
  if (std::fread(&r.h.base, sizeof r.h.base, 1, fp) < 1) { std::fclose(fp); return "short header"; }
  auto &b = r.h.base;
  if (b.id != REPLAY_ID_YRP1 && b.id != REPLAY_ID_YRP2) { std::fclose(fp); return "bad id"; }
  if (b.version < 0x12d0u) { std::fclose(fp); return "version too old"; }
  if (b.version >= 0x1353u && !(b.flag & REPLAY_UNIFORM)) { std::fclose(fp); return "missing UNIFORM flag"; }
  if (b.id == REPLAY_ID_YRP2 &&
      std::fread(reinterpret_cast<uint8_t *>(&r.h) + sizeof r.h.base, sizeof r.h - sizeof r.h.base, 1, fp) < 1) {
    std::fclose(fp);
    return "short extended header";
  }
  if (b.flag & REPLAY_COMPRESSED) {
    std::vector<uint8_t> comp(MAX_COMP_SIZE);
    size_t comp_size = std::fread(comp.data(), 1, comp.size(), fp);
    std::fclose(fp);
    if (b.datasize > MAX_REPLAY_SIZE) return "datasize too large";
    r.data.resize(b.datasize);
    lzma_filter filter;
    filter.id = LZMA_FILTER_LZMA1;
    filter.options = nullptr;
    if (lzma_properties_decode(&filter, nullptr, b.props, 5) != LZMA_OK) return "bad LZMA props";
    auto *opt = static_cast<lzma_options_lzma *>(filter.options);
    lzma_set_ext_size(*opt, b.datasize);
    opt->ext_flags = LZMA_LZMA1EXT_ALLOW_EOPM;
    filter.id = LZMA_FILTER_LZMA1EXT;
    lzma_filter filters[2] = {filter, {LZMA_VLI_UNKNOWN, nullptr}};
    size_t in_pos = 0, out_pos = 0;
    lzma_ret lret = lzma_raw_buffer_decode(filters, nullptr, comp.data(), &in_pos, comp_size, r.data.data(),
                                           &out_pos, r.data.size());
    lzma_filters_free(filters, nullptr);
    if (lret != LZMA_OK || out_pos != b.datasize) return "LZMA decode failed";
  } else {
    r.data.resize(MAX_REPLAY_SIZE);
    r.data.resize(std::fread(r.data.data(), 1, MAX_REPLAY_SIZE, fp));
    std::fclose(fp);
  }
  // ReadInfo (two players, no tag, no single mode)
  if (b.flag & (REPLAY_TAG | REPLAY_SINGLE_MODE)) return "tag/single mode not supported by this verifier";
  uint16_t name[20];
  for (int i = 0; i < 2; ++i)
    if (!r.read(name, sizeof name)) return "short names";
  r.start_lp = r.get<int32_t>();
  r.start_hand = r.get<int32_t>();
  r.draw_count = r.get<int32_t>();
  r.duel_flag = r.get<uint32_t>();
  for (int p = 0; p < 2; ++p) {
    std::vector<uint32_t> m(r.get<uint32_t>());
    if (m.size() > 60 || !r.read(m.data(), m.size() * 4)) return "bad main deck";
    std::vector<uint32_t> e(r.get<uint32_t>());
    if (e.size() > 60 || !r.read(e.data(), e.size() * 4)) return "bad extra deck";
    r.main.push_back(m);
    r.extra.push_back(e);
  }
  r.info_end = r.pos;
  return "";
}

// ---------------------------------------------------------------- playback
enum class Step { Continue, Win, Retry, OutOfResponses, Unknown };

struct Player {
  intptr_t pduel;
  Replay &r;
  int responses = 0, turns = 0;
  int last_msg = 0;
  uint8_t winner = 255, reason = 0;

  bool respond() {
    uint8_t len;
    uint8_t resp[SIZE_RETURN_VALUE] = {0};
    if (!r.read(&len, 1) || !r.read(resp, len)) return false;
    if (std::getenv("YRP_DEBUG")) {
      std::printf("    response %d to msg %d:", responses + 1, last_msg);
      for (int i = 0; i < len; ++i) std::printf(" %02x", resp[i]);
      std::printf("\n");
    }
    set_responseb(pduel, resp);
    ++responses;
    return true;
  }

  // message walk of ReplayMode::ReplayAnalyze (client display calls dropped)
  Step analyze(uint8_t *msg, uint32_t len) {
    uint8_t *p = msg;
    auto u8 = [&]() { return *p++; };
    int count;
    while (p - msg < (int)len) {
      int m = u8();
      last_msg = m;
      switch (m) {
      case MSG_RETRY: return Step::Retry;
      case MSG_HINT: p += 6; break;
      case MSG_WIN: winner = p[0]; reason = p[1]; p += 2; return Step::Win;
      case MSG_SELECT_BATTLECMD:
        u8(); count = u8(); p += count * 11; count = u8(); p += count * 8 + 2;
        return respond() ? Step::Continue : Step::OutOfResponses;
      case MSG_SELECT_IDLECMD:
        u8();
        for (int i = 0; i < 5; ++i) { count = u8(); p += count * 7; }
        count = u8(); p += count * 11 + 3;
        return respond() ? Step::Continue : Step::OutOfResponses;
      case MSG_SELECT_EFFECTYN: u8(); p += 12; return respond() ? Step::Continue : Step::OutOfResponses;
      case MSG_SELECT_YESNO: u8(); p += 4; return respond() ? Step::Continue : Step::OutOfResponses;
      case MSG_SELECT_OPTION: u8(); count = u8(); p += count * 4; return respond() ? Step::Continue : Step::OutOfResponses;
      case MSG_SELECT_CARD:
      case MSG_SELECT_TRIBUTE: u8(); p += 3; count = u8(); p += count * 8; return respond() ? Step::Continue : Step::OutOfResponses;
      case MSG_SELECT_UNSELECT_CARD:
        u8(); p += 4; count = u8(); p += count * 8; count = u8(); p += count * 8;
        return respond() ? Step::Continue : Step::OutOfResponses;
      case MSG_SELECT_CHAIN: u8(); count = u8(); p += 9 + count * 14; return respond() ? Step::Continue : Step::OutOfResponses;
      case MSG_SELECT_PLACE:
      case MSG_SELECT_DISFIELD:
      case MSG_SELECT_POSITION: u8(); p += 5; return respond() ? Step::Continue : Step::OutOfResponses;
      case MSG_SELECT_COUNTER: u8(); p += 4; count = u8(); p += count * 9; return respond() ? Step::Continue : Step::OutOfResponses;
      case MSG_SELECT_SUM:
        p++; u8(); p += 6; count = u8(); p += count * 11; count = u8(); p += count * 11;
        return respond() ? Step::Continue : Step::OutOfResponses;
      case MSG_SORT_CARD: u8(); count = u8(); p += count * 7; return respond() ? Step::Continue : Step::OutOfResponses;
      case MSG_CONFIRM_DECKTOP:
      case MSG_CONFIRM_EXTRATOP: u8(); count = u8(); p += count * 7; break;
      case MSG_CONFIRM_CARDS: u8(); p += 1; count = u8(); p += count * 7; break;
      case MSG_SHUFFLE_DECK: u8(); break;
      case MSG_SHUFFLE_HAND:
      case MSG_SHUFFLE_EXTRA: u8(); count = u8(); p += count * 4; break;
      case MSG_REFRESH_DECK: p++; break;
      case MSG_SWAP_GRAVE_DECK: u8(); break;
      case MSG_REVERSE_DECK: break;
      case MSG_DECK_TOP: p += 6; break;
      case MSG_SHUFFLE_SET_CARD: p++; count = u8(); p += count * 8; break;
      case MSG_NEW_TURN: u8(); ++turns; break;
      case MSG_NEW_PHASE: p += 2; break;
      case MSG_MOVE: p += 16; break;
      case MSG_POS_CHANGE: p += 9; break;
      case MSG_SET: p += 8; break;
      case MSG_SWAP: p += 16; break;
      case MSG_FIELD_DISABLED: p += 4; break;
      case MSG_SUMMONING:
      case MSG_SPSUMMONING:
      case MSG_FLIPSUMMONING: p += 8; break;
      case MSG_SUMMONED:
      case MSG_SPSUMMONED:
      case MSG_FLIPSUMMONED: break;
      case MSG_CHAINING: p += 16; break;
      case MSG_CHAINED:
      case MSG_CHAIN_SOLVING:
      case MSG_CHAIN_SOLVED:
      case MSG_CHAIN_NEGATED:
      case MSG_CHAIN_DISABLED: p++; break;
      case MSG_CHAIN_END: break;
      case MSG_CARD_SELECTED:
      case MSG_RANDOM_SELECTED: u8(); count = u8(); p += count * 4; break;
      case MSG_BECOME_TARGET: count = u8(); p += count * 4; break;
      case MSG_DRAW: u8(); count = u8(); p += count * 4; break;
      case MSG_DAMAGE:
      case MSG_RECOVER:
      case MSG_LPUPDATE:
      case MSG_PAY_LPCOST: p += 5; break;
      case MSG_EQUIP: p += 8; break;
      case MSG_UNEQUIP: p += 4; break;
      case MSG_CARD_TARGET:
      case MSG_CANCEL_TARGET: p += 8; break;
      case MSG_ADD_COUNTER:
      case MSG_REMOVE_COUNTER: p += 7; break;
      case MSG_ATTACK: p += 8; break;
      case MSG_BATTLE: p += 26; break;
      case MSG_ATTACK_DISABLED:
      case MSG_DAMAGE_STEP_START:
      case MSG_DAMAGE_STEP_END: break;
      case MSG_MISSED_EFFECT: p += 8; break;
      case MSG_TOSS_COIN:
      case MSG_TOSS_DICE: u8(); count = u8(); p += count; break;
      case MSG_ROCK_PAPER_SCISSORS: u8(); return respond() ? Step::Continue : Step::OutOfResponses;
      case MSG_HAND_RES: p += 1; break;
      case MSG_ANNOUNCE_RACE:
      case MSG_ANNOUNCE_ATTRIB: u8(); p += 5; return respond() ? Step::Continue : Step::OutOfResponses;
      case MSG_ANNOUNCE_CARD:
      case MSG_ANNOUNCE_NUMBER: u8(); count = u8(); p += 4 * count; return respond() ? Step::Continue : Step::OutOfResponses;
      case MSG_CARD_HINT: p += 9; break;
      case MSG_PLAYER_HINT: p += 6; break;
      case MSG_MATCH_KILL: p += 4; break;
      case MSG_TAG_SWAP: p += p[2] * 4 + p[4] * 4 + 9; break;
      case MSG_RELOAD_FIELD:
        p++;
        for (int pl = 0; pl < 2; ++pl) {
          p += 4;
          for (int s = 0; s < 7; ++s) if (u8()) p += 2;
          for (int s = 0; s < 8; ++s) if (u8()) p++;
          p += 6;
        }
        p++;
        break;
      case MSG_AI_NAME:
      case MSG_SHOW_HINT: { uint16_t l; std::memcpy(&l, p, 2); p += 2 + l + 1; break; }
      default: return Step::Unknown;
      }
    }
    return Step::Continue;
  }
};

static bool verify(const char *path) {
  Replay r;
  std::string err = open_replay(path, r);
  const auto &b = r.h.base;
  std::printf("%s\n", path);
  if (!err.empty()) { std::printf("  FAIL: %s\n", err.c_str()); return false; }
  std::printf("  id=%s version=0x%x flag=0x%x datasize=%u decks %zu/%zu vs %zu/%zu\n",
              b.id == REPLAY_ID_YRP2 ? "yrp2" : "yrp1", b.version, b.flag, b.datasize,
              r.main[0].size(), r.extra[0].size(), r.main[1].size(), r.extra[1].size());
  intptr_t pduel;
  if (b.id == REPLAY_ID_YRP1) {
    std::mt19937 rnd(b.seed);
    pduel = create_duel(rnd());
  } else {
    pduel = create_duel_v2(r.h.seed_sequence);
  }
  set_player_info(pduel, 0, r.start_lp, r.start_hand, r.draw_count);
  set_player_info(pduel, 1, r.start_lp, r.start_hand, r.draw_count);
  for (int i = 0; i < 2; ++i) {
    for (auto c : r.main[i]) new_card(pduel, c, i, i, LOCATION_DECK, 0, POS_FACEDOWN_DEFENSE);
    for (auto c : r.extra[i]) new_card(pduel, c, i, i, LOCATION_EXTRA, 0, POS_FACEDOWN_DEFENSE);
  }
  start_duel(pduel, r.duel_flag);
  r.pos = r.info_end;
  size_t body = r.data.size() - r.info_end;

  Player pl{pduel, r};
  std::vector<uint8_t> buf(SIZE_MESSAGE_BUFFER);
  Step st = Step::Continue;
  for (int iter = 0; iter < 2000000 && st == Step::Continue; ++iter) {
    uint32_t res = process(pduel);
    uint32_t len = res & PROCESSOR_BUFFER_LEN;
    if (len) {
      if (len > buf.size()) buf.resize(len);
      get_message(pduel, buf.data());
      st = pl.analyze(buf.data(), len);
    } else if (res & PROCESSOR_END) {
      break;
    }
  }
  end_duel(pduel);
  size_t left = r.data.size() - r.pos;
  bool pass = st == Step::Win && left == 0;
  const char *why = st == Step::Win ? (left ? "MSG_WIN reached but responses left over" : "")
                    : st == Step::Retry ? "core rejected a response (MSG_RETRY)"
                    : st == Step::OutOfResponses ? "ran out of responses before MSG_WIN (not fully recorded)"
                    : st == Step::Unknown ? "unknown message in stream"
                    : "duel ended without MSG_WIN";
  std::printf("  %s: %d responses used, %zu of %zu body bytes left, %d turns, winner %d (reason %d), "
              "last msg %d%s%s\n",
              pass ? "PASS" : "FAIL", pl.responses, left, body, pl.turns, pl.winner, pl.reason, pl.last_msg,
              *why ? " - " : "", why);
  return pass;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <cards.cdb> <script_dir> <file.yrp> [...]\n", argv[0]);
    return 2;
  }
  if (sqlite3_open_v2(argv[1], &g_db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }
  g_script_dir = argv[2];
  set_card_reader(card_reader_cb);
  set_script_reader(script_reader_cb);
  set_message_handler(message_handler_cb);
  int fails = 0;
  for (int i = 3; i < argc; ++i) fails += !verify(argv[i]);
  std::printf("%d of %d replays passed%s\n", argc - 3 - fails, argc - 3,
              g_lua_errors ? " (Lua errors were logged)" : "");
  sqlite3_close(g_db);
  return fails ? 1 : 0;
}
