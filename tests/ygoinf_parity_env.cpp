// Test-only build of the YGOPro env that also records what a game client sees.
//
// The env's Python interface returns only the encoded observation. The ygoinf
// parity test (tools/ygoinf_parity.py) needs the raw information a client
// would have at each decision: the core's message stream and the state of
// every card. This module compiles the unchanged ygoenv/ygopro/ygopro.h and
// interposes two ygopro-core calls with the preprocessor:
//
//   get_message  -> record the message buffer, then a query of every location
//                   of both players (the field state at the end of the buffer)
//   start_duel   -> record a duel boundary
//
// Nothing in the env's control flow changes. The parity test checks that this
// build produces the same observations as the real ygopro_ygoenv module for
// the same seed before it compares anything else.
//
// drain() returns the records since the last call as a list of (kind, bytes):
//   kind 0: duel start
//   kind 1: message buffer as returned by get_message
//   kind 2: field snapshot, repeated {u8 player, u8 location, u32 len, len bytes
//           of query_field_card(player, location, kSnapshotFlags)}
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "ygopro-core/common.h"
#include "ygopro-core/card_data.h"
#include "ygopro-core/duel.h"
#include "ygopro-core/ocgapi.h"
#include "ygopro-core/field.h"
#include "ygopro-core/effect.h"
#include "ygopro-core/card.h"
#include "ygoenv/ygopro/duel_registry.h"

namespace ygo_parity {

// Same flags as the env's get_cards_in_location.
constexpr uint32_t kSnapshotFlags =
    QUERY_CODE | QUERY_POSITION | QUERY_LEVEL | QUERY_RANK | QUERY_ATTACK |
    QUERY_DEFENSE | QUERY_EQUIP_CARD | QUERY_OVERLAY_CARD | QUERY_COUNTERS |
    QUERY_STATUS | QUERY_LSCALE | QUERY_RSCALE | QUERY_LINK;

inline std::mutex g_mutex;
inline std::vector<std::pair<int, std::string>> g_log;
inline bool g_enabled = true;

inline void record(int kind, const void *data, size_t n) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_enabled) {
    g_log.emplace_back(kind, std::string(static_cast<const char *>(data), n));
  }
}

inline void snapshot(intptr_t pduel) {
  static thread_local std::vector<uint8_t> buf(0x20000);
  std::string out;
  for (uint8_t p = 0; p < 2; ++p) {
    for (uint8_t loc : {LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE,
                        LOCATION_SZONE, LOCATION_GRAVE, LOCATION_REMOVED,
                        LOCATION_EXTRA}) {
      int32_t n = query_field_card(pduel, p, loc, kSnapshotFlags, buf.data(), 0);
      uint32_t len = n > 0 ? static_cast<uint32_t>(n) : 0;
      out.push_back(static_cast<char>(p));
      out.push_back(static_cast<char>(loc));
      out.append(reinterpret_cast<const char *>(&len), 4);
      out.append(reinterpret_cast<const char *>(buf.data()), len);
    }
  }
  record(2, out.data(), out.size());
}

}  // namespace ygo_parity

inline int32_t parity_get_message(intptr_t pduel, uint8_t *buf) {
  int32_t n = ::get_message(pduel, buf);
  ygo_parity::record(1, buf, n > 0 ? static_cast<size_t>(n) : 0);
  ygo_parity::snapshot(pduel);
  return n;
}

inline void parity_start_duel(intptr_t pduel, uint32_t options) {
  ygo_parity::record(0, "", 0);
  ::start_duel(pduel, options);
}

#define get_message parity_get_message
#define start_duel parity_start_duel
// pybind11 registers types by C++ type across modules: give the env classes a
// different namespace so this module can be loaded next to ygopro_ygoenv.
#define ygopro ygopro_parity_env
#include "ygoenv/ygopro/ygopro.h"
#undef get_message
#undef start_duel

#include "ygoenv/core/py_envpool.h"

using YGOProEnvSpec = PyEnvSpec<ygopro::YGOProEnvSpec>;  // ygopro_parity_env::
using YGOProEnvPool = PyEnvPool<ygopro::YGOProEnvPool>;

PYBIND11_MODULE(ygopro_parity, m) {
  REGISTER(m, YGOProEnvSpec, YGOProEnvPool)

  m.def("init_module", &ygopro::init_module);
  m.def("drain", []() {
    std::vector<std::pair<int, std::string>> log;
    {
      std::lock_guard<std::mutex> lock(ygo_parity::g_mutex);
      log.swap(ygo_parity::g_log);
    }
    py::list out;
    for (auto &[kind, data] : log) {
      out.append(py::make_tuple(kind, py::bytes(data)));
    }
    return out;
  });
  m.def("set_enabled", [](bool on) {
    std::lock_guard<std::mutex> lock(ygo_parity::g_mutex);
    ygo_parity::g_enabled = on;
  });
}
