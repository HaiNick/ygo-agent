#pragma once
// ygopro-core keeps every duel in one process-wide std::set (ocgapi.cpp:
// duel_set). create_duel / create_duel_v2 insert into it and end_duel erases
// from it, without any lock. Each env runs on its own thread, so concurrent
// calls corrupt the tree (segfault in std::_Rb_tree_rebalance_for_erase inside
// end_duel). All duel creation and destruction goes through this lock.
#include <cstdint>
#include <mutex>

#include "ygopro-core/common.h"
#include "ygopro-core/ocgapi.h"

namespace ygo_registry {

inline std::mutex g_mutex;

inline intptr_t create_duel(uint32_t (&seed_sequence)[SEED_COUNT]) {
  std::lock_guard<std::mutex> lock(g_mutex);
  return create_duel_v2(seed_sequence);
}

inline void end_duel(intptr_t pduel) {
  std::lock_guard<std::mutex> lock(g_mutex);
  ::end_duel(pduel);
}

}  // namespace ygo_registry
