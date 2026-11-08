// Regression test for the training crash of 2026-11-08 (segfault in
// std::_Rb_tree_rebalance_for_erase inside end_duel): ygopro-core's global duel
// registry (ocgapi.cpp duel_set) is modified by create_duel_v2 / end_duel
// without a lock, and each env runs on its own thread.
//
//   duel_registry_stress locked   [threads] [iterations]  -> must finish, exit 0
//   duel_registry_stress unlocked [threads] [iterations]  -> calls the core
//       directly, as the env did; expected to crash (reproduces the bug)
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "ygoenv/ygopro/duel_registry.h"
#include "ygopro-core/card_data.h"

static uint8_t *no_script(const char *, int *) { return nullptr; }  // keep duels cheap
static uint32_t blank_card(uint32_t, card_data *d) { d->clear(); return 0; }
static uint32_t quiet(intptr_t, uint32_t) { return 0; }

int main(int argc, char **argv) {
  if (argc < 2 || (std::strcmp(argv[1], "locked") && std::strcmp(argv[1], "unlocked"))) {
    std::fprintf(stderr, "usage: %s locked|unlocked [threads] [iterations]\n", argv[0]);
    return 2;
  }
  const bool locked = !std::strcmp(argv[1], "locked");
  const int threads = argc > 2 ? std::atoi(argv[2]) : 32;
  const int iters = argc > 3 ? std::atoi(argv[3]) : 2000;
  set_script_reader(no_script);
  set_card_reader(blank_card);
  set_message_handler(quiet);

  std::atomic<long> done{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      for (int i = 0; i < iters; ++i) {
        uint32_t seq[SEED_COUNT];
        for (int k = 0; k < SEED_COUNT; ++k) seq[k] = static_cast<uint32_t>(t * 1000003 + i * 7 + k);
        intptr_t pd = locked ? ygo_registry::create_duel(seq) : create_duel_v2(seq);
        if (locked) ygo_registry::end_duel(pd); else end_duel(pd);
        ++done;
      }
    });
  }
  for (auto &th : pool) th.join();
  std::printf("%s: %ld duels created and ended on %d threads without a crash\n",
              locked ? "locked" : "unlocked", done.load(), threads);
  return 0;
}
