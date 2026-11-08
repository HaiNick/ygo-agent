// Run a Lua file inside a ygopro-core duel and print its Debug.Message output.
//
//   core_lua_probe <script_dir> <lua_file>
//
// <script_dir> provides constant.lua / utility.lua / procedure.lua, which the
// core loads when a duel is created. Lua errors go to stderr and make the exit
// code 1. Used by tools/check_category_sset.py.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "ygopro-core/card_data.h"
#include "ygopro-core/common.h"
#include "ygopro-core/ocgapi.h"

static std::string g_script_dir;
static std::string g_test_file;
static std::vector<char> g_buf;
static int g_errors = 0;

static bool read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    return false;
  }
  g_buf.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return true;
}

static uint8_t *probe_script_reader(const char *name, int *len) {
  std::string n(name);
  bool ok;
  if (n == "./script/probe.lua") {
    ok = read_file(g_test_file);
  } else {
    const std::string prefix = "./script/";
    std::string rel = n.rfind(prefix, 0) == 0 ? n.substr(prefix.size()) : n;
    ok = read_file(g_script_dir + "/" + rel);
  }
  if (!ok) {
    return nullptr;
  }
  *len = static_cast<int>(g_buf.size());
  return reinterpret_cast<uint8_t *>(g_buf.data());
}

static uint32_t probe_card_reader(uint32_t, card_data *data) {
  data->clear();
  return 0;
}

static uint32_t probe_message_handler(intptr_t pduel, uint32_t type) {
  char buf[1024] = {0};
  get_log_message(pduel, buf);
  if (type == 1) {
    std::fprintf(stderr, "[lua-error] %s\n", buf);
    ++g_errors;
  } else {
    std::printf("%s\n", buf);
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <script_dir> <lua_file>\n", argv[0]);
    return 2;
  }
  g_script_dir = argv[1];
  g_test_file = argv[2];
  set_script_reader(probe_script_reader);
  set_card_reader(probe_card_reader);
  set_message_handler(probe_message_handler);

  uint32_t seeds[SEED_COUNT] = {1, 2, 3, 4, 5, 6, 7, 8};
  intptr_t pduel = create_duel_v2(seeds);
  int32_t ok = preload_script(pduel, "./script/probe.lua");
  end_duel(pduel);
  if (!ok) {
    std::fprintf(stderr, "probe script failed to load or run\n");
    return 1;
  }
  return g_errors ? 1 : 0;
}
