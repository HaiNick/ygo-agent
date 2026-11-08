#pragma once
// Crash info for reproducing env crashes.
//
// If the environment variable YGO_CRASH_FILE is set when the module is
// initialised, fatal signals (SIGSEGV, SIGABRT, SIGBUS, SIGFPE, SIGILL) append
// the current duel of every env in the process (deck pair, duel seed, seed
// sequence, episode, turn) to that file, marking the env that was running on
// the crashing thread with '*'. The previous handler (or the default action,
// i.e. a core dump) runs afterwards. Uncaught C++ exceptions in env threads end
// in std::terminate -> SIGABRT and are logged too.
//
// The handler only uses async-signal-safe calls (open/write/close, no malloc).
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>

namespace ygo_crash {

constexpr int kSlots = 4096;  // env ids beyond this are not tracked
constexpr int kLen = 384;

inline char g_slots[kSlots][kLen];
inline std::atomic<pthread_t> g_thread[kSlots];
inline char g_path[1024];
inline struct sigaction g_old[NSIG];
inline std::atomic<bool> g_installed{false};

inline void write_str(int fd, const char *s, size_t n) {
  while (n > 0) {
    ssize_t r = write(fd, s, n);
    if (r <= 0) return;
    s += r;
    n -= static_cast<size_t>(r);
  }
}
inline void write_str(int fd, const char *s) { write_str(fd, s, strlen(s)); }
inline void write_int(int fd, long v) {
  char b[24];
  int i = 23;
  b[i] = 0;
  unsigned long u = v < 0 ? static_cast<unsigned long>(-v) : static_cast<unsigned long>(v);
  do {
    b[--i] = static_cast<char>('0' + u % 10);
    u /= 10;
  } while (u);
  if (v < 0) b[--i] = '-';
  write_str(fd, b + i);
}

inline void handler(int sig, siginfo_t *, void *) {
  int fd = open(g_path, O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd >= 0) {
    write_str(fd, "=== ygoenv fatal signal ");
    write_int(fd, sig);
    write_str(fd, ", pid ");
    write_int(fd, getpid());
    write_str(fd, ", unix time ");
    write_int(fd, static_cast<long>(time(nullptr)));
    write_str(fd, "\n");
    pthread_t self = pthread_self();
    for (int i = 0; i < kSlots; ++i) {
      if (!g_slots[i][0]) continue;
      bool here = pthread_equal(g_thread[i].load(std::memory_order_relaxed), self);
      write_str(fd, here ? "* " : "  ");
      write_str(fd, g_slots[i], strnlen(g_slots[i], kLen));
      write_str(fd, "\n");
    }
    write_str(fd, "('*' = env that was running on the crashing thread)\n");
    close(fd);
  }
  sigaction(sig, &g_old[sig], nullptr);
  raise(sig);
}

inline void install_from_env() {
  bool expected = false;
  if (!g_installed.compare_exchange_strong(expected, true)) return;
  const char *p = std::getenv("YGO_CRASH_FILE");
  if (!p || !*p) return;
  std::strncpy(g_path, p, sizeof(g_path) - 1);
  struct sigaction sa;
  std::memset(&sa, 0, sizeof(sa));
  sa.sa_sigaction = handler;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  for (int sig : {SIGSEGV, SIGABRT, SIGBUS, SIGFPE, SIGILL}) sigaction(sig, &sa, &g_old[sig]);
}

// Env ids repeat within a process (cleanba runs several env pools, each with
// ids 0..n-1), so slots are handed out by a process-wide counter.
inline std::atomic<int> g_next_slot{0};
inline int allocate_slot() {
  int s = g_next_slot.fetch_add(1, std::memory_order_relaxed);
  return s < kSlots ? s : -1;
}

inline void mark_thread(int slot) {
  if (slot >= 0 && slot < kSlots) g_thread[slot].store(pthread_self(), std::memory_order_relaxed);
}

template <class... Args>
inline void set_slot(int slot, const char *fmt, Args... args) {
  if (slot < 0 || slot >= kSlots || !g_path[0]) return;
  char tmp[kLen];
  std::snprintf(tmp, kLen, fmt, args...);
  std::memcpy(g_slots[slot], tmp, kLen);
}

}  // namespace ygo_crash
