// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// bench_main.cpp -- minimal raw-LOOKUP harness for the ABI store crossover
// benchmark. It drives the C++ WAM SeekFactSource read path DIRECTLY (the same
// wam_cpp::SeekFactSource that the compiled resolver dispatches through for a
// store fact source), NOT the JS/wamjs lmdb backend. The whole point of the
// measurement is the C++ L1 (direct-mapped) + L2 (FIFO) app caches that only
// this backend has; the indexed backend has none and leans on the OS page
// cache, so under memory pressure it does real disk seeks.
//
// It reads a list of query keys from a file, looks each up REPEATED R times in
// ONE process, and prints the D43 deterministic I/O stats (bytes read, read
// count, per-source L1/L2 hits + cache misses) plus wall time as a single JSON
// line on stdout.
//
// Usage:
//   bench_main <indexed|lmdb> <store-path> <keys-file> <R>
// store-path: indexed -> the UWFI/UWIX prefix (Prefix.data + Prefix.idx);
//             lmdb    -> the LMDB env directory (data.mdb + lock.mdb).
// Cache sizing (lmdb only) via env: UW_WAM_LMDB_L1_SLOTS, UW_WAM_LMDB_L2_CAP.

#include "wam_runtime.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

using namespace wam_cpp;

static const char* env_or(const char* name, const char* dflt) {
  const char* v = std::getenv(name);
  return (v && *v) ? v : dflt;
}

// Evict one file's pages from the OS page cache (no root needed):
// posix_fadvise(DONTNEED) drops the clean cached pages for the inode, so the
// next read faults from disk. This is the root-free "store not resident under
// memory pressure" proxy used when systemd/cgroup hard caps are unavailable.
static void evict_file(const std::string& p) {
  int fd = ::open(p.c_str(), O_RDONLY);
  if (fd < 0) return;
  struct stat st{};
  if (::fstat(fd, &st) == 0 && st.st_size > 0) {
    // Pre-fault the file so its pages are actually RESIDENT, THEN drop them:
    // POSIX_FADV_DONTNEED is a no-op on non-resident pages, so without the
    // streaming read below the "cold" run would be cold-in-name-only. The
    // sequential read forces the pages in; DONTNEED then evicts them, so the
    // subsequent lookups genuinely fault from disk.
    char buf[1 << 16];
    ssize_t n;
    while ((n = ::read(fd, buf, sizeof(buf))) > 0) { /* force pages resident */ }
    ::posix_fadvise(fd, 0, st.st_size, POSIX_FADV_DONTNEED);
  }
  ::close(fd);
}

// Evict the store files backing this run so a cold-cache wall time can be
// measured. Enabled by UW_BENCH_EVICT=1.
static void evict_store(const std::string& kind, const std::string& path) {
  if (kind == "indexed") {
    evict_file(path + ".data");
    evict_file(path + ".idx");
  } else {
    evict_file(path + "/data.mdb");
  }
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
      "usage: %s <indexed|lmdb> <store-path> <keys-file> <R>\n", argv[0]);
    return 2;
  }
  const std::string kind = argv[1];
  const std::string path = argv[2];
  const std::string keysFile = argv[3];
  const int R = std::atoi(argv[4]);

  // Load query keys.
  std::vector<std::string> keys;
  {
    std::ifstream in(keysFile);
    if (!in.is_open()) { std::fprintf(stderr, "cannot open keys file %s\n", keysFile.c_str()); return 1; }
    std::string line;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (!line.empty()) keys.push_back(line);
    }
  }
  if (keys.empty()) { std::fprintf(stderr, "no keys loaded\n"); return 1; }

  const bool evict = std::string(env_or("UW_BENCH_EVICT", "0")) == "1";
  if (evict) evict_store(kind, path);

  SeekFactSource src(kind, path);
  reset_fact_io();

  std::uint64_t rowsFound = 0;
  std::uint64_t lookups = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < R; ++r) {
    for (const std::string& k : keys) {
      // Mirror WamState::dispatch_foreign_call: a bound atomic A1 is encoded via
      // encode_store_key (0x41 atom tag + utf8) before the keyed seek.
      auto rows = src.rows(std::optional<std::string>(encode_store_key(Value::Atom(k))));
      rowsFound += rows.size();
      ++lookups;
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::printf(
    "{\"kind\":\"%s\",\"R\":%d,\"nkeys\":%zu,\"lookups\":%llu,\"rows_found\":%llu,"
    "\"fact_io_bytes\":%llu,\"fact_io_reads\":%llu,\"fact_io_data_size\":%llu,"
    "\"l1_hits\":%llu,\"l2_hits\":%llu,\"cache_misses\":%llu,"
    "\"l1_slots\":\"%s\",\"l2_cap\":\"%s\",\"evict\":%d,\"wall_ms\":%.3f}\n",
    kind.c_str(), R, keys.size(),
    (unsigned long long)lookups, (unsigned long long)rowsFound,
    (unsigned long long)fact_io_bytes(), (unsigned long long)fact_io_reads(),
    (unsigned long long)fact_io_data_size(),
    (unsigned long long)src.l1_hits(), (unsigned long long)src.l2_hits(),
    (unsigned long long)src.cache_misses(),
    env_or("UW_WAM_LMDB_L1_SLOTS", "default"),
    env_or("UW_WAM_LMDB_L2_CAP", "default"),
    evict ? 1 : 0,
    wall_ms);
  return 0;
}
