// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (@s243a)
//
// cost_model_probe.c -- measure this machine's single-page read latency for the
// store-crossover cost model, WITHOUT needing any memory-cap mechanism:
//   t_seek : COLD single 4KB page read (posix_fadvise(DONTNEED) the file first)
//   t_mem  : WARM single 4KB page read (page already resident)
// Times N page reads at pseudo-random page-aligned offsets, reports median and
// p10/p90 (ns). Also reports the /proc/self/io read_bytes delta so we can see
// whether this host accounts physical disk reads at all (WSL2 often does not).
//
// Usage: cost_model_probe <file> [N]

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>

static int cmp_ll(const void* a, const void* b) {
  long long x = *(const long long*)a, y = *(const long long*)b;
  return (x > y) - (x < y);
}
static long long now_ns(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
static long long read_bytes_now(void) {
  FILE* f = fopen("/proc/self/io", "r");
  if (!f) return -1;
  char k[64]; long long v, out = -1;
  while (fscanf(f, "%63[^:]: %lld\n", k, &v) == 2) {
    if (strcmp(k, "read_bytes") == 0) { out = v; break; }
  }
  fclose(f);
  return out;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <file> [N]\n", argv[0]); return 2; }
  const char* path = argv[1];
  long N = argc > 2 ? atol(argv[2]) : 2000;
  int fd = open(path, O_RDONLY);
  if (fd < 0) { perror("open"); return 1; }
  struct stat st; if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
  off_t size = st.st_size;
  long pages = size / 4096; if (pages < 1) pages = 1;

  // Deterministic pseudo-random distinct-ish page offsets.
  unsigned int s = 2463534242u;
  off_t* offs = malloc(sizeof(off_t) * N);
  for (long i = 0; i < N; i++) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    offs[i] = ((off_t)(s % pages)) * 4096;
  }
  char* buf = malloc(4096);
  long long* lat = malloc(sizeof(long long) * N);

  // ---- COLD: drop the whole file from cache, then read each page once ----
  posix_fadvise(fd, 0, size, POSIX_FADV_DONTNEED);
  long long rb0 = read_bytes_now();
  for (long i = 0; i < N; i++) {
    // Re-drop this page right before reading so it is cold even if a neighbor
    // read pulled it in (readahead). DONTNEED on a 4KB range is cheap.
    posix_fadvise(fd, offs[i], 4096, POSIX_FADV_DONTNEED);
    long long t0 = now_ns();
    ssize_t n = pread(fd, buf, 4096, offs[i]);
    long long t1 = now_ns();
    (void)n; lat[i] = t1 - t0;
  }
  long long rb1 = read_bytes_now();
  qsort(lat, N, sizeof(long long), cmp_ll);
  long long cold_med = lat[N/2], cold_p10 = lat[N/10], cold_p90 = lat[(N*9)/10];

  // ---- WARM: read the same pages again (now resident) ----
  for (long i = 0; i < N; i++) (void)pread(fd, buf, 4096, offs[i]); // prime
  for (long i = 0; i < N; i++) {
    long long t0 = now_ns();
    ssize_t n = pread(fd, buf, 4096, offs[i]);
    long long t1 = now_ns();
    (void)n; lat[i] = t1 - t0;
  }
  qsort(lat, N, sizeof(long long), cmp_ll);
  long long warm_med = lat[N/2], warm_p10 = lat[N/10], warm_p90 = lat[(N*9)/10];

  printf("{\"file\":\"%s\",\"size\":%lld,\"pages\":%ld,\"N\":%ld,"
         "\"t_seek_cold_ns\":{\"median\":%lld,\"p10\":%lld,\"p90\":%lld},"
         "\"t_mem_warm_ns\":{\"median\":%lld,\"p10\":%lld,\"p90\":%lld},"
         "\"proc_read_bytes_delta_cold\":%lld}\n",
         path, (long long)size, pages, N,
         cold_med, cold_p10, cold_p90, warm_med, warm_p10, warm_p90,
         (rb0 < 0 || rb1 < 0) ? -1 : (rb1 - rb0));
  free(offs); free(buf); free(lat); close(fd);
  return 0;
}
