#!/usr/bin/env python3
"""
Benchmark the WAM-C bidirectional ancestor kernel without generating a full
effective-distance WAM-C project.

The script reuses benchmark TSV inputs, builds optional reverse-CSR artifacts,
compiles a tiny C harness against the WAM-C runtime, and times
`wam_collect_bidirectional_ancestor_hops` over sampled category/root queries.
This isolates native child-search behavior from the expensive full facts-program
generation path.
"""

from __future__ import annotations

import argparse
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "examples" / "benchmark"
RUNTIME_INCLUDE_DIR = ROOT / "src" / "unifyweaver" / "targets" / "wam_c_runtime"
WAM_C_TARGET = ROOT / "src" / "unifyweaver" / "targets" / "wam_c_target.pl"
sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_wam_c_child_csr_scale_sweep import (  # noqa: E402
    BENCH_DIR,
    category_id_map,
    file_tree_size_bytes,
    read_tsv_column,
    read_tsv_pairs,
    scale_names,
    write_reverse_csr_artifact,
)


DEFAULT_SCALES = "10k"
DEFAULT_MODES = "scan,sorted_array"
MODE_BACKENDS = {
    "scan": None,
    "sorted_array": "sorted_array",
    "sorted_array_pread_drop": "sorted_array",
    "lmdb_offset": "lmdb_offset",
    "lmdb_offset_pread_drop": "lmdb_offset",
}


HARNESS_C = r"""
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "wam_runtime.h"

typedef struct {
    char **categories;
    char **roots;
    int count;
} QueryList;

static double elapsed_ms(struct timespec start, struct timespec end) {
    double seconds = (double)(end.tv_sec - start.tv_sec);
    double nanos = (double)(end.tv_nsec - start.tv_nsec);
    return seconds * 1000.0 + nanos / 1000000.0;
}

static char *clone_range(const char *start, size_t len) {
    char *copy = (char *)malloc(len + 1);
    if (!copy) return NULL;
    memcpy(copy, start, len);
    copy[len] = 0;
    return copy;
}

static void trim_right(char *value) {
    size_t len = strlen(value);
    while (len > 0) {
        char c = value[len - 1];
        if (c != '\n' && c != '\r' && c != ' ' && c != '\t') break;
        value[--len] = 0;
    }
}

static int append_query(QueryList *queries, const char *category, const char *root) {
    int next_count = queries->count + 1;
    char **next_categories = (char **)realloc(
        queries->categories, (size_t)next_count * sizeof(char *));
    if (!next_categories) return 0;
    queries->categories = next_categories;
    char **next_roots = (char **)realloc(
        queries->roots, (size_t)next_count * sizeof(char *));
    if (!next_roots) return 0;
    queries->roots = next_roots;
    queries->categories[queries->count] = clone_range(category, strlen(category));
    queries->roots[queries->count] = clone_range(root, strlen(root));
    if (!queries->categories[queries->count] || !queries->roots[queries->count]) {
        free(queries->categories[queries->count]);
        free(queries->roots[queries->count]);
        return 0;
    }
    queries->count = next_count;
    return 1;
}

static void close_queries(QueryList *queries) {
    for (int i = 0; i < queries->count; i++) {
        free(queries->categories[i]);
        free(queries->roots[i]);
    }
    free(queries->categories);
    free(queries->roots);
    memset(queries, 0, sizeof(QueryList));
}

static int read_queries(const char *path, QueryList *queries) {
    FILE *file = fopen(path, "r");
    if (!file) return 0;
    char line[8192];
    while (fgets(line, sizeof(line), file)) {
        char *sep = strchr(line, '\t');
        if (!sep) continue;
        *sep = 0;
        char *category = line;
        char *root = sep + 1;
        trim_right(category);
        trim_right(root);
        if (!category[0] || !root[0]) continue;
        if (!append_query(queries, category, root)) {
            fclose(file);
            return 0;
        }
    }
    fclose(file);
    return queries->count > 0;
}

static int load_category_ids(WamState *state, const char *path) {
    FILE *file = fopen(path, "r");
    if (!file) return 0;
    char line[8192];
    while (fgets(line, sizeof(line), file)) {
        char *sep = strchr(line, '\t');
        if (!sep) continue;
        *sep = 0;
        char *atom = line;
        char *id_text = sep + 1;
        trim_right(atom);
        trim_right(id_text);
        char *end = NULL;
        long id = strtol(id_text, &end, 10);
        if (end == id_text || id < INT32_MIN || id > INT32_MAX) {
            fclose(file);
            return 0;
        }
        wam_register_category_id(state, atom, (int)id);
    }
    fclose(file);
    return 1;
}

static int load_parent_edges(WamState *state, const char *path) {
    WamFactSource source;
    wam_fact_source_init(&source);
    if (!wam_fact_source_load_tsv(state, &source, path)) {
        wam_fact_source_close(&source);
        return 0;
    }
    int ok = wam_register_category_parent_fact_source(state, &source) ? 1 : 0;
    wam_fact_source_close(&source);
    return ok;
}

static int load_artifact(WamReverseCsrArtifact *artifact,
                         const char *mode,
                         const char *index_path,
                         const char *values_path,
                         const char *offset_env_path) {
    if (strcmp(mode, "scan") == 0) return 1;
    if (strcmp(mode, "sorted_array") == 0) {
        return wam_reverse_csr_load(artifact, index_path, values_path) ? 1 : 0;
    }
    if (strcmp(mode, "sorted_array_pread_drop") == 0) {
        return wam_reverse_csr_load_pread_drop(artifact, index_path, values_path) ? 1 : 0;
    }
    if (strcmp(mode, "lmdb_offset") == 0) {
        return wam_reverse_csr_load_lmdb_offset(artifact, values_path, offset_env_path, "offsets") ? 1 : 0;
    }
    if (strcmp(mode, "lmdb_offset_pread_drop") == 0) {
        return wam_reverse_csr_load_lmdb_offset_pread_drop(artifact, values_path, offset_env_path, "offsets") ? 1 : 0;
    }
    return 0;
}

static int run_loop(WamState *state,
                    QueryList queries,
                    long long *total_results_out,
                    long long *checksum_out) {
    long long total_results = 0;
    long long checksum = 0;
    for (int qi = 0; qi < queries.count; qi++) {
        state->A[0] = val_atom(wam_intern_atom(state, queries.categories[qi]));
        state->A[1] = val_atom(wam_intern_atom(state, queries.roots[qi]));
        WamBidirectionalAncestorResults results;
        wam_bidirectional_ancestor_results_init(&results);
        int ok = wam_collect_bidirectional_ancestor_hops(state, &results) ? 1 : 0;
        if (!ok) {
            wam_bidirectional_ancestor_results_close(&results);
            continue;
        }
        total_results += results.count;
        for (int ri = 0; ri < results.count; ri++) {
            checksum += ((long long)results.values[ri].total_hops * 1000003LL);
            checksum += ((long long)results.values[ri].parent_hops * 1009LL);
            checksum += (long long)results.values[ri].child_hops;
        }
        wam_bidirectional_ancestor_results_close(&results);
    }
    *total_results_out = total_results;
    *checksum_out = checksum;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 14) {
        fprintf(stderr,
                "usage: %s MODE PARENT_TSV IDS_TSV IDX VAL OFFSET_LMDB QUERIES ITERATIONS WARMUPS MAX_DEPTH PARENT_COST CHILD_COST BUDGET\n",
                argv[0]);
        return 2;
    }

    const char *mode = argv[1];
    const char *parent_tsv = argv[2];
    const char *ids_tsv = argv[3];
    const char *index_path = argv[4];
    const char *values_path = argv[5];
    const char *offset_env_path = argv[6];
    const char *queries_path = argv[7];
    int iterations = atoi(argv[8]);
    int warmups = atoi(argv[9]);
    int max_depth = atoi(argv[10]);
    double parent_cost = atof(argv[11]);
    double child_cost = atof(argv[12]);
    double budget = atof(argv[13]);
    if (iterations <= 0 || warmups < 0 || max_depth <= 0 ||
        parent_cost <= 0.0 || child_cost <= 0.0 || budget <= 0.0) {
        return 3;
    }

    struct timespec setup_start;
    struct timespec setup_end;
    clock_gettime(CLOCK_MONOTONIC, &setup_start);

    WamState state;
    wam_state_init(&state);
    WamReverseCsrArtifact artifact;
    wam_reverse_csr_init(&artifact);
    QueryList queries;
    memset(&queries, 0, sizeof(QueryList));

    int attached_artifact = 0;
    if (!load_parent_edges(&state, parent_tsv) ||
        !load_category_ids(&state, ids_tsv) ||
        !read_queries(queries_path, &queries)) {
        close_queries(&queries);
        wam_reverse_csr_close(&artifact);
        wam_free_state(&state);
        return 4;
    }
    if (!load_artifact(&artifact, mode, index_path, values_path, offset_env_path)) {
        close_queries(&queries);
        wam_reverse_csr_close(&artifact);
        wam_free_state(&state);
        return 5;
    }
    if (strcmp(mode, "scan") != 0) {
        wam_attach_bidirectional_child_csr(&state, &artifact);
        attached_artifact = 1;
    }
    wam_register_bidirectional_ancestor_kernel(
        &state, "bidirectional_ancestor/5", max_depth, parent_cost, child_cost, budget);

    clock_gettime(CLOCK_MONOTONIC, &setup_end);
    double setup_ms = elapsed_ms(setup_start, setup_end);

    long long total_results = 0;
    long long checksum = 0;
    for (int wi = 0; wi < warmups; wi++) {
        if (!run_loop(&state, queries, &total_results, &checksum)) {
            close_queries(&queries);
            if (attached_artifact) wam_attach_bidirectional_child_csr(&state, NULL);
            wam_reverse_csr_close(&artifact);
            wam_free_state(&state);
            return 6;
        }
    }

    printf("iteration\telapsed_ms\tsetup_ms\ttotal_results\tchecksum\n");
    for (int it = 0; it < iterations; it++) {
        struct timespec start;
        struct timespec end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        if (!run_loop(&state, queries, &total_results, &checksum)) {
            close_queries(&queries);
            if (attached_artifact) wam_attach_bidirectional_child_csr(&state, NULL);
            wam_reverse_csr_close(&artifact);
            wam_free_state(&state);
            return 7;
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        printf("%d\t%.6f\t%.6f\t%lld\t%lld\n",
               it + 1, elapsed_ms(start, end), setup_ms, total_results, checksum);
    }

    close_queries(&queries);
    if (attached_artifact) wam_attach_bidirectional_child_csr(&state, NULL);
    wam_reverse_csr_close(&artifact);
    wam_free_state(&state);
    return 0;
}
"""


@dataclass(frozen=True)
class KernelRow:
    scale: str
    mode: str
    sample_queries: int
    sample_roots: int
    iterations: int
    warmups: int
    parent_edge_count: int
    category_count: int
    max_depth: int
    parent_step_cost: float
    child_step_cost: float
    budget: float
    setup_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    query_us_per_query: float
    total_results: int
    checksum: int
    harness_compile_s: float
    artifact_build_s: float
    reverse_csr_index_bytes: int
    reverse_csr_values_bytes: int
    reverse_csr_offsets_lmdb_bytes: int


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--sample-queries", type=int, default=1000)
    parser.add_argument("--sample-roots", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--parent-step-cost", type=float, default=1.0)
    parser.add_argument("--child-step-cost", type=float, default=3.0)
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--keep-work", action="store_true", help="keep temporary artifacts and harness files")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def parse_modes(value: str) -> list[str]:
    modes = [part.strip() for part in value.split(",") if part.strip()]
    if not modes:
        raise ValueError("--modes must list at least one mode")
    unknown = sorted(set(modes) - set(MODE_BACKENDS))
    if unknown:
        raise ValueError(f"unsupported mode(s): {', '.join(unknown)}")
    return modes


def generate_runtime_c(path: Path) -> None:
    command = [
        "swipl",
        "-q",
        "-g",
        (
            f"use_module('{WAM_C_TARGET.as_posix()}'), "
            "wam_c_target:compile_wam_runtime_to_c([], C), write(C), halt"
        ),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"WAM-C runtime generation failed:\n{result.stderr}")
    path.write_text(result.stdout, encoding="utf-8")


def compile_harness(work_dir: Path) -> tuple[Path, float]:
    runtime_c = work_dir / "wam_runtime.c"
    harness_c = work_dir / "wam_c_bidirectional_kernel_harness.c"
    binary = work_dir / "wam_c_bidirectional_kernel_harness"
    started = time.perf_counter()
    generate_runtime_c(runtime_c)
    harness_c.write_text(HARNESS_C, encoding="utf-8")
    command = [
        "gcc",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-DWAM_C_ENABLE_LMDB",
        "-I",
        str(RUNTIME_INCLUDE_DIR),
        str(runtime_c),
        str(harness_c),
        "-lm",
        "-llmdb",
        "-o",
        str(binary),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"WAM-C bidirectional kernel harness compile failed:\n{result.stderr}")
    return binary, time.perf_counter() - started


def write_parent_runtime_tsv(path: Path, category_parents: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for child, parent in category_parents:
            stream.write(f"{child}\t{parent}\n")


def write_category_ids(path: Path, ids: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for category, category_id in sorted(ids.items(), key=lambda item: item[1]):
            stream.write(f"{category}\t{category_id}\n")


def selected_roots(root_categories: list[str], sample_roots: int, seed: int) -> list[str]:
    if not root_categories:
        raise ValueError("scale has no root categories")
    if sample_roots <= 0 or sample_roots >= len(root_categories):
        return list(root_categories)
    rng = random.Random(seed)
    return sorted(rng.sample(root_categories, sample_roots))


def sampled_queries(
    article_categories: list[tuple[str, str]],
    category_parents: list[tuple[str, str]],
    roots: list[str],
    sample_queries: int,
    seed: int,
) -> list[tuple[str, str]]:
    if not article_categories:
        raise ValueError("scale has no article_category rows")
    if not roots:
        raise ValueError("expected at least one sampled root")
    if sample_queries <= 0:
        raise ValueError("--sample-queries must be positive")
    rng = random.Random(seed)
    article_category_values = [category for _article, category in article_categories]
    article_category_set = set(article_category_values)
    reachable = reachable_article_categories_by_root(category_parents, article_category_set, roots)
    queries: list[tuple[str, str]] = []
    for root in roots[:sample_queries]:
        category_choices = reachable.get(root) or article_category_values
        category = category_choices[rng.randrange(len(category_choices))]
        queries.append((category, root))
    while len(queries) < sample_queries:
        root = roots[rng.randrange(len(roots))]
        category_choices = reachable.get(root) or article_category_values
        category = category_choices[rng.randrange(len(category_choices))]
        queries.append((category, root))
    return queries


def reachable_article_categories_by_root(
    category_parents: list[tuple[str, str]],
    article_category_set: set[str],
    roots: list[str],
) -> dict[str, list[str]]:
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for child, parent in category_parents:
        children_by_parent[parent].append(child)

    reachable: dict[str, list[str]] = {}
    for root in roots:
        seen = {root}
        queue: deque[str] = deque([root])
        hits: list[str] = []
        while queue:
            node = queue.popleft()
            if node in article_category_set:
                hits.append(node)
            for child in children_by_parent.get(node, []):
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
        reachable[root] = sorted(hits)
    return reachable


def write_queries(path: Path, queries: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for category, root in queries:
            stream.write(f"{category}\t{root}\n")


def build_scale_inputs(
    scale: str,
    out_root: Path,
    modes: list[str],
    sample_roots: int,
    sample_queries: int,
    seed: int,
) -> tuple[dict[str, Path], dict[str, float], Path, Path, Path, int, int, int]:
    scale_dir = BENCH_DIR / scale
    category_parent_path = scale_dir / "category_parent.tsv"
    if not category_parent_path.exists():
        raise FileNotFoundError(category_parent_path)

    category_parents = read_tsv_pairs(category_parent_path)
    article_categories = read_tsv_pairs(scale_dir / "article_category.tsv")
    root_categories = read_tsv_column(scale_dir / "root_categories.tsv")
    ids = category_id_map(category_parents, article_categories, root_categories)
    roots = selected_roots(root_categories, sample_roots, seed)
    queries = sampled_queries(article_categories, category_parents, roots, sample_queries, seed + 1)

    scale_root = out_root / scale
    scale_root.mkdir(parents=True, exist_ok=True)
    parent_runtime_tsv = scale_root / "category_parent.runtime.tsv"
    ids_tsv = scale_root / "category_ids.tsv"
    queries_tsv = scale_root / "queries.tsv"
    write_parent_runtime_tsv(parent_runtime_tsv, category_parents)
    write_category_ids(ids_tsv, ids)
    write_queries(queries_tsv, queries)

    artifact_dirs: dict[str, Path] = {}
    artifact_build_s: dict[str, float] = {mode: 0.0 for mode in modes}
    needed_backends = sorted({MODE_BACKENDS[mode] for mode in modes if MODE_BACKENDS[mode] is not None})
    for backend in needed_backends:
        out_dir = scale_root / backend
        if out_dir.exists():
            shutil.rmtree(out_dir)
        started = time.perf_counter()
        write_reverse_csr_artifact(category_parents, ids, out_dir, backend)
        elapsed = time.perf_counter() - started
        artifact_dirs[backend] = out_dir
        for mode in modes:
            if MODE_BACKENDS[mode] == backend:
                artifact_build_s[mode] = elapsed
    return (
        artifact_dirs,
        artifact_build_s,
        parent_runtime_tsv,
        ids_tsv,
        queries_tsv,
        len(category_parents),
        len(ids),
        len(set(root for _category, root in queries)),
    )


def run_harness(
    binary: Path,
    mode: str,
    artifact_dir: Path,
    parent_runtime_tsv: Path,
    ids_tsv: Path,
    queries_tsv: Path,
    iterations: int,
    warmups: int,
    max_depth: int,
    parent_step_cost: float,
    child_step_cost: float,
    budget: float,
) -> tuple[list[float], float, int, int]:
    command = [
        str(binary),
        mode,
        str(parent_runtime_tsv),
        str(ids_tsv),
        str(artifact_dir / "category_child.csr.idx"),
        str(artifact_dir / "category_child.csr.val"),
        str(artifact_dir / "category_child.csr.offsets.lmdb"),
        str(queries_tsv),
        str(iterations),
        str(warmups),
        str(max_depth),
        f"{parent_step_cost:g}",
        f"{child_step_cost:g}",
        f"{budget:g}",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"WAM-C bidirectional kernel harness failed for {mode} "
            f"with status {result.returncode}:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    expected_header = ["iteration", "elapsed_ms", "setup_ms", "total_results", "checksum"]
    if not lines or lines[0].split("\t") != expected_header:
        raise RuntimeError(f"unexpected harness output:\n{result.stdout}")
    times: list[float] = []
    setup_ms = 0.0
    total_results = 0
    checksum = 0
    for line in lines[1:]:
        _iteration, elapsed_ms, setup_ms_text, total_results_text, checksum_text = line.split("\t")
        times.append(float(elapsed_ms))
        setup_ms = float(setup_ms_text)
        total_results = int(total_results_text)
        checksum = int(checksum_text)
    return times, setup_ms, total_results, checksum


def benchmark_scale(
    scale: str,
    modes: list[str],
    artifact_root: Path,
    harness_binary: Path,
    harness_compile_s: float,
    sample_roots: int,
    sample_queries: int,
    iterations: int,
    warmups: int,
    seed: int,
    max_depth: int,
    parent_step_cost: float,
    child_step_cost: float,
    budget: float,
) -> list[KernelRow]:
    (
        artifact_dirs,
        artifact_build_s,
        parent_runtime_tsv,
        ids_tsv,
        queries_tsv,
        parent_edge_count,
        category_count,
        distinct_roots,
    ) = build_scale_inputs(scale, artifact_root, modes, sample_roots, sample_queries, seed)
    empty_artifact_dir = artifact_root / scale / "scan"
    empty_artifact_dir.mkdir(parents=True, exist_ok=True)

    rows: list[KernelRow] = []
    expected_checksum: int | None = None
    for mode in modes:
        backend = MODE_BACKENDS[mode]
        artifact_dir = empty_artifact_dir if backend is None else artifact_dirs[backend]
        times, setup_ms, total_results, checksum = run_harness(
            harness_binary,
            mode,
            artifact_dir,
            parent_runtime_tsv,
            ids_tsv,
            queries_tsv,
            iterations,
            warmups,
            max_depth,
            parent_step_cost,
            child_step_cost,
            budget,
        )
        if expected_checksum is None:
            expected_checksum = checksum
        elif checksum != expected_checksum:
            raise RuntimeError(f"checksum mismatch for mode {mode}: {checksum} != {expected_checksum}")
        median_ms = statistics.median(times)
        rows.append(
            KernelRow(
                scale=scale,
                mode=mode,
                sample_queries=sample_queries,
                sample_roots=distinct_roots,
                iterations=iterations,
                warmups=warmups,
                parent_edge_count=parent_edge_count,
                category_count=category_count,
                max_depth=max_depth,
                parent_step_cost=parent_step_cost,
                child_step_cost=child_step_cost,
                budget=budget,
                setup_ms=setup_ms,
                median_ms=median_ms,
                min_ms=min(times),
                max_ms=max(times),
                query_us_per_query=(median_ms * 1000.0) / max(sample_queries, 1),
                total_results=total_results,
                checksum=checksum,
                harness_compile_s=harness_compile_s,
                artifact_build_s=artifact_build_s[mode],
                reverse_csr_index_bytes=file_tree_size_bytes(artifact_dir / "category_child.csr.idx"),
                reverse_csr_values_bytes=file_tree_size_bytes(artifact_dir / "category_child.csr.val"),
                reverse_csr_offsets_lmdb_bytes=file_tree_size_bytes(artifact_dir / "category_child.csr.offsets.lmdb"),
            )
        )
    return rows


def print_rows(rows: list[KernelRow]) -> None:
    headers = [
        "scale",
        "mode",
        "sample_queries",
        "sample_roots",
        "iterations",
        "warmups",
        "parent_edge_count",
        "category_count",
        "max_depth",
        "parent_step_cost",
        "child_step_cost",
        "budget",
        "setup_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "query_us_per_query",
        "total_results",
        "checksum",
        "harness_compile_s",
        "artifact_build_s",
        "reverse_csr_index_bytes",
        "reverse_csr_values_bytes",
        "reverse_csr_offsets_lmdb_bytes",
    ]
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    row.scale,
                    row.mode,
                    str(row.sample_queries),
                    str(row.sample_roots),
                    str(row.iterations),
                    str(row.warmups),
                    str(row.parent_edge_count),
                    str(row.category_count),
                    str(row.max_depth),
                    f"{row.parent_step_cost:g}",
                    f"{row.child_step_cost:g}",
                    f"{row.budget:g}",
                    f"{row.setup_ms:.6f}",
                    f"{row.median_ms:.6f}",
                    f"{row.min_ms:.6f}",
                    f"{row.max_ms:.6f}",
                    f"{row.query_us_per_query:.6f}",
                    str(row.total_results),
                    str(row.checksum),
                    f"{row.harness_compile_s:.3f}",
                    f"{row.artifact_build_s:.3f}",
                    str(row.reverse_csr_index_bytes),
                    str(row.reverse_csr_values_bytes),
                    str(row.reverse_csr_offsets_lmdb_bytes),
                ]
            )
        )


def run(args: argparse.Namespace) -> int:
    try:
        modes = parse_modes(args.modes)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    scales = scale_names(args.scales)
    if args.dry_run:
        root = args.artifact_root or Path("<temporary artifact root>")
        for scale in scales:
            print(
                f"wam-c-bidirectional-kernel scale={scale} modes={','.join(modes)} "
                f"sample_queries={args.sample_queries} sample_roots={args.sample_roots} root={root}"
            )
        return 0

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.artifact_root is None:
        if args.keep_work:
            artifact_root = Path(tempfile.mkdtemp(prefix="wam-c-bidirectional-kernel-"))
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="wam-c-bidirectional-kernel-")
            artifact_root = Path(temp_dir.name)
    else:
        artifact_root = args.artifact_root
        artifact_root.mkdir(parents=True, exist_ok=True)

    try:
        harness_binary, harness_compile_s = compile_harness(artifact_root)
        rows: list[KernelRow] = []
        for scale in scales:
            rows.extend(
                benchmark_scale(
                    scale,
                    modes,
                    artifact_root,
                    harness_binary,
                    harness_compile_s,
                    args.sample_roots,
                    args.sample_queries,
                    args.iterations,
                    args.warmups,
                    args.seed,
                    args.max_depth,
                    args.parent_step_cost,
                    args.child_step_cost,
                    args.budget,
                )
            )
        print_rows(rows)
        if args.keep_work and args.artifact_root is None:
            print(f"kept_work_dir\t{artifact_root}")
        return 0
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
