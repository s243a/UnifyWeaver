#!/usr/bin/env python3
"""
Benchmark WAM-C reverse CSR child lookups without running effective distance.

This measures the actual WAM-C runtime API used by generated child-search code:
`wam_reverse_csr_lookup_children`. The script builds reverse CSR artifacts from
benchmark TSVs, compiles a tiny C harness against the generated WAM-C runtime,
and reports lookup timings for the supported file-backed index modes.
"""

from __future__ import annotations

import argparse
import random
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "examples" / "benchmark"
RUNTIME_INCLUDE_DIR = ROOT / "src" / "unifyweaver" / "targets" / "wam_c_runtime"
WAM_C_TARGET = ROOT / "src" / "unifyweaver" / "targets" / "wam_c_target.pl"
sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_wam_c_child_csr_scale_sweep import (  # noqa: E402
    BENCH_DIR,
    I32,
    IDX_RECORD,
    category_id_map,
    file_tree_size_bytes,
    read_tsv_column,
    read_tsv_pairs,
    scale_names,
    write_reverse_csr_artifact,
)


DEFAULT_SCALES = "10k"
DEFAULT_MODES = "sorted_array,sorted_array_pread_drop,lmdb_offset,lmdb_offset_pread_drop"
MODE_BACKENDS = {
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
    int *values;
    int count;
} ParentList;

static double elapsed_ms(struct timespec start, struct timespec end) {
    double seconds = (double)(end.tv_sec - start.tv_sec);
    double nanos = (double)(end.tv_nsec - start.tv_nsec);
    return seconds * 1000.0 + nanos / 1000000.0;
}

static ParentList read_parent_file(const char *path) {
    ParentList parents;
    parents.values = NULL;
    parents.count = 0;
    int cap = 0;
    FILE *file = fopen(path, "r");
    if (!file) return parents;
    char line[64];
    while (fgets(line, sizeof(line), file)) {
        char *end = NULL;
        long value = strtol(line, &end, 10);
        if (end == line) continue;
        if (value < INT32_MIN || value > INT32_MAX) continue;
        if (parents.count == cap) {
            int next_cap = cap == 0 ? 64 : cap * 2;
            int *next = (int *)realloc(parents.values, (size_t)next_cap * sizeof(int));
            if (!next) {
                free(parents.values);
                parents.values = NULL;
                parents.count = 0;
                fclose(file);
                return parents;
            }
            parents.values = next;
            cap = next_cap;
        }
        parents.values[parents.count++] = (int)value;
    }
    fclose(file);
    return parents;
}

static int load_artifact(WamReverseCsrArtifact *artifact,
                         const char *mode,
                         const char *index_path,
                         const char *values_path,
                         const char *offset_env_path) {
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

static int run_loop(WamReverseCsrArtifact *artifact,
                    ParentList parents,
                    int *children,
                    int max_children,
                    long long *total_children_out,
                    long long *checksum_out) {
    long long total_children = 0;
    long long checksum = 0;
    for (int pi = 0; pi < parents.count; pi++) {
        int count = wam_reverse_csr_lookup_children(artifact, parents.values[pi], children, max_children);
        if (count < 0) return 0;
        total_children += count;
        int read_count = count < max_children ? count : max_children;
        for (int ci = 0; ci < read_count; ci++) {
            checksum += children[ci];
        }
    }
    *total_children_out = total_children;
    *checksum_out = checksum;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 9) {
        fprintf(stderr, "usage: %s MODE IDX VAL OFFSET_LMDB PARENTS ITERATIONS MAX_CHILDREN WARMUPS\n", argv[0]);
        return 2;
    }

    const char *mode = argv[1];
    const char *index_path = argv[2];
    const char *values_path = argv[3];
    const char *offset_env_path = argv[4];
    const char *parents_path = argv[5];
    int iterations = atoi(argv[6]);
    int max_children = atoi(argv[7]);
    int warmups = atoi(argv[8]);
    if (iterations <= 0 || max_children <= 0 || warmups < 0) return 3;

    ParentList parents = read_parent_file(parents_path);
    if (!parents.values || parents.count <= 0) return 4;
    int *children = (int *)calloc((size_t)max_children, sizeof(int));
    if (!children) {
        free(parents.values);
        return 5;
    }

    WamReverseCsrArtifact artifact;
    wam_reverse_csr_init(&artifact);
    if (!load_artifact(&artifact, mode, index_path, values_path, offset_env_path)) {
        free(children);
        free(parents.values);
        return 6;
    }

    long long total_children = 0;
    long long checksum = 0;
    for (int wi = 0; wi < warmups; wi++) {
        if (!run_loop(&artifact, parents, children, max_children, &total_children, &checksum)) {
            wam_reverse_csr_close(&artifact);
            free(children);
            free(parents.values);
            return 7;
        }
    }

    printf("iteration\telapsed_ms\ttotal_children\tchecksum\n");
    for (int it = 0; it < iterations; it++) {
        struct timespec start;
        struct timespec end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        if (!run_loop(&artifact, parents, children, max_children, &total_children, &checksum)) {
            wam_reverse_csr_close(&artifact);
            free(children);
            free(parents.values);
            return 8;
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        printf("%d\t%.6f\t%lld\t%lld\n", it + 1, elapsed_ms(start, end), total_children, checksum);
    }

    wam_reverse_csr_close(&artifact);
    free(children);
    free(parents.values);
    return 0;
}
"""


@dataclass(frozen=True)
class LookupRow:
    scale: str
    mode: str
    sample_parents: int
    iterations: int
    parent_count: int
    edge_count: int
    max_children: int
    total_children: int
    checksum: int
    median_ms: float
    min_ms: float
    max_ms: float
    lookup_us_per_parent: float
    reverse_csr_index_bytes: int
    reverse_csr_values_bytes: int
    reverse_csr_offsets_lmdb_bytes: int


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--sample-parents", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
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


def idx_records(idx_path: Path) -> list[tuple[int, int, int]]:
    data = idx_path.read_bytes()
    if len(data) % IDX_RECORD.size != 0:
        raise ValueError(f"CSR index file has partial record: {idx_path}")
    return [
        IDX_RECORD.unpack_from(data, offset)
        for offset in range(0, len(data), IDX_RECORD.size)
    ]


def sample_parent_ids(records: list[tuple[int, int, int]], sample_count: int, seed: int) -> list[int]:
    parents = [parent for parent, _offset, count in records if count > 0]
    if sample_count <= 0 or sample_count >= len(parents):
        return list(parents)
    rng = random.Random(seed)
    return sorted(rng.sample(parents, sample_count))


def write_parent_sample(path: Path, parents: list[int]) -> None:
    path.write_text("".join(f"{parent}\n" for parent in parents), encoding="ascii")


def build_scale_artifacts(scale: str, out_root: Path, modes: list[str]) -> dict[str, Path]:
    scale_dir = BENCH_DIR / scale
    category_parent_path = scale_dir / "category_parent.tsv"
    if not category_parent_path.exists():
        raise FileNotFoundError(category_parent_path)

    category_parents = read_tsv_pairs(category_parent_path)
    article_categories = read_tsv_pairs(scale_dir / "article_category.tsv")
    root_categories = read_tsv_column(scale_dir / "root_categories.tsv")
    ids = category_id_map(category_parents, article_categories, root_categories)

    artifact_dirs: dict[str, Path] = {}
    needed_backends = sorted({MODE_BACKENDS[mode] for mode in modes})
    for backend in needed_backends:
        out_dir = out_root / scale / backend
        if out_dir.exists():
            for child in out_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)
        write_reverse_csr_artifact(category_parents, ids, out_dir, backend)
        artifact_dirs[backend] = out_dir
    return artifact_dirs


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


def compile_harness(work_dir: Path) -> Path:
    runtime_c = work_dir / "wam_runtime.c"
    harness_c = work_dir / "wam_c_reverse_csr_lookup_harness.c"
    binary = work_dir / "wam_c_reverse_csr_lookup_harness"
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
        raise RuntimeError(f"WAM-C CSR lookup harness compile failed:\n{result.stderr}")
    return binary


def run_harness(
    binary: Path,
    mode: str,
    artifact_dir: Path,
    parent_sample_path: Path,
    iterations: int,
    max_children: int,
    warmups: int,
) -> tuple[list[float], int, int]:
    command = [
        str(binary),
        mode,
        str(artifact_dir / "category_child.csr.idx"),
        str(artifact_dir / "category_child.csr.val"),
        str(artifact_dir / "category_child.csr.offsets.lmdb"),
        str(parent_sample_path),
        str(iterations),
        str(max_children),
        str(warmups),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"WAM-C CSR lookup harness failed for {mode}:\n{result.stderr}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines or lines[0].split("\t") != ["iteration", "elapsed_ms", "total_children", "checksum"]:
        raise RuntimeError(f"unexpected harness output:\n{result.stdout}")
    times: list[float] = []
    total_children = 0
    checksum = 0
    for line in lines[1:]:
        _iteration, elapsed_ms, total_children_text, checksum_text = line.split("\t")
        times.append(float(elapsed_ms))
        total_children = int(total_children_text)
        checksum = int(checksum_text)
    return times, total_children, checksum


def benchmark_scale(
    scale: str,
    modes: list[str],
    artifact_root: Path,
    harness_binary: Path,
    sample_parents: int,
    iterations: int,
    warmups: int,
    seed: int,
) -> list[LookupRow]:
    artifact_dirs = build_scale_artifacts(scale, artifact_root, modes)
    sample_artifact_dir = artifact_dirs.get("sorted_array") or next(iter(artifact_dirs.values()))
    records = idx_records(sample_artifact_dir / "category_child.csr.idx")
    parents = sample_parent_ids(records, sample_parents, seed)
    max_children = max((count for _parent, _offset, count in records), default=1)
    parent_sample_path = artifact_root / scale / "sample_parents.txt"
    write_parent_sample(parent_sample_path, parents)
    parent_count = len(records)
    edge_count = sum(count for _parent, _offset, count in records)

    rows: list[LookupRow] = []
    expected_total_children: int | None = None
    expected_checksum: int | None = None
    for mode in modes:
        backend = MODE_BACKENDS[mode]
        artifact_dir = artifact_dirs[backend]
        times, total_children, checksum = run_harness(
            harness_binary,
            mode,
            artifact_dir,
            parent_sample_path,
            iterations,
            max_children,
            warmups,
        )
        if expected_total_children is None:
            expected_total_children = total_children
            expected_checksum = checksum
        elif total_children != expected_total_children or checksum != expected_checksum:
            raise RuntimeError(
                f"lookup parity mismatch for mode {mode}: "
                f"total/checksum={total_children}/{checksum}, "
                f"expected={expected_total_children}/{expected_checksum}"
            )
        median_ms = statistics.median(times)
        rows.append(
            LookupRow(
                scale=scale,
                mode=mode,
                sample_parents=len(parents),
                iterations=iterations,
                parent_count=parent_count,
                edge_count=edge_count,
                max_children=max_children,
                total_children=total_children,
                checksum=checksum,
                median_ms=median_ms,
                min_ms=min(times),
                max_ms=max(times),
                lookup_us_per_parent=(median_ms * 1000.0) / max(len(parents), 1),
                reverse_csr_index_bytes=file_tree_size_bytes(artifact_dir / "category_child.csr.idx"),
                reverse_csr_values_bytes=file_tree_size_bytes(artifact_dir / "category_child.csr.val"),
                reverse_csr_offsets_lmdb_bytes=file_tree_size_bytes(artifact_dir / "category_child.csr.offsets.lmdb"),
            )
        )
    return rows


def print_rows(rows: list[LookupRow]) -> None:
    headers = [
        "scale",
        "mode",
        "sample_parents",
        "iterations",
        "parent_count",
        "edge_count",
        "max_children",
        "total_children",
        "checksum",
        "median_ms",
        "min_ms",
        "max_ms",
        "lookup_us_per_parent",
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
                    str(row.sample_parents),
                    str(row.iterations),
                    str(row.parent_count),
                    str(row.edge_count),
                    str(row.max_children),
                    str(row.total_children),
                    str(row.checksum),
                    f"{row.median_ms:.6f}",
                    f"{row.min_ms:.6f}",
                    f"{row.max_ms:.6f}",
                    f"{row.lookup_us_per_parent:.6f}",
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
            print(f"wam-c-reverse-csr-lookup scale={scale} modes={','.join(modes)} root={root}")
        return 0

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.artifact_root is None:
        if args.keep_work:
            artifact_root = Path(tempfile.mkdtemp(prefix="wam-c-reverse-csr-lookup-"))
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="wam-c-reverse-csr-lookup-")
            artifact_root = Path(temp_dir.name)
    else:
        artifact_root = args.artifact_root
        artifact_root.mkdir(parents=True, exist_ok=True)

    try:
        harness_binary = compile_harness(artifact_root)
        rows: list[LookupRow] = []
        for scale in scales:
            rows.extend(
                benchmark_scale(
                    scale,
                    modes,
                    artifact_root,
                    harness_binary,
                    args.sample_parents,
                    args.iterations,
                    args.warmups,
                    args.seed,
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
