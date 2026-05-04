#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

from benchmark_common import is_termux_environment


@dataclass(frozen=True)
class TargetInfo:
    name: str
    category: str
    description: str


@dataclass(frozen=True)
class KernelPairInfo:
    family: str
    mode: str
    kernels_target: str
    no_kernels_target: str


TARGETS: dict[str, TargetInfo] = {
    "csharp-query": TargetInfo(
        "csharp-query",
        "query-engine",
        "Parameterized C# query runtime",
    ),
    "csharp-dfs": TargetInfo(
        "csharp-dfs",
        "direct-pipeline",
        "Generated C# DFS executable",
    ),
    "rust-dfs": TargetInfo(
        "rust-dfs",
        "direct-pipeline",
        "Generated Rust DFS executable",
    ),
    "go-dfs": TargetInfo(
        "go-dfs",
        "direct-pipeline",
        "Generated Go DFS executable",
    ),
    "prolog-seeded": TargetInfo(
        "prolog-seeded",
        "optimized-prolog",
        "Generated Prolog seeded benchmark",
    ),
    "prolog-accumulated": TargetInfo(
        "prolog-accumulated",
        "optimized-prolog",
        "Generated Prolog accumulated benchmark",
    ),
    "wam-rust-seeded": TargetInfo(
        "wam-rust-seeded",
        "hybrid-wam",
        "Hybrid WAM Rust benchmark with seeded host accumulation",
    ),
    "wam-rust-accumulated": TargetInfo(
        "wam-rust-accumulated",
        "hybrid-wam",
        "Hybrid WAM Rust benchmark with optimized accumulated helpers",
    ),
    "wam-rust-seeded-no-kernels": TargetInfo(
        "wam-rust-seeded-no-kernels",
        "hybrid-wam",
        "Hybrid WAM Rust benchmark with seeded host accumulation and no_kernels(true)",
    ),
    "wam-rust-accumulated-no-kernels": TargetInfo(
        "wam-rust-accumulated-no-kernels",
        "hybrid-wam",
        "Hybrid WAM Rust benchmark with optimized accumulated helpers and no_kernels(true)",
    ),
    "go-wam-accumulated": TargetInfo(
        "go-wam-accumulated",
        "hybrid-wam",
        "Hybrid WAM Go benchmark with optimized accumulated helpers and kernels enabled",
    ),
    "go-wam-accumulated-no-kernels": TargetInfo(
        "go-wam-accumulated-no-kernels",
        "hybrid-wam",
        "Hybrid WAM Go benchmark with optimized accumulated helpers and no_kernels(true)",
    ),
    "scala-wam-seeded": TargetInfo(
        "scala-wam-seeded",
        "hybrid-wam",
        "Hybrid WAM Scala effective-distance runner with seeded helpers and kernels enabled",
    ),
    "scala-wam-seeded-artifact": TargetInfo(
        "scala-wam-seeded-artifact",
        "hybrid-wam",
        "Hybrid WAM Scala effective-distance runner with seeded helpers, kernels enabled, and file-backed artifact data",
    ),
    "scala-wam-seeded-no-kernels": TargetInfo(
        "scala-wam-seeded-no-kernels",
        "hybrid-wam",
        "Hybrid WAM Scala effective-distance runner with seeded helpers and no_kernels(true)",
    ),
    "scala-wam-accumulated": TargetInfo(
        "scala-wam-accumulated",
        "hybrid-wam",
        "Hybrid WAM Scala effective-distance runner with optimized accumulated helpers and kernels enabled",
    ),
    "scala-wam-accumulated-artifact": TargetInfo(
        "scala-wam-accumulated-artifact",
        "hybrid-wam",
        "Hybrid WAM Scala effective-distance runner with optimized accumulated helpers, kernels enabled, and file-backed artifact data",
    ),
    "scala-wam-accumulated-no-kernels": TargetInfo(
        "scala-wam-accumulated-no-kernels",
        "hybrid-wam",
        "Hybrid WAM Scala effective-distance runner with optimized accumulated helpers and no_kernels(true)",
    ),
    "clojure-wam-seeded": TargetInfo(
        "clojure-wam-seeded",
        "hybrid-wam",
        "Hybrid WAM Clojure generated project with seeded helpers and kernels enabled",
    ),
    "clojure-wam-seeded-no-kernels": TargetInfo(
        "clojure-wam-seeded-no-kernels",
        "hybrid-wam",
        "Hybrid WAM Clojure generated project with seeded helpers and no_kernels(true)",
    ),
    "clojure-wam-seeded-artifact": TargetInfo(
        "clojure-wam-seeded-artifact",
        "hybrid-wam",
        "Hybrid WAM Clojure generated project with seeded helpers, kernels enabled, and preprocessed artifact data",
    ),
    "clojure-wam-seeded-no-kernels-artifact": TargetInfo(
        "clojure-wam-seeded-no-kernels-artifact",
        "hybrid-wam",
        "Hybrid WAM Clojure generated project with seeded helpers, no_kernels(true), and preprocessed artifact data",
    ),
    "clojure-wam-accumulated": TargetInfo(
        "clojure-wam-accumulated",
        "hybrid-wam",
        "Hybrid WAM Clojure generated project with optimized accumulated helpers and a result-producing runner",
    ),
    "clojure-wam-accumulated-no-kernels": TargetInfo(
        "clojure-wam-accumulated-no-kernels",
        "hybrid-wam",
        "Hybrid WAM Clojure generated project with optimized accumulated helpers and no_kernels(true)",
    ),
    "clojure-wam-accumulated-artifact": TargetInfo(
        "clojure-wam-accumulated-artifact",
        "hybrid-wam",
        "Hybrid WAM Clojure generated project with optimized accumulated helpers, kernels enabled, and preprocessed artifact data",
    ),
    "clojure-wam-accumulated-no-kernels-artifact": TargetInfo(
        "clojure-wam-accumulated-no-kernels-artifact",
        "hybrid-wam",
        "Hybrid WAM Clojure generated project with optimized accumulated helpers, no_kernels(true), and preprocessed artifact data",
    ),
    "haskell-pure-interp": TargetInfo(
        "haskell-pure-interp",
        "hybrid-wam",
        "Optimized Prolog -> WAM Haskell interpreter with no_kernels(true)",
    ),
    "haskell-interp-ffi": TargetInfo(
        "haskell-interp-ffi",
        "hybrid-wam",
        "Optimized Prolog -> WAM Haskell interpreter with kernels enabled",
    ),
    "haskell-interp-ffi-auto": TargetInfo(
        "haskell-interp-ffi-auto",
        "hybrid-wam",
        "Optimized Prolog -> WAM Haskell interpreter, kernels enabled, "
        "use_lmdb(auto) — resolver picks IntMap or LMDB based on fact_count "
        "and ghc-pkg availability of the lmdb package",
    ),
    "haskell-lowered-only": TargetInfo(
        "haskell-lowered-only",
        "optimized-prolog",
        "Optimized Prolog -> lowered Haskell functions with no_kernels(true)",
    ),
    "haskell-lowered-ffi": TargetInfo(
        "haskell-lowered-ffi",
        "optimized-prolog",
        "Optimized Prolog -> lowered Haskell functions with kernels enabled",
    ),
    "rust-pure-interp": TargetInfo(
        "rust-pure-interp",
        "hybrid-wam",
        "Optimized Prolog -> WAM Rust interpreter with no_kernels(true)",
    ),
    "rust-interp-ffi": TargetInfo(
        "rust-interp-ffi",
        "hybrid-wam",
        "Optimized Prolog -> WAM Rust interpreter with kernels enabled",
    ),
    "rust-lowered-only": TargetInfo(
        "rust-lowered-only",
        "optimized-prolog",
        "Optimized Prolog -> lowered Rust functions with no_kernels(true)",
    ),
    "rust-lowered-ffi": TargetInfo(
        "rust-lowered-ffi",
        "optimized-prolog",
        "Optimized Prolog -> lowered Rust functions with kernels enabled",
    ),
}


KERNEL_TARGET_PAIRS: tuple[KernelPairInfo, ...] = (
    KernelPairInfo(
        "rust",
        "seeded",
        "wam-rust-seeded",
        "wam-rust-seeded-no-kernels",
    ),
    KernelPairInfo(
        "rust",
        "accumulated",
        "wam-rust-accumulated",
        "wam-rust-accumulated-no-kernels",
    ),
    KernelPairInfo(
        "rust",
        "interpreter",
        "rust-interp-ffi",
        "rust-pure-interp",
    ),
    KernelPairInfo(
        "rust",
        "lowered",
        "rust-lowered-ffi",
        "rust-lowered-only",
    ),
    KernelPairInfo(
        "go",
        "accumulated",
        "go-wam-accumulated",
        "go-wam-accumulated-no-kernels",
    ),
    KernelPairInfo(
        "scala",
        "seeded",
        "scala-wam-seeded",
        "scala-wam-seeded-no-kernels",
    ),
    KernelPairInfo(
        "scala",
        "accumulated",
        "scala-wam-accumulated",
        "scala-wam-accumulated-no-kernels",
    ),
    KernelPairInfo(
        "clojure",
        "seeded",
        "clojure-wam-seeded",
        "clojure-wam-seeded-no-kernels",
    ),
    KernelPairInfo(
        "clojure",
        "accumulated",
        "clojure-wam-accumulated",
        "clojure-wam-accumulated-no-kernels",
    ),
    KernelPairInfo(
        "clojure",
        "seeded-artifact",
        "clojure-wam-seeded-artifact",
        "clojure-wam-seeded-no-kernels-artifact",
    ),
    KernelPairInfo(
        "clojure",
        "accumulated-artifact",
        "clojure-wam-accumulated-artifact",
        "clojure-wam-accumulated-no-kernels-artifact",
    ),
    KernelPairInfo(
        "haskell",
        "interpreter",
        "haskell-interp-ffi",
        "haskell-pure-interp",
    ),
    KernelPairInfo(
        "haskell",
        "lowered",
        "haskell-lowered-ffi",
        "haskell-lowered-only",
    ),
)


TARGET_SETS: dict[str, list[str]] = {
    "termux-smoke": [
        "prolog-accumulated",
        "go-wam-accumulated-no-kernels",
        "go-dfs",
    ],
    "optimized-prolog": [
        "prolog-seeded",
        "prolog-accumulated",
        "haskell-lowered-only",
        "haskell-lowered-ffi",
        "rust-lowered-only",
        "rust-lowered-ffi",
    ],
    "hybrid-wam": [
        "wam-rust-seeded",
        "wam-rust-accumulated",
        "wam-rust-seeded-no-kernels",
        "wam-rust-accumulated-no-kernels",
        "go-wam-accumulated",
        "go-wam-accumulated-no-kernels",
        "scala-wam-seeded",
        "scala-wam-seeded-no-kernels",
        "scala-wam-accumulated",
        "scala-wam-accumulated-no-kernels",
        "clojure-wam-accumulated",
        "clojure-wam-accumulated-no-kernels",
        "clojure-wam-seeded",
        "clojure-wam-seeded-no-kernels",
        "haskell-pure-interp",
        "haskell-interp-ffi",
        "haskell-lowered-only",
        "haskell-lowered-ffi",
        "rust-pure-interp",
        "rust-interp-ffi",
        "rust-lowered-only",
        "rust-lowered-ffi",
    ],
    "clojure-wam-scaffold": [],
    "scala-wam": [
        "scala-wam-seeded",
        "scala-wam-seeded-no-kernels",
        "scala-wam-accumulated",
        "scala-wam-accumulated-no-kernels",
    ],
    "scala-wam-artifact": [
        "scala-wam-seeded",
        "scala-wam-seeded-artifact",
        "scala-wam-accumulated",
        "scala-wam-accumulated-artifact",
    ],
    "clojure-wam": [
        "clojure-wam-accumulated",
        "clojure-wam-seeded",
        "clojure-wam-seeded-no-kernels",
        "clojure-wam-accumulated-no-kernels",
    ],
    "clojure-wam-artifact": [
        "clojure-wam-accumulated",
        "clojure-wam-accumulated-artifact",
        "clojure-wam-accumulated-no-kernels",
        "clojure-wam-accumulated-no-kernels-artifact",
        "clojure-wam-seeded",
        "clojure-wam-seeded-artifact",
        "clojure-wam-seeded-no-kernels",
        "clojure-wam-seeded-no-kernels-artifact",
    ],
    "direct-pipeline": [
        "rust-dfs",
        "go-dfs",
        "csharp-dfs",
    ],
    "query-engine": [
        "csharp-query",
    ],
    "portable-default": [
        "prolog-accumulated",
        "wam-rust-accumulated",
        "go-wam-accumulated",
        "haskell-lowered-ffi",
        "rust-dfs",
        "go-dfs",
    ],
    "desktop-default": [
        "prolog-accumulated",
        "wam-rust-accumulated",
        "go-wam-accumulated",
        "haskell-lowered-ffi",
        "rust-dfs",
        "go-dfs",
        "csharp-query",
        "csharp-dfs",
    ],
}

TARGET_SETS["all"] = list(TARGETS.keys())


def default_target_set_name() -> str:
    return "termux-smoke" if is_termux_environment() else "desktop-default"


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def list_targets_text() -> str:
    lines = ["name\tcategory\tdescription"]
    for name in sorted(TARGETS):
        info = TARGETS[name]
        lines.append(f"{info.name}\t{info.category}\t{info.description}")
    lines.append("")
    lines.append("target_set\ttargets")
    for set_name in sorted(TARGET_SETS):
        lines.append(f"{set_name}\t{','.join(TARGET_SETS[set_name])}")
    return "\n".join(lines)


def list_kernel_pairs_text() -> str:
    lines = ["family\tmode\tkernels_target\tno_kernels_target"]
    for pair in KERNEL_TARGET_PAIRS:
        lines.append(
            f"{pair.family}\t{pair.mode}\t{pair.kernels_target}\t{pair.no_kernels_target}"
        )
    return "\n".join(lines)


def resolve_targets(
    *,
    explicit_targets: list[str] | None,
    target_set_names: list[str] | None,
    include_targets: list[str] | None = None,
    exclude_targets: list[str] | None = None,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add_target(name: str) -> None:
        if name not in TARGETS:
            raise ValueError(f"unknown benchmark target: {name}")
        if name not in seen:
            ordered.append(name)
            seen.add(name)

    if explicit_targets:
        for name in explicit_targets:
            add_target(name)
    else:
        set_names = target_set_names or [default_target_set_name()]
        for set_name in set_names:
            if set_name not in TARGET_SETS:
                raise ValueError(f"unknown target set: {set_name}")
            for name in TARGET_SETS[set_name]:
                add_target(name)

    for name in include_targets or []:
        add_target(name)

    excluded = set(exclude_targets or [])
    unknown_excluded = sorted(name for name in excluded if name not in TARGETS)
    if unknown_excluded:
        raise ValueError(f"unknown benchmark targets in exclude list: {', '.join(unknown_excluded)}")
    return [name for name in ordered if name not in excluded]
