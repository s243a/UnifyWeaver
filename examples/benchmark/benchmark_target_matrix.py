#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

from benchmark_common import is_termux_environment


@dataclass(frozen=True)
class TargetInfo:
    name: str
    category: str
    description: str


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
}


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
    ],
    "hybrid-wam": [
        "wam-rust-seeded",
        "wam-rust-accumulated",
        "go-wam-accumulated",
        "go-wam-accumulated-no-kernels",
        "haskell-pure-interp",
        "haskell-interp-ffi",
        "haskell-lowered-only",
        "haskell-lowered-ffi",
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
