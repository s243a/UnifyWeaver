#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


SCALE_MAP = {
    "300": 300,
    "1k": 1000,
    "5k": 5000,
    "10k": 10000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic dependency DAG benchmark data")
    parser.add_argument("--output-dir", required=True, help="Directory to write benchmark TSV/facts files into")
    parser.add_argument("--scale", help="Named scale such as 300, 1k, 5k, 10k")
    parser.add_argument("--packages", type=int, help="Override package count directly")
    parser.add_argument("--projects", type=int, help="Override project count")
    return parser.parse_args()


def package_count_from_args(args: argparse.Namespace) -> int:
    if args.packages is not None:
        return args.packages
    if args.scale is not None:
        if args.scale not in SCALE_MAP:
            raise SystemExit(f"Unknown scale {args.scale!r}; expected one of {', '.join(SCALE_MAP)}")
        return SCALE_MAP[args.scale]
    raise SystemExit("One of --scale or --packages is required")


def build_dependency_edges(package_count: int) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for i in range(1, package_count + 1):
        child = f"pkg_{i:05d}"
        parents: list[int] = []
        if i > 1:
            parents.append(i - 1)
        if i > 2:
            parents.append(i // 2)
        if i > 5 and i % 3 == 0:
            parents.append(i - 3)
        if i > 9 and i % 5 == 0:
            parents.append(i - 7)
        for parent_idx in sorted(set(p for p in parents if p >= 1 and p < i)):
            edges.append((child, f"pkg_{parent_idx:05d}"))
    return edges


def build_project_deps(package_count: int, project_count: int) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    span = max(11, package_count // max(1, project_count))
    for i in range(1, project_count + 1):
        project = f"proj_{i:05d}"
        start = max(1, package_count - (i - 1) * span)
        deps = [start, max(1, start - 3), max(1, start - 9)]
        for dep_idx in sorted(set(deps), reverse=True):
            edges.append((project, f"pkg_{dep_idx:05d}"))
    return edges


def write_tsv(path: Path, header: tuple[str, str], rows: list[tuple[str, str]]) -> None:
    lines = ["\t".join(header)]
    lines.extend(f"{left}\t{right}" for left, right in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_facts(path: Path, project_deps: list[tuple[str, str]], package_edges: list[tuple[str, str]]) -> None:
    lines: list[str] = []
    for project, dep in project_deps:
        lines.append(f"article_category('{project}', '{dep}').")
    for child, parent in package_edges:
        lines.append(f"category_parent('{child}', '{parent}').")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    package_count = package_count_from_args(args)
    project_count = args.projects if args.projects is not None else max(12, package_count // 20)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    package_edges = build_dependency_edges(package_count)
    project_deps = build_project_deps(package_count, project_count)

    write_tsv(out_dir / "category_parent.tsv", ("child", "parent"), package_edges)
    write_tsv(out_dir / "article_category.tsv", ("article", "category"), project_deps)
    write_facts(out_dir / "facts.pl", project_deps, package_edges)

    print(f"Generated dependency benchmark data in {out_dir}")
    print(f"  packages={package_count}")
    print(f"  projects={project_count}")
    print(f"  dependency_edges={len(package_edges)}")
    print(f"  project_edges={len(project_deps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
