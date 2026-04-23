#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=check,
        capture_output=capture_output,
        text=True,
    )


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def is_termux_environment() -> bool:
    prefix = os.environ.get("PREFIX", "")
    return "com.termux" in prefix or "termux" in prefix.lower() or "TERMUX_VERSION" in os.environ


def scale_sort_key(scale: str) -> tuple[int, str]:
    digits = "".join(ch for ch in scale if ch.isdigit())
    suffix = "".join(ch for ch in scale if not ch.isdigit())
    if not digits:
        return (0, scale)
    value = int(digits)
    if suffix.lower() == "k":
        value *= 1000
    return (value, scale)


def available_targets(requested: list[str]) -> list[str]:
    targets: list[str] = []
    for target in requested:
        if target.startswith("csharp-") and shutil.which("dotnet") is None:
            print(f"skip {target}: dotnet not found", file=sys.stderr)
            continue
        if target.startswith("haskell-") and (shutil.which("cabal") is None or shutil.which("ghc") is None):
            print(f"skip {target}: cabal or ghc not found", file=sys.stderr)
            continue
        if target.startswith("rust-") and shutil.which("rustc") is None:
            print(f"skip {target}: rustc not found", file=sys.stderr)
            continue
        if target.startswith("go-") and shutil.which("go") is None:
            print(f"skip {target}: go not found", file=sys.stderr)
            continue
        if target.startswith("prolog-") and shutil.which("swipl") is None:
            print(f"skip {target}: swipl not found", file=sys.stderr)
            continue
        if target.startswith("wam-") and (shutil.which("swipl") is None or shutil.which("cargo") is None):
            print(f"skip {target}: swipl or cargo not found", file=sys.stderr)
            continue
        if target.startswith("go-wam-") and (shutil.which("swipl") is None or shutil.which("go") is None):
            print(f"skip {target}: swipl or go not found", file=sys.stderr)
            continue
        if target.startswith("clojure-wam-") and (shutil.which("swipl") is None or shutil.which("java") is None):
            print(f"skip {target}: swipl or java not found", file=sys.stderr)
            continue
        targets.append(target)
    return targets


def generate_pipeline_source(
    generator: Path,
    facts_path: Path,
    workload: str,
    target: str,
    output_dir: Path,
    *,
    root: str | None = None,
    output_name: str | None = None,
) -> Path:
    ext = {"csharp": ".cs", "rust": ".rs", "go": ".go", "csharp_query": ".cs"}[target]
    filename = output_name or ("Program.cs" if target.startswith("csharp") else f"{workload}{ext}")
    output = output_dir / filename
    cmd = [
        sys.executable,
        str(generator),
        "--facts",
        str(facts_path),
    ]
    if root is not None:
        cmd.extend(["--root", root])
    cmd.extend([
        "--workload",
        workload,
        "--target",
        target,
        "--output",
        str(output),
    ])
    run_command(cmd)
    return output


def generate_pipeline_package(
    generator: Path,
    facts_path: Path,
    workload: str,
    target: str,
    output_dir: Path,
    *,
    root: str | None = None,
) -> Path:
    cmd = [
        sys.executable,
        str(generator),
        "--facts",
        str(facts_path),
    ]
    if root is not None:
        cmd.extend(["--root", root])
    cmd.extend([
        "--workload",
        workload,
        "--target",
        target,
        "--output-dir",
        str(output_dir),
    ])
    run_command(cmd)
    return output_dir


def build_csharp_package(
    generator: Path,
    facts_path: Path,
    workload: str,
    target: str,
    project_dir: Path,
    *,
    root: str | None = None,
) -> list[str]:
    generate_pipeline_package(generator, facts_path, workload, target, project_dir, root=root)
    run_command(["dotnet", "build", "benchmark.csproj", "-c", "Release"], cwd=project_dir)
    return ["dotnet", str(project_dir / "bin" / "Release" / "net9.0" / "benchmark.dll")]


def build_rust_binary(
    generator: Path,
    facts_path: Path,
    workload: str,
    project_dir: Path,
    binary_name: str,
    *,
    root: str | None = None,
    source_name: str | None = None,
) -> list[str]:
    project_dir.mkdir(parents=True, exist_ok=True)
    source = generate_pipeline_source(
        generator,
        facts_path,
        workload,
        "rust",
        project_dir,
        root=root,
        output_name=source_name,
    )
    binary = project_dir / binary_name
    run_command(["rustc", "-O", str(source), "-o", str(binary)])
    return [str(binary)]


def build_go_binary(
    generator: Path,
    facts_path: Path,
    workload: str,
    project_dir: Path,
    binary_name: str,
    *,
    root: str | None = None,
    source_name: str | None = None,
) -> list[str]:
    project_dir.mkdir(parents=True, exist_ok=True)
    source = generate_pipeline_source(
        generator,
        facts_path,
        workload,
        "go",
        project_dir,
        root=root,
        output_name=source_name,
    )
    binary = project_dir / binary_name
    go_cache = project_dir / ".gocache"
    go_cache.mkdir(exist_ok=True)
    env = dict(os.environ, GOCACHE=str(go_cache))
    run_command(["go", "build", "-o", str(binary), str(source)], env=env)
    return [str(binary)]


def build_haskell_project(project_dir: Path, executable_name: str) -> list[str]:
    run_command(["cabal", "build", f"exe:{executable_name}"], cwd=project_dir)
    result = run_command(["cabal", "list-bin", f"exe:{executable_name}"], cwd=project_dir)
    binary = result.stdout.strip()
    if not binary:
        raise RuntimeError(f"could not resolve cabal binary for {executable_name}")
    return [binary]


def digest_normalized_output(normalized: str) -> tuple[str, int]:
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    row_count = max(0, len(normalized.splitlines()) - 1)
    return digest, row_count


def normalize_sorted_lines(output: str) -> str:
    lines = output.splitlines()
    header = lines[:1]
    body = sorted(lines[1:])
    return "\n".join(header + body)


def normalize_two_column_float_rows(
    output: str,
    *,
    decimals: int = 9,
    descending_numeric: bool = False,
) -> str:
    lines = output.splitlines()
    if not lines:
        return ""
    header = lines[0]
    rows: list[tuple[str, float]] = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        rows.append((parts[0], round(float(parts[1]), decimals)))
    rows.sort(key=lambda item: ((-item[1]) if descending_numeric else item[1], item[0]))
    normalized = [header]
    for label, value in rows:
        normalized.append(f"{label}\t{value:.{decimals}f}")
    return "\n".join(normalized)


def normalize_three_column_float_rows(
    output: str,
    *,
    decimals: int = 9,
) -> str:
    lines = output.splitlines()
    if not lines:
        return ""
    header = lines[0]
    rows: list[tuple[str, str, float]] = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        rows.append((parts[0], parts[1], round(float(parts[2]), decimals)))
    rows.sort(key=lambda item: (item[2], item[0], item[1]))
    normalized = [header]
    for col1, col2, value in rows:
        normalized.append(f"{col1}\t{col2}\t{value:.{decimals}f}")
    return "\n".join(normalized)


def three_column_parity_vs_reference(
    normalized_output: str,
    reference_path: Path,
    *,
    decimals: int = 6,
) -> str:
    """Compare a normalized three-column benchmark output against a
    native-SWI reference TSV. Returns a short status string suitable for
    a benchmark-table column:

      - "match"       — byte-for-byte identical after normalization.
      - "no-ref"      — reference_path does not exist.
      - "diff:<N>/ours:<A>/ref:<B>"
                      — N articles shared but with different deff values,
                        A articles only in ours, B articles only in ref.
    """
    if not reference_path.exists():
        return "no-ref"
    try:
        reference_raw = reference_path.read_text(encoding="utf-8")
    except OSError:
        return "no-ref"
    reference_normalized = normalize_three_column_float_rows(
        reference_raw, decimals=decimals
    )
    if normalized_output == reference_normalized:
        return "match"

    our_rows = normalized_output.splitlines()[1:]
    ref_rows = reference_normalized.splitlines()[1:]
    our_by_key = {row.split("\t")[0]: row for row in our_rows if "\t" in row}
    ref_by_key = {row.split("\t")[0]: row for row in ref_rows if "\t" in row}

    only_ours = len(set(our_by_key) - set(ref_by_key))
    only_ref = len(set(ref_by_key) - set(our_by_key))
    shared = set(our_by_key) & set(ref_by_key)
    diff_rows = sum(1 for k in shared if our_by_key[k] != ref_by_key[k])
    return f"diff:{diff_rows}/ours:{only_ours}/ref:{only_ref}"


def group_results_by_scale(results: list[object], sort_key=scale_sort_key) -> list[tuple[str, list[object]]]:
    by_scale: dict[str, list[object]] = {}
    for result in results:
        by_scale.setdefault(result.scale, []).append(result)
    return [(scale, by_scale[scale]) for scale in sorted(by_scale.keys(), key=sort_key)]


def print_result_table(entries: list[object], scale: str) -> None:
    for result in sorted(entries, key=lambda item: item.target):
        print(
            f"{scale}\t{result.target}\t{result.median:.3f}\t"
            f"{min(result.times):.3f}\t{max(result.times):.3f}\t"
            f"{result.row_count}\t{result.stdout_sha256[:12]}"
        )


def find_result(entries: list[object], target: str) -> object | None:
    return next((item for item in entries if item.target == target), None)


def print_match_status(scale: str, label: str, entries: list[object]) -> None:
    hashes = {item.stdout_sha256 for item in entries}
    print(f"{scale}\t{label}\t{'match' if len(hashes) == 1 else 'MISMATCH'}")


def print_pair_match_status(scale: str, label: str, left: object | None, right: object | None) -> None:
    if left and right:
        print(f"{scale}\t{label}\t{'match' if left.stdout_sha256 == right.stdout_sha256 else 'DIFFERENT'}")


def print_speedup(scale: str, label: str, faster_baseline: object | None, measured: object | None) -> None:
    if faster_baseline and measured:
        print(f"{scale}\t{label}\t{faster_baseline.median / measured.median:.2f}x")


def print_phase_metrics(scale: str, label: str, result: object | None) -> None:
    if result and result.stderr:
        phase_lines = [line.strip() for line in result.stderr.splitlines() if "=" in line]
        if phase_lines:
            print(f"{scale}\t{label}\t" + " ".join(phase_lines))
