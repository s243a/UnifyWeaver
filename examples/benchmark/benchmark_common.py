#!/usr/bin/env python3
from __future__ import annotations

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
        if target.startswith("rust-") and shutil.which("rustc") is None:
            print(f"skip {target}: rustc not found", file=sys.stderr)
            continue
        if target.startswith("go-") and shutil.which("go") is None:
            print(f"skip {target}: go not found", file=sys.stderr)
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
