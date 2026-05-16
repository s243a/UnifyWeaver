#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
CATEGORY_ONLY_GENERATOR = ROOT / "examples" / "benchmark" / "generate_category_only_benchmark.py"
LMDB_PROJECT = (
    ROOT
    / "src"
    / "unifyweaver"
    / "targets"
    / "csharp_query_runtime_lmdb"
    / "UnifyWeaver.QueryRuntime.Lmdb.csproj"
)
CORE_DLL = (
    ROOT
    / "src"
    / "unifyweaver"
    / "targets"
    / "csharp_query_runtime"
    / "bin"
    / "Debug"
    / "net9.0"
    / "UnifyWeaver.QueryRuntime.Core.dll"
)
LMDB_DLL = (
    ROOT
    / "src"
    / "unifyweaver"
    / "targets"
    / "csharp_query_runtime_lmdb"
    / "bin"
    / "Debug"
    / "net9.0"
    / "UnifyWeaver.QueryRuntime.Lmdb.dll"
)
LIGHTNINGDB_PACKAGE = Path.home() / ".nuget" / "packages" / "lightningdb" / "0.21.0"
LIGHTNINGDB_DLL = LIGHTNINGDB_PACKAGE / "lib" / "net9.0" / "LightningDB.dll"
AUTO_PREPARED_LARGE_SCALES = ("50k_cats", "100k_cats")
PREPARED_LARGE_SCALE_LABELS = AUTO_PREPARED_LARGE_SCALES + ("500k_cats", "1m_cats")
DEFAULT_DB_CANDIDATES = (
    ROOT / "data" / "simplewiki" / "simplewiki_categories.db",
    ROOT / "context" / "gemini" / "UnifyWeaver" / "data" / "simplewiki" / "simplewiki_categories.db",
    ROOT.parent.parent / "gemini" / "UnifyWeaver" / "data" / "simplewiki" / "simplewiki_categories.db",
)


HEADERS = [
    "scale",
    "run",
    "mode",
    "rows",
    "distinct_categories",
    "lookup_keys",
    "artifact_bytes",
    "open_ms",
    "lookup_ms",
    "lookup_col1_ms",
    "bucket_ms",
    "bucket_col1_ms",
    "scan_ms",
    "retained_bytes",
    "scan_hash",
    "lookup_hash",
    "lookup_col1_hash",
    "bucket_hash",
    "bucket_col1_hash",
]

RAW_HEADERS = [column for column in HEADERS if column != "run"]

SUMMARY_HEADERS = [
    "scale",
    "rows",
    "distinct_categories",
    "lookup_keys",
    "best_lookup_mode",
    "best_lookup_col1_mode",
    "best_bucket_mode",
    "best_bucket_col1_mode",
    "best_scan_mode",
    "smallest_artifact_mode",
    "lookup_ms_by_mode",
    "lookup_col1_ms_by_mode",
    "bucket_ms_by_mode",
    "bucket_col1_ms_by_mode",
    "scan_ms_by_mode",
    "artifact_bytes_by_mode",
]


@dataclass(frozen=True)
class BenchmarkRow:
    values: dict[str, str]


@dataclass(frozen=True)
class SummaryRow:
    values: dict[str, str]


def run_checked(command: list[str], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def build_lmdb_runtime() -> None:
    if shutil.which("dotnet") is None:
        raise RuntimeError("dotnet is not available")
    if not LIGHTNINGDB_PACKAGE.exists():
        raise RuntimeError("LightningDB 0.21.0 package is not available in the local NuGet cache")

    run_checked(["dotnet", "build", str(LMDB_PROJECT)], cwd=ROOT, timeout=120)
    for path in (CORE_DLL, LMDB_DLL, LIGHTNINGDB_DLL):
        if not path.exists():
            raise RuntimeError(f"required C# benchmark dependency is missing: {path}")


def write_benchmark_project(project_dir: Path) -> Path:
    project_path = project_dir / "CSharpQueryEffectiveDistanceArtifactBackends.csproj"
    program_path = project_dir / "Program.cs"
    project_path.write_text(
        textwrap.dedent(
            f"""\
            <Project Sdk="Microsoft.NET.Sdk">
              <PropertyGroup>
                <OutputType>Exe</OutputType>
                <TargetFramework>net9.0</TargetFramework>
                <Nullable>enable</Nullable>
                <ImplicitUsings>enable</ImplicitUsings>
                <NuGetAudit>false</NuGetAudit>
              </PropertyGroup>
              <ItemGroup>
                <Reference Include="UnifyWeaver.QueryRuntime.Core">
                  <HintPath>{CORE_DLL}</HintPath>
                </Reference>
                <Reference Include="UnifyWeaver.QueryRuntime.Lmdb">
                  <HintPath>{LMDB_DLL}</HintPath>
                </Reference>
                <Reference Include="LightningDB">
                  <HintPath>{LIGHTNINGDB_DLL}</HintPath>
                </Reference>
              </ItemGroup>
            </Project>
            """
        ),
        encoding="utf-8",
    )
    program_path.write_text(BENCHMARK_PROGRAM, encoding="utf-8")
    return project_path


def parse_rows(output: str, run_index: int) -> list[BenchmarkRow]:
    reader = csv.DictReader(output.splitlines(), delimiter="\t")
    if reader.fieldnames != RAW_HEADERS:
        raise RuntimeError(f"unexpected benchmark headers: {reader.fieldnames}")
    rows = []
    for row in reader:
        values = dict(row)
        values["run"] = str(run_index)
        rows.append(BenchmarkRow(values))
    return rows


def run_scale(
    scale: str,
    benchmark_root: Path,
    lookup_keys: int,
    lookup_repetitions: int,
    run_index: int,
    keep_temp: bool,
    artifact_root: Path | None,
    refresh_artifacts: bool,
    use_scale_lmdb_artifact: bool,
    preserve_numeric_ids: bool,
    lmdb_only: bool,
) -> list[BenchmarkRow]:
    scale_dir = benchmark_root / scale
    if not lmdb_only and not (scale_dir / "category_parent.tsv").exists():
        raise RuntimeError(f"scale has no category_parent.tsv: {scale_dir}")
    if lmdb_only and not (scale_dir / "category_parent.lmdb.manifest.json").exists():
        raise RuntimeError(f"scale has no category_parent.lmdb.manifest.json: {scale_dir}")

    build_lmdb_runtime()
    temp_path = Path(tempfile.mkdtemp(prefix=f"uw-csharp-effective-distance-artifacts-{scale}-"))
    try:
        project_path = write_benchmark_project(temp_path)
        run_checked(["dotnet", "build", str(project_path)], cwd=temp_path, timeout=120)

        output_dir = temp_path / "bin" / "Debug" / "net9.0"
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (
            str(output_dir)
            if not env.get("LD_LIBRARY_PATH")
            else f"{output_dir}:{env['LD_LIBRARY_PATH']}"
        )
        result = run_checked(
            [
                "dotnet",
                str(output_dir / "CSharpQueryEffectiveDistanceArtifactBackends.dll"),
                "--scale",
                scale,
                "--scale-dir",
                str(scale_dir),
                "--lookup-keys",
                str(lookup_keys),
                "--lookup-repetitions",
                str(lookup_repetitions),
                "--artifact-root",
                str((artifact_root / scale) if artifact_root is not None else temp_path),
                "--refresh-artifacts",
                "true" if refresh_artifacts else "false",
                "--use-scale-lmdb-artifact",
                "true" if use_scale_lmdb_artifact else "false",
                "--preserve-numeric-ids",
                "true" if preserve_numeric_ids else "false",
                "--lmdb-only",
                "true" if lmdb_only else "false",
            ],
            cwd=temp_path,
            timeout=240,
        )
        return parse_rows(result.stdout, run_index)
    finally:
        if keep_temp:
            print(f"kept benchmark project: {temp_path}", file=sys.stderr)
        else:
            shutil.rmtree(temp_path, ignore_errors=True)


def render_tsv(rows: list[BenchmarkRow]) -> str:
    lines = ["\t".join(HEADERS)]
    lines.extend("\t".join(row.values[column] for column in HEADERS) for row in rows)
    return "\n".join(lines)


def render_markdown(rows: list[BenchmarkRow]) -> str:
    lines = [
        "| Scale | Run | Mode | Rows | Categories | Lookup keys | Artifact bytes | Open ms | Lookup c0 ms | Lookup c1 ms | Bucket c0 ms | Bucket c1 ms | Scan ms | Retained bytes |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        value = row.values
        lines.append(
            "| {scale} | {run} | {mode} | {rows} | {distinct_categories} | {lookup_keys} | {artifact_bytes} | {open_ms} | {lookup_ms} | {lookup_col1_ms} | {bucket_ms} | {bucket_col1_ms} | {scan_ms} | {retained_bytes} |".format(
                **value
            )
        )
    return "\n".join(lines)


def numeric(row: BenchmarkRow, column: str) -> float:
    return float(row.values[column])


def scale_numeric_value(scale: str) -> int | None:
    label = scale.strip().lower()
    multiplier = 1
    if label.endswith("k"):
        multiplier = 1_000
        label = label[:-1]
    elif label.endswith("m"):
        multiplier = 1_000_000
        label = label[:-1]

    try:
        return int(float(label) * multiplier)
    except ValueError:
        return None


def scale_sort_key(scale: str) -> tuple[int, str]:
    numeric_value = scale_numeric_value(scale)
    return (numeric_value if numeric_value is not None else sys.maxsize, scale)


def is_large_scale(scale: str) -> bool:
    numeric_value = scale_numeric_value(scale)
    return (numeric_value is not None and numeric_value >= 50_000) or scale in PREPARED_LARGE_SCALE_LABELS


def scale_seed_cap(scale: str) -> int | None:
    if scale == "50k_cats":
        return 50_000
    if scale == "100k_cats":
        return None
    raise ValueError(
        f"automatic fixture preparation only supports {', '.join(AUTO_PREPARED_LARGE_SCALES)}; "
        f"prepare data/benchmark/{scale} separately or omit --prepare-missing-large-scales"
    )


def resolve_simplewiki_db(path: Path | None) -> Path:
    if path is not None:
        if path.exists():
            return path
        raise FileNotFoundError(path)
    for candidate in DEFAULT_DB_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = "\n".join(f"  - {candidate}" for candidate in DEFAULT_DB_CANDIDATES)
    raise FileNotFoundError(
        "simplewiki_categories.db not found. Run examples/benchmark/parse_simplewiki_dump.py "
        "or pass --db.\nSearched:\n" + searched
    )


def fixture_ready(scale: str) -> bool:
    scale_dir = BENCH_DIR / scale
    return (scale_dir / "category_parent.tsv").exists() and (scale_dir / "facts.pl").exists()


def prepare_large_fixture(scale: str, db_path: Path) -> None:
    if fixture_ready(scale):
        print(f"[fixture] reuse {BENCH_DIR / scale}", file=sys.stderr)
        return
    cmd = [
        sys.executable,
        str(CATEGORY_ONLY_GENERATOR),
        "--output",
        str(BENCH_DIR / scale),
        "--db",
        str(db_path),
    ]
    cap = scale_seed_cap(scale)
    if cap is not None:
        cmd.extend(["--max-seeds", str(cap)])
    print(f"[fixture] generate {scale}", file=sys.stderr)
    run_checked(cmd, cwd=ROOT, timeout=600)


def parse_mem_available_mib(meminfo_text: str) -> int | None:
    for line in meminfo_text.splitlines():
        if not line.startswith("MemAvailable:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            return int(parts[1]) // 1024
        except ValueError:
            return None
    return None


def read_mem_available_mib(meminfo_path: Path = Path("/proc/meminfo")) -> int | None:
    try:
        return parse_mem_available_mib(meminfo_path.read_text(encoding="utf-8"))
    except OSError:
        return None


def competing_processes(cpu_threshold: float) -> list[str]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,%cpu=,args="],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    processes = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            cpu = float(parts[1])
        except ValueError:
            continue
        if pid == os.getpid() or cpu < cpu_threshold:
            continue
        processes.append(f"competing process {pid} uses {cpu:.1f}% CPU: {parts[2]}")
    return processes


def resource_preflight_failures(
    *,
    min_free_memory_mib: int,
    require_idle: bool,
    max_competing_cpu_percent: float,
) -> list[str]:
    failures: list[str] = []
    memory_mib = read_mem_available_mib()
    if memory_mib is None:
        failures.append("could not determine available memory from /proc/meminfo")
    elif memory_mib < min_free_memory_mib:
        failures.append(
            f"available memory {memory_mib} MiB is below required {min_free_memory_mib} MiB"
        )

    if require_idle:
        failures.extend(competing_processes(max_competing_cpu_percent))
    return failures


def summarize(rows: list[BenchmarkRow]) -> list[SummaryRow]:
    grouped: dict[str, list[BenchmarkRow]] = {}
    for row in rows:
        grouped.setdefault(row.values["scale"], []).append(row)

    summaries: list[SummaryRow] = []
    for scale in sorted(grouped, key=scale_sort_key):
        scale_rows = grouped[scale]
        modes = sorted({row.values["mode"] for row in scale_rows})

        def median_by_mode(column: str) -> dict[str, float]:
            return {
                mode: statistics.median(numeric(row, column) for row in scale_rows if row.values["mode"] == mode)
                for mode in modes
            }

        lookup = median_by_mode("lookup_ms")
        lookup_col1 = median_by_mode("lookup_col1_ms")
        bucket = median_by_mode("bucket_ms")
        bucket_col1 = median_by_mode("bucket_col1_ms")
        scan = median_by_mode("scan_ms")
        artifact = median_by_mode("artifact_bytes")
        row0 = scale_rows[0].values
        summaries.append(
            SummaryRow(
                {
                    "scale": scale,
                    "rows": row0["rows"],
                    "distinct_categories": row0["distinct_categories"],
                    "lookup_keys": row0["lookup_keys"],
                    "best_lookup_mode": min(lookup, key=lookup.get),
                    "best_lookup_col1_mode": min(lookup_col1, key=lookup_col1.get),
                    "best_bucket_mode": min(bucket, key=bucket.get),
                    "best_bucket_col1_mode": min(bucket_col1, key=bucket_col1.get),
                    "best_scan_mode": min(scan, key=scan.get),
                    "smallest_artifact_mode": min(
                        (mode for mode in artifact if mode != "preload"),
                        key=artifact.get,
                    ),
                    "lookup_ms_by_mode": format_mode_values(lookup),
                    "lookup_col1_ms_by_mode": format_mode_values(lookup_col1),
                    "bucket_ms_by_mode": format_mode_values(bucket),
                    "bucket_col1_ms_by_mode": format_mode_values(bucket_col1),
                    "scan_ms_by_mode": format_mode_values(scan),
                    "artifact_bytes_by_mode": format_mode_values(artifact, digits=0),
                }
            )
        )
    return summaries


def format_mode_values(values: dict[str, float], digits: int = 3) -> str:
    def format_value(value: float) -> str:
        return f"{value:.{digits}f}" if digits > 0 else str(int(value))

    return "|".join(f"{mode}:{format_value(values[mode])}" for mode in sorted(values))


def render_summary_tsv(rows: list[SummaryRow]) -> str:
    lines = ["\t".join(SUMMARY_HEADERS)]
    lines.extend("\t".join(row.values[column] for column in SUMMARY_HEADERS) for row in rows)
    return "\n".join(lines)


def render_summary_markdown(rows: list[SummaryRow]) -> str:
    lines = [
        "| Scale | Rows | Categories | Lookup keys | Best lookup c0 | Best lookup c1 | Best bucket c0 | Best bucket c1 | Best scan | Smallest artifact |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        value = row.values
        lines.append(
            "| {scale} | {rows} | {distinct_categories} | {lookup_keys} | {best_lookup_mode} | {best_lookup_col1_mode} | {best_bucket_mode} | {best_bucket_col1_mode} | {best_scan_mode} | {smallest_artifact_mode} |".format(
                **value
            )
        )
    lines.append("")
    lines.append("Median timing/detail fields are available in `summary-tsv`.")
    return "\n".join(lines)


def render_summary_full_markdown(rows: list[SummaryRow]) -> str:
    lines = [
        "| Scale | Rows | Categories | Lookup keys | Best lookup c0 | Best lookup c1 | Best bucket c0 | Best bucket c1 | Best scan | Smallest artifact | Lookup c0 ms by mode | Lookup c1 ms by mode | Bucket c0 ms by mode | Bucket c1 ms by mode | Scan ms by mode | Artifact bytes by mode |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        value = row.values
        lines.append(
            "| {scale} | {rows} | {distinct_categories} | {lookup_keys} | {best_lookup_mode} | {best_lookup_col1_mode} | {best_bucket_mode} | {best_bucket_col1_mode} | {best_scan_mode} | {smallest_artifact_mode} | {lookup_ms_by_mode} | {lookup_col1_ms_by_mode} | {bucket_ms_by_mode} | {bucket_col1_ms_by_mode} | {scan_ms_by_mode} | {artifact_bytes_by_mode} |".format(
                **value
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare C# query artifact backends on the real effective-distance category_parent/2 relation."
    )
    parser.add_argument("--scales", default="300,1k,5k,10k", help="comma-separated scales from data/benchmark")
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=BENCH_DIR,
        help="directory containing benchmark scale subdirectories",
    )
    parser.add_argument("--lookup-keys", type=int, default=64, help="number of category IDs to probe")
    parser.add_argument("--lookup-repetitions", type=int, default=5, help="lookup passes per mode")
    parser.add_argument("--repetitions", type=int, default=1, help="independent benchmark runs per scale")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="persistent artifact directory; when set, per-scale artifacts are reused across runs",
    )
    parser.add_argument(
        "--refresh-artifacts",
        action="store_true",
        help="rebuild artifacts under --artifact-root instead of reusing existing manifests",
    )
    parser.add_argument(
        "--use-scale-lmdb-artifact",
        action="store_true",
        help=(
            "prefer data/benchmark/<scale>/category_parent.lmdb.manifest.json for the LMDB mode "
            "when present; falls back to --artifact-root generation"
        ),
    )
    parser.add_argument(
        "--lmdb-only",
        action="store_true",
        help="benchmark only a scale-local LMDB artifact; does not require category_parent.tsv",
    )
    parser.add_argument(
        "--preserve-numeric-ids",
        action="store_true",
        help=(
            "preserve numeric category_parent.tsv IDs instead of re-interning them densely; "
            "implied by --use-scale-lmdb-artifact only when a scale-local LMDB manifest exists"
        ),
    )
    parser.add_argument(
        "--prepare-missing-large-scales",
        action="store_true",
        help="generate supported SimpleWiki category-only fixtures when 50k_cats/100k_cats are requested",
    )
    parser.add_argument(
        "--skip-missing-scales",
        action="store_true",
        help="skip requested scales whose data/benchmark/<scale>/category_parent.tsv is not present",
    )
    parser.add_argument("--db", type=Path, default=None, help="path to simplewiki_categories.db for fixture preparation")
    parser.add_argument(
        "--min-free-memory-mib",
        type=int,
        default=1024,
        help="minimum MemAvailable MiB required before running large scales",
    )
    parser.add_argument(
        "--skip-resource-check",
        action="store_true",
        help="skip the large-scale free-memory preflight check",
    )
    parser.add_argument(
        "--require-idle",
        action="store_true",
        help="for large scales, fail if competing CPU-heavy processes are active",
    )
    parser.add_argument(
        "--max-competing-cpu-percent",
        type=float,
        default=25.0,
        help="CPU-percent threshold for --require-idle competing-process detection",
    )
    parser.add_argument(
        "--format",
        choices=("tsv", "markdown", "summary-tsv", "summary-markdown", "summary-full-markdown"),
        default="tsv",
    )
    parser.add_argument("--keep-temp", action="store_true", help="keep the generated C# benchmark project")
    args = parser.parse_args(argv)

    if args.lookup_keys <= 0:
        parser.error("--lookup-keys must be positive")
    if args.lookup_repetitions <= 0:
        parser.error("--lookup-repetitions must be positive")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.min_free_memory_mib <= 0:
        parser.error("--min-free-memory-mib must be positive")
    if args.max_competing_cpu_percent < 0:
        parser.error("--max-competing-cpu-percent must be non-negative")

    scales = [scale.strip() for scale in args.scales.split(",") if scale.strip()]
    if not scales:
        parser.error("--scales must include at least one scale")
    large_scales = [scale for scale in scales if is_large_scale(scale)]

    if args.prepare_missing_large_scales:
        db_path = resolve_simplewiki_db(args.db)
        for scale in large_scales:
            if scale in AUTO_PREPARED_LARGE_SCALES:
                prepare_large_fixture(scale, db_path)

    missing = [
        scale
        for scale in scales
        if not (
            (args.benchmark_root / scale / "category_parent.tsv").exists()
            or (args.lmdb_only and (args.benchmark_root / scale / "category_parent.lmdb.manifest.json").exists())
        )
    ]
    if missing:
        if args.skip_missing_scales:
            print(f"skipping missing benchmark scale(s): {', '.join(missing)}", file=sys.stderr)
            scales = [scale for scale in scales if scale not in missing]
            large_scales = [scale for scale in scales if is_large_scale(scale)]
            if not scales:
                parser.error("all requested scales were missing after --skip-missing-scales")
        else:
            hints = []
            supported_missing = [scale for scale in missing if scale in AUTO_PREPARED_LARGE_SCALES]
            prepared_only_missing = [scale for scale in missing if scale in PREPARED_LARGE_SCALE_LABELS]
            if supported_missing:
                hints.append("pass --prepare-missing-large-scales to generate supported SimpleWiki category-only fixtures")
            if prepared_only_missing:
                hints.append("500k_cats and 1m_cats must be prepared separately from enwiki-scale data before this wrapper can run them")
            hints.append("or prepare data/benchmark/<scale>/category_parent.tsv before running")
            parser.error(f"missing benchmark scale(s): {', '.join(missing)}; {'; '.join(hints)}")

    if large_scales and not args.skip_resource_check:
        failures = resource_preflight_failures(
            min_free_memory_mib=args.min_free_memory_mib,
            require_idle=args.require_idle,
            max_competing_cpu_percent=args.max_competing_cpu_percent,
        )
        if failures:
            joined = "\n  - ".join(failures)
            parser.error(f"large-scale resource preflight failed:\n  - {joined}")

    rows: list[BenchmarkRow] = []
    for scale in scales:
        for run_index in range(1, args.repetitions + 1):
            rows.extend(
                run_scale(
                    scale,
                    args.benchmark_root,
                    args.lookup_keys,
                    args.lookup_repetitions,
                    run_index,
                    args.keep_temp,
                    args.artifact_root,
                    args.refresh_artifacts,
                    args.use_scale_lmdb_artifact,
                    args.preserve_numeric_ids,
                    args.lmdb_only,
                )
            )

    if args.format == "summary-tsv":
        print(render_summary_tsv(summarize(rows)))
    elif args.format == "summary-full-markdown":
        print(render_summary_full_markdown(summarize(rows)))
    elif args.format == "summary-markdown":
        print(render_summary_markdown(summarize(rows)))
    elif args.format == "markdown":
        print(render_markdown(rows))
    else:
        print(render_tsv(rows))
    return 0


BENCHMARK_PROGRAM = r"""
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using LightningDB;
using UnifyWeaver.QueryRuntime;

static string ReadArg(string[] args, string name, string defaultValue)
{
    for (var index = 0; index < args.Length - 1; index++)
    {
        if (args[index] == name)
        {
            return args[index + 1];
        }
    }

    return defaultValue;
}

static int ReadIntArg(string[] args, string name, int defaultValue)
{
    return int.TryParse(ReadArg(args, name, ""), out var parsed) ? parsed : defaultValue;
}

static bool ReadBoolArg(string[] args, string name, bool defaultValue)
{
    var raw = ReadArg(args, name, defaultValue ? "true" : "false");
    return raw.Equals("true", StringComparison.OrdinalIgnoreCase) ||
        raw.Equals("1", StringComparison.OrdinalIgnoreCase) ||
        raw.Equals("yes", StringComparison.OrdinalIgnoreCase);
}

static long DirectorySize(string path)
{
    if (!Directory.Exists(path))
    {
        return File.Exists(path) ? new FileInfo(path).Length : 0L;
    }

    return Directory.EnumerateFiles(path, "*", SearchOption.AllDirectories)
        .Select(file => new FileInfo(file).Length)
        .Sum();
}

static string HashRows(IEnumerable<object[]> rows)
{
    var texts = rows.Select(row => string.Join("\t", row.Select(value => Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture) ?? "")))
        .OrderBy(value => value, StringComparer.Ordinal);
    using var sha = SHA256.Create();
    foreach (var text in texts)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        sha.TransformBlock(bytes, 0, bytes.Length, null, 0);
        sha.TransformBlock(new byte[] { 10 }, 0, 1, null, 0);
    }
    sha.TransformFinalBlock(Array.Empty<byte>(), 0, 0);
    return Convert.ToHexString(sha.Hash!).ToLowerInvariant();
}

static double Measure(Action action)
{
    var stopwatch = Stopwatch.StartNew();
    action();
    stopwatch.Stop();
    return stopwatch.Elapsed.TotalMilliseconds;
}

static int Intern(Dictionary<string, int> ids, string value)
{
    if (ids.TryGetValue(value, out var id))
    {
        return id;
    }

    id = ids.Count;
    ids[value] = id;
    return id;
}

static IReadOnlyList<(int Child, int Parent)> ReadEdges(string scaleDir, bool preserveNumericIds, out int distinctCategories)
{
    if (preserveNumericIds)
    {
        return ReadNumericEdges(scaleDir, out distinctCategories);
    }

    return ReadAndInternEdges(scaleDir, out distinctCategories);
}

static IReadOnlyList<(int Child, int Parent)> ReadNumericEdges(string scaleDir, out int distinctCategories)
{
    var ids = new HashSet<int>();
    var rows = new List<(int Child, int Parent)>();
    foreach (var line in File.ReadLines(Path.Combine(scaleDir, "category_parent.tsv")).Skip(1))
    {
        if (string.IsNullOrWhiteSpace(line))
        {
            continue;
        }

        var tab = line.IndexOf('\t');
        if (tab < 0)
        {
            throw new InvalidDataException($"category_parent.tsv row has no tab: {line}");
        }

        var child = int.Parse(line.Substring(0, tab), System.Globalization.CultureInfo.InvariantCulture);
        var parent = int.Parse(line.Substring(tab + 1), System.Globalization.CultureInfo.InvariantCulture);
        rows.Add((child, parent));
        ids.Add(child);
        ids.Add(parent);
    }

    distinctCategories = ids.Count;
    return rows;
}

static IReadOnlyList<(int Child, int Parent)> ReadAndInternEdges(string scaleDir, out int distinctCategories)
{
    var ids = new Dictionary<string, int>(StringComparer.Ordinal);
    var rows = new List<(int Child, int Parent)>();
    foreach (var line in File.ReadLines(Path.Combine(scaleDir, "category_parent.tsv")).Skip(1))
    {
        if (string.IsNullOrWhiteSpace(line))
        {
            continue;
        }

        var tab = line.IndexOf('\t');
        if (tab < 0)
        {
            throw new InvalidDataException($"category_parent.tsv row has no tab: {line}");
        }

        var child = line.Substring(0, tab);
        var parent = line.Substring(tab + 1);
        rows.Add((Intern(ids, child), Intern(ids, parent)));
    }

    distinctCategories = ids.Count;
    return rows;
}

static void WriteInput(string path, IReadOnlyList<(int Child, int Parent)> rows)
{
    using var writer = new StreamWriter(path, append: false, Encoding.UTF8);
    writer.WriteLine("child\tparent");
    foreach (var row in rows)
    {
        writer.Write(row.Child.ToString(System.Globalization.CultureInfo.InvariantCulture));
        writer.Write('\t');
        writer.Write(row.Parent.ToString(System.Globalization.CultureInfo.InvariantCulture));
        writer.WriteLine();
    }
}

static string WriteLmdbArtifact(
    string root,
    PredicateId predicate,
    IReadOnlyList<(int Child, int Parent)> rows,
    string sourcePath)
{
    var envPath = Path.Combine(root, "lmdb-category-parent");
    if (Directory.Exists(envPath))
    {
        Directory.Delete(envPath, recursive: true);
    }
    Directory.CreateDirectory(envPath);
    var mapSize = Math.Max(256L * 1024L * 1024L, rows.Count * 256L);
    using (var env = new LightningEnvironment(envPath) { MaxDatabases = 4, MapSize = mapSize })
    {
        env.Open();
        using var tx = env.BeginTransaction();
        using var db = tx.OpenDatabase("main", new DatabaseConfiguration
        {
            Flags = DatabaseOpenFlags.Create | DatabaseOpenFlags.DuplicatesSort,
        });

        foreach (var row in rows)
        {
            tx.Put(
                db,
                Encoding.UTF8.GetBytes(row.Child.ToString(System.Globalization.CultureInfo.InvariantCulture)),
                Encoding.UTF8.GetBytes(row.Parent.ToString(System.Globalization.CultureInfo.InvariantCulture)));
        }

        tx.Commit();
    }

    var manifest = new LmdbRelationArtifactManifest
    {
        PredicateName = predicate.Name,
        Arity = predicate.Arity,
        EnvironmentPath = Path.GetFileName(envPath),
        DatabaseName = "main",
        DupSort = true,
        KeyEncoding = "utf8",
        ValueEncoding = "utf8",
        RowCount = rows.Count,
        SourcePath = sourcePath,
        SourceLength = new FileInfo(sourcePath).Length,
        SourceSha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(sourcePath))).ToLowerInvariant(),
    };
    var manifestPath = Path.Combine(root, "category_parent.lmdb.manifest.json");
    File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }), Encoding.UTF8);
    return manifestPath;
}

static string ExistingOrBuild(string manifestPath, bool refreshArtifacts, Func<string> build)
{
    if (!refreshArtifacts && File.Exists(manifestPath))
    {
        return manifestPath;
    }

    return build();
}

static string ResolveManifestEnvironmentPath(LmdbRelationArtifactManifest manifest, string manifestPath)
{
    return Path.IsPathRooted(manifest.EnvironmentPath)
        ? manifest.EnvironmentPath
        : Path.Combine(Path.GetDirectoryName(Path.GetFullPath(manifestPath)) ?? ".", manifest.EnvironmentPath);
}

static string? ExistingScaleLmdbManifest(string scaleDir)
{
    var manifestPath = Path.Combine(scaleDir, "category_parent.lmdb.manifest.json");
    if (!File.Exists(manifestPath))
    {
        return null;
    }

    var manifest = LmdbRelationArtifactReader.LoadManifest(manifestPath);
    if (manifest.PredicateName != "category_parent" || manifest.Arity != 2)
    {
        throw new InvalidDataException($"Scale-local LMDB manifest describes {manifest.PredicateName}/{manifest.Arity}, not category_parent/2: {manifestPath}");
    }

    var environmentPath = ResolveManifestEnvironmentPath(manifest, manifestPath);
    if (!Directory.Exists(environmentPath))
    {
        throw new DirectoryNotFoundException($"Scale-local LMDB manifest environment does not exist: {environmentPath}");
    }

    return manifestPath;
}

static object[][] ReadLmdbRowsForPlanning(PredicateId predicate, string manifestPath)
{
    var provider = new LmdbRelationProvider();
    provider.RegisterArtifact(predicate, manifestPath);
    return provider.GetFacts(predicate).ToArray();
}

static IReadOnlyList<object> LookupKeysFromRows(object[][] rows, int lookupKeyCount)
{
    return LookupKeysFromRowsByColumn(rows, 0, lookupKeyCount);
}

static IReadOnlyList<object> LookupKeysFromRowsByColumn(object[][] rows, int columnIndex, int lookupKeyCount)
{
    return rows.Select(row => row[columnIndex])
        .Distinct()
        .OrderBy(value => Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture), StringComparer.Ordinal)
        .Take(Math.Min(lookupKeyCount, rows.Length))
        .ToArray();
}

static int DistinctCategoryCount(object[][] rows)
{
    return rows.SelectMany(row => row.Take(2))
        .Select(value => Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture) ?? "")
        .Distinct(StringComparer.Ordinal)
        .Count();
}

static string HashBuckets(IEnumerable<IndexedRelationBucket> buckets, int valueColumnIndex)
{
    return HashRows(buckets.Select(bucket => new object[]
    {
        bucket.Key,
        bucket.Rows.Count,
        string.Join(",", bucket.Rows.Select(row => Convert.ToString(row[valueColumnIndex], System.Globalization.CultureInfo.InvariantCulture)).OrderBy(value => value, StringComparer.Ordinal))
    }));
}

static double MeasureLookup(
    IRelationProvider provider,
    PredicateId predicate,
    int columnIndex,
    IReadOnlyList<object> lookupKeys,
    int lookupRepetitions,
    List<object[]> lookupRows)
{
    return Measure(() =>
    {
        for (var iteration = 0; iteration < lookupRepetitions; iteration++)
        {
            if (provider is IIndexedRelationProvider indexed &&
                indexed.TryLookupFacts(predicate, columnIndex, lookupKeys, out var indexedRows))
            {
                lookupRows.AddRange(indexedRows);
            }
            else
            {
                var keySet = lookupKeys.Select(Convert.ToString).ToHashSet(StringComparer.Ordinal);
                lookupRows.AddRange(provider.GetFacts(predicate).Where(row => keySet.Contains(Convert.ToString(row[columnIndex]))));
            }
        }
    });
}

static double MeasureBuckets(
    IRelationProvider provider,
    PredicateId predicate,
    int columnIndex,
    int valueColumnIndex,
    out string bucketHash)
{
    var measuredBucketHash = "";
    var elapsedMs = Measure(() =>
    {
        if (provider is IIndexedRelationBucketProvider bucketProvider &&
            bucketProvider.TryReadIndexedBuckets(predicate, columnIndex, out var buckets))
        {
            measuredBucketHash = HashBuckets(buckets, valueColumnIndex);
        }
        else
        {
            measuredBucketHash = HashBuckets(provider.GetFacts(predicate)
                .GroupBy(row => row[columnIndex])
                .OrderBy(group => Convert.ToString(group.Key, System.Globalization.CultureInfo.InvariantCulture), StringComparer.Ordinal)
                .Select(group => new IndexedRelationBucket(group.Key, group.ToArray())), valueColumnIndex);
        }
    });
    bucketHash = measuredBucketHash;
    return elapsedMs;
}

static void Emit(
    string scale,
    string mode,
    int rowCount,
    int distinctCategories,
    int lookupKeys,
    long artifactBytes,
    double openMs,
    double lookupMs,
    double lookupCol1Ms,
    double bucketMs,
    double bucketCol1Ms,
    double scanMs,
    long retainedBytes,
    string scanHash,
    string lookupHash,
    string lookupCol1Hash,
    string bucketHash,
    string bucketCol1Hash)
{
    Console.WriteLine(string.Join('\t', new[]
    {
        scale,
        mode,
        rowCount.ToString(System.Globalization.CultureInfo.InvariantCulture),
        distinctCategories.ToString(System.Globalization.CultureInfo.InvariantCulture),
        lookupKeys.ToString(System.Globalization.CultureInfo.InvariantCulture),
        artifactBytes.ToString(System.Globalization.CultureInfo.InvariantCulture),
        openMs.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture),
        lookupMs.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture),
        lookupCol1Ms.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture),
        bucketMs.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture),
        bucketCol1Ms.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture),
        scanMs.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture),
        retainedBytes.ToString(System.Globalization.CultureInfo.InvariantCulture),
        scanHash,
        lookupHash,
        lookupCol1Hash,
        bucketHash,
        bucketCol1Hash,
    }));
}

static void BenchmarkProvider(
    string scale,
    string mode,
    Func<IRelationProvider> openProvider,
    PredicateId predicate,
    IReadOnlyList<object> lookupKeys,
    IReadOnlyList<object> lookupKeysColumn1,
    int lookupRepetitions,
    long artifactBytes,
    int rowCount,
    int distinctCategories)
{
    GC.Collect();
    GC.WaitForPendingFinalizers();
    GC.Collect();
    var before = GC.GetTotalMemory(forceFullCollection: true);
    IRelationProvider? provider = null;
    var openMs = Measure(() => provider = openProvider());
    var after = GC.GetTotalMemory(forceFullCollection: true);
    var retainedBytes = Math.Max(0L, after - before);

    var lookupRows = new List<object[]>();
    var lookupMs = MeasureLookup(provider!, predicate, 0, lookupKeys, lookupRepetitions, lookupRows);

    var lookupRowsColumn1 = new List<object[]>();
    var lookupCol1Ms = MeasureLookup(provider!, predicate, 1, lookupKeysColumn1, lookupRepetitions, lookupRowsColumn1);

    var bucketMs = MeasureBuckets(provider!, predicate, 0, 1, out var bucketHash);
    var bucketCol1Ms = MeasureBuckets(provider!, predicate, 1, 0, out var bucketCol1Hash);

    object[][] scanRows = Array.Empty<object[]>();
    var scanMs = Measure(() => scanRows = provider!.GetFacts(predicate).ToArray());

    Emit(
        scale,
        mode,
        rowCount,
        distinctCategories,
        lookupKeys.Count,
        artifactBytes,
        openMs,
        lookupMs,
        lookupCol1Ms,
        bucketMs,
        bucketCol1Ms,
        scanMs,
        retainedBytes,
        HashRows(scanRows),
        HashRows(lookupRows),
        HashRows(lookupRowsColumn1),
        bucketHash,
        bucketCol1Hash);
}

var scale = ReadArg(args, "--scale", "dev");
var scaleDir = ReadArg(args, "--scale-dir", "");
var lookupKeyCount = Math.Max(1, ReadIntArg(args, "--lookup-keys", 64));
var lookupRepetitions = Math.Max(1, ReadIntArg(args, "--lookup-repetitions", 5));
var root = Path.GetFullPath(ReadArg(args, "--artifact-root", Directory.GetCurrentDirectory()));
var refreshArtifacts = ReadBoolArg(args, "--refresh-artifacts", false);
var useScaleLmdbArtifact = ReadBoolArg(args, "--use-scale-lmdb-artifact", false);
var requestedPreserveNumericIds = ReadBoolArg(args, "--preserve-numeric-ids", false);
var lmdbOnly = ReadBoolArg(args, "--lmdb-only", false);
Directory.CreateDirectory(root);
var predicate = new PredicateId("category_parent", 2);
var scaleLmdbManifest = (!refreshArtifacts && (useScaleLmdbArtifact || lmdbOnly)) ? ExistingScaleLmdbManifest(scaleDir) : null;
if (lmdbOnly && scaleLmdbManifest is null)
{
    throw new InvalidOperationException("--lmdb-only requires a scale-local category_parent.lmdb.manifest.json");
}
if (lmdbOnly)
{
    var lmdbOnlyManifest = scaleLmdbManifest!;
    var lmdbOnlyMetadata = LmdbRelationArtifactReader.LoadManifest(lmdbOnlyManifest);
    var lmdbOnlyDir = ResolveManifestEnvironmentPath(lmdbOnlyMetadata, lmdbOnlyManifest);
    var planningRows = ReadLmdbRowsForPlanning(predicate, lmdbOnlyManifest);
    var lookupKeysOnly = LookupKeysFromRows(planningRows, lookupKeyCount);
    var lookupKeysColumn1Only = LookupKeysFromRowsByColumn(planningRows, 1, lookupKeyCount);
    Console.WriteLine("scale\tmode\trows\tdistinct_categories\tlookup_keys\tartifact_bytes\topen_ms\tlookup_ms\tlookup_col1_ms\tbucket_ms\tbucket_col1_ms\tscan_ms\tretained_bytes\tscan_hash\tlookup_hash\tlookup_col1_hash\tbucket_hash\tbucket_col1_hash");
    BenchmarkProvider(
        scale,
        "lmdb",
        () =>
        {
            var provider = new LmdbRelationProvider();
            provider.RegisterArtifact(predicate, lmdbOnlyManifest);
            return provider;
        },
        predicate,
        lookupKeysOnly,
        lookupKeysColumn1Only,
        lookupRepetitions,
        DirectorySize(lmdbOnlyDir) + new FileInfo(lmdbOnlyManifest).Length,
        planningRows.Length,
        DistinctCategoryCount(planningRows));
    return;
}
var preserveNumericIds = requestedPreserveNumericIds || scaleLmdbManifest is not null;
var rows = ReadEdges(scaleDir, preserveNumericIds, out var distinctCategories);
var lookupKeys = rows.Select(row => row.Child)
    .Distinct()
    .OrderBy(value => value)
    .Take(Math.Min(lookupKeyCount, rows.Count))
    .Select(value => (object)value.ToString(System.Globalization.CultureInfo.InvariantCulture))
    .ToArray();
var lookupKeysColumn1 = rows.Select(row => row.Parent)
    .Distinct()
    .OrderBy(value => value)
    .Take(Math.Min(lookupKeyCount, rows.Count))
    .Select(value => (object)value.ToString(System.Globalization.CultureInfo.InvariantCulture))
    .ToArray();

var inputPath = Path.Combine(root, "category_parent_ids.tsv");
if (refreshArtifacts || !File.Exists(inputPath))
{
    WriteInput(inputPath, rows);
}
var source = new DelimitedRelationSource(inputPath, '\t', 1, 2);

var binaryDir = Path.Combine(root, "binary-artifact");
var binaryManifestPath = Path.Combine(binaryDir, "category_parent_2.uwbr.json");
var binaryManifest = ExistingOrBuild(
    binaryManifestPath,
    refreshArtifacts,
    () => BinaryRelationArtifactBuilder.BuildFromDelimited(predicate, source, binaryDir));
var delimitedDir = Path.Combine(root, "delimited-artifact");
var delimitedManifestPath = Path.Combine(delimitedDir, "category_parent_2.uwdr.json");
var delimitedManifest = ExistingOrBuild(
    delimitedManifestPath,
    refreshArtifacts,
    () => DelimitedRelationArtifactBuilder.BuildFromDelimited(predicate, source, delimitedDir));
var lmdbManifestPath = Path.Combine(root, "category_parent.lmdb.manifest.json");
var lmdbManifest = scaleLmdbManifest ?? ExistingOrBuild(
    lmdbManifestPath,
    refreshArtifacts,
    () => WriteLmdbArtifact(root, predicate, rows, inputPath));
var lmdbMetadata = LmdbRelationArtifactReader.LoadManifest(lmdbManifest);
var lmdbDir = ResolveManifestEnvironmentPath(lmdbMetadata, lmdbManifest);
var mmapDir = Path.Combine(root, "mmap-array-artifact");
var mmapManifestPath = Path.Combine(mmapDir, "category_parent_2.uwa.json");
var mmapManifest = ExistingOrBuild(
    mmapManifestPath,
    refreshArtifacts,
    () => MmapArrayRelationArtifactBuilder.BuildFromDelimited(predicate, source, mmapDir));

Console.WriteLine("scale\tmode\trows\tdistinct_categories\tlookup_keys\tartifact_bytes\topen_ms\tlookup_ms\tlookup_col1_ms\tbucket_ms\tbucket_col1_ms\tscan_ms\tretained_bytes\tscan_hash\tlookup_hash\tlookup_col1_hash\tbucket_hash\tbucket_col1_hash");
BenchmarkProvider(
    scale,
    "preload",
    () =>
    {
        var provider = new InMemoryRelationProvider();
        provider.AddFacts(predicate, rows.Select(row => new object[] { row.Child, row.Parent }));
        return provider;
    },
    predicate,
    lookupKeys,
    lookupKeysColumn1,
    lookupRepetitions,
    0,
    rows.Count,
    distinctCategories);
BenchmarkProvider(
    scale,
    "binary-artifact",
    () =>
    {
        var provider = new BinaryRelationArtifactProvider();
        provider.RegisterArtifact(predicate, binaryManifest);
        return provider;
    },
    predicate,
    lookupKeys,
    lookupKeysColumn1,
    lookupRepetitions,
    DirectorySize(binaryDir),
    rows.Count,
    distinctCategories);
BenchmarkProvider(
    scale,
    "delimited-artifact",
    () =>
    {
        var provider = new DelimitedRelationArtifactProvider();
        provider.RegisterArtifact(predicate, delimitedManifest);
        return provider;
    },
    predicate,
    lookupKeys,
    lookupKeysColumn1,
    lookupRepetitions,
    DirectorySize(delimitedDir),
    rows.Count,
    distinctCategories);
BenchmarkProvider(
    scale,
    "lmdb",
    () =>
    {
        var provider = new LmdbRelationProvider();
        provider.RegisterArtifact(predicate, lmdbManifest);
        return provider;
    },
    predicate,
    lookupKeys,
    lookupKeysColumn1,
    lookupRepetitions,
    DirectorySize(lmdbDir) + new FileInfo(lmdbManifest).Length,
    rows.Count,
    distinctCategories);
BenchmarkProvider(
    scale,
    "mmap-array",
    () =>
    {
        var provider = new MmapArrayRelationArtifactProvider();
        provider.RegisterArtifact(predicate, mmapManifest);
        return provider;
    },
    predicate,
    lookupKeys,
    lookupKeysColumn1,
    lookupRepetitions,
    DirectorySize(mmapDir),
    rows.Count,
    distinctCategories);
"""


if __name__ == "__main__":
    raise SystemExit(main())
