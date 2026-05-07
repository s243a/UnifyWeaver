#!/usr/bin/env python3
"""
Benchmark the effective-distance workload for the C# query engine, seeded
Prolog, optional direct article/root and bound-root Prolog variants, and
the compiled DFS pipelines.

Default targets:
  - csharp-query  : current C# query runtime using PathAwareTransitiveClosureNode
  - prolog-seeded : generated Prolog using seeded counted-closure reuse
  - prolog-accumulated : generated Prolog using seeded pre-aggregated weight sums
  - csharp-dfs    : generated C# DFS pipeline
  - rust-dfs      : generated Rust DFS pipeline

The script builds temporary binaries locally, runs them against one or more
benchmark scales, and reports median wall-clock times plus output agreement.
"""

from __future__ import annotations

import argparse
import os
import shlex
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from benchmark_common import (
    available_targets,
    build_csharp_package,
    build_go_binary,
    build_haskell_project,
    build_rust_binary,
    add_csharp_query_source_mode_arg,
    append_csharp_query_source_mode_metric,
    csharp_query_env,
    csharp_query_results,
    csharp_query_source_modes_from_args,
    csharp_query_target_label,
    digest_normalized_output,
    find_result,
    find_csharp_query_result,
    group_results_by_scale,
    normalize_three_column_float_rows,
    print_bucket_strategy_metrics,
    print_csharp_query_source_mode_summary,
    print_match_status,
    print_pair_match_status,
    print_phase_metrics,
    print_result_table,
    print_speedup,
    require_file,
    run_command,
)


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
GENERATOR = ROOT / "examples" / "benchmark" / "generate_pipeline.py"
PROLOG_GENERATOR = ROOT / "examples" / "benchmark" / "generate_prolog_effective_distance_benchmark.pl"
WAM_GENERATOR = ROOT / "examples" / "benchmark" / "generate_wam_effective_distance_benchmark.pl"
# Use the variant-aware optimized Haskell WAM generator so seeded and
# accumulated targets compile the same Prolog optimization surfaces as Rust.
WAM_HASKELL_GENERATOR = ROOT / "examples" / "benchmark" / "generate_wam_haskell_optimized_benchmark.pl"
WAM_ELIXIR_GENERATOR = ROOT / "examples" / "benchmark" / "generate_wam_elixir_effective_distance_benchmark.pl"
SEMANTIC_PROLOG_GENERATOR = ROOT / "examples" / "benchmark" / "generate_prolog_min_semantic_distance_benchmark.pl"
EFF_SEMANTIC_PROLOG_GENERATOR = ROOT / "examples" / "benchmark" / "generate_prolog_effective_semantic_distance_benchmark.pl"
EDGE_WEIGHT_SCRIPT = ROOT / "examples" / "benchmark" / "precompute_edge_weights.py"


@dataclass
class RunResult:
    target: str
    scale: str
    times: list[float]
    stdout_sha256: str
    row_count: int
    stderr: str

    @property
    def median(self) -> float:
        return statistics.median(self.times)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales",
        default="300,1k,5k,10k",
        help="Comma-separated benchmark scales from data/benchmark/",
    )
    parser.add_argument(
        "--targets",
        default="csharp-query,csharp-dfs,rust-dfs,go-dfs,prolog-accumulated",
        help="Comma-separated targets: csharp-query,csharp-dfs,rust-dfs,go-dfs,prolog-seeded,prolog-pruned,prolog-accumulated,prolog-article-accumulated,prolog-root-accumulated,wam-rust-seeded,wam-rust-accumulated,wam-rust-seeded-no-kernels,wam-rust-accumulated-no-kernels,haskell-wam-seeded,haskell-wam-accumulated,haskell-wam-seeded-no-kernels,haskell-wam-accumulated-no-kernels,wam-elixir-int-tuple,wam-elixir-lmdb-int-ids",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of timed runs per target/scale",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary build directory for inspection",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=None,
        help=(
            "Persistent build/artifact directory. Use this for larger runs so "
            "generated projects and Elixir LMDB artifacts can be reused."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build target projects/artifacts, including reusable LMDB stores, but do not run timed benchmarks.",
    )
    add_csharp_query_source_mode_arg(parser)
    return parser.parse_args()


def benchmark_temp_parent() -> Path:
    """Pick a writable temp parent, including Termux's $PREFIX/tmp."""
    candidates: list[Path] = []
    for var in ("TMPDIR", "TMP", "TEMP"):
        raw = os.environ.get(var)
        if raw:
            candidates.append(Path(raw))
    prefix = os.environ.get("PREFIX")
    if prefix:
        candidates.append(Path(prefix) / "tmp")
    candidates.append(ROOT / "output")
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".uw_tmp_probe"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise RuntimeError("no writable temporary directory found")


def build_csharp_query(root: Path) -> list[str]:
    return build_csharp_package(
        GENERATOR, BENCH_DIR / "10k" / "facts.pl", "effective_distance", "csharp_query", root / "csharp_query", root="Physics"
    )


def build_csharp_dfs(root: Path) -> list[str]:
    return build_csharp_package(
        GENERATOR, BENCH_DIR / "10k" / "facts.pl", "effective_distance", "csharp", root / "csharp_dfs", root="Physics"
    )


def build_rust_dfs(root: Path) -> list[str]:
    return build_rust_binary(
        GENERATOR, BENCH_DIR / "10k" / "facts.pl", "effective_distance", root / "rust_dfs", "effective_distance_rust", root="Physics"
    )


def build_go_dfs(root: Path) -> list[str]:
    return build_go_binary(
        GENERATOR, BENCH_DIR / "10k" / "facts.pl", "effective_distance", root / "go_dfs", "effective_distance_go", root="Physics"
    )


def build_prolog_effective_distance(root: Path, scale: str, variant: str) -> list[str]:
    facts_path = require_file(BENCH_DIR / scale / "facts.pl")
    script_path = root / f"prolog_{variant}" / scale / f"effective_distance_{variant}.pl"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "swipl",
            "-q",
            "-s",
            str(PROLOG_GENERATOR),
            "--",
            str(facts_path),
            str(script_path),
            variant,
        ]
    )
    return ["swipl", "-q", "-s", str(script_path)]


def build_prolog_semantic_min(root: Path, scale: str) -> list[str]:
    """Build the min semantic distance Prolog benchmark.

    Requires precomputed edge weights in data/benchmark/<scale>/edge_weights.pl.
    Generate them first with:
        python precompute_edge_weights.py data/benchmark/<scale>/category_parent.tsv data/benchmark/<scale>/
    """
    facts_path = require_file(BENCH_DIR / scale / "facts.pl")
    weights_path = BENCH_DIR / scale / "edge_weights.pl"
    if not weights_path.exists():
        # Try to precompute if sentence-transformers is available
        edges_path = BENCH_DIR / scale / "category_parent.tsv"
        if edges_path.exists():
            print(f"  Precomputing edge weights for {scale}...", file=sys.stderr)
            run_command(
                [sys.executable, str(EDGE_WEIGHT_SCRIPT), str(edges_path), str(BENCH_DIR / scale)],
                check=False,
            )
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Edge weights not found: {weights_path}\n"
                f"Run: python precompute_edge_weights.py {BENCH_DIR / scale / 'category_parent.tsv'} {BENCH_DIR / scale}/"
            )
    weights_path = require_file(weights_path)

    script_path = root / "prolog_semantic_min" / scale / "min_semantic_distance.pl"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "swipl", "-q", "-s", str(SEMANTIC_PROLOG_GENERATOR),
            "--", str(facts_path), str(weights_path), str(script_path),
        ]
    )
    return ["swipl", "-q", "-s", str(script_path)]


def build_prolog_effective_semantic(root: Path, scale: str) -> list[str]:
    """Build the effective semantic distance Prolog benchmark (power-mean over weighted paths)."""
    facts_path = require_file(BENCH_DIR / scale / "facts.pl")
    weights_path = BENCH_DIR / scale / "edge_weights.pl"
    if not weights_path.exists():
        edges_path = BENCH_DIR / scale / "category_parent.tsv"
        if edges_path.exists():
            print(f"  Precomputing edge weights for {scale}...", file=sys.stderr)
            run_command(
                [sys.executable, str(EDGE_WEIGHT_SCRIPT), str(edges_path), str(BENCH_DIR / scale)],
                check=False,
            )
        if not weights_path.exists():
            raise FileNotFoundError(f"Edge weights not found: {weights_path}")
    weights_path = require_file(weights_path)

    script_path = root / "prolog_eff_semantic" / scale / "effective_semantic_distance.pl"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "swipl", "-q", "-s", str(EFF_SEMANTIC_PROLOG_GENERATOR),
            "--", str(facts_path), str(weights_path), str(script_path),
        ]
    )
    return ["swipl", "-q", "-s", str(script_path)]


def build_wam_rust_effective_distance(root: Path, scale: str, variant: str) -> list[str]:
    facts_path = require_file(BENCH_DIR / scale / "facts.pl")
    project_dir = root / f"wam_rust_{variant}" / scale
    project_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "swipl",
            "-q",
            "-s",
            str(WAM_GENERATOR),
            "--",
            str(facts_path),
            str(project_dir),
            variant,
        ],
        cwd=ROOT,
    )
    run_command(["cargo", "build", "--release"], cwd=project_dir)
    binary = project_dir / "target" / "release" / "hybrid_ed_bench"
    scale_dir = require_file(BENCH_DIR / scale / "category_parent.tsv").parent
    return [str(binary), str(scale_dir)]


def build_haskell_wam_effective_distance(root: Path, scale: str, variant: str) -> list[str]:
    facts_path = require_file(BENCH_DIR / scale / "facts.pl")
    project_dir = root / f"haskell_wam_{variant}" / scale
    project_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "swipl",
            "-q",
            "-s",
            str(WAM_HASKELL_GENERATOR),
            "--",
            str(facts_path),
            str(project_dir),
            variant,
        ],
        cwd=ROOT,
    )
    command = build_haskell_project(project_dir, "wam-haskell-bench") + [
        str(require_file(BENCH_DIR / scale / "category_parent.tsv").parent)
    ]
    haskell_rts = os.environ.get("HASKELL_RTS", "").strip()
    if haskell_rts:
        command.extend(haskell_rts.split())
    return command


ELIXIR_LMDB_BENCH_MODULES = r'''
defmodule Elmdb do
  @moduledoc false

  def env_open(path, opts) when is_binary(path) do
    :elmdb.env_open(String.to_charlist(path), opts)
  end

  def env_close(env), do: :elmdb.env_close(env)

  def db_open(env, name, opts) do
    :elmdb.db_open(env, to_db_name(name), normalize_db_opts(opts))
  end

  def rw_txn_begin(env), do: :elmdb.txn_begin(env)
  def ro_txn_begin(env), do: :elmdb.ro_txn_begin(env)
  def txn_commit(txn), do: :elmdb.txn_commit(txn)
  def txn_abort(txn), do: :elmdb.txn_abort(txn)
  def ro_txn_commit(txn), do: :elmdb.ro_txn_commit(txn)

  def txn_get(txn, dbi, key) do
    bin_key = to_bin(key)

    try do
      :elmdb.txn_get(txn, dbi, bin_key)
    rescue
      ArgumentError -> :elmdb.ro_txn_get(txn, dbi, bin_key)
    end
  end

  def txn_put(txn, dbi, key, value), do: :elmdb.txn_put(txn, dbi, to_bin(key), to_bin(value))
  def ro_txn_cursor_open(txn, dbi), do: :elmdb.ro_txn_cursor_open(txn, dbi)
  def ro_txn_cursor_close(cur), do: :elmdb.ro_txn_cursor_close(cur)
  def ro_txn_cursor_get(cur, op, arg), do: :elmdb.ro_txn_cursor_get(cur, normalize_cursor_op(op, arg))

  defp to_db_name(name) when is_binary(name), do: name
  defp to_db_name(name) when is_atom(name), do: Atom.to_string(name)
  defp to_bin(value) when is_binary(value), do: value
  defp to_bin(value) when is_list(value), do: :erlang.iolist_to_binary(value)

  defp normalize_db_opts(opts) do
    opts
    |> Enum.map(fn
      :dupsort -> :dup_sort
      other -> other
    end)
    |> add_create()
    |> Enum.uniq()
  end

  defp add_create(opts), do: [:create | opts]
  defp normalize_cursor_op(:set, key), do: {:set, key}
  defp normalize_cursor_op(op, _arg), do: op
end


defmodule LmdbSetup do
  alias WamRuntime.FactSource.LmdbIntIds

  def run(facts_dir) do
    env_path = Path.expand("../lmdb_int_ids", __DIR__)

    if File.exists?(Path.join(env_path, "data.mdb")) do
      IO.puts("lmdb_int_ids_reused=true")
    else
      File.rm_rf!(env_path)
      {handle, env} = open_handle(env_path)

      pairs =
        facts_dir
        |> Path.join("category_parent.tsv")
        |> File.stream!()
        |> Stream.drop(1)
        |> Stream.map(fn line ->
          [child, parent] = line |> String.trim() |> String.split("\t")
          {child, parent}
        end)
        |> Enum.to_list()

      {:ok, result} = LmdbIntIds.ingest_pairs(handle, pairs)
      Elmdb.env_close(env)
      IO.puts("lmdb_int_ids_reused=false")
      IO.puts("lmdb_int_ids_pairs=#{result.pairs_seen}")
      IO.puts("lmdb_int_ids_new_ids=#{result.new_ids}")
      IO.puts("lmdb_int_ids_next_id=#{result.next_id}")
    end
  end

  defp open_handle(env_path) do
    {:ok, env} = Elmdb.env_open(env_path, [{:map_size, 512 * 1024 * 1024}, {:max_dbs, 8}])
    {:ok, facts} = Elmdb.db_open(env, "facts", [:dupsort])
    {:ok, key_to_id} = Elmdb.db_open(env, "key_to_id", [])
    {:ok, id_to_key} = Elmdb.db_open(env, "id_to_key", [])

    handle =
      LmdbIntIds.open(
        %{
          env: env,
          facts_dbi: facts,
          key_to_id_dbi: key_to_id,
          id_to_key_dbi: id_to_key,
          arity: 2,
          dupsort: true
        },
        2,
        nil
      )

    {handle, env}
  end
end


defmodule BenchDriver do
  alias WamRuntime.FactSource.LmdbIntIds

  @max_depth 10
  @dimension_n 5
  @neg_n -@dimension_n

  def run_all(facts_dir) do
    root = facts_dir |> Path.join("root_categories.tsv") |> load_single() |> List.first()
    article_cats = facts_dir |> Path.join("article_category.tsv") |> load_tsv()
    seed_cats = article_cats |> Enum.map(fn {_, c} -> c end) |> Enum.uniq() |> Enum.sort()

    env_path = Path.expand("../lmdb_int_ids", __DIR__)
    {handle, env} = open_handle(env_path)
    root_id = LmdbIntIds.lookup_id(handle, root)

    seed_weight_sums =
      for cat <- seed_cats, into: %{} do
        {cat, run_seeded(handle, cat, root_id)}
      end

    Elmdb.env_close(env)

    article_sums =
      Enum.reduce(article_cats, %{}, fn {art, cat}, acc ->
        ws = Map.get(seed_weight_sums, cat, 0.0)
        val = if cat == root, do: ws + 1.0, else: ws
        Map.update(acc, art, val, &(&1 + val))
      end)

    results =
      for {art, ws} <- article_sums, ws > 0.0 do
        deff = :math.pow(ws, -1 / @dimension_n)
        {deff, art}
      end
      |> Enum.sort()

    IO.puts("article\troot_category\teffective_distance")

    for {deff, art} <- results do
      :io.format("~ts\t~ts\t~.6f~n", [art, root, deff])
    end
  end

  defp run_seeded(handle, cat, root_id) do
    case LmdbIntIds.lookup_id(handle, cat) do
      nil ->
        0.0

      cat_id ->
        if cat_id == root_id do
          0.0
        else
          dests_fn = fn node ->
            handle
            |> LmdbIntIds.lookup_by_arg1_id(node, nil)
            |> Enum.map(fn {_from, to} -> to end)
          end

          WamRuntime.GraphKernel.CategoryAncestor.fold_hops_with_dests_seeded(
            dests_fn,
            cat_id,
            root_id,
            @max_depth,
            0.0,
            fn hop, acc -> acc + :math.pow(hop + 1, @neg_n) end
          )
        end
    end
  end

  defp open_handle(env_path) do
    {:ok, env} = Elmdb.env_open(env_path, [{:map_size, 512 * 1024 * 1024}, {:max_dbs, 8}])
    {:ok, facts} = Elmdb.db_open(env, "facts", [:dupsort])
    {:ok, key_to_id} = Elmdb.db_open(env, "key_to_id", [])
    {:ok, id_to_key} = Elmdb.db_open(env, "id_to_key", [])

    handle =
      LmdbIntIds.open(
        %{
          env: env,
          facts_dbi: facts,
          key_to_id_dbi: key_to_id,
          id_to_key_dbi: id_to_key,
          arity: 2,
          dupsort: true
        },
        2,
        nil
      )

    {handle, env}
  end

  defp load_tsv(path) do
    path
    |> File.stream!()
    |> Stream.drop(1)
    |> Stream.map(fn line ->
      [a, b] = line |> String.trim() |> String.split("\t")
      {a, b}
    end)
    |> Enum.to_list()
  end

  defp load_single(path) do
    path
    |> File.stream!()
    |> Stream.drop(1)
    |> Stream.map(&String.trim/1)
    |> Enum.to_list()
  end
end
'''


ELIXIR_INT_TUPLE_BENCH_MODULES = r'''
defmodule BenchDriver do
  @max_depth 10
  @dimension_n 5
  @neg_n -@dimension_n

  def run_all(facts_dir) do
    started = monotonic_ms()
    root = facts_dir |> Path.join("root_categories.tsv") |> load_single() |> List.first()
    article_cats = facts_dir |> Path.join("article_category.tsv") |> load_tsv()
    category_edges = facts_dir |> Path.join("category_parent.tsv") |> load_tsv()
    seed_cats = article_cats |> Enum.map(fn {_, c} -> c end) |> Enum.uniq() |> Enum.sort()
    {id_map, adjacency} = build_int_tuple(category_edges, root, seed_cats)
    root_id = Map.fetch!(id_map, root)
    setup_done = monotonic_ms()

    seed_weight_sums =
      for cat <- seed_cats, into: %{} do
        {cat, run_seeded(id_map, adjacency, cat, root_id)}
      end

    query_done = monotonic_ms()

    article_sums =
      Enum.reduce(article_cats, %{}, fn {art, cat}, acc ->
        ws = Map.get(seed_weight_sums, cat, 0.0)
        val = if cat == root, do: ws + 1.0, else: ws
        Map.update(acc, art, val, &(&1 + val))
      end)

    results =
      for {art, ws} <- article_sums, ws > 0.0 do
        deff = :math.pow(ws, -1 / @dimension_n)
        {deff, art}
      end
      |> Enum.sort()

    finished = monotonic_ms()
    IO.puts(:stderr, "mode=wam_elixir_int_tuple setup_ms=#{setup_done - started} query_ms=#{query_done - setup_done} aggregation_ms=#{finished - query_done} total_ms=#{finished - started} seed_count=#{length(seed_cats)} tuple_count=#{map_size(seed_weight_sums)} article_count=#{length(results)} node_count=#{map_size(id_map)}")
    IO.puts("article\troot_category\teffective_distance")

    for {deff, art} <- results do
      :io.format("~ts\t~ts\t~.6f~n", [art, root, deff])
    end
  end

  defp run_seeded(id_map, adjacency, cat, root_id) do
    case Map.fetch(id_map, cat) do
      :error ->
        0.0

      {:ok, cat_id} ->
        if cat_id == root_id do
          0.0
        else
          dests_fn = fn node -> elem(adjacency, node) end

          WamRuntime.GraphKernel.CategoryAncestor.fold_hops_with_dests_seeded(
            dests_fn,
            cat_id,
            root_id,
            @max_depth,
            0.0,
            fn hop, acc -> acc + :math.pow(hop + 1, @neg_n) end
          )
        end
    end
  end

  defp build_int_tuple(edges, root, seed_cats) do
    {id_map, next_id} =
      Enum.reduce(edges, {%{}, 0}, fn {child, parent}, acc ->
        acc |> intern(child) |> intern(parent)
      end)

    {id_map, next_id} =
      Enum.reduce([root | seed_cats], {id_map, next_id}, fn key, acc -> intern(acc, key) end)

    adjacency =
      edges
      |> Enum.reduce(List.duplicate([], next_id), fn {child, parent}, acc ->
        child_id = Map.fetch!(id_map, child)
        parent_id = Map.fetch!(id_map, parent)
        List.update_at(acc, child_id, fn parents -> [parent_id | parents] end)
      end)
      |> Enum.map(&Enum.reverse/1)
      |> List.to_tuple()

    {id_map, adjacency}
  end

  defp intern({map, next_id}, key) do
    case Map.fetch(map, key) do
      {:ok, _} -> {map, next_id}
      :error -> {Map.put(map, key, next_id), next_id + 1}
    end
  end

  defp load_tsv(path) do
    path
    |> File.stream!()
    |> Stream.drop(1)
    |> Stream.map(fn line ->
      [a, b] = line |> String.trim() |> String.split("\t")
      {a, b}
    end)
    |> Enum.to_list()
  end

  defp load_single(path) do
    path
    |> File.stream!()
    |> Stream.drop(1)
    |> Stream.map(&String.trim/1)
    |> Enum.to_list()
  end

  defp monotonic_ms do
    System.monotonic_time(:millisecond)
  end
end
'''


def patch_wam_elixir_mix(project_dir: Path, deps: str = "[]") -> None:
    mix_path = require_file(project_dir / "mix.exs")
    mix_path.write_text(
        f'''defmodule WamElixirBench.MixProject do
  use Mix.Project

  def project do
    [
      app: :wam_elixir_bench,
      version: "0.1.0",
      elixir: "~> 1.14",
      deps: {deps}
    ]
  end
end
''',
        encoding="utf-8",
    )


def write_wam_elixir_runner(project_dir: Path, scale_dir: Path, name: str) -> Path:
    runner = project_dir / name
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(project_dir))}\n"
        "exec mix run --no-compile -e "
        + shlex.quote(f'BenchDriver.run_all("{scale_dir}")')
        + "\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    return runner


def generate_wam_elixir_project(project_dir: Path, scale: str, mode: str = "full") -> None:
    facts_path = require_file(BENCH_DIR / scale / "facts.pl")
    project_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "swipl",
        "-q",
        "-s",
        str(WAM_ELIXIR_GENERATOR),
        "--",
        str(facts_path),
        str(project_dir),
    ]
    if mode != "full":
        command.append(mode)
    run_command(
        command,
        cwd=ROOT,
    )


def build_wam_elixir_int_tuple_effective_distance(root: Path, scale: str) -> list[str]:
    scale_dir = require_file(BENCH_DIR / scale / "category_parent.tsv").parent
    project_dir = root / "wam_elixir_int_tuple" / scale
    generate_wam_elixir_project(project_dir, scale, "runtime_only")
    patch_wam_elixir_mix(project_dir)
    bench_modules = project_dir / "lib" / "int_tuple_bench_modules.ex"
    bench_modules.write_text(ELIXIR_INT_TUPLE_BENCH_MODULES, encoding="utf-8")
    run_command(["mix", "compile"], cwd=project_dir)
    return [str(write_wam_elixir_runner(project_dir, scale_dir, "run_int_tuple"))]


def build_wam_elixir_lmdb_int_ids_effective_distance(root: Path, scale: str) -> list[str]:
    scale_dir = require_file(BENCH_DIR / scale / "category_parent.tsv").parent
    project_dir = root / "wam_elixir_lmdb_int_ids" / scale
    generate_wam_elixir_project(project_dir, scale, "runtime_only")
    patch_wam_elixir_mix(project_dir, '[{:elmdb, "~> 0.4.1"}]')
    bench_modules = project_dir / "lib" / "lmdb_bench_modules.ex"
    bench_modules.write_text(ELIXIR_LMDB_BENCH_MODULES, encoding="utf-8")
    run_command(["mix", "deps.get"], cwd=project_dir)
    run_command(["mix", "compile"], cwd=project_dir)
    run_command(["mix", "run", "--no-compile", "-e", f'LmdbSetup.run("{scale_dir}")'], cwd=project_dir)
    return [str(write_wam_elixir_runner(project_dir, scale_dir, "run_lmdb_int_ids"))]


def benchmark_target(
    command: list[str],
    scale: str,
    repetitions: int,
    target: str,
    csharp_query_source_mode: str = "auto",
    artifact_dir: Path | None = None,
    result_target: str | None = None,
) -> RunResult:
    times: list[float] = []
    last_stdout = ""
    last_stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        if target.startswith("prolog-") or target.startswith("wam-") or target.startswith("haskell-wam-"):
            result = run_command(command)
        else:
            scale_dir = require_file(BENCH_DIR / scale / "category_parent.tsv").parent
            edge_path = scale_dir / "category_parent.tsv"
            article_path = scale_dir / "article_category.tsv"
            env = csharp_query_env(csharp_query_source_mode, artifact_dir) if target == "csharp-query" else None
            result = run_command(command + [str(edge_path), str(article_path)], env=env)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        last_stdout = result.stdout
        last_stderr = result.stderr
        if target == "csharp-query":
            last_stderr = append_csharp_query_source_mode_metric(last_stderr, csharp_query_source_mode)

    normalized = normalize_three_column_float_rows(last_stdout, decimals=6)
    digest, rows = digest_normalized_output(normalized)
    return RunResult(
        target=result_target or target,
        scale=scale,
        times=times,
        stdout_sha256=digest,
        row_count=rows,
        stderr=last_stderr,
    )


def print_summary(results: list[RunResult]) -> None:
    seed_subset_probe = wam_seed_subset_probe_enabled()
    print("scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256")
    for scale, entries in group_results_by_scale(results, sort_key=scale_sort_key):
        print_result_table(entries, scale)

        qe = find_csharp_query_result(entries)
        csharp_dfs = find_result(entries, "csharp-dfs")
        rust_dfs = find_result(entries, "rust-dfs")
        prolog_seeded = find_result(entries, "prolog-seeded")
        prolog_pruned = find_result(entries, "prolog-pruned")
        prolog_accumulated = find_result(entries, "prolog-accumulated")
        prolog_article_accumulated = find_result(entries, "prolog-article-accumulated")
        prolog_root_accumulated = find_result(entries, "prolog-root-accumulated")
        wam_rust_seeded = find_result(entries, "wam-rust-seeded")
        wam_rust_accumulated = find_result(entries, "wam-rust-accumulated")
        wam_rust_seeded_no_kernels = find_result(entries, "wam-rust-seeded-no-kernels")
        wam_rust_accumulated_no_kernels = find_result(entries, "wam-rust-accumulated-no-kernels")
        haskell_wam_seeded = find_result(entries, "haskell-wam-seeded")
        haskell_wam_accumulated = find_result(entries, "haskell-wam-accumulated")
        haskell_wam_seeded_no_kernels = find_result(entries, "haskell-wam-seeded-no-kernels")
        haskell_wam_accumulated_no_kernels = find_result(entries, "haskell-wam-accumulated-no-kernels")
        wam_elixir_int_tuple = find_result(entries, "wam-elixir-int-tuple")
        wam_elixir_lmdb_int_ids = find_result(entries, "wam-elixir-lmdb-int-ids")
        prolog_semantic_min = find_result(entries, "prolog-semantic-min")
        prolog_eff_semantic = find_result(entries, "prolog-eff-semantic")
        dfs_like = [item for item in entries if item.target in {"csharp-dfs", "rust-dfs", "go-dfs"}]

        if len(dfs_like) > 1:
            print_match_status(scale, "dfs_outputs", dfs_like)
        print_pair_match_status(scale, "query_vs_csharp_dfs", qe, csharp_dfs)
        print_pair_match_status(scale, "query_vs_prolog_seeded", qe, prolog_seeded)
        print_pair_match_status(scale, "query_vs_prolog_pruned", qe, prolog_pruned)
        print_pair_match_status(scale, "query_vs_prolog_accumulated", qe, prolog_accumulated)
        print_pair_match_status(scale, "query_vs_prolog_article_accumulated", qe, prolog_article_accumulated)
        print_pair_match_status(scale, "query_vs_prolog_root_accumulated", qe, prolog_root_accumulated)
        print_pair_match_status(scale, "query_vs_wam_rust_seeded", qe, wam_rust_seeded)
        print_pair_match_status(scale, "query_vs_wam_rust_accumulated", qe, wam_rust_accumulated)
        print_no_kernel_match_status(scale, "query_vs_wam_rust_seeded_no_kernels", qe, wam_rust_seeded_no_kernels, seed_subset_probe)
        print_no_kernel_match_status(scale, "query_vs_wam_rust_accumulated_no_kernels", qe, wam_rust_accumulated_no_kernels, seed_subset_probe)
        print_pair_match_status(scale, "query_vs_haskell_wam_seeded", qe, haskell_wam_seeded)
        print_pair_match_status(scale, "query_vs_haskell_wam_accumulated", qe, haskell_wam_accumulated)
        print_pair_match_status(scale, "query_vs_wam_elixir_int_tuple", qe, wam_elixir_int_tuple)
        print_pair_match_status(scale, "query_vs_wam_elixir_lmdb_int_ids", qe, wam_elixir_lmdb_int_ids)
        print_no_kernel_match_status(scale, "query_vs_haskell_wam_seeded_no_kernels", qe, haskell_wam_seeded_no_kernels, seed_subset_probe)
        print_no_kernel_match_status(scale, "query_vs_haskell_wam_accumulated_no_kernels", qe, haskell_wam_accumulated_no_kernels, seed_subset_probe)
        print_pair_match_status(scale, "prolog_vs_wam_rust_seeded", prolog_accumulated, wam_rust_seeded)
        print_pair_match_status(scale, "prolog_vs_wam_rust_accumulated", prolog_accumulated, wam_rust_accumulated)
        print_no_kernel_match_status(scale, "prolog_vs_wam_rust_seeded_no_kernels", prolog_accumulated, wam_rust_seeded_no_kernels, seed_subset_probe)
        print_no_kernel_match_status(scale, "prolog_vs_wam_rust_accumulated_no_kernels", prolog_accumulated, wam_rust_accumulated_no_kernels, seed_subset_probe)
        print_pair_match_status(scale, "prolog_vs_haskell_wam_seeded", prolog_accumulated, haskell_wam_seeded)
        print_pair_match_status(scale, "prolog_vs_haskell_wam_accumulated", prolog_accumulated, haskell_wam_accumulated)
        print_pair_match_status(scale, "prolog_vs_wam_elixir_int_tuple", prolog_accumulated, wam_elixir_int_tuple)
        print_pair_match_status(scale, "prolog_vs_wam_elixir_lmdb_int_ids", prolog_accumulated, wam_elixir_lmdb_int_ids)
        print_no_kernel_match_status(scale, "prolog_vs_haskell_wam_seeded_no_kernels", prolog_accumulated, haskell_wam_seeded_no_kernels, seed_subset_probe)
        print_no_kernel_match_status(scale, "prolog_vs_haskell_wam_accumulated_no_kernels", prolog_accumulated, haskell_wam_accumulated_no_kernels, seed_subset_probe)
        print_pair_match_status(scale, "wam_rust_vs_haskell_wam_seeded", wam_rust_seeded, haskell_wam_seeded)
        print_pair_match_status(scale, "wam_rust_vs_haskell_wam_accumulated", wam_rust_accumulated, haskell_wam_accumulated)
        print_pair_match_status(scale, "wam_rust_vs_wam_elixir_int_tuple", wam_rust_accumulated, wam_elixir_int_tuple)
        print_pair_match_status(scale, "wam_rust_vs_wam_elixir_lmdb_int_ids", wam_rust_accumulated, wam_elixir_lmdb_int_ids)
        print_pair_match_status(scale, "haskell_wam_vs_wam_elixir_int_tuple", haskell_wam_accumulated, wam_elixir_int_tuple)
        print_pair_match_status(scale, "haskell_wam_vs_wam_elixir_lmdb_int_ids", haskell_wam_accumulated, wam_elixir_lmdb_int_ids)
        print_pair_match_status(scale, "wam_elixir_int_tuple_vs_lmdb_int_ids", wam_elixir_int_tuple, wam_elixir_lmdb_int_ids)
        print_no_kernel_match_status(scale, "wam_rust_vs_haskell_wam_seeded_no_kernels", wam_rust_seeded_no_kernels, haskell_wam_seeded_no_kernels, seed_subset_probe)
        print_no_kernel_match_status(scale, "wam_rust_vs_haskell_wam_accumulated_no_kernels", wam_rust_accumulated_no_kernels, haskell_wam_accumulated_no_kernels, seed_subset_probe)
        print_pair_match_status(scale, "query_vs_prolog_semantic_min", qe, prolog_semantic_min)
        print_pair_match_status(scale, "query_vs_prolog_eff_semantic", qe, prolog_eff_semantic)
        print_speedup(scale, "speedup_vs_csharp_dfs", csharp_dfs, qe)
        print_speedup(scale, "speedup_vs_rust_dfs", rust_dfs, qe)
        print_speedup(scale, "speedup_vs_prolog_seeded", prolog_seeded, qe)
        print_speedup(scale, "speedup_vs_prolog_pruned", prolog_pruned, qe)
        print_speedup(scale, "speedup_vs_prolog_accumulated", prolog_accumulated, qe)
        print_speedup(scale, "speedup_vs_prolog_article_accumulated", prolog_article_accumulated, qe)
        print_speedup(scale, "speedup_vs_prolog_root_accumulated", prolog_root_accumulated, qe)
        print_speedup(scale, "speedup_vs_wam_rust_seeded", wam_rust_seeded, qe)
        print_speedup(scale, "speedup_vs_wam_rust_accumulated", wam_rust_accumulated, qe)
        print_no_kernel_speedup(scale, "speedup_vs_wam_rust_seeded_no_kernels", wam_rust_seeded_no_kernels, qe, seed_subset_probe)
        print_no_kernel_speedup(scale, "speedup_vs_wam_rust_accumulated_no_kernels", wam_rust_accumulated_no_kernels, qe, seed_subset_probe)
        print_speedup(scale, "speedup_vs_haskell_wam_seeded", haskell_wam_seeded, qe)
        print_speedup(scale, "speedup_vs_haskell_wam_accumulated", haskell_wam_accumulated, qe)
        print_speedup(scale, "speedup_vs_wam_elixir_int_tuple", wam_elixir_int_tuple, qe)
        print_speedup(scale, "speedup_vs_wam_elixir_lmdb_int_ids", wam_elixir_lmdb_int_ids, qe)
        print_no_kernel_speedup(scale, "speedup_vs_haskell_wam_seeded_no_kernels", haskell_wam_seeded_no_kernels, qe, seed_subset_probe)
        print_no_kernel_speedup(scale, "speedup_vs_haskell_wam_accumulated_no_kernels", haskell_wam_accumulated_no_kernels, qe, seed_subset_probe)
        print_speedup(scale, "wam_rust_speedup_vs_haskell_seeded", haskell_wam_seeded, wam_rust_seeded)
        print_speedup(scale, "wam_rust_speedup_vs_haskell_accumulated", haskell_wam_accumulated, wam_rust_accumulated)
        print_speedup(scale, "wam_rust_speedup_vs_wam_elixir_int_tuple", wam_elixir_int_tuple, wam_rust_accumulated)
        print_speedup(scale, "wam_rust_speedup_vs_wam_elixir_lmdb_int_ids", wam_elixir_lmdb_int_ids, wam_rust_accumulated)
        print_speedup(scale, "haskell_wam_speedup_vs_wam_elixir_int_tuple", wam_elixir_int_tuple, haskell_wam_accumulated)
        print_speedup(scale, "haskell_wam_speedup_vs_wam_elixir_lmdb_int_ids", wam_elixir_lmdb_int_ids, haskell_wam_accumulated)
        print_speedup(scale, "wam_elixir_int_tuple_speedup_vs_lmdb_int_ids", wam_elixir_lmdb_int_ids, wam_elixir_int_tuple)
        print_no_kernel_speedup(scale, "wam_rust_speedup_vs_haskell_seeded_no_kernels", haskell_wam_seeded_no_kernels, wam_rust_seeded_no_kernels, seed_subset_probe)
        print_no_kernel_speedup(scale, "wam_rust_speedup_vs_haskell_accumulated_no_kernels", haskell_wam_accumulated_no_kernels, wam_rust_accumulated_no_kernels, seed_subset_probe)
        print_speedup(scale, "speedup_vs_prolog_semantic_min", prolog_semantic_min, qe)
        print_speedup(scale, "speedup_vs_prolog_eff_semantic", prolog_eff_semantic, qe)
        for csharp_entry in csharp_query_results(entries):
            metric_label = f"{csharp_entry.target}-metrics" if csharp_entry.target != "csharp-query" else "csharp-query-metrics"
            bucket_label = (
                f"{csharp_entry.target}-bucket-strategies"
                if csharp_entry.target != "csharp-query"
                else "csharp-query-bucket-strategies"
            )
            print_phase_metrics(scale, metric_label, csharp_entry)
            print_bucket_strategy_metrics(scale, bucket_label, csharp_entry)
        print_csharp_query_source_mode_summary(scale, entries)
        print_phase_metrics(scale, "prolog-seeded-metrics", prolog_seeded)
        print_phase_metrics(scale, "prolog-pruned-metrics", prolog_pruned)
        print_phase_metrics(scale, "prolog-accumulated-metrics", prolog_accumulated)
        print_phase_metrics(scale, "prolog-article-accumulated-metrics", prolog_article_accumulated)
        print_phase_metrics(scale, "prolog-root-accumulated-metrics", prolog_root_accumulated)
        print_phase_metrics(scale, "wam-rust-seeded-metrics", wam_rust_seeded)
        print_phase_metrics(scale, "wam-rust-accumulated-metrics", wam_rust_accumulated)
        print_phase_metrics(scale, "wam-rust-seeded-no-kernels-metrics", wam_rust_seeded_no_kernels)
        print_phase_metrics(scale, "wam-rust-accumulated-no-kernels-metrics", wam_rust_accumulated_no_kernels)
        print_phase_metrics(scale, "haskell-wam-seeded-metrics", haskell_wam_seeded)
        print_phase_metrics(scale, "haskell-wam-accumulated-metrics", haskell_wam_accumulated)
        print_phase_metrics(scale, "wam-elixir-int-tuple-metrics", wam_elixir_int_tuple)
        print_phase_metrics(scale, "wam-elixir-lmdb-int-ids-metrics", wam_elixir_lmdb_int_ids)
        print_phase_metrics(scale, "haskell-wam-seeded-no-kernels-metrics", haskell_wam_seeded_no_kernels)
        print_phase_metrics(scale, "haskell-wam-accumulated-no-kernels-metrics", haskell_wam_accumulated_no_kernels)
        print_phase_metrics(scale, "prolog-semantic-min-metrics", prolog_semantic_min)
        print_phase_metrics(scale, "prolog-eff-semantic-metrics", prolog_eff_semantic)


def wam_seed_subset_probe_enabled() -> bool:
    return bool(os.environ.get("WAM_SEED_LIMIT") or os.environ.get("WAM_SEED_FILTER"))


def print_no_kernel_match_status(
    scale: str,
    label: str,
    left: RunResult | None,
    right: RunResult | None,
    seed_subset_probe: bool,
) -> None:
    if not (left and right):
        return
    if seed_subset_probe:
        print(f"{scale}\t{label}\tSKIPPED_SEED_SUBSET")
        return
    print_pair_match_status(scale, label, left, right)


def print_no_kernel_speedup(
    scale: str,
    label: str,
    faster_baseline: RunResult | None,
    measured: RunResult | None,
    seed_subset_probe: bool,
) -> None:
    if seed_subset_probe:
        return
    print_speedup(scale, label, faster_baseline, measured)


def scale_sort_key(scale: str) -> tuple[int, str]:
    digits = "".join(ch for ch in scale if ch.isdigit())
    suffix = "".join(ch for ch in scale if not ch.isdigit())
    if scale == "dev":
        return (0, scale)
    if not digits:
        return (10**9, scale)
    value = int(digits)
    multiplier = 1000 if suffix.lower().startswith("k") else 1
    return (value * multiplier, scale)


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    csharp_query_source_modes = csharp_query_source_modes_from_args(args)
    requested_targets = [part.strip() for part in args.targets.split(",") if part.strip()]
    targets = available_targets(requested_targets)
    if not targets:
        print("no benchmark targets available", file=sys.stderr)
        return 1

    temp_ctx = None
    if args.build_root is not None:
        temp_root = args.build_root
        temp_root.mkdir(parents=True, exist_ok=True)
    elif args.keep_temp:
        temp_parent = benchmark_temp_parent()
        temp_root = Path(tempfile.mkdtemp(prefix="uw-effective-distance-", dir=temp_parent))
    else:
        temp_parent = benchmark_temp_parent()
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-effective-distance-", dir=temp_parent)
        temp_root = Path(temp_ctx.name)
    try:
        commands: dict[str, list[str]] = {}
        for target in targets:
            if target == "csharp-query":
                commands[target] = build_csharp_query(temp_root)
            elif target == "csharp-dfs":
                commands[target] = build_csharp_dfs(temp_root)
            elif target == "rust-dfs":
                commands[target] = build_rust_dfs(temp_root)
            elif target == "go-dfs":
                commands[target] = build_go_dfs(temp_root)
            elif target == "prolog-seeded":
                continue
            elif target == "prolog-pruned":
                continue
            elif target == "prolog-accumulated":
                continue
            elif target == "prolog-article-accumulated":
                continue
            elif target == "prolog-root-accumulated":
                continue
            # Hybrid WAM variants are generated per scale because facts and
            # optional optimized helpers are loaded into the generated project.
            elif target == "wam-rust-seeded":
                continue
            elif target == "wam-rust-accumulated":
                continue
            elif target == "wam-rust-seeded-no-kernels":
                continue
            elif target == "wam-rust-accumulated-no-kernels":
                continue
            elif target == "haskell-wam-seeded":
                continue
            elif target == "haskell-wam-accumulated":
                continue
            elif target == "haskell-wam-seeded-no-kernels":
                continue
            elif target == "haskell-wam-accumulated-no-kernels":
                continue
            elif target == "wam-elixir-int-tuple":
                continue
            elif target == "wam-elixir-lmdb-int-ids":
                continue
            elif target == "prolog-semantic-min":
                continue
            elif target == "prolog-eff-semantic":
                continue
            else:
                raise ValueError(f"unsupported target: {target}")

        results: list[RunResult] = []
        for scale in scales:
            for target in targets:
                if target == "prolog-seeded":
                    command = build_prolog_effective_distance(temp_root, scale, "seeded")
                elif target == "prolog-pruned":
                    command = build_prolog_effective_distance(temp_root, scale, "pruned")
                elif target == "prolog-accumulated":
                    command = build_prolog_effective_distance(temp_root, scale, "accumulated")
                elif target == "prolog-article-accumulated":
                    command = build_prolog_effective_distance(temp_root, scale, "article_accumulated")
                elif target == "prolog-root-accumulated":
                    command = build_prolog_effective_distance(temp_root, scale, "root_accumulated")
                elif target == "wam-rust-seeded":
                    command = build_wam_rust_effective_distance(temp_root, scale, "seeded")
                elif target == "wam-rust-accumulated":
                    command = build_wam_rust_effective_distance(temp_root, scale, "accumulated")
                elif target == "wam-rust-seeded-no-kernels":
                    command = build_wam_rust_effective_distance(temp_root, scale, "seeded_no_kernels")
                elif target == "wam-rust-accumulated-no-kernels":
                    command = build_wam_rust_effective_distance(temp_root, scale, "accumulated_no_kernels")
                elif target == "haskell-wam-seeded":
                    command = build_haskell_wam_effective_distance(temp_root, scale, "seeded")
                elif target == "haskell-wam-accumulated":
                    command = build_haskell_wam_effective_distance(temp_root, scale, "accumulated")
                elif target == "haskell-wam-seeded-no-kernels":
                    command = build_haskell_wam_effective_distance(temp_root, scale, "seeded_no_kernels")
                elif target == "haskell-wam-accumulated-no-kernels":
                    command = build_haskell_wam_effective_distance(temp_root, scale, "accumulated_no_kernels")
                elif target == "wam-elixir-int-tuple":
                    command = build_wam_elixir_int_tuple_effective_distance(temp_root, scale)
                elif target == "wam-elixir-lmdb-int-ids":
                    command = build_wam_elixir_lmdb_int_ids_effective_distance(temp_root, scale)
                elif target == "prolog-semantic-min":
                    command = build_prolog_semantic_min(temp_root, scale)
                elif target == "prolog-eff-semantic":
                    command = build_prolog_effective_semantic(temp_root, scale)
                else:
                    command = commands[target]
                if args.prepare_only:
                    print(f"prepared {scale}/{target}: {' '.join(command)}", file=sys.stderr)
                    continue
                if target == "csharp-query":
                    for source_mode in csharp_query_source_modes:
                        artifact_dir = temp_root / "artifacts" / target / source_mode / scale
                        results.append(
                            benchmark_target(
                                command,
                                scale,
                                args.repetitions,
                                target,
                                source_mode,
                                artifact_dir,
                                csharp_query_target_label(source_mode, csharp_query_source_modes),
                            )
                        )
                else:
                    results.append(
                        benchmark_target(
                            command,
                            scale,
                            args.repetitions,
                            target,
                        )
                    )

        if args.prepare_only:
            print(f"prepared build directory: {temp_root}", file=sys.stderr)
        else:
            print_summary(results)
        if args.keep_temp or args.build_root is not None:
            print(f"kept temp build directory: {temp_root}", file=sys.stderr)
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())
