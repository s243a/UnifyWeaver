:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/wam_elixir_target').
:- use_module('../../src/unifyweaver/targets/wam_elixir_utils', [reg_id/2, camel_case/2]).
:- use_module(library(option)).
:- use_module(library(lists)).

%% generate_wam_elixir_effective_distance_benchmark.pl
%%
%% Generates a lowered Elixir WAM benchmark for the effective-distance workload.
%%
%% Usage:
%%   swipl -q -s generate_wam_elixir_effective_distance_benchmark.pl -- \
%%       <facts.pl> <output-dir>

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [FactsPath, OutputDir]
    ->  true
    ;   format(user_error, 'Usage: ... -- <facts.pl> <output-dir>~n', []),
        halt(1)
    ),
    generate_wam_elixir_benchmark(FactsPath, OutputDir),
    halt(0).

generate_wam_elixir_benchmark(FactsPath, OutputDir) :-
    benchmark_workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),
    load_files(FactsPath, [silent(true)]),
    
    % Use setup_call_cleanup to restore the original mode state
    setup_call_cleanup(
        ( retractall(user:mode(category_ancestor(_, _, _, _))),
          assertz(user:mode(category_ancestor(-, +, -, +))) ),
        generate_wam_elixir_benchmark_scoped(OutputDir),
        ( retractall(user:mode(category_ancestor(-, +, -, +))) )
    ).

generate_wam_elixir_benchmark_scoped(OutputDir) :-
    % Define the predicates to compile
    Predicates = [
        dimension_n/1,
        max_depth/1,
        category_ancestor/4,
        category_parent/2,
        article_category/2,
        root_category/1
    ],
    
    Options = [
        module_name('wam_elixir_bench'),
        emit_mode(lowered)
    ],

    % Convert predicate indicators to Pred/Arity-WamCode pairs
    findall(P/A-WamCode, (
        member(P/A, Predicates),
        (   wam_target:compile_predicate_to_wam(P/A, [], WamCode) -> true
        ;   wam_target:compile_predicate_to_wam(user:P/A, [], WamCode) -> true
        ;   format(user_error, '[WARN] Could not compile ~w/~w, skipping~n', [P, A]), fail
        )
    ), PredWamPairs),

    write_wam_elixir_project(PredWamPairs, Options, OutputDir),
    
    % Write the benchmark driver in Elixir
    directory_file_path(OutputDir, 'test_bench.exs', BenchPath),
    write_bench_driver(BenchPath, Options),
    
    format(user_error, '[WAM-Elixir] Benchmark project generated at: ~w~n', [OutputDir]).

write_bench_driver(Path, Options) :-
    option(module_name(ModName), Options, wam_elixir_bench),
    camel_case(ModName, CamelMod),
    open(Path, write, S),
    format(S, 'Code.require_file("lib/wam_runtime.ex", __DIR__)~n', []),
    format(S, 'Code.require_file("lib/wam_dispatcher.ex", __DIR__)~n', []),
    format(S, 'Code.require_file("lib/dimension_n.ex", __DIR__)~n', []),
    format(S, 'Code.require_file("lib/max_depth.ex", __DIR__)~n', []),
    format(S, 'Code.require_file("lib/category_ancestor.ex", __DIR__)~n', []),
    format(S, 'Code.require_file("lib/category_parent.ex", __DIR__)~n', []),
    format(S, 'Code.require_file("lib/article_category.ex", __DIR__)~n', []),
    format(S, 'Code.require_file("lib/root_category.ex", __DIR__)~n', []),
    format(S, '~n', []),
    format(S, 'defmodule BenchDriver do~n', []),
    format(S, '  def run_all(facts_dir) do~n', []),
    format(S, '    article_cats = load_tsv(Path.join(facts_dir, "article_category.tsv"))~n', []),
    format(S, '    root_cats = load_single(Path.join(facts_dir, "root_categories.tsv"))~n', []),
    format(S, '    root = List.first(root_cats)~n', []),
    format(S, '~n', []),
    format(S, '    seed_cats = article_cats |> Enum.map(fn {_, c} -> c end) |> Enum.uniq() |> Enum.sort()~n', []),
    format(S, '~n', []),
    format(S, '    seed_weight_sums = for cat <- seed_cats, into: %{} do~n', []),
    format(S, '      {cat, run_seeded(cat, root)}~n', []),
    format(S, '    end~n', []),
    format(S, '~n', []),
    format(S, '    article_sums = Enum.reduce(article_cats, %{}, fn {art, cat}, acc ->~n', []),
    format(S, '      ws = Map.get(seed_weight_sums, cat, 0.0)~n', []),
    format(S, '      val = if cat == root, do: ws + 1.0, else: ws~n', []),
    format(S, '      Map.update(acc, art, val, &(&1 + val))~n', []),
    format(S, '    end)~n', []),
    format(S, '~n', []),
    format(S, '    results = for {art, ws} <- article_sums, ws > 0.0 do~n', []),
    format(S, '      deff = :math.pow(ws, -1/5)~n', []),
    format(S, '      {deff, art}~n', []),
    format(S, '    end |> Enum.sort()~n', []),
    format(S, '~n', []),
    format(S, '    IO.puts("article\\troot_category\\teffective_distance")~n', []),
    format(S, '    for {deff, art} <- results do~n', []),
    format(S, '      :io.format("~~s\\t~~s\\t~~.6f~~n", [art, root, deff])~n', []),
    format(S, '    end~n', []),
    format(S, '  end~n', []),
    format(S, '~n', []),
    format(S, '  def run_seeded(cat, root) do~n', []),
    format(S, '    args = [cat, root, {:unbound, 3}, [cat]]~n', []),
    format(S, '    case ~w.CategoryAncestor.run(args) do~n', [CamelMod]),
    format(S, '      {:ok, state} -> execute_backtrack(state, 0.0)~n', []),
    format(S, '      :fail -> 0.0~n', []),
    format(S, '    end~n', []),
    format(S, '  end~n', []),
    format(S, '~n', []),
    format(S, '  defp execute_backtrack(state, acc) do~n', []),
    format(S, '    hops = Map.get(state.regs, 3)~n', []),
    format(S, '    h_val = if is_integer(hops), do: hops, else: String.to_integer(hops)~n', []),
    format(S, '    weight = :math.pow(h_val + 1, -5)~n', []),
    format(S, '    new_acc = acc + weight~n', []),
    format(S, '    case WamRuntime.backtrack(state) do~n', []),
    format(S, '      {:ok, next_state} -> execute_backtrack(next_state, new_acc)~n', []),
    format(S, '      :fail -> new_acc~n', []),
    format(S, '    end~n', []),
    format(S, '  end~n', []),
    format(S, '~n', []),
    format(S, '  defp load_tsv(path) do~n', []),
    format(S, '    File.stream!(path) |> Stream.drop(1) |> Stream.map(fn line ->~n', []),
    format(S, '      [a, b] = line |> String.trim() |> String.split("\\t")~n', []),
    format(S, '      {a, b}~n', []),
    format(S, '    end) |> Enum.to_list()~n', []),
    format(S, '  end~n', []),
    format(S, '~n', []),
    format(S, '  defp load_single(path) do~n', []),
    format(S, '    File.stream!(path) |> Stream.drop(1) |> Stream.map(&String.trim/1) |> Enum.to_list()~n', []),
    format(S, '  end~n', []),
    format(S, 'end~n', []),
    format(S, '~n', []),
    format(S, 'args = System.argv()~n', []),
    format(S, 'if length(args) < 1 do~n', []),
    format(S, '  IO.puts("Usage: elixir test_bench.exs <facts-dir>")~n', []),
    format(S, 'else~n', []),
    format(S, '  BenchDriver.run_all(List.first(args))~n', []),
    format(S, 'end~n', []),
    close(S).
