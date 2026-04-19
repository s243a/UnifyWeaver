:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% debug_wam_elixir_ancestor.pl - Minimal reproducer for the recursion
% gap documented in docs/design/WAM_ELIXIR_CORRECTNESS_GAPS.md.
%
% Defines a trivial recursive ancestor/2 over a 4-node chain, compiles
% it through the lowered Elixir pipeline, runs it, and reports all
% solutions found for ?- ancestor("a", Y).
%
% Expected: 3 solutions (b, c, d). The effective-distance benchmark
% finds fewer rows than reference, suggesting multi-hop recursion
% fails. This script isolates that behavior with no benchmark scaffold.
%
% Usage:
%   swipl -q -s examples/debug_wam_elixir_ancestor.pl -t halt

:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_elixir_target').
:- use_module('../src/unifyweaver/targets/wam_elixir_utils', [camel_case/2]).
:- use_module(library(option)).
:- use_module(library(lists)).
:- use_module(library(process)).

% --- Test predicate definition ---

:- dynamic user:parent/2.
:- dynamic user:ancestor/2.

:- dynamic user:ancestor_h/3.

load_test_predicates :-
    retractall(user:parent(_, _)),
    retractall(user:ancestor(_, _)),
    retractall(user:ancestor_h(_, _, _)),
    assertz((user:parent("a", "b"))),
    assertz((user:parent("b", "c"))),
    assertz((user:parent("c", "d"))),
    assertz((user:parent("d", "e"))),
    assertz((user:ancestor(X, Y) :- user:parent(X, Y))),
    assertz((user:ancestor(X, Y) :- user:parent(X, Z), user:ancestor(Z, Y))),
    % Mirror category_ancestor's arithmetic-hops pattern:
    %   ancestor_h(X, Y, 1) :- parent(X, Y).
    %   ancestor_h(X, Y, N) :- parent(X, Z), ancestor_h(Z, Y, N1), N is N1 + 1.
    assertz((user:ancestor_h(X, Y, 1) :- user:parent(X, Y))),
    assertz((user:ancestor_h(X, Y, N) :-
             user:parent(X, Z),
             user:ancestor_h(Z, Y, N1),
             N is N1 + 1)).

% --- Pipeline ---

main :-
    load_test_predicates,
    tmp_dir(TmpDir),
    format('Generating Elixir project at: ~w~n', [TmpDir]),

    Predicates = [parent/2, ancestor/2, ancestor_h/3],
    Options = [module_name('wam_elixir_ancestor'), emit_mode(lowered)],

    findall(P/A-WamCode, (
        member(P/A, Predicates),
        (   wam_target:compile_predicate_to_wam(P/A, [], WamCode) -> true
        ;   wam_target:compile_predicate_to_wam(user:P/A, [], WamCode) -> true
        ;   format(user_error, '[FAIL] could not compile ~w/~w~n', [P, A]), fail
        )
    ), PredWamPairs),

    write_wam_elixir_project(PredWamPairs, Options, TmpDir),
    write_driver(TmpDir, Options),

    run_elixir(TmpDir),
    halt(0).

tmp_dir(Dir) :-
    get_time(T),
    (   getenv('TMPDIR', Base) -> true
    ;   getenv('PREFIX', Prefix) -> atomic_list_concat([Prefix, '/tmp'], Base)
    ;   Base = '/tmp'
    ),
    format(atom(Dir), '~w/uw-elixir-ancestor-~w', [Base, T]),
    make_directory(Dir).

write_driver(TmpDir, Options) :-
    option(module_name(ModName), Options, wam_elixir_ancestor),
    camel_case(ModName, CamelMod),
    directory_file_path(TmpDir, 'test_ancestor.exs', Path),
    open(Path, write, S),
    format(S, 'Code.require_file("lib/wam_runtime.ex", __DIR__)~n', []),
    format(S, 'Code.require_file("lib/wam_dispatcher.ex", __DIR__)~n', []),
    format(S, 'Code.require_file("lib/parent.ex", __DIR__)~n', []),
    format(S, 'Code.require_file("lib/ancestor.ex", __DIR__)~n', []),
    format(S, 'Code.require_file("lib/ancestor_h.ex", __DIR__)~n', []),
    format(S, '~n', []),
    format(S, 'defmodule Driver do~n', []),
    format(S, '  def run do~n', []),
    format(S, '    IO.puts("=== Test 1: ancestor(\\"a\\", Y) -- expect [b, c, d] ===")~n', []),
    format(S, '    test1()~n', []),
    format(S, '    IO.puts("")~n', []),
    format(S, '    IO.puts("=== Test 2: ancestor_h(\\"a\\", \\"d\\", N) -- expect N=3 ===")~n', []),
    format(S, '    test2()~n', []),
    format(S, '    IO.puts("")~n', []),
    format(S, '    IO.puts("=== Test 3: ancestor_h(\\"a\\", Y, N) -- expect 3 solutions ===")~n', []),
    format(S, '    test3()~n', []),
    format(S, '  end~n', []),
    format(S, '~n', []),
    format(S, '  defp test1 do~n', []),
    format(S, '    args = ["a", {:unbound, 2}]~n', []),
    format(S, '    case ~w.Ancestor.run(args) do~n', [CamelMod]),
    format(S, '      {:ok, state} ->~n', []),
    format(S, '        y = WamRuntime.deref_var(state, Map.get(state.regs, 2))~n', []),
    format(S, '        IO.puts("  Y=#{inspect(y)}")~n', []),
    format(S, '        enumerate1(state, 1)~n', []),
    format(S, '      :fail -> IO.puts("  no solutions")~n', []),
    format(S, '      other -> IO.puts("  unexpected: #{inspect(other)}")~n', []),
    format(S, '    end~n', []),
    format(S, '  end~n', []),
    format(S, '~n', []),
    format(S, '  defp enumerate1(state, n) when n < 20 do~n', []),
    format(S, '    case WamRuntime.next_solution(state) do~n', []),
    format(S, '      {:ok, next_state} ->~n', []),
    format(S, '        y = WamRuntime.deref_var(next_state, Map.get(next_state.regs, 2))~n', []),
    format(S, '        IO.puts("  Y=#{inspect(y)}")~n', []),
    format(S, '        enumerate1(next_state, n + 1)~n', []),
    format(S, '      :fail -> IO.puts("  total: #{n}")~n', []),
    format(S, '      other -> IO.puts("  unexpected: #{inspect(other)}")~n', []),
    format(S, '    end~n', []),
    format(S, '  end~n', []),
    format(S, '  defp enumerate1(_state, n), do: IO.puts("  aborted at #{n}")~n', []),
    format(S, '~n', []),
    format(S, '  defp test2 do~n', []),
    format(S, '    args = ["a", "d", {:unbound, 3}]~n', []),
    format(S, '    case ~w.AncestorH.run(args) do~n', [CamelMod]),
    format(S, '      {:ok, state} ->~n', []),
    format(S, '        n = WamRuntime.deref_var(state, Map.get(state.regs, 3))~n', []),
    format(S, '        IO.puts("  N=#{inspect(n)}")~n', []),
    format(S, '        enumerate2(state, 1)~n', []),
    format(S, '      :fail -> IO.puts("  no solutions")~n', []),
    format(S, '      other -> IO.puts("  unexpected: #{inspect(other)}")~n', []),
    format(S, '    end~n', []),
    format(S, '  end~n', []),
    format(S, '~n', []),
    format(S, '  defp enumerate2(state, n) when n < 20 do~n', []),
    format(S, '    case WamRuntime.next_solution(state) do~n', []),
    format(S, '      {:ok, next_state} ->~n', []),
    format(S, '        v = WamRuntime.deref_var(next_state, Map.get(next_state.regs, 3))~n', []),
    format(S, '        IO.puts("  N=#{inspect(v)}")~n', []),
    format(S, '        enumerate2(next_state, n + 1)~n', []),
    format(S, '      :fail -> IO.puts("  total: #{n}")~n', []),
    format(S, '      other -> IO.puts("  unexpected: #{inspect(other)}")~n', []),
    format(S, '    end~n', []),
    format(S, '  end~n', []),
    format(S, '  defp enumerate2(_state, n), do: IO.puts("  aborted at #{n}")~n', []),
    format(S, '~n', []),
    format(S, '  defp test3 do~n', []),
    format(S, '    args = ["a", {:unbound, 2}, {:unbound, 3}]~n', []),
    format(S, '    case ~w.AncestorH.run(args) do~n', [CamelMod]),
    format(S, '      {:ok, state} ->~n', []),
    format(S, '        y = WamRuntime.deref_var(state, Map.get(state.regs, 2))~n', []),
    format(S, '        n = WamRuntime.deref_var(state, Map.get(state.regs, 3))~n', []),
    format(S, '        IO.puts("  Y=#{inspect(y)} N=#{inspect(n)}")~n', []),
    format(S, '        enumerate3(state, 1)~n', []),
    format(S, '      :fail -> IO.puts("  no solutions")~n', []),
    format(S, '      other -> IO.puts("  unexpected: #{inspect(other)}")~n', []),
    format(S, '    end~n', []),
    format(S, '  end~n', []),
    format(S, '~n', []),
    format(S, '  defp enumerate3(state, n) when n < 20 do~n', []),
    format(S, '    case WamRuntime.next_solution(state) do~n', []),
    format(S, '      {:ok, next_state} ->~n', []),
    format(S, '        y = WamRuntime.deref_var(next_state, Map.get(next_state.regs, 2))~n', []),
    format(S, '        v = WamRuntime.deref_var(next_state, Map.get(next_state.regs, 3))~n', []),
    format(S, '        IO.puts("  Y=#{inspect(y)} N=#{inspect(v)}")~n', []),
    format(S, '        enumerate3(next_state, n + 1)~n', []),
    format(S, '      :fail -> IO.puts("  total: #{n}")~n', []),
    format(S, '      other -> IO.puts("  unexpected: #{inspect(other)}")~n', []),
    format(S, '    end~n', []),
    format(S, '  end~n', []),
    format(S, '  defp enumerate3(_state, n), do: IO.puts("  aborted at #{n}")~n', []),
    format(S, 'end~n', []),
    format(S, '~n', []),
    format(S, 'Driver.run()~n', []),
    close(S).

run_elixir(TmpDir) :-
    directory_file_path(TmpDir, 'test_ancestor.exs', ScriptPath),
    format('~nRunning: elixir ~w~n', [ScriptPath]),
    format('----------------------------------------~n', []),
    process_create(path(elixir), [ScriptPath],
        [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
    read_string(Out, _, StdOut),
    read_string(Err, _, StdErr),
    process_wait(Pid, _Status),
    close(Out),
    close(Err),
    format('~s', [StdOut]),
    format('----------------------------------------~n', []),
    (   StdErr = "" -> true
    ;   format('stderr:~n~s~n', [StdErr])
    ).

:- initialization(main, main).
