:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% run_wam_kotlin_lowered_vs_interpreter.pl
%
% In-process timing of emit_mode(interpreter) vs emit_mode(functions) for the
% Kotlin hybrid WAM target. Times tryRun loops inside the generated Main
% (benchmark_main) — NOT gradle/JVM startup.
%
% Usage:
%   LANG=C.UTF-8 swipl -q -s examples/benchmark/run_wam_kotlin_lowered_vs_interpreter.pl \
%       -g main -t halt
%
% Optional argv after -- :
%   <output-dir>   (default: output/kotlin_lowered_bench)

:- use_module(library(lists)).
:- use_module(library(filesex), [
    make_directory_path/1,
    delete_directory_and_contents/1,
    directory_file_path/3
]).
:- use_module(library(process)).
:- use_module(library(option)).
:- use_module('../../src/unifyweaver/targets/wam_kotlin_target').

:- dynamic user:bk_fact/2.
:- dynamic user:bk_make_list/3.
:- dynamic user:bk_color/1.
:- dynamic user:bk_t4/2.
:- dynamic user:bk_member/2.
:- dynamic user:bk_append/3.

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [OutDir|_] -> true
    ;   OutDir = 'output/kotlin_lowered_bench'
    ),
    (   exists_directory(OutDir)
    ->  delete_directory_and_contents(OutDir)
    ;   true
    ),
    make_directory_path(OutDir),
    setup_benchmark_predicates,
    findall(Row, run_case(OutDir, Row), Rows),
    cleanup_benchmark_predicates,
    print_results(Rows),
    write_results_markdown(OutDir, Rows),
    (   forall(member(row(_, _, _, _, _, ok), Rows), true)
    ->  halt(0)
    ;   halt(1)
    ).

setup_benchmark_predicates :-
    cleanup_benchmark_predicates,
    assertz(user:bk_fact(alpha, beta)),
    assertz(user:(bk_make_list(A, B, [A, B]))),
    assertz(user:bk_color(red)),
    assertz(user:bk_color(green)),
    assertz(user:bk_color(blue)),
    assertz(user:bk_t4(_, a)),
    assertz(user:bk_t4(_, b)),
    assertz(user:bk_member(X, [X|_])),
    assertz(user:(bk_member(X, [_|T]) :- bk_member(X, T))),
    assertz(user:bk_append([], L, L)),
    assertz(user:(bk_append([H|T], L, [H|R]) :- bk_append(T, L, R))).

cleanup_benchmark_predicates :-
    retractall(user:bk_fact(_, _)),
    retractall(user:bk_make_list(_, _, _)),
    retractall(user:bk_color(_)),
    retractall(user:bk_t4(_, _)),
    retractall(user:bk_member(_, _)),
    retractall(user:bk_append(_, _, _)).

%% case(Name, Preds, PredKey, Iterations, StateSetupKotlin, ExpectNativeKeys)
bench_case(fact,
           [user:bk_fact/2],
           'bk_fact/2',
           40000,
           'stateFromCliArgs(listOf("alpha", "beta"))',
           ['bk_fact/2']).
bench_case(list_builder,
           [user:bk_make_list/3],
           'bk_make_list/3',
           20000,
           'stateFromCliArgs(listOf("alpha", "beta"))',
           ['bk_make_list/3']).
bench_case(t5_color,
           [user:bk_color/1],
           'bk_color/1',
           40000,
           'stateFromCliArgs(listOf("blue"))',
           ['bk_color/1']).
bench_case(t4_second_arg,
           [user:bk_t4/2],
           'bk_t4/2',
           30000,
           'stateFromCliArgs(listOf("x", "b"))',
           ['bk_t4/2']).
bench_case(member_100,
           [user:bk_member/2],
           'bk_member/2',
           2000,
           'run { val st = WamState(); st.writeRegister("A1", Value.IntVal(100L)); st.writeRegister("A2", intRangeList(100)); st }',
           ['bk_member/2']).
bench_case(member_500,
           [user:bk_member/2],
           'bk_member/2',
           400,
           'run { val st = WamState(); st.writeRegister("A1", Value.IntVal(500L)); st.writeRegister("A2", intRangeList(500)); st }',
           ['bk_member/2']).
bench_case(append_100,
           [user:bk_append/3],
           'bk_append/3',
           1500,
           'run { val st = WamState(); val a = intRangeList(100); val b = intRangeList(100); st.writeRegister("A1", a); st.writeRegister("A2", b); st.writeRegister("A3", st.newVariable("R")); st }',
           ['bk_append/3']).
bench_case(append_500,
           [user:bk_append/3],
           'bk_append/3',
           300,
           'run { val st = WamState(); val a = intRangeList(500); val b = intRangeList(500); st.writeRegister("A1", a); st.writeRegister("A2", b); st.writeRegister("A3", st.newVariable("R")); st }',
           ['bk_append/3']).

run_case(OutDir, row(Name, InterpMs, LowerMs, Speedup, NativeOk, Status)) :-
    bench_case(Name, Preds, PredKey, Iters, StateSetup, NativeKeys),
    format(atom(CaseDir), '~w/~w', [OutDir, Name]),
    make_directory_path(CaseDir),
    directory_file_path(CaseDir, 'interpreter', InterpDir),
    directory_file_path(CaseDir, 'functions', FunDir),
    format('~n=== case ~w (~w iters) ===~n', [Name, Iters]),
    BenchOpts = [
        benchmark_main(PredKey, Iters),
        benchmark_warmup(2),
        benchmark_batches(5),
        benchmark_state_setup(StateSetup)
    ],
    time_mode(interpreter, Preds, BenchOpts, InterpDir, InterpMs, InterpOk),
    time_mode(functions, Preds, BenchOpts, FunDir, LowerMs, LowerOk),
    confirm_native(FunDir, NativeKeys, NativeOk),
    (   InterpOk == true, LowerOk == true, NativeOk == true,
        number(InterpMs), number(LowerMs), LowerMs > 0
    ->  Speedup0 is InterpMs / LowerMs,
        format(atom(Speedup), '~2f', [Speedup0]),
        Status = ok
    ;   Speedup = nan,
        Status = fail
    ),
    format('  interpreter median_ms=~2f  lowered median_ms=~2f  speedup=~w×  native=~w  status=~w~n',
           [InterpMs, LowerMs, Speedup, NativeOk, Status]).

time_mode(Mode, Preds, BenchOpts, Dir, MedianMs, Ok) :-
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    append([emit_mode(Mode)], BenchOpts, Options),
    catch(
        wam_kotlin_target:write_wam_kotlin_project(Preds, Options, Dir),
        E,
        ( format(user_error, 'project gen failed (~w): ~q~n', [Mode, E]), fail )
    ),
    !,
    run_gradle_bench(Dir, Out, Status),
    (   Status == exit(0),
        parse_bench_line(Out, MedianMs)
    ->  Ok = true
    ;   format(user_error, 'bench run failed mode=~w status=~q out=~w~n',
               [Mode, Status, Out]),
        MedianMs = nan,
        Ok = false
    ).
time_mode(_, _, _, _, nan, false).

run_gradle_bench(Dir, Out, Status) :-
    setup_call_cleanup(
        process_create(path(gradle), ['-q', 'run'],
                       [cwd(Dir), stdout(pipe(O)), stderr(pipe(E)), process(PID)]),
        (   read_string(O, _, Out0),
            read_string(E, _, _Err),
            process_wait(PID, Status),
            Out = Out0
        ),
        (   close(O), close(E) )
    ).

parse_bench_line(Out, MedianMs) :-
    split_string(Out, "\n", "", Lines),
    member(Line, Lines),
    sub_string(Line, _, _, _, "BENCH "),
    sub_string(Line, _, _, _, "median_ms="),
    sub_string(Line, _, _, _, "ok=true"),
    once(( sub_string(Line, B, _, After, "median_ms="),
           sub_string(Line, B, Len, After, _),
           Start is B + Len,
           sub_string(Line, Start, _, 0, Rest),
           split_string(Rest, " \t", " \t", [MsStr|_]),
           number_string(MedianMs, MsStr)
         )).

confirm_native(FunDir, Keys, Ok) :-
    directory_file_path(FunDir,
        'src/main/kotlin/generated/wam/Main.kt', MainPath),
    (   exists_file(MainPath)
    ->  read_file_to_string(MainPath, Main, []),
        (   forall(member(Key, Keys),
                   ( format(string(Reg), 'registerNative("~w"', [Key]),
                     sub_string(Main, _, _, _, Reg),
                     format(string(Fun), 'fun lowered_', []),
                     sub_string(Main, _, _, _, Fun)
                   ))
        ->  Ok = true
        ;   format(user_error, 'native confirm failed for ~w~n', [Keys]),
            Ok = false
        )
    ;   Ok = false
    ).

print_results(Rows) :-
    nl,
    format('~w~n', ['program                  | interpreter ms | lowered ms | speedup | native | status']),
    format('~w~n', ['-------------------------|----------------|------------|---------|--------|-------']),
    forall(member(row(Name, I, L, S, N, St), Rows),
           format('~w~t~25+| ~w~t~15+| ~w~t~11+| ~w~t~8+| ~w~t~7+| ~w~n',
                  [Name, I, L, S, N, St])).

write_results_markdown(OutDir, Rows) :-
    directory_file_path(OutDir, 'RESULTS.md', Path),
    setup_call_cleanup(
        open(Path, write, Stream),
        (   format(Stream,
'# Kotlin WAM: interpreter vs lowered (in-process)~n~n', []),
            format(Stream,
'Measured via `benchmark_main` — warmup + median of timed `tryRun` batches inside one JVM.~n~n', []),
            format(Stream,
'| program | interpreter ms | lowered ms | speedup | native registered | status |~n', []),
            format(Stream,
'|---|---:|---:|---:|:---:|:---:|~n', []),
            forall(member(row(Name, I, L, S, N, St), Rows),
                   format(Stream, '| ~w | ~w | ~w | ~w | ~w | ~w |~n',
                          [Name, I, L, S, N, St]))
        ),
        close(Stream)
    ),
    format('Wrote ~w~n', [Path]).
