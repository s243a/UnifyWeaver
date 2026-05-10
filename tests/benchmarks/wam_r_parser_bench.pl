:- encoding(utf8).
%% Microbenchmark for the WAM R hybrid target's parser layer.
%% Compares the inline R parser (WamRuntime$wam_parse_expr / parse_term)
%% against the compiled-from-Prolog cross-target parser
%% (prolog_term_parser:parse_term_from_atom). The two parsers produce
%% the same terms (proven by tests/test_prolog_term_parser_wam_r_compile),
%% so this benchmark is purely about the cost of swapping out the
%% inline path for the compiled one as the runtime's parser.
%%
%% For each input the benchmark drives `Rscript --bench N` against:
%%   inline    -- a driver that calls read_term_from_atom/2, which
%%                dispatches into the inline parser.
%%   compiled  -- a driver that calls
%%                prolog_term_parser:parse_term_from_atom/3 with the
%%                canonical op table built per-iteration. The op-table
%%                construction is included in the timing because it
%%                would also be required at every call site if the
%%                runtime swapped to the compiled parser internally.
%%
%% Inputs cover the parser surface:
%%   atom      `foo`
%%   integer   `42`
%%   compound  `foo(a, b)`
%%   arith     `1 + 2 * 3`     (operator precedence)
%%   list      `[a, b, c]`
%%   nested    `foo(bar(1), [a, b])`
%%
%% Usage:
%%   swipl -g main -t halt tests/benchmarks/wam_r_parser_bench.pl
%%   swipl -g main -t halt tests/benchmarks/wam_r_parser_bench.pl -- --inner 2000
%%
%% Inner iterations default to 1000. Honours WAM_R_BENCH_KEEP=1 to
%% retain the generated project.

:- use_module('../../src/unifyweaver/core/prolog_term_parser').
:- use_module('../../src/unifyweaver/targets/wam_r_target').
:- use_module(library(filesex), [directory_file_path/3,
                                  delete_directory_and_contents/1]).
:- use_module(library(process)).
:- use_module(library(readutil)).

rscript_available :-
    catch(
        (   process_create(path('Rscript'), ['--version'],
                [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0))
        ),
        _, fail).

%% ========================================================================
%% Driver predicates (asserted into user)
%% ========================================================================

setup_drivers :-
    retractall(user:bench_inline_atom/0),
    retractall(user:bench_inline_int/0),
    retractall(user:bench_inline_compound/0),
    retractall(user:bench_inline_arith/0),
    retractall(user:bench_inline_list/0),
    retractall(user:bench_inline_nested/0),
    retractall(user:bench_compiled_atom/0),
    retractall(user:bench_compiled_int/0),
    retractall(user:bench_compiled_compound/0),
    retractall(user:bench_compiled_arith/0),
    retractall(user:bench_compiled_list/0),
    retractall(user:bench_compiled_nested/0),
    assertz((user:bench_inline_atom :-
        read_term_from_atom('foo', _))),
    assertz((user:bench_inline_int :-
        read_term_from_atom('42', _))),
    assertz((user:bench_inline_compound :-
        read_term_from_atom('foo(a, b)', _))),
    assertz((user:bench_inline_arith :-
        read_term_from_atom('1 + 2 * 3', _))),
    assertz((user:bench_inline_list :-
        read_term_from_atom('[a, b, c]', _))),
    assertz((user:bench_inline_nested :-
        read_term_from_atom('foo(bar(1), [a, b])', _))),
    assertz((user:bench_compiled_atom :-
        canonical_op_table(O), parse_term_from_atom('foo', O, _))),
    assertz((user:bench_compiled_int :-
        canonical_op_table(O), parse_term_from_atom('42', O, _))),
    assertz((user:bench_compiled_compound :-
        canonical_op_table(O), parse_term_from_atom('foo(a, b)', O, _))),
    assertz((user:bench_compiled_arith :-
        canonical_op_table(O), parse_term_from_atom('1 + 2 * 3', O, _))),
    assertz((user:bench_compiled_list :-
        canonical_op_table(O), parse_term_from_atom('[a, b, c]', O, _))),
    assertz((user:bench_compiled_nested :-
        canonical_op_table(O),
        parse_term_from_atom('foo(bar(1), [a, b])', O, _))).

%% ========================================================================
%% Project builder
%% ========================================================================

parser_module_predicates(Preds) :-
    findall(prolog_term_parser:N/A,
            (   current_predicate(prolog_term_parser:N/A),
                functor(H, N, A),
                once(clause(prolog_term_parser:H, _)),
                \+ predicate_property(prolog_term_parser:H,
                                       imported_from(_))
            ),
            Raw),
    sort(Raw, Preds).

driver_predicates([
    user:bench_inline_atom/0,
    user:bench_inline_int/0,
    user:bench_inline_compound/0,
    user:bench_inline_arith/0,
    user:bench_inline_list/0,
    user:bench_inline_nested/0,
    user:bench_compiled_atom/0,
    user:bench_compiled_int/0,
    user:bench_compiled_compound/0,
    user:bench_compiled_arith/0,
    user:bench_compiled_list/0,
    user:bench_compiled_nested/0
]).

build_project(ProjectDir) :-
    catch(delete_directory_and_contents(ProjectDir), _, true),
    setup_drivers,
    parser_module_predicates(ParserPreds),
    driver_predicates(Drivers),
    append(Drivers, ParserPreds, Compile),
    write_wam_r_project(Compile, [], ProjectDir).

%% ========================================================================
%% Rscript helpers (mirrors wam_r_fact_source_bench)
%% ========================================================================

run_rscript_bench(RDir, N, PredKey, BenchSec) :-
    atom_string(N, NStr),
    ProcArgs = ['generated_program.R', '--bench', NStr, PredKey],
    process_create(path('Rscript'), ProcArgs,
                   [cwd(RDir), stdout(pipe(O)), stderr(pipe(E)),
                    process(Pid)]),
    read_string(O, _, OutStr), read_string(E, _, ErrStr),
    close(O), close(E),
    process_wait(Pid, exit(EC)),
    (   (EC =:= 0 ; EC =:= 1)
    ->  parse_bench_line(OutStr, BenchSec)
    ;   throw(error(rscript_bench_failed(EC, PredKey, ErrStr), _))
    ).

parse_bench_line(Str, Seconds) :-
    split_string(Str, "\n", "", Lines),
    member(Line, Lines),
    sub_string(Line, _, _, _, "BENCH"),
    sub_string(Line, B, _, _, "elapsed="),
    Bp is B + 8,
    sub_string(Line, Bp, _, 0, Tail),
    split_string(Tail, " \n", "", [SecStr | _]),
    number_string(Seconds, SecStr), !.

%% ========================================================================
%% Bench driver
%% ========================================================================

input_cases([
    'atom'-bench_inline_atom-bench_compiled_atom,
    'integer'-bench_inline_int-bench_compiled_int,
    'compound'-bench_inline_compound-bench_compiled_compound,
    'arith'-bench_inline_arith-bench_compiled_arith,
    'list'-bench_inline_list-bench_compiled_list,
    'nested'-bench_inline_nested-bench_compiled_nested
]).

run_bench(InnerN) :-
    bench_proj_path(ProjectDir),
    format("[INFO] inner=~w iterations per case~n", [InnerN]),
    format("[INFO] building project...~n", []),
    build_project(ProjectDir),
    directory_file_path(ProjectDir, 'R', RDir),
    format("[INFO] running benchmarks via Rscript --bench ~w...~n~n",
           [InnerN]),
    format("~`-t~64|~n", []),
    format("~t~w~10| ~t~w~22| ~t~w~36| ~t~w~52| ~t~w~64|~n",
           ['input', 'inline (s)', 'compiled (s)',
            'per-iter (us)', 'ratio']),
    format("~`-t~64|~n", []),
    input_cases(Cases),
    forall(member(Name-IPred-CPred, Cases),
        ( inline_pred_key(IPred, IKey),
          inline_pred_key(CPred, CKey),
          run_rscript_bench(RDir, InnerN, IKey, ISec),
          run_rscript_bench(RDir, InnerN, CKey, CSec),
          Ratio is CSec / ISec,
          PerIterUs is (CSec / InnerN) * 1_000_000,
          format("~t~w~10| ~t~6f~22| ~t~6f~36| ~t~1f~52| ~t~1fx~64|~n",
                 [Name, ISec, CSec, PerIterUs, Ratio])
        )),
    format("~`-t~64|~n~n", []),
    cleanup_after(ProjectDir).

inline_pred_key(Name, Key) :-
    atom_string(Name, NameStr),
    string_concat(NameStr, "/0", Key).

bench_proj_path(P) :-
    absolute_file_name('_tmp_wam_r_parser_bench', P).

cleanup_after(ProjectDir) :-
    (   getenv('WAM_R_BENCH_KEEP', "1")
    ->  format("[INFO] keeping project at ~w (WAM_R_BENCH_KEEP=1)~n",
               [ProjectDir])
    ;   catch(delete_directory_and_contents(ProjectDir), _, true)
    ).

%% ========================================================================
%% Entry point
%% ========================================================================

inner_iterations(I) :-
    current_prolog_flag(argv, Argv),
    append(_, ['--inner', IStr], Argv),
    atom_number(IStr, I),
    integer(I), I > 0, !.
inner_iterations(1000).

main :-
    (   rscript_available
    ->  inner_iterations(I),
        run_bench(I)
    ;   format(user_error, "[SKIP] Rscript not on PATH~n", [])
    ).
