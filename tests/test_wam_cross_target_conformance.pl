:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_wam_cross_target_conformance.pl
%
% Cross-target WAM conformance harness. Compiles the SHARED classic
% programs (wam_conformance_fixtures.pl) to each WAM backend that has a
% toolchain on PATH, runs every backend against the SAME query vectors,
% and asserts the results match the single shared spec.
%
% Why: per-target classic-program tests already exist, but each
% re-declares its own fixtures, so a backend can silently diverge from
% the others without any test noticing. This harness is the safety net
% for exactly that: the Haskell `Proceed` and WAT `allocate`
% first-argument-indexing bugs (member/2 wrongly succeeding) are the kind
% of divergence it catches.
%
% INVOCATION is per-target (each adapter owns build + run):
%   - scala  passes the query args straight to GeneratedProgram (its
%     runtime parses "[a,b,c]" etc). It is NOT given 0-arity wrappers:
%     the Scala backend currently loops compiling 0-arity predicates.
%   - elixir / wat synthesise a ground 0-arity wrapper per query
%     (`ctw_N :- pred(args).`) and ask whether ctw_N succeeds — the shape
%     their own runtime tests use, sidestepping per-backend quirks in
%     feeding list args into a predicate entry point.
%
% SPEED. Builds are PER PROGRAM (one small project per program per
% backend) and the query set can be random-sampled to keep CI cheap while
% coverage accumulates across runs. Knobs (environment variables):
%   CONFORMANCE_TARGETS  = scala,elixir   limit which backends run
%   CONFORMANCE_PROGRAMS = member,fib     limit which programs run
%   CONFORMANCE_SAMPLE   = N              random N queries per program
%   CONFORMANCE_SEED     = N              seed the sampler (reproducible)
%
% Adding a target = implement ct_toolchain/2 + ct_build/4 + ct_run/5
% (+ ct_teardown/2) and register it in conformance_target/1. Missing
% toolchains skip rather than fail. Known-divergent (target, program)
% pairs are tracked in ct_xfail/2 so the suite stays green while the
% underlying gap is tracked separately.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(random)).
:- use_module('helpers/smoke_paths', [tmp_root/1]).
:- use_module('wam_conformance_fixtures').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_scala_target').
:- use_module('../src/unifyweaver/targets/wam_elixir_target').
:- use_module('../src/unifyweaver/targets/wam_wat_target').

% ============================================================
% Target registry + known-divergence (xfail) registry
% ============================================================

conformance_target(scala).
conformance_target(elixir).
conformance_target(wat).

%% ct_default_target(Target): runs unless CONFORMANCE_TARGETS overrides.
%  WAT is wired up but NOT a default: its backend currently diverges on
%  most list/arithmetic programs (see ct_xfail/ct_skip below), so it is
%  opt-in via CONFORMANCE_TARGETS=wat while those backend bugs are fixed.
ct_default_target(scala).
ct_default_target(elixir).

% Per-target adapter clauses are grouped by target, not by predicate.
:- discontiguous ct_build/4.
:- discontiguous ct_run/5.
:- discontiguous ct_teardown/2.

%% ct_xfail(Target, ProgramName)
%  (Target, program) pairs known to diverge from the shared spec.
%  Mismatches are tolerated (logged, not failed); an unexpected full
%  match (xpass) is logged so the entry can be retired once the gap is
%  fixed.
%
%  WAT member/append/reverse: WAT's runtime read-mode structure/list
%  argument unification is unimplemented (unify_* read-mode branches are
%  nops; no S-register), so get_structure/get_list match only the
%  functor and element mismatches go undetected. See
%  docs/WAM_SWITCH_INDEXING_CROSS_TARGET.md.
ct_xfail(wat, member).
ct_xfail(wat, append).
ct_xfail(wat, reverse).
%  WAT fib: is/2 with an already-bound LHS doesn't verify the computed
%  value — cfib(10,54) returns true though fib(10)=55 (the result is
%  stored over the bound arg instead of being unified/checked).
ct_xfail(wat, fib).
%  WAT builtins: cbi_arith uses // (integer div) and mod, which the WAT
%  backend does not evaluate correctly (returns false). cmp/eq are fine.
ct_xfail(wat, builtins).

%  Elixir append/reverse: the lowered Elixir backend fails to unify a
%  freshly-constructed list against an already-GROUND compound argument
%  in the clause head — e.g. capp([a],[b],[a,b]) (3rd arg ground) returns
%  false, while capp([a],[b],X), X=[a,b] succeeds. member passes because
%  it only matches an input list, never constructs+unifies a ground
%  output. Scala handles both, so this is a genuine backend divergence
%  the harness surfaced (not a harness artifact).
ct_xfail(elixir, append).
ct_xfail(elixir, reverse).

%% ct_skip(Target, ProgramName)
%  Stronger than xfail: do NOT even build/run this (target, program).
%  Used when *generation itself* is unusable, not just the answer.
%
%  WAT append/reverse: the WAT generator loops re-emitting millions of
%  "unrecognized instruction" warnings on recursive list-BUILDING
%  predicates (put_list/unify_* on a constructed tail), so the project
%  takes minutes and gigabytes of log to write. (member is fine — it only
%  matches an input list — so it stays as an xfail demonstrating the
%  read-mode divergence.) This is a separate WAT codegen bug from the
%  read-mode-unify gap; both are flagged for follow-up.
ct_skip(wat, append).
ct_skip(wat, reverse).

% ============================================================
% Toolchain probes
% ============================================================

ct_toolchain(scala,  [scalac]).
ct_toolchain(elixir, [elixir]).
ct_toolchain(wat,    [wat2wasm, node]).

ct_available(scala) :-
    ct_enabled(scala),
    exe_on_path(scalac),
    (   exe_on_path(java), scala_runtime_jars(_)
    ;   exe_on_path(scala)
    ),
    !.
ct_available(Target) :-
    Target \= scala,
    ct_enabled(Target),
    ct_toolchain(Target, Exes),
    forall(member(E, Exes), exe_on_path(E)).

ct_enabled(Target) :-
    (   getenv('CONFORMANCE_TARGETS', Spec), Spec \== ''
    ->  split_string(Spec, ",", " ", Parts), atom_string(Target, TS),
        memberchk(TS, Parts)
    ;   ct_default_target(Target) ).

exe_on_path(Exe) :-
    catch(
        ( process_create(path(Exe), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, _) ),
        _, fail).

% ============================================================
% Tests — one per target, looping the (sampled) shared programs
% ============================================================

:- begin_tests(wam_cross_target_conformance).

test(scala,  [condition(ct_available(scala))])  :- run_target_conformance(scala).
test(elixir, [condition(ct_available(elixir))]) :- run_target_conformance(elixir).
test(wat,    [condition(ct_available(wat))])    :- run_target_conformance(wat).

:- end_tests(wam_cross_target_conformance).

% ============================================================
% Driver
% ============================================================

run_target_conformance(Target) :-
    seed_sampler,
    selected_programs(Programs),
    nb_setval(ct_wrapper_counter, 0),
    foldl(run_program(Target), Programs, [], Failures),
    (   Failures == []
    ->  true
    ;   throw(error(conformance_failures(Target, Failures), _))
    ).

seed_sampler :-
    ( getenv('CONFORMANCE_SEED', S), atom_number(S, Seed)
    -> set_random(seed(Seed)) ; true ).

selected_programs(Programs) :-
    findall(P, conformance_program(P, _), All0),
    list_to_set(All0, All),
    (   getenv('CONFORMANCE_PROGRAMS', Spec), Spec \== ''
    ->  split_string(Spec, ",", " ", Parts),
        include([P]>>( atom_string(P, PS), memberchk(PS, Parts) ), All, Programs)
    ;   Programs = All ).

run_program(Target, Program, In, Out) :-
    (   ct_skip(Target, Program)
    ->  log_line(Target, Program, "SKIP (generation unusable, tracked)", []),
        Out = In
    ;   run_program_(Target, Program, In, Out) ).

run_program_(Target, Program, In, Out) :-
    conformance_program(Program, Preds),
    sampled_queries(Program, Queries),      % [q(K,A,E)...]
    (   Queries == []
    ->  Out = In
    ;   (   catch(
                setup_call_cleanup(
                    ct_build(Target, Preds, Queries, Ctx),
                    run_queries(Target, Ctx, Program, Queries, In, Out),
                    ct_teardown(Target, Ctx)),
                Err,
                build_error(Target, Program, Err, In, Out))
        ->  true
        ;   build_error(Target, Program, build_failed, In, Out) )
    ).

build_error(Target, Program, Err, In, Out) :-
    (   ct_xfail(Target, Program)
    ->  log_line(Target, Program, "XFAIL (build/run error, tolerated): ~w", [Err]),
        Out = In
    ;   format(atom(F), '~w/~w: build/run error: ~w', [Target, Program, Err]),
        Out = [F|In] ).

run_queries(Target, Ctx, Program, Queries, In, Out) :-
    foldl(run_query(Target, Ctx, Program), Queries, In, Out).

run_query(Target, Ctx, Program, q(K, A, Expected), In, Out) :-
    (   catch(ct_run(Target, Ctx, K, A, Got), E, ( Got = error(E), true ))
    ->  true ; Got = error(run_failed) ),
    (   Got == Expected
    ->  (   ct_xfail(Target, Program)
        ->  log_line(Target, Program, "XPASS ~w(~w) matched under xfail", [K, A]),
            Out = In
        ;   Out = In )
    ;   (   ct_xfail(Target, Program)
        ->  log_line(Target, Program,
                     "xfail ~w(~w): expected ~w got ~w (tolerated)",
                     [K, A, Expected, Got]),
            Out = In
        ;   format(atom(F), '~w/~w: ~w(~w) expected ~w got ~w',
                   [Target, Program, K, A, Expected, Got]),
            Out = [F|In] ) ).

log_line(Target, Program, Fmt, Args) :-
    format(string(Body), Fmt, Args),
    format(user_error, '  [conformance] ~w/~w: ~w~n', [Target, Program, Body]).

% ============================================================
% Query sampling
% ============================================================

sampled_queries(Program, Picked) :-
    findall(q(K, A, E), conformance_query(Program, K, A, E), Qs),
    (   getenv('CONFORMANCE_SAMPLE', NS), atom_number(NS, N), integer(N), N >= 0
    ->  random_subset(Qs, N, Picked)
    ;   Picked = Qs ).

random_subset(List, N, Picked) :-
    length(List, Len),
    (   N >= Len -> Picked = List
    ;   random_permutation(List, Shuffled),
        length(Picked, N), append(Picked, _, Shuffled) ).

% ============================================================
% Shared helpers
% ============================================================

%% render_arg(+Term, -Atom): textual form the arg-passing drivers parse:
%  ints -> '10', atoms -> 'a', atom-lists -> '[a,b,c]'.
render_arg(Term, Atom) :- format(atom(Atom), '~w', [Term]).

bool_of_string(S, B) :-
    string_to_atom(S, A),
    ( A == true -> B = true ; A == false -> B = false ; B = A ).

ct_tmp_dir(Prefix, Dir) :-
    get_time(T), Stamp is floor(T * 1000000),
    tmp_root(Root),
    format(atom(Base), '~w_~w', [Prefix, Stamp]),
    directory_file_path(Root, Base, Dir),
    make_directory_path(Dir).

cleanup_dir(Dir) :-
    ( atom(Dir), exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

strip_pred(_Mod:N/A, N/A) :- !.
strip_pred(N/A, N/A).

run_proc(Exe, Args, Cwd, Exit, ErrStr) :-
    process_create(path(Exe), Args,
                   [cwd(Cwd), stdout(null), stderr(pipe(Err)), process(Pid)]),
    read_string(Err, _, ErrStr), close(Err), process_wait(Pid, exit(Exit)).

run_proc_out(Exe, Args, Cwd, Exit, OutStr) :-
    process_create(path(Exe), Args,
                   [cwd(Cwd), stdout(pipe(Out)), stderr(null), process(Pid)]),
    read_string(Out, _, OutStr), close(Out), process_wait(Pid, exit(Exit)).

run_proc_out_err(Exe, Args, Cwd, Exit, OutStr, ErrStr) :-
    process_create(path(Exe), Args,
                   [cwd(Cwd), stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
    read_string(Out, _, OutStr),
    read_string(Err, _, ErrStr),
    close(Out), close(Err),
    process_wait(Pid, exit(Exit)).

%% Wrapper synthesis (elixir / wat). One ground 0-arity wrapper per
%  query, asserted into user as `ctw_N :- pred(Args).`. The global
%  counter keeps names unique across programs in one run. Returns the
%  wrapper predicate indicators and a (K-A)->WName lookup map.
synth_wrappers([], [], []).
synth_wrappers([q(K, A, _E)|Rest], [WName/0|WPs], [(K-A)-WName|Map]) :-
    nb_getval(ct_wrapper_counter, I0), I is I0 + 1,
    nb_setval(ct_wrapper_counter, I),
    atom_concat(ctw_, I, WName),
    pred_name_of_key(K, PredName),
    Goal =.. [PredName|A],
    assertz(user:(WName :- Goal)),
    synth_wrappers(Rest, WPs, Map).

abolish_wrappers([]).
abolish_wrappers([(_K-_A)-WName|Rest]) :-
    ( current_predicate(user:WName/0) -> abolish(user:WName/0) ; true ),
    abolish_wrappers(Rest).

pred_name_of_key(Key, Name) :-
    atomic_list_concat([NameA|_], '/', Key), atom_string(Name, NameA).

% ============================================================
% Adapter: Scala  (direct args: GeneratedProgram <predkey> <args>)
% ============================================================

ct_build(scala, Preds, _Queries, scala_ctx(Dir)) :-
    ct_tmp_dir('tmp_ct_scala', Dir),
    Opts = [ package('generated.wam_ct.core'),
             runtime_package('generated.wam_ct.core'), module_name('wam-ct') ],
    write_wam_scala_project(Preds, Opts, Dir),
    absolute_file_name(Dir, Abs),
    directory_file_path(Abs, 'classes', ClassDir), make_directory_path(ClassDir),
    directory_file_path(Abs, 'src', SrcDir),
    findall(F, ( directory_member(SrcDir, RelF,
                     [extensions([scala]), recursive(true)]),
                 directory_file_path(SrcDir, RelF, F) ), Sources),
    Sources \= [],
    run_proc(scalac, ['-d', ClassDir | Sources], Abs, Exit, Err),
    ( Exit =:= 0 -> true ; throw(scala_compile_failed(Exit, Err)) ).

ct_run(scala, scala_ctx(Dir), PredKey, Args, Bool) :-
    absolute_file_name(Dir, Abs),
    directory_file_path(Abs, 'classes', ClassDir),
    atom_string(PredKey, PredStr),
    maplist(render_arg, Args, ArgAtoms),
    maplist([X,S]>>atom_string(X,S), ArgAtoms, ArgStrs),
    scala_run_command(ClassDir, 'generated.wam_ct.core.GeneratedProgram', PredStr,
                      ArgStrs, Exe, ProcArgs),
    run_proc_out_err(Exe, ProcArgs, Abs, Exit, OutStr, ErrStr),
    (   Exit =:= 0
    ->  normalize_space(string(Out), OutStr), bool_of_string(Out, Bool)
    ;   throw(scala_run_failed(Exit, PredKey, Args, ErrStr))
    ).

scala_run_command(ClassDir, MainClass, PredStr, ArgStrs, java, ProcArgs) :-
    scala_runtime_classpath(ClassDir, ClassPath),
    !,
    append(['-cp', ClassPath, MainClass, PredStr], ArgStrs, ProcArgs).
scala_run_command(ClassDir, MainClass, PredStr, ArgStrs, scala, ProcArgs) :-
    append(['-classpath', ClassDir, MainClass, PredStr], ArgStrs, ProcArgs).

scala_runtime_classpath(ClassDir, ClassPath) :-
    scala_runtime_jars(Jars),
    path_list_separator(Sep),
    atomic_list_concat([ClassDir|Jars], Sep, ClassPath).

scala_runtime_jars(Jars) :-
    findall(Jar, scala_runtime_jar(Jar), Jars0),
    sort(Jars0, Jars),
    Jars \= [].

path_list_separator(';') :- current_prolog_flag(windows, true), !.
path_list_separator(':').

scala_runtime_jar(Jar) :-
    scala_lang_maven_root(Root),
    member(Artifact, ['scala-library', 'scala3-library_3', 'tasty-core_3']),
    directory_file_path(Root, Artifact, ArtifactDir),
    exists_directory(ArtifactDir),
    directory_member(ArtifactDir, Rel,
                     [extensions([jar]), recursive(true)]),
    directory_file_path(ArtifactDir, Rel, Jar).

scala_lang_maven_root(Root) :-
    getenv('SCALA_MAVEN_ROOT', Raw),
    Raw \== '',
    exists_directory(Raw),
    !,
    Root = Raw.
scala_lang_maven_root(Root) :-
    getenv('PREFIX', Prefix),
    Prefix \== '',
    directory_file_path(Prefix, 'opt/scala/maven2/org/scala-lang', Root),
    exists_directory(Root).

ct_teardown(scala, scala_ctx(Dir)) :- cleanup_dir(Dir).

% ============================================================
% Adapter: Elixir  (0-arity wrapper via run_classic.exs <Module> <wname>/0)
% ============================================================

ct_build(elixir, Preds, Queries, elixir_ctx(Dir, Map)) :-
    ct_tmp_dir('tmp_ct_elixir', Dir),
    synth_wrappers(Queries, WPreds, Map),
    maplist(strip_pred, Preds, BarePreds),
    append(WPreds, BarePreds, AllPreds),
    ct_compile_preds_to_wam(AllPreds, [], PredWamPairs),
    write_wam_elixir_project(PredWamPairs,
        [ module_name('wam_ct'), emit_mode(lowered), source_module(user) ], Dir),
    classic_elixir_driver(DriverSrc),
    directory_file_path(Dir, 'run_classic.exs', DriverDst),
    copy_file(DriverSrc, DriverDst).

ct_run(elixir, elixir_ctx(Dir, Map), K, A, Bool) :-
    memberchk((K-A)-WName, Map),
    absolute_file_name(Dir, Abs),
    format(atom(PredKey), '~w/0', [WName]), atom_string(PredKey, PredStr),
    run_proc_out(elixir, ['run_classic.exs', "WamCt", PredStr], Abs, _Exit, OutStr),
    normalize_space(string(Out), OutStr), bool_of_string(Out, Bool).

ct_teardown(elixir, elixir_ctx(Dir, Map)) :-
    cleanup_dir(Dir), abolish_wrappers(Map).

ct_compile_preds_to_wam([], _Opts, []).
ct_compile_preds_to_wam([Pred|Rest], Opts, [Name/Arity-WamCode|RestPairs]) :-
    strip_pred(Pred, Name/Arity),
    (   wam_target:compile_predicate_to_wam(Name/Arity, Opts, WamCode) -> true
    ;   wam_target:compile_predicate_to_wam(user:Name/Arity, Opts, WamCode) -> true
    ;   throw(wam_compile_failed(Pred))
    ),
    ct_compile_preds_to_wam(Rest, Opts, RestPairs).

:- ( prolog_load_context(directory, Dir),
     directory_file_path(Dir, 'elixir_e2e/run_classic.exs', P),
     assertz(classic_elixir_driver_fact(P))
   ; true ).
classic_elixir_driver(P) :- classic_elixir_driver_fact(P).

% ============================================================
% Adapter: WAT  (0-arity wrapper -> wasm export <wname>_0 -> 1/0)
% ============================================================

ct_build(wat, Preds, Queries, wat_ctx(Dir, HarnessJs, WasmFile, Map)) :-
    ct_tmp_dir('tmp_ct_wat', Dir),
    synth_wrappers(Queries, WPreds, Map),
    maplist(strip_pred, Preds, BarePreds),
    append(WPreds, BarePreds, AllPreds),
    directory_file_path(Dir, 'prog.wat', WatFile),
    write_wam_wat_project(AllPreds, [module_name(wam_ct)], WatFile),
    directory_file_path(Dir, 'prog.wasm', WasmFile),
    run_proc(wat2wasm, [WatFile, '-o', WasmFile], Dir, CExit, CErr),
    ( CExit =:= 0 -> true ; throw(wat2wasm_failed(CExit, CErr)) ),
    wat_write_harness(Dir, HarnessJs).

ct_run(wat, wat_ctx(_Dir, HarnessJs, WasmFile, Map), K, A, Bool) :-
    memberchk((K-A)-WName, Map),
    format(atom(Export), '~w_0', [WName]), atom_string(Export, ExportStr),
    run_proc_out(node, [HarnessJs, WasmFile, ExportStr], '.', _Exit, OutStr),
    normalize_space(string(Out), OutStr),
    ( Out == "1" -> Bool = true ; Out == "0" -> Bool = false ; atom_string(Bool, Out) ).

ct_teardown(wat, wat_ctx(Dir, _, _, Map)) :-
    cleanup_dir(Dir), abolish_wrappers(Map).

wat_write_harness(Dir, HarnessJs) :-
    directory_file_path(Dir, 'harness.js', HarnessJs),
    Src = "const fs=require('fs');\n\c
const [,,wasmPath,exportName]=process.argv;\n\c
const buf=fs.readFileSync(wasmPath);\n\c
const imports={env:{print_i64:()=>{},print_char:()=>{},print_newline:()=>{}}};\n\c
WebAssembly.instantiate(buf,imports).then(({instance})=>{\n\c
  const fn=instance.exports[exportName];\n\c
  if(typeof fn!=='function'){console.log('ERR');process.exit(1);}\n\c
  console.log(fn());\n\c
}).catch(e=>{console.log('TRAP');process.exit(2);});\n",
    open(HarnessJs, write, S), write(S, Src), close(S).
