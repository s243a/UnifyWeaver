:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_js_pattern_cross_target_conformance.pl
%
% Cross-target parity harness for the JavaScript *pattern* targets. It is
% the JS-pattern analogue of test_wam_cross_target_conformance.pl: it
% compiles the SHARED fixture set (js_pattern_conformance_fixtures.pl) with
% each JS pattern backend that has a toolchain on PATH, runs the generated
% code under that backend's runtime, and diffs the result against the single
% shared Prolog-oracle spec.
%
% Why: the WAM backends have such a harness; the pattern JS targets did not.
% Per-target pattern tests exist, but each re-declares its own expectations,
% so a pattern backend can silently diverge from the shared semantics with
% no test noticing.
%
% ARMS: typescript, annotated_js, vanilla_js, clojurescript.
%
%   typescript    - compile_recursion/3 (linear/tail/list-fold canned
%                   templates) and compile_module/3 (factorial) emit clean,
%                   tsc-clean, node-runnable functions. Built with a small
%                   appended CLI driver, type-checked+emitted with `tsc`
%                   (or `npx tsc`), then run with `node`.
%   annotated_js  - a fuller JS pattern target being built in parallel; may
%   vanilla_js      not be registered in this worktree. Each arm is guarded
%                   by a target-exists check and SKIPS cleanly when absent.
%                   Run with `node`.
%   clojurescript - compiled with compile_predicate_to_clojurescript/3 and
%                   run with `nbb` (node-babashka). Its recursion-pattern
%                   codegen is not yet clean (see ja_xfail below); with nbb
%                   absent the whole arm skips at the availability gate.
%
% SKIP, NEVER FAIL. Missing runtimes (node / tsc / nbb) and not-yet-existing
% targets (annotated_js / vanilla_js unregistered) make the corresponding
% arm's plunit `condition/1` fail, so the arm is reported SKIPPED, not
% failed. Within a running arm, a program whose family the arm does not
% support is skipped and logged; a genuine result MISMATCH on a supported
% program is a real failure (that is the point of the safety net), unless
% the (arm, program) pair is registered in ja_xfail/2 (tolerated, tracked).
%
% ENV KNOBS (analogous to CONFORMANCE_TARGETS / CONFORMANCE_PROGRAMS):
%   JS_CONFORMANCE_TARGETS  = typescript,clojurescript  limit which arms run
%   JS_CONFORMANCE_PROGRAMS = fib,sum                   limit which programs
%
% Entry point:
%   swipl -q -g run_tests -t halt tests/test_js_pattern_cross_target_conformance.pl

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('js_pattern_conformance_fixtures').

% Pattern targets. Import only the entry points we call module-qualified, so
% the two modules' shared compile_predicate/3 export does not clash on import.
:- use_module('../src/unifyweaver/targets/typescript_target',
              [ compile_recursion/3, compile_module/3 ]).
:- use_module('../src/unifyweaver/targets/clojurescript_target',
              [ compile_predicate_to_clojurescript/3 ]).
:- use_module('../src/unifyweaver/core/target_registry',
              [ target_module/2 ]).

% ============================================================
% Arm registry + known-divergence (xfail) registry
% ============================================================

% Per-arm adapter clauses are grouped by arm, not by predicate.
:- discontiguous ja_build/3.
:- discontiguous ja_run/4.
:- discontiguous ja_teardown/2.

ja_arm(typescript).
ja_arm(annotated_js).
ja_arm(vanilla_js).
ja_arm(clojurescript).

%% ja_default_arm(Arm): runs unless JS_CONFORMANCE_TARGETS overrides.
%  All four arms are on by default; absent runtimes / targets skip cleanly.
ja_default_arm(typescript).
ja_default_arm(annotated_js).
ja_default_arm(vanilla_js).
ja_default_arm(clojurescript).

%% ja_supported_family(Arm, Family)
%  Which fixture families an arm can actually compile+run.
ja_supported_family(typescript,    numeric).
ja_supported_family(clojurescript, numeric).
% The fuller targets (built in parallel) aim to cover both families.
ja_supported_family(annotated_js,  numeric).
ja_supported_family(annotated_js,  structural).
ja_supported_family(vanilla_js,    numeric).
ja_supported_family(vanilla_js,    structural).

%% ja_xfail(Arm, Program)
%  (arm, program) pairs known to diverge from the shared spec. Mismatches
%  and build/run errors are tolerated (logged, not failed); an unexpected
%  full match (xpass) is logged so the entry can be retired.
%
%  clojurescript numeric: the CLJS pattern target inherits clojure_target's
%  native-clause lowering, which does not yet emit clean recursive numeric
%  code (recursion currently reaches the shared TypeScript recursion-pattern
%  clause, not a ClojureScript one). Tracked here so the arm stays green if
%  an nbb runtime is present, until a ClojureScript recursion path lands.
:- dynamic ja_xfail/2.
ja_xfail(clojurescript, fib).
ja_xfail(clojurescript, factorial).
ja_xfail(clojurescript, sum).
ja_xfail(clojurescript, listsum).

% ============================================================
% Availability: enabled + target present + runtime present
% ============================================================

ja_enabled(Arm) :-
    (   getenv('JS_CONFORMANCE_TARGETS', Spec), Spec \== ''
    ->  split_string(Spec, ",", " ", Parts), atom_string(Arm, AS),
        memberchk(AS, Parts)
    ;   ja_default_arm(Arm) ).

%% ja_target_present(+Arm)
%  The typescript and clojurescript targets are loaded above (built in).
%  annotated_js / vanilla_js must be registered in the target registry AND
%  expose a compile_predicate/3 in their module, or the arm skips.
ja_target_present(typescript)    :- !.
ja_target_present(clojurescript) :- !.
ja_target_present(Arm) :-
    catch(target_module(Arm, Module), _, fail),
    current_predicate(Module:compile_predicate/3).

%% ja_runtime_present(+Arm)
ja_runtime_present(typescript)    :- exe_on_path(node), ts_compiler(_, _).
ja_runtime_present(annotated_js)  :- exe_on_path(node).
ja_runtime_present(vanilla_js)    :- exe_on_path(node).
ja_runtime_present(clojurescript) :- exe_on_path(nbb).

ja_available(Arm) :-
    ja_enabled(Arm),
    ja_target_present(Arm),
    ja_runtime_present(Arm).

%% ts_compiler(-Exe, -PreArgs): resolve a TypeScript compiler invocation.
%  Prefer a direct `tsc`; fall back to `npx tsc`.
ts_compiler(tsc, []) :- exe_on_path(tsc), !.
ts_compiler(npx, [tsc]) :- exe_on_path(npx).

exe_on_path(Exe) :-
    catch(
        ( process_create(path(Exe), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, _) ),
        _, fail).

% ============================================================
% Tests — one per arm, plus an always-run oracle self-check
% ============================================================

:- begin_tests(js_pattern_cross_target_conformance).

% Always runs (no toolchain needed): proves the hand-specified Expected
% values match the live Prolog oracle for every program/query.
test(js_oracle_self_check) :-
    findall(P-I-E, js_conformance_query(P, I, E), Queries),
    foldl(check_oracle_query, Queries, [], Bad),
    (   Bad == []
    ->  true
    ;   throw(error(js_oracle_mismatch(Bad), _))
    ).

test(typescript,    [condition(ja_available(typescript))])    :- run_arm_conformance(typescript).
test(annotated_js,  [condition(ja_available(annotated_js))])  :- run_arm_conformance(annotated_js).
test(vanilla_js,    [condition(ja_available(vanilla_js))])    :- run_arm_conformance(vanilla_js).
test(clojurescript, [condition(ja_available(clojurescript))]) :- run_arm_conformance(clojurescript).

:- end_tests(js_pattern_cross_target_conformance).

check_oracle_query(P-I-E, In, Out) :-
    (   catch(js_oracle(P, I, Val), _, fail),
        render_value(Val, VS),
        render_value(E, ES),
        VS == ES
    ->  Out = In
    ;   ( catch(js_oracle(P, I, Val2), _, Val2 = oracle_error) -> true ; Val2 = oracle_failed ),
        format(atom(F), '~w~w: oracle ~w expected ~w', [P, I, Val2, E]),
        Out = [F|In]
    ).

% ============================================================
% Driver
% ============================================================

run_arm_conformance(Arm) :-
    selected_programs(Programs),
    foldl(run_program(Arm), Programs, [], Failures),
    (   Failures == []
    ->  true
    ;   throw(error(js_conformance_failures(Arm, Failures), _))
    ).

selected_programs(Programs) :-
    findall(P, js_conformance_program(P, _, _, _, _, _), All0),
    list_to_set(All0, All),
    (   getenv('JS_CONFORMANCE_PROGRAMS', Spec), Spec \== ''
    ->  split_string(Spec, ",", " ", Parts),
        include([P]>>( atom_string(P, PS), memberchk(PS, Parts) ), All, Programs)
    ;   Programs = All ).

run_program(Arm, Program, In, Out) :-
    js_conformance_program(Program, Family, _Pattern, _Fn, _ArgKind, _Preds),
    (   \+ ja_supported_family(Arm, Family)
    ->  log_line(Arm, Program, "SKIP (family ~w not supported by this arm)", [Family]),
        Out = In
    ;   run_program_(Arm, Program, In, Out) ).

run_program_(Arm, Program, In, Out) :-
    program_queries(Program, Queries),
    (   Queries == []
    ->  Out = In
    ;   (   catch(
                setup_call_cleanup(
                    ja_build(Arm, Program, Ctx),
                    run_queries(Arm, Ctx, Program, Queries, In, Out),
                    ja_teardown(Arm, Ctx)),
                Err,
                build_error(Arm, Program, Err, In, Out))
        ->  true
        ;   build_error(Arm, Program, build_failed, In, Out) )
    ).

build_error(Arm, Program, Err, In, Out) :-
    (   ja_xfail(Arm, Program)
    ->  log_line(Arm, Program, "XFAIL (build/run error, tolerated): ~w", [Err]),
        Out = In
    ;   format(atom(F), '~w/~w: build/run error: ~w', [Arm, Program, Err]),
        Out = [F|In] ).

run_queries(Arm, Ctx, Program, Queries, In, Out) :-
    foldl(run_query(Arm, Ctx, Program), Queries, In, Out).

run_query(Arm, Ctx, Program, q(Inputs, Expected), In, Out) :-
    (   catch(ja_run(Arm, Ctx, Inputs, Got), E, ( Got = error(E), true ))
    ->  true ; Got = error(run_failed) ),
    render_value(Expected, ExpStr),
    (   Got == ExpStr
    ->  (   ja_xfail(Arm, Program)
        ->  log_line(Arm, Program, "XPASS ~w matched under xfail", [Inputs]),
            Out = In
        ;   Out = In )
    ;   (   ja_xfail(Arm, Program)
        ->  log_line(Arm, Program,
                     "xfail ~w: expected ~w got ~w (tolerated)",
                     [Inputs, ExpStr, Got]),
            Out = In
        ;   format(atom(F), '~w/~w: ~w expected ~w got ~w',
                   [Arm, Program, Inputs, ExpStr, Got]),
            Out = [F|In] ) ).

program_queries(Program, Queries) :-
    findall(q(I, E), js_conformance_query(Program, I, E), Queries).

log_line(Arm, Program, Fmt, Args) :-
    format(string(Body), Fmt, Args),
    format(user_error, '  [js-conformance] ~w/~w: ~w~n', [Arm, Program, Body]).

% ============================================================
% Shared helpers
% ============================================================

%% render_value(+Term, -Str): canonical textual form for comparison.
%  ints -> "55", atoms -> "positive", true/false -> "true"/"false".
render_value(Term, Str) :- format(string(Str), '~w', [Term]).

ja_tmp_dir(Prefix, Dir) :-
    get_time(T), Stamp is floor(T * 1000000),
    tmp_root(Root),
    format(atom(Base), '~w_~w', [Prefix, Stamp]),
    directory_file_path(Root, Base, Dir),
    make_directory_path(Dir).

run_proc(Exe, Args, Cwd, Exit, ErrStr) :-
    process_create(path(Exe), Args,
                   [cwd(Cwd), stdout(null), stderr(pipe(Err)), process(Pid)]),
    read_string(Err, _, ErrStr), close(Err), process_wait(Pid, exit(Exit)).

run_proc_out(Exe, Args, Cwd, Exit, OutStr) :-
    process_create(path(Exe), Args,
                   [cwd(Cwd), stdout(pipe(Out)), stderr(null), process(Pid)]),
    read_string(Out, _, OutStr), close(Out), process_wait(Pid, exit(Exit)).

%% render_input_arg(+ArgKind, +Term, -CliArg): the CLI string the driver
%  parses back. num -> "10"; numlist -> "1,2,3,4".
render_input_arg(numlist, List, CliArg) :-
    !,
    maplist([X,S]>>format(atom(S), '~w', [X]), List, Parts),
    atomic_list_concat(Parts, ',', CliArg).
render_input_arg(_Kind, Term, CliArg) :-
    format(atom(CliArg), '~w', [Term]).

%% driver_arg_expr(+ArgKind, -Expr): the JS expression that parses argv[2].
driver_arg_expr(numlist, 'String(__g.process.argv[2]).split(",").map(Number)').
driver_arg_expr(num,     'parseInt(__g.process.argv[2])').

% ============================================================
% Adapter: TypeScript
%   compile via the recursion-pattern / module path, append a CLI driver,
%   type-check + emit with tsc (or npx tsc), run with node.
% ============================================================

ja_build(typescript, Program, ts_ctx(Dir, ArgKind)) :-
    js_conformance_program(Program, numeric, Pattern, FnName, ArgKind, Preds),
    Preds = [_:_/Arity | _],
    ts_generate(Pattern, FnName, Arity, Code0),
    driver_arg_expr(ArgKind, ArgExpr),
    format(string(Driver),
'\n// UnifyWeaver JS pattern conformance driver\nconst __g: any = globalThis;\nconsole.log(~w(~w));\n',
           [FnName, ArgExpr]),
    string_concat(Code0, Driver, Code),
    ja_tmp_dir('tmp_jsct_ts', Dir),
    directory_file_path(Dir, 'prog.ts', TsFile),
    setup_call_cleanup(open(TsFile, write, S), write(S, Code), close(S)),
    ts_compiler(Exe, PreArgs),
    append(PreArgs,
           ['--strict', '--noEmitOnError', '--target', 'es2017',
            '--module', 'commonjs', 'prog.ts'],
           TscArgs),
    run_proc(Exe, TscArgs, Dir, CExit, CErr),
    ( CExit =:= 0 -> true ; throw(ts_compile_failed(CExit, CErr)) ).

ts_generate(factorial, FnName, _Arity, Code) :-
    !,
    compile_module([pred(FnName, 1, factorial)], [module_name('jsct')], Code).
ts_generate(Pattern, FnName, _Arity, Code) :-
    memberchk(Pattern, [linear_recursion, tail_recursion, list_fold]),
    compile_recursion(FnName/2, [pattern(Pattern)], Code).

ja_run(typescript, ts_ctx(Dir, ArgKind), Inputs, Value) :-
    Inputs = [Input | _],
    render_input_arg(ArgKind, Input, CliArg),
    run_proc_out(node, ['prog.js', CliArg], Dir, Exit, OutStr),
    ( Exit =:= 0
    ->  normalize_space(string(Value), OutStr)
    ;   throw(ts_run_failed(Exit, Inputs)) ).

ja_teardown(typescript, ts_ctx(Dir, _)) :- clean_dir(Dir).

% ============================================================
% Adapter: annotated_js / vanilla_js  (fuller JS pattern targets)
%   Guarded by ja_target_present/1; these arms only run if the target is
%   registered AND its module exposes compile_predicate/3. Absent here, so
%   the arms skip at the availability gate. The adapter compiles via the
%   registry module, writes prog.js, and runs it with node.
% ============================================================

ja_build(Arm, Program, node_ctx(Dir, ArgKind)) :-
    memberchk(Arm, [annotated_js, vanilla_js]),
    target_module(Arm, Module),
    js_conformance_program(Program, _Family, _Pattern, FnName, ArgKind, Preds),
    Preds = [_:Name/Arity | _],
    Module:compile_predicate(Name/Arity, [module_name(jsct), function_name(FnName)], Code0),
    driver_arg_expr_js(ArgKind, ArgExpr),
    format(string(Driver),
'\n// UnifyWeaver JS pattern conformance driver\nconsole.log(~w(~w));\n',
           [FnName, ArgExpr]),
    string_concat(Code0, Driver, Code),
    ja_tmp_dir('tmp_jsct_node', Dir),
    directory_file_path(Dir, 'prog.js', JsFile),
    setup_call_cleanup(open(JsFile, write, S), write(S, Code), close(S)).

ja_run(Arm, node_ctx(Dir, ArgKind), Inputs, Value) :-
    memberchk(Arm, [annotated_js, vanilla_js]),
    Inputs = [Input | _],
    render_input_arg(ArgKind, Input, CliArg),
    run_proc_out(node, ['prog.js', CliArg], Dir, Exit, OutStr),
    ( Exit =:= 0
    ->  normalize_space(string(Value), OutStr)
    ;   throw(node_run_failed(Exit, Inputs)) ).

ja_teardown(Arm, node_ctx(Dir, _)) :-
    memberchk(Arm, [annotated_js, vanilla_js]),
    clean_dir(Dir).

driver_arg_expr_js(numlist, 'String(process.argv[2]).split(",").map(Number)').
driver_arg_expr_js(num,     'parseInt(process.argv[2])').

% ============================================================
% Adapter: ClojureScript  (compile with the CLJS target, run under nbb)
%   nbb (node-babashka) provides the ClojureScript runtime. With nbb absent
%   the arm skips at the availability gate. Numeric programs are xfail today
%   (see ja_xfail) because the CLJS pattern recursion path is not yet clean.
% ============================================================

ja_build(clojurescript, Program, cljs_ctx(Dir)) :-
    js_conformance_program(Program, _Family, _Pattern, _FnName, _ArgKind, Preds),
    Preds = [_:Name/Arity | _],
    compile_predicate_to_clojurescript(Name/Arity, [], Code),
    ja_tmp_dir('tmp_jsct_cljs', Dir),
    directory_file_path(Dir, 'prog.cljs', CljsFile),
    setup_call_cleanup(open(CljsFile, write, S), write(S, Code), close(S)).

ja_run(clojurescript, cljs_ctx(Dir), Inputs, Value) :-
    Inputs = [Input | _],
    ( is_list(Input)
    ->  maplist([X,S]>>format(atom(S), '~w', [X]), Input, Ps),
        atomic_list_concat(Ps, ',', CliArg)
    ;   format(atom(CliArg), '~w', [Input]) ),
    run_proc_out(nbb, ['prog.cljs', CliArg], Dir, Exit, OutStr),
    ( Exit =:= 0
    ->  normalize_space(string(Value), OutStr)
    ;   throw(nbb_run_failed(Exit, Inputs)) ).

ja_teardown(clojurescript, cljs_ctx(Dir)) :- clean_dir(Dir).
