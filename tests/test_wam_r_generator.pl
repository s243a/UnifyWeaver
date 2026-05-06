:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_wam_r_generator.pl -- scaffold-level smoke tests for the
% WAM-to-R hybrid target. Modelled on test_wam_scala_generator.pl.
%
% Coverage at this stage:
%   - module loads and exports the documented entry points
%   - write_wam_r_project/3 produces the expected file layout
%   - generated R sources contain the seed atoms, instruction array,
%     label map, and predicate wrappers
%
% Running the generated R is out of scope for this file; that requires
% an R toolchain and lives in a separate test once the runtime grows
% past the scaffold subset.
%
% Usage: swipl -g run_tests -t halt tests/test_wam_r_generator.pl

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module('../src/unifyweaver/targets/wam_r_target').
:- use_module('../src/unifyweaver/targets/wam_r_lowered_emitter').
:- use_module('../src/unifyweaver/bindings/r_wam_bindings').
:- use_module('../src/unifyweaver/targets/wam_target').

:- begin_tests(wam_r_generator).

:- dynamic user:wam_r_fact/1.
:- dynamic user:wam_r_choice_fact/1.
:- dynamic user:wam_r_caller/1.
:- dynamic user:wam_r_pair/1.
:- dynamic user:wam_r_list/1.

user:wam_r_fact(a).
user:wam_r_choice_fact(a).
user:wam_r_choice_fact(b).
user:wam_r_choice_fact(c).
user:wam_r_caller(X) :- user:wam_r_fact(X).
user:wam_r_pair(f(a, b)).
user:wam_r_list([a, b, c]).

% ------------------------------------------------------------------
% Test 1: module exports the documented entry points
% ------------------------------------------------------------------
test(exports) :-
    assertion(current_predicate(wam_r_target:write_wam_r_project/3)),
    assertion(current_predicate(wam_r_target:compile_wam_predicate_to_r/4)),
    assertion(current_predicate(wam_r_target:r_foreign_predicate/3)),
    assertion(current_predicate(wam_r_target:init_r_atom_intern_table/0)).

% ------------------------------------------------------------------
% Test 2: project layout -- runtime + program + DESCRIPTION written
% ------------------------------------------------------------------
test(project_layout) :-
    once((
        unique_r_tmp_dir('tmp_r_layout', TmpDir),
        write_wam_r_project(
            [user:wam_r_fact/1],
            [module_name('wam.r.layout.test')],
            TmpDir),
        directory_file_path(TmpDir, 'DESCRIPTION', Desc),
        directory_file_path(TmpDir, 'R/wam_runtime.R', Runtime),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        assertion(exists_file(Desc)),
        assertion(exists_file(Runtime)),
        assertion(exists_file(Program)),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test 3: generated program seeds the intern table and emits the
%         instruction array body
% ------------------------------------------------------------------
test(intern_and_instructions) :-
    once((
        unique_r_tmp_dir('tmp_r_instrs', TmpDir),
        write_wam_r_project(
            [user:wam_r_fact/1],
            [],
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        assertion(sub_string(Code, _, _, _, 'intern_seed <- c(')),
        assertion(sub_string(Code, _, _, _, 'shared_instructions <- list(')),
        assertion(sub_string(Code, _, _, _, 'GetConstant(Atom(')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test 4: label map carries the predicate entry
% ------------------------------------------------------------------
test(label_map_has_pred_entry) :-
    once((
        unique_r_tmp_dir('tmp_r_labels', TmpDir),
        write_wam_r_project(
            [user:wam_r_fact/1],
            [],
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        assertion(sub_string(Code, _, _, _, '"wam_r_fact/1" =')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test 5: per-predicate wrapper functions are emitted
% ------------------------------------------------------------------
test(predicate_wrappers) :-
    once((
        unique_r_tmp_dir('tmp_r_wrappers', TmpDir),
        write_wam_r_project(
            [user:wam_r_caller/1, user:wam_r_fact/1],
            [],
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        assertion(sub_string(Code, _, _, _, 'wam_r_caller <- function(')),
        assertion(sub_string(Code, _, _, _, 'wam_r_fact <- function(')),
        assertion(sub_string(Code, _, _, _, 'WamRuntime$run_predicate(shared_program,')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test 6: choice-point instructions are emitted for multi-clause facts
% ------------------------------------------------------------------
test(choice_points_emitted) :-
    once((
        unique_r_tmp_dir('tmp_r_choice', TmpDir),
        write_wam_r_project(
            [user:wam_r_choice_fact/1],
            [],
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        assertion(sub_string(Code, _, _, _, 'TryMeElse(')),
        assertion(sub_string(Code, _, _, _, 'TrustMe()')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test: structure-bearing fact emits GetStructure with arity + Unify*
% ------------------------------------------------------------------
test(structure_instructions_emitted) :-
    once((
        unique_r_tmp_dir('tmp_r_struct', TmpDir),
        write_wam_r_project(
            [user:wam_r_pair/1],
            [],
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        % Codegen carries arity on GetStructure (3-argument constructor).
        assertion(sub_string(Code, _, _, _, 'GetStructure(')),
        assertion(sub_string(Code, _, _, _, ', 1, 2)')),
        assertion(sub_string(Code, _, _, _, 'UnifyConstant(')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test: list fact emits GetList + UnifyConstant chain
% ------------------------------------------------------------------
test(list_instructions_emitted) :-
    once((
        unique_r_tmp_dir('tmp_r_list', TmpDir),
        write_wam_r_project(
            [user:wam_r_list/1],
            [],
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        assertion(sub_string(Code, _, _, _, 'GetList(')),
        assertion(sub_string(Code, _, _, _, 'UnifyVariable(')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test: foreign-predicate body is replaced by CallForeign + Proceed
%       and the handler shows up in foreign_handlers
% ------------------------------------------------------------------
test(foreign_handler_wiring) :-
    once((
        unique_r_tmp_dir('tmp_r_foreign', TmpDir),
        % r_double/2 is foreign; it is not defined in user: at all.
        write_wam_r_project(
            [],
            [ foreign_predicates([r_double/2]),
              r_foreign_handlers([
                handler(r_double/2,
                  'function(state, args, table) {\n  n <- args[[1]]\n  if (is.null(n) || is.null(n$tag) || n$tag != "int") return(FALSE)\n  list(ok = TRUE, bindings = list(list(idx = 2L, val = IntTerm(2L * n$val))))\n}')
              ])
            ],
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        % Foreign body is the two-instruction stub: CallForeign + Proceed.
        assertion(sub_string(Code, _, _, _, 'CallForeign("r_double", 2)')),
        % Handler is registered against the "pred/arity" key.
        assertion(sub_string(Code, _, _, _, '"r_double/2" = function(')),
        % The supplied handler body is verbatim in the file.
        assertion(sub_string(Code, _, _, _, 'list(ok = TRUE, bindings = list(list(idx = 2L,')),
        % Top-level wrapper exists too.
        assertion(sub_string(Code, _, _, _, 'r_double <- function(')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test: end-to-end Rscript dispatch through a foreign handler.
% Skipped automatically when Rscript is not on PATH, so the suite
% still passes in environments without R.
% ------------------------------------------------------------------
test(foreign_handler_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_foreign_handler_via_rscript
    ;   true
    )).

e2e_foreign_handler_via_rscript :-
    HandlerSrc =
        'function(state, args, table) {\n  n <- args[[1]]\n  if (is.null(n) || is.null(n$tag) || n$tag != "int") return(FALSE)\n  list(ok = TRUE, bindings = list(list(idx = 2L, val = IntTerm(2L * n$val))))\n}',
    % Caller predicates ground both args at the call site, so the
    % foreign handler's output binding must unify with the literal
    % second arg. check_ok passes 14 (matches 2*7); check_bad passes 99
    % (does not match) -- this exercises the bindings + unify path
    % without dragging in arithmetic builtins.
    assertz((user:check_ok  :- user:r_double(7, 14))),
    assertz((user:check_bad :- user:r_double(7, 99))),
    unique_r_tmp_dir('tmp_r_foreign_e2e', TmpDir),
    write_wam_r_project(
        [user:check_ok/0, user:check_bad/0],
        [ foreign_predicates([r_double/2]),
          r_foreign_handlers([handler(r_double/2, HandlerSrc)])
        ],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    run_rscript_query(RDir, 'check_ok/0',  OkOut),
    run_rscript_query(RDir, 'check_bad/0', BadOut),
    assertion(sub_string(OkOut,  _, _, _, "true")),
    assertion(sub_string(BadOut, _, _, _, "false")),
    delete_directory_and_contents(TmpDir).

run_rscript_query(RDir, Query, Out) :-
    process_create(path('Rscript'),
                   ['generated_program.R', Query],
                   [ cwd(RDir),
                     stdout(pipe(OutStream)),
                     stderr(pipe(ErrStream)),
                     process(PID)
                   ]),
    read_string(OutStream, _, Out), close(OutStream),
    read_string(ErrStream, _, _),   close(ErrStream),
    process_wait(PID, _).

rscript_available :-
    catch((
        process_create(path('Rscript'), ['--version'],
                       [ stdout(null), stderr(null), process(PID) ]),
        process_wait(PID, exit(0))
    ), _, fail).

% ------------------------------------------------------------------
% Test: codegen emits BuiltinCall for arithmetic guards (is/2, >/2)
% ------------------------------------------------------------------
test(builtin_call_emitted) :-
    once((
        unique_r_tmp_dir('tmp_r_builtin', TmpDir),
        assertz((user:wam_r_double(X, Y) :- Y is X * 2)),
        assertz((user:wam_r_positive(X) :- X > 0)),
        write_wam_r_project(
            [user:wam_r_double/2, user:wam_r_positive/1],
            [],
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        assertion(sub_string(Code, _, _, _, 'BuiltinCall("is/2", 2)')),
        assertion(sub_string(Code, _, _, _, 'BuiltinCall(">/2", 2)')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test: end-to-end Rscript run for arithmetic builtins.
% Skipped when Rscript is not on PATH.
% ------------------------------------------------------------------
test(builtin_arith_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_builtin_arith_via_rscript
    ;   true
    )).

e2e_builtin_arith_via_rscript :-
    assertz((user:wam_r_double(X, Y)   :- Y is X * 2)),
    assertz((user:wam_r_positive(X)    :- X > 0)),
    assertz((user:wam_r_eq(X, Y)       :- X =:= Y)),
    assertz((user:wam_r_complex_ok     :- X is 3 * 4 + 1, X =:= 13)),
    assertz((user:wam_r_complex_bad    :- X is 3 * 4 + 1, X =:= 14)),
    assertz((user:wam_r_double_ok      :- wam_r_double(7, 14))),
    assertz((user:wam_r_double_bad     :- wam_r_double(7, 99))),
    assertz((user:wam_r_pos_ok         :- wam_r_positive(5))),
    assertz((user:wam_r_pos_bad        :- wam_r_positive(-3))),
    unique_r_tmp_dir('tmp_r_builtin_e2e', TmpDir),
    write_wam_r_project(
        [ user:wam_r_double/2, user:wam_r_positive/1, user:wam_r_eq/2,
          user:wam_r_complex_ok/0, user:wam_r_complex_bad/0,
          user:wam_r_double_ok/0, user:wam_r_double_bad/0,
          user:wam_r_pos_ok/0,    user:wam_r_pos_bad/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    run_rscript_query(RDir, 'wam_r_complex_ok/0',  C1),
    run_rscript_query(RDir, 'wam_r_complex_bad/0', C2),
    run_rscript_query(RDir, 'wam_r_double_ok/0',   D1),
    run_rscript_query(RDir, 'wam_r_double_bad/0',  D2),
    run_rscript_query(RDir, 'wam_r_pos_ok/0',      P1),
    run_rscript_query(RDir, 'wam_r_pos_bad/0',     P2),
    assertion(sub_string(C1, _, _, _, "true")),
    assertion(sub_string(C2, _, _, _, "false")),
    assertion(sub_string(D1, _, _, _, "true")),
    assertion(sub_string(D2, _, _, _, "false")),
    assertion(sub_string(P1, _, _, _, "true")),
    assertion(sub_string(P2, _, _, _, "false")),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% Test: lowered emitter exports + lowerability semantics (Phase 2)
% ------------------------------------------------------------------
test(lowered_emitter_phase2) :-
    assertion(current_predicate(wam_r_lowered_emitter:wam_r_lowerable/3)),
    assertion(current_predicate(wam_r_lowered_emitter:lower_predicate_to_r/4)),
    assertion(current_predicate(wam_r_lowered_emitter:r_lowered_func_name/2)),
    % Multi-clause predicates are not lowerable (need backtracking).
    compile_predicate_to_wam_string(user:wam_r_choice_fact/1, ChoiceWam),
    \+ wam_r_lowered_emitter:wam_r_lowerable(user:wam_r_choice_fact/1,
                                             ChoiceWam, _),
    % Single-clause deterministic rule is lowerable.
    compile_predicate_to_wam_string(user:wam_r_caller/1, CallerWam),
    wam_r_lowered_emitter:wam_r_lowerable(user:wam_r_caller/1,
                                          CallerWam, deterministic).

compile_predicate_to_wam_string(Pred, WamStr) :-
    wam_target:compile_predicate_to_wam(Pred, [], WamCode),
    (   string(WamCode) -> WamStr = WamCode
    ;   atom_string(WamCode, WamStr)
    ).

% ------------------------------------------------------------------
% Test: emit_mode(interpreter) keeps the wrapper on the array path and
%       emits no lowered_<...>_<n> functions.
% ------------------------------------------------------------------
test(emit_mode_interpreter_default) :-
    once((
        unique_r_tmp_dir('tmp_r_mode_interp', TmpDir),
        write_wam_r_project(
            [user:wam_r_caller/1],
            [],   % default mode = interpreter
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        assertion(sub_string(Code, _, _, _, 'WamRuntime$run_predicate(shared_program,')),
        assertion(\+ sub_string(Code, _, _, _, 'lowered_wam_r_caller_1')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test: emit_mode(functions) emits a lowered_<name>_<n> definition
%       and points the wrapper at it.
% ------------------------------------------------------------------
test(emit_mode_functions_lowers_caller) :-
    once((
        unique_r_tmp_dir('tmp_r_mode_funcs', TmpDir),
        write_wam_r_project(
            [user:wam_r_caller/1, user:wam_r_fact/1],
            [emit_mode(functions)],
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        assertion(sub_string(Code, _, _, _, 'lowered_wam_r_caller_1 <- function(program, state)')),
        assertion(sub_string(Code, _, _, _, 'isTRUE(lowered_wam_r_caller_1(shared_program, state))')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test: multi-clause predicates fall back to interpreter even under
%       emit_mode(functions).
% ------------------------------------------------------------------
test(emit_mode_functions_skips_multi_clause) :-
    once((
        unique_r_tmp_dir('tmp_r_mode_skip', TmpDir),
        write_wam_r_project(
            [user:wam_r_choice_fact/1],
            [emit_mode(functions)],
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        assertion(\+ sub_string(Code, _, _, _, 'lowered_wam_r_choice_fact_1')),
        assertion(sub_string(Code, _, _, _, 'WamRuntime$run_predicate(shared_program,')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Test: end-to-end Rscript execution of a lowered predicate.
% ------------------------------------------------------------------
test(lowered_emitter_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_lowered_via_rscript
    ;   true
    )).

e2e_lowered_via_rscript :-
    % Multi-clause fact wam_r_e2e_fact/1 stays on the array path; the
    % single-clause deterministic caller wam_r_e2e_caller/1 is lowered
    % and proves through it. We assert that calling the lowered wrapper
    % with arg = a succeeds and arg = z fails.
    assertz(user:wam_r_e2e_fact(a)),
    assertz(user:wam_r_e2e_fact(b)),
    assertz((user:wam_r_e2e_caller(X) :- user:wam_r_e2e_fact(X))),
    assertz((user:e2e_ok  :- user:wam_r_e2e_caller(a))),
    assertz((user:e2e_bad :- user:wam_r_e2e_caller(z))),
    unique_r_tmp_dir('tmp_r_lowered_e2e', TmpDir),
    write_wam_r_project(
        [ user:wam_r_e2e_fact/1, user:wam_r_e2e_caller/1,
          user:e2e_ok/0, user:e2e_bad/0 ],
        [emit_mode(functions)],
        TmpDir),
    directory_file_path(TmpDir, 'R/generated_program.R', Program),
    read_file_to_string(Program, Code, []),
    assertion(sub_string(Code, _, _, _, 'lowered_wam_r_e2e_caller_1 <- function')),
    directory_file_path(TmpDir, 'R', RDir),
    run_rscript_query(RDir, 'e2e_ok/0',  OkOut),
    run_rscript_query(RDir, 'e2e_bad/0', BadOut),
    assertion(sub_string(OkOut,  _, _, _, "true")),
    assertion(sub_string(BadOut, _, _, _, "false")),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% Test 8: r_wam bindings module loads
% ------------------------------------------------------------------
test(r_wam_bindings_loads) :-
    assertion(current_predicate(r_wam_bindings:init_r_wam_bindings/0)),
    assertion(current_predicate(r_wam_bindings:r_wam_binding/5)).

:- end_tests(wam_r_generator).

% ------------------------------------------------------------------
% Helpers
% ------------------------------------------------------------------
unique_r_tmp_dir(Prefix, Dir) :-
    get_time(Now),
    StampF is Now,
    format(atom(Dir), '/tmp/~w_~w', [Prefix, StampF]),
    (   exists_directory(Dir)
    ->  delete_directory_and_contents(Dir)
    ;   true
    ).
