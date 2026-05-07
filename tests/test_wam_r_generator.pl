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
% Test: lowered emitter exports + lowerability semantics (Phase 3)
% ------------------------------------------------------------------
test(lowered_emitter_phase3) :-
    assertion(current_predicate(wam_r_lowered_emitter:wam_r_lowerable/3)),
    assertion(current_predicate(wam_r_lowered_emitter:lower_predicate_to_r/4)),
    assertion(current_predicate(wam_r_lowered_emitter:r_lowered_func_name/2)),
    % Multi-clause predicates are now lowerable as multi_clause_1
    % (clause 1 inline + fallback to the array path on failure).
    compile_predicate_to_wam_string(user:wam_r_choice_fact/1, ChoiceWam),
    once(wam_r_lowered_emitter:wam_r_lowerable(user:wam_r_choice_fact/1,
                                               ChoiceWam, multi_clause_1)),
    % Single-clause deterministic rule is lowered with reason
    % `deterministic`.
    compile_predicate_to_wam_string(user:wam_r_caller/1, CallerWam),
    once(wam_r_lowered_emitter:wam_r_lowerable(user:wam_r_caller/1,
                                               CallerWam, deterministic)).

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
% Test: multi-clause predicates lower to a multi_clause_1 wrapper
%       (clause 1 inline; backtrack into the array path on failure).
% ------------------------------------------------------------------
test(emit_mode_functions_lowers_multi_clause) :-
    once((
        unique_r_tmp_dir('tmp_r_mode_multi', TmpDir),
        write_wam_r_project(
            [user:wam_r_choice_fact/1],
            [emit_mode(functions)],
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        % Lowered fn definition exists.
        assertion(sub_string(Code, _, _, _, 'lowered_wam_r_choice_fact_1 <- function(program, state)')),
        % multi_clause_1 marker is in the comment header.
        assertion(sub_string(Code, _, _, _, 'multi-clause; clause 1 inline, fall back to array')),
        % Clause-1 closure structure is in place.
        assertion(sub_string(Code, _, _, _, 'clause1_ok <- (function() {')),
        assertion(sub_string(Code, _, _, _, 'WamRuntime$backtrack(state)')),
        % Wrapper points at the lowered fn.
        assertion(sub_string(Code, _, _, _, 'isTRUE(lowered_wam_r_choice_fact_1(shared_program, state))')),
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
% Phase 3: native per-instruction inlining replaces step delegation
% for the simple register ops (allocate, deallocate, put_constant,
% put_variable, put_value, get_variable). Verify the codegen actually
% emits the inline form and stops emitting WamRuntime$step for them.
% ------------------------------------------------------------------
test(phase3_inlines_simple_ops) :-
    once((
        % p3_dual(X) :- p3_a(X), p3_b(X). forces variable preservation
        % across two calls, so the WAM emits get_variable + put_value
        % rather than optimizing them away. p3_a/p3_b are stubs solely
        % to give the call sites a target.
        assertz(user:p3_a(_)),
        assertz(user:p3_b(_)),
        assertz((user:p3_dual(X) :- user:p3_a(X), user:p3_b(X))),
        unique_r_tmp_dir('tmp_r_inline', TmpDir),
        write_wam_r_project(
            [user:p3_dual/1, user:p3_a/1, user:p3_b/1],
            [emit_mode(functions)],
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        % Inline put_reg/get_reg replaces step delegation for put_value
        % and get_variable.
        assertion(sub_string(Code, _, _, _, 'WamRuntime$put_reg(state, 201, WamRuntime$get_reg(state, 1))')),
        assertion(sub_string(Code, _, _, _, 'WamRuntime$put_reg(state, 1, WamRuntime$get_reg(state, 201))')),
        % Allocate inlines as a direct stack push.
        assertion(sub_string(Code, _, _, _, 'state$stack <- c(state$stack, list(list(cp = state$cp')),
        % And we no longer see the step-dispatch wrapper for these ops.
        assertion(\+ sub_string(Code, _, _, _, 'WamRuntime$step(program, state, Allocate())')),
        assertion(\+ sub_string(Code, _, _, _, 'WamRuntime$step(program, state, GetVariable')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% Phase 3 e2e: multi-clause lowered fact + backtracking. Clause 1
% matches `a`; for `b` and `c` we should backtrack into clause 2/3.
% ------------------------------------------------------------------
test(phase3_multi_clause_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_phase3_multi_clause
    ;   true
    )).

e2e_phase3_multi_clause :-
    assertz(user:p3_fact(a)),
    assertz(user:p3_fact(b)),
    assertz(user:p3_fact(c)),
    assertz((user:p3_a :- user:p3_fact(a))),
    assertz((user:p3_b :- user:p3_fact(b))),
    assertz((user:p3_c :- user:p3_fact(c))),
    assertz((user:p3_z :- user:p3_fact(z))),
    unique_r_tmp_dir('tmp_r_phase3_e2e', TmpDir),
    write_wam_r_project(
        [ user:p3_fact/1,
          user:p3_a/0, user:p3_b/0, user:p3_c/0, user:p3_z/0 ],
        [emit_mode(functions)],
        TmpDir),
    directory_file_path(TmpDir, 'R/generated_program.R', Program),
    read_file_to_string(Program, Code, []),
    % p3_fact/1 should be lowered as multi_clause_1.
    assertion(sub_string(Code, _, _, _, 'lowered_p3_fact_1 <- function(program, state)')),
    directory_file_path(TmpDir, 'R', RDir),
    run_rscript_query(RDir, 'p3_a/0', A),  % matches clause 1
    run_rscript_query(RDir, 'p3_b/0', B),  % must backtrack to clause 2
    run_rscript_query(RDir, 'p3_c/0', C),  % must backtrack to clause 3
    run_rscript_query(RDir, 'p3_z/0', Z),  % all three clauses fail
    assertion(sub_string(A, _, _, _, "true")),
    assertion(sub_string(B, _, _, _, "true")),
    assertion(sub_string(C, _, _, _, "true")),
    assertion(sub_string(Z, _, _, _, "false")),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% Phase-3 follow-up: extended builtins.
% Type checks, term equality / identity, standard order of terms,
% and the cut (!/0). Each predicate compiles to the corresponding
% builtin_call literal; here we verify the codegen emits them.
% ------------------------------------------------------------------
test(extended_builtins_emitted) :-
    once((
        unique_r_tmp_dir('tmp_r_extbi', TmpDir),
        % `atomic/1` and `ground/1` are *runtime-supported* but the WAM
        % compiler emits them as plain Execute, not BuiltinCall, so we
        % can't structurally assert them in the generated program.
        assertz((user:bi_atom(X)        :- atom(X))),
        assertz((user:bi_integer(X)     :- integer(X))),
        assertz((user:bi_var(X)         :- var(X))),
        assertz((user:bi_is_list(X)     :- is_list(X))),
        assertz((user:bi_eq(X, Y)       :- X = Y)),
        assertz((user:bi_neq(X, Y)      :- X \= Y)),
        assertz((user:bi_id(X, Y)       :- X == Y)),
        assertz((user:bi_nid(X, Y)      :- X \== Y)),
        assertz((user:bi_ord(X, Y)      :- X @< Y)),
        assertz((user:bi_cut            :- !)),
        write_wam_r_project(
            [ user:bi_atom/1, user:bi_integer/1, user:bi_var/1,
              user:bi_is_list/1,
              user:bi_eq/2, user:bi_neq/2, user:bi_id/2, user:bi_nid/2,
              user:bi_ord/2, user:bi_cut/0 ],
            [],
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        assertion(sub_string(Code, _, _, _, 'BuiltinCall("atom/1", 1)')),
        assertion(sub_string(Code, _, _, _, 'BuiltinCall("integer/1", 1)')),
        assertion(sub_string(Code, _, _, _, 'BuiltinCall("var/1", 1)')),
        assertion(sub_string(Code, _, _, _, 'BuiltinCall("is_list/1", 1)')),
        assertion(sub_string(Code, _, _, _, 'BuiltinCall("=/2", 2)')),
        assertion(sub_string(Code, _, _, _, 'BuiltinCall("==/2", 2)')),
        assertion(sub_string(Code, _, _, _, 'BuiltinCall("@</2", 2)')),
        assertion(sub_string(Code, _, _, _, 'BuiltinCall("!/0", 0)')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% End-to-end Rscript run for the extended builtins. Runs a battery of
% predicates through Rscript and asserts the truth pattern. Auto-skips
% when Rscript is not on PATH.
% ------------------------------------------------------------------
test(extended_builtins_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_extended_builtins_via_rscript
    ;   true
    )).

e2e_extended_builtins_via_rscript :-
    % atomic/ground are runtime-supported but compile to Execute, so
    % we leave them out of the e2e battery (they would dispatch as
    % missing predicates). Coverage is across the actually-emitted
    % builtin family.
    assertz((user:bi_atom_y     :- atom(a))),
    assertz((user:bi_atom_n     :- atom(7))),
    assertz((user:bi_int_y      :- integer(7))),
    assertz((user:bi_int_n      :- integer(a))),
    assertz((user:bi_num_y      :- number(7))),
    assertz((user:bi_num_n      :- number(a))),
    assertz((user:bi_compound_y :- compound(f(a)))),
    assertz((user:bi_compound_n :- compound(a))),
    assertz((user:bi_is_list_y  :- is_list([a, b, c]))),
    assertz((user:bi_is_list_e  :- is_list([]))),
    assertz((user:bi_is_list_n  :- is_list(f(a)))),
    % Term equality / identity.
    assertz((user:bi_eq_y       :- a = a)),
    assertz((user:bi_eq_n       :- a = b)),
    assertz((user:bi_neq_y      :- a \= b)),
    assertz((user:bi_neq_n      :- a \= a)),
    assertz((user:bi_id_y       :- a == a)),
    assertz((user:bi_id_n       :- a == b)),
    assertz((user:bi_nid_y      :- a \== b)),
    assertz((user:bi_nid_n      :- a \== a)),
    % Standard order of terms.
    assertz((user:bi_lt_y       :- a @< b)),
    assertz((user:bi_lt_n       :- b @< a)),
    assertz((user:bi_le_y       :- a @=< a)),
    assertz((user:bi_gt_y       :- b @> a)),
    % Cut: succeeds (it's just `:- !`).
    assertz((user:bi_cut_y      :- !)),
    unique_r_tmp_dir('tmp_r_extbi_e2e', TmpDir),
    write_wam_r_project(
        [ user:bi_atom_y/0, user:bi_atom_n/0,
          user:bi_int_y/0, user:bi_int_n/0,
          user:bi_num_y/0, user:bi_num_n/0,
          user:bi_compound_y/0, user:bi_compound_n/0,
          user:bi_is_list_y/0, user:bi_is_list_e/0, user:bi_is_list_n/0,
          user:bi_eq_y/0, user:bi_eq_n/0,
          user:bi_neq_y/0, user:bi_neq_n/0,
          user:bi_id_y/0, user:bi_id_n/0,
          user:bi_nid_y/0, user:bi_nid_n/0,
          user:bi_lt_y/0, user:bi_lt_n/0, user:bi_le_y/0, user:bi_gt_y/0,
          user:bi_cut_y/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [bi_atom_y, bi_int_y, bi_num_y,
           bi_compound_y, bi_is_list_y, bi_is_list_e,
           bi_eq_y, bi_neq_y, bi_id_y, bi_nid_y,
           bi_lt_y, bi_le_y, bi_gt_y, bi_cut_y],
    No  = [bi_atom_n, bi_int_n, bi_num_n, bi_compound_n,
           bi_is_list_n, bi_eq_n, bi_neq_n, bi_id_n, bi_nid_n, bi_lt_n],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    forall(member(P, No), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "false"))
    )),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% Term inspection: functor/3, arg/3, =../2 emit BuiltinCall literals.
% ------------------------------------------------------------------
test(term_inspection_emitted) :-
    once((
        unique_r_tmp_dir('tmp_r_terminsp', TmpDir),
        assertz((user:ti_funct(T, F, A) :- functor(T, F, A))),
        assertz((user:ti_arg(N, T, A)   :- arg(N, T, A))),
        assertz((user:ti_univ(T, L)     :- T =.. L)),
        write_wam_r_project(
            [user:ti_funct/3, user:ti_arg/3, user:ti_univ/2],
            [],
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        assertion(sub_string(Code, _, _, _, 'BuiltinCall("functor/3", 3)')),
        assertion(sub_string(Code, _, _, _, 'BuiltinCall("arg/3", 3)')),
        assertion(sub_string(Code, _, _, _, 'BuiltinCall("=../2", 2)')),
        delete_directory_and_contents(TmpDir)
    )).

% ------------------------------------------------------------------
% End-to-end Rscript run for term inspection. Covers:
%   - functor/3 decompose for struct, atom, integer
%   - functor/3 construct (fresh var args)
%   - arg/3 extraction
%   - =../2 decompose for struct, atom
%   - =../2 construct from a list
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(term_inspection_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_term_inspection_via_rscript
    ;   true
    )).

e2e_term_inspection_via_rscript :-
    % Decompose: pull functor name + arity out of a struct.
    assertz((user:ti_dec_struct  :- functor(f(a, b), f, 2))),
    assertz((user:ti_dec_struct_n :- functor(f(a, b), g, 2))),
    assertz((user:ti_dec_atom    :- functor(hello, hello, 0))),
    assertz((user:ti_dec_int     :- functor(7, 7, 0))),
    % Construct: build a fresh struct from name + arity.
    assertz((user:ti_con_struct :- functor(T, f, 2), arg(1, T, _), arg(2, T, _))),
    assertz((user:ti_con_atom   :- functor(T, hello, 0), T == hello)),
    % arg/3 indexing (1-based).
    assertz((user:ti_arg_first  :- arg(1, f(a, b, c), a))),
    assertz((user:ti_arg_third  :- arg(3, f(a, b, c), c))),
    assertz((user:ti_arg_oob    :- arg(4, f(a, b, c), _))),
    % =../2 decompose: f(a,b) =.. [f, a, b].
    assertz((user:ti_univ_dec   :- f(a, b) =.. [f, a, b])),
    assertz((user:ti_univ_dec_n :- f(a, b) =.. [g, a, b])),
    assertz((user:ti_univ_atom  :- hello =.. [hello])),
    % =../2 construct: T =.. [f, a, b]  ->  T = f(a, b).
    assertz((user:ti_univ_con   :- T =.. [f, a, b], T == f(a, b))),
    assertz((user:ti_univ_con_a :- T =.. [hello],   T == hello)),
    unique_r_tmp_dir('tmp_r_terminsp_e2e', TmpDir),
    write_wam_r_project(
        [ user:ti_dec_struct/0, user:ti_dec_struct_n/0,
          user:ti_dec_atom/0, user:ti_dec_int/0,
          user:ti_con_struct/0, user:ti_con_atom/0,
          user:ti_arg_first/0, user:ti_arg_third/0, user:ti_arg_oob/0,
          user:ti_univ_dec/0, user:ti_univ_dec_n/0, user:ti_univ_atom/0,
          user:ti_univ_con/0, user:ti_univ_con_a/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [ti_dec_struct, ti_dec_atom, ti_dec_int,
           ti_con_struct, ti_con_atom,
           ti_arg_first, ti_arg_third,
           ti_univ_dec, ti_univ_atom,
           ti_univ_con, ti_univ_con_a],
    No  = [ti_dec_struct_n, ti_arg_oob, ti_univ_dec_n],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    forall(member(P, No), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "false"))
    )),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end: list / atom library builtins.
% length/2 and append/3 dispatch via builtin_call; atom_codes/2,
% atom_chars/2, atom_length/2, atom_concat/3, reverse/2 and last/2
% dispatch via the runtime's call_library fallback when the WAM-emitted
% Execute / Call hits a missing label. Auto-skips when Rscript is not
% on PATH.
% ------------------------------------------------------------------
test(list_atom_builtins_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_list_atom_builtins_via_rscript
    ;   true
    )).

e2e_list_atom_builtins_via_rscript :-
    % length/2 (both deterministic modes).
    assertz((user:lb_len_count :- length([a, b, c], 3))),
    assertz((user:lb_len_zero  :- length([], 0))),
    assertz((user:lb_len_wrong :- length([a, b, c], 4))),
    assertz((user:lb_len_build :- length(L, 3), is_list(L))),
    % append/3 deterministic mode.
    assertz((user:lb_app_ok    :- append([1, 2], [3, 4], [1, 2, 3, 4]))),
    assertz((user:lb_app_empty :- append([], [a, b], [a, b]))),
    assertz((user:lb_app_no    :- append([1, 2], [3, 4], [1, 2, 3, 5]))),
    % reverse/2.
    assertz((user:lb_rev_ok :- reverse([a, b, c], [c, b, a]))),
    assertz((user:lb_rev_no :- reverse([a, b, c], [a, b, c]))),
    % last/2.
    assertz((user:lb_last_ok  :- last([a, b, c], c))),
    assertz((user:lb_last_no  :- last([a, b, c], a))),
    % atom_length/2 / atom_codes/2 / atom_chars/2 / atom_concat/3.
    assertz((user:lb_alen_ok    :- atom_length(hello, 5))),
    assertz((user:lb_alen_no    :- atom_length(hello, 4))),
    assertz((user:lb_acodes_ok  :- atom_codes(hi, [104, 105]))),
    assertz((user:lb_achars_ok  :- atom_chars(hi, [h, i]))),
    assertz((user:lb_aconcat_ok :- atom_concat(hello, world, helloworld))),
    assertz((user:lb_aconcat_no :- atom_concat(hello, world, hellounlikely))),
    unique_r_tmp_dir('tmp_r_listatom_e2e', TmpDir),
    write_wam_r_project(
        [ user:lb_len_count/0, user:lb_len_zero/0, user:lb_len_wrong/0,
          user:lb_len_build/0,
          user:lb_app_ok/0, user:lb_app_empty/0, user:lb_app_no/0,
          user:lb_rev_ok/0, user:lb_rev_no/0,
          user:lb_last_ok/0, user:lb_last_no/0,
          user:lb_alen_ok/0, user:lb_alen_no/0,
          user:lb_acodes_ok/0, user:lb_achars_ok/0,
          user:lb_aconcat_ok/0, user:lb_aconcat_no/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [lb_len_count, lb_len_zero, lb_len_build,
           lb_app_ok, lb_app_empty,
           lb_rev_ok, lb_last_ok,
           lb_alen_ok, lb_acodes_ok, lb_achars_ok, lb_aconcat_ok],
    No  = [lb_len_wrong, lb_app_no, lb_rev_no, lb_last_no,
           lb_alen_no, lb_aconcat_no],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    forall(member(P, No), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "false"))
    )),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for extended arithmetic in is/2.
% Covers atom constants (pi, e), unary transcendentals (sqrt, exp, log,
% sin, cos), rounding family (floor, ceiling, truncate, round, sign),
% and binary ops (^, **, atan2, gcd). Auto-skips when Rscript is not
% on PATH.
% ------------------------------------------------------------------
test(extended_arith_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_extended_arith_via_rscript
    ;   true
    )).

e2e_extended_arith_via_rscript :-
    % Atom constants.
    assertz((user:ar_pi      :- X is pi,         X > 3.14, X < 3.15)),
    assertz((user:ar_e       :- X is e,          X > 2.71, X < 2.72)),
    % Unary transcendentals.
    assertz((user:ar_sqrt    :- X is sqrt(16),   X =:= 4)),
    assertz((user:ar_sqrt_f  :- X is sqrt(2),    X > 1.4, X < 1.5)),
    assertz((user:ar_exp     :- X is exp(0),     X =:= 1)),
    assertz((user:ar_log     :- X is log(1),     X =:= 0)),
    assertz((user:ar_sin     :- X is sin(0),     X =:= 0)),
    assertz((user:ar_cos     :- X is cos(0),     X =:= 1)),
    % Rounding family.
    assertz((user:ar_floor   :- X is floor(3.7),    X =:= 3)),
    assertz((user:ar_ceil    :- X is ceiling(3.2),  X =:= 4)),
    assertz((user:ar_trunc   :- X is truncate(3.9), X =:= 3)),
    assertz((user:ar_round_d :- X is round(3.5),    X =:= 4)),
    assertz((user:ar_round_u :- X is round(3.4),    X =:= 3)),
    assertz((user:ar_sign_p  :- X is sign(7),       X =:= 1)),
    assertz((user:ar_sign_n  :- X is sign(-7),      X =:= -1)),
    assertz((user:ar_sign_z  :- X is sign(0),       X =:= 0)),
    assertz((user:ar_abs     :- X is abs(-12),      X =:= 12)),
    % Binary.
    assertz((user:ar_pow     :- X is 2^10,           X =:= 1024)),
    assertz((user:ar_starpow :- X is 2**10,          X =:= 1024)),
    assertz((user:ar_atan2   :- X is atan2(1, 1),    X > 0.78, X < 0.79)),
    assertz((user:ar_gcd     :- X is gcd(12, 8),     X =:= 4)),
    assertz((user:ar_gcd2    :- X is gcd(100, 75),   X =:= 25)),
    % Failure cases.
    assertz((user:ar_unknown :- X is bogus(1))),
    unique_r_tmp_dir('tmp_r_extarith_e2e', TmpDir),
    write_wam_r_project(
        [ user:ar_pi/0, user:ar_e/0,
          user:ar_sqrt/0, user:ar_sqrt_f/0, user:ar_exp/0, user:ar_log/0,
          user:ar_sin/0, user:ar_cos/0,
          user:ar_floor/0, user:ar_ceil/0, user:ar_trunc/0,
          user:ar_round_d/0, user:ar_round_u/0,
          user:ar_sign_p/0, user:ar_sign_n/0, user:ar_sign_z/0, user:ar_abs/0,
          user:ar_pow/0, user:ar_starpow/0, user:ar_atan2/0,
          user:ar_gcd/0, user:ar_gcd2/0,
          user:ar_unknown/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [ar_pi, ar_e,
           ar_sqrt, ar_sqrt_f, ar_exp, ar_log, ar_sin, ar_cos,
           ar_floor, ar_ceil, ar_trunc, ar_round_d, ar_round_u,
           ar_sign_p, ar_sign_n, ar_sign_z, ar_abs,
           ar_pow, ar_starpow, ar_atan2, ar_gcd, ar_gcd2],
    No  = [ar_unknown],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    forall(member(P, No), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "false"))
    )),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for bitwise arithmetic, copy_term/2,
% number<->atom conversions, and msort/2 / sort/2.
% ------------------------------------------------------------------
test(stdlib_round4_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_stdlib_round4_via_rscript
    ;   true
    )).

e2e_stdlib_round4_via_rscript :-
    % Bitwise arithmetic.
    assertz((user:bw_and    :- X is 12 /\ 10, X =:= 8)),
    assertz((user:bw_or     :- X is 12 \/ 10, X =:= 14)),
    assertz((user:bw_xor    :- X is 12 xor 10, X =:= 6)),
    assertz((user:bw_shl    :- X is 1 << 4,   X =:= 16)),
    assertz((user:bw_shr    :- X is 32 >> 2,  X =:= 8)),
    assertz((user:bw_chain  :- X is (255 /\ 0xF0) >> 4, X =:= 15)),
    % copy_term/2.
    assertz((user:ct_atom   :- copy_term(hello, hello))),
    assertz((user:ct_struct :- copy_term(f(a, b), f(a, b)))),
    % copy_term should produce a structurally-equal but variable-fresh copy.
    assertz((user:ct_fresh  :- copy_term(f(X, X), f(Y, Z)), Y == Z)),
    assertz((user:ct_indep  :- copy_term(f(X), f(Y)),
                              X = same, Y \== same)),
    % Numeric <-> atom conversions.
    % Note: WAM text loses the atom-vs-number quoting distinction (the
    % literal `'42'` and the integer `42` both serialize as a bare 42),
    % so tests for the *atom* side of these conversions are written as
    % round-trips through atom_codes / number_codes rather than using
    % atom-of-digits literals.
    assertz((user:nc_dec     :- number_codes(123, [49, 50, 51]))),
    assertz((user:nc_con     :- number_codes(N, [52, 50]), N =:= 42)),
    assertz((user:nch_round  :- number_chars(123, Cs), number_chars(N, Cs), N =:= 123)),
    assertz((user:an_round   :- atom_number(A, 42), atom_codes(A, [52, 50]))),
    assertz((user:an_to_num  :- atom_codes(AS, [51, 46, 49, 52]),  % "3.14"
                                atom_number(AS, N), N > 3.13, N < 3.15)),
    assertz((user:an_no      :- atom_codes(AS, [97, 98, 99]),       % "abc"
                                atom_number(AS, _))),
    % msort and sort.
    assertz((user:ms_keep   :- msort([3, 1, 2, 1], [1, 1, 2, 3]))),
    assertz((user:so_dedup  :- sort([3, 1, 2, 1, 3], [1, 2, 3]))),
    assertz((user:so_atoms  :- sort([c, a, b, a], [a, b, c]))),
    assertz((user:so_mixed  :- sort([2, a, 1, b], [1, 2, a, b]))),
    unique_r_tmp_dir('tmp_r_round4_e2e', TmpDir),
    write_wam_r_project(
        [ user:bw_and/0, user:bw_or/0, user:bw_xor/0,
          user:bw_shl/0, user:bw_shr/0, user:bw_chain/0,
          user:ct_atom/0, user:ct_struct/0, user:ct_fresh/0, user:ct_indep/0,
          user:nc_dec/0, user:nc_con/0,
          user:nch_round/0,
          user:an_round/0, user:an_to_num/0, user:an_no/0,
          user:ms_keep/0,
          user:so_dedup/0, user:so_atoms/0, user:so_mixed/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [bw_and, bw_or, bw_xor, bw_shl, bw_shr, bw_chain,
           ct_atom, ct_struct, ct_fresh, ct_indep,
           nc_dec, nc_con, nch_round,
           an_round, an_to_num,
           ms_keep, so_dedup, so_atoms, so_mixed],
    No  = [an_no],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    forall(member(P, No), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "false"))
    )),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for negation-as-failure (\+/1, not/1) and
% meta-call (call/1). Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(negation_meta_call_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_negation_meta_call_via_rscript
    ;   true
    )).

e2e_negation_meta_call_via_rscript :-
    % Retract any leftover clauses from prior plunit runs so multi-clause
    % accumulation doesn't change WAM dispatch shape between runs.
    forall(member(P/A, [
        nm_fact/1, nm_neg_builtin/0, nm_neg_member/0, nm_neg_user/0,
        nm_neg_no/0, nm_neg_no2/0, nm_not_builtin/0,
        nm_call_atom/0, nm_call_fail/0, nm_call_user/0, nm_call_user_n/0,
        nm_call_lib/0, nm_call_arith/0,
        nm_dneg/0, nm_dneg_no/0,
        nm_helper_succ/1, nm_helper_fail/1,
        nm_neg_succ_helper/0, nm_neg_fail_helper/0
    ]), (
        functor(H, P, A),
        catch(retractall(user:H), _, true)
    )),
    % User predicates that the goal evaluator will dispatch via labels.
    assertz(user:nm_fact(a)),
    assertz(user:nm_fact(b)),
    % \+/1 against builtin / library / user-defined predicates.
    assertz((user:nm_neg_builtin :- \+ atom(7))),       % atom(7) fails -> \+ succeeds
    assertz((user:nm_neg_member  :- \+ member(z, [a, b, c]))),
    assertz((user:nm_neg_user    :- \+ nm_fact(z))),
    assertz((user:nm_neg_no      :- \+ atom(a))),       % atom(a) succeeds -> \+ fails
    assertz((user:nm_neg_no2     :- \+ member(b, [a, b, c]))),
    % not/1 alias.
    assertz((user:nm_not_builtin :- not(atom(7)))),
    % call/1 with various callable shapes.
    assertz((user:nm_call_atom   :- call(true))),
    assertz((user:nm_call_fail   :- call(fail))),
    assertz((user:nm_call_user   :- call(nm_fact(a)))),
    assertz((user:nm_call_user_n :- call(nm_fact(z)))),
    assertz((user:nm_call_lib    :- call(member(b, [a, b, c])))),
    assertz((user:nm_call_arith  :- call(>(5, 3)))),
    % Double negation -- a classic ground-check idiom.
    assertz((user:nm_dneg        :- \+ \+ member(a, [a, b, c]))),
    assertz((user:nm_dneg_no     :- \+ \+ member(z, [a, b, c]))),
    % \+ rolls back bindings made by the goal: after `\+ X = bound`,
    % X is still unbound. Test the round-trip via a helper so the
    % conjunction has something useful to compare against. Here the
    % helper succeeds (its goal `b == bound` succeeds against atom b),
    % `\+ helper` fails, the whole conjunction fails.
    assertz((user:nm_helper_succ(X) :- X == bound)),
    assertz((user:nm_helper_fail(X) :- X == other)),
    % \+ helper_succ(bound) -> helper succeeds -> \+ fails -> conjunction fails.
    assertz((user:nm_neg_succ_helper :- \+ user:nm_helper_succ(bound))),
    % \+ helper_fail(bound) -> helper fails -> \+ succeeds.
    assertz((user:nm_neg_fail_helper :- \+ user:nm_helper_fail(bound))),
    unique_r_tmp_dir('tmp_r_negmeta_e2e', TmpDir),
    write_wam_r_project(
        [ user:nm_fact/1,
          user:nm_neg_builtin/0, user:nm_neg_member/0, user:nm_neg_user/0,
          user:nm_neg_no/0, user:nm_neg_no2/0,
          user:nm_not_builtin/0,
          user:nm_call_atom/0, user:nm_call_fail/0,
          user:nm_call_user/0, user:nm_call_user_n/0,
          user:nm_call_lib/0, user:nm_call_arith/0,
          user:nm_dneg/0, user:nm_dneg_no/0,
          user:nm_helper_succ/1, user:nm_helper_fail/1,
          user:nm_neg_succ_helper/0, user:nm_neg_fail_helper/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [nm_neg_builtin, nm_neg_member, nm_neg_user,
           nm_not_builtin,
           nm_call_atom, nm_call_user, nm_call_lib, nm_call_arith,
           nm_dneg,
           nm_neg_fail_helper],
    No  = [nm_neg_no, nm_neg_no2,
           nm_call_fail, nm_call_user_n,
           nm_dneg_no,
           nm_neg_succ_helper],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    forall(member(P, No), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "false"))
    )),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for higher-order builtins (maplist/2,
% maplist/3) and the deterministic list/utility family (compare/3,
% nth0/3, nth1/3, select/3, delete/3, succ/2).
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(higherorder_listutil_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_higherorder_listutil_via_rscript
    ;   true
    )).

e2e_higherorder_listutil_via_rscript :-
    % maplist/2: pure type checks and predicates with one trailing arg.
    assertz((user:hl_maplist_int    :- maplist(integer, [1, 2, 3]))),
    assertz((user:hl_maplist_int_n  :- maplist(integer, [1, a, 3]))),
    assertz((user:hl_maplist_atom   :- maplist(atom, [a, b, c]))),
    assertz((user:hl_maplist_empty  :- maplist(integer, []))),
    % maplist/3: lockstep transformation. succ/2 is itself in
    % call_library, so this exercises both.
    assertz((user:hl_maplist3_succ  :- maplist(succ, [1, 2, 3], [2, 3, 4]))),
    assertz((user:hl_maplist3_n     :- maplist(succ, [1, 2, 3], [2, 9, 4]))),
    assertz((user:hl_maplist3_build :- maplist(succ, [10, 20, 30], Xs),
                                       Xs = [11, 21, 31])),
    % compare/3.
    assertz((user:hl_cmp_lt :- compare(Order, 1, 2), Order == '<')),
    assertz((user:hl_cmp_eq :- compare(Order, foo, foo), Order == '=')),
    assertz((user:hl_cmp_gt :- compare(Order, b, a), Order == '>')),
    % nth0 / nth1.
    assertz((user:hl_nth0_first :- nth0(0, [a, b, c], a))),
    assertz((user:hl_nth0_last  :- nth0(2, [a, b, c], c))),
    assertz((user:hl_nth0_oob   :- nth0(3, [a, b, c], _))),
    assertz((user:hl_nth1_first :- nth1(1, [a, b, c], a))),
    assertz((user:hl_nth1_oob   :- nth1(0, [a, b, c], _))),
    % select/3.
    assertz((user:hl_select_mid :- select(b, [a, b, c], [a, c]))),
    assertz((user:hl_select_no  :- select(z, [a, b, c], _))),
    assertz((user:hl_select_one :- select(only, [only], []))),
    % delete/3 (==/2 semantics, no var binding).
    assertz((user:hl_del_all    :- delete([a, b, a, c, a], a, [b, c]))),
    assertz((user:hl_del_none   :- delete([a, b, c], z, [a, b, c]))),
    % succ/2 in both directions.
    assertz((user:hl_succ_fwd   :- succ(3, 4))),
    assertz((user:hl_succ_back  :- succ(X, 5), X =:= 4)),
    assertz((user:hl_succ_zero  :- succ(0, 1))),
    assertz((user:hl_succ_neg   :- succ(_, 0))),
    unique_r_tmp_dir('tmp_r_holu_e2e', TmpDir),
    write_wam_r_project(
        [ user:hl_maplist_int/0, user:hl_maplist_int_n/0,
          user:hl_maplist_atom/0, user:hl_maplist_empty/0,
          user:hl_maplist3_succ/0, user:hl_maplist3_n/0, user:hl_maplist3_build/0,
          user:hl_cmp_lt/0, user:hl_cmp_eq/0, user:hl_cmp_gt/0,
          user:hl_nth0_first/0, user:hl_nth0_last/0, user:hl_nth0_oob/0,
          user:hl_nth1_first/0, user:hl_nth1_oob/0,
          user:hl_select_mid/0, user:hl_select_no/0, user:hl_select_one/0,
          user:hl_del_all/0, user:hl_del_none/0,
          user:hl_succ_fwd/0, user:hl_succ_back/0, user:hl_succ_zero/0,
          user:hl_succ_neg/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [hl_maplist_int, hl_maplist_atom, hl_maplist_empty,
           hl_maplist3_succ, hl_maplist3_build,
           hl_cmp_lt, hl_cmp_eq, hl_cmp_gt,
           hl_nth0_first, hl_nth0_last, hl_nth1_first,
           hl_select_mid, hl_select_one,
           hl_del_all, hl_del_none,
           hl_succ_fwd, hl_succ_back, hl_succ_zero],
    No  = [hl_maplist_int_n, hl_maplist3_n,
           hl_nth0_oob, hl_nth1_oob,
           hl_select_no,
           hl_succ_neg],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    forall(member(P, No), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "false"))
    )),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for I/O builtins (write/1, nl/0, writeln/1,
% print/1, format/1, format/2) and between/3 (deterministic mode).
% Asserts both the truth result AND the captured stdout.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(io_between_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_io_between_via_rscript
    ;   true
    )).

e2e_io_between_via_rscript :-
    assertz((user:io_w_basic   :- write(hello), nl)),
    assertz((user:io_w_int     :- write(42), nl)),
    assertz((user:io_writeln_a :- writeln(world))),
    assertz((user:io_print_x   :- print(x), nl)),
    assertz((user:io_format1   :- format("plain~n"))),
    assertz((user:io_format2   :- format("~w + ~w = ~w~n", [2, 3, 5]))),
    assertz((user:io_format_list :- format("list = ~w~n", [[a, b, c]]))),
    assertz((user:io_format_struct :- format("term = ~w~n", [f(1, two)]))),
    assertz((user:io_format_tilde  :- format("100~~ done~n"))),
    % between/3
    assertz((user:bt_in    :- between(1, 10, 5))),
    assertz((user:bt_lo    :- between(5, 5, 5))),
    assertz((user:bt_oob_h :- between(1, 10, 99))),
    assertz((user:bt_oob_l :- between(1, 10, 0))),
    assertz((user:bt_bad   :- between(10, 1, 5))),
    assertz((user:bt_gen   :- between(3, 7, X), X =:= 3)),
    unique_r_tmp_dir('tmp_r_io_e2e', TmpDir),
    write_wam_r_project(
        [ user:io_w_basic/0, user:io_w_int/0, user:io_writeln_a/0,
          user:io_print_x/0, user:io_format1/0, user:io_format2/0,
          user:io_format_list/0, user:io_format_struct/0,
          user:io_format_tilde/0,
          user:bt_in/0, user:bt_lo/0, user:bt_oob_h/0, user:bt_oob_l/0,
          user:bt_bad/0, user:bt_gen/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    % Truth-only checks (output content irrelevant for these).
    Yes = [io_w_basic, io_w_int, io_writeln_a, io_print_x,
           io_format1, io_format2, io_format_list, io_format_struct,
           io_format_tilde,
           bt_in, bt_lo, bt_gen],
    No  = [bt_oob_h, bt_oob_l, bt_bad],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    forall(member(P, No), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "false"))
    )),
    % Output-content checks for the I/O family.
    run_rscript_query(RDir, 'io_w_basic/0',       OB),
    assertion(sub_string(OB, _, _, _, "hello\n")),
    run_rscript_query(RDir, 'io_format2/0',       OF2),
    assertion(sub_string(OF2, _, _, _, "2 + 3 = 5\n")),
    run_rscript_query(RDir, 'io_format_list/0',   OFL),
    assertion(sub_string(OFL, _, _, _, "list = [a,b,c]\n")),
    run_rscript_query(RDir, 'io_format_struct/0', OFS),
    assertion(sub_string(OFS, _, _, _, "term = f(1,two)\n")),
    run_rscript_query(RDir, 'io_format_tilde/0',  OFT),
    assertion(sub_string(OFT, _, _, _, "100~ done\n")),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for the string family (string_concat,
% string_length, atom_string, string_to_atom, string_chars,
% string_codes, string_upper, string_lower, string_code,
% split_string). The codegen collapses strings into atoms (the
% WAM-text serialisation doesn't distinguish them), so the string_*
% predicates either alias onto their atom_* counterpart or are
% implemented fresh.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(string_ops_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_string_ops_via_rscript
    ;   true
    )).

e2e_string_ops_via_rscript :-
    assertz((user:str_concat   :- string_concat("foo", "bar", "foobar"))),
    assertz((user:str_concat_n :- string_concat("foo", "bar", "fooXbar"))),
    assertz((user:str_length   :- string_length("hello", 5))),
    assertz((user:str_length_n :- string_length("hello", 4))),
    assertz((user:str_atom     :- atom_string(hello, "hello"))),
    assertz((user:str_to_atom  :- string_to_atom("hello", hello))),
    assertz((user:str_chars    :- string_chars("hi", [h, i]))),
    assertz((user:str_codes    :- string_codes("AB", [65, 66]))),
    assertz((user:str_upper    :- string_upper("abc", "ABC"))),
    assertz((user:str_lower    :- string_lower("ABC", "abc"))),
    assertz((user:str_code     :- string_code(2, "abc", 98))),
    assertz((user:str_code_n   :- string_code(2, "abc", 99))),
    assertz((user:str_code_oob :- string_code(99, "abc", _))),
    assertz((user:str_split    :- split_string("a,b,c", ",", "", ["a", "b", "c"]))),
    assertz((user:str_split_p  :- split_string(" a , b , c ", ",", " ", ["a", "b", "c"]))),
    assertz((user:str_split_e  :- split_string("abc", "", "", ["abc"]))),
    unique_r_tmp_dir('tmp_r_strops_e2e', TmpDir),
    write_wam_r_project(
        [ user:str_concat/0, user:str_concat_n/0,
          user:str_length/0, user:str_length_n/0,
          user:str_atom/0, user:str_to_atom/0,
          user:str_chars/0, user:str_codes/0,
          user:str_upper/0, user:str_lower/0,
          user:str_code/0, user:str_code_n/0, user:str_code_oob/0,
          user:str_split/0, user:str_split_p/0, user:str_split_e/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [str_concat, str_length,
           str_atom, str_to_atom,
           str_chars, str_codes,
           str_upper, str_lower,
           str_code,
           str_split, str_split_p, str_split_e],
    No  = [str_concat_n, str_length_n,
           str_code_n, str_code_oob],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    forall(member(P, No), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "false"))
    )),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for assert/retract/abolish (runtime dynamic
% predicate store). Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(dynamic_preds_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_dynamic_preds_via_rscript
    ;   true
    )).

e2e_dynamic_preds_via_rscript :-
    % assertz + dispatch through the dynamic store.
    assertz((user:dyn_basic_test :- assertz(d_fact(a)), d_fact(a))),
    % Multiple clauses get dispatched in order.
    assertz((user:dyn_multi :- assertz(p(1)), assertz(p(2)), assertz(p(3)),
                                p(1), p(2), p(3))),
    % asserta prepends, so an earlier `assertz(q(later))` followed by
    % `asserta(q(first))` should leave `q(first)` matchable first.
    assertz((user:dyn_order :- assertz(q(later)), asserta(q(first)),
                                q(first))),
    % Asserting a rule that uses arithmetic; the body runs through
    % call_goal, which reaches builtins.
    assertz((user:dyn_rule :- assertz((dbl(X, Y) :- Y is X * 2)),
                              dbl(5, R), R == 10)),
    % retract removes the matching clause; the second one survives.
    assertz((user:dyn_retract :- assertz(r(a)), assertz(r(b)),
                                  retract(r(a)), \+ r(a), r(b))),
    % Failure case: retract a clause that doesn't exist.
    assertz((user:dyn_retract_no :- assertz(s(a)), retract(s(z)))),
    % abolish wipes the whole entry.
    assertz((user:dyn_abolish :- assertz(g(x)), abolish(g/1), \+ g(x))),
    % Cross-call: dynamic clause invoked indirectly through call/1.
    assertz((user:dyn_via_call :- assertz(t(ok)), call(t(ok)))),
    % Negative: querying a never-asserted predicate.
    assertz((user:dyn_unknown_no :- never_asserted(_))),
    unique_r_tmp_dir('tmp_r_dynamic_e2e', TmpDir),
    write_wam_r_project(
        [ user:dyn_basic_test/0, user:dyn_multi/0, user:dyn_order/0,
          user:dyn_rule/0, user:dyn_retract/0, user:dyn_retract_no/0,
          user:dyn_abolish/0, user:dyn_via_call/0,
          user:dyn_unknown_no/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [dyn_basic_test, dyn_multi, dyn_order, dyn_rule,
           dyn_retract, dyn_abolish, dyn_via_call],
    No  = [dyn_retract_no, dyn_unknown_no],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    forall(member(P, No), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "false"))
    )),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for findall/3 (multi-solution machinery
% via BeginAggregate / EndAggregate). Auto-skips when Rscript is not
% on PATH.
% ------------------------------------------------------------------
test(findall_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_findall_via_rscript
    ;   true
    )).

e2e_findall_via_rscript :-
    % Static multi-clause source.
    assertz(user:fa_f(a)),
    assertz(user:fa_f(b)),
    assertz(user:fa_f(c)),
    assertz(user:fa_n(1)),
    assertz(user:fa_n(2)),
    assertz(user:fa_n(3)),
    % Basic enumeration.
    assertz((user:fa_basic    :- findall(X, user:fa_f(X), L),
                                  L == [a, b, c])),
    % Empty case (goal has no solutions).
    assertz((user:fa_empty    :- findall(X, user:fa_nope(X), L),
                                  L == [])),
    % Conjunctive goal with a guard.
    assertz((user:fa_filter   :- findall(X, (user:fa_n(X), X > 1), L),
                                  L == [2, 3])),
    % Bag with non-trivial template (X*X is a struct, not evaluated).
    assertz((user:fa_template :- findall(X * X, user:fa_n(X), L),
                                  L = [_, _, _])),
    % findall over a dynamic predicate (asserted at runtime). This
    % exercises the multi-solution path through try_dynamic.
    assertz((user:fa_dyn      :- assertz(fa_d(x)), assertz(fa_d(y)),
                                  assertz(fa_d(z)),
                                  findall(V, fa_d(V), L),
                                  L == [x, y, z])),
    % Length of the bag.
    assertz((user:fa_count    :- findall(X, user:fa_f(X), L),
                                  length(L, 3))),
    % Negative case: condition makes the WHOLE pred fail.
    assertz((user:fa_basic_no :- findall(X, user:fa_f(X), L),
                                  L == [a, b])),
    unique_r_tmp_dir('tmp_r_findall_e2e', TmpDir),
    write_wam_r_project(
        [ user:fa_f/1, user:fa_n/1,
          user:fa_basic/0, user:fa_empty/0, user:fa_filter/0,
          user:fa_template/0, user:fa_dyn/0, user:fa_count/0,
          user:fa_basic_no/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [fa_basic, fa_empty, fa_filter, fa_template,
           fa_dyn, fa_count],
    No  = [fa_basic_no],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    forall(member(P, No), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "false"))
    )),
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
