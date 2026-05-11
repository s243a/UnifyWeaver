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
    assertion(current_predicate(wam_r_target:classify_r_fact_predicate/4)),
    assertion(current_predicate(wam_r_target:split_wam_into_segments_r/2)),
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
% Test 4b: R fact-shape classifier mirrors Haskell F1 policy basics
% ------------------------------------------------------------------
test(fact_shape_classifier_fact_only) :-
    once((
        retractall(user:r_f1_color(_)),
        assertz(user:r_f1_color(red)),
        assertz(user:r_f1_color(green)),
        assertz(user:r_f1_color(blue)),
        wam_target:compile_predicate_to_wam(r_f1_color/1, [], WamCode),
        atom_string(WamCode, WamStr),
        split_string(WamStr, "\n", "", Lines),
        wam_r_target:classify_r_fact_predicate(r_f1_color/1, Lines, [], Info),
        assertion(Info = fact_shape_info(3, true, all_ground, compiled)),
        retractall(user:r_f1_color(_))
    )).

test(fact_shape_classifier_rule_not_fact_only) :-
    once((
        retractall(user:r_f1_parent(_, _)),
        retractall(user:r_f1_ancestor(_, _)),
        assertz(user:r_f1_parent(tom, bob)),
        assertz((user:r_f1_ancestor(X, Y) :- user:r_f1_parent(X, Y))),
        assertz((user:r_f1_ancestor(X, Y) :-
            user:r_f1_parent(X, Z),
            user:r_f1_ancestor(Z, Y))),
        wam_target:compile_predicate_to_wam(r_f1_ancestor/2, [], WamCode),
        atom_string(WamCode, WamStr),
        split_string(WamStr, "\n", "", Lines),
        wam_r_target:classify_r_fact_predicate(r_f1_ancestor/2, Lines, [], Info),
        Info = fact_shape_info(_, FactOnly, _, _),
        assertion(FactOnly == false),
        retractall(user:r_f1_ancestor(_, _)),
        retractall(user:r_f1_parent(_, _))
    )).

test(fact_shape_layout_auto_threshold) :-
    once((
        retractall(user:r_f1_big(_)),
        forall(between(1, 6, I), (
            atom_number(A, I),
            assertz(user:r_f1_big(A))
        )),
        wam_target:compile_predicate_to_wam(r_f1_big/1, [], WamCode),
        atom_string(WamCode, WamStr),
        split_string(WamStr, "\n", "", Lines),
        wam_r_target:classify_r_fact_predicate(
            r_f1_big/1, Lines, [fact_count_threshold(5)], Info),
        assertion(Info = fact_shape_info(6, true, all_ground, inline_data([]))),
        retractall(user:r_f1_big(_))
    )).

test(fact_shape_layout_user_override) :-
    once((
        retractall(user:r_f1_override(_)),
        assertz(user:r_f1_override(a)),
        wam_target:compile_predicate_to_wam(r_f1_override/1, [], WamCode),
        atom_string(WamCode, WamStr),
        split_string(WamStr, "\n", "", Lines),
        wam_r_target:classify_r_fact_predicate(
            r_f1_override/1,
            Lines,
            [fact_layout(r_f1_override/1, external_source(tsv("data.tsv")))],
            Info),
        assertion(Info = fact_shape_info(_, _, _, external_source(tsv("data.tsv")))),
        retractall(user:r_f1_override(_))
    )).

test(fact_shape_comment_emitted) :-
    once((
        unique_r_tmp_dir('tmp_r_fact_shape', TmpDir),
        write_wam_r_project(
            [user:wam_r_fact/1],
            [],
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        assertion(sub_string(Code, _, _, _, 'Fact shape classification')),
        assertion(sub_string(Code, _, _, _, 'wam_r_fact/1: fact_only=true')),
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
        assertion(sub_string(Code, _, _, _, 'pred_wam_r_caller <- function(')),
        assertion(sub_string(Code, _, _, _, 'pred_wam_r_fact <- function(')),
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
        assertion(sub_string(Code, _, _, _, 'pred_r_double <- function(')),
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

% Like run_rscript_query but takes an explicit list of additional
% argv entries (after the predicate indicator). Used by the CLI
% arg-parser test, which exercises the operator-precedence parser
% on the command-line args.
run_rscript_with_args(RDir, Query, ExtraArgs, Out) :-
    append(['generated_program.R', Query], ExtraArgs, Argv),
    process_create(path('Rscript'), Argv,
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
        % fact_table_layout(off) keeps wam_r_choice_fact/1 on the
        % multi_clause_1 lowered path; without it the new fact-table
        % path would pre-empt this test's assertions.
        write_wam_r_project(
            [user:wam_r_choice_fact/1],
            [emit_mode(functions), fact_table_layout(off)],
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
        % fact_table_layout(off) forces the multi_clause_1 lowered
        % path that this test was written to cover; the new fact-
        % table path is exercised separately by
        % fact_table_e2e_rscript.
        [emit_mode(functions), fact_table_layout(off)],
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
% Mode-analysis phase 1 (visibility-only). With mode_comments(on) in
% Options, the lowered emitter prepends a `# Mode analysis:` comment
% block above each lowered function summarising the analyser's per-
% clause head-binding states. Pure visibility -- no runtime / codegen
% behaviour change. Foundation for phase 2 specialisations.
%
% This is a structural test (no Rscript needed); it asserts on the
% generated R file contents directly.
% ------------------------------------------------------------------
test(mode_analysis_phase1_comments) :-
    once((
        % Cleanup previous assertions if any.
        retractall(user:ma_caller/1),
        retractall(user:ma_callee/1),
        retractall(user:mode(ma_caller(_))),
        % Single-clause deterministic predicate so the lowered emitter
        % picks it up. The mode declaration seeds the analyser: arg 1
        % is input (-> bound at clause entry).
        assertz(user:mode(ma_caller(+))),
        assertz(user:ma_callee(_)),
        assertz((user:ma_caller(X) :- user:ma_callee(X))),
        unique_r_tmp_dir('tmp_r_mode_comments', TmpDir),
        write_wam_r_project(
            [ user:ma_caller/1, user:ma_callee/1 ],
            [emit_mode(functions), mode_comments(on)],
            TmpDir),
        directory_file_path(TmpDir, 'R/generated_program.R', Program),
        read_file_to_string(Program, Code, []),
        % The lowered function for ma_caller/1 should be preceded by
        % the mode-analysis comment block, including the head-binding
        % summary that mirrors the input-mode declaration.
        assertion(sub_string(Code, _, _, _,
            '# Mode analysis (phase 1, visibility-only):')),
        assertion(sub_string(Code, _, _, _,
            'mode_decl=[input]')),
        % And the lowered function itself is still emitted.
        assertion(sub_string(Code, _, _, _,
            'lowered_ma_caller_1 <- function(program, state)')),
        % Sanity: with the option OFF, the comment block is absent.
        unique_r_tmp_dir('tmp_r_mode_comments_off', TmpDir2),
        write_wam_r_project(
            [ user:ma_caller/1, user:ma_callee/1 ],
            [emit_mode(functions)],
            TmpDir2),
        directory_file_path(TmpDir2, 'R/generated_program.R', Program2),
        read_file_to_string(Program2, Code2, []),
        assertion(\+ sub_string(Code2, _, _, _,
            '# Mode analysis (phase 1, visibility-only):')),
        % Cleanup.
        retractall(user:mode(ma_caller(_))),
        delete_directory_and_contents(TmpDir),
        delete_directory_and_contents(TmpDir2)
    )).

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
    % length/2 deterministic and generative modes.
    assertz((user:lb_len_count :- length([a, b, c], 3))),
    assertz((user:lb_len_zero  :- length([], 0))),
    assertz((user:lb_len_wrong :- length([a, b, c], 4))),
    assertz((user:lb_len_build :- length(L, 3), is_list(L))),
    assertz((user:lb_len_gen_zero :- length(L, N), L == [], N == 0)),
    assertz((user:lb_len_gen_two :-
        length(L, N),
        N == 2,
        L = [_, _])),
    assertz((user:lb_len_gen_three :-
        length(L, N),
        N == 3,
        L = [_, _, _])),
    % append/3 deterministic and finite split modes.
    assertz((user:lb_app_ok    :- append([1, 2], [3, 4], [1, 2, 3, 4]))),
    assertz((user:lb_app_empty :- append([], [a, b], [a, b]))),
    assertz((user:lb_app_no    :- append([1, 2], [3, 4], [1, 2, 3, 5]))),
    assertz((user:lb_app_split_all :-
        findall(P-S, append(P, S, [a, b]), Splits),
        Splits == [ []-[a, b], [a]-[b], [a, b]-[] ])),
    assertz((user:lb_app_split_later :-
        append(P, S, [a, b, c]),
        P == [a, b],
        S == [c])),
    assertz((user:lb_app_split_empty :-
        append(P, S, []),
        P == [],
        S == [])),
    assertz((user:lb_app_split_no :-
        append(P, S, [a, b]),
        P == [b],
        S == [a])),
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
          user:lb_len_build/0, user:lb_len_gen_zero/0,
          user:lb_len_gen_two/0, user:lb_len_gen_three/0,
          user:lb_app_ok/0, user:lb_app_empty/0, user:lb_app_no/0,
          user:lb_app_split_all/0, user:lb_app_split_later/0,
          user:lb_app_split_empty/0, user:lb_app_split_no/0,
          user:lb_rev_ok/0, user:lb_rev_no/0,
          user:lb_last_ok/0, user:lb_last_no/0,
          user:lb_alen_ok/0, user:lb_alen_no/0,
          user:lb_acodes_ok/0, user:lb_achars_ok/0,
          user:lb_aconcat_ok/0, user:lb_aconcat_no/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [lb_len_count, lb_len_zero, lb_len_build,
           lb_len_gen_zero, lb_len_gen_two, lb_len_gen_three,
           lb_app_ok, lb_app_empty, lb_app_split_all,
           lb_app_split_later, lb_app_split_empty,
           lb_rev_ok, lb_last_ok,
           lb_alen_ok, lb_acodes_ok, lb_achars_ok, lb_aconcat_ok],
    No  = [lb_len_wrong, lb_app_no, lb_app_split_no,
           lb_rev_no, lb_last_no,
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
    assertz((user:hl_select_dup_all :-
        findall(Rest, select(a, [a, b, a], Rest), Rests),
        Rests == [[b, a], [a, b]])),
    assertz((user:hl_select_var_all :-
        findall(X-Rest, select(X, [a, b, a], Rest), Pairs),
        Pairs == [a-[b, a], b-[a, a], a-[a, b]])),
    assertz((user:hl_select_bound_rest :-
        select(X, [a, b, c], [a, c]),
        X == b)),
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
          user:hl_select_dup_all/0, user:hl_select_var_all/0,
          user:hl_select_bound_rest/0,
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
           hl_select_mid, hl_select_one, hl_select_dup_all,
           hl_select_var_all, hl_select_bound_rest,
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
% End-to-end Rscript run for bagof/3, setof/3, once/1, forall/2.
% These build on the multi-solution machinery and cover user
% predicates as aggregate goals. Builtin enumerators are covered
% separately by enumerable_builtins_e2e_rscript.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(bagof_setof_once_forall_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_bagof_setof_once_forall_via_rscript
    ;   true
    )).

e2e_bagof_setof_once_forall_via_rscript :-
    % Multi-clause sources for enumeration.
    assertz(user:bso_letter(a)),
    assertz(user:bso_letter(b)),
    assertz(user:bso_letter(c)),
    assertz(user:bso_num(3)),
    assertz(user:bso_num(1)),
    assertz(user:bso_num(2)),
    assertz(user:bso_num(1)),
    % bagof/3: same as findall but fails on empty.
    assertz((user:bg_basic    :- bagof(X, user:bso_letter(X), L), L == [a, b, c])),
    assertz((user:bg_empty_no :- bagof(X, user:nope(X), _))),
    % setof/3: bag + sort + dedup.
    assertz((user:so_basic    :- setof(X, user:bso_num(X), L), L == [1, 2, 3])),
    assertz((user:so_atoms    :- setof(X, user:bso_letter(X), L), L == [a, b, c])),
    assertz((user:so_empty_no :- setof(X, user:nope(X), _))),
    % once/1: commits to first solution; outer backtracking can't
    % re-enter the goal.
    assertz((user:on_basic    :- once(user:bso_letter(X)), X == a)),
    assertz((user:on_inner_no :- once(user:nope(_)))),
    % forall/2 over a user predicate. fa_pos and fa_atom both succeed
    % because every element of the source matches Action.
    assertz((user:fa_atom     :- forall(user:bso_letter(X), atom(X)))),
    assertz((user:fa_atom_no  :- forall(user:bso_letter(X), integer(X)))),
    unique_r_tmp_dir('tmp_r_bso_e2e', TmpDir),
    write_wam_r_project(
        [ user:bso_letter/1, user:bso_num/1,
          user:bg_basic/0, user:bg_empty_no/0,
          user:so_basic/0, user:so_atoms/0, user:so_empty_no/0,
          user:on_basic/0, user:on_inner_no/0,
          user:fa_atom/0, user:fa_atom_no/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [bg_basic, so_basic, so_atoms,
           on_basic,
           fa_atom],
    No  = [bg_empty_no, so_empty_no,
           on_inner_no,
           fa_atom_no],
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
% End-to-end Rscript run for enumerable builtins inside aggregators.
% Tests that findall/3, bagof/3, setof/3, forall/2 all enumerate over
% member/2, between/3, and `,/2` conjunctions thereof.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(enumerable_builtins_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_enumerable_builtins_via_rscript
    ;   true
    )).

e2e_enumerable_builtins_via_rscript :-
    % Multi-clause source for regression checks (must keep working).
    assertz(user:enu_letter(a)),
    assertz(user:enu_letter(b)),
    assertz(user:enu_letter(c)),
    % findall over member.
    assertz((user:enu_fa_mem    :- findall(X, member(X, [1, 2, 3]), L),
                                    L == [1, 2, 3])),
    % bagof over member.
    assertz((user:enu_bg_mem    :- bagof(X, member(X, [a, b, c]), L),
                                    L == [a, b, c])),
    % setof over member with duplicates.
    assertz((user:enu_so_mem    :- setof(X, member(X, [c, a, b, a]), L),
                                    L == [a, b, c])),
    % findall over between.
    assertz((user:enu_fa_btw    :- findall(X, between(1, 5, X), L),
                                    L == [1, 2, 3, 4, 5])),
    % bagof over between.
    assertz((user:enu_bg_btw    :- bagof(X, between(2, 4, X), L),
                                    L == [2, 3, 4])),
    % findall conjunction (member + guard).
    assertz((user:enu_fa_conj   :- findall(X, (member(X, [1, 2, 3, 4, 5]),
                                                X > 2), L),
                                    L == [3, 4, 5])),
    % bagof conjunction.
    assertz((user:enu_bg_conj   :- bagof(X, (member(X, [1, 2, 3, 4]),
                                              X > 2), L),
                                    L == [3, 4])),
    % forall with all elements satisfying Action.
    assertz((user:enu_fa_all    :- forall(member(X, [1, 2, 3]),
                                           integer(X)))),
    % forall with one element not satisfying Action.
    assertz((user:enu_fa_some_no :- forall(member(X, [1, foo, 3]),
                                            integer(X)))),
    % forall over between.
    assertz((user:enu_fa_btw_all :- forall(between(1, 10, X), integer(X)))),
    % Regression: multi-clause user predicate still works.
    assertz((user:enu_fa_user   :- findall(X, user:enu_letter(X), L),
                                    L == [a, b, c])),
    unique_r_tmp_dir('tmp_r_enumerable_e2e', TmpDir),
    write_wam_r_project(
        [ user:enu_letter/1,
          user:enu_fa_mem/0, user:enu_bg_mem/0, user:enu_so_mem/0,
          user:enu_fa_btw/0, user:enu_bg_btw/0,
          user:enu_fa_conj/0, user:enu_bg_conj/0,
          user:enu_fa_all/0, user:enu_fa_some_no/0,
          user:enu_fa_btw_all/0,
          user:enu_fa_user/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [enu_fa_mem, enu_bg_mem, enu_so_mem,
           enu_fa_btw, enu_bg_btw,
           enu_fa_conj, enu_bg_conj,
           enu_fa_all, enu_fa_btw_all,
           enu_fa_user],
    No  = [enu_fa_some_no],
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
% End-to-end Rscript run for catch/3 + throw/1 (exception handling)
% and runtime aggregator support for dynamic predicates.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(catch_throw_dyn_aggregator_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_catch_throw_dyn_via_rscript
    ;   true
    )).

e2e_catch_throw_dyn_via_rscript :-
    % catch/throw basics.
    assertz((user:ct_basic     :- catch(throw(boom), boom, true))),
    % Throw a structured term; recovery sees the unified Catcher var.
    assertz((user:ct_unify     :- catch(throw(error(my_err, ctx)),
                                         error(E, _), E == my_err))),
    % Goal succeeds with no throw -- catch just succeeds, no recovery.
    assertz((user:ct_no_throw  :- catch(true, _, fail))),
    % Catcher doesn't unify -> rethrown -> top-level returns FALSE.
    assertz((user:ct_uncaught  :- catch(throw(boom), other, true))),
    % Nested catch: outer catches what inner rethrows.
    assertz((user:ct_nested    :- catch(catch(throw(inner), other, fail),
                                         inner, true))),
    % Bare throw without any catch -> top-level returns FALSE.
    assertz((user:ct_bare_throw :- throw(uncaught))),
    % Runtime aggregators over dynamic-store predicates.
    assertz((user:dyn_bg :- assertz(dynbg(a)), assertz(dynbg(b)),
                            assertz(dynbg(c)),
                            bagof(X, dynbg(X), L), L == [a, b, c])),
    assertz((user:dyn_so :- assertz(dynso(c)), assertz(dynso(a)),
                            assertz(dynso(b)), assertz(dynso(a)),
                            setof(X, dynso(X), L), L == [a, b, c])),
    assertz((user:dyn_fa :- assertz(dynfa(1)), assertz(dynfa(2)),
                            assertz(dynfa(3)),
                            forall(dynfa(X), integer(X)))),
    assertz((user:dyn_fa_no :- assertz(dynfan(1)), assertz(dynfan(foo)),
                                assertz(dynfan(3)),
                                forall(dynfan(X), integer(X)))),
    unique_r_tmp_dir('tmp_r_catch_dyn_e2e', TmpDir),
    write_wam_r_project(
        [ user:ct_basic/0, user:ct_unify/0, user:ct_no_throw/0,
          user:ct_uncaught/0, user:ct_nested/0, user:ct_bare_throw/0,
          user:dyn_bg/0, user:dyn_so/0, user:dyn_fa/0, user:dyn_fa_no/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [ct_basic, ct_unify, ct_no_throw, ct_nested,
           dyn_bg, dyn_so, dyn_fa],
    No  = [ct_uncaught, ct_bare_throw, dyn_fa_no],
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
% End-to-end Rscript run for the standard-library polish round:
% numlist/3, tab/1, permutation/2 (det check + identity), sub_atom/5
% (forward mode), char_type/2, plus the term_to_atom/2 builtin which
% uses the new tokenizer + parser.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(stdlib_polish_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_stdlib_polish_via_rscript
    ;   true
    )).

e2e_stdlib_polish_via_rscript :-
    % numlist/3.
    assertz((user:nl_basic   :- numlist(1, 5, L), L == [1, 2, 3, 4, 5])),
    assertz((user:nl_one     :- numlist(7, 7, L), L == [7])),
    assertz((user:nl_empty_no :- numlist(5, 1, _))),  % Hi < Lo -> fail
    % permutation/2 (deterministic check).
    assertz((user:pm_check   :- permutation([1, 2, 3], [3, 2, 1]))),
    assertz((user:pm_check_no :- permutation([1, 2, 3], [3, 2, 4]))),
    % sub_atom/5 (forward mode: Before+Length both bound).
    assertz((user:sa_basic   :- sub_atom(hello, 1, 3, A, S),
                                 A =:= 1, S == ell)),
    assertz((user:sa_full    :- sub_atom(abc, 0, 3, 0, abc))),
    assertz((user:sa_oob_no  :- sub_atom(abc, 1, 99, _, _))),
    % char_type/2.
    assertz((user:ct_alpha   :- char_type(a, alpha))),
    assertz((user:ct_alpha_no :- char_type('1', alpha))),
    assertz((user:ct_digit   :- char_type('5', digit))),
    assertz((user:ct_upper   :- char_type('Z', upper))),
    assertz((user:ct_lower_no :- char_type('Z', lower))),
    % term_to_atom/2 forward (Term bound -> Atom rendered).
    % Uses atom_length to side-step the literal `'42'` vs integer 42
    % collision in WAM-text serialisation.
    assertz((user:t2a_fwd_int    :- term_to_atom(42, A), atom_length(A, 2))),
    assertz((user:t2a_fwd_struct :- term_to_atom(f(a, b), A),
                                     atom_length(A, 6))),
    % term_to_atom/2 reverse (Atom bound -> parse to Term).
    % Build the source atom via atom_codes so the generated WAM
    % carries the actual character string we want to parse.
    assertz((user:t2a_parse_int :- atom_codes(A, [52, 50]),  % "42"
                                    term_to_atom(T, A), T =:= 42)),
    assertz((user:t2a_parse_atom :- atom_codes(A, [104, 105]), % "hi"
                                     term_to_atom(T, A), T == hi)),
    assertz((user:t2a_parse_list :- atom_codes(A, [91, 49, 44, 50, 44, 51, 93]),  % "[1,2,3]"
                                     term_to_atom(T, A),
                                     T == [1, 2, 3])),
    assertz((user:t2a_parse_struct :- atom_codes(A, [102, 40, 97, 44, 98, 41]),  % "f(a,b)"
                                       term_to_atom(T, A),
                                       T == f(a, b))),
    % term_to_atom round-trip: render then re-parse should give the
    % structurally equal term.
    assertz((user:t2a_round :- atom_codes(A, [102, 40, 49, 44, 50, 41]),  % "f(1,2)"
                                term_to_atom(T, A),
                                term_to_atom(T, A2),
                                A2 == A)),
    unique_r_tmp_dir('tmp_r_polish_e2e', TmpDir),
    write_wam_r_project(
        [ user:nl_basic/0, user:nl_one/0, user:nl_empty_no/0,
          user:pm_check/0, user:pm_check_no/0,
          user:sa_basic/0, user:sa_full/0, user:sa_oob_no/0,
          user:ct_alpha/0, user:ct_alpha_no/0, user:ct_digit/0,
          user:ct_upper/0, user:ct_lower_no/0,
          user:t2a_fwd_int/0, user:t2a_fwd_struct/0,
          user:t2a_parse_int/0, user:t2a_parse_atom/0,
          user:t2a_parse_list/0, user:t2a_parse_struct/0,
          user:t2a_round/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [nl_basic, nl_one,
           pm_check,
           sa_basic, sa_full,
           ct_alpha, ct_digit, ct_upper,
           t2a_fwd_int, t2a_fwd_struct,
           t2a_parse_int, t2a_parse_atom,
           t2a_parse_list, t2a_parse_struct,
           t2a_round],
    No  = [nl_empty_no,
           pm_check_no,
           sa_oob_no,
           ct_alpha_no, ct_lower_no],
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
% End-to-end Rscript run for operator-precedence parsing in
% term_to_atom/2 reverse mode. Asserts that the parser respects
% operator associativity (yfx, xfy) and precedence ordering, and
% that prefix operators (\+) work.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(operator_parser_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_operator_parser_via_rscript
    ;   true
    )).

e2e_operator_parser_via_rscript :-
    % "1+2" -> +(1, 2).
    assertz((user:op_simple :-
        atom_codes(A, [49, 43, 50]),
        term_to_atom(T, A), T == 1+2)),
    % "1+2*3" -> +(1, *(2, 3)) (mul binds tighter).
    assertz((user:op_prec :-
        atom_codes(A, [49, 43, 50, 42, 51]),
        term_to_atom(T, A), T == 1+2*3)),
    % "1+2+3" -> +(+(1, 2), 3) (yfx, left associative).
    assertz((user:op_left_assoc :-
        atom_codes(A, [49, 43, 50, 43, 51]),
        term_to_atom(T, A), T == 1+2+3)),
    % "2^3^4" -> ^(2, ^(3, 4)) (xfy, right associative).
    assertz((user:op_right_assoc :-
        atom_codes(A, [50, 94, 51, 94, 52]),
        term_to_atom(T, A), T == 2^3^4)),
    % Whitespace tolerated: "1 + 2" -> +(1, 2).
    assertz((user:op_spaces :-
        atom_codes(A, [49, 32, 43, 32, 50]),
        term_to_atom(T, A), T == 1+2)),
    % "a=b" -> =(a, b).
    assertz((user:op_eq :-
        atom_codes(A, [97, 61, 98]),
        term_to_atom(T, A), T == (a=b))),
    % "1<2" -> <(1, 2).
    assertz((user:op_compare :-
        atom_codes(A, [49, 60, 50]),
        term_to_atom(T, A), T == (1<2))),
    % Multi-char operator: "1+1 =:= 2" -> =:=(+(1, 1), 2).
    assertz((user:op_arith_eq :-
        atom_codes(A, [49, 43, 49, 32, 61, 58, 61, 32, 50]),
        term_to_atom(T, A), T == (1+1 =:= 2))),
    % Prefix operator: "\+foo" -> \+(foo).
    assertz((user:op_prefix_neg :-
        atom_codes(A, [92, 43, 102, 111, 111]),
        term_to_atom(T, A), T == (\+ foo))),
    % Comma as operator inside parens: "(a,b,c)" -> ','(a, ','(b, c)).
    assertz((user:op_comma :-
        atom_codes(A, [40, 97, 44, 98, 44, 99, 41]),
        term_to_atom(T, A), T == (a,b,c))),
    % Negative integer literal preserved: "f(-3)" -> f(-3).
    assertz((user:op_neg_lit :-
        atom_codes(A, [102, 40, 45, 51, 41]),
        term_to_atom(T, A), T == f(-3))),
    % Round-trip: render an arithmetic term and re-parse to the
    % structurally equal term.
    assertz((user:op_round_trip :-
        T0 = 1+2*3,
        term_to_atom(T0, A),
        term_to_atom(T1, A),
        T1 == T0)),
    unique_r_tmp_dir('tmp_r_op_parser_e2e', TmpDir),
    write_wam_r_project(
        [ user:op_simple/0, user:op_prec/0, user:op_left_assoc/0,
          user:op_right_assoc/0, user:op_spaces/0, user:op_eq/0,
          user:op_compare/0, user:op_arith_eq/0, user:op_prefix_neg/0,
          user:op_comma/0, user:op_neg_lit/0, user:op_round_trip/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [op_simple, op_prec, op_left_assoc, op_right_assoc,
           op_spaces, op_eq, op_compare, op_arith_eq,
           op_prefix_neg, op_comma, op_neg_lit, op_round_trip],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for the runtime op/3 builtin. The driver
% predicates declare a custom operator at runtime and then exercise
% read_term_from_atom and current_op/3 against the augmented parser
% table. Covers infix add, prefix add, xfy right-associativity for a
% custom op, op(0, ...) removal, and current_op/3 enumeration.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(op_3_runtime_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_op_3_runtime_via_rscript
    ;   true
    )).

e2e_op_3_runtime_via_rscript :-
    setup_call_cleanup(
        e2e_op_3_runtime_setup(TmpDir),
        e2e_op_3_runtime_body(TmpDir),
        e2e_op_3_runtime_cleanup(TmpDir)).

e2e_op_3_runtime_setup(TmpDir) :-
    % op/3 adds an infix op; subsequent read_term_from_atom uses it.
    % Decompose via =.. to avoid needing the op at SWI-Prolog level.
    assertz((user:op_runtime_infix :-
        op(900, xfy, '==>'),
        read_term_from_atom('a==>b', T),
        T =.. [F, A, B],
        F == '==>', A == a, B == b)),
    % xfy right-associativity: 'a ==> b ==> c' parses as (a ==> (b ==> c)).
    assertz((user:op_runtime_xfy :-
        op(900, xfy, '==>'),
        read_term_from_atom('a==>b==>c', T),
        T =.. [_, _, RHS],
        RHS =.. [F2, B, C],
        F2 == '==>', B == b, C == c)),
    % Prefix op: 'neg foo' parses as neg(foo). The space matters --
    % 'neg(foo)' would parse as a function call regardless of op decl.
    assertz((user:op_runtime_prefix :-
        op(200, fy, neg),
        read_term_from_atom('neg foo', T),
        T =.. [F, X],
        F == neg, X == foo)),
    % current_op/3 enumeration finds the new op.
    assertz((user:op_runtime_current :-
        op(900, xfy, '==>'),
        findall(P, current_op(P, xfy, '==>'), Precs),
        Precs == [900])),
    % op(0, T, N) removes; subsequent parse fails.
    assertz((user:op_runtime_remove :-
        op(900, xfy, '==>'),
        op(0, xfy, '==>'),
        \+ read_term_from_atom('a==>b', _))),
    unique_r_tmp_dir('tmp_r_op_3_runtime_e2e', TmpDir).

e2e_op_3_runtime_body(TmpDir) :-
    write_wam_r_project(
        [user:op_runtime_infix/0, user:op_runtime_xfy/0,
         user:op_runtime_prefix/0, user:op_runtime_current/0,
         user:op_runtime_remove/0],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [op_runtime_infix, op_runtime_xfy, op_runtime_prefix,
           op_runtime_current, op_runtime_remove],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )).

e2e_op_3_runtime_cleanup(TmpDir) :-
    retractall(user:op_runtime_infix),
    retractall(user:op_runtime_xfy),
    retractall(user:op_runtime_prefix),
    retractall(user:op_runtime_current),
    retractall(user:op_runtime_remove),
    ignore(catch(delete_directory_and_contents(TmpDir), _, true)).

% ------------------------------------------------------------------
% End-to-end Rscript run for the r_op_decls codegen option. The
% generated program seeds the operator table at init time, so no
% runtime op/3 call is needed -- read_term_from_atom recognises
% the custom ops out of the box.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(op_3_decl_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_op_3_decl_via_rscript
    ;   true
    )).

e2e_op_3_decl_via_rscript :-
    setup_call_cleanup(
        e2e_op_3_decl_setup(TmpDir),
        e2e_op_3_decl_body(TmpDir),
        e2e_op_3_decl_cleanup(TmpDir)).

e2e_op_3_decl_setup(TmpDir) :-
    assertz((user:op_decl_infix :-
        read_term_from_atom('a==>b', T),
        T =.. [F, A, B],
        F == '==>', A == a, B == b)),
    assertz((user:op_decl_prefix :-
        read_term_from_atom('neg foo', T),
        T =.. [F, X],
        F == neg, X == foo)),
    % Multi-name list form: r_op_decls accepts op(P, T, [N1, N2, ...]).
    assertz((user:op_decl_listform :-
        read_term_from_atom('a~b', T1),
        T1 =.. [F1, _, _], F1 == '~',
        read_term_from_atom('a@b', T2),
        T2 =.. [F2, _, _], F2 == '@')),
    unique_r_tmp_dir('tmp_r_op_3_decl_e2e', TmpDir).

e2e_op_3_decl_body(TmpDir) :-
    write_wam_r_project(
        [user:op_decl_infix/0, user:op_decl_prefix/0,
         user:op_decl_listform/0],
        [r_op_decls([op(900, xfy, '==>'),
                     op(200, fy, neg),
                     op(700, xfx, ['~', '@'])])],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [op_decl_infix, op_decl_prefix, op_decl_listform],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    % Structural: codegen emits one op_set call per name (list-form
    % expanded into a `for (nm in c(...))` loop).
    directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
    read_file_to_string(ProgPath, Code, []),
    assertion(sub_string(Code, _, _, _, 'WamRuntime$op_set("==>", 900L, "xfy")')),
    assertion(sub_string(Code, _, _, _, 'WamRuntime$op_set("neg", 200L, "fy")')),
    assertion(sub_string(Code, _, _, _, 'for (nm in c("~", "@")) WamRuntime$op_set(nm, 700L, "xfx")')).

e2e_op_3_decl_cleanup(TmpDir) :-
    retractall(user:op_decl_infix),
    retractall(user:op_decl_prefix),
    retractall(user:op_decl_listform),
    ignore(catch(delete_directory_and_contents(TmpDir), _, true)).

% ------------------------------------------------------------------
% End-to-end Rscript run for multi-solution retract/1. Asserts a
% handful of facts, drives findall(X, retract(p(X)), L) so the
% retract iterates across all matches via the iter-CP retry, and
% verifies (a) the bag captures every original X and (b) the dynamic
% store is empty afterwards.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(multi_solution_retract_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_multi_solution_retract_via_rscript
    ;   true
    )).

e2e_multi_solution_retract_via_rscript :-
    % Drive: assert three facts, retract them all via findall +
    % retract, check the bag is [1,2,3] (in assertion order) and
    % that the store is now empty (a fourth retract fails).
    assertz((user:msr_all :-
        assertz(p(1)), assertz(p(2)), assertz(p(3)),
        findall(X, retract(p(X)), L),
        L == [1, 2, 3],
        \+ retract(p(_)))),
    % Pattern-restricted retract: an inline filter inside the goal
    % still iterates retract over every clause (retract's removal
    % side-effect is permanent regardless of subsequent goal
    % failure). Odds collects only the survivors of the filter; the
    % store is empty afterwards.
    assertz((user:msr_filtered :-
        assertz(q(1)), assertz(q(2)), assertz(q(3)), assertz(q(4)),
        findall(X, (retract(q(X)), X mod 2 =:= 1), Odds),
        Odds == [1, 3],
        % All q clauses retracted.
        findall(Y, retract(q(Y)), Rest),
        Rest == [])),
    % Pattern-arg retract: retract clauses whose second arg matches
    % a pattern. Survivors stay in the store.
    assertz((user:msr_pattern :-
        assertz(qq(1, odd)),  assertz(qq(2, even)),
        assertz(qq(3, odd)),  assertz(qq(4, even)),
        findall(X, retract(qq(X, even)), Evens),
        Evens == [2, 4],
        findall(Y, retract(qq(Y, odd)), OddsLeft),
        OddsLeft == [1, 3])),
    % Rule retract: retract clauses whose body matches.
    assertz((user:msr_rule :-
        assertz((rr(1) :- true)),
        assertz((rr(2) :- fail)),
        assertz((rr(3) :- true)),
        % Retract rules whose body is `true`.
        findall(X, retract((rr(X) :- true)), Trues),
        Trues == [1, 3],
        % rr(2) (with body `fail`) survives.
        findall(Y, retract((rr(Y) :- _)), Rest),
        Rest == [2])),
    % Live (immediate-update) view: assertz that happens between
    % retract solutions is visible to subsequent solutions. The
    % fail-driven loop retracts m/1 and asserts m(X*10) on the way
    % out. With live-store iteration the next backtrack sees the
    % new clauses and retracts them in turn until the X*10 result
    % >= 1000. SWI's logical-update view would stop after the
    % original m(1) is retracted (snapshot doesn't see the asserts).
    %
    % Result accumulator uses dynamic retracted/1 so we don't have
    % to use findall + nested if-then-else here -- the compiler's
    % handling of nested ITE inside disjunction bodies has a
    % separate gap (Call(";", 2) emitted instead of inline ITE),
    % so this test stays in flat-conjunction territory.
    assertz((user:msr_live_assert :-
        assertz(m(1)),
        (   retract(m(X)),
            assertz(retracted(X)),
            Y is X * 10,
            Y < 1000,
            assertz(m(Y)),
            fail
        ;   true
        ),
        findall(R, retract(retracted(R)), Rs),
        msort(Rs, Sorted),
        Sorted == [1, 10, 100],
        \+ retract(m(_)))),
    unique_r_tmp_dir('tmp_r_msr_e2e', TmpDir),
    write_wam_r_project(
        [user:msr_all/0, user:msr_filtered/0, user:msr_pattern/0,
         user:msr_rule/0, user:msr_live_assert/0],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [msr_all, msr_filtered, msr_pattern, msr_rule,
           msr_live_assert],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for nested control flow inside conjunction
% bodies. `compile_inner_call_goals` (used by findall / aggregate
% bodies, by if-then-else cond-and-branch goals, and by both arms
% of a bare disjunction) used to fall through to `compile_goal_call`
% for any nested `(C -> T ; E)` or `(A ; B)` sub-goal -- emitting a
% useless `Call(";", 2)` (with the term constructed as data via
% PutStructure / SetConstant) that has no runtime handler and just
% fails. Locks in the recursive dispatch so an inner ITE / disjunction
% compiles to inline try/cut/trust + jump.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(nested_ite_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_nested_ite_via_rscript
    ;   true
    )).

e2e_nested_ite_via_rscript :-
    % ITE inside findall body: findall iterates retract(m(X)); on
    % each solution an inner ITE asserts m(99) when X =:= 1. With
    % the fix, the inner ITE compiles and runs; live retract sees
    % the newly-asserted clause; the bag is [1, 2, 99].
    assertz((user:ite_in_findall :-
        assertz(m(1)), assertz(m(2)),
        findall(X,
                ( retract(m(X)),
                  ( X =:= 1 -> assertz(m(99)) ; true )
                ),
                Bag),
        Bag == [1, 2, 99],
        \+ retract(m(_)))),
    % ITE inside a disjunction's left branch (fail-driven loop).
    % Without the fix this also routed through Call(";", 2) and
    % silently failed; with the fix odd/even classification runs
    % and prints in order.
    assertz((user:ite_in_disj :-
        findall(Tag-V,
                ( member(V, [1, 2, 3]),
                  ( V mod 2 =:= 0 -> Tag = even ; Tag = odd )
                ),
                Tagged),
        Tagged == [odd-1, even-2, odd-3])),
    % Bare disjunction (no ->) inside findall body. Same root cause
    % path; with the fix the disjunction compiles as inline
    % try_me_else + trust_me and routes both alternatives.
    assertz((user:disj_in_findall :-
        findall(X,
                ( member(X, [1, 2, 3, 4, 5]),
                  ( X > 3 ; X < 2 )
                ),
                Bag),
        Bag == [1, 4, 5])),
    % bare `(C -> T)` (implicit `; fail`) inside findall body.
    % The compiler now treats this as `(C -> T ; fail)` per SWI
    % convention -- prior to the fix it emitted Call("->", 2).
    assertz((user:bare_arrow_in_findall :-
        findall(X,
                ( member(X, [1, 2, 3]),
                  ( X >= 2 -> true )
                ),
                Bag),
        Bag == [2, 3])),
    unique_r_tmp_dir('tmp_r_nested_ite_e2e', TmpDir),
    write_wam_r_project(
        [user:ite_in_findall/0, user:ite_in_disj/0,
         user:disj_in_findall/0, user:bare_arrow_in_findall/0],
        [intern_atoms([even, odd])],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [ite_in_findall, ite_in_disj, disj_in_findall, bare_arrow_in_findall],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for fact_in_range/5: range queries on
% fact-tabled predicates via the per-arg sorted index. The codegen
% builds the sorted index alongside the existing hash index; the
% builtin binary-searches for tuples whose ArgPos value is in
% [Lo, Hi] and iterates via fact_table_iter_subset's iter-CP. Each
% sub-test uses a fail-driven loop with assertz/retract accumulators
% to drive backtracking without findall -- a separate
% PutVariable + PutStructure-on-A1 interaction in the WAM compiler
% leaks the template var when findall captures a non-anonymous
% template that also appears inside a struct arg of the call (filed
% as a follow-up; orthogonal to fact_in_range).
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(fact_in_range_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_fact_in_range_via_rscript
    ;   true
    )).

e2e_fact_in_range_via_rscript :-
    % price/2: fact table with int values at arg 2.
    assertz(user:price(apple,     5)),
    assertz(user:price(banana,    8)),
    assertz(user:price(cherry,   12)),
    assertz(user:price(date,     15)),
    assertz(user:price(eggplant,  3)),
    % q_mid: tuples with arg2 in [5,10] -> apple, banana.
    assertz((user:q_mid :-
        (   fact_in_range(price/2, 2, 5, 10, [Item, _P]),
            assertz(found(Item)),
            fail
        ;   true
        ),
        findall(X, retract(found(X)), Xs),
        msort(Xs, Sorted),
        Sorted == [apple, banana])),
    % q_high: arg2 in [12,99] -> cherry, date.
    assertz((user:q_high :-
        (   fact_in_range(price/2, 2, 12, 99, [Item, _P]),
            assertz(found(Item)),
            fail
        ;   true
        ),
        findall(X, retract(found(X)), Xs),
        msort(Xs, Sorted),
        Sorted == [cherry, date])),
    % q_empty: arg2 in [100,200] -> empty.
    assertz((user:q_empty :-
        \+ fact_in_range(price/2, 2, 100, 200, [_Item, _P]))),
    % q_exact: arg2 = exactly 12 -> just cherry.
    assertz((user:q_exact :-
        (   fact_in_range(price/2, 2, 12, 12, [Item, _P]),
            assertz(found(Item)),
            fail
        ;   true
        ),
        findall(X, retract(found(X)), Xs),
        Xs == [cherry])),
    % q_atom_arg: ArgPos 1 has only atom values, no sorted index
    %             -> always fails.
    assertz((user:q_atom_arg :-
        \+ fact_in_range(price/2, 1, 0, 100, [_Item, _P]))),
    % q_bad_pos: out-of-range ArgPos -> fails.
    assertz((user:q_bad_pos :-
        \+ fact_in_range(price/2, 99, 0, 100, [_Item, _P]))),
    unique_r_tmp_dir('tmp_r_fact_range_e2e', TmpDir),
    write_wam_r_project(
        [user:price/2, user:q_mid/0, user:q_high/0, user:q_empty/0,
         user:q_exact/0, user:q_atom_arg/0, user:q_bad_pos/0],
        [intern_atoms([apple, banana, cherry, date, eggplant])],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [q_mid, q_high, q_empty, q_exact, q_atom_arg, q_bad_pos],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for findall whose template var appears
% inside a struct arg of the inner goal. Prior to the put_variable
% self-init fix in compile_aggregate_all, the compiler emitted
% `put_variable Y_template, A1` -- sharing a fresh unbound between
% the Y register and A1. When the inner goal's first arg was then
% built as a compound (e.g. `p/2`), append_build_arg's auto-bind
% logic bound the shared unbound to the new struct, silently
% capturing the inner goal's first arg into the template var.
% Now put_variable self-inits (put_variable Y, Y) so A1 is left
% untouched until the call's arg construction.
%
% Three sub-tests:
%   - template inside list arg: findall(X, p(.../2, ..., [X, _]), L)
%   - template inside a compound arg: findall(Y, q(f(Y)), L)
%   - sanity: template as a direct call arg still works (member/2)
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(findall_template_in_struct_arg_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_findall_template_in_struct_arg_via_rscript
    ;   true
    )).

e2e_findall_template_in_struct_arg_via_rscript :-
    % t_list: template inside a list arg of fact_in_range.
    assertz(user:tprice(apple,  5)),
    assertz(user:tprice(banana, 8)),
    assertz(user:tprice(cherry, 12)),
    assertz((user:t_list :-
        findall(Item,
                fact_in_range(tprice/2, 2, 5, 10, [Item, _P]),
                L),
        L == [apple, banana])),
    % t_compound: template inside a compound (non-list) arg.
    % Defines q/1 dynamically so the inner goal compiles as a Call
    % whose first arg is built via put_structure.
    assertz((user:tq(f(red)))),
    assertz((user:tq(f(green)))),
    assertz((user:tq(f(blue)))),
    assertz((user:t_compound :-
        findall(Y, tq(f(Y)), Ys),
        Ys == [red, green, blue])),
    % t_member: template as a direct call arg (sanity, was always OK).
    assertz((user:t_member :-
        findall(X, member(X, [1, 2, 3]), Xs),
        Xs == [1, 2, 3])),
    unique_r_tmp_dir('tmp_r_findall_struct_e2e', TmpDir),
    write_wam_r_project(
        [user:tprice/2, user:tq/1, user:t_list/0, user:t_compound/0,
         user:t_member/0],
        [intern_atoms([apple, banana, cherry, red, green, blue, f])],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [t_list, t_compound, t_member],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for `^/2` existential scope and first
% free-variable grouping in bagof/setof. Asserts 2-arg dynamic
% predicates, then drives bagof(X, Y^p(X,Y), L) and
% bagof(X, p(X,Y), L). The `^/2` wrapper keeps Y existential;
% without it, Y is bound to the selected witness group.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(bagof_setof_existential_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_bagof_setof_existential_via_rscript
    ;   true
    )).

e2e_bagof_setof_existential_via_rscript :-
    % bagof + ^/2 over a 2-arg dynamic predicate. Y is existentially
    % scoped, so all X bindings are collected (with duplicates).
    assertz((user:be_bagof_basic :-
        assertz(pp(1, a)), assertz(pp(1, b)), assertz(pp(2, a)),
        bagof(X, Y^pp(X, Y), L),
        L == [1, 1, 2])),
    % setof + ^/2 dedups and sorts.
    assertz((user:be_setof_basic :-
        assertz(qq(3, x)), assertz(qq(1, y)), assertz(qq(3, z)),
        assertz(qq(2, x)),
        setof(X, Y^qq(X, Y), L),
        L == [1, 2, 3])),
    % Nested existentials: Y^Z^Goal unwraps both layers.
    assertz((user:be_nested_caret :-
        assertz(rr(1, a, p)), assertz(rr(2, b, q)),
        assertz(rr(2, c, r)), assertz(rr(1, d, s)),
        bagof(X, Y^Z^rr(X, Y, Z), L),
        L == [1, 2, 2, 1])),
    % findall + ^/2 (compiled aggregate path). findall already
    % ignores witness scoping; ^/2 must just be transparent.
    assertz((user:be_findall_caret :-
        assertz(ss(1, x)), assertz(ss(2, y)), assertz(ss(3, z)),
        findall(X, Y^ss(X, Y), L),
        L == [1, 2, 3])),
    % Empty bag fails (bagof semantics).
    assertz((user:be_bagof_empty_no :-
        bagof(X, Y^nonexistent(X, Y), _))),
    % Non-existential free witness: first group binds Y and collects
    % only matching X values.
    assertz((user:be_bagof_group_first :-
        assertz(gg(1, a)), assertz(gg(2, a)), assertz(gg(3, b)),
        bagof(X, gg(X, Y), L),
        Y == a,
        L == [1, 2])),
    % setof does the same grouping, then sorts and dedups the selected
    % group's template values.
    assertz((user:be_setof_group_first :-
        assertz(hh(3, a)), assertz(hh(1, a)), assertz(hh(3, a)),
        assertz(hh(2, b)),
        setof(X, hh(X, Y), L),
        Y == a,
        L == [1, 3])),
    % Additional witness groups are enumerated on backtracking.
    assertz((user:be_bagof_group_backtrack :-
        assertz(ii(1, a)), assertz(ii(2, b)),
        bagof(X, ii(X, Y), L),
        Y == b,
        L == [2])),
    assertz((user:be_setof_group_backtrack :-
        assertz(jj(3, a)), assertz(jj(1, a)), assertz(jj(3, a)),
        assertz(jj(2, b)),
        setof(X, jj(X, Y), L),
        Y == b,
        L == [2])),
    % Nested compiled findall around runtime bagof/setof: compound
    % findall template vars must not alias the inner call's A-register
    % temporaries.
    assertz((user:be_findall_bagof_groups :-
        assertz(kk(1, a)), assertz(kk(2, b)),
        findall(Y-L, bagof(X, kk(X, Y), L), Groups),
        Groups == [a-[1], b-[2]])),
    assertz((user:be_findall_setof_groups :-
        assertz(ll(3, a)), assertz(ll(1, a)), assertz(ll(3, a)),
        assertz(ll(2, b)),
        findall(Y-L, setof(X, ll(X, Y), L), Groups),
        Groups == [a-[1, 3], b-[2]])),
    unique_r_tmp_dir('tmp_r_caret_e2e', TmpDir),
    write_wam_r_project(
        [ user:be_bagof_basic/0, user:be_setof_basic/0,
          user:be_nested_caret/0, user:be_findall_caret/0,
          user:be_bagof_empty_no/0, user:be_bagof_group_first/0,
          user:be_setof_group_first/0,
          user:be_bagof_group_backtrack/0,
          user:be_setof_group_backtrack/0,
          user:be_findall_bagof_groups/0,
          user:be_findall_setof_groups/0 ],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [be_bagof_basic, be_setof_basic, be_nested_caret,
           be_findall_caret, be_bagof_group_first,
           be_setof_group_first, be_bagof_group_backtrack,
           be_setof_group_backtrack, be_findall_bagof_groups,
           be_findall_setof_groups],
    No  = [be_bagof_empty_no],
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
% End-to-end Rscript run for structured CLI argument parsing. The
% generated program's main no longer falls back to atom-or-integer
% only -- it parses each argv entry through the runtime's operator-
% precedence parser, so lists, structs, and arithmetic expressions
% reach the predicate as proper WAM terms.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(cli_arg_parser_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_cli_arg_parser_via_rscript
    ;   true
    )).

e2e_cli_arg_parser_via_rscript :-
    % `=/2` is a builtin, so we can call it directly with two args
    % and rely on the parser to produce structurally identical terms
    % on both sides.
    assertz((user:check_eq(X, Y) :- X == Y)),
    % Tests with bound args.
    assertz((user:check_list_3([1, 2, 3]))),
    assertz((user:check_struct(f(a, b)))),
    assertz((user:check_nested(g(h(1), [2, 3])))),
    assertz((user:check_arith_expr(3 + 2 * 2))),  % Not evaluated, just structural.
    unique_r_tmp_dir('tmp_r_cli_e2e', TmpDir),
    write_wam_r_project(
        [ user:check_eq/2,
          user:check_list_3/1,
          user:check_struct/1,
          user:check_nested/1,
          user:check_arith_expr/1 ],
        [intern_atoms([a, b, h])],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    % Each entry: (Predicate/arity, list-of-args, expected stdout).
    % The args go through the new CLI parser and must produce terms
    % that unify with the asserted clause's pattern.
    Cases = [
        % Two atoms via shared parse state -- both X and Y are atoms.
        'check_eq/2'-["foo", "foo"]-"true",
        'check_eq/2'-["foo", "bar"]-"false",
        % Same struct on both sides.
        'check_eq/2'-["f(a, b)", "f(a, b)"]-"true",
        % Different struct -> mismatch.
        'check_eq/2'-["f(a, b)", "f(a, c)"]-"false",
        % List input.
        'check_list_3/1'-["[1, 2, 3]"]-"true",
        'check_list_3/1'-["[1, 2, 4]"]-"false",
        % Compound term.
        'check_struct/1'-["f(a, b)"]-"true",
        'check_struct/1'-["f(b, a)"]-"false",
        % Nested compound + list.
        'check_nested/1'-["g(h(1), [2, 3])"]-"true",
        % Arithmetic expression as a structural term (not evaluated).
        'check_arith_expr/1'-["3 + 2 * 2"]-"true"
    ],
    forall(member(Pred-Args-Expected, Cases), (
        run_rscript_with_args(RDir, Pred, Args, Out),
        (   sub_string(Out, _, _, _, Expected)
        ->  true
        ;   format(user_error,
                   'CLI test FAILED: pred=~w args=~w expected=~w got=~w~n',
                   [Pred, Args, Expected, Out]),
            assertion(false)
        )
    )),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for predicates whose names collide with
% base R functions. The wrapper-name generator prefixes every
% per-predicate wrapper with `pred_` so user predicates named `c`,
% `t`, `q`, `cat`, `paste`, ... don't shadow the runtime's own
% calls to those base functions. Without the prefix, asserting
% `c/2` would replace `base::c` at the top level and the runtime's
% tokenizer (and other paths that build vectors via `c(...)`) would
% crash.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(base_name_clash_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_base_name_clash_via_rscript
    ;   true
    )).

e2e_base_name_clash_via_rscript :-
    assertz((user:c(1, 2))),
    assertz((user:c(2, 3))),
    assertz((user:t(hello))),
    assertz((user:q(forty_two, 42))),
    assertz((user:cat(meow))),
    assertz((user:check_c   :- c(1, 2))),
    assertz((user:check_t   :- t(hello))),
    assertz((user:check_q   :- q(forty_two, X), X =:= 42)),
    assertz((user:check_cat :- cat(meow))),
    % Also invoke the parser path directly (term_to_atom reverse
    % mode walks tokenize_term, which builds vectors via c(...) --
    % this would have crashed with the old wrapper naming when c/2
    % was asserted above).
    assertz((user:check_parser :-
        atom_codes(A, [102, 40, 49, 41]),  % "f(1)"
        term_to_atom(T, A),
        T == f(1))),
    unique_r_tmp_dir('tmp_r_baseclash_e2e', TmpDir),
    write_wam_r_project(
        [user:c/2, user:t/1, user:q/2, user:cat/1,
         user:check_c/0, user:check_t/0, user:check_q/0,
         user:check_cat/0, user:check_parser/0],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [check_c, check_t, check_q, check_cat, check_parser],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for read_term_from_atom/2,3 and clause/2.
% Both build on existing infrastructure: read_term_from_atom shares
% the parser with term_to_atom/2 reverse mode; clause/2 walks the
% dynamic store via clause_iter (same iter-CP shape as retract_iter
% from PR #1900, but without the removal side-effect).
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(read_term_clause_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_read_term_clause_via_rscript
    ;   true
    )).

e2e_read_term_clause_via_rscript :-
    % read_term_from_atom/2: parse a structural term.
    assertz((user:rt_struct :-
        atom_codes(A, [102, 40, 49, 44, 50, 41]),  % "f(1,2)"
        read_term_from_atom(A, T),
        T == f(1, 2))),
    % read_term_from_atom/2: parse an operator expression.
    assertz((user:rt_op :-
        atom_codes(A, [49, 43, 50, 42, 51]),  % "1+2*3"
        read_term_from_atom(A, T),
        T == 1+2*3)),
    % read_term_from_atom/3: options arg is accepted (ignored).
    assertz((user:rt_three :-
        atom_codes(A, [104, 105]),  % "hi"
        read_term_from_atom(A, T, []),
        T == hi)),
    % Garbage input fails.
    assertz((user:rt_fail :-
        atom_codes(A, [102, 40]),  % "f("
        read_term_from_atom(A, _))),
    % clause/2: collect every fact of fact/1.
    assertz((user:cl_collect :-
        assertz(fact(1)), assertz(fact(2)), assertz(fact(3)),
        findall(X, clause(fact(X), true), L),
        L == [1, 2, 3])),
    % clause/2: rule with body. The body in the dynamic store comes
    % out via the second arg.
    assertz((user:cl_body :-
        assertz((rul(X) :- X > 0)),
        clause(rul(_), B),
        B = (_ > 0))),
    % clause/2 unifies head -- selects which clauses match.
    assertz((user:cl_filter :-
        assertz(typed(1, odd)),
        assertz(typed(2, even)),
        assertz(typed(3, odd)),
        findall(X, clause(typed(X, odd), true), Odds),
        Odds == [1, 3])),
    % clause/2 over an unknown predicate fails.
    assertz((user:cl_no :- clause(nope(_), _))),
    % clause/2 doesn't remove anything; the store survives.
    assertz((user:cl_nondestr :-
        assertz(persists(1)), assertz(persists(2)),
        findall(X, clause(persists(X), _), L1),
        L1 == [1, 2],
        % After clause/2, the facts must still be there.
        findall(Y, clause(persists(Y), _), L2),
        L2 == [1, 2])),
    unique_r_tmp_dir('tmp_r_readterm_clause_e2e', TmpDir),
    write_wam_r_project(
        [user:rt_struct/0, user:rt_op/0, user:rt_three/0, user:rt_fail/0,
         user:cl_collect/0, user:cl_body/0, user:cl_filter/0,
         user:cl_no/0, user:cl_nondestr/0],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [rt_struct, rt_op, rt_three,
           cl_collect, cl_body, cl_filter, cl_nondestr],
    No  = [rt_fail, cl_no],
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
% End-to-end Rscript run for DCG (--> notation). DCG rules are
% translated at SWI's term-expansion time (or via
% dcg_translate_rule/2 at runtime) into ordinary clauses with two
% extra difference-list args, so by the time the WAM compiler
% reads them they look like normal predicates and need no special
% handling. The runtime side just needs phrase/2 and phrase/3 to
% bridge the user-level call into the translated /N+2 form.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(dcg_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_dcg_via_rscript
    ;   true
    )).

e2e_dcg_via_rscript :-
    use_module(library(dcg/basics)),
    % Translate a handful of DCG rules at runtime via
    % dcg_translate_rule/2, then assertz the resulting clauses.
    % This proves the WAM target handles the translated form
    % regardless of whether SWI did the translation at consult
    % time or asynchronously.
    dcg_translate_rule((g_greet --> [hello]), Cg),  assertz(user:Cg),
    dcg_translate_rule((g_pair  --> [a], [b]), Cp), assertz(user:Cp),
    % Recursive grammar that emits N..1 down to 0.
    dcg_translate_rule((g_seq(0) --> []), Cs0),     assertz(user:Cs0),
    dcg_translate_rule(
        (g_seq(N) --> {N > 0}, [N], {N1 is N - 1}, g_seq(N1)), Csn),
    assertz(user:Csn),
    % Test predicates that exercise phrase/2 and phrase/3.
    assertz((user:dcg_simple :- phrase(g_greet, [hello]))),
    assertz((user:dcg_pair   :- phrase(g_pair, [a, b]))),
    assertz((user:dcg_seq3   :- phrase(g_seq(3), [3, 2, 1]))),
    assertz((user:dcg_seq0   :- phrase(g_seq(0), []))),
    % phrase/3 with a non-empty rest: parse a prefix, leave the rest.
    assertz((user:dcg_phrase3 :-
        phrase(g_pair, [a, b, x, y], [x, y]))),
    % Negative cases.
    assertz((user:dcg_no   :- phrase(g_pair, [a, c]))),
    assertz((user:dcg_no2  :- phrase(g_seq(2), [2, 1, 0]))),
    unique_r_tmp_dir('tmp_r_dcg_e2e', TmpDir),
    write_wam_r_project(
        [user:g_greet/2, user:g_pair/2, user:g_seq/3,
         user:dcg_simple/0, user:dcg_pair/0, user:dcg_seq3/0,
         user:dcg_seq0/0, user:dcg_phrase3/0,
         user:dcg_no/0, user:dcg_no2/0],
        [intern_atoms([hello, a, b, c, x, y])],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [dcg_simple, dcg_pair, dcg_seq3, dcg_seq0, dcg_phrase3],
    No  = [dcg_no, dcg_no2],
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
% End-to-end Rscript run for stream I/O. Drives a write-then-read
% round trip through a real file: open/3 + writeln/2 + format/3 +
% close/1, then re-open the file for reading and read/2 each term
% back, verifying the parser handles operator notation and EOF
% returns the `end_of_file` atom.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(streams_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_streams_via_rscript
    ;   true
    )).

e2e_streams_via_rscript :-
    unique_r_tmp_dir('tmp_r_streams_e2e', TmpDir),
    directory_file_path(TmpDir, 'stream_data.txt', DataFile),
    atom_string(DataFile, DataFileStr),
    % Driver predicates. write_round writes a few terms (each
    % terminated by a `.` so read/2 can re-parse them); read_round
    % opens for read, pulls terms back, and asserts they unify with
    % the expected forms.
    assertz((user:s_write(File) :-
        open(File, write, S),
        writeln(S, 'fact(1).'),
        writeln(S, 'fact(2).'),
        format(S, 'sum(~w).~n', [42]),
        format(S, 'expr(~w).~n', [1+2*3]),
        close(S))),
    assertz((user:s_read_back(File) :-
        open(File, read, S),
        read(S, T1), T1 == fact(1),
        read(S, T2), T2 == fact(2),
        read(S, T3), T3 == sum(42),
        read(S, T4), T4 == expr(1+2*3),
        read(S, T5), T5 == end_of_file,
        close(S))),
    assertz((user:s_round_trip(File) :- s_write(File), s_read_back(File))),
    % open/3 on a non-existent file with mode=read fails.
    assertz((user:s_open_no(File) :- open(File, read, _))),
    % close/1 on a non-stream arg fails.
    assertz((user:s_close_no :- close(not_a_stream))),
    write_wam_r_project(
        [user:s_write/1, user:s_read_back/1, user:s_round_trip/1,
         user:s_open_no/1, user:s_close_no/0],
        [intern_atoms([fact, sum, expr, end_of_file, '+', '*'])],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    % Path is passed bare; the CLI parser falls back to atom when
    % the slash-tokenised string fails to parse as a Prolog term.
    run_rscript_with_args(RDir, 's_round_trip/1', [DataFileStr], OutOk),
    assertion(sub_string(OutOk, _, _, _, "true")),
    % Bogus path fails the read-mode open.
    run_rscript_with_args(RDir, 's_open_no/1', ["/no/such/path.txt"], OutNo),
    assertion(sub_string(OutNo, _, _, _, "false")),
    run_rscript_query(RDir, 's_close_no/0', OutCloseNo),
    assertion(sub_string(OutCloseNo, _, _, _, "false")),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for read/2 across multi-line clauses.
% Writes a file with terms whose source spans several lines (open
% paren on one line, args on the next, closing `).` on the line
% after) and reads them back, asserting the parsed terms match the
% expected single-line forms. Also covers a clause that contains a
% `.` literal mid-string (the `.` should not be mistaken for the
% clause terminator).
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(streams_multiline_read_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_streams_multiline_via_rscript
    ;   true
    )).

e2e_streams_multiline_via_rscript :-
    unique_r_tmp_dir('tmp_r_streams_mlread', TmpDir),
    directory_file_path(TmpDir, 'multiline.txt', DataFile),
    atom_string(DataFile, DataFileStr),
    % Write a file with three terms:
    %   1. A two-line compound `foo(\n  bar,\n  baz\n).`
    %   2. A single-line term to verify backward compat.
    %   3. A four-line operator term `expr(\n  1 +\n  2 *\n  3\n).`
    assertz((user:s_ml_write(File) :-
        open(File, write, S),
        format(S, 'foo(~n  bar,~n  baz~n).~n', []),
        writeln(S, 'simple(1).'),
        format(S, 'expr(~n  1 +~n  2 *~n  3~n).~n', []),
        close(S))),
    assertz((user:s_ml_read_back(File) :-
        open(File, read, S),
        read(S, T1), T1 == foo(bar, baz),
        read(S, T2), T2 == simple(1),
        read(S, T3), T3 == expr(1+2*3),
        read(S, T4), T4 == end_of_file,
        close(S))),
    assertz((user:s_ml_round_trip(File) :-
        s_ml_write(File), s_ml_read_back(File))),
    write_wam_r_project(
        [user:s_ml_write/1, user:s_ml_read_back/1, user:s_ml_round_trip/1],
        [intern_atoms([foo, bar, baz, simple, expr,
                       end_of_file, '+', '*'])],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    run_rscript_with_args(RDir, 's_ml_round_trip/1', [DataFileStr], Out),
    assertion(sub_string(Out, _, _, _, "true")),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for the fact-table lowering path. Pure-fact
% predicates (every clause is `get_constant + proceed`, no body
% calls) are emitted as a flat R list of arg tuples plus a one-line
% lowered function -- bypassing the WAM stepping engine. A first-
% arg hash index lets ground-arg queries hit a bucket directly
% instead of scanning the full tuple list.
% Also exercises the fact_table_layout(off) escape hatch by
% generating a parallel project that forces the regular compiled
% path and asserting both paths agree on every query.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(fact_table_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_fact_table_via_rscript
    ;   true
    )).

e2e_fact_table_via_rscript :-
    setup_call_cleanup(
        e2e_fact_table_setup(TmpDir),
        e2e_fact_table_body(TmpDir),
        e2e_fact_table_cleanup(TmpDir)).

e2e_fact_table_setup(TmpDir) :-
    assertz(user:edge(a, b)),
    assertz(user:edge(b, c)),
    assertz(user:edge(c, d)),
    assertz(user:edge(d, a)),
    % Driver predicates: cover ground/1, ground/2, both-ground,
    % multi-solution via findall, and a no-match case.
    assertz((user:ft_first   :- edge(a, b))),
    assertz((user:ft_third   :- edge(c, d))),
    assertz((user:ft_no      :- edge(a, c))),
    assertz((user:ft_findall :- findall(X-Y, edge(X, Y), L),
                                 L == [a-b, b-c, c-d, d-a])),
    assertz((user:ft_succ    :- edge(b, X), X == c)),
    % Mixed first-arg type: integers as well as atoms.
    assertz(user:ival(1, one)),
    assertz(user:ival(2, two)),
    assertz(user:ival(3, three)),
    assertz((user:ft_int  :- ival(2, two))),
    assertz((user:ft_int_no :- ival(2, three))),
    unique_r_tmp_dir('tmp_r_facttable_e2e', TmpDir).

e2e_fact_table_body(TmpDir) :-
    write_wam_r_project(
        [user:edge/2, user:ival/2,
         user:ft_first/0, user:ft_third/0, user:ft_no/0,
         user:ft_findall/0, user:ft_succ/0,
         user:ft_int/0, user:ft_int_no/0],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [ft_first, ft_third, ft_findall, ft_succ, ft_int],
    No  = [ft_no, ft_int_no],
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
    % Check that the generator emits the expected fact-table data.
    directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
    read_file_to_string(ProgPath, Code, []),
    assertion(sub_string(Code, _, _, _, '# Fact table for edge/2')),
    assertion(sub_string(Code, _, _, _, 'pred_edge_facts <- list(')),
    % Per-arg indexes (one env per arg position) bundled into a list.
    assertion(sub_string(Code, _, _, _, 'pred_edge_index_arg1 <- new.env')),
    assertion(sub_string(Code, _, _, _, 'pred_edge_index_arg2 <- new.env')),
    assertion(sub_string(Code, _, _, _, 'pred_edge_indexes <- list(pred_edge_index_arg1, pred_edge_index_arg2)')),
    % Both arg envs should be populated: each fact contributes one
    % entry to each arg position's index. 4 facts, 4 distinct atoms
    % at arg1 (a, b, c, d) and 4 distinct atoms at arg2 (b, c, d, a)
    % -> 4 `assign(..., envir = <env>)` lines per env. We don't bind
    % to specific atom-ids since interning order is not stable.
    aggregate_all(count,
        sub_string(Code, _, _, _, ', envir = pred_edge_index_arg1)'),
        Arg1Count),
    aggregate_all(count,
        sub_string(Code, _, _, _, ', envir = pred_edge_index_arg2)'),
        Arg2Count),
    assertion(Arg1Count =:= 4),
    assertion(Arg2Count =:= 4).

e2e_fact_table_cleanup(TmpDir) :-
    retractall(user:edge(_, _)),
    retractall(user:ival(_, _)),
    retractall(user:ft_first),
    retractall(user:ft_third),
    retractall(user:ft_no),
    retractall(user:ft_findall),
    retractall(user:ft_succ),
    retractall(user:ft_int),
    retractall(user:ft_int_no),
    ignore(catch(delete_directory_and_contents(TmpDir), _, true)).

% ------------------------------------------------------------------
% End-to-end Rscript run for multi-arg fact-table indexing.
% A 3-arg edge_w/3 table with deliberately skewed buckets exercises
% the smallest-bucket pick: arg1 has 5 values for one of them and
% just 1 for another, while arg2 / arg3 have different selectivities.
% Drivers query with just arg2 bound, just arg3 bound, and (arg2, arg3)
% bound -- all three should hit the per-arg indexes rather than fall
% back to a full scan. We assert on the answers (correctness) and on
% the emitted index code (the indexes list contains 3 envs).
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(fact_table_multi_arg_index_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_multi_arg_index_via_rscript
    ;   true
    )).

e2e_multi_arg_index_via_rscript :-
    setup_call_cleanup(
        e2e_multi_arg_index_setup(TmpDir),
        e2e_multi_arg_index_body(TmpDir),
        e2e_multi_arg_index_cleanup(TmpDir)).

e2e_multi_arg_index_setup(TmpDir) :-
    % edge_w(Src, Dst, Weight) -- 6 facts. Skew arg1: `a` has 4 outgoing,
    % everyone else has 1. So a query with arg2=z bound reaches 1 fact;
    % a query with arg1=a bound reaches 4. Smallest-bucket pick should
    % use arg2 in those cases.
    assertz(user:edge_w(a, x, 1)),
    assertz(user:edge_w(a, y, 2)),
    assertz(user:edge_w(a, z, 3)),
    assertz(user:edge_w(a, w, 4)),
    assertz(user:edge_w(b, x, 5)),
    assertz(user:edge_w(c, x, 6)),
    % q_arg2_only:  edge_w(_, z, _) -- 1 match (a, z, 3).
    % q_arg3_only:  edge_w(_, _, 5) -- 1 match (b, x, 5).
    % q_arg23:      edge_w(_, x, 6) -- 1 match (c, x, 6).
    % q_arg2_no:    edge_w(_, q, _) -- 0 matches (atom q absent at arg2).
    % q_arg3_no:    edge_w(_, _, 99) -- 0 matches.
    assertz((user:q_arg2_only :- edge_w(_, z, W), W == 3)),
    assertz((user:q_arg3_only :- edge_w(S, _, 5), S == b)),
    assertz((user:q_arg23     :- edge_w(S, x, 6), S == c)),
    assertz((user:q_arg2_no   :- edge_w(_, q, _))),
    assertz((user:q_arg3_no   :- edge_w(_, _, 99))),
    unique_r_tmp_dir('tmp_r_multi_arg_index_e2e', TmpDir).

e2e_multi_arg_index_body(TmpDir) :-
    write_wam_r_project(
        [user:edge_w/3,
         user:q_arg2_only/0, user:q_arg3_only/0,
         user:q_arg23/0,
         user:q_arg2_no/0, user:q_arg3_no/0],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [q_arg2_only, q_arg3_only, q_arg23],
    No  = [q_arg2_no, q_arg3_no],
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
    % Structural: 3 per-arg index envs are emitted and bundled.
    directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
    read_file_to_string(ProgPath, Code, []),
    assertion(sub_string(Code, _, _, _, 'pred_edge_w_index_arg1 <- new.env')),
    assertion(sub_string(Code, _, _, _, 'pred_edge_w_index_arg2 <- new.env')),
    assertion(sub_string(Code, _, _, _, 'pred_edge_w_index_arg3 <- new.env')),
    assertion(sub_string(Code, _, _, _,
        'pred_edge_w_indexes <- list(pred_edge_w_index_arg1, pred_edge_w_index_arg2, pred_edge_w_index_arg3)')).

e2e_multi_arg_index_cleanup(TmpDir) :-
    retractall(user:edge_w(_, _, _)),
    retractall(user:q_arg2_only),
    retractall(user:q_arg3_only),
    retractall(user:q_arg23),
    retractall(user:q_arg2_no),
    retractall(user:q_arg3_no),
    ignore(catch(delete_directory_and_contents(TmpDir), _, true)).

% ------------------------------------------------------------------
% End-to-end Rscript run for the recursive-kernel detector. The
% ancestor/2 predicate matches the transitive_closure2 pattern from
% recursive_kernel_detection.pl, so the codegen swaps its WAM body
% for a native R BFS over the underlying parent/2 fact-table. Tests
% direct hit, multi-hop reach, branch traversal, no-path failure,
% and findall-style enumeration through the new lowered_dispatch
% tier in dispatch_call / Call / Execute.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(kernel_tc2_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_kernel_tc2_via_rscript
    ;   true
    )).

e2e_kernel_tc2_via_rscript :-
    retractall(user:parent_of(_, _)),
    retractall(user:anc(_, _)),
    assertz(user:parent_of(alice, bob)),
    assertz(user:parent_of(bob, carol)),
    assertz(user:parent_of(carol, dan)),
    assertz(user:parent_of(alice, eve)),
    assertz(user:parent_of(eve, frank)),
    assertz((user:anc(X, Y) :- user:parent_of(X, Y))),
    assertz((user:anc(X, Y) :- user:parent_of(X, Z), user:anc(Z, Y))),
    assertz((user:tc_direct  :- anc(alice, bob))),
    assertz((user:tc_deep    :- anc(alice, dan))),
    assertz((user:tc_branch  :- anc(alice, frank))),
    assertz((user:tc_no_back :- anc(bob, alice))),
    assertz((user:tc_disjoint :- anc(eve, carol))),
    assertz((user:tc_findall :-
        findall(Y, anc(alice, Y), L),
        msort(L, S),
        S == [bob, carol, dan, eve, frank])),
    unique_r_tmp_dir('tmp_r_kernel_tc2_e2e', TmpDir),
    write_wam_r_project(
        [user:parent_of/2, user:anc/2,
         user:tc_direct/0, user:tc_deep/0, user:tc_branch/0,
         user:tc_no_back/0, user:tc_disjoint/0, user:tc_findall/0],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [tc_direct, tc_deep, tc_branch, tc_findall],
    No  = [tc_no_back, tc_disjoint],
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
    % The generator should emit the kernel function and register it
    % in the lowered-dispatch env.
    directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
    read_file_to_string(ProgPath, Code, []),
    assertion(sub_string(Code, _, _, _, 'pred_anc_kernel_tc2 <- function(')),
    assertion(sub_string(Code, _, _, _,
        'assign("anc/2", pred_anc_kernel_tc2, envir = shared_program$lowered_dispatch)')),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for the transitive_distance3 kernel. Same
% BFS shape as transitive_closure2 but tracks depth-from-source and
% yields (target, distance) pairs. Direct hits, multi-hop reach,
% wrong-distance failure, branch traversal, and findall over the
% reachable set. The lowered_dispatch tier means internal calls
% (from tc_findall's body) hit the kernel just like top-level CLI.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(kernel_td3_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_kernel_td3_via_rscript
    ;   true
    )).

e2e_kernel_td3_via_rscript :-
    retractall(user:edge(_, _)),
    retractall(user:tdist(_, _, _)),
    assertz(user:edge(a, b)),
    assertz(user:edge(b, c)),
    assertz(user:edge(c, d)),
    assertz(user:edge(a, e)),
    assertz(user:edge(e, f)),
    assertz((user:tdist(X, Y, 1) :- user:edge(X, Y))),
    assertz((user:tdist(X, Y, D) :- user:edge(X, Z), user:tdist(Z, Y, D1),
                                     D is D1 + 1)),
    assertz((user:td_one  :- tdist(a, b, 1))),
    assertz((user:td_two  :- tdist(a, c, 2))),
    assertz((user:td_three :- tdist(a, d, 3))),
    assertz((user:td_branch :- tdist(a, e, 1))),
    assertz((user:td_deep_branch :- tdist(a, f, 2))),
    assertz((user:td_wrong_dist :- tdist(a, c, 1))),
    assertz((user:td_no_path    :- tdist(b, a, _))),
    assertz((user:td_findall :-
        findall(Y-D, tdist(a, Y, D), L),
        msort(L, S),
        S == [b-1, c-2, d-3, e-1, f-2])),
    unique_r_tmp_dir('tmp_r_kernel_td3_e2e', TmpDir),
    write_wam_r_project(
        [user:edge/2, user:tdist/3,
         user:td_one/0, user:td_two/0, user:td_three/0,
         user:td_branch/0, user:td_deep_branch/0,
         user:td_wrong_dist/0, user:td_no_path/0, user:td_findall/0],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [td_one, td_two, td_three, td_branch, td_deep_branch, td_findall],
    No  = [td_wrong_dist, td_no_path],
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
    directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
    read_file_to_string(ProgPath, Code, []),
    assertion(sub_string(Code, _, _, _, 'pred_tdist_kernel_td3 <- function(')),
    assertion(sub_string(Code, _, _, _,
        'assign("tdist/3", pred_tdist_kernel_td3, envir = shared_program$lowered_dispatch)')),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for the weighted_shortest_path3 kernel.
% Dijkstra over a weighted edge predicate of arity 3 -- yields
% (target, total-weight) for the SHORTEST path per reachable node,
% not all paths. Tests cover:
%   * shorter detour beating a single direct heavy edge,
%   * multi-hop sum,
%   * wrong-distance failure (heavy direct edge),
%   * findall over the reachable set with shortest weights.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(kernel_wsp3_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_kernel_wsp3_via_rscript
    ;   true
    )).

e2e_kernel_wsp3_via_rscript :-
    retractall(user:wedge(_, _, _)),
    retractall(user:wsp(_, _, _)),
    % a -1-> b -2-> c -3-> d, plus a heavy direct a-5->c.
    assertz(user:wedge(a, b, 1)),
    assertz(user:wedge(b, c, 2)),
    assertz(user:wedge(c, d, 3)),
    assertz(user:wedge(a, c, 5)),
    assertz((user:wsp(X, Y, W) :- user:wedge(X, Y, W))),
    assertz((user:wsp(X, Y, W) :- user:wedge(X, Z, W1),
                                   user:wsp(Z, Y, W2),
                                   W is W1 + W2)),
    assertz((user:wsp_direct  :- wsp(a, b, 1))),
    assertz((user:wsp_shorter :- wsp(a, c, 3))),       % via b, not 5
    assertz((user:wsp_threehop :- wsp(a, d, 6))),
    assertz((user:wsp_no_heavy :- wsp(a, c, 5))),      % wrong weight
    assertz((user:wsp_no_back  :- wsp(d, a, _))),
    assertz((user:wsp_findall :-
        findall(Y-W, wsp(a, Y, W), L),
        msort(L, S),
        S == [b-1, c-3, d-6])),
    unique_r_tmp_dir('tmp_r_kernel_wsp3_e2e', TmpDir),
    write_wam_r_project(
        [user:wedge/3, user:wsp/3,
         user:wsp_direct/0, user:wsp_shorter/0, user:wsp_threehop/0,
         user:wsp_no_heavy/0, user:wsp_no_back/0, user:wsp_findall/0],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [wsp_direct, wsp_shorter, wsp_threehop, wsp_findall],
    No  = [wsp_no_heavy, wsp_no_back],
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
    directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
    read_file_to_string(ProgPath, Code, []),
    assertion(sub_string(Code, _, _, _, 'pred_wsp_kernel_wsp3 <- function(')),
    assertion(sub_string(Code, _, _, _,
        'assign("wsp/3", pred_wsp_kernel_wsp3, envir = shared_program$lowered_dispatch)')),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for the transitive_parent_distance4
% kernel. BFS that tracks the immediate predecessor of each
% reachable node alongside its distance, yielding (target, parent,
% distance) triples. Direct/multi-hop checks, branch traversal,
% wrong-parent failure, and a findall over the reachable set
% (using a struct template t(Y,P,D) since the WAM compiler doesn't
% accept pair templates like Y-P-D).
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(kernel_tpd4_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_kernel_tpd4_via_rscript
    ;   true
    )).

e2e_kernel_tpd4_via_rscript :-
    retractall(user:pedge(_, _)),
    retractall(user:pd(_, _, _, _)),
    % a -> b -> c -> d, plus branch a -> e -> f.
    assertz(user:pedge(a, b)),
    assertz(user:pedge(b, c)),
    assertz(user:pedge(c, d)),
    assertz(user:pedge(a, e)),
    assertz(user:pedge(e, f)),
    assertz((user:pd(X, Y, X, 1) :- user:pedge(X, Y))),
    assertz((user:pd(X, Y, P, D) :- user:pedge(X, Z),
                                     user:pd(Z, Y, P, D1),
                                     D is D1 + 1)),
    assertz((user:tpd_direct  :- pd(a, b, a, 1))),
    assertz((user:tpd_two     :- pd(a, c, b, 2))),
    assertz((user:tpd_three   :- pd(a, d, c, 3))),
    assertz((user:tpd_branch  :- pd(a, e, a, 1))),
    assertz((user:tpd_branch2 :- pd(a, f, e, 2))),
    assertz((user:tpd_wrong   :- pd(a, c, a, 2))),
    assertz((user:tpd_no_back :- pd(b, a, _, _))),
    assertz((user:tpd_findall :-
        findall(t(Y, P, D), pd(a, Y, P, D), L),
        msort(L, S),
        S == [t(b, a, 1), t(c, b, 2), t(d, c, 3), t(e, a, 1), t(f, e, 2)])),
    unique_r_tmp_dir('tmp_r_kernel_tpd4_e2e', TmpDir),
    write_wam_r_project(
        [user:pedge/2, user:pd/4,
         user:tpd_direct/0, user:tpd_two/0, user:tpd_three/0,
         user:tpd_branch/0, user:tpd_branch2/0,
         user:tpd_wrong/0, user:tpd_no_back/0, user:tpd_findall/0],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [tpd_direct, tpd_two, tpd_three, tpd_branch, tpd_branch2,
           tpd_findall],
    No  = [tpd_wrong, tpd_no_back],
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
    directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
    read_file_to_string(ProgPath, Code, []),
    assertion(sub_string(Code, _, _, _, 'pred_pd_kernel_tpd4 <- function(')),
    assertion(sub_string(Code, _, _, _,
        'assign("pd/4", pred_pd_kernel_tpd4, envir = shared_program$lowered_dispatch)')),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for the transitive_step_parent_distance5
% kernel. BFS that records, for each reachable node, the FIRST
% hop neighbour of source on the path (step) plus immediate
% predecessor (parent) and distance. Yields 4-tuple results via
% kernel_quad_iter. Direct/multi-hop checks, branch traversal,
% wrong-step / wrong-parent failures, and findall over the full
% reachable set with all four slots.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(kernel_tspd5_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_kernel_tspd5_via_rscript
    ;   true
    )).

e2e_kernel_tspd5_via_rscript :-
    retractall(user:sedge(_, _)),
    retractall(user:tspd(_, _, _, _, _)),
    assertz(user:sedge(a, b)),
    assertz(user:sedge(b, c)),
    assertz(user:sedge(c, d)),
    assertz(user:sedge(a, e)),
    assertz(user:sedge(e, f)),
    % Canonical tspd5 shape -- the 3rd head arg in the recursive
    % clause IS the second arg of the edge call (not a separate
    % variable bound later), which is what the detector requires.
    assertz((user:tspd(X, Y, Y, X, 1) :- user:sedge(X, Y))),
    assertz((user:tspd(X, Y, M, P, D) :-
        user:sedge(X, M), user:tspd(M, Y, _, P, D1), D is D1 + 1)),
    % Direct edge: step==target, parent==source, dist=1.
    assertz((user:tspd_direct  :- tspd(a, b, b, a, 1))),
    % Two-hop a->b->c: step=b (first hop), parent=b (predecessor of c), dist=2.
    assertz((user:tspd_two     :- tspd(a, c, b, b, 2))),
    % Three-hop a->b->c->d: step=b, parent=c, dist=3.
    assertz((user:tspd_three   :- tspd(a, d, b, c, 3))),
    % Branch a->e direct: step=e, parent=a, dist=1.
    assertz((user:tspd_branch  :- tspd(a, e, e, a, 1))),
    % Branch two-hop a->e->f: step=e, parent=e, dist=2.
    assertz((user:tspd_branch2 :- tspd(a, f, e, e, 2))),
    % Wrong step: claims step=e for c, but actual is b.
    assertz((user:tspd_no_step :- tspd(a, c, e, b, 2))),
    % Wrong parent.
    assertz((user:tspd_no_par  :- tspd(a, c, b, a, 2))),
    % No reachable.
    assertz((user:tspd_no_back :- tspd(d, a, _, _, _))),
    % findall over the full reachable set.
    assertz((user:tspd_findall :-
        findall(t(Y, S, P, D), tspd(a, Y, S, P, D), L),
        msort(L, Sorted),
        Sorted == [t(b, b, a, 1), t(c, b, b, 2), t(d, b, c, 3),
                    t(e, e, a, 1), t(f, e, e, 2)])),
    unique_r_tmp_dir('tmp_r_kernel_tspd5_e2e', TmpDir),
    write_wam_r_project(
        [user:sedge/2, user:tspd/5,
         user:tspd_direct/0, user:tspd_two/0, user:tspd_three/0,
         user:tspd_branch/0, user:tspd_branch2/0,
         user:tspd_no_step/0, user:tspd_no_par/0, user:tspd_no_back/0,
         user:tspd_findall/0],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [tspd_direct, tspd_two, tspd_three,
           tspd_branch, tspd_branch2, tspd_findall],
    No  = [tspd_no_step, tspd_no_par, tspd_no_back],
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
    directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
    read_file_to_string(ProgPath, Code, []),
    assertion(sub_string(Code, _, _, _, 'pred_tspd_kernel_tspd5 <- function(')),
    assertion(sub_string(Code, _, _, _,
        'assign("tspd/5", pred_tspd_kernel_tspd5, envir = shared_program$lowered_dispatch)')),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for the category_ancestor kernel.
% Hierarchical BFS with explicit depth cap (read from
% user:max_depth/1 at codegen time) and visited-set cycle
% detection. Result layout is tuple(1) -- yields each reached
% ancestor. The Visited and Hops slots in the user-side call are
% inputs / discarded outputs that the native impl doesn't touch.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(kernel_ca_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_kernel_ca_via_rscript
    ;   true
    )).

e2e_kernel_ca_via_rscript :-
    retractall(user:cparent(_, _)),
    retractall(user:cat_anc(_, _, _, _)),
    retractall(user:max_depth(_)),
    % Required by the category_ancestor detector at codegen time.
    assertz(user:max_depth(3)),
    % A small category tree:
    %   animal -> mammal -> dog -> poodle
    %   animal -> fish   -> salmon
    assertz(user:cparent(animal, mammal)),
    assertz(user:cparent(mammal, dog)),
    assertz(user:cparent(dog, poodle)),
    assertz(user:cparent(animal, fish)),
    assertz(user:cparent(fish, salmon)),
    assertz((user:cat_anc(Cat, Anc, Visited, 0) :-
        \+ member(Cat, Visited),
        user:cparent(Cat, Anc))),
    assertz((user:cat_anc(Cat, Anc, Visited, Hops) :-
        \+ member(Cat, Visited),
        user:cparent(Cat, Mid),
        user:cat_anc(Mid, Anc, [Cat | Visited], Hops0),
        Hops is Hops0 + 1)),
    assertz((user:ca_direct  :- cat_anc(animal, mammal, [], _))),
    assertz((user:ca_two     :- cat_anc(animal, dog, [], _))),
    assertz((user:ca_three   :- cat_anc(animal, poodle, [], _))),
    assertz((user:ca_branch  :- cat_anc(animal, salmon, [], _))),
    assertz((user:ca_no_back :- cat_anc(mammal, animal, [], _))),
    assertz((user:ca_findall :-
        findall(A, cat_anc(animal, A, [], _), L),
        msort(L, S),
        S == [dog, fish, mammal, poodle, salmon])),
    unique_r_tmp_dir('tmp_r_kernel_ca_e2e', TmpDir),
    write_wam_r_project(
        [user:cparent/2, user:cat_anc/4,
         user:ca_direct/0, user:ca_two/0, user:ca_three/0,
         user:ca_branch/0, user:ca_no_back/0, user:ca_findall/0],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [ca_direct, ca_two, ca_three, ca_branch, ca_findall],
    No  = [ca_no_back],
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
    directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
    read_file_to_string(ProgPath, Code, []),
    assertion(sub_string(Code, _, _, _, 'pred_cat_anc_kernel_ca <- function(')),
    assertion(sub_string(Code, _, _, _,
        'assign("cat_anc/4", pred_cat_anc_kernel_ca, envir = shared_program$lowered_dispatch)')),
    % max_depth(3) should be embedded into the generated dispatch call.
    assertion(sub_string(Code, _, _, _, '3L, source, ancestor')),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for the astar_shortest_path4 kernel.
% Goal-directed shortest-path search with a user-supplied heuristic
% (read from user:direct_dist_pred/1 at codegen time). Result
% layout is tuple(1) -- single shortest distance. Admissibility of
% the heuristic is the user's responsibility; this test uses
% under-estimates so the path returned is optimal.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(kernel_astar4_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_kernel_astar4_via_rscript
    ;   true
    )).

e2e_kernel_astar4_via_rscript :-
    retractall(user:wedge2(_, _, _)),
    retractall(user:h_dist2(_, _, _)),
    retractall(user:astar(_, _, _, _)),
    retractall(user:direct_dist_pred(_)),
    % a -1-> b -2-> c -3-> d, plus a heavy direct a -5-> c.
    assertz(user:wedge2(a, b, 1)),
    assertz(user:wedge2(b, c, 2)),
    assertz(user:wedge2(c, d, 3)),
    assertz(user:wedge2(a, c, 5)),
    % Heuristic to goal d (admissible: each value <= true cost).
    assertz(user:h_dist2(a, d, 5)),  % true cost is 6
    assertz(user:h_dist2(b, d, 4)),  % true cost is 5
    assertz(user:h_dist2(c, d, 3)),  % exact
    assertz(user:h_dist2(d, d, 0)),
    assertz(user:direct_dist_pred(h_dist2/3)),
    % Canonical astar shape: 4-arity with Dim passthrough.
    assertz((user:astar(X, Y, _, W) :- user:wedge2(X, Y, W))),
    assertz((user:astar(X, Y, D, W) :-
        user:wedge2(X, Z, W1),
        user:astar(Z, Y, D, RestW),
        W is W1 + RestW)),
    assertz((user:astar_direct  :- astar(a, b, 5, 1))),
    assertz((user:astar_shorter :- astar(a, c, 5, 3))),     % via b, not 5
    assertz((user:astar_three   :- astar(a, d, 5, 6))),
    assertz((user:astar_no_heavy :- astar(a, c, 5, 5))),     % wrong cost
    assertz((user:astar_no_back :- astar(d, a, 5, _))),
    unique_r_tmp_dir('tmp_r_kernel_astar4_e2e', TmpDir),
    write_wam_r_project(
        [user:wedge2/3, user:h_dist2/3, user:astar/4,
         user:astar_direct/0, user:astar_shorter/0, user:astar_three/0,
         user:astar_no_heavy/0, user:astar_no_back/0],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [astar_direct, astar_shorter, astar_three],
    No  = [astar_no_heavy, astar_no_back],
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
    directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
    read_file_to_string(ProgPath, Code, []),
    assertion(sub_string(Code, _, _, _, 'pred_astar_kernel_astar4 <- function(')),
    assertion(sub_string(Code, _, _, _,
        'assign("astar/4", pred_astar_kernel_astar4, envir = shared_program$lowered_dispatch)')),
    % Heuristic-pred name should appear in the generated dispatch call.
    assertion(sub_string(Code, _, _, _, '"h_dist2", "h_dist2/3"')),
    delete_directory_and_contents(TmpDir).

% ------------------------------------------------------------------
% End-to-end Rscript run for external fact sources via CSV. The
% predicate has no Prolog clauses; the codegen emits a runtime
% loader that reads the file at program-init time and dispatches
% via the same fact_table_dispatch path used by inline fact tables
% (PR #1921). The CLI hits the loader through a 1-instruction
% Execute body that falls into lowered_dispatch.
% Auto-skips when Rscript is not on PATH.
% ------------------------------------------------------------------
test(external_fact_source_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_external_fact_source_via_rscript
    ;   true
    )).

e2e_external_fact_source_via_rscript :-
    retractall(user:cpedge(_, _)),
    retractall(user:fs_check/0),
    unique_r_tmp_dir('tmp_r_csv_fact_e2e', TmpDir),
    make_directory_path(TmpDir),
    directory_file_path(TmpDir, 'cpedge.csv', CsvPath),
    atom_string(CsvPath, CsvPathStr),
    setup_call_cleanup(
        open(CsvPath, write, Stream),
        ( write(Stream, '# Auto-generated for the CSV fact-source test.'),
          nl(Stream),
          write(Stream, 'alice,bob'), nl(Stream),
          write(Stream, 'bob,carol'), nl(Stream),
          write(Stream, 'carol,dan'), nl(Stream),
          write(Stream, 'alice,eve'), nl(Stream),
          write(Stream, 'eve,frank'), nl(Stream) ),
        close(Stream)),
    % cpedge/2 has no clauses; the loader populates pred_cpedge_facts.
    assertz((user:fs_direct  :- cpedge(alice, bob))),
    assertz((user:fs_branch  :- cpedge(alice, eve))),
    assertz((user:fs_no_back :- cpedge(bob, alice))),
    assertz((user:fs_findall :-
        findall(Y, cpedge(alice, Y), L),
        msort(L, S),
        S == [bob, eve])),
    assertz((user:fs_findall_all :-
        findall(X-Y, cpedge(X, Y), L),
        length(L, N),
        N == 5)),
    write_wam_r_project(
        [user:cpedge/2, user:fs_direct/0, user:fs_branch/0,
         user:fs_no_back/0, user:fs_findall/0, user:fs_findall_all/0],
        [intern_atoms([alice, bob, carol, dan, eve, frank]),
         r_fact_sources([source(cpedge/2, file(CsvPathStr))])],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [fs_direct, fs_branch, fs_findall, fs_findall_all],
    No  = [fs_no_back],
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
    directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
    read_file_to_string(ProgPath, Code, []),
    assertion(sub_string(Code, _, _, _, '# External fact source for cpedge/2')),
    assertion(sub_string(Code, _, _, _,
        'pred_cpedge_facts <- WamRuntime$read_facts_csv(')),
    assertion(sub_string(Code, _, _, _,
        'assign("cpedge/2", pred_cpedge_fact_iter, envir = shared_program$lowered_dispatch)')),
    delete_directory_and_contents(TmpDir).

% End-to-end Rscript run for runtime postfix-operator support. Covers
% op/3 accepting xf/yf without throwing, the parser wrapping a primary
% as a postfix struct, mixed infix+postfix in one expression, yf
% chaining, current_op/3 enumeration, and op(0, ...) removal. Uses
% =.. to decompose terms so SWI doesn't need the operator declared.
test(op_3_postfix_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_op_3_postfix_via_rscript
    ;   true
    )).

e2e_op_3_postfix_via_rscript :-
    retractall(user:op_postfix_yf/0),
    retractall(user:op_postfix_xf/0),
    retractall(user:op_postfix_yf_chain/0),
    retractall(user:op_postfix_with_infix/0),
    retractall(user:op_postfix_current/0),
    retractall(user:op_postfix_remove/0),
    assertz((user:op_postfix_yf :-
        op(100, yf, '!'),
        read_term_from_atom('5!', T),
        T =.. [F, A],
        F == '!', A == 5)),
    assertz((user:op_postfix_xf :-
        op(100, xf, '!'),
        read_term_from_atom('5!', T),
        T =.. [F, A],
        F == '!', A == 5)),
    % yf permits the operand at the op's own precedence, so `5!!`
    % parses as `'!'('!'(5))`. The parser keeps wrapping while the
    % postfix entry matches.
    assertz((user:op_postfix_yf_chain :-
        op(100, yf, '!'),
        read_term_from_atom('5!!', T),
        T =.. [F, Inner],
        F == '!',
        Inner =.. [F, A],
        A == 5)),
    % Mixed: postfix binds tighter than the surrounding infix, so
    % `5! + 3` parses as `+('!'(5), 3)`.
    assertz((user:op_postfix_with_infix :-
        op(100, yf, '!'),
        read_term_from_atom('5! + 3', T),
        T =.. ['+', L, R],
        L =.. [F, A],
        F == '!', A == 5, R == 3)),
    assertz((user:op_postfix_current :-
        op(100, yf, '!'),
        findall(Type, current_op(100, Type, '!'), Types),
        Types == [yf])),
    % op(0, yf, ...) removes the postfix entry; subsequent parse fails.
    assertz((user:op_postfix_remove :-
        op(100, yf, '!'),
        op(0, yf, '!'),
        \+ read_term_from_atom('5!', _))),
    unique_r_tmp_dir('tmp_r_op_3_postfix_e2e', TmpDir),
    write_wam_r_project(
        [user:op_postfix_yf/0, user:op_postfix_xf/0,
         user:op_postfix_yf_chain/0, user:op_postfix_with_infix/0,
         user:op_postfix_current/0, user:op_postfix_remove/0],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [op_postfix_yf, op_postfix_xf, op_postfix_yf_chain,
           op_postfix_with_infix, op_postfix_current,
           op_postfix_remove],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    delete_directory_and_contents(TmpDir).

test(external_fact_source_grouped_tsv_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_external_fact_source_grouped_tsv_via_rscript
    ;   true
    )).

e2e_external_fact_source_grouped_tsv_via_rscript :-
    retractall(user:gpedge(_, _)),
    unique_r_tmp_dir('tmp_r_grouped_tsv_fact_e2e', TmpDir),
    make_directory_path(TmpDir),
    directory_file_path(TmpDir, 'gpedge.tsv', TsvPath),
    atom_string(TsvPath, TsvPathStr),
    % grouped-by-first shape: <key>\t<v1>\t<v2>...; the loader explodes
    % each row into multiple (key, vK) tuples. The 'alice' row tests
    % multi-value rows; the 'eve' row tests a single-value row;
    % '# comment' and the blank line test the skip rules.
    setup_call_cleanup(
        open(TsvPath, write, Stream),
        ( write(Stream, '# Auto-generated for the grouped-by-first TSV test.'),
          nl(Stream),
          write(Stream, 'alice\tbob\tcarol\teve'), nl(Stream),
          write(Stream, ''), nl(Stream),
          write(Stream, 'bob\tdan'), nl(Stream),
          write(Stream, 'eve\tfrank'), nl(Stream) ),
        close(Stream)),
    assertz((user:gs_direct  :- gpedge(alice, bob))),
    assertz((user:gs_third   :- gpedge(alice, eve))),
    assertz((user:gs_no_back :- gpedge(bob, alice))),
    assertz((user:gs_findall :-
        findall(Y, gpedge(alice, Y), L),
        msort(L, S),
        S == [bob, carol, eve])),
    assertz((user:gs_findall_all :-
        findall(X-Y, gpedge(X, Y), L),
        length(L, N),
        N == 5)),
    write_wam_r_project(
        [user:gpedge/2, user:gs_direct/0, user:gs_third/0,
         user:gs_no_back/0, user:gs_findall/0, user:gs_findall_all/0],
        [intern_atoms([alice, bob, carol, dan, eve, frank]),
         r_fact_sources([source(gpedge/2,
                                grouped_by_first(TsvPathStr))])],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [gs_direct, gs_third, gs_findall, gs_findall_all],
    No  = [gs_no_back],
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
    directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
    read_file_to_string(ProgPath, Code, []),
    assertion(sub_string(Code, _, _, _,
        '# External fact source for gpedge/2 (grouped-by-first tsv file:')),
    assertion(sub_string(Code, _, _, _,
        'pred_gpedge_facts <- WamRuntime$read_facts_grouped_tsv(')),
    assertion(sub_string(Code, _, _, _,
        'assign("gpedge/2", pred_gpedge_fact_iter, envir = shared_program$lowered_dispatch)')),
    delete_directory_and_contents(TmpDir).

% End-to-end Rscript run for the LMDB external fact-source backend.
% Step-1 semantics (load-everything): the runtime reads all key/value
% pairs from the LMDB env at program-load time and feeds them through
% the same build_fact_indexes + fact_table_dispatch pipeline as the
% inline / CSV / grouped-TSV backends, so per-arg indexing and
% backtracking behaviour are identical. Step-2 (probe-on-demand) is
% tracked as a follow-up; see docs/handoff/wam_r_session_handoff.md
% item #1.
%
% Auto-skips when Rscript or an R LMDB binding (thor / lmdbr) isn't
% installed. Documented install steps live in docs/WAM_R_TARGET.md.
test(external_fact_source_lmdb_e2e_rscript) :-
    once((
        rscript_available,
        r_lmdb_pkg_available
    ->  e2e_external_fact_source_lmdb_via_rscript
    ;   true
    )).

% Succeeds when an R LMDB binding (thor or lmdbr) is installed in the
% Rscript environment. Used as an auto-skip guard for the LMDB e2e
% test so the suite still passes on machines without liblmdb / the
% R wrapper.
r_lmdb_pkg_available :-
    catch((
        process_create(path('Rscript'),
                       ['-e',
                        'q(status = if (requireNamespace("thor", quietly = TRUE) || requireNamespace("lmdbr", quietly = TRUE)) 0L else 1L)'],
                       [ stdout(null), stderr(null), process(PID) ]),
        process_wait(PID, exit(0))
    ), _, fail).

e2e_external_fact_source_lmdb_via_rscript :-
    retractall(user:lpedge(_, _)),
    unique_r_tmp_dir('tmp_r_lmdb_fact_e2e', TmpDir),
    make_directory_path(TmpDir),
    directory_file_path(TmpDir, 'lpedge.lmdb', LmdbPath),
    atom_string(LmdbPath, LmdbPathStr),
    % Seed the LMDB env via a small Rscript that writes the tab-encoded
    % `tag:payload` values matching read_facts_lmdb's decoder. The five
    % tuples mirror the CSV / grouped-TSV e2e tests so the assertions
    % can stay analogous.
    seed_lmdb_for_test(TmpDir, LmdbPath,
        ["a:alice\ta:bob",
         "a:bob\ta:carol",
         "a:carol\ta:dan",
         "a:alice\ta:eve",
         "a:eve\ta:frank"]),
    % lpedge/2 has no clauses; the loader populates pred_lpedge_facts.
    assertz((user:ls_direct  :- lpedge(alice, bob))),
    assertz((user:ls_branch  :- lpedge(alice, eve))),
    assertz((user:ls_no_back :- lpedge(bob, alice))),
    assertz((user:ls_findall :-
        findall(Y, lpedge(alice, Y), L),
        msort(L, S),
        S == [bob, eve])),
    assertz((user:ls_findall_all :-
        findall(X-Y, lpedge(X, Y), L),
        length(L, N),
        N == 5)),
    write_wam_r_project(
        [user:lpedge/2, user:ls_direct/0, user:ls_branch/0,
         user:ls_no_back/0, user:ls_findall/0, user:ls_findall_all/0],
        [intern_atoms([alice, bob, carol, dan, eve, frank]),
         r_fact_sources([source(lpedge/2, lmdb(LmdbPathStr))])],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [ls_direct, ls_branch, ls_findall, ls_findall_all],
    No  = [ls_no_back],
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
    directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
    read_file_to_string(ProgPath, Code, []),
    assertion(sub_string(Code, _, _, _,
        '# External fact source for lpedge/2 (lmdb env:')),
    assertion(sub_string(Code, _, _, _,
        'pred_lpedge_facts <- WamRuntime$read_facts_lmdb(')),
    assertion(sub_string(Code, _, _, _,
        'assign("lpedge/2", pred_lpedge_fact_iter, envir = shared_program$lowered_dispatch)')),
    delete_directory_and_contents(TmpDir).

% Writes a tiny R script that opens an LMDB env at LmdbPath via thor
% and puts each Value at a key "kN". Order doesn't matter -- the
% load-everything reader iterates all keys and feeds them through
% build_fact_indexes. Used by the LMDB e2e test.
seed_lmdb_for_test(TmpDir, LmdbPath, EncodedTuples) :-
    directory_file_path(TmpDir, 'seed_lmdb.R', SeedScript),
    atom_string(LmdbPath, LmdbPathStr),
    findall(PutLine,
            (   nth1(I, EncodedTuples, Tuple),
                format(string(KeyLit), '"k~d"', [I]),
                r_double_quoted_literal(Tuple, TupleLit),
                format(string(PutLine), 'env$put(~w, ~w)',
                       [KeyLit, TupleLit])
            ),
            PutLines),
    atomic_list_concat(PutLines, '\n', PutBlock),
    r_double_quoted_literal(LmdbPathStr, LmdbPathLit),
    format(string(R),
'library(thor)
env <- thor::mdb_env(~w, create = TRUE)
~w
env$close()
',
           [LmdbPathLit, PutBlock]),
    setup_call_cleanup(
        open(SeedScript, write, S),
        write(S, R),
        close(S)),
    process_create(path('Rscript'), [SeedScript],
                   [ stdout(null), stderr(null), process(PID) ]),
    process_wait(PID, exit(0)).

% Minimal double-quoted-R-string escaper (test-only). The strings we
% pass in are TmpDir paths and tab-encoded ASCII; the only chars that
% need attention are backslash and double-quote. Tab characters are
% passed through as literal "\t" via the `\\t` substitution so they
% survive the file write -> Rscript read trip as actual tabs.
r_double_quoted_literal(Str, Lit) :-
    string_chars(Str, Chars),
    r_escape_chars(Chars, Escaped),
    string_chars(Body, Escaped),
    format(string(Lit), '"~w"', [Body]).

r_escape_chars([], []).
r_escape_chars([C | Rest], Out) :-
    (   C == '"'  -> Out = ['\\', '"' | OutRest]
    ;   C == '\\' -> Out = ['\\', '\\' | OutRest]
    ;   C == '\t' -> Out = ['\\', 't'  | OutRest]
    ;   C == '\n' -> Out = ['\\', 'n'  | OutRest]
    ;   Out = [C | OutRest]
    ),
    r_escape_chars(Rest, OutRest).

% ------------------------------------------------------------------
% End-to-end: cut barrier truncates state$cps back to the depth at the
% predicate's call site, so a `!` in a clause body that has just
% returned from a multi-clause helper drops both the helper's leftover
% CP and the predicate's own try-chain CP. Pre-fix, `!/0` only popped
% the most-recent CP, leaving alternatives alive whenever any inner
% multi-clause goal sat between clause-head match and the cut.
% Auto-skips when Rscript isn't on PATH.
test(cut_barrier_after_helper_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_cut_barrier_after_helper_via_rscript
    ;   true
    )).

e2e_cut_barrier_after_helper_via_rscript :-
    retractall(user:cb_helper/1),
    retractall(user:cb_caller/1),
    retractall(user:cb_drv/0),
    % cb_helper has three clauses, all succeed via head match. Its
    % TryMeElse CP stays alive after clause 1 returns, exposing the
    % old "drop-topmost" cut bug.
    assertz(user:cb_helper(1)),
    assertz(user:cb_helper(2)),
    assertz(user:cb_helper(3)),
    % cb_caller's clause 1 cut must commit to "selected 1": with proper
    % cut-barrier scope it drops both cb_helper's CP and cb_caller's
    % own try-chain CP, so a downstream backtrack (`fail`) must not
    % land on cb_caller's clause 2 (the fallback) or re-enter the
    % helper. Pre-fix, the runtime would print "selected 1" then fall
    % through to "fallback" (or worse).
    assertz((user:cb_caller(X) :-
        cb_helper(X),
        !,
        write('selected '), write(X), nl)),
    assertz((user:cb_caller(_) :-
        write('fallback'), nl)),
    % Driver: collect every solution and assert the result is a
    % single "selected 1" line. No fallback, no second helper value.
    assertz((user:cb_drv :-
        ( cb_caller(_), fail ; true ))),
    unique_r_tmp_dir('tmp_r_cut_barrier_e2e', TmpDir),
    write_wam_r_project([user:cb_helper/1, user:cb_caller/1, user:cb_drv/0],
                        [], TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    run_rscript_query(RDir, 'cb_drv/0', Out),
    assertion(sub_string(Out, _, _, _, "selected 1")),
    assertion(\+ sub_string(Out, _, _, _, "selected 2")),
    assertion(\+ sub_string(Out, _, _, _, "selected 3")),
    assertion(\+ sub_string(Out, _, _, _, "fallback")),
    assertion(sub_string(Out, _, _, _, "true")),
    delete_directory_and_contents(TmpDir).

% End-to-end: each predicate's clauses Allocate per-clause (the WAM
% emit's standard shape). Failed clauses leave stale env frames on
% state$stack; before the fix, an outer Deallocate later popped a
% stale frame and restored the wrong continuation pointer, silently
% re-running post-call code. Backtrack now truncates state$stack
% back to the depth recorded at the CP's TryMeElse, dropping
% stale frames before the next clause's Allocate.
test(stack_frame_cleanup_on_backtrack_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_stack_frame_cleanup_via_rscript
    ;   true
    )).

e2e_stack_frame_cleanup_via_rscript :-
    retractall(user:sf_pick/2),
    retractall(user:sf_drv/0),
    % sf_pick has three clauses each with a body (forces Allocate
    % per clause). Clauses 1 and 2 reject via head guard; clause 3
    % matches. Pre-fix, clauses 1 and 2 each leave a stale frame
    % whose saved cp later mis-targets the runner.
    assertz((user:sf_pick(a, X) :- X is 1)),
    assertz((user:sf_pick(b, X) :- X is 2)),
    assertz((user:sf_pick(c, X) :- X is 3)),
    assertz((user:sf_drv :-
        sf_pick(c, V),
        write('value='), write(V), nl)),
    unique_r_tmp_dir('tmp_r_stack_cleanup_e2e', TmpDir),
    write_wam_r_project([user:sf_pick/2, user:sf_drv/0], [], TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    run_rscript_query(RDir, 'sf_drv/0', Out),
    assertion(sub_string(Out, _, _, _, "value=3")),
    assertion(sub_string(Out, _, _, _, "true")),
    % Single line (post-frame-cleanup the runtime doesn't loop into
    % stale post-call code).
    split_string(Out, "\n", "", Lines),
    include([L]>>( sub_string(L, _, _, _, "value=") ), Lines, ValueLines),
    assertion(length(ValueLines, 1)),
    delete_directory_and_contents(TmpDir).

% End-to-end: `( A -> B ; C )` soft-cut commits past CPs that A's
% evaluation left alive. Pre-fix CutIte popped only the topmost CP,
% which silently picked up an inner Cond CP instead of the if-then-
% else's own choice point -- so a downstream `fail` would re-enter
% A and rerun B. With the fix, mark_ite_try_me_else tags the
% if-then-else CP and CutIte truncates state$cps back to that CP's
% pre-push depth.
test(cut_ite_barrier_after_helper_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_cut_ite_barrier_after_helper_via_rscript
    ;   true
    )).

e2e_cut_ite_barrier_after_helper_via_rscript :-
    retractall(user:ci_pick/1),
    retractall(user:ci_caller/1),
    retractall(user:ci_drv/0),
    % ci_pick is multi-clause and leaves a CP alive after returning
    % the first solution. Without proper soft-cut barrier, that CP
    % sits between CutIte and the if-then-else's own CP, so the
    % naive "drop topmost" semantic kills the wrong one.
    assertz(user:ci_pick(1)),
    assertz(user:ci_pick(2)),
    assertz(user:ci_pick(3)),
    % If ci_pick succeeds (it does), commit to the "then" branch.
    % "else" should never run after a successful Cond; a downstream
    % `fail` should not re-enter Cond either.
    assertz((user:ci_caller(X) :-
        ( ci_pick(X) -> write('then '), write(X), nl
        ; write('else'), nl ))),
    assertz((user:ci_drv :-
        ( ci_caller(_), fail ; true ))),
    unique_r_tmp_dir('tmp_r_cut_ite_barrier_e2e', TmpDir),
    write_wam_r_project([user:ci_pick/1, user:ci_caller/1, user:ci_drv/0],
                        [], TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    run_rscript_query(RDir, 'ci_drv/0', Out),
    assertion(sub_string(Out, _, _, _, "then 1")),
    assertion(\+ sub_string(Out, _, _, _, "then 2")),
    assertion(\+ sub_string(Out, _, _, _, "then 3")),
    assertion(\+ sub_string(Out, _, _, _, "else")),
    assertion(sub_string(Out, _, _, _, "true")),
    delete_directory_and_contents(TmpDir).

% End-to-end: chained `(A -> B ; C -> D ; E)` if-then-else compiles
% as nested cut_ite/try_me_else_ite pairs, so each branch dispatches
% on its own Cond. Pre-fix, only the outermost `(A -> B ; rest)` was
% recognised as an if-then-else; the inner `(C -> D ; E)` in Else
% position emitted as `Call("->", 2)` (no runtime implementation),
% so any input that should land on a non-first branch silently
% failed.
test(nested_if_then_else_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_nested_if_then_else_via_rscript
    ;   true
    )).

e2e_nested_if_then_else_via_rscript :-
    retractall(user:nite_classify/2),
    retractall(user:nite_drv_one/0),
    retractall(user:nite_drv_two/0),
    retractall(user:nite_drv_three/0),
    retractall(user:nite_drv_other/0),
    % Three-way chain: matches arg 1 against 1, 2, 3 in order; falls
    % through to "other". Each branch sets a distinct atom in arg 2.
    assertz((user:nite_classify(X, Tag) :-
        ( X =:= 1 -> Tag = one
        ; X =:= 2 -> Tag = two
        ; X =:= 3 -> Tag = three
        ; Tag = other ))),
    assertz((user:nite_drv_one   :- nite_classify(1, T), write(T), nl)),
    assertz((user:nite_drv_two   :- nite_classify(2, T), write(T), nl)),
    assertz((user:nite_drv_three :- nite_classify(3, T), write(T), nl)),
    assertz((user:nite_drv_other :- nite_classify(7, T), write(T), nl)),
    unique_r_tmp_dir('tmp_r_nested_ite_e2e', TmpDir),
    write_wam_r_project([user:nite_classify/2,
                         user:nite_drv_one/0, user:nite_drv_two/0,
                         user:nite_drv_three/0, user:nite_drv_other/0],
                        [], TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    run_rscript_query(RDir, 'nite_drv_one/0', Out1),
    assertion(sub_string(Out1, _, _, _, "one")),
    run_rscript_query(RDir, 'nite_drv_two/0', Out2),
    assertion(sub_string(Out2, _, _, _, "two")),
    run_rscript_query(RDir, 'nite_drv_three/0', Out3),
    assertion(sub_string(Out3, _, _, _, "three")),
    run_rscript_query(RDir, 'nite_drv_other/0', Out4),
    assertion(sub_string(Out4, _, _, _, "other")),
    delete_directory_and_contents(TmpDir).

% Compile-time: deeply-nested if-then-else bodies don't blow the
% clause_body_analysis stack. Pre-fix `disjunction_alternatives/2`
% unified an unbound goal with `(Left ; Right)`, binding Left and
% Right to fresh unbounds and recursing into them indefinitely.
% The nonvar guard makes the function safe to call with any term;
% as a side benefit the parser's natural-form `parse_op_loop` /
% `parse_list_elems` bodies (triple-nested ITE) compile cleanly.
% End-to-end: the runtime parser (WamRuntime$wam_parse_expr) enforces
% strict (xf / xfx / xfy / fx) vs non-strict (yf / yfx / fy) lhs-prec
% rules. yf / fy permit chaining `5!!` / `neg neg foo`; xf / fx
% reject the second application because the operand precedence is
% required to be strictly less than the operator precedence.
test(strict_xf_chain_fails_e2e_rscript) :-
    once((
        rscript_available
    ->  e2e_strict_xf_chain_fails_via_rscript
    ;   true
    )).

e2e_strict_xf_chain_fails_via_rscript :-
    retractall(user:strict_xf_single_ok/0),
    retractall(user:strict_xf_chain_fails/0),
    retractall(user:strict_fx_chain_fails/0),
    retractall(user:nonstrict_yf_chain_ok/0),
    retractall(user:nonstrict_fy_chain_ok/0),
    % yf permits chaining (5!! parses to !(!(5))).
    assertz((user:nonstrict_yf_chain_ok :-
        op(100, yf, '!'),
        read_term_from_atom('5!!', T),
        T =.. [F, Inner],
        F == '!',
        Inner =.. [F, A],
        A == 5)),
    % xf single application still works.
    assertz((user:strict_xf_single_ok :-
        op(100, xf, '!'),
        read_term_from_atom('5!', T),
        T =.. [F, A],
        F == '!', A == 5)),
    % xf chained: parser refuses 5!! because the inner !(5) has prec
    % 100 and the outer ! requires its operand at < 100.
    assertz((user:strict_xf_chain_fails :-
        op(100, xf, '!'),
        \+ read_term_from_atom('5!!', _))),
    % fy permits chaining; fx rejects.
    assertz((user:nonstrict_fy_chain_ok :-
        op(900, fy, neg),
        read_term_from_atom('neg neg foo', T),
        T =.. [F, Inner],
        F == neg,
        Inner =.. [F, A],
        A == foo)),
    assertz((user:strict_fx_chain_fails :-
        op(900, fx, neg),
        \+ read_term_from_atom('neg neg foo', _))),
    unique_r_tmp_dir('tmp_r_strict_xf_chain_e2e', TmpDir),
    write_wam_r_project(
        [user:nonstrict_yf_chain_ok/0, user:strict_xf_single_ok/0,
         user:strict_xf_chain_fails/0, user:nonstrict_fy_chain_ok/0,
         user:strict_fx_chain_fails/0],
        [],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir),
    Yes = [nonstrict_yf_chain_ok, strict_xf_single_ok,
           strict_xf_chain_fails, nonstrict_fy_chain_ok,
           strict_fx_chain_fails],
    forall(member(P, Yes), (
        format(string(Q), '~w/0', [P]),
        run_rscript_query(RDir, Q, Out),
        assertion(sub_string(Out, _, _, _, "true"))
    )),
    delete_directory_and_contents(TmpDir).

test(deeply_nested_ite_compiles) :-
    retractall(user:dni_chain/1),
    % Triple-nested if-then-else. The exact shape that previously
    % stack-overflowed clause_body_analysis at compile time.
    assertz((user:dni_chain(X) :-
        ( integer(X), X > 0
        -> Y = pos_int
        ; integer(X), X < 0
        -> Y = neg_int
        ; atom(X)
        -> Y = atom
        ; Y = other
        ),
        write(Y), nl)),
    set_prolog_flag(stack_limit, 67_108_864),
    wam_target:compile_predicate_to_wam(user:dni_chain/1, [], Code),
    string_length(Code, CodeLen),
    assertion(CodeLen > 0),
    retractall(user:dni_chain/1).

% ------------------------------------------------------------------
% Test 8: r_wam bindings module loads
% ------------------------------------------------------------------
test(r_wam_bindings_loads) :-
    assertion(current_predicate(r_wam_bindings:init_r_wam_bindings/0)),
    assertion(current_predicate(r_wam_bindings:r_wam_binding/5)).

test(r_wam_bindings_list_parity) :-
    assertion(r_wam_bindings:r_wam_binding(nth0/3, _, _, _, _)),
    assertion(r_wam_bindings:r_wam_binding(nth1/3, _, _, _, _)),
    assertion(r_wam_bindings:r_wam_binding(is_list/1, _, _, _, _)),
    assertion(r_wam_bindings:r_wam_binding(ground/1, _, _, _, _)).

:- end_tests(wam_r_generator).

% ------------------------------------------------------------------
% Helpers
% ------------------------------------------------------------------
unique_r_tmp_dir(Prefix, Dir) :-
    r_tmp_root(Root),
    get_time(Now),
    StampF is Now,
    format(atom(Base), '~w_~w', [Prefix, StampF]),
    directory_file_path(Root, Base, Dir),
    (   exists_directory(Dir)
    ->  delete_directory_and_contents(Dir)
    ;   true
    ).

r_tmp_root(Root) :-
    getenv('TMPDIR', EnvRoot),
    exists_directory(EnvRoot),
    access_file(EnvRoot, write),
    !,
    Root = EnvRoot.
r_tmp_root('/data/data/com.termux/files/usr/tmp') :-
    exists_directory('/data/data/com.termux/files/usr/tmp'),
    access_file('/data/data/com.termux/files/usr/tmp', write),
    !.
r_tmp_root('/tmp').
