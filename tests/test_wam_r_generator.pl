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
:- use_module('../src/unifyweaver/targets/wam_r_target').
:- use_module('../src/unifyweaver/targets/wam_r_lowered_emitter').
:- use_module('../src/unifyweaver/bindings/r_wam_bindings').

:- begin_tests(wam_r_generator).

:- dynamic user:wam_r_fact/1.
:- dynamic user:wam_r_choice_fact/1.
:- dynamic user:wam_r_caller/1.

user:wam_r_fact(a).
user:wam_r_choice_fact(a).
user:wam_r_choice_fact(b).
user:wam_r_choice_fact(c).
user:wam_r_caller(X) :- user:wam_r_fact(X).

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
% Test 7: lowered emitter scaffold is loadable and refuses lowering
% ------------------------------------------------------------------
test(lowered_emitter_phase1_stub) :-
    assertion(current_predicate(wam_r_lowered_emitter:wam_r_lowerable/3)),
    assertion(current_predicate(wam_r_lowered_emitter:lower_predicate_to_r/4)),
    assertion(current_predicate(wam_r_lowered_emitter:r_lowered_func_name/2)),
    % Phase 1 stub: must always fail.
    \+ wam_r_lowered_emitter:wam_r_lowerable(user:wam_r_fact/1, "", _Reason).

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
