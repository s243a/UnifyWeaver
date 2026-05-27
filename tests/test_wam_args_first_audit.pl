:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_wam_args_first_audit.pl
%
% Cross-target audit for the args_first_emission default-flip
% (PR #2285). The flip protects targets with flat-contiguous heap
% representations against the nested-compound bug that was originally
% surfaced as Elixir's bagof_with_quantifier failure.
%
% This audit:
%
% 1. Compiles a Prolog predicate with a nested-compound term to WAM
%    bytecode using the default args_first_emission(true).
%
% 2. Verifies the bytecode shape: ALL outer set_variable / set_value
%    instructions for the outer compound's args appear BEFORE the
%    nested put_structure that constructs the inner compound. This is
%    the property the flag enforces; if interleaved emission happened,
%    a flat-heap target would corrupt the outer compound's arg slots.
%
% 3. Runs the same predicate through Rust and Haskell write_*_project
%    pipelines to confirm those codegens accept the bytecode shape
%    without errors. This is a code-generation audit, not a runtime
%    audit — full runtime e2e would require cargo / ghc installation
%    and is out of scope for this fixture.
%
% Same shape used by Elixir's elx_bagof_with_quantifier test:
% nested compounds inside list cells trigger the same emission path.
% The simpler fixture here (`pair(left(a), right(b))`) is sufficient
% to exercise the outer-args-vs-nested-put_structure ordering.

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_rust_target').
:- use_module('../src/unifyweaver/targets/wam_haskell_target').

% ============================================================
% Fixture
% ============================================================

% sink/1 is a dummy callee — exists only so build_nested can call
% something with a compound arg. Without it the compound would not
% reach write-mode emission.
:- dynamic user:sink/1.
user:sink(_).

% build_nested/0 puts a nested compound on the heap and passes it
% as the arg to sink/1. Write-mode emission: put_structure for pair/2
% must place set_variable for ALL its args BEFORE the nested
% put_structure emissions for left/1 and right/1. Otherwise the
% nested heap allocations corrupt pair/2's arg slots on flat-heap
% targets (Elixir's bagof_with_quantifier bug, fixed by PR #2285).
:- dynamic user:build_nested/0.
user:build_nested :- sink(pair(left(a), right(b))).

% ============================================================
% Helpers
% ============================================================

%% args_first_property(+WamCodeString)
%
%  Verifies the args-first emission property in a WAM bytecode
%  listing. Specifically: every outer put_structure should be
%  followed by its own set_variable / set_value instructions for
%  its args BEFORE any nested put_structure interleaves.
%
%  This is verified by checking that, for any put_structure line
%  that emits an arg-bearing functor (arity > 0), the immediately
%  following lines until the next put_structure are all
%  set_variable / set_value / set_constant / get_constant — the
%  arg-emission instruction family — and not another put_structure.
%
%  We allow nested put_structures AFTER the outer's args are all
%  emitted; the check is: between an outer put_structure and the
%  point its arg-emission completes, no nested put_structure
%  appears.
args_first_property(WamCode) :-
    split_string(WamCode, "\n", " \t", Lines0),
    exclude([L]>>(string_length(L, 0)), Lines0, Lines),
    check_args_first_lines(Lines).

check_args_first_lines([]).
check_args_first_lines([Line | Rest]) :-
    (   sub_string(Line, _, _, _, "put_structure"),
        sub_string(Line, _, _, _, "/")
    ->  % Found a put_structure with a functor. Count its arity.
        parse_put_structure_arity(Line, Arity),
        check_next_args(Rest, Arity, RestAfterArgs),
        check_args_first_lines(RestAfterArgs)
    ;   check_args_first_lines(Rest)
    ).

check_next_args(Lines, 0, Lines) :- !.
check_next_args([Line | Rest], Arity, Final) :-
    Arity > 0,
    (   is_arg_emission_instr(Line)
    ->  Arity1 is Arity - 1,
        check_next_args(Rest, Arity1, Final)
    ;   sub_string(Line, _, _, _, "put_structure")
    ->  % Nested put_structure before outer args complete — fails the property.
        throw(error(args_first_property_violation(Line, Arity), _))
    ;   % Other instruction (proceed, allocate, etc.) — pass through.
        check_next_args(Rest, Arity, Final)
    ).
check_next_args([], _, []).

is_arg_emission_instr(Line) :-
    (   sub_string(Line, _, _, _, "set_variable") ;
        sub_string(Line, _, _, _, "set_value") ;
        sub_string(Line, _, _, _, "set_constant")
    ), !.

parse_put_structure_arity(Line, Arity) :-
    % Match "put_structure name/N, A1" — extract N.
    split_string(Line, "/,", " ", Parts),
    nth0(1, Parts, AritySegment),
    split_string(AritySegment, " ", "", [AStr | _]),
    number_string(Arity, AStr).

% ============================================================
% Tests
% ============================================================

:- begin_tests(wam_args_first_audit).

test(wam_bytecode_emits_args_first) :-
    % Default args_first_emission(true) is now the default.
    wam_target:compile_predicate_to_wam(user:build_nested/0, [], WamCode),
    % The bytecode should satisfy the args-first property.
    args_first_property(WamCode).

test(legacy_off_violates_property,
     [throws(error(args_first_property_violation(_, _), _))]) :-
    % Explicit opt-out reproduces the pre-PR-#2285 emission order.
    % This should TRIGGER the property check — proving the flag
    % default-on is doing real work.
    wam_target:compile_predicate_to_wam(user:build_nested/0,
                                        [args_first_emission(false)],
                                        WamCode),
    args_first_property(WamCode).

test(rust_codegen_accepts_nested_compound) :-
    % Pipe through the Rust codegen — should produce a project
    % without errors.
    unique_tmp_dir('tmp_audit_rust', TmpDir),
    setup_call_cleanup(
        true,
        once((
            catch(
                write_wam_rust_project([user:build_nested/0],
                                       [module_name('args_first_audit_rust'),
                                        wam_fallback(true)],
                                       TmpDir),
                Err,
                throw(error(rust_codegen_failed(Err), _))
            ),
            % Sanity: project dir exists and has some Rust files.
            directory_file_path(TmpDir, 'src/lib.rs', Lib),
            exists_file(Lib)
        )),
        delete_directory_and_contents(TmpDir)).

test(haskell_codegen_accepts_nested_compound) :-
    % Pipe through the Haskell codegen — should produce a project
    % without errors.
    unique_tmp_dir('tmp_audit_haskell', TmpDir),
    setup_call_cleanup(
        true,
        once((
            catch(
                write_wam_haskell_project([user:build_nested/0],
                                          [module_name('args_first_audit_haskell')],
                                          TmpDir),
                Err,
                throw(error(haskell_codegen_failed(Err), _))
            ),
            % Sanity: project dir exists. The per-target file layout
            % is the codegen contract; the codegen printing
            % "Generated project at:" is sufficient signal here.
            exists_directory(TmpDir)
        )),
        delete_directory_and_contents(TmpDir)).

:- end_tests(wam_args_first_audit).

% ============================================================
% Helpers
% ============================================================

unique_tmp_dir(Prefix, TmpDir) :-
    get_time(T),
    Stamp is floor(T * 1000000),
    (   getenv('TMPDIR', Base) -> true
    ;   Base = '/tmp'
    ),
    format(atom(TmpDir), '~w/~w_~w', [Base, Prefix, Stamp]).
