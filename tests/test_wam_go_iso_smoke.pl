:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_go_iso_smoke.pl — ISO three-form error handling for WAM-Go
%
% Closes the ISO-GO card in docs/WAM_FLEET_GAP_TASKS.md. Structured
% after tests/test_wam_haskell_iso_smoke.pl and checked against the
% seven adoption criteria in
% docs/design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md §"What Counts As
% Adoption":
%
%   1. catch/3 + throw/1 runtime support        — go_catch_recovers_*,
%                                                 go_throw_plain_ball
%   2. error(ErrorType, Context) constructors   — go_runtime_error_ctors
%   3. shared config shape + inline overrides   — go_inline_override_*
%   4. per-predicate rewrite of default keys    — go_rewrite_*
%   5. explicit _iso/_lax keys survive flips    — go_explicit_*_survives
%   6. ISO throws / lax fails / overrides       — the execution block
%   7. audit report                             — go_audit_*
%
% The execution block builds and runs the generated Go and is skipped
% (with a message) when `go` isn't on PATH; everything else always runs.
%
% Usage: swipl -q -g run_tests -t halt tests/test_wam_go_iso_smoke.pl

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_go_target').
:- use_module('../src/unifyweaver/core/iso_errors').

:- begin_tests(wam_go_iso_smoke).

:- dynamic user:goiso_div/2.
:- dynamic user:goiso_unbound/1.
:- dynamic user:goiso_cmp/2.
:- dynamic user:goiso_succ/1.
:- dynamic user:goiso_catch_div/1.
:- dynamic user:goiso_catch_other/1.
:- dynamic user:goiso_throw_plain/1.

goiso_predicates([user:goiso_div/2, user:goiso_unbound/1, user:goiso_cmp/2,
                  user:goiso_succ/1, user:goiso_catch_div/1,
                  user:goiso_catch_other/1, user:goiso_throw_plain/1]).

goiso_assert :-
    assertz((user:goiso_div(X, Y) :- Y is X // 0)),
    assertz((user:goiso_unbound(X) :- _Z is X + 1)),
    assertz((user:goiso_cmp(A, B) :- A < B)),
    assertz((user:goiso_succ(X) :- succ(foo, X))),
    assertz((user:goiso_catch_div(R) :- catch((_Y is 1 // 0), error(E, _), true), R = E)),
    assertz((user:goiso_catch_other(R) :- catch((_Y is 1 // 0), type_error(_, _), true), R = nope)),
    assertz((user:goiso_throw_plain(R) :- catch(throw(my_ball), B, true), R = B)).

goiso_retract :-
    retractall(user:goiso_div(_, _)),
    retractall(user:goiso_unbound(_)),
    retractall(user:goiso_cmp(_, _)),
    retractall(user:goiso_succ(_)),
    retractall(user:goiso_catch_div(_)),
    retractall(user:goiso_catch_other(_)),
    retractall(user:goiso_throw_plain(_)).

% =====================================================================
% (4,5) Key tables and WAM-text rewrite
% =====================================================================

test(go_iso_key_tables_registered) :-
    forall(member(Key-IsoKey-LaxKey,
                  [ "is/2"-"is_iso/2"-"is_lax/2",
                    "</2"-"<_iso/2"-"<_lax/2",
                    ">/2"-">_iso/2"-">_lax/2",
                    ">=/2"-">=_iso/2"-">=_lax/2",
                    "=</2"-"=<_iso/2"-"=<_lax/2",
                    "=:=/2"-"=:=_iso/2"-"=:=_lax/2",
                    "=\\=/2"-"=\\=_iso/2"-"=\\=_lax/2",
                    "succ/2"-"succ_iso/2"-"succ_lax/2" ]),
           ( assertion(iso_errors:iso_errors_default_to_iso(Key, IsoKey)),
             assertion(iso_errors:iso_errors_default_to_lax(Key, LaxKey)) )).

% Every ISO/lax key must also be a Go direct builtin. Without an entry
% the emitter falls through to Execute{Pred: ...}, which looks the key
% up as an indexed fact table and silently fails.
test(go_iso_keys_are_direct_builtins) :-
    forall(member(Key, ["is_iso/2", "is_lax/2", "succ_iso/2", "succ_lax/2",
                        "<_iso/2", "<_lax/2", ">_iso/2", ">_lax/2",
                        "=<_iso/2", "=<_lax/2", ">=_iso/2", ">=_lax/2",
                        "=:=_iso/2", "=:=_lax/2", "=\\=_iso/2", "=\\=_lax/2",
                        "catch/3", "throw/1"]),
           assertion(wam_go_target:wam_go_direct_builtin(Key, _, _))).

test(go_rewrite_is_to_iso) :-
    wam_go_target:iso_errors_rewrite_text(iso_config(true, []), test/0,
        "    builtin_call is/2, 2\n    proceed\n", Rewritten),
    assertion(sub_string(Rewritten, _, _, _, "is_iso/2")).

test(go_rewrite_is_to_lax) :-
    wam_go_target:iso_errors_rewrite_text(iso_config(false, []), test/0,
        "    builtin_call is/2, 2\n    proceed\n", Rewritten),
    assertion(sub_string(Rewritten, _, _, _, "is_lax/2")).

test(go_rewrite_comparison_and_succ) :-
    wam_go_target:iso_errors_rewrite_text(iso_config(true, []), test/0,
        "    builtin_call </2, 2\n    builtin_call succ/2, 2\n", Rewritten),
    assertion(sub_string(Rewritten, _, _, _, "<_iso/2")),
    assertion(sub_string(Rewritten, _, _, _, "succ_iso/2")).

% execute is a rewritable shape too: a clause whose last goal is one of
% these builtins compiles to `execute <key>`, not `builtin_call`.
test(go_rewrite_execute_shape) :-
    wam_go_target:iso_errors_rewrite_text(iso_config(true, []), test/0,
        "    execute succ/2\n", Rewritten),
    assertion(sub_string(Rewritten, _, _, _, "succ_iso/2")).

test(go_explicit_iso_survives_lax_mode) :-
    wam_go_target:iso_errors_rewrite_text(iso_config(false, []), test/0,
        "    builtin_call is_iso/2, 2\n", Rewritten),
    assertion(sub_string(Rewritten, _, _, _, "is_iso/2")),
    assertion(\+ sub_string(Rewritten, _, _, _, "is_lax/2")).

test(go_explicit_lax_survives_iso_mode) :-
    wam_go_target:iso_errors_rewrite_text(iso_config(true, []), test/0,
        "    builtin_call is_lax/2, 2\n", Rewritten),
    assertion(sub_string(Rewritten, _, _, _, "is_lax/2")),
    assertion(\+ sub_string(Rewritten, _, _, _, "is_iso/2")).

test(go_rewrite_leaves_untabled_keys_alone) :-
    wam_go_target:iso_errors_rewrite_text(iso_config(true, []), test/0,
        "    builtin_call atom_length/2, 2\n", Rewritten),
    assertion(sub_string(Rewritten, _, _, _, "atom_length/2")).

% =====================================================================
% (3) Shared config shape, including per-predicate inline overrides
% =====================================================================

test(go_inline_override_flips_one_predicate) :-
    iso_errors_resolve_options([iso_errors(false), iso_errors(p/1, true)], Config),
    wam_go_target:iso_errors_rewrite_text(Config, p/1,
        "    builtin_call is/2, 2\n", Overridden),
    wam_go_target:iso_errors_rewrite_text(Config, q/1,
        "    builtin_call is/2, 2\n", Default),
    assertion(sub_string(Overridden, _, _, _, "is_iso/2")),
    assertion(sub_string(Default, _, _, _, "is_lax/2")).

test(go_inline_default_true_applies_everywhere) :-
    iso_errors_resolve_options([iso_errors(true)], Config),
    wam_go_target:iso_errors_rewrite_text(Config, anything/3,
        "    builtin_call is/2, 2\n", Rewritten),
    assertion(sub_string(Rewritten, _, _, _, "is_iso/2")).

% =====================================================================
% (7) Audit
% =====================================================================

test(go_audit_reports_resolved_keys) :-
    setup_call_cleanup(
        goiso_assert,
        ( wam_go_iso_audit([user:goiso_div/2], [iso_errors(true)], Audit),
          assertion(Audit = [audit(_, true, Sites)]),
          Audit = [audit(_, _, Sites)],
          assertion(Sites \== []),
          assertion(( member(site(_, "is/2", "is_iso/2", _, _), Sites) )),
          with_output_to(string(Report), wam_go_iso_audit_report(Audit)),
          assertion(sub_string(Report, _, _, _, "is_iso/2")) ),
        goiso_retract).

% =====================================================================
% Codegen: the rewrite reaches the emitted project, and only when asked
% =====================================================================

test(go_project_iso_mode_emits_iso_keys) :-
    setup_call_cleanup(
        goiso_assert,
        goiso_with_project([iso_errors(true)], [go_project_has_iso_keys]),
        goiso_retract).

test(go_project_lax_mode_emits_lax_keys) :-
    setup_call_cleanup(
        goiso_assert,
        goiso_with_project([iso_errors(false)], [go_project_has_lax_keys]),
        goiso_retract).

% No iso_errors option at all: the WAM text is untouched and the
% project keeps the plain builtin keys it has always used.
test(go_project_without_config_keeps_plain_keys) :-
    setup_call_cleanup(
        goiso_assert,
        goiso_with_project([], [go_project_has_plain_keys]),
        goiso_retract).

go_project_has_iso_keys(_TmpDir, LibCode) :-
    assertion(sub_string(LibCode, _, _, _, 'Op: "is_iso/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "<_iso/2"')),
    assertion(sub_string(LibCode, _, _, _, 'succ_iso/2')),
    assertion(\+ sub_string(LibCode, _, _, _, 'Op: "is/2"')).

go_project_has_lax_keys(_TmpDir, LibCode) :-
    assertion(sub_string(LibCode, _, _, _, 'Op: "is_lax/2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "<_lax/2"')),
    assertion(\+ sub_string(LibCode, _, _, _, 'Op: "is/2"')).

go_project_has_plain_keys(_TmpDir, LibCode) :-
    assertion(sub_string(LibCode, _, _, _, 'Op: "is/2"')),
    assertion(\+ sub_string(LibCode, _, _, _, 'is_iso/2')),
    assertion(\+ sub_string(LibCode, _, _, _, 'is_lax/2')).

% =====================================================================
% (1,2,6) Execution: ISO throws, lax fails, catch/3 recovers
% =====================================================================

test(go_runtime_error_ctors_present) :-
    read_file_to_string('templates/targets/go_wam/state.go.mustache', State, []),
    assertion(sub_string(State, _, _, _, 'func makeErrorTerm(formal Value, context Value) Value')),
    assertion(sub_string(State, _, _, _, 'func makeInstantiationError(context string) Value')),
    assertion(sub_string(State, _, _, _, 'func makeTypeError(expected string, culprit Value, context string) Value')),
    assertion(sub_string(State, _, _, _, 'func makeEvaluationError(what string, context string) Value')),
    assertion(sub_string(State, _, _, _, 'type prologBall struct')),
    assertion(sub_string(State, _, _, _, 'func (vm *WamState) builtinCatch(goal Value, catcher Value, recovery Value) (result bool)')),
    % Run recovers an escaped ball rather than letting the panic kill
    % the process; that lives in the transpiled runtime.
    wam_go_target:compile_wam_runtime_to_go([], RuntimeCode),
    atom_string(RuntimeCode, Runtime),
    assertion(sub_string(Runtime, _, _, _, 'vm.UncaughtBall = thrown.Ball')).

test(go_iso_execution) :-
    setup_call_cleanup(
        goiso_assert,
        goiso_with_project([iso_errors(true)], [go_run_iso_driver]),
        goiso_retract).

test(go_lax_execution) :-
    setup_call_cleanup(
        goiso_assert,
        goiso_with_project([iso_errors(false)], [go_run_lax_driver]),
        goiso_retract).

go_run_iso_driver(TmpDir, _LibCode) :-
    goiso_run_driver(TmpDir, Output),
    (   Output == skipped
    ->  true
    ;   % ISO mode: each faulty call throws the ISO error it warrants.
        assertion(sub_string(Output, _, _, _,
            "div ok=false R=_R uncaught=error(evaluation_error(zero_divisor), is/2)")),
        assertion(sub_string(Output, _, _, _,
            "unbound ok=false R=_R uncaught=error(instantiation_error, is/2)")),
        assertion(sub_string(Output, _, _, _,
            "succ ok=false R=_R uncaught=error(type_error(integer, foo), succ/2)")),
        assertion(sub_string(Output, _, _, _, "cmp ok=false R=_R uncaught=error(type_error(evaluable,")),
        % catch/3 recovers a matching ball and binds through it.
        assertion(sub_string(Output, _, _, _,
            "catch_div ok=true R=evaluation_error(zero_divisor)")),
        % A non-matching catcher re-throws for an outer handler.
        assertion(sub_string(Output, _, _, _,
            "catch_other ok=false R=_R uncaught=error(evaluation_error(zero_divisor), is/2)")),
        % throw/1 with a plain (non-error) ball.
        assertion(sub_string(Output, _, _, _, "throw_plain ok=true R=my_ball"))
    ).

go_run_lax_driver(TmpDir, _LibCode) :-
    goiso_run_driver(TmpDir, Output),
    (   Output == skipped
    ->  true
    ;   % Lax mode: the same calls fail silently, nothing is thrown.
        assertion(sub_string(Output, _, _, _, "div ok=false R=_R uncaught=-")),
        assertion(sub_string(Output, _, _, _, "unbound ok=false R=_R uncaught=-")),
        assertion(sub_string(Output, _, _, _, "succ ok=false R=_R uncaught=-")),
        assertion(sub_string(Output, _, _, _, "cmp ok=false R=_R uncaught=-")),
        assertion(sub_string(Output, _, _, _, "catch_div ok=false R=_R uncaught=-")),
        % An explicit throw/1 is not an arithmetic error and still works.
        assertion(sub_string(Output, _, _, _, "throw_plain ok=true R=my_ball"))
    ).

% =====================================================================
% Harness
% =====================================================================

%% goiso_with_project(+ExtraOptions, +Checks)
%  Generate the ISO fixture project, run each check against it, clean up.
goiso_with_project(ExtraOptions, Checks) :-
    get_time(T),
    format(atom(TmpDir), 'tmp_wam_go_iso_~w', [T]),
    append([module_name(goiso_test), prefer_wam(true)], ExtraOptions, Options),
    goiso_predicates(Predicates),
    setup_call_cleanup(
        write_wam_go_project(Predicates, Options, TmpDir),
        ( directory_file_path(TmpDir, 'lib.go', LibPath),
          read_file_to_string(LibPath, LibCode, []),
          forall(member(Check, Checks), call(Check, TmpDir, LibCode)) ),
        ( exists_directory(TmpDir) -> delete_directory_and_contents(TmpDir) ; true )
    ).

%% goiso_run_driver(+TmpDir, -Output)
%  Build and run the generated project through a driver that reports
%  each predicate's success plus any uncaught ball. Output is the atom
%  `skipped` when the Go toolchain is unavailable.
goiso_run_driver(TmpDir, Output) :-
    (   catch(process_create(path(go), ['version'],
                             [stdout(null), stderr(null)]), _, fail)
    ->  directory_file_path(TmpDir, 'cmd', CmdDir),
        directory_file_path(CmdDir, 'goiso', DriverDir),
        make_directory_path(DriverDir),
        directory_file_path(DriverDir, 'main.go', MainPath),
        goiso_driver_source(Source),
        goiso_write_file(MainPath, Source),
        format(string(RunCmd), "cd ~w && go run ./cmd/goiso 2>&1", [TmpDir]),
        process_create(path(sh), ['-c', RunCmd],
                       [stdout(pipe(Out)), process(Pid)]),
        read_string(Out, _, Output),
        process_wait(Pid, Exit),
        format('~nWAM-Go ISO driver output:~n~s~n', [Output]),
        assertion(Exit == exit(0))
    ;   format('~n[skip] go not on PATH — skipping WAM-Go ISO execution check~n'),
        Output = skipped
    ).

goiso_driver_source(
'package main

import (
	"fmt"
	wam "goiso_test"
)

// probe runs one predicate with an unbound output register and reports
// both its result and any ball that escaped every catch/3.
func probe(name string, code []wam.Instruction, labels map[string]int, pc int, in ...wam.Value) {
	vm := wam.NewWamState(code, labels)
	vm.PC = pc
	for i, v := range in {
		vm.Regs[i] = v
	}
	out := &wam.Unbound{Name: "R", Idx: 0}
	vm.Regs[len(in)] = out
	ok := vm.Run()
	fmt.Printf("%s ok=%v R=%s uncaught=%s\\n",
		name, ok, vm.WriteTerm(vm.Deref(out)), vm.WriteTerm(vm.UncaughtBall))
}

func main() {
	probe("div", wam.Goiso_divCode, wam.Goiso_divLabels, wam.Goiso_divStartPC, &wam.Integer{Val: 5})
	probe("unbound", wam.Goiso_unboundCode, wam.Goiso_unboundLabels, wam.Goiso_unboundStartPC)
	probe("succ", wam.Goiso_succCode, wam.Goiso_succLabels, wam.Goiso_succStartPC)
	probe("catch_div", wam.Goiso_catch_divCode, wam.Goiso_catch_divLabels, wam.Goiso_catch_divStartPC)
	probe("catch_other", wam.Goiso_catch_otherCode, wam.Goiso_catch_otherLabels, wam.Goiso_catch_otherStartPC)
	probe("throw_plain", wam.Goiso_throw_plainCode, wam.Goiso_throw_plainLabels, wam.Goiso_throw_plainStartPC)

	cmp := wam.NewWamState(wam.Goiso_cmpCode, wam.Goiso_cmpLabels)
	cmp.PC = wam.Goiso_cmpStartPC
	cmp.Regs[0] = &wam.Integer{Val: 1}
	cmp.Regs[1] = &wam.Atom{Name: "notanum"}
	okc := cmp.Run()
	fmt.Printf("cmp ok=%v R=_R uncaught=%s\\n", okc, cmp.WriteTerm(cmp.UncaughtBall))
}
').

goiso_write_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).

:- end_tests(wam_go_iso_smoke).
