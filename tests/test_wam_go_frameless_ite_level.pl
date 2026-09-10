:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_go_frameless_ite_level.pl
%
% Probe for the WAM_FLEET_GAPS gap-A2 hazard in its *frameless-Y-write* form
% on wam_go. Ported from tests/test_wam_python_frameless_ite_level.pl (ledger
% row D52), which ported the wam_rust finding of row D50.
%
% The shape
% ---------
% `compile_if_then_else/7` in the shared emitter (`wam_target.pl`) reserves a
% permanent Y register for the if-then-else barrier AFTER it has decided
% whether the clause needs an environment. So a clause that needs no
% environment still gets `get_level Yn` ... `cut Yn` -- with NO `allocate`.
% `sat/2` clause 2 below is exactly that clause (`\+ G` inlines to
% `(G -> fail ; true)` under `ite_use_y_level(true)`, which EVERY wam_go
% compile enables), and `pick_a/4` / `pick_b/4` are callers that DO hold an
% environment with live Y registers across the call.
%
% Go keeps Y registers in the flat, global `vm.Regs[200..299]`, so
% `GetLevel Y1` used to write a choice-point depth straight over the
% caller's permanent variable Y1.
%
% Two entry lanes, and they behaved differently before the fix
% -----------------------------------------------------------
%   * INTERPRETER lane -- the caller reached through the WAM `Call`
%     instruction. Safe since ledger D51: `Call` pushes a snapshot of
%     `Regs[200:300]` (`vm.YSaves`) and `Proceed` pops it, so the callee's
%     scribble is repaired on return (the wam_javascript model).
%   * LOWERED lane -- a `lowered.go` method (`vm.PredPick_b4()`), reached the
%     way `examples/pkg_resolver/go/shim.go` and any embedder reach a
%     predicate. `wam_go_lowered_emitter.pl` emits `call p/N` as raw label
%     dispatch (`vm.PC = labels[...]; vm.Run()`) with NO `pushCallFrame`, so
%     no Y save is pushed and nothing is restored. On the pristine tree:
%
%       PredPick_b4(3, gte(1), tagX, Out)  ->  ok=true,  Out = 0   (a CP depth)
%       PredPick_a4(3, gte(1), tagX, Out)  ->  ok=false            (silent fail)
%
%     i.e. a silent wrong answer, not a crash -- SWI gives Out = tagX for both.
%
% The fix keeps ITE barrier levels on the if-then-else's own choice point
% (`ChoicePoint.Levels` + `recordIteLevel`/`lookupIteLevel` in
% `templates/targets/go_wam/state.go.mustache`), so the level never touches a
% register and is per-activation for free -- the wam_rust `ChoicePoint::levels`
% / wam_python `ChoicePoint.levels` model. Both lanes are then correct, and the
% fix does not depend on how the predicate was entered.
%
% This suite pins BOTH halves:
%   * emission -- `sat/2` really does carry `get_level` in an `allocate`-less
%     clause (if the shared emitter ever stops doing that the probe would go
%     vacuously green, so we assert the shape), and `lowered.go` really does
%     carry the lowered callers (so the exposed lane cannot silently vanish);
%   * behaviour -- the generated Go project agrees with SWI as oracle on BOTH
%     lanes, including a recursive caller that keeps several activations of
%     the same if-then-else live.
%
% Skipped automatically when the Go toolchain is unavailable.
%
%   swipl -q -g run_tests -t halt tests/test_wam_go_frameless_ite_level.pl

:- module(test_wam_go_frameless_ite_level,
          [test_wam_go_frameless_ite_level/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3,
                                 delete_directory_and_contents/1]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_go_target',
              [write_wam_go_project/3]).
:- use_module('../src/unifyweaver/targets/wam_target',
              [compile_predicate_to_wam_text/3]).

:- dynamic user:lt/2, user:sat/2, user:pick_a/4, user:pick_b/4, user:wpick/3.

% --- the probe program (also the SWI oracle) -------------------------------

user:lt(A, B) :- A < B.

% Multi-clause callee. Clause 2 needs NO environment: its only permanent
% would be the if-then-else barrier the emitter reserves for the inlined
% negation -- so it emits `get_level Y1` with no `allocate`.
user:sat(_V, any).
user:sat(V, gte(G)) :- \+ user:lt(V, G).

% Callers that DO hold an environment across the call. The Y numbering
% follows first use in the body, so pick_a parks the (unbound) output in Y1
% and pick_b parks the (bound) input tag in Y1 -- the first shape used to
% FAIL, the second used to return the clobbering choice-point index.
user:pick_a(Ver, C, Tag, Out) :- user:sat(Ver, C), Out = Tag.
user:pick_b(Ver, C, Tag, Out) :- user:sat(Ver, C), Tag = Out.

% Recursive caller: several activations of sat/2's if-then-else are live at
% once, each with its own barrier. A register-held level is not
% per-activation; a level on the guard choice point is.
user:wpick(0, Tag, Tag).
user:wpick(N, Tag, Out) :-
    N > 0,
    user:sat(N, gte(1)),
    M is N - 1,
    user:wpick(M, Tag, Out).

probe_preds([user:lt/2, user:sat/2, user:pick_a/4, user:pick_b/4,
             user:wpick/3]).

% Lowered Go methods that must exist for the lowered lane to be exercised.
% If lowering ever stops covering these the execution arm would still pass
% while testing nothing, so their absence is a loud failure.
lowered_method('PredPick_a4').
lowered_method('PredPick_b4').

% --- oracle cases ----------------------------------------------------------
%
% probe_case(Id, Label, LoweredMethod, Args, OutIndex, Goal, OutVar)
%   LoweredMethod : the lowered.go method, or `none` to run only the
%                   interpreter lane.
%   Args      : Go argument expressions for Regs[0..], excluding the output.
%   OutIndex  : 0-based Regs index the fresh output variable goes into.
%   OutVar    : the SWI variable holding the answer, or 0 for success/failure.
%
% `wpick_deep` runs the interpreter lane only. Its lowered method loses the
% output binding, but that is a SEPARATE, pre-existing lowered-lane defect,
% not this one: the control `wcall/3` -- same recursive shape, a plain
% `call okp/1` in the body, no if-then-else or negation anywhere -- fails
% identically, and both fail the same way before and after the barrier fix.
% See the residual note in docs/WAM_GO_STATUS.md.

probe_case(pick_a_true, 'pick_a/4', 'PredPick_a4',
           [int(3), gte(1), atom(tagX)], 3,
           user:pick_a(3, gte(1), tagX, O), O).
probe_case(pick_b_true, 'pick_b/4', 'PredPick_b4',
           [int(3), gte(1), atom(tagX)], 3,
           user:pick_b(3, gte(1), tagX, O), O).
probe_case(pick_a_fail, 'pick_a/4', 'PredPick_a4',
           [int(0), gte(1), atom(tagX)], 3,
           user:pick_a(0, gte(1), tagX, _), 0).
probe_case(pick_b_fail, 'pick_b/4', 'PredPick_b4',
           [int(0), gte(1), atom(tagX)], 3,
           user:pick_b(0, gte(1), tagX, _), 0).
probe_case(pick_a_any,  'pick_a/4', 'PredPick_a4',
           [int(3), atom(any), atom(tagX)], 3,
           user:pick_a(3, any, tagX, O), O).
probe_case(wpick_deep,  'wpick/3',  none,
           [int(4), atom(tagY)], 2,
           user:wpick(4, tagY, O), O).

go_available :-
    catch(( process_create(path(go), ['version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

test_wam_go_frameless_ite_level :-
    run_tests(wam_go_frameless_ite_level).

:- begin_tests(wam_go_frameless_ite_level).

% The probe is only meaningful while the shared emitter still produces the
% hazard shape. Assert it directly on the emitted text.
test(sat_clause_has_get_level_without_allocate) :-
    compile_predicate_to_wam_text(user:sat/2, [ite_use_y_level(true)], Text),
    atom_string(Text, S),
    assertion(sub_string(S, _, _, _, "get_level Y")),
    assertion(\+ sub_string(S, _, _, _, "allocate")).

% ... and while the callers really do park a permanent across the call.
test(caller_allocates_and_holds_y1) :-
    compile_predicate_to_wam_text(user:pick_b/4, [ite_use_y_level(true)], Text),
    atom_string(Text, S),
    assertion(sub_string(S, _, _, _, "allocate")),
    assertion(sub_string(S, _, _, _, "Y1")),
    assertion(sub_string(S, _, _, _, "call sat/2")).

test(matches_swi_on_both_lanes, [condition(go_available)]) :-
    once(run_frameless_probe).

:- end_tests(wam_go_frameless_ite_level).

% --- harness ---------------------------------------------------------------

run_frameless_probe :-
    probe_preds(Preds),
    Dir = 'output/test_wam_go_frameless_ite_level_gen',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    write_wam_go_project(Preds,
                         [prefer_wam(true), module_name(gfprobe),
                          package_name(wam)], Dir),
    !,
    directory_file_path(Dir, 'lib.go', LibPath),
    read_file_to_string(LibPath, Lib, []),
    (   sub_string(Lib, _, _, _, ': compilation failed')
    ->  format(user_error,
               'frameless-ITE probe: a predicate failed to compile~n', []),
        fail
    ;   true
    ),
    directory_file_path(Dir, 'lowered.go', LoweredPath),
    read_file_to_string(LoweredPath, LoweredSrc, []),
    forall(lowered_method(M),
           (   sub_string(LoweredSrc, _, _, _, M)
           ->  true
           ;   format(user_error,
                      'frameless-ITE probe: lowered.go has no ~w -- the \c
                       lowered entry lane is no longer covered~n', [M]),
               fail
           )),
    directory_file_path(Dir, 'cmd', CmdDir),
    directory_file_path(CmdDir, 'run', RunDir),
    make_directory_path(RunDir),
    directory_file_path(RunDir, 'main.go', MainPath),
    probe_go_main(MainSrc),
    setup_call_cleanup(open(MainPath, write, MS, [encoding(utf8)]),
                       write(MS, MainSrc), close(MS)),
    directory_file_path(Dir, 'go.mod', GoModPath),
    read_file_to_string(GoModPath, GoModOld, []),
    atomic_list_concat([GoModOld, "\nreplace gfprobe => ../../\n"], GoModNew),
    setup_call_cleanup(open(GoModPath, write, GS), write(GS, GoModNew),
                       close(GS)),
    format(atom(BuildCmd), 'cd ~w && go build -o runprobe . 2>&1', [RunDir]),
    process_create(path(sh), ['-c', BuildCmd],
                   [stdout(pipe(BO)), stderr(std), process(BP)]),
    read_string(BO, _, BuildOut), close(BO),
    process_wait(BP, BuildStatus),
    (   BuildStatus == exit(0)
    ->  true
    ;   format(user_error, '~n[go build failed]~n~w~n', [BuildOut]),
        throw(go_build_failed(BuildStatus))
    ),
    format(atom(RunCmd), 'cd ~w && ./runprobe 2>&1', [RunDir]),
    process_create(path(sh), ['-c', RunCmd],
                   [stdout(pipe(RO)), stderr(std), process(RP)]),
    read_string(RO, _, RunOut), close(RO),
    process_wait(RP, RunStatus),
    (   RunStatus == exit(0), sub_string(RunOut, _, _, _, "ALL PASS")
    ->  true
    ;   format(user_error,
               "~n[frameless-ITE-level harness output]~n~w~n", [RunOut]),
        throw(wam_go_frameless_ite_level_failed(RunStatus))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

% One chk(...) pair per case (interpreter lane + lowered lane), with SWI
% supplying the expectation.
probe_go_main(Src) :-
    findall(Line, case_check_line(Line), Lines),
    atomic_list_concat(Lines, '\n', ChecksBlock),
    go_prelude(Prelude),
    go_epilogue(Epilogue),
    atomic_list_concat([Prelude, ChecksBlock, Epilogue], '\n', Src).

case_check_line(Line) :-
    probe_case(Id, Label, Method, Args, OutIdx, Goal, OutVar),
    maplist(go_arg, Args, ArgStrs),
    atomic_list_concat(ArgStrs, ', ', ArgList),
    (   catch(Goal, _, fail)
    ->  (   OutVar == 0
        ->  Expect = 'true'
        ;   go_answer_literal(OutVar, Expect)
        )
    ;   Expect = 'FAIL'
    ),
    (   Method == none
    ->  format(atom(Line),
               "\tchk(\"~w/interp\", runInterp(\"~w\", []wam.Value{~w}, ~w), \"~w\")",
               [Id, Label, ArgList, OutIdx, Expect])
    ;   format(atom(Line),
               "\tchk(\"~w/interp\", runInterp(\"~w\", []wam.Value{~w}, ~w), \"~w\")\n\c
                \tchk(\"~w/lowered\", runLowered((*wam.WamState).~w, []wam.Value{~w}, ~w), \"~w\")",
               [Id, Label, ArgList, OutIdx, Expect,
                Id, Method, ArgList, OutIdx, Expect])
    ).

% The harness reduces an answer to a bare atom name or decimal integer, so
% the expectation SWI computes is spelled the same way. A goal with no output
% variable reports plain success as "true".
go_answer_literal(V, S) :- number(V), !, format(atom(S), '~w', [V]).
go_answer_literal(V, S) :- format(atom(S), '~w', [V]).

go_arg(int(N), S)  :- format(atom(S), '&wam.Integer{Val: ~w}', [N]).
go_arg(atom(A), S) :- format(atom(S), 'wam.InternAtom("~w")', [A]).
go_arg(gte(N), S)  :-
    format(atom(S),
           '&wam.Structure{Functor: "gte/1", Arity: 1, Args: []wam.Value{&wam.Integer{Val: ~w}}}',
           [N]).

go_prelude(
'package main

import (
	"fmt"
	"os"

	wam "gfprobe"
)

// answer reduces a solution to a comparable string: the atom name, the
// decimal integer, "UNBOUND" for a success that left the output variable
// unbound, or "FAIL".
func answer(ok bool, v wam.Value) string {
	if !ok {
		return "FAIL"
	}
	switch t := v.(type) {
	case *wam.Atom:
		return t.Name
	case *wam.Integer:
		return fmt.Sprintf("%d", t.Val)
	case *wam.Unbound:
		return "UNBOUND"
	}
	return fmt.Sprintf("%v", v)
}

func seed(args []wam.Value, outIdx int) (*wam.WamState, *wam.Unbound) {
	vm := wam.NewWamState(wam.SharedWamCode, wam.SharedWamLabels)
	out := &wam.Unbound{Name: "Out", Idx: 900001}
	for i, a := range args {
		vm.Regs[i] = a
	}
	vm.Regs[outIdx] = out
	return vm, out
}

// Lane 1: the interpreter, entered at the predicate label the way a shim
// does. Inside it, `call sat/2` pushes a Y save that Proceed pops.
func runInterp(label string, args []wam.Value, outIdx int) string {
	vm, out := seed(args, outIdx)
	pc, ok := vm.Ctx.Labels[label]
	if !ok {
		return "NOLABEL:" + label
	}
	vm.PC = pc
	r := vm.Run()
	return answer(r, vm.Deref(out))
}

// Lane 2: the lowered Go method. Its `call` is raw label dispatch with no
// pushCallFrame, so nothing saves or restores the caller Y registers it
// wrote directly into vm.Regs[200...].
func runLowered(fn func(*wam.WamState) bool, args []wam.Value, outIdx int) string {
	vm, out := seed(args, outIdx)
	r := fn(vm)
	return answer(r, vm.Deref(out))
}

var fails int

func chk(name, got, want string) {
	if got != want {
		fails++
		fmt.Printf("FAIL %s got %q want %q\\n", name, got, want)
	}
}

func main() {').

go_epilogue(
'	if fails == 0 {
		fmt.Println("ALL PASS")
	} else {
		fmt.Printf("FAILURES: %d\\n", fails)
		os.Exit(1)
	}
}
').
