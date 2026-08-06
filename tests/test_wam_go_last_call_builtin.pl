:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_go_last_call_builtin.pl — three WAM-Go codegen/runtime parity fixes
%
% 1. `execute <builtin>/N` (a clause whose *last* goal is a builtin, e.g.
%    `s(X) :- succ(1, X).`) used to emit a bare BuiltinCall. BuiltinCall
%    advances to the following instruction, but `execute` has none: the
%    predicate bound its output arguments correctly and still reported
%    failure. Now emits BuiltinExecute, which runs the builtin and then
%    takes Proceed's return path.
%
% 2. Go has no overloading, so a project carrying the same predicate name
%    at two arities (e.g. the portable parser's `read_term_from_atom/2`
%    and `/3`) emitted two identically named `func`s and failed to compile
%    with "redeclared in this block". Overloaded names now get an arity
%    suffix; names that are unique in the project are unchanged.
%
% 3. The write-family builtins rendered a compound via Value.String(),
%    which prints a *Structure as bare "functor/arity" and drops every
%    argument — `write(foo(a,b))` printed "foo/2". They now go through
%    writeTermString, the unquoted sibling of the write_canonical/1
%    renderer.
%
% The end-to-end section is skipped when `go` isn't on PATH; the codegen
% assertions always run.

:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_go_target').
:- use_module(library(filesex)).
:- use_module(library(process)).

:- begin_tests(wam_go_last_call_builtin).

:- dynamic user:golc_succ_last/1.
:- dynamic user:golc_succ_mid/1.
:- dynamic user:golc_is_last/1.
:- dynamic user:golc_write_compound/0.
:- dynamic user:golc_dup/1.
:- dynamic user:golc_dup/2.

golc_predicates([golc_succ_last/1, golc_succ_mid/1, golc_is_last/1,
                 golc_write_compound/0, golc_dup/1, golc_dup/2]).

golc_assert_predicates :-
    assertz((user:golc_succ_last(X) :- succ(1, X))),
    assertz((user:golc_succ_mid(X) :- succ(1, Y), X is Y * 1)),
    assertz((user:golc_is_last(X) :- X is 1 + 1)),
    assertz((user:golc_write_compound :- write(pair(a, b)), nl)),
    assertz(user:golc_dup(one)),
    assertz(user:golc_dup(one, two)).

golc_retract_predicates :-
    retractall(user:golc_succ_last(_)),
    retractall(user:golc_succ_mid(_)),
    retractall(user:golc_is_last(_)),
    retractall(user:golc_write_compound),
    retractall(user:golc_dup(_)),
    retractall(user:golc_dup(_, _)).

test(last_call_builtin_and_overload_codegen) :-
    get_time(T),
    format(atom(TmpDir), 'tmp_wam_go_last_call_~w', [T]),
    setup_call_cleanup(
        golc_assert_predicates,
        run_golc_codegen_test(TmpDir),
        ( golc_retract_predicates,
          ( exists_directory(TmpDir) -> delete_directory_and_contents(TmpDir) ; true ) )
    ).

run_golc_codegen_test(TmpDir) :-
    golc_predicates(Predicates),
    write_wam_go_project(Predicates, [module_name(golc_test), prefer_wam(true)], TmpDir),

    directory_file_path(TmpDir, 'lib.go', LibPath),
    read_file_to_string(LibPath, LibCode, []),

    % (1) A last-goal builtin uses BuiltinExecute; the same builtin in
    % mid-body position (golc_succ_mid, where succ/2 is followed by is/2)
    % still uses BuiltinCall, which advances to the next instruction.
    assertion(sub_string(LibCode, _, _, _, '&BuiltinExecute{Op: "succ/2"')),
    assertion(sub_string(LibCode, _, _, _, '&BuiltinCall{Op: "succ/2"')),
    assertion(sub_string(LibCode, _, _, _, '&BuiltinCall{Op: "is/2"')),
    assertion(sub_string(LibCode, _, _, _, '&BuiltinCall{Op: "write/1"')),

    % (2) the overloaded name is arity-suffixed; unique names are not.
    assertion(sub_string(LibCode, _, _, _, 'func Golc_dup1(a1 Value) bool')),
    assertion(sub_string(LibCode, _, _, _, 'func Golc_dup2(a1 Value, a2 Value) bool')),
    assertion(\+ sub_string(LibCode, _, _, _, 'func Golc_dup(')),
    assertion(sub_string(LibCode, _, _, _, 'func Golc_succ_last(a1 Value) bool')),

    % (3) the write-family builtins render terms instead of stringifying
    % the Value directly.
    directory_file_path(TmpDir, 'state.go', StatePath),
    read_file_to_string(StatePath, StateCode, []),
    assertion(sub_string(StateCode, _, _, _,
        'func (vm *WamState) writeTermString(value Value) string')),
    assertion(sub_string(StateCode, _, _, _,
        'case "write/1":\n\t\tfmt.Print(vm.writeTermString(arg1))')),
    assertion(sub_string(StateCode, _, _, _,
        'case "writeln/1":\n\t\tfmt.Println(vm.writeTermString(arg1))')),

    % The generated BuiltinExecute case must return through the caller
    % rather than falling through to the next instruction.
    directory_file_path(TmpDir, 'runtime.go', RuntimePath),
    read_file_to_string(RuntimePath, RuntimeCode, []),
    assertion(sub_string(RuntimeCode, _, _, _, 'case *BuiltinExecute:')),

    run_golc_go_binary(TmpDir).

%% run_golc_go_binary(+TmpDir)
%  Build and run the generated project through a driver. Skipped (with a
%  message) when the Go toolchain isn't available.
run_golc_go_binary(TmpDir) :-
    (   catch(process_create(path(go), ['version'],
                             [stdout(null), stderr(null)]), _, fail)
    ->  golc_run_driver(TmpDir)
    ;   format('~n[skip] go not on PATH — skipping WAM-Go execution check~n')
    ).

golc_run_driver(TmpDir) :-
    directory_file_path(TmpDir, 'cmd', CmdDir),
    directory_file_path(CmdDir, 'golc', DriverDir),
    make_directory_path(DriverDir),
    directory_file_path(DriverDir, 'main.go', MainPath),
    golc_driver_source(Source),
    golc_write_file(MainPath, Source),

    format(string(RunCmd), "cd ~w && go run ./cmd/golc 2>&1", [TmpDir]),
    process_create(path(sh), ['-c', RunCmd], [stdout(pipe(Out)), process(Pid)]),
    read_string(Out, _, Output),
    process_wait(Pid, Exit),
    format('~nWAM-Go last-call driver output:~n~s~n', [Output]),
    assertion(Exit == exit(0)),

    % succ/2 as the last goal now succeeds *and* binds.
    assertion(sub_string(Output, _, _, _, "succ_last ok=true X=2")),
    assertion(sub_string(Output, _, _, _, "succ_mid ok=true X=2")),
    assertion(sub_string(Output, _, _, _, "is_last ok=true X=2")),
    % Both arities of the overloaded predicate are callable.
    assertion(sub_string(Output, _, _, _, "dup1 ok=true")),
    assertion(sub_string(Output, _, _, _, "dup2 ok=true")),
    % write/1 keeps the compound's arguments.
    assertion(sub_string(Output, _, _, _, "pair(a, b)")).

golc_driver_source(
'package main

import (
	"fmt"
	wam "golc_test"
)

func probe(name string, code []wam.Instruction, labels map[string]int, pc int) {
	vm := wam.NewWamState(code, labels)
	vm.PC = pc
	x := &wam.Unbound{Name: "X", Idx: 0}
	vm.Regs[0] = x
	ok := vm.Run()
	fmt.Printf("%s ok=%v X=%v\\n", name, ok, vm.Deref(x))
}

func main() {
	probe("succ_last", wam.Golc_succ_lastCode, wam.Golc_succ_lastLabels, wam.Golc_succ_lastStartPC)
	probe("succ_mid", wam.Golc_succ_midCode, wam.Golc_succ_midLabels, wam.Golc_succ_midStartPC)
	probe("is_last", wam.Golc_is_lastCode, wam.Golc_is_lastLabels, wam.Golc_is_lastStartPC)

	d1 := wam.NewWamState(wam.Golc_dup1Code, wam.Golc_dup1Labels)
	d1.PC = wam.Golc_dup1StartPC
	d1.Regs[0] = &wam.Unbound{Name: "A", Idx: 0}
	fmt.Printf("dup1 ok=%v\\n", d1.Run())

	d2 := wam.NewWamState(wam.Golc_dup2Code, wam.Golc_dup2Labels)
	d2.PC = wam.Golc_dup2StartPC
	d2.Regs[0] = &wam.Unbound{Name: "A", Idx: 0}
	d2.Regs[1] = &wam.Unbound{Name: "B", Idx: 1}
	fmt.Printf("dup2 ok=%v\\n", d2.Run())

	w := wam.NewWamState(wam.Golc_write_compoundCode, wam.Golc_write_compoundLabels)
	w.PC = wam.Golc_write_compoundStartPC
	fmt.Printf("write ok=%v out=", w.Run())
}
').

golc_write_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).

:- end_tests(wam_go_last_call_builtin).
