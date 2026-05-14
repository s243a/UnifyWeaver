:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_go_target').
:- use_module(library(filesex)).

:- begin_tests(go_wam_builtins).

:- dynamic user:test_builtins/1.
:- dynamic user:test_term_builtins/0.
:- dynamic user:test_member_collect/0.

test(builtins_execution) :-
    get_time(T),
    format(atom(TmpDir), 'tmp_wam_builtins_~w', [T]),
    setup_call_cleanup(
        ( assertz(user:test_builtins(X) :-
            (   X is 1 + 2,
                X < 5,
                X =:= 3,
                X =< 3,
                is_list([a,b]),
                display(ok),
                nl,
                atom(foo),
                \+ atom(5)
            )),
          assertz(user:test_term_builtins :-
            (   functor(f(a, 7), F, A),
                F == f,
                A =:= 2,
                arg(2, f(a, 7), V),
                V =:= 7,
                f(a, 7) =.. L,
                length(L, 3),
                member(f, L),
                member(a, L),
                member(7, L),
                copy_term(f(X, X), C),
                arg(1, C, Y),
                arg(2, C, Z),
                Y == Z
            )),
          assertz(user:test_member_collect :-
            (   findall(X, member(X, [a,b]), L),
                length(L, 2),
                member(a, L),
                member(b, L)
            ))
        ),
        run_builtins_test(TmpDir),
        ( retractall(user:test_builtins(_)),
          retractall(user:test_term_builtins),
          retractall(user:test_member_collect),
          delete_directory_and_contents(TmpDir) )
    ).

run_builtins_test(TmpDir) :-
    Predicates = [test_builtins/1, test_term_builtins/0, test_member_collect/0],
    Options = [module_name(builtin_test)],

    write_wam_go_project(Predicates, Options, TmpDir),

    % Verify generated code has our fixes
    directory_file_path(TmpDir, 'value.go', ValuePath),
    read_file_to_string(ValuePath, ValueCode, []),
    assertion(sub_string(ValueCode, _, _, _, "type Structure struct")),
    directory_file_path(TmpDir, 'lib.go', LibPath),
    read_file_to_string(LibPath, LibCode, []),
    assertion(sub_string(LibCode, _, _, _, 'Op: "=</2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "is_list/1"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "display/1"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "functor/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "arg/3"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "=../2"')),
    assertion(sub_string(LibCode, _, _, _, 'Op: "copy_term/2"')),

    % Add a main.go to run the test in a separate cmd directory
    directory_file_path(TmpDir, 'cmd', CmdDir),
    directory_file_path(CmdDir, 'test', TestDir),
    make_directory_path(TestDir),
    directory_file_path(TestDir, 'main.go', MainPath),
    write_file(MainPath,
'package main

import (
	"fmt"
	wam "builtin_test"
)

func main() {
	vm := wam.NewWamState(wam.Test_builtinsCode, wam.Test_builtinsLabels)
	vm.PC = wam.Test_builtinsStartPC
	// A1 will hold X — use named unbound so we can deref after execution
	x := &wam.Unbound{Name: "X", Idx: 0}
	vm.Regs[0] = x

	if vm.Run() {
		fmt.Printf("SUCCESS: X=%v\\n", vm.Deref(x))
	} else {
		fmt.Println("FAILURE")
	}

	termVM := wam.NewWamState(wam.Test_term_builtinsCode, wam.Test_term_builtinsLabels)
	termVM.PC = wam.Test_term_builtinsStartPC
	if termVM.Run() {
		fmt.Println("TERM_SUCCESS")
	} else {
		fmt.Println("TERM_FAILURE")
	}

	memberVM := wam.NewWamState(wam.Test_member_collectCode, wam.Test_member_collectLabels)
	memberVM.PC = wam.Test_member_collectStartPC
	if memberVM.Run() {
		fmt.Println("MEMBER_SUCCESS")
	} else {
		fmt.Println("MEMBER_FAILURE")
	}
}
'),

    % Update go.mod to include local replace for the module
    directory_file_path(TmpDir, 'go.mod', GoModPath),
    read_file_to_string(GoModPath, GoModOld, []),
    atomic_list_concat([GoModOld, "\nreplace builtin_test => ../../\n"], GoModNew),
    write_file(GoModPath, GoModNew),

    % Verify it compiles and runs
    (   catch(process_create(path(go), ['version'], [stdout(null), stderr(null)]), _, fail)
    ->  format(string(RunCmd), "cd ~w && go run main.go 2>&1", [TestDir]),
        process_create(path(sh), ['-c', RunCmd], [stdout(pipe(Out)), process(Pid)]),
        read_string(Out, _, FullOutput),
        process_wait(Pid, Exit),
        format('Full output from Go: ~s~n', [FullOutput]),
        assertion(Exit == exit(0)),
        assertion(sub_string(FullOutput, _, _, _, "ok")),
        assertion(sub_string(FullOutput, _, _, _, "SUCCESS: X=3")),
        assertion(sub_string(FullOutput, _, _, _, "TERM_SUCCESS")),
        assertion(sub_string(FullOutput, _, _, _, "MEMBER_SUCCESS"))
    ;   format("Go not found, skipping execution test.~n")
    ).

:- end_tests(go_wam_builtins).

write_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).
