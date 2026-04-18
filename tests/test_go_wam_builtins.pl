:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_go_target').
:- use_module(library(filesex)).

:- begin_tests(go_wam_builtins).

test(builtins_execution) :-
    get_time(T),
    format(atom(TmpDir), 'tmp_wam_builtins_~w', [T]),
    setup_call_cleanup(
        assertz(user:test_builtins(X) :- (X is 1 + 2, X < 5, X =:= 3, atom(foo), \+ atom(5))),
        run_builtins_test(TmpDir),
        ( retractall(user:test_builtins(_)),
          delete_directory_and_contents(TmpDir) )
    ).

run_builtins_test(TmpDir) :-
    Predicates = [test_builtins/1],
    Options = [module_name(builtin_test)],

    write_wam_go_project(Predicates, Options, TmpDir),

    % Verify generated code has our fixes
    directory_file_path(TmpDir, 'value.go', ValuePath),
    read_file_to_string(ValuePath, ValueCode, []),
    assertion(sub_string(ValueCode, _, _, _, "type Structure struct")),

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
	x := &wam.Unbound{Name: "X"}
	vm.Regs["A1"] = x

	if vm.Run() {
		fmt.Printf("SUCCESS: X=%v\\n", vm.Deref(x))
	} else {
		fmt.Println("FAILURE")
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
        assertion(sub_string(FullOutput, _, _, _, "SUCCESS: X=3"))
    ;   format("Go not found, skipping execution test.~n")
    ).

:- end_tests(go_wam_builtins).

delete_directory_and_contents(Dir) :-
    (   exists_directory(Dir)
    ->  delete_directory_contents(Dir),
        delete_directory(Dir)
    ;   true
    ).

delete_directory_contents(Dir) :-
    directory_files(Dir, Files),
    member(File, Files),
    File \== '.', File \== '..',
    directory_file_path(Dir, File, Path),
    (   exists_directory(Path)
    ->  delete_directory_and_contents(Path)
    ;   delete_file(Path)
    ),
    fail.
delete_directory_contents(_).

write_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).
