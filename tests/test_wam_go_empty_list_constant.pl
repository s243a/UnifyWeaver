% test_wam_go_empty_list_constant.pl
%
% append([], [], L) used to yield a zero-length *List, while a clause
% head `p([])` compiles to get_constant []. Atom.Equals is pointer-only,
% so get_constant failed and the recursive resolve_pending([], Acc, Acc)
% clause in uw-resolve never fired. Empty-list representations (interned
% [] atom and empty *List) are now equivalent in valueEquals, and
% list-producing builtins emit the interned atom when the result is empty.
%
%   swipl -q -g run_tests -t halt tests/test_wam_go_empty_list_constant.pl

:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_go_target').

:- dynamic user:gempty/1.
:- dynamic user:gappnil/0.
:- dynamic user:gsortnil/0.

user:gempty([]).
user:gappnil :- append([], [], L), gempty(L).
user:gsortnil :- sort([], L), gempty(L).

go_available :-
    catch(
        ( process_create(path(go), ['version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

:- begin_tests(wam_go_empty_list_constant, [condition(go_available)]).

test(append_empty_matches_get_constant_nil) :-
    Proj = 'output/test_wam_go_emptylist_gen',
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ),
    write_wam_go_project([user:gempty/1, user:gappnil/0, user:gsortnil/0],
                         [module_name(emptylist), prefer_wam(true)], Proj),
    directory_file_path(Proj, 'cmd', CmdDir),
    directory_file_path(CmdDir, 'run', RunDir),
    make_directory_path(RunDir),
    directory_file_path(RunDir, 'main.go', MainPath),
    setup_call_cleanup(
        open(MainPath, write, MS),
        write(MS,
'package main

import (
	"fmt"
	wam "emptylist"
)

func run(code []wam.Instruction, labels map[string]int, pc int) bool {
	vm := wam.NewWamState(code, labels)
	vm.PC = pc
	return vm.Run()
}

func main() {
	fmt.Printf("APPNIL=%v\\n", run(wam.GappnilCode, wam.GappnilLabels, wam.GappnilStartPC))
	fmt.Printf("SORTNIL=%v\\n", run(wam.GsortnilCode, wam.GsortnilLabels, wam.GsortnilStartPC))
}
'),
        close(MS)),
    directory_file_path(Proj, 'go.mod', GoModPath),
    read_file_to_string(GoModPath, GoModOld, []),
    atomic_list_concat([GoModOld, "\nreplace emptylist => ../../\n"], GoModNew),
    setup_call_cleanup(
        open(GoModPath, write, GS),
        write(GS, GoModNew),
        close(GS)),
    format(atom(RunCmd), 'cd ~w && go run main.go 2>&1', [RunDir]),
    process_create(path(sh), ['-c', RunCmd],
                   [stdout(pipe(Out)), process(Pid)]),
    read_string(Out, _, OutStr),
    close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0)
    ->  true
    ;   format(user_error, "~n[go run output]~n~w~n", [OutStr]),
        throw(go_run_failed(Status))
    ),
    assertion(sub_string(OutStr, _, _, _, "APPNIL=true")),
    assertion(sub_string(OutStr, _, _, _, "SORTNIL=true")),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ).

:- end_tests(wam_go_empty_list_constant).
