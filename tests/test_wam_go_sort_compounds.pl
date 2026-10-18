% test_wam_go_sort_compounds.pl
%
% sort/2 used compareValues, which ranks atoms/numbers and treats every
% compound as equal. ISO sort is unique, so sort([a-1,b-1,c-1], L) kept
% one pair. uw-resolve's sort(Acc, Selection) then dropped every pick
% but the last.
%
%   swipl -q -g run_tests -t halt tests/test_wam_go_sort_compounds.pl

:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_go_target').

:- dynamic user:gsortpairs/1.
:- dynamic user:gthreepairs/0.

user:gsortpairs(L) :- sort([c-1, a-1, b-1], L).
user:gthreepairs :- gsortpairs(L), L = [a-1, b-1, c-1].

go_available :-
    catch(
        ( process_create(path(go), ['version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

:- begin_tests(wam_go_sort_compounds, [condition(go_available)]).

test(sort_unique_keeps_distinct_pairs) :-
    Proj = 'output/test_wam_go_sortpairs_gen',
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ),
    write_wam_go_project([user:gsortpairs/1, user:gthreepairs/0],
                         [module_name(sortpairs), prefer_wam(true)], Proj),
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
	wam "sortpairs"
)

func run(code []wam.Instruction, labels map[string]int, pc int) bool {
	vm := wam.NewWamState(code, labels)
	vm.PC = pc
	return vm.Run()
}

func main() {
	fmt.Printf("THREE=%v\\n", run(wam.GthreepairsCode, wam.GthreepairsLabels, wam.GthreepairsStartPC))
}
'),
        close(MS)),
    directory_file_path(Proj, 'go.mod', GoModPath),
    read_file_to_string(GoModPath, GoModOld, []),
    atomic_list_concat([GoModOld, "\nreplace sortpairs => ../../\n"], GoModNew),
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
    assertion(sub_string(OutStr, _, _, _, "THREE=true")),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ).

:- end_tests(wam_go_sort_compounds).
