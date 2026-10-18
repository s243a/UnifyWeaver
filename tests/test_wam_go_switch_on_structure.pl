% test_wam_go_switch_on_structure.pl
%
% switch_on_structure table entries are Functor/Arity:Label. Emitting
% them with ConstCase {Val, Label} (the constant-index shape) does not
% compile: StructCase has Functor, not Val. Forced by catalog/6 vs
% catalog/9 first-arg indexing in uw-resolve.
%
%   swipl -q -g run_tests -t halt tests/test_wam_go_switch_on_structure.pl

:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_go_target').

:- dynamic user:gcat/2.
:- dynamic user:gsix/0.
:- dynamic user:gnine/0.

user:gcat(catalog(_,_,_,_,_,_), six).
user:gcat(catalog(_,_,_,_,_,_,_,_,_), nine).
user:gsix :- gcat(catalog(a,b,c,d,e,f), six).
user:gnine :- gcat(catalog(a,b,c,d,e,f,g,h,i), nine).

go_available :-
    catch(
        ( process_create(path(go), ['version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

:- begin_tests(wam_go_switch_on_structure, [condition(go_available)]).

test(catalog6_and_catalog9_both_match) :-
    Proj = 'output/test_wam_go_switchstruct_gen',
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ),
    write_wam_go_project([user:gcat/2, user:gsix/0, user:gnine/0],
                         [module_name(switchstruct), prefer_wam(true)], Proj),
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
	wam "switchstruct"
)

func run(code []wam.Instruction, labels map[string]int, pc int) bool {
	vm := wam.NewWamState(code, labels)
	vm.PC = pc
	return vm.Run()
}

func main() {
	fmt.Printf("SIX=%v\\n", run(wam.GsixCode, wam.GsixLabels, wam.GsixStartPC))
	fmt.Printf("NINE=%v\\n", run(wam.GnineCode, wam.GnineLabels, wam.GnineStartPC))
}
'),
        close(MS)),
    directory_file_path(Proj, 'go.mod', GoModPath),
    read_file_to_string(GoModPath, GoModOld, []),
    atomic_list_concat([GoModOld, "\nreplace switchstruct => ../../\n"], GoModNew),
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
    assertion(sub_string(OutStr, _, _, _, "SIX=true")),
    assertion(sub_string(OutStr, _, _, _, "NINE=true")),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ).

:- end_tests(wam_go_switch_on_structure).
