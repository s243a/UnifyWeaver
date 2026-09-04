% test_wam_go_switch_default_chain.pl
%
% pick_need/8 indexes on Mode: classic is the switch "default" and has
% TWO clauses (real package, then provider). Jumping to
% indexedClauseBodyStart(PC+1) skipped TryMeElse and kept only the first
% body, so a virtual-only Provides resolve failed. Default must fall
% through into the try/retry chain.
%
%   swipl -q -g run_tests -t halt tests/test_wam_go_switch_default_chain.pl

:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_go_target').

:- dynamic user:gpick/3.
:- dynamic user:greal/0.
:- dynamic user:gvirt/0.
:- dynamic user:gfailreal/0.

user:gpick(classic, Name, Name) :- Name == real, !.
user:gpick(classic, _Name, provider).
user:gpick(layered, _Name, layer).
user:greal :- gpick(classic, real, real).
user:gvirt :- gpick(classic, virt, provider).
user:gfailreal :- gpick(classic, virt, real).

go_available :-
    catch(
        ( process_create(path(go), ['version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

:- begin_tests(wam_go_switch_default_chain, [condition(go_available)]).

test(default_key_tries_second_clause) :-
    Proj = 'output/test_wam_go_switchdef_gen',
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ),
    write_wam_go_project(
        [user:gpick/3, user:greal/0, user:gvirt/0, user:gfailreal/0],
        [module_name(switchdef), prefer_wam(true)], Proj),
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
	wam "switchdef"
)

func run(code []wam.Instruction, labels map[string]int, pc int) bool {
	vm := wam.NewWamState(code, labels)
	vm.PC = pc
	return vm.Run()
}

func main() {
	fmt.Printf("REAL=%v\\n", run(wam.GrealCode, wam.GrealLabels, wam.GrealStartPC))
	fmt.Printf("VIRT=%v\\n", run(wam.GvirtCode, wam.GvirtLabels, wam.GvirtStartPC))
	fmt.Printf("FAILREAL=%v\\n", run(wam.GfailrealCode, wam.GfailrealLabels, wam.GfailrealStartPC))
}
'),
        close(MS)),
    directory_file_path(Proj, 'go.mod', GoModPath),
    read_file_to_string(GoModPath, GoModOld, []),
    atomic_list_concat([GoModOld, "\nreplace switchdef => ../../\n"], GoModNew),
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
    assertion(sub_string(OutStr, _, _, _, "REAL=true")),
    assertion(sub_string(OutStr, _, _, _, "VIRT=true")),
    assertion(sub_string(OutStr, _, _, _, "FAILREAL=false")),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ).

:- end_tests(wam_go_switch_default_chain).
