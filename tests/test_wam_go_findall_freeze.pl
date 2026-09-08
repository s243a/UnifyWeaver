% test_wam_go_findall_freeze.pl
%
% findall/bagof run the inner goal on a Clone(). Collecting
% sub.deref(template) is shallow: nested Unbounds stay bound only in
% the clone's Bindings table. After the clone is discarded the parent
% sees those slots as unbound. freezeTerm copies the template with
% every Ref/Unbound chased first.
%
%   swipl -q -g run_tests -t halt tests/test_wam_go_findall_freeze.pl

:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_go_target').

:- dynamic user:gfp/1.
:- dynamic user:gfbag/1.
:- dynamic user:gfok/0.

user:gfp(a).
user:gfp(b).
user:gfbag(L) :- findall(f(X), gfp(X), L).
user:gfok :- gfbag([f(a), f(b)]).

go_available :-
    catch(
        ( process_create(path(go), ['version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

:- begin_tests(wam_go_findall_freeze, [condition(go_available)]).

test(findall_compound_template_keeps_bindings) :-
    Proj = 'output/test_wam_go_findall_gen',
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ),
    write_wam_go_project([user:gfp/1, user:gfbag/1, user:gfok/0],
                         [module_name(findfreeze), prefer_wam(true)], Proj),
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
	wam "findfreeze"
)

func run(code []wam.Instruction, labels map[string]int, pc int) bool {
	vm := wam.NewWamState(code, labels)
	vm.PC = pc
	return vm.Run()
}

func main() {
	fmt.Printf("FOK=%v\\n", run(wam.GfokCode, wam.GfokLabels, wam.GfokStartPC))
}
'),
        close(MS)),
    directory_file_path(Proj, 'go.mod', GoModPath),
    read_file_to_string(GoModPath, GoModOld, []),
    atomic_list_concat([GoModOld, "\nreplace findfreeze => ../../\n"], GoModNew),
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
    assertion(sub_string(OutStr, _, _, _, "FOK=true")),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ).

:- end_tests(wam_go_findall_freeze).
