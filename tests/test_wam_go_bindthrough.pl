% test_wam_go_bindthrough.pl
%
% Regression test for the M139/M140 PutStructure A-register bind-through
% class on the Go WAM target: building a goal structure that embeds a
% still-live head variable into the same A register must NOT bind that
% variable to the structure's own cell (cyclic X = f(X), which wrong-fails
% a later X = 1). Found by the cross-target probe sweep after the same
% bug was fixed in Rust; the fix conditions the bind-through on the
% register class (A regs < 100 are staging and never bind; X/Y keep the
% top-down structure-chaining bind-through).
%
% gbt exercises the bug shape; glist guards the class-2 (non-atom list
% head) pattern, which was verified CLEAN on Go and must stay that way;
% glist_chain guards the legitimate X-register bind-through (nested term
% built via set_variable placeholder) that the fix must NOT break.
%
% Skipped automatically when `go` is not on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_go_target').

:- dynamic user:mk1/2.
:- dynamic user:bindthrough/2.
:- dynamic user:gbt/0.
:- dynamic user:lhead/2.
:- dynamic user:glist/0.
:- dynamic user:glist_chain/0.

user:mk1(T, T).
user:bindthrough(X, W) :- mk1(f(X), W), X = 1.
user:gbt :- bindthrough(_X, W), W == f(1).
user:lhead([H|_], H).
user:glist :- lhead([7,8], V), V =:= 7.
user:glist_chain :- mk1(g(f(1), 2), W), W == g(f(1), 2).

go_available :-
    catch(
        ( process_create(path(go), ['version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

:- begin_tests(wam_go_bindthrough, [condition(go_available)]).

test(a_register_bind_through_and_list_heads) :-
    Proj = 'output/test_wam_go_bindthrough_gen',
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ),
    write_wam_go_project([user:mk1/2, user:bindthrough/2, user:gbt/0,
                          user:lhead/2, user:glist/0, user:glist_chain/0],
                         [module_name(bindthrough), prefer_wam(true)], Proj),
    directory_file_path(Proj, 'cmd', CmdDir),
    directory_file_path(CmdDir, 'run', RunDir),
    make_directory_path(RunDir),
    directory_file_path(RunDir, 'main.go', MainPath),
    setup_call_cleanup(
        open(MainPath, write, MS),
        write(MS,
'package main\n\nimport (\n\t"fmt"\n\twam "bindthrough"\n)\n\nfunc run(code []wam.Instruction, labels map[string]int, pc int) bool {\n\tvm := wam.NewWamState(code, labels)\n\tvm.PC = pc\n\treturn vm.Run()\n}\n\nfunc main() {\n\tfmt.Printf("GBT=%v\\n", run(wam.GbtCode, wam.GbtLabels, wam.GbtStartPC))\n\tfmt.Printf("GLIST=%v\\n", run(wam.GlistCode, wam.GlistLabels, wam.GlistStartPC))\n\tfmt.Printf("GCHAIN=%v\\n", run(wam.Glist_chainCode, wam.Glist_chainLabels, wam.Glist_chainStartPC))\n}\n'),
        close(MS)),
    directory_file_path(Proj, 'go.mod', GoModPath),
    read_file_to_string(GoModPath, GoModOld, []),
    atomic_list_concat([GoModOld, "\nreplace bindthrough => ../../\n"], GoModNew),
    setup_call_cleanup(
        open(GoModPath, write, GS),
        write(GS, GoModNew),
        close(GS)),
    format(atom(RunCmd), 'cd ~w && timeout 120 go run main.go 2>&1', [RunDir]),
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
    assertion(sub_string(OutStr, _, _, _, "GBT=true")),
    assertion(sub_string(OutStr, _, _, _, "GLIST=true")),
    assertion(sub_string(OutStr, _, _, _, "GCHAIN=true")),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ).

:- end_tests(wam_go_bindthrough).
