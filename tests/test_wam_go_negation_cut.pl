% test_wam_go_negation_cut.pl
%
% End-to-end execution test for a cut INSIDE a negated goal on the Go WAM
% target. \+ G desugars to (G -> fail ; true), so a cut in G lands in the
% if-then-else CONDITION, which is opaque to cut (local to the condition,
% like call/1). Under ite_use_y_level the Go target now emits M17
% get_level/cut for the condition's cut, so it prunes only the condition's
% own choicepoints and leaves the negation's choicepoint intact.
%
% gnc :- \+ (ggen(_), !, fail).  The cut commits ggen's first solution and
% `fail` then fails the goal, so the negation SUCCEEDS (true). Before the
% M17 get_level/cut + ite_use_y_level change the inline cut escaped and the
% negation wrongly failed.
%
% Skipped automatically when `go` is not on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_go_target').

:- dynamic user:ggen/1.
:- dynamic user:gnc/0.

user:ggen(1).
user:ggen(2).
user:ggen(3).
user:gnc :- \+ (ggen(_), !, fail).

go_available :-
    catch(
        ( process_create(path(go), ['version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

:- begin_tests(wam_go_negation_cut, [condition(go_available)]).

test(inline_cut_in_negation_succeeds) :-
    Proj = 'output/test_wam_go_negcut_gen',
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ),
    write_wam_go_project([user:ggen/1, user:gnc/0],
                         [module_name(negcut), prefer_wam(true)], Proj),
    % Drive the gnc/0 WAM entry and print the boolean result.
    directory_file_path(Proj, 'cmd', CmdDir),
    directory_file_path(CmdDir, 'run', RunDir),
    make_directory_path(RunDir),
    directory_file_path(RunDir, 'main.go', MainPath),
    setup_call_cleanup(
        open(MainPath, write, MS),
        write(MS,
'package main\n\nimport (\n\t"fmt"\n\twam "negcut"\n)\n\nfunc main() {\n\tvm := wam.NewWamState(wam.GncCode, wam.GncLabels)\n\tvm.PC = wam.GncStartPC\n\tif vm.Run() {\n\t\tfmt.Println("GNC=true")\n\t} else {\n\t\tfmt.Println("GNC=false")\n\t}\n}\n'),
        close(MS)),
    directory_file_path(Proj, 'go.mod', GoModPath),
    read_file_to_string(GoModPath, GoModOld, []),
    atomic_list_concat([GoModOld, "\nreplace negcut => ../../\n"], GoModNew),
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
    assertion(sub_string(OutStr, _, _, _, "GNC=true")),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ).

:- end_tests(wam_go_negation_cut).
