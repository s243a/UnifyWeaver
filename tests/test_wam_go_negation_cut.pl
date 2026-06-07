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

% Clause-cut B0: a plain cut commits the clause and a cut inside a CALLED
% predicate stays local to it. proper_length-style: the var-guard clause
% must reject a partial list. These exercise the call-time B0 cut + the
% GetConstant deref fix together.
:- dynamic user:gpacc/3.
:- dynamic user:gplen/2.
:- dynamic user:gpl_proper/0.
:- dynamic user:gpl_partial/0.
:- dynamic user:gpick/1.
:- dynamic user:gcommit/0.

user:gpacc(L, _, _) :- var(L), !, fail.
user:gpacc([], N, N) :- !.
user:gpacc([_|T], A, N) :- A1 is A + 1, gpacc(T, A1, N).
user:gplen(L, N) :- gpacc(L, 0, N).
user:gpl_proper :- gplen([a,b,c], N), N =:= 3.   % proper list → length 3
user:gpl_partial :- \+ gplen([a,b|_], _).        % partial list → gplen fails → \+ succeeds
user:gpick(a) :- !.
user:gpick(b).
user:gcommit :- gpick(X), X == a.                % neck cut commits to a

% forall over a multi-solution generator. forall drives backtracking by
% failing after each test-success; the generator variable must be rebound
% across backtracks (the register-alias rewrite in bindUnbound is trailed
% so it survives backtrack). Before the fix, forall saw only the first
% element and so wrongly succeeded for the failing cases.
:- dynamic user:gfa_all/0.
:- dynamic user:gfa_neg/0.
:- dynamic user:gfa_posfail/0.
user:gfa_all :- forall(member(X, [1,2,3]), X >= 1).          % all hold → true
user:gfa_neg :- \+ forall(member(X, [1,2,3]), X =:= 1).      % not all → \+ true
user:gfa_posfail :- forall(member(X, [1,2,3]), X =:= 1).     % not all → forall false

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

test(clause_cut_b0_and_partial_list) :-
    Proj = 'output/test_wam_go_b0cut_gen',
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ),
    write_wam_go_project([user:gpacc/3, user:gplen/2,
                          user:gpl_proper/0, user:gpl_partial/0,
                          user:gpick/1, user:gcommit/0],
                         [module_name(b0cut), prefer_wam(true)], Proj),
    directory_file_path(Proj, 'cmd', CmdDir),
    directory_file_path(CmdDir, 'run', RunDir),
    make_directory_path(RunDir),
    directory_file_path(RunDir, 'main.go', MainPath),
    setup_call_cleanup(
        open(MainPath, write, MS),
        write(MS,
'package main\n\nimport (\n\t"fmt"\n\twam "b0cut"\n)\n\nfunc run(code []wam.Instruction, labels map[string]int, pc int) bool {\n\tvm := wam.NewWamState(code, labels)\n\tvm.PC = pc\n\treturn vm.Run()\n}\n\nfunc main() {\n\tfmt.Printf("PROPER=%v\\n", run(wam.Gpl_properCode, wam.Gpl_properLabels, wam.Gpl_properStartPC))\n\tfmt.Printf("PARTIAL=%v\\n", run(wam.Gpl_partialCode, wam.Gpl_partialLabels, wam.Gpl_partialStartPC))\n\tfmt.Printf("COMMIT=%v\\n", run(wam.GcommitCode, wam.GcommitLabels, wam.GcommitStartPC))\n}\n'),
        close(MS)),
    directory_file_path(Proj, 'go.mod', GoModPath),
    read_file_to_string(GoModPath, GoModOld, []),
    atomic_list_concat([GoModOld, "\nreplace b0cut => ../../\n"], GoModNew),
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
    assertion(sub_string(OutStr, _, _, _, "PROPER=true")),
    assertion(sub_string(OutStr, _, _, _, "PARTIAL=true")),
    assertion(sub_string(OutStr, _, _, _, "COMMIT=true")),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ).

test(forall_over_generator) :-
    Proj = 'output/test_wam_go_forall_gen',
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ),
    write_wam_go_project([user:gfa_all/0, user:gfa_neg/0, user:gfa_posfail/0],
                         [module_name(faall), prefer_wam(true)], Proj),
    directory_file_path(Proj, 'cmd', CmdDir),
    directory_file_path(CmdDir, 'run', RunDir),
    make_directory_path(RunDir),
    directory_file_path(RunDir, 'main.go', MainPath),
    setup_call_cleanup(
        open(MainPath, write, MS),
        write(MS,
'package main\n\nimport (\n\t"fmt"\n\twam "faall"\n)\n\nfunc run(code []wam.Instruction, labels map[string]int, pc int) bool {\n\tvm := wam.NewWamState(code, labels)\n\tvm.PC = pc\n\treturn vm.Run()\n}\n\nfunc main() {\n\tfmt.Printf("ALL=%v\\n", run(wam.Gfa_allCode, wam.Gfa_allLabels, wam.Gfa_allStartPC))\n\tfmt.Printf("NEG=%v\\n", run(wam.Gfa_negCode, wam.Gfa_negLabels, wam.Gfa_negStartPC))\n\tfmt.Printf("POSFAIL=%v\\n", run(wam.Gfa_posfailCode, wam.Gfa_posfailLabels, wam.Gfa_posfailStartPC))\n}\n'),
        close(MS)),
    directory_file_path(Proj, 'go.mod', GoModPath),
    read_file_to_string(GoModPath, GoModOld, []),
    atomic_list_concat([GoModOld, "\nreplace faall => ../../\n"], GoModNew),
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
    assertion(sub_string(OutStr, _, _, _, "ALL=true")),
    assertion(sub_string(OutStr, _, _, _, "NEG=true")),
    assertion(sub_string(OutStr, _, _, _, "POSFAIL=false")),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ).

:- end_tests(wam_go_negation_cut).
