% test_wam_go_varid_collision.pl
%
% allocVarId used to start at 1000 and climb through Idx 10000, which
% drivers mint for output Unbounds (`Idx: 10000+i`). After ~9000
% PutVariable/SetVariable cells, Bindings[10002] aliased
% resolve_layered's Selection; sort/2 unified a real Acc with [] and
% the 5k catalog failed. allocVarId now jumps over 10000..10999.
%
%   swipl -q -g run_tests -t halt tests/test_wam_go_varid_collision.pl

:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_go_target').

:- dynamic user:gwalk/1.

user:gwalk([]).
user:gwalk([_|T]) :- gwalk(T).

go_available :-
    catch(
        ( process_create(path(go), ['version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

:- begin_tests(wam_go_varid_collision, [condition(go_available)]).

test(driver_idx_10002_survives_long_walk) :-
    Proj = 'output/test_wam_go_varid_gen',
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ),
    write_wam_go_project([user:gwalk/1],
                         [module_name(varidcol), prefer_wam(true)], Proj),
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
	wam "varidcol"
)

func main() {
	items := make([]wam.Value, 9500)
	for i := range items {
		items[i] = wam.InternAtom("x")
	}
	vm := wam.NewWamState(wam.GwalkCode, wam.GwalkLabels)
	vm.PC = wam.GwalkStartPC
	vm.Regs[0] = &wam.List{Elements: items}
	canary := &wam.Unbound{Name: "Canary", Idx: 10002}
	ok := vm.Run()
	d := vm.Deref(canary)
	_, stillVar := d.(*wam.Unbound)
	fmt.Printf("WALK=%v CANARY_VAR=%v\\n", ok, stillVar)
}
'),
        close(MS)),
    directory_file_path(Proj, 'go.mod', GoModPath),
    read_file_to_string(GoModPath, GoModOld, []),
    atomic_list_concat([GoModOld, "\nreplace varidcol => ../../\n"], GoModNew),
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
    assertion(sub_string(OutStr, _, _, _, "WALK=true")),
    assertion(sub_string(OutStr, _, _, _, "CANARY_VAR=true")),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ).

:- end_tests(wam_go_varid_collision).
