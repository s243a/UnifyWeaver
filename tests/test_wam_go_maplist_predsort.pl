% test_wam_go_maplist_predsort.pl
%
% P3 uw-resolve emits BuiltinCall maplist/N (is_v3/1) and predsort/3
% (cmp_ver/3). Pin the Go runtime ports plus string_codes/2 empty↔[]
% and the UW_WAM_WARN_UNKNOWN Call fallback (A3).
%
%   swipl -q -g run_tests -t halt tests/test_wam_go_maplist_predsort.pl

:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_go_target').

:- dynamic user:is_v3/1.
:- dynamic user:inc1/2.
:- dynamic user:gmapv3/0.
:- dynamic user:gmapdeb/0.
:- dynamic user:gmapinc/0.
:- dynamic user:gscodes/0.
:- dynamic user:empty_codes/1.
:- dynamic user:gpredsort/0.
:- dynamic user:gunknown/0.

user:is_v3(v(_, _, _)).
user:inc1(X, Y) :- Y is X + 1.
user:gmapv3 :- maplist(is_v3, [v(1, 0, 0), v(2, 0, 0)]).
user:gmapdeb :- maplist(is_v3, [deb(0, [], [])]).
user:gmapinc :- maplist(inc1, [1, 2, 3], [2, 3, 4]).
user:empty_codes([]).
user:gscodes :- string_codes('', L), empty_codes(L).
user:gpredsort :- predsort(compare, [v(2, 0, 0), v(1, 0, 0)], [v(1, 0, 0), v(2, 0, 0)]).
user:gunknown :- totally_missing_pred(x).

go_available :-
    catch(
        ( process_create(path(go), ['version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

write_probe_main(MainPath) :-
    setup_call_cleanup(
        open(MainPath, write, MS),
        write(MS,
'package main

import (
	"fmt"
	wam "maplistprobe"
)

func run(code []wam.Instruction, labels map[string]int, pc int) bool {
	vm := wam.NewWamState(code, labels)
	vm.PC = pc
	return vm.Run()
}

func main() {
	fmt.Printf("MAPV3=%v\\n", run(wam.Gmapv3Code, wam.Gmapv3Labels, wam.Gmapv3StartPC))
	fmt.Printf("MAPDEB=%v\\n", run(wam.GmapdebCode, wam.GmapdebLabels, wam.GmapdebStartPC))
	fmt.Printf("MAPINC=%v\\n", run(wam.GmapincCode, wam.GmapincLabels, wam.GmapincStartPC))
	fmt.Printf("SCODES=%v\\n", run(wam.GscodesCode, wam.GscodesLabels, wam.GscodesStartPC))
	fmt.Printf("PREDSORT=%v\\n", run(wam.GpredsortCode, wam.GpredsortLabels, wam.GpredsortStartPC))
	fmt.Printf("UNKNOWN=%v\\n", run(wam.GunknownCode, wam.GunknownLabels, wam.GunknownStartPC))
}
'),
        close(MS)).

:- begin_tests(wam_go_maplist_predsort, [condition(go_available)]).

test(maplist_predsort_string_codes_warn) :-
    Proj = 'output/test_wam_go_maplist_gen',
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ),
    write_wam_go_project(
        [user:is_v3/1, user:inc1/2, user:gmapv3/0, user:gmapdeb/0,
         user:gmapinc/0, user:empty_codes/1, user:gscodes/0,
         user:gpredsort/0, user:gunknown/0],
        [module_name(maplistprobe), prefer_wam(true)], Proj),
    directory_file_path(Proj, 'cmd', CmdDir),
    directory_file_path(CmdDir, 'run', RunDir),
    make_directory_path(RunDir),
    directory_file_path(RunDir, 'main.go', MainPath),
    write_probe_main(MainPath),
    directory_file_path(Proj, 'go.mod', GoModPath),
    read_file_to_string(GoModPath, GoModOld, []),
    atomic_list_concat([GoModOld, "\nreplace maplistprobe => ../../\n"], GoModNew),
    setup_call_cleanup(
        open(GoModPath, write, GS),
        write(GS, GoModNew),
        close(GS)),
    format(atom(RunCmd),
           'cd ~w && UW_WAM_WARN_UNKNOWN=1 go run main.go 2>../../warn.err',
           [RunDir]),
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
    assertion(sub_string(OutStr, _, _, _, "MAPV3=true")),
    assertion(sub_string(OutStr, _, _, _, "MAPDEB=false")),
    assertion(sub_string(OutStr, _, _, _, "MAPINC=true")),
    assertion(sub_string(OutStr, _, _, _, "SCODES=true")),
    assertion(sub_string(OutStr, _, _, _, "PREDSORT=true")),
    assertion(sub_string(OutStr, _, _, _, "UNKNOWN=false")),
    directory_file_path(Proj, 'warn.err', WarnPath),
    read_file_to_string(WarnPath, WarnStr, []),
    assertion(sub_string(WarnStr, _, _, _, "[wam_go]")),
    assertion(sub_string(WarnStr, _, _, _, "unresolved goal")),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ).

:- end_tests(wam_go_maplist_predsort).
