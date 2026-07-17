:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0

% Executable regression for transactional, correlated foreign-result tuples
% in the generated Go WAM runtime.

:- use_module(library(filesex)).
:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_go_target').

go_available :-
    catch(
        ( process_create(path(go), [version],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _,
        fail).

:- begin_tests(wam_go_foreign_tuple_aliases,
               [condition(go_available)]).

test(correlated_alias_candidates_are_transactional) :-
    Dir = 'output/test_wam_go_foreign_tuple_aliases',
    setup_call_cleanup(
        prepare_go_alias_project(Dir),
        run_go_alias_test(Dir),
        cleanup_test_dir(Dir)).

:- end_tests(wam_go_foreign_tuple_aliases).

prepare_go_alias_project(Dir) :-
    cleanup_test_dir(Dir),
    write_wam_go_project([], [module_name(aliasruntime), package_name(wam)], Dir),
    directory_file_path(Dir, 'foreign_tuple_aliases_test.go', TestPath),
    go_alias_test_source(Source),
    setup_call_cleanup(
        open(TestPath, write, Stream, [encoding(utf8)]),
        format(Stream, '~s', [Source]),
        close(Stream)).

run_go_alias_test(Dir) :-
    absolute_file_name(Dir, AbsDir),
    directory_file_path(AbsDir, '.gocache', CacheDir),
    make_directory_path(CacheDir),
    process_create(path(go), ['test', './...'],
                   [cwd(AbsDir), environment(['GOCACHE'=CacheDir, 'HOME'='/tmp']),
                    stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
    read_string(Out, _, OutText),
    read_string(Err, _, ErrText),
    close(Out),
    close(Err),
    process_wait(Pid, Status),
    (   Status == exit(0)
    ->  true
    ;   throw(error(go_alias_test_failed(Status, OutText, ErrText), _))
    ).

cleanup_test_dir(Dir) :-
    (   exists_directory(Dir)
    ->  delete_directory_and_contents(Dir)
    ;   true
    ).

go_alias_test_source(
"package wam

import \"testing\"

func aliasTuple(left, right string, distance int64) Value {
	return &Compound{Functor: \"__tuple__\", Args: []Value{
		internAtom(left), internAtom(right), &Integer{Val: distance},
	}}
}

func aliasMachine() (*WamState, *Unbound, *Unbound) {
	vm := NewWamState(nil, nil)
	vm.Ctx.ForeignResultLayouts[\"alias/3\"] = \"tuple:3\"
	vm.Ctx.ForeignResultModes[\"alias/3\"] = \"stream\"
	shared := &Unbound{Name: \"Shared\", Idx: vm.allocVarId()}
	distance := &Unbound{Name: \"Distance\", Idx: vm.allocVarId()}
	vm.Regs[0] = shared
	vm.Regs[1] = shared
	vm.Regs[2] = distance
	return vm, shared, distance
}

func requireAliasAtom(t *testing.T, vm *WamState, value Value, expected string) {
	t.Helper()
	actual, ok := vm.deref(value).(*Atom)
	if !ok || actual.Name != expected {
		t.Fatalf(\"expected atom %q, got %T %v\", expected, vm.deref(value), vm.deref(value))
	}
}

func requireAliasInteger(t *testing.T, vm *WamState, value Value, expected int64) {
	t.Helper()
	actual, ok := vm.deref(value).(*Integer)
	if !ok || actual.Val != expected {
		t.Fatalf(\"expected integer %d, got %T %v\", expected, vm.deref(value), vm.deref(value))
	}
}

func TestSkipsIncompatibleFirstTupleWithoutLeak(t *testing.T) {
	vm, shared, distance := aliasMachine()
	results := []Value{
		aliasTuple(\"left\", \"right\", 1),
		aliasTuple(\"same\", \"same\", 2),
	}
	if !vm.finishForeignResults(\"alias/3\", []int{0, 1, 2}, results) {
		t.Fatal(\"expected later alias-compatible tuple to succeed\")
	}
	requireAliasAtom(t, vm, shared, \"same\")
	requireAliasInteger(t, vm, distance, 2)
	if len(vm.ChoicePoints) != 0 {
		t.Fatalf(\"incompatible prefix leaked a choice point: %d\", len(vm.ChoicePoints))
	}
}

func TestRetrySkipsIncompatibleTupleAndFindsLaterMatch(t *testing.T) {
	vm, shared, distance := aliasMachine()
	results := []Value{
		aliasTuple(\"first\", \"first\", 1),
		aliasTuple(\"bad\", \"worse\", 2),
		aliasTuple(\"second\", \"second\", 3),
	}
	if !vm.finishForeignResults(\"alias/3\", []int{0, 1, 2}, results) {
		t.Fatal(\"first tuple should succeed\")
	}
	if len(vm.ChoicePoints) != 1 {
		t.Fatalf(\"expected one retry choice point, got %d\", len(vm.ChoicePoints))
	}
	if !vm.backtrack() {
		t.Fatal(\"retry should skip the incompatible tuple and find the final tuple\")
	}
	requireAliasAtom(t, vm, shared, \"second\")
	requireAliasInteger(t, vm, distance, 3)
	if len(vm.ChoicePoints) != 0 {
		t.Fatalf(\"final result leaked a choice point: %d\", len(vm.ChoicePoints))
	}
}

func TestExhaustedRetryRestoresBindingsTrailAndChoicePoint(t *testing.T) {
	vm, shared, distance := aliasMachine()
	results := []Value{
		aliasTuple(\"first\", \"first\", 1),
		aliasTuple(\"bad\", \"worse\", 2),
	}
	if !vm.finishForeignResults(\"alias/3\", []int{0, 1, 2}, results) {
		t.Fatal(\"first tuple should succeed\")
	}
	if vm.backtrack() {
		t.Fatal(\"the only retry tuple is alias-incompatible\")
	}
	if _, ok := vm.deref(shared).(*Unbound); !ok {
		t.Fatalf(\"shared output binding leaked after exhaustion: %v\", vm.deref(shared))
	}
	if _, ok := vm.deref(distance).(*Unbound); !ok {
		t.Fatalf(\"distance binding leaked after exhaustion: %v\", vm.deref(distance))
	}
	if vm.TrailLen != 0 || len(vm.Trail) != 0 {
		t.Fatalf(\"trail leaked after exhaustion: TrailLen=%d len=%d\", vm.TrailLen, len(vm.Trail))
	}
	if len(vm.ChoicePoints) != 0 {
		t.Fatalf(\"choice point leaked after exhaustion: %d\", len(vm.ChoicePoints))
	}
}
").
