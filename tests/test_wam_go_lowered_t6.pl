% test_wam_go_lowered_t6.pl
%
% End-to-end test for the Go T6 lowering — first-argument indexing (native Go
% string `switch`), lowering type T6 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md.
%
% Go represents atoms as `&Atom{Name: "..."}` interned BY NAME, so the
% discriminator is keyed by string — Go is an ATOM-KEYED target (like Rust/C++/
% F#), not int-interned. Its T6 back-end is a native Go `switch t6atom.Name`
% (which the Go compiler lowers to a hash/length jump) replacing the linear
% `valueEquals` if-cascade. Benchmarked O(1) vs the cascade's O(n): 4.8x at 8
% clauses, 31x at 64, 59x at 256 (see docs/reports/wam_go_dispatch_t6_perf.md).
%
% Gated like Rust/C++/F#: T6 fires only when every clause discriminates on a
% distinct ATOM and there are >= t6_min_clauses of them (default 8). Below the
% threshold the few-clause predicate stays the T5 cascade.
%
% The build drops lib.go — the older native clause-parallel strategy, which is
% independently broken for these predicate shapes (a pre-existing go_target
% native-codegen bug unrelated to WAM lowering). The T6 deliverable lives
% entirely in lowered.go, mirroring how the Go T5 test isolates lowered.go.
%
% Skipped automatically when `go` is not on PATH.

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_go_target').
:- use_module('../src/unifyweaver/targets/wam_go_lowered_emitter').

:- dynamic user:shade/1, user:grade/2, user:few/1.

user:shade(s01). user:shade(s02). user:shade(s03). user:shade(s04).
user:shade(s05). user:shade(s06). user:shade(s07). user:shade(s08).
user:shade(s09). user:shade(s10).

user:grade(g01, R) :- R is 1 + 0.
user:grade(g02, R) :- R is 1 + 1.
user:grade(g03, R) :- R is 1 + 2.
user:grade(g04, R) :- R is 1 + 3.
user:grade(g05, R) :- R is 1 + 4.
user:grade(g06, R) :- R is 1 + 5.
user:grade(g07, R) :- R is 1 + 6.
user:grade(g08, R) :- R is 1 + 7.
user:grade(g09, R) :- R is 1 + 8.
user:grade(g10, R) :- R is 1 + 9.

user:few(a). user:few(b). user:few(c).

go_available :-
    catch(( process_create(path(go), ['version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_go_lowered_t6, [condition(go_available)]).

% The many-clause atom predicates emit the native string switch (T6); the
% few-clause one stays the if-cascade (T5). Threshold is configurable.
test(gate_picks_t6_for_many_t5_for_few) :-
    wam_target:compile_predicate_to_wam(shade/1, [], Ws),
    lower_predicate_to_go(shade/1, Ws, [], ShadeLines),
    atomic_list_concat(ShadeLines, '\n', ShadeCode),
    assertion(sub_string(ShadeCode, _, _, _, "T6 first-argument indexing")),
    assertion(sub_string(ShadeCode, _, _, _, "switch t6atom.Name")),
    wam_target:compile_predicate_to_wam(grade/2, [], Wg),
    lower_predicate_to_go(grade/2, Wg, [], GradeLines),
    atomic_list_concat(GradeLines, '\n', GradeCode),
    assertion(sub_string(GradeCode, _, _, _, "T6 first-argument indexing")),
    wam_target:compile_predicate_to_wam(few/1, [], Wf),
    lower_predicate_to_go(few/1, Wf, [], FewLines),
    atomic_list_concat(FewLines, '\n', FewCode),
    assertion(sub_string(FewCode, _, _, _, "T5 first-argument dispatch")),
    assertion(\+ sub_string(FewCode, _, _, _, "T6 first-argument indexing")),
    % threshold override: few/1 lowers as T6 when the gate is lowered to 3.
    lower_predicate_to_go(few/1, Wf, [t6_min_clauses(3)], FewT6Lines),
    atomic_list_concat(FewT6Lines, '\n', FewT6Code),
    assertion(sub_string(FewT6Code, _, _, _, "T6 first-argument indexing")).

% Build + run: the T6 native-switch dispatch returns the Prolog-correct result
% for clause hits (including non-first clauses), no-match, and a grade remainder
% (guard-body) mismatch.
test(t6_exec) :-
    Proj = 'output/test_wam_go_t6_gen',
    Dir = 'output/test_wam_go_t6_exec',
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    write_wam_go_project(
        [user:shade/1, user:grade/2, user:few/1],
        [module_name('t6exec'), wam_fallback(true)], Proj),
    atomic_list_concat([Proj, '/lowered.go'], LoweredPath),
    read_file_to_string(LoweredPath, LSrc, []),
    assertion(sub_string(LSrc, _, _, _, "T6 first-argument indexing")),
    % Build lowered.go + runtime in isolation (drop the broken lib.go).
    atomic_list_concat(['cp ', Proj, '/*.go ', Dir, '/ && rm -f ', Dir, '/lib.go'], CpCmd),
    shell_ok_t6(CpCmd),
    write_file_t6(Dir/'go.mod', "module t6exec\n\ngo 1.21\n"),
    go_t6_source(Src),
    write_file_t6(Dir/'t6_exec_test.go', Src),
    format(atom(TestCmd), 'cd ~w && go test ./... 2>&1', [Dir]),
    process_create(path(sh), ['-c', TestCmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0)
    ->  true
    ;   format(user_error, "~n[go t6 test output]~n~w~n", [OutStr]),
        throw(go_t6_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ).

:- end_tests(wam_go_lowered_t6).

shell_ok_t6(Cmd) :-
    process_create(path(sh), ['-c', Cmd], [process(Pid)]),
    process_wait(Pid, exit(0)).

write_file_t6(PathSpec, Text) :-
    ( atom(PathSpec) -> Path = PathSpec ; PathSpec = D/F, atomic_list_concat([D, '/', F], Path) ),
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]), write(S, Text), close(S)).

% Harness: bound first arg loaded into Regs[0..], call each lowered method.
% Cases cover clause-1, several non-first clauses (the native-switch payoff),
% no-match, a grade remainder mismatch, and few/1 (the T5 control).
go_t6_source(
"package wam

import \"testing\"

func callPredT6(setup func(vm *WamState), pred func(vm *WamState) bool) bool {
	vm := NewWamState(nil, nil)
	setup(vm)
	return pred(vm)
}

func TestT6Dispatch(t *testing.T) {
	I := func(n int64) Value { return &Integer{Val: n} }
	A := func(s string) Value { return internAtom(s) }
	set := func(vals ...Value) func(vm *WamState) {
		return func(vm *WamState) {
			for i, v := range vals {
				vm.Regs[i] = v
			}
		}
	}
	cases := []struct {
		name string
		got  bool
		want bool
	}{
		{\"shade(s01)\", callPredT6(set(A(\"s01\")), (*WamState).PredShade1), true},
		{\"shade(s05)\", callPredT6(set(A(\"s05\")), (*WamState).PredShade1), true},
		{\"shade(s10)\", callPredT6(set(A(\"s10\")), (*WamState).PredShade1), true},
		{\"shade(zz)\", callPredT6(set(A(\"zz\")), (*WamState).PredShade1), false},
		{\"grade(g01,1)\", callPredT6(set(A(\"g01\"), I(1)), (*WamState).PredGrade2), true},
		{\"grade(g05,5)\", callPredT6(set(A(\"g05\"), I(5)), (*WamState).PredGrade2), true},
		{\"grade(g10,10)\", callPredT6(set(A(\"g10\"), I(10)), (*WamState).PredGrade2), true},
		{\"grade(g05,9)\", callPredT6(set(A(\"g05\"), I(9)), (*WamState).PredGrade2), false},
		{\"grade(zz,1)\", callPredT6(set(A(\"zz\"), I(1)), (*WamState).PredGrade2), false},
		{\"few(b)\", callPredT6(set(A(\"b\")), (*WamState).PredFew1), true},
		{\"few(z)\", callPredT6(set(A(\"z\")), (*WamState).PredFew1), false},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf(\"%s: got %v want %v\", c.name, c.got, c.want)
		}
	}
}
").
