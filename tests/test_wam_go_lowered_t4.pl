% test_wam_go_lowered_t4.pl
%
% End-to-end execution test for the Go T4 lowering — "multi-clause, all
% clauses" (lowering type T4 / multi_clause_n in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md), ported from Scala/Rust.
%
% A multi-clause predicate whose clauses are all supported deterministic
% bodies but do NOT discriminate on a distinct first-argument constant (so T5
% declines) now lowers to ALL clauses inline: each clause is an
% immediately-invoked func() bool, tried in order with a
% trail/register/heap/stack restore (vm.LoRestoreClause) between attempts. The
% first clause that succeeds wins (first-solution / deterministic-prefix); the
% method never returns to the interpreter for clauses 2+, unlike multi_clause_1.
%
% Pins (BOUND first arg; the payoff is the non-first clauses running natively):
%   * grade/2 — fact chain with a REPEATED first arg (alice in clauses 1 & 3);
%   * rel/2   — RULE chain with a VARIABLE first arg (=/2 body; clause 1
%               clobbers A2, which the restore must undo).
%
% Like the T5 test, the older native clause-parallel lib.go is dropped from
% the build (it is independently broken for these shapes — a pre-existing
% go_target bug unrelated to WAM lowering); the T4 deliverable lives in
% lowered.go. Skipped automatically when `go` is not on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_go_target').
:- use_module('../src/unifyweaver/targets/wam_go_lowered_emitter').

:- dynamic user:grade/2.
:- dynamic user:rel/2.

user:grade(alice, a).
user:grade(bob,   b).
user:grade(alice, c).

user:rel(X, one) :- X = p.
user:rel(X, two) :- X = q.

go_available :-
    catch(
        ( process_create(path(go), ['version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

:- begin_tests(wam_go_lowered_t4, [condition(go_available)]).

test(gate_picks_multi_clause_n) :-
    forall(member(PI, [grade/2, rel/2]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_go_lowerable(PI, W, Reason),
             assertion(Reason == multi_clause_n) )).

test(t4_exec_parity) :-
    Dir  = 'output/test_wam_go_t4_exec',
    Proj = 'output/test_wam_go_t4_exec_gen',
    ( exists_directory(Dir)  -> delete_directory_and_contents(Dir)  ; true ),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ),
    make_directory_path(Dir),
    write_wam_go_project(
        [user:grade/2, user:rel/2],
        [module_name('t4exec'), wam_fallback(true)], Proj),
    atomic_list_concat([Proj, '/lowered.go'], LoweredPath),
    ( exists_file(LoweredPath) -> read_file_to_string(LoweredPath, LSrc, []) ; LSrc = "" ),
    assertion(sub_string(LSrc, _, _, _, "T4 all-clauses inline")),
    atomic_list_concat(['cp ', Proj, '/*.go ', Dir, '/ && rm -f ', Dir, '/lib.go'], CpCmd),
    shell_ok_t4(CpCmd),
    write_file_t4(Dir/'go.mod', "module t4exec\n\ngo 1.21\n"),
    go_t4_source(Src),
    write_file_t4(Dir/'t4_exec_test.go', Src),
    format(atom(TestCmd), 'cd ~w && go test ./... 2>&1', [Dir]),
    process_create(path(sh), ['-c', TestCmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0)
    ->  true
    ;   format(user_error, "~n[go test output]~n~w~n", [OutStr]),
        throw(go_t4_test_failed(Status))
    ),
    ( exists_directory(Dir)  -> delete_directory_and_contents(Dir)  ; true ),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ).

:- end_tests(wam_go_lowered_t4).

shell_ok_t4(Cmd) :-
    process_create(path(sh), ['-c', Cmd], [process(Pid)]),
    process_wait(Pid, exit(0)).

write_file_t4(PathSpec, Text) :-
    ( atom(PathSpec) -> Path = PathSpec ; PathSpec = D/F, atomic_list_concat([D, '/', F], Path) ),
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]), write(S, Text), close(S)).

% Calls each lowered method with the query args loaded into Regs[0..] (first
% arg bound). Exercises the non-first clauses (grade clauses 2 & 3, rel clause
% 2) — the T4 payoff — plus no-match cases.
go_t4_source(
"package wam

import \"testing\"

func callPredT4(setup func(vm *WamState), pred func(vm *WamState) bool) bool {
	vm := NewWamState(nil, nil)
	setup(vm)
	return pred(vm)
}

func TestT4MultiClause(t *testing.T) {
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
		{\"grade(alice,a)\", callPredT4(set(A(\"alice\"), A(\"a\")), (*WamState).PredGrade2), true},
		{\"grade(bob,b)\",   callPredT4(set(A(\"bob\"),   A(\"b\")), (*WamState).PredGrade2), true},
		{\"grade(alice,c)\", callPredT4(set(A(\"alice\"), A(\"c\")), (*WamState).PredGrade2), true},
		{\"grade(alice,b)\", callPredT4(set(A(\"alice\"), A(\"b\")), (*WamState).PredGrade2), false},
		{\"grade(carol,a)\", callPredT4(set(A(\"carol\"), A(\"a\")), (*WamState).PredGrade2), false},
		{\"grade(bob,c)\",   callPredT4(set(A(\"bob\"),   A(\"c\")), (*WamState).PredGrade2), false},
		{\"rel(p,one)\", callPredT4(set(A(\"p\"), A(\"one\")), (*WamState).PredRel2), true},
		{\"rel(q,two)\", callPredT4(set(A(\"q\"), A(\"two\")), (*WamState).PredRel2), true},
		{\"rel(p,two)\", callPredT4(set(A(\"p\"), A(\"two\")), (*WamState).PredRel2), false},
		{\"rel(q,one)\", callPredT4(set(A(\"q\"), A(\"one\")), (*WamState).PredRel2), false},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf(\"%s: got %v want %v\", c.name, c.got, c.want)
		}
	}
}
").
