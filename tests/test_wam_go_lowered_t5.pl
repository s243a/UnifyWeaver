% test_wam_go_lowered_t5.pl
%
% End-to-end execution test for the Go T5 lowering — "multi-clause as an
% if-then-else chain" (lowering type T5 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md), ported from Scala/Rust
% via the shared wam_clause_chain front-end.
%
% A predicate whose clauses discriminate on a DISTINCT first-argument
% constant now lowers to a bound-checked first-arg dispatch over ALL clauses
% (non-first clauses become fast-path too when A1 is bound); an unbound first
% argument returns false and defers to the entry wrapper's interpreter
% fallback. Pins (BOUND first arg, exercising every clause incl. non-first):
%   * color/1 — fact chain;
%   * sz/2    — second head match in each remainder;
%   * op/2    — RULE chain (each remainder runs an is/2 builtin).
%
% Skipped automatically when `go` is not on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_go_target').
:- use_module('../src/unifyweaver/targets/wam_go_lowered_emitter').

:- dynamic user:color/1.
:- dynamic user:sz/2.
:- dynamic user:op/2.

user:color(red).
user:color(green).
user:color(blue).

user:sz(small, 1).
user:sz(medium, 2).
user:sz(large, 3).

user:op(add, R) :- R is 1 + 1.
user:op(mul, R) :- R is 2 * 3.
user:op(neg, R) :- R is 0 - 1.

go_available :-
    catch(
        ( process_create(path(go), ['version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

:- begin_tests(wam_go_lowered_t5, [condition(go_available)]).

test(gate_picks_clause_chain) :-
    forall(member(PI, [color/1, sz/2, op/2]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_go_lowerable(PI, W, Reason),
             assertion(Reason == clause_chain) )).

test(t5_exec_parity) :-
    Dir = 'output/test_wam_go_t5_exec',
    Proj = 'output/test_wam_go_t5_exec_gen',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ),
    make_directory_path(Dir),
    write_wam_go_project(
        [user:color/1, user:sz/2, user:op/2],
        [module_name('t5exec'), wam_fallback(true)], Proj),
    % Sanity: the lowered code must be the T5 first-argument dispatch (not the
    % old multi_clause_1 fast path).
    atomic_list_concat([Proj, '/lowered.go'], LoweredPath),
    ( exists_file(LoweredPath) -> read_file_to_string(LoweredPath, LSrc, []) ; LSrc = "" ),
    assertion(sub_string(LSrc, _, _, _, "T5 first-argument dispatch")),
    % Copy the runtime + lowered sources into a clean module and DROP lib.go.
    % lib.go is the older native clause-parallel strategy (goroutines), which
    % is independently broken for these predicate shapes (it emits `arg2 = ...`
    % against an undeclared variable for rule clauses, and an unused `ctx` for
    % arg-less fact predicates) — a pre-existing go_target native-codegen bug
    % unrelated to WAM lowering. The T5 deliverable lives entirely in
    % lowered.go, so we build and exercise that in isolation, mirroring how the
    % Rust/Scala T5 tests target their lowered functions.
    atomic_list_concat(['cp ', Proj, '/*.go ', Dir, '/ && rm -f ', Dir, '/lib.go'], CpCmd),
    shell_ok_t5(CpCmd),
    write_file_t5(Dir/'go.mod', "module t5exec\n\ngo 1.21\n"),
    go_t5_source(Src),
    write_file_t5(Dir/'t5_exec_test.go', Src),
    format(atom(TestCmd), 'cd ~w && go test ./... 2>&1', [Dir]),
    process_create(path(sh), ['-c', TestCmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0)
    ->  true
    ;   format(user_error, "~n[go test output]~n~w~n", [OutStr]),
        throw(go_t5_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ).

:- end_tests(wam_go_lowered_t5).

shell_ok_t5(Cmd) :-
    process_create(path(sh), ['-c', Cmd], [process(Pid)]),
    process_wait(Pid, exit(0)).

write_file_t5(PathSpec, Text) :-
    ( atom(PathSpec) -> Path = PathSpec ; PathSpec = D/F, atomic_list_concat([D, '/', F], Path) ),
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]), write(S, Text), close(S)).

% Calls each lowered method with the query args loaded into Regs[0..] (first
% arg bound). Exercises every clause incl. non-first (green/blue, medium/
% large, mul/neg) — the T5 payoff — plus the no-match cases.
go_t5_source(
"package wam

import \"testing\"

func callPredT5(setup func(vm *WamState), pred func(vm *WamState) bool) bool {
	vm := NewWamState(nil, nil)
	setup(vm)
	return pred(vm)
}

func TestT5Dispatch(t *testing.T) {
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
		{\"color(red)\", callPredT5(set(A(\"red\")), (*WamState).PredColor1), true},
		{\"color(green)\", callPredT5(set(A(\"green\")), (*WamState).PredColor1), true},
		{\"color(blue)\", callPredT5(set(A(\"blue\")), (*WamState).PredColor1), true},
		{\"color(yellow)\", callPredT5(set(A(\"yellow\")), (*WamState).PredColor1), false},
		{\"sz(small,1)\", callPredT5(set(A(\"small\"), I(1)), (*WamState).PredSz2), true},
		{\"sz(medium,2)\", callPredT5(set(A(\"medium\"), I(2)), (*WamState).PredSz2), true},
		{\"sz(large,3)\", callPredT5(set(A(\"large\"), I(3)), (*WamState).PredSz2), true},
		{\"sz(small,2)\", callPredT5(set(A(\"small\"), I(2)), (*WamState).PredSz2), false},
		{\"sz(big,1)\", callPredT5(set(A(\"big\"), I(1)), (*WamState).PredSz2), false},
		{\"op(add,2)\", callPredT5(set(A(\"add\"), I(2)), (*WamState).PredOp2), true},
		{\"op(mul,6)\", callPredT5(set(A(\"mul\"), I(6)), (*WamState).PredOp2), true},
		{\"op(neg,-1)\", callPredT5(set(A(\"neg\"), I(-1)), (*WamState).PredOp2), true},
		{\"op(add,3)\", callPredT5(set(A(\"add\"), I(3)), (*WamState).PredOp2), false},
		{\"op(div,1)\", callPredT5(set(A(\"div\"), I(1)), (*WamState).PredOp2), false},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf(\"%s: got %v want %v\", c.name, c.got, c.want)
		}
	}
}
").
