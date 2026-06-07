% test_wam_go_lowered_ite_exec.pl
%
% End-to-end execution test for Go if-then-else / negation / once lowering.
%
% Generates a WAM Go project for a set of ITE predicates, compiles the WAM
% runtime + lowered functions with the real `go` toolchain, and runs a Go
% test that calls each lowered function and asserts the boolean result is
% correct. This is the execution-level counterpart to the structural checks
% in test_wam_go_lowered_phase3.pl, and it specifically pins:
%
%   * sequential ITEs   — seqite(10,pos,small) must be false (the old
%                         epilogue heuristic mislowered the 2nd block into
%                         the 1st's else and wrongly returned true);
%   * nested ITEs       — nestite/2;
%   * negation (\+)     — neg/1, which commits with !/0 not cut_ite;
%   * binding condition — undoite/2, whose condition binds a fresh var then
%                         fails, so the trail unwind + register restore +
%                         fresh-var reset must leave it unbound for X = Y.
%
% Skipped automatically when `go` is not on PATH (mirrors the cargo gating
% in test_wam_rust_runtime.pl).

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_go_target').

:- dynamic user:gite/2.
:- dynamic user:gneg/1.
:- dynamic user:gseqite/3.
:- dynamic user:gnestite/2.
:- dynamic user:gundoite/2.

user:gite(X, Y)      :- ( X > 0 -> Y = pos ; Y = nonpos ).
user:gneg(X)         :- \+ X > 0.
user:gseqite(X, Y, Z) :- ( X > 0 -> Y = pos ; Y = nonpos ),
                         ( X > 5 -> Z = big ; Z = small ).
user:gnestite(X, Y)  :- ( X > 0 -> ( X > 10 -> Y = big ; Y = small ) ; Y = neg ).
user:gundoite(X, R)  :- ( (Y = a, Y = b) -> R = then ; R = els ), X = Y.

go_available :-
    catch(
        ( process_create(path(go), ['version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

:- begin_tests(wam_go_lowered_ite_exec, [condition(go_available)]).

test(ite_exec_parity) :-
    Dir = 'output/test_wam_go_ite_exec',
    Proj = 'output/test_wam_go_ite_exec_gen',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    ( exists_directory(Proj) -> delete_directory_and_contents(Proj) ; true ),
    make_directory_path(Dir),
    % 1. Generate the WAM Go project (lowered functions + runtime).
    write_wam_go_project(
        [user:gite/2, user:gneg/1, user:gseqite/3, user:gnestite/2, user:gundoite/2],
        [module_name('iteexec'), wam_fallback(true)], Proj),
    % 1b. The whole generated project must compile. This guards lib.go's
    %     computed imports, the clause-parallel goroutine template, and the
    %     routing of -> / \+ predicates away from the broken native strategy
    %     to WAM (go_pred_has_control_constructs/1).
    format(atom(BuildCmd), 'cd ~w && go build ./... 2>&1', [Proj]),
    process_create(path(sh), ['-c', BuildCmd],
                   [stdout(pipe(BOut)), stderr(std), process(BPid)]),
    read_string(BOut, _, BOutStr), close(BOut),
    process_wait(BPid, BStatus),
    ( BStatus == exit(0)
    ->  true
    ;   format(user_error, "~n[go build ./... output]~n~w~n", [BOutStr]),
        throw(go_project_build_failed(BStatus))
    ),
    % 2. Copy the WAM runtime + lowered sources into a clean module to call
    %    the lowered functions directly. lib.go is excluded only because the
    %    direct-call harness needs no WAM entry points, not because it is
    %    broken (1b already proved the whole project builds).
    atomic_list_concat(
        ['cp ', Proj, '/*.go ', Dir, '/ && rm -f ', Dir, '/lib.go'], CpCmd),
    shell_ok(CpCmd),
    % 3. go.mod + the execution test.
    write_file(Dir/'go.mod', "module iteexec\n\ngo 1.21\n"),
    go_test_source(GoTestSrc),
    write_file(Dir/'ite_exec_test.go', GoTestSrc),
    % 4. Compile + run.
    format(atom(TestCmd), 'cd ~w && go test ./... 2>&1', [Dir]),
    process_create(path(sh), ['-c', TestCmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0)
    ->  true
    ;   format(user_error, "~n[go test output]~n~w~n", [OutStr]),
        throw(go_test_failed(Status))
    ).

:- end_tests(wam_go_lowered_ite_exec).

shell_ok(Cmd) :-
    process_create(path(sh), ['-c', Cmd], [process(Pid)]),
    process_wait(Pid, exit(0)).

write_file(PathSpec, Text) :-
    ( atom(PathSpec) -> Path = PathSpec ; PathSpec = D/F, atomic_list_concat([D, '/', F], Path) ),
    setup_call_cleanup(open(Path, write, S), write(S, Text), close(S)).

% Calls each lowered method (A-registers Regs[0..] hold the query args) and
% asserts the boolean outcome. seqite(10,pos,small)=false and undoite(c,els)
% =true are the discriminating cases for the bugs this fix addresses.
go_test_source(
"package wam

import \"testing\"

func callPred(setup func(vm *WamState), pred func(vm *WamState) bool) bool {
	vm := NewWamState(nil, nil)
	setup(vm)
	return pred(vm)
}

func TestITELowering(t *testing.T) {
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
		{\"gite(5,pos)\", callPred(set(I(5), A(\"pos\")), (*WamState).PredGite2), true},
		{\"gite(5,nonpos)\", callPred(set(I(5), A(\"nonpos\")), (*WamState).PredGite2), false},
		{\"gite(-1,nonpos)\", callPred(set(I(-1), A(\"nonpos\")), (*WamState).PredGite2), true},
		{\"gite(-1,pos)\", callPred(set(I(-1), A(\"pos\")), (*WamState).PredGite2), false},
		{\"gneg(5)\", callPred(set(I(5)), (*WamState).PredGneg1), false},
		{\"gneg(-1)\", callPred(set(I(-1)), (*WamState).PredGneg1), true},
		{\"gneg(0)\", callPred(set(I(0)), (*WamState).PredGneg1), true},
		{\"gseqite(10,pos,big)\", callPred(set(I(10), A(\"pos\"), A(\"big\")), (*WamState).PredGseqite3), true},
		{\"gseqite(10,pos,small)\", callPred(set(I(10), A(\"pos\"), A(\"small\")), (*WamState).PredGseqite3), false},
		{\"gseqite(3,pos,small)\", callPred(set(I(3), A(\"pos\"), A(\"small\")), (*WamState).PredGseqite3), true},
		{\"gseqite(-1,nonpos,small)\", callPred(set(I(-1), A(\"nonpos\"), A(\"small\")), (*WamState).PredGseqite3), true},
		{\"gnestite(20,big)\", callPred(set(I(20), A(\"big\")), (*WamState).PredGnestite2), true},
		{\"gnestite(5,small)\", callPred(set(I(5), A(\"small\")), (*WamState).PredGnestite2), true},
		{\"gnestite(-1,neg)\", callPred(set(I(-1), A(\"neg\")), (*WamState).PredGnestite2), true},
		{\"gnestite(20,small)\", callPred(set(I(20), A(\"small\")), (*WamState).PredGnestite2), false},
		{\"gundoite(c,els)\", callPred(set(A(\"c\"), A(\"els\")), (*WamState).PredGundoite2), true},
		{\"gundoite(c,then)\", callPred(set(A(\"c\"), A(\"then\")), (*WamState).PredGundoite2), false},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf(\"%s: got %v, want %v\", c.name, c.got, c.want)
		}
	}
}
").
