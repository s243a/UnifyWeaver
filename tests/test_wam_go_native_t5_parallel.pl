% test_wam_go_native_t5_parallel.pl
%
% T5 first-argument dispatch in the NATIVE clause-parallel Go path.
%
% An order-independent multi-clause predicate whose clauses discriminate on a
% DISTINCT first-argument constant used to compile to a goroutine fan-out
% (one goroutine per clause, a results channel, context cancellation). Because
% at most one such clause can ever match, that parallelism is pure overhead —
% and the goroutine codegen was additionally broken for these shapes (an
% unused `ctx` for arg-less predicates, an undeclared output variable for rule
% clauses). It now lowers to a deterministic `switch arg1 { ... }`, the
% native-path analogue of the WAM-lowered T5 dispatch (wam_clause_chain).
%
% Pins:
%   * sz/2 — fact chain, returns the second argument;
%   * op/2 — RULE chain, each clause computes its output with is/2;
%   * boundary: a body-discriminated order-independent predicate (the
%     first head arg is a variable) still uses the goroutine fan-out.
%
% The exec portion is skipped automatically when `go` is not on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_go_target').
:- use_module('../src/unifyweaver/targets/wam_go_lowered_emitter').
:- use_module('../src/unifyweaver/targets/go_target').
:- use_module('../src/unifyweaver/core/clause_body_analysis').

:- dynamic user:sz/2.
:- dynamic user:op/2.

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

:- begin_tests(wam_go_native_t5_parallel).

% Distinct first-arg constants -> deterministic switch, no goroutines.
test(sz_lowers_to_switch) :-
    Clauses = [ (sz(small, 1)  - true),
                (sz(medium, 2) - true),
                (sz(large, 3)  - true) ],
    go_target:compile_clause_parallel_to_go(sz/2, Clauses, Code),
    assertion(sub_string(Code, _, _, _, "switch arg1")),
    assertion(sub_string(Code, _, _, _, "case \"small\":")),
    assertion(\+ sub_string(Code, _, _, _, "go func()")),
    assertion(\+ sub_string(Code, _, _, _, "sync.WaitGroup")).

% Rule clauses computing the output must return the expression directly
% (regression: previously emitted `arg2 = (1 + 1)` against an undeclared var).
test(op_lowers_to_switch_with_return_expr) :-
    Clauses = [ (op(add, A) - (A is 1 + 1)),
                (op(mul, B) - (B is 2 * 3)),
                (op(neg, C) - (C is 0 - 1)) ],
    go_target:compile_clause_parallel_to_go(op/2, Clauses, Code),
    assertion(sub_string(Code, _, _, _, "switch arg1")),
    assertion(sub_string(Code, _, _, _, "return (1 + 1)")),
    assertion(\+ sub_string(Code, _, _, _, "arg2 = ")),
    assertion(\+ sub_string(Code, _, _, _, "go func()")).

% Boundary: a body-discriminated predicate (variable first head arg) is not
% T5-eligible and keeps the goroutine fan-out.
test(body_discriminated_stays_goroutine) :-
    assertz(clause_body_analysis:order_independent(nc/2)),
    Clauses = [ (nc(X, Y) - (X = red,  Y = 1)),
                (nc(X, Y) - (X = blue, Y = 2)) ],
    go_target:compile_clause_parallel_to_go(nc/2, Clauses, Code),
    retract(clause_body_analysis:order_independent(nc/2)),
    assertion(sub_string(Code, _, _, _, "go func()")),
    assertion(sub_string(Code, _, _, _, "sync.WaitGroup")),
    assertion(\+ sub_string(Code, _, _, _, "switch arg1")).

% End-to-end: the whole native project compiles (the T5 switch is valid Go,
% unlike the goroutine fan-out it replaces) and the functions return the
% right values.
test(t5_native_exec, [condition(go_available)]) :-
    Dir = 'output/test_wam_go_native_t5_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    write_wam_go_project(
        [user:sz/2, user:op/2],
        [module_name('nt5'), wam_fallback(true)], Dir),
    atomic_list_concat([Dir, '/lib.go'], LibPath),
    read_file_to_string(LibPath, LibSrc, []),
    assertion(sub_string(LibSrc, _, _, _, "T5 first-argument dispatch")),
    write_native_t5_test(Dir),
    format(atom(Cmd), 'cd ~w && go test ./... 2>&1', [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0)
    ->  true
    ;   format(user_error, "~n[go test output]~n~w~n", [OutStr]),
        throw(go_native_t5_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_go_native_t5_parallel).

write_native_t5_test(Dir) :-
    atomic_list_concat([Dir, '/t5_native_test.go'], Path),
    Src =
"package wam

import \"testing\"

func TestNativeT5Parallel(t *testing.T) {
	if got := sz(\"small\"); got != 1 { t.Errorf(\"sz(small)=%v want 1\", got) }
	if got := sz(\"medium\"); got != 2 { t.Errorf(\"sz(medium)=%v want 2\", got) }
	if got := sz(\"large\"); got != 3 { t.Errorf(\"sz(large)=%v want 3\", got) }
	if got := op(\"add\"); got != 2 { t.Errorf(\"op(add)=%v want 2\", got) }
	if got := op(\"mul\"); got != 6 { t.Errorf(\"op(mul)=%v want 6\", got) }
	if got := op(\"neg\"); got != -1 { t.Errorf(\"op(neg)=%v want -1\", got) }
	func() {
		defer func() {
			if r := recover(); r == nil { t.Errorf(\"sz(big) should panic\") }
		}()
		sz(\"big\")
	}()
}
",
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]), write(S, Src), close(S)).
