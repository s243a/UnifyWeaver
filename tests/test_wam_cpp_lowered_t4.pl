% test_wam_cpp_lowered_t4.pl
%
% End-to-end execution test for the C++ T4 lowering — "multi-clause, all
% clauses" (lowering type T4 / multi_clause_n in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md), ported from
% Scala/Rust/Go.
%
% A multi-clause predicate whose clauses are all supported deterministic
% bodies but do NOT discriminate on a distinct first-argument constant (so T5
% declines) now lowers to ALL clauses inline: each clause is an
% immediately-invoked lambda, tried in order with a trail/register/env/cut
% restore (vm->lo_restore_clause) between attempts. The first clause that
% succeeds wins (first-solution / deterministic-prefix); the function never
% returns to the interpreter for clauses 2+, unlike multi_clause_1.
%
% Pins (BOUND first arg; the payoff is the non-first clauses running natively):
%   * grade/2 — fact chain with a REPEATED first arg (alice in clauses 1 & 3);
%   * rel/2   — RULE chain with a VARIABLE first arg (=/2 body; clause 1
%               clobbers A2, which the restore must undo).
%
% Skipped automatically when no host C++17 compiler is available.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_cpp_target').
:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').

:- dynamic user:grade/2.
:- dynamic user:rel/2.

user:grade(alice, a).
user:grade(bob,   b).
user:grade(alice, c).

user:rel(X, one) :- X = p.
user:rel(X, two) :- X = q.

cc_ok(CC) :-
    catch(( process_create(path(CC), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

cpp_compiler(CC) :-
    ( cc_ok('g++') -> CC = 'g++'
    ; cc_ok('clang++') -> CC = 'clang++'
    ; fail ).

:- begin_tests(wam_cpp_lowered_t4, [condition(cpp_compiler(_))]).

% Both predicates must lower as T4 (multi_clause_n), not multi_clause_1.
test(gate_picks_multi_clause_n) :-
    forall(member(PI, [grade/2, rel/2]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_cpp_lowerable(PI, W, Reason),
             assertion(Reason == multi_clause_n) )).

test(t4_exec_parity) :-
    cpp_compiler(CC),
    Dir = 'output/test_wam_cpp_t4_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_cpp_project(
        [user:grade/2, user:rel/2],
        [module_name('t4cpp'), wam_fallback(true), emit_mode(functions)], Dir),
    atomic_list_concat([Dir, '/cpp/generated_program.cpp'], GenPath),
    ( exists_file(GenPath) -> read_file_to_string(GenPath, GSrc, []) ; GSrc = "" ),
    assertion(sub_string(GSrc, _, _, _, "T4 all-clauses inline")),
    atomic_list_concat([Dir, '/cpp/test_t4.cpp'], TestPath),
    cpp_t4_source(Src),
    setup_call_cleanup(open(TestPath, write, S), write(S, Src), close(S)),
    atomic_list_concat([Dir, '/cpp'], CppDir),
    format(atom(Cmd),
        '~w -std=c++17 -O0 ~w/test_t4.cpp ~w/generated_program.cpp ~w/wam_runtime.cpp -o ~w/t4_test 2>&1 && ~w/t4_test',
        [CC, CppDir, CppDir, CppDir, CppDir, CppDir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 10 PASS")
    ->  true
    ;   format(user_error, "~n[cpp t4 test output]~n~w~n", [OutStr]),
        throw(cpp_t4_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_cpp_lowered_t4).

% Calls each lowered function with a BOUND first argument and asserts the
% boolean outcome. Exercises the non-first clauses (grade clauses 2 & 3, rel
% clause 2) — the T4 payoff — plus the no-match cases.
cpp_t4_source(
"#include \"wam_runtime.h\"
#include <iostream>
bool lowered_grade_2(WamState*); bool lowered_rel_2(WamState*);
static int failures = 0;
static void chk(const char* n, bool g, bool w) {
    if (g != w) { std::cerr << \"FAIL \" << n << \": got \" << g << \" want \" << w << \"\\n\"; failures++; }
}
static Value A(const char* s) { return Value::Atom(s); }
int main() {
    { WamState v; v.put_reg(\"A1\", A(\"alice\")); v.put_reg(\"A2\", A(\"a\")); chk(\"grade(alice,a)\", lowered_grade_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"bob\"));   v.put_reg(\"A2\", A(\"b\")); chk(\"grade(bob,b)\",   lowered_grade_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"alice\")); v.put_reg(\"A2\", A(\"c\")); chk(\"grade(alice,c)\", lowered_grade_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"alice\")); v.put_reg(\"A2\", A(\"b\")); chk(\"grade(alice,b)\", lowered_grade_2(&v), false); }
    { WamState v; v.put_reg(\"A1\", A(\"carol\")); v.put_reg(\"A2\", A(\"a\")); chk(\"grade(carol,a)\", lowered_grade_2(&v), false); }
    { WamState v; v.put_reg(\"A1\", A(\"bob\"));   v.put_reg(\"A2\", A(\"c\")); chk(\"grade(bob,c)\",   lowered_grade_2(&v), false); }
    { WamState v; v.put_reg(\"A1\", A(\"p\")); v.put_reg(\"A2\", A(\"one\")); chk(\"rel(p,one)\", lowered_rel_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"q\")); v.put_reg(\"A2\", A(\"two\")); chk(\"rel(q,two)\", lowered_rel_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"p\")); v.put_reg(\"A2\", A(\"two\")); chk(\"rel(p,two)\", lowered_rel_2(&v), false); }
    { WamState v; v.put_reg(\"A1\", A(\"q\")); v.put_reg(\"A2\", A(\"one\")); chk(\"rel(q,one)\", lowered_rel_2(&v), false); }
    if (failures == 0) { std::cout << \"ALL 10 PASS\\n\"; return 0; }
    std::cerr << failures << \" FAILURES\\n\"; return 1;
}
").
