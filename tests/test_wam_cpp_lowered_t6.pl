% test_wam_cpp_lowered_t6.pl
%
% End-to-end execution test for the C++ T6 lowering — "first-argument
% indexing" (lowering type T6 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md).
%
% T6 shares the wam_clause_chain front-end with T5 but, when the
% discriminators are all atoms AND the clause count meets the gate
% (`t6_min_clauses`, default 8), replaces the T5 if-cascade with a static
% std::unordered_map<std::string,int> looked up by the deref'd first-arg atom
% (via the no-copy match_reg_atom_str), then a switch (jump table) on the
% index. C++ has no native string switch, so the map IS the indexing
% structure: one hash + one compare + a jump instead of up to N derefs +
% string compares. For few clauses the gate declines (the compiler flattens
% the cascade), so T6 only fires above the threshold.
%
% Pins:
%   * shade/1 (12 atom facts) — must lower as T6, dispatching correctly for a
%     first / middle / last clause, a no-match atom, and an unbound / non-atom
%     first argument (both defer/fail -> false).
%   * grade/2 (12 atom RULE clauses, each remainder runs an is/2 builtin) —
%     T6 with a non-trivial remainder.
%   * few/1 (3 atom facts) — below the gate, must STAY T5 (the cascade).
%
% Skipped automatically when no host C++17 compiler is available.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_cpp_target').
:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').

:- dynamic user:shade/1.
:- dynamic user:grade/2.
:- dynamic user:few/1.

user:shade(s01). user:shade(s02). user:shade(s03). user:shade(s04).
user:shade(s05). user:shade(s06). user:shade(s07). user:shade(s08).
user:shade(s09). user:shade(s10). user:shade(s11). user:shade(s12).

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
user:grade(g11, R) :- R is 1 + 10.
user:grade(g12, R) :- R is 1 + 11.

user:few(a). user:few(b). user:few(c).

cc_ok(CC) :-
    catch(( process_create(path(CC), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

cpp_compiler(CC) :-
    ( cc_ok('g++') -> CC = 'g++'
    ; cc_ok('clang++') -> CC = 'clang++'
    ; fail ).

:- begin_tests(wam_cpp_lowered_t6, [condition(cpp_compiler(_))]).

% The many-clause atom predicates lower via the clause_chain front-end; few/1
% does too (it is just gated to the T5 back-end at emit time).
test(gate_picks_clause_chain) :-
    forall(member(PI, [shade/1, grade/2, few/1]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_cpp_lowerable(PI, W, Reason),
             assertion(Reason == clause_chain) )).

% The emitted source must use the map/switch for the many-clause atom
% predicates and the cascade for the few-clause one.
test(emits_t6_for_many_t5_for_few) :-
    Dir = 'output/test_wam_cpp_t6_shape',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_cpp_project(
        [user:shade/1, user:grade/2, user:few/1],
        [module_name('t6shape'), wam_fallback(true), emit_mode(functions)], Dir),
    atomic_list_concat([Dir, '/cpp/generated_program.cpp'], GenPath),
    read_file_to_string(GenPath, Src, []),
    assertion(sub_string(Src, _, _, _, "lowered_shade_1")),
    assertion(sub_string(Src, _, _, _, "(T6 first-argument indexing / map+switch)")),
    assertion(sub_string(Src, _, _, _, "lowered_grade_2")),
    % few/1 (3 clauses, below gate) must stay the T5 cascade.
    assertion(sub_string(Src, _, _, _, "lowered_few_1 — lowered from few/1 (T5 first-argument dispatch)")),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

test(t6_exec_parity) :-
    cpp_compiler(CC),
    Dir = 'output/test_wam_cpp_t6_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_cpp_project(
        [user:shade/1, user:grade/2],
        [module_name('t6cpp'), wam_fallback(true), emit_mode(functions)], Dir),
    atomic_list_concat([Dir, '/cpp/generated_program.cpp'], GenPath),
    ( exists_file(GenPath) -> read_file_to_string(GenPath, GSrc, []) ; GSrc = "" ),
    assertion(sub_string(GSrc, _, _, _, "T6 first-argument indexing")),
    atomic_list_concat([Dir, '/cpp/test_t6.cpp'], TestPath),
    cpp_t6_source(Src),
    setup_call_cleanup(open(TestPath, write, S), write(S, Src), close(S)),
    atomic_list_concat([Dir, '/cpp'], CppDir),
    format(atom(Cmd),
        '~w -std=c++17 -O0 ~w/test_t6.cpp ~w/generated_program.cpp ~w/wam_runtime.cpp -o ~w/t6_test 2>&1 && ~w/t6_test',
        [CC, CppDir, CppDir, CppDir, CppDir, CppDir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 11 PASS")
    ->  true
    ;   format(user_error, "~n[cpp t6 test output]~n~w~n", [OutStr]),
        throw(cpp_t6_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_cpp_lowered_t6).

% Calls each lowered function with a BOUND first argument and asserts the
% boolean outcome: first / middle / last clause hit, a no-match atom, a
% non-atom value, and an unbound register (the last two defer/fail -> false).
cpp_t6_source(
"#include \"wam_runtime.h\"
#include <iostream>
bool lowered_shade_1(WamState*); bool lowered_grade_2(WamState*);
static int failures = 0;
static void chk(const char* n, bool g, bool w) {
    if (g != w) { std::cerr << \"FAIL \" << n << \": got \" << g << \" want \" << w << \"\\n\"; failures++; }
}
static Value I(long long n) { return Value::Integer(n); }
static Value A(const char* s) { return Value::Atom(s); }
int main() {
    { WamState v; v.put_reg(\"A1\", A(\"s01\")); chk(\"shade(s01)\", lowered_shade_1(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"s07\")); chk(\"shade(s07)\", lowered_shade_1(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"s12\")); chk(\"shade(s12)\", lowered_shade_1(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"s99\")); chk(\"shade(s99)\", lowered_shade_1(&v), false); }
    { WamState v; v.put_reg(\"A1\", I(42));     chk(\"shade(42)\",  lowered_shade_1(&v), false); }
    { WamState v;                              chk(\"shade(_)\",   lowered_shade_1(&v), false); }
    { WamState v; v.put_reg(\"A1\", A(\"g01\")); v.put_reg(\"A2\", I(1));  chk(\"grade(g01,1)\",  lowered_grade_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"g06\")); v.put_reg(\"A2\", I(6));  chk(\"grade(g06,6)\",  lowered_grade_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"g12\")); v.put_reg(\"A2\", I(12)); chk(\"grade(g12,12)\", lowered_grade_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"g06\")); v.put_reg(\"A2\", I(9));  chk(\"grade(g06,9)\",  lowered_grade_2(&v), false); }
    { WamState v; v.put_reg(\"A1\", A(\"gxx\")); v.put_reg(\"A2\", I(1));  chk(\"grade(gxx,1)\",  lowered_grade_2(&v), false); }
    if (failures == 0) { std::cout << \"ALL 11 PASS\\n\"; return 0; }
    std::cerr << failures << \" FAILURES\\n\"; return 1;
}
").
