% test_wam_cpp_lowered_t5.pl
%
% End-to-end execution test for the C++ T5 lowering — "multi-clause as an
% if-then-else chain" (lowering type T5 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md), ported from the
% Scala/Rust/Go/Haskell/F#/LLVM emitters via the shared wam_clause_chain
% front-end.
%
% A predicate whose clauses discriminate on a DISTINCT first-argument
% constant now lowers to a bound-checked first-arg dispatch over ALL clauses
% (non-first clauses become fast-path too when A1 is bound), instead of
% multi_clause_1 (clause 1 inline, clauses 2+ via the interpreter fallback).
% An unbound first argument returns false and defers to the entry wrapper's
% interpreter fallback.
%
% Pins (the harness preloads a BOUND first arg, exercising every clause incl.
% the non-first ones — the T5 payoff):
%   * color/1 — fact chain, atom discriminators;
%   * sz/2    — fact chain with a second head match in each remainder;
%   * op/2    — RULE chain (each remainder runs an is/2 builtin).
%
% Skipped automatically when no host C++17 compiler is available.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_cpp_target').
:- use_module('../src/unifyweaver/targets/wam_cpp_lowered_emitter').

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

cc_ok(CC) :-
    catch(( process_create(path(CC), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

cpp_compiler(CC) :-
    ( cc_ok('g++') -> CC = 'g++'
    ; cc_ok('clang++') -> CC = 'clang++'
    ; fail ).

:- begin_tests(wam_cpp_lowered_t5, [condition(cpp_compiler(_))]).

% The three predicates must lower as T5 (clause_chain), not multi_clause_1.
test(gate_picks_clause_chain) :-
    forall(member(PI, [color/1, sz/2, op/2]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_cpp_lowerable(PI, W, Reason),
             assertion(Reason == clause_chain) )).

test(t5_exec_parity) :-
    cpp_compiler(CC),
    Dir = 'output/test_wam_cpp_t5_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    % 1. Generate the WAM C++ project with the lowered emitter enabled.
    write_wam_cpp_project(
        [user:color/1, user:sz/2, user:op/2],
        [module_name('t5cpp'), wam_fallback(true), emit_mode(functions)], Dir),
    % Sanity: the generated lowered code must be the T5 dispatch.
    atomic_list_concat([Dir, '/cpp/generated_program.cpp'], GenPath),
    ( exists_file(GenPath) -> read_file_to_string(GenPath, GSrc, []) ; GSrc = "" ),
    assertion(sub_string(GSrc, _, _, _, "T5 first-argument dispatch")),
    % 2. Test harness alongside the generated sources.
    atomic_list_concat([Dir, '/cpp/test_t5.cpp'], TestPath),
    cpp_t5_source(Src),
    setup_call_cleanup(open(TestPath, write, S), write(S, Src), close(S)),
    % 3. Compile + run.
    atomic_list_concat([Dir, '/cpp'], CppDir),
    format(atom(Cmd),
        '~w -std=c++17 -O0 ~w/test_t5.cpp ~w/generated_program.cpp ~w/wam_runtime.cpp -o ~w/t5_test 2>&1 && ~w/t5_test',
        [CC, CppDir, CppDir, CppDir, CppDir, CppDir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 14 PASS")
    ->  true
    ;   format(user_error, "~n[cpp t5 test output]~n~w~n", [OutStr]),
        throw(cpp_t5_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_cpp_lowered_t5).

% Calls each lowered function with a BOUND first argument and asserts the
% boolean outcome. Exercises every clause including the non-first ones
% (green/blue, medium/large, mul/neg) — the T5 payoff — plus the no-match
% cases (yellow, sz big, op div). C++ atoms are plain strings, so no atom-id
% coordination is needed.
cpp_t5_source(
"#include \"wam_runtime.h\"
#include <iostream>
bool lowered_color_1(WamState*); bool lowered_sz_2(WamState*); bool lowered_op_2(WamState*);
static int failures = 0;
static void chk(const char* n, bool g, bool w) {
    if (g != w) { std::cerr << \"FAIL \" << n << \": got \" << g << \" want \" << w << \"\\n\"; failures++; }
}
static Value I(long long n) { return Value::Integer(n); }
static Value A(const char* s) { return Value::Atom(s); }
int main() {
    { WamState v; v.put_reg(\"A1\", A(\"red\"));    chk(\"color(red)\",    lowered_color_1(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"green\"));  chk(\"color(green)\",  lowered_color_1(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"blue\"));   chk(\"color(blue)\",   lowered_color_1(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"yellow\")); chk(\"color(yellow)\", lowered_color_1(&v), false); }
    { WamState v; v.put_reg(\"A1\", A(\"small\"));  v.put_reg(\"A2\", I(1)); chk(\"sz(small,1)\",  lowered_sz_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"medium\")); v.put_reg(\"A2\", I(2)); chk(\"sz(medium,2)\", lowered_sz_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"large\"));  v.put_reg(\"A2\", I(3)); chk(\"sz(large,3)\",  lowered_sz_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"small\"));  v.put_reg(\"A2\", I(2)); chk(\"sz(small,2)\",  lowered_sz_2(&v), false); }
    { WamState v; v.put_reg(\"A1\", A(\"big\"));    v.put_reg(\"A2\", I(1)); chk(\"sz(big,1)\",    lowered_sz_2(&v), false); }
    { WamState v; v.put_reg(\"A1\", A(\"add\")); v.put_reg(\"A2\", I(2));  chk(\"op(add,2)\",  lowered_op_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"mul\")); v.put_reg(\"A2\", I(6));  chk(\"op(mul,6)\",  lowered_op_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"neg\")); v.put_reg(\"A2\", I(-1)); chk(\"op(neg,-1)\", lowered_op_2(&v), true); }
    { WamState v; v.put_reg(\"A1\", A(\"add\")); v.put_reg(\"A2\", I(3));  chk(\"op(add,3)\",  lowered_op_2(&v), false); }
    { WamState v; v.put_reg(\"A1\", A(\"div\")); v.put_reg(\"A2\", I(1));  chk(\"op(div,1)\",  lowered_op_2(&v), false); }
    if (failures == 0) { std::cout << \"ALL 14 PASS\\n\"; return 0; }
    std::cerr << failures << \" FAILURES\\n\"; return 1;
}
").
