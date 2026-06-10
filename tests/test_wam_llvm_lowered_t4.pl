% test_wam_llvm_lowered_t4.pl
%
% End-to-end execution test for the LLVM T4 lowering — "multi-clause, all
% clauses" (lowering type T4 / multi_clause_n in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md), ported from
% Scala/Rust/Go/C++/Haskell/F#/Clojure/Lua.
%
% A multi-clause predicate whose clauses are all supported deterministic
% bodies but do NOT discriminate on a distinct first-argument constant (so T5
% declines) now lowers ALL of its clauses into one LLVM function, instead of
% lowering only clause 1 (multi_clause_c1) and reaching clauses 2+ through the
% bytecode interpreter on backtrack. Each clause is tried in order; ANY
% instruction failure restores the entry register snapshot (memcpy, exactly as
% a bytecode try_me_else choice point saves the registers) and the trail mark,
% then falls through to the next clause. First-solution / deterministic-prefix
% semantics; the interpreter is never entered for the predicate.
%
% Pins (FACT chains with a REPEATED first-argument constant, so they are not
% distinct-first-arg (T5) chains; the get_constant-only bodies avoid the
% emitter's known-broken put_structure/is/2 path):
%   * grade/2 — alice in clauses 1 & 3; grade(alice,c) needs clause 3;
%   * dup/2   — first arg `a` in both clauses; dup(a,2) needs clause 2.
%
% The C harness calls each lowered_<pred>_<arity>(%WamState*) kernel directly
% on a fresh state with the argument registers set, exercising the non-first
% clauses (the T4 payoff) and the restore/fall-through across clauses. Skipped
% unless clang is on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../src/unifyweaver/targets/wam_llvm_lowered_emitter').

:- dynamic user:grade/2.
:- dynamic user:dup/2.

user:grade(alice, a).
user:grade(bob,   b).
user:grade(alice, c).

user:dup(a, 1).
user:dup(a, 2).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_llvm_lowered_t4, [condition(clang_available)]).

test(gate_picks_multi_clause_n) :-
    forall(member(PI, [grade/2, dup/2]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_llvm_lowerable(PI, W, Shape),
             assertion(Shape == multi_clause_n) )).

test(t4_exec_parity) :-
    Dir = 'output/test_wam_llvm_t4_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    atomic_list_concat([Dir, '/t4proj.ll'], LLPath),
    write_wam_llvm_project(
        [user:grade/2, user:dup/2],
        [module_name('t4proj'), emit_mode(functions)], LLPath),
    read_file_to_string(LLPath, LLSrc, []),
    forall(member(F, ['@lowered_grade_2(', '@lowered_dup_2(',
                      "T4 all-clauses inline"]),
           assertion(sub_string(LLSrc, _, _, _, F))),
    maplist(disc_id, [alice, bob, a, b, c], [Alice, Bob, AId, BId, CId]),
    atomic_list_concat([Dir, '/harness.c'], HPath),
    harness_source_t4(Alice, Bob, AId, BId, CId, HSrc),
    setup_call_cleanup(open(HPath, write, S), write(S, HSrc), close(S)),
    atomic_list_concat([Dir, '/t4_test'], ExePath),
    format(atom(Cmd),
        'clang -w ~w ~w -o ~w -lm 2>&1 && ~w',
        [HPath, LLPath, ExePath, ExePath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 10 PASS")
    ->  true
    ;   format(user_error, "~n[llvm t4 test output]~n~w~n", [OutStr]),
        throw(llvm_t4_test_failed(Status))
    ).

:- end_tests(wam_llvm_lowered_t4).

disc_id(Atom, Id) :-
    wam_llvm_lowered_emitter:parse_constant(Atom, 0, Id).

% Calls each lowered kernel directly on a fresh %WamState with the argument
% registers set. Exercises the non-first clauses (grade clauses 2 & 3, dup
% clause 2) — the T4 payoff, reached via the restore/fall-through — plus
% no-match cases (the no-match atom id 999999 is never interned).
harness_source_t4(Alice, Bob, AId, BId, CId, Src) :-
    format(atom(Src),
"#include <stdio.h>
typedef struct WamState WamState;
extern WamState* wam_state_new(void* code, int n, int* labels, int nl);
extern void wam_set_reg_int(WamState*, int, long);
extern void wam_set_reg_atom_id(WamState*, int, long);
extern unsigned char lowered_grade_2(WamState*);
extern unsigned char lowered_dup_2(WamState*);

#define ALICE ~w
#define BOB ~w
#define A ~w
#define B ~w
#define C ~w
#define NOMATCH 999999

static int fails = 0, total = 0;
static WamState* mk(void){ return wam_state_new(0,0,0,0); }
static int b(unsigned char r){ return r & 1; }
static void check(const char* name, int got, int want){
  total++;
  if(got != want){ fails++; printf(\"FAIL %s: got %d want %d\\n\", name, got, want); }
}

int main(void){
  WamState* s;
  s=mk(); wam_set_reg_atom_id(s,0,ALICE); wam_set_reg_atom_id(s,1,A); check(\"grade(alice,a)\", b(lowered_grade_2(s)), 1);
  s=mk(); wam_set_reg_atom_id(s,0,BOB);   wam_set_reg_atom_id(s,1,B); check(\"grade(bob,b)\",   b(lowered_grade_2(s)), 1);
  s=mk(); wam_set_reg_atom_id(s,0,ALICE); wam_set_reg_atom_id(s,1,C); check(\"grade(alice,c)\", b(lowered_grade_2(s)), 1);
  s=mk(); wam_set_reg_atom_id(s,0,ALICE); wam_set_reg_atom_id(s,1,B); check(\"grade(alice,b)\", b(lowered_grade_2(s)), 0);
  s=mk(); wam_set_reg_atom_id(s,0,NOMATCH);wam_set_reg_atom_id(s,1,A);check(\"grade(carol,a)\", b(lowered_grade_2(s)), 0);
  s=mk(); wam_set_reg_atom_id(s,0,BOB);   wam_set_reg_atom_id(s,1,C); check(\"grade(bob,c)\",   b(lowered_grade_2(s)), 0);
  s=mk(); wam_set_reg_atom_id(s,0,A); wam_set_reg_int(s,1,1); check(\"dup(a,1)\", b(lowered_dup_2(s)), 1);
  s=mk(); wam_set_reg_atom_id(s,0,A); wam_set_reg_int(s,1,2); check(\"dup(a,2)\", b(lowered_dup_2(s)), 1);
  s=mk(); wam_set_reg_atom_id(s,0,A); wam_set_reg_int(s,1,3); check(\"dup(a,3)\", b(lowered_dup_2(s)), 0);
  s=mk(); wam_set_reg_atom_id(s,0,B); wam_set_reg_int(s,1,1); check(\"dup(b,1)\", b(lowered_dup_2(s)), 0);

  if(fails==0) printf(\"ALL %d PASS\\n\", total);
  else printf(\"%d FAILURES\\n\", fails);
  return fails ? 1 : 0;
}
",
        [Alice, Bob, AId, BId, CId]).
