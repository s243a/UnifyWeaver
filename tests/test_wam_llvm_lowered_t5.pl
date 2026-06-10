% test_wam_llvm_lowered_t5.pl
%
% End-to-end execution test for the LLVM T5 lowering — "multi-clause as a
% first-argument dispatch" (lowering type T5 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md), ported from the
% Scala/Rust/Go/Haskell/F# emitters via the shared wam_clause_chain front-end.
%
% A predicate whose clauses discriminate on a DISTINCT first-argument
% constant now lowers ALL of its clauses into one LLVM function as a
% first-argument dispatch, instead of lowering only clause 1 (multi_clause_c1)
% and reaching clauses 2+ through the bytecode interpreter on backtrack. When
% the first argument is bound this is deterministic dispatch with no
% interpreter hop; when it is unbound the kernel returns false and the hybrid
% wrapper re-runs the full bytecode (which enumerates every clause).
%
% Pins (the harness preloads a BOUND first arg, exercising every clause incl.
% the non-first ones — the T5 payoff):
%   * color/1 — fact chain, atom discriminators;
%   * sz/2    — fact chain with a second head match in each remainder.
%
% op/2 (a RULE chain whose remainders run an is/2 over a built +/2 structure)
% also lowers as the T5 dispatch — the test asserts @lowered_op_2 is emitted —
% but it is NOT exercised at runtime: the LLVM lowered emitter's
% put_structure/set_constant -> is/2 path is independently broken (a
% single-clause `d(X,R):-R is X+X` lowered kernel also returns the wrong
% result), a pre-existing limitation unrelated to T5. So op/2 is pinned for
% EMISSION, while color/1 and sz/2 are pinned for EXECUTION.
%
% The C harness calls each lowered_<pred>_<arity>(%WamState*) kernel directly
% on a fresh state with the argument registers set. Atom ids are read back
% from the generated module's intern table via parse_constant/3, so the test
% does not hard-code interning offsets. Skipped unless clang is on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../src/unifyweaver/targets/wam_llvm_lowered_emitter').

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

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_llvm_lowered_t5, [condition(clang_available)]).

test(t5_exec_parity) :-
    Dir = 'output/test_wam_llvm_t5_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    atomic_list_concat([Dir, '/t5proj.ll'], LLPath),
    % 1. Generate the WAM LLVM module with the lowered emitter enabled.
    write_wam_llvm_project(
        [user:color/1, user:sz/2, user:op/2],
        [module_name('t5proj'), emit_mode(functions)], LLPath),
    % Sanity: the three predicates must be lowered as the T5 dispatch.
    read_file_to_string(LLPath, LLSrc, []),
    forall(member(F, ['@lowered_color_1(', '@lowered_sz_2(', '@lowered_op_2(']),
           assertion(sub_string(LLSrc, _, _, _, F))),
    assertion(sub_string(LLSrc, _, _, _, "T5 first-argument dispatch")),
    % 2. Read the interned atom ids the kernels compare against (color + sz;
    %    op/2 is emission-only, see file header).
    maplist(disc_id, [red, green, blue, small, medium, large],
            [R, G, B, Sm, Md, Lg]),
    % 3. Write the C harness.
    atomic_list_concat([Dir, '/harness.c'], HPath),
    harness_source(R, G, B, Sm, Md, Lg, HSrc),
    setup_call_cleanup(open(HPath, write, S), write(S, HSrc), close(S)),
    % 4. Compile + run.
    atomic_list_concat([Dir, '/t5_test'], ExePath),
    format(atom(Cmd),
        'clang -w ~w ~w -o ~w -lm 2>&1 && ~w',
        [HPath, LLPath, ExePath, ExePath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 9 PASS")
    ->  true
    ;   format(user_error, "~n[llvm t5 test output]~n~w~n", [OutStr]),
        throw(llvm_t5_test_failed(Status))
    ).

:- end_tests(wam_llvm_lowered_t5).

%% disc_id(+Atom, -Id) — the interned id the generated kernel compares against.
disc_id(Atom, Id) :-
    wam_llvm_lowered_emitter:parse_constant(Atom, 0, Id).

% Calls each lowered kernel directly on a fresh %WamState with a BOUND first
% argument. Exercises every clause incl. the non-first ones (green/blue,
% medium/large) — the T5 payoff — plus the no-match cases (the no-match atom
% id 999999 is never interned, so it matches no clause).
harness_source(R, G, B, Sm, Md, Lg, Src) :-
    format(atom(Src),
"#include <stdio.h>
typedef struct WamState WamState;
extern WamState* wam_state_new(void* code, int n, int* labels, int nl);
extern void wam_set_reg_int(WamState*, int, long);
extern void wam_set_reg_atom_id(WamState*, int, long);
extern unsigned char lowered_color_1(WamState*);
extern unsigned char lowered_sz_2(WamState*);

#define RED ~w
#define GREEN ~w
#define BLUE ~w
#define SMALL ~w
#define MEDIUM ~w
#define LARGE ~w
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
  s=mk(); wam_set_reg_atom_id(s,0,RED);     check(\"color(red)\",    b(lowered_color_1(s)), 1);
  s=mk(); wam_set_reg_atom_id(s,0,GREEN);   check(\"color(green)\",  b(lowered_color_1(s)), 1);
  s=mk(); wam_set_reg_atom_id(s,0,BLUE);    check(\"color(blue)\",   b(lowered_color_1(s)), 1);
  s=mk(); wam_set_reg_atom_id(s,0,NOMATCH); check(\"color(yellow)\", b(lowered_color_1(s)), 0);
  s=mk(); wam_set_reg_atom_id(s,0,SMALL);  wam_set_reg_int(s,1,1); check(\"sz(small,1)\",  b(lowered_sz_2(s)), 1);
  s=mk(); wam_set_reg_atom_id(s,0,MEDIUM); wam_set_reg_int(s,1,2); check(\"sz(medium,2)\", b(lowered_sz_2(s)), 1);
  s=mk(); wam_set_reg_atom_id(s,0,LARGE);  wam_set_reg_int(s,1,3); check(\"sz(large,3)\",  b(lowered_sz_2(s)), 1);
  s=mk(); wam_set_reg_atom_id(s,0,SMALL);  wam_set_reg_int(s,1,2); check(\"sz(small,2)\",  b(lowered_sz_2(s)), 0);
  s=mk(); wam_set_reg_atom_id(s,0,NOMATCH);wam_set_reg_int(s,1,1); check(\"sz(big,1)\",    b(lowered_sz_2(s)), 0);

  if(fails==0) printf(\"ALL %d PASS\\n\", total);
  else printf(\"%d FAILURES\\n\", fails);
  return fails ? 1 : 0;
}
",
    [R, G, B, Sm, Md, Lg]).
