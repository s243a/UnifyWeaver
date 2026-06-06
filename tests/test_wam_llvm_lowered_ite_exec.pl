% test_wam_llvm_lowered_ite_exec.pl
%
% End-to-end execution test for LLVM if-then-else / negation / once lowering
% (emit_mode(functions)).
%
% Generates a WAM LLVM module with the lowered emitter enabled, compiles it
% with clang together with a small C harness that calls each lowered kernel
% (lowered_<pred>_<arity>(%WamState*)) on a fresh state with the argument
% registers set, and asserts the i1 (success/failure) outcome. Counterpart
% to the Go, Rust, C++, Haskell, F# and Clojure exec tests. Pins:
%
%   * sequential ITEs   — lseqite(10,pos,small) must fail;
%   * nested ITEs       — lnestite/2 (inner block in the then-arm);
%   * negation (\+)      — lneg/1 (commit is the !/0 builtin, then=fail/0,
%                          else=true/0, else runs after a trail rollback);
%   * simple ITEs        — lite/2.
%
% LLVM previously did NOT lower any predicate with an if-then-else in clause 1
% (the gate rejected the internal try_me_else and the soft-cut commit was not
% in the supported set); the shared-structurer conversion enabled it. The
% behaviour was already SOUND before this change — ITE predicates fell back to
% the full WAM interpreter emitted in LLVM IR — so this verifies the native
% basic-block lowering produces the same results. Skipped unless clang is on
% PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

:- dynamic user:lite/2.
:- dynamic user:lneg/1.
:- dynamic user:lseqite/3.
:- dynamic user:lnestite/2.

user:lite(X, Y)       :- ( X > 0 -> Y = pos ; Y = nonpos ).
user:lneg(X)          :- \+ X > 0.
user:lseqite(X, Y, Z) :- ( X > 0 -> Y = pos ; Y = nonpos ),
                         ( X > 5 -> Z = big ; Z = small ).
user:lnestite(X, Y)   :- ( X > 0 -> ( X > 10 -> Y = big ; Y = small ) ; Y = neg ).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_llvm_lowered_ite_exec, [condition(clang_available)]).

test(ite_exec_parity) :-
    Dir = 'output/test_wam_llvm_ite_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    atomic_list_concat([Dir, '/iteproj.ll'], LLPath),
    % 1. Generate the WAM LLVM module with the lowered emitter enabled.
    write_wam_llvm_project(
        [user:lite/2, user:lneg/1, user:lseqite/3, user:lnestite/2],
        [module_name('iteproj'), emit_mode(functions)], LLPath),
    % Sanity: the four predicates must have been lowered natively (not the
    % WAM-interpreter fallback), otherwise the test would pass vacuously.
    read_file_to_string(LLPath, LLSrc, []),
    forall(member(F, ['@lowered_lite_2(', '@lowered_lneg_1(',
                      '@lowered_lseqite_3(', '@lowered_lnestite_2(']),
           assertion(sub_string(LLSrc, _, _, _, F))),
    % 2. Write the C harness next to the module.
    atomic_list_concat([Dir, '/harness.c'], HPath),
    harness_source(HSrc),
    setup_call_cleanup(open(HPath, write, S), write(S, HSrc), close(S)),
    % 3. Compile (clang links the module + harness; -lm for the math builtins
    %    referenced by the WAM arithmetic runtime) and run.
    atomic_list_concat([Dir, '/ite_test'], ExePath),
    format(atom(Cmd),
        'clang -w ~w ~w -o ~w -lm 2>&1 && ~w',
        [HPath, LLPath, ExePath, ExePath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 15 PASS")
    ->  true
    ;   format(user_error, "~n[llvm ite test output]~n~w~n", [OutStr]),
        throw(llvm_test_failed(Status))
    ).

:- end_tests(wam_llvm_lowered_ite_exec).

% Calls each lowered kernel directly on a fresh %WamState with the argument
% registers set via @wam_set_reg_int / @wam_set_reg_atom_id. The atom ids
% (pos=97, nonpos=98, big=99, small=100, neg=101) are the interned ids the
% generated kernels compare against — passing the same id makes =/2 succeed,
% a different id makes it fail. lseqite(10,pos,small)=false and the lnestite
% rows are the sequential / nested discriminators.
harness_source(
"#include <stdio.h>
typedef struct WamState WamState;
extern WamState* wam_state_new(void* code, int n, int* labels, int nl);
extern void wam_set_reg_int(WamState*, int, long);
extern void wam_set_reg_atom_id(WamState*, int, long);
extern unsigned char lowered_lite_2(WamState*);
extern unsigned char lowered_lneg_1(WamState*);
extern unsigned char lowered_lseqite_3(WamState*);
extern unsigned char lowered_lnestite_2(WamState*);

#define POS 97
#define NONPOS 98
#define BIG 99
#define SMALL 100
#define NEG 101

static int fails = 0, total = 0;
static WamState* mk(void){ return wam_state_new(0,0,0,0); }
static int b(unsigned char r){ return r & 1; }
static void check(const char* name, int got, int want){
  total++;
  if(got != want){ fails++; printf(\"FAIL %s: got %d want %d\\n\", name, got, want); }
}

int main(void){
  WamState* s;
  s=mk(); wam_set_reg_int(s,0,5);  wam_set_reg_atom_id(s,1,POS);    check(\"lite(5,pos)\",    b(lowered_lite_2(s)), 1);
  s=mk(); wam_set_reg_int(s,0,5);  wam_set_reg_atom_id(s,1,NONPOS); check(\"lite(5,nonpos)\", b(lowered_lite_2(s)), 0);
  s=mk(); wam_set_reg_int(s,0,-1); wam_set_reg_atom_id(s,1,NONPOS); check(\"lite(-1,nonpos)\",b(lowered_lite_2(s)), 1);
  s=mk(); wam_set_reg_int(s,0,-1); wam_set_reg_atom_id(s,1,POS);    check(\"lite(-1,pos)\",   b(lowered_lite_2(s)), 0);
  s=mk(); wam_set_reg_int(s,0,5);  check(\"lneg(5)\",  b(lowered_lneg_1(s)), 0);
  s=mk(); wam_set_reg_int(s,0,-1); check(\"lneg(-1)\", b(lowered_lneg_1(s)), 1);
  s=mk(); wam_set_reg_int(s,0,0);  check(\"lneg(0)\",  b(lowered_lneg_1(s)), 1);
  s=mk(); wam_set_reg_int(s,0,10); wam_set_reg_atom_id(s,1,POS);    wam_set_reg_atom_id(s,2,BIG);   check(\"lseqite(10,pos,big)\",     b(lowered_lseqite_3(s)), 1);
  s=mk(); wam_set_reg_int(s,0,10); wam_set_reg_atom_id(s,1,POS);    wam_set_reg_atom_id(s,2,SMALL); check(\"lseqite(10,pos,small)\",   b(lowered_lseqite_3(s)), 0);
  s=mk(); wam_set_reg_int(s,0,3);  wam_set_reg_atom_id(s,1,POS);    wam_set_reg_atom_id(s,2,SMALL); check(\"lseqite(3,pos,small)\",    b(lowered_lseqite_3(s)), 1);
  s=mk(); wam_set_reg_int(s,0,-1); wam_set_reg_atom_id(s,1,NONPOS); wam_set_reg_atom_id(s,2,SMALL); check(\"lseqite(-1,nonpos,small)\",b(lowered_lseqite_3(s)), 1);
  s=mk(); wam_set_reg_int(s,0,20); wam_set_reg_atom_id(s,1,BIG);   check(\"lnestite(20,big)\",  b(lowered_lnestite_2(s)), 1);
  s=mk(); wam_set_reg_int(s,0,5);  wam_set_reg_atom_id(s,1,SMALL); check(\"lnestite(5,small)\", b(lowered_lnestite_2(s)), 1);
  s=mk(); wam_set_reg_int(s,0,-1); wam_set_reg_atom_id(s,1,NEG);   check(\"lnestite(-1,neg)\",  b(lowered_lnestite_2(s)), 1);
  s=mk(); wam_set_reg_int(s,0,20); wam_set_reg_atom_id(s,1,SMALL); check(\"lnestite(20,small)\",b(lowered_lnestite_2(s)), 0);

  if(fails==0) printf(\"ALL %d PASS\\n\", total);
  else printf(\"%d FAILURES\\n\", fails);
  return fails ? 1 : 0;
}
").
