% test_wam_llvm_lowered_ite.pl
%
% Structural tests for LLVM if-then-else / negation / once lowering
% (the shared-structurer conversion that enabled ITE lowering for the
% wam_llvm_lowered_emitter).
%
% LLVM previously did NOT lower any predicate whose clause 1 contained an
% if-then-else (is_deterministic_pred_llvm/1 rejected the internal
% try_me_else, and the soft-cut commit instructions were not in the
% supported set), so such predicates fell back to the WAM bytecode
% interpreter emitted in LLVM IR — sound, but not lowered. Wiring clause-1
% emission through wam_ite_structurer emits native basic-block branching:
% a condition block, a trail-rollback else block, and a shared
% continuation.
%
% NOTE: the LLVM pipeline compiles ITE with ite_use_y_level(true) (the
% default forced by compile_predicates_for_llvm/4), so the soft cut is the
% `cut Yn` form preceded by a `get_level Yn` snapshot — NOT the naive
% cut_ite. These tests therefore use that option to exercise the real path.
%
% These checks assert on the emitted LLVM *text* (the generated basic-block
% structure). End-to-end compile+run verification (clang + the WAM runtime)
% lives in tests/test_wam_llvm_lowered_ite_exec.pl and is gated on the LLVM
% toolchain; this suite runs anywhere SWI-Prolog does.

:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../src/unifyweaver/targets/wam_llvm_lowered_emitter').

:- dynamic user:lite/2.
:- dynamic user:lneg/1.
:- dynamic user:lseqite/3.
:- dynamic user:lnestite/2.

user:lite(X, Y)       :- ( X > 0 -> Y = pos ; Y = nonpos ).
user:lneg(X)          :- \+ X > 0.
user:lseqite(X, Y, Z) :- ( X > 0 -> Y = pos ; Y = nonpos ),
                         ( X > 5 -> Z = big ; Z = small ).
user:lnestite(X, Y)   :- ( X > 0 -> ( X > 10 -> Y = big ; Y = small ) ; Y = neg ).

% Compile with ite_use_y_level(true) — the form the LLVM pipeline uses.
ite_opts([ite_use_y_level(true)]).

lowered_code(PI, Code) :-
    ite_opts(Opts),
    compile_predicate_to_wam(user:PI, Opts, W),
    lower_predicate_to_llvm(PI, W, Opts, Code).

count_substr(Str, Sub, N) :-
    findall(x, sub_atom(Str, _, _, _, Sub), Xs),
    length(Xs, N).

:- begin_tests(wam_llvm_lowered_ite).

% The gate now accepts ITE predicates as single-clause lowerable
% (previously rejected on the internal try_me_else).
test(all_lowerable) :-
    ite_opts(Opts),
    forall(member(PI, [lite/2, lneg/1, lseqite/3, lnestite/2]),
           ( compile_predicate_to_wam(user:PI, Opts, W),
             assertion(wam_llvm_lowerable(PI, W, single_clause)) )).

% Simple ITE emits the condition / else-rollback / continuation block group
% and no leftover structural markers.
test(simple_ite_blocks) :-
    lowered_code(lite/2, Code),
    assertion(sub_atom(Code, _, _, _, 'ite_')),
    assertion(sub_atom(Code, _, _, _, '_else:')),
    assertion(sub_atom(Code, _, _, _, '@unwind_trail(%WamState* %vm')),
    % no structural choice-point markers survived into the emitted body
    % (the soft-cut commit is realised by the block layout, not an instr)
    assertion(\+ sub_atom(Code, _, _, _, 'try_me_else')),
    assertion(\+ sub_atom(Code, _, _, _, 'trust_me')).

% The condition's failure branch is redirected to the else block (not the
% function epilogue %lowered_fail).
test(condition_branches_to_else) :-
    lowered_code(lite/2, Code),
    assertion(sub_atom(Code, _, _, _, ', label %ite_')).

% Negation lowers with the !/0-commit shape: the then-arm is the fail/0
% builtin, the else-arm the true/0 builtin.
test(negation_uses_true_fail) :-
    lowered_code(lneg/1, Code),
    assertion(sub_atom(Code, _, _, _, 'builtin_call fail/0')),
    assertion(sub_atom(Code, _, _, _, 'builtin_call true/0')),
    assertion(sub_atom(Code, _, _, _, '@unwind_trail(%WamState* %vm')).

% get_level Yn is emitted as a no-op (soft cut is structural).
test(get_level_is_noop) :-
    lowered_code(lite/2, Code),
    assertion(sub_atom(Code, _, _, _, 'get_level')),
    assertion(sub_atom(Code, _, _, _, 'no-op')).

% Sequential ITE emits two sibling ite block groups.
test(sequential_two_ites) :-
    lowered_code(lseqite/3, Code),
    count_substr(Code, '_else:', N),
    assertion(N >= 2).

% Nested ITE emits a nested ite group (inner block in the then-arm).
test(nested_two_ites) :-
    lowered_code(lnestite/2, Code),
    count_substr(Code, '_else:', N),
    assertion(N >= 2),
    assertion(\+ sub_atom(Code, _, _, _, 'try_me_else')),
    assertion(\+ sub_atom(Code, _, _, _, 'trust_me')).

% Each lowered ITE function still defines the single shared succeed/fail
% epilogue exactly once (no duplicate basic-block labels → valid SSA).
test(single_epilogue) :-
    lowered_code(lite/2, Code),
    count_substr(Code, 'lowered_succeed:', NS),
    count_substr(Code, 'lowered_fail:', NF),
    assertion(NS =:= 1),
    assertion(NF =:= 1).

:- end_tests(wam_llvm_lowered_ite).
