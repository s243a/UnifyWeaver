% test_wam_fsharp_lowered_ite.pl
%
% Structural tests for F# if-then-else / negation / once lowering
% (the shared-structurer conversion that enabled ITE lowering for F#).
%
% F# previously did NOT lower any predicate whose clause 1 contained an
% if-then-else (the gate's is_deterministic_pred_fs/1 rejected the internal
% try_me_else, so such predicates fell back to the interpreter — sound but
% not lowered). Wiring clause-1 emission through wam_ite_structurer enables
% lowering of simple, sequential, negation and nested ITE.
%
% These checks assert on the emitted F# *text* (the generated match
% structure). End-to-end compile+run verification needs the dotnet/F#
% toolchain (mirroring the GHC test for Haskell) and is left as a gated
% follow-up; this suite runs anywhere SWI-Prolog does.

:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../src/unifyweaver/targets/wam_fsharp_target').
:- use_module('../src/unifyweaver/targets/wam_fsharp_lowered_emitter').

:- dynamic user:fite/2.
:- dynamic user:fneg/1.
:- dynamic user:fseqite/3.
:- dynamic user:fnestite/2.

user:fite(X, Y)       :- ( X > 0 -> Y = pos ; Y = nonpos ).
user:fneg(X)          :- \+ X > 0.
user:fseqite(X, Y, Z) :- ( X > 0 -> Y = pos ; Y = nonpos ),
                         ( X > 5 -> Z = big ; Z = small ).
user:fnestite(X, Y)   :- ( X > 0 -> ( X > 10 -> Y = big ; Y = small ) ; Y = neg ).

lowered_code(PI, Code) :-
    compile_predicate_to_wam(user:PI, [], W),
    lower_predicate_to_fsharp(PI, W, [], lowered(_, _, Code)).

count_substr(Str, Sub, N) :-
    findall(x, sub_string(Str, _, _, _, Sub), Xs),
    length(Xs, N).

:- begin_tests(wam_fsharp_lowered_ite).

% The gate now accepts ITE predicates (previously rejected).
test(all_lowerable) :-
    forall(member(PI, [fite/2, fneg/1, fseqite/3, fnestite/2]),
           ( compile_predicate_to_wam(user:PI, [], W),
             assertion(wam_fsharp_lowerable(PI, W, _)) )).

% Simple ITE emits an F# match with Some/None arms.
test(simple_ite_match) :-
    lowered_code(fite/2, Code),
    assertion(sub_string(Code, _, _, _, "match (")),
    assertion(sub_string(Code, _, _, _, "| Some ")),
    assertion(sub_string(Code, _, _, _, "| None ->")).

% Negation lowers with the !/0-commit shape: then = fail/0, else = true/0.
test(negation_uses_true_fail) :-
    lowered_code(fneg/1, Code),
    assertion(sub_string(Code, _, _, _, "fail/0")),
    assertion(sub_string(Code, _, _, _, "true/0")).

% Sequential ITE emits two sibling match blocks.
test(sequential_two_matches) :-
    lowered_code(fseqite/3, Code),
    count_substr(Code, "match (", N),
    assertion(N >= 2).

% Nested ITE emits a nested match (inner block inside the then-arm).
test(nested_match) :-
    lowered_code(fnestite/2, Code),
    count_substr(Code, "match (", N),
    assertion(N >= 2),
    % no stray choice-point markers leaked into the emitted code
    assertion(\+ sub_string(Code, _, _, _, "try_me_else")),
    assertion(\+ sub_string(Code, _, _, _, "trust_me")).

:- end_tests(wam_fsharp_lowered_ite).
