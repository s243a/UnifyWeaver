:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_rust_builtin_dispatch.pl
%
% D124: the Rust WAM `execute_builtin` matches the four core builtins
% (true/0, fail/0, !/0, =/2) BEFORE the builtin family cascade
% (arith, io, type, term, ext, meta). Before D124 they were the last arms,
% reached only after every family match had failed. Moving them first is exact
% only while no family claims one of these names. This test pins that, and
% the shape of the emitted dispatcher, at the generator level. The generated
% crate's `mod d124_builtin_dispatch_tests` checks the same at runtime.
%
%   swipl -q -g run_tests -t halt tests/test_wam_rust_builtin_dispatch.pl

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_rust_target').

core_builtin('true/0').
core_builtin('fail/0').
core_builtin('!/0').
core_builtin('=/2').

family(arith).
family(io).
family(type).
family(term).
family(ext).
family(meta).

family_code(Family, Code) :-
    atomic_list_concat([compile_execute_, Family, '_builtin_to_rust'], Pred),
    Goal =.. [Pred, Code],
    call(wam_rust_target:Goal).

%  claims_pattern(+Code, +Name): the quoted literal "Name" appears in a match
%  pattern position: followed (after whitespace) by `=>` or `|`, or preceded
%  (after whitespace) by `|`. This over-approximates (an inner match or a
%  `||` would also count), which can only make the disjointness test
%  stricter.
claims_pattern(Code, Name) :-
    format(atom(Quoted), '"~w"', [Name]),
    sub_atom(Code, Before, Len, _, Quoted),
    End is Before + Len,
    (   sub_atom(Code, End, _, 0, Rest),
        skip_ws(Rest, Rest1),
        ( sub_atom(Rest1, 0, _, _, '=>') ; sub_atom(Rest1, 0, _, _, '|') )
    ->  true
    ;   sub_atom(Code, 0, Before, _, Prefix),
        atom_codes(Prefix, PCs),
        reverse(PCs, RevPCs),
        skip_ws_codes(RevPCs, [0'||_])
    ),
    !.

skip_ws(Atom, Out) :-
    atom_codes(Atom, Cs),
    skip_ws_codes(Cs, Cs1),
    atom_codes(Out, Cs1).

skip_ws_codes([C|Cs], Out) :-
    code_type(C, space), !,
    skip_ws_codes(Cs, Out).
skip_ws_codes(Cs, Cs).

:- begin_tests(wam_rust_builtin_dispatch).

test(detector_finds_known_family_patterns) :-
    % Sanity check for claims_pattern/2: names each family does handle are
    % found, and a literal used as data is not.
    family_code(ext, Ext),
    assertion(claims_pattern(Ext, 'compare/3')),
    assertion(claims_pattern(Ext, 'sort/2')),
    family_code(arith, Arith),
    assertion(claims_pattern(Arith, 'is/2')),
    assertion(claims_pattern(Arith, '==/2')),
    % "=</2" sits in the middle of an or-pattern ("... | "=</2" | ...").
    assertion(claims_pattern(Arith, '=</2')),
    family_code(meta, Meta),
    assertion(claims_pattern(Meta, 'maplist/2')),
    family_code(type, Type),
    assertion(claims_pattern(Type, 'atom/1')),
    family_code(term, Term),
    % term builds "=/2".to_string() as data; that is not a pattern.
    assertion(sub_atom(Term, _, _, _, '"=/2".to_string()')),
    assertion(\+ claims_pattern(Term, '=/2')).

test(no_family_claims_a_core_builtin) :-
    forall(( family(F), core_builtin(N) ),
           (   family_code(F, Code),
               assertion(\+ claims_pattern(Code, N))
           )).

test(core_arms_precede_the_family_cascade_in_order) :-
    wam_rust_target:compile_execute_builtin_to_rust(Code),
    forall(core_builtin(N),
           assertion(claims_pattern(Code, N))),
    findall(Pos,
            ( family(F),
              format(atom(Call), 'self.execute_~w_builtin(op, arity)', [F]),
              once(sub_atom(Code, Pos, _, _, Call))
            ),
            Positions),
    assertion(length(Positions, 6)),
    % The cascade order is unchanged: arith, io, type, term, ext, meta.
    msort(Positions, Sorted),
    assertion(Positions == Sorted),
    Positions = [FirstCall|_],
    forall(core_builtin(N),
           (   format(atom(Arm), '"~w" =>', [N]),
               once(sub_atom(Code, ArmPos, _, _, Arm)),
               assertion(ArmPos < FirstCall)
           )).

:- end_tests(wam_rust_builtin_dispatch).
