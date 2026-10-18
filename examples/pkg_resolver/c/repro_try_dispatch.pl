:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Minimal reproduction for C indexed-dispatch codegen: `try Label`
% (wam_target.pl format_dispatch_chain/5) for resolver.pl order_lt/2
% (two list-headed clauses). Expects compile_wam_predicate_to_c/4 to
% emit INSTR_TRY (distinct from try_me_else).
%
%   swipl -q -g main -t halt examples/pkg_resolver/c/repro_try_dispatch.pl

:- use_module('../../../src/unifyweaver/targets/wam_target',
              [compile_predicate_to_wam/3]).
:- use_module('../../../src/unifyweaver/targets/wam_c_target',
              [compile_wam_predicate_to_c/4]).

order_lt([], []) :- !, fail.
order_lt([], [C|_]) :-
    order_val(C, V),
    0 < V.
order_lt([C|_], []) :-
    order_val(C, V),
    V < 0.
order_lt([A|As], [B|Bs]) :-
    order_val(A, VA),
    order_val(B, VB),
    (   VA < VB
    ->  true
    ;   VA =:= VB,
        order_lt(As, Bs)
    ).

order_val(126, -1) :- !.
order_val(C, C) :-
    C >= 65, C =< 90, !.
order_val(C, C) :-
    C >= 97, C =< 122, !.
order_val(C, V) :-
    V is C + 256.

main :-
    compile_predicate_to_wam(user:order_lt/2, [], Wam),
    format("WAM for order_lt/2:~n~w~n", [Wam]),
    (   sub_string(Wam, _, _, _, "try L_order_lt_2_3_body")
    ->  format("FOUND indexed-dispatch: try L_order_lt_2_3_body~n", [])
    ;   format("NOTE: try L_order_lt_2_3_body not in WAM text~n", [])
    ),
    catch(
        compile_wam_predicate_to_c(user:order_lt/2, Wam, [], C),
        Error,
        (   format("C compile error: ~q~n", [Error]),
            halt(1)
        )
    ),
    (   sub_string(C, _, _, _, "INSTR_TRY")
    ->  format("C compile succeeded with INSTR_TRY~n", []),
        halt(0)
    ;   format("C compile succeeded but INSTR_TRY missing~n", []),
        halt(1)
    ).
