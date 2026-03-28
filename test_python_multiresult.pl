:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').

%% Multi-result: if-then-else binding 2+ shared vars in single clause
:- dynamic classify_and_score/3.
classify_and_score(X, Class, Score) :-
    (X > 0 -> Class = positive, Score is X * 10
    ; Class = negative, Score is X * -10).

%% Multi-result: disjunction binding 2+ vars
:- dynamic color_rgb/4.
color_rgb(red, 255, 0, 0).
color_rgb(green, 0, 255, 0).
color_rgb(blue, 0, 0, 255).

try(Label, Pred/Arity) :-
    format('~w: ', [Label]),
    (   catch(recursive_compiler:compile_recursive(Pred/Arity, [target(python)], Code), _, fail)
    ->  atom_string(Pred, PS),
        split_string(Code, "\n", "", Lines),
        format(atom(Prefix), "def ~w", [PS]),
        (nth1(I, Lines, L), sub_string(L, _, _, _, Prefix) ->
            End is min(I + 12, 99999),
            forall((between(I, End, J), nth1(J, Lines, LJ),
                    (LJ \= "" ; J =:= I)), writeln(LJ))
        ; writeln('  (function not found)')
        ),
        writeln(ok)
    ;   writeln('COMPILE FAIL')
    ).

run :-
    try('multi-result ite (2 vars)', classify_and_score/3),
    try('multi-result facts (3 vars)', color_rgb/4),
    writeln('=== MULTI-RESULT TESTS DONE ===').
