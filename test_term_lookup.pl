:- encoding(utf8).
% Test term lookup in VarMap

adult(X, Age) :- Age > 18.

test :-
    Head = adult(_, _),
    findall(Head-Body, user:clause(Head, Body), [H-B]),
    H =.. [_|HeadArgs],
    write('HeadArgs: '), write(HeadArgs), nl,

    % Build VarMap
    build_var_map(HeadArgs, 1, VarMap),
    write('VarMap: '), write(VarMap), nl,

    % Extract constraint
    B = (Age > 18),
    write('Age variable: '), write(Age), nl,

    % Test lookup
    (   member((Age, Pos), VarMap) ->
        write('Found Age at position: '), write(Pos), nl
    ;   write('Age NOT found in VarMap!'), nl
    ).

build_var_map([], _, []).
build_var_map([Arg|Rest], Pos, [(Arg, Pos)|RestMap]) :-
    NextPos is Pos + 1,
    build_var_map(Rest, NextPos, RestMap).

:- initialization(test).
