:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/recursive_compiler').
:- use_module('src/unifyweaver/core/clause_body_analysis').

%% Test predicates
:- dynamic classify_sign/2, safe_double/2, greet/2.

classify_sign(0, zero).
classify_sign(X, positive) :- X > 0.
classify_sign(X, negative) :- X < 0.

safe_double(X, Y) :- X > 0, Y is X * 2.
safe_double(X, 0) :- X =< 0.

greet(hello, 'Hello!').
greet(_, 'Hi!').

try(Label, Target, Pred/Arity) :-
    format('~w (~w): ', [Label, Target]),
    (   catch(recursive_compiler:compile_recursive(Pred/Arity, [target(Target)], Code), _, fail)
    ->  (sub_string(Code, _, _, _, "def ") ; sub_string(Code, _, _, _, "func ") ; sub_string(Code, _, _, _, "function "))
    ->  writeln(ok)
    ;   writeln('compiled but no function')
    ;   writeln('FAIL')
    ).

show(Target, Pred/Arity) :-
    (   catch(recursive_compiler:compile_recursive(Pred/Arity, [target(Target)], Code), _, fail)
    ->  atom_string(Pred, PS),
        split_string(Code, "\n", "", Lines),
        (   Target == go -> format(atom(Prefix), "func ~w", [PS])
        ;   Target == lua -> format(atom(Prefix), "function ~w", [PS])
        ;   format(atom(Prefix), "def ~w", [PS])
        ),
        (   nth1(I, Lines, L), sub_string(L, _, _, _, Prefix)
        ->  End is min(I + 12, 99999),
            forall((between(I, End, J), nth1(J, Lines, LJ),
                    (LJ \= "" ; J =:= I)), writeln(LJ))
        ;   writeln('  (function not found)')
        )
    ;   writeln('  (compile failed)')
    ).

%% Test shared compile_expression with Go hooks
test_go_expression :-
    writeln('=== TEST: Go compile_expression ==='),
    build_head_varmap([X, L], 1, VarMap),
    Goal = (X > 0 -> L = positive ; L = negative),
    (   compile_expression(go, Goal, VarMap, Code, OutputVars, _)
    ->  format('  Code: ~w~n  Outputs: ~w~n', [Code, OutputVars]),
        writeln('  PASS')
    ;   writeln('  FAIL: compile_expression failed')
    ).

%% Test shared compile_expression with Lua hooks
test_lua_expression :-
    writeln('=== TEST: Lua compile_expression ==='),
    build_head_varmap([X, L], 1, VarMap),
    Goal = (X > 0 -> L = positive ; L = negative),
    (   compile_expression(lua, Goal, VarMap, Code, OutputVars, _)
    ->  format('  Code: ~w~n  Outputs: ~w~n', [Code, OutputVars]),
        writeln('  PASS')
    ;   writeln('  FAIL: compile_expression failed')
    ).

run_tests :-
    %% Test hooks via shared compile_expression
    test_go_expression,
    test_lua_expression,
    nl,
    %% Test full compilation
    try('classify_sign', go, classify_sign/2),
    try('classify_sign', lua, classify_sign/2),
    try('safe_double', go, safe_double/2),
    try('safe_double', lua, safe_double/2),
    try('greet', go, greet/2),
    try('greet', lua, greet/2),
    nl,
    %% Show generated functions
    writeln('--- Go ---'),
    show(go, classify_sign/2), nl,
    show(go, safe_double/2), nl,
    writeln('--- Lua ---'),
    show(lua, classify_sign/2), nl,
    show(lua, safe_double/2), nl,
    writeln('=== GO + LUA HOOK TESTS DONE ===').
