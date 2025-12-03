:- module(test_common_generator, [
    test_common_generator/0
]).

test_common_generator :-
    run_tests(common_generator).

:- begin_tests(common_generator).
:- use_module('../../src/unifyweaver/targets/common_generator').

test(build_variable_map) :-
    Goal1 = parent(X, Y),
    Goal2 = ancestor(Y, Z),
    build_variable_map([Goal1-"fact", Goal2-"other"], VarMap),
    member(X-source("fact", 0), VarMap),
    member(Y-source("fact", 1), VarMap),
    member(Z-source("other", 1), VarMap),
    !.

test(translate_expr_python) :-
    Config = [
        access_fmt-"~w.get('arg~w')",
        atom_fmt-"'~w'",
        null_val-"None",
        ops-[plus-"+", mod-"%"]
    ],
    VarMap = [X-source("fact", 0)],
    translate_expr_common(X, VarMap, Config, ResVar),
    assertion(ResVar == "fact.get('arg0')"),
    translate_expr_common(123, VarMap, Config, ResNum),
    assertion(ResNum == "123"),
    translate_expr_common(foo, VarMap, Config, ResAtom),
    assertion(ResAtom == "'foo'"),
    translate_expr_common(plus(X, 10), VarMap, Config, ResCompound),
    assertion(ResCompound == "(fact.get('arg0') + 10)"),
    !.

test(translate_builtin_csharp) :-
    Config = [
        access_fmt-"~w[\"arg~w\"]",
        atom_fmt-"\"~w\"",
        null_val-"null",
        ops-[> -">", is-"=="]
    ],
    VarMap = [A-source("fact", 0)],
    translate_builtin_common(A > 10, VarMap, Config, Res),
    assertion(Res == "fact[\"arg0\"] > 10"),
    !.

test(prepare_negation) :-
    Config = [
        access_fmt-"~w.get('arg~w')",
        atom_fmt-"'~w'",
        null_val-"None"
    ],
    VarMap = [X-source("fact", 0)],
    Goal = married(X, yes),
    prepare_negation_data(Goal, VarMap, Config, Pairs),
    member(relation-Rel, Pairs),
    assertion(Rel == "'married'"),
    member(arg0-Arg0, Pairs),
    assertion(Arg0 == "fact.get('arg0')"),
    member(arg1-Arg1, Pairs),
    assertion(Arg1 == "'yes'"),
    !.

:- end_tests(common_generator).
