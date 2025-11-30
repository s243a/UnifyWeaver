:- module(common_generator, [
    build_variable_map/2,
    translate_expr_common/4,
    translate_builtin_common/4,
    prepare_negation_data/4
]).

%% build_variable_map(+GoalSourcePairs, -VarMap)
%  GoalSourcePairs is list of Goal-SourceString
%  VarMap is list of Var-source(SourceString, ArgIndex)
build_variable_map(GoalSourcePairs, VarMap) :-
    findall(Var-source(Source, Idx),
        (   member(Goal-Source, GoalSourcePairs),
            Goal =.. [_ | Args],
            nth0(Idx, Args, Var),
            var(Var)
        ),
        VarMap).

%% translate_expr_common(+Expr, +VarMap, +Config, -Result)
%  Config is a list of options:
%    access_fmt: Format string for variable access, e.g. "~w.get('arg~w')"
%    atom_fmt: Format string for atoms, e.g. "'~w'"
%    null_val: String for unbound/null, e.g. "None"
%    ops: List of Op-String pairs for operators
translate_expr_common(Var, VarMap, Config, Result) :-
    var(Var), !,
    (   memberchk(Var-source(Source, Idx), VarMap)
    ->  get_option(access_fmt, Config, Fmt),
        format(string(Result), Fmt, [Source, Idx])
    ;   get_option(null_val, Config, Result)
    ).
translate_expr_common(Num, _, _, Result) :-
    number(Num), !,
    format(string(Result), "~w", [Num]).
translate_expr_common(Atom, _, Config, Result) :-
    atom(Atom), !,
    get_option(atom_fmt, Config, Fmt),
    format(string(Result), Fmt, [Atom]).
translate_expr_common(Expr, VarMap, Config, Result) :-
    compound(Expr),
    Expr =.. [Op, Left, Right],
    get_option(ops, Config, Ops),
    (   memberchk(Op-OpStr, Ops)
    ->  translate_expr_common(Left, VarMap, Config, LeftStr),
        translate_expr_common(Right, VarMap, Config, RightStr),
        format(string(Result), "(~w ~w ~w)", [LeftStr, OpStr, RightStr])
    ;   format(user_error, "Op ~w not found in ~w~n", [Op, Ops]),
        fail
    ).
translate_expr_common(_, _, Config, Result) :-
    get_option(null_val, Config, Result).

%% translate_builtin_common(+Goal, +VarMap, +Config, -Result)
%  Handles is/2, >/2, etc.
%  Config must include 'ops' mapping for comparison operators too.
translate_builtin_common(Goal, VarMap, Config, Result) :-
    Goal =.. [Op, Left, Right],
    get_option(ops, Config, Ops),
    memberchk(Op-OpStr, Ops),
    !,
    translate_expr_common(Left, VarMap, Config, LeftStr),
    translate_expr_common(Right, VarMap, Config, RightStr),
    format(string(Result), "~w ~w ~w", [LeftStr, OpStr, RightStr]).

%% prepare_negation_data(+Goal, +VarMap, +Config, -Pairs)
%  Returns list of Key-ValueString pairs for the negated goal.
%  Key is 'relation' or 'argN'.
%  ValueString is the translated expression for that argument.
prepare_negation_data(Goal, VarMap, Config, Pairs) :-
    Goal =.. [Pred | Args],
    findall(Key-ValStr,
        (   nth0(Idx, Args, Arg),
            translate_expr_common(Arg, VarMap, Config, ValStr),
            format(atom(Key), "arg~w", [Idx])
        ),
        ArgPairs),
    get_option(atom_fmt, Config, AtomFmt),
    format(string(RelVal), AtomFmt, [Pred]),
    Pairs = [relation-RelVal | ArgPairs].

%% Helper
get_option(Key, Config, Value) :-
    memberchk(Key-Value, Config).
