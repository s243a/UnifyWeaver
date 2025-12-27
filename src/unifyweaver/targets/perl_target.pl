:- module(perl_target, [
    compile_predicate_to_perl/3
]).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module('../core/binding_registry').
:- use_module('../bindings/perl_bindings').

%% compile_predicate_to_perl(+Pred/Arity, +Options, -Code)
compile_predicate_to_perl(Pred/Arity, _Options, Code) :-
    functor(Head, Pred, Arity),
    findall(HeadCopy-BodyCopy,
            ( clause(user:Head, Body),
              copy_term((Head, Body), (HeadCopy, BodyCopy))
            ),
            Clauses),
    
    % Generate Perl header
    format(string(Header), "#!/usr/bin/env perl\nuse strict;\nuse warnings;\n\n", []),
    
    % Compile clauses
    compile_clauses(Pred, Arity, Clauses, ClauseCode),
    
    format(string(Code), "~s~s", [Header, ClauseCode]).

compile_clauses(Pred, Arity, Clauses, Code) :-
    % Check for facts vs rules
    (   forall(member(_-Body, Clauses), Body == true)
    ->  compile_facts(Pred, Arity, Clauses, Code)
    ;   compile_rules(Pred, Arity, Clauses, Code)
    ).

compile_facts(Pred, _Arity, Clauses, Code) :-
    format(string(Start), "sub ~w {\n    my $callback = shift;\n    my @facts = (\n", [Pred]),
    findall(FactStr, (
        member(Head-true, Clauses),
        Head =.. [_|Args],
        format_fact_args(Args, ArgStr),
        format(string(FactStr), "        [~s]", [ArgStr])
    ), FactStrings),
    atomic_list_concat(FactStrings, ",\n", FactsBody),
    format(string(End), "\n    );\n    foreach my $fact (@facts) {\n        $callback->(@$fact);\n    }\n}\n", []),
    format(string(Code), "~s~s~s", [Start, FactsBody, End]).

compile_rules(Pred, Arity, Clauses, Code) :-
    generate_arg_list(Arity, ArgList),
    format(string(Start), "sub ~w {\n    my $callback = shift;\n    my (~s) = @_ ;\n", [Pred, ArgList]),
    maplist(compile_rule_clause(Arity), Clauses, ClauseCodes),
    atomic_list_concat(ClauseCodes, "\n", BodyCode),
    format(string(End), "}\n", []),
    format(string(Code), "~s~s~s", [Start, BodyCode, End]).

generate_arg_list(0, "") :- !.
generate_arg_list(N, Str) :-
    numlist(1, N, Indices),
    maplist(format_arg_var, Indices, Vars),
    atomic_list_concat(Vars, ", ", Str).

format_arg_var(N, V) :- format(string(V), "$arg~w", [N]).

compile_rule_clause(_Arity, Head-Body, Code) :-
    Head =.. [_|HeadArgs],
    
    % Pre-seed map for head args to use $arg1..$argN
    map_head_to_args(HeadArgs, 1, HeadMap),
    
    % Map remaining variables
    term_variables((Head, Body), AllVars),
    exclude(is_in_map(HeadMap), AllVars, BodyVars),
    map_vars(BodyVars, 0, BodyMap),
    append(HeadMap, BodyMap, VarMap),
    
    goals_to_list(Body, Goals),
    
    % DEBUG
    % format(user_error, "HeadMap: ~w\nBodyMap: ~w\nGoals: ~w\n", [HeadMap, BodyMap, Goals]),
    
    compile_chain(Goals, HeadArgs, VarMap, 1, ChainCode),
    Code = ChainCode.

is_in_map(Map, Var) :- member(V-_, Map), V == Var.

map_head_to_args([], _, []).
map_head_to_args([Arg|Rest], Idx, [Arg-Name|Map]) :-
    var(Arg),
    !,
    format(string(Name), "$arg~w", [Idx]),
    Next is Idx + 1,
    map_head_to_args(Rest, Next, Map).
map_head_to_args([_|Rest], Idx, Map) :-
    % Skip constants in head for now (requires filtering logic in generated code)
    Next is Idx + 1,
    map_head_to_args(Rest, Next, Map).

compile_chain([], HeadArgs, VarMap, Indent, Code) :-
    map_args_to_perl(HeadArgs, VarMap, ArgStr),
    indent(Indent, I),
    format(string(Code), "~s$callback->(~s);\n", [I, ArgStr]).

compile_chain([Goal|Rest], HeadArgs, VarMap, Indent, Code) :-
    Goal =.. [Pred|Args],
    map_args_to_perl(Args, VarMap, ParamList),
    indent(Indent, I),
    NextIndent is Indent + 1,
    compile_chain(Rest, HeadArgs, VarMap, NextIndent, InnerCode),
    format(string(Code), 
"~s~w(sub {\n~s    my (~s) = @_;\n~s~s});\n", 
    [I, Pred, I, ParamList, InnerCode, I]).

goals_to_list(true, []) :- !.
goals_to_list((A, B), [GoalA|Rest]) :- !,
    strip_module(A, _, GoalA),
    goals_to_list(B, Rest).
goals_to_list(Goal0, [Goal]) :-
    strip_module(Goal0, _, Goal).

map_vars([], _, []).
map_vars([V|Rest], Idx, [V-Name|Map]) :-
    format(string(Name), "$v~w", [Idx]),
    Next is Idx + 1,
    map_vars(Rest, Next, Map).

map_args_to_perl([], _, "").
map_args_to_perl([Arg|Rest], VarMap, Str) :-
    arg_to_perl(Arg, VarMap, A),
    map_args_to_perl(Rest, VarMap, R),
    (   R == "" -> Str = A
    ;   format(string(Str), "~s, ~s", [A, R])
    ).

arg_to_perl(Arg, VarMap, Name) :-
    var(Arg), member(V-Name, VarMap), V == Arg, !.
arg_to_perl(Arg, _, Str) :-
    format_fact_arg(Arg, Str).

format_fact_args([], "").
format_fact_args([Arg|Rest], Str) :-
    format_fact_arg(Arg, A),
    format_fact_args(Rest, R),
    (   R == "" -> Str = A
    ;   format(string(Str), "~s, ~s", [A, R])
    ).

format_fact_arg(A, S) :- number(A), format(string(S), "~w", [A]).
format_fact_arg(A, S) :- atom(A), format(string(S), "'~w'", [A]).
format_fact_arg(A, S) :- string(A), format(string(S), "'~w'", [A]).

indent(0, "").
indent(N, Str) :-
    N > 0, N1 is N - 1, indent(N1, S), string_concat("    ", S, Str).
