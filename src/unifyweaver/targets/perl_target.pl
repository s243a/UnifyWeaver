:- module(perl_target, [
    compile_predicate_to_perl/3
]).

:- use_module(library(lists)).
:- use_module('../core/binding_registry').
:- use_module('../bindings/perl_bindings').

%% compile_predicate_to_perl(+Pred/Arity, +Options, -Code)
compile_predicate_to_perl(Pred/Arity, _Options, Code) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, clause(user:Head, Body), Clauses),
    
    % Generate Perl header
    format(string(Header), "#!/usr/bin/env perl\nuse strict;\nuse warnings;\n\n", []),
    
    % Compile clauses
    compile_clauses(Pred, Arity, Clauses, ClauseCode),
    
    format(string(Code), "~s~s", [Header, ClauseCode]).

compile_clauses(Pred, Arity, Clauses, Code) :-
    % Check for facts vs rules
    (   forall(member(_-true, Clauses), true)
    ->  compile_facts(Pred, Arity, Clauses, Code)
    ;   compile_rules(Pred, Arity, Clauses, Code)
    ).

compile_facts(Pred, _Arity, Clauses, Code) :-
    format(string(Start), "sub ~w {\n    my $callback = shift;\n    my @facts = (\n", [Pred]),
    findall(FactStr, (
        member(Head-true, Clauses),
        Head =.. [_|Args],
        format_args(Args, ArgStr),
        format(string(FactStr), "        [~s]", [ArgStr])
    ), FactStrings),
    atomic_list_concat(FactStrings, ",\n", FactsBody),
    format(string(End), "\n    );\n    foreach my $fact (@facts) {\n        $callback->(@$fact);\n    }\n}\n", []),
    format(string(Code), "~s~s~s", [Start, FactsBody, End]).

compile_rules(Pred, _Arity, Clauses, Code) :-
    format(string(Start), "sub ~w {\n    my $callback = shift;\n    my ($arg1, $arg2) = @_;
", [Pred]), % Simplified args
    compile_body_clauses(Clauses, BodyCode),
    format(string(End), "}\n", []),
    format(string(Code), "~s~s~s", [Start, BodyCode, End]).

compile_body_clauses([], "").
compile_body_clauses([Head-Body|Rest], Code) :-
    Head =.. [_|Args],
    % Bind head args to perl vars
    map_head_args(Args, 0, _Bindings),
    compile_goal(Body, BodyStr),
    compile_body_clauses(Rest, RestStr),
    format(string(Code), "    # Clause\n~s~s", [BodyStr, RestStr]).

compile_goal(true, "") :- !.
compile_goal((A, B), Code) :- !,
    compile_goal(A, CodeA),
    compile_goal(B, CodeB),
    format(string(Code), "~s~s", [CodeA, CodeB]).
compile_goal(Goal, Code) :-
    Goal =.. [Pred|_Args],
    % Simplified: assuming all are local calls or built-ins
    format(string(Code), "    ~w(sub { ... }, ...);\n", [Pred]). % Placeholder

format_args([], "").
format_args([Arg|Rest], Str) :-
    format_arg(Arg, A),
    format_args(Rest, R),
    (   R == "" -> Str = A
    ;   format(string(Str), "~s, ~s", [A, R])
    ).

format_arg(A, S) :-
    number(A), format(string(S), "~w", [A]).
format_arg(A, S) :-
    atom(A), format(string(S), "'~w'", [A]).

map_head_args([], _, []).
map_head_args([_|Rest], Idx, [_|RestBindings]) :-
    Next is Idx + 1,
    map_head_args(Rest, Next, RestBindings).