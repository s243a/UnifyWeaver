:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% bash_target.pl - Bash Target for UnifyWeaver
% Compiles Prolog predicates to Bash scripts using bindings.

:- module(bash_target, [
    compile_predicate_to_bash/3,
    init_bash_target/0
]).

:- use_module(library(lists)).
:- use_module(library(gensym)).
:- use_module('../core/binding_registry').
:- use_module('../bindings/bash_bindings').

%% init_bash_target
%  Initialize the Bash target by loading bindings.
init_bash_target :-
    init_bash_bindings.

%% compile_predicate_to_bash(+Predicate, +Options, -BashCode)
%  Compile a predicate to a Bash function.
compile_predicate_to_bash(PredIndicator, _Options, BashCode) :-
    PredIndicator = Pred/Arity,
    functor(Head, Pred, Arity),
    findall(Head-Body, clause(user:Head, Body), Clauses),
    format('DEBUG: Clauses for ~w/~w: ~w~n', [Pred, Arity, Clauses]),
    (   Clauses = [Head-Body]
    ->  compile_rule(Head, Body, BashCode)
    ;   format(string(BashCode), '# Multiple clauses not supported yet for ~w/~w', [Pred, Arity])
    ).

compile_rule(Head, Body, Code) :-
    Head =.. [Pred|Args],
    
    % Generate variable mapping for arguments
    map_args(Args, 1, VarMap, ArgInitCode),
    
    % Compile body
    compile_body(Body, VarMap, _, BodyCode),
    
    format(string(Code),
'~w() {
~s
~s
}
', [Pred, ArgInitCode, BodyCode]).

map_args([], _, [], "").
map_args([Arg|Rest], Idx, [Arg-VarName|MapRest], Code) :-
    format(atom(VarName), 'arg~w', [Idx]),
    format(string(Line), '    local ~w="$~w"~n', [VarName, Idx]),
    NextIdx is Idx + 1,
    map_args(Rest, NextIdx, MapRest, RestCode),
    string_concat(Line, RestCode, Code).

compile_body(true, V, V, "") :- !.
compile_body((A, B), V0, V2, Code) :- !,
    compile_body(A, V0, V1, C1),
    compile_body(B, V1, V2, C2),
    format(string(Code), '~s~n~s', [C1, C2]).
compile_body(Goal, V0, V1, Code) :-
    functor(Goal, Pred, Arity),
    (   binding(bash, Pred/Arity, Pattern, Inputs, Outputs, Options)
    ->  Goal =.. [_|Args],
        length(Inputs, InCount),
        length(InArgs, InCount),
        append(InArgs, OutArgs, Args),
        
        maplist(resolve_val(V0), InArgs, BashInArgs),
        format_pattern(Pattern, BashInArgs, Expr),
        
        (   Outputs = []
        ->  V1 = V0,
            format(string(Code), '    ~s || return 1', [Expr])
        ;   OutArgs = [OutVar],
            ensure_var(V0, OutVar, BashOutVar, V1),
            
            (   member(pattern(expansion), Options) ->
                format(string(Code), '    local ~w=~s', [BashOutVar, Expr])
            ;   member(pattern(arithmetic), Options) ->
                format(string(Code), '    local ~w=~s', [BashOutVar, Expr])
            ;   member(pattern(command_substitution), Options) ->
                format(string(Code), '    local ~w=$(~s)', [BashOutVar, Expr])
            ;   % Default
                format(string(Code), '    local ~w=$(~s)', [BashOutVar, Expr])
            )
        )
    ;   V1 = V0,
        format(string(Code), '    # Unknown predicate: ~w', [Goal])
    ).

resolve_val(VarMap, Var, Val) :-
    var(Var), lookup_var(Var, VarMap, Name), !,
    format(string(Val), '~w', [Name]).
resolve_val(_, Val, Val).

ensure_var(VarMap, Var, Name, VarMap) :-
    lookup_var(Var, VarMap, Name), !.
ensure_var(VarMap, Var, Name, [Var-Name|VarMap]) :-
    gensym(v, Name).

lookup_var(Var, [V-Name|_], Name) :- Var == V, !.
lookup_var(Var, [_|Rest], Name) :- lookup_var(Var, Rest, Name).

format_pattern(Pattern, Args, Cmd) :-
    format(string(Cmd), Pattern, Args).
