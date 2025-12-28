:- module(perl_target, [
    compile_predicate_to_perl/3
]).

:- use_module(library(lists)).
:- use_module(library(apply)).

%% compile_predicate_to_perl(+Pred/Arity, +Options, -Code)
%%
%% Compiles a Prolog predicate to Perl code using continuation-passing style.
%% Handles facts, rules with joins, and recursive predicates with memoization.
compile_predicate_to_perl(Pred/Arity, Options, Code) :-
    functor(Head, Pred, Arity),
    findall(HeadCopy-BodyCopy,
            ( clause(user:Head, Body),
              copy_term((Head, Body), (HeadCopy, BodyCopy))
            ),
            Clauses),

    % Check if predicate is recursive
    (   is_recursive(Pred, Arity, Clauses)
    ->  IsRecursive = true
    ;   IsRecursive = false
    ),

    % Generate Perl header
    format(string(Header), "#!/usr/bin/env perl~nuse strict;~nuse warnings;~n~n", []),

    % Compile clauses
    compile_clauses(Pred, Arity, Clauses, IsRecursive, Options, ClauseCode),

    format(string(Code), "~s~s", [Header, ClauseCode]).

%% is_recursive(+Pred, +Arity, +Clauses)
%% Check if any clause body contains a call to Pred/Arity
is_recursive(Pred, Arity, Clauses) :-
    member(_-Body, Clauses),
    body_contains_call(Body, Pred, Arity),
    !.

body_contains_call(Goal, Pred, Arity) :-
    Goal = (A, B), !,
    (body_contains_call(A, Pred, Arity) ; body_contains_call(B, Pred, Arity)).
body_contains_call(Goal, Pred, Arity) :-
    Goal = (A ; B), !,
    (body_contains_call(A, Pred, Arity) ; body_contains_call(B, Pred, Arity)).
body_contains_call(Goal, Pred, Arity) :-
    strip_module(Goal, _, G),
    functor(G, Pred, Arity).

compile_clauses(Pred, Arity, Clauses, IsRecursive, _Options, Code) :-
    % Check for facts vs rules
    (   forall(member(_-Body, Clauses), Body == true)
    ->  compile_facts(Pred, Arity, Clauses, Code)
    ;   compile_rules(Pred, Arity, Clauses, IsRecursive, Code)
    ).

%% ============================================
%% FACTS COMPILATION
%% ============================================

compile_facts(Pred, _Arity, Clauses, Code) :-
    format(string(Start), "sub ~w {~n    my $callback = shift;~n    my @facts = (~n", [Pred]),
    findall(FactStr, (
        member(Head-true, Clauses),
        Head =.. [_|Args],
        format_fact_args(Args, ArgStr),
        format(string(FactStr), "        [~s]", [ArgStr])
    ), FactStrings),
    atomic_list_concat(FactStrings, ",\n", FactsBody),
    format(string(End), "~n    );~n    foreach my $fact (@facts) {~n        $callback->(@$fact);~n    }~n}~n", []),
    format(string(Code), "~s~s~s", [Start, FactsBody, End]).

%% ============================================
%% RULES COMPILATION
%% ============================================

compile_rules(Pred, Arity, Clauses, IsRecursive, Code) :-
    (   IsRecursive == true
    ->  compile_recursive_rules(Pred, Arity, Clauses, Code)
    ;   compile_nonrecursive_rules(Pred, Arity, Clauses, Code)
    ).

%% Non-recursive rules use simple CPS
compile_nonrecursive_rules(Pred, Arity, Clauses, Code) :-
    format(string(Start), "sub ~w {~n    my $callback = shift;~n", [Pred]),
    compile_all_clauses(Pred, Arity, Clauses, 0, ClauseCodes),
    atomic_list_concat(ClauseCodes, "\n", BodyCode),
    format(string(End), "}~n", []),
    format(string(Code), "~s~s~s", [Start, BodyCode, End]).

%% Recursive rules use semi-naive iteration
compile_recursive_rules(Pred, Arity, Clauses, Code) :-
    % Separate base cases (non-recursive) from recursive cases
    partition(is_recursive_clause(Pred, Arity), Clauses, RecClauses, BaseClauses),

    % Generate semi-naive iteration code
    format(string(Header), "sub ~w {~n    my $callback = shift;~n    my @delta;~n    my %seen;~n~n", [Pred]),

    % Generate base case collection
    format(string(BaseComment), "    # Base cases~n", []),
    compile_base_clauses_for_delta(Pred, Arity, BaseClauses, 0, BaseCode, Counter1),

    % Generate recursive expansion
    format(string(RecComment), "~n    # Semi-naive iteration~n    while (@delta) {~n        my $item = shift @delta;~n        $callback->(@$item);~n~n", []),
    compile_recursive_expansion(Pred, Arity, RecClauses, Counter1, RecCode),
    format(string(RecEnd), "    }~n}~n", []),

    format(string(Code), "~s~s~s~s~s~s", [Header, BaseComment, BaseCode, RecComment, RecCode, RecEnd]).

is_recursive_clause(Pred, Arity, _Head-Body) :-
    body_contains_call(Body, Pred, Arity).

%% Compile base clauses to add to @delta
compile_base_clauses_for_delta(_, _, [], Counter, "", Counter).
compile_base_clauses_for_delta(Pred, Arity, [Clause|Rest], Counter, Code, FinalCounter) :-
    compile_base_clause_for_delta(Pred, Arity, Clause, Counter, ClauseCode, Counter1),
    compile_base_clauses_for_delta(Pred, Arity, Rest, Counter1, RestCode, FinalCounter),
    string_concat(ClauseCode, RestCode, Code).

compile_base_clause_for_delta(_Pred, _Arity, Head-Body, Counter, Code, FinalCounter) :-
    Head =.. [_|HeadArgs],
    goals_to_list(Body, Goals),
    compile_goal_chain_for_delta(Goals, HeadArgs, [], Counter, 1, Code, FinalCounter).

%% Like compile_goal_chain but pushes to @delta instead of calling callback
compile_goal_chain_for_delta([], HeadArgs, Bindings, Counter, Indent, Code, Counter) :-
    map_args_to_perl(HeadArgs, Bindings, ArgStr),
    indent(Indent, I),
    format(string(Code), "~smy $key = join('\\0', ~s);~n~sunless ($seen{$key}++) { push @delta, [~s]; }~n", [I, ArgStr, I, ArgStr]).

compile_goal_chain_for_delta([Goal|Rest], HeadArgs, Bindings, VarCounter, Indent, Code, FinalCounter) :-
    Goal =.. [GoalPred|GoalArgs],
    indent(Indent, I),
    length(GoalArgs, GoalArity),
    create_callback_params(GoalArity, VarCounter, ParamNames, Counter1),
    build_join_conditions(GoalArgs, ParamNames, Bindings, JoinConds, NewBindings),
    merge_bindings(Bindings, NewBindings, MergedBindings),
    atomic_list_concat(ParamNames, ", ", ParamList),
    NextIndent is Indent + 1,
    compile_goal_chain_for_delta(Rest, HeadArgs, MergedBindings, Counter1, NextIndent, InnerCode, FinalCounter),
    (   JoinConds == []
    ->  format(string(Code), "~s~w(sub {~n~s    my (~s) = @_;~n~s~s});~n", [I, GoalPred, I, ParamList, InnerCode, I])
    ;   format_join_conditions(JoinConds, JoinCode),
        indent(NextIndent, I2),
        format(string(Code), "~s~w(sub {~n~s    my (~s) = @_;~n~s~s~s~s});~n", [I, GoalPred, I, ParamList, I2, JoinCode, InnerCode, I])
    ).

%% Compile recursive clause expansion (runs inside while loop)
compile_recursive_expansion(_, _, [], _, "").
compile_recursive_expansion(Pred, Arity, [Clause|Rest], Counter, Code) :-
    compile_recursive_clause_expansion(Pred, Arity, Clause, Counter, ClauseCode, Counter1),
    compile_recursive_expansion(Pred, Arity, Rest, Counter1, RestCode),
    string_concat(ClauseCode, RestCode, Code).

compile_recursive_clause_expansion(Pred, Arity, Head-Body, Counter, Code, FinalCounter) :-
    Head =.. [_|HeadArgs],
    goals_to_list(Body, Goals),

    % Find which goal is the recursive call and separate
    partition(is_recursive_goal(Pred, Arity), Goals, [RecGoal|_], NonRecGoals),

    % The recursive call binds from $item, non-recursive goals filter/join
    RecGoal =.. [_|RecArgs],

    % Build initial bindings from $item for the recursive call's output
    build_item_bindings(RecArgs, HeadArgs, 0, ItemBindings, Counter, Counter1),

    % Compile non-recursive goals with join conditions
    compile_expansion_chain(NonRecGoals, HeadArgs, ItemBindings, Counter1, 2, Code, FinalCounter).

is_recursive_goal(Pred, Arity, Goal) :-
    strip_module(Goal, _, G),
    functor(G, Pred, Arity).

%% Build bindings from $item->[i] for recursive call outputs
build_item_bindings([], _, _, [], Counter, Counter).
build_item_bindings([Arg|Rest], HeadArgs, Idx, Bindings, Counter, FinalCounter) :-
    Idx1 is Idx + 1,
    build_item_bindings(Rest, HeadArgs, Idx1, RestBindings, Counter, FinalCounter),
    (   var(Arg)
    ->  format(string(Name), "$item->[~w]", [Idx]),
        Bindings = [Arg-Name|RestBindings]
    ;   Bindings = RestBindings
    ).

%% Compile expansion chain - non-recursive goals that join with $item
compile_expansion_chain([], HeadArgs, Bindings, Counter, Indent, Code, Counter) :-
    map_args_to_perl(HeadArgs, Bindings, ArgStr),
    indent(Indent, I),
    format(string(Code), "~smy $key = join('\\0', ~s);~n~sunless ($seen{$key}++) { push @delta, [~s]; }~n", [I, ArgStr, I, ArgStr]).

compile_expansion_chain([Goal|Rest], HeadArgs, Bindings, VarCounter, Indent, Code, FinalCounter) :-
    Goal =.. [GoalPred|GoalArgs],
    indent(Indent, I),
    length(GoalArgs, GoalArity),
    create_callback_params(GoalArity, VarCounter, ParamNames, Counter1),
    build_join_conditions(GoalArgs, ParamNames, Bindings, JoinConds, NewBindings),
    merge_bindings(Bindings, NewBindings, MergedBindings),
    atomic_list_concat(ParamNames, ", ", ParamList),
    NextIndent is Indent + 1,
    compile_expansion_chain(Rest, HeadArgs, MergedBindings, Counter1, NextIndent, InnerCode, FinalCounter),
    (   JoinConds == []
    ->  format(string(Code), "~s~w(sub {~n~s    my (~s) = @_;~n~s~s});~n", [I, GoalPred, I, ParamList, InnerCode, I])
    ;   format_join_conditions(JoinConds, JoinCode),
        indent(NextIndent, I2),
        format(string(Code), "~s~w(sub {~n~s    my (~s) = @_;~n~s~s~s~s});~n", [I, GoalPred, I, ParamList, I2, JoinCode, InnerCode, I])
    ).

compile_all_clauses(_, _, [], _, []).
compile_all_clauses(Pred, Arity, [Clause|Rest], VarCounter, [Code|Codes]) :-
    compile_rule_clause(Pred, Arity, Clause, VarCounter, Code, NextCounter),
    compile_all_clauses(Pred, Arity, Rest, NextCounter, Codes).

compile_rule_clause(_Pred, _Arity, Head-Body, VarCounter, Code, FinalCounter) :-
    Head =.. [_|HeadArgs],

    % Get all goals from the body
    goals_to_list(Body, Goals),

    % Start with empty bindings - variables get bound as we process goals
    % Head variables will be bound when they first appear in a goal
    compile_goal_chain(Goals, HeadArgs, [], VarCounter, 1, ChainCode, FinalCounter),

    Code = ChainCode.

%% create_head_bindings(+Args, +Counter, -Bindings, -NextCounter)
%% Create fresh variable names for head arguments
create_head_bindings([], Counter, [], Counter).
create_head_bindings([Arg|Rest], Counter, Bindings, FinalCounter) :-
    (   var(Arg)
    ->  format(string(VarName), "$v~w", [Counter]),
        Counter1 is Counter + 1,
        Bindings = [Arg-VarName|RestBindings]
    ;   % Constants don't need bindings
        Counter1 = Counter,
        Bindings = RestBindings
    ),
    create_head_bindings(Rest, Counter1, RestBindings, FinalCounter).

%% compile_goal_chain(+Goals, +HeadArgs, +Bindings, +VarCounter, +Indent, -Code, -FinalCounter)
compile_goal_chain([], HeadArgs, Bindings, Counter, Indent, Code, Counter) :-
    % Base case: emit callback with head arguments
    map_args_to_perl(HeadArgs, Bindings, ArgStr),
    indent(Indent, I),
    format(string(Code), "~s$callback->(~s);~n", [I, ArgStr]).

compile_goal_chain([Goal|Rest], HeadArgs, Bindings, VarCounter, Indent, Code, FinalCounter) :-
    Goal =.. [GoalPred|GoalArgs],
    indent(Indent, I),

    % Create fresh variable names for this callback's parameters
    length(GoalArgs, GoalArity),
    create_callback_params(GoalArity, VarCounter, ParamNames, Counter1),

    % Determine which parameters need join conditions
    build_join_conditions(GoalArgs, ParamNames, Bindings, JoinConds, NewBindings),

    % Merge new bindings
    merge_bindings(Bindings, NewBindings, MergedBindings),

    % Format parameter list for callback
    atomic_list_concat(ParamNames, ", ", ParamList),

    % Compile the rest of the chain
    NextIndent is Indent + 1,
    compile_goal_chain(Rest, HeadArgs, MergedBindings, Counter1, NextIndent, InnerCode, FinalCounter),

    % Generate the goal call with callback
    (   JoinConds == []
    ->  % No join conditions needed
        format(string(Code),
            "~s~w(sub {~n~s    my (~s) = @_;~n~s~s});~n",
            [I, GoalPred, I, ParamList, InnerCode, I])
    ;   % Add join condition guards
        format_join_conditions(JoinConds, JoinCode),
        indent(NextIndent, I2),
        format(string(Code),
            "~s~w(sub {~n~s    my (~s) = @_;~n~s~s~s~s});~n",
            [I, GoalPred, I, ParamList, I2, JoinCode, InnerCode, I])
    ).

%% create_callback_params(+Arity, +Counter, -ParamNames, -NextCounter)
create_callback_params(0, Counter, [], Counter) :- !.
create_callback_params(N, Counter, [Name|Rest], FinalCounter) :-
    format(string(Name), "$p~w", [Counter]),
    Counter1 is Counter + 1,
    N1 is N - 1,
    create_callback_params(N1, Counter1, Rest, FinalCounter).

%% build_join_conditions(+GoalArgs, +ParamNames, +Bindings, -JoinConds, -NewBindings)
%% For each goal argument:
%% - If it's a variable already bound, add a join condition
%% - If it's a new variable, add it to bindings
%% - If it's a constant, add an equality check
build_join_conditions([], [], _, [], []).
build_join_conditions([Arg|Args], [Param|Params], Bindings, JoinConds, NewBindings) :-
    build_join_conditions(Args, Params, Bindings, RestConds, RestNewBindings),
    (   var(Arg),
        lookup_binding(Arg, Bindings, ExistingName)
    ->  % Variable already bound - need join condition
        JoinConds = [Param-ExistingName|RestConds],
        NewBindings = RestNewBindings
    ;   var(Arg)
    ->  % New variable - add to bindings
        JoinConds = RestConds,
        NewBindings = [Arg-Param|RestNewBindings]
    ;   % Constant - need equality check
        format_fact_arg(Arg, ConstStr),
        JoinConds = [Param-ConstStr|RestConds],
        NewBindings = RestNewBindings
    ).

lookup_binding(Var, [V-Name|_], Name) :- V == Var, !.
lookup_binding(Var, [_|Rest], Name) :- lookup_binding(Var, Rest, Name).

merge_bindings(Old, [], Old).
merge_bindings(Old, [V-N|Rest], Merged) :-
    (   lookup_binding(V, Old, _)
    ->  merge_bindings(Old, Rest, Merged)  % Already exists
    ;   merge_bindings([V-N|Old], Rest, Merged)
    ).

format_join_conditions([], "").
format_join_conditions([Param-Expected|Rest], Code) :-
    format_join_conditions(Rest, RestCode),
    % Use 'eq' for string comparison
    format(string(Cond), "return unless ~s eq ~s;~n", [Param, Expected]),
    string_concat(Cond, RestCode, Code).

%% ============================================
%% UTILITY PREDICATES
%% ============================================

goals_to_list(true, []) :- !.
goals_to_list((A, B), [GoalA|Rest]) :- !,
    strip_module(A, _, GoalA),
    goals_to_list(B, Rest).
goals_to_list(Goal0, [Goal]) :-
    strip_module(Goal0, _, Goal).

map_args_to_perl([], _, "").
map_args_to_perl([Arg|Rest], Bindings, Str) :-
    arg_to_perl(Arg, Bindings, A),
    map_args_to_perl(Rest, Bindings, R),
    (   R == "" -> Str = A
    ;   format(string(Str), "~s, ~s", [A, R])
    ).

arg_to_perl(Arg, Bindings, Name) :-
    var(Arg), lookup_binding(Arg, Bindings, Name), !.
arg_to_perl(Arg, _, Str) :-
    format_fact_arg(Arg, Str).

format_fact_args([], "").
format_fact_args([Arg|Rest], Str) :-
    format_fact_arg(Arg, A),
    format_fact_args(Rest, R),
    (   R == "" -> Str = A
    ;   format(string(Str), "~s, ~s", [A, R])
    ).

format_fact_arg(A, S) :- number(A), !, format(string(S), "~w", [A]).
format_fact_arg(A, S) :- atom(A), !, format(string(S), "'~w'", [A]).
format_fact_arg(A, S) :- string(A), !, format(string(S), "'~w'", [A]).
format_fact_arg(A, S) :- format(string(S), "'~w'", [A]).

indent(0, "") :- !.
indent(N, Str) :-
    N > 0, N1 is N - 1, indent(N1, S), string_concat("    ", S, Str).

%% ============================================
%% TEST SUPPORT
%% ============================================

:- if(current_prolog_flag(verbose, true)).
test_perl_target :-
    format("~n=== Perl Target Tests ===~n"),

    % Test 1: Facts
    retractall(user:parent(_,_)),
    assertz(user:parent(alice, bob)),
    assertz(user:parent(bob, charlie)),
    compile_predicate_to_perl(parent/2, [], Code1),
    (   sub_string(Code1, _, _, _, "@facts")
    ->  format("  ✓ Facts compilation passed~n")
    ;   format("  ✗ Facts compilation failed~n")
    ),

    % Test 2: Simple rule with join
    retractall(user:grandparent(_,_)),
    assertz((user:grandparent(X, Z) :- user:parent(X, Y), user:parent(Y, Z))),
    compile_predicate_to_perl(grandparent/2, [], Code2),
    (   sub_string(Code2, _, _, _, "return unless")
    ->  format("  ✓ Join conditions passed~n")
    ;   format("  ✗ Join conditions failed~n")
    ),

    % Test 3: Recursive predicate has memoization
    retractall(user:ancestor(_,_)),
    assertz((user:ancestor(X, Y) :- user:parent(X, Y))),
    assertz((user:ancestor(X, Z) :- user:parent(X, Y), user:ancestor(Y, Z))),
    compile_predicate_to_perl(ancestor/2, [], Code3),
    (   sub_string(Code3, _, _, _, "_seen")
    ->  format("  ✓ Memoization passed~n")
    ;   format("  ✗ Memoization failed~n")
    ).
:- endif.
