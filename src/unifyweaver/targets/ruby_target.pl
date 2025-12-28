:- module(ruby_target, [
    compile_predicate_to_ruby/3
]).

:- use_module(library(lists)).
:- use_module(library(apply)).

%% compile_predicate_to_ruby(+Pred/Arity, +Options, -Code)
%%
%% Compiles a Prolog predicate to Ruby code using blocks and yield.
%% Handles facts, rules with joins, and recursive predicates with memoization.
compile_predicate_to_ruby(Pred/Arity, Options, Code) :-
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

    % Generate Ruby header
    format(string(Header), "#!/usr/bin/env ruby~nrequire 'set'~n~n", []),

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
    format(string(Start), "def ~w~n  facts = [~n", [Pred]),
    findall(FactStr, (
        member(Head-true, Clauses),
        Head =.. [_|Args],
        format_fact_args(Args, ArgStr),
        format(string(FactStr), "    [~s]", [ArgStr])
    ), FactStrings),
    atomic_list_concat(FactStrings, ",\n", FactsBody),
    format(string(End), "~n  ]~n  facts.each { |fact| yield(*fact) }~nend~n", []),
    format(string(Code), "~s~s~s", [Start, FactsBody, End]).

%% ============================================
%% RULES COMPILATION
%% ============================================

compile_rules(Pred, Arity, Clauses, IsRecursive, Code) :-
    (   IsRecursive == true
    ->  compile_recursive_rules(Pred, Arity, Clauses, Code)
    ;   compile_nonrecursive_rules(Pred, Arity, Clauses, Code)
    ).

%% Non-recursive rules use nested blocks with yield
compile_nonrecursive_rules(Pred, Arity, Clauses, Code) :-
    format(string(Start), "def ~w~n", [Pred]),
    compile_all_clauses(Pred, Arity, Clauses, 0, ClauseCodes),
    atomic_list_concat(ClauseCodes, "\n", BodyCode),
    format(string(End), "end~n", []),
    format(string(Code), "~s~s~s", [Start, BodyCode, End]).

%% Recursive rules use semi-naive iteration
compile_recursive_rules(Pred, Arity, Clauses, Code) :-
    % Separate base cases (non-recursive) from recursive cases
    partition(is_recursive_clause(Pred, Arity), Clauses, RecClauses, BaseClauses),

    % Generate semi-naive iteration code
    format(string(Header), "def ~w~n  delta = []~n  seen = Set.new~n~n", [Pred]),

    % Generate base case collection
    format(string(BaseComment), "  # Base cases~n", []),
    compile_base_cases_for_delta(BaseClauses, 0, BaseCaseCode),

    % Generate semi-naive iteration loop
    format(string(LoopStart), "~n  # Semi-naive iteration~n  until delta.empty?~n    item = delta.shift~n    yield(*item)~n~n", []),

    % Generate recursive expansion
    compile_recursive_expansion(RecClauses, Pred, Arity, RecExpansionCode),

    format(string(LoopEnd), "  end~nend~n", []),

    format(string(Code), "~s~s~s~s~s~s",
           [Header, BaseComment, BaseCaseCode, LoopStart, RecExpansionCode, LoopEnd]).

%% Check if a clause is recursive
is_recursive_clause(Pred, Arity, _Head-Body) :-
    body_contains_call(Body, Pred, Arity).

%% Compile base cases that push to delta instead of yielding
compile_base_cases_for_delta([], _, "").
compile_base_cases_for_delta([Clause|Rest], Index, Code) :-
    compile_base_case_for_delta(Clause, Index, ClauseCode),
    NextIndex is Index + 1,
    compile_base_cases_for_delta(Rest, NextIndex, RestCode),
    format(string(Code), "~s~s", [ClauseCode, RestCode]).

compile_base_case_for_delta(Head-Body, _Index, Code) :-
    Head =.. [_|HeadArgs],
    % Get goals from body
    body_to_goals(Body, Goals),
    % Start with empty bindings
    compile_goals_for_delta(Goals, HeadArgs, [], 0, "  ", Code).

%% Compile goals that push result to delta
compile_goals_for_delta([], HeadArgs, Bindings, _ParamCounter, Indent, Code) :-
    % Build the output tuple from head args
    build_output_args(HeadArgs, Bindings, OutputArgs),
    format(string(Code), "~s    key = [~s]~n~s    unless seen.include?(key)~n~s      seen.add(key)~n~s      delta << key~n~s    end~n",
           [Indent, OutputArgs, Indent, Indent, Indent, Indent]).

compile_goals_for_delta([Goal|Goals], HeadArgs, Bindings, ParamCounter, Indent, Code) :-
    Goal =.. [GoalPred|GoalArgs],
    length(GoalArgs, GoalArity),

    % Create unique parameter names for this goal
    create_callback_params(GoalArity, ParamCounter, Params, NewParamCounter),

    % Build join conditions and update bindings
    build_join_conditions(GoalArgs, Params, Bindings, JoinConds, NewBindings),

    % Format parameter list
    format_param_list(Params, ParamList),

    % Generate the block call
    format(string(BlockStart), "~s~w do |~s|~n", [Indent, GoalPred, ParamList]),

    % Generate join conditions
    format_join_conditions(JoinConds, Indent, JoinCode),

    % Recursively compile remaining goals
    atom_concat(Indent, "  ", NewIndent),
    compile_goals_for_delta(Goals, HeadArgs, NewBindings, NewParamCounter, NewIndent, InnerCode),

    format(string(BlockEnd), "~send~n", [Indent]),

    format(string(Code), "~s~s~s~s", [BlockStart, JoinCode, InnerCode, BlockEnd]).

%% Compile recursive expansion inside the loop
compile_recursive_expansion([], _, _, "").
compile_recursive_expansion([Clause|Rest], Pred, Arity, Code) :-
    compile_single_recursive_expansion(Clause, Pred, Arity, ClauseCode),
    compile_recursive_expansion(Rest, Pred, Arity, RestCode),
    format(string(Code), "~s~s", [ClauseCode, RestCode]).

compile_single_recursive_expansion(Head-Body, Pred, Arity, Code) :-
    Head =.. [_|HeadArgs],
    body_to_goals(Body, Goals),
    % Separate recursive call from other goals
    partition(goal_matches(Pred, Arity), Goals, [_RecGoal], NonRecGoals),
    % The recursive call binds to item, non-recursive goals iterate
    % Start with empty bindings
    compile_expansion_goals(NonRecGoals, HeadArgs, Pred, Arity, Body, 0, "    ", [], Code).

goal_matches(Pred, Arity, Goal) :-
    Goal =.. [Pred|Args],
    length(Args, Arity).

%% Compile expansion goals for semi-naive iteration
%% Now tracks bindings from non-recursive goals
compile_expansion_goals([], HeadArgs, Pred, Arity, Body, _ParamCounter, Indent, Bindings, Code) :-
    % Find the recursive call and determine which item position maps to which head arg
    body_to_goals(Body, AllGoals),
    include(goal_matches(Pred, Arity), AllGoals, [RecGoal]),
    RecGoal =.. [_|RecArgs],
    % Build output using item references for recursive args AND bindings from non-rec goals
    build_expansion_output(HeadArgs, RecArgs, Bindings, OutputArgs),
    format(string(Code), "~skey = [~s]~n~sunless seen.include?(key)~n~s  seen.add(key)~n~s  delta << key~n~send~n",
           [Indent, OutputArgs, Indent, Indent, Indent, Indent]).

compile_expansion_goals([Goal|Goals], HeadArgs, Pred, Arity, Body, ParamCounter, Indent, Bindings, Code) :-
    Goal =.. [GoalPred|GoalArgs],
    length(GoalArgs, GoalArity),

    % Create unique parameter names
    create_callback_params(GoalArity, ParamCounter, Params, NewParamCounter),

    % Build join conditions - need to check against item array
    build_expansion_join_conditions(GoalArgs, Params, Pred, Arity, Body, JoinConds),

    % Update bindings with new variables from this goal
    update_expansion_bindings(GoalArgs, Params, Bindings, NewBindings),

    % Format parameter list
    format_param_list(Params, ParamList),

    format(string(BlockStart), "~s~w do |~s|~n", [Indent, GoalPred, ParamList]),
    format_join_conditions(JoinConds, Indent, JoinCode),

    atom_concat(Indent, "  ", NewIndent),
    compile_expansion_goals(Goals, HeadArgs, Pred, Arity, Body, NewParamCounter, NewIndent, NewBindings, InnerCode),

    format(string(BlockEnd), "~send~n", [Indent]),

    format(string(Code), "~s~s~s~s", [BlockStart, JoinCode, InnerCode, BlockEnd]).

%% Build join conditions for expansion, checking against item array
build_expansion_join_conditions([], [], _, _, _, []).
build_expansion_join_conditions([Arg|Args], [Param|Params], Pred, Arity, Body, JoinConds) :-
    build_expansion_join_conditions(Args, Params, Pred, Arity, Body, RestConds),
    (   var(Arg)
    ->  % Check if this variable appears in the recursive call
        body_to_goals(Body, AllGoals),
        include(goal_matches(Pred, Arity), AllGoals, [RecGoal]),
        RecGoal =.. [_|RecArgs],
        (   nth0(ItemIdx, RecArgs, Arg2), Arg2 == Arg
        ->  format(string(CondStr), "item[~d]", [ItemIdx]),
            JoinConds = [Param-CondStr|RestConds]
        ;   JoinConds = RestConds
        )
    ;   % Constant
        format_fact_arg(Arg, ConstStr),
        JoinConds = [Param-ConstStr|RestConds]
    ).

%% Update bindings from goal arguments
update_expansion_bindings([], [], Bindings, Bindings).
update_expansion_bindings([Arg|Args], [Param|Params], Bindings, NewBindings) :-
    update_expansion_bindings(Args, Params, Bindings, RestBindings),
    (   var(Arg), \+ lookup_binding(Arg, Bindings, _)
    ->  NewBindings = [Arg-Param|RestBindings]
    ;   NewBindings = RestBindings
    ).

%% Build output for expansion using item references AND bindings from non-rec goals
build_expansion_output([], _, _, "").
build_expansion_output([Arg|Args], RecArgs, Bindings, Output) :-
    build_expansion_output(Args, RecArgs, Bindings, RestOutput),
    (   var(Arg),
        nth0(Idx, RecArgs, Arg2), Arg2 == Arg
    ->  % Variable from recursive call - use item reference
        format(string(ArgStr), "item[~d]", [Idx])
    ;   var(Arg), lookup_binding(Arg, Bindings, BoundName)
    ->  % Variable bound from non-recursive goal
        atom_string(BoundName, ArgStr)
    ;   var(Arg)
    ->  ArgStr = "nil"  % Unbound variable (shouldn't happen)
    ;   format_fact_arg(Arg, ArgStr)
    ),
    (   RestOutput == ""
    ->  Output = ArgStr
    ;   format(string(Output), "~s, ~s", [ArgStr, RestOutput])
    ).

%% ============================================
%% CLAUSE COMPILATION
%% ============================================

compile_all_clauses(_, _, [], _, []).
compile_all_clauses(Pred, Arity, [Clause|Rest], Index, [Code|Codes]) :-
    compile_single_clause(Pred, Arity, Clause, Index, Code),
    NextIndex is Index + 1,
    compile_all_clauses(Pred, Arity, Rest, NextIndex, Codes).

compile_single_clause(_Pred, _Arity, Head-Body, _Index, Code) :-
    Head =.. [_|HeadArgs],
    % Get goals from body
    body_to_goals(Body, Goals),
    % Start with empty bindings - head args are outputs, not inputs
    compile_goals(Goals, HeadArgs, [], 0, "  ", Code).

%% ============================================
%% GOAL COMPILATION
%% ============================================

compile_goals([], HeadArgs, Bindings, _ParamCounter, Indent, Code) :-
    % Final yield with head arguments
    build_output_args(HeadArgs, Bindings, OutputArgs),
    format(string(Code), "~syield(~s)~n", [Indent, OutputArgs]).

compile_goals([Goal|Goals], HeadArgs, Bindings, ParamCounter, Indent, Code) :-
    Goal =.. [GoalPred|GoalArgs],
    length(GoalArgs, GoalArity),

    % Create unique parameter names for this goal
    create_callback_params(GoalArity, ParamCounter, Params, NewParamCounter),

    % Build join conditions and update bindings
    build_join_conditions(GoalArgs, Params, Bindings, JoinConds, NewBindings),

    % Format parameter list
    format_param_list(Params, ParamList),

    % Generate the block call
    format(string(BlockStart), "~s~w do |~s|~n", [Indent, GoalPred, ParamList]),

    % Generate join conditions
    format_join_conditions(JoinConds, Indent, JoinCode),

    % Recursively compile remaining goals
    atom_concat(Indent, "  ", NewIndent),
    compile_goals(Goals, HeadArgs, NewBindings, NewParamCounter, NewIndent, InnerCode),

    format(string(BlockEnd), "~send~n", [Indent]),

    format(string(Code), "~s~s~s~s", [BlockStart, JoinCode, InnerCode, BlockEnd]).

%% ============================================
%% HELPER PREDICATES
%% ============================================

%% Convert body to list of goals
body_to_goals(true, []) :- !.
body_to_goals((A, B), Goals) :- !,
    body_to_goals(A, GoalsA),
    body_to_goals(B, GoalsB),
    append(GoalsA, GoalsB, Goals).
body_to_goals(Goal, [Goal]).

%% Create unique callback parameter names
create_callback_params(0, Counter, [], Counter) :- !.
create_callback_params(N, Counter, [Param|Params], FinalCounter) :-
    N > 0,
    format(atom(Param), "p~d", [Counter]),
    NextCounter is Counter + 1,
    N1 is N - 1,
    create_callback_params(N1, NextCounter, Params, FinalCounter).

%% Build join conditions for shared variables
build_join_conditions([], [], Bindings, [], Bindings).
build_join_conditions([Arg|Args], [Param|Params], Bindings, JoinConds, NewBindings) :-
    build_join_conditions(Args, Params, Bindings, RestConds, RestNewBindings),
    (   var(Arg), lookup_binding(Arg, Bindings, ExistingName)
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

%% Lookup a variable in bindings
lookup_binding(Var, [V-Name|_], Name) :-
    Var == V, !.
lookup_binding(Var, [_|Rest], Name) :-
    lookup_binding(Var, Rest, Name).

%% Format parameter list for block
format_param_list([], "").
format_param_list([P], P) :- !.
format_param_list([P|Ps], Result) :-
    format_param_list(Ps, Rest),
    format(atom(Result), "~w, ~s", [P, Rest]).

%% Format join conditions as next unless statements
format_join_conditions([], _, "").
format_join_conditions([Param-Value|Rest], Indent, Code) :-
    format_join_conditions(Rest, Indent, RestCode),
    format(string(CondCode), "~s  next unless ~w == ~s~n", [Indent, Param, Value]),
    format(string(Code), "~s~s", [CondCode, RestCode]).

%% Build output arguments from head args and bindings
build_output_args([], _, "").
build_output_args([Arg], Bindings, Output) :- !,
    (   var(Arg), lookup_binding(Arg, Bindings, Name)
    ->  atom_string(Name, Output)
    ;   var(Arg)
    ->  Output = "nil"
    ;   format_fact_arg(Arg, Output)
    ).
build_output_args([Arg|Args], Bindings, Output) :-
    Args \= [],
    build_output_args([Arg], Bindings, ArgStr),
    build_output_args(Args, Bindings, RestStr),
    format(string(Output), "~s, ~s", [ArgStr, RestStr]).

%% Format fact arguments
format_fact_args([], "").
format_fact_args([Arg], ArgStr) :- !,
    format_fact_arg(Arg, ArgStr).
format_fact_args([Arg|Args], Result) :-
    Args \= [],
    format_fact_arg(Arg, ArgStr),
    format_fact_args(Args, RestStr),
    format(string(Result), "~s, ~s", [ArgStr, RestStr]).

format_fact_arg(Arg, Str) :-
    atom(Arg), !,
    format(string(Str), "\"~w\"", [Arg]).
format_fact_arg(Arg, Str) :-
    number(Arg), !,
    format(string(Str), "~w", [Arg]).
format_fact_arg(Arg, Str) :-
    string(Arg), !,
    format(string(Str), "\"~s\"", [Arg]).
format_fact_arg(Arg, Str) :-
    format(string(Str), "~w", [Arg]).
