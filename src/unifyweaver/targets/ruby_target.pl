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
    ->  % Check for tail recursion pattern first
        (   can_compile_tail_recursion(Pred, Arity, Clauses)
        ->  compile_tail_recursive_rules(Pred, Arity, Clauses, Code)
        ;   can_compile_linear_recursion(Pred, Arity, Clauses)
        ->  compile_linear_recursive_rules(Pred, Arity, Clauses, Code)
        ;   compile_recursive_rules(Pred, Arity, Clauses, Code)
        )
    ;   compile_nonrecursive_rules(Pred, Arity, Clauses, Code)
    ).

%% ============================================
%% LINEAR RECURSION WITH MEMOIZATION
%% ============================================

can_compile_linear_recursion(Pred, Arity, Clauses) :-
    partition(is_recursive_clause(Pred, Arity), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [],
    forall(member(BC, BaseClauses), is_value_base_case(BC)),
    forall(member(RC, RecClauses), is_linear_recursive_clause(Pred, Arity, RC)).

is_value_base_case(Head-true) :-
    Head =.. [_|Args], Args \= [],
    last(Args, Result),
    (number(Result) ; atom(Result) ; var(Result)), !.
is_value_base_case(Head-Body) :-
    Head =.. [_|Args], Args \= [],
    body_to_goals(Body, Goals),
    forall(member(G, Goals), is_simple_condition(G)).

is_linear_recursive_clause(Pred, Arity, _Head-Body) :-
    body_to_goals(Body, Goals),
    Goals \= [],
    last(Goals, LastGoal),
    \+ (strip_module(LastGoal, _, G), functor(G, Pred, Arity)),
    member(_ is _, Goals).

compile_linear_recursive_rules(Pred, Arity, Clauses, Code) :-
    partition(is_recursive_clause(Pred, Arity), Clauses, RecClauses, BaseClauses),

    InputArity is Arity - 1,
    numlist(1, InputArity, Indices),
    maplist([I, Name]>>format(atom(Name), "arg~w", [I]), Indices, ParamNames),
    atomic_list_concat(ParamNames, ", ", ParamList),

    compile_ruby_linear_base_cases(BaseClauses, ParamNames, BaseCaseCode),

    RecClauses = [RecClause|_],
    compile_ruby_linear_recursive_case(RecClause, Pred, Arity, ParamNames, RecCaseCode),

    format(string(Code),
"def ~w(~s, &block)
  @memo ||= {}
  key = [~s]
  if @memo.key?(key)
    return block.call(@memo[key])
  end

~s
~s
end
", [Pred, ParamList, ParamList, BaseCaseCode, RecCaseCode]).

compile_ruby_linear_base_cases([], _, "").
compile_ruby_linear_base_cases([Clause|Rest], ParamNames, Code) :-
    compile_ruby_linear_base_case(Clause, ParamNames, ClauseCode),
    compile_ruby_linear_base_cases(Rest, ParamNames, RestCode),
    format(string(Code), "~s~s", [ClauseCode, RestCode]).

compile_ruby_linear_base_case(Head-_Body, ParamNames, Code) :-
    Head =.. [_|Args],
    append(InputArgs, [Result], Args),
    build_ruby_base_condition(InputArgs, ParamNames, Condition),
    (   var(Result)
    ->  (   nth1(Idx, InputArgs, RA), RA == Result
        ->  nth1(Idx, ParamNames, ResultExpr)
        ;   ResultExpr = "nil"
        )
    ;   format(string(ResultExpr), "~w", [Result])
    ),
    format(string(Code),
"  if ~s
    @memo[key] = ~s
    return block.call(~s)
  end
", [Condition, ResultExpr, ResultExpr]).

build_ruby_base_condition([], [], "true").
build_ruby_base_condition([Arg|Args], [Param|Params], Condition) :-
    build_ruby_base_condition(Args, Params, RestCond),
    (   number(Arg) -> format(string(ArgCond), "~w == ~w", [Param, Arg])
    ;   atom(Arg) -> format(string(ArgCond), "~w == '~w'", [Param, Arg])
    ;   ArgCond = "true"
    ),
    (   RestCond == "true" -> Condition = ArgCond
    ;   ArgCond == "true" -> Condition = RestCond
    ;   format(string(Condition), "~s && ~s", [ArgCond, RestCond])
    ).

compile_ruby_linear_recursive_case(Head-Body, Pred, Arity, ParamNames, Code) :-
    Head =.. [_|HeadArgs],
    append(_, [ResultVar], HeadArgs),
    body_to_goals(Body, Goals),

    partition(goal_matches(Pred, Arity), Goals, RecGoals, NonRecGoals),
    partition(is_ruby_computation, NonRecGoals, Computations, _Conditions),

    find_ruby_final_expr(Computations, ResultVar, FinalExpr),
    collect_ruby_computed_vars(Computations, ResultVar, 0, ComputedVars),
    compile_ruby_linear_pre_computations(Computations, ResultVar, HeadArgs, ParamNames, ComputedVars, PreCompCode),
    compile_ruby_nested_recursive_calls(RecGoals, Pred, HeadArgs, ParamNames, ComputedVars, FinalExpr, RecCallCode),

    format(string(Code), "~s~s", [PreCompCode, RecCallCode]).

is_ruby_computation(_ is _).

find_ruby_final_expr([], _, 0).
find_ruby_final_expr([RV is Expr|_], ResultVar, Expr) :- RV == ResultVar, !.
find_ruby_final_expr([_|Rest], ResultVar, Expr) :- find_ruby_final_expr(Rest, ResultVar, Expr).

collect_ruby_computed_vars([], _, _, []).
collect_ruby_computed_vars([Var is _|Rest], ResultVar, Counter, Vars) :-
    var(Var), Var \== ResultVar, !,
    format(atom(Name), "tmp~w", [Counter]),
    Counter1 is Counter + 1,
    collect_ruby_computed_vars(Rest, ResultVar, Counter1, RestVars),
    Vars = [Var-Name|RestVars].
collect_ruby_computed_vars([_|Rest], ResultVar, Counter, Vars) :-
    collect_ruby_computed_vars(Rest, ResultVar, Counter, Vars).

compile_ruby_linear_pre_computations([], _, _, _, _, "").
compile_ruby_linear_pre_computations([Var is Expr|Rest], ResultVar, HeadArgs, ParamNames, ComputedVars, Code) :-
    var(Var), Var \== ResultVar, !,
    get_ruby_var_name(Var, HeadArgs, ParamNames, ComputedVars, VarName),
    compile_ruby_expr(Expr, HeadArgs, ParamNames, ComputedVars, ExprCode),
    compile_ruby_linear_pre_computations(Rest, ResultVar, HeadArgs, ParamNames, ComputedVars, RestCode),
    format(string(Code), "  ~s = ~s~n~s", [VarName, ExprCode, RestCode]).
compile_ruby_linear_pre_computations([_|Rest], ResultVar, HeadArgs, ParamNames, ComputedVars, Code) :-
    compile_ruby_linear_pre_computations(Rest, ResultVar, HeadArgs, ParamNames, ComputedVars, Code).

compile_ruby_nested_recursive_calls(RecGoals, Pred, HeadArgs, ParamNames, ComputedVars, FinalExpr, Code) :-
    compile_ruby_nested_recursive_calls_(RecGoals, Pred, HeadArgs, ParamNames, ComputedVars, FinalExpr, 0, Code).

compile_ruby_nested_recursive_calls_([], _Pred, HeadArgs, ParamNames, ComputedVars, FinalExpr, _Counter, Code) :-
    compile_ruby_expr(FinalExpr, HeadArgs, ParamNames, ComputedVars, ExprCode),
    format(string(Code),
"      result = ~s
      @memo[key] = result
      block.call(result)
", [ExprCode]).

compile_ruby_nested_recursive_calls_([RecGoal|Rest], Pred, HeadArgs, ParamNames, ComputedVars, FinalExpr, Counter, Code) :-
    RecGoal =.. [_|RecArgs],
    append(RecInputs, [RecResult], RecArgs),

    maplist(compile_ruby_rec_arg(HeadArgs, ParamNames, ComputedVars), RecInputs, ArgCodes),
    atomic_list_concat(ArgCodes, ", ", ArgStr),

    format(atom(ResName), "r~w", [Counter]),
    NewComputedVars = [RecResult-ResName|ComputedVars],
    NextCounter is Counter + 1,

    compile_ruby_nested_recursive_calls_(Rest, Pred, HeadArgs, ParamNames, NewComputedVars, FinalExpr, NextCounter, InnerCode),

    format(string(Code),
"  ~w(~s) do |~s|
~s
  end
", [Pred, ArgStr, ResName, InnerCode]).

compile_ruby_rec_arg(HeadArgs, ParamNames, ComputedVars, Arg, Code) :-
    (   var(Arg) -> get_ruby_var_name(Arg, HeadArgs, ParamNames, ComputedVars, Code)
    ;   number(Arg) -> format(string(Code), "~w", [Arg])
    ;   format(string(Code), "'~w'", [Arg])
    ).

%% ============================================
%% TAIL RECURSION DETECTION AND COMPILATION
%% ============================================

%% can_compile_tail_recursion(+Pred, +Arity, +Clauses)
%% Check if this is a tail-recursive predicate that can be optimized to a loop.
can_compile_tail_recursion(Pred, Arity, Clauses) :-
    partition(is_recursive_clause(Pred, Arity), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [],
    forall(member(Clause, RecClauses), is_tail_recursive_clause(Pred, Arity, Clause)),
    forall(member(BaseClause, BaseClauses), is_accumulator_base_case(BaseClause)).

%% is_tail_recursive_clause(+Pred, +Arity, +Clause)
is_tail_recursive_clause(Pred, Arity, _Head-Body) :-
    body_to_goals(Body, Goals),
    Goals \= [],
    last(Goals, LastGoal),
    strip_module(LastGoal, _, G),
    functor(G, Pred, Arity).

%% is_accumulator_base_case(+Clause)
is_accumulator_base_case(Head-true) :-
    Head =.. [_|Args],
    length(Args, Len), Len >= 2,
    append(_, [Acc, Result], Args),
    Acc == Result, !.
is_accumulator_base_case(Head-Body) :-
    Head =.. [_|Args],
    length(Args, Len), Len >= 2,
    append(_, [Acc, Result], Args),
    Acc == Result,
    body_to_goals(Body, Goals),
    forall(member(G, Goals), is_simple_condition(G)).

is_simple_condition(G) :-
    functor(G, Op, 2),
    member(Op, [=, ==, \=, \==, <, >, =<, >=, =:=, =\=, is]).

%% compile_tail_recursive_rules(+Pred, +Arity, +Clauses, -Code)
compile_tail_recursive_rules(Pred, Arity, Clauses, Code) :-
    partition(is_recursive_clause(Pred, Arity), Clauses, RecClauses, BaseClauses),

    % Generate parameter names
    numlist(1, Arity, Indices),
    maplist([I, Name]>>format(atom(Name), "arg~w", [I]), Indices, ParamNames),
    atomic_list_concat(ParamNames, ", ", ParamList),

    % Find accumulator position
    Arity >= 2,
    AccIdx is Arity - 1,

    % Generate base case condition
    compile_tail_base_conditions(BaseClauses, AccIdx, ParamNames, BaseCondCode),

    % Generate loop body from recursive clause
    RecClauses = [RecClause|_],
    compile_tail_loop_body(RecClause, Pred, Arity, ParamNames, LoopBodyCode),

    format(string(Code),
"def ~w(~s, &block)
  # Tail recursion optimized to loop
  loop do
~s
~s
  end
end
", [Pred, ParamList, BaseCondCode, LoopBodyCode]).

%% compile_tail_base_conditions(+BaseClauses, +AccIdx, +ParamNames, -Code)
compile_tail_base_conditions(BaseClauses, AccIdx, ParamNames, Code) :-
    BaseClauses = [Head-Body|_],
    Head =.. [_|Args],
    nth1(AccIdx, ParamNames, AccParam),
    (   Body == true
    ->  Args = [FirstArg|_],
        (   FirstArg == 0
        ->  format(string(Code),
"    if arg1 == 0
      block.call(~w)
      break
    end", [AccParam])
        ;   FirstArg == []
        ->  format(string(Code),
"    if arg1.empty?
      block.call(~w)
      break
    end", [AccParam])
        ;   format(string(Code),
"    if arg1 == ~w
      block.call(~w)
      break
    end", [FirstArg, AccParam])
        )
    ;   body_to_goals(Body, Goals),
        compile_ruby_conditions(Goals, ParamNames, CondCode),
        format(string(Code),
"    if ~s
      block.call(~w)
      break
    end", [CondCode, AccParam])
    ).

compile_ruby_conditions([], _, "true").
compile_ruby_conditions([Goal|Rest], ParamNames, Code) :-
    compile_ruby_condition(Goal, ParamNames, GoalCode),
    compile_ruby_conditions(Rest, ParamNames, RestCode),
    (   RestCode == "true"
    ->  Code = GoalCode
    ;   format(string(Code), "~s && ~s", [GoalCode, RestCode])
    ).

compile_ruby_condition(_ == Y, _, Code) :- format(string(Code), "arg1 == ~w", [Y]).
compile_ruby_condition(_ =:= Y, _, Code) :- format(string(Code), "arg1 == ~w", [Y]).
compile_ruby_condition(_ < Y, _, Code) :- format(string(Code), "arg1 < ~w", [Y]).
compile_ruby_condition(_ > Y, _, Code) :- format(string(Code), "arg1 > ~w", [Y]).
compile_ruby_condition(_ =< Y, _, Code) :- format(string(Code), "arg1 <= ~w", [Y]).
compile_ruby_condition(_ >= Y, _, Code) :- format(string(Code), "arg1 >= ~w", [Y]).
compile_ruby_condition(_, _, "true").

%% compile_tail_loop_body(+RecClause, +Pred, +Arity, +ParamNames, -Code)
compile_tail_loop_body(Head-Body, Pred, Arity, ParamNames, Code) :-
    Head =.. [_|HeadArgs],
    body_to_goals(Body, Goals),
    partition(goal_matches(Pred, Arity), Goals, [RecGoal|_], OtherGoals),
    RecGoal =.. [_|RecArgs],

    % Build computed variables mapping
    collect_computed_vars(OtherGoals, 0, ComputedVars),

    % Compile intermediate computations
    compile_tail_computations(OtherGoals, HeadArgs, ParamNames, ComputedVars, CompCode),

    % Generate variable updates
    compile_tail_updates(RecArgs, HeadArgs, ParamNames, ComputedVars, UpdateCode),

    format(string(Code), "~s~s", [CompCode, UpdateCode]).

collect_computed_vars([], _, []).
collect_computed_vars([Var is _|Rest], Counter, [Var-Name|RestVars]) :-
    var(Var), !,
    format(atom(Name), "tmp~w", [Counter]),
    Counter1 is Counter + 1,
    collect_computed_vars(Rest, Counter1, RestVars).
collect_computed_vars([_|Rest], Counter, Vars) :-
    collect_computed_vars(Rest, Counter, Vars).

compile_tail_computations([], _, _, _, "").
compile_tail_computations([Goal|Rest], HeadArgs, ParamNames, ComputedVars, Code) :-
    compile_tail_computation(Goal, HeadArgs, ParamNames, ComputedVars, GoalCode),
    compile_tail_computations(Rest, HeadArgs, ParamNames, ComputedVars, RestCode),
    format(string(Code), "~s~s", [GoalCode, RestCode]).

compile_tail_computation(Var is Expr, HeadArgs, ParamNames, ComputedVars, Code) :-
    var(Var), !,
    get_ruby_var_name(Var, HeadArgs, ParamNames, ComputedVars, VarName),
    compile_ruby_expr(Expr, HeadArgs, ParamNames, ComputedVars, ExprCode),
    format(string(Code), "    ~s = ~s~n", [VarName, ExprCode]).
compile_tail_computation(_, _, _, _, "").

get_ruby_var_name(Var, _, _, ComputedVars, Name) :-
    member(V-Name, ComputedVars), V == Var, !.
get_ruby_var_name(Var, HeadArgs, ParamNames, _, Name) :-
    nth1(Idx, HeadArgs, Arg), Arg == Var, !,
    nth1(Idx, ParamNames, Name).
get_ruby_var_name(_, _, _, _, "unknown").

compile_ruby_expr(Expr, HeadArgs, ParamNames, ComputedVars, Code) :-
    var(Expr), !,
    get_ruby_var_name(Expr, HeadArgs, ParamNames, ComputedVars, Code).
compile_ruby_expr(N, _, _, _, Code) :- number(N), !, format(string(Code), "~w", [N]).
compile_ruby_expr(Atom, _, _, _, Code) :- atom(Atom), !, format(string(Code), "'~w'", [Atom]).
compile_ruby_expr(X + Y, HeadArgs, ParamNames, ComputedVars, Code) :-
    number(Y), Y < 0, !,
    NegY is -Y,
    compile_ruby_expr(X, HeadArgs, ParamNames, ComputedVars, XCode),
    format(string(Code), "(~s - ~w)", [XCode, NegY]).
compile_ruby_expr(X + Y, HeadArgs, ParamNames, ComputedVars, Code) :- !,
    compile_ruby_expr(X, HeadArgs, ParamNames, ComputedVars, XCode),
    compile_ruby_expr(Y, HeadArgs, ParamNames, ComputedVars, YCode),
    format(string(Code), "(~s + ~s)", [XCode, YCode]).
compile_ruby_expr(X - Y, HeadArgs, ParamNames, ComputedVars, Code) :- !,
    compile_ruby_expr(X, HeadArgs, ParamNames, ComputedVars, XCode),
    compile_ruby_expr(Y, HeadArgs, ParamNames, ComputedVars, YCode),
    format(string(Code), "(~s - ~s)", [XCode, YCode]).
compile_ruby_expr(X * Y, HeadArgs, ParamNames, ComputedVars, Code) :- !,
    compile_ruby_expr(X, HeadArgs, ParamNames, ComputedVars, XCode),
    compile_ruby_expr(Y, HeadArgs, ParamNames, ComputedVars, YCode),
    format(string(Code), "(~s * ~s)", [XCode, YCode]).
compile_ruby_expr(X / Y, HeadArgs, ParamNames, ComputedVars, Code) :- !,
    compile_ruby_expr(X, HeadArgs, ParamNames, ComputedVars, XCode),
    compile_ruby_expr(Y, HeadArgs, ParamNames, ComputedVars, YCode),
    format(string(Code), "(~s / ~s)", [XCode, YCode]).
compile_ruby_expr(X, _, _, _, Code) :- format(string(Code), "~w", [X]).

compile_tail_updates(RecArgs, HeadArgs, ParamNames, ComputedVars, Code) :-
    length(RecArgs, Len),
    numlist(1, Len, Indices),
    maplist(compile_ruby_single_update(HeadArgs, ParamNames, ComputedVars), Indices, RecArgs, UpdateCodes),
    atomic_list_concat(UpdateCodes, "", Code).

compile_ruby_single_update(HeadArgs, ParamNames, ComputedVars, Idx, RecArg, Code) :-
    nth1(Idx, ParamNames, ParamName),
    length(ParamNames, TotalArgs),
    (   Idx == TotalArgs
    ->  Code = ""  % Don't update result argument
    ;   var(RecArg)
    ->  (   member(V-ComputedName, ComputedVars), V == RecArg
        ->  format(string(Code), "    ~s = ~s~n", [ParamName, ComputedName])
        ;   nth1(OldIdx, HeadArgs, OldArg), OldArg == RecArg
        ->  nth1(OldIdx, ParamNames, OldName),
            (   OldName == ParamName
            ->  Code = ""
            ;   format(string(Code), "    ~s = ~s~n", [ParamName, OldName])
            )
        ;   Code = ""
        )
    ;   number(RecArg)
    ->  format(string(Code), "    ~s = ~w~n", [ParamName, RecArg])
    ;   format(string(Code), "    ~s = '~w'~n", [ParamName, RecArg])
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
