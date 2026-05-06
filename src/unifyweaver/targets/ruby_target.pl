:- module(ruby_target, [
    compile_predicate_to_ruby/3
]).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module('../core/clause_body_analysis').

%% compile_predicate_to_ruby(+Pred/Arity, +Options, -Code)
%%
%% Compiles a Prolog predicate to Ruby code using blocks and yield.
%% Handles facts, rules with joins, and recursive predicates with memoization.
%%
%% Options:
%%   json_output - Wrap predicate to output JSON array
%%   json_input  - Read facts from JSON input
%%   pipeline    - Generate pipeline-compatible code (read stdin, write stdout)
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

    % Generate Ruby header based on options
    (   (member(json_output, Options) ; member(json_input, Options) ; member(pipeline, Options))
    ->  format(string(Header), "#!/usr/bin/env ruby~nrequire 'set'~nrequire 'json'~n~n", [])
    ;   format(string(Header), "#!/usr/bin/env ruby~nrequire 'set'~n~n", [])
    ),

    % Compile clauses
    compile_clauses(Pred, Arity, Clauses, IsRecursive, Options, ClauseCode),

    % Add JSON wrapper if requested
    (   member(json_output, Options)
    ->  generate_ruby_json_wrapper(Pred, Arity, WrapperCode)
    ;   member(pipeline, Options)
    ->  generate_ruby_pipeline_wrapper(Pred, Arity, WrapperCode)
    ;   WrapperCode = ""
    ),

    format(string(Code), "~s~s~s", [Header, ClauseCode, WrapperCode]).

%% generate_ruby_json_wrapper(+Pred, +Arity, -Code)
generate_ruby_json_wrapper(Pred, Arity, Code) :-
    numlist(0, Arity, Indices0),
    Indices0 = [_|Indices],
    maplist([I, Name]>>format(atom(Name), "args[~w]", [I]), Indices, VarRefs),
    atomic_list_concat(VarRefs, ", ", VarList),
    format(string(Code),
"
# JSON output wrapper
def ~w_json
  results = []
  ~w { |*args| results << [~s] }
  puts results.to_json
end

# Run if executed directly
~w_json if __FILE__ == $0
", [Pred, Pred, VarList, Pred]).

%% generate_ruby_pipeline_wrapper(+Pred, +Arity, -Code)
generate_ruby_pipeline_wrapper(Pred, Arity, Code) :-
    numlist(0, Arity, Indices0),
    Indices0 = [_|Indices],
    maplist([I, Name]>>format(atom(Name), "args[~w]", [I]), Indices, VarRefs),
    atomic_list_concat(VarRefs, ", ", VarList),
    format(string(Code),
"
# Pipeline mode - outputs JSON to stdout
def run_pipeline
  results = []
  ~w { |*args| results << [~s] }
  puts results.to_json
end

run_pipeline if __FILE__ == $0
", [Pred, VarList]).

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
    % Check for facts vs rules vs aggregations
    (   forall(member(_-Body, Clauses), Body == true)
    ->  compile_facts(Pred, Arity, Clauses, Code)
    ;   is_aggregation_predicate(Clauses)
    ->  compile_ruby_aggregation(Pred, Arity, Clauses, Code)
    ;   compile_rules(Pred, Arity, Clauses, IsRecursive, Code)
    ).

%% ============================================
%% AGGREGATION COMPILATION
%% ============================================

is_aggregation_predicate(Clauses) :-
    member(_-Body, Clauses),
    body_contains_aggregate(Body), !.

body_contains_aggregate(aggregate_all(_, _, _)) :- !.
body_contains_aggregate((A, _)) :- body_contains_aggregate(A), !.
body_contains_aggregate((_, B)) :- body_contains_aggregate(B), !.

compile_ruby_aggregation(Pred, _Arity, Clauses, Code) :-
    Clauses = [Head-Body|_],
    Head =.. [_|HeadArgs],

    extract_ruby_aggregate(Body, AggType, Template, Goal, ResultVar),

    findall(Idx-Arg, (nth1(Idx, HeadArgs, Arg), Arg \== ResultVar), InputPairs),
    length(InputPairs, NumInputs),
    (   NumInputs > 0
    ->  numlist(1, NumInputs, InputIndices),
        maplist([I, Name]>>format(atom(Name), "arg~w", [I]), InputIndices, ParamNames),
        atomic_list_concat(ParamNames, ", ", ParamList),
        format(string(ParamDecl), "def ~w(~s, &block)~n", [Pred, ParamList])
    ;   format(string(ParamDecl), "def ~w(&block)~n", [Pred])
    ),

    ruby_agg_init_and_update(AggType, Template, InitCode, UpdateCode, FinalCode),

    (   Goal = _:InnerGoal
    ->  InnerGoal =.. [GoalPred|GoalArgs]
    ;   Goal =.. [GoalPred|GoalArgs]
    ),
    length(GoalArgs, GoalArity),
    numlist(1, GoalArity, GoalIndices),
    maplist([I, Name]>>format(atom(Name), "g~w", [I]), GoalIndices, GoalParamNames),
    atomic_list_concat(GoalParamNames, ", ", GoalParamList),

    extract_ruby_template_vars(Template, GoalArgs, GoalParamNames, TemplateExtract),

    format(string(Code),
"~s~s
  ~w do |~s|
~s~s
  end
~s
  block.call(agg_result)
end
", [ParamDecl, InitCode, GoalPred, GoalParamList, TemplateExtract, UpdateCode, FinalCode]).

extract_ruby_aggregate(aggregate_all(AggType, Goal, ResultVar), AggType, Template, Goal, ResultVar) :-
    (   AggType = count -> Template = 1
    ;   AggType = sum(Template) -> true
    ;   AggType = min(Template) -> true
    ;   AggType = max(Template) -> true
    ;   AggType = avg(Template) -> true
    ;   Template = AggType
    ), !.
extract_ruby_aggregate((A, _), AggType, Template, Goal, ResultVar) :-
    extract_ruby_aggregate(A, AggType, Template, Goal, ResultVar), !.
extract_ruby_aggregate((_, B), AggType, Template, Goal, ResultVar) :-
    extract_ruby_aggregate(B, AggType, Template, Goal, ResultVar).

ruby_agg_init_and_update(count, _, Init, Update, Final) :-
    Init = "  agg_result = 0\n",
    Update = "    agg_result += 1\n",
    Final = "".
ruby_agg_init_and_update(sum(_), _, Init, Update, Final) :-
    Init = "  agg_result = 0\n",
    Update = "    agg_result += tmpl_val\n",
    Final = "".
ruby_agg_init_and_update(min(_), _, Init, Update, Final) :-
    Init = "  agg_result = nil\n",
    Update = "    agg_result = tmpl_val if agg_result.nil? || tmpl_val < agg_result\n",
    Final = "".
ruby_agg_init_and_update(max(_), _, Init, Update, Final) :-
    Init = "  agg_result = nil\n",
    Update = "    agg_result = tmpl_val if agg_result.nil? || tmpl_val > agg_result\n",
    Final = "".
ruby_agg_init_and_update(avg(_), _, Init, Update, Final) :-
    Init = "  agg_sum = 0\n  agg_count = 0\n",
    Update = "    agg_sum += tmpl_val\n    agg_count += 1\n",
    Final = "  agg_result = agg_count > 0 ? agg_sum.to_f / agg_count : 0\n".

extract_ruby_template_vars(Template, GoalArgs, GoalParamNames, Code) :-
    (   var(Template)
    ->  (   nth1(Idx, GoalArgs, Arg), Arg == Template
        ->  nth1(Idx, GoalParamNames, ParamName),
            format(string(Code), "    tmpl_val = ~w~n", [ParamName])
        ;   Code = "    tmpl_val = 1\n"
        )
    ;   number(Template)
    ->  format(string(Code), "    tmpl_val = ~w~n", [Template])
    ;   compile_ruby_template_expr(Template, GoalArgs, GoalParamNames, ExprCode),
        format(string(Code), "    tmpl_val = ~s~n", [ExprCode])
    ).

compile_ruby_template_expr(Expr, GoalArgs, GoalParamNames, Code) :-
    var(Expr), !,
    (   nth1(Idx, GoalArgs, Arg), Arg == Expr
    ->  nth1(Idx, GoalParamNames, Code)
    ;   Code = "0"
    ).
compile_ruby_template_expr(N, _, _, Code) :- number(N), !, format(string(Code), "~w", [N]).
compile_ruby_template_expr(X + Y, GoalArgs, GoalParamNames, Code) :- !,
    compile_ruby_template_expr(X, GoalArgs, GoalParamNames, XCode),
    compile_ruby_template_expr(Y, GoalArgs, GoalParamNames, YCode),
    format(string(Code), "(~s + ~s)", [XCode, YCode]).
compile_ruby_template_expr(X * Y, GoalArgs, GoalParamNames, Code) :- !,
    compile_ruby_template_expr(X, GoalArgs, GoalParamNames, XCode),
    compile_ruby_template_expr(Y, GoalArgs, GoalParamNames, YCode),
    format(string(Code), "(~s * ~s)", [XCode, YCode]).
compile_ruby_template_expr(E, _, _, Code) :- format(string(Code), "~w", [E]).

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

%% Non-recursive rules — try native lowering first
compile_nonrecursive_rules(Pred, Arity, Clauses, Code) :-
    native_ruby_clause_body(Pred/Arity, Clauses, FuncBody),
    !,
    atom_string(Pred, PredStr),
    Arity1 is Arity - 1,
    build_ruby_arg_list(Arity1, ArgList),
    format(string(Code),
'# Generated by UnifyWeaver Ruby Target - Native Clause Lowering
# Predicate: ~w/~w

def ~w(~w)
~w
end

# CLI entry point
if ARGV.length > 0
  puts ~w(ARGV[0].to_i)
end
', [PredStr, Arity, PredStr, ArgList, FuncBody, PredStr]).

%% Fallback: non-recursive rules use nested blocks with yield
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

% ============================================================================
% NATIVE CLAUSE BODY LOWERING
% ============================================================================

%% build_ruby_arg_list(+N, -ArgList)
build_ruby_arg_list(0, "") :- !.
build_ruby_arg_list(N, ArgList) :-
    findall(Arg, (
        between(1, N, I),
        format(string(Arg), 'arg~w', [I])
    ), Args),
    atomic_list_concat(Args, ', ', ArgList).

%% native_ruby_clause_body(+PredSpec, +Clauses, -Code)

% Single clause
native_ruby_clause_body(PredSpec, [Head-Body], Code) :-
    native_ruby_clause(PredSpec, Head, Body, Condition, ClauseCode),
    !,
    (   Condition == "true"
    ->  format(string(Code), '  ~w', [ClauseCode])
    ;   format(string(Code),
'  if ~w
    ~w
  else
    raise "No matching clause for ~w"
  end', [Condition, ClauseCode, PredSpec])
    ).

% Multi-clause → if/elsif/else
native_ruby_clause_body(PredSpec, Clauses, Code) :-
    Clauses = [_|[_|_]],
    maplist(native_ruby_clause_pair(PredSpec), Clauses, Branches),
    Branches \= [],
    branches_to_ruby_if_chain(Branches, PredSpec, Code).

native_ruby_clause_pair(PredSpec, Head-Body, branch(Condition, ClauseCode)) :-
    native_ruby_clause(PredSpec, Head, Body, Condition, ClauseCode),
    !.

%% native_ruby_clause(+PredSpec, +Head, +Body, -Condition, -Code)
native_ruby_clause(_PredSpec, Head, Body, Condition, Code) :-
    Head =.. [_Pred|HeadArgs],
    length(HeadArgs, Arity),
    build_head_varmap(HeadArgs, 1, VarMap),
    (   Arity > 1
    ->  append(_InputHeadArgs, [OutputHeadArg], HeadArgs),
        ruby_head_conditions(HeadArgs, 1, Arity, HeadConditions)
    ;   OutputHeadArg = _,
        ruby_head_conditions(HeadArgs, 1, Arity, HeadConditions)
    ),
    normalize_goals(Body, Goals),
    (   Goals == []
    ->  ruby_resolve_value(VarMap, OutputHeadArg, Code),
        GoalConditions = []
    ;   (   Arity > 1, nonvar(OutputHeadArg)
        ->  clause_guard_output_split(Goals, VarMap, GuardGoals, OutputGoals),
            maplist(ruby_guard_condition(VarMap), GuardGoals, GoalConditions),
            (   OutputGoals == []
            ->  ruby_literal(OutputHeadArg, Code)
            ;   ruby_output_goals(OutputGoals, VarMap, Code)
            )
        ;   native_ruby_goal_sequence(Goals, VarMap, GoalConditions, Code)
        )
    ),
    append(HeadConditions, GoalConditions, AllConditions),
    combine_ruby_conditions(AllConditions, Condition).

%% ruby_head_conditions(+HeadArgs, +Index, +Arity, -Conditions)
ruby_head_conditions([], _, _, []).
ruby_head_conditions([_], _, Arity, []) :- Arity > 1, !.
ruby_head_conditions([HeadArg|Rest], Index, Arity, Conditions) :-
    (   var(HeadArg)
    ->  Conditions = RestConditions
    ;   format(string(ArgName), 'arg~w', [Index]),
        ruby_literal(HeadArg, Literal),
        format(string(Cond), '~w == ~w', [ArgName, Literal]),
        Conditions = [Cond|RestConditions]
    ),
    NextIndex is Index + 1,
    ruby_head_conditions(Rest, NextIndex, Arity, RestConditions).

%% native_ruby_goal_sequence(+Goals, +VarMap, -Conditions, -Code)
%  Uses classify_goal_sequence for advanced pattern detection.
%  Falls back to clause_guard_output_split if classification fails.
native_ruby_goal_sequence(Goals, VarMap, Conditions, Code) :-
    classify_goal_sequence(Goals, VarMap, ClassifiedGoals),
    ClassifiedGoals \= [],
    ruby_render_classified_goals(ClassifiedGoals, VarMap, Conditions, Lines),
    Lines \= [],
    atomic_list_concat(Lines, '\n', Code),
    !.
native_ruby_goal_sequence(Goals, VarMap, Conditions, Code) :-
    clause_guard_output_split(Goals, VarMap, GuardGoals, OutputGoals),
    maplist(ruby_guard_condition(VarMap), GuardGoals, Conditions),
    ruby_output_goals(OutputGoals, VarMap, Code).

%% ruby_render_classified_goals(+ClassifiedGoals, +VarMap, -Conditions, -Lines)
ruby_render_classified_goals([], _VarMap, [], []).
ruby_render_classified_goals([Classified], VarMap, Conds, Lines) :-
    !,
    ruby_render_classified_last(Classified, VarMap, Conds, Lines).
%% Guarded tail: output followed by guard(s)
ruby_render_classified_goals([output(Goal, _, _)|Rest], VarMap, [], Lines) :-
    Rest = [guard(_, _)|_],
    !,
    ruby_output_goal(Goal, VarMap, AssignLine, VarMap1),
    ruby_collect_trailing_guards(Rest, VarMap1, GuardGoals, _Remaining),
    maplist(ruby_guard_condition(VarMap1), GuardGoals, GuardConds),
    atomic_list_concat(GuardConds, ' && ', GuardExpr),
    (   goal_output_var(Goal, OutVar), lookup_var(OutVar, VarMap1, OutName)
    ->  true
    ;   OutName = "nil"
    ),
    format(string(IfLine), '  if ~w', [GuardExpr]),
    format(string(RetLine), '    return ~w', [OutName]),
    EndLine = '  end',
    Lines = [AssignLine, IfLine, RetLine, EndLine].
ruby_render_classified_goals([Classified|Rest], VarMap, Conds, Lines) :-
    ruby_render_classified_mid(Classified, VarMap, MidConds, MidLines, VarMap1),
    ruby_render_classified_goals(Rest, VarMap1, RestConds, RestLines),
    append(MidConds, RestConds, Conds),
    append(MidLines, RestLines, Lines).

%% ruby_render_classified_mid(+Classified, +VarMap, -Conds, -Lines, -VarMapOut)
ruby_render_classified_mid(guard(Goal, _), VarMap, [Cond], [], VarMap) :-
    ruby_guard_condition(VarMap, Goal, Cond).
ruby_render_classified_mid(output(Goal, _, _), VarMap0, [], [Line], VarMapOut) :-
    ruby_output_goal(Goal, VarMap0, Line, VarMapOut).
ruby_render_classified_mid(output_ite(If, Then, Else, _SharedVars), VarMap0, [], Lines, VarMap0) :-
    ruby_guard_condition(VarMap0, If, Cond),
    ruby_branch_value(Then, VarMap0, ThenExpr),
    ruby_branch_value(Else, VarMap0, ElseExpr),
    format(string(IfLine), '  if ~w', [Cond]),
    format(string(ThenLine), '    return ~w', [ThenExpr]),
    ElseLine = '  else',
    format(string(ElseRetLine), '    return ~w', [ElseExpr]),
    Lines = [IfLine, ThenLine, ElseLine, ElseRetLine, '  end'].
ruby_render_classified_mid(passthrough(Goal), VarMap0, [], [Line], VarMapOut) :-
    ruby_output_goal(Goal, VarMap0, Line, VarMapOut).
ruby_render_classified_mid(_, VarMap, [], [], VarMap).

%% ruby_render_classified_last(+Classified, +VarMap, -Conds, -Lines)
ruby_render_classified_last(guard(Goal, _), VarMap, [Cond], []) :-
    ruby_guard_condition(VarMap, Goal, Cond).
ruby_render_classified_last(output(Goal, _, _), VarMap, [], Lines) :-
    ruby_output_goal_last_lines(Goal, VarMap, Lines).
ruby_render_classified_last(output_ite(If, Then, Else, _), VarMap, [], Lines) :-
    ruby_guard_condition(VarMap, If, Cond),
    ruby_branch_value(Then, VarMap, ThenExpr),
    ruby_branch_value(Else, VarMap, ElseExpr),
    format(string(IfLine), '  if ~w', [Cond]),
    format(string(ThenLine), '    return ~w', [ThenExpr]),
    ElseLine = '  else',
    format(string(ElseRetLine), '    return ~w', [ElseExpr]),
    Lines = [IfLine, ThenLine, ElseLine, ElseRetLine, '  end'].
ruby_render_classified_last(output_disj(Alternatives, _SharedVars), VarMap, [], Lines) :-
    ruby_disj_if_chain(Alternatives, VarMap, Lines).
ruby_render_classified_last(passthrough(Goal), VarMap, [], Lines) :-
    ruby_output_goal_last_lines(Goal, VarMap, Lines).
ruby_render_classified_last(_, _, [], []).

%% ruby_output_goal_last_lines(+Goal, +VarMap, -Lines)
ruby_output_goal_last_lines(Goal, VarMap, [Line]) :-
    ruby_output_goal(Goal, VarMap, AssignLine, VarMapOut),
    (   goal_output_var(Goal, OutVar), lookup_var(OutVar, VarMapOut, OutName)
    ->  format(string(RetPart), '\n  return ~w', [OutName]),
        atom_concat(AssignLine, RetPart, Line)
    ;   Line = AssignLine
    ).
ruby_output_goal_last_lines(Goal, VarMap, [Line]) :-
    ruby_branch_value(Goal, VarMap, Expr),
    format(string(Line), '  return ~w', [Expr]).

%% ruby_collect_trailing_guards(+ClassifiedGoals, +VarMap, -GuardGoals, -Remaining)
ruby_collect_trailing_guards([guard(Goal, _)|Rest], VarMap, [Goal|Guards], Remaining) :-
    !, ruby_collect_trailing_guards(Rest, VarMap, Guards, Remaining).
ruby_collect_trailing_guards(Remaining, _, [], Remaining).

%% ruby_disj_if_chain(+Alternatives, +VarMap, -Lines)
ruby_disj_if_chain([], _, []).
ruby_disj_if_chain([Alt], VarMap, [ElseLine, RetLine, EndLine]) :-
    !,
    ruby_branch_value(Alt, VarMap, ValExpr),
    ElseLine = '  else',
    format(string(RetLine), '    return ~w', [ValExpr]),
    EndLine = '  end'.
ruby_disj_if_chain([Alt|Rest], VarMap, Lines) :-
    normalize_goals(Alt, Goals),
    clause_guard_output_split(Goals, VarMap, Guards, _Outputs),
    (   Guards \= []
    ->  maplist(ruby_guard_condition(VarMap), Guards, CondStrs),
        atomic_list_concat(CondStrs, ' && ', CondExpr)
    ;   CondExpr = "true"
    ),
    ruby_branch_value(Alt, VarMap, ValExpr),
    format(string(IfLine), '  if ~w', [CondExpr]),
    format(string(RetLine), '    return ~w', [ValExpr]),
    ruby_disj_elsif_chain(Rest, VarMap, RestLines),
    append([IfLine, RetLine], RestLines, Lines).

ruby_disj_elsif_chain([], _, []).
ruby_disj_elsif_chain([Alt], VarMap, [ElseLine, RetLine, EndLine]) :-
    !,
    ruby_branch_value(Alt, VarMap, ValExpr),
    ElseLine = '  else',
    format(string(RetLine), '    return ~w', [ValExpr]),
    EndLine = '  end'.
ruby_disj_elsif_chain([Alt|Rest], VarMap, [ElsifLine, RetLine|RestLines]) :-
    normalize_goals(Alt, Goals),
    clause_guard_output_split(Goals, VarMap, Guards, _Outputs),
    (   Guards \= []
    ->  maplist(ruby_guard_condition(VarMap), Guards, CondStrs),
        atomic_list_concat(CondStrs, ' && ', CondExpr)
    ;   CondExpr = "true"
    ),
    ruby_branch_value(Alt, VarMap, ValExpr),
    format(string(ElsifLine), '  elsif ~w', [CondExpr]),
    format(string(RetLine), '    return ~w', [ValExpr]),
    ruby_disj_elsif_chain(Rest, VarMap, RestLines).

%% ruby_guard_condition(+VarMap, +Goal, -Condition)
ruby_guard_condition(VarMap, _Module:Goal, Condition) :-
    !, ruby_guard_condition(VarMap, Goal, Condition).
ruby_guard_condition(VarMap, Goal, Condition) :-
    compound(Goal),
    Goal =.. [Op, Left, Right],
    expr_op(Op, StdOp),
    !,
    ruby_expr(Left, VarMap, RLeft),
    ruby_expr(Right, VarMap, RRight),
    ruby_op(StdOp, ROp),
    format(string(Condition), '~w ~w ~w', [RLeft, ROp, RRight]).

%% ruby_output_goals(+Goals, +VarMap, -Code)
ruby_output_goals([], _VarMap, '"error"') :- !.
ruby_output_goals([Goal], VarMap, Code) :-
    !, ruby_output_goal_last(Goal, VarMap, Code).
ruby_output_goals([Goal|Rest], VarMap0, Code) :-
    ruby_output_goal(Goal, VarMap0, _Line, VarMap1),
    ruby_output_goals(Rest, VarMap1, Code).

%% ruby_output_goal_last — produce the return expression
ruby_output_goal_last(_Module:Goal, VarMap, Code) :-
    !, ruby_output_goal_last(Goal, VarMap, Code).
ruby_output_goal_last(Goal, VarMap, Code) :-
    if_then_else_goal(Goal, IfGoal, ThenGoal, ElseGoal),
    !,
    ruby_if_then_else_output(IfGoal, ThenGoal, ElseGoal, VarMap, Code).
ruby_output_goal_last(=(Var, Expr), VarMap, Code) :-
    var(Var), !,
    ruby_expr(Expr, VarMap, Code).
ruby_output_goal_last(is(Var, Expr), VarMap, Code) :-
    var(Var), !,
    ruby_expr(Expr, VarMap, Code).

%% ruby_output_goal — produce a local assignment (not used as return)
ruby_output_goal(_Module:Goal, VarMap0, Line, VarMapOut) :-
    !, ruby_output_goal(Goal, VarMap0, Line, VarMapOut).
ruby_output_goal(=(Var, Expr), VarMap0, Line, VarMapOut) :-
    var(Var), !,
    ensure_var(VarMap0, Var, VarName, VarMapOut),
    ruby_expr(Expr, VarMap0, RExpr),
    format(string(Line), '~w = ~w', [VarName, RExpr]).
ruby_output_goal(is(Var, Expr), VarMap0, Line, VarMapOut) :-
    var(Var), !,
    ensure_var(VarMap0, Var, VarName, VarMapOut),
    ruby_expr(Expr, VarMap0, RExpr),
    format(string(Line), '~w = ~w', [VarName, RExpr]).

%% ruby_if_then_else_output — generate ternary expressions
ruby_if_then_else_output(IfGoal, ThenGoal, ElseGoal, VarMap, Code) :-
    flatten_ruby_if_branches(IfGoal, ThenGoal, ElseGoal, Branches, DefaultGoal),
    ruby_branches_to_ternary(Branches, DefaultGoal, VarMap, Code).

flatten_ruby_if_branches(If, Then, Else, [branch(If, Then)|RestBranches], Default) :-
    if_then_else_goal(Else, If2, Then2, Else2),
    !,
    flatten_ruby_if_branches(If2, Then2, Else2, RestBranches, Default).
flatten_ruby_if_branches(If, Then, Else, [branch(If, Then)], Else).

ruby_branches_to_ternary([branch(If, Then)], DefaultGoal, VarMap, Code) :-
    !,
    ruby_guard_condition(VarMap, If, IfCond),
    ruby_branch_value(Then, VarMap, ThenVal),
    ruby_branch_value(DefaultGoal, VarMap, ElseVal),
    format(string(Code), '(~w) ? ~w : ~w', [IfCond, ThenVal, ElseVal]).
ruby_branches_to_ternary([branch(If, Then)|Rest], DefaultGoal, VarMap, Code) :-
    ruby_guard_condition(VarMap, If, IfCond),
    ruby_branch_value(Then, VarMap, ThenVal),
    ruby_branches_to_ternary(Rest, DefaultGoal, VarMap, ElseCode),
    format(string(Code), '(~w) ? ~w : ~w', [IfCond, ThenVal, ElseCode]).

%% ruby_branch_value — extract result value from a branch
ruby_branch_value(_Module:Goal, VarMap, Value) :-
    !, ruby_branch_value(Goal, VarMap, Value).
ruby_branch_value(Goal, VarMap, Value) :-
    if_then_else_goal(Goal, If, Then, Else),
    !,
    ruby_guard_condition(VarMap, If, Cond),
    ruby_branch_value(Then, VarMap, ThenVal),
    ruby_branch_value(Else, VarMap, ElseVal),
    format(string(Value), '(~w) ? ~w : ~w', [Cond, ThenVal, ElseVal]).
ruby_branch_value((A, B), VarMap, Value) :-
    !,
    normalize_goals((A, B), Goals),
    last(Goals, LastGoal),
    ruby_branch_value(LastGoal, VarMap, Value).
ruby_branch_value(=(_, Expr), VarMap, Value) :-
    !, ruby_expr(Expr, VarMap, Value).
ruby_branch_value(is(_, Expr), VarMap, Value) :-
    !, ruby_expr(Expr, VarMap, Value).
ruby_branch_value(Goal, VarMap, Value) :-
    ruby_expr(Goal, VarMap, Value).

% ============================================================================
% MULTIFILE HOOKS — Register Ruby renderers for shared compile_expression
% ============================================================================

clause_body_analysis:render_output_goal(ruby, Goal, VarMap, Line, VarName, VarMapOut) :-
    ruby_output_goal(Goal, VarMap, Line, VarMapOut),
    (   goal_output_var(Goal, OutVar), lookup_var(OutVar, VarMapOut, VarName)
    ->  true
    ;   VarName = "_"
    ).

clause_body_analysis:render_guard_condition(ruby, Goal, VarMap, CondStr) :-
    ruby_guard_condition(VarMap, Goal, CondStr).

clause_body_analysis:render_branch_value(ruby, Branch, VarMap, ExprStr) :-
    ruby_branch_value(Branch, VarMap, ExprStr).

clause_body_analysis:render_ite_block(ruby, Cond, ThenLines, ElseLines, Indent, _ReturnVars, Lines) :-
    format(string(IfLine), '~wif ~w', [Indent, Cond]),
    ruby_indent_lines(ThenLines, Indent, IndentedThen),
    (   ElseLines \= []
    ->  format(string(ElseLine), '~welsif', [Indent]),
        ruby_indent_lines(ElseLines, Indent, IndentedElse),
        format(string(EndLine), '~wend', [Indent]),
        append([IfLine|IndentedThen], [ElseLine|IndentedElse], PreEnd),
        append(PreEnd, [EndLine], Lines)
    ;   format(string(EndLine), '~wend', [Indent]),
        append([IfLine|IndentedThen], [EndLine], Lines)
    ).

ruby_indent_lines([], _, []).
ruby_indent_lines([Line|Rest], Indent, [Indented|RestIndented]) :-
    format(string(Indented), '~w    ~w', [Indent, Line]),
    ruby_indent_lines(Rest, Indent, RestIndented).

%% ruby_expr — convert Prolog expression to Ruby syntax
ruby_expr(Var, VarMap, RExpr) :-
    var(Var), !,
    (   lookup_var(Var, VarMap, Name)
    ->  RExpr = Name
    ;   term_string(Var, RExpr)
    ).
ruby_expr(Expr, VarMap, RExpr) :-
    compound(Expr),
    Expr =.. [Op, Left, Right],
    expr_op(Op, StdOp),
    !,
    ruby_expr(Left, VarMap, RLeft),
    ruby_expr(Right, VarMap, RRight),
    ruby_op(StdOp, ROp),
    format(string(RExpr), '(~w ~w ~w)', [RLeft, ROp, RRight]).
ruby_expr(-Expr, VarMap, RExpr) :-
    !,
    ruby_expr(Expr, VarMap, Inner),
    format(string(RExpr), '(-~w)', [Inner]).
ruby_expr(abs(Expr), VarMap, RExpr) :-
    !,
    ruby_expr(Expr, VarMap, Inner),
    format(string(RExpr), '~w.abs', [Inner]).
ruby_expr(Atom, _VarMap, RExpr) :-
    atom(Atom), !,
    ruby_literal(Atom, RExpr).
ruby_expr(Number, _VarMap, RExpr) :-
    number(Number), !,
    format(string(RExpr), '~w', [Number]).
ruby_expr(String, _VarMap, RExpr) :-
    string(String), !,
    format(string(RExpr), '"~w"', [String]).

%% ruby_literal — convert Prolog value to Ruby literal
ruby_literal(Value, 'nil') :- var(Value), !.
ruby_literal(true, 'true') :- !.
ruby_literal(false, 'false') :- !.
ruby_literal(Value, RubyLiteral) :-
    number(Value), !,
    format(string(RubyLiteral), '~w', [Value]).
ruby_literal(Value, RubyLiteral) :-
    atom(Value), !,
    format(string(RubyLiteral), '"~w"', [Value]).
ruby_literal(Value, RubyLiteral) :-
    string(Value), !,
    format(string(RubyLiteral), '"~w"', [Value]).
ruby_literal(Value, RubyLiteral) :-
    term_string(Value, S),
    format(string(RubyLiteral), '"~w"', [S]).

%% ruby_resolve_value — resolve variable or constant to Ruby expression
ruby_resolve_value(VarMap, Var, RExpr) :-
    var(Var), !,
    lookup_var(Var, VarMap, RExpr).
ruby_resolve_value(_VarMap, Value, RExpr) :-
    ruby_literal(Value, RExpr).

%% ruby_op — map standard operator to Ruby syntax
ruby_op('>', '>').
ruby_op('<', '<').
ruby_op('>=', '>=').
ruby_op('<=', '<=').
ruby_op('==', '==').
ruby_op('!=', '!=').
ruby_op('+', '+').
ruby_op('-', '-').
ruby_op('*', '*').
ruby_op('/', '/').
ruby_op('%', '%').
ruby_op('&&', '&&').
ruby_op('||', '||').

%% combine_ruby_conditions — join conditions with &&
combine_ruby_conditions([], "true") :- !.
combine_ruby_conditions([Condition], Condition) :- !.
combine_ruby_conditions(Conditions, Combined) :-
    atomic_list_concat(Conditions, ' && ', Combined).

%% branches_to_ruby_if_chain — build Ruby if/elsif/else/end chain
branches_to_ruby_if_chain(Branches, PredSpec, Code) :-
    branches_to_ruby_if_lines(Branches, PredSpec, Lines),
    atomic_list_concat(Lines, '\n', Code).

branches_to_ruby_if_lines([branch(Condition, ClauseCode)], PredSpec, [IfLine, RetLine, ElseLine, ErrLine, EndLine]) :-
    !,
    format(string(IfLine), '  if ~w', [Condition]),
    format(string(RetLine), '    ~w', [ClauseCode]),
    ElseLine = '  else',
    format(string(ErrLine), '    raise "No matching clause for ~w"', [PredSpec]),
    EndLine = '  end'.
branches_to_ruby_if_lines([branch(Condition, ClauseCode)|Rest], PredSpec, [IfLine, RetLine|RestLines]) :-
    format(string(IfLine), '  if ~w', [Condition]),
    format(string(RetLine), '    ~w', [ClauseCode]),
    branches_to_ruby_elsif_lines(Rest, PredSpec, RestLines).

branches_to_ruby_elsif_lines([branch(Condition, ClauseCode)], PredSpec, [ElifLine, RetLine, ElseLine, ErrLine, EndLine]) :-
    !,
    format(string(ElifLine), '  elsif ~w', [Condition]),
    format(string(RetLine), '    ~w', [ClauseCode]),
    ElseLine = '  else',
    format(string(ErrLine), '    raise "No matching clause for ~w"', [PredSpec]),
    EndLine = '  end'.
branches_to_ruby_elsif_lines([branch(Condition, ClauseCode)|Rest], PredSpec, [ElifLine, RetLine|RestLines]) :-
    format(string(ElifLine), '  elsif ~w', [Condition]),
    format(string(RetLine), '    ~w', [ClauseCode]),
    branches_to_ruby_elsif_lines(Rest, PredSpec, RestLines).

% ============================================================================
% MULTIFILE DISPATCH - Tail Recursion
% ============================================================================

:- use_module('../core/advanced/tail_recursion').
:- multifile tail_recursion:compile_tail_pattern/9.

tail_recursion:compile_tail_pattern(ruby, PredStr, Arity, _BaseClauses, _RecClauses, _AccPos, StepOp, _ExitAfterResult, Code) :-
    step_op_to_ruby(StepOp, RubyStepExpr),
    (   Arity =:= 3 ->
        format(string(Code),
'# Generated by UnifyWeaver Ruby Target - Tail Recursion (multifile dispatch)
# Predicate: ~w/~w

def ~w(items, acc = 0)
  items.each do |item|
    acc = ~w
  end
  acc
end

if __FILE__ == $0
  if ARGV.length >= 1
    items = ARGV[0].split(",").map(&:to_i)
    puts ~w(items)
  end
end
', [PredStr, Arity, PredStr, RubyStepExpr, PredStr])
    ;   Arity =:= 2 ->
        format(string(Code),
'# Generated by UnifyWeaver Ruby Target - Tail Recursion (binary, multifile dispatch)
# Predicate: ~w/~w

def ~w(items)
  count = 0
  items.each { |_| count += 1 }
  count
end

if __FILE__ == $0
  if ARGV.length >= 1
    items = ARGV[0].split(",")
    puts ~w(items)
  end
end
', [PredStr, Arity, PredStr, PredStr])
    ;   fail
    ).

%% step_op_to_ruby(+StepOp, -RubyExpr)
step_op_to_ruby(arithmetic(Expr), RubyExpr) :-
    expr_to_ruby(Expr, RubyExpr).
step_op_to_ruby(unknown, 'acc + 1').

expr_to_ruby(_ + Const, RubyExpr) :- integer(Const), !,
    format(atom(RubyExpr), 'acc + ~w', [Const]).
expr_to_ruby(_ + _, 'acc + item') :- !.
expr_to_ruby(_ - _, 'acc - item') :- !.
expr_to_ruby(_ * _, 'acc * item') :- !.
expr_to_ruby(_, 'acc + 1').

% ============================================================================
% MULTIFILE DISPATCH - Linear Recursion
% ============================================================================

:- use_module('../core/advanced/linear_recursion').
:- multifile linear_recursion:compile_linear_pattern/8.

linear_recursion:compile_linear_pattern(ruby, PredStr, Arity, BaseClauses, RecClauses, MemoEnabled, _MemoStrategy, Code) :-
    (   Arity =:= 2 ->
        linear_fold_ruby(PredStr, BaseClauses, RecClauses, MemoEnabled, Code)
    ;   linear_generic_ruby(PredStr, Arity, MemoEnabled, Code)
    ).

%% linear_fold_ruby(+PredStr, +BaseClauses, +RecClauses, +MemoEnabled, -Code)
linear_fold_ruby(PredStr, BaseClauses, _RecClauses, MemoEnabled, Code) :-
    linear_recursion:extract_base_case_info(BaseClauses, BaseInput, BaseOutput),
    linear_recursion:detect_input_type(BaseInput, InputType),
    (   MemoEnabled = true ->
        format(string(MemoDecl), '@~w_memo = {}', [PredStr]),
        format(string(MemoCheck), '  return @~w_memo[n] if @~w_memo.key?(n)', [PredStr, PredStr]),
        format(string(MemoStore), '  @~w_memo[n] = result', [PredStr])
    ;   MemoDecl = '# Memoization disabled',
        MemoCheck = '',
        MemoStore = ''
    ),
    % Extract fold operation
    atom_string(Pred, PredStr),
    functor(Head, Pred, 2),
    findall(clause(Head, Body), user:clause(Head, Body), AllClauses),
    partition(linear_recursion:is_recursive_clause(Pred), AllClauses, ActualRec, _),
    (   ActualRec = [clause(RHead, RBody)|_] ->
        RHead =.. [_, InputVar, _],
        linear_recursion:find_recursive_call(RBody, RecCall),
        RecCall =.. [_, _, AccVar],
        linear_recursion:find_last_is_expression(RBody, _ is FoldExpr),
        translate_fold_expr_ruby(FoldExpr, InputVar, AccVar, RubyOp)
    ;   RubyOp = "current * acc"
    ),
    (   InputType = numeric ->
        format(string(Code),
'# Generated by UnifyWeaver Ruby Target - Linear Recursion (numeric, multifile dispatch)
# Predicate: ~w/2

~w

def ~w(n)
~w
  return ~w if n == ~w
  result = ~w.downto(1).reduce(~w) { |acc, current| ~w }
~w
  result
end

if __FILE__ == $0
  puts ~w(ARGV[0].to_i) if ARGV.length >= 1
end
', [PredStr, MemoDecl, PredStr, MemoCheck, BaseOutput, BaseInput, 'n', BaseOutput, RubyOp, MemoStore, PredStr])
    ;   InputType = list ->
        % Re-extract fold with head variable for list patterns
        (   ActualRec = [clause(LRHead, LRBody)|_] ->
            linear_recursion:find_last_is_expression(LRBody, _ is LFoldExpr),
            linear_recursion:find_recursive_call(LRBody, LRecCall),
            LRecCall =.. [_, _, LAccVar],
            LRHead =.. [_, [LHeadVar|_], _],
            translate_list_fold_ruby(LFoldExpr, LHeadVar, LAccVar, ListFoldOp)
        ;   ListFoldOp = "acc + current"
        ),
        format(string(Code),
'# Generated by UnifyWeaver Ruby Target - Linear Recursion (list, multifile dispatch)
# Predicate: ~w/2

def ~w(lst)
  return ~w if lst.empty?
  lst.reduce(~w) { |acc, current| ~w }
end

if __FILE__ == $0
  puts ~w(ARGV[0].split(",").map(&:to_i)) if ARGV.length >= 1
end
', [PredStr, PredStr, BaseOutput, BaseOutput, ListFoldOp, PredStr])
    ;   linear_generic_ruby(PredStr, 2, MemoEnabled, Code)
    ).

%% linear_generic_ruby(+PredStr, +Arity, +MemoEnabled, -Code)
linear_generic_ruby(PredStr, Arity, MemoEnabled, Code) :-
    (   MemoEnabled = true ->
        format(string(MemoDecl), '@~w_memo = {}', [PredStr]),
        format(string(MemoCheck), '  return @~w_memo[n] if @~w_memo.key?(n)', [PredStr, PredStr]),
        format(string(MemoStore), '  @~w_memo[n] = result', [PredStr])
    ;   MemoDecl = '# Memoization disabled',
        MemoCheck = '',
        MemoStore = ''
    ),
    format(string(Code),
'# Generated by UnifyWeaver Ruby Target - Linear Recursion (generic, multifile dispatch)
# Predicate: ~w/~w

~w

def ~w(n)
~w
  return 0 if n <= 0
  return 1 if n == 1
  result = ~w(n - 1) + n
~w
  result
end

if __FILE__ == $0
  puts ~w(ARGV[0].to_i) if ARGV.length >= 1
end
', [PredStr, Arity, MemoDecl, PredStr, MemoCheck, PredStr, MemoStore, PredStr]).

%% translate_fold_expr_ruby(+PrologExpr, +InputVar, +AccVar, -RubyExpr)
translate_fold_expr_ruby(A * B, InputVar, AccVar, Expr) :-
    translate_ruby_term(A, InputVar, AccVar, AT),
    translate_ruby_term(B, InputVar, AccVar, BT),
    format(string(Expr), '~w * ~w', [AT, BT]).
translate_fold_expr_ruby(A + B, InputVar, AccVar, Expr) :-
    translate_ruby_term(A, InputVar, AccVar, AT),
    translate_ruby_term(B, InputVar, AccVar, BT),
    format(string(Expr), '~w + ~w', [AT, BT]).
translate_fold_expr_ruby(A - B, InputVar, AccVar, Expr) :-
    translate_ruby_term(A, InputVar, AccVar, AT),
    translate_ruby_term(B, InputVar, AccVar, BT),
    format(string(Expr), '~w - ~w', [AT, BT]).
translate_fold_expr_ruby(Term, InputVar, AccVar, Expr) :-
    translate_ruby_term(Term, InputVar, AccVar, Expr).

translate_ruby_term(Term, InputVar, _AccVar, 'current') :- Term == InputVar, !.
translate_ruby_term(Term, _InputVar, AccVar, 'acc') :- Term == AccVar, !.
translate_ruby_term(Number, _, _, RubyTerm) :- integer(Number), !,
    format(string(RubyTerm), '~w', [Number]).
translate_ruby_term(Atom, _, _, RubyTerm) :-
    format(string(RubyTerm), '~w', [Atom]).

%% translate_list_fold_ruby(+PrologExpr, +HeadVar, +AccVar, -RubyExpr)
%  Like translate_fold_expr_ruby but maps HeadVar → 'current' (list element in reduce)
%  and AccVar → 'acc' (accumulator in reduce block).
translate_list_fold_ruby(A * B, HeadVar, AccVar, Expr) :-
    translate_list_term_ruby(A, HeadVar, AccVar, AT),
    translate_list_term_ruby(B, HeadVar, AccVar, BT),
    format(string(Expr), '~w * ~w', [AT, BT]).
translate_list_fold_ruby(A + B, HeadVar, AccVar, Expr) :-
    translate_list_term_ruby(A, HeadVar, AccVar, AT),
    translate_list_term_ruby(B, HeadVar, AccVar, BT),
    format(string(Expr), '~w + ~w', [AT, BT]).
translate_list_fold_ruby(A - B, HeadVar, AccVar, Expr) :-
    translate_list_term_ruby(A, HeadVar, AccVar, AT),
    translate_list_term_ruby(B, HeadVar, AccVar, BT),
    format(string(Expr), '~w - ~w', [AT, BT]).
translate_list_fold_ruby(Term, HeadVar, AccVar, Expr) :-
    translate_list_term_ruby(Term, HeadVar, AccVar, Expr).

translate_list_term_ruby(Term, HeadVar, _AccVar, 'current') :- Term == HeadVar, !.
translate_list_term_ruby(Term, _HeadVar, AccVar, 'acc') :- Term == AccVar, !.
translate_list_term_ruby(Number, _, _, RubyTerm) :- integer(Number), !,
    format(string(RubyTerm), '~w', [Number]).
translate_list_term_ruby(Atom, _, _, RubyTerm) :-
    format(string(RubyTerm), '~w', [Atom]).

% ============================================================================
% TREE RECURSION - Ruby target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/tree_recursion').
:- multifile tree_recursion:compile_tree_pattern/6.

tree_recursion:compile_tree_pattern(ruby, _Pattern, Pred, _Arity, _UseMemo, RubyCode) :-
    atom_string(Pred, PredStr),
    format(string(RubyCode),
'# Generated by UnifyWeaver Ruby Target - Tree Recursion (multifile dispatch)
@~w_memo = {}

def ~w(n)
  return @~w_memo[n] if @~w_memo.key?(n)
  return 0 if n <= 0
  return 1 if n == 1
  result = ~w(n - 1) + ~w(n - 2)
  @~w_memo[n] = result
  result
end

if __FILE__ == $0
  if ARGV.length >= 1
    puts ~w(ARGV[0].to_i)
  end
end
', [PredStr, PredStr, PredStr, PredStr, PredStr, PredStr, PredStr, PredStr]).

% ============================================================================
% MULTICALL LINEAR RECURSION - Ruby target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/multicall_linear_recursion').
:- multifile multicall_linear_recursion:compile_multicall_pattern/6.

multicall_linear_recursion:compile_multicall_pattern(ruby, PredStr, BaseClauses, _RecClauses, _MemoEnabled, RubyCode) :-
    findall(BaseCaseCode, (
        member(clause(BHead, _), BaseClauses),
        BHead =.. [_P, BInput, BOutput],
        format(string(BaseCaseCode), '  return ~w if n == ~w', [BOutput, BInput])
    ), BaseCaseCodes0),
    sort(BaseCaseCodes0, BaseCaseCodes),
    atomic_list_concat(BaseCaseCodes, '\n', BaseCaseStr),
    format(string(RubyCode),
'# Generated by UnifyWeaver Ruby Target - Multicall Linear Recursion (multifile dispatch)
@~w_memo = {}

def ~w(n)
  return @~w_memo[n] if @~w_memo.key?(n)
~w
  result = ~w(n - 1) + ~w(n - 2)
  @~w_memo[n] = result
  result
end

if __FILE__ == $0
  if ARGV.length >= 1
    puts ~w(ARGV[0].to_i)
  end
end
', [PredStr, PredStr, PredStr, PredStr, BaseCaseStr, PredStr, PredStr, PredStr, PredStr]).

% ============================================================================
% DIRECT MULTICALL RECURSION - Ruby target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/direct_multi_call_recursion').
:- multifile direct_multi_call_recursion:compile_direct_multicall_pattern/5.

direct_multi_call_recursion:compile_direct_multicall_pattern(ruby, PredStr, BaseClauses, _RecClause, RubyCode) :-
    findall(BaseCaseCode, (
        member(clause(BHead, _), BaseClauses),
        BHead =.. [_P, BInput, BOutput],
        format(string(BaseCaseCode), '  return @~w_memo[~w] = ~w if n == ~w', [PredStr, BInput, BOutput, BInput])
    ), BaseCaseCodes0),
    sort(BaseCaseCodes0, BaseCaseCodes),
    atomic_list_concat(BaseCaseCodes, '\n', BaseCaseStr),
    format(string(RubyCode),
'# Generated by UnifyWeaver Ruby Target - Direct Multicall Recursion (multifile dispatch)
@~w_memo = {}

def ~w(n)
  return @~w_memo[n] if @~w_memo.key?(n)
~w
  result = ~w(n - 1) + ~w(n - 2)
  @~w_memo[n] = result
  result
end

if __FILE__ == $0
  if ARGV.length >= 1
    puts ~w(ARGV[0].to_i)
  end
end
', [PredStr, PredStr, PredStr, PredStr, BaseCaseStr, PredStr, PredStr, PredStr, PredStr]).

% ============================================================================
% MUTUAL RECURSION - Ruby target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/mutual_recursion').
:- use_module('../core/advanced/pattern_matchers', [is_per_path_visited_pattern/4]).
:- multifile mutual_recursion:compile_mutual_pattern/5.

mutual_recursion:compile_mutual_pattern(ruby, Predicates, MemoEnabled, _MemoStrategy, RubyCode) :-
    mutual_functions_ruby(Predicates, Predicates, MemoEnabled, FuncCodes),
    atomic_list_concat(FuncCodes, '\n\n', FunctionsCode),
    mutual_dispatch_ruby(Predicates, DispatchCode),
    format(string(RubyCode),
'# Generated by UnifyWeaver Ruby Target - Mutual Recursion (multifile dispatch)

~w

if __FILE__ == $0
  func = ARGV[0]
  n = ARGV[1].to_i if ARGV.length >= 2
~w
end
', [FunctionsCode, DispatchCode]).

mutual_functions_ruby([], _AllPreds, _MemoEnabled, []).
mutual_functions_ruby([Pred/Arity|Rest], AllPreds, MemoEnabled, [FuncCode|RestCodes]) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),
    partition(mutual_recursion:is_mutual_recursive_clause(AllPreds), Clauses, RecClauses, BaseClauses),
    findall(BaseLine, (
        member(clause(BHead, true), BaseClauses),
        BHead =.. [_P, BValue],
        format(string(BaseLine), '  return true if n == ~w', [BValue])
    ), BaseLines),
    atomic_list_concat(BaseLines, '\n', BaseCode),
    (   RecClauses = [clause(_RHead, RBody)|_] ->
        extract_mutual_rec_info_ruby(RBody, Guard, CalledPred, Step),
        atom_string(CalledPred, CalledStr),
        (   Guard = (N > Threshold), var(N) ->
            (   MemoEnabled = true ->
                format(string(RecCode),
'  if n > ~w
    key = "~w:#{n}"
    return @mutual_memo[key] unless @mutual_memo[key].nil?
    result = ~w(n ~w)
    @mutual_memo[key] = result
    result
  else
    false
  end', [Threshold, PredStr, CalledStr, Step])
            ;   format(string(RecCode),
'  n > ~w ? ~w(n ~w) : false', [Threshold, CalledStr, Step])
            )
        ;   format(string(RecCode), '  ~w(n ~w)', [CalledStr, Step])
        )
    ;   RecCode = '  false'
    ),
    (   MemoEnabled = true ->
        format(string(FuncCode),
'@mutual_memo ||= {}

def ~w(n)
~w
~w
end', [PredStr, BaseCode, RecCode])
    ;   format(string(FuncCode),
'def ~w(n)
~w
~w
end', [PredStr, BaseCode, RecCode])
    ),
    mutual_functions_ruby(Rest, AllPreds, MemoEnabled, RestCodes).

mutual_dispatch_ruby(Predicates, Code) :-
    findall(DispatchLine, (
        member(Pred/_Arity, Predicates),
        atom_string(Pred, PredStr),
        format(string(DispatchLine), '  puts ~w(n) if func == "~w"', [PredStr, PredStr])
    ), Lines),
    atomic_list_concat(Lines, '\n', Code).

extract_mutual_rec_info_ruby(Body, Guard, CalledPred, Step) :-
    extract_goals_ruby(Body, Goals),
    (   member(Guard, Goals), Guard = (_ > _) -> true
    ;   Guard = none
    ),
    member(Call, Goals),
    compound(Call),
    Call \= (_ is _), Call \= (_ > _), Call \= (_ < _),
    Call \= (_ >= _), Call \= (_ =< _),
    functor(Call, CalledPred, _),
    (   member(_ is _ - K, Goals), integer(K) ->
        format(string(Step), '- ~w', [K])
    ;   member(_ is _ + K, Goals), integer(K), K < 0 ->
        AbsK is abs(K),
        format(string(Step), '- ~w', [AbsK])
    ;   Step = "- 1"
    ).

extract_goals_ruby((A, B), Goals) :- !,
    extract_goals_ruby(A, GA),
    extract_goals_ruby(B, GB),
    append(GA, GB, Goals).
extract_goals_ruby(true, []) :- !.
extract_goals_ruby(Goal, [Goal]).

% ============================================================================
% GENERAL RECURSIVE PATTERN (visited-set cycle detection)
% ============================================================================

:- multifile advanced_recursive_compiler:compile_general_recursive_pattern/6.

%% No-visited-pattern — plain recursive without cycle detection
advanced_recursive_compiler:compile_general_recursive_pattern(ruby, PredStr, Arity, BaseClauses, RecClauses, Code) :-
    atom_string(Pred, PredStr),
    append(BaseClauses, RecClauses, AllClauses),
    \+ is_per_path_visited_pattern(Pred, Arity, AllClauses, _),
    !,
    (   BaseClauses = [(BH, true)|_]
    ->  BH =.. [_|BaseArgs], last(BaseArgs, BaseVal),
        BaseArgs = [BaseKey|_],
        format(string(BaseCheck), '    return ["~w"] if arg1 == "~w"', [BaseVal, BaseKey])
    ;   BaseCheck = '    # no base case extracted'
    ),
    format(string(Code),
'# General recursive: ~w (plain, no visited pattern)\n\c
def ~w(arg1)\n\c
~w\n\c
    ~w(arg1)\n\c
end\n',
    [PredStr, PredStr, BaseCheck, PredStr]).

%% Arity-2: wrapper + worker with base case check and recursive accumulation
advanced_recursive_compiler:compile_general_recursive_pattern(ruby, PredStr, 2, BaseClauses, RecClauses, Code) :-
    atom_string(PredAtom, PredStr),
    atom_concat(PredAtom, '_worker', WorkerAtom),
    atom_string(WorkerAtom, WorkerStr),
    %% Extract base case key/value from first base clause
    (   BaseClauses = [(BH, true)|_]
    ->  BH =.. [_, BaseKey, BaseVal],
        format(string(BaseCheck),
            '    return ["~w"] if arg1 == "~w"', [BaseVal, BaseKey])
    ;   BaseCheck = '    # no base case extracted'
    ),
    %% Extract recursive step from first recursive clause
    (   RecClauses = [(_, RecBody)|_]
    ->  extract_rec_call_ruby(RecBody, PredStr, WorkerStr, RecCallExpr)
    ;   format(string(RecCallExpr), '~w(arg1, visited)', [WorkerStr])
    ),
    format(string(Code),
'# General recursive: ~w (with cycle detection)\n\c
require \"set\"\n\c
\n\c
def ~w(arg1)\n\c
    ~w(arg1, Set.new)\n\c
end\n\c
\n\c
def ~w(arg1, visited)\n\c
    return [] if visited.include?(arg1)\n\c
    visited.add(arg1)\n\c
~w\n\c
    sub = ~w\n\c
    sub\n\c
end\n',
    [PredStr, PredStr, WorkerStr, WorkerStr, BaseCheck, RecCallExpr]).

%% Arity-3: wrapper + worker with counter/output style
advanced_recursive_compiler:compile_general_recursive_pattern(ruby, PredStr, 3, BaseClauses, RecClauses, Code) :-
    atom_string(PredAtom, PredStr),
    atom_concat(PredAtom, '_worker', WorkerAtom),
    atom_string(WorkerAtom, WorkerStr),
    (   BaseClauses = [(BH, true)|_]
    ->  BH =.. [_, BaseKey, _, BaseVal],
        format(string(BaseCheck),
            '    return ["~w"] if arg1 == "~w"', [BaseVal, BaseKey])
    ;   BaseCheck = '    # no base case extracted'
    ),
    (   RecClauses = [(_, RecBody)|_]
    ->  extract_rec_call_ruby(RecBody, PredStr, WorkerStr, RecCallExpr)
    ;   format(string(RecCallExpr), '~w(arg1, visited)', [WorkerStr])
    ),
    format(string(Code),
'# General recursive: ~w (with cycle detection)\n\c
require \"set\"\n\c
\n\c
def ~w(arg1)\n\c
    ~w(arg1, Set.new)\n\c
end\n\c
\n\c
def ~w(arg1, visited)\n\c
    return [] if visited.include?(arg1)\n\c
    visited.add(arg1)\n\c
~w\n\c
    ~w\n\c
end\n',
    [PredStr, PredStr, WorkerStr, WorkerStr, BaseCheck, RecCallExpr]).

extract_rec_call_ruby((A, _), PredStr, WorkerStr, Expr) :-
    nonvar(A),
    functor(A, Pred, _),
    atom_string(Pred, PredStr), !,
    A =.. [_|CallArgs],
    (   CallArgs = [Arg1|_]
    ->  format(string(Expr), '~w(~w, visited)', [WorkerStr, Arg1])
    ;   format(string(Expr), '~w(arg1, visited)', [WorkerStr])
    ).
extract_rec_call_ruby((_, B), PredStr, WorkerStr, Expr) :- !,
    extract_rec_call_ruby(B, PredStr, WorkerStr, Expr).
extract_rec_call_ruby(Goal, PredStr, WorkerStr, Expr) :-
    nonvar(Goal),
    functor(Goal, Pred, _),
    atom_string(Pred, PredStr), !,
    Goal =.. [_|CallArgs],
    (   CallArgs = [Arg1|_]
    ->  format(string(Expr), '~w(~w, visited)', [WorkerStr, Arg1])
    ;   format(string(Expr), '~w(arg1, visited)', [WorkerStr])
    ).
extract_rec_call_ruby(_, _PredStr, WorkerStr, Expr) :-
    format(string(Expr), '~w(arg1, visited)', [WorkerStr]).
