:- module(perl_target, [
    compile_predicate_to_perl/3
]).

:- use_module(library(lists)).
:- use_module(library(apply)).

%% compile_predicate_to_perl(+Pred/Arity, +Options, -Code)
%%
%% Compiles a Prolog predicate to Perl code using continuation-passing style.
%% Handles facts, rules with joins, and recursive predicates with memoization.
%%
%% Options:
%%   json_output - Wrap predicate to output JSON array
%%   json_input  - Read facts from JSON input
%%   pipeline    - Generate pipeline-compatible code (read stdin, write stdout)
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

    % Generate Perl header based on options
    (   (member(json_output, Options) ; member(json_input, Options) ; member(pipeline, Options))
    ->  format(string(Header), "#!/usr/bin/env perl~nuse strict;~nuse warnings;~nuse JSON;~n~n", [])
    ;   format(string(Header), "#!/usr/bin/env perl~nuse strict;~nuse warnings;~n~n", [])
    ),

    % Compile clauses
    compile_clauses(Pred, Arity, Clauses, IsRecursive, Options, ClauseCode),

    % Add JSON wrapper if requested
    (   member(json_output, Options)
    ->  generate_json_output_wrapper(Pred, Arity, WrapperCode)
    ;   member(pipeline, Options)
    ->  generate_pipeline_wrapper(Pred, Arity, WrapperCode)
    ;   WrapperCode = ""
    ),

    format(string(Code), "~s~s~s", [Header, ClauseCode, WrapperCode]).

%% generate_json_output_wrapper(+Pred, +Arity, -Code)
generate_json_output_wrapper(Pred, Arity, Code) :-
    numlist(1, Arity, Indices),
    maplist([I, Name]>>format(atom(Name), "$_[~w]", [I]), Indices, VarRefs),
    atomic_list_concat(VarRefs, ", ", VarList),
    format(string(Code),
"
# JSON output wrapper
sub ~w_json {
    my @results;
    ~w(sub {
        push @results, [~s];
    });
    print encode_json(\\@results);
}

# Run if executed directly
~w_json() unless caller;
", [Pred, Pred, VarList, Pred]).

%% generate_pipeline_wrapper(+Pred, +Arity, -Code)
generate_pipeline_wrapper(Pred, Arity, Code) :-
    numlist(1, Arity, Indices),
    maplist([I, Name]>>format(atom(Name), "$_[~w]", [I]), Indices, VarRefs),
    atomic_list_concat(VarRefs, ", ", VarList),
    format(string(Code),
"
# Pipeline mode - reads JSON from stdin, outputs JSON to stdout
sub run_pipeline {
    my @results;
    ~w(sub {
        push @results, [~s];
    });
    print encode_json(\\@results) . \"\\n\";
}

run_pipeline() unless caller;
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
    ->  compile_aggregation(Pred, Arity, Clauses, Code)
    ;   compile_rules(Pred, Arity, Clauses, IsRecursive, Code)
    ).

%% ============================================
%% AGGREGATION COMPILATION
%% ============================================

%% is_aggregation_predicate(+Clauses)
%% Check if clauses use aggregate_all/3
is_aggregation_predicate(Clauses) :-
    member(_-Body, Clauses),
    body_contains_aggregate(Body), !.

body_contains_aggregate(aggregate_all(_, _, _)) :- !.
body_contains_aggregate((A, _)) :- body_contains_aggregate(A), !.
body_contains_aggregate((_, B)) :- body_contains_aggregate(B), !.

%% compile_aggregation(+Pred, +Arity, +Clauses, -Code)
compile_aggregation(Pred, _Arity, Clauses, Code) :-
    Clauses = [Head-Body|_],
    Head =.. [_|HeadArgs],

    % Extract the aggregate_all call
    extract_aggregate(Body, AggType, Template, Goal, ResultVar),

    % Generate parameter names for non-result args
    findall(Idx-Arg, (nth1(Idx, HeadArgs, Arg), Arg \== ResultVar), InputPairs),
    length(InputPairs, NumInputs),
    (   NumInputs > 0
    ->  numlist(1, NumInputs, InputIndices),
        maplist([I, Name]>>format(atom(Name), "$arg~w", [I]), InputIndices, ParamNames),
        atomic_list_concat(ParamNames, ", ", ParamList),
        format(string(ParamDecl), "    my (~s) = @_;~n", [ParamList])
    ;   ParamDecl = ""
    ),

    % Determine aggregation operation
    agg_init_and_update(AggType, Template, InitCode, UpdateCode, FinalCode),

    % Compile the goal - strip module prefix if present
    (   Goal = _:InnerGoal
    ->  InnerGoal =.. [GoalPred|GoalArgs]
    ;   Goal =.. [GoalPred|GoalArgs]
    ),
    length(GoalArgs, GoalArity),
    numlist(1, GoalArity, GoalIndices),
    maplist([I, Name]>>format(atom(Name), "$g~w", [I]), GoalIndices, GoalParamNames),
    atomic_list_concat(GoalParamNames, ", ", GoalParamList),

    % Build template variable extraction
    extract_template_vars(Template, GoalArgs, GoalParamNames, TemplateExtract),

    format(string(Code),
"sub ~w {
    my $callback = shift;
~s~s
    ~w(sub {
        my (~s) = @_;
~s~s
    });
~s
    $callback->($agg_result);
}
", [Pred, ParamDecl, InitCode, GoalPred, GoalParamList, TemplateExtract, UpdateCode, FinalCode]).

%% extract_aggregate(+Body, -AggType, -Template, -Goal, -ResultVar)
extract_aggregate(aggregate_all(AggType, Goal, ResultVar), AggType, Template, Goal, ResultVar) :-
    (   AggType = count -> Template = 1
    ;   AggType = sum(Template) -> true
    ;   AggType = min(Template) -> true
    ;   AggType = max(Template) -> true
    ;   AggType = avg(Template) -> true
    ;   Template = AggType
    ), !.
extract_aggregate((A, _), AggType, Template, Goal, ResultVar) :-
    extract_aggregate(A, AggType, Template, Goal, ResultVar), !.
extract_aggregate((_, B), AggType, Template, Goal, ResultVar) :-
    extract_aggregate(B, AggType, Template, Goal, ResultVar).

%% agg_init_and_update(+AggType, +Template, -InitCode, -UpdateCode, -FinalCode)
agg_init_and_update(count, _, Init, Update, Final) :-
    Init = "    my $agg_result = 0;\n",
    Update = "        $agg_result++;\n",
    Final = "".
agg_init_and_update(sum(_), _, Init, Update, Final) :-
    Init = "    my $agg_result = 0;\n",
    Update = "        $agg_result += $tmpl_val;\n",
    Final = "".
agg_init_and_update(min(_), _, Init, Update, Final) :-
    Init = "    my $agg_result;\n",
    Update = "        $agg_result = $tmpl_val if !defined($agg_result) || $tmpl_val < $agg_result;\n",
    Final = "".
agg_init_and_update(max(_), _, Init, Update, Final) :-
    Init = "    my $agg_result;\n",
    Update = "        $agg_result = $tmpl_val if !defined($agg_result) || $tmpl_val > $agg_result;\n",
    Final = "".
agg_init_and_update(avg(_), _, Init, Update, Final) :-
    Init = "    my $agg_sum = 0;\n    my $agg_count = 0;\n",
    Update = "        $agg_sum += $tmpl_val;\n        $agg_count++;\n",
    Final = "    my $agg_result = $agg_count > 0 ? $agg_sum / $agg_count : 0;\n".

%% extract_template_vars(+Template, +GoalArgs, +GoalParamNames, -Code)
extract_template_vars(Template, GoalArgs, GoalParamNames, Code) :-
    (   var(Template)
    ->  % Template is a variable - find which goal arg it matches
        (   nth1(Idx, GoalArgs, Arg), Arg == Template
        ->  nth1(Idx, GoalParamNames, ParamName),
            format(string(Code), "        my $tmpl_val = ~w;~n", [ParamName])
        ;   Code = "        my $tmpl_val = 1;\n"
        )
    ;   number(Template)
    ->  format(string(Code), "        my $tmpl_val = ~w;~n", [Template])
    ;   % Template is an expression - compile it
        compile_template_expr(Template, GoalArgs, GoalParamNames, ExprCode),
        format(string(Code), "        my $tmpl_val = ~s;~n", [ExprCode])
    ).

compile_template_expr(Expr, GoalArgs, GoalParamNames, Code) :-
    var(Expr), !,
    (   nth1(Idx, GoalArgs, Arg), Arg == Expr
    ->  nth1(Idx, GoalParamNames, Code)
    ;   Code = "0"
    ).
compile_template_expr(N, _, _, Code) :- number(N), !, format(string(Code), "~w", [N]).
compile_template_expr(X + Y, GoalArgs, GoalParamNames, Code) :- !,
    compile_template_expr(X, GoalArgs, GoalParamNames, XCode),
    compile_template_expr(Y, GoalArgs, GoalParamNames, YCode),
    format(string(Code), "(~s + ~s)", [XCode, YCode]).
compile_template_expr(X - Y, GoalArgs, GoalParamNames, Code) :- !,
    compile_template_expr(X, GoalArgs, GoalParamNames, XCode),
    compile_template_expr(Y, GoalArgs, GoalParamNames, YCode),
    format(string(Code), "(~s - ~s)", [XCode, YCode]).
compile_template_expr(X * Y, GoalArgs, GoalParamNames, Code) :- !,
    compile_template_expr(X, GoalArgs, GoalParamNames, XCode),
    compile_template_expr(Y, GoalArgs, GoalParamNames, YCode),
    format(string(Code), "(~s * ~s)", [XCode, YCode]).
compile_template_expr(X / Y, GoalArgs, GoalParamNames, Code) :- !,
    compile_template_expr(X, GoalArgs, GoalParamNames, XCode),
    compile_template_expr(Y, GoalArgs, GoalParamNames, YCode),
    format(string(Code), "(~s / ~s)", [XCode, YCode]).
compile_template_expr(E, _, _, Code) :- format(string(Code), "~w", [E]).

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

%% can_compile_linear_recursion(+Pred, +Arity, +Clauses)
%% Detect linear recursion pattern (recursive calls not in tail position,
%% returns a computed value). Example: fibonacci, tree traversal.
can_compile_linear_recursion(Pred, Arity, Clauses) :-
    partition(is_recursive_clause(Pred, Arity), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [],
    % Must have base cases that set result directly
    forall(member(BC, BaseClauses), is_value_base_case(BC)),
    % Recursive clause should compute result from recursive calls
    forall(member(RC, RecClauses), is_linear_recursive_clause(Pred, Arity, RC)).

%% is_value_base_case(+Clause) - base case returns a constant or input value
is_value_base_case(Head-true) :-
    Head =.. [_|Args],
    Args \= [],
    last(Args, Result),
    (number(Result) ; atom(Result) ; var(Result)), !.
is_value_base_case(Head-Body) :-
    Head =.. [_|Args],
    Args \= [],
    goals_to_list(Body, Goals),
    forall(member(G, Goals), is_simple_condition(G)).

%% is_linear_recursive_clause(+Pred, +Arity, +Clause)
%% Check that recursive calls are not in tail position and result is computed
is_linear_recursive_clause(Pred, Arity, _Head-Body) :-
    goals_to_list(Body, Goals),
    Goals \= [],
    % Recursive call should NOT be last (not tail recursive)
    last(Goals, LastGoal),
    \+ (strip_module(LastGoal, _, G), functor(G, Pred, Arity)),
    % Should have 'is' computation that uses recursive results
    member(_ is _, Goals).

%% compile_linear_recursive_rules(+Pred, +Arity, +Clauses, -Code)
%% Generate memoized linear recursion with CPS
compile_linear_recursive_rules(Pred, Arity, Clauses, Code) :-
    partition(is_recursive_clause(Pred, Arity), Clauses, RecClauses, BaseClauses),

    % Generate parameter names (excluding result - last arg)
    InputArity is Arity - 1,
    numlist(1, InputArity, Indices),
    maplist([I, Name]>>format(atom(Name), "$arg~w", [I]), Indices, ParamNames),
    atomic_list_concat(ParamNames, ", ", ParamList),

    % Generate memo key
    format(string(MemoKey), "join('\\0', ~s)", [ParamList]),

    % Compile base cases
    compile_linear_base_cases(BaseClauses, ParamNames, BaseCaseCode),

    % Compile recursive case
    RecClauses = [RecClause|_],
    compile_linear_recursive_case(RecClause, Pred, Arity, ParamNames, RecCaseCode),

    format(string(Code),
"{
    my %%memo;
    sub ~w {
        my $callback = shift;
        my (~s) = @_;
        my $key = ~s;
        if (exists $memo{$key}) {
            return $callback->($memo{$key});
        }
~s
~s
    }
}
", [Pred, ParamList, MemoKey, BaseCaseCode, RecCaseCode]).

%% compile_linear_base_cases(+BaseClauses, +ParamNames, -Code)
compile_linear_base_cases([], _, "").
compile_linear_base_cases([Clause|Rest], ParamNames, Code) :-
    compile_linear_base_case(Clause, ParamNames, ClauseCode),
    compile_linear_base_cases(Rest, ParamNames, RestCode),
    format(string(Code), "~s~s", [ClauseCode, RestCode]).

compile_linear_base_case(Head-_Body, ParamNames, Code) :-
    Head =.. [_|Args],
    % Get input args and result
    append(InputArgs, [Result], Args),
    % Build condition from input args
    build_base_condition(InputArgs, ParamNames, Condition),
    % Get result value
    (   var(Result)
    ->  % Result is bound to an input arg
        (   nth1(Idx, InputArgs, RA), RA == Result
        ->  nth1(Idx, ParamNames, ResultExpr)
        ;   ResultExpr = "undef"
        )
    ;   format(string(ResultExpr), "~w", [Result])
    ),
    format(string(Code),
"        if (~s) {
            $memo{$key} = ~s;
            return $callback->(~s);
        }
", [Condition, ResultExpr, ResultExpr]).

build_base_condition([], [], "1").
build_base_condition([Arg|Args], [Param|Params], Condition) :-
    build_base_condition(Args, Params, RestCond),
    (   number(Arg)
    ->  format(string(ArgCond), "~s == ~w", [Param, Arg])
    ;   atom(Arg)
    ->  format(string(ArgCond), "~s eq '~w'", [Param, Arg])
    ;   ArgCond = "1"
    ),
    (   RestCond == "1"
    ->  Condition = ArgCond
    ;   ArgCond == "1"
    ->  Condition = RestCond
    ;   format(string(Condition), "~s && ~s", [ArgCond, RestCond])
    ).

%% compile_linear_recursive_case(+Clause, +Pred, +Arity, +ParamNames, -Code)
compile_linear_recursive_case(Head-Body, Pred, Arity, ParamNames, Code) :-
    Head =.. [_|HeadArgs],
    append(_, [ResultVar], HeadArgs),
    goals_to_list(Body, Goals),

    % Separate: conditions, computations, recursive calls, final computation
    partition(is_recursive_goal(Pred, Arity), Goals, RecGoals, NonRecGoals),
    partition(is_computation, NonRecGoals, Computations, _Conditions),

    % Find the final result computation (ResultVar is ...)
    find_final_expr(Computations, ResultVar, FinalExpr),

    % Build computed vars for intermediate values
    collect_linear_computed_vars(Computations, ResultVar, 0, ComputedVars),

    % Generate pre-computations (before recursive calls)
    compile_linear_pre_computations(Computations, ResultVar, HeadArgs, ParamNames, ComputedVars, PreCompCode),

    % Generate nested recursive calls
    compile_nested_recursive_calls(RecGoals, Pred, HeadArgs, ParamNames, ComputedVars, FinalExpr, ResultVar, RecCallCode),

    format(string(Code), "~s~s", [PreCompCode, RecCallCode]).

%% find_final_expr(+Computations, +ResultVar, -FinalExpr)
find_final_expr([], _, 0).
find_final_expr([RV is Expr|_], ResultVar, Expr) :-
    RV == ResultVar, !.
find_final_expr([_|Rest], ResultVar, Expr) :-
    find_final_expr(Rest, ResultVar, Expr).

is_computation(_ is _).

collect_linear_computed_vars([], _, _, []).
collect_linear_computed_vars([Var is _|Rest], ResultVar, Counter, Vars) :-
    var(Var), Var \== ResultVar, !,
    format(atom(Name), "$tmp~w", [Counter]),
    Counter1 is Counter + 1,
    collect_linear_computed_vars(Rest, ResultVar, Counter1, RestVars),
    Vars = [Var-Name|RestVars].
collect_linear_computed_vars([_|Rest], ResultVar, Counter, Vars) :-
    collect_linear_computed_vars(Rest, ResultVar, Counter, Vars).

compile_linear_pre_computations([], _, _, _, _, "").
compile_linear_pre_computations([Var is Expr|Rest], ResultVar, HeadArgs, ParamNames, ComputedVars, Code) :-
    var(Var), Var \== ResultVar, !,
    get_var_name(Var, HeadArgs, ParamNames, ComputedVars, VarName),
    compile_perl_expr_ext(Expr, HeadArgs, ParamNames, ComputedVars, ExprCode),
    compile_linear_pre_computations(Rest, ResultVar, HeadArgs, ParamNames, ComputedVars, RestCode),
    format(string(Code), "        my ~s = ~s;~n~s", [VarName, ExprCode, RestCode]).
compile_linear_pre_computations([_|Rest], ResultVar, HeadArgs, ParamNames, ComputedVars, Code) :-
    compile_linear_pre_computations(Rest, ResultVar, HeadArgs, ParamNames, ComputedVars, Code).

%% compile_nested_recursive_calls(+RecGoals, +Pred, +HeadArgs, +ParamNames, +ComputedVars, +FinalExpr, +ResultVar, +Counter, -Code)
compile_nested_recursive_calls(RecGoals, Pred, HeadArgs, ParamNames, ComputedVars, FinalExpr, ResultVar, Code) :-
    compile_nested_recursive_calls_(RecGoals, Pred, HeadArgs, ParamNames, ComputedVars, FinalExpr, ResultVar, 0, Code).

compile_nested_recursive_calls_([], _Pred, HeadArgs, ParamNames, ComputedVars, FinalExpr, _ResultVar, _Counter, Code) :-
    compile_perl_expr_ext(FinalExpr, HeadArgs, ParamNames, ComputedVars, ExprCode),
    format(string(Code),
"            my $result = ~s;
            $memo{$key} = $result;
            $callback->($result);
", [ExprCode]).

compile_nested_recursive_calls_([RecGoal|Rest], Pred, HeadArgs, ParamNames, ComputedVars, FinalExpr, ResultVar, Counter, Code) :-
    RecGoal =.. [_|RecArgs],
    append(RecInputs, [RecResult], RecArgs),

    % Compile arguments for recursive call
    maplist(compile_rec_arg(HeadArgs, ParamNames, ComputedVars), RecInputs, ArgCodes),
    atomic_list_concat(ArgCodes, ", ", ArgStr),

    % Name for this recursive result (use counter for uniqueness)
    format(atom(ResName), "$r~w", [Counter]),
    NewComputedVars = [RecResult-ResName|ComputedVars],
    NextCounter is Counter + 1,

    % Compile inner nested calls or final computation
    compile_nested_recursive_calls_(Rest, Pred, HeadArgs, ParamNames, NewComputedVars, FinalExpr, ResultVar, NextCounter, InnerCode),

    format(string(Code),
"        ~w(sub {
            my (~s) = @_;
~s
        }, ~s);
", [Pred, ResName, InnerCode, ArgStr]).

compile_rec_arg(HeadArgs, ParamNames, ComputedVars, Arg, Code) :-
    (   var(Arg)
    ->  get_var_name(Arg, HeadArgs, ParamNames, ComputedVars, Code)
    ;   number(Arg)
    ->  format(string(Code), "~w", [Arg])
    ;   format(string(Code), "'~w'", [Arg])
    ).

%% ============================================
%% TAIL RECURSION DETECTION AND COMPILATION
%% ============================================

%% can_compile_tail_recursion(+Pred, +Arity, +Clauses)
%% Check if this is a tail-recursive predicate that can be optimized to a loop.
%% Pattern: base case returns accumulator, recursive case has recursive call last.
%% Example: factorial(0, Acc, Acc). factorial(N, Acc, R) :- N1 is N-1, Acc1 is Acc*N, factorial(N1, Acc1, R).
can_compile_tail_recursion(Pred, Arity, Clauses) :-
    % Separate base and recursive cases
    partition(is_recursive_clause(Pred, Arity), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [],
    % All recursive clauses must have tail call
    forall(member(Clause, RecClauses), is_tail_recursive_clause(Pred, Arity, Clause)),
    % Base cases should unify result with accumulator (last two args same var)
    forall(member(BaseClause, BaseClauses), is_accumulator_base_case(BaseClause)).

%% is_tail_recursive_clause(+Pred, +Arity, +Clause)
%% Check if the recursive call is in tail position (last goal in body)
is_tail_recursive_clause(Pred, Arity, _Head-Body) :-
    goals_to_list(Body, Goals),
    Goals \= [],
    last(Goals, LastGoal),
    strip_module(LastGoal, _, G),
    functor(G, Pred, Arity).

%% is_accumulator_base_case(+Clause)
%% Check if base case returns accumulator (last two args are same variable)
is_accumulator_base_case(Head-true) :-
    Head =.. [_|Args],
    length(Args, Len),
    Len >= 2,
    append(_, [Acc, Result], Args),
    Acc == Result, !.
is_accumulator_base_case(Head-Body) :-
    Head =.. [_|Args],
    length(Args, Len),
    Len >= 2,
    append(_, [Acc, Result], Args),
    Acc == Result,
    % Body should be simple conditions
    goals_to_list(Body, Goals),
    forall(member(G, Goals), is_simple_condition(G)).

%% is_simple_condition(+Goal)
%% Check if goal is a simple condition (comparison, arithmetic)
is_simple_condition(G) :-
    functor(G, Op, 2),
    member(Op, [=, ==, \=, \==, <, >, =<, >=, =:=, =\=, is]).

%% compile_tail_recursive_rules(+Pred, +Arity, +Clauses, -Code)
%% Compile tail recursion to an iterative loop
compile_tail_recursive_rules(Pred, Arity, Clauses, Code) :-
    partition(is_recursive_clause(Pred, Arity), Clauses, RecClauses, BaseClauses),

    % Generate parameter names
    numlist(1, Arity, Indices),
    maplist([I, Name]>>format(atom(Name), "$arg~w", [I]), Indices, ParamNames),
    atomic_list_concat(ParamNames, ", ", ParamList),

    % Find which arg is the accumulator (second to last in result position)
    Arity >= 2,
    AccIdx is Arity - 1,
    _ResultIdx is Arity,  % ResultIdx not needed currently

    % Generate base case condition
    compile_tail_base_conditions(BaseClauses, AccIdx, BaseCondCode),

    % Generate loop body from recursive clause
    RecClauses = [RecClause|_],  % Use first recursive clause
    compile_tail_loop_body(RecClause, Pred, Arity, ParamNames, LoopBodyCode),

    % AccIdx and ResultIdx identify positions for base case check
    % (AccParam and ResultParam not currently used in template)

    format(string(Code),
"sub ~w {
    my $callback = shift;
    my (~s) = @_;

    # Tail recursion optimized to loop
    while (1) {
~s
~s
    }
}
", [Pred, ParamList, BaseCondCode, LoopBodyCode]).

%% compile_tail_base_conditions(+BaseClauses, +AccIdx, -Code)
compile_tail_base_conditions(BaseClauses, AccIdx, Code) :-
    BaseClauses = [Head-Body|_],
    Head =.. [_|Args],
    % Find the base case condition (usually first arg == 0 or similar)
    (   Body == true
    ->  % Check first argument for base condition
        Args = [FirstArg|_],
        (   FirstArg == 0
        ->  format(string(Code),
"        if ($arg1 == 0) {
            $callback->($arg~w);
            return;
        }", [AccIdx])
        ;   FirstArg == []
        ->  format(string(Code),
"        if (!@{$arg1}) {
            $callback->($arg~w);
            return;
        }", [AccIdx])
        ;   format(string(Code),
"        # Base case
        if (defined $arg1 && $arg1 eq '~w') {
            $callback->($arg~w);
            return;
        }", [FirstArg, AccIdx])
        )
    ;   % Has conditions in body
        goals_to_list(Body, Goals),
        compile_perl_conditions(Goals, CondCode),
        format(string(Code),
"        if (~s) {
            $callback->($arg~w);
            return;
        }", [CondCode, AccIdx])
    ).

%% compile_perl_conditions(+Goals, -Code)
compile_perl_conditions([], "1").
compile_perl_conditions([Goal|Rest], Code) :-
    compile_perl_condition(Goal, GoalCode),
    compile_perl_conditions(Rest, RestCode),
    (   RestCode == "1"
    ->  Code = GoalCode
    ;   format(string(Code), "~s && ~s", [GoalCode, RestCode])
    ).

compile_perl_condition(_ == Y, Code) :-
    format(string(Code), "$arg1 == ~w", [Y]).
compile_perl_condition(_ =:= Y, Code) :-
    format(string(Code), "$arg1 == ~w", [Y]).
compile_perl_condition(_ < Y, Code) :-
    format(string(Code), "$arg1 < ~w", [Y]).
compile_perl_condition(_ > Y, Code) :-
    format(string(Code), "$arg1 > ~w", [Y]).
compile_perl_condition(_ =< Y, Code) :-
    format(string(Code), "$arg1 <= ~w", [Y]).
compile_perl_condition(_ >= Y, Code) :-
    format(string(Code), "$arg1 >= ~w", [Y]).
compile_perl_condition(_, "1").

%% compile_tail_loop_body(+RecClause, +Pred, +Arity, +ParamNames, -Code)
compile_tail_loop_body(Head-Body, Pred, Arity, ParamNames, Code) :-
    Head =.. [_|HeadArgs],
    goals_to_list(Body, Goals),

    % Separate the recursive call from other goals
    partition(is_recursive_goal(Pred, Arity), Goals, [RecGoal|_], OtherGoals),
    RecGoal =.. [_|RecArgs],

    % Build a mapping of computed variables
    collect_computed_vars(OtherGoals, 0, ComputedVars),

    % Compile the intermediate computations with computed var tracking
    compile_tail_computations(OtherGoals, HeadArgs, ParamNames, ComputedVars, CompCode),

    % Generate the variable updates for next iteration
    compile_tail_updates(RecArgs, HeadArgs, ParamNames, ComputedVars, UpdateCode),

    format(string(Code), "~s~s", [CompCode, UpdateCode]).

%% collect_computed_vars(+Goals, +Counter, -ComputedVars)
%% Build a list of Var-Name pairs for computed variables
collect_computed_vars([], _, []).
collect_computed_vars([Var is _|Rest], Counter, [Var-Name|RestVars]) :-
    var(Var), !,
    format(atom(Name), "$tmp~w", [Counter]),
    Counter1 is Counter + 1,
    collect_computed_vars(Rest, Counter1, RestVars).
collect_computed_vars([_|Rest], Counter, Vars) :-
    collect_computed_vars(Rest, Counter, Vars).

%% compile_tail_computations(+Goals, +HeadArgs, +ParamNames, +ComputedVars, -Code)
compile_tail_computations([], _, _, _, "").
compile_tail_computations([Goal|Rest], HeadArgs, ParamNames, ComputedVars, Code) :-
    compile_tail_computation(Goal, HeadArgs, ParamNames, ComputedVars, GoalCode),
    compile_tail_computations(Rest, HeadArgs, ParamNames, ComputedVars, RestCode),
    format(string(Code), "~s~s", [GoalCode, RestCode]).

compile_tail_computation(Var is Expr, HeadArgs, ParamNames, ComputedVars, Code) :-
    var(Var), !,
    % Find the variable name (check computed vars first, then head args)
    get_var_name(Var, HeadArgs, ParamNames, ComputedVars, VarName),
    compile_perl_expr_ext(Expr, HeadArgs, ParamNames, ComputedVars, ExprCode),
    format(string(Code), "        my ~s = ~s;~n", [VarName, ExprCode]).
compile_tail_computation(_, _, _, _, "").

%% get_var_name(+Var, +HeadArgs, +ParamNames, +ComputedVars, -Name)
get_var_name(Var, _, _, ComputedVars, Name) :-
    member(V-Name, ComputedVars),
    V == Var, !.
get_var_name(Var, HeadArgs, ParamNames, _, Name) :-
    nth1(Idx, HeadArgs, Arg),
    Arg == Var, !,
    nth1(Idx, ParamNames, Name).
get_var_name(_, _, _, _, "$unknown").

%% compile_perl_expr_ext - extended to handle computed vars
compile_perl_expr_ext(Expr, HeadArgs, ParamNames, ComputedVars, Code) :-
    var(Expr), !,
    get_var_name(Expr, HeadArgs, ParamNames, ComputedVars, Code).
compile_perl_expr_ext(N, _, _, _, Code) :-
    number(N), !,
    format(string(Code), "~w", [N]).
compile_perl_expr_ext(Atom, _, _, _, Code) :-
    atom(Atom), !,
    format(string(Code), "'~w'", [Atom]).
%% Handle X + (-Y) as X - Y (Prolog sometimes represents subtraction this way)
compile_perl_expr_ext(X + Y, HeadArgs, ParamNames, ComputedVars, Code) :-
    number(Y), Y < 0, !,
    NegY is -Y,
    compile_perl_expr_ext(X, HeadArgs, ParamNames, ComputedVars, XCode),
    format(string(Code), "(~s - ~w)", [XCode, NegY]).
compile_perl_expr_ext(X + Y, HeadArgs, ParamNames, ComputedVars, Code) :- !,
    compile_perl_expr_ext(X, HeadArgs, ParamNames, ComputedVars, XCode),
    compile_perl_expr_ext(Y, HeadArgs, ParamNames, ComputedVars, YCode),
    format(string(Code), "(~s + ~s)", [XCode, YCode]).
compile_perl_expr_ext(X - Y, HeadArgs, ParamNames, ComputedVars, Code) :- !,
    compile_perl_expr_ext(X, HeadArgs, ParamNames, ComputedVars, XCode),
    compile_perl_expr_ext(Y, HeadArgs, ParamNames, ComputedVars, YCode),
    format(string(Code), "(~s - ~s)", [XCode, YCode]).
compile_perl_expr_ext(X * Y, HeadArgs, ParamNames, ComputedVars, Code) :- !,
    compile_perl_expr_ext(X, HeadArgs, ParamNames, ComputedVars, XCode),
    compile_perl_expr_ext(Y, HeadArgs, ParamNames, ComputedVars, YCode),
    format(string(Code), "(~s * ~s)", [XCode, YCode]).
compile_perl_expr_ext(X / Y, HeadArgs, ParamNames, ComputedVars, Code) :- !,
    compile_perl_expr_ext(X, HeadArgs, ParamNames, ComputedVars, XCode),
    compile_perl_expr_ext(Y, HeadArgs, ParamNames, ComputedVars, YCode),
    format(string(Code), "(~s / ~s)", [XCode, YCode]).
compile_perl_expr_ext(X, _, _, _, Code) :-
    format(string(Code), "~w", [X]).

%% compile_perl_expr(+Expr, +HeadArgs, +ParamNames, -Code)
compile_perl_expr(Expr, HeadArgs, ParamNames, Code) :-
    var(Expr), !,
    get_perl_var_name(Expr, HeadArgs, ParamNames, Code).
compile_perl_expr(N, _, _, Code) :-
    number(N), !,
    format(string(Code), "~w", [N]).
compile_perl_expr(Atom, _, _, Code) :-
    atom(Atom), !,
    format(string(Code), "'~w'", [Atom]).
compile_perl_expr(X + Y, HeadArgs, ParamNames, Code) :- !,
    compile_perl_expr(X, HeadArgs, ParamNames, XCode),
    compile_perl_expr(Y, HeadArgs, ParamNames, YCode),
    format(string(Code), "(~s + ~s)", [XCode, YCode]).
compile_perl_expr(X - Y, HeadArgs, ParamNames, Code) :- !,
    compile_perl_expr(X, HeadArgs, ParamNames, XCode),
    compile_perl_expr(Y, HeadArgs, ParamNames, YCode),
    format(string(Code), "(~s - ~s)", [XCode, YCode]).
compile_perl_expr(X * Y, HeadArgs, ParamNames, Code) :- !,
    compile_perl_expr(X, HeadArgs, ParamNames, XCode),
    compile_perl_expr(Y, HeadArgs, ParamNames, YCode),
    format(string(Code), "(~s * ~s)", [XCode, YCode]).
compile_perl_expr(X / Y, HeadArgs, ParamNames, Code) :- !,
    compile_perl_expr(X, HeadArgs, ParamNames, XCode),
    compile_perl_expr(Y, HeadArgs, ParamNames, YCode),
    format(string(Code), "(~s / ~s)", [XCode, YCode]).
compile_perl_expr(X, _, _, Code) :-
    format(string(Code), "~w", [X]).

%% get_perl_var_name(+Var, +HeadArgs, +ParamNames, -Name)
get_perl_var_name(Var, HeadArgs, ParamNames, Name) :-
    nth1(Idx, HeadArgs, Arg),
    Arg == Var, !,
    nth1(Idx, ParamNames, Name).
get_perl_var_name(_, _, _, "$tmp").

%% compile_tail_updates(+RecArgs, +HeadArgs, +ParamNames, +ComputedVars, -Code)
compile_tail_updates(RecArgs, HeadArgs, ParamNames, ComputedVars, Code) :-
    length(RecArgs, Len),
    numlist(1, Len, Indices),
    maplist(compile_single_update(HeadArgs, ParamNames, ComputedVars), Indices, RecArgs, UpdateCodes),
    atomic_list_concat(UpdateCodes, "", Code).

compile_single_update(HeadArgs, ParamNames, ComputedVars, Idx, RecArg, Code) :-
    nth1(Idx, ParamNames, ParamName),
    Len is Idx,  % Skip the result argument (last one)
    length(ParamNames, TotalArgs),
    (   Len == TotalArgs
    ->  Code = ""  % Don't update result argument
    ;   var(RecArg)
    ->  % Find what this variable maps to - check computed vars first, then head args
        (   member(V-ComputedName, ComputedVars), V == RecArg
        ->  format(string(Code), "        ~s = ~s;~n", [ParamName, ComputedName])
        ;   nth1(OldIdx, HeadArgs, OldArg), OldArg == RecArg
        ->  nth1(OldIdx, ParamNames, OldName),
            (   OldName == ParamName
            ->  Code = ""  % No change needed
            ;   format(string(Code), "        ~s = ~s;~n", [ParamName, OldName])
            )
        ;   % Unknown variable - shouldn't happen
            format(string(Code), "        # ~s: unknown source~n", [ParamName])
        )
    ;   number(RecArg)
    ->  format(string(Code), "        ~s = ~w;~n", [ParamName, RecArg])
    ;   format(string(Code), "        ~s = '~w';~n", [ParamName, RecArg])
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
