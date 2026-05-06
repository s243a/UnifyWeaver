:- module(perl_target, [
    compile_predicate_to_perl/3
]).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module('../core/clause_body_analysis').

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
    my %memo;
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

%% Non-recursive rules — try native lowering first
compile_nonrecursive_rules(Pred, Arity, Clauses, Code) :-
    native_perl_clause_body(Pred/Arity, Clauses, FuncBody),
    !,
    atom_string(Pred, PredStr),
    Arity1 is Arity - 1,
    build_perl_arg_list(Arity1, ArgList),
    format(string(Code),
'sub ~w {
~w~w
}

# CLI entry point
if (@ARGV) {
    print ~w($ARGV[0]), "\\n";
}
', [PredStr, ArgList, FuncBody, PredStr]).

%% Fallback: non-recursive rules use simple CPS
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
%% ============================================
%% NATIVE CLAUSE BODY LOWERING
%% ============================================

%% build_perl_arg_list(+N, -ArgList)
%  Generates "    my ($arg1, $arg2) = @_;\n" or "" for arity 0
build_perl_arg_list(0, "") :- !.
build_perl_arg_list(N, ArgList) :-
    findall(Arg, (
        between(1, N, I),
        format(string(Arg), '$arg~w', [I])
    ), Args),
    atomic_list_concat(Args, ', ', ArgStr),
    format(string(ArgList), '    my (~w) = @_;~n', [ArgStr]).

%% native_perl_clause_body(+PredSpec, +Clauses, -Code)

% Single clause
native_perl_clause_body(PredSpec, [Head-Body], Code) :-
    native_perl_clause(PredSpec, Head, Body, Condition, ClauseCode),
    !,
    (   Condition == "true"
    ->  format(string(Code), '    return ~w;', [ClauseCode])
    ;   format(string(Code),
'    if (~w) {
        return ~w;
    }
    die "No matching clause for ~w\\n";', [Condition, ClauseCode, PredSpec])
    ).

% Multi-clause → if/elsif/else
native_perl_clause_body(PredSpec, Clauses, Code) :-
    Clauses = [_|[_|_]],
    maplist(native_perl_clause_pair(PredSpec), Clauses, Branches),
    Branches \= [],
    branches_to_perl_if_chain(Branches, PredSpec, Code).

native_perl_clause_pair(PredSpec, Head-Body, branch(Condition, ClauseCode)) :-
    native_perl_clause(PredSpec, Head, Body, Condition, ClauseCode),
    !.

%% native_perl_clause(+PredSpec, +Head, +Body, -Condition, -Code)
native_perl_clause(_PredSpec, Head, Body, Condition, Code) :-
    Head =.. [_Pred|HeadArgs],
    length(HeadArgs, Arity),
    build_head_varmap(HeadArgs, 1, VarMap),
    (   Arity > 1
    ->  append(_InputHeadArgs, [OutputHeadArg], HeadArgs),
        perl_head_conditions(HeadArgs, 1, Arity, HeadConditions)
    ;   OutputHeadArg = _,
        perl_head_conditions(HeadArgs, 1, Arity, HeadConditions)
    ),
    normalize_goals(Body, Goals),
    (   Goals == []
    ->  perl_resolve_value(VarMap, OutputHeadArg, Code),
        GoalConditions = []
    ;   (   Arity > 1, nonvar(OutputHeadArg)
        ->  clause_guard_output_split(Goals, VarMap, GuardGoals, OutputGoals),
            maplist(perl_guard_condition(VarMap), GuardGoals, GoalConditions),
            (   OutputGoals == []
            ->  perl_literal(OutputHeadArg, Code)
            ;   perl_output_goals(OutputGoals, VarMap, Code)
            )
        ;   native_perl_goal_sequence(Goals, VarMap, GoalConditions, Code)
        )
    ),
    append(HeadConditions, GoalConditions, AllConditions),
    combine_perl_conditions(AllConditions, Condition).

%% perl_head_conditions(+HeadArgs, +Index, +Arity, -Conditions)
perl_head_conditions([], _, _, []).
perl_head_conditions([_], _, Arity, []) :- Arity > 1, !.
perl_head_conditions([HeadArg|Rest], Index, Arity, Conditions) :-
    (   var(HeadArg)
    ->  Conditions = RestConditions
    ;   format(string(ArgName), '$arg~w', [Index]),
        perl_literal(HeadArg, Literal),
        (   atom(HeadArg)
        ->  format(string(Cond), '~w eq ~w', [ArgName, Literal])
        ;   format(string(Cond), '~w == ~w', [ArgName, Literal])
        ),
        Conditions = [Cond|RestConditions]
    ),
    NextIndex is Index + 1,
    perl_head_conditions(Rest, NextIndex, Arity, RestConditions).

%% native_perl_goal_sequence(+Goals, +VarMap, -Conditions, -Code)
%  Uses classify_goal_sequence for advanced pattern detection.
%  Falls back to clause_guard_output_split if classification fails.
native_perl_goal_sequence(Goals, VarMap, Conditions, Code) :-
    classify_goal_sequence(Goals, VarMap, ClassifiedGoals),
    ClassifiedGoals \= [],
    perl_render_classified_goals(ClassifiedGoals, VarMap, Conditions, Lines),
    Lines \= [],
    atomic_list_concat(Lines, '\n', Code),
    !.
native_perl_goal_sequence(Goals, VarMap, Conditions, Code) :-
    clause_guard_output_split(Goals, VarMap, GuardGoals, OutputGoals),
    maplist(perl_guard_condition(VarMap), GuardGoals, Conditions),
    perl_output_goals(OutputGoals, VarMap, Code).

%% perl_render_classified_goals(+ClassifiedGoals, +VarMap, -Conditions, -Lines)
perl_render_classified_goals([], _VarMap, [], []).
perl_render_classified_goals([Classified], VarMap, Conds, Lines) :-
    !,
    perl_render_classified_last(Classified, VarMap, Conds, Lines).
%% Guarded tail: output followed by guard(s)
perl_render_classified_goals([output(Goal, _, _)|Rest], VarMap, [], Lines) :-
    Rest = [guard(_, _)|_],
    !,
    perl_output_goal(Goal, VarMap, AssignLine, VarMap1),
    perl_collect_trailing_guards(Rest, VarMap1, GuardGoals, _Remaining),
    maplist(perl_guard_condition(VarMap1), GuardGoals, GuardConds),
    atomic_list_concat(GuardConds, ' && ', GuardExpr),
    (   goal_output_var(Goal, OutVar), lookup_var(OutVar, VarMap1, OutName)
    ->  format(string(OutRef), '$~w', [OutName])
    ;   OutRef = "undef"
    ),
    format(string(IfLine), '    if (~w) {', [GuardExpr]),
    format(string(RetLine), '        return ~w;', [OutRef]),
    CloseLine = '    }',
    Lines = [AssignLine, IfLine, RetLine, CloseLine].
perl_render_classified_goals([Classified|Rest], VarMap, Conds, Lines) :-
    perl_render_classified_mid(Classified, VarMap, MidConds, MidLines, VarMap1),
    perl_render_classified_goals(Rest, VarMap1, RestConds, RestLines),
    append(MidConds, RestConds, Conds),
    append(MidLines, RestLines, Lines).

%% perl_render_classified_mid(+Classified, +VarMap, -Conds, -Lines, -VarMapOut)
perl_render_classified_mid(guard(Goal, _), VarMap, [Cond], [], VarMap) :-
    perl_guard_condition(VarMap, Goal, Cond).
perl_render_classified_mid(output(Goal, _, _), VarMap0, [], [Line], VarMapOut) :-
    perl_output_goal(Goal, VarMap0, Line, VarMapOut).
perl_render_classified_mid(output_ite(If, Then, Else, _SharedVars), VarMap0, [], Lines, VarMap0) :-
    perl_guard_condition(VarMap0, If, Cond),
    perl_branch_value(Then, VarMap0, ThenExpr),
    perl_branch_value(Else, VarMap0, ElseExpr),
    format(string(IfLine), '    if (~w) {', [Cond]),
    format(string(ThenLine), '        return ~w;', [ThenExpr]),
    ElseLine = '    } else {',
    format(string(ElseRetLine), '        return ~w;', [ElseExpr]),
    Lines = [IfLine, ThenLine, ElseLine, ElseRetLine, '    }'].
perl_render_classified_mid(passthrough(Goal), VarMap0, [], [Line], VarMapOut) :-
    perl_output_goal(Goal, VarMap0, Line, VarMapOut).
perl_render_classified_mid(_, VarMap, [], [], VarMap).

%% perl_render_classified_last(+Classified, +VarMap, -Conds, -Lines)
perl_render_classified_last(guard(Goal, _), VarMap, [Cond], []) :-
    perl_guard_condition(VarMap, Goal, Cond).
perl_render_classified_last(output(Goal, _, _), VarMap, [], Lines) :-
    perl_output_goal_last_lines(Goal, VarMap, Lines).
perl_render_classified_last(output_ite(If, Then, Else, _), VarMap, [], Lines) :-
    perl_guard_condition(VarMap, If, Cond),
    perl_branch_value(Then, VarMap, ThenExpr),
    perl_branch_value(Else, VarMap, ElseExpr),
    format(string(IfLine), '    if (~w) {', [Cond]),
    format(string(ThenLine), '        return ~w;', [ThenExpr]),
    ElseLine = '    } else {',
    format(string(ElseRetLine), '        return ~w;', [ElseExpr]),
    Lines = [IfLine, ThenLine, ElseLine, ElseRetLine, '    }'].
perl_render_classified_last(output_disj(Alternatives, _SharedVars), VarMap, [], Lines) :-
    perl_disj_if_chain(Alternatives, VarMap, Lines).
perl_render_classified_last(passthrough(Goal), VarMap, [], Lines) :-
    perl_output_goal_last_lines(Goal, VarMap, Lines).
perl_render_classified_last(_, _, [], []).

%% perl_output_goal_last_lines(+Goal, +VarMap, -Lines)
perl_output_goal_last_lines(Goal, VarMap, [Line]) :-
    perl_output_goal(Goal, VarMap, AssignLine, VarMapOut),
    (   goal_output_var(Goal, OutVar), lookup_var(OutVar, VarMapOut, OutName)
    ->  format(string(RetPart), '\n    return $~w;', [OutName]),
        atom_concat(AssignLine, RetPart, Line)
    ;   Line = AssignLine
    ).
perl_output_goal_last_lines(Goal, VarMap, [Line]) :-
    perl_branch_value(Goal, VarMap, Expr),
    format(string(Line), '    return ~w;', [Expr]).

%% perl_collect_trailing_guards(+ClassifiedGoals, +VarMap, -GuardGoals, -Remaining)
perl_collect_trailing_guards([guard(Goal, _)|Rest], VarMap, [Goal|Guards], Remaining) :-
    !, perl_collect_trailing_guards(Rest, VarMap, Guards, Remaining).
perl_collect_trailing_guards(Remaining, _, [], Remaining).

%% perl_disj_if_chain(+Alternatives, +VarMap, -Lines)
perl_disj_if_chain([], _, []).
perl_disj_if_chain([Alt], VarMap, [ElseLine, RetLine, CloseLine]) :-
    !,
    perl_branch_value(Alt, VarMap, ValExpr),
    ElseLine = '    } else {',
    format(string(RetLine), '        return ~w;', [ValExpr]),
    CloseLine = '    }'.
perl_disj_if_chain([Alt|Rest], VarMap, Lines) :-
    normalize_goals(Alt, Goals),
    clause_guard_output_split(Goals, VarMap, Guards, _Outputs),
    (   Guards \= []
    ->  maplist(perl_guard_condition(VarMap), Guards, CondStrs),
        atomic_list_concat(CondStrs, ' && ', CondExpr)
    ;   CondExpr = "1"
    ),
    perl_branch_value(Alt, VarMap, ValExpr),
    format(string(IfLine), '    if (~w) {', [CondExpr]),
    format(string(RetLine), '        return ~w;', [ValExpr]),
    perl_disj_elsif_chain(Rest, VarMap, RestLines),
    append([IfLine, RetLine], RestLines, Lines).

perl_disj_elsif_chain([], _, []).
perl_disj_elsif_chain([Alt], VarMap, [ElseLine, RetLine, CloseLine]) :-
    !,
    perl_branch_value(Alt, VarMap, ValExpr),
    ElseLine = '    } else {',
    format(string(RetLine), '        return ~w;', [ValExpr]),
    CloseLine = '    }'.
perl_disj_elsif_chain([Alt|Rest], VarMap, [ElsifLine, RetLine|RestLines]) :-
    normalize_goals(Alt, Goals),
    clause_guard_output_split(Goals, VarMap, Guards, _Outputs),
    (   Guards \= []
    ->  maplist(perl_guard_condition(VarMap), Guards, CondStrs),
        atomic_list_concat(CondStrs, ' && ', CondExpr)
    ;   CondExpr = "1"
    ),
    perl_branch_value(Alt, VarMap, ValExpr),
    format(string(ElsifLine), '    } elsif (~w) {', [CondExpr]),
    format(string(RetLine), '        return ~w;', [ValExpr]),
    perl_disj_elsif_chain(Rest, VarMap, RestLines).

%% perl_guard_condition(+VarMap, +Goal, -Condition)
perl_guard_condition(VarMap, _Module:Goal, Condition) :-
    !, perl_guard_condition(VarMap, Goal, Condition).
perl_guard_condition(VarMap, Goal, Condition) :-
    compound(Goal),
    Goal =.. [Op, Left, Right],
    expr_op(Op, StdOp),
    !,
    perl_expr(Left, VarMap, PLeft),
    perl_expr(Right, VarMap, PRight),
    (   (atom(Left) ; atom(Right)), (StdOp == '==' ; StdOp == '!=')
    ->  (StdOp == '==' -> POp = 'eq' ; POp = 'ne')
    ;   perl_op(StdOp, POp)
    ),
    format(string(Condition), '~w ~w ~w', [PLeft, POp, PRight]).

%% perl_output_goals(+Goals, +VarMap, -Code)
perl_output_goals([], _VarMap, '"error"') :- !.
perl_output_goals([Goal], VarMap, Code) :-
    !, perl_output_goal_last(Goal, VarMap, Code).
perl_output_goals([Goal|Rest], VarMap0, Code) :-
    perl_output_goal(Goal, VarMap0, _Line, VarMap1),
    perl_output_goals(Rest, VarMap1, Code).

%% perl_output_goal_last — produce the return expression
perl_output_goal_last(_Module:Goal, VarMap, Code) :-
    !, perl_output_goal_last(Goal, VarMap, Code).
perl_output_goal_last(Goal, VarMap, Code) :-
    if_then_else_goal(Goal, IfGoal, ThenGoal, ElseGoal),
    !,
    perl_if_then_else_output(IfGoal, ThenGoal, ElseGoal, VarMap, Code).
perl_output_goal_last(=(Var, Expr), VarMap, Code) :-
    var(Var), !,
    perl_expr(Expr, VarMap, Code).
perl_output_goal_last(is(Var, Expr), VarMap, Code) :-
    var(Var), !,
    perl_expr(Expr, VarMap, Code).

%% perl_output_goal — produce a my assignment (not used as return)
perl_output_goal(_Module:Goal, VarMap0, Line, VarMapOut) :-
    !, perl_output_goal(Goal, VarMap0, Line, VarMapOut).
perl_output_goal(=(Var, Expr), VarMap0, Line, VarMapOut) :-
    var(Var), !,
    ensure_var(VarMap0, Var, VarName, VarMapOut),
    perl_expr(Expr, VarMap0, PExpr),
    format(string(Line), 'my $~w = ~w;', [VarName, PExpr]).
perl_output_goal(is(Var, Expr), VarMap0, Line, VarMapOut) :-
    var(Var), !,
    ensure_var(VarMap0, Var, VarName, VarMapOut),
    perl_expr(Expr, VarMap0, PExpr),
    format(string(Line), 'my $~w = ~w;', [VarName, PExpr]).

%% perl_if_then_else_output — generate ternary expressions
perl_if_then_else_output(IfGoal, ThenGoal, ElseGoal, VarMap, Code) :-
    flatten_perl_if_branches(IfGoal, ThenGoal, ElseGoal, Branches, DefaultGoal),
    perl_branches_to_ternary(Branches, DefaultGoal, VarMap, Code).

flatten_perl_if_branches(If, Then, Else, [branch(If, Then)|RestBranches], Default) :-
    if_then_else_goal(Else, If2, Then2, Else2),
    !,
    flatten_perl_if_branches(If2, Then2, Else2, RestBranches, Default).
flatten_perl_if_branches(If, Then, Else, [branch(If, Then)], Else).

perl_branches_to_ternary([branch(If, Then)], DefaultGoal, VarMap, Code) :-
    !,
    perl_guard_condition(VarMap, If, IfCond),
    perl_branch_value(Then, VarMap, ThenVal),
    perl_branch_value(DefaultGoal, VarMap, ElseVal),
    format(string(Code), '(~w) ? ~w : ~w', [IfCond, ThenVal, ElseVal]).
perl_branches_to_ternary([branch(If, Then)|Rest], DefaultGoal, VarMap, Code) :-
    perl_guard_condition(VarMap, If, IfCond),
    perl_branch_value(Then, VarMap, ThenVal),
    perl_branches_to_ternary(Rest, DefaultGoal, VarMap, ElseCode),
    format(string(Code), '(~w) ? ~w : ~w', [IfCond, ThenVal, ElseCode]).

%% perl_branch_value — extract result value from a branch
perl_branch_value(_Module:Goal, VarMap, Value) :-
    !, perl_branch_value(Goal, VarMap, Value).
perl_branch_value(Goal, VarMap, Value) :-
    if_then_else_goal(Goal, If, Then, Else),
    !,
    perl_guard_condition(VarMap, If, Cond),
    perl_branch_value(Then, VarMap, ThenVal),
    perl_branch_value(Else, VarMap, ElseVal),
    format(string(Value), '(~w) ? ~w : ~w', [Cond, ThenVal, ElseVal]).
perl_branch_value((A, B), VarMap, Value) :-
    !,
    normalize_goals((A, B), Goals),
    last(Goals, LastGoal),
    perl_branch_value(LastGoal, VarMap, Value).
perl_branch_value(=(_, Expr), VarMap, Value) :-
    !, perl_expr(Expr, VarMap, Value).
perl_branch_value(is(_, Expr), VarMap, Value) :-
    !, perl_expr(Expr, VarMap, Value).
perl_branch_value(Goal, VarMap, Value) :-
    perl_expr(Goal, VarMap, Value).

% ============================================================================
% MULTIFILE HOOKS — Register Perl renderers for shared compile_expression
% ============================================================================

clause_body_analysis:render_output_goal(perl, Goal, VarMap, Line, VarName, VarMapOut) :-
    perl_output_goal(Goal, VarMap, Line, VarMapOut),
    (   goal_output_var(Goal, OutVar), lookup_var(OutVar, VarMapOut, VarName)
    ->  true
    ;   VarName = "_"
    ).

clause_body_analysis:render_guard_condition(perl, Goal, VarMap, CondStr) :-
    perl_guard_condition(VarMap, Goal, CondStr).

clause_body_analysis:render_branch_value(perl, Branch, VarMap, ExprStr) :-
    perl_branch_value(Branch, VarMap, ExprStr).

clause_body_analysis:render_ite_block(perl, Cond, ThenLines, ElseLines, Indent, _ReturnVars, Lines) :-
    format(string(IfLine), '~wif (~w) {', [Indent, Cond]),
    perl_indent_lines(ThenLines, Indent, IndentedThen),
    (   ElseLines \= []
    ->  format(string(ElseLine), '~w} else {', [Indent]),
        perl_indent_lines(ElseLines, Indent, IndentedElse),
        format(string(EndLine), '~w}', [Indent]),
        append([IfLine|IndentedThen], [ElseLine|IndentedElse], PreEnd),
        append(PreEnd, [EndLine], Lines)
    ;   format(string(EndLine), '~w}', [Indent]),
        append([IfLine|IndentedThen], [EndLine], Lines)
    ).

perl_indent_lines([], _, []).
perl_indent_lines([Line|Rest], Indent, [Indented|RestIndented]) :-
    format(string(Indented), '~w    ~w', [Indent, Line]),
    perl_indent_lines(Rest, Indent, RestIndented).

%% perl_expr — convert Prolog expression to Perl syntax
perl_expr(Var, VarMap, PExpr) :-
    var(Var), !,
    (   lookup_var(Var, VarMap, Name)
    ->  format(string(PExpr), '$~w', [Name])
    ;   term_string(Var, PExpr)
    ).
perl_expr(Expr, VarMap, PExpr) :-
    compound(Expr),
    Expr =.. [Op, Left, Right],
    expr_op(Op, StdOp),
    !,
    perl_expr(Left, VarMap, PLeft),
    perl_expr(Right, VarMap, PRight),
    perl_op(StdOp, POp),
    format(string(PExpr), '(~w ~w ~w)', [PLeft, POp, PRight]).
perl_expr(-Expr, VarMap, PExpr) :-
    !,
    perl_expr(Expr, VarMap, Inner),
    format(string(PExpr), '(-~w)', [Inner]).
perl_expr(abs(Expr), VarMap, PExpr) :-
    !,
    perl_expr(Expr, VarMap, Inner),
    format(string(PExpr), 'abs(~w)', [Inner]).
perl_expr(Atom, _VarMap, PExpr) :-
    atom(Atom), !,
    perl_literal(Atom, PExpr).
perl_expr(Number, _VarMap, PExpr) :-
    number(Number), !,
    format(string(PExpr), '~w', [Number]).
perl_expr(String, _VarMap, PExpr) :-
    string(String), !,
    format(string(PExpr), '"~w"', [String]).

%% perl_literal — convert Prolog value to Perl literal
perl_literal(Value, 'undef') :- var(Value), !.
perl_literal(true, '1') :- !.
perl_literal(false, '0') :- !.
perl_literal(Value, PerlLiteral) :-
    number(Value), !,
    format(string(PerlLiteral), '~w', [Value]).
perl_literal(Value, PerlLiteral) :-
    atom(Value), !,
    format(string(PerlLiteral), '"~w"', [Value]).
perl_literal(Value, PerlLiteral) :-
    string(Value), !,
    format(string(PerlLiteral), '"~w"', [Value]).
perl_literal(Value, PerlLiteral) :-
    term_string(Value, S),
    format(string(PerlLiteral), '"~w"', [S]).

%% perl_resolve_value — resolve variable or constant to Perl expression
perl_resolve_value(VarMap, Var, PExpr) :-
    var(Var), !,
    lookup_var(Var, VarMap, Name),
    format(string(PExpr), '$~w', [Name]).
perl_resolve_value(_VarMap, Value, PExpr) :-
    perl_literal(Value, PExpr).

%% perl_op — map standard operator to Perl syntax
perl_op('>', '>').
perl_op('<', '<').
perl_op('>=', '>=').
perl_op('<=', '<=').
perl_op('==', '==').
perl_op('!=', '!=').
perl_op('+', '+').
perl_op('-', '-').
perl_op('*', '*').
perl_op('/', '/').
perl_op('%', '%').
perl_op('&&', '&&').
perl_op('||', '||').

%% combine_perl_conditions — join conditions with &&
combine_perl_conditions([], "true") :- !.
combine_perl_conditions([Condition], Condition) :- !.
combine_perl_conditions(Conditions, Combined) :-
    atomic_list_concat(Conditions, ' && ', Combined).

%% branches_to_perl_if_chain — build Perl if/elsif/else chain
branches_to_perl_if_chain(Branches, PredSpec, Code) :-
    branches_to_perl_if_lines(Branches, PredSpec, Lines),
    atomic_list_concat(Lines, '\n', Code).

branches_to_perl_if_lines([branch(Condition, ClauseCode)], PredSpec, [IfLine, RetLine, ElseLine, ErrLine, CloseLine]) :-
    !,
    format(string(IfLine), '    if (~w) {', [Condition]),
    format(string(RetLine), '        return ~w;', [ClauseCode]),
    ElseLine = '    } else {',
    format(string(ErrLine), '        die "No matching clause for ~w\\n";', [PredSpec]),
    CloseLine = '    }'.
branches_to_perl_if_lines([branch(Condition, ClauseCode)|Rest], PredSpec, [IfLine, RetLine|RestLines]) :-
    format(string(IfLine), '    if (~w) {', [Condition]),
    format(string(RetLine), '        return ~w;', [ClauseCode]),
    branches_to_perl_elsif_lines(Rest, PredSpec, RestLines).

branches_to_perl_elsif_lines([branch(Condition, ClauseCode)], PredSpec, [ElifLine, RetLine, ElseLine, ErrLine, CloseLine]) :-
    !,
    format(string(ElifLine), '    } elsif (~w) {', [Condition]),
    format(string(RetLine), '        return ~w;', [ClauseCode]),
    ElseLine = '    } else {',
    format(string(ErrLine), '        die "No matching clause for ~w\\n";', [PredSpec]),
    CloseLine = '    }'.
branches_to_perl_elsif_lines([branch(Condition, ClauseCode)|Rest], PredSpec, [ElifLine, RetLine|RestLines]) :-
    format(string(ElifLine), '    } elsif (~w) {', [Condition]),
    format(string(RetLine), '        return ~w;', [ClauseCode]),
    branches_to_perl_elsif_lines(Rest, PredSpec, RestLines).

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

% ============================================================================
% MULTIFILE DISPATCH - Tail Recursion
% ============================================================================

:- use_module('../core/advanced/tail_recursion').
:- multifile tail_recursion:compile_tail_pattern/9.

tail_recursion:compile_tail_pattern(perl, PredStr, Arity, _BaseClauses, _RecClauses, _AccPos, StepOp, _ExitAfterResult, Code) :-
    step_op_to_perl(StepOp, PerlStepExpr),
    (   Arity =:= 3 ->
        format(string(Code),
'#!/usr/bin/env perl
# Generated by UnifyWeaver Perl Target - Tail Recursion (multifile dispatch)
# Predicate: ~w/~w
use strict;
use warnings;

sub ~w {
    my ($items_ref, $acc) = @_;
    $acc //= 0;
    for my $item (@$items_ref) {
        $acc = ~w;
    }
    return $acc;
}

if (@ARGV) {
    my @items = split(/,/, $ARGV[0]);
    print ~w(\\@items), "\\n";
}
', [PredStr, Arity, PredStr, PerlStepExpr, PredStr])
    ;   Arity =:= 2 ->
        format(string(Code),
'#!/usr/bin/env perl
# Generated by UnifyWeaver Perl Target - Tail Recursion (binary, multifile dispatch)
# Predicate: ~w/~w
use strict;
use warnings;

sub ~w {
    my ($items_ref) = @_;
    my $count = 0;
    $count++ for @$items_ref;
    return $count;
}

if (@ARGV) {
    my @items = split(/,/, $ARGV[0]);
    print ~w(\\@items), "\\n";
}
', [PredStr, Arity, PredStr, PredStr])
    ;   fail
    ).

%% step_op_to_perl(+StepOp, -PerlExpr)
step_op_to_perl(arithmetic(Expr), PerlExpr) :-
    expr_to_perl(Expr, PerlExpr).
step_op_to_perl(unknown, '$acc + 1').

expr_to_perl(_ + Const, PerlExpr) :- integer(Const), !,
    format(atom(PerlExpr), '$acc + ~w', [Const]).
expr_to_perl(_ + _, '$acc + $item') :- !.
expr_to_perl(_ - _, '$acc - $item') :- !.
expr_to_perl(_ * _, '$acc * $item') :- !.
expr_to_perl(_, '$acc + 1').

% ============================================================================
% MULTIFILE DISPATCH - Linear Recursion
% ============================================================================

:- use_module('../core/advanced/linear_recursion').
:- multifile linear_recursion:compile_linear_pattern/8.

linear_recursion:compile_linear_pattern(perl, PredStr, Arity, BaseClauses, _RecClauses, MemoEnabled, _MemoStrategy, Code) :-
    (   Arity =:= 2 ->
        linear_fold_perl(PredStr, BaseClauses, MemoEnabled, Code)
    ;   linear_generic_perl(PredStr, Arity, MemoEnabled, Code)
    ).

%% linear_fold_perl(+PredStr, +BaseClauses, +MemoEnabled, -Code)
linear_fold_perl(PredStr, BaseClauses, MemoEnabled, Code) :-
    linear_recursion:extract_base_case_info(BaseClauses, BaseInput, BaseOutput),
    linear_recursion:detect_input_type(BaseInput, InputType),
    (   MemoEnabled = true ->
        format(string(MemoDecl), 'my %~w_memo;', [PredStr]),
        format(string(MemoCheck), '    return $~w_memo{$n} if exists $~w_memo{$n};', [PredStr, PredStr]),
        format(string(MemoStore), '    $~w_memo{$n} = $result;', [PredStr])
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
        translate_fold_expr_perl(FoldExpr, InputVar, AccVar, PerlOp)
    ;   PerlOp = "$current * $acc"
    ),
    (   InputType = numeric ->
        format(string(Code),
'#!/usr/bin/env perl
# Generated by UnifyWeaver Perl Target - Linear Recursion (numeric, multifile dispatch)
# Predicate: ~w/2
use strict;
use warnings;
use List::Util qw(reduce);

~w

sub ~w {
    my ($n) = @_;
~w
    return ~w if $n == ~w;
    my $result = reduce { ~w } ~w, reverse(1 .. $n);
~w
    return $result;
}

print ~w($ARGV[0]), "\\n" if @ARGV;
', [PredStr, MemoDecl, PredStr, MemoCheck, BaseOutput, BaseInput, PerlOp, BaseOutput, MemoStore, PredStr])
    ;   InputType = list ->
        % Re-extract fold with head variable for list patterns
        (   ActualRec = [clause(LRHead, LRBody)|_] ->
            linear_recursion:find_last_is_expression(LRBody, _ is LFoldExpr),
            linear_recursion:find_recursive_call(LRBody, LRecCall),
            LRecCall =.. [_, _, LAccVar],
            LRHead =.. [_, [LHeadVar|_], _],
            translate_list_fold_perl(LFoldExpr, LHeadVar, LAccVar, ListFoldOp)
        ;   ListFoldOp = "$a + $b"
        ),
        format(string(Code),
'#!/usr/bin/env perl
# Generated by UnifyWeaver Perl Target - Linear Recursion (list, multifile dispatch)
# Predicate: ~w/2
use strict;
use warnings;
use List::Util qw(reduce);

sub ~w {
    my ($lst_ref) = @_;
    return ~w unless @$lst_ref;
    return reduce { ~w } ~w, @$lst_ref;
}

if (@ARGV) {
    my @items = split(/,/, $ARGV[0]);
    print ~w(\\@items), "\\n";
}
', [PredStr, PredStr, BaseOutput, ListFoldOp, BaseOutput, PredStr])
    ;   linear_generic_perl(PredStr, 2, MemoEnabled, Code)
    ).

%% linear_generic_perl(+PredStr, +Arity, +MemoEnabled, -Code)
linear_generic_perl(PredStr, Arity, MemoEnabled, Code) :-
    (   MemoEnabled = true ->
        format(string(MemoDecl), 'my %~w_memo;', [PredStr]),
        format(string(MemoCheck), '    return $~w_memo{$n} if exists $~w_memo{$n};', [PredStr, PredStr]),
        format(string(MemoStore), '    $~w_memo{$n} = $result;', [PredStr])
    ;   MemoDecl = '# Memoization disabled',
        MemoCheck = '',
        MemoStore = ''
    ),
    format(string(Code),
'#!/usr/bin/env perl
# Generated by UnifyWeaver Perl Target - Linear Recursion (generic, multifile dispatch)
# Predicate: ~w/~w
use strict;
use warnings;

~w

sub ~w {
    my ($n) = @_;
~w
    return 0 if $n <= 0;
    return 1 if $n == 1;
    my $result = ~w($n - 1) + $n;
~w
    return $result;
}

print ~w($ARGV[0]), "\\n" if @ARGV;
', [PredStr, Arity, MemoDecl, PredStr, MemoCheck, PredStr, MemoStore, PredStr]).

%% translate_fold_expr_perl(+PrologExpr, +InputVar, +AccVar, -PerlExpr)
%  Used in linear recursion (reduce context) where Perl's $a/$b are the
%  special accumulator variables. Tail recursion uses expr_to_perl instead,
%  which maps to $acc/$item for the loop context.
translate_fold_expr_perl(A * B, InputVar, AccVar, Expr) :-
    translate_perl_reduce_term(A, InputVar, AccVar, AT),
    translate_perl_reduce_term(B, InputVar, AccVar, BT),
    format(string(Expr), '~w * ~w', [AT, BT]).
translate_fold_expr_perl(A + B, InputVar, AccVar, Expr) :-
    translate_perl_reduce_term(A, InputVar, AccVar, AT),
    translate_perl_reduce_term(B, InputVar, AccVar, BT),
    format(string(Expr), '~w + ~w', [AT, BT]).
translate_fold_expr_perl(A - B, InputVar, AccVar, Expr) :-
    translate_perl_reduce_term(A, InputVar, AccVar, AT),
    translate_perl_reduce_term(B, InputVar, AccVar, BT),
    format(string(Expr), '~w - ~w', [AT, BT]).
translate_fold_expr_perl(Term, InputVar, AccVar, Expr) :-
    translate_perl_reduce_term(Term, InputVar, AccVar, Expr).

% Maps Prolog variables to Perl's reduce special vars ($a=accumulator, $b=current)
translate_perl_reduce_term(Term, InputVar, _AccVar, '$a') :- Term == InputVar, !.
translate_perl_reduce_term(Term, _InputVar, AccVar, '$b') :- Term == AccVar, !.
translate_perl_reduce_term(Number, _, _, PerlTerm) :- integer(Number), !,
    format(string(PerlTerm), '~w', [Number]).
translate_perl_reduce_term(Atom, _, _, PerlTerm) :-
    format(string(PerlTerm), '~w', [Atom]).

%% translate_list_fold_perl(+PrologExpr, +HeadVar, +AccVar, -PerlExpr)
%  Like translate_fold_expr_perl but maps HeadVar → '$b' (current element in reduce)
%  and AccVar → '$a' (accumulator in reduce). Perl's reduce uses $a/$b.
translate_list_fold_perl(A * B, HeadVar, AccVar, Expr) :-
    translate_list_term_perl(A, HeadVar, AccVar, AT),
    translate_list_term_perl(B, HeadVar, AccVar, BT),
    format(string(Expr), '~w * ~w', [AT, BT]).
translate_list_fold_perl(A + B, HeadVar, AccVar, Expr) :-
    translate_list_term_perl(A, HeadVar, AccVar, AT),
    translate_list_term_perl(B, HeadVar, AccVar, BT),
    format(string(Expr), '~w + ~w', [AT, BT]).
translate_list_fold_perl(A - B, HeadVar, AccVar, Expr) :-
    translate_list_term_perl(A, HeadVar, AccVar, AT),
    translate_list_term_perl(B, HeadVar, AccVar, BT),
    format(string(Expr), '~w - ~w', [AT, BT]).
translate_list_fold_perl(Term, HeadVar, AccVar, Expr) :-
    translate_list_term_perl(Term, HeadVar, AccVar, Expr).

translate_list_term_perl(Term, HeadVar, _AccVar, '$b') :- Term == HeadVar, !.
translate_list_term_perl(Term, _HeadVar, AccVar, '$a') :- Term == AccVar, !.
translate_list_term_perl(Number, _, _, PerlTerm) :- integer(Number), !,
    format(string(PerlTerm), '~w', [Number]).
translate_list_term_perl(Atom, _, _, PerlTerm) :-
    format(string(PerlTerm), '~w', [Atom]).

% ============================================================================
% TREE RECURSION - Perl target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/tree_recursion').
:- multifile tree_recursion:compile_tree_pattern/6.

tree_recursion:compile_tree_pattern(perl, _Pattern, Pred, _Arity, _UseMemo, PerlCode) :-
    atom_string(Pred, PredStr),
    format(string(PerlCode),
'#!/usr/bin/env perl
# Generated by UnifyWeaver Perl Target - Tree Recursion (multifile dispatch)
use strict;
use warnings;

my %~w_memo;

sub ~w {
    my ($n) = @_;
    return $~w_memo{$n} if exists $~w_memo{$n};
    return 0 if $n <= 0;
    return 1 if $n == 1;
    my $result = ~w($n - 1) + ~w($n - 2);
    $~w_memo{$n} = $result;
    return $result;
}

print ~w($ARGV[0]), "\\n" if @ARGV;
', [PredStr, PredStr, PredStr, PredStr, PredStr, PredStr, PredStr, PredStr]).

% ============================================================================
% MULTICALL LINEAR RECURSION - Perl target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/multicall_linear_recursion').
:- multifile multicall_linear_recursion:compile_multicall_pattern/6.

multicall_linear_recursion:compile_multicall_pattern(perl, PredStr, BaseClauses, _RecClauses, _MemoEnabled, PerlCode) :-
    findall(BaseCaseCode, (
        member(clause(BHead, _), BaseClauses),
        BHead =.. [_P, BInput, BOutput],
        format(string(BaseCaseCode), '    return ~w if $n == ~w;', [BOutput, BInput])
    ), BaseCaseCodes0),
    sort(BaseCaseCodes0, BaseCaseCodes),
    atomic_list_concat(BaseCaseCodes, '\n', BaseCaseStr),
    format(string(PerlCode),
'#!/usr/bin/env perl
# Generated by UnifyWeaver Perl Target - Multicall Linear Recursion (multifile dispatch)
use strict;
use warnings;

my %~w_memo;

sub ~w {
    my ($n) = @_;
    return $~w_memo{$n} if exists $~w_memo{$n};
~w
    my $result = ~w($n - 1) + ~w($n - 2);
    $~w_memo{$n} = $result;
    return $result;
}

print ~w($ARGV[0]), "\\n" if @ARGV;
', [PredStr, PredStr, PredStr, PredStr, BaseCaseStr, PredStr, PredStr, PredStr, PredStr]).

% ============================================================================
% DIRECT MULTICALL RECURSION - Perl target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/direct_multi_call_recursion').
:- multifile direct_multi_call_recursion:compile_direct_multicall_pattern/5.

direct_multi_call_recursion:compile_direct_multicall_pattern(perl, PredStr, BaseClauses, _RecClause, PerlCode) :-
    findall(BaseCaseCode, (
        member(clause(BHead, _), BaseClauses),
        BHead =.. [_P, BInput, BOutput],
        format(string(BaseCaseCode), '    $~w_memo{~w} = ~w; return ~w if $n == ~w;', [PredStr, BInput, BOutput, BOutput, BInput])
    ), BaseCaseCodes0),
    sort(BaseCaseCodes0, BaseCaseCodes),
    atomic_list_concat(BaseCaseCodes, '\n', BaseCaseStr),
    format(string(PerlCode),
'#!/usr/bin/env perl
# Generated by UnifyWeaver Perl Target - Direct Multicall Recursion (multifile dispatch)
use strict;
use warnings;

my %~w_memo;

sub ~w {
    my ($n) = @_;
    return $~w_memo{$n} if exists $~w_memo{$n};
~w
    my $result = ~w($n - 1) + ~w($n - 2);
    $~w_memo{$n} = $result;
    return $result;
}

print ~w($ARGV[0]), "\\n" if @ARGV;
', [PredStr, PredStr, PredStr, PredStr, BaseCaseStr, PredStr, PredStr, PredStr, PredStr]).

% ============================================================================
% MUTUAL RECURSION - Perl target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/mutual_recursion').
:- use_module('../core/advanced/pattern_matchers', [is_per_path_visited_pattern/4]).
:- multifile mutual_recursion:compile_mutual_pattern/5.

mutual_recursion:compile_mutual_pattern(perl, Predicates, MemoEnabled, _MemoStrategy, PerlCode) :-
    mutual_functions_perl(Predicates, Predicates, MemoEnabled, FuncCodes),
    atomic_list_concat(FuncCodes, '\n\n', FunctionsCode),
    mutual_dispatch_perl(Predicates, DispatchCode),
    format(string(PerlCode),
'#!/usr/bin/env perl
# Generated by UnifyWeaver Perl Target - Mutual Recursion (multifile dispatch)
use strict;
use warnings;

my %mutual_memo;

~w

if (@ARGV >= 2) {
    my $func = $ARGV[0];
    my $n = $ARGV[1];
~w
}
', [FunctionsCode, DispatchCode]).

mutual_functions_perl([], _AllPreds, _MemoEnabled, []).
mutual_functions_perl([Pred/Arity|Rest], AllPreds, MemoEnabled, [FuncCode|RestCodes]) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),
    partition(mutual_recursion:is_mutual_recursive_clause(AllPreds), Clauses, RecClauses, BaseClauses),
    findall(BaseLine, (
        member(clause(BHead, true), BaseClauses),
        BHead =.. [_P, BValue],
        format(string(BaseLine), '    return 1 if $n == ~w;', [BValue])
    ), BaseLines),
    atomic_list_concat(BaseLines, '\n', BaseCode),
    (   RecClauses = [clause(_RHead, RBody)|_] ->
        extract_mutual_rec_info_perl(RBody, Guard, CalledPred, Step),
        atom_string(CalledPred, CalledStr),
        (   Guard = (N > Threshold), var(N) ->
            (   MemoEnabled = true ->
                format(string(RecCode),
'    if ($n > ~w) {
        my $key = "~w:$n";
        return $mutual_memo{$key} if exists $mutual_memo{$key};
        my $result = ~w($n ~w);
        $mutual_memo{$key} = $result;
        return $result;
    }
    return 0;', [Threshold, PredStr, CalledStr, Step])
            ;   format(string(RecCode),
'    return $n > ~w ? ~w($n ~w) : 0;', [Threshold, CalledStr, Step])
            )
        ;   format(string(RecCode), '    return ~w($n ~w);', [CalledStr, Step])
        )
    ;   RecCode = '    return 0;'
    ),
    format(string(FuncCode),
'sub ~w {
    my ($n) = @_;
~w
~w
}', [PredStr, BaseCode, RecCode]),
    mutual_functions_perl(Rest, AllPreds, MemoEnabled, RestCodes).

mutual_dispatch_perl(Predicates, Code) :-
    findall(DispatchLine, (
        member(Pred/_Arity, Predicates),
        atom_string(Pred, PredStr),
        format(string(DispatchLine), '    print ~w($n), "\\n" if $func eq "~w";', [PredStr, PredStr])
    ), Lines),
    atomic_list_concat(Lines, '\n', Code).

extract_mutual_rec_info_perl(Body, Guard, CalledPred, Step) :-
    extract_goals_perl(Body, Goals),
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

extract_goals_perl((A, B), Goals) :- !,
    extract_goals_perl(A, GA),
    extract_goals_perl(B, GB),
    append(GA, GB, Goals).
extract_goals_perl(true, []) :- !.
extract_goals_perl(Goal, [Goal]).

% ============================================================================
% GENERAL RECURSIVE PATTERN (visited-set cycle detection)
% ============================================================================

:- multifile advanced_recursive_compiler:compile_general_recursive_pattern/6.

%% No-visited-pattern — plain recursive without cycle detection
advanced_recursive_compiler:compile_general_recursive_pattern(perl, PredStr, Arity, BaseClauses, RecClauses, Code) :-
    atom_string(Pred, PredStr),
    append(BaseClauses, RecClauses, AllClauses),
    \+ is_per_path_visited_pattern(Pred, Arity, AllClauses, _),
    !,
    (   BaseClauses = [(BH, true)|_]
    ->  BH =.. [_|BaseArgs], last(BaseArgs, BaseVal),
        BaseArgs = [BaseKey|_],
        format(string(BaseCheck), '    return ("~w") if $arg1 eq "~w";', [BaseVal, BaseKey])
    ;   BaseCheck = '    # no base case extracted'
    ),
    format(string(Code),
'# General recursive: ~w (plain, no visited pattern)\n\c
sub ~w {\n\c
    my ($arg1) = @_;\n\c
~w\n\c
    return ~w($arg1);\n\c
}\n',
    [PredStr, PredStr, BaseCheck, PredStr]).

%% Arity-2: wrapper + worker with base case check and recursive accumulation
advanced_recursive_compiler:compile_general_recursive_pattern(perl, PredStr, 2, BaseClauses, RecClauses, Code) :-
    atom_string(PredAtom, PredStr),
    atom_concat(PredAtom, '_worker', WorkerAtom),
    atom_string(WorkerAtom, WorkerStr),
    %% Extract base case key/value from first base clause
    (   BaseClauses = [(BH, true)|_]
    ->  BH =.. [_, BaseKey, BaseVal],
        format(string(BaseCheck),
            '    return ("~w") if $arg1 eq "~w";', [BaseVal, BaseKey])
    ;   BaseCheck = '    # no base case extracted'
    ),
    %% Extract recursive step from first recursive clause
    (   RecClauses = [(_, RecBody)|_]
    ->  extract_rec_call_perl(RecBody, PredStr, WorkerStr, RecCallExpr)
    ;   format(string(RecCallExpr), '~w($arg1, $visited)', [WorkerStr])
    ),
    format(string(Code),
'# General recursive: ~w (with cycle detection)\n\c
sub ~w {\n\c
    my ($arg1) = @_;\n\c
    return ~w($arg1, {});\n\c
}\n\c
\n\c
sub ~w {\n\c
    my ($arg1, $visited) = @_;\n\c
    return () if $visited->{$arg1};\n\c
    $visited->{$arg1} = 1;\n\c
~w\n\c
    my @sub = ~w;\n\c
    return @sub;\n\c
}\n',
    [PredStr, PredStr, WorkerStr, WorkerStr, BaseCheck, RecCallExpr]).

%% Arity-3: wrapper + worker with counter/output style
advanced_recursive_compiler:compile_general_recursive_pattern(perl, PredStr, 3, BaseClauses, RecClauses, Code) :-
    atom_string(PredAtom, PredStr),
    atom_concat(PredAtom, '_worker', WorkerAtom),
    atom_string(WorkerAtom, WorkerStr),
    (   BaseClauses = [(BH, true)|_]
    ->  BH =.. [_, BaseKey, _, BaseVal],
        format(string(BaseCheck),
            '    return ("~w") if $arg1 eq "~w";', [BaseVal, BaseKey])
    ;   BaseCheck = '    # no base case extracted'
    ),
    (   RecClauses = [(_, RecBody)|_]
    ->  extract_rec_call_perl(RecBody, PredStr, WorkerStr, RecCallExpr)
    ;   format(string(RecCallExpr), '~w($arg1, $visited)', [WorkerStr])
    ),
    format(string(Code),
'# General recursive: ~w (with cycle detection)\n\c
sub ~w {\n\c
    my ($arg1) = @_;\n\c
    return ~w($arg1, {});\n\c
}\n\c
\n\c
sub ~w {\n\c
    my ($arg1, $visited) = @_;\n\c
    return () if $visited->{$arg1};\n\c
    $visited->{$arg1} = 1;\n\c
~w\n\c
    return ~w;\n\c
}\n',
    [PredStr, PredStr, WorkerStr, WorkerStr, BaseCheck, RecCallExpr]).

extract_rec_call_perl((A, _), PredStr, WorkerStr, Expr) :-
    nonvar(A),
    functor(A, Pred, _),
    atom_string(Pred, PredStr), !,
    A =.. [_|CallArgs],
    (   CallArgs = [Arg1|_]
    ->  format(string(Expr), '~w(~w, $visited)', [WorkerStr, Arg1])
    ;   format(string(Expr), '~w($arg1, $visited)', [WorkerStr])
    ).
extract_rec_call_perl((_, B), PredStr, WorkerStr, Expr) :- !,
    extract_rec_call_perl(B, PredStr, WorkerStr, Expr).
extract_rec_call_perl(Goal, PredStr, WorkerStr, Expr) :-
    nonvar(Goal),
    functor(Goal, Pred, _),
    atom_string(Pred, PredStr), !,
    Goal =.. [_|CallArgs],
    (   CallArgs = [Arg1|_]
    ->  format(string(Expr), '~w(~w, $visited)', [WorkerStr, Arg1])
    ;   format(string(Expr), '~w($arg1, $visited)', [WorkerStr])
    ).
extract_rec_call_perl(_, _PredStr, WorkerStr, Expr) :-
    format(string(Expr), '~w($arg1, $visited)', [WorkerStr]).
