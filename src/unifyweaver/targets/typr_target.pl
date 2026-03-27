:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% typr_target.pl - TypR code generation target

:- module(typr_target, [
    target_info/1,
    compile_predicate/3,
    compile_predicate_to_typr/3,
    compile_recursive_predicate_to_typr/3,
    generated_typr_is_valid/2
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(readutil)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../core/template_system').
:- use_module('../core/binding_registry', [binding/6]).
:- use_module('../core/advanced/pattern_matchers', [
    is_tail_recursive_accumulator/2,
    split_body_at_recursive_call/5
]).
:- use_module('../core/advanced/mutual_recursion', [
    is_mutual_recursive_clause/2,
    parse_recursive_body/5
]).
:- use_module('../core/advanced/linear_recursion', [
    extract_step_info/4
]).
:- use_module('r_target', [compile_predicate_to_r/3, init_r_target/0, infer_clauses_return_type/2]).
:- use_module('type_declarations').

:- multifile mutual_recursion:compile_mutual_pattern/5.

target_info(info{
    name: "TypR",
    family: r,
    file_extension: ".typr",
    runtime: typr,
    features: [types, gradual_typing, s3, transpiles_to_r],
    recursion_patterns: [transitive_closure, tail_recursion, linear_recursion, tree_recursion, mutual_recursion],
    compile_command: "typr"
}).

compile_predicate(PredArity, Options, Code) :-
    compile_predicate_to_typr(PredArity, Options, Code).

compile_predicate_to_typr(PredIndicator, Options, Code) :-
    pred_indicator_parts(PredIndicator, Module, Pred, Arity),
    option(global_typed_mode(GlobalMode), Options, infer),
    resolve_typed_mode(Pred/Arity, Options, GlobalMode, TypedMode),
    (   option(base_pred(BasePredOption), Options)
    ->  normalize_base_pred(BasePredOption, BasePred),
        compile_typr_transitive_closure(Module, Pred/Arity, BasePred, TypedMode, Code)
    ;   detect_transitive_closure(Module, Pred, Arity, BasePred)
    ->  compile_typr_transitive_closure(Module, Pred/Arity, BasePred, TypedMode, Code)
    ;   compile_generic_typr(Module, Pred/Arity, Options, TypedMode, Code)
    ),
    finalize_type_diagnostics_report(Options).

compile_recursive_predicate_to_typr(PredIndicator, Options, Code) :-
    pred_indicator_parts(PredIndicator, Module, Pred, Arity),
    option(global_typed_mode(GlobalMode), Options, infer),
    resolve_typed_mode(Pred/Arity, Options, GlobalMode, TypedMode),
    findall(Head-Body, predicate_clause(Module, Pred, Arity, Head, Body), Clauses),
    (   compile_typr_tail_recursive_accumulator(Module:Pred/Arity, TypedMode, Clauses, Code)
    ;   compile_typr_linear_recursive_numeric(Module:Pred/Arity, TypedMode, Clauses, Code)
    ;   compile_typr_linear_recursive_list(Module:Pred/Arity, TypedMode, Clauses, Code)
    ;   compile_typr_tree_recursive_structural(Module:Pred/Arity, TypedMode, Clauses, Code)
    ;   compile_typr_tree_recursive_numeric(Module:Pred/Arity, TypedMode, Clauses, Code)
    ),
    finalize_type_diagnostics_report(Options).

mutual_recursion:compile_mutual_pattern(typr, Predicates, MemoEnabled, _MemoStrategy, Code) :-
    compile_typr_mutual_recursion_group(Predicates, MemoEnabled, Code).

compile_typr_mutual_recursion_group(Predicates0, MemoEnabled, Code) :-
    sort(Predicates0, Predicates),
    (   maplist(typr_mutual_numeric_spec(Predicates), Predicates, Specs)
    ;   maplist(typr_mutual_list_spec(Predicates), Predicates, Specs)
    ;   maplist(typr_mutual_tree_dual_value_full_body_spec(Predicates), Predicates, Specs)
    ;   maplist(typr_mutual_tree_dual_value_branch_spec(Predicates), Predicates, Specs)
    ;   maplist(typr_mutual_tree_dual_value_spec(Predicates), Predicates, Specs)
    ;   maplist(typr_mutual_tree_dual_body_spec(Predicates), Predicates, Specs)
    ;   maplist(typr_mutual_tree_dual_context_spec(Predicates), Predicates, Specs)
    ;   maplist(typr_mutual_tree_dual_branch_spec(Predicates), Predicates, Specs)
    ;   maplist(typr_mutual_tree_dual_spec(Predicates), Predicates, Specs)
    ;   maplist(typr_mutual_tree_spec(Predicates), Predicates, Specs)
    ),
    typr_mutual_group_code(Specs, MemoEnabled, Code).

typr_mutual_numeric_spec(GroupPredicates, Pred/Arity, Spec) :-
    Arity =:= 1,
    findall(Head-Body, predicate_clause(user, Pred, Arity, Head, Body), Clauses),
    Clauses \= [],
    generic_typr_return_type(Pred/Arity, Clauses, "bool"),
    build_typed_arg_list(Pred/Arity, none, Arity, explicit, TypedArgList),
    findall(clause(Head, Body), predicate_clause(user, Pred, Arity, Head, Body), ClauseTerms),
    partition(typr_is_mutual_recursive_clause(GroupPredicates), ClauseTerms, RecClauses, BaseClauses),
    RecClauses = [clause(RecHead, RecBody)],
    BaseClauses \= [],
    maplist(typr_mutual_base_case_value, BaseClauses, BaseValues),
    RecHead =.. [_PredName, HeadVar],
    parse_recursive_body(RecBody, HeadVar, Conditions, Computations, RecCall),
    RecCall \= none,
    RecCall =.. [NextPred, RecArg],
    memberchk(NextPred/1, GroupPredicates),
    typr_mutual_guard_expr(HeadVar, Conditions, GuardExpr),
    typr_mutual_step_expr(HeadVar, Computations, RecArg, StepExpr),
    atom_string(Pred, PredStr),
    atom_string(NextPred, NextPredStr),
    format(string(HelperName), '~w_impl', [PredStr]),
    format(string(NextHelperName), '~w_impl', [NextPredStr]),
    Spec = mutual_spec{
        kind: numeric,
        pred: Pred,
        arity: Arity,
        pred_str: PredStr,
        typed_arg_list: TypedArgList,
        return_type: "bool",
        helper_name: HelperName,
        next_helper_name: NextHelperName,
        base_values: BaseValues,
        guard_expr: GuardExpr,
        step_expr: StepExpr
    }.

typr_mutual_list_spec(GroupPredicates, Pred/Arity, Spec) :-
    Arity =:= 1,
    findall(Head-Body, predicate_clause(user, Pred, Arity, Head, Body), Clauses),
    Clauses \= [],
    generic_typr_return_type(Pred/Arity, Clauses, "bool"),
    build_typed_arg_list(Pred/Arity, none, Arity, explicit, TypedArgList),
    findall(clause(Head, Body), predicate_clause(user, Pred, Arity, Head, Body), ClauseTerms),
    partition(typr_is_mutual_recursive_clause(GroupPredicates), ClauseTerms, RecClauses, BaseClauses),
    RecClauses = [clause(RecHead, RecBody)],
    BaseClauses \= [],
    maplist(typr_mutual_list_base_case_length, BaseClauses, BaseLengths0),
    sort(BaseLengths0, BaseLengths),
    RecHead =.. [_PredName, RecInputPattern],
    RecInputPattern = [HeadVar|TailVar],
    var(TailVar),
    parse_recursive_body(RecBody, HeadVar, Conditions, Computations, RecCall),
    Conditions == [],
    Computations == [],
    RecCall \= none,
    RecCall =.. [NextPred, RecArg],
    memberchk(NextPred/1, GroupPredicates),
    RecArg == TailVar,
    atom_string(Pred, PredStr),
    atom_string(NextPred, NextPredStr),
    format(string(HelperName), '~w_impl', [PredStr]),
    format(string(NextHelperName), '~w_impl', [NextPredStr]),
    Spec = mutual_spec{
        kind: list,
        pred: Pred,
        arity: Arity,
        pred_str: PredStr,
        typed_arg_list: TypedArgList,
        return_type: "bool",
        helper_name: HelperName,
        next_helper_name: NextHelperName,
        base_lengths: BaseLengths,
        guard_expr: 'length(current_input) > 0',
        step_expr: 'tail(current_input, -1)'
    }.

typr_mutual_base_case_value(clause(Head, true), BaseExpr) :-
    Head =.. [_Pred, BaseValue],
    integer(BaseValue),
    typr_translate_r_expr(BaseValue, [], BaseExpr).

typr_mutual_list_base_case_length(clause(Head, true), 0) :-
    Head =.. [_Pred, []].
typr_mutual_list_base_case_length(clause(Head, true), 1) :-
    Head =.. [_Pred, [Elem]],
    var(Elem).

typr_mutual_tree_dual_value_spec(GroupPredicates, Pred/Arity, Spec) :-
    Arity >= 2,
    findall(Head-Body, predicate_clause(user, Pred, Arity, Head, Body), Clauses),
    Clauses \= [],
    generic_typr_return_type(Pred/Arity, Clauses, "int"),
    build_typed_arg_list(Pred/Arity, none, Arity, explicit, TypedArgList),
    findall(clause(Head, Body), predicate_clause(user, Pred, Arity, Head, Body), ClauseTerms),
    partition(is_mutual_recursive_clause(GroupPredicates), ClauseTerms, RecClauses, BaseClauses),
    RecClauses = [clause(RecHead, RecBody)],
    BaseClauses \= [],
    RecHead =.. [_PredName, RecInputPattern|ContextAndOutputVars],
    append(ContextVars, [OutputVar], ContextAndOutputVars),
    var(OutputVar),
    RecInputPattern = [ValueVar, LeftVar, RightVar],
    typr_mutual_tree_context_param_names(ContextVars, ContextParamNames),
    maplist(typr_mutual_tree_value_base_case(ContextParamNames), BaseClauses, BaseCases),
    typr_mutual_goal_list(RecBody, Goals0),
    maplist(typr_strip_module_goal, Goals0, Goals),
    split_typr_mutual_multicall_goals(GroupPredicates, Goals, PreGoals, RecGoals, PostGoals),
    typr_mutual_tree_alias_map([], LeftVar, RightVar, AliasMap0),
    typr_mutual_tree_context_fields(Pred, ContextVars, HelperName, ExtraArgMap0, HelperParams, WrapperCallArgs, MemoKeyExpr),
    typr_mutual_tree_branch_prework(PreGoals, ValueVar, AliasMap0, ExtraArgMap0, AliasMap, ExtraArgMap, PreGuardExpr),
    maplist(typr_mutual_tree_value_recursive_goal(GroupPredicates, AliasMap, ExtraArgMap), RecGoals, CallSpecs0),
    select(value_call_spec(left, LeftNextPred, LeftCallArgs, LeftOutputVar), CallSpecs0, CallSpecs1),
    select(value_call_spec(right, RightNextPred, RightCallArgs, RightOutputVar), CallSpecs1, []),
    typr_mutual_tree_top_guard_expr(PreGuardExpr, GuardExpr),
    typr_mutual_tree_condition_varmap(ValueVar, AliasMap, ExtraArgMap, ResultVarMap0),
    update_typr_expr_varmap(ResultVarMap0, LeftOutputVar, left_result, ResultVarMap1),
    update_typr_expr_varmap(ResultVarMap1, RightOutputVar, right_result, ResultVarMap),
    typr_goals_to_body(PostGoals, PostBody),
    linear_recursive_output_expr(PostBody, OutputVar, ResultVarMap, ResultExpr),
    atom_string(Pred, PredStr),
    atom_string(LeftNextPred, LeftNextPredStr),
    atom_string(RightNextPred, RightNextPredStr),
    format(string(LeftNextHelperName), '~w_impl', [LeftNextPredStr]),
    format(string(RightNextHelperName), '~w_impl', [RightNextPredStr]),
    Spec = mutual_spec{
        kind: tree_dual_value,
        pred: Pred,
        arity: Arity,
        pred_str: PredStr,
        typed_arg_list: TypedArgList,
        return_type: "int",
        helper_name: HelperName,
        helper_params: HelperParams,
        wrapper_call_args: WrapperCallArgs,
        memo_key_expr: MemoKeyExpr,
        base_cases: BaseCases,
        guard_expr: GuardExpr,
        left_call: branch_call(LeftNextHelperName, LeftCallArgs),
        right_call: branch_call(RightNextHelperName, RightCallArgs),
        result_expr: ResultExpr
    }.

typr_mutual_tree_value_base_case(ContextParamNames, clause(Head, true), base_case(BaseCondition, OutputExpr)) :-
    typr_mutual_tree_base_case_condition(clause(Head, true), BaseCondition),
    Head =.. [_Pred, BaseTree|ContextAndOutputVars],
    append(ContextVars, [BaseOutput], ContextAndOutputVars),
    typr_mutual_tree_base_case_varmap(BaseTree, ContextVars, ContextParamNames, VarMap),
    typr_translate_r_expr(BaseOutput, VarMap, OutputExpr).

typr_mutual_tree_base_case_varmap(BaseTree, ContextVars, ContextParamNames, VarMap) :-
    typr_mutual_tree_context_varmap(ContextVars, ContextParamNames, ContextVarMap),
    typr_mutual_tree_base_tree_varmap(BaseTree, TreeVarMap),
    append(TreeVarMap, ContextVarMap, VarMap).

typr_mutual_tree_base_tree_varmap([], []) :-
    !.
typr_mutual_tree_base_tree_varmap([ValueVar, [], []], [ValueVar-'.subset2(current_input, 1)']) :-
    var(ValueVar),
    !.
typr_mutual_tree_base_tree_varmap([_Value, [], []], []).

typr_mutual_tree_dual_value_full_body_spec(GroupPredicates, Pred/Arity, Spec) :-
    Arity >= 2,
    findall(Head-Body, predicate_clause(user, Pred, Arity, Head, Body), Clauses),
    Clauses \= [],
    generic_typr_return_type(Pred/Arity, Clauses, "int"),
    build_typed_arg_list(Pred/Arity, none, Arity, explicit, TypedArgList),
    findall(clause(Head, Body), predicate_clause(user, Pred, Arity, Head, Body), ClauseTerms),
    partition(typr_is_mutual_recursive_clause(GroupPredicates), ClauseTerms, RecClauses, BaseClauses),
    RecClauses = [clause(RecHead, RecBody)],
    BaseClauses \= [],
    RecHead =.. [_PredName, RecInputPattern|ContextAndOutputVars],
    append(ContextVars, [OutputVar], ContextAndOutputVars),
    var(OutputVar),
    RecInputPattern = [ValueVar, LeftVar, RightVar],
    typr_mutual_tree_context_param_names(ContextVars, ContextParamNames),
    maplist(typr_mutual_tree_value_base_case(ContextParamNames), BaseClauses, BaseCases),
    typr_mutual_goal_list(RecBody, Goals0),
    maplist(typr_strip_module_goal, Goals0, Goals),
    typr_mutual_tree_alias_map([], LeftVar, RightVar, AliasMap0),
    typr_mutual_tree_context_fields(
        Pred,
        ContextVars,
        HelperName,
        ExtraArgMap0,
        HelperParams,
        WrapperCallArgs,
        MemoKeyExpr
    ),
    typr_mutual_tree_value_branch_body(
        GroupPredicates,
        Goals,
        ValueVar,
        AliasMap0,
        ExtraArgMap0,
        true,
        OutputVar,
        Body
    ),
    atom_string(Pred, PredStr),
    Spec = mutual_spec{
        kind: tree_dual_value_full_body,
        pred: Pred,
        arity: Arity,
        pred_str: PredStr,
        typed_arg_list: TypedArgList,
        return_type: "int",
        helper_name: HelperName,
        helper_params: HelperParams,
        wrapper_call_args: WrapperCallArgs,
        memo_key_expr: MemoKeyExpr,
        base_cases: BaseCases,
        guard_expr: 'length(current_input) == 3',
        body: Body
    }.

typr_mutual_tree_dual_value_branch_spec(GroupPredicates, Pred/Arity, Spec) :-
    Arity >= 2,
    findall(Head-Body, predicate_clause(user, Pred, Arity, Head, Body), Clauses),
    Clauses \= [],
    generic_typr_return_type(Pred/Arity, Clauses, "int"),
    build_typed_arg_list(Pred/Arity, none, Arity, explicit, TypedArgList),
    findall(clause(Head, Body), predicate_clause(user, Pred, Arity, Head, Body), ClauseTerms),
    partition(typr_is_mutual_recursive_clause(GroupPredicates), ClauseTerms, RecClauses, BaseClauses),
    RecClauses = [clause(RecHead, RecBody)],
    BaseClauses \= [],
    RecHead =.. [_PredName, RecInputPattern|ContextAndOutputVars],
    append(ContextVars, [OutputVar], ContextAndOutputVars),
    var(OutputVar),
    RecInputPattern = [ValueVar, LeftVar, RightVar],
    typr_mutual_tree_context_param_names(ContextVars, ContextParamNames),
    maplist(typr_mutual_tree_value_base_case(ContextParamNames), BaseClauses, BaseCases),
    typr_mutual_goal_list(RecBody, Goals0),
    maplist(typr_strip_module_goal, Goals0, Goals),
    append([BranchGoal, GoalA, GoalB], PostGoals, Goals),
    PostGoals \= [],
    typr_if_then_else_goal(BranchGoal, IfGoal, ThenGoal0, ElseGoal0),
    typr_mutual_goal_list(ThenGoal0, ThenGoals0),
    maplist(typr_strip_module_goal, ThenGoals0, ThenGoals),
    maplist(typr_mutual_tree_alias_goal, ThenGoals),
    typr_mutual_goal_list(ElseGoal0, ElseGoals0),
    maplist(typr_strip_module_goal, ElseGoals0, ElseGoals),
    maplist(typr_mutual_tree_alias_goal, ElseGoals),
    typr_mutual_tree_alias_map([], LeftVar, RightVar, AliasMap0),
    typr_mutual_tree_context_fields(Pred, ContextVars, HelperName, ExtraArgMap0, HelperParams, WrapperCallArgs, MemoKeyExpr),
    typr_mutual_tree_alias_map(ThenGoals, LeftVar, RightVar, ThenAliasMap),
    typr_mutual_tree_alias_map(ElseGoals, LeftVar, RightVar, ElseAliasMap),
    maplist(
        typr_mutual_tree_value_recursive_goal(GroupPredicates, ThenAliasMap, ExtraArgMap0),
        [GoalA, GoalB],
        ThenGoalSpecs0
    ),
    maplist(
        typr_mutual_tree_value_recursive_goal(GroupPredicates, ElseAliasMap, ExtraArgMap0),
        [GoalA, GoalB],
        ElseGoalSpecs0
    ),
    typr_mutual_tree_value_side_calls(ThenGoalSpecs0, ThenLeftCall, ThenRightCall, ThenLeftOutputVar, ThenRightOutputVar),
    typr_mutual_tree_value_side_calls(ElseGoalSpecs0, ElseLeftCall, ElseRightCall, ElseLeftOutputVar, ElseRightOutputVar),
    typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap0, ExtraArgMap0, BranchConditionExpr),
    typr_goals_to_body(PostGoals, PostBody),
    typr_mutual_tree_condition_varmap(ValueVar, AliasMap0, ExtraArgMap0, ResultVarMap0),
    update_typr_expr_varmap(ResultVarMap0, ThenLeftOutputVar, left_result, ThenResultVarMap1),
    update_typr_expr_varmap(ThenResultVarMap1, ThenRightOutputVar, right_result, ThenResultVarMap),
    update_typr_expr_varmap(ResultVarMap0, ElseLeftOutputVar, left_result, ElseResultVarMap1),
    update_typr_expr_varmap(ElseResultVarMap1, ElseRightOutputVar, right_result, ElseResultVarMap),
    linear_recursive_output_expr(PostBody, OutputVar, ThenResultVarMap, ThenResultExpr),
    linear_recursive_output_expr(PostBody, OutputVar, ElseResultVarMap, ElseResultExpr),
    atom_string(Pred, PredStr),
    Spec = mutual_spec{
        kind: tree_dual_value_branch,
        pred: Pred,
        arity: Arity,
        pred_str: PredStr,
        typed_arg_list: TypedArgList,
        return_type: "int",
        helper_name: HelperName,
        helper_params: HelperParams,
        wrapper_call_args: WrapperCallArgs,
        memo_key_expr: MemoKeyExpr,
        base_cases: BaseCases,
        guard_expr: 'length(current_input) == 3',
        branch_condition_expr: BranchConditionExpr,
        then_left_call: ThenLeftCall,
        then_right_call: ThenRightCall,
        else_left_call: ElseLeftCall,
        else_right_call: ElseRightCall,
        then_result_expr: ThenResultExpr,
        else_result_expr: ElseResultExpr
    }.

typr_mutual_tree_dual_value_branch_spec(GroupPredicates, Pred/Arity, Spec) :-
    Arity >= 2,
    findall(Head-Body, predicate_clause(user, Pred, Arity, Head, Body), Clauses),
    Clauses \= [],
    generic_typr_return_type(Pred/Arity, Clauses, "int"),
    build_typed_arg_list(Pred/Arity, none, Arity, explicit, TypedArgList),
    findall(clause(Head, Body), predicate_clause(user, Pred, Arity, Head, Body), ClauseTerms),
    partition(typr_is_mutual_recursive_clause(GroupPredicates), ClauseTerms, RecClauses, BaseClauses),
    RecClauses = [clause(RecHead, RecBody)],
    BaseClauses \= [],
    RecHead =.. [_PredName, RecInputPattern|ContextAndOutputVars],
    append(ContextVars, [OutputVar], ContextAndOutputVars),
    var(OutputVar),
    RecInputPattern = [ValueVar, LeftVar, RightVar],
    typr_mutual_tree_context_param_names(ContextVars, ContextParamNames),
    maplist(typr_mutual_tree_value_base_case(ContextParamNames), BaseClauses, BaseCases),
    typr_mutual_goal_list(RecBody, Goals0),
    maplist(typr_strip_module_goal, Goals0, Goals),
    Goals = [BranchGoal|PostGoals],
    PostGoals \= [],
    typr_if_then_else_goal(BranchGoal, IfGoal, ThenGoal0, ElseGoal0),
    typr_goals_to_body(PostGoals, PostBody),
    typr_mutual_goal_list(ThenGoal0, ThenGoals0),
    maplist(typr_strip_module_goal, ThenGoals0, ThenGoals),
    typr_mutual_goal_list(ElseGoal0, ElseGoals0),
    maplist(typr_strip_module_goal, ElseGoals0, ElseGoals),
    typr_mutual_tree_alias_map([], LeftVar, RightVar, AliasMap0),
    typr_mutual_tree_context_fields(Pred, ContextVars, HelperName, ExtraArgMap0, HelperParams, WrapperCallArgs, MemoKeyExpr),
    typr_mutual_tree_value_branch_body(
        GroupPredicates,
        ThenGoals,
        ValueVar,
        AliasMap0,
        ExtraArgMap0,
        PostBody,
        OutputVar,
        ThenBody
    ),
    typr_mutual_tree_value_branch_body(
        GroupPredicates,
        ElseGoals,
        ValueVar,
        AliasMap0,
        ExtraArgMap0,
        PostBody,
        OutputVar,
        ElseBody
    ),
    typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap0, ExtraArgMap0, BranchConditionExpr),
    atom_string(Pred, PredStr),
    Spec = mutual_spec{
        kind: tree_dual_value_body,
        pred: Pred,
        arity: Arity,
        pred_str: PredStr,
        typed_arg_list: TypedArgList,
        return_type: "int",
        helper_name: HelperName,
        helper_params: HelperParams,
        wrapper_call_args: WrapperCallArgs,
        memo_key_expr: MemoKeyExpr,
        base_cases: BaseCases,
        guard_expr: 'length(current_input) == 3',
        branch_condition_expr: BranchConditionExpr,
        then_body: ThenBody,
        else_body: ElseBody
    }.

typr_mutual_tree_dual_value_branch_spec(GroupPredicates, Pred/Arity, Spec) :-
    Arity >= 2,
    findall(Head-Body, predicate_clause(user, Pred, Arity, Head, Body), Clauses),
    Clauses \= [],
    generic_typr_return_type(Pred/Arity, Clauses, "int"),
    build_typed_arg_list(Pred/Arity, none, Arity, explicit, TypedArgList),
    findall(clause(Head, Body), predicate_clause(user, Pred, Arity, Head, Body), ClauseTerms),
    partition(typr_is_mutual_recursive_clause(GroupPredicates), ClauseTerms, RecClauses, BaseClauses),
    RecClauses = [clause(RecHead, RecBody)],
    BaseClauses \= [],
    RecHead =.. [_PredName, RecInputPattern|ContextAndOutputVars],
    append(ContextVars, [OutputVar], ContextAndOutputVars),
    var(OutputVar),
    RecInputPattern = [ValueVar, LeftVar, RightVar],
    typr_mutual_tree_context_param_names(ContextVars, ContextParamNames),
    maplist(typr_mutual_tree_value_base_case(ContextParamNames), BaseClauses, BaseCases),
    typr_mutual_goal_list(RecBody, Goals0),
    maplist(typr_strip_module_goal, Goals0, Goals),
    Goals = [BranchGoal|PostGoals],
    PostGoals \= [],
    typr_if_then_else_goal(BranchGoal, IfGoal, ThenGoal0, ElseGoal0),
    typr_mutual_goal_list(ThenGoal0, ThenGoals0),
    maplist(typr_strip_module_goal, ThenGoals0, ThenGoals),
    typr_mutual_goal_list(ElseGoal0, ElseGoals0),
    maplist(typr_strip_module_goal, ElseGoals0, ElseGoals),
    typr_mutual_tree_alias_map([], LeftVar, RightVar, AliasMap0),
    typr_mutual_tree_context_fields(Pred, ContextVars, HelperName, ExtraArgMap0, HelperParams, WrapperCallArgs, MemoKeyExpr),
    maplist(
        typr_mutual_tree_value_recursive_goal(GroupPredicates, AliasMap0, ExtraArgMap0),
        ThenGoals,
        ThenGoalSpecs0
    ),
    maplist(
        typr_mutual_tree_value_recursive_goal(GroupPredicates, AliasMap0, ExtraArgMap0),
        ElseGoals,
        ElseGoalSpecs0
    ),
    typr_mutual_tree_value_side_calls(ThenGoalSpecs0, ThenLeftCall, ThenRightCall, ThenLeftOutputVar, ThenRightOutputVar),
    typr_mutual_tree_value_side_calls(ElseGoalSpecs0, ElseLeftCall, ElseRightCall, ElseLeftOutputVar, ElseRightOutputVar),
    typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap0, ExtraArgMap0, BranchConditionExpr),
    typr_goals_to_body(PostGoals, PostBody),
    typr_mutual_tree_condition_varmap(ValueVar, AliasMap0, ExtraArgMap0, ResultVarMap0),
    update_typr_expr_varmap(ResultVarMap0, ThenLeftOutputVar, left_result, ThenResultVarMap1),
    update_typr_expr_varmap(ThenResultVarMap1, ThenRightOutputVar, right_result, ThenResultVarMap),
    update_typr_expr_varmap(ResultVarMap0, ElseLeftOutputVar, left_result, ElseResultVarMap1),
    update_typr_expr_varmap(ElseResultVarMap1, ElseRightOutputVar, right_result, ElseResultVarMap),
    linear_recursive_output_expr(PostBody, OutputVar, ThenResultVarMap, ThenResultExpr),
    linear_recursive_output_expr(PostBody, OutputVar, ElseResultVarMap, ElseResultExpr),
    atom_string(Pred, PredStr),
    Spec = mutual_spec{
        kind: tree_dual_value_branch,
        pred: Pred,
        arity: Arity,
        pred_str: PredStr,
        typed_arg_list: TypedArgList,
        return_type: "int",
        helper_name: HelperName,
        helper_params: HelperParams,
        wrapper_call_args: WrapperCallArgs,
        memo_key_expr: MemoKeyExpr,
        base_cases: BaseCases,
        guard_expr: 'length(current_input) == 3',
        branch_condition_expr: BranchConditionExpr,
        then_left_call: ThenLeftCall,
        then_right_call: ThenRightCall,
        else_left_call: ElseLeftCall,
        else_right_call: ElseRightCall,
        then_result_expr: ThenResultExpr,
        else_result_expr: ElseResultExpr
    }.

typr_mutual_tree_spec(GroupPredicates, Pred/Arity, Spec) :-
    Arity =:= 1,
    findall(Head-Body, predicate_clause(user, Pred, Arity, Head, Body), Clauses),
    Clauses \= [],
    generic_typr_return_type(Pred/Arity, Clauses, "bool"),
    build_typed_arg_list(Pred/Arity, none, Arity, explicit, TypedArgList),
    findall(clause(Head, Body), predicate_clause(user, Pred, Arity, Head, Body), ClauseTerms),
    partition(is_mutual_recursive_clause(GroupPredicates), ClauseTerms, RecClauses, BaseClauses),
    RecClauses = [clause(RecHead, RecBody)],
    BaseClauses \= [],
    maplist(typr_mutual_tree_base_case_condition, BaseClauses, BaseConds0),
    sort(BaseConds0, BaseConditions),
    RecHead =.. [_PredName, RecInputPattern],
    RecInputPattern = [ValueVar, LeftVar, RightVar],
    parse_recursive_body(RecBody, ValueVar, Conditions, Computations, RecCall),
    Conditions == [],
    Computations == [],
    RecCall \= none,
    RecCall =.. [NextPred, RecArg],
    memberchk(NextPred/1, GroupPredicates),
    (   RecArg == LeftVar ->
        StepExpr = '.subset2(current_input, 2)'
    ;   RecArg == RightVar ->
        StepExpr = '.subset2(current_input, 3)'
    ),
    atom_string(Pred, PredStr),
    atom_string(NextPred, NextPredStr),
    format(string(HelperName), '~w_impl', [PredStr]),
    format(string(NextHelperName), '~w_impl', [NextPredStr]),
    Spec = mutual_spec{
        kind: tree,
        pred: Pred,
        arity: Arity,
        pred_str: PredStr,
        typed_arg_list: TypedArgList,
        return_type: "bool",
        helper_name: HelperName,
        next_helper_name: NextHelperName,
        base_conditions: BaseConditions,
        guard_expr: 'length(current_input) == 3',
        step_expr: StepExpr
    }.

typr_mutual_tree_dual_spec(GroupPredicates, Pred/Arity, Spec) :-
    Arity =:= 1,
    findall(Head-Body, predicate_clause(user, Pred, Arity, Head, Body), Clauses),
    Clauses \= [],
    generic_typr_return_type(Pred/Arity, Clauses, "bool"),
    build_typed_arg_list(Pred/Arity, none, Arity, explicit, TypedArgList),
    findall(clause(Head, Body), predicate_clause(user, Pred, Arity, Head, Body), ClauseTerms),
    partition(is_mutual_recursive_clause(GroupPredicates), ClauseTerms, RecClauses, BaseClauses),
    RecClauses = [clause(RecHead, RecBody)],
    BaseClauses \= [],
    maplist(typr_mutual_tree_base_case_condition, BaseClauses, BaseConds0),
    sort(BaseConds0, BaseConditions),
    RecHead =.. [_PredName, RecInputPattern],
    RecInputPattern = [_ValueVar, LeftVar, RightVar],
    typr_mutual_goal_list(RecBody, Goals0),
    maplist(typr_strip_module_goal, Goals0, Goals),
    typr_mutual_tree_split_pre_goals(GroupPredicates, Goals, PreGoals, RecGoals),
    RecGoals = [GoalA, GoalB],
    typr_mutual_tree_alias_map(PreGoals, LeftVar, RightVar, AliasMap),
    maplist(typr_mutual_tree_recursive_goal(GroupPredicates, AliasMap), [GoalA, GoalB], GoalSpecs0),
    select(call_spec(left, LeftNextPred, LeftCallArgs), GoalSpecs0, GoalSpecs1),
    select(call_spec(right, RightNextPred, RightCallArgs), GoalSpecs1, []),
    LeftCallArgs = ['.subset2(current_input, 2)'],
    RightCallArgs = ['.subset2(current_input, 3)'],
    atom_string(Pred, PredStr),
    atom_string(LeftNextPred, LeftNextPredStr),
    atom_string(RightNextPred, RightNextPredStr),
    format(string(HelperName), '~w_impl', [PredStr]),
    format(string(LeftNextHelperName), '~w_impl', [LeftNextPredStr]),
    format(string(RightNextHelperName), '~w_impl', [RightNextPredStr]),
    Spec = mutual_spec{
        kind: tree_dual,
        pred: Pred,
        arity: Arity,
        pred_str: PredStr,
        typed_arg_list: TypedArgList,
        return_type: "bool",
        helper_name: HelperName,
        left_call: branch_call(LeftNextHelperName, LeftCallArgs),
        right_call: branch_call(RightNextHelperName, RightCallArgs),
        base_conditions: BaseConditions,
        guard_expr: 'length(current_input) == 3'
    }.

typr_mutual_tree_dual_context_spec(GroupPredicates, Pred/Arity, Spec) :-
    Arity >= 2,
    findall(Head-Body, predicate_clause(user, Pred, Arity, Head, Body), Clauses),
    Clauses \= [],
    generic_typr_return_type(Pred/Arity, Clauses, "bool"),
    build_typed_arg_list(Pred/Arity, none, Arity, explicit, TypedArgList),
    findall(clause(Head, Body), predicate_clause(user, Pred, Arity, Head, Body), ClauseTerms),
    partition(is_mutual_recursive_clause(GroupPredicates), ClauseTerms, RecClauses, BaseClauses),
    RecClauses = [clause(RecHead, RecBody)],
    BaseClauses \= [],
    maplist(typr_mutual_tree_base_case_condition, BaseClauses, BaseConds0),
    sort(BaseConds0, BaseConditions),
    RecHead =.. [_PredName, RecInputPattern|ContextVars],
    RecInputPattern = [ValueVar, LeftVar, RightVar],
    typr_mutual_goal_list(RecBody, Goals0),
    maplist(typr_strip_module_goal, Goals0, Goals),
    typr_mutual_tree_alias_map([], LeftVar, RightVar, AliasMap0),
    typr_mutual_tree_context_fields(Pred, ContextVars, HelperName, ExtraArgMap0, HelperParams, WrapperCallArgs, MemoKeyExpr),
    typr_mutual_tree_ordered_dual_context_goals(
        GroupPredicates,
        Goals,
        ValueVar,
        AliasMap0,
        ExtraArgMap0,
        Body
    ),
    atom_string(Pred, PredStr),
    Spec = mutual_spec{
        kind: tree_dual_body,
        pred: Pred,
        arity: Arity,
        pred_str: PredStr,
        typed_arg_list: TypedArgList,
        return_type: "bool",
        helper_name: HelperName,
        helper_params: HelperParams,
        wrapper_call_args: WrapperCallArgs,
        memo_key_expr: MemoKeyExpr,
        base_conditions: BaseConditions,
        guard_expr: 'length(current_input) == 3',
        body: Body
    }.

typr_mutual_tree_dual_context_spec(GroupPredicates, Pred/Arity, Spec) :-
    Arity >= 2,
    findall(Head-Body, predicate_clause(user, Pred, Arity, Head, Body), Clauses),
    Clauses \= [],
    generic_typr_return_type(Pred/Arity, Clauses, "bool"),
    build_typed_arg_list(Pred/Arity, none, Arity, explicit, TypedArgList),
    findall(clause(Head, Body), predicate_clause(user, Pred, Arity, Head, Body), ClauseTerms),
    partition(is_mutual_recursive_clause(GroupPredicates), ClauseTerms, RecClauses, BaseClauses),
    RecClauses = [clause(RecHead, RecBody)],
    BaseClauses \= [],
    maplist(typr_mutual_tree_base_case_condition, BaseClauses, BaseConds0),
    sort(BaseConds0, BaseConditions),
    RecHead =.. [_PredName, RecInputPattern|ContextVars],
    RecInputPattern = [ValueVar, LeftVar, RightVar],
    typr_mutual_goal_list(RecBody, Goals0),
    maplist(typr_strip_module_goal, Goals0, Goals),
    append(PreGoals, RecGoals, Goals),
    RecGoals = [GoalA, GoalB],
    typr_mutual_tree_alias_map([], LeftVar, RightVar, AliasMap0),
    typr_mutual_tree_context_fields(Pred, ContextVars, HelperName, ExtraArgMap0, HelperParams, WrapperCallArgs, MemoKeyExpr),
    typr_mutual_tree_branch_prework(PreGoals, ValueVar, AliasMap0, ExtraArgMap0, AliasMap, ExtraArgMap, PreGuardExpr),
    maplist(
        typr_mutual_tree_recursive_goal(GroupPredicates, AliasMap, ExtraArgMap),
        [GoalA, GoalB],
        GoalSpecs0
    ),
    select(call_spec(left, LeftNextPred, LeftCallArgs), GoalSpecs0, GoalSpecs1),
    select(call_spec(right, RightNextPred, RightCallArgs), GoalSpecs1, []),
    LeftCallArgs = ['.subset2(current_input, 2)'|CurrentCtxExprs],
    RightCallArgs = ['.subset2(current_input, 3)'|CurrentCtxExprs],
    atom_string(Pred, PredStr),
    atom_string(LeftNextPred, LeftNextPredStr),
    atom_string(RightNextPred, RightNextPredStr),
    format(string(LeftNextHelperName), '~w_impl', [LeftNextPredStr]),
    format(string(RightNextHelperName), '~w_impl', [RightNextPredStr]),
    typr_mutual_tree_top_guard_expr(PreGuardExpr, GuardExpr),
    Spec = mutual_spec{
        kind: tree_dual,
        pred: Pred,
        arity: Arity,
        pred_str: PredStr,
        typed_arg_list: TypedArgList,
        return_type: "bool",
        helper_name: HelperName,
        helper_params: HelperParams,
        wrapper_call_args: WrapperCallArgs,
        memo_key_expr: MemoKeyExpr,
        left_call: branch_call(LeftNextHelperName, LeftCallArgs),
        right_call: branch_call(RightNextHelperName, RightCallArgs),
        base_conditions: BaseConditions,
        guard_expr: GuardExpr
    }.

typr_mutual_tree_context_fields(Pred, ContextVars, HelperName, ExtraArgMap, HelperParams, WrapperCallArgs, MemoKeyExpr) :-
    atom_string(Pred, PredStr),
    format(string(HelperName), '~w_impl', [PredStr]),
    typr_mutual_tree_context_param_names(ContextVars, ContextParamNames),
    typr_mutual_tree_context_varmap(ContextVars, ContextParamNames, ExtraArgMap),
    HelperParams = ["current_input"|ContextParamNames],
    length(ContextVars, ContextArity),
    WrapperArity is ContextArity + 1,
    typr_mutual_wrapper_arg_names(WrapperArity, WrapperCallArgs),
    atomic_list_concat(HelperParams, ', ', HelperParamText),
    format(
        string(MemoKeyExpr),
        'paste0("~w:", paste(deparse(list(~w)), collapse=""))',
        [PredStr, HelperParamText]
    ).

typr_mutual_tree_context_param_names([], []).
typr_mutual_tree_context_param_names([_], ["current_ctx"]) :-
    !.
typr_mutual_tree_context_param_names([_|Rest], ["current_ctx"|Names]) :-
    typr_mutual_tree_context_param_names(Rest, 2, Names).

typr_mutual_tree_context_param_names([], _Index, []).
typr_mutual_tree_context_param_names([_|Rest], Index, [Name|Names]) :-
    format(string(Name), 'current_ctx_~d', [Index]),
    NextIndex is Index + 1,
    typr_mutual_tree_context_param_names(Rest, NextIndex, Names).

typr_mutual_tree_context_varmap([], [], []).
typr_mutual_tree_context_varmap([Var|Vars], [Name|Names], [Var-Name|Pairs]) :-
    typr_mutual_tree_context_varmap(Vars, Names, Pairs).

typr_mutual_wrapper_arg_names(Arity, Names) :-
    findall(Name, (
        between(1, Arity, Index),
        format(string(Name), 'arg~d', [Index])
    ), Names).

typr_mutual_tree_dual_body_spec(GroupPredicates, Pred/Arity, Spec) :-
    Arity >= 1,
    findall(Head-Body, predicate_clause(user, Pred, Arity, Head, Body), Clauses),
    Clauses \= [],
    generic_typr_return_type(Pred/Arity, Clauses, "bool"),
    build_typed_arg_list(Pred/Arity, none, Arity, explicit, TypedArgList),
    findall(clause(Head, Body), predicate_clause(user, Pred, Arity, Head, Body), ClauseTerms),
    partition(typr_is_mutual_recursive_clause(GroupPredicates), ClauseTerms, RecClauses, BaseClauses),
    RecClauses = [clause(RecHead, RecBody)],
    BaseClauses \= [],
    maplist(typr_mutual_tree_base_case_condition, BaseClauses, BaseConds0),
    sort(BaseConds0, BaseConditions),
    RecHead =.. [_PredName, RecInputPattern|ContextVars],
    RecInputPattern = [ValueVar, LeftVar, RightVar],
    typr_mutual_goal_list(RecBody, Goals0),
    maplist(typr_strip_module_goal, Goals0, Goals),
    typr_mutual_tree_alias_map([], LeftVar, RightVar, AliasMap0),
    typr_mutual_tree_context_fields(
        Pred,
        ContextVars,
        HelperName,
        ExtraArgMap0,
        HelperParams,
        WrapperCallArgs,
        MemoKeyExpr
    ),
    typr_mutual_tree_branch_body(
        GroupPredicates,
        Goals,
        ValueVar,
        AliasMap0,
        ExtraArgMap0,
        Body
    ),
    atom_string(Pred, PredStr),
    Spec = mutual_spec{
        kind: tree_dual_body,
        pred: Pred,
        arity: Arity,
        pred_str: PredStr,
        typed_arg_list: TypedArgList,
        return_type: "bool",
        helper_name: HelperName,
        helper_params: HelperParams,
        wrapper_call_args: WrapperCallArgs,
        memo_key_expr: MemoKeyExpr,
        base_conditions: BaseConditions,
        guard_expr: 'length(current_input) == 3',
        body: Body
    }.

typr_mutual_tree_dual_branch_spec(GroupPredicates, Pred/Arity, Spec) :-
    Arity =:= 1,
    findall(Head-Body, predicate_clause(user, Pred, Arity, Head, Body), Clauses),
    Clauses \= [],
    generic_typr_return_type(Pred/Arity, Clauses, "bool"),
    build_typed_arg_list(Pred/Arity, none, Arity, explicit, TypedArgList),
    findall(clause(Head, Body), predicate_clause(user, Pred, Arity, Head, Body), ClauseTerms),
    partition(typr_is_mutual_recursive_clause(GroupPredicates), ClauseTerms, RecClauses, BaseClauses),
    RecClauses = [clause(RecHead, RecBody)],
    BaseClauses \= [],
    maplist(typr_mutual_tree_base_case_condition, BaseClauses, BaseConds0),
    sort(BaseConds0, BaseConditions),
    RecHead =.. [_PredName, RecInputPattern],
    RecInputPattern = [ValueVar, LeftVar, RightVar],
    typr_mutual_tree_alias_map([], LeftVar, RightVar, AliasMap0),
    typr_mutual_goal_list(RecBody, Goals0),
    maplist(typr_strip_module_goal, Goals0, Goals),
    Goals = [BranchGoal],
    typr_if_then_else_goal(BranchGoal, IfGoal, ThenGoal0, ElseGoal0),
    typr_mutual_goal_list(ThenGoal0, ThenGoals0),
    maplist(typr_strip_module_goal, ThenGoals0, ThenGoals),
    typr_mutual_tree_branch_body(GroupPredicates, ThenGoals, ValueVar, AliasMap0, ThenBody),
    typr_mutual_goal_list(ElseGoal0, ElseGoals0),
    maplist(typr_strip_module_goal, ElseGoals0, ElseGoals),
    typr_mutual_tree_branch_body(GroupPredicates, ElseGoals, ValueVar, AliasMap0, ElseBody),
    typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap0, BranchConditionExpr),
    atom_string(Pred, PredStr),
    format(string(HelperName), '~w_impl', [PredStr]),
    Spec = mutual_spec{
        kind: tree_dual_branch,
        pred: Pred,
        arity: Arity,
        pred_str: PredStr,
        typed_arg_list: TypedArgList,
        return_type: "bool",
        helper_name: HelperName,
        base_conditions: BaseConditions,
        guard_expr: 'length(current_input) == 3',
        branch_condition_expr: BranchConditionExpr,
        then_body: ThenBody,
        else_body: ElseBody
    }.

typr_mutual_tree_dual_branch_spec(GroupPredicates, Pred/Arity, Spec) :-
    Arity =:= 1,
    findall(Head-Body, predicate_clause(user, Pred, Arity, Head, Body), Clauses),
    Clauses \= [],
    generic_typr_return_type(Pred/Arity, Clauses, "bool"),
    build_typed_arg_list(Pred/Arity, none, Arity, explicit, TypedArgList),
    findall(clause(Head, Body), predicate_clause(user, Pred, Arity, Head, Body), ClauseTerms),
    partition(is_mutual_recursive_clause(GroupPredicates), ClauseTerms, RecClauses, BaseClauses),
    RecClauses = [clause(RecHead, RecBody)],
    BaseClauses \= [],
    maplist(typr_mutual_tree_base_case_condition, BaseClauses, BaseConds0),
    sort(BaseConds0, BaseConditions),
    RecHead =.. [_PredName, RecInputPattern],
    RecInputPattern = [ValueVar, LeftVar, RightVar],
    typr_mutual_goal_list(RecBody, Goals0),
    maplist(typr_strip_module_goal, Goals0, Goals),
    Goals = [BranchGoal, GoalA, GoalB],
    typr_if_then_else_goal(BranchGoal, IfGoal, ThenGoal0, ElseGoal0),
    typr_mutual_goal_list(ThenGoal0, ThenGoals0),
    maplist(typr_strip_module_goal, ThenGoals0, ThenGoals),
    maplist(typr_mutual_tree_alias_goal, ThenGoals),
    typr_mutual_goal_list(ElseGoal0, ElseGoals0),
    maplist(typr_strip_module_goal, ElseGoals0, ElseGoals),
    maplist(typr_mutual_tree_alias_goal, ElseGoals),
    typr_mutual_tree_alias_map(ThenGoals, LeftVar, RightVar, ThenAliasMap),
    typr_mutual_tree_alias_map(ElseGoals, LeftVar, RightVar, ElseAliasMap),
    maplist(typr_mutual_tree_recursive_goal(GroupPredicates, ThenAliasMap), [GoalA, GoalB], ThenGoalSpecs),
    maplist(typr_mutual_tree_recursive_goal(GroupPredicates, ElseAliasMap), [GoalA, GoalB], ElseGoalSpecs),
    typr_mutual_tree_call_specs_cover_both_sides(ThenGoalSpecs),
    typr_mutual_tree_call_specs_cover_both_sides(ElseGoalSpecs),
    typr_mutual_tree_alias_map([], LeftVar, RightVar, AliasMap0),
    typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap0, BranchConditionExpr),
    maplist(typr_mutual_tree_branch_call, ThenGoalSpecs, ThenCalls),
    maplist(typr_mutual_tree_branch_call, ElseGoalSpecs, ElseCalls),
    atom_string(Pred, PredStr),
    format(string(HelperName), '~w_impl', [PredStr]),
    Spec = mutual_spec{
        kind: tree_dual_branch,
        pred: Pred,
        arity: Arity,
        pred_str: PredStr,
        typed_arg_list: TypedArgList,
        return_type: "bool",
        helper_name: HelperName,
        base_conditions: BaseConditions,
        guard_expr: 'length(current_input) == 3',
        branch_condition_expr: BranchConditionExpr,
        then_body: branch_calls(ThenCalls),
        else_body: branch_calls(ElseCalls)
    }.

typr_mutual_tree_dual_branch_spec(GroupPredicates, Pred/Arity, Spec) :-
    Arity >= 2,
    findall(Head-Body, predicate_clause(user, Pred, Arity, Head, Body), Clauses),
    Clauses \= [],
    generic_typr_return_type(Pred/Arity, Clauses, "bool"),
    build_typed_arg_list(Pred/Arity, none, Arity, explicit, TypedArgList),
    findall(clause(Head, Body), predicate_clause(user, Pred, Arity, Head, Body), ClauseTerms),
    partition(typr_is_mutual_recursive_clause(GroupPredicates), ClauseTerms, RecClauses, BaseClauses),
    RecClauses = [clause(RecHead, RecBody)],
    BaseClauses \= [],
    maplist(typr_mutual_tree_base_case_condition, BaseClauses, BaseConds0),
    sort(BaseConds0, BaseConditions),
    RecHead =.. [_PredName, RecInputPattern|ContextVars],
    RecInputPattern = [ValueVar, LeftVar, RightVar],
    typr_mutual_tree_alias_map([], LeftVar, RightVar, AliasMap0),
    typr_mutual_tree_context_fields(Pred, ContextVars, HelperName, ExtraArgMap, HelperParams, WrapperCallArgs, MemoKeyExpr),
    typr_mutual_goal_list(RecBody, Goals0),
    maplist(typr_strip_module_goal, Goals0, Goals),
    Goals = [BranchGoal],
    typr_if_then_else_goal(BranchGoal, IfGoal, ThenGoal0, ElseGoal0),
    typr_mutual_goal_list(ThenGoal0, ThenGoals0),
    maplist(typr_strip_module_goal, ThenGoals0, ThenGoals),
    typr_mutual_tree_branch_body(GroupPredicates, ThenGoals, ValueVar, AliasMap0, ExtraArgMap, ThenBody),
    typr_mutual_goal_list(ElseGoal0, ElseGoals0),
    maplist(typr_strip_module_goal, ElseGoals0, ElseGoals),
    typr_mutual_tree_branch_body(GroupPredicates, ElseGoals, ValueVar, AliasMap0, ExtraArgMap, ElseBody),
    typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap0, ExtraArgMap, BranchConditionExpr),
    atom_string(Pred, PredStr),
    Spec = mutual_spec{
        kind: tree_dual_branch,
        pred: Pred,
        arity: Arity,
        pred_str: PredStr,
        typed_arg_list: TypedArgList,
        return_type: "bool",
        helper_name: HelperName,
        helper_params: HelperParams,
        wrapper_call_args: WrapperCallArgs,
        memo_key_expr: MemoKeyExpr,
        base_conditions: BaseConditions,
        guard_expr: 'length(current_input) == 3',
        branch_condition_expr: BranchConditionExpr,
        then_body: ThenBody,
        else_body: ElseBody
    }.

typr_mutual_tree_dual_branch_spec(GroupPredicates, Pred/Arity, Spec) :-
    Arity >= 2,
    findall(Head-Body, predicate_clause(user, Pred, Arity, Head, Body), Clauses),
    Clauses \= [],
    generic_typr_return_type(Pred/Arity, Clauses, "bool"),
    build_typed_arg_list(Pred/Arity, none, Arity, explicit, TypedArgList),
    findall(clause(Head, Body), predicate_clause(user, Pred, Arity, Head, Body), ClauseTerms),
    partition(is_mutual_recursive_clause(GroupPredicates), ClauseTerms, RecClauses, BaseClauses),
    RecClauses = [clause(RecHead, RecBody)],
    BaseClauses \= [],
    maplist(typr_mutual_tree_base_case_condition, BaseClauses, BaseConds0),
    sort(BaseConds0, BaseConditions),
    RecHead =.. [_PredName, RecInputPattern|ContextVars],
    RecInputPattern = [ValueVar, LeftVar, RightVar],
    typr_mutual_goal_list(RecBody, Goals0),
    maplist(typr_strip_module_goal, Goals0, Goals),
    Goals = [BranchGoal, GoalA, GoalB],
    typr_if_then_else_goal(BranchGoal, IfGoal, ThenGoal0, ElseGoal0),
    typr_mutual_goal_list(ThenGoal0, ThenGoals0),
    maplist(typr_strip_module_goal, ThenGoals0, ThenGoals),
    typr_mutual_goal_list(ElseGoal0, ElseGoals0),
    maplist(typr_strip_module_goal, ElseGoals0, ElseGoals),
    typr_mutual_tree_alias_map([], LeftVar, RightVar, AliasMap0),
    typr_mutual_tree_context_fields(Pred, ContextVars, HelperName, ExtraArgMap0, HelperParams, WrapperCallArgs, MemoKeyExpr),
    typr_mutual_tree_branch_prework(
        ThenGoals,
        ValueVar,
        AliasMap0,
        ExtraArgMap0,
        ThenAliasMap,
        ThenExtraArgMap,
        ThenGuardExpr
    ),
    typr_mutual_tree_branch_prework(
        ElseGoals,
        ValueVar,
        AliasMap0,
        ExtraArgMap0,
        ElseAliasMap,
        ElseExtraArgMap,
        ElseGuardExpr
    ),
    maplist(
        typr_mutual_tree_recursive_goal(GroupPredicates, ThenAliasMap, ThenExtraArgMap),
        [GoalA, GoalB],
        ThenGoalSpecs
    ),
    maplist(
        typr_mutual_tree_recursive_goal(GroupPredicates, ElseAliasMap, ElseExtraArgMap),
        [GoalA, GoalB],
        ElseGoalSpecs
    ),
    typr_mutual_tree_call_specs_cover_both_sides(ThenGoalSpecs),
    typr_mutual_tree_call_specs_cover_both_sides(ElseGoalSpecs),
    typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap0, ExtraArgMap0, BranchConditionExpr),
    maplist(typr_mutual_tree_branch_call, ThenGoalSpecs, ThenCalls),
    maplist(typr_mutual_tree_branch_call, ElseGoalSpecs, ElseCalls),
    typr_mutual_tree_guard_wrap(ThenGuardExpr, branch_calls(ThenCalls), ThenBody),
    typr_mutual_tree_guard_wrap(ElseGuardExpr, branch_calls(ElseCalls), ElseBody),
    atom_string(Pred, PredStr),
    Spec = mutual_spec{
        kind: tree_dual_branch,
        pred: Pred,
        arity: Arity,
        pred_str: PredStr,
        typed_arg_list: TypedArgList,
        return_type: "bool",
        helper_name: HelperName,
        helper_params: HelperParams,
        wrapper_call_args: WrapperCallArgs,
        memo_key_expr: MemoKeyExpr,
        base_conditions: BaseConditions,
        guard_expr: 'length(current_input) == 3',
        branch_condition_expr: BranchConditionExpr,
        then_body: ThenBody,
        else_body: ElseBody
    }.

typr_mutual_tree_base_case_condition(clause(Head, true), 'length(current_input) == 0') :-
    Head =.. [_Pred, FirstArg|_],
    FirstArg == [].
typr_mutual_tree_base_case_condition(clause(Head, true), 'length(current_input) == 3 && length(.subset2(current_input, 2)) == 0 && length(.subset2(current_input, 3)) == 0') :-
    Head =.. [_Pred, [Val, [], []]|_],
    var(Val).

typr_mutual_guard_expr(_HeadVar, [], 'TRUE') :-
    !.
typr_mutual_guard_expr(HeadVar, Conditions, GuardExpr) :-
    VarMap = [HeadVar-"current_input"],
    findall(Expr0, (
        member(Condition, Conditions),
        typr_translate_r_expr(Condition, VarMap, RawExpr),
        typr_top_level_guard_expr(RawExpr, Expr0)
    ), Exprs0),
    list_to_set(Exprs0, Exprs),
    atomic_list_concat(Exprs, ' && ', GuardExpr).

typr_mutual_step_expr(HeadVar, _Computations, RecArg, "current_input") :-
    var(RecArg),
    RecArg == HeadVar,
    !.
typr_mutual_step_expr(HeadVar, Computations, RecArg, StepExpr) :-
    member(RecArg is StepTerm, Computations),
    !,
    typr_translate_r_expr(StepTerm, [HeadVar-"current_input"], StepExpr).
typr_mutual_step_expr(_HeadVar, _Computations, RecArg, StepExpr) :-
    atomic(RecArg),
    !,
    typr_translate_r_expr(RecArg, [], StepExpr).

typr_mutual_goal_list((A, B), Goals) :-
    !,
    typr_mutual_goal_list(A, GoalsA),
    typr_mutual_goal_list(B, GoalsB),
    append(GoalsA, GoalsB, Goals).
typr_mutual_goal_list(Goal, [Goal]).

split_typr_mutual_multicall_goals(GroupPredicates, Goals, PreGoals, RecGoals, PostGoals) :-
    split_typr_mutual_non_recursive_prefix(GroupPredicates, Goals, PreGoals, RecAndPostGoals),
    take_typr_mutual_recursive_goal_prefix(GroupPredicates, RecAndPostGoals, RecGoals, PostGoals),
    RecGoals = [_|[_|_]],
    PostGoals \= [],
    \+ typr_contains_mutual_recursive_goal(GroupPredicates, PostGoals).

split_typr_mutual_multicall_goals_allow_empty_post(GroupPredicates, Goals, PreGoals, RecGoals, PostGoals) :-
    split_typr_mutual_non_recursive_prefix(GroupPredicates, Goals, PreGoals, RecAndPostGoals),
    take_typr_mutual_recursive_goal_prefix(GroupPredicates, RecAndPostGoals, RecGoals, PostGoals),
    RecGoals = [_|[_|_]],
    \+ typr_contains_mutual_recursive_goal(GroupPredicates, PostGoals).

split_typr_mutual_recursive_goals_allow_empty_post(GroupPredicates, Goals, PreGoals, RecGoals, PostGoals) :-
    split_typr_mutual_non_recursive_prefix(GroupPredicates, Goals, PreGoals, RecAndPostGoals),
    take_typr_mutual_recursive_goal_prefix(GroupPredicates, RecAndPostGoals, RecGoals, PostGoals),
    RecGoals = [_|_],
    \+ typr_contains_mutual_recursive_goal(GroupPredicates, PostGoals).

split_typr_mutual_non_recursive_prefix(_GroupPredicates, [], [], []).
split_typr_mutual_non_recursive_prefix(GroupPredicates, [Goal|Rest], [], [Goal|Rest]) :-
    typr_goal_calls_mutual_group(GroupPredicates, Goal),
    !.
split_typr_mutual_non_recursive_prefix(GroupPredicates, [Goal|Rest], [Goal|PreGoals], PostGoals) :-
    split_typr_mutual_non_recursive_prefix(GroupPredicates, Rest, PreGoals, PostGoals).

take_typr_mutual_recursive_goal_prefix(GroupPredicates, [Goal|Rest], [Goal|RecGoals], PostGoals) :-
    typr_goal_calls_mutual_group(GroupPredicates, Goal),
    !,
    take_typr_mutual_recursive_goal_prefix(GroupPredicates, Rest, RecGoals, PostGoals).
take_typr_mutual_recursive_goal_prefix(_GroupPredicates, Goals, [], Goals).

typr_goal_calls_mutual_group(GroupPredicates, Goal0) :-
    nonvar(Goal0),
    typr_strip_module_goal(Goal0, Goal),
    compound(Goal),
    functor(Goal, Pred, Arity),
    memberchk(Pred/Arity, GroupPredicates).

typr_contains_mutual_recursive_goal(GroupPredicates, [Goal|_]) :-
    typr_goal_calls_mutual_group(GroupPredicates, Goal),
    !.
typr_contains_mutual_recursive_goal(GroupPredicates, [_|Rest]) :-
    typr_contains_mutual_recursive_goal(GroupPredicates, Rest).

typr_strip_module_goal(_Module:Goal, StrippedGoal) :-
    !,
    typr_strip_module_goal(Goal, StrippedGoal).
typr_strip_module_goal(Goal, Goal).

typr_mutual_tree_split_pre_goals(GroupPredicates, Goals, PreGoals, RecGoals) :-
    append(PreGoals, RecGoals, Goals),
    RecGoals = [_, _],
    maplist(typr_mutual_tree_alias_goal, PreGoals),
    maplist(typr_mutual_recursive_group_goal(GroupPredicates), RecGoals).

typr_mutual_recursive_group_goal(GroupPredicates, Goal0) :-
    typr_strip_module_goal(Goal0, Goal),
    Goal =.. [NextPred, RecArg],
    memberchk(NextPred/1, GroupPredicates),
    var(RecArg).

typr_mutual_tree_alias_goal(Goal) :-
    Goal = (Left = Right),
    var(Left),
    var(Right).

typr_mutual_tree_alias_map(PreGoals, LeftVar, RightVar, AliasMap) :-
    typr_mutual_tree_alias_map(PreGoals, [LeftVar-left, RightVar-right], AliasMap).

typr_mutual_tree_alias_map([], AliasMap, AliasMap).
typr_mutual_tree_alias_map([Goal|Rest], AliasMap0, AliasMap) :-
    typr_mutual_tree_alias_binding(Goal, AliasMap0, AliasMap1),
    typr_mutual_tree_alias_map(Rest, AliasMap1, AliasMap).

typr_mutual_tree_alias_binding(Left = Right, AliasMap0, AliasMap) :-
    (   typr_mutual_tree_lookup_side(Left, AliasMap0, Side) ->
        typr_mutual_tree_bind_var(Right, Side, AliasMap0, AliasMap)
    ;   typr_mutual_tree_lookup_side(Right, AliasMap0, Side) ->
        typr_mutual_tree_bind_var(Left, Side, AliasMap0, AliasMap)
    ).

typr_mutual_tree_bind_var(Var, Side, AliasMap0, AliasMap) :-
    (   typr_mutual_tree_lookup_side(Var, AliasMap0, ExistingSide) ->
        ExistingSide == Side,
        AliasMap = AliasMap0
    ;   AliasMap = [Var-Side|AliasMap0]
    ).

typr_mutual_tree_lookup_side(Var, [KnownVar-Side|_], Side) :-
    Var == KnownVar,
    !.
typr_mutual_tree_lookup_side(Var, [_|Rest], Side) :-
    typr_mutual_tree_lookup_side(Var, Rest, Side).

typr_mutual_tree_side_step_expr(left, '.subset2(current_input, 2)').
typr_mutual_tree_side_step_expr(right, '.subset2(current_input, 3)').

typr_is_mutual_recursive_clause(GroupPredicates, clause(_Head, Body)) :-
    sub_term(SubGoal0, Body),
    nonvar(SubGoal0),
    typr_strip_module_goal(SubGoal0, SubGoal),
    compound(SubGoal),
    functor(SubGoal, Pred, Arity),
    memberchk(Pred/Arity, GroupPredicates),
    !.

typr_mutual_tree_recursive_goal(GroupPredicates, AliasMap, Goal0, CallSpec) :-
    typr_mutual_tree_recursive_goal(GroupPredicates, AliasMap, [], Goal0, CallSpec).

typr_mutual_tree_recursive_goal(GroupPredicates, AliasMap, ExtraArgMap, Goal0, call_spec(Side, NextPred, CallArgs)) :-
    typr_strip_module_goal(Goal0, Goal),
    Goal =.. [NextPred, RecArg|ExtraArgs],
    length(ExtraArgs, ExtraArity),
    Arity is ExtraArity + 1,
    memberchk(NextPred/Arity, GroupPredicates),
    typr_mutual_tree_lookup_side(RecArg, AliasMap, Side),
    typr_mutual_tree_side_step_expr(Side, StepExpr),
    maplist(typr_mutual_extra_arg_expr(ExtraArgMap), ExtraArgs, ExtraArgExprs),
    CallArgs = [StepExpr|ExtraArgExprs].

typr_mutual_tree_value_recursive_goal(GroupPredicates, AliasMap, ExtraArgMap, Goal0, value_call_spec(Side, NextPred, CallArgs, OutputVar)) :-
    typr_strip_module_goal(Goal0, Goal),
    Goal =.. [NextPred, RecArg|ExtraAndOutputArgs],
    append(ExtraArgs, [OutputVar], ExtraAndOutputArgs),
    var(OutputVar),
    length(ExtraAndOutputArgs, TailArity),
    Arity is TailArity + 1,
    memberchk(NextPred/Arity, GroupPredicates),
    typr_mutual_tree_lookup_side(RecArg, AliasMap, Side),
    typr_mutual_tree_side_step_expr(Side, StepExpr),
    maplist(typr_mutual_extra_arg_expr(ExtraArgMap), ExtraArgs, ExtraArgExprs),
    CallArgs = [StepExpr|ExtraArgExprs].

typr_mutual_tree_value_side_calls(CallSpecs0, LeftCall, RightCall, LeftOutputVar, RightOutputVar) :-
    select(value_call_spec(left, LeftNextPred, LeftCallArgs, LeftOutputVar), CallSpecs0, CallSpecs1),
    select(value_call_spec(right, RightNextPred, RightCallArgs, RightOutputVar), CallSpecs1, []),
    atom_string(LeftNextPred, LeftNextPredStr),
    atom_string(RightNextPred, RightNextPredStr),
    format(string(LeftHelperName), '~w_impl', [LeftNextPredStr]),
    format(string(RightHelperName), '~w_impl', [RightNextPredStr]),
    LeftCall = branch_call(LeftHelperName, LeftCallArgs),
    RightCall = branch_call(RightHelperName, RightCallArgs).

typr_mutual_tree_value_call_bindings(CallSpecs0, CallBindings, ResultBindings) :-
    findall(
        Side,
        member(value_call_spec(Side, _NextPred, _CallArgs, _OutputVar), CallSpecs0),
        Sides0
    ),
    sort(Sides0, UniqueSides),
    length(Sides0, SideCount),
    length(UniqueSides, SideCount),
    typr_mutual_tree_value_call_binding(CallSpecs0, left, LeftCallBindings, LeftResultBindings),
    typr_mutual_tree_value_call_binding(CallSpecs0, right, RightCallBindings, RightResultBindings),
    append(LeftCallBindings, RightCallBindings, CallBindings),
    append(LeftResultBindings, RightResultBindings, ResultBindings).

typr_mutual_tree_value_call_binding(CallSpecs0, Side, [value_call_binding(Side, Call)], [result_binding(OutputVar, ResultName)]) :-
    member(value_call_spec(Side, NextPred, CallArgs, OutputVar), CallSpecs0),
    !,
    atom_string(NextPred, NextPredStr),
    format(string(HelperName), '~w_impl', [NextPredStr]),
    Call = branch_call(HelperName, CallArgs),
    typr_mutual_tree_result_name(Side, ResultName).
typr_mutual_tree_value_call_binding(_CallSpecs0, _Side, [], []).

typr_mutual_tree_result_name(left, left_result).
typr_mutual_tree_result_name(right, right_result).

typr_mutual_tree_value_branch_body(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap0, PostBody, OutputVar, Body) :-
    append(PreGoals, [NestedIfGoal|TailPostGoals], Goals),
    typr_mutual_tree_branch_prework(PreGoals, ValueVar, AliasMap0, ExtraArgMap0, AliasMap, ExtraArgMap1, GuardExpr),
    GuardExpr == none,
    typr_if_then_else_goal(NestedIfGoal, IfGoal, ThenGoal0, ElseGoal0),
    typr_mutual_tree_value_branch_post_body(TailPostGoals, PostBody, CombinedPostBody),
    typr_mutual_goal_list(ThenGoal0, ThenGoals0),
    maplist(typr_strip_module_goal, ThenGoals0, ThenGoals),
    typr_mutual_tree_value_branch_body(
        GroupPredicates,
        ThenGoals,
        ValueVar,
        AliasMap,
        ExtraArgMap1,
        CombinedPostBody,
        OutputVar,
        ThenBody
    ),
    typr_mutual_goal_list(ElseGoal0, ElseGoals0),
    maplist(typr_strip_module_goal, ElseGoals0, ElseGoals),
    typr_mutual_tree_value_branch_body(
        GroupPredicates,
        ElseGoals,
        ValueVar,
        AliasMap,
        ExtraArgMap1,
        CombinedPostBody,
        OutputVar,
        ElseBody
    ),
    typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap, ExtraArgMap1, BranchConditionExpr),
    Body = value_branch_if(BranchConditionExpr, ThenBody, ElseBody).
typr_mutual_tree_value_branch_body(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap0, PostBody, OutputVar, Body) :-
    split_typr_mutual_recursive_goals_allow_empty_post(GroupPredicates, Goals, PreGoals, RecGoals, BranchPostGoals),
    typr_mutual_tree_branch_prework(PreGoals, ValueVar, AliasMap0, ExtraArgMap0, AliasMap, ExtraArgMap1, GuardExpr),
    GuardExpr == none,
    maplist(
        typr_mutual_tree_value_recursive_goal(GroupPredicates, AliasMap, ExtraArgMap1),
        RecGoals,
        GoalSpecs0
    ),
    typr_mutual_tree_value_call_bindings(GoalSpecs0, CallBindings, ResultBindings),
    typr_mutual_tree_value_branch_post_body(BranchPostGoals, PostBody, CombinedPostBody),
    typr_mutual_tree_value_result_expr_from_bindings(
        CombinedPostBody,
        OutputVar,
        ValueVar,
        AliasMap,
        ExtraArgMap1,
        ResultBindings,
        ResultExpr
    ),
    Body = value_branch_leaf(CallBindings, ResultExpr).

typr_mutual_tree_value_branch_post_body(BranchPostGoals, PostBody, CombinedPostBody) :-
    typr_mutual_tree_value_post_body_goals(PostBody, SharedPostGoals),
    append(BranchPostGoals, SharedPostGoals, CombinedPostGoals),
    typr_goals_to_body(CombinedPostGoals, CombinedPostBody).

typr_mutual_tree_value_post_body_goals(true, []) :-
    !.
typr_mutual_tree_value_post_body_goals(PostBody, Goals) :-
    typr_mutual_goal_list(PostBody, Goals0),
    maplist(typr_strip_module_goal, Goals0, Goals).

typr_mutual_tree_value_result_expr(PostBody, OutputVar, ValueVar, AliasMap, ExtraArgMap, LeftOutputVar, RightOutputVar, ResultExpr) :-
    typr_mutual_tree_condition_varmap(ValueVar, AliasMap, ExtraArgMap, ResultVarMap0),
    update_typr_expr_varmap(ResultVarMap0, LeftOutputVar, left_result, ResultVarMap1),
    update_typr_expr_varmap(ResultVarMap1, RightOutputVar, right_result, ResultVarMap),
    linear_recursive_output_expr(PostBody, OutputVar, ResultVarMap, ResultExpr).

typr_mutual_tree_value_result_expr_from_bindings(PostBody, OutputVar, ValueVar, AliasMap, ExtraArgMap, ResultBindings, ResultExpr) :-
    typr_mutual_tree_condition_varmap(ValueVar, AliasMap, ExtraArgMap, ResultVarMap0),
    foldl(typr_mutual_tree_result_binding_varmap, ResultBindings, ResultVarMap0, ResultVarMap),
    linear_recursive_output_expr(PostBody, OutputVar, ResultVarMap, ResultExpr).

typr_mutual_tree_result_binding_varmap(result_binding(OutputVar, ResultName), VarMap0, VarMap) :-
    update_typr_expr_varmap(VarMap0, OutputVar, ResultName, VarMap).

typr_mutual_extra_arg_expr(ExtraArgMap, Arg, Expr) :-
    var(Arg),
    member(StoredArg-Expr, ExtraArgMap),
    StoredArg == Arg,
    !.
typr_mutual_extra_arg_expr(ExtraArgMap, Arg, Expr) :-
    typr_translate_r_expr(Arg, ExtraArgMap, Expr).

typr_mutual_tree_call_specs_cover_both_sides(CallSpecs) :-
    findall(Side, member(call_spec(Side, _NextPred, _StepExpr), CallSpecs), Sides0),
    msort(Sides0, [left, right]).

typr_mutual_tree_top_guard_expr(none, 'length(current_input) == 3') :-
    !.
typr_mutual_tree_top_guard_expr(PreGuardExpr, GuardExpr) :-
    combine_typr_conditions(['length(current_input) == 3', PreGuardExpr], CombinedGuard),
    typr_condition_expr_text(CombinedGuard, GuardExpr).

typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap, BranchConditionExpr) :-
    typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap, [], BranchConditionExpr).

typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap, ExtraArgMap, BranchConditionExpr) :-
    typr_mutual_tree_condition_varmap(ValueVar, AliasMap, ExtraArgMap, VarMap),
    native_typr_if_condition(IfGoal, VarMap, BranchCondition0),
    typr_condition_expr_text(BranchCondition0, BranchConditionExpr).

typr_mutual_tree_ordered_dual_context_goals(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap0, Body) :-
    typr_mutual_tree_goal_after_first_call(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap0),
    typr_mutual_tree_branch_body(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap0, Body).

typr_mutual_tree_goal_after_first_call(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap0) :-
    typr_mutual_tree_goal_after_first_call(before_call, GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap0).

typr_mutual_tree_goal_after_first_call(_State, _GroupPredicates, [], _ValueVar, _AliasMap, _ExtraArgMap) :-
    fail.
typr_mutual_tree_goal_after_first_call(before_call, GroupPredicates, [Goal0|Rest], ValueVar, AliasMap0, ExtraArgMap0) :-
    typr_strip_module_goal(Goal0, Goal),
    (   typr_mutual_tree_recursive_goal(GroupPredicates, AliasMap0, ExtraArgMap0, Goal, _GoalSpec)
    ->  typr_mutual_tree_goal_after_first_call(after_first_call, GroupPredicates, Rest, ValueVar, AliasMap0, ExtraArgMap0)
    ;   typr_mutual_tree_alias_goal(Goal)
    ->  typr_mutual_tree_alias_map([Goal], AliasMap0, AliasMap1),
        typr_mutual_tree_goal_after_first_call(before_call, GroupPredicates, Rest, ValueVar, AliasMap1, ExtraArgMap0)
    ;   typr_mutual_tree_symbolic_extra_arg_goal(Goal, ValueVar, AliasMap0, ExtraArgMap0, ExtraArgMap1)
    ->  typr_mutual_tree_goal_after_first_call(before_call, GroupPredicates, Rest, ValueVar, AliasMap0, ExtraArgMap1)
    ;   \+ typr_mutual_tree_nested_goal(Goal),
        typr_mutual_tree_condition_varmap(ValueVar, AliasMap0, ExtraArgMap0, VarMap),
        native_typr_guard_goal(Goal, VarMap, _GuardCondition)
    ->  typr_mutual_tree_goal_after_first_call(before_call, GroupPredicates, Rest, ValueVar, AliasMap0, ExtraArgMap0)
    ).
typr_mutual_tree_goal_after_first_call(after_first_call, GroupPredicates, [Goal0|Rest], ValueVar, AliasMap0, ExtraArgMap0) :-
    typr_strip_module_goal(Goal0, Goal),
    (   typr_mutual_tree_recursive_goal(GroupPredicates, AliasMap0, ExtraArgMap0, Goal, _GoalSpec)
    ->  Rest \= []
    ;   typr_mutual_tree_alias_goal(Goal)
    ->  true
    ;   typr_mutual_tree_symbolic_extra_arg_goal(Goal, ValueVar, AliasMap0, ExtraArgMap0, _ExtraArgMap1)
    ->  true
    ;   typr_mutual_tree_nested_goal(Goal)
    ->  true
    ;   \+ typr_mutual_tree_nested_goal(Goal),
        typr_mutual_tree_condition_varmap(ValueVar, AliasMap0, ExtraArgMap0, VarMap),
        native_typr_guard_goal(Goal, VarMap, _GuardCondition)
    ->  true
    ).

typr_mutual_tree_branch_prework([], _ValueVar, AliasMap, AliasMap, none).
typr_mutual_tree_branch_prework(Goals, ValueVar, AliasMap0, AliasMap, GuardExpr) :-
    typr_mutual_tree_branch_prework(Goals, ValueVar, AliasMap0, [], AliasMap, _ExtraArgMap, GuardExpr).

typr_mutual_tree_branch_prework(Goals, ValueVar, AliasMap0, ExtraArgMap0, AliasMap, GuardExpr) :-
    typr_mutual_tree_branch_prework(Goals, ValueVar, AliasMap0, ExtraArgMap0, AliasMap, _ExtraArgMap, GuardExpr).

typr_mutual_tree_branch_prework([], _ValueVar, AliasMap, ExtraArgMap, AliasMap, ExtraArgMap, none).
typr_mutual_tree_branch_prework([Goal|Rest], ValueVar, AliasMap0, ExtraArgMap0, AliasMap, ExtraArgMap, GuardExpr) :-
    (   typr_mutual_tree_alias_goal(Goal)
    ->  typr_mutual_tree_alias_map([Goal], AliasMap0, AliasMap1),
        typr_mutual_tree_branch_prework(Rest, ValueVar, AliasMap1, ExtraArgMap0, AliasMap, ExtraArgMap, GuardExpr)
    ;   typr_mutual_tree_symbolic_extra_arg_goal(Goal, ValueVar, AliasMap0, ExtraArgMap0, ExtraArgMap1)
    ->  typr_mutual_tree_branch_prework(Rest, ValueVar, AliasMap0, ExtraArgMap1, AliasMap, ExtraArgMap, GuardExpr)
    ;   typr_mutual_tree_condition_varmap(ValueVar, AliasMap0, ExtraArgMap0, VarMap),
        native_typr_guard_goal(Goal, VarMap, GuardCondition),
        typr_mutual_tree_branch_prework(Rest, ValueVar, AliasMap0, ExtraArgMap0, AliasMap, ExtraArgMap, RestGuardExpr),
        typr_mutual_tree_guard_expr_join(GuardCondition, RestGuardExpr, GuardExpr)
    ).

typr_mutual_tree_symbolic_extra_arg_goal(Goal, ValueVar, AliasMap, ExtraArgMap0, ExtraArgMap) :-
    typr_mutual_tree_condition_varmap(ValueVar, AliasMap, ExtraArgMap0, VarMap0),
    linear_recursive_post_goal_varmap(Goal, VarMap0, VarMap1),
    varmap_changed_vars(VarMap0, VarMap1, ChangedVars0),
    unique_vars_by_identity(ChangedVars0, ChangedVars),
    ChangedVars \= [],
    foldl(typr_mutual_tree_store_symbolic_expr(VarMap1), ChangedVars, ExtraArgMap0, ExtraArgMap).

typr_mutual_tree_store_symbolic_expr(VarMap, Var, ExtraArgMap0, ExtraArgMap) :-
    lookup_typr_var(Var, VarMap, Expr),
    update_typr_expr_varmap(ExtraArgMap0, Var, Expr, ExtraArgMap).

typr_mutual_tree_guard_expr_join(GuardCondition, none, GuardCondition) :-
    !.
typr_mutual_tree_guard_expr_join(GuardCondition, RestGuardExpr, GuardExpr) :-
    combine_typr_conditions([GuardCondition, RestGuardExpr], GuardExpr).

typr_mutual_tree_guard_wrap(none, Body, Body) :-
    !.
typr_mutual_tree_guard_wrap(GuardExpr, Body, branch_guard(GuardExpr, Body)).

typr_mutual_tree_nested_goal(Goal) :-
    typr_if_then_else_goal(Goal, _IfGoal, _ThenGoal, _ElseGoal),
    !.
typr_mutual_tree_nested_goal(Goal) :-
    typr_if_then_goal(Goal, _IfGoal, _ThenGoal).

typr_mutual_tree_condition_varmap(ValueVar, AliasMap, VarMap) :-
    typr_mutual_tree_condition_varmap(ValueVar, AliasMap, [], VarMap).

typr_mutual_tree_condition_varmap(ValueVar, AliasMap, ExtraArgMap, VarMap) :-
    findall(
        Var-Expr,
        (
            member(Var-Side, AliasMap),
            typr_mutual_tree_side_step_expr(Side, Expr)
        ),
        VarMap0
    ),
    append(ExtraArgMap, VarMap0, VarMap1),
    (   var(ValueVar)
    ->  VarMap = [ValueVar-'.subset2(current_input, 1)'|VarMap1]
    ;   VarMap = VarMap1
    ).

typr_mutual_tree_branch_call(call_spec(_Side, NextPred, CallArgs), branch_call(HelperName, CallArgs)) :-
    atom_string(NextPred, NextPredStr),
    format(string(HelperName), '~w_impl', [NextPredStr]).

typr_mutual_tree_branch_body(GroupPredicates, Goals, ValueVar, AliasMap0, Body) :-
    typr_mutual_tree_branch_body(GroupPredicates, Goals, ValueVar, AliasMap0, [], Body).

typr_mutual_tree_branch_body(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap, Body) :-
    append(PreGoals, [NestedIfGoal], Goals),
    typr_mutual_tree_branch_prework(PreGoals, ValueVar, AliasMap0, ExtraArgMap, AliasMap, ExtraArgMap1, GuardExpr),
    typr_if_then_else_goal(NestedIfGoal, IfGoal, ThenGoal0, ElseGoal0),
    typr_mutual_goal_list(ThenGoal0, ThenGoals0),
    maplist(typr_strip_module_goal, ThenGoals0, ThenGoals),
    typr_mutual_tree_branch_body(GroupPredicates, ThenGoals, ValueVar, AliasMap, ExtraArgMap1, ThenBody),
    typr_mutual_goal_list(ElseGoal0, ElseGoals0),
    maplist(typr_strip_module_goal, ElseGoals0, ElseGoals),
    typr_mutual_tree_branch_body(GroupPredicates, ElseGoals, ValueVar, AliasMap, ExtraArgMap1, ElseBody),
    typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap, ExtraArgMap1, BranchConditionExpr),
    typr_mutual_tree_guard_wrap(
        GuardExpr,
        branch_if(BranchConditionExpr, ThenBody, ElseBody),
        Body
    ).
typr_mutual_tree_branch_body(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap, Body) :-
    append(PreGoals, [FirstGoal0, NestedIfGoal], Goals),
    typr_mutual_tree_branch_prework(PreGoals, ValueVar, AliasMap0, ExtraArgMap, AliasMap, ExtraArgMap1, GuardExpr),
    typr_mutual_tree_recursive_goal(GroupPredicates, AliasMap, ExtraArgMap1, FirstGoal0, FirstGoalSpec),
    typr_mutual_tree_branch_call(FirstGoalSpec, FirstCall),
    typr_if_then_else_goal(NestedIfGoal, IfGoal, ThenGoal0, ElseGoal0),
    typr_mutual_goal_list(ThenGoal0, ThenGoals0),
    maplist(typr_strip_module_goal, ThenGoals0, ThenGoals),
    typr_mutual_tree_tail_body(
        GroupPredicates,
        ThenGoals,
        ValueVar,
        AliasMap,
        ExtraArgMap1,
        ThenBody,
        ThenGoalSpecs
    ),
    typr_mutual_tree_call_specs_cover_both_sides([FirstGoalSpec|ThenGoalSpecs]),
    typr_mutual_goal_list(ElseGoal0, ElseGoals0),
    maplist(typr_strip_module_goal, ElseGoals0, ElseGoals),
    typr_mutual_tree_tail_body(
        GroupPredicates,
        ElseGoals,
        ValueVar,
        AliasMap,
        ExtraArgMap1,
        ElseBody,
        ElseGoalSpecs
    ),
    typr_mutual_tree_call_specs_cover_both_sides([FirstGoalSpec|ElseGoalSpecs]),
    typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap, ExtraArgMap1, BranchConditionExpr),
    typr_mutual_tree_guard_wrap(
        GuardExpr,
        branch_after_call(FirstCall, BranchConditionExpr, ThenBody, ElseBody),
        Body
    ).
typr_mutual_tree_branch_body(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap, Body) :-
    append(PreGoals, [GoalA0, GoalB0, NestedIfGoal], Goals),
    typr_mutual_tree_branch_prework(PreGoals, ValueVar, AliasMap0, ExtraArgMap, AliasMap, ExtraArgMap1, GuardExpr),
    maplist(typr_mutual_tree_recursive_goal(GroupPredicates, AliasMap, ExtraArgMap1), [GoalA0, GoalB0], GoalSpecs),
    typr_mutual_tree_call_specs_cover_both_sides(GoalSpecs),
    maplist(typr_mutual_tree_branch_call, GoalSpecs, Calls),
    typr_if_then_else_goal(NestedIfGoal, IfGoal, ThenGoal0, ElseGoal0),
    typr_mutual_goal_list(ThenGoal0, ThenGoals0),
    maplist(typr_strip_module_goal, ThenGoals0, ThenGoals),
    typr_mutual_tree_tail_body(
        GroupPredicates,
        ThenGoals,
        ValueVar,
        AliasMap,
        ExtraArgMap1,
        ThenBody,
        ThenGoalSpecs
    ),
    ThenGoalSpecs = [],
    typr_mutual_goal_list(ElseGoal0, ElseGoals0),
    maplist(typr_strip_module_goal, ElseGoals0, ElseGoals),
    typr_mutual_tree_tail_body(
        GroupPredicates,
        ElseGoals,
        ValueVar,
        AliasMap,
        ExtraArgMap1,
        ElseBody,
        ElseGoalSpecs
    ),
    ElseGoalSpecs = [],
    typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap, ExtraArgMap1, BranchConditionExpr),
    typr_mutual_tree_guard_wrap(
        GuardExpr,
        branch_after_calls(Calls, BranchConditionExpr, ThenBody, ElseBody),
        Body
    ).
typr_mutual_tree_branch_body(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap0, Body) :-
    append(PrefixGoals, [NestedIfGoal], Goals),
    PrefixGoals \= [],
    typr_if_then_else_goal(NestedIfGoal, IfGoal, ThenGoal0, ElseGoal0),
    typr_mutual_tree_branch_sequence_state(
        GroupPredicates,
        PrefixGoals,
        ValueVar,
        AliasMap0,
        ExtraArgMap0,
        GoalSpecs,
        Steps,
        AliasMap,
        ExtraArgMap1
    ),
    typr_mutual_tree_call_specs_cover_both_sides(GoalSpecs),
    typr_mutual_goal_list(ThenGoal0, ThenGoals0),
    maplist(typr_strip_module_goal, ThenGoals0, ThenGoals),
    typr_mutual_tree_tail_body(
        GroupPredicates,
        ThenGoals,
        ValueVar,
        AliasMap,
        ExtraArgMap1,
        ThenBody,
        ThenGoalSpecs
    ),
    ThenGoalSpecs = [],
    typr_mutual_goal_list(ElseGoal0, ElseGoals0),
    maplist(typr_strip_module_goal, ElseGoals0, ElseGoals),
    typr_mutual_tree_tail_body(
        GroupPredicates,
        ElseGoals,
        ValueVar,
        AliasMap,
        ExtraArgMap1,
        ElseBody,
        ElseGoalSpecs
    ),
    ElseGoalSpecs = [],
    typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap, ExtraArgMap1, BranchConditionExpr),
    Body = branch_steps_if(Steps, BranchConditionExpr, ThenBody, ElseBody).
typr_mutual_tree_branch_body(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap, Body) :-
    typr_mutual_tree_split_pre_goals(GroupPredicates, Goals, PreGoals, RecGoals),
    RecGoals = [GoalA, GoalB],
    typr_mutual_tree_branch_prework(PreGoals, ValueVar, AliasMap0, ExtraArgMap, AliasMap, ExtraArgMap1, GuardExpr),
    maplist(typr_mutual_tree_recursive_goal(GroupPredicates, AliasMap, ExtraArgMap1), [GoalA, GoalB], GoalSpecs),
    typr_mutual_tree_call_specs_cover_both_sides(GoalSpecs),
    maplist(typr_mutual_tree_branch_call, GoalSpecs, Calls),
    typr_mutual_tree_guard_wrap(GuardExpr, branch_calls(Calls), Body).
typr_mutual_tree_branch_body(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap, Body) :-
    typr_mutual_tree_branch_sequence(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap, GoalSpecs, Steps),
    typr_mutual_tree_call_specs_cover_both_sides(GoalSpecs),
    typr_mutual_tree_branch_body_from_steps(Steps, Body).

typr_mutual_tree_branch_sequence(_GroupPredicates, [], _ValueVar, _AliasMap, _ExtraArgMap, [], []).
typr_mutual_tree_branch_sequence(GroupPredicates, [Goal0|Rest], ValueVar, AliasMap0, ExtraArgMap, GoalSpecs, Steps) :-
    typr_mutual_tree_branch_sequence_state(
        GroupPredicates,
        [Goal0|Rest],
        ValueVar,
        AliasMap0,
        ExtraArgMap,
        GoalSpecs,
        Steps,
        _AliasMap,
        _ExtraArgMap
    ).

typr_mutual_tree_branch_sequence_state(_GroupPredicates, [], _ValueVar, AliasMap, ExtraArgMap, [], [], AliasMap, ExtraArgMap).
typr_mutual_tree_branch_sequence_state(GroupPredicates, [Goal0|Rest], ValueVar, AliasMap0, ExtraArgMap0, GoalSpecs, Steps, AliasMap, ExtraArgMap) :-
    typr_strip_module_goal(Goal0, Goal),
    (   typr_mutual_tree_recursive_goal(GroupPredicates, AliasMap0, ExtraArgMap0, Goal, GoalSpec)
    ->  typr_mutual_tree_branch_call(GoalSpec, Call),
        GoalSpecs = [GoalSpec|RestGoalSpecs],
        Steps = [step_call(Call)|RestSteps],
        AliasMap1 = AliasMap0,
        ExtraArgMap1 = ExtraArgMap0
    ;   typr_mutual_tree_alias_goal(Goal)
    ->  GoalSpecs = RestGoalSpecs,
        Steps = RestSteps,
        typr_mutual_tree_alias_map([Goal], AliasMap0, AliasMap1),
        ExtraArgMap1 = ExtraArgMap0
    ;   typr_mutual_tree_symbolic_extra_arg_goal(Goal, ValueVar, AliasMap0, ExtraArgMap0, ExtraArgMap1)
    ->  GoalSpecs = RestGoalSpecs,
        Steps = RestSteps,
        AliasMap1 = AliasMap0
    ;   \+ typr_mutual_tree_nested_goal(Goal),
        typr_mutual_tree_condition_varmap(ValueVar, AliasMap0, ExtraArgMap0, VarMap),
        native_typr_guard_goal(Goal, VarMap, GuardCondition)
    ->  GoalSpecs = RestGoalSpecs,
        Steps = [step_guard(GuardCondition)|RestSteps],
        AliasMap1 = AliasMap0,
        ExtraArgMap1 = ExtraArgMap0
    ),
    typr_mutual_tree_branch_sequence_state(
        GroupPredicates,
        Rest,
        ValueVar,
        AliasMap1,
        ExtraArgMap1,
        RestGoalSpecs,
        RestSteps,
        AliasMap,
        ExtraArgMap
    ).

typr_mutual_tree_branch_body_from_steps([step_call(Call1), step_call(Call2)], branch_calls([Call1, Call2])) :-
    !.
typr_mutual_tree_branch_body_from_steps(Steps, branch_steps(Steps)).

typr_mutual_tree_nonrecursive_tail_body(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap0, Body) :-
    append(PreGoals, [NestedIfGoal], Goals),
    typr_mutual_tree_branch_prework(PreGoals, ValueVar, AliasMap0, ExtraArgMap0, AliasMap, ExtraArgMap1, GuardExpr),
    typr_if_then_else_goal(NestedIfGoal, IfGoal, ThenGoal0, ElseGoal0),
    typr_mutual_goal_list(ThenGoal0, ThenGoals0),
    maplist(typr_strip_module_goal, ThenGoals0, ThenGoals),
    typr_mutual_tree_nonrecursive_tail_body(GroupPredicates, ThenGoals, ValueVar, AliasMap, ExtraArgMap1, ThenBody),
    typr_mutual_goal_list(ElseGoal0, ElseGoals0),
    maplist(typr_strip_module_goal, ElseGoals0, ElseGoals),
    typr_mutual_tree_nonrecursive_tail_body(GroupPredicates, ElseGoals, ValueVar, AliasMap, ExtraArgMap1, ElseBody),
    typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap, ExtraArgMap1, BranchConditionExpr),
    typr_mutual_tree_guard_wrap(
        GuardExpr,
        branch_if(BranchConditionExpr, ThenBody, ElseBody),
        Body
    ).
typr_mutual_tree_nonrecursive_tail_body(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap0, Body) :-
    typr_mutual_tree_branch_sequence_state(
        GroupPredicates,
        Goals,
        ValueVar,
        AliasMap0,
        ExtraArgMap0,
        GoalSpecs,
        Steps,
        _AliasMap,
        _ExtraArgMap
    ),
    GoalSpecs = [],
    typr_mutual_tree_branch_body_from_steps(Steps, Body).

typr_mutual_tree_tail_body(GroupPredicates, Goals, ValueVar, AliasMap0, Body, GoalSpecs) :-
    typr_mutual_tree_tail_body(GroupPredicates, Goals, ValueVar, AliasMap0, [], Body, GoalSpecs).

typr_mutual_tree_tail_body(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap, Body, []) :-
    typr_mutual_tree_nonrecursive_tail_body(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap, Body).
typr_mutual_tree_tail_body(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap, Body, [FirstGoalSpec]) :-
    append(PreGoals, [FirstGoal0, NestedIfGoal], Goals),
    typr_mutual_tree_branch_prework(PreGoals, ValueVar, AliasMap0, ExtraArgMap, AliasMap, ExtraArgMap1, GuardExpr),
    typr_mutual_tree_recursive_goal(GroupPredicates, AliasMap, ExtraArgMap1, FirstGoal0, FirstGoalSpec),
    typr_mutual_tree_branch_call(FirstGoalSpec, FirstCall),
    typr_if_then_else_goal(NestedIfGoal, IfGoal, ThenGoal0, ElseGoal0),
    typr_mutual_goal_list(ThenGoal0, ThenGoals0),
    maplist(typr_strip_module_goal, ThenGoals0, ThenGoals),
    typr_mutual_tree_nonrecursive_tail_body(
        GroupPredicates,
        ThenGoals,
        ValueVar,
        AliasMap,
        ExtraArgMap1,
        ThenBody
    ),
    typr_mutual_goal_list(ElseGoal0, ElseGoals0),
    maplist(typr_strip_module_goal, ElseGoals0, ElseGoals),
    typr_mutual_tree_nonrecursive_tail_body(
        GroupPredicates,
        ElseGoals,
        ValueVar,
        AliasMap,
        ExtraArgMap1,
        ElseBody
    ),
    typr_mutual_tree_branch_condition_expr(IfGoal, ValueVar, AliasMap, ExtraArgMap1, BranchConditionExpr),
    typr_mutual_tree_guard_wrap(
        GuardExpr,
        branch_after_call(FirstCall, BranchConditionExpr, ThenBody, ElseBody),
        Body
    ).
typr_mutual_tree_tail_body(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap, Body, GoalSpecs) :-
    typr_mutual_tree_branch_sequence(GroupPredicates, Goals, ValueVar, AliasMap0, ExtraArgMap, GoalSpecs, Steps),
    typr_mutual_tree_branch_body_from_steps(Steps, Body).

typr_mutual_group_code(Specs, MemoEnabled, Code) :-
    findall(GroupLabel, (
        member(Spec, Specs),
        PredStr = Spec.pred_str,
        Arity = Spec.arity,
        format(string(GroupLabel), '~w/~w', [PredStr, Arity])
    ), GroupLabels),
    atomic_list_concat(GroupLabels, ', ', GroupText),
    findall(PredStr, (
        member(Spec, Specs),
        PredStr = Spec.pred_str
    ), PredStrs),
    atomic_list_concat(PredStrs, '_', GroupName),
    findall(WrapperCode, (
        member(Spec, Specs),
        typr_mutual_wrapper_code(GroupName, Specs, MemoEnabled, Spec, WrapperCode)
    ), WrapperCodes),
    atomic_list_concat(WrapperCodes, '\n\n', WrapperCodeText),
    format(string(Code),
'# Generated by UnifyWeaver TypR Target
# Mutual recursion group: ~w

~w
', [GroupText, WrapperCodeText]).

typr_mutual_spec_field_or_default(Spec, Field, Default, Value) :-
    (   get_dict(Field, Spec, Value0)
    ->  Value = Value0
    ;   Value = Default
    ).

typr_mutual_helper_params(Spec, Params) :-
    typr_mutual_spec_field_or_default(Spec, helper_params, ["current_input"], Params).

typr_mutual_wrapper_call_args(Spec, CallArgs) :-
    typr_mutual_spec_field_or_default(Spec, wrapper_call_args, ["arg1"], CallArgs).

typr_mutual_call_expr(Name, CallArgs, Expr) :-
    atomic_list_concat(CallArgs, ', ', CallArgsText),
    format(string(Expr), '~w(~w)', [Name, CallArgsText]).

typr_mutual_helper_line(Spec, HelperName, HelperLine) :-
    typr_mutual_helper_params(Spec, HelperParams),
    atomic_list_concat(HelperParams, ', ', HelperParamsText),
    format_string_return('    ~w <- function(~w) {', [HelperName, HelperParamsText], HelperLine).

typr_mutual_wrapper_helper_call_expr(Spec, HelperName, HelperCallExpr) :-
    typr_mutual_wrapper_call_args(Spec, WrapperCallArgs),
    typr_mutual_call_expr(HelperName, WrapperCallArgs, HelperCallExpr).

typr_mutual_tree_key_line(Spec, PredStr, KeyLine) :-
    (   get_dict(memo_key_expr, Spec, MemoKeyExpr)
    ->  format_string_return('        key <- ~w;', [MemoKeyExpr], KeyLine)
    ;   format_string_return('        key <- paste0("~w:", paste(deparse(current_input), collapse=""));', [PredStr], KeyLine)
    ).

typr_mutual_wrapper_code(GroupName, Specs, true, Spec, Code) :-
    PredStr = Spec.pred_str,
    TypedArgList = Spec.typed_arg_list,
    ReturnType = Spec.return_type,
    HelperName = Spec.helper_name,
    typr_mutual_wrapper_helper_call_expr(Spec, HelperName, HelperCallExpr),
    findall(HelperCode, (
        member(HelperSpec, Specs),
        typr_mutual_helper_code(GroupName, true, HelperSpec, HelperCode)
    ), HelperCodes),
    atomic_list_concat(HelperCodes, '\n', HelpersText),
    indent_text(HelpersText, "\t", IndentedHelpersText),
    format(string(Code),
'let ~w <- fn(~w): ~w {
\tresult <- @{
\tlocal({
\t    ~w_memo <- new.env(hash=TRUE, parent=emptyenv());
~w
\t    ~w
\t})
\t}@;
\tresult
};', [PredStr, TypedArgList, ReturnType, GroupName, IndentedHelpersText, HelperCallExpr]).
typr_mutual_wrapper_code(GroupName, Specs, false, Spec, Code) :-
    PredStr = Spec.pred_str,
    TypedArgList = Spec.typed_arg_list,
    ReturnType = Spec.return_type,
    HelperName = Spec.helper_name,
    typr_mutual_wrapper_helper_call_expr(Spec, HelperName, HelperCallExpr),
    findall(HelperCode, (
        member(HelperSpec, Specs),
        typr_mutual_helper_code(GroupName, false, HelperSpec, HelperCode)
    ), HelperCodes),
    atomic_list_concat(HelperCodes, '\n', HelpersText),
    indent_text(HelpersText, "\t", IndentedHelpersText),
    format(string(Code),
'let ~w <- fn(~w): ~w {
\tresult <- @{
\tlocal({
~w
\t    ~w
\t})
\t}@;
\tresult
};', [PredStr, TypedArgList, ReturnType, IndentedHelpersText, HelperCallExpr]).

typr_mutual_helper_code(GroupName, true, Spec, Code) :-
    Kind = Spec.kind,
    Kind == numeric,
    PredStr = Spec.pred_str,
    HelperName = Spec.helper_name,
    NextHelperName = Spec.next_helper_name,
    BaseValues = Spec.base_values,
    GuardExpr = Spec.guard_expr,
    StepExpr = Spec.step_expr,
    typr_mutual_base_branch_lines(GroupName, BaseValues, BaseLines),
    typr_mutual_recursive_branch_lines(GroupName, GuardExpr, NextHelperName, StepExpr, RecursiveLines),
    typr_mutual_false_lines(GroupName, FalseLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    format_string_return('        key <- paste0("~w:", current_input);', [PredStr], KeyLine),
    format_string_return('        if (exists(key, envir=~w_memo, inherits=FALSE)) {', [GroupName], MemoCheckLine),
    format_string_return('            get(key, envir=~w_memo, inherits=FALSE)', [GroupName], MemoGetLine),
    append([HelperLine, KeyLine, MemoCheckLine, MemoGetLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, FalseLines, Lines2),
    append(Lines2, ['    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(_GroupName, false, Spec, Code) :-
    Kind = Spec.kind,
    Kind == numeric,
    HelperName = Spec.helper_name,
    NextHelperName = Spec.next_helper_name,
    BaseValues = Spec.base_values,
    GuardExpr = Spec.guard_expr,
    StepExpr = Spec.step_expr,
    typr_mutual_base_branch_lines_no_memo(BaseValues, BaseLines),
    typr_mutual_recursive_branch_lines_no_memo(GuardExpr, NextHelperName, StepExpr, RecursiveLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    append([HelperLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, ['        } else {', '            FALSE', '        }', '    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(GroupName, true, Spec, Code) :-
    Kind = Spec.kind,
    Kind == list,
    PredStr = Spec.pred_str,
    HelperName = Spec.helper_name,
    NextHelperName = Spec.next_helper_name,
    BaseLengths = Spec.base_lengths,
    GuardExpr = Spec.guard_expr,
    StepExpr = Spec.step_expr,
    typr_mutual_list_base_branch_lines(GroupName, BaseLengths, BaseLines),
    typr_mutual_recursive_branch_lines(GroupName, GuardExpr, NextHelperName, StepExpr, RecursiveLines),
    typr_mutual_false_lines(GroupName, FalseLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    format_string_return('        key <- paste0("~w:", paste(deparse(current_input), collapse=""));', [PredStr], KeyLine),
    format_string_return('        if (exists(key, envir=~w_memo, inherits=FALSE)) {', [GroupName], MemoCheckLine),
    format_string_return('            get(key, envir=~w_memo, inherits=FALSE)', [GroupName], MemoGetLine),
    append([HelperLine, KeyLine, MemoCheckLine, MemoGetLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, FalseLines, Lines2),
    append(Lines2, ['    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(_GroupName, false, Spec, Code) :-
    Kind = Spec.kind,
    Kind == list,
    HelperName = Spec.helper_name,
    NextHelperName = Spec.next_helper_name,
    BaseLengths = Spec.base_lengths,
    GuardExpr = Spec.guard_expr,
    StepExpr = Spec.step_expr,
    typr_mutual_list_base_branch_lines_no_memo(BaseLengths, BaseLines),
    typr_mutual_recursive_branch_lines_no_memo(GuardExpr, NextHelperName, StepExpr, RecursiveLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    append([HelperLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, ['        } else {', '            FALSE', '        }', '    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(GroupName, true, Spec, Code) :-
    Kind = Spec.kind,
    Kind == tree_dual_value_full_body,
    PredStr = Spec.pred_str,
    HelperName = Spec.helper_name,
    BaseCases = Spec.base_cases,
    GuardExpr = Spec.guard_expr,
    Body = Spec.body,
    typr_mutual_tree_value_base_branch_lines(GroupName, BaseCases, BaseLines),
    typr_mutual_tree_dual_value_full_body_recursive_branch_lines(
        GroupName,
        GuardExpr,
        Body,
        RecursiveLines
    ),
    typr_mutual_stop_lines(PredStr, StopLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    typr_mutual_tree_key_line(Spec, PredStr, KeyLine),
    format_string_return('        if (exists(key, envir=~w_memo, inherits=FALSE)) {', [GroupName], MemoCheckLine),
    format_string_return('            get(key, envir=~w_memo, inherits=FALSE)', [GroupName], MemoGetLine),
    append([HelperLine, KeyLine, MemoCheckLine, MemoGetLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, StopLines, Lines2),
    append(Lines2, ['    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(_GroupName, false, Spec, Code) :-
    Kind = Spec.kind,
    Kind == tree_dual_value_full_body,
    PredStr = Spec.pred_str,
    HelperName = Spec.helper_name,
    BaseCases = Spec.base_cases,
    GuardExpr = Spec.guard_expr,
    Body = Spec.body,
    typr_mutual_tree_value_base_branch_lines_no_memo(BaseCases, BaseLines),
    typr_mutual_tree_dual_value_full_body_recursive_branch_lines_no_memo(
        GuardExpr,
        Body,
        RecursiveLines
    ),
    typr_mutual_stop_lines_no_memo(PredStr, StopLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    append([HelperLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, StopLines, Lines2),
    append(Lines2, ['    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(GroupName, true, Spec, Code) :-
    Kind = Spec.kind,
    Kind == tree_dual_value,
    PredStr = Spec.pred_str,
    HelperName = Spec.helper_name,
    BaseCases = Spec.base_cases,
    GuardExpr = Spec.guard_expr,
    LeftCall = Spec.left_call,
    RightCall = Spec.right_call,
    ResultExpr = Spec.result_expr,
    typr_mutual_tree_value_base_branch_lines(GroupName, BaseCases, BaseLines),
    typr_mutual_tree_dual_value_recursive_branch_lines(
        GroupName,
        GuardExpr,
        LeftCall,
        RightCall,
        ResultExpr,
        RecursiveLines
    ),
    typr_mutual_stop_lines(PredStr, StopLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    typr_mutual_tree_key_line(Spec, PredStr, KeyLine),
    format_string_return('        if (exists(key, envir=~w_memo, inherits=FALSE)) {', [GroupName], MemoCheckLine),
    format_string_return('            get(key, envir=~w_memo, inherits=FALSE)', [GroupName], MemoGetLine),
    append([HelperLine, KeyLine, MemoCheckLine, MemoGetLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, StopLines, Lines2),
    append(Lines2, ['    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(_GroupName, false, Spec, Code) :-
    Kind = Spec.kind,
    Kind == tree_dual_value,
    PredStr = Spec.pred_str,
    HelperName = Spec.helper_name,
    BaseCases = Spec.base_cases,
    GuardExpr = Spec.guard_expr,
    LeftCall = Spec.left_call,
    RightCall = Spec.right_call,
    ResultExpr = Spec.result_expr,
    typr_mutual_tree_value_base_branch_lines_no_memo(BaseCases, BaseLines),
    typr_mutual_tree_dual_value_recursive_branch_lines_no_memo(
        GuardExpr,
        LeftCall,
        RightCall,
        ResultExpr,
        RecursiveLines
    ),
    typr_mutual_stop_lines_no_memo(PredStr, StopLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    append([HelperLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, StopLines, Lines2),
    append(Lines2, ['    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(GroupName, true, Spec, Code) :-
    Kind = Spec.kind,
    Kind == tree_dual_value_branch,
    PredStr = Spec.pred_str,
    HelperName = Spec.helper_name,
    BaseCases = Spec.base_cases,
    GuardExpr = Spec.guard_expr,
    BranchConditionExpr = Spec.branch_condition_expr,
    ThenLeftCall = Spec.then_left_call,
    ThenRightCall = Spec.then_right_call,
    ElseLeftCall = Spec.else_left_call,
    ElseRightCall = Spec.else_right_call,
    ThenResultExpr = Spec.then_result_expr,
    ElseResultExpr = Spec.else_result_expr,
    typr_mutual_tree_value_base_branch_lines(GroupName, BaseCases, BaseLines),
    typr_mutual_tree_dual_value_branch_recursive_branch_lines(
        GroupName,
        GuardExpr,
        BranchConditionExpr,
        ThenLeftCall,
        ThenRightCall,
        ElseLeftCall,
        ElseRightCall,
        ThenResultExpr,
        ElseResultExpr,
        RecursiveLines
    ),
    typr_mutual_stop_lines(PredStr, StopLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    typr_mutual_tree_key_line(Spec, PredStr, KeyLine),
    format_string_return('        if (exists(key, envir=~w_memo, inherits=FALSE)) {', [GroupName], MemoCheckLine),
    format_string_return('            get(key, envir=~w_memo, inherits=FALSE)', [GroupName], MemoGetLine),
    append([HelperLine, KeyLine, MemoCheckLine, MemoGetLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, StopLines, Lines2),
    append(Lines2, ['    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(GroupName, true, Spec, Code) :-
    Kind = Spec.kind,
    Kind == tree_dual_value_body,
    PredStr = Spec.pred_str,
    HelperName = Spec.helper_name,
    BaseCases = Spec.base_cases,
    GuardExpr = Spec.guard_expr,
    BranchConditionExpr = Spec.branch_condition_expr,
    ThenBody = Spec.then_body,
    ElseBody = Spec.else_body,
    typr_mutual_tree_value_base_branch_lines(GroupName, BaseCases, BaseLines),
    typr_mutual_tree_dual_value_body_recursive_branch_lines(
        GroupName,
        GuardExpr,
        BranchConditionExpr,
        ThenBody,
        ElseBody,
        RecursiveLines
    ),
    typr_mutual_stop_lines(PredStr, StopLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    typr_mutual_tree_key_line(Spec, PredStr, KeyLine),
    format_string_return('        if (exists(key, envir=~w_memo, inherits=FALSE)) {', [GroupName], MemoCheckLine),
    format_string_return('            get(key, envir=~w_memo, inherits=FALSE)', [GroupName], MemoGetLine),
    append([HelperLine, KeyLine, MemoCheckLine, MemoGetLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, StopLines, Lines2),
    append(Lines2, ['    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(_GroupName, false, Spec, Code) :-
    Kind = Spec.kind,
    Kind == tree_dual_value_branch,
    PredStr = Spec.pred_str,
    HelperName = Spec.helper_name,
    BaseCases = Spec.base_cases,
    GuardExpr = Spec.guard_expr,
    BranchConditionExpr = Spec.branch_condition_expr,
    ThenLeftCall = Spec.then_left_call,
    ThenRightCall = Spec.then_right_call,
    ElseLeftCall = Spec.else_left_call,
    ElseRightCall = Spec.else_right_call,
    ThenResultExpr = Spec.then_result_expr,
    ElseResultExpr = Spec.else_result_expr,
    typr_mutual_tree_value_base_branch_lines_no_memo(BaseCases, BaseLines),
    typr_mutual_tree_dual_value_branch_recursive_branch_lines_no_memo(
        GuardExpr,
        BranchConditionExpr,
        ThenLeftCall,
        ThenRightCall,
        ElseLeftCall,
        ElseRightCall,
        ThenResultExpr,
        ElseResultExpr,
        RecursiveLines
    ),
    typr_mutual_stop_lines_no_memo(PredStr, StopLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    append([HelperLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, StopLines, Lines2),
    append(Lines2, ['    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(_GroupName, false, Spec, Code) :-
    Kind = Spec.kind,
    Kind == tree_dual_value_body,
    PredStr = Spec.pred_str,
    HelperName = Spec.helper_name,
    BaseCases = Spec.base_cases,
    GuardExpr = Spec.guard_expr,
    BranchConditionExpr = Spec.branch_condition_expr,
    ThenBody = Spec.then_body,
    ElseBody = Spec.else_body,
    typr_mutual_tree_value_base_branch_lines_no_memo(BaseCases, BaseLines),
    typr_mutual_tree_dual_value_body_recursive_branch_lines_no_memo(
        GuardExpr,
        BranchConditionExpr,
        ThenBody,
        ElseBody,
        RecursiveLines
    ),
    typr_mutual_stop_lines_no_memo(PredStr, StopLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    append([HelperLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, StopLines, Lines2),
    append(Lines2, ['    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(GroupName, true, Spec, Code) :-
    Kind = Spec.kind,
    Kind == tree_dual_branch,
    PredStr = Spec.pred_str,
    HelperName = Spec.helper_name,
    BaseConditions = Spec.base_conditions,
    GuardExpr = Spec.guard_expr,
    BranchConditionExpr = Spec.branch_condition_expr,
    ThenBody = Spec.then_body,
    ElseBody = Spec.else_body,
    typr_mutual_tree_base_branch_lines(GroupName, BaseConditions, BaseLines),
    typr_mutual_tree_dual_branch_recursive_branch_lines(
        GroupName,
        GuardExpr,
        BranchConditionExpr,
        ThenBody,
        ElseBody,
        RecursiveLines
    ),
    typr_mutual_false_lines(GroupName, FalseLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    typr_mutual_tree_key_line(Spec, PredStr, KeyLine),
    format_string_return('        if (exists(key, envir=~w_memo, inherits=FALSE)) {', [GroupName], MemoCheckLine),
    format_string_return('            get(key, envir=~w_memo, inherits=FALSE)', [GroupName], MemoGetLine),
    append([HelperLine, KeyLine, MemoCheckLine, MemoGetLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, FalseLines, Lines2),
    append(Lines2, ['    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(_GroupName, false, Spec, Code) :-
    Kind = Spec.kind,
    Kind == tree_dual_branch,
    HelperName = Spec.helper_name,
    BaseConditions = Spec.base_conditions,
    GuardExpr = Spec.guard_expr,
    BranchConditionExpr = Spec.branch_condition_expr,
    ThenBody = Spec.then_body,
    ElseBody = Spec.else_body,
    typr_mutual_tree_base_branch_lines_no_memo(BaseConditions, BaseLines),
    typr_mutual_tree_dual_branch_recursive_branch_lines_no_memo(
        GuardExpr,
        BranchConditionExpr,
        ThenBody,
        ElseBody,
        RecursiveLines
    ),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    append([HelperLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, ['        } else {', '            FALSE', '        }', '    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(GroupName, true, Spec, Code) :-
    Kind = Spec.kind,
    Kind == tree_dual_body,
    PredStr = Spec.pred_str,
    HelperName = Spec.helper_name,
    BaseConditions = Spec.base_conditions,
    GuardExpr = Spec.guard_expr,
    Body = Spec.body,
    typr_mutual_tree_base_branch_lines(GroupName, BaseConditions, BaseLines),
    typr_mutual_tree_dual_body_recursive_branch_lines(
        GroupName,
        GuardExpr,
        Body,
        RecursiveLines
    ),
    typr_mutual_false_lines(GroupName, FalseLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    typr_mutual_tree_key_line(Spec, PredStr, KeyLine),
    format_string_return('        if (exists(key, envir=~w_memo, inherits=FALSE)) {', [GroupName], MemoCheckLine),
    format_string_return('            get(key, envir=~w_memo, inherits=FALSE)', [GroupName], MemoGetLine),
    append([HelperLine, KeyLine, MemoCheckLine, MemoGetLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, FalseLines, Lines2),
    append(Lines2, ['    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(_GroupName, false, Spec, Code) :-
    Kind = Spec.kind,
    Kind == tree_dual_body,
    HelperName = Spec.helper_name,
    BaseConditions = Spec.base_conditions,
    GuardExpr = Spec.guard_expr,
    Body = Spec.body,
    typr_mutual_tree_base_branch_lines_no_memo(BaseConditions, BaseLines),
    typr_mutual_tree_dual_body_recursive_branch_lines_no_memo(
        GuardExpr,
        Body,
        RecursiveLines
    ),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    append([HelperLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, ['        } else {', '            FALSE', '        }', '    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(GroupName, true, Spec, Code) :-
    Kind = Spec.kind,
    Kind == tree_dual,
    PredStr = Spec.pred_str,
    HelperName = Spec.helper_name,
    LeftCall = Spec.left_call,
    RightCall = Spec.right_call,
    BaseConditions = Spec.base_conditions,
    GuardExpr = Spec.guard_expr,
    typr_mutual_tree_base_branch_lines(GroupName, BaseConditions, BaseLines),
    typr_mutual_tree_dual_recursive_branch_lines(
        GroupName,
        GuardExpr,
        LeftCall,
        RightCall,
        RecursiveLines
    ),
    typr_mutual_false_lines(GroupName, FalseLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    typr_mutual_tree_key_line(Spec, PredStr, KeyLine),
    format_string_return('        if (exists(key, envir=~w_memo, inherits=FALSE)) {', [GroupName], MemoCheckLine),
    format_string_return('            get(key, envir=~w_memo, inherits=FALSE)', [GroupName], MemoGetLine),
    append([HelperLine, KeyLine, MemoCheckLine, MemoGetLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, FalseLines, Lines2),
    append(Lines2, ['    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(_GroupName, false, Spec, Code) :-
    Kind = Spec.kind,
    Kind == tree_dual,
    HelperName = Spec.helper_name,
    LeftCall = Spec.left_call,
    RightCall = Spec.right_call,
    BaseConditions = Spec.base_conditions,
    GuardExpr = Spec.guard_expr,
    typr_mutual_tree_base_branch_lines_no_memo(BaseConditions, BaseLines),
    typr_mutual_tree_dual_recursive_branch_lines_no_memo(
        GuardExpr,
        LeftCall,
        RightCall,
        RecursiveLines
    ),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    append([HelperLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, ['        } else {', '            FALSE', '        }', '    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(GroupName, true, Spec, Code) :-
    Kind = Spec.kind,
    Kind == tree,
    PredStr = Spec.pred_str,
    HelperName = Spec.helper_name,
    NextHelperName = Spec.next_helper_name,
    BaseConditions = Spec.base_conditions,
    GuardExpr = Spec.guard_expr,
    StepExpr = Spec.step_expr,
    typr_mutual_tree_base_branch_lines(GroupName, BaseConditions, BaseLines),
    typr_mutual_recursive_branch_lines(GroupName, GuardExpr, NextHelperName, StepExpr, RecursiveLines),
    typr_mutual_false_lines(GroupName, FalseLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    typr_mutual_tree_key_line(Spec, PredStr, KeyLine),
    format_string_return('        if (exists(key, envir=~w_memo, inherits=FALSE)) {', [GroupName], MemoCheckLine),
    format_string_return('            get(key, envir=~w_memo, inherits=FALSE)', [GroupName], MemoGetLine),
    append([HelperLine, KeyLine, MemoCheckLine, MemoGetLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, FalseLines, Lines2),
    append(Lines2, ['    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).
typr_mutual_helper_code(_GroupName, false, Spec, Code) :-
    Kind = Spec.kind,
    Kind == tree,
    HelperName = Spec.helper_name,
    NextHelperName = Spec.next_helper_name,
    BaseConditions = Spec.base_conditions,
    GuardExpr = Spec.guard_expr,
    StepExpr = Spec.step_expr,
    typr_mutual_tree_base_branch_lines_no_memo(BaseConditions, BaseLines),
    typr_mutual_recursive_branch_lines_no_memo(GuardExpr, NextHelperName, StepExpr, RecursiveLines),
    typr_mutual_helper_line(Spec, HelperName, HelperLine),
    append([HelperLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, ['        } else {', '            FALSE', '        }', '    };'], Lines),
    atomic_list_concat(Lines, '\n', Code).

format_string_return(Format, Args, String) :-
    format(string(String), Format, Args).

typr_mutual_base_branch_lines(GroupName, [BaseValue|Rest], Lines) :-
    format(string(IfLine), '        } else if (identical(current_input, ~w)) {', [BaseValue]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    Lines = [IfLine, '            result = TRUE;', AssignLine, '            result'|RestLines],
    typr_mutual_base_branch_lines_rest(GroupName, Rest, RestLines).

typr_mutual_base_branch_lines_rest(_GroupName, [], []).
typr_mutual_base_branch_lines_rest(GroupName, [BaseValue|Rest], [
    IfLine, '            result = TRUE;', AssignLine, '            result'|RestLines
]) :-
    format(string(IfLine), '        } else if (identical(current_input, ~w)) {', [BaseValue]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    typr_mutual_base_branch_lines_rest(GroupName, Rest, RestLines).

typr_mutual_base_branch_lines_no_memo([BaseValue|Rest], Lines) :-
    format(string(IfLine), '        if (identical(current_input, ~w)) {', [BaseValue]),
    Lines = [IfLine, '            TRUE'|RestLines],
    typr_mutual_base_branch_lines_no_memo_rest(Rest, RestLines).

typr_mutual_base_branch_lines_no_memo_rest([], []).
typr_mutual_base_branch_lines_no_memo_rest([BaseValue|Rest], [IfLine, '            TRUE'|RestLines]) :-
    format(string(IfLine), '        } else if (identical(current_input, ~w)) {', [BaseValue]),
    typr_mutual_base_branch_lines_no_memo_rest(Rest, RestLines).

typr_mutual_list_base_branch_lines(GroupName, [BaseLength|Rest], Lines) :-
    format(string(IfLine), '        } else if (length(current_input) == ~w) {', [BaseLength]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    Lines = [IfLine, '            result = TRUE;', AssignLine, '            result'|RestLines],
    typr_mutual_list_base_branch_lines_rest(GroupName, Rest, RestLines).

typr_mutual_list_base_branch_lines_rest(_GroupName, [], []).
typr_mutual_list_base_branch_lines_rest(GroupName, [BaseLength|Rest], [
    IfLine, '            result = TRUE;', AssignLine, '            result'|RestLines
]) :-
    format(string(IfLine), '        } else if (length(current_input) == ~w) {', [BaseLength]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    typr_mutual_list_base_branch_lines_rest(GroupName, Rest, RestLines).

typr_mutual_list_base_branch_lines_no_memo([BaseLength|Rest], Lines) :-
    format(string(IfLine), '        if (length(current_input) == ~w) {', [BaseLength]),
    Lines = [IfLine, '            TRUE'|RestLines],
    typr_mutual_list_base_branch_lines_no_memo_rest(Rest, RestLines).

typr_mutual_list_base_branch_lines_no_memo_rest([], []).
typr_mutual_list_base_branch_lines_no_memo_rest([BaseLength|Rest], [IfLine, '            TRUE'|RestLines]) :-
    format(string(IfLine), '        } else if (length(current_input) == ~w) {', [BaseLength]),
    typr_mutual_list_base_branch_lines_no_memo_rest(Rest, RestLines).

typr_mutual_tree_base_branch_lines(GroupName, [BaseCondition|Rest], Lines) :-
    format(string(IfLine), '        } else if (~w) {', [BaseCondition]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    Lines = [IfLine, '            result = TRUE;', AssignLine, '            result'|RestLines],
    typr_mutual_tree_base_branch_lines_rest(GroupName, Rest, RestLines).

typr_mutual_tree_base_branch_lines_rest(_GroupName, [], []).
typr_mutual_tree_base_branch_lines_rest(GroupName, [BaseCondition|Rest], [
    IfLine, '            result = TRUE;', AssignLine, '            result'|RestLines
]) :-
    format(string(IfLine), '        } else if (~w) {', [BaseCondition]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    typr_mutual_tree_base_branch_lines_rest(GroupName, Rest, RestLines).

typr_mutual_tree_base_branch_lines_no_memo([BaseCondition|Rest], Lines) :-
    format(string(IfLine), '        if (~w) {', [BaseCondition]),
    Lines = [IfLine, '            TRUE'|RestLines],
    typr_mutual_tree_base_branch_lines_no_memo_rest(Rest, RestLines).

typr_mutual_tree_base_branch_lines_no_memo_rest([], []).
typr_mutual_tree_base_branch_lines_no_memo_rest([BaseCondition|Rest], [IfLine, '            TRUE'|RestLines]) :-
    format(string(IfLine), '        } else if (~w) {', [BaseCondition]),
    typr_mutual_tree_base_branch_lines_no_memo_rest(Rest, RestLines).

typr_mutual_tree_value_base_branch_lines(GroupName, [base_case(BaseCondition, BaseExpr)|Rest], Lines) :-
    format(string(IfLine), '        } else if (~w) {', [BaseCondition]),
    format(string(ResultLine), '            result = ~w;', [BaseExpr]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    Lines = [IfLine, ResultLine, AssignLine, '            result'|RestLines],
    typr_mutual_tree_value_base_branch_lines_rest(GroupName, Rest, RestLines).

typr_mutual_tree_value_base_branch_lines_rest(_GroupName, [], []).
typr_mutual_tree_value_base_branch_lines_rest(GroupName, [base_case(BaseCondition, BaseExpr)|Rest], [
    IfLine, ResultLine, AssignLine, '            result'|RestLines
]) :-
    format(string(IfLine), '        } else if (~w) {', [BaseCondition]),
    format(string(ResultLine), '            result = ~w;', [BaseExpr]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    typr_mutual_tree_value_base_branch_lines_rest(GroupName, Rest, RestLines).

typr_mutual_tree_value_base_branch_lines_no_memo([base_case(BaseCondition, BaseExpr)|Rest], Lines) :-
    format(string(IfLine), '        if (~w) {', [BaseCondition]),
    format(string(ResultLine), '            ~w', [BaseExpr]),
    Lines = [IfLine, ResultLine|RestLines],
    typr_mutual_tree_value_base_branch_lines_no_memo_rest(Rest, RestLines).

typr_mutual_tree_value_base_branch_lines_no_memo_rest([], []).
typr_mutual_tree_value_base_branch_lines_no_memo_rest([base_case(BaseCondition, BaseExpr)|Rest], [IfLine, ResultLine|RestLines]) :-
    format(string(IfLine), '        } else if (~w) {', [BaseCondition]),
    format(string(ResultLine), '            ~w', [BaseExpr]),
    typr_mutual_tree_value_base_branch_lines_no_memo_rest(Rest, RestLines).

typr_mutual_recursive_branch_lines(GroupName, 'TRUE', NextHelperName, StepExpr, Lines) :-
    !,
    format(string(ResultLine), '            result = ~w(~w);', [NextHelperName, StepExpr]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    Lines = [
        '        } else {',
        ResultLine,
        AssignLine,
        '            result'
    ].
typr_mutual_recursive_branch_lines(GroupName, GuardExpr, NextHelperName, StepExpr, Lines) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    format(string(ResultLine), '            result = ~w(~w);', [NextHelperName, StepExpr]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    Lines = [
        IfLine,
        ResultLine,
        AssignLine,
        '            result'
    ].

typr_mutual_recursive_branch_lines_no_memo('TRUE', NextHelperName, StepExpr, Lines) :-
    !,
    format(string(ResultLine), '            ~w(~w)', [NextHelperName, StepExpr]),
    Lines = [
        '        } else {',
        ResultLine
    ].
typr_mutual_recursive_branch_lines_no_memo(GuardExpr, NextHelperName, StepExpr, Lines) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    format(string(ResultLine), '            ~w(~w)', [NextHelperName, StepExpr]),
    Lines = [
        IfLine,
        ResultLine
    ].

typr_mutual_tree_dual_recursive_branch_lines(
    GroupName,
    'TRUE',
    LeftCall,
    RightCall,
    Lines
) :-
    !,
    typr_mutual_tree_call_expr(LeftCall, LeftExpr),
    typr_mutual_tree_call_expr(RightCall, RightExpr),
    format(string(LeftLine), '            left_result = ~w;', [LeftExpr]),
    format(string(RightLine), '            right_result = ~w;', [RightExpr]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    Lines = [
        '        } else {',
        LeftLine,
        RightLine,
        '            result = left_result && right_result;',
        AssignLine,
        '            result'
    ].
typr_mutual_tree_dual_recursive_branch_lines(
    GroupName,
    GuardExpr,
    LeftCall,
    RightCall,
    Lines
) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    typr_mutual_tree_call_expr(LeftCall, LeftExpr),
    typr_mutual_tree_call_expr(RightCall, RightExpr),
    format(string(LeftLine), '            left_result = ~w;', [LeftExpr]),
    format(string(RightLine), '            right_result = ~w;', [RightExpr]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    Lines = [
        IfLine,
        LeftLine,
        RightLine,
        '            result = left_result && right_result;',
        AssignLine,
        '            result'
    ].

typr_mutual_tree_dual_recursive_branch_lines_no_memo(
    'TRUE',
    LeftCall,
    RightCall,
    Lines
) :-
    !,
    typr_mutual_tree_call_expr(LeftCall, LeftExpr),
    typr_mutual_tree_call_expr(RightCall, RightExpr),
    format(string(LeftLine), '            left_result = ~w;', [LeftExpr]),
    format(string(RightLine), '            right_result = ~w;', [RightExpr]),
    Lines = [
        '        } else {',
        LeftLine,
        RightLine,
        '            left_result && right_result'
    ].
typr_mutual_tree_dual_recursive_branch_lines_no_memo(
    GuardExpr,
    LeftCall,
    RightCall,
    Lines
) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    typr_mutual_tree_call_expr(LeftCall, LeftExpr),
    typr_mutual_tree_call_expr(RightCall, RightExpr),
    format(string(LeftLine), '            left_result = ~w;', [LeftExpr]),
    format(string(RightLine), '            right_result = ~w;', [RightExpr]),
    Lines = [
        IfLine,
        LeftLine,
        RightLine,
        '            left_result && right_result'
    ].

typr_mutual_tree_dual_value_recursive_branch_lines(
    GroupName,
    'TRUE',
    LeftCall,
    RightCall,
    ResultExpr,
    Lines
) :-
    !,
    typr_mutual_tree_call_expr(LeftCall, LeftExpr),
    typr_mutual_tree_call_expr(RightCall, RightExpr),
    format(string(LeftLine), '            left_result = ~w;', [LeftExpr]),
    format(string(RightLine), '            right_result = ~w;', [RightExpr]),
    format(string(ResultLine), '            result = ~w;', [ResultExpr]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    Lines = [
        '        } else {',
        LeftLine,
        RightLine,
        ResultLine,
        AssignLine,
        '            result'
    ].
typr_mutual_tree_dual_value_recursive_branch_lines(
    GroupName,
    GuardExpr,
    LeftCall,
    RightCall,
    ResultExpr,
    Lines
) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    typr_mutual_tree_call_expr(LeftCall, LeftExpr),
    typr_mutual_tree_call_expr(RightCall, RightExpr),
    format(string(LeftLine), '            left_result = ~w;', [LeftExpr]),
    format(string(RightLine), '            right_result = ~w;', [RightExpr]),
    format(string(ResultLine), '            result = ~w;', [ResultExpr]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    Lines = [
        IfLine,
        LeftLine,
        RightLine,
        ResultLine,
        AssignLine,
        '            result'
    ].

typr_mutual_tree_dual_value_recursive_branch_lines_no_memo(
    'TRUE',
    LeftCall,
    RightCall,
    ResultExpr,
    Lines
) :-
    !,
    typr_mutual_tree_call_expr(LeftCall, LeftExpr),
    typr_mutual_tree_call_expr(RightCall, RightExpr),
    format(string(LeftLine), '            left_result = ~w;', [LeftExpr]),
    format(string(RightLine), '            right_result = ~w;', [RightExpr]),
    format(string(ResultLine), '            ~w', [ResultExpr]),
    Lines = [
        '        } else {',
        LeftLine,
        RightLine,
        ResultLine
    ].
typr_mutual_tree_dual_value_recursive_branch_lines_no_memo(
    GuardExpr,
    LeftCall,
    RightCall,
    ResultExpr,
    Lines
) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    typr_mutual_tree_call_expr(LeftCall, LeftExpr),
    typr_mutual_tree_call_expr(RightCall, RightExpr),
    format(string(LeftLine), '            left_result = ~w;', [LeftExpr]),
    format(string(RightLine), '            right_result = ~w;', [RightExpr]),
    format(string(ResultLine), '            ~w', [ResultExpr]),
    Lines = [
        IfLine,
        LeftLine,
        RightLine,
        ResultLine
    ].

typr_mutual_tree_dual_value_full_body_recursive_branch_lines(
    GroupName,
    'TRUE',
    Body,
    Lines
) :-
    !,
    typr_mutual_tree_value_body_lines(Body, '            ', BodyLines),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    append(['        } else {'], BodyLines, Lines0),
    append(Lines0, [AssignLine, '            result'], Lines).
typr_mutual_tree_dual_value_full_body_recursive_branch_lines(
    GroupName,
    GuardExpr,
    Body,
    Lines
) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    typr_mutual_tree_value_body_lines(Body, '            ', BodyLines),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    append([IfLine], BodyLines, Lines0),
    append(Lines0, [AssignLine, '            result'], Lines).

typr_mutual_tree_dual_value_full_body_recursive_branch_lines_no_memo(
    'TRUE',
    Body,
    Lines
) :-
    !,
    typr_mutual_tree_value_body_lines_no_memo(Body, '            ', BodyLines),
    append(['        } else {'], BodyLines, Lines).
typr_mutual_tree_dual_value_full_body_recursive_branch_lines_no_memo(
    GuardExpr,
    Body,
    Lines
) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    typr_mutual_tree_value_body_lines_no_memo(Body, '            ', BodyLines),
    append([IfLine], BodyLines, Lines).

typr_mutual_tree_dual_value_branch_recursive_branch_lines(
    GroupName,
    'TRUE',
    BranchConditionExpr,
    ThenLeftCall,
    ThenRightCall,
    ElseLeftCall,
    ElseRightCall,
    ThenResultExpr,
    ElseResultExpr,
    Lines
) :-
    !,
    typr_mutual_tree_call_expr(ThenLeftCall, ThenLeftExpr),
    typr_mutual_tree_call_expr(ThenRightCall, ThenRightExpr),
    typr_mutual_tree_call_expr(ElseLeftCall, ElseLeftExpr),
    typr_mutual_tree_call_expr(ElseRightCall, ElseRightExpr),
    format(string(BranchIfLine), '            if (~w) {', [BranchConditionExpr]),
    format(string(ThenLeftLine), '                left_result = ~w;', [ThenLeftExpr]),
    format(string(ThenRightLine), '                right_result = ~w;', [ThenRightExpr]),
    format(string(ThenResultLine), '                result = ~w;', [ThenResultExpr]),
    format(string(ElseLeftLine), '                left_result = ~w;', [ElseLeftExpr]),
    format(string(ElseRightLine), '                right_result = ~w;', [ElseRightExpr]),
    format(string(ElseResultLine), '                result = ~w;', [ElseResultExpr]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    Lines = [
        '        } else {',
        BranchIfLine,
        ThenLeftLine,
        ThenRightLine,
        ThenResultLine,
        '            } else {',
        ElseLeftLine,
        ElseRightLine,
        ElseResultLine,
        '            }',
        AssignLine,
        '            result'
    ].
typr_mutual_tree_dual_value_branch_recursive_branch_lines(
    GroupName,
    GuardExpr,
    BranchConditionExpr,
    ThenLeftCall,
    ThenRightCall,
    ElseLeftCall,
    ElseRightCall,
    ThenResultExpr,
    ElseResultExpr,
    Lines
) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    typr_mutual_tree_call_expr(ThenLeftCall, ThenLeftExpr),
    typr_mutual_tree_call_expr(ThenRightCall, ThenRightExpr),
    typr_mutual_tree_call_expr(ElseLeftCall, ElseLeftExpr),
    typr_mutual_tree_call_expr(ElseRightCall, ElseRightExpr),
    format(string(BranchIfLine), '            if (~w) {', [BranchConditionExpr]),
    format(string(ThenLeftLine), '                left_result = ~w;', [ThenLeftExpr]),
    format(string(ThenRightLine), '                right_result = ~w;', [ThenRightExpr]),
    format(string(ThenResultLine), '                result = ~w;', [ThenResultExpr]),
    format(string(ElseLeftLine), '                left_result = ~w;', [ElseLeftExpr]),
    format(string(ElseRightLine), '                right_result = ~w;', [ElseRightExpr]),
    format(string(ElseResultLine), '                result = ~w;', [ElseResultExpr]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    Lines = [
        IfLine,
        BranchIfLine,
        ThenLeftLine,
        ThenRightLine,
        ThenResultLine,
        '            } else {',
        ElseLeftLine,
        ElseRightLine,
        ElseResultLine,
        '            }',
        AssignLine,
        '            result'
    ].

typr_mutual_tree_dual_value_branch_recursive_branch_lines_no_memo(
    'TRUE',
    BranchConditionExpr,
    ThenLeftCall,
    ThenRightCall,
    ElseLeftCall,
    ElseRightCall,
    ThenResultExpr,
    ElseResultExpr,
    Lines
) :-
    !,
    typr_mutual_tree_call_expr(ThenLeftCall, ThenLeftExpr),
    typr_mutual_tree_call_expr(ThenRightCall, ThenRightExpr),
    typr_mutual_tree_call_expr(ElseLeftCall, ElseLeftExpr),
    typr_mutual_tree_call_expr(ElseRightCall, ElseRightExpr),
    format(string(BranchIfLine), '            if (~w) {', [BranchConditionExpr]),
    format(string(ThenLeftLine), '                left_result = ~w;', [ThenLeftExpr]),
    format(string(ThenRightLine), '                right_result = ~w;', [ThenRightExpr]),
    format(string(ThenResultLine), '                ~w', [ThenResultExpr]),
    format(string(ElseLeftLine), '                left_result = ~w;', [ElseLeftExpr]),
    format(string(ElseRightLine), '                right_result = ~w;', [ElseRightExpr]),
    format(string(ElseResultLine), '                ~w', [ElseResultExpr]),
    Lines = [
        '        } else {',
        BranchIfLine,
        ThenLeftLine,
        ThenRightLine,
        ThenResultLine,
        '            } else {',
        ElseLeftLine,
        ElseRightLine,
        ElseResultLine,
        '            }'
    ].
typr_mutual_tree_dual_value_branch_recursive_branch_lines_no_memo(
    GuardExpr,
    BranchConditionExpr,
    ThenLeftCall,
    ThenRightCall,
    ElseLeftCall,
    ElseRightCall,
    ThenResultExpr,
    ElseResultExpr,
    Lines
) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    typr_mutual_tree_call_expr(ThenLeftCall, ThenLeftExpr),
    typr_mutual_tree_call_expr(ThenRightCall, ThenRightExpr),
    typr_mutual_tree_call_expr(ElseLeftCall, ElseLeftExpr),
    typr_mutual_tree_call_expr(ElseRightCall, ElseRightExpr),
    format(string(BranchIfLine), '            if (~w) {', [BranchConditionExpr]),
    format(string(ThenLeftLine), '                left_result = ~w;', [ThenLeftExpr]),
    format(string(ThenRightLine), '                right_result = ~w;', [ThenRightExpr]),
    format(string(ThenResultLine), '                ~w', [ThenResultExpr]),
    format(string(ElseLeftLine), '                left_result = ~w;', [ElseLeftExpr]),
    format(string(ElseRightLine), '                right_result = ~w;', [ElseRightExpr]),
    format(string(ElseResultLine), '                ~w', [ElseResultExpr]),
    Lines = [
        IfLine,
        BranchIfLine,
        ThenLeftLine,
        ThenRightLine,
        ThenResultLine,
        '            } else {',
        ElseLeftLine,
        ElseRightLine,
        ElseResultLine,
        '            }'
    ].

typr_mutual_tree_dual_value_body_recursive_branch_lines(
    GroupName,
    GuardExpr,
    BranchConditionExpr,
    ThenBody,
    ElseBody,
    Lines
) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    format(string(BranchIfLine), '            if (~w) {', [BranchConditionExpr]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    typr_mutual_tree_value_body_lines(ThenBody, '                ', ThenLines),
    typr_mutual_tree_value_body_lines(ElseBody, '                ', ElseLines),
    append([IfLine, BranchIfLine], ThenLines, Lines0),
    append(Lines0, ['            } else {'], Lines1),
    append(Lines1, ElseLines, Lines2),
    append(Lines2, ['            }', AssignLine, '            result'], Lines).

typr_mutual_tree_dual_value_body_recursive_branch_lines_no_memo(
    GuardExpr,
    BranchConditionExpr,
    ThenBody,
    ElseBody,
    Lines
) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    format(string(BranchIfLine), '            if (~w) {', [BranchConditionExpr]),
    typr_mutual_tree_value_body_lines_no_memo(ThenBody, '                ', ThenLines),
    typr_mutual_tree_value_body_lines_no_memo(ElseBody, '                ', ElseLines),
    append([IfLine, BranchIfLine], ThenLines, Lines0),
    append(Lines0, ['            } else {'], Lines1),
    append(Lines1, ElseLines, Lines2),
    append(Lines2, ['            }'], Lines).

typr_mutual_tree_dual_branch_recursive_branch_lines(
    GroupName,
    GuardExpr,
    BranchConditionExpr,
    ThenBody,
    ElseBody,
    Lines
) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    format(string(BranchIfLine), '            if (~w) {', [BranchConditionExpr]),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    typr_mutual_tree_branch_body_lines(ThenBody, '                ', ThenLines),
    typr_mutual_tree_branch_body_lines(ElseBody, '                ', ElseLines),
    append([IfLine, BranchIfLine], ThenLines, Lines0),
    append(Lines0, ['            } else {'], Lines1),
    append(Lines1, ElseLines, Lines2),
    append(Lines2, ['            }', AssignLine, '            result'], Lines).

typr_mutual_tree_dual_branch_recursive_branch_lines_no_memo(
    GuardExpr,
    BranchConditionExpr,
    ThenBody,
    ElseBody,
    Lines
) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    format(string(BranchIfLine), '            if (~w) {', [BranchConditionExpr]),
    typr_mutual_tree_branch_body_lines_no_memo(ThenBody, '                ', ThenLines),
    typr_mutual_tree_branch_body_lines_no_memo(ElseBody, '                ', ElseLines),
    append([IfLine, BranchIfLine], ThenLines, Lines0),
    append(Lines0, ['            } else {'], Lines1),
    append(Lines1, ElseLines, Lines2),
    append(Lines2, ['            }'], Lines).

typr_mutual_tree_dual_body_recursive_branch_lines(
    GroupName,
    'TRUE',
    Body,
    Lines
) :-
    !,
    typr_mutual_tree_branch_body_lines(Body, '            ', BodyLines),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    append(['        } else {'], BodyLines, Lines0),
    append(Lines0, [AssignLine, '            result'], Lines).
typr_mutual_tree_dual_body_recursive_branch_lines(
    GroupName,
    GuardExpr,
    Body,
    Lines
) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    typr_mutual_tree_branch_body_lines(Body, '            ', BodyLines),
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]),
    append([IfLine], BodyLines, Lines0),
    append(Lines0, [AssignLine, '            result'], Lines).

typr_mutual_tree_dual_body_recursive_branch_lines_no_memo(
    'TRUE',
    Body,
    Lines
) :-
    !,
    typr_mutual_tree_branch_body_lines_no_memo(Body, '            ', BodyLines),
    append(['        } else {'], BodyLines, Lines).
typr_mutual_tree_dual_body_recursive_branch_lines_no_memo(
    GuardExpr,
    Body,
    Lines
) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    typr_mutual_tree_branch_body_lines_no_memo(Body, '            ', BodyLines),
    append([IfLine], BodyLines, Lines).

typr_mutual_tree_value_body_lines(Body, Prefix, Lines) :-
    typr_mutual_tree_value_body_lines_from(Body, Prefix, Lines).

typr_mutual_tree_value_body_lines_from(value_branch_if(BranchConditionExpr, ThenBody, ElseBody), Prefix, Lines) :-
    format(string(IfLine), '~wif (~w) {', [Prefix, BranchConditionExpr]),
    atom_concat(Prefix, '    ', NestedPrefix),
    typr_mutual_tree_value_body_lines_from(ThenBody, NestedPrefix, ThenLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    typr_mutual_tree_value_body_lines_from(ElseBody, NestedPrefix, ElseLines),
    format(string(EndLine), '~w}', [Prefix]),
    append([IfLine], ThenLines, Lines0),
    append(Lines0, [ElseLine], Lines1),
    append(Lines1, ElseLines, Lines2),
    append(Lines2, [EndLine], Lines).
typr_mutual_tree_value_body_lines_from(value_branch_leaf(CallBindings, ResultExpr), Prefix, Lines) :-
    typr_mutual_tree_value_call_binding_lines(CallBindings, Prefix, CallLines),
    format(string(ResultLine), '~wresult = ~w;', [Prefix, ResultExpr]),
    append(CallLines, [ResultLine], Lines).

typr_mutual_tree_value_body_lines_no_memo(Body, Prefix, Lines) :-
    typr_mutual_tree_value_body_lines_no_memo_from(Body, Prefix, Lines).

typr_mutual_tree_value_body_lines_no_memo_from(value_branch_if(BranchConditionExpr, ThenBody, ElseBody), Prefix, Lines) :-
    format(string(IfLine), '~wif (~w) {', [Prefix, BranchConditionExpr]),
    atom_concat(Prefix, '    ', NestedPrefix),
    typr_mutual_tree_value_body_lines_no_memo_from(ThenBody, NestedPrefix, ThenLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    typr_mutual_tree_value_body_lines_no_memo_from(ElseBody, NestedPrefix, ElseLines),
    format(string(EndLine), '~w}', [Prefix]),
    append([IfLine], ThenLines, Lines0),
    append(Lines0, [ElseLine], Lines1),
    append(Lines1, ElseLines, Lines2),
    append(Lines2, [EndLine], Lines).
typr_mutual_tree_value_body_lines_no_memo_from(value_branch_leaf(CallBindings, ResultExpr), Prefix, Lines) :-
    typr_mutual_tree_value_call_binding_lines(CallBindings, Prefix, CallLines),
    format(string(ResultLine), '~w~w', [Prefix, ResultExpr]),
    append(CallLines, [ResultLine], Lines).

typr_mutual_tree_value_call_binding_lines([], _Prefix, []).
typr_mutual_tree_value_call_binding_lines([value_call_binding(Side, Call)|Rest], Prefix, [CallLine|RestLines]) :-
    typr_mutual_tree_result_name(Side, ResultName),
    typr_mutual_tree_call_expr(Call, CallExpr),
    format(string(CallLine), '~w~w = ~w;', [Prefix, ResultName, CallExpr]),
    typr_mutual_tree_value_call_binding_lines(Rest, Prefix, RestLines).

typr_mutual_tree_branch_body_lines(Body, Prefix, Lines) :-
    typr_mutual_tree_branch_body_lines_from(Body, Prefix, 1, Lines).

typr_mutual_tree_branch_body_lines_from(branch_after_call(BranchCall, BranchConditionExpr, ThenBody, ElseBody), Prefix, Index, Lines) :-
    typr_mutual_tree_step_result_name(Index, ResultName),
    typr_mutual_tree_call_expr(BranchCall, CallExpr),
    format(string(CallLine), '~w~w = ~w;', [Prefix, ResultName, CallExpr]),
    format(string(IfLine), '~wif (~w) {', [Prefix, ResultName]),
    atom_concat(Prefix, '    ', BranchPrefix),
    format(string(BranchIfLine), '~wif (~w) {', [BranchPrefix, BranchConditionExpr]),
    atom_concat(BranchPrefix, '    ', NestedPrefix),
    NextIndex is Index + 1,
    typr_mutual_tree_branch_body_lines_from(ThenBody, NestedPrefix, NextIndex, ThenLines),
    format(string(BranchElseLine), '~w} else {', [BranchPrefix]),
    typr_mutual_tree_branch_body_lines_from(ElseBody, NestedPrefix, NextIndex, ElseLines),
    format(string(BranchEndLine), '~w}', [BranchPrefix]),
    format(string(PrefixElseLine), '~w} else {', [Prefix]),
    format(string(FalseLine), '~w    result = FALSE;', [Prefix]),
    format(string(EndLine), '~w}', [Prefix]),
    append([CallLine, IfLine, BranchIfLine], ThenLines, Lines0),
    append(Lines0, [BranchElseLine], Lines1),
    append(Lines1, ElseLines, Lines2),
    append(Lines2, [BranchEndLine, PrefixElseLine, FalseLine, EndLine], Lines).
typr_mutual_tree_branch_body_lines_from(branch_after_calls(Calls, BranchConditionExpr, ThenBody, ElseBody), Prefix, _Index, Lines) :-
    typr_mutual_tree_dual_branch_call_prefix_lines(Calls, Prefix, CallLines),
    format(string(IfLine), '~wif (~w) {', [Prefix, BranchConditionExpr]),
    atom_concat(Prefix, '    ', NestedPrefix),
    typr_mutual_tree_branch_body_lines_from(ThenBody, NestedPrefix, 3, ThenLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    typr_mutual_tree_branch_body_lines_from(ElseBody, NestedPrefix, 3, ElseLines),
    format(string(EndLine), '~w}', [Prefix]),
    append(CallLines, [IfLine|ThenLines], Lines0),
    append(Lines0, [ElseLine], Lines1),
    append(Lines1, ElseLines, Lines2),
    append(Lines2, [EndLine], Lines).
typr_mutual_tree_branch_body_lines_from(branch_steps_if(Steps, BranchConditionExpr, ThenBody, ElseBody), Prefix, Index, Lines) :-
    typr_mutual_tree_branch_steps_if_lines(Steps, Prefix, Index, BranchConditionExpr, ThenBody, ElseBody, Lines).
typr_mutual_tree_branch_body_lines_from(branch_if(BranchConditionExpr, ThenBody, ElseBody), Prefix, Index, Lines) :-
    format(string(IfLine), '~wif (~w) {', [Prefix, BranchConditionExpr]),
    atom_concat(Prefix, '    ', NestedPrefix),
    typr_mutual_tree_branch_body_lines_from(ThenBody, NestedPrefix, Index, ThenLines),
    typr_mutual_tree_branch_body_lines_from(ElseBody, NestedPrefix, Index, ElseLines),
    append([IfLine], ThenLines, Lines0),
    format(string(ElseLine), '~w} else {', [Prefix]),
    format(string(EndLine), '~w}', [Prefix]),
    append(Lines0, [ElseLine], Lines1),
    append(Lines1, ElseLines, Lines2),
    append(Lines2, [EndLine], Lines).
typr_mutual_tree_branch_body_lines_from(branch_guard(GuardExpr, Body), Prefix, Index, Lines) :-
    format(string(IfLine), '~wif (~w) {', [Prefix, GuardExpr]),
    atom_concat(Prefix, '    ', NestedPrefix),
    typr_mutual_tree_branch_body_lines_from(Body, NestedPrefix, Index, BodyLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    format(string(FalseLine), '~w    result = FALSE;', [Prefix]),
    format(string(EndLine), '~w}', [Prefix]),
    append([IfLine], BodyLines, Lines0),
    append(Lines0, [ElseLine, FalseLine, EndLine], Lines).
typr_mutual_tree_branch_body_lines_from(branch_steps(Steps), Prefix, Index, Lines) :-
    typr_mutual_tree_branch_steps_lines(Steps, Prefix, Index, Lines).
typr_mutual_tree_branch_body_lines_from(branch_calls(Calls), Prefix, 1, Lines) :-
    typr_mutual_tree_dual_branch_call_lines(Calls, Prefix, Lines).

typr_mutual_tree_branch_body_lines_no_memo(Body, Prefix, Lines) :-
    typr_mutual_tree_branch_body_lines_no_memo_from(Body, Prefix, 1, Lines).

typr_mutual_tree_branch_body_lines_no_memo_from(branch_after_call(BranchCall, BranchConditionExpr, ThenBody, ElseBody), Prefix, Index, Lines) :-
    typr_mutual_tree_step_result_name(Index, ResultName),
    typr_mutual_tree_call_expr(BranchCall, CallExpr),
    format(string(CallLine), '~w~w = ~w;', [Prefix, ResultName, CallExpr]),
    format(string(IfLine), '~wif (~w) {', [Prefix, ResultName]),
    atom_concat(Prefix, '    ', BranchPrefix),
    format(string(BranchIfLine), '~wif (~w) {', [BranchPrefix, BranchConditionExpr]),
    atom_concat(BranchPrefix, '    ', NestedPrefix),
    NextIndex is Index + 1,
    typr_mutual_tree_branch_body_lines_no_memo_from(ThenBody, NestedPrefix, NextIndex, ThenLines),
    format(string(BranchElseLine), '~w} else {', [BranchPrefix]),
    typr_mutual_tree_branch_body_lines_no_memo_from(ElseBody, NestedPrefix, NextIndex, ElseLines),
    format(string(BranchEndLine), '~w}', [BranchPrefix]),
    format(string(PrefixElseLine), '~w} else {', [Prefix]),
    format(string(FalseLine), '~w    result = FALSE;', [Prefix]),
    format(string(EndLine), '~w}', [Prefix]),
    append([CallLine, IfLine, BranchIfLine], ThenLines, Lines0),
    append(Lines0, [BranchElseLine], Lines1),
    append(Lines1, ElseLines, Lines2),
    append(Lines2, [BranchEndLine, PrefixElseLine, FalseLine, EndLine], Lines).
typr_mutual_tree_branch_body_lines_no_memo_from(branch_after_calls(Calls, BranchConditionExpr, ThenBody, ElseBody), Prefix, _Index, Lines) :-
    typr_mutual_tree_dual_branch_call_prefix_lines_no_memo(Calls, Prefix, CallLines),
    format(string(IfLine), '~wif (~w) {', [Prefix, BranchConditionExpr]),
    atom_concat(Prefix, '    ', NestedPrefix),
    typr_mutual_tree_branch_body_lines_no_memo_from(ThenBody, NestedPrefix, 3, ThenLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    typr_mutual_tree_branch_body_lines_no_memo_from(ElseBody, NestedPrefix, 3, ElseLines),
    format(string(EndLine), '~w}', [Prefix]),
    append(CallLines, [IfLine|ThenLines], Lines0),
    append(Lines0, [ElseLine], Lines1),
    append(Lines1, ElseLines, Lines2),
    append(Lines2, [EndLine], Lines).
typr_mutual_tree_branch_body_lines_no_memo_from(branch_steps_if(Steps, BranchConditionExpr, ThenBody, ElseBody), Prefix, Index, Lines) :-
    typr_mutual_tree_branch_steps_if_lines_no_memo(Steps, Prefix, Index, BranchConditionExpr, ThenBody, ElseBody, Lines).
typr_mutual_tree_branch_body_lines_no_memo_from(branch_if(BranchConditionExpr, ThenBody, ElseBody), Prefix, Index, Lines) :-
    format(string(IfLine), '~wif (~w) {', [Prefix, BranchConditionExpr]),
    atom_concat(Prefix, '    ', NestedPrefix),
    typr_mutual_tree_branch_body_lines_no_memo_from(ThenBody, NestedPrefix, Index, ThenLines),
    typr_mutual_tree_branch_body_lines_no_memo_from(ElseBody, NestedPrefix, Index, ElseLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    format(string(EndLine), '~w}', [Prefix]),
    append([IfLine], ThenLines, Lines0),
    append(Lines0, [ElseLine], Lines1),
    append(Lines1, ElseLines, Lines2),
    append(Lines2, [EndLine], Lines).
typr_mutual_tree_branch_body_lines_no_memo_from(branch_guard(GuardExpr, Body), Prefix, Index, Lines) :-
    format(string(IfLine), '~wif (~w) {', [Prefix, GuardExpr]),
    atom_concat(Prefix, '    ', NestedPrefix),
    typr_mutual_tree_branch_body_lines_no_memo_from(Body, NestedPrefix, Index, BodyLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    format(string(FalseLine), '~w    FALSE', [Prefix]),
    format(string(EndLine), '~w}', [Prefix]),
    append([IfLine], BodyLines, Lines0),
    append(Lines0, [ElseLine, FalseLine, EndLine], Lines).
typr_mutual_tree_branch_body_lines_no_memo_from(branch_steps(Steps), Prefix, Index, Lines) :-
    typr_mutual_tree_branch_steps_lines_no_memo(Steps, Prefix, Index, Lines).
typr_mutual_tree_branch_body_lines_no_memo_from(branch_calls(Calls), Prefix, 1, Lines) :-
    typr_mutual_tree_dual_branch_call_lines_no_memo(Calls, Prefix, Lines).

typr_mutual_tree_branch_steps_lines([], Prefix, _Index, [ResultLine]) :-
    format(string(ResultLine), '~wresult = left_result && right_result;', [Prefix]).
typr_mutual_tree_branch_steps_lines([step_guard(GuardExpr)|Rest], Prefix, Index, Lines) :-
    format(string(IfLine), '~wif (~w) {', [Prefix, GuardExpr]),
    atom_concat(Prefix, '    ', NestedPrefix),
    typr_mutual_tree_branch_steps_lines(Rest, NestedPrefix, Index, StepLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    format(string(FalseLine), '~w    result = FALSE;', [Prefix]),
    format(string(EndLine), '~w}', [Prefix]),
    append([IfLine], StepLines, Lines0),
    append(Lines0, [ElseLine, FalseLine, EndLine], Lines).
typr_mutual_tree_branch_steps_lines([step_call(BranchCall)|Rest], Prefix, Index, Lines) :-
    typr_mutual_tree_step_result_name(Index, ResultName),
    typr_mutual_tree_call_expr(BranchCall, CallExpr),
    format(string(CallLine), '~w~w = ~w;', [Prefix, ResultName, CallExpr]),
    format(string(IfLine), '~wif (~w) {', [Prefix, ResultName]),
    atom_concat(Prefix, '    ', NestedPrefix),
    NextIndex is Index + 1,
    typr_mutual_tree_branch_steps_lines(Rest, NestedPrefix, NextIndex, StepLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    format(string(FalseLine), '~w    result = FALSE;', [Prefix]),
    format(string(EndLine), '~w}', [Prefix]),
    append([CallLine, IfLine], StepLines, Lines0),
    append(Lines0, [ElseLine, FalseLine, EndLine], Lines).

typr_mutual_tree_branch_steps_if_lines([], Prefix, Index, BranchConditionExpr, ThenBody, ElseBody, Lines) :-
    format(string(IfLine), '~wif (~w) {', [Prefix, BranchConditionExpr]),
    atom_concat(Prefix, '    ', NestedPrefix),
    typr_mutual_tree_branch_body_lines_from(ThenBody, NestedPrefix, Index, ThenLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    typr_mutual_tree_branch_body_lines_from(ElseBody, NestedPrefix, Index, ElseLines),
    format(string(EndLine), '~w}', [Prefix]),
    append([IfLine], ThenLines, Lines0),
    append(Lines0, [ElseLine], Lines1),
    append(Lines1, ElseLines, Lines2),
    append(Lines2, [EndLine], Lines).
typr_mutual_tree_branch_steps_if_lines([step_guard(GuardExpr)|Rest], Prefix, Index, BranchConditionExpr, ThenBody, ElseBody, Lines) :-
    format(string(IfLine), '~wif (~w) {', [Prefix, GuardExpr]),
    atom_concat(Prefix, '    ', NestedPrefix),
    typr_mutual_tree_branch_steps_if_lines(Rest, NestedPrefix, Index, BranchConditionExpr, ThenBody, ElseBody, StepLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    format(string(FalseLine), '~w    result = FALSE;', [Prefix]),
    format(string(EndLine), '~w}', [Prefix]),
    append([IfLine], StepLines, Lines0),
    append(Lines0, [ElseLine, FalseLine, EndLine], Lines).
typr_mutual_tree_branch_steps_if_lines([step_call(BranchCall)|Rest], Prefix, Index, BranchConditionExpr, ThenBody, ElseBody, Lines) :-
    typr_mutual_tree_step_result_name(Index, ResultName),
    typr_mutual_tree_call_expr(BranchCall, CallExpr),
    format(string(CallLine), '~w~w = ~w;', [Prefix, ResultName, CallExpr]),
    format(string(IfLine), '~wif (~w) {', [Prefix, ResultName]),
    atom_concat(Prefix, '    ', NestedPrefix),
    NextIndex is Index + 1,
    typr_mutual_tree_branch_steps_if_lines(Rest, NestedPrefix, NextIndex, BranchConditionExpr, ThenBody, ElseBody, StepLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    format(string(FalseLine), '~w    result = FALSE;', [Prefix]),
    format(string(EndLine), '~w}', [Prefix]),
    append([CallLine, IfLine], StepLines, Lines0),
    append(Lines0, [ElseLine, FalseLine, EndLine], Lines).

typr_mutual_tree_branch_steps_lines_no_memo([], Prefix, _Index, [ResultLine]) :-
    format(string(ResultLine), '~wresult', [Prefix]).
typr_mutual_tree_branch_steps_lines_no_memo([step_guard(GuardExpr)|Rest], Prefix, Index, Lines) :-
    format(string(IfLine), '~wif (~w) {', [Prefix, GuardExpr]),
    atom_concat(Prefix, '    ', NestedPrefix),
    typr_mutual_tree_branch_steps_lines_no_memo(Rest, NestedPrefix, Index, StepLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    format(string(FalseLine), '~w    result = FALSE;', [Prefix]),
    format(string(EndLine), '~w}', [Prefix]),
    append([IfLine], StepLines, Lines0),
    append(Lines0, [ElseLine, FalseLine, EndLine], Lines).
typr_mutual_tree_branch_steps_lines_no_memo([step_call(BranchCall)|Rest], Prefix, Index, Lines) :-
    typr_mutual_tree_step_result_name(Index, ResultName),
    typr_mutual_tree_call_expr(BranchCall, CallExpr),
    format(string(CallLine), '~w~w = ~w;', [Prefix, ResultName, CallExpr]),
    format(string(IfLine), '~wif (~w) {', [Prefix, ResultName]),
    atom_concat(Prefix, '    ', NestedPrefix),
    NextIndex is Index + 1,
    typr_mutual_tree_branch_steps_lines_no_memo(Rest, NestedPrefix, NextIndex, StepLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    format(string(FalseLine), '~w    result = FALSE;', [Prefix]),
    format(string(EndLine), '~w}', [Prefix]),
    append([CallLine, IfLine], StepLines, Lines0),
    append(Lines0, [ElseLine, FalseLine, EndLine], Lines).

typr_mutual_tree_branch_steps_if_lines_no_memo([], Prefix, Index, BranchConditionExpr, ThenBody, ElseBody, Lines) :-
    format(string(IfLine), '~wif (~w) {', [Prefix, BranchConditionExpr]),
    atom_concat(Prefix, '    ', NestedPrefix),
    typr_mutual_tree_branch_body_lines_no_memo_from(ThenBody, NestedPrefix, Index, ThenLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    typr_mutual_tree_branch_body_lines_no_memo_from(ElseBody, NestedPrefix, Index, ElseLines),
    format(string(EndLine), '~w}', [Prefix]),
    append([IfLine], ThenLines, Lines0),
    append(Lines0, [ElseLine], Lines1),
    append(Lines1, ElseLines, Lines2),
    append(Lines2, [EndLine], Lines).
typr_mutual_tree_branch_steps_if_lines_no_memo([step_guard(GuardExpr)|Rest], Prefix, Index, BranchConditionExpr, ThenBody, ElseBody, Lines) :-
    format(string(IfLine), '~wif (~w) {', [Prefix, GuardExpr]),
    atom_concat(Prefix, '    ', NestedPrefix),
    typr_mutual_tree_branch_steps_if_lines_no_memo(Rest, NestedPrefix, Index, BranchConditionExpr, ThenBody, ElseBody, StepLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    format(string(FalseLine), '~w    result = FALSE;', [Prefix]),
    format(string(EndLine), '~w}', [Prefix]),
    append([IfLine], StepLines, Lines0),
    append(Lines0, [ElseLine, FalseLine, EndLine], Lines).
typr_mutual_tree_branch_steps_if_lines_no_memo([step_call(BranchCall)|Rest], Prefix, Index, BranchConditionExpr, ThenBody, ElseBody, Lines) :-
    typr_mutual_tree_step_result_name(Index, ResultName),
    typr_mutual_tree_call_expr(BranchCall, CallExpr),
    format(string(CallLine), '~w~w = ~w;', [Prefix, ResultName, CallExpr]),
    format(string(IfLine), '~wif (~w) {', [Prefix, ResultName]),
    atom_concat(Prefix, '    ', NestedPrefix),
    NextIndex is Index + 1,
    typr_mutual_tree_branch_steps_if_lines_no_memo(Rest, NestedPrefix, NextIndex, BranchConditionExpr, ThenBody, ElseBody, StepLines),
    format(string(ElseLine), '~w} else {', [Prefix]),
    format(string(FalseLine), '~w    result = FALSE;', [Prefix]),
    format(string(EndLine), '~w}', [Prefix]),
    append([CallLine, IfLine], StepLines, Lines0),
    append(Lines0, [ElseLine, FalseLine, EndLine], Lines).

typr_mutual_tree_step_result_name(1, left_result).
typr_mutual_tree_step_result_name(2, right_result).

typr_mutual_tree_call_expr(branch_call(HelperName, CallArgs), CallExpr) :-
    atomic_list_concat(CallArgs, ', ', CallArgsText),
    format(string(CallExpr), '~w(~w)', [HelperName, CallArgsText]).

typr_mutual_tree_dual_branch_call_lines(
    [FirstCall, SecondCall],
    Prefix,
    [FirstLine, SecondLine, ResultLine]
) :-
    typr_mutual_tree_call_expr(FirstCall, FirstCallExpr),
    typr_mutual_tree_call_expr(SecondCall, SecondCallExpr),
    format(string(FirstLine), '~wleft_result = ~w;', [Prefix, FirstCallExpr]),
    format(string(SecondLine), '~wright_result = ~w;', [Prefix, SecondCallExpr]),
    format(string(ResultLine), '~wresult = left_result && right_result;', [Prefix]).

typr_mutual_tree_dual_branch_call_prefix_lines(
    [FirstCall, SecondCall],
    Prefix,
    [FirstLine, SecondLine]
) :-
    typr_mutual_tree_call_expr(FirstCall, FirstCallExpr),
    typr_mutual_tree_call_expr(SecondCall, SecondCallExpr),
    format(string(FirstLine), '~wleft_result = ~w;', [Prefix, FirstCallExpr]),
    format(string(SecondLine), '~wright_result = ~w;', [Prefix, SecondCallExpr]).

typr_mutual_tree_dual_branch_call_lines_no_memo(
    [FirstCall, SecondCall],
    Prefix,
    [FirstLine, SecondLine, ResultLine]
) :-
    typr_mutual_tree_call_expr(FirstCall, FirstCallExpr),
    typr_mutual_tree_call_expr(SecondCall, SecondCallExpr),
    format(string(FirstLine), '~wleft_result = ~w;', [Prefix, FirstCallExpr]),
    format(string(SecondLine), '~wright_result = ~w;', [Prefix, SecondCallExpr]),
    format(string(ResultLine), '~wleft_result && right_result', [Prefix]).

typr_mutual_tree_dual_branch_call_prefix_lines_no_memo(
    [FirstCall, SecondCall],
    Prefix,
    [FirstLine, SecondLine]
) :-
    typr_mutual_tree_call_expr(FirstCall, FirstCallExpr),
    typr_mutual_tree_call_expr(SecondCall, SecondCallExpr),
    format(string(FirstLine), '~wleft_result = ~w;', [Prefix, FirstCallExpr]),
    format(string(SecondLine), '~wright_result = ~w;', [Prefix, SecondCallExpr]).

typr_mutual_false_lines(GroupName, [
    '        } else {',
    '            result = FALSE;',
    AssignLine,
    '            result',
    '        }'
]) :-
    format(string(AssignLine), '            assign(key, result, envir=~w_memo);', [GroupName]).

typr_mutual_stop_lines(PredStr, [
    '        } else {',
    StopLine,
    '        }'
]) :-
    format(string(StopLine), '            stop("No matching mutual recursive clause for ~w")', [PredStr]).

typr_mutual_stop_lines_no_memo(PredStr, [
    '        } else {',
    StopLine,
    '        }'
]) :-
    format(string(StopLine), '            stop("No matching mutual recursive clause for ~w")', [PredStr]).

compile_typr_tail_recursive_accumulator(_Module:Pred/Arity, TypedMode, Clauses, Code) :-
    Arity =:= 3,
    is_tail_recursive_accumulator(Pred/Arity, _),
    single_tail_recursive_clause_pair(Pred, Arity, Clauses, BaseClause, RecClause),
    tail_recursive_loop_spec(Pred, BaseClause, RecClause, LoopSpec),
    build_typed_arg_list(Pred/Arity, none, Arity, TypedMode, TypedArgList),
    generic_typr_return_type(Pred/Arity, Clauses, ReturnType),
    build_typr_tail_recursive_body(RecClause, LoopSpec, Body),
    atom_string(Pred, PredStr),
    indent_text(Body, "\t", IndentedBody),
    format(string(Code),
'# Generated by UnifyWeaver TypR Target
# Predicate: ~w/~w

let ~w <- fn(~w): ~w {
~w
};
', [PredStr, Arity, PredStr, TypedArgList, ReturnType, IndentedBody]).

single_tail_recursive_clause_pair(Pred, Arity, Clauses, BaseClause, RecClause) :-
    partition(clause_calls_predicate(Pred, Arity), Clauses, RecClauses, BaseClauses),
    BaseClauses = [BaseClause],
    RecClauses = [RecClause].

single_linear_recursive_clause_pair(Pred, Arity, Clauses, BaseClause, RecClause) :-
    partition(clause_calls_predicate(Pred, Arity), Clauses, RecClauses, BaseClauses),
    BaseClauses = [BaseClause],
    RecClauses = [RecClause].

single_tree_recursive_clause_set(Pred, Arity, Clauses, BaseClauses, RecClause) :-
    partition(clause_calls_predicate(Pred, Arity), Clauses, RecClauses, BaseClauses),
    BaseClauses \= [],
    RecClauses = [RecClause],
    RecClause = (_Head-RecBody),
    recursive_goal_count(Pred, RecBody, Count),
    Count >= 2.

compile_typr_linear_recursive_numeric(_Module:Pred/Arity, TypedMode, Clauses, Code) :-
    single_linear_recursive_clause_pair(Pred, Arity, Clauses, BaseClause, RecClause),
    linear_recursive_numeric_spec(Pred, Arity, BaseClause, RecClause, LoopSpec),
    build_typed_arg_list(Pred/Arity, none, Arity, TypedMode, TypedArgList),
    generic_typr_return_type(Pred/Arity, Clauses, ReturnType),
    build_typr_linear_recursive_body(Pred, LoopSpec, Body),
    atom_string(Pred, PredStr),
    indent_text(Body, "\t", IndentedBody),
    format(string(Code),
'# Generated by UnifyWeaver TypR Target
# Predicate: ~w/~w

let ~w <- fn(~w): ~w {
~w
};
', [PredStr, Arity, PredStr, TypedArgList, ReturnType, IndentedBody]).

compile_typr_linear_recursive_list(_Module:Pred/Arity, TypedMode, Clauses, Code) :-
    single_linear_recursive_clause_pair(Pred, Arity, Clauses, BaseClause, RecClause),
    linear_recursive_list_spec(Pred, Arity, BaseClause, RecClause, LoopSpec),
    build_typed_arg_list(Pred/Arity, none, Arity, TypedMode, TypedArgList),
    generic_typr_return_type(Pred/Arity, Clauses, ReturnType),
    build_typr_linear_recursive_list_body(LoopSpec, Body),
    atom_string(Pred, PredStr),
    indent_text(Body, "\t", IndentedBody),
    format(string(Code),
'# Generated by UnifyWeaver TypR Target
# Predicate: ~w/~w

let ~w <- fn(~w): ~w {
~w
};
', [PredStr, Arity, PredStr, TypedArgList, ReturnType, IndentedBody]).

compile_typr_tree_recursive_structural(_Module:Pred/Arity, TypedMode, Clauses, Code) :-
    Arity >= 2,
    single_tree_recursive_clause_set(Pred, Arity, Clauses, BaseClauses, RecClause),
    structural_tree_recursive_spec(Pred, Arity, BaseClauses, RecClause, TreeSpec),
    build_typed_arg_list(Pred/Arity, none, Arity, TypedMode, TypedArgList),
    generic_typr_return_type(Pred/Arity, Clauses, ReturnType),
    build_typr_structural_tree_recursive_body(Pred, TreeSpec, Body),
    atom_string(Pred, PredStr),
    indent_text(Body, "\t", IndentedBody),
    format(string(Code),
'# Generated by UnifyWeaver TypR Target
# Predicate: ~w/~w

let ~w <- fn(~w): ~w {
~w
};
', [PredStr, Arity, PredStr, TypedArgList, ReturnType, IndentedBody]).

compile_typr_tree_recursive_numeric(_Module:Pred/Arity, TypedMode, Clauses, Code) :-
    Arity =:= 2,
    single_tree_recursive_clause_set(Pred, Arity, Clauses, BaseClauses, RecClause),
    tree_recursive_numeric_spec(Pred, BaseClauses, RecClause, TreeSpec),
    build_typed_arg_list(Pred/Arity, none, Arity, TypedMode, TypedArgList),
    generic_typr_return_type(Pred/Arity, Clauses, ReturnType),
    build_typr_tree_recursive_body(Pred, TreeSpec, Body),
    atom_string(Pred, PredStr),
    indent_text(Body, "\t", IndentedBody),
    format(string(Code),
'# Generated by UnifyWeaver TypR Target
# Predicate: ~w/~w

let ~w <- fn(~w): ~w {
~w
};
', [PredStr, Arity, PredStr, TypedArgList, ReturnType, IndentedBody]).

linear_recursive_numeric_spec(
    Pred,
    Arity,
    BaseHead-BaseBody,
    RecHead-RecBody,
    linear_loop_spec{
        base_input_literal: BaseInputLiteral,
        base_output_expr: BaseOutputExpr,
        recursive_guard_expr: RecursiveGuardExpr,
        seq_expr: SeqExpr,
        fold_expr: FoldExpr,
        input_arg_name: InputArgName,
        output_arg_name: OutputArgName,
        result_name: ResultName
    }
) :-
    BaseBody == true,
    BaseHead =.. [_BasePredName|BaseArgs],
    RecHead =.. [_RecPredName|RecHeadArgs],
    split_body_at_recursive_call(RecBody, Pred, PreGoals, RecCall, PostGoals),
    RecCall =.. [_PredName|RecCallArgs],
    linear_recursive_arg_spec(
        Arity,
        BaseArgs,
        RecHeadArgs,
        RecCallArgs,
        BaseOutput,
        OutputVar,
        RecResultVar,
        DriverPos,
        BaseInput,
        InputVar,
        _RecInputArg,
        BaseInvariantVarMap,
        RecInvariantVarMap,
        InputArgName,
        OutputArgName,
        ResultName
    ),
    number(BaseInput),
    var(InputVar),
    extract_typr_linear_step_info(RecBody, DriverPos, RecCallArgs, InputVar, Step, Direction),
    append([InputVar-"current", RecResultVar-"acc"], RecInvariantVarMap, FoldVarMap),
    append([InputVar-"current_input"], RecInvariantVarMap, GuardVarMap),
    linear_recursive_guard_expr(PreGoals, GuardVarMap, RecursiveGuardExpr),
    linear_recursive_output_expr(PostGoals, OutputVar, FoldVarMap, FoldExpr),
    typr_translate_r_expr(BaseOutput, BaseInvariantVarMap, BaseOutputExpr),
    r_literal(BaseInput, BaseInputLiteral),
    typr_linear_seq_expr(BaseInput, Step, Direction, SeqExpr).

linear_recursive_list_spec(
    Pred,
    Arity,
    BaseHead-BaseBody,
    RecHead-RecBody,
    list_loop_spec{
        base_output_expr: BaseOutputExpr,
        fold_expr: FoldExpr,
        input_arg_name: InputArgName,
        output_arg_name: OutputArgName,
        result_name: ResultName
    }
) :-
    BaseBody == true,
    BaseHead =.. [_BasePredName|BaseArgs],
    RecHead =.. [_RecPredName|RecHeadArgs],
    split_body_at_recursive_call(RecBody, Pred, PreGoals, RecCall, PostGoals),
    RecCall =.. [_PredName|RecCallArgs],
    linear_recursive_arg_spec(
        Arity,
        BaseArgs,
        RecHeadArgs,
        RecCallArgs,
        BaseOutput,
        OutputVar,
        RecResultVar,
        _DriverPos,
        BaseInput,
        InputPattern,
        RecInputArg,
        BaseInvariantVarMap,
        RecInvariantVarMap,
        InputArgName,
        OutputArgName,
        ResultName
    ),
    BaseInput == [],
    InputPattern = [HeadVar|TailVar],
    var(TailVar),
    PreGoals == true,
    RecInputArg == TailVar,
    append([HeadVar-"current", RecResultVar-"acc"], RecInvariantVarMap, FoldVarMap),
    linear_recursive_output_expr(PostGoals, OutputVar, FoldVarMap, FoldExpr),
    typr_translate_r_expr(BaseOutput, BaseInvariantVarMap, BaseOutputExpr).

linear_recursive_arg_spec(
    Arity,
    BaseArgs,
    RecHeadArgs,
    RecCallArgs,
    BaseOutput,
    OutputVar,
    RecResultVar,
    DriverPos,
    BaseInput,
    DriverHeadArg,
    DriverRecArg,
    BaseInvariantVarMap,
    RecInvariantVarMap,
    InputArgName,
    OutputArgName,
    ResultName
) :-
    Arity >= 2,
    OutputPos is Arity,
    nth1(OutputPos, BaseArgs, BaseOutput),
    nth1(OutputPos, RecHeadArgs, OutputVar),
    var(OutputVar),
    nth1(OutputPos, RecCallArgs, RecResultVar),
    var(RecResultVar),
    single_linear_recursive_driver_pos(Arity, RecHeadArgs, RecCallArgs, DriverPos),
    nth1(DriverPos, BaseArgs, BaseInput),
    nth1(DriverPos, RecHeadArgs, DriverHeadArg),
    nth1(DriverPos, RecCallArgs, DriverRecArg),
    linear_recursive_invariant_positions(Arity, DriverPos, InvariantPositions),
    base_linear_invariants_ok(BaseArgs, RecHeadArgs, InvariantPositions),
    build_linear_recursive_varmap(BaseArgs, InvariantPositions, BaseInvariantVarMap),
    build_linear_recursive_varmap(RecHeadArgs, InvariantPositions, RecInvariantVarMap),
    format(string(InputArgName), 'arg~w', [DriverPos]),
    format(string(OutputArgName), 'arg~w', [OutputPos]),
    linear_recursive_result_name(Arity, ResultName).

tree_recursive_numeric_spec(
    Pred,
    BaseClauses,
    RecHead-RecBody,
    tree_recursive_spec{
        base_cases: BaseCases,
        recursive_guard_expr: GuardExpr,
        step_lines: StepLines,
        call_lines: CallLines,
        result_expr: ResultExpr,
        input_arg_name: InputArgName,
        output_arg_name: OutputArgName,
        result_name: ResultName,
        helper_name: HelperName,
        memo_name: MemoName
    }
) :-
    maplist(tree_recursive_base_case_spec, BaseClauses, BaseCases),
    RecHead =.. [_PredName, InputVar, OutputVar],
    var(InputVar),
    var(OutputVar),
    normalize_typr_goals(RecBody, Goals),
    split_typr_multicall_goals(Pred, Goals, PreGoals, RecCalls, PostGoals),
    typr_goals_to_body(PreGoals, PreBody),
    compile_tail_recursive_pre_goals(PreBody, [InputVar-"current_input"], VarMap0, GuardConditions, StepLines),
    raw_guard_expr(GuardConditions, GuardExpr),
    atom_string(Pred, PredStr),
    format(string(HelperName), '~w_impl', [PredStr]),
    format(string(MemoName), '~w_memo', [PredStr]),
    build_typr_tree_recursive_call_lines(RecCalls, HelperName, VarMap0, VarMap1, CallLines),
    typr_goals_to_body(PostGoals, PostBody),
    normalize_typr_goals(PostBody, [OutputVar is AggExpr]),
    typr_translate_r_expr(AggExpr, VarMap1, ResultExpr),
    InputArgName = "arg1",
    OutputArgName = "arg2",
    linear_recursive_result_name(2, ResultName).

tree_recursive_base_case_spec(Head-Body, base_case(InputLiteral, OutputExpr)) :-
    Body == true,
    Head =.. [_PredName, BaseInput, BaseOutput],
    nonvar(BaseInput),
    nonvar(BaseOutput),
    r_literal(BaseInput, InputLiteral),
    r_literal(BaseOutput, OutputExpr).

structural_tree_recursive_spec(
    Pred,
    Arity,
    BaseClauses,
    RecHead-RecBody,
    structural_tree_spec{
        base_output_expr: BaseOutputExpr,
        guard_expr: GuardExpr,
        step_lines: StepLines,
        call_lines: CallLines,
        result_expr: ResultExpr,
        input_arg_name: _InputArgName,
        output_arg_name: OutputArgName,
        result_name: ResultName,
        helper_name: HelperName,
        helper_param_list: HelperParamList,
        helper_call_arg_list: HelperCallArgList
    }
) :-
    RecHead =.. [_PredName|RecHeadArgs],
    normalize_typr_goals(RecBody, Goals),
    split_typr_multicall_goals(Pred, Goals, PreGoals, RecCalls, PostGoals),
    structural_tree_arg_spec(
        Arity,
        BaseClauses,
        RecHeadArgs,
        RecCalls,
        BaseOutputExpr,
        OutputVar,
        DriverPos,
        ValueVar,
        LeftVar,
        RightVar,
        RecInvariantVarMap,
        InputArgName,
        OutputArgName,
        ResultName
    ),
    typr_goals_to_body(PreGoals, PreBody),
    PreVarMap0 = [ValueVar-"value", LeftVar-"left", RightVar-"right"|RecInvariantVarMap],
    compile_tail_recursive_pre_goals(PreBody, PreVarMap0, PreVarMap, GuardConditions, StepLines),
    raw_guard_expr(GuardConditions, GuardExpr),
    atom_string(Pred, PredStr),
    format(string(HelperName), '~w_impl', [PredStr]),
    structural_tree_helper_call_plan(
        HelperName,
        Arity,
        DriverPos,
        RecHeadArgs,
        RecCalls,
        LeftVar,
        RightVar,
        InputArgName,
        PreVarMap,
        HelperParamList,
        HelperCallArgList,
        CallLines,
        VarMap
    ),
    typr_goals_to_body(PostGoals, PostBody),
    linear_recursive_output_expr(PostBody, OutputVar, VarMap, ResultExpr).
structural_tree_recursive_spec(
    Pred,
    Arity,
    BaseClauses,
    RecHead-RecBody,
    structural_tree_spec{
        base_output_expr: BaseOutputExpr,
        guard_expr: 'TRUE',
        step_lines: StepLines,
        call_lines: [],
        result_expr: "branch_result",
        input_arg_name: _InputArgName,
        output_arg_name: OutputArgName,
        result_name: ResultName,
        helper_name: HelperName,
        helper_param_list: HelperParamList,
        helper_call_arg_list: HelperCallArgList
    }
) :-
    RecHead =.. [_PredName|RecHeadArgs],
    typr_if_then_else_goal(RecBody, IfGoal, ThenGoal, ElseGoal),
    structural_tree_goal_rec_calls(Pred, ThenGoal, ThenRecCalls),
    structural_tree_arg_spec(
        Arity,
        BaseClauses,
        RecHeadArgs,
        ThenRecCalls,
        BaseOutputExpr,
        OutputVar,
        DriverPos,
        ValueVar,
        LeftVar,
        RightVar,
        RecInvariantVarMap,
        InputArgName,
        OutputArgName,
        ResultName
    ),
    PreVarMap0 = [ValueVar-"value", LeftVar-"left", RightVar-"right"|RecInvariantVarMap],
    atom_string(Pred, PredStr),
    format(string(HelperName), '~w_impl', [PredStr]),
    structural_tree_helper_param_lists(
        Arity,
        DriverPos,
        RecHeadArgs,
        InputArgName,
        PreVarMap0,
        HelperParamList,
        HelperCallArgList
    ),
    structural_tree_if_branch_step_lines(
        Pred,
        IfGoal,
        ThenGoal,
        ElseGoal,
        PreVarMap0,
        HelperName,
        DriverPos,
        Arity,
        LeftVar,
        RightVar,
        OutputVar,
        StepLines
    ).

structural_tree_arg_spec(
    Arity,
    BaseClauses,
    RecHeadArgs,
    RecCalls,
    BaseOutputExpr,
    OutputVar,
    DriverPos,
    ValueVar,
    LeftVar,
    RightVar,
    RecInvariantVarMap,
    InputArgName,
    OutputArgName,
    ResultName
) :-
    OutputPos is Arity,
    nth1(OutputPos, RecHeadArgs, OutputVar),
    var(OutputVar),
    RecCalls = [_, _],
    structural_tree_driver_pos(Arity, RecHeadArgs, RecCalls, DriverPos, ValueVar, LeftVar, RightVar),
    linear_recursive_invariant_positions(Arity, DriverPos, InvariantPositions),
    structural_tree_base_case_spec(
        BaseClauses,
        DriverPos,
        OutputPos,
        InvariantPositions,
        RecHeadArgs,
        BaseOutputExpr
    ),
    build_linear_recursive_varmap(RecHeadArgs, InvariantPositions, RecInvariantVarMap),
    format(string(InputArgName), 'arg~w', [DriverPos]),
    format(string(OutputArgName), 'arg~w', [OutputPos]),
    linear_recursive_result_name(Arity, ResultName).

structural_tree_driver_pos(Arity, RecHeadArgs, RecCalls, DriverPos, ValueVar, LeftVar, RightVar) :-
    InputLast is Arity - 1,
    structural_tree_driver_pos(1, InputLast, RecHeadArgs, RecCalls, DriverPos, ValueVar, LeftVar, RightVar).

structural_tree_driver_pos(Pos, InputLast, RecHeadArgs, _RecCalls, DriverPos, ValueVar, LeftVar, RightVar) :-
    Pos =< InputLast,
    nth1(Pos, RecHeadArgs, TreeArg),
    nonvar(TreeArg),
    TreeArg = [ValueVar, LeftVar, RightVar],
    var(LeftVar),
    var(RightVar),
    !,
    DriverPos = Pos.
structural_tree_driver_pos(Pos, InputLast, RecHeadArgs, RecCalls, DriverPos, ValueVar, LeftVar, RightVar) :-
    Pos < InputLast,
    NextPos is Pos + 1,
    structural_tree_driver_pos(NextPos, InputLast, RecHeadArgs, RecCalls, DriverPos, ValueVar, LeftVar, RightVar).

structural_tree_base_case_spec(BaseClauses, DriverPos, OutputPos, InvariantPositions, RecHeadArgs, BaseOutputExpr) :-
    BaseClauses = [Head-Body],
    Body == true,
    Head =.. [_PredName|BaseArgs],
    nth1(DriverPos, BaseArgs, BaseTree),
    BaseTree == [],
    base_linear_invariants_ok(BaseArgs, RecHeadArgs, InvariantPositions),
    build_linear_recursive_varmap(BaseArgs, InvariantPositions, BaseInvariantVarMap),
    nth1(OutputPos, BaseArgs, BaseOutput),
    typr_translate_r_expr(BaseOutput, BaseInvariantVarMap, BaseOutputExpr).

structural_tree_helper_call_plan(
    HelperName,
    Arity,
    DriverPos,
    RecHeadArgs,
    RecCalls,
    LeftVar,
    RightVar,
    InputArgName,
    VarMap0,
    HelperParamList,
    HelperCallArgList,
    CallLines,
    VarMap
) :-
    OutputPos is Arity,
    linear_recursive_invariant_positions(Arity, DriverPos, InvariantPositions),
    structural_tree_helper_invariant_names(RecHeadArgs, InvariantPositions, VarMap0, InvariantNames),
    append(["current_tree"], InvariantNames, HelperParams),
    atomic_list_concat(HelperParams, ', ', HelperParamList),
    append([InputArgName], InvariantNames, HelperCallArgs),
    atomic_list_concat(HelperCallArgs, ', ', HelperCallArgList),
    build_structural_tree_call_lines(
        RecCalls,
        HelperName,
        DriverPos,
        OutputPos,
        LeftVar,
        RightVar,
        InvariantPositions,
        VarMap0,
        VarMap,
        CallLines
    ).

structural_tree_helper_param_lists(
    Arity,
    DriverPos,
    RecHeadArgs,
    InputArgName,
    VarMap0,
    HelperParamList,
    HelperCallArgList
) :-
    linear_recursive_invariant_positions(Arity, DriverPos, InvariantPositions),
    structural_tree_helper_invariant_names(RecHeadArgs, InvariantPositions, VarMap0, InvariantNames),
    append(["current_tree"], InvariantNames, HelperParams),
    atomic_list_concat(HelperParams, ', ', HelperParamList),
    append([InputArgName], InvariantNames, HelperCallArgs),
    atomic_list_concat(HelperCallArgs, ', ', HelperCallArgList).

structural_tree_helper_invariant_names(_RecHeadArgs, [], _VarMap, []).
structural_tree_helper_invariant_names(RecHeadArgs, [Pos|Rest], VarMap, [Name|RestNames]) :-
    nth1(Pos, RecHeadArgs, HeadArg),
    lookup_typr_var(HeadArg, VarMap, Name),
    structural_tree_helper_invariant_names(RecHeadArgs, Rest, VarMap, RestNames).

build_structural_tree_call_lines(RecCalls, HelperName, DriverPos, OutputPos, LeftVar, RightVar, InvariantPositions, VarMap0, VarMap, Lines) :-
    build_structural_tree_call_lines(RecCalls, HelperName, DriverPos, OutputPos, LeftVar, RightVar, InvariantPositions, 1, VarMap0, VarMap, Lines).

build_structural_tree_call_lines([], _HelperName, _DriverPos, _OutputPos, _LeftVar, _RightVar, _InvariantPositions, _Index, VarMap, VarMap, []).
build_structural_tree_call_lines([RecCall|Rest], HelperName, DriverPos, OutputPos, LeftVar, RightVar, InvariantPositions, Index, VarMap0, VarMap, [Line|RestLines]) :-
    RecCall =.. [_PredName|CallArgs],
    nth1(OutputPos, CallArgs, CallOutputVar),
    var(CallOutputVar),
    nth1(DriverPos, CallArgs, CallInput),
    typr_translate_r_expr(CallInput, VarMap0, CallInputExpr),
    structural_tree_call_invariant_exprs(CallArgs, InvariantPositions, VarMap0, InvariantExprs),
    append([CallInputExpr], InvariantExprs, HelperArgs),
    atomic_list_concat(HelperArgs, ', ', HelperArgList),
    structural_tree_call_result_name(Index, CallInput, LeftVar, RightVar, ResultName),
    format(string(Line), '            ~w = ~w(~w);', [ResultName, HelperName, HelperArgList]),
    update_typr_expr_varmap(VarMap0, CallOutputVar, ResultName, VarMap1),
    NextIndex is Index + 1,
    build_structural_tree_call_lines(Rest, HelperName, DriverPos, OutputPos, LeftVar, RightVar, InvariantPositions, NextIndex, VarMap1, VarMap, RestLines).

structural_tree_call_result_name(_Index, CallInput, LeftVar, _RightVar, "left_result") :-
    CallInput == LeftVar,
    !.
structural_tree_call_result_name(_Index, CallInput, _LeftVar, RightVar, "right_result") :-
    CallInput == RightVar,
    !.
structural_tree_call_result_name(1, _CallInput, _LeftVar, _RightVar, "left_result") :-
    !.
structural_tree_call_result_name(2, _CallInput, _LeftVar, _RightVar, "right_result") :-
    !.
structural_tree_call_result_name(Index, _CallInput, _LeftVar, _RightVar, ResultName) :-
    format(string(ResultName), 'call_~w', [Index]).

structural_tree_call_invariant_exprs(_CallArgs, [], _VarMap, []).
structural_tree_call_invariant_exprs(CallArgs, [Pos|Rest], VarMap, [Expr|RestExprs]) :-
    nth1(Pos, CallArgs, CallArg),
    typr_translate_r_expr(CallArg, VarMap, Expr),
    structural_tree_call_invariant_exprs(CallArgs, Rest, VarMap, RestExprs).

structural_tree_if_branch_step_lines(
    Pred,
    IfGoal,
    ThenGoal,
    ElseGoal,
    VarMap0,
    HelperName,
    DriverPos,
    Arity,
    LeftVar,
    RightVar,
    OutputVar,
    Lines
) :-
    native_typr_if_condition(IfGoal, VarMap0, IfCondition0),
    typr_condition_expr_text(IfCondition0, IfCondition),
    structural_tree_branch_body_lines(
        Pred,
        ThenGoal,
        VarMap0,
        HelperName,
        DriverPos,
        Arity,
        LeftVar,
        RightVar,
        OutputVar,
        ThenLines
    ),
    structural_tree_branch_body_lines(
        Pred,
        ElseGoal,
        VarMap0,
        HelperName,
        DriverPos,
        Arity,
        LeftVar,
        RightVar,
        OutputVar,
        ElseLines
    ),
    indent_lines(ThenLines, '    ', IndentedThenLines),
    indent_lines(ElseLines, '    ', IndentedElseLines),
    format(string(IfLine), '        if (~w) {', [IfCondition]),
    append([IfLine|IndentedThenLines], ['        } else {'|IndentedElseLines], Lines0),
    append(Lines0, ['        };'], Lines).

structural_tree_branch_body_lines(
    Pred,
    BranchGoal,
    VarMap0,
    HelperName,
    DriverPos,
    Arity,
    LeftVar,
    RightVar,
    OutputVar,
    Lines
) :-
    structural_tree_nested_if_goal_any(BranchGoal, PreGoals, NestedIfGoal, PostGoals),
    typr_goals_to_body(PreGoals, PreBody),
    compile_tail_recursive_pre_goals(PreBody, VarMap0, PreVarMap, GuardConditions, StepLines),
    tail_recursive_pre_branch_lines(GuardConditions, StepLines, BranchPreLines),
    typr_if_then_else_goal(NestedIfGoal, IfGoal, ThenGoal, ElseGoal),
    native_typr_if_condition(IfGoal, PreVarMap, IfCondition0),
    typr_condition_expr_text(IfCondition0, IfCondition),
    structural_tree_recursive_prefix_lines(
        Pred,
        ThenGoal,
        PreVarMap,
        HelperName,
        DriverPos,
        Arity,
        LeftVar,
        RightVar,
        ThenVarMap,
        ThenLines
    ),
    structural_tree_recursive_prefix_lines(
        Pred,
        ElseGoal,
        PreVarMap,
        HelperName,
        DriverPos,
        Arity,
        LeftVar,
        RightVar,
        ElseVarMap,
        ElseLines
    ),
    structural_tree_branch_result_expr(PostGoals, OutputVar, ThenVarMap, ThenBranchResultExpr),
    structural_tree_branch_result_expr(PostGoals, OutputVar, ElseVarMap, ElseBranchResultExpr),
    format(string(ThenResultLine), '        branch_result = ~w;', [ThenBranchResultExpr]),
    format(string(ElseResultLine), '        branch_result = ~w;', [ElseBranchResultExpr]),
    append(ThenLines, [ThenResultLine], ThenLinesWithResult),
    append(ElseLines, [ElseResultLine], ElseLinesWithResult),
    indent_lines(ThenLinesWithResult, '    ', IndentedThenLines),
    indent_lines(ElseLinesWithResult, '    ', IndentedElseLines),
    format(string(IfLine), '        if (~w) {', [IfCondition]),
    append(BranchPreLines, [IfLine|IndentedThenLines], Lines0),
    append(Lines0, ['        } else {'|IndentedElseLines], Lines1),
    append(Lines1, ['        };'], Lines).
structural_tree_branch_body_lines(
    Pred,
    BranchGoal,
    VarMap0,
    HelperName,
    DriverPos,
    Arity,
    LeftVar,
    RightVar,
    OutputVar,
    Lines
) :-
    OutputPos is Arity,
    linear_recursive_invariant_positions(Arity, DriverPos, InvariantPositions),
    normalize_typr_goals(BranchGoal, Goals),
    split_typr_multicall_goals(Pred, Goals, PreGoals, RecCalls, PostGoals),
    typr_goals_to_body(PreGoals, PreBody),
    compile_tail_recursive_pre_goals(PreBody, VarMap0, PreVarMap, GuardConditions, StepLines),
    tail_recursive_pre_branch_lines(GuardConditions, StepLines, BranchPreLines),
    build_structural_tree_call_lines(
        RecCalls,
        HelperName,
        DriverPos,
        OutputPos,
        LeftVar,
        RightVar,
        InvariantPositions,
        PreVarMap,
        VarMap,
        CallLines
    ),
    typr_goals_to_body(PostGoals, PostBody),
    linear_recursive_output_expr(PostBody, OutputVar, VarMap, BranchResultExpr),
    format(string(ResultLine), '        branch_result = ~w;', [BranchResultExpr]),
    append(BranchPreLines, CallLines, Lines0),
    append(Lines0, [ResultLine], Lines).

structural_tree_branch_result_expr([], OutputVar, VarMap, ResultExpr) :-
    !,
    lookup_typr_var(OutputVar, VarMap, ResultExpr).
structural_tree_branch_result_expr(PostGoals, OutputVar, VarMap, ResultExpr) :-
    typr_goals_to_body(PostGoals, PostBody),
    linear_recursive_output_expr(PostBody, OutputVar, VarMap, ResultExpr).

structural_tree_goal_rec_calls(Pred, Goal, RecCalls) :-
    normalize_typr_goals(Goal, Goals),
    (   split_typr_multicall_goals(Pred, Goals, _PreGoals, RecCalls, _PostGoals)
    ;   split_typr_multicall_goal_prefix(Pred, Goals, _PreGoals, RecCalls)
    ),
    !.
structural_tree_goal_rec_calls(Pred, Goal, RecCalls) :-
    typr_if_then_else_goal(Goal, _IfGoal, ThenGoal, _ElseGoal),
    !,
    structural_tree_goal_rec_calls(Pred, ThenGoal, RecCalls).
structural_tree_goal_rec_calls(Pred, Goal, RecCalls) :-
    structural_tree_nested_if_goal_any(Goal, _PreGoals, NestedIfGoal, _PostGoals),
    typr_if_then_else_goal(NestedIfGoal, _IfGoal, ThenGoal, _ElseGoal),
    structural_tree_goal_rec_calls(Pred, ThenGoal, RecCalls).

structural_tree_nested_if_goal(Goal, PreGoals, NestedIfGoal, PostGoals) :-
    normalize_typr_goals(Goal, Goals),
    append(PreGoals, [NestedIfGoal|PostGoals], Goals),
    PostGoals \= [],
    typr_if_then_else_goal(NestedIfGoal, _IfGoal, _ThenGoal, _ElseGoal),
    !.

structural_tree_nested_if_goal_any(Goal, PreGoals, NestedIfGoal, PostGoals) :-
    normalize_typr_goals(Goal, Goals),
    append(PreGoals, [NestedIfGoal|PostGoals], Goals),
    typr_if_then_else_goal(NestedIfGoal, _IfGoal, _ThenGoal, _ElseGoal),
    !.

structural_tree_recursive_prefix_lines(
    _Pred,
    true,
    VarMap,
    _HelperName,
    _DriverPos,
    _Arity,
    _LeftVar,
    _RightVar,
    VarMap,
    []
) :-
    !.

structural_tree_recursive_prefix_lines(
    Pred,
    BranchGoal,
    VarMap0,
    HelperName,
    DriverPos,
    Arity,
    LeftVar,
    RightVar,
    VarMap,
    Lines
) :-
    structural_tree_nested_if_goal_any(BranchGoal, PreGoals, NestedIfGoal, PostGoals),
    typr_goals_to_body(PreGoals, PreBody),
    compile_tail_recursive_pre_goals(PreBody, VarMap0, PreVarMap, GuardConditions, StepLines),
    tail_recursive_pre_branch_lines(GuardConditions, StepLines, BranchPreLines),
    typr_if_then_else_goal(NestedIfGoal, IfGoal, ThenGoal, ElseGoal),
    native_typr_if_condition(IfGoal, PreVarMap, IfCondition0),
    typr_condition_expr_text(IfCondition0, IfCondition),
    structural_tree_recursive_prefix_lines(
        Pred,
        ThenGoal,
        PreVarMap,
        HelperName,
        DriverPos,
        Arity,
        LeftVar,
        RightVar,
        ThenVarMap0,
        ThenLines
    ),
    structural_tree_recursive_prefix_lines(
        Pred,
        ElseGoal,
        PreVarMap,
        HelperName,
        DriverPos,
        Arity,
        LeftVar,
        RightVar,
        ElseVarMap0,
        ElseLines
    ),
    linear_recursive_post_goals_varmap(PostGoals, ThenVarMap0, ThenVarMap),
    linear_recursive_post_goals_varmap(PostGoals, ElseVarMap0, ElseVarMap),
    varmap_changed_vars(PreVarMap, ThenVarMap, ThenChangedVars0),
    varmap_changed_vars(PreVarMap, ElseVarMap, ElseChangedVars0),
    unique_vars_by_identity(ThenChangedVars0, ThenChangedVars),
    unique_vars_by_identity(ElseChangedVars0, ElseChangedVars),
    include_vars_by_identity(ThenChangedVars, ElseChangedVars, SharedChangedVars),
    SharedChangedVars \= [],
    merge_linear_recursive_branch_vars(
        SharedChangedVars,
        IfCondition,
        ThenVarMap,
        ElseVarMap,
        PreVarMap,
        MergedVarMap
    ),
    typr_goals_to_body(PostGoals, PostBody),
    structural_tree_recursive_prefix_lines(
        Pred,
        PostBody,
        MergedVarMap,
        HelperName,
        DriverPos,
        Arity,
        LeftVar,
        RightVar,
        VarMap,
        PostLines
    ),
    indent_lines(ThenLines, '    ', IndentedThenLines),
    indent_lines(ElseLines, '    ', IndentedElseLines),
    format(string(IfLine), '        if (~w) {', [IfCondition]),
    append(BranchPreLines, [IfLine|IndentedThenLines], Lines0),
    append(Lines0, ['        } else {'|IndentedElseLines], Lines1),
    append(Lines1, ['        };'], Lines2),
    append(Lines2, PostLines, Lines).
structural_tree_recursive_prefix_lines(
    Pred,
    BranchGoal,
    VarMap0,
    HelperName,
    DriverPos,
    Arity,
    LeftVar,
    RightVar,
    VarMap,
    Lines
) :-
    OutputPos is Arity,
    linear_recursive_invariant_positions(Arity, DriverPos, InvariantPositions),
    normalize_typr_goals(BranchGoal, Goals),
    (   split_typr_multicall_goals(Pred, Goals, PreGoals, RecCalls, PostGoals)
    ;   split_typr_multicall_goal_prefix(Pred, Goals, PreGoals, RecCalls),
        PostGoals = []
    ),
    typr_goals_to_body(PreGoals, PreBody),
    compile_tail_recursive_pre_goals(PreBody, VarMap0, PreVarMap, GuardConditions, StepLines),
    tail_recursive_pre_branch_lines(GuardConditions, StepLines, BranchPreLines),
    build_structural_tree_call_lines(
        RecCalls,
        HelperName,
        DriverPos,
        OutputPos,
        LeftVar,
        RightVar,
        InvariantPositions,
        PreVarMap,
        CallVarMap,
        CallLines
    ),
    linear_recursive_post_goals_varmap(PostGoals, CallVarMap, VarMap),
    append(BranchPreLines, CallLines, Lines).
structural_tree_recursive_prefix_lines(
    Pred,
    BranchGoal,
    VarMap0,
    _HelperName,
    _DriverPos,
    _Arity,
    _LeftVar,
    _RightVar,
    VarMap,
    []
) :-
    normalize_typr_goals(BranchGoal, Goals),
    \+ contains_recursive_goal(Pred, Goals),
    linear_recursive_post_goals_varmap(Goals, VarMap0, VarMap).

add_structural_tree_value_var(ValueVar, VarMap, [ValueVar-"value"|VarMap]) :-
    var(ValueVar),
    !.
add_structural_tree_value_var(_ValueVar, VarMap, VarMap).

split_typr_multicall_goals(Pred, Goals, PreGoals, RecCalls, PostGoals) :-
    split_non_recursive_prefix(Pred, Goals, PreGoals, RecAndPostGoals),
    take_recursive_goal_prefix(Pred, RecAndPostGoals, RecCalls, PostGoals),
    RecCalls = [_|[_|_]],
    PostGoals \= [],
    \+ contains_recursive_goal(Pred, PostGoals).

split_typr_multicall_goal_prefix(Pred, Goals, PreGoals, RecCalls) :-
    split_non_recursive_prefix(Pred, Goals, PreGoals, RecAndPostGoals),
    take_recursive_goal_prefix(Pred, RecAndPostGoals, RecCalls, PostGoals),
    RecCalls = [_|[_|_]],
    PostGoals == [].

split_non_recursive_prefix(_Pred, [], [], []).
split_non_recursive_prefix(Pred, [Goal|Rest], [], [Goal|Rest]) :-
    goal_calls_predicate(Pred, Goal),
    !.
split_non_recursive_prefix(Pred, [Goal|Rest], [Goal|PreGoals], PostGoals) :-
    split_non_recursive_prefix(Pred, Rest, PreGoals, PostGoals).

take_recursive_goal_prefix(Pred, [Goal|Rest], [Goal|RecGoals], PostGoals) :-
    goal_calls_predicate(Pred, Goal),
    !,
    take_recursive_goal_prefix(Pred, Rest, RecGoals, PostGoals).
take_recursive_goal_prefix(_Pred, Goals, [], Goals).

contains_recursive_goal(Pred, [Goal|_]) :-
    goal_calls_predicate(Pred, Goal),
    !.
contains_recursive_goal(Pred, [_|Rest]) :-
    contains_recursive_goal(Pred, Rest).

goal_calls_predicate(Pred, Goal0) :-
    nonvar(Goal0),
    Goal0 = _Module:Goal,
    !,
    goal_calls_predicate(Pred, Goal).
goal_calls_predicate(Pred, Goal) :-
    nonvar(Goal),
    compound(Goal),
    functor(Goal, Pred, _).

recursive_goal_count(Pred, Body, Count) :-
    findall(
        Goal,
        (
            sub_term(Goal, Body),
            goal_calls_predicate(Pred, Goal)
        ),
        RecGoals
    ),
    length(RecGoals, Count).

typr_goals_to_body([], true).
typr_goals_to_body([Goal], Goal) :-
    !.
typr_goals_to_body([Goal|Rest], (Goal, RestBody)) :-
    typr_goals_to_body(Rest, RestBody).

build_typr_tree_recursive_call_lines(RecCalls, HelperName, VarMap0, VarMapOut, Lines) :-
    build_typr_tree_recursive_call_lines(RecCalls, HelperName, 1, VarMap0, VarMapOut, Lines).

build_typr_tree_recursive_call_lines([], _HelperName, _Index, VarMap, VarMap, []).
build_typr_tree_recursive_call_lines([RecCall|Rest], HelperName, Index, VarMap0, VarMapOut, [Line|RestLines]) :-
    RecCall =.. [_PredName, CallInput, CallOutput],
    var(CallOutput),
    typr_translate_r_expr(CallInput, VarMap0, CallInputExpr),
    format(string(CallName), 'call_~w', [Index]),
    format(string(Line), '        ~w = ~w(~w);', [CallName, HelperName, CallInputExpr]),
    update_typr_expr_varmap(VarMap0, CallOutput, CallName, VarMap1),
    NextIndex is Index + 1,
    build_typr_tree_recursive_call_lines(Rest, HelperName, NextIndex, VarMap1, VarMapOut, RestLines).

single_linear_recursive_driver_pos(Arity, RecHeadArgs, RecCallArgs, DriverPos) :-
    InputLast is Arity - 1,
    findall(Pos, (
        between(1, InputLast, Pos),
        linear_recursive_changed_arg(RecHeadArgs, RecCallArgs, Pos)
    ), DriverPositions),
    DriverPositions = [DriverPos].

linear_recursive_changed_arg(HeadArgs, CallArgs, Pos) :-
    nth1(Pos, HeadArgs, HeadArg),
    nth1(Pos, CallArgs, CallArg),
    \+ linear_recursive_same_arg(HeadArg, CallArg).

linear_recursive_same_arg(Left, Right) :-
    Left == Right,
    !.
linear_recursive_same_arg(Left, Right) :-
    nonvar(Left),
    nonvar(Right),
    Left =@= Right.

linear_recursive_invariant_positions(Arity, DriverPos, Positions) :-
    InputLast is Arity - 1,
    findall(Pos, (
        between(1, InputLast, Pos),
        Pos =\= DriverPos
    ), Positions).

base_linear_invariants_ok(_BaseArgs, _RecHeadArgs, []).
base_linear_invariants_ok(BaseArgs, RecHeadArgs, [Pos|Rest]) :-
    nth1(Pos, BaseArgs, BaseArg),
    nth1(Pos, RecHeadArgs, RecArg),
    linear_recursive_base_invariant_ok(BaseArg, RecArg),
    base_linear_invariants_ok(BaseArgs, RecHeadArgs, Rest).

linear_recursive_base_invariant_ok(BaseArg, _RecArg) :-
    var(BaseArg),
    !.
linear_recursive_base_invariant_ok(BaseArg, RecArg) :-
    nonvar(RecArg),
    BaseArg =@= RecArg.

build_linear_recursive_varmap(Args, Positions, VarMap) :-
    build_linear_recursive_varmap(Args, Positions, [], RevVarMap),
    reverse(RevVarMap, VarMap).

build_linear_recursive_varmap(_Args, [], VarMap, VarMap).
build_linear_recursive_varmap(Args, [Pos|Rest], VarMap0, VarMap) :-
    nth1(Pos, Args, Var),
    (   var(Var)
    ->  format(string(ArgName), 'arg~w', [Pos]),
        VarMap1 = [Var-ArgName|VarMap0]
    ;   VarMap1 = VarMap0
    ),
    build_linear_recursive_varmap(Args, Rest, VarMap1, VarMap).

linear_recursive_result_name(Arity, ResultName) :-
    ResultIndex is Arity + 1,
    format(string(ResultName), 'v~w', [ResultIndex]).

linear_recursive_guard_expr(true, _VarMap, 'TRUE') :-
    !.
linear_recursive_guard_expr(PreGoals, VarMap, GuardExpr) :-
    normalize_typr_goals(PreGoals, Goals0),
    exclude(linear_recursive_step_goal, Goals0, GuardGoals),
    maplist(linear_recursive_guard_condition(VarMap), GuardGoals, GuardConditions),
    combine_typr_conditions(GuardConditions, GuardExpr).

linear_recursive_step_goal(Goal) :-
    Goal = (_Var is _Expr).

linear_recursive_guard_condition(VarMap, Goal, GuardCondition) :-
    native_typr_guard_goal(Goal, VarMap, GuardCondition).

linear_recursive_output_expr(PostGoals, OutputVar, VarMap, OutputExpr) :-
    normalize_typr_goals(PostGoals, [OutputVar is FoldTerm]),
    !,
    typr_translate_r_expr(FoldTerm, VarMap, OutputExpr).
linear_recursive_output_expr(PostGoals, OutputVar, VarMap, OutputExpr) :-
    typr_if_then_else_goal(PostGoals, IfGoal, ThenGoal, ElseGoal),
    linear_recursive_branch_output_expr(ThenGoal, OutputVar, VarMap, ThenExpr),
    linear_recursive_branch_output_expr(ElseGoal, OutputVar, VarMap, ElseExpr),
    native_typr_if_condition(IfGoal, VarMap, IfCondition),
    format(string(OutputExpr), 'if (~w) { ~w } else { ~w }', [IfCondition, ThenExpr, ElseExpr]).
linear_recursive_output_expr(PostGoals, OutputVar, VarMap, OutputExpr) :-
    normalize_typr_goals(PostGoals, Goals),
    linear_recursive_post_goals_varmap(Goals, VarMap, ResolvedVarMap),
    lookup_typr_var(OutputVar, ResolvedVarMap, OutputExpr).

linear_recursive_branch_output_expr(Goal, OutputVar, VarMap, OutputExpr) :-
    normalize_typr_goals(Goal, [OutputVar is FoldTerm]),
    typr_translate_r_expr(FoldTerm, VarMap, OutputExpr).

linear_recursive_post_goals_varmap([], VarMap, VarMap).
linear_recursive_post_goals_varmap([Goal|Rest], VarMap0, VarMap) :-
    linear_recursive_post_goal_varmap(Goal, VarMap0, VarMap1),
    linear_recursive_post_goals_varmap(Rest, VarMap1, VarMap).

linear_recursive_post_goal_varmap(Goal, VarMap0, VarMap) :-
    Goal = (Var is Expr),
    !,
    typr_translate_r_expr(Expr, VarMap0, ResolvedExpr),
    update_typr_expr_varmap(VarMap0, Var, ResolvedExpr, VarMap).
linear_recursive_post_goal_varmap(Goal, VarMap0, VarMap) :-
    typr_if_then_else_goal(Goal, IfGoal, ThenGoal, ElseGoal),
    !,
    native_typr_if_condition(IfGoal, VarMap0, IfCondition),
    normalize_typr_goals(ThenGoal, ThenGoals),
    normalize_typr_goals(ElseGoal, ElseGoals),
    linear_recursive_post_goals_varmap(ThenGoals, VarMap0, ThenVarMap),
    linear_recursive_post_goals_varmap(ElseGoals, VarMap0, ElseVarMap),
    varmap_changed_vars(VarMap0, ThenVarMap, ThenChangedVars0),
    varmap_changed_vars(VarMap0, ElseVarMap, ElseChangedVars0),
    unique_vars_by_identity(ThenChangedVars0, ThenChangedVars),
    unique_vars_by_identity(ElseChangedVars0, ElseChangedVars),
    include_vars_by_identity(ThenChangedVars, ElseChangedVars, SharedChangedVars),
    SharedChangedVars \= [],
    merge_linear_recursive_branch_vars(
        SharedChangedVars,
        IfCondition,
        ThenVarMap,
        ElseVarMap,
        VarMap0,
        VarMap
    ).

merge_linear_recursive_branch_vars([], _IfCondition, _ThenVarMap, _ElseVarMap, VarMap, VarMap).
merge_linear_recursive_branch_vars([Var|Rest], IfCondition, ThenVarMap, ElseVarMap, VarMap0, VarMap) :-
    lookup_typr_var(Var, ThenVarMap, ThenExpr),
    lookup_typr_var(Var, ElseVarMap, ElseExpr),
    format(string(MergedExpr), '(if (~w) { ~w } else { ~w })', [IfCondition, ThenExpr, ElseExpr]),
    update_typr_expr_varmap(VarMap0, Var, MergedExpr, VarMap1),
    merge_linear_recursive_branch_vars(Rest, IfCondition, ThenVarMap, ElseVarMap, VarMap1, VarMap).

update_typr_expr_varmap(VarMap0, Var, Expr, [Var-Expr|FilteredVarMap]) :-
    remove_var_mapping(Var, VarMap0, FilteredVarMap).

varmap_changed_vars(VarMap0, VarMap1, ChangedVars) :-
    varmap_changed_vars(VarMap0, VarMap1, [], RevChangedVars),
    reverse(RevChangedVars, ChangedVars).

varmap_changed_vars(_VarMap0, [], ChangedVars, ChangedVars).
varmap_changed_vars(VarMap0, [Var-Expr1|Rest], ChangedVars0, ChangedVars) :-
    (   lookup_typr_var(Var, VarMap0, Expr0)
    ->  (   Expr0 \= Expr1
        ->  ChangedVars1 = [Var|ChangedVars0]
        ;   ChangedVars1 = ChangedVars0
        )
    ;   ChangedVars1 = [Var|ChangedVars0]
    ),
    varmap_changed_vars(VarMap0, Rest, ChangedVars1, ChangedVars).

typr_linear_seq_expr(BaseInput, Step, down, SeqExpr) :-
    !,
    LoopEnd is BaseInput + Step,
    (   Step =:= 1
    ->  format(string(SeqExpr), 'seq(current_input, ~w)', [LoopEnd])
    ;   NegStep is -Step,
        format(string(SeqExpr), 'seq(current_input, ~w, by = ~w)', [LoopEnd, NegStep])
    ).
typr_linear_seq_expr(BaseInput, Step, up, SeqExpr) :-
    LoopStart is BaseInput + Step,
    (   Step =:= 1
    ->  format(string(SeqExpr), 'seq(~w, current_input)', [LoopStart])
    ;   format(string(SeqExpr), 'seq(~w, current_input, by = ~w)', [LoopStart, Step])
    ).

build_typr_linear_recursive_body(
    Pred,
    linear_loop_spec{
        base_input_literal: BaseInputLiteral,
        base_output_expr: BaseOutputExpr,
        recursive_guard_expr: RecursiveGuardExpr,
        seq_expr: SeqExpr,
        fold_expr: FoldExpr,
        input_arg_name: InputArgName,
        output_arg_name: OutputArgName,
        result_name: ResultName
    },
    Body
) :-
    linear_recursive_guard_lines(RecursiveGuardExpr, Pred, GuardLines),
    format(string(BaseLine), '        ~w', [BaseOutputExpr]),
    format(string(AccInitLine), '        acc = ~w;', [BaseOutputExpr]),
    format(string(LoopLine), '        for (current in ~w) {', [SeqExpr]),
    format(string(AccStepLine), '            acc = ~w;', [FoldExpr]),
    format(string(BaseCaseIfLine), '    if (identical(current_input, ~w)) {', [BaseInputLiteral]),
    format(string(ResultIntroLine), 'let ~w <- @{', [ResultName]),
    format(string(CurrentInputLine), '    current_input = ~w;', [InputArgName]),
    format(string(AssignResultLine), '~w <- ~w;', [OutputArgName, ResultName]),
    atomic_list_concat(
        [
            ResultIntroLine,
            'local({',
            CurrentInputLine,
            BaseCaseIfLine,
            BaseLine,
            '    } else {'
        ],
        '\n',
        Prefix
    ),
    append(
        [
            Prefix
        ],
        GuardLines,
        RawLines0
    ),
    append(
        RawLines0,
        [
            AccInitLine,
            LoopLine,
            AccStepLine,
            '        };',
            '        acc',
            '    }',
            '})',
            '}@;',
            AssignResultLine,
            OutputArgName
        ],
        RawLines
    ),
    atomic_list_concat(RawLines, '\n', Body).

build_typr_linear_recursive_list_body(
    list_loop_spec{
        base_output_expr: BaseOutputExpr,
        fold_expr: FoldExpr,
        input_arg_name: InputArgName,
        output_arg_name: OutputArgName,
        result_name: ResultName
    },
    Body
) :-
    format(string(BaseLine), '        ~w', [BaseOutputExpr]),
    format(string(AccInitLine), '        acc = ~w;', [BaseOutputExpr]),
    format(string(AccStepLine), '            acc = ~w;', [FoldExpr]),
    format(string(ResultIntroLine), 'let ~w <- @{', [ResultName]),
    format(string(CurrentInputLine), '    current_input = ~w;', [InputArgName]),
    format(string(AssignResultLine), '~w <- ~w;', [OutputArgName, ResultName]),
    atomic_list_concat(
        [
            ResultIntroLine,
            'local({',
            CurrentInputLine,
            '    if (length(current_input) == 0) {',
            BaseLine,
            '    } else {',
            AccInitLine,
            '        for (current in rev(current_input)) {',
            AccStepLine,
            '        };',
            '        acc',
            '    }',
            '})',
            '}@;',
            AssignResultLine,
            OutputArgName
        ],
        '\n',
        Body
    ).

build_typr_tree_recursive_body(
    _Pred,
    tree_recursive_spec{
        base_cases: BaseCases,
        recursive_guard_expr: GuardExpr,
        step_lines: StepLines,
        call_lines: CallLines,
        result_expr: ResultExpr,
        input_arg_name: InputArgName,
        output_arg_name: OutputArgName,
        result_name: ResultName,
        helper_name: HelperName,
        memo_name: MemoName
    },
    Body
) :-
    format(string(ResultIntroLine), 'let ~w <- @{', [ResultName]),
    format(string(MemoLine), '    ~w <- new.env(hash=TRUE, parent=emptyenv());', [MemoName]),
    format(string(HelperLine), '    ~w <- function(current_input) {', [HelperName]),
    format(string(KeyLine), '        key <- as.character(current_input);', []),
    format(string(HelperCallLine), '    ~w(~w)', [HelperName, InputArgName]),
    format(string(AssignResultLine), '~w <- ~w;', [OutputArgName, ResultName]),
    tree_recursive_dispatch_lines(BaseCases, GuardExpr, StepLines, CallLines, ResultExpr, MemoName, DispatchLines),
    append(
        [
            ResultIntroLine,
            'local({',
            MemoLine,
            HelperLine,
            KeyLine
        ],
        DispatchLines,
        RawLines0
    ),
    append(
        RawLines0,
        [
            '    };',
            HelperCallLine,
            '})',
            '}@;',
            AssignResultLine,
            OutputArgName
        ],
        RawLines
    ),
    atomic_list_concat(RawLines, '\n', Body).

build_typr_structural_tree_recursive_body(
    Pred,
    structural_tree_spec{
        base_output_expr: BaseOutputExpr,
        guard_expr: GuardExpr,
        step_lines: StepLines,
        call_lines: CallLines,
        result_expr: ResultExpr,
        input_arg_name: _InputArgName,
        output_arg_name: OutputArgName,
        result_name: ResultName,
        helper_name: HelperName,
        helper_param_list: HelperParamList,
        helper_call_arg_list: HelperCallArgList
    },
    Body
) :-
    format(string(ResultIntroLine), 'let ~w <- @{', [ResultName]),
    format(string(HelperLine), '    ~w <- function(~w) {', [HelperName, HelperParamList]),
    format(string(HelperCallLine), '    ~w(~w)', [HelperName, HelperCallArgList]),
    format(string(AssignResultLine), '~w <- ~w;', [OutputArgName, ResultName]),
    structural_tree_dispatch_lines(Pred, BaseOutputExpr, GuardExpr, StepLines, CallLines, ResultExpr, DispatchLines),
    append(
        [
            ResultIntroLine,
            'local({',
            HelperLine
        ],
        DispatchLines,
        RawLines0
    ),
    append(
        RawLines0,
        [
            '    };',
            HelperCallLine,
            '})',
            '}@;',
            AssignResultLine,
            OutputArgName
        ],
        RawLines
    ),
    atomic_list_concat(RawLines, '\n', Body).

structural_tree_dispatch_lines(Pred, BaseOutputExpr, GuardExpr, StepLines, CallLines, ResultExpr, Lines) :-
    format(string(BaseLine), '            ~w', [BaseOutputExpr]),
    format(string(ResultLine), '            result = ~w;', [ResultExpr]),
    format(string(StopLine), '            stop("No matching recursive clause for ~w")', [Pred]),
    structural_tree_guard_lines(GuardExpr, Pred, GuardLines),
    indent_lines(StepLines, '    ', IndentedStepLines),
    append([
        '        if (length(current_tree) == 0) {',
        BaseLine,
        '        } else if (length(current_tree) == 3) {',
        '            value = .subset2(current_tree, 1);',
        '            left = .subset2(current_tree, 2);',
        '            right = .subset2(current_tree, 3);'
    ], GuardLines, Lines0),
    append(Lines0, IndentedStepLines, Lines1),
    append(Lines1, CallLines, Lines2),
    append(Lines2, [
        ResultLine,
        '            result',
        '        } else {',
        StopLine,
        '        }'
    ], Lines).

structural_tree_guard_lines('TRUE', _PredName, []) :-
    !.
structural_tree_guard_lines(GuardExpr, PredName, [
    IfLine,
    StopLine,
    '            };'
]) :-
    format_string('            if (!(~w)) {', [GuardExpr], IfLine),
    format_string('                stop("No matching recursive clause for ~w")', [PredName], StopLine).

indent_lines([], _Prefix, []).
indent_lines([Line|Rest], Prefix, [IndentedLine|IndentedRest]) :-
    string_concat(Prefix, Line, IndentedLine),
    indent_lines(Rest, Prefix, IndentedRest).

tree_recursive_dispatch_lines(BaseCases, GuardExpr, StepLines, CallLines, ResultExpr, MemoName, Lines) :-
    format(string(MemoCheckLine), '        if (exists(key, envir=~w, inherits=FALSE)) {', [MemoName]),
    format(string(MemoGetLine), '            get(key, envir=~w, inherits=FALSE)', [MemoName]),
    base_case_branch_lines(BaseCases, BaseLines),
    recursive_branch_lines(GuardExpr, StepLines, CallLines, ResultExpr, MemoName, RecursiveLines),
    append([MemoCheckLine, MemoGetLine], BaseLines, Lines0),
    append(Lines0, RecursiveLines, Lines1),
    append(Lines1, [], Lines).

base_case_branch_lines([], []).
base_case_branch_lines([base_case(InputLiteral, OutputExpr)|Rest], [IfLine, OutputLine|RestLines]) :-
    format(string(IfLine), '        } else if (identical(current_input, ~w)) {', [InputLiteral]),
    format(string(OutputLine), '            ~w', [OutputExpr]),
    base_case_branch_lines(Rest, RestLines).

recursive_branch_lines('TRUE', StepLines, CallLines, ResultExpr, MemoName, Lines) :-
    !,
    recursive_block_lines(StepLines, CallLines, ResultExpr, MemoName, InnerLines),
    append(['        } else {'], InnerLines, Lines0),
    append(Lines0, ['        }'], Lines).
recursive_branch_lines(GuardExpr, StepLines, CallLines, ResultExpr, MemoName, [IfLine|Rest]) :-
    format(string(IfLine), '        } else if (~w) {', [GuardExpr]),
    recursive_block_lines(StepLines, CallLines, ResultExpr, MemoName, InnerLines),
    append(
        InnerLines,
        [
            '        } else {',
            '            stop("No matching recursive clause for recursive helper")',
            '        }'
        ],
        Rest
    ).

recursive_block_lines(StepLines, CallLines, ResultExpr, MemoName, Lines) :-
    format(string(ResultLine), '        result = ~w;', [ResultExpr]),
    format(string(AssignLine), '        assign(key, result, envir=~w);', [MemoName]),
    append(
        StepLines,
        CallLines,
        Lines0
    ),
    append(
        Lines0,
        [
            ResultLine,
            AssignLine,
            '        result'
        ],
        Lines
    ).

extract_typr_linear_step_info(RecBody, DriverPos, RecCallArgs, InputVar, Step, Direction) :-
    nth1(DriverPos, RecCallArgs, RecInputVar),
    normalize_typr_goals(RecBody, Goals),
    member(RecInputVar is StepExpr, Goals),
    extract_typr_linear_step_from_expr(StepExpr, InputVar, Step, Direction),
    !.

extract_typr_linear_step_from_expr(A - B, InputVar, Step, down) :-
    A == InputVar,
    integer(B),
    B > 0,
    !,
    Step = B.
extract_typr_linear_step_from_expr(A + B, InputVar, Step, up) :-
    A == InputVar,
    integer(B),
    B > 0,
    !,
    Step = B.
extract_typr_linear_step_from_expr(A + B, InputVar, Step, down) :-
    A == InputVar,
    integer(B),
    B < 0,
    !,
    Step is abs(B).

linear_recursive_guard_lines('TRUE', _Pred, []) :-
    !.
linear_recursive_guard_lines(GuardExpr, Pred, [IfLine, StopLine, '        };']) :-
    format_string('        if (!(~w)) {', [GuardExpr], IfLine),
    format_string('            stop("No matching recursive clause for ~w")', [Pred], StopLine).

clause_calls_predicate(Pred, Arity, _Head-Body) :-
    calls_predicate(Pred, Arity, Body).

tail_recursive_loop_spec(
    Pred,
    BaseHead-BaseBody,
    RecHead-RecBody,
    loop_spec{
        base_input_literal: BaseInputLiteral,
        guard_expr: GuardExpr,
        step_lines: StepLines,
        next_input_expr: NextInputExpr,
        next_acc_expr: NextAccExpr
    }
) :-
    BaseBody == true,
    BaseHead =.. [_PredName, BaseInput, BaseAcc, BaseOut],
    var(BaseAcc),
    var(BaseOut),
    BaseAcc == BaseOut,
    r_literal(BaseInput, BaseInputLiteral),
    RecHead =.. [_PredName2, InputVar, AccVar, OutVar],
    var(InputVar),
    var(AccVar),
    var(OutVar),
    split_body_at_recursive_call(RecBody, Pred, PreGoals, RecCall, PostGoals),
    PostGoals == true,
    RecCall =.. [_PredName3, NextInputTerm, NextAccTerm, NextOutVar],
    NextOutVar == OutVar,
    VarMap0 = [InputVar-"current_input", AccVar-"current_acc", OutVar-"arg3"],
    compile_tail_recursive_pre_goals(PreGoals, VarMap0, VarMap, GuardConditions, StepLines),
    raw_guard_expr(GuardConditions, GuardExpr),
    typr_translate_r_expr(NextInputTerm, VarMap, NextInputExpr),
    typr_translate_r_expr(NextAccTerm, VarMap, NextAccExpr).

compile_tail_recursive_pre_goals(true, VarMap, VarMap, [], []) :-
    !.
compile_tail_recursive_pre_goals(PreGoals, VarMap0, VarMap, GuardConditions, StepLines) :-
    normalize_typr_goals(PreGoals, Goals),
    compile_tail_recursive_pre_goals_list(Goals, VarMap0, VarMap, GuardConditions, StepLines).

compile_tail_recursive_pre_goals_list([], VarMap, VarMap, [], []).
compile_tail_recursive_pre_goals_list([Goal|Rest], VarMap0, VarMap, GuardConditions, StepLines) :-
    tail_recursive_pre_goal(Goal, VarMap0, VarMap1, GoalGuards, GoalLines),
    compile_tail_recursive_pre_goals_list(Rest, VarMap1, VarMap, RestGuards, RestLines),
    append(GoalGuards, RestGuards, GuardConditions),
    append(GoalLines, RestLines, StepLines).

tail_recursive_pre_goal(_Module:Goal, VarMap0, VarMap, GuardConditions, StepLines) :-
    !,
    tail_recursive_pre_goal(Goal, VarMap0, VarMap, GuardConditions, StepLines).
tail_recursive_pre_goal(Goal, VarMap0, VarMap, [], Lines) :-
    typr_if_then_else_goal(Goal, IfGoal, ThenGoal, ElseGoal),
    !,
    compile_tail_recursive_pre_if_then_else_goal(
        IfGoal,
        ThenGoal,
        ElseGoal,
        VarMap0,
        VarMap,
        Lines
    ).
tail_recursive_pre_goal(Goal, VarMap0, VarMap, [], [Line]) :-
    Goal = (Var is Expr),
    !,
    ensure_tail_temp_var(VarMap0, Var, TempName, VarMap),
    typr_translate_r_expr(Expr, VarMap0, ResolvedExpr),
    format(string(Line), '        ~w = ~w;', [TempName, ResolvedExpr]).
tail_recursive_pre_goal(Goal, VarMap0, VarMap, [], [Line]) :-
    Goal = (Var = Expr),
    var(Var),
    !,
    ensure_tail_temp_var(VarMap0, Var, TempName, VarMap),
    typr_translate_r_expr(Expr, VarMap0, ResolvedExpr),
    format(string(Line), '        ~w = ~w;', [TempName, ResolvedExpr]).
tail_recursive_pre_goal(Goal, VarMap, VarMap, [GuardCondition], []) :-
    native_typr_guard_goal(Goal, VarMap, GuardCondition),
    !.

compile_tail_recursive_pre_if_then_else_goal(
    IfGoal,
    ThenGoal,
    ElseGoal,
    VarMap0,
    VarMap,
    Lines
) :-
    native_typr_if_condition(IfGoal, VarMap0, IfCondition0),
    typr_condition_expr_text(IfCondition0, IfCondition),
    compile_tail_recursive_pre_goals(ThenGoal, VarMap0, VarMap1, ThenGuardConditions, ThenStepLines),
    compile_tail_recursive_pre_goals(ElseGoal, VarMap1, VarMap, ElseGuardConditions, ElseStepLines),
    tail_recursive_pre_branch_lines(ThenGuardConditions, ThenStepLines, ThenLines),
    tail_recursive_pre_branch_lines(ElseGuardConditions, ElseStepLines, ElseLines),
    indent_lines(ThenLines, '    ', IndentedThenLines),
    indent_lines(ElseLines, '    ', IndentedElseLines),
    format(string(IfLine), '        if (~w) {', [IfCondition]),
    append(
        [IfLine|IndentedThenLines],
        ['        } else {'|IndentedElseLines],
        Lines0
    ),
    append(Lines0, ['        };'], Lines).

tail_recursive_pre_branch_lines(GuardConditions, StepLines, Lines) :-
    raw_guard_expr(GuardConditions, GuardExpr),
    tail_recursive_pre_branch_guard_lines(GuardExpr, GuardLines),
    append(GuardLines, StepLines, Lines).

tail_recursive_pre_branch_guard_lines('TRUE', []) :-
    !.
tail_recursive_pre_branch_guard_lines(GuardExpr, [
    IfLine,
    StopLine,
    '        };'
]) :-
    format_string('        if (!(~w)) {', [GuardExpr], IfLine),
    StopLine = '            stop("No matching recursive branch")'.

ensure_tail_temp_var(VarMap, Var, Name, VarMap, existing) :-
    lookup_typr_var(Var, VarMap, Name),
    !.
ensure_tail_temp_var(VarMap0, Var, Name, [Var-Name|VarMap0], new) :-
    next_tail_temp_index(VarMap0, NextIndex),
    format(string(Name), 'step_~w', [NextIndex]).
ensure_tail_temp_var(VarMap, Var, Name, VarMapOut) :-
    ensure_tail_temp_var(VarMap, Var, Name, VarMapOut, _).

next_tail_temp_index(VarMap, NextIndex) :-
    findall(Index, (
        member(_-Name, VarMap),
        string(Name),
        sub_string(Name, 0, 5, _, "step_"),
        sub_string(Name, 5, _, 0, Digits),
        number_string(Index, Digits)
    ), Indices),
    (   Indices = []
    ->  NextIndex = 1
    ;   max_list(Indices, MaxIndex),
        NextIndex is MaxIndex + 1
    ).

raw_guard_expr([], 'TRUE').
raw_guard_expr(GuardConditions, GuardExpr) :-
    combine_typr_conditions(GuardConditions, CombinedCondition),
    typr_condition_expr_text(CombinedCondition, GuardExpr).

build_typr_tail_recursive_body(
    RecHead-_RecBody,
    loop_spec{
        base_input_literal: BaseInputLiteral,
        guard_expr: GuardExpr,
        step_lines: StepLines,
        next_input_expr: NextInputExpr,
        next_acc_expr: NextAccExpr
    },
    Body
) :-
    RecHead =.. [PredNameAtom, HeadInputVar, HeadAccVar, HeadOutVar],
    atom_string(PredNameAtom, PredName),
    build_head_varmap([HeadInputVar, HeadAccVar, HeadOutVar], 1, HeadVarMap0),
    reserve_typr_internal_var(HeadVarMap0, _ResultToken, ResultName, _HeadVarMap, new),
    lookup_typr_var(HeadOutVar, HeadVarMap0, OutputName),
    tail_recursive_guard_lines(GuardExpr, PredName, GuardLines),
    format_string('    while (!identical(current_input, ~w)) {', [BaseInputLiteral], WhileLine),
    format_string('        current_input = ~w;', [NextInputExpr], NextInputLine),
    format_string('        current_acc = ~w;', [NextAccExpr], NextAccLine),
    append(
        [
            'local({',
            '    current_input = arg1;',
            '    current_acc = arg2;',
            WhileLine
        ],
        GuardLines,
        RawLines0
    ),
    append(
        RawLines0,
        StepLines,
        RawLines1
    ),
    append(
        RawLines1,
        [
            NextInputLine,
            NextAccLine,
            '    };',
            '    current_acc',
            '})'
        ],
        RawLines
    ),
    atomic_list_concat(RawLines, '\n', RawExpr),
    format(string(Body),
'let ~w <- @{
~w
}@;
~w <- ~w;
~w', [ResultName, RawExpr, OutputName, ResultName, OutputName]).

tail_recursive_guard_lines('TRUE', _PredName, []) :-
    !.
tail_recursive_guard_lines(GuardExpr, PredName, [IfLine, StopLine, '        };']) :-
    format_string('        if (!(~w)) {', [GuardExpr], IfLine),
    format_string('            stop("No matching recursive clause for ~w")', [PredName], StopLine).

format_string(Format, Args, String) :-
    format(string(String), Format, Args).

compile_typr_transitive_closure(Module, Pred/Arity, BasePred, TypedMode, Code) :-
    resolve_node_type(Pred/Arity, BasePred/2, NodeTypeTerm),
    read_file_to_string('templates/targets/typr/transitive_closure.mustache', Template, []),
    atom_string(Pred, PredStr),
    atom_string(BasePred, BaseStr),
    annotation_suffix(TypedMode, NodeTypeTerm, NodeAnnotation),
    annotation_suffix(TypedMode, NodeTypeTerm, AddFromAnnotation),
    annotation_suffix(TypedMode, NodeTypeTerm, AddToAnnotation),
    annotation_suffix(TypedMode, list(NodeTypeTerm), AllReturnAnnotation),
    annotation_suffix(TypedMode, boolean, CheckReturnAnnotation),
    empty_collection_expr(NodeTypeTerm, EmptyNodesExpr),
    base_seed_code(Module, BasePred, SeedCode),
    render_template(Template, [
        pred=PredStr,
        base=BaseStr,
        add_from_annotation=AddFromAnnotation,
        add_to_annotation=AddToAnnotation,
        node_annotation=NodeAnnotation,
        all_return_annotation=AllReturnAnnotation,
        check_return_annotation=CheckReturnAnnotation,
        empty_nodes_expr=EmptyNodesExpr,
        seed_code=SeedCode
    ], Code).

compile_generic_typr(Module, Pred/Arity, Options, TypedMode, Code) :-
    init_r_target,
    findall(Head-Body, predicate_clause(Module, Pred, Arity, Head, Body), Clauses),
    build_typr_function(Module:Pred/Arity, Options, TypedMode, Clauses, Code).

annotation_suffix(off, _TypeTerm, "") :- !.
annotation_suffix(Mode, TypeTerm, Annotation) :-
    should_emit_annotation(Mode, TypeTerm),
    !,
    resolve_type(TypeTerm, typr, ConcreteType),
    format(string(Annotation), ": ~w", [ConcreteType]).
annotation_suffix(_, _, "").

should_emit_annotation(explicit, TypeTerm) :-
    nonvar(TypeTerm).
should_emit_annotation(infer, any).
should_emit_annotation(infer, TypeTerm) :-
    compound(TypeTerm),
    TypeTerm \= boolean.
should_emit_annotation(infer, TypeTerm) :-
    atomic(TypeTerm),
    TypeTerm \= atom,
    TypeTerm \= string,
    TypeTerm \= integer,
    TypeTerm \= float,
    TypeTerm \= number,
    TypeTerm \= boolean.

resolve_node_type(PredSpec, _BasePredSpec, TypeTerm) :-
    resolved_type_term(PredSpec, none, 1, TypeTerm),
    !.
resolve_node_type(_PredSpec, BasePredSpec, TypeTerm) :-
    resolved_type_term(BasePredSpec, none, 1, TypeTerm),
    !.
resolve_node_type(_PredSpec, _BasePredSpec, atom).

resolved_type_term(PredSpec, _FallbackPredSpec, ArgIndex, TypeTerm) :-
    predicate_arg_type(PredSpec, ArgIndex, TypeTerm),
    !.
resolved_type_term(_PredSpec, FallbackPredSpec, ArgIndex, TypeTerm) :-
    FallbackPredSpec \== none,
    predicate_arg_type(FallbackPredSpec, ArgIndex, TypeTerm).

build_typed_arg_list(PredSpec, FallbackPredSpec, Arity, TypedMode, ArgList) :-
    findall(ArgName, (
        between(1, Arity, Index),
        typed_argument_name(PredSpec, FallbackPredSpec, Index, TypedMode, ArgName)
    ), ArgNames),
    atomic_list_concat(ArgNames, ', ', ArgList).

typed_argument_name(PredSpec, FallbackPredSpec, Index, TypedMode, ArgName) :-
    format(string(BaseName), 'arg~w', [Index]),
    (   resolved_type_term(PredSpec, FallbackPredSpec, Index, TypeTerm)
    ->  annotation_suffix(TypedMode, TypeTerm, Annotation),
        string_concat(BaseName, Annotation, ArgName)
    ;   ArgName = BaseName
    ).

normalize_base_pred(BasePred/_, BasePred) :- !.
normalize_base_pred(BasePred, BasePred).

pred_indicator_parts(Module:Pred/Arity, Module, Pred, Arity) :- !.
pred_indicator_parts(Pred/Arity, user, Pred, Arity).

detect_transitive_closure(Module, Pred, 2, BasePred) :-
    functor(Head, Pred, 2),
    findall(Body, clause(Module:Head, Body), Bodies),
    Bodies \= [],
    partition(calls_predicate(Pred, 2), Bodies, RecursiveBodies, BaseBodies),
    BaseBodies \= [],
    RecursiveBodies \= [],
    transitive_closure_pattern(Pred, BaseBodies, RecursiveBodies, BasePred).

calls_predicate(Pred, Arity, Goal) :-
    sub_term(SubGoal, Goal),
    nonvar(SubGoal),
    compound(SubGoal),
    functor(SubGoal, Pred, Arity).

contains_goal((Left, _Right), Goal) :-
    contains_goal(Left, Goal).
contains_goal((_Left, Right), Goal) :-
    contains_goal(Right, Goal).
contains_goal(Goal, Goal) :-
    compound(Goal),
    Goal \= (_,_).

transitive_closure_pattern(Pred, BaseBodies, RecursiveBodies, BasePred) :-
    member(BaseBody, BaseBodies),
    BaseBody \= true,
    functor(BaseBody, BasePred, 2),
    BasePred \= Pred,
    member(RecBody, RecursiveBodies),
    RecBody = (BaseCall, RecCall),
    functor(BaseCall, BasePred, 2),
    functor(RecCall, Pred, 2),
    (   BaseCall =.. [BasePred, _X, Y],
        RecCall =.. [Pred, Y2, _Z],
        Y == Y2
    ;   BaseCall =.. [BasePred, Y, _X],
        RecCall =.. [Pred, Y2, _Z],
        Y == Y2
    ).

predicate_clause(Module, Pred, Arity, Head, Body) :-
    functor(Head, Pred, Arity),
    clause(Module:Head, Body).

build_typr_function(Module:Pred/Arity, Options, TypedMode, Clauses, Code) :-
    atom_string(Pred, PredStr),
    build_typed_arg_list(Pred/Arity, none, Arity, TypedMode, TypedArgList),
    typr_function_body(Module:Pred/Arity, Options, TypedMode, Clauses, ReturnType, RawBody),
    indent_text(RawBody, "\t", Body),
    format(string(Code),
'# Generated by UnifyWeaver TypR Target
# Predicate: ~w/~w

let ~w <- fn(~w): ~w {
~w
};
', [PredStr, Arity, PredStr, TypedArgList, ReturnType, Body]).

typr_function_body(_PredSpec, _Options, _TypedMode, [], "bool", "result <- false;\nresult").
typr_function_body(PredSpec, _Options, _TypedMode, Clauses, "bool", Body) :-
    all_fact_clauses(Clauses),
    fact_match_expression(PredSpec, Clauses, MatchExpr),
    !,
    format(string(Body), 'result <- @{ local({ ~w }) }@;\nresult', [MatchExpr]).
typr_function_body(PredSpec, Options, _TypedMode, Clauses, ReturnType, Body) :-
    generic_typr_return_type(PredSpec, Clauses, ReturnType),
    (   native_typr_clause_body(PredSpec, Clauses, Body)
    ->  true
    ;   wrapped_r_body_expression(PredSpec, Options, WrappedExpr),
        format(string(Body), 'result <- @{ ~w }@;\nresult', [WrappedExpr])
    ).

typr_declared_return_type(PredSpec, ReturnType) :-
    resolved_return_type(PredSpec, typr, ReturnType).

inferred_typr_return_type(Clauses, ReturnType) :-
    infer_clauses_return_type(Clauses, AbstractType),
    resolve_type(AbstractType, typr, ReturnType).

generic_typr_return_type(PredSpec, Clauses, ReturnType) :-
    (   typr_declared_return_type(PredSpec, ReturnType)
    ->  true
    ;   inferred_typr_return_type(Clauses, ReturnType)
    ->  true
    ;   ReturnType = "Any"
    ).

all_fact_clauses([]).
all_fact_clauses([_-true|Rest]) :-
    all_fact_clauses(Rest).

fact_match_expression(PredSpec, Clauses, Expr) :-
    findall(ClauseExpr, (
        member(Head-true, Clauses),
        fact_clause_expression(PredSpec, Head, ClauseExpr)
    ), ClauseExprs),
    ClauseExprs \= [],
    atomic_list_concat(ClauseExprs, ' || ', Expr).

fact_clause_expression(_PredSpec, Head, Expr) :-
    Head =.. [_|HeadArgs],
    findall(Condition, (
        nth1(Index, HeadArgs, HeadArg),
        fact_arg_condition(Index, HeadArg, Condition)
    ), Conditions0),
    exclude(=(true), Conditions0, Conditions),
    combine_typr_conditions(Conditions, Expr).

fact_arg_condition(_Index, HeadArg, true) :-
    var(HeadArg),
    !.
fact_arg_condition(Index, HeadArg, Condition) :-
    format(string(ArgName), 'arg~w', [Index]),
    r_literal(HeadArg, Literal),
    format(string(Condition), 'identical(~w, ~w)', [ArgName, Literal]).

base_seed_code(Module, BasePred, SeedCode) :-
    functor(BaseHead, BasePred, 2),
    findall(From-To, (
        clause(Module:BaseHead, true),
        BaseHead =.. [BasePred, From, To],
        nonvar(From),
        nonvar(To)
    ), Pairs),
    findall(Statement, (
        nth1(Index, Pairs, From-To),
        seed_statement(Index, BasePred, From, To, Statement)
    ), Statements),
    (   Statements = []
    ->  SeedCode = ""
    ;   atomic_list_concat(Statements, '\n', SeedCode)
    ).

seed_statement(Index, BasePred, From, To, Statement) :-
    r_literal(From, FromLiteral),
    r_literal(To, ToLiteral),
    format(string(Statement), '_seed_~w_~w <- add_~w(~w, ~w);', [BasePred, Index, BasePred, FromLiteral, ToLiteral]).

empty_collection_expr(atom, 'character()').
empty_collection_expr(string, 'character()').
empty_collection_expr(integer, 'integer()').
empty_collection_expr(float, 'numeric()').
empty_collection_expr(number, 'numeric()').
empty_collection_expr(boolean, 'logical()').
empty_collection_expr(any, 'c()').
empty_collection_expr(_, 'c()').

r_literal(Value, Literal) :-
    var(Value),
    !,
    Literal = 'NULL'.
r_literal(Value, Literal) :-
    string(Value),
    !,
    format(string(Literal), '"~s"', [Value]).
r_literal(Value, Literal) :-
    atom(Value),
    !,
    (   Value == true
    ->  Literal = 'TRUE'
    ;   Value == false
    ->  Literal = 'FALSE'
    ;   format(string(Literal), '"~w"', [Value])
    ).
r_literal(Value, Literal) :-
    number(Value),
    !,
    format(string(Literal), '~w', [Value]).
r_literal(Value, Literal) :-
    term_string(Value, Literal).

native_typr_clause_body(PredSpec, [Head-Body], Code) :-
    native_typr_clause(Head, Body, Condition, ClauseCode),
    !,
    (   Condition == 'TRUE'
    ->  Code = ClauseCode
    ;   pred_spec_name(PredSpec, PredName),
        branch_safe_typr_code(ClauseCode, BranchClauseCode),
        indent_text(BranchClauseCode, "\t", BranchCode),
        format(string(Code),
'if (~w) {
~w
} else {
	stop("No matching clause for ~w")
}', [Condition, BranchCode, PredName])
    ).
native_typr_clause_body(PredSpec, Clauses, Code) :-
    maplist(native_typr_clause_pair, Clauses, Branches),
    Branches \= [],
    pred_spec_name(PredSpec, PredName),
    branches_to_typr_if_chain(Branches, IfChain),
    format(string(Code), '~w else {\n\tstop("No matching clause for ~w")\n}', [IfChain, PredName]).

native_typr_clause_pair(Head-Body, branch(Condition, ClauseCode)) :-
    native_typr_clause(Head, Body, Condition, ClauseCode),
    !.

native_typr_clause(Head, Body, Condition, ClauseCode) :-
    Head =.. [Pred|HeadArgs],
    build_head_varmap(HeadArgs, 1, VarMap),
    typr_head_condition(HeadArgs, 1, HeadConditions),
    atom_string(Pred, PredName),
    native_typr_goal_sequence(Body, VarMap, PredName, GoalConditions, ClauseCode),
    append(HeadConditions, GoalConditions, Conditions),
    combine_typr_conditions(Conditions, Condition).

native_typr_goal_sequence(Body, VarMap, PredName, Conditions, Code) :-
    native_typr_goal_sequence(Body, VarMap, PredName, Conditions, Code, _VarMapOut).

native_typr_goal_sequence(Body, VarMap, PredName, Conditions, Code, VarMapOut) :-
    normalize_typr_goals(Body, Goals),
    Goals \= [],
    native_typr_prefix_goals(
        Goals,
        VarMap,
        PredName,
        false,
        none,
        Conditions,
        PrefixLines,
        TailCode,
        VarMapOut
    ),
    append(PrefixLines, [TailCode], BodyLines),
    atomic_list_concat(BodyLines, '\n', Code).

normalize_typr_goals(true, []) :- !.
normalize_typr_goals((Left, Right), Goals) :-
    !,
    normalize_typr_goals(Left, LeftGoals),
    normalize_typr_goals(Right, RightGoals),
    append(LeftGoals, RightGoals, Goals).
normalize_typr_goals(_Module:Goal, Goals) :-
    !,
    normalize_typr_goals(Goal, Goals).
normalize_typr_goals(Goal, [Goal]).

native_typr_prefix_goals([], VarMap, _PredName, _SeenOutput, none, [], [], 'true', VarMap).
native_typr_prefix_goals([], VarMap, _PredName, _SeenOutput, LastExpr, [], [], LastExpr, VarMap) :-
    LastExpr \== none.
native_typr_prefix_goals([Goal|Rest], VarMap0, PredName, _SeenOutput, _LastExpr0, Conditions, GoalLines, TailCode, VarMapOut) :-
    native_typr_multi_result_output_goal(Goal, VarMap0, PredName, VarMap1, OutputLines, OutExpr),
    append(OutputLines, RestLines, GoalLines),
    native_typr_prefix_goals(Rest, VarMap1, PredName, true, OutExpr, Conditions, RestLines, TailCode, VarMapOut).
native_typr_prefix_goals([Goal|Rest], VarMap0, PredName, _SeenOutput, _LastExpr0, Conditions, [Line|RestLines], TailCode, VarMapOut) :-
    native_typr_output_goal(Goal, VarMap0, PredName, VarMap1, Line, OutExpr),
    native_typr_prefix_goals(Rest, VarMap1, PredName, true, OutExpr, Conditions, RestLines, TailCode, VarMapOut).
native_typr_prefix_goals([Goal|Rest], VarMap0, PredName, false, LastExpr, [GuardCondition|RestConditions], RestLines, TailCode, VarMapOut) :-
    native_typr_guard_goal(Goal, VarMap0, GuardCondition),
    native_typr_prefix_goals(Rest, VarMap0, PredName, false, LastExpr, RestConditions, RestLines, TailCode, VarMapOut).
native_typr_prefix_goals([Goal|Rest], VarMap0, PredName, true, LastExpr, [], [], TailCode, VarMapOut) :-
    native_typr_guarded_tail_sequence([Goal|Rest], VarMap0, PredName, LastExpr, TailCode, VarMapOut).

native_typr_guarded_tail_sequence(Goals, VarMap0, PredName, LastExpr, Code, VarMapOut) :-
    native_typr_guarded_tail_sequence(Goals, VarMap0, PredName, LastExpr, [], Lines, FinalExpr, VarMapOut),
    append(Lines, [FinalExpr], BodyLines),
    atomic_list_concat(BodyLines, '\n', Code).

native_typr_guarded_tail_sequence([], VarMap, PredName, LastExpr, PendingGuards, [], FinalExpr, VarMap) :-
    guard_tail_final_expression(PendingGuards, PredName, LastExpr, FinalExpr).
native_typr_guarded_tail_sequence([Goal|Rest], VarMap0, PredName, _LastExpr0, [], GoalLines, FinalExpr, VarMapOut) :-
    native_typr_multi_result_output_goal(Goal, VarMap0, PredName, VarMap1, OutputLines, OutExpr),
    append(OutputLines, RestLines, GoalLines),
    native_typr_guarded_tail_sequence(Rest, VarMap1, PredName, OutExpr, [], RestLines, FinalExpr, VarMapOut).
native_typr_guarded_tail_sequence([Goal|Rest], VarMap0, PredName, _LastExpr0, PendingGuards, [Line|RestLines], FinalExpr, VarMapOut) :-
    native_typr_output_expr(Goal, VarMap0, PredName, VarMap1, OutVar, OutputExpr, IntroKind),
    guard_condition_expression(PendingGuards, GuardExpr),
    conditional_output_line(GuardExpr, PredName, IntroKind, OutVar, OutputExpr, Line),
    native_typr_guarded_tail_sequence(Rest, VarMap1, PredName, OutVar, [], RestLines, FinalExpr, VarMapOut).
native_typr_guarded_tail_sequence([Goal|Rest], VarMap0, PredName, LastExpr, PendingGuards0, Lines, FinalExpr, VarMapOut) :-
    native_typr_guard_goal(Goal, VarMap0, GuardCondition),
    append(PendingGuards0, [GuardCondition], PendingGuards),
    native_typr_guarded_tail_sequence(Rest, VarMap0, PredName, LastExpr, PendingGuards, Lines, FinalExpr, VarMapOut).

native_typr_output_goal(_Module:Goal, VarMap0, PredName, VarMap, Line, FinalExpr) :-
    !,
    native_typr_output_goal(Goal, VarMap0, PredName, VarMap, Line, FinalExpr).
native_typr_output_goal(Goal, VarMap0, PredName, VarMap, Line, FinalExpr) :-
    native_typr_output_expr(Goal, VarMap0, PredName, VarMap, FinalExpr, OutputExpr, IntroKind),
    typr_assignment_line(IntroKind, FinalExpr, OutputExpr, Line).

native_typr_output_expr(_Module:Goal, VarMap0, PredName, VarMap, FinalExpr, OutputExpr, IntroKind) :-
    !,
    native_typr_output_expr(Goal, VarMap0, PredName, VarMap, FinalExpr, OutputExpr, IntroKind).
native_typr_output_expr(Goal, VarMap0, PredName, VarMap, FinalExpr, OutputExpr, IntroKind) :-
    typr_if_then_else_goal(Goal, IfGoal, ThenGoal, ElseGoal),
    native_typr_if_then_else_output_expr(
        IfGoal,
        ThenGoal,
        ElseGoal,
        VarMap0,
        PredName,
        VarMap,
        FinalExpr,
        OutputExpr,
        IntroKind
    ),
    !.
native_typr_output_expr(Goal, VarMap0, PredName, VarMap, FinalExpr, OutputExpr, IntroKind) :-
    typr_if_then_goal(Goal, IfGoal, ThenGoal),
    native_typr_if_then_output_expr(
        IfGoal,
        ThenGoal,
        VarMap0,
        PredName,
        VarMap,
        FinalExpr,
        OutputExpr,
        IntroKind
    ),
    !.
native_typr_output_expr(Goal, VarMap0, PredName, VarMap, FinalExpr, OutputExpr, IntroKind) :-
    typr_disjunction_alternatives(Goal, Alternatives),
    native_typr_disjunction_output_expr(Alternatives, VarMap0, PredName, VarMap, FinalExpr, OutputExpr, IntroKind),
    !.
native_typr_output_expr(filter(DF, Expr, Out), VarMap0, _PredName, VarMap, FinalExpr, OutputExpr, IntroKind) :-
    !,
    ensure_typr_var(VarMap0, Out, FinalExpr, VarMap, IntroKind),
    typr_resolve_value(VarMap0, DF, RDF),
    typr_translate_r_expr(Expr, VarMap0, RExpr),
    format(string(OutputExpr), '@{ subset(~w, ~w) }@', [RDF, RExpr]).
native_typr_output_expr(sort_by(DF, Col, Out), VarMap0, _PredName, VarMap, FinalExpr, OutputExpr, IntroKind) :-
    !,
    ensure_typr_var(VarMap0, Out, FinalExpr, VarMap, IntroKind),
    typr_resolve_value(VarMap0, DF, RDF),
    typr_resolve_value(VarMap0, Col, RCol),
    format(string(OutputExpr), '@{ ~w[order(~w[[~w]]), ] }@', [RDF, RDF, RCol]).
native_typr_output_expr(group_by(DF, Col, Out), VarMap0, _PredName, VarMap, FinalExpr, OutputExpr, IntroKind) :-
    !,
    ensure_typr_var(VarMap0, Out, FinalExpr, VarMap, IntroKind),
    typr_resolve_value(VarMap0, DF, RDF),
    typr_resolve_value(VarMap0, Col, RCol),
    format(string(OutputExpr), '@{ aggregate(. ~~ ~w, data=~w, FUN=list) }@', [RCol, RDF]).
native_typr_output_expr(Goal, VarMap0, _PredName, VarMap, FinalExpr, OutputExpr, IntroKind) :-
    functor(Goal, Pred, Arity),
    binding(r, Pred/Arity, TargetName, Inputs, Outputs, _Options),
    Outputs = [_],
    simple_r_binding_target(TargetName),
    Goal =.. [_|Args],
    length(Inputs, InCount),
    length(InArgs, InCount),
    append(InArgs, [OutArg], Args),
    maplist(typr_resolve_value(VarMap0), InArgs, ResolvedInArgs),
    atomic_list_concat(ResolvedInArgs, ', ', RArgsStr),
    ensure_typr_var(VarMap0, OutArg, FinalExpr, VarMap, IntroKind),
    format(string(OutputExpr), '@{ ~w(~w) }@', [TargetName, RArgsStr]).

typr_disjunction_alternatives((Left ; Right), Alternatives) :-
    !,
    typr_disjunction_alternatives(Left, LeftAlternatives),
    typr_disjunction_alternatives(Right, RightAlternatives),
    append(LeftAlternatives, RightAlternatives, Alternatives).
typr_disjunction_alternatives(Goal, [Goal]).

typr_if_then_else_goal((IfGoal -> ThenGoal ; ElseGoal), IfGoal, ThenGoal, ElseGoal) :-
    !.
typr_if_then_else_goal(;(->(IfGoal, ThenGoal), ElseGoal), IfGoal, ThenGoal, ElseGoal) :-
    !.
typr_if_then_goal((IfGoal -> ThenGoal), IfGoal, ThenGoal) :-
    !.
typr_if_then_goal(->(IfGoal, ThenGoal), IfGoal, ThenGoal) :-
    !.

native_typr_if_then_else_output_expr(
    IfGoal,
    ThenGoal,
    ElseGoal,
    VarMap0,
    PredName,
    VarMap,
    FinalExpr,
    OutputExpr,
    IntroKind
) :-
    native_typr_if_condition(IfGoal, VarMap0, IfCondition),
    typr_if_then_else_shared_output_var(ThenGoal, ElseGoal, VarMap0, SharedVar),
    ensure_typr_var(VarMap0, SharedVar, FinalExpr, VarMap, IntroKind),
    native_typr_local_goal_sequence(ThenGoal, VarMap0, PredName, ThenCode, _ThenVarMap),
    native_typr_local_goal_sequence(ElseGoal, VarMap0, PredName, ElseCode, _ElseVarMap),
    branches_to_typr_output_if_chain(
        [branch(IfCondition, ThenCode), branch('TRUE', ElseCode)],
        PredName,
        OutputExpr
    ).

native_typr_if_then_output_expr(
    IfGoal,
    ThenGoal,
    VarMap0,
    PredName,
    VarMap,
    FinalExpr,
    OutputExpr,
    IntroKind
) :-
    native_typr_if_condition(IfGoal, VarMap0, IfCondition),
    typr_if_then_output_var(ThenGoal, VarMap0, OutputVar),
    ensure_typr_var(VarMap0, OutputVar, FinalExpr, VarMap, IntroKind),
    native_typr_local_goal_sequence(ThenGoal, VarMap0, PredName, ThenCode, _ThenVarMap),
    branches_to_typr_output_if_chain(
        [branch(IfCondition, ThenCode)],
        PredName,
        OutputExpr
    ).

native_typr_disjunction_output_expr(Alternatives, VarMap0, PredName, VarMap, FinalExpr, OutputExpr, IntroKind) :-
    Alternatives = [_|[_|_]],
    typr_disjunction_shared_output_var(Alternatives, VarMap0, SharedVar),
    ensure_typr_var(VarMap0, SharedVar, FinalExpr, VarMap, IntroKind),
    maplist(native_typr_alternative_branch(VarMap0, PredName, SharedVar), Alternatives, Branches),
    branches_to_typr_output_if_chain(Branches, PredName, OutputExpr).

native_typr_multi_result_output_goal(_Module:Goal, VarMap0, PredName, VarMapOut, OutputLines, FinalExpr) :-
    !,
    native_typr_multi_result_output_goal(Goal, VarMap0, PredName, VarMapOut, OutputLines, FinalExpr).
native_typr_multi_result_output_goal(Goal, VarMap0, PredName, VarMapOut, OutputLines, FinalExpr) :-
    typr_if_then_else_goal(Goal, IfGoal, ThenGoal, ElseGoal),
    native_typr_if_then_else_multi_result_output_goal(
        IfGoal,
        ThenGoal,
        ElseGoal,
        VarMap0,
        PredName,
        VarMapOut,
        OutputLines,
        FinalExpr
    ),
    !.
native_typr_multi_result_output_goal(Goal, VarMap0, PredName, VarMapOut, OutputLines, FinalExpr) :-
    typr_if_then_goal(Goal, IfGoal, ThenGoal),
    native_typr_if_then_multi_result_output_goal(
        IfGoal,
        ThenGoal,
        VarMap0,
        PredName,
        VarMapOut,
        OutputLines,
        FinalExpr
    ),
    !.
native_typr_multi_result_output_goal(Goal, VarMap0, PredName, VarMapOut, OutputLines, FinalExpr) :-
    typr_disjunction_alternatives(Goal, Alternatives),
    Alternatives = [_|[_|_]],
    typr_disjunction_shared_output_vars(Alternatives, VarMap0, SharedVars),
    length(SharedVars, SharedCount),
    SharedCount > 1,
    reserve_typr_internal_var(VarMap0, ContainerToken, ContainerName, VarMap1, new),
    ensure_typr_vars(SharedVars, VarMap1, SharedNamePairs, VarMap2),
    remove_var_mapping(ContainerToken, VarMap2, VarMapOut),
    maplist(native_typr_multi_result_branch(VarMap0, PredName, SharedVars), Alternatives, Branches),
    branches_to_typr_output_if_chain(Branches, PredName, ContainerExpr),
    typr_assignment_line(new, ContainerName, ContainerExpr, ContainerLine),
    build_typr_extraction_lines(ContainerName, SharedNamePairs, ExtractionLines),
    append([ContainerLine], ExtractionLines, OutputLines),
    last(SharedNamePairs, _-FinalExpr).

native_typr_if_then_multi_result_output_goal(
    IfGoal,
    ThenGoal,
    VarMap0,
    PredName,
    VarMapOut,
    OutputLines,
    FinalExpr
) :-
    native_typr_if_condition(IfGoal, VarMap0, IfCondition),
    typr_if_then_output_vars(ThenGoal, VarMap0, OutputVars),
    length(OutputVars, OutputCount),
    OutputCount > 1,
    reserve_typr_internal_var(VarMap0, ContainerToken, ContainerName, VarMap1, new),
    ensure_typr_vars(OutputVars, VarMap1, SharedNamePairs, VarMap2),
    remove_var_mapping(ContainerToken, VarMap2, VarMapOut),
    native_typr_local_goal_sequence(ThenGoal, VarMap0, PredName, ThenCode0, ThenVarMap),
    shared_output_list_expr(OutputVars, ThenVarMap, ThenListExpr),
    replace_final_expression(ThenCode0, ThenListExpr, ThenCode),
    branches_to_typr_output_if_chain(
        [branch(IfCondition, ThenCode)],
        PredName,
        ContainerExpr
    ),
    typr_assignment_line(new, ContainerName, ContainerExpr, ContainerLine),
    build_typr_extraction_lines(ContainerName, SharedNamePairs, ExtractionLines),
    append([ContainerLine], ExtractionLines, OutputLines),
    last(SharedNamePairs, _-FinalExpr).

native_typr_if_then_guard_condition(IfGoal, ThenGoal, VarMap, GuardCondition) :-
    native_typr_guard_sequence(IfGoal, VarMap, IfConditions),
    native_typr_guard_sequence(ThenGoal, VarMap, ThenConditions),
    append(IfConditions, ThenConditions, Conditions),
    Conditions \= [],
    combine_typr_conditions(Conditions, GuardCondition).

native_typr_if_then_else_guard_condition(IfGoal, ThenGoal, ElseGoal, VarMap, GuardCondition) :-
    native_typr_guard_sequence(IfGoal, VarMap, IfConditions),
    IfConditions \= [],
    combine_typr_conditions(IfConditions, IfCondition),
    native_typr_optional_guard_condition(ThenGoal, VarMap, ThenCondition),
    native_typr_optional_guard_condition(ElseGoal, VarMap, ElseCondition),
    typr_condition_expr_text(IfCondition, IfExpr),
    typr_condition_expr_text(ThenCondition, ThenExpr),
    typr_condition_expr_text(ElseCondition, ElseExpr),
    format(
        string(GuardCondition),
        '@{ ifelse(~w, ~w, ~w) }@',
        [IfExpr, ThenExpr, ElseExpr]
    ).

native_typr_optional_guard_condition(Body, VarMap, GuardCondition) :-
    normalize_typr_goals(Body, Goals),
    (   Goals = []
    ->  GuardCondition = 'TRUE'
    ;   maplist(native_typr_guard_goal_with_varmap(VarMap), Goals, Conditions),
        combine_typr_conditions(Conditions, GuardCondition)
    ).

native_typr_guard_sequence(Body, VarMap, Conditions) :-
    normalize_typr_goals(Body, Goals),
    Goals \= [],
    maplist(native_typr_guard_goal_with_varmap(VarMap), Goals, Conditions).

native_typr_guard_goal_with_varmap(VarMap, Goal, GuardCondition) :-
    native_typr_guard_goal(Goal, VarMap, GuardCondition).

native_typr_if_then_else_multi_result_output_goal(
    IfGoal,
    ThenGoal,
    ElseGoal,
    VarMap0,
    PredName,
    VarMapOut,
    OutputLines,
    FinalExpr
) :-
    native_typr_if_condition(IfGoal, VarMap0, IfCondition),
    typr_if_then_else_shared_output_vars(ThenGoal, ElseGoal, VarMap0, SharedVars),
    length(SharedVars, SharedCount),
    SharedCount > 1,
    reserve_typr_internal_var(VarMap0, ContainerToken, ContainerName, VarMap1, new),
    ensure_typr_vars(SharedVars, VarMap1, SharedNamePairs, VarMap2),
    remove_var_mapping(ContainerToken, VarMap2, VarMapOut),
    native_typr_local_goal_sequence(ThenGoal, VarMap0, PredName, ThenCode0, ThenVarMap),
    native_typr_local_goal_sequence(ElseGoal, VarMap0, PredName, ElseCode0, ElseVarMap),
    shared_output_list_expr(SharedVars, ThenVarMap, ThenListExpr),
    shared_output_list_expr(SharedVars, ElseVarMap, ElseListExpr),
    replace_final_expression(ThenCode0, ThenListExpr, ThenCode),
    replace_final_expression(ElseCode0, ElseListExpr, ElseCode),
    branches_to_typr_output_if_chain(
        [branch(IfCondition, ThenCode), branch('TRUE', ElseCode)],
        PredName,
        ContainerExpr
    ),
    typr_assignment_line(new, ContainerName, ContainerExpr, ContainerLine),
    build_typr_extraction_lines(ContainerName, SharedNamePairs, ExtractionLines),
    append([ContainerLine], ExtractionLines, OutputLines),
    last(SharedNamePairs, _-FinalExpr).

typr_disjunction_shared_output_var([Alternative|Rest], VarMap, SharedVar) :-
    typr_alternative_output_var(Alternative, SharedVar),
    var(SharedVar),
    typr_disjunction_output_var_allowed(VarMap, SharedVar),
    maplist(typr_alternative_output_var_matches(SharedVar), Rest).

typr_if_then_else_shared_output_var(ThenGoal, ElseGoal, VarMap, SharedVar) :-
    typr_alternative_output_var(ThenGoal, SharedVar),
    var(SharedVar),
    typr_disjunction_output_var_allowed(VarMap, SharedVar),
    typr_alternative_output_var(ElseGoal, ElseVar),
    ElseVar == SharedVar.

typr_if_then_output_var(ThenGoal, VarMap, OutputVar) :-
    typr_alternative_output_var(ThenGoal, OutputVar),
    var(OutputVar),
    typr_disjunction_output_var_allowed(VarMap, OutputVar).

typr_disjunction_shared_output_vars([Alternative|Rest], VarMap, SharedVars) :-
    typr_alternative_output_vars(Alternative, FirstOutputVars0),
    exclude_varmap_vars(VarMap, FirstOutputVars0, FirstOutputVars),
    foldl(intersect_output_vars, Rest, FirstOutputVars, SharedVars),
    SharedVars \= [].

typr_if_then_else_shared_output_vars(ThenGoal, ElseGoal, VarMap, SharedVars) :-
    typr_alternative_output_vars(ThenGoal, ThenOutputVars0),
    exclude_varmap_vars(VarMap, ThenOutputVars0, ThenOutputVars),
    intersect_output_vars(ElseGoal, ThenOutputVars, SharedVars),
    SharedVars \= [].

typr_if_then_output_vars(ThenGoal, VarMap, OutputVars) :-
    typr_alternative_output_vars(ThenGoal, ThenOutputVars0),
    exclude_varmap_vars(VarMap, ThenOutputVars0, OutputVars),
    OutputVars \= [].

typr_disjunction_output_var_allowed(VarMap, SharedVar) :-
    \+ varmap_contains_var(VarMap, SharedVar),
    !.
typr_disjunction_output_var_allowed(VarMap, SharedVar) :-
    lookup_typr_var(SharedVar, VarMap, ArgName),
    sub_string(ArgName, 0, 3, _, "arg").

typr_alternative_output_var_matches(ExpectedVar, Alternative) :-
    typr_alternative_output_var(Alternative, ActualVar),
    ActualVar == ExpectedVar.

typr_alternative_output_vars(Alternative, OutputVars) :-
    normalize_typr_goals(Alternative, Goals),
    collect_typr_goal_output_vars(Goals, OutputVars0),
    unique_vars_by_identity(OutputVars0, OutputVars).

collect_typr_goal_output_vars([], []).
collect_typr_goal_output_vars([Goal|Rest], OutputVars) :-
    (   typr_goal_output_vars(Goal, GoalOutputVars)
    ->  append(GoalOutputVars, RestOutputs, OutputVars)
    ;   OutputVars = RestOutputs
    ),
    collect_typr_goal_output_vars(Rest, RestOutputs).

typr_alternative_output_var(Alternative, OutputVar) :-
    normalize_typr_goals(Alternative, Goals),
    reverse(Goals, ReversedGoals),
    member(Goal, ReversedGoals),
    typr_goal_output_var(Goal, OutputVar),
    !.

typr_goal_output_vars(_Module:Goal, OutputVars) :-
    !,
    typr_goal_output_vars(Goal, OutputVars).
typr_goal_output_vars(Goal, OutputVars) :-
    typr_if_then_else_goal(Goal, _IfGoal, ThenGoal, ElseGoal),
    typr_if_then_else_goal_output_vars(ThenGoal, ElseGoal, OutputVars),
    !.
typr_goal_output_vars(Goal, OutputVars) :-
    typr_if_then_goal(Goal, _IfGoal, ThenGoal),
    typr_if_then_goal_output_vars(ThenGoal, OutputVars),
    !.
typr_goal_output_vars(Goal, OutputVars) :-
    typr_disjunction_alternatives(Goal, Alternatives),
    Alternatives = [_|[_|_]],
    typr_disjunction_goal_output_vars(Alternatives, OutputVars),
    !.
typr_goal_output_vars(Goal, [OutputVar]) :-
    typr_goal_output_var_simple(Goal, OutputVar).

typr_goal_output_var(_Module:Goal, OutputVar) :-
    !,
    typr_goal_output_var(Goal, OutputVar).
typr_goal_output_var(Goal, OutputVar) :-
    typr_goal_output_vars(Goal, [OutputVar]),
    !.
typr_goal_output_var(Goal, OutputVar) :-
    typr_goal_output_var_simple(Goal, OutputVar).

typr_disjunction_goal_output_vars([Alternative|Rest], OutputVars) :-
    typr_alternative_output_vars(Alternative, FirstOutputVars),
    foldl(intersect_output_vars, Rest, FirstOutputVars, OutputVars),
    OutputVars \= [].

typr_if_then_else_goal_output_vars(ThenGoal, ElseGoal, OutputVars) :-
    typr_alternative_output_vars(ThenGoal, ThenOutputVars),
    intersect_output_vars(ElseGoal, ThenOutputVars, OutputVars),
    OutputVars \= [].

typr_if_then_goal_output_vars(ThenGoal, OutputVars) :-
    typr_alternative_output_vars(ThenGoal, OutputVars),
    OutputVars \= [].

typr_goal_output_var_simple(_Module:Goal, OutputVar) :-
    !,
    typr_goal_output_var_simple(Goal, OutputVar).
typr_goal_output_var_simple(filter(_, _, OutputVar), OutputVar).
typr_goal_output_var_simple(sort_by(_, _, OutputVar), OutputVar).
typr_goal_output_var_simple(group_by(_, _, OutputVar), OutputVar).
typr_goal_output_var_simple(Goal, OutputVar) :-
    functor(Goal, Pred, Arity),
    binding(r, Pred/Arity, TargetName, Inputs, Outputs, _Options),
    Outputs = [_],
    simple_r_binding_target(TargetName),
    Goal =.. [_|Args],
    length(Inputs, InCount),
    length(Outputs, OutCount),
    length(InArgs, InCount),
    length(OutArgs, OutCount),
    append(InArgs, OutArgs, Args),
    OutArgs = [OutputVar].

native_typr_alternative_branch(VarMap0, PredName, SharedVar, Alternative, branch(Condition, BranchCode)) :-
    typr_alternative_output_var_matches(SharedVar, Alternative),
    native_typr_goal_sequence(Alternative, VarMap0, PredName, Conditions, BranchCode),
    combine_typr_conditions(Conditions, Condition).

native_typr_multi_result_branch(VarMap0, PredName, SharedVars, Alternative, branch(Condition, BranchCode)) :-
    native_typr_goal_sequence(Alternative, VarMap0, PredName, Conditions, BranchCode0, BranchVarMap),
    shared_output_list_expr(SharedVars, BranchVarMap, SharedListExpr),
    replace_final_expression(BranchCode0, SharedListExpr, BranchCode),
    combine_typr_conditions(Conditions, Condition).

native_typr_local_goal_sequence(Body, VarMap0, PredName, Code, VarMapOut) :-
    native_typr_goal_sequence(Body, VarMap0, PredName, Conditions, RawCode, VarMapOut),
    (   Conditions = []
    ->  Code = RawCode
    ;   combine_typr_conditions(Conditions, GuardExpr),
        branch_safe_typr_code(RawCode, SafeRawCode),
        indent_text(SafeRawCode, "\t", IndentedCode),
        format(
            string(Code),
'if (~w) {
~w
} else {
	stop("No matching clause for ~w")
}', [GuardExpr, IndentedCode, PredName]
        )
    ).

native_typr_if_condition(IfGoal, VarMap, IfCondition) :-
    normalize_typr_goals(IfGoal, Goals),
    findall(Condition, (
        member(Goal, Goals),
        native_typr_guard_goal(Goal, VarMap, Condition)
    ), Conditions),
    combine_typr_conditions(Conditions, IfCondition).

varmap_contains_var([StoredVar-_|_], Var) :-
    Var == StoredVar,
    !.
varmap_contains_var([_|Rest], Var) :-
    varmap_contains_var(Rest, Var).

exclude_varmap_vars(_VarMap, [], []).
exclude_varmap_vars(VarMap, [Var|Rest], Filtered) :-
    (   varmap_contains_var(VarMap, Var)
    ->  exclude_varmap_vars(VarMap, Rest, Filtered)
    ;   Filtered = [Var|FilteredRest],
        exclude_varmap_vars(VarMap, Rest, FilteredRest)
    ).

intersect_output_vars(Alternative, Candidates0, Candidates) :-
    typr_alternative_output_vars(Alternative, OutputVars),
    include_vars_by_identity(Candidates0, OutputVars, Candidates).

include_vars_by_identity([], _Allowed, []).
include_vars_by_identity([Var|Rest], Allowed, Included) :-
    (   var_member_by_identity(Var, Allowed)
    ->  Included = [Var|IncludedRest]
    ;   Included = IncludedRest
    ),
    include_vars_by_identity(Rest, Allowed, IncludedRest).

unique_vars_by_identity([], []).
unique_vars_by_identity([Var|Rest], Unique) :-
    unique_vars_by_identity(Rest, RestUnique),
    (   var_member_by_identity(Var, RestUnique)
    ->  Unique = RestUnique
    ;   Unique = [Var|RestUnique]
    ).

var_member_by_identity(Var, [Candidate|_]) :-
    Var == Candidate,
    !.
var_member_by_identity(Var, [_|Rest]) :-
    var_member_by_identity(Var, Rest).

reserve_typr_internal_var(VarMap0, Token, Name, VarMap, IntroKind) :-
    ensure_typr_var(VarMap0, Token, Name, VarMap, IntroKind).

ensure_typr_vars([], VarMap, [], VarMap).
ensure_typr_vars([Var|Rest], VarMap0, [Var-Name|RestPairs], VarMapOut) :-
    ensure_typr_var(VarMap0, Var, Name, VarMap1, _),
    ensure_typr_vars(Rest, VarMap1, RestPairs, VarMapOut).

remove_var_mapping(_Var, [], []).
remove_var_mapping(Var, [StoredVar-Name|Rest], Filtered) :-
    (   Var == StoredVar
    ->  Filtered = FilteredRest
    ;   Filtered = [StoredVar-Name|FilteredRest]
    ),
    remove_var_mapping(Var, Rest, FilteredRest).

shared_output_list_expr(SharedVars, VarMap, ListExpr) :-
    findall(Name, (
        member(SharedVar, SharedVars),
        lookup_typr_var(SharedVar, VarMap, Name)
    ), SharedNames),
    atomic_list_concat(SharedNames, ', ', SharedNamesText),
    format(string(ListExpr), '@{ list(~w) }@', [SharedNamesText]).

replace_final_expression(Code, NewFinalExpr, RewrittenCode) :-
    split_string(Code, "\n", "", Lines),
    (   wrapped_guard_body_lines(Lines, HeaderLine, BodyLines, ElseLine, StopLine, ClosingLine)
    ->  atomic_list_concat(BodyLines, '\n', BodyCode),
        replace_final_expression(BodyCode, NewFinalExpr, RewrittenBodyCode),
        atomic_list_concat(
            [HeaderLine, RewrittenBodyCode, ElseLine, StopLine, ClosingLine],
            '\n',
            RewrittenCode
        )
    ;   append(PrefixLines, [OldFinalExpr], Lines),
        OldFinalExpr \= "}",
        append(PrefixLines, [NewFinalExpr], RewrittenLines),
        atomic_list_concat(RewrittenLines, '\n', RewrittenCode)
    ).

wrapped_guard_body_lines(
    [HeaderLine|RestLines],
    HeaderLine,
    BodyLines,
    ElseLine,
    StopLine,
    ClosingLine
) :-
    sub_string(HeaderLine, 0, 4, _, "if ("),
    append(BodyAndElseLines, [StopLine, ClosingLine], RestLines),
    append(BodyLines, [ElseLine], BodyAndElseLines),
    ElseLine = "} else {",
    sub_string(StopLine, _, _, _, "stop(\"No matching clause for "),
    ClosingLine = "}".

build_typr_extraction_lines(ContainerName, SharedNamePairs, Lines) :-
    build_typr_extraction_lines(ContainerName, SharedNamePairs, 1, Lines).

build_typr_extraction_lines(_ContainerName, [], _Index, []).
build_typr_extraction_lines(ContainerName, [_Var-Name|Rest], Index, [Line|RestLines]) :-
    format(string(Line), 'let ~w <- @{ .subset2(~w, ~w) }@;', [Name, ContainerName, Index]),
    NextIndex is Index + 1,
    build_typr_extraction_lines(ContainerName, Rest, NextIndex, RestLines).

native_typr_guard_goal(_Module:Goal, VarMap, GuardCondition) :-
    !,
    native_typr_guard_goal(Goal, VarMap, GuardCondition).
native_typr_guard_goal(Goal, VarMap, GuardCondition) :-
    typr_if_then_else_goal(Goal, IfGoal, ThenGoal, ElseGoal),
    native_typr_if_then_else_guard_condition(IfGoal, ThenGoal, ElseGoal, VarMap, GuardCondition),
    !.
native_typr_guard_goal(Goal, VarMap, GuardCondition) :-
    typr_if_then_goal(Goal, IfGoal, ThenGoal),
    native_typr_if_then_guard_condition(IfGoal, ThenGoal, VarMap, GuardCondition),
    !.
native_typr_guard_goal(Goal, VarMap, GuardCondition) :-
    typr_native_guard_expr(Goal, VarMap, GuardCondition),
    !.
native_typr_guard_goal(Goal, VarMap, GuardCondition) :-
    functor(Goal, Pred, Arity),
    binding(r, Pred/Arity, TargetName, Inputs, [], Options),
    member(pattern(command), Options),
    simple_r_binding_target(TargetName),
    Goal =.. [_|Args],
    length(Inputs, InCount),
    length(InArgs, InCount),
    append(InArgs, [], Args),
    maplist(typr_resolve_value(VarMap), InArgs, ResolvedInArgs),
    typr_guard_expression(TargetName, ResolvedInArgs, GuardCondition).

typr_native_guard_expr(Expr, VarMap, GuardCondition) :-
    compound(Expr),
    Expr =.. [Op, _Left, _Right],
    r_expr_op_map(Op, _),
    typr_translate_r_expr(Expr, VarMap, ResolvedExpr0),
    typr_top_level_guard_expr(ResolvedExpr0, ResolvedExpr),
    format(string(GuardCondition), '@{ ~w }@', [ResolvedExpr]).

simple_r_binding_target(TargetName) :-
    atom(TargetName),
    \+ sub_atom(TargetName, _, _, _, '~').
simple_r_binding_target(TargetName) :-
    string(TargetName),
    \+ sub_string(TargetName, _, _, _, "~").

typr_resolve_value(VarMap, Var, Value) :-
    var(Var),
    lookup_typr_var(Var, VarMap, Value),
    !.
typr_resolve_value(_VarMap, Value, Literal) :-
    r_literal(Value, Literal).

typr_guard_expression(TargetName, [], GuardCondition) :-
    format(string(GuardCondition), '@{ ~w }@', [TargetName]).
typr_guard_expression(TargetName, ResolvedInArgs, GuardCondition) :-
    atomic_list_concat(ResolvedInArgs, ', ', RArgsStr),
    format(string(GuardCondition), '@{ ~w(~w) }@', [TargetName, RArgsStr]).

guard_condition_expression([], none).
guard_condition_expression(Conditions, GuardExpr) :-
    Conditions \= [],
    combine_typr_conditions(Conditions, GuardExpr).

combine_typr_conditions([], 'TRUE').
combine_typr_conditions([Condition], Condition) :-
    !.
combine_typr_conditions(Conditions, Combined) :-
    maplist(typr_condition_expr_text, Conditions, RawExprs),
    !,
    atomic_list_concat(RawExprs, ' && ', InnerExpr),
    format(string(Combined), '@{ ~w }@', [InnerExpr]).
combine_typr_conditions(Conditions, Combined) :-
    atomic_list_concat(Conditions, ' && ', Combined).

typr_condition_expr_text(Condition, Expr) :-
    raw_guard_expr_text(Condition, Expr),
    !.
typr_condition_expr_text(Condition, Condition).

raw_guard_expr_text(Condition, Expr) :-
    string(Condition),
    sub_string(Condition, 0, 3, _, "@{ "),
    sub_string(Condition, _, 3, 0, " }@"),
    sub_string(Condition, 3, _, 3, Expr).

typr_binding_prefix(existing, '').
typr_binding_prefix(new, 'let ').

typr_assignment_line(IntroKind, OutVar, OutputExpr, Line) :-
    typr_binding_prefix(IntroKind, Prefix),
    format(string(Line), '~w~w <- ~w;', [Prefix, OutVar, OutputExpr]).

conditional_output_line(none, _PredName, IntroKind, OutVar, OutputExpr, Line) :-
    typr_assignment_line(IntroKind, OutVar, OutputExpr, Line).
conditional_output_line(GuardExpr, PredName, IntroKind, OutVar, OutputExpr, Line) :-
    typr_binding_prefix(IntroKind, Prefix),
    format(string(Line),
'~w~w <- if (~w) {
	~w
} else {
	stop("No matching clause for ~w")
};', [Prefix, OutVar, GuardExpr, OutputExpr, PredName]).

guard_tail_final_expression([], _PredName, none, 'true').
guard_tail_final_expression([], _PredName, LastExpr, LastExpr) :-
    LastExpr \== none.
guard_tail_final_expression(PendingGuards, PredName, LastExpr, FinalExpr) :-
    PendingGuards \= [],
    guard_condition_expression(PendingGuards, GuardExpr),
    (   LastExpr == none
    ->  BaseExpr = 'true'
    ;   BaseExpr = LastExpr
    ),
    format(string(FinalExpr),
'if (~w) {
	~w
} else {
	stop("No matching clause for ~w")
}', [GuardExpr, BaseExpr, PredName]).

typr_translate_r_expr(Var, VarMap, Resolved) :-
    var(Var),
    !,
    lookup_typr_var(Var, VarMap, Resolved).
typr_translate_r_expr(Atom, _VarMap, Resolved) :-
    atom(Atom),
    !,
    format(string(Resolved), '~w', [Atom]).
typr_translate_r_expr(Text, _VarMap, Resolved) :-
    string(Text),
    !,
    format(string(Resolved), '"~s"', [Text]).
typr_translate_r_expr(Number, _VarMap, Resolved) :-
    number(Number),
    !,
    format(string(Resolved), '~w', [Number]).
typr_translate_r_expr(-Expr, VarMap, Resolved) :-
    !,
    typr_translate_r_expr(Expr, VarMap, InnerResolved),
    format(string(Resolved), '-~w', [InnerResolved]).
typr_translate_r_expr(Expr, VarMap, Resolved) :-
    compound(Expr),
    Expr =.. [Op, Left, Right],
    typr_translate_r_expr(Left, VarMap, LeftResolved),
    typr_translate_r_expr(Right, VarMap, RightResolved),
    r_expr_op_map(Op, ROp),
    format(string(Resolved), '(~w ~w ~w)', [LeftResolved, ROp, RightResolved]).
typr_translate_r_expr(Expr, VarMap, Resolved) :-
    compound(Expr),
    Expr =.. [Fn|Args],
    Args \= [],
    maplist(typr_translate_r_expr_arg(VarMap), Args, ResolvedArgs),
    atomic_list_concat(ResolvedArgs, ', ', ArgsText),
    format(string(Resolved), '~w(~w)', [Fn, ArgsText]).

typr_translate_r_expr_arg(VarMap, Arg, Resolved) :-
    typr_translate_r_expr(Arg, VarMap, Resolved).

typr_top_level_guard_expr(ResolvedExpr0, ResolvedExpr) :-
    (   sub_string(ResolvedExpr0, 0, 1, _, "("),
        sub_string(ResolvedExpr0, _, 1, 0, ")")
    ->  sub_string(ResolvedExpr0, 1, _, 1, ResolvedExpr)
    ;   ResolvedExpr = ResolvedExpr0
    ).

r_expr_op_map(>, '>').
r_expr_op_map(<, '<').
r_expr_op_map(>=, '>=').
r_expr_op_map(=<, '<=').
r_expr_op_map(=:=, '==').
r_expr_op_map(==, '==').
r_expr_op_map(\=, '!=').
r_expr_op_map(+, '+').
r_expr_op_map(-, '-').
r_expr_op_map(*, '*').
r_expr_op_map(/, '/').
r_expr_op_map(and, '&').
r_expr_op_map(or, '|').

ensure_typr_var(VarMap, Var, Name, VarMap, existing) :-
    lookup_typr_var(Var, VarMap, Name),
    !.
ensure_typr_var(VarMap, Var, Name, [Var-Name|VarMap], new) :-
    next_typr_var_index(VarMap, NextIndex),
    format(string(Name), 'v~w', [NextIndex]).
ensure_typr_var(VarMap, Var, Name, VarMapOut) :-
    ensure_typr_var(VarMap, Var, Name, VarMapOut, _).

next_typr_var_index([], 1).
next_typr_var_index(VarMap, NextIndex) :-
    findall(Index, (
        member(_-Name, VarMap),
        typr_name_index(Name, Index)
    ), Indices),
    max_list(Indices, MaxIndex),
    NextIndex is MaxIndex + 1.

typr_name_index(Name, Index) :-
    string(Name),
    (   sub_string(Name, 0, 3, _, "arg")
    ->  sub_string(Name, 3, _, 0, Digits)
    ;   sub_string(Name, 0, 1, _, "v"),
        sub_string(Name, 1, _, 0, Digits)
    ),
    number_string(Index, Digits).

lookup_typr_var(Var, [StoredVar-Name|_], Name) :-
    Var == StoredVar,
    !.
lookup_typr_var(Var, [_|Rest], Name) :-
    lookup_typr_var(Var, Rest, Name).

build_head_varmap([], _, []).
build_head_varmap([Arg|Args], Index, [Arg-ArgName|Rest]) :-
    var(Arg),
    !,
    format(string(ArgName), 'arg~w', [Index]),
    NextIndex is Index + 1,
    build_head_varmap(Args, NextIndex, Rest).
build_head_varmap([_|Args], Index, Rest) :-
    NextIndex is Index + 1,
    build_head_varmap(Args, NextIndex, Rest).

typr_head_condition([], _, []).
typr_head_condition([HeadArg|Rest], Index, Conditions) :-
    (   fact_arg_condition(Index, HeadArg, Condition),
        Condition \== true
    ->  Conditions = [Condition|RestConditions]
    ;   Conditions = RestConditions
    ),
    NextIndex is Index + 1,
    typr_head_condition(Rest, NextIndex, RestConditions).

branches_to_typr_if_chain([branch(Condition, Code)], IfChain) :-
    branch_safe_typr_code(Code, BranchCode),
    indent_text(BranchCode, "\t", IndentedCode),
    format(string(IfChain), 'if (~w) {\n~w\n}', [Condition, IndentedCode]).
branches_to_typr_if_chain([branch(Condition, Code)|Rest], IfChain) :-
    branches_to_typr_if_chain(Rest, RestChain),
    branch_safe_typr_code(Code, BranchCode),
    indent_text(BranchCode, "\t", IndentedCode),
    format(string(IfChain), 'if (~w) {\n~w\n} else ~w', [Condition, IndentedCode, RestChain]).

branches_to_typr_output_if_chain([branch(Condition, Code)], PredName, OutputExpr) :-
    branch_safe_typr_code(Code, BranchCode),
    indent_text(BranchCode, "\t", IndentedCode),
    format(string(OutputExpr),
'if (~w) {
~w
} else {
	stop("No matching clause for ~w")
}', [Condition, IndentedCode, PredName]).
branches_to_typr_output_if_chain([branch(Condition, Code)|Rest], PredName, OutputExpr) :-
    Rest \= [],
    branch_safe_typr_code(Code, BranchCode),
    indent_text(BranchCode, "\t", IndentedCode),
    branches_to_typr_output_if_chain(Rest, PredName, RestChain),
    format(string(OutputExpr), 'if (~w) {\n~w\n} else ~w', [Condition, IndentedCode, RestChain]).

pred_spec_name(_Module:Pred/_, PredName) :-
    !,
    atom_string(Pred, PredName).
pred_spec_name(Pred/_, PredName) :-
    atom_string(Pred, PredName).

indent_text(Text, Prefix, Indented) :-
    split_string(Text, "\n", "", Lines),
    findall(IndentedLine, (
        member(Line, Lines),
        string_concat(Prefix, Line, IndentedLine)
    ), IndentedLines),
    atomic_list_concat(IndentedLines, '\n', Indented).

branch_safe_typr_code(Code, SafeCode) :-
    split_string(Code, "\n", "", Lines),
    maplist(branch_safe_typr_line, Lines, SafeLines),
    atomic_list_concat(SafeLines, '\n', SafeCode).

branch_safe_typr_line(Line, SafeLine) :-
    leading_layout(Line, Layout, Content),
    (   sub_string(Content, 0, 4, _, "let ")
    ->  SafeLine = Line
    ;   sub_string(Content, _, _, _, " <- "),
        sub_string(Content, _, 1, 0, ";")
    ->  format(string(SafeLine), '~wlet ~w', [Layout, Content])
    ;   SafeLine = Line
    ).

leading_layout(Line, Layout, Content) :-
    string_codes(Line, Codes),
    take_layout_codes(Codes, LayoutCodes, ContentCodes),
    string_codes(Layout, LayoutCodes),
    string_codes(Content, ContentCodes).

take_layout_codes([Code|Rest], [Code|LayoutCodes], ContentCodes) :-
    code_type(Code, space),
    !,
    take_layout_codes(Rest, LayoutCodes, ContentCodes).
take_layout_codes(ContentCodes, [], ContentCodes).

finalize_type_diagnostics_report(Options) :-
    (   memberchk(type_diagnostics_report(Report), Options),
        var(Report)
    ->  Report = []
    ;   true
    ).

wrapped_r_body_expression(Module:Pred/Arity, Options, WrappedExpr) :-
    !,
    init_r_target,
    compile_predicate_to_r(Module:Pred/Arity, Options, RCode),
    r_function_body(RCode, Body),
    raw_arg_list(Arity, ArgList),
    format(string(WrappedExpr), '(function(~w) {\n~w\n})(~w)', [ArgList, Body, ArgList]).
wrapped_r_body_expression(Pred/Arity, Options, WrappedExpr) :-
    wrapped_r_body_expression(user:Pred/Arity, Options, WrappedExpr).

r_function_body(RCode, Body) :-
    split_string(RCode, "\n", "\n", Lines),
    Lines = [_Header|Rest],
    append(BodyLines, ["}"], Rest),
    atomic_list_concat(BodyLines, '\n', Body).

raw_arg_list(Arity, ArgList) :-
    findall(ArgName, (
        between(1, Arity, Index),
        format(string(ArgName), 'arg~w', [Index])
    ), ArgNames),
    atomic_list_concat(ArgNames, ', ', ArgList).

generated_typr_is_valid(Code, Result) :-
    tmp_file(typr_validation, RootDir),
    make_directory(RootDir),
    setup_call_cleanup(
        true,
        (
            run_typr_command(RootDir, ['new', 'validation_project'], exit(0)),
            directory_file_path(RootDir, 'validation_project', ProjectDir),
            directory_file_path(ProjectDir, 'TypR/main.ty', MainFile),
            setup_call_cleanup(
                open(MainFile, write, Stream),
                write(Stream, Code),
                close(Stream)
            ),
            run_typr_command(ProjectDir, ['check'], Status),
            Result = Status
        ),
        delete_directory_and_contents(RootDir)
    ).

run_typr_command(ProjectDir, Args, Status) :-
    process_create(
        path(typr),
        Args,
        [ cwd(ProjectDir),
          stdout(pipe(Stdout)),
          stderr(pipe(Stderr)),
          process(Pid)
        ]
    ),
    read_string(Stdout, _, _),
    read_string(Stderr, _, _),
    close(Stdout),
    close(Stderr),
    process_wait(Pid, Status).
