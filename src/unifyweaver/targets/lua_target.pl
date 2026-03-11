% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% lua_target.pl - Lua Target for UnifyWeaver
% Generates standalone Lua scripts for record processing.
% Supports sequential and fixpoint evaluation.

:- module(lua_target, [
    compile_predicate_to_lua/3,     % +Predicate, +Options, -LuaCode
    compile_facts_to_lua/3,         % +Predicate, +Options, -LuaCode
    init_lua_target/0,              % Initialize Lua target with bindings
    compile_lua_pipeline/3,         % +Predicates, +Options, -LuaCode
    test_lua_pipeline/0             % Test pipeline generation
]).

:- use_module(library(lists)).
:- use_module(library(gensym)).
:- use_module(common_generator).
:- use_module('../core/binding_registry').
:- use_module('../bindings/lua_bindings').
:- use_module('../core/advanced/tree_recursion').
:- use_module('../core/advanced/tail_recursion').
:- use_module('../core/advanced/multicall_linear_recursion').
:- use_module('../core/advanced/direct_multi_call_recursion').
:- use_module('../core/advanced/linear_recursion').
:- use_module('../core/advanced/mutual_recursion').

%% init_lua_target
%  Initialize the Lua target by loading bindings.
init_lua_target :-
    init_lua_bindings.

% ============================================================================
% FACT COMPILATION
% ============================================================================

%% compile_facts_to_lua(+PredIndicator, +Options, -LuaCode)
%  Compiles facts to a Lua function using CPS (callback) pattern.
compile_facts_to_lua(Pred/Arity, _Options, LuaCode) :-
    functor(Head, Pred, Arity),
    findall(Args, (user:clause(Head, true), Head =.. [_|Args]), AllFacts),
    findall(Entry, (
        member(Args, AllFacts),
        maplist(format_lua_fact_arg, Args, FormattedArgs),
        atomic_list_concat(FormattedArgs, ', ', ArgsStr),
        format(string(Entry), '        {~w}', [ArgsStr])
    ), Entries),
    atomic_list_concat(Entries, ',\n', FactLines),
    format(string(LuaCode),
'local function ~w(callback)
    local facts = {
~w
    }
    for _, fact in ipairs(facts) do
        callback(table.unpack(fact))
    end
end
', [Pred, FactLines]).

format_lua_fact_arg(Arg, Formatted) :-
    (   number(Arg) -> format(string(Formatted), '~w', [Arg])
    ;   format(string(Formatted), '"~w"', [Arg])
    ).

% ============================================================================
% PREDICATE COMPILATION (with multi-clause OR and join handling)
% ============================================================================

%% compile_predicate_to_lua(+PredIndicator, +Options, -LuaCode)
compile_predicate_to_lua(Pred/Arity, Options, LuaCode) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   Clauses = []
    ->  format(string(LuaCode), '-- No clauses found for ~w/~w', [Pred, Arity])
    ;   % Check if all clauses are facts
        forall(member(_-Body, Clauses), Body == true)
    ->  compile_facts_to_lua(Pred/Arity, Options, LuaCode)
    ;   % Check for recursive predicates
        (   member(_-Body, Clauses),
            body_calls_pred(Body, Pred)
        )
    ->  compile_recursive_rules(Pred, Arity, Clauses, LuaCode)
    ;   % Non-recursive rules (single or multiple)
        compile_nonrecursive_rules(Pred, Arity, Clauses, LuaCode)
    ).

body_calls_pred((A, B), Pred) :- !,
    (body_calls_pred(A, Pred) ; body_calls_pred(B, Pred)).
body_calls_pred(_:Goal, Pred) :- !, body_calls_pred(Goal, Pred).
body_calls_pred(Goal, Pred) :-
    functor(Goal, Pred, _).

%% compile_nonrecursive_rules(+Pred, +Arity, +Clauses, -Code)
%  Compiles non-recursive rules with CPS callbacks and join handling.
compile_nonrecursive_rules(Pred, _Arity, Clauses, LuaCode) :-
    findall(ClauseCode, (
        member(Head-Body, Clauses),
        Head =.. [_|HeadArgs],
        compile_clause_body(Body, HeadArgs, 0, ClauseCode)
    ), ClauseCodes),
    atomic_list_concat(ClauseCodes, '\n', AllClausesCode),
    format(string(LuaCode),
'local function ~w(callback)
~w
end
', [Pred, AllClausesCode]).

%% compile_clause_body(+Body, +HeadArgs, +ParamCounter, -Code)
%  Compiles a single clause body with join handling.
compile_clause_body(true, HeadArgs, _Counter, Code) :- !,
    maplist(format_head_arg, HeadArgs, ArgStrs),
    atomic_list_concat(ArgStrs, ', ', ArgsStr),
    format(string(Code), '    callback(~w)', [ArgsStr]).
compile_clause_body(Body, HeadArgs, Counter, Code) :-
    body_to_goals(Body, Goals),
    compile_goals_nested(Goals, HeadArgs, Counter, [], Code).

body_to_goals((A, B), Goals) :- !,
    body_to_goals(A, GA),
    body_to_goals(B, GB),
    append(GA, GB, Goals).
body_to_goals(_:Goal, Goals) :- !, body_to_goals(Goal, Goals).
body_to_goals(Goal, [Goal]).

%% compile_goals_nested(+Goals, +HeadArgs, +Counter, +Bindings, -Code)
%  Generates nested callback calls with join conditions.
compile_goals_nested([], HeadArgs, _Counter, Bindings, Code) :- !,
    maplist(resolve_head_arg(Bindings), HeadArgs, ResolvedArgs),
    atomic_list_concat(ResolvedArgs, ', ', ArgsStr),
    format(string(Code), '    callback(~w)', [ArgsStr]).
compile_goals_nested([Goal|Rest], HeadArgs, Counter, Bindings, Code) :-
    (   Goal = (_ is _) ->
        % Arithmetic: skip for now in CPS mode
        compile_goals_nested(Rest, HeadArgs, Counter, Bindings, Code)
    ;   Goal =.. [GoalPred|GoalArgs],
        length(GoalArgs, GoalArity),
        create_callback_params(GoalArity, Counter, Params, NextCounter),
        build_join_conditions(GoalArgs, Params, Bindings, JoinConds, NewBindings),
        atomic_list_concat(Params, ', ', ParamList),
        format_join_guards(JoinConds, GuardCode),
        compile_goals_nested(Rest, HeadArgs, NextCounter, NewBindings, InnerCode),
        (   GuardCode = "" ->
            format(string(Code),
'    ~w(function(~w)
~w
    end)', [GoalPred, ParamList, InnerCode])
        ;   format(string(Code),
'    ~w(function(~w)
~w
~w
    end)', [GoalPred, ParamList, GuardCode, InnerCode])
        )
    ).

%% create_callback_params(+N, +Counter, -Params, -NextCounter)
create_callback_params(0, C, [], C) :- !.
create_callback_params(N, C, [Param|Rest], Final) :-
    N > 0,
    format(atom(Param), 'p~d', [C]),
    C1 is C + 1,
    N1 is N - 1,
    create_callback_params(N1, C1, Rest, Final).

%% build_join_conditions(+GoalArgs, +Params, +Bindings, -JoinConds, -NewBindings)
build_join_conditions([], [], Bindings, [], Bindings).
build_join_conditions([Arg|Args], [Param|Params], Bindings, JoinConds, NewBindings) :-
    (   var(Arg), lookup_binding(Arg, Bindings, ExistingName) ->
        % Variable already bound — generate join condition
        JoinConds = [Param-ExistingName|RestConds],
        build_join_conditions(Args, Params, Bindings, RestConds, NewBindings)
    ;   var(Arg) ->
        % New variable — add to bindings
        build_join_conditions(Args, Params, [Arg-Param|Bindings], RestConds, NewBindings),
        JoinConds = RestConds
    ;   % Constant — generate equality check
        format_lua_fact_arg(Arg, FormattedArg),
        JoinConds = [Param-FormattedArg|RestConds],
        build_join_conditions(Args, Params, Bindings, RestConds, NewBindings)
    ).

lookup_binding(Var, [V-Name|_], Name) :- Var == V, !.
lookup_binding(Var, [_|Rest], Name) :- lookup_binding(Var, Rest, Name).

%% format_join_guards(+JoinConds, -GuardCode)
format_join_guards([], "") :- !.
format_join_guards(Conds, Code) :-
    findall(Guard, (
        member(Param-Expected, Conds),
        format(string(Guard), '        if ~w ~~= ~w then return end', [Param, Expected])
    ), Guards),
    atomic_list_concat(Guards, '\n', Code).

%% resolve_head_arg(+Bindings, +Arg, -Resolved)
resolve_head_arg(Bindings, Arg, Resolved) :-
    (   var(Arg), lookup_binding(Arg, Bindings, Name) ->
        Resolved = Name
    ;   var(Arg) ->
        Resolved = 'nil'
    ;   format_lua_fact_arg(Arg, Resolved)
    ).

format_head_arg(Arg, Str) :-
    (   number(Arg) -> format(string(Str), '~w', [Arg])
    ;   atom(Arg) -> format(string(Str), '"~w"', [Arg])
    ;   format(string(Str), '~w', [Arg])
    ).

% ============================================================================
% SEMI-NAIVE RECURSIVE COMPILATION
% ============================================================================

%% compile_recursive_rules(+Pred, +Arity, +Clauses, -LuaCode)
%  Compiles recursive predicates using semi-naive iteration with delta/seen.
compile_recursive_rules(Pred, Arity, Clauses, LuaCode) :-
    partition(clause_is_recursive(Pred), Clauses, RecClauses, BaseClauses),
    compile_base_cases_for_delta(Pred, Arity, BaseClauses, BaseCode),
    compile_recursive_expansion(Pred, Arity, RecClauses, ExpansionCode),
    format(string(LuaCode),
'local function ~w(callback)
    local delta = {}
    local seen = {}

    -- Base cases: seed the worklist
~w

    -- Semi-naive iteration
    while #delta > 0 do
        local item = table.remove(delta, 1)
        callback(table.unpack(item))

        -- Expand: find new results via recursive clauses
~w
    end
end
', [Pred, BaseCode, ExpansionCode]).

clause_is_recursive(Pred, _Head-Body) :-
    body_calls_pred(Body, Pred).

%% compile_base_cases_for_delta(+Pred, +Arity, +BaseClauses, -Code)
compile_base_cases_for_delta(_Pred, Arity, BaseClauses, Code) :-
    findall(CaseCode, (
        member(Head-Body, BaseClauses),
        Head =.. [_|HeadArgs],
        compile_base_case_delta(Body, HeadArgs, Arity, 0, CaseCode)
    ), CaseCodes),
    atomic_list_concat(CaseCodes, '\n', Code).

compile_base_case_delta(true, HeadArgs, _Arity, _Counter, Code) :- !,
    maplist(format_head_arg, HeadArgs, ArgStrs),
    atomic_list_concat(ArgStrs, ', ', ArgsStr),
    key_expr_from_args(ArgStrs, KeyExpr),
    format(string(Code),
'    do
        local key = ~w
        if not seen[key] then
            seen[key] = true
            table.insert(delta, {~w})
        end
    end', [KeyExpr, ArgsStr]).
compile_base_case_delta(Body, HeadArgs, _Arity, Counter, Code) :-
    body_to_goals(Body, Goals),
    compile_base_goals_for_delta(Goals, HeadArgs, Counter, [], Code).

compile_base_goals_for_delta([], HeadArgs, _Counter, Bindings, Code) :- !,
    maplist(resolve_head_arg(Bindings), HeadArgs, ResolvedArgs),
    atomic_list_concat(ResolvedArgs, ', ', ArgsStr),
    key_expr_from_list(ResolvedArgs, KeyExpr),
    format(string(Code),
'        local key = ~w
        if not seen[key] then
            seen[key] = true
            table.insert(delta, {~w})
        end', [KeyExpr, ArgsStr]).
compile_base_goals_for_delta([Goal|Rest], HeadArgs, Counter, Bindings, Code) :-
    (   Goal = (_ is _) ->
        compile_base_goals_for_delta(Rest, HeadArgs, Counter, Bindings, Code)
    ;   Goal =.. [GoalPred|GoalArgs],
        length(GoalArgs, GoalArity),
        create_callback_params(GoalArity, Counter, Params, NextCounter),
        build_join_conditions(GoalArgs, Params, Bindings, JoinConds, NewBindings),
        atomic_list_concat(Params, ', ', ParamList),
        format_join_guards(JoinConds, GuardCode),
        compile_base_goals_for_delta(Rest, HeadArgs, NextCounter, NewBindings, InnerCode),
        (   GuardCode = "" ->
            format(string(Code),
'    ~w(function(~w)
~w
    end)', [GoalPred, ParamList, InnerCode])
        ;   format(string(Code),
'    ~w(function(~w)
~w
~w
    end)', [GoalPred, ParamList, GuardCode, InnerCode])
        )
    ).

%% compile_recursive_expansion(+Pred, +Arity, +RecClauses, -Code)
compile_recursive_expansion(Pred, _Arity, RecClauses, Code) :-
    findall(ExpCode, (
        member(Head-Body, RecClauses),
        Head =.. [_|HeadArgs],
        body_to_goals(Body, Goals),
        % Find the recursive call to get item bindings
        find_recursive_goal(Goals, Pred, RecGoal),
        RecGoal =.. [_|RecArgs],
        % Bind recursive call args to item[N] positions (1-indexed for Lua)
        bind_to_item_positions(RecArgs, 1, ItemBindings),
        % Get non-recursive, non-arithmetic goals
        exclude(goal_is_recursive_or_arith(Pred), Goals, NonRecGoals),
        compile_expansion_goals(NonRecGoals, HeadArgs, 0, ItemBindings, ExpCode)
    ), ExpCodes),
    atomic_list_concat(ExpCodes, '\n', Code).

find_recursive_goal([Goal|_], Pred, Goal) :-
    Goal =.. [Pred|_], !.
find_recursive_goal([_:Goal|_], Pred, Goal) :-
    Goal =.. [Pred|_], !.
find_recursive_goal([_|Rest], Pred, Goal) :-
    find_recursive_goal(Rest, Pred, Goal).

goal_is_recursive_or_arith(Pred, Goal) :-
    (   Goal = (_ is _)
    ;   Goal = (_ > _)
    ;   Goal = (_ < _)
    ;   Goal = (_ >= _)
    ;   Goal = (_ =< _)
    ;   Goal =.. [Pred|_]
    ;   Goal = _:G, G =.. [Pred|_]
    ).

%% bind_to_item_positions(+Args, +Pos, -Bindings)
%  Bind variables to item[Pos] references.
bind_to_item_positions([], _, []).
bind_to_item_positions([Arg|Rest], Pos, Bindings) :-
    NextPos is Pos + 1,
    (   var(Arg) ->
        format(atom(ItemRef), 'item[~w]', [Pos]),
        Bindings = [Arg-ItemRef|RestBindings],
        bind_to_item_positions(Rest, NextPos, RestBindings)
    ;   bind_to_item_positions(Rest, NextPos, Bindings)
    ).

%% compile_expansion_goals(+Goals, +HeadArgs, +Counter, +Bindings, -Code)
compile_expansion_goals([], HeadArgs, _Counter, Bindings, Code) :- !,
    maplist(resolve_expansion_arg(Bindings), HeadArgs, ResolvedArgs),
    atomic_list_concat(ResolvedArgs, ', ', ArgsStr),
    key_expr_from_list(ResolvedArgs, KeyExpr),
    format(string(Code),
'        local key = ~w
        if not seen[key] then
            seen[key] = true
            table.insert(delta, {~w})
        end', [KeyExpr, ArgsStr]).
compile_expansion_goals([Goal|Rest], HeadArgs, Counter, Bindings, Code) :-
    Goal =.. [GoalPred|GoalArgs],
    length(GoalArgs, GoalArity),
    create_callback_params(GoalArity, Counter, Params, NextCounter),
    build_expansion_join_conditions(GoalArgs, Params, Bindings, JoinConds, NewBindings),
    atomic_list_concat(Params, ', ', ParamList),
    format_join_guards(JoinConds, GuardCode),
    compile_expansion_goals(Rest, HeadArgs, NextCounter, NewBindings, InnerCode),
    (   GuardCode = "" ->
        format(string(Code),
'        ~w(function(~w)
~w
        end)', [GoalPred, ParamList, InnerCode])
    ;   format(string(Code),
'        ~w(function(~w)
~w
~w
        end)', [GoalPred, ParamList, GuardCode, InnerCode])
    ).

%% build_expansion_join_conditions(+GoalArgs, +Params, +Bindings, -JoinConds, -NewBindings)
build_expansion_join_conditions([], [], Bindings, [], Bindings).
build_expansion_join_conditions([Arg|Args], [Param|Params], Bindings, JoinConds, NewBindings) :-
    (   var(Arg), lookup_binding(Arg, Bindings, ExistingName) ->
        JoinConds = [Param-ExistingName|RestConds],
        build_expansion_join_conditions(Args, Params, Bindings, RestConds, NewBindings)
    ;   var(Arg) ->
        build_expansion_join_conditions(Args, Params, [Arg-Param|Bindings], RestConds, NewBindings),
        JoinConds = RestConds
    ;   format_lua_fact_arg(Arg, FormattedArg),
        JoinConds = [Param-FormattedArg|RestConds],
        build_expansion_join_conditions(Args, Params, Bindings, RestConds, NewBindings)
    ).

%% resolve_expansion_arg(+Bindings, +Arg, -Resolved)
resolve_expansion_arg(Bindings, Arg, Resolved) :-
    (   var(Arg), lookup_binding(Arg, Bindings, Name) ->
        Resolved = Name
    ;   var(Arg) ->
        Resolved = 'nil'
    ;   format_lua_fact_arg(Arg, Resolved)
    ).

% Helper to build dedup key from argument strings
key_expr_from_args(ArgStrs, KeyExpr) :-
    findall(Part, (
        member(A, ArgStrs),
        format(string(Part), 'tostring(~w)', [A])
    ), Parts),
    atomic_list_concat(Parts, ' .. "\\0" .. ', KeyExpr).

key_expr_from_list(ArgList, KeyExpr) :-
    findall(Part, (
        member(A, ArgList),
        format(string(Part), 'tostring(~w)', [A])
    ), Parts),
    atomic_list_concat(Parts, ' .. "\\0" .. ', KeyExpr).

% Variable resolution helpers (kept for binding-based compilation)
resolve_val(VarMap, Var, Val) :- var(Var), lookup_binding(Var, VarMap, Name), !, format(string(Val), '~w', [Name]).
resolve_val(_, Val, StrVal) :- format(string(StrVal), '~w', [Val]).

ensure_var(VarMap, Var, Name, VarMap) :- lookup_binding(Var, VarMap, Name), !.
ensure_var(VarMap, Var, Name, [Var-Name|VarMap]) :- gensym(v, Name).


% ============================================================================
% PIPELINE MODE
% ============================================================================

compile_lua_pipeline(Predicates, Options, LuaCode) :-
    (member(pipeline_name(PipelineName), Options) -> true ; PipelineName = 'pipeline'),
    (member(pipeline_mode(Mode), Options) -> true ; Mode = sequential),
    
    lua_pipeline_header(HeaderCode),
    generate_lua_stage_functions(Predicates, StageFunctions),
    generate_lua_pipeline_connector(Predicates, PipelineName, Mode, ConnectorCode),
    
    format(string(LuaCode), '~s~n~n~s~n~n~s',
           [HeaderCode, StageFunctions, ConnectorCode]).

lua_pipeline_header(Code) :-
    format(string(Code),
'-- Auto-generated Lua Pipeline
local json = require("cjson") -- assuming cjson is available for json parsing

-- Helper for reading JSONL
local function read_jsonl(file_path)
    local lines = {}
    local file = io.open(file_path, "r")
    if file then
        for line in file:lines() do
            table.insert(lines, json.decode(line))
        end
        file:close()
    end
    return lines
end
', []).

generate_lua_stage_functions([], "").
generate_lua_stage_functions([Pred|Rest], Code) :-
    compile_predicate_to_lua(Pred, [], StageCode),
    generate_lua_stage_functions(Rest, RestCode),
    (RestCode = "" -> Code = StageCode ; format(string(Code), '~s~n~n~s', [StageCode, RestCode])).

generate_lua_pipeline_connector(Predicates, PipelineName, sequential, Code) :-
    maplist(get_pred_name, Predicates, StageNames),
    % sequential nested calls: e.g. stage3(stage2(stage1(row)))
    generate_nested_calls(StageNames, "row", NestedCall),
    format(string(Code),
'-- Sequential pipeline connector: ~w
local function ~w(data)
    local output = {}
    for _, row in ipairs(data) do
        local result = ~w
        table.insert(output, result)
    end
    return output
end
', [PipelineName, PipelineName, NestedCall]).

generate_lua_pipeline_connector(Predicates, PipelineName, generator, Code) :-
    maplist(get_pred_name, Predicates, StageNames),
    generate_nested_calls(StageNames, "row", NestedCall),
    format(string(Code),
'-- Fixpoint pipeline connector: ~w
-- Iterates until no new rows are produced.
local function ~w(data)
    local seen = {}
    local function hash_row(row)
        return json.encode(row) -- quick and dirty hash for lua tables
    end
    
    local records = {}
    local output_records = {}
    for _, row in ipairs(data) do
        seen[hash_row(row)] = true
        table.insert(records, row)
        table.insert(output_records, row)
    end
    
    local changed = true
    while changed do
        changed = false
        local current_records = records
        records = {}
        
        for _, row in ipairs(current_records) do
            local new_row = ~w
            local key = hash_row(new_row)
            if not seen[key] then
                seen[key] = true
                table.insert(records, new_row)
                table.insert(output_records, new_row)
                changed = true
            end
        end
    end
    return output_records
end
', [PipelineName, PipelineName, NestedCall]).

get_pred_name(Pred/_Arity, Pred).

generate_nested_calls([], Var, Var).
generate_nested_calls([Stage|Rest], Var, Result) :-
    format(string(CallStr), '~w(~w)', [Stage, Var]),
    generate_nested_calls(Rest, CallStr, Result).


% ============================================================================
% MULTIFILE RECURSION PATTERNS FOR LUA
% ============================================================================

:- multifile tree_recursion:compile_tree_pattern/6.
tree_recursion:compile_tree_pattern(lua, fibonacci, Pred, _Arity, UseMemo, LuaCode) :-
    atom_string(Pred, PredStr),
    (   UseMemo = true ->
        MemoDecl = 'local _memo = {}'
    ;   MemoDecl = '-- Memoization disabled'
    ),
    format(string(LuaCode),
'-- ~w/2 - tree recursive pattern (Fibonacci-like in Lua)
~w
local function ~w(n, expected)
    if n == 0 then return 0 end
    if n == 1 then return 1 end
    
    local key = tostring(n)
    local result
    if _memo and _memo[key] then
        result = _memo[key]
    else
        result = ~w(n - 1) + ~w(n - 2)
        if _memo then _memo[key] = result end
    end
    
    if expected ~~~= nil then
        return result == expected
    end
    return result
end
', [PredStr, MemoDecl, PredStr, PredStr, PredStr]).

tree_recursion:compile_tree_pattern(lua, binary_tree, Pred, Arity, _UseMemo, LuaCode) :-
    atom_string(Pred, PredStr),
    format(string(LuaCode), '-- ~w/~w - binary tree recursion (Lua)\n-- Not fully implemented', [PredStr, Arity]).

tree_recursion:compile_tree_pattern(lua, generic, Pred, Arity, _UseMemo, LuaCode) :-
    atom_string(Pred, PredStr),
    format(string(LuaCode), '-- ~w/~w - generic tree recursion (Lua)\n-- Not fully implemented', [PredStr, Arity]).

:- multifile tail_recursion:compile_tail_pattern/9.
tail_recursion:compile_tail_pattern(lua, PredStr, Arity, _BaseClauses, _RecClauses, _AccPos, StepOp, _ExitAfterResult, LuaCode) :-
    (   Arity =:= 3 ->
        generate_ternary_tail_loop_lua(PredStr, StepOp, LuaCode)
    ;   Arity =:= 2 ->
        generate_binary_tail_loop_lua(PredStr, LuaCode)
    ;   format('Warning: tail recursion in Lua with arity ~w not yet supported~n', [Arity]),
        fail
    ).

step_op_to_lua(arithmetic(Expr), LuaCode) :-
    expr_to_lua(Expr, LuaExpr),
    format(atom(LuaCode), 'current_acc = ~w', [LuaExpr]).
step_op_to_lua(unknown, 'current_acc = current_acc + 1').

expr_to_lua(_ + Const, LuaExpr) :- integer(Const), !, format(atom(LuaExpr), 'current_acc + ~w', [Const]).
expr_to_lua(_ + _, 'current_acc + item') :- !.
expr_to_lua(_ - _, 'current_acc - item') :- !.
expr_to_lua(_ * _, 'current_acc * item') :- !.
expr_to_lua(_, 'current_acc + 1').

generate_ternary_tail_loop_lua(PredStr, StepOp, LuaCode) :-
    step_op_to_lua(StepOp, LStepOp),
    format(string(LuaCode),
'-- ~w - tail recursive accumulator pattern (Lua)
local function ~w(items, acc)
    local current_acc = acc or 0
    for _, item in ipairs(items) do
        ~w
    end
    return current_acc
end
', [PredStr, PredStr, LStepOp]).

generate_binary_tail_loop_lua(PredStr, LuaCode) :-
    format(string(LuaCode),
'-- ~w - tail recursive binary pattern (Lua)
local function ~w(items)
    local count = 0
    for _ in ipairs(items) do
        count = count + 1
    end
    return count
end
', [PredStr, PredStr]).

:- multifile multicall_linear_recursion:compile_multicall_pattern/6.
multicall_linear_recursion:compile_multicall_pattern(lua, PredStr, _BaseClauses, _RecClauses, _MemoEnabled, LuaCode) :-
    format(string(LuaCode), '-- ~w/2 - multicall linear recursion (Lua)\n-- Not fully implemented', [PredStr]).

:- multifile direct_multi_call_recursion:compile_direct_multicall_pattern/5.
direct_multi_call_recursion:compile_direct_multicall_pattern(lua, PredStr, _BaseClauses, _RecClause, LuaCode) :-
    format(string(LuaCode), '-- ~w/2 - direct multicall recursion (Lua)\n-- Not fully implemented', [PredStr]).

:- multifile linear_recursion:compile_linear_pattern/8.
linear_recursion:compile_linear_pattern(lua, PredStr, Arity, BaseClauses, RecClauses, MemoEnabled, _MemoStrategy, LuaCode) :-
    (   Arity =:= 2 ->
        generate_fold_based_recursion_lua(PredStr, BaseClauses, RecClauses, MemoEnabled, LuaCode)
    ;   generate_generic_linear_recursion_lua(PredStr, Arity, MemoEnabled, LuaCode)
    ).

generate_fold_based_recursion_lua(PredStr, BaseClauses, RecClauses, MemoEnabled, LuaCode) :-
    linear_recursion:extract_base_case_info(BaseClauses, BaseInput, BaseOutput),
    linear_recursion:detect_input_type(BaseInput, InputType),
    (   InputType = numeric ->
        generate_numeric_fold_lua(PredStr, BaseClauses, RecClauses, BaseInput, BaseOutput, MemoEnabled, LuaCode)
    ;   InputType = list ->
        generate_list_fold_lua(PredStr, BaseClauses, RecClauses, BaseInput, BaseOutput, MemoEnabled, LuaCode)
    ;   generate_generic_linear_recursion_lua(PredStr, 2, MemoEnabled, LuaCode)
    ).

generate_numeric_fold_lua(PredStr, _BaseClauses, RecClauses, BaseInput, BaseOutput, MemoEnabled, LuaCode) :-
    RecClauses = [clause(RHead, RBody)|_],
    RHead =.. [_, InputVar, _],
    linear_recursion:find_recursive_call(RBody, RecCall),
    RecCall =.. [_, _, AccVar],
    linear_recursion:find_last_is_expression(RBody, _ is ActualFoldExpr),
    translate_fold_expr_lua(ActualFoldExpr, InputVar, AccVar, LFoldOp),
    (   MemoEnabled = true ->
        format(string(LuaCode),
'-- ~w - fold-based linear recursion (numeric, Lua)
local ~w_memo = {}

local function ~w(n)
    if ~w_memo[n] then return ~w_memo[n] end
    if n == ~w then return ~w end
    local result = ~w
    for current = n, 1, -1 do
        result = ~w
    end
    ~w_memo[n] = result
    return result
end
', [PredStr, PredStr, PredStr, PredStr, PredStr, BaseInput, BaseOutput, BaseOutput, LFoldOp, PredStr])
    ;   format(string(LuaCode),
'-- ~w - fold-based linear recursion (numeric, Lua)
local function ~w(n)
    if n == ~w then return ~w end
    local result = ~w
    for current = n, 1, -1 do
        result = ~w
    end
    return result
end
', [PredStr, PredStr, BaseInput, BaseOutput, BaseOutput, LFoldOp])
    ).

generate_list_fold_lua(PredStr, _BaseClauses, RecClauses, _BaseInput, BaseOutput, MemoEnabled, LuaCode) :-
    RecClauses = [clause(RHead, RBody)|_],
    RHead =.. [_, _, _],
    linear_recursion:find_recursive_call(RBody, RecCall),
    RecCall =.. [_, _, AccVar],
    linear_recursion:find_last_is_expression(RBody, _ is ActualFoldExpr),
    translate_fold_expr_lua(ActualFoldExpr, _, AccVar, LFoldOp),
    (   MemoEnabled = true ->
        format(string(LuaCode),
'-- ~w - fold-based linear recursion (list, Lua)
local ~w_memo = {}

local function ~w(lst)
    local key = table.concat(lst, ",")
    if ~w_memo[key] then return ~w_memo[key] end
    if #lst == 0 then return ~w end
    local result = ~w
    for i = #lst, 1, -1 do
        local current = lst[i]
        result = ~w
    end
    ~w_memo[key] = result
    return result
end
', [PredStr, PredStr, PredStr, PredStr, PredStr, BaseOutput, BaseOutput, LFoldOp, PredStr])
    ;   format(string(LuaCode),
'-- ~w - fold-based linear recursion (list, Lua)
local function ~w(lst)
    if #lst == 0 then return ~w end
    local result = ~w
    for i = #lst, 1, -1 do
        local current = lst[i]
        result = ~w
    end
    return result
end
', [PredStr, PredStr, BaseOutput, BaseOutput, LFoldOp])
    ).

generate_generic_linear_recursion_lua(PredStr, Arity, MemoEnabled, LuaCode) :-
    (   MemoEnabled = true ->
        format(string(LuaCode),
'-- ~w/~w - generic linear recursion (Lua)
local ~w_memo = {}

local function ~w(n)
    if ~w_memo[n] then return ~w_memo[n] end
    -- Generic linear recursion: implement specific logic here
    return nil
end
', [PredStr, Arity, PredStr, PredStr, PredStr, PredStr])
    ;   format(string(LuaCode),
'-- ~w/~w - generic linear recursion (Lua)
local function ~w(n)
    -- Generic linear recursion: implement specific logic here
    return nil
end
', [PredStr, Arity, PredStr])
    ).

% Translate fold expressions to Lua syntax
translate_fold_expr_lua(Expr, InputVar, AccVar, LuaExpr) :-
    translate_fold_term_lua(Expr, InputVar, AccVar, LuaExpr).

% Variables must be checked BEFORE compound term patterns to prevent
% unification of variables with *(X,Y) / +(X,Y) / -(X,Y) patterns.
translate_fold_term_lua(Var, InputVar, _AccVar, 'current') :-
    var(Var), Var == InputVar, !.
translate_fold_term_lua(Var, _InputVar, AccVar, 'result') :-
    var(Var), Var == AccVar, !.
translate_fold_term_lua(Var, _, _, 'current') :-
    var(Var), !.
translate_fold_term_lua(Num, _, _, Str) :-
    number(Num), !, format(atom(Str), '~w', [Num]).
translate_fold_term_lua(X * Y, InputVar, AccVar, LuaExpr) :- !,
    translate_fold_term_lua(X, InputVar, AccVar, LX),
    translate_fold_term_lua(Y, InputVar, AccVar, LY),
    format(atom(LuaExpr), '~w * ~w', [LX, LY]).
translate_fold_term_lua(X + Y, InputVar, AccVar, LuaExpr) :- !,
    translate_fold_term_lua(X, InputVar, AccVar, LX),
    translate_fold_term_lua(Y, InputVar, AccVar, LY),
    format(atom(LuaExpr), '~w + ~w', [LX, LY]).
translate_fold_term_lua(X - Y, InputVar, AccVar, LuaExpr) :- !,
    translate_fold_term_lua(X, InputVar, AccVar, LX),
    translate_fold_term_lua(Y, InputVar, AccVar, LY),
    format(atom(LuaExpr), '~w - ~w', [LX, LY]).
translate_fold_term_lua(Term, _, _, Str) :-
    format(atom(Str), '~w', [Term]).

:- multifile mutual_recursion:compile_mutual_pattern/5.
mutual_recursion:compile_mutual_pattern(lua, Predicates, MemoEnabled, _MemoStrategy, LuaCode) :-
    % Generate forward declarations
    findall(FwdDecl, (
        member(Pred/_, Predicates),
        format(string(FwdDecl), 'local ~w', [Pred])
    ), FwdDecls),
    atomic_list_concat(FwdDecls, '\n', ForwardDecls),
    % Generate each function
    mutual_functions_lua(Predicates, Predicates, MemoEnabled, subsequent, FuncCodes),
    atomic_list_concat(FuncCodes, '\n\n', FunctionsCode),
    format(string(LuaCode),
'-- Mutually recursive group (Lua)
~w

~w
', [ForwardDecls, FunctionsCode]).

mutual_functions_lua([], _AllPreds, _MemoEnabled, _Position, []).
mutual_functions_lua([Pred/Arity|Rest], AllPreds, MemoEnabled, _Position, [Code|RestCodes]) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   Clauses = [] ->
        format(string(Code), '~w = function(n)\n    return nil\nend', [PredStr])
    ;   compile_mutual_function_lua(PredStr, Arity, Clauses, AllPreds, MemoEnabled, Code)
    ),
    mutual_functions_lua(Rest, AllPreds, MemoEnabled, subsequent, RestCodes).

compile_mutual_function_lua(PredStr, _Arity, Clauses, _AllPreds, MemoEnabled, Code) :-
    findall(CaseCode, (
        member(Head-Body, Clauses),
        Head =.. [_|HeadArgs],
        compile_mutual_clause(HeadArgs, Body, CaseCode)
    ), CaseCodes),
    atomic_list_concat(CaseCodes, '\n', CasesCode),
    (   MemoEnabled = true ->
        format(string(Code),
'local ~w_memo = {}
~w = function(n)
    if ~w_memo[n] ~~= nil then return ~w_memo[n] end
~w
    ~w_memo[n] = result
    return result
end', [PredStr, PredStr, PredStr, PredStr, CasesCode, PredStr])
    ;   format(string(Code),
'~w = function(n)
~w
end', [PredStr, CasesCode])
    ).

compile_mutual_clause(HeadArgs, fail, Code) :- !,
    HeadArgs = [InputArg|_],
    (   number(InputArg) ->
        format(string(Code), '    if n == ~w then return false end', [InputArg])
    ;   format(string(Code), '    if n == "~w" then return false end', [InputArg])
    ).
compile_mutual_clause(HeadArgs, true, Code) :- !,
    HeadArgs = [InputArg|_],
    (   number(InputArg) ->
        format(string(Code), '    if n == ~w then return true end', [InputArg])
    ;   format(string(Code), '    if n == "~w" then return true end', [InputArg])
    ).
compile_mutual_clause(HeadArgs, Body, Code) :-
    HeadArgs = [InputArg|_],
    % Collect arithmetic bindings (N1 is N - 1 => N1 -> n - 1)
    collect_arith_bindings(Body, HeadArgs, ArithBindings),
    compile_mutual_body(Body, ArithBindings, BodyExpr),
    (   InputArg == 0 ->
        format(string(Code), '    if n == 0 then return ~w end', [BodyExpr])
    ;   number(InputArg) ->
        format(string(Code), '    if n == ~w then return ~w end', [InputArg, BodyExpr])
    ;   var(InputArg) ->
        format(string(Code), '    local result = ~w', [BodyExpr])
    ;   format(string(Code), '    if n == "~w" then return ~w end', [InputArg, BodyExpr])
    ).

%% collect_arith_bindings(+Body, +HeadArgs, -Bindings)
%  Scan body for `Var is Expr` patterns and build substitution map.
collect_arith_bindings((A, B), HeadArgs, Bindings) :- !,
    collect_arith_bindings(A, HeadArgs, B1),
    collect_arith_bindings(B, HeadArgs, B2),
    append(B1, B2, Bindings).
collect_arith_bindings(Var is Expr, HeadArgs, [Var-LuaExpr]) :- !,
    compile_mutual_arg(Expr, HeadArgs, [], LuaExpr).
collect_arith_bindings(_, _, []).

compile_mutual_body((A, B), Bindings, Expr) :- !,
    compile_mutual_body(B, Bindings, Expr),
    (   A = (_ > _) -> true  % Skip guards
    ;   A = (_ is _) -> true % Skip arithmetic (already collected)
    ;   true
    ).
compile_mutual_body(_:Goal, Bindings, Expr) :- !, compile_mutual_body(Goal, Bindings, Expr).
compile_mutual_body(Goal, Bindings, Expr) :-
    Goal =.. [Pred|Args],
    (   Args = [Arg] ->
        compile_mutual_arg(Arg, [], Bindings, ArgStr),
        format(atom(Expr), '~w(~w)', [Pred, ArgStr])
    ;   Args = [A1, A2] ->
        compile_mutual_arg(A1, [], Bindings, A1Str),
        compile_mutual_arg(A2, [], Bindings, A2Str),
        format(atom(Expr), '~w(~w, ~w)', [Pred, A1Str, A2Str])
    ;   format(atom(Expr), '~w(n)', [Pred])
    ).

%% compile_mutual_arg(+Arg, +HeadArgs, +ArithBindings, -Str)
compile_mutual_arg(Arg, _HeadArgs, _Bindings, Str) :-
    number(Arg), !, format(atom(Str), '~w', [Arg]).
compile_mutual_arg(Arg, _HeadArgs, Bindings, Str) :-
    var(Arg), lookup_binding(Arg, Bindings, BoundExpr), !,
    Str = BoundExpr.
compile_mutual_arg(Arg, _HeadArgs, _Bindings, 'n') :-
    var(Arg), !.
compile_mutual_arg(X - Y, HeadArgs, Bindings, Str) :- !,
    compile_mutual_arg(X, HeadArgs, Bindings, XS),
    compile_mutual_arg(Y, HeadArgs, Bindings, YS),
    format(atom(Str), '~w - ~w', [XS, YS]).
compile_mutual_arg(X + Y, HeadArgs, Bindings, Str) :- !,
    compile_mutual_arg(X, HeadArgs, Bindings, XS),
    (   number(Y), Y < 0 ->
        AbsY is abs(Y),
        format(atom(Str), '~w - ~w', [XS, AbsY])
    ;   compile_mutual_arg(Y, HeadArgs, Bindings, YS),
        format(atom(Str), '~w + ~w', [XS, YS])
    ).
compile_mutual_arg(Term, _, _, Str) :-
    format(atom(Str), '~w', [Term]).


% ============================================================================
% TESTING
% ============================================================================

test_lua_pipeline :-
    init_lua_target,
    asserta((user:step1(In, Out) :- true)),
    asserta((user:step2(In, Out) :- true)),
    compile_lua_pipeline([step1/2, step2/2], [pipeline_name(my_lua_pipe), pipeline_mode(sequential)], CodeSeq),
    format('~n=== Lua Sequential Pipeline Test ===~n~s~n', [CodeSeq]),
    compile_lua_pipeline([step1/2, step2/2], [pipeline_name(my_lua_pipe_gen), pipeline_mode(generator)], CodeGen),
    format('~n=== Lua Generator Pipeline Test ===~n~s~n', [CodeGen]),
    retractall(user:step1(_, _)),
    retractall(user:step2(_, _)).
