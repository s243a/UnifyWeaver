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

%% compile_facts_to_lua(+PredIndicator, +Options, -LuaCode)
compile_facts_to_lua(Pred/Arity, _Options, LuaCode) :-
    functor(Head, Pred, Arity),
    findall(Head, user:clause(Head, true), _Facts),
    format(string(LuaCode), '-- Facts for ~w/~w not fully implemented in generic Lua yet.', [Pred, Arity]).

%% compile_predicate_to_lua(+PredIndicator, +Options, -LuaCode)
compile_predicate_to_lua(PredIndicator, _Options, LuaCode) :-
    PredIndicator = Pred/Arity,
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   Clauses = [Head-Body]
    ->  compile_rule(Head, Body, LuaCode)
    ;   format(string(LuaCode), '-- Multiple clauses not supported yet for ~w/~w', [Pred, Arity])
    ).

compile_rule(Head, Body, Code) :-
    Head =.. [Pred|Args],
    map_args(Args, 1, VarMap, ArgNames),
    atomic_list_concat(ArgNames, ', ', ArgList),
    compile_body(Body, VarMap, _, BodyCode),
    format(string(Code),
'local function ~w(~w)
~s
end
', [Pred, ArgList, BodyCode]).

map_args([], _, [], []).
map_args([Arg|Rest], Idx, [Arg-VarName|MapRest], [VarName|NamesRest]) :-
    format(atom(VarName), 'arg~w', [Idx]),
    NextIdx is Idx + 1,
    map_args(Rest, NextIdx, MapRest, NamesRest).

compile_body(true, V, V, "") :- !.
compile_body((A, B), V0, V2, Code) :- !,
    compile_body(A, V0, V1, C1),
    compile_body(B, V1, V2, C2),
    format(string(Code), '~s~n~s', [C1, C2]).
compile_body(_M:Goal, V0, V1, Code) :- !,
    compile_body(Goal, V0, V1, Code).
compile_body(Goal, V0, V1, Code) :-
    functor(Goal, Pred, Arity),
    (   binding(lua, Pred/Arity, TargetName, Inputs, Outputs, _Options)
    ->  Goal =.. [_|Args],
        length(Inputs, InCount),
        length(InArgs, InCount),
        append(InArgs, OutArgs, Args),
        
        maplist(resolve_val(V0), InArgs, LInArgs),
        atomic_list_concat(LInArgs, ', ', LArgsStr),
        format(string(Expr), '~w(~w)', [TargetName, LArgsStr]),
        
        (   Outputs = []
        ->  V1 = V0,
            format(string(Code), '    ~w', [Expr])
        ;   OutArgs = [OutVar],
            ensure_var(V0, OutVar, LOutVar, V1),
            format(string(Code), '    local ~w = ~w', [LOutVar, Expr])
        )
    ;   V1 = V0,
        format(string(Code), '    -- Unknown predicate: ~w', [Goal])
    ).

resolve_val(VarMap, Var, Val) :- var(Var), lookup_var(Var, VarMap, Name), !, format(string(Val), '~w', [Name]).
resolve_val(_, Val, StrVal) :- format(string(StrVal), '~w', [Val]).

ensure_var(VarMap, Var, Name, VarMap) :- lookup_var(Var, VarMap, Name), !.
ensure_var(VarMap, Var, Name, [Var-Name|VarMap]) :- gensym(v, Name).

lookup_var(Var, [V-Name|_], Name) :- Var == V, !.
lookup_var(Var, [_|Rest], Name) :- lookup_var(Var, Rest, Name).


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
    
    if expected ~~= nil then
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
tail_recursion:compile_tail_pattern(lua, PredStr, Arity, _BaseClauses, _RecClauses, AccPos, StepOp, ExitAfterResult, LuaCode) :-
    (   Arity =:= 3 ->
        generate_ternary_tail_loop_lua(PredStr, AccPos, StepOp, ExitAfterResult, LuaCode)
    ;   Arity =:= 2 ->
        generate_binary_tail_loop_lua(PredStr, ExitAfterResult, LuaCode)
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

generate_ternary_tail_loop_lua(PredStr, _AccPos, StepOp, ExitAfterResult, LuaCode) :-
    step_op_to_lua(StepOp, LStepOp),
    (   ExitAfterResult = true ->
        ExitStatement = "        return current_acc  -- Unique constraint"
    ;   ExitStatement = ""
    ),

    format(string(LuaCode),
'-- ~w - tail recursive accumulator pattern (Lua)
local function ~w(input, acc)
    local items = input
    -- Check if input is a JSON string (simulate array)
    if type(input) == "string" then
        -- Extremely simplified split, assuming a JSON-like array string "[1, 2, 3]"
        items = {}
        local clean = input:gsub("%[", ""):gsub("%]", "")
        for val in clean:gmatch("[^,]+") do
            table.insert(items, tonumber(val))
        end
    end
    
    local current_acc = acc
    for _, item in ipairs(items) do
        ~w
~w
    end
    return current_acc
end

local function ~w_eval(input)
    return ~w(input, 0)
end
', [PredStr, PredStr, LStepOp, ExitStatement, PredStr, PredStr]).

generate_binary_tail_loop_lua(PredStr, ExitAfterResult, LuaCode) :-
    (   ExitAfterResult = true ->
        ExitStatement = "    return count  -- Unique constraint"
    ;   ExitStatement = ""
    ),

    format(string(LuaCode),
'-- ~w - tail recursive binary pattern (Lua)
local function ~w(input, expected)
    local count = 0
    local items = input
    if type(input) == "string" then
        items = {}
        local clean = input:gsub("%[", ""):gsub("%]", "")
        for val in clean:gmatch("[^,]+") do
            table.insert(items, tonumber(val))
        end
    end
    
    local i = 1
    while items[i] ~~= nil do
        count = count + 1
        i = i + 1
    end
    
    if expected ~~= nil then
        return count == expected
    end
    return count
~w
end
', [PredStr, PredStr, ExitStatement]).

:- multifile multicall_linear_recursion:compile_multicall_pattern/6.
multicall_linear_recursion:compile_multicall_pattern(lua, PredStr, _BaseClauses, _RecClauses, _MemoEnabled, LuaCode) :-
    format(string(LuaCode), '-- ~w/2 - multicall linear recursion (Lua)\n-- Not fully implemented', [PredStr]).

:- multifile direct_multi_call_recursion:compile_direct_multicall_pattern/5.
direct_multi_call_recursion:compile_direct_multicall_pattern(lua, PredStr, _BaseClauses, _RecClause, LuaCode) :-
    format(string(LuaCode), '-- ~w/2 - direct multicall recursion (Lua)\n-- Not fully implemented', [PredStr]).

:- multifile linear_recursion:compile_linear_pattern/8.
linear_recursion:compile_linear_pattern(lua, PredStr, Arity, BaseClauses, RecClauses, MemoEnabled, MemoStrategy, LuaCode) :-
    (   Arity =:= 2 ->
        generate_fold_based_recursion_lua(PredStr, BaseClauses, RecClauses, MemoEnabled, MemoStrategy, LuaCode)
    ;   generate_generic_linear_recursion_lua(PredStr, Arity, BaseClauses, RecClauses, MemoEnabled, MemoStrategy, LuaCode)
    ).

generate_fold_based_recursion_lua(PredStr, BaseClauses, RecClauses, MemoEnabled, MemoStrategy, LuaCode) :-
    % Extract pattern info
    linear_recursion:extract_base_case_info(BaseClauses, BaseInput, BaseOutput),
    linear_recursion:detect_input_type(BaseInput, InputType),
    linear_recursion:extract_fold_operation(RecClauses, FoldExpr),

    (   InputType = numeric ->
        generate_numeric_fold_lua(PredStr, BaseInput, BaseOutput, FoldExpr, MemoEnabled, MemoStrategy, LuaCode)
    ;   InputType = list ->
        generate_list_fold_lua(PredStr, BaseInput, BaseOutput, FoldExpr, MemoEnabled, MemoStrategy, LuaCode)
    ;   generate_generic_linear_recursion_lua(PredStr, 2, BaseClauses, RecClauses, MemoEnabled, MemoStrategy, LuaCode)
    ).

generate_numeric_fold_lua(PredStr, BaseInput, BaseOutput, _FoldExpr, MemoEnabled, _MemoStrategy, LuaCode) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, 2),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),
    partition(linear_recursion:is_recursive_clause(Pred), Clauses, [clause(RHead, RBody)|_], _BaseClauses),
    
    % Find variable mapping
    RHead =.. [_Pred, InputVar, _OutputVar],
    linear_recursion:find_recursive_call(RBody, RecCall),
    RecCall =.. [_RecPred, _RecInput, AccVar],
    linear_recursion:find_last_is_expression(RBody, _ is ActualFoldExpr),
    
    linear_recursion:translate_fold_expr(ActualFoldExpr, InputVar, AccVar, LFoldOp),
    
    (   MemoEnabled = true ->
        format(string(MemoDecl), '-- Memoization table~nlocal _memo = {}~n', []),
        format(string(MemoCheckCode), '    local key = tostring(n)~n    if _memo[key] then~n        if expected ~~= nil then return _memo[key] == expected end~n        return _memo[key]~n    end~n', []),
        format(string(MemoStoreCode), '    _memo[key] = result~n', [])
    ;   MemoDecl = '-- Memoization disabled\n',
        MemoCheckCode = '',
        MemoStoreCode = ''
    ),

    format(string(LuaCode),
'-- ~w - fold-based linear recursion (numeric, Lua)
~w
local function ~w_op(current, acc)
    return ~w
end

local function ~w(n, expected)
~w
    if n == ~w then
        local result = ~w
~w
        if expected ~~= nil then return result == expected end
        return result
    end

    local result = 1 -- init value (TODO: dynamic init based on fold base case)
    for i = 1, n do
        -- Simulate range down or up depending on operation
        -- For factorial, it is 1..n
        result = ~w_op(i, result)
    end
~w
    if expected ~~= nil then return result == expected end
    return result
end
', [PredStr, MemoDecl, PredStr, LFoldOp, PredStr, MemoCheckCode, BaseInput, BaseOutput, MemoStoreCode, PredStr, MemoStoreCode]).

generate_list_fold_lua(PredStr, BaseInput, BaseOutput, _FoldExpr, MemoEnabled, _MemoStrategy, LuaCode) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, 2),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),
    partition(linear_recursion:is_recursive_clause(Pred), Clauses, [clause(RHead, RBody)|_], _BaseClauses),
    
    RHead =.. [_Pred, _InputVar, _OutputVar],
    linear_recursion:find_recursive_call(RBody, RecCall),
    RecCall =.. [_RecPred, _RecInput, AccVar],
    linear_recursion:find_last_is_expression(RBody, _ is ActualFoldExpr),
    
    linear_recursion:translate_fold_expr(ActualFoldExpr, _DummyInput, AccVar, LFoldOp),

    (   MemoEnabled = true ->
        format(string(MemoDecl), '-- Memoization table~nlocal _memo = {}~n', []),
        format(string(MemoCheckCode), '    local key = table.concat(lst, ",")~n    if _memo[key] then~n        if expected ~~= nil then return _memo[key] == expected end~n        return _memo[key]~n    end~n', []),
        format(string(MemoStoreCode), '    _memo[key] = result~n', [])
    ;   MemoDecl = '-- Memoization disabled\n',
        MemoCheckCode = '',
        MemoStoreCode = ''
    ),

    format(string(LuaCode),
'-- ~w - fold-based linear recursion (list, Lua)
~w
local function ~w_op(acc, current)
    return ~w
end

local function ~w(lst_input, expected)
    local lst = lst_input
    if type(lst_input) == "string" then
        lst = {}
        local clean = lst_input:gsub("%[", ""):gsub("%]", "")
        for val in clean:gmatch("[^,]+") do
            table.insert(lst, tonumber(val))
        end
    end

~w
    if #lst == 0 or (#lst == 1 and lst[1] == "~w") then
        local result = ~w
~w
        if expected ~~= nil then return result == expected end
        return result
    end

    local result = ~w
    for i = #lst, 1, -1 do
        result = ~w_op(result, lst[i])
    end
~w
    if expected ~~= nil then return result == expected end
    return result
end
', [PredStr, MemoDecl, PredStr, LFoldOp, PredStr, MemoCheckCode, BaseInput, BaseOutput, MemoStoreCode, BaseOutput, PredStr, MemoStoreCode]).

generate_generic_linear_recursion_lua(PredStr, Arity, _BaseClauses, _RecClauses, MemoEnabled, _MemoStrategy, LuaCode) :-
    (   MemoEnabled = true ->
        MemoDecl = 'local _memo = {}'
    ;   MemoDecl = '-- Memoization disabled'
    ),
    format(string(LuaCode),
'-- ~w/~w - generic linear recursion (Lua)
~w
local function ~w(...)
    print("Generic linear recursion not fully implemented in Lua")
    return nil
end
', [PredStr, Arity, MemoDecl, PredStr]).

:- multifile mutual_recursion:compile_mutual_pattern/5.
mutual_recursion:compile_mutual_pattern(lua, Predicates, _MemoEnabled, _MemoStrategy, LuaCode) :-
    format(string(LuaCode), '-- Mutually recursive group: ~w (Lua)\n-- Not fully implemented', [Predicates]).


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
