:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_target.pl - WAM (Warren Abstract Machine) Code Generation Target
% Compiles Prolog predicates to symbolic WAM instructions.
% This serves as a universal low-level fallback hub.

:- module(wam_target, [
    target_info/1,
    compile_predicate_to_wam/3,          % +PredIndicator, +Options, -WAMCode
    compile_predicate/3,                 % +PredIndicator, +Options, -WAMCode (dispatch alias)
    compile_facts_to_wam/3,              % +Pred, +Arity, -WAMCode
    compile_wam_module/3,                % +Predicates, +Options, -WAMCode
    write_wam_program/2,                 % +Code, +Filename
    init_wam_target/0                    % Initialize target
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module('../core/clause_body_analysis').
:- use_module('../core/template_system').

%% target_info(-Info)
target_info(info{
    name: "WAM",
    family: low_level,
    file_extension: ".wam",
    runtime: wam,
    features: [backtracking, unification, choice_points, environments, tail_call_optimization],
    recursion_patterns: [tail_recursion, linear_recursion, tree_recursion, mutual_recursion],
    compile_command: "wam_asm"
}).

%% init_wam_target
init_wam_target :-
    % Initialize any WAM-specific state or bindings if needed
    true.

%% compile_predicate/3 - dispatch alias for target_registry
compile_predicate(PredArity, Options, Code) :-
    compile_predicate_to_wam(PredArity, Options, Code).

%% compile_predicate_to_wam(+PredIndicator, +Options, -Code)
compile_predicate_to_wam(PredIndicator, Options, Code) :-
    % Handle module qualification
    (   PredIndicator = Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity -> option(module(Module), Options, user)
    ;   format(user_error, 'WAM target: invalid predicate indicator ~w~n', [PredIndicator]),
        fail
    ),
    functor(Head, Pred, Arity),
    % Find all clauses for the predicate in the specified module
    findall(Head-Body, clause(Module:Head, Body), Clauses),
    (   Clauses = []
    ->  format(user_error, 'WAM target: no clauses for ~w:~w/~w~n', [Module, Pred, Arity]),
        fail
    ;   compile_clauses_to_wam(Pred, Arity, Clauses, Options, Code)
    ).

%% compile_wam_module(+Predicates, +Options, -Code) is det.
%
%   Compiles a list of predicates to a single WAM module using templates.
compile_wam_module(Predicates, Options, Code) :-
    maplist({Options}/[PI, PredCode]>> (
        compile_predicate_to_wam(PI, Options, PredCode)
    ), Predicates, PredCodes),
    
    atomic_list_concat(PredCodes, '\n\n', AllPredsCode),
    
    option(module_name(ModuleName), Options, 'GeneratedWAM'),
    get_time(TimeStamp),
    format_time(string(Date), "%Y-%m-%d %H:%M:%S", TimeStamp),
    
    % template_system:render_named_template/3 takes [Key=Value] list
    TemplateData = [
        module_name=ModuleName,
        target_name="UnifyWeaver WAM",
        date=Date,
        predicates_code=AllPredsCode
    ],
    
    % Use named template from template_system
    render_named_template(wam_module, TemplateData, CodeAtom),
    atom_string(CodeAtom, Code).

%% compile_clauses_to_wam(+Pred, +Arity, +Clauses, +Options, -Code)
compile_clauses_to_wam(Pred, Arity, Clauses, Options, Code) :-
    format(string(Label), "~w/~w:", [Pred, Arity]),
    (   length(Clauses, 1)
    ->  Clauses = [Clause],
        compile_single_clause_wam(Clause, Options, ClausesCode0)
    ;   compile_multi_clause_wam(Pred, Arity, Clauses, Options, ClausesCode0)
    ),
    % Apply peephole optimization
    peephole_optimize(ClausesCode0, ClausesCode),
    % Generate argument index (try first arg, fall back to second arg)
    (   length(Clauses, NC), NC > 1,
        (   build_first_arg_index(Pred, Arity, Clauses, IndexCode)
        ;   Arity >= 2,
            build_second_arg_index(Pred, Arity, Clauses, IndexCode)
        )
    ->  format(string(Code), "~w~n~w~n~w", [Label, IndexCode, ClausesCode])
    ;   format(string(Code), "~w~n~w", [Label, ClausesCode])
    ).

%% build_first_arg_index(+Pred, +Arity, +Clauses, -IndexCode)
%  Analyzes first arguments of all clauses and emits indexing instructions.
%  - All atomic first args → switch_on_constant
%  - All compound first args → switch_on_structure
%  - Mixed types → switch_on_term (type-based dispatch)
%  - Any variable first args → no indexing (variable matches anything)
build_first_arg_index(Pred, Arity, Clauses, IndexCode) :-
    classify_first_args(Clauses, Types),
    \+ member(variable, Types),  % can't index if any clause has a variable first arg
    (   forall(member(T, Types), T = constant)
    ->  build_constant_index(Clauses, 1, Pred, Arity, Entries),
        Entries \= [],
        format_index_entries(Entries, EntriesStr),
        format(string(IndexCode), "    switch_on_constant ~w", [EntriesStr])
    ;   forall(member(T, Types), T = structure)
    ->  build_structure_index(Clauses, 1, Pred, Arity, Entries),
        Entries \= [],
        format_index_entries(Entries, EntriesStr),
        format(string(IndexCode), "    switch_on_structure ~w", [EntriesStr])
    ;   % Mixed — emit switch_on_term with type-based dispatch
        build_term_index(Clauses, 1, Pred, Arity, Types, ConstEntries, StructEntries),
        format_switch_on_term(ConstEntries, StructEntries, IndexCode)
    ).

classify_first_args([], []).
classify_first_args([Head-_|Rest], [Type|RestTypes]) :-
    Head =.. [_|[FirstArg|_]],
    (   var(FirstArg) -> Type = variable
    ;   atomic(FirstArg) -> Type = constant
    ;   is_list_term(FirstArg) -> Type = structure  % lists are './2' structures
    ;   compound(FirstArg) -> Type = structure
    ;   Type = variable
    ),
    classify_first_args(Rest, RestTypes).

build_constant_index([], _, _, _, []).
build_constant_index([Head-_|Rest], I, Pred, Arity, [FirstArg-Label|RestEntries]) :-
    Head =.. [_|[FirstArg|_]],
    (   I == 1 -> Label = default
    ;   format(atom(Label), "L_~w_~w_~w", [Pred, Arity, I])
    ),
    NextI is I + 1,
    build_constant_index(Rest, NextI, Pred, Arity, RestEntries).

build_structure_index([], _, _, _, []).
build_structure_index([Head-_|Rest], I, Pred, Arity, [FN-Label|RestEntries]) :-
    Head =.. [_|[FirstArg|_]],
    FirstArg =.. [F|SubArgs],
    length(SubArgs, SArity),
    format(atom(FN), "~w/~w", [F, SArity]),
    (   I == 1 -> Label = default
    ;   format(atom(Label), "L_~w_~w_~w", [Pred, Arity, I])
    ),
    NextI is I + 1,
    build_structure_index(Rest, NextI, Pred, Arity, RestEntries).

build_term_index([], _, _, _, _, [], []).
build_term_index([Head-_|Rest], I, Pred, Arity, [Type|RestTypes],
                 ConstEntries, StructEntries) :-
    Head =.. [_|[FirstArg|_]],
    (   I == 1 -> Label = default
    ;   format(atom(Label), "L_~w_~w_~w", [Pred, Arity, I])
    ),
    NextI is I + 1,
    build_term_index(Rest, NextI, Pred, Arity, RestTypes,
                     RestConst, RestStruct),
    (   Type = constant
    ->  ConstEntries = [FirstArg-Label|RestConst],
        StructEntries = RestStruct
    ;   Type = structure
    ->  FirstArg =.. [F|SubArgs],
        length(SubArgs, SArity),
        format(atom(FN), "~w/~w", [F, SArity]),
        ConstEntries = RestConst,
        StructEntries = [FN-Label|RestStruct]
    ;   ConstEntries = RestConst,
        StructEntries = RestStruct
    ).

format_switch_on_term(ConstEntries, StructEntries, IndexCode) :-
    (   ConstEntries \= []
    ->  format_index_entries(ConstEntries, CStr),
        ConstPart = CStr
    ;   ConstPart = "none"
    ),
    (   StructEntries \= []
    ->  format_index_entries(StructEntries, SStr),
        StructPart = SStr
    ;   StructPart = "none"
    ),
    format(string(IndexCode),
           "    switch_on_term constant:~w, structure:~w",
           [ConstPart, StructPart]).

%% build_second_arg_index(+Pred, +Arity, +Clauses, -IndexCode)
%  When first-arg indexing fails (e.g., all variable first args),
%  try indexing on the second argument instead.
build_second_arg_index(Pred, Arity, Clauses, IndexCode) :-
    classify_second_args(Clauses, Types),
    \+ member(variable, Types),
    forall(member(T, Types), T = constant),
    build_constant_index_on(Clauses, 2, 1, Pred, Arity, Entries),
    Entries \= [],
    format_index_entries(Entries, EntriesStr),
    format(string(IndexCode), "    switch_on_constant_a2 ~w", [EntriesStr]).

classify_second_args([], []).
classify_second_args([Head-_|Rest], [Type|RestTypes]) :-
    Head =.. [_|Args],
    (   length(Args, L), L >= 2, nth1(2, Args, SecondArg)
    ->  (   var(SecondArg) -> Type = variable
        ;   atomic(SecondArg) -> Type = constant
        ;   Type = variable
        )
    ;   Type = variable
    ),
    classify_second_args(Rest, RestTypes).

build_constant_index_on([], _, _, _, _, []).
build_constant_index_on([Head-_|Rest], ArgPos, I, Pred, Arity, [Val-Label|RestEntries]) :-
    Head =.. [_|Args],
    nth1(ArgPos, Args, Val),
    (   I == 1 -> Label = default
    ;   format(atom(Label), "L_~w_~w_~w", [Pred, Arity, I])
    ),
    NextI is I + 1,
    build_constant_index_on(Rest, ArgPos, NextI, Pred, Arity, RestEntries).

format_index_entries(Entries, Str) :-
    maplist([K-V, S]>>format(atom(S), "~w:~w", [K, V]), Entries, Parts),
    atomic_list_concat(Parts, ', ', Str).

%% compile_single_clause_wam(+Clause, +Options, -Code)
compile_single_clause_wam(Head-Body, Options, Code) :-
    Head =.. [_|Args],
    normalize_goals(Body, Goals),
    empty_varmap(V0),
    (   length(Goals, N), N > 1
    ->  % Pre-assign Yi registers, emit allocate before head so Yi
        % registers can be stored in the environment frame immediately.
        pre_assign_permanent_vars(Goals, V0, V0a),
        compile_head_arguments(Args, 1, V0a, V1, HeadCode),
        compile_goals(Goals, V1, yes, _, GoalsCode),
        format(string(Code), "    allocate~n~w~n~w", [HeadCode, GoalsCode])
    ;   compile_head_arguments(Args, 1, V0, V1, HeadCode),
        (   Goals == []
        ->  BodyCode = "    proceed"
        ;   compile_body_goals(Goals, V1, Options, BodyCode)
        ),
        format(string(Code), "~w~n~w", [HeadCode, BodyCode])
    ).

%% compile_multi_clause_wam(+Pred, +Arity, +Clauses, +Options, -Code)
compile_multi_clause_wam(Pred, Arity, Clauses, Options, Code) :-
    length(Clauses, N),
    compile_clauses_with_choice_points(Clauses, 1, N, Pred, Arity, Options, Code).

compile_clauses_with_choice_points([], _, _, _, _, _, "").
compile_clauses_with_choice_points([Head-Body|Rest], I, N, Pred, Arity, Options, Code) :-
    (   I == 1
    ->  format(string(Choice), "    try_me_else L_~w_~w_~w", [Pred, Arity, 2])
    ;   I == N
    ->  format(string(Choice), "L_~w_~w_~w:~n    trust_me", [Pred, Arity, I])
    ;   Next is I + 1,
        format(string(Choice), "L_~w_~w_~w:~n    retry_me_else L_~w_~w_~w", [Pred, Arity, I, Pred, Arity, Next])
    ),
    % Compile clause body — pre-assign Yi for permanent vars before head
    Head =.. [_|Args],
    normalize_goals(Body, Goals),
    empty_varmap(V0),
    (   length(Goals, NG), NG > 1
    ->  pre_assign_permanent_vars(Goals, V0, V0a),
        compile_head_arguments(Args, 1, V0a, V1, HeadCode0),
        compile_goals(Goals, V1, yes, _, GoalsCode),
        format(string(HeadCode), "    allocate~n~w", [HeadCode0]),
        BodyCode = GoalsCode
    ;   compile_head_arguments(Args, 1, V0, V1, HeadCode),
        (   Goals == []
        ->  BodyCode = "    proceed"
        ;   compile_body_goals(Goals, V1, Options, BodyCode)
        )
    ),
    NextI is I + 1,
    compile_clauses_with_choice_points(Rest, NextI, N, Pred, Arity, Options, RestCode),
    (   RestCode == ""
    ->  format(string(Code), "~w~n~w~n~w", [Choice, HeadCode, BodyCode])
    ;   format(string(Code), "~w~n~w~n~w~n~w", [Choice, HeadCode, BodyCode, RestCode])
    ).

%% compile_head_arguments(+Args, +ArgIndex, +VIn, -VOut, -Code)
compile_head_arguments([], _, V, V, "").
compile_head_arguments([Arg|Rest], I, V0, Vf, Code) :-
    compile_head_argument(Arg, I, V0, V1, ArgCode),
    NI is I + 1,
    compile_head_arguments(Rest, NI, V1, Vf, RestCode),
    (   RestCode == ""
    ->  Code = ArgCode
    ;   format(string(Code), "~w~n~w", [ArgCode, RestCode])
    ).

compile_head_argument(Arg, I, V0, V1, Code) :-
    (   var(Arg)
    ->  (   get_var_reg(Arg, V0, Reg)
        ->  format(string(Code), "    get_value ~w, A~w", [Reg, I]),
            V1 = V0
        ;   get_yi_alloc(Arg, V0, YReg, V1)
        ->  format(string(Code), "    get_variable ~w, A~w", [YReg, I])
        ;   next_x_reg(V0, XReg, V_temp),
            bind_var(Arg, XReg, V_temp, V1),
            format(string(Code), "    get_variable ~w, A~w", [XReg, I])
        )
    ;   atomic(Arg)
    ->  format(string(Code), "    get_constant ~w, A~w", [Arg, I]),
        V1 = V0
    ;   is_list_term(Arg)
    ->  Arg = [H|T],
        format(string(Fst), "    get_list A~w", [I]),
        compile_unify_arguments([H, T], V0, V1, SubCode),
        format(string(Code), "~w~n~w", [Fst, SubCode])
    ;   compound(Arg)
    ->  Arg =.. [F|SubArgs],
        length(SubArgs, Arity),
        format(string(Fst), "    get_structure ~w/~w, A~w", [F, Arity, I]),
        compile_unify_arguments(SubArgs, V0, V1, SubCode),
        format(string(Code), "~w~n~w", [Fst, SubCode])
    ).

%% is_list_term(+Term) — true if Term is a non-empty list cons cell [H|T].
is_list_term(Term) :- nonvar(Term), Term = [_|_].

compile_unify_arguments([], V, V, "").
compile_unify_arguments([Arg|Rest], V0, Vf, Code) :-
    (   var(Arg)
    ->  (   get_var_reg(Arg, V0, Reg)
        ->  format(string(ArgCode), "    unify_value ~w", [Reg]),
            V1 = V0
        ;   get_yi_alloc(Arg, V0, YReg, V1)
        ->  format(string(ArgCode), "    unify_variable ~w", [YReg])
        ;   next_x_reg(V0, XReg, V_temp),
            bind_var(Arg, XReg, V_temp, V1),
            format(string(ArgCode), "    unify_variable ~w", [XReg])
        )
    ;   atomic(Arg)
    ->  format(string(ArgCode), "    unify_constant ~w", [Arg]),
        V1 = V0
    ;   % Nested structure — emit unify_variable for a temp register,
        % then get_structure + unify_* for the nested sub-arguments.
        compound(Arg)
    ->  Arg =.. [F|NestedArgs],
        length(NestedArgs, NArity),
        next_x_reg(V0, XReg, V_temp),
        bind_var(Arg, XReg, V_temp, V1a),
        format(string(UnifyCode), "    unify_variable ~w", [XReg]),
        format(string(GetCode), "    get_structure ~w/~w, ~w", [F, NArity, XReg]),
        compile_unify_arguments(NestedArgs, V1a, V1, NestedCode),
        format(string(ArgCode), "~w~n~w~n~w", [UnifyCode, GetCode, NestedCode])
    ;   % Fallback
        next_x_reg(V0, XReg, V_temp),
        bind_var(Arg, XReg, V_temp, V1),
        format(string(ArgCode), "    unify_variable ~w", [XReg])
    ),
    compile_unify_arguments(Rest, V1, Vf, RestCode),
    (   RestCode == ""
    ->  Code = ArgCode
    ;   format(string(Code), "~w~n~w", [ArgCode, RestCode])
    ).

%% compile_body_goals(+Goals, +VarMap, +Options, -Code)
compile_body_goals(Goals, V, _Options, Code) :-
    length(Goals, N),
    (   N > 1
    ->  % allocate + Yi promotion handled by the clause compiler.
        compile_goals(Goals, V, yes, _, GoalsCode),
        format(string(Code), "    allocate~n~w", [GoalsCode])
    ;   compile_goals(Goals, V, no, _, Code)
    ).

%% pre_assign_permanent_vars(+Goals, +VarMapIn, -VarMapOut)
%  Identifies permanent variables and pre-assigns them Yi registers.
%  A variable is permanent if it is used in any non-first goal (i.e., it
%  must survive across at least one call instruction). This includes
%  head-bound variables referenced after the first call.
pre_assign_permanent_vars(Goals, vmap(Bindings, X), vmap(NewBindings, X)) :-
    collect_goal_vars(Goals, GoalVarSets),
    find_permanent_vars(GoalVarSets, PermVars),
    reassign_to_yi(Bindings, PermVars, 1, ReassignedBindings, NextY),
    pre_bind_unbound_yi(PermVars, ReassignedBindings, NextY, NewBindings).

collect_goal_vars([], []).
collect_goal_vars([Goal|Rest], [Vars|RestVars]) :-
    term_variables(Goal, Vars),
    collect_goal_vars(Rest, RestVars).

%% find_permanent_vars(+GoalVarSets, -PermVars)
%  A variable is permanent if it appears in any non-first goal, since the
%  first goal's call instruction may clobber Xi registers. This captures
%  both cross-goal variables and head-bound variables used after a call.
find_permanent_vars(GoalVarSets, PermVars) :-
    (   GoalVarSets = [_|RestGoalSets]
    ->  append(RestGoalSets, AllLaterVars),
        unique_vars(AllLaterVars, PermVars)
    ;   PermVars = []
    ).

unique_vars([], []).
unique_vars([V|Rest], Result) :-
    unique_vars(Rest, Acc),
    (   member(A, Acc), A == V
    ->  Result = Acc
    ;   Result = [V|Acc]
    ).

var_in_list(List, Var) :-
    member(V, List), V == Var, !.

union_vars([], Acc, Acc).
union_vars([V|Rest], Acc, Result) :-
    (   member(A, Acc), A == V
    ->  union_vars(Rest, Acc, Result)
    ;   union_vars(Rest, [V|Acc], Result)
    ).

%% reassign_to_yi(+Bindings, +PermVars, +YI, -NewBindings, -NextY)
%  Reassigns already-bound permanent variables from Xi to Yi.
reassign_to_yi([], _, YI, [], YI).
reassign_to_yi([b(Var, _Reg)|Rest], PermVars, YI, [b(Var, YReg)|NewRest], NextY) :-
    member(PV, PermVars), PV == Var, !,
    format(atom(YReg), "Y~w", [YI]),
    NYI is YI + 1,
    reassign_to_yi(Rest, PermVars, NYI, NewRest, NextY).
reassign_to_yi([B|Rest], PermVars, YI, [B|NewRest], NextY) :-
    reassign_to_yi(Rest, PermVars, YI, NewRest, NextY).

%% pre_bind_unbound_yi(+PermVars, +Bindings, +YI, -NewBindings)
%  For permanent variables not yet in the varmap, pre-allocate a Yi register
%  using y_alloc (not yet seen — will be promoted to b() on first use).
pre_bind_unbound_yi([], Bindings, _, Bindings).
pre_bind_unbound_yi([Var|Rest], Bindings, YI, NewBindings) :-
    (   (member(b(V, _), Bindings), V == Var ; member(y_alloc(V, _), Bindings), V == Var)
    ->  pre_bind_unbound_yi(Rest, Bindings, YI, NewBindings)
    ;   format(atom(YReg), "Y~w", [YI]),
        NYI is YI + 1,
        pre_bind_unbound_yi(Rest, [y_alloc(Var, YReg)|Bindings], NYI, NewBindings)
    ).

%% compile_goals(+Goals, +VarMap, +HasEnv, -Vf, -Code)
compile_goals([], V, _, V, "").
compile_goals([Goal|Rest], V0, HasEnv, Vf, Code) :-
    % Check for aggregate_all/findall first — these are always compiled inline
    (   Goal = aggregate_all(Template, InnerGoal, Result)
    ->  compile_aggregate_all(Template, InnerGoal, Result, V0, V1, GoalCode),
        (   Rest == []
        ->  Vf = V1,
            (   HasEnv == yes
            ->  format(string(Code), "~w~n    deallocate~n    proceed", [GoalCode])
            ;   format(string(Code), "~w~n    proceed", [GoalCode])
            )
        ;   compile_goals(Rest, V1, HasEnv, Vf, RestCode),
            format(string(Code), "~w~n~w", [GoalCode, RestCode])
        )
    ;   Goal = findall(Template, InnerGoal, Result)
    ->  compile_findall(Template, InnerGoal, Result, V0, V1, GoalCode),
        (   Rest == []
        ->  Vf = V1,
            (   HasEnv == yes
            ->  format(string(Code), "~w~n    deallocate~n    proceed", [GoalCode])
            ;   format(string(Code), "~w~n    proceed", [GoalCode])
            )
        ;   compile_goals(Rest, V1, HasEnv, Vf, RestCode),
            format(string(Code), "~w~n~w", [GoalCode, RestCode])
        )
    ;   Rest == []
    ->  % Last goal: execute (Tail Call Optimization)
        (   HasEnv == yes
        ->  Goal =.. [Pred|Args],
            length(Args, Arity),
            compile_put_arguments(Args, 1, V0, Vf, PutCode),
            (   is_builtin_pred(Pred, Arity)
            ->  format(string(ExecCode), "    builtin_call ~w/~w, ~w~n    proceed", [Pred, Arity, Arity])
            ;   format(string(ExecCode), "    execute ~w/~w", [Pred, Arity])
            ),
            (   PutCode == ""
            ->  format(string(Code), "    deallocate~n~w", [ExecCode])
            ;   format(string(Code), "~w~n    deallocate~n~w", [PutCode, ExecCode])
            )
        ;   compile_goal_execute(Goal, V0, Vf, Code)
        )
    ;   % Non-last goal: call
        compile_goal_call(Goal, V0, V1, GoalCode),
        compile_goals(Rest, V1, HasEnv, Vf, RestCode),
        format(string(Code), "~w~n~w", [GoalCode, RestCode])
    ).

%% compile_aggregate_all(+Template, +InnerGoal, +Result, +V0, -Vf, -Code)
%  Compile aggregate_all(Template, Goal, Result) to WAM instructions.
%  Emits: begin_aggregate, Goal body, end_aggregate
%  The WAM runtime handles solution collection and aggregation.
compile_aggregate_all(Template, InnerGoal, Result, V0, Vf, Code) :-
    % Determine aggregation type from Template
    (   Template = sum(ValueVar) -> AggType = sum
    ;   Template = count       -> AggType = count, ValueVar = 1
    ;   Template = max(ValueVar) -> AggType = max
    ;   Template = min(ValueVar) -> AggType = min
    ;   AggType = collect, ValueVar = Template  % default: collect all values
    ),
    % Find or allocate the Result register (where output goes)
    (   var(Result), get_var_reg(Result, V0, ResultReg0)
    ->  V1 = V0
    ;   allocate_var(Result, V0, V1, ResultReg0)
    ),
    % Compile the Value register (what gets collected per solution)
    (   var(ValueVar)
    ->  % ValueVar is a Prolog variable — allocate a Y-register for it
        allocate_var(ValueVar, V1, V2, ValueReg),
        % Emit put_variable to actually create the Y-register in the env frame
        format(string(InitValueCode), "    put_variable ~w, A1", [ValueReg])
    ;   % Constant value (e.g., count uses 1) — use A1 as placeholder
        ValueReg = 'A1', V2 = V1, InitValueCode = ""
    ),
    % Flatten the InnerGoal conjunction into a list of goals
    flatten_conjunction(InnerGoal, GoalList),
    % Compile each inner goal as a call (never TCO/execute) so control
    % returns to end_aggregate after each solution
    compile_inner_call_goals(GoalList, V2, Vf, InnerCode),
    (   InitValueCode \= ""
    ->  format(string(Code),
            "~w~n    begin_aggregate ~w, ~w, ~w~n~w~n    end_aggregate ~w",
            [InitValueCode, AggType, ValueReg, ResultReg0, InnerCode, ValueReg])
    ;   format(string(Code),
            "    begin_aggregate ~w, ~w, ~w~n~w~n    end_aggregate ~w",
            [AggType, ValueReg, ResultReg0, InnerCode, ValueReg])
    ).

%% compile_findall(+Template, +InnerGoal, +Result, +V0, -Vf, -Code)
compile_findall(Template, InnerGoal, Result, V0, Vf, Code) :-
    compile_aggregate_all(collect-Template, InnerGoal, Result, V0, Vf, Code).

%% flatten_conjunction(+Conj, -GoalList)
%  Flatten (A, B, C) into [A, B, C].
flatten_conjunction((A, B), Goals) :- !,
    flatten_conjunction(A, AG),
    flatten_conjunction(B, BG),
    append(AG, BG, Goals).
flatten_conjunction(Goal, [Goal]).

%% compile_inner_call_goals(+Goals, +V0, -Vf, -Code)
%  Compile all goals as calls (never execute/TCO) for use inside aggregate bodies.
compile_inner_call_goals([], V, V, "").
compile_inner_call_goals([Goal|Rest], V0, Vf, Code) :-
    compile_goal_call(Goal, V0, V1, GoalCode),
    compile_inner_call_goals(Rest, V1, Vf, RestCode),
    (   RestCode == ""
    ->  Code = GoalCode
    ;   format(string(Code), "~w~n~w", [GoalCode, RestCode])
    ).

%% allocate_var(+Var, +VarMapIn, -VarMapOut, -Register)
%  Allocate a Y-register for a variable if not already allocated.
allocate_var(Var, VIn, VOut, Reg) :-
    (   get_var_reg(Var, VIn, ExistingReg)
    ->  Reg = ExistingReg, VOut = VIn
    ;   get_yi_alloc(Var, VIn, Reg, VOut)
    ->  true
    ;   next_x_reg(VIn, XReg, V_temp),
        bind_var(Var, XReg, V_temp, VOut),
        Reg = XReg
    ).

compile_goal_call(Goal, V0, Vf, Code) :-
    Goal =.. [Pred|Args],
    length(Args, Arity),
    compile_put_arguments(Args, 1, V0, Vf, PutCode),
    (   is_builtin_pred(Pred, Arity)
    ->  format(string(CallCode), "    builtin_call ~w/~w, ~w", [Pred, Arity, Arity])
    ;   format(string(CallCode), "    call ~w/~w, ~w", [Pred, Arity, Arity])
    ),
    (   PutCode == ""
    ->  Code = CallCode
    ;   format(string(Code), "~w~n~w", [PutCode, CallCode])
    ).

compile_goal_execute(Goal, V0, Vf, Code) :-
    Goal =.. [Pred|Args],
    length(Args, Arity),
    compile_put_arguments(Args, 1, V0, Vf, PutCode),
    (   is_builtin_pred(Pred, Arity)
    ->  format(string(BuiltinCode), "    builtin_call ~w/~w, ~w", [Pred, Arity, Arity]),
        format(string(ExecCode), "~w~n    proceed", [BuiltinCode])
    ;   format(string(ExecCode), "    execute ~w/~w", [Pred, Arity])
    ),
    (   PutCode == ""
    ->  Code = ExecCode
    ;   format(string(Code), "~w~n~w", [PutCode, ExecCode])
    ).

%% is_builtin_pred(+Pred, +Arity)
%  Recognized built-in predicates that the WAM runtime handles directly.
%  Delegates to clause_body_analysis for guard/comparison detection,
%  with explicit entries for arithmetic and control builtins.
is_builtin_pred(Pred, Arity) :-
    % Build a goal term to test with is_guard_goal/2
    length(MockArgs, Arity),
    Goal =.. [Pred|MockArgs],
    is_guard_goal(Goal, []),  % empty varmap — we just need structural match
    !.
is_builtin_pred(is, 2).      % arithmetic evaluation (output goal, not a guard)
is_builtin_pred(true, 0).    % control
is_builtin_pred(fail, 0).
is_builtin_pred('!', 0).     % cut
is_builtin_pred(\+, 1).      % negation-as-failure
is_builtin_pred(member, 2).  % list operations
is_builtin_pred(append, 3).
is_builtin_pred(length, 2).
is_builtin_pred(functor, 3). % term inspection: name/arity read or construct
is_builtin_pred(arg, 3).     % term inspection: Nth argument access
is_builtin_pred((=..), 2).   % term inspection: univ (decompose/compose)
is_builtin_pred(copy_term, 2). % term inspection: fresh-variable copy

compile_put_arguments([], _, V, V, "").
compile_put_arguments([Arg|Rest], I, V0, Vf, Code) :-
    compile_put_argument(Arg, I, V0, V1, ArgCode),
    NI is I + 1,
    compile_put_arguments(Rest, NI, V1, Vf, RestCode),
    (   RestCode == ""
    ->  Code = ArgCode
    ;   format(string(Code), "~w~n~w", [ArgCode, RestCode])
    ).

compile_put_argument(Arg, I, V0, V1, Code) :-
    (   var(Arg)
    ->  (   get_var_reg(Arg, V0, Reg)
        ->  format(string(Code), "    put_value ~w, A~w", [Reg, I]),
            V1 = V0
        ;   get_yi_alloc(Arg, V0, YReg, V1)
        ->  format(string(Code), "    put_variable ~w, A~w", [YReg, I])
        ;   next_x_reg(V0, XReg, V_temp),
            bind_var(Arg, XReg, V_temp, V1),
            format(string(Code), "    put_variable ~w, A~w", [XReg, I])
        )
    ;   atomic(Arg)
    ->  format(string(Code), "    put_constant ~w, A~w", [Arg, I]),
        V1 = V0
    ;   is_list_term(Arg)
    ->  Arg = [H|T],
        next_x_reg(V0, XReg, V_temp),
        bind_var(Arg, XReg, V_temp, V2),
        format(string(ListCode), "    put_list A~w", [I]),
        compile_set_arguments([H, T], V2, V1, SetCode),
        (   SetCode == ""
        ->  Code = ListCode
        ;   format(string(Code), "~w~n~w", [ListCode, SetCode])
        )
    ;   compound(Arg)
    ->  Arg =.. [F|SubArgs],
        length(SubArgs, SArity),
        next_x_reg(V0, XReg, V_temp),
        bind_var(Arg, XReg, V_temp, V2),
        format(string(StructCode), "    put_structure ~w/~w, A~w", [F, SArity, I]),
        compile_set_arguments(SubArgs, V2, V1, SetCode),
        (   SetCode == ""
        ->  Code = StructCode
        ;   format(string(Code), "~w~n~w", [StructCode, SetCode])
        )
    ;   % Fallback for unknown terms — allocate a fresh variable
        next_x_reg(V0, XReg, V_temp),
        bind_var(Arg, XReg, V_temp, V1),
        format(string(Code), "    put_variable ~w, A~w", [XReg, I])
    ).

%% compile_set_arguments(+Args, +VIn, -VOut, -Code)
%  Emits set_value/set_variable instructions for put_structure sub-arguments.
compile_set_arguments([], V, V, "").
compile_set_arguments([Arg|Rest], V0, Vf, Code) :-
    (   var(Arg)
    ->  (   get_var_reg(Arg, V0, Reg)
        ->  format(string(ArgCode), "    set_value ~w", [Reg]),
            V1 = V0
        ;   get_yi_alloc(Arg, V0, YReg, V1)
        ->  format(string(ArgCode), "    set_variable ~w", [YReg])
        ;   next_x_reg(V0, XReg, V_temp),
            bind_var(Arg, XReg, V_temp, V1),
            format(string(ArgCode), "    set_variable ~w", [XReg])
        )
    ;   atomic(Arg)
    ->  % For atomic sub-args, emit set_constant directly
        V1 = V0,
        format(string(ArgCode), "    set_constant ~w", [Arg])
    ;   % Nested compound — recursively emit put_structure + set_* for sub-args
        compound(Arg)
    ->  Arg =.. [F|NestedArgs],
        length(NestedArgs, NArity),
        next_x_reg(V0, XReg, V_temp),
        bind_var(Arg, XReg, V_temp, V1a),
        format(string(SetCode), "    set_variable ~w", [XReg]),
        format(string(PutCode), "    put_structure ~w/~w, ~w", [F, NArity, XReg]),
        compile_set_arguments(NestedArgs, V1a, V1, NestedCode),
        (   NestedCode == ""
        ->  format(string(ArgCode), "~w~n~w", [SetCode, PutCode])
        ;   format(string(ArgCode), "~w~n~w~n~w", [SetCode, PutCode, NestedCode])
        )
    ;   % Fallback
        next_x_reg(V0, XReg, V_temp),
        bind_var(Arg, XReg, V_temp, V1),
        format(string(ArgCode), "    set_variable ~w", [XReg])
    ),
    compile_set_arguments(Rest, V1, Vf, RestCode),
    (   RestCode == ""
    ->  Code = ArgCode
    ;   format(string(Code), "~w~n~w", [ArgCode, RestCode])
    ).

%% Variable Mapping Helpers
%  Bindings use b(Var, Reg) for seen variables, and
%  y_alloc(Var, Reg) for pre-allocated Yi registers not yet seen.
empty_varmap(vmap([], 1)).

get_var_reg(Var, vmap(Bindings, _), Reg) :-
    member(b(V, Reg), Bindings),
    V == Var, !.

%% get_yi_alloc(+Var, +VarMap, -YReg, -VarMapOut)
%  If Var has a pre-allocated Yi register, return it and promote to seen.
get_yi_alloc(Var, vmap(Bindings, X), YReg, vmap(NewBindings, X)) :-
    select(y_alloc(V, YReg), Bindings, Rest),
    V == Var, !,
    NewBindings = [b(Var, YReg)|Rest].

bind_var(Var, Reg, vmap(Bs, X), vmap([b(Var, Reg)|Bs], X)).

next_x_reg(vmap(Bs, X), XReg, vmap(Bs, NX)) :-
    format(atom(XReg), "X~w", [X]),
    NX is X + 1.

%% compile_facts_to_wam(+Pred, +Arity, -Code)
compile_facts_to_wam(PredIndicator, Arity, Code) :-
    % Handle module qualification
    (   PredIndicator = Module:Pred -> true
    ;   PredIndicator = Pred, Module = user
    ),
    functor(Head, Pred, Arity),
    findall(Head-true, clause(Module:Head, true), Clauses),
    (   Clauses = []
    ->  format(user_error, 'WAM target: no facts for ~w:~w/~w~n', [Module, Pred, Arity]),
        fail
    ;   compile_clauses_to_wam(Pred, Arity, Clauses, [], Code)
    ).

%% =====================================================
%% Peephole Optimization
%% =====================================================

%% peephole_optimize(+CodeStr, -OptimizedStr)
%  Applies peephole optimizations to a WAM instruction string.
%  Operates on the string representation, line by line.
peephole_optimize(Code, Optimized) :-
    split_string(Code, "\n", "", Lines),
    peephole_lines(Lines, OptLines),
    atomic_list_concat(OptLines, '\n', Optimized).

peephole_lines([], []).
% Eliminate put_value Xn, Ai followed by get_variable Xn, Ai (identity)
peephole_lines([L1, L2|Rest], Result) :-
    normalize_ws(L1, N1),
    normalize_ws(L2, N2),
    atom_string(N1, S1), atom_string(N2, S2),
    match_put_get_identity(S1, S2), !,
    peephole_lines(Rest, Result).
% Eliminate put_value Xn, Ai immediately followed by put_value Xn, Ai (duplicate)
peephole_lines([L1, L2|Rest], [L1|Result]) :-
    normalize_ws(L1, N1),
    normalize_ws(L2, N2),
    N1 == N2,
    atom_string(N1, S1),
    sub_string(S1, 0, _, _, "put_"), !,
    peephole_lines(Rest, Result).
% Eliminate get_variable Xn, Ai followed by put_value Xn, Ai (pass-through)
% Only safe if Xn is not referenced by any later instruction.
peephole_lines([L1, L2|Rest], Result) :-
    normalize_ws(L1, N1),
    normalize_ws(L2, N2),
    atom_string(N1, S1), atom_string(N2, S2),
    match_get_put_passthrough(S1, S2, Reg),
    \+ reg_used_in_rest(Reg, Rest), !,
    peephole_lines(Rest, Result).
% Eliminate put_variable Xn, Ai followed by put_value Xn, Ai (same register, same arg)
peephole_lines([L1, L2|Rest], [L1|Result]) :-
    normalize_ws(L1, N1),
    normalize_ws(L2, N2),
    atom_string(N1, S1), atom_string(N2, S2),
    match_put_variable_put_value(S1, S2), !,
    peephole_lines(Rest, Result).
peephole_lines([L|Rest], [L|Result]) :-
    peephole_lines(Rest, Result).

normalize_ws(Str, Normalized) :-
    split_string(Str, " \t", " \t", Parts),
    delete(Parts, "", Clean),
    atomic_list_concat(Clean, ' ', Normalized).

%% match_put_get_identity(+PutStr, +GetStr)
%  put_value Xn/Yn, Ai followed by get_variable Xn/Yn, Ai — the get is redundant.
match_put_get_identity(Put, Get) :-
    split_string(Put, " ,", " ,", ["put_value", Reg, Ai]),
    split_string(Get, " ,", " ,", ["get_variable", Reg, Ai]).

%% match_get_put_passthrough(+GetStr, +PutStr, -Reg)
%  get_variable Xn, Ai followed by put_value Xn, Ai — both redundant
%  (the value is already in Ai and doesn't need round-tripping through Xn).
match_get_put_passthrough(Get, Put, Reg) :-
    split_string(Get, " ,", " ,", ["get_variable", Reg, Ai]),
    split_string(Put, " ,", " ,", ["put_value", Reg, Ai]).

%% match_put_variable_put_value(+PutVarStr, +PutValStr)
%  put_variable Xn/Yn, Ai followed by put_value Xn/Yn, Ai where both
%  use the same register and arg — the put_value is redundant.
match_put_variable_put_value(PutVar, PutVal) :-
    split_string(PutVar, " ,", " ,", ["put_variable", Reg, Ai]),
    split_string(PutVal, " ,", " ,", ["put_value", Reg, Ai]).

%% reg_used_in_rest(+Reg, +Lines)
%  True if the register name appears in any subsequent line.
reg_used_in_rest(Reg, Lines) :-
    member(Line, Lines),
    atom_string(Line, LineStr),
    sub_string(LineStr, _, _, _, Reg), !.

%% write_wam_program(+Code, +Filename)
write_wam_program(Code, Filename) :-
    setup_call_cleanup(
        open(Filename, write, Stream),
        format(Stream, "~w~n", [Code]),
        close(Stream)
    ).
