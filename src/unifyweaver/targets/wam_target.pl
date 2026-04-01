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
        compile_single_clause_wam(Clause, Options, ClausesCode)
    ;   compile_multi_clause_wam(Pred, Arity, Clauses, Options, ClausesCode)
    ),
    format(string(Code), "~w~n~w", [Label, ClausesCode]).

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
    ;   compound(Arg)
    ->  Arg =.. [F|SubArgs],
        length(SubArgs, Arity),
        format(string(Fst), "    get_structure ~w/~w, A~w", [F, Arity, I]),
        compile_unify_arguments(SubArgs, V0, V1, SubCode),
        format(string(Code), "~w~n~w", [Fst, SubCode])
    ).

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
    ;   % Nested structure
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
    (   Rest == []
    ->  % Last goal: execute (Tail Call Optimization)
        % For TCO with environments, put arguments BEFORE deallocate
        % so that Yi registers are still accessible.
        (   HasEnv == yes
        ->  Goal =.. [Pred|Args],
            length(Args, Arity),
            compile_put_arguments(Args, 1, V0, Vf, PutCode),
            format(string(ExecCode), "    execute ~w/~w", [Pred, Arity]),
            (   PutCode == ""
            ->  format(string(Code), "    deallocate~n~w", [ExecCode])
            ;   format(string(Code), "~w~n    deallocate~n~w", [PutCode, ExecCode])
            )
        ;   compile_goal_execute(Goal, V0, Vf, Code)
        )
    ;   compile_goal_call(Goal, V0, V1, GoalCode),
        compile_goals(Rest, V1, HasEnv, Vf, RestCode),
        format(string(Code), "~w~n~w", [GoalCode, RestCode])
    ).

compile_goal_call(Goal, V0, Vf, Code) :-
    Goal =.. [Pred|Args],
    length(Args, Arity),
    compile_put_arguments(Args, 1, V0, Vf, PutCode),
    format(string(CallCode), "    call ~w/~w, ~w", [Pred, Arity, Arity]),
    (   PutCode == ""
    ->  Code = CallCode
    ;   format(string(Code), "~w~n~w", [PutCode, CallCode])
    ).

compile_goal_execute(Goal, V0, Vf, Code) :-
    Goal =.. [Pred|Args],
    length(Args, Arity),
    compile_put_arguments(Args, 1, V0, Vf, PutCode),
    format(string(ExecCode), "    execute ~w/~w", [Pred, Arity]),
    (   PutCode == ""
    ->  Code = ExecCode
    ;   format(string(Code), "~w~n~w", [PutCode, ExecCode])
    ).

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
    ;   % Nested compound — allocate a register for the sub-structure
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

%% write_wam_program(+Code, +Filename)
write_wam_program(Code, Filename) :-
    setup_call_cleanup(
        open(Filename, write, Stream),
        format(Stream, "~w~n", [Code]),
        close(Stream)
    ).
