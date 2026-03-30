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
compile_predicate_to_wam(Pred/Arity, Options, Code) :-
    functor(Head, Pred, Arity),
    % Find all clauses for the predicate
    (   current_module(Module),
        findall(Head-Body, clause(Module:Head, Body), Clauses),
        Clauses \= []
    ->  true
    ;   findall(Head-Body, user:clause(Head, Body), Clauses)
    ),
    (   Clauses = []
    ->  format(user_error, 'WAM target: no clauses for ~w/~w~n', [Pred, Arity]),
        fail
    ;   compile_clauses_to_wam(Pred, Arity, Clauses, Options, Code)
    ).

%% compile_wam_module(+Predicates, +Options, -Code) is det.
%
%   Compiles a list of predicates to a single WAM module using templates.
compile_wam_module(Predicates, Options, Code) :-
    maplist({Options}/[P/A, PredCode]>> (
        compile_predicate_to_wam(P/A, Options, PredCode)
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
compile_single_clause_wam(Head-Body, _Options, Code) :-
    % Head unification
    Head =.. [_|Args],
    compile_head_arguments(Args, 1, HeadCode),
    % Body execution
    normalize_goals(Body, Goals),
    (   Goals == []
    ->  BodyCode = "    proceed"
    ;   compile_body_goals(Goals, BodyCode)
    ),
    format(string(Code), "~w~n~w", [HeadCode, BodyCode]).

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
    % Compile clause body
    Head =.. [_|Args],
    compile_head_arguments(Args, 1, HeadCode),
    normalize_goals(Body, Goals),
    (   Goals == []
    ->  BodyCode = "    proceed"
    ;   compile_body_goals(Goals, BodyCode)
    ),
    NextI is I + 1,
    compile_clauses_with_choice_points(Rest, NextI, N, Pred, Arity, Options, RestCode),
    (   RestCode == ""
    ->  format(string(Code), "~w~n~w~n~w", [Choice, HeadCode, BodyCode])
    ;   format(string(Code), "~w~n~w~n~w~n~w", [Choice, HeadCode, BodyCode, RestCode])
    ).

%% compile_head_arguments(+Args, +ArgIndex, -Code)
compile_head_arguments([], _, "").
compile_head_arguments([Arg|Rest], I, Code) :-
    compile_head_argument(Arg, I, ArgCode),
    NI is I + 1,
    compile_head_arguments(Rest, NI, RestCode),
    (   RestCode == ""
    ->  Code = ArgCode
    ;   format(string(Code), "~w~n~w", [ArgCode, RestCode])
    ).

compile_head_argument(Arg, I, Code) :-
    (   var(Arg)
    ->  format(string(Code), "    get_variable X~w, A~w", [I, I]) % Simplified mapping
    ;   atomic(Arg)
    ->  format(string(Code), "    get_constant ~w, A~w", [Arg, I])
    ;   compound(Arg)
    ->  Arg =.. [F|SubArgs],
        length(SubArgs, Arity),
        format(string(Fst), "    get_structure ~w/~w, A~w", [F, Arity, I]),
        compile_unify_arguments(SubArgs, SubCode),
        format(string(Code), "~w~n~w", [Fst, SubCode])
    ).

compile_unify_arguments([], "").
compile_unify_arguments([Arg|Rest], Code) :-
    (   var(Arg)
    ->  ArgCode = "    unify_variable X_" % Placeholder
    ;   atomic(Arg)
    ->  format(string(ArgCode), "    unify_constant ~w", [Arg])
    ;   ArgCode = "    unify_variable X_" % Simplified
    ),
    compile_unify_arguments(Rest, RestCode),
    (   RestCode == ""
    ->  Code = ArgCode
    ;   format(string(Code), "~w~n~w", [ArgCode, RestCode])
    ).

%% compile_body_goals(+Goals, -Code)
compile_body_goals(Goals, Code) :-
    % Need to handle environment allocation if permanent variables exist
    % For now, simplified version
    compile_goals(Goals, Code).

compile_goals([], "").
compile_goals([Goal|Rest], Code) :-
    (   Rest == []
    ->  % Last goal: execute (Tail Call Optimization)
        compile_goal_execute(Goal, Code)
    ;   compile_goal_call(Goal, GoalCode),
        compile_goals(Rest, RestCode),
        format(string(Code), "~w~n~w", [GoalCode, RestCode])
    ).

compile_goal_call(Goal, Code) :-
    Goal =.. [Pred|Args],
    length(Args, Arity),
    compile_put_arguments(Args, 1, PutCode),
    format(string(CallCode), "    call ~w/~w, ~w", [Pred, Arity, Arity]),
    (   PutCode == ""
    ->  Code = CallCode
    ;   format(string(Code), "~w~n~w", [PutCode, CallCode])
    ).

compile_goal_execute(Goal, Code) :-
    Goal =.. [Pred|Args],
    length(Args, Arity),
    compile_put_arguments(Args, 1, PutCode),
    format(string(ExecCode), "    execute ~w/~w", [Pred, Arity]),
    (   PutCode == ""
    ->  Code = ExecCode
    ;   format(string(Code), "~w~n~w", [PutCode, ExecCode])
    ).

compile_put_arguments([], _, "").
compile_put_arguments([Arg|Rest], I, Code) :-
    compile_put_argument(Arg, I, ArgCode),
    NI is I + 1,
    compile_put_arguments(Rest, NI, RestCode),
    (   RestCode == ""
    ->  Code = ArgCode
    ;   format(string(Code), "~w~n~w", [ArgCode, RestCode])
    ).

compile_put_argument(Arg, I, Code) :-
    (   var(Arg)
    ->  format(string(Code), "    put_variable X~w, A~w", [I, I])
    ;   atomic(Arg)
    ->  format(string(Code), "    put_constant ~w, A~w", [Arg, I])
    ;   format(string(Code), "    put_variable X~w, A~w", [I, I]) % Simplified
    ).

%% compile_facts_to_wam(+Pred, +Arity, -Code)
compile_facts_to_wam(Pred, Arity, Code) :-
    functor(Head, Pred, Arity),
    (   current_module(Module),
        findall(Head-true, clause(Module:Head, Body), Clauses),
        Body == true, % Ensure they are facts
        Clauses \= []
    ->  true
    ;   findall(Head-true, user:clause(Head, true), Clauses)
    ),
    compile_clauses_to_wam(Pred, Arity, Clauses, [], Code).

%% write_wam_program(+Code, +Filename)
write_wam_program(Code, Filename) :-
    setup_call_cleanup(
        open(Filename, write, Stream),
        format(Stream, "~w~n", [Code]),
        close(Stream)
    ).
