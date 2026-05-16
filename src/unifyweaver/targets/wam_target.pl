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
    init_wam_target/0,                   % Initialize target
    % WAM constant quoting — exported so downstream tokenizers can
    % share the same rules (see wam_elixir_lowered_emitter:tokenize_wam_line/2).
    quote_wam_constant/2                 % +Value, -QuotedString
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module('../core/clause_body_analysis').
:- use_module('../core/template_system').
:- use_module('../core/binding_state_analysis').

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
    nb_setval(wam_ite_counter, 0).

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
    (   option(inline_bagof_setof(true), Options)
    ->  b_setval(wam_inline_bagof_setof, true)
    ;   b_setval(wam_inline_bagof_setof, false)
    ),
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
    ;   forall(member(T, Types), T = structure),
        \+ first_args_contain_list(Clauses)
    ->  build_structure_index(Clauses, 1, Pred, Arity, Entries),
        Entries \= [],
        format_index_entries(Entries, EntriesStr),
        format(string(IndexCode), "    switch_on_structure ~w", [EntriesStr])
    ;   % Mixed — emit switch_on_term with type-based dispatch
        build_term_index(Clauses, 1, Pred, Arity, Types, ConstEntries, StructEntries, ListLabel),
        format_switch_on_term(ConstEntries, StructEntries, ListLabel, IndexCode)
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

first_args_contain_list([Head-_|_]) :-
    Head =.. [_|[FirstArg|_]],
    is_list_term(FirstArg), !.
first_args_contain_list([_|Rest]) :-
    first_args_contain_list(Rest).

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

build_term_index([], _, _, _, _, [], [], none).
build_term_index([Head-_|Rest], I, Pred, Arity, [Type|RestTypes],
                 ConstEntries, StructEntries, ListLabel) :-
    Head =.. [_|[FirstArg|_]],
    (   I == 1 -> Label = default
    ;   format(atom(Label), "L_~w_~w_~w", [Pred, Arity, I])
    ),
    NextI is I + 1,
    build_term_index(Rest, NextI, Pred, Arity, RestTypes,
                     RestConst, RestStruct, RestListLabel),
    (   Type = constant
    ->  ConstEntries = [FirstArg-Label|RestConst],
        StructEntries = RestStruct,
        ListLabel = RestListLabel
    ;   Type = structure,
        is_list_term(FirstArg)
    ->  ConstEntries = RestConst,
        StructEntries = RestStruct,
        ListLabel = Label
    ;   Type = structure
    ->  FirstArg =.. [F|SubArgs],
        length(SubArgs, SArity),
        format(atom(FN), "~w/~w", [F, SArity]),
        ConstEntries = RestConst,
        StructEntries = [FN-Label|RestStruct],
        ListLabel = RestListLabel
    ;   ConstEntries = RestConst,
        StructEntries = RestStruct,
        ListLabel = RestListLabel
    ).

format_switch_on_term(ConstEntries, StructEntries, ListLabel, IndexCode) :-
    length(ConstEntries, CLen),
    length(StructEntries, SLen),
    format_index_entries(ConstEntries, CStr),
    format_index_entries(StructEntries, SStr),
    format(string(IndexCode),
           "    switch_on_term ~w ~w ~w ~w ~w",
           [CLen, CStr, SLen, SStr, ListLabel]).

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
    maplist([K-V, S]>>(quote_wam_constant(K, KStr),
                       format(atom(S), "~w:~w", [KStr, V])),
            Entries, Parts),
    atomic_list_concat(Parts, ' ', Str).

% ---------------------------------------------------------------------------
% Constant quoting
% ---------------------------------------------------------------------------
%
% The symbolic WAM text uses ` `, `,`, and `\t` as token separators and
% `:` as the key/label separator inside `switch_on_constant` entries.
% Prolog atoms freely contain any of those characters, so a naive
% `~w` serialisation produces output the downstream line tokenizer
% cannot reparse (`get_constant Washington,_D.C., A1` splits on the
% embedded comma and emits a `raw/1` catch-all).
%
% `quote_wam_constant/2` wraps constants that need quoting in
% single quotes, escaping embedded `'` as `\'` and `\` as `\\`.
% Unambiguous unquoted atoms (identifier-like, numeric) pass through
% unchanged, keeping the common case readable.
%
% Downstream tokenizers (see `wam_elixir_lowered_emitter:tokenize_wam_line/2`)
% must recognise the same quote syntax.

%% quote_wam_constant(+Value, -QuotedString)
%  Value is an atom, string, or number. QuotedString is a string
%  suitable to embed after `get_constant`, `put_constant`, etc.
%
%  Atom-vs-number disambiguation: when an atom''s textual form would
%  re-parse as a number (`''5''`, `''42''`, `''-3.14''`), the emitted
%  text gets a `\\x01` marker INSIDE the quotes — e.g. `''\\x015''`.
%  After the tokenizer strips the outer quotes, the token is
%  `\\x015`. Constant-emitters that recognise the marker treat the
%  token as an atom (stripping the marker); ones that don''t still
%  produce a (visibly-weird-but-valid) atom name, which is better
%  than the previous silent-coercion-to-integer.
quote_wam_constant(Value, Quoted) :-
    (   number(Value)
    ->  format(string(Quoted), "~w", [Value])
    ;   ( atom(Value) -> atom_string(Value, Str) ; Str = Value ),
        (   atom_looks_like_number(Str)
        ->  % Quote with marker so the round-trip preserves Atom-ness
            % vs Number even when the textual form is numeric.
            escape_for_wam_quoting(Str, Escaped),
            format(string(Quoted), "'~w'", [Escaped])
        ;   constant_needs_quoting(Str)
        ->  escape_for_wam_quoting(Str, Escaped),
            format(string(Quoted), "'~w'", [Escaped])
        ;   Quoted = Str
        )
    ).

%% atom_looks_like_number(+Str) is semidet.
%  True iff Str (which is the textual form of an atom) would re-parse
%  as a number. Used by quote_wam_constant to decide whether to add
%  the atom-marker.
atom_looks_like_number(Str) :-
    catch(number_string(_, Str), _, fail).

constant_needs_quoting("") :- !.
constant_needs_quoting(Str) :-
    string_chars(Str, Chars),
    member(C, Chars),
    separator_or_special_char(C), !.

separator_or_special_char(' ').
separator_or_special_char('\t').
separator_or_special_char(',').
separator_or_special_char(':').
separator_or_special_char('\'').
separator_or_special_char('\\').

escape_for_wam_quoting(Str, Escaped) :-
    string_chars(Str, Chars),
    maplist(escape_wam_char, Chars, NestedChars),
    append(NestedChars, EscChars),
    string_chars(Escaped, EscChars).

escape_wam_char('\\', ['\\', '\\']).
escape_wam_char('\'', ['\\', '\'']).
escape_wam_char(C, [C]).

%% compile_single_clause_wam(+Clause, +Options, -Code)
compile_single_clause_wam(Head-Body, Options, Code) :-
    set_clause_binding_context(Head, Body),
    set_clause_visited_context(Head),
    Head =.. [_|Args],
    normalize_goals(Body, Goals),
    empty_varmap(V0),
    % Force permanent variable allocation when the body contains a
    % Call (including aggregate_all/findall whose inner goals contain
    % Calls). Without this, a single-goal clause like
    %   p(X,Y) :- aggregate_all(sum(W), q(X,…), Y).
    % assigns X/Y to X-registers (< 200) which get clobbered by the
    % inner Call to q.
    % Expand aggregates so pre_assign_permanent_vars sees the inner
    % goals. Variables shared between the head and the aggregate body
    % must be permanent (Y-registers) because the inner Call clobbers
    % X-registers.
    expand_aggregate_goals_for_perm_vars(Goals, ExpandedGoals),
    (   ( length(ExpandedGoals, N), N > 1
        ; goals_contain_call_or_aggregate(Goals)
        )
    ->  % Pre-assign Yi registers, emit allocate before head so Yi
        % registers can be stored in the environment frame immediately.
        pre_assign_permanent_vars(ExpandedGoals, V0, V0a),
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

%% goals_contain_call_or_aggregate(+Goals)
%  True if any goal in the list is a Call to a user predicate or an
%  aggregate_all/findall/bagof/setof that internally produces Call
%  instructions.
%  Used to force permanent variable allocation in single-goal clauses
%  that would otherwise skip it (since length(Goals) == 1).
goals_contain_call_or_aggregate(Goals) :-
    member(G, Goals),
    ( G = aggregate_all(_, _, _)
    ; G = findall(_, _, _)
    ; wam_inline_bagof_setof_enabled, G = bagof(_, _, _)
    ; wam_inline_bagof_setof_enabled, G = setof(_, _, _)
    ; callable(G), functor(G, F, _), \+ is_builtin_goal(F)
    ),
    !.

wam_inline_bagof_setof_enabled :-
    catch(b_getval(wam_inline_bagof_setof, true), _, fail).

is_builtin_goal(is).
is_builtin_goal(=).
is_builtin_goal(\=).
is_builtin_goal(>).
is_builtin_goal(<).
is_builtin_goal(>=).
is_builtin_goal(=<).
is_builtin_goal(=:=).
is_builtin_goal(=\=).
is_builtin_goal(true).
is_builtin_goal(fail).
is_builtin_goal(write).
is_builtin_goal(display).
is_builtin_goal(nl).
is_builtin_goal(format).
is_builtin_goal(member).
is_builtin_goal(length).
is_builtin_goal(is_list).
is_builtin_goal(append).
is_builtin_goal((\+)).

%% expand_aggregate_goals_for_perm_vars(+Goals, -Expanded)
%  For permanent-variable detection, expand aggregate_all/findall/
%  bagof/setof into their inner goals + a synthetic "result" goal. This ensures
%  variables shared between the clause head and the aggregate body
%  are detected as permanent (appear in >1 "goal").
expand_aggregate_goals_for_perm_vars([], []).
expand_aggregate_goals_for_perm_vars([G|Rest], Expanded) :-
    (   G = aggregate_all(_Template, InnerGoal, _Result)
    ->  flatten_conjunction(InnerGoal, InnerGoals),
        % Add the aggregate itself as a separate "goal" so that
        % variables appearing in the aggregate arguments AND in the
        % inner body register as permanent (cross-goal).
        append([G|InnerGoals], RestExpanded, Expanded)
    ;   G = findall(_Template, InnerGoal, _Result)
    ->  flatten_conjunction(InnerGoal, InnerGoals),
        append([G|InnerGoals], RestExpanded, Expanded)
    ;   wam_inline_bagof_setof_enabled,
        G = bagof(_Template, InnerGoal, _Result)
    ->  flatten_conjunction(InnerGoal, InnerGoals),
        append([G|InnerGoals], RestExpanded, Expanded)
    ;   wam_inline_bagof_setof_enabled,
        G = setof(_Template, InnerGoal, _Result)
    ->  flatten_conjunction(InnerGoal, InnerGoals),
        append([G|InnerGoals], RestExpanded, Expanded)
    ;   ( G = (_;_) ; G = (_->_) )
    ->  % ITE / soft-cut: expose branch goals so variables shared
        %  between the clause head and the ITE branches are detected as
        %  permanent (they appear in the "later" expanded goals).
        (   G = (If -> Then ; Else)
        ->  flatten_conjunction(If,   IfGoals),
            flatten_conjunction(Then, ThenGoals),
            flatten_conjunction(Else, ElseGoals),
            append(IfGoals, ThenGoals, IfThen),
            append(IfThen,  ElseGoals, BranchGoals)
        ;   G = (If -> Then)
        ->  flatten_conjunction(If,  IfGoals),
            flatten_conjunction(Then, ThenGoals),
            append(IfGoals, ThenGoals, BranchGoals)
        ;   G = (A ; B)
        ->  flatten_conjunction(A, AGls),
            flatten_conjunction(B, BGls),
            append(AGls, BGls, BranchGoals)
        ;   BranchGoals = [G]
        ),
        append([G|BranchGoals], RestExpanded, Expanded)
    ;   Expanded = [G|RestExpanded]
    ),
    expand_aggregate_goals_for_perm_vars(Rest, RestExpanded).

%% compile_multi_clause_wam(+Pred, +Arity, +Clauses, +Options, -Code)
compile_multi_clause_wam(Pred, Arity, Clauses, Options, Code) :-
    length(Clauses, N),
    % Build a list of per-clause fragment strings, then atomic_list_concat
    % once at the end. The previous nested format/3 recursion rebuilt the
    % growing accumulator at each step — O(N²) for fact-only predicates
    % with thousands of clauses (6009-clause category_parent/2 took ~85s
    % before; ~3s after).
    compile_clauses_fragments(Clauses, 1, N, Pred, Arity, Options, Fragments),
    atomic_list_concat(Fragments, '\n', Code).

compile_clauses_fragments([], _, _, _, _, _, []).
compile_clauses_fragments([Head-Body|Rest], I, N, Pred, Arity, Options,
                         [Choice, HeadCode, BodyCode | RestFragments]) :-
    (   I == 1
    ->  format(string(Choice), "    try_me_else L_~w_~w_~w", [Pred, Arity, 2])
    ;   I == N
    ->  format(string(Choice), "L_~w_~w_~w:~n    trust_me", [Pred, Arity, I])
    ;   Next is I + 1,
        format(string(Choice), "L_~w_~w_~w:~n    retry_me_else L_~w_~w_~w", [Pred, Arity, I, Pred, Arity, Next])
    ),
    % Compile clause body — pre-assign Yi for permanent vars before head
    set_clause_binding_context(Head, Body),
    set_clause_visited_context(Head),
    Head =.. [_|Args],
    normalize_goals(Body, Goals),
    empty_varmap(V0),
    % Force permanent variable allocation when the body contains a Call
    % or aggregate (findall/aggregate_all), or a Cut (!/0), even for
    % single-goal bodies.
    % Without an env frame, the post-aggregate continuation has nowhere
    % to retrieve the caller's saved cp from, so finalise_aggregate's
    % update_topmost_agg_cp + restored.cp.(restored) chain loops back
    % into k2 forever. Also, Cut needs an environment frame to access
    % the cut barrier in some targets (like LLVM).
    % The single-clause path applies the same condition at line ~327;
    % this keeps multi-clause in sync.
    expand_aggregate_goals_for_perm_vars(Goals, ExpandedGoals),
    (   ( length(ExpandedGoals, NG), NG > 1
        ; goals_contain_call_or_aggregate(Goals)
        ; member(builtin_call('!/0', _), Goals)
        )
    ->  pre_assign_permanent_vars(ExpandedGoals, V0, V0a),
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
    compile_clauses_fragments(Rest, NextI, N, Pred, Arity, Options, RestFragments).

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
    ->  quote_wam_constant(Arg, ArgStr),
        format(string(Code), "    get_constant ~w, A~w", [ArgStr, I]),
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
    ->  quote_wam_constant(Arg, ArgStr),
        format(string(ArgCode), "    unify_constant ~w", [ArgStr]),
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
pre_assign_permanent_vars(Goals, vmap(Bindings, X), vmap(NewBindings, XOut)) :-
    collect_goal_vars(Goals, GoalVarSets),
    find_permanent_vars(GoalVarSets, PermVars),
    reassign_to_yi(Bindings, PermVars, 1, ReassignedBindings, NextY),
    pre_bind_unbound_yi(PermVars, ReassignedBindings, NextY, NewBindings),
    %% Bump the X register counter past all allocated Yi indices.
    %% Yi and Xi share the same register-file range (both map to
    %% N + 31 in reg_name_to_index), so a subsequent next_x_reg
    %% must not hand out an Xi that collides with an allocated Yi.
    max_yi_index(NewBindings, 0, MaxY),
    XOut is max(X, MaxY + 1).

%% max_yi_index(+Bindings, +Acc, -Max)
%  Finds the highest Y-register number among b(_, Yi) and y_alloc(_, Yi)
%  entries. Returns 0 if there are no Y registers.
max_yi_index([], Max, Max).
max_yi_index([Entry|Rest], Acc, Max) :-
    (   ( Entry = b(_, Reg) ; Entry = y_alloc(_, Reg) ),
        atom_string(Reg, S),
        string_codes(S, [0'Y|Digits]),
        number_codes(N, Digits)
    ->  NewAcc is max(Acc, N)
    ;   NewAcc = Acc
    ),
    max_yi_index(Rest, NewAcc, Max).

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
    % Check for aggregate_all/findall/bagof/setof first — these are
    % always compiled inline.
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
    ;   wam_inline_bagof_setof_enabled,
        Goal = bagof(Template, InnerGoal, Result)
    ->  compile_bagof(Template, InnerGoal, Result, V0, V1, GoalCode),
        (   Rest == []
        ->  Vf = V1,
            (   HasEnv == yes
            ->  format(string(Code), "~w~n    deallocate~n    proceed", [GoalCode])
            ;   format(string(Code), "~w~n    proceed", [GoalCode])
            )
        ;   compile_goals(Rest, V1, HasEnv, Vf, RestCode),
            format(string(Code), "~w~n~w", [GoalCode, RestCode])
        )
    ;   wam_inline_bagof_setof_enabled,
        Goal = setof(Template, InnerGoal, Result)
    ->  compile_setof(Template, InnerGoal, Result, V0, V1, GoalCode),
        (   Rest == []
        ->  Vf = V1,
            (   HasEnv == yes
            ->  format(string(Code), "~w~n    deallocate~n    proceed", [GoalCode])
            ;   format(string(Code), "~w~n    proceed", [GoalCode])
            )
        ;   compile_goals(Rest, V1, HasEnv, Vf, RestCode),
            format(string(Code), "~w~n~w", [GoalCode, RestCode])
        )
    % If-then-else: (Cond -> Then ; Else)
    ;   Goal = (CondGoal -> ThenGoal ; ElseGoal)
    ->  compile_if_then_else(CondGoal, ThenGoal, ElseGoal, V0, V1, HasEnv, GoalCode),
        (   Rest == []
        ->  Vf = V1,
            (   HasEnv == yes
            ->  format(string(Code), "~w~n    deallocate~n    proceed", [GoalCode])
            ;   format(string(Code), "~w~n    proceed", [GoalCode])
            )
        ;   compile_goals(Rest, V1, HasEnv, Vf, RestCode),
            format(string(Code), "~w~n~w", [GoalCode, RestCode])
        )
    % Bare if-then: (Cond -> Then) without an Else clause. Reuses the
    % if-then-else compiler with Else=fail — semantically identical
    % for the success path; Cond-failure just falls through to fail
    % (which is what bare ->/2 does).
    ;   Goal = (CondGoal -> ThenGoal)
    ->  compile_if_then_else(CondGoal, ThenGoal, fail, V0, V1, HasEnv, GoalCode),
        (   Rest == []
        ->  Vf = V1,
            (   HasEnv == yes
            ->  format(string(Code), "~w~n    deallocate~n    proceed", [GoalCode])
            ;   format(string(Code), "~w~n    proceed", [GoalCode])
            )
        ;   compile_goals(Rest, V1, HasEnv, Vf, RestCode),
            format(string(Code), "~w~n~w", [GoalCode, RestCode])
        )
    % Bare disjunction: (A ; B) without ->
    ;   Goal = (LeftGoal ; RightGoal),
        \+ (LeftGoal = (_ -> _))
    ->  compile_disjunction(LeftGoal, RightGoal, V0, V1, HasEnv, GoalCode),
        (   Rest == []
        ->  Vf = V1,
            (   HasEnv == yes
            ->  format(string(Code), "~w~n    deallocate~n    proceed", [GoalCode])
            ;   format(string(Code), "~w~n    proceed", [GoalCode])
            )
        ;   compile_goals(Rest, V1, HasEnv, Vf, RestCode),
            format(string(Code), "~w~n~w", [GoalCode, RestCode])
        )
    % once(G) — succeed once with G''s first solution, fail if G has
    % none. Desugars to (G -> true): if-then-else with Else=fail
    % already gives the right semantics (bare ->/2 fails on Cond
    % failure). Recurse so the rewritten goal flows through the
    % normal dispatch (including if-then-else inlining).
    ;   Goal = once(OnceGoal)
    ->  compile_goals([(OnceGoal -> true) | Rest], V0, HasEnv, Vf, Code)
    % forall(G, T) — for every solution of G, T must succeed.
    % Desugars to \+ (G, \+ T): negation-as-failure over the
    % conjunction of generator + negated test. Recursion routes
    % through compile_goal_call, which emits `call \+/1, 1` and the
    % runtime handles negation via the builtin path.
    ;   Goal = forall(GenGoal, TestGoal)
    ->  compile_goals([\+ (GenGoal, \+ TestGoal) | Rest],
                      V0, HasEnv, Vf, Code)
    ;   Rest == []
    ->  % Last goal: execute (Tail Call Optimization)
        (   %% Term-construction builtins (=../2 and functor/3) compose-mode
            %% lowering routes through compile_goal_execute which dispatches
            %% to emit_put_structure_dyn_lowering when the binding-state
            %% preconditions hold. We add the deallocate here (HasEnv == yes
            %% case) so the resulting tail-call stays TCO-correct.
            is_term_construction_goal(Goal),
            HasEnv == yes
        ->  compile_goal_execute(Goal, V0, Vf, ExecCode),
            format(string(Code), "    deallocate~n~w", [ExecCode])
        ;   is_term_construction_goal(Goal)
        ->  compile_goal_execute(Goal, V0, Vf, Code)
        ;   %% Visited-set lowering (Layer 2.5) routes through
            %% compile_goal_execute where the rewrite clause fires.
            %% Without this the inline put_arguments path below would
            %% bypass the rewrite for TCO-position goals.
            goal_has_visited_set_arg(Goal),
            HasEnv == yes
        ->  compile_goal_execute(Goal, V0, Vf, ExecCode),
            format(string(Code), "    deallocate~n~w", [ExecCode])
        ;   goal_has_visited_set_arg(Goal)
        ->  compile_goal_execute(Goal, V0, Vf, Code)
        ;   HasEnv == yes
        ->  Goal =.. [Pred|Args],
            length(Args, Arity),
            compile_put_arguments(Args, 1, V0, Vf, PutCode),
            (   is_builtin_pred(Pred, Arity)
            ->  format(string(ExecCode), "    builtin_call ~w/~w, ~w", [Pred, Arity, Arity]),
                (   PutCode == ""
                ->  format(string(Code), "~w~n    deallocate~n    proceed", [ExecCode])
                ;   format(string(Code), "~w~n~w~n    deallocate~n    proceed", [PutCode, ExecCode])
                )
            ;   format(string(ExecCode), "    execute ~w/~w", [Pred, Arity]),
                (   PutCode == ""
                ->  format(string(Code), "    deallocate~n~w", [ExecCode])
                ;   format(string(Code), "~w~n    deallocate~n~w", [PutCode, ExecCode])
                )
            )
        ;   compile_goal_execute(Goal, V0, Vf, Code)
        )
    ;   % Non-last goal: call
        compile_goal_call(Goal, V0, V1, GoalCode),
        advance_clause_goal_idx,
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
    ;   Template = bag(ValueVar) -> AggType = bag
    ;   Template = set(ValueVar) -> AggType = set
    ;   Template = bagof-BagVar -> AggType = bagof, ValueVar = BagVar
    ;   Template = setof-SetVar -> AggType = setof, ValueVar = SetVar
    % findall/3 → compile_findall/5 wraps Template as `collect-Template`.
    % Unwrap to expose the real ValueVar so the var(ValueVar) branch
    % below allocates a Y-register and emits put_variable Y_n, A1.
    % Without this, ValueReg defaults to A1 and survives the inner
    % call only when no instruction along the inner-goal path
    % overwrites A1 — false in the module-qualified case where the
    % `:/2` builtin lowering puts the module-name string into A1
    % (Phase 3 finding from #1647). Y-reg version is preserved
    % across any inner-call register churn.
    ;   Template = collect-CollectVar -> AggType = collect, ValueVar = CollectVar
    ;   AggType = collect, ValueVar = Template  % default: direct callers
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
        % Emit put_variable to actually create the Y-register in the env
        % frame. Use the SELF-INIT form (put_variable Y_n, Y_n) rather
        % than (put_variable Y_n, A1): the latter would share a fresh
        % unbound between Y_n and A1, and if the inner goal's first
        % arg is a constructed compound (e.g.
        % `findall(X, fact_in_range(p/2, ..., [X,_]), L)` where A1
        % holds `p/2`), put_structure_'s auto-bind in append_build_arg
        % would bind the shared unbound to the new struct -- silently
        % capturing the inner goal's first arg into the template var.
        % Self-init avoids the A1 share entirely; the inner goal's
        % compilation emits `put_value Y_n, A_k` when the template
        % var IS a direct call arg, so cases that don't have the
        % struct-first-arg conflict are unaffected.
        format(string(InitValueCode), "    put_variable ~w, ~w",
               [ValueReg, ValueReg]),
        ConstructionCode = ""
    ;   compound(ValueVar)
    ->  % Compound Template — `findall(p(X, Y), Goal, L)` and similar.
        % Allocate Y-regs for each variable arg, init via put_variable so
        % each iteration starts with fresh refs (the inner goal binds
        % them through head unification). After the inner returns, build
        % the Template structure on the heap via put_structure +
        % set_value/set_constant. ValueReg = A1 holds the heap ref so
        % end_aggregate captures it. aggregate_collect on the Elixir
        % runtime side deep-copies the heap structure into a self-
        % contained value before backtrack rewinds the heap.
        compile_compound_template(ValueVar, V1, V2, InitValueCode, ConstructionCode),
        ValueReg = 'A1'
    ;   % Constant value (e.g., count uses 1) — use A1 as placeholder
        ValueReg = 'A1', V2 = V1, InitValueCode = "", ConstructionCode = ""
    ),
    % Flatten the InnerGoal conjunction into a list of goals
    flatten_conjunction(InnerGoal, GoalList),
    % Compile each inner goal as a call (never TCO/execute) so control
    % returns to end_aggregate after each solution.
    % Note: permanent variable allocation for inner-goal variables that
    % survive across Calls is handled by the OUTER clause''s
    % pre_assign_permanent_vars (via expand_aggregate_goals_for_perm_vars
    % in compile_single_clause_wam). Do NOT add a second
    % pre_assign_permanent_vars call here — it would reassign variables
    % that already have Y-register slots from the outer pass.
    compile_inner_call_goals(GoalList, V2, Vf, InnerCode),
    % For compound Templates, the construction code runs AFTER the
    % inner-goal call but BEFORE end_aggregate captures.
    (   ConstructionCode == ""
    ->  FullInnerCode = InnerCode
    ;   format(string(FullInnerCode), "~w~n~w", [InnerCode, ConstructionCode])
    ),
    % For bagof/setof: compute free witnesses (vars in InnerGoal NOT
    % in Template and NOT under ^/2) and emit their registers as a
    % 4th `begin_aggregate` arg. Runtimes that recognise the 4-arg
    % form use the witness regs for ISO grouping. The 3-arg form
    % stays untouched for findall/count/sum/min/max/bag/set —
    % grouping doesn''t apply there.
    % Use Vf (post-inner-compile varmap) so witness vars allocated
    % during inner-goal compilation have register slots available
    % for lookup.
    aggregate_witness_clause(AggType, ValueVar, InnerGoal, Vf,
                             WitnessRegsClause),
    (   InitValueCode \= ""
    ->  format(string(Code),
            "~w~n    begin_aggregate ~w, ~w, ~w~w~n~w~n    end_aggregate ~w",
            [InitValueCode, AggType, ValueReg, ResultReg0,
             WitnessRegsClause, FullInnerCode, ValueReg])
    ;   format(string(Code),
            "    begin_aggregate ~w, ~w, ~w~w~n~w~n    end_aggregate ~w",
            [AggType, ValueReg, ResultReg0, WitnessRegsClause,
             FullInnerCode, ValueReg])
    ).

%% aggregate_witness_clause(+AggType, +ValueVar, +InnerGoal, +V, -Clause)
%  Build the trailing ", 'W1;W2;...'" string for begin_aggregate
%  when AggType is bagof or setof. Returns "" for other kinds (so
%  the existing 3-arg shape stays).
aggregate_witness_clause(AggType, ValueVar, InnerGoal, V, Clause) :-
    (   ( AggType == bagof ; AggType == setof )
    ->  find_free_witnesses(ValueVar, InnerGoal, Witnesses),
        witness_var_regs(Witnesses, V, WitnessRegs),
        atomic_list_concat(WitnessRegs, ';', WitnessStr),
        format(string(Clause), ", '~w'", [WitnessStr])
    ;   Clause = ""
    ).

%% find_free_witnesses(+Template, +InnerGoal, -Witnesses)
%  Vars in InnerGoal not in Template and not under ^/2.
find_free_witnesses(Template, InnerGoal, Witnesses) :-
    term_variables(Template, TemplateVars),
    free_witness_walk(InnerGoal, TemplateVars, [], WitnessesRev),
    list_to_set_var(WitnessesRev, Witnesses).

free_witness_walk(Var, Exclude, Acc, Out) :-
    var(Var), !,
    (   memberchk_var(Var, Exclude)
    ->  Out = Acc
    ;   Out = [Var|Acc]
    ).
free_witness_walk(Atomic, _, Acc, Acc) :- atomic(Atomic), !.
free_witness_walk(LHS^RHS, Exclude, Acc, Out) :- !,
    term_variables(LHS, LHSVars),
    append(LHSVars, Exclude, Exclude2),
    free_witness_walk(RHS, Exclude2, Acc, Out).
% Nested aggregate-style binders introduce their own scope: vars in
% the inner Template (and the inner Result) are local to the inner
% aggregate and don''t escape as witnesses of the outer. Walk only
% the inner Goal with those vars added to the exclude set.
free_witness_walk(bagof(T, G, R), Exclude, Acc, Out) :- !,
    nested_aggregate_walk(T, G, R, Exclude, Acc, Out).
free_witness_walk(setof(T, G, R), Exclude, Acc, Out) :- !,
    nested_aggregate_walk(T, G, R, Exclude, Acc, Out).
free_witness_walk(findall(T, G, R), Exclude, Acc, Out) :- !,
    nested_aggregate_walk(T, G, R, Exclude, Acc, Out).
free_witness_walk(aggregate_all(T, G, R), Exclude, Acc, Out) :- !,
    nested_aggregate_walk(T, G, R, Exclude, Acc, Out).
free_witness_walk(Compound, Exclude, Acc, Out) :-
    Compound =.. [_|Args],
    free_witness_walk_list(Args, Exclude, Acc, Out).

nested_aggregate_walk(T, G, R, Exclude, Acc, Out) :-
    term_variables(T, TVars),
    term_variables(R, RVars),
    append(TVars, RVars, Local0),
    append(Local0, Exclude, Exclude2),
    free_witness_walk(G, Exclude2, Acc, Out).

free_witness_walk_list([], _, Acc, Acc).
free_witness_walk_list([G|Gs], Exclude, Acc, Out) :-
    free_witness_walk(G, Exclude, Acc, Acc2),
    free_witness_walk_list(Gs, Exclude, Acc2, Out).

memberchk_var(V, [X|_]) :- V == X, !.
memberchk_var(V, [_|Rest]) :- memberchk_var(V, Rest).

list_to_set_var([], []).
list_to_set_var([V|Rest], [V|Out]) :-
    \+ memberchk_var(V, Rest), !,
    list_to_set_var(Rest, Out).
list_to_set_var([_|Rest], Out) :-
    list_to_set_var(Rest, Out).

%% witness_var_regs(+Vars, +Varmap, -Regs)
%  Look up each witness var''s register name in Varmap. If a var
%  doesn''t have a register yet (rare — would mean it''s an
%  unallocated singleton), it''s skipped silently.
witness_var_regs([], _, []).
witness_var_regs([V|Vs], Varmap, [Reg|Rest]) :-
    catch(get_var_reg(V, Varmap, Reg), _, fail), !,
    witness_var_regs(Vs, Varmap, Rest).
witness_var_regs([_|Vs], Varmap, Rest) :-
    witness_var_regs(Vs, Varmap, Rest).

%% compile_compound_template(+Template, +V0, -Vf, -InitCode, -ConstructionCode)
%  For findall(Functor(Arg1, Arg2, ...), Goal, L) with compound Template:
%  - Each Arg is either a variable or a constant
%  - Variables get Y-regs allocated (via allocate_var) and initialized
%    directly so they have stable slots across the inner-goal call.
%    Do not initialize through A-registers: those are scratch argument
%    registers for the inner goal, and aliasing a template variable
%    through A2 can bind it when the inner goal builds its own A2 term
%    (for example findall(Y-L, bagof(X, p(X,Y), L), Groups)).
%  - ConstructionCode emits put_structure + set_value/set_constant
%    after the inner goal returns; A1 holds the heap ref to the
%    constructed Template
%  - Compound args within the Template (e.g., `findall(p(f(X)), ...)`)
%    are not yet supported — would need recursive heap construction.
compile_compound_template(Template, V0, Vf, InitCode, ConstructionCode) :-
    Template =.. [Functor | Args],
    length(Args, Arity),
    compile_template_arg_init(Args, 1, V0, V1, InitLines),
    (   InitLines == []
    ->  InitCode = ""
    ;   atomic_list_concat(InitLines, '\n', InitCode)
    ),
    format(string(PutStrucCode), "    put_structure ~w/~w, A1", [Functor, Arity]),
    compile_template_arg_set(Args, V1, Vf, SetLines),
    atomic_list_concat([PutStrucCode | SetLines], '\n', ConstructionCode).

compile_template_arg_init([], _, V, V, []).
compile_template_arg_init([Arg | Rest], I, V0, Vf, Lines) :-
    (   var(Arg)
    ->  allocate_var(Arg, V0, V1, Reg),
        format(string(Line), "    put_variable ~w, ~w", [Reg, Reg]),
        Lines = [Line | RestLines]
    ;   atomic(Arg)
    ->  V1 = V0,
        Lines = RestLines
    ;   % Compound or other — not yet supported
        throw(error(unsupported_template_arg(Arg), compile_compound_template/5))
    ),
    NI is I + 1,
    compile_template_arg_init(Rest, NI, V1, Vf, RestLines).

compile_template_arg_set([], V, V, []).
compile_template_arg_set([Arg | Rest], V0, Vf, [Line | Lines]) :-
    (   var(Arg)
    ->  get_var_reg(Arg, V0, Reg),
        format(string(Line), "    set_value ~w", [Reg]),
        V1 = V0
    ;   atomic(Arg)
    ->  V1 = V0,
        quote_wam_constant(Arg, ArgStr),
        format(string(Line), "    set_constant ~w", [ArgStr])
    ;   throw(error(unsupported_template_arg(Arg), compile_compound_template/5))
    ),
    compile_template_arg_set(Rest, V1, Vf, Lines).

%% compile_findall(+Template, +InnerGoal, +Result, +V0, -Vf, -Code)
compile_findall(Template, InnerGoal, Result, V0, Vf, Code) :-
    compile_aggregate_all(collect-Template, InnerGoal, Result, V0, Vf, Code).

%% compile_bagof(+Template, +InnerGoal, +Result, +V0, -Vf, -Code)
%  Compile the no-witness-group subset of bagof/3. The runtime fails
%  empty bags, unlike findall/3.
compile_bagof(Template, InnerGoal, Result, V0, Vf, Code) :-
    compile_aggregate_all(bagof-Template, InnerGoal, Result, V0, Vf, Code).

%% compile_setof(+Template, +InnerGoal, +Result, +V0, -Vf, -Code)
%  Compile the no-witness-group subset of setof/3. The runtime fails
%  empty sets, then sorts/deduplicates collected values.
compile_setof(Template, InnerGoal, Result, V0, Vf, Code) :-
    compile_aggregate_all(setof-Template, InnerGoal, Result, V0, Vf, Code).

%% next_ite_label(-ElseLabel, -ContLabel)
%  Generate unique labels for if-then-else compilation.
next_ite_label(ElseLabel, ContLabel) :-
    (   nb_current(wam_ite_counter, N) -> true ; N = 0 ),
    N1 is N + 1,
    nb_setval(wam_ite_counter, N1),
    format(atom(ElseLabel), "L_ite_else_~w", [N1]),
    format(atom(ContLabel), "L_ite_cont_~w", [N1]).

%% flatten_conjunction(+Conj, -GoalList)
%  Flatten (A, B, C) into [A, B, C].
flatten_conjunction((A, B), Goals) :- !,
    flatten_conjunction(A, AG),
    flatten_conjunction(B, BG),
    append(AG, BG, Goals).
flatten_conjunction(Goal, [Goal]).

%% compile_if_then_else(+Cond, +Then, +Else, +V0, -Vf, +HasEnv, -Code)
%  Compile (Cond -> Then ; Else) to WAM try/cut/trust + jump pattern.
%  The condition runs in a temporary choice point; if it succeeds, cut
%  commits to Then. If it fails, backtrack to Else.
%
%  When Else is itself another if-then-else (`(C2 -> T2 ; E2)`) or a
%  bare `(C2 -> T2)` (implicit `; fail`), we recurse so a chain like
%  `(A -> B ; C -> D ; E)` compiles to nested cut_ite/try_me_else
%  pairs. Without this, the second `->` in Else position would emit
%  as a regular `Call("->", 2)` -- which has no runtime
%  implementation and just fails.
compile_if_then_else(CondGoal, ThenGoal, ElseGoal, V0, Vf, _HasEnv, Code) :-
    next_ite_label(ElseLabel, ContLabel),
    % Flatten condition into a goal list
    flatten_conjunction(CondGoal, CondGoals),
    compile_inner_call_goals(CondGoals, V0, V1, CondCode),
    % Then and Else can each be themselves a nested if-then-else --
    % compile_ite_branch recurses on those forms. Else starts from
    % V0 since backtrack restores to before the condition.
    compile_ite_branch(ThenGoal, V1, V2, ThenCode),
    compile_ite_branch(ElseGoal, V0, V3, ElseCode),
    % Use the wider variable map as output
    (   V2 = V3 -> Vf = V2
    ;   Vf = V2  % prefer then-branch vars (else-branch is alternative)
    ),
    % Emit: try_me_else ElseLabel / Cond / !/0 / Then / jump ContLabel
    %        ElseLabel: trust_me / Else / ContLabel:
    % Use cut_ite (soft cut) instead of !/0 — pops only the if-then-else
    % CP, preserving aggregate frames and outer choice points.
    % ContLabel marks the continuation after both branches.
    format(string(Code),
        "    try_me_else ~w~n~w~n    cut_ite~n~w~n    jump ~w~n~w:~n    trust_me~n~w~n~w:",
        [ElseLabel, CondCode, ThenCode, ContLabel, ElseLabel, ElseCode, ContLabel]).

%% compile_ite_branch(+Branch, +V0, -Vf, -Code) is det.
%  Compile a Then- or Else-branch of an if-then-else. If the branch
%  is itself another if-then-else (`(C -> T ; E)` or bare `(C -> T)`,
%  the latter treated as `(C -> T ; fail)` to match SWI), recurse;
%  otherwise compile as a flat goal sequence.
compile_ite_branch((Cond2 -> Then2 ; Else2), V0, Vf, Code) :-
    !,
    compile_if_then_else(Cond2, Then2, Else2, V0, Vf, no, Code).
compile_ite_branch((Cond2 -> Then2), V0, Vf, Code) :-
    !,
    compile_if_then_else(Cond2, Then2, fail, V0, Vf, no, Code).
compile_ite_branch(Branch, V0, Vf, Code) :-
    flatten_conjunction(Branch, BranchGoals),
    compile_inner_call_goals(BranchGoals, V0, Vf, Code).

%% compile_disjunction(+Left, +Right, +V0, -Vf, +HasEnv, -Code)
%  Compile (A ; B) to WAM try/trust + jump pattern (no cut).
compile_disjunction(LeftGoal, RightGoal, V0, Vf, _HasEnv, Code) :-
    next_ite_label(RightLabel, ContLabel),
    flatten_conjunction(LeftGoal, LeftGoals),
    flatten_conjunction(RightGoal, RightGoals),
    compile_inner_call_goals(LeftGoals, V0, V1, LeftCode),
    compile_inner_call_goals(RightGoals, V0, V2, RightCode),
    (   V1 = V2 -> Vf = V1 ; Vf = V1 ),
    format(string(Code),
        "    try_me_else ~w~n~w~n    jump ~w~n~w:~n    trust_me~n~w~n~w:",
        [RightLabel, LeftCode, ContLabel, RightLabel, RightCode, ContLabel]).

%% compile_inner_call_goals(+Goals, +V0, -Vf, -Code)
%  Compile all goals as calls (never execute/TCO) for use inside aggregate
%  bodies, if-then-else cond clauses, ite-branch bodies, and disjunction
%  arms. Dispatches on `(C -> T ; E)` / `(A ; B)` the same way the outer
%  `compile_goals` does, so nested if-then-else / disjunction inside a
%  conjunction body compiles to inline try/cut/trust + jump rather than
%  falling through to a generic `Call(";", 2)` (which has no runtime
%  handler and silently fails).
compile_inner_call_goals([], V, V, "").
compile_inner_call_goals([Goal|Rest], V0, Vf, Code) :-
    (   Goal = (CondGoal -> ThenGoal ; ElseGoal)
    ->  compile_if_then_else(CondGoal, ThenGoal, ElseGoal, V0, V1, no, GoalCode),
        compile_inner_call_goals(Rest, V1, Vf, RestCode),
        join_goal_codes(GoalCode, RestCode, Code)
    ;   Goal = (CondGoal -> ThenGoal)
    ->  compile_if_then_else(CondGoal, ThenGoal, fail, V0, V1, no, GoalCode),
        compile_inner_call_goals(Rest, V1, Vf, RestCode),
        join_goal_codes(GoalCode, RestCode, Code)
    ;   Goal = (LeftGoal ; RightGoal),
        \+ (LeftGoal = (_ -> _))
    ->  compile_disjunction(LeftGoal, RightGoal, V0, V1, no, GoalCode),
        compile_inner_call_goals(Rest, V1, Vf, RestCode),
        join_goal_codes(GoalCode, RestCode, Code)
    % once(G) / forall(G, T) — same desugarings as compile_goals.
    % Recurse with the rewritten goal so the if-then-else / negation
    % machinery picks it up.
    ;   Goal = once(OnceGoal)
    ->  compile_inner_call_goals([(OnceGoal -> true) | Rest], V0, Vf, Code)
    ;   Goal = forall(GenGoal, TestGoal)
    ->  compile_inner_call_goals([\+ (GenGoal, \+ TestGoal) | Rest],
                                 V0, Vf, Code)
    ;   compile_goal_call(Goal, V0, V1, GoalCode),
        compile_inner_call_goals(Rest, V1, Vf, RestCode),
        join_goal_codes(GoalCode, RestCode, Code)
    ).

join_goal_codes(GoalCode, "", GoalCode) :- !.
join_goal_codes(GoalCode, RestCode, Code) :-
    format(string(Code), "~w~n~w", [GoalCode, RestCode]).

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

% Static module qualifier unwrap. UnifyWeaver's targets resolve
% predicates by name only — there is no per-module dispatch table —
% so `M:p(X)` with a static atom (or string) module name is
% semantically identical to `p(X)`. Recursively compiling the inner
% goal emits a regular `call p/Arity, N` instead of routing through
% the `:/2` builtin path. Cross-target win: every target avoids the
% meta-call overhead (heap walk for the goal structure, register
% shuffling, dispatch table lookup) on the common static-qualifier
% case. The dynamic case (`Module = m, Module:p(X)`) where M is a
% Prolog variable at compile time still falls through to `:/2` —
% that genuinely needs a module registry which no target has yet.
% String + atom guard per #1647 follow-up review (Perplexity)
% covers any code path that represents module names as strings.
compile_goal_call(M:InnerGoal, V0, Vf, Code) :-
    (atom(M) ; string(M)),
    !,
    compile_goal_call(InnerGoal, V0, Vf, Code).
%% Visited-set arg rewrite (Layer 2.5 of the IntSet design). When a
%% goal calls a predicate with `:- visited_set(Pred/Arity, ArgN)` and
%% the arg at position N is a recognised list shape (`[Cat]` bootstrap
%% or `[X|V_visited]` recursive extension), rewrite the arg to a
%% fresh var bound to the constructed-VSet register and INLINE the
%% put_arguments + call. We do NOT recurse on the rewritten goal —
%% that's what caused the previous Layer 2 attempt to hang via the
%% TCO-dispatch / rewrite interaction.
compile_goal_call(Goal, V0, Vf, Code) :-
    nonvar(Goal),
    Goal =.. [Pred|Args],
    Args \== [],
    length(Args, Arity),
    rewrite_visited_set_args(Pred/Arity, Args, NewArgs, ConstructCode, V0, V1),
    !,
    compile_put_arguments(NewArgs, 1, V1, Vf, PutCode),
    (   is_builtin_pred(Pred, Arity)
    ->  format(string(CallCode), "    builtin_call ~w/~w, ~w", [Pred, Arity, Arity])
    ;   format(string(CallCode), "    call ~w/~w, ~w", [Pred, Arity, Arity])
    ),
    join_nonempty([ConstructCode, PutCode, CallCode], Code).
%% =../2 compose-mode lowering: T =.. [Name | FixedArgs] where T is
%% provably unbound and Name is provably bound. Emits PutStructureDyn
%% rather than the generic =../2 builtin call. Falls through to the
%% builtin path if the binding-state preconditions cannot be proved.
compile_goal_call(T =.. L, V0, Vf, Code) :-
    parse_univ_list_pattern(L, NameVar, FixedArgs),
    current_clause_binding_env(BeforeEnv),
    binding_state_analysis:binding_state_at_var(BeforeEnv, T, unbound),
    binding_state_analysis:binding_state_at_var(BeforeEnv, NameVar, bound),
    !,
    emit_put_structure_dyn_lowering(T, NameVar, FixedArgs, V0, Vf, Code).
%% functor/3 compose-mode lowering: functor(T, Name, Arity) where T is
%% provably unbound, Name is provably bound, and Arity is a literal
%% non-negative integer. Lowers to the same PutStructureDyn shape as
%% the =../2 case by synthesising Arity fresh unbound argument slots.
compile_goal_call(functor(T, NameVar, Arity), V0, Vf, Code) :-
    integer(Arity), Arity >= 0,
    var(T), var(NameVar),
    current_clause_binding_env(BeforeEnv),
    binding_state_analysis:binding_state_at_var(BeforeEnv, T, unbound),
    binding_state_analysis:binding_state_at_var(BeforeEnv, NameVar, bound),
    !,
    length(FreshArgs, Arity),
    emit_put_structure_dyn_lowering(T, NameVar, FreshArgs, V0, Vf, Code).
%% arg/3 lowering: arg(N, T, A) where N is a literal positive integer
%% and T is provably bound. Emits a single specialised `arg` WAM
%% instruction that the runtime translates to the Arg ADT constructor,
%% skipping the put_constant + put_value + builtin_call dispatch chain.
%% A may be bound or unbound; the runtime handles both via unification.
compile_goal_call(arg(N, T, A), V0, Vf, Code) :-
    integer(N), N >= 1,
    var(T), var(A),
    current_clause_binding_env(BeforeEnv),
    binding_state_analysis:binding_state_at_var(BeforeEnv, T, bound),
    !,
    emit_arg_lowering(N, T, A, V0, Vf, Code).
%% \+ member(X, V) lowering for visited-set variables (Phase H).
%% Fires when V is a head-arg variable at a position declared via
%% `:- visited_set(Pred/Arity, ArgN)` for this clause's predicate.
%% Emits `not_member_set` (O(log N)) instead of `not_member_list`
%% (O(N)). Must come BEFORE the NotMemberList clause so the more
%% specific case wins.
compile_goal_call(\+ member(X, V), V0, Vf, Code) :-
    var(X), var(V),
    is_visited_set_var(V),
    !,
    emit_not_member_set_lowering(X, V, V0, Vf, Code).
%% \+ member(X, [a,b,c,...]) lowering for compile-time-ground atom
%% lists. Bakes the atoms into a single NotMemberConstAtoms WAM
%% instruction — zero heap allocation, zero list-walk dispatch. Fires
%% whenever the second arg is a proper list of ground atoms in the
%% source, regardless of X's binding state (the runtime checks at
%% dispatch). Must come BEFORE the var(L) clause so the more specific
%% case wins.
compile_goal_call(\+ member(X, L), V0, Vf, Code) :-
    var(X),
    is_ground_atom_list(L, Atoms),
    \+ lowering_disabled(ground_member),
    !,
    emit_not_member_const_atoms_lowering(X, Atoms, V0, Vf, Code).
%% \+ member(X, L) lowering: emits a specialised NotMemberList WAM
%% instruction that walks L inline and skips both the put_structure
%% goal-term construction AND the builtin_call dispatch (4 dispatches +
%% one heap allocation per call). Fires when X and L are Prolog
%% variables that the binding-state analyser proves are both `bound`
%% at the goal site — typical of visited-set checks in graph
%% traversal: `parent(X, Z), \+ member(Z, V), recurse(Z, [Z|V])`.
compile_goal_call(\+ member(X, L), V0, Vf, Code) :-
    var(X), var(L),
    current_clause_binding_env(BeforeEnv),
    binding_state_analysis:binding_state_at_var(BeforeEnv, X, bound),
    binding_state_analysis:binding_state_at_var(BeforeEnv, L, bound),
    !,
    emit_not_member_list_lowering(X, L, V0, Vf, Code).
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

compile_goal_execute(M:InnerGoal, V0, Vf, Code) :-
    (atom(M) ; string(M)),
    !,
    compile_goal_execute(InnerGoal, V0, Vf, Code).
%% Visited-set arg rewrite, tail-call form. Mirrors compile_goal_call
%% but ends in `execute Pred/Arity` instead of `call`. Same single-pass
%% emission — no recursion on the rewritten goal.
compile_goal_execute(Goal, V0, Vf, Code) :-
    nonvar(Goal),
    Goal =.. [Pred|Args],
    Args \== [],
    length(Args, Arity),
    rewrite_visited_set_args(Pred/Arity, Args, NewArgs, ConstructCode, V0, V1),
    !,
    compile_put_arguments(NewArgs, 1, V1, Vf, PutCode),
    (   is_builtin_pred(Pred, Arity)
    ->  format(string(BuiltinCode), "    builtin_call ~w/~w, ~w", [Pred, Arity, Arity]),
        format(string(ExecCode), "~w~n    proceed", [BuiltinCode])
    ;   format(string(ExecCode), "    execute ~w/~w", [Pred, Arity])
    ),
    join_nonempty([ConstructCode, PutCode, ExecCode], Code).
%% =../2 compose-mode lowering, tail-call form. Produces the same
%% PutStructureDyn lowering followed by `proceed`.
compile_goal_execute(T =.. L, V0, Vf, Code) :-
    parse_univ_list_pattern(L, NameVar, FixedArgs),
    current_clause_binding_env(BeforeEnv),
    binding_state_analysis:binding_state_at_var(BeforeEnv, T, unbound),
    binding_state_analysis:binding_state_at_var(BeforeEnv, NameVar, bound),
    !,
    emit_put_structure_dyn_lowering(T, NameVar, FixedArgs, V0, Vf, BodyCode),
    format(string(Code), "~w~n    proceed", [BodyCode]).
%% functor/3 compose-mode lowering, tail-call form. Same shape as the
%% call form: synthesise Arity fresh unbound slots and reuse the =../2
%% emit helper.
compile_goal_execute(functor(T, NameVar, Arity), V0, Vf, Code) :-
    integer(Arity), Arity >= 0,
    var(T), var(NameVar),
    current_clause_binding_env(BeforeEnv),
    binding_state_analysis:binding_state_at_var(BeforeEnv, T, unbound),
    binding_state_analysis:binding_state_at_var(BeforeEnv, NameVar, bound),
    !,
    length(FreshArgs, Arity),
    emit_put_structure_dyn_lowering(T, NameVar, FreshArgs, V0, Vf, BodyCode),
    format(string(Code), "~w~n    proceed", [BodyCode]).
%% arg/3 lowering, tail-call form.
compile_goal_execute(arg(N, T, A), V0, Vf, Code) :-
    integer(N), N >= 1,
    var(T), var(A),
    current_clause_binding_env(BeforeEnv),
    binding_state_analysis:binding_state_at_var(BeforeEnv, T, bound),
    !,
    emit_arg_lowering(N, T, A, V0, Vf, BodyCode),
    format(string(Code), "~w~n    proceed", [BodyCode]).
%% \+ member(X, V) for visited-set V, tail-call form.
compile_goal_execute(\+ member(X, V), V0, Vf, Code) :-
    var(X), var(V),
    is_visited_set_var(V),
    !,
    emit_not_member_set_lowering(X, V, V0, Vf, BodyCode),
    format(string(Code), "~w~n    proceed", [BodyCode]).
%% \+ member(X, [a,b,c,...]) ground-list lowering, tail-call form.
compile_goal_execute(\+ member(X, L), V0, Vf, Code) :-
    var(X),
    is_ground_atom_list(L, Atoms),
    \+ lowering_disabled(ground_member),
    !,
    emit_not_member_const_atoms_lowering(X, Atoms, V0, Vf, BodyCode),
    format(string(Code), "~w~n    proceed", [BodyCode]).
%% \+ member(X, L) lowering, tail-call form.
compile_goal_execute(\+ member(X, L), V0, Vf, Code) :-
    var(X), var(L),
    current_clause_binding_env(BeforeEnv),
    binding_state_analysis:binding_state_at_var(BeforeEnv, X, bound),
    binding_state_analysis:binding_state_at_var(BeforeEnv, L, bound),
    !,
    emit_not_member_list_lowering(X, L, V0, Vf, BodyCode),
    format(string(Code), "~w~n    proceed", [BodyCode]).
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
is_builtin_pred(write, 1).  % I/O — useful for runtime instrumentation.
is_builtin_pred(display, 1).
is_builtin_pred(nl, 0).
is_builtin_pred(format, 1).  % I/O — formatted output, ~-directives.
is_builtin_pred(format, 2).
is_builtin_pred(format, 3).
is_builtin_pred(atom_codes, 2).   % atom ↔ list of integer codes.
is_builtin_pred(atom_chars, 2).   % atom ↔ list of single-char atoms.
is_builtin_pred(number_codes, 2). % number ↔ list of integer codes.
is_builtin_pred(atom_concat, 3).  % (+, +, ?) — concatenation.
is_builtin_pred(atom_length, 2).  % atom → length in chars.
is_builtin_pred(char_code, 2).    % char-atom ↔ integer code.
is_builtin_pred(assertz, 1).      % dynamic db: append fact.
is_builtin_pred(asserta, 1).      % dynamic db: prepend fact.
% retract/1 is nondeterministic — dispatched via the Call/Execute
% step arms (like findall/sub_atom) so the CP iterator can backtrack
% through subsequent matches. NOT in is_builtin_pred so the compiler
% emits `call retract/1, 1` rather than `builtin_call`.
is_builtin_pred(retractall, 1).   % dynamic db: remove all matches.
is_builtin_pred(nb_setval, 2).    % mutable global: replace value.
is_builtin_pred(nb_getval, 2).    % mutable global: read value.
is_builtin_pred(b_setval, 2).     % backtrackable mutable global: bind.
is_builtin_pred(b_getval, 2).     % backtrackable mutable global: read.
is_builtin_pred(@<, 2).           % standard order: less than.
is_builtin_pred(@=<, 2).          % standard order: less or equal.
is_builtin_pred(@>, 2).           % standard order: greater than.
is_builtin_pred(@>=, 2).          % standard order: greater or equal.
is_builtin_pred(compare, 3).      % standard order: -1 / 0 / +1 as atom.
is_builtin_pred(char_type, 2).    % char classification + case conv.
is_builtin_pred(upcase_atom, 2).  % whole-atom case conversion: upper.
is_builtin_pred(downcase_atom, 2).% whole-atom case conversion: lower.
is_builtin_pred(numlist, 3).      % integer range generator: [Lo..Hi].
is_builtin_pred(sort, 2).         % stable sort + dedup (std order).
is_builtin_pred(msort, 2).        % stable sort, NO dedup (std order).
is_builtin_pred(select, 3).       % nondet list element selection.
is_builtin_pred(maplist, 2).      % apply goal to each list element.
is_builtin_pred(maplist, 3).
is_builtin_pred(maplist, 4).
is_builtin_pred(maplist, 5).
is_builtin_pred(include, 3).      % filter: keep elems where Goal succeeds.
is_builtin_pred(exclude, 3).      % filter: drop elems where Goal succeeds.
is_builtin_pred(partition, 4).    % split into pass + fail lists.
is_builtin_pred(foldl, 4).        % left fold over one list.
is_builtin_pred(foldl, 5).        % left fold over two lists in parallel.
is_builtin_pred(keysort, 2).      % stable sort of Key-Value pairs by Key.
is_builtin_pred(pairs_keys, 2).   % extract keys from Key-Value pairs.
is_builtin_pred(pairs_values, 2). % extract values from Key-Value pairs.
is_builtin_pred(pairs_keys_values, 3). % split pairs into parallel lists.
% Note: sub_atom/5 is nondeterministic; like findall/bagof/setof it
% goes through the Call/Execute dispatch path (not is_builtin_pred)
% so dispatch_sub_atom can manage its own CP machinery.
is_builtin_pred(=, 2).       % unification: bind / structurally unify two terms.
is_builtin_pred(is_list, 1).
                             % Without this entry, X = Y in a body goal
                             % would compile to `execute =/2`, which looks
                             % up "=/2" as a user predicate, misses, and
                             % resolves to PC=0 at runtime — silently
                             % re-entering the project's first predicate
                             % and corrupting execution state.

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
    ->  quote_wam_constant(Arg, ArgStr),
        format(string(Code), "    put_constant ~w, A~w", [ArgStr, I]),
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
        quote_wam_constant(Arg, ArgStr),
        format(string(ArgCode), "    set_constant ~w", [ArgStr])
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

%% ========================================================================
%% Binding-state analysis plumbing for =../2 compose-mode lowering.
%%
%% The binding analyser produces a list of `goal_binding(Idx, Before,
%% After)` records per clause. We stash that list in a non-backtrackable
%% global before invoking the body walk, and the new `compile_goal_call`
%% / `compile_goal_execute` clauses for `T =.. L` consult it.
%%
%% The analysis is *additive* — if the global is unset (e.g. when a
%% target invokes `compile_goal_call/4` directly without going through
%% `compile_single_clause_wam`), the lowering simply falls through to
%% the existing builtin path. No regression for any non-WAM-Haskell
%% caller.
%% ========================================================================

%% set_clause_binding_context(+Head, +Body)
%  Run the binding analyser on the clause and stash the resulting list
%  of `goal_binding/3` records in `wam_clause_binding_records`. Also
%  resets the per-goal index counter to 1.
set_clause_binding_context(Head, Body) :-
    catch(
        binding_state_analysis:analyse_clause_bindings(Head, Body, Bindings),
        _Err,
        Bindings = []
    ),
    b_setval(wam_clause_binding_records, Bindings),
    b_setval(wam_clause_goal_idx, 1).

%% current_clause_binding_env(-BeforeEnv)
%  Returns the BeforeEnv of the goal currently being compiled. Defaults
%  to the empty env when no analysis is in scope.
current_clause_binding_env(BeforeEnv) :-
    (   catch(b_getval(wam_clause_binding_records, Bindings), _, fail)
    ->  true
    ;   Bindings = []
    ),
    (   catch(b_getval(wam_clause_goal_idx, Idx), _, fail)
    ->  true
    ;   Idx = 1
    ),
    (   member(goal_binding(Idx, Env, _), Bindings)
    ->  BeforeEnv = Env
    ;   binding_state_analysis:empty_binding_env(BeforeEnv)
    ).

%% advance_clause_goal_idx
%  Step the per-clause goal index forward by one. Called from the body
%  walk between goals so the binding lookup sees the right BeforeEnv.
advance_clause_goal_idx :-
    (   catch(b_getval(wam_clause_goal_idx, Idx), _, fail)
    ->  Idx1 is Idx + 1,
        b_setval(wam_clause_goal_idx, Idx1)
    ;   true
    ).

%% ========================================================================
%% Visited-set context (Layer 2 of the IntSet visited design).
%%
%% A predicate-arg position can be marked as a visited-set via
%%
%%     :- visited_set(Pred/Arity, ArgN).
%%
%% asserted into the user module before compilation. The compiler then:
%%
%%   1. At clause entry, captures which head-arg variables of THIS
%%      clause sit at declared visited-set positions and stashes them
%%      in `wam_clause_visited_vars` (a non-backtrackable global).
%%
%%   2. At a body goal `\+ member(X, V)` where V is in that set, emits
%%      `not_member_set` instead of `not_member_list`.
%%
%%   3. At a CALL site whose target predicate-arg matches a directive
%%      AND whose actual arg is a list-shaped term:
%%        * `[Cat]` (1-elem ground literal)  ⇒
%%            build_empty_set + set_insert(Cat) sequence
%%        * `[X|V_visited]` (cons of head's visited-set var) ⇒
%%            set_insert(X, V_visited, Fresh)
%%      bound to a fresh Prolog variable, which the standard
%%      put_argument path then stores in the call's argument register.
%%
%% Soundness: if `:- visited_set/2` is wrong (e.g. claims an arg is a
%% set but the runtime sees a non-Atom element), `set_insert` /
%% `not_member_set` return Nothing → the goal fails. No silent
%% miscompilation.
%% ========================================================================

%% set_clause_visited_context(+Head)
%  Read user:visited_set(Pred/Arity, ArgN) declarations and capture
%  the head-arg variables matching this clause as the per-clause
%  visited-set var set. Stored as a list because Prolog vars are not
%  groundable for use as map keys without copy_term.
set_clause_visited_context(Head) :-
    (   compound(Head)
    ->  Head =.. [Pred|HeadArgs],
        length(HeadArgs, Arity),
        %% Walk HeadArgs in place (NOT via findall, which would copy
        %% the captured variables and break == identity later).
        collect_visited_vars(HeadArgs, 1, Pred/Arity, VisitedVars)
    ;   VisitedVars = []
    ),
    b_setval(wam_clause_visited_vars, VisitedVars).

%% collect_visited_vars(+Args, +N, +Pred/+Arity, -CapturedVars)
%  Walk Args; for each variable position whose index N matches a
%  visited_set directive on Pred/Arity, capture the variable in
%  CapturedVars (preserving its identity — no findall copy).
collect_visited_vars([], _, _, []).
collect_visited_vars([Arg|Rest], N, PA, [Arg|MoreVars]) :-
    var(Arg),
    is_visited_set_arg(PA, N),
    !,
    N1 is N + 1,
    collect_visited_vars(Rest, N1, PA, MoreVars).
collect_visited_vars([_|Rest], N, PA, MoreVars) :-
    N1 is N + 1,
    collect_visited_vars(Rest, N1, PA, MoreVars).

%% is_visited_set_var(+Var)
%  True when Var is a Prolog variable currently in the per-clause
%  visited-set context (captured by set_clause_visited_context/1).
is_visited_set_var(Var) :-
    var(Var),
    catch(b_getval(wam_clause_visited_vars, Vars), _, fail),
    member(V, Vars),
    V == Var.

%% is_visited_set_arg(+Pred/+Arity, +ArgN)
%  True when the directive `user:visited_set(Pred/Arity, ArgN)` has
%  been asserted.
is_visited_set_arg(Pred/Arity, ArgN) :-
    current_predicate(user:visited_set/2),
    user:visited_set(Pred/Arity, ArgN).


%% is_term_construction_goal(+Goal)
%
%  True when Goal matches a builtin that the WAM compiler recognises
%  for binding-analysis-driven lowering: `T =.. L`, `functor(T, N, A)`,
%  `arg(N, T, A)`, or `\+ member(X, L)`. Used by the TCO routing in
%  compile_goals/5 to direct these goals through compile_goal_execute
%  so the lowering preconditions can be checked.
is_term_construction_goal(Goal) :-
    nonvar(Goal),
    (   Goal = (_ =.. _)
    ;   Goal = functor(_, _, _)
    ;   Goal = arg(_, _, _)
    ;   Goal = (\+ member(_, _))
    ).

%% emit_not_member_list_lowering(+X, +L, +V0, -Vf, -Code)
%
%  Emits a single `not_member_list XReg LReg` WAM instruction. Both
%  X and L are required (by the caller's binding-analysis check) to
%  already have allocated registers on entry; this helper simply
%  looks them up. Falls back to allocating fresh X registers if the
%  variables somehow do not yet have register assignments — a defensive
%  path that should not fire when the precondition holds.
emit_not_member_list_lowering(X, L, V0, Vf, Code) :-
    (   get_var_reg(X, V0, XReg)
    ->  V1 = V0,
        XPrefix = ""
    ;   next_x_reg(V0, XReg, V_temp1),
        bind_var(X, XReg, V_temp1, V1),
        format(string(XPrefix), "    put_variable ~w, A1", [XReg])
    ),
    (   get_var_reg(L, V1, LReg)
    ->  Vf = V1,
        LPrefix = ""
    ;   next_x_reg(V1, LReg, V_temp2),
        bind_var(L, LReg, V_temp2, Vf),
        format(string(LPrefix), "    put_variable ~w, A2", [LReg])
    ),
    format(string(Body), "    not_member_list ~w, ~w", [XReg, LReg]),
    join_nonempty([XPrefix, LPrefix, Body], Code).

%% emit_not_member_set_lowering(+X, +V, +V0, -Vf, -Code)
%
%  Emits a single `not_member_set XReg VReg` WAM instruction (Layer 2
%  of the IntSet visited design). Mirrors the not_member_list helper
%  above; the runtime semantic is O(log N) IntSet lookup instead of
%  O(N) list walk. Pre-condition (checked by the caller): V is in the
%  per-clause visited-set context.
emit_not_member_set_lowering(X, V, V0, Vf, Code) :-
    (   get_var_reg(X, V0, XReg)
    ->  V1 = V0,
        XPrefix = ""
    ;   next_x_reg(V0, XReg, V_temp1),
        bind_var(X, XReg, V_temp1, V1),
        format(string(XPrefix), "    put_variable ~w, A1", [XReg])
    ),
    (   get_var_reg(V, V1, VReg)
    ->  Vf = V1,
        VPrefix = ""
    ;   next_x_reg(V1, VReg, V_temp2),
        bind_var(V, VReg, V_temp2, Vf),
        format(string(VPrefix), "    put_variable ~w, A2", [VReg])
    ),
    format(string(Body), "    not_member_set ~w, ~w", [XReg, VReg]),
    join_nonempty([XPrefix, VPrefix, Body], Code).

%% lowering_disabled(+Tag) is semidet.
%
%  Per-process opt-out hook used by benchmarks to compile a baseline
%  project with a specific lowering turned off. Currently recognised:
%
%    ground_member — disables the literal-ground-list `\+ member`
%                    lowering (NotMemberConstAtoms). Falls through to
%                    the standard builtin dispatch path.
%
%  Default: not disabled. Bench scripts assert
%  `wam_target:lowering_disabled(ground_member)` before generating an
%  unlowered baseline project, then retract afterwards.
:- dynamic(lowering_disabled/1).

%% is_ground_atom_list(+L, -Atoms) is semidet.
%
%  Succeeds when L is a proper list of ground atoms, binding Atoms to
%  the same list. Fails for variables, partial lists, lists containing
%  numbers, structures, or unbound elements. Used to detect the
%  literal-list shape of `\+ member(X, [a, b, c])` for codegen lowering.
is_ground_atom_list(L, Atoms) :-
    nonvar(L),
    proper_list(L, Items),
    Items \== [],
    maplist(atom, Items),
    Atoms = Items.

proper_list(T, []) :- T == [], !.
proper_list([H|T], [H|R]) :- proper_list(T, R).

%% emit_not_member_const_atoms_lowering(+X, +Atoms, +V0, -Vf, -Code)
%
%  Emits a single `not_member_const_atoms XReg A1 A2 ... AN` WAM
%  instruction. Atoms are interned at the WAM-haskell-target stage,
%  so what we emit here is just the atom names space-separated. X must
%  resolve to a register; if it doesn't already have one, allocate
%  fresh.
emit_not_member_const_atoms_lowering(X, Atoms, V0, Vf, Code) :-
    (   get_var_reg(X, V0, XReg)
    ->  Vf = V0,
        XPrefix = ""
    ;   next_x_reg(V0, XReg, V_temp),
        bind_var(X, XReg, V_temp, Vf),
        format(string(XPrefix), "    put_variable ~w, A1", [XReg])
    ),
    atomic_list_concat(Atoms, ' ', AtomsStr),
    format(string(Body), "    not_member_const_atoms ~w ~w", [XReg, AtomsStr]),
    join_nonempty([XPrefix, Body], Code).

%% rewrite_visited_set_args(+Pred/+Arity, +Args, -NewArgs, -Code, +V0, -Vf)
%
%  Rewrite call-site args at positions declared as visited-sets via
%  `:- visited_set(Pred/Arity, ArgN)`. Recognises two list shapes:
%
%    Pattern 1 — bootstrap `[X]` (1-elem list, var head):
%      build_empty_set Rs ; set_insert XReg, Rs, Rs
%      Replaces the arg with a fresh var bound to Rs.
%
%    Pattern 2 — recursive `[X|V_visited]` (cons of head's visited-set
%      var): set_insert XReg, V_visited_Reg, Rfresh
%      Replaces the arg with a fresh var bound to Rfresh.
%
%  Anything else passes through unchanged. Succeeds only when at
%  least one position rewrites — otherwise fails so the caller falls
%  through to the standard compile_put_arguments path.
rewrite_visited_set_args(PA, Args, NewArgs, Code, V0, Vf) :-
    rewrite_args_loop(Args, 1, PA, NewArgs, V0, Vf, Pieces),
    Pieces \= [],
    join_nonempty(Pieces, Code).

rewrite_args_loop([], _, _, [], V, V, []).
rewrite_args_loop([Arg|Rest], N, PA, [NewArg|NewRest], V0, Vf, Pieces) :-
    (   is_visited_set_arg(PA, N),
        rewrite_visited_arg(Arg, NewArg, V0, V1, ArgCode)
    ->  Pieces = [ArgCode|RestPieces]
    ;   NewArg = Arg, V1 = V0, Pieces = RestPieces
    ),
    N1 is N + 1,
    rewrite_args_loop(Rest, N1, PA, NewRest, V1, Vf, RestPieces).

%% rewrite_visited_arg(+Arg, -NewArg, +V0, -Vf, -Code)
%
%  Recursive extension `[X|V_visited]` (cons of head's visited-set
%  var). Tested FIRST so the more specific case wins — clause-head
%  unification of `[X]` would otherwise bind V_visited=[] and fall
%  through to the bootstrap path with wrong semantics.
rewrite_visited_arg(L, NewArg, V0, Vf, Code) :-
    nonvar(L),
    L = [X|V],
    var(X), var(V),
    is_visited_set_var(V),
    !,
    (   get_var_reg(X, V0, XReg)
    ->  V1 = V0
    ;   next_x_reg(V0, XReg, V_t1),
        bind_var(X, XReg, V_t1, V1)
    ),
    (   get_var_reg(V, V1, VReg)
    ->  V2 = V1
    ;   next_x_reg(V1, VReg, V_t2),
        bind_var(V, VReg, V_t2, V2)
    ),
    next_x_reg(V2, NewSetReg, V3),
    bind_var(NewArg, NewSetReg, V3, Vf),
    format(string(Code), "    set_insert ~w, ~w, ~w", [XReg, VReg, NewSetReg]).
%% Bootstrap: `[X]` (1-elem list with var head and atom-`[]` tail)
%% -> build_empty_set + set_insert(X). The tail-is-atom-[] guard
%% prevents this from matching `[X|V_var]` where V_var would get
%% silently bound to [] by Prolog clause-head unification.
rewrite_visited_arg(L, NewArg, V0, Vf, Code) :-
    nonvar(L),
    L = [X|T],
    var(X),
    nonvar(T), T == [],
    !,
    (   get_var_reg(X, V0, XReg)
    ->  V1 = V0
    ;   next_x_reg(V0, XReg, V_t1),
        bind_var(X, XReg, V_t1, V1)
    ),
    next_x_reg(V1, SetReg, V2),
    bind_var(NewArg, SetReg, V2, Vf),
    format(string(Code),
        "    build_empty_set ~w~n    set_insert ~w, ~w, ~w",
        [SetReg, XReg, SetReg, SetReg]).

%% goal_has_visited_set_arg(+Goal)
%  True when Goal calls a predicate with at least one visited_set
%  declaration. Used by the TCO dispatch in compile_goals/5 to route
%  the goal through compile_goal_execute (where the rewrite fires)
%  instead of the inline put_arguments path.
goal_has_visited_set_arg(Goal) :-
    nonvar(Goal),
    Goal =.. [Pred|Args],
    Args \== [],
    length(Args, Arity),
    is_visited_set_arg(Pred/Arity, _).

%% emit_arg_lowering(+N, +T, +A, +V0, -Vf, -Code)
%
%  Emits a single `arg N TReg AReg` WAM instruction. Allocates X
%  registers for T and A as needed (mirrors the conventions in
%  emit_put_structure_dyn_lowering/6).
emit_arg_lowering(N, T, A, V0, Vf, Code) :-
    %% TReg: T must already have a register (precondition: bound). If
    %% not allocated, allocate one and emit put_variable for safety.
    (   get_var_reg(T, V0, TReg)
    ->  V1 = V0,
        TPrefix = ""
    ;   next_x_reg(V0, TReg, V_temp1),
        bind_var(T, TReg, V_temp1, V1),
        format(string(TPrefix), "    put_variable ~w, A1", [TReg])
    ),
    %% AReg: allocate if not present.
    (   get_var_reg(A, V1, AReg)
    ->  Vf = V1
    ;   next_x_reg(V1, AReg, V_temp2),
        bind_var(A, AReg, V_temp2, Vf)
    ),
    format(string(ArgCode), "    arg ~w, ~w, ~w", [N, TReg, AReg]),
    (   TPrefix == ""
    ->  Code = ArgCode
    ;   format(string(Code), "~w~n~w", [TPrefix, ArgCode])
    ).

%% parse_univ_list_pattern(+List, -NameVar, -FixedArgs)
%
%  Recognises the list literal `[NameVar | FixedArgs]` where NameVar is
%  a Prolog variable and FixedArgs is a fixed-length proper list
%  (length ≥ 0, no tail variables). Used by the =../2 compose-mode
%  lowering to extract the dynamic functor name and the fixed
%  argument list.
parse_univ_list_pattern(List, _NameVar, _FixedArgs) :-
    var(List), !, fail.
parse_univ_list_pattern([NameVar|Rest], NameVar, FixedArgs) :-
    var(NameVar),
    proper_fixed_list(Rest, FixedArgs).

proper_fixed_list(L, _) :- var(L), !, fail.
proper_fixed_list([], []) :- !.
proper_fixed_list([H|T], [H|FixedT]) :-
    proper_fixed_list(T, FixedT).

%% emit_put_structure_dyn_lowering(+T, +NameVar, +FixedArgs, +V0, -Vf, -Code)
%
%  Emit the WAM instruction sequence that constructs a structure whose
%  functor name comes from a register (`NameVar`) and whose arity is
%  the literal length of FixedArgs.
%
%  Sequence:
%      put_value Reg(NameVar), A1     # nameReg
%      put_constant N, A2             # arityReg (literal int)
%      put_structure_dyn A1, A2, A3   # construct Str at A3
%      <set_value/set_variable for each FixedArg>
%      get_value Reg(T), A3           # unify with T
%
%  Pre-condition: the caller has verified that NameVar is `bound` and
%  T is `unbound` in the BeforeEnv.
emit_put_structure_dyn_lowering(T, NameVar, FixedArgs, V0, Vf, Code) :-
    %% A1 = NameVar's register (must already exist; if not, allocate).
    (   get_var_reg(NameVar, V0, NameReg)
    ->  V1 = V0,
        format(string(NameCode), "    put_value ~w, A1", [NameReg])
    ;   next_x_reg(V0, NameReg, V_temp1),
        bind_var(NameVar, NameReg, V_temp1, V1),
        format(string(NameCode), "    put_variable ~w, A1", [NameReg])
    ),
    %% A2 = literal arity.
    length(FixedArgs, Arity),
    format(string(ArityCode), "    put_constant ~w, A2", [Arity]),
    %% A3 = constructed term register.
    next_x_reg(V1, TermReg, V2),
    format(string(StructCode), "    put_structure_dyn A1, A2, ~w", [TermReg]),
    %% Emit set_value/set_variable for each FixedArg.
    compile_set_arguments(FixedArgs, V2, V3, SetCode),
    %% Bind T to the result. If T already has a reg, emit get_value;
    %% otherwise bind T to TermReg directly (T is now aliased to the
    %% constructed term).
    (   get_var_reg(T, V3, TReg)
    ->  Vf = V3,
        format(string(GetCode), "    get_value ~w, ~w", [TReg, TermReg])
    ;   bind_var(T, TermReg, V3, Vf),
        GetCode = ""
    ),
    %% Stitch together, dropping empty pieces.
    join_nonempty(
        [NameCode, ArityCode, StructCode, SetCode, GetCode],
        Code).

join_nonempty(Pieces, Code) :-
    exclude(=(""), Pieces, NonEmpty),
    atomic_list_concat(NonEmpty, '\n', CodeAtom),
    atom_string(CodeAtom, Code).
