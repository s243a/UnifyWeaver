:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% mutual_recursion.pl - Compile mutually recursive predicates
% Uses SCC detection to identify mutual recursion groups
% Compiles with shared memoization tables

:- module(mutual_recursion, [
    compile_mutual_recursion/3,     % +Predicates, +Options, -BashCode
    can_compile_mutual_recursion/1,  % +Predicates
    test_mutual_recursion/0        % Test predicate
]).

:- use_module(library(lists)).
:- use_module('../template_system').
:- use_module('../constraint_analyzer').
:- use_module('call_graph').
:- use_module('scc_detection').

%% can_compile_mutual_recursion(+Predicates)
%  Check if predicates form a mutually recursive group
%  Predicates = [pred1/arity1, pred2/arity2, ...]
can_compile_mutual_recursion(Predicates) :-
    length(Predicates, Len),
    Len > 1,  % Need at least 2 predicates for mutual recursion

    % Build call graph
    build_call_graph(Predicates, Graph),

    % Find SCCs
    find_sccs(Graph, SCCs),

    % Check if predicates are in same non-trivial SCC
    member(SCC, SCCs),
    \+ is_trivial_scc(SCC),
    forall(member(Pred, Predicates), member(Pred, SCC)).

%% merge_scc_constraints(+Predicates, -MergedConstraints)
%  Merge constraints from all predicates in an SCC
%  Strategy: Most restrictive wins
%  - If ANY predicate has unique(false), result is unique(false)
%  - If ANY predicate has unordered(false) (ordered), result is unordered(false)
merge_scc_constraints(Predicates, MergedConstraints) :-
    % Get constraints for each predicate
    findall(Constraints,
        (   member(Pred, Predicates),
            get_constraints(Pred, Constraints)
        ),
        AllConstraints),

    % Flatten and merge
    append(AllConstraints, FlatConstraints),

    % Determine unique constraint (if any has false, use false)
    (   member(unique(false), FlatConstraints) ->
        UniqueConstraint = unique(false)
    ;   UniqueConstraint = unique(true)
    ),

    % Determine unordered constraint (if any has false, use false)
    (   member(unordered(false), FlatConstraints) ->
        UnorderedConstraint = unordered(false)
    ;   UnorderedConstraint = unordered(true)
    ),

    MergedConstraints = [UniqueConstraint, UnorderedConstraint].

%% compile_mutual_recursion(+Predicates, +Options, -BashCode)
%  Compile a group of mutually recursive predicates
%  Generates bash code with shared memoization
%  Options: List of Key=Value pairs (e.g., [unique=true, ordered=false])
%
%  Constraint effects for mutual recursion:
%  - unique(false): Disable shared memoization for the entire SCC
%  - ordered(true): Use hash-based shared memoization (preserves order)
%  - ordered(false): Use standard shared memoization (default)
%
%  NOTE: When predicates in an SCC have different constraints, we take the
%  intersection of constraints (most restrictive wins):
%  - If ANY predicate has unique(false), memoization is disabled for all
%  - If ANY predicate requires ordered, hash-based memo is used for all
compile_mutual_recursion(Predicates, Options, BashCode) :-
    format('  Compiling mutual recursion group: ~w~n', [Predicates]),

    % Merge constraints from all predicates in the SCC
    merge_scc_constraints(Predicates, SCCConstraints),
    format('  Merged SCC constraints: ~w~n', [SCCConstraints]),

    % Merge runtime options with merged constraints (runtime options override)
    append(Options, SCCConstraints, AllOptions),
    format('  Final options: ~w~n', [AllOptions]),

    % Determine memoization strategy based on constraints
    (   member(unique(false), AllOptions) ->
        format('  Applying unique(false): Shared memo disabled~n', []),
        MemoEnabled = false
    ;   MemoEnabled = true
    ),

    % Determine memo lookup strategy
    (   member(unordered(false), AllOptions) ->  % ordered = true
        format('  Applying ordered constraint: Using hash-based shared memo~n', []),
        MemoStrategy = hash
    ;   MemoStrategy = standard
    ),

    % Generate bash code for the group
    generate_mutual_recursion_bash(Predicates, MemoEnabled, MemoStrategy, BashCode).

%% generate_mutual_recursion_bash(+Predicates, +MemoEnabled, +MemoStrategy, -BashCode)
generate_mutual_recursion_bash(Predicates, MemoEnabled, MemoStrategy, BashCode) :-
    % Extract predicate names
    findall(PredStr,
        (   member(Pred/_Arity, Predicates),
            atom_string(Pred, PredStr)
        ),
        PredStrs),
    atomic_list_concat(PredStrs, '_', GroupName),

    % Generate header with constraint info
    generate_mutual_header(GroupName, MemoEnabled, MemoStrategy, HeaderCode),

    % Generate function for each predicate
    findall(FuncCode,
        (   member(Pred/Arity, Predicates),
            generate_mutual_function(Pred, Arity, GroupName, Predicates, MemoEnabled, FuncCode)
        ),
        FuncCodes),
    atomic_list_concat(FuncCodes, '\n\n', FunctionsCode),

    % Combine
    atomic_list_concat([HeaderCode, FunctionsCode], '\n\n', BashCode).

%% generate_mutual_header(+GroupName, +MemoEnabled, +MemoStrategy, -HeaderCode)
generate_mutual_header(GroupName, MemoEnabled, MemoStrategy, HeaderCode) :-
    % Generate memo declaration (if enabled)
    (   MemoEnabled = true ->
        format(string(MemoDecl), 'declare -gA ~w_memo', [GroupName]),
        format(string(MemoComment), 'Shared memoization for mutual recursion (~w strategy)', [MemoStrategy])
    ;   MemoDecl = '# Shared memoization disabled (unique=false constraint)',
        MemoComment = 'Shared memoization disabled'
    ),

    TemplateLines = [
        "#!/bin/bash",
        "# Mutually recursive group: {{group}}",
        "# {{memo_comment}}",
        "",
        "# Shared memo table for all predicates in this group",
        "{{memo_decl}}"
    ],

    atomic_list_concat(TemplateLines, '\n', Template),
    render_template(Template, [group=GroupName, memo_decl=MemoDecl, memo_comment=MemoComment], HeaderCode).

%% generate_mutual_function(+Pred, +Arity, +GroupName, +AllPredicates, +MemoEnabled, -FuncCode)
%  Generate bash function for one predicate in mutual group
generate_mutual_function(Pred, Arity, GroupName, AllPredicates, MemoEnabled, FuncCode) :-
    atom_string(Pred, PredStr),

    % Get clauses for this predicate - use user:clause to access predicates from any module (including test predicates)
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),

    % Separate base and recursive cases (check for calls to any predicate in the group)
    partition(is_mutual_recursive_clause(AllPredicates), Clauses, RecClauses, BaseClauses),

    % Generate base case code
    generate_base_cases_code(GroupName, BaseClauses, MemoEnabled, BaseCode),

    % Generate recursive case code
    generate_recursive_cases_code(GroupName, RecClauses, MemoEnabled, RecCode),

    % Generate function template
    generate_function_template(PredStr, Arity, GroupName, MemoEnabled, BaseCode, RecCode, FuncCode).

%% is_mutual_recursive_clause(+AllPredicates, +Clause)
%  Check if clause contains a call to any predicate in the mutual group
is_mutual_recursive_clause(AllPredicates, clause(_Head, Body)) :-
    member(Pred/Arity, AllPredicates),
    functor(Goal, Pred, Arity),
    contains_goal(Body, Goal).

%% contains_goal(+Body, +TemplateGoal)
contains_goal((A, B), Goal) :-
    !,
    (   contains_goal(A, Goal)
    ;   contains_goal(B, Goal)
    ).
contains_goal(Body, TemplateGoal) :-
    compound(Body),
    functor(Body, F, A),
    functor(TemplateGoal, F, A).

%% is_recursive_clause_any(+Pred, +Clause)
%  Check if clause contains any recursive call
is_recursive_clause_any(Pred, clause(_Head, Body)) :-
    contains_any_call(Body, Pred).

%% contains_any_call(+Body, +Pred)
contains_any_call(Body, Pred) :-
    extract_goal_simple(Body, Goal),
    functor(Goal, Pred, _).

%% extract_goal_simple(+Body, -Goal)
extract_goal_simple(Goal, Goal) :-
    compound(Goal),
    \+ Goal = (_,_),
    \+ Goal = (_;_).
extract_goal_simple((A, _), Goal) :- extract_goal_simple(A, Goal).
extract_goal_simple((_, B), Goal) :- extract_goal_simple(B, Goal).

%% generate_base_cases_code(+GroupName, +BaseClauses, +MemoEnabled, -BaseCode)
generate_base_cases_code(GroupName, BaseClauses, MemoEnabled, BaseCode) :-
    length(BaseClauses, NumBase),
    format(string(Header), '# ~w base case(s)', [NumBase]),
    (   BaseClauses = [] ->
        BaseCode = Header
    ;   findall(Code,
            (   member(Clause, BaseClauses),
                generate_base_case_code(GroupName, Clause, MemoEnabled, Code)
            ),
            BaseCodes),
        atomic_list_concat([Header|BaseCodes], '\n    ', BaseCode)
    ).

%% generate_base_case_code(+GroupName, +Clause, +MemoEnabled, -Code)
generate_base_case_code(GroupName, clause(Head, true), MemoEnabled, Code) :-
    % Simple fact: pred(Value)
    Head =.. [_Pred, Value],
    % Generate memo store code (if enabled)
    (   MemoEnabled = true ->
        format(string(MemoStore), '~w_memo[\"$key\"]=\"$result\"~n        ', [GroupName])
    ;   MemoStore = ''
    ),
    format(string(Code), 'if [[ \"$arg1\" == \"~w\" ]]; then\n        local result=\"true\"\n        ~wecho \"$result\"\n        return 0\n    fi', [Value, MemoStore]).


%% generate_condition_code(+Body, -CondCode)
generate_condition_code((A, B), Code) :-
    !,
    % Conjunction
    generate_condition_code(A, CodeA),
    generate_condition_code(B, CodeB),
    format(string(Code), '~w && ~w', [CodeA, CodeB]).
generate_condition_code(A == B, Code) :-
    !,
    translate_comparison_term(A, TermA),
    translate_comparison_term(B, TermB),
    format(string(Code), '[[ \"~w\" == \"~w\" ]]', [TermA, TermB]).
generate_condition_code(A =:= B, Code) :-
    !,
    translate_comparison_term(A, TermA),
    translate_comparison_term(B, TermB),
    format(string(Code), '[[ ~w -eq ~w ]]', [TermA, TermB]).
generate_condition_code(A > B, Code) :-
    !,
    translate_comparison_term(A, TermA),
    translate_comparison_term(B, TermB),
    format(string(Code), '[[ ~w -gt ~w ]]', [TermA, TermB]).
generate_condition_code(A < B, Code) :-
    !,
    translate_comparison_term(A, TermA),
    translate_comparison_term(B, TermB),
    format(string(Code), '[[ ~w -lt ~w ]]', [TermA, TermB]).
generate_condition_code(A >= B, Code) :-
    !,
    translate_comparison_term(A, TermA),
    translate_comparison_term(B, TermB),
    format(string(Code), '[[ ~w -ge ~w ]]', [TermA, TermB]).
generate_condition_code(A =< B, Code) :-
    !,
    translate_comparison_term(A, TermA),
    translate_comparison_term(B, TermB),
    format(string(Code), '[[ ~w -le ~w ]]', [TermA, TermB]).
generate_condition_code(_, 'true').

%% translate_comparison_term(+Term, -BashTerm)
translate_comparison_term(Term, '"$arg1"') :-
    var(Term), !.
translate_comparison_term(Term, BashTerm) :-
    atomic(Term),
    format(string(BashTerm), '~w', [Term]).

%% generate_recursive_cases_code(+GroupName, +RecClauses, +MemoEnabled, -RecCode)
generate_recursive_cases_code(GroupName, RecClauses, MemoEnabled, RecCode) :-
    length(RecClauses, NumRec),
    format(string(Header), '# ~w recursive case(s)', [NumRec]),
    (   RecClauses = [] ->
        RecCode = Header
    ;   findall(Code,
            (   member(Clause, RecClauses),
                generate_recursive_case_code(GroupName, Clause, MemoEnabled, Code)
            ),
            RecCodes),
        atomic_list_concat([Header|RecCodes], '\n    ', RecCode)
    ).

%% generate_recursive_case_code(+GroupName, +Clause, +MemoEnabled, -Code)
generate_recursive_case_code(GroupName, clause(Head, Body), MemoEnabled, Code) :-
    % Extract head argument to identify the variable
    Head =.. [_Pred, HeadVar],

    % Parse body to find: conditions, computations, recursive call
    parse_recursive_body(Body, HeadVar, Conditions, Computations, RecCall),

    % Generate bash code with head variable mapping
    generate_recursive_bash(GroupName, HeadVar, Conditions, Computations, RecCall, MemoEnabled, Code).

%% parse_recursive_body(+Body, +HeadVar, -Conditions, -Computations, -RecCall)
parse_recursive_body(Body, HeadVar, Conditions, Computations, RecCall) :-
    extract_goals(Body, Goals),
    partition(is_condition_goal(HeadVar), Goals, Conditions, Rest),
    partition(is_computation_goal, Rest, Computations, RecCallList),
    (   RecCallList = [RecCall|_] -> true
    ;   RecCall = none
    ).

%% extract_goals(+Body, -Goals)
extract_goals((A, B), Goals) :-
    !,
    extract_goals(A, GoalsA),
    extract_goals(B, GoalsB),
    append(GoalsA, GoalsB, Goals).
extract_goals(Goal, [Goal]).

%% is_condition_goal(+HeadVar, +Goal)
is_condition_goal(HeadVar, Goal) :-
    (   Goal = (_ > _)
    ;   Goal = (_ < _)
    ;   Goal = (_ >= _)
    ;   Goal = (_ =< _)
    ;   Goal = (_ =:= _)
    ;   Goal = (_ == _)
    ),
    % Make sure it involves the head variable
    term_variables(Goal, Vars),
    member(V, Vars),
    V == HeadVar.

%% is_computation_goal(+Goal)
is_computation_goal(Goal) :-
    Goal = (_ is _).

%% generate_recursive_bash(+GroupName, +HeadVar, +Conditions, +Computations, +RecCall, +MemoEnabled, -Code)
generate_recursive_bash(GroupName, HeadVar, Conditions, Computations, RecCall, MemoEnabled, Code) :-
    % Generate condition check
    (   Conditions = [] ->
        CondCode = ''
    ;   maplist(generate_condition_code_with_var(HeadVar), Conditions, CondCodes),
        atomic_list_concat(CondCodes, ' && ', CondStr),
        format(string(CondCode), '~w', [CondStr])
    ),

    % Generate computations
    maplist(generate_computation_code_with_var(HeadVar), Computations, CompCodes),
    atomic_list_concat(CompCodes, '\n        ', CompCode),

    % Generate recursive call
    (   RecCall = none ->
        RecCode = ''
    ;   generate_rec_call_code_with_var(HeadVar, RecCall, GroupName, MemoEnabled, RecCode)
    ),

    % Assemble the if-block
    (   CondCode = '' ->
        % No condition - always execute
        format(string(Code), '# Recursive case\n    ~w\n    ~w', [CompCode, RecCode])
    ;   format(string(Code), 'if ~w; then\n        ~w\n        ~w\n    fi', [CondCode, CompCode, RecCode])
    ).

%% generate_condition_code_with_var(+HeadVar, +Goal, -Code)
generate_condition_code_with_var(HeadVar, Goal, Code) :-
    translate_goal_with_var(HeadVar, Goal, Code).

%% generate_computation_code_with_var(+HeadVar, +Goal, -Code)
generate_computation_code_with_var(HeadVar, VarOut is Expr, Code) :-
    translate_expr_with_var(HeadVar, Expr, BashExpr),
    format(atom(VarOutAtom), '~w', [VarOut]),
    atom_string(VarOutAtom, VarOutStr),
    to_lower_case(VarOutStr, VarOutLower),
    format(string(Code), 'local ~w=$(( ~w ))', [VarOutLower, BashExpr]).

%% generate_rec_call_code_with_var(+HeadVar, +RecCall, -Code)
generate_rec_call_code_with_var(HeadVar, RecCall, GroupName, MemoEnabled, Code) :-
    RecCall =.. [Pred|Args],
    atom_string(Pred, PredStr),
    maplist(translate_call_arg_with_var(HeadVar), Args, BashArgs),
    atomic_list_concat(BashArgs, ' ', ArgsStr),
    (   MemoEnabled = true ->
        format(string(MemoStore), '            ~w_memo[\"$key\"]=\"$result\"~n', [GroupName])
    ;   MemoStore = ''
    ),
    format(string(Code),
           'local rec_result=$(~w ~w)\n        if [[ $? -eq 0 && \"$rec_result\" == \"true\" ]]; then\n            local result=\"true\"\n~w            echo \"$result\"\n            return 0\n        else\n            return 1\n        fi',
           [PredStr, ArgsStr, MemoStore]).

%% translate_goal_with_var(+HeadVar, +Goal, -Code)
translate_goal_with_var(HeadVar, Goal, Code) :-
    Goal =.. [Op, A, B],
    member(Op, [>, <, >=, =<, =:=, ==]),
    translate_term_with_var(HeadVar, A, TermA),
    translate_term_with_var(HeadVar, B, TermB),
    bash_comparison_op(Op, BashOp),
    format(string(Code), '[[ ~w ~w ~w ]]', [TermA, BashOp, TermB]).

bash_comparison_op(>, '-gt').
bash_comparison_op(<, '-lt').
bash_comparison_op(>=, '-ge').
bash_comparison_op(=<, '-le').
bash_comparison_op(=:=, '-eq').
bash_comparison_op(==, '==').

%% translate_term_with_var(+HeadVar, +Term, -BashTerm)
translate_term_with_var(HeadVar, Term, '"$arg1"') :-
    Term == HeadVar, !.
translate_term_with_var(_HeadVar, Term, BashTerm) :-
    var(Term), !,
    format(atom(VarAtom), '~w', [Term]),
    atom_string(VarAtom, VarStr),
    to_lower_case(VarStr, VarLower),
    format(string(BashTerm), '"$~w"', [VarLower]).
translate_term_with_var(_HeadVar, Term, BashTerm) :-
    atomic(Term),
    format(string(BashTerm), '~w', [Term]).

%% translate_expr_with_var(+HeadVar, +Expr, -BashExpr)
translate_expr_with_var(HeadVar, Var, '$arg1') :-
    Var == HeadVar, !.
translate_expr_with_var(_HeadVar, Var, BashExpr) :-
    var(Var), !,
    format(atom(VarAtom), '~w', [Var]),
    atom_string(VarAtom, VarStr),
    to_lower_case(VarStr, VarLower),
    format(string(BashExpr), '$~w', [VarLower]).
translate_expr_with_var(_HeadVar, Atom, BashExpr) :-
    atomic(Atom), !,
    format(string(BashExpr), '~w', [Atom]).
translate_expr_with_var(HeadVar, A + B, BashExpr) :-
    !,
    (   (number(B), B < 0) ->
        translate_expr_with_var(HeadVar, A, BA),
        Pos is -B,
        format(string(BashExpr), '~w - ~w', [BA, Pos])
    ;   translate_expr_with_var(HeadVar, A, BA),
        translate_expr_with_var(HeadVar, B, BB),
        format(string(BashExpr), '~w + ~w', [BA, BB])
    ).
translate_expr_with_var(HeadVar, A - B, BashExpr) :-
    !,
    translate_expr_with_var(HeadVar, A, BA),
    translate_expr_with_var(HeadVar, B, BB),
    format(string(BashExpr), '~w - ~w', [BA, BB]).
translate_expr_with_var(HeadVar, A * B, BashExpr) :-
    !,
    translate_expr_with_var(HeadVar, A, BA),
    translate_expr_with_var(HeadVar, B, BB),
    format(string(BashExpr), '~w * ~w', [BA, BB]).

%% translate_call_arg_with_var(+HeadVar, +Arg, -BashArg)
translate_call_arg_with_var(HeadVar, Var, '"$arg1"') :-
    Var == HeadVar, !.
translate_call_arg_with_var(_HeadVar, Var, BashArg) :-
    var(Var), !,
    format(atom(VarAtom), '~w', [Var]),
    atom_string(VarAtom, VarStr),
    to_lower_case(VarStr, VarLower),
    format(string(BashArg), '"$~w"', [VarLower]).
translate_call_arg_with_var(_HeadVar, Atom, BashArg) :-
    atomic(Atom), !,
    format(string(BashArg), '~w', [Atom]).

%% translate_expr(+Expr, -BashExpr)
translate_expr(Var, BashExpr) :-
    var(Var),
    !,
    format(atom(VarAtom), '~w', [Var]),
    atom_string(VarAtom, VarStr),
    to_lower_case(VarStr, VarLower),
    format(string(BashExpr), '$~w', [VarLower]).
translate_expr(Atom, BashExpr) :-
    atomic(Atom),
    !,
    format(string(BashExpr), '~w', [Atom]).
translate_expr(A + B, BashExpr) :-
    !,
    % Handle "N + (-1)" as "N - 1"
    (   (number(B), B < 0) ->
        translate_expr(A, BA),
        Pos is -B,
        format(string(BashExpr), '~w - ~w', [BA, Pos])
    ;   translate_expr(A, BA),
        translate_expr(B, BB),
        format(string(BashExpr), '~w + ~w', [BA, BB])
    ).
translate_expr(A - B, BashExpr) :-
    !,
    translate_expr(A, BA),
    translate_expr(B, BB),
    format(string(BashExpr), '~w - ~w', [BA, BB]).
translate_expr(A * B, BashExpr) :-
    !,
    translate_expr(A, BA),
    translate_expr(B, BB),
    format(string(BashExpr), '~w * ~w', [BA, BB]).
translate_expr(Term, _) :-
    format('ERROR: Unable to translate expression: ~w~n', [Term]),
    fail.

%% to_lower_case(+String, -LowerString)
to_lower_case(Str, Lower) :-
    string_lower(Str, Lower).

%% generate_rec_call_code(+RecCall, -Code)
generate_rec_call_code(RecCall, Code) :-
    RecCall =.. [Pred|Args],
    atom_string(Pred, PredStr),
    maplist(translate_call_arg, Args, BashArgs),
    atomic_list_concat(BashArgs, ' ', ArgsStr),
    format(string(Code), 'local rec_result=$(~w ~w)\n        if [[ $? -eq 0 && \"$rec_result\" == \"true\" ]]; then\n            result=\"true\"\n        else\n            return 1\n        fi', [PredStr, ArgsStr]).

%% translate_call_arg(+Arg, -BashArg)
translate_call_arg(Var, BashArg) :-
    var(Var),
    !,
    format(atom(VarAtom), '~w', [Var]),
    atom_string(VarAtom, VarStr),
    to_lower_case(VarStr, VarLower),
    format(string(BashArg), '"$~w"', [VarLower]).
translate_call_arg(Atom, BashArg) :-
    atomic(Atom),
    !,
    format(string(BashArg), '~w', [Atom]).

%% generate_function_template(+PredStr, +Arity, +GroupName, +MemoEnabled, +BaseCode, +RecCode, -FuncCode)
generate_function_template(PredStr, Arity, GroupName, MemoEnabled, BaseCode, RecCode, FuncCode) :-
    % Generate memo check code (if enabled)
    (   MemoEnabled = true ->
        format(string(MemoCheckCode1), '    # Check shared memo table~n    if [[ -n \"${~w_memo[$key]}\" ]]; then~n        echo \"${~w_memo[$key]}\"~n        return 0~n    fi~n    ', [GroupName, GroupName])
    ;   MemoCheckCode1 = '    # Memoization disabled (unique=false constraint)\n    '
    ),

    TemplateLines = [
        "# {{pred}}/{{arity}} - part of mutual recursion group",
        "{{pred}}() {",
        "    local arg1=\"$1\"",
        "    local key=\"{{pred}}:$*\"",
        "    ",
        "{{memo_check}}",
        "    {{base_code}}",
        "    ",
        "    {{rec_code}}",
        "    ",
        "    # No match found",
        "    return 1",
        "}",
        "",
        "{{pred}}_stream() {",
        "    {{pred}} \"$@\"",
        "}"
    ],

    atomic_list_concat(TemplateLines, '\n', Template),
    render_template(Template, [
        pred=PredStr,
        arity=Arity,
        group=GroupName,
        base_code=BaseCode,
        rec_code=RecCode,
        memo_check=MemoCheckCode1
    ], FuncCode).

%% ============================================
%% TESTS
%% ============================================

test_mutual_recursion :-
    writeln('=== MUTUAL RECURSION COMPILER TESTS ==='),

    % Setup output directory
    (   exists_directory('output/advanced') -> true
    ;   make_directory('output/advanced')
    ),

    % Clear test predicates
    catch(abolish(is_even/1), _, true),
    catch(abolish(is_odd/1), _, true),

    % Test 1: Classic even/odd mutual recursion
    writeln('Test 1: Compile is_even/1 and is_odd/1 (mutually recursive)'),
    assertz(user:is_even(0)),
    assertz(user:(is_even(N) :- N > 0, N1 is N - 1, is_odd(N1))),
    assertz(user:is_odd(1)),
    assertz(user:(is_odd(N) :- N > 1, N1 is N - 1, is_even(N1))),

    Predicates1 = [is_even/1, is_odd/1],

    (   can_compile_mutual_recursion(Predicates1) ->
        writeln('  ✓ Mutual recursion detected'),
        compile_mutual_recursion(Predicates1, [], Code1),
        write_bash_file('output/advanced/even_odd.sh', Code1),
        writeln('  ✓ Compiled to output/advanced/even_odd.sh')
    ;   writeln('  ✗ FAIL - should detect mutual recursion')
    ),

    % Test 2: Check non-mutual case
    writeln('Test 2: Verify non-mutual predicates are rejected'),
    catch(abolish(independent1/1), _, true),
    catch(abolish(independent2/1), _, true),
    assertz(user:independent1(a)),
    assertz(user:independent2(b)),

    Predicates2 = [independent1/1, independent2/1],
    (   can_compile_mutual_recursion(Predicates2) ->
        writeln('  ✗ FAIL - should not detect mutual recursion')
    ;   writeln('  ✓ PASS - correctly rejected non-mutual predicates')
    ),

    writeln('=== MUTUAL RECURSION COMPILER TESTS COMPLETE ===').

%% Helper to write bash files
write_bash_file(Path, Content) :-
    open(Path, write, Stream),
    write(Stream, Content),
    close(Stream),
    % Make executable
    atom_concat('chmod +x ', Path, ChmodCmd),
    shell(ChmodCmd).
