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

%% compile_mutual_recursion(+Predicates, +Options, -BashCode)
%  Compile a group of mutually recursive predicates
%  Generates bash code with shared memoization
%  Options: List of Key=Value pairs (e.g., [unique=true, ordered=false])
%  Currently mutual recursive predicates return single values (not sets),
%  so deduplication constraints don't apply. Options are reserved for
%  future use (e.g., output language selection).
compile_mutual_recursion(Predicates, Options, BashCode) :-
    format('  Compiling mutual recursion group: ~w~n', [Predicates]),

    % Query constraints for the first predicate (all share same group constraints)
    (   Predicates = [FirstPred|_] ->
        get_constraints(FirstPred, Constraints),
        format('  Constraints: ~w~n', [Constraints]),

        % Merge runtime options with constraints
        append(Options, Constraints, _AllOptions),
        format('  Final options: ~w~n', [Options])
    ;   true
    ),

    % TODO: Mutually recursive patterns return single values, not sets.
    % Deduplication constraints (unique, unordered) may not apply here.
    % Options are kept for future extensibility (e.g., output_lang=python).

    % Generate bash code for the group
    generate_mutual_recursion_bash(Predicates, BashCode).

%% generate_mutual_recursion_bash(+Predicates, -BashCode)
generate_mutual_recursion_bash(Predicates, BashCode) :-
    % Extract predicate names
    findall(PredStr,
        (   member(Pred/_Arity, Predicates),
            atom_string(Pred, PredStr)
        ),
        PredStrs),
    atomic_list_concat(PredStrs, '_', GroupName),

    % Generate header
    generate_mutual_header(GroupName, HeaderCode),

    % Generate function for each predicate
    findall(FuncCode,
        (   member(Pred/Arity, Predicates),
            generate_mutual_function(Pred, Arity, GroupName, Predicates, FuncCode)
        ),
        FuncCodes),
    atomic_list_concat(FuncCodes, '\n\n', FunctionsCode),

    % Combine
    atomic_list_concat([HeaderCode, FunctionsCode], '\n\n', BashCode).

%% generate_mutual_header(+GroupName, -HeaderCode)
generate_mutual_header(GroupName, HeaderCode) :-
    TemplateLines = [
        "#!/bin/bash",
        "# Mutually recursive group: {{group}}",
        "# Shared memoization for mutual recursion",
        "",
        "# Shared memo table for all predicates in this group",
        "declare -gA {{group}}_memo"
    ],

    atomic_list_concat(TemplateLines, '\n', Template),
    render_template(Template, [group=GroupName], HeaderCode).

%% generate_mutual_function(+Pred, +Arity, +GroupName, +AllPredicates, -FuncCode)
%  Generate bash function for one predicate in mutual group
generate_mutual_function(Pred, Arity, GroupName, AllPredicates, FuncCode) :-
    atom_string(Pred, PredStr),

    % Get clauses for this predicate - use user:clause to access predicates from any module (including test predicates)
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),

    % Separate base and recursive cases (check for calls to any predicate in the group)
    partition(is_mutual_recursive_clause(AllPredicates), Clauses, RecClauses, BaseClauses),

    % Generate base case code
    generate_base_cases_code(GroupName, BaseClauses, BaseCode),

    % Generate recursive case code
    generate_recursive_cases_code(GroupName, RecClauses, RecCode),

    % Generate function template
    generate_function_template(PredStr, Arity, GroupName, BaseCode, RecCode, FuncCode).

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

%% generate_base_cases_code(+GroupName, +BaseClauses, -BaseCode)
generate_base_cases_code(GroupName, BaseClauses, BaseCode) :-
    length(BaseClauses, NumBase),
    format(string(Header), '# ~w base case(s)', [NumBase]),
    (   BaseClauses = [] ->
        BaseCode = Header
    ;   findall(Code,
            (   member(Clause, BaseClauses),
                generate_base_case_code(GroupName, Clause, Code)
            ),
            BaseCodes),
        atomic_list_concat([Header|BaseCodes], '\n    ', BaseCode)
    ).

%% generate_base_case_code(+GroupName, +Clause, -Code)
generate_base_case_code(GroupName, clause(Head, true), Code) :-
    % Simple fact: pred(Value)
    Head =.. [_Pred, Value],
    format(string(CodeTemplate), 'if [[ \"$arg1\" == \"~w\" ]]; then\n        local result=\"true\"\n        {{group}}_memo[\"$key\"]=\"$result\"\n        echo \"$result\"\n        return 0\n    fi', [Value]),
    render_template(CodeTemplate, [group=GroupName], Code).


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

%% generate_recursive_cases_code(+GroupName, +RecClauses, -RecCode)
generate_recursive_cases_code(GroupName, RecClauses, RecCode) :-
    length(RecClauses, NumRec),
    format(string(Header), '# ~w recursive case(s)', [NumRec]),
    (   RecClauses = [] ->
        RecCode = Header
    ;   findall(Code,
            (   member(Clause, RecClauses),
                generate_recursive_case_code(GroupName, Clause, Code)
            ),
            RecCodes),
        atomic_list_concat([Header|RecCodes], '\n    ', RecCode)
    ).

%% generate_recursive_case_code(+GroupName, +Clause, -Code)
generate_recursive_case_code(GroupName, clause(Head, Body), Code) :-
    % Extract head argument to identify the variable
    Head =.. [_Pred, HeadVar],

    % Parse body to find: conditions, computations, recursive call
    parse_recursive_body(Body, HeadVar, Conditions, Computations, RecCall),

    % Generate bash code with head variable mapping
    generate_recursive_bash(GroupName, HeadVar, Conditions, Computations, RecCall, Code).

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

%% generate_recursive_bash(+GroupName, +HeadVar, +Conditions, +Computations, +RecCall, -Code)
generate_recursive_bash(GroupName, HeadVar, Conditions, Computations, RecCall, Code) :-
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
    ;   generate_rec_call_code_with_var(HeadVar, RecCall, RecCode)
    ),

    % Assemble the if-block
    (   CondCode = '' ->
        % No condition - always execute
        format(string(CodeTemplate), '# Recursive case\n    ~w\n    ~w\n    local result=\"true\"\n    {{group}}_memo[\"$key\"]=\"$result\"\n    echo \"$result\"\n    return 0', [CompCode, RecCode])
    ;   format(string(CodeTemplate), 'if ~w; then\n        ~w\n        ~w\n        local result=\"true\"\n        {{group}}_memo[\"$key\"]=\"$result\"\n        echo \"$result\"\n        return 0\n    fi', [CondCode, CompCode, RecCode])
    ),
    render_template(CodeTemplate, [group=GroupName], Code).

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
generate_rec_call_code_with_var(HeadVar, RecCall, Code) :-
    RecCall =.. [Pred|Args],
    atom_string(Pred, PredStr),
    maplist(translate_call_arg_with_var(HeadVar), Args, BashArgs),
    atomic_list_concat(BashArgs, ' ', ArgsStr),
    format(string(Code), 'local rec_result=$(~w ~w)\n        if [[ $? -eq 0 && \"$rec_result\" == \"true\" ]]; then\n            result=\"true\"\n        else\n            return 1\n        fi', [PredStr, ArgsStr]).

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

%% generate_function_template(+PredStr, +Arity, +GroupName, +BaseCode, +RecCode, -FuncCode)
generate_function_template(PredStr, Arity, GroupName, BaseCode, RecCode, FuncCode) :-
    TemplateLines = [
        "# {{pred}}/{{arity}} - part of mutual recursion group",
        "{{pred}}() {",
        "    local arg1=\"$1\"",
        "    local key=\"{{pred}}:$*\"",
        "    ",
        "    # Check shared memo table",
        "    if [[ -n \"${{{group}}_memo[$key]}\" ]]; then",
        "        echo \"${{{group}}_memo[$key]}\"",
        "        return 0",
        "    fi",
        "    ",
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
        rec_code=RecCode
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
