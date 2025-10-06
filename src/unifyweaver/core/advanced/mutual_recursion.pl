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
        append(Options, Constraints, AllOptions),
        format('  Final options: ~w~n', [AllOptions])
    ;   AllOptions = Options
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
            generate_mutual_function(Pred, Arity, GroupName, FuncCode)
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

%% generate_mutual_function(+Pred, +Arity, +GroupName, -FuncCode)
%  Generate bash function for one predicate in mutual group
generate_mutual_function(Pred, Arity, GroupName, FuncCode) :-
    atom_string(Pred, PredStr),

    % Get clauses for this predicate - use user:clause to access predicates from any module (including test predicates)
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),

    % Separate base and recursive cases
    partition(is_recursive_clause_any(Pred), Clauses, RecClauses, BaseClauses),

    % Generate base case code
    generate_base_cases_code(PredStr, BaseClauses, BaseCode),

    % Generate recursive case code
    generate_recursive_cases_code(PredStr, RecClauses, RecCode),

    % Generate function template
    generate_function_template(PredStr, Arity, GroupName, BaseCode, RecCode, FuncCode).

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

%% generate_base_cases_code(+PredStr, +BaseClauses, -BaseCode)
generate_base_cases_code(_PredStr, BaseClauses, BaseCode) :-
    length(BaseClauses, NumBase),
    format(string(BaseCode), '# ~w base case(s)\n    # TODO: Implement base cases', [NumBase]).

%% generate_recursive_cases_code(+PredStr, +RecClauses, -RecCode)
generate_recursive_cases_code(_PredStr, RecClauses, RecCode) :-
    length(RecClauses, NumRec),
    format(string(RecCode), '# ~w recursive case(s)\n    # TODO: Implement recursive cases', [NumRec]).

%% generate_function_template(+PredStr, +Arity, +GroupName, +BaseCode, +RecCode, -FuncCode)
generate_function_template(PredStr, Arity, GroupName, BaseCode, RecCode, FuncCode) :-
    TemplateLines = [
        "# {{pred}}/{{arity}} - part of mutual recursion group",
        "{{pred}}() {",
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
        "    # Cache and return result",
        "    local result=\"$*\"  # Placeholder",
        "    {{group}}_memo[\"$key\"]=\"$result\"",
        "    echo \"$result\"",
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
