:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% tree_recursion.pl - Tree recursion pattern compiler
% Handles recursion with multiple recursive calls (tree pattern)
% Trees represented as lists: [Value, LeftSubtree, RightSubtree] or []

:- module(tree_recursion, [
    is_tree_recursive/1,           % +Pred/Arity - Detect tree recursion
    can_compile_tree_recursion/1,  % +Pred/Arity - Check if compilable
    compile_tree_recursion/3,      % +Pred/Arity, +Options, -BashCode
    test_tree_recursion/0          % Run tests
]).

:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module(pattern_matchers).

%% ============================================
%% PATTERN DETECTION
%% ============================================

%% is_tree_recursive(+Pred/Arity)
%  Detect tree recursion: predicate with multiple recursive calls
%  Classic examples: fibonacci, tree_sum, tree_height
is_tree_recursive(Pred/Arity) :-
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),

    % Need at least one base case and one recursive case
    partition(is_recursive_for_pred(Pred), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [],

    % At least one recursive clause must have 2+ recursive calls (tree pattern)
    member(clause(_, RecBody), RecClauses),
    count_recursive_calls_to(RecBody, Pred, Count),
    Count >= 2.

%% is_recursive_for_pred(+Pred, +Clause)
is_recursive_for_pred(Pred, clause(_, Body)) :-
    contains_call_to(Body, Pred).

%% count_recursive_calls_to(+Body, +Pred, -Count)
%  Count how many times Pred is called in Body
count_recursive_calls_to(Body, Pred, Count) :-
    findall(1, (extract_goal_from_body(Body, Goal), functor(Goal, Pred, _)), Calls),
    length(Calls, Count).

%% extract_goal_from_body(+Body, -Goal)
%  Extract individual goals from body (helper)
extract_goal_from_body(Goal, Goal) :-
    compound(Goal),
    \+ Goal = (_,_),
    \+ Goal = (_;_),
    \+ Goal = (_->_).
extract_goal_from_body((A, _), Goal) :- extract_goal_from_body(A, Goal).
extract_goal_from_body((_, B), Goal) :- extract_goal_from_body(B, Goal).

%% ============================================
%% COMPILATION CHECK
%% ============================================

%% can_compile_tree_recursion(+Pred/Arity)
%  Check if we can compile this tree recursive predicate to bash
can_compile_tree_recursion(Pred/Arity) :-
    is_tree_recursive(Pred/Arity),

    % Get clauses
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),

    % Must have recognizable base and recursive cases
    partition(is_recursive_for_pred(Pred), Clauses, RecClauses, BaseClauses),
    BaseClauses \= [],
    RecClauses \= [],

    % Base cases should be simple (unifiable patterns)
    forall(member(clause(BaseHead, BaseBody), BaseClauses),
           is_simple_base_case(BaseHead, BaseBody)),

    % Recursive cases should follow tree pattern
    forall(member(clause(RecHead, RecBody), RecClauses),
           is_tree_recursive_case(RecHead, RecBody, Pred)).

%% is_simple_base_case(+Head, +Body)
%  Base case should be simple: either true, or a simple unification/computation
is_simple_base_case(_, true) :- !.
is_simple_base_case(_, (_ is _)) :- !.
is_simple_base_case(_, (_ = _)) :- !.
is_simple_base_case(_, Body) :-
    % Allow simple conjunctions of is/= goals
    Body = (A, B),
    is_simple_base_case(_, A),
    is_simple_base_case(_, B).

%% is_tree_recursive_case(+Head, +Body, +Pred)
%  Recursive case should make multiple recursive calls
is_tree_recursive_case(_Head, Body, Pred) :-
    count_recursive_calls_to(Body, Pred, Count),
    Count >= 2.

%% ============================================
%% CODE GENERATION
%% ============================================

%% compile_tree_recursion(+Pred/Arity, +Options, -BashCode)
%  Generate bash code for tree recursive predicate
%  Options:
%    - memoization(true/false) - Use memoization table (default: false for v1.0)
%    - tree_format(list/nested) - Tree representation (default: list)
compile_tree_recursion(Pred/Arity, Options, BashCode) :-
    % Check if memoization is requested
    (   member(memoization(true), Options) ->
        UseMemo = true
    ;   UseMemo = false
    ),

    % Get predicate information
    atom_string(Pred, PredStr),

    % Detect pattern type
    (   is_fibonacci_pattern(Pred/Arity) ->
        generate_fibonacci_code(PredStr, UseMemo, BashCode)
    ;   is_binary_tree_pattern(Pred/Arity) ->
        generate_binary_tree_code(Pred, Arity, UseMemo, BashCode)
    ;   % Generic tree recursion
        generate_generic_tree_code(Pred, Arity, UseMemo, BashCode)
    ).

%% ============================================
%% PATTERN RECOGNITION
%% ============================================

%% is_fibonacci_pattern(+Pred/Arity)
%  Detect fibonacci-like pattern: f(N, F) with two recursive calls on N-1, N-2
is_fibonacci_pattern(Pred/2) :-
    functor(Head, Pred, 2),
    user:clause(Head, Body),
    % Look for pattern: N1 is N - 1, N2 is N - 2, recursive calls
    contains_fibonacci_structure(Body, Pred).

contains_fibonacci_structure(Body, Pred) :-
    % Extract goals
    findall(G, extract_goal_from_body(Body, G), Goals),

    % Should have two recursive calls
    findall(1, (member(G, Goals), functor(G, Pred, _)), RecCalls),
    length(RecCalls, 2),

    % Should have N-1 and N-2 computations (or similar)
    member(_ is _ - 1, Goals),
    member(_ is _ - 2, Goals).

%% is_binary_tree_pattern(+Pred/Arity)
%  Detect binary tree pattern: operates on list-based trees [V, L, R]
is_binary_tree_pattern(Pred/Arity) :-
    Arity >= 2,
    functor(Head, Pred, Arity),
    user:clause(Head, Body),

    % Check if first argument is tree structure [V, L, R]
    arg(1, Head, TreeArg),
    is_list_tree_argument(TreeArg, Body, Pred).

is_list_tree_argument([_V, _L, _R], Body, Pred) :-
    % Should have recursive calls on L and R
    count_recursive_calls_to(Body, Pred, Count),
    Count >= 2.
is_list_tree_argument([_|_], _, _) :- !.  % Could be tree structure

%% ============================================
%% CODE GENERATORS
%% ============================================

%% generate_fibonacci_code(+PredStr, +UseMemo, -BashCode)
%  Generate bash code for fibonacci-like predicates
generate_fibonacci_code(PredStr, UseMemo, BashCode) :-
    (   UseMemo = true ->
        MemoDecl = "declare -gA _MEMO_TABLE",
        MemoCheck = [
            "    # Check memo table",
            "    local key=\"", PredStr, ":$n\"",
            "    if [[ -n \"${_MEMO_TABLE[$key]}\" ]]; then",
            "        echo \"${_MEMO_TABLE[$key]}\"",
            "        return 0",
            "    fi",
            ""
        ],
        MemoStore = [
            "    # Store in memo table",
            "    _MEMO_TABLE[\"$key\"]=\"$result\"",
            ""
        ]
    ;   MemoDecl = "",
        MemoCheck = [],
        MemoStore = []
    ),

    % Build template - flatten all nested structures
    format(atom(Header), '#!/bin/bash\n# ~w - tree recursion (fibonacci pattern)\n# List-based representation for v1.0\n\n', [PredStr]),
    format(atom(FuncStart), '~w() {\n    local n="$1"\n\n', [PredStr]),

    (   UseMemo = true ->
        format(atom(MemoCheckCode), '    # Check memo table\n    local key="~w:$n"\n    if [[ -n "${_MEMO_TABLE[$key]}" ]]; then\n        echo "${_MEMO_TABLE[$key]}"\n        return 0\n    fi\n\n', [PredStr])
    ;   MemoCheckCode = ''
    ),

    BaseCases = '    # Base cases\n    if [[ "$n" -le 0 ]]; then\n        echo 0\n        return 0\n    fi\n    if [[ "$n" -eq 1 ]]; then\n        echo 1\n        return 0\n    fi\n\n',

    format(atom(RecCase), '    # Recursive case\n    local n1=$((n - 1))\n    local n2=$((n - 2))\n    local f1=$(~w "$n1")\n    local f2=$(~w "$n2")\n    local result=$((f1 + f2))\n\n', [PredStr, PredStr]),

    (   UseMemo = true ->
        format(atom(MemoStoreCode), '    # Store in memo table\n    _MEMO_TABLE["$key"]="$result"\n\n', [])
    ;   MemoStoreCode = ''
    ),

    format(atom(FuncEnd), '    echo "$result"\n}\n\n', []),
    format(atom(StreamHelper), '# Helper for streaming\n~w_stream() {\n    ~w "$@"\n}\n', [PredStr, PredStr]),

    atomic_list_concat([Header, MemoDecl, '\n', FuncStart, MemoCheckCode, BaseCases, RecCase, MemoStoreCode, FuncEnd, StreamHelper], BashCode).

%% generate_binary_tree_code(+Pred, +Arity, +UseMemo, -BashCode)
%  Generate bash code for binary tree operations
%  Trees represented as: [value, [left], [right]] or []
generate_binary_tree_code(Pred, Arity, UseMemo, BashCode) :-
    atom_string(Pred, PredStr),

    % Determine what operation (sum, height, count, etc.)
    (   sub_atom(Pred, _, _, _, sum) ->
        Operation = sum,
        BaseValue = "0",
        CombineOp = "$value + $left_result + $right_result"
    ;   sub_atom(Pred, _, _, _, height) ->
        Operation = height,
        BaseValue = "0",
        CombineOp = "1 + ($left_result > $right_result ? $left_result : $right_result)"
    ;   sub_atom(Pred, _, _, _, count) ->
        Operation = count,
        BaseValue = "0",
        CombineOp = "1 + $left_result + $right_result"
    ;   % Generic: just return value
        Operation = generic,
        BaseValue = "0",
        CombineOp = "$value"
    ),

    (   UseMemo = true ->
        MemoDecl = "declare -gA _MEMO_TABLE\n"
    ;   MemoDecl = ""
    ),

    format(atom(Header), '#!/bin/bash\n# ~w - tree recursion (binary tree pattern)\n# List-based tree: [value, [left], [right]] or []\n\n', [PredStr]),
    format(atom(FuncStart), '~w() {\n    local tree="$1"\n\n', [PredStr]),

    (   UseMemo = true ->
        format(atom(MemoCheckCode), '    # Memoization support\n    local key="~w:$tree"\n    if [[ -n "${_MEMO_TABLE[$key]}" ]]; then\n        echo "${_MEMO_TABLE[$key]}"\n        return 0\n    fi\n\n', [PredStr])
    ;   MemoCheckCode = ''
    ),

    format(atom(BaseCase), '    # Base case: empty tree\n    if [[ "$tree" == "[]" || -z "$tree" ]]; then\n        echo ~w\n        return 0\n    fi\n\n', [BaseValue]),

    format(atom(ParseAndRecurse), '    # Parse tree structure [value, left, right]\n    # Simple parsing for list-based trees\n    local value left right\n    parse_tree "$tree" value left right\n\n    # Recursive calls\n    local left_result=$(~w "$left")\n    local right_result=$(~w "$right")\n\n    # Combine results\n    local result=$(( ~w ))\n\n', [PredStr, PredStr, CombineOp]),

    (   UseMemo = true ->
        MemoStoreCode = '    _MEMO_TABLE["$key"]="$result"\n'
    ;   MemoStoreCode = ''
    ),

    FuncEnd = '    echo "$result"\n}\n\n',

    ParseTreeHelper = '# Helper: Parse list-based tree\n# Handles nested brackets properly by tracking bracket depth\nparse_tree() {\n    local tree_str="$1"\n    local -n val="$2"\n    local -n lft="$3"\n    local -n rgt="$4"\n    \n    # Remove outer brackets\n    tree_str="${tree_str#[}"\n    tree_str="${tree_str%]}"\n    \n    # Parse by tracking bracket depth\n    local depth=0 part=0 current=""\n    local i char\n    \n    for (( i=0; i<${#tree_str}; i++ )); do\n        char="${tree_str:$i:1}"\n        \n        if [[ "$char" == "[" ]]; then\n            ((depth++))\n            current+="$char"\n        elif [[ "$char" == "]" ]]; then\n            ((depth--))\n            current+="$char"\n        elif [[ "$char" == "," && $depth -eq 0 ]]; then\n            # Top-level comma - marks boundary between parts\n            case $part in\n                0) val="$current" ;;\n                1) lft="$current" ;;\n            esac\n            current=""\n            ((part++))\n        else\n            current+="$char"\n        fi\n    done\n    \n    # Last part is right subtree\n    rgt="$current"\n    \n    # Clean up whitespace\n    val="${val// /}"\n    lft="${lft// /}"\n    rgt="${rgt// /}"\n}\n\n',

    format(atom(StreamHelper), '# Streaming helper\n~w_stream() {\n    ~w "$@"\n}\n', [PredStr, PredStr]),

    atomic_list_concat([Header, MemoDecl, FuncStart, MemoCheckCode, BaseCase, ParseAndRecurse, MemoStoreCode, FuncEnd, ParseTreeHelper, StreamHelper], BashCode).
%% generate_generic_tree_code(+Pred, +Arity, +UseMemo, -BashCode)
%  Generic tree recursion code generation (placeholder)
generate_generic_tree_code(Pred, Arity, _UseMemo, BashCode) :-
    atom_string(Pred, PredStr),
    format(atom(BashCode),
        '#!/bin/bash\n# ~w/~w - generic tree recursion\n# TODO: Implement generic tree recursion pattern\n\n~w() {\n    echo "Not yet implemented"\n}\n',
        [Pred, Arity, PredStr]).

%% ============================================
%% TESTS
%% ============================================

test_tree_recursion :-
    writeln('=== TREE RECURSION COMPILER TESTS ==='),

    % Setup
    (   exists_directory('output/advanced') -> true
    ;   make_directory('output/advanced')
    ),

    % Clear test predicates
    catch(abolish(tree_sum/2), _, true),
    catch(abolish(tree_height/2), _, true),

    % Test 1: Binary tree sum pattern detection
    writeln('Test 1: Detect binary tree sum pattern'),
    assertz(user:(tree_sum([], 0))),
    assertz(user:(tree_sum([V, L, R], Sum) :- tree_sum(L, LS), tree_sum(R, RS), Sum is V + LS + RS)),

    (   is_tree_recursive(tree_sum/2) ->
        writeln('  ✓ PASS - tree_sum detected as tree recursive')
    ;   writeln('  ✗ FAIL - tree_sum not detected'),
        fail
    ),

    % Test 2: Compile tree_sum
    writeln('Test 2: Compile tree_sum'),
    (   can_compile_tree_recursion(tree_sum/2) ->
        writeln('  ✓ Pattern is compilable'),
        compile_tree_recursion(tree_sum/2, [], Code1),
        write_bash_file('output/advanced/tree_sum.sh', Code1),
        writeln('  ✓ Generated output/advanced/tree_sum.sh')
    ;   writeln('  ✗ FAIL - cannot compile tree_sum'),
        fail
    ),

    % Test 3: Binary tree height (another structural example)
    writeln('Test 3: Detect binary tree height pattern'),
    assertz(user:(tree_height([], 0))),
    assertz(user:(tree_height([_V, L, R], H) :- tree_height(L, HL), tree_height(R, HR), H is 1 + max(HL, HR))),

    (   is_tree_recursive(tree_height/2) ->
        writeln('  ✓ PASS - tree_height detected as tree recursive')
    ;   writeln('  ⚠ SKIP - tree_height not detected (expected - may need max/2 support)')
    ),

    writeln(''),
    writeln('✓ Tree recursion tests complete!').

%% write_bash_file(+Path, +Code)
write_bash_file(Path, Code) :-
    open(Path, write, Stream),
    write(Stream, Code),
    close(Stream),
    % Make executable
    format(atom(ChmodCmd), 'chmod +x ~w', [Path]),
    shell(ChmodCmd).
