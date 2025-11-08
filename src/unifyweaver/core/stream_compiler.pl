:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% stream_compiler.pl - UnifyWeaver streaming compiler for non-recursive predicates
% Compiles Prolog facts and rules into bash streaming pipelines
% Handles: facts, single rules, multiple rules (OR), inequality constraints
% Does not handle: recursion, negation, complex built-ins

:- module(stream_compiler, [
    compile_predicate/3,
    compile_predicate/2,
    compile_facts/4,         % Export for testing
    test_stream_compiler/0
]).

:- use_module(library(lists)).
:- use_module('constraint_analyzer').
:- use_module('template_system').

%% Main compilation entry point
compile_predicate(Pred/Arity, Options) :-
    compile_predicate(Pred/Arity, Options, _).

compile_predicate(PredIndicator, Options, BashCode) :-
    PredIndicator = PredAtom/Arity,
    format('=== Compiling ~w/~w ===~n', [PredAtom, Arity]),

    % Get constraints for this predicate (from declarations or defaults)
    get_constraints(PredAtom/Arity, Constraints),
    format('  Constraints: ~w~n', [Constraints]),

    % Merge with any runtime options (options take precedence)
    merge_options(Options, Constraints, MergedOptions),

    % Create head with correct arity
    functor(Head, PredAtom, Arity),

    % Get all clauses for this predicate
    findall(Body, clause(Head, Body), Bodies),
    length(Bodies, NumClauses),
    
    % Determine compilation strategy
    (   Bodies = [] ->
        format('ERROR: No clauses found for ~w/~w~n', [PredAtom, Arity]),
        fail
    ;   Bodies = [true|Rest], forall(member(B, Rest), B = true) ->
        % All bodies are just 'true' - these are facts
        format('Type: facts (~w clauses)~n', [NumClauses]),
        compile_facts(PredAtom, Arity, MergedOptions, BashCode)
    ;   Bodies = [SingleBody], SingleBody \= true ->
        % Single rule
        format('Type: single_rule (~w clauses)~n', [NumClauses]),
        extract_predicates(SingleBody, Predicates),
        format('  Body predicates: ~w~n', [Predicates]),
        compile_single_rule(PredAtom, SingleBody, MergedOptions, BashCode)
    ;   % Multiple rules (OR pattern)
        format('Type: multiple_rules (~w clauses)~n', [NumClauses]),
        format('  ~w alternatives~n', [NumClauses]),
        compile_multiple_rules(PredAtom, Bodies, MergedOptions, BashCode)
    ).

%% Merge runtime options with constraint-based options
%  Runtime options take precedence over constraints
merge_options(RuntimeOpts, Constraints, Merged) :-
    % Extract unique and unordered from constraints
    (member(unique(ConstraintUnique), Constraints) -> true ; ConstraintUnique = true),
    (member(unordered(ConstraintUnordered), Constraints) -> true ; ConstraintUnordered = true),

    % Check if runtime overrides exist, otherwise use constraint values
    (member(unique(RuntimeUnique), RuntimeOpts) -> FinalUnique = RuntimeUnique ; FinalUnique = ConstraintUnique),
    (member(unordered(RuntimeUnordered), RuntimeOpts) -> FinalUnordered = RuntimeUnordered ; FinalUnordered = ConstraintUnordered),

    % Build merged list with final values
    Merged = [unique(FinalUnique), unordered(FinalUnordered)].

%% Extract predicates from a body
extract_predicates(true, []) :- !.
extract_predicates((A, B), Predicates) :- !,
    extract_predicates(A, P1),
    extract_predicates(B, P2),
    append(P1, P2, Predicates).
extract_predicates(\+ A, Predicates) :- !,
    % For negation, we still need to know about the predicates inside.
    extract_predicates(A, Predicates).
% Guard against variables in Goal - treat as producing no predicates
extract_predicates(Goal, []) :-
    var(Goal), !.
% Skip inequality and other arithmetic operators - they are handled by the shell.
extract_predicates(_ \= _, []) :- !.
extract_predicates(\=(_, _), []) :- !.
extract_predicates(_ > _, []) :- !.
extract_predicates(_ < _, []) :- !.
extract_predicates(_ >= _, []) :- !.
extract_predicates(_ =< _, []) :- !.
extract_predicates(is(_, _), []) :- !.
extract_predicates(Goal, [Pred]) :-
    functor(Goal, Pred, _),
    Pred \= ',',
    Pred \= true.

%% Compile facts into bash lookup
compile_facts(Pred, Arity, Options, BashCode) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, Arity),
    
    % Collect all facts
    findall(Head, clause(Head, true), Facts),
    
    % Get deduplication strategy from options
    get_dedup_strategy(Options, Strategy),
    
    % Handle different strategies
    (   Strategy = no_dedup ->
        % For no_dedup, use regular array to preserve duplicates
        compile_facts_no_dedup(Pred, Arity, Facts, PredStr, BashCode)
    ;   % For other strategies, use associative array
        % Build array entries
        findall(Entry,
            (   member(Fact, Facts),
                Fact =.. [_|Args],
                format_fact_entry(Args, Entry)
            ),
            Entries),
        atomic_list_concat(Entries, '\n    ', EntriesStr),
        
        % Render template with associative array
        compose_templates(
            ['bash/header','facts/array_binary','facts/lookup_binary','facts/stream_binary', 'facts/reverse_stream_binary'],
            [pred=PredStr, entries=EntriesStr, strategy=Strategy],
            BashCode
        )
    ),
    !.

%% compile_facts_no_dedup(+Pred, +Arity, +Facts, +PredStr, -BashCode)
%  Compile facts without deduplication using regular array
compile_facts_no_dedup(_Pred, Arity, Facts, PredStr, BashCode) :-
    % Build array entries as strings (not key=value pairs)
    findall(Entry,
        (   member(Fact, Facts),
            Fact =.. [_|Args],
            atomic_list_concat(Args, ':', Entry)
        ),
        Entries),
    atomic_list_concat(Entries, '"\n    "', EntriesStr),
    
    % Generate bash code with regular array
    (   Arity = 1 ->
        format(string(BashCode), '#!/bin/bash
# ~s - fact lookup (no deduplication)
~s_data=(
    "~s"
)
~s() {
  local query="$1"
  for item in "${~s_data[@]}"; do
    [[ "$item" == "$query" ]] && echo "$item"
  done
}
~s_stream() {
  for item in "${~s_data[@]}"; do
    echo "$item"
  done
}
# Execute stream function when script is run directly
~s_stream', [PredStr, PredStr, EntriesStr, PredStr, PredStr, PredStr, PredStr, PredStr])
    ;   Arity = 2 ->
        format(string(BashCode), '#!/bin/bash
# ~s - fact lookup (no deduplication)
~s_data=(
    "~s"
)
~s() {
  local key="$1:$2"
  for item in "${~s_data[@]}"; do
    [[ "$item" == "$key" ]] && echo "$item"
  done
}
~s_stream() {
  for item in "${~s_data[@]}"; do
    echo "$item"
  done
}
~s_reverse_stream() {
  for item in "${~s_data[@]}"; do
    IFS=":" read -r a b <<< "$item"
    echo "$b:$a"
  done
}
# Execute stream function when script is run directly
~s_stream', [PredStr, PredStr, EntriesStr, PredStr, PredStr, PredStr, PredStr, PredStr, PredStr, PredStr])
    ;   % For higher arities, use a generic approach
        format(string(BashCode), '#!/bin/bash
# ~s - fact lookup (no deduplication)
~s_data=(
    "~s"
)
~s() {
  echo "Error: ~s with arity ~w not yet supported for no_dedup" >&2
  return 1
}
~s_stream() {
  for item in "${~s_data[@]}"; do
    echo "$item"
  done
}', [PredStr, PredStr, EntriesStr, PredStr, PredStr, Arity, PredStr, PredStr])
    ).

%% format_fact_entry(+Args, -Entry)
format_fact_entry(Args, Entry) :-
    atomic_list_concat(Args, ':', Key),
    format(string(Entry), '[~w]=1', [Key]).

% Helper: materialize facts into a bash array literal body as lines.
gather_fact_entries(Pred, 1, EntriesStr) :-
    functor(Head, Pred, 1),
    findall(Line, (clause(Head, true),
                   Head =.. [Pred, A1],
                   shell_quote(A1, Q1),
                   format(string(Line), "  \"~w\"", [Q1])),
            Lines),
    atomic_list_concat(Lines, "\n", EntriesStr).
gather_fact_entries(Pred, 2, EntriesStr) :-
    functor(Head, Pred, 2),
    findall(Line, (clause(Head, true),
                   Head =.. [Pred, A1, A2],
                   shell_quote(A1, Q1),
                   shell_quote(A2, Q2),
                   format(string(Line), "  [\"~w:~w\"]=1", [Q1, Q2])),
            Lines),
    atomic_list_concat(Lines, "\n", EntriesStr).

% Minimal shell quoting for literals used inside array items.
shell_quote(Atom, Q) :-
    atom_string(Atom, S0),
    replace_in_string(S0, "\"", "\\\"", S1),
    replace_in_string(S1, "\n", "\\n", Q).

replace_in_string(String, Find, Replace, Result) :-
    atomic_list_concat(Split, Find, String),
    atomic_list_concat(Split, Replace, Result).

%% compile_facts_debug(+Pred, +Arity, +_MergedOptions, -BashCode)
%  Debug version of compile_facts with extensive logging.
compile_facts_debug(Pred, Arity, _MergedOptions, BashCode) :-
    format('DEBUG: Starting compile_facts for ~w/~w~n', [Pred, Arity]),
    atom_string(Pred, PredStr),
    format('DEBUG: PredStr = ~w~n', [PredStr]),
    
    functor(Head, Pred, Arity),
    findall(Head, clause(Head, true), Facts),
    format('DEBUG: Found ~w facts~n', [length(Facts, _)]),
    
    % Try template rendering with explicit error handling
    (   catch(
            compose_templates(
                ['bash/header','facts/array_binary','facts/lookup_binary','facts/stream_binary','facts/reverse_stream_binary'],
                [pred=PredStr, entries='[test]=1', strategy=sort_u],
                BashCode
            ),
            Error,
            (format(user_error, 'DEBUG: Template rendering error: ~w~n', [Error]), fail)
        ) ->
        format('DEBUG: Template rendered successfully~n'),
        format('DEBUG: BashCode length: ~w~n', [string_length(BashCode, _)])
    ;   format('DEBUG: Template rendering failed~n'),
        fail
    ).

%% compile_single_rule(+Pred, +Body, +Options, -BashCode)
%  Compile single rule into a bash function that evaluates the rule's body.
compile_single_rule(Pred, Body, Options, BashCode) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, _),
    arg(1, Head, A), % Assume variables are named A, B, C...
    arg(2, Head, B),
    
    % Translate the body of the rule to a bash script
    translate_body_to_bash(Body, BashBody),
    
    % For now, we assume a simple structure with 2 variables.
    % This needs to be generalized.
    format(string(Pipeline), 'local A="$1"
    local B="$2"
    ~w', [BashBody]),

    generate_dedup_wrapper(PredStr, "", Pipeline, Options, BashCode).

%% translate_body_to_bash(+Goal, -Bash)
%  Translates a Prolog goal or a conjunction of goals into bash code.
translate_body_to_bash((A, B), Bash) :- !,
    translate_body_to_bash(A, BashA),
    translate_body_to_bash(B, BashB),
    format(string(Bash), '~w &&\n    ~w', [BashA, BashB]).
translate_body_to_bash(\+ Goal, Bash) :- !,
    translate_body_to_bash(Goal, BashGoal),
    format(string(Bash), '! { ~w }', [BashGoal]).
translate_body_to_bash(is(Var, Expr), Bash) :- !,
    translate_expr(Expr, BashExpr),
    format(string(Bash), '~w=$((~s))', [Var, BashExpr]).
translate_body_to_bash(Goal, Bash) :-
    goal_to_bash_operator(Goal, Op), !,
    Goal =.. [_, A, B],
    format(string(Bash), '[[ "$~w" -~w "$~w" ]]', [A, Op, B]).
translate_body_to_bash(Goal, Bash) :-
    % Default case for calling another predicate
    functor(Goal, Functor, _),
    atom_string(Functor, FuncStr),
    Goal =.. [_|Args],
    maplist(format_arg, Args, BashArgs),
    atomic_list_concat(BashArgs, " ", BashArgsStr),
    format(string(Bash), '~w ~w', [FuncStr, BashArgsStr]).

%% format_arg(+Arg, -BashArg)
%  Formats a Prolog term as a bash argument.
format_arg(Var, BashArg) :- var(Var), !, format(string(BashArg), '"$~w"', [Var]).
format_arg(Atom, BashArg) :- atom(Atom), !, format(string(BashArg), '"~w"', [Atom]).
format_arg(Number, BashArg) :- number(Number), !, format(string(BashArg), '~w', [Number]).

%% goal_to_bash_operator(+Goal, -BashOperator)
%  Maps Prolog comparison operators to bash test operators.
goal_to_bash_operator(_ > _, 'gt').
goal_to_bash_operator(_ < _, 'lt').
goal_to_bash_operator(_ >= _, 'ge').
goal_to_bash_operator(_ =< _, 'le').
goal_to_bash_operator(_ =:= _, 'eq').
goal_to_bash_operator(_ =\= _, 'ne').

%% translate_expr(+PrologExpr, -BashExpr)
%  Translates a Prolog arithmetic expression to a bash one.
translate_expr(A + B, BashExpr) :- !,
    translate_expr(A, BashA),
    translate_expr(B, BashB),
    format(string(BashExpr), '~w + ~w', [BashA, BashB]).
translate_expr(A - B, BashExpr) :- !,
    translate_expr(A, BashA),
    translate_expr(B, BashB),
    format(string(BashExpr), '~w - ~w', [BashA, BashB]).
translate_expr(Var, BashExpr) :- var(Var), !, format(string(BashExpr), '$~w', [Var]).
translate_expr(Number, BashExpr) :- number(Number), !, format(string(BashExpr), '~w', [Number]).


%% Check for inequality constraints (cut-safe, terminating)
has_inequality((A, B)) :- !,
    (has_inequality(A) ; has_inequality(B)).
has_inequality(_ \= _) :- !.
has_inequality(\=(_, _)) :- !.
has_inequality(Body) :-
    nonvar(Body), !,
    fail.

% Keep helper for compatibility; add cuts to avoid fallback loops
has_inequality_in_conjunction((A, B)) :- !,
    (has_inequality(A) ; has_inequality(B)).
has_inequality_in_conjunction(Goal) :- !,
    has_inequality(Goal).

%% Compile with inequality handling
compile_with_inequality(Pred, Body, BashCode) :-
    atom_string(Pred, PredStr),
    extract_predicates(Body, Predicates),
    (   Predicates = [parent, parent] ->  % Fix: check for 'parent' atom, not variable
        % Special case: sibling pattern with parent predicates
        format(string(BashCode), '#!/bin/bash
# ~s - with inequality constraint

~s() {
    declare -A seen
    
    for key1 in "${!parent_data[@]}"; do
        IFS=":" read -r p1 c1 <<< "$key1"
        
        for key2 in "${!parent_data[@]}"; do
            IFS=":" read -r p2 c2 <<< "$key2"
            
            # Same parent, different children
            if [[ "$p1" == "$p2" && "$c1" != "$c2" ]]; then
                pair="$c1:$c2"
                if [[ -z "${seen[$pair]}" ]]; then
                    seen[$pair]=1
                    echo "$pair"
                fi
            fi
        done
    done
}

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, PredStr, PredStr, PredStr])
    ;   % General inequality case
        format(string(BashCode), '#!/bin/bash
# ~s - with inequality constraint
~s() { echo "TODO: general inequality"; }

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, PredStr, PredStr, PredStr])
    ).

%% Write bash code to file with Unix line endings
write_bash_file(File, BashCode) :-
    % Convert all line endings to Unix format
    % Replace \r\n with \n, and standalone \r with \n
    atom_string(BashCode, BashStr),
    split_string(BashStr, "\r\n", "", Lines1),  % Split on Windows CRLF
    atomic_list_concat(Lines1, "\n", TempStr),   % Join with Unix LF
    split_string(TempStr, "\r", "", Lines2),     % Split on any remaining CR
    atomic_list_concat(Lines2, "\n", UnixStr),   % Join with Unix LF
    
    % Write with binary mode to preserve exact line endings
    open(File, write, Stream, [type(binary)]),
    string_codes(UnixStr, Codes),
    maplist(put_byte(Stream), Codes),
    close(Stream).

%% Main test
test_stream_compiler :-
    writeln('=== STREAM COMPILER TEST ==='),
    writeln('Testing basic non-recursive predicate compilation'),
    
    % Setup output directory
    (   exists_directory('output') -> true
    ;   make_directory('output')
    ),
    writeln('Output directory: output/'),
    
    % Clear any existing predicates
    abolish(parent/2),
    abolish(grandparent/2),
    abolish(sibling/2),
    abolish(related/2),
    
    % Define test predicates (facts) - with siblings
    assertz(parent(alice, bob)),
    assertz(parent(alice, barbara)),     % alice has two children
    assertz(parent(bob, charlie)),
    assertz(parent(bob, cathy)),         % bob has two children  
    assertz(parent(charlie, diana)),
    assertz(parent(diana, eve)),
    assertz(parent(diana, emily)),       % diana has two children
    assertz(parent(eve, frank)),
    
    % Define test predicates (rules)
    assertz((grandparent(X, Z) :- parent(X, Y), parent(Y, Z))),
    assertz((sibling(X, Y) :- parent(P, X), parent(P, Y), X \= Y)),
    
    % OR pattern - multiple ways to be related
    assertz((related(X, Y) :- parent(X, Y))),       % Forward: X is parent of Y
    assertz((related(X, Y) :- parent(Y, X))),       % Reverse: Y is parent of X (X is child of Y)
    assertz((related(X, Y) :- sibling(X, Y))),
    
    % Declare constraints (using defaults: unique=true, unordered=true)
    % No need to declare for grandparent/sibling/related - they use defaults
    % But we'll be explicit for documentation
    declare_constraint(grandparent/2, [unique, unordered]),
    declare_constraint(sibling/2, [unique, unordered]),
    declare_constraint(related/2, [unique, unordered]),

    % Compile predicates (constraints come from declarations, not options)
    writeln('--- Compiling predicates ---'),
    compile_predicate(parent/2, [], ParentCode),
    write_bash_file('output/parent.sh', ParentCode),

    compile_predicate(grandparent/2, [], GrandparentCode),
    write_bash_file('output/grandparent.sh', GrandparentCode),

    compile_predicate(sibling/2, [], SiblingCode),
    write_bash_file('output/sibling.sh', SiblingCode),

    compile_predicate(related/2, [], RelatedCode),
    write_bash_file('output/related.sh', RelatedCode),
    
    % Generate test script
    generate_test_script,
    
    writeln('--- Test Complete ---'),
    writeln('Check files in output/'),
    writeln('Run: bash output/test.sh').

%% Generate deduplication wrapper based on constraints
%  For single-rule predicates
generate_dedup_wrapper(PredStr, JoinCode, Pipeline, Options, BashCode) :-
    (   constraint_implies_sort_u(Options) ->
        % Use sort -u for unique + unordered
        format(string(BashCode), '#!/bin/bash
# ~s - streaming pipeline with uniqueness (sort -u)

~s

~s() {
    ~s | sort -u
}

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, JoinCode, PredStr, Pipeline, PredStr, PredStr])
    ;   constraint_implies_hash(Options) ->
        % Use hash-based dedup for unique + ordered
        format(string(BashCode), '#!/bin/bash
# ~s - streaming pipeline with hash-based deduplication (preserves order)

~s

~s() {
    declare -A seen
    ~s | while IFS= read -r line; do
        if [[ -z "${seen[$line]}" ]]; then
            seen[$line]=1
            echo "$line"
        fi
    done
}

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, JoinCode, PredStr, Pipeline, PredStr, PredStr])
    ;   % No deduplication
        format(string(BashCode), '#!/bin/bash
# ~s - streaming pipeline (no deduplication)

~s

~s() {
    ~s
}

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, JoinCode, PredStr, Pipeline, PredStr, PredStr])
    ).

%% Generate deduplication wrapper for multiple-rule predicates
generate_dedup_wrapper_multi(PredStr, NumAlts, JoinFuncsCode, FunctionsCode, Pipeline, Options, BashCode) :-
    (   constraint_implies_sort_u(Options) ->
        format(string(BashCode), '#!/bin/bash
# ~s - OR pattern with ~w alternatives (sort -u)

~s

~s

# Main function - combine alternatives with uniqueness
~s() {
    ~s | sort -u
}', [PredStr, NumAlts, JoinFuncsCode, FunctionsCode, PredStr, Pipeline])
    ;   constraint_implies_hash(Options) ->
        format(string(BashCode), '#!/bin/bash
# ~s - OR pattern with ~w alternatives (hash dedup, preserves order)

~s

~s

# Main function - combine alternatives preserving order
~s() {
    declare -A seen
    ~s | while IFS= read -r line; do
        if [[ -z "${seen[$line]}" ]]; then
            seen[$line]=1
            echo "$line"
        fi
    done
}', [PredStr, NumAlts, JoinFuncsCode, FunctionsCode, PredStr, Pipeline])
    ;   % No deduplication
        format(string(BashCode), '#!/bin/bash
# ~s - OR pattern with ~w alternatives (no deduplication)

~s

~s

# Main function - combine alternatives
~s() {
    ~s
}', [PredStr, NumAlts, JoinFuncsCode, FunctionsCode, PredStr, Pipeline])
    ).

%% Generate test script
generate_test_script :-
    TestScript = '#!/bin/bash
# Test script for compiled predicates

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source files relative to script directory
source "$SCRIPT_DIR/parent.sh"
source "$SCRIPT_DIR/grandparent.sh"
source "$SCRIPT_DIR/sibling.sh"
source "$SCRIPT_DIR/related.sh"

echo "=== Testing parent ==="
echo "parent alice bob: $(parent alice bob)"
echo "parent stream:"
parent_stream | head -3

echo ""
echo "=== Testing grandparent ==="
echo "grandparent alice charlie:"
grandparent | grep "alice:charlie" || echo "Not found"
echo "All grandparents:"
grandparent

echo ""
echo "=== Testing sibling ==="
echo "Siblings:"
sibling

echo ""
echo "=== Testing related ==="
echo "Related pairs:"
related | head -5',
    
    write_bash_file('output/test.sh', TestScript).
