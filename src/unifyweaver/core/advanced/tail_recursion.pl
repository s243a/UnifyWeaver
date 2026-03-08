:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% tail_recursion.pl - Compile tail recursive predicates to iterative loops
% Uses list-of-strings template style for better readability

:- module(tail_recursion, [
    compile_tail_recursion/3,     % +Pred/Arity, +Options, -BashCode
    can_compile_tail_recursion/1,  % +Pred/Arity
    test_tail_recursion/0        % Test predicate
]).

:- use_module(library(lists)).
:- use_module('../template_system').
:- use_module('../constraint_analyzer').
:- use_module('pattern_matchers').

%% can_compile_tail_recursion(+Pred/Arity)
%  Check if predicate can be compiled as tail recursion
can_compile_tail_recursion(Pred/Arity) :-
    is_tail_recursive_accumulator(Pred/Arity, _AccInfo).

%% compile_tail_recursion(+Pred/Arity, +Options, -BashCode)
%  Compile tail recursive predicate to bash while loop
%  Options: List of Key=Value pairs (e.g., [unique=true, ordered=false])
%  Currently tail recursive predicates return single values (not sets),
%  so deduplication constraints don't apply. Options are reserved for
%  future use (e.g., output language selection).
compile_tail_recursion(Pred/Arity, Options, Code) :-
    format('  Compiling tail recursion: ~w/~w~n', [Pred, Arity]),

    % Query constraints (for logging and future use)
    get_constraints(Pred/Arity, Constraints),
    format('  Constraints: ~w~n', [Constraints]),

    % Merge runtime options with constraints
    append(Options, Constraints, AllOptions),
    format('  Final options: ~w~n', [AllOptions]),
    
    % Determine target (default to bash)
    (   member(target(Target), AllOptions) -> true
    ;   Target = bash
    ),
    format('  Target: ~w~n', [Target]),

    % Apply unique constraint optimization
    % Tail recursive predicates with unique(true) can exit after first result
    (   member(unique(true), AllOptions) ->
        format('  Applying unique constraint optimization~n', []),
        ExitAfterResult = true
    ;   ExitAfterResult = false
    ),

    % Get accumulator pattern info
    is_tail_recursive_accumulator(Pred/Arity, AccInfo),
    AccInfo = acc_pattern(BaseClauses, RecClauses, AccPos),

    % Extract the step operation pattern
    extract_accumulator_pattern(Pred/Arity, Pattern),
    Pattern = pattern(_InitValue, StepOp, _UnifyType),

    % Generate code based on pattern and target
    atom_string(Pred, PredStr),
    (   Target == r ->
        generate_tail_recursion_r(PredStr, Arity, BaseClauses, RecClauses, AccPos, StepOp, ExitAfterResult, Code)
    ;   Target == bash ->
        generate_tail_recursion_bash(PredStr, Arity, BaseClauses, RecClauses, AccPos, StepOp, ExitAfterResult, Code)
    ;   format('Error: Unsupported target ~w for tail recursion~n', [Target]),
        fail
    ).

%% generate_tail_recursion_r(+PredStr, +Arity, +BaseClauses, +RecClauses, +AccPos, +StepOp, +ExitAfterResult, -RCode)
generate_tail_recursion_r(PredStr, Arity, _BaseClauses, _RecClauses, AccPos, StepOp, ExitAfterResult, RCode) :-
    % Determine which pattern to use based on arity
    (   Arity =:= 3 ->
        generate_ternary_tail_loop_r(PredStr, AccPos, StepOp, ExitAfterResult, RCode)
    ;   Arity =:= 2 ->
        generate_binary_tail_loop_r(PredStr, ExitAfterResult, RCode)
    ;   % Unsupported arity
        format('Warning: tail recursion in R with arity ~w not yet supported~n', [Arity]),
        fail
    ).

%% step_op_to_r(+StepOp, -RCode)
step_op_to_r(arithmetic(Expr), RCode) :-
    expr_to_r(Expr, RExpr),
    format(atom(RCode), 'current_acc <- ~w', [RExpr]).
step_op_to_r(unknown, 'current_acc <- current_acc + 1').  % Fallback

%% expr_to_r(+PrologExpr, -RExpr)
expr_to_r(_ + Const, RExpr) :- integer(Const), !, format(atom(RExpr), 'current_acc + ~w', [Const]).
expr_to_r(_ + _, 'current_acc + item') :- !.
expr_to_r(_ - _, 'current_acc - item') :- !.
expr_to_r(_ * _, 'current_acc * item') :- !.
expr_to_r(_, 'current_acc + 1').  % Fallback

%% generate_ternary_tail_loop_r(+PredStr, +AccPos, +StepOp, +ExitAfterResult, -RCode)
generate_ternary_tail_loop_r(PredStr, _AccPos, StepOp, ExitAfterResult, RCode) :-
    step_op_to_r(StepOp, RStepOp),
    (   ExitAfterResult = true ->
        ExitStatement = "        return(current_acc)  # Unique constraint: only one result"
    ;   ExitStatement = ""
    ),

    TemplateLines = [
        "# {{pred}} - tail recursive accumulator pattern (R)",
        "{{pred}} <- function(input, acc) {",
        "    items <- input",
        "    if (is.character(input)) {",
        "       # parse string input if necessary",
        "       items <- unlist(strsplit(gsub(\"\\\\[|\\\\]\", \"\", input), \",\"))",
        "       items <- as.numeric(items)",
        "    }",
        "    current_acc <- acc",
        "    for (item in items) {",
        "        {{step_op}}",
        "{{exit_statement}}",
        "    }",
        "    return(current_acc)",
        "}",
        "{{pred}}_eval <- function(input) {",
        "    return({{pred}}(input, 0))",
        "}"
    ],

    atomic_list_concat(TemplateLines, '\n', Template),
    render_template(Template, [pred=PredStr, step_op=RStepOp, exit_statement=ExitStatement], RCode).

%% generate_binary_tail_loop_r(+PredStr, +ExitAfterResult, -RCode)
generate_binary_tail_loop_r(PredStr, ExitAfterResult, RCode) :-
    (   ExitAfterResult = true ->
        ExitStatement = "    return(count)  # Unique constraint: only one result"
    ;   ExitStatement = ""
    ),

    TemplateLines = [
        "# {{pred}} - tail recursive binary pattern (R)",
        "{{pred}} <- function(input, expected=NULL) {",
        "    count <- 0",
        "    items <- input",
        "    while (length(items) > 0) {",
        "        count <- count + 1",
        "        items <- items[-1]",
        "    }",
        "    if (!is.null(expected)) {",
        "        return(count == expected)",
        "    }",
        "    return(count)",
        "{{exit_statement}}",
        "}"
    ],

    atomic_list_concat(TemplateLines, '\n', Template),
    render_template(Template, [pred=PredStr, exit_statement=ExitStatement], RCode).

%% generate_tail_recursion_bash(+PredStr, +Arity, +BaseClauses, +RecClauses, +AccPos, +StepOp, +ExitAfterResult, -BashCode)
generate_tail_recursion_bash(PredStr, Arity, _BaseClauses, _RecClauses, AccPos, StepOp, ExitAfterResult, BashCode) :-
    % Determine which pattern to use based on arity
    (   Arity =:= 3 ->
        % Standard accumulator pattern: pred(Input, Accumulator, Result)
        generate_ternary_tail_loop(PredStr, AccPos, StepOp, ExitAfterResult, BashCode)
    ;   Arity =:= 2 ->
        % Binary pattern: pred(Input, Result) with implicit accumulator
        generate_binary_tail_loop(PredStr, ExitAfterResult, BashCode)
    ;   % Unsupported arity
        format('Warning: tail recursion with arity ~w not yet supported~n', [Arity]),
        fail
    ).

%% step_op_to_bash(+StepOp, -BashCode)
%  Convert Prolog step operation to bash arithmetic
%  StepOp: arithmetic(Expr) where Expr contains variables and operations
step_op_to_bash(arithmetic(Expr), BashCode) :-
    % Convert Prolog expression to bash
    expr_to_bash(Expr, BashExpr),
    format(atom(BashCode), 'current_acc=$((~w))', [BashExpr]).
step_op_to_bash(unknown, 'current_acc=$((current_acc + 1))').  % Fallback

%% expr_to_bash(+PrologExpr, -BashExpr)
%  Convert Prolog arithmetic expression to bash syntax
expr_to_bash(_ + Const, BashExpr) :-
    % Constant addition (like +1)
    integer(Const),
    !,
    format(atom(BashExpr), 'current_acc + ~w', [Const]).
expr_to_bash(_ + _, BashExpr) :-
    % Variable + Variable (assume accumulator + list element)
    !,
    BashExpr = 'current_acc + item'.
expr_to_bash(_ - _, 'current_acc - item') :- !.
expr_to_bash(_ * _, 'current_acc * item') :- !.
expr_to_bash(_, 'current_acc + 1').  % Fallback

%% generate_ternary_tail_loop(+PredStr, +AccPos, +StepOp, +ExitAfterResult, -BashCode)
%  Generate bash code for arity-3 tail recursive predicates
%  Pattern: count([H|T], Acc, N) :- Acc1 is Acc + 1, count(T, Acc1, N).
%  StepOp: arithmetic(Expr) where Expr is the step operation
%  ExitAfterResult: true if unique constraint applies (exit after first result)
generate_ternary_tail_loop(PredStr, AccPos, StepOp, ExitAfterResult, BashCode) :-
    % Convert step operation to bash code
    step_op_to_bash(StepOp, BashStepOp),

    % Generate return statement if unique constraint
    (   ExitAfterResult = true ->
        ExitStatement = "        return 0  # Unique constraint: only one result"
    ;   ExitStatement = ""
    ),

    % Use list-of-strings template style
    TemplateLines = [
        "#!/bin/bash",
        "# {{pred}} - tail recursive accumulator pattern",
        "# Compiled to iterative while loop",
        "",
        "{{pred}}() {",
        "    local input=\"$1\"",
        "    local acc=\"$2\"",
        "    local result_var=\"$3\"",
        "    ",
        "    # Convert input to array if it's a list notation",
        "    if [[ \"$input\" =~ ^\\[.*\\]$ ]]; then",
        "        # Remove brackets and split by comma",
        "        input=\"${input#[}\"",
        "        input=\"${input%]}\"",
        "        IFS=',' read -ra items <<< \"$input\"",
        "    else",
        "        items=()",
        "    fi",
        "    ",
        "    local current_acc=\"$acc\"",
        "    ",
        "    # Iterative loop (tail recursion optimization)",
        "    for item in \"${items[@]}\"; do",
        "        # Step operation",
        "        {{step_op}}",
        "    done",
        "    ",
        "    # Return result",
        "    if [[ -n \"$result_var\" ]]; then",
        "        eval \"$result_var=$current_acc\"",
        "    else",
        "        echo \"$current_acc\"",
        "    fi",
        "{{exit_statement}}",
        "}",
        "",
        "# Helper function for common use case",
        "{{pred}}_eval() {",
        "    {{pred}} \"$1\" 0 result",
        "    echo \"$result\"",
        "}"
    ],

    % Join lines and render template
    atomic_list_concat(TemplateLines, '\n', Template),
    render_template(Template, [pred=PredStr, acc_pos=AccPos, step_op=BashStepOp, exit_statement=ExitStatement], BashCode).

%% generate_binary_tail_loop(+PredStr, +ExitAfterResult, -BashCode)
%  Generate bash code for arity-2 tail recursive predicates
%  Pattern: length([_|T], N) :- length(T, N1), N is N1 + 1.
%  ExitAfterResult: true if unique constraint applies (exit after first result)
generate_binary_tail_loop(PredStr, ExitAfterResult, BashCode) :-
    % Generate return statement if unique constraint
    (   ExitAfterResult = true ->
        ExitStatement = "    return 0  # Unique constraint: only one result"
    ;   ExitStatement = ""
    ),

    TemplateLines = [
        "#!/bin/bash",
        "# {{pred}} - tail recursive binary pattern",
        "",
        "{{pred}}() {",
        "    local input=\"$1\"",
        "    local expected=\"$2\"",
        "    ",
        "    # Process input iteratively",
        "    local count=0",
        "    ",
        "    # Simple iteration for demonstration",
        "    # TODO: Customize based on actual predicate logic",
        "    while [[ -n \"$input\" ]]; do",
        "        count=$((count + 1))",
        "        # Process next item (simplified)",
        "        break",
        "    done",
        "    ",
        "    if [[ -n \"$expected\" ]]; then",
        "        [[ \"$count\" == \"$expected\" ]] && echo \"true\" || echo \"false\"",
        "    else",
        "        echo \"$count\"",
        "    fi",
        "{{exit_statement}}",
        "}"
    ],

    atomic_list_concat(TemplateLines, '\n', Template),
    render_template(Template, [pred=PredStr, exit_statement=ExitStatement], BashCode).

%% ============================================
%% TESTS
%% ============================================

test_tail_recursion :-
    writeln('=== TAIL RECURSION COMPILER TESTS ==='),

    % Setup output directory
    (   exists_directory('output/advanced') -> true
    ;   make_directory('output/advanced')
    ),

    % Clear test predicates
    catch(abolish(count_items/3), _, true),
    catch(abolish(sum_list/3), _, true),

    % Test 1: Count items with accumulator
    writeln('Test 1: Compile count_items/3 (tail recursive)'),
    assertz(user:(count_items([], Acc, Acc))),
    assertz(user:(count_items([_|T], Acc, N) :- Acc1 is Acc + 1, count_items(T, Acc1, N))),

    (   can_compile_tail_recursion(count_items/3) ->
        writeln('  ✓ Pattern detected'),
        compile_tail_recursion(count_items/3, [target(bash)], Code1),
        write_bash_file('output/advanced/count_items.sh', Code1),
        writeln('  ✓ Compiled to output/advanced/count_items.sh (bash)'),
        compile_tail_recursion(count_items/3, [target(r)], Code1R),
        write_bash_file('output/advanced/count_items.R', Code1R),
        writeln('  ✓ Compiled to output/advanced/count_items.R (r)')
    ;   writeln('  ✗ FAIL - should detect tail recursion')
    ),

    % Test 2: Sum list with accumulator
    writeln('Test 2: Compile sum_list/3 (tail recursive)'),
    assertz(user:(sum_list([], Acc, Acc))),
    assertz(user:(sum_list([H|T], Acc, Sum) :- Acc1 is Acc + H, sum_list(T, Acc1, Sum))),

    (   can_compile_tail_recursion(sum_list/3) ->
        writeln('  ✓ Pattern detected'),
        compile_tail_recursion(sum_list/3, [target(bash)], Code2),
        write_bash_file('output/advanced/sum_list.sh', Code2),
        writeln('  ✓ Compiled to output/advanced/sum_list.sh (bash)'),
        compile_tail_recursion(sum_list/3, [target(r)], Code2R),
        write_bash_file('output/advanced/sum_list.R', Code2R),
        writeln('  ✓ Compiled to output/advanced/sum_list.R (r)')
    ;   writeln('  ✗ FAIL - should detect tail recursion')
    ),

    writeln('=== TAIL RECURSION COMPILER TESTS COMPLETE ===').

%% Helper to write bash files
write_bash_file(Path, Content) :-
    open(Path, write, Stream),
    write(Stream, Content),
    close(Stream),
    % Make executable
    atom_concat('chmod +x ', Path, ChmodCmd),
    shell(ChmodCmd).
