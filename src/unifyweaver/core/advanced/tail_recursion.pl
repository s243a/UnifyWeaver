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
:- use_module('pattern_matchers').

%% can_compile_tail_recursion(+Pred/Arity)
%  Check if predicate can be compiled as tail recursion
can_compile_tail_recursion(Pred/Arity) :-
    is_tail_recursive_accumulator(Pred/Arity, _AccInfo).

%% compile_tail_recursion(+Pred/Arity, +Options, -BashCode)
%  Compile tail recursive predicate to bash while loop
compile_tail_recursion(Pred/Arity, _Options, BashCode) :-
    format('  Compiling tail recursion: ~w/~w~n', [Pred, Arity]),

    % Get accumulator pattern info
    is_tail_recursive_accumulator(Pred/Arity, AccInfo),
    AccInfo = acc_pattern(BaseClauses, RecClauses, AccPos),

    % Generate bash code based on pattern
    atom_string(Pred, PredStr),
    generate_tail_recursion_bash(PredStr, Arity, BaseClauses, RecClauses, AccPos, BashCode).

%% generate_tail_recursion_bash(+PredStr, +Arity, +BaseClauses, +RecClauses, +AccPos, -BashCode)
generate_tail_recursion_bash(PredStr, Arity, _BaseClauses, _RecClauses, AccPos, BashCode) :-
    % Determine which pattern to use based on arity
    (   Arity =:= 3 ->
        % Standard accumulator pattern: pred(Input, Accumulator, Result)
        generate_ternary_tail_loop(PredStr, AccPos, BashCode)
    ;   Arity =:= 2 ->
        % Binary pattern: pred(Input, Result) with implicit accumulator
        generate_binary_tail_loop(PredStr, BashCode)
    ;   % Unsupported arity
        format('Warning: tail recursion with arity ~w not yet supported~n', [Arity]),
        fail
    ).

%% generate_ternary_tail_loop(+PredStr, +AccPos, -BashCode)
%  Generate bash code for arity-3 tail recursive predicates
%  Pattern: count([H|T], Acc, N) :- Acc1 is Acc + 1, count(T, Acc1, N).
generate_ternary_tail_loop(PredStr, AccPos, BashCode) :-
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
        "        # Step operation - increment accumulator",
        "        current_acc=$((current_acc + 1))",
        "    done",
        "    ",
        "    # Return result",
        "    if [[ -n \"$result_var\" ]]; then",
        "        eval \"$result_var=$current_acc\"",
        "    else",
        "        echo \"$current_acc\"",
        "    fi",
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
    render_template(Template, [pred=PredStr, acc_pos=AccPos], BashCode).

%% generate_binary_tail_loop(+PredStr, -BashCode)
%  Generate bash code for arity-2 tail recursive predicates
%  Pattern: length([_|T], N) :- length(T, N1), N is N1 + 1.
generate_binary_tail_loop(PredStr, BashCode) :-
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
        "}"
    ],

    atomic_list_concat(TemplateLines, '\n', Template),
    render_template(Template, [pred=PredStr], BashCode).

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
        compile_tail_recursion(count_items/3, [], Code1),
        write_bash_file('output/advanced/count_items.sh', Code1),
        writeln('  ✓ Compiled to output/advanced/count_items.sh')
    ;   writeln('  ✗ FAIL - should detect tail recursion')
    ),

    % Test 2: Sum list with accumulator
    writeln('Test 2: Compile sum_list/3 (tail recursive)'),
    assertz(user:(sum_list([], Acc, Acc))),
    assertz(user:(sum_list([H|T], Acc, Sum) :- Acc1 is Acc + H, sum_list(T, Acc1, Sum))),

    (   can_compile_tail_recursion(sum_list/3) ->
        writeln('  ✓ Pattern detected'),
        compile_tail_recursion(sum_list/3, [], Code2),
        write_bash_file('output/advanced/sum_list.sh', Code2),
        writeln('  ✓ Compiled to output/advanced/sum_list.sh')
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
