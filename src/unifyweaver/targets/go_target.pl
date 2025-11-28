:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% go_target.pl - Go Target for UnifyWeaver
% Generates standalone Go programs for record/field processing
% Supports configurable delimiters, quoting, and regex matching

:- module(go_target, [
    compile_predicate_to_go/3,      % +Predicate, +Options, -GoCode
    write_go_program/2              % +GoCode, +FilePath
]).

:- use_module(library(lists)).

%% ============================================
%% PUBLIC API
%% ============================================

%% compile_predicate_to_go(+Predicate, +Options, -GoCode)
%  Compile a Prolog predicate to Go code
%
%  @arg Predicate Predicate indicator (Name/Arity)
%  @arg Options List of options
%  @arg GoCode Generated Go code as atom
%
%  Options:
%  - record_delimiter(null|newline|Char) - Record separator (default: newline)
%  - field_delimiter(colon|tab|comma|Char) - Field separator (default: colon)
%  - quoting(csv|none) - Quoting style (default: none)
%  - escape_char(Char) - Escape character (default: backslash)
%  - include_package(true|false) - Include package main (default: true)
%  - unique(true|false) - Deduplicate results (default: true)
%
compile_predicate_to_go(PredIndicator, Options, GoCode) :-
    PredIndicator = Pred/Arity,
    format('=== Compiling ~w/~w to Go ===~n', [Pred, Arity]),

    % Get options
    option(record_delimiter(RecordDelim), Options, newline),
    option(field_delimiter(FieldDelim), Options, colon),
    option(quoting(Quoting), Options, none),
    option(escape_char(EscapeChar), Options, backslash),
    option(include_package(IncludePackage), Options, true),
    option(unique(Unique), Options, true),

    % Create head with correct arity
    functor(Head, Pred, Arity),

    % Get all clauses for this predicate
    findall(Head-Body, user:clause(Head, Body), Clauses),

    % Determine compilation strategy
    (   Clauses = [] ->
        format('ERROR: No clauses found for ~w/~w~n', [Pred, Arity]),
        fail
    ;   maplist(is_fact_clause, Clauses) ->
        % All bodies are 'true' - these are facts
        format('Type: facts (~w clauses)~n', [length(Clauses, _)]),
        compile_facts_to_go(Pred, Arity, Clauses, RecordDelim, FieldDelim,
                           Unique, ScriptBody)
    ;   Clauses = [SingleHead-SingleBody], SingleBody \= true ->
        % Single rule
        format('Type: single_rule~n'),
        compile_single_rule_to_go(Pred, Arity, SingleHead, SingleBody, RecordDelim,
                                 FieldDelim, Unique, ScriptBody)
    ;   % Multiple rules (OR pattern)
        format('Type: multiple_rules (~w clauses)~n', [length(Clauses, _)]),
        compile_multiple_rules_to_go(Pred, Arity, Clauses, RecordDelim,
                                    FieldDelim, Unique, ScriptBody)
    ),

    % Generate complete Go program
    (   IncludePackage ->
        generate_go_program(Pred, Arity, RecordDelim, FieldDelim, Quoting,
                           EscapeChar, ScriptBody, GoCode)
    ;   GoCode = ScriptBody
    ),
    !.

%% Helper to check if a clause is a fact (body is just 'true')
is_fact_clause(_-true).

%% ============================================
%% FACTS COMPILATION
%% ============================================

%% compile_facts_to_go(+Pred, +Arity, +Clauses, +RecordDelim, +FieldDelim, +Unique, -GoCode)
%  Compile fact predicates to Go map lookup
%
compile_facts_to_go(Pred, Arity, Clauses, _RecordDelim, FieldDelim, Unique, GoCode) :-
    atom_string(Pred, PredStr),

    % Build map entries
    findall(Entry,
        (   member(Fact-true, Clauses),
            Fact =.. [_|Args],
            format_go_fact_entry(Args, FieldDelim, Entry)
        ),
        Entries),
    atomic_list_concat(Entries, ',\n\t\t', EntriesStr),

    % Generate Go code
    format(string(GoCode), '
\tfacts := map[string]bool{
\t\t~s,
\t}

\tfor key := range facts {
\t\tfmt.Println(key)
\t}
', [EntriesStr]).

%% format_go_fact_entry(+Args, +FieldDelim, -Entry)
%  Format a fact as a Go map entry
format_go_fact_entry(Args, FieldDelim, Entry) :-
    map_field_delimiter(FieldDelim, DelimChar),
    maplist(atom_string, Args, ArgStrs),
    atomic_list_concat(ArgStrs, DelimChar, Key),
    format(string(Entry), '"~s": true', [Key]).

%% ============================================
%% SINGLE RULE COMPILATION
%% ============================================

%% compile_single_rule_to_go(+Pred, +Arity, +Head, +Body, +RecordDelim, +FieldDelim, +Unique, -GoCode)
%  Compile single rule to Go code
%
compile_single_rule_to_go(Pred, Arity, Head, Body, RecordDelim, FieldDelim, Unique, GoCode) :-
    atom_string(Pred, PredStr),

    % Build variable mapping from head arguments
    Head =.. [_|HeadArgs],
    build_var_map(HeadArgs, VarMap),
    format('  Variable map: ~w~n', [VarMap]),

    % Extract predicates from body
    extract_predicates(Body, Predicates),
    format('  Body predicates: ~w~n', [Predicates]),

    % For now, handle simple case: single predicate in body
    (   Predicates = [SinglePred] ->
        compile_single_predicate_rule_go(PredStr, HeadArgs, SinglePred, VarMap, FieldDelim, Unique, GoCode)
    ;   % Multiple predicates or no predicates - TODO
        GoCode = '\t// TODO: Multi-predicate or constraint-only rules not yet implemented\n'
    ).

%% compile_single_predicate_rule_go(+PredStr, +HeadArgs, +BodyPred, +VarMap, +FieldDelim, +Unique, -GoCode)
%  Compile a rule with single predicate in body (e.g., child(C,P) :- parent(P,C))
%
compile_single_predicate_rule_go(PredStr, HeadArgs, BodyPred, VarMap, FieldDelim, Unique, GoCode) :-
    % Get the body predicate name and args
    BodyPred =.. [BodyPredName|BodyArgs],
    atom_string(BodyPredName, BodyPredStr),
    map_field_delimiter(FieldDelim, DelimChar),

    % Build output format by mapping body args to field positions
    % For each arg in head, find which position in body has the identical variable
    findall(OutputPart,
        (   nth1(HeadPos, HeadArgs, HeadArg),
            (   var(HeadArg),
                % Find which position in BodyArgs has the same variable
                nth1(BodyPos, BodyArgs, BodyArg),
                HeadArg == BodyArg
            ->  format(atom(OutputPart), 'field~w', [BodyPos])
            ;   % Constant in head
                atom_string(HeadArg, HeadArgStr),
                format(atom(OutputPart), '"~s"', [HeadArgStr])
            )
        ),
        OutputParts),

    % Build Go expression with delimiter between parts
    build_go_concat_expr(OutputParts, DelimChar, OutputExpr),

    % Generate Go code that reads from stdin and processes records
    length(BodyArgs, NumFields),
    generate_field_assignments(BodyArgs, FieldAssignments),
    format(string(GoCode), '
\t// Read from stdin and process ~s records
\tscanner := bufio.NewScanner(os.Stdin)
\tseen := make(map[string]bool)
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tparts := strings.Split(line, "~s")
\t\tif len(parts) == ~w {
\t\t\t~s
\t\t\tresult := ~s
\t\t\tif !seen[result] {
\t\t\t\tseen[result] = true
\t\t\t\tfmt.Println(result)
\t\t\t}
\t\t}
\t}
', [BodyPredStr, DelimChar, NumFields,
    FieldAssignments, OutputExpr]).

%% generate_field_assignments(+Args, -Code)
%  Generate field assignment statements
generate_field_assignments(Args, Code) :-
    findall(Assignment,
        (   nth1(N, Args, _),
            I is N - 1,
            format(atom(Assignment), 'field~w := parts[~w]', [N, I])
        ),
        Assignments),
    atomic_list_concat(Assignments, '\n\t\t\t', Code).

%% build_go_concat_expr(+Parts, +Delimiter, -Expression)
%  Build a Go string concatenation expression with delimiters
%  e.g., [field1, field2] with ":" -> field1 + ":" + field2
build_go_concat_expr([], _, '""') :- !.
build_go_concat_expr([Single], _, Single) :- !.
build_go_concat_expr([First|Rest], Delim, Expr) :-
    build_go_concat_expr_rest(Rest, Delim, RestExpr),
    format(atom(Expr), '~w + "~s" + ~w', [First, Delim, RestExpr]).

build_go_concat_expr_rest([Single], _, Single) :- !.
build_go_concat_expr_rest([First|Rest], Delim, Expr) :-
    build_go_concat_expr_rest(Rest, Delim, RestExpr),
    format(atom(Expr), '~w + "~s" + ~w', [First, Delim, RestExpr]).

%% Helper functions
build_var_map(HeadArgs, VarMap) :-
    build_var_map_(HeadArgs, 1, VarMap).

build_var_map_([], _, []).
build_var_map_([Arg|Rest], Pos, [(Arg, Pos)|RestMap]) :-
    NextPos is Pos + 1,
    build_var_map_(Rest, NextPos, RestMap).

extract_predicates(true, []) :- !.
extract_predicates((A, B), Predicates) :- !,
    extract_predicates(A, P1),
    extract_predicates(B, P2),
    append(P1, P2, Predicates).
extract_predicates(Goal, [Goal]) :-
    functor(Goal, Functor, _),
    Functor \= ',',
    Functor \= true.

%% ============================================
%% MULTIPLE RULES COMPILATION
%% ============================================

%% compile_multiple_rules_to_go(+Pred, +Arity, +Clauses, +RecordDelim, +FieldDelim, +Unique, -GoCode)
%  Compile multiple rules (OR pattern) to Go code
%
compile_multiple_rules_to_go(_Pred, _Arity, _Clauses, _RecordDelim, _FieldDelim, _Unique, GoCode) :-
    % TODO: Implement multiple rules compilation
    GoCode = '\t// TODO: Multiple rules compilation not yet implemented\n'.

%% ============================================
%% GO PROGRAM GENERATION
%% ============================================

%% generate_go_program(+Pred, +Arity, +RecordDelim, +FieldDelim, +Quoting, +EscapeChar, +Body, -GoCode)
%  Generate complete Go program with imports and main function
%
generate_go_program(Pred, Arity, RecordDelim, FieldDelim, Quoting, EscapeChar, Body, GoCode) :-
    atom_string(Pred, PredStr),

    % Generate program template
    % TODO: Dynamically include only needed imports based on Body content
    format(string(GoCode), 'package main

import (
\t"bufio"
\t"fmt"
\t"os"
\t"strings"
)

func main() {
~s}
', [Body]).

%% ============================================
%% UTILITY FUNCTIONS
%% ============================================

%% map_field_delimiter(+Delimiter, -String)
%  Map delimiter atom to string
map_field_delimiter(colon, ':') :- !.
map_field_delimiter(tab, '\t') :- !.
map_field_delimiter(comma, ',') :- !.
map_field_delimiter(pipe, '|') :- !.
map_field_delimiter(Char, Char) :- atom(Char), atom_length(Char, 1), !.

%% write_go_program(+GoCode, +FilePath)
%  Write Go code to file
%
write_go_program(GoCode, FilePath) :-
    open(FilePath, write, Stream),
    write(Stream, GoCode),
    close(Stream),
    format('Go program written to: ~w~n', [FilePath]).
