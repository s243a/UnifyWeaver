:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% awk_target.pl - AWK Target for UnifyWeaver
% Generates self-contained AWK scripts for pattern matching and data processing
% Leverages AWK's associative arrays and regex capabilities

:- module(awk_target, [
    compile_predicate_to_awk/3,     % +Predicate, +Options, -AwkCode
    write_awk_script/2              % +AwkCode, +FilePath
]).

:- use_module(library(lists)).

%% ============================================
%% PUBLIC API
%% ============================================

%% compile_predicate_to_awk(+Predicate, +Options, -AwkCode)
%  Compile a Prolog predicate to AWK code
%
%  @arg Predicate Predicate indicator (Name/Arity)
%  @arg Options List of options
%  @arg AwkCode Generated AWK code as atom
%
%  Options:
%  - record_format(jsonl|tsv|csv) - Input format (default: tsv)
%  - field_separator(Char) - Field separator for tsv (default: '\t')
%  - include_header(true|false) - Include shebang (default: true)
%  - unique(true|false) - Deduplicate results (default: true)
%  - unordered(true|false) - Allow unordered output (default: true)
%  - aggregation(sum|count|max|min|avg) - Aggregation operation
%
compile_predicate_to_awk(PredIndicator, Options, AwkCode) :-
    PredIndicator = Pred/Arity,
    format('=== Compiling ~w/~w to AWK ===~n', [Pred, Arity]),

    % Check if this is an aggregation operation
    (   option(aggregation(AggOp), Options, none),
        AggOp \= none
    ->  % Compile as aggregation
        compile_aggregation_to_awk(Pred, Arity, AggOp, Options, AwkCode)
    ;   % Check if this is a tail-recursive pattern
        functor(Head, Pred, Arity),
        findall(Head-Body, user:clause(Head, Body), Clauses),
        is_tail_recursive_pattern(Pred, Clauses)
    ->  % Compile as tail recursion (while loop)
        compile_tail_recursion_to_awk(Pred, Arity, Clauses, Options, AwkCode)
    ;   % Continue with normal compilation
        compile_predicate_to_awk_normal(Pred, Arity, Options, AwkCode)
    ).

%% compile_predicate_to_awk_normal(+Pred, +Arity, +Options, -AwkCode)
%  Normal (non-aggregation) compilation path
%
compile_predicate_to_awk_normal(Pred, Arity, Options, AwkCode) :-

    % Get options
    option(record_format(RecordFormat), Options, tsv),
    option(field_separator(FieldSep), Options, '\t'),
    option(include_header(IncludeHeader), Options, true),
    option(unique(Unique), Options, true),
    option(unordered(Unordered), Options, true),

    % Create head with correct arity
    functor(Head, Pred, Arity),

    % Get all clauses for this predicate (preserving variable sharing)
    findall(Head-Body, user:clause(Head, Body), Clauses),

    % Determine compilation strategy
    (   Clauses = [] ->
        format('ERROR: No clauses found for ~w/~w~n', [Pred, Arity]),
        fail
    ;   maplist(is_fact_clause, Clauses) ->
        % All bodies are 'true' - these are facts
        format('Type: facts (~w clauses)~n', [length(Clauses)]),
        compile_facts_to_awk(Pred, Arity, Clauses, RecordFormat, FieldSep,
                            Unique, Unordered, ScriptBody)
    ;   Clauses = [SingleHead-SingleBody], SingleBody \= true ->
        % Single rule
        format('Type: single_rule~n'),
        compile_single_rule_to_awk(Pred, Arity, SingleHead, SingleBody, RecordFormat,
                                  FieldSep, Unique, Unordered, ScriptBody)
    ;   % Multiple rules (OR pattern)
        format('Type: multiple_rules (~w clauses)~n', [length(Clauses)]),
        compile_multiple_rules_to_awk(Pred, Arity, Clauses, RecordFormat,
                                     FieldSep, Unique, Unordered, ScriptBody)
    ),

    % Optionally add shebang header
    (   IncludeHeader ->
        generate_awk_header(Pred, Arity, RecordFormat, Header),
        atomic_list_concat([Header, '\n', ScriptBody], AwkCode)
    ;   AwkCode = ScriptBody
    ),
    !.

%% write_awk_script(+AwkCode, +FilePath)
%  Write AWK script to file and make it executable
%
write_awk_script(AwkCode, FilePath) :-
    open(FilePath, write, Stream),
    format(Stream, '~w', [AwkCode]),
    close(Stream),
    % Make executable
    format(atom(ChmodCmd), 'chmod +x ~w', [FilePath]),
    shell(ChmodCmd),
    format('[AwkTarget] Generated executable AWK script: ~w~n', [FilePath]).

%% ============================================
%% AGGREGATION PATTERN COMPILATION
%% ============================================

%% compile_aggregation_to_awk(+Pred, +Arity, +AggOp, +Options, -AwkCode)
%  Compile aggregation operations (sum, count, max, min, avg)
%
compile_aggregation_to_awk(Pred, Arity, AggOp, Options, AwkCode) :-
    atom_string(Pred, _PredStr),
    option(field_separator(FieldSep), Options, '\t'),
    option(include_header(IncludeHeader), Options, true),

    format('  Aggregation type: ~w~n', [AggOp]),

    % Generate aggregation AWK code based on operation
    generate_aggregation_awk(AggOp, Arity, FieldSep, ScriptBody),

    % Optionally add shebang header
    (   IncludeHeader ->
        generate_awk_header(Pred, Arity, tsv, Header),
        atomic_list_concat([Header, '\n', ScriptBody], AwkCode)
    ;   AwkCode = ScriptBody
    ).

%% generate_aggregation_awk(+AggOp, +Arity, +FieldSep, -AwkCode)
%  Generate AWK code for specific aggregation operations
%
generate_aggregation_awk(sum, _Arity, FieldSep, AwkCode) :-
    format(atom(AwkCode),
'BEGIN { FS = "~w" }
{ sum += $1 }
END { print sum }', [FieldSep]).

generate_aggregation_awk(count, _Arity, FieldSep, AwkCode) :-
    format(atom(AwkCode),
'BEGIN { FS = "~w" }
{ count++ }
END { print count }', [FieldSep]).

generate_aggregation_awk(max, _Arity, FieldSep, AwkCode) :-
    format(atom(AwkCode),
'BEGIN { FS = "~w"; max = -999999999 }
{ if ($1 > max) max = $1 }
END { print max }', [FieldSep]).

generate_aggregation_awk(min, _Arity, FieldSep, AwkCode) :-
    format(atom(AwkCode),
'BEGIN { FS = "~w"; min = 999999999 }
{ if (NR == 1 || $1 < min) min = $1 }
END { print min }', [FieldSep]).

generate_aggregation_awk(avg, _Arity, FieldSep, AwkCode) :-
    format(atom(AwkCode),
'BEGIN { FS = "~w"; sum = 0; count = 0 }
{ sum += $1; count++ }
END { if (count > 0) print sum/count; else print 0 }', [FieldSep]).

%% ============================================
%% TAIL RECURSION → WHILE LOOP COMPILATION
%% ============================================

%% is_tail_recursive_pattern(+Pred, +Clauses)
%  Detect if clauses form a tail-recursive pattern
%  Pattern: base case + recursive case with self-call at end
%
is_tail_recursive_pattern(Pred, Clauses) :-
    length(Clauses, Len),
    Len >= 2,  % Need at least base case + recursive case
    % Check for base case (no body or simple body)
    member(_Head1-Body1, Clauses),
    is_base_case(Body1),
    % Check for recursive case
    member(_Head2-Body2, Clauses),
    is_tail_recursive_body(Pred, Body2).

%% is_base_case(+Body)
%  Check if body is a base case (true or simple unification)
%
is_base_case(true) :- !.
is_base_case(_A = _B) :- !.  % Simple unification

%% is_tail_recursive_body(+Pred, +Body)
%  Check if body ends with recursive call to Pred
%
is_tail_recursive_body(Pred, Body) :-
    extract_last_goal(Body, LastGoal),
    functor(LastGoal, Pred, _).

%% extract_last_goal(+Body, -LastGoal)
%  Extract the last goal from a body
%
extract_last_goal((_, B), LastGoal) :- !,
    extract_last_goal(B, LastGoal).
extract_last_goal(Goal, Goal).

%% compile_tail_recursion_to_awk(+Pred, +Arity, +Clauses, +Options, -AwkCode)
%  Compile tail-recursive pattern as AWK while loop
%
compile_tail_recursion_to_awk(Pred, Arity, Clauses, Options, AwkCode) :-
    format('  Type: tail_recursion~n'),

    option(field_separator(FieldSep), Options, '\t'),
    option(include_header(IncludeHeader), Options, true),

    % Separate base case and recursive case
    separate_tail_rec_clauses(Pred, Clauses, BaseClause, RecClause),

    % Extract loop components
    BaseClause = BaseHead-_BaseBody,
    RecClause = RecHead-RecBody,

    BaseHead =.. [_|BaseArgs],
    RecHead =.. [_|RecArgs],

    % Analyze the pattern
    analyze_tail_recursion(BaseArgs, RecArgs, RecBody, LoopInfo),

    % Generate AWK while loop
    generate_tail_rec_awk(Pred, Arity, LoopInfo, FieldSep, ScriptBody),

    % Optionally add shebang header
    (   IncludeHeader ->
        generate_awk_header(Pred, Arity, tsv, Header),
        atomic_list_concat([Header, '\n', ScriptBody], AwkCode)
    ;   AwkCode = ScriptBody
    ).

%% separate_tail_rec_clauses(+Pred, +Clauses, -BaseClause, -RecClause)
%  Separate base case from recursive case
%
separate_tail_rec_clauses(Pred, Clauses, BaseClause, RecClause) :-
    % Find base case
    member(BaseClause, Clauses),
    BaseClause = _-Body1,
    is_base_case(Body1),
    % Find recursive case
    member(RecClause, Clauses),
    RecClause = _-Body2,
    is_tail_recursive_body(Pred, Body2),
    BaseClause \= RecClause.

%% analyze_tail_recursion(+BaseArgs, +RecArgs, +RecBody, -LoopInfo)
%  Analyze tail recursion to extract loop components
%
%  For factorial(N, Acc, Result):
%  - Base: factorial(0, Acc, Acc)
%  - Rec:  factorial(N, Acc, F) :- N > 0, N1 is N-1, Acc1 is Acc*N, factorial(N1, Acc1, F)
%
%  Extract:
%  - Loop variable (N) and condition (N > 0)
%  - Accumulator updates (Acc1 is Acc*N)
%  - Loop variable update (N1 is N-1)
%
analyze_tail_recursion(BaseArgs, RecArgs, RecBody, LoopInfo) :-
    % Extract constraints and updates from recursive body
    extract_loop_components(RecBody, Constraints, Updates, RecCall),

    % Build loop info using functor (now including RecCall)
    LoopInfo = loop_info(BaseArgs, RecArgs, Constraints, Updates, RecCall).

%% extract_loop_components(+Body, -Constraints, -Updates, -RecCall)
%  Extract constraints, updates, and recursive call from body
%
extract_loop_components(Body, Constraints, Updates, RecCall) :-
    body_to_list(Body, Goals),
    partition_goals(Goals, Constraints, Updates, RecCall).

%% body_to_list(+Body, -Goals)
%  Convert body to list of goals
%
body_to_list((A, B), Goals) :- !,
    body_to_list(A, G1),
    body_to_list(B, G2),
    append(G1, G2, Goals).
body_to_list(Goal, [Goal]).

%% partition_goals(+Goals, -Constraints, -Updates, -RecCall)
%  Partition goals into constraints, updates, and recursive call
%
partition_goals([], [], [], none).
partition_goals([Goal|Rest], [Goal|RestC], Updates, RecCall) :-
    is_constraint_goal(Goal), !,
    partition_goals(Rest, RestC, Updates, RecCall).
partition_goals([Goal|Rest], Constraints, [Goal|RestU], RecCall) :-
    is_update_goal(Goal), !,
    partition_goals(Rest, Constraints, RestU, RecCall).
partition_goals([Goal|Rest], Constraints, Updates, Goal) :-
    % Last goal should be recursive call
    Rest = [], !,
    partition_goals(Rest, Constraints, Updates, _).
partition_goals([_|Rest], Constraints, Updates, RecCall) :-
    % Skip other goals
    partition_goals(Rest, Constraints, Updates, RecCall).

%% is_constraint_goal(+Goal)
%  Check if goal is a constraint (comparison)
%
is_constraint_goal(_ > _).
is_constraint_goal(_ < _).
is_constraint_goal(_ >= _).
is_constraint_goal(_ =< _).
is_constraint_goal(_ =:= _).
is_constraint_goal(_ =\= _).
is_constraint_goal(_ \= _).

%% is_update_goal(+Goal)
%  Check if goal is an update (is/2)
%
is_update_goal(is(_, _)).

%% generate_tail_rec_awk(+Pred, +Arity, +LoopInfo, +FieldSep, -AwkCode)
%  Generate AWK while loop from tail recursion pattern
%
%  Example: factorial(N, Acc, Result)
%  Base: factorial(0, Acc, Acc)
%  Rec:  factorial(N, Acc, F) :- N > 0, N1 is N-1, Acc1 is Acc*N, factorial(N1, Acc1, F)
%
%  AWK:
%  BEGIN { FS = "\t" }
%  {
%      n = $1; acc = $2
%      while (n > 0) {
%          new_acc = acc * n
%          new_n = n - 1
%          acc = new_acc
%          n = new_n
%      }
%      print acc
%  }
%
generate_tail_rec_awk(_Pred, Arity, LoopInfo, FieldSep, AwkCode) :-
    % Extract from functor-based structure
    LoopInfo = loop_info(BaseArgs, RecArgs, Constraints, Updates, RecCall),

    % Build variable initializations from input fields
    length(RecArgs, NumArgs),
    NumArgs =:= Arity,
    findall(Init,
        (   between(1, Arity, I),
            nth1(I, RecArgs, Arg),
            var_to_awk_name(Arg, VarName),
            format(atom(Init), '    ~w = $~w', [VarName, I])
        ),
        Inits),
    atomic_list_concat(Inits, '\n', InitCode),

    % Build loop condition
    (   Constraints = [Constraint|_] ->
        constraint_to_awk_simple(Constraint, CondCode)
    ;   CondCode = '1'  % Default: always loop once
    ),

    % Extract recursive call arguments to map temp vars to loop vars
    RecCall =.. [_|RecCallArgs],

    % Build loop body: compute new values, then update loop variables
    findall(UpdateStmt,
        (   member(is(Var, Expr), Updates),
            var_to_awk_name(Var, VarName),
            expr_to_awk_simple(Expr, ExprCode),
            format(atom(UpdateStmt), '        ~w = ~w', [VarName, ExprCode])
        ),
        TempUpdates),

    % Add statements to copy temp vars back to loop vars
    findall(CopyStmt,
        (   between(1, Arity, I),
            nth1(I, RecCallArgs, RecArg),
            nth1(I, RecArgs, OrigArg),
            % Only copy if the recursive call uses a different variable
            RecArg \== OrigArg,
            var_to_awk_name(OrigArg, OrigName),
            var_to_awk_name(RecArg, RecName),
            format(atom(CopyStmt), '        ~w = ~w', [OrigName, RecName])
        ),
        CopyStmts),

    append(TempUpdates, CopyStmts, AllUpdates),
    atomic_list_concat(AllUpdates, '\n', UpdateCode),

    % Find result variable from base case
    % In base case like factorial(0, Acc, Acc), the result (last arg)
    % is unified with another variable. Find which one.
    nth1(Arity, BaseArgs, BaseResultArg),
    % Check if the last base arg is the same variable as an earlier position
    (   findall(Pos,
            (   between(1, Arity, Pos),
                Pos < Arity,
                nth1(Pos, BaseArgs, BaseVar),
                BaseVar == BaseResultArg
            ),
            [ResultPos|_])
    ->  % Found earlier position with same variable
        nth1(ResultPos, RecArgs, ResultVar)
    ;   % No match, use last argument
        nth1(Arity, RecArgs, ResultVar)
    ),
    var_to_awk_name(ResultVar, ResultName),

    format(atom(AwkCode),
'BEGIN { FS = "~w" }
{
~w
    while (~w) {
~w
    }
    print ~w
}', [FieldSep, InitCode, CondCode, UpdateCode, ResultName]).

%% var_to_awk_name(+Var, -AwkName)
%  Convert Prolog variable to AWK variable name
%
var_to_awk_name(Var, AwkName) :-
    var(Var), !,
    term_to_atom(Var, VarAtom),
    downcase_atom(VarAtom, Lower),
    atom_string(Lower, AwkName).
var_to_awk_name(Atom, AwkName) :-
    atom(Atom), !,
    downcase_atom(Atom, Lower),
    atom_string(Lower, AwkName).
var_to_awk_name(Term, AwkName) :-
    term_to_atom(Term, Atom),
    downcase_atom(Atom, Lower),
    atom_string(Lower, AwkName).

%% constraint_to_awk_simple(+Constraint, -AwkCode)
%  Convert simple constraint to AWK condition
%
constraint_to_awk_simple(A > B, AwkCode) :-
    expr_to_awk_simple(A, AwkA),
    expr_to_awk_simple(B, AwkB),
    format(atom(AwkCode), '~w > ~w', [AwkA, AwkB]).
constraint_to_awk_simple(A < B, AwkCode) :-
    expr_to_awk_simple(A, AwkA),
    expr_to_awk_simple(B, AwkB),
    format(atom(AwkCode), '~w < ~w', [AwkA, AwkB]).
constraint_to_awk_simple(A >= B, AwkCode) :-
    expr_to_awk_simple(A, AwkA),
    expr_to_awk_simple(B, AwkB),
    format(atom(AwkCode), '~w >= ~w', [AwkA, AwkB]).
constraint_to_awk_simple(A =< B, AwkCode) :-
    expr_to_awk_simple(A, AwkA),
    expr_to_awk_simple(B, AwkB),
    format(atom(AwkCode), '~w =< ~w', [AwkA, AwkB]).

%% expr_to_awk_simple(+Expr, -AwkExpr)
%  Convert simple expression to AWK
%
expr_to_awk_simple(Var, AwkVar) :-
    var(Var), !,
    var_to_awk_name(Var, AwkVar).
expr_to_awk_simple(Atom, AwkAtom) :-
    atom(Atom), !,
    downcase_atom(Atom, Lower),
    atom_string(Lower, AwkAtom).
expr_to_awk_simple(Num, Num) :-
    number(Num), !.
% Handle A + (-N) as A - N
expr_to_awk_simple(A + B, AwkExpr) :-
    number(B),
    B < 0, !,
    PosB is -B,
    expr_to_awk_simple(A, AwkA),
    format(atom(AwkExpr), '(~w - ~w)', [AwkA, PosB]).
expr_to_awk_simple(A + B, AwkExpr) :- !,
    expr_to_awk_simple(A, AwkA),
    expr_to_awk_simple(B, AwkB),
    format(atom(AwkExpr), '(~w + ~w)', [AwkA, AwkB]).
expr_to_awk_simple(A - B, AwkExpr) :- !,
    expr_to_awk_simple(A, AwkA),
    expr_to_awk_simple(B, AwkB),
    format(atom(AwkExpr), '(~w - ~w)', [AwkA, AwkB]).
expr_to_awk_simple(A * B, AwkExpr) :- !,
    expr_to_awk_simple(A, AwkA),
    expr_to_awk_simple(B, AwkB),
    format(atom(AwkExpr), '(~w * ~w)', [AwkA, AwkB]).
expr_to_awk_simple(A / B, AwkExpr) :- !,
    expr_to_awk_simple(A, AwkA),
    expr_to_awk_simple(B, AwkB),
    format(atom(AwkExpr), '(~w / ~w)', [AwkA, AwkB]).

%% ============================================
%% HEADER GENERATION
%% ============================================

generate_awk_header(Pred, Arity, RecordFormat, Header) :-
    get_unifyweaver_version(Version),
    get_time(Timestamp),
    format_time(atom(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),

    format(atom(Header),
'#!/usr/bin/awk -f
# Generated by UnifyWeaver ~w
# Target: AWK
# Generated: ~w
# Predicate: ~w/~w
# Input format: ~w
', [Version, DateStr, Pred, Arity, RecordFormat]).

get_unifyweaver_version('v0.0.3').

%% ============================================
%% FACT COMPILATION
%% ============================================

%% compile_facts_to_awk(+Pred, +Arity, +Clauses, +RecordFormat, +FieldSep,
%%                      +Unique, +Unordered, -AwkCode)
%  Compile facts to AWK using associative arrays
%
compile_facts_to_awk(Pred, Arity, Clauses, _RecordFormat, FieldSep,
                     Unique, _Unordered, AwkCode) :-
    atom_string(Pred, _PredStr),

    % Extract facts from clauses
    findall(Args,
        (member(Head-true, Clauses), Head =.. [_|Args]),
        AllFacts),

    % Generate BEGIN block with fact data
    generate_facts_begin_block(AllFacts, FieldSep, BeginBlock),

    % Generate main block for matching
    generate_facts_main_block(Arity, FieldSep, Unique, MainBlock),

    % Generate END block if needed (for unique output)
    (   Unique ->
        generate_unique_end_block(EndBlock)
    ;   EndBlock = ''
    ),

    % Assemble complete AWK script
    atomic_list_concat([BeginBlock, '\n', MainBlock, '\n', EndBlock], AwkCode).

%% generate_facts_begin_block(+Facts, +FieldSep, -BeginBlock)
%  Generate BEGIN block that loads facts into associative array
%
generate_facts_begin_block(Facts, FieldSep, BeginBlock) :-
    % Build array initialization statements
    findall(InitStmt,
        (   member(Args, Facts),
            atomic_list_concat(Args, ':', Key),
            format(atom(InitStmt), '    facts["~w"] = 1', [Key])
        ),
        InitStmts),
    atomic_list_concat(InitStmts, '\n', InitStmtsStr),

    format(atom(BeginBlock),
'BEGIN {
    FS = "~w"
~w
}', [FieldSep, InitStmtsStr]).

%% generate_facts_main_block(+Arity, +FieldSep, +Unique, -MainBlock)
%  Generate main pattern-action block for fact lookup
%
generate_facts_main_block(Arity, _FieldSep, Unique, MainBlock) :-
    % Build field concatenation
    numlist(1, Arity, FieldNums),
    findall(Field,
        (member(N, FieldNums), format(atom(Field), '$~w', [N])),
        Fields),
    atomic_list_concat(Fields, ' ":" ', FieldsConcat),

    % Build lookup and output
    (   Unique ->
        % Store in seen array for deduplication
        format(atom(MainBlock),
'{
    key = ~w
    if (key in facts && !(key in seen)) {
        seen[key] = 1
        print $0
    }
}', [FieldsConcat])
    ;   % No deduplication
        format(atom(MainBlock),
'{
    key = ~w
    if (key in facts) {
        print $0
    }
}', [FieldsConcat])
    ).

%% generate_unique_end_block(-EndBlock)
%  Generate END block (currently empty, but available for future use)
%
generate_unique_end_block('').

%% ============================================
%% SINGLE RULE COMPILATION
%% ============================================

%% compile_single_rule_to_awk(+Pred, +Arity, +Head, +Body, +RecordFormat,
%%                            +FieldSep, +Unique, +Unordered, -AwkCode)
%  Compile a single rule to AWK
%
compile_single_rule_to_awk(Pred, Arity, Head, Body, _RecordFormat, FieldSep,
                          Unique, _Unordered, AwkCode) :-
    atom_string(Pred, PredStr),

    % Build variable mapping from head arguments to field positions
    Head =.. [_|HeadArgs],
    build_var_map(HeadArgs, VarMap),
    format('  Variable map: ~w~n', [VarMap]),

    % Extract predicates and constraints from body
    extract_predicates(Body, Predicates),
    extract_constraints(Body, Constraints),
    format('  Body predicates: ~w~n', [Predicates]),
    format('  Constraints: ~w~n', [Constraints]),

    % Determine compilation strategy
    (   Predicates = [] ->
        % No predicates - just constraints or empty body
        compile_constraint_only_rule(PredStr, Arity, Constraints, VarMap, FieldSep, Unique, AwkCode)
    ;   Predicates = [SinglePred] ->
        % Single predicate - simple lookup with optional constraints
        compile_single_predicate_rule(PredStr, Arity, SinglePred, Constraints, VarMap,
                                     FieldSep, Unique, AwkCode)
    ;   % Multiple predicates - hash join pipeline
        compile_multi_predicate_rule(PredStr, Arity, Predicates, Constraints, VarMap,
                                    FieldSep, Unique, AwkCode)
    ).

%% build_var_map(+HeadArgs, -VarMap)
%  Build a mapping from variables to field positions
%  e.g., [X, Y, Z] -> [(X, 1), (Y, 2), (Z, 3)]
%
build_var_map(HeadArgs, VarMap) :-
    build_var_map_(HeadArgs, 1, VarMap).

build_var_map_([], _, []).
build_var_map_([Arg|Rest], Pos, [(Arg, Pos)|RestMap]) :-
    NextPos is Pos + 1,
    build_var_map_(Rest, NextPos, RestMap).

%% var_map_lookup(+Var, +VarMap, -Pos)
%  Look up a variable in VarMap using identity (==) not unification (=)
%
var_map_lookup(Var, [(MapVar, Pos)|_], Pos) :-
    Var == MapVar, !.
var_map_lookup(Var, [_|Rest], Pos) :-
    var_map_lookup(Var, Rest, Pos).

%% compile_constraint_only_rule(+PredStr, +Arity, +Constraints, +VarMap, +FieldSep, +Unique, -AwkCode)
%  Compile a rule with no predicates (just constraints or empty body)
%
compile_constraint_only_rule(_PredStr, _Arity, Constraints, VarMap, FieldSep, Unique, AwkCode) :-
    (   Constraints = [] ->
        % No constraints - just pass through (always true)
        MainBlock = '{ print $0 }'
    ;   % Has constraints - generate AWK condition
        generate_constraint_code(Constraints, VarMap, ConstraintCode),
        (   Unique ->
            format(atom(MainBlock),
'{
    if (~w) {
        key = $0
        if (!(key in seen)) {
            seen[key] = 1
            print $0
        }
    }
}', [ConstraintCode])
        ;   format(atom(MainBlock),
'{
    if (~w) {
        print $0
    }
}', [ConstraintCode])
        )
    ),
    format(atom(BeginBlock), 'BEGIN { FS = "~w" }', [FieldSep]),
    atomic_list_concat([BeginBlock, '\n', MainBlock], AwkCode).

%% compile_single_predicate_rule(+PredStr, +Arity, +Predicate, +Constraints,
%%                                +VarMap, +FieldSep, +Unique, -AwkCode)
%  Compile a rule with a single predicate (simple lookup)
%
compile_single_predicate_rule(_PredStr, _Arity, Predicate, Constraints, VarMap, FieldSep, Unique, AwkCode) :-
    % Need to load the predicate's facts
    atom_string(Predicate, PredLookupStr),

    % Get the predicate's arity by finding first clause
    current_predicate(Predicate/PredArity),
    functor(PredGoal, Predicate, PredArity),
    findall(PredGoal, user:clause(PredGoal, true), PredFacts),

    % Generate fact loading in BEGIN block
    findall(InitStmt,
        (   member(Fact, PredFacts),
            Fact =.. [_|Args],
            atomic_list_concat(Args, ':', Key),
            format(atom(InitStmt), '    ~w_data["~w"] = 1', [PredLookupStr, Key])
        ),
        InitStmts),
    atomic_list_concat(InitStmts, '\n', InitStmtsStr),

    % Generate main lookup logic
    % This depends on how the input maps to the predicate arguments
    % For now, assume direct mapping
    (   Constraints = [] ->
        ConstraintCheck = ''
    ;   generate_constraint_code(Constraints, VarMap, ConstraintCode),
        format(atom(ConstraintCheck), ' && ~w', [ConstraintCode])
    ),

    % Build field concatenation based on predicate arity
    numlist(1, PredArity, FieldNums),
    findall(Field,
        (member(N, FieldNums), format(atom(Field), '$~w', [N])),
        Fields),
    atomic_list_concat(Fields, ' ":" ', FieldsConcat),

    (   Unique ->
        format(atom(MainBlock),
'{
    key = ~w
    if (key in ~w_data~w) {
        if (!(key in seen)) {
            seen[key] = 1
            print $0
        }
    }
}', [FieldsConcat, PredLookupStr, ConstraintCheck])
    ;   format(atom(MainBlock),
'{
    key = ~w
    if (key in ~w_data~w) {
        print $0
    }
}', [FieldsConcat, PredLookupStr, ConstraintCheck])
    ),

    format(atom(BeginBlock),
'BEGIN {
    FS = "~w"
~w
}', [FieldSep, InitStmtsStr]),
    atomic_list_concat([BeginBlock, '\n', MainBlock], AwkCode).

%% compile_multi_predicate_rule(+PredStr, +Arity, +Predicates, +Constraints,
%%                              +VarMap, +FieldSep, +Unique, -AwkCode)
%  Compile a rule with multiple predicates (hash join)
%
compile_multi_predicate_rule(_PredStr, _Arity, Predicates, Constraints, _VarMap, FieldSep, Unique, AwkCode) :-
    % For multiple predicates, we need to implement a hash join
    % Strategy: Load all predicate facts, then perform nested joins

    % Load all predicates' facts into arrays
    findall(LoadCode,
        (   member(Pred, Predicates),
            generate_predicate_loader(Pred, LoadCode)
        ),
        LoadCodes),
    atomic_list_concat(LoadCodes, '\n', AllLoadCode),

    % Generate join logic
    generate_join_logic(Predicates, Constraints, Unique, JoinLogic),

    format(atom(BeginBlock),
'BEGIN {
    FS = "~w"
~w
}', [FieldSep, AllLoadCode]),

    format(atom(EndBlock),
'END {
~w
}', [JoinLogic]),

    atomic_list_concat([BeginBlock, '\n# No input processing needed\n', EndBlock], AwkCode).

%% ============================================
%% HELPER FUNCTIONS FOR SINGLE RULE COMPILATION
%% ============================================

%% generate_constraint_code(+Constraints, +VarMap, -AwkCode)
%  Generate AWK code for constraint checking
%
generate_constraint_code([], _, 'true') :- !.
generate_constraint_code(Constraints, VarMap, AwkCode) :-
    findall(ConstraintCode,
        (   member(Constraint, Constraints),
            constraint_to_awk(Constraint, VarMap, ConstraintCode)
        ),
        ConstraintCodes),
    atomic_list_concat(ConstraintCodes, ' && ', AwkCode).

%% constraint_to_awk(+Constraint, +VarMap, -AwkCode)
%  Convert a single constraint to AWK code
%
constraint_to_awk(inequality(A, B), VarMap, AwkCode) :-
    term_to_awk_expr(A, VarMap, AwkA),
    term_to_awk_expr(B, VarMap, AwkB),
    format(atom(AwkCode), '(~w != ~w)', [AwkA, AwkB]).
constraint_to_awk(gt(A, B), VarMap, AwkCode) :-
    term_to_awk_expr(A, VarMap, AwkA),
    term_to_awk_expr(B, VarMap, AwkB),
    format(atom(AwkCode), '(~w > ~w)', [AwkA, AwkB]).
constraint_to_awk(lt(A, B), VarMap, AwkCode) :-
    term_to_awk_expr(A, VarMap, AwkA),
    term_to_awk_expr(B, VarMap, AwkB),
    format(atom(AwkCode), '(~w < ~w)', [AwkA, AwkB]).
constraint_to_awk(gte(A, B), VarMap, AwkCode) :-
    term_to_awk_expr(A, VarMap, AwkA),
    term_to_awk_expr(B, VarMap, AwkB),
    format(atom(AwkCode), '(~w >= ~w)', [AwkA, AwkB]).
constraint_to_awk(lte(A, B), VarMap, AwkCode) :-
    term_to_awk_expr(A, VarMap, AwkA),
    term_to_awk_expr(B, VarMap, AwkB),
    format(atom(AwkCode), '(~w <= ~w)', [AwkA, AwkB]).
constraint_to_awk(eq(A, B), VarMap, AwkCode) :-
    term_to_awk_expr(A, VarMap, AwkA),
    term_to_awk_expr(B, VarMap, AwkB),
    format(atom(AwkCode), '(~w == ~w)', [AwkA, AwkB]).
constraint_to_awk(neq(A, B), VarMap, AwkCode) :-
    term_to_awk_expr(A, VarMap, AwkA),
    term_to_awk_expr(B, VarMap, AwkB),
    format(atom(AwkCode), '(~w != ~w)', [AwkA, AwkB]).
constraint_to_awk(is(A, B), VarMap, AwkCode) :-
    term_to_awk_expr(A, VarMap, AwkA),
    term_to_awk_expr(B, VarMap, AwkB),
    format(atom(AwkCode), '((~w = ~w), 1)', [AwkA, AwkB]).

%% term_to_awk_expr(+Term, +VarMap, -AwkExpr)
%  Convert a Prolog term to an AWK expression using variable mapping
%
term_to_awk_expr(Term, VarMap, AwkExpr) :-
    var(Term), !,
    % It's a Prolog variable - look it up in VarMap using identity check
    (   var_map_lookup(Term, VarMap, Pos) ->
        format(atom(AwkExpr), '$~w', [Pos])
    ;   % Variable not in map - might be from nested term
        AwkExpr = Term
    ).
term_to_awk_expr(Term, _, AwkExpr) :-
    atom(Term), !,
    % Atom constant - quote it for AWK
    format(atom(AwkExpr), '"~w"', [Term]).
term_to_awk_expr(Term, _, Term) :-
    number(Term), !.
term_to_awk_expr(Term, VarMap, AwkExpr) :-
    compound(Term), !,
    Term =.. [Op, Left, Right],
    term_to_awk_expr(Left, VarMap, AwkLeft),
    term_to_awk_expr(Right, VarMap, AwkRight),
    format(atom(AwkExpr), '(~w ~w ~w)', [AwkLeft, Op, AwkRight]).
term_to_awk_expr(Term, _, Term).

%% generate_predicate_loader(+Predicate, -LoadCode)
%  Generate AWK code to load a predicate's facts into an array
%
generate_predicate_loader(Predicate, LoadCode) :-
    atom_string(Predicate, PredStr),
    current_predicate(Predicate/PredArity),
    functor(PredGoal, Predicate, PredArity),
    findall(PredGoal, user:clause(PredGoal, true), PredFacts),
    findall(InitStmt,
        (   member(Fact, PredFacts),
            Fact =.. [_|Args],
            atomic_list_concat(Args, ':', Key),
            format(atom(InitStmt), '    ~w_data["~w"] = 1', [PredStr, Key])
        ),
        InitStmts),
    atomic_list_concat(InitStmts, '\n', LoadCode).

%% generate_join_logic(+Predicates, +Constraints, +Unique, -JoinLogic)
%  Generate AWK code for hash join of multiple predicates
%
generate_join_logic([Pred1, Pred2|_Rest], _Constraints, Unique, JoinLogic) :-
    % Simple two-way join for now
    % More complex joins would require analyzing variable bindings
    atom_string(Pred1, Pred1Str),
    atom_string(Pred2, Pred2Str),

    % Generate nested loop join (simple but works)
    (   Unique ->
        format(atom(JoinLogic),
'    for (key1 in ~w_data) {
        split(key1, a, ":")
        for (key2 in ~w_data) {
            split(key2, b, ":")
            # Simple join condition: last field of first matches first field of second
            if (a[length(a)] == b[1]) {
                # Build result
                result = a[1]
                for (i = 2; i <= length(a); i++) result = result ":" a[i]
                for (i = 2; i <= length(b); i++) result = result ":" b[i]
                if (!(result in seen)) {
                    seen[result] = 1
                    print result
                }
            }
        }
    }', [Pred1Str, Pred2Str])
    ;   format(atom(JoinLogic),
'    for (key1 in ~w_data) {
        split(key1, a, ":")
        for (key2 in ~w_data) {
            split(key2, b, ":")
            # Simple join condition: last field of first matches first field of second
            if (a[length(a)] == b[1]) {
                # Build result
                result = a[1]
                for (i = 2; i <= length(a); i++) result = result ":" a[i]
                for (i = 2; i <= length(b); i++) result = result ":" b[i]
                print result
            }
        }
    }', [Pred1Str, Pred2Str])
    ).

%% ============================================
%% MULTIPLE RULES COMPILATION
%% ============================================

%% compile_multiple_rules_to_awk(+Pred, +Arity, +Clauses, +RecordFormat,
%%                               +FieldSep, +Unique, +Unordered, -AwkCode)
%  Compile multiple rules (OR pattern) to AWK
%
compile_multiple_rules_to_awk(Pred, Arity, Clauses, _RecordFormat, FieldSep,
                             Unique, _Unordered, AwkCode) :-
    atom_string(Pred, _PredStr),

    % Collect all predicates used across all rule bodies
    findall(BodyPred,
        (   member(_Head-Body, Clauses),
            extract_predicates(Body, BodyPreds),
            member(BodyPred, BodyPreds)
        ),
        AllBodyPreds),
    list_to_set(AllBodyPreds, UniqueBodyPreds),

    % Load all predicate facts in BEGIN block
    findall(LoadCode,
        (   member(BodyPred, UniqueBodyPreds),
            generate_predicate_loader(BodyPred, LoadCode)
        ),
        LoadCodes),
    atomic_list_concat(LoadCodes, '\n', AllLoadCode),

    % Generate main block with OR logic for all alternatives
    generate_multi_rules_main_block(Arity, Clauses, UniqueBodyPreds, Unique, MainBlock),

    % Assemble complete script
    format(atom(BeginBlock),
'BEGIN {
    FS = "~w"
~w
}', [FieldSep, AllLoadCode]),
    atomic_list_concat([BeginBlock, '\n', MainBlock], AwkCode).

%% generate_multi_rules_main_block(+Arity, +Clauses, +LoadedPreds, +Unique, -MainBlock)
%  Generate main block for multiple rules (OR pattern)
%
generate_multi_rules_main_block(_Arity, Clauses, LoadedPreds, Unique, MainBlock) :-
    % Generate condition for each rule alternative
    findall(RuleCondition,
        (   member(Head-Body, Clauses),
            Head =.. [_|HeadArgs],
            build_var_map(HeadArgs, VarMap),
            generate_rule_condition(Body, VarMap, LoadedPreds, RuleCondition)
        ),
        RuleConditions),

    % Combine all conditions with OR logic
    (   Unique ->
        atomic_list_concat(RuleConditions, ' || ', CombinedCondition),
        format(atom(MainBlock),
'{
    # Multiple rules (OR pattern)
    if (~w) {
        key = $0
        if (!(key in seen)) {
            seen[key] = 1
            print $0
        }
    }
}', [CombinedCondition])
    ;   atomic_list_concat(RuleConditions, ' || ', CombinedCondition),
        format(atom(MainBlock),
'{
    # Multiple rules (OR pattern)
    if (~w) {
        print $0
    }
}', [CombinedCondition])
    ).

%% generate_rule_condition(+Body, +VarMap, +LoadedPreds, -Condition)
%  Generate AWK condition for a single rule body
%
generate_rule_condition(Body, VarMap, _LoadedPreds, Condition) :-
    extract_predicates(Body, Predicates),
    extract_constraints(Body, Constraints),

    % Generate predicate checks
    (   Predicates = [] ->
        PredicateCheck = '1'  % Always true if no predicates
    ;   Predicates = [SinglePred] ->
        % Single predicate lookup
        atom_string(SinglePred, PredStr),
        current_predicate(SinglePred/PredArity),
        numlist(1, PredArity, FieldNums),
        findall(Field,
            (member(N, FieldNums), format(atom(Field), '$~w', [N])),
            Fields),
        atomic_list_concat(Fields, ' ":" ', FieldsConcat),
        format(atom(PredicateCheck), '(~w in ~w_data)', [FieldsConcat, PredStr])
    ;   % Multiple predicates - for now, just check first one
        % TODO: Implement proper join logic
        Predicates = [FirstPred|_],
        atom_string(FirstPred, PredStr),
        current_predicate(FirstPred/PredArity),
        numlist(1, PredArity, FieldNums),
        findall(Field,
            (member(N, FieldNums), format(atom(Field), '$~w', [N])),
            Fields),
        atomic_list_concat(Fields, ' ":" ', FieldsConcat),
        format(atom(PredicateCheck), '(~w in ~w_data)', [FieldsConcat, PredStr])
    ),

    % Generate constraint checks
    (   Constraints = [] ->
        ConstraintCheck = ''
    ;   generate_constraint_code(Constraints, VarMap, ConstraintCode),
        format(atom(ConstraintCheck), ' && ~w', [ConstraintCode])
    ),

    % Combine checks
    format(atom(Condition), '(~w~w)', [PredicateCheck, ConstraintCheck]).

%% ============================================
%% HELPER PREDICATES
%% ============================================

%% is_fact_clause(+Clause)
%  Check if a clause is a fact (body is 'true')
%
is_fact_clause(_-true).

%% extract_predicates(+Body, -Predicates)
%  Extract predicate names from a body
%
extract_predicates(true, []) :- !.
extract_predicates((A, B), Predicates) :- !,
    extract_predicates(A, P1),
    extract_predicates(B, P2),
    append(P1, P2, Predicates).
extract_predicates(Goal, []) :-
    var(Goal), !.
% Skip inequality operators
extract_predicates(_ \= _, []) :- !.
extract_predicates(\=(_, _), []) :- !.
% Skip arithmetic operators
extract_predicates(_ > _, []) :- !.
extract_predicates(_ < _, []) :- !.
extract_predicates(_ >= _, []) :- !.
extract_predicates(_ =< _, []) :- !.
extract_predicates(_ =:= _, []) :- !.
extract_predicates(_ =\= _, []) :- !.
extract_predicates(is(_, _), []) :- !.
% Skip negation wrapper
extract_predicates(\+ A, Predicates) :- !,
    extract_predicates(A, Predicates).
extract_predicates(Goal, [Pred]) :-
    functor(Goal, Pred, _),
    Pred \= ',',
    Pred \= true.

%% extract_constraints(+Body, -Constraints)
%  Extract constraints (inequalities, arithmetic) from body
%
extract_constraints(true, []) :- !.
extract_constraints((A, B), Constraints) :- !,
    extract_constraints(A, C1),
    extract_constraints(B, C2),
    append(C1, C2, Constraints).
extract_constraints(Goal, []) :-
    var(Goal), !.
% Capture inequality constraints
extract_constraints(A \= B, [inequality(A, B)]) :- !.
extract_constraints(\=(A, B), [inequality(A, B)]) :- !.
% Capture arithmetic constraints
extract_constraints(A > B, [gt(A, B)]) :- !.
extract_constraints(A < B, [lt(A, B)]) :- !.
extract_constraints(A >= B, [gte(A, B)]) :- !.
extract_constraints(A =< B, [lte(A, B)]) :- !.
extract_constraints(A =:= B, [eq(A, B)]) :- !.
extract_constraints(A =\= B, [neq(A, B)]) :- !.
extract_constraints(is(A, B), [is(A, B)]) :- !.
% Skip predicates
extract_constraints(Goal, []) :-
    functor(Goal, Pred, _),
    Pred \= ',',
    Pred \= true.

%% option(+Option, +Options, +Default)
%  Get option value with default fallback
%
option(Option, Options, Default) :-
    (   member(Option, Options)
    ->  true
    ;   Option =.. [Key, Default],
        \+ member(Key=_, Options)
    ).
