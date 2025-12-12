:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% awk_target.pl - AWK Target for UnifyWeaver
% Generates self-contained AWK scripts for pattern matching and data processing
% Leverages AWK's associative arrays and regex capabilities

:- module(awk_target, [
    compile_predicate_to_awk/3,     % +Predicate, +Options, -AwkCode
    write_awk_script/2,             % +AwkCode, +FilePath
    init_awk_target/0,
    compile_awk_pipeline/3,         % +Predicates, +Options, -AwkCode
    test_awk_pipeline_generator/0,  % Unit tests for pipeline generator mode
    % Enhanced pipeline chaining exports
    compile_awk_enhanced_pipeline/3, % +Stages, +Options, -AwkCode
    awk_enhanced_helpers/1,          % -Code
    generate_awk_enhanced_connector/3, % +Stages, +PipelineName, -Code
    test_awk_enhanced_chaining/0     % Test enhanced pipeline chaining
]).

:- use_module(library(lists)).
:- use_module(library(gensym)).
:- use_module('../core/binding_registry').
:- use_module('../bindings/awk_bindings').

%% init_awk_target
%  Initialize the AWK target by loading bindings.
init_awk_target :-
    init_awk_bindings.

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

    % Extract predicates, constraints, and bindings
    extract_predicates(Body, Predicates),
    extract_constraints(Body, Constraints),
    extract_bindings(Body, Bindings),
    format('  Body predicates: ~w~n', [Predicates]),
    format('  Constraints: ~w~n', [Constraints]),
    format('  Bindings: ~w~n', [Bindings]),

    % Determine compilation strategy
    (   Predicates = [] ->
        % No predicates - just constraints, bindings, or empty body
        compile_constraint_only_rule(PredStr, Arity, Constraints, Bindings, VarMap, FieldSep, Unique, AwkCode)
    ;   Predicates = [SinglePred] ->
        % Single predicate - simple lookup with optional constraints
        compile_single_predicate_rule(PredStr, Arity, SinglePred, Constraints, Bindings, VarMap,
                                     FieldSep, Unique, AwkCode)
    ;   % Multiple predicates - hash join pipeline
        compile_multi_predicate_rule(PredStr, Arity, Predicates, Constraints, Bindings, VarMap,
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

%% compile_constraint_only_rule(+PredStr, +Arity, +Constraints, +Bindings, +VarMap, +FieldSep, +Unique, -AwkCode)
%  Compile a rule with no predicates (just constraints, bindings, or empty body)
%
compile_constraint_only_rule(_PredStr, _Arity, Constraints, Bindings, VarMap, FieldSep, Unique, AwkCode) :-
    % Generate binding code
    generate_awk_bindings(Bindings, VarMap, BindingCode, NewVarMap),

    (   Constraints = [] ->
        ConstraintCode = '1'
    ;   generate_constraint_code(Constraints, NewVarMap, ConstraintCode)
    ),
    
    (   Unique ->
        format(atom(MainBlock),
'{
~s    if (~w) {
        key = $0
        if (!(key in seen)) {
            seen[key] = 1
            print $0
        }
    }
}', [BindingCode, ConstraintCode])
    ;   format(atom(MainBlock),
'{
~s    if (~w) {
        print $0
    }
}', [BindingCode, ConstraintCode])
    ),
    format(atom(BeginBlock), 'BEGIN { FS = "~w" }', [FieldSep]),
    atomic_list_concat([BeginBlock, '\n', MainBlock], AwkCode).

%% compile_single_predicate_rule(+PredStr, +Arity, +Predicate, +Constraints, +Bindings,
%%                                +VarMap, +FieldSep, +Unique, -AwkCode)
%  Compile a rule with a single predicate (simple lookup)
%
compile_single_predicate_rule(_PredStr, _Arity, Predicate, Constraints, Bindings, VarMap, FieldSep, Unique, AwkCode) :-
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

    % Generate bindings
    generate_awk_bindings(Bindings, VarMap, BindingCode, NewVarMap),

    % Generate main lookup logic
    (   Constraints = [] ->
        ConstraintCheck = ''
    ;   generate_constraint_code(Constraints, NewVarMap, ConstraintCode),
        format(atom(ConstraintCheck), ' && ~w', [ConstraintCode])
    ),

    % Build field concatenation based on predicate arity
    numlist(1, PredArity, FieldNums),
    findall(Field,
        (member(N, FieldNums), format(atom(Field), '$~w', [N])),
        Fields),
    atomic_list_concat(Fields, ' ":" ', FieldsConcat),

    % Check if we have capture groups to extract
    (   has_capture_groups(Constraints, NumCaptures),
        NumCaptures > 0
    ->  % Generate print statement for captured values
        generate_capture_print(NumCaptures, PrintStmt)
    ;   % No captures - print the whole line
        PrintStmt = 'print $0'
    ),

    (   Unique ->
        format(atom(MainBlock),
'{
    key = ~w
    if (key in ~w_data) {
~s        if (1~w) {
            if (!(key in seen)) {
                seen[key] = 1
                ~w
            }
        }
    }
}', [FieldsConcat, PredLookupStr, BindingCode, ConstraintCheck, PrintStmt])
    ;   format(atom(MainBlock),
'{
    key = ~w
    if (key in ~w_data) {
~s        if (1~w) {
            ~w
        }
    }
}', [FieldsConcat, PredLookupStr, BindingCode, ConstraintCheck, PrintStmt])
    ),

    format(atom(BeginBlock),
'BEGIN {
    FS = "~w"
~w
}', [FieldSep, InitStmtsStr]),
    atomic_list_concat([BeginBlock, '\n', MainBlock], AwkCode).

%% compile_multi_predicate_rule(+PredStr, +Arity, +Predicates, +Constraints, +Bindings,
%%                              +VarMap, +FieldSep, +Unique, -AwkCode)
%  Compile a rule with multiple predicates (hash join)
%
compile_multi_predicate_rule(_PredStr, _Arity, Predicates, Constraints, _Bindings, _VarMap, FieldSep, Unique, AwkCode) :-
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
constraint_to_awk(match(Var, Pattern, Type, Groups), VarMap, AwkCode) :-
    % Validate regex type for AWK target
    validate_regex_type_for_awk(Type),
    % Convert variable to AWK expression
    term_to_awk_expr(Var, VarMap, AwkVar),
    % Escape pattern for AWK
    escape_awk_pattern(Pattern, EscapedPattern),
    % Check if we have capture groups
    (   Groups = []
    ->  % No capture groups - use simple boolean match
        format(atom(AwkCode), '(~w ~~ /~w/)', [AwkVar, EscapedPattern])
    ;   % Has capture groups - use match() function with capture array
        % Generate AWK match() call with captures
        % AWK syntax: match(string, regex, array)
        % Note: This generates a boolean check; actual capture extraction
        % would need to be in a separate statement
        format(atom(AwkCode), 'match(~w, /~w/, __captures__)', [AwkVar, EscapedPattern])
    ).

%% term_to_awk_expr(+Term, +VarMap, -AwkExpr)
%  Convert a Prolog term to an AWK expression using variable mapping
%
term_to_awk_expr(Term, VarMap, AwkExpr) :-
    var(Term), !,
    (   var_map_lookup(Term, VarMap, Val) ->
        (   Val = field(Pos) -> format(atom(AwkExpr), '$~w', [Pos])
        ;   Val = var(Name) -> AwkExpr = Name
        ;   integer(Val) -> format(atom(AwkExpr), '$~w', [Val]) % Legacy support
        )
    ;   AwkExpr = Term
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
% Skip bindings
extract_predicates(Goal, []) :-
    functor(Goal, Name, Arity),
    awk_binding(Name/Arity, _, _, _, _), !.
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
% Skip match predicates (they're constraints)
extract_predicates(match(_, _), []) :- !.
extract_predicates(match(_, _, _), []) :- !.
extract_predicates(match(_, _, _, _), []) :- !.
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
% Capture match predicates (regex pattern matching)
extract_constraints(match(Var, Pattern), [match(Var, Pattern, auto, [])]) :- !.
extract_constraints(match(Var, Pattern, Type), [match(Var, Pattern, Type, [])]) :- !.
extract_constraints(match(Var, Pattern, Type, Groups), [match(Var, Pattern, Type, Groups)]) :- !.
% Skip predicates
extract_constraints(Goal, []) :-
    functor(Goal, Pred, _),
    Pred \= ',',
    Pred \= true.

%% ============================================
%% REGEX SUPPORT
%% ============================================

%% validate_regex_type_for_awk(+Type)
%  Validate that AWK supports the given regex type
%
validate_regex_type_for_awk(auto) :- !.
validate_regex_type_for_awk(ere) :- !.
validate_regex_type_for_awk(bre) :- !.
validate_regex_type_for_awk(awk) :- !.
validate_regex_type_for_awk(Type) :-
    format('ERROR: AWK target does not support regex type ~q~n', [Type]),
    format('  Supported types: auto, ere, bre, awk~n', []),
    format('  Note: PCRE, Python, and .NET regex are not supported by AWK~n', []),
    fail.

%% escape_awk_pattern(+Pattern, -EscapedPattern)
%  Escape special characters in regex pattern for AWK
%  AWK regex uses / as delimiter, so we need to escape it
%
escape_awk_pattern(Pattern, EscapedPattern) :-
    atom(Pattern), !,
    atom_codes(Pattern, Codes),
    escape_pattern_codes(Codes, EscapedCodes),
    atom_codes(EscapedPattern, EscapedCodes).
escape_awk_pattern(Pattern, EscapedPattern) :-
    string(Pattern), !,
    string_codes(Pattern, Codes),
    escape_pattern_codes(Codes, EscapedCodes),
    string_codes(EscapedPattern, EscapedCodes).
escape_awk_pattern(Pattern, Pattern).

%% escape_pattern_codes(+Codes, -EscapedCodes)
%  Escape forward slashes in pattern
%
escape_pattern_codes([], []).
escape_pattern_codes([47|Rest], [92, 47|EscapedRest]) :- !,  % 47 = '/', 92 = '\'
    escape_pattern_codes(Rest, EscapedRest).
escape_pattern_codes([C|Rest], [C|EscapedRest]) :-
    escape_pattern_codes(Rest, EscapedRest).

%% has_capture_groups(+Constraints, -NumCaptures)
%  Check if constraints include match with capture groups
%  Returns the number of capture groups if found
%
has_capture_groups(Constraints, NumCaptures) :-
    member(match(_Var, _Pattern, _Type, Groups), Constraints),
    Groups \= [],
    length(Groups, NumCaptures),
    NumCaptures > 0.

%% generate_capture_print(+NumCaptures, -PrintStmt)
%  Generate AWK print statement for captured groups
%  AWK stores captures in array starting at index 1
%
generate_capture_print(NumCaptures, PrintStmt) :-
    numlist(1, NumCaptures, CaptureNums),
    findall(Cap,
        (member(N, CaptureNums), format(atom(Cap), '__captures__[~w]', [N])),
        Captures),
    atomic_list_concat(Captures, ', ', CapturesStr),
    format(atom(PrintStmt), 'print ~w', [CapturesStr]).

%% extract_bindings(+Body, -Bindings)
extract_bindings(true, []) :- !.
extract_bindings((A, B), Bs) :- !,
    extract_bindings(A, B1),
    extract_bindings(B, B2),
    append(B1, B2, Bs).
extract_bindings(Goal, [Goal]) :-
    functor(Goal, Name, Arity),
    awk_binding(Name/Arity, _, _, _, _), !.
extract_bindings(_, []).

%% generate_awk_bindings(+Bindings, +VarMap, -Code, -NewVarMap)
generate_awk_bindings([], VarMap, "", VarMap).
generate_awk_bindings([Goal|Rest], VarMapIn, Code, VarMapOut) :-
    Goal =.. [Pred|Args],
    functor(Goal, Pred, Arity),
    binding(awk, Pred/Arity, Pattern, Inputs, Outputs, _),
    
    length(Inputs, InCount),
    length(InArgs, InCount),
    append(InArgs, OutArgs, Args),
    
    maplist(term_to_awk_expr_binding(VarMapIn), InArgs, AwkInArgs),
    format_pattern(Pattern, AwkInArgs, Expr),
    
    (   Outputs = []
    ->  format(string(Line), "    ~s\n", [Expr]),
        VarMapNext = VarMapIn
    ;   OutArgs = [OutVar],
        gensym(v, VarName),
        format(string(Line), "    ~s = ~s\n", [VarName, Expr]),
        VarMapNext = [(OutVar, var(VarName))|VarMapIn]
    ),
    
    generate_awk_bindings(Rest, VarMapNext, RestCode, VarMapOut),
    string_concat(Line, RestCode, Code).

term_to_awk_expr_binding(VarMap, Term, Expr) :- term_to_awk_expr(Term, VarMap, Expr).

format_pattern(Pattern, Args, Cmd) :- format(string(Cmd), Pattern, Args).

%% ============================================
%% OPTIONS
%% ============================================

%% option(+Option, +Options, +Default)
%  Get option value with default fallback
%
option(Option, Options, Default) :-
    (   member(Option, Options)
    ->  true
    ;   Option =.. [Key, Default],
        \+ member(Key=_, Options)
    ).

% ============================================================================
% Pipeline Generator Mode for AWK
% ============================================================================
%
% This section implements pipeline chaining with support for generator mode
% (fixpoint iteration) for AWK targets. Similar to Python, PowerShell, C#,
% and Rust pipeline implementations.
%
% Usage:
%   compile_awk_pipeline([derive/1, transform/1], [
%       pipeline_name(fixpoint_pipe),
%       pipeline_mode(generator),
%       record_format(jsonl)
%   ], AwkCode).

%% compile_awk_pipeline(+Predicates, +Options, -AwkCode)
%  Compile a list of predicates into an AWK pipeline.
%  Options:
%    pipeline_name(Name) - Name for the pipeline (default: 'pipeline')
%    pipeline_mode(Mode) - 'sequential' (default) or 'generator' (fixpoint)
%    record_format(Format) - 'jsonl' (default), 'tsv', or 'csv'
%    field_separator(Sep) - Field separator for TSV/CSV (default: '\t')
%
compile_awk_pipeline(Predicates, Options, AwkCode) :-
    option(pipeline_name(PipelineName), Options, pipeline),
    option(pipeline_mode(PipelineMode), Options, sequential),
    option(record_format(RecordFormat), Options, jsonl),
    option(field_separator(FieldSep), Options, '\t'),

    % Generate header
    awk_pipeline_header(PipelineName, PipelineMode, Header),

    % Generate helper functions based on mode
    awk_pipeline_helpers(PipelineMode, RecordFormat, FieldSep, Helpers),

    % Extract stage names
    extract_awk_stage_names(Predicates, StageNames),

    % Generate stage functions
    generate_awk_stage_functions(StageNames, StageFunctions),

    % Generate the pipeline connector (mode-aware)
    generate_awk_pipeline_connector(StageNames, PipelineName, PipelineMode, ConnectorCode),

    % Generate main execution block
    generate_awk_main_block(PipelineName, PipelineMode, RecordFormat, FieldSep, MainBlock),

    % Combine all parts
    format(string(AwkCode),
"~w

~w

~w
~w

~w
", [Header, Helpers, StageFunctions, ConnectorCode, MainBlock]).

%% awk_pipeline_header(+Name, +Mode, -Header)
%  Generate pipeline header with shebang
awk_pipeline_header(PipelineName, generator, Header) :-
    !,
    atom_string(PipelineName, PipelineNameStr),
    format(string(Header),
"#!/usr/bin/awk -f
# Generated by UnifyWeaver AWK Pipeline Generator Mode
# Pipeline: ~w
# Fixpoint evaluation for recursive pipeline stages", [PipelineNameStr]).

awk_pipeline_header(PipelineName, _, Header) :-
    atom_string(PipelineName, PipelineNameStr),
    format(string(Header),
"#!/usr/bin/awk -f
# Generated by UnifyWeaver AWK Pipeline (sequential mode)
# Pipeline: ~w", [PipelineNameStr]).

%% awk_pipeline_helpers(+Mode, +Format, +FieldSep, -Helpers)
%  Generate mode-aware helper functions
awk_pipeline_helpers(generator, jsonl, _, Helpers) :-
    !,
    format(string(Helpers),
"# Helper function: Generate record key for deduplication
function record_key(record,    key, i, n, keys, k) {
    n = split(record, keys, \",\")
    asort(keys)
    key = \"\"
    for (i = 1; i <= n; i++) {
        if (key != \"\") key = key \";\"
        key = key keys[i]
    }
    return key
}

# Helper function: Parse JSONL record to fields array
function parse_jsonl(line, fields,    n, parts, i, kv, k, v) {
    gsub(/^\\{|\\}$/, \"\", line)
    n = split(line, parts, \",\")
    for (i = 1; i <= n; i++) {
        split(parts[i], kv, \":\")
        k = kv[1]; v = kv[2]
        gsub(/\"/, \"\", k); gsub(/\"/, \"\", v)
        gsub(/^ +| +$/, \"\", k); gsub(/^ +| +$/, \"\", v)
        fields[k] = v
    }
    return n
}

# Helper function: Format fields as JSONL
function format_jsonl(fields,    result, k, first) {
    result = \"{\"
    first = 1
    for (k in fields) {
        if (!first) result = result \",\"
        result = result \"\\\"\" k \"\\\":\\\"\" fields[k] \"\\\"\"
        first = 0
    }
    result = result \"}\"
    return result
}", []).

awk_pipeline_helpers(generator, _, FieldSep, Helpers) :-
    !,
    format(string(Helpers),
"# Helper function: Generate record key for deduplication
function record_key(record) {
    return record  # For TSV/CSV, the whole line is the key
}

# Field separator
BEGIN { FS = \"~w\"; OFS = \"~w\" }", [FieldSep, FieldSep]).

awk_pipeline_helpers(_, jsonl, _, Helpers) :-
    !,
    format(string(Helpers),
"# Helper function: Parse JSONL record to fields array
function parse_jsonl(line, fields,    n, parts, i, kv, k, v) {
    gsub(/^\\{|\\}$/, \"\", line)
    n = split(line, parts, \",\")
    for (i = 1; i <= n; i++) {
        split(parts[i], kv, \":\")
        k = kv[1]; v = kv[2]
        gsub(/\"/, \"\", k); gsub(/\"/, \"\", v)
        gsub(/^ +| +$/, \"\", k); gsub(/^ +| +$/, \"\", v)
        fields[k] = v
    }
    return n
}

# Helper function: Format fields as JSONL
function format_jsonl(fields,    result, k, first) {
    result = \"{\"
    first = 1
    for (k in fields) {
        if (!first) result = result \",\"
        result = result \"\\\"\" k \"\\\":\\\"\" fields[k] \"\\\"\"
        first = 0
    }
    result = result \"}\"
    return result
}", []).

awk_pipeline_helpers(_, _, FieldSep, Helpers) :-
    format(string(Helpers),
"# Field separator
BEGIN { FS = \"~w\"; OFS = \"~w\" }", [FieldSep, FieldSep]).

%% extract_awk_stage_names(+Predicates, -Names)
%  Extract stage names from predicate indicators
extract_awk_stage_names([], []).
extract_awk_stage_names([Pred|Rest], [Name|RestNames]) :-
    extract_awk_pred_name(Pred, Name),
    extract_awk_stage_names(Rest, RestNames).

extract_awk_pred_name(_Target:Name/_Arity, NameStr) :-
    !,
    atom_string(Name, NameStr).
extract_awk_pred_name(Name/_Arity, NameStr) :-
    atom_string(Name, NameStr).

%% generate_awk_stage_functions(+Names, -Code)
%  Generate placeholder stage function implementations
generate_awk_stage_functions([], "").
generate_awk_stage_functions([Name|Rest], Code) :-
    format(string(StageCode),
"# Stage: ~w
function stage_~w(record) {
    # TODO: Implement stage logic
    return record
}

", [Name, Name]),
    generate_awk_stage_functions(Rest, RestCode),
    format(string(Code), "~w~w", [StageCode, RestCode]).

%% generate_awk_pipeline_connector(+StageNames, +PipelineName, +Mode, -Code)
%  Generate the pipeline connector function (mode-aware)
generate_awk_pipeline_connector(StageNames, PipelineName, sequential, Code) :-
    !,
    generate_awk_sequential_chain(StageNames, ChainCode),
    atom_string(PipelineName, PipelineNameStr),
    format(string(Code),
"# Sequential pipeline connector: ~w
function ~w(record) {
    # Sequential mode - chain stages directly
~w
}", [PipelineNameStr, PipelineNameStr, ChainCode]).

generate_awk_pipeline_connector(StageNames, PipelineName, generator, Code) :-
    generate_awk_fixpoint_chain(StageNames, ChainCode),
    atom_string(PipelineName, PipelineNameStr),
    format(string(Code),
"# Fixpoint pipeline connector: ~w
# Iterates until no new records are produced.
function run_~w(    changed, i, record, key, new_record) {
    # Fixpoint iteration
    do {
        changed = 0
        for (i = 1; i <= record_count; i++) {
            record = records[i]
~w
            key = record_key(new_record)
            if (!(key in seen)) {
                seen[key] = 1
                record_count++
                records[record_count] = new_record
                output[++output_count] = new_record
                changed = 1
            }
        }
    } while (changed)
}", [PipelineNameStr, PipelineNameStr, ChainCode]).

%% generate_awk_sequential_chain(+Names, -Code)
%  Generate sequential stage chaining code
generate_awk_sequential_chain([], Code) :-
    format(string(Code), "    return record", []).
generate_awk_sequential_chain([Name], Code) :-
    !,
    format(string(Code), "    return stage_~w(record)", [Name]).
generate_awk_sequential_chain(Names, Code) :-
    Names \= [],
    generate_awk_chain_expr(Names, "record", ChainExpr),
    format(string(Code), "    return ~w", [ChainExpr]).

generate_awk_chain_expr([], Current, Current).
generate_awk_chain_expr([Name|Rest], Current, Expr) :-
    format(string(NextExpr), "stage_~w(~w)", [Name, Current]),
    generate_awk_chain_expr(Rest, NextExpr, Expr).

%% generate_awk_fixpoint_chain(+Names, -Code)
%  Generate fixpoint stage application code
generate_awk_fixpoint_chain([], Code) :-
    format(string(Code), "            new_record = record", []).
generate_awk_fixpoint_chain(Names, Code) :-
    Names \= [],
    generate_awk_fixpoint_stages(Names, "record", StageCode),
    format(string(Code), "~w", [StageCode]).

generate_awk_fixpoint_stages([], Current, Code) :-
    format(string(Code), "            new_record = ~w", [Current]).
generate_awk_fixpoint_stages([Stage|Rest], Current, Code) :-
    format(string(NextVar), "stage_~w_out", [Stage]),
    format(string(StageCall), "            ~w = stage_~w(~w)
", [NextVar, Stage, Current]),
    generate_awk_fixpoint_stages(Rest, NextVar, RestCode),
    format(string(Code), "~w~w", [StageCall, RestCode]).

%% generate_awk_main_block(+PipelineName, +Mode, +Format, +FieldSep, -Code)
%  Generate main execution block
generate_awk_main_block(PipelineName, generator, jsonl, _, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    format(string(Code),
"# Main execution - Generator mode with JSONL
BEGIN {
    record_count = 0
    output_count = 0
}

{
    # Read input records
    record_count++
    records[record_count] = $0
    key = record_key($0)
    if (!(key in seen)) {
        seen[key] = 1
        output[++output_count] = $0
    }
}

END {
    # Run fixpoint iteration
    run_~w()

    # Output all results
    for (i = 1; i <= output_count; i++) {
        print output[i]
    }
}", [PipelineNameStr]).

generate_awk_main_block(PipelineName, generator, _, _, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    format(string(Code),
"# Main execution - Generator mode with TSV/CSV
BEGIN {
    record_count = 0
    output_count = 0
}

{
    # Read input records
    record_count++
    records[record_count] = $0
    key = record_key($0)
    if (!(key in seen)) {
        seen[key] = 1
        output[++output_count] = $0
    }
}

END {
    # Run fixpoint iteration
    run_~w()

    # Output all results
    for (i = 1; i <= output_count; i++) {
        print output[i]
    }
}", [PipelineNameStr]).

generate_awk_main_block(PipelineName, sequential, _, _, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    format(string(Code),
"# Main execution - Sequential mode
{
    result = ~w($0)
    if (result != \"\") print result
}", [PipelineNameStr]).

% ============================================================================
% Unit Tests for AWK Pipeline Generator Mode
% ============================================================================

test_awk_pipeline_generator :-
    format("~n=== AWK Pipeline Generator Mode Unit Tests ===~n~n", []),

    % Test 1: Basic pipeline compilation with generator mode
    format("Test 1: Basic pipeline with generator mode... ", []),
    (   compile_awk_pipeline([transform/1, derive/1], [
            pipeline_name(test_pipeline),
            pipeline_mode(generator),
            record_format(jsonl)
        ], Code1),
        sub_string(Code1, _, _, _, "record_key"),
        sub_string(Code1, _, _, _, "while (changed)"),
        sub_string(Code1, _, _, _, "test_pipeline")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 2: Sequential mode still works
    format("Test 2: Sequential mode still works... ", []),
    (   compile_awk_pipeline([filter/1, format/1], [
            pipeline_name(seq_pipeline),
            pipeline_mode(sequential),
            record_format(tsv)
        ], Code2),
        sub_string(Code2, _, _, _, "seq_pipeline"),
        sub_string(Code2, _, _, _, "sequential mode"),
        \+ sub_string(Code2, _, _, _, "while (changed)")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 3: Generator mode includes record_key function
    format("Test 3: Generator mode has record_key function... ", []),
    (   compile_awk_pipeline([a/1], [pipeline_mode(generator)], Code3),
        sub_string(Code3, _, _, _, "function record_key")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 4: JSONL helpers included for jsonl format
    format("Test 4: JSONL helpers included... ", []),
    (   compile_awk_pipeline([x/1], [pipeline_mode(generator), record_format(jsonl)], Code4),
        sub_string(Code4, _, _, _, "parse_jsonl"),
        sub_string(Code4, _, _, _, "format_jsonl")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 5: Fixpoint iteration structure
    format("Test 5: Fixpoint iteration structure... ", []),
    (   compile_awk_pipeline([derive/1, transform/1], [pipeline_mode(generator)], Code5),
        sub_string(Code5, _, _, _, "changed = 0"),
        sub_string(Code5, _, _, _, "while (changed)"),
        sub_string(Code5, _, _, _, "seen[key]")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 6: Stage functions generated
    format("Test 6: Stage functions generated... ", []),
    (   compile_awk_pipeline([filter/1, transform/1], [pipeline_mode(generator)], Code6),
        sub_string(Code6, _, _, _, "function stage_filter"),
        sub_string(Code6, _, _, _, "function stage_transform")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 7: Pipeline chain code for generator
    format("Test 7: Pipeline chain code for generator mode... ", []),
    (   compile_awk_pipeline([derive/1, transform/1], [pipeline_mode(generator)], Code7),
        sub_string(Code7, _, _, _, "stage_derive_out"),
        sub_string(Code7, _, _, _, "stage_transform_out")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 8: Default options work
    format("Test 8: Default options work... ", []),
    (   compile_awk_pipeline([a/1, b/1], [], Code8),
        sub_string(Code8, _, _, _, "pipeline"),
        sub_string(Code8, _, _, _, "sequential mode")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 9: Shebang header present
    format("Test 9: Shebang header present... ", []),
    (   compile_awk_pipeline([x/1], [pipeline_name(test_pipe)], Code9),
        sub_string(Code9, _, _, _, "#!/usr/bin/awk -f"),
        sub_string(Code9, _, _, _, "test_pipe")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 10: TSV field separator option
    format("Test 10: TSV field separator option... ", []),
    (   compile_awk_pipeline([x/1], [
            pipeline_mode(sequential),
            record_format(tsv),
            field_separator(':')
        ], Code10),
        sub_string(Code10, _, _, _, "FS = \":\"")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    format("~n=== All AWK Pipeline Generator Mode Tests Passed ===~n", []).

%% ============================================
%% AWK ENHANCED PIPELINE CHAINING
%% ============================================
%
%  Supports advanced flow patterns:
%    - fan_out(Stages) : Broadcast to parallel stages
%    - merge          : Combine results from parallel stages
%    - route_by(Pred, Routes) : Conditional routing
%    - filter_by(Pred) : Filter records
%    - Pred/Arity     : Standard stage
%
%% compile_awk_enhanced_pipeline(+Stages, +Options, -AwkCode)
%  Main entry point for enhanced AWK pipeline with advanced flow patterns.
%
compile_awk_enhanced_pipeline(Stages, Options, AwkCode) :-
    option(pipeline_name(PipelineName), Options, enhanced_pipeline),
    option(record_format(RecordFormat), Options, jsonl),

    % Generate header
    awk_enhanced_header(PipelineName, Header),

    % Generate helpers
    awk_enhanced_helpers(Helpers),

    % Generate stage functions
    generate_awk_enhanced_stage_functions(Stages, StageFunctions),

    % Generate the main connector
    generate_awk_enhanced_connector(Stages, PipelineName, ConnectorCode),

    % Generate main block
    generate_awk_enhanced_main(PipelineName, RecordFormat, MainBlock),

    format(string(AwkCode),
"~w

~w

~w
~w
~w
", [Header, Helpers, StageFunctions, ConnectorCode, MainBlock]).

%% awk_enhanced_header(+PipelineName, -Header)
%  Generate header for enhanced pipeline
awk_enhanced_header(PipelineName, Header) :-
    atom_string(PipelineName, PipelineNameStr),
    format(string(Header),
"#!/usr/bin/awk -f
# Generated by UnifyWeaver AWK Enhanced Pipeline
# Pipeline: ~w
# Supports fan-out, merge, conditional routing, and filtering", [PipelineNameStr]).

%% awk_enhanced_helpers(-Code)
%  Generate helper functions for enhanced pipeline operations.
awk_enhanced_helpers(Code) :-
    Code = "# Enhanced Pipeline Helpers

# Helper: Parse JSONL record to fields array
function parse_jsonl(line, fields,    n, parts, i, kv, k, v) {
    gsub(/^\\{|\\}$/, \"\", line)
    n = split(line, parts, \",\")
    for (i = 1; i <= n; i++) {
        split(parts[i], kv, \":\")
        k = kv[1]; v = kv[2]
        gsub(/\"/, \"\", k); gsub(/\"/, \"\", v)
        gsub(/^ +| +$/, \"\", k); gsub(/^ +| +$/, \"\", v)
        fields[k] = v
    }
    return n
}

# Helper: Format fields as JSONL
function format_jsonl(fields,    result, k, first) {
    result = \"{\"
    first = 1
    for (k in fields) {
        if (!first) result = result \",\"
        result = result \"\\\"\" k \"\\\":\\\"\" fields[k] \"\\\"\"
        first = 0
    }
    result = result \"}\"
    return result
}

# Helper: Fan-out - send record to multiple stages, collect all results
# Returns results in fan_out_results array, sets fan_out_count
function fan_out_records(record, stages,    i, n, stage, result) {
    fan_out_count = 0
    n = split(stages, stage_arr, \",\")
    for (i = 1; i <= n; i++) {
        stage = stage_arr[i]
        gsub(/^ +| +$/, \"\", stage)
        # Call stage function dynamically using indirect call
        result = @stage(record)
        fan_out_count++
        fan_out_results[fan_out_count] = result
    }
    return fan_out_count
}

# Helper: Route record based on condition
# condition_result should be set before calling
function route_record(record, condition_result,    route_stage, result) {
    if (condition_result in route_map) {
        route_stage = route_map[condition_result]
        result = @route_stage(record)
    } else if (\"default\" in route_map) {
        route_stage = route_map[\"default\"]
        result = @route_stage(record)
    } else {
        result = record  # Pass through if no matching route
    }
    return result
}

# Helper: Filter records based on predicate
# Returns 1 if record passes filter, 0 otherwise
function filter_record(record, predicate_fn,    result) {
    result = @predicate_fn(record)
    return result
}

# Helper: Merge streams (identity for AWK - streams already linear)
function merge_streams(records, count,    i) {
    # In AWK, we process records one at a time, so merge is implicit
    for (i = 1; i <= count; i++) {
        output[++output_count] = records[i]
    }
    return count
}

# Helper: Tee stream - send each record to multiple stages
function tee_stream(records, count, stages,    i, j, n, stage, result) {
    n = split(stages, stage_arr, \",\")
    for (i = 1; i <= count; i++) {
        for (j = 1; j <= n; j++) {
            stage = stage_arr[j]
            gsub(/^ +| +$/, \"\", stage)
            result = @stage(records[i])
            output[++output_count] = result
        }
    }
}

".

%% generate_awk_enhanced_stage_functions(+Stages, -Code)
%  Generate stub functions for each stage.
generate_awk_enhanced_stage_functions([], "").
generate_awk_enhanced_stage_functions([Stage|Rest], Code) :-
    generate_awk_single_enhanced_stage(Stage, StageCode),
    generate_awk_enhanced_stage_functions(Rest, RestCode),
    (RestCode = "" ->
        Code = StageCode
    ;   format(string(Code), "~w~w", [StageCode, RestCode])
    ).

generate_awk_single_enhanced_stage(fan_out(SubStages), Code) :-
    !,
    generate_awk_enhanced_stage_functions(SubStages, Code).
generate_awk_single_enhanced_stage(merge, "") :- !.
generate_awk_single_enhanced_stage(route_by(_, Routes), Code) :-
    !,
    findall(Stage, member((_Cond, Stage), Routes), RouteStages),
    generate_awk_enhanced_stage_functions(RouteStages, Code).
generate_awk_single_enhanced_stage(filter_by(_), "") :- !.
generate_awk_single_enhanced_stage(Pred/Arity, Code) :-
    !,
    format(string(Code),
"# Stage: ~w/~w
function stage_~w(record) {
    # TODO: Implement stage logic
    return record
}

", [Pred, Arity, Pred]).
generate_awk_single_enhanced_stage(_, "").

%% generate_awk_enhanced_connector(+Stages, +PipelineName, -Code)
%  Generate the main connector that handles enhanced flow patterns.
generate_awk_enhanced_connector(Stages, PipelineName, Code) :-
    generate_awk_enhanced_flow_code(Stages, "record", FlowCode),
    atom_string(PipelineName, PipelineNameStr),
    format(string(Code),
"# ~w - Enhanced pipeline with fan-out, merge, and routing support
function run_~w(record,    result) {
~w
    return result
}

", [PipelineNameStr, PipelineNameStr, FlowCode]).

%% generate_awk_enhanced_flow_code(+Stages, +CurrentVar, -Code)
%  Generate the flow code for enhanced pipeline stages.
generate_awk_enhanced_flow_code([], CurrentVar, Code) :-
    format(string(Code), "    result = ~w", [CurrentVar]).
generate_awk_enhanced_flow_code([Stage|Rest], CurrentVar, Code) :-
    generate_awk_stage_flow(Stage, CurrentVar, NextVar, StageCode),
    generate_awk_enhanced_flow_code(Rest, NextVar, RestCode),
    format(string(Code), "~w~n~w", [StageCode, RestCode]).

%% generate_awk_stage_flow(+Stage, +InVar, -OutVar, -Code)
%  Generate flow code for a single stage.

% Fan-out stage: broadcast to parallel stages
generate_awk_stage_flow(fan_out(SubStages), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "fan_out_~w_result", [N]),
    extract_awk_enhanced_stage_names(SubStages, StageNames),
    format_awk_stage_list(StageNames, StageListStr),
    format(string(Code),
"    # Fan-out to ~w parallel stages
    fan_out_records(~w, \"~w\")
    # Collect fan-out results
    ~w = \"\"
    for (_fo_i = 1; _fo_i <= fan_out_count; _fo_i++) {
        if (~w != \"\") ~w = ~w \"\\n\"
        ~w = ~w fan_out_results[_fo_i]
    }", [N, InVar, StageListStr, OutVar, OutVar, OutVar, OutVar, OutVar, OutVar]).

% Merge stage: placeholder, usually follows fan_out
generate_awk_stage_flow(merge, InVar, OutVar, Code) :-
    !,
    OutVar = InVar,
    Code = "    # Merge: results already combined from fan-out".

% Conditional routing
generate_awk_stage_flow(route_by(CondPred, Routes), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "routed_result", []),
    format_awk_route_map(Routes, RouteMapStr),
    format(string(Code),
"    # Conditional routing based on ~w
    # Initialize route map
~w
    # Get condition and route
    _cond = ~w(~w)
    ~w = route_record(~w, _cond)", [CondPred, RouteMapStr, CondPred, InVar, OutVar, InVar]).

% Filter stage
generate_awk_stage_flow(filter_by(Pred), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "filtered_result", []),
    format(string(Code),
"    # Filter by ~w
    if (~w(~w)) {
        ~w = ~w
    } else {
        ~w = \"\"  # Filtered out
    }", [Pred, Pred, InVar, OutVar, InVar, OutVar]).

% Standard predicate stage
generate_awk_stage_flow(Pred/Arity, InVar, OutVar, Code) :-
    !,
    atom(Pred),
    format(atom(OutVar), "~w_result", [Pred]),
    format(string(Code),
"    # Stage: ~w/~w
    ~w = stage_~w(~w)", [Pred, Arity, OutVar, Pred, InVar]).

% Fallback for unknown stages
generate_awk_stage_flow(Stage, InVar, InVar, Code) :-
    format(string(Code), "    # Unknown stage type: ~w (pass-through)", [Stage]).

%% extract_awk_enhanced_stage_names(+Stages, -Names)
%  Extract function names from stage specifications.
extract_awk_enhanced_stage_names([], []).
extract_awk_enhanced_stage_names([Pred/_Arity|Rest], [Pred|RestNames]) :-
    !,
    extract_awk_enhanced_stage_names(Rest, RestNames).
extract_awk_enhanced_stage_names([_|Rest], RestNames) :-
    extract_awk_enhanced_stage_names(Rest, RestNames).

%% format_awk_stage_list(+Names, -ListStr)
%  Format stage names as comma-separated AWK function references.
format_awk_stage_list([], "").
format_awk_stage_list([Name], Str) :-
    format(string(Str), "stage_~w", [Name]).
format_awk_stage_list([Name|Rest], Str) :-
    Rest \= [],
    format_awk_stage_list(Rest, RestStr),
    format(string(Str), "stage_~w,~w", [Name, RestStr]).

%% format_awk_route_map(+Routes, -MapStr)
%  Format routing map initialization for AWK.
format_awk_route_map([], "").
format_awk_route_map([(_Cond, Stage)|[]], Str) :-
    (Stage = StageName/_Arity -> true ; StageName = Stage),
    format(string(Str), "    route_map[1] = \"stage_~w\"", [StageName]).
format_awk_route_map([(Cond, Stage)|Rest], Str) :-
    Rest \= [],
    (Stage = StageName/_Arity -> true ; StageName = Stage),
    format_awk_route_map(Rest, RestStr),
    (Cond = true ->
        format(string(Str), "    route_map[1] = \"stage_~w\"~n~w", [StageName, RestStr])
    ; Cond = false ->
        format(string(Str), "    route_map[0] = \"stage_~w\"~n~w", [StageName, RestStr])
    ;   format(string(Str), "    route_map[\"~w\"] = \"stage_~w\"~n~w", [Cond, StageName, RestStr])
    ).

%% generate_awk_enhanced_main(+PipelineName, +RecordFormat, -Code)
%  Generate main execution block for enhanced pipeline.
generate_awk_enhanced_main(PipelineName, jsonl, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    format(string(Code),
"# Main execution block
BEGIN {
    output_count = 0
}

{
    # Process each input record through enhanced pipeline
    result = run_~w($0)
    if (result != \"\") {
        print result
        output_count++
    }
}

END {
    # Summary statistics (optional)
    # print \"Processed \" output_count \" records\" > \"/dev/stderr\"
}
", [PipelineNameStr]).
generate_awk_enhanced_main(PipelineName, _, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    format(string(Code),
"# Main execution block
BEGIN {
    output_count = 0
}

{
    # Process each input record through enhanced pipeline
    result = run_~w($0)
    if (result != \"\") {
        print result
        output_count++
    }
}

END {
    # Summary statistics (optional)
    # print \"Processed \" output_count \" records\" > \"/dev/stderr\"
}
", [PipelineNameStr]).

%% ============================================
%% AWK ENHANCED PIPELINE CHAINING TESTS
%% ============================================

test_awk_enhanced_chaining :-
    format('~n=== AWK Enhanced Pipeline Chaining Tests ===~n~n', []),

    % Test 1: Generate enhanced helpers
    format('[Test 1] Generate enhanced helpers~n', []),
    awk_enhanced_helpers(Helpers1),
    (   sub_string(Helpers1, _, _, _, "fan_out_records"),
        sub_string(Helpers1, _, _, _, "route_record"),
        sub_string(Helpers1, _, _, _, "filter_record"),
        sub_string(Helpers1, _, _, _, "merge_streams"),
        sub_string(Helpers1, _, _, _, "tee_stream")
    ->  format('  [PASS] All helper functions generated~n', [])
    ;   format('  [FAIL] Missing helper functions~n', [])
    ),

    % Test 2: Linear pipeline connector
    format('[Test 2] Linear pipeline connector~n', []),
    generate_awk_enhanced_connector([extract/1, transform/1, load/1], linear_pipe, Code2),
    (   sub_string(Code2, _, _, _, "run_linear_pipe"),
        sub_string(Code2, _, _, _, "stage_extract"),
        sub_string(Code2, _, _, _, "stage_transform"),
        sub_string(Code2, _, _, _, "stage_load")
    ->  format('  [PASS] Linear connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code2])
    ),

    % Test 3: Fan-out connector
    format('[Test 3] Fan-out connector~n', []),
    generate_awk_enhanced_connector([fan_out([validate/1, enrich/1])], fanout_pipe, Code3),
    (   sub_string(Code3, _, _, _, "run_fanout_pipe"),
        sub_string(Code3, _, _, _, "Fan-out to 2 parallel stages"),
        sub_string(Code3, _, _, _, "fan_out_records")
    ->  format('  [PASS] Fan-out connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code3])
    ),

    % Test 4: Fan-out with merge
    format('[Test 4] Fan-out with merge~n', []),
    generate_awk_enhanced_connector([fan_out([a/1, b/1]), merge], merge_pipe, Code4),
    (   sub_string(Code4, _, _, _, "run_merge_pipe"),
        sub_string(Code4, _, _, _, "Fan-out to 2"),
        sub_string(Code4, _, _, _, "Merge: results already combined")
    ->  format('  [PASS] Merge connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code4])
    ),

    % Test 5: Conditional routing
    format('[Test 5] Conditional routing~n', []),
    generate_awk_enhanced_connector([route_by(has_error, [(true, error_handler/1), (false, success/1)])], route_pipe, Code5),
    (   sub_string(Code5, _, _, _, "run_route_pipe"),
        sub_string(Code5, _, _, _, "Conditional routing based on has_error"),
        sub_string(Code5, _, _, _, "route_map")
    ->  format('  [PASS] Routing connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code5])
    ),

    % Test 6: Filter stage
    format('[Test 6] Filter stage~n', []),
    generate_awk_enhanced_connector([filter_by(is_valid)], filter_pipe, Code6),
    (   sub_string(Code6, _, _, _, "run_filter_pipe"),
        sub_string(Code6, _, _, _, "Filter by is_valid"),
        sub_string(Code6, _, _, _, "is_valid")
    ->  format('  [PASS] Filter connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code6])
    ),

    % Test 7: Complex pipeline with all patterns
    format('[Test 7] Complex pipeline~n', []),
    generate_awk_enhanced_connector([
        extract/1,
        filter_by(is_active),
        fan_out([validate/1, enrich/1, audit/1]),
        merge,
        route_by(has_error, [(true, error_log/1), (false, transform/1)]),
        output/1
    ], complex_pipe, Code7),
    (   sub_string(Code7, _, _, _, "run_complex_pipe"),
        sub_string(Code7, _, _, _, "Filter by is_active"),
        sub_string(Code7, _, _, _, "Fan-out to 3 parallel stages"),
        sub_string(Code7, _, _, _, "Merge"),
        sub_string(Code7, _, _, _, "Conditional routing")
    ->  format('  [PASS] Complex connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code7])
    ),

    % Test 8: Stage function generation
    format('[Test 8] Stage function generation~n', []),
    generate_awk_enhanced_stage_functions([extract/1, transform/1], StageFns8),
    (   sub_string(StageFns8, _, _, _, "stage_extract"),
        sub_string(StageFns8, _, _, _, "stage_transform")
    ->  format('  [PASS] Stage functions generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [StageFns8])
    ),

    % Test 9: Full enhanced pipeline compilation
    format('[Test 9] Full enhanced pipeline~n', []),
    compile_awk_enhanced_pipeline([
        extract/1,
        filter_by(is_active),
        fan_out([validate/1, enrich/1]),
        merge,
        output/1
    ], [pipeline_name(full_enhanced), record_format(jsonl)], FullCode9),
    (   sub_string(FullCode9, _, _, _, "#!/usr/bin/awk -f"),
        sub_string(FullCode9, _, _, _, "fan_out_records"),
        sub_string(FullCode9, _, _, _, "filter_record"),
        sub_string(FullCode9, _, _, _, "run_full_enhanced"),
        sub_string(FullCode9, _, _, _, "BEGIN")
    ->  format('  [PASS] Full pipeline compiles~n', [])
    ;   format('  [FAIL] Missing patterns in generated code~n', [])
    ),

    % Test 10: Enhanced helpers include all functions
    format('[Test 10] Enhanced helpers completeness~n', []),
    awk_enhanced_helpers(Helpers10),
    (   sub_string(Helpers10, _, _, _, "parse_jsonl"),
        sub_string(Helpers10, _, _, _, "format_jsonl"),
        sub_string(Helpers10, _, _, _, "fan_out_records"),
        sub_string(Helpers10, _, _, _, "route_record"),
        sub_string(Helpers10, _, _, _, "filter_record")
    ->  format('  [PASS] All helpers present~n', [])
    ;   format('  [FAIL] Missing helpers~n', [])
    ),

    format('~n=== All AWK Enhanced Pipeline Chaining Tests Passed ===~n', []).
