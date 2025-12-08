:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% powershell_compiler.pl - Compile Prolog predicates to PowerShell scripts
% Strategy: Generate bash code via existing compilers, wrap in PowerShell compatibility layer
% This allows reuse of all bash templates while providing PowerShell target support
%
% Future: Phase 2 will add native PowerShell code generation with object pipeline support

:- module(powershell_compiler, [
    compile_to_powershell/3,      % compile_to_powershell(+Predicate, +Options, -PowerShellCode)
    compile_to_powershell/2,      % compile_to_powershell(+Predicate, -PowerShellCode)
    powershell_wrapper/3,         % powershell_wrapper(+BashCode, +Options, -PowerShellCode)
    test_powershell_compiler/0    % Run tests
]).

:- use_module(library(lists)).
:- use_module('stream_compiler').
:- use_module('recursive_compiler').
:- use_module('firewall_v2').
% Load source modules without importing predicates (we call them with module qualification)
:- use_module('../sources/csv_source', []).
:- use_module('../sources/json_source', []).
:- use_module('../sources/http_source', []).
:- use_module('../sources/xml_source', []).
:- use_module('../sources/dotnet_source', []).

%% compile_to_powershell(+Predicate, -PowerShellCode)
%  Simplified interface with default options
compile_to_powershell(Predicate, PowerShellCode) :-
    compile_to_powershell(Predicate, [], PowerShellCode).

%% compile_to_powershell(+Predicate, +Options, -PowerShellCode)
%  Main compilation entry point
%
%  Predicate: Pred/Arity indicator (e.g., grandparent/2)
%  Options:
%    - compiler(stream|recursive) - which bash compiler to use (default: stream)
%    - wrapper_style(inline|tempfile) - how to invoke bash (default: inline)
%    - compat_check(true|false) - add compatibility layer check (default: true)
%    - output_file(Path) - write to file instead of returning code
%    - script_name(Name) - name for generated script (default: derived from predicate)
%    - powershell_mode(baas|pure|auto) - pure PowerShell or bash-as-a-service (default: auto)
%    ... other options passed to bash compiler
%
compile_to_powershell(Predicate, Options, PowerShellCode) :-
    Predicate = PredAtom/Arity,
    format('[PowerShell Compiler] Compiling ~w/~w to PowerShell~n', [PredAtom, Arity]),

    % Determine PowerShell compilation mode (consult firewall if auto)
    option(powershell_mode(UserMode), Options, auto),

    % Get source type for firewall consultation
    (   member(source_type(SourceType), Options)
    ->  true
    ;   SourceType = unknown
    ),

    % Resolve mode (may consult firewall)
    resolve_powershell_mode(UserMode, SourceType, ResolvedMode),
    format('[PowerShell Compiler] PowerShell mode: ~w (resolved from ~w)~n', [ResolvedMode, UserMode]),

    % Check if pure mode and predicate supports it
    (   (ResolvedMode = pure ; ResolvedMode = auto ; ResolvedMode = auto_with_preference(pure)),
        supports_pure_powershell(Predicate, Options)
    ->  % Use pure PowerShell template
        format('[PowerShell Compiler] Using pure PowerShell templates~n', []),
        compile_to_pure_powershell(Predicate, Options, PowerShellCode)
    ;   % Fall back to bash-as-a-service
        format('[PowerShell Compiler] Using bash-as-a-service (BaaS) mode~n', []),
        compile_to_baas_powershell(Predicate, Options, PowerShellCode)
    ),

    % Optionally write to file
    (   option(output_file(OutputFile), Options)
    ->  write_powershell_file(OutputFile, PowerShellCode),
        format('[PowerShell Compiler] Written to: ~w~n', [OutputFile])
    ;   true
    ).

%% resolve_powershell_mode(+UserMode, +SourceType, -ResolvedMode)
%  Resolve PowerShell mode based on user preference and firewall policies.
%
%  UserMode: pure | baas | auto (from user options)
%  SourceType: csv | json | http | awk | python | unknown
%  ResolvedMode: pure | baas | auto | auto_with_preference(Mode)
resolve_powershell_mode(UserMode, SourceType, ResolvedMode) :-
    (   UserMode = auto
    ->  % Auto mode: consult firewall
        (   catch(firewall_v2:derive_powershell_mode(SourceType, FirewallMode), _, fail)
        ->  ResolvedMode = FirewallMode,
            format('[Firewall] Derived mode: ~w for source type: ~w~n', [FirewallMode, SourceType])
        ;   % Firewall not available or failed, use default auto
            ResolvedMode = auto
        )
    ;   % User explicitly specified mode, respect it
        ResolvedMode = UserMode
    ).

%% supports_pure_powershell(+Predicate, +Options)
%  Check if predicate can be compiled to pure PowerShell
supports_pure_powershell(_Predicate, Options) :-
    % Check if it's a dynamic source with pure PowerShell support
    (   member(source_type(csv), Options) -> true
    ;   member(source_type(json), Options) -> true
    ;   member(source_type(http), Options) -> true
    ;   member(source_type(xml), Options) -> true
    ;   member(source_type(dotnet), Options) -> true
    ;   % NEW: Support basic Prolog facts/joins in pure PowerShell
        \+ member(source_type(_), Options)  % No special source = basic Prolog
    ).

%% compile_to_pure_powershell(+Predicate, +Options, -PowerShellCode)
%  Compile using pure PowerShell templates (no bash dependency)
compile_to_pure_powershell(Predicate, Options, PowerShellCode) :-
    % Add template_suffix to request _powershell_pure templates
    append(Options, [template_suffix('_powershell_pure')], PureOptions),

    % Dispatch to appropriate source compiler based on source_type
    (   member(source_type(csv), PureOptions)
    ->  csv_source:compile_source(Predicate, PureOptions, [], PowerShellCode)
    ;   member(source_type(json), PureOptions)
    ->  json_source:compile_source(Predicate, PureOptions, [], PowerShellCode)
    ;   member(source_type(http), PureOptions)
    ->  http_source:compile_source(Predicate, PureOptions, [], PowerShellCode)
    ;   member(source_type(xml), PureOptions)
    ->  xml_source:compile_source(Predicate, PureOptions, [], PowerShellCode)
    ;   member(source_type(dotnet), PureOptions)
    ->  dotnet_source:compile_source(Predicate, PureOptions, [], PowerShellCode)
    ;   % NEW: Compile basic Prolog logic (facts, joins) to pure PowerShell
        compile_prolog_to_pure_powershell(Predicate, Options, PowerShellCode)
    ).

%% compile_prolog_to_pure_powershell(+Predicate, +Options, -PowerShellCode)
%  Compile Prolog facts and rules to pure PowerShell (no bash dependency)
compile_prolog_to_pure_powershell(Predicate, Options, PowerShellCode) :-
    Predicate = Pred/Arity,
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    length(Clauses, NumClauses),
    format('[PowerShell Pure] Compiling ~w/~w (~w clauses)~n', [Pred, Arity, NumClauses]),
    
    % Check for generator mode (fixpoint)
    (   member(mode(generator), Options)
    ->  format('[PowerShell Pure] Mode: generator (fixpoint)~n', []),
        compile_generator_mode_powershell(Pred, Arity, Clauses, Options, PowerShellCode)
    ;   % Non-generator mode - determine type
        % Check if predicate is recursive
        (   is_recursive_predicate_ps(Pred, Clauses)
        ->  format('[PowerShell Pure] Type: recursive~n', []),
            compile_recursive_to_powershell(Pred, Arity, Clauses, Options, PowerShellCode)
        ;   forall(member(_-Body, Clauses), Body = true)
        ->  % All facts
            format('[PowerShell Pure] Type: facts~n', []),
            compile_facts_to_powershell(Pred, Arity, Clauses, Options, PowerShellCode)
        ;   % Has rules - compile first rule
            member(H-Body, Clauses),
            Body \= true
        ->  format('[PowerShell Pure] Type: rule with body~n', []),
            compile_rule_to_powershell(Pred, Arity, H, Body, Options, PowerShellCode)
        ;   format('[PowerShell Pure] Error: No clauses found~n', []),
            fail
        )
    ).

%% is_recursive_predicate_ps(+Pred, +Clauses)
%  Check if any clause body contains a call to the same predicate
is_recursive_predicate_ps(Pred, Clauses) :-
    member(_Head-Body, Clauses),
    Body \= true,
    body_contains_pred_ps(Pred, Body),
    !.

%% body_contains_pred_ps(+Pred, +Body)
%  Check if body contains a call to Pred
body_contains_pred_ps(Pred, (A, B)) :- !,
    (   body_contains_pred_ps(Pred, A)
    ;   body_contains_pred_ps(Pred, B)
    ).
body_contains_pred_ps(Pred, \+ Goal) :- !,
    body_contains_pred_ps(Pred, Goal).
body_contains_pred_ps(Pred, Goal) :-
    Goal \= true,
    Goal =.. [Pred|_].

%% compile_facts_to_powershell(+Pred, +Arity, +Clauses, +Options, -Code)
%  Compile simple facts to pure PowerShell
compile_facts_to_powershell(Pred, Arity, Clauses, _Options, Code) :-
    atom_string(Pred, PredStr),
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),
    
    % Build fact array entries
    findall(FactEntry, (
        member(Head-true, Clauses),
        Head =.. [_|Args],
        format_ps_fact_entry(Args, FactEntry)
    ), FactEntries),
    atomic_list_concat(FactEntries, '\n', FactsStr),
    
    % Generate parameter string based on arity
    (   Arity = 1
    ->  ParamStr = "[string]$Key"
    ;   Arity = 2
    ->  ParamStr = "[string]$X, [string]$Y"
    ;   ParamStr = "[string]$X, [string]$Y, [string]$Z"  % Generic for now
    ),
    
    % Generate filter logic based on arity
    (   Arity = 1
    ->  FilterLogic = "
    if ($Key) {
        $facts | Where-Object { $_ -eq $Key }
    } else {
        $facts
    }"
    ;   Arity = 2
    ->  FilterLogic = "
    if ($X -and $Y) {
        $facts | Where-Object { $_.X -eq $X -and $_.Y -eq $Y }
    } elseif ($X) {
        $facts | Where-Object { $_.X -eq $X } | ForEach-Object { $_.Y }
    } elseif ($Y) {
        $facts | Where-Object { $_.Y -eq $Y } | ForEach-Object { $_.X }
    } else {
        $facts | ForEach-Object { \"$($_.X):$($_.Y)\" }
    }"
    ;   FilterLogic = "
    $facts | ForEach-Object { $_ }"  % Generic fallback
    ),
    
    % Generate appropriate fact array based on arity
    (   Arity = 1
    ->  format(string(Code),
"# Generated Pure PowerShell Script
# Predicate: ~w/~w
# Generated: ~w
# Generated by UnifyWeaver PowerShell Compiler (Pure Mode)

function ~w {
    param(~w)
    
    $facts = @(~w)
~w
}

# Stream all facts
~w
", [Pred, Arity, DateStr, PredStr, ParamStr, FactsStr, FilterLogic, PredStr])
    ;   % Binary facts use PSCustomObject
        findall(ObjEntry, (
            member(Head-true, Clauses),
            Head =.. [_|[A1, A2|_]],
            format(string(ObjEntry), "        [PSCustomObject]@{ X='~w'; Y='~w' }", [A1, A2])
        ), ObjEntries),
        atomic_list_concat(ObjEntries, ',\n', ObjStr),
        format(string(Code),
"# Generated Pure PowerShell Script
# Predicate: ~w/~w
# Generated: ~w
# Generated by UnifyWeaver PowerShell Compiler (Pure Mode)

function ~w {
    param(~w)
    
    $facts = @(
~w
    )
~w
}

# Stream all facts
~w
", [Pred, Arity, DateStr, PredStr, ParamStr, ObjStr, FilterLogic, PredStr])
    ).

%% format_ps_fact_entry(+Args, -Entry)
%  Format fact arguments for PowerShell array
format_ps_fact_entry([Arg], Entry) :-
    format(string(Entry), "        '~w'", [Arg]).
format_ps_fact_entry([A, B|_], Entry) :-
    format(string(Entry), "        [PSCustomObject]@{ X='~w'; Y='~w' }", [A, B]).

%% compile_rule_to_powershell(+Pred, +Arity, +Head, +Body, +Options, -Code)
%  Compile a rule with joins and optional negation to pure PowerShell
compile_rule_to_powershell(Pred, Arity, _Head, Body, _Options, Code) :-
    atom_string(Pred, PredStr),
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),
    
    % Extract body goals
    body_to_list_ps(Body, Goals),
    length(Goals, NumGoals),
    format('[PowerShell Pure] Body has ~w goals~n', [NumGoals]),
    
    % Partition into positive and negated goals
    partition(is_negated_goal_ps, Goals, NegatedGoals, PositiveGoals),
    length(NegatedGoals, NumNeg),
    format('[PowerShell Pure] ~w negated goals~n', [NumNeg]),
    
    % Handle rules based on structure
    (   PositiveGoals = [Goal1, Goal2],
        Goal1 =.. [Pred1|_],
        Goal2 =.. [Pred2|_]
    ->  % Join between two predicates with optional negation
        atom_string(Pred1, Pred1Str),
        atom_string(Pred2, Pred2Str),
        (   NegatedGoals \= []
        ->  generate_negation_check_ps(NegatedGoals, NegCheckCode),
            generate_negated_facts_loaders(NegatedGoals, NegLoaderCode)
        ;   NegCheckCode = "",
            NegLoaderCode = ""
        ),
        format(string(Code),
"# Generated Pure PowerShell Script
# Predicate: ~w/~w (join~w)
# Generated: ~w
# Generated by UnifyWeaver PowerShell Compiler (Pure Mode)

function ~w {
    param([string]$X, [string]$Z)
    
    # Get facts from both relations
    $rel1 = ~w
    $rel2 = ~w
~w
    # Nested loop join: rel1.Y = rel2.X
    $results = foreach ($r1 in $rel1) {
        foreach ($r2 in $rel2) {
            if ($r1.Y -eq $r2.X) {
                $x = $r1.X
                $z = $r2.Y~w
                [PSCustomObject]@{
                    X = $x
                    Z = $z
                }
            }
        }
    }
    
    if ($X -and $Z) {
        $results | Where-Object { $_.X -eq $X -and $_.Z -eq $Z }
    } elseif ($X) {
        $results | Where-Object { $_.X -eq $X } | ForEach-Object { $_.Z }
    } elseif ($Z) {
        $results | Where-Object { $_.Z -eq $Z } | ForEach-Object { $_.X }
    } else {
        $results | ForEach-Object { \"$($_.X):$($_.Z)\" }
    }
}

# Stream all results
~w
", [Pred, Arity, (NegatedGoals \= [] -> " with negation" ; ""), DateStr, PredStr, Pred1Str, Pred2Str, 
   NegLoaderCode, NegCheckCode, PredStr])
    ;   PositiveGoals = [Goal1],
        Goal1 =.. [Pred1|_],
        NegatedGoals \= []
    ->  % Single positive goal with negation
        atom_string(Pred1, Pred1Str),
        generate_negation_check_ps(NegatedGoals, NegCheckCode),
        generate_negated_facts_loaders(NegatedGoals, NegLoaderCode),
        format(string(Code),
"# Generated Pure PowerShell Script
# Predicate: ~w/~w (filter with negation)
# Generated: ~w
# Generated by UnifyWeaver PowerShell Compiler (Pure Mode)

function ~w {
    param([string]$X, [string]$Y)
    
    # Get facts from source relation
    $facts = ~w
~w
    # Filter with negation check
    $results = foreach ($f in $facts) {
        $x = $f.X
        $y = $f.Y~w
        [PSCustomObject]@{ X = $x; Y = $y }
    }
    
    if ($X -and $Y) {
        $results | Where-Object { $_.X -eq $X -and $_.Y -eq $Y }
    } elseif ($X) {
        $results | Where-Object { $_.X -eq $X } | ForEach-Object { $_.Y }
    } elseif ($Y) {
        $results | Where-Object { $_.Y -eq $Y } | ForEach-Object { $_.X }
    } else {
        $results | ForEach-Object { \"$($_.X):$($_.Y)\" }
    }
}

# Stream all results
~w
", [Pred, Arity, DateStr, PredStr, Pred1Str, 
   NegLoaderCode, NegCheckCode, PredStr])
    ;   % Unsupported rule pattern - provide informative error
        format(string(Code),
"# Generated Pure PowerShell Script
# Predicate: ~w/~w
# Generated: ~w
# ERROR: Complex rule pattern not yet supported in pure PowerShell mode
# Please use BaaS mode: compile_to_powershell(~w/~w, [powershell_mode(baas)], Code)

Write-Error 'Complex rule pattern not supported in pure PowerShell mode'
", [Pred, Arity, DateStr, Pred, Arity])
    ).

%% is_negated_goal_ps(+Goal)
%  Check if goal is a negation
is_negated_goal_ps(\+ _).
is_negated_goal_ps(not(_)).

%% generate_negation_check_ps(+NegatedGoals, -Code)
%  Generate PowerShell code to check negated facts
generate_negation_check_ps([], "").
generate_negation_check_ps([NegGoal|Rest], Code) :-
    extract_negated_pred_ps(NegGoal, NegPred),
    atom_string(NegPred, NegPredStr),
    format(string(Check), "
                # Negation check: \\+~w
                $negKey = \"$x:$y\"
                if ($~w_set.ContainsKey($negKey)) { continue }", [NegPred, NegPredStr]),
    (   Rest = []
    ->  Code = Check
    ;   generate_negation_check_ps(Rest, RestCode),
        format(string(Code), "~w~w", [Check, RestCode])
    ).

%% extract_negated_pred_ps(+NegGoal, -Pred)
extract_negated_pred_ps(\+ Goal, Pred) :-
    Goal =.. [Pred|_].
extract_negated_pred_ps(not(Goal), Pred) :-
    Goal =.. [Pred|_].

%% generate_negated_facts_loaders(+NegatedGoals, -Code)
%  Generate code to load negated facts into hashtables for efficient lookup
generate_negated_facts_loaders([], "").
generate_negated_facts_loaders(NegatedGoals, Code) :-
    NegatedGoals \= [],
    findall(LoaderCode, (
        member(NegGoal, NegatedGoals),
        extract_negated_pred_ps(NegGoal, NegPred),
        atom_string(NegPred, NegPredStr),
        format(string(LoaderCode), "
    # Load negated facts: ~w
    $~w_set = @{}
    foreach ($nf in ~w) { $~w_set[\"$($nf.X):$($nf.Y)\"] = $true }
", [NegPred, NegPredStr, NegPredStr, NegPredStr])
    ), LoaderCodes),
    atomic_list_concat(LoaderCodes, '', Code).

%% body_to_list_ps(+Body, -Goals)
%  Convert conjunctive body to list of goals
body_to_list_ps((A, B), [A|Rest]) :- !,
    body_to_list_ps(B, Rest).
body_to_list_ps(true, []) :- !.
body_to_list_ps(Goal, [Goal]).

%% ============================================================================
%% PROCEDURAL RECURSION
%% ============================================================================

%% compile_recursive_to_powershell(+Pred, +Arity, +Clauses, +Options, -Code)
%  Compile recursive Prolog predicates to PowerShell functions
compile_recursive_to_powershell(Pred, Arity, Clauses, _Options, Code) :-
    atom_string(Pred, PredStr),
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),
    
    % Separate base cases (facts) from recursive rules
    partition(is_fact_clause_ps, Clauses, BaseCases, RecursiveCases),
    length(BaseCases, NumBase),
    length(RecursiveCases, NumRec),
    format('[PowerShell Pure] ~w base cases, ~w recursive cases~n', [NumBase, NumRec]),
    
    % Generate parameter list based on arity
    generate_ps_params(Arity, ParamStr, ArgNames),
    
    % Generate base case code
    generate_base_cases_ps(BaseCases, ArgNames, BaseCaseCode),
    
    % Generate recursive case code
    generate_recursive_cases_ps(RecursiveCases, PredStr, ArgNames, RecursiveCaseCode),
    
    format(string(Code),
"# Generated Pure PowerShell Script (Recursive)
# Predicate: ~w/~w
# Generated: ~w
# Generated by UnifyWeaver PowerShell Compiler (Pure Mode)

function ~w {
~w
~w
~w
}

# Call the function
~w
", [Pred, Arity, DateStr, PredStr, ParamStr, BaseCaseCode, RecursiveCaseCode, PredStr]).

%% is_fact_clause_ps(+Clause)
is_fact_clause_ps(_-true).

%% generate_ps_params(+Arity, -ParamStr, -ArgNames)
generate_ps_params(1, "    param([int]$N)", ["N"]).
generate_ps_params(2, "    param($Arg1, $Arg2)", ["Arg1", "Arg2"]).
generate_ps_params(3, "    param($List, $Acc, $Result)", ["List", "Acc", "Result"]).
generate_ps_params(Arity, ParamStr, ArgNames) :-
    Arity > 3,
    numlist(1, Arity, Nums),
    maplist([N, Name]>>format(atom(Name), '$Arg~w', [N]), Nums, ArgNames),
    atomic_list_concat(ArgNames, ', ', ArgsJoined),
    format(string(ParamStr), "    param(~w)", [ArgsJoined]).

%% generate_base_cases_ps(+BaseCases, +ArgNames, -Code)
generate_base_cases_ps([], _, "    # No explicit base cases").
generate_base_cases_ps([Head-true|Rest], ArgNames, Code) :-
    Head =.. [_|Args],
    generate_base_case_check_ps(Args, ArgNames, CheckCode),
    (   Rest = []
    ->  format(string(Code), "    # Base case
~w", [CheckCode])
    ;   generate_base_cases_ps(Rest, ArgNames, RestCode),
        format(string(Code), "    # Base case
~w
~w", [CheckCode, RestCode])
    ).

%% generate_base_case_check_ps(+Args, +ArgNames, -Code)
generate_base_case_check_ps([Val], [ArgName], Code) :-
    format(string(Code), "    if ($~w -eq ~w) { return ~w }", [ArgName, Val, Val]).
generate_base_case_check_ps([Val1, Val2], [Arg1, Arg2], Code) :-
    format(string(Code), "    if ($~w -eq ~w -and $~w -eq ~w) { return @($~w, $~w) }", 
           [Arg1, Val1, Arg2, Val2, Arg1, Arg2]).
% Accumulator pattern: sum_list([], Acc, Acc)
generate_base_case_check_ps([[], AccVal, AccVal], [ListArg, AccArg, _ResultArg], Code) :-
    format(string(Code), "    if ($~w.Count -eq 0) { return $~w }", [ListArg, AccArg]).
generate_base_case_check_ps([Val1, Val2, Val3], [Arg1, Arg2, Arg3], Code) :-
    format(string(Code), "    if ($~w -eq ~w -and $~w -eq ~w -and $~w -eq ~w) { return @($~w, $~w, $~w) }", 
           [Arg1, Val1, Arg2, Val2, Arg3, Val3, Arg1, Arg2, Arg3]).

%% generate_recursive_cases_ps(+RecursiveCases, +PredStr, +ArgNames, -Code)
generate_recursive_cases_ps([], _, _, "    # No recursive cases").
generate_recursive_cases_ps([Head-Body|_], PredStr, ArgNames, Code) :-
    % Analyze the body to generate recursive call
    body_to_list_ps(Body, Goals),
    % Find and classify the recursive pattern
    find_recursive_pattern_ps(Goals, PredStr, Pattern),
    format('[PowerShell Pure] Detected pattern: ~w~n', [Pattern]),
    generate_recursive_code_ps(Pattern, PredStr, ArgNames, Code).

%% find_recursive_pattern_ps(+Goals, +PredStr, -Pattern)
%  Classify the recursion pattern based on body structure
find_recursive_pattern_ps(Goals, PredStr, Pattern) :-
    atom_string(PredAtom, PredStr),
    % Count recursive calls
    findall(Call, (member(Call, Goals), Call =.. [PredAtom|_]), RecCalls),
    length(RecCalls, NumRec),
    
    % Look for arithmetic operations
    findall(ArithGoal, (member(ArithGoal, Goals), ArithGoal = (_ is _)), ArithGoals),
    
    % Look for list operations (head/tail)
    findall(ListGoal, (member(ListGoal, Goals), is_list_op_ps(ListGoal)), ListGoals),
    
    % Classify pattern
    (   NumRec = 2
    ->  % Fibonacci-like: two recursive calls
        Pattern = fibonacci
    ;   NumRec = 1, ListGoals \= []
    ->  % Accumulator pattern: list processing with recursion
        Pattern = accumulator
    ;   NumRec = 1, ArithGoals \= []
    ->  % Factorial-like: single recursion with arithmetic
        Pattern = factorial
    ;   NumRec = 1
    ->  % Simple recursion
        Pattern = simple_recursive
    ;   Pattern = unknown
    ).

%% is_list_op_ps(+Goal)
%  Check if goal is a list operation (unification with [H|T])
is_list_op_ps(Goal) :-
    Goal = (_ = [_|_]).
is_list_op_ps(Goal) :-
    Goal =.. [=, _, List],
    is_list(List).

%% generate_recursive_code_ps(+Pattern, +PredStr, +ArgNames, -Code)
%  Generate PowerShell code for each recursion pattern

% Fibonacci pattern: fib(N) = fib(N-1) + fib(N-2)
generate_recursive_code_ps(fibonacci, PredStr, [ArgName|_], Code) :-
    format(string(Code),
"    # Fibonacci pattern: two recursive calls
    if ($~w -le 1) { return $~w }
    
    $n1 = $~w - 1
    $n2 = $~w - 2
    $r1 = ~w $n1
    $r2 = ~w $n2
    return $r1 + $r2", [ArgName, ArgName, ArgName, ArgName, PredStr, PredStr]).

% Factorial pattern: fact(N) = N * fact(N-1)
generate_recursive_code_ps(factorial, PredStr, [ArgName|_], Code) :-
    format(string(Code),
"    # Factorial pattern: N * recursive_call(N-1)
    $n1 = $~w - 1
    $result = ~w $n1
    return $~w * $result", [ArgName, PredStr, ArgName]).

% Accumulator pattern: process list with accumulator
generate_recursive_code_ps(accumulator, PredStr, ArgNames, Code) :-
    (   ArgNames = [ListArg, AccArg, ResultArg|_]
    ->  format(string(Code),
"    # Accumulator pattern: process list head, recurse on tail
    if ($~w.Count -eq 0) { return $~w }
    
    $head = $~w[0]
    $tail = $~w[1..($~w.Count - 1)]
    $newAcc = $~w + $head
    return ~w $tail $newAcc $~w", 
           [ListArg, AccArg, ListArg, ListArg, ListArg, AccArg, PredStr, ResultArg])
    ;   ArgNames = [ListArg, AccArg|_]
    ->  format(string(Code),
"    # Accumulator pattern: process list head, recurse on tail
    if ($~w.Count -eq 0) { return $~w }
    
    $head = $~w[0]
    $tail = $~w[1..($~w.Count - 1)]
    $newAcc = $~w + $head
    return ~w $tail $newAcc", 
           [ListArg, AccArg, ListArg, ListArg, ListArg, AccArg, PredStr])
    ;   Code = "    # Accumulator pattern requires list and accumulator arguments
    Write-Error 'Invalid accumulator arguments'"
    ).

% Simple recursive pattern
generate_recursive_code_ps(simple_recursive, PredStr, [ArgName|_], Code) :-
    format(string(Code),
"    # Simple recursive pattern
    $n1 = $~w - 1
    return ~w $n1", [ArgName, PredStr]).

% Unknown pattern - fallback
generate_recursive_code_ps(unknown, _, _, 
"    # Unknown recursion pattern
    Write-Error 'Complex recursion pattern not supported'").

%% ============================================================================
%% GENERATOR MODE (FIXPOINT)
%% ============================================================================

%% compile_generator_mode_powershell(+Pred, +Arity, +Clauses, +Options, -Code)
%  Compile to fixpoint-style generator (like Go generator mode)
compile_generator_mode_powershell(Pred, Arity, Clauses, Options, Code) :-
    atom_string(Pred, PredStr),
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),
    
    % Gather all dependency predicates
    gather_dependencies_ps(Clauses, Pred, Dependencies),
    format('[PowerShell Generator] Dependencies: ~w~n', [Dependencies]),
    
    % Separate facts from rules
    partition(is_fact_clause_ps, Clauses, FactClauses, RuleClauses),
    
    % Generate initial facts
    generate_initial_facts_ps(FactClauses, Dependencies, Options, InitialFactsCode),
    
    % Generate rule application functions
    generate_rule_functions_ps(Pred, RuleClauses, RuleFunctionsCode),
    
    % Generate the Solve-Fixpoint function
    generate_fixpoint_loop_ps(PredStr, SolveCode),
    
    format(string(Code),
"# Generated Pure PowerShell Script (Generator/Fixpoint Mode)
# Predicate: ~w/~w
# Generated: ~w
# Generated by UnifyWeaver PowerShell Compiler (Generator Mode)

# Fact representation
class Fact {
    [string]$Pred
    [object[]]$Args
    
    Fact([string]$pred, [object[]]$args) {
        $this.Pred = $pred
        $this.Args = $args
    }
    
    [string] Key() {
        return \"$($this.Pred):$($this.Args -join ':')\")
    }
}

# Global fact storage
$script:total = @{}
$script:delta = @{}

~w

~w

~w

# Run fixpoint computation
Solve-~w
", [Pred, Arity, DateStr, InitialFactsCode, RuleFunctionsCode, SolveCode, PredStr]).

%% gather_dependencies_ps(+Clauses, +Pred, -Dependencies)
gather_dependencies_ps(Clauses, Pred, Dependencies) :-
    findall(Dep, (
        member(_-Body, Clauses),
        Body \= true,
        body_to_list_ps(Body, Goals),
        member(Goal, Goals),
        Goal \= (\+ _),
        Goal =.. [Dep|_],
        Dep \= Pred,
        Dep \= is,
        Dep \= '>',
        Dep \= '<',
        Dep \= '>=',
        Dep \= '=<'
    ), DepsWithDups),
    sort(DepsWithDups, Dependencies).

%% generate_initial_facts_ps(+FactClauses, +Dependencies, +Options, -Code)
generate_initial_facts_ps(FactClauses, Dependencies, _Options, Code) :-
    % Generate fact loading for main predicate
    findall(FactCode, (
        member(Head-true, FactClauses),
        Head =.. [Pred|Args],
        format_args_ps(Args, ArgsStr),
        format(string(FactCode), "$script:total[[Fact]::new('~w', @(~w)).Key()] = [Fact]::new('~w', @(~w))", 
               [Pred, ArgsStr, Pred, ArgsStr])
    ), FactCodes),
    (   FactCodes = []
    ->  MainFactsCode = "# No initial facts"
    ;   atomic_list_concat(FactCodes, '\n', MainFactsCode)
    ),
    
    % Generate dependency fact loading
    findall(DepCode, (
        member(DepPred, Dependencies),
        % Find all facts for this predicate
        user:current_predicate(DepPred/Arity),
        functor(DepHead, DepPred, Arity),
        user:clause(DepHead, true),
        DepHead =.. [_|DepArgs],
        format_args_ps(DepArgs, DepArgsStr),
        format(string(DepCode), "    $script:total[[Fact]::new('~w', @(~w)).Key()] = [Fact]::new('~w', @(~w))",
               [DepPred, DepArgsStr, DepPred, DepArgsStr])
    ), DepCodes),
    
    atomic_list_concat(DepCodes, '\n', DepFactsCode),
    format(string(Code), 
"# Initial facts
function Initialize-Facts {
~w
~w
}", [MainFactsCode, DepFactsCode]).

%% format_args_ps(+Args, -Str)
format_args_ps([], "").
format_args_ps([Arg], Str) :-
    format(string(Str), "'~w'", [Arg]).
format_args_ps([A|Rest], Str) :-
    Rest \= [],
    format_args_ps(Rest, RestStr),
    format(string(Str), "'~w', ~w", [A, RestStr]).

%% generate_rule_functions_ps(+Pred, +RuleClauses, -Code)
%  Generate Apply-* functions for all rules
generate_rule_functions_ps(Pred, RuleClauses, Code) :-
    atom_string(Pred, PredStr),
    (   RuleClauses = []
    ->  Code = "# No rules"
    ;   % Generate code for each rule
        findall(RuleCode-RuleType, (
            member(_Head-Body, RuleClauses),
            body_to_list_ps(Body, Goals),
            partition(is_recursive_goal_ps(PredStr), Goals, RecGoals, BaseGoals),
            length(RecGoals, NumRec),
            (   NumRec = 0
            ->  RuleType = base
            ;   RuleType = recursive
            ),
            generate_apply_rule_ps(PredStr, Goals, RuleCode)
        ), RuleCodePairs),
        % Separate base and recursive rules
        findall(C, member(C-base, RuleCodePairs), BaseCodes),
        findall(C, member(C-recursive, RuleCodePairs), RecCodes),
        % Combine codes
        append(BaseCodes, RecCodes, AllCodes),
        atomic_list_concat(AllCodes, '\n\n', Code)
    ).

%% generate_apply_rule_ps(+PredStr, +Goals, -Code)
%  Generates rule application code for different patterns:
%  1. Two base goals (join between two different relations)
%  2. One base + one recursive (self-join pattern like ancestor)
%  3. Single base goal (copy pattern)
generate_apply_rule_ps(PredStr, Goals, Code) :-
    % Partition goals into recursive and non-recursive
    partition(is_recursive_goal_ps(PredStr), Goals, RecGoals, BaseGoals),
    length(BaseGoals, NumBase),
    length(RecGoals, NumRec),
    format('[PowerShell Generator] ~w base goals, ~w recursive goals~n', [NumBase, NumRec]),
    
    (   NumBase = 2, NumRec = 0
    ->  % Pattern 1: Two base goals (regular join)
        BaseGoals = [Goal1, Goal2|_],
        Goal1 =.. [Pred1|_],
        Goal2 =.. [Pred2|_],
        atom_string(Pred1, Pred1Str),
        atom_string(Pred2, Pred2Str),
        generate_two_base_join_ps(PredStr, Pred1Str, Pred2Str, Code)
    
    ;   NumBase = 1, NumRec = 1
    ->  % Pattern 2: Self-join (one base + one recursive)
        BaseGoals = [BaseGoal],
        BaseGoal =.. [BasePred|_],
        atom_string(BasePred, BasePredStr),
        generate_self_join_ps(PredStr, BasePredStr, Code)
    
    ;   NumBase = 1, NumRec = 0
    ->  % Pattern 3: Single base goal (copy pattern - like ancestor(X,Y) :- parent(X,Y))
        BaseGoals = [BaseGoal],
        BaseGoal =.. [BasePred|_],
        atom_string(BasePred, BasePredStr),
        generate_copy_rule_ps(PredStr, BasePredStr, Code)
    
    ;   % Unsupported pattern
        format(string(Code),
"# Unsupported rule pattern: ~w base, ~w recursive goals
function Apply-~wRule {
    param($facts)
    # Rule pattern not yet supported
}", [NumBase, NumRec, PredStr])
    ).

%% generate_two_base_join_ps(+PredStr, +Pred1Str, +Pred2Str, -Code)
%  Generate join between two different base relations
generate_two_base_join_ps(PredStr, Pred1Str, Pred2Str, Code) :-
    format(string(Code),
"# Apply rule: ~w from ~w and ~w (two-base join)
function Apply-~wRule {
    param($facts)
    
    $rel1 = $facts.Values | Where-Object { $_.Pred -eq '~w' }
    $rel2 = $facts.Values | Where-Object { $_.Pred -eq '~w' }
    
    foreach ($f1 in $rel1) {
        foreach ($f2 in $rel2) {
            if ($f1.Args[1] -eq $f2.Args[0]) {
                $newFact = [Fact]::new('~w', @($f1.Args[0], $f2.Args[1]))
                $key = $newFact.Key()
                if (-not $facts.ContainsKey($key)) {
                    $script:delta[$key] = $newFact
                }
            }
        }
    }
}", [PredStr, Pred1Str, Pred2Str, PredStr, Pred1Str, Pred2Str, PredStr]).

%% generate_self_join_ps(+PredStr, +BasePredStr, -Code)
%  Generate self-join: base relation joined with derived relation
%  Example: ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)
generate_self_join_ps(PredStr, BasePredStr, Code) :-
    format(string(Code),
"# Apply rule: ~w from ~w and ~w (self-join/transitive closure)
function Apply-~wRule {
    param($facts)
    
    # Base relation (e.g., parent)
    $base = $facts.Values | Where-Object { $_.Pred -eq '~w' }
    # Derived relation (e.g., ancestor) - includes what we've computed so far
    $derived = $facts.Values | Where-Object { $_.Pred -eq '~w' }
    
    # Join: base.Y = derived.X (transitive step)
    foreach ($b in $base) {
        foreach ($d in $derived) {
            if ($b.Args[1] -eq $d.Args[0]) {
                $newFact = [Fact]::new('~w', @($b.Args[0], $d.Args[1]))
                $key = $newFact.Key()
                if (-not $facts.ContainsKey($key)) {
                    $script:delta[$key] = $newFact
                }
            }
        }
    }
}", [PredStr, BasePredStr, PredStr, PredStr, BasePredStr, PredStr, PredStr]).

%% generate_copy_rule_ps(+PredStr, +BasePredStr, -Code)
%  Generate copy rule: derived = base (e.g., ancestor(X,Y) :- parent(X,Y))
generate_copy_rule_ps(PredStr, BasePredStr, Code) :-
    format(string(Code),
"# Apply rule: ~w from ~w (copy/base case rule)
function Apply-~wBaseRule {
    param($facts)
    
    # Copy base facts to derived relation
    $base = $facts.Values | Where-Object { $_.Pred -eq '~w' }
    
    foreach ($b in $base) {
        $newFact = [Fact]::new('~w', @($b.Args[0], $b.Args[1]))
        $key = $newFact.Key()
        if (-not $facts.ContainsKey($key)) {
            $script:delta[$key] = $newFact
        }
    }
}", [PredStr, BasePredStr, PredStr, BasePredStr, PredStr]).

%% is_recursive_goal_ps(+PredStr, +Goal)
is_recursive_goal_ps(PredStr, Goal) :-
    Goal =.. [Pred|_],
    atom_string(Pred, PredStr).

%% generate_fixpoint_loop_ps(+PredStr, -Code)
generate_fixpoint_loop_ps(PredStr, Code) :-
    format(string(Code),
"# Fixpoint computation
function Solve-~w {
    Initialize-Facts
    
    # First, apply base rule (copy from base relation)
    $script:delta = @{}
    Apply-~wBaseRule $script:total
    foreach ($key in $script:delta.Keys) {
        $script:total[$key] = $script:delta[$key]
    }
    Write-Host \"Base rule applied: $($script:delta.Count) initial derived facts\" -ForegroundColor Yellow
    
    # Then iterate until fixpoint
    $iteration = 0
    do {
        $iteration++
        $script:delta = @{}
        
        # Apply recursive rule (self-join)
        Apply-~wRule $script:total
        
        # Merge delta into total
        foreach ($key in $script:delta.Keys) {
            $script:total[$key] = $script:delta[$key]
        }
        
        Write-Host \"Iteration $iteration: ~w new facts\" -ForegroundColor Cyan
    } while ($script:delta.Count -gt 0)
    
    Write-Host \"Fixpoint reached after $iteration iterations\" -ForegroundColor Green
    
    # Output results
    $script:total.Values | Where-Object { $_.Pred -eq '~w' } | ForEach-Object {
        \"$($_.Args -join ':')\"
    }
}", [PredStr, PredStr, PredStr, '$($script:delta.Count)', PredStr]).

%% compile_to_baas_powershell(+Predicate, +Options, -PowerShellCode)
%  Compile using bash-as-a-service approach (original implementation)
compile_to_baas_powershell(Predicate, Options, PowerShellCode) :-
    % Determine which bash compiler to use
    option(compiler(Compiler), Options, stream),
    format('[PowerShell Compiler] Using ~w compiler~n', [Compiler]),

    % Generate bash code using selected compiler
    compile_to_bash(Compiler, Predicate, Options, BashCode),

    % Wrap bash code in PowerShell compatibility layer invocation
    powershell_wrapper(BashCode, Options, PowerShellCode).

%% compile_to_bash(+Compiler, +Predicate, +Options, -BashCode)
%  Internal: dispatch to appropriate bash compiler
compile_to_bash(stream, Predicate, Options, BashCode) :-
    stream_compiler:compile_predicate(Predicate, Options, BashCode).

compile_to_bash(recursive, Predicate, Options, BashCode) :-
    recursive_compiler:compile_predicate(Predicate, Options, BashCode).

%% powershell_wrapper(+BashCode, +Options, -PowerShellCode)
%  Wraps bash code in PowerShell compatibility layer invocation
%
%  Options:
%    - wrapper_style(inline|tempfile) - default: inline
%    - compat_check(true|false) - default: true
%    - script_name(Name) - for comments
%    - predicate(Pred/Arity) - for comments
%
powershell_wrapper(BashCode, Options, PowerShellCode) :-
    option(wrapper_style(Style), Options, inline),
    option(compat_check(CompatCheck), Options, true),
    option(script_name(ScriptName), Options, 'generated_script'),

    format('[PowerShell Compiler] Wrapper style: ~w~n', [Style]),

    % Generate appropriate wrapper
    generate_powershell_wrapper(Style, BashCode, ScriptName, CompatCheck, PowerShellCode).

%% generate_powershell_wrapper(+Style, +BashCode, +ScriptName, +CompatCheck, -PowerShellCode)
%  Generate PowerShell wrapper based on style

% Inline style: Bash code as heredoc, executed via uw-bash
generate_powershell_wrapper(inline, BashCode, ScriptName, CompatCheck, PowerShellCode) :-
    % Get timestamp
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),

    % Build compatibility check code
    (   CompatCheck = true
    ->  CompatCheckCode = '# Ensure compatibility layer is available\nif (-not (Get-Command uw-bash -ErrorAction SilentlyContinue)) {\n    Write-Error "UnifyWeaver PowerShell compatibility layer not loaded"\n    Write-Host "Please run: . .\\scripts\\init_unify_compat.ps1" -ForegroundColor Yellow\n    exit 1\n}\n\n'
    ;   CompatCheckCode = ''
    ),

    % Construct PowerShell script
    atomic_list_concat([
        '# Generated PowerShell Script\n',
        '# Script: ', ScriptName, '.ps1\n',
        '# Generated by UnifyWeaver PowerShell Compiler\n',
        '# Generated: ', DateStr, '\n',
        '#\n',
        '# This script wraps a bash implementation via the PowerShell compatibility layer.\n',
        '# The bash code is executed using uw-bash from UnifyWeaver compatibility layer.\n',
        '\n',
        CompatCheckCode,
        '# Embedded bash implementation\n',
        '$bashScript = @''\n',
        BashCode, '\n',
        '''@\n',
        '\n',
        '# Execute via compatibility layer\n',
        'if ($Input) {\n',
        '    $Input | uw-bash -c $bashScript\n',
        '} else {\n',
        '    uw-bash -c $bashScript\n',
        '}\n'
    ], PowerShellCode).

% Tempfile style: Write bash to temp file, execute, cleanup
generate_powershell_wrapper(tempfile, BashCode, ScriptName, CompatCheck, PowerShellCode) :-
    % Get timestamp
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),

    % Build compatibility check code
    (   CompatCheck = true
    ->  CompatCheckCode = '# Ensure compatibility layer is available\nif (-not (Get-Command uw-bash -ErrorAction SilentlyContinue)) {\n    Write-Error "UnifyWeaver PowerShell compatibility layer not loaded"\n    Write-Host "Please run: . .\\scripts\\init_unify_compat.ps1" -ForegroundColor Yellow\n    exit 1\n}\n\n'
    ;   CompatCheckCode = ''
    ),

    % Construct PowerShell script with tempfile approach
    atomic_list_concat([
        '# Generated PowerShell Script\n',
        '# Script: ', ScriptName, '.ps1\n',
        '# Generated by UnifyWeaver PowerShell Compiler\n',
        '# Generated: ', DateStr, '\n',
        '#\n',
        '# This script creates a temporary bash file and executes it.\n',
        '\n',
        CompatCheckCode,
        '# Create temporary bash script\n',
        '$tempFile = [System.IO.Path]::GetTempFileName() + ".sh"\n',
        '\n',
        '# Embedded bash implementation\n',
        '$bashScript = @''\n',
        BashCode, '\n',
        '''@\n',
        '\n',
        '# Write to temp file and execute\n',
        'try {\n',
        '    Set-Content -Path $tempFile -Value $bashScript -Encoding UTF8\n',
        '    \n',
        '    if ($Input) {\n',
        '        $Input | uw-bash $tempFile\n',
        '    } else {\n',
        '        uw-bash $tempFile\n',
        '    }\n',
        '} finally {\n',
        '    # Cleanup temp file\n',
        '    if (Test-Path $tempFile) {\n',
        '        Remove-Item $tempFile -ErrorAction SilentlyContinue\n',
        '    }\n',
        '}\n'
    ], PowerShellCode).

%% write_powershell_file(+FilePath, +PowerShellCode)
%  Write PowerShell code to file
write_powershell_file(FilePath, PowerShellCode) :-
    open(FilePath, write, Stream, [encoding(utf8)]),
    write(Stream, PowerShellCode),
    close(Stream).

%% option(+Key, +Options, -Value)
%% option(+Key, +Options, -Value, +Default)
%  Helper to extract option from list
option(Key, Options, Default) :-
    (   member(Key, Options)
    ->  true
    ;   Key =.. [Functor, Default],
        \+ member(Functor=_, Options)
    ).

%% ============================================================================
%% TESTS
%% ============================================================================

test_powershell_compiler :-
    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  PowerShell Compiler Tests            ║~n', []),
    format('╚════════════════════════════════════════╝~n~n', []),

    % Test 1: Basic inline wrapper
    format('~n[Test 1] Inline wrapper generation~n', []),
    test_inline_wrapper,

    % Test 2: Tempfile wrapper
    format('~n[Test 2] Tempfile wrapper generation~n', []),
    test_tempfile_wrapper,

    % Test 3: Compilation with options
    format('~n[Test 3] Compilation with various options~n', []),
    test_compilation_options,

    format('~n╔════════════════════════════════════════╗~n', []),
    format('║  All PowerShell Compiler Tests Passed ║~n', []),
    format('╚════════════════════════════════════════╝~n', []).

test_inline_wrapper :-
    BashCode = '#!/bin/bash\necho "Hello from bash"',
    powershell_wrapper(BashCode, [wrapper_style(inline), script_name(test_inline)], PSCode),

    % Verify it contains key components
    sub_string(PSCode, _, _, _, '$bashScript = @'''),
    sub_string(PSCode, _, _, _, 'uw-bash -c $bashScript'),
    sub_string(PSCode, _, _, _, 'if (-not (Get-Command uw-bash'),

    format('[✓] Inline wrapper contains expected components~n', []).

test_tempfile_wrapper :-
    BashCode = '#!/bin/bash\necho "Hello from bash"',
    powershell_wrapper(BashCode, [wrapper_style(tempfile), script_name(test_tempfile)], PSCode),

    % Verify it contains key components
    sub_string(PSCode, _, _, _, '$tempFile = [System.IO.Path]::GetTempFileName'),
    sub_string(PSCode, _, _, _, 'Set-Content -Path $tempFile'),
    sub_string(PSCode, _, _, _, 'Remove-Item $tempFile'),

    format('[✓] Tempfile wrapper contains expected components~n', []).

test_compilation_options :-
    % Test without compat check
    BashCode = '#!/bin/bash\necho "Test"',
    powershell_wrapper(BashCode, [wrapper_style(inline), compat_check(false)], PSCode1),
    \+ sub_string(PSCode1, _, _, _, 'if (-not (Get-Command uw-bash'),
    format('[✓] compat_check(false) omits compatibility check~n', []),

    % Test with compat check (default)
    powershell_wrapper(BashCode, [wrapper_style(inline)], PSCode2),
    sub_string(PSCode2, _, _, _, 'if (-not (Get-Command uw-bash'),
    format('[✓] Default includes compatibility check~n', []).
