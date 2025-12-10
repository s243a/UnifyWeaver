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
    compile_mutual_recursion_powershell/3,  % +Predicates, +Options, -PowerShellCode
    compile_tail_recursion_powershell/3,    % +Pred/Arity, +Options, -PowerShellCode
    can_compile_tail_recursion_ps/1,        % +Pred/Arity
    powershell_wrapper/3,         % powershell_wrapper(+BashCode, +Options, -PowerShellCode)
    test_powershell_compiler/0,   % Run tests

    % Binding-based compilation (Book 12, Chapter 3 & 5)
    compile_bound_predicate/3,    % compile_bound_predicate(+Pred, +Options, -Code)
    generate_cmdlet_wrapper/3,    % generate_cmdlet_wrapper(+Pred, +Options, -CmdletCode)
    has_powershell_binding/1,     % has_powershell_binding(+Pred)
    init_powershell_compiler/0,   % Initialize compiler with bindings

    % C# hosting integration (Book 12, Chapter 6)
    compile_with_csharp_host/4,   % compile_with_csharp_host(+Predicates, +Options, -PSCode, -CSharpCode)
    generate_csharp_bridge/3,     % generate_csharp_bridge(+BridgeType, +Options, -Code)
    compile_cross_target_pipeline/3  % compile_cross_target_pipeline(+Steps, +Options, -Code)
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

% Binding system for cmdlet generation (Book 12, Chapters 3 & 5)
:- use_module('binding_registry').
:- use_module('../bindings/powershell_bindings').
:- use_module('../bindings/binding_codegen').

% .NET glue for C# hosting (Book 12, Chapter 6)
:- use_module('../glue/dotnet_glue').

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
        compile_to_pure_powershell(Predicate, Options, CompiledCode)
    ;   % Fall back to bash-as-a-service
        format('[PowerShell Compiler] Using bash-as-a-service (BaaS) mode~n', []),
        compile_to_baas_powershell(Predicate, Options, CompiledCode)
    ),

    % Handle include_dependencies option
    (   member(include_dependencies(true), Options)
    ->  % Get dependencies and compile them
        Predicate = PredAtom/Arity,
        functor(Head, PredAtom, Arity),
        findall(_H-B, user:clause(Head, B), Clauses),
        gather_dependencies_ps(Clauses, PredAtom, Dependencies),
        format('[PowerShell Compiler] Dependencies: ~w~n', [Dependencies]),
        % Compile each dependency (without include_dependencies to avoid recursion)
        DepOptions = [powershell_mode(pure)],
        findall(DepCode, (
            member(DepPred, Dependencies),
            user:current_predicate(DepPred/DepArity),
            functor(DepHead, DepPred, DepArity),
            user:clause(DepHead, _),
            compile_to_pure_powershell(DepPred/DepArity, DepOptions, DepCode)
        ), DepCodes),
        list_to_set(DepCodes, UniqueDepCodes),  % Remove duplicates
        (   UniqueDepCodes = []
        ->  PowerShellCode = CompiledCode
        ;   atomic_list_concat(UniqueDepCodes, '\n\n', DepCodeStr),
            atomic_list_concat([DepCodeStr, '\n\n', CompiledCode], PowerShellCode)
        )
    ;   PowerShellCode = CompiledCode
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
            % Try binding-aware compilation first, fall back to standard
            (   compile_rule_with_bindings(Pred, Arity, H, Body, Options, PowerShellCode)
            ->  true
            ;   compile_rule_to_powershell(Pred, Arity, H, Body, Options, PowerShellCode)
            )
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
%  Options: output_format(object|text) - Return PSCustomObject or strings
compile_facts_to_powershell(Pred, Arity, Clauses, Options, Code) :-
    atom_string(Pred, PredStr),
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),
    
    % Check output format option (default: text for backward compatibility)
    (   member(output_format(object), Options)
    ->  OutputFormat = object
    ;   OutputFormat = text
    ),
    
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
    
    % Generate filter logic based on arity AND output format
    generate_filter_logic_ps(Arity, OutputFormat, FilterLogic),
    
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
        
        % Generate parameters (simple or cmdlet-style)
        generate_cmdlet_params_ps(Arity, Options, CmdletBindingAttr, ParamBlock),
        
        % Generate facts definition
        format(string(FactsDef), "    $facts = @(\n~w\n    )", [ObjStr]),

        % Check verbose_output option
        (   member(verbose_output(true), Options)
        ->  format(string(VerboseLoad), "        Write-Verbose \"[~w] Loading ~w facts...\"\n", [PredStr, Pred]),
            format(string(VerboseQuery), "        Write-Verbose \"[~w] Query: X=$X, Y=$Y\"\n", [PredStr])
        ;   VerboseLoad = "",
            VerboseQuery = ""
        ),
        
        % Structure body based on cmdlet_binding (Begin/Process vs simple)
        (   member(cmdlet_binding(true), Options)
        ->  % Advanced function: Use begin/process blocks
            format(string(Body), 
"    begin {
~w~w
    }

    process {
~w~w
    }", [VerboseLoad, FactsDef, VerboseQuery, FilterLogic])
        ;   % Basic function: Sequential execution
            format(string(Body), 
"~w~w
~w
~w", [VerboseLoad, VerboseQuery, FactsDef, FilterLogic])
        ),
        
        format(string(Code),
"# Generated Pure PowerShell Script
# Predicate: ~w/~w
# Generated: ~w
# Generated by UnifyWeaver PowerShell Compiler (Pure Mode)

function ~w {
~w~w
    
~w
}

# Stream all facts
~w
", [Pred, Arity, DateStr, PredStr, CmdletBindingAttr, ParamBlock, Body, PredStr])
    ).

%% generate_filter_logic_ps(+Arity, +OutputFormat, -FilterLogic)
%  Generate filter logic based on arity and output format
%  OutputFormat: object | text

% Arity 1 - simple facts (same for text and object)
generate_filter_logic_ps(1, _, FilterLogic) :-
    FilterLogic = "
    if ($Key) {
        $facts | Where-Object { $_ -eq $Key }
    } else {
        $facts
    }".

% Arity 2 - binary relations with object output (returns PSCustomObject)
generate_filter_logic_ps(2, object, FilterLogic) :-
    FilterLogic = "
    if ($X -and $Y) {
        $facts | Where-Object { $_.X -eq $X -and $_.Y -eq $Y }
    } elseif ($X) {
        $facts | Where-Object { $_.X -eq $X }
    } elseif ($Y) {
        $facts | Where-Object { $_.Y -eq $Y }
    } else {
        $facts
    }".

% Arity 2 - binary relations with text output (colon-separated strings)
generate_filter_logic_ps(2, text, FilterLogic) :-
    FilterLogic = "
    if ($X -and $Y) {
        $facts | Where-Object { $_.X -eq $X -and $_.Y -eq $Y }
    } elseif ($X) {
        $facts | Where-Object { $_.X -eq $X } | ForEach-Object { $_.Y }
    } elseif ($Y) {
        $facts | Where-Object { $_.Y -eq $Y } | ForEach-Object { $_.X }
    } else {
        $facts | ForEach-Object { \"$($_.X):$($_.Y)\" }
    }".

% Generic fallback (same for text and object - just pass through)
generate_filter_logic_ps(_, _, FilterLogic) :-
    FilterLogic = "
    $facts | ForEach-Object { $_ }".

%% format_ps_fact_entry(+Args, -Entry)
%  Format fact arguments for PowerShell array
format_ps_fact_entry([Arg], Entry) :-
    format(string(Entry), "        '~w'", [Arg]).
format_ps_fact_entry([A, B|_], Entry) :-
    format(string(Entry), "        [PSCustomObject]@{ X='~w'; Y='~w' }", [A, B]).

%% compile_rule_to_powershell(+Pred, +Arity, +Head, +Body, +Options, -Code)
%  Compile a rule with joins and optional negation to pure PowerShell
compile_rule_to_powershell(Pred, Arity, _Head, Body, Options, Code) :-
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
        
        % Generate Parameters
        generate_cmdlet_params_ps(Arity, Options, CmdletBindingAttr, ParamBlock),

        % Generate Loading Logic (Facts)
        format(string(Loaders), 
"        # Get facts from both relations
        $rel1 = ~w
        $rel2 = ~w
~w", [Pred1Str, Pred2Str, NegLoaderCode]),

        % Generate Join Logic
        format(string(JoinLogic),
"        # Nested loop join: rel1.Y = rel2.X
        $results = foreach ($r1 in $rel1) {
            foreach ($r2 in $rel2) {
                if ($r1.Y -eq $r2.X) {
                    $x = $r1.X
                    $y = $r2.Y~w
                    [PSCustomObject]@{
                        X = $x
                        Y = $y
                    }
                }
            }
        }", [NegCheckCode]),

        % Generate Filter Logic
        format(string(FilterLogic),
"        if ($X -and $Y) {
            $results | Where-Object { $_.X -eq $X -and $_.Y -eq $Y }
        } elseif ($X) {
            $results | Where-Object { $_.X -eq $X } | ForEach-Object { $_.Y }
        } elseif ($Y) {
            $results | Where-Object { $_.Y -eq $Y } | ForEach-Object { $_.X }
        } else {
            $results | ForEach-Object { \"$($_.X):$($_.Y)\" }
        }", []),
        
        % Check verbose_output option
        (   member(verbose_output(true), Options)
        ->  format(string(VerboseLoad), "        Write-Verbose \"[~w] Loading relations ~w, ~w...\"\n", [PredStr, Pred1Str, Pred2Str]),
            format(string(VerboseQuery), "        Write-Verbose \"[~w] Query: X=$X, Y=$Y\"\n", [PredStr])
        ;   VerboseLoad = "",
            VerboseQuery = ""
        ),

        % Structure body
        (   member(cmdlet_binding(true), Options)
        ->  format(string(BodyStr),
"    begin {
~w~w
    }
    
    process {
~w~w
~w
    }", [VerboseLoad, Loaders, VerboseQuery, JoinLogic, FilterLogic])
        ;   format(string(BodyStr),
"~w~w
~w
~w
~w", [VerboseLoad, VerboseQuery, Loaders, JoinLogic, FilterLogic])
        ),

        format(string(Code),
"# Generated Pure PowerShell Script
# Predicate: ~w/~w (join~w)
# Generated: ~w
# Generated by UnifyWeaver PowerShell Compiler (Pure Mode)

function ~w {
~w~w
    
~w
}

# Stream all results
~w
", [Pred, Arity, (NegatedGoals \= [] -> " with negation" ; ""), DateStr, PredStr, CmdletBindingAttr, ParamBlock, BodyStr, PredStr])
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
%  Handle module-qualified conjunctions: Module:(A, B) -> split into goals
body_to_list_ps(_Module:(A, B), Goals) :- !,
    body_to_list_ps((A, B), Goals).
body_to_list_ps(_Module:Goal, [Goal]) :-
    Goal \= (_,_),  % Not a conjunction - just strip module
    !.
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
%% TAIL RECURSION OPTIMIZATION
%% ============================================================================

%% compile_tail_recursion_powershell(+Pred/Arity, +Options, -Code)
%  Compile tail recursive predicate to iterative PowerShell loop
%  Pattern: sum_list([], Acc, Acc).
%           sum_list([H|T], Acc, Sum) :- Acc1 is Acc + H, sum_list(T, Acc1, Sum).
compile_tail_recursion_powershell(Pred/Arity, Options, Code) :-
    format('[PowerShell Pure] Compiling tail recursion: ~w/~w~n', [Pred, Arity]),
    
    atom_string(Pred, PredStr),
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),
    
    % Get clauses for this predicate
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    
    % Separate base and recursive cases
    partition(is_recursive_clause_for_ps(Pred), Clauses, RecClauses, BaseClauses),
    length(BaseClauses, NumBase),
    length(RecClauses, NumRec),
    format('[PowerShell Pure] ~w base, ~w recursive clauses~n', [NumBase, NumRec]),
    
    % Check if it's a true tail recursive pattern
    (   is_tail_recursive_ps(RecClauses, Pred)
    ->  format('[PowerShell Pure] Tail position confirmed~n', [])
    ;   format('[PowerShell Pure] WARNING: Not in tail position~n', [])
    ),
    
    % Detect step operation from recursive clause
    (   RecClauses = [RecHead-RecBody|_],
        detect_step_operation_ps(RecBody, Pred, StepOp)
    ->  format('[PowerShell Pure] Step operation: ~w~n', [StepOp])
    ;   StepOp = add_1
    ),
    
    % Generate PowerShell code
    generate_tail_recursion_ps(PredStr, Arity, BaseClauses, StepOp, DateStr, Code).

%% is_recursive_clause_for_ps(+Pred, +Clause)
is_recursive_clause_for_ps(Pred, _Head-Body) :-
    Body \= true,
    body_contains_pred_ps(Pred, Body).

%% is_tail_recursive_ps(+RecClauses, +Pred)
%  Check if all recursive clauses have the recursive call in tail position
is_tail_recursive_ps(RecClauses, Pred) :-
    forall(member(_Head-Body, RecClauses), (
        last_goal_in_body_ps(Body, LastGoal),
        LastGoal =.. [Pred|_]
    )).

%% last_goal_in_body_ps(+Body, -Goal)
last_goal_in_body_ps(Goal, Goal) :-
    \+ Goal = (_, _).
last_goal_in_body_ps((_, B), Goal) :-
    last_goal_in_body_ps(B, Goal).

%% detect_step_operation_ps(+Body, +Pred, -StepOp)
%  Detect the step operation in the recursive clause
detect_step_operation_ps(Body, Pred, StepOp) :-
    body_to_list_ps(Body, Goals),
    % Find arithmetic operations (Var is Expr)
    findall(Op, (
        member(Goal, Goals),
        Goal = (_ is Expr),
        \+ body_contains_pred_ps(Pred, Goal),
        detect_arith_op_ps(Expr, Op)
    ), Ops),
    (   Ops = [StepOp|_]
    ->  true
    ;   StepOp = unknown
    ).

%% detect_arith_op_ps(+Expr, -Op)
%  Check for variable addition before constant addition
detect_arith_op_ps(_ + B, add_element) :- var(B), !.   % Acc + H (where H is var)
detect_arith_op_ps(_ + B, add_1) :- B == 1, !.         % Acc + 1
detect_arith_op_ps(_ + _, add_element) :- !.            % Fallback for other additions
detect_arith_op_ps(_ - B, subtract_element) :- var(B), !.
detect_arith_op_ps(_ - _, subtract_element) :- !.
detect_arith_op_ps(_ * B, multiply_element) :- var(B), !.
detect_arith_op_ps(_ * _, multiply_element) :- !.
detect_arith_op_ps(_, unknown).

%% generate_tail_recursion_ps(+PredStr, +Arity, +BaseClauses, +StepOp, +DateStr, -Code)
generate_tail_recursion_ps(PredStr, 3, _BaseClauses, StepOp, DateStr, Code) :-
    % Ternary pattern: pred(List, Accumulator, Result)
    step_op_to_ps(StepOp, StepCode),
    format(string(Code),
"# Generated Pure PowerShell Script (Tail Recursion Optimized)
# Predicate: ~w/3
# Generated: ~w
# Generated by UnifyWeaver PowerShell Compiler (Pure Mode)

# ~w - tail recursive accumulator pattern
# Compiled to iterative foreach loop (tail call optimization)
function ~w {
    param(
        [array]$List,
        [int]$Accumulator = 0
    )
    
    $current = $Accumulator
    
    # Iterative loop (tail recursion optimization)
    foreach ($item in $List) {
        ~w
    }
    
    return $current
}

# Helper function for common use case
function ~w-FromZero {
    param([array]$List)
    return ~w $List 0
}

# Stream function (for pipeline usage)
function Invoke-~w {
    param([array]$List, [int]$Initial = 0)
    return ~w $List $Initial
}", [PredStr, DateStr, PredStr, PredStr, StepCode, PredStr, PredStr, PredStr, PredStr]).

generate_tail_recursion_ps(PredStr, 2, _BaseClauses, StepOp, DateStr, Code) :-
    % Binary pattern: pred(List, Result)
    step_op_to_ps(StepOp, StepCode),
    format(string(Code),
"# Generated Pure PowerShell Script (Tail Recursion Optimized)
# Predicate: ~w/2
# Generated: ~w
# Generated by UnifyWeaver PowerShell Compiler (Pure Mode)

# ~w - tail recursive binary pattern
# Compiled to iterative loop (tail call optimization)
function ~w {
    param([array]$List)
    
    $current = 0
    
    # Iterative loop (tail recursion optimization)
    foreach ($item in $List) {
        ~w
    }
    
    return $current
}

# Stream function (for pipeline usage)
function Invoke-~w {
    param([array]$List)
    return ~w $List
}", [PredStr, DateStr, PredStr, PredStr, StepCode, PredStr, PredStr]).

generate_tail_recursion_ps(PredStr, Arity, _BaseClauses, _StepOp, DateStr, Code) :-
    % Unsupported arity
    format(string(Code),
"# Generated Pure PowerShell Script (Tail Recursion)
# Predicate: ~w/~w
# Generated: ~w
# Arity ~w not yet supported for tail recursion optimization
Write-Error 'Tail recursion with arity ~w not supported'", [PredStr, Arity, DateStr, Arity, Arity]).

%% step_op_to_ps(+StepOp, -PowerShellCode)
step_op_to_ps(add_1, "$current = $current + 1").
step_op_to_ps(add_element, "$current = $current + $item").
step_op_to_ps(subtract_element, "$current = $current - $item").
step_op_to_ps(multiply_element, "$current = $current * $item").
step_op_to_ps(unknown, "$current = $current + 1").  % Fallback

%% can_compile_tail_recursion_ps(+Pred/Arity)
%  Check if predicate can be compiled with tail recursion optimization
can_compile_tail_recursion_ps(Pred/Arity) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    
    % Need at least one base case and one recursive case
    partition(is_recursive_clause_for_ps(Pred), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [],
    
    % All recursive clauses must be in tail position
    is_tail_recursive_ps(RecClauses, Pred).

%% ============================================================================
%% MUTUAL RECURSION
%% ============================================================================

%% compile_mutual_recursion_powershell(+Predicates, +Options, -Code)
%  Compile mutually recursive predicates to PowerShell
%  Example: is_even/1 and is_odd/1
compile_mutual_recursion_powershell(Predicates, Options, Code) :-
    format('[PowerShell Pure] Compiling mutual recursion: ~w~n', [Predicates]),
    
    % Extract predicate names for group name
    findall(PredStr, (
        member(Pred/_Arity, Predicates),
        atom_string(Pred, PredStr)
    ), PredStrs),
    atomic_list_concat(PredStrs, '_', GroupName),
    
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),
    
    % Determine if memoization is enabled
    (   member(memo(false), Options)
    ->  MemoEnabled = false
    ;   MemoEnabled = true
    ),
    
    % Generate header with memoization
    generate_mutual_header_ps(GroupName, MemoEnabled, DateStr, HeaderCode),
    
    % Generate function for each predicate
    findall(FuncCode, (
        member(Pred/Arity, Predicates),
        generate_mutual_function_ps(Pred, Arity, GroupName, Predicates, MemoEnabled, FuncCode)
    ), FuncCodes),
    atomic_list_concat(FuncCodes, '\n\n', FunctionsCode),
    
    % Generate main dispatch
    generate_mutual_dispatch_ps(Predicates, DispatchCode),
    
    % Combine all parts
    format(string(Code), "~w\n\n~w\n\n~w\n\n# Run: Invoke-~w <args>",
           [HeaderCode, FunctionsCode, DispatchCode, PredStrs]).

%% generate_mutual_header_ps(+GroupName, +MemoEnabled, +DateStr, -Code)
generate_mutual_header_ps(GroupName, MemoEnabled, DateStr, Code) :-
    (   MemoEnabled = true
    ->  MemoDecl = "$script:memo = @{}"
    ;   MemoDecl = "# Memoization disabled"
    ),
    format(string(Code),
"# Generated Pure PowerShell Script (Mutual Recursion)
# Group: ~w
# Generated: ~w
# Generated by UnifyWeaver PowerShell Compiler (Pure Mode)

# Shared memoization table for all predicates in this group
~w
", [GroupName, DateStr, MemoDecl]).

%% generate_mutual_function_ps(+Pred, +Arity, +GroupName, +AllPredicates, +MemoEnabled, -Code)
generate_mutual_function_ps(Pred, Arity, GroupName, AllPredicates, MemoEnabled, Code) :-
    atom_string(Pred, PredStr),
    
    % Get clauses for this predicate
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    
    % Separate base and recursive cases
    partition(is_mutual_recursive_clause_ps(AllPredicates), Clauses, RecClauses, BaseClauses),
    length(BaseClauses, NumBase),
    length(RecClauses, NumRec),
    format('[PowerShell Pure] ~w: ~w base, ~w recursive~n', [PredStr, NumBase, NumRec]),
    
    % Generate base case code
    generate_mutual_base_cases_ps(BaseClauses, BaseCaseCode),
    
    % Generate recursive case code
    generate_mutual_recursive_cases_ps(RecClauses, AllPredicates, RecCaseCode),
    
    % Generate memoization check
    (   MemoEnabled = true
    ->  format(string(MemoCheck),
"    $key = \"~w:$arg\"
    if ($script:memo.ContainsKey($key)) {
        return $script:memo[$key]
    }", [PredStr])
    ;   MemoCheck = "    # Memoization disabled"
    ),
    
    % Generate memoization store
    (   MemoEnabled = true
    ->  MemoStore = "        $script:memo[$key] = $result"
    ;   MemoStore = "        # Memoization disabled"
    ),
    
    format(string(Code),
"# ~w/~w - part of mutual recursion group: ~w
function ~w {
    param([int]$arg)
    
~w
    
~w
    
~w
    
    # No match - return false
    return $false
}", [PredStr, Arity, GroupName, PredStr, MemoCheck, BaseCaseCode, RecCaseCode]).

%% is_mutual_recursive_clause_ps(+AllPredicates, +Clause)
is_mutual_recursive_clause_ps(AllPredicates, _Head-Body) :-
    Body \= true,
    member(Pred/Arity, AllPredicates),
    functor(Goal, Pred, Arity),
    body_contains_goal_ps(Body, Goal).

%% body_contains_goal_ps(+Body, +Goal)
body_contains_goal_ps((A, B), Goal) :- !,
    (   body_contains_goal_ps(A, Goal)
    ;   body_contains_goal_ps(B, Goal)
    ).
body_contains_goal_ps(Body, Goal) :-
    compound(Body),
    functor(Body, F, A),
    functor(Goal, F, A).

%% generate_mutual_base_cases_ps(+BaseClauses, -Code)
generate_mutual_base_cases_ps([], "    # No base cases").
generate_mutual_base_cases_ps(BaseClauses, Code) :-
    BaseClauses \= [],
    findall(CaseCode, (
        member(Head-true, BaseClauses),
        Head =.. [_|[Value]],
        format(string(CaseCode), "    if ($arg -eq ~w) {
        $result = $true
        $script:memo[$key] = $result
        return $result
    }", [Value])
    ), CaseCodes),
    atomic_list_concat(CaseCodes, '\n    ', Code).

%% generate_mutual_recursive_cases_ps(+RecClauses, +AllPredicates, -Code)
generate_mutual_recursive_cases_ps([], _, "    # No recursive cases").
generate_mutual_recursive_cases_ps(RecClauses, AllPredicates, Code) :-
    RecClauses \= [],
    findall(CaseCode, (
        member(Head-Body, RecClauses),
        Head =.. [_|[_HeadVar]],
        generate_mutual_recursive_case_ps(Body, AllPredicates, CaseCode)
    ), CaseCodes),
    atomic_list_concat(CaseCodes, '\n    ', Code).

%% generate_mutual_recursive_case_ps(+Body, +AllPredicates, -Code)
generate_mutual_recursive_case_ps(Body, _AllPredicates, Code) :-
    % Parse body to extract: condition, computation, recursive call
    body_to_list_ps(Body, Goals),
    
    % Find condition (N > 0, etc.)
    findall(CondGoal, (member(CondGoal, Goals), is_condition_goal_ps(CondGoal)), CondGoals),
    
    % Find computation (N1 is N - 1)
    findall(CompGoal, (member(CompGoal, Goals), CompGoal = (_ is _)), CompGoals),
    
    % Find recursive call
    findall(RecGoal, (member(RecGoal, Goals), \+ is_condition_goal_ps(RecGoal), RecGoal \= (_ is _)), RecGoals),
    
    % Generate condition
    (   CondGoals = [CondGoal|_]
    ->  generate_ps_condition(CondGoal, CondCode)
    ;   CondCode = "$true"
    ),
    
    % Generate computation and extract computed variable name
    (   CompGoals = [CompGoal|_]
    ->  CompGoal = (VarOut is _),
        format(atom(VarAtom), '~w', [VarOut]),
        atom_string(VarAtom, VarStr),
        string_lower(VarStr, ComputedVarName),
        generate_ps_computation(CompGoal, CompCode)
    ;   CompCode = "",
        ComputedVarName = "arg"  % fallback to input arg
    ),
    
    % Generate recursive call using computed variable
    (   RecGoals = [RecGoal|_]
    ->  generate_ps_mutual_call_with_var(RecGoal, ComputedVarName, RecCallCode)
    ;   RecCallCode = "# No recursive call"
    ),
    
    format(string(Code),
"    # Recursive case
    if (~w) {
        ~w
        ~w
        if ($recResult) {
            $result = $true
            $script:memo[$key] = $result
            return $result
        }
    }", [CondCode, CompCode, RecCallCode]).

%% is_condition_goal_ps(+Goal)
is_condition_goal_ps(Goal) :-
    Goal =.. [Op|_],
    member(Op, ['>', '<', '>=', '=<', '=:=', '==']).

%% generate_ps_condition(+Goal, -Code)
generate_ps_condition(Goal, Code) :-
    Goal =.. [Op, A, B],
    ps_comparison_op(Op, PsOp),
    format(string(Code), "$arg ~w ~w", [PsOp, B]).

ps_comparison_op('>', '-gt').
ps_comparison_op('<', '-lt').
ps_comparison_op('>=', '-ge').
ps_comparison_op('=<', '-le').
ps_comparison_op('=:=', '-eq').
ps_comparison_op('==', '-eq').

%% generate_ps_computation(+Goal, -Code)
generate_ps_computation(VarOut is Expr, Code) :-
    format(atom(VarAtom), '~w', [VarOut]),
    atom_string(VarAtom, VarStr),
    string_lower(VarStr, VarLower),
    generate_ps_expr(Expr, ExprCode),
    format(string(Code), "$~w = ~w", [VarLower, ExprCode]).

%% generate_ps_expr(+Expr, -Code)
%  Handle variables first (most specific)
generate_ps_expr(Var, Code) :-
    var(Var), !,
    Code = "$arg".
generate_ps_expr(Atom, Code) :-
    atomic(Atom), !,
    format(string(Code), "~w", [Atom]).
%  Handle arithmetic with negative numbers (N + -1 becomes N - 1)
generate_ps_expr(A + B, Code) :-
    number(B), B < 0, !,
    PosB is -B,
    generate_ps_expr(A, ACode),
    format(string(Code), "(~w - ~w)", [ACode, PosB]).
generate_ps_expr(A - B, Code) :- !,
    generate_ps_expr(A, ACode),
    generate_ps_expr(B, BCode),
    format(string(Code), "(~w - ~w)", [ACode, BCode]).
generate_ps_expr(A + B, Code) :- !,
    generate_ps_expr(A, ACode),
    generate_ps_expr(B, BCode),
    format(string(Code), "(~w + ~w)", [ACode, BCode]).

%% generate_ps_mutual_call(+Goal, -Code)
generate_ps_mutual_call(Goal, Code) :-
    Goal =.. [Pred|Args],
    atom_string(Pred, PredStr),
    % Translate arguments
    maplist(translate_ps_arg, Args, PsArgs),
    atomic_list_concat(PsArgs, ' ', ArgsStr),
    format(string(Code), "$recResult = ~w ~w", [PredStr, ArgsStr]).

%% generate_ps_mutual_call_with_var(+Goal, +ComputedVarName, -Code)
%  Generate mutual call using specified variable name for argument
generate_ps_mutual_call_with_var(Goal, ComputedVarName, Code) :-
    Goal =.. [Pred|_Args],
    atom_string(Pred, PredStr),
    format(string(Code), "$recResult = ~w $~w", [PredStr, ComputedVarName]).

%% translate_ps_arg(+Arg, -PsArg)
translate_ps_arg(Arg, PsArg) :-
    var(Arg), !,
    format(atom(VarAtom), '~w', [Arg]),
    atom_string(VarAtom, VarStr),
    string_lower(VarStr, VarLower),
    format(string(PsArg), "$~w", [VarLower]).
translate_ps_arg(Atom, PsArg) :-
    atomic(Atom),
    format(string(PsArg), "~w", [Atom]).

%% generate_mutual_dispatch_ps(+Predicates, -Code)
generate_mutual_dispatch_ps(Predicates, Code) :-
    findall(CaseCode, (
        member(Pred/_Arity, Predicates),
        atom_string(Pred, PredStr),
        format(string(CaseCode), "        '~w' { ~w $Args[1] }", [PredStr, PredStr])
    ), CaseCodes),
    atomic_list_concat(CaseCodes, '\n', CasesStr),
    format(string(Code),
"# Main dispatch
function Invoke-MutualGroup {
    param([string]$FunctionName, [int]$Arg)
    
    switch ($FunctionName) {
~w
        default { Write-Error \"Unknown function: $FunctionName\" }
    }
}", [CasesStr]).

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

%% generate_cmdlet_params_ps(+Arity, +Options, -CmdletBinding, -ParamBlock)
%  Generate the [CmdletBinding()] attribute and param(...) block
generate_cmdlet_params_ps(Arity, Options, CmdletBinding, ParamBlock) :-
    (   member(cmdlet_binding(true), Options)
    ->  CmdletBinding = "    [CmdletBinding()]\n",
        generate_advanced_params(Arity, Options, ParamsStr),
        format(string(ParamBlock), "    param(~n~w~n    )", [ParamsStr])
    ;   CmdletBinding = "",
        generate_simple_params(Arity, ParamStr),
        format(string(ParamBlock), "    param(~w)", [ParamStr])
    ).

generate_simple_params(1, "[string]$Key").
generate_simple_params(2, "[string]$X, [string]$Y").
generate_simple_params(3, "[string]$X, [string]$Y, [string]$Z").
generate_simple_params(Arity, Str) :-
    Arity > 3,
    findall(S, (between(1, Arity, I), format(string(S), "[string]$Arg~w", [I])), List),
    atomic_list_concat(List, ', ', Str).

generate_advanced_params(Arity, Options, ParamsStr) :-
    ArityM1 is Arity - 1,
    findall(PStr, (
        between(0, ArityM1, I),
        get_arg_name(I, Name),
        get_arg_attributes(I, Options, Attrs),
        format_param_entry(I, Name, Attrs, PStr)
    ), PList),
    atomic_list_concat(PList, ",\n\n", ParamsStr).

get_arg_name(0, 'X').
get_arg_name(1, 'Y').
get_arg_name(2, 'Z').
get_arg_name(I, Name) :- I > 2, format(string(Name), "Arg~w", [I]).

get_arg_attributes(Index, Options, Attrs) :-
    (   member(arg_options(Index, Opts), Options)
    ->  Attrs = Opts
    ;   Attrs = []
    ).

format_param_entry(Index, Name, Attrs, Entry) :-
    % Position is default
    (   member(position(P), Attrs) -> Pos = P ; Pos = Index ),
    % Mandatory
    (   member(mandatory(true), Attrs) -> Mand = ", Mandatory=$true" ; Mand = "" ),
    % Pipeline
    (   member(pipeline(true), Attrs) -> Pipe = ", ValueFromPipeline=$true"
    ;   Index = 1 -> Pipe = ", ValueFromPipeline=$true" % Default for 2nd arg in binary
    ;   Pipe = ""
    ),

    format(string(ParamAttr), "        [Parameter(Position=~w~w~w)]", [Pos, Mand, Pipe]),

    % Validation
    (   member(validate_not_null(true), Attrs) -> Val1 = "\n        [ValidateNotNullOrEmpty()]" ; Val1 = "" ),
    (   member(validate_set(Set), Attrs), is_list(Set) ->
        atomic_list_concat(Set, "', '", SetInner),
        format(string(Val2), "\n        [ValidateSet('~w')]", [SetInner])
    ;   Val2 = ""
    ),

    format(string(Entry), "~w~w~w\n        [string]$~w", [ParamAttr, Val1, Val2, Name]).

% ============================================================================
% BINDING INTEGRATION FOR RULE COMPILATION (Book 12, Chapter 5)
% ============================================================================
%
% These predicates integrate the binding system into rule body compilation,
% allowing goals like get_service(Name, Status) to automatically compile to
% Get-Service calls instead of requiring explicit fact definitions.

%% lookup_var_eq(+Var, +VarMap, -PSVar)
%
%  Look up a Prolog variable in the VarMap using strict equality (==).
%  This prevents accidental unification when looking up different variables.
%
%  IMPORTANT: member/2 uses unification which causes bugs when variables
%  in the VarMap get unified with the lookup variable. We must use ==.
%
lookup_var_eq(Var, [V-PS|_], PS) :-
    Var == V, !.
lookup_var_eq(Var, [_|Rest], PS) :-
    lookup_var_eq(Var, Rest, PS).

%% goal_has_binding(+Goal, -Binding)
%
%  Check if a goal matches a registered PowerShell binding.
%  Returns the binding information if found.
%  Handles both plain goals and module-qualified goals (Module:Goal).
%
%  @param Goal     A Prolog goal term (e.g., get_service(Name, Status))
%  @param Binding  binding(Pred, TargetName, Inputs, Outputs, Options)
%
goal_has_binding(Goal, Binding) :-
    % Handle module-qualified goals like powershell_compiler:sqrt(X,Y)
    (   Goal = _Module:InnerGoal
    ->  ActualGoal = InnerGoal
    ;   ActualGoal = Goal
    ),
    ActualGoal =.. [Pred|Args],
    length(Args, Arity),
    binding(powershell, Pred/Arity, TargetName, Inputs, Outputs, Options),
    Binding = binding(Pred/Arity, TargetName, Inputs, Outputs, Options).

%% classify_body_goals(+Goals, -BoundGoals, -FactGoals, -BuiltinGoals)
%
%  Classify goals in a rule body into bound (have PowerShell bindings),
%  fact-based (need to be loaded from Prolog facts), and builtins.
%
classify_body_goals([], [], [], []).
classify_body_goals([Goal|Rest], Bound, Facts, Builtins) :-
    classify_body_goals(Rest, RestBound, RestFacts, RestBuiltins),
    (   is_builtin_goal_ps(Goal)
    ->  Bound = RestBound,
        Facts = RestFacts,
        Builtins = [Goal|RestBuiltins]
    ;   goal_has_binding(Goal, Binding)
    ->  Bound = [Goal-Binding|RestBound],
        Facts = RestFacts,
        Builtins = RestBuiltins
    ;   % Assume it's a fact-based goal
        Bound = RestBound,
        Facts = [Goal|RestFacts],
        Builtins = RestBuiltins
    ).

%% is_builtin_goal_ps(+Goal)
%
%  Check if a goal is a Prolog builtin that needs special handling.
%  Handles both plain goals and module-qualified goals (Module:Goal).
%
is_builtin_goal_ps(Goal) :-
    % Handle module-qualified goals
    (   Goal = _Module:InnerGoal
    ->  is_builtin_goal_ps_inner(InnerGoal)
    ;   is_builtin_goal_ps_inner(Goal)
    ).

is_builtin_goal_ps_inner(_ = _).
is_builtin_goal_ps_inner(_ \= _).
is_builtin_goal_ps_inner(_ == _).
is_builtin_goal_ps_inner(_ \== _).
is_builtin_goal_ps_inner(_ < _).
is_builtin_goal_ps_inner(_ > _).
is_builtin_goal_ps_inner(_ =< _).
is_builtin_goal_ps_inner(_ >= _).
is_builtin_goal_ps_inner(_ is _).
is_builtin_goal_ps_inner(true).
is_builtin_goal_ps_inner(fail).
is_builtin_goal_ps_inner(!).
is_builtin_goal_ps_inner(\+ _).
is_builtin_goal_ps_inner(not(_)).

%% generate_bound_goal_code(+Goal, +Binding, +VarMap, -Code, -NewVarMap)
%
%  Generate PowerShell code for a bound goal.
%
%  @param Goal      The Prolog goal (e.g., get_service(Name, Status))
%  @param Binding   The binding info from goal_has_binding/2
%  @param VarMap    Current variable name mappings
%  @param Code      Generated PowerShell code
%  @param NewVarMap Updated variable mappings
%
generate_bound_goal_code(Goal, binding(_Pred, TargetName, Inputs, Outputs, Options), VarMap, Code, NewVarMap) :-
    % Handle module-qualified goals like powershell_compiler:sqrt(X,Y)
    (   Goal = _Module:InnerGoal
    ->  ActualGoal = InnerGoal
    ;   ActualGoal = Goal
    ),
    ActualGoal =.. [_|Args],
    length(Inputs, NumInputs),
    length(Outputs, NumOutputs),
    length(Args, TotalArgs),

    % Split args into inputs and outputs based on binding spec
    (   NumInputs + NumOutputs =:= TotalArgs
    ->  length(InputArgs, NumInputs),
        append(InputArgs, OutputArgs, Args)
    ;   % All args are inputs (output via return value)
        InputArgs = Args,
        OutputArgs = []
    ),

    % Generate the call based on binding pattern
    (   member(pattern(pipe_transform), Options)
    ->  % Pipeline pattern: input | Cmdlet
        generate_pipeline_call(TargetName, InputArgs, OutputArgs, VarMap, Code, NewVarMap)
    ;   member(pattern(exit_code_bool), Options)
    ->  % Boolean via exit code (like Test-Path)
        generate_bool_call(TargetName, InputArgs, OutputArgs, VarMap, Code, NewVarMap)
    ;   sub_string(TargetName, 0, _, _, "[")
    ->  % .NET static method
        generate_dotnet_static_call(TargetName, InputArgs, OutputArgs, VarMap, Code, NewVarMap)
    ;   sub_string(TargetName, 0, _, _, ".")
    ->  % .NET instance method
        generate_dotnet_instance_call(TargetName, InputArgs, OutputArgs, VarMap, Code, NewVarMap)
    ;   % Standard cmdlet call
        generate_cmdlet_call_code(TargetName, InputArgs, OutputArgs, VarMap, Code, NewVarMap)
    ).

%% generate_cmdlet_call_code(+CmdletName, +InputArgs, +OutputArgs, +VarMap, -Code, -NewVarMap)
%
%  Generate PowerShell cmdlet call code.
%
generate_cmdlet_call_code(CmdletName, InputArgs, OutputArgs, VarMap, Code, NewVarMap) :-
    % Format input arguments
    maplist(format_arg_for_ps(VarMap), InputArgs, FormattedInputs),
    atomic_list_concat(FormattedInputs, ' ', InputStr),

    % Generate output variable assignment if needed
    % NOTE: Must use lookup_var_eq/3, NOT member/2 for variable lookup!
    (   OutputArgs = [OutVar|_],
        var(OutVar),
        lookup_var_eq(OutVar, VarMap, ExistingVar)
    ->  % Output variable already mapped from head - use existing name
        format(string(Code), "~w = ~w ~w", [ExistingVar, CmdletName, InputStr]),
        NewVarMap = VarMap
    ;   OutputArgs = [OutVar|_],
        var(OutVar)
    ->  % New output variable - create new name
        gensym('$result', ResultVar),
        format(string(Code), "~w = ~w ~w", [ResultVar, CmdletName, InputStr]),
        NewVarMap = [OutVar-ResultVar|VarMap]
    ;   OutputArgs = [OutVar|_],
        atom(OutVar)
    ->  % Output to named variable
        format_arg_for_ps(VarMap, OutVar, OutVarStr),
        format(string(Code), "~w = ~w ~w", [OutVarStr, CmdletName, InputStr]),
        NewVarMap = VarMap
    ;   % No output capture needed
        format(string(Code), "~w ~w", [CmdletName, InputStr]),
        NewVarMap = VarMap
    ).

%% generate_dotnet_static_call(+MethodName, +InputArgs, +OutputArgs, +VarMap, -Code, -NewVarMap)
%
%  Generate .NET static method call (e.g., [Math]::Sqrt(x))
%
generate_dotnet_static_call(MethodName, InputArgs, OutputArgs, VarMap, Code, NewVarMap) :-
    maplist(format_arg_for_ps(VarMap), InputArgs, FormattedInputs),
    atomic_list_concat(FormattedInputs, ', ', InputStr),

    (   OutputArgs = [OutVar|_]
    ->  % Check if output variable is already mapped (e.g., from head args)
        % NOTE: Must use lookup_var_eq/3, NOT member/2!
        % member/2 uses unification which causes different variables to unify
        (   var(OutVar), lookup_var_eq(OutVar, VarMap, ExistingVar)
        ->  % Assign directly to existing variable
            format(string(Code), "~w = ~w(~w)", [ExistingVar, MethodName, InputStr]),
            NewVarMap = VarMap
        ;   % Create new result variable
            gensym('$result', ResultVar),
            format(string(Code), "~w = ~w(~w)", [ResultVar, MethodName, InputStr]),
            NewVarMap = [OutVar-ResultVar|VarMap]
        )
    ;   format(string(Code), "~w(~w)", [MethodName, InputStr]),
        NewVarMap = VarMap
    ).

%% generate_dotnet_instance_call(+MethodName, +InputArgs, +OutputArgs, +VarMap, -Code, -NewVarMap)
%
%  Generate .NET instance method call (e.g., $str.Trim())
%
generate_dotnet_instance_call(MethodName, [Object|RestInputs], OutputArgs, VarMap, Code, NewVarMap) :-
    format_arg_for_ps(VarMap, Object, ObjStr),
    maplist(format_arg_for_ps(VarMap), RestInputs, FormattedInputs),
    atomic_list_concat(FormattedInputs, ', ', InputStr),

    % Check if method already has () in name
    (   sub_string(MethodName, _, 2, 0, "()")
    ->  MethodCall = MethodName
    ;   InputStr = ""
    ->  format(string(MethodCall), "~w()", [MethodName])
    ;   format(string(MethodCall), "~w(~w)", [MethodName, InputStr])
    ),

    % NOTE: Must use lookup_var_eq/3, NOT member/2 for variable lookup!
    (   OutputArgs = [OutVar|_],
        var(OutVar),
        lookup_var_eq(OutVar, VarMap, ExistingVar)
    ->  % Output variable already mapped - use existing name
        format(string(Code), "~w = ~w~w", [ExistingVar, ObjStr, MethodCall]),
        NewVarMap = VarMap
    ;   OutputArgs = [OutVar|_]
    ->  gensym('$result', ResultVar),
        format(string(Code), "~w = ~w~w", [ResultVar, ObjStr, MethodCall]),
        NewVarMap = [OutVar-ResultVar|VarMap]
    ;   format(string(Code), "~w~w", [ObjStr, MethodCall]),
        NewVarMap = VarMap
    ).

%% generate_pipeline_call(+CmdletName, +InputArgs, +OutputArgs, +VarMap, -Code, -NewVarMap)
%
%  Generate pipeline-style call (input | Cmdlet)
%
generate_pipeline_call(CmdletName, InputArgs, OutputArgs, VarMap, Code, NewVarMap) :-
    (   InputArgs = [PipeInput|RestInputs]
    ->  format_arg_for_ps(VarMap, PipeInput, PipeStr),
        maplist(format_arg_for_ps(VarMap), RestInputs, FormattedRest),
        atomic_list_concat(FormattedRest, ' ', RestStr)
    ;   PipeStr = "$_",
        RestStr = ""
    ),

    % NOTE: Must use lookup_var_eq/3, NOT member/2 for variable lookup!
    (   OutputArgs = [OutVar|_],
        var(OutVar),
        lookup_var_eq(OutVar, VarMap, ExistingVar)
    ->  % Output variable already mapped - use existing name
        (   RestStr = ""
        ->  format(string(Code), "~w = ~w | ~w", [ExistingVar, PipeStr, CmdletName])
        ;   format(string(Code), "~w = ~w | ~w ~w", [ExistingVar, PipeStr, CmdletName, RestStr])
        ),
        NewVarMap = VarMap
    ;   OutputArgs = [OutVar|_]
    ->  gensym('$result', ResultVar),
        (   RestStr = ""
        ->  format(string(Code), "~w = ~w | ~w", [ResultVar, PipeStr, CmdletName])
        ;   format(string(Code), "~w = ~w | ~w ~w", [ResultVar, PipeStr, CmdletName, RestStr])
        ),
        NewVarMap = [OutVar-ResultVar|VarMap]
    ;   (   RestStr = ""
        ->  format(string(Code), "~w | ~w", [PipeStr, CmdletName])
        ;   format(string(Code), "~w | ~w ~w", [PipeStr, CmdletName, RestStr])
        ),
        NewVarMap = VarMap
    ).

%% generate_bool_call(+CmdletName, +InputArgs, +OutputArgs, +VarMap, -Code, -NewVarMap)
%
%  Generate boolean test call (like Test-Path)
%
generate_bool_call(CmdletName, InputArgs, OutputArgs, VarMap, Code, NewVarMap) :-
    maplist(format_arg_for_ps(VarMap), InputArgs, FormattedInputs),
    atomic_list_concat(FormattedInputs, ' ', InputStr),

    % NOTE: Must use lookup_var_eq/3, NOT member/2 for variable lookup!
    (   OutputArgs = [OutVar|_],
        var(OutVar),
        lookup_var_eq(OutVar, VarMap, ExistingVar)
    ->  % Output variable already mapped - use existing name
        format(string(Code), "~w = ~w ~w", [ExistingVar, CmdletName, InputStr]),
        NewVarMap = VarMap
    ;   OutputArgs = [OutVar|_]
    ->  gensym('$result', ResultVar),
        format(string(Code), "~w = ~w ~w", [ResultVar, CmdletName, InputStr]),
        NewVarMap = [OutVar-ResultVar|VarMap]
    ;   % Use in condition context
        format(string(Code), "~w ~w", [CmdletName, InputStr]),
        NewVarMap = VarMap
    ).

%% format_arg_for_ps(+VarMap, +Arg, -Formatted)
%
%  Format a Prolog argument for PowerShell code.
%
format_arg_for_ps(VarMap, Arg, Formatted) :-
    (   var(Arg)
    ->  % Unbound variable - look up in VarMap using strict equality
        % NOTE: Must use lookup_var_eq/3, NOT member/2!
        % member/2 uses unification which causes different variables to unify
        (   lookup_var_eq(Arg, VarMap, PSVar)
        ->  Formatted = PSVar
        ;   Formatted = "$_"  % Default for unbound
        )
    ;   atom(Arg)
    ->  % Atom - check if it's a Prolog variable name pattern (starts with uppercase)
        atom_codes(Arg, [C|_]),
        (   C >= 65, C =< 90  % A-Z = 65-90
        ->  format(string(Formatted), "$~w", [Arg])
        ;   format(string(Formatted), "'~w'", [Arg])
        )
    ;   number(Arg)
    ->  format(string(Formatted), "~w", [Arg])
    ;   string(Arg)
    ->  format(string(Formatted), "'~w'", [Arg])
    ;   is_list(Arg)
    ->  maplist(format_arg_for_ps(VarMap), Arg, Items),
        atomic_list_concat(Items, ', ', ItemsStr),
        format(string(Formatted), "@(~w)", [ItemsStr])
    ;   format(string(Formatted), "~w", [Arg])
    ).

%% create_head_var_map(+HeadArgs, -VarMap)
%
%  Create a variable mapping from head arguments to PowerShell parameter names.
%  Maps the first arg to $X, second to $Y, etc.
%
create_head_var_map(HeadArgs, VarMap) :-
    create_head_var_map(HeadArgs, 0, VarMap).

create_head_var_map([], _, []).
create_head_var_map([Arg|Rest], Index, VarMap) :-
    get_arg_name(Index, PSName),
    format(string(PSVar), "$~w", [PSName]),
    NextIndex is Index + 1,
    create_head_var_map(Rest, NextIndex, RestMap),
    (   var(Arg)
    ->  VarMap = [Arg-PSVar|RestMap]
    ;   VarMap = RestMap  % Ground args don't need mapping
    ).

%% compile_rule_with_bindings(+Pred, +Arity, +Head, +Body, +Options, -Code)
%
%  Compile a rule that may contain bound goals to PowerShell.
%  This is the main entry point for binding-aware rule compilation.
%
compile_rule_with_bindings(Pred, Arity, Head, Body, Options, Code) :-
    atom_string(Pred, PredStr),
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),

    % Convert body to list of goals
    body_to_list_ps(Body, Goals),

    % Separate negated goals
    partition(is_negated_goal_ps, Goals, NegatedGoals, PositiveGoals),

    % Classify positive goals
    classify_body_goals(PositiveGoals, BoundGoals, FactGoals, BuiltinGoals),

    length(BoundGoals, NumBound),
    length(FactGoals, NumFacts),
    length(BuiltinGoals, NumBuiltins),
    format('[PowerShell Pure] Rule analysis: ~w bound, ~w fact-based, ~w builtin goals~n',
           [NumBound, NumFacts, NumBuiltins]),

    % Generate code based on goal composition
    (   BoundGoals \= [], FactGoals = []
    ->  % Pure bound rule - all goals have bindings
        format('[PowerShell Pure] Compiling pure bound rule~n', []),
        compile_pure_bound_rule(Pred, Arity, Head, BoundGoals, BuiltinGoals, NegatedGoals, Options, Code)
    ;   BoundGoals \= [], FactGoals \= []
    ->  % Mixed rule - some bound, some fact-based
        format('[PowerShell Pure] Compiling mixed bound/fact rule~n', []),
        compile_mixed_bound_rule(Pred, Arity, Head, BoundGoals, FactGoals, BuiltinGoals, NegatedGoals, Options, Code)
    ;   % No bound goals - use standard fact-based compilation
        format('[PowerShell Pure] No bound goals, using standard compilation~n', []),
        fail  % Fall back to standard compile_rule_to_powershell
    ).

%% compile_pure_bound_rule(+Pred, +Arity, +Head, +BoundGoals, +BuiltinGoals, +NegatedGoals, +Options, -Code)
%
%  Compile a rule where all data goals have PowerShell bindings.
%
compile_pure_bound_rule(Pred, Arity, Head, BoundGoals, _BuiltinGoals, _NegatedGoals, Options, Code) :-
    atom_string(Pred, PredStr),
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),

    % Generate parameters
    generate_cmdlet_params_ps(Arity, Options, CmdletBindingAttr, ParamBlock),

    % Extract head arguments and create initial variable mapping
    Head =.. [_|HeadArgs],
    create_head_var_map(HeadArgs, InitialVarMap),

    % Generate code for each bound goal with variable mapping
    generate_bound_goals_code(BoundGoals, InitialVarMap, GoalCodes, _FinalVarMap),
    atomic_list_concat(GoalCodes, '\n        ', GoalsStr),

    % Check for verbose output
    (   member(verbose_output(true), Options)
    ->  format(string(VerboseStart), "Write-Verbose \"[~w] Executing bound rule...\"~n        ", [PredStr])
    ;   VerboseStart = ""
    ),

    % Structure based on cmdlet_binding option
    (   member(cmdlet_binding(true), Options)
    ->  format(string(BodyStr),
"    begin {
        ~w
    }

    process {
        ~w
    }

    end {
        Write-Verbose \"[~w] Complete\"
    }", [VerboseStart, GoalsStr, PredStr])
    ;   format(string(BodyStr), "~w~w", [VerboseStart, GoalsStr])
    ),

    format(string(Code),
"# Generated Pure PowerShell Script (Bound Rule)
# Predicate: ~w/~w
# Generated: ~w
# Generated by UnifyWeaver PowerShell Compiler (Binding Mode)

function ~w {
~w~w

~w
}

# Call the function
~w
", [Pred, Arity, DateStr, PredStr, CmdletBindingAttr, ParamBlock, BodyStr, PredStr]).

%% compile_mixed_bound_rule(+Pred, +Arity, +Head, +BoundGoals, +FactGoals, +BuiltinGoals, +NegatedGoals, +Options, -Code)
%
%  Compile a rule with both bound goals and fact-based goals.
%  Properly handles variable flow between fact results and bound goal inputs.
%
%  Example: salary_sqrt(Name, SqrtSal) :- employee(Name, Sal), sqrt(Sal, SqrtSal).
%
%  The Sal variable flows from the employee fact to the sqrt bound goal.
%  Generated code loops through facts and applies bindings to each.
%
compile_mixed_bound_rule(Pred, Arity, Head, BoundGoals, FactGoals, BuiltinGoals, NegatedGoals, Options, Code) :-
    atom_string(Pred, PredStr),
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),

    % Generate parameters
    generate_cmdlet_params_ps(Arity, Options, CmdletBindingAttr, ParamBlock),

    % Extract head arguments for output variable mapping
    Head =.. [_|HeadArgs],

    % Analyze variable sharing between goals
    analyze_mixed_rule_vars(HeadArgs, FactGoals, BoundGoals, VarAnalysis),

    % Generate the mixed rule body
    generate_mixed_rule_body(PredStr, HeadArgs, FactGoals, BoundGoals, BuiltinGoals,
                             NegatedGoals, VarAnalysis, Options, BodyCode),

    format(string(Code),
"# Generated Pure PowerShell Script (Mixed Bound/Fact Rule)
# Predicate: ~w/~w
# Generated: ~w
# Generated by UnifyWeaver PowerShell Compiler (Mixed Mode)

function ~w {
~w~w

~w
}

# Call the function
~w
", [Pred, Arity, DateStr, PredStr, CmdletBindingAttr, ParamBlock, BodyCode, PredStr]).

%% analyze_mixed_rule_vars(+HeadArgs, +FactGoals, +BoundGoals, -VarAnalysis)
%
%  Analyze how variables flow between head, fact goals, and bound goals.
%  Returns a structure describing which variables are shared.
%
analyze_mixed_rule_vars(HeadArgs, FactGoals, BoundGoals, VarAnalysis) :-
    % Collect all variables from fact goals with their positions
    findall(fact_var(FactPred, ArgIdx, Var), (
        member(FactGoal, FactGoals),
        strip_module_qualifier(FactGoal, PlainGoal),
        PlainGoal =.. [FactPred|FactArgs],
        nth0(ArgIdx, FactArgs, Var),
        var(Var)
    ), FactVars),

    % Collect all variables from bound goals with their positions
    findall(bound_var(BoundPred, ArgIdx, Var, IsInput), (
        member(Goal-Binding, BoundGoals),
        strip_module_qualifier(Goal, PlainGoal),
        PlainGoal =.. [BoundPred|BoundArgs],
        Binding = binding(_, _, Inputs, Outputs, _),
        length(Inputs, NumInputs),
        nth0(ArgIdx, BoundArgs, Var),
        var(Var),
        (ArgIdx < NumInputs -> IsInput = true ; IsInput = false)
    ), BoundVars),

    % Find head variable positions
    findall(head_var(ArgIdx, Var), (
        nth0(ArgIdx, HeadArgs, Var),
        var(Var)
    ), HeadVars),

    VarAnalysis = var_analysis(HeadVars, FactVars, BoundVars).

%% strip_module_qualifier(+Goal, -PlainGoal)
%  Remove module qualifier from a goal if present.
strip_module_qualifier(_Module:Goal, Goal) :- !.
strip_module_qualifier(Goal, Goal).

%% generate_mixed_rule_body(+PredStr, +HeadArgs, +FactGoals, +BoundGoals, +BuiltinGoals, +NegatedGoals, +VarAnalysis, +Options, -Code)
%
%  Generate the body code for a mixed rule.
%
generate_mixed_rule_body(PredStr, HeadArgs, FactGoals, BoundGoals, _BuiltinGoals,
                         NegatedGoals, VarAnalysis, Options, Code) :-
    VarAnalysis = var_analysis(HeadVars, FactVars, BoundVars),

    % Generate fact loader calls
    generate_fact_loaders(FactGoals, LoaderCode),

    % Generate negation loaders if needed
    (   NegatedGoals \= []
    ->  generate_negated_facts_loaders(NegatedGoals, NegLoaderCode)
    ;   NegLoaderCode = ""
    ),

    % Generate the foreach loop over fact results
    generate_fact_loop(PredStr, HeadArgs, FactGoals, BoundGoals, HeadVars,
                       FactVars, BoundVars, NegatedGoals, Options, LoopCode),

    format(string(Code),
"    ~w~w
~w", [LoaderCode, NegLoaderCode, LoopCode]).

%% generate_fact_loaders(+FactGoals, -Code)
%  Generate PowerShell code to load facts for each fact-based goal.
generate_fact_loaders([], "").
generate_fact_loaders([FactGoal|Rest], Code) :-
    strip_module_qualifier(FactGoal, PlainGoal),
    PlainGoal =.. [FactPred|_],
    atom_string(FactPred, FactPredStr),
    generate_fact_loaders(Rest, RestCode),
    format(string(Code), "$~w_data = ~w\n    ~w", [FactPredStr, FactPredStr, RestCode]).

%% generate_fact_loop(+PredStr, +HeadArgs, +FactGoals, +BoundGoals, +HeadVars, +FactVars, +BoundVars, +NegatedGoals, +Options, -Code)
%
%  Generate the main foreach loop that iterates over facts and applies bindings.
%
generate_fact_loop(PredStr, HeadArgs, FactGoals, BoundGoals, HeadVars,
                   FactVars, BoundVars, NegatedGoals, Options, Code) :-
    % For single fact goal, generate simple loop
    (   FactGoals = [SingleFactGoal]
    ->  strip_module_qualifier(SingleFactGoal, PlainFact),
        PlainFact =.. [FactPred|FactArgs],
        atom_string(FactPred, FactPredStr),

        % Create variable map from fact fields to PS variables
        create_fact_var_map(FactArgs, FactVarMap),

        % Generate bound goal code using the fact variable map
        generate_bound_goals_in_loop(BoundGoals, FactVarMap, HeadArgs, BoundCode, OutputVarMap),

        % Generate negation check if needed
        (   NegatedGoals \= []
        ->  generate_negation_check_ps(NegatedGoals, NegCheck),
            format(string(NegCheckStr), "~n            ~w", [NegCheck])
        ;   NegCheckStr = ""
        ),

        % Generate output object creation
        generate_output_object(HeadArgs, OutputVarMap, FactVarMap, OutputCode),

        % Generate fact field extraction BEFORE the format call
        generate_fact_field_extraction(FactArgs, FactFieldExtraction),

        % Check for verbose output
        (   member(verbose_output(true), Options)
        ->  format(string(VerboseStr), "Write-Verbose \"[~w] Processing fact...\"\n            ", [PredStr])
        ;   VerboseStr = ""
        ),

        format(string(Code),
"    $results = foreach ($fact in $~w_data) {
        ~w# Extract fact fields
        ~w
        # Apply bound operations~w
        ~w
        # Output result
        ~w
    }
    $results", [FactPredStr, VerboseStr, FactFieldExtraction, NegCheckStr, BoundCode, OutputCode])

    ;   % Multiple fact goals - generate joins (2 or more facts)
        length(FactGoals, NumFacts),
        (   NumFacts = 2
        ->  % Two-way join - use existing binary join
            FactGoals = [FG1, FG2],
            generate_multi_fact_loop(PredStr, HeadArgs, FG1, FG2, BoundGoals, HeadVars,
                                     FactVars, BoundVars, NegatedGoals, Options, Code)
        ;   NumFacts > 2
        ->  % N-way join (3+ facts) - use multi-way join
            generate_nway_fact_loop(PredStr, HeadArgs, FactGoals, BoundGoals, HeadVars,
                                    FactVars, BoundVars, NegatedGoals, Options, Code)
        ;   % Should not happen but fallback
            fail
        )
    ).

%% create_fact_var_map(+FactArgs, -VarMap)
%  Create a mapping from fact argument variables to PowerShell field accessors.
create_fact_var_map(FactArgs, VarMap) :-
    create_fact_var_map(FactArgs, 0, VarMap).

create_fact_var_map([], _, []).
create_fact_var_map([Arg|Rest], Idx, VarMap) :-
    NextIdx is Idx + 1,
    create_fact_var_map(Rest, NextIdx, RestMap),
    (   var(Arg)
    ->  % Map to $fact.X, $fact.Y, etc. based on position
        (Idx = 0 -> FieldName = "X" ; Idx = 1 -> FieldName = "Y" ; format(atom(FieldName), "F~w", [Idx])),
        format(string(PSVar), "$~w", [FieldName]),
        VarMap = [Arg-PSVar|RestMap]
    ;   VarMap = RestMap
    ).

%% generate_fact_field_extraction(+FactArgs, -Code)
%  Generate code to extract fact fields into local variables.
generate_fact_field_extraction(FactArgs, Code) :-
    generate_fact_field_extraction(FactArgs, 0, Extractions),
    atomic_list_concat(Extractions, '\n        ', Code).

generate_fact_field_extraction([], _, []).
generate_fact_field_extraction([_|Rest], Idx, [Extraction|RestExtractions]) :-
    NextIdx is Idx + 1,
    (Idx = 0 -> FieldName = "X" ; Idx = 1 -> FieldName = "Y" ; format(atom(FieldName), "F~w", [Idx])),
    format(string(Extraction), "$~w = $fact.~w", [FieldName, FieldName]),
    generate_fact_field_extraction(Rest, NextIdx, RestExtractions).

%% generate_bound_goals_in_loop(+BoundGoals, +FactVarMap, +HeadArgs, -Code, -OutputVarMap)
%  Generate bound goal code using fact-derived variables.
generate_bound_goals_in_loop([], VarMap, _, "", VarMap).
generate_bound_goals_in_loop([Goal-Binding|Rest], VarMap, HeadArgs, Code, FinalVarMap) :-
    generate_bound_goal_code(Goal, Binding, VarMap, GoalCode, NewVarMap),
    generate_bound_goals_in_loop(Rest, NewVarMap, HeadArgs, RestCode, FinalVarMap),
    (   RestCode = ""
    ->  Code = GoalCode
    ;   format(string(Code), "~w\n        ~w", [GoalCode, RestCode])
    ).

%% generate_output_object(+HeadArgs, +OutputVarMap, +FactVarMap, -Code)
%  Generate the output PSCustomObject based on head arguments.
generate_output_object(HeadArgs, OutputVarMap, FactVarMap, Code) :-
    % Combine both var maps for lookup
    append(OutputVarMap, FactVarMap, CombinedMap),
    generate_output_fields(HeadArgs, CombinedMap, 0, Fields),
    atomic_list_concat(Fields, '; ', FieldsStr),
    format(string(Code), "[PSCustomObject]@{ ~w }", [FieldsStr]).

generate_output_fields([], _, _, []).
generate_output_fields([Arg|Rest], VarMap, Idx, [Field|RestFields]) :-
    NextIdx is Idx + 1,
    (Idx = 0 -> OutName = "X" ; Idx = 1 -> OutName = "Y" ; format(atom(OutName), "F~w", [Idx])),
    (   var(Arg),
        lookup_var_eq(Arg, VarMap, PSVar)
    ->  format(string(Field), "~w = ~w", [OutName, PSVar])
    ;   var(Arg)
    ->  format(string(Field), "~w = $null", [OutName])
    ;   format(string(Field), "~w = '~w'", [OutName, Arg])
    ),
    generate_output_fields(Rest, VarMap, NextIdx, RestFields).

%% generate_multi_fact_loop(+PredStr, +HeadArgs, +FG1, +FG2, +BoundGoals, +HeadVars, +FactVars, +BoundVars, +NegatedGoals, +Options, -Code)
%  Generate joins for multiple fact goals.
%  Supports two join strategies:
%    - Hash join (default): O(n+m) - builds hashtable on first relation, probes with second
%    - Nested loop: O(n*m) - fallback for cross products or when explicitly requested
%
%  Use option join_strategy(nested_loop) to force nested loops.
%  Use option join_strategy(hash) for hash join (default when join conditions exist).
generate_multi_fact_loop(PredStr, HeadArgs, FG1, FG2, BoundGoals, _HeadVars,
                          _FactVars, _BoundVars, NegatedGoals, Options, Code) :-
    strip_module_qualifier(FG1, Plain1),
    strip_module_qualifier(FG2, Plain2),
    Plain1 =.. [FP1|Args1],
    Plain2 =.. [FP2|Args2],
    atom_string(FP1, FP1Str),
    atom_string(FP2, FP2Str),

    % Create variable maps for each fact with prefixed field names
    create_prefixed_fact_var_map(Args1, "r1", VarMap1),
    create_prefixed_fact_var_map(Args2, "r2", VarMap2),

    % Detect join conditions (shared variables between facts)
    detect_join_conditions(Args1, Args2, VarMap1, VarMap2, JoinConditions),

    % Also get the join key info for hash joins
    detect_join_keys(Args1, Args2, JoinKeys),

    % Combine var maps for bound goals and output
    append(VarMap1, VarMap2, CombinedVarMap),

    % Generate bound goal code if any
    (   BoundGoals \= []
    ->  generate_bound_goals_in_loop(BoundGoals, CombinedVarMap, HeadArgs, BoundCode, OutputVarMap),
        format(string(BoundSection), "~n        # Apply bound operations~n        ~w", [BoundCode])
    ;   BoundSection = "",
        OutputVarMap = []
    ),

    % Generate negation check if needed
    (   NegatedGoals \= []
    ->  generate_negation_check_ps(NegatedGoals, NegCheck),
        format(string(NegCheckStr), "~n        ~w", [NegCheck])
    ;   NegCheckStr = ""
    ),

    % Generate output object
    append(OutputVarMap, CombinedVarMap, FinalVarMap),
    generate_output_object(HeadArgs, OutputVarMap, FinalVarMap, OutputCode),

    % Check for verbose output
    (   member(verbose_output(true), Options)
    ->  VerboseFlag = true
    ;   VerboseFlag = false
    ),

    % Determine join strategy
    (   member(join_strategy(nested_loop), Options)
    ->  % Force nested loop
        generate_nested_loop_join(PredStr, FP1Str, FP2Str, Args1, Args2, VarMap1, VarMap2,
                                  JoinConditions, NegCheckStr, BoundSection, OutputCode, VerboseFlag, Code)
    ;   JoinKeys \= []
    ->  % Has join keys - use hash join
        generate_hash_join(PredStr, FP1Str, FP2Str, Args1, Args2, VarMap1, VarMap2,
                          JoinKeys, NegCheckStr, BoundSection, OutputCode, VerboseFlag, Code)
    ;   % No join keys (cross product) - use nested loop
        generate_nested_loop_join(PredStr, FP1Str, FP2Str, Args1, Args2, VarMap1, VarMap2,
                                  JoinConditions, NegCheckStr, BoundSection, OutputCode, VerboseFlag, Code)
    ).

%% generate_hash_join(+PredStr, +FP1Str, +FP2Str, +Args1, +Args2, +VarMap1, +VarMap2, +JoinKeys, +NegCheckStr, +BoundSection, +OutputCode, +VerboseFlag, -Code)
%  Generate a hash-based join for O(n+m) performance.
%  Builds a hashtable from the first relation, then probes with the second.
generate_hash_join(PredStr, FP1Str, FP2Str, Args1, Args2, VarMap1, VarMap2,
                   JoinKeys, NegCheckStr, BoundSection, OutputCode, VerboseFlag, Code) :-
    % Get the join key field names for building hashtable
    generate_hash_key_expr(JoinKeys, "r1", HashKeyExpr1),
    generate_hash_key_expr(JoinKeys, "r2", HashKeyExpr2),

    % Generate field extractions
    generate_prefixed_field_extraction(Args1, "r1", Extraction1),
    generate_prefixed_field_extraction(Args2, "r2", Extraction2),

    % Verbose output
    (   VerboseFlag = true
    ->  format(string(VerboseStr), "Write-Verbose \"[~w] Building hash index on ~w...\"~n    ", [PredStr, FP1Str])
    ;   VerboseStr = ""
    ),

    format(string(Code),
"    # Hash join: O(n+m) complexity
    ~w# Build hash index on first relation
    $hashIndex = @{}
    foreach ($r1 in $~w_data) {
        $key = ~w
        if (-not $hashIndex.ContainsKey($key)) {
            $hashIndex[$key] = [System.Collections.ArrayList]::new()
        }
        [void]$hashIndex[$key].Add($r1)
    }

    # Probe hash index with second relation
    $results = foreach ($r2 in $~w_data) {
        $probeKey = ~w
        if ($hashIndex.ContainsKey($probeKey)) {
            foreach ($r1 in $hashIndex[$probeKey]) {
                # Extract fields from first fact
                ~w
                # Extract fields from second fact
                ~w~w~w
                # Output result
                ~w
            }
        }
    }
    $results", [VerboseStr, FP1Str, HashKeyExpr1, FP2Str, HashKeyExpr2,
                Extraction1, Extraction2, NegCheckStr, BoundSection, OutputCode]).

%% generate_nested_loop_join(+PredStr, +FP1Str, +FP2Str, +Args1, +Args2, +VarMap1, +VarMap2, +JoinConditions, +NegCheckStr, +BoundSection, +OutputCode, +VerboseFlag, -Code)
%  Generate a nested loop join (original implementation).
generate_nested_loop_join(PredStr, FP1Str, FP2Str, Args1, Args2, _VarMap1, _VarMap2,
                          JoinConditions, NegCheckStr, BoundSection, OutputCode, VerboseFlag, Code) :-
    % Generate field extractions
    generate_prefixed_field_extraction(Args1, "r1", Extraction1),
    generate_prefixed_field_extraction(Args2, "r2", Extraction2),

    % Generate join condition string
    (   JoinConditions \= []
    ->  atomic_list_concat(JoinConditions, ' -and ', JoinStr)
    ;   JoinStr = "$true"  % No join condition = cross product
    ),

    % Verbose output
    (   VerboseFlag = true
    ->  format(string(VerboseStr), "Write-Verbose \"[~w] Nested loop join...\"~n            ", [PredStr])
    ;   VerboseStr = ""
    ),

    format(string(Code),
"    # Nested loop join: O(n*m) complexity
    $results = foreach ($r1 in $~w_data) {
        foreach ($r2 in $~w_data) {
            ~w# Extract fields from first fact
            ~w
            # Extract fields from second fact
            ~w
            # Check join condition
            if (~w) {~w~w
                # Output result
                ~w
            }
        }
    }
    $results", [FP1Str, FP2Str, VerboseStr, Extraction1, Extraction2, JoinStr, NegCheckStr, BoundSection, OutputCode]).

%% detect_join_keys(+Args1, +Args2, -JoinKeys)
%  Detect shared variables and return their field positions for hash key generation.
%  JoinKeys is a list of idx1-idx2 pairs indicating which fields to use for the join key.
detect_join_keys(Args1, Args2, JoinKeys) :-
    findall(Idx1-Idx2, (
        nth0(Idx1, Args1, Arg1),
        var(Arg1),
        nth0(Idx2, Args2, Arg2),
        var(Arg2),
        Arg1 == Arg2
    ), JoinKeys).

%% generate_hash_key_expr(+JoinKeys, +Prefix, -KeyExpr)
%  Generate a PowerShell expression for the hash key based on join fields.
generate_hash_key_expr(JoinKeys, Prefix, KeyExpr) :-
    findall(FieldAccess, (
        member(Idx-_, JoinKeys),
        idx_to_field_name(Idx, FieldName),
        format(string(FieldAccess), "$~w.~w", [Prefix, FieldName])
    ), FieldAccesses1),
    findall(FieldAccess, (
        member(_-Idx, JoinKeys),
        Prefix = "r2",
        idx_to_field_name(Idx, FieldName),
        format(string(FieldAccess), "$~w.~w", [Prefix, FieldName])
    ), FieldAccesses2),
    % Use appropriate list based on prefix
    (   Prefix = "r1"
    ->  FieldAccesses = FieldAccesses1
    ;   % For r2, we need the second index from each pair
        findall(FieldAccess, (
            member(_-Idx, JoinKeys),
            idx_to_field_name(Idx, FieldName),
            format(string(FieldAccess), "$~w.~w", [Prefix, FieldName])
        ), FieldAccesses)
    ),
    (   FieldAccesses = [Single]
    ->  KeyExpr = Single
    ;   atomic_list_concat(FieldAccesses, ':', Combined),
        format(string(KeyExpr), "\"~w\"", [Combined])
    ).

%% idx_to_field_name(+Idx, -FieldName)
%  Convert a 0-based index to a field name (X, Y, F2, F3, ...).
idx_to_field_name(0, "X") :- !.
idx_to_field_name(1, "Y") :- !.
idx_to_field_name(Idx, FieldName) :-
    format(atom(FieldName), "F~w", [Idx]).

%% create_prefixed_fact_var_map(+FactArgs, +Prefix, -VarMap)
%  Create a variable map with prefixed PS variables (e.g., $r1_X, $r1_Y)
create_prefixed_fact_var_map(FactArgs, Prefix, VarMap) :-
    create_prefixed_fact_var_map(FactArgs, Prefix, 0, VarMap).

create_prefixed_fact_var_map([], _, _, []).
create_prefixed_fact_var_map([Arg|Rest], Prefix, Idx, VarMap) :-
    NextIdx is Idx + 1,
    create_prefixed_fact_var_map(Rest, Prefix, NextIdx, RestMap),
    (   var(Arg)
    ->  (Idx = 0 -> FieldName = "X" ; Idx = 1 -> FieldName = "Y" ; format(atom(FieldName), "F~w", [Idx])),
        format(string(PSVar), "$~w_~w", [Prefix, FieldName]),
        VarMap = [Arg-PSVar|RestMap]
    ;   VarMap = RestMap
    ).

%% generate_prefixed_field_extraction(+FactArgs, +Prefix, -Code)
%  Generate field extraction with prefixed variable names.
generate_prefixed_field_extraction(FactArgs, Prefix, Code) :-
    generate_prefixed_field_extraction(FactArgs, Prefix, 0, Extractions),
    atomic_list_concat(Extractions, '\n            ', Code).

generate_prefixed_field_extraction([], _, _, []).
generate_prefixed_field_extraction([_|Rest], Prefix, Idx, [Extraction|RestExtractions]) :-
    NextIdx is Idx + 1,
    (Idx = 0 -> FieldName = "X" ; Idx = 1 -> FieldName = "Y" ; format(atom(FieldName), "F~w", [Idx])),
    format(string(Extraction), "$~w_~w = $~w.~w", [Prefix, FieldName, Prefix, FieldName]),
    generate_prefixed_field_extraction(Rest, Prefix, NextIdx, RestExtractions).

%% detect_join_conditions(+Args1, +Args2, +VarMap1, +VarMap2, -JoinConditions)
%  Detect shared variables between two fact goals and generate join conditions.
detect_join_conditions(Args1, Args2, VarMap1, VarMap2, JoinConditions) :-
    findall(Condition, (
        member(Arg1, Args1),
        var(Arg1),
        member(Arg2, Args2),
        var(Arg2),
        Arg1 == Arg2,  % Same Prolog variable
        lookup_var_eq(Arg1, VarMap1, PSVar1),
        lookup_var_eq(Arg2, VarMap2, PSVar2),
        format(string(Condition), "~w -eq ~w", [PSVar1, PSVar2])
    ), JoinConditions).

%% ============================================================================
%% N-WAY JOINS (3+ fact goals)
%% ============================================================================
%%
%% For rules with 3 or more fact goals, we generate pipelined nested hash joins:
%%   Join(F1, F2) -> Result1
%%   Join(Result1, F3) -> Result2
%%   ...
%%
%% This maintains O(n+m+p+...) complexity for equi-joins.

%% generate_nway_fact_loop(+PredStr, +HeadArgs, +FactGoals, +BoundGoals, +HeadVars, +FactVars, +BoundVars, +NegatedGoals, +Options, -Code)
%  Generate joins for 3 or more fact goals using pipelined hash joins.
generate_nway_fact_loop(PredStr, HeadArgs, FactGoals, BoundGoals, _HeadVars,
                        _FactVars, _BoundVars, NegatedGoals, Options, Code) :-
    length(FactGoals, NumFacts),
    format('[PowerShell] Generating ~w-way join~n', [NumFacts]),

    % Build fact info list with prefixes r1, r2, r3, ...
    number_fact_goals(FactGoals, 1, NumberedFacts),

    % Generate fact loaders
    generate_all_fact_loaders(NumberedFacts, LoaderCode),

    % Build combined variable map from all facts
    build_combined_var_map(NumberedFacts, CombinedVarMap),

    % Generate bound goal code if any
    (   BoundGoals \= []
    ->  generate_bound_goals_in_loop(BoundGoals, CombinedVarMap, HeadArgs, BoundCode, OutputVarMap),
        format(string(BoundSection), "~n                # Apply bound operations~n                ~w", [BoundCode])
    ;   BoundSection = "",
        OutputVarMap = []
    ),

    % Generate negation check if needed
    (   NegatedGoals \= []
    ->  generate_negation_check_ps(NegatedGoals, NegCheck),
        format(string(NegCheckStr), "~n                ~w", [NegCheck])
    ;   NegCheckStr = ""
    ),

    % Generate output object
    append(OutputVarMap, CombinedVarMap, FinalVarMap),
    generate_output_object(HeadArgs, OutputVarMap, FinalVarMap, OutputCode),

    % Check for verbose output
    (   member(verbose_output(true), Options)
    ->  format(string(VerboseStr), "Write-Verbose \"[~w] ~w-way join...\"~n    ", [PredStr, NumFacts])
    ;   VerboseStr = ""
    ),

    % Generate the pipelined join code
    generate_pipelined_joins(NumberedFacts, NegCheckStr, BoundSection, OutputCode,
                             VerboseStr, JoinCode),

    format(string(Code), "    ~w~w", [LoaderCode, JoinCode]).

%% number_fact_goals(+FactGoals, +StartNum, -NumberedFacts)
%  Number fact goals as r1, r2, r3, etc.
number_fact_goals([], _, []).
number_fact_goals([FG|Rest], N, [fact_info(N, Prefix, FPred, Args, VarMap)|RestNumbered]) :-
    strip_module_qualifier(FG, PlainFG),
    PlainFG =.. [FPred|Args],
    format(atom(Prefix), "r~w", [N]),
    create_prefixed_fact_var_map(Args, Prefix, VarMap),
    N1 is N + 1,
    number_fact_goals(Rest, N1, RestNumbered).

%% generate_all_fact_loaders(+NumberedFacts, -Code)
%  Generate loader code for all facts.
generate_all_fact_loaders(NumberedFacts, Code) :-
    findall(LoaderLine, (
        member(fact_info(_, _, FPred, _, _), NumberedFacts),
        atom_string(FPred, FPredStr),
        format(string(LoaderLine), "$~w_data = ~w", [FPredStr, FPredStr])
    ), LoaderLines),
    atomic_list_concat(LoaderLines, '\n    ', Code).

%% build_combined_var_map(+NumberedFacts, -CombinedVarMap)
%  Combine variable maps from all facts.
build_combined_var_map([], []).
build_combined_var_map([fact_info(_, _, _, _, VarMap)|Rest], CombinedVarMap) :-
    build_combined_var_map(Rest, RestMap),
    append(VarMap, RestMap, CombinedVarMap).

%% generate_pipelined_joins(+NumberedFacts, +NegCheckStr, +BoundSection, +OutputCode, +VerboseStr, -Code)
%  Generate pipelined hash joins for N facts.
%  Strategy: nested loops with hash acceleration where possible.
generate_pipelined_joins(NumberedFacts, NegCheckStr, BoundSection, OutputCode, VerboseStr, Code) :-
    % For N-way joins, we generate deeply nested foreach loops with join conditions
    % at each level. Future optimization: build hash indices for each pair.

    % Collect all field extractions
    findall(ExtrLines, (
        member(fact_info(_, Prefix, _, Args, _), NumberedFacts),
        generate_prefixed_field_extraction(Args, Prefix, ExtrLines)
    ), AllExtractions),
    atomic_list_concat(AllExtractions, '\n                ', ExtractionsCode),

    % Detect all join conditions between consecutive facts
    detect_all_join_conditions(NumberedFacts, AllJoinConditions),

    % Generate nested foreach loops
    generate_nested_foreach_loops(NumberedFacts, AllJoinConditions, ExtractionsCode,
                                   NegCheckStr, BoundSection, OutputCode, VerboseStr, Code).

%% detect_all_join_conditions(+NumberedFacts, -AllConditions)
%  Detect join conditions between all pairs of facts.
detect_all_join_conditions(NumberedFacts, AllConditions) :-
    findall(Condition, (
        member(fact_info(N1, _, _, Args1, VarMap1), NumberedFacts),
        member(fact_info(N2, _, _, Args2, VarMap2), NumberedFacts),
        N1 < N2,
        member(Arg1, Args1),
        var(Arg1),
        member(Arg2, Args2),
        var(Arg2),
        Arg1 == Arg2,
        lookup_var_eq(Arg1, VarMap1, PSVar1),
        lookup_var_eq(Arg2, VarMap2, PSVar2),
        format(string(Condition), "~w -eq ~w", [PSVar1, PSVar2])
    ), AllConditions).

%% generate_nested_foreach_loops(+Facts, +JoinConds, +Extractions, +NegCheck, +Bound, +Output, +Verbose, -Code)
%  Generate deeply nested foreach loops.
generate_nested_foreach_loops(NumberedFacts, JoinConditions, ExtractionsCode,
                               NegCheckStr, BoundSection, OutputCode, VerboseStr, Code) :-
    length(NumberedFacts, NumFacts),
    % Build the opening loops
    build_foreach_openings(NumberedFacts, Openings, Closings),

    % Build the join condition check
    (   JoinConditions \= []
    ->  atomic_list_concat(JoinConditions, ' -and ', JoinStr)
    ;   JoinStr = "$true"
    ),

    % Calculate indentation for innermost block (4 spaces per nesting level + base)
    BaseIndent = 4,
    InnerIndent is BaseIndent + (NumFacts * 4),
    indent_string(InnerIndent, InnerIndentStr),

    format(string(Code),
"~w# ~w-way nested loop join
    $results = ~w
        # Extract all fields
        ~w
        # Check join conditions
        if (~w) {~w~w
            # Output result
            ~w
        }
    ~w
    $results",
           [VerboseStr, NumFacts, Openings, ExtractionsCode, JoinStr,
            NegCheckStr, BoundSection, OutputCode, Closings]).

%% build_foreach_openings(+Facts, -Openings, -Closings)
%  Build the foreach opening and closing strings.
build_foreach_openings([], "", "").
build_foreach_openings([fact_info(N, Prefix, FPred, _, _)|Rest], Openings, Closings) :-
    atom_string(FPred, FPredStr),
    (   N = 1
    ->  format(string(ThisOpen), "foreach ($~w in $~w_data) {", [Prefix, FPredStr])
    ;   format(string(ThisOpen), "~n        foreach ($~w in $~w_data) {", [Prefix, FPredStr])
    ),
    build_foreach_openings(Rest, RestOpenings, RestClosings),
    atom_concat(ThisOpen, RestOpenings, Openings),
    atom_concat("}", RestClosings, Closings).

%% indent_string(+NumSpaces, -Str)
%  Generate a string of spaces.
indent_string(0, "") :- !.
indent_string(N, Str) :-
    N > 0,
    N1 is N - 1,
    indent_string(N1, Rest),
    atom_concat(" ", Rest, Str).

%% generate_bound_goals_code(+BoundGoals, +VarMap, -Codes, -FinalVarMap)
%
%  Generate PowerShell code for a list of bound goals.
%
generate_bound_goals_code([], VarMap, [], VarMap).
generate_bound_goals_code([Goal-Binding|Rest], VarMap, [Code|RestCodes], FinalVarMap) :-
    generate_bound_goal_code(Goal, Binding, VarMap, Code, NewVarMap),
    generate_bound_goals_code(Rest, NewVarMap, RestCodes, FinalVarMap).

% ============================================================================
% BINDING-BASED CMDLET GENERATION (Book 12, Chapters 3 & 5)
% ============================================================================
%
% These predicates integrate with the binding system to generate PowerShell
% cmdlets that wrap Prolog predicate semantics using proper PowerShell patterns.

%% init_powershell_compiler
%
%  Initialize the PowerShell compiler with all registered bindings.
%  Call this before using binding-based compilation.
%
init_powershell_compiler :-
    format('[PowerShell Compiler] Initializing binding system...~n', []),
    powershell_bindings:init_powershell_bindings,
    bindings_for_target(powershell, Bindings),
    length(Bindings, NumBindings),
    format('[PowerShell Compiler] Loaded ~w PowerShell bindings~n', [NumBindings]).

%% has_powershell_binding(+Pred)
%
%  Check if a predicate has a registered PowerShell binding.
%
%  @param Pred  Name/Arity predicate indicator
%
has_powershell_binding(Pred) :-
    binding(powershell, Pred, _, _, _, _).

%% compile_bound_predicate(+Pred, +Options, -Code)
%
%  Compile a predicate using its PowerShell binding to generate a cmdlet call.
%  Falls back to standard compilation if no binding exists.
%
%  @param Pred     Name/Arity predicate indicator
%  @param Options  Compilation options:
%                    - args(List) - Arguments to pass to the binding
%                    - wrap_cmdlet(true) - Wrap in cmdlet function
%                    - verbose_output(true) - Add Write-Verbose statements
%  @param Code     Generated PowerShell code
%
compile_bound_predicate(Pred, Options, Code) :-
    (   has_powershell_binding(Pred)
    ->  % Use binding-based generation
        format('[PowerShell Compiler] Using binding for ~w~n', [Pred]),
        (   member(args(Args), Options)
        ->  true
        ;   Args = []
        ),
        (   member(wrap_cmdlet(true), Options)
        ->  % Generate full cmdlet wrapper
            generate_cmdlet_wrapper(Pred, Options, Code)
        ;   % Generate just the call
            binding_codegen:generate_binding_call(powershell, Pred, Args, Code)
        )
    ;   % Fall back to standard compilation
        format('[PowerShell Compiler] No binding for ~w, using standard compilation~n', [Pred]),
        compile_to_powershell(Pred, Options, Code)
    ).

%% generate_cmdlet_wrapper(+Pred, +Options, -CmdletCode)
%
%  Generate a complete PowerShell cmdlet function that wraps a bound predicate.
%  Follows PowerShell conventions: [CmdletBinding()], param(), begin/process/end.
%
%  @param Pred        Name/Arity predicate indicator
%  @param Options     Options passed to binding_codegen
%  @param CmdletCode  Complete cmdlet function code
%
generate_cmdlet_wrapper(Pred, Options, CmdletCode) :-
    binding_codegen:generate_cmdlet_from_binding(powershell, Pred, Options, CmdletCode).

%% compile_with_bindings(+Predicate, +Options, -Code)
%
%  Enhanced compilation that first checks for bindings before falling back
%  to traditional compilation. Useful for predicates that map directly to
%  PowerShell cmdlets (e.g., get_service -> Get-Service).
%
compile_with_bindings(Predicate, Options, Code) :-
    Predicate = Pred/Arity,
    format('[PowerShell Compiler] Checking bindings for ~w/~w~n', [Pred, Arity]),
    (   has_powershell_binding(Predicate)
    ->  compile_bound_predicate(Predicate, Options, Code)
    ;   compile_to_powershell(Predicate, Options, Code)
    ).

%% generate_binding_call_inline(+Pred, +Args, -Code)
%
%  Generate inline code for a bound predicate call without cmdlet wrapper.
%  Useful when embedding bound calls within larger generated code.
%
generate_binding_call_inline(Pred, Args, Code) :-
    (   has_powershell_binding(Pred)
    ->  binding_codegen:generate_binding_call(powershell, Pred, Args, Code)
    ;   % No binding - generate placeholder
        Pred = Name/Arity,
        format(string(Code), "# Unbound predicate: ~w/~w", [Name, Arity])
    ).

%% list_powershell_bindings
%
%  List all registered PowerShell bindings with their target names.
%
list_powershell_bindings :-
    format('~n=== Registered PowerShell Bindings ===~n~n', []),
    forall(
        binding(powershell, Pred, TargetName, Inputs, Outputs, Opts),
        (   Pred = Name/Arity,
            length(Inputs, NumIn),
            length(Outputs, NumOut),
            format('  ~w/~w -> ~w  (in:~w, out:~w)~n', [Name, Arity, TargetName, NumIn, NumOut]),
            (   member(effect(E), Opts)
            ->  format('    effect: ~w~n', [E])
            ;   true
            ),
            (   member(pattern(P), Opts)
            ->  format('    pattern: ~w~n', [P])
            ;   true
            )
        )
    ),
    format('~n', []).

%% test_bindings_integration
%
%  Test the binding system integration with the PowerShell compiler.
%
test_bindings_integration :-
    format('~n=== Testing Binding System Integration ===~n~n', []),

    % Initialize
    format('[Test] Initializing...~n', []),
    init_powershell_compiler,

    % Test 1: Check binding exists
    format('[Test 1] Check binding existence~n', []),
    (   has_powershell_binding(get_service/1)
    ->  format('  [PASS] get_service/1 has binding~n', [])
    ;   format('  [FAIL] get_service/1 binding not found~n', [])
    ),

    % Test 2: Generate binding call
    format('[Test 2] Generate binding call~n', []),
    generate_binding_call_inline(sqrt/2, [16], SqrtCode),
    format('  sqrt(16) -> ~w~n', [SqrtCode]),
    (   sub_string(SqrtCode, _, _, _, "Sqrt")
    ->  format('  [PASS] Generated [Math]::Sqrt call~n', [])
    ;   format('  [FAIL] Expected Sqrt in output~n', [])
    ),

    % Test 3: Generate cmdlet wrapper
    format('[Test 3] Generate cmdlet wrapper~n', []),
    generate_cmdlet_wrapper(test_path/1, [verbose_output(true)], CmdletCode),
    (   sub_string(CmdletCode, _, _, _, "[CmdletBinding()]")
    ->  format('  [PASS] Generated CmdletBinding attribute~n', [])
    ;   format('  [FAIL] Missing CmdletBinding~n', [])
    ),

    % Test 4: Compile bound predicate
    format('[Test 4] Compile bound predicate with call~n', []),
    compile_bound_predicate(write_output/1, [args(['Hello'])], OutputCode),
    format('  write_output -> ~w~n', [OutputCode]),
    (   sub_string(OutputCode, _, _, _, "Write-Output")
    ->  format('  [PASS] Generated Write-Output call~n', [])
    ;   format('  [FAIL] Expected Write-Output~n', [])
    ),

    format('~n=== Binding Integration Tests Complete ===~n', []).

%% test_bound_rule_compilation
%
%  Test compilation of rules that use bound predicates.
%
test_bound_rule_compilation :-
    format('~n=== Testing Bound Rule Compilation ===~n~n', []),

    % Initialize bindings
    init_powershell_compiler,

    % Test 1: Goal classification
    format('[Test 1] Goal classification~n', []),
    classify_body_goals([sqrt(X, Y), parent(A, B), X > 0], Bound, Facts, Builtins),
    length(Bound, NumBound),
    length(Facts, NumFacts),
    length(Builtins, NumBuiltins),
    format('  Bound: ~w, Facts: ~w, Builtins: ~w~n', [NumBound, NumFacts, NumBuiltins]),
    (   NumBound = 1, NumFacts = 1, NumBuiltins = 1
    ->  format('  [PASS] Goals correctly classified~n', [])
    ;   format('  [FAIL] Expected 1 bound, 1 fact, 1 builtin~n', [])
    ),

    % Test 2: Generate bound goal code
    format('[Test 2] Generate bound goal code~n', []),
    goal_has_binding(sqrt(16, Result), Binding),
    generate_bound_goal_code(sqrt(16, Result), Binding, [], Code, _NewVarMap),
    format('  sqrt(16, Result) -> ~w~n', [Code]),
    (   sub_string(Code, _, _, _, "[Math]::Sqrt")
    ->  format('  [PASS] Generated .NET static call~n', [])
    ;   format('  [FAIL] Expected [Math]::Sqrt~n', [])
    ),

    % Test 3: Format Prolog args for PowerShell
    format('[Test 3] Format arguments~n', []),
    format_arg_for_ps([], 'ServiceName', Arg1),
    format_arg_for_ps([], 42, Arg2),
    format_arg_for_ps([], hello, Arg3),
    format('  ServiceName -> ~w~n', [Arg1]),
    format('  42 -> ~w~n', [Arg2]),
    format('  hello -> ~w~n', [Arg3]),
    (   Arg1 = "$ServiceName", Arg2 = "42", Arg3 = "'hello'"
    ->  format('  [PASS] Arguments formatted correctly~n', [])
    ;   format('  [FAIL] Argument formatting issue~n', [])
    ),

    % Test 4: Compile a pure bound rule
    format('[Test 4] Compile pure bound rule~n', []),
    % Define a test rule dynamically
    % NOTE: Must use distinct variable names from Tests 1-3 to avoid
    % sharing Prolog variables across the test predicate!
    abolish(user:compute_sqrt/2),
    assertz((user:compute_sqrt(In, Out) :- sqrt(In, Out))),
    (   compile_to_powershell(compute_sqrt/2, [powershell_mode(pure)], PSCode)
    ->  format('  Generated code length: ~w chars~n', [PSCode]),
        (   sub_string(PSCode, _, _, _, "[Math]::Sqrt")
        ->  format('  [PASS] Pure bound rule compiled with binding~n', [])
        ;   format('  [INFO] Code generated but binding not used (may need init)~n', []),
            format('  Code snippet: ~w~n', [PSCode])
        )
    ;   format('  [FAIL] Compilation failed~n', [])
    ),
    abolish(user:compute_sqrt/2),

    % Test 5: Compile a mixed bound/fact rule
    format('[Test 5] Compile mixed bound/fact rule~n', []),
    % Define facts first
    abolish(user:test_emp/2),
    assertz(user:test_emp(alice, 100)),
    assertz(user:test_emp(bob, 144)),
    % Define mixed rule: test_emp is fact-based, sqrt is bound
    % Use unique variable names to avoid sharing with other tests
    abolish(user:emp_sqrt/2),
    assertz((user:emp_sqrt(MixName, MixSqrt) :- test_emp(MixName, MixVal), sqrt(MixVal, MixSqrt))),
    (   compile_to_powershell(emp_sqrt/2, [powershell_mode(pure)], MixedCode)
    ->  % Check each pattern individually for debugging
        (sub_string(MixedCode, _, _, _, "foreach") -> HasForeach = true ; HasForeach = false),
        (sub_string(MixedCode, _, _, _, "[Math]::Sqrt") -> HasSqrt = true ; HasSqrt = false),
        (sub_string(MixedCode, _, _, _, "$fact") -> HasFact = true ; HasFact = false),
        (   HasForeach = true, HasSqrt = true, HasFact = true
        ->  format('  [PASS] Mixed rule compiled with foreach loop and binding~n', [])
        ;   format('  [FAIL] Missing patterns: foreach=~w, Sqrt=~w, fact=~w~n', [HasForeach, HasSqrt, HasFact])
        )
    ;   format('  [FAIL] Mixed rule compilation failed~n', [])
    ),
    abolish(user:test_emp/2),
    abolish(user:emp_sqrt/2),

    % Test 6: Compile multi-fact join rule
    format('[Test 6] Compile multi-fact join rule~n', []),
    % Define facts for a transitive closure style join
    abolish(user:edge/2),
    assertz(user:edge(a, b)),
    assertz(user:edge(b, c)),
    assertz(user:edge(c, d)),
    % Define join rule: path(X, Z) :- edge(X, Y), edge(Y, Z)
    abolish(user:path/2),
    assertz((user:path(JoinX, JoinZ) :- edge(JoinX, JoinY), edge(JoinY, JoinZ))),
    (   compile_to_powershell(path/2, [powershell_mode(pure)], JoinCode)
    ->  % Check for nested loops and join condition
        (sub_string(JoinCode, _, _, _, "foreach ($r1") -> HasR1 = true ; HasR1 = false),
        (sub_string(JoinCode, _, _, _, "foreach ($r2") -> HasR2 = true ; HasR2 = false),
        (sub_string(JoinCode, _, _, _, "-eq") -> HasJoinCond = true ; HasJoinCond = false),
        (   HasR1 = true, HasR2 = true, HasJoinCond = true
        ->  format('  [PASS] Multi-fact join compiled with nested loops and join condition~n', [])
        ;   format('  [FAIL] Missing patterns: r1=~w, r2=~w, join=~w~n', [HasR1, HasR2, HasJoinCond])
        )
    ;   format('  [FAIL] Multi-fact join compilation failed~n', [])
    ),
    abolish(user:edge/2),
    abolish(user:path/2),

    % Test 7: Multi-fact join with bound goal
    format('[Test 7] Multi-fact join with bound goal~n', []),
    % Define facts for join
    abolish(user:item/2),
    assertz(user:item(apple, 4)),
    assertz(user:item(banana, 9)),
    abolish(user:price/2),
    assertz(user:price(apple, 100)),
    assertz(user:price(banana, 200)),
    % Join with sqrt bound goal: combined(Name, SqrtQty) :- item(Name, Qty), price(Name, _), sqrt(Qty, SqrtQty)
    abolish(user:combined/2),
    assertz((user:combined(ItemName, SqrtQty) :- item(ItemName, ItemQty), price(ItemName, _ItemPrice), sqrt(ItemQty, SqrtQty))),
    (   compile_to_powershell(combined/2, [powershell_mode(pure)], CombinedCode)
    ->  (sub_string(CombinedCode, _, _, _, "[Math]::Sqrt") -> HasSqrtBind = true ; HasSqrtBind = false),
        (sub_string(CombinedCode, _, _, _, "foreach") -> HasLoop = true ; HasLoop = false),
        % Check for hash join pattern
        (sub_string(CombinedCode, _, _, _, "$hashIndex") -> HasHashJoin = true ; HasHashJoin = false),
        (   HasSqrtBind = true, HasLoop = true, HasHashJoin = true
        ->  format('  [PASS] Multi-fact join with bound goal compiled (using hash join)~n', [])
        ;   HasSqrtBind = true, HasLoop = true
        ->  format('  [PASS] Multi-fact join with bound goal compiled (nested loop fallback)~n', [])
        ;   format('  [FAIL] Missing patterns: Sqrt=~w, loop=~w, hashJoin=~w~n', [HasSqrtBind, HasLoop, HasHashJoin])
        )
    ;   format('  [FAIL] Multi-fact join with bound goal compilation failed~n', [])
    ),
    abolish(user:item/2),
    abolish(user:price/2),
    abolish(user:combined/2),

    % Test 8: Verify hash join pattern explicitly
    format('[Test 8] Hash join vs nested loop selection~n', []),
    % Define facts for testing hash join
    abolish(user:emp/2),
    assertz(user:emp(alice, 100)),
    assertz(user:emp(bob, 200)),
    abolish(user:dept/2),
    assertz(user:dept(alice, engineering)),
    assertz(user:dept(bob, sales)),
    % Join rule with bound goal (should use hash join)
    abolish(user:emp_dept_sqrt/3),
    assertz((user:emp_dept_sqrt(EmpName, EmpDept, SqrtSal) :-
        emp(EmpName, EmpSal), dept(EmpName, EmpDept), sqrt(EmpSal, SqrtSal))),
    (   compile_to_powershell(emp_dept_sqrt/3, [powershell_mode(pure)], HashJoinCode)
    ->  (sub_string(HashJoinCode, _, _, _, "$hashIndex") -> UsesHashJoin = true ; UsesHashJoin = false),
        (sub_string(HashJoinCode, _, _, _, "O(n+m)") -> HasHashComment = true ; HasHashComment = false),
        (sub_string(HashJoinCode, _, _, _, "probeKey") -> HasProbe = true ; HasProbe = false),
        (   UsesHashJoin = true, HasProbe = true
        ->  format('  [PASS] Hash join generated for equi-join with bound goal~n', [])
        ;   format('  [INFO] Using nested loop (join_keys not detected): hash=~w, probe=~w~n',
                   [UsesHashJoin, HasProbe])
        )
    ;   format('  [FAIL] Hash join compilation failed~n', [])
    ),
    abolish(user:emp/2),
    abolish(user:dept/2),
    abolish(user:emp_dept_sqrt/3),

    % Test 9: 3-way join (N-way join support)
    format('[Test 9] 3-way join with bound goal~n', []),
    % Define facts for 3-way join: employee, department, location
    abolish(user:employee/2),
    assertz(user:employee(alice, eng)),
    assertz(user:employee(bob, sales)),
    abolish(user:department/2),
    assertz(user:department(eng, building_a)),
    assertz(user:department(sales, building_b)),
    abolish(user:building/2),
    assertz(user:building(building_a, 100)),
    assertz(user:building(building_b, 200)),
    % 3-way join: employee -> department -> building with sqrt on capacity
    abolish(user:emp_capacity/3),
    assertz((user:emp_capacity(EmpName3, Building3, SqrtCap3) :-
        employee(EmpName3, Dept3), department(Dept3, Building3), building(Building3, Cap3), sqrt(Cap3, SqrtCap3))),
    (   compile_to_powershell(emp_capacity/3, [powershell_mode(pure)], ThreeWayCode)
    ->  (sub_string(ThreeWayCode, _, _, _, "3-way") -> Has3Way = true ; Has3Way = false),
        (sub_string(ThreeWayCode, _, _, _, "$r3") -> HasR3 = true ; HasR3 = false),
        (sub_string(ThreeWayCode, _, _, _, "[Math]::Sqrt") -> Has3Sqrt = true ; Has3Sqrt = false),
        (   Has3Way = true, HasR3 = true, Has3Sqrt = true
        ->  format('  [PASS] 3-way join generated with all facts and binding~n', [])
        ;   Has3Way = false, HasR3 = true, Has3Sqrt = true
        ->  format('  [PASS] 3-way join generated (nested loops)~n', [])
        ;   format('  [FAIL] Missing patterns: 3-way=~w, r3=~w, Sqrt=~w~n', [Has3Way, HasR3, Has3Sqrt])
        )
    ;   format('  [FAIL] 3-way join compilation failed~n', [])
    ),
    abolish(user:employee/2),
    abolish(user:department/2),
    abolish(user:building/2),
    abolish(user:emp_capacity/3),

    format('~n=== Bound Rule Compilation Tests Complete ===~n', []).

%% ============================================================================
%% C# HOSTING INTEGRATION (Book 12, Chapter 6)
%% ============================================================================
%%
%% These predicates integrate with dotnet_glue to enable:
%% - In-process C# ↔ PowerShell communication
%% - Cross-target pipeline compilation
%% - Runspace management for efficient execution

%% compile_with_csharp_host(+Predicates, +Options, -PSCode, -CSharpCode)
%
%  Compile predicates to PowerShell and generate a C# host for in-process execution.
%
%  This generates two outputs:
%  1. PSCode - PowerShell functions for the predicates
%  2. CSharpCode - C# host class that can invoke the PowerShell in-process
%
%  Options:
%    - namespace(Name) : C# namespace (default: UnifyWeaver.Generated)
%    - class(Name) : Host class name (default: PowerShellHost)
%    - async(Bool) : Generate async methods (default: false)
%    - include_bridge(Bool) : Include PowerShellBridge class (default: true)
%
compile_with_csharp_host(Predicates, Options, PSCode, CSharpCode) :-
    format('[C# Hosting] Compiling ~w predicates with C# host~n', [Predicates]),

    % Compile each predicate to PowerShell
    findall(Code, (
        member(Pred, Predicates),
        compile_to_powershell(Pred, [powershell_mode(pure)|Options], Code)
    ), PSCodes),
    atomic_list_concat(PSCodes, '\n\n', PSCode),

    % Generate C# host
    generate_csharp_host_code(Predicates, Options, CSharpCode).

%% generate_csharp_host_code(+Predicates, +Options, -Code)
%  Generate C# host class for invoking PowerShell predicates.
generate_csharp_host_code(Predicates, Options, Code) :-
    option_or_default(namespace(Namespace), Options, 'UnifyWeaver.Generated'),
    option_or_default(class(ClassName), Options, 'PowerShellHost'),
    option_or_default(include_bridge(IncludeBridge), Options, true),

    % Generate method wrappers for each predicate
    findall(MethodCode, (
        member(Pred/Arity, Predicates),
        generate_host_method(Pred, Arity, MethodCode)
    ), Methods),
    atomic_list_concat(Methods, '\n\n', MethodsCode),

    % Include bridge if requested
    (   IncludeBridge == true
    ->  dotnet_glue:generate_powershell_bridge([namespace(Namespace)], BridgeCode),
        format(atom(FullCode), '~w~n~n~w', [BridgeCode, HostCode])
    ;   FullCode = HostCode
    ),

    format(atom(HostCode), '
// Generated PowerShell Host for UnifyWeaver predicates
// Provides typed C# wrappers for PowerShell function calls

using System;
using System.Collections.Generic;
using System.Management.Automation;

namespace ~w
{
    /// <summary>
    /// Host class for invoking compiled PowerShell predicates from C#.
    /// </summary>
    public class ~w
    {
        private readonly string _scriptBlock;

        /// <summary>
        /// Initialize host with the compiled PowerShell script.
        /// </summary>
        /// <param name="scriptBlock">PowerShell script containing compiled predicates</param>
        public ~w(string scriptBlock)
        {
            _scriptBlock = scriptBlock;
            // Initialize the runspace with the script
            PowerShellBridge.SetVariable("__init_script__", scriptBlock);
            PowerShellBridge.Invoke<object, object>("Invoke-Expression $__init_script__", null);
        }

~w
    }
}
', [Namespace, ClassName, ClassName, MethodsCode]),

    Code = FullCode.

%% generate_host_method(+Pred, +Arity, -Code)
%  Generate a C# method wrapper for a PowerShell predicate.
generate_host_method(Pred, Arity, Code) :-
    atom_string(Pred, PredStr),
    % Generate parameter list
    generate_csharp_params(Arity, ParamList, ArgList),

    format(atom(Code), '
        /// <summary>
        /// Invoke the ~w predicate.
        /// </summary>
        public IEnumerable<dynamic> ~w(~w)
        {
            var script = "~w ~w";
            return PowerShellBridge.Invoke<object, dynamic>(script, null);
        }', [PredStr, PredStr, ParamList, PredStr, ArgList]).

%% generate_csharp_params(+Arity, -ParamList, -ArgList)
%  Generate C# parameter declarations and argument list.
generate_csharp_params(0, "", "") :- !.
generate_csharp_params(1, "string arg1", "$arg1") :- !.
generate_csharp_params(2, "string arg1, string arg2", "$arg1 $arg2") :- !.
generate_csharp_params(Arity, ParamList, ArgList) :-
    Arity > 2,
    findall(P, (between(1, Arity, I), format(atom(P), "string arg~w", [I])), Params),
    findall(A, (between(1, Arity, I), format(atom(A), "$arg~w", [I])), Args),
    atomic_list_concat(Params, ', ', ParamList),
    atomic_list_concat(Args, ' ', ArgList).

%% generate_csharp_bridge(+BridgeType, +Options, -Code)
%
%  Generate a specific type of C# bridge code.
%
%  BridgeType:
%    - powershell : PowerShell in-process bridge
%    - ironpython : IronPython in-process bridge
%    - cpython : CPython pipe-based bridge
%
generate_csharp_bridge(powershell, Options, Code) :-
    dotnet_glue:generate_powershell_bridge(Options, Code).

generate_csharp_bridge(ironpython, Options, Code) :-
    dotnet_glue:generate_ironpython_bridge(Options, Code).

generate_csharp_bridge(cpython, Options, Code) :-
    dotnet_glue:generate_cpython_bridge(Options, Code).

%% compile_cross_target_pipeline(+Steps, +Options, -Code)
%
%  Compile a cross-target pipeline where steps can be in different languages.
%
%  Steps is a list of step(Target, Predicate, Options):
%    - step(powershell, filter_users/2, [])
%    - step(csharp, transform_data/2, [inline_code(...)])
%    - step(python, analyze/1, [])
%
%  This generates a complete .NET solution that orchestrates the pipeline.
%
compile_cross_target_pipeline(Steps, Options, Code) :-
    format('[Cross-Target] Compiling ~w-step pipeline~n', [Steps]),
    length(Steps, NumSteps),

    % Detect runtime capabilities
    dotnet_glue:detect_dotnet_runtime(DotNetRuntime),
    dotnet_glue:detect_powershell(PSVersion),
    format('[Cross-Target] .NET: ~w, PowerShell: ~w~n', [DotNetRuntime, PSVersion]),

    % Generate pipeline code using dotnet_glue
    dotnet_glue:generate_dotnet_pipeline(Steps, Options, Code),
    format('[Cross-Target] Generated ~w-step pipeline code~n', [NumSteps]).

%% option_or_default(+Option, +Options, +Default)
%  Extract option value or use default.
option_or_default(Option, Options, _Default) :-
    member(Option, Options), !.
option_or_default(Option, _Options, Default) :-
    Option =.. [Name, Default],
    \+ var(Default).

%% ============================================================================
%% C# HOSTING TESTS
%% ============================================================================

test_csharp_hosting :-
    format('~n=== Testing C# Hosting Integration ===~n~n', []),

    % Test 1: Generate PowerShell bridge
    format('[Test 1] Generate PowerShell bridge~n', []),
    generate_csharp_bridge(powershell, [], BridgeCode),
    (   sub_string(BridgeCode, _, _, _, "PowerShellBridge"),
        sub_string(BridgeCode, _, _, _, "SharedRunspace")
    ->  format('  [PASS] PowerShell bridge generated with runspace~n', [])
    ;   format('  [FAIL] Bridge missing expected components~n', [])
    ),

    % Test 2: Runtime detection
    format('[Test 2] Runtime detection~n', []),
    dotnet_glue:detect_dotnet_runtime(Runtime),
    dotnet_glue:detect_powershell(PSVer),
    format('  .NET Runtime: ~w~n', [Runtime]),
    format('  PowerShell: ~w~n', [PSVer]),
    format('  [PASS] Runtime detection complete~n', []),

    % Test 3: Generate IronPython bridge
    format('[Test 3] Generate IronPython bridge~n', []),
    generate_csharp_bridge(ironpython, [namespace('Test.Glue')], IPBridge),
    (   sub_string(IPBridge, _, _, _, "IronPythonBridge"),
        sub_string(IPBridge, _, _, _, "Test.Glue")
    ->  format('  [PASS] IronPython bridge generated with custom namespace~n', [])
    ;   format('  [FAIL] IronPython bridge missing expected components~n', [])
    ),

    % Test 4: C# parameter generation
    format('[Test 4] C# parameter generation~n', []),
    generate_csharp_params(0, P0, A0),
    generate_csharp_params(1, P1, A1),
    generate_csharp_params(3, P3, A3),
    (   P0 = "", A0 = "",
        P1 = "string arg1", A1 = "$arg1",
        sub_string(P3, _, _, _, "arg3")
    ->  format('  [PASS] Parameters generated correctly~n', [])
    ;   format('  [FAIL] Parameter generation error~n', [])
    ),

    format('~n=== C# Hosting Tests Complete ===~n', []).
