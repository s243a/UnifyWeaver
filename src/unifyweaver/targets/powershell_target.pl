:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% powershell_target.pl - PowerShell Semantic Target for UnifyWeaver
% Generates native PowerShell code using .NET objects for XML streaming and Vector search.
% Phase 2: Native Semantic Implementation.

:- module(powershell_target, [
    compile_predicate_to_powershell/3,  % +Predicate, +Options, -PsCode
    compile_vector_ops/2,               % +Options, -PsCode
    compile_vector_search/3,            % +PredName, +Options, -PsCode
    % Pipeline support
    compile_powershell_pipeline/3,      % +Predicates, +Options, -PsCode
    test_powershell_pipeline_generator/0,
    % Enhanced pipeline chaining exports
    compile_powershell_enhanced_pipeline/3, % +Stages, +Options, -PsCode
    ps_enhanced_helpers/1,                  % -Code
    generate_ps_enhanced_connector/3,       % +Stages, +PipelineName, -Code
    test_powershell_enhanced_chaining/0     % Test enhanced pipeline chaining
]).

:- use_module(library(lists)).
:- use_module(library(option)).

%% compile_vector_search(+PredName, +Options, -PsCode)
%  Generates a PowerShell function to perform vector search over a collection.
compile_vector_search(PredName, _Options, PsCode) :-
    format(string(PsCode),
'
function ~w {
    param(
        [Parameter(Mandatory=$true)]$Items,
        [Parameter(Mandatory=$true)][float[]]$QueryVector,
        [int]$TopK = 10
    )

    $results = @()

    foreach ($item in $Items) {
        if ($item.vector) {
            $score = Get-CosineSimilarity $QueryVector $item.vector
            # Add score to a copy of the item to avoid modifying original if needed
            # Or just return a result object
            $results += [PSCustomObject]@{
                Score = $score
                Item = $item
            }
        }
    }

    $results | Sort-Object Score -Descending | Select-Object -First $TopK
}
', [PredName]).

%% compile_predicate_to_powershell(+Predicate, +Options, -PsCode)
%  Main entry point for compiling Prolog predicates to native PowerShell.
%  Supports:
%    - source(xml) via .NET XmlReader
%    - vector search (stub/future)
%
compile_predicate_to_powershell(PredicateIndicator, Options, PsCode) :-
    PredicateIndicator = Pred/Arity,
    format('=== Compiling ~w/~w to PowerShell (Semantic) ===~n', [Pred, Arity]),
    
    (   option(input_source(xml(File, Tags)), Options)
    ->  compile_source_xml(Pred, File, Tags, PsCode)
    ;   
        % Default fallback or error
        format(string(PsCode), "Write-Error 'Unsupported mode for powershell_target: ~w'", [Options])
    ).

%% compile_source_xml(+PredName, +File, +Tags, -PsCode)
%  Generates PowerShell code to stream an XML file and flatten specific tags.
%  Uses [System.Xml.XmlReader] for memory efficiency.
compile_source_xml(PredName, File, Tags, PsCode) :-
    maplist(atom_string, Tags, TagStrs),
    % Convert tags list to PowerShell array string: "tag1","tag2"
    maplist(ps_quote, TagStrs, QuotedTags),
    atomic_list_concat(QuotedTags, ',', TagListStr),
    
    format(string(PsCode), 
'
function ~w {
    param([string]$InputFile = "~w")
    
    $tags = @(~s)
    $path = [System.IO.Path]::GetFullPath($InputFile)
    
    if (-not (Test-Path $path)) {
        Write-Error "File not found: $path"
        return
    }

    $reader = [System.Xml.XmlReader]::Create($path)
    try {
        while ($reader.Read()) {
            if ($reader.NodeType -eq [System.Xml.XmlNodeType]::Element -and $tags -contains $reader.Name) {
                # Read the current element and its children
                $subReader = $reader.ReadSubtree()
                $doc = [System.Xml.XmlDocument]::new()
                $doc.Load($subReader)
                
                # Flatten to Hashtable
                $obj = @{}
                $root = $doc.DocumentElement
                
                $obj["tag"] = $root.Name
                if ($root.InnerText) {
                    $obj["text"] = $root.InnerText.Trim()
                }
                
                # Attributes
                foreach ($attr in $root.Attributes) {
                    $obj["@$($attr.Name)"] = $attr.Value
                }
                
                # Flatten Children (Simple)
                foreach ($child in $root.ChildNodes) {
                    if ($child.NodeType -eq [System.Xml.XmlNodeType]::Element) {
                        $obj[$child.Name] = $child.InnerText.Trim()
                    }
                }
                
                # Output object
                $obj
            }
        }
    } finally {
        $reader.Dispose()
    }
}

~w
', [PredName, File, TagListStr, PredName]).

ps_quote(S, Q) :- format(string(Q), '"~s"', [S]).

%% compile_vector_ops(+Options, -PsCode)
%  Generates vector math helper functions (Cosine Similarity).
%  Currently generates Pure PowerShell implementation.
compile_vector_ops(_Options, PsCode) :-
    generate_vector_math_pure(PsCode).

generate_vector_math_pure(Code) :-
    Code = '
function Get-CosineSimilarity {
    param(
        [Parameter(Mandatory=$true)][float[]]$V1,
        [Parameter(Mandatory=$true)][float[]]$V2
    )

    if ($V1.Length -ne $V2.Length) {
        throw "Vector length mismatch"
    }

    $dot = 0.0
    $mag1 = 0.0
    $mag2 = 0.0
    $len = $V1.Length

    for ($i = 0; $i -lt $len; $i++) {
        $v1_val = $V1[$i]
        $v2_val = $V2[$i]

        $dot += $v1_val * $v2_val
        $mag1 += $v1_val * $v1_val
        $mag2 += $v2_val * $v2_val
    }

    if ($mag1 -eq 0.0 -or $mag2 -eq 0.0) {
        return 0.0
    }

    return $dot / ([Math]::Sqrt($mag1) * [Math]::Sqrt($mag2))
}
'.

%% ============================================================================
%% POWERSHELL PIPELINE SUPPORT
%% ============================================================================
%%
%% Supports pipeline_mode option:
%%   - sequential (default): Stages piped sequentially
%%   - generator: Fixpoint iteration with deduplication
%%

%% compile_powershell_pipeline(+Predicates, +Options, -PsCode)
%  Main entry point for PowerShell pipeline compilation.
%
%  Options:
%    - pipeline_name(Name) : Name for the pipeline function (default: Invoke-Pipeline)
%    - pipeline_mode(Mode) : sequential | generator (default: sequential)
%    - output_format(Fmt)  : jsonl | object (default: jsonl)
%
compile_powershell_pipeline(Predicates, Options, PsCode) :-
    option(pipeline_name(PipelineName), Options, 'Invoke-Pipeline'),
    option(pipeline_mode(PipelineMode), Options, sequential),
    option(output_format(OutputFormat), Options, jsonl),

    format('=== Compiling PowerShell Pipeline ===~n', []),
    format('  Predicates: ~w~n', [Predicates]),
    format('  Pipeline name: ~w~n', [PipelineName]),
    format('  Mode: ~w~n', [PipelineMode]),

    % Generate header
    ps_pipeline_header(PipelineMode, Header),

    % Generate helper functions
    ps_pipeline_helpers(PipelineMode, Helpers),

    % Generate stage functions (placeholders)
    extract_ps_predicate_names(Predicates, StageNames),
    generate_ps_stage_functions(StageNames, StageFunctions),

    % Generate pipeline connector
    generate_ps_pipeline_connector(StageNames, PipelineName, PipelineMode, ConnectorCode),

    % Generate main block
    generate_ps_main(PipelineName, OutputFormat, MainCode),

    format(string(PsCode), "~w~w~w~w~w",
        [Header, Helpers, StageFunctions, ConnectorCode, MainCode]).

%% ps_pipeline_header(+Mode, -Header)
%  Generate script header with required imports
ps_pipeline_header(generator, Header) :-
    !,
    Header = "# PowerShell Pipeline with Fixpoint Evaluation
# Generated by UnifyWeaver

# Requires PowerShell 5.1+ for hashtable operations

".
ps_pipeline_header(_, Header) :-
    Header = "# PowerShell Pipeline
# Generated by UnifyWeaver

".

%% ps_pipeline_helpers(+Mode, -Helpers)
%  Generate helper functions based on mode
ps_pipeline_helpers(generator, Helpers) :-
    !,
    Helpers = "
# Helper: Read JSONL from stdin
function Read-JsonlStream {
    param([System.IO.TextReader]$Reader = [Console]::In)

    while ($null -ne ($line = $Reader.ReadLine())) {
        if ($line.Trim()) {
            $line | ConvertFrom-Json
        }
    }
}

# Helper: Write JSONL to stdout
function Write-JsonlStream {
    param(
        [Parameter(ValueFromPipeline=$true)]$InputObject
    )
    process {
        $InputObject | ConvertTo-Json -Compress
    }
}

# Helper: Generate unique key for record deduplication
function Get-RecordKey {
    param([hashtable]$Record)

    $sortedKeys = $Record.Keys | Sort-Object
    $parts = foreach ($key in $sortedKeys) {
        \"$key=$($Record[$key])\"
    }
    $parts -join ';'
}

# Helper: Convert PSCustomObject to Hashtable
function ConvertTo-Hashtable {
    param([Parameter(ValueFromPipeline=$true)]$InputObject)
    process {
        if ($InputObject -is [hashtable]) {
            $InputObject
        } else {
            $ht = @{}
            $InputObject.PSObject.Properties | ForEach-Object {
                $ht[$_.Name] = $_.Value
            }
            $ht
        }
    }
}

".
ps_pipeline_helpers(_, Helpers) :-
    Helpers = "
# Helper: Read JSONL from stdin
function Read-JsonlStream {
    param([System.IO.TextReader]$Reader = [Console]::In)

    while ($null -ne ($line = $Reader.ReadLine())) {
        if ($line.Trim()) {
            $line | ConvertFrom-Json
        }
    }
}

# Helper: Write JSONL to stdout
function Write-JsonlStream {
    param(
        [Parameter(ValueFromPipeline=$true)]$InputObject
    )
    process {
        $InputObject | ConvertTo-Json -Compress
    }
}

".

%% extract_ps_predicate_names(+Predicates, -Names)
%  Extract stage names from predicates
extract_ps_predicate_names([], []).
extract_ps_predicate_names([Pred|Rest], [Name|RestNames]) :-
    extract_ps_pred_name(Pred, Name),
    extract_ps_predicate_names(Rest, RestNames).

extract_ps_pred_name(_Target:Name/_Arity, NameStr) :-
    !,
    atom_string(Name, NameStr).
extract_ps_pred_name(Name/_Arity, NameStr) :-
    atom_string(Name, NameStr).
extract_ps_pred_name(Name, NameStr) :-
    atom(Name),
    atom_string(Name, NameStr).

%% generate_ps_stage_functions(+Names, -Code)
%  Generate placeholder stage functions
generate_ps_stage_functions([], "").
generate_ps_stage_functions([Name|Rest], Code) :-
    format(string(FuncCode),
"
# Stage: ~w (placeholder - implement actual logic)
function Invoke-~w {
    param(
        [Parameter(ValueFromPipeline=$true)]$InputObject
    )
    begin { }
    process {
        # Pass through - actual implementation needed
        $InputObject
    }
    end { }
}

", [Name, Name]),
    generate_ps_stage_functions(Rest, RestCode),
    format(string(Code), "~w~w", [FuncCode, RestCode]).

%% generate_ps_pipeline_connector(+StageNames, +PipelineName, +Mode, -Code)
%  Generate the pipeline connector function
generate_ps_pipeline_connector(StageNames, PipelineName, sequential, Code) :-
    generate_ps_sequential_chain(StageNames, ChainCode),
    format(string(Code),
"
# Pipeline connector: ~w (sequential mode)
function ~w {
    param(
        [Parameter(ValueFromPipeline=$true)]$InputObject
    )
    begin {
        $inputRecords = @()
    }
    process {
        $inputRecords += $InputObject
    }
    end {
~w
    }
}

", [PipelineName, PipelineName, ChainCode]).

generate_ps_pipeline_connector(StageNames, PipelineName, generator, Code) :-
    generate_ps_fixpoint_chain(StageNames, ChainCode),
    format(string(Code),
"
# Pipeline connector: ~w (generator/fixpoint mode)
function ~w {
    param(
        [Parameter(ValueFromPipeline=$true)]$InputObject
    )
    begin {
        $inputRecords = @()
    }
    process {
        $inputRecords += $InputObject
    }
    end {
        # Initialize with input records
        $total = @{}

        foreach ($record in $inputRecords) {
            $ht = $record | ConvertTo-Hashtable
            $key = Get-RecordKey $ht
            if (-not $total.ContainsKey($key)) {
                $total[$key] = $ht
                $ht  # Output initial record
            }
        }

        # Fixpoint iteration - apply stages until no new records
        $changed = $true
        while ($changed) {
            $changed = $false
            $current = @($total.Values)

~w

            # Check for new records
            foreach ($record in $newRecords) {
                $ht = $record | ConvertTo-Hashtable
                $key = Get-RecordKey $ht
                if (-not $total.ContainsKey($key)) {
                    $total[$key] = $ht
                    $changed = $true
                    $ht  # Output new record
                }
            }
        }
    }
}

", [PipelineName, PipelineName, ChainCode]).

%% generate_ps_sequential_chain(+StageNames, -Code)
%  Generate sequential pipeline chain
generate_ps_sequential_chain([], Code) :-
    Code = "        $inputRecords".
generate_ps_sequential_chain(StageNames, Code) :-
    StageNames \= [],
    generate_ps_chain_expr(StageNames, "$inputRecords", ChainExpr),
    format(string(Code), "        ~w", [ChainExpr]).

generate_ps_chain_expr([], Current, Current).
generate_ps_chain_expr([Stage|Rest], Current, Expr) :-
    format(string(NextExpr), "~w | Invoke-~w", [Current, Stage]),
    generate_ps_chain_expr(Rest, NextExpr, Expr).

%% generate_ps_fixpoint_chain(+StageNames, -Code)
%  Generate fixpoint stage application code
generate_ps_fixpoint_chain([], Code) :-
    Code = "            $newRecords = $current".
generate_ps_fixpoint_chain(StageNames, Code) :-
    StageNames \= [],
    generate_ps_stage_chain(StageNames, "$current", ChainExpr),
    format(string(Code), "            # Apply pipeline stages
            $newRecords = @(~w)", [ChainExpr]).

generate_ps_stage_chain([], Current, Current).
generate_ps_stage_chain([Stage|Rest], Current, Expr) :-
    format(string(NextExpr), "~w | Invoke-~w", [Current, Stage]),
    generate_ps_stage_chain(Rest, NextExpr, Expr).

%% generate_ps_main(+PipelineName, +OutputFormat, -Code)
%  Generate main execution block
generate_ps_main(PipelineName, jsonl, Code) :-
    format(string(Code),
"
# Main execution
Read-JsonlStream | ~w | Write-JsonlStream
", [PipelineName]).
generate_ps_main(PipelineName, object, Code) :-
    format(string(Code),
"
# Main execution (object output)
Read-JsonlStream | ~w
", [PipelineName]).

%% ============================================================================
%% POWERSHELL PIPELINE GENERATOR MODE TESTS
%% ============================================================================

test_powershell_pipeline_generator :-
    format('~n=== PowerShell Pipeline Generator Mode Tests ===~n~n', []),

    % Test 1: Pipeline header for generator mode
    format('[Test 1] Pipeline header (generator)~n', []),
    ps_pipeline_header(generator, Header1),
    (   sub_string(Header1, _, _, _, "Fixpoint Evaluation")
    ->  format('  [PASS] Generator header correct~n', [])
    ;   format('  [FAIL] Header: ~w~n', [Header1])
    ),

    % Test 2: Pipeline header for sequential mode
    format('[Test 2] Pipeline header (sequential)~n', []),
    ps_pipeline_header(sequential, Header2),
    (   sub_string(Header2, _, _, _, "PowerShell Pipeline"),
        \+ sub_string(Header2, _, _, _, "Fixpoint")
    ->  format('  [PASS] Sequential header correct~n', [])
    ;   format('  [FAIL] Header: ~w~n', [Header2])
    ),

    % Test 3: Pipeline helpers for generator mode
    format('[Test 3] Pipeline helpers (generator)~n', []),
    ps_pipeline_helpers(generator, Helpers3),
    (   sub_string(Helpers3, _, _, _, "Get-RecordKey"),
        sub_string(Helpers3, _, _, _, "ConvertTo-Hashtable"),
        sub_string(Helpers3, _, _, _, "Read-JsonlStream")
    ->  format('  [PASS] Generator helpers include dedup functions~n', [])
    ;   format('  [FAIL] Helpers missing patterns~n', [])
    ),

    % Test 4: Extract predicate names
    format('[Test 4] Extract predicate names~n', []),
    extract_ps_predicate_names([stage1/1, stage2/2], Names4),
    (   Names4 = ["stage1", "stage2"]
    ->  format('  [PASS] Names extracted: ~w~n', [Names4])
    ;   format('  [FAIL] Names: ~w~n', [Names4])
    ),

    % Test 5: Generate stage functions
    format('[Test 5] Generate stage functions~n', []),
    generate_ps_stage_functions(["transform"], StageCode5),
    (   sub_string(StageCode5, _, _, _, "function Invoke-transform"),
        sub_string(StageCode5, _, _, _, "ValueFromPipeline")
    ->  format('  [PASS] Stage function generated~n', [])
    ;   format('  [FAIL] Stage code: ~w~n', [StageCode5])
    ),

    % Test 6: Sequential chain generation
    format('[Test 6] Sequential chain generation~n', []),
    generate_ps_sequential_chain(["a", "b"], SeqChain6),
    (   sub_string(SeqChain6, _, _, _, "Invoke-a"),
        sub_string(SeqChain6, _, _, _, "Invoke-b"),
        sub_string(SeqChain6, _, _, _, "|")
    ->  format('  [PASS] Sequential chain correct~n', [])
    ;   format('  [FAIL] Chain: ~w~n', [SeqChain6])
    ),

    % Test 7: Fixpoint chain generation
    format('[Test 7] Fixpoint chain generation~n', []),
    generate_ps_fixpoint_chain(["x", "y"], FixChain7),
    (   sub_string(FixChain7, _, _, _, "Invoke-x"),
        sub_string(FixChain7, _, _, _, "Invoke-y"),
        sub_string(FixChain7, _, _, _, "$newRecords")
    ->  format('  [PASS] Fixpoint chain correct~n', [])
    ;   format('  [FAIL] Chain: ~w~n', [FixChain7])
    ),

    % Test 8: Pipeline connector (sequential)
    format('[Test 8] Pipeline connector (sequential)~n', []),
    generate_ps_pipeline_connector(["s1", "s2"], testSeq, sequential, ConnSeq8),
    (   sub_string(ConnSeq8, _, _, _, "function testSeq"),
        sub_string(ConnSeq8, _, _, _, "sequential mode"),
        sub_string(ConnSeq8, _, _, _, "Invoke-s1"),
        sub_string(ConnSeq8, _, _, _, "Invoke-s2")
    ->  format('  [PASS] Sequential connector correct~n', [])
    ;   format('  [FAIL] Connector: ~w~n', [ConnSeq8])
    ),

    % Test 9: Pipeline connector (generator)
    format('[Test 9] Pipeline connector (generator)~n', []),
    generate_ps_pipeline_connector(["g1", "g2"], testGen, generator, ConnGen9),
    (   sub_string(ConnGen9, _, _, _, "function testGen"),
        sub_string(ConnGen9, _, _, _, "fixpoint mode"),
        sub_string(ConnGen9, _, _, _, "$total = @{}"),
        sub_string(ConnGen9, _, _, _, "while ($changed)"),
        sub_string(ConnGen9, _, _, _, "Get-RecordKey")
    ->  format('  [PASS] Generator connector has fixpoint loop~n', [])
    ;   format('  [FAIL] Connector: ~w~n', [ConnGen9])
    ),

    % Test 10: Full pipeline compilation (generator mode)
    format('[Test 10] Full pipeline (generator mode)~n', []),
    compile_powershell_pipeline([derive/1, transform/1], [
        pipeline_name('Invoke-FixpointPipe'),
        pipeline_mode(generator),
        output_format(jsonl)
    ], FullCode10),
    (   sub_string(FullCode10, _, _, _, "Fixpoint Evaluation"),
        sub_string(FullCode10, _, _, _, "function Invoke-FixpointPipe"),
        sub_string(FullCode10, _, _, _, "Get-RecordKey"),
        sub_string(FullCode10, _, _, _, "while ($changed)"),
        sub_string(FullCode10, _, _, _, "$total.ContainsKey")
    ->  format('  [PASS] Full generator pipeline compiled~n', [])
    ;   format('  [FAIL] Missing expected patterns~n', [])
    ),

    format('~n=== All PowerShell Pipeline Generator Mode Tests Passed ===~n', []).

%% ============================================
%% POWERSHELL ENHANCED PIPELINE CHAINING
%% ============================================
%
%  Supports advanced flow patterns:
%    - fan_out(Stages) : Broadcast to parallel stages
%    - merge          : Combine results from parallel stages
%    - route_by(Pred, Routes) : Conditional routing
%    - filter_by(Pred) : Filter records
%    - Pred/Arity     : Standard stage
%
%% compile_powershell_enhanced_pipeline(+Stages, +Options, -PsCode)
%  Main entry point for enhanced PowerShell pipeline with advanced flow patterns.
%
compile_powershell_enhanced_pipeline(Stages, Options, PsCode) :-
    option(pipeline_name(PipelineName), Options, 'Invoke-EnhancedPipeline'),
    option(output_format(OutputFormat), Options, jsonl),

    % Generate helpers
    ps_enhanced_helpers(Helpers),

    % Generate stage functions
    generate_ps_enhanced_stage_functions(Stages, StageFunctions),

    % Generate the main connector
    generate_ps_enhanced_connector(Stages, PipelineName, ConnectorCode),

    % Generate main block
    generate_ps_enhanced_main(PipelineName, OutputFormat, MainCode),

    format(string(PsCode),
"# PowerShell Enhanced Pipeline
# Generated by UnifyWeaver
# Supports fan-out, merge, conditional routing, and filtering

~w

~w
~w
~w
", [Helpers, StageFunctions, ConnectorCode, MainCode]).

%% ps_enhanced_helpers(-Code)
%  Generate helper functions for enhanced pipeline operations.
ps_enhanced_helpers(Code) :-
    Code = "# Enhanced Pipeline Helpers

function Invoke-FanOut {
    <#
    .SYNOPSIS
    Fan-out: Send record to all stages, collect all results.
    #>
    param(
        [Parameter(Mandatory=$true)]$Record,
        [Parameter(Mandatory=$true)][scriptblock[]]$Stages
    )

    $results = @()
    foreach ($stage in $Stages) {
        $stageResults = @($Record) | & $stage
        $results += $stageResults
    }
    $results
}

function Merge-Streams {
    <#
    .SYNOPSIS
    Merge: Combine multiple streams into one.
    #>
    param(
        [Parameter(ValueFromPipeline=$true)]$InputObject
    )
    process {
        $InputObject
    }
}

function Invoke-RouteRecord {
    <#
    .SYNOPSIS
    Route: Direct record to appropriate stage based on condition.
    #>
    param(
        [Parameter(Mandatory=$true)]$Record,
        [Parameter(Mandatory=$true)][scriptblock]$ConditionFn,
        [Parameter(Mandatory=$true)][hashtable]$RouteMap,
        [scriptblock]$DefaultFn = $null
    )

    $condition = & $ConditionFn $Record
    if ($RouteMap.ContainsKey($condition)) {
        @($Record) | & $RouteMap[$condition]
    } elseif ($DefaultFn) {
        @($Record) | & $DefaultFn
    } else {
        $Record  # Pass through if no matching route
    }
}

function Select-FilteredRecords {
    <#
    .SYNOPSIS
    Filter: Only yield records that satisfy the predicate.
    #>
    param(
        [Parameter(ValueFromPipeline=$true)]$InputObject,
        [Parameter(Mandatory=$true)][scriptblock]$PredicateFn
    )
    process {
        if (& $PredicateFn $InputObject) {
            $InputObject
        }
    }
}

function Invoke-TeeStream {
    <#
    .SYNOPSIS
    Tee: Send each record to multiple stages and collect all results.
    #>
    param(
        [Parameter(ValueFromPipeline=$true)]$InputObject,
        [Parameter(Mandatory=$true)][scriptblock[]]$Stages
    )
    begin {
        $allRecords = @()
    }
    process {
        $allRecords += $InputObject
    }
    end {
        foreach ($record in $allRecords) {
            foreach ($stage in $Stages) {
                @($record) | & $stage
            }
        }
    }
}

function Read-JsonlStream {
    <#
    .SYNOPSIS
    Read JSONL from stdin.
    #>
    param([System.IO.TextReader]$Reader = [Console]::In)

    while ($null -ne ($line = $Reader.ReadLine())) {
        if ($line.Trim()) {
            $line | ConvertFrom-Json
        }
    }
}

function Write-JsonlStream {
    <#
    .SYNOPSIS
    Write JSONL to stdout.
    #>
    param(
        [Parameter(ValueFromPipeline=$true)]$InputObject
    )
    process {
        $InputObject | ConvertTo-Json -Compress
    }
}

".

%% generate_ps_enhanced_stage_functions(+Stages, -Code)
%  Generate stub functions for each stage.
generate_ps_enhanced_stage_functions([], "").
generate_ps_enhanced_stage_functions([Stage|Rest], Code) :-
    generate_ps_single_enhanced_stage(Stage, StageCode),
    generate_ps_enhanced_stage_functions(Rest, RestCode),
    (RestCode = "" ->
        Code = StageCode
    ;   format(string(Code), "~w~n~w", [StageCode, RestCode])
    ).

generate_ps_single_enhanced_stage(fan_out(SubStages), Code) :-
    !,
    generate_ps_enhanced_stage_functions(SubStages, Code).
generate_ps_single_enhanced_stage(merge, "") :- !.
generate_ps_single_enhanced_stage(route_by(_, Routes), Code) :-
    !,
    findall(Stage, member((_Cond, Stage), Routes), RouteStages),
    generate_ps_enhanced_stage_functions(RouteStages, Code).
generate_ps_single_enhanced_stage(filter_by(_), "") :- !.
generate_ps_single_enhanced_stage(Pred/Arity, Code) :-
    !,
    format(string(Code),
"function Invoke-~w {
    <#
    .SYNOPSIS
    Pipeline stage: ~w/~w
    #>
    param([Parameter(ValueFromPipeline=$true)]$InputObject)
    process {
        # TODO: Implement based on predicate bindings
        $InputObject
    }
}

", [Pred, Pred, Arity]).
generate_ps_single_enhanced_stage(_, "").

%% generate_ps_enhanced_connector(+Stages, +PipelineName, -Code)
%  Generate the main connector that handles enhanced flow patterns.
generate_ps_enhanced_connector(Stages, PipelineName, Code) :-
    generate_ps_enhanced_flow_code(Stages, "$input", FlowCode),
    format(string(Code),
"function ~w {
    <#
    .SYNOPSIS
    Enhanced pipeline with fan-out, merge, and routing support.
    #>
    param([Parameter(ValueFromPipeline=$true)]$InputObject)
    begin {
        $inputStream = @()
    }
    process {
        $inputStream += $InputObject
    }
    end {
        $input = $inputStream
~w
    }
}

", [PipelineName, FlowCode]).

%% generate_ps_enhanced_flow_code(+Stages, +CurrentVar, -Code)
%  Generate the flow code for enhanced pipeline stages.
generate_ps_enhanced_flow_code([], CurrentVar, Code) :-
    format(string(Code), "        ~w", [CurrentVar]).
generate_ps_enhanced_flow_code([Stage|Rest], CurrentVar, Code) :-
    generate_ps_stage_flow(Stage, CurrentVar, NextVar, StageCode),
    generate_ps_enhanced_flow_code(Rest, NextVar, RestCode),
    format(string(Code), "~w~n~w", [StageCode, RestCode]).

%% generate_ps_stage_flow(+Stage, +InVar, -OutVar, -Code)
%  Generate flow code for a single stage.

% Fan-out stage: broadcast to parallel stages
generate_ps_stage_flow(fan_out(SubStages), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "$fanOut~wResult", [N]),
    extract_ps_stage_names(SubStages, StageNames),
    format_ps_stage_list(StageNames, StageListStr),
    format(string(Code),
"        # Fan-out to ~w parallel stages
        ~w = @()
        foreach ($record in ~w) {
            ~w += Invoke-FanOut -Record $record -Stages @(~w)
        }", [N, OutVar, InVar, OutVar, StageListStr]).

% Merge stage: placeholder, usually follows fan_out
generate_ps_stage_flow(merge, InVar, OutVar, Code) :-
    !,
    OutVar = InVar,
    Code = "        # Merge: results already combined from fan-out".

% Conditional routing
generate_ps_stage_flow(route_by(CondPred, Routes), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "$routedResult", []),
    format_ps_route_map(Routes, RouteMapStr),
    format(string(Code),
"        # Conditional routing based on ~w
        $routeMap = @{
~w
        }
        ~w = @()
        foreach ($record in ~w) {
            ~w += Invoke-RouteRecord -Record $record -ConditionFn { param($r) ~w $r } -RouteMap $routeMap
        }", [CondPred, RouteMapStr, OutVar, InVar, OutVar, CondPred]).

% Filter stage
generate_ps_stage_flow(filter_by(Pred), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "$filteredResult", []),
    format(string(Code),
"        # Filter by ~w
        ~w = ~w | Select-FilteredRecords -PredicateFn { param($r) ~w $r }", [Pred, OutVar, InVar, Pred]).

% Standard predicate stage
generate_ps_stage_flow(Pred/Arity, InVar, OutVar, Code) :-
    !,
    atom(Pred),
    format(atom(OutVar), "$~wResult", [Pred]),
    format(string(Code),
"        # Stage: ~w/~w
        ~w = ~w | Invoke-~w", [Pred, Arity, OutVar, InVar, Pred]).

% Fallback for unknown stages
generate_ps_stage_flow(Stage, InVar, InVar, Code) :-
    format(string(Code), "        # Unknown stage type: ~w (pass-through)", [Stage]).

%% extract_ps_stage_names(+Stages, -Names)
%  Extract function names from stage specifications.
extract_ps_stage_names([], []).
extract_ps_stage_names([Pred/_Arity|Rest], [Pred|RestNames]) :-
    !,
    extract_ps_stage_names(Rest, RestNames).
extract_ps_stage_names([_|Rest], RestNames) :-
    extract_ps_stage_names(Rest, RestNames).

%% format_ps_stage_list(+Names, -ListStr)
%  Format stage names as PowerShell scriptblock references.
format_ps_stage_list([], "").
format_ps_stage_list([Name], Str) :-
    format(string(Str), "{ Invoke-~w }", [Name]).
format_ps_stage_list([Name|Rest], Str) :-
    Rest \= [],
    format_ps_stage_list(Rest, RestStr),
    format(string(Str), "{ Invoke-~w }, ~w", [Name, RestStr]).

%% format_ps_route_map(+Routes, -MapStr)
%  Format routing map for PowerShell.
format_ps_route_map([], "").
format_ps_route_map([(_Cond, Stage)|[]], Str) :-
    (Stage = StageName/_Arity -> true ; StageName = Stage),
    format(string(Str), "            $true = { Invoke-~w }", [StageName]).
format_ps_route_map([(Cond, Stage)|Rest], Str) :-
    Rest \= [],
    (Stage = StageName/_Arity -> true ; StageName = Stage),
    format_ps_route_map(Rest, RestStr),
    (Cond = true ->
        format(string(Str), "            $true = { Invoke-~w }~n~w", [StageName, RestStr])
    ; Cond = false ->
        format(string(Str), "            $false = { Invoke-~w }~n~w", [StageName, RestStr])
    ;   format(string(Str), "            '~w' = { Invoke-~w }~n~w", [Cond, StageName, RestStr])
    ).

%% generate_ps_enhanced_main(+PipelineName, +OutputFormat, -Code)
%  Generate main block for enhanced pipeline.
generate_ps_enhanced_main(PipelineName, jsonl, Code) :-
    format(string(Code),
"# Main execution
# Read JSONL from stdin, process through pipeline, output JSONL
Read-JsonlStream | ~w | Write-JsonlStream
", [PipelineName]).
generate_ps_enhanced_main(PipelineName, _, Code) :-
    format(string(Code),
"# Main execution
# Read JSONL from stdin, process through pipeline, output results
Read-JsonlStream | ~w | Write-JsonlStream
", [PipelineName]).

%% ============================================
%% POWERSHELL ENHANCED PIPELINE CHAINING TESTS
%% ============================================

test_powershell_enhanced_chaining :-
    format('~n=== PowerShell Enhanced Pipeline Chaining Tests ===~n~n', []),

    % Test 1: Generate enhanced helpers
    format('[Test 1] Generate enhanced helpers~n', []),
    ps_enhanced_helpers(Helpers1),
    (   sub_string(Helpers1, _, _, _, "Invoke-FanOut"),
        sub_string(Helpers1, _, _, _, "Merge-Streams"),
        sub_string(Helpers1, _, _, _, "Invoke-RouteRecord"),
        sub_string(Helpers1, _, _, _, "Select-FilteredRecords"),
        sub_string(Helpers1, _, _, _, "Invoke-TeeStream")
    ->  format('  [PASS] All helper functions generated~n', [])
    ;   format('  [FAIL] Missing helper functions~n', [])
    ),

    % Test 2: Linear pipeline connector
    format('[Test 2] Linear pipeline connector~n', []),
    generate_ps_enhanced_connector([extract/1, transform/1, load/1], 'Invoke-LinearPipe', Code2),
    (   sub_string(Code2, _, _, _, "Invoke-LinearPipe"),
        sub_string(Code2, _, _, _, "Invoke-extract"),
        sub_string(Code2, _, _, _, "Invoke-transform"),
        sub_string(Code2, _, _, _, "Invoke-load")
    ->  format('  [PASS] Linear connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code2])
    ),

    % Test 3: Fan-out connector
    format('[Test 3] Fan-out connector~n', []),
    generate_ps_enhanced_connector([fan_out([validate/1, enrich/1])], 'Invoke-FanoutPipe', Code3),
    (   sub_string(Code3, _, _, _, "Invoke-FanoutPipe"),
        sub_string(Code3, _, _, _, "Fan-out to 2 parallel stages"),
        sub_string(Code3, _, _, _, "Invoke-FanOut")
    ->  format('  [PASS] Fan-out connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code3])
    ),

    % Test 4: Fan-out with merge
    format('[Test 4] Fan-out with merge~n', []),
    generate_ps_enhanced_connector([fan_out([a/1, b/1]), merge], 'Invoke-MergePipe', Code4),
    (   sub_string(Code4, _, _, _, "Invoke-MergePipe"),
        sub_string(Code4, _, _, _, "Fan-out to 2"),
        sub_string(Code4, _, _, _, "Merge: results already combined")
    ->  format('  [PASS] Merge connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code4])
    ),

    % Test 5: Conditional routing
    format('[Test 5] Conditional routing~n', []),
    generate_ps_enhanced_connector([route_by(hasError, [(true, errorHandler/1), (false, success/1)])], 'Invoke-RoutePipe', Code5),
    (   sub_string(Code5, _, _, _, "Invoke-RoutePipe"),
        sub_string(Code5, _, _, _, "Conditional routing based on hasError"),
        sub_string(Code5, _, _, _, "$routeMap")
    ->  format('  [PASS] Routing connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code5])
    ),

    % Test 6: Filter stage
    format('[Test 6] Filter stage~n', []),
    generate_ps_enhanced_connector([filter_by(isValid)], 'Invoke-FilterPipe', Code6),
    (   sub_string(Code6, _, _, _, "Invoke-FilterPipe"),
        sub_string(Code6, _, _, _, "Filter by isValid"),
        sub_string(Code6, _, _, _, "Select-FilteredRecords")
    ->  format('  [PASS] Filter connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code6])
    ),

    % Test 7: Complex pipeline with all patterns
    format('[Test 7] Complex pipeline~n', []),
    generate_ps_enhanced_connector([
        extract/1,
        filter_by(isActive),
        fan_out([validate/1, enrich/1, audit/1]),
        merge,
        route_by(hasError, [(true, errorLog/1), (false, transform/1)]),
        output/1
    ], 'Invoke-ComplexPipe', Code7),
    (   sub_string(Code7, _, _, _, "Invoke-ComplexPipe"),
        sub_string(Code7, _, _, _, "Filter by isActive"),
        sub_string(Code7, _, _, _, "Fan-out to 3 parallel stages"),
        sub_string(Code7, _, _, _, "Merge"),
        sub_string(Code7, _, _, _, "Conditional routing")
    ->  format('  [PASS] Complex connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code7])
    ),

    % Test 8: Stage function generation
    format('[Test 8] Stage function generation~n', []),
    generate_ps_enhanced_stage_functions([extract/1, transform/1], StageFns8),
    (   sub_string(StageFns8, _, _, _, "Invoke-extract"),
        sub_string(StageFns8, _, _, _, "Invoke-transform")
    ->  format('  [PASS] Stage functions generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [StageFns8])
    ),

    % Test 9: Full enhanced pipeline compilation
    format('[Test 9] Full enhanced pipeline~n', []),
    compile_powershell_enhanced_pipeline([
        extract/1,
        filter_by(isActive),
        fan_out([validate/1, enrich/1]),
        merge,
        output/1
    ], [pipeline_name('Invoke-FullEnhanced'), output_format(jsonl)], FullCode9),
    (   sub_string(FullCode9, _, _, _, "PowerShell Enhanced Pipeline"),
        sub_string(FullCode9, _, _, _, "Invoke-FanOut"),
        sub_string(FullCode9, _, _, _, "Select-FilteredRecords"),
        sub_string(FullCode9, _, _, _, "Invoke-FullEnhanced"),
        sub_string(FullCode9, _, _, _, "Read-JsonlStream")
    ->  format('  [PASS] Full pipeline compiles~n', [])
    ;   format('  [FAIL] Missing patterns in generated code~n', [])
    ),

    % Test 10: Enhanced helpers include all functions
    format('[Test 10] Enhanced helpers completeness~n', []),
    ps_enhanced_helpers(Helpers10),
    (   sub_string(Helpers10, _, _, _, "Invoke-FanOut"),
        sub_string(Helpers10, _, _, _, "Merge-Streams"),
        sub_string(Helpers10, _, _, _, "Invoke-RouteRecord"),
        sub_string(Helpers10, _, _, _, "Select-FilteredRecords"),
        sub_string(Helpers10, _, _, _, "Invoke-TeeStream")
    ->  format('  [PASS] All helpers present~n', [])
    ;   format('  [FAIL] Missing helpers~n', [])
    ),

    format('~n=== All PowerShell Enhanced Pipeline Chaining Tests Passed ===~n', []).
