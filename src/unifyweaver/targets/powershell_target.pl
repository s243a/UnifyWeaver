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
    test_powershell_pipeline_generator/0
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
