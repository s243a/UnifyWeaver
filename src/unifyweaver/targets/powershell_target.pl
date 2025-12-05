:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% powershell_target.pl - PowerShell Semantic Target for UnifyWeaver
% Generates native PowerShell code using .NET objects for XML streaming and Vector search.
% Phase 2: Native Semantic Implementation.

:- module(powershell_target, [
    compile_predicate_to_powershell/3,  % +Predicate, +Options, -PsCode
    compile_vector_ops/2                % +Options, -PsCode
]).

:- use_module(library(lists)).
:- use_module(library(option)).

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
