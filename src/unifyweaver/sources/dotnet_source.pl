:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% dotnet_source.pl - .NET/C# source plugin for PowerShell targets
% Compiles predicates that execute inline C# code via PowerShell Add-Type
%
% This plugin enables:
% - Inline C# code in Prolog predicates
% - Pre-compilation to DLL for performance
% - Assembly caching to avoid recompilation
% - Access to full .NET Framework/Core libraries

:- module(dotnet_source, []).

:- use_module(library(lists)).
:- use_module(library(readutil)).
:- use_module('../core/template_system').
:- use_module('../core/dynamic_source_compiler').

%% Register this plugin on load
:- initialization(
    register_source_type(dotnet, dotnet_source),
    now
).

%% ============================================
%% PLUGIN INTERFACE
%% ============================================

%% source_info(-Info)
%  Provide information about this source plugin
source_info(info(
    name('.NET/C# Inline Source'),
    version('1.0.0'),
    description('Execute inline C# code via PowerShell Add-Type with optional pre-compilation'),
    supported_arities([1, 2, 3, 4, 5])
)).

%% validate_config(+Config)
%  Validate configuration for .NET source
validate_config(Config) :-
    % Must have either csharp_inline or csharp_file
    (   member(csharp_inline(_), Config)
    ->  true
    ;   member(csharp_file(File), Config)
    ->  (   exists_file(File)
        ->  true
        ;   format('Error: C# file ~w does not exist~n', [File]),
            fail
        )
    ;   format('Error: .NET source requires csharp_inline(Code) or csharp_file(File)~n', []),
        fail
    ),

    % Validate namespace if specified
    (   member(namespace(NS), Config)
    ->  atom(NS)
    ;   true
    ),

    % Validate class_name if specified
    (   member(class_name(Class), Config)
    ->  atom(Class)
    ;   true
    ),

    % Validate method_name if specified
    (   member(method_name(Method), Config)
    ->  atom(Method)
    ;   true
    ),

    % Validate pre_compile option
    (   member(pre_compile(Value), Config)
    ->  (   memberchk(Value, [true, false])
        ->  true
        ;   format('Error: pre_compile must be true or false, got ~w~n', [Value]),
            fail
        )
    ;   true
    ),

    % Validate references if specified
    (   member(references(Refs), Config)
    ->  is_list(Refs)
    ;   true
    ).

%% compile_source(+Pred/Arity, +Config, +Options, -PowerShellCode)
%  Compile .NET source to PowerShell code
compile_source(Pred/Arity, Config, Options, PowerShellCode) :-
    format('  Compiling .NET source: ~w/~w~n', [Pred, Arity]),

    % Validate configuration
    validate_config(Config),

    % Merge config and options
    append(Config, Options, AllOptions),

    % Extract C# code source
    (   member(csharp_inline(CSharpCode), AllOptions)
    ->  SourceMode = inline
    ;   member(csharp_file(File), AllOptions),
        read_csharp_file(File, CSharpCode),
        SourceMode = file
    ),

    % Extract optional parameters with defaults
    (   member(namespace(Namespace), AllOptions)
    ->  true
    ;   atom_concat('UnifyWeaver.Generated.', Pred, Namespace)
    ),
    (   member(class_name(ClassName), AllOptions)
    ->  true
    ;   upcase_atom(Pred, Upper),
        atom_concat(Upper, 'Handler', ClassName)
    ),
    (   member(method_name(MethodName), AllOptions)
    ->  true
    ;   atom_concat('Process', Pred, TempMethod),
        upcase_atom(TempMethod, MethodName)
    ),
    (   member(pre_compile(PreCompile), AllOptions)
    ->  true
    ;   PreCompile = false  % Default: no pre-compilation
    ),
    (   member(cache_dir(CacheDir), AllOptions)
    ->  true
    ;   CacheDir = '$env:TEMP/unifyweaver_dotnet_cache'
    ),
    (   member(references(References), AllOptions)
    ->  true
    ;   References = []
    ),
    (   member(output_format(OutputFormat), AllOptions)
    ->  true
    ;   OutputFormat = text
    ),

    % Generate PowerShell code using template
    atom_string(Pred, PredStr),
    atom_string(Namespace, NamespaceStr),
    atom_string(ClassName, ClassNameStr),
    atom_string(MethodName, MethodNameStr),

    generate_dotnet_powershell(PredStr, Arity, CSharpCode, NamespaceStr, ClassNameStr,
                               MethodNameStr, PreCompile, CacheDir, References,
                               OutputFormat, SourceMode, AllOptions, PowerShellCode).

%% ============================================
%% C# CODE READING
%% ============================================

%% read_csharp_file(+File, -Code)
%  Read C# code from external file
read_csharp_file(File, Code) :-
    catch(
        read_file_to_string(File, Code, []),
        Error,
        (   format('Error reading C# file ~w: ~w~n', [File, Error]),
            fail
        )
    ).

%% ============================================
%% POWERSHELL CODE GENERATION
%% ============================================

%% generate_dotnet_powershell(+PredStr, +Arity, +CSharpCode, +Namespace, +ClassName,
%%                            +MethodName, +PreCompile, +CacheDir, +References,
%%                            +OutputFormat, +SourceMode, +Options, -PowerShellCode)
%  Generate PowerShell code for .NET source
generate_dotnet_powershell(PredStr, Arity, CSharpCode, Namespace, ClassName,
                          MethodName, PreCompile, CacheDir, References,
                          OutputFormat, SourceMode, _Options, PowerShellCode) :-

    % Escape C# code for PowerShell here-string
    escape_csharp_for_powershell(CSharpCode, EscapedCode),

    % Generate reference assemblies string
    generate_references_string(References, ReferencesStr),

    % Determine compilation mode
    (   PreCompile = true
    ->  CompileMode = precompiled,
        format(atom(CacheKey), '~w_~w_~w', [PredStr, ClassName, MethodName])
    ;   CompileMode = inline,
        CacheKey = ''
    ),

    % Select template based on compilation mode
    (   CompileMode = precompiled
    ->  TemplateName = dotnet_source_precompiled
    ;   TemplateName = dotnet_source_inline
    ),

    % Render template
    render_named_template(TemplateName,
        [pred=PredStr, arity=Arity, csharp_code=EscapedCode,
         namespace=Namespace, class_name=ClassName, method_name=MethodName,
         references=ReferencesStr, cache_dir=CacheDir, cache_key=CacheKey,
         output_format=OutputFormat, source_mode=SourceMode],
        [source_order([file, generated])],
        PowerShellCode).

%% escape_csharp_for_powershell(+Code, -Escaped)
%  Escape C# code for safe usage in PowerShell here-string
escape_csharp_for_powershell(Code, Escaped) :-
    % For now, simple pass-through - PowerShell here-strings handle most escaping
    % The @' '@ syntax preserves everything except the closing '@
    Escaped = Code.

%% generate_references_string(+References, -String)
%  Generate PowerShell Add-Type -ReferencedAssemblies parameter
generate_references_string([], '') :- !.
generate_references_string(References, String) :-
    maplist(quote_reference, References, Quoted),
    atomic_list_concat(Quoted, ',', RefList),
    format(atom(String), ' -ReferencedAssemblies @(~w)', [RefList]).

quote_reference(Ref, Quoted) :-
    format(atom(Quoted), '''~w''', [Ref]).

%% ============================================
%% POWERSHELL TEMPLATES
%% ============================================

:- multifile template_system:template/2.

% Template for inline compilation (no pre-compilation, Add-Type on each run)
template_system:template(dotnet_source_inline, '# {{pred}} - .NET inline source ({{source_mode}} mode)
# Generated by UnifyWeaver - Inline C# compilation
# Namespace: {{namespace}}
# Class: {{class_name}}
# Method: {{method_name}}

function {{pred}} {
    param([Parameter(ValueFromPipeline=$true)]$InputData)

    begin {
        # Compile C# code inline using Add-Type
        $csharpCode = @''
{{csharp_code}}
''@

        try {
            Add-Type -TypeDefinition $csharpCode{{references}} -Language CSharp -ErrorAction Stop
            Write-Verbose "[{{pred}}] C# code compiled successfully"
        } catch {
            Write-Error "Failed to compile C# code: $_"
            throw
        }

        # Create instance of the class
        $handler = New-Object {{namespace}}.{{class_name}}
    }

    process {
        try {
            # Call the method with input data
            if ($InputData) {
                $result = $handler.{{method_name}}($InputData)
            } else {
                $result = $handler.{{method_name}}()
            }

            # Output result
            Write-Output $result
        } catch {
            Write-Error "Error executing {{method_name}}: $_"
        }
    }

    end {
        # Cleanup if needed
        $handler = $null
    }
}

function {{pred}}_stream {
    $input | {{pred}}
}

# Auto-execute when run directly (not when dot-sourced)
if ($MyInvocation.InvocationName -ne ''.'') {
    if ($input) {
        $input | {{pred}} @args
    } else {
        {{pred}} @args
    }
}
').

% Template for pre-compiled DLL mode (compile once, cache, and load from DLL)
template_system:template(dotnet_source_precompiled, '# {{pred}} - .NET pre-compiled source
# Generated by UnifyWeaver - Pre-compiled C# with caching
# Namespace: {{namespace}}
# Class: {{class_name}}
# Method: {{method_name}}
# Cache: {{cache_dir}}

function {{pred}} {
    param([Parameter(ValueFromPipeline=$true)]$InputData)

    begin {
        # Setup cache directory
        $cacheDir = "{{cache_dir}}"
        if (-not (Test-Path $cacheDir)) {
            New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
        }

        # Generate cache key based on code hash
        $cacheKey = "{{cache_key}}"
        $dllPath = Join-Path $cacheDir "$cacheKey.dll"

        # Check if DLL exists and is valid
        if (Test-Path $dllPath) {
            try {
                # Load pre-compiled assembly
                [System.Reflection.Assembly]::LoadFrom($dllPath) | Out-Null
                Write-Verbose "[{{pred}}] Loaded pre-compiled assembly from cache"
            } catch {
                Write-Warning "Cached DLL invalid, recompiling: $_"
                Remove-Item $dllPath -ErrorAction SilentlyContinue
                $dllPath = $null
            }
        }

        # Compile if no valid DLL exists
        if (-not (Test-Path $dllPath)) {
            $csharpCode = @''
{{csharp_code}}
''@

            try {
                # Compile to DLL
                Add-Type -TypeDefinition $csharpCode{{references}} -OutputAssembly $dllPath -Language CSharp -ErrorAction Stop
                Write-Verbose "[{{pred}}] C# code compiled and cached to $dllPath"
            } catch {
                Write-Error "Failed to compile C# code: $_"
                throw
            }
        }

        # Create instance of the class
        $handler = New-Object {{namespace}}.{{class_name}}
    }

    process {
        try {
            # Call the method with input data
            if ($InputData) {
                $result = $handler.{{method_name}}($InputData)
            } else {
                $result = $handler.{{method_name}}()
            }

            # Output result
            Write-Output $result
        } catch {
            Write-Error "Error executing {{method_name}}: $_"
        }
    }

    end {
        # Cleanup if needed
        $handler = $null
    }
}

function {{pred}}_stream {
    $input | {{pred}}
}

function {{pred}}_clear_cache {
    # Clear the cached DLL to force recompilation
    $cacheDir = "{{cache_dir}}"
    $cacheKey = "{{cache_key}}"
    $dllPath = Join-Path $cacheDir "$cacheKey.dll"

    if (Test-Path $dllPath) {
        Remove-Item $dllPath -Force
        Write-Host "Cleared cache for {{pred}}"
    } else {
        Write-Host "No cache found for {{pred}}"
    }
}

# Auto-execute when run directly (not when dot-sourced)
if ($MyInvocation.InvocationName -ne ''.'') {
    if ($input) {
        $input | {{pred}} @args
    } else {
        {{pred}} @args
    }
}
').
