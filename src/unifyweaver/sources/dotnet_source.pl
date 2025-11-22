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
    (   member(external_compile(Value), Config)
    ->  (   memberchk(Value, [true, false])
        ->  true
        ;   format('Error: external_compile must be true or false, got ~w~n', [Value]),
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
    ;   % Generate PascalCase namespace from predicate name
        capitalize_atom(Pred, PascalPred),
        atom_concat('UnifyWeaver.Generated.', PascalPred, Namespace)
    ),
    (   member(class_name(ClassName), AllOptions)
    ->  true
    ;   % Generate PascalCase class name
        capitalize_atom(Pred, PascalPred),
        atom_concat(PascalPred, 'Handler', ClassName)
    ),
    (   member(method_name(MethodName), AllOptions)
    ->  true
    ;   % Generate PascalCase method name
        capitalize_atom(Pred, PascalPred),
        atom_concat('Process', PascalPred, MethodName)
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

    % Determine compilation mode with .NET SDK detection and fallback
    (   member(external_compile(UserExternalCompile), AllOptions)
    ->  % User explicitly requested external_compile
        (   UserExternalCompile = true
        ->  check_dotnet_sdk_available(DotnetAvailable),
            (   DotnetAvailable = true
            ->  ExternalCompile = true
            ;   format('Warning: .NET SDK not available, falling back to pre_compile mode for ~w~n', [Pred]),
                ExternalCompile = false
            )
        ;   ExternalCompile = false
        )
    ;   member(pre_compile(true), AllOptions)
    ->  % User explicitly requested pre_compile
        ExternalCompile = false
    ;   % No explicit choice - auto-detect and prefer external_compile if available
        check_dotnet_sdk_available(DotnetAvailable),
        (   DotnetAvailable = true
        ->  ExternalCompile = true
        ;   ExternalCompile = false
        )
    ),

    % Generate PowerShell code using template
    atom_string(Pred, PredStr),
    atom_string(Namespace, NamespaceStr),
    atom_string(ClassName, ClassNameStr),
    atom_string(MethodName, MethodNameStr),

    (   member(output_format(OutputFormat), AllOptions)
    ->  true
    ;   OutputFormat = text % Default
    ),

    generate_dotnet_powershell(PredStr, Arity, CSharpCode, NamespaceStr, ClassNameStr,
                               MethodNameStr, PreCompile, CacheDir, References,
                               OutputFormat, SourceMode, ExternalCompile, AllOptions, PowerShellCode).

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
%% .NET SDK DETECTION
%% ============================================

%% check_dotnet_sdk_available(-Available)
%  Check if dotnet SDK is available on the system
check_dotnet_sdk_available(true) :-
    catch(
        (process_create(path(dotnet), ['--version'], [stdout(null), stderr(null), process(PID)]),
         process_wait(PID, exit(0))),
        _,
        fail
    ),
    !.
check_dotnet_sdk_available(false).

%% ============================================
%% POWERSHELL CODE GENERATION
%% ============================================

%% generate_dotnet_powershell(+PredStr, +Arity, +CSharpCode, +Namespace, +ClassName,
%%                            +MethodName, +PreCompile, +CacheDir, +References,
%%                            +OutputFormat, +SourceMode, +ExternalCompile, +Options, -PowerShellCode)
%  Generate PowerShell code for .NET source
generate_dotnet_powershell(PredStr, Arity, CSharpCode, Namespace, ClassName,
                          MethodName, PreCompile, CacheDir, References,
                          OutputFormat, SourceMode, ExternalCompile, _Options, PowerShellCode) :-

    % Escape C# code for PowerShell here-string
    escape_csharp_for_powershell(CSharpCode, EscapedCode),

    % Generate reference assemblies string
    generate_references_string(References, ReferencesStr),

    % Extract DLL file references for dependency loading
    extract_dll_references(References, DllRefs),
    format_dll_references_for_powershell(DllRefs, DllRefsStr),

    % Determine compilation mode
    (   ExternalCompile = true
    ->  TemplateName = dotnet_source_external_compile,
        generate_package_references_string(References, PackageRefsStr),
        generate_assembly_references_string(References, AssemblyRefsStr),
        ExtraParams = [package_references=PackageRefsStr, assembly_references=AssemblyRefsStr],
        CacheKey = ''
    ;   PreCompile = true
    ->  format(atom(CacheKey), '~w_~w_~w', [PredStr, ClassName, MethodName]),
        TemplateName = dotnet_source_precompiled,
        ExtraParams = []
    ;   CacheKey = '',
        TemplateName = dotnet_source_inline,
        ExtraParams = []
    ),

    % Prepare template parameters
    BaseParams = [pred=PredStr, arity=Arity, csharp_code=EscapedCode,
                  namespace=Namespace, class_name=ClassName, method_name=MethodName,
                  references=ReferencesStr, dll_references=DllRefsStr,
                  cache_dir=CacheDir, cache_key=CacheKey,
                  output_format=OutputFormat, source_mode=SourceMode],
    
    append(BaseParams, ExtraParams, AllParams),

    % Render template
    render_named_template(TemplateName, AllParams,
        [source_order([file, generated]), template_extension('.tmpl.ps1')],
        PowerShellCode).

%% escape_csharp_for_powershell(+Code, -Escaped)
%  Escape C# code for safe usage in PowerShell here-string
%  PowerShell here-strings preserve literal content, but Prolog's atom/string
%  conversion interprets escape sequences like \n as newlines.
%  We need to double backslashes so C# escape sequences are preserved.
escape_csharp_for_powershell(Code, Escaped) :-
    atom_string(Code, CodeStr),
    % Replace single backslash with double backslash to preserve C# escape sequences
    split_string(CodeStr, "\\", "", Parts),
    atomic_list_concat(Parts, "\\\\", Escaped).

%% generate_references_string(+References, -String)
%  Generate PowerShell Add-Type -ReferencedAssemblies parameter
%  Automatically adds 'netstandard' when DLL files are referenced
generate_references_string([], '') :- !.
generate_references_string(References, String) :-
    % Check if any references are DLL files (indicating .NET Standard libraries)
    (   member(Ref, References),
        atom(Ref),
        atom_concat(_, '.dll', Ref)
    ->  NeedsNetStandard = true
    ;   NeedsNetStandard = false
    ),

    % Add netstandard, System.Linq, and System.Linq.Expressions if needed and not already present
    (   NeedsNetStandard = true
    ->  (   \+ member(netstandard, References)
        ->  TempRefs1 = [netstandard|References]
        ;   TempRefs1 = References
        ),
        (   \+ member('System.Linq', TempRefs1)
        ->  TempRefs2 = ['System.Linq'|TempRefs1]
        ;   TempRefs2 = TempRefs1
        ),
        (   \+ member('System.Linq.Expressions', TempRefs2)
        ->  AllReferences = ['System.Linq.Expressions'|TempRefs2]
        ;   AllReferences = TempRefs2
        )
    ;   AllReferences = References
    ),

    maplist(quote_reference, AllReferences, Quoted),
    atomic_list_concat(Quoted, ',', RefList),
    format(atom(String), ' -ReferencedAssemblies @(~w)', [RefList]).

quote_reference(Ref, Quoted) :-
    format(atom(Quoted), '''~w''', [Ref]).

%% extract_dll_references(+References, -DllRefs)
%  Extract only .dll file references from the references list
extract_dll_references(References, DllRefs) :-
    include(is_dll_reference, References, DllRefs).

is_dll_reference(Ref) :-
    atom(Ref),
    atom_concat(_, '.dll', Ref).

%% format_dll_references_for_powershell(+DllRefs, -FormattedString)
%  Format DLL references as PowerShell array elements
format_dll_references_for_powershell([], '') :- !.
format_dll_references_for_powershell(DllRefs, FormattedString) :-
    maplist(quote_reference, DllRefs, Quoted),
    atomic_list_concat(Quoted, ',', FormattedString).

%% generate_package_references_string(+References, -String)
%  Generate XML for PackageReference items in .csproj
generate_package_references_string(References, String) :-
    findall(Xml,
        (member(Ref, References),
         generate_single_package_reference(Ref, Xml)),
        XmlList),
    atomic_list_concat(XmlList, '\n', String).

generate_single_package_reference(Ref, '') :-
    atom(Ref),
    atom_concat(_, '.dll', Ref), !. % Skip DLLs
generate_single_package_reference('System.Text.Json', '<PackageReference Include="System.Text.Json" Version="9.0.0" />') :- !.
generate_single_package_reference(Ref, Xml) :-
    % Default fallback for other packages - assume version *
    format(atom(Xml), '    <PackageReference Include="~w" Version="*" />', [Ref]).

%% generate_assembly_references_string(+References, -String)
%  Generate XML for Reference items (DLLs) in .csproj
generate_assembly_references_string(References, String) :-
    findall(Xml,
        (member(Ref, References),
         generate_single_assembly_reference(Ref, Xml)),
        XmlList),
    atomic_list_concat(XmlList, '\n', String).

generate_single_assembly_reference(Ref, Xml) :-
    atom(Ref),
    atom_concat(_, '.dll', Ref),
    !,
    absolute_file_name(Ref, AbsPath),
    file_base_name(Ref, BaseName),
    file_name_extension(Name, _, BaseName),
    format(atom(Xml), '    <Reference Include="~w"><HintPath>~w</HintPath></Reference>', [Name, AbsPath]).
generate_single_assembly_reference(_, '').

%% capitalize_atom(+Atom, -Capitalized)
%  Convert atom to PascalCase (capitalize first letter of each word)
%  Examples: test_string_reverser -> TestStringReverser
%            my_handler -> MyHandler
capitalize_atom(Atom, Capitalized) :-
    atom_string(Atom, String),
    % Split by underscore
    split_string(String, "_", "", Parts),
    % Capitalize each part
    maplist(capitalize_string, Parts, CapitalizedParts),
    % Join without separator
    atomic_list_concat(CapitalizedParts, Capitalized).

%% capitalize_string(+String, -Capitalized)
%  Capitalize first letter of a string
capitalize_string(String, Capitalized) :-
    string_chars(String, [First|Rest]),
    upcase_atom(First, UpperFirst),
    atomic_list_concat([UpperFirst|Rest], Capitalized).

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
    {{pred}} @args
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
        # Setup cache directory (cross-platform)
        $tempBase = if ($env:TEMP) {
            $env:TEMP
        } elseif ($env:TMPDIR) {
            $env:TMPDIR
        } else {
            "/tmp"
        }
        $cacheDir = Join-Path $tempBase "unifyweaver_dotnet_cache"
        if (-not (Test-Path $cacheDir)) {
            New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
        }

        # Generate cache key based on code hash
        $cacheKey = "{{cache_key}}"
        $dllPath = Join-Path $cacheDir "$cacheKey.dll"

        # Check if DLL exists and is valid
        if (Test-Path $dllPath) {
            try {
                # Load DLL dependencies first
                $dllRefs = @({{dll_references}})
                foreach ($dllRef in $dllRefs) {
                    if (Test-Path $dllRef) {
                        Add-Type -Path $dllRef -ErrorAction SilentlyContinue
                    }
                }

                # Load pre-compiled assembly
                Add-Type -Path $dllPath -ErrorAction Stop
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

                # Load the compiled DLL into the session
                # Load dependencies first
                $dllRefs = @({{dll_references}})
                foreach ($dllRef in $dllRefs) {
                    if (Test-Path $dllRef) {
                        Add-Type -Path $dllRef -ErrorAction SilentlyContinue
                    }
                }
                Add-Type -Path $dllPath -ErrorAction Stop
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
    $tempBase = if ($env:TEMP) { $env:TEMP } elseif ($env:TMPDIR) { $env:TMPDIR } else { "/tmp" }
    $cacheDir = Join-Path $tempBase "unifyweaver_dotnet_cache"
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
    {{pred}} @args
}
').

