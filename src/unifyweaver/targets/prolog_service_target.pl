:- module(prolog_service_target, [
    generate_bash_with_prolog_service/3,  % +UserPredicates, +Options, -BashCode
    analyze_dependencies/2,               % +UserPredicates, -Dependencies (reuse from prolog_target)
    write_bash_script/2                   % +BashCode, +OutputPath
]).

%! generate_bash_with_prolog_service(+UserPredicates, +Options, -BashCode) is det.
%
%  Generate a bash script that uses Prolog-as-Service pattern.
%
%  @arg UserPredicates List of predicate indicators (Name/Arity)
%  @arg Options Compilation options:
%               - entry_point(Goal): Main goal to execute
%               - service_name(Name): Name for the Prolog service function
%               - source_file(File): Source file path (for comments)
%               - target_platform(Platform): Target platform (linux/windows/darwin/auto)
%               - heredoc_method(Method): Force specific method (fd3/tempfile/auto)
%  @arg BashCode Generated bash script code
%
%  Platform Detection:
%  - linux/darwin: Use /dev/fd/3 (default)
%  - windows: Use temp files (default)
%  - auto: Detect from current system
%
%  Heredoc Method:
%  - fd3: Use /dev/fd/3 file descriptor (faster, no disk I/O)
%  - tempfile: Use mktemp temporary files (more portable)
%  - auto: Choose based on target_platform
%
%  Example:
%  ```
%  ?- generate_bash_with_prolog_service(
%         [partition_stdin/2, write_partitions/1],
%         [entry_point(partition_stdin(3, P)),
%          service_name(partition_service),
%          target_platform(linux)],
%         Code
%     ).
%  ```
generate_bash_with_prolog_service(UserPredicates, Options, BashCode) :-
    % Analyze dependencies
    prolog_target:analyze_dependencies(UserPredicates, Dependencies),

    % Determine heredoc method based on platform
    determine_heredoc_method(Options, HeredocMethod),

    % Add method to options for downstream use
    OptionsWithMethod = [heredoc_method(HeredocMethod) | Options],

    % Generate bash script components
    generate_bash_shebang(ShebangCode),
    generate_bash_header(UserPredicates, OptionsWithMethod, HeaderCode),
    generate_bash_env_setup(EnvSetupCode),
    generate_prolog_service_function(UserPredicates, Dependencies, OptionsWithMethod, ServiceCode),
    generate_bash_main(OptionsWithMethod, MainCode),

    % Combine all parts
    atomic_list_concat([
        ShebangCode,
        HeaderCode,
        EnvSetupCode,
        ServiceCode,
        MainCode
    ], '\n\n', BashCode).

%% determine_heredoc_method(+Options, -Method)
%  Determine which heredoc method to use based on options and platform
%
%  Priority:
%  1. Explicit heredoc_method(Method) option
%  2. Explicit target_platform(Platform) option
%  3. Auto-detect current platform
determine_heredoc_method(Options, Method) :-
    % Check for explicit heredoc method
    (   member(heredoc_method(ExplicitMethod), Options),
        ExplicitMethod \= auto
    ->  Method = ExplicitMethod
    ;   % Determine from target platform
        determine_target_platform(Options, Platform),
        platform_default_method(Platform, Method)
    ).

%% determine_target_platform(+Options, -Platform)
%  Determine target platform from options or auto-detect
determine_target_platform(Options, Platform) :-
    (   member(target_platform(ExplicitPlatform), Options),
        ExplicitPlatform \= auto
    ->  Platform = ExplicitPlatform
    ;   % Auto-detect current platform
        detect_current_platform(Platform)
    ).

%% detect_current_platform(-Platform)
%  Detect current platform (linux/windows/darwin)
detect_current_platform(Platform) :-
    current_prolog_flag(windows, true),
    !,
    Platform = windows.
detect_current_platform(Platform) :-
    current_prolog_flag(apple, true),
    !,
    Platform = darwin.
detect_current_platform(linux).  % Default to linux for other Unix systems

%% platform_default_method(+Platform, -Method)
%  Map platform to default heredoc method
platform_default_method(windows, tempfile) :- !.
platform_default_method(linux, fd3) :- !.
platform_default_method(darwin, fd3) :- !.
platform_default_method(_, tempfile).  % Safe fallback

%% generate_bash_shebang(-Code)
%  Generate bash shebang line
generate_bash_shebang('#!/bin/bash').

%% generate_bash_header(+UserPredicates, +Options, -Code)
%  Generate bash script header with metadata
generate_bash_header(UserPredicates, Options, Code) :-
    % Get current date/time
    get_time(Timestamp),
    format_time(atom(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),

    % Extract source file if provided
    (   member(source_file(SourceFile), Options)
    ->  format(atom(SourceComment), '# Source: ~w', [SourceFile])
    ;   SourceComment = ''
    ),

    % Get heredoc method
    (   member(heredoc_method(Method), Options)
    ->  format(atom(MethodComment), '# Heredoc method: ~w', [Method])
    ;   MethodComment = '# Heredoc method: auto'
    ),

    % Format predicate list
    length(UserPredicates, NumPreds),
    format(atom(PredsStr), '~w', [UserPredicates]),

    % Build header lines
    format(atom(DateLine), '# Generated: ~w', [DateStr]),
    format(atom(NumPredsLine), '# Predicates transpiled: ~w', [NumPreds]),
    format(atom(PredsLine), '# ~w', [PredsStr]),

    % Combine
    BaseLines = ['# Generated by UnifyWeaver v0.0.3',
                 '# Target: Bash with Prolog-as-Service',
                 DateLine,
                 MethodComment],

    % Add source comment if present
    (   SourceComment = ''
    ->  LinesWithSource = BaseLines
    ;   append(BaseLines, [SourceComment], LinesWithSource)
    ),

    % Add predicate info
    append(LinesWithSource, [NumPredsLine, PredsLine], Lines),
    atomic_list_concat(Lines, '\n', Code).

%% generate_bash_env_setup(-Code)
%  Generate bash environment setup (UNIFYWEAVER_HOME detection)
generate_bash_env_setup(Code) :-
    Lines = [
        '# Set up UnifyWeaver runtime environment',
        'setup_unifyweaver_env() {',
        '    if [ -n "$UNIFYWEAVER_HOME" ]; then',
        '        # Use provided UNIFYWEAVER_HOME',
        '        export UNIFYWEAVER_HOME',
        '    else',
        '        # Try to detect from script location',
        '        local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        '        if [ -d "$script_dir/../src/unifyweaver" ]; then',
        '            export UNIFYWEAVER_HOME="$script_dir/../src/unifyweaver"',
        '        elif [ -d "$script_dir/src/unifyweaver" ]; then',
        '            export UNIFYWEAVER_HOME="$script_dir/src/unifyweaver"',
        '        else',
        '            # Assume installed as SWI-Prolog pack',
        '            export UNIFYWEAVER_HOME=""',
        '        fi',
        '    fi',
        '}',
        '',
        '# Initialize environment',
        'setup_unifyweaver_env'
    ],
    atomic_list_concat(Lines, '\n', Code).

%% generate_prolog_service_function(+UserPredicates, +Dependencies, +Options, -Code)
%  Generate bash function containing Prolog service as heredoc
%
%  Supports two methods:
%  - fd3: Use /dev/fd/3 file descriptor (Linux/Unix)
%  - tempfile: Use mktemp temporary files (Windows/portable)
generate_prolog_service_function(UserPredicates, Dependencies, Options, Code) :-
    % Get service name
    (   member(service_name(ServiceName), Options)
    ->  true
    ;   ServiceName = prolog_service
    ),

    % Get heredoc method
    (   member(heredoc_method(Method), Options)
    ->  true
    ;   Method = tempfile  % Fallback
    ),

    % Generate Prolog code for the heredoc
    generate_prolog_service_code(UserPredicates, Dependencies, Options, PrologCode),

    % Generate appropriate bash function for the method
    (   Method = fd3
    ->  generate_fd3_service_function(ServiceName, PrologCode, Code)
    ;   Method = tempfile
    ->  generate_tempfile_service_function(ServiceName, PrologCode, Code)
    ;   % Unknown method, use tempfile as fallback
        format(atom(Warning), '# Warning: Unknown heredoc method "~w", using tempfile', [Method]),
        generate_tempfile_service_function(ServiceName, PrologCode, TempCode),
        atomic_list_concat([Warning, TempCode], '\n', Code)
    ).

%% generate_fd3_service_function(+ServiceName, +PrologCode, -Code)
%  Generate service using /dev/fd/3 file descriptor
%
%  Advantages:
%  - No disk I/O (faster)
%  - No temp file cleanup needed
%  - More secure (code never on disk)
%
%  Works on: Linux, macOS, other Unix systems with /dev/fd
generate_fd3_service_function(ServiceName, PrologCode, Code) :-
    format(atom(FunctionHeader), '# Prolog service function (fd3 with tempfile fallback)\n~w() {', [ServiceName]),

    Lines = [
        FunctionHeader,
        '    # Save original stdin to fd4 so the Prolog code can reopen it as /dev/fd/4',
        '    exec 4<&0',
        '    ',
        '    local status=0',
        '    swipl -q -g "consult(user), main, halt" -t halt <<\'PROLOG\'',
        PrologCode,
        'PROLOG',
        '    status=$?',
        '    if [ "$status" -eq 86 ]; then',
        '        echo "[UnifyWeaver] fd3 setup failed, retrying with temp file" >&2',
        '        local temp_pl=$(mktemp --suffix=.pl)',
        '        cat <<\'PROLOG\' > "$temp_pl"',
        PrologCode,
        'PROLOG',
        '        UNIFYWEAVER_SKIP_FD4=1 swipl -q -f "$temp_pl" -g main -t halt',
        '        status=$?',
        '        rm -f "$temp_pl"',
        '    fi',
        '    exec 4<&-',
        '    return $status',
        '}'
    ],
    atomic_list_concat(Lines, '\n', Code).

%% generate_tempfile_service_function(+ServiceName, +PrologCode, -Code)
%  Generate service using temporary files
%
%  Advantages:
%  - More portable (works on Windows)
%  - Compatible with all bash implementations
%
%  Works on: All platforms with mktemp
generate_tempfile_service_function(ServiceName, PrologCode, Code) :-
    format(atom(FunctionHeader), '# Prolog service function (using temp file)\n~w() {', [ServiceName]),

    Lines = [
        FunctionHeader,
        '    local temp_pl=$(mktemp --suffix=.pl)',
        '    cat << \'PROLOG\' > "$temp_pl"',
        PrologCode,
        'PROLOG',
        '    swipl -q -f "$temp_pl" -g main -t halt',
        '    rm -f "$temp_pl"',
        '}'
    ],
    atomic_list_concat(Lines, '\n', Code).

%% generate_prolog_service_code(+UserPredicates, +Dependencies, +Options, -Code)
%  Generate Prolog code to embed in heredoc
generate_prolog_service_code(UserPredicates, Dependencies, Options, Code) :-
    % Get heredoc method to determine if we need stdin redirection
    (   member(heredoc_method(Method), Options)
    ->  true
    ;   Method = tempfile
    ),

    % Generate components
    generate_prolog_search_path_setup(SearchPathCode),

    % Add stdin redirection for fd3 method (stdin saved to fd4)
    (   Method = fd3
    ->  StdinSetup = '% Redirect user_input to saved fd4 after loading\n:- dynamic unifyweaver_fd4_stream/1.\n:- initialization(unifyweaver_setup_fd4_stdin, after_load).\n\nunifyweaver_setup_fd4_stdin :-\n    (   getenv(\'UNIFYWEAVER_SKIP_FD4\', Skip), Skip \\= \'\'\n    ->  true\n    ;   (   catch(unifyweaver_redirect_fd4_to_user_input, Error,\n                  (unifyweaver_report_fd4_error(Error), fail))\n        ->  true\n        ;   halt(86)\n        )\n    ).\n\nunifyweaver_report_fd4_error(Error) :-\n    message_to_string(Error, Msg),\n    format(user_error, \'[UnifyWeaver] fd4 setup failed: ~s~n\', [Msg]).\n\nunifyweaver_redirect_fd4_to_user_input :-\n    catch(open(\'/dev/fd/4\', read, Fd4Stream, [buffer(line)]), OpenError,\n          throw(error(unifyweaver_fd4_open, OpenError))),\n    set_input(Fd4Stream),\n    catch(set_stream(Fd4Stream, alias(user_input)), AliasError,\n          throw(error(unifyweaver_alias_fd4, AliasError))),\n    asserta(unifyweaver_fd4_stream(Fd4Stream)),\n    at_halt(unifyweaver_close_fd4_stream).\n\nunifyweaver_close_fd4_stream :-\n    (   retract(unifyweaver_fd4_stream(Stream))\n    ->  catch(close(Stream), CloseError,\n              ( message_to_string(CloseError, Msg),\n                format(user_error, \'[UnifyWeaver] fd4 close warning: ~s~n\', [Msg])\n              ))\n    ;   true\n    ).'
    ;   StdinSetup = ''
    ),

    generate_prolog_imports(Dependencies, ImportsCode),
    generate_prolog_user_code(UserPredicates, UserCode),
    generate_prolog_entry_point(Options, EntryCode),

    % Combine with proper indentation (4 spaces for heredoc content)
    indent_code(SearchPathCode, 4, IndentedSearchPath),
    (   StdinSetup \= ''
    ->  indent_code(StdinSetup, 4, IndentedStdinSetup),
        Components = [IndentedSearchPath, IndentedStdinSetup]
    ;   Components = [IndentedSearchPath]
    ),
    indent_code(ImportsCode, 4, IndentedImports),
    indent_code(UserCode, 4, IndentedUserCode),
    indent_code(EntryCode, 4, IndentedEntryCode),

    append(Components, [IndentedImports, IndentedUserCode, IndentedEntryCode], AllComponents),
    atomic_list_concat(AllComponents, '\n\n', Code).

%% generate_prolog_search_path_setup(-Code)
%  Generate Prolog code to set up search path from bash environment
generate_prolog_search_path_setup(Code) :-
    Lines = [
        '% Set up search path from UNIFYWEAVER_HOME environment variable',
        ':- ( getenv(\'UNIFYWEAVER_HOME\', Home), Home \\= \'\'',
        '   -> asserta(file_search_path(unifyweaver, Home))',
        '   ;  true  % Assume installed as pack',
        '   ).'
    ],
    atomic_list_concat(Lines, '\n', Code).

%% generate_prolog_imports(+Dependencies, -Code)
%  Generate Prolog import directives
generate_prolog_imports(Dependencies, Code) :-
    findall(Import, (
        member(Dep, Dependencies),
        prolog_target:dependency_to_import(Dep, Import)
    ), Imports),

    (   Imports = []
    ->  Code = '% No dependencies'
    ;   atomic_list_concat(Imports, '\n', Code)
    ).

%% generate_prolog_user_code(+UserPredicates, -Code)
%  Generate user predicates code
generate_prolog_user_code(UserPredicates, Code) :-
    format(atom(Header), '% === User Code (Transpiled) ===', []),

    % Copy each predicate
    findall(PredCode, (
        member(Pred, UserPredicates),
        prolog_target:copy_predicate_clauses(Pred, PredCode)
    ), PredCodes),

    % Combine
    atomic_list_concat([Header | PredCodes], '\n\n', Code).

%% generate_prolog_entry_point(+Options, -Code)
%  Generate Prolog entry point
generate_prolog_entry_point(Options, Code) :-
    % Get entry point goal
    (   member(entry_point(EntryGoal), Options)
    ->  format(atom(MainBody), '    ~w,', [EntryGoal])
    ;   MainBody = '    true,'
    ),

    % Build lines
    atomic_list_concat([
        '% === Entry Point ===',
        'main :-',
        MainBody,
        '    halt(0).',
        '',
        'main :-',
        '    % If main goal fails, exit with error',
        '    format(user_error, \'Error: Execution failed~n\', []),',
        '    halt(1).'
    ], '\n', Code).

%% generate_bash_main(+Options, -Code)
%  Generate bash main entry point
generate_bash_main(Options, Code) :-
    % Get service name
    (   member(service_name(ServiceName), Options)
    ->  format(atom(ServiceCall), '    ~w', [ServiceName])
    ;   ServiceCall = '    prolog_service'
    ),

    atomic_list_concat([
        '# Main entry point',
        'main() {',
        ServiceCall,
        '}',
        '',
        '# Run main if script is executed directly',
        'if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then',
        '    main "$@"',
        'fi'
    ], '\n', Code).

%% indent_code(+Code, +Spaces, -IndentedCode)
%  Indent each line of code by given number of spaces
indent_code(Code, Spaces, IndentedCode) :-
    atom_string(Code, CodeStr),
    split_string(CodeStr, "\n", "", Lines),

    % Create indentation string
    length(SpaceList, Spaces),
    maplist(=(' '), SpaceList),
    atomic_list_concat(SpaceList, Indent),

    % Indent each line
    maplist(indent_line(Indent), Lines, IndentedLines),
    atomic_list_concat(IndentedLines, '\n', IndentedCode).

indent_line(Indent, Line, IndentedLine) :-
    (   Line = ""
    ->  IndentedLine = ""  % Don't indent empty lines
    ;   atomic_list_concat([Indent, Line], IndentedLine)
    ).

%% write_bash_script(+BashCode, +OutputPath)
%  Write bash script to file and make it executable
write_bash_script(BashCode, OutputPath) :-
    open(OutputPath, write, Stream),
    write(Stream, BashCode),
    close(Stream),

    % Make executable
    format(atom(ChmodCmd), 'chmod +x ~w', [OutputPath]),
    shell(ChmodCmd),

    format('[PrologService] Generated executable bash script: ~w~n', [OutputPath]).

%% analyze_dependencies(+UserPredicates, -Dependencies)
%  Reuse dependency analysis from prolog_target
analyze_dependencies(UserPredicates, Dependencies) :-
    prolog_target:analyze_dependencies(UserPredicates, Dependencies).
