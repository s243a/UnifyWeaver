:- encoding(utf8).
%% firewall.pl - Control Plane Firewall System
%
% This module enforces security policies for backend and service usage.
% It defines hard boundaries that must be satisfied before compilation proceeds.
%
% @author John William Creighton (@s243a)
% @license MIT OR Apache-2.0

:- module(firewall, [
    rule_firewall/2,
    firewall_default/1,
    validate_against_firewall/3,
    validate_service/2,
    validate_network_access/2,
    validate_python_imports/2,
    validate_file_access/3,
    validate_cache_directory/2
]).

:- use_module(library(lists)).

%% rule_firewall(?PredicateIndicator, ?PolicyTerms) is nondet.
%
% Declares predicate-specific firewall policies. This is the primary way to define
% security boundaries for individual predicates.
%
% @arg PredicateIndicator The predicate in Functor/Arity form (e.g., ancestor/2)
% @arg PolicyTerms List of policy terms that define security constraints
%
% Policy terms include:
% - execution([backend1, backend2, ...]) - Whitelist of allowed execution backends
% - services([service1, service2, ...]) - Whitelist of allowed external services
% - denied([target1, service1, ...]) - Blacklist that always takes precedence
% - max_cost(low|medium|high) - Abstract cost limit (future enhancement)
% - network_access(allowed|denied) - Control network access for HTTP sources
% - network_hosts([pattern1, pattern2, ...]) - Whitelist of allowed host patterns
% - python_modules([module1, module2, ...]) - Whitelist of allowed Python imports
% - file_read_patterns([pattern1, pattern2, ...]) - Allowed file read patterns
% - file_write_patterns([pattern1, pattern2, ...]) - Allowed file write patterns
% - cache_dirs([dir1, dir2, ...]) - Allowed cache directories
%
% @example Deny LLM service for sensitive predicate
%   :- assertz(firewall:rule_firewall(sensitive_pred/2, [denied([llm])])).
%
% @example Allow only bash execution with SQL service
%   :- assertz(firewall:rule_firewall(db_query/2, [execution([bash]), services([sql])])).
:- dynamic rule_firewall/2.

%% firewall_default(?PolicyTerms) is nondet.
%
% Declares global default firewall policies. These apply to all predicates that
% don't have specific rule_firewall/2 declarations. If neither rule_firewall nor
% firewall_default is defined, the system defaults to implicit allow.
%
% @arg PolicyTerms List of policy terms (same format as rule_firewall/2)
%
% @example Set global default to deny LLM services
%   :- assertz(firewall:firewall_default([denied([llm])])).
%
% @example Allow only bash by default
%   :- assertz(firewall:firewall_default([execution([bash])])).
:- dynamic firewall_default/1.

%% validate_against_firewall(+Target, +FinalOptions, +Firewall) is semidet.
%
% Validates a compilation request against the firewall policy. This is called
% by the main compiler dispatcher before proceeding with code generation.
%
% Validation steps (in order of precedence):
% 1. Check denied list - any match causes immediate failure
% 2. Check execution allowlist - target must be in list (if list exists)
% 3. Check services allowlist - all requested services must be allowed (if list exists)
%
% @arg Target The target execution backend (e.g., bash, python)
% @arg FinalOptions Merged options that may include use_services([...])
% @arg Firewall The policy terms for this predicate
%
% @throws Firewall Violation Prints error and fails if validation fails
%
% @example Validate bash compilation with no services
%   ?- validate_against_firewall(bash, [], [execution([bash])]).
%   true.
%
% @example Validate denied service (fails)
%   ?- validate_against_firewall(bash, [use_services([llm])], [denied([llm])]).
%   Firewall Violation: Service llm is denied.
%   false.
validate_against_firewall(Target, FinalOptions, Firewall) :-
    % Extract policy terms
    findall(X, (member(execution(Xs), Firewall), member(X, Xs)), ExecAllowFlat),
    findall(S, (member(services(Ss), Firewall), member(S, Ss)), ServiceAllowFlat),
    findall(D, (member(denied(Ds), Firewall), member(D, Ds)), DenyFlat),

    % 1) Explicit deny wins
    (   member(Target, DenyFlat) ->
        format(user_error, 'Firewall Violation: Target ~w is denied.~n', [Target]),
        fail
    ;   true
    ),
    findall(Svc, (member(use_services(Svc), FinalOptions)), RequestedSvcs),
    (   intersection(RequestedSvcs, DenyFlat, [Forbidden|_]) ->
        format(user_error, 'Firewall Violation: Service ~w is denied.~n', [Forbidden]),
        fail
    ;   true
    ),

    % 2) Execution allowlist (if present)
    (   ExecAllowFlat == [] -> true
    ;   member(Target, ExecAllowFlat)
    ->  true
    ;   format(user_error, 'Firewall Violation: Target ~w is not in execution allowlist.~n', [Target]),
        fail
    ),

    % 3) Services allowlist (if present)
    (   ServiceAllowFlat == [] -> true
    ;   forall(member(Svc, RequestedSvcs), member(Svc, ServiceAllowFlat))
    ->  true
    ;   subtract(RequestedSvcs, ServiceAllowFlat, [ForbiddenSvc|_]),
        format(user_error, 'Firewall Violation: Service ~w is not in services allowlist.~n', [ForbiddenSvc]),
        fail
    ),

    % 4) Network access validation (for HTTP sources)
    (   member(url(URL), FinalOptions)
    ->  validate_network_access(URL, Firewall)
    ;   true
    ),

    % 5) Python module validation (for Python sources)
    (   member(python_inline(Code), FinalOptions)
    ->  validate_python_imports(Code, Firewall)
    ;   member(python_file(File), FinalOptions)
    ->  (   exists_file(File)
        ->  read_file_to_string(File, FileCode, []),
            validate_python_imports(FileCode, Firewall)
        ;   true
        )
    ;   true
    ),

    % 6) File access validation
    (   member(csv_file(CsvFile), FinalOptions)
    ->  validate_file_access(CsvFile, read, Firewall)
    ;   true
    ),
    (   member(json_file(JsonFile), FinalOptions)
    ->  validate_file_access(JsonFile, read, Firewall)
    ;   true
    ),
    (   member(cache_file(CacheFile), FinalOptions)
    ->  validate_cache_directory(CacheFile, Firewall)
    ;   true
    ).

%% ============================================
%% ENHANCED VALIDATION PREDICATES
%% ============================================

%% validate_service(+Service, +Firewall) is semidet.
%
% Validates that a specific service is allowed by the firewall policy.
% This is a utility predicate that can be called by individual source plugins.
%
% @arg Service The service to validate (e.g., python3, curl, jq)
% @arg Firewall The firewall policy terms
%
% @example Validate python3 service
%   ?- validate_service(python3, [services([python3, curl])]).
%   true.
validate_service(Service, Firewall) :-
    findall(S, (member(services(Ss), Firewall), member(S, Ss)), ServiceAllowFlat),
    findall(D, (member(denied(Ds), Firewall), member(D, Ds)), DenyFlat),
    
    % Check deny list first
    (   member(Service, DenyFlat)
    ->  format(user_error, 'Firewall blocks service: ~w~n', [Service]),
        fail
    ;   true
    ),
    
    % Check allow list (if present)
    (   ServiceAllowFlat == []
    ->  true  % No restrictions
    ;   member(Service, ServiceAllowFlat)
    ->  true
    ;   format(user_error, 'Firewall blocks service: ~w (not in allowlist)~n', [Service]),
        fail
    ).

%% validate_network_access(+URL, +Firewall) is semidet.
%
% Validates network access for HTTP sources against firewall policy.
%
% @arg URL The URL being accessed
% @arg Firewall The firewall policy terms
validate_network_access(URL, Firewall) :-
    % Check if network access is explicitly denied
    (   member(network_access(denied), Firewall)
    ->  format(user_error, 'Firewall blocks network access to: ~w~n', [URL]),
        fail
    ;   true
    ),
    
    % Check host patterns (if present)
    findall(P, member(network_hosts(Ps), Firewall), Patterns),
    flatten(Patterns, FlatPatterns),
    (   FlatPatterns == []
    ->  true  % No host restrictions
    ;   validate_url_against_patterns(URL, FlatPatterns)
    ->  true
    ;   format(user_error, 'Firewall blocks network access to host: ~w~n', [URL]),
        fail
    ).

%% validate_python_imports(+Code, +Firewall) is semidet.
%
% Validates Python import statements against firewall policy.
%
% @arg Code The Python code to validate
% @arg Firewall The firewall policy terms
validate_python_imports(Code, Firewall) :-
    findall(M, member(python_modules(Ms), Firewall), Modules),
    flatten(Modules, AllowedModules),
    (   AllowedModules == []
    ->  true  % No import restrictions
    ;   extract_python_imports(Code, ImportedModules),
        forall(member(Module, ImportedModules), 
               (   member(Module, AllowedModules)
               ->  true
               ;   format(user_error, 'Firewall blocks Python import: ~w~n', [Module]),
                   fail
               ))
    ).

%% validate_file_access(+File, +Mode, +Firewall) is semidet.
%
% Validates file access against firewall policy.
%
% @arg File The file path to validate
% @arg Mode The access mode (read or write)
% @arg Firewall The firewall policy terms
validate_file_access(File, Mode, Firewall) :-
    (   Mode = read
    ->  PatternKey = file_read_patterns
    ;   Mode = write
    ->  PatternKey = file_write_patterns
    ;   format(user_error, 'Invalid file access mode: ~w~n', [Mode]),
        fail
    ),
    
    findall(P, (TermPattern =.. [PatternKey, Ps], member(TermPattern, Firewall), member(P, Ps)), Patterns),
    (   Patterns == []
    ->  true  % No file restrictions
    ;   validate_file_against_patterns(File, Patterns)
    ->  true
    ;   format(user_error, 'Firewall blocks ~w access to file: ~w~n', [Mode, File]),
        fail
    ).

%% validate_cache_directory(+CacheFile, +Firewall) is semidet.
%
% Validates cache file access against firewall policy.
%
% @arg CacheFile The cache file path
% @arg Firewall The firewall policy terms
validate_cache_directory(CacheFile, Firewall) :-
    findall(D, member(cache_dirs(Ds), Firewall), Dirs),
    flatten(Dirs, AllowedDirs),
    (   AllowedDirs == []
    ->  true  % No cache restrictions
    ;   file_directory_name(CacheFile, CacheDir),
        member_pattern(CacheDir, AllowedDirs)
    ->  true
    ;   format(user_error, 'Firewall blocks cache access: ~w~n', [CacheFile]),
        fail
    ).

%% ============================================
%% UTILITY PREDICATES
%% ============================================

%% validate_url_against_patterns(+URL, +Patterns) is semidet.
%
% Check if URL matches any of the allowed host patterns.
validate_url_against_patterns(URL, Patterns) :-
    uri_parse(URL, uri(_, _, Host, _, _)),
    member_pattern(Host, Patterns).

%% validate_file_against_patterns(+File, +Patterns) is semidet.
%
% Check if file path matches any of the allowed patterns.
validate_file_against_patterns(File, Patterns) :-
    member_pattern(File, Patterns).

%% member_pattern(+Item, +Patterns) is semidet.
%
% Check if Item matches any pattern in Patterns (supports wildcards).
member_pattern(Item, Patterns) :-
    member(Pattern, Patterns),
    (   sub_atom(Pattern, _, _, _, '*')
    ->  wildcard_match(Pattern, Item)
    ;   Item = Pattern
    ).

%% wildcard_match(+Pattern, +String) is semidet.
%
% Simple wildcard matching (supports * as wildcard).
wildcard_match(Pattern, String) :-
    split_string(Pattern, '*', '', Parts),
    wildcard_match_parts(Parts, String, 0).

wildcard_match_parts([], _, _) :- !.
wildcard_match_parts([Part], String, Offset) :-
    !,
    string_length(Part, PartLen),
    string_length(String, StringLen),
    StartPos is StringLen - PartLen,
    StartPos >= Offset,
    sub_string(String, StartPos, PartLen, 0, Part).
wildcard_match_parts([Part|Rest], String, Offset) :-
    string_length(Part, PartLen),
    sub_string(String, Offset, PartLen, _, Part),
    !,
    NewOffset is Offset + PartLen,
    wildcard_match_parts(Rest, String, NewOffset).
wildcard_match_parts([Part|Rest], String, Offset) :-
    string_length(String, StringLen),
    Offset < StringLen,
    NextOffset is Offset + 1,
    wildcard_match_parts([Part|Rest], String, NextOffset).

%% extract_python_imports(+Code, -Modules) is det.
%
% Extract import statements from Python code.
extract_python_imports(Code, Modules) :-
    split_string(Code, '\n', '', Lines),
    findall(Module, 
            (   member(Line, Lines),
                extract_import_from_line(Line, Module)
            ), Modules).

%% extract_import_from_line(+Line, -Module) is nondet.
%
% Extract module name from a Python import line.
extract_import_from_line(Line, Module) :-
    % Handle "import module" statements
    (   re_matchsub('import\\s+([a-zA-Z_][a-zA-Z0-9_]*)', Line, Sub, [])
    ->  get_dict(1, Sub, Module)
    ;   % Handle "from module import ..." statements
        re_matchsub('from\\s+([a-zA-Z_][a-zA-Z0-9_.]*)\\s+import', Line, Sub, [])
    ->  get_dict(1, Sub, ModuleAtom),
        atom_string(ModuleAtom, ModuleString),
        split_string(ModuleString, '.', '', [FirstPart|_]),
        atom_string(Module, FirstPart)
    ;   fail
    ).
