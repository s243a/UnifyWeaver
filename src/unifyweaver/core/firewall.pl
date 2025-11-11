% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

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
    get_firewall_policy/2,
    set_firewall_mode/1,
    current_firewall_mode/1,
    enforce_firewall/4,
    validate_against_firewall/3,
    validate_service/2,
    validate_network_access/2,
    validate_python_imports/2,
    validate_file_access/3,
    validate_cache_directory/2,
    % Higher-order firewall implications
    firewall_implies/2,
    firewall_implies_default/2,
    firewall_implies_disabled/2,
    derive_policy/2,
    check_derived_policy/3
]).

:- use_module(library(lists)).
:- use_module(library(uri)).

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
:- dynamic firewall_policy_warning_shown/0.
:- dynamic firewall_runtime_mode/1.

%% get_firewall_policy(+PredicateIndicator, -Firewall) is det.
%
% Resolves the applicable firewall policy for a predicate by checking
% rule-specific policies first, then falling back to the global default.
% If no policies are defined, returns [] and prints a one-time reminder.
get_firewall_policy(PredIndicator, Firewall) :-
    (   rule_firewall(PredIndicator, Firewall)
    ->  true
    ;   firewall_default(Firewall)
    ->  true
    ;   Firewall = [],
        (   firewall_policy_warning_shown
        ->  true
        ;   format(user_error, '~nINFO: No firewall rules defined. Using implicit allow.~n', []),
            format(user_error,
                   'Set firewall:firewall_default([...]) to configure security policy.~n~n', []),
            assertz(firewall_policy_warning_shown)
        )
    ).

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

%% enforce_firewall(+Context, +Target, +FinalOptions, +Firewall) is semidet.
%
% Wrapper around validate_against_firewall/3 that respects firewall modes.
% Context is an atom or predicate indicator used in warning messages.
enforce_firewall(Context, Target, FinalOptions, Firewall) :-
    (   validate_against_firewall(Target, FinalOptions, Firewall)
    ->  true
    ;   handle_firewall_mode(Context)
    ).

handle_firewall_mode(Context) :-
    current_firewall_mode(Mode),
    violation_action(Mode, Action),
    (   Action = deny
    ->  fail
    ;   Action = warn
    ->  format(user_error,
               'Firewall warning (mode ~w): continuing with ~w despite violation.~n',
               [Mode, Context])
    ;   true  % allow
    ).

violation_action(strict, deny).
violation_action(warn, warn).
violation_action(permissive, allow).
violation_action(disabled, allow).
violation_action(_, deny).

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
    findall(Ps, member(network_hosts(Ps), Firewall), Patterns),
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
    findall(Ms, member(python_modules(Ms), Firewall), Modules),
    flatten(Modules, AllowedModules),
    (   AllowedModules == []
    ->  true  % No import restrictions
    ;   extract_python_imports(Code, ImportedModules),
        forall(member(Module, ImportedModules), 
               (   % Normalize both to atoms for comparison
                   (atom(Module) -> ModuleAtom = Module ; atom_string(ModuleAtom, Module)),
                   (   member(ModuleAtom, AllowedModules)
                   ;   member(ModuleStr, AllowedModules), atom_string(ModuleAtom, ModuleStr)
                   )
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
    findall(Ds, member(cache_dirs(Ds), Firewall), Dirs),
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
    uri_components(URL, uri_components(_, Authority, _, _, _)),
    (   var(Authority)
    ->  fail  % No authority in URL
    ;   uri_authority_components(Authority, uri_authority(_, _, Host, _)),
        (   var(Host)
        ->  fail  % No host in authority
        ;   atom_string(Host, HostStr),
            member_pattern(HostStr, Patterns)
        )
    ).

%% validate_file_against_patterns(+File, +Patterns) is semidet.
%
% Check if file path matches any of the allowed patterns.
validate_file_against_patterns(File, Patterns) :-
    member_pattern(File, Patterns).

%% member_pattern(+Item, +Patterns) is semidet.
%
% Check if Item matches any pattern in Patterns (supports wildcards).
member_pattern(Item, Patterns) :-
    % Convert Item to string for consistent matching
    (   var(Item) -> fail  % Fail if Item is unbound
    ;   atom(Item) -> atom_string(Item, ItemStr)
    ;   string(Item) -> ItemStr = Item
    ;   format(atom(ItemStr), '~w', [Item])  % Fallback conversion
    ),
    member(Pattern, Patterns),
    % Convert Pattern to string
    (   var(Pattern) -> fail  % Fail if Pattern is unbound
    ;   atom(Pattern) -> atom_string(Pattern, PatternStr)
    ;   string(Pattern) -> PatternStr = Pattern
    ;   format(atom(PatternAtom), '~w', [Pattern]), atom_string(PatternAtom, PatternStr)
    ),
    (   sub_string(PatternStr, _, _, _, "*")
    ->  wildcard_match(PatternStr, ItemStr)
    ;   ItemStr = PatternStr
    ).

%% wildcard_match(+Pattern, +String) is semidet.
%
% Simple wildcard matching (supports * as wildcard).
% Both Pattern and String should be strings.
wildcard_match(Pattern, String) :-
    % Ensure both are strings
    (   string(Pattern) -> PatternStr = Pattern
    ;   atom(Pattern) -> atom_string(Pattern, PatternStr)
    ;   PatternStr = Pattern
    ),
    (   string(String) -> StringStr = String
    ;   atom(String) -> atom_string(String, StringStr)
    ;   StringStr = String
    ),
    split_string(PatternStr, '*', '', Parts),
    wildcard_match_parts(Parts, StringStr, 0).

wildcard_match_parts([], _, _) :- !.
wildcard_match_parts([Part|Rest], String, Offset) :-
    string_length(Part, PartLen),
    (   PartLen =:= 0
    ->  % Empty part means wildcard matched zero or more chars, continue with rest
        wildcard_match_parts(Rest, String, Offset)
    ;   Rest == []
    ->  % Last part must match at end
        string_length(String, StringLen),
        StartPos is StringLen - PartLen,
        StartPos >= Offset,
        sub_string(String, StartPos, PartLen, 0, Part)
    ;   % Middle part - find it and continue
        sub_string(String, Offset, PartLen, _, Part),
        !,
        NewOffset is Offset + PartLen,
        wildcard_match_parts(Rest, String, NewOffset)
    ).
wildcard_match_parts([Part|Rest], String, Offset) :-
    % Try next position if current doesn't match
    string_length(Part, PartLen),
    PartLen > 0,  % Only advance if part is non-empty
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
    % Handle "from module import ..." statements first (more specific)
    (   re_matchsub('^\\s*from\\s+([a-zA-Z_][a-zA-Z0-9_.]*)\\s+import', Line, Sub, [])
    ->  get_dict(1, Sub, ModuleAtom),
        atom_string(ModuleAtom, ModuleString),
        split_string(ModuleString, '.', '', [FirstPart|_]),
        atom_string(Module, FirstPart)
    ;   % Handle "import module" statements
        re_matchsub('^\\s*import\\s+([a-zA-Z_][a-zA-Z0-9_]*)', Line, Sub, [])
    ->  get_dict(1, Sub, Module)
    ;   fail
    ).

%% ============================================
%% HIGHER-ORDER FIREWALL IMPLICATIONS
%% ============================================
%
% This system provides logical inference for deriving firewall policies from
% fundamental rules. It demonstrates Prolog's strength in declarative reasoning
% while remaining practical and user-controllable.
%
% Architecture:
% 1. firewall_implies_default/2 - Built-in default implications (overridable)
% 2. firewall_implies/2 - User-defined implications (can override defaults)
% 3. firewall_implies_disabled/2 - Explicit disabling of implications
%
% Users can:
% - Use default implications as-is
% - Disable specific default implications
% - Add custom implications that extend or replace defaults

%% firewall_implies(+Condition, -Consequence) is nondet.
%
% Derives firewall consequences from conditions using logical inference.
% This is the main entry point that combines default and user-defined implications.
%
% First checks user-defined implications, then falls back to defaults if not disabled.
%
% @arg Condition The triggering condition (e.g., no_bash_available)
% @arg Consequence The derived policy (e.g., denied_service(powershell, executable(bash)))
%
% @example Derive that no bash means pure PowerShell required
%   ?- firewall_implies(no_bash_available, Consequence).
%   Consequence = denied_service(powershell, executable(bash)).
:- dynamic firewall_implies/2.

%% firewall_implies_default(+Condition, -Consequence) is nondet.
%
% Default built-in implications that can be overridden by users.
% These represent sensible defaults for common security scenarios.
%
% @arg Condition The triggering condition
% @arg Consequence The derived policy consequence
:- dynamic firewall_implies_default/2.

%% firewall_implies_disabled(+Condition, +Consequence) is nondet.
%
% Explicitly disabled implications. Users can assert these to prevent
% specific default implications from taking effect.
%
% @arg Condition The condition to disable
% @arg Consequence The consequence to disable
%
% @example Disable the default "no bash → deny bash service" implication
%   :- assertz(firewall:firewall_implies_disabled(no_bash_available,
%                                                  denied_service(powershell, executable(bash)))).
:- dynamic firewall_implies_disabled/2.

%% derive_policy(+Condition, -Policies) is det.
%
% Derives all firewall policies from a given condition.
% Collects both user-defined and default implications (unless disabled).
%
% @arg Condition The triggering condition
% @arg Policies List of derived policy consequences
%
% @example Derive all policies from no_bash_available
%   ?- derive_policy(no_bash_available, Policies).
%   Policies = [denied_service(powershell, executable(bash))].
derive_policy(Condition, Policies) :-
    findall(Policy, (
        % User-defined implications take precedence
        (   firewall_implies(Condition, Policy)
        ;   % Fall back to defaults if not disabled
            firewall_implies_default(Condition, Policy),
            \+ firewall_implies_disabled(Condition, Policy)
        )
    ), Policies).

%% check_derived_policy(+Condition, +ExpectedPolicies, -Result) is det.
%
% Check if a condition derives expected policies. Useful for testing.
%
% @arg Condition The triggering condition
% @arg ExpectedPolicies List of expected policy consequences
% @arg Result true if all expected policies are derived, false otherwise
%
% @example Check if no_bash_available denies bash service
%   ?- check_derived_policy(no_bash_available,
%                          [denied_service(powershell, executable(bash))],
%                          Result).
%   Result = true.
check_derived_policy(Condition, ExpectedPolicies, Result) :-
    derive_policy(Condition, DerivedPolicies),
    (   subset(ExpectedPolicies, DerivedPolicies)
    ->  Result = true
    ;   Result = false
    ).

%% ============================================
%% DEFAULT IMPLICATIONS
%% ============================================
%
% Built-in default implications for common security scenarios.
% Users can disable these by asserting firewall_implies_disabled/2.

%% 1. No bash available → Deny bash service for PowerShell
%
% If bash is not available on the system, PowerShell targets cannot use
% Bash-as-a-Service mode and must use pure PowerShell implementations.
firewall_implies_default(no_bash_available,
                        denied(service(powershell, executable(bash)))).

%% 2. Bash target denied → Deny bash service for all targets
%
% If bash itself is denied as a target, then no other target can use
% bash as a service either.
firewall_implies_default(denied_target_language(bash),
                        denied(service(_, executable(bash)))).

%% 3. Network access denied → Deny all network services
%
% If network access is globally denied, block all services that require
% network connectivity.
firewall_implies_default(network_access(denied),
                        denied(service(_, network_access(_)))).

firewall_implies_default(network_access(denied),
                        network_access(denied)).

%% 4. Specific executable denied → Deny as service
%
% If a specific executable is denied (e.g., python), deny it as a service
% for all targets.
firewall_implies_default(denied_executable(Tool),
                        denied(service(_, executable(Tool)))).

%% 5. Strict security policy → Prefer built-in features over executables
%
% In strict security mode, prefer language built-ins (cmdlets, built-in modules)
% over external executable invocations.
firewall_implies_default(security_policy(strict),
                        prefer(service(powershell, cmdlet(_)),
                               service(powershell, executable(_)))).

firewall_implies_default(security_policy(strict),
                        prefer(service(python, builtin(_)),
                               service(python, executable(_)))).

%% 6. Restricted environment → Deny external service calls
%
% In restricted/sandboxed environments, deny all external service invocations
% and only allow language built-ins.
firewall_implies_default(environment(restricted),
                        denied(service(_, executable(_)))).

firewall_implies_default(environment(restricted),
                        denied(service(_, network_access(_)))).

%% 7. Target language not allowed → Deny all services for that target
%
% If a target language is not in the allowed list, deny all services for it.
firewall_implies_default(denied_target_language(Target),
                        denied(execution(Target))).

%% 8. Pure mode preference → Prefer pure implementations
%
% When pure mode is preferred (no external dependencies), prefer native
% implementations over Bash-as-a-Service or other external tools.
firewall_implies_default(prefer_pure_mode(powershell),
                        prefer(service(powershell, cmdlet(_)),
                               service(powershell, executable(bash)))).

firewall_implies_default(prefer_pure_mode(python),
                        prefer(service(python, library(_)),
                               service(python, executable(_)))).

%% 9. Offline mode → Deny network access
%
% In offline mode, deny all network-based services.
firewall_implies_default(mode(offline),
                        network_access(denied)).

firewall_implies_default(mode(offline),
                        denied(service(_, http(_)))).

%% 10. Portable/cross-platform requirement → Prefer portable tools
%
% When portability is required, prefer tools available on all platforms.
firewall_implies_default(require_portable,
                        prefer(service(_, cmdlet(_)),
                               service(_, executable(awk)))).  % AWK not always available

firewall_implies_default(require_portable,
                        prefer(service(_, builtin(_)),
                               service(_, executable(_)))).

%% ============================================
%% NETWORK ACCESS IMPLICATIONS
%% ============================================
%
% Higher-order implications for network access control based on
% security contexts and environmental constraints.

%% 11. External network denied → Restrict to internal hosts only
%
% When external network access is denied, only allow connections
% to internal/localhost addresses.
firewall_implies_default(deny_external_network,
                        denied(service(_, network_access(external)))).

firewall_implies_default(deny_external_network,
                        network_hosts(['localhost', '127.0.0.1', '*.local', '*.internal.*'])).

%% 12. Corporate environment → Whitelist internal domains
%
% In corporate environments, typically only allow connections to
% company-internal domains and trusted external APIs.
firewall_implies_default(environment(corporate),
                        network_hosts(['*.internal.company.com', 'localhost'])).

firewall_implies_default(environment(corporate),
                        denied(service(_, network_access(untrusted)))).

%% 13. Sandboxed/restricted environment → Deny external network
%
% Restricted environments should not access external networks.
firewall_implies_default(environment(restricted),
                        denied(service(_, network_access(external)))).

%% 14. Development environment → Allow localhost only
%
% Development environments typically only need localhost access.
firewall_implies_default(environment(development),
                        network_hosts(['localhost', '127.0.0.1', '*.local'])).

%% 15. Production environment → Require explicit whitelisting
%
% Production should only access pre-approved external services.
firewall_implies_default(environment(production),
                        require_network_whitelist).

%% 16. Offline mode → Deny all network access
%
% Offline mode extends to HTTP sources and any network-based data access.
firewall_implies_default(mode(offline),
                        denied(service(_, source(http)))).

%% 17. Privacy-sensitive mode → Block external tracking/analytics
%
% When privacy is a concern, block common analytics and tracking domains.
firewall_implies_default(privacy_mode(enabled),
                        denied(network_hosts(['*analytics*', '*tracking*', '*doubleclick*']))).

%% 18. Testing/CI environment → Allow test APIs only
%
% CI/CD environments should only access test/mock APIs.
firewall_implies_default(environment(ci),
                        network_hosts(['*.test.*', 'localhost', 'mock.*'])).

%% 19. Air-gapped system → Complete network denial
%
% Air-gapped systems have no network access whatsoever.
firewall_implies_default(system_type(air_gapped),
                        network_access(denied)).

firewall_implies_default(system_type(air_gapped),
                        denied(service(_, network_access(_)))).

%% 20. VPN-required policy → Enforce VPN for external access
%
% Some organizations require VPN for any external network access.
% This is informational - actual VPN enforcement is outside firewall scope.
firewall_implies_default(network_policy(vpn_required),
                        require_vpn_for_external).

%% 21. Termux/Mobile environment → Alternative SSH port (8022)
%
% Termux on Android uses port 8022 for SSH instead of standard port 22,
% which is typically blocked on mobile devices for security reasons.
% This allows SSH-based services to work in Termux environments.
firewall_implies_default(environment(termux),
                        network_hosts(['localhost:8022', '127.0.0.1:8022', 'localhost', '127.0.0.1'])).

firewall_implies_default(environment(termux),
                        prefer(service(_, port(8022)), service(_, port(22)))).

%% 22. Mobile/restricted port environment → Prefer alternative ports
%
% Some mobile or restricted environments block standard service ports.
% Allow common alternative ports (8080 for HTTP, 8443 for HTTPS, 8022 for SSH).
firewall_implies_default(environment(mobile_restricted_ports),
                        network_hosts(['*:8080', '*:8443', '*:8022', '*:3000'])).

firewall_implies_default(environment(mobile_restricted_ports),
                        prefer(service(_, port(Alternative)), service(_, port(Standard)))) :-
    member(Alternative-Standard, [8022-22, 8080-80, 8443-443, 3000-80]).
%% set_firewall_mode(+Mode) is det.
%
% Sets the firewall enforcement mode.
% Supported modes:
% - strict     : Violations block compilation (default)
% - warn       : Allow compilation but emit warning message
% - permissive : Allow compilation, no extra warning beyond failure reason
% - disabled   : Skip enforcement (firewall checks still run for logging)
set_firewall_mode(Mode) :-
    retractall(firewall_runtime_mode(_)),
    assertz(firewall_runtime_mode(Mode)).

%% current_firewall_mode(-Mode) is det.
%
% Retrieves the current firewall mode. Falls back to strict if undefined.
% If firewall_v2 is loaded, uses its mode setting unless overridden locally.
current_firewall_mode(Mode) :-
    (   firewall_runtime_mode(Mode)
    ->  true
    ;   current_module(firewall_v2),
        catch(firewall_v2:firewall_mode(Mode), _, fail)
    ->  true
    ;   Mode = strict
    ).
