% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
%% firewall_v2.pl - Fundamental Rules Firewall Architecture
%
% Extension to the existing firewall system with fundamental target language
% and service policies. This module provides the foundation for deriving
% higher-level preferences like powershell_mode selection.
%
% @author John William Creighton (@s243a), Claude Code
% @license MIT OR Apache-2.0
% @see planning/FIREWALL_TODO.md for design rationale

:- module(firewall_v2, [
    % Fundamental policy predicates
    allowed_target_language/1,
    denied_target_language/1,
    allowed_service/2,
    denied_service/2,
    preferred_service/2,

    % Firewall mode
    firewall_mode/1,
    set_firewall_mode/1,

    % Implication system
    firewall_implies/2,
    firewall_implies_default/2,
    firewall_implies_disabled/2,

    % Query predicates for compilers
    check_target_language/2,
    check_service/3,
    derive_powershell_mode/2,
    supports_pure_powershell/1,

    % PowerShell mode preferences
    preferred_powershell_mode/1,
    set_preferred_powershell_mode/1,

    % Predicate-level firewall rules (Phase 10)
    predicate_firewall_mode/2,
    set_predicate_firewall_mode/2,
    clear_predicate_firewall_mode/1,
    predicate_allowed_service/3,
    predicate_denied_service/3,
    set_predicate_service_policy/4,
    check_predicate_firewall/3,
    derive_predicate_powershell_mode/3,
    effective_powershell_mode/3,

    % Tool availability predicates
    tool_availability_policy/1,
    require_tool/1,
    prefer_available_tools/1,
    allowed_tool/1,
    denied_tool/1,
    check_tool_availability/3,
    derive_compilation_mode_with_tools/3,
    derive_mode_with_alternatives/3,
    detect_and_configure_bash_availability/0,
    auto_configure_firewall_for_missing_tools/1,

    % Network access predicates
    network_access_policy/1,
    allowed_url_pattern/1,
    denied_url_pattern/1,
    allowed_domain/1,
    denied_domain/1,
    check_url_access/2,
    check_domain_access/2,
    extract_domain/2,
    url_matches_pattern/2,
    domain_matches_pattern/2,

    % Policy templates
    load_firewall_policy/1,

    % Testing
    test_predicate_firewall/0
]).

:- use_module(library(lists)).
:- use_module('tool_detection').

% Discontiguous for supports_pure_powershell (source types + predicate indicators)
:- discontiguous supports_pure_powershell/1.

%% ============================================
%% DYNAMIC PREDICATES
%% ============================================

%% Fundamental firewall rules (asserted by policies)
:- dynamic allowed_target_language/1.
:- dynamic denied_target_language/1.
:- dynamic allowed_service/2.      % allowed_service(TargetLang, Service)
:- dynamic denied_service/2.       % denied_service(TargetLang, Service)
:- dynamic preferred_service/2.    % preferred_service(TargetLang, Service)

%% Firewall mode
:- dynamic firewall_mode/1.        % strict | permissive | disabled

%% Implication system
:- dynamic firewall_implies_disabled/2.  % User overrides

%% Tool availability policies
:- dynamic tool_availability_policy/1.   % warn | forbid | ignore | list_required
:- dynamic require_tool/1.               % require_tool(ToolName)
:- dynamic prefer_available_tools/1.     % prefer_available_tools(true|false)
:- dynamic allowed_tool/1.               % allowed_tool(ToolName)
:- dynamic denied_tool/1.                % denied_tool(ToolName)

%% Network access policies
:- dynamic network_access_policy/1.      % allow_all | deny_all | whitelist | blacklist
:- dynamic allowed_url_pattern/1.        % allowed_url_pattern(Pattern)
:- dynamic denied_url_pattern/1.         % denied_url_pattern(Pattern)
:- dynamic allowed_domain/1.             % allowed_domain(Domain)
:- dynamic denied_domain/1.              % denied_domain(Domain)

%% PowerShell mode preferences
:- dynamic preferred_powershell_mode/1.  % pure | baas | auto (default: auto)

%% Predicate-level firewall rules (Phase 10)
%% These override default-level rules for specific predicates
:- dynamic predicate_firewall_mode/2.    % predicate_firewall_mode(Pred/Arity, Mode)
:- dynamic predicate_allowed_service/3.  % predicate_allowed_service(Pred/Arity, Target, Service)
:- dynamic predicate_denied_service/3.   % predicate_denied_service(Pred/Arity, Target, Service)

%% ============================================
%% DEFAULT CONFIGURATION
%% ============================================

% Default mode: permissive (developer-friendly)
firewall_mode(permissive) :-
    \+ (firewall_mode(strict) ; firewall_mode(disabled)).

%% set_firewall_mode(+Mode)
%  Set firewall mode (strict|permissive|disabled)
set_firewall_mode(Mode) :-
    retractall(firewall_mode(_)),
    assertz(firewall_mode(Mode)),
    format('[Firewall] Mode set to: ~w~n', [Mode]).

%% set_preferred_powershell_mode(+Mode)
%  Set preferred PowerShell compilation mode (pure|baas|auto)
%
%  - pure: Always use pure PowerShell when supported
%  - baas: Always use Bash-as-a-Service when allowed
%  - auto: Let firewall automatically decide (default)
%
%  Example:
%    ?- set_preferred_powershell_mode(pure).  % Forces pure mode
set_preferred_powershell_mode(Mode) :-
    (   memberchk(Mode, [pure, baas, auto])
    ->  retractall(preferred_powershell_mode(_)),
        assertz(preferred_powershell_mode(Mode)),
        format('[Firewall] PowerShell mode preference set to: ~w~n', [Mode])
    ;   format('[Firewall Error] Invalid mode: ~w (expected: pure|baas|auto)~n', [Mode]),
        fail
    ).

%% ============================================
%% DEFAULT TARGET LANGUAGE POLICIES
%% ============================================

% By default, allow common target languages (unless overridden)
allowed_target_language(bash) :-
    \+ denied_target_language(bash).

allowed_target_language(powershell) :-
    \+ denied_target_language(powershell).

%% ============================================
%% DEFAULT SERVICE POLICIES
%% ============================================

% Bash services (default allowed unless explicitly denied)
allowed_service(bash, executable(awk)) :-
    \+ denied_service(bash, executable(awk)).

allowed_service(bash, executable(sed)) :-
    \+ denied_service(bash, executable(sed)).

allowed_service(bash, executable(grep)) :-
    \+ denied_service(bash, executable(grep)).

allowed_service(bash, executable(jq)) :-
    \+ denied_service(bash, executable(jq)).

allowed_service(bash, executable(curl)) :-
    \+ denied_service(bash, executable(curl)).

allowed_service(bash, executable(python3)) :-
    \+ denied_service(bash, executable(python3)).

allowed_service(bash, executable(sqlite3)) :-
    \+ denied_service(bash, executable(sqlite3)).

% PowerShell services (default allowed unless explicitly denied)
allowed_service(powershell, cmdlet(import_csv)) :-
    \+ denied_service(powershell, cmdlet(import_csv)).

allowed_service(powershell, cmdlet(convertfrom_json)) :-
    \+ denied_service(powershell, cmdlet(convertfrom_json)).

allowed_service(powershell, cmdlet(invoke_restmethod)) :-
    \+ denied_service(powershell, cmdlet(invoke_restmethod)).

allowed_service(powershell, cmdlet(get_content)) :-
    \+ denied_service(powershell, cmdlet(get_content)).

% PowerShell can use bash via compatibility layer (default)
allowed_service(powershell, executable(bash)) :-
    \+ denied_service(powershell, executable(bash)).

%% ============================================
%% FIREWALL IMPLIES - DEFAULT IMPLICATIONS
%% ============================================

%% firewall_implies(+Condition, -Consequence)
%  Derives firewall policies from fundamental rules and conditions.
%  Users can override defaults by asserting firewall_implies_disabled/2.

firewall_implies(Condition, Consequence) :-
    firewall_implies_default(Condition, Consequence),
    \+ firewall_implies_disabled(Condition, Consequence).

% Default implication: no bash available → deny bash service for PowerShell
firewall_implies_default(no_bash_available,
                        denied_service(powershell, executable(bash))).

% Default implication: bash target denied → deny bash service for PowerShell
firewall_implies_default(denied_target_language(bash),
                        denied_service(powershell, executable(bash))).

% Default implication: denied executable → deny service for all targets
firewall_implies_default(denied_executable(Tool),
                        denied_service(_, executable(Tool))).

% Default implication: network access denied → deny network services
firewall_implies_default(network_access(denied),
                        denied_service(_, network_access(_))).

%% ============================================
%% QUERY PREDICATES FOR COMPILERS
%% ============================================

%% check_target_language(+Language, -Result)
%  Check if a target language is allowed by firewall.
%  Result: allow | deny(Reason) | warn(Reason)
check_target_language(Language, Result) :-
    (   denied_target_language(Language)
    ->  Result = deny(target_language_denied)
    ;   allowed_target_language(Language)
    ->  Result = allow
    ;   firewall_mode(Mode),
        (   Mode = strict
        ->  Result = deny(not_in_allowlist)
        ;   Mode = permissive
        ->  Result = warn(not_explicitly_allowed)
        ;   Mode = disabled
        ->  Result = allow
        )
    ).

%% check_service(+TargetLanguage, +Service, -Result)
%  Check if a service is allowed for a target language.
%  Result: allow | deny(Reason) | warn(Reason)
check_service(TargetLanguage, Service, Result) :-
    (   denied_service(TargetLanguage, Service)
    ->  Result = deny(service_denied)
    ;   firewall_implies(_, denied_service(TargetLanguage, Service))
    ->  Result = deny(service_denied_by_implication)
    ;   allowed_service(TargetLanguage, Service)
    ->  (   preferred_service(TargetLanguage, Service)
        ->  Result = allow
        ;   Result = allow
        )
    ;   firewall_mode(Mode),
        (   Mode = strict
        ->  Result = deny(not_in_allowlist)
        ;   Mode = permissive
        ->  Result = warn(not_explicitly_allowed)
        ;   Mode = disabled
        ->  Result = allow
        )
    ).

%% derive_powershell_mode(+SourceType, -Mode)
%  Derive PowerShell compilation mode from firewall policies.
%  This is called by powershell_compiler when mode is 'auto'.
%
%  Mode: pure | baas | auto_with_preference(pure|baas)
derive_powershell_mode(SourceType, Mode) :-
    % Check if bash service is allowed for PowerShell
    check_service(powershell, executable(bash), BashResult),

    % Check if source type supports pure PowerShell
    (   supports_pure_powershell(SourceType)
    ->  PureSupported = true
    ;   PureSupported = false
    ),

    % Derive mode based on firewall and capability
    (   BashResult = deny(_), PureSupported = true
    ->  % Bash denied but pure supported → must use pure
        Mode = pure
    ;   BashResult = deny(_), PureSupported = false
    ->  % Bash denied and pure not supported → fail
        format('[Firewall Error] Bash denied for PowerShell, but ~w has no pure implementation~n', [SourceType]),
        fail
    ;   BashResult = allow, PureSupported = true
    ->  % Both allowed → prefer pure (native cmdlets better)
        Mode = auto_with_preference(pure)
    ;   BashResult = allow, PureSupported = false
    ->  % Only BaaS available
        Mode = baas
    ;   BashResult = warn(_), PureSupported = true
    ->  % Bash allowed with warning → prefer pure
        Mode = auto_with_preference(pure)
    ;   % Default: auto
        Mode = auto
    ).

% Helper: Check if source type supports pure PowerShell
% Phase 1: External data sources
supports_pure_powershell(csv).
supports_pure_powershell(json).
supports_pure_powershell(http).

% Phase 1: Internal predicates
supports_pure_powershell(facts).         % Static facts
supports_pure_powershell(rules).         % Rules with joins/negation

% Phase 2: Recursion patterns
supports_pure_powershell(recursion).     % All recursion patterns
supports_pure_powershell(fixpoint).      % Transitive closure

% Phase 3: Data partitioning
supports_pure_powershell(partitioning).  % Data partitioning strategies

%% ============================================
%% POLICY TEMPLATES
%% ============================================

%% load_firewall_policy(+PolicyName)
%  Load a predefined firewall policy template.
%
%  Available policies:
%  - permissive - Allow everything (default)
%  - strict_security - Deny by default, explicit allow only
%  - pure_powershell - Force pure PowerShell for CSV/JSON/HTTP
%  - no_network - Block all network access
load_firewall_policy(permissive) :-
    format('[Firewall] Loading permissive policy~n', []),
    set_firewall_mode(permissive),
    % Allow all common languages and services (defaults handle this)
    !.

load_firewall_policy(strict_security) :-
    format('[Firewall] Loading strict_security policy~n', []),
    set_firewall_mode(strict),
    % In strict mode, only explicitly allowed items pass
    % Add your explicit allows here
    assertz(allowed_target_language(bash)),
    assertz(allowed_target_language(powershell)),
    !.

load_firewall_policy(pure_powershell) :-
    format('[Firewall] Loading pure_powershell policy~n', []),
    set_firewall_mode(permissive),
    % Deny bash service for PowerShell → forces pure mode
    assertz(denied_service(powershell, executable(bash))),
    format('[Firewall] PowerShell will use pure mode (no bash)~n', []),
    !.

load_firewall_policy(no_network) :-
    format('[Firewall] Loading no_network policy~n', []),
    set_firewall_mode(permissive),
    % Deny all network access
    assertz(network_access_policy(deny_all)),
    assertz(denied_service(_, network_access(_))),
    assertz(denied_service(_, executable(curl))),
    assertz(denied_service(_, executable(wget))),
    format('[Firewall] Network access blocked~n', []),
    !.

load_firewall_policy(whitelist_domains(Domains)) :-
    format('[Firewall] Loading whitelist_domains policy~n', []),
    set_firewall_mode(permissive),
    % Set whitelist mode
    assertz(network_access_policy(whitelist)),
    % Add allowed domains
    forall(member(Domain, Domains),
           assertz(allowed_domain(Domain))),
    format('[Firewall] Only domains ~w are allowed~n', [Domains]),
    !.

load_firewall_policy(auto_detect_environment) :-
    format('[Firewall] Auto-detecting environment~n', []),
    set_firewall_mode(permissive),
    detect_and_configure_bash_availability,
    !.

load_firewall_policy(PolicyName) :-
    format('[Firewall Error] Unknown policy: ~w~n', [PolicyName]),
    fail.

%% ============================================
%% TOOL AVAILABILITY INTEGRATION
%% ============================================

%% detect_and_configure_bash_availability
%  Detect if bash is available and configure firewall accordingly.
%  This is useful for auto-detecting restricted environments (e.g., Windows without WSL).
%
%  Detection checks:
%  1. Standard bash in PATH
%  2. WSL bash (wsl.exe --version)
%  3. Git Bash (common Windows install location)
%  4. Cygwin bash (common Windows install location)
%
%  If bash is unavailable, optionally adds firewall rule to deny bash for PowerShell.

detect_and_configure_bash_availability :-
    % Check if bash is available
    tool_detection:detect_tool_availability(bash, BashStatus),

    (   BashStatus = available
    ->  % Bash is available
        format('[Firewall] Bash detected - available~n', []),
        % Don't add any firewall rules (allow bash)
        true

    ;   BashStatus = unavailable(_)
    ->  % Bash not available
        format('[Firewall] Bash not detected - unavailable~n', []),

        % Check if we should auto-configure firewall
        (   auto_configure_firewall_for_missing_tools(true)
        ->  % Add firewall rule to deny bash for PowerShell
            assertz(denied_service(powershell, executable(bash))),
            format('[Firewall] Added rule: deny bash for PowerShell (not available)~n', [])
        ;   % Just inform, don't add rule
            format('[Firewall] Note: bash unavailable but no auto-configuration~n', [])
        )
    ).

% Helper predicate - can be overridden by user
:- dynamic auto_configure_firewall_for_missing_tools/1.
% Default: don't auto-configure (let user decide)
% User can enable with: assertz(auto_configure_firewall_for_missing_tools(true)).

%% check_tool_availability(+Tool, +TargetLanguage, -Result)
%  Check if a tool is available, considering firewall policies.
%  This integrates tool_detection with firewall rules.
%
%  Tool: Tool name (e.g., bash, jq, import_csv)
%  TargetLanguage: Target language (e.g., bash, powershell)
%  Result: available | unavailable(Reason) | denied(Reason)

check_tool_availability(Tool, TargetLanguage, Result) :-
    % Check firewall policy first
    (   denied_tool(Tool)
    ->  Result = denied(tool_denied_by_firewall)

    % Check if tool is denied for this target language
    ;   tool_detection:tool_executable(Tool, _),
        denied_service(TargetLanguage, executable(Tool))
    ->  Result = denied(service_denied)

    % Check actual availability
    ;   tool_detection:detect_tool_availability(Tool, Status),
        (   Status = available
        ->  Result = available
        ;   Status = unavailable(Reason)
        ->  Result = unavailable(Reason)
        )
    ).

%% derive_compilation_mode_with_tools(+SourceType, +RequiredTools, -Mode)
%  Derive compilation mode considering both firewall policies
%  and tool availability.
%
%  This extends derive_powershell_mode/2 with tool availability checking.

derive_compilation_mode_with_tools(SourceType, RequiredTools, Mode) :-
    % Check if all required tools are available
    tool_detection:check_all_tools(RequiredTools, ToolStatus),

    % Get preference for available tools
    (   prefer_available_tools(true)
    ->  PreferAvailable = true
    ;   PreferAvailable = false
    ),

    % Derive mode based on tool availability and preferences
    (   ToolStatus = all_available
    ->  % All tools available, use normal derivation
        derive_powershell_mode(SourceType, Mode)

    ;   ToolStatus = missing(MissingTools),
        PreferAvailable = true
    ->  % Some tools missing, prefer alternatives
        derive_mode_with_alternatives(SourceType, MissingTools, Mode)

    ;   ToolStatus = missing(MissingTools)
    ->  % Tools missing, check policy
        (   tool_availability_policy(forbid)
        ->  format('[Firewall Error] Required tools missing: ~w~n', [MissingTools]),
            fail
        ;   tool_availability_policy(warn)
        ->  format('[Firewall Warning] Missing tools: ~w~n', [MissingTools]),
            derive_powershell_mode(SourceType, Mode)
        ;   % Default: list required
            derive_powershell_mode(SourceType, Mode)
        )
    ).

%% derive_mode_with_alternatives(+SourceType, +MissingTools, -Mode)
%  Derive compilation mode preferring available alternatives

derive_mode_with_alternatives(SourceType, MissingTools, Mode) :-
    % If bash is missing and source supports pure PowerShell
    (   memberchk(bash, MissingTools),
        supports_pure_powershell(SourceType)
    ->  Mode = pure,
        format('[Firewall] Bash unavailable, using pure PowerShell mode~n', [])

    % Otherwise use standard derivation
    ;   derive_powershell_mode(SourceType, Mode)
    ).

%% ============================================
%% NETWORK ACCESS CONTROL
%% ============================================

%% check_url_access(+URL, -Result)
%  Check if a URL is allowed by firewall policies.
%  Result: allow | deny(Reason) | warn(Reason)
%
%  Checks in this order:
%  1. Global network policy (deny_all, allow_all)
%  2. Specific URL patterns (denied_url_pattern, allowed_url_pattern)
%  3. Domain patterns (denied_domain, allowed_domain)
%  4. Firewall mode (strict/permissive)

check_url_access(URL, Result) :-
    % Check global network policy first
    (   network_access_policy(deny_all)
    ->  Result = deny(network_access_denied)

    ;   network_access_policy(allow_all)
    ->  Result = allow

    % Check denied URL patterns
    ;   denied_url_pattern(Pattern),
        url_matches_pattern(URL, Pattern)
    ->  Result = deny(url_pattern_denied(Pattern))

    % Check allowed URL patterns in whitelist mode (only if URL patterns are defined)
    ;   network_access_policy(whitelist),
        allowed_url_pattern(_),  % Only check if we have URL patterns
        \+ (allowed_url_pattern(Pattern),
            url_matches_pattern(URL, Pattern))
    ->  Result = deny(not_in_whitelist)

    % Check domain-based rules
    ;   extract_domain(URL, Domain),
        check_domain_access(Domain, DomainResult)
    ->  Result = DomainResult

    % Check firewall mode
    ;   firewall_mode(strict)
    ->  Result = deny(strict_mode_requires_explicit_allow)

    ;   firewall_mode(permissive)
    ->  Result = allow

    ;   firewall_mode(disabled)
    ->  Result = allow
    ).

%% check_domain_access(+Domain, -Result)
%  Check if a domain is allowed.

check_domain_access(Domain, Result) :-
    % Check denied domains
    (   denied_domain(DeniedPattern),
        domain_matches_pattern(Domain, DeniedPattern)
    ->  Result = deny(domain_denied(DeniedPattern))

    % Check allowed domains (whitelist mode)
    ;   network_access_policy(whitelist),
        \+ (allowed_domain(AllowedPattern),
            domain_matches_pattern(Domain, AllowedPattern))
    ->  Result = deny(domain_not_in_whitelist)

    % Check if specific domain is allowed
    ;   allowed_domain(AllowedPattern),
        domain_matches_pattern(Domain, AllowedPattern)
    ->  Result = allow

    % Default: check firewall mode
    ;   firewall_mode(strict)
    ->  Result = deny(strict_mode_requires_explicit_allow)

    ;   Result = allow
    ).

%% url_matches_pattern(+URL, +Pattern)
%  Check if URL matches a pattern (supports wildcards and regex)

url_matches_pattern(URL, Pattern) :-
    % Simple substring match for now
    (   atom(Pattern)
    ->  sub_atom(URL, _, _, _, Pattern)
    ;   % Pattern could be more complex (regex, etc.)
        sub_string(URL, _, _, _, Pattern)
    ).

%% domain_matches_pattern(+Domain, +Pattern)
%  Check if domain matches a pattern

domain_matches_pattern(Domain, Pattern) :-
    % Exact match
    (   Domain = Pattern
    ->  true
    % Subdomain match (e.g., Pattern = 'example.com', Domain = 'api.example.com')
    ;   atom_concat(_, Pattern, Domain)
    % Pattern with wildcard (e.g., '*.example.com')
    ;   sub_atom(Pattern, 0, 1, _, '*'),
        sub_atom(Pattern, 1, _, 0, BaseDomain),
        atom_concat(_, BaseDomain, Domain)
    ).

%% extract_domain(+URL, -Domain)
%  Extract domain from URL

extract_domain(URL, Domain) :-
    % Remove protocol (http://, https://)
    (   sub_atom(URL, Before, 3, _, '://')
    ->  Start is Before + 3,
        sub_atom(URL, Start, _, 0, URLNoProt)
    ;   URLNoProt = URL
    ),

    % Extract domain (everything before first '/')
    (   sub_atom(URLNoProt, DomainEnd, _, _, '/')
    ->  sub_atom(URLNoProt, 0, DomainEnd, _, Domain)
    ;   Domain = URLNoProt
    ).

%% ============================================
%% PREDICATE-LEVEL FIREWALL RULES (Phase 10)
%% ============================================
%%
%% Predicate-level rules override default-level rules for specific predicates.
%% This allows fine-grained control over compilation modes and service access.
%%
%% Priority order (highest to lowest):
%% 1. Predicate-level denied_service (always blocks)
%% 2. Predicate-level firewall_mode (overrides default)
%% 3. Default-level denied_service
%% 4. Default-level firewall_mode
%%
%% Usage examples:
%%   % Force pure PowerShell for user/2 (no bash allowed)
%%   ?- set_predicate_firewall_mode(user/2, pure).
%%
%%   % Allow bash for log_parser/3 even if default is pure
%%   ?- set_predicate_firewall_mode(log_parser/3, baas).
%%
%%   % Deny network access for sensitive_data/2
%%   ?- set_predicate_service_policy(sensitive_data/2, powershell, network_access, deny).
%%
%% ============================================

%% set_predicate_firewall_mode(+Predicate, +Mode)
%  Set firewall mode for a specific predicate.
%  Mode: pure | baas | auto | inherit (use default)
%
%  Example:
%    ?- set_predicate_firewall_mode(user/2, pure).
%    [Firewall] Predicate user/2 firewall mode set to: pure
set_predicate_firewall_mode(Predicate, Mode) :-
    (   memberchk(Mode, [pure, baas, auto, inherit])
    ->  retractall(predicate_firewall_mode(Predicate, _)),
        (   Mode = inherit
        ->  % inherit means remove predicate-level rule
            format('[Firewall] Predicate ~w will inherit default firewall mode~n', [Predicate])
        ;   assertz(predicate_firewall_mode(Predicate, Mode)),
            format('[Firewall] Predicate ~w firewall mode set to: ~w~n', [Predicate, Mode])
        )
    ;   format('[Firewall Error] Invalid mode: ~w (expected: pure|baas|auto|inherit)~n', [Mode]),
        fail
    ).

%% clear_predicate_firewall_mode(+Predicate)
%  Remove predicate-level firewall mode (inherit from default).
clear_predicate_firewall_mode(Predicate) :-
    retractall(predicate_firewall_mode(Predicate, _)),
    retractall(predicate_allowed_service(Predicate, _, _)),
    retractall(predicate_denied_service(Predicate, _, _)),
    format('[Firewall] Predicate ~w firewall rules cleared~n', [Predicate]).

%% set_predicate_service_policy(+Predicate, +Target, +Service, +Policy)
%  Set service policy for a specific predicate.
%  Policy: allow | deny
%
%  Example:
%    ?- set_predicate_service_policy(api_call/2, powershell, executable(bash), deny).
%    [Firewall] Predicate api_call/2: deny powershell service executable(bash)
set_predicate_service_policy(Predicate, Target, Service, allow) :-
    retractall(predicate_denied_service(Predicate, Target, Service)),
    assertz(predicate_allowed_service(Predicate, Target, Service)),
    format('[Firewall] Predicate ~w: allow ~w service ~w~n', [Predicate, Target, Service]).

set_predicate_service_policy(Predicate, Target, Service, deny) :-
    retractall(predicate_allowed_service(Predicate, Target, Service)),
    assertz(predicate_denied_service(Predicate, Target, Service)),
    format('[Firewall] Predicate ~w: deny ~w service ~w~n', [Predicate, Target, Service]).

%% check_predicate_firewall(+Predicate, +Target, -Result)
%  Check firewall rules for a specific predicate.
%  Returns the effective mode considering predicate-level and default rules.
%
%  Result: allow(Mode) | deny(Reason)
check_predicate_firewall(Predicate, Target, Result) :-
    % Check predicate-level denied services first (highest priority)
    (   predicate_denied_service(Predicate, Target, Service)
    ->  Result = deny(predicate_service_denied(Service))

    % Check predicate-level mode
    ;   predicate_firewall_mode(Predicate, PredicateMode)
    ->  (   PredicateMode = pure
        ->  % Check if pure is supported
            (   supports_pure_powershell(Predicate)
            ->  Result = allow(pure)
            ;   Result = deny(pure_not_supported)
            )
        ;   PredicateMode = baas
        ->  % Check if bash service is allowed at default level
            (   check_service(Target, executable(bash), ServiceResult),
                ServiceResult = allow
            ->  Result = allow(baas)
            ;   Result = deny(bash_denied_at_default_level)
            )
        ;   PredicateMode = auto
        ->  Result = allow(auto)
        )

    % Fall back to default-level rules
    ;   check_service(Target, executable(bash), BashResult),
        (   BashResult = deny(_)
        ->  (   supports_pure_powershell(Predicate)
            ->  Result = allow(pure)
            ;   Result = deny(no_viable_mode)
            )
        ;   Result = allow(auto)
        )
    ).

%% derive_predicate_powershell_mode(+Predicate, +SourceType, -Mode)
%  Derive PowerShell compilation mode for a specific predicate.
%  Considers both predicate-level and default-level rules.
%
%  Mode: pure | baas | auto_with_preference(pure|baas)
derive_predicate_powershell_mode(Predicate, SourceType, Mode) :-
    % Check predicate-level mode first
    (   predicate_firewall_mode(Predicate, PredicateMode),
        PredicateMode \= inherit
    ->  % Predicate has explicit mode
        (   PredicateMode = pure
        ->  (   supports_pure_powershell(SourceType)
            ->  Mode = pure
            ;   format('[Firewall Error] ~w requires pure mode but ~w has no pure implementation~n',
                       [Predicate, SourceType]),
                fail
            )
        ;   PredicateMode = baas
        ->  % Check if bash is allowed
            (   \+ predicate_denied_service(Predicate, powershell, executable(bash)),
                check_service(powershell, executable(bash), BashResult),
                BashResult \= deny(_)
            ->  Mode = baas
            ;   format('[Firewall Error] ~w requires baas mode but bash is denied~n', [Predicate]),
                fail
            )
        ;   PredicateMode = auto
        ->  % Use default derivation
            derive_powershell_mode(SourceType, Mode)
        )

    % Check predicate-level denied bash service
    ;   predicate_denied_service(Predicate, powershell, executable(bash))
    ->  % Bash denied for this predicate, must use pure
        (   supports_pure_powershell(SourceType)
        ->  Mode = pure,
            format('[Firewall] ~w: bash denied, using pure PowerShell~n', [Predicate])
        ;   format('[Firewall Error] ~w: bash denied but ~w has no pure implementation~n',
                   [Predicate, SourceType]),
            fail
        )

    % Fall back to default-level derivation
    ;   derive_powershell_mode(SourceType, Mode)
    ).

%% effective_powershell_mode(+Predicate, +SourceType, -Mode)
%  Get the effective PowerShell mode for a predicate.
%  This is the main entry point for compilers to determine mode.
%
%  Respects priority: predicate-level > default-level > auto-detection
effective_powershell_mode(Predicate, SourceType, Mode) :-
    % Try predicate-level derivation first
    (   derive_predicate_powershell_mode(Predicate, SourceType, DerivedMode)
    ->  Mode = DerivedMode
    % Fall back to default
    ;   derive_powershell_mode(SourceType, Mode)
    ).

%% Helper: supports_pure_powershell for predicates
%  Overload to check if a predicate supports pure PowerShell
%  Note: This version just assumes pure is supported for predicates
%  without a source type (rules/facts). The actual source type check
%  happens in powershell_compiler.pl.
supports_pure_powershell(Pred/Arity) :-
    % For predicate indicators, assume pure is supported
    % (the actual check happens in powershell_compiler with source_type)
    atom(Pred),
    integer(Arity),
    Arity >= 0.

%% ============================================
%% PREDICATE-LEVEL FIREWALL TESTS
%% ============================================

test_predicate_firewall :-
    format('~n=== Testing Predicate-Level Firewall (Phase 10) ===~n~n', []),

    % Test 1: Set predicate firewall mode
    format('[Test 1] Set predicate firewall mode~n', []),
    set_predicate_firewall_mode(test_pred/2, pure),
    (   predicate_firewall_mode(test_pred/2, pure)
    ->  format('  [PASS] Predicate mode set correctly~n', [])
    ;   format('  [FAIL] Predicate mode not set~n', [])
    ),

    % Test 2: Predicate-level service deny
    format('[Test 2] Predicate-level service deny~n', []),
    set_predicate_service_policy(secure_pred/3, powershell, executable(bash), deny),
    (   predicate_denied_service(secure_pred/3, powershell, executable(bash))
    ->  format('  [PASS] Service denial recorded~n', [])
    ;   format('  [FAIL] Service denial not recorded~n', [])
    ),

    % Test 3: Check predicate firewall with explicit mode
    format('[Test 3] Check predicate firewall with explicit mode~n', []),
    check_predicate_firewall(test_pred/2, powershell, Result1),
    (   Result1 = allow(pure)
    ->  format('  [PASS] Predicate firewall check returned pure~n', [])
    ;   format('  [FAIL] Expected allow(pure), got ~w~n', [Result1])
    ),

    % Test 4: Check predicate firewall with denied service
    format('[Test 4] Check predicate firewall with denied service~n', []),
    check_predicate_firewall(secure_pred/3, powershell, Result2),
    (   Result2 = deny(predicate_service_denied(_))
    ->  format('  [PASS] Service denial enforced~n', [])
    ;   format('  [FAIL] Expected deny, got ~w~n', [Result2])
    ),

    % Test 5: Clear predicate firewall rules
    format('[Test 5] Clear predicate firewall rules~n', []),
    clear_predicate_firewall_mode(test_pred/2),
    (   \+ predicate_firewall_mode(test_pred/2, _)
    ->  format('  [PASS] Predicate rules cleared~n', [])
    ;   format('  [FAIL] Predicate rules not cleared~n', [])
    ),

    % Test 6: Effective mode derivation
    format('[Test 6] Effective mode derivation~n', []),
    set_predicate_firewall_mode(csv_pred/2, pure),
    effective_powershell_mode(csv_pred/2, csv, Mode1),
    (   Mode1 = pure
    ->  format('  [PASS] Effective mode = pure~n', [])
    ;   format('  [FAIL] Expected pure, got ~w~n', [Mode1])
    ),

    % Cleanup
    clear_predicate_firewall_mode(csv_pred/2),
    clear_predicate_firewall_mode(secure_pred/3),

    format('~n=== Predicate-Level Firewall Tests Complete ===~n', []).

%% ============================================
%% INITIALIZATION
%% ============================================

% Default: permissive mode
:- initialization(
    (   \+ firewall_mode(_)
    ->  set_firewall_mode(permissive)
    ;   true
    ),
    now
).
