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

    % Policy templates
    load_firewall_policy/1
]).

:- use_module(library(lists)).

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
supports_pure_powershell(csv).
supports_pure_powershell(json).
supports_pure_powershell(http).

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
    % Deny all network services
    assertz(denied_service(_, network_access(_))),
    assertz(denied_service(_, executable(curl))),
    format('[Firewall] Network access blocked~n', []),
    !.

load_firewall_policy(PolicyName) :-
    format('[Firewall Error] Unknown policy: ~w~n', [PolicyName]),
    fail.

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
