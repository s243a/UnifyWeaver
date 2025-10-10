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
    validate_against_firewall/3
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
    ).