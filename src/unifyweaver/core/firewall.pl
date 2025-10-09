:- encoding(utf8).

:- module(firewall, [
    rule_firewall/2,
    firewall_default/1,
    validate_against_firewall/3
]).

:- use_module(library(lists)).

:- dynamic rule_firewall/2.
:- dynamic firewall_default/1.

%% validate_against_firewall(+Target, +FinalOptions, +Firewall) is semidet.
%  Target is the primary execution backend (e.g., bash).
%  FinalOptions may request services; Firewall is the merged policy for the predicate.
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