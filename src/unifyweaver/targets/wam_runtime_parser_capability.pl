:- module(wam_runtime_parser_capability, [
    wam_target_runtime_parser/3,
    parser_dependent_builtin/1,
    target_supports_runtime_parser_mode/2
]).

:- use_module(library(error)).
:- use_module(library(option)).

%% wam_target_runtime_parser(+Target, +Options, -Mode)
%
% Resolve the runtime Prolog-term parser mode for a WAM target.
% Mode is one of:
%   - none
%   - native(Entry)
%   - compiled(prolog_term_parser)
%
% Options:
%   - runtime_parser(auto)     target default
%   - runtime_parser(off)      no source-term parser
%   - runtime_parser(native)   require a native target parser
%   - runtime_parser(compiled) require the portable compiled parser
wam_target_runtime_parser(Target, Options, Mode) :-
    must_be(atom, Target),
    must_be(list, Options),
    normalize_runtime_parser_target(Target, Canonical),
    option(runtime_parser(Request), Options, auto),
    resolve_runtime_parser_request(Request, Canonical, Mode).

%% parser_dependent_builtin(?Name/Arity)
%
% Builtins/features that need source text parsed into runtime terms.
parser_dependent_builtin(read/2).
parser_dependent_builtin(read_term_from_atom/2).
parser_dependent_builtin(read_term_from_atom/3).
parser_dependent_builtin(term_to_atom/2).

%% target_supports_runtime_parser_mode(+Target, ?Mode)
%
% Capability facts used by the resolver. Keep this conservative:
% only advertise compiled parser support for targets with compile and
% runtime proof tests.
target_supports_runtime_parser_mode(Target, Mode) :-
    must_be(atom, Target),
    normalize_runtime_parser_target(Target, Canonical),
    target_runtime_parser_mode_(Canonical, Mode).

resolve_runtime_parser_request(auto, Target, Mode) :-
    !,
    (   target_runtime_parser_default(Target, Mode)
    ->  true
    ;   Mode = none
    ).
resolve_runtime_parser_request(off, _Target, none) :- !.
resolve_runtime_parser_request(native, Target, Mode) :-
    !,
    (   target_supports_runtime_parser_mode(Target, Mode),
        Mode = native(_)
    ->  true
    ;   domain_error(runtime_parser_mode(Target), native)
    ).
resolve_runtime_parser_request(compiled, Target, Mode) :-
    !,
    (   target_supports_runtime_parser_mode(Target,
                                            compiled(prolog_term_parser))
    ->  Mode = compiled(prolog_term_parser)
    ;   domain_error(runtime_parser_mode(Target), compiled)
    ).
resolve_runtime_parser_request(Request, _Target, _Mode) :-
    domain_error(runtime_parser_request, Request).

target_runtime_parser_default(wam_r, native(parse_term)).

target_runtime_parser_mode_(wam_r, native(parse_term)).
target_runtime_parser_mode_(wam_r, compiled(prolog_term_parser)).

normalize_runtime_parser_target(r, wam_r) :- !.
normalize_runtime_parser_target(wam_r, wam_r) :- !.
normalize_runtime_parser_target(Target, Target).
