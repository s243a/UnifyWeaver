:- module(wam_runtime_parser_capability, [
    wam_target_runtime_parser/3,
    parser_dependent_builtin/1,
    parser_dependent_goal/2,
    parser_dependent_body_goal/2,
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
parser_dependent_builtin(read/1).
parser_dependent_builtin(read/2).
parser_dependent_builtin(read_term/1).
parser_dependent_builtin(read_term_from_atom/2).
parser_dependent_builtin(read_term_from_atom/3).
parser_dependent_builtin(term_to_atom/2).

%% parser_dependent_goal(+Goal, -Builtin)
%
% True when Goal is a statically visible call shape that needs a
% runtime source-term parser. `term_to_atom/2` is mode-sensitive:
% forward rendering does not parse, but a visible nonvar atom/text
% argument means reverse parsing is possible and requires a parser.
parser_dependent_goal(Goal0, Builtin) :-
    nonvar(Goal0),
    strip_module_qualifier(Goal0, Goal),
    parser_dependent_goal_(Goal, Builtin).

parser_dependent_goal_(term_to_atom(_Term, AtomText), term_to_atom/2) :-
    !,
    nonvar(AtomText).
parser_dependent_goal_(Goal, Builtin) :-
    callable(Goal),
    functor(Goal, Name, Arity),
    Builtin = Name/Arity,
    parser_dependent_builtin(Builtin),
    Builtin \= term_to_atom/2.

%% parser_dependent_body_goal(+BodyGoal, -Builtin)
%
% True when BodyGoal contains a statically visible parser-dependent
% goal. Walks the common source-level control forms target writers see
% before WAM lowering.
parser_dependent_body_goal(Body0, Builtin) :-
    nonvar(Body0),
    strip_module_qualifier(Body0, Body),
    (   parser_dependent_goal(Body, Builtin)
    ->  true
    ;   parser_body_child(Body, Child),
        parser_dependent_body_goal(Child, Builtin)
    ).

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
% C++ has a hand-written canonical-form parser in the runtime
% (LmdbFactSource lives next to it). It powers atom_to_term/3,
% term_to_atom/2's reverse mode, and read_term/1. It does NOT
% currently support operator notation -- 1+2 must be written
% as +(1, 2). Compiling the portable parser in would lift that
% restriction; we register both modes here and leave native as
% the default since it ships today.
target_runtime_parser_default(wam_cpp, native(parse_term)).
% F# has no native runtime parser today; the F# target compiles the
% portable parser through WAM and that path now executes correctly
% end-to-end (42/42 inputs in test_wam_fsharp_parser_smoke.pl, including
% prefix-directive '`:- p`', list patterns, and clauses with bodies).
% A series of runtime fixes -- PR #2407 (B0 cut-barrier protocol),
% PR #2408 (VList/Str list-encoding equivalence), PR #2415 (member/2
% via MemberRetry), PR #2419 (==/2 list-encoding, findall result-reg
% seeding), PR #2423 (dispatchCall WsCP reset), PR #2422 (lowered
% quoted-atom rendering) -- closed the gaps that the earlier version
% of this comment was warning about (Proceed-stubbed instructions).
%
% Default is kept at `none` so existing F# projects don't silently
% pull in the parser library; `runtime_parser(compiled)` is opt-in.
% See docs/WAM_RUNTIME_PARSER_STATUS.md for the cross-target picture
% and why we don't flip the default.
target_runtime_parser_default(wam_fsharp, none).
target_runtime_parser_default(wam_haskell, none).

target_runtime_parser_mode_(wam_r, native(parse_term)).
target_runtime_parser_mode_(wam_r, compiled(prolog_term_parser)).
target_runtime_parser_mode_(wam_cpp, native(parse_term)).
target_runtime_parser_mode_(wam_cpp, compiled(prolog_term_parser)).
target_runtime_parser_mode_(wam_python, compiled(prolog_term_parser)).
target_runtime_parser_mode_(wam_fsharp, compiled(prolog_term_parser)).
target_runtime_parser_mode_(wam_haskell, compiled(prolog_term_parser)).

normalize_runtime_parser_target(r, wam_r) :- !.
normalize_runtime_parser_target(wam_r, wam_r) :- !.
normalize_runtime_parser_target(cpp, wam_cpp) :- !.
normalize_runtime_parser_target(wam_cpp, wam_cpp) :- !.
normalize_runtime_parser_target(fsharp, wam_fsharp) :- !.
normalize_runtime_parser_target(wam_fsharp, wam_fsharp) :- !.
normalize_runtime_parser_target(haskell, wam_haskell) :- !.
normalize_runtime_parser_target(wam_haskell, wam_haskell) :- !.
normalize_runtime_parser_target(Target, Target).

strip_module_qualifier(Module:Goal, Stripped) :-
    atom(Module),
    nonvar(Goal),
    !,
    strip_module_qualifier(Goal, Stripped).
strip_module_qualifier(Goal, Goal).

parser_body_child((A, _), A).
parser_body_child((_, B), B).
parser_body_child((A ; _), A).
parser_body_child((_ ; B), B).
parser_body_child((A -> _), A).
parser_body_child((_ -> B), B).
parser_body_child((*->(A, _)), A).
parser_body_child((*->(_, B)), B).
parser_body_child(\+(A), A).
parser_body_child(not(A), A).
parser_body_child(once(A), A).
parser_body_child(call(A), A).
