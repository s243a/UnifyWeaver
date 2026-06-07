% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% iso_errors.pl - Shared ISO error config, mode resolution, and audit
% helpers for WAM targets.
%
% Cross-target ISO contract:
%   docs/design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md
%   docs/design/WAM_CPP_ISO_ERRORS_SPECIFICATION.md
%
% Background
% ----------
% C++, Elixir, Python, and F# all adopted near-identical Prolog
% plumbing for ISO error config loading, per-predicate mode
% resolution, and audit. Each target was carrying its own ~340-line
% copy of the same code. This module extracts the parts that are
% byte-identical (mod whitespace and comments) so target files can
% import them.
%
% What lives here vs. per-target
% ------------------------------
% Shared (this module):
%   - Config schema parsing: iso_errors_load_config/2,
%     iso_errors_extract_terms/5, iso_errors_read_terms/2.
%   - Option resolution: iso_errors_resolve_options/2 and its
%     iso_errors_inline_* / iso_errors_merge_overrides helpers.
%   - PI matching: iso_errors_valid_pi/1, iso_errors_pi_matches/2,
%     iso_errors_shadowed/2.
%   - Mode resolution: iso_errors_mode_for/3.
%   - Multi-module warning: iso_errors_warn_multi_module/2,
%     iso_errors_check_override_scope/2, iso_errors_pi_module/4.
%   - Item-level rewrite: iso_errors_rewrite/4, iso_errors_rewrite_item/3.
%   - Audit walker helpers shared across targets:
%     iso_errors_audit_normalise_pi/2, iso_errors_audit_walk/5,
%     iso_errors_audit_classify/5, iso_errors_resolve_default/3,
%     iso_errors_other_mode/2, iso_errors_key_has_suffix/2.
%
% Per-target (stays in wam_<target>_target.pl):
%   - iso_errors_default_to_iso/2 and iso_errors_default_to_lax/2
%     facts -- these are declared :- multifile here so each target
%     adds its own table entries.  A target should NOT add an entry
%     before its runtime has a matching builtin branch; otherwise the
%     rewrite silently routes calls to dead keys.
%   - iso_errors_rewrite_text/4 and friends: Python/F# and Elixir use
%     slightly different WAM-text parsing strategies; reconciliation
%     is a separate concern from extracting the shared parts.
%   - The audit predicate itself (wam_<target>_iso_audit/3) and its
%     pretty-printer -- they consume the shared walker helpers.

:- module(iso_errors, [
    iso_errors_resolve_options/2,        % +Options, -Config
    iso_errors_load_config/2,            % +File, -Config
    iso_errors_mode_for/3,               % +Config, +PI, -Mode
    iso_errors_warn_multi_module/2,      % +Config, +Predicates
    iso_errors_rewrite/4,                % +Config, +PI, +Items0, -Items
    iso_errors_rewrite_item/3,           % +Mode, +Item0, -Item
    iso_errors_valid_pi/1,               % +PI
    iso_errors_pi_matches/2,             % +PatternPI, +TargetPI
    iso_errors_audit_normalise_pi/2,     % +PI, -NormPI
    iso_errors_audit_walk/5,             % +Items, +PC, +Mode, +Acc, -Sites
    iso_errors_resolve_default/3,        % +Key, +Mode, -ResolvedKey
    iso_errors_other_mode/2,             % +Mode, -OtherMode
    iso_errors_key_has_suffix/2,         % +Key, +Suffix
    % Multifile tables -- each target asserts its own entries.
    iso_errors_default_to_iso/2,         % +DefaultKey, -IsoKey
    iso_errors_default_to_lax/2          % +DefaultKey, -LaxKey
]).

:- use_module(library(lists)).
:- use_module(library(option)).

% Each WAM target asserts its own entries; this module ships none.
% Targets typically include is/2, the six arithmetic comparisons, and
% succ/2.  Python additionally maps read_term_from_atom/2,3, read/1,2,
% and read_term/1 because it has the runtime-parser substrate; F# and
% Elixir do not (yet).
:- multifile iso_errors_default_to_iso/2.
:- multifile iso_errors_default_to_lax/2.
:- dynamic   iso_errors_default_to_iso/2.
:- dynamic   iso_errors_default_to_lax/2.

% ============================================================================
% Option resolution
% ============================================================================

%% iso_errors_resolve_options(+Options, -Config)
%  Merges optional file config with inline options into
%  iso_config(Default, Overrides).  Inline options override file
%  entries for the same PI.
iso_errors_resolve_options(Options, iso_config(Default, Overrides)) :-
    (   option(iso_errors_config(File), Options)
    ->  iso_errors_load_config(File, iso_config(FileDefault, FileOv))
    ;   FileDefault = false, FileOv = []
    ),
    iso_errors_inline_default(Options, FileDefault, Default),
    iso_errors_inline_overrides(Options, InlineOv),
    iso_errors_merge_overrides(FileOv, InlineOv, Overrides).

iso_errors_inline_default(Options, FileDefault, Default) :-
    (   member(iso_errors(M), Options),
        (M == true ; M == false)
    ->  Default = M
    ;   Default = FileDefault
    ).

iso_errors_inline_overrides(Options, InlineOv) :-
    findall(PI-Mode,
        ( member(iso_errors(PI, Mode), Options),
          (Mode == true ; Mode == false),
          iso_errors_valid_pi(PI)
        ),
        InlineOv).

iso_errors_valid_pi(Name/Arity) :- atom(Name), integer(Arity), Arity >= 0, !.
iso_errors_valid_pi(Module:Name/Arity) :-
    atom(Module), atom(Name), integer(Arity), Arity >= 0.

iso_errors_merge_overrides(FileOv, InlineOv, Merged) :-
    exclude(iso_errors_shadowed(InlineOv), FileOv, Kept),
    append(Kept, InlineOv, Merged).

iso_errors_shadowed(InlineOv, PI-_) :-
    member(InlinePI-_, InlineOv),
    iso_errors_pi_matches(InlinePI, PI), !.

% Bare Name/Arity matches Module:Name/Arity in any module, and vice
% versa.  Same-shape PIs match by unification.
iso_errors_pi_matches(PI, PI) :- !.
iso_errors_pi_matches(Name/Arity, _:Name/Arity) :-
    atom(Name), integer(Arity), !.
iso_errors_pi_matches(_:Name/Arity, Name/Arity) :-
    atom(Name), integer(Arity), !.

% ============================================================================
% Config-file loader
% ============================================================================

%% iso_errors_load_config(+File, -Config)
%  Reads iso_errors_default/1 and iso_errors_override/2 facts.
%  Unknown facts and I/O failures are ignored, yielding
%  iso_config(false, []).
iso_errors_load_config(File, iso_config(Default, Overrides)) :-
    catch(
        setup_call_cleanup(
            open(File, read, Stream),
            iso_errors_read_terms(Stream, RawTerms),
            close(Stream)),
        _,
        RawTerms = []),
    iso_errors_extract_terms(RawTerms, false, [], Default, RevOv),
    reverse(RevOv, Overrides).

iso_errors_read_terms(Stream, Terms) :-
    read_term(Stream, T, []),
    (   T == end_of_file
    ->  Terms = []
    ;   Terms = [T|Rest],
        iso_errors_read_terms(Stream, Rest)
    ).

iso_errors_extract_terms([], D, Ov, D, Ov).
iso_errors_extract_terms([T|Rest], D0, Ov0, D, Ov) :-
    (   T = iso_errors_default(NewD), (NewD == true ; NewD == false)
    ->  iso_errors_extract_terms(Rest, NewD, Ov0, D, Ov)
    ;   T = iso_errors_override(PI, Mode),
        (Mode == true ; Mode == false),
        iso_errors_valid_pi(PI)
    ->  iso_errors_extract_terms(Rest, D0, [PI-Mode|Ov0], D, Ov)
    ;   iso_errors_extract_terms(Rest, D0, Ov0, D, Ov)
    ).

% ============================================================================
% Mode resolution
% ============================================================================

%% iso_errors_mode_for(+Config, +PI, -Mode)
iso_errors_mode_for(iso_config(Default, Overrides), PI, Mode) :-
    (   member(OvPI-OvMode, Overrides),
        iso_errors_pi_matches(OvPI, PI)
    ->  Mode = OvMode
    ;   Mode = Default
    ).

% ============================================================================
% Multi-module bare-PI warning
% ============================================================================

%% iso_errors_warn_multi_module(+Config, +Predicates)
%  Warns when a bare override matches predicates from multiple modules.
iso_errors_warn_multi_module(iso_config(_, Overrides), Predicates) :-
    forall(member(OvPI-_, Overrides),
           iso_errors_check_override_scope(OvPI, Predicates)).

iso_errors_check_override_scope(Name/Arity, Predicates) :-
    atom(Name), integer(Arity), !,
    findall(M, ( member(P, Predicates),
                 iso_errors_pi_module(P, Name, Arity, M)
               ), Modules),
    list_to_set(Modules, Unique),
    (   Unique = [_, _ | _]
    ->  length(Unique, N),
        format(user_error,
               'Warning: iso_errors_override(~w/~w, _) matches ~w predicates~n         in different modules (~w).~n         Qualify with `mod:~w/~w` for module-scoped overrides.~n',
               [Name, Arity, N, Unique, Name, Arity])
    ;   true
    ).
iso_errors_check_override_scope(_, _).

iso_errors_pi_module(Module:Name/Arity, Name, Arity, Module) :- !.
iso_errors_pi_module(Name/Arity, Name, Arity, user).
iso_errors_pi_module(Pred/Arity-_, Name, Arity, user) :-
    atom(Pred), Pred = Name, !.

% ============================================================================
% Item-level rewrite
% ============================================================================

%% iso_errors_rewrite(+Config, +PI, +Items0, -Items)
%  Item-level rewrite API.  Some targets prefer text-level rewrite
%  (Python/F#/Elixir) and keep their own iso_errors_rewrite_text/4
%  alongside this; both are equivalent in semantics.
iso_errors_rewrite(Config, PI, Items0, Items) :-
    iso_errors_mode_for(Config, PI, Mode),
    maplist(iso_errors_rewrite_item(Mode), Items0, Items).

iso_errors_rewrite_item(true, builtin_call(Key, N), builtin_call(IsoKey, N)) :-
    iso_errors_default_to_iso(Key, IsoKey), !.
iso_errors_rewrite_item(true, put_structure(Key, Reg), put_structure(IsoKey, Reg)) :-
    iso_errors_default_to_iso(Key, IsoKey), !.
iso_errors_rewrite_item(true, call(Key, N), call(IsoKey, N)) :-
    iso_errors_default_to_iso(Key, IsoKey), !.
iso_errors_rewrite_item(true, execute(Key), execute(IsoKey)) :-
    iso_errors_default_to_iso(Key, IsoKey), !.
iso_errors_rewrite_item(false, builtin_call(Key, N), builtin_call(LaxKey, N)) :-
    iso_errors_default_to_lax(Key, LaxKey), !.
iso_errors_rewrite_item(false, put_structure(Key, Reg), put_structure(LaxKey, Reg)) :-
    iso_errors_default_to_lax(Key, LaxKey), !.
iso_errors_rewrite_item(false, call(Key, N), call(LaxKey, N)) :-
    iso_errors_default_to_lax(Key, LaxKey), !.
iso_errors_rewrite_item(false, execute(Key), execute(LaxKey)) :-
    iso_errors_default_to_lax(Key, LaxKey), !.
iso_errors_rewrite_item(_, Item, Item).

% ============================================================================
% Audit walker helpers
% ============================================================================
%
% These helpers do the per-call-site bookkeeping shared by every
% target's wam_<target>_iso_audit/3.  The target predicate is
% responsible for compiling the predicate to WAM text, parsing the
% lines into items the walker accepts, and rendering the result;
% the walking + classification + flip-detection happens here.

iso_errors_audit_normalise_pi(Pred/Arity-_, Pred/Arity) :- !.
iso_errors_audit_normalise_pi(Module:Pred/Arity, Module:Pred/Arity) :- !.
iso_errors_audit_normalise_pi(PI, PI).

iso_errors_audit_walk([], _, _, Acc, Acc).
iso_errors_audit_walk([label|Rest], PC, Mode, Acc, Out) :- !,
    iso_errors_audit_walk(Rest, PC, Mode, Acc, Out).
iso_errors_audit_walk([builtin_call(Key, _)|Rest], PC, Mode, Acc, Out) :- !,
    iso_errors_audit_classify(Key, Mode, Source, Resolved, Flip),
    PC1 is PC + 1,
    iso_errors_audit_walk(Rest, PC1, Mode, [site(PC, Key, Resolved, Source, Flip)|Acc], Out).
iso_errors_audit_walk([_|Rest], PC, Mode, Acc, Out) :-
    PC1 is PC + 1,
    iso_errors_audit_walk(Rest, PC1, Mode, Acc, Out).

iso_errors_audit_classify(Key, _Mode, explicit_iso, Key, false) :-
    iso_errors_key_has_suffix(Key, "_iso"), !.
iso_errors_audit_classify(Key, _Mode, explicit_lax, Key, false) :-
    iso_errors_key_has_suffix(Key, "_lax"), !.
iso_errors_audit_classify(Key, Mode, default, Resolved, Flip) :-
    iso_errors_resolve_default(Key, Mode, Resolved),
    iso_errors_other_mode(Mode, OtherMode),
    iso_errors_resolve_default(Key, OtherMode, OtherResolved),
    ( Resolved == OtherResolved -> Flip = false ; Flip = true ).

iso_errors_resolve_default(Key, true, IsoKey) :-
    iso_errors_default_to_iso(Key, IsoKey), !.
iso_errors_resolve_default(Key, false, LaxKey) :-
    iso_errors_default_to_lax(Key, LaxKey), !.
iso_errors_resolve_default(Key, _, Key).

iso_errors_other_mode(true, false).
iso_errors_other_mode(false, true).

iso_errors_key_has_suffix(Key, Suffix) :-
    ( atom(Key) -> atom_string(Key, KS) ; KS = Key ),
    split_string(KS, "/", "", [Name | _]),
    string_concat(_, Suffix, Name).
