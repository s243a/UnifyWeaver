% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
%% preferences.pl - Control Plane Preference System
%
% This module manages layered configuration preferences that guide the compiler
% on which implementation to choose from the options permitted by the firewall.
% Preferences are flexible and can be changed freely without security implications.
%
% @author John William Creighton (@s243a)
% @license MIT OR Apache-2.0

:- module(preferences, [
    rule_preferences/2,
    preferences_default/1,
    get_final_options/3
]).

:- use_module(library(dicts)).

%% rule_preferences(?PredicateIndicator, ?PreferenceTerms) is nondet.
%
% Declares predicate-specific preference settings. These override global defaults
% and are themselves overridden by runtime options.
%
% @arg PredicateIndicator The predicate in Functor/Arity form (e.g., ancestor/2)
% @arg PreferenceTerms List of preference terms that guide implementation choices
%
% Preference terms include:
% - prefer([backend1, service1, ...]) - Desired order of implementation choice
% - fallback_order([backend2, service2, ...]) - Order to try if preferred fails
% - optimization(speed|memory|balance) - Hint for code generation strategy
% - service_mode(embedded|remote) - Whether to prefer local or remote services
%
% @example Prefer bash with embedded SQL
%   :- assertz(preferences:rule_preferences(db_query/2, [prefer([bash]), service_mode(embedded)])).
%
% @example Optimize for speed
%   :- assertz(preferences:rule_preferences(fast_pred/2, [optimization(speed)])).
:- dynamic rule_preferences/2.

%% preferences_default(?PreferenceTerms) is nondet.
%
% Declares global default preference settings. These apply to all predicates that
% don't have specific rule_preferences/2 declarations.
%
% @arg PreferenceTerms List of preference terms (same format as rule_preferences/2)
%
% @example Set global default to prefer bash and optimize for balance
%   :- assertz(preferences:preferences_default([prefer([bash]), optimization(balance)])).
:- dynamic preferences_default/1.

%% get_final_options(+PredicateIndicator, +RuntimeOptions, -FinalOptions) is det.
%
% Merges preferences from all configuration layers into a final options list.
% The merge follows a strict precedence hierarchy where more specific layers
% override more general ones.
%
% Merge precedence (highest to lowest):
% 1. RuntimeOptions - Options passed directly to compile_recursive/3
% 2. Rule-specific - Declared via rule_preferences/2
% 3. Global defaults - Declared via preferences_default/1
%
% The merging is done using dict-based functor/arity keying, which means that
% if multiple layers define the same option (e.g., optimization/1), the higher
% precedence layer wins completely (no partial merging of option arguments).
%
% @arg PredicateIndicator The predicate in Functor/Arity form (e.g., ancestor/2)
% @arg RuntimeOptions Options passed at compilation time (highest precedence)
% @arg FinalOptions The merged result combining all layers
%
% @example Get options with runtime override
%   ?- assertz(preferences:preferences_default([optimization(balance)])),
%      assertz(preferences:rule_preferences(foo/2, [optimization(speed)])),
%      get_final_options(foo/2, [optimization(memory)], Final).
%   Final = [optimization(memory)].  % Runtime wins
%
% @example Get options with only defaults
%   ?- assertz(preferences:preferences_default([prefer([bash])])),
%      get_final_options(bar/2, [], Final).
%   Final = [prefer([bash])].
get_final_options(PredArity, Runtime, Final) :-
    % Defaults
    (   preferences_default(Defaults) -> true ; Defaults = [] ),
    % Rule-specific
    (   rule_preferences(PredArity, Rule) -> true ; Rule = [] ),
    % Merge with precedence
    merge_opts(Defaults, Rule, M1),
    merge_opts(M1, Runtime, Final).

% Merge helper: right-hand list overrides by functor/arity key.
merge_opts(Base, Override, Merged) :-
    index_by_key(Base, BaseI),
    index_by_key(Override, OverI),
    dict_pairs(OverI, _, OverPairs),
    foldl(put_pair, OverPairs, BaseI, OutI),
    dict_pairs(OutI, _, Pairs),
    pairs_values(Pairs, Merged).

index_by_key(Options, Dict) :-
    maplist(to_kv, Options, Pairs),
    dict_create(Dict, opts, Pairs).

to_kv(Opt, Key-Opt) :-
    functor(Opt, F, A),
    format(atom(Key), '~w/~w', [F, A]).

put_pair(K-V, D0, D) :- put_dict(K, D0, V, D).