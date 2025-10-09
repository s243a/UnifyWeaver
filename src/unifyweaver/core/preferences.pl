:- encoding(utf8).

:- module(preferences, [
    rule_preferences/2,
    preferences_default/1,
    get_final_options/3
]).

:- use_module(library(dicts)).

:- dynamic rule_preferences/2.
:- dynamic preferences_default/1.

%% get_final_options(+Pred/Arity, +RuntimeOptions, -FinalOptions) is det.
%  Merge precedence: Runtime > Rule > Default.
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