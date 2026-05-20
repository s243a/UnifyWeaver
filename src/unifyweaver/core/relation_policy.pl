%% relation_policy.pl
%% Cross-target relation policy registry.
%%
%% Phase 1: parser + registry + lookup API. NO enforcement.
%% Per docs/design/RELATION_POLICY_DECLARATIONS.md.
%%
%% User code declares policies via:
%%
%%   :- use_module(library(unifyweaver/core/relation_policy)).
%%   :- relation_policy(edge/2,
%%         [ key([arg(1), arg(2)]),
%%           order(natural),
%%           unique(true),
%%           on_duplicate(throw),
%%           determinism(semidet)
%%         ]).
%%
%% Backends (codegen) query via:
%%
%%   get_relation_policy(edge/2, unique, Value)        % semidet
%%   get_relation_policy(edge/2, unique, V, false)     % det, with default
%%   get_effective_policy(edge/2, SourceOpts, K, V)    % merge override
%%
%% No backend reads the registry yet -- that's Phase 2, per-backend.

:- module(relation_policy, [
    relation_policy/2,
    get_relation_policy/3,           % +PredArity, +Key, -Value
    get_relation_policy/4,           % +PredArity, +Key, -Value, +Default
    get_effective_policy/4,          % +PredArity, +Override, +Key, -Value
    current_relation_policy/3,       % ?PredArity, ?Key, ?Value
    clear_relation_policies/0,
    relation_policy_default/2,       % +Key, -Default
    relation_policy_option_keys/1    % -ListOfKeys
]).

:- dynamic stored_relation_policy/3.   % PredArity, Key, Value

%% relation_policy(+PredArity, +Options) is det.
%
% Declare a policy for Pred/Arity. Validates the indicator and
% every option key/value pair; stores them in the registry. Calling
% twice for the same Pred/Arity overwrites earlier declarations for
% the same keys (latest-wins per-key, not per-declaration).
relation_policy(PredArity, Options) :-
    validate_pred_arity(PredArity),
    must_be(list, Options),
    forall(member(Opt, Options), validate_option(Opt)),
    forall(member(Opt, Options), (
        Opt =.. [Key, Value],
        retractall(stored_relation_policy(PredArity, Key, _)),
        assertz(stored_relation_policy(PredArity, Key, Value))
    )).

%% get_relation_policy(+PredArity, +Key, -Value) is semidet.
%
% Look up the declared value for Key. Fails if no declaration.
get_relation_policy(PredArity, Key, Value) :-
    stored_relation_policy(PredArity, Key, Value).

%% get_relation_policy(+PredArity, +Key, -Value, +Default) is det.
%
% Same as /3 but returns Default when no declaration exists.
get_relation_policy(PredArity, Key, Value, _) :-
    stored_relation_policy(PredArity, Key, Value), !.
get_relation_policy(_, _, Value, Default) :-
    Value = Default.

%% get_effective_policy(+PredArity, +OverrideOpts, +Key, -Value) is det.
%
% Source-level override merge: if Key is in OverrideOpts, use that
% value; else fall back to the registry; else fall back to the
% built-in default. Used by backends consuming both predicate-level
% declarations and per-source spec options like
%   source(edge/2, lmdb('graph.mdb', [on_duplicate(warn)]))
get_effective_policy(_, OverrideOpts, Key, Value) :-
    member(Opt, OverrideOpts),
    Opt =.. [Key, Value], !.
get_effective_policy(PredArity, _, Key, Value) :-
    stored_relation_policy(PredArity, Key, Value), !.
get_effective_policy(_, _, Key, Value) :-
    relation_policy_default(Key, Value).

%% current_relation_policy(?PredArity, ?Key, ?Value) is nondet.
%
% Enumerate all declarations. Introspection helper.
current_relation_policy(PredArity, Key, Value) :-
    stored_relation_policy(PredArity, Key, Value).

%% clear_relation_policies is det.
%
% Wipe the registry. Test helper.
clear_relation_policies :-
    retractall(stored_relation_policy(_, _, _)).

%% relation_policy_default(+Key, -Default)
%
% The implicit defaults that apply when neither a declaration nor a
% source-level override provides a value. Matches today's implicit
% behaviour so existing code without declarations is unaffected.
% Per the doc's "Defaults summary" section.
relation_policy_default(key,           all).
relation_policy_default(order,         natural).
relation_policy_default(unique,        false).
relation_policy_default(on_duplicate,  keep_all).
relation_policy_default(determinism,   nondet).
relation_policy_default(cardinality,   unknown).

%% relation_policy_option_keys(-Keys) is det.
%
% The set of recognised option keys. Used by validation.
relation_policy_option_keys([key, order, unique, on_duplicate,
                             determinism, cardinality]).

% ----------------------------------------------------------------
% Validation
% ----------------------------------------------------------------

validate_pred_arity(Functor/Arity) :- !,
    must_be(atom, Functor),
    must_be(nonneg, Arity).
validate_pred_arity(Other) :-
    type_error(predicate_indicator, Other).

validate_option(Opt) :-
    Opt =.. [Key | Args], !,
    ( Args = [Value]
    -> validate_option_value(Key, Value)
    ;  domain_error(relation_policy_option, Opt)
    ).
validate_option(Opt) :-
    type_error(relation_policy_option, Opt).

validate_option_value(key, V)          :- !, validate_key_spec(V).
validate_option_value(order, V)        :- !, validate_order_spec(V).
validate_option_value(unique, V)       :- !, must_be(boolean, V).
validate_option_value(on_duplicate, V) :- !, validate_dup_policy(V).
validate_option_value(determinism, V)  :- !, validate_determinism(V).
validate_option_value(cardinality, V)  :- !, validate_cardinality(V).
validate_option_value(Key, _) :-
    domain_error(relation_policy_key, Key).

validate_key_spec(all) :- !.
validate_key_spec(arg(N)) :- !, must_be(positive_integer, N).
validate_key_spec([])    :- !.
validate_key_spec([arg(N) | Rest]) :- !,
    must_be(positive_integer, N),
    validate_key_spec(Rest).
validate_key_spec(Other) :-
    domain_error(relation_policy_key_spec, Other).

validate_order_spec(natural)   :- !.
validate_order_spec(insertion) :- !.
validate_order_spec([])        :- !.
validate_order_spec([H | T])   :- !,
    validate_order_term(H),
    validate_order_spec(T).
validate_order_spec(Other) :-
    domain_error(relation_policy_order_spec, Other).

validate_order_term(arg(N))      :- !, must_be(positive_integer, N).
validate_order_term(asc(arg(N))) :- !, must_be(positive_integer, N).
validate_order_term(desc(arg(N))):- !, must_be(positive_integer, N).
validate_order_term(Other) :-
    domain_error(relation_policy_order_term, Other).

validate_dup_policy(throw)      :- !.
validate_dup_policy(warn)       :- !.
validate_dup_policy(overwrite)  :- !.
validate_dup_policy(first_wins) :- !.
validate_dup_policy(keep_all)   :- !.
validate_dup_policy(fallback(P)) :- !, validate_dup_policy(P).
validate_dup_policy(Other) :-
    domain_error(relation_policy_on_duplicate, Other).

validate_determinism(det)     :- !.
validate_determinism(semidet) :- !.
validate_determinism(nondet)  :- !.
validate_determinism(multi)   :- !.
validate_determinism(Other) :-
    domain_error(relation_policy_determinism, Other).

validate_cardinality(unknown) :- !.
validate_cardinality(small)   :- !.
validate_cardinality(medium)  :- !.
validate_cardinality(large)   :- !.
validate_cardinality(N) :- integer(N), N >= 0, !.
validate_cardinality(Other) :-
    domain_error(relation_policy_cardinality, Other).
