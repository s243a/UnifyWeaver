:- module(predicate_preprocessing, [
    declared_preprocess/3,          % +PredIndicator, -Mode, -Info
    declared_preprocess_metadata/4, % +PredIndicator, -Mode, -Info, -Metadata
    preprocess_mode/2,              % +Spec, -Mode
    preprocess_metadata/3           % +Spec, -Mode, -Metadata
]).

:- use_module(library(lists), [memberchk/2]).

%% predicate_preprocessing.pl
%%
%% Shared preprocessing declaration surface for targets that want to
%% consume source-level artifact/materialization intent without hardcoding
%% benchmark-local predicates.
%%
%% Supported declaration shapes (all via user:preprocess/2):
%%
%%   preprocess(edge/2, artifact).
%%   preprocess(edge/2, artifact([format(grouped_tsv_arg1)])).
%%   preprocess(edge/2, exact_hash_index([key([1]), values([2])])).
%%   preprocess(edge/2, adjacency_index([key([1]), values([2])])).
%%   preprocess(edge/2, relation_rows([format(edn_rows)])).
%%   preprocess(edge/2, inline_data([])).
%%
%% The normalised Mode is intentionally small for now:
%%   artifact | sidecar | inline
%%
%% Info preserves the original kind/options so target-specific code can
%% grow into richer providers later.

declared_preprocess(PredIndicator, Mode, preprocess_info(Kind, Options)) :-
    current_predicate(user:preprocess/2),
    preprocess_lookup(PredIndicator, Spec),
    normalize_preprocess_spec(Spec, Kind, Mode, Options),
    !.

declared_preprocess_metadata(PredIndicator, Mode, Info, Metadata) :-
    declared_preprocess(PredIndicator, Mode, Info),
    Info = preprocess_info(Kind, Options),
    metadata_for_preprocess(Kind, Options, Metadata).

preprocess_mode(Spec, Mode) :-
    normalize_preprocess_spec(Spec, _Kind, Mode, _Options).

preprocess_metadata(Spec, Mode, Metadata) :-
    normalize_preprocess_spec(Spec, Kind, Mode, Options),
    metadata_for_preprocess(Kind, Options, Metadata).

preprocess_lookup(Module:Pred/Arity, Spec) :-
    !,
    (   call(user:preprocess(Module:Pred/Arity, Spec))
    ;   call(user:preprocess(Pred/Arity, Spec))
    ).
preprocess_lookup(Pred/Arity, Spec) :-
    call(user:preprocess(Pred/Arity, Spec)).

normalize_preprocess_spec(artifact, artifact, artifact, []) :- !.
normalize_preprocess_spec(artifact(Options), artifact, artifact, Options) :- !.
normalize_preprocess_spec(exact_hash_index(Options), exact_hash_index, artifact, Options) :- !.
normalize_preprocess_spec(adjacency_index(Options), adjacency_index, artifact, Options) :- !.
normalize_preprocess_spec(grouped_tsv(Options), grouped_tsv, artifact, Options) :- !.
normalize_preprocess_spec(relation_rows, relation_rows, sidecar, []) :- !.
normalize_preprocess_spec(relation_rows(Options), relation_rows, sidecar, Options) :- !.
normalize_preprocess_spec(sidecar, sidecar, sidecar, []) :- !.
normalize_preprocess_spec(sidecar(Options), sidecar, sidecar, Options) :- !.
normalize_preprocess_spec(inline_data, inline_data, inline, []) :- !.
normalize_preprocess_spec(inline_data(Options), inline_data, inline, Options) :- !.
normalize_preprocess_spec(inline, inline, inline, []) :- !.
normalize_preprocess_spec(inline(Options), inline, inline, Options) :- !.
normalize_preprocess_spec(benchmark_mode(Mode), benchmark_mode, Mode, []) :-
    valid_mode(Mode),
    !.
normalize_preprocess_spec(benchmark_mode(Mode, Options), benchmark_mode, Mode, Options) :-
    valid_mode(Mode),
    !.

valid_mode(Mode) :-
    memberchk(Mode, [artifact, sidecar, inline]).

metadata_for_preprocess(Kind, Options,
                        preprocess_metadata{
                            kind: Kind,
                            format: Format,
                            access_contracts: AccessContracts,
                            options: Options
                        }) :-
    preprocess_format(Kind, Options, Format),
    preprocess_access_contracts(Kind, Options, AccessContracts).

preprocess_format(_Kind, Options, Format) :-
    memberchk(format(Format), Options),
    !.
preprocess_format(artifact, _Options, artifact).
preprocess_format(exact_hash_index, _Options, exact_hash_index).
preprocess_format(adjacency_index, _Options, adjacency_index).
preprocess_format(grouped_tsv, _Options, grouped_tsv).
preprocess_format(relation_rows, _Options, relation_rows).
preprocess_format(sidecar, _Options, sidecar).
preprocess_format(inline_data, _Options, inline_data).
preprocess_format(inline, _Options, inline).
preprocess_format(benchmark_mode, _Options, benchmark_mode).

preprocess_access_contracts(_Kind, Options, AccessContracts) :-
    memberchk(access(Declared), Options),
    !,
    sort(Declared, AccessContracts).
preprocess_access_contracts(exact_hash_index, Options, AccessContracts) :-
    !,
    preprocess_index_access_contracts(Options,
                                      [exact_key_lookup, scan],
                                      AccessContracts).
preprocess_access_contracts(adjacency_index, Options, AccessContracts) :-
    !,
    preprocess_index_access_contracts(Options,
                                      [adjacency_lookup, scan],
                                      AccessContracts).
preprocess_access_contracts(grouped_tsv, Options, AccessContracts) :-
    !,
    preprocess_index_access_contracts(Options,
                                      [grouped_lookup, scan],
                                      AccessContracts).
preprocess_access_contracts(relation_rows, _Options, [scan]).
preprocess_access_contracts(sidecar, _Options, [scan]).
preprocess_access_contracts(inline_data, _Options, [scan]).
preprocess_access_contracts(inline, _Options, [scan]).
preprocess_access_contracts(artifact, _Options, [scan]).
preprocess_access_contracts(benchmark_mode, _Options, [scan]).

preprocess_index_access_contracts(Options, Base, AccessContracts) :-
    findall(Contract,
            preprocess_option_access_contract(Options, Contract),
            OptionContracts0),
    append(Base, OptionContracts0, Contracts0),
    sort(Contracts0, AccessContracts).

preprocess_option_access_contract(Options, arg_position_lookup(Arg)) :-
    memberchk(key(Keys), Options),
    member(Arg, Keys).
preprocess_option_access_contract(Options, grouped_values_lookup(Args)) :-
    memberchk(values(Args), Options).
