/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Target Mapping - Maps predicates to compilation targets and locations
 *
 * This module tracks which target should compile each predicate, and
 * where that predicate should execute (location) and how it connects
 * to other predicates (transport).
 */

:- module(target_mapping, [
    % Target declarations
    declare_target/2,           % declare_target(Pred/Arity, Target)
    declare_target/3,           % declare_target(Pred/Arity, Target, Options)
    undeclare_target/1,         % undeclare_target(Pred/Arity)

    % Location declarations
    declare_location/2,         % declare_location(Pred/Arity, Options)
    undeclare_location/1,       % undeclare_location(Pred/Arity)

    % Connection declarations
    declare_connection/3,       % declare_connection(Pred1/Arity1, Pred2/Arity2, Options)
    undeclare_connection/2,     % undeclare_connection(Pred1/Arity1, Pred2/Arity2)

    % Queries
    predicate_target/2,         % predicate_target(Pred/Arity, Target)
    predicate_target_options/3, % predicate_target_options(Pred/Arity, Target, Options)
    predicate_location/2,       % predicate_location(Pred/Arity, Location)
    connection_transport/3,     % connection_transport(Pred1, Pred2, Transport)
    connection_options/3,       % connection_options(Pred1, Pred2, Options)

    % Resolution (with defaults)
    resolve_location/2,         % resolve_location(Pred/Arity, Location)
    resolve_transport/3,        % resolve_transport(Pred1, Pred2, Transport)

    % Listing
    list_mappings/1,            % list_mappings(Mappings)
    list_connections/1,         % list_connections(Connections)

    % Validation
    validate_mapping/2          % validate_mapping(Pred/Arity, Errors)
]).

:- use_module(library(lists)).
:- use_module(target_registry).

%% ============================================
%% Dynamic storage
%% ============================================

:- dynamic user_target/3.       % user_target(Pred/Arity, Target, Options)
:- dynamic user_location/2.     % user_location(Pred/Arity, Options)
:- dynamic user_connection/3.   % user_connection(Pred1/Arity1, Pred2/Arity2, Options)

%% ============================================
%% Target Declarations
%% ============================================

%% declare_target(+Pred/Arity, +Target)
%  Declare which target should compile a predicate.
%
%  Example:
%    :- declare_target(filter_logs/2, awk).
%
declare_target(Pred/Arity, Target) :-
    declare_target(Pred/Arity, Target, []).

%% declare_target(+Pred/Arity, +Target, +Options)
%  Declare target with additional options.
%
%  Options:
%    - optimize(speed | size | memory)
%    - streaming(true | false)
%    - priority(N) - for conflict resolution
%
%  Example:
%    :- declare_target(process/2, go, [optimize(speed), streaming(true)]).
%
declare_target(Pred/Arity, Target, Options) :-
    atom(Pred),
    integer(Arity),
    Arity >= 0,
    atom(Target),
    is_list(Options),
    % Remove existing mapping if present
    retractall(user_target(Pred/Arity, _, _)),
    assertz(user_target(Pred/Arity, Target, Options)).

%% undeclare_target(+Pred/Arity)
%  Remove a target declaration.
%
undeclare_target(Pred/Arity) :-
    retractall(user_target(Pred/Arity, _, _)).

%% ============================================
%% Location Declarations
%% ============================================

%% declare_location(+Pred/Arity, +Options)
%  Declare where a predicate should execute.
%
%  Options:
%    - process(same | separate)
%    - host(Hostname)
%    - port(Port)
%
%  Example:
%    :- declare_location(ml_model/2, [host('ml-server'), port(8080)]).
%
declare_location(Pred/Arity, Options) :-
    atom(Pred),
    integer(Arity),
    Arity >= 0,
    is_list(Options),
    retractall(user_location(Pred/Arity, _)),
    assertz(user_location(Pred/Arity, Options)).

%% undeclare_location(+Pred/Arity)
%  Remove a location declaration.
%
undeclare_location(Pred/Arity) :-
    retractall(user_location(Pred/Arity, _)).

%% ============================================
%% Connection Declarations
%% ============================================

%% declare_connection(+Pred1/Arity1, +Pred2/Arity2, +Options)
%  Declare how two predicates should communicate.
%
%  Options:
%    - transport(pipe | socket | http | grpc)
%    - format(tsv | json | binary)
%    - buffer(none | line | block(Size))
%    - timeout(Seconds)
%    - retry(Count)
%
%  Example:
%    :- declare_connection(producer/2, consumer/2, [transport(pipe), format(json)]).
%
declare_connection(Pred1/Arity1, Pred2/Arity2, Options) :-
    atom(Pred1), integer(Arity1), Arity1 >= 0,
    atom(Pred2), integer(Arity2), Arity2 >= 0,
    is_list(Options),
    % Store in canonical order (alphabetically by predicate name)
    canonical_pair(Pred1/Arity1, Pred2/Arity2, P1, P2),
    retractall(user_connection(P1, P2, _)),
    assertz(user_connection(P1, P2, Options)).

%% undeclare_connection(+Pred1/Arity1, +Pred2/Arity2)
%  Remove a connection declaration.
%
undeclare_connection(Pred1/Arity1, Pred2/Arity2) :-
    canonical_pair(Pred1/Arity1, Pred2/Arity2, P1, P2),
    retractall(user_connection(P1, P2, _)).

%% canonical_pair(+A, +B, -First, -Second)
%  Order a pair canonically for storage.
%
canonical_pair(A, B, A, B) :- A @=< B, !.
canonical_pair(A, B, B, A).

%% ============================================
%% Queries
%% ============================================

%% predicate_target(?Pred/Arity, ?Target)
%  Query which target compiles a predicate.
%
predicate_target(Pred/Arity, Target) :-
    user_target(Pred/Arity, Target, _).

%% predicate_target_options(?Pred/Arity, ?Target, ?Options)
%  Query target with options.
%
predicate_target_options(Pred/Arity, Target, Options) :-
    user_target(Pred/Arity, Target, Options).

%% predicate_location(?Pred/Arity, ?Location)
%  Query explicit location options for a predicate.
%
predicate_location(Pred/Arity, Location) :-
    user_location(Pred/Arity, Location).

%% connection_transport(+Pred1, +Pred2, -Transport)
%  Get the declared transport between two predicates.
%
connection_transport(Pred1, Pred2, Transport) :-
    connection_options(Pred1, Pred2, Options),
    member(transport(Transport), Options).

%% connection_options(?Pred1, ?Pred2, ?Options)
%  Query connection options between predicates.
%
connection_options(Pred1, Pred2, Options) :-
    canonical_pair(Pred1, Pred2, P1, P2),
    user_connection(P1, P2, Options).

%% ============================================
%% Resolution (with defaults)
%% ============================================

%% resolve_location(+Pred/Arity, -Location)
%  Resolve the location for a predicate, using defaults if not declared.
%
resolve_location(Pred/Arity, Location) :-
    % Check for explicit location
    user_location(Pred/Arity, Options),
    !,
    options_to_location(Options, Location).

resolve_location(Pred/Arity, Location) :-
    % Use default based on target
    predicate_target(Pred/Arity, Target),
    !,
    target_registry:default_location(Target, Location).

resolve_location(_, local_process).  % Ultimate fallback

%% options_to_location(+Options, -Location)
%  Convert location options to a location term.
%
options_to_location(Options, remote(Host)) :-
    member(host(Host), Options),
    !.
options_to_location(Options, in_process) :-
    member(process(same), Options),
    !.
options_to_location(Options, local_process) :-
    member(process(separate), Options),
    !.
options_to_location(_, local_process).  % Default

%% resolve_transport(+Pred1, +Pred2, -Transport)
%  Resolve the transport between two predicates, using defaults if not declared.
%
resolve_transport(Pred1, Pred2, Transport) :-
    % Check for explicit connection
    connection_transport(Pred1, Pred2, Transport),
    !.

resolve_transport(Pred1, Pred2, Transport) :-
    % Use default based on locations
    resolve_location(Pred1, Loc1),
    resolve_location(Pred2, Loc2),
    target_registry:default_transport(Loc1, Loc2, Transport).

%% ============================================
%% Listing
%% ============================================

%% list_mappings(-Mappings)
%  Get all predicate-to-target mappings.
%  Returns list of mapping(Pred/Arity, Target, Options).
%
list_mappings(Mappings) :-
    findall(
        mapping(Pred/Arity, Target, Options),
        user_target(Pred/Arity, Target, Options),
        Mappings
    ).

%% list_connections(-Connections)
%  Get all connection declarations.
%  Returns list of connection(Pred1, Pred2, Options).
%
list_connections(Connections) :-
    findall(
        connection(P1, P2, Options),
        user_connection(P1, P2, Options),
        Connections
    ).

%% ============================================
%% Validation
%% ============================================

%% validate_mapping(+Pred/Arity, -Errors)
%  Validate a mapping and return any errors.
%
validate_mapping(Pred/Arity, Errors) :-
    findall(Error, mapping_error(Pred/Arity, Error), Errors).

%% mapping_error(+Pred/Arity, -Error)
%  Check for specific mapping errors.
%
mapping_error(Pred/Arity, error(no_target, Pred/Arity)) :-
    \+ user_target(Pred/Arity, _, _).

mapping_error(Pred/Arity, error(unknown_target, Target)) :-
    user_target(Pred/Arity, Target, _),
    \+ target_registry:target_exists(Target).

mapping_error(Pred/Arity, error(invalid_location_option, Option)) :-
    user_location(Pred/Arity, Options),
    member(Option, Options),
    \+ valid_location_option(Option).

%% valid_location_option(+Option)
%  Check if a location option is valid.
%
valid_location_option(process(same)).
valid_location_option(process(separate)).
valid_location_option(host(H)) :- atom(H).
valid_location_option(port(P)) :- integer(P), P > 0, P < 65536.

%% ============================================
%% Directive support
%% ============================================

%% Allow using as directives
:- multifile user:term_expansion/2.

user:term_expansion((:- target(Pred, Target)), []) :-
    declare_target(Pred, Target).

user:term_expansion((:- target(Pred, Target, Options)), []) :-
    declare_target(Pred, Target, Options).

user:term_expansion((:- location(Pred, Options)), []) :-
    declare_location(Pred, Options).

user:term_expansion((:- connection(P1, P2, Options)), []) :-
    declare_connection(P1, P2, Options).
