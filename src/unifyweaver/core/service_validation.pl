% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% service_validation.pl - Service Definition Validation
% Validates service definitions for the client-server architecture.

:- module(service_validation, [
    validate_service/2,
    is_valid_service/1,
    is_valid_handler_spec/1,
    is_valid_service_operation/1,
    validate_service_options/2,
    service_type/2,
    % Phase 2: Transport helpers
    get_service_transport/2,
    get_service_protocol/2,
    get_service_timeout/2,
    is_cross_process_service/1,
    is_network_service/1
]).

:- use_module(library(lists)).

%% validate_service(+Service, -Errors)
%  Validates a service definition and returns list of errors.
%  Empty list means valid service.
validate_service(service(Name, HandlerSpec), Errors) :-
    !,
    validate_service_name(Name, NameErrors),
    validate_handler_spec(HandlerSpec, HandlerErrors),
    append(NameErrors, HandlerErrors, Errors).

validate_service(service(Name, Options, HandlerSpec), Errors) :-
    !,
    validate_service_name(Name, NameErrors),
    validate_service_options(Options, OptionErrors),
    validate_handler_spec(HandlerSpec, HandlerErrors),
    append([NameErrors, OptionErrors, HandlerErrors], Errors).

validate_service(Invalid, [error(invalid_service_definition, Invalid)]).

%% validate_service_name(+Name, -Errors)
validate_service_name(Name, []) :-
    atom(Name),
    !.
validate_service_name(Name, [error(invalid_service_name, Name)]).

%% is_valid_service(+Service)
%  Succeeds if Service is a valid service definition.
is_valid_service(service(Name, HandlerSpec)) :-
    atom(Name),
    is_valid_handler_spec(HandlerSpec).

is_valid_service(service(Name, Options, HandlerSpec)) :-
    atom(Name),
    is_list(Options),
    maplist(is_valid_service_option, Options),
    is_valid_handler_spec(HandlerSpec).

%% is_valid_handler_spec(+HandlerSpec)
%  Validates the handler specification (list of operations).
is_valid_handler_spec([]).
is_valid_handler_spec([Op|Rest]) :-
    is_valid_service_operation(Op),
    is_valid_handler_spec(Rest).

%% validate_handler_spec(+HandlerSpec, -Errors)
validate_handler_spec(HandlerSpec, Errors) :-
    is_list(HandlerSpec),
    !,
    validate_operations(HandlerSpec, 1, Errors).
validate_handler_spec(Invalid, [error(handler_spec_not_list, Invalid)]).

validate_operations([], _, []).
validate_operations([Op|Rest], N, Errors) :-
    ( is_valid_service_operation(Op) ->
        OpErrors = []
    ;
        OpErrors = [error(invalid_operation_at_position, N, Op)]
    ),
    N1 is N + 1,
    validate_operations(Rest, N1, RestErrors),
    append(OpErrors, RestErrors, Errors).

%% is_valid_service_operation(+Operation)
%  Validates individual service operations.

% receive(Variable) - Bind request to variable
is_valid_service_operation(receive(_Var)) :- !.

% respond(Value) - Send response
is_valid_service_operation(respond(_Value)) :- !.

% respond_error(Error) - Send error response
is_valid_service_operation(respond_error(_Error)) :- !.

% Predicate/Arity - Execute predicate
is_valid_service_operation(Pred/Arity) :-
    atom(Pred),
    integer(Arity),
    Arity >= 0,
    !.

% Atom predicate (arity inferred)
is_valid_service_operation(Pred) :-
    atom(Pred),
    Pred \= receive,
    Pred \= respond,
    Pred \= respond_error,
    !.

% transform(In, Out) or transform(In, Out, Goal)
is_valid_service_operation(transform(_, _)) :- !.
is_valid_service_operation(transform(_, _, _)) :- !.

% State operations (for stateful services)
is_valid_service_operation(state_get(Key, _Value)) :-
    atom(Key),
    !.
is_valid_service_operation(state_put(Key, _Value)) :-
    atom(Key),
    !.
is_valid_service_operation(state_modify(Key, _Func)) :-
    atom(Key),
    !.
is_valid_service_operation(state_delete(Key)) :-
    atom(Key),
    !.

% Call another service
is_valid_service_operation(call_service(Name, _Request, _Response)) :-
    atom(Name),
    !.
is_valid_service_operation(call_service(Name, _Request, _Response, Options)) :-
    atom(Name),
    is_list(Options),
    !.

% Route by field value
is_valid_service_operation(route_by(Field, Routes)) :-
    atom(Field),
    is_list(Routes),
    maplist(is_valid_route, Routes),
    !.

% Branch on condition
is_valid_service_operation(branch(Cond, TrueOps, FalseOps)) :-
    ( atom(Cond) ; Cond = _/_ ),
    is_valid_handler_spec(TrueOps),
    is_valid_handler_spec(FalseOps),
    !.

% Nested handler spec (for routing)
is_valid_service_operation(Ops) :-
    is_list(Ops),
    is_valid_handler_spec(Ops),
    !.

%% is_valid_route(+Route)
is_valid_route((Value, Handler)) :-
    is_valid_handler_spec(Handler),
    ( atom(Value) ; number(Value) ; Value = _ ).

%% Service Options Validation

validate_service_options(Options, Errors) :-
    is_list(Options),
    !,
    findall(Error,
        (member(Opt, Options), \+ is_valid_service_option(Opt), Error = error(invalid_option, Opt)),
        Errors).
validate_service_options(Invalid, [error(options_not_list, Invalid)]).

is_valid_service_option(stateful(Bool)) :-
    ( Bool = true ; Bool = false ).

is_valid_service_option(transport(Transport)) :-
    is_valid_transport(Transport).

is_valid_service_option(protocol(Protocol)) :-
    is_valid_protocol(Protocol).

is_valid_service_option(timeout(Ms)) :-
    integer(Ms),
    Ms > 0.

is_valid_service_option(max_concurrent(N)) :-
    integer(N),
    N > 0.

is_valid_service_option(on_error(Handler)) :-
    ( atom(Handler) ; Handler = _/_ ).

%% Transport validation
is_valid_transport(in_process).
is_valid_transport(unix_socket(Path)) :- atom(Path).
is_valid_transport(tcp(Host, Port)) :- atom(Host), integer(Port), Port > 0.
is_valid_transport(http(Endpoint)) :- atom(Endpoint).
is_valid_transport(http(Endpoint, Options)) :- atom(Endpoint), is_list(Options).

%% Protocol validation
is_valid_protocol(jsonl).
is_valid_protocol(json).
is_valid_protocol(messagepack).
is_valid_protocol(protobuf(Schema)) :- atom(Schema).

%% service_type(+Service, -Type)
%  Determines the type of service (stateful/stateless, transport type).
service_type(service(_, Options, _), Type) :-
    !,
    ( member(stateful(true), Options) ->
        Stateful = stateful
    ;
        Stateful = stateless
    ),
    ( member(transport(Transport), Options) ->
        transport_category(Transport, TransportCat)
    ;
        TransportCat = in_process
    ),
    Type = service_type(Stateful, TransportCat).

service_type(service(_, _), service_type(stateless, in_process)).

transport_category(in_process, in_process).
transport_category(unix_socket(_), cross_process).
transport_category(tcp(_, _), network).
transport_category(http(_), network).
transport_category(http(_, _), network).

%% Utility predicates for querying services

%% has_state_operations(+HandlerSpec)
%  Succeeds if handler spec contains state operations.
has_state_operations(HandlerSpec) :-
    member(Op, HandlerSpec),
    ( Op = state_get(_, _)
    ; Op = state_put(_, _)
    ; Op = state_modify(_, _)
    ; Op = state_delete(_)
    ),
    !.

%% extract_receive_var(+HandlerSpec, -Var)
%  Extract the variable bound by receive operation.
extract_receive_var([receive(Var)|_], Var) :- !.
extract_receive_var([_|Rest], Var) :-
    extract_receive_var(Rest, Var).

%% extract_respond_value(+HandlerSpec, -Value)
%  Extract the value from respond operation.
extract_respond_value(HandlerSpec, Value) :-
    member(respond(Value), HandlerSpec),
    !.

%% ============================================
%% Phase 2: Transport Helper Predicates
%% ============================================

%% get_service_transport(+Service, -Transport)
%  Extract transport configuration from service definition.
get_service_transport(service(_, Options, _), Transport) :-
    is_list(Options),
    member(transport(Transport), Options),
    !.
get_service_transport(service(_, _), in_process).
get_service_transport(service(_, Options, _), in_process) :-
    is_list(Options),
    \+ member(transport(_), Options).

%% get_service_protocol(+Service, -Protocol)
%  Extract protocol configuration from service definition.
get_service_protocol(service(_, Options, _), Protocol) :-
    is_list(Options),
    member(protocol(Protocol), Options),
    !.
get_service_protocol(service(_, _), jsonl).
get_service_protocol(service(_, Options, _), jsonl) :-
    is_list(Options),
    \+ member(protocol(_), Options).

%% get_service_timeout(+Service, -Timeout)
%  Extract timeout configuration from service definition.
get_service_timeout(service(_, Options, _), Timeout) :-
    is_list(Options),
    member(timeout(Timeout), Options),
    !.
get_service_timeout(_, none).

%% is_cross_process_service(+Service)
%  Succeeds if service uses cross-process transport (Unix sockets).
is_cross_process_service(Service) :-
    get_service_transport(Service, Transport),
    transport_category(Transport, cross_process).

%% is_network_service(+Service)
%  Succeeds if service uses network transport (TCP/HTTP).
is_network_service(Service) :-
    get_service_transport(Service, Transport),
    transport_category(Transport, network).
