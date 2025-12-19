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
    is_valid_service_option/1,
    validate_service_options/2,
    service_type/2,
    % Phase 2: Transport helpers
    get_service_transport/2,
    get_service_protocol/2,
    get_service_timeout/2,
    is_cross_process_service/1,
    is_network_service/1,
    % Phase 4: Service mesh helpers
    get_load_balance_strategy/2,
    get_circuit_breaker_config/2,
    get_retry_config/2,
    get_backends/2,
    has_load_balancing/1,
    has_circuit_breaker/1,
    has_retry/1,
    is_service_mesh_service/1,
    % Phase 5: Polyglot service helpers
    get_target_language/2,
    get_service_dependencies/2,
    get_service_endpoint/2,
    is_polyglot_service/1,
    is_valid_target_language/1,
    is_valid_service_dependency/1,
    is_valid_endpoint_spec/1,
    % Phase 6: Distributed service helpers
    get_replication_factor/2,
    get_consistency_level/2,
    get_sharding_strategy/2,
    get_partition_key/2,
    get_cluster_config/2,
    is_distributed_service/1,
    is_valid_consistency_level/1,
    is_valid_sharding_strategy/1,
    % Phase 7: Service Discovery helpers
    is_discovery_enabled/1,
    get_health_check_config/2,
    get_discovery_ttl/2,
    get_discovery_backend/2,
    get_discovery_tags/2,
    is_valid_health_check_config/1,
    is_valid_discovery_backend/1,
    % Phase 8: Service Tracing helpers
    is_tracing_enabled/1,
    get_trace_sampling/2,
    get_trace_exporter/2,
    get_trace_service_name/2,
    get_trace_propagation/2,
    get_trace_attributes/2,
    is_valid_trace_exporter/1,
    is_valid_trace_propagation/1,
    % Phase 3 KG Topology: Kleinberg routing helpers
    is_valid_routing_strategy/1,
    is_valid_kleinberg_option/1,
    is_valid_discovery_metadata_entry/1,
    is_kleinberg_routed/1,
    get_kleinberg_options/2,
    get_kleinberg_alpha/2,
    get_kleinberg_max_hops/2,
    get_kleinberg_parallel_paths/2,
    get_kleinberg_similarity_threshold/2,
    get_kleinberg_path_folding/2,
    get_semantic_centroid/2,
    get_interface_topics/2
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

%% Phase 4: Service Mesh Options

% Load balancing strategies
is_valid_service_option(load_balance(Strategy)) :-
    is_valid_load_balance_strategy(Strategy).

is_valid_service_option(load_balance(Strategy, Options)) :-
    is_valid_load_balance_strategy(Strategy),
    is_list(Options),
    maplist(is_valid_load_balance_option, Options).

% Circuit breaker configuration
is_valid_service_option(circuit_breaker(Threshold, Timeout)) :-
    is_valid_circuit_breaker_threshold(Threshold),
    is_valid_circuit_breaker_timeout(Timeout).

is_valid_service_option(circuit_breaker(Threshold, Timeout, Option)) :-
    is_valid_circuit_breaker_threshold(Threshold),
    is_valid_circuit_breaker_timeout(Timeout),
    is_valid_circuit_breaker_option(Option).

is_valid_service_option(circuit_breaker(Threshold, Timeout, Option1, Option2)) :-
    is_valid_circuit_breaker_threshold(Threshold),
    is_valid_circuit_breaker_timeout(Timeout),
    is_valid_circuit_breaker_option(Option1),
    is_valid_circuit_breaker_option(Option2).

is_valid_service_option(circuit_breaker(Options)) :-
    is_list(Options),
    maplist(is_valid_circuit_breaker_option, Options).

% Retry with backoff
is_valid_service_option(retry(N, Strategy)) :-
    integer(N),
    N > 0,
    is_valid_retry_strategy(Strategy).

is_valid_service_option(retry(N, Strategy, Option)) :-
    integer(N),
    N > 0,
    is_valid_retry_strategy(Strategy),
    is_valid_retry_option(Option).

is_valid_service_option(retry(N, Strategy, Option1, Option2)) :-
    integer(N),
    N > 0,
    is_valid_retry_strategy(Strategy),
    is_valid_retry_option(Option1),
    is_valid_retry_option(Option2).

is_valid_service_option(retry(N, Strategy, Options)) :-
    integer(N),
    N > 0,
    is_valid_retry_strategy(Strategy),
    is_list(Options),
    maplist(is_valid_retry_option, Options).

% Service discovery
is_valid_service_option(discovery(Method)) :-
    is_valid_discovery_method(Method).

% Backend pool for load balancing
is_valid_service_option(backends(Backends)) :-
    is_list(Backends),
    maplist(is_valid_backend, Backends).

%% Phase 5: Polyglot Service Options

% Target language for cross-language service calls
is_valid_service_option(target_language(Lang)) :-
    is_valid_target_language(Lang).

% Service dependencies (other services this service calls)
is_valid_service_option(depends_on(Services)) :-
    is_list(Services),
    maplist(is_valid_service_dependency, Services).

% Mark as polyglot-aware service
is_valid_service_option(polyglot(Bool)) :-
    ( Bool = true ; Bool = false ).

% Service endpoint for cross-language calls
is_valid_service_option(endpoint(Endpoint)) :-
    ( atom(Endpoint) ; is_valid_endpoint_spec(Endpoint) ).

%% Phase 6: Distributed Service Options

% Mark as distributed service
is_valid_service_option(distributed(Bool)) :-
    ( Bool = true ; Bool = false ).

% Replication factor
is_valid_service_option(replication(N)) :-
    integer(N),
    N > 0.

% Consistency level
is_valid_service_option(consistency(Level)) :-
    is_valid_consistency_level(Level).

% Sharding strategy
is_valid_service_option(sharding(Strategy)) :-
    is_valid_sharding_strategy(Strategy).

% Partition key for sharding
is_valid_service_option(partition_key(Key)) :-
    atom(Key).

% Cluster configuration
is_valid_service_option(cluster(Config)) :-
    is_valid_cluster_config(Config).

%% Load balance validation
is_valid_load_balance_strategy(round_robin).
is_valid_load_balance_strategy(random).
is_valid_load_balance_strategy(least_connections).
is_valid_load_balance_strategy(weighted).
is_valid_load_balance_strategy(ip_hash).

is_valid_load_balance_option(sticky(Bool)) :- ( Bool = true ; Bool = false ).
is_valid_load_balance_option(health_check(Interval)) :- integer(Interval), Interval > 0.

%% Circuit breaker validation
is_valid_circuit_breaker_threshold(threshold(N)) :- integer(N), N > 0.
is_valid_circuit_breaker_timeout(timeout(Ms)) :- integer(Ms), Ms > 0.

is_valid_circuit_breaker_option(threshold(N)) :- integer(N), N > 0.
is_valid_circuit_breaker_option(timeout(Ms)) :- integer(Ms), Ms > 0.
is_valid_circuit_breaker_option(half_open_requests(N)) :- integer(N), N > 0.
is_valid_circuit_breaker_option(success_threshold(N)) :- integer(N), N > 0.

%% Retry validation
is_valid_retry_strategy(fixed).
is_valid_retry_strategy(linear).
is_valid_retry_strategy(exponential).

is_valid_retry_option(delay(Ms)) :- integer(Ms), Ms >= 0.
is_valid_retry_option(max_delay(Ms)) :- integer(Ms), Ms > 0.
is_valid_retry_option(jitter(Bool)) :- ( Bool = true ; Bool = false ).

%% Discovery validation
is_valid_discovery_method(static).
is_valid_discovery_method(dns).
is_valid_discovery_method(consul).
is_valid_discovery_method(etcd).

%% Backend validation
is_valid_backend(backend(Name, Transport)) :-
    atom(Name),
    is_valid_transport(Transport).
is_valid_backend(backend(Name, Transport, Options)) :-
    atom(Name),
    is_valid_transport(Transport),
    is_list(Options).

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

%% Phase 5: Polyglot Service Validation

% Valid target languages
is_valid_target_language(python).
is_valid_target_language(go).
is_valid_target_language(rust).
is_valid_target_language(csharp).
is_valid_target_language(java).
is_valid_target_language(javascript).
is_valid_target_language(typescript).

% Service dependency validation
is_valid_service_dependency(ServiceName) :- atom(ServiceName).
is_valid_service_dependency(dep(ServiceName, Lang)) :-
    atom(ServiceName),
    is_valid_target_language(Lang).
is_valid_service_dependency(dep(ServiceName, Lang, Transport)) :-
    atom(ServiceName),
    is_valid_target_language(Lang),
    is_valid_transport(Transport).

% Endpoint specification validation
is_valid_endpoint_spec(http(Host, Port)) :-
    atom(Host),
    integer(Port),
    Port > 0.
is_valid_endpoint_spec(http(Host, Port, Path)) :-
    atom(Host),
    integer(Port),
    Port > 0,
    atom(Path).
is_valid_endpoint_spec(grpc(Host, Port)) :-
    atom(Host),
    integer(Port),
    Port > 0.

%% Phase 6: Distributed Service Validation

% Consistency levels
is_valid_consistency_level(eventual).
is_valid_consistency_level(strong).
is_valid_consistency_level(causal).
is_valid_consistency_level(read_your_writes).
is_valid_consistency_level(monotonic_reads).
is_valid_consistency_level(quorum).

% Sharding strategies
is_valid_sharding_strategy(hash).
is_valid_sharding_strategy(range).
is_valid_sharding_strategy(consistent_hash).
is_valid_sharding_strategy(geographic).
is_valid_sharding_strategy(custom(Pred)) :- atom(Pred).

% Cluster configuration validation
is_valid_cluster_config(nodes(Nodes)) :-
    is_list(Nodes),
    maplist(is_valid_node_spec, Nodes).
is_valid_cluster_config(discovery(Method)) :-
    is_valid_discovery_method(Method).
is_valid_cluster_config(config(Options)) :-
    is_list(Options).

% Node specification validation
is_valid_node_spec(node(Name, Host, Port)) :-
    atom(Name),
    atom(Host),
    integer(Port),
    Port > 0.
is_valid_node_spec(node(Name, Endpoint)) :-
    atom(Name),
    ( atom(Endpoint) ; is_valid_endpoint_spec(Endpoint) ).

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

%% ============================================
%% Phase 4: Service Mesh Helper Predicates
%% ============================================

%% get_load_balance_strategy(+Service, -Strategy)
%  Extract load balancing strategy from service definition.
get_load_balance_strategy(service(_, Options, _), Strategy) :-
    is_list(Options),
    ( member(load_balance(Strategy), Options) ->
        true
    ; member(load_balance(Strategy, _), Options) ->
        true
    ),
    !.
get_load_balance_strategy(_, none).

%% get_circuit_breaker_config(+Service, -Config)
%  Extract circuit breaker configuration from service definition.
get_circuit_breaker_config(service(_, Options, _), config(Threshold, Timeout)) :-
    is_list(Options),
    member(circuit_breaker(threshold(Threshold), timeout(Timeout)), Options),
    !.
get_circuit_breaker_config(service(_, Options, _), config(Threshold, Timeout, HalfOpen, SuccessThresh)) :-
    is_list(Options),
    member(circuit_breaker(CBOptions), Options),
    is_list(CBOptions),
    ( member(threshold(Threshold), CBOptions) -> true ; Threshold = 5 ),
    ( member(timeout(Timeout), CBOptions) -> true ; Timeout = 30000 ),
    ( member(half_open_requests(HalfOpen), CBOptions) -> true ; HalfOpen = 1 ),
    ( member(success_threshold(SuccessThresh), CBOptions) -> true ; SuccessThresh = 1 ),
    !.
get_circuit_breaker_config(_, none).

%% get_retry_config(+Service, -Config)
%  Extract retry configuration from service definition.
get_retry_config(service(_, Options, _), config(N, Strategy, Delay, MaxDelay, Jitter)) :-
    is_list(Options),
    ( member(retry(N, Strategy, RetryOptions), Options) ->
        ( member(delay(Delay), RetryOptions) -> true ; Delay = 100 ),
        ( member(max_delay(MaxDelay), RetryOptions) -> true ; MaxDelay = 30000 ),
        ( member(jitter(Jitter), RetryOptions) -> true ; Jitter = false )
    ; member(retry(N, Strategy), Options) ->
        Delay = 100,
        MaxDelay = 30000,
        Jitter = false
    ),
    !.
get_retry_config(_, none).

%% get_backends(+Service, -Backends)
%  Extract backend pool from service definition.
get_backends(service(_, Options, _), Backends) :-
    is_list(Options),
    member(backends(Backends), Options),
    !.
get_backends(_, []).

%% has_load_balancing(+Service)
%  Succeeds if service has load balancing configured.
has_load_balancing(service(_, Options, _)) :-
    is_list(Options),
    ( member(load_balance(_), Options) ; member(load_balance(_, _), Options) ),
    !.

%% has_circuit_breaker(+Service)
%  Succeeds if service has circuit breaker configured.
has_circuit_breaker(service(_, Options, _)) :-
    is_list(Options),
    ( member(circuit_breaker(_, _), Options) ; member(circuit_breaker(_), Options) ),
    !.

%% has_retry(+Service)
%  Succeeds if service has retry configured.
has_retry(service(_, Options, _)) :-
    is_list(Options),
    ( member(retry(_, _), Options) ; member(retry(_, _, _), Options) ),
    !.

%% is_service_mesh_service(+Service)
%  Succeeds if service has any service mesh features.
is_service_mesh_service(Service) :-
    ( has_load_balancing(Service)
    ; has_circuit_breaker(Service)
    ; has_retry(Service)
    ),
    !.

%% ============================================
%% Phase 5: Polyglot Service Helper Predicates
%% ============================================

%% get_target_language(+Service, -Language)
%  Extract target language from service definition.
get_target_language(service(_, Options, _), Language) :-
    is_list(Options),
    member(target_language(Language), Options),
    !.
get_target_language(_, none).

%% get_service_dependencies(+Service, -Dependencies)
%  Extract service dependencies from service definition.
get_service_dependencies(service(_, Options, _), Dependencies) :-
    is_list(Options),
    member(depends_on(Dependencies), Options),
    !.
get_service_dependencies(_, []).

%% is_polyglot_service(+Service)
%  Succeeds if service is marked as polyglot or has cross-language dependencies.
is_polyglot_service(service(_, Options, _)) :-
    is_list(Options),
    ( member(polyglot(true), Options)
    ; member(depends_on(Deps), Options), Deps \= []
    ; member(target_language(_), Options)
    ),
    !.

%% get_service_endpoint(+Service, -Endpoint)
%  Extract endpoint configuration from service definition.
get_service_endpoint(service(_, Options, _), Endpoint) :-
    is_list(Options),
    member(endpoint(Endpoint), Options),
    !.
get_service_endpoint(Service, Endpoint) :-
    % Fall back to transport-derived endpoint
    get_service_transport(Service, Transport),
    transport_to_endpoint(Transport, Endpoint),
    !.
get_service_endpoint(_, none).

%% transport_to_endpoint(+Transport, -Endpoint)
%  Convert transport to endpoint specification.
transport_to_endpoint(tcp(Host, Port), http(Host, Port)) :- !.
transport_to_endpoint(http(Path), http(localhost, 8080, Path)) :- !.
transport_to_endpoint(http(Path, _Options), http(localhost, 8080, Path)) :- !.
transport_to_endpoint(_, none).

%% ============================================
%% Phase 6: Distributed Service Helper Predicates
%% ============================================

%% get_replication_factor(+Service, -Factor)
%  Extract replication factor from service definition.
get_replication_factor(service(_, Options, _), Factor) :-
    is_list(Options),
    member(replication(Factor), Options),
    !.
get_replication_factor(_, 1).

%% get_consistency_level(+Service, -Level)
%  Extract consistency level from service definition.
get_consistency_level(service(_, Options, _), Level) :-
    is_list(Options),
    member(consistency(Level), Options),
    !.
get_consistency_level(_, eventual).

%% get_sharding_strategy(+Service, -Strategy)
%  Extract sharding strategy from service definition.
get_sharding_strategy(service(_, Options, _), Strategy) :-
    is_list(Options),
    member(sharding(Strategy), Options),
    !.
get_sharding_strategy(_, hash).

%% get_partition_key(+Service, -Key)
%  Extract partition key from service definition.
get_partition_key(service(_, Options, _), Key) :-
    is_list(Options),
    member(partition_key(Key), Options),
    !.
get_partition_key(_, id).

%% get_cluster_config(+Service, -Config)
%  Extract cluster configuration from service definition.
get_cluster_config(service(_, Options, _), Config) :-
    is_list(Options),
    member(cluster(Config), Options),
    !.
get_cluster_config(_, none).

%% is_distributed_service(+Service)
%  Succeeds if service is marked as distributed.
is_distributed_service(service(_, Options, _)) :-
    is_list(Options),
    ( member(distributed(true), Options)
    ; member(replication(N), Options), N > 1
    ; member(sharding(_), Options)
    ; member(cluster(_), Options)
    ),
    !.

%% ============================================
%% Phase 7: Service Discovery Options
%% ============================================

%% discovery_enabled - Enable service discovery
is_valid_service_option(discovery_enabled(Bool)) :-
    ( Bool = true ; Bool = false ).

%% health_check - Health check configuration
is_valid_service_option(health_check(Config)) :-
    is_valid_health_check_config(Config).

%% discovery_ttl - Time-to-live for service registration
is_valid_service_option(discovery_ttl(Seconds)) :-
    integer(Seconds),
    Seconds > 0.

%% discovery_backend - Service discovery backend
is_valid_service_option(discovery_backend(Backend)) :-
    is_valid_discovery_backend(Backend).

%% discovery_tags - Tags for service filtering
is_valid_service_option(discovery_tags(Tags)) :-
    is_list(Tags),
    maplist(atom, Tags).

%% discovery_metadata - Additional metadata for registration
is_valid_service_option(discovery_metadata(Metadata)) :-
    is_list(Metadata).

%% Health check configuration validation
is_valid_health_check_config(http(Path, Interval)) :-
    atom(Path),
    integer(Interval),
    Interval > 0.
is_valid_health_check_config(http(Path, Interval, Timeout)) :-
    atom(Path),
    integer(Interval),
    Interval > 0,
    integer(Timeout),
    Timeout > 0.
is_valid_health_check_config(tcp(Port, Interval)) :-
    integer(Port),
    Port > 0,
    integer(Interval),
    Interval > 0.
is_valid_health_check_config(grpc(Interval)) :-
    integer(Interval),
    Interval > 0.
is_valid_health_check_config(script(Command, Interval)) :-
    atom(Command),
    integer(Interval),
    Interval > 0.
is_valid_health_check_config(config(Options)) :-
    is_list(Options).

%% Discovery backend validation
is_valid_discovery_backend(consul).
is_valid_discovery_backend(consul(Host, Port)) :-
    atom(Host),
    integer(Port),
    Port > 0.
is_valid_discovery_backend(etcd).
is_valid_discovery_backend(etcd(Endpoints)) :-
    is_list(Endpoints).
is_valid_discovery_backend(dns).
is_valid_discovery_backend(dns(Domain)) :-
    atom(Domain).
is_valid_discovery_backend(kubernetes).
is_valid_discovery_backend(kubernetes(Namespace)) :-
    atom(Namespace).
is_valid_discovery_backend(zookeeper(Hosts)) :-
    is_list(Hosts).
is_valid_discovery_backend(eureka(Url)) :-
    atom(Url).

%% ============================================
%% Phase 7: Service Discovery Helper Predicates
%% ============================================

%% is_discovery_enabled(+Service)
%  Succeeds if service has discovery enabled.
is_discovery_enabled(service(_, Options, _)) :-
    is_list(Options),
    ( member(discovery_enabled(true), Options)
    ; member(discovery_backend(_), Options)
    ; member(health_check(_), Options)
    ),
    !.

%% get_health_check_config(+Service, -Config)
%  Extract health check configuration from service definition.
get_health_check_config(service(_, Options, _), Config) :-
    is_list(Options),
    member(health_check(Config), Options),
    !.
get_health_check_config(_, http('/health', 30000)).

%% get_discovery_ttl(+Service, -TTL)
%  Extract discovery TTL from service definition.
get_discovery_ttl(service(_, Options, _), TTL) :-
    is_list(Options),
    member(discovery_ttl(TTL), Options),
    !.
get_discovery_ttl(_, 60).

%% get_discovery_backend(+Service, -Backend)
%  Extract discovery backend from service definition.
get_discovery_backend(service(_, Options, _), Backend) :-
    is_list(Options),
    member(discovery_backend(Backend), Options),
    !.
get_discovery_backend(_, consul).

%% get_discovery_tags(+Service, -Tags)
%  Extract discovery tags from service definition.
get_discovery_tags(service(_, Options, _), Tags) :-
    is_list(Options),
    member(discovery_tags(Tags), Options),
    !.
get_discovery_tags(_, []).

%% ============================================
%% Phase 8: Service Tracing Options
%% ============================================

%% tracing - Enable distributed tracing
is_valid_service_option(tracing(Bool)) :-
    ( Bool = true ; Bool = false ).

%% trace_sampling - Sampling rate (0.0 to 1.0)
is_valid_service_option(trace_sampling(Rate)) :-
    number(Rate),
    Rate >= 0.0,
    Rate =< 1.0.

%% trace_exporter - Tracing exporter backend
is_valid_service_option(trace_exporter(Exporter)) :-
    is_valid_trace_exporter(Exporter).

%% trace_service_name - Service name for tracing
is_valid_service_option(trace_service_name(Name)) :-
    atom(Name).

%% trace_propagation - Context propagation format
is_valid_service_option(trace_propagation(Format)) :-
    is_valid_trace_propagation(Format).

%% trace_attributes - Additional span attributes
is_valid_service_option(trace_attributes(Attrs)) :-
    is_list(Attrs).

%% trace_batch_size - Batch size for exporting spans
is_valid_service_option(trace_batch_size(Size)) :-
    integer(Size),
    Size > 0.

%% trace_export_interval - Export interval in milliseconds
is_valid_service_option(trace_export_interval(Ms)) :-
    integer(Ms),
    Ms > 0.

%% Trace exporter validation
is_valid_trace_exporter(jaeger).
is_valid_trace_exporter(jaeger(Endpoint)) :-
    atom(Endpoint).
is_valid_trace_exporter(jaeger(Host, Port)) :-
    atom(Host),
    integer(Port),
    Port > 0.
is_valid_trace_exporter(zipkin).
is_valid_trace_exporter(zipkin(Endpoint)) :-
    atom(Endpoint).
is_valid_trace_exporter(otlp).
is_valid_trace_exporter(otlp(Endpoint)) :-
    atom(Endpoint).
is_valid_trace_exporter(otlp(Endpoint, Options)) :-
    atom(Endpoint),
    is_list(Options).
is_valid_trace_exporter(datadog).
is_valid_trace_exporter(datadog(AgentHost)) :-
    atom(AgentHost).
is_valid_trace_exporter(console).
is_valid_trace_exporter(none).

%% Trace propagation format validation
is_valid_trace_propagation(w3c).
is_valid_trace_propagation(b3).
is_valid_trace_propagation(b3_multi).
is_valid_trace_propagation(jaeger).
is_valid_trace_propagation(xray).
is_valid_trace_propagation(datadog).

%% ============================================
%% Phase 8: Service Tracing Helper Predicates
%% ============================================

%% is_tracing_enabled(+Service)
%  Succeeds if service has tracing enabled.
is_tracing_enabled(service(_, Options, _)) :-
    is_list(Options),
    ( member(tracing(true), Options)
    ; member(trace_exporter(_), Options)
    ; member(trace_sampling(_), Options)
    ),
    !.

%% get_trace_sampling(+Service, -Rate)
%  Extract trace sampling rate from service definition.
get_trace_sampling(service(_, Options, _), Rate) :-
    is_list(Options),
    member(trace_sampling(Rate), Options),
    !.
get_trace_sampling(_, 1.0).

%% get_trace_exporter(+Service, -Exporter)
%  Extract trace exporter from service definition.
get_trace_exporter(service(_, Options, _), Exporter) :-
    is_list(Options),
    member(trace_exporter(Exporter), Options),
    !.
get_trace_exporter(_, otlp).

%% get_trace_service_name(+Service, -Name)
%  Extract trace service name from service definition.
get_trace_service_name(service(Name, Options, _), ServiceName) :-
    is_list(Options),
    member(trace_service_name(ServiceName), Options),
    !.
get_trace_service_name(service(Name, _, _), Name) :- !.
get_trace_service_name(service(Name, _), Name).

%% get_trace_propagation(+Service, -Format)
%  Extract trace propagation format from service definition.
get_trace_propagation(service(_, Options, _), Format) :-
    is_list(Options),
    member(trace_propagation(Format), Options),
    !.
get_trace_propagation(_, w3c).

%% get_trace_attributes(+Service, -Attributes)
%  Extract trace attributes from service definition.
get_trace_attributes(service(_, Options, _), Attrs) :-
    is_list(Options),
    member(trace_attributes(Attrs), Options),
    !.
get_trace_attributes(_, []).

%% ============================================
%% KG Topology Phase 3: Kleinberg Routing Options
%% ============================================

%% routing - Query routing strategy for distributed KG topology
is_valid_service_option(routing(Strategy)) :-
    is_valid_routing_strategy(Strategy).

%% Valid routing strategies
is_valid_routing_strategy(direct).           % No routing, local only
is_valid_routing_strategy(round_robin).      % Simple load balancing
is_valid_routing_strategy(kleinberg).        % Semantic small-world routing
is_valid_routing_strategy(kleinberg(Opts)) :-
    is_list(Opts),
    maplist(is_valid_kleinberg_option, Opts).

%% Kleinberg routing options
is_valid_kleinberg_option(alpha(A)) :-
    number(A), A > 0.
is_valid_kleinberg_option(max_hops(N)) :-
    integer(N), N > 0.
is_valid_kleinberg_option(parallel_paths(N)) :-
    integer(N), N > 0.
is_valid_kleinberg_option(similarity_threshold(T)) :-
    number(T), T >= 0, T =< 1.
is_valid_kleinberg_option(path_folding(Bool)) :-
    ( Bool = true ; Bool = false ).
is_valid_kleinberg_option(embedding_model(Model)) :-
    atom(Model).

%% Semantic discovery metadata entries (for interface centroid publishing)
is_valid_discovery_metadata_entry(semantic_centroid(Centroid)) :-
    ( is_list(Centroid) ; atom(Centroid) ).  % List of floats or base64 string
is_valid_discovery_metadata_entry(interface_topics(Topics)) :-
    is_list(Topics),
    maplist(atom, Topics).
is_valid_discovery_metadata_entry(embedding_model(Model)) :-
    atom(Model).

%% ============================================
%% KG Topology Phase 3: Kleinberg Routing Helper Predicates
%% ============================================

%% is_kleinberg_routed(+Service)
%  Succeeds if service uses Kleinberg routing.
is_kleinberg_routed(service(_, Options, _)) :-
    is_list(Options),
    ( member(routing(kleinberg), Options)
    ; member(routing(kleinberg(_)), Options)
    ),
    !.

%% get_kleinberg_options(+Service, -Options)
%  Extract Kleinberg routing options from service definition.
get_kleinberg_options(service(_, Options, _), KleinbergOpts) :-
    is_list(Options),
    member(routing(kleinberg(KleinbergOpts)), Options),
    !.
get_kleinberg_options(service(_, Options, _), []) :-
    is_list(Options),
    member(routing(kleinberg), Options),
    !.
get_kleinberg_options(_, []).

%% get_kleinberg_alpha(+Service, -Alpha)
%  Extract Kleinberg alpha parameter (link distribution exponent).
get_kleinberg_alpha(Service, Alpha) :-
    get_kleinberg_options(Service, Opts),
    member(alpha(Alpha), Opts),
    !.
get_kleinberg_alpha(_, 2.0).

%% get_kleinberg_max_hops(+Service, -MaxHops)
%  Extract maximum hops (HTL) for Kleinberg routing.
get_kleinberg_max_hops(Service, MaxHops) :-
    get_kleinberg_options(Service, Opts),
    member(max_hops(MaxHops), Opts),
    !.
get_kleinberg_max_hops(_, 10).

%% get_kleinberg_parallel_paths(+Service, -Paths)
%  Extract number of parallel paths for Kleinberg routing.
get_kleinberg_parallel_paths(Service, Paths) :-
    get_kleinberg_options(Service, Opts),
    member(parallel_paths(Paths), Opts),
    !.
get_kleinberg_parallel_paths(_, 1).

%% get_kleinberg_similarity_threshold(+Service, -Threshold)
%  Extract similarity threshold for Kleinberg routing.
get_kleinberg_similarity_threshold(Service, Threshold) :-
    get_kleinberg_options(Service, Opts),
    member(similarity_threshold(Threshold), Opts),
    !.
get_kleinberg_similarity_threshold(_, 0.5).

%% get_kleinberg_path_folding(+Service, -Enabled)
%  Check if path folding is enabled for Kleinberg routing.
get_kleinberg_path_folding(Service, Enabled) :-
    get_kleinberg_options(Service, Opts),
    member(path_folding(Enabled), Opts),
    !.
get_kleinberg_path_folding(_, true).

%% get_semantic_centroid(+Service, -Centroid)
%  Extract semantic centroid from discovery metadata.
get_semantic_centroid(service(_, Options, _), Centroid) :-
    is_list(Options),
    member(discovery_metadata(Metadata), Options),
    is_list(Metadata),
    member(semantic_centroid(Centroid), Metadata),
    !.
get_semantic_centroid(_, none).

%% get_interface_topics(+Service, -Topics)
%  Extract interface topics from discovery metadata.
get_interface_topics(service(_, Options, _), Topics) :-
    is_list(Options),
    member(discovery_metadata(Metadata), Options),
    is_list(Metadata),
    member(interface_topics(Topics), Metadata),
    !.
get_interface_topics(_, []).
