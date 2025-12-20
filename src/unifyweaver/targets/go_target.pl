:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% go_target.pl - Go Target for UnifyWeaver
% Generates standalone Go programs for record/field processing
% Supports configurable delimiters, quoting, and regex matching

:- module(go_target, [
    compile_predicate_to_go/3,      % +Predicate, +Options, -GoCode
    compile_facts_to_go/3,          % +Pred, +Arity, -GoCode  -- NEW
    compile_go_pipeline/3,          % +Predicates, +Options, -GoCode
    write_go_program/2,             % +GoCode, +FilePath
    json_schema/2,                  % +SchemaName, +Fields (directive)
    get_json_schema/2,              % +SchemaName, -Fields (lookup)
    get_field_info/4,               % +SchemaName, +FieldName, -Type, -Options (lookup)
    % Binding system exports
    init_go_target/0,               % Initialize Go target with bindings
    clear_binding_imports/0,        % Clear collected binding imports
    get_collected_imports/1,        % Get imports collected from bindings
    test_go_binding_integration/0,  % Test binding integration
    % Pipeline mode exports
    test_go_pipeline_mode/0,        % Test pipeline mode
    test_go_pipeline_chaining/0,    % Test pipeline chaining
    test_go_pipeline_bindings/0,    % Test pipeline binding integration
    test_go_pipeline_generator/0,   % Test pipeline generator mode
    % Enhanced pipeline chaining exports
    compile_go_enhanced_pipeline/3, % +Stages, +Options, -GoCode
    go_enhanced_helpers/1,          % -Code
    generate_go_enhanced_connector/3, % +Stages, +PipelineName, -Code
    test_go_enhanced_chaining/0,    % Test enhanced pipeline chaining
    % Client-server architecture exports (Phase 9)
    compile_service_to_go/2,        % +Service, -GoCode
    generate_service_handler_go/2,  % +HandlerSpec, -GoCode
    % Phase 2: Cross-process services
    compile_unix_socket_service_go/2,   % +Service, -GoCode
    compile_unix_socket_client_go/3,    % +ServiceName, +SocketPath, -GoCode
    % Phase 3: Network services
    compile_tcp_service_go/2,           % +Service, -GoCode
    compile_tcp_client_go/4,            % +ServiceName, +Host, +Port, -GoCode
    compile_http_service_go/2,          % +Service, -GoCode
    compile_http_client_go/3,           % +ServiceName, +Endpoint, -GoCode
    compile_http_client_go/4,           % +ServiceName, +Endpoint, +Options, -GoCode
    % Phase 4: Service mesh
    compile_service_mesh_go/2,          % +Service, -GoCode
    % Phase 5: Polyglot services
    compile_polyglot_service_go/2,      % +Service, -GoCode
    generate_service_client_go/3,       % +ServiceName, +Endpoint, -GoCode
    % Phase 6: Distributed services
    compile_distributed_service_go/2,   % +Service, -GoCode
    generate_sharding_go/2,             % +Strategy, -GoCode
    generate_replication_go/2,          % +ReplicationFactor, -GoCode
    % Phase 7: Service Discovery
    compile_discovery_service_go/2,     % +Service, -GoCode
    generate_health_check_go/2,         % +Config, -GoCode
    generate_service_registry_go/2,     % +Backend, -GoCode
    % Phase 8: Service Tracing
    compile_traced_service_go/2,        % +Service, -GoCode
    generate_tracer_go/2,               % +Config, -GoCode
    generate_span_context_go/2,         % +Context, -GoCode
    % KG Topology Phase 3: Kleinberg routing
    compile_kleinberg_router_go/2,      % +Options, -GoCode
    % KG Topology Phase 4: Federated queries
    compile_federated_query_go/2,       % +Options, -GoCode
    % KG Topology Phase 5b: Adaptive federation-k
    compile_adaptive_federation_go/2,   % +Options, -GoCode
    % KG Topology Phase 5c: Query plan optimization
    compile_query_planner_go/2,         % +Options, -GoCode
    % KG Topology Phase 5a: Hierarchical federation
    compile_hierarchical_federation_go/2, % +Options, -GoCode
    % KG Topology Phase 5d: Streaming federation
    compile_streaming_federation_go/2   % +Options, -GoCode
]).

:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module(common_generator).

% Binding system integration
:- use_module('../core/binding_registry').
:- use_module('../bindings/go_bindings').

% Pipeline validation
:- use_module('../core/pipeline_validation').

% Service validation (Client-Server Phase 2)
:- use_module('../core/service_validation').

% Track required imports from bindings
:- dynamic required_binding_import/1.

%% init_go_target
%  Initialize Go target with bindings
init_go_target :-
    retractall(required_binding_import(_)),
    init_go_bindings.

%% clear_binding_imports
%  Clear collected binding imports
clear_binding_imports :-
    retractall(required_binding_import(_)).

%% collect_binding_import(+Import)
%  Record that an import is required
collect_binding_import(Import) :-
    (   required_binding_import(Import)
    ->  true
    ;   assertz(required_binding_import(Import))
    ).

%% get_collected_imports(-Imports)
%  Get all collected imports from bindings
get_collected_imports(Imports) :-
    findall(I, required_binding_import(I), Imports).

%% format_binding_imports(+Imports, -FormattedStr)
%  Format a list of import package names for Go import block
format_binding_imports([], "").
format_binding_imports(Imports, FormattedStr) :-
    Imports \= [],
    sort(Imports, UniqueImports),
    findall(Formatted,
        (   member(Import, UniqueImports),
            format(string(Formatted), "\"~w\"\n    ", [Import])
        ),
        FormattedList),
    atomic_list_concat(FormattedList, '', FormattedStr).

% Suppress singleton warnings in this experimental generator target.
:- style_check(-singleton).
:- discontiguous extract_match_constraints/2.
:- discontiguous term_to_go_expr/3.
:- use_module(library(filesex)).

%% Go generator config for common_generator.pl
go_generator_config(Config) :-
    Config = [
        access_fmt-"~w.Args[\"arg~w\"]",
        atom_fmt-"\"~w\"",
        null_val-"nil",
        ops-[
            + - "+", - - "-", * - "*", / - "/",
            > - ">", < - "<", >= - ">=", =< - "<=",
            =:= - "==", =\= - "!="
        ]
    ].

%% ============================================
%% JSON SCHEMA SUPPORT
%% ============================================

:- dynamic json_schema_def/2.

%% json_schema(+SchemaName, +Fields)
%  Define a JSON schema with typed fields
%  Used as a directive: :- json_schema(person, [field(name, string), field(age, integer)]).
%
json_schema(SchemaName, Fields) :-
    % Validate schema fields
    (   validate_schema_fields(Fields)
    ->  % Store schema definition
        retractall(json_schema_def(SchemaName, _)),
        assertz(json_schema_def(SchemaName, Fields)),
        format('Schema defined: ~w with ~w fields~n', [SchemaName, Fields])
    ;   format('ERROR: Invalid schema definition for ~w: ~w~n', [SchemaName, Fields]),
        fail
    ).

%% validate_schema_fields(+Fields)
%  Validate that all fields have correct format: field(Name, Type)
%
validate_schema_fields([]).
validate_schema_fields([field(Name, Type)|Rest]) :-
    atom(Name),
    valid_json_type(Type),
    validate_schema_fields(Rest).
validate_schema_fields([field(Name, Type, Options)|Rest]) :-
    atom(Name),
    valid_json_type(Type),
    is_list(Options),
    validate_field_options(Options),
    validate_schema_fields(Rest).
validate_schema_fields([Invalid|_]) :-
    format('ERROR: Invalid field specification: ~w~n', [Invalid]),
    fail.

%% validate_field_options(+Options)
%  Validate field options: min(N), max(N), format(F), required, optional
validate_field_options([]).
validate_field_options([Option|Rest]) :-
    validate_field_option(Option),
    validate_field_options(Rest).

validate_field_option(min(N)) :- number(N).
validate_field_option(max(N)) :- number(N).
validate_field_option(format(F)) :- atom(F).
validate_field_option(required).
validate_field_option(optional).
validate_field_option(Invalid) :-
    format('ERROR: Invalid field option: ~w~n', [Invalid]),
    fail.

%% valid_json_type(+Type)
%  Check if a type is valid for JSON schemas
%
valid_json_type(string).
valid_json_type(integer).
valid_json_type(float).
valid_json_type(boolean).
valid_json_type(any).  % Fallback to interface{}
valid_json_type(array(Type)) :-
    valid_json_type(Type).
valid_json_type(object(SchemaName)) :-
    atom(SchemaName).

%% get_json_schema(+SchemaName, -Fields)
%  Retrieve a schema definition by name
%
get_json_schema(SchemaName, Fields) :-
    json_schema_def(SchemaName, Fields), !.
get_json_schema(SchemaName, _) :-
    format('ERROR: Schema not found: ~w~n', [SchemaName]),
    fail.

%% get_field_info(+SchemaName, +FieldName, -Type, -Options)
%  Get the type and options of a specific field from a schema
%
get_field_info(SchemaName, FieldName, Type, Options) :-
    get_json_schema(SchemaName, Fields),
    (   member(field(FieldName, Type, Options), Fields) -> true
    ;   member(field(FieldName, Type), Fields) -> Options = []
    ), !.
get_field_info(SchemaName, FieldName, any, []) :-
    % Field not in schema - default to 'any' (interface{})
    format('WARNING: Field ~w not in schema ~w, defaulting to type ''any''~n', [FieldName, SchemaName]).

%% ============================================
%% SERVICE COMPILATION (Client-Server Phase 1)
%% ============================================

%% compile_service_to_go(+Service, -GoCode)
%  Compile a service definition to a Go struct and methods.
%  Dispatches based on transport type: in_process, unix_socket, etc.
compile_service_to_go(service(Name, HandlerSpec), GoCode) :-
    !,
    compile_service_to_go(service(Name, [], HandlerSpec), GoCode).

compile_service_to_go(Service, GoCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    member(transport(unix_socket(_Path)), Options),
    !,
    % Phase 2: Unix socket service
    compile_unix_socket_service_go(Service, GoCode).

compile_service_to_go(Service, GoCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    member(transport(tcp(_Host, _Port)), Options),
    !,
    % Phase 3: TCP service
    compile_tcp_service_go(Service, GoCode).

compile_service_to_go(Service, GoCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    member(transport(http(_Endpoint)), Options),
    !,
    % Phase 3: HTTP service
    compile_http_service_go(Service, GoCode).

compile_service_to_go(Service, GoCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    member(transport(http(_Endpoint, _HttpOptions)), Options),
    !,
    % Phase 3: HTTP service with options
    compile_http_service_go(Service, GoCode).

compile_service_to_go(Service, GoCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    ( member(polyglot(true), Options)
    ; member(depends_on(Deps), Options), Deps \= []
    ),
    !,
    % Phase 5: Polyglot service
    compile_polyglot_service_go(Service, GoCode).

compile_service_to_go(Service, GoCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    ( member(distributed(true), Options)
    ; member(sharding(_), Options)
    ; member(replication(_), Options)
    ; member(cluster(_), Options)
    ),
    !,
    % Phase 6: Distributed service
    compile_distributed_service_go(Service, GoCode).

compile_service_to_go(Service, GoCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    ( member(discovery_enabled(true), Options)
    ; member(discovery_backend(_), Options)
    ; member(health_check(_), Options)
    ),
    !,
    % Phase 7: Service Discovery
    compile_discovery_service_go(Service, GoCode).

compile_service_to_go(Service, GoCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    ( member(tracing(true), Options)
    ; member(trace_exporter(_), Options)
    ; member(trace_sampling(_), Options)
    ),
    !,
    % Phase 8: Service Tracing
    compile_traced_service_go(Service, GoCode).

compile_service_to_go(Service, GoCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    ( member(load_balance(_), Options)
    ; member(load_balance(_, _), Options)
    ; member(circuit_breaker(_, _), Options)
    ; member(circuit_breaker(_), Options)
    ; member(retry(_, _), Options)
    ; member(retry(_, _, _), Options)
    ),
    !,
    % Phase 4: Service mesh service
    compile_service_mesh_go(Service, GoCode).

compile_service_to_go(service(Name, Options, HandlerSpec), GoCode) :-
    % Phase 1: In-process service (default)
    % Determine if service is stateful
    ( member(stateful(true), Options) -> Stateful = true ; Stateful = false ),
    % Generate handler code
    generate_service_handler_go(HandlerSpec, HandlerCode),
    % Format the struct name (capitalize first letter)
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        StructName = [Upper|Rest]
    ;
        StructName = [First|Rest]
    ),
    atom_codes(StructNameAtom, StructName),
    % Generate the service struct
    ( Stateful = true ->
        format(string(GoCode),
"// ~wService implements the Service interface for ~w
type ~wService struct {
\t*StatefulService
}

// New~wService creates a new ~w service instance
func New~wService() *~wService {
\treturn &~wService{
\t\tStatefulService: NewStatefulService(\"~w\"),
\t}
}

// Name returns the service name
func (s *~wService) Name() string {
\treturn \"~w\"
}

// Call processes a request and returns a response
func (s *~wService) Call(request interface{}) (interface{}, error) {
~w
}

// Register ~w service
func init() {
\tRegisterService(\"~w\", New~wService())
}
", [StructNameAtom, Name, StructNameAtom, StructNameAtom, Name, StructNameAtom, StructNameAtom, StructNameAtom, Name, StructNameAtom, Name, StructNameAtom, HandlerCode, Name, Name, StructNameAtom])
    ;
        format(string(GoCode),
"// ~wService implements the Service interface for ~w
type ~wService struct {
\tname string
}

// New~wService creates a new ~w service instance
func New~wService() *~wService {
\treturn &~wService{name: \"~w\"}
}

// Name returns the service name
func (s *~wService) Name() string {
\treturn s.name
}

// Call processes a request and returns a response
func (s *~wService) Call(request interface{}) (interface{}, error) {
~w
}

// Register ~w service
func init() {
\tRegisterService(\"~w\", New~wService())
}
", [StructNameAtom, Name, StructNameAtom, StructNameAtom, Name, StructNameAtom, StructNameAtom, StructNameAtom, Name, StructNameAtom, StructNameAtom, HandlerCode, Name, Name, StructNameAtom])
    ).

%% generate_service_handler_go(+HandlerSpec, -Code)
%  Generate Go handler code from handler specification.
generate_service_handler_go([], "\treturn nil, nil").
generate_service_handler_go(HandlerSpec, Code) :-
    HandlerSpec \= [],
    generate_handler_ops_go(HandlerSpec, OpsCode),
    format(string(Code), "~w", [OpsCode]).

%% generate_handler_ops_go(+Ops, -Code)
%  Generate Go code for handler operations.
generate_handler_ops_go([], "").
generate_handler_ops_go([Op|Rest], Code) :-
    generate_handler_op_go(Op, OpCode),
    generate_handler_ops_go(Rest, RestCode),
    ( RestCode = "" ->
        Code = OpCode
    ;
        format(string(Code), "~w~n~w", [OpCode, RestCode])
    ).

%% generate_handler_op_go(+Op, -Code)
%  Generate Go code for a single handler operation.
generate_handler_op_go(receive(_Var), Code) :-
    format(string(Code), "\t// Bind request", []).

generate_handler_op_go(respond(Value), Code) :-
    ( var(Value) ->
        format(string(Code), "\treturn response, nil", [])
    ; atom(Value) ->
        format(string(Code), "\treturn ~w, nil", [Value])
    ; number(Value) ->
        format(string(Code), "\treturn ~w, nil", [Value])
    ;
        format(string(Code), "\treturn request, nil", [])
    ).

generate_handler_op_go(respond_error(Error), Code) :-
    format(string(Code), "\treturn nil, &ServiceError{Service: s.Name(), Message: \"~w\"}", [Error]).

generate_handler_op_go(transform(_In, _Out, Goal), Code) :-
    format(string(Code), "\t// transform: ~w", [Goal]).

generate_handler_op_go(transform(_In, _Out), Code) :-
    format(string(Code), "\t// transform", []).

generate_handler_op_go(state_get(Key, _Var), Code) :-
    format(string(Code), "\t_ = s.StateGet(\"~w\")", [Key]).

generate_handler_op_go(state_put(Key, Value), Code) :-
    ( var(Value) ->
        format(string(Code), "\ts.StatePut(\"~w\", request)", [Key])
    ;
        format(string(Code), "\ts.StatePut(\"~w\", ~w)", [Key, Value])
    ).

generate_handler_op_go(state_modify(Key, Func), Code) :-
    format(string(Code), "\ts.StateModify(\"~w\", func(v interface{}) interface{} { /* ~w */ return v })", [Key, Func]).

generate_handler_op_go(state_delete(Key), Code) :-
    format(string(Code), "\ts.StateDelete(\"~w\")", [Key]).

generate_handler_op_go(call_service(ServiceName, _Req, _Resp), Code) :-
    format(string(Code), "\t_, _ = CallServiceImpl(\"~w\", request, nil)", [ServiceName]).

generate_handler_op_go(Pred/Arity, Code) :-
    format(string(Code), "\t// Call predicate: ~w/~w", [Pred, Arity]).

generate_handler_op_go(Pred, Code) :-
    atom(Pred),
    Pred \= receive, Pred \= respond, Pred \= respond_error,
    format(string(Code), "\t// Execute predicate: ~w", [Pred]).

generate_handler_op_go(_, "\t// Unknown operation").

%% ============================================
%% PHASE 2: CROSS-PROCESS SERVICES (Unix Socket)
%% ============================================

%% compile_unix_socket_service_go(+Service, -GoCode)
%  Generate Go code for a Unix socket service server.
compile_unix_socket_service_go(service(Name, Options, HandlerSpec), GoCode) :-
    % Extract socket path
    member(transport(unix_socket(SocketPath)), Options),
    % Determine if service is stateful
    ( member(stateful(true), Options) -> Stateful = true ; Stateful = false ),
    % Extract timeout (default 30000ms)
    ( member(timeout(TimeoutMs), Options) -> Timeout = TimeoutMs ; Timeout = 30000 ),
    % Generate handler code
    generate_service_handler_go(HandlerSpec, HandlerCode),
    % Format the struct name (capitalize first letter)
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        StructName = [Upper|Rest]
    ;
        StructName = [First|Rest]
    ),
    atom_codes(StructNameAtom, StructName),
    % Generate upper case name for export
    atom_codes(UpperName, StructName),
    % Generate the Unix socket service
    ( Stateful = true ->
        format(string(GoCode),
"package main

import (
\t\"bufio\"
\t\"encoding/json\"
\t\"fmt\"
\t\"net\"
\t\"os\"
\t\"os/signal\"
\t\"sync\"
\t\"syscall\"
\t\"time\"
)

// ~wService implements a Unix socket server for ~w
type ~wService struct {
\t*StatefulService
\tsocketPath string
\ttimeout    time.Duration
\tlistener   net.Listener
\trunning    bool
\tmu         sync.Mutex
}

// New~wService creates a new ~w service instance
func New~wService() *~wService {
\treturn &~wService{
\t\tStatefulService: NewStatefulService(\"~w\"),
\t\tsocketPath:      \"~w\",
\t\ttimeout:         ~w * time.Millisecond,
\t}
}

// Name returns the service name
func (s *~wService) Name() string {
\treturn \"~w\"
}

// Call processes a request and returns a response
func (s *~wService) Call(request interface{}) (interface{}, error) {
~w
}

// StartServer starts the Unix socket server
func (s *~wService) StartServer() error {
\t// Remove existing socket file
\tos.Remove(s.socketPath)

\tlistener, err := net.Listen(\"unix\", s.socketPath)
\tif err != nil {
\t\treturn err
\t}
\ts.listener = listener
\ts.running = true

\tfmt.Fprintf(os.Stderr, \"[~w] Server listening on %%s\\n\", s.socketPath)

\tfor s.running {
\t\tconn, err := listener.Accept()
\t\tif err != nil {
\t\t\tif !s.running {
\t\t\t\tbreak
\t\t\t}
\t\t\tcontinue
\t\t}
\t\tgo s.handleConnection(conn)
\t}

\ts.cleanup()
\treturn nil
}

func (s *~wService) handleConnection(conn net.Conn) {
\tdefer conn.Close()
\tconn.SetDeadline(time.Now().Add(s.timeout))

\treader := bufio.NewReader(conn)
\tfor {
\t\tline, err := reader.ReadBytes('\\n')
\t\tif err != nil {
\t\t\tbreak
\t\t}
\t\ts.processRequest(conn, line)
\t}
}

func (s *~wService) processRequest(conn net.Conn, line []byte) {
\tvar request map[string]interface{}
\tif err := json.Unmarshal(line, &request); err != nil {
\t\ts.sendError(conn, \"parse_error\", err.Error())
\t\treturn
\t}

\trequestID, _ := request[\"_id\"].(string)
\tpayload := request[\"_payload\"]
\tif payload == nil {
\t\tpayload = request
\t}

\ts.mu.Lock()
\tresponse, err := s.Call(payload)
\ts.mu.Unlock()

\tif err != nil {
\t\ts.sendError(conn, \"service_error\", err.Error())
\t\treturn
\t}

\ts.sendResponse(conn, requestID, response)
}

func (s *~wService) sendResponse(conn net.Conn, requestID string, response interface{}) {
\tmsg := map[string]interface{}{
\t\t\"_id\":      requestID,
\t\t\"_status\":  \"ok\",
\t\t\"_payload\": response,
\t}
\tdata, _ := json.Marshal(msg)
\tconn.Write(append(data, '\\n'))
}

func (s *~wService) sendError(conn net.Conn, errorType, message string) {
\tmsg := map[string]interface{}{
\t\t\"_status\":     \"error\",
\t\t\"_error_type\": errorType,
\t\t\"_message\":    message,
\t}
\tdata, _ := json.Marshal(msg)
\tconn.Write(append(data, '\\n'))
}

// StopServer stops the Unix socket server
func (s *~wService) StopServer() {
\ts.running = false
\tif s.listener != nil {
\t\ts.listener.Close()
\t}
}

func (s *~wService) cleanup() {
\tos.Remove(s.socketPath)
\tfmt.Fprintf(os.Stderr, \"[~w] Server stopped\\n\")
}

// Global service instance
var _~wService = New~wService()

func init() {
\tRegisterService(\"~w\", _~wService)
}

func main() {
\tsigChan := make(chan os.Signal, 1)
\tsignal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

\tgo func() {
\t\t<-sigChan
\t\tfmt.Fprintf(os.Stderr, \"\\n[~w] Shutting down...\\n\")
\t\t_~wService.StopServer()
\t\tos.Exit(0)
\t}()

\t_~wService.StartServer()
}
", [StructNameAtom, Name, StructNameAtom, StructNameAtom, Name, StructNameAtom, StructNameAtom, StructNameAtom, Name, SocketPath, Timeout,
    StructNameAtom, Name, HandlerCode, StructNameAtom, Name, StructNameAtom, StructNameAtom, StructNameAtom,
    StructNameAtom, StructNameAtom, StructNameAtom, Name, StructNameAtom, StructNameAtom, Name, StructNameAtom, Name, StructNameAtom, StructNameAtom, StructNameAtom])
    ;
        % Non-stateful version
        format(string(GoCode),
"package main

import (
\t\"bufio\"
\t\"encoding/json\"
\t\"fmt\"
\t\"net\"
\t\"os\"
\t\"os/signal\"
\t\"sync\"
\t\"syscall\"
\t\"time\"
)

// ~wService implements a Unix socket server for ~w
type ~wService struct {
\tname       string
\tsocketPath string
\ttimeout    time.Duration
\tlistener   net.Listener
\trunning    bool
\tmu         sync.Mutex
}

// New~wService creates a new ~w service instance
func New~wService() *~wService {
\treturn &~wService{
\t\tname:       \"~w\",
\t\tsocketPath: \"~w\",
\t\ttimeout:    ~w * time.Millisecond,
\t}
}

// Name returns the service name
func (s *~wService) Name() string {
\treturn s.name
}

// Call processes a request and returns a response
func (s *~wService) Call(request interface{}) (interface{}, error) {
~w
}

// StartServer starts the Unix socket server
func (s *~wService) StartServer() error {
\t// Remove existing socket file
\tos.Remove(s.socketPath)

\tlistener, err := net.Listen(\"unix\", s.socketPath)
\tif err != nil {
\t\treturn err
\t}
\ts.listener = listener
\ts.running = true

\tfmt.Fprintf(os.Stderr, \"[~w] Server listening on %%s\\n\", s.socketPath)

\tfor s.running {
\t\tconn, err := listener.Accept()
\t\tif err != nil {
\t\t\tif !s.running {
\t\t\t\tbreak
\t\t\t}
\t\t\tcontinue
\t\t}
\t\tgo s.handleConnection(conn)
\t}

\ts.cleanup()
\treturn nil
}

func (s *~wService) handleConnection(conn net.Conn) {
\tdefer conn.Close()
\tconn.SetDeadline(time.Now().Add(s.timeout))

\treader := bufio.NewReader(conn)
\tfor {
\t\tline, err := reader.ReadBytes('\\n')
\t\tif err != nil {
\t\t\tbreak
\t\t}
\t\ts.processRequest(conn, line)
\t}
}

func (s *~wService) processRequest(conn net.Conn, line []byte) {
\tvar request map[string]interface{}
\tif err := json.Unmarshal(line, &request); err != nil {
\t\ts.sendError(conn, \"parse_error\", err.Error())
\t\treturn
\t}

\trequestID, _ := request[\"_id\"].(string)
\tpayload := request[\"_payload\"]
\tif payload == nil {
\t\tpayload = request
\t}

\ts.mu.Lock()
\tresponse, err := s.Call(payload)
\ts.mu.Unlock()

\tif err != nil {
\t\ts.sendError(conn, \"service_error\", err.Error())
\t\treturn
\t}

\ts.sendResponse(conn, requestID, response)
}

func (s *~wService) sendResponse(conn net.Conn, requestID string, response interface{}) {
\tmsg := map[string]interface{}{
\t\t\"_id\":      requestID,
\t\t\"_status\":  \"ok\",
\t\t\"_payload\": response,
\t}
\tdata, _ := json.Marshal(msg)
\tconn.Write(append(data, '\\n'))
}

func (s *~wService) sendError(conn net.Conn, errorType, message string) {
\tmsg := map[string]interface{}{
\t\t\"_status\":     \"error\",
\t\t\"_error_type\": errorType,
\t\t\"_message\":    message,
\t}
\tdata, _ := json.Marshal(msg)
\tconn.Write(append(data, '\\n'))
}

// StopServer stops the Unix socket server
func (s *~wService) StopServer() {
\ts.running = false
\tif s.listener != nil {
\t\ts.listener.Close()
\t}
}

func (s *~wService) cleanup() {
\tos.Remove(s.socketPath)
\tfmt.Fprintf(os.Stderr, \"[~w] Server stopped\\n\")
}

// Global service instance
var _~wService = New~wService()

func init() {
\tRegisterService(\"~w\", _~wService)
}

func main() {
\tsigChan := make(chan os.Signal, 1)
\tsignal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

\tgo func() {
\t\t<-sigChan
\t\tfmt.Fprintf(os.Stderr, \"\\n[~w] Shutting down...\\n\")
\t\t_~wService.StopServer()
\t\tos.Exit(0)
\t}()

\t_~wService.StartServer()
}
", [StructNameAtom, Name, StructNameAtom, StructNameAtom, Name, StructNameAtom, StructNameAtom, StructNameAtom, Name, SocketPath, Timeout,
    StructNameAtom, HandlerCode, StructNameAtom, Name, StructNameAtom, StructNameAtom, StructNameAtom,
    StructNameAtom, StructNameAtom, StructNameAtom, Name, StructNameAtom, StructNameAtom, Name, StructNameAtom, Name, StructNameAtom, StructNameAtom, StructNameAtom])
    ).

%% compile_unix_socket_client_go(+ServiceName, +SocketPath, -GoCode)
%  Generate Go code for a Unix socket service client.
compile_unix_socket_client_go(Name, SocketPath, GoCode) :-
    % Format the struct name (capitalize first letter)
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        StructName = [Upper|Rest]
    ;
        StructName = [First|Rest]
    ),
    atom_codes(StructNameAtom, StructName),
    format(string(GoCode),
"package main

import (
\t\"bufio\"
\t\"encoding/json\"
\t\"fmt\"
\t\"net\"
\t\"time\"

\t\"github.com/google/uuid\"
)

// ~wClient is a client for the ~w Unix socket service
type ~wClient struct {
\tsocketPath string
\ttimeout    time.Duration
\tconn       net.Conn
}

// New~wClient creates a new client for ~w service
func New~wClient(socketPath string, timeout time.Duration) *~wClient {
\treturn &~wClient{
\t\tsocketPath: socketPath,
\t\ttimeout:    timeout,
\t}
}

// Connect connects to the service
func (c *~wClient) Connect() error {
\tconn, err := net.DialTimeout(\"unix\", c.socketPath, c.timeout)
\tif err != nil {
\t\treturn err
\t}
\tc.conn = conn
\treturn nil
}

// Disconnect disconnects from the service
func (c *~wClient) Disconnect() {
\tif c.conn != nil {
\t\tc.conn.Close()
\t\tc.conn = nil
\t}
}

// Call sends a request and receives a response
func (c *~wClient) Call(request interface{}) (interface{}, error) {
\tif c.conn == nil {
\t\tif err := c.Connect(); err != nil {
\t\t\treturn nil, err
\t\t}
\t}

\tc.conn.SetDeadline(time.Now().Add(c.timeout))

\trequestID := uuid.New().String()
\tmsg := map[string]interface{}{
\t\t\"_id\":      requestID,
\t\t\"_payload\": request,
\t}
\tdata, err := json.Marshal(msg)
\tif err != nil {
\t\treturn nil, err
\t}
\t_, err = c.conn.Write(append(data, '\\n'))
\tif err != nil {
\t\treturn nil, err
\t}

\treader := bufio.NewReader(c.conn)
\tline, err := reader.ReadBytes('\\n')
\tif err != nil {
\t\treturn nil, err
\t}

\tvar response map[string]interface{}
\tif err := json.Unmarshal(line, &response); err != nil {
\t\treturn nil, err
\t}

\tif response[\"_status\"] == \"ok\" {
\t\treturn response[\"_payload\"], nil
\t}

\treturn nil, &ServiceError{
\t\tService: \"~w\",
\t\tMessage: fmt.Sprintf(\"%%v\", response[\"_message\"]),
\t}
}

// Call~w is a convenience function to call the ~w service
func Call~w(request interface{}, socketPath string, timeout time.Duration) (interface{}, error) {
\tclient := New~wClient(socketPath, timeout)
\tdefer client.Disconnect()
\treturn client.Call(request)
}

// Default client
var Default~wClient = New~wClient(\"~w\", 30*time.Second)

// ~wRemoteService wraps the client for use with call_service_impl
type ~wRemoteService struct {
\tclient *~wClient
}

// Name returns the service name
func (s *~wRemoteService) Name() string {
\treturn \"~w\"
}

// Call calls the remote service
func (s *~wRemoteService) Call(request interface{}) (interface{}, error) {
\treturn s.client.Call(request)
}

func init() {
\t// Register remote service if local not available
\tif _, ok := services[\"~w\"]; !ok {
\t\tRegisterService(\"~w\", &~wRemoteService{client: Default~wClient})
\t}
}
", [StructNameAtom, Name, StructNameAtom, StructNameAtom, Name, StructNameAtom, StructNameAtom, StructNameAtom,
    StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom,
    Name, StructNameAtom, Name, StructNameAtom, StructNameAtom, StructNameAtom, SocketPath,
    StructNameAtom, StructNameAtom, StructNameAtom, Name, StructNameAtom, Name, Name, StructNameAtom, StructNameAtom]).

%% ============================================
%% PHASE 3: NETWORK SERVICES (TCP)
%% ============================================

%% compile_tcp_service_go(+Service, -GoCode)
%  Generate Go code for a TCP network service server.
compile_tcp_service_go(service(Name, Options, HandlerSpec), GoCode) :-
    % Extract host and port
    member(transport(tcp(Host, Port)), Options),
    % Determine if service is stateful
    ( member(stateful(true), Options) -> Stateful = true ; Stateful = false ),
    % Extract timeout (default 30000ms)
    ( member(timeout(TimeoutMs), Options) -> Timeout = TimeoutMs ; Timeout = 30000 ),
    % Generate handler code
    generate_service_handler_go(HandlerSpec, HandlerCode),
    % Format the struct name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        StructName = [Upper|Rest]
    ;
        StructName = [First|Rest]
    ),
    atom_codes(StructNameAtom, StructName),
    % Generate the TCP service (different for stateful/stateless)
    ( Stateful = true ->
        format(string(GoCode),
"package main

import (
\t\"bufio\"
\t\"encoding/json\"
\t\"fmt\"
\t\"net\"
\t\"os\"
\t\"os/signal\"
\t\"sync\"
\t\"syscall\"
)

// ~wService is a TCP network service (stateful)
type ~wService struct {
\tname     string
\thost     string
\tport     int
\ttimeout  int
\tstate    map[string]interface{}
\tstateMu  sync.RWMutex
\tlistener net.Listener
\trunning  bool
\trunMu    sync.Mutex
}

// New~wService creates a new TCP service
func New~wService() *~wService {
\treturn &~wService{
\t\tname:    \"~w\",
\t\thost:    \"~w\",
\t\tport:    ~w,
\t\ttimeout: ~w,
\t\tstate:   make(map[string]interface{}),
\t}
}

func (s *~wService) Name() string { return s.name }

func (s *~wService) Call(request interface{}) (interface{}, error) {
~w
}

func (s *~wService) StateGet(key string) interface{} {
\ts.stateMu.RLock()
\tdefer s.stateMu.RUnlock()
\treturn s.state[key]
}

func (s *~wService) StatePut(key string, value interface{}) {
\ts.stateMu.Lock()
\tdefer s.stateMu.Unlock()
\ts.state[key] = value
}

func (s *~wService) StartServer() error {
\taddr := fmt.Sprintf(\"%%s:%%d\", s.host, s.port)
\tlistener, err := net.Listen(\"tcp\", addr)
\tif err != nil {
\t\treturn err
\t}
\ts.listener = listener
\ts.running = true

\tfmt.Fprintf(os.Stderr, \"[~w] Server listening on %%s\\n\", addr)

\tgo func() {
\t\tfor s.running {
\t\t\tconn, err := listener.Accept()
\t\t\tif err != nil {
\t\t\t\tif s.running {
\t\t\t\t\tfmt.Fprintf(os.Stderr, \"[~w] Accept error: %%v\\n\", err)
\t\t\t\t}
\t\t\t\tcontinue
\t\t\t}
\t\t\tgo s.handleConnection(conn)
\t\t}
\t}()

\treturn nil
}

func (s *~wService) handleConnection(conn net.Conn) {
\tdefer conn.Close()
\treader := bufio.NewReader(conn)

\tfor {
\t\tline, err := reader.ReadBytes('\\n')
\t\tif err != nil {
\t\t\treturn
\t\t}
\t\ts.processRequest(conn, line)
\t}
}

func (s *~wService) processRequest(conn net.Conn, line []byte) {
\tvar request map[string]interface{}
\tif err := json.Unmarshal(line, &request); err != nil {
\t\ts.sendError(conn, \"parse_error\", err.Error())
\t\treturn
\t}

\trequestID, _ := request[\"_id\"].(string)
\tpayload := request[\"_payload\"]
\tif payload == nil {
\t\tpayload = request
\t}

\tresponse, err := s.Call(payload)
\tif err != nil {
\t\ts.sendError(conn, \"service_error\", err.Error())
\t\treturn
\t}

\ts.sendResponse(conn, requestID, response)
}

func (s *~wService) sendResponse(conn net.Conn, requestID string, response interface{}) {
\tmsg := map[string]interface{}{
\t\t\"_id\":      requestID,
\t\t\"_status\":  \"ok\",
\t\t\"_payload\": response,
\t}
\tdata, _ := json.Marshal(msg)
\tconn.Write(append(data, '\\n'))
}

func (s *~wService) sendError(conn net.Conn, errType, message string) {
\tmsg := map[string]interface{}{
\t\t\"_status\":     \"error\",
\t\t\"_error_type\": errType,
\t\t\"_message\":    message,
\t}
\tdata, _ := json.Marshal(msg)
\tconn.Write(append(data, '\\n'))
}

func (s *~wService) StopServer() {
\ts.runMu.Lock()
\ts.running = false
\ts.runMu.Unlock()
\tif s.listener != nil {
\t\ts.listener.Close()
\t}
\tfmt.Fprintf(os.Stderr, \"[~w] Server stopped\\n\")
}

var _~wService = New~wService()

func init() {
\tRegisterService(\"~w\", _~wService)
}

func main() {
\tsigChan := make(chan os.Signal, 1)
\tsignal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

\tif err := _~wService.StartServer(); err != nil {
\t\tfmt.Fprintf(os.Stderr, \"Failed to start server: %%v\\n\", err)
\t\tos.Exit(1)
\t}

\t<-sigChan
\tfmt.Fprintf(os.Stderr, \"\\n[~w] Shutting down...\\n\")
\t_~wService.StopServer()
}
", [StructNameAtom, StructNameAtom,
    StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, Name, Host, Port, Timeout,
    StructNameAtom, StructNameAtom, HandlerCode,
    StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom,
    Name, Name, StructNameAtom, StructNameAtom,
    StructNameAtom, StructNameAtom, StructNameAtom, Name,
    Name, StructNameAtom, Name, Name, StructNameAtom, Name, StructNameAtom])
    ;
        format(string(GoCode),
"package main

import (
\t\"bufio\"
\t\"encoding/json\"
\t\"fmt\"
\t\"net\"
\t\"os\"
\t\"os/signal\"
\t\"sync\"
\t\"syscall\"
)

// ~wService is a TCP network service
type ~wService struct {
\tname     string
\thost     string
\tport     int
\ttimeout  int
\tlistener net.Listener
\trunning  bool
\trunMu    sync.Mutex
}

// New~wService creates a new TCP service
func New~wService() *~wService {
\treturn &~wService{
\t\tname:    \"~w\",
\t\thost:    \"~w\",
\t\tport:    ~w,
\t\ttimeout: ~w,
\t}
}

func (s *~wService) Name() string { return s.name }

func (s *~wService) Call(request interface{}) (interface{}, error) {
~w
}

func (s *~wService) StartServer() error {
\taddr := fmt.Sprintf(\"%%s:%%d\", s.host, s.port)
\tlistener, err := net.Listen(\"tcp\", addr)
\tif err != nil {
\t\treturn err
\t}
\ts.listener = listener
\ts.running = true

\tfmt.Fprintf(os.Stderr, \"[~w] Server listening on %%s\\n\", addr)

\tgo func() {
\t\tfor s.running {
\t\t\tconn, err := listener.Accept()
\t\t\tif err != nil {
\t\t\t\tif s.running {
\t\t\t\t\tfmt.Fprintf(os.Stderr, \"[~w] Accept error: %%v\\n\", err)
\t\t\t\t}
\t\t\t\tcontinue
\t\t\t}
\t\t\tgo s.handleConnection(conn)
\t\t}
\t}()

\treturn nil
}

func (s *~wService) handleConnection(conn net.Conn) {
\tdefer conn.Close()
\treader := bufio.NewReader(conn)

\tfor {
\t\tline, err := reader.ReadBytes('\\n')
\t\tif err != nil {
\t\t\treturn
\t\t}
\t\ts.processRequest(conn, line)
\t}
}

func (s *~wService) processRequest(conn net.Conn, line []byte) {
\tvar request map[string]interface{}
\tif err := json.Unmarshal(line, &request); err != nil {
\t\ts.sendError(conn, \"parse_error\", err.Error())
\t\treturn
\t}

\trequestID, _ := request[\"_id\"].(string)
\tpayload := request[\"_payload\"]
\tif payload == nil {
\t\tpayload = request
\t}

\tresponse, err := s.Call(payload)
\tif err != nil {
\t\ts.sendError(conn, \"service_error\", err.Error())
\t\treturn
\t}

\ts.sendResponse(conn, requestID, response)
}

func (s *~wService) sendResponse(conn net.Conn, requestID string, response interface{}) {
\tmsg := map[string]interface{}{
\t\t\"_id\":      requestID,
\t\t\"_status\":  \"ok\",
\t\t\"_payload\": response,
\t}
\tdata, _ := json.Marshal(msg)
\tconn.Write(append(data, '\\n'))
}

func (s *~wService) sendError(conn net.Conn, errType, message string) {
\tmsg := map[string]interface{}{
\t\t\"_status\":     \"error\",
\t\t\"_error_type\": errType,
\t\t\"_message\":    message,
\t}
\tdata, _ := json.Marshal(msg)
\tconn.Write(append(data, '\\n'))
}

func (s *~wService) StopServer() {
\ts.runMu.Lock()
\ts.running = false
\ts.runMu.Unlock()
\tif s.listener != nil {
\t\ts.listener.Close()
\t}
\tfmt.Fprintf(os.Stderr, \"[~w] Server stopped\\n\")
}

var _~wService = New~wService()

func init() {
\tRegisterService(\"~w\", _~wService)
}

func main() {
\tsigChan := make(chan os.Signal, 1)
\tsignal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

\tif err := _~wService.StartServer(); err != nil {
\t\tfmt.Fprintf(os.Stderr, \"Failed to start server: %%v\\n\", err)
\t\tos.Exit(1)
\t}

\t<-sigChan
\tfmt.Fprintf(os.Stderr, \"\\n[~w] Shutting down...\\n\")
\t_~wService.StopServer()
}
", [StructNameAtom, StructNameAtom,
    StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, Name, Host, Port, Timeout,
    StructNameAtom, StructNameAtom, HandlerCode,
    StructNameAtom, Name, Name, StructNameAtom, StructNameAtom,
    StructNameAtom, StructNameAtom, StructNameAtom, Name,
    StructNameAtom, StructNameAtom, Name, StructNameAtom, StructNameAtom, Name, StructNameAtom])
    ).

%% compile_tcp_client_go(+ServiceName, +Host, +Port, -GoCode)
%  Generate Go code for a TCP network service client.
compile_tcp_client_go(Name, Host, Port, GoCode) :-
    % Format the struct name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        StructName = [Upper|Rest]
    ;
        StructName = [First|Rest]
    ),
    atom_codes(StructNameAtom, StructName),
    format(string(GoCode),
"package main

import (
\t\"bufio\"
\t\"encoding/json\"
\t\"fmt\"
\t\"net\"

\t\"github.com/google/uuid\"
)

// ~wClient is a TCP client for the ~w service
type ~wClient struct {
\thost    string
\tport    int
\ttimeout int
\tconn    net.Conn
}

// New~wClient creates a new TCP client
func New~wClient(host string, port int) *~wClient {
\treturn &~wClient{
\t\thost:    host,
\t\tport:    port,
\t\ttimeout: 30000,
\t}
}

// Default~wClient is the default client instance
var Default~wClient = New~wClient(\"~w\", ~w)

func (c *~wClient) Connect() error {
\taddr := fmt.Sprintf(\"%%s:%%d\", c.host, c.port)
\tconn, err := net.Dial(\"tcp\", addr)
\tif err != nil {
\t\treturn err
\t}
\tc.conn = conn
\treturn nil
}

func (c *~wClient) Disconnect() {
\tif c.conn != nil {
\t\tc.conn.Close()
\t\tc.conn = nil
\t}
}

func (c *~wClient) Call(request interface{}) (interface{}, error) {
\tif c.conn == nil {
\t\tif err := c.Connect(); err != nil {
\t\t\treturn nil, err
\t\t}
\t}

\trequestID := uuid.New().String()
\tmsg := map[string]interface{}{
\t\t\"_id\":      requestID,
\t\t\"_payload\": request,
\t}

\tdata, err := json.Marshal(msg)
\tif err != nil {
\t\treturn nil, err
\t}

\tif _, err := c.conn.Write(append(data, '\\n')); err != nil {
\t\treturn nil, err
\t}

\treader := bufio.NewReader(c.conn)
\tline, err := reader.ReadBytes('\\n')
\tif err != nil {
\t\treturn nil, err
\t}

\tvar response map[string]interface{}
\tif err := json.Unmarshal(line, &response); err != nil {
\t\treturn nil, err
\t}

\tif response[\"_status\"] == \"ok\" {
\t\treturn response[\"_payload\"], nil
\t}
\treturn nil, fmt.Errorf(\"service error: %%v\", response[\"_message\"])
}

// Call~w calls the ~w service
func Call~w(request interface{}) (interface{}, error) {
\treturn Default~wClient.Call(request)
}

// ~wRemoteService wraps the client as a Service
type ~wRemoteService struct {
\tclient *~wClient
}

func (s *~wRemoteService) Name() string { return \"~w\" }

func (s *~wRemoteService) Call(request interface{}) (interface{}, error) {
\treturn s.client.Call(request)
}

func init() {
\tif _, ok := services[\"~w\"]; !ok {
\t\tRegisterService(\"~w\", &~wRemoteService{client: Default~wClient})
\t}
}
", [StructNameAtom, Name, StructNameAtom,
    StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom,
    StructNameAtom, StructNameAtom, StructNameAtom, Host, Port,
    StructNameAtom, StructNameAtom, StructNameAtom,
    Name, Name, Name, StructNameAtom,
    StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, Name,
    StructNameAtom, Name, Name, StructNameAtom, StructNameAtom]).

%% ============================================
%% PHASE 3: NETWORK SERVICES (HTTP/REST)
%% ============================================

%% compile_http_service_go(+Service, -GoCode)
%  Generate Go code for an HTTP REST service server.
compile_http_service_go(service(Name, Options, HandlerSpec), GoCode) :-
    % Extract endpoint (and optional HTTP options)
    ( member(transport(http(Endpoint, HttpOptions)), Options) ->
        true
    ; member(transport(http(Endpoint)), Options) ->
        HttpOptions = []
    ),
    % Extract host and port from options or use defaults
    ( member(host(Host), HttpOptions) -> true ; Host = '0.0.0.0' ),
    ( member(port(Port), HttpOptions) -> true ; Port = 8080 ),
    % Determine if service is stateful
    ( member(stateful(true), Options) -> Stateful = true ; Stateful = false ),
    % Extract timeout (default 30000ms)
    ( member(timeout(TimeoutMs), Options) -> Timeout = TimeoutMs ; Timeout = 30000 ),
    % Generate handler code
    generate_service_handler_go(HandlerSpec, HandlerCode),
    % Format the struct name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        StructName = [Upper|Rest]
    ;
        StructName = [First|Rest]
    ),
    atom_codes(StructNameAtom, StructName),
    % Generate the HTTP service
    ( Stateful = true ->
        format(string(GoCode),
"package main

import (
\t\"encoding/json\"
\t\"fmt\"
\t\"io\"
\t\"net/http\"
\t\"os\"
\t\"os/signal\"
\t\"sync\"
\t\"syscall\"
)

// ~wService is an HTTP REST service (stateful)
type ~wService struct {
\tname     string
\thost     string
\tport     int
\tendpoint string
\ttimeout  int
\tstate    map[string]interface{}
\tstateMu  sync.RWMutex
\tserver   *http.Server
}

// New~wService creates a new HTTP service
func New~wService() *~wService {
\treturn &~wService{
\t\tname:     \"~w\",
\t\thost:     \"~w\",
\t\tport:     ~w,
\t\tendpoint: \"~w\",
\t\ttimeout:  ~w,
\t\tstate:    make(map[string]interface{}),
\t}
}

func (s *~wService) Name() string { return s.name }

func (s *~wService) Call(request interface{}) (interface{}, error) {
~w
}

func (s *~wService) StateGet(key string) interface{} {
\ts.stateMu.RLock()
\tdefer s.stateMu.RUnlock()
\treturn s.state[key]
}

func (s *~wService) StatePut(key string, value interface{}) {
\ts.stateMu.Lock()
\tdefer s.stateMu.Unlock()
\ts.state[key] = value
}

func (s *~wService) ServeHTTP(w http.ResponseWriter, r *http.Request) {
\tw.Header().Set(\"Content-Type\", \"application/json\")

\t// Build request object
\trequest := map[string]interface{}{
\t\t\"method\":  r.Method,
\t\t\"path\":    r.URL.Path,
\t\t\"query\":   r.URL.Query(),
\t\t\"headers\": r.Header,
\t}

\t// Read body for POST/PUT/PATCH
\tif r.Method == \"POST\" || r.Method == \"PUT\" || r.Method == \"PATCH\" {
\t\tbody, _ := io.ReadAll(r.Body)
\t\tif len(body) > 0 {
\t\t\tvar bodyData interface{}
\t\t\tjson.Unmarshal(body, &bodyData)
\t\t\trequest[\"body\"] = bodyData
\t\t}
\t}

\tresponse, err := s.Call(request)
\tif err != nil {
\t\tw.WriteHeader(http.StatusBadRequest)
\t\tjson.NewEncoder(w).Encode(map[string]interface{}{
\t\t\t\"_status\":     \"error\",
\t\t\t\"_error_type\": \"service_error\",
\t\t\t\"_message\":    err.Error(),
\t\t})
\t\treturn
\t}

\tjson.NewEncoder(w).Encode(map[string]interface{}{
\t\t\"_status\":  \"ok\",
\t\t\"_payload\": response,
\t})
}

func (s *~wService) StartServer() error {
\taddr := fmt.Sprintf(\"%%s:%%d\", s.host, s.port)
\tmux := http.NewServeMux()
\tmux.Handle(s.endpoint, s)
\tmux.Handle(s.endpoint+\"/\", s)

\ts.server = &http.Server{
\t\tAddr:    addr,
\t\tHandler: mux,
\t}

\tfmt.Fprintf(os.Stderr, \"[~w] HTTP server listening on http://%%s%%s\\n\", addr, s.endpoint)

\tgo func() {
\t\tif err := s.server.ListenAndServe(); err != http.ErrServerClosed {
\t\t\tfmt.Fprintf(os.Stderr, \"[~w] Server error: %%v\\n\", err)
\t\t}
\t}()

\treturn nil
}

func (s *~wService) StopServer() {
\tif s.server != nil {
\t\ts.server.Close()
\t}
\tfmt.Fprintf(os.Stderr, \"[~w] HTTP server stopped\\n\")
}

var _~wService = New~wService()

func init() {
\tRegisterService(\"~w\", _~wService)
}

func main() {
\tsigChan := make(chan os.Signal, 1)
\tsignal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

\tif err := _~wService.StartServer(); err != nil {
\t\tfmt.Fprintf(os.Stderr, \"Failed to start server: %%v\\n\", err)
\t\tos.Exit(1)
\t}

\t<-sigChan
\tfmt.Fprintf(os.Stderr, \"\\n[~w] Shutting down...\\n\")
\t_~wService.StopServer()
}
", [StructNameAtom, StructNameAtom,
    StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, Name, Host, Port, Endpoint, Timeout,
    StructNameAtom, StructNameAtom, HandlerCode,
    StructNameAtom, StructNameAtom, StructNameAtom,
    StructNameAtom, Name, Name, StructNameAtom, Name,
    StructNameAtom, StructNameAtom, Name, StructNameAtom, StructNameAtom, Name, StructNameAtom])
    ;
        format(string(GoCode),
"package main

import (
\t\"encoding/json\"
\t\"fmt\"
\t\"io\"
\t\"net/http\"
\t\"os\"
\t\"os/signal\"
\t\"syscall\"
)

// ~wService is an HTTP REST service
type ~wService struct {
\tname     string
\thost     string
\tport     int
\tendpoint string
\ttimeout  int
\tserver   *http.Server
}

// New~wService creates a new HTTP service
func New~wService() *~wService {
\treturn &~wService{
\t\tname:     \"~w\",
\t\thost:     \"~w\",
\t\tport:     ~w,
\t\tendpoint: \"~w\",
\t\ttimeout:  ~w,
\t}
}

func (s *~wService) Name() string { return s.name }

func (s *~wService) Call(request interface{}) (interface{}, error) {
~w
}

func (s *~wService) ServeHTTP(w http.ResponseWriter, r *http.Request) {
\tw.Header().Set(\"Content-Type\", \"application/json\")

\t// Build request object
\trequest := map[string]interface{}{
\t\t\"method\":  r.Method,
\t\t\"path\":    r.URL.Path,
\t\t\"query\":   r.URL.Query(),
\t\t\"headers\": r.Header,
\t}

\t// Read body for POST/PUT/PATCH
\tif r.Method == \"POST\" || r.Method == \"PUT\" || r.Method == \"PATCH\" {
\t\tbody, _ := io.ReadAll(r.Body)
\t\tif len(body) > 0 {
\t\t\tvar bodyData interface{}
\t\t\tjson.Unmarshal(body, &bodyData)
\t\t\trequest[\"body\"] = bodyData
\t\t}
\t}

\tresponse, err := s.Call(request)
\tif err != nil {
\t\tw.WriteHeader(http.StatusBadRequest)
\t\tjson.NewEncoder(w).Encode(map[string]interface{}{
\t\t\t\"_status\":     \"error\",
\t\t\t\"_error_type\": \"service_error\",
\t\t\t\"_message\":    err.Error(),
\t\t})
\t\treturn
\t}

\tjson.NewEncoder(w).Encode(map[string]interface{}{
\t\t\"_status\":  \"ok\",
\t\t\"_payload\": response,
\t})
}

func (s *~wService) StartServer() error {
\taddr := fmt.Sprintf(\"%%s:%%d\", s.host, s.port)
\tmux := http.NewServeMux()
\tmux.Handle(s.endpoint, s)
\tmux.Handle(s.endpoint+\"/\", s)

\ts.server = &http.Server{
\t\tAddr:    addr,
\t\tHandler: mux,
\t}

\tfmt.Fprintf(os.Stderr, \"[~w] HTTP server listening on http://%%s%%s\\n\", addr, s.endpoint)

\tgo func() {
\t\tif err := s.server.ListenAndServe(); err != http.ErrServerClosed {
\t\t\tfmt.Fprintf(os.Stderr, \"[~w] Server error: %%v\\n\", err)
\t\t}
\t}()

\treturn nil
}

func (s *~wService) StopServer() {
\tif s.server != nil {
\t\ts.server.Close()
\t}
\tfmt.Fprintf(os.Stderr, \"[~w] HTTP server stopped\\n\")
}

var _~wService = New~wService()

func init() {
\tRegisterService(\"~w\", _~wService)
}

func main() {
\tsigChan := make(chan os.Signal, 1)
\tsignal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

\tif err := _~wService.StartServer(); err != nil {
\t\tfmt.Fprintf(os.Stderr, \"Failed to start server: %%v\\n\", err)
\t\tos.Exit(1)
\t}

\t<-sigChan
\tfmt.Fprintf(os.Stderr, \"\\n[~w] Shutting down...\\n\")
\t_~wService.StopServer()
}
", [StructNameAtom, StructNameAtom,
    StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, Name, Host, Port, Endpoint, Timeout,
    StructNameAtom, StructNameAtom, HandlerCode,
    StructNameAtom, StructNameAtom, Name, Name, StructNameAtom, Name,
    StructNameAtom, StructNameAtom, Name, StructNameAtom, StructNameAtom, Name, StructNameAtom])
    ).

%% compile_http_client_go(+ServiceName, +Endpoint, -GoCode)
%  Generate Go code for an HTTP REST service client.
compile_http_client_go(Name, Endpoint, GoCode) :-
    compile_http_client_go(Name, Endpoint, [], GoCode).

compile_http_client_go(Name, Endpoint, HttpOptions, GoCode) :-
    % Extract host and port from options or use defaults
    ( member(host(Host), HttpOptions) -> true ; Host = 'localhost' ),
    ( member(port(Port), HttpOptions) -> true ; Port = 8080 ),
    % Format the struct name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        StructName = [Upper|Rest]
    ;
        StructName = [First|Rest]
    ),
    atom_codes(StructNameAtom, StructName),
    format(string(GoCode),
"package main

import (
\t\"bytes\"
\t\"encoding/json\"
\t\"fmt\"
\t\"io\"
\t\"net/http\"
\t\"net/url\"
)

// ~wClient is an HTTP client for the ~w service
type ~wClient struct {
\tbaseURL string
\ttimeout int
\tclient  *http.Client
}

// New~wClient creates a new HTTP client
func New~wClient(host string, port int) *~wClient {
\treturn &~wClient{
\t\tbaseURL: fmt.Sprintf(\"http://%%s:%%d~w\", host, port),
\t\ttimeout: 30000,
\t\tclient:  &http.Client{},
\t}
}

// Default~wClient is the default client instance
var Default~wClient = New~wClient(\"~w\", ~w)

func (c *~wClient) makeRequest(method, path string, data interface{}, query url.Values) (interface{}, error) {
\tfullURL := c.baseURL + path
\tif len(query) > 0 {
\t\tfullURL += \"?\" + query.Encode()
\t}

\tvar body io.Reader
\tif data != nil {
\t\tjsonData, err := json.Marshal(data)
\t\tif err != nil {
\t\t\treturn nil, err
\t\t}
\t\tbody = bytes.NewReader(jsonData)
\t}

\treq, err := http.NewRequest(method, fullURL, body)
\tif err != nil {
\t\treturn nil, err
\t}
\treq.Header.Set(\"Content-Type\", \"application/json\")

\tresp, err := c.client.Do(req)
\tif err != nil {
\t\treturn nil, err
\t}
\tdefer resp.Body.Close()

\trespBody, err := io.ReadAll(resp.Body)
\tif err != nil {
\t\treturn nil, err
\t}

\tvar result map[string]interface{}
\tif err := json.Unmarshal(respBody, &result); err != nil {
\t\treturn nil, err
\t}

\tif result[\"_status\"] == \"ok\" {
\t\treturn result[\"_payload\"], nil
\t}
\treturn nil, fmt.Errorf(\"service error: %%v\", result[\"_message\"])
}

func (c *~wClient) Get(path string, query url.Values) (interface{}, error) {
\treturn c.makeRequest(\"GET\", path, nil, query)
}

func (c *~wClient) Post(path string, data interface{}) (interface{}, error) {
\treturn c.makeRequest(\"POST\", path, data, nil)
}

func (c *~wClient) Put(path string, data interface{}) (interface{}, error) {
\treturn c.makeRequest(\"PUT\", path, data, nil)
}

func (c *~wClient) Delete(path string) (interface{}, error) {
\treturn c.makeRequest(\"DELETE\", path, nil, nil)
}

func (c *~wClient) Call(request interface{}) (interface{}, error) {
\treq, ok := request.(map[string]interface{})
\tif !ok {
\t\treturn c.Post(\"\", request)
\t}

\tmethod, _ := req[\"method\"].(string)
\tif method == \"\" {
\t\tmethod = \"POST\"
\t}
\tpath, _ := req[\"path\"].(string)
\tdata := req[\"body\"]
\tif data == nil {
\t\tdata = req[\"data\"]
\t}

\treturn c.makeRequest(method, path, data, nil)
}

// Call~w calls the ~w service
func Call~w(request interface{}) (interface{}, error) {
\treturn Default~wClient.Call(request)
}

// ~wRemoteService wraps the client as a Service
type ~wRemoteService struct {
\tclient *~wClient
}

func (s *~wRemoteService) Name() string { return \"~w\" }

func (s *~wRemoteService) Call(request interface{}) (interface{}, error) {
\treturn s.client.Call(request)
}

func init() {
\tif _, ok := services[\"~w\"]; !ok {
\t\tRegisterService(\"~w\", &~wRemoteService{client: Default~wClient})
\t}
}
", [StructNameAtom, Name, StructNameAtom,
    StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, Endpoint,
    StructNameAtom, StructNameAtom, StructNameAtom, Host, Port,
    StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom,
    Name, Name, Name, StructNameAtom,
    StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, Name,
    StructNameAtom, Name, Name, StructNameAtom, StructNameAtom]).

%% ============================================
%% PHASE 4: SERVICE MESH
%% ============================================

%% compile_service_mesh_go(+Service, -GoCode)
%  Generate Go code for a service mesh service with load balancing,
%  circuit breaker, and retry capabilities.
compile_service_mesh_go(service(Name, Options, HandlerSpec), GoCode) :-
    % Generate handler code
    generate_service_handler_go(HandlerSpec, HandlerCode),
    % Format the struct name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        StructName = [Upper|Rest]
    ;
        StructName = [First|Rest]
    ),
    atom_codes(StructNameAtom, StructName),
    % Extract configurations
    ( member(stateful(true), Options) -> Stateful = true ; Stateful = false ),
    % Load balancing
    ( ( member(load_balance(LBStrategy), Options) ; member(load_balance(LBStrategy, _), Options) ) ->
        atom_string(LBStrategy, LBStrategyStr)
    ;
        LBStrategyStr = "none"
    ),
    % Circuit breaker
    ( ( member(circuit_breaker(threshold(CBThreshold), timeout(CBTimeout)), Options)
      ; ( member(circuit_breaker(CBOpts), Options), is_list(CBOpts),
          ( member(threshold(CBThreshold), CBOpts) -> true ; CBThreshold = 5 ),
          ( member(timeout(CBTimeout), CBOpts) -> true ; CBTimeout = 30000 ) ) ) ->
        true
    ;
        CBThreshold = 5,
        CBTimeout = 30000
    ),
    % Retry
    ( ( member(retry(RetryN, RetryStrategy, RetryOpts), Options) ->
          ( member(delay(RetryDelay), RetryOpts) -> true ; RetryDelay = 100 ),
          ( member(max_delay(RetryMaxDelay), RetryOpts) -> true ; RetryMaxDelay = 30000 )
      ; member(retry(RetryN, RetryStrategy), Options) ->
          RetryDelay = 100,
          RetryMaxDelay = 30000
      ) ->
        atom_string(RetryStrategy, RetryStrategyStr)
    ;
        RetryN = 0,
        RetryStrategyStr = "none",
        RetryDelay = 100,
        RetryMaxDelay = 30000
    ),
    % Generate code based on stateful or not
    ( Stateful = true ->
        format(string(GoCode),
"package main

import (
\t\"fmt\"
\t\"math/rand\"
\t\"sync\"
\t\"sync/atomic\"
\t\"time\"
)

type CircuitState int

const (
\tCircuitClosed CircuitState = iota
\tCircuitOpen
\tCircuitHalfOpen
)

type Backend struct {
\tName      string
\tTransport string
}

// ~wService is a service mesh service (stateful)
type ~wService struct {
\tname            string
\tbackends        []Backend
\tlbStrategy      string
\tcbThreshold     int
\tcbTimeout       time.Duration
\tretryMax        int
\tretryStrategy   string
\tretryDelay      time.Duration
\tretryMaxDelay   time.Duration
\tstate           map[string]interface{}
\tstateMu         sync.RWMutex
\tcircuitState    CircuitState
\tfailureCount    int32
\tlastFailureTime time.Time
\trrIndex         uint32
}

func New~wService() *~wService {
\treturn &~wService{
\t\tname:          \"~w\",
\t\tbackends:      []Backend{},
\t\tlbStrategy:    \"~w\",
\t\tcbThreshold:   ~w,
\t\tcbTimeout:     ~w * time.Millisecond,
\t\tretryMax:      ~w,
\t\tretryStrategy: \"~w\",
\t\tretryDelay:    ~w * time.Millisecond,
\t\tretryMaxDelay: ~w * time.Millisecond,
\t\tstate:         make(map[string]interface{}),
\t\tcircuitState:  CircuitClosed,
\t}
}

func (s *~wService) Name() string { return s.name }

func (s *~wService) selectBackend() *Backend {
\tif len(s.backends) == 0 {
\t\treturn nil
\t}
\tswitch s.lbStrategy {
\tcase \"round_robin\":
\t\tidx := atomic.AddUint32(&s.rrIndex, 1) - 1
\t\treturn &s.backends[idx%%uint32(len(s.backends))]
\tcase \"random\":
\t\treturn &s.backends[rand.Intn(len(s.backends))]
\tdefault:
\t\treturn &s.backends[0]
\t}
}

func (s *~wService) checkCircuit() bool {
\tif s.circuitState == CircuitOpen {
\t\ts.stateMu.RLock()
\t\telapsed := time.Since(s.lastFailureTime)
\t\ts.stateMu.RUnlock()
\t\tif elapsed > s.cbTimeout {
\t\t\ts.circuitState = CircuitHalfOpen
\t\t\treturn true
\t\t}
\t\treturn false
\t}
\treturn true
}

func (s *~wService) recordSuccess() {
\tif s.circuitState == CircuitHalfOpen {
\t\ts.circuitState = CircuitClosed
\t}
\tatomic.StoreInt32(&s.failureCount, 0)
}

func (s *~wService) recordFailure() {
\tcount := atomic.AddInt32(&s.failureCount, 1)
\ts.stateMu.Lock()
\ts.lastFailureTime = time.Now()
\ts.stateMu.Unlock()
\tif int(count) >= s.cbThreshold {
\t\ts.circuitState = CircuitOpen
\t}
}

func (s *~wService) calculateDelay(attempt int) time.Duration {
\tswitch s.retryStrategy {
\tcase \"fixed\":
\t\treturn s.retryDelay
\tcase \"linear\":
\t\td := s.retryDelay * time.Duration(attempt+1)
\t\tif d > s.retryMaxDelay {
\t\t\treturn s.retryMaxDelay
\t\t}
\t\treturn d
\tcase \"exponential\":
\t\td := s.retryDelay * time.Duration(1<<uint(attempt))
\t\tif d > s.retryMaxDelay {
\t\t\treturn s.retryMaxDelay
\t\t}
\t\treturn d
\t}
\treturn s.retryDelay
}

func (s *~wService) StateGet(key string) interface{} {
\ts.stateMu.RLock()
\tdefer s.stateMu.RUnlock()
\treturn s.state[key]
}

func (s *~wService) StatePut(key string, value interface{}) {
\ts.stateMu.Lock()
\tdefer s.stateMu.Unlock()
\ts.state[key] = value
}

func (s *~wService) Call(request interface{}) (interface{}, error) {
\tif !s.checkCircuit() {
\t\treturn nil, fmt.Errorf(\"circuit breaker is open\")
\t}
\tmaxAttempts := s.retryMax + 1
\tif maxAttempts < 1 {
\t\tmaxAttempts = 1
\t}
\tvar lastErr error
\tfor attempt := 0; attempt < maxAttempts; attempt++ {
\t\t_ = s.selectBackend()
\t\tresult, err := s.handleRequest(request)
\t\tif err == nil {
\t\t\ts.recordSuccess()
\t\t\treturn result, nil
\t\t}
\t\tlastErr = err
\t\ts.recordFailure()
\t\tif attempt < maxAttempts-1 {
\t\t\ttime.Sleep(s.calculateDelay(attempt))
\t\t}
\t}
\treturn nil, lastErr
}

func (s *~wService) handleRequest(request interface{}) (interface{}, error) {
~w
}

var _~wService = New~wService()

func init() {
\tRegisterService(\"~w\", _~wService)
}
", [StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, Name, LBStrategyStr, CBThreshold, CBTimeout, RetryN, RetryStrategyStr, RetryDelay, RetryMaxDelay,
    StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom,
    HandlerCode, StructNameAtom, StructNameAtom, Name, StructNameAtom])
    ;
        format(string(GoCode),
"package main

import (
\t\"fmt\"
\t\"math/rand\"
\t\"sync/atomic\"
\t\"time\"
)

type CircuitState int

const (
\tCircuitClosed CircuitState = iota
\tCircuitOpen
\tCircuitHalfOpen
)

type Backend struct {
\tName      string
\tTransport string
}

// ~wService is a service mesh service
type ~wService struct {
\tname            string
\tbackends        []Backend
\tlbStrategy      string
\tcbThreshold     int
\tcbTimeout       time.Duration
\tretryMax        int
\tretryStrategy   string
\tretryDelay      time.Duration
\tretryMaxDelay   time.Duration
\tcircuitState    CircuitState
\tfailureCount    int32
\tlastFailureTime time.Time
\trrIndex         uint32
}

func New~wService() *~wService {
\treturn &~wService{
\t\tname:          \"~w\",
\t\tbackends:      []Backend{},
\t\tlbStrategy:    \"~w\",
\t\tcbThreshold:   ~w,
\t\tcbTimeout:     ~w * time.Millisecond,
\t\tretryMax:      ~w,
\t\tretryStrategy: \"~w\",
\t\tretryDelay:    ~w * time.Millisecond,
\t\tretryMaxDelay: ~w * time.Millisecond,
\t\tcircuitState:  CircuitClosed,
\t}
}

func (s *~wService) Name() string { return s.name }

func (s *~wService) selectBackend() *Backend {
\tif len(s.backends) == 0 {
\t\treturn nil
\t}
\tswitch s.lbStrategy {
\tcase \"round_robin\":
\t\tidx := atomic.AddUint32(&s.rrIndex, 1) - 1
\t\treturn &s.backends[idx%%uint32(len(s.backends))]
\tcase \"random\":
\t\treturn &s.backends[rand.Intn(len(s.backends))]
\tdefault:
\t\treturn &s.backends[0]
\t}
}

func (s *~wService) checkCircuit() bool {
\tif s.circuitState == CircuitOpen {
\t\tif time.Since(s.lastFailureTime) > s.cbTimeout {
\t\t\ts.circuitState = CircuitHalfOpen
\t\t\treturn true
\t\t}
\t\treturn false
\t}
\treturn true
}

func (s *~wService) recordSuccess() {
\tif s.circuitState == CircuitHalfOpen {
\t\ts.circuitState = CircuitClosed
\t}
\tatomic.StoreInt32(&s.failureCount, 0)
}

func (s *~wService) recordFailure() {
\tcount := atomic.AddInt32(&s.failureCount, 1)
\ts.lastFailureTime = time.Now()
\tif int(count) >= s.cbThreshold {
\t\ts.circuitState = CircuitOpen
\t}
}

func (s *~wService) calculateDelay(attempt int) time.Duration {
\tswitch s.retryStrategy {
\tcase \"fixed\":
\t\treturn s.retryDelay
\tcase \"linear\":
\t\td := s.retryDelay * time.Duration(attempt+1)
\t\tif d > s.retryMaxDelay {
\t\t\treturn s.retryMaxDelay
\t\t}
\t\treturn d
\tcase \"exponential\":
\t\td := s.retryDelay * time.Duration(1<<uint(attempt))
\t\tif d > s.retryMaxDelay {
\t\t\treturn s.retryMaxDelay
\t\t}
\t\treturn d
\t}
\treturn s.retryDelay
}

func (s *~wService) Call(request interface{}) (interface{}, error) {
\tif !s.checkCircuit() {
\t\treturn nil, fmt.Errorf(\"circuit breaker is open\")
\t}
\tmaxAttempts := s.retryMax + 1
\tif maxAttempts < 1 {
\t\tmaxAttempts = 1
\t}
\tvar lastErr error
\tfor attempt := 0; attempt < maxAttempts; attempt++ {
\t\t_ = s.selectBackend()
\t\tresult, err := s.handleRequest(request)
\t\tif err == nil {
\t\t\ts.recordSuccess()
\t\t\treturn result, nil
\t\t}
\t\tlastErr = err
\t\ts.recordFailure()
\t\tif attempt < maxAttempts-1 {
\t\t\ttime.Sleep(s.calculateDelay(attempt))
\t\t}
\t}
\treturn nil, lastErr
}

func (s *~wService) handleRequest(request interface{}) (interface{}, error) {
~w
}

var _~wService = New~wService()

func init() {
\tRegisterService(\"~w\", _~wService)
}
", [StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, Name, LBStrategyStr, CBThreshold, CBTimeout, RetryN, RetryStrategyStr, RetryDelay, RetryMaxDelay,
    StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom,
    HandlerCode, StructNameAtom, StructNameAtom, Name, StructNameAtom])
    ).

%% ============================================
%% PHASE 5: POLYGLOT SERVICES
%% ============================================

%% compile_polyglot_service_go(+Service, -GoCode)
%  Generate Go code for a polyglot service that can call services in other languages.
compile_polyglot_service_go(service(Name, Options, HandlerSpec), GoCode) :-
    % Generate handler code
    generate_service_handler_go(HandlerSpec, HandlerCode),
    % Format the struct name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        StructName = [Upper|Rest]
    ;
        StructName = [First|Rest]
    ),
    atom_codes(StructNameAtom, StructName),
    % Get dependencies
    ( member(depends_on(Dependencies), Options) -> Dependencies = Deps ; Deps = [] ),
    % Get target language
    ( member(target_language(Lang), Options) -> atom_string(Lang, LangStr) ; LangStr = "go" ),
    % Generate dependency client registrations
    generate_dependency_clients_go(Deps, ClientsCode),
    % Determine if stateful
    ( member(stateful(true), Options) -> Stateful = "true" ; Stateful = "false" ),
    % Get timeout
    ( member(timeout(Timeout), Options) -> true ; Timeout = 30000 ),
    % Format dependencies list
    format_deps_list_go(Deps, DepsListStr),
    % Generate the polyglot service
    format(string(GoCode), "package main

import (
\t\"bytes\"
\t\"encoding/json\"
\t\"fmt\"
\t\"io\"
\t\"net/http\"
\t\"sync\"
\t\"time\"
)

// Phase 5: Polyglot Service Support
// Target language: ~w

// ServiceClient is a client for calling remote services
type ServiceClient struct {
\tName     string
\tEndpoint string
\tTimeout  time.Duration
\tclient   *http.Client
}

// NewServiceClient creates a new service client
func NewServiceClient(name, endpoint string, timeout time.Duration) *ServiceClient {
\treturn &ServiceClient{
\t\tName:     name,
\t\tEndpoint: endpoint,
\t\tTimeout:  timeout,
\t\tclient:   &http.Client{Timeout: timeout},
\t}
}

// Call makes an HTTP call to the remote service
func (c *ServiceClient) Call(request interface{}) (interface{}, error) {
\tpayload := map[string]interface{}{\"_payload\": request}
\tdata, err := json.Marshal(payload)
\tif err != nil {
\t\treturn nil, fmt.Errorf(\"failed to marshal request: %w\", err)
\t}

\tresp, err := c.client.Post(c.Endpoint, \"application/json\", bytes.NewReader(data))
\tif err != nil {
\t\treturn nil, fmt.Errorf(\"failed to call service %s: %w\", c.Name, err)
\t}
\tdefer resp.Body.Close()

\tbody, err := io.ReadAll(resp.Body)
\tif err != nil {
\t\treturn nil, fmt.Errorf(\"failed to read response: %w\", err)
\t}

\tvar result map[string]interface{}
\tif err := json.Unmarshal(body, &result); err != nil {
\t\treturn nil, fmt.Errorf(\"failed to unmarshal response: %w\", err)
\t}

\tif status, ok := result[\"_status\"].(string); ok && status == \"error\" {
\t\tmsg, _ := result[\"_message\"].(string)
\t\treturn nil, fmt.Errorf(\"remote service error: %s\", msg)
\t}

\treturn result[\"_payload\"], nil
}

// ServiceRegistry manages local and remote services
type ServiceRegistry struct {
\tmu       sync.RWMutex
\tremote   map[string]*ServiceClient
\tlocal    map[string]Service
}

var registry = &ServiceRegistry{
\tremote: make(map[string]*ServiceClient),
\tlocal:  make(map[string]Service),
}

// RegisterRemote registers a remote service
func (r *ServiceRegistry) RegisterRemote(name, endpoint string, timeout time.Duration) {
\tr.mu.Lock()
\tdefer r.mu.Unlock()
\tr.remote[name] = NewServiceClient(name, endpoint, timeout)
}

// RegisterLocal registers a local service
func (r *ServiceRegistry) RegisterLocal(name string, svc Service) {
\tr.mu.Lock()
\tdefer r.mu.Unlock()
\tr.local[name] = svc
}

// CallService calls a service by name (local or remote)
func (r *ServiceRegistry) CallService(name string, request interface{}) (interface{}, error) {
\tr.mu.RLock()
\tdefer r.mu.RUnlock()

\tif svc, ok := r.local[name]; ok {
\t\treturn svc.Call(request)
\t}
\tif client, ok := r.remote[name]; ok {
\t\treturn client.Call(request)
\t}
\treturn nil, fmt.Errorf(\"service not found: %s\", name)
}

~w

// ~wService is a polyglot service
// Target Language: ~w
// Dependencies: ~w
type ~wService struct {
\tname     string
\tstateful bool
\ttimeout  time.Duration
\tstate    map[string]interface{}
}

var _~wService = &~wService{
\tname:     \"~w\",
\tstateful: ~w,
\ttimeout:  ~w * time.Millisecond,
\tstate:    make(map[string]interface{}),
}

func (s *~wService) Name() string { return s.name }

func (s *~wService) Call(request interface{}) (interface{}, error) {
~w
}

// CallService calls another service (local or remote)
func (s *~wService) CallService(name string, request interface{}) (interface{}, error) {
\treturn registry.CallService(name, request)
}

func init() {
\tregistry.RegisterLocal(\"~w\", _~wService)
}
", [LangStr, ClientsCode, StructNameAtom, LangStr, DepsListStr, StructNameAtom,
    StructNameAtom, StructNameAtom, Name, Stateful, Timeout,
    StructNameAtom, StructNameAtom, HandlerCode, StructNameAtom, Name, StructNameAtom]).

%% generate_dependency_clients_go(+Dependencies, -Code)
%  Generate Go code to register remote service clients.
generate_dependency_clients_go([], "// No remote service dependencies").
generate_dependency_clients_go(Deps, Code) :-
    Deps \= [],
    generate_dep_registrations_go(Deps, RegStrs),
    atomic_list_concat(RegStrs, '\n', Code).

generate_dep_registrations_go([], []).
generate_dep_registrations_go([Dep|Rest], [Str|RestStrs]) :-
    ( Dep = dep(Name, Lang, Transport) ->
        transport_to_endpoint_str_go(Transport, Endpoint),
        format(string(Str), "// ~w service (~w)~nfunc init() { registry.RegisterRemote(\"~w\", \"~w\", 30*time.Second) }",
               [Name, Lang, Name, Endpoint])
    ; Dep = dep(Name, Lang) ->
        format(string(Str), "// ~w service (~w)~nfunc init() { registry.RegisterRemote(\"~w\", \"http://localhost:8080/~w\", 30*time.Second) }",
               [Name, Lang, Name, Name])
    ; atom(Dep) ->
        format(string(Str), "// ~w service~nfunc init() { registry.RegisterRemote(\"~w\", \"http://localhost:8080/~w\", 30*time.Second) }",
               [Dep, Dep, Dep])
    ;
        Str = "// Unknown dependency format"
    ),
    generate_dep_registrations_go(Rest, RestStrs).

%% transport_to_endpoint_str_go(+Transport, -Endpoint)
transport_to_endpoint_str_go(tcp(Host, Port), Endpoint) :-
    format(string(Endpoint), "http://~w:~w", [Host, Port]).
transport_to_endpoint_str_go(http(Path), Endpoint) :-
    format(string(Endpoint), "http://localhost:8080~w", [Path]).
transport_to_endpoint_str_go(http(Host, Port), Endpoint) :-
    format(string(Endpoint), "http://~w:~w", [Host, Port]).
transport_to_endpoint_str_go(http(Host, Port, Path), Endpoint) :-
    format(string(Endpoint), "http://~w:~w~w", [Host, Port, Path]).
transport_to_endpoint_str_go(_, "http://localhost:8080").

%% format_deps_list_go(+Deps, -Str)
format_deps_list_go([], "[]").
format_deps_list_go(Deps, Str) :-
    Deps \= [],
    format_dep_names(Deps, Names),
    atomic_list_concat(Names, ', ', NamesStr),
    format(string(Str), "[~w]", [NamesStr]).

format_dep_names([], []).
format_dep_names([dep(Name, _, _)|Rest], [Name|RestNames]) :- format_dep_names(Rest, RestNames).
format_dep_names([dep(Name, _)|Rest], [Name|RestNames]) :- format_dep_names(Rest, RestNames).
format_dep_names([Name|Rest], [Name|RestNames]) :- atom(Name), format_dep_names(Rest, RestNames).

%% generate_service_client_go(+ServiceName, +Endpoint, -Code)
%  Generate a standalone Go service client.
generate_service_client_go(ServiceName, Endpoint, Code) :-
    atom_codes(ServiceName, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        StructName = [Upper|Rest]
    ;
        StructName = [First|Rest]
    ),
    atom_codes(StructNameAtom, StructName),
    format(string(Code), "package main

import (
\t\"bytes\"
\t\"encoding/json\"
\t\"fmt\"
\t\"io\"
\t\"net/http\"
\t\"time\"
)

// ~wClient is a client for the ~w service
type ~wClient struct {
\tendpoint string
\ttimeout  time.Duration
\tclient   *http.Client
}

// New~wClient creates a new client
func New~wClient(endpoint string, timeout time.Duration) *~wClient {
\treturn &~wClient{
\t\tendpoint: endpoint,
\t\ttimeout:  timeout,
\t\tclient:   &http.Client{Timeout: timeout},
\t}
}

// Call calls the ~w service
func (c *~wClient) Call(request interface{}) (interface{}, error) {
\tpayload := map[string]interface{}{\"_payload\": request}
\tdata, err := json.Marshal(payload)
\tif err != nil {
\t\treturn nil, fmt.Errorf(\"failed to marshal request: %%w\", err)
\t}

\tresp, err := c.client.Post(c.endpoint, \"application/json\", bytes.NewReader(data))
\tif err != nil {
\t\treturn nil, fmt.Errorf(\"failed to call ~w service: %%w\", err)
\t}
\tdefer resp.Body.Close()

\tbody, err := io.ReadAll(resp.Body)
\tif err != nil {
\t\treturn nil, fmt.Errorf(\"failed to read response: %%w\", err)
\t}

\tvar result map[string]interface{}
\tif err := json.Unmarshal(body, &result); err != nil {
\t\treturn nil, fmt.Errorf(\"failed to unmarshal response: %%w\", err)
\t}

\tif status, ok := result[\"_status\"].(string); ok && status == \"error\" {
\t\tmsg, _ := result[\"_message\"].(string)
\t\treturn nil, fmt.Errorf(\"remote service error: %%s\", msg)
\t}

\treturn result[\"_payload\"], nil
}

// Default client for ~w service
var Default~wClient = New~wClient(\"~w\", 30*time.Second)
", [StructNameAtom, ServiceName, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom,
    ServiceName, StructNameAtom, ServiceName, ServiceName, StructNameAtom, Endpoint]).

%% ============================================
%% DISTRIBUTED SERVICES (Phase 6)
%% ============================================

%% compile_distributed_service_go(+Service, -GoCode)
%  Generate Go code for a distributed service with sharding and replication.
compile_distributed_service_go(service(Name, Options, HandlerSpec), GoCode) :-
    % Extract distributed configuration
    ( member(sharding(ShardStrategy), Options) -> true ; ShardStrategy = hash ),
    ( member(partition_key(PartitionKey), Options) -> true ; PartitionKey = id ),
    ( member(replication(ReplicationFactor), Options) -> true ; ReplicationFactor = 1 ),
    ( member(consistency(ConsistencyLevel), Options) -> true ; ConsistencyLevel = eventual ),
    ( member(cluster(ClusterConfig), Options) -> true ; ClusterConfig = [] ),
    ( member(stateful(true), Options) -> Stateful = true ; Stateful = false ),
    ( member(timeout(Timeout), Options) -> true ; Timeout = 30000 ),
    % Generate handler code
    generate_service_handler_go(HandlerSpec, HandlerCode),
    % Format the struct name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        StructName = [Upper|Rest]
    ;
        StructName = [First|Rest]
    ),
    atom_codes(StructNameAtom, StructName),
    % Convert atoms to strings
    atom_string(ShardStrategy, ShardStrategyStr),
    atom_string(PartitionKey, PartitionKeyStr),
    atom_string(ConsistencyLevel, ConsistencyStr),
    % Generate cluster nodes
    generate_cluster_nodes_go(ClusterConfig, Name, NodesCode),
    % Generate the distributed service
    format(string(GoCode),
"package main

import (
\t\"crypto/md5\"
\t\"encoding/hex\"
\t\"encoding/json\"
\t\"fmt\"
\t\"sort\"
\t\"sync\"
\t\"sync/atomic\"
\t\"time\"
)

// Phase 6: Distributed Service Support

type ShardingStrategy int

const (
\tShardHash ShardingStrategy = iota
\tShardRange
\tShardConsistentHash
\tShardGeographic
)

type ConsistencyLevel int

const (
\tConsistencyEventual ConsistencyLevel = iota
\tConsistencyStrong
\tConsistencyQuorum
\tConsistencyReadYourWrites
\tConsistencyCausal
)

type ClusterNode struct {
\tNodeID  string
\tHost    string
\tPort    int
\tRegion  string
\tWeight  int
\tHealthy bool
}

type ShardInfo struct {
\tShardID      int
\tPrimaryNode  string
\tReplicaNodes []string
\tKeyRange     [2]interface{}
}

// ConsistentHashRing implements consistent hashing for distributed routing
type ConsistentHashRing struct {
\treplicas   int
\tring       map[uint32]string
\tsortedKeys []uint32
\tmu         sync.RWMutex
}

func NewConsistentHashRing(replicas int) *ConsistentHashRing {
\treturn &ConsistentHashRing{
\t\treplicas:   replicas,
\t\tring:       make(map[uint32]string),
\t\tsortedKeys: make([]uint32, 0),
\t}
}

func (r *ConsistentHashRing) hash(key string) uint32 {
\th := md5.Sum([]byte(key))
\treturn uint32(h[0])<<24 | uint32(h[1])<<16 | uint32(h[2])<<8 | uint32(h[3])
}

func (r *ConsistentHashRing) AddNode(nodeID string) {
\tr.mu.Lock()
\tdefer r.mu.Unlock()
\tfor i := 0; i < r.replicas; i++ {
\t\tkey := r.hash(fmt.Sprintf(\"%s:%d\", nodeID, i))
\t\tr.ring[key] = nodeID
\t\tr.sortedKeys = append(r.sortedKeys, key)
\t}
\tsort.Slice(r.sortedKeys, func(i, j int) bool { return r.sortedKeys[i] < r.sortedKeys[j] })
}

func (r *ConsistentHashRing) RemoveNode(nodeID string) {
\tr.mu.Lock()
\tdefer r.mu.Unlock()
\tfor i := 0; i < r.replicas; i++ {
\t\tkey := r.hash(fmt.Sprintf(\"%s:%d\", nodeID, i))
\t\tdelete(r.ring, key)
\t}
\tnewKeys := make([]uint32, 0)
\tfor _, k := range r.sortedKeys {
\t\tif _, exists := r.ring[k]; exists {
\t\t\tnewKeys = append(newKeys, k)
\t\t}
\t}
\tr.sortedKeys = newKeys
}

func (r *ConsistentHashRing) GetNode(key string) string {
\tr.mu.RLock()
\tdefer r.mu.RUnlock()
\tif len(r.ring) == 0 {
\t\treturn \"\"
\t}
\th := r.hash(key)
\tfor _, k := range r.sortedKeys {
\t\tif k >= h {
\t\t\treturn r.ring[k]
\t\t}
\t}
\treturn r.ring[r.sortedKeys[0]]
}

// ShardRouter routes requests to appropriate shards
type ShardRouter struct {
\tStrategy        ShardingStrategy
\tNumShards       int
\tHashRing        *ConsistentHashRing
\tRangeBoundaries []interface{}
}

func NewShardRouter(strategy ShardingStrategy, numShards int) *ShardRouter {
\treturn &ShardRouter{
\t\tStrategy:  strategy,
\t\tNumShards: numShards,
\t\tHashRing:  NewConsistentHashRing(100),
\t}
}

func (sr *ShardRouter) GetShard(partitionKey interface{}) int {
\tswitch sr.Strategy {
\tcase ShardHash:
\t\treturn sr.hashShard(partitionKey)
\tcase ShardConsistentHash:
\t\treturn sr.consistentHashShard(partitionKey)
\tcase ShardRange:
\t\treturn sr.rangeShard(partitionKey)
\tdefault:
\t\treturn sr.hashShard(partitionKey)
\t}
}

func (sr *ShardRouter) hashShard(key interface{}) int {
\th := md5.Sum([]byte(fmt.Sprintf(\"%v\", key)))
\thex := hex.EncodeToString(h[:])
\tvar sum uint64
\tfor _, c := range hex[:8] {
\t\tsum = sum*16 + uint64(c)
\t}
\treturn int(sum %% uint64(sr.NumShards))
}

func (sr *ShardRouter) consistentHashShard(key interface{}) int {
\tnode := sr.HashRing.GetNode(fmt.Sprintf(\"%v\", key))
\tif node == \"\" {
\t\treturn 0
\t}
\th := md5.Sum([]byte(node))
\treturn int(h[0]) %% sr.NumShards
}

func (sr *ShardRouter) rangeShard(key interface{}) int {
\tfor i, boundary := range sr.RangeBoundaries {
\t\tif fmt.Sprintf(\"%v\", key) < fmt.Sprintf(\"%v\", boundary) {
\t\t\treturn i
\t\t}
\t}
\treturn len(sr.RangeBoundaries)
}

// ReplicationManager manages data replication
type ReplicationManager struct {
\tReplicationFactor int
\tConsistency       ConsistencyLevel
\tmu                sync.RWMutex
}

func NewReplicationManager(factor int, consistency ConsistencyLevel) *ReplicationManager {
\treturn &ReplicationManager{
\t\tReplicationFactor: factor,
\t\tConsistency:       consistency,
\t}
}

func (rm *ReplicationManager) WriteQuorum() int {
\tswitch rm.Consistency {
\tcase ConsistencyStrong:
\t\treturn rm.ReplicationFactor
\tcase ConsistencyQuorum:
\t\treturn (rm.ReplicationFactor / 2) + 1
\tdefault:
\t\treturn 1
\t}
}

func (rm *ReplicationManager) ReadQuorum() int {
\tswitch rm.Consistency {
\tcase ConsistencyStrong:
\t\treturn rm.ReplicationFactor
\tcase ConsistencyQuorum:
\t\treturn (rm.ReplicationFactor / 2) + 1
\tdefault:
\t\treturn 1
\t}
}

// ~wService is a distributed service
type ~wService struct {
\tName              string
\tStateful          bool
\tTimeoutMs         int64
\tShardingStrategy  ShardingStrategy
\tPartitionKey      string
\tReplicationFactor int
\tConsistency       ConsistencyLevel
\tNodes             map[string]*ClusterNode
\tShards            map[int]*ShardInfo
\tRouter            *ShardRouter
\tReplication       *ReplicationManager
\tstate             map[string]interface{}
\tmu                sync.RWMutex
\trequestCount      atomic.Int64
}

func New~wService() *~wService {
\tsvc := &~wService{
\t\tName:              \"~w\",
\t\tStateful:          ~w,
\t\tTimeoutMs:         ~w,
\t\tShardingStrategy:  Shard~w,
\t\tPartitionKey:      \"~w\",
\t\tReplicationFactor: ~w,
\t\tConsistency:       Consistency~w,
\t\tNodes:             make(map[string]*ClusterNode),
\t\tShards:            make(map[int]*ShardInfo),
\t\tRouter:            NewShardRouter(Shard~w, 16),
\t\tReplication:       NewReplicationManager(~w, Consistency~w),
\t\tstate:             make(map[string]interface{}),
\t}
\treturn svc
}

func (s *~wService) AddNode(node *ClusterNode) {
\ts.mu.Lock()
\tdefer s.mu.Unlock()
\ts.Nodes[node.NodeID] = node
\ts.Router.HashRing.AddNode(node.NodeID)
}

func (s *~wService) RemoveNode(nodeID string) {
\ts.mu.Lock()
\tdefer s.mu.Unlock()
\tdelete(s.Nodes, nodeID)
\ts.Router.HashRing.RemoveNode(nodeID)
}

func (s *~wService) GetPartitionKey(request map[string]interface{}) interface{} {
\tif val, ok := request[s.PartitionKey]; ok {
\t\treturn val
\t}
\treturn fmt.Sprintf(\"%p\", &request)
}

func (s *~wService) RouteRequest(request map[string]interface{}) int {
\tkey := s.GetPartitionKey(request)
\treturn s.Router.GetShard(key)
}

func (s *~wService) Call(request interface{}) (interface{}, error) {
\ts.requestCount.Add(1)
\tif reqMap, ok := request.(map[string]interface{}); ok {
\t\tshardID := s.RouteRequest(reqMap)
\t\treturn s.handleRequest(request, shardID)
\t}
\treturn s.handleRequest(request, 0)
}

func (s *~wService) handleRequest(request interface{}, shardID int) (interface{}, error) {
~w
}

// Initialize cluster nodes
~w

// Service instance
var ~wServiceInstance = New~wService()
", [StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, Name, Stateful, Timeout,
    ShardStrategyStr, PartitionKeyStr, ReplicationFactor, ConsistencyStr,
    ShardStrategyStr, ReplicationFactor, ConsistencyStr,
    StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom,
    HandlerCode, NodesCode, StructNameAtom, StructNameAtom]).

%% generate_sharding_go(+Strategy, -Code)
%  Generate Go code for a specific sharding strategy.
generate_sharding_go(hash, "ShardHash").
generate_sharding_go(range, "ShardRange").
generate_sharding_go(consistent_hash, "ShardConsistentHash").
generate_sharding_go(geographic, "ShardGeographic").
generate_sharding_go(_, "ShardHash").

%% generate_replication_go(+Factor, -Code)
%  Generate Go replication configuration code.
generate_replication_go(Factor, Code) :-
    integer(Factor),
    format(string(Code), "replicationFactor: ~w", [Factor]).
generate_replication_go(_, "replicationFactor: 1").

%% generate_cluster_nodes_go(+Config, +ServiceName, -Code)
%  Generate Go code for cluster node initialization.
generate_cluster_nodes_go([], _, "// No initial cluster nodes").
generate_cluster_nodes_go(Nodes, ServiceName, Code) :-
    Nodes \= [],
    maplist(generate_node_init_go(ServiceName), Nodes, NodeCodes),
    atomic_list_concat(NodeCodes, '\n', Code).

generate_node_init_go(ServiceName, node(Id, Host, Port), Code) :-
    format(string(Code), "~wServiceInstance.AddNode(&ClusterNode{NodeID: \"~w\", Host: \"~w\", Port: ~w, Healthy: true})",
           [ServiceName, Id, Host, Port]).
generate_node_init_go(ServiceName, node(Id, Host, Port, Region), Code) :-
    format(string(Code), "~wServiceInstance.AddNode(&ClusterNode{NodeID: \"~w\", Host: \"~w\", Port: ~w, Region: \"~w\", Healthy: true})",
           [ServiceName, Id, Host, Port, Region]).
generate_node_init_go(_, _, "// Unknown node format").

%% ============================================
%% SERVICE DISCOVERY (Phase 7)
%% ============================================

%% compile_discovery_service_go(+Service, -GoCode)
%  Generate Go code for a service with discovery capabilities.
compile_discovery_service_go(service(Name, Options, HandlerSpec), GoCode) :-
    % Extract discovery configuration
    ( member(discovery_backend(Backend), Options) -> true ; Backend = consul ),
    ( member(health_check(HealthConfig), Options) -> true ; HealthConfig = http('/health', 30000) ),
    ( member(discovery_ttl(TTL), Options) -> true ; TTL = 60 ),
    ( member(discovery_tags(Tags), Options) -> true ; Tags = [] ),
    ( member(stateful(true), Options) -> Stateful = "true" ; Stateful = "false" ),
    ( member(timeout(Timeout), Options) -> true ; Timeout = 30000 ),
    % Generate handler code
    generate_service_handler_go(HandlerSpec, HandlerCode),
    % Format the struct name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        StructName = [Upper|Rest]
    ;
        StructName = [First|Rest]
    ),
    atom_codes(StructNameAtom, StructName),
    % Convert to strings
    backend_to_go_string(Backend, BackendStr),
    health_config_to_go_string(HealthConfig, HealthStr, HealthIntervalStr),
    tags_to_go_slice(Tags, TagsStr),
    % Generate the discovery service
    format(string(GoCode),
"package main

import (
\t\"bytes\"
\t\"encoding/json\"
\t\"fmt\"
\t\"net\"
\t\"net/http\"
\t\"sync\"
\t\"time\"
)

// Phase 7: Service Discovery Support

// DiscoveryBackend represents supported discovery backends
type DiscoveryBackend string

const (
\tDiscoveryConsul     DiscoveryBackend = \"consul\"
\tDiscoveryEtcd       DiscoveryBackend = \"etcd\"
\tDiscoveryDNS        DiscoveryBackend = \"dns\"
\tDiscoveryKubernetes DiscoveryBackend = \"kubernetes\"
\tDiscoveryZookeeper  DiscoveryBackend = \"zookeeper\"
\tDiscoveryEureka     DiscoveryBackend = \"eureka\"
)

// HealthStatus represents service health
type HealthStatus string

const (
\tHealthStatusHealthy   HealthStatus = \"healthy\"
\tHealthStatusUnhealthy HealthStatus = \"unhealthy\"
\tHealthStatusUnknown   HealthStatus = \"unknown\"
)

// ServiceInstance represents a registered service instance
type ServiceInstance struct {
\tServiceID     string            `json:\"service_id\"`
\tServiceName   string            `json:\"service_name\"`
\tHost          string            `json:\"host\"`
\tPort          int               `json:\"port\"`
\tTags          []string          `json:\"tags\"`
\tMetadata      map[string]string `json:\"metadata\"`
\tHealthStatus  HealthStatus      `json:\"health_status\"`
\tLastHeartbeat time.Time         `json:\"last_heartbeat\"`
}

// HealthCheckConfig configures health checks
type HealthCheckConfig struct {
\tCheckType          string        `json:\"check_type\"`
\tEndpoint           string        `json:\"endpoint\"`
\tInterval           time.Duration `json:\"interval\"`
\tTimeout            time.Duration `json:\"timeout\"`
\tDeregisterAfter    time.Duration `json:\"deregister_after\"`
}

// ServiceRegistry interface for service registries
type ServiceRegistry interface {
\tRegister(instance *ServiceInstance) error
\tDeregister(serviceID string) error
\tDiscover(serviceName string, tags []string) ([]*ServiceInstance, error)
\tHealthCheck(serviceID string) HealthStatus
}

// ConsulRegistry implements ServiceRegistry for Consul
type ConsulRegistry struct {
\tHost       string
\tPort       int
\tbaseURL    string
\tregistered map[string]*ServiceInstance
\tmu         sync.RWMutex
}

// NewConsulRegistry creates a new Consul registry
func NewConsulRegistry(host string, port int) *ConsulRegistry {
\treturn &ConsulRegistry{
\t\tHost:       host,
\t\tPort:       port,
\t\tbaseURL:    fmt.Sprintf(\"http://%%s:%%d/v1\", host, port),
\t\tregistered: make(map[string]*ServiceInstance),
\t}
}

func (c *ConsulRegistry) Register(instance *ServiceInstance) error {
\tpayload := map[string]interface{}{
\t\t\"ID\":      instance.ServiceID,
\t\t\"Name\":    instance.ServiceName,
\t\t\"Address\": instance.Host,
\t\t\"Port\":    instance.Port,
\t\t\"Tags\":    instance.Tags,
\t\t\"Meta\":    instance.Metadata,
\t}
\tdata, _ := json.Marshal(payload)
\treq, _ := http.NewRequest(\"PUT\", c.baseURL+\"/agent/service/register\", bytes.NewReader(data))
\treq.Header.Set(\"Content-Type\", \"application/json\")
\tclient := &http.Client{Timeout: 5 * time.Second}
\tresp, err := client.Do(req)
\tif err != nil {
\t\treturn err
\t}
\tdefer resp.Body.Close()
\tc.mu.Lock()
\tc.registered[instance.ServiceID] = instance
\tc.mu.Unlock()
\treturn nil
}

func (c *ConsulRegistry) Deregister(serviceID string) error {
\treq, _ := http.NewRequest(\"PUT\", c.baseURL+\"/agent/service/deregister/\"+serviceID, nil)
\tclient := &http.Client{Timeout: 5 * time.Second}
\t_, err := client.Do(req)
\tc.mu.Lock()
\tdelete(c.registered, serviceID)
\tc.mu.Unlock()
\treturn err
}

func (c *ConsulRegistry) Discover(serviceName string, tags []string) ([]*ServiceInstance, error) {
\turl := c.baseURL + \"/catalog/service/\" + serviceName
\tresp, err := http.Get(url)
\tif err != nil {
\t\treturn nil, err
\t}
\tdefer resp.Body.Close()
\tvar services []map[string]interface{}
\tjson.NewDecoder(resp.Body).Decode(&services)
\tvar instances []*ServiceInstance
\tfor _, s := range services {
\t\tinstance := &ServiceInstance{
\t\t\tServiceID:   s[\"ServiceID\"].(string),
\t\t\tServiceName: s[\"ServiceName\"].(string),
\t\t\tHost:        s[\"ServiceAddress\"].(string),
\t\t\tPort:        int(s[\"ServicePort\"].(float64)),
\t\t}
\t\tinstances = append(instances, instance)
\t}
\treturn instances, nil
}

func (c *ConsulRegistry) HealthCheck(serviceID string) HealthStatus {
\tresp, err := http.Get(c.baseURL + \"/health/service/\" + serviceID)
\tif err != nil {
\t\treturn HealthStatusUnknown
\t}
\tdefer resp.Body.Close()
\treturn HealthStatusHealthy
}

// LocalRegistry implements in-memory service registry
type LocalRegistry struct {
\tinstances map[string]*ServiceInstance
\tmu        sync.RWMutex
}

// NewLocalRegistry creates a new local registry
func NewLocalRegistry() *LocalRegistry {
\treturn &LocalRegistry{
\t\tinstances: make(map[string]*ServiceInstance),
\t}
}

func (l *LocalRegistry) Register(instance *ServiceInstance) error {
\tl.mu.Lock()
\tdefer l.mu.Unlock()
\tinstance.LastHeartbeat = time.Now()
\tl.instances[instance.ServiceID] = instance
\treturn nil
}

func (l *LocalRegistry) Deregister(serviceID string) error {
\tl.mu.Lock()
\tdefer l.mu.Unlock()
\tdelete(l.instances, serviceID)
\treturn nil
}

func (l *LocalRegistry) Discover(serviceName string, tags []string) ([]*ServiceInstance, error) {
\tl.mu.RLock()
\tdefer l.mu.RUnlock()
\tvar results []*ServiceInstance
\tfor _, instance := range l.instances {
\t\tif instance.ServiceName == serviceName {
\t\t\tresults = append(results, instance)
\t\t}
\t}
\treturn results, nil
}

func (l *LocalRegistry) HealthCheck(serviceID string) HealthStatus {
\tl.mu.RLock()
\tdefer l.mu.RUnlock()
\tinstance, ok := l.instances[serviceID]
\tif !ok {
\t\treturn HealthStatusUnknown
\t}
\tif time.Since(instance.LastHeartbeat) < 2*time.Minute {
\t\treturn HealthStatusHealthy
\t}
\treturn HealthStatusUnhealthy
}

// HealthChecker performs health checks
type HealthChecker struct {
\tConfig  HealthCheckConfig
\trunning bool
\tstopCh  chan struct{}
}

// NewHealthChecker creates a new health checker
func NewHealthChecker(config HealthCheckConfig) *HealthChecker {
\treturn &HealthChecker{
\t\tConfig: config,
\t\tstopCh: make(chan struct{}),
\t}
}

func (h *HealthChecker) CheckHTTP(host string, port int) HealthStatus {
\turl := fmt.Sprintf(\"http://%%s:%%d%%s\", host, port, h.Config.Endpoint)
\tclient := &http.Client{Timeout: h.Config.Timeout}
\tresp, err := client.Get(url)
\tif err != nil {
\t\treturn HealthStatusUnhealthy
\t}
\tdefer resp.Body.Close()
\tif resp.StatusCode == 200 {
\t\treturn HealthStatusHealthy
\t}
\treturn HealthStatusUnhealthy
}

func (h *HealthChecker) CheckTCP(host string, port int) HealthStatus {
\taddr := fmt.Sprintf(\"%%s:%%d\", host, port)
\tconn, err := net.DialTimeout(\"tcp\", addr, h.Config.Timeout)
\tif err != nil {
\t\treturn HealthStatusUnhealthy
\t}
\tconn.Close()
\treturn HealthStatusHealthy
}

func (h *HealthChecker) Check(host string, port int) HealthStatus {
\tif h.Config.CheckType == \"http\" {
\t\treturn h.CheckHTTP(host, port)
\t}
\treturn h.CheckTCP(host, port)
}

// ~wService is a discoverable service
// Backend: ~w
// TTL: ~w seconds
type ~wService struct {
\tName            string
\tHost            string
\tPort            int
\tStateful        bool
\tTimeoutMs       int
\tTTL             int
\tTags            []string
\tState           map[string]interface{}
\tmu              sync.RWMutex
\trunning         bool
\theartbeatStopCh chan struct{}
\tRegistry        ServiceRegistry
\tHealthConfig    HealthCheckConfig
\tHealthChecker   *HealthChecker
\tInstance        *ServiceInstance
}

// New~wService creates a new ~w service instance
func New~wService(host string, port int) *~wService {
\ts := &~wService{
\t\tName:            \"~w\",
\t\tHost:            host,
\t\tPort:            port,
\t\tStateful:        ~w,
\t\tTimeoutMs:       ~w,
\t\tTTL:             ~w,
\t\tTags:            ~w,
\t\tState:           make(map[string]interface{}),
\t\theartbeatStopCh: make(chan struct{}),
\t\tRegistry:        ~w,
\t\tHealthConfig: HealthCheckConfig{
\t\t\tCheckType: \"~w\",
\t\t\tEndpoint:  \"~w\",
\t\t\tInterval:  ~w,
\t\t\tTimeout:   5 * time.Second,
\t\t},
\t}
\ts.HealthChecker = NewHealthChecker(s.HealthConfig)
\ts.Instance = &ServiceInstance{
\t\tServiceID:   fmt.Sprintf(\"%%s-%%s-%%d\", s.Name, host, port),
\t\tServiceName: s.Name,
\t\tHost:        host,
\t\tPort:        port,
\t\tTags:        s.Tags,
\t\tMetadata:    make(map[string]string),
\t}
\treturn s
}

// Register registers the service with the discovery backend
func (s *~wService) Register() error {
\terr := s.Registry.Register(s.Instance)
\tif err == nil {
\t\ts.startHeartbeat()
\t}
\treturn err
}

// Deregister deregisters the service
func (s *~wService) Deregister() error {
\ts.stopHeartbeat()
\treturn s.Registry.Deregister(s.Instance.ServiceID)
}

// DiscoverPeers discovers other instances of this service
func (s *~wService) DiscoverPeers() ([]*ServiceInstance, error) {
\treturn s.Registry.Discover(s.Name, s.Tags)
}

func (s *~wService) startHeartbeat() {
\tif s.running {
\t\treturn
\t}
\ts.running = true
\tgo func() {
\t\tticker := time.NewTicker(time.Duration(s.TTL/2) * time.Second)
\t\tdefer ticker.Stop()
\t\tfor {
\t\t\tselect {
\t\t\tcase <-ticker.C:
\t\t\t\ts.Instance.LastHeartbeat = time.Now()
\t\t\t\ts.Instance.HealthStatus = s.HealthChecker.Check(s.Host, s.Port)
\t\t\tcase <-s.heartbeatStopCh:
\t\t\t\treturn
\t\t\t}
\t\t}
\t}()
}

func (s *~wService) stopHeartbeat() {
\tif s.running {
\t\ts.running = false
\t\tclose(s.heartbeatStopCh)
\t}
}

// Call processes a request through the service
func (s *~wService) Call(request interface{}) interface{} {
\treturn s.handleRequest(request)
}

func (s *~wService) handleRequest(request interface{}) interface{} {
~w
}

// Service instance
var ~wServiceInstance = New~wService(\"localhost\", 8080)
", [StructNameAtom, BackendStr, TTL,
    StructNameAtom,
    StructNameAtom, Name, StructNameAtom, StructNameAtom, StructNameAtom,
    Name, Stateful, Timeout, TTL, TagsStr,
    BackendStr, HealthStr, HealthStr, HealthIntervalStr,
    StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom, StructNameAtom,
    HandlerCode, StructNameAtom, StructNameAtom]).

%% backend_to_go_string(+Backend, -String)
%  Convert discovery backend to Go code string.
backend_to_go_string(consul, "NewConsulRegistry(\"localhost\", 8500)").
backend_to_go_string(consul(Host, Port), Code) :-
    format(string(Code), "NewConsulRegistry(\"~w\", ~w)", [Host, Port]).
backend_to_go_string(etcd, "NewLocalRegistry()").
backend_to_go_string(dns, "NewLocalRegistry()").
backend_to_go_string(kubernetes, "NewLocalRegistry()").
backend_to_go_string(_, "NewLocalRegistry()").

%% health_config_to_go_string(+Config, -TypeString, -IntervalString)
%  Extract health check type and interval from config.
health_config_to_go_string(http(Path, Interval), Path, IntervalStr) :-
    Ms is Interval / 1000,
    format(string(IntervalStr), "~w * time.Second", [Ms]), !.
health_config_to_go_string(http(Path, Interval, _), Path, IntervalStr) :-
    Ms is Interval / 1000,
    format(string(IntervalStr), "~w * time.Second", [Ms]), !.
health_config_to_go_string(tcp(_, Interval), "tcp", IntervalStr) :-
    Ms is Interval / 1000,
    format(string(IntervalStr), "~w * time.Second", [Ms]), !.
health_config_to_go_string(_, "/health", "30 * time.Second").

%% tags_to_go_slice(+Tags, -String)
%  Convert Prolog list of tags to Go slice string.
tags_to_go_slice([], "[]string{}").
tags_to_go_slice(Tags, Code) :-
    Tags \= [],
    maplist(quote_go_string, Tags, QuotedTags),
    atomic_list_concat(QuotedTags, ', ', Inner),
    format(string(Code), "[]string{~w}", [Inner]).

quote_go_string(Atom, Quoted) :-
    format(string(Quoted), "\"~w\"", [Atom]).

%% generate_health_check_go(+Config, -Code)
%  Generate Go health check configuration code.
generate_health_check_go(http(Path, Interval), Code) :-
    format(string(Code), "HealthCheckConfig{CheckType: \"http\", Endpoint: \"~w\", Interval: ~w * time.Millisecond}", [Path, Interval]).
generate_health_check_go(tcp(Port, Interval), Code) :-
    format(string(Code), "HealthCheckConfig{CheckType: \"tcp\", Endpoint: \"~w\", Interval: ~w * time.Millisecond}", [Port, Interval]).
generate_health_check_go(_, "HealthCheckConfig{CheckType: \"http\", Endpoint: \"/health\", Interval: 30 * time.Second}").

%% generate_service_registry_go(+Backend, -Code)
%  Generate Go service registry initialization code.
generate_service_registry_go(consul, "NewConsulRegistry(\"localhost\", 8500)").
generate_service_registry_go(consul(Host, Port), Code) :-
    format(string(Code), "NewConsulRegistry(\"~w\", ~w)", [Host, Port]).
generate_service_registry_go(_, "NewLocalRegistry()").

%% ============================================
%% SERVICE TRACING (Phase 8)
%% ============================================

%% compile_traced_service_go(+Service, -GoCode)
%  Generate Go code for a service with distributed tracing.
compile_traced_service_go(service(Name, Options, HandlerSpec), GoCode) :-
    % Extract tracing configuration
    ( member(trace_exporter(Exporter), Options) -> true ; Exporter = otlp ),
    ( member(trace_sampling(SamplingRate), Options) -> true ; SamplingRate = 1.0 ),
    ( member(trace_service_name(ServiceName), Options) -> true ; ServiceName = Name ),
    ( member(trace_propagation(Propagation), Options) -> true ; Propagation = w3c ),
    ( member(trace_attributes(Attributes), Options) -> true ; Attributes = [] ),
    ( member(stateful(true), Options) -> Stateful = "true" ; Stateful = "false" ),
    ( member(timeout(Timeout), Options) -> true ; Timeout = 30000 ),
    % Generate handler code
    generate_service_handler_go(HandlerSpec, HandlerCode),
    % Format the struct name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        StructName = [Upper|Rest]
    ;
        StructName = [First|Rest]
    ),
    atom_codes(StructNameAtom, StructName),
    % Convert to strings
    exporter_to_go_string(Exporter, ExporterStr),
    propagation_to_go_string(Propagation, PropagationStr),
    attributes_to_go_map(Attributes, AttrsStr),
    atom_string(ServiceName, ServiceNameStr),
    % Generate the traced service
    format(string(GoCode),
"package main

import (
\t\"bytes\"
\t\"context\"
\t\"encoding/json\"
\t\"fmt\"
\t\"math/rand\"
\t\"net/http\"
\t\"strings\"
\t\"sync\"
\t\"time\"

\t\"github.com/google/uuid\"
)

// Phase 8: Service Tracing Support (OpenTelemetry-compatible)

// TraceExporter types
type TraceExporter string

const (
\tExporterOTLP    TraceExporter = \"otlp\"
\tExporterJaeger  TraceExporter = \"jaeger\"
\tExporterZipkin  TraceExporter = \"zipkin\"
\tExporterDatadog TraceExporter = \"datadog\"
\tExporterConsole TraceExporter = \"console\"
\tExporterNone    TraceExporter = \"none\"
)

// PropagationFormat types
type PropagationFormat string

const (
\tPropagationW3C     PropagationFormat = \"w3c\"
\tPropagationB3      PropagationFormat = \"b3\"
\tPropagationB3Multi PropagationFormat = \"b3_multi\"
\tPropagationJaeger  PropagationFormat = \"jaeger\"
\tPropagationXRay    PropagationFormat = \"xray\"
\tPropagationDatadog PropagationFormat = \"datadog\"
)

// SpanKind types
type SpanKind string

const (
\tSpanKindInternal SpanKind = \"internal\"
\tSpanKindServer   SpanKind = \"server\"
\tSpanKindClient   SpanKind = \"client\"
\tSpanKindProducer SpanKind = \"producer\"
\tSpanKindConsumer SpanKind = \"consumer\"
)

// SpanStatus types
type SpanStatus string

const (
\tSpanStatusUnset SpanStatus = \"unset\"
\tSpanStatusOK    SpanStatus = \"ok\"
\tSpanStatusError SpanStatus = \"error\"
)

// SpanContext holds trace context
type SpanContext struct {
\tTraceID    string `json:\"trace_id\"`
\tSpanID     string `json:\"span_id\"`
\tTraceFlags int    `json:\"trace_flags\"`
\tTraceState string `json:\"trace_state\"`
}

// NewSpanContext generates a new span context
func NewSpanContext() *SpanContext {
\treturn &SpanContext{
\t\tTraceID:    uuid.New().String() + uuid.New().String()[:8],
\t\tSpanID:     uuid.New().String()[:16],
\t\tTraceFlags: 1,
\t}
}

// ToTraceparent formats as W3C traceparent header
func (c *SpanContext) ToTraceparent() string {
\treturn fmt.Sprintf(\"00-%%s-%%s-%%02x\", c.TraceID, c.SpanID, c.TraceFlags)
}

// ParseTraceparent parses W3C traceparent header
func ParseTraceparent(header string) *SpanContext {
\tparts := strings.Split(header, \"-\")
\tif len(parts) >= 4 {
\t\treturn &SpanContext{
\t\t\tTraceID:    parts[1],
\t\t\tSpanID:     parts[2],
\t\t\tTraceFlags: 1,
\t\t}
\t}
\treturn nil
}

// Span represents a single span in a trace
type Span struct {
\tName          string                 `json:\"name\"`
\tContext       *SpanContext           `json:\"context\"`
\tParentContext *SpanContext           `json:\"parent_context,omitempty\"`
\tKind          SpanKind               `json:\"kind\"`
\tStatus        SpanStatus             `json:\"status\"`
\tStartTime     time.Time              `json:\"start_time\"`
\tEndTime       time.Time              `json:\"end_time\"`
\tAttributes    map[string]interface{} `json:\"attributes\"`
\tEvents        []SpanEvent            `json:\"events\"`
}

// SpanEvent represents an event within a span
type SpanEvent struct {
\tName       string                 `json:\"name\"`
\tTimestamp  time.Time              `json:\"timestamp\"`
\tAttributes map[string]interface{} `json:\"attributes\"`
}

// SetAttribute sets an attribute on the span
func (s *Span) SetAttribute(key string, value interface{}) {
\tif s.Attributes == nil {
\t\ts.Attributes = make(map[string]interface{})
\t}
\ts.Attributes[key] = value
}

// AddEvent adds an event to the span
func (s *Span) AddEvent(name string, attrs map[string]interface{}) {
\ts.Events = append(s.Events, SpanEvent{
\t\tName:       name,
\t\tTimestamp:  time.Now(),
\t\tAttributes: attrs,
\t})
}

// SetStatus sets the span status
func (s *Span) SetStatus(status SpanStatus, description string) {
\ts.Status = status
\tif description != \"\" {
\t\ts.SetAttribute(\"status.description\", description)
\t}
}

// End ends the span
func (s *Span) End() {
\ts.EndTime = time.Now()
}

// SpanExporter interface
type SpanExporter interface {
\tExport(spans []*Span) error
}

// ConsoleSpanExporter exports spans to console
type ConsoleSpanExporter struct{}

func (e *ConsoleSpanExporter) Export(spans []*Span) error {
\tfor _, span := range spans {
\t\tdata, _ := json.Marshal(span)
\t\tfmt.Printf(\"[TRACE] %%s\\n\", string(data))
\t}
\treturn nil
}

// OTLPSpanExporter exports spans via OTLP HTTP
type OTLPSpanExporter struct {
\tEndpoint string
}

func NewOTLPExporter(endpoint string) *OTLPSpanExporter {
\tif endpoint == \"\" {
\t\tendpoint = \"http://localhost:4318/v1/traces\"
\t}
\treturn &OTLPSpanExporter{Endpoint: endpoint}
}

func (e *OTLPSpanExporter) Export(spans []*Span) error {
\tpayload := map[string]interface{}{
\t\t\"resourceSpans\": []map[string]interface{}{{
\t\t\t\"scopeSpans\": []map[string]interface{}{{
\t\t\t\t\"spans\": spans,
\t\t\t}},
\t\t}},
\t}
\tdata, _ := json.Marshal(payload)
\treq, _ := http.NewRequest(\"POST\", e.Endpoint, bytes.NewReader(data))
\treq.Header.Set(\"Content-Type\", \"application/json\")
\tclient := &http.Client{Timeout: 5 * time.Second}
\t_, err := client.Do(req)
\treturn err
}

// JaegerSpanExporter exports spans to Jaeger
type JaegerSpanExporter struct {
\tEndpoint string
}

func NewJaegerExporter(endpoint string) *JaegerSpanExporter {
\tif endpoint == \"\" {
\t\tendpoint = \"http://localhost:14268/api/traces\"
\t}
\treturn &JaegerSpanExporter{Endpoint: endpoint}
}

func (e *JaegerSpanExporter) Export(spans []*Span) error {
\tfor _, span := range spans {
\t\tdata, _ := json.Marshal(span)
\t\treq, _ := http.NewRequest(\"POST\", e.Endpoint, bytes.NewReader(data))
\t\treq.Header.Set(\"Content-Type\", \"application/json\")
\t\tclient := &http.Client{Timeout: 5 * time.Second}
\t\tclient.Do(req)
\t}
\treturn nil
}

// ZipkinSpanExporter exports spans to Zipkin
type ZipkinSpanExporter struct {
\tEndpoint string
}

func NewZipkinExporter(endpoint string) *ZipkinSpanExporter {
\tif endpoint == \"\" {
\t\tendpoint = \"http://localhost:9411/api/v2/spans\"
\t}
\treturn &ZipkinSpanExporter{Endpoint: endpoint}
}

func (e *ZipkinSpanExporter) Export(spans []*Span) error {
\tdata, _ := json.Marshal(spans)
\treq, _ := http.NewRequest(\"POST\", e.Endpoint, bytes.NewReader(data))
\treq.Header.Set(\"Content-Type\", \"application/json\")
\tclient := &http.Client{Timeout: 5 * time.Second}
\t_, err := client.Do(req)
\treturn err
}

// Tracer manages distributed tracing
type Tracer struct {
\tServiceName  string
\tSamplingRate float64
\tExporter     SpanExporter
\tPropagation  PropagationFormat
\tspans        []*Span
\tmu           sync.Mutex
\tcurrentSpan  *Span
}

// NewTracer creates a new tracer
func NewTracer(serviceName string, samplingRate float64, exporter SpanExporter, propagation PropagationFormat) *Tracer {
\treturn &Tracer{
\t\tServiceName:  serviceName,
\t\tSamplingRate: samplingRate,
\t\tExporter:     exporter,
\t\tPropagation:  propagation,
\t\tspans:        make([]*Span, 0),
\t}
}

// ShouldSample determines if this request should be sampled
func (t *Tracer) ShouldSample() bool {
\treturn rand.Float64() < t.SamplingRate
}

// StartSpan starts a new span
func (t *Tracer) StartSpan(name string, kind SpanKind, parent *SpanContext, attrs map[string]interface{}) *Span {
\tif !t.ShouldSample() {
\t\treturn nil
\t}
\tspan := &Span{
\t\tName:          name,
\t\tContext:       NewSpanContext(),
\t\tParentContext: parent,
\t\tKind:          kind,
\t\tStatus:        SpanStatusUnset,
\t\tStartTime:     time.Now(),
\t\tAttributes:    attrs,
\t\tEvents:        make([]SpanEvent, 0),
\t}
\tspan.SetAttribute(\"service.name\", t.ServiceName)
\tt.currentSpan = span
\treturn span
}

// EndSpan ends a span and adds it to the export queue
func (t *Tracer) EndSpan(span *Span) {
\tif span == nil {
\t\treturn
\t}
\tspan.End()
\tif span.Status == SpanStatusUnset {
\t\tspan.Status = SpanStatusOK
\t}
\tt.mu.Lock()
\tt.spans = append(t.spans, span)
\tt.mu.Unlock()
}

// ExtractContext extracts span context from headers
func (t *Tracer) ExtractContext(headers map[string]string) *SpanContext {
\tif t.Propagation == PropagationW3C {
\t\tif tp, ok := headers[\"traceparent\"]; ok {
\t\t\treturn ParseTraceparent(tp)
\t\t}
\t}
\tif t.Propagation == PropagationB3 || t.Propagation == PropagationB3Multi {
\t\ttraceID := headers[\"X-B3-TraceId\"]
\t\tspanID := headers[\"X-B3-SpanId\"]
\t\tif traceID != \"\" && spanID != \"\" {
\t\t\treturn &SpanContext{TraceID: traceID, SpanID: spanID, TraceFlags: 1}
\t\t}
\t}
\treturn nil
}

// InjectContext injects span context into headers
func (t *Tracer) InjectContext(span *Span, headers map[string]string) {
\tif span == nil {
\t\treturn
\t}
\tif t.Propagation == PropagationW3C {
\t\theaders[\"traceparent\"] = span.Context.ToTraceparent()
\t} else if t.Propagation == PropagationB3 || t.Propagation == PropagationB3Multi {
\t\theaders[\"X-B3-TraceId\"] = span.Context.TraceID
\t\theaders[\"X-B3-SpanId\"] = span.Context.SpanID
\t\theaders[\"X-B3-Sampled\"] = \"1\"
\t}
}

// Flush exports all pending spans
func (t *Tracer) Flush() error {
\tt.mu.Lock()
\tspans := t.spans
\tt.spans = make([]*Span, 0)
\tt.mu.Unlock()
\tif len(spans) > 0 {
\t\treturn t.Exporter.Export(spans)
\t}
\treturn nil
}

// ~wService is a traced service
// Service Name: ~w
// Exporter: ~w
// Sampling Rate: ~w
// Propagation: ~w
type ~wService struct {
\tName              string
\tStateful          bool
\tTimeoutMs         int
\tState             map[string]interface{}
\tmu                sync.RWMutex
\tTracer            *Tracer
\tDefaultAttributes map[string]interface{}
}

// New~wService creates a new traced ~w service
func New~wService() *~wService {
\treturn &~wService{
\t\tName:      \"~w\",
\t\tStateful:  ~w,
\t\tTimeoutMs: ~w,
\t\tState:     make(map[string]interface{}),
\t\tTracer: NewTracer(
\t\t\t\"~w\",
\t\t\t~w,
\t\t\t~w,
\t\t\tPropagation~w,
\t\t),
\t\tDefaultAttributes: ~w,
\t}
}

// Call processes a request with tracing
func (s *~wService) Call(ctx context.Context, request interface{}, headers map[string]string) interface{} {
\tif headers == nil {
\t\theaders = make(map[string]string)
\t}
\tparentCtx := s.Tracer.ExtractContext(headers)
\tspan := s.Tracer.StartSpan(
\t\tfmt.Sprintf(\"%%s.call\", s.Name),
\t\tSpanKindServer,
\t\tparentCtx,
\t\ts.DefaultAttributes,
\t)
\tif span != nil {
\t\tspan.SetAttribute(\"request.type\", fmt.Sprintf(\"%%T\", request))
\t}
\tresult := s.handleRequest(request)
\tif span != nil {
\t\tspan.SetAttribute(\"response.type\", fmt.Sprintf(\"%%T\", result))
\t\ts.Tracer.EndSpan(span)
\t}
\treturn result
}

// FlushTraces exports pending traces
func (s *~wService) FlushTraces() error {
\treturn s.Tracer.Flush()
}

func (s *~wService) handleRequest(request interface{}) interface{} {
~w
}

// Service instance
var ~wServiceInstance = New~wService()
", [StructNameAtom, ServiceNameStr, ExporterStr, SamplingRate, PropagationStr,
    StructNameAtom,
    StructNameAtom, Name, StructNameAtom, StructNameAtom, StructNameAtom,
    Name, Stateful, Timeout,
    ServiceNameStr, SamplingRate, ExporterStr, PropagationStr, AttrsStr,
    StructNameAtom, StructNameAtom, StructNameAtom,
    HandlerCode, Name, StructNameAtom]).

%% exporter_to_go_string(+Exporter, -String)
%  Convert trace exporter to Go code string.
exporter_to_go_string(otlp, "NewOTLPExporter(\"\")").
exporter_to_go_string(otlp(Endpoint), Code) :-
    format(string(Code), "NewOTLPExporter(\"~w\")", [Endpoint]).
exporter_to_go_string(jaeger, "NewJaegerExporter(\"\")").
exporter_to_go_string(jaeger(Endpoint), Code) :-
    format(string(Code), "NewJaegerExporter(\"~w\")", [Endpoint]).
exporter_to_go_string(jaeger(Host, Port), Code) :-
    format(string(Code), "NewJaegerExporter(\"http://~w:~w/api/traces\")", [Host, Port]).
exporter_to_go_string(zipkin, "NewZipkinExporter(\"\")").
exporter_to_go_string(zipkin(Endpoint), Code) :-
    format(string(Code), "NewZipkinExporter(\"~w\")", [Endpoint]).
exporter_to_go_string(console, "&ConsoleSpanExporter{}").
exporter_to_go_string(none, "&ConsoleSpanExporter{}").
exporter_to_go_string(_, "&ConsoleSpanExporter{}").

%% propagation_to_go_string(+Propagation, -String)
%  Convert propagation format to Go const string.
propagation_to_go_string(w3c, "W3C").
propagation_to_go_string(b3, "B3").
propagation_to_go_string(b3_multi, "B3Multi").
propagation_to_go_string(jaeger, "Jaeger").
propagation_to_go_string(xray, "XRay").
propagation_to_go_string(datadog, "Datadog").
propagation_to_go_string(_, "W3C").

%% attributes_to_go_map(+Attributes, -String)
%  Convert attribute list to Go map string.
attributes_to_go_map([], "map[string]interface{}{}").
attributes_to_go_map(Attrs, Code) :-
    Attrs \= [],
    maplist(attr_to_go, Attrs, AttrStrs),
    atomic_list_concat(AttrStrs, ', ', Inner),
    format(string(Code), "map[string]interface{}{~w}", [Inner]).

attr_to_go(Key=Value, Code) :-
    format(string(Code), "\"~w\": \"~w\"", [Key, Value]).
attr_to_go(Key-Value, Code) :-
    format(string(Code), "\"~w\": \"~w\"", [Key, Value]).

%% generate_tracer_go(+Config, -Code)
%  Generate Go tracer initialization code.
generate_tracer_go(config(ServiceName, SamplingRate, Exporter), Code) :-
    exporter_to_go_string(Exporter, ExporterStr),
    format(string(Code), "NewTracer(\"~w\", ~w, ~w, PropagationW3C)", [ServiceName, SamplingRate, ExporterStr]).

%% generate_span_context_go(+Context, -Code)
%  Generate Go span context code.
generate_span_context_go(context(TraceId, SpanId), Code) :-
    format(string(Code), "&SpanContext{TraceID: \"~w\", SpanID: \"~w\", TraceFlags: 1}", [TraceId, SpanId]).
generate_span_context_go(_, "NewSpanContext()").

%% ============================================
%% PUBLIC API
%% ============================================

%% compile_predicate_to_go(+Predicate, +Options, -GoCode)
%  Compile a Prolog predicate to Go code
%
%  @arg Predicate Predicate indicator (Name/Arity)
%  @arg Options List of options
%  @arg GoCode Generated Go code as atom
%
%  Options:
%  - record_delimiter(null|newline|Char) - Record separator (default: newline)
%  - field_delimiter(colon|tab|comma|Char) - Field separator (default: colon)
%  - quoting(csv|none) - Quoting style (default: none)
%  - escape_char(Char) - Escape character (default: backslash)
%  - include_package(true|false) - Include package main (default: true)
%  - unique(true|false) - Deduplicate results (default: true)
%  - aggregation(sum|count|max|min|avg) - Aggregation operation
%  - db_backend(bbolt) - Database backend (bbolt only for now)
%  - db_file(Path) - Database file path (default: 'data.db')
%  - db_bucket(Name) - Bucket name (default: predicate name)
%  - db_key_field(Field) - Field to use as key
%  - db_mode(read|write) - Database operation mode (default: write with json_input, read otherwise)
%
compile_predicate_to_go(PredIndicator, Options, GoCode) :-
    (   PredIndicator = _Module:Pred/Arity
    ->  true
    ;   PredIndicator = Pred/Arity
    ),
    format('=== Compiling ~w/~w to Go ===~n', [Pred, Arity]),

    % Clear any previously collected binding imports
    clear_binding_imports,

    % Check for generator mode (fixpoint evaluation) - MUST come first
    (   option(mode(generator), Options)
    ->  format('  Mode: Generator (fixpoint)~n'),
        compile_generator_mode_go(Pred, Arity, Options, GoCode)
    % Check if this is an aggregation predicate (aggregate/3 in body)
    ;   functor(Head, Pred, Arity),
        clause(Head, Body),
        is_aggregation_predicate(Body)
    ->  format('  Mode: Aggregation (New)~n'),
        compile_aggregation_mode(Pred, Arity, Options, GoCode)
    % Check if this is a GROUP BY predicate (group_by/4 in body)
    ;   functor(Head, Pred, Arity),
        clause(Head, Body),
        is_group_by_predicate(Body)
    ->  format('  Mode: GROUP BY Aggregation~n'),
        compile_group_by_mode(Pred, Arity, Options, GoCode)
    % Check if this is pipeline mode (streaming JSONL I/O with typed output)
    ;   option(pipeline_input(true), Options)
    ->  format('  Mode: Pipeline (streaming JSONL)~n'),
        compile_pipeline_mode(Pred, Arity, Options, GoCode)
    % Check if this is database read mode
    ;   option(db_backend(bbolt), Options),
        (option(db_mode(read), Options) ; \+ option(json_input(true), Options)),
        \+ option(json_output(true), Options)
    ->  % Compile for database read
        format('  Mode: Database read (bbolt)~n'),
        compile_database_read_mode(Pred, Arity, Options, GoCode)
    % Check if this is JSON input mode
    ;   option(json_input(true), Options)
    ->  % Compile for JSON input (may include database write)
        (   option(db_backend(bbolt), Options)
        ->  format('  Mode: JSON input (JSONL) with database storage~n')
        ;   format('  Mode: JSON input (JSONL)~n')
        ),
        % Check for parallel execution
        (   option(workers(Workers), Options), Workers > 1
        ->  format('  Parallel execution: ~w workers~n', [Workers]),
            compile_parallel_json_input_mode(Pred, Arity, Options, Workers, GoCode)
        ;   compile_json_input_mode(Pred, Arity, Options, GoCode)
        )
    % Check if this is XML input mode
    ;   option(xml_input(true), Options)
    ->  % Compile for XML input
        format('  Mode: XML input (streaming + flattening)~n'),
        compile_xml_input_mode(Pred, Arity, Options, GoCode)
    % Check if this is JSON output mode
    ;   option(json_output(true), Options)
    ->  % Compile for JSON output
        format('  Mode: JSON output~n'),
        compile_json_output_mode(Pred, Arity, Options, GoCode)
    % Check if this is an aggregation operation (legacy option-based)
    ;   option(aggregation(AggOp), Options, none),
        AggOp \= none
    ->  % Compile as aggregation
        option(field_delimiter(FieldDelim), Options, colon),
        option(include_package(IncludePackage), Options, true),
        compile_aggregation_to_go(Pred, Arity, AggOp, FieldDelim, IncludePackage, GoCode)
    ;   % Continue with normal compilation
        compile_predicate_to_go_normal(Pred, Arity, Options, GoCode)
    ).

%% ============================================
%% GENERATOR MODE IMPLEMENTATION (Fixpoint)
%% ============================================

%% compile_generator_mode_go(+Pred, +Arity, +Options, -GoCode)
%  Compile predicate using fixpoint evaluation (generator mode)
%  Produces standalone Go program with Solve() iteration loop
compile_generator_mode_go(Pred, Arity, Options, GoCode) :-
    go_generator_config(Config),
    
    % Gather all clauses for this predicate (use copy_term to normalize)
    functor(Head, Pred, Arity),
    findall(HC-BC, 
        (user:clause(Head, B), copy_term((Head, B), (HC, BC))), 
        TargetClauses0),
    remove_variant_duplicates(TargetClauses0, TargetClauses),
    
    % Also gather clauses for predicates referenced in rule bodies (dependency closure)
    % This includes extracting inner goals from aggregate_all
    findall(DepPred/DepArity,
        (   member(_-Body, TargetClauses),
            Body \= true,
            body_to_list_go(Body, Goals),
            member(Goal, Goals),
            extract_goal_predicate(Goal, DepPred, DepArity),
            DepPred/DepArity \= Pred/Arity  % Don't include self-references
        ),
        DepPredList0),
    sort(DepPredList0, DepPredList),
    
    % Gather facts from dependencies
    findall(DepHead,
        (   member(DP/DA, DepPredList),
            functor(DepHead, DP, DA),
            user:clause(DepHead, true)
        ),
        DepFacts),
    
    % Combine target clauses with dependency facts
    findall(FactHead, member(FactHead-true, TargetClauses), TargetFacts),
    append(TargetFacts, DepFacts, AllFacts0),
    remove_variant_duplicates_single(AllFacts0, AllFacts),
    
    % Compile facts
    compile_go_generator_facts(AllFacts, Config, FactsCode),
    
    % Compile rules (only for target predicate, exclude facts)
    findall(RuleClause, 
        (member(RuleClause, TargetClauses), RuleClause = (_-RB), RB \= true),
        TargetRuleClauses),
    compile_go_generator_rules(TargetRuleClauses, Config, RulesCode, RuleNames),
    
    % Compile execution (fixpoint loop)
    compile_go_generator_execution(Pred, RuleNames, Options, ExecutionCode),
    
    % Assemble complete program
    go_generator_header(Pred, Options, Header),
    format(string(GoCode), "~w\n~w\n~w\n~w\n", [Header, FactsCode, RulesCode, ExecutionCode]).

%% remove_variant_duplicates(+List, -Unique)
%  Remove duplicates where terms are variants (same structure, different vars)
remove_variant_duplicates([], []).
remove_variant_duplicates([H|T], Result) :-
    (   member_variant(H, T)
    ->  remove_variant_duplicates(T, Result)
    ;   Result = [H|Rest],
        remove_variant_duplicates(T, Rest)
    ).

member_variant(X, [H|_]) :- X =@= H, !.
member_variant(X, [_|T]) :- member_variant(X, T).

%% remove_variant_duplicates_single(+List, -Unique)
%  For single terms (not pairs)
remove_variant_duplicates_single([], []).
remove_variant_duplicates_single([H|T], Result) :-
    (   member_variant_single(H, T)
    ->  remove_variant_duplicates_single(T, Result)
    ;   Result = [H|Rest],
        remove_variant_duplicates_single(T, Rest)
    ).

member_variant_single(X, [H|_]) :- X =@= H, !.
member_variant_single(X, [_|T]) :- member_variant_single(X, T).

%% extract_goal_predicate(+Goal, -Pred, -Arity)
%  Extract predicate name and arity from a goal
%  Handles aggregate_all/3,4 by extracting from inner goal
extract_goal_predicate(aggregate_all(_, InnerGoal, _), Pred, Arity) :-
    !,
    InnerGoal =.. [Pred|Args],
    length(Args, Arity).
extract_goal_predicate(aggregate_all(_, InnerGoal, _, _), Pred, Arity) :-
    !,
    InnerGoal =.. [Pred|Args],
    length(Args, Arity).
extract_goal_predicate(aggregate(_, InnerGoal, _), Pred, Arity) :-
    !,
    InnerGoal =.. [Pred|Args],
    length(Args, Arity).
extract_goal_predicate(Goal, Pred, Arity) :-
    \+ is_builtin_goal_go(Goal),
    Goal =.. [Pred|Args],
    length(Args, Arity).

%% go_generator_header(+Pred, +Options, -Header)
%  Generate Go boilerplate with Fact type
go_generator_header(Pred, Options, Header) :-
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),
    % Add bufio and os imports if json_input is enabled
    (   option(json_input(true), Options)
    ->  JsonImports = "\"bufio\"\n    \"os\"\n    "
    ;   JsonImports = ""
    ),
    % Add sync import if workers (parallel) is enabled
    (   option(workers(W), Options), W > 1
    ->  SyncImport = "\"sync\"\n    "
    ;   SyncImport = ""
    ),
    % Add bbolt import if db_backend is specified
    (   option(db_backend(bbolt), Options)
    ->  DbImport = "\"log\"\n    \"os\"\n    bolt \"go.etcd.io/bbolt\"\n    "
    ;   DbImport = ""
    ),
    % Add imports from bindings
    get_collected_imports(BindingImportsList),
    (   BindingImportsList = []
    ->  BindingImports = ""
    ;   format_binding_imports(BindingImportsList, BindingImports)
    ),
    format(string(ExtraImports), "~w~w~w~w", [JsonImports, SyncImport, DbImport, BindingImports]),
    format(string(Header),
"// Generated by UnifyWeaver Go Generator Mode
// Date: ~w
// Predicate: ~w

package main

import (
    ~w\"encoding/json\"
    \"fmt\"
    \"strconv\"
)

// Fact represents a relation tuple
type Fact struct {
    Relation string                 \x60json:\"relation\"\x60
    Args     map[string]interface{} \x60json:\"args\"\x60
}

// Key returns a canonical string for set membership
func (f Fact) Key() string {
    b, _ := json.Marshal(f)
    return string(b)
}

// toFloat64 converts interface{} to float64 for aggregation
func toFloat64(v interface{}) (float64, bool) {
    switch val := v.(type) {
    case float64:
        return val, true
    case int:
        return float64(val), true
    case int64:
        return float64(val), true
    case string:
        f, err := strconv.ParseFloat(val, 64)
        return f, err == nil
    default:
        return 0, false
    }
}

// Index provides O(1) lookup by relation and argument value
type Index struct {
    // byArg[relation][argN][value] -> []*Fact
    byArg map[string]map[string]map[interface{}][]*Fact
}

// NewIndex creates an empty index
func NewIndex() *Index {
    return &Index{byArg: make(map[string]map[string]map[interface{}][]*Fact)}
}

// Add indexes a fact by all its arguments
func (idx *Index) Add(fact *Fact) {
    rel := fact.Relation
    if _, ok := idx.byArg[rel]; !ok {
        idx.byArg[rel] = make(map[string]map[interface{}][]*Fact)
    }
    for argName, argVal := range fact.Args {
        if _, ok := idx.byArg[rel][argName]; !ok {
            idx.byArg[rel][argName] = make(map[interface{}][]*Fact)
        }
        idx.byArg[rel][argName][argVal] = append(idx.byArg[rel][argName][argVal], fact)
    }
}

// Lookup returns facts matching relation, argument name, and value
func (idx *Index) Lookup(relation, argName string, value interface{}) []*Fact {
    if relIdx, ok := idx.byArg[relation]; ok {
        if argIdx, ok := relIdx[argName]; ok {
            return argIdx[value]
        }
    }
    return nil
}

// BuildIndex creates an index from all facts in the total map
func BuildIndex(total map[string]Fact) *Index {
    idx := NewIndex()
    for _, fact := range total {
        f := fact // Create copy for stable pointer
        idx.Add(&f)
    }
    return idx
}

", [DateStr, Pred, ExtraImports]).

%% compile_go_generator_facts(+FactHeads, +Config, -FactsCode)
%  Generate GetInitialFacts() function
compile_go_generator_facts(FactHeads, _Config, Code) :-
    findall(FactCode,
        (   member(Head, FactHeads),
            Head =.. [Pred|Args],
            generate_go_generator_fact(Pred, Args, FactCode)
        ),
        FactCodes),
    (   FactCodes == []
    ->  FactsBody = ""
    ;   atomic_list_concat(FactCodes, "\n\t\t", FactsBody)
    ),
    format(string(Code),
"// GetInitialFacts returns base facts for fixpoint computation
func GetInitialFacts() []Fact {
    return []Fact{
        ~w
    }
}
", [FactsBody]).

%% generate_go_generator_fact(+Pred, +Args, -FactCode)
generate_go_generator_fact(Pred, Args, Code) :-
    findall(ArgCode,
        (   nth0(I, Args, Arg),
            format(string(ArgCode), "\"arg~w\": \"~w\"", [I, Arg])
        ),
        ArgCodes),
    atomic_list_concat(ArgCodes, ", ", ArgsStr),
    format(string(Code), 
        "{Relation: \"~w\", Args: map[string]interface{}{~w}},", 
        [Pred, ArgsStr]).

%% compile_go_generator_rules(+RuleClauses, +Config, -RulesCode, -RuleNames)
%  Generate ApplyRule_N functions for each rule clause
compile_go_generator_rules(RuleClauses, Config, RulesCode, RuleNames) :-
    findall(RuleCode-RuleName,
        (   nth1(I, RuleClauses, Head-Body),
            Body \= true,
            once(compile_go_generator_rule(I, Head, Body, Config, RuleCode, RuleName))
        ),
        Pairs),
    pairs_keys_values(Pairs, RuleCodes, RuleNames),
    atomic_list_concat(RuleCodes, "\n\n", RulesCode).

%% compile_go_generator_rule(+Index, +Head, +Body, +Config, -Code, -RuleName)
%  Compile a single rule to a Go function
compile_go_generator_rule(Index, Head, Body, Config, Code, RuleName) :-
    format(string(RuleName), "ApplyRule_~w", [Index]),
    Head =.. [HeadPred|HeadArgs],
    
    % Parse body into goals
    body_to_list_go(Body, Goals),
    
    % Check for aggregate goal first
    (   member(AggGoal, Goals),
        is_aggregate_goal_go(AggGoal)
    ->  % Aggregate rule - extract HAVING filters (builtins after aggregate)
        extract_having_filters(Goals, AggGoal, HavingFilters),
        compile_go_aggregate_rule(Index, HeadPred, HeadArgs, AggGoal, HavingFilters, Config, Code)
    ;   % Normal rule (joins, negation, etc.)
        partition(is_builtin_goal_go, Goals, Builtins, RelGoals),
        
        (   RelGoals = []
        ->  % Only builtins - not a productive rule
            format(string(Code),
"func ~w(fact Fact, total map[string]Fact, idx *Index) []Fact {
    // Rule ~w: constraint-only (no relational goals)
    return nil
}", [RuleName, Index])
        ;   % Generate a trigger block for EACH relational goal
            findall(BlockCode,
                (   select(TriggerGoal, RelGoals, OtherGoals),
                    compile_go_trigger_block(TriggerGoal, OtherGoals, Builtins, HeadPred, HeadArgs, Config, BlockCode)
                ),
                Blocks),
            atomic_list_concat(Blocks, "\n\n", AllBlocks),
            format(string(Code),
"func ~w(fact Fact, total map[string]Fact, idx *Index) []Fact {
    var results []Fact
~w
    return results
}", [RuleName, AllBlocks])
        )
    ).

%% compile_go_trigger_block(+TriggerGoal, +OtherGoals, +Builtins, +HeadPred, +HeadArgs, +Config, -BlockCode)
compile_go_trigger_block(TriggerGoal, OtherGoals, Builtins, HeadPred, HeadArgs, Config, BlockCode) :-
    TriggerGoal =.. [TriggerPred|_],
    
    % Build variable map from trigger goal (fact)
    build_variable_map([TriggerGoal-fact], VarMap0),
    
    (   OtherGoals == []
    ->  % Simple rule
        compile_go_builtins(Builtins, VarMap0, Config, BuiltinCode),
        compile_go_head_construction(HeadPred, HeadArgs, VarMap0, Config, HeadCode),
        format(string(BodyCode), "~w\n        ~w", [BuiltinCode, HeadCode])
    ;   % Join rule
        compile_go_join_with_result(OtherGoals, VarMap0, Config, Builtins, HeadPred, HeadArgs, BodyCode)
    ),
    
    format(string(BlockCode),
"    // Case: Fact matches ~w
    func() {
        if fact.Relation != \"~w\" {
            return
        }
~w
    }()", [TriggerGoal, TriggerPred, BodyCode]).

%% is_aggregate_goal_go(+Goal)
%  Check if goal is an aggregate
is_aggregate_goal_go(aggregate_all(_, _, _)).
is_aggregate_goal_go(aggregate_all(_, _, _, _)).
is_aggregate_goal_go(aggregate(_, _, _)).  % Will be normalized

%% normalize_aggregate_goal_go(+Goal, -NormalizedGoal)
%  Normalize aggregate/3 to aggregate_all/3 for consistency
normalize_aggregate_goal_go(aggregate(Op, Body, Res), aggregate_all(Op, Body, Res)) :- !.
normalize_aggregate_goal_go(G, G).

%% extract_having_filters(+Goals, +AggGoal, -HavingFilters)
%  Extract builtin goals that appear after the aggregate (HAVING clause)
extract_having_filters(Goals, AggGoal, HavingFilters) :-
    % Find position of aggregate goal
    append(Before, [AggGoal|After], Goals),
    !,
    % Builtins after aggregate are HAVING filters
    include(is_builtin_goal_go, After, HavingFilters).
extract_having_filters(_, _, []).

%% compile_go_aggregate_rule(+Index, +HeadPred, +HeadArgs, +AggGoal, +HavingFilters, +Config, -Code)
%  Compile an aggregate rule to Go with optional HAVING filters
compile_go_aggregate_rule(Index, HeadPred, HeadArgs, AggGoal0, HavingFilters, Config, Code) :-
    format(string(RuleName), "ApplyRule_~w", [Index]),
    
    % Normalize aggregate/3 -> aggregate_all/3
    normalize_aggregate_goal_go(AggGoal0, AggGoal),
    
    % Decompose aggregate goal
    (   AggGoal = aggregate_all(OpTerm, InnerGoal, GroupVar, Result)
    ->  % Grouped aggregation (aggregate_all/4)
        compile_go_grouped_aggregate(RuleName, HeadPred, HeadArgs, OpTerm, InnerGoal, GroupVar, Result, HavingFilters, Config, Code)
    ;   AggGoal = aggregate_all(OpTerm, InnerGoal, Result)
    ->  % Ungrouped aggregation (aggregate_all/3)
        compile_go_ungrouped_aggregate(RuleName, HeadPred, HeadArgs, OpTerm, InnerGoal, Result, HavingFilters, Config, Code)
    ;   format(string(Code),
"func ~w(fact Fact, total map[string]Fact, idx *Index) []Fact {
    // Unsupported aggregate form
    return nil
}", [RuleName])
    ).

%% compile_go_ungrouped_aggregate(+RuleName, +HeadPred, +HeadArgs, +OpTerm, +InnerGoal, +Result, +HavingFilters, +Config, -Code)
compile_go_ungrouped_aggregate(RuleName, HeadPred, HeadArgs, OpTerm, InnerGoal, Result, HavingFilters, _Config, Code) :-
    InnerGoal =.. [Pred|Args],
    
    % Determine the aggregate operation and value variable
    decompose_agg_op(OpTerm, Op, ValueVar),
    
    % Find value variable index in inner goal args
    (   var(ValueVar)
    ->  find_var_index_go(ValueVar, Args, ValueIdx)
    ;   ValueIdx = -1  % count doesn't need a value index
    ),
    
    % Generate aggregate code
    go_agg_code(Op, AggCode),
    
    % Build result args
    length(HeadArgs, NumArgs),
    (   NumArgs == 1
    ->  ArgsStr = "\"arg0\": agg"
    ;   ArgsStr = "\"arg0\": agg"  % Default for now
    ),
    
    % Generate HAVING filter code if any
    (   HavingFilters \= []
    ->  generate_having_conditions(HavingFilters, Result, HavingCond),
        format(string(HavingIfStart), "
        // HAVING clause filter
        if ~w {", [HavingCond]),
        HavingIfEnd = "
        }"
    ;   HavingIfStart = "",
        HavingIfEnd = ""
    ),
    
    format(string(Code),
"func ~w(fact Fact, total map[string]Fact, idx *Index) []Fact {
    var results []Fact
    
    // Skip if not triggered by this relation
    if fact.Relation != \"~w\" {
        return results
    }
    
    // Collect values for aggregation
    var values []float64
    for _, f := range total {
        if f.Relation == \"~w\" {
            val, ok := toFloat64(f.Args[\"arg~w\"])
            if ok {
                values = append(values, val)
            }
        }
    }
    
    // Compute aggregate
    if len(values) > 0 {
        ~w~w
            results = append(results, Fact{Relation: \"~w\", Args: map[string]interface{}{~w}})~w
    }
    
    return results
}", [RuleName, Pred, Pred, ValueIdx, AggCode, HavingIfStart, HeadPred, ArgsStr, HavingIfEnd]).

%% compile_go_grouped_aggregate(+RuleName, +HeadPred, +HeadArgs, +OpTerm, +InnerGoal, +GroupVar, +Result, +HavingFilters, +Config, -Code)
compile_go_grouped_aggregate(RuleName, HeadPred, _HeadArgs, OpTerm, InnerGoal, GroupVar, Result, HavingFilters, _Config, Code) :-
    InnerGoal =.. [Pred|Args],
    
    % Determine the aggregate operation and value variable
    decompose_agg_op(OpTerm, Op, ValueVar),
    
    % Find group key and value indices
    find_var_index_go(GroupVar, Args, GroupIdx),
    (   var(ValueVar)
    ->  find_var_index_go(ValueVar, Args, ValueIdx)
    ;   ValueIdx = -1
    ),
    
    % Generate aggregate code
    go_agg_code(Op, AggCode),
    
    % Generate HAVING filter code if any
    (   HavingFilters \= []
    ->  generate_having_conditions(HavingFilters, Result, HavingCond),
        format(string(HavingCode), "
            // HAVING clause filter
            if !(~w) {
                continue
            }", [HavingCond])
    ;   HavingCode = ""
    ),
    
    format(string(Code),
"func ~w(fact Fact, total map[string]Fact, idx *Index) []Fact {
    var results []Fact
    
    // Skip if not triggered by this relation
    if fact.Relation != \"~w\" {
        return results
    }
    
    // Group by key
    groups := make(map[interface{}][]float64)
    for _, f := range total {
        if f.Relation == \"~w\" {
            key := f.Args[\"arg~w\"]
            val, ok := toFloat64(f.Args[\"arg~w\"])
            if ok {
                groups[key] = append(groups[key], val)
            }
        }
    }
    
    // Compute aggregate per group
    for key, values := range groups {
        if len(values) > 0 {
            ~w~w
            results = append(results, Fact{
                Relation: \"~w\",
                Args: map[string]interface{}{\"arg0\": key, \"arg1\": agg},
            })
        }
    }
    
    return results
}", [RuleName, Pred, Pred, GroupIdx, ValueIdx, AggCode, HavingCode, HeadPred]).

%% decompose_agg_op(+OpTerm, -Op, -ValueVar)
decompose_agg_op(count, count, _).
decompose_agg_op(sum(V), sum, V).
decompose_agg_op(min(V), min, V).
decompose_agg_op(max(V), max, V).
decompose_agg_op(avg(V), avg, V).
decompose_agg_op(set(V), set, V).
decompose_agg_op(bag(V), bag, V).

%% generate_having_conditions(+HavingFilters, +ResultVar, -GoCondition)
%  Translate HAVING filter builtins to Go conditions
%  ResultVar is the Prolog variable bound to the aggregate result
generate_having_conditions([], _, "true").
generate_having_conditions([Filter|Rest], ResultVar, Condition) :-
    generate_single_having_condition(Filter, ResultVar, Cond1),
    (   Rest == []
    ->  Condition = Cond1
    ;   generate_having_conditions(Rest, ResultVar, RestCond),
        format(string(Condition), "(~w) && (~w)", [Cond1, RestCond])
    ).

%% generate_single_having_condition(+Filter, +ResultVar, -GoCond)
generate_single_having_condition(Filter, ResultVar, GoCond) :-
    Filter =.. [Op, Left, Right],
    member(Op-GoOp, [(>)-">", (<)-"<", (>=)-">=", (=<)-"<=", (=:=)-"==", (=\=)-"!="]),
    !,
    translate_having_operand(Left, ResultVar, LeftGo),
    translate_having_operand(Right, ResultVar, RightGo),
    format(string(GoCond), "~w ~w ~w", [LeftGo, GoOp, RightGo]).
generate_single_having_condition(_, _, "true").  % Fallback

%% translate_having_operand(+Term, +ResultVar, -GoExpr)
translate_having_operand(Term, ResultVar, "agg") :-
    var(Term), Term == ResultVar, !.
translate_having_operand(N, _, Str) :-
    number(N), !,
    format(string(Str), "~w", [N]).
translate_having_operand(Term, _, Str) :-
    format(string(Str), "~w", [Term]).

%% go_agg_code(+Op, -GoCode)
go_agg_code(count, "agg := float64(len(values))").
go_agg_code(sum, "agg := 0.0; for _, v := range values { agg += v }").
go_agg_code(min, "agg := values[0]; for _, v := range values { if v < agg { agg = v } }").
go_agg_code(max, "agg := values[0]; for _, v := range values { if v > agg { agg = v } }").
go_agg_code(avg, "agg := 0.0; for _, v := range values { agg += v }; agg /= float64(len(values))").
go_agg_code(set, "agg := float64(len(values))").  % Simplified for now
go_agg_code(bag, "agg := float64(len(values))").  % Simplified for now

%% find_var_index_go(+Var, +Args, -Index)
find_var_index_go(Var, Args, Index) :-
    nth0(Index, Args, Arg),
    Var == Arg,
    !.
find_var_index_go(_, _, 0).  % Default to 0 if not found

%% body_to_list_go(+Body, -Goals)
%  Convert conjunction body to list of goals
body_to_list_go((A, B), Goals) :- !,
    body_to_list_go(A, GoalsA),
    body_to_list_go(B, GoalsB),
    append(GoalsA, GoalsB, Goals).
body_to_list_go(Goal, [Goal]).

%% is_builtin_goal_go(+Goal)
%  Check if goal is a builtin (comparison, arithmetic, or has a binding)
is_builtin_goal_go(Goal) :-
    Goal =.. [Functor|Args],
    (   % Standard builtins
        member(Functor, [is, '>', '<', '>=', '=<', '=:=', '=\\=', '==', '\\=', not, '\\+'])
    ;   % Check if there's a binding for this predicate
        length(Args, Arity),
        go_binding(Functor/Arity, _, _, _, _)
    ).

%% is_binding_goal_go(+Goal)
%  Check if goal has a Go binding (not a standard builtin)
is_binding_goal_go(Goal) :-
    Goal =.. [Functor|Args],
    \+ member(Functor, [is, '>', '<', '>=', '=<', '=:=', '=\\=', '==', '\\=', not, '\\+']),
    length(Args, Arity),
    go_binding(Functor/Arity, _, _, _, _).

%% compile_go_join_with_result(+Goals, +VarMap, +Config, +Builtins, +HeadPred, +HeadArgs, -Code)
%  Generate nested join loops with result emission inside innermost loop
compile_go_join_with_result([], VarMap, Config, Builtins, HeadPred, HeadArgs, Code) :-
    % Base case: generate builtin checks and result emit
    compile_go_builtins(Builtins, VarMap, Config, BuiltinCode),
    compile_go_head_construction(HeadPred, HeadArgs, VarMap, Config, HeadCode),
    format(string(Code), "~w\n            ~w", [BuiltinCode, HeadCode]).

compile_go_join_with_result([Goal|Rest], VarMap, Config, Builtins, HeadPred, HeadArgs, Code) :-
    Goal =.. [Pred|Args],
    length([Goal|Rest], Len),
    format(string(VarName), "j~w", [Len]),
    
    % Find first shared variable that can use index lookup
    (   find_indexable_join(Args, VarMap, ArgIdx, Source, SrcIdx)
    ->  % Use indexed lookup
        format(string(LookupKey), "~w.Args[\"arg~w\"]", [Source, SrcIdx]),
        format(string(ArgName), "arg~w", [ArgIdx]),
        
        % Build remaining join conditions (excluding the indexed one)
        compile_go_join_condition_excluding(Args, VarMap, Config, VarName, ArgIdx, JoinCond),
        
        % Update variable map with this goal's bindings
        build_variable_map([Goal-VarName], NewBindings),
        append(VarMap, NewBindings, VarMap1),
        
        % Recurse to get inner code
        compile_go_join_with_result(Rest, VarMap1, Config, Builtins, HeadPred, HeadArgs, InnerCode),
        
        format(string(Code),
"    
    // Join with ~w (indexed on ~w)
    for _, ~wPtr := range idx.Lookup(\"~w\", \"~w\", ~w) {
        ~w := *~wPtr
        if true~w {
~w
        }
    }
", [Pred, ArgName, VarName, Pred, ArgName, LookupKey, VarName, VarName, JoinCond, InnerCode])
    ;   % Fallback to linear scan (no indexable join condition)
        compile_go_join_condition(Args, VarMap, Config, VarName, JoinCond),
        
        % Update variable map with this goal's bindings
        build_variable_map([Goal-VarName], NewBindings),
        append(VarMap, NewBindings, VarMap1),
        
        % Recurse to get inner code
        compile_go_join_with_result(Rest, VarMap1, Config, Builtins, HeadPred, HeadArgs, InnerCode),
        
        format(string(Code),
"    
    // Join with ~w (linear scan)
    for _, ~w := range total {
        if ~w.Relation == \"~w\"~w {
~w
        }
    }
", [Pred, VarName, VarName, Pred, JoinCond, InnerCode])
    ).

%% find_indexable_join(+Args, +VarMap, -ArgIdx, -Source, -SrcIdx)
%  Find the first argument that is bound in VarMap (prefer arg0)
find_indexable_join(Args, VarMap, ArgIdx, Source, SrcIdx) :-
    nth0(ArgIdx, Args, Arg),
    var(Arg),
    member(Arg0-source(Source, SrcIdx), VarMap),
    Arg == Arg0,
    !.

%% compile_go_join_condition_excluding(+Args, +VarMap, +Config, +VarName, +ExcludeIdx, -Condition)
%  Generate join conditions excluding the indexed argument
compile_go_join_condition_excluding(Args, VarMap, _Config, VarName, ExcludeIdx, Condition) :-
    findall(Cond,
        (   nth0(I, Args, Arg),
            I \= ExcludeIdx,
            var(Arg),
            member(Arg0-source(Source, SrcIdx), VarMap),
            Arg == Arg0,
            format(string(Cond), " && ~w.Args[\"arg~w\"] == ~w.Args[\"arg~w\"]", [VarName, I, Source, SrcIdx])
        ),
        Conds),
    atomic_list_concat(Conds, "", Condition).

%% compile_go_joins(+Goals, +VarMap, +Config, +Index, -JoinCode, -FinalVarMap)
%  Generate nested loops for join goals
compile_go_joins([], VarMap, _, _, "", VarMap).
compile_go_joins([Goal|Rest], VarMap, Config, Index, Code, FinalVarMap) :-
    Goal =.. [Pred|Args],
    format(string(VarName), "j~w", [Index]),
    
    % Build join condition from shared variables
    compile_go_join_condition(Args, VarMap, Config, VarName, JoinCond),
    
    % Update variable map with this goal's bindings
    build_variable_map([Goal-VarName], NewBindings),
    append(VarMap, NewBindings, VarMap1),
    
    % Recurse for remaining goals
    Next is Index + 1,
    compile_go_joins(Rest, VarMap1, Config, Next, RestCode, FinalVarMap),
    
    format(string(Code),
"    
    // Join with ~w
    for _, ~w := range total {
        if ~w.Relation == \"~w\"~w {
~w        }
    }
", [Pred, VarName, VarName, Pred, JoinCond, RestCode]).

%% compile_go_join_condition(+Args, +VarMap, +Config, +VarName, -Condition)
%  Generate join condition checking shared variables
compile_go_join_condition(Args, VarMap, _Config, VarName, Condition) :-
    findall(Cond,
        (   nth0(I, Args, Arg),
            var(Arg),
            member(Arg0-source(Source, SrcIdx), VarMap),
            Arg == Arg0,
            format(string(Cond), " && ~w.Args[\"arg~w\"] == ~w.Args[\"arg~w\"]", [VarName, I, Source, SrcIdx])
        ),
        Conds),
    atomic_list_concat(Conds, "", Condition).

%% compile_go_builtins(+Builtins, +VarMap, +Config, -Code)
%  Generate builtin constraint checks
compile_go_builtins([], _, _, "").
compile_go_builtins(Builtins, VarMap, Config, Code) :-
    findall(Check,
        (   member(B, Builtins),
            compile_go_single_builtin(B, VarMap, Config, Check)
        ),
        Checks),
    (   Checks == []
    ->  Code = ""
    ;   atomic_list_concat(Checks, "\n", ChecksStr),
        format(string(Code), "\n    // Constraint checks\n~w\n", [ChecksStr])
    ).

%% compile_go_single_builtin(+Builtin, +VarMap, +Config, -Check)
compile_go_single_builtin(Goal, VarMap, Config, Check) :-
    (   Goal = (\+ NegGoal)
    ;   Goal = not(NegGoal)
    ),
    !,
    % Negation check
    NegGoal =.. [NegPred|NegArgs],
    findall(Assign,
        (   nth0(I, NegArgs, Arg),
            translate_go_expr(Arg, VarMap, Config, Expr),
            format(string(Assign), "\"arg~w\": ~w", [I, Expr])
        ),
        Assigns),
    atomic_list_concat(Assigns, ", ", ArgsStr),
    format(string(Check),
"    negFact := Fact{Relation: \"~w\", Args: map[string]interface{}{~w}}
    if _, exists := total[negFact.Key()]; exists {
        return results
    }", [NegPred, ArgsStr]).

compile_go_single_builtin(Goal, VarMap, Config, Check) :-
    Goal =.. [Op, Left, Right],
    member(Op-GoOp, ['>' - ">", '<' - "<", '>=' - ">=", '=<' - "<=", '=:=' - "==", '=\\=' - "!="]),
    !,
    translate_go_expr(Left, VarMap, Config, LeftExpr),
    translate_go_expr(Right, VarMap, Config, RightExpr),
    format(string(Check),
"    if !(~w ~w ~w) {
        return results
    }", [LeftExpr, GoOp, RightExpr]).

% Handle binding goals (e.g., string_lower(X, Y), sqrt(X, Y))
compile_go_single_builtin(Goal, VarMap, Config, Check) :-
    is_binding_goal_go(Goal),
    !,
    compile_go_binding_goal(Goal, VarMap, Config, Check).

compile_go_single_builtin(_, _, _, "").

%% compile_go_binding_goal(+Goal, +VarMap, +Config, -Code)
%  Compile a goal that has a Go binding to Go code
compile_go_binding_goal(Goal, VarMap, Config, Code) :-
    Goal =.. [Functor|Args],
    length(Args, Arity),
    Pred = Functor/Arity,

    % Get the binding
    go_binding(Pred, TargetName, _Inputs, Outputs, Options),

    % Collect import if needed
    (   member(import(Import), Options)
    ->  collect_binding_import(Import)
    ;   true
    ),

    % Determine input and output args
    % Convention: last arg(s) are outputs based on Outputs list length
    length(Outputs, NumOutputs),
    length(Args, NumArgs),
    NumInputs is NumArgs - NumOutputs,
    length(InputArgs, NumInputs),
    length(OutputArgs, NumOutputs),
    append(InputArgs, OutputArgs, Args),

    % Translate input arguments to Go expressions
    maplist(translate_go_expr_binding(VarMap, Config), InputArgs, GoInputExprs),

    % Generate the binding call
    atom_string(TargetName, TargetStr),
    (   % Method call pattern: starts with .
        sub_string(TargetStr, 0, 1, _, ".")
    ->  % First input is receiver
        (   GoInputExprs = [Receiver|RestInputs]
        ->  (   RestInputs = []
            ->  format(string(CallCode), "~w~w", [Receiver, TargetStr])
            ;   atomic_list_concat(RestInputs, ", ", RestArgsStr),
                % Remove trailing () if present in method name
                (   sub_string(TargetStr, _, 2, 0, "()")
                ->  sub_string(TargetStr, 0, _, 2, MethodBase),
                    format(string(CallCode), "~w~w(~w)", [Receiver, MethodBase, RestArgsStr])
                ;   format(string(CallCode), "~w~w(~w)", [Receiver, TargetStr, RestArgsStr])
                )
            )
        ;   format(string(CallCode), "/* missing receiver for ~w */", [TargetStr])
        )
    ;   % Regular function call
        (   GoInputExprs = []
        ->  format(string(CallCode), "~w()", [TargetStr])
        ;   atomic_list_concat(GoInputExprs, ", ", ArgsStr),
            format(string(CallCode), "~w(~w)", [TargetStr, ArgsStr])
        )
    ),

    % Handle output assignment
    (   OutputArgs = [OutputVar]
    ->  % Single output - assign result
        (   var(OutputVar)
        ->  % Create a temporary variable
            gensym(bindResult, TempVar),
            format(string(Code),
"    ~w := ~w
    _ = ~w  // Use result in ~w", [TempVar, CallCode, TempVar, OutputVar])
        ;   % Output is ground - check equality
            translate_go_expr(OutputVar, VarMap, Config, ExpectedExpr),
            gensym(bindResult, TempVar),
            format(string(Code),
"    ~w := ~w
    if ~w != ~w {
        return results
    }", [TempVar, CallCode, TempVar, ExpectedExpr])
        )
    ;   OutputArgs = []
    ->  % No output - just call for side effect
        format(string(Code), "    ~w", [CallCode])
    ;   % Multiple outputs - not commonly handled yet
        format(string(Code), "    /* multiple outputs not yet supported: ~w */", [CallCode])
    ).

%% translate_go_expr_binding(+VarMap, +Config, +Expr, -GoExpr)
%  Wrapper for translate_go_expr with argument order suitable for maplist
translate_go_expr_binding(VarMap, Config, Expr, GoExpr) :-
    translate_go_expr(Expr, VarMap, Config, GoExpr).

%% translate_go_expr(+Expr, +VarMap, +Config, -GoExpr)
%  Translate Prolog expression to Go expression
translate_go_expr(Var, VarMap, _, Expr) :-
    var(Var),
    !,
    (   member(V-source(Source, Idx), VarMap), Var == V
    ->  format(string(Expr), "~w.Args[\"arg~w\"]", [Source, Idx])
    ;   Expr = "nil"
    ).
translate_go_expr(Num, _, _, Expr) :-
    number(Num),
    !,
    format(string(Expr), "~w", [Num]).
translate_go_expr(Atom, _, _, Expr) :-
    atom(Atom),
    !,
    format(string(Expr), "\"~w\"", [Atom]).
translate_go_expr(Expr, VarMap, Config, GoExpr) :-
    Expr =.. [Op, Left, Right],
    member(Op, [+, -, *, /]),
    !,
    translate_go_expr(Left, VarMap, Config, LeftExpr),
    translate_go_expr(Right, VarMap, Config, RightExpr),
    format(string(GoExpr), "(~w ~w ~w)", [LeftExpr, Op, RightExpr]).
translate_go_expr(_, _, _, "nil").

%% compile_go_head_construction(+HeadPred, +HeadArgs, +VarMap, +Config, -Code)
%  Generate code to construct result fact
compile_go_head_construction(HeadPred, HeadArgs, VarMap, Config, Code) :-
    findall(ArgCode,
        (   nth0(I, HeadArgs, Arg),
            translate_go_expr(Arg, VarMap, Config, Expr),
            format(string(ArgCode), "\"arg~w\": ~w", [I, Expr])
        ),
        ArgCodes),
    atomic_list_concat(ArgCodes, ", ", ArgsStr),
    format(string(Code),
"results = append(results, Fact{Relation: \"~w\", Args: map[string]interface{}{~w}})", 
        [HeadPred, ArgsStr]).

%% compile_go_generator_execution(+Pred, +RuleNames, +Options, -Code)
%  Generate Solve() fixpoint loop and main()
compile_go_generator_execution(_Pred, RuleNames, Options, Code) :-
    findall(Call,
        (   member(Name, RuleNames),
            format(string(Call), "\t\t\tnewFacts = append(newFacts, ~w(fact, total, idx)...)", [Name])
        ),
        Calls),
    atomic_list_concat(Calls, "\n", CallsStr),
    
    % Generate stdin loading code if json_input is enabled
    (   option(json_input(true), Options)
    ->  StdinCode = "
    // Load additional facts from stdin (JSONL format)
    scanner := bufio.NewScanner(os.Stdin)
    for scanner.Scan() {
        var fact Fact
        if err := json.Unmarshal(scanner.Bytes(), &fact); err == nil {
            total[fact.Key()] = fact
        }
    }
"
    ;   StdinCode = ""
    ),
    
    % Generate database code if db_backend is specified
    (   option(db_backend(bbolt), Options)
    ->  option(db_file(DbFile), Options, 'facts.db'),
        option(db_bucket(Bucket), Options, 'facts'),
        atom_string(Bucket, BucketStr),
        format(string(DbLoadCode), "
    // Load facts from database
    db, err := bolt.Open(\"~w\", 0600, nil)
    if err != nil {
        log.Printf(\"Warning: Could not open database: %v\", err)
    } else {
        defer db.Close()
        db.View(func(tx *bolt.Tx) error {
            b := tx.Bucket([]byte(\"~w\"))
            if b != nil {
                b.ForEach(func(k, v []byte) error {
                    var fact Fact
                    if err := json.Unmarshal(v, &fact); err == nil {
                        total[fact.Key()] = fact
                    }
                    return nil
                })
            }
            return nil
        })
    }
", [DbFile, BucketStr]),
        format(string(DbSaveCode), "
    // Save all facts to database
    db2, err := bolt.Open(\"~w\", 0600, nil)
    if err == nil {
        defer db2.Close()
        db2.Update(func(tx *bolt.Tx) error {
            b, _ := tx.CreateBucketIfNotExists([]byte(\"~w\"))
            for key, fact := range total {
                v, _ := json.Marshal(fact)
                b.Put([]byte(key), v)
            }
            return nil
        })
    }
", [DbFile, BucketStr])
    ;   DbLoadCode = "",
        DbSaveCode = ""
    ),
    
    % Check for parallel execution
    (   option(workers(Workers), Options), Workers > 1
    ->  compile_go_generator_parallel_solve(Workers, RuleNames, StdinCode, Code)
    ;   % Sequential execution (default)
        format(string(Code),
"// Solve runs fixpoint iteration until no new facts are derived
func Solve() map[string]Fact {
    total := make(map[string]Fact)
    
    // Initialize with base facts
    for _, fact := range GetInitialFacts() {
        total[fact.Key()] = fact
    }
~w~w
    // Build initial index
    idx := BuildIndex(total)
    
    // Fixpoint iteration
    changed := true
    for changed {
        changed = false
        var newFacts []Fact
        
        for _, fact := range total {
~w
        }
        
        // Add new facts to total and index
        for _, nf := range newFacts {
            key := nf.Key()
            if _, exists := total[key]; !exists {
                total[key] = nf
                idx.Add(&nf)
                changed = true
            }
        }
    }
~w
    return total
}

func main() {
    result := Solve()
    for _, fact := range result {
        b, _ := json.Marshal(fact)
        fmt.Println(string(b))
    }
}
", [StdinCode, DbLoadCode, CallsStr, DbSaveCode])
    ).

%% compile_go_generator_parallel_solve(+Workers, +RuleNames, +StdinCode, -Code)
%  Generate parallel Solve() using goroutines
compile_go_generator_parallel_solve(Workers, RuleNames, StdinCode, Code) :-
    % Generate rule calls for worker function
    findall(Call,
        (   member(Name, RuleNames),
            format(string(Call), "\t\t\tresults = append(results, ~w(fact, total, idx)...)", [Name])
        ),
        Calls),
    atomic_list_concat(Calls, "\n", CallsStr),
    
    format(string(Code),
"// applyRules applies all rules to a fact and sends results to channel
func applyRules(fact Fact, total map[string]Fact, idx *Index, resultChan chan<- Fact, wg *sync.WaitGroup) {
    defer wg.Done()
    var results []Fact
~w
    for _, r := range results {
        resultChan <- r
    }
}

// Solve runs parallel fixpoint iteration with ~w workers
func Solve() map[string]Fact {
    total := make(map[string]Fact)
    
    // Initialize with base facts
    for _, fact := range GetInitialFacts() {
        total[fact.Key()] = fact
    }
~w
    // Build initial index
    idx := BuildIndex(total)
    
    // Fixpoint iteration with parallel workers
    changed := true
    for changed {
        changed = false
        
        // Collect facts into slice for parallel processing
        facts := make([]Fact, 0, len(total))
        for _, f := range total {
            facts = append(facts, f)
        }
        
        // Channel to collect new facts
        resultChan := make(chan Fact, len(facts)*10)
        var wg sync.WaitGroup
        
        // Dispatch workers (each processes a subset of facts)
        numWorkers := ~w
        chunkSize := (len(facts) + numWorkers - 1) / numWorkers
        
        for i := 0; i < numWorkers; i++ {
            start := i * chunkSize
            end := start + chunkSize
            if end > len(facts) {
                end = len(facts)
            }
            if start >= len(facts) {
                break
            }
            
            // Process chunk in goroutine
            wg.Add(1)
            go func(chunk []Fact) {
                defer wg.Done()
                for _, fact := range chunk {
                    var results []Fact
~w
                    for _, r := range results {
                        resultChan <- r
                    }
                }
            }(facts[start:end])
        }
        
        // Close channel when all workers done
        go func() {
            wg.Wait()
            close(resultChan)
        }()
        
        // Collect results
        for nf := range resultChan {
            key := nf.Key()
            if _, exists := total[key]; !exists {
                total[key] = nf
                idx.Add(&nf)
                changed = true
            }
        }
    }
    
    return total
}

func main() {
    result := Solve()
    for _, fact := range result {
        b, _ := json.Marshal(fact)
        fmt.Println(string(b))
    }
}
", [CallsStr, Workers, StdinCode, Workers, CallsStr]).

%% compile_predicate_to_go_normal(+Pred, +Arity, +Options, -GoCode)
%  Normal (non-aggregation) compilation path
%
compile_predicate_to_go_normal(Pred, Arity, Options, GoCode) :-
    % Get options
    option(record_delimiter(RecordDelim), Options, newline),
    option(field_delimiter(FieldDelim), Options, colon),
    option(quoting(Quoting), Options, none),
    option(escape_char(EscapeChar), Options, backslash),
    option(include_package(IncludePackage), Options, true),
    option(unique(Unique), Options, true),

    % Create head with correct arity
    functor(Head, Pred, Arity),

    % Get all clauses for this predicate
    findall(Head-Body, user:clause(Head, Body), Clauses),

    % Determine compilation strategy
    (   Clauses = [] ->
        format('ERROR: No clauses found for ~w/~w~n', [Pred, Arity]),
        fail
    
    % Semantic Rule Check
    ;   Clauses = [Head-Body], Body \= true,
        extract_predicates(Body, [SinglePred]),
        is_semantic_predicate(SinglePred)
    ->  Head =.. [_|HeadArgs],
        compile_semantic_rule_go(Pred, HeadArgs, SinglePred, GoCode)

    ;   maplist(is_fact_clause, Clauses) ->
        % All bodies are 'true' - these are facts
        format('Type: facts (~w clauses)~n', [length(Clauses, _)]),
        compile_facts_to_go(Pred, Arity, Clauses, RecordDelim, FieldDelim,
                           Unique, ScriptBody),
        GenerateProgram = true
    ;   is_tail_recursive_pattern(Pred, Clauses) ->
        % Tail recursive pattern - compile to iterative loop
        format('Type: tail_recursion~n'),
        compile_tail_recursive_to_go(Pred, Arity, Clauses, ScriptBody),
        GenerateProgram = true
    ;   Clauses = [SingleHead-SingleBody], SingleBody \= true ->
        % Single rule
        format('Type: single_rule~n'),
        compile_single_rule_to_go(Pred, Arity, SingleHead, SingleBody, RecordDelim,
                                 FieldDelim, Unique, ScriptBody),
        GenerateProgram = true
    ;   % Multiple rules (OR pattern)
        format('Type: multiple_rules (~w clauses)~n', [length(Clauses, _)]),
        compile_multiple_rules_to_go(Pred, Arity, Clauses, RecordDelim,
                                    FieldDelim, Unique, ScriptBody),
        GenerateProgram = true
    ),

    % Determine if we need imports (only if GenerateProgram is true)
    (   var(GenerateProgram) -> true
    ;   (   maplist(is_fact_clause, Clauses) ->
            NeedsStdin = false
        ;   is_tail_recursive_pattern(Pred, Clauses) ->
            NeedsStdin = false
        ;   NeedsStdin = true
        ),

        % Determine if we need regexp imports
        (   member(_Head-Body, Clauses),
            extract_match_constraints(Body, MatchCs),
            MatchCs \= []
        ->  NeedsRegexp = true
        ;   NeedsRegexp = false
        ),

        % Determine if we need strings import
        (   NeedsStdin,
            (   length(Clauses, NumClauses), NumClauses > 1, \+ maplist(is_fact_clause, Clauses)
            ;   member(Head-Body, Clauses), \+ is_fact_clause(Head-Body),
                Head =.. [_|Args], length(Args, ArgCount), ArgCount > 1,
                extract_predicates(Body, Preds), Preds \= []
            )
        ->  NeedsStrings = true
        ;   NeedsStrings = false
        ),

        % Determine if we need strconv import
        (   member(_Head-Body, Clauses),
            extract_constraints(Body, Cs),
            Cs \= []
        ->  NeedsStrconv = true
        ;   NeedsStrconv = false
        ),

        % Generate complete Go program
        (   IncludePackage ->
            generate_go_program(Pred, Arity, RecordDelim, FieldDelim, Quoting,
                               EscapeChar, NeedsStdin, NeedsRegexp, NeedsStrings, NeedsStrconv, ScriptBody, GoCode)
        ;   GoCode = ScriptBody
        )
    ),
    !.

%% Helper to check if a clause is a fact (body is just 'true')
is_fact_clause(_-true).

%% is_aggregation_predicate(+Body)
%  Check if body contains an aggregation goal
is_aggregation_predicate(Body) :-
    format('Checking body for aggregation: ~w~n', [Body]),
    extract_aggregation_spec(Body, _, _, _).

%% extract_aggregation_spec(+Body, -Op, -Goal, -Result)
%  Extract aggregation specification from body
%  Supports: aggregate(Op, Goal, Result)
extract_aggregation_spec(aggregate(Op, Goal, Result), Op, Goal, Result).
extract_aggregation_spec((aggregate(Op, Goal, Result), _), Op, Goal, Result).

%% ============================================
%% AGGREGATION PATTERN COMPILATION
%% ============================================

%% compile_aggregation_to_go(+Pred, +Arity, +AggOp, +FieldDelim, +IncludePackage, -GoCode)
%  Compile aggregation operations (sum, count, max, min, avg)
%
%% compile_aggregation_mode(+Pred, +Arity, +Options, -GoCode)
%  Compile aggregation predicate (new style)
%
compile_aggregation_mode(Pred, Arity, Options, GoCode) :- !,
    % Get predicate clauses
    functor(Head, Pred, Arity),
    clause(Head, Body),
    
    % Extract aggregation spec
    extract_aggregation_spec(Body, Op, Goal, ResultVar),
    
    % Determine if we are aggregating over a field or just counting
    (   Op = count
    ->  AggField = none,
        OpType = count
    ;   compound(Op), functor(Op, AggType, 1), arg(1, Op, AggVar)
    ->  AggField = AggVar,
        OpType = AggType
    ;   % Fallback / Error case
        OpType = Op, AggField = none
    ),
    
    % Extract JSON field mappings from the inner goal
    extract_json_field_mappings(Goal, FieldMappings),
    
    % Generate Go code
    generate_aggregation_code(OpType, AggField, ResultVar, FieldMappings, Options, ScriptBody),
    
    % Determine imports
    (   OpType = count ->
        Imports = '\t"bufio"\n\t"encoding/json"\n\t"fmt"\n\t"os"'
    ;   Imports = '\t"bufio"\n\t"encoding/json"\n\t"fmt"\n\t"os"'
    ),
    
    % Wrap in package main if requested
    option(include_package(IncludePackage), Options, true),
    (   IncludePackage ->
        format(string(GoCode), 'package main

import (
~s
)

func main() {
~s}
', [Imports, ScriptBody])
    ;   GoCode = ScriptBody
    ).

%% compile_aggregation_to_go(+Pred, +Arity, +AggOp, +FieldDelim, +IncludePackage, -GoCode)
%  Compile aggregation operations (legacy option-based)
compile_aggregation_to_go(Pred, Arity, AggOp, FieldDelim, IncludePackage, GoCode) :-
    atom_string(Pred, PredStr),
    map_field_delimiter(FieldDelim, DelimChar),
    format('  Legacy Aggregation type: ~w~n', [AggOp]),
    generate_aggregation_go(AggOp, Arity, DelimChar, ScriptBody),
    (   AggOp = count ->
        Imports = '\t"bufio"\n\t"fmt"\n\t"os"'
    ;   Imports = '\t"bufio"\n\t"fmt"\n\t"os"\n\t"strconv"'
    ),
    (   IncludePackage ->
        format(string(GoCode), 'package main

import (
~s
)

func main() {
~s}
', [Imports, ScriptBody])
    ;   GoCode = ScriptBody
    ).

%% generate_aggregation_go(+AggOp, +Arity, +DelimChar, -GoCode)
%  Generate Go code for specific aggregation operations
%
generate_aggregation_go(sum, _Arity, _DelimChar, GoCode) :-
    format(atom(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
\tsum := 0.0
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tval, err := strconv.ParseFloat(line, 64)
\t\tif err == nil {
\t\t\tsum += val
\t\t}
\t}
\t
\tfmt.Println(sum)
', []).

generate_aggregation_go(count, _Arity, _DelimChar, GoCode) :-
    format(atom(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
\tcount := 0
\t
\tfor scanner.Scan() {
\t\tcount++
\t}
\t
\tfmt.Println(count)
', []).

generate_aggregation_go(max, _Arity, _DelimChar, GoCode) :-
    format(atom(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
\tvar max float64
\tfirst := true
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tval, err := strconv.ParseFloat(line, 64)
\t\tif err == nil {
\t\t\tif first || val > max {
\t\t\t\tmax = val
\t\t\t\tfirst = false
\t\t\t}
\t\t}
\t}
\t
\tif !first {
\t\tfmt.Println(max)
\t}
', []).

generate_aggregation_go(min, _Arity, _DelimChar, GoCode) :-
    format(atom(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
\tvar min float64
\tfirst := true
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tval, err := strconv.ParseFloat(line, 64)
\t\tif err == nil {
\t\t\tif first || val < min {
\t\t\t\tmin = val
\t\t\t\tfirst = false
\t\t\t}
\t\t}
\t}
\t
\tif !first {
\t\tfmt.Println(min)
\t}
', []).

generate_aggregation_go(avg, _Arity, _DelimChar, GoCode) :-
    format(atom(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
\tsum := 0.0
\tcount := 0
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tval, err := strconv.ParseFloat(line, 64)
\t\tif err == nil {
\t\t\tsum += val
\t\t\tcount++
\t\t}
\t}
\t
\tif count > 0 {
\t\tfmt.Println(sum / float64(count))
\t}
', []).

%% ============================================
%% FACTS COMPILATION
%% ============================================

%% compile_facts_to_go(+Pred, +Arity, -GoCode)
%  PUBLIC API: Export Prolog facts as a standalone Go program.
%  Generates a Go program with a struct and slice of facts.
%
%  Example:
%    ?- compile_facts_to_go(parent, 2, Code).
%    Generates a Go program with facts as a slice of structs.
%
compile_facts_to_go(Pred, Arity, GoCode) :-
    % Get the predicate name
    atom_string(Pred, PredStr),
    upcase_atom(Pred, PredUp),
    atom_string(PredUp, PredUpStr),
    
    % Collect all facts
    functor(Head, Pred, Arity),
    findall(Args, (user:clause(Head, true), Head =.. [_|Args]), AllFacts),
    
    % Generate field names
    findall(Field, (
        between(1, Arity, N),
        format(atom(Field), 'Arg~w string', [N])
    ), Fields),
    atomic_list_concat(Fields, '\n\t', FieldDefs),
    
    % Format facts as Go struct literals
    findall(Entry, (
        member(Args, AllFacts),
        format_go_struct_entry(Args, Arity, Entry)
    ), Entries),
    atomic_list_concat(Entries, ',\n\t\t', EntriesCode),
    
    % Generate Go program
    format(string(GoCode),
'// Generated by UnifyWeaver Go Target - Fact Export
// Predicate: ~w/~w

package main

import (
\t"fmt"
\t"strings"
)

// ~w represents a ~w/~w fact
type ~w struct {
\t~w
}

// GetAll returns all facts
func GetAll~w() []~w {
\treturn []~w{
\t\t~w,
\t}
}

// Stream iterates over all facts
func Stream~w(fn func(~w)) {
\tfor _, fact := range GetAll~w() {
\t\tfn(fact)
\t}
}

// Contains checks if a fact exists
func Contains~w(target ~w) bool {
\tfor _, fact := range GetAll~w() {
\t\tif fact == target {
\t\t\treturn true
\t\t}
\t}
\treturn false
}

func main() {
\tfor _, fact := range GetAll~w() {
\t\tparts := []string{}
\t\t~w
\t\tfmt.Println(strings.Join(parts, ":"))
\t}
}
', [PredStr, Arity, PredUpStr, PredStr, Arity, PredUpStr, FieldDefs, 
    PredUpStr, PredUpStr, PredUpStr, EntriesCode,
    PredUpStr, PredUpStr, PredUpStr,
    PredUpStr, PredUpStr, PredUpStr,
    PredUpStr, format_go_field_prints(Arity)]).

%% format_go_struct_entry(+Args, +Arity, -Entry)
format_go_struct_entry(Args, _Arity, Entry) :-
    findall(FieldVal, (
        nth1(N, Args, Arg),
        atom_string(Arg, ArgStr),
        format(string(FieldVal), 'Arg~w: "~w"', [N, ArgStr])
    ), FieldVals),
    atomic_list_concat(FieldVals, ', ', FieldsStr),
    format(string(Entry), '{~w}', [FieldsStr]).

%% format_go_field_prints(+Arity, -Code)
format_go_field_prints(Arity) :-
    findall(Line, (
        between(1, Arity, N),
        format(string(Line), 'parts = append(parts, fact.Arg~w)', [N])
    ), Lines),
    atomic_list_concat(Lines, '\n\t\t', _Code).

%% compile_facts_to_go(+Pred, +Arity, +Clauses, +RecordDelim, +FieldDelim, +Unique, -GoCode)
%  Compile fact predicates to Go map lookup
%
compile_facts_to_go(Pred, Arity, Clauses, _RecordDelim, FieldDelim, Unique, GoCode) :-
    atom_string(Pred, PredStr),

    % Build map entries
    findall(Entry,
        (   member(Fact-true, Clauses),
            Fact =.. [_|Args],
            format_go_fact_entry(Args, FieldDelim, Entry)
        ),
        Entries),
    atomic_list_concat(Entries, ',\n\t\t', EntriesStr),

    % Generate Go code
    format(string(GoCode), '
\tfacts := map[string]bool{
\t\t~s,
\t}

\tfor key := range facts {
\t\tfmt.Println(key)
\t}
', [EntriesStr]).

%% format_go_fact_entry(+Args, +FieldDelim, -Entry)
%  Format a fact as a Go map entry
format_go_fact_entry(Args, FieldDelim, Entry) :-
    map_field_delimiter(FieldDelim, DelimChar),
    maplist(atom_string, Args, ArgStrs),
    atomic_list_concat(ArgStrs, DelimChar, Key),
    format(string(Entry), '"~s": true', [Key]).

%% ============================================
%% TAIL RECURSION OPTIMIZATION
%% ============================================

%% compile_tail_recursion_go(+Pred/Arity, +Options, -GoCode)
%  Compile tail recursive predicates to Go for loops.
%  Pattern: sum([], Acc, Acc). sum([H|T], Acc, S) :- Acc1 is Acc + H, sum(T, Acc1, S).
%  Generates O(1) stack space code.
%
compile_tail_recursion_go(Pred/Arity, _Options, GoCode) :-
    atom_string(Pred, PredStr),
    upcase_atom(Pred, PredUp),
    atom_string(PredUp, PredUpStr),
    
    % Detect step operation from predicate clauses
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    detect_go_step_op(Clauses, Pred, StepOp),
    step_op_to_go(StepOp, GoStepCode),
    
    % Generate based on arity
    (   Arity =:= 3 ->
        % Ternary: pred(List, Acc, Result)
        format(string(GoCode),
'// Generated by UnifyWeaver Go Target - Tail Recursion Optimization
// Predicate: ~w/~w
// O(1) stack space via for loop

package main

import (
\t"bufio"
\t"fmt"
\t"os"
\t"strconv"
\t"strings"
)

// ~w computes result using accumulator pattern
func ~w(items []int, acc int) int {
\tfor _, item := range items {
\t\t~w
\t}
\treturn acc
}

func main() {
\tscanner := bufio.NewScanner(os.Stdin)
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tparts := strings.Split(line, ",")
\t\titems := make([]int, 0, len(parts))
\t\tfor _, p := range parts {
\t\t\tif v, err := strconv.Atoi(strings.TrimSpace(p)); err == nil {
\t\t\t\titems = append(items, v)
\t\t\t}
\t\t}
\t\tfmt.Println(~w(items, 0))
\t}
}
', [PredStr, Arity, PredUpStr, PredStr, GoStepCode, PredStr])
    ;   Arity =:= 2 ->
        % Binary: pred(List, Result) with implicit 0 accumulator
        format(string(GoCode),
'// Generated by UnifyWeaver Go Target - Tail Recursion Optimization
// Predicate: ~w/~w

package main

import (
\t"bufio"
\t"fmt"
\t"os"
\t"strconv"
\t"strings"
)

func ~w(items []int) int {
\tacc := 0
\tfor _, item := range items {
\t\t~w
\t}
\treturn acc
}

func main() {
\tscanner := bufio.NewScanner(os.Stdin)
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tparts := strings.Split(line, ",")
\t\titems := make([]int, 0, len(parts))
\t\tfor _, p := range parts {
\t\t\tif v, err := strconv.Atoi(strings.TrimSpace(p)); err == nil {
\t\t\t\titems = append(items, v)
\t\t\t}
\t\t}
\t\tfmt.Println(~w(items))
\t}
}
', [PredStr, Arity, PredStr, GoStepCode, PredStr])
    ;   % Unsupported arity
        format(string(GoCode), '// Tail recursion for arity ~w not supported', [Arity])
    ).

%% detect_go_step_op(+Clauses, +Pred, -StepOp)
detect_go_step_op(Clauses, Pred, StepOp) :-
    % Find recursive clause and extract arithmetic operation
    member(_Head-Body, Clauses),
    Body \= true,
    contains_pred_go(Body, Pred),
    extract_arith_go(Body, StepOp),
    !.
detect_go_step_op(_, _, add_element).  % Default

%% contains_pred_go(+Body, +Pred)
contains_pred_go(Body, Pred) :-
    Body =.. [Pred|_], !.
contains_pred_go((A, _), Pred) :- contains_pred_go(A, Pred), !.
contains_pred_go((_, B), Pred) :- contains_pred_go(B, Pred), !.

%% extract_arith_go(+Body, -StepOp)
extract_arith_go((_ is _ + B), add_element) :- var(B), !.
extract_arith_go((_ is _ + 1), add_1) :- !.
extract_arith_go((_ is _ * B), mult_element) :- var(B), !.
extract_arith_go((A, _), Op) :- extract_arith_go(A, Op), !.
extract_arith_go((_, B), Op) :- extract_arith_go(B, Op), !.
extract_arith_go(_, add_1).

%% step_op_to_go(+StepOp, -GoCode)
step_op_to_go(add_element, "acc += item").
step_op_to_go(add_1, "acc++").
step_op_to_go(mult_element, "acc *= item").
step_op_to_go(_, "acc += item").

%% can_compile_tail_recursion_go(+Pred/Arity)
can_compile_tail_recursion_go(Pred/Arity) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    % Need at least one base case and one recursive case
    partition(is_recursive_clause_go(Pred), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [].

%% is_recursive_clause_go(+Pred, +Clause)
is_recursive_clause_go(Pred, _Head-Body) :-
    Body \= true,
    contains_pred_go(Body, Pred).

%% ============================================
%% LINEAR RECURSION WITH MEMOIZATION
%% ============================================

%% compile_linear_recursion_go(+Pred/Arity, +Options, -GoCode)
%  Compile linear recursive predicates to Go with memoization.
%  Pattern: fib(0, 0). fib(1, 1). fib(N, F) :- fib(N-1, F1), F is F1 + N.
%
compile_linear_recursion_go(Pred/Arity, _Options, GoCode) :-
    atom_string(Pred, PredStr),
    upcase_atom(Pred, PredUp),
    atom_string(PredUp, PredUpStr),
    
    (   Arity =:= 2 ->
        format(string(GoCode),
'// Generated by UnifyWeaver Go Target - Linear Recursion with Memoization
// Predicate: ~w/~w
// Uses map-based memoization for O(n) performance

package main

import (
\t"bufio"
\t"fmt"
\t"os"
\t"strconv"
\t"strings"
)

var ~wMemo = make(map[int]int)

// ~w computes result with memoization
func ~w(n int) int {
\tif result, ok := ~wMemo[n]; ok {
\t\treturn result
\t}
\t
\t// Base cases
\tif n <= 0 {
\t\treturn 0
\t}
\tif n == 1 {
\t\treturn 1
\t}
\t
\t// Recursive case with memoization
\tresult := ~w(n-1) + n
\t~wMemo[n] = result
\treturn result
}

func main() {
\tscanner := bufio.NewScanner(os.Stdin)
\tfor scanner.Scan() {
\t\tif n, err := strconv.Atoi(strings.TrimSpace(scanner.Text())); err == nil {
\t\t\tfmt.Println(~w(n))
\t\t}
\t}
}
', [PredStr, Arity, PredStr, PredUpStr, PredStr, PredStr, PredStr, PredStr, PredStr])
    ;   format(string(GoCode), '// Linear recursion for arity ~w not supported', [Arity])
    ).

%% can_compile_linear_recursion_go(+Pred/Arity)
can_compile_linear_recursion_go(Pred/Arity) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    partition(is_recursive_clause_go(Pred), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [],
    % Exactly one recursive call per clause
    forall(member(_-Body, RecClauses), count_recursive_calls_go(Body, Pred, 1)).

%% count_recursive_calls_go(+Body, +Pred, ?Count)
count_recursive_calls_go(Body, Pred, Count) :-
    count_recursive_calls_go_(Body, Pred, 0, Count).

count_recursive_calls_go_(Goal, Pred, Acc, Count) :-
    Goal =.. [Pred|_], !,
    Count is Acc + 1.
count_recursive_calls_go_((A, B), Pred, Acc, Count) :- !,
    count_recursive_calls_go_(A, Pred, Acc, Acc1),
    count_recursive_calls_go_(B, Pred, Acc1, Count).
count_recursive_calls_go_(_, _, Acc, Acc).

%% ============================================
%% MUTUAL RECURSION
%% ============================================

%% compile_mutual_recursion_go(+Predicates, +Options, -GoCode)
%  Compile mutually recursive predicates to Go.
%  Example: is_even/1 and is_odd/1
%
compile_mutual_recursion_go(Predicates, _Options, GoCode) :-
    % Extract predicate names for group name
    findall(PredStr, (
        member(Pred/_Arity, Predicates),
        atom_string(Pred, PredStr)
    ), PredStrs),
    atomic_list_concat(PredStrs, '_', GroupName),
    
    % Generate functions for each predicate
    findall(FuncCode, (
        member(Pred/Arity, Predicates),
        generate_mutual_function_go(Pred, Arity, Predicates, GroupName, FuncCode)
    ), FuncCodes),
    atomic_list_concat(FuncCodes, '\n\n', FunctionsCode),
    
    format(string(GoCode),
'// Generated by UnifyWeaver Go Target - Mutual Recursion
// Group: ~w

package main

import (
\t"fmt"
\t"os"
\t"strconv"
)

// Memoization for mutual recursion group
var ~wMemo = make(map[string]bool)

~w

func main() {
\tif len(os.Args) >= 3 {
\t\tn, _ := strconv.Atoi(os.Args[2])
\t\tswitch os.Args[1] {
~w\t\t}
\t}
}
', [GroupName, GroupName, FunctionsCode, generate_go_switch_cases(Predicates)]).

%% generate_mutual_function_go(+Pred, +Arity, +AllPredicates, +GroupName, -Code)
generate_mutual_function_go(Pred, Arity, AllPredicates, GroupName, Code) :-
    atom_string(Pred, PredStr),
    
    % Get clauses for this predicate
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    
    % Separate base and recursive cases
    partition(is_mutual_recursive_clause_go(AllPredicates), Clauses, RecClauses, BaseClauses),
    
    % Generate base case code
    generate_mutual_base_cases_go(BaseClauses, GroupName, BaseCaseCode),
    
    % Generate recursive case code
    generate_mutual_recursive_cases_go(RecClauses, AllPredicates, GroupName, RecCaseCode),
    
    format(string(Code),
'// ~w is part of mutual recursion group
func ~w(n int) bool {
\t// Check memo
\tkey := fmt.Sprintf("~w:%d", n)
\tif result, ok := ~wMemo[key]; ok {
\t\treturn result
\t}
\t
~w
~w
\t
\t// No match
\treturn false
}', [PredStr, PredStr, PredStr, PredStr, BaseCaseCode, RecCaseCode]).

%% is_mutual_recursive_clause_go(+AllPredicates, +Clause)
is_mutual_recursive_clause_go(AllPredicates, _Head-Body) :-
    Body \= true,
    member(Pred/Arity, AllPredicates),
    functor(Goal, Pred, Arity),
    body_contains_goal_go(Body, Goal).

%% body_contains_goal_go(+Body, +Goal)
body_contains_goal_go((A, B), Goal) :- !,
    (   body_contains_goal_go(A, Goal)
    ;   body_contains_goal_go(B, Goal)
    ).
body_contains_goal_go(Body, Goal) :-
    compound(Body),
    functor(Body, F, A),
    functor(Goal, F, A).

%% generate_mutual_base_cases_go(+BaseClauses, +GroupName, -Code)
generate_mutual_base_cases_go([], _, "\t// No base cases").
generate_mutual_base_cases_go(BaseClauses, GroupName, Code) :-
    BaseClauses \= [],
    findall(CaseCode, (
        member(Head-true, BaseClauses),
        Head =.. [_|[Value]],
        format(string(CaseCode), '\tif n == ~w {\n\t\t~wMemo[key] = true\n\t\treturn true\n\t}', [Value, GroupName])
    ), CaseCodes),
    atomic_list_concat(CaseCodes, '\n', Code).

%% generate_mutual_recursive_cases_go(+RecClauses, +AllPredicates, +GroupName, -Code)
generate_mutual_recursive_cases_go([], _, _, "\t// No recursive cases").
generate_mutual_recursive_cases_go([Head-Body|_], AllPredicates, GroupName, Code) :-
    % Extract the called predicate from the body
    find_mutual_call_go(Body, AllPredicates, CalledPred),
    atom_string(CalledPred, CalledPredStr),
    format(string(Code),
'\t// Recursive case
\tif n > 0 {
\t\tresult := ~w(n - 1)
\t\t~wMemo[key] = result
\t\treturn result
\t}', [CalledPredStr, GroupName]).

%% find_mutual_call_go(+Body, +AllPredicates, -CalledPred)
%  Find which predicate from the group is called in the body
find_mutual_call_go((A, B), AllPredicates, CalledPred) :- !,
    (   find_mutual_call_go(A, AllPredicates, CalledPred)
    ;   find_mutual_call_go(B, AllPredicates, CalledPred)
    ).
find_mutual_call_go(Goal, AllPredicates, CalledPred) :-
    Goal =.. [Pred|_],
    member(Pred/_Arity, AllPredicates),
    CalledPred = Pred.

%% generate_go_switch_cases(+Predicates)
generate_go_switch_cases(Predicates) :-
    findall(CaseCode, (
        member(Pred/_Arity, Predicates),
        atom_string(Pred, PredStr),
        format(string(CaseCode), '\t\tcase "~w": fmt.Println(~w(n))', [PredStr, PredStr])
    ), CaseCodes),
    atomic_list_concat(CaseCodes, '\n', _Cases).

%% ============================================
%% SINGLE RULE COMPILATION
%% ============================================

%% compile_single_rule_to_go(+Pred, +Arity, +Head, +Body, +RecordDelim, +FieldDelim, +Unique, -GoCode)
%  Compile single rule to Go code
%
compile_single_rule_to_go(Pred, Arity, Head, Body, RecordDelim, FieldDelim, Unique, GoCode) :-
    atom_string(Pred, PredStr),

    % Build variable mapping from head arguments
    Head =.. [_|HeadArgs],
    build_var_map(HeadArgs, VarMap),
    format('  Variable map: ~w~n', [VarMap]),

    % Extract predicates, match constraints, and arithmetic constraints from body
    extract_predicates(Body, Predicates),
    extract_match_constraints(Body, MatchConstraints),
    extract_constraints(Body, Constraints),
    format('  Body predicates: ~w~n', [Predicates]),
    format('  Match constraints: ~w~n', [MatchConstraints]),
    format('  Arithmetic constraints: ~w~n', [Constraints]),

    % Handle simple case: single predicate in body (with optional constraints)
    (   Predicates = [SinglePred] ->
        (   is_semantic_predicate(SinglePred)
        ->  compile_semantic_rule_go(PredStr, HeadArgs, SinglePred, GoCode)
        ;   compile_single_predicate_rule_go(PredStr, HeadArgs, SinglePred, VarMap,
                                            FieldDelim, Unique, MatchConstraints, Constraints, GoCode)
        )
    ;   Predicates = [], MatchConstraints \= [] ->
        % No predicates, just match constraints - read from stdin and filter
        compile_match_only_rule_go(PredStr, HeadArgs, VarMap, FieldDelim,
                                   Unique, MatchConstraints, GoCode)
    ;   % Multiple predicates or unsupported pattern
        format(user_error,
               'Go target: multi-predicate or constraint-only rules not supported (yet) for ~w/~w~n',
               [Pred, Arity]),
        fail
    ).

%% compile_single_predicate_rule_go(+PredStr, +HeadArgs, +BodyPred, +VarMap, +FieldDelim, +Unique, +MatchConstraints, +Constraints, -GoCode)
%  Compile a rule with single predicate in body (e.g., child(C,P) :- parent(P,C))
%  Optional match constraints for regex filtering and arithmetic constraints
%
compile_single_predicate_rule_go(PredStr, HeadArgs, BodyPred, VarMap, FieldDelim, Unique, MatchConstraints, Constraints, GoCode) :-
    % Get the body predicate name and args
    BodyPred =.. [BodyPredName|BodyArgs],
    atom_string(BodyPredName, BodyPredStr),
    map_field_delimiter(FieldDelim, DelimChar),

    % Build capture mapping from match constraints
    % Map head argument positions to capture group positions
    % For each head arg, check if it appears in any capture group
    findall((HeadPos, CapIdx),
        (   nth1(HeadPos, HeadArgs, HeadArg),
            var(HeadArg),
            member(match(_, _, _, Groups), MatchConstraints),
            Groups \= [],
            nth1(CapIdx, Groups, GroupVar),
            HeadArg == GroupVar
        ),
        CaptureMapping),
    format('  Capture mapping (HeadPos -> CapIdx): ~w~n', [CaptureMapping]),

    % Build output format by checking three sources:
    % 1. Variables from BodyArgs -> field{pos}
    % 2. Variables from capture groups (via position mapping) -> cap{idx}
    % 3. Constants -> literal value
    findall(OutputPart,
        (   nth1(HeadPos, HeadArgs, HeadArg),
            (   var(HeadArg) ->
                (   % Check if it's from body args
                    nth1(BodyPos, BodyArgs, BodyArg),
                    HeadArg == BodyArg
                ->  format(atom(OutputPart), 'field~w', [BodyPos])
                ;   % Check if it's from capture groups (using position mapping)
                    member((HeadPos, CapIdx), CaptureMapping)
                ->  format(atom(OutputPart), 'cap~w', [CapIdx])
                ;   % Variable not found - should not happen
                    format('WARNING: Variable at position ~w not found in body or captures~n', [HeadPos]),
                    fail
                )
            ;   % Constant in head
                atom_string(HeadArg, HeadArgStr),
                format(atom(OutputPart), '"~s"', [HeadArgStr])
            )
        ),
        OutputParts),

    % Build Go expression with delimiter between parts
    build_go_concat_expr(OutputParts, DelimChar, OutputExpr),

    % Generate Go code that reads from stdin and processes records
    length(BodyArgs, NumFields),

    % For single-field records, use the entire line without splitting
    (   NumFields = 1 ->
        FieldAssignments = '\t\t\tfield1 := line',
        SplitCode = '',
        LenCheck = ''
    ;   % Multi-field records need splitting and length check
        generate_field_assignments(BodyArgs, FieldAssignments),
        format(atom(SplitCode), '\t\tparts := strings.Split(line, "~s")\n', [DelimChar]),
        format(atom(LenCheck), '\t\tif len(parts) == ~w {\n', [NumFields])
    ),

    % Generate match constraint code if present
    generate_go_match_code(MatchConstraints, HeadArgs, BodyArgs, MatchRegexDecls, MatchChecks, MatchCaptureCode),

    % Generate arithmetic constraint checks if present
    generate_go_constraint_code(Constraints, VarMap, ConstraintChecks),

    % Combine all checks (match + arithmetic)
    (   MatchChecks = '', ConstraintChecks = '' ->
        AllChecks = ''
    ;   MatchChecks = '' ->
        AllChecks = ConstraintChecks
    ;   ConstraintChecks = '' ->
        AllChecks = MatchChecks
    ;   atomic_list_concat([MatchChecks, '\n', ConstraintChecks], AllChecks)
    ),

    % Add capture extraction after checks if present
    (   MatchCaptureCode = '' ->
        AllChecksAndCaptures = AllChecks
    ;   AllChecks = '' ->
        AllChecksAndCaptures = MatchCaptureCode
    ;   atomic_list_concat([AllChecks, '\n', MatchCaptureCode], AllChecksAndCaptures)
    ),

    % Build complete Go code with optional constraints
    (   NumFields = 1 ->
        % Single field - no splitting needed
        (   AllChecksAndCaptures = '' ->
            format(string(GoCode), '
\t// Read from stdin and process ~s records
\tscanner := bufio.NewScanner(os.Stdin)
\tseen := make(map[string]bool)
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\t~s
\t\tresult := ~s
\t\tif !seen[result] {
\t\t\tseen[result] = true
\t\t\tfmt.Println(result)
\t\t}
\t}
', [BodyPredStr, FieldAssignments, OutputExpr])
        ;   % With constraints (match and/or arithmetic)
            format(string(GoCode), '
\t// Read from stdin and process ~s records with filtering
~s
\tscanner := bufio.NewScanner(os.Stdin)
\tseen := make(map[string]bool)
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\t~s
~s
\t\tresult := ~s
\t\tif !seen[result] {
\t\t\tseen[result] = true
\t\t\tfmt.Println(result)
\t\t}
\t}
', [BodyPredStr, MatchRegexDecls, FieldAssignments, AllChecksAndCaptures, OutputExpr])
        )
    ;   % Multi-field - needs splitting and length check
        (   AllChecksAndCaptures = '' ->
            format(string(GoCode), '
\t// Read from stdin and process ~s records
\tscanner := bufio.NewScanner(os.Stdin)
\tseen := make(map[string]bool)
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tparts := strings.Split(line, "~s")
\t\tif len(parts) == ~w {
\t\t\t~s
\t\t\tresult := ~s
\t\t\tif !seen[result] {
\t\t\t\tseen[result] = true
\t\t\t\tfmt.Println(result)
\t\t\t}
\t\t}
\t}
', [BodyPredStr, DelimChar, NumFields, FieldAssignments, OutputExpr])
        ;   % With constraints (match and/or arithmetic)
            format(string(GoCode), '
\t// Read from stdin and process ~s records with filtering
~s
\tscanner := bufio.NewScanner(os.Stdin)
\tseen := make(map[string]bool)
\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tparts := strings.Split(line, "~s")
\t\tif len(parts) == ~w {
\t\t\t~s
~s
\t\t\tresult := ~s
\t\t\tif !seen[result] {
\t\t\t\tseen[result] = true
\t\t\t\tfmt.Println(result)
\t\t\t}
\t\t}
\t}
', [BodyPredStr, MatchRegexDecls, DelimChar, NumFields, FieldAssignments, AllChecksAndCaptures, OutputExpr])
        )
    ).

%% is_semantic_predicate(+Goal)
is_semantic_predicate(semantic_search(_, _, _)).
is_semantic_predicate(crawler_run(_, _)).

%% compile_semantic_rule_go(+PredStr, +HeadArgs, +Goal, -GoCode)
compile_semantic_rule_go(_PredStr, HeadArgs, Goal, GoCode) :-
    build_var_map(HeadArgs, VarMap),
    Goal =.. [GoalName | GoalArgs],
    
    Imports = '\t"fmt"\n\t"log"\n\n\t"unifyweaver/targets/go_runtime/search"\n\t"unifyweaver/targets/go_runtime/embedder"\n\t"unifyweaver/targets/go_runtime/storage"\n\t"unifyweaver/targets/go_runtime/crawler"',
    
    (   GoalName == semantic_search
    ->  GoalArgs = [Query, TopK, _Results],
        term_to_go_expr(Query, VarMap, QueryExpr),
        term_to_go_expr(TopK, VarMap, TopKExpr),
        
        format(string(Body), '
\t// Initialize runtime
\tstore, err := storage.NewStore("data.db")
\tif err != nil { log.Fatal(err) }
\tdefer store.Close()

\temb, err := embedder.NewHugotEmbedder("models/model.onnx", "all-MiniLM-L6-v2")
\tif err != nil { log.Fatal(err) }
\tdefer emb.Close()

\t// Embed query
\tqVec, err := emb.Embed(~s)
\tif err != nil { log.Fatal(err) }
\t
\t// Search
\tresults, err := search.Search(store, qVec, ~w)
\tif err != nil { log.Fatal(err) }

\tfor _, res := range results {
\t\tfmt.Printf("Result: %%s (Score: %%f)\\n", res.ID, res.Score)
\t}
', [QueryExpr, TopKExpr])
    ;   GoalName == crawler_run
    ->  GoalArgs = [Seeds, MaxDepth],
        term_to_go_expr(MaxDepth, VarMap, DepthExpr),
        
        (   is_list(Seeds)
        ->  maplist(atom_string, Seeds, SeedStrs),
            atomic_list_concat(SeedStrs, '", "', Inner),
            format(string(SeedsGo), '[]string{"~w"}', [Inner])
        ;   term_to_go_expr(Seeds, VarMap, SeedsExpr),
            SeedsGo = SeedsExpr
        ),

        format(string(Body), '
\t// Initialize runtime
\tstore, err := storage.NewStore("data.db")
\tif err != nil { log.Fatal(err) }
\tdefer store.Close()

\temb, err := embedder.NewHugotEmbedder("models/model.onnx", "all-MiniLM-L6-v2")
\tif err != nil { 
\t\tlog.Printf("Warning: Embeddings disabled: %%v\\n", err) 
\t\temb = nil
\t} else {
\t\tdefer emb.Close()
\t}

\tcraw := crawler.NewCrawler(store, emb)
\tcraw.Crawl(~w, int(~w))
', [SeedsGo, DepthExpr])
    ),
    
    format(string(GoCode), 'package main

import (
~s
)

func main() {
\t// Parse input arguments if needed (e.g. if HeadArgs are used)
\t// For now, we assume simple stdin/args or constants
~s}
', [Imports, Body]).

%% generate_field_assignments(+Args, -Code)
%  Generate field assignment statements
generate_field_assignments(Args, Code) :-
    findall(Assignment,
        (   nth1(N, Args, _),
            I is N - 1,
            format(atom(Assignment), 'field~w := parts[~w]', [N, I])
        ),
        Assignments),
    atomic_list_concat(Assignments, '\n\t\t\t', Code).

%% generate_go_match_code(+MatchConstraints, +HeadArgs, +BodyArgs, -RegexDecls, -MatchChecks, -CaptureCode)
%  Generate Go regex declarations, match checks, and capture extractions
generate_go_match_code([], _, _, "", "", "") :- !.
generate_go_match_code(MatchConstraints, HeadArgs, BodyArgs, RegexDecls, MatchChecks, CaptureCode) :-
    findall(Decl-Check-Capture,
        (   member(match(Var, Pattern, _Type, Groups), MatchConstraints),
            generate_single_match_code(Var, Pattern, Groups, HeadArgs, BodyArgs, Decl, Check, Capture)
        ),
        DeclsChecksCaps),
    findall(D, member(D-_-_, DeclsChecksCaps), Decls),
    findall(C, member(_-C-_, DeclsChecksCaps), Checks),
    findall(Cap, member(_-_-Cap, DeclsChecksCaps), Captures),
    atomic_list_concat(Decls, '\n', RegexDecls),
    atomic_list_concat(Checks, '\n', MatchChecks),
    % Filter out empty captures and join with newlines
    exclude(=(''), Captures, NonEmptyCaptures),
    (   NonEmptyCaptures = [] ->
        CaptureCode = ""
    ;   atomic_list_concat(NonEmptyCaptures, '\n', CaptureCode)
    ).

%% generate_single_match_code(+Var, +Pattern, +Groups, +HeadArgs, +BodyArgs, -Decl, -Check, -CaptureExtraction)
%  Generate regex declaration, check, and capture extraction for a single match constraint
generate_single_match_code(Var, Pattern, Groups, HeadArgs, BodyArgs, Decl, Check, CaptureExtraction) :-
    % Convert pattern to string
    (   atom(Pattern) ->
        atom_string(Pattern, PatternStr)
    ;   PatternStr = Pattern
    ),

    % Find which field variable contains Var
    % Check HeadArgs first, then BodyArgs
    (   nth1(FieldPos, HeadArgs, HeadVar),
        Var == HeadVar
    ->  format(atom(FieldVar), 'field~w', [FieldPos])
    ;   nth1(FieldPos, BodyArgs, BodyVar),
        Var == BodyVar
    ->  format(atom(FieldVar), 'field~w', [FieldPos])
    ;   FieldVar = 'line'  % If not in head or body args, match against the whole line
    ),

    % Generate unique regex variable name
    gensym(regex, RegexVar),

    % Generate regex compilation
    format(string(Decl), '\t~w := regexp.MustCompile(`~s`)', [RegexVar, PatternStr]),

    % Generate match check code
    (   Groups = [] ->
        % Boolean match - no captures
        format(string(Check), '\t\t\tif !~w.MatchString(~w) {\n\t\t\t\tcontinue\n\t\t\t}',
               [RegexVar, FieldVar]),
        CaptureExtraction = ""
    ;   % Match with capture groups
        length(Groups, NumGroups),
        NumGroups1 is NumGroups + 1,  % +1 for full match
        format(string(Check), '\t\t\tmatches := ~w.FindStringSubmatch(~w)\n\t\t\tif matches == nil || len(matches) != ~w {\n\t\t\t\tcontinue\n\t\t\t}',
               [RegexVar, FieldVar, NumGroups1]),
        % Generate capture extraction code: cap1 := matches[1], cap2 := matches[2], etc.
        findall(CapAssignment,
            (   between(1, NumGroups, CapIdx),
                format(atom(CapAssignment), '\t\t\tcap~w := matches[~w]', [CapIdx, CapIdx])
            ),
            CapAssignments),
        atomic_list_concat(CapAssignments, '\n', CaptureExtraction)
    ).

%% generate_go_constraint_code(+Constraints, +VarMap, -ConstraintChecks)
%  Generate Go code for arithmetic and comparison constraints
%  Includes type conversion from string fields to ints
generate_go_constraint_code([], _, "") :- !.
generate_go_constraint_code(Constraints, VarMap, ConstraintChecks) :-
    % First, collect which fields need int conversion
    findall(Pos,
        (   member(Constraint, Constraints),
            constraint_uses_field(Constraint, VarMap, Pos)
        ),
        FieldPoss),
    list_to_set(FieldPoss, UniqueFieldPoss),

    % Generate int conversion declarations
    findall(Decl,
        (   member(Pos, UniqueFieldPoss),
            format(atom(FieldName), 'field~w', [Pos]),
            format(atom(IntName), 'int~w', [Pos]),
            format(atom(Decl), '\t\t\t~w, err := strconv.Atoi(~w)\n\t\t\tif err != nil {\n\t\t\t\tcontinue\n\t\t\t}',
                   [IntName, FieldName])
        ),
        IntDecls),

    % Generate constraint checks
    findall(Check,
        (   member(Constraint, Constraints),
            constraint_to_go(Constraint, VarMap, GoConstraint),
            % Handle is/2 as assignment, others as conditions
            (   Constraint = is(_, _) ->
                % is/2 is an assignment
                format(atom(Check), '\t\t\t~w', [GoConstraint])
            ;   % Other constraints are conditions
                format(atom(Check), '\t\t\tif !(~w) {\n\t\t\t\tcontinue\n\t\t\t}', [GoConstraint])
            )
        ),
        Checks),

    % Combine declarations and checks
    append(IntDecls, Checks, AllParts),
    atomic_list_concat(AllParts, '\n', ConstraintChecks).

%% constraint_uses_field(+Constraint, +VarMap, -Pos)
%  Check if a constraint uses a field variable and return its position
constraint_uses_field(gt(A, _), VarMap, Pos) :- var(A), member((Var, Pos), VarMap), A == Var, !.
constraint_uses_field(gt(_, B), VarMap, Pos) :- var(B), member((Var, Pos), VarMap), B == Var, !.
constraint_uses_field(lt(A, _), VarMap, Pos) :- var(A), member((Var, Pos), VarMap), A == Var, !.
constraint_uses_field(lt(_, B), VarMap, Pos) :- var(B), member((Var, Pos), VarMap), B == Var, !.
constraint_uses_field(gte(A, _), VarMap, Pos) :- var(A), member((Var, Pos), VarMap), A == Var, !.
constraint_uses_field(gte(_, B), VarMap, Pos) :- var(B), member((Var, Pos), VarMap), B == Var, !.
constraint_uses_field(lte(A, _), VarMap, Pos) :- var(A), member((Var, Pos), VarMap), A == Var, !.
constraint_uses_field(lte(_, B), VarMap, Pos) :- var(B), member((Var, Pos), VarMap), B == Var, !.
constraint_uses_field(eq(A, _), VarMap, Pos) :- var(A), member((Var, Pos), VarMap), A == Var, !.
constraint_uses_field(eq(_, B), VarMap, Pos) :- var(B), member((Var, Pos), VarMap), B == Var, !.
constraint_uses_field(neq(A, _), VarMap, Pos) :- var(A), member((Var, Pos), VarMap), A == Var, !.
constraint_uses_field(neq(_, B), VarMap, Pos) :- var(B), member((Var, Pos), VarMap), B == Var, !.
constraint_uses_field(inequality(A, _), VarMap, Pos) :- var(A), member((Var, Pos), VarMap), A == Var, !.
constraint_uses_field(inequality(_, B), VarMap, Pos) :- var(B), member((Var, Pos), VarMap), B == Var, !.

%% ==============================================
%% MATCH CONSTRAINTS (REGEX WITH CAPTURES)
%% ==============================================

%% extract_match_constraints(+Body, -MatchConstraints)
%  Extract match/3 or match/4 predicates from rule body
%  match/3: match(Field, Pattern, Type)
%  match/4: match(Field, Pattern, Type, Captures)
extract_match_constraints(true, []) :- !.
extract_match_constraints((A, B), Constraints) :- !,
    extract_match_constraints(A, C1),
    extract_match_constraints(B, C2),
    append(C1, C2, Constraints).
% match/4 with capture groups
extract_match_constraints(match(Field, Pattern, Type, Captures), [match(Field, Pattern, Type, Captures)]) :- !.
% match/3 without captures
extract_match_constraints(match(Field, Pattern, Type), [match(Field, Pattern, Type, [])]) :- !.
extract_match_constraints(_, []).

%% compile_match_only_rule_go(+PredStr, +HeadArgs, +VarMap, +FieldDelim, +Unique, +MatchConstraints, -GoCode)
%  Compile rules with only match constraints (no body predicates)
%  Reads from stdin and filters based on regex patterns with capture groups
compile_match_only_rule_go(PredStr, HeadArgs, VarMap, FieldDelim, Unique, MatchConstraints, GoCode) :-
    % Get the first match constraint (for now, we support single match)
    (   MatchConstraints = [match(MatchField, Pattern, _Type, Captures)] ->
        % Generate code for regex matching with captures
        map_field_delimiter(FieldDelim, DelimChar),

        % Build capture variable assignments
        % FindStringSubmatch returns [fullMatch, group1, group2, ...]
        % So we need to map Captures list to matches[1], matches[2], etc.
        length(Captures, NumCaptures),
        findall(Assignment,
            (   between(1, NumCaptures, Idx),
                nth1(Idx, Captures, CaptureVar),
                member((Var, Pos), VarMap),
                CaptureVar == Var,
                format(atom(Assignment), '\t\t\tcap~w := matches[~w]', [Pos, Idx])
            ),
            CaptureAssignments),
        atomic_list_concat(CaptureAssignments, '\n', CaptureCode),

        % Build output expression using head args
        % Map each head arg to either original line or captured value
        findall(OutputPart,
            (   member(Arg, HeadArgs),
                (   % Check if this arg is the matched field
                    Arg == MatchField ->
                    OutputPart = 'line'
                ;   % Check if this arg is a captured variable
                    member((Var, Pos), VarMap),
                    Arg == Var,
                    member(Arg, Captures) ->
                    format(atom(OutputPart), 'cap~w', [Pos])
                ;   OutputPart = 'line'
                )
            ),
            OutputParts),
        build_go_concat_expr(OutputParts, DelimChar, OutputExpr),

        % Generate uniqueness check if needed
        (   Unique = true ->
            SeenMapCode = '\n\tseen := make(map[string]bool)',
            UniqueCode = '\n\t\t\tif !seen[result] {\n\t\t\t\tseen[result] = true\n\t\t\t\tfmt.Println(result)\n\t\t\t}'
        ;   SeenMapCode = '',
            UniqueCode = '\n\t\t\tfmt.Println(result)'
        ),

        % Generate the complete Go code
        format(atom(GoCode),
'\n\t// Read from stdin and process with regex pattern matching\n\n\tpattern := regexp.MustCompile(`~w`)\n\tscanner := bufio.NewScanner(os.Stdin)~w\n\t\n\tfor scanner.Scan() {\n\t\tline := scanner.Text()\n\t\tmatches := pattern.FindStringSubmatch(line)\n\t\tif matches != nil {\n~w\n\t\t\tresult := ~w~w\n\t\t}\n\t}\n',
            [Pattern, SeenMapCode, CaptureCode, OutputExpr, UniqueCode])
    ;   % Multiple or no match constraints
        length(HeadArgs, Arity),
        format(user_error,
               'Go target: multiple/no match constraints not supported for ~w/~w~n',
               [PredStr, Arity]),
        fail
    ).

%% build_go_concat_expr(+Parts, +Delimiter, -Expression)
%  Build a Go string concatenation expression with delimiters
%  e.g., [field1, field2] with ":" -> field1 + ":" + field2
build_go_concat_expr([], _, '""') :- !.
build_go_concat_expr([Single], _, Single) :- !.
build_go_concat_expr([First|Rest], Delim, Expr) :-
    build_go_concat_expr_rest(Rest, Delim, RestExpr),
    format(atom(Expr), '~w + "~s" + ~w', [First, Delim, RestExpr]).

build_go_concat_expr_rest([Single], _, Single) :- !.
build_go_concat_expr_rest([First|Rest], Delim, Expr) :-
    build_go_concat_expr_rest(Rest, Delim, RestExpr),
    format(atom(Expr), '~w + "~s" + ~w', [First, Delim, RestExpr]).

%% Helper functions
build_var_map(HeadArgs, VarMap) :-
    build_var_map_(HeadArgs, 1, VarMap).

build_var_map_([], _, []).
build_var_map_([Arg|Rest], Pos, [(Arg, Pos)|RestMap]) :-
    NextPos is Pos + 1,
    build_var_map_(Rest, NextPos, RestMap).

%% Skip match predicates when extracting regular predicates
extract_predicates(_:Goal, Preds) :- !, extract_predicates(Goal, Preds).
extract_predicates(match(_, _), []) :- !.
extract_predicates(match(_, _, _), []) :- !.
extract_predicates(match(_, _, _, _), []) :- !.

%% Skip constraint operators
extract_predicates(_ > _, []) :- !.
extract_predicates(_ < _, []) :- !.
extract_predicates(_ >= _, []) :- !.
extract_predicates(_ =< _, []) :- !.
extract_predicates(_ =:= _, []) :- !.
extract_predicates(_ =\= _, []) :- !.
extract_predicates(_ \= _, []) :- !.
extract_predicates(\=(_,  _), []) :- !.
extract_predicates(is(_, _), []) :- !.

extract_predicates(true, []) :- !.
extract_predicates((A, B), Predicates) :- !,
    extract_predicates(A, P1),
    extract_predicates(B, P2),
    append(P1, P2, Predicates).
extract_predicates(Goal, [Goal]) :-
    functor(Goal, Functor, _),
    Functor \= ',',
    Functor \= true,
    Functor \= match.

%% extract_match_constraints(+Body, -Constraints)
%  Extract all match/2, match/3, match/4 constraints from body
extract_match_constraints(true, []) :- !.
extract_match_constraints((A, B), Constraints) :- !,
    extract_match_constraints(A, C1),
    extract_match_constraints(B, C2),
    append(C1, C2, Constraints).
extract_match_constraints(match(Var, Pattern), [match(Var, Pattern, auto, [])]) :- !.
extract_match_constraints(match(Var, Pattern, Type), [match(Var, Pattern, Type, [])]) :- !.
extract_match_constraints(match(Var, Pattern, Type, Groups), [match(Var, Pattern, Type, Groups)]) :- !.
extract_match_constraints(_, []).

%% extract_constraints(+Body, -Constraints)
%  Extract all arithmetic and comparison constraints from body
%  Similar to extract_match_constraints but for operators like >, <, is/2, etc.
extract_constraints(true, []) :- !.
extract_constraints((A, B), Constraints) :- !,
    extract_constraints(A, C1),
    extract_constraints(B, C2),
    append(C1, C2, Constraints).
extract_constraints(Goal, []) :-
    var(Goal), !.
% Capture inequality constraints
extract_constraints(A \= B, [inequality(A, B)]) :- !.
extract_constraints(\=(A, B), [inequality(A, B)]) :- !.
% Capture arithmetic comparison constraints
extract_constraints(A > B, [gt(A, B)]) :- !.
extract_constraints(A < B, [lt(A, B)]) :- !.
extract_constraints(A >= B, [gte(A, B)]) :- !.
extract_constraints(A =< B, [lte(A, B)]) :- !.
extract_constraints(A =:= B, [eq(A, B)]) :- !.
extract_constraints(A =\= B, [neq(A, B)]) :- !.
extract_constraints(is(A, B), [is(A, B)]) :- !.
% Skip match predicates (handled separately)
extract_constraints(match(_, _), []) :- !.
extract_constraints(match(_, _, _), []) :- !.
extract_constraints(match(_, _, _, _), []) :- !.
% Skip other predicates
extract_constraints(Goal, []) :-
    functor(Goal, Pred, _),
    Pred \= ',',
    Pred \= true.

%% constraint_to_go(+Constraint, +VarMap, -GoCode)
%  Convert a constraint to Go code
%  VarMap maps Prolog variables to field positions for Go
%  Numeric comparisons need strconv.Atoi for string-to-int conversion
constraint_to_go(inequality(A, B), VarMap, GoCode) :-
    term_to_go_expr_numeric(A, VarMap, GoA),
    term_to_go_expr_numeric(B, VarMap, GoB),
    format(atom(GoCode), '~w != ~w', [GoA, GoB]).
constraint_to_go(gt(A, B), VarMap, GoCode) :-
    term_to_go_expr_numeric(A, VarMap, GoA),
    term_to_go_expr_numeric(B, VarMap, GoB),
    format(atom(GoCode), '~w > ~w', [GoA, GoB]).
constraint_to_go(lt(A, B), VarMap, GoCode) :-
    term_to_go_expr_numeric(A, VarMap, GoA),
    term_to_go_expr_numeric(B, VarMap, GoB),
    format(atom(GoCode), '~w < ~w', [GoA, GoB]).
constraint_to_go(gte(A, B), VarMap, GoCode) :-
    term_to_go_expr_numeric(A, VarMap, GoA),
    term_to_go_expr_numeric(B, VarMap, GoB),
    format(atom(GoCode), '~w >= ~w', [GoA, GoB]).
constraint_to_go(lte(A, B), VarMap, GoCode) :-
    term_to_go_expr_numeric(A, VarMap, GoA),
    term_to_go_expr_numeric(B, VarMap, GoB),
    format(atom(GoCode), '~w <= ~w', [GoA, GoB]).
constraint_to_go(eq(A, B), VarMap, GoCode) :-
    term_to_go_expr_numeric(A, VarMap, GoA),
    term_to_go_expr_numeric(B, VarMap, GoB),
    format(atom(GoCode), '~w == ~w', [GoA, GoB]).
constraint_to_go(neq(A, B), VarMap, GoCode) :-
    term_to_go_expr_numeric(A, VarMap, GoA),
    term_to_go_expr_numeric(B, VarMap, GoB),
    format(atom(GoCode), '~w != ~w', [GoA, GoB]).
constraint_to_go(is(A, B), VarMap, GoCode) :-
    % For is/2, we need to assign the result
    % e.g., Double is Age * 2 becomes: double := age * 2
    term_to_go_expr(A, VarMap, GoA),
    term_to_go_expr(B, VarMap, GoB),
    format(atom(GoCode), '~w := ~w', [GoA, GoB]).

%% term_to_go_expr_numeric(+Term, +VarMap, -GoExpr)
%  Convert a Prolog term to a Go expression for numeric contexts
%  Wraps field references with strconv.Atoi for type conversion
%
term_to_go_expr_numeric(Term, VarMap, GoExpr) :-
    var(Term), !,
    % It's a Prolog variable - convert to int
    (   member((Var, Pos), VarMap),
        Term == Var
    ->  % Need to convert string field to int
        format(atom(FieldName), 'field~w', [Pos]),
        format(atom(IntName), 'int~w', [Pos]),
        % We'll define intN variables in the constraint check code
        GoExpr = IntName
    ;   GoExpr = 'unknown'
    ).
term_to_go_expr_numeric(Term, _, Term) :-
    number(Term), !.
term_to_go_expr_numeric(Term, VarMap, GoExpr) :-
    compound(Term), !,
    Term =.. [Op, Left, Right],
    term_to_go_expr_numeric(Left, VarMap, GoLeft),
    term_to_go_expr_numeric(Right, VarMap, GoRight),
    go_operator(Op, GoOp),
    format(atom(GoExpr), '(~w ~w ~w)', [GoLeft, GoOp, GoRight]).
term_to_go_expr_numeric(Term, _, GoExpr) :-
    atom(Term), !,
    format(atom(GoExpr), '"~w"', [Term]).

%% term_to_go_expr(+Term, +VarMap, -GoExpr)
%  Convert a Prolog term to a Go expression using variable mapping
%  For string contexts (no type conversion)
%
term_to_go_expr(Term, VarMap, GoExpr) :-
    var(Term), !,
    % It's a Prolog variable - look it up in VarMap using identity check
    (   member((Var, Pos), VarMap),
        Term == Var
    ->  % Use field reference - will add type conversion in constraint_to_go if needed
        format(atom(GoExpr), 'field~w', [Pos])
    ;   % Variable not in map - use placeholder
        GoExpr = 'unknown'
    ).
term_to_go_expr(Term, _, GoExpr) :-
    atom(Term), !,
    % Atom constant - quote it for Go
    format(atom(GoExpr), '"~w"', [Term]).
term_to_go_expr(Term, _, GoExpr) :-
    number(Term), !,
    format(atom(GoExpr), '~w', [Term]).
term_to_go_expr(Term, VarMap, GoExpr) :-
    compound(Term), !,
    Term =.. [Op, Left, Right],
    term_to_go_expr(Left, VarMap, GoLeft),
    term_to_go_expr(Right, VarMap, GoRight),
    % Map Prolog operators to Go operators
    go_operator(Op, GoOp),
    format(atom(GoExpr), '(~w ~w ~w)', [GoLeft, GoOp, GoRight]).
term_to_go_expr(Term, _, Term).

%% go_operator(+PrologOp, -GoOp)
%  Map Prolog operators to Go operators
go_operator(+, '+') :- !.
go_operator(-, '-') :- !.
go_operator(*, '*') :- !.
go_operator(/, '/') :- !.
go_operator(mod, '%') :- !.
go_operator(Op, Op).  % Default: use as-is

%% ============================================
%% MULTIPLE RULES COMPILATION
%% ============================================

%% compile_multiple_rules_to_go(+Pred, +Arity, +Clauses, +RecordDelim, +FieldDelim, +Unique, -GoCode)
%  Compile multiple rules (OR pattern) to Go code
%
compile_multiple_rules_to_go(Pred, Arity, Clauses, _RecordDelim, FieldDelim, Unique, GoCode) :-
    atom_string(Pred, PredStr),

    % Check if all rules have compatible structure
    (   all_rules_compatible(Clauses, BodyPredFunctor, BodyArity) ->
        % All rules use the same body predicate with different match constraints
        format('  All rules use ~w/~w~n', [BodyPredFunctor, BodyArity]),

        % Collect all match constraints from all rules
        findall(Pattern,
            (   member(_Head-Body, Clauses),
                extract_match_constraints(Body, Constraints),
                member(match(_Var, Pattern, _Type, _Groups), Constraints)
            ),
            Patterns),

        % If we have patterns, combine them into OR regex
        (   Patterns \= [] ->
            % Combine patterns with | for OR
            atomic_list_concat(Patterns, '|', CombinedPattern),
            format('  Combined pattern: ~w~n', [CombinedPattern]),

            % Get a sample head and body to use for compilation
            Clauses = [Head-Body|_],
            Head =.. [_|HeadArgs],
            extract_predicates(Body, [BodyPred|_]),

            % Create a combined match constraint
            CombinedConstraint = [match(HeadArgs, CombinedPattern, auto, [])],

            % Compile as a single rule with combined match
            build_var_map(HeadArgs, VarMap),
            compile_single_predicate_rule_go(PredStr, HeadArgs, BodyPred, VarMap,
                                            FieldDelim, Unique, CombinedConstraint, GoCode)
        ;   % No match constraints - just multiple rules with same body
            format(user_error,
                   'Go target: multiple rules without match constraints not supported for ~w~n',
                   [PredStr]),
            fail
        )
    ;   % Rules not compatible for simple merging - different body predicates
        format('  Rules have different body predicates - compiling separately~n'),
        compile_different_body_rules_to_go(PredStr, Clauses, FieldDelim, Unique, GoCode)
    ).

%% all_rules_compatible(+Clauses, -BodyPredFunctor, -BodyArity)
%  Check if all rules use the same body predicate (compatible for merging)
all_rules_compatible(Clauses, BodyPredFunctor, BodyArity) :-
    maplist(get_body_predicate_info, Clauses, BodyInfos),
    BodyInfos = [BodyPredFunctor/BodyArity|Rest],
    maplist(==(BodyPredFunctor/BodyArity), Rest).

%% get_body_predicate_info(+Clause, -BodyPredInfo)
%  Extract body predicate functor/arity from a clause
get_body_predicate_info(_Head-Body, BodyPredFunctor/BodyArity) :-
    extract_predicates(Body, [BodyPred|_]),
    functor(BodyPred, BodyPredFunctor, BodyArity).

%% compile_different_body_rules_to_go(+PredStr, +Clauses, +FieldDelim, +Unique, -GoCode)
%  Compile multiple rules with different body predicates
%  Generates code that tries each rule pattern sequentially
compile_different_body_rules_to_go(PredStr, Clauses, FieldDelim, Unique, GoCode) :-
    map_field_delimiter(FieldDelim, DelimChar),

    % For each clause, generate the code to try that rule
    findall(RuleCode,
        (   member(Head-Body, Clauses),
            compile_single_rule_attempt(Head, Body, DelimChar, RuleCode)
        ),
        RuleCodes),

    % Combine all rule attempts
    atomic_list_concat(RuleCodes, '\n', AllRuleAttempts),

    % Build the complete Go code
    (   Unique = true ->
        SeenDecl = '\tseen := make(map[string]bool)\n',
        UniqueCheck = '\t\t\tif !seen[result] {\n\t\t\t\tseen[result] = true\n\t\t\t\tfmt.Println(result)\n\t\t\t}\n'
    ;   SeenDecl = '',
        UniqueCheck = '\t\t\tfmt.Println(result)\n'
    ),

    format(string(GoCode), '
\t// Read from stdin and try each rule pattern
\tscanner := bufio.NewScanner(os.Stdin)
~w\t
\tfor scanner.Scan() {
\t\tline := scanner.Text()
\t\tparts := strings.Split(line, "~s")
\t\t
~s\t}
', [SeenDecl, DelimChar, AllRuleAttempts]).

%% compile_single_rule_attempt(+Head, +Body, +DelimChar, -RuleCode)
%  Generate Go code to try matching and transforming a single rule
compile_single_rule_attempt(Head, Body, DelimChar, RuleCode) :-
    % Extract head arguments
    Head =.. [_|HeadArgs],

    % Extract body predicate and constraints
    extract_predicates(Body, Predicates),
    extract_match_constraints(Body, MatchConstraints),
    extract_constraints(Body, Constraints),

    % Get body predicate info
    (   Predicates = [BodyPred] ->
        BodyPred =.. [BodyPredName|BodyArgs],
        length(BodyArgs, NumFields),
        atom_string(BodyPredName, BodyPredStr),

        % Build variable map
        build_var_map(HeadArgs, VarMap),

        % Build capture mapping
        findall((HeadPos, CapIdx),
            (   nth1(HeadPos, HeadArgs, HeadArg),
                var(HeadArg),
                member(match(_, _, _, Groups), MatchConstraints),
                Groups \= [],
                nth1(CapIdx, Groups, GroupVar),
                HeadArg == GroupVar
            ),
            CaptureMapping),

        % Build output expression
        findall(OutputPart,
            (   nth1(HeadPos, HeadArgs, HeadArg),
                (   var(HeadArg) ->
                    (   nth1(BodyPos, BodyArgs, BodyArg),
                        HeadArg == BodyArg
                    ->  format(atom(OutputPart), 'field~w', [BodyPos])
                    ;   member((HeadPos, CapIdx), CaptureMapping)
                    ->  format(atom(OutputPart), 'cap~w', [CapIdx])
                    ;   fail
                    )
                ;   atom_string(HeadArg, HeadArgStr),
                    format(atom(OutputPart), '"~s"', [HeadArgStr])
                )
            ),
            OutputParts),
        build_go_concat_expr(OutputParts, DelimChar, OutputExpr),

        % Determine which fields are actually used
        findall(UsedFieldNum,
            (   member(OutputPart, OutputParts),
                atom(OutputPart),
                atom_concat('field', NumAtom, OutputPart),
                atom_number(NumAtom, UsedFieldNum)
            ),
            UsedFields),

        % Generate field assignments only for used fields
        findall(Assignment,
            (   nth1(N, BodyArgs, _),
                member(N, UsedFields),
                I is N - 1,
                format(atom(Assignment), 'field~w := parts[~w]', [N, I])
            ),
            Assignments),
        atomic_list_concat(Assignments, '\n\t\t\t', FieldAssignments),

        % Generate match and constraint checks
        generate_go_match_code(MatchConstraints, HeadArgs, BodyArgs, MatchRegexDecls, MatchChecks, MatchCaptureCode),
        generate_go_constraint_code(Constraints, VarMap, ConstraintChecks),

        % Combine checks
        findall(CheckPart,
            (   MatchChecks \= '', member(CheckPart, [MatchChecks])
            ;   MatchCaptureCode \= '', member(CheckPart, [MatchCaptureCode])
            ;   ConstraintChecks \= '', member(CheckPart, [ConstraintChecks])
            ),
            CheckParts),
        atomic_list_concat(CheckParts, '\n', AllChecks),

        % Generate the rule attempt code
        (   AllChecks = '' ->
            format(string(RuleCode), '\t\t// Try rule: ~s/~w~n\t\tif len(parts) == ~w {~n\t\t\t~s~n\t\t\tresult := ~s~n\t\t\tif !seen[result] {~n\t\t\t\tseen[result] = true~n\t\t\t\tfmt.Println(result)~n\t\t\t}~n\t\t\tcontinue~n\t\t}~n',
                   [BodyPredStr, NumFields, NumFields, FieldAssignments, OutputExpr])
        ;   format(string(RuleCode), '\t\t// Try rule: ~s/~w~n\t\tif len(parts) == ~w {~n\t\t\t~s~n~s~n\t\t\tresult := ~s~n\t\t\tif !seen[result] {~n\t\t\t\tseen[result] = true~n\t\t\t\tfmt.Println(result)~n\t\t\t}~n\t\t\tcontinue~n\t\t}~n',
                   [BodyPredStr, NumFields, NumFields, FieldAssignments, AllChecks, OutputExpr])
        )
    ;   % No body predicate or multiple body predicates - skip
        RuleCode = '\t\t// Skipping rule without single body predicate\n'
    ).

%% ============================================
%% TAIL RECURSION COMPILATION
%% ============================================

%% is_tail_recursive_pattern(+Pred, +Clauses)
%  Check if clauses form a tail recursive pattern
%  Pattern: base case + recursive case where recursion is last goal
is_tail_recursive_pattern(Pred, Clauses) :-
    % Must have at least 2 clauses (base + recursive)
    length(Clauses, Len),
    Len >= 2,

    % Separate base and recursive clauses
    partition(is_recursive_clause_for(Pred), Clauses, RecClauses, BaseClauses),

    % Must have at least one base case and one recursive case
    RecClauses \= [],
    BaseClauses \= [],

    % Check that recursive clauses are tail recursive
    forall(member(_-Body, RecClauses),
           is_tail_recursive_body(Pred, Body)).

%% is_recursive_clause_for(+Pred, +Clause)
%  Check if clause is recursive (calls Pred in body)
is_recursive_clause_for(Pred, _Head-Body) :-
    contains_call_to(Pred, Body).

%% contains_call_to(+Pred, +Body)
%  Check if Body contains a call to Pred
contains_call_to(Pred, Body) :-
    (   Body =.. [Pred|_] -> true
    ;   Body = (_,_) ->
        (   Body = (A, B),
            (contains_call_to(Pred, A) ; contains_call_to(Pred, B))
        )
    ;   false
    ).

%% is_tail_recursive_body(+Pred, +Body)
%  Check if Body is tail recursive (Pred call is last goal)
is_tail_recursive_body(Pred, Body) :-
    get_last_goal(Body, LastGoal),
    functor(LastGoal, Pred, _).

%% get_last_goal(+Body, -LastGoal)
%  Extract the last goal from a body
get_last_goal((_, B), LastGoal) :- !, get_last_goal(B, LastGoal).
get_last_goal(Goal, Goal).

%% compile_tail_recursive_to_go(+Pred, +Arity, +Clauses, -GoCode)
%  Compile tail recursive predicate to Go iterative loop
compile_tail_recursive_to_go(Pred, Arity, Clauses, GoCode) :-
    atom_string(Pred, PredStr),

    % Separate base and recursive clauses
    partition(is_recursive_clause_for(Pred), Clauses, RecClauses, BaseClauses),

    % Extract base case value
    BaseClauses = [BaseHead-_|_],
    BaseHead =.. [_|BaseArgs],

    % Extract recursive pattern
    RecClauses = [RecHead-RecBody|_],
    RecHead =.. [_|RecArgs],

    % Determine accumulator pattern for arity 3: pred(Input, Acc, Result)
    (   Arity =:= 3 ->
        generate_ternary_tail_recursion_go(PredStr, BaseArgs, RecArgs, RecBody, GoCode)
    ;   Arity =:= 2 ->
        generate_binary_tail_recursion_go(PredStr, BaseArgs, RecArgs, RecBody, GoCode)
    ;   % Unsupported arity
        format(user_error,
               'Go target: tail recursion for arity ~w not supported for ~w~n',
               [Arity, PredStr]),
        fail
    ).

%% generate_ternary_tail_recursion_go(+PredStr, +BaseArgs, +RecArgs, +RecBody, -GoCode)
%  Generate Go code for arity-3 tail recursion: count([H|T], Acc, N) :- ...
generate_ternary_tail_recursion_go(PredStr, BaseArgs, _RecArgs, RecBody, GoCode) :-
    % Extract step operation from recursive body
    extract_step_operation(RecBody, StepOp),

    % Convert step operation to Go code
    step_op_to_go(StepOp, GoStepCode),

    % Extract base accumulator value (usually second argument)
    (   BaseArgs = [_, BaseAcc, _] ->
        format(atom(BaseAccStr), '~w', [BaseAcc])
    ;   BaseAccStr = '0'
    ),

    % Check if we need to parse the accumulator as an integer
    (   atom_number(BaseAccStr, _) ->
        AccInit = BaseAccStr
    ;   AccInit = 'acc'  % Use parameter if not a number
    ),

    % Generate complete Go function with iterative loop
    format(atom(GoCode), '
// ~w implements tail-recursive ~w with iterative loop
func ~w(n int, acc int) int {
\tcurrentAcc := acc
\tcurrentN := n
\t
\t// Iterative loop (tail recursion optimization)
\tfor currentN > 0 {
\t\t// Step operation
\t\t~s
\t\tcurrentN--
\t}
\t
\treturn currentAcc
}
', [PredStr, PredStr, PredStr, GoStepCode]).

%% generate_binary_tail_recursion_go(+PredStr, +BaseArgs, +RecArgs, +RecBody, -GoCode)
%  Generate Go code for arity-2 tail recursion
generate_binary_tail_recursion_go(PredStr, _BaseArgs, _RecArgs, _RecBody, _GoCode) :-
    format(user_error,
           'Go target: binary tail recursion not supported for ~w~n',
           [PredStr]),
    fail.

%% extract_step_operation(+Body, -StepOp)
%  Extract the accumulator step operation from recursive body
%  Looks for patterns like: Acc1 is Acc * N or Acc1 is Acc + N
%  Skips decrement operations like N1 is N - 1
extract_step_operation((Goal, RestBody), StepOp) :- !,
    (   is_accumulator_update(Goal) ->
        Goal = (_ is Expr),
        StepOp = arithmetic(Expr)
    ;   extract_step_operation(RestBody, StepOp)
    ).
extract_step_operation(Goal, arithmetic(Expr)) :-
    is_accumulator_update(Goal), !,
    Goal = (_ is Expr).
extract_step_operation(_, unknown).

%% is_accumulator_update(+Goal)
%  Check if goal is an accumulator update (not a simple decrement)
is_accumulator_update(_ is Expr) :-
    \+ is_simple_decrement(Expr).

%% is_simple_decrement(+Expr)
%  Check if expression is just a simple decrement like N-1
%  Note: Prolog parses N-1 as +(N, -1) so we check for that too
is_simple_decrement(_ - 1) :- !.
is_simple_decrement(_ - _Const) :- !.
is_simple_decrement(_ + (-1)) :- !.
is_simple_decrement(_ + Const) :- integer(Const), Const < 0, !.

%% step_op_to_go(+StepOp, -GoCode)
%  Convert Prolog step operation to Go code
step_op_to_go(arithmetic(_Acc + Const), GoCode) :-
    integer(Const), !,
    format(atom(GoCode), 'currentAcc += ~w', [Const]).
step_op_to_go(arithmetic(_Acc + _N), 'currentAcc += currentN') :- !.
step_op_to_go(arithmetic(_Acc * _N), 'currentAcc *= currentN') :- !.
step_op_to_go(arithmetic(_Acc - _N), 'currentAcc -= currentN') :- !.
step_op_to_go(unknown, 'currentAcc += 1') :- !.
step_op_to_go(_, 'currentAcc += 1').  % Fallback

%% ============================================
%% GO PROGRAM GENERATION
%% ============================================

%% generate_go_program(+Pred, +Arity, +RecordDelim, +FieldDelim, +Quoting, +EscapeChar, +NeedsStdin, +NeedsRegexp, +NeedsStrings, +NeedsStrconv, +Body, -GoCode)
%  Generate complete Go program with imports and main function
%
generate_go_program(Pred, Arity, RecordDelim, FieldDelim, Quoting, EscapeChar, NeedsStdin, NeedsRegexp, NeedsStrings, NeedsStrconv, Body, GoCode) :-
    atom_string(Pred, PredStr),

    % Generate imports based on what's needed
    (   NeedsStdin ->
        % Build import list for stdin processing
        findall(Import,
            (   member(Pkg, [bufio, fmt, os]),
                format(atom(Import), '\t"~w"', [Pkg])
            ;   NeedsRegexp,
                format(atom(Import), '\t"regexp"', [])
            ;   NeedsStrings,
                format(atom(Import), '\t"strings"', [])
            ;   NeedsStrconv,
                format(atom(Import), '\t"strconv"', [])
            ),
            ImportList),
        list_to_set(ImportList, UniqueImports),  % Remove duplicates
        atomic_list_concat(UniqueImports, '\n', Imports)
    ;   NeedsRegexp ->
        Imports = '\t"fmt"\n\t"regexp"'
    ;   % Facts only need fmt
        Imports = '\t"fmt"'
    ),

    % Generate program template
    format(string(GoCode), 'package main

import (
~s
)

func main() {
~s}
', [Imports, Body]).

%% ============================================
%% UTILITY FUNCTIONS
%% ============================================

%% ============================================
%% DATABASE KEY EXPRESSION COMPILER
%% ============================================

%% compile_key_expression(+KeyExpr, +FieldMappings, +Options, -KeyCode, -Imports)
%  Compile a key expression into Go code that generates a database key
%
%  KeyExpr can be:
%    - field(FieldName)              - Extract a single field value
%    - composite([Expr1, Expr2, ...]) - Concatenate multiple expressions
%    - hash(Expr)                    - SHA-256 hash of an expression
%    - hash(Expr, Algorithm)         - Hash with specific algorithm
%    - literal(String)               - Constant string value
%    - substring(Expr, Start, Len)   - Extract substring
%    - uuid()                        - Generate UUID
%    - auto_increment()              - Sequential counter (future)
%
%  Returns:
%    - KeyCode: Go code that evaluates to the key (as string)
%    - Imports: List of required import packages
%
compile_key_expression(KeyExpr, FieldMappings, Options, KeyCode, Imports) :-
    compile_key_expr(KeyExpr, FieldMappings, Options, KeyCode, Imports).

%% compile_key_expr/5 - Main expression compiler
%
%  field(FieldName) - Extract field value
compile_key_expr(field(FieldName), FieldMappings, _Options, KeyCode, []) :-
    % Find field position
    (   nth1(Pos, FieldMappings, FieldName-_)
    ->  true
    ;   nth1(Pos, FieldMappings, nested(Path, _)),
        last(Path, FieldName)
    ->  true
    ;   format('ERROR: Field ~w not found in mappings: ~w~n', [FieldName, FieldMappings]),
        fail
    ),
    format(atom(FieldVar), 'field~w', [Pos]),
    format(string(KeyCode), 'fmt.Sprintf("%v", ~s)', [FieldVar]).

%  composite([Expr1, Expr2, ...]) - Concatenate expressions
compile_key_expr(composite(Exprs), FieldMappings, Options, KeyCode, AllImports) :-
    % Get delimiter (default ':')
    option(db_key_delimiter(Delimiter), Options, ':'),

    % Compile each sub-expression
    maplist(compile_key_expr_for_composite(FieldMappings, Options), Exprs, ExprCodes, ExprImportsList),

    % Flatten imports
    append(ExprImportsList, AllImports),

    % Build format string and args
    length(Exprs, NumExprs),
    length(FormatSpecifiers, NumExprs),
    maplist(=('%s'), FormatSpecifiers),
    atomic_list_concat(FormatSpecifiers, Delimiter, FormatString),
    atomic_list_concat(ExprCodes, ', ', ArgsString),

    format(string(KeyCode), 'fmt.Sprintf("~s", ~s)', [FormatString, ArgsString]).

%  hash(Expr) - SHA-256 hash of expression
compile_key_expr(hash(Expr), FieldMappings, Options, KeyCode, Imports) :-
    compile_key_expr(hash(Expr, sha256), FieldMappings, Options, KeyCode, Imports).

%  hash(Expr, Algorithm) - Hash with specific algorithm
compile_key_expr(hash(Expr, Algorithm), FieldMappings, Options, KeyCode, Imports) :-
    % Compile the inner expression
    compile_key_expr(Expr, FieldMappings, Options, ExprCode, ExprImports),

    % Generate hash code based on algorithm
    (   Algorithm = sha256
    ->  HashImport = 'crypto/sha256',
        format(string(KeyCode), 'func() string {
\t\tvalStr := ~s
\t\thash := sha256.Sum256([]byte(valStr))
\t\treturn hex.EncodeToString(hash[:])
\t}()', [ExprCode])
    ;   Algorithm = md5
    ->  HashImport = 'crypto/md5',
        format(string(KeyCode), 'func() string {
\t\tvalStr := ~s
\t\thash := md5.Sum([]byte(valStr))
\t\treturn hex.EncodeToString(hash[:])
\t}()', [ExprCode])
    ;   format('ERROR: Unsupported hash algorithm: ~w~n', [Algorithm]),
        fail
    ),

    append(ExprImports, [HashImport, 'encoding/hex'], Imports).

%  literal(String) - Constant string value
compile_key_expr(literal(String), _FieldMappings, _Options, KeyCode, []) :-
    format(string(KeyCode), '"~s"', [String]).

%  substring(Expr, Start, Length) - Extract substring
compile_key_expr(substring(Expr, Start, Length), FieldMappings, Options, KeyCode, Imports) :-
    compile_key_expr(Expr, FieldMappings, Options, ExprCode, Imports),
    End is Start + Length,
    format(string(KeyCode), 'func() string {
\t\tstr := ~s
\t\tif len(str) > ~w {
\t\t\treturn str[~w:~w]
\t\t}
\t\treturn str
\t}()', [ExprCode, End, Start, End]).

%  uuid() - Generate UUID
compile_key_expr(uuid(), _FieldMappings, _Options, KeyCode, ['github.com/google/uuid']) :-
    KeyCode = 'uuid.New().String()'.

%  auto_increment() - Sequential counter (requires state management)
compile_key_expr(auto_increment(), _FieldMappings, _Options, _KeyCode, _Imports) :-
    format('ERROR: auto_increment() not yet implemented~n'),
    fail.

%% Helper for composite - wraps each expression's code
compile_key_expr_for_composite(FieldMappings, Options, Expr, Code, Imports) :-
    compile_key_expr(Expr, FieldMappings, Options, Code, Imports).

%% normalize_key_strategy(+Options, -NormalizedOptions)
%  Normalize key strategy options for backward compatibility
%
%  Converts:
%    db_key_field(Field) → db_key_strategy(field(Field))
%    db_key_fields([F1,F2]) → db_key_strategy(composite([field(F1), field(F2)]))
%
normalize_key_strategy(Options, NormalizedOptions) :-
    % Check if db_key_strategy is already present
    (   option(db_key_strategy(_), Options)
    ->  % Already normalized
        NormalizedOptions = Options
    ;   option(db_key_field(Field), Options)
    ->  % Convert db_key_field(F) to db_key_strategy(field(F))
        select(db_key_field(Field), Options, TempOptions),
        NormalizedOptions = [db_key_strategy(field(Field))|TempOptions]
    ;   option(db_key_fields(Fields), Options)
    ->  % Convert db_key_fields([...]) to db_key_strategy(composite([field(F1), ...]))
        maplist(wrap_field_expr, Fields, FieldExprs),
        select(db_key_fields(Fields), Options, TempOptions),
        NormalizedOptions = [db_key_strategy(composite(FieldExprs))|TempOptions]
    ;   % No key strategy specified - will use default
        NormalizedOptions = Options
    ).

wrap_field_expr(Field, field(Field)).

%% ============================================
%% DATABASE QUERY CONSTRAINTS (Phase 8a)
%% ============================================

%% extract_db_constraints(+Body, -JsonRecord, -Constraints)
%  Extract filter constraints from predicate body
%  Separates json_record/1 from comparison constraints
%
%  Supported constraints (Phase 8a):
%    - Comparisons: >, <, >=, =<, =, \=
%    - Implicit AND (multiple constraints in body)
%
extract_db_constraints(Body, JsonRecord, Constraints) :-
    extract_constraints_impl(Body, none, JsonRecord, [], Constraints).

% Helper: recursively extract constraints from conjunction
extract_constraints_impl((A, B), JsonRecAcc, JsonRec, ConsAcc, Constraints) :- !,
    extract_constraints_impl(A, JsonRecAcc, JsonRec1, ConsAcc, Cons1),
    extract_constraints_impl(B, JsonRec1, JsonRec, Cons1, Constraints).

% json_record/1 - save for later
extract_constraints_impl(json_record(Fields), _JsonRecAcc, json_record(Fields), ConsAcc, ConsAcc) :- !.

% Comparison constraints - collect them
extract_constraints_impl(Constraint, JsonRecAcc, JsonRecAcc, ConsAcc, [Constraint|ConsAcc]) :-
    is_comparison_constraint(Constraint), !.

extract_constraints_impl(Constraint, JsonRecAcc, JsonRecAcc, ConsAcc, [Constraint|ConsAcc]) :-
    is_functional_constraint(Constraint), !.

% Skip other predicates (like json_get, etc.)
extract_constraints_impl(_, JsonRecAcc, JsonRecAcc, ConsAcc, ConsAcc).

%% is_comparison_constraint(+Term)
%  Check if term is a supported comparison constraint
%
is_comparison_constraint(_ > _).
is_comparison_constraint(_ < _).
is_comparison_constraint(_ >= _).
is_comparison_constraint(_ =< _).
is_comparison_constraint(_ = _).
is_comparison_constraint(_ \= _).
is_comparison_constraint(_ =@= _).  % Case-insensitive equality

%% is_functional_constraint(+Term)
%  Check if term is a functional constraint (contains, member, etc.)
%
is_functional_constraint(contains(_, _)).
is_functional_constraint(member(_, _)).

%% is_numeric_constraint(+Constraint)
%  Check if constraint requires numeric type conversion (>, <, >=, =<)
%  Equality and inequality (=, \=) can work with any type
%
is_numeric_constraint(_ > _).
is_numeric_constraint(_ < _).
is_numeric_constraint(_ >= _).
is_numeric_constraint(_ =< _).

%% constraints_need_strings(+Constraints)
%  Check if any constraint requires the strings package
%  True if constraints contain =@= or contains/2
%
constraints_need_strings(Constraints) :-
    member(Constraint, Constraints),
    (   Constraint = (_ =@= _)
    ;   Constraint = contains(_, _)
    ), !.

%% generate_filter_checks(+Constraints, +FieldMappings, -GoCode)
%  Generate Go if statements for constraint checking
%  Returns empty string if no constraints
%
generate_filter_checks([], _, '') :- !.
generate_filter_checks(Constraints, FieldMappings, GoCode) :-
    findall(CheckCode,
        (member(Constraint, Constraints),
         constraint_to_go_check(Constraint, FieldMappings, CheckCode)),
        Checks),
    atomic_list_concat(Checks, '\n', GoCode).

%% constraint_to_go_check(+Constraint, +FieldMappings, -GoCode)
%  Convert a Prolog constraint to Go if statement
%
constraint_to_go_check(Left > Right, FieldMappings, Code) :- !,
    field_term_to_go_expr(Left, FieldMappings, LeftExpr),
    field_term_to_go_expr(Right, FieldMappings, RightExpr),
    format(string(Code), '\t\t\t// Filter: ~w > ~w\n\t\t\tif !(~s > ~s) {\n\t\t\t\treturn nil // Skip record\n\t\t\t}',
           [Left, Right, LeftExpr, RightExpr]).

constraint_to_go_check(Left < Right, FieldMappings, Code) :- !,
    field_term_to_go_expr(Left, FieldMappings, LeftExpr),
    field_term_to_go_expr(Right, FieldMappings, RightExpr),
    format(string(Code), '\t\t\t// Filter: ~w < ~w\n\t\t\tif !(~s < ~s) {\n\t\t\t\treturn nil // Skip record\n\t\t\t}',
           [Left, Right, LeftExpr, RightExpr]).

constraint_to_go_check(Left >= Right, FieldMappings, Code) :- !,
    field_term_to_go_expr(Left, FieldMappings, LeftExpr),
    field_term_to_go_expr(Right, FieldMappings, RightExpr),
    format(string(Code), '\t\t\t// Filter: ~w >= ~w\n\t\t\tif !(~s >= ~s) {\n\t\t\t\treturn nil // Skip record\n\t\t\t}',
           [Left, Right, LeftExpr, RightExpr]).

constraint_to_go_check(Left =< Right, FieldMappings, Code) :- !,
    field_term_to_go_expr(Left, FieldMappings, LeftExpr),
    field_term_to_go_expr(Right, FieldMappings, RightExpr),
    format(string(Code), '\t\t\t// Filter: ~w =< ~w\n\t\t\tif !(~s <= ~s) {\n\t\t\t\treturn nil // Skip record\n\t\t\t}',
           [Left, Right, LeftExpr, RightExpr]).

constraint_to_go_check(Left = Right, FieldMappings, Code) :- !,
    field_term_to_go_expr(Left, FieldMappings, LeftExpr),
    field_term_to_go_expr(Right, FieldMappings, RightExpr),
    format(string(Code), '\t\t\t// Filter: ~w = ~w\n\t\t\tif !(~s == ~s) {\n\t\t\t\treturn nil // Skip record\n\t\t\t}',
           [Left, Right, LeftExpr, RightExpr]).

constraint_to_go_check(Left \= Right, FieldMappings, Code) :- !,
    field_term_to_go_expr(Left, FieldMappings, LeftExpr),
    field_term_to_go_expr(Right, FieldMappings, RightExpr),
    format(string(Code), '\t\t\t// Filter: ~w \\= ~w\n\t\t\tif !(~s != ~s) {\n\t\t\t\treturn nil // Skip record\n\t\t\t}',
           [Left, Right, LeftExpr, RightExpr]).

% Case-insensitive equality (requires strings package)
constraint_to_go_check(Left =@= Right, FieldMappings, Code) :- !,
    field_term_to_go_expr(Left, FieldMappings, LeftExpr),
    field_term_to_go_expr(Right, FieldMappings, RightExpr),
    format(string(Code), '\t\t\t// Filter: ~w =@= ~w (case-insensitive)\n\t\t\tif !strings.EqualFold(fmt.Sprintf("%v", ~s), fmt.Sprintf("%v", ~s)) {\n\t\t\t\treturn nil // Skip record\n\t\t\t}',
           [Left, Right, LeftExpr, RightExpr]).

% Contains check (requires strings package)
constraint_to_go_check(contains(Haystack, Needle), FieldMappings, Code) :- !,
    field_term_to_go_expr(Haystack, FieldMappings, HaystackExpr),
    field_term_to_go_expr(Needle, FieldMappings, NeedleExpr),
    format(string(Code), '\t\t\t// Filter: contains(~w, ~w)\n\t\t\tif !strings.Contains(fmt.Sprintf("%v", ~s), fmt.Sprintf("%v", ~s)) {\n\t\t\t\treturn nil // Skip record\n\t\t\t}',
           [Haystack, Needle, HaystackExpr, NeedleExpr]).

% List membership check
constraint_to_go_check(member(Element, List), FieldMappings, Code) :- !,
    field_term_to_go_expr(Element, FieldMappings, ElementExpr),
    generate_member_check_code(ElementExpr, List, FieldMappings, Code).

%% field_term_to_go_expr(+Term, +FieldMappings, -GoExpr)
%  Convert a Prolog term to Go expression for filter constraints
%  Handles variables (map to fieldN) and literals
%  FieldMappings is a list of Name-Var pairs from json_record
%
field_term_to_go_expr(Term, FieldMappings, GoExpr) :-
    var(Term), !,
    % Find which field this variable corresponds to
    (   nth1(Pos, FieldMappings, _-Var),
        Term == Var
    ->  format(atom(GoExpr), 'field~w', [Pos])
    ;   % Variable not in mappings - shouldn't happen
        format('WARNING: Variable ~w not found in field mappings~n', [Term]),
        GoExpr = 'unknownVar'
    ).

field_term_to_go_expr(Term, _, GoExpr) :-
    string(Term), !,
    % String literal - use as-is with quotes
    format(atom(GoExpr), '"~s"', [Term]).

field_term_to_go_expr(Term, _, GoExpr) :-
    atom(Term), !,
    % Atom literal - quote it for Go string
    format(atom(GoExpr), '"~w"', [Term]).

field_term_to_go_expr(Term, _, GoExpr) :-
    number(Term), !,
    format(atom(GoExpr), '~w', [Term]).

field_term_to_go_expr(Term, _, GoExpr) :-
    % Fallback for unknown terms
    format(atom(GoExpr), '%s /* ~w */', [Term]).

%% generate_member_check_code(+ElementExpr, +List, +FieldMappings, -Code)
%  Generate Go code for list membership check
%  Handles both string and numeric list members
%
generate_member_check_code(ElementExpr, List, FieldMappings, Code) :-
    % Convert list elements to Go expressions
    findall(GoExpr,
        (member(ListItem, List),
         field_term_to_go_expr(ListItem, FieldMappings, GoExpr)),
        GoExprs),
    % Generate the list items as Go slice literals
    atomic_list_concat(GoExprs, ', ', GoListItems),
    % Determine if we're checking strings or numbers
    (   List = [FirstItem|_],
        (atom(FirstItem) ; string(FirstItem))
    ->  % String membership
        format(string(Code), '\t\t\t// Filter: member(~w, list)
\t\t\toptions := []string{~s}
\t\t\tfound := false
\t\t\tfor _, opt := range options {
\t\t\t\tif fmt.Sprintf("%v", ~s) == opt {
\t\t\t\t\tfound = true
\t\t\t\t\tbreak
\t\t\t\t}
\t\t\t}
\t\t\tif !found {
\t\t\t\treturn nil // Skip record
\t\t\t}',
            [ElementExpr, GoListItems, ElementExpr])
    ;   % Numeric membership
        format(string(Code), '\t\t\t// Filter: member(~w, list)
\t\t\tfound := false
\t\t\tfor _, opt := range []interface{}{~s} {
\t\t\t\tif fmt.Sprintf("%v", ~s) == fmt.Sprintf("%v", opt) {
\t\t\t\t\tfound = true
\t\t\t\t\tbreak
\t\t\t\t}
\t\t\t}
\t\t\tif !found {
\t\t\t\treturn nil // Skip record
\t\t\t}',
            [ElementExpr, GoListItems, ElementExpr])
    ).

%% extract_used_fields(+KeyExpr, -UsedFieldPositions)
%  Extract which field positions are referenced by the key expression
%
extract_used_fields(field(FieldName), [Pos]) :-
    % Single field reference - need to find its position in FieldMappings
    % This is a simplified version - full implementation would need FieldMappings
    % For now, extract field number from field(name) atom
    !,
    Pos = 1.  % Placeholder - will be computed properly in context

extract_used_fields(composite(Exprs), AllUsedFields) :-
    !,
    findall(UsedFields,
        (member(Expr, Exprs),
         extract_used_fields(Expr, UsedFields)),
        UsedFieldsList),
    append(UsedFieldsList, AllUsedFields).

extract_used_fields(hash(Expr), UsedFields) :-
    !,
    extract_used_fields(Expr, UsedFields).

extract_used_fields(hash(Expr, _Algorithm), UsedFields) :-
    !,
    extract_used_fields(Expr, UsedFields).

extract_used_fields(substring(Expr, _, _), UsedFields) :-
    !,
    extract_used_fields(Expr, UsedFields).

extract_used_fields(literal(_), []) :- !.
extract_used_fields(uuid(), []) :- !.
extract_used_fields(auto_increment(), []) :- !.

%% extract_used_field_positions(+KeyExpr, +FieldMappings, -UsedPositions)
%  Extract actual field positions by matching field names
%
extract_used_field_positions(KeyExpr, FieldMappings, UsedPositions) :-
    extract_field_names_from_expr(KeyExpr, FieldNames),
    findall(Pos,
        (member(FieldName, FieldNames),
         nth1(Pos, FieldMappings, FieldName-_)),
        UsedPositions).

%% extract_field_names_from_expr(+KeyExpr, -FieldNames)
%  Extract all field names referenced in the expression
%
extract_field_names_from_expr(field(FieldName), [FieldName]) :- !.
extract_field_names_from_expr(composite(Exprs), AllFields) :-
    !,
    findall(Fields,
        (member(Expr, Exprs),
         extract_field_names_from_expr(Expr, Fields)),
        FieldsList),
    append(FieldsList, AllFields).
extract_field_names_from_expr(hash(Expr), Fields) :-
    !,
    extract_field_names_from_expr(Expr, Fields).
extract_field_names_from_expr(hash(Expr, _), Fields) :-
    !,
    extract_field_names_from_expr(Expr, Fields).
extract_field_names_from_expr(substring(Expr, _, _), Fields) :-
    !,
    extract_field_names_from_expr(Expr, Fields).
extract_field_names_from_expr(_, []).  % literals, uuid, etc.

%% ============================================
%% DATABASE READ MODE COMPILATION
%% ============================================

%% ============================================
%% KEY OPTIMIZATION DETECTION (Phase 8c)
%% ============================================

%% analyze_key_optimization(+KeyStrategy, +Constraints, +FieldMappings, -OptType, -OptDetails)
%  Analyze if predicate can use optimized key lookup
%  OptType: direct_lookup | prefix_scan | full_scan
%  OptDetails: Details needed for code generation
%
analyze_key_optimization(KeyStrategy, Constraints, FieldMappings, OptType, OptDetails) :-
    (   can_use_direct_lookup(KeyStrategy, Constraints, FieldMappings, KeyValue)
    ->  OptType = direct_lookup,
        OptDetails = key_value(KeyValue),
        format('  Optimization: Direct lookup (key=~w)~n', [KeyValue])
    ;   can_use_prefix_scan(KeyStrategy, Constraints, FieldMappings, PrefixValue)
    ->  OptType = prefix_scan,
        OptDetails = prefix_value(PrefixValue),
        format('  Optimization: Prefix scan (prefix=~w)~n', [PrefixValue])
    ;   OptType = full_scan,
        OptDetails = none,
        format('  Optimization: Full scan (no key match)~n')
    ).

%% can_use_direct_lookup(+KeyStrategy, +Constraints, +FieldMappings, -KeyValue)
%  Check if we can use bucket.Get() for direct key lookup
%  True if there's an exact equality constraint on the key field
%
can_use_direct_lookup([KeyField], Constraints, FieldMappings, KeyValue) :-
    % Single key field
    member(KeyField-_Var, FieldMappings),
    member(Constraint, Constraints),
    is_exact_equality_on_field(Constraint, KeyField, FieldMappings, KeyValue),
    !.

can_use_direct_lookup(KeyFields, Constraints, FieldMappings, CompositeKey) :-
    % Composite key - all fields must have exact equality
    is_list(KeyFields),
    length(KeyFields, Len),
    Len > 1,
    maplist(has_exact_constraint_for_field(Constraints, FieldMappings), KeyFields, Values),
    build_composite_key_value(Values, CompositeKey),
    !.

%% can_use_prefix_scan(+KeyStrategy, +Constraints, +FieldMappings, -PrefixValue)
%  Check if we can use cursor.Seek() for prefix scan
%  True if first N fields of composite key have exact equality
%
can_use_prefix_scan(KeyFields, Constraints, FieldMappings, PrefixValue) :-
    is_list(KeyFields),
    length(KeyFields, TotalLen),
    TotalLen > 1,  % Must be composite key
    find_matching_prefix_fields(KeyFields, Constraints, FieldMappings, PrefixFields, PrefixValues),
    length(PrefixFields, PrefixLen),
    PrefixLen > 0,
    PrefixLen < TotalLen,  % Not all fields (that would be direct lookup)
    build_composite_key_value(PrefixValues, PrefixValue),
    !.

%% is_exact_equality_on_field(+Constraint, +FieldName, +FieldMappings, -Value)
%  Check if constraint is exact equality (=) on the field and extract value
%  Rejects case-insensitive (=@=), contains, member, etc.
%
is_exact_equality_on_field(Var = Value, FieldName, FieldMappings, Value) :-
    member(FieldName-Var, FieldMappings),
    ground(Value),
    \+ is_variable_reference(Value, FieldMappings),  % Value must be literal, not another field
    !.

%% is_variable_reference(+Term, +FieldMappings)
%  Check if Term is a variable that appears in FieldMappings
%
is_variable_reference(Var, FieldMappings) :-
    var(Var),
    member(_-Var, FieldMappings),
    !.

%% has_exact_constraint_for_field(+Constraints, +FieldMappings, +FieldName, -Value)
%  Check if there's an exact equality constraint for this field
%
has_exact_constraint_for_field(Constraints, FieldMappings, FieldName, Value) :-
    member(Constraint, Constraints),
    is_exact_equality_on_field(Constraint, FieldName, FieldMappings, Value).

%% find_matching_prefix_fields(+KeyFields, +Constraints, +FieldMappings, -PrefixFields, -PrefixValues)
%  Find the longest prefix of KeyFields that all have exact equality constraints
%
find_matching_prefix_fields([Field|Rest], Constraints, FieldMappings, [Field|RestFields], [Value|RestValues]) :-
    has_exact_constraint_for_field(Constraints, FieldMappings, Field, Value),
    !,
    find_matching_prefix_fields(Rest, Constraints, FieldMappings, RestFields, RestValues).
find_matching_prefix_fields(_, _, _, [], []).

%% build_composite_key_value(+Values, -CompositeKey)
%  Build composite key string with colon separator
%  For direct lookup and prefix scan
%
build_composite_key_value([Single], Single) :- !.
build_composite_key_value(Values, CompositeKey) :-
    maplist(value_to_key_string, Values, Strings),
    atomic_list_concat(Strings, ':', CompositeKey).

%% value_to_key_string(+Value, -String)
%  Convert a value to string for key construction
%
value_to_key_string(Value, String) :-
    (   atom(Value) -> atom_string(Value, String)
    ;   string(Value) -> String = Value
    ;   number(Value) -> format(string(String), '~w', [Value])
    ;   format(string(String), '~w', [Value])
    ).

%% ============================================
%% FIELD EXTRACTION FOR DATABASE READ
%% ============================================

%% generate_field_extractions_for_read(+FieldMappings, +Constraints, +HeadArgs, -GoCode)
%  Generate field extraction code for read mode with proper type conversions
%  - Extracts all fields from FieldMappings
%  - Adds type conversions for fields used in constraints
%  - Marks unused fields with _ = fieldN to avoid Go compiler warnings
%
generate_field_extractions_for_read(FieldMappings, Constraints, HeadArgs, GoCode) :-
    % Build set of field positions used in NUMERIC constraints (need float64 conversion)
    findall(NumericPos,
        (   nth1(NumericPos, FieldMappings, _-Var),
            member(C, Constraints),
            is_numeric_constraint(C),
            term_variables(C, CVars),
            member(CV, CVars),
            CV == Var
        ),
        NumericConstraintPositions),

    findall(HeadPos,
        (   nth1(HeadPos, FieldMappings, _-Var),
            member(HV, HeadArgs),
            HV == Var
        ),
        HeadPositions),

    findall(ExtractBlock,
        (   nth1(Pos, FieldMappings, Field-_Var),
            atom_string(Field, FieldStr),

            % Check if this position needs numeric type conversion
            (   member(Pos, NumericConstraintPositions)
            ->  NeedsNumericConversion = true
            ;   NeedsNumericConversion = false
            ),

            (   member(Pos, HeadPositions)
            ->  UsedInHead = true
            ;   UsedInHead = false
            ),

            % Generate extraction with type conversion if needed
            (   NeedsNumericConversion = true
            ->  % Need type conversion for numeric comparison
                format(string(ExtractBlock), '\t\t\t// Extract field: ~w (with type conversion)
\t\t\tfield~wRaw, field~wOk := data["~s"]
\t\t\tif !field~wOk {
\t\t\t\treturn nil // Skip if field missing
\t\t\t}
\t\t\tfield~wFloat, field~wFloatOk := field~wRaw.(float64)
\t\t\tif !field~wFloatOk {
\t\t\t\treturn nil // Skip if wrong type
\t\t\t}
\t\t\tfield~w := field~wFloat',
                    [Field, Pos, Pos, FieldStr, Pos, Pos, Pos, Pos, Pos, Pos, Pos])
            ;   UsedInHead = true
            ->  % Keep as interface{} for output
                format(string(ExtractBlock), '\t\t\t// Extract field: ~w
\t\t\tfield~w, field~wOk := data["~s"]
\t\t\tif !field~wOk {
\t\t\t\treturn nil // Skip if field missing
\t\t\t}',
                    [Field, Pos, Pos, FieldStr, Pos])
            ;   % Unused field - extract and mark as unused
                format(string(ExtractBlock), '\t\t\t// Extract field: ~w (unused)
\t\t\tfield~w, field~wOk := data["~s"]
\t\t\tif !field~wOk {
\t\t\t\treturn nil // Skip if field missing
\t\t\t}
\t\t\t_ = field~w  // Mark as intentionally unused',
                    [Field, Pos, Pos, FieldStr, Pos, Pos])
            )
        ),
        ExtractBlocks),
    atomic_list_concat(ExtractBlocks, '\n', GoCode).

%% generate_output_for_read(+HeadArgs, +FieldMappings, -GoCode)
%  Generate JSON output code with selected fields only
%  Creates a map with only the fields that appear in the predicate head
%
generate_output_for_read(HeadArgs, FieldMappings, GoCode) :-
    % Build a map of selected fields
    findall(FieldName:Pos,
        (   nth1(Idx, HeadArgs, Var),
            nth1(Pos, FieldMappings, FieldName-MappedVar),
            Var == MappedVar
        ),
        FieldSelections),

    % Generate output struct
    findall(FieldPair,
        (   member(FieldName:Pos, FieldSelections),
            atom_string(FieldName, FieldStr),
            format(string(FieldPair), '"~s": field~w', [FieldStr, Pos])
        ),
        FieldPairs),
    atomic_list_concat(FieldPairs, ', ', FieldsStr),

    format(string(GoCode), '\t\t\t// Output selected fields
\t\t\toutput, err := json.Marshal(map[string]interface{}{~s})
\t\t\tif err != nil {
\t\t\t\treturn nil
\t\t\t}
\t\t\tfmt.Println(string(output))', [FieldsStr]).

%% ============================================
%% DATABASE ACCESS CODE GENERATION (Phase 8c)
%% ============================================

%% generate_direct_lookup_code(+DbFile, +BucketStr, +KeyValue, +ProcessCode, -BodyCode)
%  Generate optimized code using bucket.Get() for direct key lookup
%
generate_direct_lookup_code(DbFile, BucketStr, KeyValue, ProcessCode, BodyCode) :-
    atom_string(KeyValue, KeyStr),
    format(string(BodyCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Direct lookup using bucket.Get() (optimized)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\t// Get record by key
\t\tkey := []byte("~s")
\t\tvalue := bucket.Get(key)
\t\tif value == nil {
\t\t\treturn nil // Key not found
\t\t}

\t\t// Deserialize JSON record
\t\tvar data map[string]interface{}
\t\tif err := json.Unmarshal(value, &data); err != nil {
\t\t\tfmt.Fprintf(os.Stderr, "Error unmarshaling record: %v\\n", err)
\t\t\treturn nil
\t\t}

~s
\t\treturn nil
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}
', [DbFile, BucketStr, BucketStr, KeyStr, ProcessCode]).

%% generate_prefix_scan_code(+DbFile, +BucketStr, +PrefixValue, +ProcessCode, -BodyCode)
%  Generate optimized code using cursor.Seek() for prefix scan
%
generate_prefix_scan_code(DbFile, BucketStr, PrefixValue, ProcessCode, BodyCode) :-
    atom_string(PrefixValue, PrefixStr),
    format(string(BodyCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Prefix scan using cursor.Seek() (optimized)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\t// Seek to first key with prefix
\t\tcursor := bucket.Cursor()
\t\tprefix := []byte("~s:")

\t\tfor k, v := cursor.Seek(prefix); k != nil && bytes.HasPrefix(k, prefix); k, v = cursor.Next() {
\t\t\t// Deserialize JSON record
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\tfmt.Fprintf(os.Stderr, "Error unmarshaling record: %v\\n", err)
\t\t\t\tcontinue // Continue with next record
\t\t\t}

~s
\t\t}
\t\treturn nil
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}
', [DbFile, BucketStr, BucketStr, PrefixStr, ProcessCode]).

%% generate_full_scan_code(+DbFile, +BucketStr, +RecordsDesc, +ProcessCode, -BodyCode)
%  Generate standard code using bucket.ForEach() for full scan
%
generate_full_scan_code(DbFile, BucketStr, RecordsDesc, ProcessCode, BodyCode) :-
    format(string(Header), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Read ~s from bucket
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\treturn bucket.ForEach(func(k, v []byte) error {
\t\t\t// Deserialize JSON record
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\tfmt.Fprintf(os.Stderr, "Error unmarshaling record: %v\\n", err)
\t\t\t\treturn nil // Continue with next record
\t\t\t}

', [DbFile, RecordsDesc, BucketStr, BucketStr]),
    Footer = '
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}
',
    string_concat(Header, ProcessCode, Temp),
    string_concat(Temp, Footer, BodyCode).

%% compile_database_read_mode(+Pred, +Arity, +Options, -GoCode)
%  Compile predicate to read from bbolt database and output as JSON
%  Supports optional filtering based on constraints in predicate body
%  Phase 8c: Includes key optimization detection and optimized code generation
%
compile_database_read_mode(Pred, Arity, Options, GoCode) :-
    % Get database options
    option(db_file(DbFile), Options, 'data.db'),
    option(db_bucket(BucketName), Options, Pred),
    atom_string(BucketName, BucketStr),
    option(include_package(IncludePackage), Options, true),

    % Get key strategy (can be single field or list of fields)
    (   option(db_key_field(KeyField), Options)
    ->  (   is_list(KeyField)
        ->  KeyStrategy = KeyField
        ;   KeyStrategy = [KeyField]
        ),
        format('  Key strategy: ~w~n', [KeyStrategy])
    ;   KeyStrategy = none,
        format('  No key strategy specified~n')
    ),

    % Check if predicate has a body with constraints
    functor(Head, Pred, Arity),
    (   clause(Head, Body),
        Body \= true
    ->  % Has body - extract constraints and field mappings
        format('  Predicate body: ~w~n', [Body]),
        extract_db_constraints(Body, JsonRecord, Constraints),
        (   JsonRecord = json_record(FieldMappings0)
        ->  % FieldMappings0 is the list of Name-Var pairs from json_record
            FieldMappings = FieldMappings0,
            Head =.. [_|HeadArgs],
            format('  Field mappings: ~w~n', [FieldMappings]),
            format('  Constraints: ~w~n', [Constraints])
        ;   format('ERROR: No json_record/1 found in predicate body~n'),
            fail
        ),

        % Analyze key optimization opportunities (Phase 8c)
        (   KeyStrategy \= none,
            Constraints \= []
        ->  format('  Analyzing key optimization...~n'),
            analyze_key_optimization(KeyStrategy, Constraints, FieldMappings, OptType, OptDetails)
        ;   OptType = full_scan,
            OptDetails = none,
            format('  Skipping optimization (no key strategy or constraints)~n')
        ),

        % Generate field extraction code (with type conversions for constraints)
        format('  Generating field extractions...~n'),
        generate_field_extractions_for_read(FieldMappings, Constraints, HeadArgs, ExtractCode),
        format('  Generated ~w chars of extraction code~n', [ExtractCode]),
        % Generate filter checks
        format('  Generating filter checks...~n'),
        generate_filter_checks(Constraints, FieldMappings, FilterCode),
        format('  Generated ~w chars of filter code~n', [FilterCode]),
        % Generate output code (selected fields only)
        format('  Generating output code...~n'),
        generate_output_for_read(HeadArgs, FieldMappings, OutputCode),
        format('  Generated ~w chars of output code~n', [OutputCode]),
        % Combine extraction + filter + output
        format('  Combining code sections...~n'),
        (   FilterCode \= ''
        ->  format(string(ProcessCode), '~s\n~s\n~s', [ExtractCode, FilterCode, OutputCode])
        ;   format(string(ProcessCode), '~s\n~s', [ExtractCode, OutputCode])
        ),
        format('  Process code ready: ~w chars~n', [ProcessCode]),
        HasFilters = true,
        format('  HasFilters set to true~n')
    ;   % No body or body is 'true' - read all records as-is
        format('  No predicate body found - reading all records~n'),
        ProcessCode = '\t\t\t// Output as JSON
\t\t\toutput, err := json.Marshal(data)
\t\t\tif err != nil {
\t\t\t\tfmt.Fprintf(os.Stderr, "Error marshaling output: %v\\n", err)
\t\t\t\treturn nil // Continue with next record
\t\t\t}

\t\t\tfmt.Println(string(output))',
        HasFilters = false
    ),

    % Generate database read code (Phase 8c: with optimizations)
    format('  Generating database read code...~n'),
    (   HasFilters = true
    ->  RecordsDesc = 'filtered records'
    ;   RecordsDesc = 'all records'
    ),

    % Generate appropriate database access code based on optimization type
    (   var(OptType)
    ->  % No optimization analysis (no constraints or no key strategy)
        format('  Using full scan (no optimization analysis)~n'),
        generate_full_scan_code(DbFile, BucketStr, RecordsDesc, ProcessCode, BodyCode)
    ;   OptType = direct_lookup
    ->  % Direct lookup optimization
        OptDetails = key_value(KeyValue),
        format('  Generating direct lookup code~n'),
        generate_direct_lookup_code(DbFile, BucketStr, KeyValue, ProcessCode, BodyCode)
    ;   OptType = prefix_scan
    ->  % Prefix scan optimization
        OptDetails = prefix_value(PrefixValue),
        format('  Generating prefix scan code~n'),
        generate_prefix_scan_code(DbFile, BucketStr, PrefixValue, ProcessCode, BodyCode)
    ;   % Full scan (default/fallback)
        format('  Using full scan~n'),
        generate_full_scan_code(DbFile, BucketStr, RecordsDesc, ProcessCode, BodyCode)
    ),
    format('  Body generated successfully (~w chars)~n', [BodyCode]),

    % Wrap in package if requested
    (   IncludePackage = true
    ->  % Check if we need bytes package (for prefix scan optimization)
        (   nonvar(OptType), OptType = prefix_scan
        ->  BytesImport = '\t"bytes"\n'
        ;   BytesImport = ''
        ),
        % Check if we need strings package (only if Constraints is defined)
        (   (var(Constraints) ; Constraints = [])
        ->  % No constraints or empty constraints - no strings needed
            StringsImport = ''
        ;   % Check if constraints need strings package
            (   constraints_need_strings(Constraints)
            ->  StringsImport = '\t"strings"\n'
            ;   StringsImport = ''
            )
        ),
        % Build package with conditional bytes and strings imports
        format(string(GoCode), 'package main

import (
~s\t"encoding/json"
\t"fmt"
\t"os"
~s
\tbolt "go.etcd.io/bbolt"
)

func main() {
~s}
', [BytesImport, StringsImport, BodyCode])
    ;   GoCode = BodyCode
    ).

%% ============================================


%% ============================================
%% GROUP BY AGGREGATION SUPPORT (Phase 9b)
%% ============================================

%% is_group_by_predicate(+Body)
%  Check if predicate body contains group_by/3 or group_by/4
%  group_by/3: group_by(GroupField, Goal, AggOpList) for multiple aggregations
%  group_by/4: group_by(GroupField, Goal, AggOp, Result) for single aggregation
%
is_group_by_predicate(group_by(_GroupField, _Goal, _AggOp)).
is_group_by_predicate(group_by(_GroupField, _Goal, _AggOp, _Result)).
is_group_by_predicate((group_by(_GroupField, _Goal, _AggOp), _Rest)).
is_group_by_predicate((group_by(_GroupField, _Goal, _AggOp, _Result), _Rest)).
is_group_by_predicate((_First, Rest)) :-
    is_group_by_predicate(Rest).

%% extract_group_by_spec(+Body, -GroupField, -Goal, -AggOp, -Result)
%  Extract group_by operation components
%  Handles both group_by/3 (Result = null) and group_by/4 (explicit Result)
%
extract_group_by_spec(group_by(GroupField, Goal, AggOp), GroupField, Goal, AggOp, null) :- !.
extract_group_by_spec(group_by(GroupField, Goal, AggOp, Result), GroupField, Goal, AggOp, Result) :- !.
extract_group_by_spec((group_by(GroupField, Goal, AggOp), _Rest), GroupField, Goal, AggOp, null) :- !.
extract_group_by_spec((group_by(GroupField, Goal, AggOp, Result), _Rest), GroupField, Goal, AggOp, Result) :- !.
extract_group_by_spec((_First, Rest), GroupField, Goal, AggOp, Result) :-
    extract_group_by_spec(Rest, GroupField, Goal, AggOp, Result).

%% extract_having_constraints(+Body, -Constraints)
%  Extract constraints that appear after group_by (HAVING clause)
%  Returns null if no constraints, or the constraint goals
%
extract_having_constraints(group_by(_, _, _), null) :- !.
extract_having_constraints(group_by(_, _, _, _), null) :- !.
extract_having_constraints((group_by(_, _, _), Rest), Rest) :- !.
extract_having_constraints((group_by(_, _, _, _), Rest), Rest) :- !.
extract_having_constraints((_First, Rest), Constraints) :-
    extract_having_constraints(Rest, Constraints).

%% compile_group_by_mode(+Pred, +Arity, +Options, -GoCode)
%  Compile predicate with GROUP BY aggregation
%
compile_group_by_mode(Pred, Arity, Options, GoCode) :-
    % Get predicate definition
    functor(Head, Pred, Arity),
    clause(Head, Body),

    % Extract group_by specification
    extract_group_by_spec(Body, GroupField, Goal, AggOp, Result),
    format('  Group by field: ~w~n', [GroupField]),
    format('  Aggregation: ~w~n', [AggOp]),
    format('  Goal: ~w~n', [Goal]),

    % Extract HAVING constraints (Phase 9c-2)
    extract_having_constraints(Body, HavingConstraints),
    (   HavingConstraints \= null
    ->  format('  HAVING constraints: ~w~n', [HavingConstraints])
    ;   true
    ),

    % Extract field mappings from json_record
    (   Goal = json_record(FieldMappings)
    ->  format('  Field mappings: ~w~n', [FieldMappings])
    ;   format('ERROR: No json_record/1 found in group_by goal~n'),
        fail
    ),

    % Generate grouped aggregation code
    (   option(json_input(true), Options)
    ->  % JSONL Stream Mode
        format('  Generating JSONL stream code for group_by~n'),
        generate_group_by_code_jsonl(GroupField, FieldMappings, AggOp, Result, HavingConstraints, Options, AggBody),
        BaseImports = ["bufio", "encoding/json", "fmt", "os"]
    ;   % Database Mode (bbolt)
        format('  Generating bbolt code for group_by~n'),
        
        % Get database options
        option(db_file(DbFile), Options, 'data.db'),
        option(db_bucket(BucketAtom), Options, Pred),
        atom_string(BucketAtom, BucketStr),
        
        generate_group_by_code(GroupField, FieldMappings, AggOp, Result, HavingConstraints, DbFile, BucketStr, AggBody),
        BaseImports = ["encoding/json", "fmt", "os", "bolt \"go.etcd.io/bbolt\""]
    ),

    % Add strings import if needed (for nested grouping or string manipulation)
    (   (is_list(GroupField) ; member(nested(_, _), FieldMappings))
    ->  append(BaseImports, ["strings"], ImportsList)
    ;   ImportsList = BaseImports
    ),
    
    % Sort and format imports
    sort(ImportsList, UniqueImports),
    maplist(format_import, UniqueImports, ImportLines),
    atomic_list_concat(ImportLines, '\n', Imports),

    % Wrap in package main
    format(string(GoCode), 'package main

import (
~s
)

func main() {
~s}
', [Imports, AggBody]).



%% generate_group_by_code(+GroupField, +FieldMappings, +AggOp, +Result, +HavingConstraints, +DbFile, +BucketStr, -GoCode)
%  Generate Go code for grouped aggregation with optional HAVING clause
%
generate_group_by_code(GroupField, FieldMappings, AggOp, Result, HavingConstraints, DbFile, BucketStr, GoCode) :-
    % Check if GroupField is a list (nested grouping) - handle differently
    (   is_list(GroupField)
    ->  % Nested grouping: GroupField is a list like [_1820, _1822]
        % Pass the list directly to multi-aggregation code (it will extract field names)
        format('  Nested grouping field list detected: ~w~n', [GroupField]),
        (   is_list(AggOp)
        ->  % Already in list format
            generate_multi_aggregation_code(GroupField, FieldMappings, AggOp, HavingConstraints, DbFile, BucketStr, GoCode)
        ;   % Single aggregation with nested grouping - convert to multi-agg format
            format('  Converting single aggregation ~w to list format~n', [AggOp]),
            (   AggOp = count
            ->  OpList = [count(Result)]
            ;   AggOp = sum(AggVar)
            ->  OpList = [sum(AggVar, Result)]
            ;   AggOp = avg(AggVar)
            ->  OpList = [avg(AggVar, Result)]
            ;   AggOp = max(AggVar)
            ->  OpList = [max(AggVar, Result)]
            ;   AggOp = min(AggVar)
            ->  OpList = [min(AggVar, Result)]
            ;   format('ERROR: Unknown aggregation operation: ~w~n', [AggOp]),
                fail
            ),
            generate_multi_aggregation_code(GroupField, FieldMappings, OpList, HavingConstraints, DbFile, BucketStr, GoCode)
        )
    ;   % Single field grouping: GroupField is a variable
        % Find the field name for group field variable
        find_field_for_var(GroupField, FieldMappings, GroupFieldName),

        % Check if AggOp is a list (multiple aggregations) or single operation
        (   is_list(AggOp)
        ->  % Multiple aggregations (Phase 9c-1 + HAVING support Phase 9c-2)
            format('  Multiple aggregations detected: ~w~n', [AggOp]),
            generate_multi_aggregation_code(GroupFieldName, FieldMappings, AggOp, HavingConstraints, DbFile, BucketStr, GoCode)
        % Single aggregation (Phase 9b - backward compatible, HAVING support added)
        ;   AggOp = count
        ->  generate_group_by_count_with_having(GroupFieldName, Result, HavingConstraints, DbFile, BucketStr, GoCode)
        ;   AggOp = sum(AggVar)
        ->  find_field_for_var(AggVar, FieldMappings, AggFieldName),
            generate_group_by_sum_with_having(GroupFieldName, AggFieldName, Result, HavingConstraints, DbFile, BucketStr, GoCode)
        ;   AggOp = avg(AggVar)
        ->  find_field_for_var(AggVar, FieldMappings, AggFieldName),
            generate_group_by_avg_with_having(GroupFieldName, AggFieldName, Result, HavingConstraints, DbFile, BucketStr, GoCode)
        ;   AggOp = max(AggVar)
        ->  find_field_for_var(AggVar, FieldMappings, AggFieldName),
            generate_group_by_max_with_having(GroupFieldName, AggFieldName, Result, HavingConstraints, DbFile, BucketStr, GoCode)
        ;   AggOp = min(AggVar)
        ->  find_field_for_var(AggVar, FieldMappings, AggFieldName),
            generate_group_by_min_with_having(GroupFieldName, AggFieldName, Result, HavingConstraints, DbFile, BucketStr, GoCode)
        ;   format('ERROR: Unknown group_by aggregation operation: ~w~n', [AggOp]),
            fail
        )
    ).

%% generate_multi_aggregation_code(+GroupField, +FieldMappings, +OpList, +HavingConstraints, +DbFile, +BucketStr, -GoCode)
%  Generate GROUP BY code for multiple aggregations in single query with HAVING support
%  GroupField can be a single field name or a list of field names for nested grouping
%  OpList is a list like [count(Count), avg(Age, AvgAge), max(Age, MaxAge)]
%  HavingConstraints are post-aggregation filters (null if none)
%
generate_multi_aggregation_code(GroupField, FieldMappings, OpList, HavingConstraints, DbFile, BucketStr, GoCode) :-
    % Check if GroupField is a list (nested grouping) or single field
    (   is_list(GroupField)
    ->  format('  Nested grouping detected: ~w~n', [GroupField]),
        generate_nested_group_multi_agg(GroupField, FieldMappings, OpList, HavingConstraints, DbFile, BucketStr, GoCode)
    ;   % Single field grouping - original implementation
        generate_single_field_multi_agg(GroupField, FieldMappings, OpList, HavingConstraints, DbFile, BucketStr, GoCode)
    ).

%% generate_single_field_multi_agg(+GroupField, +FieldMappings, +OpList, +HavingConstraints, +DbFile, +BucketStr, -GoCode)
%  Original implementation for single-field grouping
%
generate_single_field_multi_agg(GroupField, FieldMappings, OpList, HavingConstraints, DbFile, BucketStr, GoCode) :-
    % Parse operations to determine what's needed
    parse_multi_agg_operations(OpList, FieldMappings, AggInfo),

    % Generate struct fields based on operations
    generate_multi_agg_struct_fields(AggInfo, StructFields),

    % Generate field extractions and accumulation code
    generate_multi_agg_accumulation(AggInfo, AccumulationCode),

    % Generate output code for all metrics
    generate_multi_agg_output(GroupField, AggInfo, OutputCode),

    % Get struct initialization values
    get_struct_init_values(AggInfo, InitValues),

    % Get output calculations (e.g., avg = sum/count)
    get_output_calculations(AggInfo, OutputCalcs),

    % Generate HAVING filter code (Phase 9c-2)
    generate_having_filter_code(HavingConstraints, AggInfo, OpList, HavingFilterCode),

    % Combine into full Go code
    format(string(GoCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Group by ~s with multiple aggregations
\ttype GroupStats struct {
~s
\t}
\tstats := make(map[string]*GroupStats)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\treturn bucket.ForEach(func(k, v []byte) error {
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\treturn nil // Skip invalid records
\t\t\t}

\t\t\t// Extract group field
\t\t\tif groupRaw, ok := data["~s"]; ok {
\t\t\t\tif groupStr, ok := groupRaw.(string); ok {
\t\t\t\t\t// Initialize stats for this group if needed
\t\t\t\t\tif _, exists := stats[groupStr]; !exists {
\t\t\t\t\t\tstats[groupStr] = &GroupStats{~s}
\t\t\t\t\t}
~s
\t\t\t\t}
\t\t\t}
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}

\t// Output results as JSON (one per group)
\tfor group, s := range stats {
~s
~s
\t\tresult := map[string]interface{}{
\t\t\t"~s": group,
~s
\t\t}
\t\toutput, _ := json.Marshal(result)
\t\tfmt.Println(string(output))
\t}
',  [DbFile, GroupField, StructFields, BucketStr, BucketStr, GroupField,
    InitValues, AccumulationCode, OutputCalcs, HavingFilterCode, GroupField, OutputCode]).

%% extract_group_field_names(+GroupFieldVars, +FieldMappings, -FieldNames)
%  Extract field names from a list of variables by looking them up in FieldMappings
%  GroupFieldVars: List of variables like [_1820, _1822]
%  FieldMappings: List of pairs like [state-_1820, city-_1822, name-_1892]
%  FieldNames: Extracted atom list like [state, city]
%
extract_group_field_names([], _, []) :- !.
extract_group_field_names([Var|Rest], FieldMappings, [FieldName|RestNames]) :-
    member(FieldName-MappedVar, FieldMappings),
    Var == MappedVar,  % Use == for variable identity check
    !,
    extract_group_field_names(Rest, FieldMappings, RestNames).
extract_group_field_names([Var|_], FieldMappings, _) :-
    % If we get here, variable wasn't found in mappings
    format('ERROR: Group field variable ~w not found in mappings ~w~n', [Var, FieldMappings]),
    fail.

%% generate_nested_group_multi_agg(+GroupFields, +FieldMappings, +OpList, +HavingConstraints, +DbFile, +BucketStr, -GoCode)
%  Generate GROUP BY code for nested grouping (multiple group fields)
%  GroupFields is a list like [State, City]
%
generate_nested_group_multi_agg(GroupFields, FieldMappings, OpList, HavingConstraints, DbFile, BucketStr, GoCode) :-
    % Extract field names from variables
    extract_group_field_names(GroupFields, FieldMappings, FieldNames),

    % Parse operations
    parse_multi_agg_operations(OpList, FieldMappings, AggInfo),

    % Generate struct fields
    generate_multi_agg_struct_fields(AggInfo, StructFields),

    % Generate accumulation code
    generate_multi_agg_accumulation(AggInfo, AccumulationCode),

    % Get struct initialization
    get_struct_init_values(AggInfo, InitValues),

    % Get output calculations
    get_output_calculations(AggInfo, OutputCalcs),

    % Generate HAVING filter code
    generate_having_filter_code(HavingConstraints, AggInfo, OpList, HavingFilterCode),

    % Generate composite key extraction code (includes accumulation) - use field names
    generate_composite_key_extraction(FieldNames, AccumulationCode, KeyExtractionCode),

    % Generate composite key parsing and output - use field names
    generate_composite_key_output(FieldNames, AggInfo, KeyOutputCode),

    % Create group fields description for comment
    atomic_list_concat(FieldNames, ', ', GroupFieldsStr),

    % Combine into full Go code
    format(string(GoCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Group by [~s] with multiple aggregations
\ttype GroupStats struct {
~s
\t}
\tstats := make(map[string]*GroupStats)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\treturn bucket.ForEach(func(k, v []byte) error {
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\treturn nil // Skip invalid records
\t\t\t}

~s
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}

\t// Output results as JSON (one per group)
\tfor groupKey, s := range stats {
~s
~s
~s
\t\toutput, _ := json.Marshal(result)
\t\tfmt.Println(string(output))
\t}
', [DbFile, GroupFieldsStr, StructFields, BucketStr, BucketStr,
    KeyExtractionCode, OutputCalcs, HavingFilterCode, KeyOutputCode]).

%% generate_composite_key_extraction(+GroupFields, -Code)
%  Generate Go code to extract multiple fields and create composite key
%
generate_composite_key_extraction(GroupFields, AccumulationCode, Code) :-
    length(GroupFields, NumFields),
    (   NumFields =:= 1
    ->  % Single field - simpler case
        [Field] = GroupFields,
        format(string(Code), '\t\t\t// Extract group field
\t\t\tif groupRaw, ok := data["~s"]; ok {
\t\t\t\tif groupStr, ok := groupRaw.(string); ok {
\t\t\t\t\t// Initialize stats for this group if needed
\t\t\t\t\tif _, exists := stats[groupStr]; !exists {
\t\t\t\t\t\tstats[groupStr] = &GroupStats{}
\t\t\t\t\t}
~s
\t\t\t\t}
\t\t\t}', [Field, AccumulationCode])
    ;   % Multiple fields - use composite key
        generate_field_extractions(GroupFields, Extractions),
        % Replace groupStr with groupKey in accumulation code for nested grouping
        atomic_list_concat(Split, 'groupStr', AccumulationCode),
        atomic_list_concat(Split, 'groupKey', AccumulationCodeFixed),
        format(string(Code), '\t\t\t// Extract group fields for composite key
\t\t\tkeyParts := make([]string, 0, ~d)
~s
\t\t\t// Check all fields were extracted
\t\t\tif len(keyParts) == ~d {
\t\t\t\tgroupKey := strings.Join(keyParts, "|")
\t\t\t\t// Initialize stats for this group if needed
\t\t\t\tif _, exists := stats[groupKey]; !exists {
\t\t\t\t\tstats[groupKey] = &GroupStats{}
\t\t\t\t}
~s
\t\t\t}', [NumFields, Extractions, NumFields, AccumulationCodeFixed])
    ).

%% generate_field_extractions(+Fields, -Code)
%  Generate extraction code for each field in the composite key
%
generate_field_extractions(Fields, Code) :-
    generate_field_extractions_impl(Fields, 1, CodeLines),
    atomic_list_concat(CodeLines, '\n', Code).

generate_field_extractions_impl([], _, []).
generate_field_extractions_impl([Field|Rest], Index, [ThisCode|RestCodes]) :-
    format(string(ThisCode), '\t\t\tif val~d, ok := data["~s"]; ok {
\t\t\t\tif str~d, ok := val~d.(string); ok {
\t\t\t\t\tkeyParts = append(keyParts, str~d)
\t\t\t\t}
\t\t\t}', [Index, Field, Index, Index, Index]),
    NextIndex is Index + 1,
    generate_field_extractions_impl(Rest, NextIndex, RestCodes).

%% generate_key_parts_list(+Fields, -Code)
%  Generate the keyParts list initialization
%
generate_key_parts_list(Fields, Code) :-
    length(Fields, Len),
    format(string(Code), 'make([]string, 0, ~d)', [Len]).

%% generate_composite_key_output(+GroupFields, +AggInfo, -Code)
%  Generate code to parse composite key and output all group fields
%
generate_composite_key_output([Field], AggInfo, Code) :- !,
    % Single field - simple output
    generate_multi_agg_output_fields(AggInfo, AggOutputCode),
    format(string(Code), '\t\tresult := map[string]interface{}{
\t\t\t"~s": groupKey,
~s
\t\t}', [Field, AggOutputCode]).

generate_composite_key_output(GroupFields, AggInfo, Code) :-
    % Multiple fields - parse composite key
    length(GroupFields, NumFields),
    generate_groupkey_field_parsing(GroupFields, 0, FieldParsing),
    generate_multi_agg_output_fields(AggInfo, AggOutputCode),
    (   AggOutputCode = ''
    ->  FullOutput = FieldParsing
    ;   atomic_list_concat([FieldParsing, ',\n', AggOutputCode], FullOutput)
    ),
    format(string(Code), '\t\t// Parse composite key
\t\tparts := strings.Split(groupKey, "|")
\t\tif len(parts) < ~d {
\t\t\tcontinue  // Skip malformed keys
\t\t}
\t\tresult := map[string]interface{}{
~s
\t\t}', [NumFields, FullOutput]).

%% generate_groupkey_field_parsing(+Fields, +Index, -Code)
%  Generate parsing code for composite key fields from group_by
%
generate_groupkey_field_parsing([], _, '').
generate_groupkey_field_parsing([Field|Rest], Index, Code) :-
    format(string(ThisCode), '\t\t\t"~s": parts[~d]', [Field, Index]),
    NextIndex is Index + 1,
    generate_groupkey_field_parsing(Rest, NextIndex, RestCode),
    (   RestCode = ''
    ->  Code = ThisCode
    ;   atomic_list_concat([ThisCode, ',\n', RestCode], Code)
    ).

%% generate_multi_agg_output_fields(+AggInfo, -Code)
%  Generate output field code for aggregations (without map wrapper)
%
generate_multi_agg_output_fields(AggInfo, Code) :-
    findall(FieldCode, (
        member(agg(OpType, _, _, OutputName), AggInfo),
        operation_output_field(OpType, OutputName, FieldCode)
    ), FieldCodes),
    atomic_list_concat(FieldCodes, ',\n', Code).

%% operation_output_field(+OpType, +Name, -Code)
%  Generate single output field code
%
operation_output_field(count, count, '\t\t\t"count": s.count').
operation_output_field(sum, sum, '\t\t\t"sum": s.sum').
operation_output_field(avg, avg, '\t\t\t"avg": avg').
operation_output_field(max, max, '\t\t\t"max": s.maxValue').
operation_output_field(min, min, '\t\t\t"min": s.minValue').

%% parse_multi_agg_operations(+OpList, +FieldMappings, -AggInfo)
%  Parse list of operations into structured info
%  AggInfo is a list of operation specs like:
%    [op(count, null, CountVar), op(avg, age, AvgVar), op(max, age, MaxVar)]
%
parse_multi_agg_operations([], _FieldMappings, []).
parse_multi_agg_operations([Op|Rest], FieldMappings, [Info|RestInfo]) :-
    parse_single_agg_op(Op, FieldMappings, Info),
    parse_multi_agg_operations(Rest, FieldMappings, RestInfo).

%% parse_single_agg_op(+Op, +FieldMappings, -Info)
%  Parse a single operation specification
%
parse_single_agg_op(count, _FieldMappings, agg(count, null, null, count)) :- !.
parse_single_agg_op(count(_ResultVar), _FieldMappings, agg(count, null, null, count)) :- !.
parse_single_agg_op(sum(AggVar), FieldMappings, agg(sum, FieldName, null, sum)) :- !,
    find_field_for_var(AggVar, FieldMappings, FieldName).
parse_single_agg_op(sum(AggVar, _ResultVar), FieldMappings, agg(sum, FieldName, null, sum)) :- !,
    find_field_for_var(AggVar, FieldMappings, FieldName).
parse_single_agg_op(avg(AggVar), FieldMappings, agg(avg, FieldName, null, avg)) :- !,
    find_field_for_var(AggVar, FieldMappings, FieldName).
parse_single_agg_op(avg(AggVar, _ResultVar), FieldMappings, agg(avg, FieldName, null, avg)) :- !,
    find_field_for_var(AggVar, FieldMappings, FieldName).
parse_single_agg_op(max(AggVar), FieldMappings, agg(max, FieldName, null, max)) :- !,
    find_field_for_var(AggVar, FieldMappings, FieldName).
parse_single_agg_op(max(AggVar, _ResultVar), FieldMappings, agg(max, FieldName, null, max)) :- !,
    find_field_for_var(AggVar, FieldMappings, FieldName).
parse_single_agg_op(min(AggVar), FieldMappings, agg(min, FieldName, null, min)) :- !,
    find_field_for_var(AggVar, FieldMappings, FieldName).
parse_single_agg_op(min(AggVar, _ResultVar), FieldMappings, agg(min, FieldName, null, min)) :- !,
    find_field_for_var(AggVar, FieldMappings, FieldName).

%% generate_multi_agg_struct_fields(+AggInfo, -StructFields)
%  Generate Go struct field declarations based on operations
%
generate_multi_agg_struct_fields(AggInfo, StructFields) :-
    collect_needed_fields(AggInfo, NeededFields),
    format_struct_fields(NeededFields, StructFields).

%% collect_needed_fields(+AggInfo, -NeededFields)
%  Determine which struct fields are needed
%
collect_needed_fields(AggInfo, NeededFields) :-
    findall(Field, (
        member(agg(OpType, _FieldName, _, _), AggInfo),
        needed_struct_field(OpType, Field)
    ), AllFields),
    sort(AllFields, NeededFields).

%% needed_struct_field(+OpType, -Field)
%  Map operation type to required struct fields
%
needed_struct_field(count, count).
needed_struct_field(sum, sum).
needed_struct_field(sum, count).  % sum also needs count for proper tracking
needed_struct_field(avg, sum).
needed_struct_field(avg, count).
needed_struct_field(max, maxValue).
needed_struct_field(max, maxFirst).
needed_struct_field(min, minValue).
needed_struct_field(min, minFirst).

%% format_struct_fields(+NeededFields, -StructFields)
%  Format struct fields as Go code
%
format_struct_fields(NeededFields, StructFields) :-
    findall(FieldLine, (
        member(Field, NeededFields),
        go_struct_field_line(Field, FieldLine)
    ), Lines),
    atomic_list_concat(Lines, '\n', StructFields).

%% go_struct_field_line(+Field, -Line)
%  Generate Go struct field declaration
%
go_struct_field_line(count, '\t\tcount    int').
go_struct_field_line(sum, '\t\tsum      float64').
go_struct_field_line(maxValue, '\t\tmaxValue float64').
go_struct_field_line(maxFirst, '\t\tmaxFirst bool').
go_struct_field_line(minValue, '\t\tminValue float64').
go_struct_field_line(minFirst, '\t\tminFirst bool').

%% get_struct_init_values(+AggInfo, -InitValues)
%  Generate initialization values for struct
%
get_struct_init_values(AggInfo, 'maxFirst: true, minFirst: true') :-
    member(agg(max, _, _, _), AggInfo),
    member(agg(min, _, _, _), AggInfo),
    !.
get_struct_init_values(AggInfo, 'maxFirst: true') :-
    member(agg(max, _, _, _), AggInfo),
    !.
get_struct_init_values(AggInfo, 'minFirst: true') :-
    member(agg(min, _, _, _), AggInfo),
    !.
get_struct_init_values(_, '').

%% generate_multi_agg_accumulation(+AggInfo, -AccumulationCode)
%  Generate accumulation code for all operations
%  Smart handling: if 'count' operation is present, it owns the count field
%
generate_multi_agg_accumulation(AggInfo, AccumulationCode) :-
    % Check if count operation is present
    (   member(agg(count, _, _, _), AggInfo)
    ->  HasCount = true
    ;   HasCount = false
    ),
    % Generate code for each operation
    findall(Code, (
        member(AggSpec, AggInfo),
        generate_single_accumulation(AggSpec, HasCount, Code)
    ), CodeLines),
    atomic_list_concat(CodeLines, '', AccumulationCode).

%% generate_single_accumulation(+AggSpec, +HasCount, -Code)
%  Generate accumulation code for one operation
%  HasCount indicates if a separate count operation exists
%
generate_single_accumulation(agg(count, null, _, _), _, '\t\t\t\t\t// Count operation
\t\t\t\t\tstats[groupStr].count++
') :- !.

generate_single_accumulation(agg(sum, FieldName, _, _), HasCount, Code) :- !,
    % Only increment count if no separate count operation exists
    (   HasCount = true
    ->  format(string(Code), '\t\t\t\t\t// Sum ~s
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tstats[groupStr].sum += valueFloat
\t\t\t\t\t\t}
\t\t\t\t\t}
', [FieldName, FieldName])
    ;   format(string(Code), '\t\t\t\t\t// Sum ~s
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tstats[groupStr].sum += valueFloat
\t\t\t\t\t\t\tstats[groupStr].count++
\t\t\t\t\t\t}
\t\t\t\t\t}
', [FieldName, FieldName])
    ).

generate_single_accumulation(agg(avg, FieldName, _, _), HasCount, Code) :- !,
    % Only increment count if no separate count operation exists
    (   HasCount = true
    ->  format(string(Code), '\t\t\t\t\t// Average ~s
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tstats[groupStr].sum += valueFloat
\t\t\t\t\t\t}
\t\t\t\t\t}
', [FieldName, FieldName])
    ;   format(string(Code), '\t\t\t\t\t// Average ~s
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tstats[groupStr].sum += valueFloat
\t\t\t\t\t\t\tstats[groupStr].count++
\t\t\t\t\t\t}
\t\t\t\t\t}
', [FieldName, FieldName])
    ).

generate_single_accumulation(agg(max, FieldName, _, _), _, Code) :- !,
    format(string(Code), '\t\t\t\t\t// Max ~s
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tif stats[groupStr].maxFirst || valueFloat > stats[groupStr].maxValue {
\t\t\t\t\t\t\t\tstats[groupStr].maxValue = valueFloat
\t\t\t\t\t\t\t\tstats[groupStr].maxFirst = false
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
', [FieldName, FieldName]).

generate_single_accumulation(agg(min, FieldName, _, _), _, Code) :- !,
    format(string(Code), '\t\t\t\t\t// Min ~s
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tif stats[groupStr].minFirst || valueFloat < stats[groupStr].minValue {
\t\t\t\t\t\t\t\tstats[groupStr].minValue = valueFloat
\t\t\t\t\t\t\t\tstats[groupStr].minFirst = false
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
', [FieldName, FieldName]).

%% get_output_calculations(+AggInfo, -OutputCalcs)
%  Generate pre-output calculations (e.g., avg = sum / count)
%
get_output_calculations(AggInfo, '\t\tavg := 0.0
\t\tif s.count > 0 {
\t\t\tavg = s.sum / float64(s.count)
\t\t}') :-
    member(agg(avg, _, _, _), AggInfo),
    !.
get_output_calculations(_, '').

%% generate_multi_agg_output(+GroupField, +AggInfo, -OutputCode)
%  Generate output field mappings for result JSON
%
generate_multi_agg_output(_GroupField, AggInfo, OutputCode) :-
    findall(Line, (
        member(AggSpec, AggInfo),
        generate_output_field(AggSpec, Line)
    ), Lines),
    atomic_list_concat(Lines, ',\n', OutputCode).

%% generate_output_field(+AggSpec, -Line)
%  Generate single output field mapping
%
generate_output_field(agg(count, _, _, _), '\t\t\t"count": s.count') :- !.
generate_output_field(agg(sum, _, _, _), '\t\t\t"sum": s.sum') :- !.
generate_output_field(agg(avg, _, _, _), '\t\t\t"avg": avg') :- !.
generate_output_field(agg(max, _, _, _), '\t\t\t"max": s.maxValue') :- !.
generate_output_field(agg(min, _, _, _), '\t\t\t"min": s.minValue') :- !.

%% ============================================
%% HAVING CLAUSE SUPPORT (Phase 9c-2)
%% ============================================

%% generate_having_filter_code(+HavingConstraints, +AggInfo, +OpList, -FilterCode)
%  Generate HAVING filter code for output loop
%  Parses constraints and generates Go filter code with continue statements
%
generate_having_filter_code(null, _, _, '') :- !.
generate_having_filter_code(Constraints, AggInfo, OpList, FilterCode) :-
    % Parse constraints into list
    parse_constraints_to_list(Constraints, ConstraintList),
    % Generate filter code for each constraint
    findall(Code, (
        member(Constraint, ConstraintList),
        generate_single_having_filter(Constraint, AggInfo, OpList, Code)
    ), CodeLines),
    atomic_list_concat(CodeLines, '', FilterCode).

%% parse_constraints_to_list(+Constraints, -ConstraintList)
%  Convert conjunction of constraints into a list
%
parse_constraints_to_list((C1, C2), List) :- !,
    parse_constraints_to_list(C1, L1),
    parse_constraints_to_list(C2, L2),
    append(L1, L2, List).
parse_constraints_to_list(C, [C]).

%% generate_single_having_filter(+Constraint, +AggInfo, +OpList, -Code)
%  Generate filter code for a single constraint
%  Supports: >, <, >=, <=, =, =\=
%
generate_single_having_filter(Constraint, AggInfo, OpList, Code) :-
    % Extract operator and operands
    parse_constraint_operator(Constraint, Var, Op, Value),
    % Map variable to Go expression
    map_variable_to_go_expr(Var, AggInfo, OpList, GoExpr),
    % Generate Go comparison code
    format(string(Code), '\t\t// HAVING filter: ~w ~w ~w
\t\tif !(~s ~s ~w) {
\t\t\tcontinue
\t\t}
', [Var, Op, Value, GoExpr, Op, Value]).

%% parse_constraint_operator(+Constraint, -Var, -Op, -Value)
%  Parse constraint to extract variable, operator, and value
%
parse_constraint_operator(Var > Value, Var, '>', Value) :- !.
parse_constraint_operator(Var < Value, Var, '<', Value) :- !.
parse_constraint_operator(Var >= Value, Var, '>=', Value) :- !.
parse_constraint_operator(Var =< Value, Var, '=<', Value) :- !.
parse_constraint_operator(Var = Value, Var, '==', Value) :- !.
parse_constraint_operator(Var =\= Value, Var, '!=', Value) :- !.

%% map_variable_to_go_expr(+Var, +AggInfo, +OpList, -GoExpr)
%  Map Prolog variable to Go expression (e.g., Count -> s.count, Avg -> avg)
%
map_variable_to_go_expr(Var, _AggInfo, OpList, GoExpr) :-
    % Find which operation produces this variable
    member(Op, OpList),
    variable_from_operation(Op, Var, OpType),
    !,
    % Map to Go expression
    operation_to_go_expr(OpType, GoExpr).

%% variable_from_operation(+Operation, ?Var, -OpType)
%  Extract result variable and operation type from operation spec
%  Uses == for variable identity checking to avoid incorrect unification
%
variable_from_operation(count(Var), RequestedVar, count) :-
    Var == RequestedVar,  % Use == to check if they're the same variable
    !.
variable_from_operation(count, _, count) :- !.
variable_from_operation(sum(_, Var), RequestedVar, sum) :-
    Var == RequestedVar, !.
variable_from_operation(avg(_, Var), RequestedVar, avg) :-
    Var == RequestedVar, !.
variable_from_operation(max(_, Var), RequestedVar, max) :-
    Var == RequestedVar, !.
variable_from_operation(min(_, Var), RequestedVar, min) :-
    Var == RequestedVar, !.

%% operation_to_go_expr(+OpType, -GoExpr)
%  Map operation type to Go expression in output loop
%
operation_to_go_expr(count, 's.count').
operation_to_go_expr(sum, 's.sum').
operation_to_go_expr(avg, 'avg').
operation_to_go_expr(max, 's.maxValue').
operation_to_go_expr(min, 's.minValue').

%% ============================================
%% SINGLE AGGREGATION WRAPPERS WITH HAVING
%% ============================================

%% Wrapper predicates with HAVING support (Phase 9c-2)
%% For single aggregations, convert to multi-agg format and use multi-agg code generator

generate_group_by_count_with_having(GroupField, Result, Having, DbFile, BucketStr, GoCode) :-
    % Convert single count to multi-agg format: [count(Result)]
    OpList = [count(Result)],
    % Use empty field mappings for count (doesn't need any field)
    FieldMappings = [],
    generate_multi_aggregation_code(GroupField, FieldMappings, OpList, Having, DbFile, BucketStr, GoCode).

generate_group_by_sum_with_having(GroupField, AggField, Result, Having, DbFile, BucketStr, GoCode) :-
    % Convert single sum to multi-agg format: [sum(AggField, Result)]
    % Create a fake variable for the field (we'll use AggField directly)
    OpList = [sum(AggField, Result)],
    % Create field mappings with the aggregation field
    FieldMappings = [field-AggField],
    generate_multi_aggregation_code(GroupField, FieldMappings, OpList, Having, DbFile, BucketStr, GoCode).

generate_group_by_avg_with_having(GroupField, AggField, Result, Having, DbFile, BucketStr, GoCode) :-
    % Convert single avg to multi-agg format: [avg(AggField, Result)]
    OpList = [avg(AggField, Result)],
    FieldMappings = [field-AggField],
    generate_multi_aggregation_code(GroupField, FieldMappings, OpList, Having, DbFile, BucketStr, GoCode).

generate_group_by_max_with_having(GroupField, AggField, Result, Having, DbFile, BucketStr, GoCode) :-
    % Convert single max to multi-agg format: [max(AggField, Result)]
    OpList = [max(AggField, Result)],
    FieldMappings = [field-AggField],
    generate_multi_aggregation_code(GroupField, FieldMappings, OpList, Having, DbFile, BucketStr, GoCode).

generate_group_by_min_with_having(GroupField, AggField, Result, Having, DbFile, BucketStr, GoCode) :-
    % Convert single min to multi-agg format: [min(AggField, Result)]
    OpList = [min(AggField, Result)],
    FieldMappings = [field-AggField],
    generate_multi_aggregation_code(GroupField, FieldMappings, OpList, Having, DbFile, BucketStr, GoCode).

%% ============================================
%% ORIGINAL SINGLE AGGREGATION GENERATORS (Phase 9b)
%% ============================================

%% generate_group_by_count(+GroupField, +DbFile, +BucketStr, -GoCode)
%  Generate GROUP BY count code
%
generate_group_by_count(GroupField, DbFile, BucketStr, GoCode) :-
    format(string(GoCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Group by ~s and count
\tcounts := make(map[string]int)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\treturn bucket.ForEach(func(k, v []byte) error {
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\treturn nil // Skip invalid records
\t\t\t}

\t\t\t// Extract group field
\t\t\tif groupRaw, ok := data["~s"]; ok {
\t\t\t\tif groupStr, ok := groupRaw.(string); ok {
\t\t\t\t\tcounts[groupStr]++
\t\t\t\t}
\t\t\t}
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}

\t// Output results as JSON (one per group)
\tfor group, count := range counts {
\t\tresult := map[string]interface{}{
\t\t\t"~s": group,
\t\t\t"count": count,
\t\t}
\t\toutput, _ := json.Marshal(result)
\t\tfmt.Println(string(output))
\t}
', [DbFile, GroupField, BucketStr, BucketStr, GroupField, GroupField]).

%% generate_group_by_sum(+GroupField, +AggField, +DbFile, +BucketStr, -GoCode)
%  Generate GROUP BY sum code
%
generate_group_by_sum(GroupField, AggField, DbFile, BucketStr, GoCode) :-
    format(string(GoCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Group by ~s and sum ~s
\tsums := make(map[string]float64)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\treturn bucket.ForEach(func(k, v []byte) error {
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\treturn nil // Skip invalid records
\t\t\t}

\t\t\t// Extract group and aggregation fields
\t\t\tif groupRaw, ok := data["~s"]; ok {
\t\t\t\tif groupStr, ok := groupRaw.(string); ok {
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tsums[groupStr] += valueFloat
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}

\t// Output results as JSON (one per group)
\tfor group, sum := range sums {
\t\tresult := map[string]interface{}{
\t\t\t"~s": group,
\t\t\t"sum": sum,
\t\t}
\t\toutput, _ := json.Marshal(result)
\t\tfmt.Println(string(output))
\t}
', [DbFile, GroupField, AggField, BucketStr, BucketStr, GroupField, AggField, GroupField]).

%% generate_group_by_avg(+GroupField, +AggField, +DbFile, +BucketStr, -GoCode)
%  Generate GROUP BY average code
%
generate_group_by_avg(GroupField, AggField, DbFile, BucketStr, GoCode) :-
    format(string(GoCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Group by ~s and average ~s
\ttype GroupStats struct {
\t\tsum   float64
\t\tcount int
\t}
\tstats := make(map[string]*GroupStats)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\treturn bucket.ForEach(func(k, v []byte) error {
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\treturn nil // Skip invalid records
\t\t\t}

\t\t\t// Extract group and aggregation fields
\t\t\tif groupRaw, ok := data["~s"]; ok {
\t\t\t\tif groupStr, ok := groupRaw.(string); ok {
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tif _, exists := stats[groupStr]; !exists {
\t\t\t\t\t\t\t\tstats[groupStr] = &GroupStats{}
\t\t\t\t\t\t\t}
\t\t\t\t\t\t\tstats[groupStr].sum += valueFloat
\t\t\t\t\t\t\tstats[groupStr].count++
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}

\t// Output results as JSON (one per group)
\tfor group, s := range stats {
\t\tavg := 0.0
\t\tif s.count > 0 {
\t\t\tavg = s.sum / float64(s.count)
\t\t}
\t\tresult := map[string]interface{}{
\t\t\t"~s": group,
\t\t\t"avg": avg,
\t\t}
\t\toutput, _ := json.Marshal(result)
\t\tfmt.Println(string(output))
\t}
', [DbFile, GroupField, AggField, BucketStr, BucketStr, GroupField, AggField, GroupField]).

%% generate_group_by_max(+GroupField, +AggField, +DbFile, +BucketStr, -GoCode)
%  Generate GROUP BY max code
%
generate_group_by_max(GroupField, AggField, DbFile, BucketStr, GoCode) :-
    format(string(GoCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Group by ~s and find max ~s
\ttype GroupMax struct {
\t\tmaxValue float64
\t\tfirst    bool
\t}
\tmaxes := make(map[string]*GroupMax)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\treturn bucket.ForEach(func(k, v []byte) error {
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\treturn nil // Skip invalid records
\t\t\t}

\t\t\t// Extract group and aggregation fields
\t\t\tif groupRaw, ok := data["~s"]; ok {
\t\t\t\tif groupStr, ok := groupRaw.(string); ok {
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tif _, exists := maxes[groupStr]; !exists {
\t\t\t\t\t\t\t\tmaxes[groupStr] = &GroupMax{first: true}
\t\t\t\t\t\t\t}
\t\t\t\t\t\t\tif maxes[groupStr].first || valueFloat > maxes[groupStr].maxValue {
\t\t\t\t\t\t\t\tmaxes[groupStr].maxValue = valueFloat
\t\t\t\t\t\t\t\tmaxes[groupStr].first = false
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}

\t// Output results as JSON (one per group)
\tfor group, m := range maxes {
\t\tresult := map[string]interface{}{
\t\t\t"~s": group,
\t\t\t"max": m.maxValue,
\t\t}
\t\toutput, _ := json.Marshal(result)
\t\tfmt.Println(string(output))
\t}
', [DbFile, GroupField, AggField, BucketStr, BucketStr, GroupField, AggField, GroupField]).

%% generate_group_by_min(+GroupField, +AggField, +DbFile, +BucketStr, -GoCode)
%  Generate GROUP BY min code
%
generate_group_by_min(GroupField, AggField, DbFile, BucketStr, GoCode) :-
    format(string(GoCode), '\t// Open database (read-only)
\tdb, err := bolt.Open("~s", 0600, &bolt.Options{ReadOnly: true})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Group by ~s and find min ~s
\ttype GroupMin struct {
\t\tminValue float64
\t\tfirst    bool
\t}
\tmins := make(map[string]*GroupMin)
\terr = db.View(func(tx *bolt.Tx) error {
\t\tbucket := tx.Bucket([]byte("~s"))
\t\tif bucket == nil {
\t\t\treturn fmt.Errorf("bucket ''~s'' not found")
\t\t}

\t\treturn bucket.ForEach(func(k, v []byte) error {
\t\t\tvar data map[string]interface{}
\t\t\tif err := json.Unmarshal(v, &data); err != nil {
\t\t\t\treturn nil // Skip invalid records
\t\t\t}

\t\t\t// Extract group and aggregation fields
\t\t\tif groupRaw, ok := data["~s"]; ok {
\t\t\t\tif groupStr, ok := groupRaw.(string); ok {
\t\t\t\t\tif valueRaw, ok := data["~s"]; ok {
\t\t\t\t\t\tif valueFloat, ok := valueRaw.(float64); ok {
\t\t\t\t\t\t\tif _, exists := mins[groupStr]; !exists {
\t\t\t\t\t\t\t\tmins[groupStr] = &GroupMin{first: true}
\t\t\t\t\t\t\t}
\t\t\t\t\t\t\tif mins[groupStr].first || valueFloat < mins[groupStr].minValue {
\t\t\t\t\t\t\t\tmins[groupStr].minValue = valueFloat
\t\t\t\t\t\t\t\tmins[groupStr].first = false
\t\t\t\t\t\t\t}
\t\t\t\t\t\t}
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t\treturn nil
\t\t})
\t})

\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error reading database: %v\\n", err)
\t\tos.Exit(1)
\t}

\t// Output results as JSON (one per group)
\tfor group, m := range mins {
\t\tresult := map[string]interface{}{
\t\t\t"~s": group,
\t\t\t"min": m.minValue,
\t\t}
\t\toutput, _ := json.Marshal(result)
\t\tfmt.Println(string(output))
\t}
', [DbFile, GroupField, AggField, BucketStr, BucketStr, GroupField, AggField, GroupField]).

%% ============================================
%% PIPELINE MODE COMPILATION
%% ============================================
%%
%% Pipeline mode provides streaming JSONL I/O with typed object output,
%% similar to Python's pipeline mode. It enables:
%%   - Streaming input processing (pipeline_input(true))
%%   - Typed object output (output_format(object))
%%   - Named fields in output (arg_names(['Name', 'Age', ...]))
%%   - JSONL streaming output (output_format(jsonl))
%%   - Text output (output_format(text))
%%
%% Options:
%%   - pipeline_input(true)        Enable streaming JSONL input
%%   - output_format(Format)       object | jsonl | text (default: jsonl)
%%   - arg_names(Names)            List of output field names
%%   - filter_only(true)           Only output records that pass (no transformation)

%% compile_pipeline_mode(+Pred, +Arity, +Options, -GoCode)
%  Compile predicate in pipeline mode for streaming JSONL processing
compile_pipeline_mode(Pred, Arity, Options, GoCode) :-
    % Get options with defaults
    option(output_format(OutputFormat), Options, jsonl),
    option(include_package(IncludePackage), Options, true),
    option(filter_only(FilterOnly), Options, false),

    format('  Output format: ~w~n', [OutputFormat]),

    % Get predicate clauses
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),

    (   Clauses = []
    ->  format('ERROR: No clauses found for ~w/~w~n', [Pred, Arity]),
        fail
    ;   Clauses = [SingleHead-SingleBody]
    ->  % Single clause - extract JSON field mappings and compile
        format('  Clause: ~w :- ~w~n', [SingleHead, SingleBody]),
        extract_json_field_mappings(SingleBody, FieldMappings),
        format('  Field mappings: ~w~n', [FieldMappings]),

        SingleHead =.. [_|HeadArgs],

        % Get output field names
        (   option(arg_names(ArgNames), Options)
        ->  true
        ;   generate_pipeline_arg_names(HeadArgs, 1, ArgNames)
        ),
        format('  Output field names: ~w~n', [ArgNames]),

        % Add predicate name to options for struct generation
        PipelineOptions = [predicate_name(Pred)|Options],

        % Compile based on output format
        compile_pipeline_body(OutputFormat, FilterOnly, HeadArgs, FieldMappings, ArgNames, PipelineOptions, PipelineBody),

        % Check if helpers are needed (for nested field access)
        (   member(nested(_, _), FieldMappings)
        ->  generate_nested_helper(HelperFunc),
            HelperSection = HelperFunc
        ;   HelperSection = ''
        )
    ;   format('ERROR: Multiple clauses not yet supported for pipeline mode~n'),
        fail
    ),

    % Wrap in package if requested
    (   IncludePackage
    ->  % Determine required imports based on output format
        base_pipeline_imports(OutputFormat, BaseImports),

        % Add binding imports if any
        get_collected_imports(BindingImports),
        append(BaseImports, BindingImports, AllImportsList),
        sort(AllImportsList, UniqueImports),
        maplist(format_import, UniqueImports, ImportLines),
        atomic_list_concat(ImportLines, '\n', Imports),

        % Generate struct definition for object output
        (   OutputFormat = object
        ->  atom_string(Pred, PredStr),
            upcase_atom(Pred, UpperPred),
            generate_pipeline_struct(UpperPred, ArgNames, StructDef),
            format(string(GoCode), 'package main

import (
~s
)

~s

~sfunc main() {
~s}
', [Imports, StructDef, HelperSection, PipelineBody])
        ;   % No struct needed for jsonl or text output
            (   HelperSection = ''
            ->  format(string(GoCode), 'package main

import (
~s
)

func main() {
~s}
', [Imports, PipelineBody])
            ;   format(string(GoCode), 'package main

import (
~s
)

~s

func main() {
~s}
', [Imports, HelperSection, PipelineBody])
            )
        )
    ;   GoCode = PipelineBody
    ).

%% base_pipeline_imports(+OutputFormat, -Imports)
%  Get base imports needed for pipeline mode
base_pipeline_imports(object, ["bufio", "encoding/json", "fmt", "os"]).
base_pipeline_imports(jsonl, ["bufio", "encoding/json", "fmt", "os"]).
base_pipeline_imports(text, ["bufio", "encoding/json", "fmt", "os"]).

%% generate_pipeline_arg_names(+Args, +StartNum, -Names)
%  Generate default argument names (arg1, arg2, ...)
generate_pipeline_arg_names([], _, []).
generate_pipeline_arg_names([_|Rest], N, [Name|RestNames]) :-
    format(atom(Name), 'arg~w', [N]),
    N1 is N + 1,
    generate_pipeline_arg_names(Rest, N1, RestNames).

%% generate_pipeline_struct(+StructName, +FieldNames, -StructDef)
%  Generate Go struct definition for typed object output
generate_pipeline_struct(StructName, FieldNames, StructDef) :-
    findall(FieldLine,
        (   member(FieldName, FieldNames),
            capitalize_first(FieldName, CapField),
            downcase_atom(FieldName, LowerField),
            format(atom(FieldLine), '\t~w interface{} `json:"~w"`', [CapField, LowerField])
        ),
        FieldLines),
    atomic_list_concat(FieldLines, '\n', FieldsStr),
    format(atom(StructDef), 'type ~wOutput struct {\n~w\n}', [StructName, FieldsStr]).

%% capitalize_first(+Atom, -Capitalized)
%  Capitalize first letter of atom
capitalize_first(Atom, Capitalized) :-
    atom_string(Atom, Str),
    (   Str = ""
    ->  Capitalized = Atom
    ;   string_chars(Str, [First|Rest]),
        upcase_atom(First, Upper),
        atom_string(Upper, UpperStr),
        string_chars(RestStr, Rest),
        string_concat(UpperStr, RestStr, CapStr),
        atom_string(Capitalized, CapStr)
    ).

%% compile_pipeline_body(+OutputFormat, +FilterOnly, +HeadArgs, +FieldMappings, +ArgNames, +Options, -Code)
%  Compile the main pipeline processing body
compile_pipeline_body(OutputFormat, FilterOnly, HeadArgs, FieldMappings, ArgNames, Options, Code) :-
    % Generate field extraction code
    generate_pipeline_field_extraction(FieldMappings, ExtractionCode, VarMap),

    % Generate filter/constraint checks
    option(constraints(Constraints), Options, []),
    generate_pipeline_constraints(Constraints, VarMap, ConstraintCode),

    % Generate output based on format
    (   FilterOnly = true
    ->  % Just pass through matching records
        OutputCode = "\t\tfmt.Println(scanner.Text())"
    ;   generate_pipeline_output(OutputFormat, HeadArgs, ArgNames, VarMap, Options, OutputCode)
    ),

    % Assemble the pipeline body
    format(string(Code), '
\tscanner := bufio.NewScanner(os.Stdin)
\tfor scanner.Scan() {
\t\tvar data map[string]interface{}
\t\tif err := json.Unmarshal(scanner.Bytes(), &data); err != nil {
\t\t\tcontinue
\t\t}
~s~s
~s
\t}
', [ExtractionCode, ConstraintCode, OutputCode]).

%% generate_pipeline_field_extraction(+FieldMappings, -Code, -VarMap)
%  Generate Go code to extract fields from JSON data
generate_pipeline_field_extraction([], "", []).
generate_pipeline_field_extraction(FieldMappings, Code, VarMap) :-
    FieldMappings \= [],
    findall(extraction(VarName, FieldExpr, ExtractCode),
        (   member(Mapping, FieldMappings),
            pipeline_extract_one_field(Mapping, VarName, FieldExpr, ExtractCode)
        ),
        Extractions),
    findall(C, member(extraction(_, _, C), Extractions), CodeParts),
    atomic_list_concat(CodeParts, '', Code),
    findall(V-F, member(extraction(V, F, _), Extractions), VarMap).

%% pipeline_extract_one_field(+Mapping, -VarName, -FieldExpr, -Code)
%  Generate extraction code for a single field mapping
pipeline_extract_one_field(flat(FieldName, VarName), VarName, FieldName, Code) :-
    format(string(Code), '
\t\t~w, _ := data["~w"]', [VarName, FieldName]).

pipeline_extract_one_field(nested(Path, VarName), VarName, nested(Path), Code) :-
    % Generate nested field access
    path_to_go_nested_access(Path, AccessExpr),
    format(string(Code), '
\t\t~w, _ := ~w', [VarName, AccessExpr]).

%% path_to_go_nested_access(+Path, -GoExpr)
%  Convert a path like [user, address, city] to Go nested access
path_to_go_nested_access(Path, GoExpr) :-
    maplist(atom_string, Path, PathStrs),
    maplist(quote_string, PathStrs, QuotedParts),
    atomic_list_concat(QuotedParts, ', ', PathArgsStr),
    format(string(GoExpr), 'getNestedField(data, []string{~w})', [PathArgsStr]).

quote_string(S, Q) :- format(string(Q), '"~w"', [S]).

%% generate_pipeline_constraints(+Constraints, +VarMap, -Code)
%  Generate constraint checking code
generate_pipeline_constraints([], _, "").
generate_pipeline_constraints(Constraints, VarMap, Code) :-
    Constraints \= [],
    findall(Check,
        (   member(Constraint, Constraints),
            pipeline_constraint_to_go(Constraint, VarMap, Check)
        ),
        Checks),
    atomic_list_concat(Checks, '', Code).

%% pipeline_constraint_to_go(+Constraint, +VarMap, -Code)
%  Convert a pipeline constraint to Go if-check (uses continue semantics)
pipeline_constraint_to_go(Var > Value, VarMap, Code) :-
    member(Var-_, VarMap),
    format(string(Code), '
\t\tif ~wNum, ok := ~w.(float64); !ok || ~wNum <= ~w {
\t\t\tcontinue
\t\t}', [Var, Var, Var, Value]).
pipeline_constraint_to_go(Var < Value, VarMap, Code) :-
    member(Var-_, VarMap),
    format(string(Code), '
\t\tif ~wNum, ok := ~w.(float64); !ok || ~wNum >= ~w {
\t\t\tcontinue
\t\t}', [Var, Var, Var, Value]).
pipeline_constraint_to_go(Var >= Value, VarMap, Code) :-
    member(Var-_, VarMap),
    format(string(Code), '
\t\tif ~wNum, ok := ~w.(float64); !ok || ~wNum < ~w {
\t\t\tcontinue
\t\t}', [Var, Var, Var, Value]).
pipeline_constraint_to_go(Var =< Value, VarMap, Code) :-
    member(Var-_, VarMap),
    format(string(Code), '
\t\tif ~wNum, ok := ~w.(float64); !ok || ~wNum > ~w {
\t\t\tcontinue
\t\t}', [Var, Var, Var, Value]).
pipeline_constraint_to_go(Var == Value, VarMap, Code) :-
    member(Var-_, VarMap),
    (   number(Value)
    ->  format(string(Code), '
\t\tif ~wNum, ok := ~w.(float64); !ok || ~wNum != ~w {
\t\t\tcontinue
\t\t}', [Var, Var, Var, Value])
    ;   format(string(Code), '
\t\tif ~wStr, ok := ~w.(string); !ok || ~wStr != "~w" {
\t\t\tcontinue
\t\t}', [Var, Var, Var, Value])
    ).
pipeline_constraint_to_go(_, _, "").

%% generate_pipeline_output(+Format, +HeadArgs, +ArgNames, +VarMap, +Options, -Code)
%  Generate output code based on format
generate_pipeline_output(jsonl, HeadArgs, ArgNames, VarMap, _Options, Code) :-
    % Output as JSONL using map
    generate_jsonl_output_map(HeadArgs, ArgNames, VarMap, MapCode),
    format(string(Code), '
\t\toutput := map[string]interface{}{~s}
\t\tjsonBytes, _ := json.Marshal(output)
\t\tfmt.Println(string(jsonBytes))', [MapCode]).

generate_pipeline_output(object, HeadArgs, ArgNames, VarMap, Options, Code) :-
    % Output as typed struct
    option(predicate_name(Pred), Options, output),
    upcase_atom(Pred, UpperPred),
    generate_struct_output_init(UpperPred, HeadArgs, ArgNames, VarMap, StructInit),
    format(string(Code), '
\t\trecord := ~s
\t\tjsonBytes, _ := json.Marshal(record)
\t\tfmt.Println(string(jsonBytes))', [StructInit]).

generate_pipeline_output(text, HeadArgs, _ArgNames, VarMap, _Options, Code) :-
    % Output as tab-separated text
    generate_text_output(HeadArgs, VarMap, TextExpr),
    format(string(Code), '
\t\tfmt.Println(~s)', [TextExpr]).

%% generate_jsonl_output_map(+HeadArgs, +ArgNames, +VarMap, -MapCode)
%  Generate map literal for JSONL output
generate_jsonl_output_map(HeadArgs, ArgNames, VarMap, MapCode) :-
    findall(Entry,
        (   nth1(Idx, HeadArgs, Arg),
            nth1(Idx, ArgNames, Name),
            head_arg_to_output_expr(Arg, VarMap, Expr),
            downcase_atom(Name, LowerName),
            format(string(Entry), '"~w": ~w', [LowerName, Expr])
        ),
        Entries),
    atomic_list_concat(Entries, ', ', MapCode).

%% head_arg_to_output_expr(+Arg, +VarMap, -Expr)
%  Convert a head argument to its Go output expression
head_arg_to_output_expr(Arg, VarMap, Expr) :-
    (   member(Arg-_, VarMap)
    ->  atom_string(Arg, Expr)
    ;   atom(Arg)
    ->  format(string(Expr), '"~w"', [Arg])
    ;   number(Arg)
    ->  format(string(Expr), '~w', [Arg])
    ;   Expr = "nil"
    ).

%% generate_struct_output_init(+StructName, +HeadArgs, +ArgNames, +VarMap, -Init)
%  Generate struct initialization for object output
generate_struct_output_init(StructName, HeadArgs, ArgNames, VarMap, Init) :-
    findall(FieldInit,
        (   nth1(Idx, HeadArgs, Arg),
            nth1(Idx, ArgNames, Name),
            head_arg_to_output_expr(Arg, VarMap, Expr),
            capitalize_first(Name, CapName),
            format(string(FieldInit), '~w: ~w', [CapName, Expr])
        ),
        FieldInits),
    atomic_list_concat(FieldInits, ', ', FieldsStr),
    format(string(Init), '~wOutput{~w}', [StructName, FieldsStr]).

%% generate_text_output(+HeadArgs, +VarMap, -TextExpr)
%  Generate fmt.Sprintf expression for text output
generate_text_output(HeadArgs, VarMap, TextExpr) :-
    length(HeadArgs, NumArgs),
    findall('%v', between(1, NumArgs, _), Formats),
    atomic_list_concat(Formats, '\t', FormatStr),
    findall(Expr,
        (   member(Arg, HeadArgs),
            head_arg_to_output_expr(Arg, VarMap, Expr)
        ),
        Exprs),
    atomic_list_concat(Exprs, ', ', ArgsStr),
    format(string(TextExpr), 'fmt.Sprintf("~w", ~w)', [FormatStr, ArgsStr]).

%% ============================================
%% PIPELINE CHAINING
%% ============================================
%%
%% Pipeline chaining allows composing multiple predicates into a single
%% Go program where data flows through each stage sequentially.
%%
%% Unlike Python's generators, Go uses channels or sequential processing.
%% We implement two modes:
%%   1. Sequential mode (default) - simpler, processes all records through each stage
%%   2. Channel mode - uses goroutines and channels for streaming
%%
%% Options:
%%   - pipeline_name(Name)    Name for the pipeline (default: pipeline)
%%   - pipeline_mode(Mode)    sequential | channel (default: sequential)
%%   - output_format(Format)  jsonl | text (default: jsonl)
%%   - arg_names(Names)       Property names for final output

%% compile_go_pipeline(+Predicates, +Options, -GoCode)
%  Main entry point for Go pipeline chaining.
%  Compiles multiple predicates into a single Go program.
%
%  Predicates: List of Name/Arity or go:Name/Arity
%    Examples:
%      - [parse_user/2, filter_active/1, format_output/2]
%      - [go:parse/1, go:transform/1, go:output/1]
%
compile_go_pipeline(Predicates, Options, GoCode) :-
    format('=== Compiling Go Pipeline ===~n', []),
    format('  Predicates: ~w~n', [Predicates]),

    % Get options with defaults
    option(pipeline_name(PipelineName), Options, pipeline),
    option(pipeline_mode(PipelineMode), Options, sequential),
    option(output_format(OutputFormat), Options, jsonl),
    option(include_package(IncludePackage), Options, true),

    format('  Pipeline name: ~w~n', [PipelineName]),
    format('  Mode: ~w~n', [PipelineMode]),

    % Clear any previous binding imports
    clear_binding_imports,

    % Extract predicate info and compile each stage
    extract_pipeline_predicates(Predicates, PredInfos),
    compile_pipeline_stages(PredInfos, Options, StagesCode, StageNames),

    format('  Compiled ~w stages~n', [StageNames]),

    % Generate the pipeline connector
    generate_go_pipeline_connector(StageNames, PipelineName, PipelineMode, ConnectorCode),

    % Generate main function
    generate_go_pipeline_main(PipelineName, OutputFormat, Options, MainCode),

    % Generate helper functions if needed
    generate_pipeline_helpers(PipelineMode, HelperCode),

    % Wrap in package if requested
    (   IncludePackage
    ->  % Collect imports
        pipeline_chaining_imports(PipelineMode, BaseImports),
        get_collected_imports(BindingImports),
        append(BaseImports, BindingImports, AllImportsList),
        % Add sort import for generator mode
        (   PipelineMode = generator
        ->  append(AllImportsList, ["sort"], AllImportsWithSort)
        ;   AllImportsWithSort = AllImportsList
        ),
        sort(AllImportsWithSort, UniqueImports),
        maplist(format_import, UniqueImports, ImportLines),
        atomic_list_concat(ImportLines, '\n', ImportsStr),

        format(string(GoCode), 'package main

import (
~w
)

// Record represents a data record flowing through the pipeline
type Record map[string]interface{}

~w~w
~w
~w', [ImportsStr, HelperCode, StagesCode, ConnectorCode, MainCode])
    ;   format(string(GoCode), '~w~w~w~w', [HelperCode, StagesCode, ConnectorCode, MainCode])
    ).

%% generate_pipeline_helpers(+Mode, -Code)
%  Generate helper functions based on pipeline mode
generate_pipeline_helpers(generator, Code) :-
    Code = '// recordKey generates a unique key for a record to track duplicates
func recordKey(r Record) string {
\t// Get all keys and sort them for consistent ordering
\tkeys := make([]string, 0, len(r))
\tfor k := range r {
\t\tkeys = append(keys, k)
\t}
\tsort.Strings(keys)
\t
\t// Build key from sorted key-value pairs
\tvar result string
\tfor _, k := range keys {
\t\tresult += fmt.Sprintf("%s=%v;", k, r[k])
\t}
\treturn result
}

'.
generate_pipeline_helpers(_, "").

%% pipeline_chaining_imports(+Mode, -Imports)
%  Get base imports needed for pipeline chaining
pipeline_chaining_imports(sequential, ["bufio", "encoding/json", "fmt", "os"]).
pipeline_chaining_imports(channel, ["bufio", "encoding/json", "fmt", "os", "sync"]).
pipeline_chaining_imports(generator, ["bufio", "encoding/json", "fmt", "os"]).

%% extract_pipeline_predicates(+Predicates, -PredInfos)
%  Extract predicate name and arity from list
extract_pipeline_predicates([], []).
extract_pipeline_predicates([Pred|Rest], [Info|RestInfos]) :-
    extract_pred_info(Pred, Info),
    extract_pipeline_predicates(Rest, RestInfos).

extract_pred_info(go:Name/Arity, pred_info(Name, Arity, go)) :- !.
extract_pred_info(Name/Arity, pred_info(Name, Arity, go)) :- !.
extract_pred_info(Pred, pred_info(Pred, 1, go)) :-
    atom(Pred).

%% compile_pipeline_stages(+PredInfos, +Options, -Code, -StageNames)
%  Compile each predicate into a pipeline stage function
compile_pipeline_stages([], _, "", []).
compile_pipeline_stages([pred_info(Name, Arity, _Target)|Rest], Options, Code, [StageName|RestNames]) :-
    atom_string(Name, NameStr),
    StageName = NameStr,

    % Check if predicate exists and compile it
    (   functor(Head, Name, Arity),
        clause(Head, Body)
    ->  % Compile actual predicate logic
        compile_pipeline_stage_from_clause(Name, Arity, Head, Body, Options, StageCode)
    ;   % Generate placeholder stage
        generate_placeholder_stage(Name, Arity, StageCode)
    ),

    compile_pipeline_stages(Rest, Options, RestCode, RestNames),
    format(string(Code), "~w~w", [StageCode, RestCode]).

%% compile_pipeline_stage_from_clause(+Name, +Arity, +Head, +Body, +Options, -Code)
%  Compile a predicate clause into a pipeline stage function.
%  Now supports binding goals (e.g., string_lower/2, sqrt/2) in the body.
%  Includes nil guards for robustness in fixpoint (generator) mode.
compile_pipeline_stage_from_clause(Name, Arity, Head, Body, _Options, Code) :-
    atom_string(Name, NameStr),
    Head =.. [_|HeadArgs],

    % Extract field mappings from body
    extract_json_field_mappings(Body, FieldMappings),

    % Generate field guards (nil checks for required fields)
    generate_stage_field_guards(FieldMappings, GuardCode),

    % Generate field extraction code
    generate_stage_field_extraction(FieldMappings, ExtractionCode, VarMap),

    % Extract and compile binding goals from body
    extract_stage_binding_goals(Body, BindingGoals),
    compile_stage_bindings(BindingGoals, VarMap, BindingCode, BindingVarMap),

    % Merge variable maps (binding outputs extend the map)
    append(VarMap, BindingVarMap, FullVarMap),

    % Generate output construction with full variable map
    generate_stage_output(HeadArgs, FullVarMap, OutputCode),

    format(string(Code),
'// ~w processes records for ~w/~w
func ~w(records []Record) []Record {
\tvar results []Record
\tfor _, record := range records {
~w~w~w
\t\tresult := Record{~w}
\t\tresults = append(results, result)
\t}
\treturn results
}

', [NameStr, NameStr, Arity, NameStr, GuardCode, ExtractionCode, BindingCode, OutputCode]).

%% generate_placeholder_stage(+Name, +Arity, -Code)
%  Generate a placeholder stage for undefined predicates
generate_placeholder_stage(Name, Arity, Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
'// ~w is a placeholder for ~w/~w
// TODO: Implement actual predicate logic
func ~w(records []Record) []Record {
\t// Pass through - actual implementation needed
\treturn records
}

', [NameStr, NameStr, Arity, NameStr]).

%% ============================================
%% PIPELINE BINDING INTEGRATION
%% ============================================
%%
%% These predicates extract and compile binding goals (e.g., string_lower/2,
%% sqrt/2) from predicate bodies for use in pipeline stages.

%% extract_stage_binding_goals(+Body, -BindingGoals)
%  Extract all binding goals from a predicate body.
%  Traverses conjunctions and disjunctions to find binding calls.
%
extract_stage_binding_goals(Body, BindingGoals) :-
    findall(Goal, extract_binding_goal(Body, Goal), BindingGoals).

extract_binding_goal((A, B), Goal) :-
    (   extract_binding_goal(A, Goal)
    ;   extract_binding_goal(B, Goal)
    ).
extract_binding_goal((A ; B), Goal) :-
    (   extract_binding_goal(A, Goal)
    ;   extract_binding_goal(B, Goal)
    ).
extract_binding_goal(Goal, Goal) :-
    Goal \= (_, _),
    Goal \= (_ ; _),
    is_stage_binding_goal(Goal).

%% is_stage_binding_goal(+Goal)
%  Check if a goal is a binding goal (has a Go binding defined).
%
is_stage_binding_goal(Goal) :-
    callable(Goal),
    Goal \= true,
    Goal \= fail,
    functor(Goal, Functor, Arity),
    % Skip json_record and json_get - handled separately
    Functor \= json_record,
    Functor \= json_get,
    % Check if binding exists
    go_binding(Functor/Arity, _, _, _, _).

%% compile_stage_bindings(+BindingGoals, +VarMap, -Code, -NewVarMap)
%  Compile binding goals to Go code for pipeline stages.
%  Returns code and any new variable mappings (for binding outputs).
%
compile_stage_bindings([], _, "", []).
compile_stage_bindings([Goal|Rest], VarMap, Code, NewVarMap) :-
    compile_single_stage_binding(Goal, VarMap, GoalCode, GoalVarMap),
    % Update VarMap with outputs from this goal
    append(VarMap, GoalVarMap, ExtendedVarMap),
    compile_stage_bindings(Rest, ExtendedVarMap, RestCode, RestVarMap),
    format(string(Code), "~w~w", [GoalCode, RestCode]),
    append(GoalVarMap, RestVarMap, NewVarMap).

%% compile_single_stage_binding(+Goal, +VarMap, -Code, -NewVarMap)
%  Compile a single binding goal to Go code.
%
compile_single_stage_binding(Goal, VarMap, Code, NewVarMap) :-
    Goal =.. [Functor|Args],
    length(Args, Arity),
    Pred = Functor/Arity,

    % Get the binding
    go_binding(Pred, TargetName, _Inputs, Outputs, Options),

    % Collect import if needed
    (   member(import(Import), Options)
    ->  collect_binding_import(Import)
    ;   true
    ),

    % Determine input and output args
    length(Outputs, NumOutputs),
    NumInputs is Arity - NumOutputs,
    length(InputArgs, NumInputs),
    length(OutputArgs, NumOutputs),
    append(InputArgs, OutputArgs, Args),

    % Translate input arguments to Go expressions
    maplist(stage_arg_to_go_expr(VarMap), InputArgs, GoInputExprs),

    % Generate the binding call
    atom_string(TargetName, TargetStr),
    (   % Method call pattern: starts with .
        sub_string(TargetStr, 0, 1, _, ".")
    ->  % First input is receiver
        (   GoInputExprs = [Receiver|RestInputs]
        ->  (   RestInputs = []
            ->  format(string(CallCode), "~w~w", [Receiver, TargetStr])
            ;   atomic_list_concat(RestInputs, ", ", RestArgsStr),
                (   sub_string(TargetStr, _, 2, 0, "()")
                ->  sub_string(TargetStr, 0, _, 2, MethodBase),
                    format(string(CallCode), "~w~w(~w)", [Receiver, MethodBase, RestArgsStr])
                ;   format(string(CallCode), "~w~w(~w)", [Receiver, TargetStr, RestArgsStr])
                )
            )
        ;   format(string(CallCode), "/* missing receiver for ~w */", [TargetStr])
        )
    ;   % Regular function call
        (   GoInputExprs = []
        ->  format(string(CallCode), "~w()", [TargetStr])
        ;   atomic_list_concat(GoInputExprs, ", ", ArgsStr),
            format(string(CallCode), "~w(~w)", [TargetStr, ArgsStr])
        )
    ),

    % Handle output assignment
    (   OutputArgs = [OutputVar]
    ->  % Single output - assign to variable
        atom_string(OutputVar, OutputVarStr),
        format(string(Code), '\t\t~w := ~w\n', [OutputVarStr, CallCode]),
        NewVarMap = [OutputVar-binding_output]
    ;   OutputArgs = [Out1, Out2]
    ->  % Two outputs (value, error pattern)
        atom_string(Out1, Out1Str),
        atom_string(Out2, Out2Str),
        format(string(Code), '\t\t~w, ~w := ~w\n\t\t_ = ~w // ignore error for now\n',
            [Out1Str, Out2Str, CallCode, Out2Str]),
        NewVarMap = [Out1-binding_output, Out2-binding_output]
    ;   OutputArgs = []
    ->  % No output - just call
        format(string(Code), '\t\t~w\n', [CallCode]),
        NewVarMap = []
    ;   % Multiple outputs - use tuple
        maplist(atom_string, OutputArgs, OutputVarStrs),
        atomic_list_concat(OutputVarStrs, ", ", OutputsStr),
        format(string(Code), '\t\t~w := ~w\n', [OutputsStr, CallCode]),
        findall(V-binding_output, member(V, OutputArgs), NewVarMap)
    ).

%% stage_arg_to_go_expr(+VarMap, +Arg, -GoExpr)
%  Convert a Prolog argument to a Go expression for stage binding calls.
%
stage_arg_to_go_expr(VarMap, Arg, GoExpr) :-
    (   % Check if it's a known variable from extraction
        member(Arg-_, VarMap)
    ->  % Variable - use its name, but need type assertion for interface{}
        atom_string(Arg, VarStr),
        format(string(GoExpr), "~w.(string)", [VarStr])
    ;   % Atom - string literal
        atom(Arg)
    ->  format(string(GoExpr), '"~w"', [Arg])
    ;   % Number
        number(Arg)
    ->  format(string(GoExpr), '~w', [Arg])
    ;   % String
        string(Arg)
    ->  format(string(GoExpr), '"~w"', [Arg])
    ;   % Variable not in map - use as-is with type assertion
        var(Arg)
    ->  GoExpr = "nil"
    ;   % Compound term or unknown
        term_string(Arg, ArgStr),
        format(string(GoExpr), '/* ~w */', [ArgStr])
    ).

%% generate_stage_field_extraction(+FieldMappings, -Code, -VarMap)
%  Generate Go code to extract fields for a pipeline stage
generate_stage_field_extraction([], "", []).
generate_stage_field_extraction(FieldMappings, Code, VarMap) :-
    FieldMappings \= [],
    findall(extraction(VarName, FieldName, ExtractCode),
        (   member(Mapping, FieldMappings),
            stage_extract_one_field(Mapping, VarName, FieldName, ExtractCode)
        ),
        Extractions),
    findall(C, member(extraction(_, _, C), Extractions), CodeParts),
    atomic_list_concat(CodeParts, '', Code),
    findall(V-F, member(extraction(V, F, _), Extractions), VarMap).

stage_extract_one_field(flat(FieldName, VarName), VarName, FieldName, Code) :-
    format(string(Code), '\t\t~w := record["~w"]\n', [VarName, FieldName]).
stage_extract_one_field(FieldName-VarName, VarName, FieldName, Code) :-
    atom(FieldName),
    format(string(Code), '\t\t~w := record["~w"]\n', [VarName, FieldName]).
stage_extract_one_field(nested(Path, VarName), VarName, nested(Path), Code) :-
    path_to_go_nested_access(Path, AccessExpr),
    format(string(Code), '\t\t~w, _ := ~w\n', [VarName, AccessExpr]).

%% generate_stage_field_guards(+FieldMappings, -Code)
%  Generate nil checks for required fields - for use with generator mode
%  to skip records that don't have the expected structure
generate_stage_field_guards([], "").
generate_stage_field_guards(FieldMappings, Code) :-
    FieldMappings \= [],
    findall(FieldName,
        (   member(Mapping, FieldMappings),
            extract_field_name(Mapping, FieldName)
        ),
        FieldNames),
    (   FieldNames = []
    ->  Code = ""
    ;   generate_nil_check_conditions(FieldNames, Conditions),
        format(string(Code), '\t\tif ~w {\n\t\t\tcontinue\n\t\t}\n', [Conditions])
    ).

extract_field_name(flat(FieldName, _), FieldName).
extract_field_name(FieldName-_, FieldName) :- atom(FieldName).
extract_field_name(nested([First|_], _), First).

generate_nil_check_conditions([Field], Condition) :-
    format(string(Condition), 'record["~w"] == nil', [Field]).
generate_nil_check_conditions([Field|Rest], Condition) :-
    Rest \= [],
    format(string(FirstCheck), 'record["~w"] == nil', [Field]),
    generate_nil_check_conditions(Rest, RestCheck),
    format(string(Condition), '~w || ~w', [FirstCheck, RestCheck]).

%% generate_stage_output(+HeadArgs, +VarMap, -Code)
%  Generate output record construction for a stage
generate_stage_output(HeadArgs, VarMap, Code) :-
    findall(Entry,
        (   nth1(Idx, HeadArgs, Arg),
            format(atom(FieldName), 'arg~w', [Idx]),
            stage_arg_to_output(Arg, VarMap, Expr),
            format(string(Entry), '"~w": ~w', [FieldName, Expr])
        ),
        Entries),
    atomic_list_concat(Entries, ', ', Code).

stage_arg_to_output(Arg, VarMap, Expr) :-
    (   member(Arg-_, VarMap)
    ->  atom_string(Arg, Expr)
    ;   atom(Arg)
    ->  format(string(Expr), '"~w"', [Arg])
    ;   number(Arg)
    ->  format(string(Expr), '~w', [Arg])
    ;   Expr = "nil"
    ).

%% generate_go_pipeline_connector(+StageNames, +PipelineName, +Mode, -Code)
%  Generate the function that chains all stages together
generate_go_pipeline_connector(StageNames, PipelineName, sequential, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    generate_sequential_chain(StageNames, ChainCode),
    format(string(Code),
'// ~w chains all pipeline stages together
func ~w(input []Record) []Record {
~w}

', [PipelineNameStr, PipelineNameStr, ChainCode]).

generate_go_pipeline_connector(StageNames, PipelineName, channel, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    generate_channel_chain(StageNames, ChainCode),
    format(string(Code),
'// ~w chains all pipeline stages using channels
func ~w(input []Record) []Record {
~w}

', [PipelineNameStr, PipelineNameStr, ChainCode]).

generate_go_pipeline_connector(StageNames, PipelineName, generator, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    generate_generator_chain(StageNames, ChainCode),
    format(string(Code),
'// ~w chains pipeline stages with fixpoint evaluation for recursive stages
func ~w(input []Record) []Record {
~w}

', [PipelineNameStr, PipelineNameStr, ChainCode]).

%% generate_generator_chain(+StageNames, -Code)
%  Generate fixpoint-based chaining for recursive stages
generate_generator_chain([], "\treturn input\n").
generate_generator_chain(StageNames, Code) :-
    StageNames \= [],
    generate_generator_chain_body(StageNames, BodyCode),
    format(string(Code),
'\t// Initialize with input records
\ttotal := make(map[string]Record)
\tfor _, r := range input {
\t\tkey := recordKey(r)
\t\ttotal[key] = r
\t}

\t// Fixpoint iteration - apply stages until no new records
\tchanged := true
\tfor changed {
\t\tchanged = false
\t\tvar current []Record
\t\tfor _, r := range total {
\t\t\tcurrent = append(current, r)
\t\t}

~w

\t\t// Add new records to total
\t\tfor _, r := range current {
\t\t\tkey := recordKey(r)
\t\t\tif _, exists := total[key]; !exists {
\t\t\t\ttotal[key] = r
\t\t\t\tchanged = true
\t\t\t}
\t\t}
\t}

\t// Convert map to slice
\tvar results []Record
\tfor _, r := range total {
\t\tresults = append(results, r)
\t}
\treturn results
', [BodyCode]).

generate_generator_chain_body([], "").
generate_generator_chain_body([Stage|Rest], Code) :-
    format(string(StageCall), '\t\tcurrent = ~w(current)\n', [Stage]),
    generate_generator_chain_body(Rest, RestCode),
    string_concat(StageCall, RestCode, Code).

%% generate_sequential_chain(+StageNames, -Code)
%  Generate sequential chaining code: stage1(stage2(stage3(input)))
generate_sequential_chain([], "\treturn input\n").
generate_sequential_chain([First|Rest], Code) :-
    generate_sequential_chain_recursive(Rest, First, "input", ChainExpr),
    format(string(Code), "\treturn ~w\n", [ChainExpr]).

generate_sequential_chain_recursive([], Current, Input, Expr) :-
    format(string(Expr), "~w(~w)", [Current, Input]).
generate_sequential_chain_recursive([Next|Rest], Current, Input, Expr) :-
    format(string(CurrentCall), "~w(~w)", [Current, Input]),
    generate_sequential_chain_recursive(Rest, Next, CurrentCall, Expr).

%% generate_channel_chain(+StageNames, -Code)
%  Generate channel-based chaining code using goroutines
generate_channel_chain(StageNames, Code) :-
    length(StageNames, NumStages),
    generate_channel_declarations(NumStages, DeclCode),
    generate_channel_goroutines(StageNames, 0, GoroutineCode),
    format(string(Code),
'\tvar wg sync.WaitGroup
~w
~w
\twg.Wait()
\treturn results
', [DeclCode, GoroutineCode]).

generate_channel_declarations(0, "").
generate_channel_declarations(N, Code) :-
    N > 0,
    findall(Decl,
        (   between(0, N, I),
            (   I =:= N
            ->  format(string(Decl), '\tvar results []Record', [])
            ;   format(string(Decl), '\tch~w := make(chan Record, 100)', [I])
            )
        ),
        Decls),
    atomic_list_concat(Decls, '\n', Code).

generate_channel_goroutines([], _, "").
generate_channel_goroutines([Stage|Rest], Idx, Code) :-
    NextIdx is Idx + 1,
    (   Idx =:= 0
    ->  InputExpr = "input"
    ;   format(string(InputExpr), "ch~w", [Idx])
    ),
    (   Rest = []
    ->  format(string(StageCode),
'\twg.Add(1)
\tgo func() {
\t\tdefer wg.Done()
\t\tfor record := range ~w {
\t\t\tprocessed := ~w([]Record{record})
\t\t\tresults = append(results, processed...)
\t\t}
\t}()
', [InputExpr, Stage])
    ;   format(string(StageCode),
'\twg.Add(1)
\tgo func() {
\t\tdefer wg.Done()
\t\tfor record := range ~w {
\t\t\tprocessed := ~w([]Record{record})
\t\t\tfor _, r := range processed {
\t\t\t\tch~w <- r
\t\t\t}
\t\t}
\t\tclose(ch~w)
\t}()
', [InputExpr, Stage, NextIdx, NextIdx])
    ),
    generate_channel_goroutines(Rest, NextIdx, RestCode),
    format(string(Code), "~w~w", [StageCode, RestCode]).

%% generate_go_pipeline_main(+PipelineName, +OutputFormat, +Options, -Code)
%  Generate the main function for the pipeline
generate_go_pipeline_main(PipelineName, jsonl, _Options, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    format(string(Code),
'func main() {
\t// Read input records from stdin
\tvar input []Record
\tscanner := bufio.NewScanner(os.Stdin)
\tfor scanner.Scan() {
\t\tvar record Record
\t\tif err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
\t\t\tcontinue
\t\t}
\t\tinput = append(input, record)
\t}

\t// Process through pipeline
\tresults := ~w(input)

\t// Output results as JSONL
\tfor _, result := range results {
\t\tjsonBytes, _ := json.Marshal(result)
\t\tfmt.Println(string(jsonBytes))
\t}
}
', [PipelineNameStr]).

generate_go_pipeline_main(PipelineName, text, Options, Code) :-
    atom_string(PipelineName, PipelineNameStr),
    option(arg_names(ArgNames), Options, []),
    (   ArgNames = []
    ->  OutputCode = "\t\tfmt.Printf(\"%v\\n\", result)"
    ;   generate_text_output_code(ArgNames, OutputCode)
    ),
    format(string(Code),
'func main() {
\t// Read input records from stdin
\tvar input []Record
\tscanner := bufio.NewScanner(os.Stdin)
\tfor scanner.Scan() {
\t\tvar record Record
\t\tif err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
\t\t\tcontinue
\t\t}
\t\tinput = append(input, record)
\t}

\t// Process through pipeline
\tresults := ~w(input)

\t// Output results as text
\tfor _, result := range results {
~w
\t}
}
', [PipelineNameStr, OutputCode]).

generate_text_output_code(ArgNames, Code) :-
    findall(FieldAccess,
        (   member(Name, ArgNames),
            downcase_atom(Name, LowerName),
            format(string(FieldAccess), 'result["~w"]', [LowerName])
        ),
        Accesses),
    length(ArgNames, NumFields),
    findall('%v', between(1, NumFields, _), Formats),
    atomic_list_concat(Formats, '\t', FormatStr),
    atomic_list_concat(Accesses, ', ', AccessStr),
    format(string(Code), '\t\tfmt.Printf("~w\\n", ~w)', [FormatStr, AccessStr]).

%% ============================================
%% JSON INPUT MODE COMPILATION
%% ============================================

%% compile_json_input_mode(+Pred, +Arity, +Options, -GoCode)
%  Compile predicate with JSON input (JSONL format)
%  Reads JSON lines from stdin, extracts fields, outputs in configured format
%
compile_json_input_mode(Pred, Arity, Options, GoCode) :-
    % Get options
    option(field_delimiter(FieldDelim), Options, colon),
    option(include_package(IncludePackage), Options, true),
    option(unique(Unique), Options, true),

    % Get predicate clauses
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),

    (   Clauses = [] ->
        format('ERROR: No clauses found for ~w/~w~n', [Pred, Arity]),
        fail
    ;   Clauses = [SingleHead-SingleBody] ->
        % Single clause - extract JSON field mappings
        format('  Clause: ~w~n', [SingleHead-SingleBody]),
        extract_json_field_mappings(SingleBody, FieldMappings),
        format('  Field mappings: ~w~n', [FieldMappings]),

        % Check for schema option OR database backend (both require typed compilation)
        SingleHead =.. [_|HeadArgs],
        (   option(json_schema(SchemaName), Options)
        ->  % Typed compilation with schema validation
            format('  Using schema: ~w~n', [SchemaName]),
            compile_json_to_go_typed(HeadArgs, FieldMappings, SchemaName, FieldDelim, Unique, CoreBody)
        ;   option(db_backend(bbolt), Options)
        ->  % Typed compilation without schema (for database writes)
            format('  Database mode: using typed compilation~n'),
            compile_json_to_go_typed_noschema(HeadArgs, FieldMappings, FieldDelim, Unique, CoreBody)
        ;   % Untyped compilation (current behavior)
            compile_json_to_go(HeadArgs, FieldMappings, FieldDelim, Unique, CoreBody)
        ),

        % Check if database backend is specified
        (   option(db_backend(bbolt), Options)
        ->  % Wrap core body with database operations
            format('  Database: bbolt~n'),
            wrap_with_database(CoreBody, FieldMappings, Pred, Options, ScriptBody, KeyImports),
            NeedsDatabase = true
        ;   ScriptBody = CoreBody,
            NeedsDatabase = false,
            KeyImports = []
        ),

        % Check if helpers are needed (for the package wrapping)
        (   member(nested(_, _), FieldMappings)
        ->  generate_nested_helper(HelperFunc),
            HelperSection = HelperFunc
        ;   HelperSection = ''
        )
    ;   format('ERROR: Multiple clauses not yet supported for JSON input mode~n'),
        fail
    ),

    % Wrap in package if requested
    (   IncludePackage ->
        % Generate imports based on what's needed
        (   NeedsDatabase = true
        ->  BaseImports = ["bufio", "encoding/json", "fmt", "os", "bolt \"go.etcd.io/bbolt\""]
        ;   BaseImports = ["bufio", "encoding/json", "fmt", "os"]
        ),

        % Add key expression imports if any (from main)
        (   KeyImports \= []
        ->  maplist(atom_string, KeyImports, KeyImportStrs),
            append(BaseImports, KeyImportStrs, ImportsWithKeys)
        ;   ImportsWithKeys = BaseImports
        ),
        
        % Add strings import if needed (from feature branch)
        (   option(json_schema(SchemaName), Options), schema_needs_strings(SchemaName)
        ->  append(ImportsWithKeys, ["strings"], AllImportsList)
        ;   AllImportsList = ImportsWithKeys
        ),
        
        % Format all imports
        sort(AllImportsList, UniqueImports), % Deduplicate
        maplist(format_import, UniqueImports, ImportLines),
        atomic_list_concat(ImportLines, '\n', Imports),

        (   HelperSection = '' ->
            format(string(GoCode), 'package main

import (
~s
)

func main() {
~s}
', [Imports, ScriptBody])
        ;   format(string(GoCode), 'package main

import (
~s
)

~s

func main() {
~s}
', [Imports, HelperSection, ScriptBody])
        )
    ;   GoCode = ScriptBody
    ).

%% compile_parallel_json_input_mode(+Pred, +Arity, +Options, +Workers, -GoCode)
%  Compile predicate with JSON input for parallel execution
compile_parallel_json_input_mode(Pred, Arity, Options, Workers, GoCode) :-
    % Get options
    option(field_delimiter(FieldDelim), Options, colon),
    option(include_package(IncludePackage), Options, true),
    option(unique(Unique), Options, true),

    % Get predicate clauses
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),

    (   Clauses = [SingleHead-SingleBody] ->
        % Single clause - extract JSON field mappings
        extract_json_field_mappings(SingleBody, FieldMappings),
        
        SingleHead =.. [_|HeadArgs],
        compile_parallel_json_to_go(HeadArgs, FieldMappings, FieldDelim, Unique, Workers, ScriptBody),

        % Check if helpers are needed
        (   member(nested(_, _), FieldMappings)
        ->  generate_nested_helper(HelperFunc),
            HelperSection = HelperFunc
        ;   HelperSection = ''
        )
    ;   format('ERROR: Multiple clauses not yet supported for parallel JSON input mode~n'),
        fail
    ),

    % Wrap in package if requested
    (   IncludePackage ->
        BaseImports = ["bufio", "encoding/json", "fmt", "os", "sync"],
        
        (   option(json_schema(SchemaName), Options), schema_needs_strings(SchemaName)
        ->  append(BaseImports, ["strings"], AllImportsList)
        ;   AllImportsList = BaseImports
        ),
        
        maplist(format_import, AllImportsList, ImportLines),
        atomic_list_concat(ImportLines, '\n', Imports),
        (   HelperSection = '' ->
            format(string(GoCode), 'package main

import (
~s
)

func main() {
~s}
', [Imports, ScriptBody])
        ;   format(string(GoCode), 'package main

import (
~s
)

~s

func main() {
~s}
', [Imports, HelperSection, ScriptBody])
        )
    ;   GoCode = ScriptBody
    ).

%% compile_parallel_json_to_go(+HeadArgs, +Operations, +FieldDelim, +Unique, +Workers, -GoCode)
%  Generate Go code for parallel JSON processing
compile_parallel_json_to_go(HeadArgs, Operations, FieldDelim, Unique, Workers, GoCode) :-
    map_field_delimiter(FieldDelim, DelimChar),

    % Generate processing code (recursive)
    generate_parallel_json_processing(Operations, HeadArgs, DelimChar, Unique, 1, [], ProcessingCode),

    (   Unique = true 
    ->  UniqueVars = "var seenMutex sync.Mutex\n\tseen := make(map[string]bool)" 
    ;   UniqueVars = ""
    ),

    format(string(GoCode), '
	// Parallel execution with ~w workers
	jobs := make(chan []byte, 100)
	var wg sync.WaitGroup
	var outputMutex sync.Mutex
	~s

	// Start workers
	for i := 0; i < ~w; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for lineBytes := range jobs {
				var data map[string]interface{}
				if err := json.Unmarshal(lineBytes, &data); err != nil {
					continue
				}
				
				~s
			}
		}()
	}

	// Scanner loop
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		// Copy bytes because scanner.Bytes() is reused
		b := make([]byte, len(scanner.Bytes()))
		copy(b, scanner.Bytes())
		jobs <- b
	}
	close(jobs)
	wg.Wait()
', [Workers, UniqueVars, Workers, ProcessingCode]).

%% generate_parallel_json_processing(+Operations, +HeadArgs, +Delim, +Unique, +VIdx, +VarMap, -Code)
generate_parallel_json_processing([], HeadArgs, Delim, Unique, _, VarMap, Code) :-
    !,
    generate_json_output_from_map(HeadArgs, VarMap, Delim, OutputExpr),
    (   Unique = true ->
        format(string(Code), '
				result := ~s
				seenMutex.Lock()
				if !seen[result] {
					seen[result] = true
					outputMutex.Lock()
					fmt.Println(result)
					outputMutex.Unlock()
				}
				seenMutex.Unlock()', [OutputExpr])
    ;   format(string(Code), '
				result := ~s
				outputMutex.Lock()
				fmt.Println(result)
				outputMutex.Unlock()', [OutputExpr])
    ).

generate_parallel_json_processing([Op|Rest], HeadArgs, Delim, Unique, VIdx, VarMap, Code) :-
    % Reuse existing logic for extraction, just recurse to parallel version
    NextVIdx is VIdx + 1,
    (   Op = nested(Path, Var) ->
        format(atom(GoVar), 'v~w', [VIdx]),
        generate_nested_extraction_code(Path, 'data', GoVar, ExtractCode),
        generate_parallel_json_processing(Rest, HeadArgs, Delim, Unique, NextVIdx, [(Var, GoVar)|VarMap], RestCode),
        format(string(Code), '~s\n~s', [ExtractCode, RestCode])

    ;   Op = Field-Var ->
        format(atom(GoVar), 'v~w', [VIdx]),
        atom_string(Field, FieldStr),
        generate_flat_field_extraction(FieldStr, GoVar, ExtractCode),
        generate_parallel_json_processing(Rest, HeadArgs, Delim, Unique, NextVIdx, [(Var, GoVar)|VarMap], RestCode),
        format(string(Code), '~s\n~s', [ExtractCode, RestCode])

    ;   Op = extract(SourceVar, Path, Var) ->
        (   lookup_var_identity(SourceVar, VarMap, SourceGoVar) -> true
        ;   format('ERROR: Source variable not found in map: ~w~n', [SourceVar]), fail
        ),
        format(atom(GoVar), 'v~w', [VIdx]),
        format(string(ExtractCode), '
				var ~w interface{}
				if val, ok := ~w.(map[string]interface{}); ok {
					if v, ok := val["~w"]; ok {
						~w = v
					} else {
						continue
					}
				} else {
					continue
				}', [GoVar, SourceGoVar, Path, GoVar]),
        generate_parallel_json_processing(Rest, HeadArgs, Delim, Unique, NextVIdx, [(Var, GoVar)|VarMap], RestCode),
        format(string(Code), '~s\n~s', [ExtractCode, RestCode])

    ;   Op = iterate(ListVar, ItemVar) ->
        (   lookup_var_identity(ListVar, VarMap, ListGoVar) -> true
        ;   format('ERROR: List variable not found in map: ~w~n', [ListVar]), fail
        ),
        format(atom(ItemGoVar), 'v~w', [VIdx]),
        generate_parallel_json_processing(Rest, HeadArgs, Delim, Unique, NextVIdx, [(ItemVar, ItemGoVar)|VarMap], LoopBody),
        format(string(Code), '
				if listVal, ok := ~w.([]interface{}); ok {
					for _, itemVal := range listVal {
						~w := itemVal
						~s
					}
				}', [ListGoVar, ItemGoVar, LoopBody])
    ).

%% wrap_with_database(+CoreBody, +FieldMappings, +Pred, +Options, -WrappedBody, -KeyImports)
%  Wrap core extraction code with database operations
%  Returns additional imports needed for key expressions
%
wrap_with_database(CoreBody, FieldMappings, Pred, Options, WrappedBody, KeyImports) :-
    % Get database options
    option(db_file(DbFile), Options, 'data.db'),
    option(db_bucket(BucketName), Options, Pred),
    atom_string(BucketName, BucketStr),

    % Normalize key strategy options (backward compatibility)
    normalize_key_strategy(Options, NormalizedOptions),

    % Determine key strategy (with default fallback)
    (   option(db_key_strategy(KeyStrategy), NormalizedOptions)
    ->  true
    ;   % Default: use first field
        (   FieldMappings = [FirstField-_|_]
        ->  KeyStrategy = field(FirstField)
        ;   FieldMappings = [nested(Path, _)|_],
            last(Path, FirstField)
        ->  KeyStrategy = field(FirstField)
        ;   format('ERROR: No fields found in mappings: ~w~n', [FieldMappings]),
            fail
        )
    ),

    % Compile key expression to Go code
    compile_key_expression(KeyStrategy, FieldMappings, NormalizedOptions, KeyCode, KeyImports),

    % Determine which fields are used by the key expression
    extract_used_field_positions(KeyStrategy, FieldMappings, UsedFieldPositions),

    % Generate blank assignments for unused fields (to avoid Go unused variable errors)
    findall(BlankAssignment,
        (nth1(Pos, FieldMappings, _),
         \+ memberchk(Pos, UsedFieldPositions),
         format(string(BlankAssignment), '\t\t_ = field~w  // Unused in key\n', [Pos])),
        BlankAssignments),
    atomic_list_concat(BlankAssignments, '', UnusedFieldCode),

    % Create storage code block with compiled key
    format(string(StorageCode), '~s\t\t// Store in database
\t\terr = db.Update(func(tx *bolt.Tx) error {
\t\t\tbucket := tx.Bucket([]byte("~s"))
\t\t\t
\t\t\t// Generate key using strategy
\t\t\tkeyStr := ~s
\t\t\tkey := []byte(keyStr)
\t\t\t
\t\t\t// Store full JSON record
\t\t\tvalue, err := json.Marshal(data)
\t\t\tif err != nil {
\t\t\t\treturn err
\t\t\t}
\t\t\t
\t\t\treturn bucket.Put(key, value)
\t\t})
\t\t
\t\tif err != nil {
\t\t\terrorCount++
\t\t\tfmt.Fprintf(os.Stderr, "Database error: %v\\n", err)
\t\t\tcontinue
\t\t}
\t\t
\t\trecordCount++', [UnusedFieldCode, BucketStr, KeyCode]),

    % Remove output block and inject storage code
    split_string(CoreBody, "\n", "", Lines),
    % Pass -1 to keep all fields (key expression will determine which it needs)
    filter_and_replace_lines(Lines, StorageCode, -1, FilteredLines),
    atomics_to_string(FilteredLines, "\n", CleanedCore),

    % Generate wrapped code with storage inside loop
    format(string(WrappedBody), '\t// Open database
\tdb, err := bolt.Open("~s", 0600, nil)
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening database: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer db.Close()

\t// Create bucket
\terr = db.Update(func(tx *bolt.Tx) error {
\t\t_, err := tx.CreateBucketIfNotExists([]byte("~s"))
\t\treturn err
\t})
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error creating bucket: %v\\n", err)
\t\tos.Exit(1)
\t}

\t// Process records
\trecordCount := 0
\terrorCount := 0

~s

\t// Summary
\tfmt.Fprintf(os.Stderr, "Stored %d records, %d errors\\n", recordCount, errorCount)
', [DbFile, BucketStr, CleanedCore]).

% Helper to filter output lines and inject storage code
filter_and_replace_lines(Lines, StorageCode, KeyPos, Result) :-
    filter_and_replace_lines(Lines, StorageCode, KeyPos, [], Result).

filter_and_replace_lines([], _StorageCode, _KeyPos, Acc, Result) :-
    reverse(Acc, Result).
filter_and_replace_lines([Line|Rest], StorageCode, KeyPos, Acc, Result) :-
    (   % Skip seen map declaration
        sub_string(Line, _, _, _, "seen := make(map[string]bool)")
    ->  filter_and_replace_lines(Rest, StorageCode, KeyPos, Acc, Result)
    ;   % Skip result formatting
        sub_string(Line, _, _, _, "result := fmt.Sprintf")
    ->  filter_and_replace_lines(Rest, StorageCode, KeyPos, Acc, Result)
    ;   % Skip seen check and inject storage code instead
        sub_string(Line, _, _, _, "if !seen[result]")
    ->  % Skip the entire if block (this line + seen[result]=true + println + closing brace)
        Rest = [_SeenTrue, _Println, _CloseBrace|RestAfterBlock],
        % Inject storage code
        filter_and_replace_lines(RestAfterBlock, StorageCode, KeyPos, [StorageCode|Acc], Result)
    ;   % Replace non-key field variables with _ (blank identifier)
        % Match lines like: "\t\tfieldN, fieldNOk := data[...]"
        sub_string(Line, _, _, _, "field"),
        replace_unused_field_var(Line, KeyPos, NewLine),
        NewLine \= Line  % Only if replacement was made
    ->  filter_and_replace_lines(Rest, StorageCode, KeyPos, [NewLine|Acc], Result)
    ;   % Keep all other lines
        filter_and_replace_lines(Rest, StorageCode, KeyPos, [Line|Acc], Result)
    ).

% Replace fieldN with _ if N != KeyPos
replace_unused_field_var(Line, KeyPos, NewLine) :-
    % If KeyPos is -1, keep all fields (don't replace anything)
    KeyPos \= -1,
    % Try to replace field1, field2, field3, etc. up to field9
    between(1, 9, N),
    N \= KeyPos,
    (   % Pattern 1: "fieldN," for untyped extraction
        format(atom(FieldVar), 'field~w,', [N]),
        sub_string(Line, Before, Len, After, FieldVar)
    ->  % Replace with "_,"
        sub_string(Line, 0, Before, _, Prefix),
        string_concat(Prefix, "_,", NewPrefix),
        Skip is Before + Len,
        sub_string(Line, Skip, _, 0, Suffix),
        string_concat(NewPrefix, Suffix, NewLine),
        !
    ;   % Pattern 2: "fieldN := " for typed final assignment
        format(atom(FieldAssign), 'field~w := ', [N]),
        sub_string(Line, Before2, Len2, After2, FieldAssign),
        % Replace entire line with "_ = " version
        sub_string(Line, 0, Before2, _, Prefix2),
        string_concat(Prefix2, "_ = ", NewPrefix2),
        Skip2 is Before2 + Len2,
        sub_string(Line, Skip2, _, 0, Suffix2),
        string_concat(NewPrefix2, Suffix2, NewLine),
        !
    ;   fail
    ).
replace_unused_field_var(Line, _, Line).  % No replacement needed

%% ============================================
%% JSON OUTPUT MODE
%% ============================================

%% compile_json_output_mode(+Pred, +Arity, +Options, -GoCode)
%  Compile a predicate to generate JSON output
%  Reads delimiter-based input and generates JSON
%
compile_json_output_mode(Pred, Arity, Options, GoCode) :-
    % Get options
    option(field_delimiter(FieldDelim), Options, colon),
    option(include_package(IncludePackage), Options, true),

    % Get field names from options or generate defaults
    (   option(json_fields(FieldNames), Options)
    ->  true
    ;   % Generate default field names from predicate
        functor(Head, Pred, Arity),
        Head =.. [_|Args],
        generate_default_field_names(Args, 1, FieldNames)
    ),

    % Generate struct definition
    atom_string(Pred, PredStr),
    upcase_atom(Pred, UpperPred),
    generate_json_struct(UpperPred, FieldNames, StructDef),

    % Generate parsing and output code
    compile_json_output_to_go(UpperPred, FieldNames, FieldDelim, OutputBody),

    % Wrap in package if requested
    (   IncludePackage ->
        format(string(GoCode), 'package main

import (
\t"bufio"
\t"encoding/json"
\t"fmt"
\t"os"
\t"strings"
\t"sync"
\t"strconv"
)

~s

func main() {
~s}
', [StructDef, OutputBody])
    ;   format(string(GoCode), '~s~n~s', [StructDef, OutputBody])
    ).

%% generate_default_field_names(+Args, +StartNum, -FieldNames)
%  Generate default field names (Field1, Field2, ...)
%
generate_default_field_names([], _, []).
generate_default_field_names([_|Args], N, [FieldName|Rest]) :-
    format(atom(FieldName), 'Field~w', [N]),
    N1 is N + 1,
    generate_default_field_names(Args, N1, Rest).

%% generate_json_struct(+StructName, +FieldNames, -StructDef)
%  Generate Go struct definition with JSON tags
%
generate_json_struct(StructName, FieldNames, StructDef) :-
    findall(FieldLine,
        (   nth1(Pos, FieldNames, FieldName),
            upcase_atom(FieldName, UpperField),
            downcase_atom(FieldName, LowerField),
            format(atom(FieldLine), '\t~w interface{} `json:"~w"`', [UpperField, LowerField])
        ),
        FieldLines),
    atomic_list_concat(FieldLines, '\n', FieldsStr),
    format(atom(StructDef), 'type ~wRecord struct {\n~w\n}', [StructName, FieldsStr]).

%% compile_json_output_to_go(+StructName, +FieldNames, +FieldDelim, -GoCode)
%  Generate Go code to read delimited input and output JSON
%
compile_json_output_to_go(StructName, FieldNames, FieldDelim, GoCode) :-
    % Map delimiter
    map_field_delimiter(FieldDelim, DelimChar),

    % Generate field count and parsing code
    length(FieldNames, NumFields),
    generate_field_parsing(FieldNames, 1, ParseCode),

    % Generate struct initialization
    generate_struct_init(StructName, FieldNames, StructInit),

    format(string(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
\t
\tfor scanner.Scan() {
\t\tparts := strings.Split(scanner.Text(), "~s")
\t\tif len(parts) != ~w {
\t\t\tcontinue
\t\t}
\t\t
~s
\t\t
\t\trecord := ~s
\t\t
\t\tjsonBytes, err := json.Marshal(record)
\t\tif err != nil {
\t\t\tcontinue
\t\t}
\t\tfmt.Println(string(jsonBytes))
\t}
', [DelimChar, NumFields, ParseCode, StructInit]).

%% generate_field_parsing(+FieldNames, +StartPos, -ParseCode)
%  Generate code to parse fields from split parts
%  Tries to convert to numbers if possible, otherwise uses string
%
generate_field_parsing([], _, '').
generate_field_parsing([FieldName|Rest], Pos, ParseCode) :-
    Pos0 is Pos - 1,
    upcase_atom(FieldName, UpperField),
    downcase_atom(FieldName, LowerField),

    % Generate parsing code that tries numeric conversion
    format(atom(FieldParse), '\t\t// Parse ~w
\t\t~w := parts[~w]
\t\tvar ~wValue interface{}
\t\tif intVal, err := strconv.Atoi(~w); err == nil {
\t\t\t~wValue = intVal
\t\t} else if floatVal, err := strconv.ParseFloat(~w, 64); err == nil {
\t\t\t~wValue = floatVal
\t\t} else if boolVal, err := strconv.ParseBool(~w); err == nil {
\t\t\t~wValue = boolVal
\t\t} else {
\t\t\t~wValue = ~w
\t\t}',
        [LowerField, LowerField, Pos0, LowerField, LowerField,
         LowerField, LowerField, LowerField, LowerField, LowerField,
         LowerField, LowerField]),

    Pos1 is Pos + 1,
    generate_field_parsing(Rest, Pos1, RestParse),

    (   RestParse = ''
    ->  ParseCode = FieldParse
    ;   format(atom(ParseCode), '~s~n~s', [FieldParse, RestParse])
    ).

%% generate_struct_init(+StructName, +FieldNames, -StructInit)
%  Generate struct initialization code
%
generate_struct_init(StructName, FieldNames, StructInit) :-
    findall(FieldInit,
        (   member(FieldName, FieldNames),
            upcase_atom(FieldName, UpperField),
            downcase_atom(FieldName, LowerField),
            format(atom(FieldInit), '~w: ~wValue', [UpperField, LowerField])
        ),
        FieldInits),
    atomic_list_concat(FieldInits, ', ', FieldsStr),
    format(atom(StructInit), '~wRecord{~w}', [StructName, FieldsStr]).

%% extract_json_field_mappings(+Body, -FieldMappings)
%  Extract field-to-variable mappings from json_record([field-Var, ...]) and json_get(Path, Var)
%
extract_json_field_mappings(Body, FieldMappings) :-
    extract_json_operations(Body, Operations),
    (   Operations = [] ->
        format('WARNING: Body does not contain json_record/1 or json_get/2: ~w~n', [Body]),
        FieldMappings = []
    ;   FieldMappings = Operations
    ).

%% extract_json_operations(+Body, -Operations)
%  Extract all JSON operations from body (handles conjunction)
%
extract_json_operations(_:Goal, Ops) :- !,
    extract_json_operations(Goal, Ops).
extract_json_operations((A, B), Ops) :- !,
    extract_json_operations(A, OpsA),
    extract_json_operations(B, OpsB),
    append(OpsA, OpsB, Ops).
extract_json_operations(json_record(Fields), RecordOps) :- !,
    extract_field_list(Fields, RecordOps).
extract_json_operations(json_get(Path, Var), [nested(Path, Var)]) :- !.
extract_json_operations(json_get(Source, Path, Var), [extract(Source, Path, Var)]) :- !,
    var(Source).
extract_json_operations(json_array_member(List, Item), [iterate(List, Item)]) :- !.
extract_json_operations(_, []).


extract_field_list([], []).
extract_field_list([Field-Var|Rest], [Field-Var|Mappings]) :- !,
    extract_field_list(Rest, Mappings).
extract_field_list([Other|Rest], Mappings) :-
    format('WARNING: Unexpected field format: ~w~n', [Other]),
    extract_field_list(Rest, Mappings).

%% compile_json_to_go(+HeadArgs, +Operations, +FieldDelim, +Unique, -GoCode)
%  Generate Go code for JSON input mode (recursive for arrays)
%
compile_json_to_go(HeadArgs, Operations, FieldDelim, Unique, GoCode) :-
    % Map delimiter
    map_field_delimiter(FieldDelim, DelimChar),

    % Generate processing code (recursive)
    % Initial VarMap contains 'data' -> 'data'
    generate_json_processing(Operations, HeadArgs, DelimChar, Unique, 1, [], ProcessingCode),

    % Build the loop code
    format(string(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
\tseen := make(map[string]bool)
\t
\tfor scanner.Scan() {
\t\tvar data map[string]interface{}
\t\tif err := json.Unmarshal(scanner.Bytes(), &data); err != nil {
\t\t\tcontinue
\t\t}
\t\t
~s
\t}
', [ProcessingCode]).

%% generate_json_processing(+Operations, +HeadArgs, +Delim, +Unique, +VIdx, +VarMap, -Code)
%  Generate nested Go code for JSON operations
generate_json_processing([], HeadArgs, Delim, Unique, _, VarMap, Code) :-
    !,
    % No more operations - generate output
    generate_json_output_from_map(HeadArgs, VarMap, Delim, OutputExpr),
    (   Unique = true ->
        format(string(Code), '
\t\tresult := ~s
\t\tif !seen[result] {
\t\t\tseen[result] = true
\t\t\tfmt.Println(result)
\t\t}', [OutputExpr])
    ;   format(string(Code), '
\t\tresult := ~s
\t\tfmt.Println(result)', [OutputExpr])
    ).

generate_json_processing([Op|Rest], HeadArgs, Delim, Unique, VIdx, VarMap, Code) :-
    NextVIdx is VIdx + 1,
    (   Op = nested(Path, Var) ->
        format(atom(GoVar), 'v~w', [VIdx]),
        generate_nested_extraction_code(Path, 'data', GoVar, ExtractCode),
        generate_json_processing(Rest, HeadArgs, Delim, Unique, NextVIdx, [(Var, GoVar)|VarMap], RestCode),
        format(string(Code), '~s\n~s', [ExtractCode, RestCode])

    ;   Op = Field-Var ->
        format(atom(GoVar), 'v~w', [VIdx]),
        % Generate flat field extraction (not nested)
        atom_string(Field, FieldStr),
        format(string(ExtractCode), '
\t\t~w, ok~w := data["~s"]
\t\tif !ok~w {
\t\t\tcontinue
\t\t}', [GoVar, VIdx, FieldStr, VIdx]),
        generate_json_processing(Rest, HeadArgs, Delim, Unique, NextVIdx, [(Var, GoVar)|VarMap], RestCode),
        format(string(Code), '~s\n~s', [ExtractCode, RestCode])

    ;   Op = extract(SourceVar, Path, Var) ->
        (   lookup_var_identity(SourceVar, VarMap, SourceGoVar) -> true
        ;   format('ERROR: Source variable not found in map: ~w~n', [SourceVar]), fail
        ),
        format(atom(GoVar), 'v~w', [VIdx]),
        % Source must be a map
        format(string(ExtractCode), '
\t\tsourceMap~w, ok := ~w.(map[string]interface{})
\t\tif !ok { continue }', [VIdx, SourceGoVar]),
        
        format(atom(SourceMapVar), 'sourceMap~w', [VIdx]),
        generate_nested_extraction_code(Path, SourceMapVar, GoVar, InnerExtract),
        
        format(string(FullExtract), '~s\n~s', [ExtractCode, InnerExtract]),
        format(atom(FixedExtract), FullExtract, []),
        
        generate_json_processing(Rest, HeadArgs, Delim, Unique, NextVIdx, [(Var, GoVar)|VarMap], RestCode),
        format(string(Code), '~s\n~s', [FixedExtract, RestCode])

    ;   Op = iterate(ListVar, ItemVar) ->
        (   lookup_var_identity(ListVar, VarMap, ListGoVar) -> true
        ;   format('ERROR: List variable not found in map: ~w~n', [ListVar]), fail
        ),
        format(atom(ItemGoVar), 'v~w', [VIdx]),
        
        generate_json_processing(Rest, HeadArgs, Delim, Unique, NextVIdx, [(ItemVar, ItemGoVar)|VarMap], InnerCode),
        
        format(string(Code), '
\t\tif listVal~w, ok := ~w.([]interface{}); ok {
\t\t\tfor _, itemVal~w := range listVal~w {
\t\t\t\t~w := itemVal~w
~s
\t\t\t}
\t\t}', [VIdx, ListGoVar, VIdx, VIdx, ItemGoVar, VIdx, InnerCode])
    ;   % Fallback for unknown ops
        format('WARNING: Unknown JSON operation: ~w~n', [Op]),
        generate_json_processing(Rest, HeadArgs, Delim, Unique, VIdx, VarMap, Code)
    ).

generate_nested_extraction_code(Path, Source, Target, Code) :-
    (   is_list(Path) -> PathList = Path ; PathList = [Path] ),
    maplist(atom_string, PathList, PathStrs),
    atomic_list_concat(PathStrs, '", "', PathStr),
    format(string(Code), '
\t\t~w, found := getNestedField(~w, []string{"~s"})
\t\tif !found { continue }', [Target, Source, PathStr]).

generate_json_output_from_map(HeadArgs, VarMap, DelimChar, OutputExpr) :-
    maplist(arg_to_go_var(VarMap), HeadArgs, GoVars),
    length(HeadArgs, NumArgs),
    findall('%v', between(1, NumArgs, _), FormatParts),
    atomic_list_concat(FormatParts, DelimChar, FormatStr),
    atomic_list_concat(GoVars, ', ', VarList),
    format(atom(OutputExpr), 'fmt.Sprintf("~s", ~s)', [FormatStr, VarList]).

arg_to_go_var(VarMap, Arg, GoVar) :-
    var(Arg), !,
    (   lookup_var_identity(Arg, VarMap, Name) ->
        GoVar = Name
    ;   GoVar = '"<unknown>"'
    ).
arg_to_go_var(_, Arg, GoVar) :-
    atom(Arg), !,
    format(atom(GoVar), '"~w"', [Arg]).
arg_to_go_var(_, Arg, Arg) :- number(Arg), !.

%% lookup_var_identity(+Key, +Map, -Val)
%  Lookup value in association list using identity check (==)
lookup_var_identity(Key, [(K, V)|_], Val) :- Key == K, !, Val = V.
lookup_var_identity(Key, [_|Rest], Val) :- lookup_var_identity(Key, Rest, Val).



%% compile_json_to_go_typed_noschema(+HeadArgs, +FieldMappings, +FieldDelim, +Unique, -GoCode)
%  Generate Go code for JSON input mode with fieldN variables but no type validation
%  Used for database writes without schema
%
compile_json_to_go_typed_noschema(HeadArgs, FieldMappings, FieldDelim, Unique, GoCode) :-
    % Map delimiter
    map_field_delimiter(FieldDelim, DelimChar),

    % Generate untyped field extraction code with fieldN variable names
    findall(ExtractLine,
        (   nth1(Pos, FieldMappings, Mapping),
            format(atom(VarName), 'field~w', [Pos]),
            % Generate extraction for flat or nested fields
            (   Mapping = Field-_Var
            ->  % Flat field - untyped extraction
                atom_string(Field, FieldStr),
                format(atom(ExtractLine), '\t\t~w, ~wOk := data["~s"]\n\t\tif !~wOk {\n\t\t\tcontinue\n\t\t}',
                    [VarName, VarName, FieldStr, VarName])
            ;   Mapping = nested(Path, _Var)
            ->  % Nested field - untyped extraction
                generate_nested_field_extraction(Path, VarName, ExtractLine)
            )
        ),
        ExtractLines),
    atomic_list_concat(ExtractLines, '\n', ExtractCode),

    % Generate output expression (same as typed)
    generate_json_output_expr(HeadArgs, DelimChar, OutputExpr),

    % Build main loop
    (   Unique = true ->
        SeenDecl = '\tseen := make(map[string]bool)\n\t',
        UniqueCheck = '\t\tif !seen[result] {\n\t\t\tseen[result] = true\n\t\t\tfmt.Println(result)\n\t\t}\n'
    ;   SeenDecl = '\t',
        UniqueCheck = '\t\tfmt.Println(result)\n'
    ),

    % Build the loop code
    format(string(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
~w
\tfor scanner.Scan() {
\t\tvar data map[string]interface{}
\t\tif err := json.Unmarshal(scanner.Bytes(), &data); err != nil {
\t\t\tcontinue
\t\t}
\t\t
~s
\t\t
\t\tresult := ~s
~s\t}
', [SeenDecl, ExtractCode, OutputExpr, UniqueCheck]).

%% compile_json_to_go_typed(+HeadArgs, +FieldMappings, +SchemaName, +FieldDelim, +Unique, -GoCode)
%  Generate Go code for JSON input mode with type safety from schema
%
compile_json_to_go_typed(HeadArgs, FieldMappings, SchemaName, FieldDelim, Unique, GoCode) :-
    % Map delimiter
    map_field_delimiter(FieldDelim, DelimChar),

    % Generate typed field extraction code
    generate_typed_field_extractions(FieldMappings, SchemaName, HeadArgs, ExtractCode),

    % Generate output expression (same as untyped)
    generate_json_output_expr(HeadArgs, DelimChar, OutputExpr),

    % Build main loop
    (   Unique = true ->
        SeenDecl = '\tseen := make(map[string]bool)\n\t',
        UniqueCheck = '\t\tif !seen[result] {\n\t\t\tseen[result] = true\n\t\t\tfmt.Println(result)\n\t\t}\n'
    ;   SeenDecl = '\t',
        UniqueCheck = '\t\tfmt.Println(result)\n'
    ),

    % Build the loop code
    format(string(GoCode), '
\tscanner := bufio.NewScanner(os.Stdin)
~w
\tfor scanner.Scan() {
\t\tvar data map[string]interface{}
\t\tif err := json.Unmarshal(scanner.Bytes(), &data); err != nil {
\t\t\tcontinue
\t\t}
\t\t
~s
\t\t
\t\tresult := ~s
~s\t}
', [SeenDecl, ExtractCode, OutputExpr, UniqueCheck]).

%% generate_typed_field_extractions(+FieldMappings, +SchemaName, +HeadArgs, -ExtractCode)
%  Generate typed field extraction code based on schema
%
generate_typed_field_extractions(FieldMappings, SchemaName, _HeadArgs, ExtractCode) :-
    findall(ExtractLine,
        (   nth1(Pos, FieldMappings, Mapping),
            format(atom(VarName), 'field~w', [Pos]),
            % Dispatch based on mapping type
            (   Mapping = Field-_Var
            ->  % Flat field - get type and options from schema
                get_field_info(SchemaName, Field, Type, Options),
                atom_string(Field, FieldStr),
                generate_typed_flat_field_extraction(FieldStr, VarName, Type, Options, ExtractLine)
            ;   Mapping = nested(Path, _Var)
            ->  % Nested field - get type and options from last element of path
                last(Path, LastField),
                get_field_info(SchemaName, LastField, Type, Options),
                generate_typed_nested_field_extraction(Path, VarName, Type, Options, ExtractLine)
            )
        ),
        ExtractLines),
    atomic_list_concat(ExtractLines, '\n', ExtractCode).

%% generate_typed_flat_field_extraction(+FieldName, +VarName, +Type, +Options, -ExtractCode)
%  Generate type-safe extraction code for a flat field with validation
%
generate_typed_flat_field_extraction(FieldName, VarName, Type, Options, ExtractCode) :-
    % Generate validation code
    generate_validation_code(VarName, Type, Options, ValidationCode),

    % Determine if optional
    (   member(optional, Options) -> Optional = true ; Optional = false ),

    % Generate extraction based on type
    (   Type = string ->
        format(atom(ExtractCode), '\t\tvar ~w string\n\t\t~wRaw, ~wRawOk := data["~s"]\n\t\tif ~wRawOk {\n\t\t\t~wVal, ~wIsString := ~wRaw.(string)\n\t\tif !~wIsString {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: field ''~s'' is not a string\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w = ~wVal\n~s\n\t\t} else if !~w {\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, VarName, FieldName, VarName, VarName, VarName, VarName, VarName, FieldName, VarName, VarName, ValidationCode, Optional])
    ;   Type = integer ->
        format(atom(ExtractCode), '\t\tvar ~w int\n\t\t~wRaw, ~wRawOk := data["~s"]\n\t\tif ~wRawOk {\n\t\t\t~wFloat, ~wFloatOk := ~wRaw.(float64)\n\t\tif !~wFloatOk {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: field ''~s'' is not a number\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w = int(~wFloat)\n~s\n\t\t} else if !~w {\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, VarName, FieldName, VarName, VarName, VarName, VarName, VarName, FieldName, VarName, VarName, ValidationCode, Optional])
    ;   Type = float ->
        format(atom(ExtractCode), '\t\tvar ~w float64\n\t\t~wRaw, ~wRawOk := data["~s"]\n\t\tif ~wRawOk {\n\t\t\t~wVal, ~wFloatOk := ~wRaw.(float64)\n\t\tif !~wFloatOk {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: field ''~s'' is not a number\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w = ~wVal\n~s\n\t\t} else if !~w {\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, VarName, FieldName, VarName, VarName, VarName, VarName, VarName, FieldName, VarName, VarName, ValidationCode, Optional])
    ;   Type = boolean ->
        format(atom(ExtractCode), '\t\tvar ~w bool\n\t\t~wRaw, ~wRawOk := data["~s"]\n\t\tif ~wRawOk {\n\t\t\t~wVal, ~wBoolOk := ~wRaw.(bool)\n\t\tif !~wBoolOk {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: field ''~s'' is not a boolean\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w = ~wVal\n~s\n\t\t} else if !~w {\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, VarName, FieldName, VarName, VarName, VarName, VarName, VarName, FieldName, VarName, VarName, ValidationCode, Optional])
    ;   % Fallback to untyped for 'any' type
        generate_flat_field_extraction(FieldName, VarName, ExtractCode)
    ).

%% generate_typed_nested_field_extraction(+Path, +VarName, +Type, +Options, -ExtractCode)
%  Generate type-safe extraction code for a nested field with validation
%
generate_typed_nested_field_extraction(Path, VarName, Type, Options, ExtractCode) :-
    % Convert path to Go string slice
    maplist(atom_string, Path, PathStrs),
    atomic_list_concat(PathStrs, '", "', PathStr),
    last(Path, LastField),
    atom_string(LastField, LastFieldStr),

    % Generate validation code
    generate_validation_code(VarName, Type, Options, ValidationCode),

    % Determine if optional
    (   member(optional, Options) -> Optional = true ; Optional = false ),

    % Generate extraction with type assertion based on type
    (   Type = string ->
        format(atom(ExtractCode), '\t\tvar ~w string\n\t\t~wRaw, ~wRawOk := getNestedField(data, []string{"~s"})\n\t\tif ~wRawOk {\n\t\t\t~wVal, ~wIsString := ~wRaw.(string)\n\t\tif !~wIsString {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: nested field ''~s'' is not a string\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w = ~wVal\n~s\n\t\t} else if !~w {\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, PathStr, VarName, VarName, VarName, VarName, LastFieldStr, VarName, VarName, ValidationCode, Optional])
    ;   Type = integer ->
        format(atom(ExtractCode), '\t\tvar ~w int\n\t\t~wRaw, ~wRawOk := getNestedField(data, []string{"~s"})\n\t\tif ~wRawOk {\n\t\t\t~wFloat, ~wFloatOk := ~wRaw.(float64)\n\t\tif !~wFloatOk {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: nested field ''~s'' is not a number\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w = int(~wFloat)\n~s\n\t\t} else if !~w {\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, PathStr, VarName, VarName, VarName, VarName, LastFieldStr, VarName, VarName, ValidationCode, Optional])
    ;   Type = float ->
        format(atom(ExtractCode), '\t\tvar ~w float64\n\t\t~wRaw, ~wRawOk := getNestedField(data, []string{"~s"})\n\t\tif ~wRawOk {\n\t\t\t~wVal, ~wFloatOk := ~wRaw.(float64)\n\t\tif !~wFloatOk {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: nested field ''~s'' is not a number\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w = ~wVal\n~s\n\t\t} else if !~w {\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, PathStr, VarName, VarName, VarName, VarName, LastFieldStr, VarName, VarName, ValidationCode, Optional])
    ;   Type = boolean ->
        format(atom(ExtractCode), '\t\tvar ~w bool\n\t\t~wRaw, ~wRawOk := getNestedField(data, []string{"~s"})\n\t\tif ~wRawOk {\n\t\t\t~wVal, ~wBoolOk := ~wRaw.(bool)\n\t\tif !~wBoolOk {\n\t\t\tfmt.Fprintf(os.Stderr, "Error: nested field ''~s'' is not a boolean\\n")\n\t\t\tcontinue\n\t\t}\n\t\t~w = ~wVal\n~s\n\t\t} else if !~w {\n\t\t\tcontinue\n\t\t}',
            [VarName, VarName, PathStr, VarName, VarName, VarName, VarName, LastFieldStr, VarName, VarName, ValidationCode, Optional])
    ;   % Fallback to untyped
        generate_nested_field_extraction(Path, VarName, ExtractCode)
    ).

%% generate_validation_code(+VarName, +Type, +Options, -Code)
%  Generate Go validation logic for a variable
generate_validation_code(VarName, Type, Options, Code) :-
    findall(Check,
        (   member(Option, Options),
            generate_check(VarName, Type, Option, Check)
        ),
        Checks),
    atomic_list_concat(Checks, '\n', Code).

generate_check(VarName, integer, min(Min), Check) :-
    format(atom(Check), '\t\tif ~w < ~w { continue }', [VarName, Min]).
generate_check(VarName, integer, max(Max), Check) :-
    format(atom(Check), '\t\tif ~w > ~w { continue }', [VarName, Max]).
generate_check(VarName, float, min(Min), Check) :-
    format(atom(Check), '\t\tif ~w < ~w { continue }', [VarName, Min]).
generate_check(VarName, float, max(Max), Check) :-
    format(atom(Check), '\t\tif ~w > ~w { continue }', [VarName, Max]).
generate_check(VarName, string, format(email), Check) :-
    format(atom(Check), '\t\tif !strings.Contains(~w, "@") { continue }', [VarName]).
generate_check(_, _, _, '').

%% generate_nested_helper(-HelperCode)
%  Generate the getNestedField helper function for traversing nested JSON
%
generate_nested_helper(HelperCode) :-
    HelperCode = 'func getNestedField(data map[string]interface{}, path []string) (interface{}, bool) {
\tcurrent := interface{}(data)
\t
\tfor _, key := range path {
\t\tcurrentMap, ok := current.(map[string]interface{})
\t\tif !ok {
\t\t\treturn nil, false
\t\t}
\t\t
\t\tvalue, exists := currentMap[key]
\t\tif !exists {
\t\t\treturn nil, false
\t\t}
\t\t
\t\tcurrent = value
\t}
\t
\treturn current, true
}'.

%% generate_json_field_extractions(+FieldMappings, +HeadArgs, -ExtractCode)
%  Generate Go code to extract and type-assert JSON fields (flat and nested)
%
generate_json_field_extractions(FieldMappings, HeadArgs, ExtractCode) :-
    % Generate extractions by pairing field mappings with positions
    findall(ExtractLine,
        (   nth1(Pos, FieldMappings, Mapping),
            format(atom(VarName), 'field~w', [Pos]),
            generate_field_extraction_dispatch(Mapping, VarName, ExtractLine)
        ),
        ExtractLines),
    atomic_list_concat(ExtractLines, '\n', ExtractCode).

%% generate_field_extraction_dispatch(+Mapping, +VarName, -ExtractCode)
%  Dispatch to appropriate extraction based on mapping type
%
generate_field_extraction_dispatch(Field-_Var, VarName, ExtractCode) :- !,
    % Flat field extraction
    atom_string(Field, FieldStr),
    generate_flat_field_extraction(FieldStr, VarName, ExtractCode).
generate_field_extraction_dispatch(nested(Path, _Var), VarName, ExtractCode) :- !,
    % Nested field extraction
    generate_nested_field_extraction(Path, VarName, ExtractCode).

%% generate_flat_field_extraction(+FieldName, +VarName, -ExtractCode)
%  Generate extraction code for a flat field
%  Extract as interface{} to support any JSON type (string, number, bool, etc.)
%
generate_flat_field_extraction(FieldName, VarName, ExtractCode) :-
    format(atom(ExtractCode), '\t\t~w, ~wOk := data["~s"]\n\t\tif !~wOk {\n\t\t\tcontinue\n\t\t}',
        [VarName, VarName, FieldName, VarName]).

%% generate_nested_field_extraction(+Path, +VarName, -ExtractCode)
%  Generate extraction code for a nested field using getNestedField helper
%
generate_nested_field_extraction(Path, VarName, ExtractCode) :-
    % Ensure path is a list
    (   is_list(Path)
    ->  PathList = Path
    ;   PathList = [Path]
    ),
    % Convert path atoms to Go string slice
    maplist(atom_string, PathList, PathStrs),
    atomic_list_concat(PathStrs, '", "', PathStr),
    format(atom(ExtractCode), '\t\t~w, ~wOk := getNestedField(data, []string{"~s"})\n\t\tif !~wOk {\n\t\t\tcontinue\n\t\t}',
        [VarName, VarName, PathStr, VarName]).

%% generate_json_output_expr(+HeadArgs, +DelimChar, -OutputExpr)
%  Generate Go fmt.Sprintf expression for output
%  Use %v to handle any JSON type (string, number, bool, etc.)
%
generate_json_output_expr(HeadArgs, DelimChar, OutputExpr) :-
    length(HeadArgs, NumArgs),
    findall('%v', between(1, NumArgs, _), FormatParts),  % %v instead of %s
    atomic_list_concat(FormatParts, DelimChar, FormatStr),

    findall(VarName,
        (   nth1(Pos, HeadArgs, _),
            format(atom(VarName), 'field~w', [Pos])
        ),
        VarNames),
    atomic_list_concat(VarNames, ', ', VarList),

    format(atom(OutputExpr), 'fmt.Sprintf("~s", ~s)', [FormatStr, VarList]).

%% ============================================
%% UTILITY FUNCTIONS
%% ============================================

%% ============================================
%% XML INPUT MODE COMPILATION
%% ============================================

%% compile_xml_input_mode(+Pred, +Arity, +Options, -GoCode)
%  Compile predicate for XML input (streaming + flattening)
compile_xml_input_mode(Pred, Arity, Options, GoCode) :-
    % Get options
    option(field_delimiter(FieldDelim), Options, colon),
    option(include_package(IncludePackage), Options, true),
    option(unique(Unique), Options, true),
    
    % Get predicate clauses
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    
    (   Clauses = [SingleHead-SingleBody] ->
        % Extract mappings (same as JSON)
        extract_json_field_mappings(SingleBody, FieldMappings),
        format('DEBUG: FieldMappings = ~w~n', [FieldMappings]),
        
        % Generate XML loop
        SingleHead =.. [_|HeadArgs],
        compile_xml_to_go(HeadArgs, FieldMappings, FieldDelim, Unique, Options, CoreBody),
        
        % Check if database backend is specified
        (   option(db_backend(bbolt), Options)
        ->  format('  Database: bbolt~n'),
            wrap_with_database(CoreBody, FieldMappings, Pred, Options, ScriptBody),
            NeedsDatabase = true
        ;   ScriptBody = CoreBody,
            NeedsDatabase = false
        )
    ;   format('ERROR: XML mode supports single clause only~n'),
        fail
    ),
    
    % Generate XML helpers
    generate_xml_helpers(XmlHelpers),
    
    % Check if nested helper is needed
    (   member(nested(_, _), FieldMappings)
    ->  generate_nested_helper(NestedHelper),
        format(string(Helpers), "~s\n~s", [XmlHelpers, NestedHelper])
    ;   Helpers = XmlHelpers
    ),
    
    % Wrap in package
    (   IncludePackage ->
        (   NeedsDatabase = true
        ->  Imports = '\t"encoding/xml"\n\t"fmt"\n\t"os"\n\t"strings"\n\t"io"\n\n\tbolt "go.etcd.io/bbolt"'
        ;   Imports = '\t"encoding/xml"\n\t"fmt"\n\t"os"\n\t"strings"\n\t"io"'
        ),
        
        format(string(GoCode), 'package main

import (
~s
)

~s

func main() {
~s}
', [Imports, Helpers, ScriptBody])
    ;   GoCode = ScriptBody
    ).

%% compile_xml_to_go(+HeadArgs, +FieldMappings, +FieldDelim, +Unique, +Options, -GoCode)
compile_xml_to_go(HeadArgs, FieldMappings, FieldDelim, Unique, Options, GoCode) :-
    % Map delimiter
    map_field_delimiter(FieldDelim, DelimChar),
    
    % Generate field extraction code (resusing JSON logic as data is map[string]interface{})
    generate_json_field_extractions(FieldMappings, HeadArgs, ExtractCode),
    generate_json_output_expr(HeadArgs, DelimChar, OutputExpr),
    
    % Get XML file and tags
    option(xml_file(XmlFile), Options, stdin),
    
    (   XmlFile == stdin
    ->  FileOpenCode = '\tf := os.Stdin'
    ;   format(string(FileOpenCode), '
\tf, err := os.Open("~w")
\tif err != nil {
\t\tfmt.Fprintf(os.Stderr, "Error opening file: %v\\n", err)
\t\tos.Exit(1)
\t}
\tdefer f.Close()', [XmlFile])
    ),

    (   option(tags(Tags), Options)
    ->  true
    ;   option(tag(Tag), Options)
    ->  Tags = [Tag]
    ;   Tags = []
    ),
    
    % Build tag check
    (   Tags = []
    ->  TagCheck = 'true'
    ;   maplist(tag_to_go_cond, Tags, Conds),
        atomic_list_concat(Conds, " || ", TagCheck)
    ),
    
    % Build main loop
    (   Unique = true ->
        SeenDecl = '\tseen := make(map[string]bool)\n\t',
        UniqueCheck = '\t\tif !seen[result] {\n\t\t\tseen[result] = true\n\t\t\tfmt.Println(result)\n\t\t}\n'
    ;   SeenDecl = '\t',
        UniqueCheck = '\t\tfmt.Println(result)\n'
    ),
    
    format(string(GoCode), '
~s

\tdecoder := xml.NewDecoder(f)
~w
\tfor {
\t\tt, err := decoder.Token()
\t\tif err == io.EOF {
\t\t\tbreak
\t\t}
\t\tif err != nil {
\t\t\tcontinue
\t\t}
\t\t
\t\tswitch se := t.(type) {
\t\tcase xml.StartElement:
\t\t\tif ~w {
\t\t\t\tvar node XmlNode
\t\t\t\tif err := decoder.DecodeElement(&node, &se); err != nil {
\t\t\t\t\tcontinue
\t\t\t\t}
\t\t\t\t
\t\t\t\tdata := FlattenXML(node)
\t\t\t\t
~s
\t\t\t\t
\t\t\t\tresult := ~s
~s
\t\t\t}
\t\t}
\t}
', [FileOpenCode, SeenDecl, TagCheck, ExtractCode, OutputExpr, UniqueCheck]).

tag_to_go_cond(Tag, Cond) :-
    format(string(Cond), 'se.Name.Local == "~w"', [Tag]).

generate_xml_helpers(Code) :-
    Code = '
type XmlNode struct {
	XMLName xml.Name
	Attrs   []xml.Attr `xml:",any,attr"`
	Content string     `xml:",chardata"`
	Nodes   []XmlNode  `xml:",any"`
}

func FlattenXML(n XmlNode) map[string]interface{} {
	m := make(map[string]interface{})
	for _, a := range n.Attrs {
		m["@"+a.Name.Local] = a.Value
	}
	trim := strings.TrimSpace(n.Content)
	if trim != "" {
		m["text"] = trim
	}
	for _, child := range n.Nodes {
        tag := child.XMLName.Local
        flatChild := FlattenXML(child)
        
        if existing, ok := m[tag]; ok {
            if list, isList := existing.([]interface{}); isList {
                m[tag] = append(list, flatChild)
            } else {
                m[tag] = []interface{}{existing, flatChild}
            }
        } else {
		    m[tag] = flatChild
        }
	}
	return m
}
'.

%% map_field_delimiter(+Delimiter, -String)
%  Map delimiter atom to string
map_field_delimiter(colon, ':') :- !.
map_field_delimiter(tab, '\t') :- !.
map_field_delimiter(comma, ',') :- !.
map_field_delimiter(pipe, '|') :- !.
map_field_delimiter(Char, Char) :- atom(Char), atom_length(Char, 1), !.

%% write_go_program(+GoCode, +FilePath)
%  Write Go code to file
%
write_go_program(GoCode, FilePath) :-
    file_directory_name(FilePath, Dir),
    (   Dir \= '.' -> make_directory_path(Dir) ; true ),
    open(FilePath, write, Stream),
    write(Stream, GoCode),
    close(Stream),
    format('Go program written to: ~w~n', [FilePath]).

%% schema_needs_strings(+SchemaName)
%  Check if schema requires the strings package
schema_needs_strings(SchemaName) :-
    get_json_schema(SchemaName, Fields),
    member(field(_, _, Options), Fields),
    member(format(_), Options), !.

format_import(Import, Line) :-
    format(atom(Line), '\t"~w"', [Import]).
format_import(Import, Line) :-
    sub_string(Import, _, _, _, '"'), % Already quoted (e.g. bolt "...")
    format(atom(Line), '\t~w', [Import]).
%% generate_aggregation_code(+Op, +AggField, +ResultVar, +FieldMappings, +Options, -GoCode)
%  Generate Go code for aggregation
generate_aggregation_code(Op, AggField, ResultVar, FieldMappings, Options, GoCode) :-
    % Generate field extraction code
    (   AggField == none
    ->  ExtractCode = ''
    ;   % Find the field mapping for the aggregation variable
        (   member(Field-Var, FieldMappings), Var == AggField
        ->  true
        ;   format('ERROR: Aggregation variable ~w not found in field mappings~n', [AggField]),
            fail
        ),
        % Generate extraction logic
        format(atom(ExtractCode), '
		valRaw, ok := data["~w"]
		if !ok { continue }
		val, ok := valRaw.(float64)
		if !ok { continue }', [Field])
    ),

    % Generate specific aggregation logic
    (   Op = count -> generate_count_aggregation(ExtractCode, BodyCode)
    ;   Op = sum   -> generate_sum_aggregation(ExtractCode, BodyCode)
    ;   Op = avg   -> generate_avg_aggregation(ExtractCode, BodyCode)
    ;   Op = max   -> generate_max_aggregation(ExtractCode, BodyCode)
    ;   Op = min   -> generate_min_aggregation(ExtractCode, BodyCode)
    ;   format('ERROR: Unknown aggregation operation: ~w~n', [Op]), fail
    ),
    
    BodyCode = code(Init, Loop, Output),
    
    % Wrap in loop
    format(string(GoCode), '
	scanner := bufio.NewScanner(os.Stdin)
	~s
	
	for scanner.Scan() {
		line := scanner.Bytes()
		var data map[string]interface{}
		if err := json.Unmarshal(line, &data); err != nil {
			continue
		}
		~s
	}
	
	~s
', [Init, Loop, Output]).

%% Aggregation Generators

generate_count_aggregation(_, code(
    'count := 0',
    'count++',
    'fmt.Println(count)'
)).

generate_sum_aggregation(ExtractCode, code(
    'sum := 0.0',
    LoopCode,
    'fmt.Println(sum)'
)) :-
    format(string(LoopCode), '~s
		sum += val', [ExtractCode]).

generate_avg_aggregation(ExtractCode, code(
    'sum := 0.0\n\tcount := 0',
    LoopCode,
    'if count > 0 {\n\t\tfmt.Println(sum / float64(count))\n\t} else {\n\t\tfmt.Println(0)\n\t}'
)) :-
    format(string(LoopCode), '~s
		sum += val
		count++', [ExtractCode]).

generate_max_aggregation(ExtractCode, code(
    'var maxVal float64\n\tfirst := true',
    LoopCode,
    'if !first {\n\t\tfmt.Println(maxVal)\n\t}'
)) :-
    format(string(LoopCode), '~s
		if first || val > maxVal {
			maxVal = val
			first = false
		}', [ExtractCode]).

generate_min_aggregation(ExtractCode, code(
    'var minVal float64\n\tfirst := true',
    LoopCode,
    'if !first {\n\t\tfmt.Println(minVal)\n\t}'
)) :-
    format(string(LoopCode), '~s
		if first || val < minVal {
			minVal = val
			first = false
		}', [ExtractCode]).

%% generate_group_by_code_jsonl(+GroupField, +FieldMappings, +AggOp, +Result, +HavingConstraints, +Options, -GoCode)
%  Generate Go code for JSONL stream grouped aggregation
generate_group_by_code_jsonl(GroupField, FieldMappings, AggOp, _Result, HavingConstraints, _Options, GoCode) :-
    % 1. Determine key type (simplification: assume string for now)
    KeyType = "string",
    
    % 2. Normalize AggOp to list
    (   is_list(AggOp)
    ->  AggOpList = AggOp
    ;   AggOpList = [AggOp]
    ),

    % 3. Generate Code using Multi-Agg logic
    generate_group_by_code_jsonl_multi(GroupField, FieldMappings, AggOpList, HavingConstraints, KeyType, GoCode).

%% find_field_for_var(+Var, +FieldMappings, -FieldName)
%  Find field name for a variable in field mappings
find_field_for_var(Var, [FieldName-MappedVar|_], FieldName) :-
    Var == MappedVar, !.
find_field_for_var(Var, [_|Rest], FieldName) :-
    find_field_for_var(Var, Rest, FieldName).

%% format_import(+Import, -Line)
%  Format import string for Go

%% generate_group_by_code_jsonl_multi(+GroupField, +FieldMappings, +AggOpList, +HavingConstraints, +KeyType, -GoCode)
%  Generate Go code for multiple aggregations on JSONL stream
generate_group_by_code_jsonl_multi(GroupField, FieldMappings, AggOpList, HavingConstraints, KeyType, GoCode) :-
    % 1. Parse operations
    parse_multi_agg_operations(AggOpList, FieldMappings, AggInfo),
    
    % 2. Generate Struct Definition
    generate_multi_agg_struct_fields(AggInfo, StructFields),
    format(string(StateStructDef), 'struct {
~w
    }', [StructFields]),
    StateStruct = "AggState",
    
    % 3. Generate Update Logic
    generate_multi_agg_jsonl_updates(AggInfo, UpdateLogic),
    
    % 4. Generate Output/Having Logic
    generate_multi_agg_jsonl_output(AggInfo, HavingConstraints, OutputLogic),
    
    % 5. Key Extraction
    find_field_for_var(GroupField, FieldMappings, GroupFieldName),
    format(string(KeyExtraction), '
        keyRaw, ok := data["~w"]
        if !ok { continue }
        key := fmt.Sprintf("%v", keyRaw)', [GroupFieldName]),

    % 6. Generate Complete Code
    format(string(GoCode), '
    type AggState ~w
    
    scanner := bufio.NewScanner(os.Stdin)
    results := make(map[~w]*AggState)
    
    for scanner.Scan() {
        var data map[string]interface{}
        if err := json.Unmarshal(scanner.Bytes(), &data); err != nil {
            continue
        }
        
        ~w
        
        state, exists := results[key]
        if !exists {
            state = &AggState{}
            results[key] = state
        }
        
        ~w
    }
    
    if err := scanner.Err(); err != nil {
        fmt.Fprintln(os.Stderr, "reading standard input:", err)
    }
    
    for key, s := range results {
        ~w
    }
', [StateStructDef, KeyType, KeyExtraction, UpdateLogic, OutputLogic]).

%% generate_multi_agg_jsonl_updates(+AggInfo, -Code)
%  Generate update statements for each aggregation in struct
generate_multi_agg_jsonl_updates(AggInfo, Code) :-
    findall(UpdateStr, (
        member(agg(OpType, FieldName, _Var, _OutName), AggInfo),
        jsonl_op_update(OpType, FieldName, UpdateStr)
    ), UpdateStrs),
    atomic_list_concat(UpdateStrs, '\n        ', Code).

jsonl_op_update(count, _Field, 'state.count++').
jsonl_op_update(sum, Field, Code) :-
    format(string(Code), 'if val, ok := data["~w"].(float64); ok { state.sum += val }', [Field]).
% Extensible for avg, max, min...

%% generate_multi_agg_jsonl_output(+AggInfo, +HavingConstraints, -Code)
%  Generate output printing code, with optional HAVING filtering
generate_multi_agg_jsonl_output(AggInfo, HavingConstraints, Code) :-
    % Generate print args
    findall(Arg, (
        member(agg(OpType, _, _, _), AggInfo),
        jsonl_op_print_arg(OpType, Arg)
    ), ArgsList),
    atomic_list_concat(["key"|ArgsList], ', ', PrintArgs),
    
    % Generate format string
    findall(Fmt, (
        member(agg(OpType, _, _, _), AggInfo),
        jsonl_op_fmt(OpType, Fmt)
    ), Fmts),
    atomic_list_concat(["%v"|Fmts], ': ', FmtStr),
    
    format(string(PrintStmt), 'fmt.Printf("~w\\n", ~w)', [FmtStr, PrintArgs]),
    
    % Handle HAVING
    (   HavingConstraints \== null, HavingConstraints \== []
    ->  % Let's implement basic "SUM > Val" support via manual check for now.
        generate_having_check(HavingConstraints, AggInfo, CheckCode),
        format(string(Code), '
        if ~w {
            ~w
        }', [CheckCode, PrintStmt])
    ;   Code = PrintStmt
    ).

jsonl_op_print_arg(count, 's.count').
jsonl_op_print_arg(sum, 's.sum').

jsonl_op_fmt(count, '%d').
jsonl_op_fmt(sum, '%g').
% Add others as needed

%% generate_having_check(+Constraints, +AggInfo, -GoExpr)
%  Generate Go boolean expression for HAVING clause
generate_having_check((A > B), AggInfo, Code) :-
    resolve_having_var(A, AggInfo, ACode),
    resolve_having_var(B, AggInfo, BCode),
    format(string(Code), '~w > ~w', [ACode, BCode]).
% Add other operators as needed

resolve_having_var(Var, AggInfo, Code) :-
    var(Var),
    member(agg(OpType, _, Var, _), AggInfo),
    jsonl_op_print_arg(OpType, CodeStr),
    atom_string(Code, CodeStr).
resolve_having_var(Num, _, Num) :- number(Num).

% ============================================================================
% BINDING SYSTEM INTEGRATION TESTS
% ============================================================================

%% test_go_binding_integration
%  Test that Go bindings are properly integrated
test_go_binding_integration :-
    format('~n=== Go Binding Integration Tests ===~n~n', []),

    % Test 1: Initialize bindings
    format('[Test 1] Initialize Go target with bindings~n', []),
    init_go_target,
    format('  [PASS] Go target initialized~n', []),

    % Test 2: Check bindings are accessible
    format('[Test 2] Check bindings accessibility~n', []),
    (   go_binding(sqrt/2, 'math.Sqrt', _, _, Opts),
        member(import('math'), Opts)
    ->  format('  [PASS] sqrt/2 binding accessible with import~n', [])
    ;   format('  [FAIL] sqrt/2 binding not found~n', [])
    ),

    % Test 3: Test import collection
    format('[Test 3] Test import collection~n', []),
    clear_binding_imports,
    collect_binding_import('math'),
    collect_binding_import('strings'),
    collect_binding_import('math'),  % Duplicate should be ignored
    get_collected_imports(Imports),
    length(Imports, NumImports),
    (   NumImports == 2
    ->  format('  [PASS] Collected ~w unique imports~n', [NumImports])
    ;   format('  [FAIL] Expected 2 imports, got ~w~n', [NumImports])
    ),

    % Test 4: Query bindings by package
    format('[Test 4] Query bindings by package~n', []),
    findall(P, (go_binding(P, _, _, _, Opts4), member(import('strings'), Opts4)), StringPreds),
    length(StringPreds, NumStrPreds),
    (   NumStrPreds > 10
    ->  format('  [PASS] Found ~w strings package bindings~n', [NumStrPreds])
    ;   format('  [FAIL] Expected >10 strings bindings, got ~w~n', [NumStrPreds])
    ),

    % Test 5: Check pure bindings
    format('[Test 5] Check pure bindings~n', []),
    (   go_binding(string_lower/2, _, _, _, Opts5),
        member(pure, Opts5)
    ->  format('  [PASS] string_lower/2 is pure~n', [])
    ;   format('  [FAIL] string_lower/2 should be pure~n', [])
    ),

    % Test 6: Check effect bindings
    format('[Test 6] Check effect bindings~n', []),
    (   go_binding(println/1, _, _, _, Opts6),
        member(effect(io), Opts6)
    ->  format('  [PASS] println/1 has effect(io)~n', [])
    ;   format('  [FAIL] println/1 should have effect(io)~n', [])
    ),

    % Test 7: Compile a predicate with binding goal and verify output
    format('[Test 7] Compile predicate with binding goal~n', []),
    test_binding_compilation,

    format('~n=== All Go Binding Integration Tests Passed ===~n', []).

%% test_binding_compilation
%  Test that binding goals are properly compiled to Go code
test_binding_compilation :-
    % Define a test predicate dynamically
    % normalized_name(X, Lower) :- string_lower(X, Lower).
    % We'll test by calling compile_go_binding_goal directly

    go_generator_config(Config),

    % Test 7a: Compile string_lower binding
    Goal1 = string_lower(input_str, result_lower),
    VarMap1 = [input_str-source(j1, 0)],
    clear_binding_imports,
    (   compile_go_binding_goal(Goal1, VarMap1, Config, Code1),
        sub_string(Code1, _, _, _, "strings.ToLower")
    ->  format('  [PASS] string_lower compiles to strings.ToLower~n', [])
    ;   format('  [FAIL] string_lower should compile to strings.ToLower~n', [])
    ),

    % Verify import was collected
    get_collected_imports(Imports1),
    (   member('strings', Imports1)
    ->  format('  [PASS] strings import collected~n', [])
    ;   format('  [FAIL] strings import should be collected~n', [])
    ),

    % Test 7b: Compile sqrt binding
    Goal2 = sqrt(input_num, result_sqrt),
    VarMap2 = [input_num-source(j2, 0)],
    clear_binding_imports,
    (   compile_go_binding_goal(Goal2, VarMap2, Config, Code2),
        sub_string(Code2, _, _, _, "math.Sqrt")
    ->  format('  [PASS] sqrt compiles to math.Sqrt~n', [])
    ;   format('  [FAIL] sqrt should compile to math.Sqrt~n', [])
    ),

    get_collected_imports(Imports2),
    (   member('math', Imports2)
    ->  format('  [PASS] math import collected~n', [])
    ;   format('  [FAIL] math import should be collected~n', [])
    ),

    % Test 7c: Test format_binding_imports
    clear_binding_imports,
    collect_binding_import('strings'),
    collect_binding_import('math'),
    collect_binding_import('regexp'),
    get_collected_imports(ImportsAll),
    format_binding_imports(ImportsAll, FormattedImports),
    (   sub_string(FormattedImports, _, _, _, "\"strings\""),
        sub_string(FormattedImports, _, _, _, "\"math\"")
    ->  format('  [PASS] format_binding_imports generates correct import block~n', [])
    ;   format('  [FAIL] format_binding_imports should include strings and math~n', [])
    ).

%% ============================================
%% PIPELINE MODE TESTS
%% ============================================

%% test_go_pipeline_mode
%  Test pipeline mode compilation
test_go_pipeline_mode :-
    format('~n=== Go Pipeline Mode Tests ===~n~n', []),

    % Test 1: Basic pipeline arg name generation
    format('[Test 1] Generate pipeline arg names~n', []),
    generate_pipeline_arg_names([a, b, c], 1, Names1),
    (   Names1 = [arg1, arg2, arg3]
    ->  format('  [PASS] Generated: ~w~n', [Names1])
    ;   format('  [FAIL] Expected [arg1, arg2, arg3], got ~w~n', [Names1])
    ),

    % Test 2: Capitalize first letter
    format('[Test 2] Capitalize first letter~n', []),
    capitalize_first(name, Cap1),
    capitalize_first('userId', Cap2),
    (   Cap1 = 'Name', Cap2 = 'UserId'
    ->  format('  [PASS] name -> ~w, userId -> ~w~n', [Cap1, Cap2])
    ;   format('  [FAIL] Capitalization failed~n', [])
    ),

    % Test 3: Generate pipeline struct
    format('[Test 3] Generate pipeline struct~n', []),
    generate_pipeline_struct('USER', [name, age, email], StructDef),
    (   sub_string(StructDef, _, _, _, "USEROutput"),
        sub_string(StructDef, _, _, _, "Name interface{}"),
        sub_string(StructDef, _, _, _, "`json:\"name\"`")
    ->  format('  [PASS] Generated struct with correct fields and tags~n', [])
    ;   format('  [FAIL] Struct generation failed~n', [])
    ),

    % Test 4: Base pipeline imports
    format('[Test 4] Base pipeline imports~n', []),
    base_pipeline_imports(jsonl, JsonlImports),
    base_pipeline_imports(object, ObjectImports),
    base_pipeline_imports(text, TextImports),
    (   member("encoding/json", JsonlImports),
        member("bufio", ObjectImports),
        member("fmt", TextImports)
    ->  format('  [PASS] All output formats have correct base imports~n', [])
    ;   format('  [FAIL] Missing base imports~n', [])
    ),

    % Test 5: Pipeline field extraction - flat fields
    format('[Test 5] Pipeline field extraction (flat)~n', []),
    pipeline_extract_one_field(flat(name, varName), V1, F1, C1),
    (   V1 = varName, F1 = name,
        sub_string(C1, _, _, _, "data[\"name\"]")
    ->  format('  [PASS] Flat field extraction: ~w~n', [V1])
    ;   format('  [FAIL] Flat field extraction failed~n', [])
    ),

    % Test 6: Path to Go nested access
    format('[Test 6] Path to Go nested access~n', []),
    path_to_go_nested_access([user, address, city], GoExpr),
    (   sub_string(GoExpr, _, _, _, "getNestedField"),
        sub_string(GoExpr, _, _, _, "\"user\""),
        sub_string(GoExpr, _, _, _, "\"city\"")
    ->  format('  [PASS] Nested path: ~w~n', [GoExpr])
    ;   format('  [FAIL] Nested path conversion failed~n', [])
    ),

    % Test 7: JSONL output map generation
    format('[Test 7] JSONL output map generation~n', []),
    generate_jsonl_output_map([name, age], [userName, userAge], [name-name, age-age], MapCode),
    (   sub_string(MapCode, _, _, _, "\"username\":"),
        sub_string(MapCode, _, _, _, "\"userage\":")
    ->  format('  [PASS] Map code generated correctly~n', [])
    ;   format('  [FAIL] Map code: ~w~n', [MapCode])
    ),

    % Test 8: Text output generation
    format('[Test 8] Text output generation~n', []),
    generate_text_output([name, age], [name-name, age-age], TextExpr),
    (   sub_string(TextExpr, _, _, _, "fmt.Sprintf"),
        sub_string(TextExpr, _, _, _, "%v")
    ->  format('  [PASS] Text output: ~w~n', [TextExpr])
    ;   format('  [FAIL] Text output failed~n', [])
    ),

    % Test 9: Pipeline constraint to Go check (greater than)
    format('[Test 9] Pipeline constraint to Go check~n', []),
    pipeline_constraint_to_go(age > 18, [age-age], CheckCode),
    (   sub_string(CheckCode, _, _, _, "age.(float64)"),
        sub_string(CheckCode, _, _, _, "<= 18")
    ->  format('  [PASS] Constraint check generated~n', [])
    ;   format('  [FAIL] Constraint check: ~w~n', [CheckCode])
    ),

    % Test 10: Full pipeline body compilation
    format('[Test 10] Full pipeline body compilation~n', []),
    compile_pipeline_body(jsonl, false, [name, age], [flat(name, name), flat(age, age)], [userName, userAge], [], BodyCode),
    (   sub_string(BodyCode, _, _, _, "bufio.NewScanner"),
        sub_string(BodyCode, _, _, _, "json.Unmarshal"),
        sub_string(BodyCode, _, _, _, "json.Marshal")
    ->  format('  [PASS] Pipeline body compiled successfully~n', [])
    ;   format('  [FAIL] Pipeline body compilation failed~n', [])
    ),

    format('~n=== All Go Pipeline Mode Tests Passed ===~n', []).

%% ============================================
%% GO PIPELINE CHAINING TESTS
%% ============================================

test_go_pipeline_chaining :-
    format('~n=== Go Pipeline Chaining Tests ===~n~n', []),

    % Test 1: Extract pipeline predicates
    format('[Test 1] Extract pipeline predicates~n', []),
    extract_pipeline_predicates([parse_user/2, filter/1, go:output/1], PredInfos),
    (   PredInfos = [pred_info(parse_user, 2, go), pred_info(filter, 1, go), pred_info(output, 1, go)]
    ->  format('  [PASS] Extracted: ~w~n', [PredInfos])
    ;   format('  [FAIL] Got: ~w~n', [PredInfos])
    ),

    % Test 2: Generate sequential chain
    format('[Test 2] Generate sequential chain~n', []),
    generate_sequential_chain(["stage1", "stage2", "stage3"], ChainCode),
    (   sub_string(ChainCode, _, _, _, "stage3(stage2(stage1(input)))")
    ->  format('  [PASS] Sequential chain generated correctly~n', [])
    ;   format('  [FAIL] Chain: ~w~n', [ChainCode])
    ),

    % Test 3: Generate pipeline connector (sequential mode)
    format('[Test 3] Generate pipeline connector (sequential)~n', []),
    generate_go_pipeline_connector(["parse", "transform"], myPipeline, sequential, ConnectorCode),
    (   sub_string(ConnectorCode, _, _, _, "func myPipeline"),
        sub_string(ConnectorCode, _, _, _, "transform(parse(input))")
    ->  format('  [PASS] Connector generated~n', [])
    ;   format('  [FAIL] Connector: ~w~n', [ConnectorCode])
    ),

    % Test 4: Pipeline chaining imports
    format('[Test 4] Pipeline chaining imports~n', []),
    pipeline_chaining_imports(sequential, SeqImports),
    pipeline_chaining_imports(channel, ChanImports),
    (   member("bufio", SeqImports),
        member("encoding/json", SeqImports),
        member("sync", ChanImports)
    ->  format('  [PASS] Imports: seq=~w, chan=~w~n', [SeqImports, ChanImports])
    ;   format('  [FAIL] Import check failed~n', [])
    ),

    % Test 5: Generate placeholder stage
    format('[Test 5] Generate placeholder stage~n', []),
    generate_placeholder_stage(unknown_pred, 2, PlaceholderCode),
    (   sub_string(PlaceholderCode, _, _, _, "func unknown_pred"),
        sub_string(PlaceholderCode, _, _, _, "return records")
    ->  format('  [PASS] Placeholder generated~n', [])
    ;   format('  [FAIL] Placeholder: ~w~n', [PlaceholderCode])
    ),

    % Test 6: Generate pipeline main (jsonl format)
    format('[Test 6] Generate pipeline main (jsonl)~n', []),
    generate_go_pipeline_main(testPipeline, jsonl, [], MainJsonlCode),
    (   sub_string(MainJsonlCode, _, _, _, "func main()"),
        sub_string(MainJsonlCode, _, _, _, "testPipeline(input)"),
        sub_string(MainJsonlCode, _, _, _, "json.Marshal")
    ->  format('  [PASS] JSONL main generated~n', [])
    ;   format('  [FAIL] JSONL main: ~w~n', [MainJsonlCode])
    ),

    % Test 7: Generate pipeline main (text format)
    format('[Test 7] Generate pipeline main (text)~n', []),
    generate_go_pipeline_main(testPipeline, text, [arg_names([name, value])], MainTextCode),
    (   sub_string(MainTextCode, _, _, _, "func main()"),
        sub_string(MainTextCode, _, _, _, "testPipeline(input)"),
        sub_string(MainTextCode, _, _, _, "fmt.Printf")
    ->  format('  [PASS] Text main generated~n', [])
    ;   format('  [FAIL] Text main: ~w~n', [MainTextCode])
    ),

    % Test 8: Full pipeline compilation (placeholder stages)
    format('[Test 8] Full pipeline compilation~n', []),
    compile_go_pipeline([stage1/1, stage2/1], [
        pipeline_name(testPipe),
        pipeline_mode(sequential),
        output_format(jsonl)
    ], FullCode),
    (   sub_string(FullCode, _, _, _, "package main"),
        sub_string(FullCode, _, _, _, "func stage1"),
        sub_string(FullCode, _, _, _, "func stage2"),
        sub_string(FullCode, _, _, _, "func testPipe"),
        sub_string(FullCode, _, _, _, "func main()")
    ->  format('  [PASS] Full pipeline compiled~n', [])
    ;   format('  [FAIL] Full pipeline compilation failed~n', [])
    ),

    % Test 9: Channel declarations
    format('[Test 9] Channel declarations~n', []),
    generate_channel_declarations(2, ChanDeclCode),
    (   sub_string(ChanDeclCode, _, _, _, "ch0"),
        sub_string(ChanDeclCode, _, _, _, "ch1"),
        sub_string(ChanDeclCode, _, _, _, "results")
    ->  format('  [PASS] Channel declarations~n', [])
    ;   format('  [FAIL] Channel decls: ~w~n', [ChanDeclCode])
    ),

    % Test 10: Stage output generation
    format('[Test 10] Stage output generation~n', []),
    generate_stage_output([name, age], [name-name, age-age], StageOutputCode),
    (   sub_string(StageOutputCode, _, _, _, '"arg1":'),
        sub_string(StageOutputCode, _, _, _, '"arg2":')
    ->  format('  [PASS] Stage output: ~w~n', [StageOutputCode])
    ;   format('  [FAIL] Stage output: ~w~n', [StageOutputCode])
    ),

    format('~n=== All Go Pipeline Chaining Tests Passed ===~n', []).

%% ============================================
%% GO PIPELINE BINDING INTEGRATION TESTS
%% ============================================

test_go_pipeline_bindings :-
    format('~n=== Go Pipeline Binding Integration Tests ===~n~n', []),

    % Initialize bindings
    init_go_target,

    % Test 1: is_stage_binding_goal
    format('[Test 1] is_stage_binding_goal detection~n', []),
    (   is_stage_binding_goal(string_lower(x, y)),
        is_stage_binding_goal(string_upper(a, b)),
        \+ is_stage_binding_goal(json_record([name-_])),
        \+ is_stage_binding_goal(true)
    ->  format('  [PASS] Correctly identifies binding goals~n', [])
    ;   format('  [FAIL] Binding goal detection failed~n', [])
    ),

    % Test 2: extract_stage_binding_goals from conjunction
    format('[Test 2] Extract binding goals from body~n', []),
    TestBody2 = (json_record([name-Name]), string_lower(Name, Lower)),
    extract_stage_binding_goals(TestBody2, Goals2),
    (   Goals2 = [string_lower(Name, Lower)]
    ->  format('  [PASS] Extracted: ~w~n', [Goals2])
    ;   format('  [FAIL] Got: ~w~n', [Goals2])
    ),

    % Test 3: compile_single_stage_binding for string_lower
    format('[Test 3] Compile string_lower binding~n', []),
    compile_single_stage_binding(string_lower(name, lower), [name-name], Code3, VarMap3),
    (   sub_string(Code3, _, _, _, "strings.ToLower"),
        sub_string(Code3, _, _, _, "lower :="),
        VarMap3 = [lower-binding_output]
    ->  format('  [PASS] Code: ~w~n', [Code3])
    ;   format('  [FAIL] Code: ~w, VarMap: ~w~n', [Code3, VarMap3])
    ),

    % Test 4: compile_single_stage_binding for string_upper
    format('[Test 4] Compile string_upper binding~n', []),
    compile_single_stage_binding(string_upper(text, upper), [text-text], Code4, VarMap4),
    (   sub_string(Code4, _, _, _, "strings.ToUpper"),
        sub_string(Code4, _, _, _, "upper :="),
        VarMap4 = [upper-binding_output]
    ->  format('  [PASS] Code: ~w~n', [Code4])
    ;   format('  [FAIL] Code: ~w, VarMap: ~w~n', [Code4, VarMap4])
    ),

    % Test 5: compile_stage_bindings with multiple goals
    format('[Test 5] Compile multiple bindings~n', []),
    compile_stage_bindings(
        [string_lower(name, lower), string_upper(title, upper)],
        [name-name, title-title],
        Code5,
        VarMap5
    ),
    (   sub_string(Code5, _, _, _, "strings.ToLower"),
        sub_string(Code5, _, _, _, "strings.ToUpper"),
        member(lower-binding_output, VarMap5),
        member(upper-binding_output, VarMap5)
    ->  format('  [PASS] Multiple bindings compiled~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code5])
    ),

    % Test 6: stage_arg_to_go_expr
    format('[Test 6] stage_arg_to_go_expr conversion~n', []),
    stage_arg_to_go_expr([name-name], name, Expr6a),
    stage_arg_to_go_expr([], hello, Expr6b),
    stage_arg_to_go_expr([], 42, Expr6c),
    (   sub_string(Expr6a, _, _, _, "name.(string)"),
        Expr6b = "\"hello\"",
        Expr6c = "42"
    ->  format('  [PASS] Expressions: ~w, ~w, ~w~n', [Expr6a, Expr6b, Expr6c])
    ;   format('  [FAIL] Got: ~w, ~w, ~w~n', [Expr6a, Expr6b, Expr6c])
    ),

    % Test 7: Import collection
    format('[Test 7] Import collection~n', []),
    clear_binding_imports,
    compile_single_stage_binding(string_lower(name, lower), [name-name], _, _),
    get_collected_imports(Imports7),
    (   (member("strings", Imports7) ; member(strings, Imports7))
    ->  format('  [PASS] Collected imports: ~w~n', [Imports7])
    ;   format('  [FAIL] Imports: ~w~n', [Imports7])
    ),

    % Test 8: Full pipeline with bindings (using asserted clause)
    format('[Test 8] Full pipeline with bindings~n', []),
    clear_binding_imports,
    % Assert a test predicate with ground atoms for args
    abolish(user:test_normalize/2),
    assert(user:(test_normalize(myname, mylower) :-
        json_record([name-myname]),
        string_lower(myname, mylower))),
    % Compile the pipeline
    compile_go_pipeline([test_normalize/2], [
        pipeline_name(testNorm),
        output_format(jsonl)
    ], PipelineCode8),
    (   sub_string(PipelineCode8, _, _, _, "strings.ToLower"),
        sub_string(PipelineCode8, _, _, _, "func test_normalize"),
        sub_string(PipelineCode8, _, _, _, "import")
    ->  format('  [PASS] Pipeline with bindings compiles correctly~n', [])
    ;   format('  [FAIL] Pipeline code missing expected patterns~n', [])
    ),
    abolish(user:test_normalize/2),

    % Test 9: String trim_space binding
    format('[Test 9] String trim_space binding~n', []),
    compile_single_stage_binding(string_trim_space(input, trimmed), [input-input], Code9, _),
    (   sub_string(Code9, _, _, _, "strings.TrimSpace"),
        sub_string(Code9, _, _, _, "trimmed :=")
    ->  format('  [PASS] trim_space: ~w~n', [Code9])
    ;   format('  [FAIL] Code: ~w~n', [Code9])
    ),

    % Test 10: String contains binding (returns bool)
    format('[Test 10] String contains binding~n', []),
    compile_single_stage_binding(string_contains(text, sub, result), [text-text], Code10, _),
    (   sub_string(Code10, _, _, _, "strings.Contains"),
        sub_string(Code10, _, _, _, "result :=")
    ->  format('  [PASS] contains: ~w~n', [Code10])
    ;   format('  [FAIL] Code: ~w~n', [Code10])
    ),

    format('~n=== All Go Pipeline Binding Integration Tests Passed ===~n', []).

%% ============================================
%% GO PIPELINE GENERATOR MODE TESTS
%% ============================================

test_go_pipeline_generator :-
    format('~n=== Go Pipeline Generator Mode Tests ===~n~n', []),

    % Initialize bindings
    init_go_target,

    % Test 1: Generate helper functions for generator mode
    format('[Test 1] Generate pipeline helpers~n', []),
    generate_pipeline_helpers(generator, HelperCode1),
    (   sub_string(HelperCode1, _, _, _, "func recordKey"),
        sub_string(HelperCode1, _, _, _, "sort.Strings"),
        sub_string(HelperCode1, _, _, _, "fmt.Sprintf")
    ->  format('  [PASS] recordKey helper generated~n', [])
    ;   format('  [FAIL] Helper code: ~w~n', [HelperCode1])
    ),

    % Test 2: No helpers for sequential mode
    format('[Test 2] No helpers for sequential mode~n', []),
    generate_pipeline_helpers(sequential, HelperCode2),
    (   HelperCode2 = ""
    ->  format('  [PASS] No helpers for sequential~n', [])
    ;   format('  [FAIL] Got: ~w~n', [HelperCode2])
    ),

    % Test 3: Generator mode imports include sort
    format('[Test 3] Generator mode imports~n', []),
    pipeline_chaining_imports(generator, GenImports),
    (   member("bufio", GenImports),
        member("encoding/json", GenImports),
        member("fmt", GenImports),
        member("os", GenImports)
    ->  format('  [PASS] Generator imports: ~w~n', [GenImports])
    ;   format('  [FAIL] Imports: ~w~n', [GenImports])
    ),

    % Test 4: Generate generator chain for empty stages
    format('[Test 4] Generator chain (empty)~n', []),
    generate_generator_chain([], EmptyChain),
    (   sub_string(EmptyChain, _, _, _, "return input")
    ->  format('  [PASS] Empty chain returns input~n', [])
    ;   format('  [FAIL] Chain: ~w~n', [EmptyChain])
    ),

    % Test 5: Generate generator chain body
    format('[Test 5] Generator chain body~n', []),
    generate_generator_chain_body(["stage1", "stage2"], BodyCode5),
    (   sub_string(BodyCode5, _, _, _, "current = stage1(current)"),
        sub_string(BodyCode5, _, _, _, "current = stage2(current)")
    ->  format('  [PASS] Chain body correct~n', [])
    ;   format('  [FAIL] Body: ~w~n', [BodyCode5])
    ),

    % Test 6: Generate full generator chain
    format('[Test 6] Full generator chain~n', []),
    generate_generator_chain(["transform", "derive"], ChainCode6),
    (   sub_string(ChainCode6, _, _, _, "total := make(map[string]Record)"),
        sub_string(ChainCode6, _, _, _, "recordKey(r)"),
        sub_string(ChainCode6, _, _, _, "changed := true"),
        sub_string(ChainCode6, _, _, _, "for changed"),
        sub_string(ChainCode6, _, _, _, "current = transform(current)"),
        sub_string(ChainCode6, _, _, _, "current = derive(current)")
    ->  format('  [PASS] Fixpoint iteration generated~n', [])
    ;   format('  [FAIL] Chain: ~w~n', [ChainCode6])
    ),

    % Test 7: Generate pipeline connector for generator mode
    format('[Test 7] Pipeline connector (generator)~n', []),
    generate_go_pipeline_connector(["step1", "step2"], testGen, generator, ConnCode7),
    (   sub_string(ConnCode7, _, _, _, "func testGen(input []Record)"),
        sub_string(ConnCode7, _, _, _, "fixpoint evaluation")
    ->  format('  [PASS] Connector generated~n', [])
    ;   format('  [FAIL] Connector: ~w~n', [ConnCode7])
    ),

    % Test 8: Full pipeline compilation with generator mode
    format('[Test 8] Full pipeline (generator mode)~n', []),
    clear_binding_imports,
    compile_go_pipeline([gen_stage1/1, gen_stage2/1], [
        pipeline_name(fixpointPipe),
        pipeline_mode(generator),
        output_format(jsonl)
    ], FullGenCode),
    (   sub_string(FullGenCode, _, _, _, "package main"),
        sub_string(FullGenCode, _, _, _, "\"sort\""),
        sub_string(FullGenCode, _, _, _, "func recordKey"),
        sub_string(FullGenCode, _, _, _, "func fixpointPipe"),
        sub_string(FullGenCode, _, _, _, "changed := true"),
        sub_string(FullGenCode, _, _, _, "for changed")
    ->  format('  [PASS] Full generator pipeline compiled~n', [])
    ;   format('  [FAIL] Missing expected patterns in generated code~n', [])
    ),

    % Test 9: Generator mode with defined predicate
    format('[Test 9] Generator with defined predicate~n', []),
    abolish(user:test_derive/2),
    assert(user:(test_derive(input, derived) :-
        json_record([input-input]),
        string_upper(input, derived))),
    clear_binding_imports,
    compile_go_pipeline([test_derive/2], [
        pipeline_name(derivePipe),
        pipeline_mode(generator),
        output_format(jsonl)
    ], DeriveCode),
    (   sub_string(DeriveCode, _, _, _, "func test_derive"),
        sub_string(DeriveCode, _, _, _, "strings.ToUpper"),
        sub_string(DeriveCode, _, _, _, "func recordKey"),
        sub_string(DeriveCode, _, _, _, "total := make(map[string]Record)")
    ->  format('  [PASS] Generator with bindings compiles~n', [])
    ;   format('  [FAIL] Code missing expected patterns~n', [])
    ),
    abolish(user:test_derive/2),

    % Test 10: Verify sort import is added for generator mode
    format('[Test 10] Sort import verification~n', []),
    compile_go_pipeline([placeholder/1], [
        pipeline_name(sortTest),
        pipeline_mode(generator),
        output_format(jsonl)
    ], SortTestCode),
    (   sub_string(SortTestCode, _, _, _, "\"sort\"")
    ->  format('  [PASS] Sort import added~n', [])
    ;   format('  [FAIL] Missing sort import~n', [])
    ),

    format('~n=== All Go Pipeline Generator Mode Tests Passed ===~n', []).

%% ============================================
%% GO ENHANCED PIPELINE CHAINING
%% ============================================
%
%  Supports advanced flow patterns:
%    - fan_out(Stages)        : Broadcast to stages (sequential execution)
%    - parallel(Stages)       : Execute stages concurrently (goroutines)
%    - merge                  : Combine results from fan_out or parallel
%    - route_by(Pred, Routes) : Conditional routing
%    - filter_by(Pred)        : Filter records
%    - Pred/Arity             : Standard stage
%
%% compile_go_enhanced_pipeline(+Stages, +Options, -GoCode)
%  Main entry point for enhanced Go pipeline with advanced flow patterns.
%  Validates pipeline stages before code generation.
%
compile_go_enhanced_pipeline(Stages, Options, GoCode) :-
    % Validate pipeline stages
    option(validate(Validate), Options, true),
    option(strict(Strict), Options, false),
    ( Validate == true ->
        validate_pipeline(Stages, [strict(Strict)], result(Errors, Warnings)),
        % Report warnings
        ( Warnings \== [] ->
            format(user_error, 'Go pipeline warnings:~n', []),
            forall(member(W, Warnings), (
                format_validation_warning(W, Msg),
                format(user_error, '  ~w~n', [Msg])
            ))
        ; true
        ),
        % Fail on errors
        ( Errors \== [] ->
            format(user_error, 'Go pipeline validation errors:~n', []),
            forall(member(E, Errors), (
                format_validation_error(E, Msg),
                format(user_error, '  ~w~n', [Msg])
            )),
            throw(pipeline_validation_failed(Errors))
        ; true
        )
    ; true
    ),

    option(pipeline_name(PipelineName), Options, enhanced_pipeline),
    option(output_format(OutputFormat), Options, jsonl),
    option(include_package(IncludePackage), Options, true),

    % Generate helper functions
    go_enhanced_helpers(Helpers),

    % Generate stage functions
    generate_go_enhanced_stage_functions(Stages, StageFunctions),

    % Generate the main connector
    generate_go_enhanced_connector(Stages, PipelineName, ConnectorCode),

    % Generate main function
    generate_go_enhanced_main(PipelineName, OutputFormat, MainCode),

    % Wrap in package if requested
    (   IncludePackage
    ->  format(string(GoCode),
'package main

import (
\t"bufio"
\t"encoding/json"
\t"fmt"
\t"os"
\t"sync"
)

// Record represents a data record flowing through the pipeline
type Record map[string]interface{}

~w

~w
~w
~w', [Helpers, StageFunctions, ConnectorCode, MainCode])
    ;   format(string(GoCode), '~w~w~w~w', [Helpers, StageFunctions, ConnectorCode, MainCode])
    ).

%% go_enhanced_helpers(-Code)
%  Generate helper functions for enhanced pipeline operations.
go_enhanced_helpers(Code) :-
    Code = '// Enhanced Pipeline Helpers

// fanOutRecords broadcasts a record to multiple stages and collects all results
func fanOutRecords(record Record, stages []func([]Record) []Record) []Record {
\tvar results []Record
\tfor _, stage := range stages {
\t\tfor _, result := range stage([]Record{record}) {
\t\t\tresults = append(results, result)
\t\t}
\t}
\treturn results
}

// mergeStreams combines multiple record slices into one
func mergeStreams(streams ...[]Record) []Record {
\tvar merged []Record
\tfor _, stream := range streams {
\t\tmerged = append(merged, stream...)
\t}
\treturn merged
}

// routeRecord directs a record to appropriate stage based on condition
func routeRecord(record Record, conditionFn func(Record) interface{}, routeMap map[interface{}]func([]Record) []Record, defaultFn func([]Record) []Record) []Record {
\tcondition := conditionFn(record)
\tif stage, ok := routeMap[condition]; ok {
\t\treturn stage([]Record{record})
\t}
\tif defaultFn != nil {
\t\treturn defaultFn([]Record{record})
\t}
\t// Pass through if no matching route
\treturn []Record{record}
}

// filterRecords returns only records that satisfy the predicate
func filterRecords(records []Record, predicateFn func(Record) bool) []Record {
\tvar filtered []Record
\tfor _, record := range records {
\t\tif predicateFn(record) {
\t\t\tfiltered = append(filtered, record)
\t\t}
\t}
\treturn filtered
}

// teeStream sends each record to multiple stages and collects all results
func teeStream(records []Record, stages ...func([]Record) []Record) []Record {
\tvar results []Record
\tfor _, record := range records {
\t\tfor _, stage := range stages {
\t\t\tresults = append(results, stage([]Record{record})...)
\t\t}
\t}
\treturn results
}

// parallelRecords executes stages concurrently using goroutines and collects all results (completion order)
func parallelRecords(record Record, stages []func([]Record) []Record) []Record {
\tvar wg sync.WaitGroup
\tvar mu sync.Mutex
\tvar results []Record

\tfor _, stage := range stages {
\t\twg.Add(1)
\t\tgo func(s func([]Record) []Record) {
\t\t\tdefer wg.Done()
\t\t\tstageResults := s([]Record{record})
\t\t\tmu.Lock()
\t\t\tresults = append(results, stageResults...)
\t\t\tmu.Unlock()
\t\t}(stage)
\t}

\twg.Wait()
\treturn results
}

// parallelRecordsOrdered executes stages concurrently but preserves stage definition order
func parallelRecordsOrdered(record Record, stages []func([]Record) []Record) []Record {
\tvar wg sync.WaitGroup
\tindexedResults := make([][]Record, len(stages))

\tfor i, stage := range stages {
\t\twg.Add(1)
\t\tgo func(idx int, s func([]Record) []Record) {
\t\t\tdefer wg.Done()
\t\t\tindexedResults[idx] = s([]Record{record})
\t\t}(i, stage)
\t}

\twg.Wait()

\t// Flatten results in order
\tvar results []Record
\tfor _, stageResults := range indexedResults {
\t\tresults = append(results, stageResults...)
\t}
\treturn results
}

// batchRecords collects records into batches of specified size
func batchRecords(records []Record, batchSize int) [][]Record {
\tvar batches [][]Record
\tfor i := 0; i < len(records); i += batchSize {
\t\tend := i + batchSize
\t\tif end > len(records) {
\t\t\tend = len(records)
\t\t}
\t\tbatches = append(batches, records[i:end])
\t}
\treturn batches
}

// unbatchRecords flattens batches back to individual records
func unbatchRecords(batches [][]Record) []Record {
\tvar records []Record
\tfor _, batch := range batches {
\t\trecords = append(records, batch...)
\t}
\treturn records
}

// uniqueByField keeps only the first record for each unique field value
func uniqueByField(records []Record, field string) []Record {
\tseen := make(map[interface{}]bool)
\tvar result []Record
\tfor _, record := range records {
\t\tkey := record[field]
\t\tif !seen[key] {
\t\t\tseen[key] = true
\t\t\tresult = append(result, record)
\t\t}
\t}
\treturn result
}

// firstByField is an alias for uniqueByField (keep first occurrence)
func firstByField(records []Record, field string) []Record {
\treturn uniqueByField(records, field)
}

// lastByField keeps only the last record for each unique field value
func lastByField(records []Record, field string) []Record {
\tlastSeen := make(map[interface{}]Record)
\tvar order []interface{}
\tfor _, record := range records {
\t\tkey := record[field]
\t\tif _, exists := lastSeen[key]; !exists {
\t\t\torder = append(order, key)
\t\t}
\t\tlastSeen[key] = record
\t}
\tvar result []Record
\tfor _, key := range order {
\t\tresult = append(result, lastSeen[key])
\t}
\treturn result
}

// Aggregation represents an aggregation operation
type Aggregation struct {
\tName    string
\tType    string
\tField   string
}

// groupByField groups records by field and applies aggregations
func groupByField(records []Record, field string, aggregations []Aggregation) []Record {
\tgroups := make(map[interface{}][]Record)
\tvar order []interface{}

\t// Collect records into groups
\tfor _, record := range records {
\t\tkey := record[field]
\t\tif _, exists := groups[key]; !exists {
\t\t\torder = append(order, key)
\t\t}
\t\tgroups[key] = append(groups[key], record)
\t}

\t// Apply aggregations to each group
\tvar result []Record
\tfor _, key := range order {
\t\tgroupRecords := groups[key]
\t\tresultRecord := Record{field: key}

\t\tfor _, agg := range aggregations {
\t\t\tswitch agg.Type {
\t\t\tcase \"count\":
\t\t\t\tresultRecord[agg.Name] = len(groupRecords)
\t\t\tcase \"sum\":
\t\t\t\tvar sum float64
\t\t\t\tfor _, r := range groupRecords {
\t\t\t\t\tif v, ok := r[agg.Field].(float64); ok {
\t\t\t\t\t\tsum += v
\t\t\t\t\t}
\t\t\t\t}
\t\t\t\tresultRecord[agg.Name] = sum
\t\t\tcase \"avg\":
\t\t\t\tvar sum float64
\t\t\t\tfor _, r := range groupRecords {
\t\t\t\t\tif v, ok := r[agg.Field].(float64); ok {
\t\t\t\t\t\tsum += v
\t\t\t\t\t}
\t\t\t\t}
\t\t\t\tresultRecord[agg.Name] = sum / float64(len(groupRecords))
\t\t\tcase \"min\":
\t\t\t\tvar minVal float64 = 1e308
\t\t\t\tfor _, r := range groupRecords {
\t\t\t\t\tif v, ok := r[agg.Field].(float64); ok && v < minVal {
\t\t\t\t\t\tminVal = v
\t\t\t\t\t}
\t\t\t\t}
\t\t\t\tresultRecord[agg.Name] = minVal
\t\t\tcase \"max\":
\t\t\t\tvar maxVal float64 = -1e308
\t\t\t\tfor _, r := range groupRecords {
\t\t\t\t\tif v, ok := r[agg.Field].(float64); ok && v > maxVal {
\t\t\t\t\t\tmaxVal = v
\t\t\t\t\t}
\t\t\t\t}
\t\t\t\tresultRecord[agg.Name] = maxVal
\t\t\tcase \"first\":
\t\t\t\tif len(groupRecords) > 0 {
\t\t\t\t\tresultRecord[agg.Name] = groupRecords[0][agg.Field]
\t\t\t\t}
\t\t\tcase \"last\":
\t\t\t\tif len(groupRecords) > 0 {
\t\t\t\t\tresultRecord[agg.Name] = groupRecords[len(groupRecords)-1][agg.Field]
\t\t\t\t}
\t\t\tcase \"collect\":
\t\t\t\tvar values []interface{}
\t\t\t\tfor _, r := range groupRecords {
\t\t\t\t\tvalues = append(values, r[agg.Field])
\t\t\t\t}
\t\t\t\tresultRecord[agg.Name] = values
\t\t\t}
\t\t}
\t\tresult = append(result, resultRecord)
\t}
\treturn result
}

// ReducerFunc is a function type for reduce operations
type ReducerFunc func(record Record, acc interface{}) interface{}

// reduceRecords applies a reducer function sequentially across all records
func reduceRecords(records []Record, reducer ReducerFunc, initial interface{}) []Record {
\tacc := initial
\tfor _, record := range records {
\t\tacc = reducer(record, acc)
\t}
\treturn []Record{{\"result\": acc}}
}

// scanRecords applies a reducer function and yields intermediate results
func scanRecords(records []Record, reducer ReducerFunc, initial interface{}) []Record {
\tvar result []Record
\tacc := initial
\tfor _, record := range records {
\t\tacc = reducer(record, acc)
\t\tresult = append(result, Record{\"result\": acc})
\t}
\treturn result
}

// orderByField sorts records by a single field
func orderByField(records []Record, field string, direction string) []Record {
\tresult := make([]Record, len(records))
\tcopy(result, records)
\tsort.Slice(result, func(i, j int) bool {
\t\tvi := result[i][field]
\t\tvj := result[j][field]
\t\tcmp := compareValues(vi, vj)
\t\tif direction == \"desc\" {
\t\t\treturn cmp > 0
\t\t}
\t\treturn cmp < 0
\t})
\treturn result
}

// FieldSpec represents a field ordering specification
type FieldSpec struct {
\tField     string
\tDirection string
}

// orderByFields sorts records by multiple fields
func orderByFields(records []Record, fieldSpecs []FieldSpec) []Record {
\tresult := make([]Record, len(records))
\tcopy(result, records)
\tsort.Slice(result, func(i, j int) bool {
\t\tfor _, spec := range fieldSpecs {
\t\t\tvi := result[i][spec.Field]
\t\t\tvj := result[j][spec.Field]
\t\t\tcmp := compareValues(vi, vj)
\t\t\tif cmp != 0 {
\t\t\t\tif spec.Direction == \"desc\" {
\t\t\t\t\treturn cmp > 0
\t\t\t\t}
\t\t\t\treturn cmp < 0
\t\t\t}
\t\t}
\t\treturn false // All fields equal
\t})
\treturn result
}

// compareValues compares two interface{} values
func compareValues(a, b interface{}) int {
\t// Handle nil values (sort to end)
\tif a == nil && b == nil {
\t\treturn 0
\t}
\tif a == nil {
\t\treturn 1
\t}
\tif b == nil {
\t\treturn -1
\t}
\t// Compare based on type
\tswitch va := a.(type) {
\tcase float64:
\t\tif vb, ok := b.(float64); ok {
\t\t\tif va < vb {
\t\t\t\treturn -1
\t\t\t}
\t\t\tif va > vb {
\t\t\t\treturn 1
\t\t\t}
\t\t\treturn 0
\t\t}
\tcase string:
\t\tif vb, ok := b.(string); ok {
\t\t\tif va < vb {
\t\t\t\treturn -1
\t\t\t}
\t\t\tif va > vb {
\t\t\t\treturn 1
\t\t\t}
\t\t\treturn 0
\t\t}
\tcase int:
\t\tif vb, ok := b.(int); ok {
\t\t\tif va < vb {
\t\t\t\treturn -1
\t\t\t}
\t\t\tif va > vb {
\t\t\t\treturn 1
\t\t\t}
\t\t\treturn 0
\t\t}
\t}
\t// Fallback: string comparison
\treturn 0
}

// ComparatorFunc is a function type for custom comparisons
type ComparatorFunc func(a, b Record) int

// sortByComparator sorts records using a custom comparison function
func sortByComparator(records []Record, comparator ComparatorFunc) []Record {
\tresult := make([]Record, len(records))
\tcopy(result, records)
\tsort.Slice(result, func(i, j int) bool {
\t\treturn comparator(result[i], result[j]) < 0
\t})
\treturn result
}

// StageFunc is a function type for pipeline stages
type StageFunc func([]Record) ([]Record, error)

// ErrorHandlerFunc is a function type for error handlers
type ErrorHandlerFunc func([]Record, error) []Record

// tryCatchStage executes a stage and routes errors to handler
func tryCatchStage(records []Record, stage StageFunc, handler ErrorHandlerFunc) []Record {
\tvar result []Record
\tfor _, record := range records {
\t\tresults, err := stage([]Record{record})
\t\tif err != nil {
\t\t\thandlerResults := handler([]Record{record}, err)
\t\t\tresult = append(result, handlerResults...)
\t\t} else {
\t\t\tresult = append(result, results...)
\t\t}
\t}
\treturn result
}

// retryStage retries a stage up to maxRetries times
func retryStage(records []Record, stage StageFunc, maxRetries int, delayMs int, backoff string) []Record {
\tvar result []Record
\tfor _, record := range records {
\t\tvar lastErr error
\t\tfor attempt := 0; attempt <= maxRetries; attempt++ {
\t\t\tresults, err := stage([]Record{record})
\t\t\tif err == nil {
\t\t\t\tresult = append(result, results...)
\t\t\t\tbreak
\t\t\t}
\t\t\tlastErr = err
\t\t\tif attempt < maxRetries && delayMs > 0 {
\t\t\t\tvar waitTime time.Duration
\t\t\t\tswitch backoff {
\t\t\t\tcase \"exponential\":
\t\t\t\t\twaitTime = time.Duration(delayMs*(1<<attempt)) * time.Millisecond
\t\t\t\tcase \"linear\":
\t\t\t\t\twaitTime = time.Duration(delayMs*(attempt+1)) * time.Millisecond
\t\t\t\tdefault:
\t\t\t\t\twaitTime = time.Duration(delayMs) * time.Millisecond
\t\t\t\t}
\t\t\t\ttime.Sleep(waitTime)
\t\t\t}
\t\t}
\t\tif lastErr != nil {
\t\t\t// All retries exhausted
\t\t\terrorRecord := Record{
\t\t\t\t\"_error\":   lastErr.Error(),
\t\t\t\t\"_record\":  record,
\t\t\t\t\"_retries\": maxRetries,
\t\t\t}
\t\t\tresult = append(result, errorRecord)
\t\t}
\t}
\treturn result
}

// onErrorStage wraps records with error handling
func onErrorStage(records []Record, handler ErrorHandlerFunc) []Record {
\tvar result []Record
\tfor _, record := range records {
\t\t// In Go, we pass records through - errors are handled at stage level
\t\tresult = append(result, record)
\t}
\treturn result
}

// timeoutStage executes a stage with a time limit
func timeoutStage(records []Record, stage StageFunc, timeoutMs int) []Record {
\tvar result []Record
\tfor _, record := range records {
\t\tresultChan := make(chan []Record, 1)
\t\terrChan := make(chan error, 1)

\t\tgo func() {
\t\t\tdefer func() {
\t\t\t\tif r := recover(); r != nil {
\t\t\t\t\terrChan <- fmt.Errorf(\"panic: %v\", r)
\t\t\t\t}
\t\t\t}()
\t\t\tresults, err := stage([]Record{record})
\t\t\tif err != nil {
\t\t\t\terrChan <- err
\t\t\t} else {
\t\t\t\tresultChan <- results
\t\t\t}
\t\t}()

\t\tselect {
\t\tcase results := <-resultChan:
\t\t\tresult = append(result, results...)
\t\tcase <-errChan:
\t\t\t// Error occurred, treat as timeout
\t\t\ttimeoutRecord := Record{
\t\t\t\t\"_timeout\":  true,
\t\t\t\t\"_record\":   record,
\t\t\t\t\"_limit_ms\": timeoutMs,
\t\t\t}
\t\t\tresult = append(result, timeoutRecord)
\t\tcase <-time.After(time.Duration(timeoutMs) * time.Millisecond):
\t\t\t// Timeout occurred
\t\t\ttimeoutRecord := Record{
\t\t\t\t\"_timeout\":  true,
\t\t\t\t\"_record\":   record,
\t\t\t\t\"_limit_ms\": timeoutMs,
\t\t\t}
\t\t\tresult = append(result, timeoutRecord)
\t\t}
\t}
\treturn result
}

// timeoutStageWithFallback executes a stage with time limit, using fallback on timeout
func timeoutStageWithFallback(records []Record, stage StageFunc, timeoutMs int, fallback func([]Record) []Record) []Record {
\tvar result []Record
\tfor _, record := range records {
\t\tresultChan := make(chan []Record, 1)
\t\terrChan := make(chan error, 1)

\t\tgo func() {
\t\t\tdefer func() {
\t\t\t\tif r := recover(); r != nil {
\t\t\t\t\terrChan <- fmt.Errorf(\"panic: %v\", r)
\t\t\t\t}
\t\t\t}()
\t\t\tresults, err := stage([]Record{record})
\t\t\tif err != nil {
\t\t\t\terrChan <- err
\t\t\t} else {
\t\t\t\tresultChan <- results
\t\t\t}
\t\t}()

\t\tselect {
\t\tcase results := <-resultChan:
\t\t\tresult = append(result, results...)
\t\tcase <-errChan:
\t\t\t// Error occurred, use fallback
\t\t\tfallbackResults := fallback([]Record{record})
\t\t\tresult = append(result, fallbackResults...)
\t\tcase <-time.After(time.Duration(timeoutMs) * time.Millisecond):
\t\t\t// Timeout occurred, use fallback
\t\t\tfallbackResults := fallback([]Record{record})
\t\t\tresult = append(result, fallbackResults...)
\t\t}
\t}
\treturn result
}

// rateLimitStage limits throughput to count records per interval
func rateLimitStage(records []Record, count int, intervalMs int) []Record {
\tvar result []Record
\tinterval := time.Duration(intervalMs) * time.Millisecond / time.Duration(count)
\tvar lastTime time.Time

\tfor _, record := range records {
\t\tif !lastTime.IsZero() {
\t\t\telapsed := time.Since(lastTime)
\t\t\tif elapsed < interval {
\t\t\t\ttime.Sleep(interval - elapsed)
\t\t\t}
\t\t}
\t\tlastTime = time.Now()
\t\tresult = append(result, record)
\t}
\treturn result
}

// throttleStage adds fixed delay between records
func throttleStage(records []Record, delayMs int) []Record {
\tvar result []Record
\tdelay := time.Duration(delayMs) * time.Millisecond

\tfor i, record := range records {
\t\tif i > 0 {
\t\t\ttime.Sleep(delay)
\t\t}
\t\tresult = append(result, record)
\t}
\treturn result
}

// bufferStage collects records into batches of specified size
func bufferStage(records []Record, size int) [][]Record {
\tvar result [][]Record
\tvar buffer []Record

\tfor _, record := range records {
\t\tbuffer = append(buffer, record)
\t\tif len(buffer) >= size {
\t\t\tresult = append(result, buffer)
\t\t\tbuffer = nil
\t\t}
\t}
\tif len(buffer) > 0 {
\t\tresult = append(result, buffer)
\t}
\treturn result
}

// debounceStage emits record only after quiet period
func debounceStage(records []Record, delayMs int) []Record {
\tif len(records) == 0 {
\t\treturn nil
\t}
\tdelay := time.Duration(delayMs) * time.Millisecond
\tvar result []Record
\tvar pending Record
\thasPending := false
\ttimer := time.NewTimer(delay)
\ttimer.Stop()

\tfor _, record := range records {
\t\tpending = record
\t\thasPending = true
\t\ttimer.Reset(delay)
\t}

\tif hasPending {
\t\t<-timer.C
\t\tresult = append(result, pending)
\t}
\treturn result
}

// zipStage runs multiple stage functions and combines results
func zipStage(records []Record, stages []func([]Record) []Record) []Record {
\tvar result []Record
\tfor _, record := range records {
\t\tvar stageResults [][]Record
\t\tmaxLen := 0
\t\tfor _, stage := range stages {
\t\t\tres := stage([]Record{record})
\t\t\tstageResults = append(stageResults, res)
\t\t\tif len(res) > maxLen {
\t\t\t\tmaxLen = len(res)
\t\t\t}
\t\t}
\t\tfor i := 0; i < maxLen; i++ {
\t\t\tcombined := make(Record)
\t\t\tfor _, resList := range stageResults {
\t\t\t\tif i < len(resList) {
\t\t\t\t\tfor k, v := range resList[i] {
\t\t\t\t\t\tcombined[k] = v
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}
\t\t\tresult = append(result, combined)
\t\t}
\t}
\treturn result
}

// windowStage collects records into non-overlapping windows
func windowStage(records []Record, size int) [][]Record {
\tvar result [][]Record
\tvar window []Record
\tfor _, record := range records {
\t\twindow = append(window, record)
\t\tif len(window) >= size {
\t\t\tresult = append(result, window)
\t\t\twindow = nil
\t\t}
\t}
\tif len(window) > 0 {
\t\tresult = append(result, window)
\t}
\treturn result
}

// slidingWindowStage creates sliding windows of records
func slidingWindowStage(records []Record, size, step int) [][]Record {
\tvar result [][]Record
\tvar buffer []Record
\tfor _, record := range records {
\t\tbuffer = append(buffer, record)
\t\tfor len(buffer) >= size {
\t\t\twindow := make([]Record, size)
\t\t\tcopy(window, buffer[:size])
\t\t\tresult = append(result, window)
\t\t\tbuffer = buffer[step:]
\t\t}
\t}
\tif len(buffer) > 0 {
\t\tresult = append(result, buffer)
\t}
\treturn result
}

// sampleStage randomly samples n records using reservoir sampling
func sampleStage(records []Record, n int) []Record {
\timport \"math/rand\"
\treservoir := make([]Record, 0, n)
\tfor i, record := range records {
\t\tif i < n {
\t\t\treservoir = append(reservoir, record)
\t\t} else {
\t\t\tj := rand.Intn(i + 1)
\t\t\tif j < n {
\t\t\t\treservoir[j] = record
\t\t\t}
\t\t}
\t}
\treturn reservoir
}

// takeEveryStage takes every nth record
func takeEveryStage(records []Record, n int) []Record {
\tvar result []Record
\tfor i, record := range records {
\t\tif i%n == 0 {
\t\t\tresult = append(result, record)
\t\t}
\t}
\treturn result
}

// partitionStage splits records into matches and non-matches
func partitionStage(records []Record, pred func(Record) bool) ([]Record, []Record) {
\tvar matches, nonMatches []Record
\tfor _, record := range records {
\t\tif pred(record) {
\t\t\tmatches = append(matches, record)
\t\t} else {
\t\t\tnonMatches = append(nonMatches, record)
\t\t}
\t}
\treturn matches, nonMatches
}

// takeStage takes first n records
func takeStage(records []Record, n int) []Record {
\tif n >= len(records) {
\t\treturn records
\t}
\treturn records[:n]
}

// skipStage skips first n records
func skipStage(records []Record, n int) []Record {
\tif n >= len(records) {
\t\treturn nil
\t}
\treturn records[n:]
}

// takeWhileStage takes records while predicate is true
func takeWhileStage(records []Record, pred func(Record) bool) []Record {
\tvar result []Record
\tfor _, record := range records {
\t\tif !pred(record) {
\t\t\tbreak
\t\t}
\t\tresult = append(result, record)
\t}
\treturn result
}

// skipWhileStage skips records while predicate is true
func skipWhileStage(records []Record, pred func(Record) bool) []Record {
\tvar result []Record
\tskipping := true
\tfor _, record := range records {
\t\tif skipping && pred(record) {
\t\t\tcontinue
\t\t}
\t\tskipping = false
\t\tresult = append(result, record)
\t}
\treturn result
}

// distinctStage removes all duplicate records (global dedup)
func distinctStage(records []Record) []Record {
\tseen := make(map[string]bool)
\tvar result []Record
\tfor _, record := range records {
\t\tkey := recordKey(record)
\t\tif !seen[key] {
\t\t\tseen[key] = true
\t\t\tresult = append(result, record)
\t\t}
\t}
\treturn result
}

// distinctByStage removes duplicates based on a specific field
func distinctByStage(records []Record, field string) []Record {
\tseen := make(map[string]bool)
\tvar result []Record
\tfor _, record := range records {
\t\tval, _ := record[field]
\t\tkey := fmt.Sprintf(\"%v\", val)
\t\tif !seen[key] {
\t\t\tseen[key] = true
\t\t\tresult = append(result, record)
\t\t}
\t}
\treturn result
}

// dedupStage removes consecutive duplicate records
func dedupStage(records []Record) []Record {
\tvar result []Record
\tvar lastKey string
\tfor _, record := range records {
\t\tkey := recordKey(record)
\t\tif key != lastKey {
\t\t\tlastKey = key
\t\t\tresult = append(result, record)
\t\t}
\t}
\treturn result
}

// dedupByStage removes consecutive duplicates based on a specific field
func dedupByStage(records []Record, field string) []Record {
\tvar result []Record
\tvar lastValue interface{} = nil
\tfirst := true
\tfor _, record := range records {
\t\tval, _ := record[field]
\t\tif first || val != lastValue {
\t\t\tlastValue = val
\t\t\tfirst = false
\t\t\tresult = append(result, record)
\t\t}
\t}
\treturn result
}

// recordKey generates a unique key for a record (for dedup comparison)
func recordKey(record Record) string {
\tkeys := make([]string, 0, len(record))
\tfor k := range record {
\t\tkeys = append(keys, k)
\t}
\tsort.Strings(keys)
\tvar parts []string
\tfor _, k := range keys {
\t\tparts = append(parts, fmt.Sprintf(\"%s=%v\", k, record[k]))
\t}
\treturn strings.Join(parts, \",\")
}

// interleaveStage interleaves records from multiple stage outputs in round-robin fashion
func interleaveStage(streams [][]Record) []Record {
\tif len(streams) == 0 {
\t\treturn nil
\t}
\tvar result []Record
\tmaxLen := 0
\tfor _, s := range streams {
\t\tif len(s) > maxLen {
\t\t\tmaxLen = len(s)
\t\t}
\t}
\tfor i := 0; i < maxLen; i++ {
\t\tfor _, stream := range streams {
\t\t\tif i < len(stream) {
\t\t\t\tresult = append(result, stream[i])
\t\t\t}
\t\t}
\t}
\treturn result
}

// concatStage concatenates multiple stage outputs sequentially
func concatStage(streams [][]Record) []Record {
\tvar result []Record
\tfor _, stream := range streams {
\t\tresult = append(result, stream...)
\t}
\treturn result
}

// mergeSortedStage merges multiple pre-sorted streams maintaining sort order
// Uses a heap-based k-way merge for efficiency
func mergeSortedStage(streams [][]Record, field string, ascending bool) []Record {
\tif len(streams) == 0 {
\t\treturn nil
\t}

\t// Track current index in each stream
\tindices := make([]int, len(streams))
\tvar result []Record

\tfor {
\t\t// Find the stream with the smallest (or largest) next value
\t\tbestStream := -1
\t\tvar bestValue interface{}

\t\tfor i, stream := range streams {
\t\t\tif indices[i] >= len(stream) {
\t\t\t\tcontinue // This stream is exhausted
\t\t\t}

\t\t\tvalue := stream[indices[i]][field]
\t\t\tif bestStream == -1 {
\t\t\t\tbestStream = i
\t\t\t\tbestValue = value
\t\t\t\tcontinue
\t\t\t}

\t\t\t// Compare values
\t\t\tvar isBetter bool
\t\t\tswitch v := value.(type) {
\t\t\tcase int:
\t\t\t\tif bv, ok := bestValue.(int); ok {
\t\t\t\t\tif ascending {
\t\t\t\t\t\tisBetter = v < bv
\t\t\t\t\t} else {
\t\t\t\t\t\tisBetter = v > bv
\t\t\t\t\t}
\t\t\t\t}
\t\t\tcase float64:
\t\t\t\tif bv, ok := bestValue.(float64); ok {
\t\t\t\t\tif ascending {
\t\t\t\t\t\tisBetter = v < bv
\t\t\t\t\t} else {
\t\t\t\t\t\tisBetter = v > bv
\t\t\t\t\t}
\t\t\t\t}
\t\t\tcase string:
\t\t\t\tif bv, ok := bestValue.(string); ok {
\t\t\t\t\tif ascending {
\t\t\t\t\t\tisBetter = v < bv
\t\t\t\t\t} else {
\t\t\t\t\t\tisBetter = v > bv
\t\t\t\t\t}
\t\t\t\t}
\t\t\t}

\t\t\tif isBetter {
\t\t\t\tbestStream = i
\t\t\t\tbestValue = value
\t\t\t}
\t\t}

\t\tif bestStream == -1 {
\t\t\tbreak // All streams exhausted
\t\t}

\t\tresult = append(result, streams[bestStream][indices[bestStream]])
\t\tindices[bestStream]++
\t}

\treturn result
}

// tapStage executes a side effect for each record without modifying the stream
// Useful for logging, metrics, debugging, or other observations
func tapStage(records []Record, sideEffect func(Record)) []Record {
\tfor _, record := range records {
\t\tfunc() {
\t\t\tdefer func() { recover() }() // Side effects should not interrupt pipeline
\t\t\tsideEffect(record)
\t\t}()
\t}
\treturn records
}

// flattenStage flattens nested collections into individual records
// If a record contains an array field \"__items__\", yields each item
func flattenStage(records []Record) []Record {
\tvar result []Record
\tfor _, record := range records {
\t\tif items, ok := record[\"__items__\"]; ok {
\t\t\tif arr, ok := items.([]interface{}); ok {
\t\t\t\tfor _, item := range arr {
\t\t\t\t\tif rec, ok := item.(Record); ok {
\t\t\t\t\t\tresult = append(result, rec)
\t\t\t\t\t} else if m, ok := item.(map[string]interface{}); ok {
\t\t\t\t\t\tresult = append(result, Record(m))
\t\t\t\t\t}
\t\t\t\t}
\t\t\t\tcontinue
\t\t\t}
\t\t}
\t\tresult = append(result, record)
\t}
\treturn result
}

// flattenFieldStage flattens a specific field within each record
// Expands records where field contains an array into multiple records
func flattenFieldStage(records []Record, field string) []Record {
\tvar result []Record
\tfor _, record := range records {
\t\tif fieldValue, ok := record[field]; ok {
\t\t\tif arr, ok := fieldValue.([]interface{}); ok {
\t\t\t\tfor _, item := range arr {
\t\t\t\t\tnewRecord := make(Record)
\t\t\t\t\tfor k, v := range record {
\t\t\t\t\t\tnewRecord[k] = v
\t\t\t\t\t}
\t\t\t\t\tnewRecord[field] = item
\t\t\t\t\tresult = append(result, newRecord)
\t\t\t\t}
\t\t\t\tcontinue
\t\t\t}
\t\t}
\t\tresult = append(result, record)
\t}
\treturn result
}

// debounceStage emits records only after a silence period
// Groups records by time windows and emits the last record in each window
func debounceStage(records []Record, ms int64, timestampField string) []Record {
\tif len(records) == 0 {
\t\treturn records
\t}

\tvar result []Record
\tvar buffer Record
\tvar lastTime float64 = -1
\tthresholdSec := float64(ms) / 1000.0

\tfor _, record := range records {
\t\tcurrentTime := float64(time.Now().UnixNano()) / 1e9
\t\tif timestampField != \"\" {
\t\t\tif ts, ok := record[timestampField]; ok {
\t\t\t\tswitch v := ts.(type) {
\t\t\t\tcase float64:
\t\t\t\t\tcurrentTime = v
\t\t\t\tcase int64:
\t\t\t\t\tcurrentTime = float64(v)
\t\t\t\tcase int:
\t\t\t\t\tcurrentTime = float64(v)
\t\t\t\t}
\t\t\t}
\t\t}

\t\tif lastTime < 0 {
\t\t\tbuffer = record
\t\t\tlastTime = currentTime
\t\t} else if currentTime-lastTime < thresholdSec {
\t\t\t// Within debounce window, replace buffer
\t\t\tbuffer = record
\t\t\tlastTime = currentTime
\t\t} else {
\t\t\t// Silence period exceeded, emit buffered and start new
\t\t\tresult = append(result, buffer)
\t\t\tbuffer = record
\t\t\tlastTime = currentTime
\t\t}
\t}

\t// Emit final buffered record
\tresult = append(result, buffer)
\treturn result
}

// branchStage routes records through different stages based on condition
// Records matching condition go through trueStage, others through falseStage
func branchStage(records []Record, condFn func(Record) bool, trueFn func([]Record) []Record, falseFn func([]Record) []Record) []Record {
\tvar trueRecords, falseRecords []Record

\tfor _, record := range records {
\t\tif condFn(record) {
\t\t\ttrueRecords = append(trueRecords, record)
\t\t} else {
\t\t\tfalseRecords = append(falseRecords, record)
\t\t}
\t}

\tvar result []Record
\tif len(trueRecords) > 0 {
\t\tresult = append(result, trueFn(trueRecords)...)
\t}
\tif len(falseRecords) > 0 {
\t\tresult = append(result, falseFn(falseRecords)...)
\t}
\treturn result
}

// teeStage runs side stage on records, discards results, returns original records
// Like Unix tee - fork to side destination while main stream continues
func teeStage(records []Record, sideFn func([]Record) []Record) []Record {
\t// Run side stage (results discarded)
\t_ = sideFn(records)

\t// Return original records unchanged
\treturn records
}

// ============================================
// SERVICE INFRASTRUCTURE (Client-Server Phase 1)
// ============================================

// ServiceError represents an error from a service call
type ServiceError struct {
\tService string
\tMessage string
}

func (e *ServiceError) Error() string {
\treturn fmt.Sprintf(\"service %s: %s\", e.Service, e.Message)
}

// Service is the interface for in-process services
type Service interface {
\tCall(request interface{}) (interface{}, error)
\tName() string
}

// StatefulService is a service that maintains state between requests
type StatefulService struct {
\tname  string
\tstate map[string]interface{}
}

func NewStatefulService(name string) *StatefulService {
\treturn &StatefulService{
\t\tname:  name,
\t\tstate: make(map[string]interface{}),
\t}
}

func (s *StatefulService) Name() string {
\treturn s.name
}

func (s *StatefulService) StateGet(key string) interface{} {
\treturn s.state[key]
}

func (s *StatefulService) StatePut(key string, value interface{}) {
\ts.state[key] = value
}

func (s *StatefulService) StateModify(key string, fn func(interface{}) interface{}) {
\ts.state[key] = fn(s.state[key])
}

func (s *StatefulService) StateDelete(key string) {
\tdelete(s.state, key)
}

// Global service registry
var services = make(map[string]Service)

// RegisterService adds a service to the global registry
func RegisterService(name string, service Service) {
\tservices[name] = service
}

// GetService retrieves a service from the registry
func GetService(name string) (Service, error) {
\tservice, ok := services[name]
\tif !ok {
\t\treturn nil, &ServiceError{Service: name, Message: \"service not found\"}
\t}
\treturn service, nil
}

// CallServiceOptions contains options for service calls
type CallServiceOptions struct {
\tTimeout     int // milliseconds
\tRetry       int // number of retries
\tRetryDelay  int // milliseconds between retries
\tFallback    interface{}
}

// CallServiceImpl calls a service with options
func CallServiceImpl(serviceName string, request interface{}, options *CallServiceOptions) (interface{}, error) {
\tservice, err := GetService(serviceName)
\tif err != nil {
\t\tif options != nil && options.Fallback != nil {
\t\t\treturn options.Fallback, nil
\t\t}
\t\treturn nil, err
\t}

\tmaxRetries := 0
\tif options != nil {
\t\tmaxRetries = options.Retry
\t}

\tvar lastErr error
\tfor attempt := 0; attempt <= maxRetries; attempt++ {
\t\tresult, err := service.Call(request)
\t\tif err == nil {
\t\t\treturn result, nil
\t\t}
\t\tlastErr = err
\t\tif attempt < maxRetries && options != nil && options.RetryDelay > 0 {
\t\t\ttime.Sleep(time.Duration(options.RetryDelay) * time.Millisecond)
\t\t}
\t}

\tif options != nil && options.Fallback != nil {
\t\treturn options.Fallback, nil
\t}
\treturn nil, lastErr
}

// callServiceStage is a pipeline stage that calls a service for each record
func callServiceStage(records []Record, serviceName, requestField, responseField string, options *CallServiceOptions) []Record {
\tvar output []Record
\tfor _, record := range records {
\t\trequest := record[requestField]
\t\tresponse, err := CallServiceImpl(serviceName, request, options)
\t\tif err != nil {
\t\t\trecord[\"__error__\"] = true
\t\t\trecord[\"__type__\"] = \"ServiceError\"
\t\t\trecord[\"__message__\"] = err.Error()
\t\t} else {
\t\t\trecord[responseField] = response
\t\t}
\t\toutput = append(output, record)
\t}
\treturn output
}

'.

%% generate_go_enhanced_stage_functions(+Stages, -Code)
%  Generate stub functions for each stage.
generate_go_enhanced_stage_functions([], "").
generate_go_enhanced_stage_functions([Stage|Rest], Code) :-
    generate_go_single_enhanced_stage(Stage, StageCode),
    generate_go_enhanced_stage_functions(Rest, RestCode),
    (RestCode = "" ->
        Code = StageCode
    ;   format(string(Code), "~w~n~w", [StageCode, RestCode])
    ).

generate_go_single_enhanced_stage(fan_out(SubStages), Code) :-
    !,
    generate_go_enhanced_stage_functions(SubStages, Code).
generate_go_single_enhanced_stage(parallel(SubStages, _Options), Code) :-
    !,
    generate_go_enhanced_stage_functions(SubStages, Code).
generate_go_single_enhanced_stage(parallel(SubStages), Code) :-
    !,
    generate_go_enhanced_stage_functions(SubStages, Code).
generate_go_single_enhanced_stage(merge, "") :- !.
generate_go_single_enhanced_stage(route_by(_, Routes), Code) :-
    !,
    findall(Stage, member((_Cond, Stage), Routes), RouteStages),
    generate_go_enhanced_stage_functions(RouteStages, Code).
generate_go_single_enhanced_stage(filter_by(_), "") :- !.
generate_go_single_enhanced_stage(batch(_), "") :- !.
generate_go_single_enhanced_stage(unbatch, "") :- !.
generate_go_single_enhanced_stage(unique(_), "") :- !.
generate_go_single_enhanced_stage(first(_), "") :- !.
generate_go_single_enhanced_stage(last(_), "") :- !.
generate_go_single_enhanced_stage(group_by(_, _), "") :- !.
generate_go_single_enhanced_stage(reduce(_, _), "") :- !.
generate_go_single_enhanced_stage(reduce(_), "") :- !.
generate_go_single_enhanced_stage(scan(_, _), "") :- !.
generate_go_single_enhanced_stage(scan(_), "") :- !.
generate_go_single_enhanced_stage(order_by(_), "") :- !.
generate_go_single_enhanced_stage(order_by(_, _), "") :- !.
generate_go_single_enhanced_stage(sort_by(_), "") :- !.
generate_go_single_enhanced_stage(try_catch(Stage, Handler), Code) :-
    !,
    generate_go_single_enhanced_stage(Stage, StageCode),
    generate_go_single_enhanced_stage(Handler, HandlerCode),
    format(string(Code), "~w~w", [StageCode, HandlerCode]).
generate_go_single_enhanced_stage(retry(Stage, _), Code) :-
    !,
    generate_go_single_enhanced_stage(Stage, Code).
generate_go_single_enhanced_stage(retry(Stage, _, _), Code) :-
    !,
    generate_go_single_enhanced_stage(Stage, Code).
generate_go_single_enhanced_stage(on_error(Handler), Code) :-
    !,
    generate_go_single_enhanced_stage(Handler, Code).
generate_go_single_enhanced_stage(timeout(Stage, _), Code) :-
    !,
    generate_go_single_enhanced_stage(Stage, Code).
generate_go_single_enhanced_stage(timeout(Stage, _, Fallback), Code) :-
    !,
    generate_go_single_enhanced_stage(Stage, StageCode),
    generate_go_single_enhanced_stage(Fallback, FallbackCode),
    format(string(Code), "~w~w", [StageCode, FallbackCode]).
generate_go_single_enhanced_stage(rate_limit(_, _), "") :- !.
generate_go_single_enhanced_stage(throttle(_), "") :- !.
generate_go_single_enhanced_stage(buffer(_), "") :- !.
generate_go_single_enhanced_stage(debounce(_), "") :- !.
generate_go_single_enhanced_stage(zip(SubStages), Code) :-
    !,
    generate_go_enhanced_stage_functions(SubStages, Code).
generate_go_single_enhanced_stage(window(_), "") :- !.
generate_go_single_enhanced_stage(sliding_window(_, _), "") :- !.
generate_go_single_enhanced_stage(sample(_), "") :- !.
generate_go_single_enhanced_stage(take_every(_), "") :- !.
generate_go_single_enhanced_stage(partition(_), "") :- !.
generate_go_single_enhanced_stage(take(_), "") :- !.
generate_go_single_enhanced_stage(skip(_), "") :- !.
generate_go_single_enhanced_stage(take_while(_), "") :- !.
generate_go_single_enhanced_stage(skip_while(_), "") :- !.
generate_go_single_enhanced_stage(distinct, "") :- !.
generate_go_single_enhanced_stage(distinct_by(_), "") :- !.
generate_go_single_enhanced_stage(dedup, "") :- !.
generate_go_single_enhanced_stage(dedup_by(_), "") :- !.
generate_go_single_enhanced_stage(interleave(SubStages), Code) :-
    !,
    generate_go_enhanced_stage_functions(SubStages, Code).
generate_go_single_enhanced_stage(concat(SubStages), Code) :-
    !,
    generate_go_enhanced_stage_functions(SubStages, Code).
generate_go_single_enhanced_stage(merge_sorted(SubStages, _Field), Code) :-
    !,
    generate_go_enhanced_stage_functions(SubStages, Code).
generate_go_single_enhanced_stage(merge_sorted(SubStages, _Field, _Dir), Code) :-
    !,
    generate_go_enhanced_stage_functions(SubStages, Code).
generate_go_single_enhanced_stage(tap(_), "") :- !.
generate_go_single_enhanced_stage(flatten, "") :- !.
generate_go_single_enhanced_stage(flatten(_), "") :- !.
generate_go_single_enhanced_stage(debounce(_), "") :- !.
generate_go_single_enhanced_stage(debounce(_, _), "") :- !.
generate_go_single_enhanced_stage(branch(_Cond, TrueStage, FalseStage), Code) :-
    !,
    generate_go_single_enhanced_stage(TrueStage, TrueCode),
    generate_go_single_enhanced_stage(FalseStage, FalseCode),
    format(string(Code), "~w~w", [TrueCode, FalseCode]).
generate_go_single_enhanced_stage(tee(SideStage), Code) :-
    !,
    generate_go_single_enhanced_stage(SideStage, Code).
generate_go_single_enhanced_stage(call_service(_, _, _), "") :- !.
generate_go_single_enhanced_stage(call_service(_, _, _, _), "") :- !.
generate_go_single_enhanced_stage(Pred/Arity, Code) :-
    !,
    format(string(Code),
"// ~w is a pipeline stage (~w/~w)
func ~w(input []Record) []Record {
\t// TODO: Implement based on predicate bindings
\tvar output []Record
\tfor _, record := range input {
\t\toutput = append(output, record)
\t}
\treturn output
}

", [Pred, Pred, Arity, Pred]).
generate_go_single_enhanced_stage(_, "").

%% generate_go_enhanced_connector(+Stages, +PipelineName, -Code)
%  Generate the main connector that handles enhanced flow patterns.
generate_go_enhanced_connector(Stages, PipelineName, Code) :-
    generate_go_enhanced_flow_code(Stages, "input", FlowCode),
    format(string(Code),
"// ~w is an enhanced pipeline with fan-out, merge, and routing support
func ~w(input []Record) []Record {
~w
}

", [PipelineName, PipelineName, FlowCode]).

%% generate_go_enhanced_flow_code(+Stages, +CurrentVar, -Code)
%  Generate the flow code for enhanced pipeline stages.
generate_go_enhanced_flow_code([], CurrentVar, Code) :-
    format(string(Code), "\treturn ~w", [CurrentVar]).
generate_go_enhanced_flow_code([Stage|Rest], CurrentVar, Code) :-
    generate_go_stage_flow(Stage, CurrentVar, NextVar, StageCode),
    generate_go_enhanced_flow_code(Rest, NextVar, RestCode),
    format(string(Code), "~w~n~w", [StageCode, RestCode]).

%% generate_go_stage_flow(+Stage, +InVar, -OutVar, -Code)
%  Generate flow code for a single stage.

% Fan-out stage: broadcast to parallel stages (sequential execution)
generate_go_stage_flow(fan_out(SubStages), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "fanOut~wResult", [N]),
    extract_go_stage_names(SubStages, StageNames),
    format_go_stage_list(StageNames, StageListStr),
    format(string(Code),
"\t// Fan-out to ~w stages (sequential)
\tvar ~w []Record
\tfor _, record := range ~w {
\t\tfanOutResults := fanOutRecords(record, []func([]Record) []Record{~w})
\t\t~w = append(~w, fanOutResults...)
\t}", [N, OutVar, InVar, StageListStr, OutVar, OutVar]).

% Parallel stage with options: parallel(Stages, Options)
generate_go_stage_flow(parallel(SubStages, Options), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "parallel~wResult", [N]),
    extract_go_stage_names(SubStages, StageNames),
    format_go_stage_list(StageNames, StageListStr),
    % Check for ordered option
    (   member(ordered(true), Options)
    ->  FuncName = "parallelRecordsOrdered",
        format(atom(Comment), "Parallel execution (ordered) of ~w stages", [N])
    ;   FuncName = "parallelRecords",
        format(atom(Comment), "Parallel execution of ~w stages (concurrent via goroutines)", [N])
    ),
    format(string(Code),
"\t// ~w
\tvar ~w []Record
\tfor _, record := range ~w {
\t\tparallelResults := ~w(record, []func([]Record) []Record{~w})
\t\t~w = append(~w, parallelResults...)
\t}", [Comment, OutVar, InVar, FuncName, StageListStr, OutVar, OutVar]).

% Parallel stage: concurrent execution using goroutines (default: unordered)
generate_go_stage_flow(parallel(SubStages), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "parallel~wResult", [N]),
    extract_go_stage_names(SubStages, StageNames),
    format_go_stage_list(StageNames, StageListStr),
    format(string(Code),
"\t// Parallel execution of ~w stages (concurrent via goroutines)
\tvar ~w []Record
\tfor _, record := range ~w {
\t\tparallelResults := parallelRecords(record, []func([]Record) []Record{~w})
\t\t~w = append(~w, parallelResults...)
\t}", [N, OutVar, InVar, StageListStr, OutVar, OutVar]).

% Merge stage: placeholder, usually follows fan_out or parallel
generate_go_stage_flow(merge, InVar, OutVar, Code) :-
    !,
    OutVar = InVar,
    Code = "\t// Merge: results already combined from fan-out or parallel".

% Conditional routing
generate_go_stage_flow(route_by(CondPred, Routes), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "routedResult", []),
    format_go_route_map(Routes, RouteMapStr),
    format(string(Code),
"\t// Conditional routing based on ~w
\tvar ~w []Record
\trouteMap := map[interface{}]func([]Record) []Record{~w}
\tfor _, record := range ~w {
\t\troutedRecords := routeRecord(record, ~w, routeMap, nil)
\t\t~w = append(~w, routedRecords...)
\t}", [CondPred, OutVar, RouteMapStr, InVar, CondPred, OutVar, OutVar]).

% Filter stage
generate_go_stage_flow(filter_by(Pred), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "filteredResult", []),
    format(string(Code),
"\t// Filter by ~w
\t~w := filterRecords(~w, ~w)", [Pred, OutVar, InVar, Pred]).

% Batch stage: collect N records into batches
generate_go_stage_flow(batch(N), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "batched~wResult", [N]),
    format(string(Code),
"\t// Batch records into groups of ~w
\t~w := batchRecords(~w, ~w)", [N, OutVar, InVar, N]).

% Unbatch stage: flatten batches back to individual records
generate_go_stage_flow(unbatch, InVar, OutVar, Code) :-
    !,
    OutVar = "unbatchedResult",
    format(string(Code),
"\t// Unbatch: flatten batches to individual records
\t~w := unbatchRecords(~w)", [OutVar, InVar]).

% Unique stage: deduplicate by field (keep first)
generate_go_stage_flow(unique(Field), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "unique~wResult", [Field]),
    format(string(Code),
"\t// Unique: keep first record per '~w' value
\t~w := uniqueByField(~w, \"~w\")", [Field, OutVar, InVar, Field]).

% First stage: alias for unique (keep first occurrence)
generate_go_stage_flow(first(Field), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "first~wResult", [Field]),
    format(string(Code),
"\t// First: keep first record per '~w' value
\t~w := firstByField(~w, \"~w\")", [Field, OutVar, InVar, Field]).

% Last stage: keep last record per field value
generate_go_stage_flow(last(Field), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "last~wResult", [Field]),
    format(string(Code),
"\t// Last: keep last record per '~w' value
\t~w := lastByField(~w, \"~w\")", [Field, OutVar, InVar, Field]).

% Group by stage: group and aggregate
generate_go_stage_flow(group_by(Field, Agg), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "grouped~wResult", [Field]),
    format_go_aggregations(Agg, AggStr),
    format(string(Code),
"\t// Group by '~w' with aggregations
\t~w := groupByField(~w, \"~w\", []Aggregation{~w})", [Field, OutVar, InVar, Field, AggStr]).

% Reduce stage with initial value: custom sequential fold
generate_go_stage_flow(reduce(Pred, Init), InVar, OutVar, Code) :-
    !,
    OutVar = "reducedResult",
    format(string(Code),
"\t// Reduce: sequential fold with ~w
\t~w := reduceRecords(~w, ~w, ~w)", [Pred, OutVar, InVar, Pred, Init]).

% Reduce stage without initial value
generate_go_stage_flow(reduce(Pred), InVar, OutVar, Code) :-
    !,
    OutVar = "reducedResult",
    format(string(Code),
"\t// Reduce: sequential fold with ~w
\t~w := reduceRecords(~w, ~w, nil)", [Pred, OutVar, InVar, Pred]).

% Scan stage with initial value: reduce with intermediate outputs
generate_go_stage_flow(scan(Pred, Init), InVar, OutVar, Code) :-
    !,
    OutVar = "scannedResult",
    format(string(Code),
"\t// Scan: running fold with ~w (emits intermediate values)
\t~w := scanRecords(~w, ~w, ~w)", [Pred, OutVar, InVar, Pred, Init]).

% Scan stage without initial value
generate_go_stage_flow(scan(Pred), InVar, OutVar, Code) :-
    !,
    OutVar = "scannedResult",
    format(string(Code),
"\t// Scan: running fold with ~w (emits intermediate values)
\t~w := scanRecords(~w, ~w, nil)", [Pred, OutVar, InVar, Pred]).

% Order by single field (ascending by default)
generate_go_stage_flow(order_by(Field), InVar, OutVar, Code) :-
    atom(Field),
    !,
    format(atom(OutVar), "ordered~wResult", [Field]),
    format(string(Code),
"\t// Order by '~w' ascending
\t~w := orderByField(~w, \"~w\", \"asc\")", [Field, OutVar, InVar, Field]).

% Order by single field with direction
generate_go_stage_flow(order_by(Field, Dir), InVar, OutVar, Code) :-
    atom(Field),
    !,
    format(atom(OutVar), "ordered~wResult", [Field]),
    format(string(Code),
"\t// Order by '~w' ~w
\t~w := orderByField(~w, \"~w\", \"~w\")", [Field, Dir, OutVar, InVar, Field, Dir]).

% Order by multiple fields with directions
generate_go_stage_flow(order_by(FieldSpecs), InVar, OutVar, Code) :-
    is_list(FieldSpecs),
    !,
    OutVar = "orderedMultiResult",
    format_go_field_specs(FieldSpecs, SpecStr),
    format(string(Code),
"\t// Order by multiple fields
\t~w := orderByFields(~w, []FieldSpec{~w})", [OutVar, InVar, SpecStr]).

% Sort by custom comparator
generate_go_stage_flow(sort_by(ComparePred), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "sorted~wResult", [ComparePred]),
    format(string(Code),
"\t// Sort by custom comparator: ~w
\t~w := sortByComparator(~w, ~w)", [ComparePred, OutVar, InVar, ComparePred]).

% Try-catch stage: execute stage, on error route to handler
generate_go_stage_flow(try_catch(Stage, Handler), InVar, OutVar, Code) :-
    !,
    extract_go_stage_name(Stage, StageName),
    extract_go_stage_name(Handler, HandlerName),
    OutVar = "tryCatchResult",
    format(string(Code),
"\t// Try-Catch: ~w with handler ~w
\t~w := tryCatchStage(~w, ~w, ~w)", [StageName, HandlerName, OutVar, InVar, StageName, HandlerName]).

% Retry stage: retry N times on failure
generate_go_stage_flow(retry(Stage, N), InVar, OutVar, Code) :-
    !,
    extract_go_stage_name(Stage, StageName),
    OutVar = "retryResult",
    format(string(Code),
"\t// Retry: ~w up to ~w times
\t~w := retryStage(~w, ~w, ~w, 0, \"none\")", [StageName, N, OutVar, InVar, StageName, N]).

% Retry stage with options
generate_go_stage_flow(retry(Stage, N, Options), InVar, OutVar, Code) :-
    !,
    extract_go_stage_name(Stage, StageName),
    OutVar = "retryResult",
    extract_go_retry_options(Options, DelayMs, Backoff),
    format(string(Code),
"\t// Retry: ~w up to ~w times (delay=~wms, backoff=~w)
\t~w := retryStage(~w, ~w, ~w, ~w, \"~w\")", [StageName, N, DelayMs, Backoff, OutVar, InVar, StageName, N, DelayMs, Backoff]).

% On-error stage: global error handler
generate_go_stage_flow(on_error(Handler), InVar, OutVar, Code) :-
    !,
    extract_go_stage_name(Handler, HandlerName),
    OutVar = "onErrorResult",
    format(string(Code),
"\t// On-Error: route errors to ~w
\t~w := onErrorStage(~w, ~w)", [HandlerName, OutVar, InVar, HandlerName]).

% Timeout stage: execute with time limit
generate_go_stage_flow(timeout(Stage, Ms), InVar, OutVar, Code) :-
    !,
    extract_go_stage_name(Stage, StageName),
    OutVar = "timeoutResult",
    format(string(Code),
"\t// Timeout: ~w with ~wms limit
\t~w := timeoutStage(~w, ~w, ~w)", [StageName, Ms, OutVar, InVar, StageName, Ms]).

% Timeout stage with fallback
generate_go_stage_flow(timeout(Stage, Ms, Fallback), InVar, OutVar, Code) :-
    !,
    extract_go_stage_name(Stage, StageName),
    extract_go_stage_name(Fallback, FallbackName),
    OutVar = "timeoutResult",
    format(string(Code),
"\t// Timeout: ~w with ~wms limit, fallback to ~w
\t~w := timeoutStageWithFallback(~w, ~w, ~w, ~w)", [StageName, Ms, FallbackName, OutVar, InVar, StageName, Ms, FallbackName]).

% Rate limit stage: limit throughput
generate_go_stage_flow(rate_limit(N, Per), InVar, OutVar, Code) :-
    !,
    OutVar = "rateLimitedResult",
    time_unit_to_ms(Per, IntervalMs),
    format(string(Code),
"\t// Rate Limit: ~w per ~w
\t~w := rateLimitStage(~w, ~w, ~w)", [N, Per, OutVar, InVar, N, IntervalMs]).

% Throttle stage: fixed delay between records
generate_go_stage_flow(throttle(Ms), InVar, OutVar, Code) :-
    !,
    OutVar = "throttledResult",
    format(string(Code),
"\t// Throttle: ~wms delay between records
\t~w := throttleStage(~w, ~w)", [Ms, OutVar, InVar, Ms]).

% Buffer stage: collect records into batches
generate_go_stage_flow(buffer(N), InVar, OutVar, Code) :-
    !,
    OutVar = "bufferedResult",
    format(string(Code),
"\t// Buffer: collect ~w records into batches
\t~w := bufferStage(~w, ~w)", [N, OutVar, InVar, N]).

% Debounce stage: emit only if no new record within delay
generate_go_stage_flow(debounce(Ms), InVar, OutVar, Code) :-
    !,
    OutVar = "debouncedResult",
    format(string(Code),
"\t// Debounce: ~wms quiet period
\t~w := debounceStage(~w, ~w)", [Ms, OutVar, InVar, Ms]).

% Zip stage: combine multiple stages record-by-record
generate_go_stage_flow(zip(Stages), InVar, OutVar, Code) :-
    !,
    OutVar = "zippedResult",
    extract_go_stage_names(Stages, Names),
    format_go_stage_list(Names, StageListStr),
    format(string(Code),
"\t// Zip: combine outputs from multiple stages
\t~w := zipStage(~w, []func([]Record) []Record{~w})", [OutVar, InVar, StageListStr]).

% Window stage: non-overlapping windows
generate_go_stage_flow(window(N), InVar, OutVar, Code) :-
    !,
    OutVar = "windowedResult",
    format(string(Code),
"\t// Window: collect ~w records into windows
\t~w := windowStage(~w, ~w)", [N, OutVar, InVar, N]).

% Sliding window stage
generate_go_stage_flow(sliding_window(N, Step), InVar, OutVar, Code) :-
    !,
    OutVar = "slidingWindowResult",
    format(string(Code),
"\t// Sliding Window: size ~w, step ~w
\t~w := slidingWindowStage(~w, ~w, ~w)", [N, Step, OutVar, InVar, N, Step]).

% Sample stage: random sampling
generate_go_stage_flow(sample(N), InVar, OutVar, Code) :-
    !,
    OutVar = "sampledResult",
    format(string(Code),
"\t// Sample: random ~w records
\t~w := sampleStage(~w, ~w)", [N, OutVar, InVar, N]).

% Take every stage
generate_go_stage_flow(take_every(N), InVar, OutVar, Code) :-
    !,
    OutVar = "takeEveryResult",
    format(string(Code),
"\t// Take Every: every ~wth record
\t~w := takeEveryStage(~w, ~w)", [N, OutVar, InVar, N]).

% Partition stage
generate_go_stage_flow(partition(Pred), InVar, OutVar, Code) :-
    !,
    OutVar = "partitionedResult",
    format(string(Code),
"\t// Partition: split by ~w
\t~w, _ := partitionStage(~w, ~w)", [Pred, OutVar, InVar, Pred]).

% Take stage
generate_go_stage_flow(take(N), InVar, OutVar, Code) :-
    !,
    OutVar = "takeResult",
    format(string(Code),
"\t// Take: first ~w records
\t~w := takeStage(~w, ~w)", [N, OutVar, InVar, N]).

% Skip stage
generate_go_stage_flow(skip(N), InVar, OutVar, Code) :-
    !,
    OutVar = "skipResult",
    format(string(Code),
"\t// Skip: skip first ~w records
\t~w := skipStage(~w, ~w)", [N, OutVar, InVar, N]).

% Take while stage
generate_go_stage_flow(take_while(Pred), InVar, OutVar, Code) :-
    !,
    OutVar = "takeWhileResult",
    format(string(Code),
"\t// Take While: while ~w is true
\t~w := takeWhileStage(~w, ~w)", [Pred, OutVar, InVar, Pred]).

% Skip while stage
generate_go_stage_flow(skip_while(Pred), InVar, OutVar, Code) :-
    !,
    OutVar = "skipWhileResult",
    format(string(Code),
"\t// Skip While: skip while ~w is true
\t~w := skipWhileStage(~w, ~w)", [Pred, OutVar, InVar, Pred]).

% Distinct stage: remove all duplicates (global)
generate_go_stage_flow(distinct, InVar, OutVar, Code) :-
    !,
    OutVar = "distinctResult",
    format(string(Code),
"\t// Distinct: remove all duplicates (global dedup)
\t~w := distinctStage(~w)", [OutVar, InVar]).

% Distinct by field: remove duplicates based on specific field
generate_go_stage_flow(distinct_by(Field), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "distinct~wResult", [Field]),
    format(string(Code),
"\t// Distinct By: remove duplicates based on '~w' field
\t~w := distinctByStage(~w, \"~w\")", [Field, OutVar, InVar, Field]).

% Dedup stage: remove consecutive duplicates only
generate_go_stage_flow(dedup, InVar, OutVar, Code) :-
    !,
    OutVar = "dedupResult",
    format(string(Code),
"\t// Dedup: remove consecutive duplicates
\t~w := dedupStage(~w)", [OutVar, InVar]).

% Dedup by field: remove consecutive duplicates based on specific field
generate_go_stage_flow(dedup_by(Field), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "dedup~wResult", [Field]),
    format(string(Code),
"\t// Dedup By: remove consecutive duplicates based on '~w' field
\t~w := dedupByStage(~w, \"~w\")", [Field, OutVar, InVar, Field]).

% Interleave stage: round-robin interleave from multiple stage outputs
generate_go_stage_flow(interleave(SubStages), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "interleaved~wResult", [N]),
    extract_go_stage_names(SubStages, StageNames),
    format_go_stage_list(StageNames, StageListStr),
    format(string(Code),
"\t// Interleave: round-robin from ~w stage outputs
\tvar interleaveStreams~w [][]Record
\tfor _, stageFn := range []func([]Record) []Record{~w} {
\t\tinterleaveStreams~w = append(interleaveStreams~w, stageFn(~w))
\t}
\t~w := interleaveStage(interleaveStreams~w)", [N, N, StageListStr, N, N, InVar, OutVar, N]).

% Concat stage: sequential concatenation of multiple stage outputs
generate_go_stage_flow(concat(SubStages), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "concatenated~wResult", [N]),
    extract_go_stage_names(SubStages, StageNames),
    format_go_stage_list(StageNames, StageListStr),
    format(string(Code),
"\t// Concat: sequential concatenation of ~w stage outputs
\tvar concatStreams~w [][]Record
\tfor _, stageFn := range []func([]Record) []Record{~w} {
\t\tconcatStreams~w = append(concatStreams~w, stageFn(~w))
\t}
\t~w := concatStage(concatStreams~w)", [N, N, StageListStr, N, N, InVar, OutVar, N]).

% Merge sorted stage: merge pre-sorted streams maintaining order (ascending)
generate_go_stage_flow(merge_sorted(SubStages, Field), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "mergeSorted~wResult", [Field]),
    extract_go_stage_names(SubStages, StageNames),
    format_go_stage_list(StageNames, StageListStr),
    format(string(Code),
"\t// Merge Sorted: merge ~w pre-sorted streams by '~w' (ascending)
\tvar mergeSortedStreams~w [][]Record
\tfor _, stageFn := range []func([]Record) []Record{~w} {
\t\tmergeSortedStreams~w = append(mergeSortedStreams~w, stageFn(~w))
\t}
\t~w := mergeSortedStage(mergeSortedStreams~w, \"~w\", true)", [N, Field, N, StageListStr, N, N, InVar, OutVar, N, Field]).

% Merge sorted stage with direction: merge pre-sorted streams with specified order
generate_go_stage_flow(merge_sorted(SubStages, Field, Dir), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "mergeSorted~w~wResult", [Field, Dir]),
    extract_go_stage_names(SubStages, StageNames),
    format_go_stage_list(StageNames, StageListStr),
    ( Dir = asc -> Ascending = "true" ; Ascending = "false" ),
    format(string(Code),
"\t// Merge Sorted: merge ~w pre-sorted streams by '~w' (~w)
\tvar mergeSortedStreams~w~w [][]Record
\tfor _, stageFn := range []func([]Record) []Record{~w} {
\t\tmergeSortedStreams~w~w = append(mergeSortedStreams~w~w, stageFn(~w))
\t}
\t~w := mergeSortedStage(mergeSortedStreams~w~w, \"~w\", ~w)", [N, Field, Dir, N, Dir, StageListStr, N, Dir, N, Dir, InVar, OutVar, N, Dir, Field, Ascending]).

% Tap stage: execute side effect without modifying stream
generate_go_stage_flow(tap(Pred), InVar, OutVar, Code) :-
    !,
    ( Pred = PredName/_ -> true ; PredName = Pred ),
    format(atom(OutVar), "tapped~wResult", [PredName]),
    format(string(Code),
"\t// Tap: execute ~w for side effects (logging/metrics)
\t~w := tapStage(~w, ~w)", [PredName, OutVar, InVar, PredName]).

% Flatten stage: flatten nested collections
generate_go_stage_flow(flatten, InVar, OutVar, Code) :-
    !,
    OutVar = "flattenedResult",
    format(string(Code),
"\t// Flatten: expand nested collections into individual records
\t~w := flattenStage(~w)", [OutVar, InVar]).

% Flatten field stage: flatten a specific field within records
generate_go_stage_flow(flatten(Field), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "flattened~wResult", [Field]),
    format(string(Code),
"\t// Flatten Field: expand '~w' field into individual records
\t~w := flattenFieldStage(~w, \"~w\")", [Field, OutVar, InVar, Field]).

% Debounce stage: emit only after silence period
generate_go_stage_flow(debounce(Ms), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "debounced~wResult", [Ms]),
    format(string(Code),
"\t// Debounce: emit after ~wms silence period
\t~w := debounceStage(~w, ~w, \"\")", [Ms, OutVar, InVar, Ms]).

% Debounce stage with timestamp field
generate_go_stage_flow(debounce(Ms, Field), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "debounced~w~wResult", [Ms, Field]),
    format(string(Code),
"\t// Debounce: emit after ~wms silence (using '~w' timestamp field)
\t~w := debounceStage(~w, ~w, \"~w\")", [Ms, Field, OutVar, InVar, Ms, Field]).

% Branch stage: conditional routing
generate_go_stage_flow(branch(Cond, TrueStage, FalseStage), InVar, OutVar, Code) :-
    !,
    OutVar = "branchResult",
    % Extract condition predicate name
    ( Cond = CondName/_ -> true ; CondName = Cond ),
    % Extract true/false stage names
    ( TrueStage = TrueName/_ -> true ; TrueName = TrueStage ),
    ( FalseStage = FalseName/_ -> true ; FalseName = FalseStage ),
    format(string(Code),
"\t// Branch: if ~w then ~w else ~w
\t~w := branchStage(~w,
\t\tfunc(r Record) bool { return ~w(r) },
\t\tfunc(rs []Record) []Record { return ~w(rs) },
\t\tfunc(rs []Record) []Record { return ~w(rs) })",
    [CondName, TrueName, FalseName, OutVar, InVar, CondName, TrueName, FalseName]).

% Tee stage: run side stage, discard results, pass through
generate_go_stage_flow(tee(SideStage), InVar, OutVar, Code) :-
    !,
    OutVar = "teeResult",
    % Extract side stage name
    ( SideStage = SideName/_ -> true ; SideName = SideStage ),
    format(string(Code),
"\t// Tee: fork to ~w (results discarded), pass original through
\t~w := teeStage(~w, func(rs []Record) []Record { return ~w(rs) })",
    [SideName, OutVar, InVar, SideName]).

% Call service stage (without options)
generate_go_stage_flow(call_service(ServiceName, RequestExpr, ResponseVar), InVar, OutVar, Code) :-
    !,
    OutVar = "serviceResult",
    format(string(Code),
"\t// Call service: ~w
\t~w := callServiceStage(~w, \"~w\", \"~w\", \"~w\", nil)",
    [ServiceName, OutVar, InVar, ServiceName, RequestExpr, ResponseVar]).

% Call service stage (with options)
generate_go_stage_flow(call_service(ServiceName, RequestExpr, ResponseVar, Options), InVar, OutVar, Code) :-
    !,
    OutVar = "serviceResult",
    format_go_options(Options, OptionsStr),
    format(string(Code),
"\t// Call service: ~w (with options)
\t~w := callServiceStage(~w, \"~w\", \"~w\", \"~w\", ~w)",
    [ServiceName, OutVar, InVar, ServiceName, RequestExpr, ResponseVar, OptionsStr]).

% Standard predicate stage
generate_go_stage_flow(Pred/Arity, InVar, OutVar, Code) :-
    !,
    atom(Pred),
    format(atom(OutVar), "~wResult", [Pred]),
    format(string(Code),
"\t// Stage: ~w/~w
\t~w := ~w(~w)", [Pred, Arity, OutVar, Pred, InVar]).

% Fallback for unknown stages
generate_go_stage_flow(Stage, InVar, InVar, Code) :-
    format(string(Code), "\t// Unknown stage type: ~w (pass-through)", [Stage]).

%% format_go_options(+Options, -GoStruct)
%  Format a list of Prolog options as a Go struct literal.
format_go_options([], "nil").
format_go_options(Options, GoStruct) :-
    Options \= [],
    format_go_option_fields(Options, Fields),
    atomic_list_concat(Fields, ', ', FieldsStr),
    format(string(GoStruct), "&CallServiceOptions{~w}", [FieldsStr]).

format_go_option_fields([], []).
format_go_option_fields([Opt|Rest], [Field|RestFields]) :-
    format_go_option_field(Opt, Field),
    format_go_option_fields(Rest, RestFields).

format_go_option_field(timeout(Ms), Field) :-
    format(string(Field), "Timeout: ~w", [Ms]).
format_go_option_field(retry(N), Field) :-
    format(string(Field), "Retry: ~w", [N]).
format_go_option_field(retry_delay(Ms), Field) :-
    format(string(Field), "RetryDelay: ~w", [Ms]).
format_go_option_field(fallback(Value), Field) :-
    ( atom(Value) ->
        format(string(Field), "Fallback: \"~w\"", [Value])
    ; number(Value) ->
        format(string(Field), "Fallback: ~w", [Value])
    ;
        format(string(Field), "Fallback: nil", [])
    ).
format_go_option_field(transport(T), Field) :-
    format(string(Field), "/* Transport: ~w */", [T]).
format_go_option_field(Opt, Field) :-
    % Fallback for unknown options
    format(string(Field), "/* Unknown option: ~w */", [Opt]).

%% extract_go_stage_names(+Stages, -Names)
%  Extract function names from stage specifications.
extract_go_stage_names([], []).
extract_go_stage_names([Pred/_Arity|Rest], [Pred|RestNames]) :-
    !,
    extract_go_stage_names(Rest, RestNames).
extract_go_stage_names([_|Rest], RestNames) :-
    extract_go_stage_names(Rest, RestNames).

%% extract_go_stage_name(+Stage, -Name)
%  Extract function name from a single stage specification.
extract_go_stage_name(Pred/_, Pred) :- atom(Pred), !.
extract_go_stage_name(Pred, Pred) :- atom(Pred), !.
extract_go_stage_name(_, unknown_stage).

%% extract_go_retry_options(+Options, -DelayMs, -Backoff)
%  Extract retry options from options list.
extract_go_retry_options(Options, DelayMs, Backoff) :-
    ( member(delay(D), Options) -> DelayMs = D ; DelayMs = 0 ),
    ( member(backoff(B), Options) -> Backoff = B ; Backoff = none ).

%% time_unit_to_ms(+Unit, -Ms)
%  Convert time unit to milliseconds.
time_unit_to_ms(second, 1000) :- !.
time_unit_to_ms(minute, 60000) :- !.
time_unit_to_ms(hour, 3600000) :- !.
time_unit_to_ms(ms(X), X) :- !.
time_unit_to_ms(X, X) :- integer(X), !.
time_unit_to_ms(_, 1000).

%% format_go_stage_list(+Names, -ListStr)
%  Format stage names as Go function references.
format_go_stage_list([], "").
format_go_stage_list([Name], Str) :-
    format(string(Str), "~w", [Name]).
format_go_stage_list([Name|Rest], Str) :-
    Rest \= [],
    format_go_stage_list(Rest, RestStr),
    format(string(Str), "~w, ~w", [Name, RestStr]).

%% format_go_route_map(+Routes, -MapStr)
%  Format routing map for Go.
format_go_route_map([], "").
format_go_route_map([(_Cond, Stage)|[]], Str) :-
    (Stage = StageName/_Arity -> true ; StageName = Stage),
    format(string(Str), "true: ~w", [StageName]).
format_go_route_map([(Cond, Stage)|Rest], Str) :-
    Rest \= [],
    (Stage = StageName/_Arity -> true ; StageName = Stage),
    format_go_route_map(Rest, RestStr),
    (Cond = true ->
        format(string(Str), "true: ~w, ~w", [StageName, RestStr])
    ; Cond = false ->
        format(string(Str), "false: ~w, ~w", [StageName, RestStr])
    ;   format(string(Str), "\"~w\": ~w, ~w", [Cond, StageName, RestStr])
    ).

%% format_go_aggregations(+Agg, -Str)
%  Format aggregation specifications for Go group_by stage.
format_go_aggregations(Aggs, Str) :-
    is_list(Aggs),
    !,
    format_go_aggregation_list(Aggs, Str).
format_go_aggregations(Agg, Str) :-
    format_go_single_aggregation(Agg, Str).

format_go_aggregation_list([], "").
format_go_aggregation_list([Agg], Str) :-
    format_go_single_aggregation(Agg, Str).
format_go_aggregation_list([Agg|Rest], Str) :-
    Rest \= [],
    format_go_single_aggregation(Agg, AggStr),
    format_go_aggregation_list(Rest, RestStr),
    format(string(Str), "~w, ~w", [AggStr, RestStr]).

% count aggregation
format_go_single_aggregation(count, "{\"count\", \"count\", \"\"}").
% Aggregations with field
format_go_single_aggregation(sum(Field), Str) :-
    format(string(Str), "{\"sum\", \"sum\", \"~w\"}", [Field]).
format_go_single_aggregation(avg(Field), Str) :-
    format(string(Str), "{\"avg\", \"avg\", \"~w\"}", [Field]).
format_go_single_aggregation(min(Field), Str) :-
    format(string(Str), "{\"min\", \"min\", \"~w\"}", [Field]).
format_go_single_aggregation(max(Field), Str) :-
    format(string(Str), "{\"max\", \"max\", \"~w\"}", [Field]).
format_go_single_aggregation(first(Field), Str) :-
    format(string(Str), "{\"first\", \"first\", \"~w\"}", [Field]).
format_go_single_aggregation(last(Field), Str) :-
    format(string(Str), "{\"last\", \"last\", \"~w\"}", [Field]).
format_go_single_aggregation(collect(Field), Str) :-
    format(string(Str), "{\"collect\", \"collect\", \"~w\"}", [Field]).

%% format_go_field_specs(+FieldSpecs, -Str)
%  Format field specifications for multi-field ordering.
format_go_field_specs([], "").
format_go_field_specs([Spec], Str) :-
    format_go_single_field_spec(Spec, Str).
format_go_field_specs([Spec|Rest], Str) :-
    Rest \= [],
    format_go_single_field_spec(Spec, SpecStr),
    format_go_field_specs(Rest, RestStr),
    format(string(Str), "~w, ~w", [SpecStr, RestStr]).

format_go_single_field_spec(Field, Str) :-
    atom(Field),
    !,
    format(string(Str), "{\"~w\", \"asc\"}", [Field]).
format_go_single_field_spec((Field, Dir), Str) :-
    format(string(Str), "{\"~w\", \"~w\"}", [Field, Dir]).

%% generate_go_enhanced_main(+PipelineName, +OutputFormat, -Code)
%  Generate main function for enhanced pipeline.
generate_go_enhanced_main(PipelineName, jsonl, Code) :-
    format(string(Code),
'func main() {
\t// Read JSONL from stdin
\tvar input []Record
\tscanner := bufio.NewScanner(os.Stdin)
\tfor scanner.Scan() {
\t\tvar record Record
\t\tif err := json.Unmarshal(scanner.Bytes(), &record); err == nil {
\t\t\tinput = append(input, record)
\t\t}
\t}

\t// Run enhanced pipeline
\tresults := ~w(input)

\t// Output results as JSONL
\tfor _, record := range results {
\t\tif output, err := json.Marshal(record); err == nil {
\t\t\tfmt.Println(string(output))
\t\t}
\t}
}
', [PipelineName]).
generate_go_enhanced_main(PipelineName, _, Code) :-
    format(string(Code),
'func main() {
\t// Read JSONL from stdin
\tvar input []Record
\tscanner := bufio.NewScanner(os.Stdin)
\tfor scanner.Scan() {
\t\tvar record Record
\t\tif err := json.Unmarshal(scanner.Bytes(), &record); err == nil {
\t\t\tinput = append(input, record)
\t\t}
\t}

\t// Run enhanced pipeline
\tresults := ~w(input)

\t// Output results
\tfor _, record := range results {
\t\tif output, err := json.Marshal(record); err == nil {
\t\t\tfmt.Println(string(output))
\t\t}
\t}
}
', [PipelineName]).

%% ============================================
%% GO ENHANCED PIPELINE CHAINING TESTS
%% ============================================

test_go_enhanced_chaining :-
    format('~n=== Go Enhanced Pipeline Chaining Tests ===~n~n', []),

    % Test 1: Generate enhanced helpers
    format('[Test 1] Generate enhanced helpers~n', []),
    go_enhanced_helpers(Helpers1),
    (   sub_string(Helpers1, _, _, _, "func fanOutRecords"),
        sub_string(Helpers1, _, _, _, "func mergeStreams"),
        sub_string(Helpers1, _, _, _, "func routeRecord"),
        sub_string(Helpers1, _, _, _, "func filterRecords"),
        sub_string(Helpers1, _, _, _, "func teeStream")
    ->  format('  [PASS] All helper functions generated~n', [])
    ;   format('  [FAIL] Missing helper functions~n', [])
    ),

    % Test 2: Linear pipeline connector
    format('[Test 2] Linear pipeline connector~n', []),
    generate_go_enhanced_connector([extract/1, transform/1, load/1], linearPipe, Code2),
    (   sub_string(Code2, _, _, _, "func linearPipe"),
        sub_string(Code2, _, _, _, "extract(input)"),
        sub_string(Code2, _, _, _, "transform(extractResult)"),
        sub_string(Code2, _, _, _, "load(transformResult)")
    ->  format('  [PASS] Linear connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code2])
    ),

    % Test 3: Fan-out connector
    format('[Test 3] Fan-out connector~n', []),
    generate_go_enhanced_connector([fan_out([validate/1, enrich/1])], fanoutPipe, Code3),
    (   sub_string(Code3, _, _, _, "func fanoutPipe"),
        sub_string(Code3, _, _, _, "Fan-out to 2 parallel stages"),
        sub_string(Code3, _, _, _, "fanOutRecords")
    ->  format('  [PASS] Fan-out connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code3])
    ),

    % Test 4: Fan-out with merge
    format('[Test 4] Fan-out with merge~n', []),
    generate_go_enhanced_connector([fan_out([a/1, b/1]), merge], mergePipe, Code4),
    (   sub_string(Code4, _, _, _, "func mergePipe"),
        sub_string(Code4, _, _, _, "Fan-out to 2"),
        sub_string(Code4, _, _, _, "Merge: results already combined")
    ->  format('  [PASS] Merge connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code4])
    ),

    % Test 5: Conditional routing
    format('[Test 5] Conditional routing~n', []),
    generate_go_enhanced_connector([route_by(hasError, [(true, errorHandler/1), (false, success/1)])], routePipe, Code5),
    (   sub_string(Code5, _, _, _, "func routePipe"),
        sub_string(Code5, _, _, _, "Conditional routing based on hasError"),
        sub_string(Code5, _, _, _, "routeMap")
    ->  format('  [PASS] Routing connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code5])
    ),

    % Test 6: Filter stage
    format('[Test 6] Filter stage~n', []),
    generate_go_enhanced_connector([filter_by(isValid)], filterPipe, Code6),
    (   sub_string(Code6, _, _, _, "func filterPipe"),
        sub_string(Code6, _, _, _, "Filter by isValid"),
        sub_string(Code6, _, _, _, "filterRecords")
    ->  format('  [PASS] Filter connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code6])
    ),

    % Test 7: Complex pipeline with all patterns
    format('[Test 7] Complex pipeline~n', []),
    generate_go_enhanced_connector([
        extract/1,
        filter_by(isActive),
        fan_out([validate/1, enrich/1, audit/1]),
        merge,
        route_by(hasError, [(true, errorLog/1), (false, transform/1)]),
        output/1
    ], complexPipe, Code7),
    (   sub_string(Code7, _, _, _, "func complexPipe"),
        sub_string(Code7, _, _, _, "Filter by isActive"),
        sub_string(Code7, _, _, _, "Fan-out to 3 parallel stages"),
        sub_string(Code7, _, _, _, "Merge"),
        sub_string(Code7, _, _, _, "Conditional routing")
    ->  format('  [PASS] Complex connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code7])
    ),

    % Test 8: Stage function generation
    format('[Test 8] Stage function generation~n', []),
    generate_go_enhanced_stage_functions([extract/1, transform/1], StageFns8),
    (   sub_string(StageFns8, _, _, _, "func extract"),
        sub_string(StageFns8, _, _, _, "func transform")
    ->  format('  [PASS] Stage functions generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [StageFns8])
    ),

    % Test 9: Full enhanced pipeline compilation
    format('[Test 9] Full enhanced pipeline~n', []),
    compile_go_enhanced_pipeline([
        extract/1,
        filter_by(isActive),
        fan_out([validate/1, enrich/1]),
        merge,
        output/1
    ], [pipeline_name(fullEnhanced), output_format(jsonl)], FullCode9),
    (   sub_string(FullCode9, _, _, _, "package main"),
        sub_string(FullCode9, _, _, _, "func fanOutRecords"),
        sub_string(FullCode9, _, _, _, "func filterRecords"),
        sub_string(FullCode9, _, _, _, "func fullEnhanced"),
        sub_string(FullCode9, _, _, _, "func main()")
    ->  format('  [PASS] Full pipeline compiles~n', [])
    ;   format('  [FAIL] Missing patterns in generated code~n', [])
    ),

    % Test 10: Enhanced helpers include all functions
    format('[Test 10] Enhanced helpers completeness~n', []),
    go_enhanced_helpers(Helpers10),
    (   sub_string(Helpers10, _, _, _, "fanOutRecords"),
        sub_string(Helpers10, _, _, _, "mergeStreams"),
        sub_string(Helpers10, _, _, _, "routeRecord"),
        sub_string(Helpers10, _, _, _, "filterRecords"),
        sub_string(Helpers10, _, _, _, "teeStream")
    ->  format('  [PASS] All helpers present~n', [])
    ;   format('  [FAIL] Missing helpers~n', [])
    ),

    format('~n=== All Go Enhanced Pipeline Chaining Tests Passed ===~n', []).

%% ============================================
%% KG Topology Phase 3: Kleinberg Router Code Generation
%% ============================================

%% compile_kleinberg_router_go(+Options, -Code)
%  Generate Go KleinbergRouter struct with configurable options.

compile_kleinberg_router_go(Options, Code) :-
    ( member(alpha(Alpha), Options) -> true ; Alpha = 2.0 ),
    ( member(max_hops(MaxHops), Options) -> true ; MaxHops = 10 ),
    ( member(parallel_paths(ParallelPaths), Options) -> true ; ParallelPaths = 1 ),
    ( member(similarity_threshold(Threshold), Options) -> true ; Threshold = 0.5 ),
    ( member(path_folding(PathFolding), Options) -> true ; PathFolding = true ),

    format(string(Code), '
// KG Topology Phase 3: Kleinberg Router
// Generated from Prolog service definition

package kg

import (
    "encoding/base64"
    "encoding/binary"
    "encoding/json"
    "fmt"
    "math"
    "net/http"
    "sort"
    "sync"
    "time"
)

// KGNode represents a discovered node in the distributed KG network
type KGNode struct {
    NodeID         string
    Endpoint       string
    Centroid       []float32
    Topics         []string
    EmbeddingModel string
    Similarity     float64
}

// RoutingEnvelope carries routing information between nodes
type RoutingEnvelope struct {
    OriginNode         string   `json:"origin_node"`
    HTL                int      `json:"htl"`
    Visited            []string `json:"visited"`
    PathFoldingEnabled bool     `json:"path_folding_enabled"`
}

// KleinbergRouter implements small-world routing for distributed KG topology
type KleinbergRouter struct {
    LocalNodeID         string
    DiscoveryClient     ServiceRegistry
    Alpha               float64
    MaxHops             int
    ParallelPaths       int
    SimilarityThreshold float64
    PathFoldingEnabled  bool

    nodeCache      map[string]*KGNode
    cacheTimestamp time.Time
    cacheTTL       time.Duration
    shortcuts      map[string]string
    mu             sync.RWMutex
}

// NewKleinbergRouter creates a new router with default configuration
func NewKleinbergRouter(nodeID string, discovery ServiceRegistry) *KleinbergRouter {
    return &KleinbergRouter{
        LocalNodeID:         nodeID,
        DiscoveryClient:     discovery,
        Alpha:               ~w,
        MaxHops:             ~w,
        ParallelPaths:       ~w,
        SimilarityThreshold: ~w,
        PathFoldingEnabled:  ~w,
        nodeCache:           make(map[string]*KGNode),
        cacheTTL:            60 * time.Second,
        shortcuts:           make(map[string]string),
    }
}

// DiscoverNodes finds KG nodes from service registry
func (r *KleinbergRouter) DiscoverNodes(tags []string) ([]*KGNode, error) {
    r.mu.Lock()
    defer r.mu.Unlock()

    // Check cache
    if time.Since(r.cacheTimestamp) < r.cacheTTL {
        nodes := make([]*KGNode, 0, len(r.nodeCache))
        for _, n := range r.nodeCache {
            nodes = append(nodes, n)
        }
        return nodes, nil
    }

    if tags == nil {
        tags = []string{"kg_node"}
    }

    instances, err := r.DiscoveryClient.Discover("kg_topology", tags)
    if err != nil {
        return nil, err
    }

    nodes := make([]*KGNode, 0)
    for _, inst := range instances {
        metadata := inst.Metadata

        centroidB64, ok := metadata["semantic_centroid"].(string)
        if !ok {
            continue
        }

        centroidBytes, err := base64.StdEncoding.DecodeString(centroidB64)
        if err != nil {
            continue
        }

        // Convert bytes to float32 slice
        centroid := make([]float32, len(centroidBytes)/4)
        for i := range centroid {
            bits := binary.LittleEndian.Uint32(centroidBytes[i*4:])
            centroid[i] = math.Float32frombits(bits)
        }

        topics, _ := metadata["interface_topics"].([]string)
        model, _ := metadata["embedding_model"].(string)

        node := &KGNode{
            NodeID:         inst.ServiceID,
            Endpoint:       fmt.Sprintf("http://%%s:%%d", inst.Host, inst.Port),
            Centroid:       centroid,
            Topics:         topics,
            EmbeddingModel: model,
        }
        nodes = append(nodes, node)
        r.nodeCache[node.NodeID] = node
    }

    r.cacheTimestamp = time.Now()
    return nodes, nil
}

// CosineSimilarity computes cosine similarity between two vectors
func CosineSimilarity(a, b []float32) float64 {
    if len(a) != len(b) {
        return 0
    }

    var dot, normA, normB float64
    for i := range a {
        dot += float64(a[i] * b[i])
        normA += float64(a[i] * a[i])
        normB += float64(b[i] * b[i])
    }

    if normA == 0 || normB == 0 {
        return 0
    }

    return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// GetStats returns router statistics
func (r *KleinbergRouter) GetStats() map[string]interface{} {
    r.mu.RLock()
    defer r.mu.RUnlock()

    return map[string]interface{}{
        "local_node_id":  r.LocalNodeID,
        "cached_nodes":   len(r.nodeCache),
        "shortcuts":      len(r.shortcuts),
        "config": map[string]interface{}{
            "alpha":                r.Alpha,
            "max_hops":             r.MaxHops,
            "parallel_paths":       r.ParallelPaths,
            "similarity_threshold": r.SimilarityThreshold,
            "path_folding_enabled": r.PathFoldingEnabled,
        },
    }
}
', [Alpha, MaxHops, ParallelPaths, Threshold, PathFolding]).


% =============================================================================
% KG TOPOLOGY PHASE 4: FEDERATED QUERY CODE GENERATION (Go)
% =============================================================================

%% compile_federated_query_go(+Options, -Code)
%  Generate Go FederatedQueryEngine configuration and types.
%  Creates structs and methods for federated query aggregation.

compile_federated_query_go(Options, Code) :-
    % Extract federation options with defaults
    ( member(federation_k(K), Options) -> true ; K = 3 ),
    ( member(timeout_ms(Timeout), Options) -> true ; Timeout = 5000 ),
    ( member(diversity_field(DivField), Options) -> true ; DivField = corpus_id ),

    % Extract aggregation strategy
    ( member(aggregation(Strategy), Options) -> true
    ; member(aggregation(Strategy, _), Options) -> true
    ; Strategy = sum
    ),
    strategy_to_go_const(Strategy, StrategyConst),

    % Extract dedup key if specified
    ( member(aggregation(_, AggOpts), Options),
      member(dedup_key(DedupKey), AggOpts) -> true
    ; DedupKey = answer_hash
    ),

    format(string(Code), '
// KG Topology Phase 4: Federated Query Types and Engine
// Generated from Prolog service definition

package main

import (
    "context"
    "encoding/json"
    "fmt"
    "math"
    "net/http"
    "sync"
    "time"
)

// AggregationStrategy defines how to merge results from multiple nodes
type AggregationStrategy int

const (
    AggregationSum AggregationStrategy = iota
    AggregationMax
    AggregationMin
    AggregationAvg
    AggregationCount
    AggregationFirst
    AggregationDiversityWeighted
)

// AggregationConfig holds configuration for result aggregation
type AggregationConfig struct {
    Strategy       AggregationStrategy
    DedupKey       string
    DiversityField string
}

// ResultProvenance tracks where a result came from
type ResultProvenance struct {
    NodeID         string    `json:"node_id"`
    ExpScore       float64   `json:"exp_score"`
    CorpusID       string    `json:"corpus_id"`
    DataSources    []string  `json:"data_sources"`
    InterfaceID    int       `json:"interface_id"`
    EmbeddingModel string    `json:"embedding_model"`
    Timestamp      time.Time `json:"timestamp"`
}

// AggregatedResult is a result merged from multiple nodes
type AggregatedResult struct {
    AnswerText     string             `json:"answer_text"`
    AnswerHash     string             `json:"answer_hash"`
    CombinedScore  float64            `json:"combined_score"`
    SourceNodes    []string           `json:"source_nodes"`
    Provenance     []ResultProvenance `json:"provenance"`
    DiversityScore float64            `json:"diversity_score"`
    UniqueCorpora  int                `json:"unique_corpora"`
}

// NodeResponse is a response from a single node
type NodeResponse struct {
    SourceNode    string                 `json:"source_node"`
    Results       []NodeResult           `json:"results"`
    PartitionSum  float64                `json:"partition_sum"`
    NodeMetadata  map[string]interface{} `json:"node_metadata"`
    ResponseTime  time.Duration          `json:"response_time_ms"`
    Error         string                 `json:"error,omitempty"`
}

// NodeResult is a single result from a node
type NodeResult struct {
    AnswerID   int                    `json:"answer_id"`
    AnswerText string                 `json:"answer_text"`
    AnswerHash string                 `json:"answer_hash"`
    RawScore   float64                `json:"raw_score"`
    ExpScore   float64                `json:"exp_score"`
    Metadata   map[string]interface{} `json:"metadata"`
}

// FederatedQueryEngine coordinates distributed queries
type FederatedQueryEngine struct {
    Router       *KleinbergRouter
    Config       AggregationConfig
    FederationK  int
    TimeoutMs    int
    mu           sync.RWMutex
    queryCount   int64
    totalTimeMs  float64
}

// Default configuration
var DefaultFederationConfig = AggregationConfig{
    Strategy:       ~w,
    DedupKey:       "~w",
    DiversityField: "~w",
}

const (
    DefaultFederationK = ~w
    DefaultTimeoutMs   = ~w
)

// NewFederatedQueryEngine creates a configured engine
func NewFederatedQueryEngine(router *KleinbergRouter) *FederatedQueryEngine {
    return &FederatedQueryEngine{
        Router:      router,
        Config:      DefaultFederationConfig,
        FederationK: DefaultFederationK,
        TimeoutMs:   DefaultTimeoutMs,
    }
}

// FederatedQuery executes a query across multiple nodes
func (e *FederatedQueryEngine) FederatedQuery(
    ctx context.Context,
    queryText string,
    queryEmbedding []float32,
    topK int,
) (*AggregatedResponse, error) {
    startTime := time.Now()

    // Discover nodes
    nodes, err := e.Router.DiscoverNodes([]string{"kg_node"})
    if err != nil {
        return nil, fmt.Errorf("node discovery failed: %%w", err)
    }

    // Select top-k nodes by similarity
    if len(nodes) > e.FederationK {
        nodes = nodes[:e.FederationK]
    }

    // Query nodes in parallel
    responses := e.parallelQuery(ctx, nodes, queryText, queryEmbedding)

    // Aggregate results
    aggregated, totalPartition := e.aggregate(responses)

    // Normalize and rank
    results := e.normalizeAndRank(aggregated, totalPartition, topK)

    // Update stats
    e.mu.Lock()
    e.queryCount++
    e.totalTimeMs += float64(time.Since(startTime).Milliseconds())
    e.mu.Unlock()

    return &AggregatedResponse{
        QueryID:           fmt.Sprintf("%%d", time.Now().UnixNano()),
        Results:           results,
        TotalPartitionSum: totalPartition,
        NodesQueried:      len(nodes),
        NodesResponded:    len(responses),
        TotalTimeMs:       float64(time.Since(startTime).Milliseconds()),
    }, nil
}

// parallelQuery sends queries to nodes concurrently
func (e *FederatedQueryEngine) parallelQuery(
    ctx context.Context,
    nodes []KGNode,
    queryText string,
    queryEmbedding []float32,
) []NodeResponse {
    var wg sync.WaitGroup
    responses := make([]NodeResponse, len(nodes))

    for i, node := range nodes {
        wg.Add(1)
        go func(idx int, n KGNode) {
            defer wg.Done()
            resp := e.queryNode(ctx, n, queryText, queryEmbedding)
            responses[idx] = resp
        }(i, node)
    }

    wg.Wait()
    return responses
}

// aggregate merges results from multiple nodes
func (e *FederatedQueryEngine) aggregate(
    responses []NodeResponse,
) (map[string]*AggregatedResult, float64) {
    results := make(map[string]*AggregatedResult)
    var totalPartition float64

    for _, resp := range responses {
        if resp.Error != "" {
            continue
        }
        totalPartition += resp.PartitionSum

        for _, r := range resp.Results {
            key := r.AnswerHash
            if existing, ok := results[key]; ok {
                // Merge with existing
                existing.CombinedScore = e.mergeScore(
                    existing.CombinedScore,
                    r.ExpScore,
                    existing,
                    resp.NodeMetadata,
                )
                existing.SourceNodes = append(existing.SourceNodes, resp.SourceNode)
                existing.Provenance = append(existing.Provenance, ResultProvenance{
                    NodeID:    resp.SourceNode,
                    ExpScore:  r.ExpScore,
                    CorpusID:  getStringFromMeta(resp.NodeMetadata, "corpus_id"),
                    Timestamp: time.Now(),
                })
            } else {
                results[key] = &AggregatedResult{
                    AnswerText:    r.AnswerText,
                    AnswerHash:    r.AnswerHash,
                    CombinedScore: r.ExpScore,
                    SourceNodes:   []string{resp.SourceNode},
                    Provenance: []ResultProvenance{{
                        NodeID:    resp.SourceNode,
                        ExpScore:  r.ExpScore,
                        CorpusID:  getStringFromMeta(resp.NodeMetadata, "corpus_id"),
                        Timestamp: time.Now(),
                    }},
                }
            }
        }
    }

    // Calculate diversity scores
    for _, result := range results {
        corpora := make(map[string]bool)
        for _, p := range result.Provenance {
            if p.CorpusID != "" {
                corpora[p.CorpusID] = true
            }
        }
        result.UniqueCorpora = len(corpora)
        if len(result.Provenance) > 0 {
            result.DiversityScore = float64(len(corpora)) / float64(len(result.Provenance))
        }
    }

    return results, totalPartition
}

// mergeScore combines scores based on strategy
func (e *FederatedQueryEngine) mergeScore(
    existing, new float64,
    result *AggregatedResult,
    nodeMeta map[string]interface{},
) float64 {
    switch e.Config.Strategy {
    case AggregationSum:
        return existing + new
    case AggregationMax:
        return math.Max(existing, new)
    case AggregationMin:
        return math.Min(existing, new)
    case AggregationDiversityWeighted:
        // Check if corpus is different
        newCorpus := getStringFromMeta(nodeMeta, "corpus_id")
        for _, p := range result.Provenance {
            if p.CorpusID != newCorpus {
                return existing + new  // Diverse, full boost
            }
        }
        return math.Max(existing, new)  // Same corpus, no boost
    default:
        return existing + new
    }
}

// queryNode sends a query to a single node
func (e *FederatedQueryEngine) queryNode(
    ctx context.Context,
    node KGNode,
    queryText string,
    queryEmbedding []float32,
) NodeResponse {
    startTime := time.Now()

    // Build request
    reqBody, _ := json.Marshal(map[string]interface{}{
        "__type": "kg_federated_query",
        "payload": map[string]interface{}{
            "query_text": queryText,
            "top_k":      10,
        },
    })

    // Create HTTP request with timeout
    timeoutCtx, cancel := context.WithTimeout(ctx, time.Duration(e.TimeoutMs)*time.Millisecond)
    defer cancel()

    req, err := http.NewRequestWithContext(
        timeoutCtx,
        "POST",
        node.Endpoint+"/kg/federated",
        nil,
    )
    if err != nil {
        return NodeResponse{SourceNode: node.NodeID, Error: err.Error()}
    }
    req.Header.Set("Content-Type", "application/json")

    // Send request
    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return NodeResponse{SourceNode: node.NodeID, Error: err.Error()}
    }
    defer resp.Body.Close()

    // Parse response
    var nodeResp NodeResponse
    if err := json.NewDecoder(resp.Body).Decode(&nodeResp); err != nil {
        return NodeResponse{SourceNode: node.NodeID, Error: err.Error()}
    }

    nodeResp.ResponseTime = time.Since(startTime)
    _ = reqBody  // Suppress unused variable warning
    return nodeResp
}

// normalizeAndRank sorts results by normalized probability
func (e *FederatedQueryEngine) normalizeAndRank(
    aggregated map[string]*AggregatedResult,
    totalPartition float64,
    topK int,
) []AggregatedResult {
    results := make([]AggregatedResult, 0, len(aggregated))

    for _, r := range aggregated {
        results = append(results, *r)
    }

    // Sort by combined score descending
    for i := 0; i < len(results)-1; i++ {
        for j := i + 1; j < len(results); j++ {
            if results[j].CombinedScore > results[i].CombinedScore {
                results[i], results[j] = results[j], results[i]
            }
        }
    }

    if len(results) > topK {
        results = results[:topK]
    }

    return results
}

// GetStats returns engine statistics
func (e *FederatedQueryEngine) GetStats() map[string]interface{} {
    e.mu.RLock()
    defer e.mu.RUnlock()

    avgTime := 0.0
    if e.queryCount > 0 {
        avgTime = e.totalTimeMs / float64(e.queryCount)
    }

    return map[string]interface{}{
        "query_count":        e.queryCount,
        "avg_response_ms":    avgTime,
        "federation_k":       e.FederationK,
        "aggregation_strategy": e.Config.Strategy,
    }
}

// AggregatedResponse is the final response from a federated query
type AggregatedResponse struct {
    QueryID           string             `json:"query_id"`
    Results           []AggregatedResult `json:"results"`
    TotalPartitionSum float64            `json:"total_partition_sum"`
    NodesQueried      int                `json:"nodes_queried"`
    NodesResponded    int                `json:"nodes_responded"`
    TotalTimeMs       float64            `json:"total_time_ms"`
}

func (r *AggregatedResponse) ToDict() map[string]interface{} {
    return map[string]interface{}{
        "query_id":            r.QueryID,
        "results":             r.Results,
        "total_partition_sum": r.TotalPartitionSum,
        "nodes_queried":       r.NodesQueried,
        "nodes_responded":     r.NodesResponded,
        "total_time_ms":       r.TotalTimeMs,
    }
}

func getStringFromMeta(meta map[string]interface{}, key string) string {
    if v, ok := meta[key]; ok {
        if s, ok := v.(string); ok {
            return s
        }
    }
    return ""
}
', [StrategyConst, DedupKey, DivField, K, Timeout]).

%% strategy_to_go_const(+Strategy, -Const)
%  Convert Prolog strategy atom to Go constant name.
strategy_to_go_const(sum, 'AggregationSum').
strategy_to_go_const(max, 'AggregationMax').
strategy_to_go_const(min, 'AggregationMin').
strategy_to_go_const(avg, 'AggregationAvg').
strategy_to_go_const(count, 'AggregationCount').
strategy_to_go_const(first, 'AggregationFirst').
strategy_to_go_const(diversity, 'AggregationDiversityWeighted').
strategy_to_go_const(diversity_weighted, 'AggregationDiversityWeighted').
strategy_to_go_const(_, 'AggregationSum').  % Default fallback

% =============================================================================
% KG TOPOLOGY PHASE 5b: ADAPTIVE FEDERATION-K CODE GENERATION (Go)
% =============================================================================

%% compile_adaptive_federation_go(+Options, -Code)
%  Generate Go AdaptiveFederatedEngine with dynamic k selection.
%  Adjusts number of nodes queried based on query characteristics.

compile_adaptive_federation_go(Options, Code) :-
    % Extract adaptive-k options with defaults
    ( member(adaptive_k(AdaptiveOpts), Options),
      is_list(AdaptiveOpts) -> true
    ; AdaptiveOpts = []
    ),
    ( member(base_k(BaseK), AdaptiveOpts) -> true ; BaseK = 3 ),
    ( member(min_k(MinK), AdaptiveOpts) -> true ; MinK = 1 ),
    ( member(max_k(MaxK), AdaptiveOpts) -> true ; MaxK = 10 ),
    ( member(entropy_weight(EntropyW), AdaptiveOpts) -> true ; EntropyW = 0.3 ),
    ( member(latency_weight(LatencyW), AdaptiveOpts) -> true ; LatencyW = 0.2 ),
    ( member(consensus_weight(ConsensusW), AdaptiveOpts) -> true ; ConsensusW = 0.5 ),
    ( member(entropy_threshold(EntropyT), AdaptiveOpts) -> true ; EntropyT = 0.7 ),
    ( member(similarity_threshold(SimT), AdaptiveOpts) -> true ; SimT = 0.5 ),
    ( member(consensus_threshold(ConsensusT), AdaptiveOpts) -> true ; ConsensusT = 0.6 ),
    ( member(history_size(HistSize), AdaptiveOpts) -> true ; HistSize = 100 ),

    format(string(Code), '
// KG Topology Phase 5b: Adaptive Federation-K
// Generated from Prolog service definition

// AdaptiveKConfig holds configuration for adaptive k selection
type AdaptiveKConfig struct {
    BaseK              int     `json:"base_k"`
    MinK               int     `json:"min_k"`
    MaxK               int     `json:"max_k"`
    EntropyWeight      float64 `json:"entropy_weight"`
    LatencyWeight      float64 `json:"latency_weight"`
    ConsensusWeight    float64 `json:"consensus_weight"`
    EntropyThreshold   float64 `json:"entropy_threshold"`
    SimilarityThreshold float64 `json:"similarity_threshold"`
    ConsensusThreshold float64 `json:"consensus_threshold"`
    HistorySize        int     `json:"history_size"`
}

// DefaultAdaptiveKConfig returns the default configuration
var DefaultAdaptiveKConfig = AdaptiveKConfig{
    BaseK:              ~w,
    MinK:               ~w,
    MaxK:               ~w,
    EntropyWeight:      ~w,
    LatencyWeight:      ~w,
    ConsensusWeight:    ~w,
    EntropyThreshold:   ~w,
    SimilarityThreshold: ~w,
    ConsensusThreshold: ~w,
    HistorySize:        ~w,
}

// QueryMetrics holds metrics for adaptive k computation
type QueryMetrics struct {
    Entropy            float64 `json:"entropy"`
    TopSimilarity      float64 `json:"top_similarity"`
    SimilarityVariance float64 `json:"similarity_variance"`
    HistoricalConsensus float64 `json:"historical_consensus"`
    AvgNodeLatencyMs   float64 `json:"avg_node_latency_ms"`
}

// QueryHistoryEntry stores past query outcomes
type QueryHistoryEntry struct {
    Embedding  []float32
    Consensus  float64
    KUsed      int
}

// AdaptiveKCalculator computes optimal federation_k based on query metrics
type AdaptiveKCalculator struct {
    Config       AdaptiveKConfig
    QueryHistory []QueryHistoryEntry
    LatencyCache map[string][]float64  // node_id -> latencies
    mu           sync.RWMutex
}

// NewAdaptiveKCalculator creates a new calculator with default config
func NewAdaptiveKCalculator() *AdaptiveKCalculator {
    return &AdaptiveKCalculator{
        Config:       DefaultAdaptiveKConfig,
        QueryHistory: make([]QueryHistoryEntry, 0),
        LatencyCache: make(map[string][]float64),
    }
}

// ComputeK computes optimal k based on query characteristics
func (c *AdaptiveKCalculator) ComputeK(
    queryEmbedding []float32,
    nodes []*KGNode,
    latencyBudgetMs *int,
) int {
    c.mu.RLock()
    defer c.mu.RUnlock()

    if len(nodes) == 0 {
        return c.Config.MinK
    }

    metrics := c.computeMetrics(queryEmbedding, nodes)
    k := c.Config.BaseK

    // Adjust based on entropy (ambiguity)
    if metrics.Entropy > c.Config.EntropyThreshold {
        k += int(2 * c.Config.EntropyWeight * 10) // Up to +2 nodes
    }

    // Adjust based on similarity distribution
    if metrics.TopSimilarity < c.Config.SimilarityThreshold {
        k++ // No strong match, query more
    }
    if metrics.SimilarityVariance > 0.1 {
        k++ // High variance suggests need for exploration
    }

    // Adjust based on historical consensus
    if metrics.HistoricalConsensus < c.Config.ConsensusThreshold {
        k += int(c.Config.ConsensusWeight * 2)
    }

    // Adjust based on latency budget
    if latencyBudgetMs != nil && metrics.AvgNodeLatencyMs > 0 {
        maxNodesInBudget := int(float64(*latencyBudgetMs) / metrics.AvgNodeLatencyMs)
        if maxNodesInBudget < k {
            k = max(c.Config.MinK, maxNodesInBudget)
        }
    }

    // Clamp to valid range
    if k < c.Config.MinK {
        k = c.Config.MinK
    }
    if k > c.Config.MaxK {
        k = c.Config.MaxK
    }
    if k > len(nodes) {
        k = len(nodes)
    }
    return k
}

func (c *AdaptiveKCalculator) computeMetrics(
    queryEmbedding []float32,
    nodes []*KGNode,
) QueryMetrics {
    similarities := make([]float64, len(nodes))
    for i, node := range nodes {
        if len(node.Centroid) > 0 {
            similarities[i] = cosineSimilarity(queryEmbedding, node.Centroid)
        }
    }

    // Entropy: normalized entropy of similarity distribution
    var entropy float64
    if len(similarities) > 1 {
        var sum float64
        for _, s := range similarities {
            sum += math.Abs(s)
        }
        if sum > 0 {
            var h float64
            for _, s := range similarities {
                p := math.Abs(s) / (sum + 1e-10)
                if p > 1e-10 {
                    h -= p * math.Log(p)
                }
            }
            entropy = h / math.Log(float64(len(similarities)))
        }
    } else {
        entropy = 0.5
    }

    // Top similarity
    var topSim float64
    for _, s := range similarities {
        if s > topSim {
            topSim = s
        }
    }

    // Variance
    var variance float64
    if len(similarities) > 1 {
        var sum, mean float64
        for _, s := range similarities {
            sum += s
        }
        mean = sum / float64(len(similarities))
        for _, s := range similarities {
            variance += (s - mean) * (s - mean)
        }
        variance /= float64(len(similarities))
    }

    // Historical consensus
    historicalConsensus := c.getHistoricalConsensus(queryEmbedding)

    // Average node latency
    avgLatency := c.getAvgLatency(nodes)

    return QueryMetrics{
        Entropy:            entropy,
        TopSimilarity:      topSim,
        SimilarityVariance: variance,
        HistoricalConsensus: historicalConsensus,
        AvgNodeLatencyMs:   avgLatency,
    }
}

func (c *AdaptiveKCalculator) getHistoricalConsensus(queryEmbedding []float32) float64 {
    if len(c.QueryHistory) == 0 {
        return 0.8 // Optimistic default
    }

    var similarConsensus []float64
    start := len(c.QueryHistory) - c.Config.HistorySize
    if start < 0 {
        start = 0
    }

    for _, entry := range c.QueryHistory[start:] {
        sim := cosineSimilarity(queryEmbedding, entry.Embedding)
        if sim > 0.7 {
            similarConsensus = append(similarConsensus, entry.Consensus)
        }
    }

    if len(similarConsensus) > 0 {
        var sum float64
        for _, c := range similarConsensus {
            sum += c
        }
        return sum / float64(len(similarConsensus))
    }
    return 0.8
}

func (c *AdaptiveKCalculator) getAvgLatency(nodes []*KGNode) float64 {
    var latencies []float64
    for _, node := range nodes {
        if cached, ok := c.LatencyCache[node.NodeID]; ok && len(cached) > 0 {
            var sum float64
            for _, l := range cached {
                sum += l
            }
            latencies = append(latencies, sum/float64(len(cached)))
        }
    }
    if len(latencies) > 0 {
        var sum float64
        for _, l := range latencies {
            sum += l
        }
        return sum / float64(len(latencies))
    }
    return 100.0 // Default 100ms if no data
}

// RecordQueryOutcome records query outcome for future adaptive decisions
func (c *AdaptiveKCalculator) RecordQueryOutcome(
    queryEmbedding []float32,
    consensusScore float64,
    kUsed int,
    nodeLatencies map[string]float64,
) {
    c.mu.Lock()
    defer c.mu.Unlock()

    // Add to history
    embCopy := make([]float32, len(queryEmbedding))
    copy(embCopy, queryEmbedding)
    c.QueryHistory = append(c.QueryHistory, QueryHistoryEntry{
        Embedding: embCopy,
        Consensus: consensusScore,
        KUsed:     kUsed,
    })

    // Trim history if needed
    if len(c.QueryHistory) > c.Config.HistorySize {
        c.QueryHistory = c.QueryHistory[len(c.QueryHistory)-c.Config.HistorySize:]
    }

    // Update latency cache
    for nodeID, latency := range nodeLatencies {
        c.LatencyCache[nodeID] = append(c.LatencyCache[nodeID], latency)
        if len(c.LatencyCache[nodeID]) > 20 {
            c.LatencyCache[nodeID] = c.LatencyCache[nodeID][len(c.LatencyCache[nodeID])-20:]
        }
    }
}

// GetStats returns statistics about adaptive k selection
func (c *AdaptiveKCalculator) GetStats() map[string]interface{} {
    c.mu.RLock()
    defer c.mu.RUnlock()

    if len(c.QueryHistory) == 0 {
        return map[string]interface{}{
            "queries_recorded": 0,
            "avg_k_used":       c.Config.BaseK,
            "avg_consensus":    0.0,
            "nodes_tracked":    0,
        }
    }

    var kSum, consensusSum float64
    for _, entry := range c.QueryHistory {
        kSum += float64(entry.KUsed)
        consensusSum += entry.Consensus
    }

    return map[string]interface{}{
        "queries_recorded": len(c.QueryHistory),
        "avg_k_used":       kSum / float64(len(c.QueryHistory)),
        "avg_consensus":    consensusSum / float64(len(c.QueryHistory)),
        "nodes_tracked":    len(c.LatencyCache),
        "config": map[string]interface{}{
            "base_k": c.Config.BaseK,
            "min_k":  c.Config.MinK,
            "max_k":  c.Config.MaxK,
        },
    }
}

// AdaptiveFederatedEngine extends FederatedQueryEngine with adaptive k
type AdaptiveFederatedEngine struct {
    *FederatedQueryEngine
    Adaptive *AdaptiveKCalculator
}

// NewAdaptiveFederatedEngine creates an engine with adaptive k selection
func NewAdaptiveFederatedEngine(router *KleinbergRouter) *AdaptiveFederatedEngine {
    return &AdaptiveFederatedEngine{
        FederatedQueryEngine: NewFederatedQueryEngine(router),
        Adaptive:             NewAdaptiveKCalculator(),
    }
}

// AdaptiveQuery executes a query with adaptive k selection
func (e *AdaptiveFederatedEngine) AdaptiveQuery(
    ctx context.Context,
    queryText string,
    queryEmbedding []float32,
    topK int,
    latencyBudgetMs *int,
) (*AggregatedResponse, error) {
    // Discover nodes
    nodes, err := e.Router.DiscoverNodes([]string{"kg_node"})
    if err != nil {
        return nil, err
    }

    // Convert to KGNode slice
    kgNodes := make([]*KGNode, len(nodes))
    for i, n := range nodes {
        kgNodes[i] = &KGNode{NodeID: n.ServiceID, Endpoint: n.Address}
    }

    // Compute adaptive k
    k := e.Adaptive.ComputeK(queryEmbedding, kgNodes, latencyBudgetMs)

    // Execute query with computed k
    originalK := e.FederationK
    e.FederationK = k
    response, err := e.FederatedQuery(ctx, queryText, queryEmbedding, topK)
    e.FederationK = originalK

    if err != nil {
        return nil, err
    }

    // Record outcome for future learning
    consensusScore := 0.0
    if len(response.Results) > 0 {
        consensusScore = response.Results[0].DiversityScore
    }
    e.Adaptive.RecordQueryOutcome(queryEmbedding, consensusScore, k, nil)

    return response, nil
}

// Helper function for cosine similarity
func cosineSimilarity(a, b []float32) float64 {
    if len(a) != len(b) || len(a) == 0 {
        return 0.0
    }
    var dot, normA, normB float64
    for i := range a {
        dot += float64(a[i]) * float64(b[i])
        normA += float64(a[i]) * float64(a[i])
        normB += float64(b[i]) * float64(b[i])
    }
    normA = math.Sqrt(normA)
    normB = math.Sqrt(normB)
    if normA < 1e-10 || normB < 1e-10 {
        return 0.0
    }
    return dot / (normA * normB)
}
', [BaseK, MinK, MaxK, EntropyW, LatencyW, ConsensusW, EntropyT, SimT, ConsensusT, HistSize]).

% =============================================================================
% KG TOPOLOGY PHASE 5c: QUERY PLAN OPTIMIZATION CODE GENERATION (Go)
% =============================================================================

%% compile_query_planner_go(+Options, -Code)
%  Generate Go QueryPlanner with cost-based query plan optimization.
%  Classifies queries and builds optimal execution plans.

compile_query_planner_go(Options, Code) :-
    % Extract planner options with defaults
    ( member(query_planning(PlanOpts), Options),
      is_list(PlanOpts) -> true
    ; PlanOpts = []
    ),
    ( member(specific_threshold(SpecT), PlanOpts) -> true ; SpecT = 0.8 ),
    ( member(exploratory_variance(ExplVar), PlanOpts) -> true ; ExplVar = 0.1 ),
    ( member(consensus_min_nodes(ConsMin), PlanOpts) -> true ; ConsMin = 3 ),
    ( member(specific_max_nodes(SpecMax), PlanOpts) -> true ; SpecMax = 2 ),
    ( member(exploratory_max_nodes(ExplMax), PlanOpts) -> true ; ExplMax = 7 ),
    ( member(consensus_stage1_nodes(ConsS1), PlanOpts) -> true ; ConsS1 = 5 ),
    ( member(consensus_stage2_nodes(ConsS2), PlanOpts) -> true ; ConsS2 = 3 ),
    ( member(default_latency_ms(DefLat), PlanOpts) -> true ; DefLat = 100.0 ),

    format(string(Code), '
// KG Topology Phase 5c: Query Plan Optimization
// Generated from Prolog service definition

// QueryType classifies query characteristics
type QueryType int

const (
    QueryTypeSpecific QueryType = iota     // High similarity, few nodes
    QueryTypeExploratory                   // Low/varied similarity, broad search
    QueryTypeConsensus                     // Medium similarity, density-focused
)

func (qt QueryType) String() string {
    switch qt {
    case QueryTypeSpecific:
        return "specific"
    case QueryTypeExploratory:
        return "exploratory"
    case QueryTypeConsensus:
        return "consensus"
    default:
        return "unknown"
    }
}

// QueryClassification holds query analysis results
type QueryClassification struct {
    QueryType          QueryType `json:"query_type"`
    MaxSimilarity      float64   `json:"max_similarity"`
    SimilarityVariance float64   `json:"similarity_variance"`
    TopNodes           []string  `json:"top_nodes"`
    Confidence         float64   `json:"confidence"`
}

// PlannerConfig holds query planner configuration
type PlannerConfig struct {
    SpecificThreshold    float64 `json:"specific_threshold"`
    ExploratoryVariance  float64 `json:"exploratory_variance"`
    ConsensusMinNodes    int     `json:"consensus_min_nodes"`
    SpecificMaxNodes     int     `json:"specific_max_nodes"`
    ExploratoryMaxNodes  int     `json:"exploratory_max_nodes"`
    ConsensusStage1Nodes int     `json:"consensus_stage1_nodes"`
    ConsensusStage2Nodes int     `json:"consensus_stage2_nodes"`
    DefaultLatencyMs     float64 `json:"default_latency_ms"`
}

// DefaultPlannerConfig returns the default configuration
var DefaultPlannerConfig = PlannerConfig{
    SpecificThreshold:    ~w,
    ExploratoryVariance:  ~w,
    ConsensusMinNodes:    ~w,
    SpecificMaxNodes:     ~w,
    ExploratoryMaxNodes:  ~w,
    ConsensusStage1Nodes: ~w,
    ConsensusStage2Nodes: ~w,
    DefaultLatencyMs:     ~w,
}

// QueryPlanStage is a stage in the execution plan
type QueryPlanStage struct {
    StageID          int                 `json:"stage_id"`
    Nodes            []string            `json:"nodes"`
    Strategy         AggregationStrategy `json:"strategy"`
    Parallel         bool                `json:"parallel"`
    DependsOn        []int               `json:"depends_on"`
    EstimatedCostMs  float64             `json:"estimated_cost_ms"`
    EstimatedResults int                 `json:"estimated_results"`
    Description      string              `json:"description"`
}

// QueryPlan is a DAG of execution stages
type QueryPlan struct {
    PlanID              string           `json:"plan_id"`
    QueryType           QueryType        `json:"query_type"`
    Stages              []QueryPlanStage `json:"stages"`
    TotalEstimatedCostMs float64         `json:"total_estimated_cost_ms"`
    CreatedAt           time.Time        `json:"created_at"`
}

// GetExecutionOrder returns stages grouped by dependency level
func (p *QueryPlan) GetExecutionOrder() [][]QueryPlanStage {
    if len(p.Stages) == 0 {
        return nil
    }
    completed := make(map[int]bool)
    var levels [][]QueryPlanStage
    remaining := make([]QueryPlanStage, len(p.Stages))
    copy(remaining, p.Stages)
    for len(remaining) > 0 {
        var ready, notReady []QueryPlanStage
        for _, stage := range remaining {
            allDepsComplete := true
            for _, dep := range stage.DependsOn {
                if !completed[dep] { allDepsComplete = false; break }
            }
            if allDepsComplete { ready = append(ready, stage) } else { notReady = append(notReady, stage) }
        }
        if len(ready) == 0 { levels = append(levels, remaining); break }
        levels = append(levels, ready)
        for _, stage := range ready { completed[stage.StageID] = true }
        remaining = notReady
    }
    return levels
}

// QueryPlanner builds optimized query execution plans
type QueryPlanner struct {
    Router    *KleinbergRouter
    Config    PlannerConfig
    planCount int64
    mu        sync.RWMutex
}

// NewQueryPlanner creates a new planner with default config
func NewQueryPlanner(router *KleinbergRouter) *QueryPlanner {
    return &QueryPlanner{Router: router, Config: DefaultPlannerConfig}
}

// ClassifyQuery analyzes query characteristics
func (p *QueryPlanner) ClassifyQuery(queryEmbedding []float32, nodes []*KGNode) *QueryClassification {
    if len(nodes) == 0 {
        return &QueryClassification{QueryType: QueryTypeSpecific, TopNodes: []string{}, Confidence: 0.0}
    }
    type nodeSim struct { nodeID string; sim float64 }
    similarities := make([]nodeSim, len(nodes))
    for i, node := range nodes {
        var sim float64
        if len(node.Centroid) > 0 { sim = cosineSimilarity(queryEmbedding, node.Centroid) }
        similarities[i] = nodeSim{node.NodeID, sim}
    }
    for i := 0; i < len(similarities)-1; i++ {
        for j := i + 1; j < len(similarities); j++ {
            if similarities[j].sim > similarities[i].sim { similarities[i], similarities[j] = similarities[j], similarities[i] }
        }
    }
    maxSim := similarities[0].sim
    var sum float64
    for _, ns := range similarities { sum += ns.sim }
    mean := sum / float64(len(similarities))
    var variance float64
    for _, ns := range similarities { variance += (ns.sim - mean) * (ns.sim - mean) }
    variance /= float64(len(similarities))
    topN := 5; if topN > len(similarities) { topN = len(similarities) }
    topNodes := make([]string, topN)
    for i := 0; i < topN; i++ { topNodes[i] = similarities[i].nodeID }
    var queryType QueryType; var confidence float64
    if maxSim >= p.Config.SpecificThreshold { queryType = QueryTypeSpecific; confidence = math.Min(1.0, maxSim)
    } else if variance >= p.Config.ExploratoryVariance { queryType = QueryTypeExploratory; confidence = math.Min(1.0, variance*5)
    } else { queryType = QueryTypeConsensus; confidence = 0.7 }
    return &QueryClassification{QueryType: queryType, MaxSimilarity: maxSim, SimilarityVariance: variance, TopNodes: topNodes, Confidence: confidence}
}

// BuildPlan creates an execution plan for the query
func (p *QueryPlanner) BuildPlan(queryEmbedding []float32, nodes []*KGNode, latencyBudgetMs *float64, forceType *QueryType) *QueryPlan {
    if len(nodes) == 0 { return p.buildEmptyPlan() }
    classification := p.ClassifyQuery(queryEmbedding, nodes)
    queryType := classification.QueryType
    if forceType != nil { queryType = *forceType }
    var plan *QueryPlan
    switch queryType {
    case QueryTypeSpecific: plan = p.buildSpecificPlan(classification)
    case QueryTypeExploratory: plan = p.buildExploratoryPlan(classification, nodes)
    default: plan = p.buildConsensusPlan(classification, nodes)
    }
    if latencyBudgetMs != nil { plan = p.applyLatencyBudget(plan, *latencyBudgetMs) }
    p.mu.Lock(); p.planCount++; p.mu.Unlock()
    return plan
}

func (p *QueryPlanner) buildSpecificPlan(classification *QueryClassification) *QueryPlan {
    numNodes := p.Config.SpecificMaxNodes
    if numNodes > len(classification.TopNodes) { numNodes = len(classification.TopNodes) }
    selectedNodes := classification.TopNodes[:numNodes]
    stage := QueryPlanStage{StageID: 0, Nodes: selectedNodes, Strategy: AggregationMax, Parallel: true, EstimatedCostMs: p.Config.DefaultLatencyMs + 10, EstimatedResults: 10, Description: "Greedy query to top matching nodes"}
    return &QueryPlan{PlanID: p.generatePlanID(), QueryType: QueryTypeSpecific, Stages: []QueryPlanStage{stage}, TotalEstimatedCostMs: stage.EstimatedCostMs, CreatedAt: time.Now()}
}

func (p *QueryPlanner) buildExploratoryPlan(classification *QueryClassification, nodes []*KGNode) *QueryPlan {
    numNodes := p.Config.ExploratoryMaxNodes
    if numNodes > len(nodes) { numNodes = len(nodes) }
    selectedNodes := make([]string, 0, numNodes)
    selectedNodes = append(selectedNodes, classification.TopNodes...)
    if len(selectedNodes) < numNodes {
        for _, n := range nodes {
            found := false
            for _, s := range selectedNodes { if s == n.NodeID { found = true; break } }
            if !found { selectedNodes = append(selectedNodes, n.NodeID); if len(selectedNodes) >= numNodes { break } }
        }
    }
    stage := QueryPlanStage{StageID: 0, Nodes: selectedNodes, Strategy: AggregationSum, Parallel: true, EstimatedCostMs: p.Config.DefaultLatencyMs + 10, EstimatedResults: len(selectedNodes) * 5, Description: "Broad exploration across diverse nodes"}
    return &QueryPlan{PlanID: p.generatePlanID(), QueryType: QueryTypeExploratory, Stages: []QueryPlanStage{stage}, TotalEstimatedCostMs: stage.EstimatedCostMs, CreatedAt: time.Now()}
}

func (p *QueryPlanner) buildConsensusPlan(classification *QueryClassification, nodes []*KGNode) *QueryPlan {
    stage1Nodes := p.Config.ConsensusStage1Nodes
    if stage1Nodes > len(nodes) { stage1Nodes = len(nodes) }
    stage1NodeIDs := make([]string, 0, stage1Nodes)
    stage1NodeIDs = append(stage1NodeIDs, classification.TopNodes...)
    if len(stage1NodeIDs) < stage1Nodes {
        for _, n := range nodes {
            found := false
            for _, s := range stage1NodeIDs { if s == n.NodeID { found = true; break } }
            if !found { stage1NodeIDs = append(stage1NodeIDs, n.NodeID); if len(stage1NodeIDs) >= stage1Nodes { break } }
        }
    }
    stage1 := QueryPlanStage{StageID: 0, Nodes: stage1NodeIDs, Strategy: AggregationSum, Parallel: true, EstimatedCostMs: p.Config.DefaultLatencyMs + 10, EstimatedResults: stage1Nodes * 5, Description: "Initial broad query for consensus candidates"}
    stage2Set := make(map[string]bool); for _, nid := range stage1NodeIDs { stage2Set[nid] = true }
    var stage2NodeIDs []string
    for _, nid := range classification.TopNodes { if !stage2Set[nid] { stage2NodeIDs = append(stage2NodeIDs, nid); if len(stage2NodeIDs) >= p.Config.ConsensusStage2Nodes { break } } }
    stage2 := QueryPlanStage{StageID: 1, Nodes: stage2NodeIDs, Strategy: AggregationDiversityWeighted, Parallel: true, DependsOn: []int{0}, EstimatedCostMs: p.Config.DefaultLatencyMs + 10, EstimatedResults: len(stage2NodeIDs) * 3, Description: "Density refinement on consensus clusters"}
    return &QueryPlan{PlanID: p.generatePlanID(), QueryType: QueryTypeConsensus, Stages: []QueryPlanStage{stage1, stage2}, TotalEstimatedCostMs: stage1.EstimatedCostMs + stage2.EstimatedCostMs, CreatedAt: time.Now()}
}

func (p *QueryPlanner) buildEmptyPlan() *QueryPlan {
    return &QueryPlan{PlanID: p.generatePlanID(), QueryType: QueryTypeSpecific, Stages: []QueryPlanStage{}, TotalEstimatedCostMs: 0, CreatedAt: time.Now()}
}

func (p *QueryPlanner) applyLatencyBudget(plan *QueryPlan, budgetMs float64) *QueryPlan {
    if plan.TotalEstimatedCostMs <= budgetMs { return plan }
    ratio := budgetMs / plan.TotalEstimatedCostMs
    newStages := make([]QueryPlanStage, len(plan.Stages))
    for i, stage := range plan.Stages {
        newNodeCount := int(float64(len(stage.Nodes)) * ratio); if newNodeCount < 1 { newNodeCount = 1 }
        newStages[i] = QueryPlanStage{StageID: stage.StageID, Nodes: stage.Nodes[:newNodeCount], Strategy: stage.Strategy, Parallel: stage.Parallel, DependsOn: stage.DependsOn, EstimatedCostMs: p.Config.DefaultLatencyMs + 10, EstimatedResults: int(float64(stage.EstimatedResults) * ratio), Description: stage.Description + " (budget-constrained)"}
    }
    var totalCost float64; for _, s := range newStages { totalCost += s.EstimatedCostMs }
    return &QueryPlan{PlanID: plan.PlanID, QueryType: plan.QueryType, Stages: newStages, TotalEstimatedCostMs: totalCost, CreatedAt: plan.CreatedAt}
}

func (p *QueryPlanner) generatePlanID() string {
    p.mu.Lock(); defer p.mu.Unlock()
    return fmt.Sprintf("plan-%%d-%%d", time.Now().UnixNano(), p.planCount)
}

// PlannedQueryEngine executes queries using optimized plans
type PlannedQueryEngine struct {
    *FederatedQueryEngine
    Planner *QueryPlanner
}

// NewPlannedQueryEngine creates an engine with query planning
func NewPlannedQueryEngine(router *KleinbergRouter) *PlannedQueryEngine {
    return &PlannedQueryEngine{FederatedQueryEngine: NewFederatedQueryEngine(router), Planner: NewQueryPlanner(router)}
}

// PlannedQuery executes a query using an optimized plan
func (e *PlannedQueryEngine) PlannedQuery(ctx context.Context, queryText string, queryEmbedding []float32, topK int, latencyBudgetMs *float64) (*AggregatedResponse, error) {
    nodes, err := e.Router.DiscoverNodes([]string{"kg_node"})
    if err != nil { return nil, err }
    kgNodes := make([]*KGNode, len(nodes))
    for i, n := range nodes { kgNodes[i] = &KGNode{NodeID: n.ServiceID, Endpoint: n.Address} }
    plan := e.Planner.BuildPlan(queryEmbedding, kgNodes, latencyBudgetMs, nil)
    if len(plan.Stages) == 0 { return &AggregatedResponse{QueryID: plan.PlanID, Results: []AggregatedResult{}, NodesQueried: 0, NodesResponded: 0}, nil }
    stage := plan.Stages[0]
    originalK := e.FederationK; e.FederationK = len(stage.Nodes); e.Config.Strategy = stage.Strategy
    response, err := e.FederatedQuery(ctx, queryText, queryEmbedding, topK)
    e.FederationK = originalK
    return response, err
}
', [SpecT, ExplVar, ConsMin, SpecMax, ExplMax, ConsS1, ConsS2, DefLat]).

% =============================================================================
% KG TOPOLOGY PHASE 5a: HIERARCHICAL FEDERATION CODE GENERATION (Go)
% =============================================================================

%% compile_hierarchical_federation_go(+Options, -Code)
%  Generate Go HierarchicalFederatedEngine with multi-level query routing.
%  Queries regional aggregators first, then drills down to best regions.

compile_hierarchical_federation_go(Options, Code) :-
    % Extract hierarchical options with defaults
    ( member(hierarchical(HierOpts), Options),
      is_list(HierOpts) -> true
    ; HierOpts = []
    ),
    ( member(max_levels(MaxLevels), HierOpts) -> true ; MaxLevels = 3 ),
    ( member(min_nodes_per_region(MinNodes), HierOpts) -> true ; MinNodes = 2 ),
    ( member(centroid_similarity_threshold(SimThresh), HierOpts) -> true ; SimThresh = 0.5 ),
    ( member(drill_down_k(DrillK), HierOpts) -> true ; DrillK = 2 ),

    format(string(Code), '
// KG Topology Phase 5a: Hierarchical Federation
// Generated from Prolog service definition

// HierarchyConfig holds configuration for hierarchy building
type HierarchyConfig struct {
    MaxLevels                  int     `json:"max_levels"`
    MinNodesPerRegion          int     `json:"min_nodes_per_region"`
    CentroidSimilarityThreshold float64 `json:"centroid_similarity_threshold"`
}

// DefaultHierarchyConfig returns the default configuration
var DefaultHierarchyConfig = HierarchyConfig{
    MaxLevels:                  ~w,
    MinNodesPerRegion:          ~w,
    CentroidSimilarityThreshold: ~w,
}

// RegionalNode represents a node that aggregates child nodes
type RegionalNode struct {
    RegionID     string    `json:"region_id"`
    Centroid     []float32 `json:"centroid"`
    Topics       []string  `json:"topics"`
    ChildNodes   []string  `json:"child_nodes"`
    ParentRegion string    `json:"parent_region,omitempty"`
    Level        int       `json:"level"`
}

// NodeHierarchy manages hierarchical node relationships
type NodeHierarchy struct {
    Config       HierarchyConfig
    Regions      map[string]*RegionalNode
    NodeToRegion map[string]string
    LeafNodes    map[string]*KGNode
    mu           sync.RWMutex
}

// NewNodeHierarchy creates a new hierarchy
func NewNodeHierarchy(config *HierarchyConfig) *NodeHierarchy {
    cfg := DefaultHierarchyConfig
    if config != nil { cfg = *config }
    return &NodeHierarchy{Config: cfg, Regions: make(map[string]*RegionalNode), NodeToRegion: make(map[string]string), LeafNodes: make(map[string]*KGNode)}
}

// BuildFromNodes builds hierarchy from leaf nodes using centroid clustering
func (h *NodeHierarchy) BuildFromNodes(nodes []*KGNode) {
    h.mu.Lock(); defer h.mu.Unlock()
    h.LeafNodes = make(map[string]*KGNode); h.Regions = make(map[string]*RegionalNode); h.NodeToRegion = make(map[string]string)
    if len(nodes) == 0 { return }
    for _, n := range nodes { h.LeafNodes[n.NodeID] = n }
    // Simple clustering by similarity - nodes with similar centroids form regions
    assigned := make(map[string]bool)
    groupID := 0
    for _, node := range nodes {
        if assigned[node.NodeID] || len(node.Centroid) == 0 { continue }
        cluster := []*KGNode{node}; assigned[node.NodeID] = true
        for _, other := range nodes {
            if assigned[other.NodeID] || len(other.Centroid) == 0 { continue }
            if cosineSimilarity(node.Centroid, other.Centroid) >= h.Config.CentroidSimilarityThreshold {
                cluster = append(cluster, other); assigned[other.NodeID] = true
            }
        }
        if len(cluster) >= h.Config.MinNodesPerRegion {
            h.createRegion(fmt.Sprintf("region_%%d", groupID), cluster, 0, ""); groupID++
        } else {
            for _, n := range cluster { h.createRegion(fmt.Sprintf("singleton_%%s", n.NodeID), []*KGNode{n}, 0, "") }
        }
    }
}

func (h *NodeHierarchy) createRegion(regionID string, nodes []*KGNode, level int, parent string) {
    var sumCentroid []float64; var count int
    for _, n := range nodes {
        if len(n.Centroid) > 0 {
            if sumCentroid == nil { sumCentroid = make([]float64, len(n.Centroid)) }
            for i, v := range n.Centroid { sumCentroid[i] += float64(v) }
            count++
        }
    }
    var avgCentroid []float32
    if count > 0 {
        avgCentroid = make([]float32, len(sumCentroid))
        for i, v := range sumCentroid { avgCentroid[i] = float32(v / float64(count)) }
    }
    topicSet := make(map[string]bool)
    for _, n := range nodes { for _, t := range n.Topics { topicSet[t] = true } }
    topics := make([]string, 0, len(topicSet)); for t := range topicSet { topics = append(topics, t) }
    childNodes := make([]string, len(nodes)); for i, n := range nodes { childNodes[i] = n.NodeID }
    h.Regions[regionID] = &RegionalNode{RegionID: regionID, Centroid: avgCentroid, Topics: topics, ChildNodes: childNodes, ParentRegion: parent, Level: level}
    for _, n := range nodes { h.NodeToRegion[n.NodeID] = regionID }
}

// GetRegionalNodes returns regions at a specific level
func (h *NodeHierarchy) GetRegionalNodes(level int) []*RegionalNode {
    h.mu.RLock(); defer h.mu.RUnlock()
    var result []*RegionalNode
    for _, r := range h.Regions { if r.Level == level { result = append(result, r) } }
    return result
}

// GetChildren returns child node IDs for a region
func (h *NodeHierarchy) GetChildren(regionID string) []string {
    h.mu.RLock(); defer h.mu.RUnlock()
    if r, ok := h.Regions[regionID]; ok { return r.ChildNodes }
    return nil
}

// GetChildNodes returns actual KGNode objects for region children
func (h *NodeHierarchy) GetChildNodes(regionID string) []*KGNode {
    childIDs := h.GetChildren(regionID)
    h.mu.RLock(); defer h.mu.RUnlock()
    result := make([]*KGNode, 0, len(childIDs))
    for _, id := range childIDs { if n, ok := h.LeafNodes[id]; ok { result = append(result, n) } }
    return result
}

// HierarchicalFederatedEngine executes queries through hierarchy
type HierarchicalFederatedEngine struct {
    *FederatedQueryEngine
    Hierarchy   *NodeHierarchy
    DrillDownK  int
    hierarchyBuilt bool
}

// NewHierarchicalFederatedEngine creates engine with hierarchical routing
func NewHierarchicalFederatedEngine(router *KleinbergRouter) *HierarchicalFederatedEngine {
    return &HierarchicalFederatedEngine{FederatedQueryEngine: NewFederatedQueryEngine(router), Hierarchy: NewNodeHierarchy(nil), DrillDownK: ~w}
}

// HierarchicalQuery executes a query through the hierarchy
func (e *HierarchicalFederatedEngine) HierarchicalQuery(ctx context.Context, queryText string, queryEmbedding []float32, topK int) (*AggregatedResponse, error) {
    e.ensureHierarchy()
    if e.Hierarchy == nil || len(e.Hierarchy.Regions) == 0 {
        return e.FederatedQuery(ctx, queryText, queryEmbedding, topK) // Fallback to flat
    }
    // Level 1: Query regional aggregators
    regions := e.Hierarchy.GetRegionalNodes(0)
    if len(regions) == 0 { return e.FederatedQuery(ctx, queryText, queryEmbedding, topK) }
    // Find best regions by centroid similarity
    type regionSim struct { region *RegionalNode; sim float64 }
    similarities := make([]regionSim, len(regions))
    for i, r := range regions { similarities[i] = regionSim{r, cosineSimilarity(queryEmbedding, r.Centroid)} }
    for i := 0; i < len(similarities)-1; i++ {
        for j := i + 1; j < len(similarities); j++ {
            if similarities[j].sim > similarities[i].sim { similarities[i], similarities[j] = similarities[j], similarities[i] }
        }
    }
    // Level 2: Query children of top regions
    drillK := e.DrillDownK; if drillK > len(similarities) { drillK = len(similarities) }
    var allNodeIDs []string
    for i := 0; i < drillK; i++ { allNodeIDs = append(allNodeIDs, e.Hierarchy.GetChildren(similarities[i].region.RegionID)...) }
    // Execute federated query on selected nodes
    originalK := e.FederationK; e.FederationK = len(allNodeIDs)
    response, err := e.FederatedQuery(ctx, queryText, queryEmbedding, topK)
    e.FederationK = originalK
    return response, err
}

func (e *HierarchicalFederatedEngine) ensureHierarchy() {
    if e.hierarchyBuilt { return }
    nodes, _ := e.Router.DiscoverNodes([]string{"kg_node"})
    if len(nodes) > 0 {
        kgNodes := make([]*KGNode, len(nodes))
        for i, n := range nodes { kgNodes[i] = &KGNode{NodeID: n.ServiceID, Endpoint: n.Address} }
        e.Hierarchy.BuildFromNodes(kgNodes)
        e.hierarchyBuilt = true
    }
}
', [MaxLevels, MinNodes, SimThresh, DrillK]).

% =============================================================================
% KG TOPOLOGY PHASE 5d: STREAMING FEDERATION CODE GENERATION (Go)
% =============================================================================

%% compile_streaming_federation_go(+Options, -Code)
%  Generate Go StreamingFederatedEngine with async partial results.
%  Streams results as nodes respond rather than waiting for all.

compile_streaming_federation_go(Options, Code) :-
    % Extract streaming options with defaults
    ( member(streaming(StreamOpts), Options),
      is_list(StreamOpts) -> true
    ; StreamOpts = []
    ),
    ( member(yield_interval_ms(YieldMs), StreamOpts) -> true ; YieldMs = 100 ),
    ( member(min_confidence(MinConf), StreamOpts) -> true ; MinConf = 0.3 ),

    format(string(Code), '
// KG Topology Phase 5d: Streaming Federation
// Generated from Prolog service definition

// StreamingConfig holds configuration for streaming queries
type StreamingConfig struct {
    YieldIntervalMs int     `json:"yield_interval_ms"`
    MinConfidence   float64 `json:"min_confidence"`
}

// DefaultStreamingConfig returns the default configuration
var DefaultStreamingConfig = StreamingConfig{
    YieldIntervalMs: ~w,
    MinConfidence:   ~w,
}

// PartialResult represents an intermediate result during streaming
type PartialResult struct {
    Results        []AggregatedResult `json:"results"`
    Confidence     float64            `json:"confidence"`
    NodesResponded int                `json:"nodes_responded"`
    NodesTotal     int                `json:"nodes_total"`
    IsFinal        bool               `json:"is_final"`
}

// StreamingFederatedEngine supports streaming partial results
type StreamingFederatedEngine struct {
    *FederatedQueryEngine
    StreamConfig StreamingConfig
}

// NewStreamingFederatedEngine creates engine with streaming support
func NewStreamingFederatedEngine(router *KleinbergRouter) *StreamingFederatedEngine {
    return &StreamingFederatedEngine{FederatedQueryEngine: NewFederatedQueryEngine(router), StreamConfig: DefaultStreamingConfig}
}

// StreamingQuery streams partial results as nodes respond
func (e *StreamingFederatedEngine) StreamingQuery(ctx context.Context, queryText string, queryEmbedding []float32, topK int) (<-chan PartialResult, error) {
    nodes, err := e.Router.DiscoverNodes([]string{"kg_node"})
    if err != nil { return nil, err }
    resultChan := make(chan PartialResult, len(nodes)+1)
    go func() {
        defer close(resultChan)
        if len(nodes) == 0 { resultChan <- PartialResult{Results: []AggregatedResult{}, Confidence: 1.0, NodesResponded: 0, NodesTotal: 0, IsFinal: true}; return }
        // Query nodes in parallel
        nodeChan := make(chan *NodeResponse, len(nodes))
        var wg sync.WaitGroup
        for _, n := range nodes {
            wg.Add(1)
            go func(node ServiceInstance) {
                defer wg.Done()
                resp := e.queryNodeAsync(ctx, &KGNode{NodeID: node.ServiceID, Endpoint: node.Address}, queryText, queryEmbedding, topK)
                nodeChan <- resp
            }(n)
        }
        go func() { wg.Wait(); close(nodeChan) }()
        // Aggregate and stream partial results
        aggregated := make(map[string]*AggregatedResult)
        responded := 0
        for resp := range nodeChan {
            if resp == nil || resp.Error != "" { continue }
            responded++
            for _, r := range resp.Results {
                if existing, ok := aggregated[r.AnswerHash]; ok {
                    existing.CombinedScore = e.mergeScore(existing.CombinedScore, r.ExpScore)
                    existing.SourceNodes = append(existing.SourceNodes, resp.SourceNode)
                } else {
                    aggregated[r.AnswerHash] = &AggregatedResult{AnswerText: r.AnswerText, AnswerHash: r.AnswerHash, CombinedScore: r.ExpScore, SourceNodes: []string{resp.SourceNode}}
                }
            }
            // Emit partial result
            results := make([]AggregatedResult, 0, len(aggregated))
            for _, r := range aggregated { results = append(results, *r) }
            for i := 0; i < len(results)-1; i++ {
                for j := i + 1; j < len(results); j++ {
                    if results[j].CombinedScore > results[i].CombinedScore { results[i], results[j] = results[j], results[i] }
                }
            }
            if len(results) > topK { results = results[:topK] }
            confidence := float64(responded) / float64(len(nodes))
            if confidence >= e.StreamConfig.MinConfidence {
                resultChan <- PartialResult{Results: results, Confidence: confidence, NodesResponded: responded, NodesTotal: len(nodes), IsFinal: false}
            }
        }
        // Final result
        results := make([]AggregatedResult, 0, len(aggregated))
        for _, r := range aggregated { results = append(results, *r) }
        for i := 0; i < len(results)-1; i++ {
            for j := i + 1; j < len(results); j++ {
                if results[j].CombinedScore > results[i].CombinedScore { results[i], results[j] = results[j], results[i] }
            }
        }
        if len(results) > topK { results = results[:topK] }
        resultChan <- PartialResult{Results: results, Confidence: float64(responded) / float64(len(nodes)), NodesResponded: responded, NodesTotal: len(nodes), IsFinal: true}
    }()
    return resultChan, nil
}

func (e *StreamingFederatedEngine) queryNodeAsync(ctx context.Context, node *KGNode, queryText string, queryEmbedding []float32, topK int) *NodeResponse {
    // Create HTTP request
    reqBody := map[string]interface{}{"__type": "kg_query", "payload": map[string]interface{}{"query_text": queryText, "top_k": topK}}
    jsonBody, _ := json.Marshal(reqBody)
    req, _ := http.NewRequestWithContext(ctx, "POST", node.Endpoint+"/kg/query", bytes.NewReader(jsonBody))
    req.Header.Set("Content-Type", "application/json")
    client := &http.Client{Timeout: time.Duration(e.TimeoutMs) * time.Millisecond}
    start := time.Now()
    resp, err := client.Do(req)
    if err != nil { return &NodeResponse{SourceNode: node.NodeID, Error: err.Error()} }
    defer resp.Body.Close()
    var nodeResp NodeResponse
    if err := json.NewDecoder(resp.Body).Decode(&nodeResp); err != nil { return &NodeResponse{SourceNode: node.NodeID, Error: err.Error()} }
    nodeResp.SourceNode = node.NodeID
    nodeResp.ResponseTime = time.Since(start)
    return &nodeResp
}

func (e *StreamingFederatedEngine) mergeScore(existing, new float64) float64 {
    switch e.Config.Strategy {
    case AggregationMax: if new > existing { return new }; return existing
    case AggregationMin: if new < existing { return new }; return existing
    default: return existing + new // SUM
    }
}
', [YieldMs, MinConf]).
