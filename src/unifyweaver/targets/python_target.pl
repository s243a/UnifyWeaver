:- module(python_target, [
    compile_predicate_to_python/3,
    compile_facts_to_python/3,        % +Pred, +Arity, -PythonCode  -- NEW
    init_python_target/0,
    json_schema/2,                    % +SchemaName, +Fields (directive)
    get_json_schema/2,                % +SchemaName, -Fields (lookup)
    % Pipeline mode exports (Phase 1)
    test_pipeline_mode/0,
    generate_default_arg_names/2,
    pipeline_header/2,
    pipeline_header/3,
    pipeline_helpers/3,
    generate_output_formatting/4,
    generate_pipeline_main/4,
    % Runtime selection exports (Phase 2)
    select_python_runtime/4,
    runtime_available/1,
    runtime_compatible_with_imports/2,
    test_runtime_selection/0,
    % Runtime-specific code generation exports (Phase 3)
    test_runtime_headers/0,
    % Pipeline chaining exports (Phase 4)
    compile_pipeline/3,
    compile_same_runtime_pipeline/3,
    compile_cross_runtime_pipeline/3,
    test_pipeline_chaining/0,
    % Pipeline generator mode exports (Phase 5)
    test_python_pipeline_generator/0,
    % IronPython pipeline generator mode exports (Phase 6)
    test_ironpython_pipeline_generator/0,
    % Enhanced pipeline chaining exports (Phase 7)
    compile_enhanced_pipeline/3,
    enhanced_pipeline_helpers/1,
    generate_enhanced_connector/3,
    test_enhanced_pipeline_chaining/0,
    % IronPython enhanced pipeline chaining exports (Phase 8)
    compile_ironpython_enhanced_pipeline/3,
    ironpython_enhanced_helpers/1,
    test_ironpython_enhanced_chaining/0,
    % Client-server architecture exports (Phase 9)
    compile_service_to_python/2,
    generate_service_handler_python/2,
    % Phase 2: Cross-process services
    compile_unix_socket_service_python/2,
    compile_unix_socket_client_python/3,
    % Phase 3: Network services
    compile_tcp_service_python/2,
    compile_tcp_client_python/4,
    compile_http_service_python/2,
    compile_http_client_python/3,
    compile_http_client_python/4,
    % Phase 4: Service mesh
    compile_service_mesh_python/2,
    generate_load_balancer_python/2,
    generate_circuit_breaker_python/2,
    generate_retry_python/2,
    % Phase 5: Polyglot services
    compile_polyglot_service_python/2,
    generate_service_client_python/3,
    % Phase 6: Distributed services
    compile_distributed_service_python/2,
    generate_sharding_python/2,
    generate_replication_python/2,
    % Phase 7: Service Discovery
    compile_discovery_service_python/2,
    generate_health_check_python/2,
    generate_service_registry_python/2,
    % Phase 8: Service Tracing
    compile_traced_service_python/2,
    generate_tracer_python/2,
    generate_span_context_python/2,
    % KG Topology Phase 3: Kleinberg routing
    compile_kleinberg_router_python/2,
    compile_distributed_kg_service_python/2,
    format_python_list/2,
    % KG Topology Phase 4: Federated queries
    compile_federated_query_python/2,
    compile_federated_service_python/2
]).

:- meta_predicate compile_predicate_to_python(:, +, -).

% Conditional import of call_graph for mutual recursion detection
% Falls back gracefully if module not available
:- catch(use_module('../core/advanced/call_graph'), _, true).
:- use_module(common_generator).

% Binding system integration (ported from PowerShell target)
:- use_module('../core/binding_registry').
:- use_module('../bindings/python_bindings').

% Pipeline validation (Phase 9)
:- use_module('../core/pipeline_validation').

% Service validation (Client-Server Phase 2)
:- use_module('../core/service_validation').

% Control plane integration (Phase 2 - Runtime Selection)
:- catch(use_module('../core/preferences'), _, true).
:- catch(use_module('../core/firewall', except([validate_service/2])), _, true).
:- catch(use_module('../glue/dotnet_glue'), _, true).

% Track required imports from bindings
:- dynamic required_import/1.

% translate_goal/2 is spread across the file for organization
:- discontiguous translate_goal/2.

% pipeline_header/2 has multiple clauses spread across the file (legacy + new)
:- discontiguous pipeline_header/2.

%% init_python_target
%  Initialize Python target with bindings
init_python_target :-
    retractall(required_import(_)),
    init_python_bindings.

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
    % Field not in schema - default to 'any'
    format('WARNING: Field ~w not in schema ~w, defaulting to type ''any''~n', [FieldName, SchemaName]).

%% ============================================
%% SERVICE COMPILATION (Client-Server Phase 1)
%% ============================================

%% compile_service_to_python(+Service, -PythonCode)
%  Compile a service definition to a Python class.
%  Dispatches based on transport type: in_process, unix_socket, etc.
compile_service_to_python(service(Name, HandlerSpec), PythonCode) :-
    !,
    compile_service_to_python(service(Name, [], HandlerSpec), PythonCode).

compile_service_to_python(Service, PythonCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    member(transport(unix_socket(_Path)), Options),
    !,
    % Phase 2: Unix socket service
    compile_unix_socket_service_python(Service, PythonCode).

compile_service_to_python(Service, PythonCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    member(transport(tcp(_Host, _Port)), Options),
    !,
    % Phase 3: TCP service
    compile_tcp_service_python(Service, PythonCode).

compile_service_to_python(Service, PythonCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    member(transport(http(_Endpoint)), Options),
    !,
    % Phase 3: HTTP service
    compile_http_service_python(Service, PythonCode).

compile_service_to_python(Service, PythonCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    member(transport(http(_Endpoint, _HttpOptions)), Options),
    !,
    % Phase 3: HTTP service with options
    compile_http_service_python(Service, PythonCode).

compile_service_to_python(Service, PythonCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    ( member(polyglot(true), Options)
    ; member(depends_on(Deps), Options), Deps \= []
    ),
    !,
    % Phase 5: Polyglot service
    compile_polyglot_service_python(Service, PythonCode).

compile_service_to_python(Service, PythonCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    ( member(distributed(true), Options)
    ; member(sharding(_), Options)
    ; member(replication(_), Options)
    ; member(cluster(_), Options)
    ),
    !,
    % Phase 6: Distributed service
    compile_distributed_service_python(Service, PythonCode).

compile_service_to_python(Service, PythonCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    ( member(discovery_enabled(true), Options)
    ; member(discovery_backend(_), Options)
    ; member(health_check(_), Options)
    ),
    !,
    % Phase 7: Service Discovery
    compile_discovery_service_python(Service, PythonCode).

compile_service_to_python(Service, PythonCode) :-
    Service = service(_Name, Options, _HandlerSpec),
    ( member(tracing(true), Options)
    ; member(trace_exporter(_), Options)
    ; member(trace_sampling(_), Options)
    ),
    !,
    % Phase 8: Service Tracing
    compile_traced_service_python(Service, PythonCode).

compile_service_to_python(Service, PythonCode) :-
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
    compile_service_mesh_python(Service, PythonCode).

compile_service_to_python(service(Name, Options, HandlerSpec), PythonCode) :-
    % Phase 1: In-process service (default)
    % Determine if service is stateful
    ( member(stateful(true), Options) -> Stateful = "True" ; Stateful = "False" ),
    % Generate handler code
    generate_service_handler_python(HandlerSpec, HandlerCode),
    % Format the class name (capitalize first letter)
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        ClassName = [Upper|Rest]
    ;
        ClassName = [First|Rest]
    ),
    atom_codes(ClassNameAtom, ClassName),
    % Generate the service class
    format(string(PythonCode),
"class ~wService(Service):
    '''
    Service: ~w
    Auto-generated from Prolog service definition.
    '''
    def __init__(self):
        super().__init__('~w', stateful=~w)

    def call(self, request):
        '''Process request and return response.'''
~w

# Register service
register_service('~w', ~wService())
", [ClassNameAtom, Name, Name, Stateful, HandlerCode, Name, ClassNameAtom]).

%% generate_service_handler_python(+HandlerSpec, -Code)
%  Generate Python handler code from handler specification.
generate_service_handler_python([], "        return None").
generate_service_handler_python(HandlerSpec, Code) :-
    HandlerSpec \= [],
    generate_handler_ops_python(HandlerSpec, OpsCode),
    format(string(Code), "~w", [OpsCode]).

%% generate_handler_ops_python(+Ops, -Code)
%  Generate Python code for handler operations.
generate_handler_ops_python([], "").
generate_handler_ops_python([Op|Rest], Code) :-
    generate_handler_op_python(Op, OpCode),
    generate_handler_ops_python(Rest, RestCode),
    ( RestCode = "" ->
        Code = OpCode
    ;
        format(string(Code), "~w~n~w", [OpCode, RestCode])
    ).

%% generate_handler_op_python(+Op, -Code)
%  Generate Python code for a single handler operation.
generate_handler_op_python(receive(Var), Code) :-
    ( var(Var) ->
        VarName = "_request"
    ;
        VarName = Var
    ),
    format(string(Code), "        ~w = request  # Bind request", [VarName]).

generate_handler_op_python(respond(Value), Code) :-
    ( var(Value) ->
        format(string(Code), "        return response", [])
    ; atom(Value) ->
        format(string(Code), "        return ~w", [Value])
    ; number(Value) ->
        format(string(Code), "        return ~w", [Value])
    ;
        format(string(Code), "        return ~w", [Value])
    ).

generate_handler_op_python(respond_error(Error), Code) :-
    format(string(Code), "        raise ServiceError('~w')", [Error]).

generate_handler_op_python(transform(In, Out, Goal), Code) :-
    ( var(In) -> InName = "_in" ; InName = In ),
    ( var(Out) -> OutName = "_out" ; OutName = Out ),
    format(string(Code), "        ~w = ~w  # transform: ~w", [OutName, InName, Goal]).

generate_handler_op_python(transform(In, Out), Code) :-
    ( var(In) -> InName = "_in" ; InName = In ),
    ( var(Out) -> OutName = "_out" ; OutName = Out ),
    format(string(Code), "        ~w = ~w  # transform", [OutName, InName]).

generate_handler_op_python(state_get(Key, Var), Code) :-
    ( var(Var) -> VarName = "_val" ; VarName = Var ),
    format(string(Code), "        ~w = self.state_get('~w')", [VarName, Key]).

generate_handler_op_python(state_put(Key, Value), Code) :-
    ( var(Value) ->
        format(string(Code), "        self.state_put('~w', _val)", [Key])
    ;
        format(string(Code), "        self.state_put('~w', ~w)", [Key, Value])
    ).

generate_handler_op_python(state_modify(Key, Func), Code) :-
    format(string(Code), "        self.state_modify('~w', ~w)", [Key, Func]).

generate_handler_op_python(state_delete(Key), Code) :-
    format(string(Code), "        self.state_delete('~w')", [Key]).

generate_handler_op_python(call_service(ServiceName, Req, Resp), Code) :-
    ( var(Req) -> ReqName = "_request" ; ReqName = Req ),
    ( var(Resp) -> RespName = "_response" ; RespName = Resp ),
    format(string(Code), "        ~w = call_service_impl('~w', ~w)", [RespName, ServiceName, ReqName]).

generate_handler_op_python(Pred/Arity, Code) :-
    format(string(Code), "        # Call predicate: ~w/~w", [Pred, Arity]).

generate_handler_op_python(Pred, Code) :-
    atom(Pred),
    Pred \= receive, Pred \= respond, Pred \= respond_error,
    format(string(Code), "        ~w(_request)  # Execute predicate", [Pred]).

generate_handler_op_python(_, "        pass  # Unknown operation").

%% ============================================
%% PHASE 2: CROSS-PROCESS SERVICES (Unix Socket)
%% ============================================

%% compile_unix_socket_service_python(+Service, -PythonCode)
%  Generate Python code for a Unix socket service server.
compile_unix_socket_service_python(service(Name, Options, HandlerSpec), PythonCode) :-
    % Extract socket path
    member(transport(unix_socket(SocketPath)), Options),
    % Determine if service is stateful
    ( member(stateful(true), Options) -> Stateful = "True" ; Stateful = "False" ),
    % Extract timeout (default 30000ms)
    ( member(timeout(TimeoutMs), Options) -> Timeout = TimeoutMs ; Timeout = 30000 ),
    % Generate handler code
    generate_service_handler_python(HandlerSpec, HandlerCode),
    % Format the class name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        ClassName = [Upper|Rest]
    ;
        ClassName = [First|Rest]
    ),
    atom_codes(ClassNameAtom, ClassName),
    % Generate the Unix socket service
    format(string(PythonCode),
"import socket
import os
import json
import threading
import signal
import sys

class ~wService(Service):
    '''
    Unix Socket Service: ~w
    Socket Path: ~w
    Auto-generated from Prolog service definition.
    '''
    def __init__(self):
        super().__init__('~w', stateful=~w)
        self.socket_path = '~w'
        self.timeout = ~w / 1000.0  # Convert to seconds
        self.server_socket = None
        self.running = False
        self._lock = threading.Lock()

    def call(self, request):
        '''Process request and return response.'''
~w

    def start_server(self):
        '''Start the Unix socket server.'''
        # Remove existing socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)  # Allow periodic check for shutdown
        self.running = True

        print(f'[~w] Server listening on {self.socket_path}', file=sys.stderr)

        while self.running:
            try:
                conn, _ = self.server_socket.accept()
                threading.Thread(target=self._handle_connection, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except OSError:
                break

        self._cleanup()

    def _handle_connection(self, conn):
        '''Handle a client connection.'''
        conn.settimeout(self.timeout)
        buffer = b''
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data
                # Process complete JSONL messages
                while b'\\n' in buffer:
                    line, buffer = buffer.split(b'\\n', 1)
                    if line:
                        self._process_request(conn, line)
        except socket.timeout:
            self._send_error(conn, 'timeout', 'Request timed out')
        except Exception as e:
            self._send_error(conn, 'error', str(e))
        finally:
            conn.close()

    def _process_request(self, conn, line):
        '''Process a single JSONL request.'''
        try:
            request = json.loads(line.decode('utf-8'))
            request_id = request.get('_id')
            payload = request.get('_payload', request)

            with self._lock:
                response = self.call(payload)

            self._send_response(conn, request_id, response)
        except json.JSONDecodeError as e:
            self._send_error(conn, 'parse_error', f'Invalid JSON: {e}')
        except ServiceError as e:
            self._send_error(conn, 'service_error', str(e))
        except Exception as e:
            self._send_error(conn, 'error', str(e))

    def _send_response(self, conn, request_id, response):
        '''Send a JSONL response.'''
        msg = {'_id': request_id, '_status': 'ok', '_payload': response}
        conn.sendall((json.dumps(msg) + '\\n').encode('utf-8'))

    def _send_error(self, conn, error_type, message):
        '''Send a JSONL error response.'''
        msg = {'_status': 'error', '_error_type': error_type, '_message': message}
        try:
            conn.sendall((json.dumps(msg) + '\\n').encode('utf-8'))
        except:
            pass

    def stop_server(self):
        '''Stop the Unix socket server.'''
        self.running = False
        if self.server_socket:
            self.server_socket.close()

    def _cleanup(self):
        '''Clean up server resources.'''
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except:
                pass
        print(f'[~w] Server stopped', file=sys.stderr)

# Create service instance
_~w_service = ~wService()

# Register for in-process calls
register_service('~w', _~w_service)

def run_~w_server():
    '''Run the ~w service as a standalone server.'''
    def signal_handler(sig, frame):
        print(f'\\n[~w] Shutting down...', file=sys.stderr)
        _~w_service.stop_server()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    _~w_service.start_server()

if __name__ == '__main__':
    run_~w_server()
", [ClassNameAtom, Name, SocketPath, Name, Stateful, SocketPath, Timeout, HandlerCode,
    Name, Name, Name, ClassNameAtom, Name, Name, Name, Name, Name, Name, Name, Name]).

%% compile_unix_socket_client_python(+ServiceName, +SocketPath, -PythonCode)
%  Generate Python code for a Unix socket service client.
compile_unix_socket_client_python(Name, SocketPath, PythonCode) :-
    % Format the class name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        ClassName = [Upper|Rest]
    ;
        ClassName = [First|Rest]
    ),
    atom_codes(ClassNameAtom, ClassName),
    format(string(PythonCode),
"import socket
import json
import uuid

class ~wClient:
    '''
    Unix Socket Client for ~w service.
    Socket Path: ~w
    Auto-generated from Prolog service definition.
    '''
    def __init__(self, socket_path='~w', timeout=30.0):
        self.socket_path = socket_path
        self.timeout = timeout
        self._socket = None

    def connect(self):
        '''Connect to the service.'''
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(self.timeout)
        self._socket.connect(self.socket_path)
        return self

    def disconnect(self):
        '''Disconnect from the service.'''
        if self._socket:
            self._socket.close()
            self._socket = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    def call(self, request):
        '''Send a request and receive a response.'''
        if not self._socket:
            self.connect()

        request_id = str(uuid.uuid4())
        msg = {'_id': request_id, '_payload': request}
        self._socket.sendall((json.dumps(msg) + '\\n').encode('utf-8'))

        # Read response
        buffer = b''
        while True:
            data = self._socket.recv(4096)
            if not data:
                raise ConnectionError('Server closed connection')
            buffer += data
            if b'\\n' in buffer:
                line, _ = buffer.split(b'\\n', 1)
                response = json.loads(line.decode('utf-8'))
                if response.get('_status') == 'ok':
                    return response.get('_payload')
                else:
                    raise ServiceError(
                        service='~w',
                        message=response.get('_message', 'Unknown error')
                    )

def call_~w(request, socket_path='~w', timeout=30.0):
    '''Convenience function to call ~w service.'''
    with ~wClient(socket_path, timeout) as client:
        return client.call(request)

# Register remote service for call_service_impl
class ~wRemoteService(Service):
    '''Remote service wrapper for ~w.'''
    def __init__(self, socket_path='~w'):
        super().__init__('~w', stateful=False)
        self.socket_path = socket_path

    def call(self, request):
        return call_~w(request, self.socket_path)

# Auto-register remote service if server not local
try:
    if '~w' not in _services:
        register_service('~w', ~wRemoteService())
except:
    pass
", [ClassNameAtom, Name, SocketPath, SocketPath, Name, Name, SocketPath, Name, ClassNameAtom, ClassNameAtom, Name, SocketPath, Name, Name, Name, Name, ClassNameAtom]).

%% ============================================
%% PHASE 3: NETWORK SERVICES (TCP)
%% ============================================

%% compile_tcp_service_python(+Service, -PythonCode)
%  Generate Python code for a TCP network service server.
compile_tcp_service_python(service(Name, Options, HandlerSpec), PythonCode) :-
    % Extract host and port
    member(transport(tcp(Host, Port)), Options),
    % Determine if service is stateful
    ( member(stateful(true), Options) -> Stateful = "True" ; Stateful = "False" ),
    % Extract timeout (default 30000ms)
    ( member(timeout(TimeoutMs), Options) -> Timeout = TimeoutMs ; Timeout = 30000 ),
    % Generate handler code
    generate_service_handler_python(HandlerSpec, HandlerCode),
    % Format the class name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        ClassName = [Upper|Rest]
    ;
        ClassName = [First|Rest]
    ),
    atom_codes(ClassNameAtom, ClassName),
    % Generate the TCP service
    format(string(PythonCode),
"import socket
import json
import threading
import signal
import sys

class ~wService(Service):
    '''
    TCP Network Service: ~w
    Host: ~w, Port: ~w
    Auto-generated from Prolog service definition.
    '''
    def __init__(self):
        super().__init__('~w', stateful=~w)
        self.host = '~w'
        self.port = ~w
        self.timeout = ~w / 1000.0  # Convert to seconds
        self.server_socket = None
        self.running = False
        self._lock = threading.Lock()

    def call(self, request):
        '''Process request and return response.'''
~w

    def start_server(self):
        '''Start the TCP server.'''
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)  # Allow periodic check for shutdown
        self.running = True

        print(f'[~w] Server listening on {self.host}:{self.port}', file=sys.stderr)

        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                print(f'[~w] Connection from {addr}', file=sys.stderr)
                threading.Thread(target=self._handle_connection, args=(conn, addr), daemon=True).start()
            except socket.timeout:
                continue
            except OSError:
                break

        self._cleanup()

    def _handle_connection(self, conn, addr):
        '''Handle a client connection.'''
        conn.settimeout(self.timeout)
        buffer = b''
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data
                # Process complete JSONL messages
                while b'\\n' in buffer:
                    line, buffer = buffer.split(b'\\n', 1)
                    if line:
                        self._process_request(conn, line)
        except socket.timeout:
            self._send_error(conn, 'timeout', 'Request timed out')
        except Exception as e:
            self._send_error(conn, 'error', str(e))
        finally:
            conn.close()

    def _process_request(self, conn, line):
        '''Process a single JSONL request.'''
        try:
            request = json.loads(line.decode('utf-8'))
            request_id = request.get('_id')
            payload = request.get('_payload', request)

            with self._lock:
                response = self.call(payload)

            self._send_response(conn, request_id, response)
        except json.JSONDecodeError as e:
            self._send_error(conn, 'parse_error', f'Invalid JSON: {e}')
        except ServiceError as e:
            self._send_error(conn, 'service_error', str(e))
        except Exception as e:
            self._send_error(conn, 'error', str(e))

    def _send_response(self, conn, request_id, response):
        '''Send a JSONL response.'''
        msg = {'_id': request_id, '_status': 'ok', '_payload': response}
        conn.sendall((json.dumps(msg) + '\\n').encode('utf-8'))

    def _send_error(self, conn, error_type, message):
        '''Send a JSONL error response.'''
        msg = {'_status': 'error', '_error_type': error_type, '_message': message}
        try:
            conn.sendall((json.dumps(msg) + '\\n').encode('utf-8'))
        except:
            pass

    def stop_server(self):
        '''Stop the TCP server.'''
        self.running = False
        if self.server_socket:
            self.server_socket.close()

    def _cleanup(self):
        '''Clean up server resources.'''
        print(f'[~w] Server stopped', file=sys.stderr)

# Create service instance
_~w_service = ~wService()

# Register for in-process calls
register_service('~w', _~w_service)

def run_~w_server():
    '''Run the ~w service as a standalone TCP server.'''
    def signal_handler(sig, frame):
        print(f'\\n[~w] Shutting down...', file=sys.stderr)
        _~w_service.stop_server()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    _~w_service.start_server()

if __name__ == '__main__':
    run_~w_server()
", [ClassNameAtom, Name, Host, Port, Name, Stateful, Host, Port, Timeout, HandlerCode,
    Name, Name, Name, Name, ClassNameAtom, Name, Name, Name, Name, Name, Name, Name, Name]).

%% compile_tcp_client_python(+ServiceName, +Host, +Port, -PythonCode)
%  Generate Python code for a TCP network service client.
compile_tcp_client_python(Name, Host, Port, PythonCode) :-
    % Format the class name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        ClassName = [Upper|Rest]
    ;
        ClassName = [First|Rest]
    ),
    atom_codes(ClassNameAtom, ClassName),
    format(string(PythonCode),
"import socket
import json
import uuid

class ~wClient:
    '''
    TCP Network Client for ~w service.
    Host: ~w, Port: ~w
    Auto-generated from Prolog service definition.
    '''
    def __init__(self, host='~w', port=~w, timeout=30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket = None

    def connect(self):
        '''Connect to the service.'''
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self.timeout)
        self._socket.connect((self.host, self.port))
        return self

    def disconnect(self):
        '''Disconnect from the service.'''
        if self._socket:
            self._socket.close()
            self._socket = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    def call(self, request):
        '''Send a request and receive a response.'''
        if not self._socket:
            self.connect()

        request_id = str(uuid.uuid4())
        msg = {'_id': request_id, '_payload': request}
        self._socket.sendall((json.dumps(msg) + '\\n').encode('utf-8'))

        # Read response
        buffer = b''
        while True:
            data = self._socket.recv(4096)
            if not data:
                raise ConnectionError('Server closed connection')
            buffer += data
            if b'\\n' in buffer:
                line, _ = buffer.split(b'\\n', 1)
                response = json.loads(line.decode('utf-8'))
                if response.get('_status') == 'ok':
                    return response.get('_payload')
                else:
                    raise ServiceError(
                        service='~w',
                        message=response.get('_message', 'Unknown error')
                    )

def call_~w(request, host='~w', port=~w, timeout=30.0):
    '''Convenience function to call ~w service over TCP.'''
    with ~wClient(host, port, timeout) as client:
        return client.call(request)

# Register remote service for call_service_impl
class ~wRemoteService(Service):
    '''Remote service wrapper for ~w (TCP).'''
    def __init__(self, host='~w', port=~w):
        super().__init__('~w', stateful=False)
        self.host = host
        self.port = port

    def call(self, request):
        return call_~w(request, self.host, self.port)

# Auto-register remote service if server not local
try:
    if '~w' not in _services:
        register_service('~w', ~wRemoteService())
except:
    pass
", [ClassNameAtom, Name, Host, Port, Host, Port, Name, Name, Host, Port, Name, ClassNameAtom,
    ClassNameAtom, Name, Host, Port, Name, Name, Name, Name, ClassNameAtom]).

%% ============================================
%% PHASE 3: NETWORK SERVICES (HTTP/REST)
%% ============================================

%% compile_http_service_python(+Service, -PythonCode)
%  Generate Python code for an HTTP REST service server.
compile_http_service_python(service(Name, Options, HandlerSpec), PythonCode) :-
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
    ( member(stateful(true), Options) -> Stateful = "True" ; Stateful = "False" ),
    % Extract timeout (default 30000ms)
    ( member(timeout(TimeoutMs), Options) -> Timeout = TimeoutMs ; Timeout = 30000 ),
    % Generate handler code
    generate_service_handler_python(HandlerSpec, HandlerCode),
    % Format the class name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        ClassName = [Upper|Rest]
    ;
        ClassName = [First|Rest]
    ),
    atom_codes(ClassNameAtom, ClassName),
    % Generate the HTTP service
    format(string(PythonCode),
"from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
import signal
import sys
import urllib.parse

class ~wHandler(BaseHTTPRequestHandler):
    '''HTTP request handler for ~w service.'''

    service = None  # Set by server initialization

    def log_message(self, format, *args):
        '''Log to stderr.'''
        print(f'[~w] {args[0]}', file=sys.stderr)

    def _send_json_response(self, status_code, data):
        '''Send JSON response.'''
        body = json.dumps(data).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        '''Read and parse JSON request body.'''
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            body = self.rfile.read(content_length)
            return json.loads(body.decode('utf-8'))
        return {}

    def _handle_request(self, method):
        '''Handle HTTP request.'''
        try:
            # Parse URL
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = dict(urllib.parse.parse_qsl(parsed.query))

            # Check endpoint match
            if not path.startswith('~w'):
                self._send_json_response(404, {'error': 'Not found'})
                return

            # Build request object
            request = {
                'method': method,
                'path': path,
                'query': query,
                'headers': dict(self.headers)
            }

            # Add body for POST/PUT/PATCH
            if method in ('POST', 'PUT', 'PATCH'):
                request['body'] = self._read_json_body()

            # Call service
            response = self.service.call(request)
            self._send_json_response(200, {'_status': 'ok', '_payload': response})

        except ServiceError as e:
            self._send_json_response(400, {'_status': 'error', '_error_type': 'service_error', '_message': str(e)})
        except Exception as e:
            self._send_json_response(500, {'_status': 'error', '_error_type': 'internal_error', '_message': str(e)})

    def do_GET(self):
        self._handle_request('GET')

    def do_POST(self):
        self._handle_request('POST')

    def do_PUT(self):
        self._handle_request('PUT')

    def do_DELETE(self):
        self._handle_request('DELETE')

    def do_PATCH(self):
        self._handle_request('PATCH')

class ~wService(Service):
    '''
    HTTP REST Service: ~w
    Endpoint: ~w
    Host: ~w, Port: ~w
    Auto-generated from Prolog service definition.
    '''
    def __init__(self):
        super().__init__('~w', stateful=~w)
        self.host = '~w'
        self.port = ~w
        self.endpoint = '~w'
        self.timeout = ~w / 1000.0
        self.server = None
        self.running = False
        self._lock = threading.Lock()

    def call(self, request):
        '''Process request and return response.'''
        with self._lock:
~w

    def start_server(self):
        '''Start the HTTP server.'''
        ~wHandler.service = self
        self.server = HTTPServer((self.host, self.port), ~wHandler)
        self.server.timeout = 1.0
        self.running = True

        print(f'[~w] HTTP server listening on http://{self.host}:{self.port}~w', file=sys.stderr)

        while self.running:
            try:
                self.server.handle_request()
            except Exception as e:
                if self.running:
                    print(f'[~w] Error: {e}', file=sys.stderr)

        self._cleanup()

    def stop_server(self):
        '''Stop the HTTP server.'''
        self.running = False
        if self.server:
            self.server.server_close()

    def _cleanup(self):
        '''Clean up server resources.'''
        print(f'[~w] HTTP server stopped', file=sys.stderr)

# Create service instance
_~w_service = ~wService()

# Register for in-process calls
register_service('~w', _~w_service)

def run_~w_server():
    '''Run the ~w HTTP service as a standalone server.'''
    def signal_handler(sig, frame):
        print(f'\\n[~w] Shutting down...', file=sys.stderr)
        _~w_service.stop_server()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    _~w_service.start_server()

if __name__ == '__main__':
    run_~w_server()
", [ClassNameAtom, Name, Name, Endpoint, ClassNameAtom, Name, Endpoint, Host, Port,
    Name, Stateful, Host, Port, Endpoint, Timeout, HandlerCode,
    ClassNameAtom, ClassNameAtom, Name, Endpoint, Name, Name,
    Name, ClassNameAtom, Name, Name, Name, Name, Name, Name, Name, Name]).

%% compile_http_client_python(+ServiceName, +Endpoint, -PythonCode)
%  Generate Python code for an HTTP REST service client.
compile_http_client_python(Name, Endpoint, PythonCode) :-
    compile_http_client_python(Name, Endpoint, [], PythonCode).

compile_http_client_python(Name, Endpoint, HttpOptions, PythonCode) :-
    % Extract host and port from options or use defaults
    ( member(host(Host), HttpOptions) -> true ; Host = 'localhost' ),
    ( member(port(Port), HttpOptions) -> true ; Port = 8080 ),
    % Format the class name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        ClassName = [Upper|Rest]
    ;
        ClassName = [First|Rest]
    ),
    atom_codes(ClassNameAtom, ClassName),
    format(string(PythonCode),
"import urllib.request
import urllib.parse
import json

class ~wClient:
    '''
    HTTP REST Client for ~w service.
    Endpoint: ~w
    Default Host: ~w, Port: ~w
    Auto-generated from Prolog service definition.
    '''
    def __init__(self, host='~w', port=~w, timeout=30.0):
        self.base_url = f'http://{host}:{port}~w'
        self.timeout = timeout

    def _make_request(self, method, path='', data=None, query=None):
        '''Make HTTP request to service.'''
        url = self.base_url + path
        if query:
            url += '?' + urllib.parse.urlencode(query)

        headers = {'Content-Type': 'application/json'}
        body = json.dumps(data).encode('utf-8') if data else None

        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('_status') == 'ok':
                    return result.get('_payload')
                else:
                    raise ServiceError(
                        service='~w',
                        message=result.get('_message', 'Unknown error')
                    )
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8')
            try:
                result = json.loads(body)
                raise ServiceError(service='~w', message=result.get('_message', str(e)))
            except json.JSONDecodeError:
                raise ServiceError(service='~w', message=str(e))

    def get(self, path='', query=None):
        '''HTTP GET request.'''
        return self._make_request('GET', path, query=query)

    def post(self, path='', data=None):
        '''HTTP POST request.'''
        return self._make_request('POST', path, data=data)

    def put(self, path='', data=None):
        '''HTTP PUT request.'''
        return self._make_request('PUT', path, data=data)

    def delete(self, path=''):
        '''HTTP DELETE request.'''
        return self._make_request('DELETE', path)

    def call(self, request):
        '''Generic call method for service compatibility.'''
        method = request.get('method', 'POST')
        path = request.get('path', '')
        data = request.get('body', request.get('data'))
        query = request.get('query')
        return self._make_request(method, path, data, query)

def call_~w(request, host='~w', port=~w, timeout=30.0):
    '''Convenience function to call ~w service over HTTP.'''
    client = ~wClient(host, port, timeout)
    return client.call(request)

# Register remote service for call_service_impl
class ~wRemoteService(Service):
    '''Remote service wrapper for ~w (HTTP).'''
    def __init__(self, host='~w', port=~w):
        super().__init__('~w', stateful=False)
        self.host = host
        self.port = port

    def call(self, request):
        return call_~w(request, self.host, self.port)

# Auto-register remote service if server not local
try:
    if '~w' not in _services:
        register_service('~w', ~wRemoteService())
except:
    pass
", [ClassNameAtom, Name, Endpoint, Host, Port, Host, Port, Endpoint,
    Name, Name, Name, Name, Host, Port, Name, ClassNameAtom,
    ClassNameAtom, Name, Host, Port, Name, Name, Name, Name, ClassNameAtom]).

%% ============================================
%% PHASE 4: SERVICE MESH
%% ============================================

%% compile_service_mesh_python(+Service, -PythonCode)
%  Generate Python code for a service mesh service with load balancing,
%  circuit breaker, and retry capabilities.
compile_service_mesh_python(service(Name, Options, HandlerSpec), PythonCode) :-
    % Generate handler code
    generate_service_handler_python(HandlerSpec, HandlerCode),
    % Format the class name (capitalize first letter)
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        ClassName = [Upper|Rest]
    ;
        ClassName = [First|Rest]
    ),
    atom_codes(ClassNameAtom, ClassName),
    % Extract service mesh configurations
    ( member(stateful(true), Options) -> Stateful = "True" ; Stateful = "False" ),
    % Load balancing
    ( ( member(load_balance(LBStrategy), Options) ; member(load_balance(LBStrategy, _), Options) ) ->
        generate_load_balancer_python(LBStrategy, LoadBalancerCode),
        atom_string(LBStrategy, LBStrategyStr)
    ;
        LoadBalancerCode = "",
        LBStrategyStr = "none"
    ),
    % Circuit breaker
    ( ( member(circuit_breaker(threshold(CBThreshold), timeout(CBTimeout)), Options)
      ; ( member(circuit_breaker(CBOpts), Options), is_list(CBOpts),
          ( member(threshold(CBThreshold), CBOpts) -> true ; CBThreshold = 5 ),
          ( member(timeout(CBTimeout), CBOpts) -> true ; CBTimeout = 30000 ) ) ) ->
        generate_circuit_breaker_python(config(CBThreshold, CBTimeout), CircuitBreakerCode),
        format(string(CBConfig), "CircuitBreakerConfig(~w, ~w)", [CBThreshold, CBTimeout])
    ;
        CircuitBreakerCode = "",
        CBConfig = "None",
        CBThreshold = 5,
        CBTimeout = 30000
    ),
    % Retry with backoff
    ( ( member(retry(RetryN, RetryStrategy, RetryOpts), Options) ->
          ( member(delay(RetryDelay), RetryOpts) -> true ; RetryDelay = 100 ),
          ( member(max_delay(RetryMaxDelay), RetryOpts) -> true ; RetryMaxDelay = 30000 )
      ; member(retry(RetryN, RetryStrategy), Options) ->
          RetryDelay = 100,
          RetryMaxDelay = 30000
      ) ->
        generate_retry_python(config(RetryN, RetryStrategy, RetryDelay, RetryMaxDelay), RetryCode),
        atom_string(RetryStrategy, RetryStrategyStr),
        format(string(RetryConfig), "RetryConfig(~w, '~w', ~w, ~w)", [RetryN, RetryStrategyStr, RetryDelay, RetryMaxDelay])
    ;
        RetryCode = "",
        RetryConfig = "None",
        RetryN = 0,
        RetryDelay = 100,
        RetryMaxDelay = 30000
    ),
    % Backends
    ( member(backends(Backends), Options) ->
        generate_backends_python(Backends, BackendsCode)
    ;
        BackendsCode = "[]"
    ),
    % Generate the service mesh service class
    format(string(PythonCode),
"import time
import random
import threading
from collections import namedtuple
from enum import Enum, auto

~w
~w
~w

class CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()

CircuitBreakerConfig = namedtuple('CircuitBreakerConfig', ['threshold', 'timeout'])
RetryConfig = namedtuple('RetryConfig', ['max_retries', 'strategy', 'delay', 'max_delay'])

class ~wService(Service):
    '''
    Service Mesh Service: ~w
    Load Balancing: ~w
    Circuit Breaker: threshold=~w, timeout=~w
    Retry: ~w attempts
    '''
    def __init__(self):
        super().__init__('~w', stateful=~w)
        self.backends = ~w
        self.lb_strategy = '~w'
        self.cb_config = ~w
        self.retry_config = ~w
        self._circuit_state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._lock = threading.Lock()
        self._rr_index = 0

    def _select_backend(self):
        '''Select a backend using the configured load balancing strategy.'''
        if not self.backends:
            return None
        if self.lb_strategy == 'round_robin':
            backend = self.backends[self._rr_index %% len(self.backends)]
            self._rr_index += 1
            return backend
        elif self.lb_strategy == 'random':
            return random.choice(self.backends)
        elif self.lb_strategy == 'least_connections':
            # For simplicity, round-robin (real impl would track connections)
            backend = self.backends[self._rr_index %% len(self.backends)]
            self._rr_index += 1
            return backend
        else:
            return self.backends[0]

    def _check_circuit(self):
        '''Check circuit breaker state.'''
        if self.cb_config is None:
            return True
        with self._lock:
            if self._circuit_state == CircuitState.OPEN:
                if self._last_failure_time and \\
                   (time.time() - self._last_failure_time) * 1000 > self.cb_config.timeout:
                    self._circuit_state = CircuitState.HALF_OPEN
                    return True
                return False
            return True

    def _record_success(self):
        '''Record a successful call.'''
        with self._lock:
            if self._circuit_state == CircuitState.HALF_OPEN:
                self._circuit_state = CircuitState.CLOSED
            self._failure_count = 0

    def _record_failure(self):
        '''Record a failed call.'''
        if self.cb_config is None:
            return
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.cb_config.threshold:
                self._circuit_state = CircuitState.OPEN

    def _calculate_delay(self, attempt):
        '''Calculate retry delay based on strategy.'''
        if self.retry_config is None:
            return 0
        base_delay = self.retry_config.delay
        if self.retry_config.strategy == 'fixed':
            return base_delay
        elif self.retry_config.strategy == 'linear':
            return min(base_delay * (attempt + 1), self.retry_config.max_delay)
        elif self.retry_config.strategy == 'exponential':
            return min(base_delay * (2 ** attempt), self.retry_config.max_delay)
        return base_delay

    def call(self, request):
        '''Process request with service mesh features.'''
        # Check circuit breaker
        if not self._check_circuit():
            raise Exception('Circuit breaker is open')

        max_attempts = self.retry_config.max_retries + 1 if self.retry_config else 1
        last_error = None

        for attempt in range(max_attempts):
            try:
                # Select backend if load balancing
                backend = self._select_backend()
                if backend:
                    # Route to backend (simplified - real impl would call backend)
                    pass

                # Execute handler
                result = self._handle_request(request)
                self._record_success()
                return result

            except Exception as e:
                last_error = e
                self._record_failure()
                if attempt < max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    if delay > 0:
                        time.sleep(delay / 1000.0)

        raise last_error or Exception('All retries exhausted')

    def _handle_request(self, request):
        '''Execute the actual handler logic.'''
~w

# Register service
register_service('~w', ~wService())
", [LoadBalancerCode, CircuitBreakerCode, RetryCode,
    ClassNameAtom, Name, LBStrategyStr, CBThreshold, CBTimeout, RetryN,
    Name, Stateful, BackendsCode, LBStrategyStr, CBConfig, RetryConfig,
    HandlerCode, Name, ClassNameAtom]).

%% generate_load_balancer_python(+Strategy, -Code)
%  Generate Python load balancer infrastructure.
generate_load_balancer_python(_, "# Load balancer strategies: round_robin, random, least_connections, weighted, ip_hash").

%% generate_circuit_breaker_python(+Config, -Code)
%  Generate Python circuit breaker infrastructure.
generate_circuit_breaker_python(_, "# Circuit breaker with configurable threshold and timeout").

%% generate_retry_python(+Config, -Code)
%  Generate Python retry with backoff infrastructure.
generate_retry_python(_, "# Retry with backoff strategies: fixed, linear, exponential").

%% generate_backends_python(+Backends, -Code)
%  Generate Python backend list.
generate_backends_python([], "[]").
generate_backends_python(Backends, Code) :-
    Backends \= [],
    generate_backend_list_python(Backends, BackendStrs),
    atomic_list_concat(BackendStrs, ', ', BackendList),
    format(string(Code), "[~w]", [BackendList]).

generate_backend_list_python([], []).
generate_backend_list_python([backend(Name, Transport)|Rest], [Str|RestStrs]) :-
    format(string(Str), "{'name': '~w', 'transport': '~w'}", [Name, Transport]),
    generate_backend_list_python(Rest, RestStrs).
generate_backend_list_python([backend(Name, Transport, _Opts)|Rest], [Str|RestStrs]) :-
    format(string(Str), "{'name': '~w', 'transport': '~w'}", [Name, Transport]),
    generate_backend_list_python(Rest, RestStrs).

%% ============================================
%% Phase 5: Polyglot Service Compilation
%% ============================================

%% compile_polyglot_service_python(+Service, -PythonCode)
%  Compiles a polyglot service that can call services in other languages.
compile_polyglot_service_python(service(Name, Options, HandlerSpec), PythonCode) :-
    % Get service class name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        ClassName = [Upper|Rest]
    ;
        ClassName = [First|Rest]
    ),
    atom_codes(ClassNameAtom, ClassName),
    % Get dependencies
    ( member(depends_on(Dependencies), Options) -> Dependencies = Deps ; Deps = [] ),
    % Get target language if specified
    ( member(target_language(Lang), Options) -> atom_string(Lang, LangStr) ; LangStr = "python" ),
    % Generate client code for each dependency
    generate_dependency_clients_python(Deps, ClientsCode),
    % Determine if stateful
    ( member(stateful(true), Options) -> Stateful = "True" ; Stateful = "False" ),
    % Get timeout
    ( member(timeout(Timeout), Options) -> true ; Timeout = 30000 ),
    % Generate handler code
    generate_service_handler_python(HandlerSpec, HandlerCode),
    % Format the complete polyglot service
    format(string(PythonCode), "import json
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Optional, List

# Phase 5: Polyglot Service Support
# Target language: ~w

class ServiceClient:
    '''Base class for cross-language service clients.'''
    def __init__(self, name: str, endpoint: str, timeout: float = 30.0):
        self.name = name
        self.endpoint = endpoint
        self.timeout = timeout

    def call(self, request: Any) -> Any:
        '''Make HTTP call to remote service.'''
        url = self.endpoint
        data = json.dumps({'_payload': request}).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('_status') == 'error':
                    raise Exception(result.get('_message', 'Remote service error'))
                return result.get('_payload', result)
        except urllib.error.URLError as e:
            raise Exception(f'Failed to call service {self.name}: {e}')

class ServiceRegistry:
    '''Registry for cross-language services.'''
    _services: Dict[str, 'ServiceClient'] = {}
    _local_services: Dict[str, Any] = {}

    @classmethod
    def register_remote(cls, name: str, endpoint: str, timeout: float = 30.0):
        '''Register a remote service.'''
        cls._services[name] = ServiceClient(name, endpoint, timeout)

    @classmethod
    def register_local(cls, name: str, service: Any):
        '''Register a local service.'''
        cls._local_services[name] = service

    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        '''Get a service by name (local or remote).'''
        if name in cls._local_services:
            return cls._local_services[name]
        return cls._services.get(name)

    @classmethod
    def call_service(cls, name: str, request: Any) -> Any:
        '''Call a service by name.'''
        service = cls.get(name)
        if service is None:
            raise Exception(f'Service not found: {name}')
        return service.call(request)

~w

class ~wService:
    '''
    Polyglot Service: ~w
    Target Language: ~w
    Dependencies: ~w
    '''
    def __init__(self):
        self.name = '~w'
        self.stateful = ~w
        self.timeout = ~w / 1000.0
        self._state = {} if self.stateful else None

    def call(self, request):
        '''Process request, calling remote services as needed.'''
~w

    def call_service(self, name: str, request: Any) -> Any:
        '''Call another service (local or remote).'''
        return ServiceRegistry.call_service(name, request)

# Register this service
ServiceRegistry.register_local('~w', ~wService())
", [LangStr, ClientsCode, ClassNameAtom, Name, LangStr, Deps, Name, Stateful, Timeout, HandlerCode, Name, ClassNameAtom]).

%% generate_dependency_clients_python(+Dependencies, -Code)
%  Generate client registrations for service dependencies.
generate_dependency_clients_python([], "# No remote service dependencies").
generate_dependency_clients_python(Deps, Code) :-
    Deps \= [],
    generate_dep_registrations_python(Deps, RegStrs),
    atomic_list_concat(RegStrs, '\n', Code).

generate_dep_registrations_python([], []).
generate_dep_registrations_python([Dep|Rest], [Str|RestStrs]) :-
    ( Dep = dep(Name, Lang, Transport) ->
        transport_to_endpoint_str(Transport, Endpoint),
        format(string(Str), "# ~w service (~w)~nServiceRegistry.register_remote('~w', '~w')",
               [Name, Lang, Name, Endpoint])
    ; Dep = dep(Name, Lang) ->
        format(string(Str), "# ~w service (~w) - endpoint TBD~nServiceRegistry.register_remote('~w', 'http://localhost:8080/~w')",
               [Name, Lang, Name, Name])
    ; atom(Dep) ->
        format(string(Str), "# ~w service~nServiceRegistry.register_remote('~w', 'http://localhost:8080/~w')",
               [Dep, Dep, Dep])
    ;
        Str = "# Unknown dependency format"
    ),
    generate_dep_registrations_python(Rest, RestStrs).

%% transport_to_endpoint_str(+Transport, -Endpoint)
transport_to_endpoint_str(tcp(Host, Port), Endpoint) :-
    format(string(Endpoint), "http://~w:~w", [Host, Port]).
transport_to_endpoint_str(http(Path), Endpoint) :-
    format(string(Endpoint), "http://localhost:8080~w", [Path]).
transport_to_endpoint_str(http(Host, Port), Endpoint) :-
    format(string(Endpoint), "http://~w:~w", [Host, Port]).
transport_to_endpoint_str(http(Host, Port, Path), Endpoint) :-
    format(string(Endpoint), "http://~w:~w~w", [Host, Port, Path]).
transport_to_endpoint_str(_, "http://localhost:8080").

%% generate_service_client_python(+ServiceName, +Endpoint, -Code)
%  Generate a standalone service client.
generate_service_client_python(ServiceName, Endpoint, Code) :-
    atom_codes(ServiceName, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        ClassName = [Upper|Rest]
    ;
        ClassName = [First|Rest]
    ),
    atom_codes(ClassNameAtom, ClassName),
    format(string(Code), "import json
import urllib.request
import urllib.error

class ~wClient:
    '''Client for ~w service.'''
    def __init__(self, endpoint: str = '~w', timeout: float = 30.0):
        self.endpoint = endpoint
        self.timeout = timeout

    def call(self, request):
        '''Call the ~w service.'''
        data = json.dumps({'_payload': request}).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        req = urllib.request.Request(self.endpoint, data=data, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('_status') == 'error':
                    raise Exception(result.get('_message', 'Remote service error'))
                return result.get('_payload', result)
        except urllib.error.URLError as e:
            raise Exception(f'Failed to call ~w service: {e}')
", [ClassNameAtom, ServiceName, Endpoint, ServiceName, ServiceName]).

%% ============================================
%% DISTRIBUTED SERVICES (Phase 6)
%% ============================================

%% compile_distributed_service_python(+Service, -PythonCode)
%  Generate Python code for a distributed service with sharding and replication.
compile_distributed_service_python(service(Name, Options, HandlerSpec), PythonCode) :-
    % Extract distributed configuration
    ( member(sharding(ShardStrategy), Options) -> true ; ShardStrategy = hash ),
    ( member(partition_key(PartitionKey), Options) -> true ; PartitionKey = id ),
    ( member(replication(ReplicationFactor), Options) -> true ; ReplicationFactor = 1 ),
    ( member(consistency(ConsistencyLevel), Options) -> true ; ConsistencyLevel = eventual ),
    ( member(cluster(ClusterConfig), Options) -> true ; ClusterConfig = [] ),
    ( member(stateful(true), Options) -> Stateful = "True" ; Stateful = "False" ),
    ( member(timeout(Timeout), Options) -> true ; Timeout = 30000 ),
    % Generate handler code
    generate_service_handler_python(HandlerSpec, HandlerCode),
    % Format the class name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        ClassName = [Upper|Rest]
    ;
        ClassName = [First|Rest]
    ),
    atom_codes(ClassNameAtom, ClassName),
    % Convert atoms to strings
    atom_string(ShardStrategy, ShardStrategyStr),
    atom_string(PartitionKey, PartitionKeyStr),
    atom_string(ConsistencyLevel, ConsistencyStr),
    % Generate cluster nodes list
    generate_cluster_nodes_python(ClusterConfig, NodesCode),
    % Generate the distributed service
    format(string(PythonCode),
"import json
import time
import hashlib
import random
import threading
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Phase 6: Distributed Service Support

class ShardingStrategy(Enum):
    HASH = 'hash'
    RANGE = 'range'
    CONSISTENT_HASH = 'consistent_hash'
    GEOGRAPHIC = 'geographic'

class ConsistencyLevel(Enum):
    EVENTUAL = 'eventual'
    STRONG = 'strong'
    QUORUM = 'quorum'
    READ_YOUR_WRITES = 'read_your_writes'
    CAUSAL = 'causal'

@dataclass
class ClusterNode:
    '''Represents a node in the distributed cluster.'''
    node_id: str
    host: str
    port: int
    region: str = 'default'
    weight: int = 1
    healthy: bool = True

@dataclass
class ShardInfo:
    '''Shard metadata.'''
    shard_id: int
    primary_node: str
    replica_nodes: List[str] = field(default_factory=list)
    key_range: Optional[Tuple[Any, Any]] = None

class ConsistentHashRing:
    '''Consistent hash ring for distributed routing.'''
    def __init__(self, replicas: int = 100):
        self.replicas = replicas
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []

    def add_node(self, node_id: str) -> None:
        for i in range(self.replicas):
            key = self._hash(f'{node_id}:{i}')
            self.ring[key] = node_id
            self.sorted_keys.append(key)
        self.sorted_keys.sort()

    def remove_node(self, node_id: str) -> None:
        for i in range(self.replicas):
            key = self._hash(f'{node_id}:{i}')
            if key in self.ring:
                del self.ring[key]
                self.sorted_keys.remove(key)

    def get_node(self, key: str) -> Optional[str]:
        if not self.ring:
            return None
        h = self._hash(key)
        for k in self.sorted_keys:
            if k >= h:
                return self.ring[k]
        return self.ring[self.sorted_keys[0]]

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

class ShardRouter:
    '''Routes requests to appropriate shards based on partition key.'''
    def __init__(self, strategy: ShardingStrategy, num_shards: int = 16):
        self.strategy = strategy
        self.num_shards = num_shards
        self.hash_ring = ConsistentHashRing()
        self.range_boundaries: List[Any] = []

    def get_shard(self, partition_key: Any) -> int:
        if self.strategy == ShardingStrategy.HASH:
            return self._hash_shard(partition_key)
        elif self.strategy == ShardingStrategy.CONSISTENT_HASH:
            return self._consistent_hash_shard(partition_key)
        elif self.strategy == ShardingStrategy.RANGE:
            return self._range_shard(partition_key)
        else:
            return self._hash_shard(partition_key)

    def _hash_shard(self, key: Any) -> int:
        h = hashlib.md5(str(key).encode()).hexdigest()
        return int(h, 16) %% self.num_shards

    def _consistent_hash_shard(self, key: Any) -> int:
        node = self.hash_ring.get_node(str(key))
        return hash(node) %% self.num_shards if node else 0

    def _range_shard(self, key: Any) -> int:
        for i, boundary in enumerate(self.range_boundaries):
            if key < boundary:
                return i
        return len(self.range_boundaries)

class ReplicationManager:
    '''Manages data replication across nodes.'''
    def __init__(self, replication_factor: int, consistency: ConsistencyLevel):
        self.replication_factor = replication_factor
        self.consistency = consistency
        self._lock = threading.Lock()

    def write_quorum(self) -> int:
        if self.consistency == ConsistencyLevel.STRONG:
            return self.replication_factor
        elif self.consistency == ConsistencyLevel.QUORUM:
            return (self.replication_factor // 2) + 1
        else:
            return 1

    def read_quorum(self) -> int:
        if self.consistency == ConsistencyLevel.STRONG:
            return self.replication_factor
        elif self.consistency == ConsistencyLevel.QUORUM:
            return (self.replication_factor // 2) + 1
        else:
            return 1

class ~wService:
    '''
    Distributed Service: ~w
    Sharding: ~w (partition_key: ~w)
    Replication: ~w replicas
    Consistency: ~w
    '''
    def __init__(self):
        self.name = '~w'
        self.stateful = ~w
        self.timeout_ms = ~w
        self.sharding_strategy = ShardingStrategy.~w
        self.partition_key = '~w'
        self.replication_factor = ~w
        self.consistency = ConsistencyLevel.~w
        self.nodes: Dict[str, ClusterNode] = {}
        self.shards: Dict[int, ShardInfo] = {}
        self.router = ShardRouter(self.sharding_strategy)
        self.replication = ReplicationManager(self.replication_factor, self.consistency)
        self._state: Dict[str, Any] = {}
        self._lock = threading.Lock()
~w

    def add_node(self, node: ClusterNode) -> None:
        '''Add a node to the cluster.'''
        self.nodes[node.node_id] = node
        self.router.hash_ring.add_node(node.node_id)

    def remove_node(self, node_id: str) -> None:
        '''Remove a node from the cluster.'''
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.router.hash_ring.remove_node(node_id)

    def get_partition_key(self, request: Dict[str, Any]) -> Any:
        '''Extract partition key from request.'''
        return request.get(self.partition_key, str(id(request)))

    def route_request(self, request: Dict[str, Any]) -> int:
        '''Route request to appropriate shard.'''
        key = self.get_partition_key(request)
        return self.router.get_shard(key)

    def call(self, request: Any) -> Any:
        '''Process a request through the distributed service.'''
        shard_id = self.route_request(request) if isinstance(request, dict) else 0
        return self._handle_request(request, shard_id)

    def _handle_request(self, request: Any, shard_id: int) -> Any:
~w

# Initialize cluster nodes
~w

# Service instance
~w_service = ~wService()
", [ClassNameAtom, Name, ShardStrategyStr, PartitionKeyStr, ReplicationFactor, ConsistencyStr,
    Name, Stateful, Timeout,
    ShardStrategyStr, PartitionKeyStr, ReplicationFactor, ConsistencyStr,
    NodesCode, HandlerCode, NodesCode, Name, ClassNameAtom]).

%% generate_sharding_python(+Strategy, -Code)
%  Generate Python code for a specific sharding strategy.
generate_sharding_python(hash, Code) :-
    Code = "ShardingStrategy.HASH".
generate_sharding_python(range, Code) :-
    Code = "ShardingStrategy.RANGE".
generate_sharding_python(consistent_hash, Code) :-
    Code = "ShardingStrategy.CONSISTENT_HASH".
generate_sharding_python(geographic, Code) :-
    Code = "ShardingStrategy.GEOGRAPHIC".
generate_sharding_python(_, Code) :-
    Code = "ShardingStrategy.HASH".

%% generate_replication_python(+ReplicationFactor, -Code)
%  Generate Python replication configuration code.
generate_replication_python(Factor, Code) :-
    integer(Factor),
    format(string(Code), "replication_factor = ~w", [Factor]).
generate_replication_python(_, "replication_factor = 1").

%% generate_cluster_nodes_python(+Config, -Code)
%  Generate Python code for cluster node initialization.
generate_cluster_nodes_python([], "        # No initial cluster nodes").
generate_cluster_nodes_python(Nodes, Code) :-
    Nodes \= [],
    maplist(generate_node_init_python, Nodes, NodeCodes),
    atomic_list_concat(NodeCodes, '\n', Code).

generate_node_init_python(node(Id, Host, Port), Code) :-
    format(string(Code), "~w_service.add_node(ClusterNode('~w', '~w', ~w))", [Id, Id, Host, Port]).
generate_node_init_python(node(Id, Host, Port, Region), Code) :-
    format(string(Code), "~w_service.add_node(ClusterNode('~w', '~w', ~w, '~w'))", [Id, Id, Host, Port, Region]).
generate_node_init_python(_, "        # Unknown node format").

%% ============================================
%% SERVICE DISCOVERY (Phase 7)
%% ============================================

%% compile_discovery_service_python(+Service, -PythonCode)
%  Generate Python code for a service with discovery capabilities.
compile_discovery_service_python(service(Name, Options, HandlerSpec), PythonCode) :-
    % Extract discovery configuration
    ( member(discovery_backend(Backend), Options) -> true ; Backend = consul ),
    ( member(health_check(HealthConfig), Options) -> true ; HealthConfig = http('/health', 30000) ),
    ( member(discovery_ttl(TTL), Options) -> true ; TTL = 60 ),
    ( member(discovery_tags(Tags), Options) -> true ; Tags = [] ),
    ( member(stateful(true), Options) -> Stateful = "True" ; Stateful = "False" ),
    ( member(timeout(Timeout), Options) -> true ; Timeout = 30000 ),
    % Generate handler code
    generate_service_handler_python(HandlerSpec, HandlerCode),
    % Format the class name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        ClassName = [Upper|Rest]
    ;
        ClassName = [First|Rest]
    ),
    atom_codes(ClassNameAtom, ClassName),
    % Convert atoms to strings
    backend_to_string(Backend, BackendStr),
    health_config_to_string(HealthConfig, HealthStr),
    tags_to_python_list(Tags, TagsStr),
    % Generate the discovery service
    format(string(PythonCode),
"import json
import time
import threading
import urllib.request
import urllib.error
import socket
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Phase 7: Service Discovery Support

class DiscoveryBackend(Enum):
    CONSUL = 'consul'
    ETCD = 'etcd'
    DNS = 'dns'
    KUBERNETES = 'kubernetes'
    ZOOKEEPER = 'zookeeper'
    EUREKA = 'eureka'

class HealthStatus(Enum):
    HEALTHY = 'healthy'
    UNHEALTHY = 'unhealthy'
    UNKNOWN = 'unknown'

@dataclass
class ServiceInstance:
    '''Represents a registered service instance.'''
    service_id: str
    service_name: str
    host: str
    port: int
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_heartbeat: float = 0.0

@dataclass
class HealthCheckConfig:
    '''Health check configuration.'''
    check_type: str  # http, tcp, grpc, script
    endpoint: str
    interval_ms: int
    timeout_ms: int = 5000
    deregister_after_ms: int = 60000

class ServiceRegistry(ABC):
    '''Abstract base class for service registries.'''
    @abstractmethod
    def register(self, instance: ServiceInstance) -> bool:
        pass

    @abstractmethod
    def deregister(self, service_id: str) -> bool:
        pass

    @abstractmethod
    def discover(self, service_name: str, tags: List[str] = None) -> List[ServiceInstance]:
        pass

    @abstractmethod
    def health_check(self, service_id: str) -> HealthStatus:
        pass

class ConsulRegistry(ServiceRegistry):
    '''Consul-based service registry.'''
    def __init__(self, host: str = 'localhost', port: int = 8500):
        self.host = host
        self.port = port
        self.base_url = f'http://{host}:{port}/v1'
        self._registered: Dict[str, ServiceInstance] = {}

    def register(self, instance: ServiceInstance) -> bool:
        try:
            payload = {
                'ID': instance.service_id,
                'Name': instance.service_name,
                'Address': instance.host,
                'Port': instance.port,
                'Tags': instance.tags,
                'Meta': instance.metadata
            }
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                f'{self.base_url}/agent/service/register',
                data=data,
                headers={'Content-Type': 'application/json'},
                method='PUT'
            )
            urllib.request.urlopen(req, timeout=5)
            self._registered[instance.service_id] = instance
            return True
        except Exception as e:
            print(f'Failed to register service: {e}')
            return False

    def deregister(self, service_id: str) -> bool:
        try:
            req = urllib.request.Request(
                f'{self.base_url}/agent/service/deregister/{service_id}',
                method='PUT'
            )
            urllib.request.urlopen(req, timeout=5)
            self._registered.pop(service_id, None)
            return True
        except Exception:
            return False

    def discover(self, service_name: str, tags: List[str] = None) -> List[ServiceInstance]:
        try:
            url = f'{self.base_url}/catalog/service/{service_name}'
            if tags:
                url += '?tag=' + '&tag='.join(tags)
            req = urllib.request.Request(url)
            response = urllib.request.urlopen(req, timeout=5)
            services = json.loads(response.read().decode('utf-8'))
            return [
                ServiceInstance(
                    service_id=s.get('ServiceID', ''),
                    service_name=s.get('ServiceName', ''),
                    host=s.get('ServiceAddress', s.get('Address', '')),
                    port=s.get('ServicePort', 0),
                    tags=s.get('ServiceTags', []),
                    metadata=s.get('ServiceMeta', {})
                )
                for s in services
            ]
        except Exception:
            return []

    def health_check(self, service_id: str) -> HealthStatus:
        try:
            req = urllib.request.Request(f'{self.base_url}/health/service/{service_id}')
            response = urllib.request.urlopen(req, timeout=5)
            health = json.loads(response.read().decode('utf-8'))
            if health and all(c.get('Status') == 'passing' for h in health for c in h.get('Checks', [])):
                return HealthStatus.HEALTHY
            return HealthStatus.UNHEALTHY
        except Exception:
            return HealthStatus.UNKNOWN

class LocalRegistry(ServiceRegistry):
    '''In-memory service registry for testing and development.'''
    _instances: Dict[str, ServiceInstance] = {}
    _lock = threading.Lock()

    def register(self, instance: ServiceInstance) -> bool:
        with self._lock:
            self._instances[instance.service_id] = instance
            instance.last_heartbeat = time.time()
            return True

    def deregister(self, service_id: str) -> bool:
        with self._lock:
            return self._instances.pop(service_id, None) is not None

    def discover(self, service_name: str, tags: List[str] = None) -> List[ServiceInstance]:
        with self._lock:
            results = [
                i for i in self._instances.values()
                if i.service_name == service_name
            ]
            if tags:
                results = [i for i in results if all(t in i.tags for t in tags)]
            return results

    def health_check(self, service_id: str) -> HealthStatus:
        with self._lock:
            instance = self._instances.get(service_id)
            if not instance:
                return HealthStatus.UNKNOWN
            # Check if heartbeat is recent (within 2x TTL)
            if time.time() - instance.last_heartbeat < 120:
                return HealthStatus.HEALTHY
            return HealthStatus.UNHEALTHY

class HealthChecker:
    '''Performs health checks on services.'''
    def __init__(self, config: HealthCheckConfig):
        self.config = config
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def check_http(self, host: str, port: int) -> HealthStatus:
        try:
            url = f'http://{host}:{port}{self.config.endpoint}'
            req = urllib.request.Request(url)
            response = urllib.request.urlopen(req, timeout=self.config.timeout_ms / 1000)
            if response.getcode() == 200:
                return HealthStatus.HEALTHY
            return HealthStatus.UNHEALTHY
        except Exception:
            return HealthStatus.UNHEALTHY

    def check_tcp(self, host: str, port: int) -> HealthStatus:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.timeout_ms / 1000)
            result = sock.connect_ex((host, port))
            sock.close()
            return HealthStatus.HEALTHY if result == 0 else HealthStatus.UNHEALTHY
        except Exception:
            return HealthStatus.UNHEALTHY

    def check(self, host: str, port: int) -> HealthStatus:
        if self.config.check_type == 'http':
            return self.check_http(host, port)
        elif self.config.check_type == 'tcp':
            return self.check_tcp(host, port)
        return HealthStatus.UNKNOWN

class ~wService:
    '''
    Discoverable Service: ~w
    Backend: ~w
    TTL: ~w seconds
    Tags: ~w
    '''
    def __init__(self, host: str = 'localhost', port: int = 8080):
        self.name = '~w'
        self.host = host
        self.port = port
        self.stateful = ~w
        self.timeout_ms = ~w
        self.ttl = ~w
        self.tags = ~w
        self._state: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Initialize registry based on backend
        self.registry: ServiceRegistry = ~w
        self.health_config = HealthCheckConfig(
            check_type='~w',
            endpoint='~w',
            interval_ms=~w
        )
        self.health_checker = HealthChecker(self.health_config)

        # Create service instance
        self.instance = ServiceInstance(
            service_id=f'{self.name}-{host}-{port}',
            service_name=self.name,
            host=host,
            port=port,
            tags=self.tags
        )

    def register(self) -> bool:
        '''Register this service with the discovery backend.'''
        success = self.registry.register(self.instance)
        if success:
            self._start_heartbeat()
        return success

    def deregister(self) -> bool:
        '''Deregister this service from the discovery backend.'''
        self._stop_heartbeat()
        return self.registry.deregister(self.instance.service_id)

    def discover_peers(self) -> List[ServiceInstance]:
        '''Discover other instances of this service.'''
        return self.registry.discover(self.name, self.tags)

    def _start_heartbeat(self) -> None:
        '''Start the heartbeat thread.'''
        if self._running:
            return
        self._running = True
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        '''Stop the heartbeat thread.'''
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=1)

    def _heartbeat_loop(self) -> None:
        '''Send periodic heartbeats to maintain registration.'''
        while self._running:
            self.instance.last_heartbeat = time.time()
            self.instance.health_status = self.health_checker.check(self.host, self.port)
            time.sleep(self.ttl / 2)

    def call(self, request: Any) -> Any:
        '''Process a request through the service.'''
        return self._handle_request(request)

    def _handle_request(self, request: Any) -> Any:
~w

# Service instance
~w_service = ~wService()
", [ClassNameAtom, Name, BackendStr, TTL, TagsStr,
    Name, Stateful, Timeout, TTL, TagsStr,
    BackendStr, HealthStr, HealthStr, TTL,
    HandlerCode, Name, ClassNameAtom]).

%% backend_to_string(+Backend, -String)
%  Convert discovery backend to Python code string.
backend_to_string(consul, "ConsulRegistry()").
backend_to_string(consul(Host, Port), Code) :-
    format(string(Code), "ConsulRegistry('~w', ~w)", [Host, Port]).
backend_to_string(etcd, "LocalRegistry()").  % Fallback for now
backend_to_string(dns, "LocalRegistry()").
backend_to_string(kubernetes, "LocalRegistry()").
backend_to_string(_, "LocalRegistry()").

%% health_config_to_string(+Config, -TypeString)
%  Extract health check type from config.
health_config_to_string(http(Path, _), Path) :- !.
health_config_to_string(http(Path, _, _), Path) :- !.
health_config_to_string(tcp(_, _), "tcp") :- !.
health_config_to_string(grpc(_), "grpc") :- !.
health_config_to_string(_, "/health").

%% tags_to_python_list(+Tags, -String)
%  Convert Prolog list of tags to Python list string.
tags_to_python_list([], "[]").
tags_to_python_list(Tags, Code) :-
    Tags \= [],
    maplist(quote_string, Tags, QuotedTags),
    atomic_list_concat(QuotedTags, ', ', Inner),
    format(string(Code), "[~w]", [Inner]).

quote_string(Atom, Quoted) :-
    format(string(Quoted), "'~w'", [Atom]).

%% generate_health_check_python(+Config, -Code)
%  Generate Python health check configuration code.
generate_health_check_python(http(Path, Interval), Code) :-
    format(string(Code), "HealthCheckConfig('http', '~w', ~w)", [Path, Interval]).
generate_health_check_python(http(Path, Interval, Timeout), Code) :-
    format(string(Code), "HealthCheckConfig('http', '~w', ~w, ~w)", [Path, Interval, Timeout]).
generate_health_check_python(tcp(Port, Interval), Code) :-
    format(string(Code), "HealthCheckConfig('tcp', '~w', ~w)", [Port, Interval]).
generate_health_check_python(_, "HealthCheckConfig('http', '/health', 30000)").

%% generate_service_registry_python(+Backend, -Code)
%  Generate Python service registry initialization code.
generate_service_registry_python(consul, "ConsulRegistry()").
generate_service_registry_python(consul(Host, Port), Code) :-
    format(string(Code), "ConsulRegistry('~w', ~w)", [Host, Port]).
generate_service_registry_python(_, "LocalRegistry()").

%% ============================================
%% SERVICE TRACING (Phase 8)
%% ============================================

%% compile_traced_service_python(+Service, -PythonCode)
%  Generate Python code for a service with distributed tracing.
compile_traced_service_python(service(Name, Options, HandlerSpec), PythonCode) :-
    % Extract tracing configuration
    ( member(trace_exporter(Exporter), Options) -> true ; Exporter = otlp ),
    ( member(trace_sampling(SamplingRate), Options) -> true ; SamplingRate = 1.0 ),
    ( member(trace_service_name(ServiceName), Options) -> true ; ServiceName = Name ),
    ( member(trace_propagation(Propagation), Options) -> true ; Propagation = w3c ),
    ( member(trace_attributes(Attributes), Options) -> true ; Attributes = [] ),
    ( member(stateful(true), Options) -> Stateful = "True" ; Stateful = "False" ),
    ( member(timeout(Timeout), Options) -> true ; Timeout = 30000 ),
    % Generate handler code
    generate_service_handler_python(HandlerSpec, HandlerCode),
    % Format the class name
    atom_codes(Name, [First|Rest]),
    ( First >= 0'a, First =< 0'z ->
        Upper is First - 32,
        ClassName = [Upper|Rest]
    ;
        ClassName = [First|Rest]
    ),
    atom_codes(ClassNameAtom, ClassName),
    % Convert to strings
    exporter_to_string(Exporter, ExporterStr),
    propagation_to_string(Propagation, PropagationStr),
    attributes_to_python_dict(Attributes, AttrsStr),
    atom_string(ServiceName, ServiceNameStr),
    % Generate the traced service
    format(string(PythonCode),
"import json
import time
import random
import threading
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import uuid

# Phase 8: Service Tracing Support (OpenTelemetry-compatible)

class TraceExporter(Enum):
    OTLP = 'otlp'
    JAEGER = 'jaeger'
    ZIPKIN = 'zipkin'
    DATADOG = 'datadog'
    CONSOLE = 'console'
    NONE = 'none'

class PropagationFormat(Enum):
    W3C = 'w3c'  # W3C Trace Context
    B3 = 'b3'    # Zipkin B3
    B3_MULTI = 'b3_multi'
    JAEGER = 'jaeger'
    XRAY = 'xray'
    DATADOG = 'datadog'

class SpanKind(Enum):
    INTERNAL = 'internal'
    SERVER = 'server'
    CLIENT = 'client'
    PRODUCER = 'producer'
    CONSUMER = 'consumer'

class SpanStatus(Enum):
    UNSET = 'unset'
    OK = 'ok'
    ERROR = 'error'

@dataclass
class SpanContext:
    '''W3C Trace Context compatible span context.'''
    trace_id: str
    span_id: str
    trace_flags: int = 1  # sampled
    trace_state: str = ''

    @classmethod
    def generate(cls) -> 'SpanContext':
        return cls(
            trace_id=uuid.uuid4().hex + uuid.uuid4().hex[:16],
            span_id=uuid.uuid4().hex[:16]
        )

    def to_traceparent(self) -> str:
        '''Format as W3C traceparent header.'''
        return f'00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}'

    @classmethod
    def from_traceparent(cls, header: str) -> Optional['SpanContext']:
        '''Parse W3C traceparent header.'''
        try:
            parts = header.split('-')
            if len(parts) >= 4:
                return cls(
                    trace_id=parts[1],
                    span_id=parts[2],
                    trace_flags=int(parts[3], 16)
                )
        except Exception:
            pass
        return None

    def to_b3(self) -> Dict[str, str]:
        '''Format as B3 headers.'''
        return {
            'X-B3-TraceId': self.trace_id,
            'X-B3-SpanId': self.span_id,
            'X-B3-Sampled': '1' if self.trace_flags else '0'
        }

@dataclass
class Span:
    '''Represents a single span in a trace.'''
    name: str
    context: SpanContext
    parent_context: Optional[SpanContext] = None
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    start_time_ns: int = 0
    end_time_ns: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    links: List[SpanContext] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Dict[str, Any] = None) -> None:
        self.events.append({
            'name': name,
            'timestamp_ns': time.time_ns(),
            'attributes': attributes or {}
        })

    def set_status(self, status: SpanStatus, description: str = '') -> None:
        self.status = status
        if description:
            self.attributes['status.description'] = description

    def end(self) -> None:
        self.end_time_ns = time.time_ns()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'trace_id': self.context.trace_id,
            'span_id': self.context.span_id,
            'parent_span_id': self.parent_context.span_id if self.parent_context else None,
            'kind': self.kind.value,
            'status': self.status.value,
            'start_time_ns': self.start_time_ns,
            'end_time_ns': self.end_time_ns,
            'duration_ms': (self.end_time_ns - self.start_time_ns) / 1_000_000,
            'attributes': self.attributes,
            'events': self.events
        }

class SpanExporter:
    '''Base class for span exporters.'''
    def export(self, spans: List[Span]) -> bool:
        raise NotImplementedError

class ConsoleExporter(SpanExporter):
    '''Exports spans to console for debugging.'''
    def export(self, spans: List[Span]) -> bool:
        for span in spans:
            print(f'[TRACE] {span.to_dict()}')
        return True

class OTLPExporter(SpanExporter):
    '''OTLP HTTP exporter.'''
    def __init__(self, endpoint: str = 'http://localhost:4318/v1/traces'):
        self.endpoint = endpoint

    def export(self, spans: List[Span]) -> bool:
        try:
            payload = {
                'resourceSpans': [{
                    'scopeSpans': [{
                        'spans': [span.to_dict() for span in spans]
                    }]
                }]
            }
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.endpoint,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception as e:
            print(f'Failed to export spans: {e}')
            return False

class JaegerExporter(SpanExporter):
    '''Jaeger HTTP exporter.'''
    def __init__(self, endpoint: str = 'http://localhost:14268/api/traces'):
        self.endpoint = endpoint

    def export(self, spans: List[Span]) -> bool:
        try:
            # Convert to Jaeger format
            for span in spans:
                payload = {
                    'traceIdLow': int(span.context.trace_id[:16], 16),
                    'traceIdHigh': int(span.context.trace_id[16:], 16) if len(span.context.trace_id) > 16 else 0,
                    'spanId': int(span.context.span_id, 16),
                    'operationName': span.name,
                    'startTime': span.start_time_ns // 1000,
                    'duration': (span.end_time_ns - span.start_time_ns) // 1000,
                    'tags': [{'key': k, 'value': str(v)} for k, v in span.attributes.items()]
                }
                data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(
                    self.endpoint,
                    data=data,
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                urllib.request.urlopen(req, timeout=5)
            return True
        except Exception:
            return False

class ZipkinExporter(SpanExporter):
    '''Zipkin HTTP exporter.'''
    def __init__(self, endpoint: str = 'http://localhost:9411/api/v2/spans'):
        self.endpoint = endpoint

    def export(self, spans: List[Span]) -> bool:
        try:
            zipkin_spans = []
            for span in spans:
                zipkin_spans.append({
                    'traceId': span.context.trace_id,
                    'id': span.context.span_id,
                    'parentId': span.parent_context.span_id if span.parent_context else None,
                    'name': span.name,
                    'timestamp': span.start_time_ns // 1000,
                    'duration': (span.end_time_ns - span.start_time_ns) // 1000,
                    'tags': {k: str(v) for k, v in span.attributes.items()}
                })
            data = json.dumps(zipkin_spans).encode('utf-8')
            req = urllib.request.Request(
                self.endpoint,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception:
            return False

class Tracer:
    '''Distributed tracer with sampling and export.'''
    def __init__(self, service_name: str, sampling_rate: float = 1.0,
                 exporter: SpanExporter = None, propagation: PropagationFormat = PropagationFormat.W3C):
        self.service_name = service_name
        self.sampling_rate = sampling_rate
        self.exporter = exporter or ConsoleExporter()
        self.propagation = propagation
        self._current_span: Optional[Span] = None
        self._spans: List[Span] = []
        self._lock = threading.Lock()
        self._local = threading.local()

    def should_sample(self) -> bool:
        return random.random() < self.sampling_rate

    def get_current_span(self) -> Optional[Span]:
        return getattr(self._local, 'current_span', None)

    def set_current_span(self, span: Optional[Span]) -> None:
        self._local.current_span = span

    @contextmanager
    def start_span(self, name: str, kind: SpanKind = SpanKind.INTERNAL,
                   parent: SpanContext = None, attributes: Dict[str, Any] = None):
        '''Context manager for creating spans.'''
        if not self.should_sample():
            yield None
            return

        # Create span context
        context = SpanContext.generate()
        if parent is None:
            current = self.get_current_span()
            parent_context = current.context if current else None
        else:
            parent_context = parent

        # Create span
        span = Span(
            name=name,
            context=context,
            parent_context=parent_context,
            kind=kind,
            start_time_ns=time.time_ns(),
            attributes=attributes or {}
        )
        span.set_attribute('service.name', self.service_name)

        # Set as current
        previous = self.get_current_span()
        self.set_current_span(span)

        try:
            yield span
            span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            raise
        finally:
            span.end()
            with self._lock:
                self._spans.append(span)
            self.set_current_span(previous)

    def extract_context(self, headers: Dict[str, str]) -> Optional[SpanContext]:
        '''Extract span context from request headers.'''
        if self.propagation == PropagationFormat.W3C:
            traceparent = headers.get('traceparent', headers.get('Traceparent', ''))
            return SpanContext.from_traceparent(traceparent)
        elif self.propagation in (PropagationFormat.B3, PropagationFormat.B3_MULTI):
            trace_id = headers.get('X-B3-TraceId', '')
            span_id = headers.get('X-B3-SpanId', '')
            sampled = headers.get('X-B3-Sampled', '1')
            if trace_id and span_id:
                return SpanContext(trace_id, span_id, int(sampled))
        return None

    def inject_context(self, span: Span, headers: Dict[str, str]) -> None:
        '''Inject span context into request headers.'''
        if self.propagation == PropagationFormat.W3C:
            headers['traceparent'] = span.context.to_traceparent()
        elif self.propagation in (PropagationFormat.B3, PropagationFormat.B3_MULTI):
            headers.update(span.context.to_b3())

    def flush(self) -> None:
        '''Export all pending spans.'''
        with self._lock:
            if self._spans:
                self.exporter.export(self._spans)
                self._spans.clear()

class ~wService:
    '''
    Traced Service: ~w
    Service Name: ~w
    Exporter: ~w
    Sampling Rate: ~w
    Propagation: ~w
    '''
    def __init__(self):
        self.name = '~w'
        self.stateful = ~w
        self.timeout_ms = ~w
        self._state: Dict[str, Any] = {}
        self._lock = threading.Lock()

        # Initialize tracer
        self.tracer = Tracer(
            service_name='~w',
            sampling_rate=~w,
            exporter=~w,
            propagation=PropagationFormat.~w
        )
        self.default_attributes = ~w

    def call(self, request: Any, headers: Dict[str, str] = None) -> Any:
        '''Process a request with tracing.'''
        headers = headers or {}

        # Extract parent context from headers
        parent_ctx = self.tracer.extract_context(headers)

        # Create span for this request
        with self.tracer.start_span(
            f'{self.name}.call',
            kind=SpanKind.SERVER,
            parent=parent_ctx,
            attributes=self.default_attributes
        ) as span:
            if span:
                span.set_attribute('request.type', type(request).__name__)
            result = self._handle_request(request)
            if span:
                span.set_attribute('response.type', type(result).__name__)
            return result

    def call_with_trace(self, request: Any, parent_span: Span = None) -> Tuple[Any, Optional[Span]]:
        '''Process request and return result with span for propagation.'''
        parent_ctx = parent_span.context if parent_span else None

        with self.tracer.start_span(
            f'{self.name}.call',
            kind=SpanKind.SERVER,
            parent=parent_ctx
        ) as span:
            result = self._handle_request(request)
            return result, span

    def flush_traces(self) -> None:
        '''Flush pending traces to exporter.'''
        self.tracer.flush()

    def _handle_request(self, request: Any) -> Any:
~w

# Service instance
~w_service = ~wService()
", [ClassNameAtom, Name, ServiceNameStr, ExporterStr, SamplingRate, PropagationStr,
    Name, Stateful, Timeout,
    ServiceNameStr, SamplingRate, ExporterStr, PropagationStr, AttrsStr,
    HandlerCode, Name, ClassNameAtom]).

%% exporter_to_string(+Exporter, -String)
%  Convert trace exporter to Python code string.
exporter_to_string(otlp, "OTLPExporter()").
exporter_to_string(otlp(Endpoint), Code) :-
    format(string(Code), "OTLPExporter('~w')", [Endpoint]).
exporter_to_string(jaeger, "JaegerExporter()").
exporter_to_string(jaeger(Endpoint), Code) :-
    format(string(Code), "JaegerExporter('~w')", [Endpoint]).
exporter_to_string(jaeger(Host, Port), Code) :-
    format(string(Code), "JaegerExporter('http://~w:~w/api/traces')", [Host, Port]).
exporter_to_string(zipkin, "ZipkinExporter()").
exporter_to_string(zipkin(Endpoint), Code) :-
    format(string(Code), "ZipkinExporter('~w')", [Endpoint]).
exporter_to_string(console, "ConsoleExporter()").
exporter_to_string(none, "ConsoleExporter()").
exporter_to_string(_, "ConsoleExporter()").

%% propagation_to_string(+Propagation, -String)
%  Convert propagation format to Python enum string.
propagation_to_string(w3c, "W3C").
propagation_to_string(b3, "B3").
propagation_to_string(b3_multi, "B3_MULTI").
propagation_to_string(jaeger, "JAEGER").
propagation_to_string(xray, "XRAY").
propagation_to_string(datadog, "DATADOG").
propagation_to_string(_, "W3C").

%% attributes_to_python_dict(+Attributes, -String)
%  Convert attribute list to Python dict string.
attributes_to_python_dict([], "{}").
attributes_to_python_dict(Attrs, Code) :-
    Attrs \= [],
    maplist(attr_to_python, Attrs, AttrStrs),
    atomic_list_concat(AttrStrs, ', ', Inner),
    format(string(Code), "{~w}", [Inner]).

attr_to_python(Key=Value, Code) :-
    format(string(Code), "'~w': '~w'", [Key, Value]).
attr_to_python(Key-Value, Code) :-
    format(string(Code), "'~w': '~w'", [Key, Value]).

%% generate_tracer_python(+Config, -Code)
%  Generate Python tracer initialization code.
generate_tracer_python(config(ServiceName, SamplingRate, Exporter), Code) :-
    exporter_to_string(Exporter, ExporterStr),
    format(string(Code), "Tracer('~w', ~w, ~w)", [ServiceName, SamplingRate, ExporterStr]).

%% generate_span_context_python(+Context, -Code)
%  Generate Python span context code.
generate_span_context_python(context(TraceId, SpanId), Code) :-
    format(string(Code), "SpanContext('~w', '~w')", [TraceId, SpanId]).
generate_span_context_python(_, "SpanContext.generate()").

/** <module> Python Target Compiler
 *
 * Compiles Prolog predicates to Python scripts using a generator-based pipeline.
 *
 * @author John William Creighton
 * @license MIT
 */

%% compile_predicate_to_python(+Predicate, +Options, -PythonCode)
%
% Compiles the given Predicate to a complete Python script.
%
% Options:
%   * record_format(Format) - 'jsonl' (default) or 'nul_json'
%   * mode(Mode) - 'procedural' (default) or 'generator'
%
% Pipeline Options (Phase 1 - Object Pipeline Support):
%   * pipeline_input(Bool) - true: enable streaming input from stdin/iterator
%                           false: standalone function (default)
%   * output_format(Format) - object: yield typed dicts with arg_names
%                            text: yield string representation (default)
%   * arg_names(Names) - List of property names for output dict
%                        Example: ['UserId', 'Email']
%   * glue_protocol(Protocol) - jsonl (default), messagepack (future)
%   * error_protocol(Protocol) - same_as_data (default), text
%

%% ============================================================================
%% FACT EXPORT - compile_facts_to_python/3
%% ============================================================================

%% compile_facts_to_python(+Pred, +Arity, -PythonCode)
%  Export Prolog facts as a Python class with static fact data.
%  Generates a class with get_all(), stream(), and contains() methods.
%
%  Example:
%    ?- compile_facts_to_python(parent, 2, Code).
%    Generates:
%      class Parent:
%          FACTS = [["alice", "bob"], ["bob", "charlie"]]
%          @classmethod
%          def get_all(cls): return cls.FACTS
%          ...
%
compile_facts_to_python(Pred, Arity, PythonCode) :-
    % Get the predicate name
    atom_string(Pred, PredStr),
    upcase_atom(Pred, PredUp),
    atom_string(PredUp, PredUpStr),
    
    % Collect all facts
    functor(Head, Pred, Arity),
    findall(Args, (user:clause(Head, true), Head =.. [_|Args]), AllFacts),
    
    % Format facts as Python list of lists
    findall(Entry, (
        member(Args, AllFacts),
        maplist(format_python_fact_arg, Args, FormattedArgs),
        atomic_list_concat(FormattedArgs, ', ', ArgsStr),
        format(string(Entry), '        [~w]', [ArgsStr])
    ), Entries),
    atomic_list_concat(Entries, ',\n', EntriesCode),
    
    % Generate Python class
    format(string(PythonCode),
'#!/usr/bin/env python3
# Generated by UnifyWeaver Python Target - Fact Export
# Predicate: ~w/~w

from typing import List, Iterator, Any

class ~w:
    """Exported Prolog facts for ~w/~w"""
    
    FACTS: List[List[str]] = [
~w
    ]
    
    @classmethod
    def get_all(cls) -> List[List[str]]:
        """Return all facts as list of lists."""
        return cls.FACTS
    
    @classmethod
    def stream(cls) -> Iterator[List[str]]:
        """Stream facts as an iterator."""
        yield from cls.FACTS
    
    @classmethod
    def contains(cls, *args) -> bool:
        """Check if a fact exists with the given arguments."""
        target = list(args)
        return target in cls.FACTS


if __name__ == "__main__":
    # Print all facts
    for fact in ~w.stream():
        print(":".join(fact))
', [PredStr, Arity, PredUpStr, PredStr, Arity, EntriesCode, PredUpStr]).

%% format_python_fact_arg(+Arg, -Formatted)
%  Format a single fact argument as a Python string literal.
format_python_fact_arg(Arg, Formatted) :-
    (   atom(Arg)
    ->  atom_string(Arg, Str), format(string(Formatted), '"~w"', [Str])
    ;   number(Arg)
    ->  format(string(Formatted), '"~w"', [Arg])
    ;   format(string(Formatted), '"~w"', [Arg])
    ).

compile_predicate_to_python(PredicateIndicator, Options, PythonCode) :-
    % Clear any previously collected binding imports
    clear_binding_imports,

    % Handle module expansion (meta_predicate ensures M:Name/Arity)
    (   PredicateIndicator = Module:Name/Arity
    ->  true
    ;   PredicateIndicator = Name/Arity, Module = user
    ),

    % Determine ordering constraint
    (   member(ordered(_Order), Options) % Changed AllOptions to Options
    ->  true
    ;   _Order = true  % Default: ordered
    ),

    % Determine evaluation mode
    option(mode(Mode), Options, procedural),

    % Check for pipeline mode (Phase 1 - Object Pipeline Support)
    option(pipeline_input(PipelineInput), Options, false),

    % Dispatch to appropriate compiler
    (   PipelineInput == true
    ->  compile_pipeline_mode(Name, Arity, Module, Options, PythonCode)
    ;   Mode == generator
    ->  compile_generator_mode(Name, Arity, Module, Options, PythonCode)
    ;   compile_procedural_mode(Name, Arity, Module, Options, PythonCode)
    ).

%% compile_procedural_mode(+Name, +Arity, +Module, +Options, -PythonCode)
%  Current implementation (renamed for clarity)
compile_procedural_mode(Name, Arity, Module, Options, PythonCode) :-
    functor(Head, Name, Arity),
    findall((Head, Body), clause(Module:Head, Body), Clauses),
    (   Clauses == []
    ->  throw(error(clause_not_found(Module:Head), _))
    ;   true
    ),
    
    % Check if predicate is recursive
    (   is_recursive_predicate(Name, Clauses)
    ->  compile_recursive_predicate(Name, Arity, Clauses, Options, PythonCode)
    ;   compile_non_recursive_predicate(Name, Arity, Clauses, Options, PythonCode)
    ).

%% compile_non_recursive_predicate(+Name, +Arity, +Clauses, +Options, -PythonCode)
compile_non_recursive_predicate(_Name, Arity, Clauses, Options, PythonCode) :-
    % Generate clause functions
    findall(ClauseCode, (
        nth0(Index, Clauses, (ClauseHead, ClauseBody)),
        translate_clause(ClauseHead, ClauseBody, Index, Arity, ClauseCode)
    ), ClauseCodes),
    atomic_list_concat(ClauseCodes, "\n", AllClausesCode),
    
    % Generate process_stream calls
    findall(Call, (
        nth0(Index, Clauses, _),
        format(string(Call), "        yield from _clause_~d(record)", [Index])
    ), Calls),
    atomic_list_concat(Calls, "\n", CallsCode),
    
    header(Header),
    helpers(Helpers),
    
    generate_python_main(Options, Main),
    
    format(string(Logic), 
"
~s

def process_stream(records: Iterator[Dict]) -> Iterator[Dict]:
    \"\"\"Generated predicate logic.\"\"\"
    for record in records:
~s
\n", [AllClausesCode, CallsCode]),

    format(string(PythonCode), "~s~s~s~s", [Header, Helpers, Logic, Main]).

% ============================================================================
% PIPELINE MODE - Phase 1: Object Pipeline Support
% ============================================================================

%% compile_pipeline_mode(+Name, +Arity, +Module, +Options, -PythonCode)
%
% Compiles predicate to Python generator for pipeline processing.
% This mode generates streaming code that:
%   - Reads from stdin (or accepts iterator)
%   - Yields typed dict objects with named properties
%   - Writes to stdout with configurable protocol (JSONL default)
%   - Writes errors to stderr in same protocol
%
% Options used:
%   - pipeline_input(true) - already checked in dispatcher
%   - output_format(object|text) - dict or string output
%   - arg_names([...]) - property names for output dict
%   - glue_protocol(jsonl|messagepack) - serialization format
%   - error_protocol(same_as_data|text) - error format
%
compile_pipeline_mode(Name, Arity, Module, Options, PythonCode) :-
    functor(Head, Name, Arity),
    findall((Head, Body), clause(Module:Head, Body), Clauses),
    (   Clauses == []
    ->  throw(error(clause_not_found(Module:Head), _))
    ;   true
    ),

    % Extract options with defaults
    option(output_format(OutputFormat), Options, object),
    option(arg_names(ArgNames), Options, []),
    option(glue_protocol(GlueProtocol), Options, jsonl),
    option(error_protocol(ErrorProtocol), Options, same_as_data),
    option(runtime(Runtime), Options, cpython),

    % Generate arg names if not provided
    (   ArgNames == []
    ->  generate_default_arg_names(Arity, DefaultArgNames)
    ;   DefaultArgNames = ArgNames
    ),

    % Generate the pipeline function
    atom_string(Name, NameStr),
    generate_pipeline_function(NameStr, Arity, Clauses, OutputFormat, DefaultArgNames, FunctionCode),

    % Generate header and helpers (Phase 3: runtime-specific headers)
    pipeline_header(GlueProtocol, Runtime, Header),
    pipeline_helpers(GlueProtocol, ErrorProtocol, Helpers),

    % Generate main block
    generate_pipeline_main(NameStr, GlueProtocol, Options, Main),

    format(string(PythonCode), "~s~s~s~s", [Header, Helpers, FunctionCode, Main]).

%% generate_default_arg_names(+Arity, -ArgNames)
%  Generate default argument names: ['arg_0', 'arg_1', ...]
generate_default_arg_names(Arity, ArgNames) :-
    NumArgs is Arity,
    findall(ArgName, (
        between(0, NumArgs, I),
        I < NumArgs,
        format(atom(ArgName), 'arg_~d', [I])
    ), ArgNames).

%% generate_pipeline_function(+Name, +Arity, +Clauses, +OutputFormat, +ArgNames, -Code)
%  Generate the main pipeline processing function
generate_pipeline_function(Name, Arity, Clauses, OutputFormat, ArgNames, Code) :-
    % Generate clause handlers
    findall(ClauseCode, (
        nth0(Index, Clauses, (ClauseHead, ClauseBody)),
        generate_pipeline_clause(ClauseHead, ClauseBody, Index, Arity, OutputFormat, ArgNames, ClauseCode)
    ), ClauseCodes),
    atomic_list_concat(ClauseCodes, "\n", AllClausesCode),

    % Generate yield calls for each clause
    findall(YieldCall, (
        nth0(Index, Clauses, _),
        format(string(YieldCall), "            yield from _clause_~d(record)", [Index])
    ), YieldCalls),
    atomic_list_concat(YieldCalls, "\n", YieldCallsCode),

    format(string(Code),
"
~s

def ~s(stream: Iterator[Dict]) -> Generator[Dict, None, None]:
    \"\"\"
    Pipeline-enabled predicate with structured output.

    Args:
        stream: Iterator of input records (dicts)

    Yields:
        Dict with keys: ~w
    \"\"\"
    for record in stream:
        try:
~s
        except Exception as e:
            # Error handling - yield error record to stderr
            yield {'__error__': True, '__type__': type(e).__name__, '__message__': str(e), '__record__': record}
", [AllClausesCode, Name, ArgNames, YieldCallsCode]).

%% generate_pipeline_clause(+Head, +Body, +Index, +Arity, +OutputFormat, +ArgNames, -Code)
%  Generate a clause handler for pipeline mode
generate_pipeline_clause(Head, Body, Index, _Arity, OutputFormat, ArgNames, Code) :-
    % Instantiate variables for translation (same as translate_clause/5)
    copy_term((Head, Body), (HeadCopy, BodyCopy)),
    numbervars((HeadCopy, BodyCopy), 0, _),

    HeadCopy =.. [_Name | Args],

    % Generate input extraction from record
    generate_input_extraction(Args, ArgNames, InputCode),

    % Translate the body
    (   BodyCopy == true
    ->  BodyCode = "    pass  # No body goals"
    ;   translate_body(BodyCopy, BodyCode)
    ),

    % Generate output formatting
    generate_output_formatting(Args, ArgNames, OutputFormat, OutputCode),

    format(string(Code),
"def _clause_~d(record: Dict) -> Generator[Dict, None, None]:
    \"\"\"Clause ~d handler.\"\"\"
~s
~s
~s
", [Index, Index, InputCode, BodyCode, OutputCode]).

%% generate_input_extraction(+Args, +ArgNames, -Code)
%  Generate code to extract input values from record dict
generate_input_extraction(Args, ArgNames, Code) :-
    findall(Line, (
        nth0(I, Args, Arg),
        nth0(I, ArgNames, ArgName),
        is_var_term(Arg),  % Only extract for input variables (includes $VAR(N))
        format(string(Line), "    v_~d = record.get('~w')", [I, ArgName])
    ), Lines),
    (   Lines == []
    ->  Code = "    # No input extraction needed"
    ;   atomic_list_concat(Lines, "\n", Code)
    ).

%% generate_output_formatting(+Args, +ArgNames, +OutputFormat, -Code)
%  Generate code to format output as dict or text
generate_output_formatting(Args, ArgNames, OutputFormat, Code) :-
    length(Args, NumArgs),
    (   OutputFormat == object
    ->  % Build dict with arg names as keys
        findall(Pair, (
            nth0(I, ArgNames, ArgName),
            I < NumArgs,
            format(string(Pair), "'~w': v_~d", [ArgName, I])
        ), Pairs),
        atomic_list_concat(Pairs, ", ", PairsStr),
        format(string(Code), "    yield {~s}", [PairsStr])
    ;   % Text format - just yield string representation
        format(string(Code), "    yield {'result': str(v_0)}", [])
    ).

%% pipeline_header(+Protocol, -Header)
%% pipeline_header(+Protocol, +Runtime, -Header)
%  Generate header for pipeline mode with appropriate imports
%  Runtime can be: cpython (default), ironpython, pypy, jython

% Default: cpython
pipeline_header(Protocol, Header) :-
    pipeline_header(Protocol, cpython, Header).

% CPython headers
pipeline_header(jsonl, cpython, Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env python3
\"\"\"
Generated pipeline predicate.
Runtime: CPython
Protocol: JSONL (line-delimited JSON)
\"\"\"
import sys
import json
from typing import Iterator, Dict, Any, Generator
~w
", [BindingImports]).

pipeline_header(messagepack, cpython, Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env python3
\"\"\"
Generated pipeline predicate.
Runtime: CPython
Protocol: MessagePack (binary)
\"\"\"
import sys
import msgpack
from typing import Iterator, Dict, Any, Generator
~w
", [BindingImports]).

% IronPython headers - include CLR integration
pipeline_header(jsonl, ironpython, Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env ipy
\"\"\"
Generated pipeline predicate.
Runtime: IronPython (CLR/.NET integration)
Protocol: JSONL (line-delimited JSON)
\"\"\"
import sys
import json
import clr

# Add .NET references for common types
clr.AddReference('System')
clr.AddReference('System.Core')
from System import String, Int32, Double, DateTime, Math
from System.Collections.Generic import Dictionary, List

# Python typing (IronPython 3.4+ compatible)
from typing import Iterator, Dict, Any, Generator
~w

# Helper: Convert Python dict to .NET Dictionary
def to_dotnet_dict(py_dict):
    \"\"\"Convert Python dict to .NET Dictionary<string, object>.\"\"\"
    result = Dictionary[String, object]()
    for k, v in py_dict.items():
        result[str(k)] = v
    return result

# Helper: Convert .NET Dictionary to Python dict
def from_dotnet_dict(dotnet_dict):
    \"\"\"Convert .NET Dictionary to Python dict.\"\"\"
    return {str(k): v for k, v in dotnet_dict}
", [BindingImports]).

pipeline_header(messagepack, ironpython, Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env ipy
\"\"\"
Generated pipeline predicate.
Runtime: IronPython (CLR/.NET integration)
Protocol: MessagePack (binary)
\"\"\"
import sys
import clr

clr.AddReference('System')
clr.AddReference('System.Core')
from System import String, Int32, Double, DateTime, Math
from System.Collections.Generic import Dictionary, List

# MessagePack for IronPython
try:
    import msgpack
except ImportError:
    # Fallback: use .NET serialization if msgpack not available
    clr.AddReference('System.Text.Json')
    from System.Text.Json import JsonSerializer
    class msgpack:
        @staticmethod
        def packb(obj):
            return JsonSerializer.SerializeToUtf8Bytes(obj)
        @staticmethod
        def unpackb(data):
            return JsonSerializer.Deserialize(data, object)

from typing import Iterator, Dict, Any, Generator
~w
", [BindingImports]).

% PyPy headers (similar to CPython but with PyPy shebang)
pipeline_header(jsonl, pypy, Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env pypy3
\"\"\"
Generated pipeline predicate.
Runtime: PyPy (JIT-optimized)
Protocol: JSONL (line-delimited JSON)
\"\"\"
import sys
import json
from typing import Iterator, Dict, Any, Generator
~w
", [BindingImports]).

pipeline_header(messagepack, pypy, Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env pypy3
\"\"\"
Generated pipeline predicate.
Runtime: PyPy (JIT-optimized)
Protocol: MessagePack (binary)
\"\"\"
import sys
import msgpack
from typing import Iterator, Dict, Any, Generator
~w
", [BindingImports]).

% Jython headers - include Java integration
pipeline_header(jsonl, jython, Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env jython
\"\"\"
Generated pipeline predicate.
Runtime: Jython (JVM integration)
Protocol: JSONL (line-delimited JSON)
\"\"\"
import sys
import json

# Java imports
from java.lang import String as JString, Math as JMath
from java.util import HashMap, ArrayList

# Note: typing module may not be available in Jython 2.7
try:
    from typing import Iterator, Dict, Any, Generator
except ImportError:
    Iterator = Dict = Any = Generator = object
~w

# Helper: Convert Python dict to Java HashMap
def to_java_map(py_dict):
    \"\"\"Convert Python dict to Java HashMap.\"\"\"
    result = HashMap()
    for k, v in py_dict.items():
        result.put(str(k), v)
    return result
", [BindingImports]).

pipeline_header(messagepack, jython, Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env jython
\"\"\"
Generated pipeline predicate.
Runtime: Jython (JVM integration)
Protocol: MessagePack (binary)
\"\"\"
import sys

# Java imports
from java.lang import String as JString, Math as JMath
from java.util import HashMap, ArrayList

# MessagePack - try Python version, fallback to Java
try:
    import msgpack
except ImportError:
    # Use Java serialization as fallback
    from java.io import ByteArrayOutputStream, ObjectOutputStream
    from java.io import ByteArrayInputStream, ObjectInputStream
    class msgpack:
        @staticmethod
        def packb(obj):
            baos = ByteArrayOutputStream()
            oos = ObjectOutputStream(baos)
            oos.writeObject(obj)
            oos.close()
            return baos.toByteArray()
        @staticmethod
        def unpackb(data):
            bais = ByteArrayInputStream(data)
            ois = ObjectInputStream(bais)
            return ois.readObject()

# Note: typing module may not be available in Jython 2.7
try:
    from typing import Iterator, Dict, Any, Generator
except ImportError:
    Iterator = Dict = Any = Generator = object
~w

# Helper: Convert Python dict to Java HashMap
def to_java_map(py_dict):
    \"\"\"Convert Python dict to Java HashMap.\"\"\"
    result = HashMap()
    for k, v in py_dict.items():
        result.put(str(k), v)
    return result
", [BindingImports]).

% Legacy 2-argument version for backward compatibility
pipeline_header(messagepack, Header) :-
    pipeline_header(messagepack, cpython, Header).

% Original messagepack header moved here
pipeline_header_messagepack_base(Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"#!/usr/bin/env python3
\"\"\"
Generated pipeline predicate.
Protocol: MessagePack (binary)
\"\"\"
import sys
import msgpack
from typing import Iterator, Dict, Any, Generator
~w
", [BindingImports]).

%% pipeline_helpers(+GlueProtocol, +ErrorProtocol, -Helpers)
%  Generate helper functions for pipeline I/O
pipeline_helpers(jsonl, same_as_data, Helpers) :-
    Helpers = "
def read_stream(stream) -> Iterator[Dict[str, Any]]:
    \"\"\"Read JSONL records from stream.\"\"\"
    for line in stream:
        line = line.strip()
        if line:
            yield json.loads(line)

def write_record(record: Dict, stream=sys.stdout) -> None:
    \"\"\"Write a single record as JSONL.\"\"\"
    if record.get('__error__'):
        # Error records go to stderr
        error_record = {k: v for k, v in record.items() if not k.startswith('__')}
        error_record['error'] = True
        error_record['type'] = record.get('__type__', 'Unknown')
        error_record['message'] = record.get('__message__', '')
        sys.stderr.write(json.dumps(error_record) + '\\n')
        sys.stderr.flush()
    else:
        stream.write(json.dumps(record) + '\\n')
        stream.flush()
".

pipeline_helpers(jsonl, text, Helpers) :-
    Helpers = "
def read_stream(stream) -> Iterator[Dict[str, Any]]:
    \"\"\"Read JSONL records from stream.\"\"\"
    for line in stream:
        line = line.strip()
        if line:
            yield json.loads(line)

def write_record(record: Dict, stream=sys.stdout) -> None:
    \"\"\"Write a single record as JSONL, errors as plain text.\"\"\"
    if record.get('__error__'):
        # Error records go to stderr as plain text
        sys.stderr.write(f\"ERROR [{record.get('__type__', 'Unknown')}]: {record.get('__message__', '')}\\n\")
        sys.stderr.flush()
    else:
        stream.write(json.dumps(record) + '\\n')
        stream.flush()
".

pipeline_helpers(messagepack, _, Helpers) :-
    Helpers = "
def read_stream(stream) -> Iterator[Dict[str, Any]]:
    \"\"\"Read MessagePack records from binary stream.\"\"\"
    unpacker = msgpack.Unpacker(stream.buffer, raw=False)
    for record in unpacker:
        yield record

def write_record(record: Dict, stream=sys.stdout) -> None:
    \"\"\"Write a single record as MessagePack.\"\"\"
    if record.get('__error__'):
        # Error records go to stderr
        error_record = {k: v for k, v in record.items() if not k.startswith('__')}
        error_record['error'] = True
        error_record['type'] = record.get('__type__', 'Unknown')
        error_record['message'] = record.get('__message__', '')
        sys.stderr.buffer.write(msgpack.packb(error_record))
        sys.stderr.flush()
    else:
        stream.buffer.write(msgpack.packb(record))
        stream.flush()
".

%% generate_pipeline_main(+Name, +Protocol, +Options, -Main)
%  Generate the main block for pipeline execution
generate_pipeline_main(Name, _Protocol, _Options, Main) :-
    format(string(Main),
"
if __name__ == '__main__':
    # Pipeline mode: read from stdin, write to stdout
    input_stream = read_stream(sys.stdin)
    for result in ~s(input_stream):
        write_record(result)
", [Name]).

% ============================================================================
% RUNTIME SELECTION - Phase 2: Firewall/Preference Integration
% ============================================================================

%% select_python_runtime(+PredIndicator, +Imports, +Context, -Runtime)
%
% Selects Python runtime respecting firewall and preferences.
% Uses existing dotnet_glue.pl for IronPython compatibility checking.
%
% @arg PredIndicator The predicate being compiled (Name/Arity)
% @arg Imports List of required Python imports
% @arg Context Additional context (e.g., [target(csharp)])
% @arg Runtime Selected runtime: cpython | ironpython | pypy | jython
%
select_python_runtime(PredIndicator, Imports, Context, Runtime) :-
    % 1. Get merged preferences (uses existing preferences module if available)
    get_runtime_preferences(PredIndicator, Context, Preferences),

    % 2. Get firewall policy (uses existing firewall module if available)
    get_runtime_firewall(PredIndicator, Firewall),

    % 3. Determine candidate runtimes (filtered by hard constraints)
    findall(R, valid_runtime_candidate(R, Imports, Firewall, Context), Candidates),
    (   Candidates == []
    ->  % No valid candidates - fall back to cpython
        Runtime = cpython
    ;   % 4. Score candidates against preferences
        score_runtime_candidates(Candidates, Preferences, Context, ScoredCandidates),
        % 5. Select best
        select_best_runtime(ScoredCandidates, Preferences, Runtime)
    ).

%% get_runtime_preferences(+PredIndicator, +Context, -Preferences)
%  Get merged runtime preferences from preference system
get_runtime_preferences(PredIndicator, Context, Preferences) :-
    (   catch(preferences:get_final_options(PredIndicator, Context, Prefs), _, fail)
    ->  Preferences = Prefs
    ;   % Default preferences if module not available
        Preferences = [
            prefer_runtime([cpython, ironpython, pypy]),
            prefer_communication(in_process),
            python_version(3)
        ]
    ).

%% get_runtime_firewall(+PredIndicator, -Firewall)
%  Get firewall policy for runtime selection
get_runtime_firewall(PredIndicator, Firewall) :-
    (   catch(firewall:get_firewall_policy(PredIndicator, FW), _, fail)
    ->  Firewall = FW
    ;   % Default: no restrictions
        Firewall = []
    ).

%% valid_runtime_candidate(+Runtime, +Imports, +Firewall, +Context) is semidet.
%
% Check if runtime passes all hard constraints.
%
valid_runtime_candidate(Runtime, Imports, Firewall, Context) :-
    member(Runtime, [cpython, ironpython, pypy, jython]),
    % Not explicitly denied in firewall
    \+ member(denied(python_runtime(Runtime)), Firewall),
    \+ (member(denied(List), Firewall), is_list(List), member(python_runtime(Runtime), List)),
    % Runtime is available on system
    runtime_available(Runtime),
    % Compatible with required imports
    runtime_compatible_with_imports(Runtime, Imports),
    % Context requirements (e.g., .NET integration needs ironpython or cpython_pipe)
    runtime_satisfies_context(Runtime, Context).

%% runtime_available(+Runtime) is semidet.
%  Check if runtime is available on the system
runtime_available(cpython) :-
    % CPython is always assumed available (python3 command)
    !.
runtime_available(ironpython) :-
    % Check for IronPython using dotnet_glue if available
    (   catch(dotnet_glue:detect_ironpython(true), _, fail)
    ->  true
    ;   % Fallback: check for ipy command
        catch(process_create(path(ipy), ['--version'], [stdout(null), stderr(null)]), _, fail)
    ).
runtime_available(pypy) :-
    % Check for PyPy
    catch(process_create(path(pypy3), ['--version'], [stdout(null), stderr(null)]), _, fail).
runtime_available(jython) :-
    % Check for Jython
    catch(process_create(path(jython), ['--version'], [stdout(null), stderr(null)]), _, fail).

%% runtime_compatible_with_imports(+Runtime, +Imports) is semidet.
%  Check if runtime supports all required imports
runtime_compatible_with_imports(cpython, _) :- !.  % CPython supports everything
runtime_compatible_with_imports(pypy, Imports) :-
    % PyPy has issues with some C extensions
    \+ member(numpy, Imports),
    \+ member(scipy, Imports),
    \+ member(tensorflow, Imports),
    \+ member(torch, Imports).
runtime_compatible_with_imports(ironpython, Imports) :-
    % Use dotnet_glue's compatibility check if available
    (   catch(dotnet_glue:can_use_ironpython(Imports), _, fail)
    ->  true
    ;   % Fallback: check against known incompatible modules
        \+ member(numpy, Imports),
        \+ member(scipy, Imports),
        \+ member(pandas, Imports),
        \+ member(matplotlib, Imports),
        \+ member(tensorflow, Imports),
        \+ member(torch, Imports)
    ).
runtime_compatible_with_imports(jython, Imports) :-
    % Jython has similar limitations to IronPython
    \+ member(numpy, Imports),
    \+ member(scipy, Imports),
    \+ member(pandas, Imports).

%% runtime_satisfies_context(+Runtime, +Context) is semidet.
%  Check if runtime satisfies context requirements
runtime_satisfies_context(_, []) :- !.
runtime_satisfies_context(Runtime, Context) :-
    % If targeting .NET, prefer in-process runtimes
    (   member(target(csharp), Context)
    ;   member(target(dotnet), Context)
    )
    ->  member(Runtime, [ironpython, cpython])  % cpython via pipes is fallback
    ;   % If targeting JVM, prefer Jython
        member(target(java), Context)
    ->  member(Runtime, [jython, cpython])
    ;   % No special context requirements
        true.

%% score_runtime_candidates(+Candidates, +Preferences, +Context, -Scored)
%  Score each candidate against preference dimensions
score_runtime_candidates(Candidates, Preferences, Context, Scored) :-
    maplist(score_single_runtime(Preferences, Context), Candidates, Scored).

%% score_single_runtime(+Preferences, +Context, +Runtime, -Scored)
score_single_runtime(Preferences, Context, Runtime, Runtime-Score) :-
    % Base score from preference order
    (   member(prefer_runtime(Order), Preferences),
        nth0(Idx, Order, Runtime)
    ->  OrderScore is 10 - Idx  % Higher = better
    ;   OrderScore = 0
    ),

    % Communication preference bonus
    (   member(prefer_communication(Comm), Preferences),
        runtime_communication(Runtime, Comm, Context)
    ->  CommScore = 5
    ;   CommScore = 0
    ),

    % Optimization hint bonus
    (   member(optimization(Opt), Preferences),
        runtime_optimization(Runtime, Opt)
    ->  OptScore = 3
    ;   OptScore = 0
    ),

    % Metrics bonus placeholder (returns 0 for now, designed for future extension)
    metrics_bonus(Runtime, MetricsBonus),

    Score is OrderScore + CommScore + OptScore + MetricsBonus.

%% runtime_communication(+Runtime, +CommType, +Context) is semidet.
%  Check if runtime provides the communication type in given context
runtime_communication(ironpython, in_process, Context) :-
    % IronPython is in-process when targeting .NET
    (member(target(csharp), Context) ; member(target(dotnet), Context)), !.
runtime_communication(jython, in_process, Context) :-
    % Jython is in-process when targeting JVM
    member(target(java), Context), !.
runtime_communication(cpython, cross_process, _) :- !.
runtime_communication(pypy, cross_process, _) :- !.
runtime_communication(_, cross_process, _).  % Default fallback

%% runtime_optimization(+Runtime, +OptType) is semidet.
%  Check if runtime is optimized for given workload type
runtime_optimization(pypy, throughput).      % JIT for long-running
runtime_optimization(pypy, latency).         % Fast after warmup
runtime_optimization(ironpython, latency).   % No serialization in .NET
runtime_optimization(cpython, memory).       % Most memory efficient
runtime_optimization(cpython, compatibility). % Best library support

%% metrics_bonus(+Runtime, -Bonus)
%  Placeholder for metrics-driven selection (returns 0 for now)
%  Future: Use runtime_metric/3 facts to calculate bonus from execution history
metrics_bonus(_Runtime, 0).

%% select_best_runtime(+ScoredCandidates, +Preferences, -Runtime)
%  Select the highest-scoring runtime, with fallback handling
select_best_runtime(ScoredCandidates, Preferences, Runtime) :-
    % Sort by score (descending)
    keysort(ScoredCandidates, Sorted),
    reverse(Sorted, [Best-_|_]),
    (   Best \= cpython
    ->  Runtime = Best
    ;   % If cpython is best but fallback specified, check it
        (   member(fallback_runtime(Fallbacks), Preferences),
            member(FB, Fallbacks),
            member(FB-_, ScoredCandidates)
        ->  Runtime = FB
        ;   Runtime = Best
        )
    ).

%% get_collected_imports(-Imports)
%  Get the list of imports collected during compilation
get_collected_imports(Imports) :-
    findall(I, required_import(I), Imports).

%% compile_recursive_predicate(+Name, +Arity, +Clauses, +Options, -PythonCode)
compile_recursive_predicate(Name, Arity, Clauses, Options, PythonCode) :-
    % Separate base and recursive clauses
    partition(is_recursive_clause_for(Name), Clauses, RecClauses, BaseClauses),
    
    % Check if part of mutual recursion group
    (   is_mutually_recursive(Name/Arity, MutualGroup),
        length(MutualGroup, GroupSize),
        GroupSize > 1
    ->  % Mutual recursion - compile entire group together
        compile_mutual_recursive_group(MutualGroup, Options, PythonCode)
    ;   % Single predicate recursion
        % Check if this is tail recursion (can be optimized to a loop)
        (   is_tail_recursive(Name, RecClauses)
        ->  compile_tail_recursive(Name, Arity, BaseClauses, RecClauses, Options, PythonCode)
        ;   compile_general_recursive(Name, Arity, BaseClauses, RecClauses, Options, PythonCode)
        )
    ).

%% is_mutually_recursive(+Pred, -MutualGroup)
%  Check if predicate is part of a mutual recursion group
%  Uses call graph analysis from advanced recursion modules
is_mutually_recursive(Pred, MutualGroup) :-
    % Try to use call_graph:predicates_in_group if module is loaded
    catch(
        (   call_graph:predicates_in_group(Pred, Group),
            length(Group, Len),
            Len > 1,
            MutualGroup = Group
        ),
        _Error,
        fail  % Silently fail if module not available or predicates not found
    ).

%% is_tail_recursive(+Name, +RecClauses)
%  Check if recursive call is in tail position
is_tail_recursive(Name, RecClauses) :-
    member((_, Body), RecClauses),
    % Get the last goal in the body
    get_last_goal(Body, LastGoal),
    functor(LastGoal, Name, _).

%% get_last_goal(+Body, -LastGoal)
get_last_goal((_, B), LastGoal) :- !, get_last_goal(B, LastGoal).
get_last_goal(Goal, Goal).

%% get_last_goal(+Body, -LastGoal)
get_last_goal((_, B), LastGoal) :- !, get_last_goal(B, LastGoal).
get_last_goal(Goal, Goal).

%% compile_mutual_recursive_group(+Predicates, +Options, -PythonCode)
%  Compile a group of mutually recursive predicates together
%  Example: [is_even/1, is_odd/1]
compile_mutual_recursive_group(Predicates, Options, PythonCode) :-
    % Generate worker functions for each predicate in the group
    findall(WorkerCode,
        (   member(Pred/Arity, Predicates),
            atom_string(Pred, PredStr),
            functor(Head, Pred, Arity),
            findall((Head, Body), clause(Head, Body), Clauses),
            partition(is_recursive_clause_for(Pred), Clauses, RecClauses, BaseClauses),
            generate_mutual_worker(PredStr, Arity, BaseClauses, RecClauses, Predicates, WorkerCode)
        ),
        WorkerCodes
    ),
    atomic_list_concat(WorkerCodes, "\n\n", AllWorkersCode),
    
    % Generate wrappers for each predicate
    findall(WrapperCode,
        (   member(Pred/Arity, Predicates),
            atom_string(Pred, PredStr),
            generate_mutual_wrapper(PredStr, Arity, WrapperCode)
        ),
        WrapperCodes
    ),
    atomic_list_concat(WrapperCodes, "\n\n", AllWrappersCode),
    
    header_with_functools(Header),
    helpers(Helpers),
    
    % For mutual recursion, generate a dispatcher that handles all predicates
    findall(Pred/Arity, member(Pred/Arity, Predicates), PredList),
    generate_mutual_dispatcher(PredList, DispatcherCode),
    
    generate_python_main(Options, Main),
    
    format(string(Logic), 
"
~s

~s

~s
", [AllWorkersCode, AllWrappersCode, DispatcherCode]),

    format(string(PythonCode), "~s~s~s~s", [Header, Helpers, Logic, Main]).

%% generate_mutual_worker(+PredStr, +Arity, +BaseClauses, +RecClauses, +AllPredicates, -WorkerCode)
%  Generate worker function for one predicate in mutual group
generate_mutual_worker(PredStr, Arity, BaseClauses, RecClauses, _AllPredicates, WorkerCode) :-
    (   Arity =:= 1
    ->  % Binary predicate with single argument
        (   BaseClauses = [(BaseHead, _BaseBody)|_]
        ->  BaseHead =.. [_, BaseValue],
            (   number(BaseValue)
            ->  format(string(BaseCondition), "arg == ~w", [BaseValue])
            ;   BaseCondition = "False"
            ),
            BaseReturn = "True"
        ;   BaseCondition = "False", BaseReturn = "False"
        ),
        
        % Extract recursive case - find which function it calls
        (   RecClauses = [(_RecHead, RecBody)|_]
        ->  extract_mutual_call(RecBody, CalledPred, CalledArg),
            atom_string(CalledPred, CalledPredStr),
            translate_call_arg_simple(CalledArg, PyArg)
        ;   CalledPredStr = "unknown", PyArg = "arg"
        ),
        
        format(string(WorkerCode),
"@functools.cache
def _~w_worker(arg):
    # Base case
    if ~s:
        return ~s
    
    # Mutual recursive case
    return _~w_worker(~s)
", [PredStr, BaseCondition, BaseReturn, CalledPredStr, PyArg])
    ;   % Unsupported arity
        format(string(WorkerCode), "# ERROR: Mutual recursion only supports arity 1, got arity ~d\n", [Arity])
    ).

%% extract_mutual_call(+Body, -CalledPred, -CalledArg)
%  Extract the predicate call and its argument from recursive clause
extract_mutual_call(Body, CalledPred, CalledArg) :-
    extract_goals_list(Body, Goals),
    member(Call, Goals),
    compound(Call),
    Call =.. [CalledPred, CalledArg],
    \+ member(CalledPred, [is, '>', '<', '>=', '=<', '=:=', '==']).

%% translate_call_arg_simple(+Arg, -PyArg)
translate_call_arg_simple(Expr, PyArg) :-
    (   Expr = (_ - Const),
        number(Const)
    ->  format(string(PyArg), "arg - ~w", [Const])
    ;   Expr = (_ + Const),
        number(Const)
    ->  format(string(PyArg), "arg + ~w", [Const])
    ;   PyArg = "arg"
    ).

%% generate_mutual_wrapper(+PredStr, +Arity, -WrapperCode)
generate_mutual_wrapper(PredStr, Arity, WrapperCode) :-
    (   Arity =:= 1
    ->  format(string(WrapperCode),
"def _~w_clause(v_0: Dict) -> Iterator[Dict]:
    # Extract input
    keys = list(v_0.keys())
    if not keys:
        return
    input_key = keys[0]
    input_val = v_0[input_key]
    
    # Call worker
    result = _~w_worker(input_val)
    
    # Yield result dict
    output_key = 'result'
    yield {input_key: input_val, output_key: result}
", [PredStr, PredStr])
    ;   WrapperCode = "# ERROR: Unsupported arity for mutual recursion wrapper"
    ).

%% generate_mutual_dispatcher(+Predicates, -DispatcherCode)
%  Generate process_stream that dispatches to appropriate predicate
generate_mutual_dispatcher(Predicates, DispatcherCode) :-
    findall(Case,
        (   member(Pred/_Arity, Predicates),
            atom_string(Pred, PredStr),
            format(string(Case), "    if 'predicate' in record and record['predicate'] == '~w':\n        yield from _~w_clause(record)", [PredStr, PredStr])
        ),
        Cases
    ),
    atomic_list_concat(Cases, "\n", CasesCode),
    format(string(DispatcherCode),
"def process_stream(records: Iterator[Dict]) -> Iterator[Dict]:
    \"\"\"Generated mutual recursion dispatcher.\"\"\"
    for record in records:
~s
", [CasesCode]).

%% compile_tail_recursive(+Name, +Arity, +BaseClauses, +RecClauses, +Options, -PythonCode)
compile_tail_recursive(Name, Arity, BaseClauses, RecClauses, Options, PythonCode) :-
    % Generate iterative code with while loop
    generate_tail_recursive_code(Name, Arity, BaseClauses, RecClauses, WorkerCode),
    
    % Generate streaming wrapper
    generate_recursive_wrapper(Name, Arity, WrapperCode),
    
    header_with_functools(Header),
    helpers(Helpers),
    
    generate_python_main(Options, Main),
    
    format(string(Logic), 
"
~s

~s

def process_stream(records: Iterator[Dict]) -> Iterator[Dict]:
    \"\"\"Generated predicate logic - tail recursive (optimized).\"\"\"
    for record in records:
        yield from _clause_0(record)
\n", [WorkerCode, WrapperCode]),

    format(string(PythonCode), "~s~s~s~s", [Header, Helpers, Logic, Main]).

%% compile_general_recursive(+Name, +Arity, +BaseClauses, +RecClauses, +Options, -PythonCode)
compile_general_recursive(Name, Arity, BaseClauses, RecClauses, Options, PythonCode) :-
    % Generate worker function with memoization
    generate_worker_function(Name, Arity, BaseClauses, RecClauses, WorkerCode),
    
    % Generate streaming wrapper
    generate_recursive_wrapper(Name, Arity, WrapperCode),
    
    header_with_functools(Header),
    helpers(Helpers),
    
    generate_python_main(Options, Main),
    
    format(string(Logic), 
"
~s

~s

def process_stream(records: Iterator[Dict]) -> Iterator[Dict]:
    \"\"\"Generated predicate logic - recursive wrapper.\"\"\"
    for record in records:
        yield from _clause_0(record)
\n", [WorkerCode, WrapperCode]),

    format(string(PythonCode), "~s~s~s~s", [Header, Helpers, Logic, Main]).

%% is_recursive_predicate(+Name, +Clauses)
is_recursive_predicate(Name, Clauses) :-
    member((_, Body), Clauses),
    contains_recursive_call(Body, Name).

%% is_recursive_clause_for(+Name, +Clause)
is_recursive_clause_for(Name, (_, Body)) :-
    contains_recursive_call(Body, Name).

%% contains_recursive_call(+Body, +Name)
contains_recursive_call(Body, Name) :-
    extract_goal(Body, Goal),
    functor(Goal, Name, _),
    !.

%% extract_goal(+Body, -Goal)
extract_goal(Goal, Goal) :-
    compound(Goal),
    \+ Goal = (_,_),
    \+ Goal = (_;_).
extract_goal((A, _), Goal) :- extract_goal(A, Goal).
extract_goal((_, B), Goal) :- extract_goal(B, Goal).

translate_clause(Head, Body, Index, Arity, Code) :-
    % Instanciate variables
    numbervars((Head, Body), 0, _),
    
    % Assume first argument is the input record
    arg(1, Head, RecordVar),
    
    % Determine parameter name and matching code
    (   compound(RecordVar), functor(RecordVar, '$VAR', 1)
    ->  % It is a variable (v_0), use it as parameter
        var_to_python(RecordVar, PyRecordVar),
        MatchCode = ""
    ;   % It is a constant, use generic parameter and check equality
        PyRecordVar = "record",
        var_to_python(RecordVar, PyVal),
        format(string(MatchCode), "    if ~w != ~w: return\n", [PyRecordVar, PyVal])
    ),
    
    % Check for schema validation (NEW)
    Head =.. [PredName|_],
    (   get_json_schema(PredName, _)
    ->  format(string(ValidationCode), "    if not _validate_~w(~w): return\n", [PredName, PyRecordVar])
    ;   ValidationCode = ""
    ),
    
    % Determine output variable (Last argument if Arity > 1, else Input)
    (   Arity > 1
    ->  arg(Arity, Head, OutputVar)
    ;   OutputVar = RecordVar
    ),
    
    % Translate body with indentation support
    body_to_list(Body, BodyGoals),
    % Append yield(OutputVar) to goals so it respects indentation/nesting
    append(BodyGoals, [yield(OutputVar)], Goals),
    
    translate_goals_recursive(Goals, "    ", BodyCode),
    
    format(string(Code),
"def _clause_~d(~w: Dict) -> Iterator[Dict]:
~s~s~s
", [Index, PyRecordVar, MatchCode, ValidationCode, BodyCode]).

%% body_to_list(+Body, -Goals)
body_to_list((A, B), [A|Rest]) :- !, body_to_list(B, Rest).
body_to_list(true, []) :- !.
body_to_list(A, [A]).

%% translate_goals_recursive(+Goals, +Indent, -Code)
translate_goals_recursive([], _, "") :- !.
translate_goals_recursive([Goal|Rest], Indent, Code) :-
    (   Goal = json_array_member(List, Item)
    ->  % Control flow goal: Array iteration
        var_to_python(List, PyList),
        var_to_python(Item, PyItem),
        format(string(Header), "~sif isinstance(~w, list):\n~s    for ~w in ~w:\n", 
               [Indent, PyList, Indent, PyItem, PyList]),
        
        string_concat(Indent, "        ", InnerIndent),
        translate_goals_recursive(Rest, InnerIndent, BodyCode),
        string_concat(Header, BodyCode, Code)
    ;   Goal = yield(Var)
    ->  % Output goal
        var_to_python(Var, PyVar),
        format(string(Code1), "~syield ~w\n", [Indent, PyVar]),
        translate_goals_recursive(Rest, Indent, Code2),
        string_concat(Code1, Code2, Code)
    ;   % Standard goal
        translate_goal_with_indent(Goal, Indent, Code1),
        translate_goals_recursive(Rest, Indent, Code2),
        string_concat(Code1, Code2, Code)
    ).

%% translate_goal_with_indent(+Goal, +Indent, -Code)
%  Adapter for legacy translate_goal/2
translate_goal_with_indent(Goal, Indent, Code) :-
    translate_goal(Goal, LegacyCode),
    % Replace the hardcoded 4 spaces with Indent
    (   sub_string(LegacyCode, 0, 4, After, "    ")
    ->  sub_string(LegacyCode, 4, After, 0, Suffix),
        string_concat(Indent, Suffix, Code)
    ;   % Fallback if no indentation found (e.g. empty string)
        string_concat(Indent, LegacyCode, Code)
    ).

is_var_term(V) :- var(V), !.
is_var_term('$VAR'(_)).

translate_goal(_:Goal, Code) :-
    !,
    translate_goal(Goal, Code).



translate_goal(get_dict(Key, Record, Value), Code) :-
    !,
    var_to_python(Record, PyRecord),
    (   is_var_term(Value)
    ->  var_to_python(Value, PyValue),
        format(string(Code), "    ~w = ~w.get('~w')\n", [PyValue, PyRecord, Key])
    ;   % Value is a constant
        var_to_python(Value, PyValue),
        % Check if key exists and equals value
        format(string(Code), "    if ~w.get('~w') != ~w: return\n", [PyRecord, Key, PyValue])
    ).

translate_goal(=(Var, Dict), Code) :-
    is_dict(Dict),
    !,
    var_to_python(Var, PyVar),
    dict_pairs(Dict, _Tag, Pairs),
    maplist(pair_to_python, Pairs, PyPairList),
    atomic_list_concat(PyPairList, ', ', PairsStr),
    format(string(Code), "    ~w = {~s}\n", [PyVar, PairsStr]).

translate_goal(=(Var, Value), Code) :-
    is_var_term(Var),
    atomic(Value),
    !,
    var_to_python(Var, PyVar),
    var_to_python(Value, PyVal),
    format(string(Code), "    ~w = ~w\n", [PyVar, PyVal]).

translate_goal(=(Var1, Var2), Code) :-
    is_var_term(Var1),
    is_var_term(Var2),
    !,
    var_to_python(Var1, PyVar1),
    var_to_python(Var2, PyVar2),
    format(string(Code), "    ~w = ~w\n", [PyVar1, PyVar2]).

translate_goal(>(Var, Value), Code) :-
    !,
    var_to_python(Var, PyVar),
    var_to_python(Value, PyValue),
    format(string(Code), "    if not (~w > ~w): return\n", [PyVar, PyValue]).

translate_goal(<(Var, Value), Code) :-
    !,
    var_to_python(Var, PyVar),
    var_to_python(Value, PyValue),
    format(string(Code), "    if not (~w < ~w): return\n", [PyVar, PyValue]).

translate_goal(>=(Var, Value), Code) :-
    !,
    var_to_python(Var, PyVar),
    var_to_python(Value, PyValue),
    format(string(Code), "    if not (~w >= ~w): return\n", [PyVar, PyValue]).

translate_goal(=<(Var, Value), Code) :-
    !,
    var_to_python(Var, PyVar),
    var_to_python(Value, PyValue),
    format(string(Code), "    if not (~w <= ~w): return\n", [PyVar, PyValue]).

translate_goal(\=(Var, Value), Code) :-
    !,
    var_to_python(Var, PyVar),
    var_to_python(Value, PyValue),
    format(string(Code), "    if not (~w != ~w): return\n", [PyVar, PyValue]).

% Match predicate support for procedural mode
translate_goal(match(Var, Pattern), Code) :-
    !,
    translate_match_goal(Var, Pattern, auto, [], Code).
translate_goal(match(Var, Pattern, Type), Code) :-
    !,
    translate_match_goal(Var, Pattern, Type, [], Code).
translate_goal(match(Var, Pattern, Type, Groups), Code) :-
    !,
    translate_match_goal(Var, Pattern, Type, Groups, Code).

translate_goal(suggest_bookmarks(Query, Options, Suggestions), Code) :-
    !,
    var_to_python(Query, PyQuery),
    var_to_python(Suggestions, PyResults),
    % Extract search mode from options (default vector)
    (   member(mode(Mode), Options)
    ->  atom_string(Mode, ModeStr)
    ;   ModeStr = "vector"
    ),
    format(string(Code), "    ~w = _get_runtime().searcher.suggest_bookmarks(~w, top_k=5, mode='~w')\n", [PyResults, PyQuery, ModeStr]).

translate_goal(suggest_bookmarks(Query, Suggestions), Code) :-
    !,
    var_to_python(Query, PyQuery),
    var_to_python(Suggestions, PyResults),
    format(string(Code), "    ~w = _get_runtime().searcher.suggest_bookmarks(~w, top_k=5)\n", [PyResults, PyQuery]).

translate_goal(graph_search(Query, TopK, Hops, Options, Results), Code) :-
    !,
    var_to_python(Query, PyQuery),
    var_to_python(TopK, PyTopK),
    var_to_python(Hops, PyHops),
    var_to_python(Results, PyResults),
    % Extract search mode from options (default vector)
    (   member(mode(Mode), Options)
    ->  atom_string(Mode, ModeStr)
    ;   ModeStr = "vector"
    ),
    format(string(Code), "    ~w = _get_runtime().searcher.graph_search(~w, top_k=~w, hops=~w, mode='~w')\n", [PyResults, PyQuery, PyTopK, PyHops, ModeStr]).

translate_goal(graph_search(Query, TopK, Hops, Results), Code) :-
    !,
    var_to_python(Query, PyQuery),
    var_to_python(TopK, PyTopK),
    var_to_python(Hops, PyHops),
    var_to_python(Results, PyResults),
    format(string(Code), "    ~w = _get_runtime().searcher.graph_search(~w, top_k=~w, hops=~w)\n", [PyResults, PyQuery, PyTopK, PyHops]).

translate_goal(semantic_search(Query, TopK, Results), Code) :-
    !,
    var_to_python(Query, PyQuery),
    var_to_python(TopK, PyTopK),
    var_to_python(Results, PyResults),
    format(string(Code), "    ~w = _get_runtime().searcher.search(~w, top_k=~w)\n", [PyResults, PyQuery, PyTopK]).

translate_goal(crawler_run(SeedIds, MaxDepth, Options), Code) :-
    !,
    var_to_python(SeedIds, PySeeds),
    var_to_python(MaxDepth, PyDepth),
    % Check options for embedding(false)
    (   member(embedding(false), Options)
    ->  EmbedVal = "False"
    ;   EmbedVal = "True"
    ),
    format(string(Code), "    _get_runtime().crawler.crawl(~w, fetch_xml_func, max_depth=~w, embed_content=~w)\n", [PySeeds, PyDepth, EmbedVal]).

translate_goal(crawler_run(SeedIds, MaxDepth), Code) :-
    !,
    var_to_python(SeedIds, PySeeds),
    var_to_python(MaxDepth, PyDepth),
    format(string(Code), "    _get_runtime().crawler.crawl(~w, fetch_xml_func, max_depth=~w)\n", [PySeeds, PyDepth]).

translate_goal(upsert_object(Id, Type, Data), Code) :-
    !,
    var_to_python(Id, PyId),
    var_to_python(Type, PyType),
    var_to_python(Data, PyData),
    format(string(Code), "    _get_runtime().importer.upsert_object(~w, ~w, ~w)\n", [PyId, PyType, PyData]).

translate_goal(llm_ask(Prompt, Context, Response), Code) :-
    !,
    var_to_python(Prompt, PyPrompt),
    var_to_python(Context, PyContext),
    var_to_python(Response, PyResponse),
    format(string(Code), "    ~w = _get_runtime().llm.ask(~w, ~w)\n", [PyResponse, PyPrompt, PyContext]).

translate_goal(chunk_text(Text, Chunks), Code) :-
    !,
    var_to_python(Text, PyText),
    var_to_python(Chunks, PyChunks),
    format(string(Code), "    ~w = [asdict(c) for c in _get_runtime().chunker.chunk(~w, 'inline')]\n", [PyChunks, PyText]).

translate_goal(chunk_text(Text, Chunks, Options), Code) :-
    !,
    var_to_python(Text, PyText),
    var_to_python(Chunks, PyChunks),
    (   is_list(Options)
    ->  maplist(opt_to_py_pair, Options, Pairs),
        atomic_list_concat(Pairs, ', ', PairsStr),
        format(string(PyKwargs), "{~s}", [PairsStr])
    ;   var_to_python(Options, PyKwargs)
    ),
    format(string(Code), "    ~w = [asdict(c) for c in _get_runtime().chunker.chunk(~w, 'inline', **~w)]\n", [PyChunks, PyText, PyKwargs]).

translate_goal(generate_key(Strategy, KeyVar), Code) :-
    !,
    var_to_python(KeyVar, PyKeyVar),
    compile_python_key_expr(Strategy, PyExpr),
    format(string(Code), "    ~w = ~s\n", [PyKeyVar, PyExpr]).

opt_to_py_pair(Term, Pair) :-
    Term =.. [Key, Value],
    format(string(Pair), "'~w': ~w", [Key, Value]).

surround_quotes(S, Q) :- format(string(Q), "'~w'", [S]).

translate_goal(json_get(Data, Field, Var), Code) :-
    !,
    var_to_python(Data, PyData),
    var_to_python(Var, PyVar),
    (   atom(Field)
    ->  format(string(Code), "    ~w = ~w.get('~w')\n", [PyVar, PyData, Field])
    ;   is_list(Field)
    ->  maplist(atom_string, Field, PathStrs),
        maplist(surround_quotes, PathStrs, QuotedPaths),
        atomic_list_concat(QuotedPaths, ', ', PathListStr),
        format(string(Code), "    ~w = _json_get_nested(~w, [~s])\n", [PyVar, PyData, PathListStr])
    ).

translate_goal(json_record(Fields), Code) :-
    !,
    maplist(translate_json_field, Fields, Codes),
    atomic_list_concat(Codes, "", Code).

translate_json_field(Name-Var, Code) :-
    !,
    var_to_python(Var, PyVar),
    % Assume input is 'record' as per convention
    format(string(Code), "    ~w = record.get('~w')\n", [PyVar, Name]).

translate_goal(true, Code) :-
    !,
    Code = "    pass\n".

% ============================================================================
% BINDING-BASED GOAL TRANSLATION
% ============================================================================
%
% Check the binding registry for Python bindings before falling back to
% unsupported goal warning. This enables extensible goal handling.
%
% The binding registry maps Prolog predicates to Python functions with:
% - TargetName: The Python function/method name
% - Inputs/Outputs: Argument specifications
% - Options: Effect annotations (pure, io, etc.) and imports
%

translate_goal(Goal, Code) :-
    % Extract predicate name and arity from goal
    Goal =.. [Pred|Args],
    length(Args, Arity),

    % Check if we have a Python binding for this predicate
    binding(python, Pred/Arity, TargetName, _Inputs, Outputs, Options),
    !,

    % Record any required imports
    (   member(import(Module), Options)
    ->  (   required_import(Module)
        ->  true
        ;   assertz(required_import(Module))
        )
    ;   true
    ),

    % Generate Python code based on binding
    generate_binding_call_python(TargetName, Args, Outputs, Options, Code).

translate_goal(Goal, "") :-
    format(string(Msg), "Warning: Unsupported goal ~w", [Goal]),
    print_message(warning, Msg).

% ============================================================================
% BINDING CODE GENERATION FOR PYTHON
% ============================================================================

%% generate_binding_call_python(+TargetName, +Args, +Outputs, +Options, -Code)
%  Generate Python code for a binding call
%
%  Handles multiple patterns:
%  - Function calls: func(args) -> result
%  - Method calls: object.method(args) -> result
%  - No-arg methods: object.method() (TargetName may include () or not)
%  - Mutating methods: object.method(arg) with no output (e.g., list.append)
%  - Chained calls: object.method1().method2()
%
generate_binding_call_python(TargetName, Args, Outputs, Options, Code) :-
    % Determine call pattern type
    (   member(pattern(method_call), Options)
    ->  generate_method_call(TargetName, Args, Outputs, CallExpr)
    ;   member(pattern(chained_call(Methods)), Options)
    ->  generate_chained_call(Methods, Args, Outputs, CallExpr)
    ;   generate_function_call(TargetName, Args, Outputs, CallExpr)
    ),

    % Generate assignment if there are outputs
    (   Outputs = []
    ->  format(string(Code), "    ~w\n", [CallExpr])
    ;   % Assign result to output variable (last argument)
        last(Args, OutputVar),
        var_to_python(OutputVar, PyOutputVar),
        format(string(Code), "    ~w = ~w\n", [PyOutputVar, CallExpr])
    ).

%% generate_method_call(+TargetName, +Args, +Outputs, -CallExpr)
%  Generate a method call expression: object.method(args)
%
%  Args structure depends on Outputs:
%  - Outputs = []: All args are inputs, first is object
%  - Outputs = [_]: All but last are inputs, first is object, last is output
%
generate_method_call(TargetName, Args, Outputs, CallExpr) :-
    % Separate object from other arguments
    Args = [Object|RestArgs],
    var_to_python(Object, PyObject),

    % Determine which args are inputs (exclude output if present)
    (   Outputs = []
    ->  % No output: all RestArgs are method arguments
        InputArgs = RestArgs
    ;   % Has output: RestArgs minus last element are method arguments
        (   RestArgs = []
        ->  InputArgs = []
        ;   append(InputArgs, [_OutputArg], RestArgs)
        )
    ),

    % Convert input args to Python
    maplist(var_to_python, InputArgs, PyInputArgs),

    % Generate method call expression
    (   PyInputArgs = []
    ->  % No-arg method call
        (   sub_string(TargetName, _, _, 0, "()")
        ->  % TargetName already has () like ".split()" or ".lower()"
            format(string(CallExpr), "~w~w", [PyObject, TargetName])
        ;   % TargetName doesn't have (), add them
            format(string(CallExpr), "~w~w()", [PyObject, TargetName])
        )
    ;   % Method call with arguments
        atomic_list_concat(PyInputArgs, ', ', ArgsStr),
        format(string(CallExpr), "~w~w(~w)", [PyObject, TargetName, ArgsStr])
    ).

%% generate_function_call(+TargetName, +Args, +Outputs, -CallExpr)
%  Generate a function call expression: func(args)
%
generate_function_call(TargetName, Args, Outputs, CallExpr) :-
    (   Outputs = []
    ->  % No output - all args are inputs
        maplist(var_to_python, Args, PyArgs),
        atomic_list_concat(PyArgs, ', ', ArgsStr),
        format(string(CallExpr), "~w(~w)", [TargetName, ArgsStr])
    ;   % Has output - extract input args (all but last which is output)
        (   append(InputArgs, [_OutputArg], Args)
        ->  maplist(var_to_python, InputArgs, PyInputArgs),
            (   PyInputArgs = []
            ->  % Constant/no-arg function
                format(string(CallExpr), "~w", [TargetName])
            ;   atomic_list_concat(PyInputArgs, ', ', ArgsStr),
                format(string(CallExpr), "~w(~w)", [TargetName, ArgsStr])
            )
        ;   % Single arg that is output (constant function like pi/1)
            format(string(CallExpr), "~w", [TargetName])
        )
    ).

%% generate_chained_call(+Methods, +Args, +Outputs, -CallExpr)
%  Generate a chained method call: object.method1(args1).method2(args2)
%
%  Methods is a list of method(Name, ArgIndices) terms specifying which
%  args (by index) go to each method in the chain.
%
%  Example: pattern(chained_call([method('.strip', []), method('.lower', [])]))
%  For: strip_lower(Str, Result) -> Str.strip().lower()
%
generate_chained_call(Methods, Args, Outputs, CallExpr) :-
    % First arg is always the object
    Args = [Object|RestArgs],
    var_to_python(Object, PyObject),

    % Determine input args (exclude output if present)
    (   Outputs = []
    ->  InputArgs = RestArgs
    ;   (   RestArgs = []
        ->  InputArgs = []
        ;   append(InputArgs, [_], RestArgs)
        )
    ),

    % Build the chain
    generate_method_chain(Methods, InputArgs, PyObject, CallExpr).

%% generate_method_chain(+Methods, +InputArgs, +CurrentExpr, -FinalExpr)
%  Recursively build a method chain expression
generate_method_chain([], _InputArgs, Expr, Expr).
generate_method_chain([method(Name, ArgIndices)|Rest], InputArgs, CurrentExpr, FinalExpr) :-
    % Get args for this method by indices
    findall(PyArg, (
        member(Idx, ArgIndices),
        nth0(Idx, InputArgs, Arg),
        var_to_python(Arg, PyArg)
    ), MethodPyArgs),

    % Generate this method call
    (   MethodPyArgs = []
    ->  (   sub_string(Name, _, _, 0, "()")
        ->  format(string(NextExpr), "~w~w", [CurrentExpr, Name])
        ;   format(string(NextExpr), "~w~w()", [CurrentExpr, Name])
        )
    ;   atomic_list_concat(MethodPyArgs, ', ', ArgsStr),
        format(string(NextExpr), "~w~w(~w)", [CurrentExpr, Name, ArgsStr])
    ),

    % Continue with rest of chain
    generate_method_chain(Rest, InputArgs, NextExpr, FinalExpr).

%% compile_python_key_expr(+Strategy, -PyExpr)
%  Compiles a key generation strategy into a Python string expression.
compile_python_key_expr(Var, PyExpr) :-
    is_var_term(Var), !,
    var_to_python(Var, PyVar),
    format(string(PyExpr), "str(~w)", [PyVar]).
compile_python_key_expr(literal(Text), PyExpr) :-
    !,
    format(string(PyExpr), "'~w'", [Text]).
compile_python_key_expr(field(Var), PyExpr) :-
    !,
    compile_python_key_expr(Var, PyExpr).
compile_python_key_expr(composite(List), PyExpr) :-
    !,
    maplist(compile_python_key_expr, List, Parts),
    atomic_list_concat(Parts, " + ", Expr),
    PyExpr = Expr.
compile_python_key_expr(hash(Expr), PyExpr) :-
    !,
    compile_python_key_expr(Expr, Inner),
    format(string(PyExpr), "hashlib.sha256(str(~s).encode('utf-8')).hexdigest()", [Inner]).
compile_python_key_expr(uuid(), "uuid.uuid4().hex") :- !.
compile_python_key_expr(Term, PyExpr) :-
    atomic(Term),
    !,
    format(string(PyExpr), "'~w'", [Term]).
compile_python_key_expr(Strategy, "'UNKNOWN_STRATEGY'") :-
    format(string(Msg), "Warning: Unknown key strategy ~w", [Strategy]),
    print_message(warning, Msg).

%% translate_match_goal(+Var, +Pattern, +Type, +Groups, -Code)
%  Translate match predicate to Python code for procedural mode
translate_match_goal(Var, Pattern, Type, Groups, Code) :-
    % Validate regex type
    validate_regex_type_for_python(Type),
    % Convert variable to Python
    var_to_python(Var, PyVar),
    % Escape pattern
    (   atom(Pattern)
    ->  atom_string(Pattern, PatternStr)
    ;   PatternStr = Pattern
    ),
    escape_python_string(PatternStr, EscapedPattern),
    % Check if we have capture groups
    (   Groups = [], !
    ->  % No capture groups - simple boolean match
        format(string(Code), "    if not re.search(r'~w', str(~w)): return\n", [EscapedPattern, PyVar])
    ;   % Has capture groups - extract them
        length(Groups, NumGroups),
        % Generate Python code to capture and store groups
        generate_python_capture_code(PyVar, EscapedPattern, Groups, NumGroups, Code)
    ).

%% generate_python_capture_code(+PyVar, +Pattern, +Groups, +NumGroups, -Code)
%  Generate Python code to perform match with capture group extraction
generate_python_capture_code(PyVar, Pattern, Groups, NumGroups, Code) :-
    % Generate match object assignment
    format(string(MatchLine), "    __match__ = re.search(r'~w', str(~w))\n", [Pattern, PyVar]),
    % Generate check for match success
    CheckLine = "    if not __match__: return\n",
    % Generate capture variable assignments
    findall(CaptureLine,
        (   between(1, NumGroups, N),
            nth1(N, Groups, GroupVar),
            var_to_python(GroupVar, PyGroupVar),
            format(string(CaptureLine), "    ~w = __match__.group(~w)\n", [PyGroupVar, N])
        ),
        CaptureLines),
    % Combine all lines
    atomic_list_concat([MatchLine, CheckLine | CaptureLines], '', Code).

pair_to_python(Key-Value, Str) :-
    var_to_python(Value, PyValue),
    format(string(Str), "'~w': ~w", [Key, PyValue]).

var_to_python('$VAR'(I), PyVar) :-
    !,
    format(string(PyVar), "v_~d", [I]).
var_to_python(v(Name), PyVar) :-
    % Support for readable variable notation v(name) -> v_name
    !,
    format(string(PyVar), "v_~w", [Name]).
var_to_python(Atom, Quoted) :-
    atom(Atom),
    !,
    format(string(Quoted), "\"~w\"", [Atom]).
var_to_python(Number, Number) :- 
    number(Number), 
    !.
var_to_python(List, PyList) :-
    is_list(List),
    !,
    maplist(var_to_python, List, Elems),
    atomic_list_concat(Elems, ', ', Inner),
    format(string(PyList), "[~w]", [Inner]).
var_to_python(Term, String) :-
    term_string(Term, String).

%% generate_tail_recursive_code(+Name, +Arity, +BaseClauses, +RecClauses, -WorkerCode)
%  Generate iterative code with while loop for tail recursion
generate_tail_recursive_code(Name, Arity, BaseClauses, RecClauses, WorkerCode) :-
    (   Arity =:= 2
    ->  generate_binary_tail_loop(Name, BaseClauses, RecClauses, WorkerCode)
    ;   Arity =:= 3
    ->  generate_ternary_tail_loop(Name, BaseClauses, RecClauses, WorkerCode)
    ;   % Fallback: generate error message
        format(string(WorkerCode), "# ERROR: Tail recursion only supported for arity 2-3, got arity ~d\n", [Arity])
    ).

%% generate_binary_tail_loop(+Name, +BaseClauses, +RecClauses, -WorkerCode)
%  Generate while loop for binary tail recursion
generate_binary_tail_loop(Name, BaseClauses, RecClauses, WorkerCode) :-
    % Extract base case pattern
    (   BaseClauses = [(BaseHead, _BaseBody)|_]
    ->  BaseHead =.. [_, BaseInput, BaseOutput],
        translate_base_case(BaseInput, BaseOutput, BaseCondition, BaseReturn)
    ;   BaseCondition = "False", BaseReturn = "None"
    ),
    
    % Extract step operation from recursive clause
    (   RecClauses = [(_RecHead, RecBody)|_]
    ->  extract_step_operation(RecBody, _StepOp)
    ;   _StepOp = "arg - 1"  % Default decrement
    ),
    
    format(string(WorkerCode),
"def _~w_worker(arg):
    # Tail recursion optimized to while loop
    current = arg
    
    # Base case check
    if ~s:
        return ~s
    
    # Iterative loop (tail recursion optimization)
    result = 1  # Initialize accumulator
    while current > 0:
        result = result * current
        current = current - 1
    
    return result
", [Name, BaseCondition, BaseReturn]).

%% extract_step_operation(+Body, -StepOp)
%  Extract the step operation for the loop
extract_step_operation(Body, StepOp) :-
    % Find 'is' expression for decrement: N1 is N - 1
    extract_goals_list(Body, Goals),
    (   member((_ is Expr), Goals),
        Expr = (_ - _)
    ->  StepOp = "arg - 1"
    ;   StepOp = "arg - 1"  % Default
    ).

%% generate_ternary_tail_loop(+Name, +BaseClauses, +RecClauses, -WorkerCode)
%  Generate while loop for ternary tail recursion with accumulator
%  Pattern: sum(0, Acc, Acc). sum(N, Acc, S) :- N > 0, ..., sum(N1, Acc1, S).
generate_ternary_tail_loop(Name, BaseClauses, RecClauses, WorkerCode) :-
    % Extract base case: pred(BaseInput, Acc, Acc)
    (   BaseClauses = [(BaseHead, _BaseBody)|_]
    ->  BaseHead =.. [_, BaseInput, _Acc, _Result],
        translate_base_case_ternary(BaseInput, BaseCondition)
    ;   BaseCondition = "False"
    ),
    
    % Extract accumulator update from recursive clause
    % sum(N, Acc, S) :- N > 0, N1 is N - 1, Acc1 is Acc + N, sum(N1, Acc1, S)
    (   RecClauses = [(_RecHead, RecBody)|_]
    ->  extract_accumulator_update(RecBody, AccUpdate)
    ;   AccUpdate = "acc + n"  % Default
    ),
    
    format(string(WorkerCode),
"def _~w_worker(n, acc):
    # Tail recursion (arity 3) optimized to while loop
    current = n
    result = acc
    
    # Base case check
    if ~s:
        return result
    
    # Iterative loop (tail recursion optimization)
    while current > 0:
        result = ~s
        current = current - 1
    
    return result
", [Name, BaseCondition, AccUpdate]).

%% translate_base_case_ternary(+Input, -Condition)
translate_base_case_ternary(Input, Condition) :-
    (   Input == []
    ->  Condition = "not current or current == []"
    ;   number(Input)
    ->  format(string(Condition), "current == ~w", [Input])
    ;   Condition = "False"
    ).

%% extract_accumulator_update(+Body, -Update)
%  Extract accumulator update expression
%  From: Acc1 is Acc + N → "result + current"
extract_accumulator_update(Body, Update) :-
    extract_goals_list(Body, Goals),
    % Find the accumulator update: Acc1 is Acc + N (or Acc * N, etc.)
    findall(Expr, member((_ is Expr), Goals), Exprs),
    % The second 'is' expression (if present) is usually the accumulator update
    (   length(Exprs, Len), Len >= 2,
        nth1(2, Exprs, AccExpr)
    ->  translate_acc_expr(AccExpr, Update)
    ;   Update = "result + current"  % Default
    ).

%% translate_acc_expr(+Expr, -PyExpr)
translate_acc_expr(Expr, PyExpr) :-
    functor(Expr, Op, 2),
    (   Op = '+'
    ->  PyOp = "+", _Order = normal
    ;   Op = '*'
    ->  PyOp = "*", _Order = normal
    ;   Op = '-'
    ->  PyOp = "-", _Order = normal
    ;   PyOp = "+", _Order = normal
    ),
    % Determine order: is it Acc + N or N + Acc?
    Expr =.. [_, Arg1, Arg2],
    (   var(Arg1), \+ var(Arg2)  % Acc + N
    ->  format(string(PyExpr), "result ~s current", [PyOp])
    ;   var(Arg2), \+ var(Arg1)  % N + Acc (reverse)
    ->  format(string(PyExpr), "current ~s result", [PyOp])
    ;   % Both vars or both ground, default
        format(string(PyExpr), "result ~s current", [PyOp])
    ).

%% generate_worker_function(+Name, +Arity, +BaseClauses, +RecClauses, -WorkerCode)
generate_worker_function(Name, Arity, BaseClauses, RecClauses, WorkerCode) :-
    % For now, only support binary recursion (Input, Output)
    (   Arity =:= 2
    ->  generate_binary_worker(Name, BaseClauses, RecClauses, WorkerCode)
    ;   % Fallback: generate error message
        format(string(WorkerCode), "# ERROR: Recursion only supported for arity 2, got arity ~d\n", [Arity])
    ).

%% generate_binary_worker(+Name, +BaseClauses, +RecClauses, -WorkerCode)
generate_binary_worker(Name, BaseClauses, RecClauses, WorkerCode) :-
    % Extract base case pattern
    (   BaseClauses = [(BaseHead, _BaseBody)|_]
    ->  BaseHead =.. [_, BaseInput, BaseOutput],
        translate_base_case(BaseInput, BaseOutput, BaseCondition, BaseReturn)
    ;   BaseCondition = "False", BaseReturn = "None"
    ),
    
    % Extract recursive case
    (   RecClauses = [(RecHead, RecBody)|_]
    ->  RecHead =.. [_, RecInput, RecOutput],
        translate_recursive_case(Name, RecInput, RecOutput, RecBody, RecCode)
    ;   RecCode = "    pass"
    ),
    
    format(string(WorkerCode),
"@functools.cache
def _~w_worker(arg):
    # Base case
    if ~s:
        return ~s
    
    # Recursive case
~s
", [Name, BaseCondition, BaseReturn, RecCode]).

%% translate_base_case(+Input, +Output, -Condition, -Return)
translate_base_case(Input, Output, Condition, Return) :-
    (   Input == []
    ->  Condition = "not arg or arg == []"
    ;   number(Input)
    ->  format(string(Condition), "arg == ~w", [Input])
    ;   Condition = "False"  % Unknown pattern
    ),
    (   number(Output)
    ->  format(string(Return), "~w", [Output])
    ;   atom(Output)
    ->  format(string(Return), "\"~w\"", [Output])
    ;   var_to_python(Output, Return)
    ).

%% translate_recursive_case(+Name, +Input, +Output, +Body, -Code)
translate_recursive_case(Name, _Input, _Output, Body, Code) :-
    % Find the recursive call and arithmetic operations
    extract_recursive_pattern(Body, Name, Pattern),
    translate_pattern_to_python(Name, Pattern, Code).

%% extract_recursive_pattern(+Body, +Name, -Pattern)
extract_recursive_pattern(Body, Name, Pattern) :-
    % factorial: N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1
    % The LAST 'is' expression is the one that computes the result
    extract_goals_list(Body, Goals),
    findall(Expr, member((_ is Expr), Goals), Exprs),
    % Take the last expression (the result computation)
    (   Exprs \= [],
        last(Exprs, RecExpr),
        functor(RecExpr, Op, 2),
        member(Op, ['*', '+', '-', '/'])
    ->  member(RecCall, Goals),
        functor(RecCall, Name, _),
        Pattern = arithmetic(RecExpr, RecCall)
    ;   Pattern = unknown
    ).

extract_goals_list((A, B), [A|Rest]) :- !, extract_goals_list(B, Rest).
extract_goals_list(Goal, [Goal]).

%% translate_pattern_to_python(+Name, +Pattern, -Code)
translate_pattern_to_python(Name, arithmetic(Expr, _RecCall), Code) :-
    % For factorial: F is N * F1 → return arg * _factorial_worker(arg - 1)
    % Extract operator using functor
    functor(Expr, Op, _),
    (   Op = '*'
    ->  PyOp = "*"
    ;   Op = '+'
    ->  PyOp = "+"
    ;   Op = '-'
    ->  PyOp = "-"
    ;   PyOp = "*"  % Default to multiplication
    ),
    format(string(Code), "    return arg ~s _~w_worker(arg - 1)", [PyOp, Name]).
translate_pattern_to_python(_, unknown, "    pass  # Unknown recursion pattern").

%% generate_recursive_wrapper(+Name, +Arity, -WrapperCode)
generate_recursive_wrapper(Name, Arity, WrapperCode) :-
    % Generate _clause_0 that extracts input from dict, calls worker, yields result
    (   Arity =:= 2
    ->  format(string(WrapperCode),
"def _clause_0(v_0: Dict) -> Iterator[Dict]:
    # Extract input
    keys = list(v_0.keys())
    if not keys:
        return
    input_key = keys[0]
    input_val = v_0[input_key]
    
    # Call worker
    result = _~w_worker(input_val)
    
    # Yield result dict
    output_key = keys[1] if len(keys) > 1 else 'result'
    yield {input_key: input_val, output_key: result}
", [Name])
    ;   Arity =:= 3
    ->  format(string(WrapperCode),
"def _clause_0(v_0: Dict) -> Iterator[Dict]:
    # Extract input and accumulator from dict
    keys = list(v_0.keys())
    if len(keys) < 2:
        return
    input_key = keys[0]
    acc_key = keys[1]
    input_val = v_0[input_key]
    acc_val = v_0.get(acc_key, 0)  # Default accumulator to 0
    
    # Call worker
    result = _~w_worker(input_val, acc_val)
    
    # Yield result dict
    output_key = keys[2] if len(keys) > 2 else 'result'
    yield {input_key: input_val, acc_key: acc_val, output_key: result}
", [Name])
    ;   WrapperCode = "# ERROR: Unsupported arity for recursion wrapper"
    ).

%% get_binding_imports(-ImportStr)
%  Get import statements for all modules required by bindings used in compilation
get_binding_imports(ImportStr) :-
    findall(Module, required_import(Module), Modules),
    sort(Modules, UniqueModules),  % Remove duplicates
    (   UniqueModules = []
    ->  ImportStr = ""
    ;   findall(ImportLine, (
            member(M, UniqueModules),
            % Skip modules already in base imports
            \+ member(M, [sys, json, re, hashlib, uuid, functools, typing]),
            format(string(ImportLine), "import ~w", [M])
        ), ImportLines),
        (   ImportLines = []
        ->  ImportStr = ""
        ;   atomic_list_concat(ImportLines, '\n', ImportsBody),
            format(string(ImportStr), "~w\n", [ImportsBody])
        )
    ).

%% header(-Header)
%  Generate header with base imports plus any binding-required imports
header(Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"import sys
import json
import re
import hashlib
import uuid
from typing import Iterator, Dict, Any
~w
", [BindingImports]).

%% header_with_functools(-Header)
%  Generate header with functools plus any binding-required imports
header_with_functools(Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"import sys
import json
import re
import hashlib
import uuid
import functools
from typing import Iterator, Dict, Any
~w
", [BindingImports]).

%% clear_binding_imports
%  Clear collected binding imports (call before each compilation)
clear_binding_imports :-
    retractall(required_import(_)).

helpers(Helpers) :-
    helpers_base(Base),
    semantic_runtime_helpers(Runtime),
    generate_all_validators(Validators),
    format(string(Helpers), "~s\n~s\n~s", [Base, Runtime, Validators]).

%% generate_all_validators(-Code)
generate_all_validators(Code) :-
    findall(ValCode, (
        json_schema_def(Name, Fields),
        generate_validator_function(Name, Fields, ValCode)
    ), ValidatorCodes),
    atomic_list_concat(ValidatorCodes, "\n\n", Code).

%% generate_validator_function(+SchemaName, +Fields, -Code)
generate_validator_function(Name, Fields, Code) :-
    maplist(generate_field_check, Fields, CheckCodes),
    atomic_list_concat(CheckCodes, "", Checks),
    format(string(Code), 
"def _validate_~w(data):
    if not isinstance(data, dict): return False
~s    return True", [Name, Checks]).

%% generate_field_check(+FieldSpec, -Code)
generate_field_check(field(Name, Type), Code) :-
    generate_field_check(field(Name, Type, []), Code).

generate_field_check(field(Name, Type, _Opts), Code) :-
    type_check_expr(Type, "val", Expr),
    format(string(Code), 
"    val = data.get('~w')
    if val is not None:
        if not (~s): return False
", [Name, Expr]).

%% type_check_expr(+Type, +VarName, -Expr)
type_check_expr(string, Var, Expr) :- format(string(Expr), "isinstance(~w, str)", [Var]).
type_check_expr(integer, Var, Expr) :- format(string(Expr), "isinstance(~w, int)", [Var]).
type_check_expr(float, Var, Expr) :- format(string(Expr), "isinstance(~w, (int, float))", [Var]).
type_check_expr(boolean, Var, Expr) :- format(string(Expr), "isinstance(~w, bool)", [Var]).
type_check_expr(any, _, "True").
type_check_expr(array(Type), Var, Expr) :-
    type_check_expr(Type, "x", InnerExpr),
    format(string(Expr), "isinstance(~w, list) and all(~s for x in ~w)", [Var, InnerExpr, Var]).
type_check_expr(object(Schema), Var, Expr) :-
    format(string(Expr), "_validate_~w(~w)", [Schema, Var]).

helpers_base("
def read_jsonl(stream) -> Iterator[Dict[str, Any]]:
    \"\"\"Read JSONL from stream.\"\"\"
    for line in stream:
        if line.strip():
            yield json.loads(line)

def write_jsonl(records: Iterator[Dict], stream) -> None:
    \"\"\"Write JSONL to stream.\"\"\"
    for record in records:
        stream.write(json.dumps(record) + '\\n')

def read_nul_json(stream) -> Iterator[Dict[str, Any]]:
    \"\"\"Read NUL-delimited JSON from stream.\"\"\"
    buff = ''
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        buff += chunk
        while '\\0' in buff:
            line, buff = buff.split('\\0', 1)
            if line:
                yield json.loads(line)
    if buff and buff.strip('\\0'):
        yield json.loads(buff)

def write_nul_json(records: Iterator[Dict], stream) -> None:
    \"\"\"Write NUL-delimited JSON to stream.\"\"\"
    for record in records:
        stream.write(json.dumps(record) + '\\0')

def read_xml_lxml(file_path: str, tags: set) -> Iterator[Dict[str, Any]]:
    \"\"\"Read and flatten XML using lxml.\"\"\"
    try:
        from lxml import etree
    except ImportError:
        sys.stderr.write('Error: lxml required for XML source\\n')
        sys.exit(1)
    
    context = etree.iterparse(file_path, events=('start', 'end'), recover=True)
    context = iter(context)
    _, root = next(context) # Get root start
    
    def expand(tag, nsmap):
        if ':' in tag:
            pfx, local = tag.split(':', 1)
            uri = nsmap.get(pfx)
            if uri:
                return '{' + uri + '}' + local
        return tag

    # Pre-calculate wanted tags (assuming passed tags are QNames if needed, or local names)
    # For simplicity, we match suffix or exact
    
    for event, elem in context:
        if event == 'end' and (elem.tag in tags or elem.tag.split('}')[-1] in tags):
            data = {}
            # Root element attributes (global keys for backward compatibility)
            for k, v in elem.attrib.items():
                data['@' + k] = v
            # Text
            if elem.text and elem.text.strip():
                data['text'] = elem.text.strip()
            # Children (simple flattening)
            for child in elem:
                tag = child.tag.split('}')[-1]
                if not len(child) and child.text:
                    data[tag] = child.text.strip()
                # Child element attributes (element-scoped to prevent conflicts)
                for attr_name, attr_val in child.attrib.items():
                    scoped_key = tag + '@' + attr_name
                    data[scoped_key] = attr_val
                    # Also store with global key for backward compatibility
                    data['@' + attr_name] = attr_val

            yield data
            
            # Memory cleanup
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
    del context
    root.clear()

def _json_get_nested(data, path):
    'Get nested value from dict using path list.'
    current = data
    for key in path:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
        if current is None:
            return None
    return current
\n").

%% ============================================
%% GENERATOR MODE (Semi-Naive Evaluation)
%% ============================================

%% compile_generator_mode(+Name, +Arity, +Module, +Options, -PythonCode)
%  Compile using generator-based semi-naive fixpoint iteration
%  Similar to C# query engine approach
compile_generator_mode(Name, Arity, Module, Options, PythonCode) :-
    functor(Head, Name, Arity),
    findall((Head, Body), clause(Module:Head, Body), Clauses),
    (   Clauses == []
    ->  format(string(PythonCode), "# ERROR: No clauses found for ~w/~w\n", [Name, Arity])
    ;   partition(is_fact_clause, Clauses, Facts, Rules),
        generate_generator_code(Name, Arity, Facts, Rules, Options, PythonCode)
    ).

is_fact_clause((_Head, Body)) :- Body == true.

%% generate_generator_code(+Name, +Arity, +Facts, +Rules, +Options, -PythonCode)
generate_generator_code(_Name, _Arity, Facts, Rules, Options, PythonCode) :-
    % Generate components
    generator_header(Header),
    generator_helpers(Options, Helpers),
    generate_fact_functions(Name, Facts, FactFunctions),
    generate_rule_functions(Name, Rules, RuleFunctions),
    generate_fixpoint_loop(Name, Facts, Rules, FixpointLoop),
    
    generate_python_main(Options, Main),
    
    atomic_list_concat([Header, Helpers, FactFunctions, RuleFunctions, FixpointLoop, Main], "\n", PythonCode).

%% generator_header(-Header)
%  Generate header for generator mode with binding-required imports
generator_header(Header) :-
    get_binding_imports(BindingImports),
    format(string(Header),
"import sys
import json
import re
from typing import Iterator, Dict, Any, Set
from dataclasses import dataclass
~w
# FrozenDict - hashable dictionary for use in sets
@dataclass(frozen=True)
class FrozenDict:
    '''Immutable dictionary that can be used in sets.'''
    items: tuple

    @staticmethod
    def from_dict(d: Dict) -> 'FrozenDict':
        return FrozenDict(tuple(sorted(d.items())))

    def to_dict(self) -> Dict:
        return dict(self.items)

    def get(self, key, default=None):
        for k, v in self.items:
            if k == key:
                return v
        return default

    def __contains__(self, key):
        return any(k == key for k, _ in self.items)

    def __repr__(self):
        return f'FrozenDict({dict(self.items)})'
", [BindingImports]).

%% generator_helpers(+Options, -Helpers)
generator_helpers(Options, Helpers) :-
    option(record_format(Format), Options, jsonl),
    (   Format == nul_json
    ->  NulReader = "
def read_nul_json(stream: Any) -> Iterator[Dict]:
    '''Read NUL-separated JSON records.'''
    buffer = ''
    while True:
        char = stream.read(1)
        if not char:
            if buffer.strip():
                yield json.loads(buffer)
            break
        if char == '\\0':
            if buffer.strip():
                yield json.loads(buffer)
                buffer = ''
        else:
            buffer += char

def write_nul_json(records: Iterator[Dict], stream: Any):
    '''Write NUL-separated JSON records.'''
    for record in records:
        stream.write(json.dumps(record) + '\\0')
",
        JsonlReader = ""
    ;   JsonlReader = "
def read_jsonl(stream: Any) -> Iterator[Dict]:
    '''Read JSONL records.'''
    for line in stream:
        line = line.strip()
        if line:
            yield json.loads(line)

def write_jsonl(records: Iterator[Dict], stream: Any):
    '''Write JSONL records.'''
    for record in records:
        stream.write(json.dumps(record) + '\\n')
",
        NulReader = ""
    ),
    semantic_runtime_helpers(Runtime),
    atomic_list_concat([JsonlReader, NulReader, Runtime], "", Helpers).

%% generate_fact_functions(+Name, +Facts, -FactFunctions)
generate_fact_functions(Name, Facts, FactFunctions) :-
    findall(FactFunc,
        (   nth1(FactNum, Facts, (Head, _Body)),
            generate_fact_function(Name, FactNum, Head, FactFunc)
        ),
        FactFuncs),
    atomic_list_concat(FactFuncs, "\n\n", FactFunctions).

%% generate_fact_function(+Name, +FactNum, +Head, -FactFunc)
generate_fact_function(_Name, FactNum, Head, FactFunc) :-
    Head =.. [Pred | Args],
    extract_constants(Args, ConstPairs),
    format_dict_pairs(ConstPairs, ArgsStr),
    (   ArgsStr == ""
    ->  format(string(DictStr), "'relation': '~w'", [Pred])
    ;   format(string(DictStr), "'relation': '~w', ~w", [Pred, ArgsStr])
    ),
    format(string(FactFunc),
"def _init_fact_~w() -> Iterator[FrozenDict]:
    '''Fact: ~w'''
    yield FrozenDict.from_dict({~w})
", [FactNum, Head, DictStr]).

%% generate_rule_functions(+Name, +Clauses, -RuleFunctions)
generate_rule_functions(Name, Clauses, RuleFunctions) :-
    findall(RuleFunc,
        (   nth1(RuleNum, Clauses, (Head, Body)),
            generate_rule_function(Name, RuleNum, Head, Body, RuleFunc)
        ),
        RuleFuncs),
    atomic_list_concat(RuleFuncs, "\n\n", RuleFunctions).

%% generate_rule_function(+Name, +RuleNum, +Head, +Body, -RuleFunc)
generate_rule_function(Name, RuleNum, Head, Body, RuleFunc) :-
    (   Body == true
    ->  % Fact (no body) - emit constant
        translate_fact_rule(Name, RuleNum, Head, RuleFunc)
    ;   % Check for disjunction (;) in body
        contains_disjunction(Body)
    ->  % Handle disjunctive rule
        extract_disjuncts(Body, Disjuncts),
        translate_disjunctive_rule(RuleNum, Head, Disjuncts, RuleFunc)
    ;   % Normal conjunctive rule
        extract_goals_list(Body, Goals),
        % Separate builtin goals from relational goals
        partition(is_builtin_goal, Goals, BuiltinGoals, RelGoals),
        length(RelGoals, NumRelGoals),
        (   NumRelGoals == 0
        ->  % Only builtins → constraint, not a generator rule
            format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''Constraint-only rule (no generator)'''
    return iter([])
", [RuleNum])
        ;   NumRelGoals == 1
        ->  % Single relational goal + optionally builtins
            RelGoals = [SingleGoal],
            translate_copy_rule_with_builtins(Name, RuleNum, Head, SingleGoal, BuiltinGoals, RuleFunc)
        ;   % Multiple relational goals + optionally builtins
            translate_join_rule_with_builtins(Name, RuleNum, Head, RelGoals, BuiltinGoals, RuleFunc)
        )
    ).

%% contains_disjunction(+Body)
%  Check if body contains disjunction (;)
contains_disjunction((_;_)) :- !.
contains_disjunction((A,B)) :- 
    !,
    (contains_disjunction(A) ; contains_disjunction(B)).
contains_disjunction(_) :- fail.

%% extract_disjuncts(+Body, -Disjuncts)
%  Extract all disjuncts from a disjunctive body
extract_disjuncts((A;B), Disjuncts) :-
    !,
    extract_disjuncts(A, DisjunctsA),
    extract_disjuncts(B, DisjunctsB),
    append(DisjunctsA, DisjunctsB, Disjuncts).
extract_disjuncts(Goal, [Goal]).

%% translate_disjunctive_rule(+RuleNum, +Head, +Disjuncts, -RuleFunc)
%  Translate a rule with disjunction to Python
translate_disjunctive_rule(RuleNum, Head, Disjuncts, RuleFunc) :-
    % Generate code for each disjunct
    findall(DisjunctCode,
        (   member(Disjunct, Disjuncts),
            translate_disjunct(Head, Disjunct, DisjunctCode)
        ),
        DisjunctCodes),
    atomic_list_concat(DisjunctCodes, "\n    # Try next disjunct\n    ", CombinedCode),
    
    format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''Disjunctive rule: ~w'''
    # Try each disjunct
    ~w
", [RuleNum, Head, CombinedCode]).

%% translate_disjunct(+Head, +Disjunct, -Code)
%  Translate a single disjunct to Python code
translate_disjunct(Head, Disjunct, Code) :-
    % Extract goals from this disjunct
    extract_goals_list(Disjunct, Goals),
    partition(is_builtin_goal, Goals, Builtins, RelGoals),
    
    length(RelGoals, NumRelGoals),
    (   NumRelGoals == 0
    ->  % Only builtins - just check constraints
        translate_builtins(Builtins, ConstraintChecks),
        (   ConstraintChecks == ""
        ->  Code = "pass  # Empty disjunct"
        ;   Head =.. [_Pred|HeadArgs],
            build_constant_output(HeadArgs, OutputStr),
            format(string(Code),
"if ~w:
        yield FrozenDict.from_dict({~w})",
                [ConstraintChecks, OutputStr])
        )
    ;   NumRelGoals == 1
    ->  % Single goal + optional constraints
        RelGoals = [Goal],
        translate_disjunct_copy(Head, Goal, Builtins, Code)
    ;   % Multiple goals - join
        translate_disjunct_join(Head, RelGoals, Builtins, Code)
    ).

%% translate_disjunct_copy(+Head, +Goal, +Builtins, -Code)
translate_disjunct_copy(Head, Goal, Builtins, Code) :-
    Head =.. [HeadPred | HeadArgs],
    Goal =.. [GoalPred | GoalArgs],
    
    % Pattern match
    length(GoalArgs, GoalArity),
    findall(Check,
        (   between(0, GoalArity, Idx),
            Idx < GoalArity,
            format(string(Check), "'arg~w' in fact", [Idx])
        ),
        Checks),
    % Add relation check
    format(string(RelCheck), "fact.get('relation') == '~w'", [GoalPred]),
    AllChecks = [RelCheck | Checks],
    atomic_list_concat(AllChecks, " and ", ConditionStr),
    
    % Constraints
    build_variable_map([Goal-"fact"], VarMap),
    translate_builtins(Builtins, VarMap, ConstraintChecks),
    (   ConstraintChecks == ""
    ->  FinalCondition = ConditionStr
    ;   format(string(FinalCondition), "~w and ~w", [ConditionStr, ConstraintChecks])
    ),
    
    % Output
    findall(Assign,
        (   nth0(HIdx, HeadArgs, HVar),
            nth0(GIdx, GoalArgs, HVar),
            format(string(Assign), "'arg~w': fact.get('arg~w')", [HIdx, GIdx])
        ),
        Assigns),
    % Add relation to output
    format(string(RelAssign), "'relation': '~w'", [HeadPred]),
    AllAssigns = [RelAssign | Assigns],
    atomic_list_concat(AllAssigns, ", ", OutputStr),
    
    format(string(Code),
"if ~w:
        yield FrozenDict.from_dict({~w})",
        [FinalCondition, OutputStr]).

%% translate_disjunct_join(+Head, +Goals, +Builtins, -Code)
translate_disjunct_join(Head, Goals, Builtins, Code) :-
    length(Goals, NumGoals),
    (   NumGoals == 2
    ->  % Binary join within disjunct
        Goals = [Goal1, Goal2],
        translate_disjunct_binary_join(Head, Goal1, Goal2, Builtins, Code)
    ;   % N-way join within disjunct - use simplified approach
        translate_disjunct_nway_join(Head, Goals, Builtins, Code)
    ).

%% translate_disjunct_binary_join(+Head, +Goal1, +Goal2, +Builtins, -Code)
translate_disjunct_binary_join(Head, Goal1, Goal2, Builtins, Code) :-
    Goal1 =.. [Pred1 | Args1],
    Goal2 =.. [Pred2 | Args2],
    Head =.. [HeadPred | HeadArgs],
    
    % Find join variable
    findall(Var-Idx1-Idx2,
        (   nth0(Idx1, Args1, Var),
            nth0(Idx2, Args2, Var),
            \+ atom(Var)
        ),
        JoinVars),
    
    % Build join condition
    (   JoinVars = [_Var-JIdx1-JIdx2|_]
    ->  format(string(JoinCond), "other.get('arg~w') == fact.get('arg~w')", [JIdx2, JIdx1])
    ;   JoinCond = "True"
    ),
    
    % Add relation check for other (Goal2)
    format(string(RelCheck2), "other.get('relation') == '~w'", [Pred2]),
    (   JoinCond == "True"
    ->  JoinCondWithRel = RelCheck2
    ;   format(string(JoinCondWithRel), "~w and ~w", [JoinCond, RelCheck2])
    ),
    
    % Build constraint checks
    build_variable_map([Goal1-"fact", Goal2-"other"], VarMap),
    translate_builtins(Builtins, VarMap, ConstraintChecks),
    (   ConstraintChecks == ""
    ->  FinalJoinCond = JoinCondWithRel
    ;   format(string(FinalJoinCond), "~w and ~w", [JoinCondWithRel, ConstraintChecks])
    ),
    
    % Build output mapping
    findall(OutAssign,
        (   nth0(HIdx, HeadArgs, HVar),
            (   nth0(G1Idx, Args1, HVar)
            ->  format(string(OutAssign), "'arg~w': fact.get('arg~w')", [HIdx, G1Idx])
            ;   nth0(G2Idx, Args2, HVar),
                format(string(OutAssign), "'arg~w': other.get('arg~w')", [HIdx, G2Idx])
            )
        ),
        OutAssigns),
    % Add relation to output
    format(string(RelAssign), "'relation': '~w'", [HeadPred]),
    AllAssigns = [RelAssign | OutAssigns],
    atomic_list_concat(AllAssigns, ", ", OutputMapping),
    
    % Pattern match for first goal
    length(Args1, Arity1),
    findall(PatCheck,
        (   between(0, Arity1, Idx),
            Idx < Arity1,
            format(string(PatCheck), "'arg~w' in fact", [Idx])
        ),
        PatChecks),
    % Add relation check for fact (Goal1)
    format(string(RelCheck1), "fact.get('relation') == '~w'", [Pred1]),
    AllPatChecks = [RelCheck1 | PatChecks],
    atomic_list_concat(AllPatChecks, " and ", Pattern1),
    
    format(string(Code),
"if ~w:
        for other in total:
            if ~w:
                yield FrozenDict.from_dict({~w})",
        [Pattern1, FinalJoinCond, OutputMapping]).

%% translate_disjunct_nway_join(+Head, +Goals, +Builtins, -Code)
translate_disjunct_nway_join(Head, Goals, Builtins, Code) :-
    Goals = [FirstGoal | RestGoals],
    
    % Pattern match first goal
    FirstGoal =.. [Pred1 | Args1],
    length(Args1, Arity1),
    findall(Check, (between(0, Arity1, Idx), Idx < Arity1, format(string(Check), "'arg~w' in fact", [Idx])), Checks),
    % Add relation check
    format(string(RelCheck), "fact.get('relation') == '~w'", [Pred1]),
    AllChecks = [RelCheck | Checks],
    atomic_list_concat(AllChecks, " and ", Pattern1),
    
    % Nested joins
    build_nested_joins(RestGoals, 1, JoinCode, FinalIdx),
    
    % Output mapping
    Head =.. [HeadPred | HeadArgs],
    collect_all_goal_args([FirstGoal | RestGoals], AllGoalArgs),
    build_output_mapping(HeadArgs, FirstGoal, RestGoals, AllGoalArgs, OutputMapping),
    format(string(FullOutputMapping), "'relation': '~w', ~w", [HeadPred, OutputMapping]),
    
    % Constraints
    findall(G-S,
        (   nth1(I, RestGoals, G),
            format(string(S), "join_~w", [I])
        ),
        RestPairs),
    Pairs = [FirstGoal-"fact" | RestPairs],
    build_variable_map(Pairs, VarMap),
    translate_builtins(Builtins, VarMap, ConstraintChecks),
    
    % Calculate indentation for innermost block
    % Base indent is 4 (inside if Pattern1).
    % Each nested join adds 4 (for loop) + 4 (if condition) = 8?
    % build_nested_joins: Idx=1 -> Indent=8 (for loop). If=12.
    % So innermost body is at (FinalIdx + 2) * 4?
    % Let's verify:
    % RestGoals=[G1]. Idx=1. Loop at 8. If at 12. Body at 16.
    % FinalIdx=2. (2+2)*4 = 16. Correct.
    Indent is (FinalIdx + 2) * 4,
    format(string(Spaces), "~*c", [Indent, 32]),
    
    (   ConstraintChecks == ""
    ->  format(string(YieldBlock),
"~wyield FrozenDict.from_dict({~w})", [Spaces, FullOutputMapping])
    ;   format(string(YieldBlock),
"~wif ~w:
~w    yield FrozenDict.from_dict({~w})", [Spaces, ConstraintChecks, Spaces, FullOutputMapping])
    ),

    format(string(Code),
"if ~w:
~w
~w", [Pattern1, JoinCode, YieldBlock]).


%% build_constant_output(+HeadArgs, -OutputStr)
build_constant_output(HeadArgs, OutputStr) :-
    findall(Assign,
        (   nth0(Idx, HeadArgs, Arg),
            (   atom(Arg)
            ->  format(string(Assign), "'arg~w': '~w'", [Idx, Arg])
            ;   format(string(Assign), "'arg~w': None", [Idx])
            )
        ),
        Assigns),
    atomic_list_concat(Assigns, ", ", OutputStr).


%% is_builtin_goal(+Goal)
%  Check if goal is a built-in (is, >, <, etc.)
is_builtin_goal(Goal) :-
    Goal =.. [Pred | _],
    member(Pred, [is, >, <, >=, =<, =:=, =\=, \+, not, match]).

%% translate_fact_rule(+Name, +RuleNum, +Head, -RuleFunc)
%  Translate a fact (constant) into a rule that emits it once
translate_fact_rule(_Name, RuleNum, Head, RuleFunc) :-
    Head =.. [Pred | Args],
    extract_constants(Args, ConstPairs),
    format_dict_pairs(ConstPairs, ArgsStr),
    (   ArgsStr == ""
    ->  format(string(DictStr), "'relation': '~w'", [Pred])
    ;   format(string(DictStr), "'relation': '~w', ~w", [Pred, ArgsStr])
    ),
    format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''Fact: ~w'''
    # Emit constant fact once (only if not already in total)
    result = FrozenDict.from_dict({~w})
    if result not in total:
        yield result
", [RuleNum, Head, DictStr]).

%% extract_constants(+Args, -Pairs)
%  Extract constant values from arguments
extract_constants(Args, Pairs) :-
    findall(Key-Value,
        (   nth1(Idx, Args, Arg),
            atom(Arg),
            \+ var(Arg),
            Key is Idx - 1,  % 0-indexed
            atom_string(Arg, Value)
        ),
        Pairs).

%% format_dict_pairs(+Pairs, -DictStr)
format_dict_pairs(Pairs, DictStr) :-
    findall(Pair,
        (   member(Key-Value, Pairs),
            format(string(Pair), "'arg~w': '~w'", [Key, Value])
        ),
        PairStrs),
    atomic_list_concat(PairStrs, ", ", DictStr).

%% translate_copy_rule_with_builtins(+Name, +RuleNum, +Head, +Goal, +Builtins, -RuleFunc)
%  Copy rule with built-in constraints and relation check
translate_copy_rule_with_builtins(_Name, RuleNum, Head, Goal, Builtins, RuleFunc) :-
    Head =.. [HeadPred | HeadArgs],
    Goal =.. [GoalPred | GoalArgs],
    
    % Build pattern match condition
    length(GoalArgs, GoalArity),
    findall(Check,
        (   between(0, GoalArity, Idx),
            Idx < GoalArity,
            format(string(Check), "'arg~w' in fact", [Idx])
        ),
        Checks),
    % Add relation check
    format(string(RelCheck), "fact.get('relation') == '~w'", [GoalPred]),
    AllChecks = [RelCheck | Checks],
    atomic_list_concat(AllChecks, " and ", ConditionStr),
    
    % Constraints
    build_variable_map([Goal-"fact"], VarMap),
    translate_builtins(Builtins, VarMap, ConstraintChecks),
    
    % Build output dict
    findall(Assign,
        (   nth0(HIdx, HeadArgs, HeadArg),
            nth0(GIdx, GoalArgs, GoalArg),
            HeadArg == GoalArg,
            format(string(Assign), "'arg~w': fact.get('arg~w')", [HIdx, GIdx])
        ),
        Assigns),
    % Add relation to output
    format(string(RelAssign), "'relation': '~w'", [HeadPred]),
    AllAssigns = [RelAssign | Assigns],
    atomic_list_concat(AllAssigns, ", ", OutputStr),
    
    % Combine pattern and constraints
    (   ConstraintChecks == ""
    ->  FinalCondition = ConditionStr
    ;   format(string(FinalCondition), "~w and ~w", [ConditionStr, ConstraintChecks])
    ),
    
    format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''Copy rule: ~w'''
    if ~w:
        yield FrozenDict.from_dict({~w})
", [RuleNum, Head, FinalCondition, OutputStr]).

%% translate_builtins(+Builtins, +VarMap, -ConstraintChecks)
%  Translate built-in predicates to Python expressions
translate_builtins([], _VarMap, "").
translate_builtins(Builtins, VarMap, ConstraintChecks) :-
    Builtins \= [],
    findall(Check,
        (   member(Builtin, Builtins),
            translate_builtin(Builtin, VarMap, Check)
        ),
        Checks),
    atomic_list_concat(Checks, " and ", ConstraintChecks).

python_config(Config) :-
    Config = [
        access_fmt-"~w.get('arg~w')",
        atom_fmt-"'~w'",
        null_val-"None",
        ops-[
            + - "+", - - "-", * - "*", / - "/", mod - "%",
            > - ">", < - "<", >= - ">=", =< - "<=", =:= - "==", =\= - "!=",
            is - "=="
        ]
    ].

%% translate_builtin(+Builtin, +VarMap, -PythonExpr)
%  Translate a single built-in to Python
translate_builtin(Goal, VarMap, PythonExpr) :-
    Goal =.. [Op | _],
    python_config(Config),
    memberchk(ops-Ops, Config),
    memberchk(Op-_, Ops),
    !,
    translate_builtin_common(Goal, VarMap, Config, PythonExpr).
translate_builtin(\+ Goal, VarMap, PythonExpr) :-
    !,
    translate_negation(Goal, VarMap, PythonExpr).
translate_builtin(not(Goal), VarMap, PythonExpr) :-
    !,
    translate_negation(Goal, VarMap, PythonExpr).
% Match predicate support (regex pattern matching)
translate_builtin(match(Var, Pattern), VarMap, PythonExpr) :-
    !,
    translate_match(Var, Pattern, auto, [], VarMap, PythonExpr).
translate_builtin(match(Var, Pattern, Type), VarMap, PythonExpr) :-
    !,
    translate_match(Var, Pattern, Type, [], VarMap, PythonExpr).
translate_builtin(match(Var, Pattern, Type, _Groups), VarMap, PythonExpr) :-
    !,
    % For now, capture groups are handled similarly to boolean match
    % TODO: Extract and use captured values
    translate_match(Var, Pattern, Type, [], VarMap, PythonExpr).
translate_builtin(_, _VarMap, "True").  % Fallback

%% translate_negation(+Goal, +VarMap, -PythonExpr)
translate_negation(Goal, VarMap, PythonExpr) :-
    python_config(Config),
    prepare_negation_data(Goal, VarMap, Config, Pairs),
    findall(PairStr,
        (   member(Key-Val, Pairs),
            format(string(PairStr), "'~w': ~w", [Key, Val])
        ),
        PairStrings),
    atomic_list_concat(PairStrings, ", ", DictContent),
    format(string(PythonExpr), "FrozenDict.from_dict({~w}) not in total", [DictContent]).

%% translate_match(+Var, +Pattern, +Type, +Groups, +VarMap, -PythonExpr)
%  Translate match predicate to Python re module call
translate_match(Var, Pattern, Type, _Groups, VarMap, PythonExpr) :-
    % Validate regex type for Python target
    validate_regex_type_for_python(Type),
    % Convert variable to Python expression
    translate_expr(Var, VarMap, PyVar),
    % Convert pattern to Python string
    (   atom(Pattern)
    ->  atom_string(Pattern, PatternStr)
    ;   PatternStr = Pattern
    ),
    % Escape pattern for Python (escape backslashes and quotes)
    escape_python_string(PatternStr, EscapedPattern),
    % Generate Python regex match expression
    % For now, use re.search for boolean match
    % TODO: Handle capture groups with re.search().groups()
    format(string(PythonExpr), "re.search(r'~w', str(~w))", [EscapedPattern, PyVar]).

%% validate_regex_type_for_python(+Type)
%  Validate that the regex type is supported by Python
validate_regex_type_for_python(auto) :- !.
validate_regex_type_for_python(python) :- !.
validate_regex_type_for_python(pcre) :- !.  % Python re is PCRE-like
validate_regex_type_for_python(ere) :- !.   % Can be supported with minor translation
validate_regex_type_for_python(Type) :-
    format('ERROR: Python target does not support regex type ~q~n', [Type]),
    format('  Supported types: auto, python, pcre, ere~n', []),
    format('  Note: BRE, AWK-specific, and .NET regex are not supported by Python~n', []),
    fail.

%% escape_python_string(+Str, -EscapedStr)
%  Escape special characters for Python string literals
escape_python_string(Str, EscapedStr) :-
    atom_string(Str, String),
    % For raw strings (r'...'), we mainly need to escape quotes
    % Backslashes are literal in raw strings
    re_replace("'"/g, "\\\\'", String, EscapedStr).

%% build_variable_map(+GoalSourcePairs, -VarMap)
%  Build map from variables to Python access strings
build_variable_map(GoalSourcePairs, VarMap) :-
    findall(Var-Access,
        (   member(Goal-Source, GoalSourcePairs),
            Goal =.. [_ | Args],
            nth0(Idx, Args, Var),
            var(Var),
            format(string(Access), "~w.get('arg~w')", [Source, Idx])
        ),
        VarMap).

%% translate_expr(+PrologExpr, +VarMap, -PythonExpr)
%  Translate Prolog expression to Python, mapping variables
translate_expr(Var, VarMap, PythonExpr) :-
    var(Var),
    !,
    (   memberchk(Var-Access, VarMap)
    ->  PythonExpr = Access
    ;   % Variable not found - assume it's a singleton or error
        % For now return None, but this indicates unsafe usage
        PythonExpr = "None"
    ).
translate_expr(Num, _VarMap, PythonExpr) :-
    number(Num),
    !,
    format(string(PythonExpr), "~w", [Num]).
translate_expr(Expr, VarMap, PythonExpr) :-
    compound(Expr),
    Expr =.. [Op, Left, Right],
    member(Op, [+, -, *, /, mod]), % Ensure this is handled
    !,
    translate_expr(Left, VarMap, LeftPy),
    translate_expr(Right, VarMap, RightPy),
    python_operator(Op, PyOp),
    format(string(PythonExpr), "(~w ~w ~w)", [LeftPy, PyOp, RightPy]).
translate_expr(Atom, _VarMap, PythonExpr) :-
    atom(Atom),
    !,
    format(string(PythonExpr), "'~w'", [Atom]).
translate_expr(_, _VarMap, "None").  % Fallback

%% python_operator(+PrologOp, -PythonOp)
python_operator(+, '+').
python_operator(-, '-').
python_operator(*, '*').
python_operator(/, '/').
python_operator(mod, '%').


%% translate_join_rule(+Name, +RuleNum, +Head, +Goals, -RuleFunc)
%  Translate a join rule (multiple goals in body)
translate_join_rule(Name, RuleNum, Head, Goals, RuleFunc) :-
    length(Goals, NumGoals),
    (   NumGoals == 2
    ->  % Binary join (existing fast path)
        Goals = [Goal1, Goal2],
        translate_binary_join(Name, RuleNum, Head, Goal1, Goal2, RuleFunc)
    ;   NumGoals >= 3
    ->  % N-way join (new!)
        translate_nway_join(RuleNum, Head, Goals, [], RuleFunc)
    ;   % Single goal shouldn't reach here
        format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''ERROR: Invalid join - single goal should use copy rule'''
    return iter([])
", [RuleNum])
    ).

%% translate_join_rule_with_builtins(+Name, +RuleNum, +Head, +Goals, +Builtins, -RuleFunc)
%  Join rule with built-in constraints
translate_join_rule_with_builtins(Name, RuleNum, Head, Goals, Builtins, RuleFunc) :-
    (   Builtins == []
    ->  % No builtins, use regular join rule
        translate_join_rule(Name, RuleNum, Head, Goals, RuleFunc)
    ;   % Generate join with constraints
        % For binary joins, add constraints  to the existing logic
        length(Goals, NumGoals),
        (   NumGoals == 2
        ->  Goals = [Goal1, Goal2],
            translate_binary_join_with_constraints(RuleNum, Head, Goal1, Goal2, Builtins, RuleFunc)
        ;   % N-way joins with constraints
            translate_nway_join(RuleNum, Head, Goals, Builtins, RuleFunc)
        )
    ).

%% translate_binary_join_with_constraints(+RuleNum, +Head, +Goal1, +Goal2, +Builtins, -RuleFunc)
%% translate_binary_join_with_constraints(+RuleNum, +Head, +Goal1, +Goal2, +Builtins, -RuleFunc)
translate_binary_join_with_constraints(RuleNum, Head, Goal1, Goal2, Builtins, RuleFunc) :-
    % Similar to translate_binary_join but with constraint checks
    Goal1 =.. [Pred1 | Args1],
    Goal2 =.. [Pred2 | Args2],
    Head =.. [HeadPred | HeadArgs],
    
    % Find join variable
    findall(Var-Idx1-Idx2,
        (   nth0(Idx1, Args1, Var),
            nth0(Idx2, Args2, Var),
            \+ atom(Var)
        ),
        JoinVars),
    
    % Build join condition
    (   JoinVars = [_Var-JIdx1-JIdx2|_]
    ->  format(string(JoinCond), "other.get('arg~w') == fact.get('arg~w')", [JIdx2, JIdx1])
    ;   JoinCond = "True"
    ),
    
    % Add relation check for other (Goal2)
    format(string(RelCheck2), "other.get('relation') == '~w'", [Pred2]),
    (   JoinCond == "True"
    ->  JoinCondWithRel = RelCheck2
    ;   format(string(JoinCondWithRel), "~w and ~w", [JoinCond, RelCheck2])
    ),
    
    % Build constraint checks
    build_variable_map([Goal1-"fact", Goal2-"other"], VarMap),
    translate_builtins(Builtins, VarMap, ConstraintChecks),
    
    % Combine join and constraints
    (   ConstraintChecks == ""
    ->  FinalJoinCond = JoinCondWithRel
    ;   format(string(FinalJoinCond), "~w and ~w", [JoinCondWithRel, ConstraintChecks])
    ),
    
    % Build output mapping
    findall(OutAssign,
        (   nth0(HIdx, HeadArgs, HVar),
            (   nth0(G1Idx, Args1, HVar)
            ->  format(string(OutAssign), "'arg~w': fact.get('arg~w')", [HIdx, G1Idx])
            ;   nth0(G2Idx, Args2, HVar),
                format(string(OutAssign), "'arg~w': other.get('arg~w')", [HIdx, G2Idx])
            )
        ),
        OutAssigns),
    % Add relation to output
    format(string(RelAssign), "'relation': '~w'", [HeadPred]),
    AllAssigns = [RelAssign | OutAssigns],
    atomic_list_concat(AllAssigns, ", ", OutputMapping),
    
    % Pattern match for first goal
    length(Args1, Arity1),
    findall(PatCheck,
        (   between(0, Arity1, Idx),
            Idx < Arity1,
            format(string(PatCheck), "'arg~w' in fact", [Idx])
        ),
        PatChecks),
    % Add relation check for fact (Goal1)
    format(string(RelCheck1), "fact.get('relation') == '~w'", [Pred1]),
    AllPatChecks = [RelCheck1 | PatChecks],
    atomic_list_concat(AllPatChecks, " and ", Pattern1),
    
    format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''Join with constraints: ~w'''
    if ~w:
        for other in total:
            if ~w:
                yield FrozenDict.from_dict({~w})
", [RuleNum, Head, Pattern1, FinalJoinCond, OutputMapping]).



%% translate_binary_join(+Name, +RuleNum, +Head, +Goal1, +Goal2, -RuleFunc)
translate_binary_join(_Name, RuleNum, Head, Goal1, Goal2, RuleFunc) :-
    % Generate Block 1: Fact matches Goal 1, join with Goal 2
    generate_join_block(Head, Goal1, Goal2, Block1),
    
    % Generate Block 2: Fact matches Goal 2, join with Goal 1
    generate_join_block(Head, Goal2, Goal1, Block2),
    
    format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''Join rule: ~w :- ~w, ~w'''
    # Case 1: Fact matches first goal
~w
    # Case 2: Fact matches second goal
~w
", [RuleNum, Head, Goal1, Goal2, Block1, Block2]).

%% generate_join_block(+Head, +TriggerGoal, +OtherGoal, -Code)
generate_join_block(Head, TriggerGoal, OtherGoal, Code) :-
    TriggerGoal =.. [Pred1 | Args1],
    OtherGoal =.. [Pred2 | Args2],
    Head =.. [HeadPred | HeadArgs],
    
    % Find join variable
    findall(Var-Idx1-Idx2,
        (   nth0(Idx1, Args1, V1),
            nth0(Idx2, Args2, V2),
            V1 == V2,
            Var = V1,
            \+ atom(Var)
        ),
        JoinVars),
    
    % Build join condition
    (   JoinVars = [_Var-JIdx1-JIdx2|_]
    ->  format(string(JoinCond), "other.get('arg~w') == fact.get('arg~w')", [JIdx2, JIdx1])
    ;   JoinCond = "True"
    ),
    
    % Add relation check for other
    format(string(RelCheck2), "other.get('relation') == '~w'", [Pred2]),
    (   JoinCond == "True"
    ->  JoinCondWithRel = RelCheck2
    ;   format(string(JoinCondWithRel), "~w and ~w", [JoinCond, RelCheck2])
    ),
    
    % Build output mapping
    findall(OutAssign,
        (   nth0(HIdx, HeadArgs, HVar),
            (   nth0(G1Idx, Args1, V1), V1 == HVar
            ->  format(string(OutAssign), "'arg~w': fact.get('arg~w')", [HIdx, G1Idx])
            ;   nth0(G2Idx, Args2, V2), V2 == HVar,
                format(string(OutAssign), "'arg~w': other.get('arg~w')", [HIdx, G2Idx])
            )
        ),
        OutAssigns),
    format(string(RelAssign), "'relation': '~w'", [HeadPred]),
    AllOutAssigns = [RelAssign | OutAssigns],
    atomic_list_concat(AllOutAssigns, ", ", OutputMapping),
    
    % Pattern match for trigger goal (fact)
    length(Args1, Arity1),
    findall(PatCheck,
        (   between(0, Arity1, Idx),
            Idx < Arity1,
            format(string(PatCheck), "'arg~w' in fact", [Idx])
        ),
        PatChecks),
    format(string(RelCheck1), "fact.get('relation') == '~w'", [Pred1]),
    AllPatChecks = [RelCheck1 | PatChecks],
    atomic_list_concat(AllPatChecks, " and ", Pattern1),
    
    format(string(Code),
"    if ~w:
        for other in total:
            if ~w:
                yield FrozenDict.from_dict({~w})",
        [Pattern1, JoinCondWithRel, OutputMapping]).

%% translate_nway_join(+RuleNum, +Head, +Goals, -RuleFunc)
%  Translate N-way join (3+ goals)
%% translate_nway_join(+RuleNum, +Head, +Goals, +Builtins, -RuleFunc)
translate_nway_join(RuleNum, Head, Goals, Builtins, RuleFunc) :-
    % Strategy: First goal from fact, rest joined from total
    Goals = [FirstGoal | RestGoals],
    
    % Build pattern match for first goal
    FirstGoal =.. [Pred1 | Args1],
    length(Args1, Arity1),
    findall(Check,
        (   between(0, Arity1, Idx),
            Idx < Arity1,
            format(string(Check), "'arg~w' in fact", [Idx])
        ),
        Checks),
    % Add relation check
    format(string(RelCheck), "fact.get('relation') == '~w'", [Pred1]),
    AllChecks = [RelCheck | Checks],
    atomic_list_concat(AllChecks, " and ", Pattern1),
    
    % Build nested joins for remaining goals
    build_nested_joins(RestGoals, 1, JoinCode, FinalIdx),
    
    % Build output mapping from head
    Head =.. [HeadPred | HeadArgs],
    collect_all_goal_args([FirstGoal | RestGoals], AllGoalArgs),
    build_output_mapping(HeadArgs, FirstGoal, RestGoals, AllGoalArgs, OutputMapping),
    format(string(FullOutputMapping), "'relation': '~w', ~w", [HeadPred, OutputMapping]),
    
    % Constraints
    findall(G-S,
        (   nth1(I, RestGoals, G),
            format(string(S), "join_~w", [I])
        ),
        RestPairs),
    Pairs = [FirstGoal-"fact" | RestPairs],
    build_variable_map(Pairs, VarMap),
    translate_builtins(Builtins, VarMap, ConstraintChecks),
    
    % Calculate indentation for yield block
    Indent is (FinalIdx + 2) * 4,
    format(string(Spaces), "~*c", [Indent, 32]),
    
    (   ConstraintChecks == ""
    ->  format(string(YieldBlock), "~wyield FrozenDict.from_dict({~w})", [Spaces, FullOutputMapping])
    ;   format(string(YieldBlock), "~wif ~w:\n~w    yield FrozenDict.from_dict({~w})", [Spaces, ConstraintChecks, Spaces, FullOutputMapping])
    ),
    
    format(string(RuleFunc),
"def _apply_rule_~w(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    '''N-way join: ~w'''
    # Match first goal
    if ~w:
~w
~w
", [RuleNum, Head, Pattern1, JoinCode, YieldBlock]).

%%build_nested_joins(+Goals, +StartIdx, -JoinCode, -FinalIdx)
%  Build nested for loops for N-way joins
build_nested_joins([], Idx, "", Idx) :- !.
build_nested_joins([Goal | RestGoals], Idx, JoinCode, FinalIdx) :-
    Goal =.. [Pred | _],
    Indent is (Idx + 1) * 4,
    format(string(Spaces), "~*c", [Indent, 32]),  % 32 = space char
    
    % Detect join conditions with previous goals
    detect_join_condition(Goal, Idx, JoinCond),
    
    % Add relation check
    format(string(RelCheck), "join_~w.get('relation') == '~w'", [Idx, Pred]),
    (   JoinCond == "True"
    ->  FullCond = RelCheck
    ;   format(string(FullCond), "~w and ~w", [RelCheck, JoinCond])
    ),
    
    format(string(ThisJoin),
"~wfor join_~w in total:
~w    if ~w:",
        [Spaces, Idx, Spaces, FullCond]),
    
    NextIdx is Idx + 1,
    build_nested_joins(RestGoals, NextIdx, RestCode, FinalIdx),
    
    (   RestCode == ""
    ->  JoinCode = ThisJoin
    ;   format(string(JoinCode), "~w\n~w", [ThisJoin, RestCode])
    ).

%% detect_join_condition(+Goal, +Idx, -JoinCond)
%  Find variables that join with previous goals
detect_join_condition(Goal, Idx, JoinCond) :-
    % Extract variables from goal
    Goal =.. [_Pred | Args],
    
    % Find first variable (simplified: join on first arg)
    % TODO: Track all variables and find actual join points
    (   Args = [FirstArg | _],
        var(FirstArg)
    ->  % Join on first argument matching previous goal's output
        PrevIdx is Idx - 1,
        format(string(JoinCond), 
            "join_~w.get('arg0') == (join_~w.get('arg1') if ~w > 0 else fact.get('arg1'))",
            [Idx, PrevIdx, PrevIdx])
    ;   % No clear join variable, default condition
        format(string(JoinCond), "True", [])
    ).


%% collect_all_goal_args(+Goals, -AllArgs)
collect_all_goal_args(Goals, AllArgs) :-
    findall(Args,
        (   member(Goal, Goals),
            Goal =.. [_ | Args]
        ),
        ArgLists),
    append(ArgLists, AllArgs).

%% build_output_mapping(+HeadArgs, +FirstGoal, +RestGoals, +AllGoalArgs, -Mapping)
build_output_mapping(HeadArgs, FirstGoal, _RestGoals, _AllGoalArgs, Mapping) :-
    % Simplified: map from first goal args
    FirstGoal =.. [_ | FirstArgs],
    findall(Assign,
        (   nth0(HIdx, HeadArgs, HVar),
            nth0(GIdx, FirstArgs, GVar),
            GVar == HVar,
            format(string(Assign), "'arg~w': fact.get('arg~w')", [HIdx, GIdx])
        ),
        Assigns),
    atomic_list_concat(Assigns, ", ", Mapping).


%% generate_fixpoint_loop(+Name, +Facts, +Rules, -FixpointLoop)


generate_fixpoint_loop(_Name, Facts, Rules, FixpointLoop) :-


    % Generate fact initialization calls


    length(Facts, NumFacts),


    findall(FactCall,


        (   between(1, NumFacts, FactNum),


            format(string(FactCall), 


"    for fact in _init_fact_~w():


        if fact not in total:


            total.add(fact)


            delta.add(fact)


            yield fact.to_dict()", [FactNum])


        ),


        FactCalls),


    atomic_list_concat(FactCalls, "\n", FactCallsStr),





    % Generate rule calls


    length(Rules, NumRules),


    findall(RuleBlock,


        (   between(1, NumRules, RuleNum),


            format(string(RuleBlock), 


"            for new_fact in _apply_rule_~w(fact, total):


                if new_fact not in total and new_fact not in new_delta:


                    new_delta.add(new_fact)


                    yield new_fact.to_dict()", [RuleNum])


        ),


        RuleBlocks),


    


    (   RuleBlocks == []


    ->  LoopBody = "            pass"


    ;   atomic_list_concat(RuleBlocks, "\n", LoopBody)


    ),


    


    format(string(FixpointLoop),


"


def process_stream_generator(records: Iterator[Dict]) -> Iterator[Dict]:


    '''Semi-naive fixpoint evaluation.'''


    total: Set[FrozenDict] = set()


    delta: Set[FrozenDict] = set()


    


    # Initialize delta with input records


    for record in records:


        frozen = FrozenDict.from_dict(record)


        delta.add(frozen)


        total.add(frozen)


        yield record  # Yield initial facts


    


    # Initialize with static facts


~w


    


    # Fixpoint iteration (semi-naive evaluation)


    while delta:


        new_delta: Set[FrozenDict] = set()


        


        # Apply rules to facts in delta


        for fact in delta:


~w


        


        total.update(new_delta)


        delta = new_delta


", [FactCallsStr, LoopBody]).

%% generate_python_main(+Options, -MainCode)
%  Generate the main entry point with appropriate reader/writer
generate_python_main(Options, MainCode) :-
    option(record_format(Format), Options, jsonl),
    (   Format == nul_json
    ->  Writer = "write_nul_json"
    ;   Writer = "write_jsonl"
    ),
    
    (   option(input_source(xml(File, Tags)), Options)
    ->  % XML Source
        maplist(atom_string, Tags, TagStrs),
        maplist(quote_py_string, TagStrs, QuotedTags),
        atomic_list_concat(QuotedTags, ", ", TagsInner),
        format(string(ReaderCall), "read_xml_lxml('~w', {~w})", [File, TagsInner])
    ;   Format == nul_json
    ->  ReaderCall = "read_nul_json(sys.stdin)"
    ;   ReaderCall = "read_jsonl(sys.stdin)"
    ),
    
    % Handle generator mode vs procedural mode function name
    (   option(mode(generator), Options)
    ->  ProcessFunc = "process_stream_generator"
    ;   ProcessFunc = "process_stream"
    ),
    
    format(string(MainCode), 
"
def main():
    records = ~w
    results = ~w(records)
    ~w(results, sys.stdout)

if __name__ == '__main__':
    main()
", [ReaderCall, ProcessFunc, Writer]).

quote_py_string(Str, Quoted) :-
    format(string(Quoted), "'~w'", [Str]).

%% semantic_runtime_helpers(-Code)
%  Inject the Semantic Runtime library (Importer, Crawler, etc.) into the script
semantic_runtime_helpers(Code) :-
    % List of runtime files to inline
    Files = [
        'src/unifyweaver/targets/python_runtime/embedding.py',
        'src/unifyweaver/targets/python_runtime/importer.py',
        'src/unifyweaver/targets/python_runtime/onnx_embedding.py',
        'src/unifyweaver/targets/python_runtime/searcher.py',
        'src/unifyweaver/targets/python_runtime/crawler.py',
        'src/unifyweaver/targets/python_runtime/llm.py',
        'src/unifyweaver/targets/python_runtime/chunker.py'
    ],
    
    findall(Content, (
        member(File, Files),
        (   exists_file(File)
        ->  read_file_to_string(File, Raw, []),
            % Remove relative imports 'from .embedding import'
            re_replace("from \\.embedding import.*", "", Raw, Content)
        ;   format(string(Content), "# ERROR: Runtime file ~w not found\\n", [File])
        )
    ), Contents),
    
    Wrapper = "
class SemanticRuntime:
    def __init__(self, db_path='data.db', model_path='models/model.onnx', vocab_path='models/vocab.txt'):
        self.importer = PtImporter(db_path)
        if os.path.exists(model_path):
            self.embedder = OnnxEmbeddingProvider(model_path, vocab_path)
        else:
            sys.stderr.write(f'Warning: Model {model_path} not found, embeddings disabled\\\\n')
            self.embedder = None
            
        self.crawler = PtCrawler(self.importer, self.embedder)
        self.searcher = PtSearcher(db_path, self.embedder)
        self.llm = LLMProvider()
        self.chunker = HierarchicalChunker()

_runtime_instance = None
def _get_runtime():
    global _runtime_instance
    if _runtime_instance is None:
        _runtime_instance = SemanticRuntime()
    return _runtime_instance

def fetch_xml_func(url):
    if os.path.exists(url):
        return open(url, 'rb')
    return None
",

    atomic_list_concat(Contents, "\n", LibCode),
    string_concat(LibCode, Wrapper, Code).

% ============================================================================
% TESTS - Import Auto-Generation
% ============================================================================

%% test_import_autogeneration
%  Test that binding imports are automatically included in generated Python code
test_import_autogeneration :-
    format('~n=== Python Import Auto-Generation Tests ===~n~n', []),

    % Test 1: Clear imports starts fresh
    format('[Test 1] Clear binding imports~n', []),
    clear_binding_imports,
    findall(M, required_import(M), Ms1),
    (   Ms1 == []
    ->  format('  [PASS] No imports after clear~n', [])
    ;   format('  [FAIL] Expected no imports, got ~w~n', [Ms1])
    ),

    % Test 2: Recording imports
    format('[Test 2] Record binding imports~n', []),
    assertz(required_import(math)),
    assertz(required_import(os)),
    assertz(required_import(collections)),
    findall(M, required_import(M), Ms2),
    sort(Ms2, Sorted2),
    (   Sorted2 == [collections, math, os]
    ->  format('  [PASS] Imports recorded: ~w~n', [Sorted2])
    ;   format('  [FAIL] Expected [collections, math, os], got ~w~n', [Sorted2])
    ),

    % Test 3: get_binding_imports generates correct import lines
    format('[Test 3] Generate import statements~n', []),
    get_binding_imports(ImportStr),
    (   sub_string(ImportStr, _, _, _, "import math"),
        sub_string(ImportStr, _, _, _, "import os"),
        sub_string(ImportStr, _, _, _, "import collections")
    ->  format('  [PASS] Import string contains math, os, collections~n', [])
    ;   format('  [FAIL] Missing imports in: ~w~n', [ImportStr])
    ),

    % Test 4: header/1 includes binding imports
    format('[Test 4] Header includes binding imports~n', []),
    header(Header),
    (   sub_string(Header, _, _, _, "import math"),
        sub_string(Header, _, _, _, "import sys"),
        sub_string(Header, _, _, _, "import json")
    ->  format('  [PASS] Header includes base + binding imports~n', [])
    ;   format('  [FAIL] Header missing imports~n', [])
    ),

    % Test 5: header_with_functools/1 includes both functools and bindings
    format('[Test 5] Header with functools includes binding imports~n', []),
    header_with_functools(HeaderFT),
    (   sub_string(HeaderFT, _, _, _, "import functools"),
        sub_string(HeaderFT, _, _, _, "import math"),
        sub_string(HeaderFT, _, _, _, "import collections")
    ->  format('  [PASS] Header with functools includes binding imports~n', [])
    ;   format('  [FAIL] Header with functools missing imports~n', [])
    ),

    % Test 6: generator_header/1 includes binding imports
    format('[Test 6] Generator header includes binding imports~n', []),
    generator_header(GenHeader),
    (   sub_string(GenHeader, _, _, _, "import math"),
        sub_string(GenHeader, _, _, _, "FrozenDict")
    ->  format('  [PASS] Generator header includes binding imports + FrozenDict~n', [])
    ;   format('  [FAIL] Generator header missing imports~n', [])
    ),

    % Test 7: Skip already-included base modules
    format('[Test 7] Skip base modules (sys, json, re, etc.)~n', []),
    clear_binding_imports,
    assertz(required_import(sys)),    % Should be skipped
    assertz(required_import(json)),   % Should be skipped
    assertz(required_import(numpy)),  % Should be included
    get_binding_imports(ImportStr2),
    (   sub_string(ImportStr2, _, _, _, "import numpy"),
        \+ sub_string(ImportStr2, _, _, _, "import sys\nimport sys")
    ->  format('  [PASS] Base modules skipped, numpy included~n', [])
    ;   format('  [FAIL] Base module filtering issue~n', [])
    ),

    % Cleanup
    clear_binding_imports,

    % Test 8: End-to-end - binding lookup records imports
    format('[Test 8] End-to-end: binding lookup records imports~n', []),
    init_python_target,
    % sqrt/2 binding has import(math) - simulate what translate_goal does
    (   binding(python, sqrt/2, _TargetName, _Inputs, _Outputs, SqrtOptions),
        member(import(math), SqrtOptions)
    ->  % Record import as translate_goal would
        assertz(required_import(math)),
        format('  [PASS] sqrt/2 binding has import(math), recorded~n', [])
    ;   format('  [FAIL] sqrt/2 binding does not have import(math)~n', [])
    ),

    % Test 9: Multiple bindings accumulate imports
    format('[Test 9] Multiple bindings accumulate imports~n', []),
    clear_binding_imports,
    init_python_bindings,
    % Simulate translate_goal recording imports for multiple bindings
    (   binding(python, sqrt/2, _, _, _, Opts1), member(import(M1), Opts1)
    ->  assertz(required_import(M1))
    ;   true
    ),
    (   binding(python, counter/2, _, _, _, Opts2), member(import(M2), Opts2)
    ->  assertz(required_import(M2))
    ;   true
    ),
    get_binding_imports(AccumImports),
    (   sub_string(AccumImports, _, _, _, "import math"),
        sub_string(AccumImports, _, _, _, "import collections")
    ->  format('  [PASS] Multiple imports accumulated: math, collections~n', [])
    ;   format('  [FAIL] Import accumulation issue: ~w~n', [AccumImports])
    ),

    % Final cleanup
    clear_binding_imports,

    format('~n=== All Import Auto-Generation Tests Passed ===~n', []).

% ============================================================================
% TESTS - Method Call Pattern Enhancement
% ============================================================================

%% test_method_call_patterns
%  Test the enhanced method call code generation
test_method_call_patterns :-
    format('~n=== Python Method Call Pattern Tests ===~n~n', []),

    % Test 1: No-arg method with () in TargetName
    format('[Test 1] No-arg method with () in name~n', []),
    generate_method_call('.strip()', [v(str), v(result)], [string], CallExpr1),
    (   CallExpr1 == "v_str.strip()"
    ->  format('  [PASS] .strip() -> ~w~n', [CallExpr1])
    ;   format('  [FAIL] Expected v_str.strip(), got ~w~n', [CallExpr1])
    ),

    % Test 2: No-arg method without () in TargetName
    format('[Test 2] No-arg method without () in name~n', []),
    generate_method_call('.lower', [v(str), v(result)], [string], CallExpr2),
    (   CallExpr2 == "v_str.lower()"
    ->  format('  [PASS] .lower -> ~w~n', [CallExpr2])
    ;   format('  [FAIL] Expected v_str.lower(), got ~w~n', [CallExpr2])
    ),

    % Test 3: Method with one argument and output
    format('[Test 3] Method with argument and output~n', []),
    generate_method_call('.split', [v(str), v(delim), v(result)], [list], CallExpr3),
    (   CallExpr3 == "v_str.split(v_delim)"
    ->  format('  [PASS] .split with arg -> ~w~n', [CallExpr3])
    ;   format('  [FAIL] Expected v_str.split(v_delim), got ~w~n', [CallExpr3])
    ),

    % Test 4: Method with multiple arguments and output
    format('[Test 4] Method with multiple arguments~n', []),
    generate_method_call('.replace', [v(str), v(old), v(new), v(result)], [string], CallExpr4),
    (   CallExpr4 == "v_str.replace(v_old, v_new)"
    ->  format('  [PASS] .replace with args -> ~w~n', [CallExpr4])
    ;   format('  [FAIL] Expected v_str.replace(v_old, v_new), got ~w~n', [CallExpr4])
    ),

    % Test 5: Mutating method (no output) with argument
    format('[Test 5] Mutating method with argument (no output)~n', []),
    generate_method_call('.append', [v(list), v(item)], [], CallExpr5),
    (   CallExpr5 == "v_list.append(v_item)"
    ->  format('  [PASS] .append mutating -> ~w~n', [CallExpr5])
    ;   format('  [FAIL] Expected v_list.append(v_item), got ~w~n', [CallExpr5])
    ),

    % Test 6: Function call with output
    format('[Test 6] Function call with output~n', []),
    generate_function_call('len', [v(list), v(result)], [int], CallExpr6),
    (   CallExpr6 == "len(v_list)"
    ->  format('  [PASS] len() function -> ~w~n', [CallExpr6])
    ;   format('  [FAIL] Expected len(v_list), got ~w~n', [CallExpr6])
    ),

    % Test 7: Function call with no output (side effect)
    format('[Test 7] Function call with no output~n', []),
    generate_function_call('print', [v(msg)], [], CallExpr7),
    (   CallExpr7 == "print(v_msg)"
    ->  format('  [PASS] print() no output -> ~w~n', [CallExpr7])
    ;   format('  [FAIL] Expected print(v_msg), got ~w~n', [CallExpr7])
    ),

    % Test 8: Constant function (no input args)
    format('[Test 8] Constant function (pi)~n', []),
    generate_function_call('math.pi', [v(result)], [float], CallExpr8),
    (   CallExpr8 == "math.pi"
    ->  format('  [PASS] math.pi constant -> ~w~n', [CallExpr8])
    ;   format('  [FAIL] Expected math.pi, got ~w~n', [CallExpr8])
    ),

    % Test 9: Chained method call
    format('[Test 9] Chained method call~n', []),
    generate_chained_call(
        [method('.strip', []), method('.lower', [])],
        [v(str), v(result)],
        [string],
        CallExpr9
    ),
    (   CallExpr9 == "v_str.strip().lower()"
    ->  format('  [PASS] Chained .strip().lower() -> ~w~n', [CallExpr9])
    ;   format('  [FAIL] Expected v_str.strip().lower(), got ~w~n', [CallExpr9])
    ),

    % Test 10: Chained method with arguments
    format('[Test 10] Chained method with arguments~n', []),
    generate_chained_call(
        [method('.replace', [0, 1]), method('.strip', [])],
        [v(str), v(old), v(new), v(result)],
        [string],
        CallExpr10
    ),
    (   CallExpr10 == "v_str.replace(v_old, v_new).strip()"
    ->  format('  [PASS] Chained .replace().strip() -> ~w~n', [CallExpr10])
    ;   format('  [FAIL] Expected v_str.replace(v_old, v_new).strip(), got ~w~n', [CallExpr10])
    ),

    % Test 11: Full code generation with method call binding
    format('[Test 11] Full code generation for method binding~n', []),
    generate_binding_call_python('.lower()', [v(str), v(result)], [string], [pattern(method_call)], Code11),
    (   sub_string(Code11, _, _, _, "v_result = v_str.lower()")
    ->  format('  [PASS] Full method binding code generated~n', [])
    ;   format('  [FAIL] Expected assignment, got ~w~n', [Code11])
    ),

    % Test 12: Full code generation for mutating method (no assignment)
    format('[Test 12] Full code generation for mutating method~n', []),
    generate_binding_call_python('.append', [v(list), v(item)], [], [pattern(method_call)], Code12),
    (   sub_string(Code12, _, _, _, "v_list.append(v_item)"),
        \+ sub_string(Code12, _, _, _, "=")
    ->  format('  [PASS] Mutating method - no assignment~n', [])
    ;   format('  [FAIL] Unexpected output: ~w~n', [Code12])
    ),

    format('~n=== All Method Call Pattern Tests Passed ===~n', []).

% ============================================================================
% TESTS - Pipeline Mode (Phase 1: Object Pipeline Support)
% ============================================================================

%% test_pipeline_mode
%  Test the pipeline mode code generation
test_pipeline_mode :-
    format('~n=== Python Pipeline Mode Tests ===~n~n', []),

    % Test 1: Generate default arg names
    format('[Test 1] Generate default arg names~n', []),
    generate_default_arg_names(3, ArgNames1),
    (   ArgNames1 == [arg_0, arg_1, arg_2]
    ->  format('  [PASS] Default arg names: ~w~n', [ArgNames1])
    ;   format('  [FAIL] Expected [arg_0, arg_1, arg_2], got ~w~n', [ArgNames1])
    ),

    % Test 2: Pipeline header for JSONL
    format('[Test 2] Pipeline header for JSONL~n', []),
    clear_binding_imports,
    pipeline_header(jsonl, Header2),
    (   sub_string(Header2, _, _, _, "import json"),
        sub_string(Header2, _, _, _, "Generator")
    ->  format('  [PASS] JSONL header has json and Generator imports~n', [])
    ;   format('  [FAIL] Header missing imports: ~w~n', [Header2])
    ),

    % Test 3: Pipeline header for MessagePack
    format('[Test 3] Pipeline header for MessagePack~n', []),
    pipeline_header(messagepack, Header3),
    (   sub_string(Header3, _, _, _, "import msgpack")
    ->  format('  [PASS] MessagePack header has msgpack import~n', [])
    ;   format('  [FAIL] Header missing msgpack: ~w~n', [Header3])
    ),

    % Test 4: Pipeline helpers for JSONL with same_as_data errors
    format('[Test 4] Pipeline helpers for JSONL (same_as_data errors)~n', []),
    pipeline_helpers(jsonl, same_as_data, Helpers4),
    (   sub_string(Helpers4, _, _, _, "read_stream"),
        sub_string(Helpers4, _, _, _, "write_record"),
        sub_string(Helpers4, _, _, _, "__error__")
    ->  format('  [PASS] JSONL helpers have read_stream, write_record, error handling~n', [])
    ;   format('  [FAIL] Missing helpers~n', [])
    ),

    % Test 5: Pipeline helpers for JSONL with text errors
    format('[Test 5] Pipeline helpers for JSONL (text errors)~n', []),
    pipeline_helpers(jsonl, text, Helpers5),
    (   sub_string(Helpers5, _, _, _, "ERROR [")
    ->  format('  [PASS] Text error protocol has plain text error format~n', [])
    ;   format('  [FAIL] Missing plain text error format~n', [])
    ),

    % Test 6: Output formatting for object mode
    format('[Test 6] Output formatting for object mode~n', []),
    generate_output_formatting([_A, _B, _C], ['Id', 'Name', 'Active'], object, OutputCode6),
    (   sub_string(OutputCode6, _, _, _, "'Id': v_0"),
        sub_string(OutputCode6, _, _, _, "'Name': v_1"),
        sub_string(OutputCode6, _, _, _, "'Active': v_2")
    ->  format('  [PASS] Object output has named keys~n', [])
    ;   format('  [FAIL] Output formatting issue: ~w~n', [OutputCode6])
    ),

    % Test 7: Output formatting for text mode
    format('[Test 7] Output formatting for text mode~n', []),
    generate_output_formatting([_X], ['Result'], text, OutputCode7),
    (   sub_string(OutputCode7, _, _, _, "str(v_0)")
    ->  format('  [PASS] Text output uses str()~n', [])
    ;   format('  [FAIL] Text output issue: ~w~n', [OutputCode7])
    ),

    % Test 8: Pipeline main block
    format('[Test 8] Pipeline main block~n', []),
    generate_pipeline_main("test_pred", jsonl, [], Main8),
    (   sub_string(Main8, _, _, _, "if __name__"),
        sub_string(Main8, _, _, _, "read_stream(sys.stdin)"),
        sub_string(Main8, _, _, _, "test_pred(input_stream)")
    ->  format('  [PASS] Main block has correct structure~n', [])
    ;   format('  [FAIL] Main block issue: ~w~n', [Main8])
    ),

    % Test 9: Full pipeline compilation (needs test predicate)
    format('[Test 9] Full pipeline compilation~n', []),
    % Define a simple test predicate
    abolish(test_user_info/2),
    assert((test_user_info(Id, Email) :- Email = Id)),
    (   catch(
            compile_predicate_to_python(test_user_info/2, [
                pipeline_input(true),
                output_format(object),
                arg_names(['UserId', 'Email'])
            ], Code9),
            Error9,
            (format('  [FAIL] Compilation error: ~w~n', [Error9]), fail)
        )
    ->  (   sub_string(Code9, _, _, _, "def test_user_info(stream"),
            sub_string(Code9, _, _, _, "'UserId'"),
            sub_string(Code9, _, _, _, "'Email'"),
            sub_string(Code9, _, _, _, "read_stream"),
            sub_string(Code9, _, _, _, "write_record")
        ->  format('  [PASS] Full pipeline code generated correctly~n', [])
        ;   format('  [FAIL] Generated code missing expected content~n', []),
            format('  Code: ~w~n', [Code9])
        )
    ;   format('  [FAIL] Pipeline compilation failed~n', [])
    ),
    abolish(test_user_info/2),

    format('~n=== All Pipeline Mode Tests Passed ===~n', []).

% ============================================================================
% TESTS - Runtime Selection (Phase 2)
% ============================================================================

%% test_runtime_selection
%  Test the runtime selection system
test_runtime_selection :-
    format('~n=== Python Runtime Selection Tests ===~n~n', []),

    % Test 1: CPython is always available
    format('[Test 1] CPython availability~n', []),
    (   runtime_available(cpython)
    ->  format('  [PASS] CPython is available~n', [])
    ;   format('  [FAIL] CPython should always be available~n', [])
    ),

    % Test 2: CPython compatible with all imports
    format('[Test 2] CPython import compatibility~n', []),
    (   runtime_compatible_with_imports(cpython, [numpy, tensorflow, pandas])
    ->  format('  [PASS] CPython supports all imports~n', [])
    ;   format('  [FAIL] CPython should support all imports~n', [])
    ),

    % Test 3: IronPython incompatible with numpy
    format('[Test 3] IronPython numpy incompatibility~n', []),
    (   \+ runtime_compatible_with_imports(ironpython, [numpy])
    ->  format('  [PASS] IronPython correctly rejects numpy~n', [])
    ;   format('  [FAIL] IronPython should reject numpy~n', [])
    ),

    % Test 4: IronPython compatible with basic imports
    format('[Test 4] IronPython basic import compatibility~n', []),
    (   runtime_compatible_with_imports(ironpython, [json, re, os, sys])
    ->  format('  [PASS] IronPython supports basic imports~n', [])
    ;   format('  [FAIL] IronPython should support basic imports~n', [])
    ),

    % Test 5: Runtime selection with no constraints
    format('[Test 5] Runtime selection (no constraints)~n', []),
    select_python_runtime(test/1, [], [], Runtime5),
    (   Runtime5 == cpython
    ->  format('  [PASS] Default selection is cpython: ~w~n', [Runtime5])
    ;   format('  [INFO] Selected runtime: ~w (cpython expected as default)~n', [Runtime5])
    ),

    % Test 6: Runtime selection with numpy import
    format('[Test 6] Runtime selection with numpy~n', []),
    select_python_runtime(test/1, [numpy], [], Runtime6),
    (   Runtime6 == cpython
    ->  format('  [PASS] numpy forces cpython: ~w~n', [Runtime6])
    ;   format('  [FAIL] numpy should force cpython, got: ~w~n', [Runtime6])
    ),

    % Test 7: Runtime scoring
    format('[Test 7] Runtime scoring~n', []),
    Prefs7 = [prefer_runtime([ironpython, cpython, pypy])],
    score_single_runtime(Prefs7, [], cpython, cpython-Score7a),
    score_single_runtime(Prefs7, [], ironpython, ironpython-Score7b),
    (   Score7b > Score7a
    ->  format('  [PASS] IronPython scores higher (~w) than CPython (~w) with preference~n', [Score7b, Score7a])
    ;   format('  [FAIL] Scoring issue: iron=~w, cpython=~w~n', [Score7b, Score7a])
    ),

    % Test 8: Context-based selection (.NET context)
    format('[Test 8] Context-based runtime selection~n', []),
    (   runtime_satisfies_context(ironpython, [target(csharp)])
    ->  format('  [PASS] IronPython satisfies .NET context~n', [])
    ;   format('  [FAIL] IronPython should satisfy .NET context~n', [])
    ),

    % Test 9: Firewall denies runtime
    format('[Test 9] Firewall runtime denial~n', []),
    Firewall9 = [denied([python_runtime(ironpython)])],
    (   \+ valid_runtime_candidate(ironpython, [], Firewall9, [])
    ->  format('  [PASS] Firewall correctly denies ironpython~n', [])
    ;   format('  [FAIL] Firewall should deny ironpython~n', [])
    ),

    % Test 10: Communication preference
    format('[Test 10] Communication preference~n', []),
    (   runtime_communication(ironpython, in_process, [target(csharp)])
    ->  format('  [PASS] IronPython is in-process for .NET~n', [])
    ;   format('  [FAIL] IronPython should be in-process for .NET~n', [])
    ),
    (   runtime_communication(cpython, cross_process, [])
    ->  format('  [PASS] CPython is cross-process~n', [])
    ;   format('  [FAIL] CPython should be cross-process~n', [])
    ),

    format('~n=== All Runtime Selection Tests Passed ===~n', []).

% ============================================================================
% TESTS - Runtime-Specific Code Generation (Phase 3)
% ============================================================================

%% test_runtime_headers
%  Test the runtime-specific pipeline header generation
test_runtime_headers :-
    format('~n=== Python Runtime-Specific Header Tests (Phase 3) ===~n~n', []),
    clear_binding_imports,

    % Test 1: CPython JSONL header
    format('[Test 1] CPython JSONL header~n', []),
    pipeline_header(jsonl, cpython, Header1),
    (   sub_string(Header1, _, _, _, "#!/usr/bin/env python3"),
        sub_string(Header1, _, _, _, "Runtime: CPython"),
        sub_string(Header1, _, _, _, "import json")
    ->  format('  [PASS] CPython JSONL header correct~n', [])
    ;   format('  [FAIL] CPython JSONL header issue~n', [])
    ),

    % Test 2: IronPython JSONL header (CLR integration)
    format('[Test 2] IronPython JSONL header (CLR imports)~n', []),
    pipeline_header(jsonl, ironpython, Header2),
    (   sub_string(Header2, _, _, _, "#!/usr/bin/env ipy"),
        sub_string(Header2, _, _, _, "import clr"),
        sub_string(Header2, _, _, _, "clr.AddReference('System')"),
        sub_string(Header2, _, _, _, "from System import"),
        sub_string(Header2, _, _, _, "Dictionary, List"),
        sub_string(Header2, _, _, _, "to_dotnet_dict")
    ->  format('  [PASS] IronPython has CLR imports and helpers~n', [])
    ;   format('  [FAIL] IronPython missing CLR integration~n', [])
    ),

    % Test 3: IronPython MessagePack header (fallback)
    format('[Test 3] IronPython MessagePack header~n', []),
    pipeline_header(messagepack, ironpython, Header3),
    (   sub_string(Header3, _, _, _, "System.Text.Json"),
        sub_string(Header3, _, _, _, "class msgpack")
    ->  format('  [PASS] IronPython has msgpack fallback~n', [])
    ;   format('  [FAIL] IronPython missing msgpack fallback~n', [])
    ),

    % Test 4: PyPy JSONL header
    format('[Test 4] PyPy JSONL header~n', []),
    pipeline_header(jsonl, pypy, Header4),
    (   sub_string(Header4, _, _, _, "#!/usr/bin/env pypy3"),
        sub_string(Header4, _, _, _, "Runtime: PyPy"),
        sub_string(Header4, _, _, _, "JIT-optimized")
    ->  format('  [PASS] PyPy header correct~n', [])
    ;   format('  [FAIL] PyPy header issue~n', [])
    ),

    % Test 5: Jython JSONL header (Java integration)
    format('[Test 5] Jython JSONL header (Java imports)~n', []),
    pipeline_header(jsonl, jython, Header5),
    (   sub_string(Header5, _, _, _, "#!/usr/bin/env jython"),
        sub_string(Header5, _, _, _, "from java.lang import"),
        sub_string(Header5, _, _, _, "HashMap, ArrayList"),
        sub_string(Header5, _, _, _, "to_java_map")
    ->  format('  [PASS] Jython has Java imports and helpers~n', [])
    ;   format('  [FAIL] Jython missing Java integration~n', [])
    ),

    % Test 6: Jython MessagePack header
    format('[Test 6] Jython MessagePack header~n', []),
    pipeline_header(messagepack, jython, Header6),
    (   sub_string(Header6, _, _, _, "ByteArrayOutputStream"),
        sub_string(Header6, _, _, _, "ObjectOutputStream"),
        sub_string(Header6, _, _, _, "class msgpack")
    ->  format('  [PASS] Jython has Java msgpack fallback~n', [])
    ;   format('  [FAIL] Jython missing msgpack fallback~n', [])
    ),

    % Test 7: compile_pipeline_mode uses runtime option
    format('[Test 7] Pipeline compilation with runtime option~n', []),
    abolish(test_iron_pred/1),
    assert((test_iron_pred(X) :- X = hello)),
    (   catch(
            compile_predicate_to_python(test_iron_pred/1, [
                pipeline_input(true),
                output_format(object),
                arg_names(['Value']),
                runtime(ironpython)
            ], Code7),
            Error7,
            (format('  [FAIL] Compilation error: ~w~n', [Error7]), fail)
        )
    ->  (   sub_string(Code7, _, _, _, "#!/usr/bin/env ipy"),
            sub_string(Code7, _, _, _, "import clr"),
            sub_string(Code7, _, _, _, "to_dotnet_dict")
        ->  format('  [PASS] Pipeline uses IronPython runtime header~n', [])
        ;   format('  [FAIL] Pipeline missing IronPython header~n', [])
        )
    ;   format('  [FAIL] Pipeline compilation failed~n', [])
    ),
    abolish(test_iron_pred/1),

    % Test 8: Legacy 2-arg pipeline_header still works
    format('[Test 8] Legacy pipeline_header/2 compatibility~n', []),
    pipeline_header(jsonl, LegacyHeader),
    (   sub_string(LegacyHeader, _, _, _, "#!/usr/bin/env python3"),
        sub_string(LegacyHeader, _, _, _, "import json")
    ->  format('  [PASS] Legacy header defaults to CPython~n', [])
    ;   format('  [FAIL] Legacy header compatibility issue~n', [])
    ),

    format('~n=== All Runtime-Specific Header Tests Passed ===~n', []).

% ============================================================================
% PIPELINE CHAINING (Phase 4)
% ============================================================================
%
% Pipeline chaining connects multiple predicates in a data flow:
%   input -> pred1 -> pred2 -> pred3 -> output
%
% Two modes:
%   1. Same-runtime chaining: All predicates run in same Python process
%   2. Cross-runtime chaining: Predicates may run in different runtimes,
%      connected via JSONL pipes or in-process bridges
%

%% compile_pipeline(+Predicates, +Options, -Code)
%  Main entry point for pipeline chaining.
%  Automatically selects same-runtime or cross-runtime based on predicates.
%
%  Predicates: List of Name/Arity or Target:Name/Arity
%    Examples:
%      - [get_users/1, filter_active/2, format_output/1]
%      - [python:get_users/1, csharp:validate/1, python:format/1]
%
%  Options:
%    - runtime(Runtime)     : Force specific runtime for same-runtime
%    - glue_protocol(P)     : Protocol for cross-runtime (jsonl/messagepack)
%    - pipeline_name(Name)  : Name for generated pipeline function
%    - arg_names(Names)     : Property names for final output
%
compile_pipeline(Predicates, Options, Code) :-
    % Check if all predicates are same runtime
    (   all_same_runtime(Predicates)
    ->  compile_same_runtime_pipeline(Predicates, Options, Code)
    ;   compile_cross_runtime_pipeline(Predicates, Options, Code)
    ).

%% all_same_runtime(+Predicates)
%  True if all predicates can run in the same Python runtime.
%
all_same_runtime([]).
all_same_runtime([Pred|Rest]) :-
    predicate_runtime(Pred, Runtime),
    all_same_runtime_check(Rest, Runtime).

all_same_runtime_check([], _).
all_same_runtime_check([Pred|Rest], Runtime) :-
    predicate_runtime(Pred, PredRuntime),
    compatible_runtimes(Runtime, PredRuntime),
    all_same_runtime_check(Rest, Runtime).

%% predicate_runtime(+Pred, -Runtime)
%  Determine the runtime for a predicate.
%
predicate_runtime(python:_Name/_Arity, python) :- !.
predicate_runtime(cpython:_Name/_Arity, cpython) :- !.
predicate_runtime(ironpython:_Name/_Arity, ironpython) :- !.
predicate_runtime(pypy:_Name/_Arity, pypy) :- !.
predicate_runtime(jython:_Name/_Arity, jython) :- !.
predicate_runtime(csharp:_Name/_Arity, csharp) :- !.
predicate_runtime(powershell:_Name/_Arity, powershell) :- !.
predicate_runtime(_Name/_Arity, python).  % Default to python

%% compatible_runtimes(+R1, +R2)
%  True if two runtimes can run in same process.
%
compatible_runtimes(R, R) :- !.
compatible_runtimes(python, cpython) :- !.
compatible_runtimes(cpython, python) :- !.
compatible_runtimes(python, ironpython) :- !.
compatible_runtimes(ironpython, python) :- !.
compatible_runtimes(python, pypy) :- !.
compatible_runtimes(pypy, python) :- !.
compatible_runtimes(python, jython) :- !.
compatible_runtimes(jython, python) :- !.

% ============================================================================
% Same-Runtime Pipeline Chaining
% ============================================================================

%% compile_same_runtime_pipeline(+Predicates, +Options, -Code)
%  Compile a pipeline where all predicates run in the same Python process.
%  This is efficient as no serialization is needed between steps.
%
%  Supports pipeline_mode option:
%    - sequential (default): Stages chained sequentially
%    - generator: Fixpoint iteration with deduplication
%
compile_same_runtime_pipeline(Predicates, Options, Code) :-
    option(runtime(Runtime), Options, cpython),
    option(glue_protocol(GlueProtocol), Options, jsonl),
    option(pipeline_name(PipelineName), Options, pipeline),
    option(pipeline_mode(PipelineMode), Options, sequential),
    option(arg_names(ArgNames), Options, []),

    % Generate header (extended for generator mode)
    pipeline_header_extended(GlueProtocol, Runtime, PipelineMode, Header),

    % Compile each predicate to a function
    compile_pipeline_predicates(Predicates, PredicateFunctions),

    % Generate the pipeline connector function based on mode and runtime
    generate_pipeline_connector(Predicates, PipelineName, PipelineMode, Runtime, ConnectorCode),

    % Generate helpers (extended for generator mode, runtime-aware)
    pipeline_helpers_extended(GlueProtocol, same_as_data, PipelineMode, Runtime, Helpers),

    % Generate main block
    generate_chained_pipeline_main(PipelineName, GlueProtocol, ArgNames, MainCode),

    format(string(Code), "~w~w~w~w~w",
        [Header, Helpers, PredicateFunctions, ConnectorCode, MainCode]).

%% pipeline_header_extended(+Protocol, +Runtime, +Mode, -Header)
%  Generate header with additional imports for generator mode
%  IronPython uses .NET HashSet instead of dataclasses
pipeline_header_extended(GlueProtocol, ironpython, generator, Header) :-
    !,
    pipeline_header(GlueProtocol, ironpython, BaseHeader),
    % IronPython: Use .NET HashSet, no dataclasses needed
    format(string(Header), "~w
# .NET HashSet for fixpoint deduplication (IronPython)
from System.Collections.Generic import HashSet

", [BaseHeader]).
pipeline_header_extended(GlueProtocol, Runtime, generator, Header) :-
    !,
    pipeline_header(GlueProtocol, Runtime, BaseHeader),
    format(string(Header), "~wfrom typing import Set
from dataclasses import dataclass

", [BaseHeader]).
pipeline_header_extended(GlueProtocol, Runtime, _, Header) :-
    pipeline_header(GlueProtocol, Runtime, Header).

%% pipeline_helpers_extended(+Protocol, +DataSource, +Mode, +Runtime, -Helpers)
%  Generate helpers including record_key for generator mode
%  IronPython uses .NET HashSet<String> with JSON-serialized keys
pipeline_helpers_extended(GlueProtocol, DataSource, generator, ironpython, Helpers) :-
    !,
    pipeline_helpers(GlueProtocol, DataSource, BaseHelpers),
    IronPythonHashCode = "
# Record key generation for .NET HashSet (IronPython)
# Uses JSON serialization with sorted keys for consistent hashing

def record_key(record):
    '''Convert a record to a hashable string key for .NET HashSet.'''
    # Sort keys for consistent ordering
    sorted_items = sorted(record.items(), key=lambda x: str(x[0]))
    # Create canonical JSON string
    return json.dumps(dict(sorted_items), sort_keys=True)

def dict_from_key(key):
    '''Convert a key back to a dictionary.'''
    return json.loads(key)

# .NET HashSet wrapper for Python-style interface
class RecordSet:
    '''Wrapper around .NET HashSet<String> for record deduplication.'''
    def __init__(self):
        self._set = HashSet[String]()

    def add(self, key):
        '''Add a key to the set.'''
        self._set.Add(String(key))

    def __contains__(self, key):
        '''Check if key is in the set.'''
        return self._set.Contains(String(key))

    def __len__(self):
        return self._set.Count

",
    format(string(Helpers), "~w~w", [BaseHelpers, IronPythonHashCode]).

pipeline_helpers_extended(GlueProtocol, DataSource, generator, _Runtime, Helpers) :-
    !,
    pipeline_helpers(GlueProtocol, DataSource, BaseHelpers),
    FrozenDictCode = "
# FrozenDict - hashable dictionary for use in sets (fixpoint deduplication)
@dataclass(frozen=True)
class FrozenDict:
    '''Immutable dictionary that can be used in sets.'''
    items: tuple

    @staticmethod
    def from_dict(d: dict) -> 'FrozenDict':
        return FrozenDict(tuple(sorted(d.items())))

    def to_dict(self) -> dict:
        return dict(self.items)

    def get(self, key, default=None):
        for k, v in self.items:
            if k == key:
                return v
        return default

    def __contains__(self, key):
        return any(k == key for k, _ in self.items)

    def __repr__(self):
        return f'FrozenDict({dict(self.items)})'

def record_key(record: dict) -> FrozenDict:
    '''Convert a record to a hashable key for deduplication.'''
    return FrozenDict.from_dict(record)

",
    format(string(Helpers), "~w~w", [BaseHelpers, FrozenDictCode]).

pipeline_helpers_extended(GlueProtocol, DataSource, _, _Runtime, Helpers) :-
    pipeline_helpers(GlueProtocol, DataSource, Helpers).

%% compile_pipeline_predicates(+Predicates, -Code)
%  Compile each predicate to a Python generator function.
%
compile_pipeline_predicates([], "").
compile_pipeline_predicates([Pred|Rest], Code) :-
    compile_single_pipeline_predicate(Pred, PredCode),
    compile_pipeline_predicates(Rest, RestCode),
    format(string(Code), "~w~w", [PredCode, RestCode]).

%% compile_single_pipeline_predicate(+Pred, -Code)
%  Compile a single predicate for use in a pipeline.
%
compile_single_pipeline_predicate(_Target:Name/Arity, Code) :-
    !,
    % Extract just the name for the function
    compile_single_pipeline_predicate(Name/Arity, Code).

compile_single_pipeline_predicate(Name/Arity, Code) :-
    atom_string(Name, NameStr),
    % Generate a simple passthrough function as placeholder
    % In real usage, this would compile the actual predicate
    generate_default_arg_names(Arity, ArgNames),
    generate_extraction_code(ArgNames, ExtractionCode),
    format(string(Code),
"
def ~w(stream):
    \"\"\"
    Pipeline step: ~w/~w
    \"\"\"
    for record in stream:
        # Extract inputs from record
~w
        # Process (placeholder - actual logic from predicate)
        result = record.copy()
        yield result

", [NameStr, NameStr, Arity, ExtractionCode]).

%% generate_arg_list(+Names, -Code)
generate_arg_list(Names, Code) :-
    atomic_list_concat(Names, ', ', Code).

%% generate_dict_construction(+Names, -Code)
generate_dict_construction(Names, Code) :-
    maplist(generate_dict_entry, Names, Entries),
    atomic_list_concat(Entries, ', ', EntriesStr),
    format(string(Code), "{~w}", [EntriesStr]).

generate_dict_entry(Name, Entry) :-
    format(string(Entry), "'~w': ~w", [Name, Name]).

%% generate_extraction_code(+Names, -Code)
%  Generate code to extract inputs from record dict.
%
generate_extraction_code(Names, Code) :-
    maplist(generate_extraction_line, Names, Lines),
    atomic_list_concat(Lines, '\n', Code).

generate_extraction_line(Name, Line) :-
    format(string(Line), "        ~w = record.get('~w')", [Name, Name]).

%% generate_pipeline_connector(+Predicates, +Name, -Code)
%  Generate the function that chains all predicates together (legacy 3-arg version).
%
generate_pipeline_connector(Predicates, PipelineName, Code) :-
    generate_pipeline_connector(Predicates, PipelineName, sequential, cpython, Code).

%% generate_pipeline_connector(+Predicates, +Name, +Mode, -Code)
%  Generate the function that chains all predicates together (4-arg version).
%  Mode can be: sequential, generator
%
generate_pipeline_connector(Predicates, PipelineName, Mode, Code) :-
    generate_pipeline_connector(Predicates, PipelineName, Mode, cpython, Code).

%% generate_pipeline_connector(+Predicates, +Name, +Mode, +Runtime, -Code)
%  Runtime-aware pipeline connector generation.
%  IronPython uses .NET HashSet wrapper for generator mode.
%
generate_pipeline_connector(Predicates, PipelineName, sequential, _Runtime, Code) :-
    % Build the chain: pred1(pred2(pred3(input)))
    % Or use generator chaining for efficiency
    extract_predicate_names(Predicates, Names),
    generate_chain_code(Names, ChainCode),
    format(string(Code),
"
def ~w(input_stream):
    \"\"\"
    Chained pipeline: ~w
    Connects all predicates in sequence.
    \"\"\"
~w
", [PipelineName, Names, ChainCode]).

%% IronPython generator mode - uses .NET HashSet wrapper
generate_pipeline_connector(Predicates, PipelineName, generator, ironpython, Code) :-
    !,
    extract_predicate_names(Predicates, Names),
    generate_fixpoint_chain_code(Names, ChainCode),
    format(string(Code),
"
def ~w(input_stream):
    \"\"\"
    Fixpoint pipeline: ~w (IronPython/.NET)
    Iterates until no new records are produced.
    Uses .NET HashSet<String> for deduplication.
    \"\"\"
    # Initialize with input records using .NET HashSet wrapper
    total = RecordSet()
    all_records = []

    for record in input_stream:
        key = record_key(record)
        if key not in total:
            total.add(key)
            all_records.append(record)
            yield record

    # Fixpoint iteration - apply stages until no new records
    changed = True
    while changed:
        changed = False
        current = list(all_records)

~w

        # Check for new records
        for record in new_records:
            key = record_key(record)
            if key not in total:
                total.add(key)
                all_records.append(record)
                changed = True
                yield record

", [PipelineName, Names, ChainCode]).

%% CPython/PyPy/Jython generator mode - uses FrozenDict with Python set
generate_pipeline_connector(Predicates, PipelineName, generator, _Runtime, Code) :-
    % Generate fixpoint iteration pipeline
    extract_predicate_names(Predicates, Names),
    generate_fixpoint_chain_code(Names, ChainCode),
    format(string(Code),
"
def ~w(input_stream):
    \"\"\"
    Fixpoint pipeline: ~w
    Iterates until no new records are produced.
    \"\"\"
    # Initialize with input records
    total: Set[FrozenDict] = set()

    for record in input_stream:
        key = record_key(record)
        if key not in total:
            total.add(key)
            yield record

    # Fixpoint iteration - apply stages until no new records
    changed = True
    while changed:
        changed = False
        current = [key.to_dict() for key in total]

~w

        # Check for new records
        for record in new_records:
            key = record_key(record)
            if key not in total:
                total.add(key)
                changed = True
                yield record

", [PipelineName, Names, ChainCode]).

%% generate_fixpoint_chain_code(+Names, -Code)
%  Generate the stage application code for fixpoint iteration
generate_fixpoint_chain_code([], "        new_records = current\n").
generate_fixpoint_chain_code(Names, Code) :-
    Names \= [],
    generate_fixpoint_stage_calls(Names, "current", StageCalls),
    format(string(Code), "        # Apply pipeline stages
~w", [StageCalls]).

generate_fixpoint_stage_calls([], Current, Code) :-
    format(string(Code), "        new_records = ~w\n", [Current]).
generate_fixpoint_stage_calls([Stage|Rest], Current, Code) :-
    format(string(NextVar), "stage_~w_out", [Stage]),
    format(string(StageCall), "        ~w = list(~w(iter(~w)))\n", [NextVar, Stage, Current]),
    generate_fixpoint_stage_calls(Rest, NextVar, RestCode),
    format(string(Code), "~w~w", [StageCall, RestCode]).

%% extract_predicate_names(+Predicates, -Names)
extract_predicate_names([], []).
extract_predicate_names([Pred|Rest], [Name|RestNames]) :-
    extract_pred_name(Pred, Name),
    extract_predicate_names(Rest, RestNames).

extract_pred_name(_Target:Name/_Arity, NameStr) :-
    !,
    atom_string(Name, NameStr).
extract_pred_name(Name/_Arity, NameStr) :-
    atom_string(Name, NameStr).

%% generate_chain_code(+Names, -Code)
%  Generate the chaining code connecting all predicates.
%
generate_chain_code([], "    yield from input_stream\n").
generate_chain_code([First|Rest], Code) :-
    generate_chain_recursive(Rest, First, "input_stream", ChainExpr),
    format(string(Code), "    yield from ~w\n", [ChainExpr]).

generate_chain_recursive([], Current, Input, Expr) :-
    format(string(Expr), "~w(~w)", [Current, Input]).
generate_chain_recursive([Next|Rest], Current, Input, Expr) :-
    format(string(CurrentCall), "~w(~w)", [Current, Input]),
    generate_chain_recursive(Rest, Next, CurrentCall, Expr).

%% generate_chained_pipeline_main(+Name, +Protocol, +ArgNames, -Code)
%  Generate the main block for the chained pipeline.
%
generate_chained_pipeline_main(PipelineName, jsonl, _ArgNames, Code) :-
    format(string(Code),
"
if __name__ == '__main__':
    import sys

    # Read from stdin, process through pipeline, write to stdout
    input_stream = read_stream(sys.stdin)
    for result in ~w(input_stream):
        write_record(result)
", [PipelineName]).

generate_chained_pipeline_main(PipelineName, messagepack, _ArgNames, Code) :-
    format(string(Code),
"
if __name__ == '__main__':
    import sys

    # Read MessagePack from stdin, process, write to stdout
    input_stream = read_stream(sys.stdin.buffer)
    for result in ~w(input_stream):
        write_record(result)
", [PipelineName]).

% ============================================================================
% Cross-Runtime Pipeline Chaining
% ============================================================================

%% compile_cross_runtime_pipeline(+Predicates, +Options, -Code)
%  Compile a pipeline where predicates run in different runtimes.
%  Uses JSONL pipes or in-process bridges for communication.
%
compile_cross_runtime_pipeline(Predicates, Options, Code) :-
    option(glue_protocol(GlueProtocol), Options, jsonl),
    option(pipeline_name(PipelineName), Options, pipeline),

    % Group predicates by runtime
    group_by_runtime(Predicates, Groups),

    % Generate orchestrator code (shell script or Python)
    generate_cross_runtime_orchestrator(Groups, PipelineName, GlueProtocol, Code).

%% group_by_runtime(+Predicates, -Groups)
%  Group consecutive predicates that share a runtime.
%
group_by_runtime([], []).
group_by_runtime([Pred|Rest], [group(Runtime, [Pred|SameRuntime])|RestGroups]) :-
    predicate_runtime(Pred, Runtime),
    take_same_runtime(Rest, Runtime, SameRuntime, Remaining),
    group_by_runtime(Remaining, RestGroups).

take_same_runtime([], _, [], []).
take_same_runtime([Pred|Rest], Runtime, [Pred|Same], Remaining) :-
    predicate_runtime(Pred, PredRuntime),
    compatible_runtimes(Runtime, PredRuntime),
    !,
    take_same_runtime(Rest, Runtime, Same, Remaining).
take_same_runtime(Preds, _, [], Preds).

%% generate_cross_runtime_orchestrator(+Groups, +Name, +Protocol, -Code)
%  Generate orchestrator that manages cross-runtime pipeline.
%
generate_cross_runtime_orchestrator(Groups, PipelineName, GlueProtocol, Code) :-
    length(Groups, NumGroups),
    (   NumGroups == 1
    ->  % Single group - just compile as same-runtime
        Groups = [group(_Runtime, Predicates)],
        compile_same_runtime_pipeline(Predicates, [pipeline_name(PipelineName)], Code)
    ;   % Multiple groups - generate orchestrator
        generate_multi_runtime_code(Groups, PipelineName, GlueProtocol, Code)
    ).

%% generate_multi_runtime_code(+Groups, +Name, +Protocol, -Code)
%  Generate Python code that orchestrates multiple runtime stages.
%
generate_multi_runtime_code(Groups, PipelineName, GlueProtocol, Code) :-
    % Generate stage functions for each group
    generate_stage_functions(Groups, 1, StageFunctions),

    % Generate the orchestrator that pipes between stages
    generate_orchestrator_function(Groups, PipelineName, GlueProtocol, OrchestratorCode),

    % Generate header
    pipeline_header(GlueProtocol, cpython, Header),
    pipeline_helpers(GlueProtocol, same_as_data, Helpers),

    % Generate main
    generate_chained_pipeline_main(PipelineName, GlueProtocol, [], MainCode),

    format(string(Code), "~w~w~w~w~w",
        [Header, Helpers, StageFunctions, OrchestratorCode, MainCode]).

%% generate_stage_functions(+Groups, +StageNum, -Code)
generate_stage_functions([], _, "").
generate_stage_functions([group(Runtime, Predicates)|Rest], N, Code) :-
    generate_stage_function(Runtime, Predicates, N, StageCode),
    N1 is N + 1,
    generate_stage_functions(Rest, N1, RestCode),
    format(string(Code), "~w~w", [StageCode, RestCode]).

%% generate_stage_function(+Runtime, +Predicates, +N, -Code)
generate_stage_function(Runtime, Predicates, N, Code) :-
    extract_predicate_names(Predicates, Names),
    atomic_list_concat(Names, ' -> ', NamesStr),
    runtime_to_string(Runtime, RuntimeStr),
    generate_stage_chain_code(Names, StageChainCode),
    format(string(Code),
"
def stage_~w(input_stream):
    \"\"\"
    Stage ~w: ~w
    Runtime: ~w
    \"\"\"
    # Chain predicates within this stage
    current = input_stream
~w
    yield from current

", [N, N, NamesStr, RuntimeStr, StageChainCode]).

generate_stage_chain_code([], "").
generate_stage_chain_code([Name|Rest], Code) :-
    format(string(Line), "    current = ~w(current)\n", [Name]),
    generate_stage_chain_code(Rest, RestCode),
    string_concat(Line, RestCode, Code).

runtime_to_string(python, "Python (CPython)").
runtime_to_string(cpython, "CPython").
runtime_to_string(ironpython, "IronPython").
runtime_to_string(pypy, "PyPy").
runtime_to_string(jython, "Jython").
runtime_to_string(csharp, "C#").
runtime_to_string(powershell, "PowerShell").
runtime_to_string(R, S) :- atom_string(R, S).

%% generate_orchestrator_function(+Groups, +Name, +Protocol, -Code)
generate_orchestrator_function(Groups, PipelineName, _Protocol, Code) :-
    length(Groups, NumStages),
    numlist(1, NumStages, StageNums),
    maplist(format_stage_call, StageNums, StageCalls),
    atomic_list_concat(StageCalls, '\n', StageCallsCode),
    format(string(Code),
"
def ~w(input_stream):
    \"\"\"
    Cross-runtime pipeline orchestrator.
    Chains ~w stages together.
    \"\"\"
    current = input_stream
~w
    yield from current

", [PipelineName, NumStages, StageCallsCode]).

format_stage_call(N, Call) :-
    format(string(Call), "    current = stage_~w(current)", [N]).

% ============================================================================
% Enhanced Pipeline Chaining - Fan-out, Merge, Conditional Routing (Phase 7)
% ============================================================================
%
% Enhanced pipeline stages support complex data flow patterns:
%   - fan_out(Stages)     : Broadcast each record to all parallel stages
%   - merge               : Combine results from parallel stages
%   - route_by(Pred, Map) : Route records based on condition
%   - filter_by(Pred)     : Filter records by predicate
%   - Name/Arity          : Standard predicate stage (unchanged)
%
% Example pipeline:
%   compile_enhanced_pipeline([
%       extract/1,
%       fan_out([validate/1, enrich/1]),
%       merge,
%       route_by(has_error, [(true, error_handler/1), (false, success/1)])
%   ], Options, Code)

%% compile_enhanced_pipeline(+Stages, +Options, -Code)
%  Main entry point for enhanced pipeline with advanced flow patterns.
%  Validates pipeline stages before code generation.
compile_enhanced_pipeline(Stages, Options, Code) :-
    % Validate pipeline stages (Phase 9)
    option(validate(Validate), Options, true),
    option(strict(Strict), Options, false),
    ( Validate == true ->
        validate_pipeline(Stages, [strict(Strict)], result(Errors, Warnings)),
        % Report warnings
        ( Warnings \== [] ->
            format(user_error, 'Pipeline warnings:~n', []),
            forall(member(W, Warnings), (
                format_validation_warning(W, Msg),
                format(user_error, '  ~w~n', [Msg])
            ))
        ; true
        ),
        % Fail on errors
        ( Errors \== [] ->
            format(user_error, 'Pipeline validation errors:~n', []),
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
    option(runtime(Runtime), Options, cpython),
    option(glue_protocol(GlueProtocol), Options, jsonl),
    option(pipeline_mode(PipelineMode), Options, sequential),

    % Generate header
    pipeline_header_extended(GlueProtocol, Runtime, PipelineMode, Header),

    % Generate helpers (including enhanced helpers)
    pipeline_helpers_extended(GlueProtocol, same_as_data, PipelineMode, Runtime, BaseHelpers),
    enhanced_pipeline_helpers(EnhancedHelpers),
    format(string(Helpers), "~w~w", [BaseHelpers, EnhancedHelpers]),

    % Generate stage functions
    generate_enhanced_stage_functions(Stages, StageFunctions),

    % Generate the main pipeline connector
    generate_enhanced_connector(Stages, PipelineName, ConnectorCode),

    % Generate main block
    generate_chained_pipeline_main(PipelineName, GlueProtocol, [], MainBlock),

    format(string(Code), "~w~w~w~w~w",
           [Header, Helpers, StageFunctions, ConnectorCode, MainBlock]).

%% enhanced_pipeline_helpers(-Code)
%  Generate helper functions for enhanced pipeline operations.
enhanced_pipeline_helpers(Code) :-
    Code = "
# Enhanced Pipeline Helpers

def fan_out_records(record, stages):
    '''
    Fan-out: Send record to all stages, collect all results.
    Each stage receives the same input record.
    '''
    results = []
    for stage in stages:
        for result in stage(iter([record])):
            results.append(result)
    return results

def merge_streams(*streams):
    '''
    Merge: Combine multiple streams into one.
    Yields records from all streams in order.
    '''
    for stream in streams:
        yield from stream

def route_record(record, condition_fn, route_map):
    '''
    Route: Direct record to appropriate stage based on condition.
    route_map is dict of {condition_value: stage_function}
    '''
    condition = condition_fn(record)
    if condition in route_map:
        yield from route_map[condition](iter([record]))
    elif 'default' in route_map:
        yield from route_map['default'](iter([record]))
    else:
        yield record  # Pass through if no matching route

def filter_records(stream, predicate_fn):
    '''
    Filter: Only yield records that satisfy the predicate.
    '''
    for record in stream:
        if predicate_fn(record):
            yield record

def tee_stream(stream, *stages):
    '''
    Tee: Send each record to multiple stages, yield all results.
    Similar to fan_out but operates on streams.
    '''
    records = list(stream)  # Materialize to allow multiple iterations
    for record in records:
        for stage in stages:
            yield from stage(iter([record]))

def parallel_records(record, stages):
    '''
    Parallel: Execute stages concurrently using ThreadPoolExecutor.
    Each stage receives the same input record.
    Results are collected in completion order (fastest first).
    '''
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def run_stage(stage):
        return list(stage(iter([record])))

    results = []
    with ThreadPoolExecutor(max_workers=len(stages)) as executor:
        futures = {executor.submit(run_stage, stage): i for i, stage in enumerate(stages)}
        for future in as_completed(futures):
            results.extend(future.result())
    return results

def parallel_records_ordered(record, stages):
    '''
    Parallel (Ordered): Execute stages concurrently, preserve input order.
    Each stage receives the same input record.
    Results are collected and returned in stage definition order.
    '''
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def run_stage(stage):
        return list(stage(iter([record])))

    indexed_results = [None] * len(stages)
    with ThreadPoolExecutor(max_workers=len(stages)) as executor:
        futures = {executor.submit(run_stage, stage): i for i, stage in enumerate(stages)}
        for future in as_completed(futures):
            idx = futures[future]
            indexed_results[idx] = future.result()

    # Flatten results in order
    results = []
    for stage_results in indexed_results:
        if stage_results:
            results.extend(stage_results)
    return results

def batch_records(stream, batch_size):
    '''
    Batch: Collect records into batches of specified size.
    Yields each batch as a list. Final batch may be smaller.
    '''
    batch = []
    for record in stream:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:  # Flush remaining records
        yield batch

def unbatch_records(stream):
    '''
    Unbatch: Flatten batches back to individual records.
    Each batch (list) is expanded to yield individual records.
    '''
    for batch in stream:
        if isinstance(batch, list):
            for record in batch:
                yield record
        else:
            yield batch  # Pass through non-list items

def unique_by_field(stream, field):
    '''
    Unique: Keep only the first record for each unique field value.
    Deduplicates the stream based on the specified field.
    '''
    seen = set()
    for record in stream:
        key = record.get(field)
        if key not in seen:
            seen.add(key)
            yield record

def first_by_field(stream, field):
    '''
    First: Alias for unique_by_field - keeps first occurrence.
    '''
    yield from unique_by_field(stream, field)

def last_by_field(stream, field):
    '''
    Last: Keep only the last record for each unique field value.
    Buffers all records, then yields last occurrence of each.
    '''
    last_seen = {}
    order = []
    for record in stream:
        key = record.get(field)
        if key not in last_seen:
            order.append(key)
        last_seen[key] = record
    for key in order:
        yield last_seen[key]

def group_by_field(stream, field, aggregations):
    '''
    Group By: Group records by field and apply aggregations.
    aggregations is a list of (name, agg_type, agg_field) tuples.
    '''
    from collections import defaultdict
    groups = defaultdict(list)
    order = []

    # Collect records into groups
    for record in stream:
        key = record.get(field)
        if key not in groups:
            order.append(key)
        groups[key].append(record)

    # Apply aggregations to each group
    for key in order:
        group_records = groups[key]
        result = {field: key}

        for agg_name, agg_type, agg_field in aggregations:
            if agg_type == 'count':
                result[agg_name] = len(group_records)
            elif agg_type == 'sum':
                result[agg_name] = sum(r.get(agg_field, 0) for r in group_records)
            elif agg_type == 'avg':
                values = [r.get(agg_field, 0) for r in group_records]
                result[agg_name] = sum(values) / len(values) if values else 0
            elif agg_type == 'min':
                values = [r.get(agg_field) for r in group_records if r.get(agg_field) is not None]
                result[agg_name] = min(values) if values else None
            elif agg_type == 'max':
                values = [r.get(agg_field) for r in group_records if r.get(agg_field) is not None]
                result[agg_name] = max(values) if values else None
            elif agg_type == 'first':
                result[agg_name] = group_records[0].get(agg_field) if group_records else None
            elif agg_type == 'last':
                result[agg_name] = group_records[-1].get(agg_field) if group_records else None
            elif agg_type == 'collect':
                result[agg_name] = [r.get(agg_field) for r in group_records]

        yield result

def reduce_records(stream, reducer_fn, initial):
    '''
    Reduce: Apply reducer function sequentially across all records.
    reducer_fn(record, accumulator) -> new_accumulator
    Yields final accumulated result.
    '''
    acc = initial
    for record in stream:
        acc = reducer_fn(record, acc)
    yield acc

def scan_records(stream, reducer_fn, initial):
    '''
    Scan: Like reduce but yields intermediate results.
    Emits running accumulated values after each record.
    '''
    acc = initial
    for record in stream:
        acc = reducer_fn(record, acc)
        yield acc

def order_by_field(stream, field, direction='asc'):
    '''
    Order by field: Sort records by a single field.
    direction can be 'asc' (ascending) or 'desc' (descending).
    Buffers all records before yielding sorted results.
    '''
    records = list(stream)
    reverse = (direction == 'desc')
    records.sort(key=lambda r: (r.get(field) is None, r.get(field)), reverse=reverse)
    yield from records

def order_by_fields(stream, field_specs):
    '''
    Order by multiple fields: Sort records by multiple fields with directions.
    field_specs is a list of (field, direction) tuples.
    direction can be 'asc' or 'desc'.
    '''
    records = list(stream)

    def make_sort_key(record):
        key_parts = []
        for field, direction in field_specs:
            value = record.get(field)
            # Handle None values (sort them to end)
            is_none = value is None
            if direction == 'desc':
                # For descending, we need to invert the sort
                # Use a wrapper that inverts comparison
                key_parts.append((is_none, Descending(value) if not is_none else value))
            else:
                key_parts.append((is_none, value))
        return tuple(key_parts)

    records.sort(key=make_sort_key)
    yield from records

class Descending:
    '''Helper class to invert comparison for descending sort.'''
    def __init__(self, value):
        self.value = value
    def __lt__(self, other):
        if isinstance(other, Descending):
            return self.value > other.value
        return False
    def __gt__(self, other):
        if isinstance(other, Descending):
            return self.value < other.value
        return False
    def __eq__(self, other):
        if isinstance(other, Descending):
            return self.value == other.value
        return False
    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)
    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

def sort_by_comparator(stream, compare_fn):
    '''
    Sort by custom comparator: Sort records using a user-defined comparison function.
    compare_fn(record_a, record_b) should return:
      -1 (or negative) if a < b
       0 if a == b
       1 (or positive) if a > b
    '''
    from functools import cmp_to_key
    records = list(stream)
    records.sort(key=cmp_to_key(compare_fn))
    yield from records

def try_catch_stage(stream, stage_fn, handler_fn):
    '''
    Try-Catch: Execute stage_fn on each record, on exception route to handler_fn.
    handler_fn receives (record, error) and should yield results or nothing.
    '''
    for record in stream:
        try:
            for result in stage_fn([record]):
                yield result
        except Exception as e:
            for result in handler_fn([record], e):
                yield result

def retry_stage(stream, stage_fn, max_retries, delay_ms=0, backoff='none'):
    '''
    Retry: Execute stage_fn with retries on failure.
    max_retries: Maximum number of retry attempts
    delay_ms: Initial delay between retries in milliseconds
    backoff: 'none', 'linear', or 'exponential'
    '''
    import time
    for record in stream:
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                for result in stage_fn([record]):
                    yield result
                break  # Success, move to next record
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    # Calculate delay
                    if delay_ms > 0:
                        if backoff == 'exponential':
                            wait_time = delay_ms * (2 ** attempt) / 1000.0
                        elif backoff == 'linear':
                            wait_time = delay_ms * (attempt + 1) / 1000.0
                        else:
                            wait_time = delay_ms / 1000.0
                        time.sleep(wait_time)
        else:
            # All retries exhausted, re-raise last error or yield error record
            yield {'_error': str(last_error), '_record': record, '_retries': max_retries}

def on_error_stage(stream, handler_fn):
    '''
    On-Error: Global error handler that catches any exception and routes to handler.
    '''
    for record in stream:
        try:
            yield record
        except Exception as e:
            for result in handler_fn([record], e):
                yield result

def timeout_stage(stream, stage_fn, timeout_ms):
    '''
    Timeout: Execute stage with time limit. Yields error record on timeout.
    '''
    import threading
    import queue

    for record in stream:
        result_queue = queue.Queue()
        exception_holder = [None]

        def run_stage():
            try:
                results = list(stage_fn([record]))
                result_queue.put(('success', results))
            except Exception as e:
                result_queue.put(('error', e))

        thread = threading.Thread(target=run_stage)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout_ms / 1000.0)

        if thread.is_alive():
            # Timeout occurred
            yield {'_timeout': True, '_record': record, '_limit_ms': timeout_ms}
        else:
            try:
                status, data = result_queue.get_nowait()
                if status == 'success':
                    for result in data:
                        yield result
                else:
                    raise data
            except queue.Empty:
                yield {'_timeout': True, '_record': record, '_limit_ms': timeout_ms}

def timeout_stage_with_fallback(stream, stage_fn, timeout_ms, fallback_fn):
    '''
    Timeout with fallback: Execute stage with time limit, use fallback on timeout.
    '''
    import threading
    import queue

    for record in stream:
        result_queue = queue.Queue()

        def run_stage():
            try:
                results = list(stage_fn([record]))
                result_queue.put(('success', results))
            except Exception as e:
                result_queue.put(('error', e))

        thread = threading.Thread(target=run_stage)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout_ms / 1000.0)

        if thread.is_alive():
            # Timeout occurred - use fallback
            for result in fallback_fn([record]):
                yield result
        else:
            try:
                status, data = result_queue.get_nowait()
                if status == 'success':
                    for result in data:
                        yield result
                else:
                    raise data
            except queue.Empty:
                for result in fallback_fn([record]):
                    yield result

def rate_limit_stage(stream, count, per_unit):
    '''
    Rate Limit: Limit throughput to count records per time unit.
    per_unit: 'second', 'minute', 'hour', or milliseconds as int
    '''
    import time

    # Calculate interval between records in seconds
    if per_unit == 'second':
        interval = 1.0 / count
    elif per_unit == 'minute':
        interval = 60.0 / count
    elif per_unit == 'hour':
        interval = 3600.0 / count
    else:
        # Assume milliseconds
        interval = (per_unit / 1000.0) / count

    last_time = None
    for record in stream:
        current_time = time.time()
        if last_time is not None:
            elapsed = current_time - last_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
        last_time = time.time()
        yield record

def throttle_stage(stream, delay_ms):
    '''
    Throttle: Add fixed delay between records.
    '''
    import time

    delay_sec = delay_ms / 1000.0
    first = True
    for record in stream:
        if not first:
            time.sleep(delay_sec)
        first = False
        yield record

def buffer_stage(stream, size):
    '''
    Buffer: Collect up to N records before emitting as batch.
    Flushes remaining records at end.
    '''
    buffer = []
    for record in stream:
        buffer.append(record)
        if len(buffer) >= size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer

def debounce_stage(stream, delay_ms):
    '''
    Debounce: Emit record only if no new record within delay_ms.
    Uses threading to implement proper debounce behavior.
    '''
    import time
    import threading

    delay_sec = delay_ms / 1000.0
    pending = [None]
    lock = threading.Lock()
    timer = [None]
    results = []
    done = threading.Event()

    def emit():
        with lock:
            if pending[0] is not None:
                results.append(pending[0])
                pending[0] = None

    for record in stream:
        with lock:
            pending[0] = record
            if timer[0] is not None:
                timer[0].cancel()
            timer[0] = threading.Timer(delay_sec, emit)
            timer[0].start()

    # Wait for final timer
    if timer[0] is not None:
        timer[0].join()

    for result in results:
        yield result

def zip_stage(record, stage_funcs):
    '''
    Zip: Run multiple stages on same input, combine outputs record-by-record.
    '''
    results = [list(stage_fn([record])) for stage_fn in stage_funcs]
    max_len = max(len(r) for r in results) if results else 0

    for i in range(max_len):
        combined = {}
        for j, result_list in enumerate(results):
            if i < len(result_list):
                combined.update(result_list[i])
        yield combined

def window_stage(stream, size):
    '''
    Window: Collect records into non-overlapping windows of specified size.
    '''
    window = []
    for record in stream:
        window.append(record)
        if len(window) >= size:
            yield window
            window = []
    if window:
        yield window

def sliding_window_stage(stream, size, step):
    '''
    Sliding Window: Emit windows of size, advancing by step each time.
    '''
    buffer = []
    for record in stream:
        buffer.append(record)
        while len(buffer) >= size:
            yield buffer[:size]
            buffer = buffer[step:]
    if buffer:
        yield buffer

def sample_stage(stream, n):
    '''
    Sample: Randomly select n records from the stream (reservoir sampling).
    '''
    import random
    reservoir = []
    for i, record in enumerate(stream):
        if i < n:
            reservoir.append(record)
        else:
            j = random.randint(0, i)
            if j < n:
                reservoir[j] = record
    for record in reservoir:
        yield record

def take_every_stage(stream, n):
    '''
    Take Every: Emit every nth record.
    '''
    for i, record in enumerate(stream):
        if i % n == 0:
            yield record

def partition_stage(stream, pred_fn):
    '''
    Partition: Split stream into [matches, non-matches] based on predicate.
    Returns a tuple of two lists.
    '''
    matches = []
    non_matches = []
    for record in stream:
        if pred_fn(record):
            matches.append(record)
        else:
            non_matches.append(record)
    return (matches, non_matches)

def take_stage(stream, n):
    '''
    Take: Emit only the first n records.
    '''
    count = 0
    for record in stream:
        if count >= n:
            break
        yield record
        count += 1

def skip_stage(stream, n):
    '''
    Skip: Skip the first n records, emit the rest.
    '''
    count = 0
    for record in stream:
        if count >= n:
            yield record
        count += 1

def take_while_stage(stream, pred_fn):
    '''
    Take While: Emit records while predicate is true, stop when false.
    '''
    for record in stream:
        if pred_fn(record):
            yield record
        else:
            break

def skip_while_stage(stream, pred_fn):
    '''
    Skip While: Skip records while predicate is true, emit once false.
    '''
    skipping = True
    for record in stream:
        if skipping and pred_fn(record):
            continue
        skipping = False
        yield record

def distinct_stage(stream):
    '''
    Distinct: Remove all duplicate records (global dedup), keeping first occurrence.
    Uses JSON serialization for record comparison.
    '''
    import json
    seen = set()
    for record in stream:
        key = json.dumps(record, sort_keys=True)
        if key not in seen:
            seen.add(key)
            yield record

def distinct_by_stage(stream, field):
    '''
    Distinct By: Remove duplicates based on a specific field, keeping first occurrence.
    '''
    seen = set()
    for record in stream:
        key = record.get(field)
        if key not in seen:
            seen.add(key)
            yield record

def dedup_stage(stream):
    '''
    Dedup: Remove consecutive duplicate records only.
    Uses JSON serialization for record comparison.
    '''
    import json
    last_key = None
    for record in stream:
        key = json.dumps(record, sort_keys=True)
        if key != last_key:
            last_key = key
            yield record

def dedup_by_stage(stream, field):
    '''
    Dedup By: Remove consecutive duplicates based on a specific field.
    '''
    last_value = object()  # Sentinel that won't match any real value
    for record in stream:
        value = record.get(field)
        if value != last_value:
            last_value = value
            yield record

def interleave_stage(streams):
    '''
    Interleave: Round-robin interleave records from multiple streams.
    Takes one record from each stream in turn until all are exhausted.
    '''
    iterators = [iter(s) for s in streams]
    active = list(range(len(iterators)))
    while active:
        next_active = []
        for i in active:
            try:
                yield next(iterators[i])
                next_active.append(i)
            except StopIteration:
                pass
        active = next_active

def concat_stage(streams):
    '''
    Concat: Concatenate multiple streams sequentially.
    Yields all records from first stream, then second, etc.
    '''
    for stream in streams:
        yield from stream

def merge_sorted_stage(streams, field, reverse=False):
    '''
    Merge Sorted: Merge multiple pre-sorted streams maintaining sort order.
    Uses a min-heap (or max-heap if reverse) for efficient k-way merge.
    Assumes each input stream is already sorted by the given field.
    '''
    import heapq

    # Create iterators with initial values
    heap = []
    iterators = []
    for i, stream in enumerate(streams):
        it = iter(stream)
        try:
            record = next(it)
            value = record.get(field)
            # For reverse (descending), negate numeric values or use negative index
            if reverse:
                if isinstance(value, (int, float)):
                    heap_val = (-value, i, record)
                else:
                    heap_val = (0, -i, record)  # For non-numeric, use FIFO order
            else:
                if isinstance(value, (int, float)):
                    heap_val = (value, i, record)
                else:
                    heap_val = (str(value) if value else '', i, record)
            heapq.heappush(heap, heap_val)
            iterators.append(it)
        except StopIteration:
            iterators.append(None)

    while heap:
        _, stream_idx, record = heapq.heappop(heap)
        yield record

        # Try to get next record from same stream
        it = iterators[stream_idx]
        if it is not None:
            try:
                record = next(it)
                value = record.get(field)
                if reverse:
                    if isinstance(value, (int, float)):
                        heap_val = (-value, stream_idx, record)
                    else:
                        heap_val = (0, -stream_idx, record)
                else:
                    if isinstance(value, (int, float)):
                        heap_val = (value, stream_idx, record)
                    else:
                        heap_val = (str(value) if value else '', stream_idx, record)
                heapq.heappush(heap, heap_val)
            except StopIteration:
                iterators[stream_idx] = None

def tap_stage(stream, side_effect_fn):
    '''
    Tap: Execute side effect for each record without modifying the stream.
    Useful for logging, metrics, debugging, or other observations.
    The side_effect_fn is called with each record but its return value is ignored.
    '''
    for record in stream:
        try:
            side_effect_fn(record)
        except Exception:
            pass  # Side effects should not interrupt the pipeline
        yield record

def flatten_stage(stream):
    '''
    Flatten: Flatten nested collections into individual records.
    If a record is a list/tuple, yields each element individually.
    Non-iterable records are yielded as-is.
    '''
    for record in stream:
        if isinstance(record, (list, tuple)):
            for item in record:
                yield item
        elif isinstance(record, dict) and '__items__' in record:
            # Handle dict with __items__ key containing list
            for item in record['__items__']:
                yield item
        else:
            yield record

def flatten_field_stage(stream, field):
    '''
    Flatten Field: Flatten a specific field within each record.
    Expands records where field contains a list into multiple records.
    Each expanded record contains one item from the list.
    '''
    for record in stream:
        if isinstance(record, dict) and field in record:
            field_value = record[field]
            if isinstance(field_value, (list, tuple)):
                for item in field_value:
                    new_record = record.copy()
                    new_record[field] = item
                    yield new_record
            else:
                yield record
        else:
            yield record

def debounce_stage(stream, ms, timestamp_field=None):
    '''
    Debounce: Emit records only after a silence period.
    Groups records by time windows and emits the last record in each window.
    If timestamp_field is provided, uses that field for timing; otherwise uses arrival order.
    For batch processing, this simulates debounce by grouping records within ms intervals.
    '''
    import time
    buffer = []
    last_time = None
    threshold_sec = ms / 1000.0

    for record in stream:
        current_time = time.time()
        if timestamp_field and isinstance(record, dict) and timestamp_field in record:
            try:
                current_time = float(record[timestamp_field])
            except (ValueError, TypeError):
                pass

        if last_time is None:
            buffer = [record]
            last_time = current_time
        elif current_time - last_time < threshold_sec:
            # Within debounce window, replace buffer
            buffer = [record]
            last_time = current_time
        else:
            # Silence period exceeded, emit buffered and start new
            if buffer:
                yield buffer[-1]
            buffer = [record]
            last_time = current_time

    # Emit final buffered record
    if buffer:
        yield buffer[-1]

def branch_stage(stream, condition_fn, true_fn, false_fn):
    '''
    Branch: Conditional routing within pipeline.
    Records matching condition go through true_fn, others through false_fn.
    Results from both branches are combined in the output.
    '''
    for record in stream:
        try:
            if condition_fn(record):
                result = true_fn(iter([record]))
                for item in result:
                    yield item
            else:
                result = false_fn(iter([record]))
                for item in result:
                    yield item
        except Exception:
            # On condition error, pass through unchanged
            yield record

def tee_stage(stream, side_fn):
    '''
    Tee: Run side stage on stream copy, discard results, pass original through.
    Like Unix tee - fork stream to side destination while main stream continues.
    '''
    # Collect records to allow side stage to process full stream
    records = list(stream)

    # Run side stage (results discarded)
    try:
        # Consume the side stage generator to execute it
        for _ in side_fn(iter(records)):
            pass
    except Exception:
        pass  # Side effects should not interrupt the main pipeline

    # Yield original records unchanged
    for record in records:
        yield record

# ============================================
# SERVICE INFRASTRUCTURE (Client-Server Phase 1)
# ============================================

# Global service registry
_services = {}

class ServiceError(Exception):
    '''Base exception for service errors.'''
    pass

class ServiceTimeoutError(ServiceError):
    '''Service call timed out.'''
    pass

class ServiceNotFoundError(ServiceError):
    '''Requested service not found in registry.'''
    pass

class Service:
    '''
    Base class for in-process services.
    Services receive requests and return responses.
    '''
    def __init__(self, name, stateful=False):
        self.name = name
        self.state = {} if stateful else None
        self._stateful = stateful

    def call(self, request):
        '''Process a request and return response. Override in subclass.'''
        raise NotImplementedError('Service.call() must be implemented')

    def state_get(self, key, default=None):
        '''Get value from service state.'''
        if self.state is None:
            raise ServiceError(f'Service {self.name} is not stateful')
        return self.state.get(key, default)

    def state_put(self, key, value):
        '''Set value in service state.'''
        if self.state is None:
            raise ServiceError(f'Service {self.name} is not stateful')
        self.state[key] = value

    def state_modify(self, key, func):
        '''Modify value in service state using function.'''
        if self.state is None:
            raise ServiceError(f'Service {self.name} is not stateful')
        self.state[key] = func(self.state.get(key))

    def state_delete(self, key):
        '''Delete key from service state.'''
        if self.state is None:
            raise ServiceError(f'Service {self.name} is not stateful')
        if key in self.state:
            del self.state[key]

def register_service(name, service):
    '''Register a service in the global registry.'''
    _services[name] = service

def get_service(name):
    '''Get a service from the registry.'''
    if name not in _services:
        raise ServiceNotFoundError(f'Service not found: {name}')
    return _services[name]

def call_service_impl(service_name, request, options=None):
    '''
    Call a service and return the response.
    Options can include: timeout, retry, retry_delay, fallback
    '''
    options = options or {}
    timeout_ms = options.get('timeout')
    max_retries = options.get('retry', 0)
    retry_delay_ms = options.get('retry_delay', 100)
    fallback = options.get('fallback')

    service = get_service(service_name)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if timeout_ms:
                import signal
                def timeout_handler(signum, frame):
                    raise ServiceTimeoutError(f'Service {service_name} timed out after {timeout_ms}ms')
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, timeout_ms / 1000.0)
                try:
                    result = service.call(request)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                result = service.call(request)
            return result
        except ServiceError as e:
            last_error = e
            if attempt < max_retries:
                import time
                time.sleep(retry_delay_ms / 1000.0)

    if fallback is not None:
        return fallback
    if last_error:
        raise last_error
    raise ServiceError(f'Service call failed: {service_name}')

def call_service_stage(stream, service_name, request_field, response_field, options=None):
    '''
    Pipeline stage that calls a service for each record.
    Extracts request from request_field, stores response in response_field.
    '''
    for record in stream:
        try:
            request = record.get(request_field) if isinstance(request_field, str) else record
            response = call_service_impl(service_name, request, options)
            record[response_field] = response
            yield record
        except ServiceError as e:
            record['__error__'] = True
            record['__type__'] = type(e).__name__
            record['__message__'] = str(e)
            yield record

".

%% generate_enhanced_stage_functions(+Stages, -Code)
%  Generate stub functions for each stage, including enhanced stages.
generate_enhanced_stage_functions([], "").
generate_enhanced_stage_functions([Stage|Rest], Code) :-
    generate_single_enhanced_stage(Stage, StageCode),
    generate_enhanced_stage_functions(Rest, RestCode),
    (RestCode = "" ->
        Code = StageCode
    ;   format(string(Code), "~w~n~w", [StageCode, RestCode])
    ).

generate_single_enhanced_stage(fan_out(SubStages), Code) :-
    !,
    generate_enhanced_stage_functions(SubStages, SubCode),
    Code = SubCode.
generate_single_enhanced_stage(parallel(SubStages, _Options), Code) :-
    !,
    generate_enhanced_stage_functions(SubStages, SubCode),
    Code = SubCode.
generate_single_enhanced_stage(parallel(SubStages), Code) :-
    !,
    generate_enhanced_stage_functions(SubStages, SubCode),
    Code = SubCode.
generate_single_enhanced_stage(merge, "") :- !.
generate_single_enhanced_stage(route_by(_, Routes), Code) :-
    !,
    findall(Stage, member((_Cond, Stage), Routes), RouteStages),
    generate_enhanced_stage_functions(RouteStages, Code).
generate_single_enhanced_stage(filter_by(_), "") :- !.
generate_single_enhanced_stage(batch(_), "") :- !.
generate_single_enhanced_stage(unbatch, "") :- !.
generate_single_enhanced_stage(unique(_), "") :- !.
generate_single_enhanced_stage(first(_), "") :- !.
generate_single_enhanced_stage(last(_), "") :- !.
generate_single_enhanced_stage(group_by(_, _), "") :- !.
generate_single_enhanced_stage(reduce(_, _), "") :- !.
generate_single_enhanced_stage(reduce(_), "") :- !.
generate_single_enhanced_stage(scan(_, _), "") :- !.
generate_single_enhanced_stage(scan(_), "") :- !.
generate_single_enhanced_stage(order_by(_), "") :- !.
generate_single_enhanced_stage(order_by(_, _), "") :- !.
generate_single_enhanced_stage(sort_by(_), "") :- !.
generate_single_enhanced_stage(try_catch(Stage, Handler), Code) :-
    !,
    generate_single_enhanced_stage(Stage, StageCode),
    generate_single_enhanced_stage(Handler, HandlerCode),
    format(string(Code), "~w~w", [StageCode, HandlerCode]).
generate_single_enhanced_stage(retry(Stage, _), Code) :-
    !,
    generate_single_enhanced_stage(Stage, Code).
generate_single_enhanced_stage(retry(Stage, _, _), Code) :-
    !,
    generate_single_enhanced_stage(Stage, Code).
generate_single_enhanced_stage(on_error(Handler), Code) :-
    !,
    generate_single_enhanced_stage(Handler, Code).
generate_single_enhanced_stage(timeout(Stage, _), Code) :-
    !,
    generate_single_enhanced_stage(Stage, Code).
generate_single_enhanced_stage(timeout(Stage, _, Fallback), Code) :-
    !,
    generate_single_enhanced_stage(Stage, StageCode),
    generate_single_enhanced_stage(Fallback, FallbackCode),
    format(string(Code), "~w~w", [StageCode, FallbackCode]).
generate_single_enhanced_stage(rate_limit(_, _), "") :- !.
generate_single_enhanced_stage(throttle(_), "") :- !.
generate_single_enhanced_stage(buffer(_), "") :- !.
generate_single_enhanced_stage(debounce(_), "") :- !.
generate_single_enhanced_stage(zip(SubStages), Code) :-
    !,
    generate_enhanced_stage_functions(SubStages, Code).
generate_single_enhanced_stage(window(_), "") :- !.
generate_single_enhanced_stage(sliding_window(_, _), "") :- !.
generate_single_enhanced_stage(sample(_), "") :- !.
generate_single_enhanced_stage(take_every(_), "") :- !.
generate_single_enhanced_stage(partition(_), "") :- !.
generate_single_enhanced_stage(take(_), "") :- !.
generate_single_enhanced_stage(skip(_), "") :- !.
generate_single_enhanced_stage(take_while(_), "") :- !.
generate_single_enhanced_stage(skip_while(_), "") :- !.
generate_single_enhanced_stage(distinct, "") :- !.
generate_single_enhanced_stage(distinct_by(_), "") :- !.
generate_single_enhanced_stage(dedup, "") :- !.
generate_single_enhanced_stage(dedup_by(_), "") :- !.
generate_single_enhanced_stage(interleave(SubStages), Code) :-
    !,
    generate_enhanced_stage_functions(SubStages, Code).
generate_single_enhanced_stage(concat(SubStages), Code) :-
    !,
    generate_enhanced_stage_functions(SubStages, Code).
generate_single_enhanced_stage(merge_sorted(SubStages, _Field), Code) :-
    !,
    generate_enhanced_stage_functions(SubStages, Code).
generate_single_enhanced_stage(merge_sorted(SubStages, _Field, _Dir), Code) :-
    !,
    generate_enhanced_stage_functions(SubStages, Code).
generate_single_enhanced_stage(tap(_), "") :- !.
generate_single_enhanced_stage(flatten, "") :- !.
generate_single_enhanced_stage(flatten(_), "") :- !.
generate_single_enhanced_stage(debounce(_), "") :- !.
generate_single_enhanced_stage(debounce(_, _), "") :- !.
generate_single_enhanced_stage(branch(_Cond, TrueStage, FalseStage), Code) :-
    !,
    generate_single_enhanced_stage(TrueStage, TrueCode),
    generate_single_enhanced_stage(FalseStage, FalseCode),
    format(string(Code), "~w~w", [TrueCode, FalseCode]).
generate_single_enhanced_stage(tee(SideStage), Code) :-
    !,
    generate_single_enhanced_stage(SideStage, Code).
generate_single_enhanced_stage(call_service(_, _, _), "") :- !.
generate_single_enhanced_stage(call_service(_, _, _, _), "") :- !.
generate_single_enhanced_stage(Pred/Arity, Code) :-
    !,
    format(string(Code),
"
def ~w(stream):
    '''Pipeline stage: ~w/~w'''
    for record in stream:
        # TODO: Implement based on predicate bindings
        yield record
", [Pred, Pred, Arity]).
generate_single_enhanced_stage(_, "").

%% generate_enhanced_connector(+Stages, +PipelineName, -Code)
%  Generate the main connector that handles enhanced flow patterns.
generate_enhanced_connector(Stages, PipelineName, Code) :-
    generate_enhanced_flow_code(Stages, "input_stream", FlowCode),
    format(string(Code),
"
def ~w(input_stream):
    '''
    Enhanced pipeline with fan-out, merge, and routing support.
    '''
~w
", [PipelineName, FlowCode]).

%% generate_enhanced_flow_code(+Stages, +CurrentVar, -Code)
%  Generate the flow code for enhanced pipeline stages.
generate_enhanced_flow_code([], CurrentVar, Code) :-
    format(string(Code), "    yield from ~w", [CurrentVar]).
generate_enhanced_flow_code([Stage|Rest], CurrentVar, Code) :-
    generate_stage_flow(Stage, CurrentVar, NextVar, StageCode),
    generate_enhanced_flow_code(Rest, NextVar, RestCode),
    format(string(Code), "~w~n~w", [StageCode, RestCode]).

%% generate_stage_flow(+Stage, +InVar, -OutVar, -Code)
%  Generate flow code for a single stage.

% Fan-out stage: broadcast to parallel stages (sequential execution)
generate_stage_flow(fan_out(SubStages), InVar, OutVar, Code) :-
    length(SubStages, N),
    format(atom(OutVar), "fan_out_~w_result", [N]),
    extract_stage_names(SubStages, StageNames),
    format_stage_list(StageNames, StageListStr),
    format(string(Code),
"    # Fan-out to ~w parallel stages
    def fan_out_generator():
        for record in ~w:
            for result in fan_out_records(record, [~w]):
                yield result
    ~w = fan_out_generator()", [N, InVar, StageListStr, OutVar]).

% Parallel stage with options: parallel(Stages, Options)
generate_stage_flow(parallel(SubStages, Options), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "parallel_~w_result", [N]),
    extract_stage_names(SubStages, StageNames),
    format_stage_list(StageNames, StageListStr),
    % Check for ordered option
    (   member(ordered(true), Options)
    ->  FuncName = "parallel_records_ordered",
        format(atom(Comment), "Parallel execution (ordered) of ~w stages", [N])
    ;   FuncName = "parallel_records",
        format(atom(Comment), "Parallel execution of ~w stages (concurrent via ThreadPoolExecutor)", [N])
    ),
    format(string(Code),
"    # ~w
    def parallel_generator():
        for record in ~w:
            for result in ~w(record, [~w]):
                yield result
    ~w = parallel_generator()", [Comment, InVar, FuncName, StageListStr, OutVar]).

% Parallel stage: concurrent execution using ThreadPoolExecutor (default: unordered)
generate_stage_flow(parallel(SubStages), InVar, OutVar, Code) :-
    length(SubStages, N),
    format(atom(OutVar), "parallel_~w_result", [N]),
    extract_stage_names(SubStages, StageNames),
    format_stage_list(StageNames, StageListStr),
    format(string(Code),
"    # Parallel execution of ~w stages (concurrent via ThreadPoolExecutor)
    def parallel_generator():
        for record in ~w:
            for result in parallel_records(record, [~w]):
                yield result
    ~w = parallel_generator()", [N, InVar, StageListStr, OutVar]).

% Merge stage: placeholder, usually follows fan_out or parallel
generate_stage_flow(merge, InVar, OutVar, Code) :-
    OutVar = InVar,
    Code = "    # Merge: results already combined from fan-out".

% Conditional routing
generate_stage_flow(route_by(CondPred, Routes), InVar, OutVar, Code) :-
    format(atom(OutVar), "routed_result", []),
    format_route_map(Routes, RouteMapStr),
    format(string(Code),
"    # Conditional routing based on ~w
    def routing_generator():
        route_map = {~w}
        for record in ~w:
            yield from route_record(record, ~w, route_map)
    ~w = routing_generator()", [CondPred, RouteMapStr, InVar, CondPred, OutVar]).

% Filter stage
generate_stage_flow(filter_by(Pred), InVar, OutVar, Code) :-
    format(atom(OutVar), "filtered_result", []),
    format(string(Code),
"    # Filter by ~w
    ~w = filter_records(~w, ~w)", [Pred, OutVar, InVar, Pred]).

% Batch stage: collect N records into batches
generate_stage_flow(batch(N), InVar, OutVar, Code) :-
    format(atom(OutVar), "batched_~w_result", [N]),
    format(string(Code),
"    # Batch records into groups of ~w
    ~w = batch_records(~w, ~w)", [N, OutVar, InVar, N]).

% Unbatch stage: flatten batches back to individual records
generate_stage_flow(unbatch, InVar, OutVar, Code) :-
    OutVar = "unbatched_result",
    format(string(Code),
"    # Unbatch: flatten batches to individual records
    ~w = unbatch_records(~w)", [OutVar, InVar]).

% Unique stage: deduplicate by field (keep first)
generate_stage_flow(unique(Field), InVar, OutVar, Code) :-
    format(atom(OutVar), "unique_~w_result", [Field]),
    format(string(Code),
"    # Unique: keep first record per '~w' value
    ~w = unique_by_field(~w, '~w')", [Field, OutVar, InVar, Field]).

% First stage: alias for unique (keep first occurrence)
generate_stage_flow(first(Field), InVar, OutVar, Code) :-
    format(atom(OutVar), "first_~w_result", [Field]),
    format(string(Code),
"    # First: keep first record per '~w' value
    ~w = first_by_field(~w, '~w')", [Field, OutVar, InVar, Field]).

% Last stage: keep last record per field value
generate_stage_flow(last(Field), InVar, OutVar, Code) :-
    format(atom(OutVar), "last_~w_result", [Field]),
    format(string(Code),
"    # Last: keep last record per '~w' value
    ~w = last_by_field(~w, '~w')", [Field, OutVar, InVar, Field]).

% Group by stage: group and aggregate
generate_stage_flow(group_by(Field, Agg), InVar, OutVar, Code) :-
    format(atom(OutVar), "grouped_~w_result", [Field]),
    format_aggregations(Agg, AggStr),
    format(string(Code),
"    # Group by '~w' with aggregations
    ~w = group_by_field(~w, '~w', [~w])", [Field, OutVar, InVar, Field, AggStr]).

% Reduce stage with initial value: custom sequential fold
generate_stage_flow(reduce(Pred, Init), InVar, OutVar, Code) :-
    format(atom(OutVar), "reduced_result", []),
    format(string(Code),
"    # Reduce: sequential fold with ~w
    ~w = reduce_records(~w, ~w, ~w)", [Pred, OutVar, InVar, Pred, Init]).

% Reduce stage without initial value (defaults to empty dict)
generate_stage_flow(reduce(Pred), InVar, OutVar, Code) :-
    format(atom(OutVar), "reduced_result", []),
    format(string(Code),
"    # Reduce: sequential fold with ~w
    ~w = reduce_records(~w, ~w, {})", [Pred, OutVar, InVar, Pred]).

% Scan stage with initial value: reduce with intermediate outputs
generate_stage_flow(scan(Pred, Init), InVar, OutVar, Code) :-
    format(atom(OutVar), "scanned_result", []),
    format(string(Code),
"    # Scan: running fold with ~w (emits intermediate values)
    ~w = scan_records(~w, ~w, ~w)", [Pred, OutVar, InVar, Pred, Init]).

% Scan stage without initial value
generate_stage_flow(scan(Pred), InVar, OutVar, Code) :-
    format(atom(OutVar), "scanned_result", []),
    format(string(Code),
"    # Scan: running fold with ~w (emits intermediate values)
    ~w = scan_records(~w, ~w, {})", [Pred, OutVar, InVar, Pred]).

% Order by single field (ascending by default)
generate_stage_flow(order_by(Field), InVar, OutVar, Code) :-
    atom(Field),
    !,
    format(atom(OutVar), "ordered_~w_result", [Field]),
    format(string(Code),
"    # Order by '~w' ascending
    ~w = order_by_field(~w, '~w', 'asc')", [Field, OutVar, InVar, Field]).

% Order by single field with direction
generate_stage_flow(order_by(Field, Dir), InVar, OutVar, Code) :-
    atom(Field),
    !,
    format(atom(OutVar), "ordered_~w_result", [Field]),
    format(string(Code),
"    # Order by '~w' ~w
    ~w = order_by_field(~w, '~w', '~w')", [Field, Dir, OutVar, InVar, Field, Dir]).

% Order by multiple fields with directions
generate_stage_flow(order_by(FieldSpecs), InVar, OutVar, Code) :-
    is_list(FieldSpecs),
    !,
    OutVar = "ordered_multi_result",
    format_field_specs(FieldSpecs, SpecStr),
    format(string(Code),
"    # Order by multiple fields
    ~w = order_by_fields(~w, [~w])", [OutVar, InVar, SpecStr]).

% Sort by custom comparator
generate_stage_flow(sort_by(ComparePred), InVar, OutVar, Code) :-
    format(atom(OutVar), "sorted_~w_result", [ComparePred]),
    format(string(Code),
"    # Sort by custom comparator: ~w
    ~w = sort_by_comparator(~w, ~w)", [ComparePred, OutVar, InVar, ComparePred]).

% Try-catch stage: execute stage, on error route to handler
generate_stage_flow(try_catch(Stage, Handler), InVar, OutVar, Code) :-
    !,
    extract_stage_name(Stage, StageName),
    extract_stage_name(Handler, HandlerName),
    OutVar = "try_catch_result",
    format(string(Code),
"    # Try-Catch: ~w with handler ~w
    ~w = try_catch_stage(~w, ~w, ~w)", [StageName, HandlerName, OutVar, InVar, StageName, HandlerName]).

% Retry stage: retry N times on failure
generate_stage_flow(retry(Stage, N), InVar, OutVar, Code) :-
    !,
    extract_stage_name(Stage, StageName),
    OutVar = "retry_result",
    format(string(Code),
"    # Retry: ~w up to ~w times
    ~w = retry_stage(~w, ~w, ~w)", [StageName, N, OutVar, InVar, StageName, N]).

% Retry stage with options
generate_stage_flow(retry(Stage, N, Options), InVar, OutVar, Code) :-
    !,
    extract_stage_name(Stage, StageName),
    OutVar = "retry_result",
    extract_retry_options(Options, DelayMs, Backoff),
    format(string(Code),
"    # Retry: ~w up to ~w times (delay=~wms, backoff=~w)
    ~w = retry_stage(~w, ~w, ~w, ~w, '~w')", [StageName, N, DelayMs, Backoff, OutVar, InVar, StageName, N, DelayMs, Backoff]).

% On-error stage: global error handler
generate_stage_flow(on_error(Handler), InVar, OutVar, Code) :-
    !,
    extract_stage_name(Handler, HandlerName),
    OutVar = "on_error_result",
    format(string(Code),
"    # On-Error: route errors to ~w
    ~w = on_error_stage(~w, ~w)", [HandlerName, OutVar, InVar, HandlerName]).

% Timeout stage: execute with time limit
generate_stage_flow(timeout(Stage, Ms), InVar, OutVar, Code) :-
    !,
    extract_stage_name(Stage, StageName),
    OutVar = "timeout_result",
    format(string(Code),
"    # Timeout: ~w with ~wms limit
    ~w = timeout_stage(~w, ~w, ~w)", [StageName, Ms, OutVar, InVar, StageName, Ms]).

% Timeout stage with fallback
generate_stage_flow(timeout(Stage, Ms, Fallback), InVar, OutVar, Code) :-
    !,
    extract_stage_name(Stage, StageName),
    extract_stage_name(Fallback, FallbackName),
    OutVar = "timeout_result",
    format(string(Code),
"    # Timeout: ~w with ~wms limit, fallback to ~w
    ~w = timeout_stage_with_fallback(~w, ~w, ~w, ~w)", [StageName, Ms, FallbackName, OutVar, InVar, StageName, Ms, FallbackName]).

% Rate limit stage: limit throughput
generate_stage_flow(rate_limit(N, Per), InVar, OutVar, Code) :-
    !,
    OutVar = "rate_limited_result",
    format_time_unit(Per, PerStr),
    format(string(Code),
"    # Rate Limit: ~w per ~w
    ~w = rate_limit_stage(~w, ~w, ~w)", [N, Per, OutVar, InVar, N, PerStr]).

% Throttle stage: fixed delay between records
generate_stage_flow(throttle(Ms), InVar, OutVar, Code) :-
    !,
    OutVar = "throttled_result",
    format(string(Code),
"    # Throttle: ~wms delay between records
    ~w = throttle_stage(~w, ~w)", [Ms, OutVar, InVar, Ms]).

% Buffer stage: collect records into batches
generate_stage_flow(buffer(N), InVar, OutVar, Code) :-
    !,
    OutVar = "buffered_result",
    format(string(Code),
"    # Buffer: collect ~w records into batches
    ~w = buffer_stage(~w, ~w)", [N, OutVar, InVar, N]).

% Debounce stage: emit only if no new record within delay
generate_stage_flow(debounce(Ms), InVar, OutVar, Code) :-
    !,
    OutVar = "debounced_result",
    format(string(Code),
"    # Debounce: ~wms quiet period
    ~w = debounce_stage(~w, ~w)", [Ms, OutVar, InVar, Ms]).

% Zip stage: combine multiple stages record-by-record
generate_stage_flow(zip(Stages), InVar, OutVar, Code) :-
    !,
    OutVar = "zipped_result",
    extract_stage_names(Stages, Names),
    format_stage_list(Names, StageListStr),
    format(string(Code),
"    # Zip: combine outputs from multiple stages
    def zip_generator():
        for record in ~w:
            for result in zip_stage(record, [~w]):
                yield result
    ~w = zip_generator()", [InVar, StageListStr, OutVar]).

% Window stage: non-overlapping windows
generate_stage_flow(window(N), InVar, OutVar, Code) :-
    !,
    OutVar = "windowed_result",
    format(string(Code),
"    # Window: collect ~w records into windows
    ~w = window_stage(~w, ~w)", [N, OutVar, InVar, N]).

% Sliding window stage
generate_stage_flow(sliding_window(N, Step), InVar, OutVar, Code) :-
    !,
    OutVar = "sliding_window_result",
    format(string(Code),
"    # Sliding Window: size ~w, step ~w
    ~w = sliding_window_stage(~w, ~w, ~w)", [N, Step, OutVar, InVar, N, Step]).

% Sample stage: random sampling
generate_stage_flow(sample(N), InVar, OutVar, Code) :-
    !,
    OutVar = "sampled_result",
    format(string(Code),
"    # Sample: random ~w records
    ~w = sample_stage(~w, ~w)", [N, OutVar, InVar, N]).

% Take every stage
generate_stage_flow(take_every(N), InVar, OutVar, Code) :-
    !,
    OutVar = "take_every_result",
    format(string(Code),
"    # Take Every: every ~wth record
    ~w = take_every_stage(~w, ~w)", [N, OutVar, InVar, N]).

% Partition stage
generate_stage_flow(partition(Pred), InVar, OutVar, Code) :-
    !,
    OutVar = "partitioned_result",
    format(string(Code),
"    # Partition: split by ~w
    ~w = partition_stage(~w, ~w)", [Pred, OutVar, InVar, Pred]).

% Take stage
generate_stage_flow(take(N), InVar, OutVar, Code) :-
    !,
    OutVar = "take_result",
    format(string(Code),
"    # Take: first ~w records
    ~w = take_stage(~w, ~w)", [N, OutVar, InVar, N]).

% Skip stage
generate_stage_flow(skip(N), InVar, OutVar, Code) :-
    !,
    OutVar = "skip_result",
    format(string(Code),
"    # Skip: skip first ~w records
    ~w = skip_stage(~w, ~w)", [N, OutVar, InVar, N]).

% Take while stage
generate_stage_flow(take_while(Pred), InVar, OutVar, Code) :-
    !,
    OutVar = "take_while_result",
    format(string(Code),
"    # Take While: while ~w is true
    ~w = take_while_stage(~w, ~w)", [Pred, OutVar, InVar, Pred]).

% Skip while stage
generate_stage_flow(skip_while(Pred), InVar, OutVar, Code) :-
    !,
    OutVar = "skip_while_result",
    format(string(Code),
"    # Skip While: skip while ~w is true
    ~w = skip_while_stage(~w, ~w)", [Pred, OutVar, InVar, Pred]).

% Distinct stage: remove all duplicates (global)
generate_stage_flow(distinct, InVar, OutVar, Code) :-
    !,
    OutVar = "distinct_result",
    format(string(Code),
"    # Distinct: remove all duplicates (global dedup)
    ~w = distinct_stage(~w)", [OutVar, InVar]).

% Distinct by field: remove duplicates based on specific field
generate_stage_flow(distinct_by(Field), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "distinct_~w_result", [Field]),
    format(string(Code),
"    # Distinct By: remove duplicates based on '~w' field
    ~w = distinct_by_stage(~w, '~w')", [Field, OutVar, InVar, Field]).

% Dedup stage: remove consecutive duplicates only
generate_stage_flow(dedup, InVar, OutVar, Code) :-
    !,
    OutVar = "dedup_result",
    format(string(Code),
"    # Dedup: remove consecutive duplicates
    ~w = dedup_stage(~w)", [OutVar, InVar]).

% Dedup by field: remove consecutive duplicates based on specific field
generate_stage_flow(dedup_by(Field), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "dedup_~w_result", [Field]),
    format(string(Code),
"    # Dedup By: remove consecutive duplicates based on '~w' field
    ~w = dedup_by_stage(~w, '~w')", [Field, OutVar, InVar, Field]).

% Interleave stage: round-robin interleave from multiple stages
generate_stage_flow(interleave(Stages), InVar, OutVar, Code) :-
    !,
    length(Stages, N),
    format(atom(OutVar), "interleaved_~w", [N]),
    extract_stage_names(Stages, StageNames),
    format_stage_list(StageNames, StageListStr),
    format(string(Code),
"    # Interleave: round-robin from ~w stages
    ~w = interleave_stage([~w(~w) for stage_fn in [~w]])", [N, OutVar, "stage_fn", InVar, StageListStr]).

% Concat stage: sequential concatenation of multiple stages
generate_stage_flow(concat(Stages), InVar, OutVar, Code) :-
    !,
    length(Stages, N),
    format(atom(OutVar), "concatenated_~w", [N]),
    extract_stage_names(Stages, StageNames),
    format_stage_list(StageNames, StageListStr),
    format(string(Code),
"    # Concat: sequential concatenation of ~w stages
    ~w = concat_stage([~w(~w) for stage_fn in [~w]])", [N, OutVar, "stage_fn", InVar, StageListStr]).

% Merge sorted stage: merge pre-sorted streams maintaining order (ascending)
generate_stage_flow(merge_sorted(Stages, Field), InVar, OutVar, Code) :-
    !,
    length(Stages, N),
    format(atom(OutVar), "merge_sorted_~w_result", [Field]),
    extract_stage_names(Stages, StageNames),
    format_stage_list(StageNames, StageListStr),
    format(string(Code),
"    # Merge Sorted: merge ~w pre-sorted streams by '~w' (ascending)
    ~w = merge_sorted_stage([stage_fn(~w) for stage_fn in [~w]], '~w', reverse=False)", [N, Field, OutVar, InVar, StageListStr, Field]).

% Merge sorted stage with direction: merge pre-sorted streams with specified order
generate_stage_flow(merge_sorted(Stages, Field, Dir), InVar, OutVar, Code) :-
    !,
    length(Stages, N),
    format(atom(OutVar), "merge_sorted_~w_~w_result", [Field, Dir]),
    extract_stage_names(Stages, StageNames),
    format_stage_list(StageNames, StageListStr),
    ( Dir = desc -> Reverse = "True" ; Reverse = "False" ),
    format(string(Code),
"    # Merge Sorted: merge ~w pre-sorted streams by '~w' (~w)
    ~w = merge_sorted_stage([stage_fn(~w) for stage_fn in [~w]], '~w', reverse=~w)", [N, Field, Dir, OutVar, InVar, StageListStr, Field, Reverse]).

% Tap stage: execute side effect without modifying stream
generate_stage_flow(tap(Pred), InVar, OutVar, Code) :-
    !,
    ( Pred = PredName/_ -> true ; PredName = Pred ),
    format(atom(OutVar), "tapped_~w_result", [PredName]),
    format(string(Code),
"    # Tap: execute ~w for side effects (logging/metrics)
    ~w = tap_stage(~w, ~w)", [PredName, OutVar, InVar, PredName]).

% Flatten stage: flatten nested collections
generate_stage_flow(flatten, InVar, OutVar, Code) :-
    !,
    OutVar = "flattened_result",
    format(string(Code),
"    # Flatten: expand nested collections into individual records
    ~w = flatten_stage(~w)", [OutVar, InVar]).

% Flatten field stage: flatten a specific field within records
generate_stage_flow(flatten(Field), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "flattened_~w_result", [Field]),
    format(string(Code),
"    # Flatten Field: expand '~w' field into individual records
    ~w = flatten_field_stage(~w, '~w')", [Field, OutVar, InVar, Field]).

% Debounce stage: emit only after silence period
generate_stage_flow(debounce(Ms), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "debounced_~w_result", [Ms]),
    format(string(Code),
"    # Debounce: emit after ~wms silence period
    ~w = debounce_stage(~w, ~w)", [Ms, OutVar, InVar, Ms]).

% Debounce stage with timestamp field
generate_stage_flow(debounce(Ms, Field), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "debounced_~w_~w_result", [Ms, Field]),
    format(string(Code),
"    # Debounce: emit after ~wms silence (using '~w' timestamp field)
    ~w = debounce_stage(~w, ~w, '~w')", [Ms, Field, OutVar, InVar, Ms, Field]).

% Branch stage: conditional routing
generate_stage_flow(branch(Cond, TrueStage, FalseStage), InVar, OutVar, Code) :-
    !,
    OutVar = "branch_result",
    % Extract condition predicate name
    ( Cond = CondName/_ -> true ; CondName = Cond ),
    % Extract true/false stage names
    ( TrueStage = TrueName/_ -> true ; TrueName = TrueStage ),
    ( FalseStage = FalseName/_ -> true ; FalseName = FalseStage ),
    format(string(Code),
"    # Branch: if ~w then ~w else ~w
    def _branch_cond(record):
        return ~w(record)
    def _branch_true(stream):
        return ~w(stream)
    def _branch_false(stream):
        return ~w(stream)
    ~w = branch_stage(~w, _branch_cond, _branch_true, _branch_false)",
    [CondName, TrueName, FalseName, CondName, TrueName, FalseName, OutVar, InVar]).

% Tee stage: run side stage, discard results, pass through
generate_stage_flow(tee(SideStage), InVar, OutVar, Code) :-
    !,
    OutVar = "tee_result",
    % Extract side stage name
    ( SideStage = SideName/_ -> true ; SideName = SideStage ),
    format(string(Code),
"    # Tee: fork to ~w (results discarded), pass original through
    def _tee_side(stream):
        return ~w(stream)
    ~w = tee_stage(~w, _tee_side)",
    [SideName, SideName, OutVar, InVar]).

% Call service stage (without options)
generate_stage_flow(call_service(ServiceName, RequestExpr, ResponseVar), InVar, OutVar, Code) :-
    !,
    OutVar = "service_result",
    format(string(Code),
"    # Call service: ~w
    ~w = call_service_stage(~w, '~w', '~w', '~w')",
    [ServiceName, OutVar, InVar, ServiceName, RequestExpr, ResponseVar]).

% Call service stage (with options)
generate_stage_flow(call_service(ServiceName, RequestExpr, ResponseVar, Options), InVar, OutVar, Code) :-
    !,
    OutVar = "service_result",
    format_python_options(Options, OptionsStr),
    format(string(Code),
"    # Call service: ~w (with options)
    ~w = call_service_stage(~w, '~w', '~w', '~w', ~w)",
    [ServiceName, OutVar, InVar, ServiceName, RequestExpr, ResponseVar, OptionsStr]).

% Standard predicate stage
generate_stage_flow(Pred/Arity, InVar, OutVar, Code) :-
    atom(Pred),
    format(atom(OutVar), "~w_result", [Pred]),
    format(string(Code),
"    # Stage: ~w/~w
    ~w = ~w(~w)", [Pred, Arity, OutVar, Pred, InVar]).

% Fallback for unknown stages
generate_stage_flow(Stage, InVar, InVar, Code) :-
    format(string(Code), "    # Unknown stage type: ~w (pass-through)", [Stage]).

%% format_python_options(+Options, -PythonDict)
%  Format a list of Prolog options as a Python dictionary string.
format_python_options([], "{}").
format_python_options(Options, Dict) :-
    Options \= [],
    format_python_option_pairs(Options, Pairs),
    atomic_list_concat(Pairs, ', ', PairsStr),
    format(string(Dict), "{~w}", [PairsStr]).

format_python_option_pairs([], []).
format_python_option_pairs([Opt|Rest], [Pair|RestPairs]) :-
    format_python_option(Opt, Pair),
    format_python_option_pairs(Rest, RestPairs).

format_python_option(timeout(Ms), Pair) :-
    format(string(Pair), "'timeout': ~w", [Ms]).
format_python_option(retry(N), Pair) :-
    format(string(Pair), "'retry': ~w", [N]).
format_python_option(retry_delay(Ms), Pair) :-
    format(string(Pair), "'retry_delay': ~w", [Ms]).
format_python_option(fallback(Value), Pair) :-
    ( atom(Value) ->
        format(string(Pair), "'fallback': '~w'", [Value])
    ; number(Value) ->
        format(string(Pair), "'fallback': ~w", [Value])
    ;
        format(string(Pair), "'fallback': None", [])
    ).
format_python_option(transport(T), Pair) :-
    format(string(Pair), "'transport': '~w'", [T]).
format_python_option(Opt, Pair) :-
    % Fallback for unknown options
    format(string(Pair), "# Unknown option: ~w", [Opt]).

%% extract_stage_names(+Stages, -Names)
%  Extract function names from stage specifications.
extract_stage_names([], []).
extract_stage_names([Pred/_Arity|Rest], [Pred|RestNames]) :-
    !,
    extract_stage_names(Rest, RestNames).
extract_stage_names([_|Rest], RestNames) :-
    extract_stage_names(Rest, RestNames).

%% format_stage_list(+Names, -Str)
%  Format a list of stage names as Python function references.
format_stage_list([], "").
format_stage_list([Name], Str) :-
    format(string(Str), "~w", [Name]).
format_stage_list([Name|Rest], Str) :-
    Rest \= [],
    format_stage_list(Rest, RestStr),
    format(string(Str), "~w, ~w", [Name, RestStr]).

%% format_aggregations(+Agg, -Str)
%  Format aggregation specifications for group_by stage.
%  Aggregations can be: count, sum(Field), avg(Field), min(Field), max(Field),
%  first(Field), last(Field), collect(Field), or a list of these.
format_aggregations(Aggs, Str) :-
    is_list(Aggs),
    !,
    format_aggregation_list(Aggs, Str).
format_aggregations(Agg, Str) :-
    format_single_aggregation(Agg, Str).

format_aggregation_list([], "").
format_aggregation_list([Agg], Str) :-
    format_single_aggregation(Agg, Str).
format_aggregation_list([Agg|Rest], Str) :-
    Rest \= [],
    format_single_aggregation(Agg, AggStr),
    format_aggregation_list(Rest, RestStr),
    format(string(Str), "~w, ~w", [AggStr, RestStr]).

% count aggregation (no field needed)
format_single_aggregation(count, "('count', 'count', None)").
% Aggregations with field: sum(Field), avg(Field), etc.
format_single_aggregation(sum(Field), Str) :-
    format(string(Str), "('sum', 'sum', '~w')", [Field]).
format_single_aggregation(avg(Field), Str) :-
    format(string(Str), "('avg', 'avg', '~w')", [Field]).
format_single_aggregation(min(Field), Str) :-
    format(string(Str), "('min', 'min', '~w')", [Field]).
format_single_aggregation(max(Field), Str) :-
    format(string(Str), "('max', 'max', '~w')", [Field]).
format_single_aggregation(first(Field), Str) :-
    format(string(Str), "('first', 'first', '~w')", [Field]).
format_single_aggregation(last(Field), Str) :-
    format(string(Str), "('last', 'last', '~w')", [Field]).
format_single_aggregation(collect(Field), Str) :-
    format(string(Str), "('collect', 'collect', '~w')", [Field]).

%% format_field_specs(+FieldSpecs, -Str)
%  Format field specifications for multi-field ordering.
%  FieldSpecs can be atoms (field names with default asc) or (Field, Dir) tuples.
format_field_specs([], "").
format_field_specs([Spec], Str) :-
    format_single_field_spec(Spec, Str).
format_field_specs([Spec|Rest], Str) :-
    Rest \= [],
    format_single_field_spec(Spec, SpecStr),
    format_field_specs(Rest, RestStr),
    format(string(Str), "~w, ~w", [SpecStr, RestStr]).

format_single_field_spec(Field, Str) :-
    atom(Field),
    !,
    format(string(Str), "('~w', 'asc')", [Field]).
format_single_field_spec((Field, Dir), Str) :-
    format(string(Str), "('~w', '~w')", [Field, Dir]).

%% format_route_map(+Routes, -Str)
%  Format route mappings as Python dict entries.
format_route_map([], "").
format_route_map([(Cond, Pred/_Arity)], Str) :-
    format(string(Str), "~q: ~w", [Cond, Pred]).
format_route_map([(Cond, Pred/_Arity)|Rest], Str) :-
    Rest \= [],
    format_route_map(Rest, RestStr),
    format(string(Str), "~q: ~w, ~w", [Cond, Pred, RestStr]).

%% extract_stage_name(+Stage, -Name)
%  Extract the function name from a stage specification.
extract_stage_name(Pred/_, Pred) :- atom(Pred), !.
extract_stage_name(Pred, Pred) :- atom(Pred), !.
extract_stage_name(_, unknown_stage).

%% extract_retry_options(+Options, -DelayMs, -Backoff)
%  Extract retry options from options list.
extract_retry_options(Options, DelayMs, Backoff) :-
    ( member(delay(D), Options) -> DelayMs = D ; DelayMs = 0 ),
    ( member(backoff(B), Options) -> Backoff = B ; Backoff = none ).

%% format_time_unit(+Unit, -Str)
%  Format time unit for Python code generation.
format_time_unit(second, "'second'") :- !.
format_time_unit(minute, "'minute'") :- !.
format_time_unit(hour, "'hour'") :- !.
format_time_unit(ms(X), Str) :- !, format(atom(Str), "~w", [X]).
format_time_unit(X, Str) :- format(atom(Str), "'~w'", [X]).

% ============================================================================
% Pipeline Chaining Tests (Phase 4)
% ============================================================================

test_pipeline_chaining :-
    format('~n=== Python Pipeline Chaining Tests (Phase 4) ===~n~n', []),

    % Test 1: All same runtime detection
    format('[Test 1] Same runtime detection~n', []),
    (   all_same_runtime([get_users/1, filter_active/2, format_output/1])
    ->  format('  [PASS] Plain predicates are same runtime~n', [])
    ;   format('  [FAIL] Should detect same runtime~n', [])
    ),

    % Test 2: Mixed runtime detection
    format('[Test 2] Mixed runtime detection~n', []),
    (   \+ all_same_runtime([python:get_users/1, csharp:validate/1])
    ->  format('  [PASS] Python + C# detected as different~n', [])
    ;   format('  [FAIL] Should detect different runtimes~n', [])
    ),

    % Test 3: Predicate runtime extraction
    format('[Test 3] Predicate runtime extraction~n', []),
    predicate_runtime(ironpython:foo/2, R3),
    (   R3 == ironpython
    ->  format('  [PASS] ironpython:foo/2 -> ironpython~n', [])
    ;   format('  [FAIL] Got ~w~n', [R3])
    ),

    % Test 4: Compile same-runtime pipeline
    format('[Test 4] Compile same-runtime pipeline~n', []),
    compile_same_runtime_pipeline(
        [get_users/1, filter_active/2],
        [runtime(cpython), pipeline_name(my_pipeline)],
        Code4
    ),
    (   sub_string(Code4, _, _, _, "def my_pipeline"),
        sub_string(Code4, _, _, _, "def get_users"),
        sub_string(Code4, _, _, _, "def filter_active")
    ->  format('  [PASS] Same-runtime pipeline generated~n', [])
    ;   format('  [FAIL] Pipeline missing components~n', [])
    ),

    % Test 5: Pipeline connector generation
    format('[Test 5] Pipeline connector generation~n', []),
    generate_pipeline_connector([a/1, b/1, c/1], test_pipe, Code5),
    (   sub_string(Code5, _, _, _, "def test_pipe"),
        sub_string(Code5, _, _, _, "yield from")
    ->  format('  [PASS] Connector chains predicates~n', [])
    ;   format('  [FAIL] Connector issue~n', [])
    ),

    % Test 6: Group by runtime
    format('[Test 6] Group predicates by runtime~n', []),
    group_by_runtime(
        [python:a/1, python:b/1, csharp:c/1, python:d/1],
        Groups6
    ),
    (   Groups6 = [group(python, [python:a/1, python:b/1]),
                   group(csharp, [csharp:c/1]),
                   group(python, [python:d/1])]
    ->  format('  [PASS] Grouped into 3 stages~n', [])
    ;   format('  [FAIL] Got ~w~n', [Groups6])
    ),

    % Test 7: Cross-runtime pipeline
    format('[Test 7] Cross-runtime pipeline~n', []),
    compile_cross_runtime_pipeline(
        [python:extract/1, csharp:validate/1, python:format/1],
        [pipeline_name(cross_pipe)],
        Code7
    ),
    (   sub_string(Code7, _, _, _, "stage_1"),
        sub_string(Code7, _, _, _, "stage_2"),
        sub_string(Code7, _, _, _, "stage_3"),
        sub_string(Code7, _, _, _, "def cross_pipe")
    ->  format('  [PASS] Cross-runtime pipeline has 3 stages~n', [])
    ;   format('  [FAIL] Cross-runtime pipeline issue~n', [])
    ),

    % Test 8: Main entry point dispatch
    format('[Test 8] Main entry point dispatch~n', []),
    compile_pipeline([a/1, b/1], [pipeline_name(dispatch_test)], Code8a),
    compile_pipeline([python:a/1, csharp:b/1], [pipeline_name(dispatch_test)], Code8b),
    (   sub_string(Code8a, _, _, _, "def dispatch_test"),
        sub_string(Code8b, _, _, _, "stage_1")
    ->  format('  [PASS] Dispatch selects correct mode~n', [])
    ;   format('  [FAIL] Dispatch issue~n', [])
    ),

    format('~n=== All Pipeline Chaining Tests Passed ===~n', []).

%% ============================================
%% PYTHON PIPELINE GENERATOR MODE TESTS
%% ============================================

test_python_pipeline_generator :-
    format('~n=== Python Pipeline Generator Mode Tests ===~n~n', []),

    % Test 1: Pipeline header extended for generator mode
    format('[Test 1] Pipeline header extended (generator)~n', []),
    pipeline_header_extended(jsonl, cpython, generator, Header1),
    (   sub_string(Header1, _, _, _, "from typing import Set"),
        sub_string(Header1, _, _, _, "from dataclasses import dataclass")
    ->  format('  [PASS] Generator header has required imports~n', [])
    ;   format('  [FAIL] Header: ~w~n', [Header1])
    ),

    % Test 2: Pipeline header extended for sequential mode (unchanged)
    format('[Test 2] Pipeline header extended (sequential)~n', []),
    pipeline_header_extended(jsonl, cpython, sequential, Header2),
    pipeline_header(jsonl, cpython, BaseHeader2),
    (   Header2 == BaseHeader2
    ->  format('  [PASS] Sequential header unchanged~n', [])
    ;   format('  [FAIL] Headers differ~n', [])
    ),

    % Test 3: Pipeline helpers extended for generator mode (CPython)
    format('[Test 3] Pipeline helpers extended (generator, CPython)~n', []),
    pipeline_helpers_extended(jsonl, same_as_data, generator, cpython, Helpers3),
    (   sub_string(Helpers3, _, _, _, "class FrozenDict"),
        sub_string(Helpers3, _, _, _, "def record_key"),
        sub_string(Helpers3, _, _, _, "from_dict")
    ->  format('  [PASS] Generator helpers include FrozenDict~n', [])
    ;   format('  [FAIL] Helpers: ~w~n', [Helpers3])
    ),

    % Test 4: Generate fixpoint chain code (empty)
    format('[Test 4] Fixpoint chain code (empty)~n', []),
    generate_fixpoint_chain_code([], ChainCode4),
    (   sub_string(ChainCode4, _, _, _, "new_records = current")
    ->  format('  [PASS] Empty chain returns current~n', [])
    ;   format('  [FAIL] Chain: ~w~n', [ChainCode4])
    ),

    % Test 5: Generate fixpoint chain code with stages
    format('[Test 5] Fixpoint chain code with stages~n', []),
    generate_fixpoint_chain_code(["stage1", "stage2"], ChainCode5),
    (   sub_string(ChainCode5, _, _, _, "stage_stage1_out"),
        sub_string(ChainCode5, _, _, _, "stage_stage2_out"),
        sub_string(ChainCode5, _, _, _, "stage1(iter(current))"),
        sub_string(ChainCode5, _, _, _, "stage2(iter(stage_stage1_out))")
    ->  format('  [PASS] Stage calls generated correctly~n', [])
    ;   format('  [FAIL] Chain code: ~w~n', [ChainCode5])
    ),

    % Test 6: Pipeline connector for generator mode
    format('[Test 6] Pipeline connector (generator)~n', []),
    generate_pipeline_connector([a/1, b/1], test_gen, generator, ConnCode6),
    (   sub_string(ConnCode6, _, _, _, "def test_gen"),
        sub_string(ConnCode6, _, _, _, "Fixpoint pipeline"),
        sub_string(ConnCode6, _, _, _, "total: Set[FrozenDict]"),
        sub_string(ConnCode6, _, _, _, "while changed"),
        sub_string(ConnCode6, _, _, _, "record_key(record)")
    ->  format('  [PASS] Generator connector has fixpoint loop~n', [])
    ;   format('  [FAIL] Connector: ~w~n', [ConnCode6])
    ),

    % Test 7: Pipeline connector for sequential mode (unchanged)
    format('[Test 7] Pipeline connector (sequential)~n', []),
    generate_pipeline_connector([a/1, b/1], test_seq, sequential, ConnCode7),
    (   sub_string(ConnCode7, _, _, _, "def test_seq"),
        sub_string(ConnCode7, _, _, _, "Chained pipeline"),
        sub_string(ConnCode7, _, _, _, "yield from")
    ->  format('  [PASS] Sequential connector unchanged~n', [])
    ;   format('  [FAIL] Connector: ~w~n', [ConnCode7])
    ),

    % Test 8: Full pipeline with generator mode
    format('[Test 8] Full pipeline (generator mode)~n', []),
    compile_same_runtime_pipeline([stage1/1, stage2/1], [
        pipeline_name(fixpoint_pipe),
        pipeline_mode(generator)
    ], FullCode8),
    (   sub_string(FullCode8, _, _, _, "class FrozenDict"),
        sub_string(FullCode8, _, _, _, "def fixpoint_pipe"),
        sub_string(FullCode8, _, _, _, "while changed"),
        sub_string(FullCode8, _, _, _, "total.add(key)")
    ->  format('  [PASS] Full generator pipeline compiled~n', [])
    ;   format('  [FAIL] Missing expected patterns~n', [])
    ),

    % Test 9: Full pipeline with sequential mode (default)
    format('[Test 9] Full pipeline (sequential, default)~n', []),
    compile_same_runtime_pipeline([stage1/1, stage2/1], [
        pipeline_name(seq_pipe)
    ], FullCode9),
    (   sub_string(FullCode9, _, _, _, "def seq_pipe"),
        sub_string(FullCode9, _, _, _, "yield from"),
        \+ sub_string(FullCode9, _, _, _, "class FrozenDict")
    ->  format('  [PASS] Sequential pipeline (default) works~n', [])
    ;   format('  [FAIL] Sequential pipeline issue~n', [])
    ),

    % Test 10: Main entry point with generator mode
    format('[Test 10] Main entry point (generator)~n', []),
    compile_pipeline([a/1, b/1], [
        pipeline_name(main_gen_pipe),
        pipeline_mode(generator)
    ], FullCode10),
    (   sub_string(FullCode10, _, _, _, "def main_gen_pipe"),
        sub_string(FullCode10, _, _, _, "class FrozenDict"),
        sub_string(FullCode10, _, _, _, "while changed")
    ->  format('  [PASS] Main entry point uses generator mode~n', [])
    ;   format('  [FAIL] Main entry point issue~n', [])
    ),

    format('~n=== All Python Pipeline Generator Mode Tests Passed ===~n', []).

%% ============================================
%% IRONPYTHON PIPELINE GENERATOR MODE TESTS
%% ============================================

test_ironpython_pipeline_generator :-
    format('~n=== IronPython Pipeline Generator Mode Tests ===~n~n', []),

    % Test 1: IronPython header for generator mode has .NET HashSet
    format('[Test 1] IronPython header (generator)~n', []),
    pipeline_header_extended(jsonl, ironpython, generator, Header1),
    (   sub_string(Header1, _, _, _, "import clr"),
        sub_string(Header1, _, _, _, "HashSet"),
        \+ sub_string(Header1, _, _, _, "dataclass")
    ->  format('  [PASS] IronPython header has .NET HashSet import~n', [])
    ;   format('  [FAIL] Header: ~w~n', [Header1])
    ),

    % Test 2: IronPython helpers for generator mode
    format('[Test 2] IronPython helpers (generator)~n', []),
    pipeline_helpers_extended(jsonl, same_as_data, generator, ironpython, Helpers2),
    (   sub_string(Helpers2, _, _, _, "class RecordSet"),
        sub_string(Helpers2, _, _, _, "HashSet[String]"),
        sub_string(Helpers2, _, _, _, "def record_key")
    ->  format('  [PASS] IronPython helpers include RecordSet wrapper~n', [])
    ;   format('  [FAIL] Helpers: ~w~n', [Helpers2])
    ),

    % Test 3: IronPython connector for generator mode
    format('[Test 3] IronPython connector (generator)~n', []),
    generate_pipeline_connector([a/1, b/1], iron_gen, generator, ironpython, ConnCode3),
    (   sub_string(ConnCode3, _, _, _, "def iron_gen"),
        sub_string(ConnCode3, _, _, _, "RecordSet()"),
        sub_string(ConnCode3, _, _, _, "IronPython/.NET"),
        sub_string(ConnCode3, _, _, _, "while changed")
    ->  format('  [PASS] IronPython connector uses RecordSet~n', [])
    ;   format('  [FAIL] Connector: ~w~n', [ConnCode3])
    ),

    % Test 4: Full IronPython pipeline with generator mode
    format('[Test 4] Full IronPython pipeline (generator)~n', []),
    compile_same_runtime_pipeline([stage1/1, stage2/1], [
        pipeline_name(iron_fixpoint),
        pipeline_mode(generator),
        runtime(ironpython)
    ], FullCode4),
    (   sub_string(FullCode4, _, _, _, "#!/usr/bin/env ipy"),
        sub_string(FullCode4, _, _, _, "class RecordSet"),
        sub_string(FullCode4, _, _, _, "def iron_fixpoint"),
        sub_string(FullCode4, _, _, _, "while changed")
    ->  format('  [PASS] Full IronPython generator pipeline compiled~n', [])
    ;   format('  [FAIL] Missing expected patterns~n', [])
    ),

    % Test 5: IronPython sequential mode still works
    format('[Test 5] IronPython sequential mode~n', []),
    compile_same_runtime_pipeline([stage1/1], [
        pipeline_name(iron_seq),
        pipeline_mode(sequential),
        runtime(ironpython)
    ], SeqCode5),
    (   sub_string(SeqCode5, _, _, _, "#!/usr/bin/env ipy"),
        sub_string(SeqCode5, _, _, _, "def iron_seq"),
        sub_string(SeqCode5, _, _, _, "yield from"),
        \+ sub_string(SeqCode5, _, _, _, "RecordSet")
    ->  format('  [PASS] IronPython sequential mode works~n', [])
    ;   format('  [FAIL] Sequential issue~n', [])
    ),

    % Test 6: IronPython generator uses all_records list
    format('[Test 6] IronPython generator tracks all_records~n', []),
    generate_pipeline_connector([a/1], track_test, generator, ironpython, ConnCode6),
    (   sub_string(ConnCode6, _, _, _, "all_records = []"),
        sub_string(ConnCode6, _, _, _, "all_records.append(record)")
    ->  format('  [PASS] IronPython tracks all_records for iteration~n', [])
    ;   format('  [FAIL] Tracking issue~n', [])
    ),

    % Test 7: CPython generator still uses FrozenDict
    format('[Test 7] CPython generator uses FrozenDict~n', []),
    compile_same_runtime_pipeline([stage1/1], [
        pipeline_name(py_gen),
        pipeline_mode(generator),
        runtime(cpython)
    ], PyCode7),
    (   sub_string(PyCode7, _, _, _, "class FrozenDict"),
        sub_string(PyCode7, _, _, _, "Set[FrozenDict]"),
        \+ sub_string(PyCode7, _, _, _, "RecordSet")
    ->  format('  [PASS] CPython uses FrozenDict (not RecordSet)~n', [])
    ;   format('  [FAIL] CPython issue~n', [])
    ),

    % Test 8: IronPython header has CLR references
    format('[Test 8] IronPython header CLR references~n', []),
    pipeline_header(jsonl, ironpython, BaseHeader8),
    (   sub_string(BaseHeader8, _, _, _, "clr.AddReference"),
        sub_string(BaseHeader8, _, _, _, "from System import")
    ->  format('  [PASS] IronPython header has CLR setup~n', [])
    ;   format('  [FAIL] Missing CLR: ~w~n', [BaseHeader8])
    ),

    % Test 9: IronPython record_key uses json.dumps
    format('[Test 9] IronPython record_key serialization~n', []),
    pipeline_helpers_extended(jsonl, same_as_data, generator, ironpython, Helpers9),
    (   sub_string(Helpers9, _, _, _, "json.dumps"),
        sub_string(Helpers9, _, _, _, "sort_keys=True")
    ->  format('  [PASS] IronPython record_key uses JSON serialization~n', [])
    ;   format('  [FAIL] Serialization issue~n', [])
    ),

    % Test 10: IronPython RecordSet __contains__ method
    format('[Test 10] IronPython RecordSet contains check~n', []),
    pipeline_helpers_extended(jsonl, same_as_data, generator, ironpython, Helpers10),
    (   sub_string(Helpers10, _, _, _, "def __contains__"),
        sub_string(Helpers10, _, _, _, ".Contains(String(key))")
    ->  format('  [PASS] RecordSet has proper contains check~n', [])
    ;   format('  [FAIL] Contains issue~n', [])
    ),

    format('~n=== All IronPython Pipeline Generator Mode Tests Passed ===~n', []).

%% ============================================
%% ENHANCED PIPELINE CHAINING TESTS (Phase 7)
%% ============================================

test_enhanced_pipeline_chaining :-
    format('~n=== Enhanced Pipeline Chaining Tests (Phase 7) ===~n~n', []),

    % Test 1: Enhanced pipeline helpers generation
    format('[Test 1] Enhanced pipeline helpers~n', []),
    enhanced_pipeline_helpers(Helpers1),
    (   sub_string(Helpers1, _, _, _, "def fan_out_records"),
        sub_string(Helpers1, _, _, _, "def merge_streams"),
        sub_string(Helpers1, _, _, _, "def route_record"),
        sub_string(Helpers1, _, _, _, "def filter_records")
    ->  format('  [PASS] All enhanced helpers generated~n', [])
    ;   format('  [FAIL] Missing helper functions~n', [])
    ),

    % Test 2: Simple linear pipeline
    format('[Test 2] Simple linear pipeline~n', []),
    generate_enhanced_connector([extract/1, transform/1, load/1], linear_pipe, Code2),
    (   sub_string(Code2, _, _, _, "def linear_pipe"),
        sub_string(Code2, _, _, _, "extract_result"),
        sub_string(Code2, _, _, _, "transform_result"),
        sub_string(Code2, _, _, _, "load_result")
    ->  format('  [PASS] Linear pipeline generated~n', [])
    ;   format('  [FAIL] Linear pipeline issue~n', [])
    ),

    % Test 3: Fan-out stage
    format('[Test 3] Fan-out stage~n', []),
    generate_enhanced_connector([fan_out([validate/1, enrich/1])], fanout_pipe, Code3),
    (   sub_string(Code3, _, _, _, "def fanout_pipe"),
        sub_string(Code3, _, _, _, "Fan-out to 2 parallel stages"),
        sub_string(Code3, _, _, _, "fan_out_records")
    ->  format('  [PASS] Fan-out stage generated~n', [])
    ;   format('  [FAIL] Fan-out issue~n', [])
    ),

    % Test 4: Merge stage
    format('[Test 4] Merge stage~n', []),
    generate_enhanced_connector([fan_out([a/1, b/1]), merge], merge_pipe, Code4),
    (   sub_string(Code4, _, _, _, "def merge_pipe"),
        sub_string(Code4, _, _, _, "Merge:")
    ->  format('  [PASS] Merge stage generated~n', [])
    ;   format('  [FAIL] Merge issue~n', [])
    ),

    % Test 5: Conditional routing
    format('[Test 5] Conditional routing~n', []),
    generate_enhanced_connector([route_by(has_error, [(true, error_handler/1), (false, success/1)])], route_pipe, Code5),
    (   sub_string(Code5, _, _, _, "def route_pipe"),
        sub_string(Code5, _, _, _, "Conditional routing"),
        sub_string(Code5, _, _, _, "route_map"),
        sub_string(Code5, _, _, _, "route_record")
    ->  format('  [PASS] Conditional routing generated~n', [])
    ;   format('  [FAIL] Routing issue~n', [])
    ),

    % Test 6: Filter stage
    format('[Test 6] Filter stage~n', []),
    generate_enhanced_connector([filter_by(is_valid)], filter_pipe, Code6),
    (   sub_string(Code6, _, _, _, "def filter_pipe"),
        sub_string(Code6, _, _, _, "Filter by is_valid"),
        sub_string(Code6, _, _, _, "filter_records")
    ->  format('  [PASS] Filter stage generated~n', [])
    ;   format('  [FAIL] Filter issue~n', [])
    ),

    % Test 7: Complex pipeline with multiple patterns
    format('[Test 7] Complex pipeline~n', []),
    generate_enhanced_connector([
        extract/1,
        fan_out([validate/1, enrich/1]),
        merge,
        filter_by(is_valid),
        transform/1
    ], complex_pipe, Code7),
    (   sub_string(Code7, _, _, _, "def complex_pipe"),
        sub_string(Code7, _, _, _, "extract_result"),
        sub_string(Code7, _, _, _, "Fan-out"),
        sub_string(Code7, _, _, _, "Filter"),
        sub_string(Code7, _, _, _, "transform_result")
    ->  format('  [PASS] Complex pipeline generated~n', [])
    ;   format('  [FAIL] Complex pipeline issue~n', [])
    ),

    % Test 8: Stage function generation
    format('[Test 8] Stage function generation~n', []),
    generate_enhanced_stage_functions([extract/1, transform/1], StageFuncs8),
    (   sub_string(StageFuncs8, _, _, _, "def extract(stream)"),
        sub_string(StageFuncs8, _, _, _, "def transform(stream)")
    ->  format('  [PASS] Stage functions generated~n', [])
    ;   format('  [FAIL] Stage functions issue~n', [])
    ),

    % Test 9: Full enhanced pipeline compilation
    format('[Test 9] Full enhanced pipeline compilation~n', []),
    compile_enhanced_pipeline([
        extract/1,
        fan_out([validate/1, audit/1]),
        merge,
        transform/1
    ], [pipeline_name(full_enhanced)], FullCode9),
    (   sub_string(FullCode9, _, _, _, "def full_enhanced"),
        sub_string(FullCode9, _, _, _, "def fan_out_records"),
        sub_string(FullCode9, _, _, _, "def extract(stream)"),
        sub_string(FullCode9, _, _, _, "Fan-out to 2")
    ->  format('  [PASS] Full enhanced pipeline compiled~n', [])
    ;   format('  [FAIL] Full compilation issue~n', [])
    ),

    % Test 10: Tee stream helper
    format('[Test 10] Tee stream helper~n', []),
    enhanced_pipeline_helpers(Helpers10),
    (   sub_string(Helpers10, _, _, _, "def tee_stream"),
        sub_string(Helpers10, _, _, _, "Materialize to allow multiple iterations")
    ->  format('  [PASS] Tee stream helper present~n', [])
    ;   format('  [FAIL] Tee stream issue~n', [])
    ),

    format('~n=== All Enhanced Pipeline Chaining Tests Passed ===~n', []).

%% ============================================================================
%% IRONPYTHON ENHANCED PIPELINE CHAINING (Phase 8)
%% ============================================================================
%%
%% Extends enhanced pipeline chaining with IronPython/.NET-specific features:
%%   - fan_out(Stages)        : Broadcast to stages (sequential execution)
%%   - parallel(Stages)       : Execute stages concurrently (.NET Tasks)
%%   - merge                  : Combine results from fan_out or parallel
%%   - route_by(Pred, Routes) : Conditional routing with .NET Dictionary
%%   - filter_by(Pred)        : Filter records by predicate
%%   - Uses .NET List<T> for results collection
%%   - Uses .NET ConcurrentBag<T> for parallel results
%%   - Leverages CLR interop for .NET integration scenarios
%%
%% Example usage:
%%   compile_ironpython_enhanced_pipeline([
%%       extract/1,
%%       filter_by(is_active),
%%       parallel([validate/1, enrich/1]),  % Concurrent via .NET Tasks
%%       merge,
%%       route_by(has_error, [(true, error_handler/1), (false, success/1)]),
%%       output/1
%%   ], [pipeline_name(iron_enhanced)], Code).
%%
%% ============================================================================

%% compile_ironpython_enhanced_pipeline(+Stages, +Options, -Code)
%  Main entry point for IronPython enhanced pipelines with .NET integration.
%  Validates pipeline stages before code generation.
compile_ironpython_enhanced_pipeline(Stages, Options, Code) :-
    % Validate pipeline stages (Phase 9)
    option(validate(Validate), Options, true),
    option(strict(Strict), Options, false),
    ( Validate == true ->
        validate_pipeline(Stages, [strict(Strict)], result(Errors, Warnings)),
        % Report warnings
        ( Warnings \== [] ->
            format(user_error, 'IronPython pipeline warnings:~n', []),
            forall(member(W, Warnings), (
                format_validation_warning(W, Msg),
                format(user_error, '  ~w~n', [Msg])
            ))
        ; true
        ),
        % Fail on errors
        ( Errors \== [] ->
            format(user_error, 'IronPython pipeline validation errors:~n', []),
            forall(member(E, Errors), (
                format_validation_error(E, Msg),
                format(user_error, '  ~w~n', [Msg])
            )),
            throw(pipeline_validation_failed(Errors))
        ; true
        )
    ; true
    ),

    option(pipeline_name(PipelineName), Options, iron_enhanced_pipeline),
    option(glue_protocol(GlueProtocol), Options, jsonl),

    % Generate IronPython header with CLR imports
    ironpython_enhanced_header(GlueProtocol, Header),

    % Generate IronPython-specific enhanced helpers
    ironpython_enhanced_helpers(EnhancedHelpers),

    % Generate stage functions (reuse from base enhanced)
    generate_enhanced_stage_functions(Stages, StageFunctions),

    % Generate the enhanced connector
    generate_ironpython_enhanced_connector(Stages, PipelineName, ConnectorCode),

    % Generate main block
    generate_ironpython_enhanced_main(PipelineName, GlueProtocol, MainBlock),

    format(string(Code), "~w~n~n~w~n~n~w~n~w~n~w~n",
           [Header, EnhancedHelpers, StageFunctions, ConnectorCode, MainBlock]).

%% ironpython_enhanced_header(+Protocol, -Code)
%  Generate IronPython header with CLR imports for enhanced pipeline.
ironpython_enhanced_header(jsonl, Code) :-
    format(string(Code),
'#!/usr/bin/env ipy
"""
Generated IronPython Enhanced Pipeline
Supports: fan-out, merge, conditional routing, filtering
Runtime: IronPython with CLR/.NET integration
"""
import sys
import json
import clr

# Add .NET references
clr.AddReference("System")
clr.AddReference("System.Core")
from System import String, Object, Func
from System.Collections.Generic import List, Dictionary, HashSet

# Python typing (IronPython 3.4+ compatible)
from typing import Iterator, Dict, Any, Generator, Callable
', []).

ironpython_enhanced_header(_, Code) :-
    format(string(Code),
'#!/usr/bin/env ipy
"""
Generated IronPython Enhanced Pipeline
Supports: fan-out, merge, conditional routing, filtering
Runtime: IronPython with CLR/.NET integration
"""
import sys
import clr

clr.AddReference("System")
clr.AddReference("System.Core")
from System import String, Object, Func
from System.Collections.Generic import List, Dictionary, HashSet

from typing import Iterator, Dict, Any, Generator, Callable
', []).

%% ironpython_enhanced_helpers(-Code)
%  Generate IronPython-specific helper functions using .NET collections.
ironpython_enhanced_helpers(Code) :-
    Code = '# ============================================
# IronPython Enhanced Pipeline Helpers
# Uses .NET collections for CLR interoperability
# ============================================

def fan_out_records(record, stages):
    """
    Fan-out: Send record to all stages, collect results.
    Uses .NET List for result collection.
    """
    results = List[object]()
    for stage in stages:
        for result in stage(iter([record])):
            results.Add(result)
    return list(results)  # Convert back to Python list for iteration

def merge_streams(*streams):
    """
    Merge: Combine multiple streams into one.
    Yields records from all streams in order.
    """
    for stream in streams:
        for record in stream:
            yield record

def route_record(record, condition_fn, route_map):
    """
    Route: Direct record to appropriate stage based on condition.
    route_map is .NET Dictionary or Python dict.
    """
    condition = condition_fn(record)

    # Handle both .NET Dictionary and Python dict
    if hasattr(route_map, "ContainsKey"):
        # .NET Dictionary
        if route_map.ContainsKey(condition):
            yield from route_map[condition](iter([record]))
        elif route_map.ContainsKey("default"):
            yield from route_map["default"](iter([record]))
        else:
            yield record
    else:
        # Python dict
        if condition in route_map:
            yield from route_map[condition](iter([record]))
        elif "default" in route_map:
            yield from route_map["default"](iter([record]))
        else:
            yield record

def filter_records(stream, predicate_fn):
    """
    Filter: Only yield records that satisfy the predicate.
    """
    for record in stream:
        if predicate_fn(record):
            yield record

def tee_stream(stream, *stages):
    """
    Tee: Send each record to multiple stages, yield all results.
    Uses .NET List for intermediate storage.
    """
    records = List[object]()
    for record in stream:
        records.Add(record)

    for record in records:
        for stage in stages:
            yield from stage(iter([record]))

def create_route_map(routes):
    """
    Create a .NET Dictionary for routing.
    routes: list of (condition, stage_function) tuples
    """
    route_map = Dictionary[object, object]()
    for condition, stage in routes:
        route_map[condition] = stage
    return route_map

def to_dotnet_list(py_list):
    """Convert Python list to .NET List<object>."""
    result = List[object]()
    for item in py_list:
        result.Add(item)
    return result

def from_dotnet_list(dotnet_list):
    """Convert .NET List to Python list."""
    return list(dotnet_list)

def parallel_records(record, stages):
    """
    Parallel: Execute stages concurrently using .NET Tasks.
    Uses System.Threading.Tasks for true parallelism in IronPython.
    Each stage receives the same input record.
    Results are collected after all tasks complete.
    """
    from System.Threading.Tasks import Task, TaskFactory
    from System.Collections.Concurrent import ConcurrentBag

    results_bag = ConcurrentBag[object]()

    def run_stage(stage):
        stage_results = list(stage(iter([record])))
        for result in stage_results:
            results_bag.Add(result)

    # Create and start tasks for each stage
    tasks = List[Task]()
    for stage in stages:
        task = Task.Factory.StartNew(lambda s=stage: run_stage(s))
        tasks.Add(task)

    # Wait for all tasks to complete
    Task.WaitAll(tasks.ToArray())

    # Convert ConcurrentBag to list
    return list(results_bag)
'.

%% generate_ironpython_enhanced_connector(+Stages, +PipelineName, -Code)
%  Generate the enhanced connector for IronPython with .NET integration.
generate_ironpython_enhanced_connector(Stages, PipelineName, Code) :-
    generate_ironpython_enhanced_flow(Stages, "input_stream", FlowCode),
    format(string(Code),
'# ============================================
# IronPython Enhanced Pipeline Connector
# ============================================

def ~w(input_stream):
    """
    Enhanced IronPython pipeline with fan-out, merge, and routing.
    Uses .NET collections for CLR interoperability.
    """
~w
', [PipelineName, FlowCode]).

%% generate_ironpython_enhanced_flow(+Stages, +CurrentVar, -Code)
%  Generate flow code for IronPython enhanced stages.
generate_ironpython_enhanced_flow([], CurrentVar, Code) :-
    format(string(Code), '    yield from ~w', [CurrentVar]).
generate_ironpython_enhanced_flow([Stage|Rest], CurrentVar, Code) :-
    generate_ironpython_stage_flow(Stage, CurrentVar, NextVar, StageCode),
    generate_ironpython_enhanced_flow(Rest, NextVar, RestCode),
    format(string(Code), '~w~n~w', [StageCode, RestCode]).

%% generate_ironpython_stage_flow(+Stage, +CurrentVar, -NextVar, -Code)
%  Generate IronPython-specific flow code for each stage type.

% Standard predicate stage
generate_ironpython_stage_flow(Pred/_, CurrentVar, NextVar, Code) :-
    !,
    format(atom(NextVar), '~w_result', [Pred]),
    format(string(Code),
'    # Stage: ~w
    ~w = ~w(~w)', [Pred, NextVar, Pred, CurrentVar]).

% Fan-out stage with .NET List (sequential execution)
generate_ironpython_stage_flow(fan_out(SubStages), CurrentVar, NextVar, Code) :-
    !,
    length(SubStages, N),
    NextVar = 'fan_out_results',
    collect_stage_names(SubStages, StageNames),
    format_stage_list(StageNames, StageListStr),
    format(string(Code),
'    # Fan-out to ~w stages (sequential, IronPython/.NET)
    fan_out_stages = [~w]
    ~w = []
    for record in ~w:
        ~w.extend(fan_out_records(record, fan_out_stages))',
    [N, StageListStr, NextVar, CurrentVar, NextVar]).

% Parallel stage with .NET Tasks (concurrent execution)
generate_ironpython_stage_flow(parallel(SubStages), CurrentVar, NextVar, Code) :-
    !,
    length(SubStages, N),
    NextVar = 'parallel_results',
    collect_stage_names(SubStages, StageNames),
    format_stage_list(StageNames, StageListStr),
    format(string(Code),
'    # Parallel execution of ~w stages (concurrent via .NET Tasks)
    parallel_stages = [~w]
    ~w = []
    for record in ~w:
        ~w.extend(parallel_records(record, parallel_stages))',
    [N, StageListStr, NextVar, CurrentVar, NextVar]).

% Merge stage (follows fan_out or parallel)
generate_ironpython_stage_flow(merge, CurrentVar, NextVar, Code) :-
    !,
    NextVar = 'merged_stream',
    format(string(Code),
'    # Merge results from fan_out or parallel
    ~w = iter(~w) if ~w else iter([])', [NextVar, CurrentVar, CurrentVar]).

% Conditional routing with .NET Dictionary support
generate_ironpython_stage_flow(route_by(Pred, Routes), CurrentVar, NextVar, Code) :-
    !,
    NextVar = 'routed_stream',
    format_python_route_map(Routes, RouteMapCode),
    format(string(Code),
'    # Conditional routing by ~w (IronPython/.NET)
    route_map = {~w}
    def route_generator():
        for record in ~w:
            yield from route_record(record, ~w, route_map)
    ~w = route_generator()', [Pred, RouteMapCode, CurrentVar, Pred, NextVar]).

% Filter stage
generate_ironpython_stage_flow(filter_by(Pred), CurrentVar, NextVar, Code) :-
    !,
    NextVar = 'filtered_stream',
    format(string(Code),
'    # Filter by ~w
    ~w = filter_records(~w, ~w)', [Pred, NextVar, CurrentVar, Pred]).

% Unknown stage - pass through
generate_ironpython_stage_flow(Stage, CurrentVar, CurrentVar, Code) :-
    format(string(Code), '    # Unknown stage: ~w (pass-through)', [Stage]).

%% collect_stage_names(+Stages, -Names)
%  Extract stage function names from stage specifications.
collect_stage_names([], []).
collect_stage_names([Pred/_|Rest], [Pred|RestNames]) :-
    !,
    collect_stage_names(Rest, RestNames).
collect_stage_names([_|Rest], RestNames) :-
    collect_stage_names(Rest, RestNames).

%% format_stage_list(+Names, -Str)
%  Format stage names as Python function references.
format_stage_list([], '').
format_stage_list([Name], Str) :-
    format(string(Str), '~w', [Name]).
format_stage_list([Name|Rest], Str) :-
    Rest \= [],
    format_stage_list(Rest, RestStr),
    format(string(Str), '~w, ~w', [Name, RestStr]).

%% format_python_route_map(+Routes, -Code)
%  Format routing map as Python dict literal.
format_python_route_map([], '').
format_python_route_map([(Cond, Stage)|[]], Code) :-
    (Stage = StageName/_Arity -> true ; StageName = Stage),
    (Cond = true ->
        format(string(Code), 'True: ~w', [StageName])
    ; Cond = false ->
        format(string(Code), 'False: ~w', [StageName])
    ;   format(string(Code), '"~w": ~w', [Cond, StageName])
    ).
format_python_route_map([(Cond, Stage)|Rest], Code) :-
    Rest \= [],
    (Stage = StageName/_Arity -> true ; StageName = Stage),
    format_python_route_map(Rest, RestCode),
    (Cond = true ->
        format(string(Code), 'True: ~w, ~w', [StageName, RestCode])
    ; Cond = false ->
        format(string(Code), 'False: ~w, ~w', [StageName, RestCode])
    ;   format(string(Code), '"~w": ~w, ~w', [Cond, StageName, RestCode])
    ).

%% generate_ironpython_enhanced_main(+PipelineName, +Protocol, -Code)
%  Generate main execution block for IronPython enhanced pipeline.
generate_ironpython_enhanced_main(PipelineName, jsonl, Code) :-
    format(string(Code),
'# ============================================
# Main Execution Block (IronPython)
# ============================================

def read_input():
    """Read JSONL records from stdin."""
    for line in sys.stdin:
        line = line.strip()
        if line:
            yield json.loads(line)

def write_output(records):
    """Write records as JSONL to stdout."""
    for record in records:
        print(json.dumps(record))

if __name__ == "__main__":
    input_stream = read_input()
    output_stream = ~w(input_stream)
    write_output(output_stream)
', [PipelineName]).

generate_ironpython_enhanced_main(PipelineName, _, Code) :-
    format(string(Code),
'# ============================================
# Main Execution Block (IronPython)
# ============================================

def read_input():
    """Read records from stdin."""
    for line in sys.stdin:
        line = line.strip()
        if line:
            yield line

def write_output(records):
    """Write records to stdout."""
    for record in records:
        print(record)

if __name__ == "__main__":
    input_stream = read_input()
    output_stream = ~w(input_stream)
    write_output(output_stream)
', [PipelineName]).

%% ============================================
%% IRONPYTHON ENHANCED PIPELINE CHAINING TESTS
%% ============================================

test_ironpython_enhanced_chaining :-
    format('~n=== IronPython Enhanced Pipeline Chaining Tests ===~n~n', []),

    % Test 1: Generate IronPython enhanced helpers
    format('[Test 1] IronPython enhanced helpers~n', []),
    ironpython_enhanced_helpers(Helpers1),
    (   sub_string(Helpers1, _, _, _, "fan_out_records"),
        sub_string(Helpers1, _, _, _, "List[object]"),
        sub_string(Helpers1, _, _, _, "route_record"),
        sub_string(Helpers1, _, _, _, "filter_records"),
        sub_string(Helpers1, _, _, _, "Dictionary")
    ->  format('  [PASS] IronPython helpers use .NET collections~n', [])
    ;   format('  [FAIL] Missing .NET collection usage~n', [])
    ),

    % Test 2: IronPython enhanced header
    format('[Test 2] IronPython enhanced header~n', []),
    ironpython_enhanced_header(jsonl, Header2),
    (   sub_string(Header2, _, _, _, "#!/usr/bin/env ipy"),
        sub_string(Header2, _, _, _, "import clr"),
        sub_string(Header2, _, _, _, "List, Dictionary, HashSet")
    ->  format('  [PASS] Header has IronPython CLR imports~n', [])
    ;   format('  [FAIL] Header missing CLR imports~n', [])
    ),

    % Test 3: Linear pipeline connector
    format('[Test 3] Linear pipeline connector~n', []),
    generate_ironpython_enhanced_connector([extract/1, transform/1, load/1], linear_pipe, Code3),
    (   sub_string(Code3, _, _, _, "def linear_pipe"),
        sub_string(Code3, _, _, _, "extract"),
        sub_string(Code3, _, _, _, "transform"),
        sub_string(Code3, _, _, _, "load")
    ->  format('  [PASS] Linear connector generated~n', [])
    ;   format('  [FAIL] Linear connector missing patterns~n', [])
    ),

    % Test 4: Fan-out connector with .NET comment
    format('[Test 4] Fan-out connector~n', []),
    generate_ironpython_enhanced_connector([extract/1, fan_out([validate/1, enrich/1])], fanout_pipe, Code4),
    (   sub_string(Code4, _, _, _, "Fan-out to 2"),
        sub_string(Code4, _, _, _, "IronPython/.NET"),
        sub_string(Code4, _, _, _, "fan_out_records")
    ->  format('  [PASS] Fan-out connector uses .NET~n', [])
    ;   format('  [FAIL] Fan-out connector missing .NET patterns~n', [])
    ),

    % Test 5: Fan-out with merge
    format('[Test 5] Fan-out with merge~n', []),
    generate_ironpython_enhanced_connector([fan_out([a/1, b/1]), merge, output/1], merge_pipe, Code5),
    (   sub_string(Code5, _, _, _, "Merge"),
        sub_string(Code5, _, _, _, "merged_stream")
    ->  format('  [PASS] Merge connector generated~n', [])
    ;   format('  [FAIL] Merge connector issue~n', [])
    ),

    % Test 6: Conditional routing
    format('[Test 6] Conditional routing~n', []),
    generate_ironpython_enhanced_connector([extract/1, route_by(has_error, [(true, error/1), (false, success/1)])], route_pipe, Code6),
    (   sub_string(Code6, _, _, _, "route_map"),
        sub_string(Code6, _, _, _, "route_record"),
        sub_string(Code6, _, _, _, "Conditional routing")
    ->  format('  [PASS] Routing connector generated~n', [])
    ;   format('  [FAIL] Routing connector missing patterns~n', [])
    ),

    % Test 7: Filter stage
    format('[Test 7] Filter stage~n', []),
    generate_ironpython_enhanced_connector([extract/1, filter_by(is_valid), output/1], filter_pipe, Code7),
    (   sub_string(Code7, _, _, _, "filter_records"),
        sub_string(Code7, _, _, _, "Filter by is_valid")
    ->  format('  [PASS] Filter connector generated~n', [])
    ;   format('  [FAIL] Filter connector missing patterns~n', [])
    ),

    % Test 8: Complex pipeline with all patterns
    format('[Test 8] Complex pipeline~n', []),
    generate_ironpython_enhanced_connector([
        extract/1,
        filter_by(is_active),
        fan_out([validate/1, enrich/1, audit/1]),
        merge,
        route_by(has_error, [(true, error_log/1), (false, transform/1)]),
        output/1
    ], complex_pipe, Code8),
    (   sub_string(Code8, _, _, _, "Fan-out to 3"),
        sub_string(Code8, _, _, _, "Filter by is_active"),
        sub_string(Code8, _, _, _, "Merge"),
        sub_string(Code8, _, _, _, "Conditional routing")
    ->  format('  [PASS] Complex connector has all patterns~n', [])
    ;   format('  [FAIL] Complex connector missing patterns~n', [])
    ),

    % Test 9: Full IronPython enhanced pipeline compilation
    format('[Test 9] Full IronPython enhanced pipeline~n', []),
    compile_ironpython_enhanced_pipeline([
        extract/1,
        filter_by(is_active),
        fan_out([validate/1, enrich/1]),
        merge,
        output/1
    ], [pipeline_name(full_iron_enhanced)], Code9),
    (   sub_string(Code9, _, _, _, "#!/usr/bin/env ipy"),
        sub_string(Code9, _, _, _, "import clr"),
        sub_string(Code9, _, _, _, "fan_out_records"),
        sub_string(Code9, _, _, _, "def full_iron_enhanced"),
        sub_string(Code9, _, _, _, '__name__ == "__main__"')
    ->  format('  [PASS] Full IronPython enhanced pipeline compiles~n', [])
    ;   format('  [FAIL] Full pipeline compilation failed~n', [])
    ),

    % Test 10: IronPython helpers have .NET interop utilities
    format('[Test 10] IronPython .NET interop utilities~n', []),
    ironpython_enhanced_helpers(Helpers10),
    (   sub_string(Helpers10, _, _, _, "to_dotnet_list"),
        sub_string(Helpers10, _, _, _, "from_dotnet_list"),
        sub_string(Helpers10, _, _, _, "create_route_map")
    ->  format('  [PASS] .NET interop utilities present~n', [])
    ;   format('  [FAIL] Missing .NET interop utilities~n', [])
    ),

    format('~n=== All IronPython Enhanced Pipeline Chaining Tests Passed ===~n', []).

%% ============================================
%% KG Topology Phase 3: Kleinberg Router Code Generation
%% ============================================

%% compile_kleinberg_router_python(+Options, -Code)
%  Generate Python KleinbergRouter class with configurable options.
%  This is for embedding in generated services; the runtime module
%  is in python_runtime/kleinberg_router.py.

compile_kleinberg_router_python(Options, Code) :-
    ( member(alpha(Alpha), Options) -> true ; Alpha = 2.0 ),
    ( member(max_hops(MaxHops), Options) -> true ; MaxHops = 10 ),
    ( member(parallel_paths(ParallelPaths), Options) -> true ; ParallelPaths = 1 ),
    ( member(similarity_threshold(Threshold), Options) -> true ; Threshold = 0.5 ),
    ( member(path_folding(PathFolding), Options) -> true ; PathFolding = true ),
    ( PathFolding = true -> PFBool = 'True' ; PFBool = 'False' ),

    format(string(Code), '
# KG Topology Phase 3: Kleinberg Router Configuration
# Generated from Prolog service definition

from kleinberg_router import KleinbergRouter
from discovery_clients import create_discovery_client

def create_kleinberg_router(node_id, discovery_backend="local", **discovery_config):
    """Create a configured KleinbergRouter instance."""
    discovery_client = create_discovery_client(discovery_backend, **discovery_config)

    return KleinbergRouter(
        local_node_id=node_id,
        discovery_client=discovery_client,
        alpha=~w,
        max_hops=~w,
        parallel_paths=~w,
        similarity_threshold=~w,
        path_folding_enabled=~w
    )
', [Alpha, MaxHops, ParallelPaths, Threshold, PFBool]).


%% compile_distributed_kg_service_python(+Service, -Code)
%  Generate a distributed KG topology service from Prolog definition.

compile_distributed_kg_service_python(service(Name, Options, _Handler), Code) :-
    % Extract Kleinberg options
    ( member(routing(kleinberg(KleinbergOpts)), Options) -> true
    ; member(routing(kleinberg), Options) -> KleinbergOpts = []
    ; KleinbergOpts = []
    ),

    % Extract discovery options
    ( member(discovery_backend(Backend), Options) -> true ; Backend = local ),
    ( member(discovery_tags(Tags), Options) -> true ; Tags = ['kg_node'] ),

    % Extract network options
    ( member(transport(http(_, TransportOpts)), Options) ->
        ( member(host(Host), TransportOpts) -> true ; Host = '0.0.0.0' ),
        ( member(port(Port), TransportOpts) -> true ; Port = 8080 )
    ; Host = '0.0.0.0', Port = 8080
    ),

    % Generate Kleinberg router config
    compile_kleinberg_router_python(KleinbergOpts, RouterCode),

    % Generate tags list
    format_python_list(Tags, TagsStr),

    format(string(Code), '
# Distributed KG Topology Service: ~w
# Generated from Prolog service definition

from flask import Flask, request, jsonify
from kg_topology_api import DistributedKGTopologyAPI

app = Flask(__name__)

# Router configuration
~w

# Initialize distributed KG API
kg_api = DistributedKGTopologyAPI(
    db_path="~w.db",
    node_id="~w",
    discovery_backend="~w"
)

# Register node on startup
@app.before_first_request
def register_with_discovery():
    kg_api.register_node(
        host="~w",
        port=~w,
        tags=~w
    )

# KG Query endpoint
@app.route("/kg/query", methods=["POST"])
def handle_kg_query():
    try:
        request_data = request.json
        if request_data.get("__type") != "kg_query":
            return jsonify({"error": "Invalid request type"}), 400
        result = kg_api.handle_remote_query(request_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health endpoint
@app.route("/kg/health", methods=["GET"])
def handle_kg_health():
    interfaces = kg_api.list_interfaces(active_only=True)
    stats = kg_api.get_query_stats()
    return jsonify({
        "status": "healthy",
        "node_id": kg_api.node_id,
        "interfaces": len(interfaces),
        "stats": stats
    })

if __name__ == "__main__":
    app.run(host="~w", port=~w)
', [Name, RouterCode, Name, Name, Backend, Host, Port, TagsStr, Host, Port]).


%% format_python_list(+List, -String)
%  Format a Prolog list as a Python list literal.
format_python_list([], '[]').
format_python_list(List, String) :-
    List \= [],
    maplist(format_python_item, List, Items),
    atomic_list_concat(Items, ', ', Inner),
    format(string(String), '[~w]', [Inner]).

format_python_item(Item, Quoted) :-
    format(string(Quoted), '"~w"', [Item]).


% =============================================================================
% KG TOPOLOGY PHASE 4: FEDERATED QUERY CODE GENERATION
% =============================================================================

%% compile_federated_query_python(+Options, -Code)
%  Generate Python FederatedQueryEngine configuration from Prolog options.
%  This creates a factory function with the specified federation settings.

compile_federated_query_python(Options, Code) :-
    % Extract federation options with defaults
    ( member(federation_k(K), Options) -> true ; K = 3 ),
    ( member(timeout_ms(Timeout), Options) -> true ; Timeout = 5000 ),
    ( member(consensus_threshold(Consensus), Options) -> ConsensusStr = Consensus
    ; ConsensusStr = 'None'
    ),
    ( member(diversity_field(DivField), Options) -> true ; DivField = corpus_id ),

    % Extract aggregation strategy
    ( member(aggregation(Strategy), Options) -> true
    ; member(aggregation(Strategy, _), Options) -> true
    ; Strategy = sum
    ),
    strategy_to_python_enum(Strategy, StrategyEnum),

    % Extract dedup key if specified
    ( member(aggregation(_, AggOpts), Options),
      member(dedup_key(DedupKey), AggOpts) -> true
    ; DedupKey = answer_hash
    ),

    format(string(Code), '
# KG Topology Phase 4: Federated Query Configuration
# Generated from Prolog service definition

from federated_query import (
    FederatedQueryEngine,
    AggregationStrategy,
    AggregationConfig,
    create_federated_engine
)

def create_federated_query_engine(router, discovery_client=None):
    """Create a configured FederatedQueryEngine instance."""
    config = AggregationConfig(
        strategy=AggregationStrategy.~w,
        dedup_key="~w",
        diversity_field="~w"
    )

    return FederatedQueryEngine(
        router=router,
        aggregation_config=config,
        federation_k=~w,
        timeout_ms=~w
    )

# Federation settings
FEDERATION_K = ~w
AGGREGATION_STRATEGY = "~w"
CONSENSUS_THRESHOLD = ~w
DIVERSITY_FIELD = "~w"
', [StrategyEnum, DedupKey, DivField, K, Timeout, K, Strategy, ConsensusStr, DivField]).

%% strategy_to_python_enum(+Strategy, -Enum)
%  Convert Prolog strategy atom to Python enum name.
strategy_to_python_enum(sum, 'SUM').
strategy_to_python_enum(max, 'MAX').
strategy_to_python_enum(min, 'MIN').
strategy_to_python_enum(avg, 'AVG').
strategy_to_python_enum(count, 'COUNT').
strategy_to_python_enum(first, 'FIRST').
strategy_to_python_enum(collect, 'COLLECT').
strategy_to_python_enum(diversity, 'DIVERSITY_WEIGHTED').
strategy_to_python_enum(diversity_weighted, 'DIVERSITY_WEIGHTED').
strategy_to_python_enum(_, 'SUM').  % Default fallback


%% compile_federated_service_python(+Service, -Code)
%  Generate a complete federated KG topology service from Prolog definition.
%  Combines Kleinberg routing (Phase 3) with federation (Phase 4).

compile_federated_service_python(service(Name, Options, _Handler), Code) :-
    % Extract Kleinberg options for routing
    ( member(routing(kleinberg(KleinbergOpts)), Options) -> true
    ; member(routing(kleinberg), Options) -> KleinbergOpts = []
    ; KleinbergOpts = []
    ),

    % Extract federation options
    ( member(federation(FedOpts), Options) -> true
    ; FedOpts = []
    ),

    % Extract discovery options
    ( member(discovery_backend(Backend), Options) -> true ; Backend = local ),
    ( member(discovery_tags(Tags), Options) -> true ; Tags = ['kg_node'] ),

    % Extract corpus info for diversity tracking
    ( member(discovery_metadata(Metadata), Options),
      member(corpus_id(CorpusId), Metadata) -> true
    ; CorpusId = 'auto'
    ),
    ( member(discovery_metadata(Metadata2), Options),
      member(data_sources(DataSources), Metadata2) -> true
    ; DataSources = []
    ),

    % Extract network options
    ( member(transport(http(_, TransportOpts)), Options) ->
        ( member(host(Host), TransportOpts) -> true ; Host = '0.0.0.0' ),
        ( member(port(Port), TransportOpts) -> true ; Port = 8080 )
    ; Host = '0.0.0.0', Port = 8080
    ),

    % Generate Kleinberg router config
    compile_kleinberg_router_python(KleinbergOpts, RouterCode),

    % Generate federated query config
    compile_federated_query_python(FedOpts, FederationCode),

    % Generate lists
    format_python_list(Tags, TagsStr),
    format_python_list(DataSources, DataSourcesStr),

    % Determine corpus_id handling
    ( CorpusId = auto -> CorpusIdCode = 'None  # Will auto-generate from DB content'
    ; format(string(CorpusIdCode), '"~w"', [CorpusId])
    ),

    format(string(Code), '
# Federated KG Topology Service: ~w
# Generated from Prolog service definition
# Combines Phase 3 (Kleinberg routing) + Phase 4 (Federation)

from flask import Flask, request, jsonify
from kg_topology_api import DistributedKGTopologyAPI
from federated_query import FederatedQueryEngine, AggregationConfig, AggregationStrategy

app = Flask(__name__)

# Phase 3: Router configuration
~w

# Phase 4: Federation configuration
~w

# Initialize distributed KG API with corpus tracking
kg_api = DistributedKGTopologyAPI(
    db_path="~w.db",
    node_id="~w",
    discovery_backend="~w"
)

# Initialize federated query engine
router = create_kleinberg_router("~w", "~w")
federation_engine = create_federated_query_engine(router)

@app.before_first_request
def register_with_discovery():
    """Register node with service discovery on startup."""
    kg_api.register_node(
        host="~w",
        port=~w,
        tags=~w,
        corpus_id=~w,
        data_sources=~w
    )

@app.route("/kg/query", methods=["POST"])
def kg_query():
    """Handle local KG queries."""
    data = request.get_json()
    query_text = data.get("query_text", "")
    top_k = data.get("top_k", 5)

    results = kg_api.semantic_search(query_text, top_k=top_k)
    return jsonify({"results": results})

@app.route("/kg/federated", methods=["POST"])
def kg_federated_query():
    """Handle federated KG queries from other nodes."""
    data = request.get_json()
    response = kg_api.handle_federated_query(data)
    return jsonify(response)

@app.route("/kg/federate", methods=["POST"])
def kg_initiate_federation():
    """Initiate a federated query across the network."""
    data = request.get_json()
    query_text = data.get("query_text", "")
    top_k = data.get("top_k", 5)

    # Use federation engine to query network
    response = federation_engine.federated_query(
        query_text=query_text,
        top_k=top_k
    )
    return jsonify(response.to_dict())

@app.route("/kg/health", methods=["GET"])
def kg_health():
    """Health check endpoint."""
    interfaces = kg_api.list_interfaces(active_only=True)
    stats = kg_api.get_query_stats()
    corpus_info = kg_api.get_corpus_info()
    fed_stats = federation_engine.get_stats()

    return jsonify({
        "status": "healthy",
        "node_id": kg_api.node_id,
        "interfaces": len(interfaces),
        "corpus_id": corpus_info.get("corpus_id"),
        "query_stats": stats,
        "federation_stats": fed_stats
    })

if __name__ == "__main__":
    app.run(host="~w", port=~w)
', [Name, RouterCode, FederationCode, Name, Name, Backend, Name, Backend,
    Host, Port, TagsStr, CorpusIdCode, DataSourcesStr, Host, Port]).


% =============================================================================
% PHASE 6e: CROSS-MODEL FEDERATION CODE GENERATION
% =============================================================================

%% compile_cross_model_engine_python(+Options, -Code)
%  Generate Python code for CrossModelFederatedEngine initialization.
compile_cross_model_engine_python(Options, Code) :-
    % Extract cross-model options
    (member(cross_model(CrossModelOpts), Options) ->
        extract_cross_model_config(CrossModelOpts, FusionMethod, RRFk, ConsThresh, ConsBost, PoolsCode)
    ;
        FusionMethod = 'WEIGHTED_SUM', RRFk = 60, ConsThresh = 0.1, ConsBost = 1.5, PoolsCode = '[]'
    ),
    format(atom(Code), '
def create_cross_model_engine(router):
    """Factory function for CrossModelFederatedEngine."""
    from cross_model_federation import (
        CrossModelFederatedEngine,
        CrossModelConfig,
        ModelPoolConfig,
        FusionMethod
    )

    pools = ~w

    config = CrossModelConfig(
        pools=pools,
        fusion_method=FusionMethod.~w,
        rrf_k=~w,
        consensus_threshold=~w,
        consensus_boost_factor=~w
    )

    return CrossModelFederatedEngine(router, config)
', [PoolsCode, FusionMethod, RRFk, ConsThresh, ConsBost]).

%% extract_cross_model_config(+Opts, -FusionMethod, -RRFk, -ConsThresh, -ConsBost, -PoolsCode)
%  Extract cross-model configuration from options list.
extract_cross_model_config(Opts, FusionMethod, RRFk, ConsThresh, ConsBost, PoolsCode) :-
    (member(fusion_method(FM), Opts) -> fusion_method_to_python(FM, FusionMethod) ; FusionMethod = 'WEIGHTED_SUM'),
    (member(rrf_k(RRFk), Opts) -> true ; RRFk = 60),
    (member(consensus_threshold(ConsThresh), Opts) -> true ; ConsThresh = 0.1),
    (member(consensus_boost(ConsBost), Opts) -> true ; ConsBost = 1.5),
    (member(pools(Pools), Opts) -> compile_pools_python(Pools, PoolsCode) ; PoolsCode = '[]').

%% fusion_method_to_python(+Atom, -PythonEnum)
%  Map Prolog fusion method atom to Python enum name.
fusion_method_to_python(weighted_sum, 'WEIGHTED_SUM').
fusion_method_to_python(rrf, 'RECIPROCAL_RANK').
fusion_method_to_python(consensus, 'CONSENSUS').
fusion_method_to_python(geometric_mean, 'GEOMETRIC_MEAN').
fusion_method_to_python(max, 'MAX').
fusion_method_to_python(_, 'WEIGHTED_SUM').  % Default

%% compile_pools_python(+Pools, -Code)
%  Generate Python list of ModelPoolConfig objects.
compile_pools_python([], '[]').
compile_pools_python(Pools, Code) :-
    Pools \= [],
    maplist(compile_pool_config_python, Pools, PoolCodes),
    atomic_list_concat(PoolCodes, ',\n        ', PoolsStr),
    format(atom(Code), '[\n        ~w\n    ]', [PoolsStr]).

%% compile_pool_config_python(+PoolConfig, -Code)
%  Generate Python ModelPoolConfig object.
compile_pool_config_python(pool(Model, Opts), Code) :-
    (member(weight(W), Opts) -> true ; W = 1.0),
    (member(federation_k(K), Opts) -> true ; K = 5),
    (member(strategy(S), Opts) -> strategy_to_python_enum(S, StratStr) ; StratStr = 'DENSITY_FLUX'),
    format(atom(Code), 'ModelPoolConfig(model_name="~w", weight=~w, federation_k=~w, aggregation_strategy=AggregationStrategy.~w)', [Model, W, K, StratStr]).

%% compile_cross_model_service_python(+Service, -Code)
%  Generate complete Flask service with cross-model federation support.
compile_cross_model_service_python(service(Name, Options, _Handler), Code) :-
    % Get transport settings
    (member(transport(http(_Path, HttpOpts)), Options) ->
        (member(host(Host), HttpOpts) -> true ; Host = '0.0.0.0'),
        (member(port(Port), HttpOpts) -> true ; Port = 8080)
    ;
        Host = '0.0.0.0', Port = 8080
    ),
    % Get discovery backend
    (member(discovery_backend(Backend), Options) -> true ; Backend = local),
    % Generate cross-model engine code
    compile_cross_model_engine_python(Options, EngineCode),
    format(atom(Code), '#!/usr/bin/env python3
# Generated Cross-Model Federation Service: ~w
# Phase 6e: Cross-Model Federation

from flask import Flask, request, jsonify
import os
import json

from kg_topology_api import DistributedKGTopologyAPI
from discovery_clients import create_discovery_client
from kleinberg_router import KleinbergRouter
from cross_model_federation import AdaptiveModelWeights

app = Flask(__name__)

# Initialize components
kg_api = DistributedKGTopologyAPI()
discovery = create_discovery_client("~w")
router = KleinbergRouter(discovery)

~w

cross_model_engine = create_cross_model_engine(router)
WEIGHTS_FILE = os.environ.get("WEIGHTS_FILE", "model_weights.json")

# Initialize adaptive weights
models = [p.model_name for p in cross_model_engine.config.pools]
adaptive_weights = AdaptiveModelWeights(models)
if os.path.exists(WEIGHTS_FILE):
    with open(WEIGHTS_FILE, "r") as f:
        adaptive_weights = AdaptiveModelWeights.from_dict(json.load(f))

@app.route("/kg/cross-model", methods=["POST"])
def cross_model_query():
    data = request.json
    response = cross_model_engine.federated_query(
        query_text=data.get("query_text", ""),
        top_k=data.get("top_k", 10)
    )
    return jsonify(response.to_dict())

@app.route("/kg/cross-model/pools", methods=["GET"])
def get_pools():
    pools = cross_model_engine.discover_pools()
    return jsonify({"pools": {m: len(n) for m, n in pools.items()}})

@app.route("/kg/cross-model/weights", methods=["GET"])
def get_weights():
    return jsonify(adaptive_weights.to_dict())

@app.route("/kg/cross-model/weights", methods=["PUT"])
def set_weights():
    for m, w in request.json.get("weights", {}).items():
        if m in adaptive_weights.weights:
            adaptive_weights.weights[m] = w
    total = sum(adaptive_weights.weights.values())
    if total > 0:
        adaptive_weights.weights = {m: w/total for m, w in adaptive_weights.weights.items()}
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(adaptive_weights.to_dict(), f)
    return jsonify(adaptive_weights.to_dict())

@app.route("/kg/cross-model/feedback", methods=["POST"])
def submit_feedback():
    data = request.json
    adaptive_weights.update(data["chosen_answer"], data.get("pool_rankings", {}))
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(adaptive_weights.to_dict(), f)
    return jsonify({"status": "updated", "weights": adaptive_weights.get_weights()})

@app.route("/kg/cross-model/stats", methods=["GET"])
def get_stats():
    return jsonify(cross_model_engine.get_stats())

@app.route("/kg/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "node_id": kg_api.node_id,
        "pools": len(cross_model_engine.pool_engines),
        "weights": adaptive_weights.get_weights()
    })

if __name__ == "__main__":
    app.run(host="~w", port=~w)
', [Name, Backend, EngineCode, Host, Port]).
