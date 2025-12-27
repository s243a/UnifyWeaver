% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% rpyc_glue.pl - Network-Based Python Communication via RPyC
%
% This module provides glue code for network-based RPC communication
% between Prolog and Python using RPyC (Remote Python Call).
%
% Unlike pipe-based communication, RPyC provides:
% - Live object proxies over the network
% - Bidirectional calls between client and server
% - Full module access on remote machines
% - SSH/SSL/unsecured security modes
%
% Unlike Janus (in-process), RPyC enables:
% - Remote machine access
% - Process isolation
% - Language-agnostic client connections
%
% Four Proxy Layers:
% - root: Direct exposed_ method access (simple RPC)
% - wrapped_root: Safe attribute access (prevent accidental execution)
% - auto_root: Automatic wrapping (general use)
% - smart_root: Local-class-aware wrapping (code generation)
%
% Requirements:
% - Python 3.8+ with rpyc package
% - Janus for Prolog-Python bridge (or external process)
% - For SSH: ssh command and optionally sshpass
% - For SSL: valid certificates
%
% Usage:
%   ?- rpyc_connect('localhost', [security(unsecured), acknowledge_risk(true)], Proxy).
%   ?- rpyc_call(Proxy, math, sqrt, [16], Result).
%   ?- rpyc_disconnect(Proxy).

:- module(rpyc_glue, [
    % Connection management
    rpyc_connect/3,           % +Host, +Options, -Proxy
    rpyc_disconnect/1,        % +Proxy
    rpyc_with_connection/3,   % +Host, +Options, :Goal

    % Remote execution (sync)
    rpyc_call/5,              % +Proxy, +Module, +Function, +Args, -Result
    rpyc_exec/3,              % +Proxy, +Code, -Namespace

    % Remote execution (async)
    rpyc_async_call/5,        % +Proxy, +Module, +Function, +Args, -AsyncResult
    rpyc_await/2,             % +AsyncResult, -Result
    rpyc_ready/1,             % +AsyncResult (succeeds if ready)

    % Module access
    rpyc_import/3,            % +Proxy, +ModuleName, -ModuleRef
    rpyc_get_module/3,        % +Proxy, +FullModuleName, -ModuleRef

    % Proxy layers
    rpyc_root/2,              % +Proxy, -RootProxy
    rpyc_wrapped_root/2,      % +Proxy, -WrappedRoot
    rpyc_auto_root/2,         % +Proxy, -AutoRoot
    rpyc_smart_root/2,        % +Proxy, -SmartRoot

    % Code generation
    generate_rpyc_client/3,   % +Predicates, +Options, -Code
    generate_rpyc_service/3,  % +Predicates, +Options, -ServiceCode
    generate_rpyc_server/2,   % +Options, -ServerScript

    % Transport registration
    rpyc_transport_type/1,

    % Testing
    test_rpyc_glue/0
]).

:- use_module(library(lists)).

% Conditionally load Janus if available
:- catch(use_module(library(janus)), _, true).

%% ============================================
%% Transport Registration
%% ============================================

%% rpyc_transport_type(-Type)
%  Register RPyC as a transport type.
rpyc_transport_type(rpyc).

%% ============================================
%% Connection Management
%% ============================================

%% rpyc_connect(+Host, +Options, -Proxy)
%  Connect to an RPyC server.
%
%  Options:
%    security(Mode) - One of: ssh, ssl, unsecured
%    remote_port(Port) - Server port (default: 18812)
%    acknowledge_risk(Bool) - Required for unsecured connections
%
%  SSH-specific options:
%    ssh_user(User) - SSH username
%    ssh_port(Port) - SSH port (default: 22)
%    ssh_key(Path) - Path to SSH key file
%
%  SSL-specific options:
%    keyfile(Path) - Client key file
%    certfile(Path) - Client certificate file
%    ca_certs(Path) - CA certificates file
%
%  Example:
%    ?- rpyc_connect('localhost', [security(unsecured), acknowledge_risk(true)], Proxy).
%
rpyc_connect(Host, Options, Proxy) :-
    atom(Host),
    is_list(Options),
    option(security(Security), Options, unsecured),
    validate_security_mode(Security, Options),
    create_bridge(Host, Security, Options, Bridge),
    connect_bridge(Bridge, Proxy).

%% validate_security_mode(+Security, +Options)
%  Validate that security requirements are met.
validate_security_mode(unsecured, Options) :-
    !,
    (   option(acknowledge_risk(true), Options)
    ->  true
    ;   throw(error(security_error('Unsecured connections require acknowledge_risk(true)'), _))
    ).
validate_security_mode(ssh, _) :- !.
validate_security_mode(ssl, _) :- !.
validate_security_mode(Mode, _) :-
    throw(error(domain_error(security_mode, Mode), _)).

%% create_bridge(+Host, +Security, +Options, -Bridge)
%  Create an RPyC bridge via Janus.
create_bridge(Host, Security, Options, Bridge) :-
    janus_available,
    !,
    janus_add_rpyc_path,
    atom_string(Host, HostStr),
    atom_string(Security, SecStr),
    build_py_bridge_call(HostStr, SecStr, Options, Bridge).
create_bridge(_, _, _, _) :-
    throw(error(janus_not_available, 'RPyC glue requires Janus for Python bridging')).

%% build_py_bridge_call(+HostStr, +SecStr, +Options, -Bridge)
%  Build the Python bridge call with options.
build_py_bridge_call(HostStr, SecStr, Options, Bridge) :-
    % Extract commonly used options
    option(remote_port(Port), Options, 18812),
    option(acknowledge_risk(AckRisk), Options, false),
    py_call(
        'unifyweaver.glue.rpyc':'ConfigurableRPyCBridge'(
            HostStr,
            tunnel_type=SecStr,
            remote_port=Port,
            acknowledge_risk=AckRisk
        ),
        Bridge
    ).

%% build_bridge_options(+Options, -BridgeOpts)
%  Convert Prolog options to Python kwargs dictionary.
build_bridge_options(Options, BridgeOpts) :-
    findall(Key=Value, (
        member(Opt, Options),
        option_to_python(Opt, Key, Value)
    ), Pairs),
    dict_create(BridgeOpts, _, Pairs).

option_to_python(remote_port(P), remote_port, P) :- !.
option_to_python(local_port(P), local_port, P) :- !.
option_to_python(ssh_user(U), ssh_user, U) :- atom(U), !.
option_to_python(ssh_port(P), ssh_port, P) :- !.
option_to_python(ssh_key(K), ssh_key, K) :- atom(K), !.
option_to_python(keyfile(F), keyfile, F) :- atom(F), !.
option_to_python(certfile(F), certfile, F) :- atom(F), !.
option_to_python(ca_certs(F), ca_certs, F) :- atom(F), !.
option_to_python(acknowledge_risk(B), acknowledge_risk, B) :- !.
option_to_python(_, _, _) :- fail.

%% connect_bridge(+Bridge, -Proxy)
%  Establish connection using the bridge.
connect_bridge(Bridge, Proxy) :-
    py_call(Bridge:connect(), Proxy).

%% rpyc_disconnect(+Proxy)
%  Disconnect from an RPyC server.
rpyc_disconnect(Proxy) :-
    catch(
        py_call(Proxy:close(), _),
        _,
        true
    ).

%% rpyc_with_connection(+Host, +Options, :Goal)
%  Execute Goal with an RPyC connection, ensuring cleanup.
%
%  Example:
%    ?- rpyc_with_connection('localhost',
%           [security(unsecured), acknowledge_risk(true)],
%           rpyc_call(Proxy, math, sqrt, [16], R)).
%
:- meta_predicate rpyc_with_connection(+, +, 0).
rpyc_with_connection(Host, Options, Goal) :-
    setup_call_cleanup(
        rpyc_connect(Host, Options, Proxy),
        call(Goal, Proxy),
        rpyc_disconnect(Proxy)
    ).

%% ============================================
%% Path Management
%% ============================================

%% janus_add_rpyc_path
%  Add the UnifyWeaver RPyC module path to Python.
janus_add_rpyc_path :-
    % Try multiple approaches to find the src directory
    (   % Method 1: From source_file
        source_file(rpyc_glue:_, SrcFile),
        file_directory_name(SrcFile, GlueDir),
        file_directory_name(GlueDir, UnifyWeaverDir),
        file_directory_name(UnifyWeaverDir, SrcDir),
        py_add_lib_dir(SrcDir)
    ;   % Method 2: From working directory
        working_directory(Cwd, Cwd),
        atomic_list_concat([Cwd, '/src'], SrcPath),
        exists_directory(SrcPath),
        py_add_lib_dir(SrcPath)
    ;   % Method 3: Just add 'src' relative path
        py_add_lib_dir(src)
    ),
    !.  % Succeed once

%% ============================================
%% Remote Execution (Sync)
%% ============================================

%% rpyc_call(+Proxy, +Module, +Function, +Args, -Result)
%  Call a Python function on the remote server.
%
%  Example:
%    ?- rpyc_connect(..., Proxy),
%       rpyc_call(Proxy, math, sqrt, [16], Result).
%    Result = 4.0.
%
rpyc_call(Proxy, Module, Function, Args, Result) :-
    atom(Module),
    atom(Function),
    is_list(Args),
    rpyc_get_module(Proxy, Module, ModRef),
    py_call(ModRef:Function, TempArgs),
    apply_args(TempArgs, Args, CallResult),
    Result = CallResult.

%% Helper to apply args - simplified approach
apply_args(Func, [], Result) :-
    !,
    py_call(Func, Result).
apply_args(Func, Args, Result) :-
    FuncWithArgs =.. [call, Func | Args],
    py_call(FuncWithArgs, Result).

%% rpyc_exec(+Proxy, +Code, -Namespace)
%  Execute Python code on the remote server.
%
%  Example:
%    ?- rpyc_exec(Proxy, "x = 42\ny = x * 2", NS),
%       rpyc_get_var(NS, y, Y).
%    Y = 84.
%
rpyc_exec(Proxy, Code, Namespace) :-
    string(Code),
    py_call(Proxy:local_methods:wrapped_exec(Code), Namespace).

%% ============================================
%% Remote Execution (Async)
%% ============================================

%% rpyc_async_call(+Proxy, +Module, +Function, +Args, -AsyncResult)
%  Make an asynchronous call to a remote function.
%
%  Example:
%    ?- rpyc_async_call(Proxy, ml, train_model, [Data], Future),
%       do_other_work,
%       rpyc_await(Future, Model).
%
rpyc_async_call(Proxy, Module, Function, Args, AsyncResult) :-
    atom(Module),
    atom(Function),
    is_list(Args),
    atom_string(Module, ModStr),
    atom_string(Function, FuncStr),
    py_call(
        'unifyweaver.glue.rpyc':async_call(Proxy, ModStr, FuncStr, Args),
        AsyncResult
    ).

%% rpyc_await(+AsyncResult, -Result)
%  Wait for and get the result of an async call.
rpyc_await(AsyncResult, Result) :-
    py_call(AsyncResult:value, Result).

%% rpyc_ready(+AsyncResult)
%  Succeeds if the async result is ready.
rpyc_ready(AsyncResult) :-
    py_call(AsyncResult:ready, true).

%% ============================================
%% Module Access
%% ============================================

%% rpyc_import(+Proxy, +ModuleName, -ModuleRef)
%  Import a module on the remote server.
%
%  Example:
%    ?- rpyc_import(Proxy, numpy, NP),
%       py_call(NP:array([1,2,3]), Arr).
%
rpyc_import(Proxy, ModuleName, ModuleRef) :-
    atom(ModuleName),
    py_call(Proxy:modules:ModuleName, ModuleRef).

%% rpyc_get_module(+Proxy, +FullModuleName, -ModuleRef)
%  Get a module by its full dotted name.
%
%  Example:
%    ?- rpyc_get_module(Proxy, 'numpy.linalg', LinAlg).
%
rpyc_get_module(Proxy, FullModuleName, ModuleRef) :-
    atom(FullModuleName),
    atom_string(FullModuleName, NameStr),
    py_call(Proxy:get_module(NameStr), ModuleRef).

%% ============================================
%% Proxy Layers
%% ============================================

%% rpyc_root(+Proxy, -RootProxy)
%  Get Layer 1 proxy: Direct exposed_ method access.
%
%  Use for simple RPC calls where you know the method name.
%
%  Example:
%    ?- rpyc_root(Proxy, Root),
%       py_call(Root:compute([1,2,3]), Result).
%
rpyc_root(Proxy, RootProxy) :-
    py_call(Proxy:root, RootProxy).

%% rpyc_wrapped_root(+Proxy, -WrappedRoot)
%  Get Layer 2 proxy: Safe attribute access.
%
%  Prevents accidental execution by wrapping all attribute access.
%
rpyc_wrapped_root(Proxy, WrappedRoot) :-
    py_call(Proxy:wrapped_root, WrappedRoot).

%% rpyc_auto_root(+Proxy, -AutoRoot)
%  Get Layer 3 proxy: Automatic wrapping.
%
%  General-purpose proxy that automatically wraps remote objects.
%
rpyc_auto_root(Proxy, AutoRoot) :-
    py_call(Proxy:auto_root, AutoRoot).

%% rpyc_smart_root(+Proxy, -SmartRoot)
%  Get Layer 4 proxy: Local-class-aware wrapping.
%
%  Only wraps when the class isn't available locally.
%  Useful for code generation scenarios.
%
rpyc_smart_root(Proxy, SmartRoot) :-
    py_call(Proxy:smart_root, SmartRoot).

%% ============================================
%% Code Generation
%% ============================================

%% generate_rpyc_client(+Predicates, +Options, -Code)
%  Generate Prolog client code for RPyC calls.
%
%  Example:
%    ?- generate_rpyc_client([
%           exposed(compute/2, [module(mymodule)])
%       ], [host('server.local')], Code).
%
generate_rpyc_client(Predicates, Options, Code) :-
    option(host(Host), Options, localhost),
    option(security(Security), Options, unsecured),
    findall(PredCode, (
        member(exposed(Pred/Arity, PredOpts), Predicates),
        generate_client_predicate(Pred, Arity, Host, Security, PredOpts, PredCode)
    ), PredCodes),
    atomic_list_concat(PredCodes, '\n\n', Code).

generate_client_predicate(Pred, Arity, Host, Security, PredOpts, Code) :-
    option(module(Module), PredOpts, builtins),
    option(function(Function), PredOpts, Pred),
    InputArity is Arity - 1,
    generate_arg_list(InputArity, ArgList),
    format(atom(Line1), '% RPyC wrapper for ~w/~w', [Pred, Arity]),
    format(atom(Line2), '% Calls ~w.~w on ~w', [Module, Function, Host]),
    format(atom(Line3), '~w(~w, Result) :-', [Pred, ArgList]),
    format(atom(Line4), '    rpyc_connect(~q, [security(~w), acknowledge_risk(true)], Proxy),', [Host, Security]),
    format(atom(Line5), '    rpyc_call(Proxy, ~w, ~w, [~w], Result),', [Module, Function, ArgList]),
    Line6 = '    rpyc_disconnect(Proxy).',
    atomic_list_concat([Line1, Line2, Line3, Line4, Line5, Line6], '\n', Code).

generate_arg_list(0, '') :- !.
generate_arg_list(1, 'Arg1') :- !.
generate_arg_list(N, ArgList) :-
    N > 1,
    findall(Arg, (
        between(1, N, I),
        format(atom(Arg), 'Arg~w', [I])
    ), Args),
    atomic_list_concat(Args, ', ', ArgList).

%% generate_rpyc_service(+Predicates, +Options, -ServiceCode)
%  Generate Python RPyC service code from predicate declarations.
%
%  Example:
%    ?- generate_rpyc_service([
%           exposed(transform_data/2, [input(list), output(list)]),
%           exposed(predict/2, [input(dict), output(float)])
%       ], [service_name('MLService')], ServiceCode).
%
generate_rpyc_service(Predicates, Options, ServiceCode) :-
    option(service_name(ServiceName), Options, 'GeneratedService'),
    option(imports(Imports), Options, []),
    generate_imports(Imports, ImportCode),
    findall(MethodCode, (
        member(exposed(Pred/_, PredOpts), Predicates),
        generate_exposed_method(Pred, PredOpts, MethodCode)
    ), MethodCodes),
    atomic_list_concat(MethodCodes, '\n\n', MethodsCode),
    format(atom(Header), '#!/usr/bin/env python3\n"""\nGenerated RPyC Service: ~w\nGenerated by UnifyWeaver\n"""', [ServiceName]),
    format(atom(ClassDef), 'class ~w(rpyc.Service):\n    ALIASES = [~q]', [ServiceName, ServiceName]),
    format(atom(MainBlock), 'if __name__ == "__main__":\n    server = ThreadedServer(~w, port=18812)\n    print("Starting ~w on port 18812...")\n    server.start()', [ServiceName, ServiceName]),
    atomic_list_concat([
        Header,
        '',
        'import rpyc',
        'from rpyc.utils.server import ThreadedServer',
        ImportCode,
        '',
        ClassDef,
        '',
        MethodsCode,
        '',
        MainBlock
    ], '\n', ServiceCode).

generate_imports([], '') :- !.
generate_imports(Imports, ImportCode) :-
    findall(Line, (
        member(Imp, Imports),
        format(atom(Line), 'import ~w', [Imp])
    ), Lines),
    atomic_list_concat(Lines, '\n', ImportCode).

generate_exposed_method(Pred, Options, Code) :-
    option(input(_), Options, any),
    option(output(_), Options, any),
    format(atom(Def), '    def exposed_~w(self, *args, **kwargs):', [Pred]),
    format(atom(Doc), '        """Auto-generated exposed method for ~w"""', [Pred]),
    format(atom(Todo), '        # TODO: Implement ~w logic', [Pred]),
    format(atom(Raise), '        raise NotImplementedError("~w not yet implemented")', [Pred]),
    atomic_list_concat([Def, Doc, Todo, Raise], '\n', Code).

%% generate_rpyc_server(+Options, -ServerScript)
%  Generate a complete RPyC server startup script.
%
%  Example:
%    ?- generate_rpyc_server([output_file('my_server.py')], Script),
%       write(Script).
%
generate_rpyc_server(Options, ServerScript) :-
    option(service_name(ServiceName), Options, 'UnifyWeaverService'),
    option(output_file(OutputFile), Options, 'rpyc_server.py'),
    (   janus_available
    ->  py_call(
            'unifyweaver.glue.rpyc':generate_rpyc_server_script(OutputFile, ServiceName),
            ServerScript
        )
    ;   format(atom(Comment), '# RPyC Server Script\n# Run: python ~w', [OutputFile]),
        format(atom(ClassDef), 'class ~w(ClassicService):\n    pass', [ServiceName]),
        format(atom(MainBlock), 'if __name__ == "__main__":\n    server = ThreadedServer(~w, port=18812)\n    print("Starting server on port 18812...")\n    server.start()', [ServiceName]),
        atomic_list_concat([
            Comment,
            '',
            'import rpyc',
            'from rpyc.utils.server import ThreadedServer',
            'from rpyc.core.service import ClassicService',
            '',
            ClassDef,
            '',
            MainBlock
        ], '\n', ServerScript)
    ).

%% ============================================
%% Testing
%% ============================================

%% test_rpyc_glue
%  Run basic tests for RPyC glue functionality.
test_rpyc_glue :-
    format('~n=== RPyC Glue Tests ===~n'),

    % Test 1: Transport registration
    format('1. Transport registration: '),
    (   rpyc_transport_type(rpyc)
    ->  format('PASS~n')
    ;   format('FAIL~n')
    ),

    % Test 2: Code generation (doesn't require connection)
    format('2. Client code generation: '),
    (   generate_rpyc_client([
            exposed(compute/2, [module(math), function(sqrt)])
        ], [host('localhost')], ClientCode),
        ClientCode \= ''
    ->  format('PASS~n')
    ;   format('FAIL~n')
    ),

    % Test 3: Service code generation
    format('3. Service code generation: '),
    (   generate_rpyc_service([
            exposed(transform/2, [input(list), output(list)])
        ], [service_name('TestService')], ServiceCode),
        ServiceCode \= ''
    ->  format('PASS~n')
    ;   format('FAIL~n')
    ),

    % Test 4: Janus availability check
    format('4. Janus availability: '),
    (   janus_available
    ->  format('PASS (available)~n')
    ;   format('SKIP (not available)~n')
    ),

    format('=== Tests Complete ===~n').

%% ============================================
%% Option Helper
%% ============================================

%% option(+Term, +Options, +Default)
%  Get option value with default.
option(Term, Options, _Default) :-
    member(Term, Options),
    !.
option(Term, _Options, Default) :-
    Term =.. [_Name, Default],
    !.
option(Term, _Options, _Default) :-
    Term =.. [Name, _],
    throw(error(existence_error(option, Name), _)).

%% janus_available
%  Check if Janus is available.
janus_available :-
    catch(
        current_module(janus),
        _,
        fail
    ).
