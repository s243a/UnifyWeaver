% SPDX-License-Identifier: MIT
% Copyright (c) 2025 John William Creighton (s243a)
%
% Python Bridges Glue - Cross-runtime Python embedding for RPyC
%
% This module generates glue code for embedding CPython in other runtimes
% and accessing RPyC through that embedded Python:
%
% .NET Bridges:
%   - Python.NET: Mature, embeds CPython in .NET CLR
%   - CSnakes: Modern .NET 8+, simpler embedding API
%
% JVM Bridges:
%   - JPype: Shared memory approach, good NumPy support
%   - jpy: Bi-directional Java↔Python, used by JetBrains
%
% Rust Bridge:
%   - PyO3: De-facto standard for Rust-Python interop
%
% Ruby Bridge:
%   - PyCall.rb: Embeds CPython in Ruby
%
% Usage:
%   ?- generate_pythonnet_rpyc_client(Options, CSharpCode).
%   ?- generate_csnakes_rpyc_client(Options, CSharpCode).
%   ?- generate_jpype_rpyc_client(Options, JavaCode).
%   ?- generate_jpy_rpyc_client(Options, JavaCode).
%   ?- generate_pyo3_rpyc_client(Options, RustCode).
%   ?- generate_pycall_rb_rpyc_client(Options, RubyCode).

:- module(python_bridges_glue, [
    % Bridge detection
    detect_pythonnet/1,             % detect_pythonnet(-Available)
    detect_csnakes/1,               % detect_csnakes(-Available)
    detect_jpype/1,                 % detect_jpype(-Available)
    detect_jpy/1,                   % detect_jpy(-Available)
    detect_pyo3/1,                  % detect_pyo3(-Available)
    detect_pycall_rb/1,             % detect_pycall_rb(-Available)
    detect_all_bridges/1,           % detect_all_bridges(-AvailableBridges)

    % Bridge requirements and validation
    bridge_requirements/2,          % bridge_requirements(+Bridge, -Requirements)
    check_bridge_ready/2,           % check_bridge_ready(+Bridge, -Status)
    validate_bridge_config/2,       % validate_bridge_config(+Bridge, +Options)

    % Auto-selection with fallback
    auto_select_bridge/2,           % auto_select_bridge(+Target, -Bridge)
    auto_select_bridge/3,           % auto_select_bridge(+Target, +Preferences, -Bridge)

    % Bridge selection (legacy)
    select_dotnet_python_bridge/2,  % select_dotnet_python_bridge(+Preferences, -Bridge)
    select_jvm_python_bridge/2,     % select_jvm_python_bridge(+Preferences, -Bridge)

    % Python.NET code generation
    generate_pythonnet_rpyc_client/2,    % generate_pythonnet_rpyc_client(+Options, -Code)
    generate_pythonnet_rpyc_service/2,   % generate_pythonnet_rpyc_service(+Options, -Code)
    generate_pythonnet_project/2,        % generate_pythonnet_project(+Options, -CsprojContent)

    % CSnakes code generation
    generate_csnakes_rpyc_client/2,      % generate_csnakes_rpyc_client(+Options, -Code)
    generate_csnakes_rpyc_service/2,     % generate_csnakes_rpyc_service(+Options, -Code)
    generate_csnakes_project/2,          % generate_csnakes_project(+Options, -CsprojContent)

    % JPype code generation
    generate_jpype_rpyc_client/2,        % generate_jpype_rpyc_client(+Options, -Code)
    generate_jpype_rpyc_service/2,       % generate_jpype_rpyc_service(+Options, -Code)
    generate_jpype_gradle/2,             % generate_jpype_gradle(+Options, -GradleContent)

    % jpy code generation
    generate_jpy_rpyc_client/2,          % generate_jpy_rpyc_client(+Options, -Code)
    generate_jpy_rpyc_service/2,         % generate_jpy_rpyc_service(+Options, -Code)
    generate_jpy_gradle/2,               % generate_jpy_gradle(+Options, -GradleContent)

    % PyO3 (Rust) code generation
    generate_pyo3_rpyc_client/2,         % generate_pyo3_rpyc_client(+Options, -Code)
    generate_pyo3_cargo_toml/2,          % generate_pyo3_cargo_toml(+Options, -TomlContent)

    % PyCall.rb (Ruby) code generation
    generate_pycall_rb_rpyc_client/2,    % generate_pycall_rb_rpyc_client(+Options, -Code)
    generate_pycall_rb_gemfile/2,        % generate_pycall_rb_gemfile(+Options, -GemfileContent)

    % Generic interface
    generate_python_bridge_client/3,     % generate_python_bridge_client(+Bridge, +Options, -Code)
    generate_python_bridge_service/3,    % generate_python_bridge_service(+Bridge, +Options, -Code)

    % Auto-generation (uses preferences + firewall)
    generate_auto_client/2,              % generate_auto_client(+Target, -Code)
    generate_auto_client/3,              % generate_auto_client(+Target, +Options, -Code)

    % Testing
    test_python_bridges_glue/0
]).

% Optional integration with preferences and firewall systems
:- use_module(library(lists)).
:- if(exists_source('../core/preferences')).
:- use_module('../core/preferences', [get_final_options/3, rule_preferences/2]).
:- endif.
:- if(exists_source('../core/firewall')).
:- use_module('../core/firewall', [
    get_firewall_policy/2,
    validate_against_firewall/3,
    derive_policy/2,
    firewall_implies_default/2
]).
:- endif.

% ============================================================================
% BRIDGE DETECTION
% ============================================================================

%% detect_pythonnet(-Available)
%  Check if Python.NET is available (pythonnet package installed).
detect_pythonnet(Available) :-
    (   catch(
            (process_create(path(python3), ['-c', 'import clr; print("ok")'],
                           [stdout(pipe(S)), stderr(null)]),
             read_line_to_string(S, "ok"),
             close(S)),
            _, fail)
    ->  Available = true
    ;   Available = false
    ).

%% detect_csnakes(-Available)
%  Check if CSnakes is available (.NET 8+ with CSnakes package).
detect_csnakes(Available) :-
    (   catch(
            (process_create(path(dotnet), ['--list-sdks'],
                           [stdout(pipe(S)), stderr(null)]),
             read_string(S, _, Output),
             close(S),
             sub_string(Output, _, _, _, "8.")),
            _, fail)
    ->  Available = true  % .NET 8+ detected, CSnakes can be installed
    ;   Available = false
    ).

%% detect_jpype(-Available)
%  Check if JPype is available.
detect_jpype(Available) :-
    (   catch(
            (process_create(path(python3), ['-c', 'import jpype; print("ok")'],
                           [stdout(pipe(S)), stderr(null)]),
             read_line_to_string(S, "ok"),
             close(S)),
            _, fail)
    ->  Available = true
    ;   Available = false
    ).

%% detect_jpy(-Available)
%  Check if jpy is available.
detect_jpy(Available) :-
    (   catch(
            (process_create(path(python3), ['-c', 'import jpy; print("ok")'],
                           [stdout(pipe(S)), stderr(null)]),
             read_line_to_string(S, "ok"),
             close(S)),
            _, fail)
    ->  Available = true
    ;   Available = false
    ).

%% detect_pyo3(-Available)
%  Check if PyO3/Rust toolchain is available.
detect_pyo3(Available) :-
    (   catch(
            (process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null)]),
             true),
            _, fail)
    ->  Available = true
    ;   Available = false
    ).

%% detect_pycall_rb(-Available)
%  Check if PyCall.rb/Ruby is available.
detect_pycall_rb(Available) :-
    (   catch(
            (process_create(path(ruby), ['-e', 'require "pycall"; puts "ok"'],
                           [stdout(pipe(S)), stderr(null)]),
             read_line_to_string(S, "ok"),
             close(S)),
            _, fail)
    ->  Available = true
    ;   % Check if Ruby is available even without pycall gem
        catch(
            (process_create(path(ruby), ['--version'],
                           [stdout(null), stderr(null)]),
             true),
            _, fail)
    ->  Available = true  % Ruby available, pycall can be installed
    ;   Available = false
    ).

%% detect_all_bridges(-AvailableBridges)
%  Detect all available Python bridges and return as a list.
%  Returns bridges in priority order (most preferred first).
detect_all_bridges(AvailableBridges) :-
    findall(Bridge, (
        member(Bridge-Detector, [
            pythonnet-detect_pythonnet,
            csnakes-detect_csnakes,
            jpype-detect_jpype,
            jpy-detect_jpy,
            pyo3-detect_pyo3,
            pycall_rb-detect_pycall_rb
        ]),
        call(Detector, true)
    ), AvailableBridges).

% ============================================================================
% BRIDGE REQUIREMENTS AND VALIDATION
% ============================================================================

%% bridge_requirements(+Bridge, -Requirements)
%  Lists requirements for a specific bridge.
%  Requirements is a list of requirement(Type, Description) terms.
bridge_requirements(pythonnet, [
    requirement(runtime, '.NET 6.0+ or Mono'),
    requirement(python_package, 'pythonnet'),
    requirement(python_package, 'rpyc'),
    requirement(environment, 'PYTHONNET_RUNTIME=coreclr (for .NET Core)')
]).
bridge_requirements(csnakes, [
    requirement(runtime, '.NET 8.0+'),
    requirement(nuget_package, 'CSnakes.Runtime'),
    requirement(python_package, 'rpyc'),
    requirement(note, 'Uses source generators - Python files must be in project')
]).
bridge_requirements(jpype, [
    requirement(runtime, 'Java 11+'),
    requirement(python_package, 'jpype1'),
    requirement(python_package, 'rpyc'),
    requirement(environment, 'JAVA_HOME must be set')
]).
bridge_requirements(jpy, [
    requirement(runtime, 'Java 11+'),
    requirement(python_package, 'jpy'),
    requirement(python_package, 'rpyc'),
    requirement(build_tool, 'Maven (for jpy build)'),
    requirement(environment, 'JAVA_HOME must be set'),
    requirement(note, 'Bi-directional - use size()/get() for Java collections')
]).
bridge_requirements(pyo3, [
    requirement(runtime, 'Rust toolchain (rustup)'),
    requirement(crate, 'pyo3'),
    requirement(python_package, 'rpyc'),
    requirement(note, 'Use import_bound() for PyO3 0.21+'),
    requirement(note, 'Requires Python dev headers for compilation')
]).
bridge_requirements(pycall_rb, [
    requirement(runtime, 'Ruby 2.7+'),
    requirement(gem, 'pycall'),
    requirement(python_package, 'rpyc'),
    requirement(system_package, 'ruby-dev (for native extensions)'),
    requirement(note, 'Use .to_f for Ruby numeric operations on RPyC floats')
]).

%% check_bridge_ready(+Bridge, -Status)
%  Check if a bridge is ready to use with detailed status.
%  Status is one of: ready, missing_runtime, missing_package, not_detected.
check_bridge_ready(Bridge, Status) :-
    (   Bridge = pythonnet
    ->  check_pythonnet_ready(Status)
    ;   Bridge = csnakes
    ->  check_csnakes_ready(Status)
    ;   Bridge = jpype
    ->  check_jpype_ready(Status)
    ;   Bridge = jpy
    ->  check_jpy_ready(Status)
    ;   Bridge = pyo3
    ->  check_pyo3_ready(Status)
    ;   Bridge = pycall_rb
    ->  check_pycall_rb_ready(Status)
    ;   Status = unknown_bridge
    ).

check_pythonnet_ready(Status) :-
    (   detect_pythonnet(true)
    ->  Status = ready
    ;   check_dotnet_available(DotNetOK),
        (   DotNetOK = false
        ->  Status = missing_runtime('.NET')
        ;   Status = missing_package(pythonnet)
        )
    ).

check_csnakes_ready(Status) :-
    (   detect_csnakes(true)
    ->  Status = ready  % .NET 8+ detected
    ;   Status = missing_runtime('.NET 8+')
    ).

check_jpype_ready(Status) :-
    (   detect_jpype(true)
    ->  Status = ready
    ;   check_java_available(JavaOK),
        (   JavaOK = false
        ->  Status = missing_runtime('Java')
        ;   Status = missing_package(jpype1)
        )
    ).

check_jpy_ready(Status) :-
    (   detect_jpy(true)
    ->  Status = ready
    ;   check_java_available(JavaOK),
        (   JavaOK = false
        ->  Status = missing_runtime('Java')
        ;   Status = missing_package(jpy)
        )
    ).

check_pyo3_ready(Status) :-
    (   detect_pyo3(true)
    ->  Status = ready
    ;   Status = missing_runtime('Rust toolchain')
    ).

check_pycall_rb_ready(Status) :-
    (   catch(
            (process_create(path(ruby), ['-e', 'require "pycall"; puts "ok"'],
                           [stdout(pipe(S)), stderr(null)]),
             read_line_to_string(S, "ok"),
             close(S)),
            _, fail)
    ->  Status = ready
    ;   check_ruby_available(RubyOK),
        (   RubyOK = false
        ->  Status = missing_runtime('Ruby')
        ;   Status = missing_package(pycall)
        )
    ).

check_ruby_available(Available) :-
    (   catch(
            (process_create(path(ruby), ['--version'],
                           [stdout(null), stderr(null)]),
             true),
            _, fail)
    ->  Available = true
    ;   Available = false
    ).

check_dotnet_available(Available) :-
    (   catch(
            (process_create(path(dotnet), ['--version'],
                           [stdout(null), stderr(null)]),
             true),
            _, fail)
    ->  Available = true
    ;   Available = false
    ).

check_java_available(Available) :-
    (   catch(
            (process_create(path(java), ['-version'],
                           [stdout(null), stderr(null)]),
             true),
            _, fail)
    ->  Available = true
    ;   Available = false
    ).

%% validate_bridge_config(+Bridge, +Options)
%  Validate that options are valid for a bridge.
%  Fails with error message if invalid.
validate_bridge_config(Bridge, Options) :-
    % Check required options have valid values
    (   member(port(Port), Options)
    ->  (   integer(Port), Port > 0, Port < 65536
        ->  true
        ;   format(user_error, 'Invalid port: ~w (must be 1-65535)~n', [Port]),
            fail
        )
    ;   true
    ),
    (   member(host(Host), Options)
    ->  (   atom(Host) ; string(Host)
        ->  true
        ;   format(user_error, 'Invalid host: ~w (must be atom or string)~n', [Host]),
            fail
        )
    ;   true
    ),
    % Bridge-specific validation
    validate_bridge_specific(Bridge, Options).

validate_bridge_specific(pythonnet, Options) :-
    (   member(target_framework(Framework), Options)
    ->  (   sub_atom(Framework, 0, 3, _, net)
        ->  true
        ;   format(user_error, 'Warning: non-standard framework ~w for pythonnet~n', [Framework])
        )
    ;   true
    ).
validate_bridge_specific(csnakes, Options) :-
    (   member(target_framework(Framework), Options)
    ->  (   sub_atom(Framework, 0, 4, _, 'net8') ; sub_atom(Framework, 0, 4, _, 'net9')
        ->  true
        ;   format(user_error, 'Warning: CSnakes requires .NET 8+, got ~w~n', [Framework])
        )
    ;   true
    ).
validate_bridge_specific(jpype, _).
validate_bridge_specific(jpy, _).

% ============================================================================
% AUTO-SELECTION WITH PREFERENCES AND FIREWALL
% ============================================================================

%% auto_select_bridge(+Target, -Bridge)
%  Auto-select the best bridge for a target platform.
%  Target is one of: dotnet, jvm, any.
%  Uses preferences and firewall if available.
auto_select_bridge(Target, Bridge) :-
    auto_select_bridge(Target, [], Bridge).

%% auto_select_bridge(+Target, +Preferences, -Bridge)
%  Auto-select with explicit preferences.
%  Preferences can include: prefer(BridgeName), fallback([B1,B2,...])
auto_select_bridge(Target, Preferences, Bridge) :-
    % Get available bridges
    detect_all_bridges(Available),

    % Filter by target platform
    filter_bridges_by_target(Target, Available, TargetBridges),

    % Apply firewall constraints if available
    filter_bridges_by_firewall(TargetBridges, AllowedBridges),

    % Apply preferences
    select_preferred_bridge(Preferences, AllowedBridges, Bridge).

%% filter_bridges_by_target(+Target, +Bridges, -Filtered)
%  Filter bridges by target platform.
filter_bridges_by_target(dotnet, Bridges, Filtered) :-
    include(is_dotnet_bridge, Bridges, Filtered).
filter_bridges_by_target(jvm, Bridges, Filtered) :-
    include(is_jvm_bridge, Bridges, Filtered).
filter_bridges_by_target(rust, Bridges, Filtered) :-
    include(is_rust_bridge, Bridges, Filtered).
filter_bridges_by_target(ruby, Bridges, Filtered) :-
    include(is_ruby_bridge, Bridges, Filtered).
filter_bridges_by_target(any, Bridges, Bridges).
filter_bridges_by_target(_, Bridges, Bridges).  % Default: allow all

is_dotnet_bridge(pythonnet).
is_dotnet_bridge(csnakes).
is_jvm_bridge(jpype).
is_jvm_bridge(jpy).
is_rust_bridge(pyo3).
is_ruby_bridge(pycall_rb).

%% filter_bridges_by_firewall(+Bridges, -Allowed)
%  Filter bridges by firewall policy.
%  Uses firewall system if loaded, otherwise allows all.
filter_bridges_by_firewall(Bridges, Allowed) :-
    (   current_predicate(firewall:get_firewall_policy/2)
    ->  % Firewall system available - check each bridge
        include(bridge_allowed_by_firewall, Bridges, Allowed)
    ;   % No firewall - allow all
        Allowed = Bridges
    ).

%% bridge_allowed_by_firewall(+Bridge)
%  Check if a bridge is allowed by firewall policy.
bridge_allowed_by_firewall(Bridge) :-
    (   current_predicate(firewall:get_firewall_policy/2)
    ->  firewall:get_firewall_policy(python_bridge/1, Policy),
        \+ member(denied(Bridge), Policy),
        (   member(services(Allowed), Policy)
        ->  member(Bridge, Allowed)
        ;   true  % No whitelist = allow all
        )
    ;   true  % No firewall = allow
    ).

%% select_preferred_bridge(+Preferences, +Available, -Selected)
%  Select the best bridge based on preferences.
select_preferred_bridge(Preferences, Available, Selected) :-
    (   Available = []
    ->  Selected = none
    ;   % Check explicit preference
        member(prefer(Preferred), Preferences),
        member(Preferred, Available)
    ->  Selected = Preferred
    ;   % Check preferences from system if available
        get_bridge_preferences(SystemPrefs),
        member(Preferred, SystemPrefs),
        member(Preferred, Available)
    ->  Selected = Preferred
    ;   % Check fallback order
        member(fallback(Fallbacks), Preferences),
        member(Bridge, Fallbacks),
        member(Bridge, Available)
    ->  Selected = Bridge
    ;   % Default: first available (priority order from detect_all_bridges)
        Available = [Selected|_]
    ).

%% get_bridge_preferences(-Preferences)
%  Get bridge preferences from the preference system if available.
get_bridge_preferences(Preferences) :-
    (   current_predicate(preferences:rule_preferences/2),
        preferences:rule_preferences(python_bridge/1, RulePrefs),
        member(prefer(BridgeList), RulePrefs)
    ->  (   is_list(BridgeList) -> Preferences = BridgeList ; Preferences = [BridgeList] )
    ;   current_predicate(preferences:preferences_default/1),
        preferences:preferences_default(DefaultPrefs),
        member(prefer_bridges(BridgeList), DefaultPrefs)
    ->  (   is_list(BridgeList) -> Preferences = BridgeList ; Preferences = [BridgeList] )
    ;   Preferences = []  % No preferences set
    ).

% ============================================================================
% AUTO-GENERATION
% ============================================================================

%% generate_auto_client(+Target, -Code)
%  Auto-generate client code for the best available bridge.
generate_auto_client(Target, Code) :-
    generate_auto_client(Target, [], Code).

%% generate_auto_client(+Target, +Options, -Code)
%  Auto-generate with options.
generate_auto_client(Target, Options, Code) :-
    auto_select_bridge(Target, Options, Bridge),
    (   Bridge = none
    ->  format(atom(Code), '// ERROR: No ~w Python bridge available~n', [Target])
    ;   validate_bridge_config(Bridge, Options),
        generate_python_bridge_client(Bridge, Options, Code)
    ).

% ============================================================================
% FIREWALL IMPLICATIONS FOR PYTHON BRIDGES
% ============================================================================

% Add firewall implications if the firewall module is available
:- if(current_predicate(firewall:firewall_implies_default/2)).

% If .NET not available, deny .NET bridges
:- assertz(firewall:firewall_implies_default(
    no_dotnet_available,
    denied(services([pythonnet, csnakes]))
)).

% If Java not available, deny JVM bridges
:- assertz(firewall:firewall_implies_default(
    no_java_available,
    denied(services([jpype, jpy]))
)).

% If rpyc not available, deny all Python bridges
:- assertz(firewall:firewall_implies_default(
    no_rpyc_available,
    denied(services([pythonnet, csnakes, jpype, jpy]))
)).

% Prefer source-generated bridges in strict security mode
:- assertz(firewall:firewall_implies_default(
    security_policy(strict),
    prefer(service(dotnet, csnakes), service(dotnet, pythonnet))
)).

% In offline mode, RPyC still works for local services
:- assertz(firewall:firewall_implies_default(
    mode(offline),
    network_hosts_for_rpyc(['localhost', '127.0.0.1'])
)).

:- endif.

% ============================================================================
% BRIDGE SELECTION (LEGACY - use auto_select_bridge instead)
% ============================================================================

%% select_dotnet_python_bridge(+Preferences, -Bridge)
%  Select best .NET Python bridge based on preferences.
%  Preferences can include: prefer(pythonnet), prefer(csnakes), modern_dotnet
select_dotnet_python_bridge(Preferences, Bridge) :-
    (   member(prefer(csnakes), Preferences),
        detect_csnakes(true)
    ->  Bridge = csnakes
    ;   member(prefer(pythonnet), Preferences),
        detect_pythonnet(true)
    ->  Bridge = pythonnet
    ;   member(modern_dotnet, Preferences),
        detect_csnakes(true)
    ->  Bridge = csnakes
    ;   detect_pythonnet(true)
    ->  Bridge = pythonnet
    ;   detect_csnakes(true)
    ->  Bridge = csnakes
    ;   Bridge = none
    ).

%% select_jvm_python_bridge(+Preferences, -Bridge)
%  Select best JVM Python bridge based on preferences.
select_jvm_python_bridge(Preferences, Bridge) :-
    (   member(prefer(jpy), Preferences),
        detect_jpy(true)
    ->  Bridge = jpy
    ;   member(prefer(jpype), Preferences),
        detect_jpype(true)
    ->  Bridge = jpype
    ;   member(bidirectional, Preferences),
        detect_jpy(true)
    ->  Bridge = jpy  % jpy is better for bidirectional
    ;   detect_jpype(true)
    ->  Bridge = jpype
    ;   detect_jpy(true)
    ->  Bridge = jpy
    ;   Bridge = none
    ).

% ============================================================================
% PYTHON.NET CODE GENERATION
% ============================================================================

%% generate_pythonnet_rpyc_client(+Options, -Code)
%  Generate C# code for an RPyC client using Python.NET.
generate_pythonnet_rpyc_client(Options, Code) :-
    option(host(Host), Options, "localhost"),
    option(port(Port), Options, 18812),
    option(namespace(Namespace), Options, "UnifyWeaver.PythonBridge"),
    option(class_name(ClassName), Options, "RPyCClient"),
    format(atom(Code), '// Generated by UnifyWeaver - Python.NET RPyC Client
// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using Python.Runtime;

namespace ~w
{
    /// <summary>
    /// RPyC client using Python.NET to embed CPython.
    /// Provides live object proxies over the network.
    /// </summary>
    public class ~w : IDisposable
    {
        private dynamic _rpyc;
        private dynamic _conn;
        private bool _disposed = false;

        public ~w()
        {
            // Initialize Python runtime
            if (!PythonEngine.IsInitialized)
            {
                PythonEngine.Initialize();
            }
        }

        /// <summary>
        /// Connect to an RPyC server.
        /// </summary>
        public void Connect(string host = "~w", int port = ~d)
        {
            using (Py.GIL())
            {
                _rpyc = Py.Import("rpyc");
                _conn = _rpyc.connect(host, port);
            }
        }

        /// <summary>
        /// Get a remote module.
        /// </summary>
        public dynamic GetModule(string moduleName)
        {
            using (Py.GIL())
            {
                return _conn.modules[moduleName];
            }
        }

        /// <summary>
        /// Call a remote function.
        /// </summary>
        public dynamic Call(string moduleName, string functionName, params object[] args)
        {
            using (Py.GIL())
            {
                dynamic module = _conn.modules[moduleName];
                dynamic func = module.GetAttr(functionName);
                return func.Invoke(args);
            }
        }

        /// <summary>
        /// Execute arbitrary Python code on the remote server.
        /// </summary>
        public dynamic Execute(string code)
        {
            using (Py.GIL())
            {
                return _conn.execute(code);
            }
        }

        /// <summary>
        /// Access the root namespace for exposed methods.
        /// </summary>
        public dynamic Root
        {
            get
            {
                using (Py.GIL())
                {
                    return _conn.root;
                }
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                using (Py.GIL())
                {
                    if (_conn != null)
                    {
                        _conn.close();
                    }
                }
                _disposed = true;
            }
        }
    }
}
', [Namespace, ClassName, ClassName, Host, Port]).

%% generate_pythonnet_rpyc_service(+Options, -Code)
%  Generate C# code for hosting an RPyC service using Python.NET.
generate_pythonnet_rpyc_service(Options, Code) :-
    option(port(Port), Options, 18812),
    option(namespace(Namespace), Options, "UnifyWeaver.PythonBridge"),
    option(class_name(ClassName), Options, "RPyCService"),
    format(atom(Code), '// Generated by UnifyWeaver - Python.NET RPyC Service Host
// SPDX-License-Identifier: MIT

using System;
using Python.Runtime;

namespace ~w
{
    /// <summary>
    /// RPyC service host using Python.NET.
    /// </summary>
    public class ~w : IDisposable
    {
        private dynamic _rpyc;
        private dynamic _server;
        private bool _disposed = false;

        public ~w()
        {
            if (!PythonEngine.IsInitialized)
            {
                PythonEngine.Initialize();
            }
        }

        /// <summary>
        /// Start the RPyC server with a custom service class.
        /// </summary>
        public void Start(int port = ~d, string serviceCode = null)
        {
            using (Py.GIL())
            {
                _rpyc = Py.Import("rpyc");
                dynamic ThreadedServer = _rpyc.utils.server.ThreadedServer;

                dynamic service;
                if (serviceCode != null)
                {
                    // Execute custom service code
                    dynamic builtins = Py.Import("builtins");
                    dynamic ns = new PyDict();
                    ns["rpyc"] = _rpyc;
                    builtins.exec(serviceCode, ns);
                    service = ns["CustomService"];
                }
                else
                {
                    service = _rpyc.Service;
                }

                _server = ThreadedServer(service, port: port);
                Console.WriteLine($"RPyC server listening on port {port}");
                _server.start();
            }
        }

        public void Stop()
        {
            using (Py.GIL())
            {
                if (_server != null)
                {
                    _server.close();
                }
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                Stop();
                _disposed = true;
            }
        }
    }
}
', [Namespace, ClassName, ClassName, Port]).

%% generate_pythonnet_project(+Options, -CsprojContent)
%  Generate .csproj file for Python.NET project.
generate_pythonnet_project(Options, CsprojContent) :-
    option(project_name(ProjectName), Options, "RPyCBridge"),
    option(target_framework(Framework), Options, "net8.0"),
    format(atom(CsprojContent), '<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Library</OutputType>
    <TargetFramework>~w</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <AssemblyName>~w</AssemblyName>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="pythonnet" Version="3.0.*" />
  </ItemGroup>

</Project>
', [Framework, ProjectName]).

% ============================================================================
% CSNAKES CODE GENERATION
% ============================================================================

%% generate_csnakes_rpyc_client(+Options, -Code)
%  Generate C# code for an RPyC client using CSnakes.
generate_csnakes_rpyc_client(Options, Code) :-
    option(host(Host), Options, "localhost"),
    option(port(Port), Options, 18812),
    option(namespace(Namespace), Options, "UnifyWeaver.PythonBridge"),
    option(class_name(ClassName), Options, "CSnakesRPyCClient"),
    format(atom(Code), '// Generated by UnifyWeaver - CSnakes RPyC Client
// SPDX-License-Identifier: MIT

using System;
using CSnakes.Runtime;

namespace ~w
{
    /// <summary>
    /// RPyC client using CSnakes for modern .NET 8+ Python embedding.
    /// </summary>
    public class ~w : IDisposable
    {
        private readonly IPythonEnvironment _python;
        private dynamic? _conn;
        private bool _disposed = false;

        public ~w()
        {
            // Initialize CSnakes Python environment
            _python = PythonEnvironment.Create();
        }

        /// <summary>
        /// Connect to an RPyC server.
        /// </summary>
        public void Connect(string host = "~w", int port = ~d)
        {
            var rpyc = _python.Import("rpyc");
            _conn = rpyc.Call("connect", host, port);
        }

        /// <summary>
        /// Get a remote module.
        /// </summary>
        public dynamic GetModule(string moduleName)
        {
            if (_conn == null) throw new InvalidOperationException("Not connected");
            return _conn.modules[moduleName];
        }

        /// <summary>
        /// Call a function on a remote module.
        /// </summary>
        public dynamic Call(string moduleName, string functionName, params object[] args)
        {
            var module = GetModule(moduleName);
            return module.Call(functionName, args);
        }

        /// <summary>
        /// Access the root namespace for exposed methods.
        /// </summary>
        public dynamic Root
        {
            get
            {
                if (_conn == null) throw new InvalidOperationException("Not connected");
                return _conn.root;
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _conn?.close();
                _python?.Dispose();
                _disposed = true;
            }
        }
    }
}
', [Namespace, ClassName, ClassName, Host, Port]).

%% generate_csnakes_rpyc_service(+Options, -Code)
%  Generate C# code for hosting an RPyC service using CSnakes.
generate_csnakes_rpyc_service(Options, Code) :-
    option(port(Port), Options, 18812),
    option(namespace(Namespace), Options, "UnifyWeaver.PythonBridge"),
    option(class_name(ClassName), Options, "CSnakesRPyCService"),
    format(atom(Code), '// Generated by UnifyWeaver - CSnakes RPyC Service Host
// SPDX-License-Identifier: MIT

using System;
using CSnakes.Runtime;

namespace ~w
{
    /// <summary>
    /// RPyC service host using CSnakes for modern .NET 8+.
    /// </summary>
    public class ~w : IDisposable
    {
        private readonly IPythonEnvironment _python;
        private dynamic? _server;
        private bool _disposed = false;

        public ~w()
        {
            _python = PythonEnvironment.Create();
        }

        /// <summary>
        /// Start the RPyC server.
        /// </summary>
        public void Start(int port = ~d)
        {
            var rpyc = _python.Import("rpyc");
            var serverModule = _python.Import("rpyc.utils.server");
            var ThreadedServer = serverModule.GetAttr("ThreadedServer");

            _server = ThreadedServer.Call(rpyc.Service, port);
            Console.WriteLine($"CSnakes RPyC server listening on port {port}");
            _server.start();
        }

        public void Stop()
        {
            _server?.close();
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                Stop();
                _python?.Dispose();
                _disposed = true;
            }
        }
    }
}
', [Namespace, ClassName, ClassName, Port]).

%% generate_csnakes_project(+Options, -CsprojContent)
%  Generate .csproj file for CSnakes project.
generate_csnakes_project(Options, CsprojContent) :-
    option(project_name(ProjectName), Options, "RPyCBridge"),
    format(atom(CsprojContent), '<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Library</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <AssemblyName>~w</AssemblyName>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="CSnakes.Runtime" Version="*" />
  </ItemGroup>

</Project>
', [ProjectName]).

% ============================================================================
% JPYPE CODE GENERATION
% ============================================================================

%% generate_jpype_rpyc_client(+Options, -Code)
%  Generate Java code for an RPyC client using JPype.
generate_jpype_rpyc_client(Options, Code) :-
    option(host(Host), Options, "localhost"),
    option(port(Port), Options, 18812),
    option(package(Package), Options, "io.unifyweaver.bridge"),
    option(class_name(ClassName), Options, "JPypeRPyCClient"),
    format(atom(Code), '// Generated by UnifyWeaver - JPype RPyC Client
// SPDX-License-Identifier: MIT

package ~w;

import org.jpype.*;

/**
 * RPyC client using JPype to embed CPython in JVM.
 * Provides live object proxies over the network.
 */
public class ~w implements AutoCloseable {
    private PyObject rpyc;
    private PyObject conn;
    private boolean closed = false;

    public ~w() {
        // Start JVM with Python if not already started
        if (!JPype.isStarted()) {
            JPype.startJVM();
        }
    }

    /**
     * Connect to an RPyC server.
     */
    public void connect(String host, int port) {
        rpyc = JPype.importModule("rpyc");
        conn = rpyc.invoke("connect", host, port);
    }

    public void connect() {
        connect("~w", ~d);
    }

    /**
     * Get a remote module.
     */
    public PyObject getModule(String moduleName) {
        PyObject modules = conn.getAttr("modules");
        return modules.getItem(moduleName);
    }

    /**
     * Call a function on a remote module.
     */
    public PyObject call(String moduleName, String functionName, Object... args) {
        PyObject module = getModule(moduleName);
        PyObject func = module.getAttr(functionName);
        return func.invoke(args);
    }

    /**
     * Access the root namespace for exposed methods.
     */
    public PyObject getRoot() {
        return conn.getAttr("root");
    }

    /**
     * Call an exposed method on the server.
     */
    public PyObject callExposed(String methodName, Object... args) {
        PyObject root = getRoot();
        PyObject method = root.getAttr(methodName);
        return method.invoke(args);
    }

    @Override
    public void close() {
        if (!closed && conn != null) {
            conn.invoke("close");
            closed = true;
        }
    }
}
', [Package, ClassName, ClassName, Host, Port]).

%% generate_jpype_rpyc_service(+Options, -Code)
%  Generate Java code for hosting an RPyC service using JPype.
generate_jpype_rpyc_service(Options, Code) :-
    option(port(Port), Options, 18812),
    option(package(Package), Options, "io.unifyweaver.bridge"),
    option(class_name(ClassName), Options, "JPypeRPyCService"),
    format(atom(Code), '// Generated by UnifyWeaver - JPype RPyC Service Host
// SPDX-License-Identifier: MIT

package ~w;

import org.jpype.*;

/**
 * RPyC service host using JPype.
 */
public class ~w implements AutoCloseable {
    private PyObject rpyc;
    private PyObject server;
    private boolean closed = false;

    public ~w() {
        if (!JPype.isStarted()) {
            JPype.startJVM();
        }
    }

    /**
     * Start the RPyC server.
     */
    public void start(int port) {
        rpyc = JPype.importModule("rpyc");
        PyObject serverUtils = JPype.importModule("rpyc.utils.server");
        PyObject ThreadedServer = serverUtils.getAttr("ThreadedServer");

        PyObject service = rpyc.getAttr("Service");
        server = ThreadedServer.invoke(service);
        server.setAttr("port", port);

        System.out.println("JPype RPyC server listening on port " + port);
        server.invoke("start");
    }

    public void start() {
        start(~d);
    }

    public void stop() {
        if (server != null) {
            server.invoke("close");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            stop();
            closed = true;
        }
    }
}
', [Package, ClassName, ClassName, Port]).

%% generate_jpype_gradle(+Options, -GradleContent)
%  Generate build.gradle for JPype project.
generate_jpype_gradle(Options, GradleContent) :-
    option(group(Group), Options, "io.unifyweaver"),
    option(version(Version), Options, "0.1.0"),
    format(atom(GradleContent), 'plugins {
    id ''java-library''
}

group = ''~w''
version = ''~w''

repositories {
    mavenCentral()
}

dependencies {
    implementation ''org.jpype:jpype:1.5.0''

    testImplementation ''org.junit.jupiter:junit-jupiter:5.10.0''
}

test {
    useJUnitPlatform()
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(17)
    }
}
', [Group, Version]).

% ============================================================================
% JPY CODE GENERATION
% ============================================================================

%% generate_jpy_rpyc_client(+Options, -Code)
%  Generate Java code for an RPyC client using jpy.
generate_jpy_rpyc_client(Options, Code) :-
    option(host(Host), Options, "localhost"),
    option(port(Port), Options, 18812),
    option(package(Package), Options, "io.unifyweaver.bridge"),
    option(class_name(ClassName), Options, "JpyRPyCClient"),
    format(atom(Code), '// Generated by UnifyWeaver - jpy RPyC Client
// SPDX-License-Identifier: MIT

package ~w;

import org.jpy.PyLib;
import org.jpy.PyModule;
import org.jpy.PyObject;

/**
 * RPyC client using jpy for bi-directional Java↔Python communication.
 */
public class ~w implements AutoCloseable {
    private PyModule rpyc;
    private PyObject conn;
    private boolean closed = false;

    public ~w() {
        // Initialize jpy Python interpreter
        if (!PyLib.isPythonRunning()) {
            PyLib.startPython();
        }
    }

    /**
     * Connect to an RPyC server.
     */
    public void connect(String host, int port) {
        rpyc = PyModule.importModule("rpyc");
        conn = rpyc.call("connect", host, port);
    }

    public void connect() {
        connect("~w", ~d);
    }

    /**
     * Get a remote module.
     */
    public PyObject getModule(String moduleName) {
        PyObject modules = conn.getAttribute("modules");
        return modules.call("__getitem__", moduleName);
    }

    /**
     * Call a function on a remote module.
     */
    public PyObject call(String moduleName, String functionName, Object... args) {
        PyObject module = getModule(moduleName);
        return module.call(functionName, args);
    }

    /**
     * Access the root namespace.
     */
    public PyObject getRoot() {
        return conn.getAttribute("root");
    }

    /**
     * Call an exposed method on the server.
     */
    public PyObject callExposed(String methodName, Object... args) {
        PyObject root = getRoot();
        return root.call(methodName, args);
    }

    /**
     * Execute Python code and return result.
     * jpy supports true bi-directional calling.
     */
    public PyObject execute(String code) {
        return PyLib.executeCode(code, PyLib.getCurrentGlobals(), PyLib.getCurrentLocals());
    }

    @Override
    public void close() {
        if (!closed && conn != null) {
            conn.call("close");
            closed = true;
        }
    }
}
', [Package, ClassName, ClassName, Host, Port]).

%% generate_jpy_rpyc_service(+Options, -Code)
%  Generate Java code for hosting an RPyC service using jpy.
generate_jpy_rpyc_service(Options, Code) :-
    option(port(Port), Options, 18812),
    option(package(Package), Options, "io.unifyweaver.bridge"),
    option(class_name(ClassName), Options, "JpyRPyCService"),
    format(atom(Code), '// Generated by UnifyWeaver - jpy RPyC Service Host
// SPDX-License-Identifier: MIT

package ~w;

import org.jpy.PyLib;
import org.jpy.PyModule;
import org.jpy.PyObject;

/**
 * RPyC service host using jpy.
 * Supports bi-directional Java↔Python communication.
 */
public class ~w implements AutoCloseable {
    private PyModule rpyc;
    private PyObject server;
    private boolean closed = false;

    public ~w() {
        if (!PyLib.isPythonRunning()) {
            PyLib.startPython();
        }
    }

    /**
     * Start the RPyC server with optional custom service.
     */
    public void start(int port, String serviceCode) {
        rpyc = PyModule.importModule("rpyc");
        PyModule serverUtils = PyModule.importModule("rpyc.utils.server");
        PyObject ThreadedServer = serverUtils.getAttribute("ThreadedServer");

        PyObject service;
        if (serviceCode != null && !serviceCode.isEmpty()) {
            // Execute custom service definition
            PyLib.executeCode(serviceCode, PyLib.getCurrentGlobals(), PyLib.getCurrentLocals());
            service = PyLib.getCurrentLocals().call("__getitem__", "CustomService");
        } else {
            service = rpyc.getAttribute("Service");
        }

        server = ThreadedServer.call(service, port);
        System.out.println("jpy RPyC server listening on port " + port);
        server.call("start");
    }

    public void start(int port) {
        start(port, null);
    }

    public void start() {
        start(~d);
    }

    public void stop() {
        if (server != null) {
            server.call("close");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            stop();
            closed = true;
        }
    }
}
', [Package, ClassName, ClassName, Port]).

%% generate_jpy_gradle(+Options, -GradleContent)
%  Generate build.gradle for jpy project.
generate_jpy_gradle(Options, GradleContent) :-
    option(group(Group), Options, "io.unifyweaver"),
    option(version(Version), Options, "0.1.0"),
    format(atom(GradleContent), 'plugins {
    id ''java-library''
}

group = ''~w''
version = ''~w''

repositories {
    mavenCentral()
}

dependencies {
    // jpy - bi-directional Java/Python bridge
    implementation ''org.jpy:jpy:0.14.0''

    testImplementation ''org.junit.jupiter:junit-jupiter:5.10.0''
}

test {
    useJUnitPlatform()

    // jpy requires these system properties
    systemProperty ''jpy.jpyLib'', System.getProperty(''jpy.jpyLib'', '''')
    systemProperty ''jpy.jdlLib'', System.getProperty(''jpy.jdlLib'', '''')
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(17)
    }
}
', [Group, Version]).

% ============================================================================
% PYO3 (RUST) CODE GENERATION
% ============================================================================

%% generate_pyo3_rpyc_client(+Options, -Code)
%  Generate Rust code for an RPyC client using PyO3.
generate_pyo3_rpyc_client(Options, Code) :-
    option(host(Host), Options, "localhost"),
    option(port(Port), Options, 18812),
    option(crate_name(CrateName), Options, "rpyc_client"),
    format(atom(Code), '//! Generated by UnifyWeaver - PyO3 RPyC Client
//! SPDX-License-Identifier: MIT

use pyo3::prelude::*;

/// RPyC client using PyO3 to embed CPython in Rust.
/// Provides live object proxies over the network.
pub struct RPyCClient {
    conn: Py<PyAny>,
}

impl RPyCClient {
    /// Connect to an RPyC server.
    pub fn connect(host: &str, port: u16) -> PyResult<Self> {
        Python::with_gil(|py| {
            let rpyc = py.import_bound("rpyc")?;
            let conn = rpyc
                .getattr("classic")?
                .call_method1("connect", (host, port))?;
            Ok(Self { conn: conn.into() })
        })
    }

    /// Connect with default host and port.
    pub fn connect_default() -> PyResult<Self> {
        Self::connect("~w", ~d)
    }

    /// Get a remote module.
    pub fn get_module(&self, module_name: &str) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let conn = self.conn.bind(py);
            let modules = conn.getattr("modules")?;
            Ok(modules.getattr(module_name)?.into())
        })
    }

    /// Call a function on a remote module.
    pub fn call<T: for<''a> FromPyObject<''a>>(
        &self,
        module_name: &str,
        function_name: &str,
        args: impl IntoPy<Py<PyTuple>>,
    ) -> PyResult<T> {
        Python::with_gil(|py| {
            let conn = self.conn.bind(py);
            let module = conn.getattr("modules")?.getattr(module_name)?;
            module.call_method1(function_name, args)?.extract()
        })
    }

    /// Access the root namespace for exposed methods.
    pub fn root(&self) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let conn = self.conn.bind(py);
            Ok(conn.getattr("root")?.into())
        })
    }

    /// Close the connection.
    pub fn close(&self) -> PyResult<()> {
        Python::with_gil(|py| {
            self.conn.bind(py).call_method0("close")?;
            Ok(())
        })
    }
}

impl Drop for RPyCClient {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn main() -> PyResult<()> {
    println!("PyO3 + RPyC Integration (~w)");

    let client = RPyCClient::connect_default()?;

    // Example: call math.sqrt
    let result: f64 = client.call("math", "sqrt", (16.0_f64,))?;
    println!("math.sqrt(16) = {}", result);

    Ok(())
}
', [Host, Port, CrateName]).

%% generate_pyo3_cargo_toml(+Options, -TomlContent)
%  Generate Cargo.toml for PyO3 project.
generate_pyo3_cargo_toml(Options, TomlContent) :-
    option(crate_name(CrateName), Options, "rpyc_client"),
    option(version(Version), Options, "0.1.0"),
    format(atom(TomlContent), '[package]
name = "~w"
version = "~w"
edition = "2021"

[dependencies]
pyo3 = { version = "0.22", features = ["auto-initialize"] }
', [CrateName, Version]).

% ============================================================================
% PYCALL.RB (RUBY) CODE GENERATION
% ============================================================================

%% generate_pycall_rb_rpyc_client(+Options, -Code)
%  Generate Ruby code for an RPyC client using PyCall.rb.
generate_pycall_rb_rpyc_client(Options, Code) :-
    option(host(Host), Options, "localhost"),
    option(port(Port), Options, 18812),
    option(class_name(ClassName), Options, "RPyCClient"),
    format(atom(Code), '#!/usr/bin/env ruby
# frozen_string_literal: true

# Generated by UnifyWeaver - PyCall.rb RPyC Client
# SPDX-License-Identifier: MIT

require ''pycall''

# RPyC client using PyCall.rb to embed CPython in Ruby.
# Provides live object proxies over the network.
class ~w
  attr_reader :conn

  def initialize(host = ''~w'', port = ~d)
    @rpyc = PyCall.import_module(''rpyc'')
    @conn = @rpyc.classic.connect(host, port)
  end

  # Get a remote module
  def get_module(module_name)
    conn.modules.send(module_name)
  end

  # Call a function on a remote module
  # Note: Use .to_f on numeric results for Ruby operations
  def call(module_name, function_name, *args)
    mod = get_module(module_name)
    mod.send(function_name, *args)
  end

  # Access the root namespace for exposed methods
  def root
    conn.root
  end

  # Call an exposed method on the server
  def call_exposed(method_name, *args)
    root.send(method_name, *args)
  end

  # Close the connection
  def close
    conn.close
  end
end

# Example usage
if __FILE__ == $PROGRAM_NAME
  puts ''PyCall.rb + RPyC Integration (~w)''

  client = ~w.new

  # Call math.sqrt (use .to_f for Ruby numeric operations)
  result = client.call(''math'', ''sqrt'', 16).to_f
  puts "math.sqrt(16) = #{result}"

  client.close
end
', [ClassName, Host, Port, ClassName, ClassName]).

%% generate_pycall_rb_gemfile(+Options, -GemfileContent)
%  Generate Gemfile for PyCall.rb project.
generate_pycall_rb_gemfile(_Options, GemfileContent) :-
    format(atom(GemfileContent), '# frozen_string_literal: true

source ''https://rubygems.org''

gem ''pycall'', ''~> 1.5''
', []).

% ============================================================================
% GENERIC INTERFACE
% ============================================================================

%% generate_python_bridge_client(+Bridge, +Options, -Code)
%  Generate client code for the specified bridge.
generate_python_bridge_client(pythonnet, Options, Code) :-
    generate_pythonnet_rpyc_client(Options, Code).
generate_python_bridge_client(csnakes, Options, Code) :-
    generate_csnakes_rpyc_client(Options, Code).
generate_python_bridge_client(jpype, Options, Code) :-
    generate_jpype_rpyc_client(Options, Code).
generate_python_bridge_client(jpy, Options, Code) :-
    generate_jpy_rpyc_client(Options, Code).
generate_python_bridge_client(pyo3, Options, Code) :-
    generate_pyo3_rpyc_client(Options, Code).
generate_python_bridge_client(pycall_rb, Options, Code) :-
    generate_pycall_rb_rpyc_client(Options, Code).

%% generate_python_bridge_service(+Bridge, +Options, -Code)
%  Generate service code for the specified bridge.
generate_python_bridge_service(pythonnet, Options, Code) :-
    generate_pythonnet_rpyc_service(Options, Code).
generate_python_bridge_service(csnakes, Options, Code) :-
    generate_csnakes_rpyc_service(Options, Code).
generate_python_bridge_service(jpype, Options, Code) :-
    generate_jpype_rpyc_service(Options, Code).
generate_python_bridge_service(jpy, Options, Code) :-
    generate_jpy_rpyc_service(Options, Code).

% ============================================================================
% TESTING
% ============================================================================

test_python_bridges_glue :-
    format('~n=== Python Bridges Glue Tests ===~n~n'),

    % Test bridge detection
    format('Bridge Detection:~n'),
    (detect_pythonnet(PNAvail) -> format('  Python.NET: ~w~n', [PNAvail]) ; format('  Python.NET: error~n')),
    (detect_csnakes(CSAvail) -> format('  CSnakes: ~w~n', [CSAvail]) ; format('  CSnakes: error~n')),
    (detect_jpype(JPAvail) -> format('  JPype: ~w~n', [JPAvail]) ; format('  JPype: error~n')),
    (detect_jpy(JpyAvail) -> format('  jpy: ~w~n', [JpyAvail]) ; format('  jpy: error~n')),
    (detect_pyo3(PyO3Avail) -> format('  PyO3: ~w~n', [PyO3Avail]) ; format('  PyO3: error~n')),
    (detect_pycall_rb(PyCallAvail) -> format('  PyCall.rb: ~w~n', [PyCallAvail]) ; format('  PyCall.rb: error~n')),

    % Test code generation
    format('~nCode Generation:~n'),

    % Python.NET
    (   generate_pythonnet_rpyc_client([host("localhost"), port(18812)], PNCode),
        atom_length(PNCode, PNLen),
        format('  Python.NET client: ~d chars~n', [PNLen])
    ;   format('  Python.NET client: FAILED~n')
    ),

    % CSnakes
    (   generate_csnakes_rpyc_client([host("localhost"), port(18812)], CSCode),
        atom_length(CSCode, CSLen),
        format('  CSnakes client: ~d chars~n', [CSLen])
    ;   format('  CSnakes client: FAILED~n')
    ),

    % JPype
    (   generate_jpype_rpyc_client([host("localhost"), port(18812)], JPCode),
        atom_length(JPCode, JPLen),
        format('  JPype client: ~d chars~n', [JPLen])
    ;   format('  JPype client: FAILED~n')
    ),

    % jpy
    (   generate_jpy_rpyc_client([host("localhost"), port(18812)], JpyCode),
        atom_length(JpyCode, JpyLen),
        format('  jpy client: ~d chars~n', [JpyLen])
    ;   format('  jpy client: FAILED~n')
    ),

    % PyO3
    (   generate_pyo3_rpyc_client([host("localhost"), port(18812)], PyO3Code),
        atom_length(PyO3Code, PyO3Len),
        format('  PyO3 client: ~d chars~n', [PyO3Len])
    ;   format('  PyO3 client: FAILED~n')
    ),

    % PyCall.rb
    (   generate_pycall_rb_rpyc_client([host("localhost"), port(18812)], PyCallCode),
        atom_length(PyCallCode, PyCallLen),
        format('  PyCall.rb client: ~d chars~n', [PyCallLen])
    ;   format('  PyCall.rb client: FAILED~n')
    ),

    format('~n=== Tests Complete ===~n').

% Helper for option extraction
option(Option, Options, _Default) :-
    member(Option, Options), !.
option(Option, _Options, Default) :-
    Option =.. [Name, Default],
    atom(Name).
