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
% Usage:
%   ?- generate_pythonnet_rpyc_client(Options, CSharpCode).
%   ?- generate_csnakes_rpyc_client(Options, CSharpCode).
%   ?- generate_jpype_rpyc_client(Options, JavaCode).
%   ?- generate_jpy_rpyc_client(Options, JavaCode).

:- module(python_bridges_glue, [
    % Bridge detection
    detect_pythonnet/1,             % detect_pythonnet(-Available)
    detect_csnakes/1,               % detect_csnakes(-Available)
    detect_jpype/1,                 % detect_jpype(-Available)
    detect_jpy/1,                   % detect_jpy(-Available)

    % Bridge selection
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

    % Generic interface
    generate_python_bridge_client/3,     % generate_python_bridge_client(+Bridge, +Options, -Code)
    generate_python_bridge_service/3,    % generate_python_bridge_service(+Bridge, +Options, -Code)

    % Testing
    test_python_bridges_glue/0
]).

:- use_module(library(lists)).

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

% ============================================================================
% BRIDGE SELECTION
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

    format('~n=== Tests Complete ===~n').

% Helper for option extraction
option(Option, Options, _Default) :-
    member(Option, Options), !.
option(Option, _Options, Default) :-
    Option =.. [Name, Default],
    atom(Name).
