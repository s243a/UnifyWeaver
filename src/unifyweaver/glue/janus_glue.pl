% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% janus_glue.pl - In-Process Python↔Prolog Communication via Janus
%
% This module provides glue code for direct in-process communication
% between Prolog and Python using SWI-Prolog's Janus library.
%
% Unlike pipe-based communication, Janus provides:
% - Zero serialization overhead for compatible types
% - Direct function calls without process spawning
% - Shared memory for large data structures
% - Bidirectional calling (Prolog→Python and Python→Prolog)
%
% Requirements:
% - SWI-Prolog 9.0+ with Janus support
% - Python 3.8+ with janus-swi package
%
% Usage:
%   ?- use_module(library(janus)).
%   ?- janus_call_python(my_module, my_function, [arg1, arg2], Result).

:- module(janus_glue, [
    % Janus availability
    janus_available/0,
    janus_available/1,

    % Python module management
    janus_import_module/2,
    janus_reload_module/2,
    janus_add_lib_path/1,
    janus_add_lib_path/2,

    % Function calling
    janus_call_python/4,
    janus_call_method/4,

    % Object management
    janus_create_object/4,
    janus_get_attribute/3,
    janus_set_attribute/3,

    % NumPy integration
    janus_numpy_available/0,
    janus_numpy_array/2,
    janus_numpy_call/3,

    % Dynamic code execution (exec pattern)
    janus_wrapped_exec/2,
    janus_wrapped_exec/3,
    janus_call_defined/4,
    janus_exec_recursive/2,

    % Code generation for compiled predicates
    generate_janus_wrapper/3,
    generate_janus_pipeline/3,

    % Glue type registration
    janus_transport_type/1,

    % Testing
    test_janus_glue/0
]).

:- use_module(library(lists)).

% Conditionally load Janus if available
:- catch(use_module(library(janus)), _, true).

%% ============================================
%% Janus Availability Detection
%% ============================================

%% janus_available
%  True if Janus library is loaded and functional.
janus_available :-
    janus_available(_).

%% janus_available(-Version)
%  Check Janus availability and return version info.
janus_available(Version) :-
    catch(
        (   current_module(janus),
            py_call(sys:version, PyVersion),
            Version = janus(PyVersion)
        ),
        _,
        fail
    ).

%% ============================================
%% Python Module Management
%% ============================================

%% janus_import_module(+ModuleName, -ModuleRef)
%  Import a Python module and return a reference to it.
%
%  Example:
%    ?- janus_import_module(numpy, NP).
%    ?- janus_call_method(NP, array, [[1,2,3]], Arr).
%
janus_import_module(ModuleName, ModuleRef) :-
    atom(ModuleName),
    py_call(importlib:import_module(ModuleName), ModuleRef).

%% janus_reload_module(+ModuleName, -ModuleRef)
%  Reload a Python module (useful during development).
janus_reload_module(ModuleName, ModuleRef) :-
    atom(ModuleName),
    py_call(importlib:import_module(ModuleName), TempRef),
    py_call(importlib:reload(TempRef), ModuleRef).

%% janus_add_lib_path(+Path)
%  Add a directory to Python's sys.path.
%  Uses py_add_lib_dir which is the proper Janus way to add paths.
janus_add_lib_path(Path) :-
    catch(
        py_add_lib_dir(Path),
        _,
        py_call(sys:path:append(Path), _)  % Fallback
    ).

%% janus_add_lib_path(+Path, +Position)
%  Add a directory to Python's sys.path at a specific position.
%  Position is 'first' or 'last'.
janus_add_lib_path(Path, Position) :-
    catch(
        py_add_lib_dir(Path, Position),
        _,
        py_call(sys:path:append(Path), _)  % Fallback
    ).

%% ============================================
%% Function Calling
%% ============================================

%% janus_call_python(+Module, +Function, +Args, -Result)
%  Call a Python function by module and function name.
%
%  Example:
%    ?- janus_call_python(math, sqrt, [16], R).
%    R = 4.0.
%
janus_call_python(Module, Function, Args, Result) :-
    janus_import_module(Module, ModRef),
    janus_call_function(ModRef, Function, Args, Result).

%% janus_call_function(+ModuleRef, +Function, +Args, -Result)
%  Call a function on an imported module reference.
janus_call_function(ModRef, Function, Args, Result) :-
    build_py_call(ModRef, Function, Args, PyCall),
    py_call(PyCall, Result).

%% build_py_call(+ModRef, +Function, +Args, -PyCall)
%  Build the py_call term dynamically.
build_py_call(ModRef, Function, [], PyCall) :-
    PyCall = ModRef:Function.
build_py_call(ModRef, Function, Args, PyCall) :-
    Args \= [],
    ArgsWithParens =.. [Function|Args],
    PyCall = ModRef:ArgsWithParens.

%% janus_call_method(+Object, +Method, +Args, -Result)
%  Call a method on a Python object.
%
%  Example:
%    ?- janus_create_object(collections, 'Counter', [[a,b,a,c,b,b]], Counter),
%       janus_call_method(Counter, most_common, [2], Top2).
%
janus_call_method(Object, Method, Args, Result) :-
    build_py_call(Object, Method, Args, PyCall),
    py_call(PyCall, Result).

%% ============================================
%% Object Management
%% ============================================

%% janus_create_object(+Module, +ClassName, +Args, -Object)
%  Create a Python object instance.
%
%  Example:
%    ?- janus_create_object(datetime, datetime, [2025, 1, 1], DT).
%
janus_create_object(Module, ClassName, Args, Object) :-
    janus_import_module(Module, ModRef),
    build_py_call(ModRef, ClassName, Args, PyCall),
    py_call(PyCall, Object).

%% janus_get_attribute(+Object, +AttrName, -Value)
%  Get an attribute from a Python object.
janus_get_attribute(Object, AttrName, Value) :-
    py_call(Object:AttrName, Value).

%% janus_set_attribute(+Object, +AttrName, +Value)
%  Set an attribute on a Python object.
janus_set_attribute(Object, AttrName, Value) :-
    py_call(setattr(Object, AttrName, Value), _).

%% ============================================
%% NumPy Integration
%% ============================================

%% janus_numpy_available
%  True if NumPy is available via Janus.
janus_numpy_available :-
    catch(
        (   janus_import_module(numpy, _),
            true
        ),
        _,
        fail
    ).

%% janus_numpy_array(+List, -NumpyArray)
%  Convert a Prolog list to a NumPy array.
janus_numpy_array(List, NumpyArray) :-
    janus_import_module(numpy, NP),
    py_call(NP:array(List), NumpyArray).

%% janus_numpy_call(+Function, +Args, -Result)
%  Call a NumPy function.
%
%  Example:
%    ?- janus_numpy_call(linalg:inv, [[[1,2],[3,4]]], Inverse).
%
janus_numpy_call(Function, Args, Result) :-
    janus_import_module(numpy, NP),
    (   Function = SubMod:Func
    ->  py_call(NP:SubMod, SubModRef),
        janus_call_function(SubModRef, Func, Args, Result)
    ;   janus_call_function(NP, Function, Args, Result)
    ).

%% ============================================
%% Dynamic Code Execution (Exec Pattern)
%% ============================================
%%
%% The exec pattern allows dynamically defining Python functions from Prolog.
%% There are two main approaches:
%%
%% 1. Non-recursive functions: Use janus_wrapped_exec/2
%%    Functions are defined in a namespace and called via janus_call_defined/4.
%%
%% 2. Recursive functions: Use janus_exec_recursive/2
%%    Uses a self-referencing namespace so functions can call themselves.
%%
%% IMPORTANT: This requires dict_wrapper.py from JanusBridge or similar.
%% Add the path with: janus_add_lib_path('/path/to/JanusBridge/src/core').

%% janus_wrapped_exec(+Code, -Namespace)
%  Execute Python code and return a namespace (DictWrapper) with definitions.
%  Non-recursive functions work fine.
%
%  Example:
%    ?- janus_wrapped_exec("
%         def double(x):
%             return x * 2
%       ", NS),
%       janus_call_defined(NS, double, [21], Result).
%    Result = 42.
%
janus_wrapped_exec(Code, Namespace) :-
    janus_import_module(dict_wrapper, DW),
    py_call(DW:wrapped_exec(Code), Namespace).

%% janus_wrapped_exec(+Code, +InitialVars, -Namespace)
%  Execute Python code with initial variables.
janus_wrapped_exec(Code, InitialVars, Namespace) :-
    janus_import_module(dict_wrapper, DW),
    py_call(DW:wrapped_exec(Code, InitialVars), Namespace).

%% janus_call_defined(+Namespace, +FuncName, +Args, -Result)
%  Call a function that was defined in a wrapped_exec namespace.
%
%  This handles the indirection needed because Janus can't directly
%  call Python callables that are stored in variables.
%
janus_call_defined(Namespace, FuncName, Args, Result) :-
    atom_string(FuncName, FuncNameStr),
    janus_import_module(dict_wrapper, DW),
    py_call(Namespace:get(FuncNameStr), Func),
    % Use call_with_args to properly expand the Args list
    py_call(DW:call_with_args(Func, Args), Result).

%% janus_exec_recursive(+Code, -Namespace)
%  Execute Python code where defined functions can call themselves recursively.
%
%  The key difference from janus_wrapped_exec is that this uses
%  the same namespace for both globals and locals, allowing a function
%  to find itself when making recursive calls.
%
%  Example:
%    ?- janus_exec_recursive("
%         def factorial(n):
%             if n <= 1:
%                 return 1
%             return n * factorial(n - 1)
%       ", NS),
%       janus_call_defined(NS, factorial, [5], Result).
%    Result = 120.
%
janus_exec_recursive(Code, Namespace) :-
    % Execute with namespace as both globals and locals
    % This allows defined functions to find themselves
    %
    % Strategy: Define a helper function in Python that:
    % 1. Executes the code with same dict as globals and locals
    % 2. Returns a DictWrapper with all definitions
    janus_import_module(dict_wrapper, DW),
    % Define the recursive exec helper - it uses exec(code, ns, ns)
    % which makes defined functions visible to themselves
    HelperCode = "
def _exec_recursive_helper(code):
    import builtins as _bi
    ns = {'__builtins__': _bi}
    exec(code, ns, ns)
    # Remove builtins from result to keep it clean
    ns.pop('__builtins__', None)
    ns.pop('_bi', None)
    return ns
",
    py_call(DW:wrapped_exec(HelperCode), HelperNS),
    py_call(HelperNS:get('_exec_recursive_helper'), ExecHelper),
    py_call(DW:call_function(ExecHelper, Code), ResultDict),
    py_call(DW:'DictWrapper'(ResultDict), Namespace).

%% ============================================
%% Code Generation for Compiled Predicates
%% ============================================

%% generate_janus_wrapper(+Predicate, +Options, -Code)
%  Generate Prolog code that wraps a compiled Python predicate
%  for in-process execution via Janus.
%
%  Options:
%    - module(ModuleName): Python module name
%    - function(FuncName): Python function name
%    - numpy(true/false): Whether to convert arrays
%
generate_janus_wrapper(Pred/Arity, Options, Code) :-
    (   member(module(Module), Options) -> true ; Module = Pred ),
    (   member(function(Func), Options) -> true ; Func = Pred ),

    % Generate argument names
    generate_arg_names(Arity, ArgNames),
    atomic_list_concat(ArgNames, ', ', ArgsStr),

    % Generate the wrapper predicate
    atom_string(Pred, PredStr),
    atom_string(Module, ModStr),
    atom_string(Func, FuncStr),

    format(string(Code),
"% Janus wrapper for ~w/~w
% Calls Python function ~w.~w in-process
~w(~w, Result) :-
    janus_glue:janus_call_python(~w, ~w, [~w], Result).
", [Pred, Arity, Module, Func, PredStr, ArgsStr, ModStr, FuncStr, ArgsStr]).

%% generate_arg_names(+Arity, -Names)
generate_arg_names(0, []) :- !.
generate_arg_names(Arity, Names) :-
    Arity > 0,
    findall(Name, (
        between(1, Arity, I),
        format(atom(Name), 'Arg~w', [I])
    ), Names).

%% generate_janus_pipeline(+Steps, +Options, -Code)
%  Generate a pipeline that uses Janus for Python steps.
%
%  Steps format: [step(name, target, pred/arity), ...]
%
generate_janus_pipeline(Steps, Options, Code) :-
    % Generate wrapper for each Python step
    findall(WrapperCode, (
        member(step(_, python, Pred/Arity), Steps),
        generate_janus_wrapper(Pred/Arity, Options, WrapperCode)
    ), WrapperCodes),

    % Generate pipeline orchestration
    generate_pipeline_runner(Steps, Options, RunnerCode),

    % Combine
    atomic_list_concat(WrapperCodes, '\n', WrappersStr),
    format(string(Code),
"% Janus Pipeline
% Generated by UnifyWeaver

:- use_module(library(janus)).
:- use_module('src/unifyweaver/glue/janus_glue').

~w

~w
", [WrappersStr, RunnerCode]).

%% generate_pipeline_runner(+Steps, +Options, -Code)
generate_pipeline_runner(Steps, _Options, Code) :-
    % Generate step calls
    findall(StepCall, (
        member(step(Name, Target, Pred/Arity), Steps),
        generate_step_call(Name, Target, Pred, Arity, StepCall)
    ), StepCalls),
    atomic_list_concat(StepCalls, ',\n    ', StepsStr),

    format(string(Code),
"run_pipeline(Input, Output) :-
    ~w.
", [StepsStr]).

generate_step_call(Name, python, Pred, _Arity, Call) :-
    format(atom(Call), "~w(~w_in, ~w_out)", [Pred, Name, Name]).
generate_step_call(Name, prolog, Pred, _Arity, Call) :-
    format(atom(Call), "~w(~w_in, ~w_out)", [Pred, Name, Name]).

%% ============================================
%% Transport Type Registration
%% ============================================

%% janus_transport_type(-Type)
%  Register Janus as a transport type for the glue system.
janus_transport_type(janus).

%% ============================================
%% Testing
%% ============================================

%% test_janus_glue
%  Run tests for Janus glue functionality.
test_janus_glue :-
    format('Testing Janus glue...~n'),

    % Test availability
    format('1. Testing Janus availability...~n'),
    (   janus_available(Version)
    ->  format('   Janus available: ~w~n', [Version])
    ;   format('   Janus NOT available (tests will skip Python calls)~n')
    ),

    % Test wrapper generation
    format('2. Testing wrapper generation...~n'),
    generate_janus_wrapper(my_func/2, [module(my_module)], WrapperCode),
    format('   Generated wrapper:~n~w~n', [WrapperCode]),

    % Test NumPy availability (if Janus available)
    format('3. Testing NumPy availability...~n'),
    (   janus_available
    ->  (   janus_numpy_available
        ->  format('   NumPy available via Janus~n')
        ;   format('   NumPy NOT available~n')
        )
    ;   format('   Skipped (Janus not available)~n')
    ),

    % Test exec pattern (if Janus and dict_wrapper available)
    format('4. Testing exec pattern...~n'),
    (   janus_available
    ->  test_exec_pattern
    ;   format('   Skipped (Janus not available)~n')
    ),

    format('Janus glue tests complete.~n').

%% test_exec_pattern
%  Test the dynamic code execution pattern.
test_exec_pattern :-
    format('   Testing non-recursive function...~n'),
    catch(
        (   janus_wrapped_exec("
def double(x):
    return x * 2
", NS1),
            janus_call_defined(NS1, double, [21], R1),
            format('      double(21) = ~w~n', [R1])
        ),
        E1,
        format('      Error (dict_wrapper not loaded?): ~w~n', [E1])
    ),

    format('   Testing recursive function...~n'),
    catch(
        (   janus_exec_recursive("
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
", NS2),
            janus_call_defined(NS2, factorial, [5], R2),
            format('      factorial(5) = ~w~n', [R2])
        ),
        E2,
        format('      Error: ~w~n', [E2])
    ).
