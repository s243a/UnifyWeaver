% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% janus_demo.pl - Janus In-Process Python↔Prolog Demo
%
% This example demonstrates using Janus for direct in-process
% communication between Prolog and Python, integrated with
% UnifyWeaver's glue system.
%
% Usage:
%   ?- [janus_demo].
%   ?- run_demo.

:- use_module(library(janus)).
:- use_module('../../src/unifyweaver/glue/janus_glue').

%% ============================================
%% Demo 1: Basic Python Calls
%% ============================================

demo_basic :-
    format('~n=== Demo 1: Basic Python Calls ===~n'),

    % Call math.sqrt directly
    format('1.1 Calling math.sqrt(16):~n'),
    janus_call_python(math, sqrt, [16], SqrtResult),
    format('    Result: ~w~n', [SqrtResult]),

    % Call json.dumps
    format('1.2 Calling json.dumps:~n'),
    py_call(json:dumps([1, 2, 3]), JsonStr),
    format('    Result: ~w~n', [JsonStr]),

    % Call builtins.sum
    format('1.3 Calling sum([1,2,3,4,5]):~n'),
    py_call(builtins:sum([1, 2, 3, 4, 5]), SumResult),
    format('    Result: ~w~n', [SumResult]),

    % Use len function
    format('1.4 Calling len():~n'),
    py_call(builtins:len([1, 2, 3, 4, 5]), Len),
    format('    len([1,2,3,4,5]) = ~w~n', [Len]).

%% ============================================
%% Demo 2: NumPy Integration
%% ============================================

demo_numpy :-
    format('~n=== Demo 2: NumPy Integration ===~n'),

    (   janus_numpy_available
    ->  % Create NumPy array and get basic stats
        format('2.1 Creating NumPy array and computing mean:~n'),
        janus_numpy_array([1, 2, 3, 4, 5], Arr),
        janus_numpy_call(mean, [Arr], Mean),
        format('    mean([1,2,3,4,5]) = ~w~n', [Mean]),

        % Sum and std
        format('2.2 Statistical functions:~n'),
        janus_numpy_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], DataArr),
        janus_numpy_call(sum, [DataArr], Sum),
        janus_numpy_call(std, [DataArr], Std),
        format('    sum = ~w, std = ~w~n', [Sum, Std]),

        % Additional NumPy operations
        format('2.3 Additional NumPy operations:~n'),
        janus_numpy_call(min, [DataArr], Min),
        janus_numpy_call(max, [DataArr], Max),
        format('    min = ~w, max = ~w~n', [Min, Max])

    ;   format('NumPy not available. Install with: pip install numpy~n')
    ).

%% ============================================
%% Demo 3: Using Standard Library Modules
%% ============================================

demo_custom_module :-
    format('~n=== Demo 3: Using Standard Library Modules ===~n'),

    % Using math module functions
    format('3.1 Math module functions:~n'),
    py_call(math:factorial(10), Fact10),
    format('    math.factorial(10) = ~w~n', [Fact10]),

    py_call(math:pow(2, 10), Power),
    format('    math.pow(2, 10) = ~w~n', [Power]),

    % Using collections
    format('3.2 Collections module:~n'),
    py_call(collections:'Counter'([a, b, a, c, b, b]), Counter),
    format('    Counter([a,b,a,c,b,b]) = ~w~n', [Counter]),

    % Using os module
    format('3.3 OS module:~n'),
    py_call(os:getcwd(), Cwd),
    format('    os.getcwd() = ~w~n', [Cwd]),

    % Using platform module
    format('3.4 Platform info:~n'),
    py_call(platform:python_version(), PyVer),
    format('    Python version: ~w~n', [PyVer]).

%% ============================================
%% Demo 4: Bidirectional Calling (Notes)
%% ============================================

% Define Prolog predicates that Python can call back to
ancestor(tom, bob).
ancestor(bob, pat).
ancestor(pat, jim).

ancestor_chain(X, Y) :- ancestor(X, Y).
ancestor_chain(X, Y) :- ancestor(X, Z), ancestor_chain(Z, Y).

demo_bidirectional :-
    format('~n=== Demo 4: Bidirectional Calling ===~n'),
    format('~n'),
    format('Janus supports bidirectional calling (Python->Prolog).~n'),
    format('This requires the janus_swi Python package:~n'),
    format('    pip install janus_swi~n'),
    format('~n'),
    format('Example Python code to query Prolog:~n'),
    format('    from janus_swi import Query~n'),
    format('    for sol in Query(\"ancestor_chain(X, jim)\"):~n'),
    format('        print(sol[\"X\"])~n'),
    format('~n'),
    format('See the JanusBridge project for more examples.~n').

%% ============================================
%% Demo 5: Wrapper Generation
%% ============================================

demo_wrapper_generation :-
    format('~n=== Demo 5: Wrapper Generation ===~n'),

    format('5.1 Generating Janus wrapper for matrix_inverse/1:~n'),
    generate_janus_wrapper(matrix_inverse/1,
        [module(numpy_linalg), function(inv)],
        WrapperCode),
    format('~w~n', [WrapperCode]),

    format('5.2 Generating pipeline with Janus:~n'),
    generate_janus_pipeline([
        step(preprocess, python, transform/1),
        step(analyze, python, analyze/1),
        step(postprocess, prolog, format_output/1)
    ], [], PipelineCode),
    format('~w~n', [PipelineCode]).

%% ============================================
%% Performance Comparison
%% ============================================

demo_performance :-
    format('~n=== Demo 6: Performance Comparison ===~n'),

    format('6.1 In-process (Janus) vs Pipe comparison:~n'),

    % Time Janus call
    format('    Timing 1000 Janus calls to math.sqrt...~n'),
    get_time(T1),
    forall(between(1, 1000, _), janus_call_python(math, sqrt, [12345], _)),
    get_time(T2),
    JanusTime is T2 - T1,
    format('    Janus time: ~3f seconds~n', [JanusTime]),

    format('    (Pipe comparison would require spawning Python process)~n'),
    format('    Janus avoids process spawn overhead entirely.~n').

%% ============================================
%% Demo 7: Dynamic Code Execution (Exec Pattern)
%% ============================================

demo_exec_pattern :-
    format('~n=== Demo 7: Dynamic Code Execution ===~n'),

    % First, we need to add the path to dict_wrapper
    format('7.1 Setting up dict_wrapper path:~n'),
    % This path should point to JanusBridge's dict_wrapper.py
    DictWrapperPath = '../../context/projects/JanusBridge/src/core',
    catch(
        (   janus_add_lib_path(DictWrapperPath),
            format('    Added path: ~w~n', [DictWrapperPath])
        ),
        _,
        format('    Warning: Could not add dict_wrapper path~n')
    ),

    % Test non-recursive function with wrapped_exec
    format('~n7.2 Non-recursive function (janus_wrapped_exec):~n'),
    catch(
        (   janus_wrapped_exec("
def double(x):
    return x * 2

def add(a, b):
    return a + b
", NS1),
            janus_call_defined(NS1, double, [21], R1),
            janus_call_defined(NS1, add, [10, 32], R2),
            format('    double(21) = ~w~n', [R1]),
            format('    add(10, 32) = ~w~n', [R2])
        ),
        E1,
        format('    Skipped (dict_wrapper not available): ~w~n', [E1])
    ),

    % Test recursive function with janus_exec_recursive
    format('~n7.3 Recursive function (janus_exec_recursive):~n'),
    catch(
        (   janus_exec_recursive("
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
", NS2),
            janus_call_defined(NS2, factorial, [5], Fact),
            janus_call_defined(NS2, fibonacci, [10], Fib),
            format('    factorial(5) = ~w~n', [Fact]),
            format('    fibonacci(10) = ~w~n', [Fib])
        ),
        E2,
        format('    Skipped: ~w~n', [E2])
    ),

    format('~nNote: The exec pattern is useful for:~n'),
    format('  - Dynamically generated Python code~n'),
    format('  - Custom algorithms defined at runtime~n'),
    format('  - Code compiled by UnifyWeaver~n').

%% ============================================
%% Main Demo Runner
%% ============================================

run_demo :-
    format('~n========================================~n'),
    format('Janus In-Process Python Integration Demo~n'),
    format('========================================~n'),

    % Check Janus availability
    (   janus_available(Version)
    ->  format('Janus available: ~w~n', [Version]),
        demo_basic,
        demo_numpy,
        demo_custom_module,
        demo_bidirectional,
        demo_wrapper_generation,
        demo_performance,
        demo_exec_pattern
    ;   format('ERROR: Janus not available.~n'),
        format('Ensure you are using SWI-Prolog 9.0+ with Janus support.~n'),
        format('Install Python package: pip install janus_swi~n')
    ),

    format('~n========================================~n'),
    format('Demo complete!~n'),
    format('========================================~n').

% Auto-run on load
:- initialization(format('~nJanus demo loaded. Run ?- run_demo. to start.~n')).
