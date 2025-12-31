/*
 * SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Integration tests for TypeScript glue modules:
 * - rpyc_security
 * - express_generator
 * - react_generator
 * - full_pipeline
 *
 * Run: swipl -g "consult('tests/integration/glue/test_typescript_glue'), run_tests, halt" -t "halt(1)"
 */

:- use_module('../../../src/unifyweaver/glue/rpyc_security').
:- use_module('../../../src/unifyweaver/glue/express_generator').
:- use_module('../../../src/unifyweaver/glue/react_generator').
:- use_module('../../../src/unifyweaver/glue/full_pipeline').

%% ============================================
%% Test Helpers
%% ============================================

:- dynamic test_passed/1.
:- dynamic test_failed/1.

run_tests :-
    retractall(test_passed(_)),
    retractall(test_failed(_)),
    format('~n========================================~n'),
    format('TypeScript Glue Integration Tests~n'),
    format('========================================~n~n'),
    run_all_tests,
    summarize_results.

run_all_tests :-
    % RPyC Security Tests
    test_rpyc_security_validation,
    test_rpyc_security_whitelist_generation,
    test_rpyc_security_validator_generation,
    test_rpyc_security_middleware_generation,

    % Express Generator Tests
    test_express_endpoint_queries,
    test_express_router_generation,
    test_express_app_generation,

    % React Generator Tests
    test_react_component_queries,
    test_react_form_component_generation,
    test_react_display_component_generation,
    test_react_styles_generation,
    test_react_hooks_generation,

    % Full Pipeline Tests
    test_pipeline_application_queries,
    test_pipeline_file_generation,
    test_pipeline_package_json,
    test_pipeline_dockerfile.

summarize_results :-
    findall(T, test_passed(T), Passed),
    findall(T, test_failed(T), Failed),
    length(Passed, PassCount),
    length(Failed, FailCount),
    Total is PassCount + FailCount,
    format('~n========================================~n'),
    format('Results: ~w/~w tests passed~n', [PassCount, Total]),
    (   FailCount > 0
    ->  format('FAILED TESTS:~n'),
        forall(member(F, Failed), format('  - ~w~n', [F])),
        format('========================================~n'),
        fail
    ;   format('All tests passed!~n'),
        format('========================================~n')
    ).

assert_contains(String, Substring, TestName) :-
    (   sub_atom(String, _, _, _, Substring)
    ->  format('  [PASS] ~w~n', [TestName]),
        assertz(test_passed(TestName))
    ;   format('  [FAIL] ~w - "~w" not found~n', [TestName, Substring]),
        assertz(test_failed(TestName))
    ).

assert_true(Goal, TestName) :-
    (   call(Goal)
    ->  format('  [PASS] ~w~n', [TestName]),
        assertz(test_passed(TestName))
    ;   format('  [FAIL] ~w~n', [TestName]),
        assertz(test_failed(TestName))
    ).

assert_equals(Value, Expected, TestName) :-
    (   Value == Expected
    ->  format('  [PASS] ~w~n', [TestName]),
        assertz(test_passed(TestName))
    ;   format('  [FAIL] ~w - expected ~w, got ~w~n', [TestName, Expected, Value]),
        assertz(test_failed(TestName))
    ).

assert_greater(Value, Min, TestName) :-
    (   Value > Min
    ->  format('  [PASS] ~w (~w > ~w)~n', [TestName, Value, Min]),
        assertz(test_passed(TestName))
    ;   format('  [FAIL] ~w - expected > ~w, got ~w~n', [TestName, Min, Value]),
        assertz(test_failed(TestName))
    ).

%% ============================================
%% RPyC Security Tests
%% ============================================

test_rpyc_security_validation :-
    format('~n--- RPyC Security Validation ---~n'),

    % Test allowed calls
    is_call_allowed(math, sqrt, MathSqrt),
    assert_equals(MathSqrt, true, 'math.sqrt is allowed'),

    is_call_allowed(numpy, mean, NumpyMean),
    assert_equals(NumpyMean, true, 'numpy.mean is allowed'),

    % Test denied calls
    is_call_allowed(os, system, OsSystem),
    assert_equals(OsSystem, false, 'os.system is denied'),

    is_call_allowed(subprocess, call, SubCall),
    assert_equals(SubCall, false, 'subprocess.call is denied'),

    % Test allowed attributes
    is_attr_allowed(math, pi, MathPi),
    assert_equals(MathPi, true, 'math.pi is allowed'),

    is_attr_allowed(numpy, '__version__', NumpyVer),
    assert_equals(NumpyVer, true, 'numpy.__version__ is allowed'),

    % Test validate_call
    validate_call(math, sqrt, Result1),
    assert_equals(Result1, ok, 'validate_call(math, sqrt) returns ok'),

    validate_call(os, system, Result2),
    assert_true(Result2 = error(_), 'validate_call(os, system) returns error').

test_rpyc_security_whitelist_generation :-
    format('~n--- RPyC Security Whitelist Generation ---~n'),

    generate_typescript_whitelist(Code),
    atom_length(Code, Len),
    assert_greater(Len, 1000, 'Whitelist code length > 1000'),

    assert_contains(Code, 'ALLOWED_MODULES', 'Contains ALLOWED_MODULES'),
    assert_contains(Code, 'ALLOWED_ATTRS', 'Contains ALLOWED_ATTRS'),
    assert_contains(Code, 'isCallAllowed', 'Contains isCallAllowed function'),
    assert_contains(Code, 'isAttrAllowed', 'Contains isAttrAllowed function'),
    assert_contains(Code, 'math', 'Contains math module'),
    assert_contains(Code, 'numpy', 'Contains numpy module'),
    assert_contains(Code, 'sqrt', 'Contains sqrt function'),
    assert_contains(Code, 'new Set', 'Uses Set for fast lookup').

test_rpyc_security_validator_generation :-
    format('~n--- RPyC Security Validator Generation ---~n'),

    generate_typescript_validator(Code),
    atom_length(Code, Len),
    assert_greater(Len, 1500, 'Validator code length > 1500'),

    assert_contains(Code, 'RATE_LIMIT', 'Contains RATE_LIMIT config'),
    assert_contains(Code, 'ValidationResult', 'Contains ValidationResult interface'),
    assert_contains(Code, 'validateCall', 'Contains validateCall function'),
    assert_contains(Code, 'validateAttr', 'Contains validateAttr function'),
    assert_contains(Code, 'sanitized', 'Contains sanitized output'),
    assert_contains(Code, 'replace(/[^a-zA-Z0-9_]/g', 'Sanitizes input').

test_rpyc_security_middleware_generation :-
    format('~n--- RPyC Security Middleware Generation ---~n'),

    generate_express_security_middleware(Code),
    atom_length(Code, Len),
    assert_greater(Len, 1000, 'Middleware code length > 1000'),

    assert_contains(Code, 'rateLimiter', 'Contains rateLimiter'),
    assert_contains(Code, 'timeoutMiddleware', 'Contains timeoutMiddleware'),
    assert_contains(Code, 'validateCallMiddleware', 'Contains validateCallMiddleware'),
    assert_contains(Code, 'Request, Response, NextFunction', 'Uses Express types'),
    assert_contains(Code, 'status(429)', 'Has rate limit response'),
    assert_contains(Code, 'status(408)', 'Has timeout response').

%% ============================================
%% Express Generator Tests
%% ============================================

test_express_endpoint_queries :-
    format('~n--- Express Endpoint Queries ---~n'),

    all_endpoints(AllEps),
    length(AllEps, TotalCount),
    assert_greater(TotalCount, 5, 'Has more than 5 endpoints'),

    endpoints_for_module(math, MathEps),
    length(MathEps, MathCount),
    assert_greater(MathCount, 0, 'Has math endpoints'),

    endpoints_for_module(numpy, NumpyEps),
    length(NumpyEps, NumpyCount),
    assert_greater(NumpyCount, 0, 'Has numpy endpoints').

test_express_router_generation :-
    format('~n--- Express Router Generation ---~n'),

    generate_express_router(test_api, Code),
    atom_length(Code, Len),
    assert_greater(Len, 3000, 'Router code length > 3000'),

    assert_contains(Code, 'import { Router', 'Imports Router from express'),
    assert_contains(Code, 'export const', 'Exports router'),
    assert_contains(Code, '.post(', 'Has POST endpoints'),
    assert_contains(Code, '.get(', 'Has GET endpoints'),
    assert_contains(Code, 'validateCall', 'Uses validateCall'),
    assert_contains(Code, 'bridge.call', 'Calls RPyC bridge'),
    assert_contains(Code, 'res.json', 'Returns JSON response'),
    assert_contains(Code, 'catch (error)', 'Has error handling').

test_express_app_generation :-
    format('~n--- Express App Generation ---~n'),

    generate_express_app(test_app, Code),
    atom_length(Code, Len),
    assert_greater(Len, 4000, 'App code length > 4000'),

    assert_contains(Code, 'import express', 'Imports express'),
    assert_contains(Code, 'import cors', 'Imports cors'),
    assert_contains(Code, 'app.use(cors())', 'Uses cors middleware'),
    assert_contains(Code, 'app.use(express.json', 'Uses JSON middleware'),
    assert_contains(Code, '/health', 'Has health check endpoint'),
    assert_contains(Code, 'app.listen', 'Starts server').

%% ============================================
%% React Generator Tests
%% ============================================

test_react_component_queries :-
    format('~n--- React Component Queries ---~n'),

    all_ui_components(AllComps),
    length(AllComps, CompCount),
    assert_greater(CompCount, 2, 'Has more than 2 components'),

    % Check specific components exist
    assert_true(ui_component(numpy_calculator, _), 'numpy_calculator exists'),
    assert_true(ui_component(math_calculator, _), 'math_calculator exists'),
    assert_true(ui_component(math_constants, _), 'math_constants exists').

test_react_form_component_generation :-
    format('~n--- React Form Component Generation ---~n'),

    generate_react_component(numpy_calculator, Code),
    atom_length(Code, Len),
    assert_greater(Len, 3000, 'Component code length > 3000'),

    assert_contains(Code, 'import React', 'Imports React'),
    assert_contains(Code, 'useState', 'Uses useState hook'),
    assert_contains(Code, 'interface', 'Has TypeScript interface'),
    assert_contains(Code, 'React.FC', 'Uses React.FC type'),
    assert_contains(Code, 'className={styles', 'Uses CSS modules'),
    assert_contains(Code, 'onChange=', 'Has input handlers'),
    assert_contains(Code, 'onClick=', 'Has button handlers'),
    assert_contains(Code, 'setLoading', 'Manages loading state'),
    assert_contains(Code, 'setError', 'Manages error state'),
    assert_contains(Code, 'fetch(', 'Makes API calls'),
    assert_contains(Code, 'JSON.stringify', 'Serializes data').

test_react_display_component_generation :-
    format('~n--- React Display Component Generation ---~n'),

    generate_react_component(math_constants, Code),
    atom_length(Code, Len),
    assert_greater(Len, 1000, 'Component code length > 1000'),

    assert_contains(Code, 'useEffect', 'Uses useEffect hook'),
    assert_contains(Code, 'ConstantValue', 'Has ConstantValue interface'),
    assert_contains(Code, 'constants.map', 'Maps over constants'),
    assert_contains(Code, '/api/math/pi', 'Fetches pi'),
    assert_contains(Code, '/api/math/e', 'Fetches e').

test_react_styles_generation :-
    format('~n--- React Styles Generation ---~n'),

    generate_component_styles(numpy_calculator, CSS),
    atom_length(CSS, Len),
    assert_greater(Len, 1000, 'CSS code length > 1000'),

    assert_contains(CSS, '.container', 'Has container class'),
    assert_contains(CSS, '.button', 'Has button class'),
    assert_contains(CSS, '.input', 'Has input class'),
    assert_contains(CSS, '.result', 'Has result class'),
    assert_contains(CSS, '.error', 'Has error class'),
    assert_contains(CSS, 'border-radius', 'Uses border-radius'),
    assert_contains(CSS, 'transition', 'Has transitions').

test_react_hooks_generation :-
    format('~n--- React Hooks Generation ---~n'),

    generate_api_hooks(python, Code),
    atom_length(Code, Len),
    assert_greater(Len, 800, 'Hooks code length > 800'),

    assert_contains(Code, 'useApiCall', 'Has useApiCall hook'),
    assert_contains(Code, 'useCallback', 'Uses useCallback'),
    assert_contains(Code, 'ApiResult', 'Has ApiResult type'),
    assert_contains(Code, 'loading', 'Tracks loading state'),
    assert_contains(Code, 'error', 'Tracks error state'),
    assert_contains(Code, 'execute', 'Has execute function'),
    assert_contains(Code, 'reset', 'Has reset function').

%% ============================================
%% Full Pipeline Tests
%% ============================================

test_pipeline_application_queries :-
    format('~n--- Full Pipeline Application Queries ---~n'),

    all_applications(AllApps),
    length(AllApps, AppCount),
    assert_greater(AppCount, 0, 'Has at least one application'),

    assert_true(application(python_bridge_demo, _), 'python_bridge_demo exists').

test_pipeline_file_generation :-
    format('~n--- Full Pipeline File Generation ---~n'),

    generate_application(python_bridge_demo, Files),
    length(Files, FileCount),
    assert_greater(FileCount, 15, 'Generates more than 15 files'),

    % Check key files exist
    assert_true(member(file('package.json', _), Files), 'Has package.json'),
    assert_true(member(file('tsconfig.json', _), Files), 'Has tsconfig.json'),
    assert_true(member(file('src/server/index.ts', _), Files), 'Has server index'),
    assert_true(member(file('src/server/router.ts', _), Files), 'Has router'),
    assert_true(member(file('src/server/whitelist.ts', _), Files), 'Has whitelist'),
    assert_true(member(file('src/App.tsx', _), Files), 'Has App.tsx'),
    assert_true(member(file('Dockerfile', _), Files), 'Has Dockerfile'),
    assert_true(member(file('README.md', _), Files), 'Has README').

test_pipeline_package_json :-
    format('~n--- Full Pipeline package.json ---~n'),

    generate_application(python_bridge_demo, Files),
    member(file('package.json', PkgJson), Files),

    assert_contains(PkgJson, '"name":', 'Has name field'),
    assert_contains(PkgJson, '"version":', 'Has version field'),
    assert_contains(PkgJson, '"scripts":', 'Has scripts'),
    assert_contains(PkgJson, '"dependencies":', 'Has dependencies'),
    assert_contains(PkgJson, 'express', 'Depends on express'),
    assert_contains(PkgJson, 'koffi', 'Depends on koffi'),
    assert_contains(PkgJson, 'typescript', 'Has TypeScript').

test_pipeline_dockerfile :-
    format('~n--- Full Pipeline Dockerfile ---~n'),

    generate_application(python_bridge_demo, Files),
    member(file('Dockerfile', Dockerfile), Files),

    assert_contains(Dockerfile, 'FROM node:', 'Based on Node image'),
    assert_contains(Dockerfile, 'python3', 'Installs Python'),
    assert_contains(Dockerfile, 'rpyc', 'Installs rpyc'),
    assert_contains(Dockerfile, 'EXPOSE', 'Exposes port'),
    assert_contains(Dockerfile, 'CMD', 'Has start command').

%% ============================================
%% Main Entry Point
%% ============================================

:- initialization((
    (run_tests -> halt(0) ; halt(1))
), main).
