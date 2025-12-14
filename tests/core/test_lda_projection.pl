% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Integration tests for LDA projection through component registry

:- encoding(utf8).

% Import only what we need from component_registry to avoid conflicts
:- use_module('../../src/unifyweaver/core/component_registry', [
    component_type/4,
    list_types/2,
    list_components/2,
    declare_component/4,
    retract_component/2,
    component/4
]).

% Load lda_projection module (it auto-registers itself)
:- use_module('../../src/unifyweaver/runtime/lda_projection', [
    type_info/1,
    validate_config/1
]).

%% ============================================
%% Test Helpers
%% ============================================

run_tests :-
    format('~n=== LDA Projection Integration Tests ===~n~n'),
    (   setup_test_matrix(TempFile)
    ->  (   run_all_tests(TempFile)
        ->  cleanup_test(TempFile),
            format('~nAll tests passed!~n')
        ;   cleanup_test(TempFile),
            format('~nTests FAILED~n'),
            halt(1)
        )
    ;   format('Failed to create test matrix~n'),
        halt(1)
    ).

run_all_tests(TempFile) :-
    test_type_info,
    test_type_registration,
    test_config_validation,
    test_component_declaration(TempFile),
    test_projection_invocation(TempFile).

assert_true(Goal, TestName) :-
    (   call(Goal)
    ->  format('  ~c ~w~n', [10003, TestName])  % checkmark
    ;   format('  X ~w FAILED~n', [TestName]),
        fail
    ).

assert_false(Goal, TestName) :-
    (   \+ call(Goal)
    ->  format('  ~c ~w~n', [10003, TestName])  % checkmark
    ;   format('  X ~w FAILED (expected false)~n', [TestName]),
        fail
    ).

%% ============================================
%% Setup: Create Test W Matrix
%% ============================================

setup_test_matrix(TempFile) :-
    format('Setting up test W matrix...~n'),
    % Create a simple 4x4 identity W matrix for testing
    get_time(Now),
    NowInt is round(Now * 1000000),
    format(atom(TempFile), '/tmp/test_W_~d.npy', [NowInt]),
    format(atom(PythonCode),
'import numpy as np
W = np.eye(4)  # Simple identity matrix
np.save("~w", W)
print("OK")
', [TempFile]),
    process_create(path(python3), ['-c', PythonCode],
                  [stdout(pipe(Out)), stderr(null), process(Proc)]),
    read_string(Out, _, Result),
    close(Out),
    process_wait(Proc, Status),
    (   Status = exit(0), sub_string(Result, _, _, _, "OK")
    ->  format('  Test matrix created: ~w~n~n', [TempFile])
    ;   format('  Failed to create test matrix~n'),
        fail
    ).

cleanup_test(TempFile) :-
    (   exists_file(TempFile)
    ->  delete_file(TempFile)
    ;   true
    ).

%% ============================================
%% Test: Type Info
%% ============================================

test_type_info :-
    format('Test: LDA projection type info~n'),
    lda_projection:type_info(Info),
    Info = info(name(Name), version(Version), _, _, _, _, _),
    assert_true(atom(Name), 'type has name'),
    assert_true(atom(Version), 'type has version'),
    format('~n').

%% ============================================
%% Test: Type Registration
%% ============================================

test_type_registration :-
    format('Test: LDA projection type registration~n'),
    % lda_projection registers itself on load
    assert_true(
        component_type(runtime, lda_projection, lda_projection, _),
        'lda_projection type registered in runtime category'
    ),
    list_types(runtime, Types),
    assert_true(member(lda_projection, Types), 'lda_projection in runtime types'),
    format('~n').

%% ============================================
%% Test: Configuration Validation
%% ============================================

test_config_validation :-
    format('Test: LDA projection config validation~n'),

    % Valid config should pass
    assert_true(
        lda_projection:validate_config([model_file('/tmp/test.npy')]),
        'valid config passes'
    ),

    % Config with optional params should pass
    assert_true(
        lda_projection:validate_config([
            model_file('/tmp/test.npy'),
            embedding_dim(384),
            lambda_reg(1.0),
            ridge(0.000001)
        ]),
        'config with optional params passes'
    ),

    % Missing model_file should fail
    assert_false(
        catch(lda_projection:validate_config([]), _, fail),
        'missing model_file fails'
    ),

    format('~n').

%% ============================================
%% Test: Component Declaration
%% ============================================

test_component_declaration(TempFile) :-
    format('Test: LDA projection component declaration~n'),

    % Declare a test component
    declare_component(runtime, test_projection, lda_projection, [
        model_file(TempFile),
        embedding_dim(4),
        initialization(lazy)
    ]),

    assert_true(
        component(runtime, test_projection, lda_projection, _),
        'component declared successfully'
    ),

    list_components(runtime, Names),
    assert_true(member(test_projection, Names), 'test_projection in components list'),

    format('~n').

%% ============================================
%% Test: Projection Invocation
%% ============================================

test_projection_invocation(TempFile) :-
    format('Test: LDA projection invocation~n'),

    % Test single query projection (through direct module call)
    QueryEmb = [1.0, 0.0, 0.0, 0.0],

    % Using the module's invoke_component directly
    Config = [model_file(TempFile), embedding_dim(4)],

    (   catch(
            lda_projection:invoke_component(test_projection, Config,
                                           query(QueryEmb), projected(ProjEmb)),
            E,
            (format('  Invocation error: ~w~n', [E]), fail)
        )
    ->  format('  Projected embedding: ~w~n', [ProjEmb]),
        assert_true(is_list(ProjEmb), 'projection returns list'),
        length(ProjEmb, Len),
        assert_true(Len =:= 4, 'projection has correct dimension')
    ;   format('  X Projection invocation FAILED~n'),
        fail
    ),

    % Test similarity computation
    DocEmb = [1.0, 0.0, 0.0, 0.0],
    (   catch(
            lda_projection:invoke_component(test_projection, Config,
                                           similarity(QueryEmb, DocEmb), score(Score)),
            E2,
            (format('  Similarity error: ~w~n', [E2]), fail)
        )
    ->  format('  Similarity score: ~w~n', [Score]),
        assert_true(number(Score), 'similarity returns number'),
        assert_true(Score > 0.99, 'identical vectors have high similarity')
    ;   format('  X Similarity invocation FAILED~n'),
        fail
    ),

    % Cleanup: remove test component
    retract_component(runtime, test_projection),

    format('~n').

%% ============================================
%% Main
%% ============================================

:- initialization(run_tests, main).
