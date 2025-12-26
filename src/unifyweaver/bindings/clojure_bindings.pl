% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% clojure_bindings.pl - Clojure-specific bindings for UnifyWeaver
%
% Clojure bindings for immutable data, lazy sequences, and functional idioms.

:- encoding(utf8).

:- module(clojure_bindings, [
    init_clojure_bindings/0,
    clj_binding/5,
    clj_binding_require/2,       % clj_binding_require(Pred, Namespace)
    test_clojure_bindings/0
]).

:- use_module('../core/binding_registry').

%% init_clojure_bindings
init_clojure_bindings :-
    register_core_bindings,
    register_collection_bindings,
    register_sequence_bindings,
    register_string_bindings,
    register_threading_bindings.

%% clj_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
clj_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(clojure, Pred, TargetName, Inputs, Outputs, Options).

%% clj_binding_require(?Pred, ?Namespace)
%  Get the require namespace for a Clojure binding.
clj_binding_require(Pred, Namespace) :-
    clj_binding(Pred, _, _, _, Options),
    member(require(Namespace), Options).

% ============================================================================
% DIRECTIVE SUPPORT
% ============================================================================

%% :- clj_binding(Pred, TargetName, Inputs, Outputs, Options)
%  Directive for user-defined Clojure bindings.
:- multifile user:term_expansion/2.

user:term_expansion(
    (:- clj_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(clojure, Pred, TargetName, Inputs, Outputs, Options)))
).

% ============================================================================
% CORE BINDINGS
% ============================================================================

register_core_bindings :-
    % Identity and type
    declare_binding(clojure, identity/2, 'identity',
        [any], [any],
        [pure, deterministic, total]),
    
    declare_binding(clojure, type/2, 'type',
        [any], [class],
        [pure, deterministic, total]),
    
    declare_binding(clojure, 'nil?'/1, 'nil?',
        [any], [boolean],
        [pure, deterministic, total]),
    
    declare_binding(clojure, 'some?'/1, 'some?',
        [any], [boolean],
        [pure, deterministic, total]),
    
    % Comparison
    declare_binding(clojure, '='/2, '=',
        [any, any], [boolean],
        [pure, deterministic, total]),
    
    declare_binding(clojure, 'not='/2, 'not=',
        [any, any], [boolean],
        [pure, deterministic, total]),
    
    % Arithmetic
    declare_binding(clojure, '+'/2, '+',
        [number, number], [number],
        [pure, deterministic, total]),
    
    declare_binding(clojure, '-'/2, '-',
        [number, number], [number],
        [pure, deterministic, total]),
    
    declare_binding(clojure, '*'/2, '*',
        [number, number], [number],
        [pure, deterministic, total]),
    
    declare_binding(clojure, '/'/2, '/',
        [number, number], [number],
        [pure, deterministic, partial]),
    
    declare_binding(clojure, mod/3, 'mod',
        [number, number], [number],
        [pure, deterministic, partial]),
    
    declare_binding(clojure, inc/2, 'inc',
        [number], [number],
        [pure, deterministic, total]),
    
    declare_binding(clojure, dec/2, 'dec',
        [number], [number],
        [pure, deterministic, total]),
    
    % I/O
    declare_binding(clojure, println/1, 'println',
        [any], [],
        [effect(io), deterministic, total]),
    
    declare_binding(clojure, print/1, 'print',
        [any], [],
        [effect(io), deterministic, total]),
    
    declare_binding(clojure, prn/1, 'prn',
        [any], [],
        [effect(io), deterministic, total]).

% ============================================================================
% COLLECTION BINDINGS
% ============================================================================

register_collection_bindings :-
    % List
    declare_binding(clojure, list/2, 'list',
        [vararg], [list],
        [pure, deterministic, total]),
    
    declare_binding(clojure, cons/3, 'cons',
        [any, seq], [seq],
        [pure, deterministic, total]),
    
    declare_binding(clojure, first/2, 'first',
        [seq], [any],
        [pure, deterministic, total]),
    
    declare_binding(clojure, rest/2, 'rest',
        [seq], [seq],
        [pure, deterministic, total]),
    
    declare_binding(clojure, next/2, 'next',
        [seq], [seq],
        [pure, deterministic, total]),
    
    % Vector
    declare_binding(clojure, vector/2, 'vector',
        [vararg], [vector],
        [pure, deterministic, total]),
    
    declare_binding(clojure, vec/2, 'vec',
        [coll], [vector],
        [pure, deterministic, total]),
    
    declare_binding(clojure, conj/3, 'conj',
        [coll, any], [coll],
        [pure, deterministic, total]),
    
    declare_binding(clojure, nth/3, 'nth',
        [coll, int], [any],
        [pure, deterministic, partial]),
    
    declare_binding(clojure, get/3, 'get',
        [coll, key], [any],
        [pure, deterministic, total]),
    
    % Map
    declare_binding(clojure, hash_map/2, 'hash-map',
        [vararg], [map],
        [pure, deterministic, total]),
    
    declare_binding(clojure, assoc/4, 'assoc',
        [map, key, val], [map],
        [pure, deterministic, total]),
    
    declare_binding(clojure, dissoc/3, 'dissoc',
        [map, key], [map],
        [pure, deterministic, total]),
    
    declare_binding(clojure, merge/3, 'merge',
        [map, map], [map],
        [pure, deterministic, total]),
    
    declare_binding(clojure, keys/2, 'keys',
        [map], [seq],
        [pure, deterministic, total]),
    
    declare_binding(clojure, vals/2, 'vals',
        [map], [seq],
        [pure, deterministic, total]),
    
    % Set
    declare_binding(clojure, set/2, 'set',
        [coll], [set],
        [pure, deterministic, total]),
    
    declare_binding(clojure, 'contains?'/2, 'contains?',
        [coll, key], [boolean],
        [pure, deterministic, total]),
    
    % Count
    declare_binding(clojure, count/2, 'count',
        [coll], [int],
        [pure, deterministic, total]),
    
    declare_binding(clojure, 'empty?'/1, 'empty?',
        [coll], [boolean],
        [pure, deterministic, total]).

% ============================================================================
% SEQUENCE BINDINGS (lazy)
% ============================================================================

register_sequence_bindings :-
    declare_binding(clojure, map/3, 'map',
        [fn, seq], [seq],
        [pure, deterministic, total]),
    
    declare_binding(clojure, filter/3, 'filter',
        [pred, seq], [seq],
        [pure, deterministic, total]),
    
    declare_binding(clojure, remove/3, 'remove',
        [pred, seq], [seq],
        [pure, deterministic, total]),
    
    declare_binding(clojure, reduce/3, 'reduce',
        [fn, seq], [any],
        [pure, deterministic, total]),
    
    declare_binding(clojure, reduce/4, 'reduce',
        [fn, init, seq], [any],
        [pure, deterministic, total]),
    
    declare_binding(clojure, mapcat/3, 'mapcat',
        [fn, seq], [seq],
        [pure, deterministic, total]),
    
    declare_binding(clojure, take/3, 'take',
        [int, seq], [seq],
        [pure, deterministic, total]),
    
    declare_binding(clojure, drop/3, 'drop',
        [int, seq], [seq],
        [pure, deterministic, total]),
    
    declare_binding(clojure, take_while/3, 'take-while',
        [pred, seq], [seq],
        [pure, deterministic, total]),
    
    declare_binding(clojure, drop_while/3, 'drop-while',
        [pred, seq], [seq],
        [pure, deterministic, total]),
    
    declare_binding(clojure, keep/3, 'keep',
        [fn, seq], [seq],
        [pure, deterministic, total]),
    
    declare_binding(clojure, lazy_seq/2, 'lazy-seq',
        [body], [seq],
        [pure, deterministic, total]),
    
    declare_binding(clojure, doall/2, 'doall',
        [seq], [seq],
        [effect(state), deterministic, total]),
    
    declare_binding(clojure, doseq/2, 'doseq',
        [bindings, body], [],
        [effect(state), deterministic, total]).

% ============================================================================
% STRING BINDINGS
% ============================================================================

register_string_bindings :-
    declare_binding(clojure, str/2, 'str',
        [vararg], [string],
        [pure, deterministic, total]),
    
    declare_binding(clojure, subs/3, 'subs',
        [string, int], [string],
        [pure, deterministic, partial]),
    
    declare_binding(clojure, subs/4, 'subs',
        [string, int, int], [string],
        [pure, deterministic, partial]),
    
    declare_binding(clojure, clojure_string_split/3, 'clojure.string/split',
        [string, regex], [vector],
        [pure, deterministic, total,
         import('[clojure.string :as str]')]),
    
    declare_binding(clojure, clojure_string_trim/2, 'clojure.string/trim',
        [string], [string],
        [pure, deterministic, total,
         import('[clojure.string :as str]')]),
    
    declare_binding(clojure, clojure_string_lower/2, 'clojure.string/lower-case',
        [string], [string],
        [pure, deterministic, total,
         import('[clojure.string :as str]')]),
    
    declare_binding(clojure, clojure_string_upper/2, 'clojure.string/upper-case',
        [string], [string],
        [pure, deterministic, total,
         import('[clojure.string :as str]')]),
    
    declare_binding(clojure, clojure_string_join/3, 'clojure.string/join',
        [sep, coll], [string],
        [pure, deterministic, total,
         import('[clojure.string :as str]')]).

% ============================================================================
% THREADING MACRO BINDINGS
% ============================================================================

register_threading_bindings :-
    declare_binding(clojure, thread_first/2, '->',
        [expr, forms], [any],
        [pure, deterministic, total, pattern(macro)]),
    
    declare_binding(clojure, thread_last/2, '->>',
        [expr, forms], [any],
        [pure, deterministic, total, pattern(macro)]),
    
    declare_binding(clojure, as_thread/3, 'as->',
        [expr, name, forms], [any],
        [pure, deterministic, total, pattern(macro)]),
    
    declare_binding(clojure, some_thread/2, 'some->',
        [expr, forms], [any],
        [pure, deterministic, total, pattern(macro)]),
    
    declare_binding(clojure, some_thread_last/2, 'some->>',
        [expr, forms], [any],
        [pure, deterministic, total, pattern(macro)]),
    
    declare_binding(clojure, cond_thread/2, 'cond->',
        [expr, clauses], [any],
        [pure, deterministic, total, pattern(macro)]).

% ============================================================================
% TESTS
% ============================================================================

test_clojure_bindings :-
    format('~n=== Clojure Bindings Tests ===~n~n'),

    format('[Test 1] Initializing Clojure bindings~n'),
    init_clojure_bindings,
    format('  [PASS] Clojure bindings initialized~n'),

    format('~n[Test 2] Checking sequence operations~n'),
    (   clj_binding(mapcat/3, 'mapcat', _, _, _)
    ->  format('  [PASS] mapcat binding exists~n')
    ;   format('  [FAIL] mapcat binding missing~n')
    ),

    format('~n[Test 3] Checking threading macros~n'),
    (   clj_binding(thread_last/2, '->>', _, _, _)
    ->  format('  [PASS] ->> binding exists~n')
    ;   format('  [FAIL] ->> binding missing~n')
    ),

    format('~n[Test 4] Counting total bindings~n'),
    findall(P, clj_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('  [INFO] Total Clojure bindings: ~w~n', [Count]),

    format('~n=== Clojure Bindings Tests Complete ===~n').
