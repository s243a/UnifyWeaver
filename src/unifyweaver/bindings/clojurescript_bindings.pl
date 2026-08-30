% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% clojurescript_bindings.pl - ClojureScript-specific bindings for UnifyWeaver
%
% Maps Prolog builtins to ClojureScript (sci) core functions. The core / seq /
% collection / string / threading surface is identical between JVM Clojure and
% ClojureScript, so this mirrors clojure_bindings.pl but registers under the
% `clojurescript` target key (so the binding registry can route CLJS compiles).
%
% These bindings are the pure-Clojure(Script) core functions that behave the
% same under every runtime variant of the ClojureScript target:
%   - scittle : Scittle/SCI in the browser
%   - nbb     : Node ClojureScript (sci)
%   - bb      : Babashka (sci, Clojure-on-JVM)
% Host-interop differences (js/parseInt vs Integer/parseInt, js/Math.abs vs
% Math/abs) are NOT expressed here -- those are handled by the runtime-variant
% interop rewrite in targets/clojurescript_target.pl, not by name bindings.

:- module(clojurescript_bindings, [
    init_clojurescript_bindings/0,
    cljs_binding/5,              % cljs_binding(Pred, TargetName, Inputs, Outputs, Options)
    cljs_binding_require/2,      % cljs_binding_require(Pred, Namespace)
    test_clojurescript_bindings/0
]).

:- use_module('../core/binding_registry').

%% init_clojurescript_bindings
init_clojurescript_bindings :-
    register_core_bindings,
    register_collection_bindings,
    register_sequence_bindings,
    register_string_bindings,
    register_threading_bindings,
    register_divergent_bindings.

%% cljs_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
cljs_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(clojurescript, Pred, TargetName, Inputs, Outputs, Options).

%% cljs_binding_require(?Pred, ?Namespace)
cljs_binding_require(Pred, Namespace) :-
    cljs_binding(Pred, _, _, _, Options),
    member(require(Namespace), Options).

% ============================================================================
% DIRECTIVE SUPPORT
% ============================================================================

%% :- cljs_binding(Pred, TargetName, Inputs, Outputs, Options)
:- multifile user:term_expansion/2.

user:term_expansion(
    (:- cljs_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(clojurescript, Pred, TargetName, Inputs, Outputs, Options)))
).

% ============================================================================
% CORE BINDINGS
% ============================================================================

register_core_bindings :-
    declare_binding(clojurescript, identity/2, 'identity',
        [any], [any], [pure, deterministic, total]),
    declare_binding(clojurescript, type/2, 'type',
        [any], [class], [pure, deterministic, total]),
    declare_binding(clojurescript, 'nil?'/1, 'nil?',
        [any], [boolean], [pure, deterministic, total]),
    declare_binding(clojurescript, 'some?'/1, 'some?',
        [any], [boolean], [pure, deterministic, total]),
    % Comparison
    declare_binding(clojurescript, '='/2, '=',
        [any, any], [boolean], [pure, deterministic, total]),
    declare_binding(clojurescript, 'not='/2, 'not=',
        [any, any], [boolean], [pure, deterministic, total]),
    % Arithmetic
    declare_binding(clojurescript, '+'/2, '+',
        [number, number], [number], [pure, deterministic, total]),
    declare_binding(clojurescript, '-'/2, '-',
        [number, number], [number], [pure, deterministic, total]),
    declare_binding(clojurescript, '*'/2, '*',
        [number, number], [number], [pure, deterministic, total]),
    declare_binding(clojurescript, '/'/2, '/',
        [number, number], [number], [pure, deterministic, partial]),
    declare_binding(clojurescript, mod/3, 'mod',
        [number, number], [number], [pure, deterministic, partial]),
    declare_binding(clojurescript, inc/2, 'inc',
        [number], [number], [pure, deterministic, total]),
    declare_binding(clojurescript, dec/2, 'dec',
        [number], [number], [pure, deterministic, total]),
    % I/O
    declare_binding(clojurescript, println/1, 'println',
        [any], [], [effect(io), deterministic, total]),
    declare_binding(clojurescript, print/1, 'print',
        [any], [], [effect(io), deterministic, total]),
    declare_binding(clojurescript, prn/1, 'prn',
        [any], [], [effect(io), deterministic, total]).

% ============================================================================
% COLLECTION BINDINGS
% ============================================================================

register_collection_bindings :-
    declare_binding(clojurescript, list/2, 'list',
        [vararg], [list], [pure, deterministic, total]),
    declare_binding(clojurescript, cons/3, 'cons',
        [any, seq], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, first/2, 'first',
        [seq], [any], [pure, deterministic, total]),
    declare_binding(clojurescript, rest/2, 'rest',
        [seq], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, next/2, 'next',
        [seq], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, vector/2, 'vector',
        [vararg], [vector], [pure, deterministic, total]),
    declare_binding(clojurescript, vec/2, 'vec',
        [coll], [vector], [pure, deterministic, total]),
    declare_binding(clojurescript, conj/3, 'conj',
        [coll, any], [coll], [pure, deterministic, total]),
    declare_binding(clojurescript, nth/3, 'nth',
        [coll, int], [any], [pure, deterministic, partial]),
    declare_binding(clojurescript, get/3, 'get',
        [coll, key], [any], [pure, deterministic, total]),
    declare_binding(clojurescript, hash_map/2, 'hash-map',
        [vararg], [map], [pure, deterministic, total]),
    declare_binding(clojurescript, assoc/4, 'assoc',
        [map, key, val], [map], [pure, deterministic, total]),
    declare_binding(clojurescript, dissoc/3, 'dissoc',
        [map, key], [map], [pure, deterministic, total]),
    declare_binding(clojurescript, merge/3, 'merge',
        [map, map], [map], [pure, deterministic, total]),
    declare_binding(clojurescript, keys/2, 'keys',
        [map], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, vals/2, 'vals',
        [map], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, set/2, 'set',
        [coll], [set], [pure, deterministic, total]),
    declare_binding(clojurescript, 'contains?'/2, 'contains?',
        [coll, key], [boolean], [pure, deterministic, total]),
    declare_binding(clojurescript, count/2, 'count',
        [coll], [int], [pure, deterministic, total]),
    declare_binding(clojurescript, 'empty?'/1, 'empty?',
        [coll], [boolean], [pure, deterministic, total]).

% ============================================================================
% SEQUENCE BINDINGS (lazy)
% ============================================================================

register_sequence_bindings :-
    declare_binding(clojurescript, map/3, 'map',
        [fn, seq], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, filter/3, 'filter',
        [pred, seq], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, remove/3, 'remove',
        [pred, seq], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, reduce/3, 'reduce',
        [fn, seq], [any], [pure, deterministic, total]),
    declare_binding(clojurescript, reduce/4, 'reduce',
        [fn, init, seq], [any], [pure, deterministic, total]),
    declare_binding(clojurescript, mapcat/3, 'mapcat',
        [fn, seq], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, take/3, 'take',
        [int, seq], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, drop/3, 'drop',
        [int, seq], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, take_while/3, 'take-while',
        [pred, seq], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, drop_while/3, 'drop-while',
        [pred, seq], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, keep/3, 'keep',
        [fn, seq], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, sort/2, 'sort',
        [coll], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, 'distinct'/2, 'distinct',
        [coll], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, range/2, 'range',
        [int], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, lazy_seq/2, 'lazy-seq',
        [body], [seq], [pure, deterministic, total]),
    declare_binding(clojurescript, doall/2, 'doall',
        [seq], [seq], [effect(state), deterministic, total]),
    declare_binding(clojurescript, doseq/2, 'doseq',
        [bindings, body], [], [effect(state), deterministic, total]).

% ============================================================================
% STRING BINDINGS
% ============================================================================

register_string_bindings :-
    declare_binding(clojurescript, str/2, 'str',
        [vararg], [string], [pure, deterministic, total]),
    declare_binding(clojurescript, subs/3, 'subs',
        [string, int], [string], [pure, deterministic, partial]),
    declare_binding(clojurescript, subs/4, 'subs',
        [string, int, int], [string], [pure, deterministic, partial]),
    declare_binding(clojurescript, clojure_string_split/3, 'clojure.string/split',
        [string, regex], [vector],
        [pure, deterministic, total, import('[clojure.string :as str]')]),
    declare_binding(clojurescript, clojure_string_trim/2, 'clojure.string/trim',
        [string], [string],
        [pure, deterministic, total, import('[clojure.string :as str]')]),
    declare_binding(clojurescript, clojure_string_lower/2, 'clojure.string/lower-case',
        [string], [string],
        [pure, deterministic, total, import('[clojure.string :as str]')]),
    declare_binding(clojurescript, clojure_string_upper/2, 'clojure.string/upper-case',
        [string], [string],
        [pure, deterministic, total, import('[clojure.string :as str]')]),
    declare_binding(clojurescript, clojure_string_join/3, 'clojure.string/join',
        [sep, coll], [string],
        [pure, deterministic, total, import('[clojure.string :as str]')]).

% ============================================================================
% THREADING MACRO BINDINGS
% ============================================================================

register_threading_bindings :-
    declare_binding(clojurescript, thread_first/2, '->',
        [expr, forms], [any], [pure, deterministic, total, pattern(macro)]),
    declare_binding(clojurescript, thread_last/2, '->>',
        [expr, forms], [any], [pure, deterministic, total, pattern(macro)]),
    declare_binding(clojurescript, as_thread/3, 'as->',
        [expr, name, forms], [any], [pure, deterministic, total, pattern(macro)]),
    declare_binding(clojurescript, some_thread/2, 'some->',
        [expr, forms], [any], [pure, deterministic, total, pattern(macro)]),
    declare_binding(clojurescript, some_thread_last/2, 'some->>',
        [expr, forms], [any], [pure, deterministic, total, pattern(macro)]),
    declare_binding(clojurescript, cond_thread/2, 'cond->',
        [expr, clauses], [any], [pure, deterministic, total, pattern(macro)]).

% ============================================================================
% HOST-DIVERGENT BINDINGS
% ============================================================================
%
% Unlike the sections above (which mirror clojure_bindings.pl verbatim -- the
% core/seq/collection/string/threading surface is identical between JVM Clojure
% and ClojureScript), these bindings map a predicate to a ClojureScript name that
% genuinely DIFFERS from the JVM Clojure default, and whose difference is NOT
% expressible by the runtime-variant interop rewrite (which only translates fixed
% host-call tokens such as Integer/parseInt or Math/abs).
%
% They exist to be routed through the `clojurescript` binding key: the CLJS
% compile path resolves predicate names with the preference list
% [clojurescript, clojure] (clojurescript_target:cljs_binding_name_rewrite/2), so
% a `clojurescript` binding here overrides the JVM `clojure` default while the 64
% shared bindings fall back to it.

register_divergent_bindings :-
    % ClojureScript's cljs.core has no `parse-double` (a JVM Clojure 1.11
    % addition backed by java.lang.Double); the idiomatic CLJS equivalent is the
    % JS global parseFloat. The JVM Clojure default for this predicate is the
    % plain function name; the interop rewrite does not touch `parse-double`, so
    % this override can only take effect via the `clojurescript` binding key.
    declare_binding(clojurescript, parse_double/2, 'js/parseFloat',
        [string], [number], [pure, deterministic, partial]).

test_clojurescript_bindings :-
    format('~n=== ClojureScript Bindings Tests ===~n~n'),

    format('[Test 1] Initializing ClojureScript bindings~n'),
    init_clojurescript_bindings,
    format('  [PASS] ClojureScript bindings initialized~n'),

    format('~n[Test 2] Checking sequence operations~n'),
    (   cljs_binding(mapcat/3, 'mapcat', _, _, _)
    ->  format('  [PASS] mapcat binding exists~n')
    ;   format('  [FAIL] mapcat binding missing~n')
    ),

    format('~n[Test 3] Checking threading macros~n'),
    (   cljs_binding(thread_last/2, '->>', _, _, _)
    ->  format('  [PASS] ->> binding exists~n')
    ;   format('  [FAIL] ->> binding missing~n')
    ),

    format('~n[Test 4] Counting total bindings~n'),
    findall(P, cljs_binding(P, _, _, _, _), Preds),
    length(Preds, Count),
    format('  [INFO] Total ClojureScript bindings: ~w~n', [Count]),

    format('~n=== ClojureScript Bindings Tests Complete ===~n').
