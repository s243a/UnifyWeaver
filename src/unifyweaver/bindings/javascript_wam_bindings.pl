% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% javascript_wam_bindings.pl - JavaScript bindings for the WAM runtime target.
%
% SCOPE. The wam_javascript target's baseline (WAMJS-1) is the *interpreter*
% emit tier: a Node WAM virtual machine that executes the shared instruction
% stream. In the interpreter tier the builtins are implemented directly in the
% JS runtime (see templates/targets/javascript_wam/runtime.js.mustache,
% Runtime.builtinCall / evalArith), so — like the C++ target, which emits its
% runtime inline and ships no bindings file — the interpreter needs no
% Prolog-side binding declarations to run.
%
% This file therefore serves two purposes:
%   1. It documents, in one place, the Prolog builtin surface the JS runtime
%      currently understands (the interpreter builtin catalogue below), so the
%      contract is visible next to the other backends' *_wam_bindings.pl.
%   2. It reserves the mapping table for the future *lowered* tier (emitting
%      idiomatic JS per predicate instead of interpreting), analogous to
%      rust_wam_bindings.pl. Lowered/FFI tiers are out of scope for the
%      WAMJS-1 spike; the js_wam_binding/5 table is intentionally minimal and
%      init_javascript_wam_bindings/0 is a no-op-safe registration hook.

:- module(javascript_wam_bindings, [
    init_javascript_wam_bindings/0,
    js_wam_binding/5,            % +PrologPred, -JsExpr, -ArgTypes, -RetType, -Props
    js_wam_type_map/2,           % +PrologType, -JsType
    js_wam_interpreter_builtin/2 % +Name, -Arity  (handled by the JS runtime)
]).

:- use_module('../core/binding_registry').

% ============================================================================
% TYPE MAPPING (JS is dynamically typed; tagged term objects carry the tag)
% ============================================================================

js_wam_type_map(value,   'object').   % { tag, ... }
js_wam_type_map(atom,    'object').   % { tag:"atom", id }
js_wam_type_map(integer, 'number').   % { tag:"int", val }
js_wam_type_map(float,   'number').   % { tag:"float", val }
js_wam_type_map(number,  'number').
js_wam_type_map(list,    'object').   % cons cells: { tag:"struct", fid:[|], args:[h,t] }
js_wam_type_map(bool,    'boolean').
js_wam_type_map(string,  'string').

% ============================================================================
% INTERPRETER BUILTIN CATALOGUE
% ------------------------------------------------------------------------
% Builtins the JS runtime evaluates directly (Runtime.builtinCall). These are
% the builtin_call opcodes the compiler may emit that the baseline supports.
% Arithmetic operators (+ - * // / mod rem min max abs ** ^) are evaluated
% inside is/2 and the arithmetic comparisons via evalArith, honouring the
% name/arity functor parse (§2) and integer result typing (§5).
% ============================================================================

js_wam_interpreter_builtin('=', 2).
js_wam_interpreter_builtin('==', 2).
js_wam_interpreter_builtin('\\==', 2).
js_wam_interpreter_builtin(is, 2).
js_wam_interpreter_builtin('=:=', 2).
js_wam_interpreter_builtin('=\\=', 2).
js_wam_interpreter_builtin('>', 2).
js_wam_interpreter_builtin('<', 2).
js_wam_interpreter_builtin('>=', 2).
js_wam_interpreter_builtin('=<', 2).
js_wam_interpreter_builtin(var, 1).
js_wam_interpreter_builtin(nonvar, 1).
js_wam_interpreter_builtin(atom, 1).
js_wam_interpreter_builtin(integer, 1).
js_wam_interpreter_builtin(float, 1).
js_wam_interpreter_builtin(number, 1).
js_wam_interpreter_builtin(compound, 1).
js_wam_interpreter_builtin(true, 0).
js_wam_interpreter_builtin(fail, 0).
js_wam_interpreter_builtin('!', 0).

% ============================================================================
% LOWERED-TIER BINDINGS (reserved; not used by the interpreter baseline)
% ============================================================================

%% js_wam_binding(+Pred, -JsExpr, -ArgTypes, -RetType, -Props)
%  Declares how a Prolog builtin would map to JS for the future lowered tier.
%  '~name' placeholders are argument slots (same convention as
%  rust_wam_binding/5). The interpreter baseline does not consume these.

js_wam_binding(length/2,
    '~list.length',
    [list-list], [integer],
    [pure, pattern(property), note('lowered tier only')]).

js_wam_binding(is_list/1,
    'Runtime.isList(~val)',
    [val-value], [bool],
    [pure, pattern(call), note('lowered tier only')]).

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_javascript_wam_bindings
%  Registers the reserved lowered-tier bindings with the binding registry.
%  Safe to call repeatedly; the interpreter baseline does not require it.
init_javascript_wam_bindings :-
    forall(
        js_wam_binding(Pred, JsExpr, ArgTypes, RetType, Props),
        (   \+ binding(javascript_wam, Pred, _, _, _, _)
        ->  declare_binding(javascript_wam, Pred, JsExpr, ArgTypes, RetType, Props)
        ;   true
        )
    ).
