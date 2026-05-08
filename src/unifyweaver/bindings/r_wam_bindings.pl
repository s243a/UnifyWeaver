:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% r_wam_bindings.pl - R bindings for hybrid WAM transpilation (scaffold)
%
% Maps Prolog builtins used by the WAM runtime to R equivalents,
% so future native lowering (wam_r_lowered_emitter.pl) can emit
% idiomatic R instead of delegating every call to the interpreter.
%
% Design intent (mirrors rust_wam_bindings.pl / python_wam_bindings.pl):
%   - Pure builtins (arithmetic, comparison, type checks) should
%     inline to native R expressions.
%   - Stateful or partial builtins (assoc, =.., functor) need a
%     small R-side helper library; these bindings reference helper
%     names defined in templates/targets/r_wam/runtime.R.mustache.
%   - Hot list operations are vectorized via base R (length, c, [[).
%
% Phase 1 (this file): a small starter set covering arithmetic,
% comparisons, list length, and basic type checks. Extend as the
% lowered emitter matures.
%
% See: docs/proposals/BINDING_PREDICATE_PROPOSAL.md
%      docs/design/WAM_HASKELL_LOWERED_SPECIFICATION.md (analogous design)

:- module(r_wam_bindings, [
    init_r_wam_bindings/0,
    r_wam_binding/5,             % +PrologPred, -RExpr, -ArgTypes, -RetType, -Props
    r_wam_type_map/2             % +PrologType, -RType
]).

:- use_module('../core/binding_registry').

% ============================================================================
% TYPE MAPPING
% ============================================================================
% R is dynamically typed; these are documentary tags for downstream
% codegen rather than enforced runtime types.

r_wam_type_map(value,        'wam_value').        % tagged list
r_wam_type_map(atom,         'wam_atom').         % list(tag="atom",  id=...)
r_wam_type_map(integer,      'integer').
r_wam_type_map(float,        'numeric').
r_wam_type_map(number,       'numeric').
r_wam_type_map(bool,         'logical').
r_wam_type_map(string,       'character').
r_wam_type_map(list,         'list').
r_wam_type_map(usize,        'integer').
r_wam_type_map(assoc,        'environment').       % named env used as hash
r_wam_type_map(instruction,  'wam_instruction').
r_wam_type_map(choice_point, 'wam_choice_point').
r_wam_type_map(stack_entry,  'wam_stack_entry').
r_wam_type_map(trail_entry,  'wam_trail_entry').

% ============================================================================
% BINDING DECLARATIONS
% ============================================================================
% Format: r_wam_binding(Pred, RExpr, ArgTypes, RetType, Props).
%   ~name placeholders in RExpr are substituted with caller arguments.

% --- Arithmetic ---

r_wam_binding('+'/2,
    '(~a + ~b)',
    [a-number, b-number], [number],
    [pure, deterministic, total, pattern(infix)]).

r_wam_binding('-'/2,
    '(~a - ~b)',
    [a-number, b-number], [number],
    [pure, deterministic, total, pattern(infix)]).

r_wam_binding('*'/2,
    '(~a * ~b)',
    [a-number, b-number], [number],
    [pure, deterministic, total, pattern(infix)]).

r_wam_binding('/'/2,
    '(~a / ~b)',
    [a-number, b-number], [number],
    [pure, deterministic, partial, pattern(infix)]).

r_wam_binding(mod/2,
    '(~a %% ~b)',
    [a-integer, b-integer], [integer],
    [pure, deterministic, partial, pattern(infix)]).

r_wam_binding('//'/2,
    '(~a %/% ~b)',
    [a-integer, b-integer], [integer],
    [pure, deterministic, partial, pattern(infix)]).

% --- Comparison ---

r_wam_binding('=:='/2,
    '(~a == ~b)',
    [a-number, b-number], [bool],
    [pure, deterministic, total, pattern(infix)]).

r_wam_binding('=\\='/2,
    '(~a != ~b)',
    [a-number, b-number], [bool],
    [pure, deterministic, total, pattern(infix)]).

r_wam_binding('<'/2,
    '(~a < ~b)',
    [a-number, b-number], [bool],
    [pure, deterministic, total, pattern(infix)]).

r_wam_binding('>'/2,
    '(~a > ~b)',
    [a-number, b-number], [bool],
    [pure, deterministic, total, pattern(infix)]).

r_wam_binding('=<'/2,
    '(~a <= ~b)',
    [a-number, b-number], [bool],
    [pure, deterministic, total, pattern(infix)]).

r_wam_binding('>='/2,
    '(~a >= ~b)',
    [a-number, b-number], [bool],
    [pure, deterministic, total, pattern(infix)]).

% --- Type checks ---
% These delegate to runtime helpers defined in runtime.R.mustache:
%   wam_is_atom(v), wam_is_integer(v), wam_is_number(v), wam_is_var(v).

r_wam_binding(atom/1,
    'wam_is_atom(~val)',
    [val-value], [bool],
    [pure, deterministic, total, pattern(call)]).

r_wam_binding(integer/1,
    'wam_is_integer(~val)',
    [val-value], [bool],
    [pure, deterministic, total, pattern(call)]).

r_wam_binding(float/1,
    'wam_is_float(~val)',
    [val-value], [bool],
    [pure, deterministic, total, pattern(call)]).

r_wam_binding(number/1,
    'wam_is_number(~val)',
    [val-value], [bool],
    [pure, deterministic, total, pattern(call)]).

r_wam_binding(var/1,
    'wam_is_var(~val)',
    [val-value], [bool],
    [pure, deterministic, total, pattern(call)]).

r_wam_binding(nonvar/1,
    '!wam_is_var(~val)',
    [val-value], [bool],
    [pure, deterministic, total, pattern(prefix_not)]).

r_wam_binding(compound/1,
    'wam_is_compound(~val)',
    [val-value], [bool],
    [pure, deterministic, total, pattern(call)]).

% --- List operations ---

r_wam_binding(length/2,
    'length(~list)',
    [list-list], [integer],
    [pure, deterministic, total, pattern(call)]).

r_wam_binding(append/3,
    'c(~list1, ~list2)',
    [list1-list, list2-list], [list],
    [pure, deterministic, total, pattern(call)]).

r_wam_binding(member/2,
    'wam_member(~elem, ~list)',
    [elem-value, list-list], [bool],
    [pure, deterministic, partial, pattern(call)]).

r_wam_binding(nth0/3,
    '~list[[~idx + 1L]]',
    [idx-usize, list-list], [value],
    [pure, deterministic, partial, pattern(index_0_based)]).

r_wam_binding(nth1/3,
    '~list[[~idx]]',
    [idx-usize, list-list], [value],
    [pure, deterministic, partial, pattern(index_1_based)]).

r_wam_binding(is_list/1,
    'wam_is_proper_list(state, ~val, table)',
    [val-value], [bool],
    [pure, deterministic, total, stateful, pattern(call)]).

r_wam_binding(ground/1,
    'wam_is_ground(state, ~val)',
    [val-value], [bool],
    [pure, deterministic, total, stateful, pattern(call)]).

% --- I/O (impure) ---

r_wam_binding(write/1,
    'cat(wam_term_to_string(~term))',
    [term-value], [],
    [impure, effect(io), pattern(call)]).

r_wam_binding(writeln/1,
    'cat(wam_term_to_string(~term), "\\n", sep="")',
    [term-value], [],
    [impure, effect(io), pattern(call)]).

r_wam_binding(nl/0,
    'cat("\\n")',
    [], [],
    [impure, effect(io), pattern(call)]).

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_r_wam_bindings
%  Registers all WAM-specific R bindings with the global binding registry.
init_r_wam_bindings :-
    forall(
        r_wam_binding(Pred, RExpr, ArgTypes, RetType, Props),
        (   \+ binding(r_wam, Pred, _, _, _, _)
        ->  declare_binding(r_wam, Pred, RExpr, ArgTypes, RetType, Props)
        ;   true
        )
    ).
