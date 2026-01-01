% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Mathematical Expression Translation
%
% This module translates Prolog mathematical expressions to target languages
% (JavaScript, Python/NumPy) for use in visualization generators.
%
% Usage:
%   % Define expressions as Prolog terms
%   Expr = sin(x) * cos(y),
%   expr_to_js(Expr, JSCode),      % -> "Math.sin(x) * Math.cos(y)"
%   expr_to_numpy(Expr, PyCode).   % -> "np.sin(x) * np.cos(y)"
%
% Supported operations:
%   - Arithmetic: +, -, *, /, ^, mod
%   - Trigonometric: sin, cos, tan, asin, acos, atan, atan2
%   - Hyperbolic: sinh, cosh, tanh
%   - Exponential: exp, log, log10, log2
%   - Power/roots: sqrt, cbrt, pow
%   - Rounding: floor, ceil, round, abs
%   - Constants: pi, e
%   - Comparison: min, max
%   - Special: sign, clamp

:- module(math_expr, [
    % Expression translation
    expr_to_js/2,                   % expr_to_js(+Expr, -JSCode)
    expr_to_numpy/2,                % expr_to_numpy(+Expr, -NumpyCode)
    expr_to_prolog/2,               % expr_to_prolog(+Expr, -PrologCode)

    % Expression with variable binding
    expr_to_js/3,                   % expr_to_js(+Expr, +VarPrefix, -JSCode)
    expr_to_numpy/3,                % expr_to_numpy(+Expr, +VarPrefix, -NumpyCode)

    % Validation
    validate_expr/1,                % validate_expr(+Expr)
    expr_variables/2,               % expr_variables(+Expr, -Variables)

    % Testing
    test_math_expr/0
]).

:- use_module(library(lists)).

% ============================================================================
% EXPRESSION TO JAVASCRIPT
% ============================================================================

%% expr_to_js(+Expr, -JSCode)
%  Convert a Prolog expression to JavaScript code.
expr_to_js(Expr, JSCode) :-
    expr_to_js(Expr, '', JSCode).

%% expr_to_js(+Expr, +VarPrefix, -JSCode)
%  Convert with optional variable prefix.
expr_to_js(Expr, Prefix, JSCode) :-
    expr_to_js_inner(Expr, Prefix, JSCode).

% Variables
expr_to_js_inner(x, Prefix, Code) :-
    (Prefix = '' -> Code = 'x' ; format(atom(Code), '~wx', [Prefix])).
expr_to_js_inner(y, Prefix, Code) :-
    (Prefix = '' -> Code = 'y' ; format(atom(Code), '~wy', [Prefix])).
expr_to_js_inner(z, Prefix, Code) :-
    (Prefix = '' -> Code = 'z' ; format(atom(Code), '~wz', [Prefix])).
expr_to_js_inner(t, Prefix, Code) :-
    (Prefix = '' -> Code = 't' ; format(atom(Code), '~wt', [Prefix])).
expr_to_js_inner(r, Prefix, Code) :-
    (Prefix = '' -> Code = 'r' ; format(atom(Code), '~wr', [Prefix])).

% Constants
expr_to_js_inner(pi, _, 'Math.PI').
expr_to_js_inner(e, _, 'Math.E').

% Numbers
expr_to_js_inner(N, _, Code) :-
    number(N),
    format(atom(Code), '~w', [N]).

% Arithmetic operators
expr_to_js_inner(A + B, Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    expr_to_js_inner(B, Prefix, BCode),
    format(atom(Code), '(~w + ~w)', [ACode, BCode]).

expr_to_js_inner(A - B, Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    expr_to_js_inner(B, Prefix, BCode),
    format(atom(Code), '(~w - ~w)', [ACode, BCode]).

expr_to_js_inner(A * B, Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    expr_to_js_inner(B, Prefix, BCode),
    format(atom(Code), '(~w * ~w)', [ACode, BCode]).

expr_to_js_inner(A / B, Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    expr_to_js_inner(B, Prefix, BCode),
    format(atom(Code), '(~w / ~w)', [ACode, BCode]).

expr_to_js_inner(A ^ B, Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    expr_to_js_inner(B, Prefix, BCode),
    format(atom(Code), 'Math.pow(~w, ~w)', [ACode, BCode]).

expr_to_js_inner(A ** B, Prefix, Code) :-
    expr_to_js_inner(A ^ B, Prefix, Code).

expr_to_js_inner(mod(A, B), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    expr_to_js_inner(B, Prefix, BCode),
    format(atom(Code), '(~w % ~w)', [ACode, BCode]).

% Unary minus
expr_to_js_inner(-A, Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), '(-~w)', [ACode]).

% Trigonometric functions
expr_to_js_inner(sin(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.sin(~w)', [ACode]).

expr_to_js_inner(cos(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.cos(~w)', [ACode]).

expr_to_js_inner(tan(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.tan(~w)', [ACode]).

expr_to_js_inner(asin(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.asin(~w)', [ACode]).

expr_to_js_inner(acos(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.acos(~w)', [ACode]).

expr_to_js_inner(atan(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.atan(~w)', [ACode]).

expr_to_js_inner(atan2(A, B), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    expr_to_js_inner(B, Prefix, BCode),
    format(atom(Code), 'Math.atan2(~w, ~w)', [ACode, BCode]).

% Hyperbolic functions
expr_to_js_inner(sinh(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.sinh(~w)', [ACode]).

expr_to_js_inner(cosh(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.cosh(~w)', [ACode]).

expr_to_js_inner(tanh(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.tanh(~w)', [ACode]).

% Exponential and logarithmic
expr_to_js_inner(exp(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.exp(~w)', [ACode]).

expr_to_js_inner(log(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.log(~w)', [ACode]).

expr_to_js_inner(ln(A), Prefix, Code) :-
    expr_to_js_inner(log(A), Prefix, Code).

expr_to_js_inner(log10(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.log10(~w)', [ACode]).

expr_to_js_inner(log2(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.log2(~w)', [ACode]).

% Power and roots
expr_to_js_inner(sqrt(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.sqrt(~w)', [ACode]).

expr_to_js_inner(cbrt(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.cbrt(~w)', [ACode]).

expr_to_js_inner(pow(A, B), Prefix, Code) :-
    expr_to_js_inner(A ^ B, Prefix, Code).

% Rounding and absolute
expr_to_js_inner(abs(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.abs(~w)', [ACode]).

expr_to_js_inner(floor(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.floor(~w)', [ACode]).

expr_to_js_inner(ceil(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.ceil(~w)', [ACode]).

expr_to_js_inner(round(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.round(~w)', [ACode]).

expr_to_js_inner(sign(A), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    format(atom(Code), 'Math.sign(~w)', [ACode]).

% Min/max
expr_to_js_inner(min(A, B), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    expr_to_js_inner(B, Prefix, BCode),
    format(atom(Code), 'Math.min(~w, ~w)', [ACode, BCode]).

expr_to_js_inner(max(A, B), Prefix, Code) :-
    expr_to_js_inner(A, Prefix, ACode),
    expr_to_js_inner(B, Prefix, BCode),
    format(atom(Code), 'Math.max(~w, ~w)', [ACode, BCode]).

% Clamp
expr_to_js_inner(clamp(Val, Min, Max), Prefix, Code) :-
    expr_to_js_inner(Val, Prefix, ValCode),
    expr_to_js_inner(Min, Prefix, MinCode),
    expr_to_js_inner(Max, Prefix, MaxCode),
    format(atom(Code), 'Math.min(Math.max(~w, ~w), ~w)', [ValCode, MinCode, MaxCode]).

% ============================================================================
% EXPRESSION TO NUMPY (Python)
% ============================================================================

%% expr_to_numpy(+Expr, -NumpyCode)
%  Convert a Prolog expression to NumPy code.
expr_to_numpy(Expr, NumpyCode) :-
    expr_to_numpy(Expr, '', NumpyCode).

%% expr_to_numpy(+Expr, +VarPrefix, -NumpyCode)
%  Convert with optional variable prefix.
expr_to_numpy(Expr, Prefix, NumpyCode) :-
    expr_to_numpy_inner(Expr, Prefix, NumpyCode).

% Variables (capitalized for numpy arrays)
expr_to_numpy_inner(x, Prefix, Code) :-
    (Prefix = '' -> Code = 'X' ; format(atom(Code), '~wX', [Prefix])).
expr_to_numpy_inner(y, Prefix, Code) :-
    (Prefix = '' -> Code = 'Y' ; format(atom(Code), '~wY', [Prefix])).
expr_to_numpy_inner(z, Prefix, Code) :-
    (Prefix = '' -> Code = 'Z' ; format(atom(Code), '~wZ', [Prefix])).
expr_to_numpy_inner(t, Prefix, Code) :-
    (Prefix = '' -> Code = 'T' ; format(atom(Code), '~wT', [Prefix])).
expr_to_numpy_inner(r, Prefix, Code) :-
    (Prefix = '' -> Code = 'R' ; format(atom(Code), '~wR', [Prefix])).

% Constants
expr_to_numpy_inner(pi, _, 'np.pi').
expr_to_numpy_inner(e, _, 'np.e').

% Numbers
expr_to_numpy_inner(N, _, Code) :-
    number(N),
    format(atom(Code), '~w', [N]).

% Arithmetic operators
expr_to_numpy_inner(A + B, Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    expr_to_numpy_inner(B, Prefix, BCode),
    format(atom(Code), '(~w + ~w)', [ACode, BCode]).

expr_to_numpy_inner(A - B, Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    expr_to_numpy_inner(B, Prefix, BCode),
    format(atom(Code), '(~w - ~w)', [ACode, BCode]).

expr_to_numpy_inner(A * B, Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    expr_to_numpy_inner(B, Prefix, BCode),
    format(atom(Code), '(~w * ~w)', [ACode, BCode]).

expr_to_numpy_inner(A / B, Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    expr_to_numpy_inner(B, Prefix, BCode),
    format(atom(Code), '(~w / ~w)', [ACode, BCode]).

expr_to_numpy_inner(A ^ B, Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    expr_to_numpy_inner(B, Prefix, BCode),
    format(atom(Code), 'np.power(~w, ~w)', [ACode, BCode]).

expr_to_numpy_inner(A ** B, Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    expr_to_numpy_inner(B, Prefix, BCode),
    format(atom(Code), '(~w ** ~w)', [ACode, BCode]).

expr_to_numpy_inner(mod(A, B), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    expr_to_numpy_inner(B, Prefix, BCode),
    format(atom(Code), 'np.mod(~w, ~w)', [ACode, BCode]).

% Unary minus
expr_to_numpy_inner(-A, Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), '(-~w)', [ACode]).

% Trigonometric functions
expr_to_numpy_inner(sin(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.sin(~w)', [ACode]).

expr_to_numpy_inner(cos(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.cos(~w)', [ACode]).

expr_to_numpy_inner(tan(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.tan(~w)', [ACode]).

expr_to_numpy_inner(asin(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.arcsin(~w)', [ACode]).

expr_to_numpy_inner(acos(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.arccos(~w)', [ACode]).

expr_to_numpy_inner(atan(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.arctan(~w)', [ACode]).

expr_to_numpy_inner(atan2(A, B), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    expr_to_numpy_inner(B, Prefix, BCode),
    format(atom(Code), 'np.arctan2(~w, ~w)', [ACode, BCode]).

% Hyperbolic functions
expr_to_numpy_inner(sinh(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.sinh(~w)', [ACode]).

expr_to_numpy_inner(cosh(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.cosh(~w)', [ACode]).

expr_to_numpy_inner(tanh(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.tanh(~w)', [ACode]).

% Exponential and logarithmic
expr_to_numpy_inner(exp(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.exp(~w)', [ACode]).

expr_to_numpy_inner(log(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.log(~w)', [ACode]).

expr_to_numpy_inner(ln(A), Prefix, Code) :-
    expr_to_numpy_inner(log(A), Prefix, Code).

expr_to_numpy_inner(log10(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.log10(~w)', [ACode]).

expr_to_numpy_inner(log2(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.log2(~w)', [ACode]).

% Power and roots
expr_to_numpy_inner(sqrt(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.sqrt(~w)', [ACode]).

expr_to_numpy_inner(cbrt(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.cbrt(~w)', [ACode]).

expr_to_numpy_inner(pow(A, B), Prefix, Code) :-
    expr_to_numpy_inner(A ^ B, Prefix, Code).

% Rounding and absolute
expr_to_numpy_inner(abs(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.abs(~w)', [ACode]).

expr_to_numpy_inner(floor(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.floor(~w)', [ACode]).

expr_to_numpy_inner(ceil(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.ceil(~w)', [ACode]).

expr_to_numpy_inner(round(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.round(~w)', [ACode]).

expr_to_numpy_inner(sign(A), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    format(atom(Code), 'np.sign(~w)', [ACode]).

% Min/max
expr_to_numpy_inner(min(A, B), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    expr_to_numpy_inner(B, Prefix, BCode),
    format(atom(Code), 'np.minimum(~w, ~w)', [ACode, BCode]).

expr_to_numpy_inner(max(A, B), Prefix, Code) :-
    expr_to_numpy_inner(A, Prefix, ACode),
    expr_to_numpy_inner(B, Prefix, BCode),
    format(atom(Code), 'np.maximum(~w, ~w)', [ACode, BCode]).

% Clamp
expr_to_numpy_inner(clamp(Val, Min, Max), Prefix, Code) :-
    expr_to_numpy_inner(Val, Prefix, ValCode),
    expr_to_numpy_inner(Min, Prefix, MinCode),
    expr_to_numpy_inner(Max, Prefix, MaxCode),
    format(atom(Code), 'np.clip(~w, ~w, ~w)', [ValCode, MinCode, MaxCode]).

% ============================================================================
% EXPRESSION TO PROLOG (for evaluation)
% ============================================================================

%% expr_to_prolog(+Expr, -PrologCode)
%  Convert expression to evaluable Prolog code string.
expr_to_prolog(Expr, PrologCode) :-
    format(atom(PrologCode), '~w', [Expr]).

% ============================================================================
% VALIDATION AND UTILITIES
% ============================================================================

%% validate_expr(+Expr)
%  Check if an expression is valid.
validate_expr(Expr) :-
    (expr_to_js(Expr, _) -> true ; fail).

%% expr_variables(+Expr, -Variables)
%  Extract all variables from an expression.
expr_variables(Expr, Variables) :-
    expr_vars(Expr, VarList),
    list_to_set(VarList, Variables).

expr_vars(x, [x]) :- !.
expr_vars(y, [y]) :- !.
expr_vars(z, [z]) :- !.
expr_vars(t, [t]) :- !.
expr_vars(r, [r]) :- !.
expr_vars(pi, []) :- !.
expr_vars(e, []) :- !.
expr_vars(N, []) :- number(N), !.
expr_vars(A + B, Vars) :- !, expr_vars(A, VA), expr_vars(B, VB), append(VA, VB, Vars).
expr_vars(A - B, Vars) :- !, expr_vars(A, VA), expr_vars(B, VB), append(VA, VB, Vars).
expr_vars(A * B, Vars) :- !, expr_vars(A, VA), expr_vars(B, VB), append(VA, VB, Vars).
expr_vars(A / B, Vars) :- !, expr_vars(A, VA), expr_vars(B, VB), append(VA, VB, Vars).
expr_vars(A ^ B, Vars) :- !, expr_vars(A, VA), expr_vars(B, VB), append(VA, VB, Vars).
expr_vars(A ** B, Vars) :- !, expr_vars(A, VA), expr_vars(B, VB), append(VA, VB, Vars).
expr_vars(-A, Vars) :- !, expr_vars(A, Vars).
expr_vars(Func, Vars) :-
    Func =.. [_|Args],
    findall(V, (member(Arg, Args), expr_vars(Arg, V)), VarLists),
    append(VarLists, Vars).

% ============================================================================
% TESTS
% ============================================================================

test_math_expr :-
    format('Testing math_expr module...~n~n'),

    % Test basic arithmetic to JS
    format('Test 1: Basic arithmetic to JS~n'),
    expr_to_js(x + y, JS1),
    (JS1 = '(x + y)'
    ->  format('  PASS: x + y -> ~w~n', [JS1])
    ;   format('  FAIL: Expected (x + y), got ~w~n', [JS1])
    ),

    % Test trig functions to JS
    format('~nTest 2: Trig functions to JS~n'),
    expr_to_js(sin(x) * cos(y), JS2),
    (sub_atom(JS2, _, _, _, 'Math.sin')
    ->  format('  PASS: sin(x) * cos(y) -> ~w~n', [JS2])
    ;   format('  FAIL: ~w~n', [JS2])
    ),

    % Test complex expression to JS
    format('~nTest 3: Complex expression to JS~n'),
    expr_to_js(exp(-(x^2 + y^2) / 2), JS3),
    (sub_atom(JS3, _, _, _, 'Math.exp')
    ->  format('  PASS: Gaussian -> ~w~n', [JS3])
    ;   format('  FAIL: ~w~n', [JS3])
    ),

    % Test basic to NumPy
    format('~nTest 4: Basic to NumPy~n'),
    expr_to_numpy(x + y, Py1),
    (Py1 = '(X + Y)'
    ->  format('  PASS: x + y -> ~w~n', [Py1])
    ;   format('  FAIL: Expected (X + Y), got ~w~n', [Py1])
    ),

    % Test trig to NumPy
    format('~nTest 5: Trig to NumPy~n'),
    expr_to_numpy(sin(x) * cos(y), Py2),
    (sub_atom(Py2, _, _, _, 'np.sin')
    ->  format('  PASS: sin(x) * cos(y) -> ~w~n', [Py2])
    ;   format('  FAIL: ~w~n', [Py2])
    ),

    % Test constants
    format('~nTest 6: Constants~n'),
    expr_to_js(2 * pi, JS4),
    expr_to_numpy(2 * pi, Py4),
    (sub_atom(JS4, _, _, _, 'Math.PI'), sub_atom(Py4, _, _, _, 'np.pi')
    ->  format('  PASS: pi in JS=~w, NumPy=~w~n', [JS4, Py4])
    ;   format('  FAIL: JS=~w, NumPy=~w~n', [JS4, Py4])
    ),

    % Test variable extraction
    format('~nTest 7: Variable extraction~n'),
    expr_variables(sin(x) * cos(y) + z, Vars),
    (msort(Vars, [x, y, z])
    ->  format('  PASS: Variables = ~w~n', [Vars])
    ;   format('  FAIL: Variables = ~w~n', [Vars])
    ),

    % Test special functions
    format('~nTest 8: Special functions~n'),
    expr_to_js(sqrt(x^2 + y^2), JS5),
    expr_to_numpy(sqrt(x^2 + y^2), Py5),
    (sub_atom(JS5, _, _, _, 'Math.sqrt'), sub_atom(Py5, _, _, _, 'np.sqrt')
    ->  format('  PASS: sqrt in JS=~w, NumPy=~w~n', [JS5, Py5])
    ;   format('  FAIL~n')
    ),

    format('~nAll tests completed.~n').

:- initialization(test_math_expr, main).
