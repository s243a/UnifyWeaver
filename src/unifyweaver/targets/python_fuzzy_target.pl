/**
 * Python Fuzzy Logic Target
 *
 * Extends the Python code generator with fuzzy logic operations.
 * Generates NumPy-based implementations for:
 * - f_and: Fuzzy AND (product t-norm)
 * - f_or: Fuzzy OR (probabilistic sum)
 * - f_dist_or: Distributed OR
 * - f_union: Non-distributed OR
 * - f_not: Fuzzy NOT
 *
 * Usage in Prolog:
 *   :- use_module(python_fuzzy_target).
 *
 *   % Compile a predicate that uses fuzzy logic
 *   compile_fuzzy_to_python(my_search/2, [], Code).
 */

:- module(python_fuzzy_target, [
    % Main compilation
    compile_fuzzy_to_python/3,
    compile_fuzzy_to_python/2,

    % Goal translation (extends python_target)
    translate_fuzzy_goal/2,

    % Header/helper generation
    fuzzy_header/1,
    fuzzy_imports/1,

    % Initialization
    init_fuzzy_target/0
]).

:- use_module(python_target, [
    compile_predicate_to_python/3,
    init_python_target/0,
    var_to_python/2
]).

% =============================================================================
% Initialization
% =============================================================================

init_fuzzy_target :-
    init_python_target.

% =============================================================================
% Main Compilation Interface
% =============================================================================

%% compile_fuzzy_to_python(+PredicateIndicator, +Options, -Code)
%  Compile a predicate with fuzzy logic support to Python.
compile_fuzzy_to_python(Pred/Arity, Options, Code) :-
    compile_predicate_to_python(Pred/Arity, Options, BaseCode),
    fuzzy_imports(Imports),
    format(string(Code), "~w\n~w", [Imports, BaseCode]).

%% compile_fuzzy_to_python(+PredicateIndicator, -Code)
%  Compile with default options.
compile_fuzzy_to_python(Pred/Arity, Code) :-
    compile_fuzzy_to_python(Pred/Arity, [], Code).

% =============================================================================
% Fuzzy Header and Imports
% =============================================================================

fuzzy_imports(Imports) :-
    Imports = "from unifyweaver.targets.python_runtime.fuzzy_logic import (
    f_and, f_or, f_dist_or, f_union, f_not,
    f_and_batch, f_or_batch, f_dist_or_batch, f_union_batch,
    multiply_scores, blend_scores, top_k, apply_filter, apply_boost
)
import numpy as np
".

fuzzy_header(Header) :-
    fuzzy_imports(Imports),
    format(string(Header), "~w\n", [Imports]).

% =============================================================================
% Goal Translation for Fuzzy Operations
% =============================================================================

%% translate_fuzzy_goal(+Goal, -Code)
%  Translate fuzzy logic goals to Python code.

% f_and/2: Fuzzy AND
% f_and([w(bash, 0.9), w(shell, 0.5)], Result)
translate_fuzzy_goal(f_and(Terms, Result), Code) :-
    !,
    translate_weighted_terms(Terms, PyTerms),
    var_to_python(Result, PyResult),
    format(string(Code),
           "    ~w = f_and(~w, _term_scores)\n",
           [PyResult, PyTerms]).

% f_or/2: Fuzzy OR
translate_fuzzy_goal(f_or(Terms, Result), Code) :-
    !,
    translate_weighted_terms(Terms, PyTerms),
    var_to_python(Result, PyResult),
    format(string(Code),
           "    ~w = f_or(~w, _term_scores)\n",
           [PyResult, PyTerms]).

% f_dist_or/3: Distributed OR with base score
translate_fuzzy_goal(f_dist_or(BaseScore, Terms, Result), Code) :-
    !,
    var_to_python(BaseScore, PyBase),
    translate_weighted_terms(Terms, PyTerms),
    var_to_python(Result, PyResult),
    format(string(Code),
           "    ~w = f_dist_or(~w, ~w, _term_scores)\n",
           [PyResult, PyBase, PyTerms]).

% f_union/3: Non-distributed OR (union)
translate_fuzzy_goal(f_union(BaseScore, Terms, Result), Code) :-
    !,
    var_to_python(BaseScore, PyBase),
    translate_weighted_terms(Terms, PyTerms),
    var_to_python(Result, PyResult),
    format(string(Code),
           "    ~w = f_union(~w, ~w, _term_scores)\n",
           [PyResult, PyBase, PyTerms]).

% f_not/2: Fuzzy NOT
translate_fuzzy_goal(f_not(Score, Result), Code) :-
    !,
    var_to_python(Score, PyScore),
    var_to_python(Result, PyResult),
    format(string(Code),
           "    ~w = f_not(~w)\n",
           [PyResult, PyScore]).

% eval_fuzzy_expr/3: Evaluate expression with term scores
translate_fuzzy_goal(eval_fuzzy_expr(Expr, TermScores, Result), Code) :-
    !,
    translate_fuzzy_expr(Expr, PyExpr),
    var_to_python(TermScores, PyScores),
    var_to_python(Result, PyResult),
    format(string(Code),
           "    _term_scores = dict(~w)\n    ~w = ~w\n",
           [PyScores, PyResult, PyExpr]).

% multiply_scores/3: Element-wise score multiplication
translate_fuzzy_goal(multiply_scores(Scores1, Scores2, Result), Code) :-
    !,
    var_to_python(Scores1, PyS1),
    var_to_python(Scores2, PyS2),
    var_to_python(Result, PyResult),
    format(string(Code),
           "    ~w = multiply_scores(np.array(~w), np.array(~w))\n",
           [PyResult, PyS1, PyS2]).

% blend_scores/4: Blend two score arrays
translate_fuzzy_goal(blend_scores(Alpha, Scores1, Scores2, Result), Code) :-
    !,
    var_to_python(Alpha, PyAlpha),
    var_to_python(Scores1, PyS1),
    var_to_python(Scores2, PyS2),
    var_to_python(Result, PyResult),
    format(string(Code),
           "    ~w = blend_scores(~w, np.array(~w), np.array(~w))\n",
           [PyResult, PyAlpha, PyS1, PyS2]).

% top_k/3: Get top K items by score
translate_fuzzy_goal(top_k(ScoredItems, K, Result), Code) :-
    !,
    var_to_python(ScoredItems, PyItems),
    var_to_python(K, PyK),
    var_to_python(Result, PyResult),
    format(string(Code),
           "    _items, _scores = zip(*~w) if ~w else ([], [])\n    ~w = top_k(list(_items), np.array(_scores), ~w)\n",
           [PyItems, PyItems, PyResult, PyK]).

% apply_filter/3: Apply boolean filter
translate_fuzzy_goal(apply_filter(ScoredItems, Filter, Result), Code) :-
    !,
    var_to_python(ScoredItems, PyItems),
    translate_filter_predicate(Filter, PyFilter),
    var_to_python(Result, PyResult),
    format(string(Code),
           "    _items, _scores = zip(*~w) if ~w else ([], [])\n    _filtered_items, _filtered_scores = apply_filter(list(_items), np.array(_scores), ~w)\n    ~w = list(zip(_filtered_items, _filtered_scores))\n",
           [PyItems, PyItems, PyFilter, PyResult]).

% apply_boost/3: Apply fuzzy boost
translate_fuzzy_goal(apply_boost(ScoredItems, BoostExpr, Result), Code) :-
    !,
    var_to_python(ScoredItems, PyItems),
    translate_boost_expr(BoostExpr, PyBoost),
    var_to_python(Result, PyResult),
    format(string(Code),
           "    _items, _scores = zip(*~w) if ~w else ([], [])\n    _boosted = apply_boost(list(_items), np.array(_scores), ~w)\n    ~w = list(zip(_items, _boosted))\n",
           [PyItems, PyItems, PyBoost, PyResult]).

% =============================================================================
% Batch Operations (for vectorized processing)
% =============================================================================

translate_fuzzy_goal(f_and_batch(Terms, TermScoresBatch, Result), Code) :-
    !,
    translate_weighted_terms(Terms, PyTerms),
    var_to_python(TermScoresBatch, PyScores),
    var_to_python(Result, PyResult),
    format(string(Code),
           "    ~w = f_and_batch(~w, ~w)\n",
           [PyResult, PyTerms, PyScores]).

translate_fuzzy_goal(f_or_batch(Terms, TermScoresBatch, Result), Code) :-
    !,
    translate_weighted_terms(Terms, PyTerms),
    var_to_python(TermScoresBatch, PyScores),
    var_to_python(Result, PyResult),
    format(string(Code),
           "    ~w = f_or_batch(~w, ~w)\n",
           [PyResult, PyTerms, PyScores]).

translate_fuzzy_goal(f_dist_or_batch(BaseScores, Terms, TermScoresBatch, Result), Code) :-
    !,
    var_to_python(BaseScores, PyBase),
    translate_weighted_terms(Terms, PyTerms),
    var_to_python(TermScoresBatch, PyScores),
    var_to_python(Result, PyResult),
    format(string(Code),
           "    ~w = f_dist_or_batch(~w, ~w, ~w)\n",
           [PyResult, PyBase, PyTerms, PyScores]).

% =============================================================================
% Helper Predicates
% =============================================================================

%% translate_weighted_terms(+Terms, -PyTerms)
%  Convert Prolog weighted terms to Python list of tuples.
%  w(bash, 0.9) -> ("bash", 0.9)
translate_weighted_terms(Terms, PyTerms) :-
    translate_terms_list(Terms, TermStrings),
    atomic_list_concat(TermStrings, ', ', Joined),
    format(string(PyTerms), "[~w]", [Joined]).

translate_terms_list([], []).
translate_terms_list([w(Term, Weight)|Rest], [PyTerm|PyRest]) :-
    !,
    format(string(PyTerm), "(\"~w\", ~w)", [Term, Weight]),
    translate_terms_list(Rest, PyRest).
translate_terms_list([Term:Weight|Rest], [PyTerm|PyRest]) :-
    !,
    format(string(PyTerm), "(\"~w\", ~w)", [Term, Weight]),
    translate_terms_list(Rest, PyRest).
translate_terms_list([Term|Rest], [PyTerm|PyRest]) :-
    atom(Term),
    !,
    format(string(PyTerm), "(\"~w\", 1.0)", [Term]),
    translate_terms_list(Rest, PyRest).

%% translate_fuzzy_expr(+Expr, -PyExpr)
%  Translate a fuzzy expression to Python.
translate_fuzzy_expr(f_and(Terms), PyExpr) :-
    !,
    translate_weighted_terms(Terms, PyTerms),
    format(string(PyExpr), "f_and(~w, _term_scores)", [PyTerms]).
translate_fuzzy_expr(f_or(Terms), PyExpr) :-
    !,
    translate_weighted_terms(Terms, PyTerms),
    format(string(PyExpr), "f_or(~w, _term_scores)", [PyTerms]).
translate_fuzzy_expr(f_dist_or(Base, Terms), PyExpr) :-
    !,
    translate_weighted_terms(Terms, PyTerms),
    format(string(PyExpr), "f_dist_or(~w, ~w, _term_scores)", [Base, PyTerms]).
translate_fuzzy_expr(f_union(Base, Terms), PyExpr) :-
    !,
    translate_weighted_terms(Terms, PyTerms),
    format(string(PyExpr), "f_union(~w, ~w, _term_scores)", [Base, PyTerms]).
translate_fuzzy_expr(f_not(Inner), PyExpr) :-
    !,
    translate_fuzzy_expr(Inner, PyInner),
    format(string(PyExpr), "f_not(~w)", [PyInner]).
translate_fuzzy_expr(Num, PyExpr) :-
    number(Num),
    !,
    format(string(PyExpr), "~w", [Num]).

%% translate_filter_predicate(+Filter, -PyFilter)
%  Translate filter predicates to Python lambda.
translate_filter_predicate(is_type(Type), PyFilter) :-
    !,
    format(string(PyFilter), "lambda item: item.get('type') == \"~w\"", [Type]).
translate_filter_predicate(has_account(Account), PyFilter) :-
    !,
    format(string(PyFilter), "lambda item: item.get('account') == \"~w\"", [Account]).
translate_filter_predicate(in_subtree(Subtree), PyFilter) :-
    !,
    format(string(PyFilter), "lambda item: \"~w\" in item.get('path', '')", [Subtree]).
translate_filter_predicate(has_parent(Parent), PyFilter) :-
    !,
    format(string(PyFilter), "lambda item: \"~w\" in item.get('parent', '')", [Parent]).
translate_filter_predicate(has_tag(Tag), PyFilter) :-
    !,
    format(string(PyFilter), "lambda item: \"~w\" in item.get('tags', [])", [Tag]).
translate_filter_predicate(child_of(Node), PyFilter) :-
    !,
    format(string(PyFilter), "lambda item: item.get('parent', '').endswith(\"~w\")", [Node]).
translate_filter_predicate(descendant_of(Node), PyFilter) :-
    !,
    format(string(PyFilter), "lambda item: \"~w\" in item.get('path', '')", [Node]).
translate_filter_predicate(has_depth(N), PyFilter) :-
    !,
    format(string(PyFilter), "lambda item: item.get('depth') == ~w", [N]).
translate_filter_predicate(depth_between(Min, Max), PyFilter) :-
    !,
    format(string(PyFilter), "lambda item: ~w <= item.get('depth', 0) <= ~w", [Min, Max]).
translate_filter_predicate(not(Inner), PyFilter) :-
    !,
    translate_filter_predicate(Inner, PyInner),
    format(string(PyFilter), "lambda item: not (~w)(item)", [PyInner]).
translate_filter_predicate(Filter, PyFilter) :-
    % Default: treat as a function name
    format(string(PyFilter), "~w", [Filter]).

%% translate_boost_expr(+BoostExpr, -PyBoost)
%  Translate boost expressions to Python lambda.
translate_boost_expr(near(Node, Decay), PyBoost) :-
    !,
    format(string(PyBoost),
           "lambda item: ~w ** abs(item.get('depth', 0) - _find_depth(\"~w\"))",
           [Decay, Node]).
translate_boost_expr(Expr, PyBoost) :-
    translate_fuzzy_expr(Expr, PyExpr),
    format(string(PyBoost), "lambda item: ~w", [PyExpr]).

% =============================================================================
% Integration with python_target.pl
% =============================================================================

% Hook into python_target's translate_goal/2 for fuzzy operations
:- multifile python_target:translate_goal/2.

python_target:translate_goal(f_and(Terms, Result), Code) :-
    translate_fuzzy_goal(f_and(Terms, Result), Code).
python_target:translate_goal(f_or(Terms, Result), Code) :-
    translate_fuzzy_goal(f_or(Terms, Result), Code).
python_target:translate_goal(f_dist_or(Base, Terms, Result), Code) :-
    translate_fuzzy_goal(f_dist_or(Base, Terms, Result), Code).
python_target:translate_goal(f_union(Base, Terms, Result), Code) :-
    translate_fuzzy_goal(f_union(Base, Terms, Result), Code).
python_target:translate_goal(f_not(Score, Result), Code) :-
    translate_fuzzy_goal(f_not(Score, Result), Code).
python_target:translate_goal(eval_fuzzy_expr(E, S, R), Code) :-
    translate_fuzzy_goal(eval_fuzzy_expr(E, S, R), Code).
python_target:translate_goal(multiply_scores(A, B, R), Code) :-
    translate_fuzzy_goal(multiply_scores(A, B, R), Code).
python_target:translate_goal(blend_scores(A, S1, S2, R), Code) :-
    translate_fuzzy_goal(blend_scores(A, S1, S2, R), Code).
python_target:translate_goal(top_k(Items, K, R), Code) :-
    translate_fuzzy_goal(top_k(Items, K, R), Code).
python_target:translate_goal(apply_filter(Items, F, R), Code) :-
    translate_fuzzy_goal(apply_filter(Items, F, R), Code).
python_target:translate_goal(apply_boost(Items, B, R), Code) :-
    translate_fuzzy_goal(apply_boost(Items, B, R), Code).
