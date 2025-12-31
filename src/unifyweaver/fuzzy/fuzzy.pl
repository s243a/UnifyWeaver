/**
 * Fuzzy Logic DSL for UnifyWeaver
 *
 * Main module that re-exports all fuzzy logic functionality.
 *
 * Usage:
 *   :- use_module(unifyweaver/fuzzy/fuzzy).
 *
 * Or import specific modules:
 *   :- use_module(unifyweaver/fuzzy/core).       % Core functors
 *   :- use_module(unifyweaver/fuzzy/boolean).    % Boolean operations
 *   :- use_module(unifyweaver/fuzzy/predicates). % Filter predicates
 *   :- use_module(unifyweaver/fuzzy/operators).  % Operator sugar (optional)
 *   :- use_module(unifyweaver/fuzzy/eval).       % Evaluation engine
 *
 * Quick start:
 *
 *   % Evaluate fuzzy AND
 *   ?- f_and([w(bash, 0.9), w(shell, 0.5)], Result).
 *
 *   % With term scores
 *   ?- eval_fuzzy_expr(
 *          f_and([w(bash, 0.9), w(shell, 0.5)]),
 *          [bash-0.8, shell-0.6],
 *          Result
 *      ).
 *
 *   % With operator syntax (after loading operators module)
 *   ?- fuzzy_and(bash:0.9 & shell:0.5, Result).
 */

:- module(fuzzy, [
    % Re-export core
    f_and/1, f_and/2,
    f_or/1, f_or/2,
    f_dist_or/2, f_dist_or/3,
    f_union/2, f_union/3,
    f_not/1, f_not/2,

    % Re-export boolean
    b_and/1, b_and/2,
    b_or/1, b_or/2,
    b_not/1, b_not/2,

    % Re-export predicates
    is_type/1,
    has_account/1,
    has_parent/1,
    in_subtree/1,
    has_tag/1,
    child_of/1,
    descendant_of/1,
    parent_of/1,
    ancestor_of/1,
    sibling_of/1,
    has_depth/1,
    depth_between/2,
    near/3,

    % Re-export evaluation
    eval_fuzzy_expr/3,
    eval_fuzzy_query/4,
    score_items/3,
    score_item/3,
    apply_filter/3,
    apply_boost/3,
    top_k/3,
    multiply_scores/3,
    blend_scores/4
]).

:- use_module(core).
:- use_module(boolean).
:- use_module(predicates).
:- use_module(eval).

% Note: operators module is NOT auto-loaded to avoid potential
% conflicts with Prolog's : operator. Import explicitly if desired:
%   :- use_module(unifyweaver/fuzzy/operators).
