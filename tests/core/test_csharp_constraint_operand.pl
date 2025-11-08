/**
 * Test cases for diagnosing "unsupported constraint operand" issue
 *
 * This module creates test cases to isolate and understand the
 * "unsupported constraint operand _168" error that Codex encountered.
 *
 * The error appears in constraint_operand/3 in csharp_query_target.pl
 * when compiling recursive plans.
 *
 * Usage:
 *   ?- test_constraint_operand_issue.
 */

:- module(test_csharp_constraint_operand, [
    test_constraint_operand_issue/0,
    test_simple_equality/0,
    test_variable_unification/0,
    test_recursive_with_constraints/0
]).

:- use_module(library(apply)).
:- use_module('../../src/unifyweaver/targets/csharp_query_target').

% =====================================================================
% Test Data
% =====================================================================

setup_constraint_test_data :-
    cleanup_constraint_test_data,

    % Simple facts
    assertz(user:person(alice, engineer)),
    assertz(user:person(bob, teacher)),
    assertz(user:person(charlie, engineer)),

    % Simple equality constraint
    assertz(user:(engineer_only(Name) :- person(Name, Job), Job = engineer)),

    % Variable unification
    assertz(user:(same_job(X, Y) :- person(X, Job), person(Y, Job), X \= Y)),

    % Recursive with constraint
    assertz(user:link(a, b)),
    assertz(user:link(b, c)),
    assertz(user:link(c, d)),
    assertz(user:(path_not_self(X, Y) :- link(X, Y), X \= Y)),
    assertz(user:(path_not_self(X, Z) :- link(X, Y), path_not_self(Y, Z), X \= Z)).

cleanup_constraint_test_data :-
    retractall(user:person(_, _)),
    retractall(user:engineer_only(_)),
    retractall(user:same_job(_, _)),
    retractall(user:link(_, _)),
    retractall(user:path_not_self(_, _)).

% =====================================================================
% Individual Test Cases
% =====================================================================

test_simple_equality :-
    format('~n=== Test 1: Simple Equality Constraint ===~n'),
    setup_constraint_test_data,
    format('Testing: engineer_only(Name) :- person(Name, Job), Job = engineer~n'),

    catch(
        (   build_query_plan(engineer_only/1, [target(csharp_query)], Plan),
            format('Plan structure:~n'),
            get_dict(root, Plan, Root),
            pretty_print_node(Root, 0),
            format('~nPlan generation: SUCCESS~n')
        ),
        Error,
        (   format('Plan generation FAILED with error:~n  ~w~n', [Error]),
            (   sub_atom(Error, _, _, _, 'unsupported constraint operand')
            ->  format('~n*** This is the constraint operand issue! ***~n')
            ;   true
            )
        )
    ),

    cleanup_constraint_test_data.

test_variable_unification :-
    format('~n=== Test 2: Variable Unification Constraint ===~n'),
    setup_constraint_test_data,
    format('Testing: same_job(X, Y) :- person(X, Job), person(Y, Job), X \\= Y~n'),

    catch(
        (   build_query_plan(same_job/2, [target(csharp_query)], Plan),
            format('Plan structure:~n'),
            get_dict(root, Plan, Root),
            pretty_print_node(Root, 0),
            format('~nPlan generation: SUCCESS~n')
        ),
        Error,
        (   format('Plan generation FAILED with error:~n  ~w~n', [Error]),
            (   sub_atom(Error, _, _, _, 'unsupported constraint operand')
            ->  format('~n*** This is the constraint operand issue! ***~n')
            ;   true
            )
        )
    ),

    cleanup_constraint_test_data.

test_recursive_with_constraints :-
    format('~n=== Test 3: Recursive Query with Constraints ===~n'),
    setup_constraint_test_data,
    format('Testing: path_not_self(X, Y) with X \\= Y constraints~n'),

    catch(
        (   build_query_plan(path_not_self/2, [target(csharp_query)], Plan),
            format('Plan structure:~n'),
            get_dict(root, Plan, Root),
            pretty_print_node(Root, 0),
            format('~nPlan generation: SUCCESS~n')
        ),
        Error,
        (   format('Plan generation FAILED with error:~n  ~w~n', [Error]),
            (   sub_atom(Error, _, _, _, 'unsupported constraint operand')
            ->  format('~n*** This is the constraint operand issue! ***~n')
            ;   true
            )
        )
    ),

    cleanup_constraint_test_data.

% =====================================================================
% Main Test Suite
% =====================================================================

test_constraint_operand_issue :-
    format('~n'),
    format('================================================================~n'),
    format('  Constraint Operand Issue Diagnostic Tests~n'),
    format('================================================================~n'),
    format('~n'),
    format('These tests isolate the "unsupported constraint operand" error.~n'),
    format('The error occurs in constraint_operand/3 when the planner~n'),
    format('encounters a constraint with an unexpected operand type.~n'),
    format('~n'),

    test_simple_equality,
    test_variable_unification,
    test_recursive_with_constraints,

    format('~n'),
    format('================================================================~n'),
    format('  Diagnostic Tests Complete~n'),
    format('================================================================~n'),
    format('~n'),
    format('If any test showed "unsupported constraint operand", that~n'),
    format('indicates the specific pattern causing the issue.~n'),
    format('~n').

% =====================================================================
% Helper: Pretty Print Query Plan Nodes
% =====================================================================

pretty_print_node(Node, Indent) :-
    indent_spaces(Indent, Spaces),
    (   is_dict(Node, relation_scan)
    ->  get_dict(predicate, Node, Pred),
        format('~wRelationScan(~w)~n', [Spaces, Pred])

    ;   is_dict(Node, join)
    ->  format('~wJoin~n', [Spaces]),
        get_dict(left, Node, Left),
        get_dict(right, Node, Right),
        Indent1 is Indent + 2,
        pretty_print_node(Left, Indent1),
        pretty_print_node(Right, Indent1)

    ;   is_dict(Node, projection)
    ->  get_dict(columns, Node, Cols),
        format('~wProject(~w)~n', [Spaces, Cols]),
        get_dict(input, Node, Input),
        Indent1 is Indent + 2,
        pretty_print_node(Input, Indent1)

    ;   is_dict(Node, selection)
    ->  get_dict(predicate, Node, Pred),
        format('~wSelection(~w)~n', [Spaces, Pred]),
        get_dict(input, Node, Input),
        Indent1 is Indent + 2,
        pretty_print_node(Input, Indent1)

    ;   is_dict(Node, fixpoint)
    ->  format('~wFixpoint~n', [Spaces]),
        Indent1 is Indent + 2,
        format('~w  Base:~n', [Spaces]),
        get_dict(base, Node, Base),
        Indent2 is Indent + 4,
        pretty_print_node(Base, Indent2),
        format('~w  Recursive:~n', [Spaces]),
        get_dict(recursive, Node, RecList),
        maplist({Indent2}/[R]>>pretty_print_node(R, Indent2), RecList)

    ;   is_dict(Node, recursive_ref)
    ->  get_dict(predicate, Node, Pred),
        get_dict(role, Node, Role),
        format('~wRecursiveRef(~w, ~w)~n', [Spaces, Pred, Role])

    ;   is_dict(Node, union)
    ->  format('~wUnion~n', [Spaces]),
        get_dict(inputs, Node, Inputs),
        Indent1 is Indent + 2,
        maplist({Indent1}/[I]>>pretty_print_node(I, Indent1), Inputs)

    ;   format('~wUnknown node type: ~w~n', [Spaces, Node])
    ).

indent_spaces(N, Spaces) :-
    length(SpaceList, N),
    maplist(=(' '), SpaceList),
    atomic_list_concat(SpaceList, '', Spaces).

% =====================================================================
% Example Standalone Usage
% =====================================================================

/**
 * Example: Run all diagnostic tests
 *
 * $ swipl -q -l tests/core/test_csharp_constraint_operand.pl \
 *         -g test_constraint_operand_issue -t halt
 */
