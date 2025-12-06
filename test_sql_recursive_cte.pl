:- encoding(utf8).
% Test SQL Recursive CTEs (WITH RECURSIVE)
% Tests for hierarchical and graph queries

:- use_module('src/unifyweaver/targets/sql_target').

%% ============================================
%% TABLE SCHEMAS
%% ============================================

% Employee hierarchy (for org chart)
:- sql_table(employees, [id-integer, name-text, manager_id-integer, dept-text]).

% Category hierarchy (for nested categories)
:- sql_table(categories, [id-integer, name-text, parent_id-integer]).

% Graph edges (for path finding)
:- sql_table(edges, [from_node-integer, to_node-integer, weight-integer]).

%% ============================================
%% TEST 1: Simple Org Chart (employee hierarchy)
%% Basic recursive query without computed columns
%% ============================================

% Declare the recursive CTE as a virtual table
:- sql_recursive_table(org_tree, [id-integer, name-text, manager_id-integer]).

% Base case: top-level employees (no manager - using IS NULL)
org_base(Id, Name, ManagerId) :-
    employees(Id, Name, ManagerId, _),
    sql_is_null(ManagerId).

% Recursive case: employees with managers in the tree
org_recursive(Id, Name, ManagerId) :-
    employees(Id, Name, ManagerId, _),
    org_tree(ManagerId, _, _).

% Main query: select from the CTE
org_result(Id, Name, ManagerId) :-
    org_tree(Id, Name, ManagerId).

test1 :-
    format('~n=== Test 1: Simple Org Chart ===~n'),
    format('Base: Employees where manager_id IS NULL~n'),
    format('Recursive: Join employees to org_tree on manager_id~n~n'),
    compile_recursive_cte(
        org_tree,
        [id, name, manager_id],
        recursive_cte(org_base/3, org_recursive/3),
        org_result/3,
        SQL
    ),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 2: Category Tree (nested categories)
%% ============================================

:- sql_recursive_table(cat_tree, [id-integer, name-text, parent_id-integer]).

% Base case: root categories (no parent)
cat_base(Id, Name, ParentId) :-
    categories(Id, Name, ParentId),
    sql_is_null(ParentId).

% Recursive case: child categories
cat_recursive(Id, Name, ParentId) :-
    categories(Id, Name, ParentId),
    cat_tree(ParentId, _, _).

% Main query
cat_result(Id, Name, ParentId) :-
    cat_tree(Id, Name, ParentId).

test2 :-
    format('~n=== Test 2: Category Tree ===~n'),
    format('Base: Root categories (parent_id IS NULL)~n'),
    format('Recursive: Join categories to cat_tree on parent_id~n~n'),
    compile_recursive_cte(
        cat_tree,
        [id, name, parent_id],
        recursive_cte(cat_base/3, cat_recursive/3),
        cat_result/3,
        SQL
    ),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 3: Graph Reachability
%% ============================================

:- sql_recursive_table(reachable, [node-integer]).

% Base case: start from node 1
reach_base(Node) :-
    edges(1, Node, _).

% Recursive case: follow edges
reach_recursive(Node) :-
    reachable(From),
    edges(From, Node, _).

% Main query
reach_result(Node) :-
    reachable(Node).

test3 :-
    format('~n=== Test 3: Graph Reachability ===~n'),
    format('Base: Direct edges from node 1~n'),
    format('Recursive: Follow edges from reachable nodes~n~n'),
    compile_recursive_cte(
        reachable,
        [node],
        recursive_cte(reach_base/1, reach_recursive/1),
        reach_result/1,
        SQL
    ),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 4: Path Finding with Distance
%% ============================================

:- sql_recursive_table(paths, [start_node-integer, end_node-integer, hops-integer]).

% Base case: direct edges (1 hop)
path_base(Start, End, Hops) :-
    edges(Start, End, _),
    Hops = 1.

% Recursive case: extend paths
path_recursive(Start, End, Hops) :-
    paths(Start, Mid, _),
    edges(Mid, End, _),
    Hops = 2.  % Simplified - just show structure

% Main query
path_result(Start, End, Hops) :-
    paths(Start, End, Hops).

test4 :-
    format('~n=== Test 4: Path Finding ===~n'),
    format('Base: Direct edges (1 hop)~n'),
    format('Recursive: Extend paths through intermediate nodes~n~n'),
    compile_recursive_cte(
        paths,
        [start_node, end_node, hops],
        recursive_cte(path_base/3, path_recursive/3),
        path_result/3,
        SQL
    ),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 5: With UNION (distinct) option
%% ============================================

test5 :-
    format('~n=== Test 5: With UNION (distinct) option ===~n'),
    compile_recursive_cte(
        org_tree,
        [id, name, manager_id],
        recursive_cte(org_base/3, org_recursive/3),
        org_result/3,
        [union_type(distinct)],
        SQL
    ),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 6: With view_name option
%% ============================================

test6 :-
    format('~n=== Test 6: With view_name option ===~n'),
    compile_recursive_cte(
        org_tree,
        [id, name, manager_id],
        recursive_cte(org_base/3, org_recursive/3),
        org_result/3,
        [view_name(employee_hierarchy)],
        SQL
    ),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 7: Ancestors query (upward traversal)
%% ============================================

:- sql_recursive_table(ancestors, [emp_id-integer, ancestor_id-integer]).

% Base case: direct manager
anc_base(EmpId, AncestorId) :-
    employees(EmpId, _, AncestorId, _),
    sql_is_not_null(AncestorId).

% Recursive case: ancestor's manager
anc_recursive(EmpId, AncestorId) :-
    ancestors(EmpId, Mid),
    employees(Mid, _, AncestorId, _),
    sql_is_not_null(AncestorId).

% Main query: find all ancestors of employee 5
anc_result(EmpId, AncestorId) :-
    ancestors(EmpId, AncestorId),
    EmpId = 5.

test7 :-
    format('~n=== Test 7: Ancestors (upward traversal) ===~n'),
    format('Base: Direct manager~n'),
    format('Recursive: Manager of ancestor~n~n'),
    compile_recursive_cte(
        ancestors,
        [emp_id, ancestor_id],
        recursive_cte(anc_base/2, anc_recursive/2),
        anc_result/2,
        SQL
    ),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% TEST 8: Bill of Materials
%% ============================================

:- sql_table(parts, [part_id-integer, part_name-text]).
:- sql_table(assemblies, [parent_part-integer, child_part-integer, quantity-integer]).
:- sql_recursive_table(bom, [part_id-integer, part_name-text]).

% Base case: top-level part
bom_base(PartId, PartName) :-
    parts(PartId, PartName),
    PartId = 1.  % Starting part

% Recursive case: component parts
bom_recursive(PartId, PartName) :-
    bom(ParentId, _),
    assemblies(ParentId, PartId, _),
    parts(PartId, PartName).

% Main query
bom_result(PartId, PartName) :-
    bom(PartId, PartName).

test8 :-
    format('~n=== Test 8: Bill of Materials ===~n'),
    format('Base: Top-level part (part_id = 1)~n'),
    format('Recursive: Join assemblies to find child parts~n~n'),
    compile_recursive_cte(
        bom,
        [part_id, part_name],
        recursive_cte(bom_base/2, bom_recursive/2),
        bom_result/2,
        SQL
    ),
    format('Generated SQL:~n~w~n', [SQL]).

%% ============================================
%% RUN ALL TESTS
%% ============================================

test_all :-
    format('~n========================================~n'),
    format('  SQL Recursive CTE Tests~n'),
    format('========================================~n'),
    test1,
    test2,
    test3,
    test4,
    test5,
    test6,
    test7,
    test8,
    format('~n========================================~n'),
    format('  All tests completed!~n'),
    format('========================================~n').

%% Entry point
main :- test_all.

:- initialization(main, main).
