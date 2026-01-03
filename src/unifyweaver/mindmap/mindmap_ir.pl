% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% mindmap_ir.pl - Mind Map Intermediate Representation
%
% This module provides a normalized intermediate representation for mind maps
% that can be processed by layout algorithms and rendered to various targets.
%
% The IR separates the graph structure from layout and styling concerns,
% enabling pipeline processing and target-independent transformations.
%
% IR Structure:
%   mindmap_ir(
%       Name,
%       Graph = graph(Nodes, Edges, Root),
%       Attributes = attrs(NodeAttrs, EdgeAttrs, GlobalAttrs),
%       Constraints = [constraint(Type, Options), ...],
%       Preferences = [preference(Type, Weight, Options), ...]
%   )

:- module(mindmap_ir, [
    % IR construction
    build_ir/2,                     % build_ir(+SpecName, -IR)
    build_ir_from_nodes/3,          % build_ir_from_nodes(+Nodes, +Edges, -IR)

    % IR access
    ir_name/2,                      % ir_name(+IR, -Name)
    ir_graph/2,                     % ir_graph(+IR, -Graph)
    ir_nodes/2,                     % ir_nodes(+IR, -Nodes)
    ir_edges/2,                     % ir_edges(+IR, -Edges)
    ir_root/2,                      % ir_root(+IR, -RootId)
    ir_attributes/2,                % ir_attributes(+IR, -Attributes)
    ir_constraints/2,               % ir_constraints(+IR, -Constraints)
    ir_preferences/2,               % ir_preferences(+IR, -Preferences)

    % Node access
    ir_node/3,                      % ir_node(+IR, +Id, -NodeData)
    ir_node_attr/4,                 % ir_node_attr(+IR, +Id, +AttrName, -Value)
    ir_node_label/3,                % ir_node_label(+IR, +Id, -Label)
    ir_node_type/3,                 % ir_node_type(+IR, +Id, -Type)

    % Edge access
    ir_edge/4,                      % ir_edge(+IR, +From, +To, -EdgeData)
    ir_children/3,                  % ir_children(+IR, +Id, -Children)
    ir_parent/3,                    % ir_parent(+IR, +Id, -Parent)
    ir_descendants/3,               % ir_descendants(+IR, +Id, -Descendants)
    ir_ancestors/3,                 % ir_ancestors(+IR, +Id, -Ancestors)

    % Tree traversal
    ir_traverse_preorder/3,         % ir_traverse_preorder(+IR, +RootId, -OrderedIds)
    ir_traverse_postorder/3,        % ir_traverse_postorder(+IR, +RootId, -OrderedIds)
    ir_traverse_levelorder/3,       % ir_traverse_levelorder(+IR, +RootId, -OrderedIds)
    ir_depth/3,                     % ir_depth(+IR, +Id, -Depth)
    ir_max_depth/2,                 % ir_max_depth(+IR, -MaxDepth)

    % IR transformation
    ir_add_positions/3,             % ir_add_positions(+IR, +Positions, -NewIR)
    ir_get_positions/2,             % ir_get_positions(+IR, -Positions)
    ir_update_node_attr/5,          % ir_update_node_attr(+IR, +Id, +Attr, +Value, -NewIR)

    % Validation
    validate_ir/1,                  % validate_ir(+IR)
    ir_is_tree/1,                   % ir_is_tree(+IR) - true if IR forms a tree
    ir_is_connected/1,              % ir_is_connected(+IR) - true if graph is connected

    % Conversion
    ir_to_dict/2,                   % ir_to_dict(+IR, -Dict) - for JSON/Python interop
    dict_to_ir/2,                   % dict_to_ir(+Dict, -IR)

    % Testing
    test_mindmap_ir/0
]).

:- use_module(library(lists)).

% Conditionally load mindmap_dsl
:- if(exists_source(library('./mindmap_dsl'))).
:- use_module('./mindmap_dsl').
:- endif.

% ============================================================================
% IR STRUCTURE
% ============================================================================

% IR term structure:
% mindmap_ir(Name, Graph, Attributes, Constraints, Preferences)
%
% Where:
%   Name = atom
%   Graph = graph(Nodes, Edges, RootId)
%     Nodes = [node(Id, Props), ...]
%     Edges = [edge(From, To, Props), ...]
%     RootId = atom
%   Attributes = attrs(NodeAttrs, EdgeAttrs, GlobalAttrs)
%     NodeAttrs = [Id-AttrDict, ...]
%     EdgeAttrs = [(From,To)-AttrDict, ...]
%     GlobalAttrs = AttrDict
%   Constraints = [constraint(Type, Options), ...]
%   Preferences = [preference(Type, Weight, Options), ...]

% ============================================================================
% IR CONSTRUCTION
% ============================================================================

%% build_ir(+SpecName, -IR)
%
%  Build intermediate representation from a mind map specification.
%
%  @param SpecName atom - name of the mind map specification
%  @param IR       term - constructed IR
%
build_ir(SpecName, IR) :-
    % Get nodes and edges from DSL
    (   catch(get_mindmap_nodes(SpecName, RawNodes), _, fail)
    ->  true
    ;   RawNodes = []
    ),
    (   catch(get_mindmap_edges(SpecName, RawEdges), _, fail)
    ->  true
    ;   RawEdges = []
    ),
    % Find root
    find_root(RawNodes, RawEdges, RootId),
    % Build graph structure
    Graph = graph(RawNodes, RawEdges, RootId),
    % Extract attributes
    extract_all_attributes(RawNodes, RawEdges, Attributes),
    % Get constraints and preferences
    collect_constraints(SpecName, Constraints),
    collect_preferences(SpecName, Preferences),
    % Construct IR
    IR = mindmap_ir(SpecName, Graph, Attributes, Constraints, Preferences).

%% build_ir_from_nodes(+Nodes, +Edges, -IR)
%
%  Build IR from explicit node and edge lists.
%
%  @param Nodes list - list of node(Id, Props) terms
%  @param Edges list - list of edge(From, To, Props) terms
%  @param IR    term - constructed IR
%
build_ir_from_nodes(Nodes, Edges, IR) :-
    find_root(Nodes, Edges, RootId),
    Graph = graph(Nodes, Edges, RootId),
    extract_all_attributes(Nodes, Edges, Attributes),
    IR = mindmap_ir(anonymous, Graph, Attributes, [], []).

%% find_root(+Nodes, +Edges, -RootId)
%
%  Find the root node of the graph.
%  Root is either marked with type(root) or has no incoming edges.
%
find_root(Nodes, Edges, RootId) :-
    % First look for explicit root
    (   member(node(RootId, Props), Nodes),
        member(type(root), Props)
    ->  true
    ;   % Find node with no incoming edges
        member(node(RootId, _), Nodes),
        \+ member(edge(_, RootId, _), Edges)
    ->  true
    ;   % Fallback to first node
        Nodes = [node(RootId, _) | _]
    ->  true
    ;   RootId = none
    ).

%% extract_all_attributes(+Nodes, +Edges, -Attributes)
%
%  Extract and normalize attributes from nodes and edges.
%
extract_all_attributes(Nodes, Edges, attrs(NodeAttrs, EdgeAttrs, GlobalAttrs)) :-
    % Node attributes
    findall(Id-AttrDict,
            (member(node(Id, Props), Nodes),
             props_to_dict(Props, AttrDict)),
            NodeAttrs),
    % Edge attributes
    findall((From,To)-AttrDict,
            (member(edge(From, To, Props), Edges),
             props_to_dict(Props, AttrDict)),
            EdgeAttrs),
    % Global attributes (empty for now)
    GlobalAttrs = global{}.

%% props_to_dict(+Props, -Dict)
%
%  Convert property list to dictionary.
%
props_to_dict(Props, Dict) :-
    findall(Key-Value, (member(Prop, Props), prop_kv(Prop, Key, Value)), Pairs),
    dict_create(Dict, attr, Pairs).

prop_kv(label(V), label, V) :- !.
prop_kv(type(V), type, V) :- !.
prop_kv(parent(V), parent, V) :- !.
prop_kv(link(V), link, V) :- !.
prop_kv(style(V), style, V) :- !.
prop_kv(cluster(V), cluster, V) :- !.
prop_kv(importance(V), importance, V) :- !.
prop_kv(position(X, Y), position, pos(X, Y)) :- !.
prop_kv(x(V), x, V) :- !.
prop_kv(y(V), y, V) :- !.
prop_kv(weight(V), weight, V) :- !.
prop_kv(implicit(V), implicit, V) :- !.
prop_kv(Term, Key, Value) :-
    Term =.. [Key, Value].

%% collect_constraints(+SpecName, -Constraints)
%
%  Collect all constraints for a specification.
%
collect_constraints(_SpecName, Constraints) :-
    findall(constraint(Type, Opts),
            mindmap_constraint(Type, Opts),
            Constraints).

%% collect_preferences(+SpecName, -Preferences)
%
%  Collect all preferences for a specification.
%
collect_preferences(_SpecName, Preferences) :-
    findall(preference(Type, Weight, Opts),
            (mindmap_preference(Type, Opts),
             (member(weight(Weight), Opts) -> true ; Weight = 1.0)),
            Preferences).

% ============================================================================
% IR ACCESS
% ============================================================================

%% ir_name(+IR, -Name)
ir_name(mindmap_ir(Name, _, _, _, _), Name).

%% ir_graph(+IR, -Graph)
ir_graph(mindmap_ir(_, Graph, _, _, _), Graph).

%% ir_nodes(+IR, -Nodes)
ir_nodes(mindmap_ir(_, graph(Nodes, _, _), _, _, _), Nodes).

%% ir_edges(+IR, -Edges)
ir_edges(mindmap_ir(_, graph(_, Edges, _), _, _, _), Edges).

%% ir_root(+IR, -RootId)
ir_root(mindmap_ir(_, graph(_, _, RootId), _, _, _), RootId).

%% ir_attributes(+IR, -Attributes)
ir_attributes(mindmap_ir(_, _, Attributes, _, _), Attributes).

%% ir_constraints(+IR, -Constraints)
ir_constraints(mindmap_ir(_, _, _, Constraints, _), Constraints).

%% ir_preferences(+IR, -Preferences)
ir_preferences(mindmap_ir(_, _, _, _, Preferences), Preferences).

% ============================================================================
% NODE ACCESS
% ============================================================================

%% ir_node(+IR, +Id, -NodeData)
%
%  Get node data by id.
%
ir_node(IR, Id, NodeData) :-
    ir_nodes(IR, Nodes),
    member(node(Id, Props), Nodes),
    NodeData = node(Id, Props).

%% ir_node_attr(+IR, +Id, +AttrName, -Value)
%
%  Get a specific attribute of a node.
%
ir_node_attr(IR, Id, AttrName, Value) :-
    ir_attributes(IR, attrs(NodeAttrs, _, _)),
    member(Id-AttrDict, NodeAttrs),
    get_dict(AttrName, AttrDict, Value).

%% ir_node_label(+IR, +Id, -Label)
%
%  Get the label of a node.
%
ir_node_label(IR, Id, Label) :-
    (   ir_node_attr(IR, Id, label, Label)
    ->  true
    ;   atom_string(Id, Label)
    ).

%% ir_node_type(+IR, +Id, -Type)
%
%  Get the type of a node.
%
ir_node_type(IR, Id, Type) :-
    (   ir_node_attr(IR, Id, type, Type)
    ->  true
    ;   Type = default
    ).

% ============================================================================
% EDGE ACCESS
% ============================================================================

%% ir_edge(+IR, +From, +To, -EdgeData)
%
%  Get edge data.
%
ir_edge(IR, From, To, EdgeData) :-
    ir_edges(IR, Edges),
    member(edge(From, To, Props), Edges),
    EdgeData = edge(From, To, Props).

%% ir_children(+IR, +Id, -Children)
%
%  Get immediate children of a node.
%
ir_children(IR, Id, Children) :-
    ir_edges(IR, Edges),
    findall(ChildId, member(edge(Id, ChildId, _), Edges), Children).

%% ir_parent(+IR, +Id, -Parent)
%
%  Get parent of a node (fails if root).
%
ir_parent(IR, Id, Parent) :-
    ir_edges(IR, Edges),
    member(edge(Parent, Id, _), Edges),
    !.

%% ir_descendants(+IR, +Id, -Descendants)
%
%  Get all descendants of a node.
%
ir_descendants(IR, Id, Descendants) :-
    ir_children(IR, Id, Children),
    (   Children = []
    ->  Descendants = []
    ;   findall(Desc, (member(C, Children), ir_descendants(IR, C, ChildDesc),
                       (Desc = C ; member(Desc, ChildDesc))), Descendants)
    ).

%% ir_ancestors(+IR, +Id, -Ancestors)
%
%  Get all ancestors of a node (path to root).
%
ir_ancestors(IR, Id, Ancestors) :-
    (   ir_parent(IR, Id, Parent)
    ->  ir_ancestors(IR, Parent, ParentAncestors),
        Ancestors = [Parent | ParentAncestors]
    ;   Ancestors = []
    ).

% ============================================================================
% TREE TRAVERSAL
% ============================================================================

%% ir_traverse_preorder(+IR, +RootId, -OrderedIds)
%
%  Pre-order traversal: visit node before children.
%
ir_traverse_preorder(IR, RootId, OrderedIds) :-
    ir_children(IR, RootId, Children),
    findall(ChildOrder,
            (member(C, Children), ir_traverse_preorder(IR, C, ChildOrder)),
            ChildOrders),
    append(ChildOrders, FlatChildren),
    OrderedIds = [RootId | FlatChildren].

%% ir_traverse_postorder(+IR, +RootId, -OrderedIds)
%
%  Post-order traversal: visit children before node.
%
ir_traverse_postorder(IR, RootId, OrderedIds) :-
    ir_children(IR, RootId, Children),
    findall(ChildOrder,
            (member(C, Children), ir_traverse_postorder(IR, C, ChildOrder)),
            ChildOrders),
    append(ChildOrders, FlatChildren),
    append(FlatChildren, [RootId], OrderedIds).

%% ir_traverse_levelorder(+IR, +RootId, -OrderedIds)
%
%  Level-order (breadth-first) traversal.
%
ir_traverse_levelorder(IR, RootId, OrderedIds) :-
    levelorder_queue(IR, [RootId], OrderedIds).

levelorder_queue(_, [], []) :- !.
levelorder_queue(IR, [Id | Rest], [Id | OrderedIds]) :-
    ir_children(IR, Id, Children),
    append(Rest, Children, NewQueue),
    levelorder_queue(IR, NewQueue, OrderedIds).

%% ir_depth(+IR, +Id, -Depth)
%
%  Get depth of a node (root = 0).
%
ir_depth(IR, Id, Depth) :-
    ir_root(IR, RootId),
    (   Id == RootId
    ->  Depth = 0
    ;   ir_ancestors(IR, Id, Ancestors),
        length(Ancestors, Depth)
    ).

%% ir_max_depth(+IR, -MaxDepth)
%
%  Get maximum depth of the tree.
%
ir_max_depth(IR, MaxDepth) :-
    ir_nodes(IR, Nodes),
    findall(D, (member(node(Id, _), Nodes), ir_depth(IR, Id, D)), Depths),
    (   Depths = []
    ->  MaxDepth = 0
    ;   max_list(Depths, MaxDepth)
    ).

% ============================================================================
% IR TRANSFORMATION
% ============================================================================

%% ir_add_positions(+IR, +Positions, -NewIR)
%
%  Add computed positions to the IR.
%  Positions is a list of position(Id, X, Y) terms.
%
ir_add_positions(mindmap_ir(Name, Graph, attrs(NodeAttrs, EdgeAttrs, GlobalAttrs), Constraints, Preferences),
                 Positions,
                 mindmap_ir(Name, Graph, attrs(NewNodeAttrs, EdgeAttrs, GlobalAttrs), Constraints, Preferences)) :-
    update_node_positions(NodeAttrs, Positions, NewNodeAttrs).

update_node_positions([], _, []).
update_node_positions([Id-AttrDict | Rest], Positions, [Id-NewAttrDict | NewRest]) :-
    (   member(position(Id, X, Y), Positions)
    ->  put_dict([x-X, y-Y], AttrDict, NewAttrDict)
    ;   NewAttrDict = AttrDict
    ),
    update_node_positions(Rest, Positions, NewRest).

%% ir_get_positions(+IR, -Positions)
%
%  Extract positions from IR.
%
ir_get_positions(IR, Positions) :-
    ir_attributes(IR, attrs(NodeAttrs, _, _)),
    findall(position(Id, X, Y),
            (member(Id-AttrDict, NodeAttrs),
             get_dict(x, AttrDict, X),
             get_dict(y, AttrDict, Y)),
            Positions).

%% ir_update_node_attr(+IR, +Id, +Attr, +Value, -NewIR)
%
%  Update a specific node attribute.
%
ir_update_node_attr(mindmap_ir(Name, Graph, attrs(NodeAttrs, EdgeAttrs, GlobalAttrs), C, P),
                    Id, Attr, Value,
                    mindmap_ir(Name, Graph, attrs(NewNodeAttrs, EdgeAttrs, GlobalAttrs), C, P)) :-
    update_single_attr(NodeAttrs, Id, Attr, Value, NewNodeAttrs).

update_single_attr([], _, _, _, []).
update_single_attr([Id-AttrDict | Rest], Id, Attr, Value, [Id-NewAttrDict | Rest]) :-
    !,
    put_dict(Attr, AttrDict, Value, NewAttrDict).
update_single_attr([Other | Rest], Id, Attr, Value, [Other | NewRest]) :-
    update_single_attr(Rest, Id, Attr, Value, NewRest).

% ============================================================================
% VALIDATION
% ============================================================================

%% validate_ir(+IR)
%
%  Validate IR structure.
%
validate_ir(IR) :-
    IR = mindmap_ir(Name, Graph, Attributes, Constraints, Preferences),
    atom(Name),
    Graph = graph(Nodes, Edges, RootId),
    is_list(Nodes),
    is_list(Edges),
    atom(RootId),
    Attributes = attrs(NodeAttrs, EdgeAttrs, _GlobalAttrs),
    is_list(NodeAttrs),
    is_list(EdgeAttrs),
    is_list(Constraints),
    is_list(Preferences),
    % Verify root exists
    (   RootId == none
    ->  true
    ;   member(node(RootId, _), Nodes)
    ).

%% ir_is_tree(+IR)
%
%  Check if the IR forms a valid tree (single root, no cycles).
%
ir_is_tree(IR) :-
    ir_root(IR, RootId),
    RootId \== none,
    ir_nodes(IR, Nodes),
    % Every non-root node has exactly one parent
    forall((member(node(Id, _), Nodes), Id \== RootId),
           (findall(P, ir_edge(IR, P, Id, _), Parents),
            length(Parents, 1))),
    % No cycles (all nodes reachable from root)
    ir_traverse_preorder(IR, RootId, Reachable),
    length(Nodes, N),
    length(Reachable, N).

%% ir_is_connected(+IR)
%
%  Check if all nodes are connected.
%
ir_is_connected(IR) :-
    ir_root(IR, RootId),
    RootId \== none,
    ir_nodes(IR, Nodes),
    ir_traverse_preorder(IR, RootId, Reachable),
    length(Nodes, N),
    length(Reachable, N).

% ============================================================================
% CONVERSION
% ============================================================================

%% ir_to_dict(+IR, -Dict)
%
%  Convert IR to dictionary for JSON/Python interop.
%
ir_to_dict(IR, Dict) :-
    ir_name(IR, Name),
    ir_nodes(IR, Nodes),
    ir_edges(IR, Edges),
    ir_root(IR, RootId),
    % Convert nodes
    findall(NodeDict,
            (member(node(Id, Props), Nodes),
             props_to_dict(Props, AttrDict),
             NodeDict = node{id: Id, attrs: AttrDict}),
            NodeDicts),
    % Convert edges
    findall(EdgeDict,
            (member(edge(From, To, Props), Edges),
             props_to_dict(Props, AttrDict),
             EdgeDict = edge{from: From, to: To, attrs: AttrDict}),
            EdgeDicts),
    Dict = mindmap{
        name: Name,
        root: RootId,
        nodes: NodeDicts,
        edges: EdgeDicts
    }.

%% dict_to_ir(+Dict, -IR)
%
%  Convert dictionary to IR.
%
dict_to_ir(Dict, IR) :-
    get_dict(name, Dict, Name),
    get_dict(nodes, Dict, NodeDicts),
    get_dict(edges, Dict, EdgeDicts),
    get_dict(root, Dict, RootId),
    % Convert nodes
    findall(node(Id, Props),
            (member(NodeDict, NodeDicts),
             get_dict(id, NodeDict, Id),
             (get_dict(attrs, NodeDict, AttrDict) -> dict_to_props(AttrDict, Props) ; Props = [])),
            Nodes),
    % Convert edges
    findall(edge(From, To, Props),
            (member(EdgeDict, EdgeDicts),
             get_dict(from, EdgeDict, From),
             get_dict(to, EdgeDict, To),
             (get_dict(attrs, EdgeDict, AttrDict) -> dict_to_props(AttrDict, Props) ; Props = [])),
            Edges),
    build_ir_from_nodes(Nodes, Edges, BaseIR),
    % Override name and root
    BaseIR = mindmap_ir(_, Graph, Attrs, Cons, Prefs),
    Graph = graph(N, E, _),
    IR = mindmap_ir(Name, graph(N, E, RootId), Attrs, Cons, Prefs).

dict_to_props(AttrDict, Props) :-
    dict_pairs(AttrDict, _, Pairs),
    findall(Prop, (member(Key-Value, Pairs), Prop =.. [Key, Value]), Props).

% ============================================================================
% TESTING
% ============================================================================

test_mindmap_ir :-
    format('~n=== Mind Map IR Tests ===~n~n'),

    % Create test data
    TestNodes = [
        node(root, [label("Root"), type(root)]),
        node(a, [label("A"), parent(root)]),
        node(b, [label("B"), parent(root)]),
        node(c, [label("C"), parent(a)])
    ],
    TestEdges = [
        edge(root, a, []),
        edge(root, b, []),
        edge(a, c, [])
    ],

    % Test 1: Build IR
    format('Test 1: Build IR from nodes...~n'),
    build_ir_from_nodes(TestNodes, TestEdges, IR),
    (   validate_ir(IR)
    ->  format('  PASS: IR built and valid~n')
    ;   format('  FAIL: IR invalid~n')
    ),

    % Test 2: Access root
    format('~nTest 2: Access root...~n'),
    ir_root(IR, RootId),
    (   RootId == root
    ->  format('  PASS: Root is ~w~n', [RootId])
    ;   format('  FAIL: Expected root, got ~w~n', [RootId])
    ),

    % Test 3: Get children
    format('~nTest 3: Get children...~n'),
    ir_children(IR, root, RootChildren),
    (   length(RootChildren, 2)
    ->  format('  PASS: Root has 2 children: ~w~n', [RootChildren])
    ;   format('  FAIL: Expected 2 children, got ~w~n', [RootChildren])
    ),

    % Test 4: Traverse preorder
    format('~nTest 4: Preorder traversal...~n'),
    ir_traverse_preorder(IR, root, PreOrder),
    format('  Order: ~w~n', [PreOrder]),
    (   PreOrder = [root | _]
    ->  format('  PASS: Preorder starts with root~n')
    ;   format('  FAIL: Invalid preorder~n')
    ),

    % Test 5: Tree check
    format('~nTest 5: Tree check...~n'),
    (   ir_is_tree(IR)
    ->  format('  PASS: IR is a valid tree~n')
    ;   format('  FAIL: IR is not a tree~n')
    ),

    % Test 6: Depth calculation
    format('~nTest 6: Depth calculation...~n'),
    ir_depth(IR, c, DepthC),
    (   DepthC =:= 2
    ->  format('  PASS: Depth of c is ~w~n', [DepthC])
    ;   format('  FAIL: Expected depth 2, got ~w~n', [DepthC])
    ),

    % Test 7: Add positions
    format('~nTest 7: Add positions...~n'),
    Positions = [position(root, 100, 100), position(a, 150, 200), position(b, 50, 200), position(c, 175, 300)],
    ir_add_positions(IR, Positions, IRWithPos),
    ir_get_positions(IRWithPos, RetrievedPos),
    length(RetrievedPos, NumPos),
    (   NumPos =:= 4
    ->  format('  PASS: ~w positions stored and retrieved~n', [NumPos])
    ;   format('  FAIL: Expected 4 positions, got ~w~n', [NumPos])
    ),

    % Test 8: Convert to dict
    format('~nTest 8: Convert to dict...~n'),
    ir_to_dict(IR, Dict),
    (   is_dict(Dict)
    ->  format('  PASS: Converted to dict~n')
    ;   format('  FAIL: Conversion failed~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% MODULE INITIALIZATION
% ============================================================================

:- initialization((
    format('Mind Map IR module loaded~n', [])
), now).
