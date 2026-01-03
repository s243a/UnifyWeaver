% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% graphviz_renderer.pl - GraphViz DOT Renderer for Mind Maps
%
% Generates GraphViz DOT format output for visualization with external tools.
% The DOT format is widely supported by:
% - GraphViz (dot, neato, fdp, etc.)
% - Online viewers (GraphViz Online, Edotor)
% - Python (graphviz, pygraphviz)
% - IDE plugins
%
% Usage:
%   ?- render_graphviz(Nodes, Edges, Positions, Options, DOT).

:- module(mindmap_render_graphviz, [
    % Component interface
    type_info/1,
    validate_config/1,
    init_component/2,
    compile_component/4,
    render/3,

    % Direct API
    render_graphviz/5,              % render_graphviz(+Nodes, +Edges, +Positions, +Options, -DOT)
    render_graphviz/4,              % render_graphviz(+Nodes, +Edges, +Options, -DOT) - no positions
    render_graphviz_clustered/6,    % render_graphviz_clustered(+Nodes, +Edges, +Clusters, +Positions, +Options, -DOT)

    % Cluster API
    define_cluster/3,               % define_cluster(+ClusterId, +NodeIds, +Options)
    get_cluster_nodes/2,            % get_cluster_nodes(+ClusterId, -NodeIds)
    clear_clusters/0,               % clear_clusters
    cluster_from_node_props/2,      % cluster_from_node_props(+Nodes, -Clusters)

    % Testing
    test_graphviz_renderer/0,
    test_cluster_support/0
]).

:- use_module(library(lists)).

% ============================================================================
% CLUSTER STORAGE
% ============================================================================

:- dynamic graphviz_cluster/3.  % graphviz_cluster(ClusterId, NodeIds, Options)

%% define_cluster(+ClusterId, +NodeIds, +Options)
%
%  Define a cluster (subgraph) containing specified nodes.
%
%  Options:
%    - label(Label) - Cluster label
%    - style(Style) - Border style: solid, dashed, dotted, bold
%    - color(Color) - Border color
%    - bgcolor(Color) - Background color
%    - fontcolor(Color) - Label font color
%    - rank(same) - Force same rank for all nodes
%
define_cluster(ClusterId, NodeIds, Options) :-
    retractall(graphviz_cluster(ClusterId, _, _)),
    assertz(graphviz_cluster(ClusterId, NodeIds, Options)).

%% get_cluster_nodes(+ClusterId, -NodeIds)
get_cluster_nodes(ClusterId, NodeIds) :-
    graphviz_cluster(ClusterId, NodeIds, _).

%% clear_clusters
clear_clusters :-
    retractall(graphviz_cluster(_, _, _)).

%% cluster_from_node_props(+Nodes, -Clusters)
%
%  Extract cluster definitions from node properties.
%  Nodes with cluster(ClusterName) property are grouped.
%
cluster_from_node_props(Nodes, Clusters) :-
    findall(ClusterName-NodeId,
        (   member(node(NodeId, Props), Nodes),
            member(cluster(ClusterName), Props)
        ),
        ClusterPairs),
    group_by_cluster(ClusterPairs, Clusters).

group_by_cluster(Pairs, Clusters) :-
    findall(ClusterName, member(ClusterName-_, Pairs), AllNames),
    sort(AllNames, UniqueNames),
    findall(cluster(Name, NodeIds, []),
        (   member(Name, UniqueNames),
            findall(Id, member(Name-Id, Pairs), NodeIds)
        ),
        Clusters).

% ============================================================================
% COMPONENT INTERFACE
% ============================================================================

type_info(info{
    name: graphviz,
    category: mindmap_renderer,
    description: "GraphViz DOT format output",
    version: "1.0.0",
    file_extension: ".dot",
    mime_type: "text/vnd.graphviz",
    parameters: [
        layout_engine - "GraphViz engine: dot, neato, fdp, circo, twopi (default dot)",
        rankdir - "Rank direction: TB, LR, BT, RL (default TB)",
        node_shape - "Node shape: ellipse, box, diamond, etc. (default ellipse)",
        use_positions - "Use provided positions with neato (default false)",
        include_urls - "Include URLs as hrefs (default true)",
        font_name - "Font name (default Helvetica)",
        font_size - "Font size (default 12)"
    ]
}).

validate_config(Config) :-
    is_list(Config),
    (   member(rankdir(R), Config)
    ->  member(R, ['TB', 'LR', 'BT', 'RL'])
    ;   true
    ).

init_component(_Name, _Config).

compile_component(_Name, _Config, _Options, '/* GraphViz DOT template */').

render(render_data(Nodes, Edges, Positions, _Styles), Options, Output) :-
    render_graphviz(Nodes, Edges, Positions, Options, Output).

% ============================================================================
% DOT RENDERING
% ============================================================================

%% render_graphviz(+Nodes, +Edges, +Positions, +Options, -DOT)
%
%  Render mind map to GraphViz DOT format with positions.
%
render_graphviz(Nodes, Edges, Positions, Options, DOT) :-
    option_or_default(layout_engine, Options, dot, _Engine),
    option_or_default(rankdir, Options, 'TB', RankDir),
    option_or_default(node_shape, Options, ellipse, NodeShape),
    option_or_default(use_positions, Options, false, UsePositions),
    option_or_default(font_name, Options, 'Helvetica', FontName),
    option_or_default(font_size, Options, 12, FontSize),
    option_or_default(title, Options, 'Mind Map', Title),

    % Build position lookup
    build_position_lookup(Positions, PosLookup),

    % Graph attributes
    (   UsePositions == true
    ->  EngineHint = '  // Use with: neato -n\n'
    ;   EngineHint = ''
    ),

    % Generate node definitions
    render_nodes_dot(Nodes, PosLookup, NodeShape, FontName, FontSize, UsePositions, Options, NodesContent),

    % Generate edge definitions
    render_edges_dot(Edges, Options, EdgesContent),

    % Assemble DOT document
    format(atom(DOT),
'// ~w
// Generated by UnifyWeaver mindmap_render_graphviz
~wdigraph mindmap {
  // Graph attributes
  graph [
    rankdir=~w
    fontname="~w"
    fontsize=~w
    label="~w"
    labelloc=t
  ];

  // Default node attributes
  node [
    shape=~w
    fontname="~w"
    fontsize=~w
    style=filled
    fillcolor="#e8f4fc"
  ];

  // Default edge attributes
  edge [
    fontname="~w"
    fontsize=~w
  ];

  // Nodes
~w
  // Edges
~w}
',
        [Title, EngineHint, RankDir, FontName, FontSize, Title,
         NodeShape, FontName, FontSize, FontName, FontSize,
         NodesContent, EdgesContent]).

%% render_graphviz(+Nodes, +Edges, +Options, -DOT)
%
%  Render without positions (let GraphViz layout).
%
render_graphviz(Nodes, Edges, Options, DOT) :-
    render_graphviz(Nodes, Edges, [], Options, DOT).

% ============================================================================
% CLUSTERED RENDERING
% ============================================================================

%% render_graphviz_clustered(+Nodes, +Edges, +Clusters, +Positions, +Options, -DOT)
%
%  Render mind map with cluster (subgraph) support.
%
%  Clusters format: [cluster(Id, NodeIds, Options), ...]
%  If Clusters is [], will auto-detect from node cluster() properties.
%
render_graphviz_clustered(Nodes, Edges, [], Positions, Options, DOT) :-
    !,
    % Auto-detect clusters from node properties
    cluster_from_node_props(Nodes, DetectedClusters),
    render_graphviz_clustered(Nodes, Edges, DetectedClusters, Positions, Options, DOT).

render_graphviz_clustered(Nodes, Edges, Clusters, Positions, Options, DOT) :-
    option_or_default(layout_engine, Options, dot, _Engine),
    option_or_default(rankdir, Options, 'TB', RankDir),
    option_or_default(node_shape, Options, ellipse, NodeShape),
    option_or_default(use_positions, Options, false, UsePositions),
    option_or_default(font_name, Options, 'Helvetica', FontName),
    option_or_default(font_size, Options, 12, FontSize),
    option_or_default(title, Options, 'Mind Map', Title),

    % Build position lookup
    build_position_lookup(Positions, PosLookup),

    % Collect all clustered node IDs
    findall(NodeId, (member(cluster(_, NIds, _), Clusters), member(NodeId, NIds)), ClusteredIds),

    % Separate clustered and unclustered nodes
    partition_nodes(Nodes, ClusteredIds, ClusteredNodes, UnclusteredNodes),

    % Generate subgraph/cluster content
    render_clusters_dot(Clusters, ClusteredNodes, PosLookup, NodeShape, FontName, FontSize, UsePositions, Options, ClustersContent),

    % Generate unclustered node definitions
    render_nodes_dot(UnclusteredNodes, PosLookup, NodeShape, FontName, FontSize, UsePositions, Options, UnclusteredContent),

    % Generate edge definitions
    render_edges_dot(Edges, Options, EdgesContent),

    % Engine hint
    (   UsePositions == true
    ->  EngineHint = '  // Use with: neato -n\n'
    ;   EngineHint = ''
    ),

    % Assemble DOT document
    format(atom(DOT),
'// ~w (with clusters)
// Generated by UnifyWeaver mindmap_render_graphviz
~wdigraph mindmap {
  // Graph attributes
  graph [
    rankdir=~w
    fontname="~w"
    fontsize=~w
    label="~w"
    labelloc=t
    compound=true
  ];

  // Default node attributes
  node [
    shape=~w
    fontname="~w"
    fontsize=~w
    style=filled
    fillcolor="#e8f4fc"
  ];

  // Default edge attributes
  edge [
    fontname="~w"
    fontsize=~w
  ];

  // Clusters (subgraphs)
~w
  // Unclustered nodes
~w
  // Edges
~w}
',
        [Title, EngineHint, RankDir, FontName, FontSize, Title,
         NodeShape, FontName, FontSize, FontName, FontSize,
         ClustersContent, UnclusteredContent, EdgesContent]).

%% partition_nodes(+Nodes, +ClusteredIds, -ClusteredNodes, -UnclusteredNodes)
partition_nodes([], _, [], []).
partition_nodes([node(Id, Props) | Rest], ClusteredIds, Clustered, Unclustered) :-
    partition_nodes(Rest, ClusteredIds, RestClustered, RestUnclustered),
    (   member(Id, ClusteredIds)
    ->  Clustered = [node(Id, Props) | RestClustered],
        Unclustered = RestUnclustered
    ;   Clustered = RestClustered,
        Unclustered = [node(Id, Props) | RestUnclustered]
    ).

%% render_clusters_dot(+Clusters, +AllNodes, +PosLookup, ..., -Content)
render_clusters_dot([], _, _, _, _, _, _, _, '').
render_clusters_dot([cluster(ClusterId, NodeIds, ClusterOpts) | Rest], AllNodes, PosLookup, Shape, FontName, FontSize, UsePos, Options, Content) :-
    % Get cluster nodes
    findall(node(Id, Props), (member(node(Id, Props), AllNodes), member(Id, NodeIds)), ClusterNodes),

    % Cluster styling
    (member(label(ClusterLabel), ClusterOpts) -> true ; atom_string(ClusterId, ClusterLabel)),
    (member(style(ClusterStyle), ClusterOpts) -> true ; ClusterStyle = solid),
    (member(color(ClusterColor), ClusterOpts) -> true ; ClusterColor = '#666666'),
    (member(bgcolor(ClusterBG), ClusterOpts) -> true ; ClusterBG = '#f5f5f5'),
    (member(fontcolor(ClusterFontColor), ClusterOpts) -> true ; ClusterFontColor = '#333333'),
    (member(rank(ClusterRank), ClusterOpts) -> RankAttr = format(atom(R), '    rank=~w;~n', [ClusterRank]), R ; RankAttr = ''),

    escape_dot_string(ClusterLabel, EscClusterLabel),

    % Render nodes within cluster
    render_nodes_dot(ClusterNodes, PosLookup, Shape, FontName, FontSize, UsePos, Options, ClusterNodesContent),

    % Format cluster subgraph
    format(atom(ClusterDOT),
'  subgraph cluster_~w {
    label="~w";
    style=~w;
    color="~w";
    bgcolor="~w";
    fontcolor="~w";
~w
~w  }

',
        [ClusterId, EscClusterLabel, ClusterStyle, ClusterColor, ClusterBG, ClusterFontColor, RankAttr, ClusterNodesContent]),

    render_clusters_dot(Rest, AllNodes, PosLookup, Shape, FontName, FontSize, UsePos, Options, RestContent),
    atom_concat(ClusterDOT, RestContent, Content).

%% build_position_lookup(+Positions, -Lookup)
build_position_lookup(Positions, Lookup) :-
    findall(Id-pos(X, Y), member(position(Id, X, Y), Positions), Lookup).

get_position(Id, Lookup, X, Y) :-
    member(Id-pos(X, Y), Lookup),
    !.
get_position(_, _, none, none).

% ============================================================================
% NODE RENDERING
% ============================================================================

render_nodes_dot([], _, _, _, _, _, _, '').
render_nodes_dot([node(Id, Props) | Rest], PosLookup, Shape, FontName, FontSize, UsePos, Options, Content) :-
    % Get properties
    (member(label(Label), Props) -> true ; atom_string(Id, Label)),
    (member(type(NodeType), Props) -> true ; NodeType = default),
    (member(link(URL), Props) -> true ; URL = ''),

    % Escape label for DOT
    escape_dot_string(Label, EscapedLabel),

    % Node styling based on type
    node_dot_style(NodeType, FillColor, FontColor),

    % Position if using fixed layout
    get_position(Id, PosLookup, PosX, PosY),
    (   UsePos == true, PosX \== none
    ->  format(atom(PosAttr), ' pos="~2f,~2f!"', [PosX, PosY])
    ;   PosAttr = ''
    ),

    % URL attribute
    option_or_default(include_urls, Options, true, IncludeURLs),
    (   IncludeURLs == true, URL \== ''
    ->  escape_dot_string(URL, EscURL),
        format(atom(URLAttr), ' URL="~w" tooltip="~w"', [EscURL, EscURL])
    ;   URLAttr = ''
    ),

    % Format node
    format(atom(NodeDOT),
        '  "~w" [label="~w" fillcolor="~w" fontcolor="~w"~w~w];~n',
        [Id, EscapedLabel, FillColor, FontColor, PosAttr, URLAttr]),

    render_nodes_dot(Rest, PosLookup, Shape, FontName, FontSize, UsePos, Options, RestContent),
    atom_concat(NodeDOT, RestContent, Content).

%% node_dot_style(+Type, -FillColor, -FontColor)
node_dot_style(root, '#4a90d9', '#ffffff') :- !.
node_dot_style(hub, '#6ab04c', '#ffffff') :- !.
node_dot_style(branch, '#f0932b', '#ffffff') :- !.
node_dot_style(leaf, '#eb4d4b', '#ffffff') :- !.
node_dot_style(_, '#e8f4fc', '#333333').

% ============================================================================
% EDGE RENDERING
% ============================================================================

render_edges_dot([], _, '').
render_edges_dot([edge(From, To, Props) | Rest], Options, Content) :-
    % Edge styling
    (member(weight(W), Props) -> true ; W = 1),
    (member(style(Style), Props) -> true ; Style = solid),
    (member(label(Label), Props) -> EdgeLabel = Label ; EdgeLabel = ''),

    % Style mapping
    edge_dot_style(Style, DotStyle),

    % Format edge
    (   EdgeLabel \== ''
    ->  escape_dot_string(EdgeLabel, EscLabel),
        format(atom(LabelAttr), ' label="~w"', [EscLabel])
    ;   LabelAttr = ''
    ),

    format(atom(EdgeDOT),
        '  "~w" -> "~w" [style=~w weight=~w~w];~n',
        [From, To, DotStyle, W, LabelAttr]),

    render_edges_dot(Rest, Options, RestContent),
    atom_concat(EdgeDOT, RestContent, Content).

%% edge_dot_style(+Style, -DotStyle)
edge_dot_style(solid, solid) :- !.
edge_dot_style(dashed, dashed) :- !.
edge_dot_style(dotted, dotted) :- !.
edge_dot_style(bold, bold) :- !.
edge_dot_style(_, solid).

% ============================================================================
% UTILITIES
% ============================================================================

option_or_default(Key, Options, Default, Value) :-
    Term =.. [Key, Value],
    (   member(Term, Options)
    ->  true
    ;   Value = Default
    ).

%% escape_dot_string(+Text, -Escaped)
%
%  Escape special characters for DOT strings.
%
escape_dot_string(Text, Escaped) :-
    atom_string(Text, Str),
    escape_dot_chars(Str, EscStr),
    atom_string(Escaped, EscStr).

escape_dot_chars([], []).
escape_dot_chars([C | Rest], Escaped) :-
    escape_dot_char(C, EscC),
    escape_dot_chars(Rest, RestEsc),
    append(EscC, RestEsc, Escaped).

escape_dot_char(0'", "\\\"") :- !.
escape_dot_char(0'\\, "\\\\") :- !.
escape_dot_char(0'\n, "\\n") :- !.
escape_dot_char(C, [C]).

% ============================================================================
% TESTING
% ============================================================================

test_graphviz_renderer :-
    format('~n=== GraphViz Renderer Tests ===~n~n'),

    Nodes = [
        node(root, [label("Central Topic"), type(root)]),
        node(a, [label("Branch A"), type(branch)]),
        node(b, [label("Branch B"), type(branch), link("https://example.com")]),
        node(c, [label("Leaf C"), type(leaf)])
    ],
    Edges = [
        edge(root, a, []),
        edge(root, b, []),
        edge(a, c, [])
    ],
    Positions = [
        position(root, 500, 400),
        position(a, 350, 550),
        position(b, 650, 550),
        position(c, 300, 700)
    ],

    % Test 1: Basic rendering
    format('Test 1: Basic DOT rendering...~n'),
    render_graphviz(Nodes, Edges, [], DOT),
    (   sub_atom(DOT, _, _, _, 'digraph')
    ->  format('  PASS: DOT document generated~n')
    ;   format('  FAIL: Invalid DOT~n')
    ),

    % Test 2: Contains nodes
    format('~nTest 2: Contains node definitions...~n'),
    (   sub_atom(DOT, _, _, _, '"root"')
    ->  format('  PASS: Nodes defined~n')
    ;   format('  FAIL: Nodes missing~n')
    ),

    % Test 3: Contains edges
    format('~nTest 3: Contains edge definitions...~n'),
    (   sub_atom(DOT, _, _, _, '->')
    ->  format('  PASS: Edges defined~n')
    ;   format('  FAIL: Edges missing~n')
    ),

    % Test 4: With positions
    format('~nTest 4: Rendering with positions...~n'),
    render_graphviz(Nodes, Edges, Positions, [use_positions(true)], DOTPos),
    (   sub_atom(DOTPos, _, _, _, 'pos=')
    ->  format('  PASS: Positions included~n')
    ;   format('  FAIL: Positions missing~n')
    ),

    % Test 5: URL inclusion
    format('~nTest 5: URL inclusion...~n'),
    (   sub_atom(DOT, _, _, _, 'URL=')
    ->  format('  PASS: URLs included~n')
    ;   format('  FAIL: URLs missing~n')
    ),

    format('~n=== Tests Complete ===~n').

%% test_cluster_support
%
%  Test subgraph/cluster functionality.
%
test_cluster_support :-
    format('~n=== GraphViz Cluster Support Tests ===~n~n'),

    % Test data with clusters
    NodesWithClusters = [
        node(root, [label("Central"), type(root)]),
        node(a1, [label("A1"), type(branch), cluster(group_a)]),
        node(a2, [label("A2"), type(leaf), cluster(group_a)]),
        node(b1, [label("B1"), type(branch), cluster(group_b)]),
        node(b2, [label("B2"), type(leaf), cluster(group_b)]),
        node(standalone, [label("Standalone"), type(leaf)])
    ],
    Edges = [
        edge(root, a1, []),
        edge(root, b1, []),
        edge(a1, a2, []),
        edge(b1, b2, []),
        edge(root, standalone, [])
    ],

    % Test 1: Auto-detect clusters from node properties
    format('Test 1: Auto-detect clusters from node properties...~n'),
    cluster_from_node_props(NodesWithClusters, DetectedClusters),
    (   length(DetectedClusters, 2),
        member(cluster(group_a, _, _), DetectedClusters),
        member(cluster(group_b, _, _), DetectedClusters)
    ->  format('  PASS: Detected 2 clusters (group_a, group_b)~n')
    ;   format('  FAIL: Cluster detection failed~n')
    ),

    % Test 2: Render with auto-detected clusters
    format('~nTest 2: Render with auto-detected clusters...~n'),
    render_graphviz_clustered(NodesWithClusters, Edges, [], [], [], ClusteredDOT1),
    (   sub_atom(ClusteredDOT1, _, _, _, 'subgraph cluster_group_a'),
        sub_atom(ClusteredDOT1, _, _, _, 'subgraph cluster_group_b')
    ->  format('  PASS: Cluster subgraphs generated~n')
    ;   format('  FAIL: Cluster subgraphs missing~n')
    ),

    % Test 3: Manually defined clusters with options
    format('~nTest 3: Manually defined clusters with styling...~n'),
    ManualClusters = [
        cluster(features, [a1, a2], [
            label("Features"),
            color('#4a90d9'),
            bgcolor('#e8f4fc'),
            style(rounded)
        ]),
        cluster(tasks, [b1, b2], [
            label("Tasks"),
            color('#6ab04c'),
            bgcolor('#f0fff0'),
            style(dashed)
        ])
    ],
    render_graphviz_clustered(NodesWithClusters, Edges, ManualClusters, [], [], ClusteredDOT2),
    (   sub_atom(ClusteredDOT2, _, _, _, 'label="Features"'),
        sub_atom(ClusteredDOT2, _, _, _, 'bgcolor="#e8f4fc"')
    ->  format('  PASS: Cluster styling applied~n')
    ;   format('  FAIL: Cluster styling missing~n')
    ),

    % Test 4: Unclustered nodes remain outside subgraphs
    format('~nTest 4: Unclustered nodes handled correctly...~n'),
    (   sub_atom(ClusteredDOT1, _, _, _, '"standalone"'),
        sub_atom(ClusteredDOT1, _, _, _, '"root"')
    ->  format('  PASS: Unclustered nodes rendered~n')
    ;   format('  FAIL: Unclustered nodes missing~n')
    ),

    % Test 5: Define and retrieve cluster via API
    format('~nTest 5: Cluster API define/get...~n'),
    define_cluster(test_cluster, [x, y, z], [label("Test"), color(red)]),
    (   get_cluster_nodes(test_cluster, TestNodes),
        TestNodes = [x, y, z]
    ->  format('  PASS: Cluster API works~n')
    ;   format('  FAIL: Cluster API failed~n')
    ),

    % Test 6: compound=true attribute for edge routing
    format('~nTest 6: Compound graph attribute...~n'),
    (   sub_atom(ClusteredDOT1, _, _, _, 'compound=true')
    ->  format('  PASS: Compound attribute set~n')
    ;   format('  FAIL: Compound attribute missing~n')
    ),

    % Cleanup
    clear_clusters,

    format('~n=== Cluster Tests Complete ===~n').

:- initialization((
    format('GraphViz renderer module loaded~n', [])
), now).
