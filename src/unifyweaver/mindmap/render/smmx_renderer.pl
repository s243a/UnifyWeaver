% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% smmx_renderer.pl - SimpleMind (.smmx) Renderer for Mind Maps
%
% Generates SimpleMind XML format. The .smmx format is a ZIP archive
% containing document/mindmap.xml.
%
% This module generates the XML content. For complete .smmx files,
% the XML should be placed in a ZIP archive with the path:
%   document/mindmap.xml
%
% Usage:
%   ?- render_smmx_xml(Nodes, Edges, Positions, Options, XML).

:- module(mindmap_render_smmx, [
    % Component interface
    type_info/1,
    validate_config/1,
    init_component/2,
    compile_component/4,
    render/3,

    % Direct API
    render_smmx_xml/5,              % render_smmx_xml(+Nodes, +Edges, +Positions, +Options, -XML)
    render_smmx_xml/4,              % render_smmx_xml(+Nodes, +Edges, +Options, -XML)

    % Testing
    test_smmx_renderer/0
]).

:- use_module(library(lists)).

% ============================================================================
% COMPONENT INTERFACE
% ============================================================================

type_info(info{
    name: smmx,
    category: mindmap_renderer,
    description: "SimpleMind .smmx format (XML content)",
    version: "1.0.0",
    file_extension: ".smmx",
    mime_type: "application/zip",
    note: "Output is XML content for document/mindmap.xml inside .smmx ZIP",
    parameters: [
        title - "Mind map title (default 'Mind Map')",
        palette - "Color palette 1-8 (default 1)",
        borderstyle - "Border style: ellipse, half_round, rectangle, diamond",
        include_positions - "Include computed positions (default true)"
    ]
}).

validate_config(Config) :-
    is_list(Config).

init_component(_Name, _Config).

compile_component(_Name, _Config, _Options, '<!-- SimpleMind template -->').

render(render_data(Nodes, Edges, Positions, _Styles), Options, Output) :-
    render_smmx_xml(Nodes, Edges, Positions, Options, Output).

% ============================================================================
% SMMX XML RENDERING
% ============================================================================

%% render_smmx_xml(+Nodes, +Edges, +Positions, +Options, -XML)
%
%  Render mind map to SimpleMind mindmap.xml format.
%
render_smmx_xml(Nodes, Edges, Positions, Options, XML) :-
    option_or_default(title, Options, 'Mind Map', Title),
    option_or_default(palette, Options, 1, Palette),

    % Build position lookup
    build_position_lookup(Positions, PosLookup),

    % Find root node
    find_root_node(Nodes, Edges, RootId),

    % Assign numeric IDs (SimpleMind uses integers)
    assign_numeric_ids(Nodes, IdMap),

    % Generate topic elements
    render_topics_smmx(Nodes, Edges, PosLookup, IdMap, RootId, Palette, Options, TopicsContent),

    % Generate relation elements (edges)
    render_relations_smmx(Edges, IdMap, RelationsContent),

    % Escape title
    escape_xml_attr(Title, EscTitle),

    % Generate timestamp (milliseconds since epoch approximation)
    get_time(Now),
    Timestamp is floor(Now * 1000),

    % Assemble XML document
    format(atom(XML),
'<?xml version="1.0" encoding="utf-8"?>
<simplemind-mindmaps doc-version="3" generator="UnifyWeaver">
  <mindmap guid="~w" save-ts="~w">
    <meta>
      <title text="~w"/>
      <note/>
    </meta>
    <topics>
~w    </topics>
    <relations>
~w    </relations>
    <style name="default" style-theme="default"/>
  </mindmap>
</simplemind-mindmaps>
', [RootId, Timestamp, EscTitle, TopicsContent, RelationsContent]).

%% render_smmx_xml(+Nodes, +Edges, +Options, -XML)
render_smmx_xml(Nodes, Edges, Options, XML) :-
    render_smmx_xml(Nodes, Edges, [], Options, XML).

% ============================================================================
% TOPIC RENDERING
% ============================================================================

render_topics_smmx([], _, _, _, _, _, _, '').
render_topics_smmx([node(Id, Props) | Rest], Edges, PosLookup, IdMap, RootId, Palette, Options, Content) :-
    % Get numeric ID
    member(Id-NumId, IdMap),

    % Get parent ID (or -1 for root)
    (   member(edge(ParentId, Id, _), Edges),
        member(ParentId-ParentNumId, IdMap)
    ->  true
    ;   ParentNumId = -1
    ),

    % Get position
    (   get_position(Id, PosLookup, X, Y),
        X \== none
    ->  true
    ;   X = 0, Y = 0
    ),

    % Get label
    (member(label(Label), Props) -> true ; atom_string(Id, Label)),
    escape_xml_attr(Label, EscLabel),

    % Get URL if present
    (   member(link(URL), Props),
        URL \== ''
    ->  escape_xml_attr(URL, EscURL),
        format(atom(LinkContent), '~n      <link urllink="~w"/>', [EscURL])
    ;   LinkContent = ''
    ),

    % Node type -> borderstyle
    (member(type(NodeType), Props) -> true ; NodeType = default),
    node_borderstyle(NodeType, BorderStyle),

    % Palette based on depth or type
    (   Id == RootId
    ->  NodePalette = 1
    ;   node_palette(NodeType, NodePalette)
    ),

    % Format topic element
    format(atom(TopicXML),
'      <topic id="~w" parent="~w" x="~2f" y="~2f" text="~w" borderstyle="~w" palette="~w">~w
      </topic>~n',
        [NumId, ParentNumId, X, Y, EscLabel, BorderStyle, NodePalette, LinkContent]),

    render_topics_smmx(Rest, Edges, PosLookup, IdMap, RootId, Palette, Options, RestContent),
    atom_concat(TopicXML, RestContent, Content).

%% node_borderstyle(+Type, -BorderStyle)
node_borderstyle(root, 'sbsEllipse') :- !.
node_borderstyle(hub, 'sbsRectangle') :- !.
node_borderstyle(branch, 'sbsHalfRound') :- !.
node_borderstyle(leaf, 'sbsDiamond') :- !.
node_borderstyle(_, 'sbsHalfRound').

%% node_palette(+Type, -Palette)
node_palette(root, 1) :- !.
node_palette(hub, 2) :- !.
node_palette(branch, 3) :- !.
node_palette(leaf, 4) :- !.
node_palette(_, 1).

% ============================================================================
% RELATION RENDERING
% ============================================================================

render_relations_smmx([], _, '').
render_relations_smmx([edge(From, To, Props) | Rest], IdMap, Content) :-
    member(From-FromId, IdMap),
    member(To-ToId, IdMap),

    % Get edge label if present
    (   member(label(Label), Props),
        Label \== ''
    ->  escape_xml_attr(Label, EscLabel),
        format(atom(LabelAttr), ' text="~w"', [EscLabel])
    ;   LabelAttr = ''
    ),

    format(atom(RelXML), '      <relation id1="~w" id2="~w"~w/>~n',
           [FromId, ToId, LabelAttr]),

    render_relations_smmx(Rest, IdMap, RestContent),
    atom_concat(RelXML, RestContent, Content).

% ============================================================================
% UTILITIES
% ============================================================================

find_root_node(Nodes, Edges, RootId) :-
    (   member(node(RootId, Props), Nodes),
        member(type(root), Props)
    ->  true
    ;   member(node(RootId, _), Nodes),
        \+ member(edge(_, RootId, _), Edges)
    ->  true
    ;   Nodes = [node(RootId, _) | _]
    ).

assign_numeric_ids(Nodes, IdMap) :-
    assign_ids_helper(Nodes, 0, IdMap).

assign_ids_helper([], _, []).
assign_ids_helper([node(Id, _) | Rest], N, [Id-N | RestMap]) :-
    N1 is N + 1,
    assign_ids_helper(Rest, N1, RestMap).

build_position_lookup(Positions, Lookup) :-
    findall(Id-pos(X, Y), member(position(Id, X, Y), Positions), Lookup).

get_position(Id, Lookup, X, Y) :-
    member(Id-pos(X, Y), Lookup),
    !.
get_position(_, _, none, none).

option_or_default(Key, Options, Default, Value) :-
    Term =.. [Key, Value],
    (   member(Term, Options)
    ->  true
    ;   Value = Default
    ).

%% escape_xml_attr(+Text, -Escaped)
escape_xml_attr(Text, Escaped) :-
    atom_string(Text, Str),
    escape_xml_chars(Str, EscStr),
    atom_string(Escaped, EscStr).

escape_xml_chars([], []).
escape_xml_chars([C | Rest], Escaped) :-
    escape_xml_char(C, EscC),
    escape_xml_chars(Rest, RestEsc),
    append(EscC, RestEsc, Escaped).

escape_xml_char(0'<, "&lt;") :- !.
escape_xml_char(0'>, "&gt;") :- !.
escape_xml_char(0'&, "&amp;") :- !.
escape_xml_char(0'", "&quot;") :- !.
escape_xml_char(0'', "&apos;") :- !.
escape_xml_char(C, [C]).

% ============================================================================
% TESTING
% ============================================================================

test_smmx_renderer :-
    format('~n=== SimpleMind Renderer Tests ===~n~n'),

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
    format('Test 1: Basic SMMX XML rendering...~n'),
    render_smmx_xml(Nodes, Edges, Positions, [title('Test Map')], XML),
    (   sub_atom(XML, _, _, _, '<simplemind-mindmaps')
    ->  format('  PASS: SMMX XML generated~n')
    ;   format('  FAIL: Invalid SMMX~n')
    ),

    % Test 2: Topics present
    format('~nTest 2: Topic elements...~n'),
    (   sub_atom(XML, _, _, _, '<topic id=')
    ->  format('  PASS: Topics generated~n')
    ;   format('  FAIL: Topics missing~n')
    ),

    % Test 3: Positions included
    format('~nTest 3: Position attributes...~n'),
    (   sub_atom(XML, _, _, _, 'x="500')
    ->  format('  PASS: Positions included~n')
    ;   format('  FAIL: Positions missing~n')
    ),

    % Test 4: URLs as links
    format('~nTest 4: URL links...~n'),
    (   sub_atom(XML, _, _, _, 'urllink=')
    ->  format('  PASS: URLs included~n')
    ;   format('  FAIL: URLs missing~n')
    ),

    % Test 5: Title in meta
    format('~nTest 5: Title in metadata...~n'),
    (   sub_atom(XML, _, _, _, 'text="Test Map"')
    ->  format('  PASS: Title set~n')
    ;   format('  FAIL: Title missing~n')
    ),

    format('~n=== Tests Complete ===~n').

:- initialization((
    format('SimpleMind (.smmx) renderer module loaded~n', [])
), now).
