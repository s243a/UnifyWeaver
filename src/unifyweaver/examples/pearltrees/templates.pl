%% pearltrees/templates.pl - Multi-format mindmap template generation
%%
%% Phase 3: Template-based mindmap generation.
%% Direct output to multiple formats from tree data.
%%
%% Supported formats:
%% - SMMX (SimpleMind)
%% - FreeMind (.mm)
%% - OPML (outline format)
%% - GraphML (yEd, Gephi)
%% - VUE (Tufts VUE)
%% - Mermaid (text diagrams)

:- module(pearltrees_templates, [
    % SMMX format
    generate_mindmap/4,
    generate_mindmap_xml/5,
    child_to_xml/2,
    escape_xml/2,

    % FreeMind format
    generate_freemind/4,
    child_to_freemind/3,

    % OPML format
    generate_opml/4,
    child_to_opml/3,

    % GraphML format
    generate_graphml/4,
    child_to_graphml_node/3,
    child_to_graphml_edge/4,

    % VUE format
    generate_vue/4,

    % Mermaid format
    generate_mermaid/4,
    child_to_mermaid/3,

    % Unified multi-format generation
    generate_all_formats/5,
    available_formats/1
]).

%% --------------------------------------------------------------------
%% XML Generation
%%
%% SMMX format is XML inside a ZIP archive.
%% This module generates the XML content.
%% --------------------------------------------------------------------

%% generate_mindmap(+TreeId, +Title, +Children, -XML) is det.
%%   Generate complete SMMX XML for a tree.
generate_mindmap(TreeId, Title, Children, XML) :-
    escape_xml(Title, EscTitle),
    maplist(child_to_xml, Children, ChildXMLs),
    atomic_list_concat(ChildXMLs, '\n      ', ChildrenStr),
    format(atom(XML), '<?xml version="1.0" encoding="UTF-8"?>
<smmx version="2">
  <mindmap>
    <topic text="~w" id="root_~w" color="#4A90D9">
      ~w
    </topic>
  </mindmap>
</smmx>', [EscTitle, TreeId, ChildrenStr]).

%% generate_mindmap_xml(+TreeId, +Title, +Uri, +Children, -XML) is det.
%%   Generate SMMX XML with URI link on root.
generate_mindmap_xml(TreeId, Title, Uri, Children, XML) :-
    escape_xml(Title, EscTitle),
    escape_xml(Uri, EscUri),
    maplist(child_to_xml, Children, ChildXMLs),
    atomic_list_concat(ChildXMLs, '\n      ', ChildrenStr),
    format(atom(XML), '<?xml version="1.0" encoding="UTF-8"?>
<smmx version="2">
  <mindmap>
    <topic text="~w" id="root_~w" color="#4A90D9">
      <link url="~w"/>
      ~w
    </topic>
  </mindmap>
</smmx>', [EscTitle, TreeId, EscUri, ChildrenStr]).

%% child_to_xml(+Child, -XML) is det.
%%   Convert a child term to XML topic element.
child_to_xml(child(pagepearl, Title, Url, Order), XML) :-
    !,
    escape_xml(Title, EscTitle),
    escape_xml(Url, EscUrl),
    format(atom(Id), 'pearl_~w', [Order]),
    (   Url \= null, Url \= ''
    ->  format(atom(XML), '<topic text="~w" id="~w" color="#6DB33F">
        <link url="~w"/>
      </topic>', [EscTitle, Id, EscUrl])
    ;   format(atom(XML), '<topic text="~w" id="~w" color="#6DB33F"/>', [EscTitle, Id])
    ).

child_to_xml(child(tree, Title, _, Order), XML) :-
    !,
    escape_xml(Title, EscTitle),
    format(atom(Id), 'tree_~w', [Order]),
    format(atom(XML), '<topic text="~w" id="~w" color="#F5A623"/>', [EscTitle, Id]).

child_to_xml(child(alias, Title, _, Order), XML) :-
    !,
    escape_xml(Title, EscTitle),
    format(atom(Id), 'alias_~w', [Order]),
    format(atom(XML), '<topic text="~w" id="~w" color="#9B59B6"/>', [EscTitle, Id]).

child_to_xml(child(section, Title, _, Order), XML) :-
    !,
    escape_xml(Title, EscTitle),
    format(atom(Id), 'section_~w', [Order]),
    format(atom(XML), '<topic text="~w" id="~w" color="#7F8C8D" style="bold"/>', [EscTitle, Id]).

child_to_xml(child(root, Title, _, _), XML) :-
    !,
    escape_xml(Title, EscTitle),
    format(atom(XML), '<!-- root: ~w -->', [EscTitle]).

child_to_xml(child(Type, Title, _, Order), XML) :-
    % Fallback for unknown types
    escape_xml(Title, EscTitle),
    format(atom(Id), '~w_~w', [Type, Order]),
    format(atom(XML), '<topic text="~w" id="~w"/>', [EscTitle, Id]).

%% escape_xml(+Text, -Escaped) is det.
%%   Escape special XML characters.
escape_xml(Text, Escaped) :-
    (   var(Text) ; Text == null
    ->  Escaped = ''
    ;   atom_string(Text, Str),
        escape_xml_chars(Str, EscStr),
        atom_string(Escaped, EscStr)
    ).

escape_xml_chars(Str, Escaped) :-
    string_codes(Str, Codes),
    maplist(escape_xml_code, Codes, EscCodeLists),
    append(EscCodeLists, EscCodes),
    string_codes(Escaped, EscCodes).

escape_xml_code(0'<, Codes) :- !, string_codes("&lt;", Codes).
escape_xml_code(0'>, Codes) :- !, string_codes("&gt;", Codes).
escape_xml_code(0'&, Codes) :- !, string_codes("&amp;", Codes).
escape_xml_code(0'", Codes) :- !, string_codes("&quot;", Codes).
escape_xml_code(0'', Codes) :- !, string_codes("&apos;", Codes).
escape_xml_code(C, [C]).

%% --------------------------------------------------------------------
%% Color scheme (matches existing Python generator)
%% --------------------------------------------------------------------
%%
%% - Root:      #4A90D9 (blue)
%% - PagePearl: #6DB33F (green)
%% - Tree:      #F5A623 (orange)
%% - Alias:     #9B59B6 (purple)
%% - Section:   #7F8C8D (gray)

%% --------------------------------------------------------------------
%% Integration with queries
%% --------------------------------------------------------------------

%% generate_tree_mindmap(+TreeId, -XML) is det.
%%   Generate mindmap XML for a tree using queries.
%%   Requires sources to be loaded.
generate_tree_mindmap(_TreeId, _XML) :-
    % This would use the actual queries module
    % pearltrees_queries:tree_with_children(TreeId, Title, Children),
    % generate_mindmap(TreeId, Title, Children, XML).
    throw(error(not_implemented, 'Requires loaded sources')).

%% ====================================================================
%% FreeMind (.mm) Format
%% ====================================================================
%%
%% FreeMind/Freeplane format. Compatible with:
%% - FreeMind (Java desktop app)
%% - Freeplane (FreeMind fork with more features)
%% - Mind42 (web-based)
%% - XMind (import)

%% generate_freemind(+TreeId, +Title, +Children, -MM) is det.
%%   Generate FreeMind .mm XML for a tree.
generate_freemind(TreeId, Title, Children, MM) :-
    escape_xml(Title, EscTitle),
    format(atom(RootId), 'root_~w', [TreeId]),
    maplist(child_to_freemind(0), Children, ChildNodes),
    atomic_list_concat(ChildNodes, '\n', ChildrenStr),
    format(atom(MM), '<?xml version="1.0" encoding="UTF-8"?>
<map version="1.0.1">
<!-- Generated by UnifyWeaver -->
<node TEXT="~w" ID="~w" COLOR="#4A90D9">
~w
</node>
</map>', [EscTitle, RootId, ChildrenStr]).

%% child_to_freemind(+Depth, +Child, -NodeXML) is det.
%%   Convert a child term to FreeMind node element.
child_to_freemind(Depth, child(pagepearl, Title, Url, Order), XML) :-
    !,
    escape_xml(Title, EscTitle),
    escape_xml(Url, EscUrl),
    format(atom(Id), 'pearl_~w', [Order]),
    indent(Depth, Indent),
    (   Depth =:= 0
    ->  (   Order mod 2 =:= 0
        ->  PosAttr = ' POSITION="right"'
        ;   PosAttr = ' POSITION="left"'
        )
    ;   PosAttr = ''
    ),
    (   Url \= null, Url \= ''
    ->  format(atom(XML), '~w<node TEXT="~w" ID="~w" COLOR="#6DB33F"~w>
~w  <attribute NAME="url" VALUE="~w"/>
~w</node>', [Indent, EscTitle, Id, PosAttr, Indent, EscUrl, Indent])
    ;   format(atom(XML), '~w<node TEXT="~w" ID="~w" COLOR="#6DB33F"~w/>', [Indent, EscTitle, Id, PosAttr])
    ).

child_to_freemind(Depth, child(tree, Title, _, Order), XML) :-
    !,
    escape_xml(Title, EscTitle),
    format(atom(Id), 'tree_~w', [Order]),
    indent(Depth, Indent),
    (   Depth =:= 0
    ->  (   Order mod 2 =:= 0
        ->  PosAttr = ' POSITION="right"'
        ;   PosAttr = ' POSITION="left"'
        )
    ;   PosAttr = ''
    ),
    format(atom(XML), '~w<node TEXT="~w" ID="~w" COLOR="#F5A623"~w FOLDED="true"/>', [Indent, EscTitle, Id, PosAttr]).

child_to_freemind(Depth, child(alias, Title, _, Order), XML) :-
    !,
    escape_xml(Title, EscTitle),
    format(atom(Id), 'alias_~w', [Order]),
    indent(Depth, Indent),
    (   Depth =:= 0
    ->  (   Order mod 2 =:= 0
        ->  PosAttr = ' POSITION="right"'
        ;   PosAttr = ' POSITION="left"'
        )
    ;   PosAttr = ''
    ),
    format(atom(XML), '~w<node TEXT="~w" ID="~w" COLOR="#9B59B6"~w>
~w  <icon BUILTIN="bookmark"/>
~w</node>', [Indent, EscTitle, Id, PosAttr, Indent, Indent]).

child_to_freemind(Depth, child(section, Title, _, Order), XML) :-
    !,
    escape_xml(Title, EscTitle),
    format(atom(Id), 'section_~w', [Order]),
    indent(Depth, Indent),
    (   Depth =:= 0
    ->  (   Order mod 2 =:= 0
        ->  PosAttr = ' POSITION="right"'
        ;   PosAttr = ' POSITION="left"'
        )
    ;   PosAttr = ''
    ),
    format(atom(XML), '~w<node TEXT="~w" ID="~w" COLOR="#7F8C8D"~w STYLE="bold"/>', [Indent, EscTitle, Id, PosAttr]).

child_to_freemind(Depth, child(Type, Title, _, Order), XML) :-
    escape_xml(Title, EscTitle),
    format(atom(Id), '~w_~w', [Type, Order]),
    indent(Depth, Indent),
    (   Depth =:= 0
    ->  (   Order mod 2 =:= 0
        ->  PosAttr = ' POSITION="right"'
        ;   PosAttr = ' POSITION="left"'
        )
    ;   PosAttr = ''
    ),
    format(atom(XML), '~w<node TEXT="~w" ID="~w"~w/>', [Indent, EscTitle, Id, PosAttr]).

indent(Depth, Indent) :-
    Spaces is Depth * 2 + 2,
    length(SpaceList, Spaces),
    maplist(=(0' ), SpaceList),
    atom_codes(Indent, SpaceList).

%% ====================================================================
%% OPML Format
%% ====================================================================
%%
%% OPML (Outline Processor Markup Language). Compatible with:
%% - Workflowy
%% - Dynalist
%% - OmniOutliner
%% - RSS readers

%% generate_opml(+TreeId, +Title, +Children, -OPML) is det.
%%   Generate OPML XML for a tree.
generate_opml(_TreeId, Title, Children, OPML) :-
    escape_xml(Title, EscTitle),
    maplist(child_to_opml(2), Children, ChildOutlines),
    atomic_list_concat(ChildOutlines, '\n', ChildrenStr),
    format(atom(OPML), '<?xml version="1.0" encoding="UTF-8"?>
<opml version="2.0">
  <head>
    <title>~w</title>
    <expansionState>0</expansionState>
  </head>
  <body>
    <outline text="~w">
~w
    </outline>
  </body>
</opml>', [EscTitle, EscTitle, ChildrenStr]).

%% child_to_opml(+Indent, +Child, -OutlineXML) is det.
%%   Convert a child term to OPML outline element.
child_to_opml(Indent, child(pagepearl, Title, Url, _Order), XML) :-
    !,
    escape_xml(Title, EscTitle),
    escape_xml(Url, EscUrl),
    indent_spaces(Indent, Spaces),
    (   Url \= null, Url \= ''
    ->  format(atom(XML), '~w<outline text="~w" url="~w" type="link"/>', [Spaces, EscTitle, EscUrl])
    ;   format(atom(XML), '~w<outline text="~w"/>', [Spaces, EscTitle])
    ).

child_to_opml(Indent, child(tree, Title, _, _Order), XML) :-
    !,
    escape_xml(Title, EscTitle),
    indent_spaces(Indent, Spaces),
    format(atom(XML), '~w<outline text="[Folder] ~w"/>', [Spaces, EscTitle]).

child_to_opml(Indent, child(alias, Title, _, _Order), XML) :-
    !,
    escape_xml(Title, EscTitle),
    indent_spaces(Indent, Spaces),
    format(atom(XML), '~w<outline text="[Alias] ~w"/>', [Spaces, EscTitle]).

child_to_opml(Indent, child(section, Title, _, _Order), XML) :-
    !,
    escape_xml(Title, EscTitle),
    indent_spaces(Indent, Spaces),
    format(atom(XML), '~w<outline text="--- ~w ---"/>', [Spaces, EscTitle]).

child_to_opml(Indent, child(_Type, Title, _, _Order), XML) :-
    escape_xml(Title, EscTitle),
    indent_spaces(Indent, Spaces),
    format(atom(XML), '~w<outline text="~w"/>', [Spaces, EscTitle]).

indent_spaces(N, Spaces) :-
    length(SpaceList, N),
    maplist(=(0' ), SpaceList),
    atom_codes(Spaces, SpaceList).

%% ====================================================================
%% GraphML Format
%% ====================================================================
%%
%% GraphML with yEd extensions. Compatible with:
%% - yEd (free graph editor)
%% - Gephi (network analysis)
%% - Cytoscape (bioinformatics)
%% - NetworkX (Python)

%% generate_graphml(+TreeId, +Title, +Children, -GraphML) is det.
%%   Generate GraphML XML for a tree.
generate_graphml(TreeId, Title, Children, GraphML) :-
    escape_xml(Title, EscTitle),
    format(atom(RootId), 'n_root_~w', [TreeId]),
    % Generate root node
    format(atom(RootNode), '    <node id="~w">
      <data key="d0">
        <y:ShapeNode>
          <y:Fill color="#4A90D9" transparent="false"/>
          <y:BorderStyle type="line" width="2.0" color="#2C5F8C"/>
          <y:NodeLabel>~w</y:NodeLabel>
          <y:Shape type="roundrectangle"/>
        </y:ShapeNode>
      </data>
    </node>', [RootId, EscTitle]),
    % Generate child nodes
    maplist(child_to_graphml_node(TreeId), Children, ChildNodes),
    % Generate edges from root to children
    maplist(child_to_graphml_edge(TreeId, RootId), Children, ChildEdges),
    atomic_list_concat([RootNode|ChildNodes], '\n', NodesStr),
    atomic_list_concat(ChildEdges, '\n', EdgesStr),
    format(atom(GraphML), '<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:y="http://www.yworks.com/xml/graphml"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
           http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd">
  <key for="node" id="d0" yfiles.type="nodegraphics"/>
  <key for="edge" id="d1" yfiles.type="edgegraphics"/>
  <key id="url" for="node" attr.name="url" attr.type="string"/>
  <graph id="~w" edgedefault="directed">
~w
~w
  </graph>
</graphml>', [EscTitle, NodesStr, EdgesStr]).

%% child_to_graphml_node(+TreeId, +Child, -NodeXML) is det.
child_to_graphml_node(TreeId, child(Type, Title, Url, Order), XML) :-
    escape_xml(Title, EscTitle),
    format(atom(NodeId), 'n_~w_~w_~w', [TreeId, Type, Order]),
    type_color(Type, FillColor, BorderColor),
    (   Url \= null, Url \= ''
    ->  escape_xml(Url, EscUrl),
        format(atom(UrlData), '\n      <data key="url">~w</data>', [EscUrl])
    ;   UrlData = ''
    ),
    format(atom(XML), '    <node id="~w">~w
      <data key="d0">
        <y:ShapeNode>
          <y:Fill color="~w" transparent="false"/>
          <y:BorderStyle type="line" width="1.0" color="~w"/>
          <y:NodeLabel>~w</y:NodeLabel>
          <y:Shape type="roundrectangle"/>
        </y:ShapeNode>
      </data>
    </node>', [NodeId, UrlData, FillColor, BorderColor, EscTitle]).

%% child_to_graphml_edge(+TreeId, +ParentId, +Child, -EdgeXML) is det.
child_to_graphml_edge(TreeId, ParentId, child(Type, _, _, Order), XML) :-
    format(atom(NodeId), 'n_~w_~w_~w', [TreeId, Type, Order]),
    format(atom(EdgeId), 'e_~w_~w', [ParentId, NodeId]),
    format(atom(XML), '    <edge id="~w" source="~w" target="~w">
      <data key="d1">
        <y:PolyLineEdge>
          <y:LineStyle type="line" width="1.0" color="#666666"/>
          <y:Arrows source="none" target="standard"/>
        </y:PolyLineEdge>
      </data>
    </edge>', [EdgeId, ParentId, NodeId]).

type_color(pagepearl, '#6DB33F', '#4A7F2B').
type_color(tree, '#F5A623', '#C78519').
type_color(alias, '#9B59B6', '#7D4592').
type_color(section, '#7F8C8D', '#5D6566').
type_color(_, '#E8E8E8', '#AAAAAA').

%% ====================================================================
%% VUE Format
%% ====================================================================
%%
%% VUE (Visual Understanding Environment) from Tufts University.
%% Free concept mapping software.

%% generate_vue(+TreeId, +Title, +Children, -VUE) is det.
%%   Generate VUE XML for a tree.
generate_vue(TreeId, Title, Children, VUE) :-
    escape_xml(Title, EscTitle),
    get_time(Now),
    Timestamp is round(Now * 1000),
    % Root node
    format(atom(RootNode), '    <child ID="100" label="~w" layerID="1"
        created="~w" x="400.0" y="300.0"
        width="120.0" height="30.0" strokeWidth="2.0"
        autoSized="true" xsi:type="node">
        <fillColor>#4A90D9</fillColor>
        <strokeColor>#2C5F8C</strokeColor>
        <textColor>#FFFFFF</textColor>
        <font>Arial-bold-14</font>
        <shape arcwidth="20.0" archeight="20.0" xsi:type="roundRect"/>
    </child>', [EscTitle, Timestamp]),
    % Child nodes (positioned radially)
    length(Children, NumChildren),
    maplist_indexed(child_to_vue_node(TreeId, Timestamp, NumChildren), Children, ChildNodes),
    % Links from root to children
    maplist_indexed(child_to_vue_link(TreeId, Timestamp, NumChildren), Children, ChildLinks),
    atomic_list_concat([RootNode|ChildNodes], '\n', NodesStr),
    atomic_list_concat(ChildLinks, '\n', LinksStr),
    format(atom(VUE), '<?xml version="1.0" encoding="UTF-8"?>
<!-- Tufts VUE concept-map -->
<!-- Generated by UnifyWeaver -->
<LW-MAP xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:noNamespaceSchemaLocation="none" ID="0" label="~w"
    created="~w" x="0.0" y="0.0" width="1.4E-45"
    height="1.4E-45" strokeWidth="0.0" autoSized="false">
    <fillColor>#FFFFFF</fillColor>
    <strokeColor>#404040</strokeColor>
    <textColor>#000000</textColor>
    <font>SansSerif-plain-14</font>
~w
~w
    <layer ID="1" label="Layer 1" created="~w" x="0.0"
        y="0.0" width="1.4E-45" height="1.4E-45" strokeWidth="0.0" autoSized="false"/>
    <userZoom>1.0</userZoom>
    <userOrigin x="-14.0" y="-14.0"/>
    <presentationBackground>#202020</presentationBackground>
    <modelVersion>6</modelVersion>
</LW-MAP>', [EscTitle, Timestamp, NodesStr, LinksStr, Timestamp]).

%% child_to_vue_node(+TreeId, +Timestamp, +Total, +Index, +Child, -NodeXML) is det.
child_to_vue_node(_TreeId, Timestamp, Total, Index, child(Type, Title, Url, Order), XML) :-
    escape_xml(Title, EscTitle),
    NodeId is 100 + Order + 1,
    % Position nodes in a circle around the root
    Angle is 2 * pi * Index / max(Total, 1),
    Radius = 200,
    X is 400 + Radius * cos(Angle),
    Y is 300 + Radius * sin(Angle),
    type_color(Type, FillColor, StrokeColor),
    (   Url \= null, Url \= ''
    ->  escape_xml(Url, EscUrl),
        format(atom(Resource), '
        <resource referenceCreated="0"
            spec="~w"
            type="2" xsi:type="URLResource">
            <property key="URL" value="~w"/>
        </resource>', [EscUrl, EscUrl])
    ;   Resource = ''
    ),
    format(atom(XML), '    <child ID="~w" label="~w" layerID="1"
        created="~w" x="~1f" y="~1f"
        width="100.0" height="28.0" strokeWidth="1.0"
        autoSized="true" xsi:type="node">~w
        <fillColor>~w</fillColor>
        <strokeColor>~w</strokeColor>
        <textColor>#000000</textColor>
        <font>Arial-plain-12</font>
        <shape arcwidth="20.0" archeight="20.0" xsi:type="roundRect"/>
    </child>', [NodeId, EscTitle, Timestamp, X, Y, Resource, FillColor, StrokeColor]).

%% child_to_vue_link(+TreeId, +Timestamp, +Total, +Index, +Child, -LinkXML) is det.
child_to_vue_link(_TreeId, Timestamp, Total, Index, child(_Type, _Title, _, Order), XML) :-
    LinkId is 1000 + Order,
    NodeId is 100 + Order + 1,
    % Calculate link endpoints
    Angle is 2 * pi * Index / max(Total, 1),
    Radius = 200,
    X2 is 400 + Radius * cos(Angle),
    Y2 is 300 + Radius * sin(Angle),
    MidX is (400 + X2) / 2,
    MidY is (300 + Y2) / 2,
    format(atom(XML), '    <child ID="~w" layerID="1" created="~w"
        x="~1f" y="~1f"
        width="~1f" height="~1f" strokeWidth="1.0"
        autoSized="false" controlCount="0" arrowState="2" xsi:type="link">
        <strokeColor>#404040</strokeColor>
        <textColor>#404040</textColor>
        <font>Arial-plain-11</font>
        <point1 x="400.0" y="300.0"/>
        <point2 x="~1f" y="~1f"/>
        <ID1 xsi:type="node">100</ID1>
        <ID2 xsi:type="node">~w</ID2>
    </child>', [LinkId, Timestamp, MidX, MidY, Radius, Radius, X2, Y2, NodeId]).

%% maplist_indexed(+Goal, +List, -Results) is det.
%%   Like maplist but passes 0-based index to Goal.
maplist_indexed(Goal, List, Results) :-
    maplist_indexed_(Goal, List, 0, Results).

maplist_indexed_(_, [], _, []).
maplist_indexed_(Goal, [H|T], Index, [R|Rs]) :-
    call(Goal, Index, H, R),
    NextIndex is Index + 1,
    maplist_indexed_(Goal, T, NextIndex, Rs).

%% ====================================================================
%% Mermaid Format
%% ====================================================================
%%
%% Mermaid is a text-based diagramming language. Compatible with:
%% - GitHub Markdown
%% - GitLab Markdown
%% - Notion
%% - Obsidian
%% - Many documentation tools

%% generate_mermaid(+TreeId, +Title, +Children, -Mermaid) is det.
%%   Generate Mermaid mindmap diagram.
generate_mermaid(TreeId, Title, Children, Mermaid) :-
    escape_mermaid(Title, EscTitle),
    format(atom(RootId), 'root_~w', [TreeId]),
    maplist(child_to_mermaid(RootId), Children, ChildLines),
    atomic_list_concat(ChildLines, '\n', ChildrenStr),
    format(atom(Mermaid), '```mermaid
mindmap
  root((~w))
~w
```', [EscTitle, ChildrenStr]).

%% child_to_mermaid(+ParentId, +Child, -Line) is det.
%%   Convert a child term to Mermaid mindmap node.
child_to_mermaid(_ParentId, child(pagepearl, Title, Url, Order), Line) :-
    !,
    escape_mermaid(Title, EscTitle),
    format(atom(NodeId), 'pearl_~w', [Order]),
    (   Url \= null, Url \= ''
    ->  % Mermaid supports click events for links
        format(atom(Line), '    ~w[~w]', [NodeId, EscTitle])
    ;   format(atom(Line), '    ~w[~w]', [NodeId, EscTitle])
    ).

child_to_mermaid(_ParentId, child(tree, Title, _, Order), Line) :-
    !,
    escape_mermaid(Title, EscTitle),
    format(atom(NodeId), 'tree_~w', [Order]),
    format(atom(Line), '    ~w{{~w}}', [NodeId, EscTitle]).

child_to_mermaid(_ParentId, child(alias, Title, _, Order), Line) :-
    !,
    escape_mermaid(Title, EscTitle),
    format(atom(NodeId), 'alias_~w', [Order]),
    format(atom(Line), '    ~w>~w]', [NodeId, EscTitle]).

child_to_mermaid(_ParentId, child(section, Title, _, Order), Line) :-
    !,
    escape_mermaid(Title, EscTitle),
    format(atom(NodeId), 'section_~w', [Order]),
    format(atom(Line), '    ~w(~w)', [NodeId, EscTitle]).

child_to_mermaid(_ParentId, child(Type, Title, _, Order), Line) :-
    escape_mermaid(Title, EscTitle),
    format(atom(NodeId), '~w_~w', [Type, Order]),
    format(atom(Line), '    ~w[~w]', [NodeId, EscTitle]).

%% escape_mermaid(+Text, -Escaped) is det.
%%   Escape special Mermaid characters.
escape_mermaid(Text, Escaped) :-
    (   var(Text) ; Text == null
    ->  Escaped = ''
    ;   atom_string(Text, Str),
        % Escape quotes and brackets that could break Mermaid syntax
        string_replace(Str, "\"", "'", S1),
        string_replace(S1, "[", "(", S2),
        string_replace(S2, "]", ")", S3),
        string_replace(S3, "{", "(", S4),
        string_replace(S4, "}", ")", Escaped)
    ).

string_replace(Str, From, To, Result) :-
    (   sub_string(Str, Before, _, After, From)
    ->  sub_string(Str, 0, Before, _, Prefix),
        sub_string(Str, _, After, 0, Suffix),
        string_concat(Prefix, To, Temp),
        string_concat(Temp, Suffix, NewStr),
        string_replace(NewStr, From, To, Result)
    ;   Result = Str
    ).

%% ====================================================================
%% Unified Multi-Format Generation
%% ====================================================================
%%
%% Generate output in all available formats from a single tree.

%% available_formats(-Formats) is det.
%%   List of all supported output formats.
available_formats([smmx, freemind, opml, graphml, vue, mermaid]).

%% format_info(+Format, -Extension, -Description) is semidet.
%%   Metadata about each format.
format_info(smmx, '.smmx', 'SimpleMind').
format_info(freemind, '.mm', 'FreeMind/Freeplane').
format_info(opml, '.opml', 'OPML Outline').
format_info(graphml, '.graphml', 'GraphML (yEd)').
format_info(vue, '.vue', 'Tufts VUE').
format_info(mermaid, '.md', 'Mermaid Diagram').

%% generate_all_formats(+TreeId, +Title, +Children, +Formats, -Results) is det.
%%   Generate output for multiple formats at once.
%%   Results is a list of format(Name, Extension, Content) terms.
%%
%%   Example:
%%   ?- generate_all_formats('123', 'My Tree', Children, [freemind, mermaid], Results).
%%   Results = [format(freemind, '.mm', '...'), format(mermaid, '.md', '...')].
generate_all_formats(TreeId, Title, Children, Formats, Results) :-
    maplist(generate_single_format(TreeId, Title, Children), Formats, Results).

generate_single_format(TreeId, Title, Children, Format, format(Format, Ext, Content)) :-
    format_info(Format, Ext, _Desc),
    generate_format(Format, TreeId, Title, Children, Content).

%% generate_format(+Format, +TreeId, +Title, +Children, -Output) is semidet.
%%   Internal dispatcher to format-specific generators.
generate_format(smmx, TreeId, Title, Children, Output) :-
    generate_mindmap(TreeId, Title, Children, Output).
generate_format(freemind, TreeId, Title, Children, Output) :-
    generate_freemind(TreeId, Title, Children, Output).
generate_format(opml, TreeId, Title, Children, Output) :-
    generate_opml(TreeId, Title, Children, Output).
generate_format(graphml, TreeId, Title, Children, Output) :-
    generate_graphml(TreeId, Title, Children, Output).
generate_format(vue, TreeId, Title, Children, Output) :-
    generate_vue(TreeId, Title, Children, Output).
generate_format(mermaid, TreeId, Title, Children, Output) :-
    generate_mermaid(TreeId, Title, Children, Output).
