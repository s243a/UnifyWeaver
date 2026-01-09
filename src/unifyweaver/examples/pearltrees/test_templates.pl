%% pearltrees/test_templates.pl - Unit tests for template generation
%%
%% Run with: swipl -g "run_tests" -t halt test_templates.pl

:- module(test_pearltrees_templates, []).

:- use_module(library(plunit)).
:- use_module(templates).

%% --------------------------------------------------------------------
%% Tests
%% --------------------------------------------------------------------

:- begin_tests(escape_xml).

test(escapes_ampersand) :-
    escape_xml('Tom & Jerry', Escaped),
    Escaped == 'Tom &amp; Jerry'.

test(escapes_less_than) :-
    escape_xml('x < y', Escaped),
    Escaped == 'x &lt; y'.

test(escapes_greater_than) :-
    escape_xml('x > y', Escaped),
    Escaped == 'x &gt; y'.

test(escapes_quotes) :-
    escape_xml('Say "hello"', Escaped),
    Escaped == 'Say &quot;hello&quot;'.

test(handles_null) :-
    escape_xml(null, Escaped),
    Escaped == ''.

test(handles_plain_text) :-
    escape_xml('Hello World', Escaped),
    Escaped == 'Hello World'.

:- end_tests(escape_xml).

:- begin_tests(child_to_xml).

test(pagepearl_with_url) :-
    child_to_xml(child(pagepearl, 'GitHub', 'https://github.com', 1), XML),
    sub_atom(XML, _, _, _, 'text="GitHub"'),
    sub_atom(XML, _, _, _, 'url="https://github.com"'),
    sub_atom(XML, _, _, _, 'color="#6DB33F"').

test(pagepearl_without_url) :-
    child_to_xml(child(pagepearl, 'Note', null, 2), XML),
    sub_atom(XML, _, _, _, 'text="Note"'),
    \+ sub_atom(XML, _, _, _, '<link').

test(tree_child) :-
    child_to_xml(child(tree, 'Subtree', null, 3), XML),
    sub_atom(XML, _, _, _, 'text="Subtree"'),
    sub_atom(XML, _, _, _, 'color="#F5A623"').

test(section_child) :-
    child_to_xml(child(section, 'Resources', null, 4), XML),
    sub_atom(XML, _, _, _, 'text="Resources"'),
    sub_atom(XML, _, _, _, 'style="bold"').

test(alias_child) :-
    child_to_xml(child(alias, 'Link to Other', null, 5), XML),
    sub_atom(XML, _, _, _, 'color="#9B59B6"').

test(escapes_special_chars_in_title) :-
    child_to_xml(child(pagepearl, 'Tom & Jerry', 'http://example.com', 1), XML),
    sub_atom(XML, _, _, _, 'text="Tom &amp; Jerry"').

:- end_tests(child_to_xml).

:- begin_tests(generate_mindmap).

test(basic_mindmap) :-
    Children = [
        child(pagepearl, 'Link 1', 'http://example.com', 1),
        child(tree, 'Subtree', null, 2)
    ],
    pearltrees_templates:generate_mindmap('12345', 'Test Tree', Children, XML),
    sub_atom(XML, _, _, _, '<?xml version="1.0"'),
    sub_atom(XML, _, _, _, '<smmx version="2">'),
    sub_atom(XML, _, _, _, 'text="Test Tree"'),
    sub_atom(XML, _, _, _, 'id="root_12345"'),
    sub_atom(XML, _, _, _, 'text="Link 1"'),
    sub_atom(XML, _, _, _, 'text="Subtree"').

test(empty_children) :-
    pearltrees_templates:generate_mindmap('99999', 'Empty', [], XML),
    sub_atom(XML, _, _, _, 'text="Empty"'),
    sub_atom(XML, _, _, _, 'id="root_99999"').

test(escapes_title) :-
    pearltrees_templates:generate_mindmap('11111', 'Science & Technology', [], XML),
    sub_atom(XML, _, _, _, 'text="Science &amp; Technology"').

:- end_tests(generate_mindmap).

:- begin_tests(generate_mindmap_xml).

test(includes_root_link) :-
    pearltrees_templates:generate_mindmap_xml('12345', 'Test', 'https://pearltrees.com/test', [], XML),
    sub_atom(XML, _, _, _, '<link url="https://pearltrees.com/test"/>').

:- end_tests(generate_mindmap_xml).

%% ====================================================================
%% FreeMind Format Tests
%% ====================================================================

:- begin_tests(freemind).

test(basic_freemind) :-
    Children = [
        child(pagepearl, 'Link 1', 'http://example.com', 1),
        child(tree, 'Subtree', null, 2)
    ],
    generate_freemind('12345', 'Test Tree', Children, MM),
    sub_atom(MM, _, _, _, '<?xml version="1.0"'),
    sub_atom(MM, _, _, _, '<map version="1.0.1">'),
    sub_atom(MM, _, _, _, 'TEXT="Test Tree"'),
    sub_atom(MM, _, _, _, 'TEXT="Link 1"'),
    sub_atom(MM, _, _, _, 'TEXT="Subtree"').

test(freemind_with_url) :-
    Children = [child(pagepearl, 'GitHub', 'https://github.com', 1)],
    generate_freemind('111', 'Links', Children, MM),
    sub_atom(MM, _, _, _, 'NAME="url"'),
    sub_atom(MM, _, _, _, 'VALUE="https://github.com"').

test(freemind_positions) :-
    Children = [
        child(pagepearl, 'Even', null, 2),
        child(pagepearl, 'Odd', null, 3)
    ],
    generate_freemind('222', 'Test', Children, MM),
    sub_atom(MM, _, _, _, 'POSITION="right"'),
    sub_atom(MM, _, _, _, 'POSITION="left"').

test(freemind_tree_folded) :-
    Children = [child(tree, 'Folder', null, 1)],
    generate_freemind('333', 'Test', Children, MM),
    sub_atom(MM, _, _, _, 'FOLDED="true"').

test(freemind_alias_bookmark) :-
    Children = [child(alias, 'Alias Link', null, 1)],
    generate_freemind('444', 'Test', Children, MM),
    sub_atom(MM, _, _, _, 'BUILTIN="bookmark"').

:- end_tests(freemind).

%% ====================================================================
%% OPML Format Tests
%% ====================================================================

:- begin_tests(opml).

test(basic_opml) :-
    Children = [
        child(pagepearl, 'Link 1', 'http://example.com', 1),
        child(tree, 'Subtree', null, 2)
    ],
    generate_opml('12345', 'Test Tree', Children, OPML),
    sub_atom(OPML, _, _, _, '<?xml version="1.0"'),
    sub_atom(OPML, _, _, _, '<opml version="2.0">'),
    sub_atom(OPML, _, _, _, '<title>Test Tree</title>'),
    sub_atom(OPML, _, _, _, 'text="Link 1"'),
    sub_atom(OPML, _, _, _, 'text="[Folder] Subtree"').

test(opml_with_url) :-
    Children = [child(pagepearl, 'GitHub', 'https://github.com', 1)],
    generate_opml('111', 'Links', Children, OPML),
    sub_atom(OPML, _, _, _, 'url="https://github.com"'),
    sub_atom(OPML, _, _, _, 'type="link"').

test(opml_section_formatting) :-
    Children = [child(section, 'Resources', null, 1)],
    generate_opml('222', 'Test', Children, OPML),
    sub_atom(OPML, _, _, _, '--- Resources ---').

test(opml_alias_prefix) :-
    Children = [child(alias, 'Linked Tree', null, 1)],
    generate_opml('333', 'Test', Children, OPML),
    sub_atom(OPML, _, _, _, '[Alias] Linked Tree').

:- end_tests(opml).

%% ====================================================================
%% GraphML Format Tests
%% ====================================================================

:- begin_tests(graphml).

test(basic_graphml) :-
    Children = [
        child(pagepearl, 'Link 1', 'http://example.com', 1),
        child(tree, 'Subtree', null, 2)
    ],
    generate_graphml('12345', 'Test Tree', Children, GraphML),
    sub_atom(GraphML, _, _, _, '<?xml version="1.0"'),
    sub_atom(GraphML, _, _, _, '<graphml'),
    sub_atom(GraphML, _, _, _, 'xmlns:y="http://www.yworks.com/xml/graphml"'),
    sub_atom(GraphML, _, _, _, '<y:NodeLabel>Test Tree</y:NodeLabel>').

test(graphml_nodes_and_edges) :-
    Children = [child(pagepearl, 'Link', null, 1)],
    generate_graphml('111', 'Test', Children, GraphML),
    sub_atom(GraphML, _, _, _, '<node id="n_root_111">'),
    sub_atom(GraphML, _, _, _, '<node id="n_111_pagepearl_1">'),
    sub_atom(GraphML, _, _, _, '<edge id=').

test(graphml_url_attribute) :-
    Children = [child(pagepearl, 'GitHub', 'https://github.com', 1)],
    generate_graphml('222', 'Test', Children, GraphML),
    sub_atom(GraphML, _, _, _, '<data key="url">https://github.com</data>').

test(graphml_colors) :-
    Children = [
        child(pagepearl, 'Pearl', null, 1),
        child(tree, 'Tree', null, 2)
    ],
    generate_graphml('333', 'Test', Children, GraphML),
    sub_atom(GraphML, _, _, _, 'color="#6DB33F"'),  % pagepearl
    sub_atom(GraphML, _, _, _, 'color="#F5A623"').  % tree

:- end_tests(graphml).

%% ====================================================================
%% VUE Format Tests
%% ====================================================================

:- begin_tests(vue).

test(basic_vue) :-
    Children = [
        child(pagepearl, 'Link 1', 'http://example.com', 1),
        child(tree, 'Subtree', null, 2)
    ],
    generate_vue('12345', 'Test Tree', Children, VUE),
    sub_atom(VUE, _, _, _, '<?xml version="1.0"'),
    sub_atom(VUE, _, _, _, '<LW-MAP'),
    sub_atom(VUE, _, _, _, 'label="Test Tree"'),
    sub_atom(VUE, _, _, _, 'xsi:type="node"').

test(vue_root_centered) :-
    Children = [],
    generate_vue('111', 'Center', Children, VUE),
    sub_atom(VUE, _, _, _, 'x="400.0" y="300.0"').

test(vue_url_resource) :-
    Children = [child(pagepearl, 'GitHub', 'https://github.com', 1)],
    generate_vue('222', 'Test', Children, VUE),
    sub_atom(VUE, _, _, _, 'xsi:type="URLResource"'),
    sub_atom(VUE, _, _, _, 'spec="https://github.com"').

test(vue_links) :-
    Children = [child(pagepearl, 'Link', null, 1)],
    generate_vue('333', 'Test', Children, VUE),
    sub_atom(VUE, _, _, _, 'xsi:type="link"'),
    sub_atom(VUE, _, _, _, '<ID1 xsi:type="node">100</ID1>').

test(vue_layer) :-
    Children = [],
    generate_vue('444', 'Test', Children, VUE),
    sub_atom(VUE, _, _, _, '<layer ID="1" label="Layer 1"').

:- end_tests(vue).

%% ====================================================================
%% Mermaid Format Tests
%% ====================================================================

:- begin_tests(mermaid).

test(basic_mermaid) :-
    Children = [
        child(pagepearl, 'Link 1', 'http://example.com', 1),
        child(tree, 'Subtree', null, 2)
    ],
    generate_mermaid('12345', 'Test Tree', Children, Mermaid),
    sub_atom(Mermaid, _, _, _, '```mermaid'),
    sub_atom(Mermaid, _, _, _, 'mindmap'),
    sub_atom(Mermaid, _, _, _, 'root((Test Tree))'),
    sub_atom(Mermaid, _, _, _, '```').

test(mermaid_pagepearl_square) :-
    Children = [child(pagepearl, 'Link', null, 1)],
    generate_mermaid('111', 'Test', Children, Mermaid),
    sub_atom(Mermaid, _, _, _, 'pearl_1[Link]').

test(mermaid_tree_hexagon) :-
    Children = [child(tree, 'Folder', null, 1)],
    generate_mermaid('222', 'Test', Children, Mermaid),
    sub_atom(Mermaid, _, _, _, 'tree_1{{Folder}}').

test(mermaid_alias_flag) :-
    Children = [child(alias, 'Alias', null, 1)],
    generate_mermaid('333', 'Test', Children, Mermaid),
    sub_atom(Mermaid, _, _, _, 'alias_1>Alias]').

test(mermaid_section_rounded) :-
    Children = [child(section, 'Section', null, 1)],
    generate_mermaid('444', 'Test', Children, Mermaid),
    sub_atom(Mermaid, _, _, _, 'section_1(Section)').

test(mermaid_escapes_brackets) :-
    Children = [child(pagepearl, 'Array[0]', null, 1)],
    generate_mermaid('555', 'Test', Children, Mermaid),
    sub_atom(Mermaid, _, _, _, 'Array(0)').

:- end_tests(mermaid).

%% ====================================================================
%% Unified Multi-Format Tests
%% ====================================================================

:- begin_tests(multi_format).

test(available_formats_list) :-
    available_formats(Formats),
    length(Formats, 6),
    member(smmx, Formats),
    member(freemind, Formats),
    member(mermaid, Formats).

test(generate_all_formats_subset) :-
    Children = [child(pagepearl, 'Link', 'http://example.com', 1)],
    generate_all_formats('123', 'Test', Children, [freemind, mermaid], Results),
    length(Results, 2),
    member(format(freemind, '.mm', _), Results),
    member(format(mermaid, '.md', _), Results).

test(generate_all_formats_full) :-
    Children = [child(pagepearl, 'Link', null, 1)],
    available_formats(AllFormats),
    generate_all_formats('456', 'Full Test', Children, AllFormats, Results),
    length(Results, 6).

test(format_result_contains_content) :-
    Children = [child(tree, 'Folder', null, 1)],
    generate_all_formats('789', 'Content Test', Children, [opml], [format(opml, Ext, Content)]),
    Ext == '.opml',
    sub_atom(Content, _, _, _, 'Content Test').

:- end_tests(multi_format).
