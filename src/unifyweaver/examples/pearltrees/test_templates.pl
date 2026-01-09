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
