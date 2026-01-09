%% pearltrees/templates.pl - SimpleMind XML template generation
%%
%% Phase 3: Template-based mindmap generation.
%% Generates SimpleMind .smmx XML from tree data.

:- module(pearltrees_templates, [
    generate_mindmap/4,
    generate_mindmap_xml/5,
    child_to_xml/2,
    escape_xml/2
]).

%% --------------------------------------------------------------------
%% XML Generation
%%
%% SimpleMind .smmx format is XML inside a ZIP archive.
%% This module generates the XML content.
%% --------------------------------------------------------------------

%% generate_mindmap(+TreeId, +Title, +Children, -XML) is det.
%%   Generate complete SimpleMind XML for a tree.
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
%%   Generate SimpleMind XML with URI link on root.
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
generate_tree_mindmap(TreeId, XML) :-
    % This would use the actual queries module
    % pearltrees_queries:tree_with_children(TreeId, Title, Children),
    % generate_mindmap(TreeId, Title, Children, XML).
    throw(error(not_implemented, 'Requires loaded sources')).
