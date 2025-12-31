:- module(pearltrees_query, [
    find_trees_by_title/2,
    find_trees_by_title/3,
    get_tree_by_id/2,
    get_tree_by_id/3,
    count_trees/1,
    count_trees/2,
    extract_tree_info/2,
    get_child_pearls/2,
    get_child_pearls/3,
    get_child_trees/2,
    get_child_trees/3,
    get_tree_hierarchy/3,
    get_tree_hierarchy/4
]).

:- use_module('xml_query').
:- use_module(library(pcre)).

%% find_trees_by_title(+SearchTerm, -Trees)
%  Find all trees whose title contains SearchTerm (case-insensitive).
%  Uses default file: context/PT/pearltrees_export.rdf
%
%  Example:
%    find_trees_by_title('physics', Trees).
find_trees_by_title(SearchTerm, Trees) :-
    find_trees_by_title('context/PT/pearltrees_export.rdf', SearchTerm, Trees).

%% find_trees_by_title(+File, +SearchTerm, -Trees)
%  Find all trees in File whose title contains SearchTerm.
%
%  Example:
%    find_trees_by_title('my_export.rdf', 'quantum', Trees).
find_trees_by_title(File, SearchTerm, Trees) :-
    xml_query:find_elements(File, 'pt:Tree', awk_pipeline,
                           contains(title, SearchTerm), Trees).

%% get_tree_by_id(+TreeId, -Tree)
%  Get a specific tree by its ID.
%  Uses default file: context/PT/pearltrees_export.rdf
%
%  Example:
%    get_tree_by_id(10647426, Tree).
get_tree_by_id(TreeId, Tree) :-
    get_tree_by_id('context/PT/pearltrees_export.rdf', TreeId, Tree).

%% get_tree_by_id(+File, +TreeId, -Tree)
%  Get a specific tree by its ID from File.
get_tree_by_id(File, TreeId, Tree) :-
    format(atom(IdPattern), 'id~w', [TreeId]),
    xml_query:find_elements(File, 'pt:Tree', awk_pipeline,
                           matches(IdPattern), [Tree|_]).

%% count_trees(+Count)
%  Count total trees in default Pearltrees export.
count_trees(Count) :-
    count_trees('context/PT/pearltrees_export.rdf', Count).

%% count_trees(+File, -Count)
%  Count total trees in File.
count_trees(File, Count) :-
    xml_query:count_elements(File, 'pt:Tree', awk_pipeline, Count).

%% get_child_pearls(+TreeId, -Pearls)
%  Get all pearls (AliasPearl elements) that belong to a tree.
%  Uses default file: context/PT/pearltrees_export.rdf
%
%  Example:
%    get_child_pearls(10647426, Pearls).
get_child_pearls(TreeId, Pearls) :-
    get_child_pearls('context/PT/pearltrees_export.rdf', TreeId, Pearls).

%% get_child_pearls(+File, +TreeId, -Pearls)
%  Get all pearls that belong to tree TreeId in File.
get_child_pearls(File, TreeId, Pearls) :-
    format(atom(TreePattern), 'id~w', [TreeId]),
    xml_query:extract_elements(File, 'pt:AliasPearl', awk_pipeline, AllPearls),
    include(belongs_to_tree(TreePattern), AllPearls, Pearls).

%% get_child_trees(+TreeId, -ChildTrees)
%  Get all child trees of a given tree (trees referenced by its pearls).
%  Returns list of child_tree(ChildId, Title, URL) dicts.
%  Uses default file: context/PT/pearltrees_export.rdf
%
%  Example:
%    get_child_trees(10647426, Children).
%    % Children = [child_tree{id:14682380, title:'Physics Education', ...}, ...]
get_child_trees(TreeId, ChildTrees) :-
    get_child_trees('context/PT/pearltrees_export.rdf', TreeId, ChildTrees).

%% get_child_trees(+File, +TreeId, -ChildTrees)
%  Get all child trees of TreeId from File.
get_child_trees(File, TreeId, ChildTrees) :-
    get_child_pearls(File, TreeId, Pearls),
    maplist(extract_child_tree_info, Pearls, ChildInfos),
    include(is_tree_reference, ChildInfos, ChildTrees).

%% get_tree_hierarchy(+TreeId, +MaxDepth, -Hierarchy)
%  Recursively build tree hierarchy up to MaxDepth levels.
%  Uses default file: context/PT/pearltrees_export.rdf
%
%  Hierarchy is: tree(Id, Title, Children) where Children is a list
%  of tree(...) structures.
%
%  Example:
%    get_tree_hierarchy(10647426, 2, Hierarchy).
%    % Hierarchy = tree(10647426, 'Physics', [
%    %   tree(14682380, 'Physics Education', [...]),
%    %   tree(14682381, 'Quantum Physics', [...]),
%    %   ...
%    % ])
get_tree_hierarchy(TreeId, MaxDepth, Hierarchy) :-
    get_tree_hierarchy('context/PT/pearltrees_export.rdf', TreeId, MaxDepth, Hierarchy).

%% get_tree_hierarchy(+File, +TreeId, +MaxDepth, -Hierarchy)
%  Recursively build tree hierarchy from File.
get_tree_hierarchy(File, TreeId, MaxDepth, tree(TreeId, Title, Children)) :-
    get_tree_by_id(File, TreeId, TreeXML),
    extract_tree_info(TreeXML, Info),
    Title = Info.title,
    (   MaxDepth > 0
    ->  get_child_trees(File, TreeId, ChildInfos),
        NextDepth is MaxDepth - 1,
        maplist(build_child_hierarchy(File, NextDepth), ChildInfos, Children)
    ;   Children = []
    ).

%% extract_tree_info(+TreeXML, -Info)
%  Extract structured information from a tree XML element.
%  Info is a dict with keys: id, title, url, last_update, privacy
%
%  Example:
%    get_tree_by_id(10647426, XML),
%    extract_tree_info(XML, Info),
%    writeln(Info.title).
extract_tree_info(TreeXML, info{
    id: TreeId,
    title: Title,
    url: URL,
    last_update: LastUpdate,
    privacy: Privacy
}) :-
    atom_codes(TreeXML, Codes),

    % Extract URL
    (   extract_attribute(Codes, 'rdf:about', URL)
    ->  true
    ;   URL = unknown
    ),

    % Extract tree ID
    (   extract_tag_text(Codes, 'pt:treeId', TreeIdText),
        atom_number(TreeIdText, TreeId)
    ->  true
    ;   TreeId = unknown
    ),

    % Extract title (handle CDATA)
    (   extract_tag_text(Codes, 'dcterms:title', RawTitle),
        strip_cdata(RawTitle, Title)
    ->  true
    ;   Title = unknown
    ),

    % Extract last update
    (   extract_tag_text(Codes, 'pt:lastUpdate', LastUpdate)
    ->  true
    ;   LastUpdate = unknown
    ),

    % Extract privacy
    (   extract_tag_text(Codes, 'pt:privacy', PrivacyText),
        atom_number(PrivacyText, Privacy)
    ->  true
    ;   Privacy = unknown
    ).

%% ============================================
%% HELPER PREDICATES
%% ============================================

belongs_to_tree(TreePattern, Pearl) :-
    % Find <pt:parentTree, then extract everything until its closing />
    sub_atom(Pearl, ParentStart, _, _, '<pt:parentTree'),
    sub_atom(Pearl, ParentStart, _, 0, ParentSection),
    sub_atom(ParentSection, CloseStart, 2, _, '/>'),
    !,  % Take first match
    sub_atom(ParentSection, 0, CloseStart, _, ParentTag),
    % Now extract rdf:resource from this tag only
    sub_atom(ParentTag, ResStart, _, _, 'rdf:resource="'),
    AfterRes is ResStart + 14,
    sub_atom(ParentTag, AfterRes, _, 0, ResRest),
    sub_atom(ResRest, URLEnd, 1, _, '"'),
    sub_atom(ResRest, 0, URLEnd, _, ParentURL),
    sub_atom(ParentURL, _, _, _, TreePattern).

extract_child_tree_info(PearlXML, child_tree{id: ChildId, title: Title, url: URL}) :-
    % Extract title using sub_atom
    (   sub_atom(PearlXML, _, _, _, '<dcterms:title>'),
        sub_atom(PearlXML, TitleStart, _, _, '<dcterms:title>'),
        TAfter is TitleStart + 15,
        sub_atom(PearlXML, TAfter, _, 0, TRest),
        sub_atom(TRest, 0, TEnd, _, '</dcterms:title>'),
        sub_atom(TRest, 0, TEnd, _, RawTitle),
        strip_cdata_atom(RawTitle, Title)
    ->  true
    ;   Title = unknown
    ),

    % Extract rdfs:seeAlso URL (look for <rdfs:seeAlso rdf:resource="...")
    (   sub_atom(PearlXML, _, _, _, '<rdfs:seeAlso'),
        sub_atom(PearlXML, SeeAlsoStart, _, _, '<rdfs:seeAlso'),
        sub_atom(PearlXML, SeeAlsoStart, _, 0, SeeSection),
        sub_atom(SeeSection, 0, SeeClose, _, '/>'),
        sub_atom(SeeSection, 0, SeeClose, _, SeeTag),
        sub_atom(SeeTag, ResStart, _, _, 'rdf:resource="'),
        AfterRes is ResStart + 14,
        sub_atom(SeeTag, AfterRes, _, 0, ResRest),
        sub_atom(ResRest, URLEnd, 1, _, '"'),
        sub_atom(ResRest, 0, URLEnd, _, URL)
    ->  true
    ;   URL = unknown
    ),

    % Extract tree ID from URL (pattern: .../id[0-9]+)
    (   URL \= unknown,
        sub_atom(URL, IDPos, _, _, '/id'),
        AfterID is IDPos + 3,
        sub_atom(URL, AfterID, _, _, IDRest),
        extract_number_atom(IDRest, IDAtom),
        IDAtom \= '',
        atom_number(IDAtom, ChildId)
    ->  true
    ;   ChildId = unknown
    ).

is_tree_reference(child_tree{id: Id}) :-
    Id \= unknown.

build_child_hierarchy(File, MaxDepth, child_tree{id: ChildId}, Hierarchy) :-
    get_tree_hierarchy(File, ChildId, MaxDepth, Hierarchy).

extract_number_codes([], []).
extract_number_codes([C|Rest], [C|NumRest]) :-
    C >= 48, C =< 57,  % 0-9
    !,
    extract_number_codes(Rest, NumRest).
extract_number_codes(_, []).

strip_cdata_atom(Text, Stripped) :-
    (   sub_atom(Text, 0, 9, _, '<![CDATA['),
        sub_atom(Text, 9, _, 3, Content),
        Stripped = Content
    ->  true
    ;   Stripped = Text
    ).

extract_number_atom(Atom, NumberAtom) :-
    atom_chars(Atom, Chars),
    extract_number_chars(Chars, NumChars),
    atom_chars(NumberAtom, NumChars).

extract_number_chars([], []).
extract_number_chars([C|Rest], [C|NumRest]) :-
    char_type(C, digit),
    !,
    extract_number_chars(Rest, NumRest).
extract_number_chars(_, []).

extract_resource_url(Codes, _TagName, URL) :-
    % Find rdf:resource="URL" and extract URL
    atom_codes(Atom, Codes),
    sub_atom(Atom, Before, 14, _, 'rdf:resource="'),
    AfterPrefix is Before + 14,
    sub_atom(Atom, AfterPrefix, _, _, Rest),
    sub_atom(Rest, 0, End, _, '"'),
    !,
    sub_atom(Rest, 0, End, _, URL).

extract_attribute(Codes, AttrName, Value) :-
    atom_codes(AttrName, AttrCodes),
    append(AttrCodes, [61, 34|Rest], Pattern),  % attr="
    append(_, Pattern, Codes),
    append(_, [61, 34|ValueRest], Codes),
    append(ValueCodes, [34|_], ValueRest),
    atom_codes(Value, ValueCodes).

extract_tag_text(Codes, TagName, Text) :-
    atom_codes(TagName, TagCodes),

    % Find opening tag
    append([60|TagCodes], [62|Rest], OpenTag),  % <tag>
    append(_, OpenTag, Codes),
    append(_, [62|AfterOpen], Codes),

    % Find closing tag
    append([60, 47|TagCodes], [62], CloseTag),  % </tag>
    append(TextCodes, CloseTag, AfterOpen),

    atom_codes(Text, TextCodes).

strip_cdata(Text, Stripped) :-
    atom_codes(Text, Codes),
    (   append([60,33,91,67,68,65,84,65,91|ContentCodes], [93,93,62], Codes)
    ->  atom_codes(Stripped, ContentCodes)
    ;   Stripped = Text
    ).
