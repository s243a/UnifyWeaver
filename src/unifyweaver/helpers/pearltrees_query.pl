:- module(pearltrees_query, [
    find_trees_by_title/2,
    find_trees_by_title/3,
    get_tree_by_id/2,
    get_tree_by_id/3,
    count_trees/1,
    count_trees/2,
    extract_tree_info/2
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
