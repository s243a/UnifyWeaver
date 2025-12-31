/**
 * Fuzzy Logic DSL - Filter Predicates
 *
 * Provides metadata and hierarchical filter predicates:
 *
 * Metadata filters:
 * - is_type/1: Match item type (tree, pearl, etc.)
 * - has_account/1: Match account
 * - has_parent/1: Match parent folder name
 * - in_subtree/1: Match path contains
 * - has_tag/1: Match tag
 *
 * Hierarchical filters:
 * - child_of/1: Direct children only
 * - descendant_of/1: Any depth below (alias for in_subtree)
 * - parent_of/1: Immediate parent
 * - ancestor_of/1: Any depth above
 * - sibling_of/1: Same parent
 * - root_of/1: Root nodes
 * - has_depth/1: At exact depth
 * - depth_between/2: Depth in range
 * - near/3: Fuzzy distance (score decays with distance)
 *
 * These predicates operate on the "current item" set by the evaluation context.
 */

:- module(fuzzy_predicates, [
    % Metadata predicates
    is_type/1,
    has_account/1,
    has_parent/1,
    in_subtree/1,
    has_tag/1,

    % Hierarchical predicates
    child_of/1,
    descendant_of/1,
    parent_of/1,
    ancestor_of/1,
    sibling_of/1,
    root_of/1,
    has_depth/1,
    depth_between/2,
    depth_at_least/1,
    depth_at_most/1,

    % Path matching
    path_matches/1,
    path_regex/1,

    % Fuzzy distance
    near/3,

    % Item context (set by evaluation)
    set_current_item/1,
    current_item/1,
    current_item_property/2,

    % Filter registration
    register_filter/1,
    registered_filter/1
]).

:- use_module(library(pcre), [re_match/2]).

% =============================================================================
% Current Item Context
% =============================================================================

%% Dynamic predicate for current item being evaluated
:- dynamic current_item_data/1.

%% set_current_item(+Item)
%  Set the current item for predicate evaluation.
%  Item should be a structure or dict with properties.
set_current_item(Item) :-
    retractall(current_item_data(_)),
    assertz(current_item_data(Item)).

%% current_item(-Item)
%  Get the current item.
current_item(Item) :-
    current_item_data(Item).

%% current_item_property(+Property, -Value)
%  Get a property of the current item.
current_item_property(Property, Value) :-
    current_item(Item),
    item_property(Item, Property, Value).

% Item property access (handles dicts and structures)
item_property(Item, Property, Value) :-
    is_dict(Item), !,
    get_dict(Property, Item, Value).
item_property(Item, Property, Value) :-
    compound(Item),
    Item =.. [_|Args],
    property_index(Property, Index),
    nth1(Index, Args, Value).

% Property indices for item structures (customize as needed)
property_index(type, 1).
property_index(account, 2).
property_index(parent, 3).
property_index(path, 4).
property_index(depth, 5).
property_index(tags, 6).

% =============================================================================
% Metadata Predicates
% =============================================================================

%% is_type(+Type)
%  True if current item has the given type.
is_type(Type) :-
    current_item_property(type, ItemType),
    ItemType == Type.

%% has_account(+Account)
%  True if current item belongs to the given account.
has_account(Account) :-
    current_item_property(account, ItemAccount),
    ItemAccount == Account.

%% has_parent(+Parent)
%  True if current item's immediate parent matches.
has_parent(Parent) :-
    current_item_property(parent, ItemParent),
    (   atom(Parent)
    ->  atom_string(Parent, ParentStr),
        (   atom(ItemParent) -> atom_string(ItemParent, ItemParentStr) ; ItemParentStr = ItemParent ),
        sub_string(ItemParentStr, _, _, _, ParentStr)
    ;   ItemParent == Parent
    ).

%% in_subtree(+Subtree)
%  True if current item's path contains the subtree.
in_subtree(Subtree) :-
    current_item_property(path, Path),
    (   atom(Subtree) -> atom_string(Subtree, SubtreeStr) ; SubtreeStr = Subtree ),
    (   atom(Path) -> atom_string(Path, PathStr) ; PathStr = Path ),
    sub_string(PathStr, _, _, _, SubtreeStr).

%% has_tag(+Tag)
%  True if current item has the given tag.
has_tag(Tag) :-
    current_item_property(tags, Tags),
    member(Tag, Tags).

% =============================================================================
% Hierarchical Predicates
% =============================================================================

%% child_of(+Node)
%  True if current item is a direct child of Node.
child_of(Node) :-
    current_item_property(parent, Parent),
    parent_matches(Parent, Node).

%% descendant_of(+Node)
%  True if current item is anywhere below Node in the hierarchy.
%  Alias for in_subtree/1.
descendant_of(Node) :-
    in_subtree(Node).

%% parent_of(+Node)
%  True if current item is the immediate parent of Node.
%  Note: Requires knowing Node's parent, typically via lookup.
parent_of(Node) :-
    current_item(Item),
    item_property(Item, id, ItemId),
    node_parent(Node, ItemId).

%% ancestor_of(+Node)
%  True if current item is anywhere above Node in the hierarchy.
ancestor_of(Node) :-
    current_item_property(path, AncestorPath),
    node_path(Node, NodePath),
    sub_string(NodePath, 0, _, _, AncestorPath),
    AncestorPath \== NodePath.

%% sibling_of(+Node)
%  True if current item has the same parent as Node.
sibling_of(Node) :-
    current_item_property(parent, Parent),
    node_parent(Node, NodeParent),
    Parent == NodeParent.

%% root_of(+Tree)
%  True if current item is a root node of the given tree/account.
root_of(Tree) :-
    current_item_property(depth, 0),
    current_item_property(account, Tree).

%% has_depth(+N)
%  True if current item is at exactly depth N.
has_depth(N) :-
    current_item_property(depth, Depth),
    Depth == N.

%% depth_between(+Min, +Max)
%  True if current item's depth is in [Min, Max].
depth_between(Min, Max) :-
    current_item_property(depth, Depth),
    Depth >= Min,
    Depth =< Max.

%% depth_at_least(+N)
%  True if current item's depth >= N.
depth_at_least(N) :-
    current_item_property(depth, Depth),
    Depth >= N.

%% depth_at_most(+N)
%  True if current item's depth <= N.
depth_at_most(N) :-
    current_item_property(depth, Depth),
    Depth =< N.

% =============================================================================
% Path Matching
% =============================================================================

%% path_matches(+Pattern)
%  True if current item's path matches a glob-style pattern.
%  Pattern uses * for single component, ** for any depth.
path_matches(Pattern) :-
    current_item_property(path, Path),
    glob_to_regex(Pattern, Regex),
    re_match(Regex, Path).

%% path_regex(+Regex)
%  True if current item's path matches a regex.
path_regex(Regex) :-
    current_item_property(path, Path),
    re_match(Regex, Path).

% Convert glob pattern to regex
glob_to_regex(Glob, Regex) :-
    atom_string(Glob, GlobStr),
    string_chars(GlobStr, Chars),
    glob_chars_to_regex(Chars, RegexChars),
    string_chars(RegexStr, RegexChars),
    atom_string(Regex, RegexStr).

glob_chars_to_regex([], []).
glob_chars_to_regex(['*', '*'|Rest], ['.', '*'|RestRegex]) :- !,
    glob_chars_to_regex(Rest, RestRegex).
glob_chars_to_regex(['*'|Rest], ['[', '^', '/', ']', '*'|RestRegex]) :- !,
    glob_chars_to_regex(Rest, RestRegex).
glob_chars_to_regex(['?'|Rest], ['[', '^', '/', ']'|RestRegex]) :- !,
    glob_chars_to_regex(Rest, RestRegex).
glob_chars_to_regex([C|Rest], [C|RestRegex]) :-
    glob_chars_to_regex(Rest, RestRegex).

% =============================================================================
% Fuzzy Distance
% =============================================================================

%% near(+Node, +Decay, -Score)
%  Returns a score that decays with tree distance from Node.
%  Score = Decay^distance, so closer items score higher.
%  Decay should be in (0, 1), e.g., 0.5 means score halves per level.
near(Node, Decay, Score) :-
    current_item(Item),
    tree_distance(Item, Node, Distance),
    Score is Decay ** Distance.

near(Node, _Decay, 0.0) :-
    \+ (current_item(Item), tree_distance(Item, Node, _)).

% Tree distance between two nodes (simplified - override for actual impl)
tree_distance(Item, Node, Distance) :-
    item_property(Item, path, ItemPath),
    node_path(Node, NodePath),
    path_distance(ItemPath, NodePath, Distance).

% Path distance: count differing components
path_distance(Path1, Path2, Distance) :-
    split_path(Path1, Parts1),
    split_path(Path2, Parts2),
    common_prefix_length(Parts1, Parts2, Common),
    length(Parts1, Len1),
    length(Parts2, Len2),
    Distance is (Len1 - Common) + (Len2 - Common).

split_path(Path, Parts) :-
    (   atom(Path) -> atom_string(Path, PathStr) ; PathStr = Path ),
    split_string(PathStr, "/", "/", Parts).

common_prefix_length([], _, 0) :- !.
common_prefix_length(_, [], 0) :- !.
common_prefix_length([H|T1], [H|T2], N) :- !,
    common_prefix_length(T1, T2, N1),
    N is N1 + 1.
common_prefix_length(_, _, 0).

% =============================================================================
% Helper predicates (to be overridden with actual data access)
% =============================================================================

:- dynamic node_parent/2.   % node_parent(Node, Parent)
:- dynamic node_path/2.     % node_path(Node, Path)

% Parent matching helper
parent_matches(Parent, Node) :-
    (   atom(Node) -> atom_string(Node, NodeStr) ; NodeStr = Node ),
    (   atom(Parent) -> atom_string(Parent, ParentStr) ; ParentStr = Parent ),
    (   sub_string(ParentStr, _, _, 0, NodeStr)  % Ends with Node
    ;   ParentStr == NodeStr
    ).

% =============================================================================
% Filter Registration
% =============================================================================

:- dynamic registered_filter_pred/1.

%% register_filter(+FilterSpec)
%  Register a custom filter for use in fuzzy expressions.
%  FilterSpec is Name/Arity, e.g., my_filter/2.
register_filter(Name/Arity) :-
    assertz(registered_filter_pred(Name/Arity)).

%% registered_filter(?FilterSpec)
%  Query registered filters.
registered_filter(Spec) :-
    registered_filter_pred(Spec).
