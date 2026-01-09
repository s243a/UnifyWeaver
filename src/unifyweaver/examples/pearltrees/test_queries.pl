%% pearltrees/test_queries.pl - Unit tests for Pearltrees queries
%%
%% Tests for aggregate queries using mock data.
%% Run with: swipl -g "load_test_files([]), run_tests" -t halt test_queries.pl

:- module(test_pearltrees_queries, []).

:- use_module(library(plunit)).

%% --------------------------------------------------------------------
%% Mock Data
%%
%% Instead of connecting to real SQLite/JSONL, we define test facts.
%% --------------------------------------------------------------------

:- dynamic mock_pearl_trees/5.
:- dynamic mock_pearl_children/6.

setup_mock_data :-
    retractall(mock_pearl_trees(_, _, _, _, _)),
    retractall(mock_pearl_children(_, _, _, _, _, _)),

    % Mock trees
    assertz(mock_pearl_trees(tree, '12345', 'Science Topics', 'https://pearltrees.com/user/science/id12345', 'https://pearltrees.com/user/root')),
    assertz(mock_pearl_trees(tree, '12346', 'Empty Tree', 'https://pearltrees.com/user/empty/id12346', 'https://pearltrees.com/user/root')),
    assertz(mock_pearl_trees(tree, '12347', 'Tech Links', 'https://pearltrees.com/user/tech/id12347', 'https://pearltrees.com/user/science/id12345')),

    % Mock children for tree 12345 (Science Topics - 3 children)
    assertz(mock_pearl_children('12345', pagepearl, 'Wikipedia Physics', 1, 'https://en.wikipedia.org/wiki/Physics', null)),
    assertz(mock_pearl_children('12345', pagepearl, 'Nature Journal', 2, 'https://nature.com', null)),
    assertz(mock_pearl_children('12345', tree, 'Tech Links', 3, null, 'https://pearltrees.com/user/tech/id12347')),

    % Mock children for tree 12346 (Empty Tree - only root)
    assertz(mock_pearl_children('12346', root, 'Empty Tree', 0, null, null)),

    % Mock children for tree 12347 (Tech Links - 2 children)
    assertz(mock_pearl_children('12347', pagepearl, 'GitHub', 1, 'https://github.com', null)),
    assertz(mock_pearl_children('12347', pagepearl, 'Stack Overflow', 2, 'https://stackoverflow.com', null)).

cleanup_mock_data :-
    retractall(mock_pearl_trees(_, _, _, _, _)),
    retractall(mock_pearl_children(_, _, _, _, _, _)).

%% --------------------------------------------------------------------
%% Test versions of queries using mock data
%% --------------------------------------------------------------------

%% tree_with_children using mock data
mock_tree_with_children(TreeId, Title, Children) :-
    mock_pearl_trees(tree, TreeId, Title, _, _),
    findall(
        child(Type, ChildTitle, Url, Order),
        mock_pearl_children(TreeId, Type, ChildTitle, Order, Url, _),
        Children
    ).

%% tree_child_count using mock data
mock_tree_child_count(TreeId, Count) :-
    mock_pearl_trees(tree, TreeId, _, _, _),
    findall(1, mock_pearl_children(TreeId, _, _, _, _, _), Matches),
    length(Matches, Count).

%% incomplete_tree using mock data
mock_incomplete_tree(TreeId, Title) :-
    mock_tree_child_count(TreeId, Count),
    Count =< 1,
    mock_pearl_trees(tree, TreeId, Title, _, _).

%% trees_by_cluster using mock data
mock_trees_by_cluster(ClusterId, Trees) :-
    mock_pearl_trees(tree, _, _, _, ClusterId),
    findall(
        tree(TreeId, Title, Uri),
        mock_pearl_trees(tree, TreeId, Title, Uri, ClusterId),
        Trees
    ),
    Trees \= [].

%% pagepearl_urls using mock data
mock_pagepearl_urls(TreeId, Urls) :-
    mock_pearl_trees(tree, TreeId, _, _, _),
    findall(
        Url,
        (mock_pearl_children(TreeId, pagepearl, _, _, Url, _), Url \= null),
        Urls
    ).

%% --------------------------------------------------------------------
%% Tests
%% --------------------------------------------------------------------

:- begin_tests(tree_with_children, [setup(setup_mock_data), cleanup(cleanup_mock_data)]).

test(science_tree_has_three_children) :-
    mock_tree_with_children('12345', Title, Children),
    Title == 'Science Topics',
    length(Children, 3).

test(science_tree_children_types) :-
    mock_tree_with_children('12345', _, Children),
    findall(Type, member(child(Type, _, _, _), Children), Types),
    msort(Types, Sorted),
    Sorted == [pagepearl, pagepearl, tree].

test(empty_tree_has_one_child) :-
    mock_tree_with_children('12346', Title, Children),
    Title == 'Empty Tree',
    length(Children, 1).

test(tech_tree_has_two_children) :-
    mock_tree_with_children('12347', _, Children),
    length(Children, 2).

:- end_tests(tree_with_children).

:- begin_tests(tree_child_count, [setup(setup_mock_data), cleanup(cleanup_mock_data)]).

test(science_tree_count) :-
    mock_tree_child_count('12345', Count),
    Count == 3.

test(empty_tree_count) :-
    mock_tree_child_count('12346', Count),
    Count == 1.

test(tech_tree_count) :-
    mock_tree_child_count('12347', Count),
    Count == 2.

:- end_tests(tree_child_count).

:- begin_tests(incomplete_tree, [setup(setup_mock_data), cleanup(cleanup_mock_data)]).

test(finds_empty_tree) :-
    findall(TreeId, mock_incomplete_tree(TreeId, _), Incomplete),
    Incomplete == ['12346'].

test(empty_tree_title) :-
    mock_incomplete_tree('12346', Title),
    Title == 'Empty Tree'.

test(science_tree_not_incomplete, [fail]) :-
    mock_incomplete_tree('12345', _).

:- end_tests(incomplete_tree).

:- begin_tests(trees_by_cluster, [setup(setup_mock_data), cleanup(cleanup_mock_data)]).

test(root_cluster_has_two_trees) :-
    mock_trees_by_cluster('https://pearltrees.com/user/root', Trees),
    length(Trees, 2).

test(science_cluster_has_one_tree) :-
    mock_trees_by_cluster('https://pearltrees.com/user/science/id12345', Trees),
    length(Trees, 1),
    Trees = [tree('12347', 'Tech Links', _)].

:- end_tests(trees_by_cluster).

:- begin_tests(pagepearl_urls, [setup(setup_mock_data), cleanup(cleanup_mock_data)]).

test(science_tree_urls) :-
    mock_pagepearl_urls('12345', Urls),
    length(Urls, 2),
    member('https://en.wikipedia.org/wiki/Physics', Urls),
    member('https://nature.com', Urls).

test(tech_tree_urls) :-
    mock_pagepearl_urls('12347', Urls),
    length(Urls, 2),
    member('https://github.com', Urls).

test(empty_tree_no_urls) :-
    mock_pagepearl_urls('12346', Urls),
    Urls == [].

:- end_tests(pagepearl_urls).

%% ====================================================================
%% Query Filter Tests
%% ====================================================================

%% Mock filter predicates

mock_has_domain_links(TreeId, Domain) :-
    mock_pearl_trees(tree, TreeId, _, _, _),
    mock_pearl_children(TreeId, pagepearl, _, _, Url, _),
    Url \= null,
    sub_atom(Url, _, _, _, Domain),
    !.

mock_has_child_type(TreeId, Type) :-
    mock_pearl_trees(tree, TreeId, _, _, _),
    mock_pearl_children(TreeId, Type, _, _, _, _),
    !.

mock_title_matches(Title, Pattern) :-
    downcase_atom(Title, LowerTitle),
    downcase_atom(Pattern, LowerPattern),
    sub_atom(LowerTitle, _, _, _, LowerPattern).

mock_trees_matching(Pattern, TreeId) :-
    mock_pearl_trees(tree, TreeId, Title, _, _),
    mock_title_matches(Title, Pattern).

mock_children_of_type(TreeId, Type, Children) :-
    findall(
        child(Type, Title, Url, Order),
        mock_pearl_children(TreeId, Type, Title, Order, Url, _),
        Children
    ).

mock_trees_with_min_children(MinCount, TreeId) :-
    mock_tree_child_count(TreeId, Count),
    Count >= MinCount.

mock_children_matching(TreeId, Pattern, Children) :-
    findall(
        child(Type, Title, Url, Order),
        (   mock_pearl_children(TreeId, Type, Title, Order, Url, _),
            mock_title_matches(Title, Pattern)
        ),
        Children
    ).

mock_filter_matches(domain(Domain), TreeId) :-
    mock_has_domain_links(TreeId, Domain).
mock_filter_matches(type(Type), TreeId) :-
    mock_has_child_type(TreeId, Type).
mock_filter_matches(min_children(N), TreeId) :-
    mock_tree_child_count(TreeId, Count),
    Count >= N.
mock_filter_matches(complete, TreeId) :-
    mock_tree_child_count(TreeId, Count),
    Count > 1.
mock_filter_matches(not(Filter), TreeId) :-
    \+ mock_filter_matches(Filter, TreeId).

mock_all_filters_match([], _).
mock_all_filters_match([Filter|Rest], TreeId) :-
    mock_filter_matches(Filter, TreeId),
    mock_all_filters_match(Rest, TreeId).

mock_apply_filters(Filters, TreeId, tree_info(TreeId, Title, ChildCount)) :-
    mock_pearl_trees(tree, TreeId, Title, _, _),
    mock_tree_child_count(TreeId, ChildCount),
    mock_all_filters_match(Filters, TreeId).

:- begin_tests(domain_filters, [setup(setup_mock_data), cleanup(cleanup_mock_data)]).

test(has_wikipedia_links) :-
    mock_has_domain_links('12345', 'wikipedia.org').

test(no_wikipedia_in_tech, [fail]) :-
    mock_has_domain_links('12347', 'wikipedia.org').

test(has_github_links) :-
    mock_has_domain_links('12347', 'github.com').

test(trees_with_github) :-
    findall(TreeId, mock_has_domain_links(TreeId, 'github.com'), Trees),
    Trees == ['12347'].

:- end_tests(domain_filters).

:- begin_tests(type_filters, [setup(setup_mock_data), cleanup(cleanup_mock_data)]).

test(has_pagepearl_type) :-
    mock_has_child_type('12345', pagepearl).

test(has_tree_type) :-
    mock_has_child_type('12345', tree).

test(no_alias_type, [fail]) :-
    mock_has_child_type('12345', alias).

test(children_of_type_pagepearl) :-
    mock_children_of_type('12345', pagepearl, Children),
    length(Children, 2).

test(children_of_type_tree) :-
    mock_children_of_type('12345', tree, Children),
    length(Children, 1).

:- end_tests(type_filters).

:- begin_tests(title_filters, [setup(setup_mock_data), cleanup(cleanup_mock_data)]).

test(title_matches_case_insensitive) :-
    mock_title_matches('Science Topics', 'science').

test(title_matches_partial) :-
    mock_title_matches('Wikipedia Physics', 'wiki').

test(trees_matching_science) :-
    findall(TreeId, mock_trees_matching('science', TreeId), Trees),
    Trees == ['12345'].

test(trees_matching_tech) :-
    findall(TreeId, mock_trees_matching('tech', TreeId), Trees),
    Trees == ['12347'].

test(children_matching_wiki) :-
    mock_children_matching('12345', 'wiki', Children),
    length(Children, 1),
    Children = [child(pagepearl, 'Wikipedia Physics', _, _)].

:- end_tests(title_filters).

:- begin_tests(count_filters, [setup(setup_mock_data), cleanup(cleanup_mock_data)]).

test(trees_with_min_2_children) :-
    findall(TreeId, mock_trees_with_min_children(2, TreeId), Trees),
    length(Trees, 2),
    member('12345', Trees),
    member('12347', Trees).

test(trees_with_min_3_children) :-
    findall(TreeId, mock_trees_with_min_children(3, TreeId), Trees),
    Trees == ['12345'].

test(no_trees_with_min_4, [fail]) :-
    mock_trees_with_min_children(4, _).

:- end_tests(count_filters).

:- begin_tests(combined_filters, [setup(setup_mock_data), cleanup(cleanup_mock_data)]).

test(filter_complete_with_pagepearls) :-
    findall(TreeId, mock_apply_filters([complete, type(pagepearl)], TreeId, _), Trees),
    length(Trees, 2).

test(filter_with_github_and_min_2) :-
    findall(TreeId, mock_apply_filters([domain('github.com'), min_children(2)], TreeId, _), Trees),
    Trees == ['12347'].

test(filter_not_incomplete) :-
    findall(TreeId, mock_apply_filters([not(min_children(2))], TreeId, _), Trees),
    Trees == ['12346'].

test(apply_filters_returns_info) :-
    mock_apply_filters([complete], '12345', Info),
    Info = tree_info('12345', 'Science Topics', 3).

:- end_tests(combined_filters).

%% --------------------------------------------------------------------
%% Run tests when loaded directly
%% --------------------------------------------------------------------

:- initialization((
    setup_mock_data,
    run_tests,
    cleanup_mock_data
), main).
