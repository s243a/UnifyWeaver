%% pearltrees/test_browser_automation.pl - Tests for abstract browser automation
%%
%% Run with: swipl -g "run_tests" -t halt test_browser_automation.pl

:- module(test_browser_automation, []).

:- use_module(library(plunit)).
:- use_module(browser_automation).

%% --------------------------------------------------------------------
%% Tests for URL template expansion
%% --------------------------------------------------------------------

:- begin_tests(expand_template).

test(simple_variable) :-
    Context = _{tree_id: "12345"},
    expand_template("id{tree_id}", Context, Result),
    Result == 'id12345'.

test(multiple_variables) :-
    Context = _{account: "user1", slug: "science", tree_id: "12345"},
    expand_template("https://example.com/{account}/{slug}/id{tree_id}", Context, Result),
    Result == 'https://example.com/user1/science/id12345'.

test(missing_variable_empty) :-
    Context = _{account: "user1"},
    expand_template("{account}/{missing}", Context, Result),
    Result == 'user1/'.

test(no_variables) :-
    Context = _{},
    expand_template("https://example.com/static", Context, Result),
    Result == 'https://example.com/static'.

test(numeric_value) :-
    Context = _{tree_id: 12345},
    expand_template("id{tree_id}", Context, Result),
    Result == 'id12345'.

:- end_tests(expand_template).

%% --------------------------------------------------------------------
%% Tests for workflow steps
%% --------------------------------------------------------------------

:- begin_tests(workflow_steps).

test(fetch_tree_has_four_steps) :-
    findall(Idx, workflow_step(fetch_tree, Idx, _), Indices),
    length(Indices, 4).

test(fetch_tree_starts_with_navigate) :-
    workflow_step(fetch_tree, 1, Step),
    Step = step(navigate, tree_page, []).

test(fetch_tree_ends_with_parse) :-
    workflow_step(fetch_tree, 4, Step),
    Step = step(parse, tree_response, []).

test(batch_fetch_has_delay_step) :-
    workflow_step(batch_fetch, 5, Step),
    Step = step(delay, human_like, []).

:- end_tests(workflow_steps).

%% --------------------------------------------------------------------
%% Tests for context helpers
%% --------------------------------------------------------------------

:- begin_tests(context).

test(put_and_get) :-
    Context = _{},
    put_context(foo, bar, Context, Context1),
    get_context(foo, Context1, Value),
    Value == bar.

test(overwrite_value) :-
    Context = _{foo: old},
    put_context(foo, new, Context, Context1),
    get_context(foo, Context1, Value),
    Value == new.

test(preserve_other_keys) :-
    Context = _{foo: 1, bar: 2},
    put_context(baz, 3, Context, Context1),
    get_context(foo, Context1, V1),
    get_context(bar, Context1, V2),
    get_context(baz, Context1, V3),
    V1 == 1, V2 == 2, V3 == 3.

:- end_tests(context).

%% --------------------------------------------------------------------
%% Tests for pearl extraction
%% --------------------------------------------------------------------

:- begin_tests(extract_pearls).

test(empty_data) :-
    extract_pearls(_{}, Pearls),
    Pearls == [].

test(missing_pearls_key) :-
    extract_pearls(_{tree: _{id: 123}}, Pearls),
    Pearls == [].

test(extracts_pearl_list) :-
    Data = _{pearls: [
        _{id: 1, contentType: 1, title: "Link 1"},
        _{id: 2, contentType: 2, title: "Subtree"}
    ]},
    extract_pearls(Data, Pearls),
    length(Pearls, 2).

test(normalizes_pearl_structure) :-
    Data = _{pearls: [_{id: 42, contentType: 1, title: "Test"}]},
    extract_pearls(Data, [Pearl]),
    Pearl = pearl(42, 1, "Test").

:- end_tests(extract_pearls).

%% --------------------------------------------------------------------
%% Tests for timing resolution
%% --------------------------------------------------------------------

:- begin_tests(timing).

test(page_load_default) :-
    resolve_wait(page_load, [], Duration),
    Duration == 3.

test(page_load_custom) :-
    resolve_wait(page_load, [seconds(5)], Duration),
    Duration == 5.

test(custom_wait) :-
    resolve_wait(custom, [seconds(10)], Duration),
    Duration == 10.

:- end_tests(timing).

%% --------------------------------------------------------------------
%% Tests for config access
%% --------------------------------------------------------------------

:- begin_tests(config_access).

test(api_endpoint_lookup) :-
    Config = _{endpoints: _{
        tree_api: _{url_template: "https://example.com/api?id={tree_id}"}
    }},
    api_endpoint(Config, tree_api, Endpoint),
    get_dict(url_template, Endpoint, Url),
    Url == "https://example.com/api?id={tree_id}".

test(api_param_lookup) :-
    Config = _{timing: _{page_load_wait: 3}},
    api_param(Config, timing, page_load_wait, Value),
    Value == 3.

test(missing_endpoint_fails, [fail]) :-
    Config = _{endpoints: _{}},
    api_endpoint(Config, nonexistent, _).

:- end_tests(config_access).
