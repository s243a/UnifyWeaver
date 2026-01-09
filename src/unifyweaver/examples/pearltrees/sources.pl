%% pearltrees/sources.pl - Data source definitions for Pearltrees processing
%%
%% Educational example showing how to define UnifyWeaver sources for
%% Pearltrees data. Each target generates its own database access code.
%%
%% This does NOT replace existing Python tools - it demonstrates how
%% UnifyWeaver can generate equivalent functionality for multiple targets.

:- module(pearltrees_sources, [
    pearl_children/6,
    pearl_trees/5,
    pearl_api_response/2
]).

%% --------------------------------------------------------------------
%% Source: Children Index (SQLite)
%%
%% Each target generates appropriate DB access:
%%   - Python: sqlite3 module
%%   - C#: Microsoft.Data.Sqlite
%%   - Go: database/sql with sqlite3 driver
%%   - Rust: rusqlite
%% --------------------------------------------------------------------

:- source(sqlite, pearl_children, [
    description('RDF-derived children index'),
    table(children),
    columns([
        parent_tree_id,   % Parent tree ID (string)
        pearl_type,       % Type: pagepearl, tree, alias, section
        title,            % Pearl title
        pos_order,        % Position/order in tree
        external_url,     % URL for pagepearls (nullable)
        see_also_uri      % Reference URI for aliases (nullable)
    ]),
    % Target-specific configuration
    target_config(python, [
        db_path('.local/data/children_index.db'),
        connection_pool(false)
    ]),
    target_config(csharp, [
        db_path('.local/data/children_index.db'),
        connection_pool(true),
        async(true)
    ]),
    target_config(go, [
        db_path('.local/data/children_index.db'),
        driver('github.com/mattn/go-sqlite3')
    ])
]).

%% --------------------------------------------------------------------
%% Source: Trees from JSONL
%%
%% Each target generates appropriate JSON parsing:
%%   - Python: json module, line-by-line iteration
%%   - C#: System.Text.Json with streaming
%%   - Go: encoding/json with bufio.Scanner
%% --------------------------------------------------------------------

:- source(jsonl, pearl_trees, [
    description('Tree definitions from JSONL export'),
    columns([
        type,       % Record type: "Tree"
        tree_id,    % Numeric tree ID
        title,      % Tree title
        uri,        % Full URI
        cluster_id  % Cluster/parent URI
    ]),
    filter(type, "Tree"),  % Only Tree records
    target_config(python, [
        file_path('reports/pearltrees_targets_trees.jsonl'),
        streaming(true)
    ]),
    target_config(csharp, [
        file_path('reports/pearltrees_targets_trees.jsonl'),
        async_enumerable(true)
    ]),
    target_config(go, [
        file_path('reports/pearltrees_targets_trees.jsonl'),
        buffer_size(65536)
    ])
]).

%% --------------------------------------------------------------------
%% Source: API Response (runtime JSON)
%%
%% For browser automation / API fetching scenarios.
%% The API response is parsed at runtime, not from a file.
%% --------------------------------------------------------------------

:- source(json_runtime, pearl_api_response, [
    description('Pearltrees API response (runtime)'),
    columns([
        tree_id,     % Tree ID from response
        response     % Full JSON response object
    ]),
    % This source is populated at runtime via API call
    runtime_input(true)
]).

%% --------------------------------------------------------------------
%% Derived predicates for API response parsing
%% --------------------------------------------------------------------

%% pearl_from_api(?TreeId, ?PearlId, ?Type, ?Title, ?Url) is nondet.
%%   Extract individual pearls from an API response.
pearl_from_api(TreeId, PearlId, Type, Title, Url) :-
    pearl_api_response(TreeId, Response),
    json_path(Response, '$.pearls[*]', Pearl),
    json_get(Pearl, id, PearlId),
    json_get(Pearl, contentType, TypeCode),
    content_type_name(TypeCode, Type),
    json_get(Pearl, title, Title),
    (   json_path(Pearl, '$.url.url', Url)
    ->  true
    ;   Url = null
    ).

%% content_type_name(+Code, -Name) is det.
%%   Map Pearltrees contentType codes to readable names.
content_type_name(1, pagepearl).
content_type_name(2, collection).
content_type_name(4, root).
content_type_name(5, shortcut).
content_type_name(7, section).
