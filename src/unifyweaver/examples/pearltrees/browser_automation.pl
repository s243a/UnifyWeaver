%% pearltrees/browser_automation.pl - Abstract browser automation workflow
%%
%% Defines abstract predicates for browser-based data fetching.
%% Concrete API endpoints and commands come from external JSON config.
%%
%% This separation allows:
%% - UnifyWeaver to define the workflow logic
%% - Config file to specify API details (URLs, selectors, etc.)
%% - Different targets to implement the actual browser commands

:- module(pearltrees_browser_automation, [
    % Workflow steps (abstract)
    workflow_step/3,
    execute_workflow/3,

    % Config loading
    load_api_config/2,
    api_endpoint/3,
    api_param/4,

    % Abstract browser operations
    browser_navigate/2,
    browser_wait/2,
    browser_fetch_api/3,
    browser_evaluate/3,

    % Data extraction
    parse_tree_response/3,
    extract_pearls/2,

    % Utilities (exported for testing)
    expand_template/3,
    get_context/3,
    put_context/4,
    resolve_wait/3
]).

%% --------------------------------------------------------------------
%% Workflow Definition (Abstract)
%%
%% Workflows are sequences of steps. Each step has:
%% - Name (atom)
%% - Type (navigate, wait, fetch, parse)
%% - Parameters (from config or computed)
%% --------------------------------------------------------------------

%% workflow_step(+WorkflowName, +StepIndex, -Step) is nondet.
%%   Define workflow steps. Steps execute in order by index.

% Fetch tree workflow
workflow_step(fetch_tree, 1, step(navigate, tree_page, [])).
workflow_step(fetch_tree, 2, step(wait, page_load, [seconds(3)])).
workflow_step(fetch_tree, 3, step(fetch, tree_api, [])).
workflow_step(fetch_tree, 4, step(parse, tree_response, [])).

% Batch fetch workflow (multiple trees)
workflow_step(batch_fetch, 1, step(navigate, tree_page, [])).
workflow_step(batch_fetch, 2, step(wait, page_load, [seconds(3)])).
workflow_step(batch_fetch, 3, step(fetch, tree_api, [])).
workflow_step(batch_fetch, 4, step(parse, tree_response, [])).
workflow_step(batch_fetch, 5, step(delay, human_like, [])).

%% execute_workflow(+WorkflowName, +Config, +Context) is det.
%%   Execute all steps in a workflow with given config and context.
%%   Context contains runtime values like tree_id, account, etc.
execute_workflow(WorkflowName, Config, Context) :-
    findall(Idx-Step, workflow_step(WorkflowName, Idx, Step), Steps),
    keysort(Steps, Sorted),
    execute_steps(Sorted, Config, Context).

execute_steps([], _, _).
execute_steps([_Idx-Step|Rest], Config, Context) :-
    execute_step(Step, Config, Context, NewContext),
    execute_steps(Rest, Config, NewContext).

%% execute_step(+Step, +Config, +Context, -NewContext) is det.
%%   Execute a single workflow step.
execute_step(step(navigate, Target, _Params), Config, Context, Context) :-
    resolve_url(Target, Config, Context, Url),
    browser_navigate(Url, Context).

execute_step(step(wait, Type, Params), _Config, Context, Context) :-
    resolve_wait(Type, Params, Duration),
    browser_wait(Duration, Context).

execute_step(step(fetch, Endpoint, _Params), Config, Context, NewContext) :-
    resolve_endpoint(Endpoint, Config, Context, Url),
    browser_fetch_api(Url, Context, Response),
    put_context(response, Response, Context, NewContext).

execute_step(step(parse, Parser, _Params), _Config, Context, NewContext) :-
    get_context(response, Context, Response),
    parse_response(Parser, Response, Data),
    put_context(parsed_data, Data, Context, NewContext).

execute_step(step(delay, Type, _Params), Config, Context, Context) :-
    resolve_delay(Type, Config, Duration),
    browser_wait(Duration, Context).

%% --------------------------------------------------------------------
%% Config Loading
%%
%% API config is loaded from external JSON file.
%% Config structure:
%% {
%%   "endpoints": {
%%     "tree_api": {"url_template": "...", "method": "GET"},
%%     ...
%%   },
%%   "urls": {
%%     "tree_page": {"template": "https://.../{account}/{slug}/id{tree_id}"}
%%   },
%%   "timing": {
%%     "page_load_wait": 3,
%%     "human_like_delay": {"distribution": "gamma", "mean": 144, "shape": 2}
%%   }
%% }
%% --------------------------------------------------------------------

:- dynamic api_config_cache/2.

%% load_api_config(+FilePath, -Config) is det.
%%   Load API configuration from JSON file.
load_api_config(FilePath, Config) :-
    (   api_config_cache(FilePath, Cached)
    ->  Config = Cached
    ;   read_json_file(FilePath, Config),
        assertz(api_config_cache(FilePath, Config))
    ).

%% api_endpoint(+Config, +Name, -Endpoint) is semidet.
%%   Get endpoint definition from config.
api_endpoint(Config, Name, Endpoint) :-
    get_dict(endpoints, Config, Endpoints),
    get_dict(Name, Endpoints, Endpoint).

%% api_param(+Config, +Section, +Key, -Value) is semidet.
%%   Get a parameter from config section.
api_param(Config, Section, Key, Value) :-
    get_dict(Section, Config, SectionDict),
    get_dict(Key, SectionDict, Value).

%% --------------------------------------------------------------------
%% URL Resolution
%%
%% Resolve URL templates using context variables.
%% Template: "https://example.com/{account}/{slug}/id{tree_id}"
%% Context: _{account: "user", slug: "science", tree_id: "12345"}
%% Result: "https://example.com/user/science/id12345"
%% --------------------------------------------------------------------

resolve_url(Target, Config, Context, Url) :-
    api_param(Config, urls, Target, UrlDef),
    get_dict(template, UrlDef, Template),
    expand_template(Template, Context, Url).

resolve_endpoint(Endpoint, Config, Context, Url) :-
    api_endpoint(Config, Endpoint, EndpointDef),
    get_dict(url_template, EndpointDef, Template),
    expand_template(Template, Context, Url).

%% expand_template(+Template, +Context, -Expanded) is det.
%%   Replace {var} placeholders with context values.
expand_template(Template, Context, Expanded) :-
    atom_string(Template, TemplateStr),
    expand_vars(TemplateStr, Context, ExpandedStr),
    atom_string(Expanded, ExpandedStr).

expand_vars(Str, Context, Result) :-
    (   sub_string(Str, Before, _, After, "{"),
        sub_string(Str, _, _, AfterClose, "}"),
        AfterClose < After
    ->  % Found a variable
        sub_string(Str, 0, Before, _, Prefix),
        VarStart is Before + 1,
        sub_string(Str, VarStart, _, _, Rest),
        sub_string(Rest, VarLen, _, _, "}"),
        sub_string(Rest, 0, VarLen, _, VarName),
        atom_string(VarAtom, VarName),
        (   get_dict(VarAtom, Context, Value)
        ->  true
        ;   Value = ""
        ),
        format(string(ValueStr), "~w", [Value]),
        SuffixStart is VarStart + VarLen + 1,
        sub_string(Str, SuffixStart, _, 0, Suffix),
        string_concat(Prefix, ValueStr, Temp),
        string_concat(Temp, Suffix, NewStr),
        expand_vars(NewStr, Context, Result)
    ;   Result = Str
    ).

%% --------------------------------------------------------------------
%% Timing Resolution
%% --------------------------------------------------------------------

resolve_wait(page_load, Params, Duration) :-
    (   member(seconds(S), Params)
    ->  Duration = S
    ;   Duration = 3
    ).

resolve_wait(custom, Params, Duration) :-
    member(seconds(Duration), Params).

resolve_delay(human_like, Config, Duration) :-
    (   api_param(Config, timing, human_like_delay, DelayDef)
    ->  generate_delay(DelayDef, Duration)
    ;   Duration = 144  % Default mean
    ).

%% generate_delay(+DelayDef, -Duration) is det.
%%   Generate delay based on distribution config.
generate_delay(DelayDef, Duration) :-
    get_dict(distribution, DelayDef, Distribution),
    (   Distribution == "gamma"
    ->  get_dict(mean, DelayDef, Mean),
        get_dict(shape, DelayDef, Shape),
        % Placeholder - actual implementation depends on target
        Duration = Mean  % Would use gamma distribution
    ;   get_dict(mean, DelayDef, Duration)
    ).

%% --------------------------------------------------------------------
%% Context Helpers
%% --------------------------------------------------------------------

get_context(Key, Context, Value) :-
    get_dict(Key, Context, Value).

put_context(Key, Value, Context, NewContext) :-
    put_dict(Key, Context, Value, NewContext).

%% --------------------------------------------------------------------
%% Abstract Browser Operations (to be implemented per target)
%% --------------------------------------------------------------------

%% browser_navigate(+Url, +Context) is det.
%%   Navigate browser to URL. Implementation depends on target.
browser_navigate(Url, _Context) :-
    format('BROWSER_NAVIGATE: ~w~n', [Url]).

%% browser_wait(+Duration, +Context) is det.
%%   Wait for specified duration in seconds.
browser_wait(Duration, _Context) :-
    format('BROWSER_WAIT: ~w seconds~n', [Duration]).

%% browser_fetch_api(+Url, +Context, -Response) is det.
%%   Fetch data from API endpoint via browser.
browser_fetch_api(Url, _Context, Response) :-
    format('BROWSER_FETCH: ~w~n', [Url]),
    Response = _{status: 200, data: null}.  % Placeholder

%% browser_evaluate(+Script, +Context, -Result) is det.
%%   Evaluate JavaScript in browser context.
browser_evaluate(Script, _Context, Result) :-
    format('BROWSER_EVALUATE: ~w~n', [Script]),
    Result = null.  % Placeholder

%% --------------------------------------------------------------------
%% Response Parsing
%% --------------------------------------------------------------------

parse_response(tree_response, Response, Data) :-
    get_dict(data, Response, RawData),
    parse_tree_response(RawData, Data).

%% parse_tree_response(+RawData, -TreeInfo, -Pearls) is det.
%%   Parse API response into tree info and pearl list.
parse_tree_response(null, _{}, []) :- !.
parse_tree_response(RawData, TreeInfo, Pearls) :-
    (   is_dict(RawData)
    ->  extract_tree_info(RawData, TreeInfo),
        extract_pearls(RawData, Pearls)
    ;   TreeInfo = _{},
        Pearls = []
    ).

extract_tree_info(Data, TreeInfo) :-
    (   get_dict(tree, Data, Tree)
    ->  TreeInfo = Tree
    ;   TreeInfo = _{}
    ).

%% extract_pearls(+Data, -Pearls) is det.
%%   Extract pearl list from API response.
extract_pearls(Data, Pearls) :-
    (   get_dict(pearls, Data, PearlList),
        is_list(PearlList)
    ->  maplist(normalize_pearl, PearlList, Pearls)
    ;   Pearls = []
    ).

normalize_pearl(Raw, Pearl) :-
    (   is_dict(Raw)
    ->  dict_get_default(id, Raw, 0, Id),
        dict_get_default(contentType, Raw, 0, Type),
        dict_get_default(title, Raw, "", Title),
        Pearl = pearl(Id, Type, Title)
    ;   Pearl = pearl(0, 0, "")
    ).

%% dict_get_default(+Key, +Dict, +Default, -Value) is det.
%%   Get value from dict, or return default if key missing.
dict_get_default(Key, Dict, Default, Value) :-
    (   get_dict(Key, Dict, V)
    ->  Value = V
    ;   Value = Default
    ).

%% --------------------------------------------------------------------
%% Utility: Read JSON file
%% --------------------------------------------------------------------

read_json_file(FilePath, Data) :-
    (   exists_file(FilePath)
    ->  setup_call_cleanup(
            open(FilePath, read, Stream),
            json_read_dict(Stream, Data),
            close(Stream)
        )
    ;   Data = _{}
    ).

:- use_module(library(http/json)).
