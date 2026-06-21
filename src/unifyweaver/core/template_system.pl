:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% template_system.pl - Robust templating system for UnifyWeaver
% Provides named placeholder substitution and composable template units

:- module(template_system, [
    render_template/3,
    render_named_template/3,
    render_named_template/4,
    compose_templates/3,
    template/2,
    generate_transitive_closure/4,
    load_template/2,
    load_template/3,
    load_template_from_file/2,
    cache_template/2,
    clear_template_cache/0,
    clear_template_cache/1,
    template_config/2,
    set_template_config/2,
    template_config_default/1,
    set_template_config_default/1,
    test_template_system/0
]).

% Templates are defined throughout the file
:- discontiguous template/2.

:- use_module(library(lists)).
:- use_module(constraint_analyzer, [get_dedup_strategy/2]).

%% ============================================
%% DYNAMIC PREDICATES (Configuration & Cache)
%% ============================================

:- dynamic cached_template/3.        % cached_template(Name, Template, Timestamp)
:- dynamic template_config/2.        % template_config(Name, Options)
:- dynamic template_config_default/1. % template_config_default(Options)

%% Default configuration
template_config_default([
    source_order([generated]),       % Start conservative: only use hardcoded templates
    template_dir('templates'),
    cache_dir('templates/cache'),
    template_extension('.tmpl.sh'),
    auto_cache(false)
]).

%% ============================================
%% CONFIGURATION MANAGEMENT
%% ============================================

%% set_template_config_default(+Options)
%  Set global default configuration
set_template_config_default(Options) :-
    retractall(template_config_default(_)),
    assertz(template_config_default(Options)).

%% set_template_config(+TemplateName, +Options)
%  Set configuration for specific template
set_template_config(TemplateName, Options) :-
    retractall(template_config(TemplateName, _)),
    assertz(template_config(TemplateName, Options)).

%% get_template_config(+TemplateName, +RuntimeOptions, -MergedConfig)
%  Merge configuration from runtime > per-template > default
get_template_config(TemplateName, RuntimeOptions, MergedConfig) :-
    % Get default config
    template_config_default(DefaultConfig),

    % Get per-template config (if exists)
    (   template_config(TemplateName, TemplateConfig)
    ->  true
    ;   TemplateConfig = []
    ),

    % Merge: runtime overrides template overrides default
    merge_options(RuntimeOptions, TemplateConfig, Temp),
    merge_options(Temp, DefaultConfig, MergedConfig).

%% merge_options(+Priority, +Fallback, -Merged)
%  Merge two option lists, Priority takes precedence
merge_options([], Fallback, Fallback) :- !.
merge_options([H|T], Fallback, Merged) :-
    H =.. [Key, _],
    delete_option(Fallback, Key, Fallback2),
    merge_options(T, Fallback2, Temp),
    Merged = [H|Temp].

%% delete_option(+Options, +Key, -Remaining)
%  Remove all options with given key
delete_option([], _, []) :- !.
delete_option([H|T], Key, Remaining) :-
    H =.. [Key, _],
    !,
    delete_option(T, Key, Remaining).
delete_option([H|T], Key, [H|Remaining]) :-
    delete_option(T, Key, Remaining).

%% get_option(+OptionPattern, +Options, +Default, -Value)
%  Get option value from list, use default if not found
get_option(Key, Options, Default, Value) :-
    OptionPattern =.. [Key, Value],
    (   member(OptionPattern, Options)
    ->  true
    ;   Value = Default
    ).

%% ============================================
%% TEMPLATE CACHING
%% ============================================

%% cache_template(+Name, +Template)
%  Cache a template in memory with timestamp
cache_template(Name, Template) :-
    get_time(Timestamp),
    retractall(cached_template(Name, _, _)),
    assertz(cached_template(Name, Template, Timestamp)).

%% get_cached_template(+Name, -Template)
%  Retrieve cached template if exists
get_cached_template(Name, Template) :-
    cached_template(Name, Template, _).

%% clear_template_cache
%  Clear all cached templates
clear_template_cache :-
    retractall(cached_template(_, _, _)).

%% clear_template_cache(+Name)
%  Clear specific cached template
clear_template_cache(Name) :-
    retractall(cached_template(Name, _, _)).

%% ============================================
%% FILE OPERATIONS
%% ============================================

%% load_template_from_file(+FilePath, -TemplateString)
%  Load template from file
load_template_from_file(FilePath, TemplateString) :-
    exists_file(FilePath),
    read_file_to_string(FilePath, TemplateString, []).

%% find_template_file(+TemplateName, +Config, -FilePath)
%  Find template file path based on configuration
find_template_file(TemplateName, Config, FilePath) :-
    % Check for explicit file_path first
    (   get_option(file_path, Config, none, ExplicitPath),
        ExplicitPath \= none
    ->  FilePath = ExplicitPath
    ;   % Otherwise construct from template_dir + name + extension
        get_option(template_dir, Config, 'templates', Dir),
        get_option(template_extension, Config, '.tmpl.sh', Ext),
        atom_string(TemplateName, NameStr),
        format(string(FilePath), '~w/~w~w', [Dir, NameStr, Ext])
    ).

%% save_template_to_cache_file(+TemplateName, +Template, +Config)
%  Save template to cache directory for inspection/editing
save_template_to_cache_file(TemplateName, Template, Config) :-
    get_option(cache_dir, Config, 'templates/cache', CacheDir),
    get_option(template_extension, Config, '.tmpl.sh', Ext),

    % Create cache directory if it doesn't exist
    (   exists_directory(CacheDir)
    ->  true
    ;   make_directory(CacheDir)
    ),

    % Write template to cache file
    atom_string(TemplateName, NameStr),
    format(string(CachePath), '~w/~w~w', [CacheDir, NameStr, Ext]),
    open(CachePath, write, Stream),
    % Force Unix line endings (LF only) for bash scripts
    set_stream(Stream, newline(posix)),
    write(Stream, Template),
    close(Stream).

%% ============================================
%% TEMPLATE LOADING STRATEGIES
%% ============================================

%% try_source(+Name, +Strategy, +Config, -Template)
%  Try loading template from specific strategy
try_source(Name, file, Config, Template) :-
    find_template_file(Name, Config, Path),
    load_template_from_file(Path, Template).

try_source(Name, cached, _Config, Template) :-
    get_cached_template(Name, Template).

try_source(Name, generated, _Config, TemplateString) :-
    template(Name, TemplateList),
    (   is_list(TemplateList) ->
        atomic_list_concat(TemplateList, '\n', TemplateString)
    ;   TemplateString = TemplateList
    ).

%% load_template_with_strategy(+Name, +Config, -Template)
%  Load template using strategy preference order
load_template_with_strategy(Name, Config, Template) :-
    get_option(source_order, Config, [generated], Order),
    try_sources(Name, Order, Config, Template).

%% try_sources(+Name, +Strategies, +Config, -Template)
%  Try strategies in order until one succeeds
try_sources(Name, [Strategy|Rest], Config, Template) :-
    (   try_source(Name, Strategy, Config, Template)
    ->  % Strategy succeeded
        % Check if we should cache this template
        (   Strategy \= cached,  % Don't re-cache already cached template
            get_option(auto_cache, Config, false, AutoCache),
            AutoCache = true
        ->  cache_template(Name, Template),
            save_template_to_cache_file(Name, Template, Config)
        ;   true
        )
    ;   % Strategy failed, try next
        try_sources(Name, Rest, Config, Template)
    ).

%% ============================================
%% PUBLIC API
%% ============================================

%% load_template(+TemplateName, -TemplateString)
%  Load template using default configuration
load_template(TemplateName, TemplateString) :-
    load_template(TemplateName, [], TemplateString).

%% load_template(+TemplateName, +Options, -TemplateString)
%  Load template with runtime options
load_template(TemplateName, Options, TemplateString) :-
    get_template_config(TemplateName, Options, Config),
    load_template_with_strategy(TemplateName, Config, TemplateString).

%% render_named_template(+TemplateName, +Dict, -Result)
%  Load and render template by name
render_named_template(TemplateName, Dict, Result) :-
    render_named_template(TemplateName, Dict, [], Result).

%% render_named_template(+TemplateName, +Dict, +Options, -Result)
%  Load and render template by name with options
render_named_template(TemplateName, Dict, Options, Result) :-
    load_template(TemplateName, Options, Template),
    render_template(Template, Dict, Result).

%% ============================================
%% ORIGINAL TEMPLATE RENDERING (unchanged)
%% ============================================

%% Named placeholder substitution + section/match blocks.
%
% Supports a mustache subset plus match/case extension:
%   {{name}}                                       -- substitution
%   {{#flag}}...{{/flag}}                           -- truthy section
%   {{^flag}}...{{/flag}}                           -- inverted section
%   {{match key}}{{case v1}}...{{case v2}}...{{default}}...{{/match}}  -- match/case dispatch
%
% Section bodies may themselves contain {{name}} substitutions and other
% sections (with different tag names; see limitation below).
%
% Truthiness rules: a key is truthy when it appears in Dict and its value
% is not one of `false`, `0`, the empty string `""` or `''`, or the empty
% list `[]`. A key not in Dict is falsy. Any other value is truthy.
%
% Limitation: same-named sections cannot nest. {{#x}}{{#x}}...{{/x}}{{/x}}
% will close the outer section at the first {{/x}}. Different-named
% sections nest fine.
render_template(Template, Dict, Result) :-
    atom_string(Template, TStr),
    expand_match_blocks(TStr, Dict, MatchExpanded),
    expand_sections(MatchExpanded, Dict, Expanded),
    render_template_string(Expanded, Dict, Result).

% Fixed version using atom_string and sub_atom for reliable replacement
render_template_string(Template, [], Template) :- !.
render_template_string(Template, [Key=Value|Rest], Result) :-
    format(atom(Placeholder), '{{~w}}', [Key]),
    atom_string(Value, ValueStr),
    atom_string(Template, TemplateStr),
    atom_string(Placeholder, PlaceholderStr),
    replace_substring(TemplateStr, PlaceholderStr, ValueStr, Mid),
    render_template_string(Mid, Rest, Result).

% Helper predicate for substring replacement
replace_substring(String, Find, Replace, Result) :-
    string_length(Find, FindLen),
    (   sub_string(String, Before, FindLen, After, Find)
    ->  sub_string(String, 0, Before, _, Prefix),
        Start is Before + FindLen,
        sub_string(String, Start, After, 0, Suffix),
        replace_substring(Suffix, Find, Replace, RestResult),
        string_concat(Prefix, Replace, Part1),
        string_concat(Part1, RestResult, Result)
    ;   Result = String
    ).

%% ============================================
%% MUSTACHE SECTION EXPANSION
%% ============================================
%
% Pre-processing pass that handles {{#flag}}...{{/flag}} and
% {{^flag}}...{{/flag}} before the substitution pass runs.

%% expand_sections(+Template, +Dict, -Result)
%  Strips or keeps section blocks based on truthiness of dict entries.
expand_sections(Template, Dict, Result) :-
    (   find_first_section(Template, Kind, Tag, Before, Body, After)
    ->  (   keep_section_block(Kind, Tag, Dict)
        ->  expand_sections(Body, Dict, BodyExpanded),
            expand_sections(After, Dict, AfterExpanded),
            string_concat(Before, BodyExpanded, P1),
            string_concat(P1, AfterExpanded, Result)
        ;   expand_sections(After, Dict, AfterExpanded),
            string_concat(Before, AfterExpanded, Result)
        )
    ;   Result = Template
    ).

%% find_first_section(+Str, -Kind, -Tag, -Before, -Body, -After)
%  Locate the first {{#Tag}} or {{^Tag}} in Str and its matching
%  {{/Tag}}.  Body is the section's contents (markers excluded);
%  Before/After are the parts of Str outside the section.
find_first_section(Str, Kind, Tag, Before, Body, After) :-
    earliest_section_open(Str, Kind, Tag, OpenIdx, BodyStart),
    format(string(CloseMarker), "{{/~w}}", [Tag]),
    string_length(CloseMarker, CMLen),
    sub_string(Str, BodyStart, _, 0, BodyAndAfter),
    sub_string(BodyAndAfter, BodyEndRel, CMLen, _, CloseMarker),
    sub_string(BodyAndAfter, 0, BodyEndRel, _, Body),
    AfterStart is BodyEndRel + CMLen,
    sub_string(BodyAndAfter, AfterStart, _, 0, After),
    sub_string(Str, 0, OpenIdx, _, Before).

%% earliest_section_open(+Str, -Kind, -Tag, -OpenIdx, -BodyStart)
%  Find the earliest {{#TAG}} or {{^TAG}} in Str.  Returns Kind in
%  {section, inverted_section}, the absolute Index of the {{ at the
%  opener, and BodyStart, the absolute index just after the closing
%  }} of the open marker.
earliest_section_open(Str, Kind, Tag, OpenIdx, BodyStart) :-
    findall(I-K-T-B,
            ( ( K = section,          Marker = "{{#"
              ; K = inverted_section, Marker = "{{^"
              ),
              section_marker_at(Str, Marker, I, T, B)
            ),
            All),
    All \= [],
    sort(All, [OpenIdx-Kind-Tag-BodyStart|_]).

%% section_marker_at(+Str, +Marker, -OpenIdx, -Tag, -BodyStart)
%  Locate `{{#TAG}}` or `{{^TAG}}` in Str.  Marker is the 3-character
%  opener (e.g. "{{#").  TAG must be alphanumeric/underscore.
section_marker_at(Str, Marker, OpenIdx, Tag, BodyStart) :-
    string_length(Marker, MLen),
    sub_string(Str, OpenIdx, MLen, _, Marker),
    AfterMarker is OpenIdx + MLen,
    sub_string(Str, AfterMarker, _, 0, Tail),
    sub_string(Tail, EndRel, 2, _, "}}"),
    sub_string(Tail, 0, EndRel, _, TagStr),
    TagStr \= "",
    valid_tag_name(TagStr),
    atom_string(Tag, TagStr),
    BodyStart is AfterMarker + EndRel + 2.

%% valid_tag_name(+TagStr)
%  A tag is valid if every character is alnum or underscore.  This
%  prevents `{{#one}}…{{/two}}` from masquerading as `{{#one two}}`
%  or matching tag boundaries with whitespace.
valid_tag_name(TagStr) :-
    string_chars(TagStr, Chars),
    Chars \= [],
    forall(member(C, Chars), tag_char(C)).

tag_char(C) :- char_type(C, alnum), !.
tag_char('_').

%% keep_section_block(+Kind, +Tag, +Dict)
%  True when the section body should be rendered.  For Kind=section
%  this means Tag is truthy in Dict.  For Kind=inverted_section it
%  means Tag is falsy.
keep_section_block(section, Tag, Dict) :-
    truthy_in(Dict, Tag).
keep_section_block(inverted_section, Tag, Dict) :-
    \+ truthy_in(Dict, Tag).

%% truthy_in(+Dict, +Tag)
%  Tag is truthy in Dict if Dict contains `Tag=Value` and Value is
%  not falsy.  Falsy values: false, 0, "", '', [].
truthy_in(Dict, Tag) :-
    member(Tag=Value, Dict),
    \+ falsy_value(Value).

falsy_value(false).
falsy_value(0).
falsy_value("").
falsy_value('').
falsy_value([]).

%% ============================================
%% MATCH/CASE BLOCK EXPANSION
%% ============================================
%
% {{match key}}{{case val1}}body1{{case val2}}body2{{default}}fallback{{/match}}
%
% Dispatches on the value of `key` in the Dict. The first {{case}}
% whose value matches is rendered; if none match, {{default}} is
% rendered (or nothing if no default). Only exact string match is
% supported currently.
%
% Future: the case_matches/2 predicate can be extended to support
% glob patterns (e.g., lmdb_*), regex (PCRE2 via re_match/2), or
% structured patterns. Python and VB use regex in match statements;
% bash's case uses glob. The predicate is factored out to make this
% extension straightforward.

%% expand_match_blocks(+Template, +Dict, -Result)
%  Find and expand all {{match key}}...{{/match}} blocks in Template.
%  Runs before section expansion so match blocks can contain sections.
expand_match_blocks(Template, Dict, Result) :-
    (   find_match_block(Template, Key, Before, MatchBody, After)
    ->  parse_match_cases(MatchBody, Cases, Default),
        resolve_match(Key, Dict, Cases, Default, Rendered),
        expand_match_blocks(Rendered, Dict, RenderedExpanded),
        expand_match_blocks(After, Dict, AfterExpanded),
        string_concat(Before, RenderedExpanded, P1),
        string_concat(P1, AfterExpanded, Result)
    ;   Result = Template
    ).

%% find_match_block(+Str, -Key, -Before, -Body, -After)
%  Locate the first {{match key}} in Str and its balanced {{/match}}.
%  Handles nested match blocks via depth counting.
find_match_block(Str, Key, Before, Body, After) :-
    sub_string(Str, OpenIdx, 8, _, "{{match "),
    AfterOpen is OpenIdx + 8,
    sub_string(Str, AfterOpen, _, 0, Tail),
    sub_string(Tail, EndRel, 2, _, "}}"),
    sub_string(Tail, 0, EndRel, _, KeyStr),
    KeyStr \= "",
    atom_string(Key, KeyStr),
    BodyStart is AfterOpen + EndRel + 2,
    sub_string(Str, BodyStart, _, 0, BodyAndAfter),
    find_balanced_match_close(BodyAndAfter, 1, 0, CloseRel),
    sub_string(BodyAndAfter, 0, CloseRel, _, Body),
    CloseEnd is CloseRel + 10,
    sub_string(BodyAndAfter, CloseEnd, _, 0, After),
    sub_string(Str, 0, OpenIdx, _, Before).

%% find_balanced_match_close(+Str, +Depth, +Pos, -ClosePos)
%  Scan Str from Pos looking for the balanced {{/match}} at Depth=0.
%  Depth starts at 1 (for the already-opened outer match block).
find_balanced_match_close(Str, Depth, Pos, ClosePos) :-
    sub_string(Str, Pos, _, 0, Rest),
    (   Depth =:= 0
    ->  ClosePos = Pos
    ;   Rest \= "",
        next_match_event(Rest, EventType, EventOffset, EventLen),
        AbsOffset is Pos + EventOffset,
        NextPos is AbsOffset + EventLen,
        (   EventType = open
        ->  Depth1 is Depth + 1,
            find_balanced_match_close(Str, Depth1, NextPos, ClosePos)
        ;   EventType = close,
            Depth1 is Depth - 1,
            (   Depth1 =:= 0
            ->  ClosePos = AbsOffset
            ;   find_balanced_match_close(Str, Depth1, NextPos, ClosePos)
            )
        )
    ).

%% next_match_event(+Str, -Type, -Offset, -Len)
%  Find the nearest {{match ...}} (open) or {{/match}} (close) in Str.
next_match_event(Str, Type, Offset, Len) :-
    (   sub_string(Str, OpenIdx, 8, _, "{{match ") -> true ; OpenIdx = 999999999 ),
    (   sub_string(Str, CloseIdx, 10, _, "{{/match}}") -> true ; CloseIdx = 999999999 ),
    OpenIdx + CloseIdx < 1999999998,  % at least one must exist
    (   OpenIdx < CloseIdx
    ->  Type = open, Offset = OpenIdx, Len = 8
    ;   Type = close, Offset = CloseIdx, Len = 10
    ).

%% parse_match_cases(+Body, -Cases, -Default)
%  Parse the body of a match block into a list of case(Value, CaseBody)
%  and an optional Default body. Cases appear as {{case value}}...
%  and {{default}}... The last segment before {{case}}/{{default}}/end
%  is the case body.
parse_match_cases(Body, Cases, Default) :-
    split_match_segments(Body, Segments),
    extract_cases(Segments, Cases, Default).

%% split_match_segments(+Body, -Segments)
%  Split match body into segments at {{case ...}} and {{default}} markers.
%  Returns a list of segment(Type, Value, Content) terms where Type is
%  'case_seg' or 'default_seg', Value is the case value (or '' for default),
%  and Content is the text content.
split_match_segments(Body, Segments) :-
    split_match_segments_(Body, [], Segments).

split_match_segments_("", Acc, Segments) :-
    reverse(Acc, Segments).
split_match_segments_(Body, Acc, Segments) :-
    Body \= "",
    (   find_next_case_marker(Body, Type, Value, Before, After)
    ->  (   Before \= "", Acc = []
        ->  % Leading text before first case — discard (whitespace/comments)
            split_match_segments_(After, [segment(Type, Value, "")|Acc], Segments)
        ;   Before \= "", Acc = [segment(PrevType, PrevVal, _)|RestAcc]
        ->  % Attach Before as content of previous segment
            split_match_segments_(After, [segment(Type, Value, "")|[segment(PrevType, PrevVal, Before)|RestAcc]], Segments)
        ;   split_match_segments_(After, [segment(Type, Value, "")|Acc], Segments)
        )
    ;   % No more markers — remaining text is content of last segment
        (   Acc = [segment(PrevType, PrevVal, _)|RestAcc]
        ->  reverse([segment(PrevType, PrevVal, Body)|RestAcc], Segments)
        ;   Segments = []  % no cases at all
        )
    ).

%% find_next_case_marker(+Str, -Type, -Value, -Before, -After)
%  Find the next top-level {{case value}} or {{default}} in Str,
%  skipping any that are inside nested {{match}}...{{/match}} blocks.
find_next_case_marker(Str, Type, Value, Before, After) :-
    find_next_case_marker_(Str, 0, 0, Type, Value, MarkerIdx, AfterIdx),
    sub_string(Str, 0, MarkerIdx, _, Before),
    sub_string(Str, AfterIdx, _, 0, After).

%% find_next_case_marker_(+Str, +Pos, +Depth, -Type, -Value, -MarkerIdx, -AfterIdx)
%  Scan from Pos at match nesting Depth for the next depth-0 case/default.
find_next_case_marker_(Str, Pos, Depth, Type, Value, MarkerIdx, AfterIdx) :-
    sub_string(Str, Pos, _, 0, Rest),
    Rest \= "",
    find_earliest_tag(Rest, TagType, TagOffset, TagVal, TagAfterRel),
    AbsIdx is Pos + TagOffset,
    AbsAfter is Pos + TagAfterRel,
    (   TagType = match_open
    ->  Depth1 is Depth + 1,
        find_next_case_marker_(Str, AbsAfter, Depth1, Type, Value, MarkerIdx, AfterIdx)
    ;   TagType = match_close
    ->  Depth1 is Depth - 1,
        (   Depth1 >= 0
        ->  find_next_case_marker_(Str, AbsAfter, Depth1, Type, Value, MarkerIdx, AfterIdx)
        ;   fail
        )
    ;   (TagType = case_tag ; TagType = default_tag),
        (   Depth =:= 0
        ->  (TagType = case_tag -> Type = case_seg ; Type = default_seg),
            Value = TagVal, MarkerIdx = AbsIdx, AfterIdx = AbsAfter
        ;   find_next_case_marker_(Str, AbsAfter, Depth, Type, Value, MarkerIdx, AfterIdx)
        )
    ).

%% find_earliest_tag(+Str, -TagType, -Offset, -Value, -AfterOffset)
%  Find the earliest of {{match }}, {{/match}}, {{case }}, {{default}} in Str.
find_earliest_tag(Str, TagType, Offset, Value, AfterOffset) :-
    findall(Idx-Type-Val-After,
            ( tag_candidate(Str, Idx, Type, Val, After) ),
            Candidates),
    Candidates \= [],
    sort(Candidates, [Offset-TagType-Value-AfterOffset|_]).

tag_candidate(Str, Idx, match_open, '', After) :-
    sub_string(Str, Idx, 8, _, "{{match "),
    Start is Idx + 8,
    sub_string(Str, Start, _, 0, Tail),
    sub_string(Tail, EndRel, 2, _, "}}"),
    After is Start + EndRel + 2.
tag_candidate(Str, Idx, match_close, '', After) :-
    sub_string(Str, Idx, 10, _, "{{/match}}"),
    After is Idx + 10.
tag_candidate(Str, Idx, case_tag, Val, After) :-
    sub_string(Str, Idx, 7, _, "{{case "),
    Start is Idx + 7,
    sub_string(Str, Start, _, 0, Tail),
    sub_string(Tail, EndRel, 2, _, "}}"),
    sub_string(Tail, 0, EndRel, _, ValStr),
    atom_string(Val, ValStr),
    After is Start + EndRel + 2.
tag_candidate(Str, Idx, default_tag, '', After) :-
    sub_string(Str, Idx, 11, _, "{{default}}"),
    After is Idx + 11.

find_case_at(Str, Idx, Value, AfterIdx) :-
    sub_string(Str, Idx, 7, _, "{{case "),
    Start is Idx + 7,
    sub_string(Str, Start, _, 0, Tail),
    sub_string(Tail, EndRel, 2, _, "}}"),
    sub_string(Tail, 0, EndRel, _, ValStr),
    atom_string(Value, ValStr),
    AfterIdx is Start + EndRel + 2.

find_default_at(Str, Idx, AfterIdx) :-
    sub_string(Str, Idx, 11, _, "{{default}}"),
    AfterIdx is Idx + 11.

%% extract_cases(+Segments, -Cases, -Default)
extract_cases([], [], "").
extract_cases(Segments, Cases, Default) :-
    Segments \= [],
    include(is_case_seg, Segments, CaseSegs),
    maplist(seg_to_case, CaseSegs, Cases),
    (   member(segment(default_seg, _, DefBody), Segments)
    ->  Default = DefBody
    ;   Default = ""
    ).

is_case_seg(segment(case_seg, _, _)).
seg_to_case(segment(case_seg, Val, Body), case(Val, Body)).

%% resolve_match(+Key, +Dict, +Cases, +Default, -Rendered)
%  Find the first matching case and return its body, or Default.
resolve_match(Key, Dict, Cases, Default, Rendered) :-
    (   member(Key=DictValue, Dict),
        atom_string(DictValue, DictValueStr),
        member(case(CaseValue, CaseBody), Cases),
        atom_string(CaseValue, CaseValueStr),
        case_matches(DictValueStr, CaseValueStr)
    ->  Rendered = CaseBody
    ;   Rendered = Default
    ).

%% case_matches(+ActualValue, +PatternValue)
%  Currently: exact string match.
%  Future extensions (see philosophy note above):
%    - Glob: case_matches_glob/2 using wildcard expansion
%    - Regex: case_matches_regex/2 using re_match/2 (PCRE2)
%    - Structured: case_matches_term/2 for Prolog term patterns
case_matches(Value, Pattern) :- Value = Pattern.

%% Compose multiple templates into one
compose_templates([], _, "") :- !.
compose_templates([Name|Rest], Dict, Result) :-
    % Name is a template name (atom); render with default options
    render_named_template(Name, Dict, [source_order([generated, file])], R1),
    compose_templates(Rest, Dict, Rs),
    string_concat(R1, Rs, Result).

%% ============================================
%% BASH CODE GENERATION TEMPLATES
%% ============================================

%% Header template
template(bash_header, '#!/bin/bash
# {{description}}
').

%% Function definition template
template(function, '
{{name}}() {
{{body}}
}').

%% WAM Module template
template(wam_module, ';;; WAM Module: {{module_name}}
;;; Generated by UnifyWeaver WAM Target
;;; Target: {{target_name}}
;;; Date: {{date}}

{{predicates_code}}

;;; End of WAM Module
').

%% Rust WAM Cargo.toml template
%%
%% LMDB crate selection is mutually exclusive — driven by
%% `lmdb_crate(lmdb_zero | heed | auto)` codegen option.  See
%% docs/design/WAM_RUST_LMDB_CRATE_DECISION.md.
%%
%%   {{#use_lmdb_zero}}lmdb-zero{{/use_lmdb_zero}}  - the default
%%   {{#use_heed}}heed{{/use_heed}}                 - opt-in alternative
%%
%% At most one of these flags is true at a time; both are false when
%% `lmdb_mode(none)`.
%%
%% Inline template is authoritative; templates/targets/rust_wam/Cargo.toml.mustache
%% is a companion mirror kept for future migration to file-based templates.
template(rust_wam_cargo, '[package]
name = "{{module_name}}"
version = "0.1.0"
edition = "2021"
description = "Generated by UnifyWeaver WAM-to-Rust transpilation"

[[bin]]
name = "bench"
path = "src/main.rs"

[dependencies]
{{#use_rayon}}rayon = "1"
{{/use_rayon}}{{#use_lmdb_zero}}lmdb-zero = "0.4"
{{/use_lmdb_zero}}{{#use_heed}}heed = "0.20"
{{/use_heed}}').

%% Rust WAM lib.rs template
%% lmdb_fact_source module is declared when either LMDB crate is active
%% (use_lmdb_zero or use_heed); the generated lmdb_fact_source.rs uses
%% the corresponding crate.  Wiring its callers into main.rs is R2/R4 work.
template(rust_wam_lib, '// Generated by UnifyWeaver WAM-to-Rust transpilation
// Module: {{module_name}}
// Date: {{date}}

pub mod value;
pub mod instructions;
pub mod state;
pub mod par_aggregate;
pub mod boundary_cache;
{{#use_lmdb_zero}}pub mod lmdb_fact_source;
{{/use_lmdb_zero}}{{#use_heed}}pub mod lmdb_fact_source;
{{/use_heed}}{{#use_csr}}pub mod csr_fact_source;
{{/use_csr}}
use std::collections::{HashMap, HashSet};
use value::Value;
use instructions::Instruction;
use state::WamState;

{{predicates_code}}
').

%% Stream check template - checks if function exists
template(stream_check, '
# Check if {{base}}_stream or {{base}} exists
{{base}}_get_stream() {
    if declare -f {{base}}_stream >/dev/null 2>&1; then
        {{base}}_stream
    elif declare -f {{base}} >/dev/null 2>&1; then
        {{base}}
    else
        echo "Error: neither {{base}}_stream nor {{base}} found" >&2
        return 1
    fi
}').

%% BFS initialization template
template(bfs_init, '
    local start="$1"
    declare -A visited
    declare -A output_seen
    local queue_file="/tmp/{{prefix}}_queue_{{tmp_suffix}}"
    local next_queue="/tmp/{{prefix}}_next_{{tmp_suffix}}"
    trap "rm -f $queue_file $next_queue" {{trap_signals}}
    echo "$start" > "$queue_file"
    visited["$start"]=1').

%% BFS loop template
template(bfs_loop, '
    while [[ -s "$queue_file" ]]; do
        > "$next_queue"
        
        while IFS= read -r current; do
            {{source}}_get_stream | grep "^$current:" | while IFS=":" read -r from to; do
                if [[ "$from" == "$current" && -z "${visited[$to]}" ]]; then
                    visited["$to"]=1
                    echo "$to" >> "$next_queue"
                    echo "$start:$to"
                fi
            done
        done < "$queue_file"
        
        mv "$next_queue" "$queue_file"
    done
    
    rm -f "$queue_file" "$next_queue"').

%% All nodes finder template
template(all_nodes, '
{{name}}_all() {
{{bfs_init}}
{{bfs_loop}}
}').

%% Check function template
template(check_function, '
{{name}}_check() {
    local start="$1"
    local target="$2"
    {{name}}_all "$start" | grep -q "^$start:$target$" && echo "$start:$target"
}').

%% Stream wrapper template
template(stream_wrapper, '
{{name}}_stream() {
    local arg="$1"
    if [[ -n "$arg" ]]; then
        {{name}}_all "$arg"
    else
        {{source}}_get_stream | while IFS=":" read -r start _; do
            echo "$start"
        done | sort -u | while read -r start; do
            {{name}}_all "$start"
        done | sort -u
    fi
}').

%% Generate complete transitive closure implementation
generate_transitive_closure(PredName, BaseName, Options, Code) :-
    atom_string(PredName, PredStr),
    atom_string(BaseName, BaseStr),
    constraint_analyzer:get_dedup_strategy(Options, Strategy),

    % Determine shell capabilities for portable code generation.
    % Options can include shell_env([no_syscalls, no_processes, ...]) to
    % declare missing capabilities. On WASM (auto-detected via
    % shell_capabilities module), these default to the restricted set.
    (   member(shell_env(EnvList), Options), is_list(EnvList)
    ->  true
    ;   detect_shell_env(EnvList)
    ),

    % When syscalls are unavailable, avoid $$ (getpid hangs on WASM).
    (   member(no_syscalls, EnvList)
    ->  TempSuffix = "$RANDOM",
        TrapSignals = "EXIT"
    ;   TempSuffix = "$$",
        TrapSignals = "EXIT"
    ),

    % When process features are limited, avoid tee >(cmd) and use
    % a sequential pipeline instead.
    (   member(no_processes, EnvList)
    ->  UseSequentialCheck = true
    ;   UseSequentialCheck = false
    ),

    % Build the _check() function body based on capabilities.
    atom_string(TempSuffix, TempSuffixStr),
    atom_string(TrapSignals, TrapSignalsStr),
    (   UseSequentialCheck = true
    ->  CheckBody = '
    local _result="/tmp/{{pred}}_result_{{tmp_suffix}}"
    {{pred}}_all "$start" 2>/dev/null > "$_result"
    if grep -q "^$start:$target$" "$_result"; then
        rm -f "$_result"
        return 0
    else
        rm -f "$_result"
        return 1
    fi'
    ;   CheckBody = '
    local tmpflag="/tmp/{{pred}}_found_{{tmp_suffix}}"
    {{pred}}_all "$start" 2>/dev/null |
    tee >(grep -q "^$start:$target$" && touch "$tmpflag") >/dev/null 2>&1
    if [[ -f "$tmpflag" ]]; then
        rm -f "$tmpflag"
        return 0
    else
        rm -f "$tmpflag"
        return 1
    fi'
    ),

    Template = '#!/bin/bash
# {{pred}} - transitive closure of {{base}}

# Check for base stream function
{{base}}_get_stream() {
    if declare -f {{base}}_stream >/dev/null 2>&1; then
        {{base}}_stream
    elif declare -f {{base}} >/dev/null 2>&1; then
        {{base}}
    else
        echo "Error: {{base}} not found" >&2
        return 1
    fi
}

# Main logic to find all descendants
{{pred}}_all() {
    local start="$1"
    declare -A visited
    local queue_file="/tmp/{{pred}}_queue_{{tmp_suffix}}"
    local next_queue="/tmp/{{pred}}_next_{{tmp_suffix}}"

    trap "rm -f $queue_file $next_queue" {{trap_signals}}

    echo "$start" > "$queue_file"
    visited["$start"]=1

    while [[ -s "$queue_file" ]]; do
        > "$next_queue"

        while IFS= read -r current; do
            while IFS=":" read -r from to; do
                if [[ "$from" == "$current" && -z "${visited[$to]}" ]]; then
                    visited["$to"]=1
                    echo "$to" >> "$next_queue"
                    echo "$start:$to"
                fi
            done < <({{base}}_get_stream | grep "^$current:")
        done < "$queue_file"

        mv "$next_queue" "$queue_file"
    done

    rm -f "$queue_file" "$next_queue"
}

# Check specific relationship
{{pred}}_check() {
    local start="$1"
    local target="$2"
{{check_body}}
}

# Main entry point
{{pred}}() {
    local start="$1"
    local target="$2"

    if [[ -z "$target" ]]; then
        # One-argument call: find all descendants
        if [[ "{{strategy}}" == "sort_u" ]]; then
            {{pred}}_all "$start" | sort -u
        elif [[ "{{strategy}}" == "hash_dedup" ]]; then
            declare -A seen
            {{pred}}_all "$start" | while IFS= read -r line; do
                if [[ -z "${seen[$line]}" ]]; then
                    seen[$line]=1
                    echo "$line"
                fi
            done
        else
            {{pred}}_all "$start"
        fi
    else
        # Two-argument call: check relationship
        if {{pred}}_check "$start" "$target"; then
            echo "$start:$target"
            return 0
        else
            return 1
        fi
    fi
}',

    % NOTE on key order: `check_body` is a string built above that
    % itself contains `{{pred}}` and `{{tmp_suffix}}` placeholders.
    % render_template_string/3 is a single-pass iterator over this
    % dict — once a key is substituted, its inserted content is not
    % re-scanned for later keys.  Keeping `check_body` FIRST means the
    % inner placeholders it introduces are still present when `pred`
    % and `tmp_suffix` come around in the same pass.  Without this
    % ordering the generated `_check()` body literally calls
    % `{{pred}}_all` and creates `/tmp/{{pred}}_found_{{tmp_suffix}}`,
    % causing tests/core/test_compiler_driver.pl to fail because the
    % generated bash script bombs out at `{{pred}}_all`.
    render_template(Template, [
        check_body = CheckBody,
        pred = PredStr,
        base = BaseStr,
        strategy = Strategy,
        tmp_suffix = TempSuffixStr,
        trap_signals = TrapSignalsStr
    ], Code).

%% Deduplication wrapper template
template(dedup_wrapper, [
'#!/bin/bash
# {{pred}} - generated with {{strategy}} deduplication

{{main_code}}

# Main entry point with deduplication
{{pred}}() {
    if [[ "{{strategy}}" == "sort_u" ]]; then
        {{pred}}_all "$@" | sort -u
    fi
    if [[ "{{strategy}}" == "hash_dedup" ]]; then
        declare -A seen
        {{pred}}_all "$@" | while IFS= read -r line; do
            if [[ -z "${seen[$line]}" ]]; then
                seen[$line]=1
                echo "$line"
            fi
        done
    fi
    if [[ "{{strategy}}" != "sort_u" && "{{strategy}}" != "hash_dedup" ]]; then
        {{pred}}_all "$@"
    fi
}'
]).

%% ============================================
%% XML FIELD EXTRACTION TEMPLATES
%% ============================================

%% AWK field extraction template
template(xml_awk_field_extraction, [
"#!/bin/bash",
"# {{pred}} - Field extraction from {{file}}",
"",
"{{pred}}() {",
"    # Extract XML elements, then extract fields",
"    awk -f scripts/utils/select_xml_elements.awk -v tag=\"{{tag}}\" {{file}} 2>/dev/null | \\",
"    awk -f {{awk_script}}",
"}",
"",
"# Invoke if executed directly",
"if [[ \"${BASH_SOURCE[0]}\" == \"${0}\" ]]; then",
"    {{pred}} \"$@\"",
"fi",
""
]).

%% ============================================
%% FACTS TEMPLATES (non-recursive predicates)
%% ============================================

% Common header
template('bash/header', [
"#!/bin/bash",
"# {{pred}} - fact lookup",
""
]).

% Data arrays
template('facts/array_unary', [
"{{pred}}_data=(",
"{{entries}}",
")",
""
]).

template('facts/array_binary', [
"declare -A {{pred}}_data=(",
"{{entries}}",
")",
""
]).

% Lookup functions
template('facts/lookup_unary', [
"{{pred}}() {",
"  local query=\"$1\"",
"  for item in \"${{{pred}}_data[@]}\"; do",
"    [[ \"$item\" == \"$query\" ]] && echo \"$item\"",
"  done",
"}",
""
]).

template('facts/lookup_binary', [
"{{pred}}() {",
"  if [[ -n \"$2\" ]]; then",
"    local key=\"$1:$2\"",
"    [[ -n \"${{{pred}}_data[$key]}\" ]] && echo \"$key\"",
"  else",
"    local _m=1",
"    for key in \"${!{{pred}}_data[@]}\"; do",
"      if [[ \"$key\" == \"$1:\"* ]]; then echo \"$key\"; _m=0; fi",
"    done",
"    return $_m",
"  fi",
"}",
""
]).

% Stream functions
template('facts/stream_unary', [
"{{pred}}_stream() {",
"  for item in \"${{{pred}}_data[@]}\"; do",
"    echo \"$item\"",
"  done",
"}",
""
]).

template('facts/stream_binary', [
"{{pred}}_stream() {",
"  for key in \"${!{{pred}}_data[@]}\"; do",
"    echo \"$key\"",
"  done",
"}",
""
]).

% Reverse stream for binary facts
template('facts/reverse_stream_binary', [
"{{pred}}_reverse_stream() {",
"  for key in \"${!{{pred}}_data[@]}\"; do",
"    IFS=\":\" read -r a b <<< \"$key\"",
"    echo \"$b:$a\"",
"  done",
"}",
""
]).

%% Test the template system
test_template_system :-
    writeln('=== Testing Template System ==='),

    % Test 1: Simple substitution
    write('Test 1 - Simple substitution: '),
    render_template('Hello {{name}}!', [name='World'], R1),
    (sub_string(R1, _, _, _, 'Hello World!') -> writeln('PASS') ; (format('FAIL: got ~w~n', [R1]), fail)),

    % Test 2: Multiple substitutions
    write('Test 2 - Multiple substitutions: '),
    render_template('{{greeting}} {{name}}', [greeting='Hello', name='Alice'], R2),
    (sub_string(R2, _, _, _, 'Hello Alice') -> writeln('PASS') ; (format('FAIL: got ~w~n', [R2]), fail)),

    % Test 3: Generate transitive closure
    writeln('Test 3 - Generate transitive closure:'),
    generate_transitive_closure(ancestor, parent, [], Code3),
    (   sub_string(Code3, _, _, _, 'ancestor_all')
    ->  writeln('PASS - contains ancestor_all function')
    ;   writeln('FAIL - missing expected function')
    ),

    % Test 4: Template caching
    write('Test 4 - Template caching: '),
    cache_template(test_cached, 'Cached {{value}}'),
    get_cached_template(test_cached, Cached),
    (Cached = 'Cached {{value}}' -> writeln('PASS') ; (format('FAIL: got ~w~n', [Cached]), fail)),

    % Test 5: Load generated template by name
    write('Test 5 - Load by name (generated): '),
    load_template(bash_header, Template5),
    (   sub_string(Template5, _, _, _, '#!/bin/bash')
    ->  writeln('PASS')
    ;   (format('FAIL: got ~w~n', [Template5]), fail)
    ),

    % Test 6: Render named template
    write('Test 6 - Render named template: '),
    render_named_template(bash_header, [description='Test Script'], Result6),
    (   sub_string(Result6, _, _, _, '# Test Script')
    ->  writeln('PASS')
    ;   (format('FAIL: got ~w~n', [Result6]), fail)
    ),

    % Test 7: Configuration merging
    write('Test 7 - Configuration merging: '),
    get_template_config(test_template, [source_order([file])], Config7),
    member(source_order([file]), Config7),
    writeln('PASS'),

    % Test 8: Truthy section keeps body
    write('Test 8 - Section, truthy: '),
    render_template('A{{#flag}}B{{/flag}}C', [flag=true], R8),
    (sub_string(R8, _, _, _, 'ABC') -> writeln('PASS') ; (format('FAIL: got ~w~n', [R8]), fail)),

    % Test 9: Falsy section drops body
    write('Test 9 - Section, falsy: '),
    render_template('A{{#flag}}B{{/flag}}C', [flag=false], R9),
    (sub_string(R9, _, _, _, 'AC'), \+ sub_string(R9, _, _, _, 'B')
     -> writeln('PASS') ; (format('FAIL: got ~w~n', [R9]), fail)),

    % Test 10: Missing key is falsy
    write('Test 10 - Missing key is falsy: '),
    render_template('A{{#absent}}B{{/absent}}C', [], R10),
    (sub_string(R10, _, _, _, 'AC'), \+ sub_string(R10, _, _, _, 'B')
     -> writeln('PASS') ; (format('FAIL: got ~w~n', [R10]), fail)),

    % Test 11: Inverted section, falsy renders
    write('Test 11 - Inverted section, falsy: '),
    render_template('A{{^flag}}B{{/flag}}C', [flag=false], R11),
    (sub_string(R11, _, _, _, 'ABC') -> writeln('PASS') ; (format('FAIL: got ~w~n', [R11]), fail)),

    % Test 12: Inverted section, truthy hides
    write('Test 12 - Inverted section, truthy: '),
    render_template('A{{^flag}}B{{/flag}}C', [flag=true], R12),
    (sub_string(R12, _, _, _, 'AC'), \+ sub_string(R12, _, _, _, 'B')
     -> writeln('PASS') ; (format('FAIL: got ~w~n', [R12]), fail)),

    % Test 13: Empty string is falsy
    write('Test 13 - Empty string is falsy: '),
    render_template('A{{#flag}}B{{/flag}}C', [flag=""], R13),
    (sub_string(R13, _, _, _, 'AC'), \+ sub_string(R13, _, _, _, 'B')
     -> writeln('PASS') ; (format('FAIL: got ~w~n', [R13]), fail)),

    % Test 14: Substitution inside truthy section
    write('Test 14 - Substitution inside section: '),
    render_template('Hello {{#greet}}{{name}}{{/greet}}!', [greet=true, name='World'], R14),
    (sub_string(R14, _, _, _, 'Hello World!') -> writeln('PASS')
     ; (format('FAIL: got ~w~n', [R14]), fail)),

    % Test 15: Nested differently-named sections
    write('Test 15 - Nested sections: '),
    render_template('[{{#a}}A{{#b}}B{{/b}}{{/a}}]', [a=true, b=true], R15a),
    render_template('[{{#a}}A{{#b}}B{{/b}}{{/a}}]', [a=true, b=false], R15b),
    render_template('[{{#a}}A{{#b}}B{{/b}}{{/a}}]', [a=false, b=true], R15c),
    (sub_string(R15a, _, _, _, '[AB]'),
     sub_string(R15b, _, _, _, '[A]'), \+ sub_string(R15b, _, _, _, 'B'),
     sub_string(R15c, _, _, _, '[]'),  \+ sub_string(R15c, _, _, _, 'A'),
     \+ sub_string(R15c, _, _, _, 'B')
     -> writeln('PASS')
     ; (format('FAIL: a=t b=t got ~w~n  a=t b=f got ~w~n  a=f b=t got ~w~n',
               [R15a, R15b, R15c]), fail)),

    % Test 16: Backward compatibility — existing pure-substitution
    % templates still render identically (no { #/^/ } syntax in them).
    write('Test 16 - No section syntax = unchanged: '),
    render_template('plain {{x}} plain {{y}} plain', [x='X', y='Y'], R16),
    (sub_string(R16, _, _, _, 'plain X plain Y plain')
     -> writeln('PASS') ; (format('FAIL: got ~w~n', [R16]), fail)),

    % === Match/Case Edge Case Tests (sections 3.1-3.8) ===

    % Test 17: Section inside case body (3.1)
    write('Test 17 - Section inside case body: '),
    render_template("{{match mode}}{{case cached}}{{#has_l2}}L2 on{{/has_l2}}{{^has_l2}}L2 off{{/has_l2}}{{case eager}}no cache{{/match}}",
                    [mode=cached, has_l2=true], R17),
    (   sub_string(R17, _, _, _, "L2 on"), \+ sub_string(R17, _, _, _, "L2 off")
    ->  writeln('PASS')
    ;   (format('FAIL: got ~w~n', [R17]), fail)
    ),

    % Test 18: Nested match blocks (3.2)
    write('Test 18 - Nested match blocks: '),
    render_template("{{match outer}}{{case a}}{{match inner}}{{case x}}AX{{case y}}AY{{/match}}{{case b}}B{{/match}}",
                    [outer=a, inner=y], R18),
    (   sub_string(R18, _, _, _, "AY")
    ->  writeln('PASS')
    ;   (format('FAIL: got ~w~n', [R18]), fail)
    ),

    % Test 19: Nested match, outer selects non-nested case (3.2 variant)
    write('Test 19 - Nested match, outer=b: '),
    render_template("{{match outer}}{{case a}}{{match inner}}{{case x}}AX{{/match}}{{case b}}B{{/match}}",
                    [outer=b], R19),
    (   sub_string(R19, _, _, _, "B"), \+ sub_string(R19, _, _, _, "AX")
    ->  writeln('PASS')
    ;   (format('FAIL: got ~w~n', [R19]), fail)
    ),

    % Test 20: Underscores in case values (3.3)
    write('Test 20 - Underscores in case values: '),
    render_template("{{match backend}}{{case lmdb_offset}}LMDB{{case sorted_array}}SORTED{{/match}}",
                    [backend=lmdb_offset], R20),
    (   sub_string(R20, _, _, _, "LMDB")
    ->  writeln('PASS')
    ;   (format('FAIL: got ~w~n', [R20]), fail)
    ),

    % Test 21: Hyphens in case values (3.4)
    write('Test 21 - Hyphens in case values: '),
    render_template("{{match target}}{{case wam-fsharp}}Fsharp{{case wam-haskell}}Haskell{{/match}}",
                    [target='wam-fsharp'], R21),
    (   sub_string(R21, _, _, _, "Fsharp")
    ->  writeln('PASS')
    ;   (format('FAIL: got ~w~n', [R21]), fail)
    ),

    % Test 22: Empty case body (3.5)
    write('Test 22 - Empty case body: '),
    render_template("{{match mode}}{{case skip}}{{case keep}}KEPT{{/match}}", [mode=skip], R22a),
    render_template("{{match mode}}{{case skip}}{{case keep}}KEPT{{/match}}", [mode=keep], R22b),
    (   R22a = "", sub_string(R22b, _, _, _, "KEPT")
    ->  writeln('PASS')
    ;   (format('FAIL: skip=~w keep=~w~n', [R22a, R22b]), fail)
    ),

    % Test 23: Match with only default (3.6)
    write('Test 23 - Only default: '),
    render_template("{{match anything}}{{default}}ALWAYS{{/match}}", [anything=whatever], R23),
    (   sub_string(R23, _, _, _, "ALWAYS")
    ->  writeln('PASS')
    ;   (format('FAIL: got ~w~n', [R23]), fail)
    ),

    % Test 24: Whitespace handling (3.7)
    write('Test 24 - Whitespace preserved: '),
    render_template("{{match mode}}\n  {{case eager}}\n    EAGER\n  {{case lazy}}\n    LAZY\n{{/match}}", [mode=eager], R24),
    (   sub_string(R24, _, _, _, "EAGER"), sub_string(R24, _, _, _, "\n")
    ->  writeln('PASS')
    ;   (format('FAIL: got ~w~n', [R24]), fail)
    ),

    % Test 25: Malformed - no closing {{/match}} (3.8)
    write('Test 25 - Malformed no /match: '),
    render_template("{{match key}}{{case a}}body", [key=a], R25),
    (   sub_string(R25, _, _, _, "{{match key}}")
    ->  writeln('PASS')
    ;   (format('FAIL: got ~w~n', [R25]), fail)
    ),

    % Test 26: Malformed - no key in {{match}} (3.8)
    write('Test 26 - Malformed no key: '),
    render_template("{{match}}{{case a}}body{{/match}}", [key=a], R26),
    (   sub_string(R26, _, _, _, "{{match}}")
    ->  writeln('PASS')
    ;   (format('FAIL: got ~w~n', [R26]), fail)
    ),

    % Test 27: Orphan {{/match}} (3.8)
    write('Test 27 - Orphan /match: '),
    render_template("hello{{/match}}world", [], R27),
    (   sub_string(R27, _, _, _, "hello"), sub_string(R27, _, _, _, "world")
    ->  writeln('PASS')
    ;   (format('FAIL: got ~w~n', [R27]), fail)
    ),

    % Test 28: Variable substitution inside matched case body
    write('Test 28 - Var substitution in case: '),
    render_template("{{match mode}}{{case cached}}cache={{size}}{{default}}none{{/match}}",
                    [mode=cached, size='64'], R28),
    (   sub_string(R28, _, _, _, "cache=64")
    ->  writeln('PASS')
    ;   (format('FAIL: got ~w~n', [R28]), fail)
    ),

    % Clean up
    clear_template_cache(test_cached),

    writeln('=== Template System Tests Complete ===').

% ============================================================================
% SHELL CAPABILITY DETECTION
% ============================================================================

%% detect_shell_env(-EnvList) is det.
%
%  Auto-detect shell environment limitations. Returns a list of atoms
%  like [no_syscalls, no_processes, no_signals] describing what the
%  target shell lacks. Empty list means full bash capabilities.
%
%  Detection order:
%    1. If shell_capabilities module is loaded, query it.
%    2. Otherwise assume full capabilities (native).
%
detect_shell_env(EnvList) :-
    (   catch(
            ( shell_capabilities:shell_lacks(syscalls) -> L1 = [no_syscalls] ; L1 = [] ),
            _, L1 = []
        ),
        catch(
            ( shell_capabilities:shell_lacks(processes) -> L2 = [no_processes] ; L2 = [] ),
            _, L2 = []
        ),
        catch(
            ( shell_capabilities:shell_lacks(signals) -> L3 = [no_signals] ; L3 = [] ),
            _, L3 = []
        ),
        append([L1, L2, L3], EnvList)
    ).
