:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% pattern_stache.pl - PROTOTYPE structural template dispatcher
%
% Standalone prototype of the pattern_stache dialect proposed in
% docs/design/STRUCTURAL_TEMPLATE_MATCHING_PHILOSOPHY.md.  It renders
% {{match}}/{{case}} blocks where case values are Prolog TERMS, dict
% values are terms, and case bodies may reference variables bound by
% the match.
%
% This module deliberately does NOT import or modify
% src/unifyweaver/core/template_system.pl.  The block-scanning code
% (find_match_block/5 and friends) is copied from that module rather
% than shared, so that nothing here can reach the 67 targets that
% depend on core case_matches/2.
%
% Dialect contract implemented here (prototype-grade):
%   - file extension .stache required by load_stache_file/2;
%   - first non-empty line must be {{! dialect(pattern_stache, 1) }};
%     a headerless .stache file is an error, an unknown version is an
%     error (fail closed, never fall back to another parser);
%   - {{case P}} reads P as a term via read_term_from_atom/3 with
%     variable_names/1, in this module's operator table;
%   - dict values are terms and must be GROUND at dispatch time
%     (dispatch only on discharged goals — see
%     DESIGN_desugaring_to_prolog_goals.md §4.1);
%   - the first case whose pattern unifies with the dict value is
%     COMMITTED to; pattern bindings are prepended to the dict as a
%     child scope for the body only (bindings shadow outer keys);
%   - case overlap is checked at load: an earlier pattern that
%     subsumes a later one makes the later case unreachable (error);
%     two patterns that unify without either subsuming the other are
%     order-dependent (warning); a later pattern subsuming an earlier
%     one is the specific-before-general idiom (silent, allowed).

:- module(pattern_stache, [
    load_stache_file/2,        % +Path, -Template
    render_stache/3,           % +Template, +Dict, -Result
    render_stache_file/3,      % +Path, +Dict, -Result
    check_stache_template/1,   % +Template (overlap + case syntax check)
    read_case_pattern/3,       % +CaseText, -Pattern, -VarNames
    goals_to_ast/3,            % +TemplatePath, +Goals, -Lines
    demo/0
]).

:- use_module(library(lists)).
:- use_module(library(apply)).

%% ============================================
%% LOADER
%% ============================================
%
% Q5 (loader dispatch): the prototype gives the dialect its own loader
% rather than a new try_source/4 strategy.  try_source strategies answer
% "where does the template text come from" (file/cache/generated); the
% dialect answers "how is the text parsed".  Conflating them would mean
% a cached or generated template has no dialect.  In core this would be
% a branch in the render path keyed on the file extension recorded at
% load time — see the prototype report.

%% load_stache_file(+Path, -Template)
%  Load a .stache file, enforcing the extension, the dialect header,
%  and the load-time case checks.  Template is stache(Version, Body)
%  with the header line removed from Body.
load_stache_file(Path, stache(Version, Body)) :-
    (   file_name_extension(_, stache, Path)
    ->  true
    ;   throw(error(pattern_stache(not_a_stache_file(Path)), _))
    ),
    read_file_to_string(Path, Text, []),
    strip_dialect_header(Path, Text, Version, Body),
    check_stache_template(stache(Version, Body)).

%% strip_dialect_header(+Path, +Text, -Version, -Body)
%  The header must be the FIRST non-empty line — fixed position, not
%  scanned for, so a {{! dialect(...) }} inside a case body can never
%  be taken for a header.
strip_dialect_header(Path, Text, Version, Body) :-
    split_string(Text, "\n", "", Lines),
    (   append(Blank, [HeaderLine|Rest], Lines),
        \+ ( member(B, Blank), \+ blank_line(B) ),
        \+ blank_line(HeaderLine)
    ->  (   parse_dialect_header(HeaderLine, Version)
        ->  (   Version == 1
            ->  atomic_list_concat(Rest, '\n', BodyAtom),
                atom_string(BodyAtom, Body)
            ;   throw(error(pattern_stache(unsupported_dialect_version(Path, Version)), _))
            )
        ;   throw(error(pattern_stache(missing_dialect_header(Path)), _))
        )
    ;   throw(error(pattern_stache(missing_dialect_header(Path)), _))
    ).

blank_line(Line) :- split_string(Line, "", " \t\r", [""]).

%% parse_dialect_header(+Line, -Version)
%  Header shape: {{! dialect(pattern_stache, V) }} — the pragma value
%  is a Prolog term read directly, no bespoke parser (philosophy doc,
%  "Prefer a term to a key: value string").
parse_dialect_header(Line, Version) :-
    split_string(Line, "", " \t\r", [Trimmed]),
    string_concat("{{!", T1, Trimmed),
    string_concat(Pragma, "}}", T1),
    catch(term_string(PragmaTerm, Pragma), _, fail),
    PragmaTerm = dialect(pattern_stache, Version).

%% ============================================
%% LOAD-TIME CHECKS (Q2 syntax, Q3 overlap)
%% ============================================

%% check_stache_template(+Template)
%  Walk every {{match}} block (including nested ones) and check that
%  each case value reads as a term and that no case is unreachable.
%  Runs at LOAD, before any dict exists, so a broken template fails
%  when it is loaded rather than when the offending case is first hit.
check_stache_template(stache(_, Body)) :-
    check_blocks_in(Body).

check_blocks_in(Text) :-
    (   find_match_block(Text, _Key, _Before, MatchBody, After)
    ->  parse_match_cases(MatchBody, Cases, Default),
        check_case_list(Cases),
        % recurse into case bodies and the default for nested blocks
        forall(member(case(_, _, CaseBody), Cases), check_blocks_in(CaseBody)),
        check_blocks_in(Default),
        check_blocks_in(After)
    ;   true
    ).

%% check_case_list(+Cases)
%  Q3: the overlap trichotomy, decided by subsumes_term/2.
%    earlier subsumes later  -> later can never fire  -> ERROR
%    later subsumes earlier  -> specific-before-general refinement -> silent
%    unify, neither subsumes -> genuinely order-dependent -> WARNING
check_case_list(Cases) :-
    forall(
        ( nth1(I, Cases, case(PI, _, _)),
          nth1(J, Cases, case(PJ, _, _)),
          I < J
        ),
        check_case_pair(I, PI, J, PJ)
    ).

check_case_pair(I, PI, J, PJ) :-
    copy_term(PI, A),
    copy_term(PJ, B),
    (   subsumes_term(A, B)
    ->  throw(error(pattern_stache(unreachable_case(case_index(J), subsumed_by(case_index(I)), pattern(PJ))), _))
    ;   subsumes_term(B, A)
    ->  true    % refinement idiom: specific case listed before general
    ;   unifiable(A, B, _)
    ->  format(user_error,
               "pattern_stache: WARNING order-dependent overlap between case ~w (~q) and case ~w (~q); first match wins~n",
               [I, PI, J, PJ])
    ;   true    % disjoint
    ).

%% ============================================
%% CASE PATTERN READING (Q2)
%% ============================================

%% read_case_pattern(+CaseText, -Pattern, -VarNames)
%  Read the text of a {{case ...}} tag as a Prolog term.
%    - variable_names(VarNames) recovers Name=Var pairs, which is the
%      channel from the pattern into the body's scope.  `_` yields no
%      pair, so it is a true wildcard that binds nothing.
%    - module(pattern_stache) pins the operator table to this module:
%      the standard table, unaffected by op/3 declarations made
%      elsewhere in the process.
%    - a syntax error (thrown by the reader) is wrapped so the message
%      names the offending case text.
read_case_pattern(CaseText, Pattern, VarNames) :-
    catch(
        read_term_from_atom(CaseText, Pattern,
                            [variable_names(VarNames), module(pattern_stache)]),
        error(syntax_error(Why), _),
        throw(error(pattern_stache(bad_case_pattern(CaseText, syntax_error(Why))), _))
    ).

%% ============================================
%% RENDERING
%% ============================================

%% render_stache_file(+Path, +Dict, -Result)
render_stache_file(Path, Dict, Result) :-
    load_stache_file(Path, Template),
    render_stache(Template, Dict, Result).

%% render_stache(+Template, +Dict, -Result)
%  Template is stache(Version, Body) from load_stache_file/2, or a
%  plain string/atom (accepted for tests; no header required then).
render_stache(stache(_, Body), Dict, Result) :-
    !,
    render_body(Body, Dict, Result).
render_stache(Text, Dict, Result) :-
    atom_string(Text, S),
    render_body(S, Dict, Result).

render_body(Text, Dict, Result) :-
    expand_match_blocks(Text, Dict, Expanded),
    substitute_placeholders(Expanded, Dict, Result).

%% expand_match_blocks(+Template, +Dict, -Result)
%  Same recursive structure as core, with two differences:
%    - resolve_match_term/6 returns a CHILD dict carrying the pattern
%      bindings, and the selected body is expanded under that child
%      scope (Q1);
%    - placeholder substitution for the body also runs under the child
%      scope, so bindings are visible to {{Var}} interpolation but do
%      not leak past {{/match}}.
expand_match_blocks(Template, Dict, Result) :-
    (   find_match_block(Template, Key, Before, MatchBody, After)
    ->  parse_match_cases(MatchBody, Cases, Default),
        resolve_match_term(Key, Dict, Cases, Default, Body, ChildDict),
        expand_match_blocks(Body, ChildDict, BodyExpanded),
        substitute_placeholders(BodyExpanded, ChildDict, BodyDone),
        expand_match_blocks(After, Dict, AfterExpanded),
        string_concat(Before, BodyDone, P1),
        string_concat(P1, AfterExpanded, Result)
    ;   Result = Template
    ).

%% resolve_match_term(+Key, +Dict, +Cases, +Default, -Body, -ChildDict)
%
%  Q1 (binding propagation): on a successful match the pattern's
%  variable_names pairs become dict entries PREPENDED to the incoming
%  dict.  Lookup everywhere in this module is member/2, so a prepended
%  binding shadows an outer key of the same name for the body's scope
%  and only there — the caller's Dict is untouched.
%
%  Q4 (determinism): the condition of the if-then-else contains ONLY
%  the dict lookup and the unification test.  member/2 enumerates
%  cases, but `->` prunes those choice points the moment one pattern
%  unifies.  Body expansion happens after the commit, so nothing that
%  goes wrong inside the body — a nested match falling through to an
%  empty default, or an exception from a malformed nested case — can
%  reconsider the case selection.  This mirrors core resolve_match/5
%  (member/2 under ->) by construction.
%
%  Q6 (dict contract): the dict value must be GROUND.  A nonground
%  value could be bound BY the pattern (dispatch deciding the data
%  instead of the data deciding the dispatch), which is the residual-
%  goal hazard §4.1 warns about.  Fail closed with a named error.
resolve_match_term(Key, Dict, Cases, Default, Body, ChildDict) :-
    (   member(Key=Value, Dict)
    ->  (   ground(Value)
        ->  true
        ;   throw(error(pattern_stache(nonground_dispatch(Key, Value)), _))
        ),
        (   member(case(Pattern0, VarNames0, CaseBody), Cases),
            copy_term(Pattern0-VarNames0, Pattern-VarNames),
            Pattern = Value                       % unify; -> commits below
        ->  Body = CaseBody,
            append(VarNames, Dict, ChildDict)     % bindings shadow outer keys
        ;   Body = Default,
            ChildDict = Dict
        )
    ;   % key absent from dict: default, no bindings
        Body = Default,
        ChildDict = Dict
    ).

%% substitute_placeholders(+Text, +Dict, -Result)
%  Replace {{Key}} and {{q:Key}} for every Key=Value in Dict.  Values
%  are terms.  Two interpolation forms exist because two consumers
%  demanded them:
%    {{Key}}    ~w, plain write — display text (the AST-emission
%               consumer); atoms render unquoted, as a template
%               author expects.
%    {{q:Key}}  ~q, quoted write — RE-READABLE text (the constraint-
%               dispatch consumer, whose rendered output is read back
%               as a term); 'hello world' keeps its quotes, plain
%               atoms are unchanged.
%  Keys not in the dict are left verbatim in both forms, exactly as
%  the core renderer leaves them (NOT replaced by the empty string —
%  see the report's "what the philosophy doc got wrong").
substitute_placeholders(Text, [], Text) :- !.
substitute_placeholders(Text, [Key=Value|Rest], Result) :-
    format(atom(QPlaceholder), '{{q:~w}}', [Key]),
    format(atom(QValueAtom), '~q', [Value]),
    atom_string(QValueAtom, QValueStr),
    atom_string(QPlaceholder, QPlaceholderStr),
    replace_substring(Text, QPlaceholderStr, QValueStr, QMid),
    format(atom(Placeholder), '{{~w}}', [Key]),
    format(atom(ValueAtom), '~w', [Value]),
    atom_string(ValueAtom, ValueStr),
    atom_string(Placeholder, PlaceholderStr),
    replace_substring(QMid, PlaceholderStr, ValueStr, Mid),
    substitute_placeholders(Mid, Rest, Result).

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
%% BLOCK SCANNING
%% ============================================
%
% Copied from src/unifyweaver/core/template_system.pl (same balanced
% depth-counting scanner) so this prototype shares no code with core.
% The one behavioural change is in tag parsing: a {{case ...}} value is
% kept as TEXT here and read as a term by parse_match_cases/3 below.

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

next_match_event(Str, Type, Offset, Len) :-
    (   sub_string(Str, OpenIdx, 8, _, "{{match ") -> true ; OpenIdx = 999999999 ),
    (   sub_string(Str, CloseIdx, 10, _, "{{/match}}") -> true ; CloseIdx = 999999999 ),
    OpenIdx + CloseIdx < 1999999998,
    (   OpenIdx < CloseIdx
    ->  Type = open, Offset = OpenIdx, Len = 8
    ;   Type = close, Offset = CloseIdx, Len = 10
    ).

%% parse_match_cases(+Body, -Cases, -Default)
%  Cases is a list of case(Pattern, VarNames, CaseBody); Default is a
%  string ("" when no {{default}} present — a non-matching match block
%  with no default renders as empty, same as core).
parse_match_cases(Body, Cases, Default) :-
    split_match_segments(Body, Segments),
    extract_cases(Segments, Cases, Default).

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
        ->  split_match_segments_(After, [segment(Type, Value, "")|[segment(PrevType, PrevVal, Before)|RestAcc]], Segments)
        ;   split_match_segments_(After, [segment(Type, Value, "")|Acc], Segments)
        )
    ;   (   Acc = [segment(PrevType, PrevVal, _)|RestAcc]
        ->  reverse([segment(PrevType, PrevVal, Body)|RestAcc], Segments)
        ;   Segments = []
        )
    ).

find_next_case_marker(Str, Type, Value, Before, After) :-
    find_next_case_marker_(Str, 0, 0, Type, Value, MarkerIdx, AfterIdx),
    sub_string(Str, 0, MarkerIdx, _, Before),
    sub_string(Str, AfterIdx, _, 0, After).

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

find_earliest_tag(Str, TagType, Offset, Value, AfterOffset) :-
    findall(Idx-Type-Val-After,
            tag_candidate(Str, Idx, Type, Val, After),
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
    atom_string(Val, ValStr),          % kept as TEXT; term read happens later
    After is Start + EndRel + 2.
tag_candidate(Str, Idx, default_tag, '', After) :-
    sub_string(Str, Idx, 11, _, "{{default}}"),
    After is Idx + 11.

%% extract_cases(+Segments, -Cases, -Default)
%  This is where the dialect diverges from core: each case value is
%  read as a term, here, once per match block per render/check.
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

seg_to_case(segment(case_seg, CaseText, Body), case(Pattern, VarNames, Body)) :-
    read_case_pattern(CaseText, Pattern, VarNames).

%% ============================================
%% WORKED EXAMPLE DRIVER
%% ============================================

%% goals_to_ast(+TemplatePath, +Goals, -Lines)
%  Render each goal through the template with a dict of [goal=Goal];
%  one output line (trimmed) per goal.  This is the motivating
%  consumer from DESIGN_desugaring_to_prolog_goals.md §4.1: goals in,
%  typed AST node text out, dispatched on goal structure.
goals_to_ast(TemplatePath, Goals, Lines) :-
    load_stache_file(TemplatePath, Template),
    maplist(goal_to_ast_line(Template), Goals, Lines).

goal_to_ast_line(Template, Goal, Line) :-
    render_stache(Template, [goal=Goal], Rendered),
    normalize_space(string(Line), Rendered).

%% demo
%  The end-to-end worked example.  Run:
%    swipl -g pattern_stache:demo -t halt pattern_stache.pl
demo :-
    module_property(pattern_stache, file(Here)),
    file_directory_name(Here, Dir),
    atomic_list_concat([Dir, '/goal_to_ast.stache'], TemplatePath),
    Goals = [
        has_type(x, substrate(pearltrees)),
        has_type(y, substrate(wikipedia)),
        has_type(j1, judge(sonnet)),
        non_amplifying(min),
        mu_bounded(path, 0.5),
        in_support(decay, d)            % no case matches -> default
    ],
    goals_to_ast(TemplatePath, Goals, Lines),
    format("=== pattern_stache demo: goals -> AST nodes ===~n"),
    forall(
        ( nth1(I, Goals, G), nth1(I, Lines, L) ),
        format("~q~n    => ~w~n", [G, L])
    ).
