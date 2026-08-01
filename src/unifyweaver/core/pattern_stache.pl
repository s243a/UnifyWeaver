:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% pattern_stache.pl - structural template dispatcher (dialect version 1)
%
% Implements the pattern_stache dialect specified in
% docs/design/SPEC_pattern_stache.md: {{match}}/{{case}} dispatch where
% case values are Prolog TERMS matched by unification against ground
% dict values, and case bodies may reference variables bound by the
% match.  Graduated from the prototype in
% prototypes/mu_cosine/pattern_stache/ after serving two consumers with
% the matching machinery unchanged (reports alongside the prototype).
%
% ADDITIVE BY CONSTRUCTION.  This module is a separate engine for a
% separate file extension (.stache).  It does not import, modify, or
% share a code path with template_system.pl, whose string semantics 67
% targets depend on; the block scanner below is a copy, not an import.
% No .mustache file can reach this parser (the loader refuses the
% extension), and no configuration option exists that could route one
% here — the decision is in the artifact (PROJECT_PHILOSOPHY §4).
%
% FAIL CLOSED.  A .stache file without the dialect header is an error;
% an unknown dialect version is an error; a case value that does not
% read as a term is an error; a nonground dispatch value is an error;
% a non-linear pattern is an error; an unreachable case is an error at
% load.  Order-dependent overlap (unifiable, neither subsumes) is
% never silent: it warns, and first-match-wins applies.
%
% Contract summary (normative text is the SPEC):
%   - loader: .stache extension + {{! dialect(pattern_stache, 1) }} as
%     the FIRST non-empty line; load-time case checks;
%   - patterns: first-order, LINEAR, guard-free terms read with the
%     standard operator table pinned to this module;
%   - dispatch: first case unifying with the ground dict value commits;
%     body failure or error never reconsiders the selection;
%   - bindings: prepended to the dict as a child scope for the case
%     body only; they shadow outer keys lexically and silently;
%   - interpolation: {{Key}} renders ~w (display), {{q:Key}} renders
%     ~q (re-readable); unknown keys are left verbatim.

:- module(pattern_stache, [
    load_stache_file/2,        % +Path, -Template
    render_stache/3,           % +Template, +Dict, -Result
    render_stache_file/3,      % +Path, +Dict, -Result
    check_stache_template/1,   % +Template (case syntax, linearity, overlap)
    read_case_pattern/3        % +CaseText, -Pattern, -VarNames
]).

:- use_module(library(lists)).
:- use_module(library(apply)).

%% ============================================
%% LOADER
%% ============================================
%
% The dialect has its own load step rather than a try_source/4
% strategy: source strategies answer "where does the text come from"
% (file/cache/generated), the dialect answers "how is it parsed", and
% cached or generated templates have no extension.  The loaded
% template is a tagged term, stache(Version, Body), so render paths
% branch on the tag, never on configuration.

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
%  be taken for a header.  An unknown version is an error, never a
%  best-effort parse with the newest available reader.
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
%  Header shape: {{! dialect(pattern_stache, V) }}.  The pragma value
%  is a Prolog term read directly — no bespoke parser, and it extends
%  by adding arguments rather than by re-parsing.
parse_dialect_header(Line, Version) :-
    split_string(Line, "", " \t\r", [Trimmed]),
    string_concat("{{!", T1, Trimmed),
    string_concat(Pragma, "}}", T1),
    catch(term_string(PragmaTerm, Pragma), _, fail),
    PragmaTerm = dialect(pattern_stache, Version).

%% ============================================
%% LOAD-TIME CHECKS
%% ============================================

%% check_stache_template(+Template)
%  Walk every {{match}} block (including nested ones) and check that
%  each case value reads as a term, is linear, and that no case is
%  unreachable.  Runs at LOAD, before any dict exists: a broken
%  template fails when it is loaded, not when the offending case is
%  first needed.
check_stache_template(stache(_, Body)) :-
    check_blocks_in(Body).

check_blocks_in(Text) :-
    (   find_match_block(Text, _Key, _Before, MatchBody, After)
    ->  parse_match_cases(MatchBody, Cases, Default),
        check_case_list(Cases),
        forall(member(case(_, _, CaseBody), Cases), check_blocks_in(CaseBody)),
        check_blocks_in(Default),
        check_blocks_in(After)
    ;   true
    ).

%% check_case_list(+Cases)
%  The overlap trichotomy, decided by subsumes_term/2:
%    earlier subsumes later  -> later can never fire  -> ERROR
%    later subsumes earlier  -> specific-before-general refinement -> silent
%    unify, neither subsumes -> genuinely order-dependent -> WARNING
%  The refinement row is load-bearing in a witnessed consumer; a flat
%  overlap ban would reject real templates.
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
    ->  true
    ;   unifiable(A, B, _)
    ->  print_message(warning, pattern_stache(order_dependent_overlap(I, PI, J, PJ)))
    ;   true
    ).

%% ============================================
%% CASE PATTERN READING
%% ============================================

%% read_case_pattern(+CaseText, -Pattern, -VarNames)
%  Read the text of a {{case ...}} tag as a Prolog term.
%    - variable_names(VarNames) recovers Name=Var pairs — the channel
%      from the pattern into the body's scope.  `_` yields no pair, so
%      it is a true wildcard that binds nothing.
%    - module(pattern_stache) pins the operator table to this module:
%      the standard table, unaffected by op/3 declarations made
%      elsewhere in the process.
%    - a syntax error is wrapped so the message names the case text.
%    - linearity is enforced here: a variable occurring twice in one
%      pattern is a load error, because non-linear patterns are
%      outside the v1 contract (SPEC, deliberate exclusions).
read_case_pattern(CaseText, Pattern, VarNames) :-
    catch(
        read_term_from_atom(CaseText, Pattern,
                            [variable_names(VarNames), module(pattern_stache)]),
        error(syntax_error(Why), _),
        throw(error(pattern_stache(bad_case_pattern(CaseText, syntax_error(Why))), _))
    ),
    (   pattern_is_linear(Pattern)
    ->  true
    ;   throw(error(pattern_stache(nonlinear_pattern(CaseText)), _))
    ).

%% pattern_is_linear(+Pattern)
%  True when no variable occurs more than once.  Each occurrence of
%  `_` is a distinct variable, so wildcards never trip this.
pattern_is_linear(Pattern) :-
    variable_occurrences(Pattern, Occurrences),
    \+ ( append(_, [V|Rest], Occurrences),
         member(W, Rest),
         V == W
       ).

variable_occurrences(Term, [Term]) :-
    var(Term),
    !.
variable_occurrences(Term, Occurrences) :-
    compound(Term),
    !,
    Term =.. [_|Args],
    maplist(variable_occurrences, Args, Nested),
    append(Nested, Occurrences).
variable_occurrences(_, []).

%% ============================================
%% RENDERING
%% ============================================

%% render_stache_file(+Path, +Dict, -Result)
render_stache_file(Path, Dict, Result) :-
    load_stache_file(Path, Template),
    render_stache(Template, Dict, Result).

%% render_stache(+Template, +Dict, -Result)
%  Template is stache(Version, Body) from load_stache_file/2, or a
%  plain string/atom (accepted for direct calls and tests; the header
%  requirement is the loader's job, not the renderer's).
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
%  The selected body is expanded and substituted under the CHILD dict
%  carrying the pattern bindings, so bindings are visible to the body
%  and invisible past {{/match}}.
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
%  Binding propagation: on a successful match the pattern's
%  variable_names pairs are PREPENDED to the incoming dict.  Lookup
%  everywhere in this module is member/2, so a prepended binding
%  shadows an outer key of the same name for the body's scope and only
%  there — the caller's Dict is untouched.
%
%  Determinism: the condition of the if-then-else contains ONLY the
%  dict lookup and the unification test.  member/2 enumerates cases,
%  but `->` prunes those choice points the moment one pattern unifies.
%  Body expansion happens after the commit, so nothing that goes wrong
%  inside the body — a nested block rendering empty, or an exception —
%  can reconsider the case selection.
%
%  Dict contract: the dict value must be GROUND.  A nonground value
%  could be bound BY the pattern — dispatch deciding the data instead
%  of the data deciding the dispatch.  Callers that carry residual
%  (nonground) goals route them away from dispatch upstream; this
%  check is the engine's backstop, not the mode system.
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
%  Replace {{q:Key}} (~q, re-readable) then {{Key}} (~w, display) for
%  every Key=Value in Dict.  Keys not in the dict are left verbatim in
%  both forms, matching the string dialect's behaviour (changing this
%  is a deliberate exclusion in the SPEC — it would be a new dialect
%  version, since it changes output).
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
% Copied from template_system.pl's balanced depth-counting scanner so
% this engine shares no code path with the string dialect.  The one
% behavioural change is in tag parsing: a {{case ...}} value is kept
% as TEXT here and read as a term by parse_match_cases/3 below.
% Depth-awareness note: nested {{match}} blocks are scanned correctly
% and current behaviour is characterized by tests, but nesting is
% outside the v1 contract (SPEC, deliberate exclusions).

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
%  string ("" when no {{default}} is present — a non-matching match
%  block with no default renders as empty).
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
%  This is where the dialect diverges from the string parser: each
%  case value is read as a term (and checked for linearity), here.
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
%% MESSAGES
%% ============================================

:- multifile prolog:message//1.

prolog:message(pattern_stache(order_dependent_overlap(I, PI, J, PJ))) -->
    [ 'pattern_stache: order-dependent overlap between case ~w (~q) and case ~w (~q); first match wins'-[I, PI, J, PJ] ].
