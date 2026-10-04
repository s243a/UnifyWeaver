% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% template_library -- target code held in template FILES, read once per process.
%
% Large target code (e.g. the LLVM runtime) used to live inside Prolog quoted
% atoms, where an apostrophe in an LLVM comment closed the atom and broke every
% build. It now lives in template files under templates/, rendered with
% template_system.pl's mustache dialect (which stays frozen). This module adds only
% what that needs: path resolution from the repository root, a per-process cache
% (a build renders the same runtime once; parsing it per call would dominate), and
% one render entry point. See docs/design/PLAN_TEMPLATE_REFACTOR.md (section 4, decision 4).

:- module(template_library, [
    template_library_text/2,     % +RelPath, -Text
    template_library_render/3,   % +RelPath, +Dict, -Rendered
    template_library_cases/2     % +RelPath, -Cases (list of Name-Body)
]).

:- use_module(template_system, [render_template/3]).

:- dynamic template_library_cache/2.
:- dynamic template_library_cases_cache/2.
:- dynamic template_library_root/1.

% The repository root: three levels above this file's directory (src/unifyweaver/core).
:- prolog_load_context(directory, Dir),
   directory_file_path(Dir, '../../..', Root0),
   absolute_file_name(Root0, Root, [file_type(directory)]),
   retractall(template_library_root(_)),
   assertz(template_library_root(Root)).

%% template_library_text(+RelPath, -Text) is det.
%  The text of the template at RelPath (relative to the repository root), read as
%  UTF-8 once per process.
template_library_text(RelPath, Text) :-
    template_library_cache(RelPath, Text0),
    !,
    Text = Text0.
template_library_text(RelPath, Text) :-
    template_library_read(RelPath, Text),
    assertz(template_library_cache(RelPath, Text)).

template_library_read(RelPath, Text) :-
    template_library_root(Root),
    directory_file_path(Root, RelPath, Path),
    read_file_to_string(Path, Text0, [encoding(utf8)]),
    atom_string(Text, Text0).

%% template_library_render(+RelPath, +Dict, -Rendered) is det.
%  Render the template at RelPath with Dict (Key=Value pairs, mustache dialect of
%  template_system.pl).
template_library_render(RelPath, Dict, Rendered) :-
    template_library_text(RelPath, Text),
    render_template(Text, Dict, Rendered0),
    atom_string(Rendered, Rendered0).

%% template_library_cases(+RelPath, -Cases) is det.
%  The cases of a case library at RelPath, as Name-Body pairs in file order, parsed
%  once per process. A case library is one template_system match block,
%
%      {{match fn}}<header>{{case Name1}}Body1{{case Name2}}Body2...{{/match}}
%
%  followed by exactly one final LF. Each body is the raw text up to the next
%  marker (never normalised: bytes inside c"..." are data); the header before the
%  first case is documentation and is dropped, as template_system drops it.
%  Malformed files, nested blocks, {{default}} and duplicate names are errors.
template_library_cases(RelPath, Cases) :-
    template_library_cases_cache(RelPath, Cases0),
    !,
    Cases = Cases0.
template_library_cases(RelPath, Cases) :-
    template_library_read(RelPath, Text),
    (   template_library_parse_cases(Text, Cases0)
    ->  true
    ;   throw(error(template_library(malformed_case_library(RelPath)), _))
    ),
    pairs_keys(Cases0, Names),
    (   msort(Names, Sorted), \+ append(_, [N, N | _], Sorted)
    ->  true
    ;   throw(error(template_library(duplicate_case(RelPath)), _))
    ),
    assertz(template_library_cases_cache(RelPath, Cases0)),
    Cases = Cases0.

template_library_parse_cases(Text, Cases) :-
    atom_concat(Block, '{{/match}}\n', Text),
    sub_atom(Block, 0, _, _, '{{match '),
    sub_atom(Block, OpenEnd0, 2, _, '}}'),
    !,
    OpenEnd is OpenEnd0 + 2,
    sub_atom(Block, OpenEnd, _, 0, Inner),
    \+ sub_atom(Inner, _, _, _, '{{match '),
    \+ sub_atom(Inner, _, _, _, '{{/match}}'),
    \+ sub_atom(Inner, _, _, _, '{{default}}'),
    atomic_list_concat([_Header | Segments], '{{case ', Inner),
    Segments \== [],
    maplist(template_library_case_segment, Segments, Cases).

template_library_case_segment(Segment, Name-Body) :-
    sub_atom(Segment, B, 2, A, '}}'),
    !,
    sub_atom(Segment, 0, B, _, Name),
    Name \== '',
    sub_atom(Segment, _, A, 0, Body).
