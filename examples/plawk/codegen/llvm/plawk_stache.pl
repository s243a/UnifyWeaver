% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk_stache -- plawk's .stache templates (pattern_stache dialect), one IR line
% per rendering. The first consumer of pattern_stache on this line (template PR 4,
% docs/design/PLAN_TEMPLATE_REFACTOR.md sections 2.3-A and 8, decision 5).
%
% A template here is one {{match op}} block whose cases are STRUCTURAL: a family of
% emitters that differed only in the values they substituted becomes one case, and
% the caller passes a ground term (elem_print(Table, Key, i64), ...). Prolog keeps
% every decision -- which kind, which SSA names, which runtime function -- the
% template only spells the line.
%
% Whitespace policy (the engine leaves it to the caller, SPEC "Whitespace"): every
% case body is exactly one line, written one case per line in the file; the
% rendering's surrounding newlines are stripped, and a rendering that is not
% exactly one line is an error.

:- module(plawk_stache, [
    plawk_render_stache/3,      % +Name, +Op, -Line
    plawk_stache_preflight/0
]).

:- use_module('../../../../src/unifyweaver/core/pattern_stache',
              [load_stache_file/2, render_stache/3]).

:- dynamic plawk_stache_cache/2.
:- dynamic plawk_stache_dir/1.

%% plawk_render_stache(+Name, +Op, -Line) is det.
%  Render template Name (templates/Name.ll.stache next to this file) for the
%  ground term Op, as an atom holding one IR line with no newline.
plawk_render_stache(Name, Op, Line) :-
    (   ground(Op)
    ->  true
    ;   throw(error(instantiation_error, context(plawk_render_stache/3, Op)))
    ),
    plawk_stache_template(Name, Template),
    render_stache(Template, [op=Op], Rendered),
    split_string(Rendered, "", "\n", [Body]),
    (   Body \== "", \+ sub_string(Body, _, _, _, "\n")
    ->  atom_string(Line, Body)
    ;   throw(error(plawk_stache(not_one_line(Name, Op, Rendered)), _))
    ).

plawk_stache_template(Name, Template) :-
    plawk_stache_cache(Name, Template0),
    !,
    Template = Template0.
plawk_stache_template(Name, Template) :-
    plawk_stache_dir(Dir),
    atomic_list_concat([Dir, '/templates/', Name, '.ll.stache'], Path),
    load_stache_file(Path, Template),
    assertz(plawk_stache_cache(Name, Template)).

:- prolog_load_context(directory, Dir),
   retractall(plawk_stache_dir(_)),
   assertz(plawk_stache_dir(Dir)).

%% plawk_stache_preflight is det.
%  Load every template and render each case once, so a broken template fails at
%  load (with the case named) instead of in the middle of some program's build.
plawk_stache_preflight :-
    forall(plawk_stache_sample(Name, Op, Want),
           (   catch(plawk_render_stache(Name, Op, Line), E, true),
               (   var(E), Line == Want
               ->  true
               ;   throw(error(plawk_stache(preflight_failed(Name, Op, E, Line)), _))
               )
           )).

% One sample per case, with the exact line it must produce.
plawk_stache_sample(assoc_elem, elem_print(3, '%k', str),
    '  call void @wam_assoc_str_print(%WamAssocI64Table* %plawk_assoc_table_3, i64 %k)').
plawk_stache_sample(assoc_elem, elem_write('%r', 2, 7, 1, inc, i64),
    '  %r = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %plawk_assoc_table_2, i64 7, i64 1)').
plawk_stache_sample(assoc_elem, elem_write('%r', 2, '%id', '%v', set, f64),
    '  %r = call double @wam_assoc_f64_set(%WamAssocI64Table* %plawk_assoc_table_2, i64 %id, double %v)').
plawk_stache_sample(assoc_elem, elem_call('%v', i64, i64, get, 3, '%k'),
    '  %v = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_3, i64 %k)').

:- initialization(plawk_stache_preflight).
