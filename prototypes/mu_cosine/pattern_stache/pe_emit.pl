:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% pe_emit.pl - PROTOTYPE third consumer for pattern_stache
%
% Template-driven emitter from Prolog goal terms to registry-v0.4
% canonical process expressions, verified byte-exact against the
% sealed golden bundle PROCESS_EXPRESSION_GOLDEN_v3.json (see
% test_pe_emit.pl).  Uses the PRODUCTION engine
% (src/unifyweaver/core/pattern_stache.pl), not the prototype witness
% in this directory.
%
% Per the owner's standing direction for this lane, the shape mirrors
% the transpiler targets: INTROSPECTION walks the goal term and
% supplies per-operator facts (a registry mirror, playing the role of
% common_generator's Config), TEMPLATES render each construct, and
% composition is RECURSION in this driver — children are emitted
% first, the template dispatches one node at a time on its structural
% form.  Exactly the driver-iterates / template-dispatches boundary
% the first two consumers established.
%
% ============================================================
% GOAL CONVENTION (the one concrete convention, documented)
% ============================================================
% A process expression is a Prolog goal term:
%
%   operator application   name(Arg1, ..., ArgN, kw1(V1), ..., kwM(VM))
%                          positional args first; each kwarg is a UNARY
%                          compound whose functor is the kwarg name (no
%                          v0.4 kwarg name collides with an operator
%                          name, so classification is unambiguous)
%   bare atom              pearltrees, haiku, e5, fs, ...
%   modified atom          mod(Base, Mod)      e.g. mod(luna, 'D')
%   pinned expression      pin(Expr, Pin)      e.g. pin(E, 'run/2026-07-25')
%   numbers                Prolog numbers      0.02, 1, -0.5
%   lists                  Prolog lists        [0.02, 0.03]
%   strings                SWI strings         "a\"b\\c"
%   enum values            atoms               estimand(ancestry), impl(structural)
%
% Example (the task's own):
%   lineage(pearltrees, mu(haiku), estimand(ancestry))
%     =>  lineage(pearltrees,decay=0.85,estimand="ancestry",mu=haiku)
%
% ============================================================
% CANONICALIZATION (mirrors process_cards.py _canonical/_render_val)
% ============================================================
%   - registry defaults resolved explicitly (v0.4 has exactly one:
%     lineage's decay=0.85); unset optional kwargs are omitted;
%   - kwargs sorted by name; no whitespace anywhere;
%   - string-kind values (estimand, impl, manifest) rendered as JSON
%     strings: double-quoted, \" and \\ escaped, non-ASCII preserved;
%   - numbers via ~w (verified against Python repr for the golden
%     corpus by the byte-equality tests);
%   - pins rendered only in the FULL form; the SEMANTIC form strips
%     them at every node.
%
% Value formatting (JSON escaping, number spelling) lives HERE, in the
% driver, not in the dialect: {{q:Key}} is Prolog-reader quoting, and
% the target language of this emitter is the v0.4 surface, which owns
% its own literal syntax — exactly as every transpiler target's Config
% owns atom_fmt.  See the third-consumer report.

:- module(pe_emit, [
    pe_semantic/2,             % +Goal, -CanonicalSemanticText
    pe_full/2                  % +Goal, -CanonicalFullText
]).

:- use_module('../../../src/unifyweaver/core/pattern_stache',
              [load_stache_file/2, render_stache/3]).
:- use_module(pe_registry_mirror,
              [pe_atom/1, pe_operator/1, pe_variadic/1, pe_kwspec/4]).
:- use_module(library(lists)).
:- use_module(library(apply)).

%% ============================================
%% REGISTRY MIRROR (v0.4)
%% ============================================
%
% The per-operator facts the introspection consults (pe_atom/1,
% pe_operator/1, pe_variadic/1, pe_kwspec/4) come from the GENERATED,
% hash-checked mirror pe_registry_mirror.pl -- regenerated from the
% sealed process_cards.py by gen_registry_mirror.py, refusing to load
% on source drift (ruling 5(b), DESIGN_prolog_elaborator.md /
% PR #4093).  The hand-maintained copy that used to live here is
% retired; the golden-bundle byte tests are the drift test proving the
% swap changed nothing.
%
% Kind vocabulary note: the generated mirror carries the registry's
% kinds verbatim -- estimand and impl are their own kinds (rendered
% like strings), and a node-valued kwarg carries its declared output
% type (mu -> judge) rather than the invented kind `expr` the retired
% copy used.

%% ============================================
%% TEMPLATE LOADING (once, cached)
%% ============================================

:- dynamic pe_template_cache/2.

pe_template(Which, Template) :-
    (   pe_template_cache(Which, Template)
    ->  true
    ;   module_property(pe_emit, file(Here)),
        file_directory_name(Here, Dir),
        atomic_list_concat([Dir, '/pe_', Which, '.stache'], Path),
        load_stache_file(Path, Template),
        assertz(pe_template_cache(Which, Template))
    ).

render_surface(Dict, Text) :-
    pe_template(surface, T),
    render_stache(T, Dict, R),
    trim(R, Text).

render_value(Form, Text) :-
    pe_template(value, T),
    render_stache(T, [val=Form], R),
    trim(R, Text).

% Case bodies carry the newline that separates them from the next
% {{case}} marker; canonical strings carry no edge whitespace, so each
% rendering is edge-trimmed.  Interior whitespace (inside string
% literals) is preserved — this is NOT normalize_space/2.
trim(S, T) :-
    split_string(S, "", " \t\r\n", [T0]),
    atom_string(T, T0).

%% ============================================
%% THE EMITTER
%% ============================================

%% pe_semantic(+Goal, -Text)
%  The semantic identity surface: pins stripped at every node.
pe_semantic(Goal, Text) :-
    pe_emit(Goal, semantic, TextA),
    atom_string(TextA, Text).

%% pe_full(+Goal, -Text)
%  The provenance surface: semantic form plus pins.
pe_full(Goal, Text) :-
    pe_emit(Goal, full, TextA),
    atom_string(TextA, Text).

% pin(E, P): full keeps the pin, semantic strips it — at every depth.
pe_emit(pin(Expr, Pin), Mode, Text) :-
    !,
    pe_emit(Expr, Mode, Inner),
    (   Mode == full
    ->  render_surface([form=pinned(Inner, Pin)], Text)
    ;   Text = Inner
    ).
% mod(B, M): emit the base, then the modified form.
pe_emit(mod(Base, Mod), Mode, Text) :-
    !,
    pe_emit(Base, Mode, Inner),
    render_surface([form=modded(Inner, Mod)], Text).
% numeric positional literal (e.g. max's first argument).
pe_emit(N, _Mode, Text) :-
    number(N),
    !,
    render_value(lit(N), Text).
% bare atom.
pe_emit(A, _Mode, Text) :-
    atom(A),
    pe_atom(A),
    !,
    render_surface([form=bare(A)], Text).
% operator application: introspect, split kwargs from positionals,
% resolve defaults, sort, recurse, dispatch.
pe_emit(Goal, Mode, Text) :-
    compound(Goal),
    compound_name_arguments(Goal, Name, RawArgs),
    pe_operator(Name),
    !,
    partition_kwargs(RawArgs, Name, Positional, KwPairs0),
    resolve_defaults(Name, KwPairs0, KwPairs),
    msort(KwPairs, SortedKw),
    maplist(render_kwarg(Name, Mode), SortedKw, KwTexts),
    atomic_list_concat(KwTexts, ',', KwCore),
    maplist(pe_emit_child(Mode), Positional, ArgTexts),
    build_form(Name, ArgTexts, Form),
    kw_tail(Positional, KwCore, KwTail),
    render_surface([form=Form, kw=KwTail], Text).
pe_emit(Goal, _Mode, _) :-
    throw(error(pe_emit(unknown_form(Goal)), _)).

pe_emit_child(Mode, Child, Text) :- pe_emit(Child, Mode, Text).

%% partition_kwargs(+RawArgs, +Op, -Positional, -KwPairs)
%  An argument is a kwarg iff it is a unary compound whose functor is
%  a declared kwarg of this operator; everything else is positional.
%  Unambiguous under v0.4: no kwarg name is an operator or atom name.
%  (The list is the first argument so clause indexing keeps the walk
%  deterministic.)
partition_kwargs([], _, [], []).
partition_kwargs([A|Rest], Op, Pos, [K-V|Kws]) :-
    compound(A),
    compound_name_arguments(A, K, [V]),
    pe_kwspec(Op, K, _, _),
    !,
    partition_kwargs(Rest, Op, Pos, Kws).
partition_kwargs([A|Rest], Op, [A|Pos], Kws) :-
    partition_kwargs(Rest, Op, Pos, Kws).

%% resolve_defaults(+Op, +Given, -Resolved)
%  Every declared default not explicitly given is added — canonical
%  strings carry resolved defaults explicitly (v0.4 rule).
resolve_defaults(Op, Given, Resolved) :-
    findall(K-D,
            ( pe_kwspec(Op, K, _, default(D)),
              \+ memberchk(K-_, Given)
            ),
            Defaults),
    append(Given, Defaults, Resolved).

%% render_kwarg(+Op, +Mode, +K-V, -Text)
%  A kwarg renders as {{K}}={{V}} through the value template's kw/2
%  case, with V rendered first according to its declared kind.
render_kwarg(Op, Mode, K-V, Text) :-
    once(pe_kwspec(Op, K, Kind, _)),
    render_kw_value(Kind, Mode, V, VText),
    render_value(kw(K, VText), Text).

render_kw_value(Kind, _Mode, V, Text) :-
    string_like_kind(Kind),
    !,
    json_escape(V, Escaped),
    render_value(str(Escaped), Text).
render_kw_value(number, _Mode, V, Text) :-
    !,
    render_value(lit(V), Text).
render_kw_value(int, _Mode, V, Text) :-
    !,
    render_value(lit(V), Text).
render_kw_value(number_list, Mode, V, Text) :-
    !,
    render_list(Mode, V, Text).
render_kw_value(int_list, Mode, V, Text) :-
    !,
    render_list(Mode, V, Text).
render_kw_value(_NodeKind, Mode, V, Text) :-
    % a kwarg whose declared kind is an output type (mu -> judge)
    % carries a node: render it as an expression
    pe_emit(V, Mode, Text).

% estimand and impl are enumerated kinds that render as strings on the
% v0.4 surface (quoted, JSON-escaped), exactly like string.
string_like_kind(string).
string_like_kind(estimand).
string_like_kind(impl).

render_list(_Mode, Items, Text) :-
    maplist([I, T]>>render_value(lit(I), T), Items, Rendered),
    atomic_list_concat(Rendered, ',', Joined),
    render_value(list(Joined), Text).

%% build_form(+Name, +ArgTexts, -Form)
%  Variadic operators take their pre-joined args in one template slot;
%  fixed-arity operators get one slot per argument; margin (0-ary
%  operator, required kwarg) dispatches as a bare tag.
build_form(Name, ArgTexts, Form) :-
    (   pe_variadic(Name)
    ->  atomic_list_concat(ArgTexts, ',', Joined),
        Form =.. [Name, Joined]
    ;   ArgTexts == []
    ->  Form = Name
    ;   Form =.. [Name|ArgTexts]
    ).

%% kw_tail(+Positional, +KwCore, -KwTail)
%  The comma between positional args and the kwarg run exists only
%  when both sides are non-empty (v0.4 canonical join rule).
kw_tail(_, '', '') :- !.
kw_tail([], KwCore, KwCore) :- !.
kw_tail(_, KwCore, KwTail) :-
    atom_concat(',', KwCore, KwTail).

%% ============================================
%% JSON STRING ESCAPING (process_cards.py: json.dumps,
%% ensure_ascii=False — non-ASCII passes through)
%% ============================================

json_escape(Value, Escaped) :-
    (   string(Value) -> S = Value
    ;   atom(Value)   -> atom_string(Value, S)
    ;   throw(error(pe_emit(bad_string_value(Value)), _))
    ),
    string_chars(S, Chars),
    maplist(json_escape_char, Chars, Pieces),
    atomic_list_concat(Pieces, Escaped).

json_escape_char('\\', '\\\\') :- !.
json_escape_char('"',  '\\"') :- !.
json_escape_char('\n', '\\n') :- !.
json_escape_char('\t', '\\t') :- !.
json_escape_char('\r', '\\r') :- !.
json_escape_char('\b', '\\b') :- !.
json_escape_char('\f', '\\f') :- !.
json_escape_char(C, Out) :-
    char_code(C, Code),
    Code < 0x20,
    !,
    format(atom(Out), '\\u~|~`0t~16r~4+', [Code]).
json_escape_char(C, C).
