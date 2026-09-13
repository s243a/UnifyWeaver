:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0

:- module(wam_cpp_templates, [cpp_render_template/3, cpp_render_template_at_root/4,
                             cpp_render_stache/3, cpp_render_stache_at_root/4,
                             cpp_preflight_stache/0,
                             cpp_with_template_root_for_test/2]).

:- use_module(library(readutil), [read_file_to_string/3]).
:- use_module(library(crypto), [crypto_file_hash/3]).
:- use_module('../core/pattern_stache', [load_stache_file/2, render_stache/3]).

:- dynamic cached_head_constant_stache/3.
:- thread_local test_cpp_template_root/1.

cpp_template_path(runtime_header, 'runtime.h.mustache').
cpp_template_path(runtime_source, 'runtime.cpp.mustache').
cpp_template_path(runtime_cell_helpers, 'runtime/cell_helpers.cpp.mustache').
cpp_template_path(runtime_arithmetic_eval, 'runtime/arithmetic_eval.cpp.mustache').
cpp_template_path(runtime_lmdb_fact_source, 'runtime/lmdb_fact_source.cpp.mustache').
cpp_template_path(main_shim, 'main.cpp.mustache').
cpp_template_path(generated_program, 'generated_program.cpp.mustache').
cpp_stache_path(head_constant, 'lowered/head_constant.cpp.stache').

% Scoped fixture root for generator tests. Production callers resolve assets
% from this module; no project option can redirect template loading.
cpp_with_template_root_for_test(Root, Goal) :-
    setup_call_cleanup(
        asserta(test_cpp_template_root(Root), Ref),
        call(Goal),
        erase(Ref)).

cpp_template_root(Root) :-
    test_cpp_template_root(Root),
    !.
cpp_template_root(Root) :-
    source_file(cpp_render_template(_, _, _), Source),
    file_directory_name(Source, ModuleDir),
    directory_file_path(ModuleDir, '../../../templates/targets/cpp_wam', Root).

% Both case bodies are required even when a particular project only uses one.
cpp_preflight_stache :-
    Vars0 = ['I'="", 'CStr'="x", 'AiStr'="A1", 'Ai'="A1"],
    cpp_render_stache(head_constant, [op=head_constant(atom("x"))|Vars0], _),
    cpp_render_stache(head_constant, [op=head_constant(value("1"))|Vars0], _).

cpp_render_stache(Id, Vars, Text) :-
    cpp_template_root(Root),
    cpp_render_stache_at_root(Root, Id, Vars, Text).

cpp_render_stache_at_root(Root, Id, Vars, Text) :-
    (   cpp_stache_path(Id, Name)
    ->  true
    ;   throw(error(domain_error(cpp_wam_stache, Id),
                    context(cpp_render_stache/3, 'unknown C++ WAM stache')))
    ),
    directory_file_path(Root, Name, Path),
    (   valid_head_constant_vars(Vars)
    ->  true
    ;   throw(error(domain_error(cpp_wam_stache_variables, Vars),
                    context(cpp_render_stache/3, Id-Path)))
    ),
    catch((load_head_constant_stache(Path, Template),
           render_stache(Template, Vars, Text)),
          Error,
          throw(error(cpp_wam_stache_failure(Id, Path, Error),
                      context(cpp_render_stache/3, emit_one(get_constant/2))))),
    (   string(Text), Text \== ""
    ->  true
    ;   throw(error(cpp_wam_stache_empty(Id, Path),
                    context(cpp_render_stache/3, emit_one(get_constant/2))))
    ).

% Read and hash on every call: a broken edit at the same path must never use
% an older successful parse, even if the edit preserves size and timestamp.
load_head_constant_stache(Path, Template) :-
    absolute_file_name(Path, FullPath, [access(read), file_errors(error)]),
    crypto_file_hash(FullPath, Digest, [algorithm(sha256)]),
    with_mutex(wam_cpp_head_constant_stache,
        cached_or_load_head_constant(FullPath, Digest, Template)).

cached_or_load_head_constant(Path, Digest, Template) :-
    (   cached_head_constant_stache(Path, Digest, Template)
    ->  true
    ;   retractall(cached_head_constant_stache(Path, _, _)),
        load_stache_file(Path, Template),
        Template = stache(_, Body),
        check_head_constant_tags(Body),
        assertz(cached_head_constant_stache(Path, Digest, Template))
    ).

valid_head_constant_vars(Vars) :-
    ground(Vars),
    Vars = [op=Op, 'I'=I, 'CStr'=CStr, 'AiStr'=Ai, 'Ai'=Reg],
    (   Op = head_constant(atom(Esc)), text_value(Esc)
    ;   Op = head_constant(value(Val)),
        nonempty_text(Val)
    ),
    maplist(text_value, [I, CStr, Ai, Reg]).

text_value(Value) :- atom(Value), !.
text_value(Value) :- string(Value).

nonempty_text(Value) :-
    text_value(Value),
    Value \== '', Value \== "".

check_head_constant_tags(Body) :-
    Prefix = "{{match op}}{{case head_constant(atom(Esc))}}",
    CaseTag = "{{case head_constant(value(CppVal))}}",
    (   string_concat(Prefix, Rest, Body),
        sub_string(Rest, CaseAt, _, _, CaseTag)
    ->  sub_string(Rest, 0, CaseAt, _, AtomBody),
        string_length(CaseTag, CaseTagLength),
        ValueAt is CaseAt + CaseTagLength,
        sub_string(Rest, ValueAt, _, 0, ValueAndClose),
        string_concat(ValueBody, "{{/match}}", ValueAndClose),
        !,
        check_head_constant_case_tags(atom, AtomBody),
        check_head_constant_case_tags(value, ValueBody)
    ;   throw(error(cpp_wam_stache_invalid_cases, _))
    ).

check_head_constant_case_tags(Branch, Body) :-
    (   sub_string(Body, Start, 2, _, "{{")
    ->  After is Start + 2,
        sub_string(Body, After, _, 0, Tail),
        (   sub_string(Tail, End, 2, _, "}}")
        ->  sub_string(Tail, 0, End, _, Tag),
            (   head_constant_tag(Branch, Tag)
            ->  true
            ;   throw(error(cpp_wam_stache_unknown_tag(Tag), _))
            ),
            Next is End + 2,
            sub_string(Tail, Next, _, 0, Rest),
            check_head_constant_case_tags(Branch, Rest)
        ;   throw(error(cpp_wam_stache_unterminated_tag(Tail), _))
        )
    ;   true
    ).

head_constant_tag(_, "I").
head_constant_tag(_, "CStr").
head_constant_tag(_, "AiStr").
head_constant_tag(_, "Ai").
head_constant_tag(atom, "Esc").
head_constant_tag(value, "CppVal").

cpp_render_template(Id, Vars, Text) :-
    cpp_template_root(Root),
    cpp_render_template_at_root(Root, Id, Vars, Text).

% Explicit root is for isolated template fixtures; emitters use the module root.
cpp_render_template_at_root(Root, Id, Vars, Text) :-
    (   cpp_template_path(Id, Name)
    ->  true
    ;   throw(error(domain_error(cpp_wam_template, Id),
                    context(cpp_render_template/3, 'unknown C++ WAM template')))
    ),
    directory_file_path(Root, Name, Path),
    (   valid_cpp_template_vars(Id, Vars)
    ->  true
    ;   throw(error(domain_error(cpp_wam_template_variables, Vars),
                    context(cpp_render_template/3, Id)))
    ),
    catch(read_file_to_string(Path, Template, [encoding(utf8)]),
          Error,
          throw(error(cpp_wam_template_load(Id, Path, Error),
                      context(cpp_render_template/3, Id)))),
    (   Template == ""
    ->  throw(error(cpp_wam_template_empty(Id, Path),
                    context(cpp_render_template/3, Id)))
    ;   true
    ),
    render_cpp_template(Id, Path, Template, Vars, Text).

valid_cpp_template_vars(generated_program,
                        [predicates_code=Predicates, setup_code=Setup]) :-
    !,
    text_value(Predicates),
    text_value(Setup).
valid_cpp_template_vars(Id, Vars) :-
    Id \== generated_program,
    Vars == [].

render_cpp_template(generated_program, Path, Template,
                    [predicates_code=Predicates, setup_code=Setup], Text) :-
    !,
    (   program_shell_parts(Template, Prefix, Middle, Suffix)
    ->  atom_string(Predicates, PredicatesText),
        atom_string(Setup, SetupText),
        string_concat(Prefix, PredicatesText, P1),
        string_concat(P1, Middle, P2),
        string_concat(P2, SetupText, P3),
        string_concat(P3, Suffix, Text)
    ;   throw(error(cpp_wam_template_tags(generated_program, Path),
                    context(cpp_render_template/3, generated_program)))
    ).
render_cpp_template(runtime_source, Path, Template, [], Text) :-
    !,
    (   runtime_shell_parts(Template, Prefix, BetweenCellAndArithmetic,
                            BetweenArithmeticAndLmdb, Suffix)
    ->  file_directory_name(Path, Root),
        cpp_render_template_at_root(Root, runtime_cell_helpers, [], Cell),
        cpp_render_template_at_root(Root, runtime_arithmetic_eval, [], Arithmetic),
        cpp_render_template_at_root(Root, runtime_lmdb_fact_source, [], Lmdb),
        % Source spans are assembled once. Child text is never scanned again.
        atomics_to_string([Prefix, Cell, BetweenCellAndArithmetic, Arithmetic,
                           BetweenArithmeticAndLmdb, Lmdb, Suffix], "", Text)
    ;   throw(error(cpp_wam_template_tags(runtime_source, Path),
                    context(cpp_render_template/3, runtime_source)))
    ).
% Whole-file assets without variables are literal source after tag lint.
% Avoid scanning the large runtime through the generic renderer at each build.
render_cpp_template(Id, Path, Template, [], Text) :-
    (   template_has_tags(Template)
    ->  throw(error(cpp_wam_template_tags(Id, Path),
                    context(cpp_render_template/3, Id)))
    ;   Text = Template
    ).

% Splice only source-template spans. Sequential Mustache replacement would
% reinterpret {{setup_code}} inside an already-inserted predicate fragment.
program_shell_parts(Template, Prefix, Middle, Suffix) :-
    PredMarker = "{{predicates_code}}",
    SetupMarker = "{{setup_code}}",
    findall(At, sub_string(Template, At, _, _, PredMarker), [PredAt]),
    findall(At, sub_string(Template, At, _, _, SetupMarker), [SetupAt]),
    string_length(PredMarker, PredLen),
    string_length(SetupMarker, SetupLen),
    PredEnd is PredAt + PredLen,
    SetupEnd is SetupAt + SetupLen,
    PredEnd =< SetupAt,
    MidLen is SetupAt - PredEnd,
    sub_string(Template, 0, PredAt, _, Prefix),
    sub_string(Template, PredEnd, MidLen, _, Middle),
    sub_string(Template, SetupEnd, _, 0, Suffix),
    \+ template_has_tags(Prefix),
    \+ template_has_tags(Middle),
    \+ template_has_tags(Suffix).

% The runtime shell has three fixed ordered slots. Reject missing, repeated,
% reordered, or unrelated source tags before loading any child section.
runtime_shell_parts(Template, Prefix, BetweenCellAndArithmetic,
                    BetweenArithmeticAndLmdb, Suffix) :-
    CellMarker = "{{cell_helpers}}",
    ArithmeticMarker = "{{arithmetic_eval}}",
    LmdbMarker = "{{lmdb_fact_source}}",
    findall(At, sub_string(Template, At, _, _, CellMarker), [CellAt]),
    findall(At, sub_string(Template, At, _, _, ArithmeticMarker), [ArithmeticAt]),
    findall(At, sub_string(Template, At, _, _, LmdbMarker), [LmdbAt]),
    string_length(CellMarker, CellLen),
    string_length(ArithmeticMarker, ArithmeticLen),
    string_length(LmdbMarker, LmdbLen),
    CellEnd is CellAt + CellLen,
    ArithmeticEnd is ArithmeticAt + ArithmeticLen,
    LmdbEnd is LmdbAt + LmdbLen,
    CellEnd =< ArithmeticAt,
    ArithmeticEnd =< LmdbAt,
    CellMiddleLen is ArithmeticAt - CellEnd,
    ArithmeticMiddleLen is LmdbAt - ArithmeticEnd,
    sub_string(Template, 0, CellAt, _, Prefix),
    sub_string(Template, CellEnd, CellMiddleLen, _, BetweenCellAndArithmetic),
    sub_string(Template, ArithmeticEnd, ArithmeticMiddleLen, _, BetweenArithmeticAndLmdb),
    sub_string(Template, LmdbEnd, _, 0, Suffix),
    \+ template_has_tags(Prefix),
    \+ template_has_tags(BetweenCellAndArithmetic),
    \+ template_has_tags(BetweenArithmeticAndLmdb),
    \+ template_has_tags(Suffix).

template_has_tags(Text) :- sub_string(Text, _, _, _, "{{"), !.
template_has_tags(Text) :- sub_string(Text, _, _, _, "}}").
