:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0

:- module(wam_cpp_templates, [cpp_render_template/3, cpp_render_template_at_root/4,
                             cpp_render_stache/3, cpp_render_stache_at_root/4,
                             cpp_preflight_stache/0,
                             cpp_with_template_root_for_test/2]).

:- use_module(library(readutil), [read_file_to_string/3]).
:- use_module(library(crypto), [crypto_file_hash/3]).
:- use_module('../core/template_system', [render_template/3]).
:- use_module('../core/pattern_stache', [load_stache_file/2, render_stache/3]).

:- dynamic cached_head_constant_stache/3.
:- thread_local test_cpp_template_root/1.

cpp_template_path(runtime_header, 'runtime.h.mustache').
cpp_template_path(main_shim, 'main.cpp.mustache').
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
    (   Vars == []
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
    % These whole-file assets have no substitutions. Reject template syntax
    % here instead of silently leaving a malformed or unknown tag in C++.
    (   (sub_string(Template, _, _, _, "{{") ;
         sub_string(Template, _, _, _, "}}"))
    ->  throw(error(cpp_wam_template_tags(Id, Path),
                    context(cpp_render_template/3, Id)))
    ;   true
    ),
    render_template(Template, [], Text).
