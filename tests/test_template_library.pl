% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% template_library case libraries, and the LLVM runtime assembled from them
% (src/unifyweaver/targets/wam_llvm_runtime_libs.pl). The assembly must be the
% old monolith byte for byte; that is checked by golden builds in the PR, here we
% pin the parser and the case-set validation that keep it so.

:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/core/template_library').
:- use_module('../src/unifyweaver/targets/wam_llvm_runtime_libs').

:- begin_tests(template_library).

fixture(Text, Path) :-
    tmp_file(tpl_lib, Path),
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]),
                       write(S, Text), close(S)).

cases_of(Text, Cases) :-
    fixture(Text, Path),
    template_library_cases(Path, Cases).

% The header before the first case is dropped; bodies are raw text up to the
% next marker, including the LF that ends each one and data bytes in c"...".
test(parses_cases_raw) :-
    cases_of('{{match fn}}\n; header\n{{case a}}\n; a\ndefine void @a() {\n  ret void\n}\n{{case b.c}}@g = private constant [3 x i8] c"%s\\00"\n}{{/match}}\n',
             Cases),
    assertion(Cases == [ a-'\n; a\ndefine void @a() {\n  ret void\n}\n',
                         'b.c'-'@g = private constant [3 x i8] c"%s\\00"\n}' ]).

test(requires_single_final_lf, [error(template_library(malformed_case_library(_)))]) :-
    cases_of('{{match fn}}{{case a}}x{{/match}}\n\n', _).

test(requires_final_lf, [error(template_library(malformed_case_library(_)))]) :-
    cases_of('{{match fn}}{{case a}}x{{/match}}', _).

test(rejects_default, [error(template_library(malformed_case_library(_)))]) :-
    cases_of('{{match fn}}{{case a}}x{{default}}y{{/match}}\n', _).

test(rejects_nested_match, [error(template_library(malformed_case_library(_)))]) :-
    cases_of('{{match fn}}{{case a}}{{match g}}{{case b}}y{{/match}}{{/match}}\n', _).

test(rejects_duplicate_case, [error(template_library(duplicate_case(_)))]) :-
    cases_of('{{match fn}}{{case a}}x{{case a}}y{{/match}}\n', _).

% Every runtime library's case set equals its table rows, and the assembled
% runtime holds every chunk once, with no markers left and the hole filled.
test(runtime_libraries_match_table) :-
    wam_llvm_runtime_check,
    wam_llvm_runtime_ir([dirent_name_ptr='HOLE_FILLED'], IR),
    assertion(\+ sub_atom(IR, _, _, _, '{{')),
    assertion(sub_atom(IR, _, _, _, 'HOLE_FILLED')),
    split_string(IR, "\n", "", Lines),
    findall(Name, ( member(L, Lines), define_name(L, Name) ), Defined),
    findall(Name, ( wam_llvm_runtime_chunk(builtin_dispatch, Name, _),
                    Name \== wam_stream_handle_globals ), Table),
    assertion(Defined == Table).

% Every unit assembles, holes and all; a unit without holes has no markers left.
test(every_unit_assembles) :-
    findall(U, wam_llvm_runtime_chunk(U, _, _), Us0),
    sort(Us0, Us),
    assertion(length(Us, 12)),
    forall(( member(U, Us), U \== builtin_dispatch, U \== meta_call ),
           ( wam_llvm_runtime_unit_ir(U, [], IR),
             assertion(\+ sub_atom(IR, _, _, _, '{{')) )).

% meta_call's six holes are filled from the dict (the 22 positional ~w of the
% format string it replaced).
test(meta_call_holes_filled) :-
    wam_llvm_runtime_unit_ir(meta_call,
        [size=3, count=2, atom_rows='ATOMS', functor_rows='FUNCTORS',
         arity_rows='ARITIES', label_rows='LABELS'], IR),
    assertion(\+ sub_atom(IR, _, _, _, '{{')),
    assertion(sub_atom(IR, _, _, _, '@wam_meta_call_atom_ids = private constant [3 x i64] [\nATOMS\n]')),
    assertion(sub_atom(IR, _, _, _, 'icmp sge i32 %gi, 2')).

test(unknown_unit_is_an_error, [error(wam_llvm_runtime(unknown_unit(nope)))]) :-
    wam_llvm_runtime_unit_ir(nope, [], _).

define_name(Line, Name) :-
    string_concat("define ", _, Line),
    sub_string(Line, At, _, _, "@"),
    !,
    Start is At + 1,
    sub_string(Line, Start, _, 0, Rest),
    sub_string(Rest, Len, _, _, "("),
    !,
    sub_string(Rest, 0, Len, _, NameS),
    atom_string(Name, NameS).

% A library that drifts from the table is an error naming what is missing.
test(case_set_mismatch_is_an_error,
     [ setup(drop_case(cache, wam_cache_close, Saved)),
       cleanup(restore_cases(cache, Saved)),
       error(wam_llvm_runtime(case_set_mismatch(cache, missing([wam_cache_close]), unknown([])))) ]) :-
    wam_llvm_runtime_check.

lib_path(Lib, Path) :-
    atomic_list_concat(['templates/targets/llvm_wam/runtime/', Lib, '.ll.mustache'], Path).

drop_case(Lib, Name, Cases) :-
    lib_path(Lib, Path),
    template_library_cases(Path, Cases),
    exclude([N-_]>>(N == Name), Cases, Fewer),
    retractall(template_library:template_library_cases_cache(Path, _)),
    assertz(template_library:template_library_cases_cache(Path, Fewer)),
    retractall(wam_llvm_runtime_libs:wam_llvm_runtime_checked).

restore_cases(Lib, Cases) :-
    lib_path(Lib, Path),
    retractall(template_library:template_library_cases_cache(Path, _)),
    assertz(template_library:template_library_cases_cache(Path, Cases)),
    retractall(wam_llvm_runtime_libs:wam_llvm_runtime_checked).

:- end_tests(template_library).
