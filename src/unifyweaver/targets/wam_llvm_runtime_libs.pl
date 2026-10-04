% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_llvm_runtime_libs -- the LLVM runtime (builtin dispatch + plawk runtime),
% assembled from concern libraries under templates/targets/llvm_wam/runtime/.
%
% Each library is a case library (template_library_cases/2): one {{case}} per
% runtime chunk, where a chunk is a define together with the comments and globals
% that precede it. wam_llvm_runtime_chunk/2 below is the single source of truth
% for which chunks exist, which library holds each, and the assembly order (the
% order of the original monolith, so the emitted module is byte-identical).
% Every library's case set must equal its rows here: a missing, unknown or
% duplicate case is an error, not a silent drop.
% See docs/design/PLAN_TEMPLATE_REFACTOR.md (sections 4 and 6, PR 2).

:- module(wam_llvm_runtime_libs, [
    wam_llvm_runtime_ir/2,       % +Dict, -IR
    wam_llvm_runtime_chunk/2,    % ?Name, ?Library
    wam_llvm_runtime_check/0
]).

:- use_module(library(lists)).
:- use_module(library(pairs)).
:- use_module('../core/template_system', [render_template/3]).
:- use_module('../core/template_library', [template_library_cases/2]).

:- dynamic wam_llvm_runtime_checked/0.

%% wam_llvm_runtime_ir(+Dict, -IR) is det.
%  The assembled runtime with its holes filled from Dict (today: dirent_name_ptr).
wam_llvm_runtime_ir(Dict, IR) :-
    wam_llvm_runtime_check,
    findall(Body,
            ( wam_llvm_runtime_chunk(Name, Lib),
              wam_llvm_runtime_lib_cases(Lib, Cases),
              memberchk(Name-Body, Cases)
            ),
            Bodies),
    atomic_list_concat(Bodies, Text),
    render_template(Text, Dict, IR0),
    atom_string(IR, IR0).

%% wam_llvm_runtime_check is det.
%  Validate every library's case set against the table, once per process.
wam_llvm_runtime_check :-
    wam_llvm_runtime_checked,
    !.
wam_llvm_runtime_check :-
    findall(Name, wam_llvm_runtime_chunk(Name, _), Names),
    (   msort(Names, Sorted), append(_, [Dup, Dup | _], Sorted)
    ->  throw(error(wam_llvm_runtime(duplicate_table_row(Dup)), _))
    ;   true
    ),
    findall(Lib, wam_llvm_runtime_chunk(_, Lib), Libs0),
    sort(Libs0, Libs),
    forall(member(Lib, Libs), wam_llvm_runtime_check_lib(Lib)),
    assertz(wam_llvm_runtime_checked).

wam_llvm_runtime_check_lib(Lib) :-
    wam_llvm_runtime_lib_cases(Lib, Cases),
    pairs_keys(Cases, Have0),
    findall(Name, wam_llvm_runtime_chunk(Name, Lib), Want0),
    sort(Have0, Have),
    sort(Want0, Want),
    subtract(Want, Have, Missing),
    subtract(Have, Want, Unknown),
    (   Missing == [], Unknown == []
    ->  true
    ;   throw(error(wam_llvm_runtime(case_set_mismatch(Lib, missing(Missing), unknown(Unknown))), _))
    ).

wam_llvm_runtime_lib_cases(Lib, Cases) :-
    atomic_list_concat(['templates/targets/llvm_wam/runtime/', Lib, '.ll.mustache'], Path),
    template_library_cases(Path, Cases).

%% wam_llvm_runtime_chunk(?Name, ?Library)
%  One row per runtime chunk, in assembly order.
wam_llvm_runtime_chunk(wam_stream_handle_globals, streams).
wam_llvm_runtime_chunk(wam_assoc_i64_new, assoc_table).
wam_llvm_runtime_chunk(wam_assoc_i64_resize, assoc_table).
wam_llvm_runtime_chunk(wam_assoc_i64_get, assoc_table).
wam_llvm_runtime_chunk(wam_assoc_i64_exists, assoc_table).
wam_llvm_runtime_chunk(wam_assoc_i64_print, assoc_table).
wam_llvm_runtime_chunk(wam_assoc_str_print, assoc_table).
wam_llvm_runtime_chunk(wam_assoc_i64_delete, assoc_table).
wam_llvm_runtime_chunk(wam_str_split_into, fields).
wam_llvm_runtime_chunk(wam_str_split_into_re, fields).
wam_llvm_runtime_chunk(wam_fields_new, fields).
wam_llvm_runtime_chunk(wam_fields_get, fields).
wam_llvm_runtime_chunk(wam_fields_set, fields).
wam_llvm_runtime_chunk(wam_fields_join, fields).
wam_llvm_runtime_chunk(wam_fields_join_str, fields).
wam_llvm_runtime_chunk(wam_fields_free, fields).
wam_llvm_runtime_chunk(wam_subsep_comp_slice, fields).
wam_llvm_runtime_chunk(wam_intern_subsep_key_comp, fields).
wam_llvm_runtime_chunk(wam_fields_new_re, fields).
wam_llvm_runtime_chunk(wam_assoc_i64_inc, assoc_table).
wam_llvm_runtime_chunk(wam_assoc_f64_add, assoc_table).
wam_llvm_runtime_chunk(wam_assoc_f64_set, assoc_table).
wam_llvm_runtime_chunk(wam_assoc_f64_print, assoc_table).
wam_llvm_runtime_chunk(wam_assoc_f64_value_at, assoc_table).
wam_llvm_runtime_chunk(wam_assoc_i64_set, assoc_table).
wam_llvm_runtime_chunk(wam_assoc_i64_free, assoc_table).
wam_llvm_runtime_chunk(wam_assoc_i64_iter_next, assoc_table).
wam_llvm_runtime_chunk(wam_assoc_i64_key_at, assoc_table).
wam_llvm_runtime_chunk(wam_assoc_i64_value_at, assoc_table).
wam_llvm_runtime_chunk(wam_cache_load, cache).
wam_llvm_runtime_chunk(wam_cache_open, cache).
wam_llvm_runtime_chunk(wam_cache_commit, cache).
wam_llvm_runtime_chunk(wam_cache_close, cache).
wam_llvm_runtime_chunk(wam_cache_commit_str, cache).
wam_llvm_runtime_chunk(wam_cache_load_str, cache).
wam_llvm_runtime_chunk(wam_regex_field_match, regex).
wam_llvm_runtime_chunk(wam_fs_regex_field_slice_value, regex).
wam_llvm_runtime_chunk(wam_fs_regex_field_count_value, regex).
wam_llvm_runtime_chunk(wam_regex_match, regex).
wam_llvm_runtime_chunk(wam_regex_gsub, regex).
wam_llvm_runtime_chunk(wam_looks_numeric, strnum).
wam_llvm_runtime_chunk(wam_strnum_cmp, strnum).
wam_llvm_runtime_chunk(wam_strnum_cmp_slices, strnum).
wam_llvm_runtime_chunk(wam_intern_i64_decimal, strnum).
wam_llvm_runtime_chunk(wam_strnum_cmp_int, strnum).
wam_llvm_runtime_chunk(wam_awk_num_is_integral, strnum).
wam_llvm_runtime_chunk(wam_awk_num_fmt, strnum).
wam_llvm_runtime_chunk(wam_print_awk_number, strnum).
wam_llvm_runtime_chunk(wam_strnum_cmp_double, strnum).
wam_llvm_runtime_chunk(wam_awk_numeric_start, strnum).
wam_llvm_runtime_chunk(wam_awk_strtod, strnum).
wam_llvm_runtime_chunk(wam_awk_f64_to_i64, strnum).
wam_llvm_runtime_chunk(wam_awk_field_int_value, strnum).
wam_llvm_runtime_chunk(wam_atom_field_f64_value, strnum).
wam_llvm_runtime_chunk(wam_stream_fail_value, streams).
wam_llvm_runtime_chunk(wam_stream_open_value, streams).
wam_llvm_runtime_chunk(wam_stream_open_fd_value, streams).
wam_llvm_runtime_chunk(wam_rs_regex_init, regex).
wam_llvm_runtime_chunk(wam_rt_clear, regex).
wam_llvm_runtime_chunk(wam_rt_set, regex).
wam_llvm_runtime_chunk(wam_stream_reader_set_replay, streams).
wam_llvm_runtime_chunk(wam_rs_regex_find, regex).
wam_llvm_runtime_chunk(wam_stream_read_line_value, streams).
wam_llvm_runtime_chunk(wam_stream_read_line_transient_value, streams).
wam_llvm_runtime_chunk(wam_stream_read_record, streams).
wam_llvm_runtime_chunk(wam_stream_close_value, streams).
wam_llvm_runtime_chunk(wam_getline_file, streams).
wam_llvm_runtime_chunk(wam_getline_main_var, streams).
wam_llvm_runtime_chunk(wam_getline_main_record, streams).
wam_llvm_runtime_chunk(wam_getline_file_record, streams).
wam_llvm_runtime_chunk(wam_getline_pipe, streams).
wam_llvm_runtime_chunk(wam_getline_pipe_record, streams).
wam_llvm_runtime_chunk(wam_environ_get, streams).
wam_llvm_runtime_chunk(plawk_cmdline_load, streams).
wam_llvm_runtime_chunk(wam_argc, streams).
wam_llvm_runtime_chunk(wam_argv_get, streams).
wam_llvm_runtime_chunk(wam_atom_prefix_value, fields).
wam_llvm_runtime_chunk(wam_is_field_whitespace, fields).
wam_llvm_runtime_chunk(wam_atom_field_eq_value, fields).
wam_llvm_runtime_chunk(wam_atom_field_slice_value, fields).
wam_llvm_runtime_chunk(wam_atom_field_count_value, fields).
wam_llvm_runtime_chunk(wam_atom_field_length_value, fields).
wam_llvm_runtime_chunk(wam_subslice_value, fields).
wam_llvm_runtime_chunk(wam_awk_field_index_checked, fields).
wam_llvm_runtime_chunk(wam_atom_field_subslice_value, fields).
wam_llvm_runtime_chunk(wam_slice_index_value, fields).
wam_llvm_runtime_chunk(wam_atom_field_index_value, fields).
wam_llvm_runtime_chunk(wam_slice_i64_parse_value, fields).
wam_llvm_runtime_chunk(wam_slice_i64_cmp_value, fields).
wam_llvm_runtime_chunk(wam_i64_cmp_value, fields).
wam_llvm_runtime_chunk(wam_atom_field_i64_value, fields).
wam_llvm_runtime_chunk(wam_atom_field_cstr, fields).
wam_llvm_runtime_chunk(wam_atom_field_strnum_cmp_int, fields).
wam_llvm_runtime_chunk(wam_atom_field_i64_cmp_value, fields).
wam_llvm_runtime_chunk(wam_atom_field_str_cmp_value, fields).
wam_llvm_runtime_chunk(wam_print_ascii_lower_slice, fields).
wam_llvm_runtime_chunk(wam_print_ascii_upper_slice, fields).
wam_llvm_runtime_chunk(wam_byte_is_ws, term_reader).
wam_llvm_runtime_chunk(wam_make_atomic, term_reader).
wam_llvm_runtime_chunk(wam_is_term_delim, term_reader).
wam_llvm_runtime_chunk(wam_skip_ws, term_reader).
wam_llvm_runtime_chunk(wam_alloc_cons, term_reader).
wam_llvm_runtime_chunk(wam_is_symbol_char, term_reader).
wam_llvm_runtime_chunk(wam_is_alnum, term_reader).
wam_llvm_runtime_chunk(wam_infix_op, term_reader).
wam_llvm_runtime_chunk(wam_build_binop, term_reader).
wam_llvm_runtime_chunk(wam_build_unop, term_reader).
wam_llvm_runtime_chunk(wam_parse_expr, term_reader).
wam_llvm_runtime_chunk(wam_parse_list, term_reader).
wam_llvm_runtime_chunk(wam_var_ref, term_reader).
wam_llvm_runtime_chunk(wam_parse_primary, term_reader).
wam_llvm_runtime_chunk(wam_sb_reserve, term_reader).
wam_llvm_runtime_chunk(wam_sb_putc, term_reader).
wam_llvm_runtime_chunk(wam_sb_puts, term_reader).
wam_llvm_runtime_chunk(wam_sb_putn, term_reader).
wam_llvm_runtime_chunk(wam_functor_is_cons, term_reader).
wam_llvm_runtime_chunk(wam_term_to_sb, term_reader).
wam_llvm_runtime_chunk(execute_builtin, builtin_dispatch).
