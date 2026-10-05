% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_llvm_runtime_libs -- the static LLVM runtime of the WAM target, assembled
% from concern libraries under templates/targets/llvm_wam/runtime/.
%
% Each library is a case library (template_library_cases/2): one {{case}} per
% runtime chunk, where a chunk is a define together with the comments and globals
% that precede it. A UNIT is what one emitter predicate of wam_llvm_target.pl
% returns (the builtin dispatch runtime, @backtrack, the .wamo loader, ...); a
% unit may draw on several libraries and a library may serve several units.
% wam_llvm_runtime_chunk/3 below is the single source of truth for which chunks
% exist, which unit and library each belongs to, and each unit's assembly order
% (the order of the quoted atom it replaced, so the emitted IR is byte-identical).
% Every library's case set must equal its rows here: a missing, unknown or
% duplicate case is an error, not a silent drop.
% See docs/design/PLAN_TEMPLATE_REFACTOR.md (sections 4 and 6, PR 2).

:- module(wam_llvm_runtime_libs, [
    wam_llvm_runtime_ir/2,       % +Dict, -IR (the builtin_dispatch unit)
    wam_llvm_runtime_unit_ir/3,  % +Unit, +Dict, -IR
    wam_llvm_runtime_chunk/3,    % ?Unit, ?Name, ?Library
    wam_llvm_runtime_check/0
]).

:- use_module(library(lists)).
:- use_module(library(pairs)).
:- use_module('../core/template_system', [render_template/3]).
:- use_module('../core/template_library', [template_library_cases/2]).

:- dynamic wam_llvm_runtime_checked/0.

%% wam_llvm_runtime_ir(+Dict, -IR) is det.
%  The builtin dispatch runtime with its hole filled from Dict (dirent_name_ptr).
wam_llvm_runtime_ir(Dict, IR) :-
    wam_llvm_runtime_unit_ir(builtin_dispatch, Dict, IR).

%% wam_llvm_runtime_unit_ir(+Unit, +Dict, -IR) is det.
%  Unit's chunks concatenated in table order, as an atom. A unit without holes
%  passes Dict = [] and is returned exactly as assembled (no render pass).
wam_llvm_runtime_unit_ir(Unit, Dict, IR) :-
    wam_llvm_runtime_check,
    (   wam_llvm_runtime_chunk(Unit, _, _)
    ->  true
    ;   throw(error(wam_llvm_runtime(unknown_unit(Unit)), _))
    ),
    findall(Body,
            ( wam_llvm_runtime_chunk(Unit, Name, Lib),
              wam_llvm_runtime_lib_cases(Lib, Cases),
              memberchk(Name-Body, Cases)
            ),
            Bodies),
    atomic_list_concat(Bodies, Text),
    (   Dict == []
    ->  IR = Text
    ;   render_template(Text, Dict, IR0),
        atom_string(IR, IR0)
    ).

%% wam_llvm_runtime_check is det.
%  Validate every library's case set against the table, once per process.
wam_llvm_runtime_check :-
    wam_llvm_runtime_checked,
    !.
wam_llvm_runtime_check :-
    findall(Name, wam_llvm_runtime_chunk(_, Name, _), Names),
    (   msort(Names, Sorted), append(_, [Dup, Dup | _], Sorted)
    ->  throw(error(wam_llvm_runtime(duplicate_table_row(Dup)), _))
    ;   true
    ),
    findall(Lib, wam_llvm_runtime_chunk(_, _, Lib), Libs0),
    sort(Libs0, Libs),
    forall(member(Lib, Libs), wam_llvm_runtime_check_lib(Lib)),
    assertz(wam_llvm_runtime_checked).

wam_llvm_runtime_check_lib(Lib) :-
    wam_llvm_runtime_lib_cases(Lib, Cases),
    pairs_keys(Cases, Have0),
    findall(Name, wam_llvm_runtime_chunk(_, Name, Lib), Want0),
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

%% wam_llvm_runtime_chunk(?Unit, ?Name, ?Library)
%  One row per runtime chunk; within a unit, rows are in assembly order.
wam_llvm_runtime_chunk(builtin_dispatch, wam_stream_handle_globals, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_i64_new, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_i64_resize, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_i64_get, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_i64_exists, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_i64_print, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_str_print, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_i64_delete, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_str_split_into, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_str_split_into_re, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_fields_new, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_fields_get, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_fields_set, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_fields_join, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_fields_join_str, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_fields_free, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_subsep_comp_slice, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_intern_subsep_key_comp, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_fields_new_re, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_i64_inc, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_f64_add, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_f64_set, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_f64_print, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_f64_value_at, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_i64_set, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_i64_free, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_i64_iter_next, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_i64_key_at, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_assoc_i64_value_at, assoc_table).
wam_llvm_runtime_chunk(builtin_dispatch, wam_cache_load, cache).
wam_llvm_runtime_chunk(builtin_dispatch, wam_cache_open, cache).
wam_llvm_runtime_chunk(builtin_dispatch, wam_cache_commit, cache).
wam_llvm_runtime_chunk(builtin_dispatch, wam_cache_close, cache).
wam_llvm_runtime_chunk(builtin_dispatch, wam_cache_commit_str, cache).
wam_llvm_runtime_chunk(builtin_dispatch, wam_cache_load_str, cache).
wam_llvm_runtime_chunk(builtin_dispatch, wam_regex_field_match, regex).
wam_llvm_runtime_chunk(builtin_dispatch, wam_fs_regex_field_slice_value, regex).
wam_llvm_runtime_chunk(builtin_dispatch, wam_fs_regex_field_count_value, regex).
wam_llvm_runtime_chunk(builtin_dispatch, wam_regex_match, regex).
wam_llvm_runtime_chunk(builtin_dispatch, wam_regex_gsub, regex).
wam_llvm_runtime_chunk(builtin_dispatch, wam_looks_numeric, strnum).
wam_llvm_runtime_chunk(builtin_dispatch, wam_strnum_cmp, strnum).
wam_llvm_runtime_chunk(builtin_dispatch, wam_strnum_cmp_slices, strnum).
wam_llvm_runtime_chunk(builtin_dispatch, wam_intern_i64_decimal, strnum).
wam_llvm_runtime_chunk(builtin_dispatch, wam_strnum_cmp_int, strnum).
wam_llvm_runtime_chunk(builtin_dispatch, wam_awk_num_is_integral, strnum).
wam_llvm_runtime_chunk(builtin_dispatch, wam_awk_num_fmt, strnum).
wam_llvm_runtime_chunk(builtin_dispatch, wam_print_awk_number, strnum).
wam_llvm_runtime_chunk(builtin_dispatch, wam_strnum_cmp_double, strnum).
wam_llvm_runtime_chunk(builtin_dispatch, wam_awk_numeric_start, strnum).
wam_llvm_runtime_chunk(builtin_dispatch, wam_awk_strtod, strnum).
wam_llvm_runtime_chunk(builtin_dispatch, wam_awk_f64_to_i64, strnum).
wam_llvm_runtime_chunk(builtin_dispatch, wam_awk_field_int_value, strnum).
wam_llvm_runtime_chunk(builtin_dispatch, wam_atom_field_f64_value, strnum).
wam_llvm_runtime_chunk(builtin_dispatch, wam_stream_fail_value, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_stream_open_value, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_stream_open_fd_value, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_rs_regex_init, regex).
wam_llvm_runtime_chunk(builtin_dispatch, wam_rt_clear, regex).
wam_llvm_runtime_chunk(builtin_dispatch, wam_rt_set, regex).
wam_llvm_runtime_chunk(builtin_dispatch, wam_stream_reader_set_replay, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_rs_regex_find, regex).
wam_llvm_runtime_chunk(builtin_dispatch, wam_stream_read_line_value, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_stream_read_line_transient_value, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_stream_read_record, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_stream_close_value, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_getline_file, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_getline_main_var, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_getline_main_record, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_getline_file_record, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_getline_pipe, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_getline_pipe_record, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_environ_get, streams).
wam_llvm_runtime_chunk(builtin_dispatch, plawk_cmdline_load, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_argc, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_argv_get, streams).
wam_llvm_runtime_chunk(builtin_dispatch, wam_atom_prefix_value, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_is_field_whitespace, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_atom_field_eq_value, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_atom_field_slice_value, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_atom_field_count_value, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_atom_field_length_value, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_subslice_value, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_awk_field_index_checked, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_atom_field_subslice_value, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_slice_index_value, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_atom_field_index_value, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_slice_i64_parse_value, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_slice_i64_cmp_value, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_i64_cmp_value, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_atom_field_i64_value, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_atom_field_cstr, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_atom_field_strnum_cmp_int, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_atom_field_i64_cmp_value, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_atom_field_str_cmp_value, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_print_ascii_lower_slice, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_print_ascii_upper_slice, fields).
wam_llvm_runtime_chunk(builtin_dispatch, wam_byte_is_ws, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_make_atomic, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_is_term_delim, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_skip_ws, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_alloc_cons, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_is_symbol_char, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_is_alnum, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_infix_op, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_build_binop, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_build_unop, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_parse_expr, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_parse_list, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_var_ref, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_parse_primary, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_sb_reserve, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_sb_putc, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_sb_puts, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_sb_putn, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_functor_is_cons, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, wam_term_to_sb, term_reader).
wam_llvm_runtime_chunk(builtin_dispatch, execute_builtin, builtin_dispatch).
wam_llvm_runtime_chunk(backtrack, backtrack, backtrack).
wam_llvm_runtime_chunk(unwind_trail, unwind_trail, backtrack).
wam_llvm_runtime_chunk(eval_arith, eval_arith, arith).
wam_llvm_runtime_chunk(eval_arith, wam_arith_name_ok, arith).
wam_llvm_runtime_chunk(eval_arith, eval_arith_value, arith).
wam_llvm_runtime_chunk(copy_term, wam_copy_term_value, copy_term).
wam_llvm_runtime_chunk(copy_term, wam_copy_term_rec, copy_term).
wam_llvm_runtime_chunk(copy_term, wam_collect_vars, copy_term).
wam_llvm_runtime_chunk(copy_term, wam_is_ground, copy_term).
wam_llvm_runtime_chunk(copy_term, wam_deref_keep_var, copy_term).
wam_llvm_runtime_chunk(copy_term, wam_strict_eq, copy_term).
wam_llvm_runtime_chunk(copy_term, wam_numbervars_walk, copy_term).
wam_llvm_runtime_chunk(copy_term, wam_freeze_value, copy_term).
wam_llvm_runtime_chunk(term_cmp, wam_term_cmp, term_cmp).
wam_llvm_runtime_chunk(ssp_emit_segment, ssp_emit_segment, ssp).
wam_llvm_runtime_chunk(dirent_probe, dirent_probe_globals, dirent).
wam_llvm_runtime_chunk(dirent_probe, wam_dirent_d_name_offset, dirent).
wam_llvm_runtime_chunk(wasm_externals, wasm_externals_globals, externals).
wam_llvm_runtime_chunk(wasm_externals, malloc, externals).
wam_llvm_runtime_chunk(wasm_externals, free, externals).
wam_llvm_runtime_chunk(wasm_externals, realloc, externals).
wam_llvm_runtime_chunk(wasm_externals, printf, externals).
wam_llvm_runtime_chunk(wasm_externals, snprintf, externals).
wam_llvm_runtime_chunk(wasm_externals, strcmp, externals).
wam_llvm_runtime_chunk(wasm_externals, putchar, externals).
wam_llvm_runtime_chunk(native_externals, native_externals_declares, externals).
wam_llvm_runtime_chunk(native_externals, plawk_redirect_begin, externals).
wam_llvm_runtime_chunk(native_externals, plawk_redirect_end, externals).
wam_llvm_runtime_chunk(meta_call, meta_call_globals, meta_call).
wam_llvm_runtime_chunk(meta_call, wam_meta_find_atom, meta_call).
wam_llvm_runtime_chunk(meta_call, wam_meta_find_compound, meta_call).
wam_llvm_runtime_chunk(meta_call, wam_dispatch_meta_call, meta_call).
wam_llvm_runtime_chunk(wamo_loader, wamo_next_int, wamo_loader).
wam_llvm_runtime_chunk(wamo_loader, wam_object_load, wamo_loader).
wam_llvm_runtime_chunk(wamo_loader, wam_object_load_bytes, wamo_loader).
wam_llvm_runtime_chunk(wamo_loader, wam_object_call_i64, wamo_loader).
wam_llvm_runtime_chunk(wamo_loader, wam_object_call_f64, wamo_loader).
wam_llvm_runtime_chunk(wamo_loader, wam_object_call_bytes, wamo_loader).
wam_llvm_runtime_chunk(wamo_loader, wam_object_call_record, wamo_loader).
wam_llvm_runtime_chunk(wamo_loader, wam_object_call_assoc, wamo_loader).
wam_llvm_runtime_chunk(wamo_loader, wam_object_call_assoc_str, wamo_loader).
wam_llvm_runtime_chunk(wamo_loader, wam_object_call_posarray, wamo_loader).
wam_llvm_runtime_chunk(wamo_loader, wam_object_call_posarray_str, wamo_loader).
wam_llvm_runtime_chunk(wamo_loader, wamo_read_file, wamo_loader).
wam_llvm_runtime_chunk(wamo_loader, wam_object_entry_index_bytes, wamo_loader).
wam_llvm_runtime_chunk(wamo_loader, wam_object_entry_index, wamo_loader).
wam_llvm_runtime_chunk(wamo_loader, wam_object_vm_entry_pc, wamo_loader).
wam_llvm_runtime_chunk(wamo_loader, wam_object_eval, wamo_loader).
wam_llvm_runtime_chunk(wamo_loader, wam_object_load_cached, wamo_loader).
