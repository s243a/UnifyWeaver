% SPDX-License-Identifier: MIT
% Copyright (c) 2026 John William Creighton (s243a)

:- module(plawk_native_codegen, [
    plawk_program_native_driver_ir/3
]).

:- use_module('../../../src/unifyweaver/targets/wam_llvm_target',
    [llvm_emit_atom_prefix_guard/5,
     llvm_emit_atom_field_eq_guard/7,
     llvm_emit_atom_field_slice/5]).

%% plawk_program_native_driver_ir(+Program, +InputPath, -DriverIR) is semidet.
%
%  Emit the first native Phase-2 PLAWK driver shape:
%
%      /^PREFIX/ { print $0 }
%      $N == "VALUE" { print $0 }
%      $N == "VALUE" { print $M, $K }
%      $N == "VALUE" { count++ } END { print count }
%      $N == "VALUE" { errors++; matches++ } END { print errors, matches }
%
%  The surrounding runtime still comes from write_wam_llvm_project/3. This
%  function emits the target-specific native main that streams the file, lowers
%  the deterministic guard, and prints matching records.
plawk_program_native_driver_ir(
    program([], [rule(Pattern, [print(Fields)])], []),
    InputPath,
    DriverIR
) :-
    atom_codes(InputPath, PathCodes),
    length(PathCodes, PathLen),
    BytesLen is PathLen + 1,
    llvm_c_bytes(PathCodes, PathBytes),
    plawk_pattern_guard_ir(Pattern, GuardGlobalIR-GuardCallIR),
    plawk_print_action_ir(Fields, PrintActionIR),
    format(atom(DriverIR),
'@.plawk_surface_path = private constant [~w x i8] c"~w\\00"
@.plawk_surface_eof = private constant [12 x i8] c"end_of_file\\00"
@.plawk_surface_print_line = private constant [4 x i8] c"%s\\0A\\00"
@.plawk_surface_print_slice = private constant [5 x i8] c"%.*s\\00"
@.plawk_surface_print_space = private constant [2 x i8] c" \\00"
@.plawk_surface_print_newline = private constant [2 x i8] c"\\0A\\00"
~w

define i32 @main() {
entry:
  %path_ptr = getelementptr [~w x i8], [~w x i8]* @.plawk_surface_path, i32 0, i32 0
  %path_id = call i64 @wam_intern_atom(i8* %path_ptr, i64 ~w)
  %path0 = insertvalue %Value undef, i32 0, 0
  %path = insertvalue %Value %path0, i64 %path_id, 1
  %handle = call %Value @wam_stream_open_value(%Value %path)
  %handle_tag = extractvalue %Value %handle, 0
  %handle_is_int = icmp eq i32 %handle_tag, 1
  br i1 %handle_is_int, label %check_handle_value, label %fail_open

check_handle_value:
  %handle_payload = extractvalue %Value %handle, 1
  %handle_ok = icmp sgt i64 %handle_payload, 0
  br i1 %handle_ok, label %loop, label %fail_open

loop:
  %line = call %Value @wam_stream_read_line_value(%Value %handle)
  %line_tag = extractvalue %Value %line, 0
  %line_payload = extractvalue %Value %line, 1
  %line_is_int = icmp eq i32 %line_tag, 1
  %line_bad_payload = icmp slt i64 %line_payload, 0
  %line_bad = and i1 %line_is_int, %line_bad_payload
  br i1 %line_bad, label %fail_read, label %check_line_atom

check_line_atom:
  %line_is_atom = icmp eq i32 %line_tag, 0
  br i1 %line_is_atom, label %check_eof, label %fail_line_tag

check_eof:
  %line_s = call i8* @wam_atom_to_string(i64 %line_payload)
  %eof_s = getelementptr [12 x i8], [12 x i8]* @.plawk_surface_eof, i32 0, i32 0
  %eof_cmp = call i32 @strcmp(i8* %line_s, i8* %eof_s)
  %is_eof = icmp eq i32 %eof_cmp, 0
  br i1 %is_eof, label %close_stream, label %lowered_match

lowered_match:
~w
  br i1 %is_match, label %print_line, label %continue_loop

print_line:
~w
  br label %continue_loop

continue_loop:
  br label %loop

close_stream:
  %close_ok = call i1 @wam_stream_close_value(%Value %handle)
  br i1 %close_ok, label %success, label %fail_close

success:
  ret i32 0

fail_open:
  ret i32 10

fail_read:
  %close_ignore_read = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 11

fail_line_tag:
  %close_ignore_line_tag = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 12

fail_close:
  ret i32 16
}
',
        [BytesLen, PathBytes,
         GuardGlobalIR,
         BytesLen, BytesLen, PathLen, GuardCallIR, PrintActionIR]).

plawk_program_native_driver_ir(
    program([], [rule(Pattern, Actions)], [end([print(PrintFields)])]),
    InputPath,
    DriverIR
) :-
    plawk_scalar_state_slots(Actions, PrintFields, Slots, ActionVars),
    atom_codes(InputPath, PathCodes),
    length(PathCodes, PathLen),
    BytesLen is PathLen + 1,
    llvm_c_bytes(PathCodes, PathBytes),
    plawk_pattern_guard_ir(Pattern, GuardGlobalIR-GuardCallIR),
    plawk_scalar_loop_phi_ir(Slots, LoopPhiIR),
    plawk_scalar_match_update_ir(Slots, ActionVars, MatchUpdateIR),
    plawk_scalar_next_phi_ir(Slots, NextPhiIR),
    plawk_scalar_end_print_ir(PrintFields, Slots, EndPrintIR),
    format(atom(DriverIR),
'@.plawk_surface_path = private constant [~w x i8] c"~w\\00"
@.plawk_surface_eof = private constant [12 x i8] c"end_of_file\\00"
@.plawk_surface_print_i64 = private constant [4 x i8] c"%ld\\00"
@.plawk_surface_print_space = private constant [2 x i8] c" \\00"
@.plawk_surface_print_newline = private constant [2 x i8] c"\\0A\\00"
~w

define i32 @main() {
entry:
  %path_ptr = getelementptr [~w x i8], [~w x i8]* @.plawk_surface_path, i32 0, i32 0
  %path_id = call i64 @wam_intern_atom(i8* %path_ptr, i64 ~w)
  %path0 = insertvalue %Value undef, i32 0, 0
  %path = insertvalue %Value %path0, i64 %path_id, 1
  %handle = call %Value @wam_stream_open_value(%Value %path)
  %handle_tag = extractvalue %Value %handle, 0
  %handle_is_int = icmp eq i32 %handle_tag, 1
  br i1 %handle_is_int, label %check_handle_value, label %fail_open

check_handle_value:
  %handle_payload = extractvalue %Value %handle, 1
  %handle_ok = icmp sgt i64 %handle_payload, 0
  br i1 %handle_ok, label %loop, label %fail_open

loop:
~w
  %line = call %Value @wam_stream_read_line_value(%Value %handle)
  %line_tag = extractvalue %Value %line, 0
  %line_payload = extractvalue %Value %line, 1
  %line_is_int = icmp eq i32 %line_tag, 1
  %line_bad_payload = icmp slt i64 %line_payload, 0
  %line_bad = and i1 %line_is_int, %line_bad_payload
  br i1 %line_bad, label %fail_read, label %check_line_atom

check_line_atom:
  %line_is_atom = icmp eq i32 %line_tag, 0
  br i1 %line_is_atom, label %check_eof, label %fail_line_tag

check_eof:
  %line_s = call i8* @wam_atom_to_string(i64 %line_payload)
  %eof_s = getelementptr [12 x i8], [12 x i8]* @.plawk_surface_eof, i32 0, i32 0
  %eof_cmp = call i32 @strcmp(i8* %line_s, i8* %eof_s)
  %is_eof = icmp eq i32 %eof_cmp, 0
  br i1 %is_eof, label %close_stream, label %lowered_match

lowered_match:
~w
  br i1 %is_match, label %apply_actions, label %keep_state

apply_actions:
~w
  br label %continue_loop

keep_state:
  br label %continue_loop

continue_loop:
~w
  br label %loop

close_stream:
  %close_ok = call i1 @wam_stream_close_value(%Value %handle)
  br i1 %close_ok, label %end_print, label %fail_close

end_print:
~w
  ret i32 0

fail_open:
  ret i32 10

fail_read:
  %close_ignore_read = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 11

fail_line_tag:
  %close_ignore_line_tag = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 12

fail_close:
  ret i32 16
}
',
        [BytesLen, PathBytes,
         GuardGlobalIR,
         BytesLen, BytesLen, PathLen, LoopPhiIR, GuardCallIR,
         MatchUpdateIR, NextPhiIR, EndPrintIR]).

llvm_c_bytes([], '').
llvm_c_bytes([Code | Rest], Bytes) :-
    llvm_c_byte(Code, Byte),
    llvm_c_bytes(Rest, Tail),
    atom_concat(Byte, Tail, Bytes).

llvm_c_byte(Code, Byte) :-
    format(atom(Byte), '\\~|~`0t~16r~2+', [Code]).

plawk_scalar_state_slots(Actions, PrintFields, Slots, ActionVars) :-
    maplist(plawk_scalar_increment_action, Actions, ActionVars),
    ActionVars \== [],
    maplist(plawk_scalar_print_expr, PrintFields, PrintVars),
    append(ActionVars, PrintVars, Names0),
    sort(Names0, Slots).

plawk_scalar_increment_action(inc(var(Name)), Name).

plawk_scalar_print_expr(var(Name), Name).

plawk_scalar_loop_phi_ir(Slots, IR) :-
    phrase(plawk_scalar_loop_phi_lines(Slots, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_scalar_loop_phi_lines([], _) -->
    [].
plawk_scalar_loop_phi_lines([_Name | Rest], Index) -->
    { format(atom(Line),
          '  %slot_~w = phi i64 [0, %check_handle_value], [%next_slot_~w, %continue_loop]',
          [Index, Index]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_scalar_loop_phi_lines(Rest, NextIndex).

plawk_scalar_match_update_ir(Slots, ActionVars, IR) :-
    phrase(plawk_scalar_match_update_lines(Slots, ActionVars, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_scalar_match_update_lines([], _ActionVars, _) -->
    [].
plawk_scalar_match_update_lines([Name | Rest], ActionVars, Index) -->
    { plawk_scalar_increment_count(Name, ActionVars, Count),
      format(atom(Line),
          '  %match_slot_~w = add i64 %slot_~w, ~w',
          [Index, Index, Count]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_scalar_match_update_lines(Rest, ActionVars, NextIndex).

plawk_scalar_increment_count(Name, ActionVars, Count) :-
    include(==(Name), ActionVars, Matches),
    length(Matches, Count).

plawk_scalar_next_phi_ir(Slots, IR) :-
    phrase(plawk_scalar_next_phi_lines(Slots, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_scalar_next_phi_lines([], _) -->
    [].
plawk_scalar_next_phi_lines([_Name | Rest], Index) -->
    { format(atom(Line),
          '  %next_slot_~w = phi i64 [%match_slot_~w, %apply_actions], [%slot_~w, %keep_state]',
          [Index, Index, Index]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_scalar_next_phi_lines(Rest, NextIndex).

plawk_scalar_end_print_ir(PrintFields, Slots, IR) :-
    phrase(plawk_scalar_end_print_lines(PrintFields, Slots, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_scalar_end_print_lines([], _Slots, _) -->
    ['  %end_newline_fmt = getelementptr [2 x i8], [2 x i8]* @.plawk_surface_print_newline, i32 0, i32 0',
     '  %printed_end_newline = call i32 (i8*, ...) @printf(i8* %end_newline_fmt)'].
plawk_scalar_end_print_lines([var(Name) | Rest], Slots, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex),
    { nth0(SlotIndex, Slots, Name),
      format(atom(FmtPtr),
          '  %end_i64_fmt_~w = getelementptr [4 x i8], [4 x i8]* @.plawk_surface_print_i64, i32 0, i32 0',
          [PrintIndex]),
      format(atom(PrintCall),
          '  %printed_end_i64_~w = call i32 (i8*, ...) @printf(i8* %end_i64_fmt_~w, i64 %slot_~w)',
          [PrintIndex, PrintIndex, SlotIndex]),
      NextPrintIndex is PrintIndex + 1
    },
    [FmtPtr, PrintCall],
    plawk_scalar_end_print_lines(Rest, Slots, NextPrintIndex).

plawk_scalar_end_separator_lines(0) -->
    !,
    [].
plawk_scalar_end_separator_lines(PrintIndex) -->
    { format(atom(SpacePtr),
          '  %end_space_fmt_~w = getelementptr [2 x i8], [2 x i8]* @.plawk_surface_print_space, i32 0, i32 0',
          [PrintIndex]),
      format(atom(SpaceCall),
          '  %printed_end_space_~w = call i32 (i8*, ...) @printf(i8* %end_space_fmt_~w)',
          [PrintIndex, PrintIndex])
    },
    [SpacePtr, SpaceCall].

plawk_pattern_guard_ir(prefix(Prefix), GuardIR) :-
    llvm_emit_atom_prefix_guard(plawk_surface_prefix, '%line', Prefix,
        '%is_match', GuardIR).

plawk_pattern_guard_ir(field_eq(Index, Value), GuardIR) :-
    llvm_emit_atom_field_eq_guard(plawk_surface_field_eq, '%line', Index, Value,
        32, '%is_match', GuardIR).

plawk_print_action_ir([field(0)], IR) :-
    !,
    IR = '  %fmt = getelementptr [4 x i8], [4 x i8]* @.plawk_surface_print_line, i32 0, i32 0
  %printed = call i32 (i8*, ...) @printf(i8* %fmt, i8* %line_s)'.
plawk_print_action_ir(Fields, IR) :-
    phrase(plawk_print_fields_ir(Fields, 0), Parts),
    atomic_list_concat(Parts, '\n', IR).

plawk_print_fields_ir([], _) -->
    ['  %newline_fmt = getelementptr [2 x i8], [2 x i8]* @.plawk_surface_print_newline, i32 0, i32 0',
     '  %printed_newline = call i32 (i8*, ...) @printf(i8* %newline_fmt)'].
plawk_print_fields_ir([Field | Rest], Index) -->
    plawk_print_separator_ir(Index),
    plawk_print_field_ir(Field, Index),
    { NextIndex is Index + 1 },
    plawk_print_fields_ir(Rest, NextIndex).

plawk_print_separator_ir(0) -->
    !,
    [].
plawk_print_separator_ir(Index) -->
    { format(atom(SpacePtr),
          '  %space_fmt_~w = getelementptr [2 x i8], [2 x i8]* @.plawk_surface_print_space, i32 0, i32 0',
          [Index]),
      format(atom(SpaceCall),
          '  %printed_space_~w = call i32 (i8*, ...) @printf(i8* %space_fmt_~w)',
          [Index, Index])
    },
    [SpacePtr, SpaceCall].

plawk_print_field_ir(field(0), Index) -->
    { format(atom(LineLen64),
          '  %line_len64_~w = call i64 @strlen(i8* %line_s)',
          [Index]),
      format(atom(LineLen),
          '  %line_len_~w = trunc i64 %line_len64_~w to i32',
          [Index, Index]),
      format(atom(FmtPtr),
          '  %line_fmt_~w = getelementptr [5 x i8], [5 x i8]* @.plawk_surface_print_slice, i32 0, i32 0',
          [Index]),
      format(atom(PrintCall),
          '  %printed_line_~w = call i32 (i8*, ...) @printf(i8* %line_fmt_~w, i32 %line_len_~w, i8* %line_s)',
          [Index, Index, Index])
    },
    [LineLen64, LineLen, FmtPtr, PrintCall].
plawk_print_field_ir(field(FieldIndex), Index) -->
    { FieldIndex > 0,
      format(atom(Base), 'plawk_field_~w', [Index]),
      llvm_emit_atom_field_slice('%line', FieldIndex, 32, Base, SliceIR),
      format(atom(FmtPtr),
          '  %slice_fmt_~w = getelementptr [5 x i8], [5 x i8]* @.plawk_surface_print_slice, i32 0, i32 0',
          [Index]),
      format(atom(PrintCall),
          '  %printed_slice_~w = call i32 (i8*, ...) @printf(i8* %slice_fmt_~w, i32 %~w_len, i8* %~w_ptr)',
          [Index, Index, Base, Base])
    },
    [SliceIR, FmtPtr, PrintCall].
