% SPDX-License-Identifier: MIT
% Copyright (c) 2026 John William Creighton (s243a)

:- module(plawk_native_codegen, [
    plawk_program_native_driver_ir/3,
    plawk_program_native_driver_ir/4,
    plawk_program_multipass_driver_ir/2,
    plawk_program_multipass_driver_ir/3,
    plawk_query_helper_name/3,
    plawk_query_helper_clause/4,
    plawk_program_query_passes/2,
    plawk_program_materialize_views/2,
    plawk_program_gen_blocks/2,
    plawk_query_pass_supported/1,
    plawk_program_foreign_specs/3,
    plawk_program_foreign_specs/4,
    plawk_program_dyncall_arities/2,
    plawk_program_dyncall_named_entries/2,
    plawk_program_dyncall_named_float_entries/2,
    plawk_program_dyncall_named_blob_entries/2,
    plawk_program_dyncall_rec_arities/2,
    plawk_program_dyncall_named_rec_entries/2,
    plawk_program_dyncall_named_assoc_entries/2,
    plawk_program_dyncall_assoc_arities/2,
    plawk_program_dyncall_named_assoc_str_entries/2,
    plawk_program_dyncall_assoc_str_arities/2,
    plawk_program_dyncall_named_posarray_entries/2,
    plawk_program_dyncall_posarray_arities/2,
    plawk_program_dyncall_named_posarray_str_entries/2,
    plawk_program_dyncall_posarray_str_arities/2,
    plawk_program_dyncall_at_arities/2,
    plawk_program_dyncall_at_named_entries/2,
    plawk_program_dyncall_at_named_float_entries/2,
    plawk_program_dyncall_at_named_blob_entries/2,
    plawk_program_dyncall_at_rec_arities/2,
    plawk_program_dyncall_at_named_rec_entries/2,
    plawk_program_dyncall_float_arities/2,
    plawk_program_dyncall_at_float_arities/2,
    plawk_program_dyncall_blob_arities/2,
    plawk_program_dyncall_at_blob_arities/2,
    plawk_program_dyncache_mode/2,
    plawk_program_dynload_path/2,
    plawk_program_evalc_path/2,
    plawk_program_compile_sites/2,
    plawk_prolog_block_preds/2
]).

:- discontiguous plawk_pattern_guard_ir/5.

%% plawk_prolog_block_preds(+Clauses, -Preds)
%
%  Install the embedded @prolog clauses a plawk_parse_source/3 parse
%  returned, so write_wam_llvm_project/3 can compile them into the
%  same binary as the driver. DCG rules expand; each defined predicate
%  is reset (retractall) before its clauses assert, so recompiling a
%  program replaces rather than accumulates definitions. Preds is the
%  deduplicated user:Name/Arity list in first-definition order --
%  exactly the predicate list write_wam_llvm_project/3 takes.
%  Directives are not clauses and reject the program.
plawk_prolog_block_preds(Clauses0, Preds) :-
    \+ member((:- _Directive), Clauses0),
    maplist(plawk_expand_block_clause, Clauses0, ClausesNested),
    append(ClausesNested, Clauses),
    findall(Name/Arity,
        ( member(Clause, Clauses),
          plawk_block_clause_head(Clause, Head),
          functor(Head, Name, Arity)
        ),
        PIs0),
    plawk_dedupe_keep_order(PIs0, PIs),
    forall(member(Name/Arity, PIs),
        ( functor(Head, Name, Arity),
          retractall(user:Head)
        )),
    forall(member(Clause, Clauses), assertz(user:Clause)),
    findall(user:PI, member(PI, PIs), Preds).

% expand_term/2 may return one clause or a list, and DCG expansion can
% interleave compile-time directives (e.g. discontiguous hints) with
% the clauses -- keep only the clauses.
plawk_expand_block_clause(Clause0, Clauses) :-
    expand_term(Clause0, Expanded),
    ( is_list(Expanded)
    ->  Expanded1 = Expanded
    ;   Expanded1 = [Expanded]
    ),
    exclude(plawk_block_directive, Expanded1, Clauses).

plawk_block_directive((:- _Goal)).

plawk_block_clause_head((Head :- _Body), Head) :- !.
plawk_block_clause_head(Head, Head).

plawk_dedupe_keep_order([], []).
plawk_dedupe_keep_order([PI | Rest0], [PI | Deduped]) :-
    exclude(==(PI), Rest0, Rest),
    plawk_dedupe_keep_order(Rest, Deduped).

:- use_module('../../../src/unifyweaver/targets/wam_llvm_target',
    [llvm_emit_atom_prefix_guard/5,
     llvm_emit_atom_field_eq_guard/7,
     llvm_emit_regex_field_match_guard/7,
     llvm_emit_atom_field_slice/5,
     llvm_emit_atom_field_count/4,
     llvm_emit_atom_field_length/5,
     llvm_emit_atom_field_subslice/7,
     llvm_emit_atom_field_index/7,
     llvm_emit_atom_field_i64_cmp_guard/7,
     llvm_emit_atom_field_i64_or_default/7,
     llvm_emit_c_string_global/5,
     llvm_emit_printf_i64/5,
     llvm_emit_printf_slice/6,
     llvm_emit_printf_string/5,
     llvm_emit_printf_string/6,
     llvm_emit_printf0/5,
     llvm_emit_stream_driver_ir/3,
     llvm_emit_binary_stream_driver_ir/4,
     llvm_emit_varlen_stream_driver_ir/5,
     llvm_emit_ascii_case_slice_print/5]).

%% plawk_program_native_driver_ir(+Program, +InputPath, -DriverIR) is semidet.
%
%  Emit the first native Phase-2 PLAWK driver shape:
%
%      /^PREFIX/ { print $0 }
%      /LITERAL/ { print $0 }
%      $N == "VALUE" { print $0 }
%      $N == "VALUE" { print $M, $K }
%      $N == "VALUE" { count++ } END { print count }
%      $N == "VALUE" { errors++; matches++ } END { print errors, matches }
%      $N == "ERROR" { errors++ } $N == "WARN" { warnings++ } END { print errors, warnings }
%      { counts[$1]++ } END { print counts["ERROR"], counts["WARN"] }
%      $N == "VALUE" { counts[$M]++ } END { print counts["KEY"] }
%      { total++; counts[$1]++ } END { print total, counts["ERROR"] }
%      { total++ } END { print "total", total }
%      BEGIN { print "kind", "count" } { total++ } END { print "total", total }
%      BEGIN { FS = ":" } $1 == "ERROR" { counts[$2]++ } END { print counts["disk"] }
%      BEGIN { FS = ":"; OFS = "," } $1 == "ERROR" { print $2, $3 }
%      $1 == "ERROR" { printf "%s=%s\n", $2, $3 }
%      $3 > 100 { big++ } END { print big }
%      $1 == "ERROR" { print $3, int($3) }
%      $1 == "ERROR" { print int($3) + 1 }
%      $1 == "ERROR" { print int($3) - 1 }
%      $1 == "ERROR" { print NR - 1, NF + 1, length($0) - 3, index($2, "sk") + 1 }
%      $1 == "ERROR" { bytes += $3; last = $3 } END { print bytes, last }
%      $1 == "ERROR" { bytes += length($0); hits += 2 } END { print bytes, hits }
%      { last_pos = index($2, "sk") + 1; total_pos += index($0, "disk") - 1 } END { print last_pos, total_pos }
%      { adjusted += length($0) - 3; width = NF; fields += NF } END { print adjusted, width, fields }
%      { last = NR; prev = NR - 1; total += NR + 1 } END { print last, prev, total }
%      $1 == "ERROR" { hits++; break } { total++ } END { print hits, total }
%      $1 == "ERROR" { last_len = length($0); hits++ } END { print hits, last_len }
%      { if ($1 == "ERROR") { errors++ } else { warnings++ } } END { print errors, warnings }
%      { if ($1 == "ERROR") { print $2, $3 } else { counts[$1]++ } } END { print counts["WARN"] }
%      { if ($1 == "ERROR") { print "error", $2 } else { print "ok", $1 } } END { print "done" }
%
%  The surrounding runtime still comes from write_wam_llvm_project/3. This
%  function emits the target-specific native main that streams the file, lowers
%  the deterministic guard, and prints matching records.
plawk_program_native_driver_ir(
    program(BeginClauses, [rule(Pattern, [Action0])], []),
    InputPath,
    DriverIR
) :-
    plawk_resolve_writebin_rules(BeginClauses, [rule(Pattern, [Action0])],
        [rule(Pattern, [Action])], WritebinPlan),
    ( Action = writebin_out(WritebinTypes, WritebinFields)
    -> plawk_writebin_args_ok(WritebinTypes, WritebinFields)
    ;  Action = writebin_arm_out(_Tag, ArmTypes, ArmFields)
    -> plawk_writebin_args_ok(ArmTypes, ArmFields)
    ;  plawk_rule_body_print_action(Action)
    ),
    plawk_output_action_exprs(Action, Exprs),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR0),
    plawk_writebin_entry_lines(WritebinPlan, WritebinEntryIR),
    plawk_join_nonempty_ir([BeginIR0, WritebinEntryIR], BeginIR),
    plawk_record_descriptor(BeginClauses, FieldSeparator),
    plawk_record_program_ok(FieldSeparator, [rule(Pattern, [Action])], []),
    plawk_pattern_guard_ir(Pattern, FieldSeparator, GuardGlobalIR-GuardCallIR),
    plawk_print_record_counter_ir(Exprs, LoopPhiIR, RecordCounterIR),
    plawk_output_action_ir(Action, FieldSeparator, OutputSeparator, PrintGlobalIR-PrintActionIR),
    plawk_writebin_globals(WritebinPlan, WritebinGlobalIR),
    format(atom(RecordIR),
'~w
~w
  br i1 %is_match, label %print_line, label %continue_loop

print_line:
~w
  br label %continue_loop',
        [RecordCounterIR, GuardCallIR, PrintActionIR]),
    format(atom(RuntimeGlobals),
'@.plawk_surface_print_line = private constant [4 x i8] c"%s\\0A\\00"
@.plawk_surface_print_slice = private constant [5 x i8] c"%.*s\\00"
@.plawk_surface_print_i64 = private constant [4 x i8] c"%ld\\00"
@.plawk_surface_print_newline = private constant [2 x i8] c"\\0A\\00"
@.plawk_surface_print_string = private constant [3 x i8] c"%s\\00"
@.plawk_surface_print_f64 = private constant [3 x i8] c"%g\\00"
~w
~w
~w
~w
',
        [BeginGlobalIR, GuardGlobalIR, PrintGlobalIR, WritebinGlobalIR]),
    plawk_emit_record_driver_ir(FieldSeparator, InputPath,
        driver_blocks(RuntimeGlobals, BeginIR, LoopPhiIR, lowered_match, RecordIR, '',
            success, 'success:\n  ret i32 0'),
        DriverIR).

% sub/gsub stream editor: a single text-mode rule `Pattern { sub/gsub(re,repl);
% ...; print $0 }`. Each substitution rewrites the whole record ($0) via
% @wam_regex_gsub (a per-site compiled ERE; `&` in the replacement expands to the
% matched text), folding a running interned-atom id; the trailing `print $0`
% emits the result. v1 scope: the target is $0, a single rule with no END, and
% the print-the-record idiom. sub/gsub into a scalar/field, and capturing the
% substitution count (`n = gsub(...)`), are follow-ons.
plawk_program_native_driver_ir(
    program(BeginClauses, [rule(Pattern, Actions)], []),
    InputPath,
    DriverIR
) :-
    plawk_gsub_body(Actions, SubActions, _PrintAction),
    \+ plawk_begin_has_binfmt(BeginClauses),
    plawk_record_descriptor(BeginClauses, FieldSeparator),
    integer(FieldSeparator),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_pattern_guard_ir(Pattern, FieldSeparator, GuardGlobalIR-GuardCallIR),
    plawk_gsub_actions_ir(SubActions, GsubGlobalIR, GsubBodyIR, FinalIdVar),
    format(atom(PrintIR),
'  %gsx_out = call i8* @wam_atom_to_string(i64 ~w)
  %gsx_out_fmt = getelementptr [4 x i8], [4 x i8]* @.plawk_surface_print_line, i32 0, i32 0
  %gsx_out_pr = call i32 (i8*, ...) @printf(i8* %gsx_out_fmt, i8* %gsx_out)',
        [FinalIdVar]),
    format(atom(RecordIR),
'~w
  br i1 %is_match, label %print_line, label %continue_loop

print_line:
~w
~w
  br label %continue_loop',
        [GuardCallIR, GsubBodyIR, PrintIR]),
    format(atom(RuntimeGlobals),
'@.plawk_surface_print_line = private constant [4 x i8] c"%s\\0A\\00"
@.plawk_surface_print_slice = private constant [5 x i8] c"%.*s\\00"
@.plawk_surface_print_i64 = private constant [4 x i8] c"%ld\\00"
@.plawk_surface_print_newline = private constant [2 x i8] c"\\0A\\00"
@.plawk_surface_print_string = private constant [3 x i8] c"%s\\00"
@.plawk_surface_print_f64 = private constant [3 x i8] c"%g\\00"
~w
~w
~w
',
        [BeginGlobalIR, GuardGlobalIR, GsubGlobalIR]),
    plawk_emit_record_driver_ir(FieldSeparator, InputPath,
        driver_blocks(RuntimeGlobals, BeginIR, '', lowered_match, RecordIR, '',
            success, 'success:\n  ret i32 0'),
        DriverIR).

%% plawk_gsub_body(+Actions, -SubActions, -PrintAction)
%
%  A sub/gsub stream-editor body: one or more `regex_sub(Global, Re, Repl)` actions
%  followed by exactly `print $0`.
plawk_gsub_body(Actions, SubActions, print([field(0)])) :-
    append(SubActions, [print([field(0)])], Actions),
    SubActions = [_ | _],
    forall(member(A, SubActions), A = regex_sub(_, _, _)).

%% plawk_gsub_actions_ir(+SubActions, -GlobalIR, -BodyIR, -FinalIdVar)
%
%  Fold the substitutions over $0: the first reads the original line atom, each
%  subsequent one reads the previous result, so chained sub/gsub compose.
plawk_gsub_actions_ir(SubActions, GlobalIR, BodyIR, FinalIdVar) :-
    plawk_gsub_fold(SubActions, 0, '%line_payload', Parts, FinalIdVar),
    pairs_keys_values(Parts, GlobalParts, BodyParts),
    plawk_join_nonempty_ir(GlobalParts, GlobalIR),
    atomic_list_concat(BodyParts, '\n', BodyIR).

plawk_gsub_fold([], _Index, CurId, [], CurId).
plawk_gsub_fold([regex_sub(Global, Regex, Repl) | Rest], Index, CurId,
        [GlobalPart-BodyPart | Parts], FinalId) :-
    plawk_gsub_one(Global, Regex, Repl, Index, CurId, GlobalPart, BodyPart, NextId),
    NextIndex is Index + 1,
    plawk_gsub_fold(Rest, NextIndex, NextId, Parts, FinalId).

%% plawk_gsub_one(+Global, +Regex, +Repl, +Index, +CurId, -GlobalPart, -Body, -NextId)
plawk_gsub_one(Global, Regex, Repl, Index, CurId, GlobalPart, BodyPart, NextId) :-
    format(atom(PatName), 'gsx_~w_pat', [Index]),
    format(atom(ReplName), 'gsx_~w_repl', [Index]),
    format(atom(CacheName), 'gsx_~w_cache', [Index]),
    llvm_emit_c_string_global(PatName, Regex, PatGlobal, _PL, PatBytes),
    llvm_emit_c_string_global(ReplName, Repl, ReplGlobal, ReplLen, ReplBytes),
    format(atom(CacheGlobal), '@~w = internal global i8* null', [CacheName]),
    atomic_list_concat([PatGlobal, ReplGlobal, CacheGlobal], '\n', GlobalPart),
    ( Global =:= 1 -> GlobalBool = true ; GlobalBool = false ),
    format(atom(NextId), '%gsx_~w_id', [Index]),
    format(atom(BodyPart),
'  %gsx_~w_str = call i8* @wam_atom_to_string(i64 ~w)
  %gsx_~w_len = call i64 @strlen(i8* %gsx_~w_str)
  %gsx_~w_patptr = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0
  %gsx_~w_replptr = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0
  ~w = call i64 @wam_regex_gsub(i8* %gsx_~w_str, i64 %gsx_~w_len, i8* %gsx_~w_patptr, i8** @~w, i8* %gsx_~w_replptr, i64 ~w, i1 ~w, i64* @plawk_gsub_count)',
        [Index, CurId,
         Index, Index,
         Index, PatBytes, PatBytes, PatName,
         Index, ReplBytes, ReplBytes, ReplName,
         NextId, Index, Index, Index, CacheName, Index, ReplLen, GlobalBool]).

% Field assignment: a single text-mode rule `Pattern { $N = expr; ...; print $0 }`.
% The record is split once into an editable field buffer (@wam_fields_new); each
% `$N = expr` mutates a slot in place (@wam_fields_set); `print $0` joins the
% fields with OFS once (@wam_fields_join) and emits them. No interning, and the
% record is scanned once regardless of how many fields are assigned -- O(record),
% not O(assignments x record). v1 scope: single-char explicit FS (so field
% splitting matches how fields are read), a single rule with no END, and the
% print-the-record idiom. RHS is a string literal, an integer literal, or another
% field `$M` (read from the current buffer). In-body field reads after assignment,
% default (space) FS, multi-rule/END programs, and `$0 = expr` are follow-ons.
plawk_program_native_driver_ir(
    program(BeginClauses, [rule(Pattern, Actions)], []),
    InputPath,
    DriverIR
) :-
    plawk_field_assign_body(Actions, SetFields, _PrintAction),
    \+ plawk_begin_has_binfmt(BeginClauses),
    plawk_record_descriptor(BeginClauses, FieldSeparator),
    integer(FieldSeparator),
    % a single explicit char (byte > 0, not whitespace 32) or the regex-FS
    % sentinel 0 (the field buffer splits on the program FS regex via _new_re)
    FieldSeparator =\= 32,
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_pattern_guard_ir(Pattern, FieldSeparator, GuardGlobalIR-GuardCallIR),
    plawk_field_assign_sets_ir(SetFields, AssignGlobalIR, SetBodyIR),
    format(atom(RecordIR),
'~w
  br i1 %is_match, label %print_line, label %continue_loop

print_line:
  %fa_fb = call %WamFieldBuf* @wam_fields_new(i8* %line_s, i8 ~w)
~w
  %fa_joined = call i8* @wam_fields_join(%WamFieldBuf* %fa_fb, i8 ~w)
  %fa_out_fmt = getelementptr [4 x i8], [4 x i8]* @.plawk_surface_print_line, i32 0, i32 0
  %fa_out_pr = call i32 (i8*, ...) @printf(i8* %fa_out_fmt, i8* %fa_joined)
  call void @free(i8* %fa_joined)
  call void @wam_fields_free(%WamFieldBuf* %fa_fb)
  br label %continue_loop',
        [GuardCallIR, FieldSeparator, SetBodyIR, OutputSeparator]),
    format(atom(RuntimeGlobals),
'@.plawk_surface_print_line = private constant [4 x i8] c"%s\\0A\\00"
@.plawk_surface_print_slice = private constant [5 x i8] c"%.*s\\00"
@.plawk_surface_print_i64 = private constant [4 x i8] c"%ld\\00"
@.plawk_surface_print_newline = private constant [2 x i8] c"\\0A\\00"
@.plawk_surface_print_string = private constant [3 x i8] c"%s\\00"
@.plawk_surface_print_f64 = private constant [3 x i8] c"%g\\00"
~w
~w
~w
',
        [BeginGlobalIR, GuardGlobalIR, AssignGlobalIR]),
    plawk_emit_record_driver_ir(FieldSeparator, InputPath,
        driver_blocks(RuntimeGlobals, BeginIR, '', lowered_match, RecordIR, '',
            success, 'success:\n  ret i32 0'),
        DriverIR).

%% plawk_field_assign_body(+Actions, -SetFields, -PrintAction)
%
%  A field-assignment rule body: one or more `set_field(N, Value)` actions
%  followed by exactly `print $0` (`print([field(0)])`). This is the canonical
%  "edit a field and re-emit the record" shape; anything else (a scalar update,
%  a field read after the assignment, a non-`$0` print) does not match, so the
%  program falls through to the general drivers.
plawk_field_assign_body(Actions, SetFields, print([field(0)])) :-
    append(SetFields, [print([field(0)])], Actions),
    SetFields = [_ | _],
    forall(member(A, SetFields), plawk_field_edit_action(A)).

% A field-editing action: a field assignment or a sub/gsub into a field.
plawk_field_edit_action(set_field(_, _)).
plawk_field_edit_action(regex_sub_field(_, _, _, _)).

%% plawk_field_assign_sets_ir(+SetFields, -GlobalIR, -SetBodyIR)
%
%  IR for the in-place field mutations against the `%fa_fb` field buffer. Each
%  `$N = Value` materializes the value as a byte range and calls @wam_fields_set;
%  a field-copy RHS (`$1=$3`) reads the current (possibly already-mutated) buffer,
%  so chained assignments observe the running record.
plawk_field_assign_sets_ir(SetFields, GlobalIR, SetBodyIR) :-
    plawk_field_assign_fold(SetFields, 0, Parts),
    pairs_keys_values(Parts, GlobalParts, BodyParts),
    plawk_join_nonempty_ir(GlobalParts, GlobalIR),
    atomic_list_concat(BodyParts, '\n', SetBodyIR).

plawk_field_assign_fold([], _Index, []).
plawk_field_assign_fold([set_field(N, Value) | Rest], Index,
        [GlobalPart-BodyPart | Parts]) :-
    plawk_field_assign_one(N, Value, Index, GlobalPart, BodyPart),
    NextIndex is Index + 1,
    plawk_field_assign_fold(Rest, NextIndex, Parts).
plawk_field_assign_fold([regex_sub_field(Global, Regex, Repl, N) | Rest], Index,
        [GlobalPart-BodyPart | Parts]) :-
    plawk_field_gsub_one(Global, Regex, Repl, N, Index, GlobalPart, BodyPart),
    NextIndex is Index + 1,
    plawk_field_assign_fold(Rest, NextIndex, Parts).

%% plawk_field_gsub_one(+Global, +Regex, +Repl, +N, +Index, -GlobalPart, -BodyPart)
%
%  `sub/gsub(/re/, repl, $N)`: read field N's current slice from the buffer, run
%  @wam_regex_gsub over it (per-site pattern/cache; `&` expands to the match),
%  resolve the interned result to text, and store it back into field N. Rebuilds
%  $0 at the join, like a field assignment.
plawk_field_gsub_one(Global, Regex, Repl, N, Index, GlobalPart, BodyPart) :-
    format(atom(PatName), 'fa_gs_~w_pat', [Index]),
    format(atom(ReplName), 'fa_gs_~w_repl', [Index]),
    format(atom(CacheName), 'fa_gs_~w_cache', [Index]),
    llvm_emit_c_string_global(PatName, Regex, PatGlobal, _PL, PatBytes),
    llvm_emit_c_string_global(ReplName, Repl, ReplGlobal, ReplLen, ReplBytes),
    format(atom(CacheGlobal), '@~w = internal global i8* null', [CacheName]),
    atomic_list_concat([PatGlobal, ReplGlobal, CacheGlobal], '\n', GlobalPart),
    ( Global =:= 1 -> GlobalBool = true ; GlobalBool = false ),
    format(atom(BodyPart),
'  %fa_gs_~w_sl = call %WamSlice @wam_fields_get(%WamFieldBuf* %fa_fb, i64 ~w)
  %fa_gs_~w_sp = extractvalue %WamSlice %fa_gs_~w_sl, 0
  %fa_gs_~w_sn = extractvalue %WamSlice %fa_gs_~w_sl, 1
  %fa_gs_~w_pp = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0
  %fa_gs_~w_rp = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0
  %fa_gs_~w_id = call i64 @wam_regex_gsub(i8* %fa_gs_~w_sp, i64 %fa_gs_~w_sn, i8* %fa_gs_~w_pp, i8** @~w, i8* %fa_gs_~w_rp, i64 ~w, i1 ~w, i64* @plawk_gsub_count)
  %fa_gs_~w_ns = call i8* @wam_atom_to_string(i64 %fa_gs_~w_id)
  %fa_gs_~w_nl = call i64 @strlen(i8* %fa_gs_~w_ns)
  call void @wam_fields_set(%WamFieldBuf* %fa_fb, i64 ~w, i8* %fa_gs_~w_ns, i64 %fa_gs_~w_nl)',
        [Index, N,
         Index, Index,
         Index, Index,
         Index, PatBytes, PatBytes, PatName,
         Index, ReplBytes, ReplBytes, ReplName,
         Index, Index, Index, Index, CacheName, Index, ReplLen, GlobalBool,
         Index, Index,
         Index, Index,
         N, Index, Index]).

%% plawk_field_assign_one(+N, +Value, +Index, -GlobalPart, -BodyPart)
%
%  One `$N = Value` mutation: materialize the value as (ptr, len), then store it
%  into slot N of the field buffer.
plawk_field_assign_one(N, Value, Index, GlobalPart, BodyPart) :-
    plawk_field_assign_value_ir(Value, Index, GlobalPart, ValueLines, ValPtr, ValLen),
    format(atom(SetLine),
        '  call void @wam_fields_set(%WamFieldBuf* %fa_fb, i64 ~w, i8* ~w, i64 ~w)',
        [N, ValPtr, ValLen]),
    append(ValueLines, [SetLine], Lines),
    atomic_list_concat(Lines, '\n', BodyPart).

%% plawk_field_assign_value_ir(+Value, +Index, -GlobalPart, -Lines, -ValPtr, -ValLen)
%
%  The new field value as (ValPtr, ValLen). A string / integer literal becomes a
%  private C-string global; a field `$M` is projected from the current field
%  buffer (its slice aliases the record text, which @wam_fields_join copies out).
%  A missing field yields a {null, 0} slice -> an empty field.
plawk_field_assign_value_ir(string(S), Index, GlobalPart, [PtrLine], ValPtr, ValLen) :-
    format(atom(GlobalName), 'fa_val_~w', [Index]),
    llvm_emit_c_string_global(GlobalName, S, GlobalPart, ValLen, BytesLen),
    format(atom(ValPtr), '%fa_~w_vp', [Index]),
    format(atom(PtrLine),
        '  ~w = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0',
        [ValPtr, BytesLen, BytesLen, GlobalName]).
plawk_field_assign_value_ir(int(V), Index, GlobalPart, Lines, ValPtr, ValLen) :-
    format(atom(S), '~w', [V]),
    plawk_field_assign_value_ir(string(S), Index, GlobalPart, Lines, ValPtr, ValLen).
plawk_field_assign_value_ir(field(M), Index, '', Lines, ValPtr, ValLen) :-
    format(atom(Slice), '%fa_~w_sl', [Index]),
    format(atom(ValPtr), '%fa_~w_vp', [Index]),
    format(atom(ValLen), '%fa_~w_vl', [Index]),
    format(atom(L0),
        '  ~w = call %WamSlice @wam_fields_get(%WamFieldBuf* %fa_fb, i64 ~w)',
        [Slice, M]),
    format(atom(L1), '  ~w = extractvalue %WamSlice ~w, 0', [ValPtr, Slice]),
    format(atom(L2), '  ~w = extractvalue %WamSlice ~w, 1', [ValLen, Slice]),
    Lines = [L0, L1, L2].

% A body-printing scalar rule chain with NO END block: all output comes from
% the per-record body (e.g. `{ i = 0; while (i < 3) { print i; i++ } }`). Same
% lowering as the scalar END-print chain but with an empty print list and NO
% end_print body -- the end_print block just returns 0 (no trailing newline).
% Requires a genuine scalar plan (a rule that updates state or prints in its
% body); writebin-only / query / single-action programs match their own
% clauses above.
plawk_program_native_driver_ir(
    program(BeginClauses, Rules0, []),
    InputPath,
    DriverIR
) :-
    \+ plawk_begin_has_binfmt(BeginClauses),
    is_list(Rules0),
    plawk_resolve_writebin_rules(BeginClauses, Rules0, Rules1, WritebinPlan),
    plawk_record_descriptor(BeginClauses, FieldSeparator),
    plawk_resolve_dynrec_view_rules(Rules1, Rules1b),
    plawk_resolve_foreach_rules(FieldSeparator, Rules1b, Rules),
    plawk_scalar_state_plan(Rules, [], StatePlan),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR0),
    plawk_writebin_entry_lines(WritebinPlan, WritebinEntryIR),
    plawk_join_nonempty_ir([BeginIR0, WritebinEntryIR], BeginIR),
    plawk_record_program_ok(FieldSeparator, Rules, []),
    plawk_scalar_rule_chain_ir(Rules, StatePlan, FieldSeparator, OutputSeparator,
        RuleGlobalIR, RuleChainIR, RuleCount, BranchControlExits),
    plawk_rules_body_print_fields(Rules, BodyPrintFields),
    plawk_rules_scalar_update_exprs(Rules, ScalarExprs),
    plawk_rules_writebin_exprs(Rules, WritebinExprs),
    append(BodyPrintFields, WritebinExprs, PrintExprs0),
    append(PrintExprs0, ScalarExprs, RecordCounterExprs),
    plawk_print_record_counter_ir(RecordCounterExprs, RecordLoopPhiIR, RecordCounterIR),
    plawk_state_loop_phi_ir(StatePlan, StateLoopPhiIR),
    plawk_join_nonempty_ir([StateLoopPhiIR, RecordLoopPhiIR], LoopPhiIR),
    plawk_join_nonempty_ir([RecordCounterIR, RuleChainIR], RecordIR),
    plawk_scalar_rule_controls(Rules, ScalarRuleControls),
    plawk_scalar_next_phi_ir(StatePlan, RuleCount, ScalarRuleControls, BranchControlExits, NextPhiIR),
    plawk_break_close_ir(StatePlan, RuleCount, ScalarRuleControls, BranchControlExits, done,
        BreakCloseIR, FinalStatePhiIR),
    plawk_writebin_globals(WritebinPlan, WritebinGlobalIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w',
        [BeginGlobalIR, RuleGlobalIR, WritebinGlobalIR]),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w
  %plawk_exit_ec = load i32, i32* @plawk_exit_code
  ret i32 %plawk_exit_ec',
        [FinalStatePhiIR]),
    plawk_emit_record_driver_ir(FieldSeparator, InputPath,
        driver_blocks(RuntimeGlobals, BeginIR, LoopPhiIR, lowered_match, RecordIR,
            NextPhiIR, BreakCloseIR, end_print, CloseOkIR),
        DriverIR).

% A NO-END program whose rule bodies mutate an assoc/positional table AND print
% per-record (`{ c[$1]++; print $1, c[$1] }`, `{ split($0,a,","); print a[1] }`).
% The assoc rule chain already emits per-record prints (the `assoc_print_action`
% block, with %line live); this clause routes a no-END assoc program through it,
% freeing the tables at exit. Tried after the scalar chain (which fails to lower
% an assoc action and backtracks here); a non-assoc program has no table, so
% plawk_assoc_runtime_record_plan fails and this clause is skipped.
plawk_program_native_driver_ir(
    program(BeginClauses, Rules, []),
    InputPath,
    DriverIR
) :-
    plawk_assoc_runtime_record_plan(Rules, AssocPlan),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_record_descriptor(BeginClauses, FieldSeparator),
    plawk_assoc_record_program_ok(FieldSeparator, Rules, []),
    plawk_assoc_entry_setup_ir(AssocPlan, EntrySetupIR),
    plawk_assoc_rule_chain_ir(AssocPlan, FieldSeparator, AssocRuleGlobalIR, AssocChainIR),
    plawk_rules_body_print_fields(Rules, BodyPrintFields),
    plawk_rules_scalar_update_exprs(Rules, ScalarExprs),
    append(BodyPrintFields, ScalarExprs, RecordCounterExprs),
    plawk_print_record_counter_ir(RecordCounterExprs, RecordLoopPhiIR, RecordCounterIR),
    plawk_join_nonempty_ir([RecordCounterIR, AssocChainIR], RecordIR),
    plawk_assoc_rule_controls(AssocPlan, AssocRuleControls),
    plawk_assoc_break_close_ir(AssocRuleControls, BreakCloseIR),
    phrase(plawk_assoc_free_lines(AssocPlan), FreeLines),
    atomic_list_concat(FreeLines, '\n', FreeIR),
    format(atom(SurfaceGlobalIR), '~w~n~w', [BeginGlobalIR, AssocRuleGlobalIR]),
    plawk_combine_entry_ir(BeginIR, EntrySetupIR, CombinedEntrySetupIR),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w
  %plawk_exit_ec = load i32, i32* @plawk_exit_code
  ret i32 %plawk_exit_ec',
        [FreeIR]),
    plawk_emit_record_driver_ir(FieldSeparator, InputPath,
        driver_blocks(RuntimeGlobals, CombinedEntrySetupIR, RecordLoopPhiIR, lowered_assoc,
            RecordIR, '', BreakCloseIR, end_print, CloseOkIR),
        DriverIR).

plawk_program_native_driver_ir(
    program(BeginClauses, Rules, [end([print(PrintFields)])]),
    InputPath,
    DriverIR
) :-
    \+ plawk_begin_has_binfmt(BeginClauses),
    plawk_mixed_state_plan(Rules, PrintFields, MixedPlan),
    MixedPlan = mixed_plan(ScalarPlan, AssocPlan, _PlannedRules),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_field_separator(BeginClauses, FieldSeparator),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_assoc_print_key_globals(PrintFields, AssocGlobalIR),
    plawk_assoc_entry_setup_ir(AssocPlan, EntrySetupIR),
    plawk_mixed_rule_chain_ir(MixedPlan, FieldSeparator, OutputSeparator,
        RuleGlobalIR, RuleChainIR, RuleCount, BranchControlExits),
    plawk_rules_body_print_fields(Rules, BodyPrintFields),
    plawk_rules_scalar_update_exprs(Rules, ScalarExprs),
    append(PrintFields, BodyPrintFields, PrintExprs),
    append(PrintExprs, ScalarExprs, RecordCounterExprs),
    plawk_print_record_counter_ir(RecordCounterExprs, RecordLoopPhiIR, RecordCounterIR),
    plawk_state_loop_phi_ir(ScalarPlan, StateLoopPhiIR),
    plawk_join_nonempty_ir([StateLoopPhiIR, RecordLoopPhiIR], LoopPhiIR),
    plawk_join_nonempty_ir([RecordCounterIR, RuleChainIR], RecordIR),
    plawk_mixed_rule_controls(MixedPlan, MixedRuleControls),
    plawk_mixed_scalar_next_phi_ir(ScalarPlan, RuleCount, MixedRuleControls, BranchControlExits, NextPhiIR),
    plawk_break_close_ir(ScalarPlan, RuleCount, MixedRuleControls, BranchControlExits, done,
        BreakCloseIR, FinalStatePhiIR),
    plawk_mixed_end_print_ir(PrintFields, ScalarPlan, AssocPlan, OutputSeparator, EndPrintIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w~n~w',
        [BeginGlobalIR, StringGlobalIR, AssocGlobalIR, RuleGlobalIR]),
    plawk_combine_entry_ir(BeginIR, EntrySetupIR, CombinedEntrySetupIR),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w~w
  %plawk_exit_ec = load i32, i32* @plawk_exit_code
  ret i32 %plawk_exit_ec',
        [FinalStatePhiIR, EndPrintIR]),
    llvm_emit_stream_driver_ir(InputPath,
        driver_blocks(RuntimeGlobals, CombinedEntrySetupIR, LoopPhiIR, lowered_mixed,
            RecordIR, NextPhiIR, BreakCloseIR, end_print, CloseOkIR),
        DriverIR).

plawk_program_native_driver_ir(
    program(BeginClauses, Rules0, [end([print(PrintFields)])]),
    InputPath,
    DriverIR
) :-
    plawk_resolve_writebin_rules(BeginClauses, Rules0, Rules1, WritebinPlan),
    plawk_record_descriptor(BeginClauses, FieldSeparator),
    plawk_resolve_dynrec_view_rules(Rules1, Rules1b),
    plawk_resolve_foreach_rules(FieldSeparator, Rules1b, Rules),
    plawk_scalar_state_plan(Rules, PrintFields, StatePlan),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR0),
    plawk_writebin_entry_lines(WritebinPlan, WritebinEntryIR),
    plawk_join_nonempty_ir([BeginIR0, WritebinEntryIR], BeginIR),
    plawk_record_program_ok(FieldSeparator, Rules, PrintFields),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_scalar_rule_chain_ir(Rules, StatePlan, FieldSeparator, OutputSeparator,
        RuleGlobalIR, RuleChainIR, RuleCount, BranchControlExits),
    plawk_rules_body_print_fields(Rules, BodyPrintFields),
    plawk_rules_scalar_update_exprs(Rules, ScalarExprs),
    plawk_rules_writebin_exprs(Rules, WritebinExprs),
    append(PrintFields, BodyPrintFields, PrintExprs0),
    append(PrintExprs0, WritebinExprs, PrintExprs),
    append(PrintExprs, ScalarExprs, RecordCounterExprs),
    plawk_print_record_counter_ir(RecordCounterExprs, RecordLoopPhiIR, RecordCounterIR),
    plawk_state_loop_phi_ir(StatePlan, StateLoopPhiIR),
    plawk_join_nonempty_ir([StateLoopPhiIR, RecordLoopPhiIR], LoopPhiIR),
    plawk_join_nonempty_ir([RecordCounterIR, RuleChainIR], RecordIR),
    plawk_scalar_rule_controls(Rules, ScalarRuleControls),
    plawk_scalar_next_phi_ir(StatePlan, RuleCount, ScalarRuleControls, BranchControlExits, NextPhiIR),
    plawk_break_close_ir(StatePlan, RuleCount, ScalarRuleControls, BranchControlExits, done,
        BreakCloseIR, FinalStatePhiIR),
    plawk_scalar_end_print_ir(PrintFields, StatePlan, OutputSeparator, EndPrintIR),
    plawk_writebin_globals(WritebinPlan, WritebinGlobalIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w~n~w',
        [BeginGlobalIR, StringGlobalIR, RuleGlobalIR, WritebinGlobalIR]),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w~w
  %plawk_exit_ec = load i32, i32* @plawk_exit_code
  ret i32 %plawk_exit_ec',
        [FinalStatePhiIR, EndPrintIR]),
    plawk_emit_record_driver_ir(FieldSeparator, InputPath,
        driver_blocks(RuntimeGlobals, BeginIR, LoopPhiIR, lowered_match, RecordIR,
            NextPhiIR, BreakCloseIR, end_print, CloseOkIR),
        DriverIR).

% END { if (COND) print ...; [else print ...] } -- a scalar-guarded print in the
% END block. COND is a scalar comparison over the final slot values; each branch
% is a single print. Same lowering as the scalar END-print clause above, but the
% end_print body is the if (condition + then/else print blocks) instead of a
% plain print. The state plan sees every field the branches print plus the
% condition variables (so their slots exist).
plawk_program_native_driver_ir(
    program(BeginClauses, Rules0, [end([if(scalar_if(Cond), ThenActions, ElseActions)])]),
    InputPath,
    DriverIR
) :-
    plawk_end_if_ok(ThenActions, ElseActions),
    plawk_end_if_print_fields(Cond, ThenActions, ElseActions, PrintFields),
    plawk_resolve_writebin_rules(BeginClauses, Rules0, Rules1, WritebinPlan),
    plawk_record_descriptor(BeginClauses, FieldSeparator),
    plawk_resolve_dynrec_view_rules(Rules1, Rules1b),
    plawk_resolve_foreach_rules(FieldSeparator, Rules1b, Rules),
    plawk_scalar_state_plan(Rules, PrintFields, StatePlan),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR0),
    plawk_writebin_entry_lines(WritebinPlan, WritebinEntryIR),
    plawk_join_nonempty_ir([BeginIR0, WritebinEntryIR], BeginIR),
    plawk_record_program_ok(FieldSeparator, Rules, PrintFields),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_scalar_rule_chain_ir(Rules, StatePlan, FieldSeparator, OutputSeparator,
        RuleGlobalIR, RuleChainIR, RuleCount, BranchControlExits),
    plawk_rules_body_print_fields(Rules, BodyPrintFields),
    plawk_rules_scalar_update_exprs(Rules, ScalarExprs),
    plawk_rules_writebin_exprs(Rules, WritebinExprs),
    append(PrintFields, BodyPrintFields, PrintExprs0),
    append(PrintExprs0, WritebinExprs, PrintExprs),
    append(PrintExprs, ScalarExprs, RecordCounterExprs),
    plawk_print_record_counter_ir(RecordCounterExprs, RecordLoopPhiIR, RecordCounterIR),
    plawk_state_loop_phi_ir(StatePlan, StateLoopPhiIR),
    plawk_join_nonempty_ir([StateLoopPhiIR, RecordLoopPhiIR], LoopPhiIR),
    plawk_join_nonempty_ir([RecordCounterIR, RuleChainIR], RecordIR),
    plawk_scalar_rule_controls(Rules, ScalarRuleControls),
    plawk_scalar_next_phi_ir(StatePlan, RuleCount, ScalarRuleControls, BranchControlExits, NextPhiIR),
    plawk_break_close_ir(StatePlan, RuleCount, ScalarRuleControls, BranchControlExits, done,
        BreakCloseIR, FinalStatePhiIR),
    plawk_scalar_end_if_ir(Cond, ThenActions, ElseActions, StatePlan,
        FieldSeparator, OutputSeparator, EndIfGlobalIR, EndIfIR),
    plawk_writebin_globals(WritebinPlan, WritebinGlobalIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w~n~w~n~w',
        [BeginGlobalIR, StringGlobalIR, EndIfGlobalIR, RuleGlobalIR, WritebinGlobalIR]),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w~w
  %plawk_exit_ec = load i32, i32* @plawk_exit_code
  ret i32 %plawk_exit_ec',
        [FinalStatePhiIR, EndIfIR]),
    plawk_emit_record_driver_ir(FieldSeparator, InputPath,
        driver_blocks(RuntimeGlobals, BeginIR, LoopPhiIR, lowered_match, RecordIR,
            NextPhiIR, BreakCloseIR, end_print, CloseOkIR),
        DriverIR).

% Tag-guard sugar: with a union BINFMT, plain rules whose patterns lead
% with TAG == K are shorthand for case blocks -- the tag test selects
% the arm and the rest of the pattern stays as the rule's own guard, so
%   TAG == 0 && $1 > 10 { hits++ }
% means exactly
%   case 0 { $1 > 10 { hits++ } }.
% Every rule must lead with a tag guard (an unguarded rule has no arm
% to type its fields against), and the tag test may appear nowhere
% else; otherwise the program is rejected.
plawk_program_native_driver_ir(
    program(BeginClauses, Rules, EndClauses),
    InputPath,
    DriverIR
) :-
    is_list(Rules),
    plawk_record_descriptor(BeginClauses, binfmt_union(_Arms)),
    plawk_tag_rules_case_blocks(Rules, CaseBlocks),
    !,
    plawk_program_native_driver_ir(
        program(BeginClauses, case_blocks(CaseBlocks), EndClauses),
        InputPath, DriverIR).

% Tagged-union programs: BINFMT = "case(arm0 | arm1 | ...)" plus
% case K { rules } blocks. Case blocks flatten into one scalar rule
% chain where each rule's guard checks the record tag before its own
% pattern, and each rule's fields type against its arm's layout. The
% END print block is optional (a pure per-arm printer needs none).
% writebin works inside case blocks: OUTFMT is program-wide, source
% fields type against each rule's arm.
plawk_program_native_driver_ir(
    program(BeginClauses, case_blocks(CaseBlocks0), EndClauses),
    InputPath,
    DriverIR
) :-
    plawk_record_descriptor(BeginClauses, Descriptor),
    Descriptor = binfmt_union(Arms),
    plawk_resolve_union_writebin_blocks(BeginClauses, CaseBlocks0, CaseBlocks,
        WritebinPlan),
    plawk_union_program_ok(Arms, CaseBlocks),
    plawk_union_flatten_rules(CaseBlocks, Arms, Rules),
    (   EndClauses = [end([print(PrintFields)])]
    ->  HasEnd = true
    ;   EndClauses == [],
        PrintFields = [],
        HasEnd = false
    ),
    (   plawk_scalar_state_plan(Rules, PrintFields, StatePlan)
    ->  true
    ;   % a pure normalizer: every rule just writebins its arm into
        % the shared output layout -- no scalar state at all
        WritebinPlan \== none,
        PrintFields == [],
        StatePlan = state_plan([])
    ),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR0),
    plawk_writebin_entry_lines(WritebinPlan, WritebinEntryIR),
    plawk_join_nonempty_ir([BeginIR0, WritebinEntryIR], BeginIR),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_scalar_rule_chain_ir(Rules, StatePlan, Descriptor, OutputSeparator,
        RuleGlobalIR, RuleChainIR, RuleCount, BranchControlExits),
    plawk_rules_body_print_fields(Rules, BodyPrintFields),
    plawk_rules_scalar_update_exprs(Rules, ScalarExprs),
    plawk_rules_writebin_exprs(Rules, WritebinExprs),
    append(PrintFields, BodyPrintFields, PrintExprs0),
    append(PrintExprs0, WritebinExprs, PrintExprs),
    append(PrintExprs, ScalarExprs, RecordCounterExprs),
    plawk_print_record_counter_ir(RecordCounterExprs, RecordLoopPhiIR, RecordCounterIR),
    plawk_state_loop_phi_ir(StatePlan, StateLoopPhiIR),
    plawk_join_nonempty_ir([StateLoopPhiIR, RecordLoopPhiIR], LoopPhiIR),
    plawk_join_nonempty_ir([RecordCounterIR, RuleChainIR], RecordIR),
    plawk_scalar_rule_controls(Rules, ScalarRuleControls),
    plawk_scalar_next_phi_ir(StatePlan, RuleCount, ScalarRuleControls, BranchControlExits, NextPhiIR),
    plawk_break_close_ir(StatePlan, RuleCount, ScalarRuleControls, BranchControlExits, done,
        BreakCloseIR, FinalStatePhiIR),
    (   HasEnd == true
    ->  plawk_scalar_end_print_ir(PrintFields, StatePlan, OutputSeparator, EndPrintIR)
    ;   EndPrintIR = ''
    ),
    plawk_writebin_globals(WritebinPlan, WritebinGlobalIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w~n~w',
        [BeginGlobalIR, StringGlobalIR, RuleGlobalIR, WritebinGlobalIR]),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w~w
  %plawk_exit_ec = load i32, i32* @plawk_exit_code
  ret i32 %plawk_exit_ec',
        [FinalStatePhiIR, EndPrintIR]),
    plawk_emit_record_driver_ir(Descriptor, InputPath,
        driver_blocks(RuntimeGlobals, BeginIR, LoopPhiIR, lowered_match, RecordIR,
            NextPhiIR, BreakCloseIR, end_print, CloseOkIR),
        DriverIR).

% Tagged-union group-by: case-block (or TAG-guarded) rules whose
% actions are assoc increments, with the shared END report. Keys are
% the raw i64 field values of each rule's own arm; every arm updates
% the same table per array name (as in awk, counts[k] is one array no
% matter which rule touched it).
plawk_program_native_driver_ir(
    program(BeginClauses, case_blocks(CaseBlocks), [end([print(PrintFields)])]),
    InputPath,
    DriverIR
) :-
    plawk_record_descriptor(BeginClauses, Descriptor),
    Descriptor = binfmt_union(Arms),
    plawk_union_flatten_rules(CaseBlocks, Arms, Rules),
    plawk_assoc_runtime_count_plan(Rules, PrintFields, AssocPlan),
    plawk_assoc_record_program_ok(Descriptor, Rules, PrintFields),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_assoc_print_key_globals(PrintFields, AssocGlobalIR),
    plawk_assoc_entry_setup_ir(AssocPlan, EntrySetupIR),
    plawk_assoc_rule_chain_ir(AssocPlan, Descriptor, AssocRuleGlobalIR, AssocChainIR),
    plawk_rules_body_print_fields(Rules, BodyPrintFields),
    plawk_rules_scalar_update_exprs(Rules, ScalarExprs),
    append(PrintFields, BodyPrintFields, PrintExprs),
    append(PrintExprs, ScalarExprs, RecordCounterExprs),
    plawk_print_record_counter_ir(RecordCounterExprs, RecordLoopPhiIR, RecordCounterIR),
    plawk_join_nonempty_ir([RecordCounterIR, AssocChainIR], RecordIR),
    plawk_assoc_rule_controls(AssocPlan, AssocRuleControls),
    plawk_assoc_break_close_ir(AssocRuleControls, BreakCloseIR),
    plawk_assoc_end_print_ir(PrintFields, AssocPlan, Descriptor, OutputSeparator, EndPrintIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w~n~w',
        [BeginGlobalIR, StringGlobalIR, AssocGlobalIR, AssocRuleGlobalIR]),
    plawk_combine_entry_ir(BeginIR, EntrySetupIR, CombinedEntrySetupIR),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w
  %plawk_exit_ec = load i32, i32* @plawk_exit_code
  ret i32 %plawk_exit_ec',
        [EndPrintIR]),
    plawk_emit_record_driver_ir(Descriptor, InputPath,
        driver_blocks(RuntimeGlobals, CombinedEntrySetupIR, RecordLoopPhiIR, lowered_assoc,
            RecordIR, '', BreakCloseIR, end_print, CloseOkIR),
        DriverIR).

% ... and the canonical group-by report: END { for (k in counts)
% print k, counts[k] } over a tagged-union stream.
plawk_program_native_driver_ir(
    program(BeginClauses, case_blocks(CaseBlocks), [end([for_in(var(LoopVar), var(ArrayName), BodyActions)])]),
    InputPath,
    DriverIR
) :-
    plawk_record_descriptor(BeginClauses, Descriptor),
    Descriptor = binfmt_union(Arms),
    plawk_union_flatten_rules(CaseBlocks, Arms, Rules),
    plawk_forin_end_plan(Rules, LoopVar, ArrayName, BodyActions, AssocPlan, PrintFields),
    plawk_assoc_record_program_ok(Descriptor, Rules, PrintFields),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_assoc_entry_setup_ir(AssocPlan, EntrySetupIR),
    plawk_assoc_rule_chain_ir(AssocPlan, Descriptor, AssocRuleGlobalIR, AssocChainIR),
    plawk_assoc_rule_controls(AssocPlan, AssocRuleControls),
    plawk_assoc_break_close_ir(AssocRuleControls, BreakCloseIR),
    plawk_forin_end_print_ir(LoopVar, ArrayName, PrintFields, AssocPlan,
        Descriptor, OutputSeparator, EndPrintIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w',
        [BeginGlobalIR, StringGlobalIR, AssocRuleGlobalIR]),
    plawk_combine_entry_ir(BeginIR, EntrySetupIR, CombinedEntrySetupIR),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w',
        [EndPrintIR]),
    plawk_emit_record_driver_ir(Descriptor, InputPath,
        driver_blocks(RuntimeGlobals, CombinedEntrySetupIR, '', lowered_assoc,
            AssocChainIR, '', BreakCloseIR, end_print, CloseOkIR),
        DriverIR).

plawk_program_native_driver_ir(
    program(BeginClauses, Rules, [end([print(PrintFields)])]),
    InputPath,
    DriverIR
) :-
    plawk_assoc_runtime_count_plan(Rules, PrintFields, AssocPlan),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_record_descriptor(BeginClauses, FieldSeparator),
    plawk_assoc_record_program_ok(FieldSeparator, Rules, PrintFields),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_assoc_print_key_globals(PrintFields, AssocGlobalIR),
    plawk_assoc_entry_setup_ir(AssocPlan, EntrySetupIR),
    plawk_assoc_rule_chain_ir(AssocPlan, FieldSeparator, AssocRuleGlobalIR, AssocChainIR),
    plawk_rules_body_print_fields(Rules, BodyPrintFields),
    plawk_rules_scalar_update_exprs(Rules, ScalarExprs),
    append(PrintFields, BodyPrintFields, PrintExprs),
    append(PrintExprs, ScalarExprs, RecordCounterExprs),
    plawk_print_record_counter_ir(RecordCounterExprs, RecordLoopPhiIR, RecordCounterIR),
    plawk_join_nonempty_ir([RecordCounterIR, AssocChainIR], RecordIR),
    plawk_assoc_rule_controls(AssocPlan, AssocRuleControls),
    plawk_assoc_break_close_ir(AssocRuleControls, BreakCloseIR),
    plawk_assoc_end_print_ir(PrintFields, AssocPlan, FieldSeparator, OutputSeparator, EndPrintIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w~n~w',
        [BeginGlobalIR, StringGlobalIR, AssocGlobalIR, AssocRuleGlobalIR]),
    plawk_combine_entry_ir(BeginIR, EntrySetupIR, CombinedEntrySetupIR),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w
  %plawk_exit_ec = load i32, i32* @plawk_exit_code
  ret i32 %plawk_exit_ec',
        [EndPrintIR]),
    plawk_emit_record_driver_ir(FieldSeparator, InputPath,
        driver_blocks(RuntimeGlobals, CombinedEntrySetupIR, RecordLoopPhiIR, lowered_assoc,
            RecordIR, '', BreakCloseIR, end_print, CloseOkIR),
        DriverIR).

% Tagged-union group-by to binary output: case-block rules count per
% arm, END iterates the table and writebins one fixed-layout record
% per group.
plawk_program_native_driver_ir(
    program(BeginClauses, case_blocks(CaseBlocks), [end([for_in(var(LoopVar), var(ArrayName), [writebin(Fields)])])]),
    InputPath,
    DriverIR
) :-
    plawk_record_descriptor(BeginClauses, Descriptor),
    Descriptor = binfmt_union(Arms),
    plawk_union_flatten_rules(CaseBlocks, Arms, Rules),
    plawk_begin_outfmt_types(BeginClauses, OutTypes),
    forall(member(OutType, OutTypes), memberchk(OutType, [i64, f64])),
    length(OutTypes, ArgCount),
    length(Fields, ArgCount),
    maplist(plawk_forin_writebin_field(LoopVar), Fields),
    findall(LookupArrayName,
        member(assoc(var(LookupArrayName), var(LoopVar)), Fields),
        LookupArrays),
    plawk_forin_assoc_plan(Rules, ArrayName, LookupArrays, AssocPlan),
    plawk_assoc_record_program_ok(Descriptor, Rules, []),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_assoc_entry_setup_ir(AssocPlan, EntrySetupIR),
    plawk_binfmt_record_size(binfmt(OutTypes), OutRecordSize),
    plawk_writebin_entry_lines(outfmt(OutTypes, OutRecordSize), WritebinEntryIR),
    plawk_assoc_rule_chain_ir(AssocPlan, Descriptor, AssocRuleGlobalIR, AssocChainIR),
    plawk_assoc_rule_controls(AssocPlan, AssocRuleControls),
    plawk_assoc_break_close_ir(AssocRuleControls, BreakCloseIR),
    plawk_forin_end_writebin_ir(LoopVar, ArrayName, OutTypes, Fields, AssocPlan,
        EndPrintIR),
    plawk_writebin_globals(outfmt(OutTypes, OutRecordSize), WritebinGlobalIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w',
        [BeginGlobalIR, AssocRuleGlobalIR, WritebinGlobalIR]),
    plawk_combine_entry_ir(BeginIR, EntrySetupIR, CombinedEntrySetupIR0),
    plawk_combine_entry_ir(CombinedEntrySetupIR0, WritebinEntryIR, CombinedEntrySetupIR),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w',
        [EndPrintIR]),
    plawk_emit_record_driver_ir(Descriptor, InputPath,
        driver_blocks(RuntimeGlobals, CombinedEntrySetupIR, '', lowered_assoc,
            AssocChainIR, '', BreakCloseIR, end_print, CloseOkIR),
        DriverIR).

% Binary group-by to binary output: iterate the table and writebin one
% fixed-layout record per group. Binary input mode only -- text-mode
% keys are interned atom ids, which would be meaningless bytes in the
% output stream.
plawk_program_native_driver_ir(
    program(BeginClauses, Rules, [end([for_in(var(LoopVar), var(ArrayName), [writebin(Fields)])])]),
    InputPath,
    DriverIR
) :-
    plawk_record_descriptor(BeginClauses, FieldSeparator),
    FieldSeparator = binfmt(_InputTypes),
    plawk_begin_outfmt_types(BeginClauses, OutTypes),
    forall(member(OutType, OutTypes), memberchk(OutType, [i64, f64])),
    length(OutTypes, ArgCount),
    length(Fields, ArgCount),
    maplist(plawk_forin_writebin_field(LoopVar), Fields),
    findall(LookupArrayName,
        member(assoc(var(LookupArrayName), var(LoopVar)), Fields),
        LookupArrays),
    plawk_forin_assoc_plan(Rules, ArrayName, LookupArrays, AssocPlan),
    plawk_assoc_record_program_ok(FieldSeparator, Rules, []),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_assoc_entry_setup_ir(AssocPlan, EntrySetupIR),
    plawk_binfmt_record_size(binfmt(OutTypes), OutRecordSize),
    plawk_writebin_entry_lines(outfmt(OutTypes, OutRecordSize), WritebinEntryIR),
    plawk_assoc_rule_chain_ir(AssocPlan, FieldSeparator, AssocRuleGlobalIR, AssocChainIR),
    plawk_assoc_rule_controls(AssocPlan, AssocRuleControls),
    plawk_assoc_break_close_ir(AssocRuleControls, BreakCloseIR),
    plawk_forin_end_writebin_ir(LoopVar, ArrayName, OutTypes, Fields, AssocPlan,
        EndPrintIR),
    plawk_writebin_globals(outfmt(OutTypes, OutRecordSize), WritebinGlobalIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w',
        [BeginGlobalIR, AssocRuleGlobalIR, WritebinGlobalIR]),
    plawk_combine_entry_ir(BeginIR, EntrySetupIR, CombinedEntrySetupIR0),
    plawk_combine_entry_ir(CombinedEntrySetupIR0, WritebinEntryIR, CombinedEntrySetupIR),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w',
        [EndPrintIR]),
    plawk_emit_record_driver_ir(FieldSeparator, InputPath,
        driver_blocks(RuntimeGlobals, CombinedEntrySetupIR, '', lowered_assoc,
            AssocChainIR, '', BreakCloseIR, end_print, CloseOkIR),
        DriverIR).

% decode-into-struct END for-in (assoc for-in, stage 3): destructure each
% iterated value `arr[k]` through a grammar into typed fields, then print
% them -- `END { for (k in arr) { (n,m) = dyncall@decode(arr[k]) as (T..) ;
% print k, n, m } }`. The per-entry value is boxed and passed as the
% grammar argument (the `forin_val` operand); the record shim + object
% handle come from the program-wide dyncall support IR the outer driver
% assembles (the collectors recurse into the for-in body and find the
% destructure). Distinct four-part body shape, so ordering is unambiguous.
plawk_program_native_driver_ir(
    program(BeginClauses, Rules,
        [end([for_in(var(LoopVar), var(ArrayName),
            [dynrec_bind(Vars, Call, Types), print(PrintFields)])])]),
    InputPath,
    DriverIR
) :-
    Call = dyncall_named(_Name, [forin_val(ArrayName)]),
    plawk_forin_end_decode_plan(Rules, ArrayName, AssocPlan),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_record_descriptor(BeginClauses, FieldSeparator),
    plawk_assoc_record_program_ok(FieldSeparator, Rules, PrintFields),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_assoc_entry_setup_ir(AssocPlan, EntrySetupIR),
    plawk_assoc_rule_chain_ir(AssocPlan, FieldSeparator, AssocRuleGlobalIR, AssocChainIR),
    plawk_assoc_rule_controls(AssocPlan, AssocRuleControls),
    plawk_assoc_break_close_ir(AssocRuleControls, BreakCloseIR),
    plawk_forin_end_decode_ir(LoopVar, ArrayName, Vars, Call, Types,
        PrintFields, AssocPlan, FieldSeparator, OutputSeparator,
        DecodeGlobalIR, EndPrintIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w~n~w',
        [BeginGlobalIR, StringGlobalIR, AssocRuleGlobalIR, DecodeGlobalIR]),
    plawk_combine_entry_ir(BeginIR, EntrySetupIR, CombinedEntrySetupIR),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w',
        [EndPrintIR]),
    plawk_emit_record_driver_ir(FieldSeparator, InputPath,
        driver_blocks(RuntimeGlobals, CombinedEntrySetupIR, '', lowered_assoc,
            AssocChainIR, '', BreakCloseIR, end_print, CloseOkIR),
        DriverIR).
% accumulate-then-print END for-in (assoc for-in, stage 2): fold the FINAL
% hash into a loop-carried scalar, then print it --
% `END { for (k in arr) acc += OPERAND ; print acc }`. The for-in loop
% threads a second phi (the accumulator) beside the slot index; the trailing
% print reads the accumulated total after the loop. Distinct two-action END
% shape, so ordering among the END clauses is unambiguous.
plawk_program_native_driver_ir(
    program(BeginClauses, Rules,
        [end([for_in(var(LoopVar), var(ArrayName), [add(var(Acc), Operand)]),
              print(PrintFields)])]),
    InputPath,
    DriverIR
) :-
    plawk_forin_end_accum_plan(Rules, ArrayName, Operand, AssocPlan),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_record_descriptor(BeginClauses, FieldSeparator),
    plawk_assoc_record_program_ok(FieldSeparator, Rules, PrintFields),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_assoc_entry_setup_ir(AssocPlan, EntrySetupIR),
    plawk_assoc_rule_chain_ir(AssocPlan, FieldSeparator, AssocRuleGlobalIR, AssocChainIR),
    plawk_assoc_rule_controls(AssocPlan, AssocRuleControls),
    plawk_assoc_break_close_ir(AssocRuleControls, BreakCloseIR),
    plawk_forin_end_accum_ir(LoopVar, ArrayName, Acc, Operand, PrintFields,
        AssocPlan, FieldSeparator, OutputSeparator, EndPrintIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w',
        [BeginGlobalIR, StringGlobalIR, AssocRuleGlobalIR]),
    plawk_combine_entry_ir(BeginIR, EntrySetupIR, CombinedEntrySetupIR),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w',
        [EndPrintIR]),
    plawk_emit_record_driver_ir(FieldSeparator, InputPath,
        driver_blocks(RuntimeGlobals, CombinedEntrySetupIR, '', lowered_assoc,
            AssocChainIR, '', BreakCloseIR, end_print, CloseOkIR),
        DriverIR).
% guarded END for-in (stage 1b): iterate the FINAL hash and filter --
% `END { for (k in arr) { if (GUARD) print ... } }`. Matched before the
% generic END for-in clause so the guarded body routes to the guarded
% emitter; otherwise identical driver wiring.
plawk_program_native_driver_ir(
    program(BeginClauses, Rules,
        [end([for_in(var(LoopVar), var(ArrayName),
            [if(Guard, [print(PrintFields)], [])])])]),
    InputPath,
    DriverIR
) :-
    plawk_forin_end_plan(Rules, LoopVar, ArrayName,
        [if(Guard, [print(PrintFields)], [])], AssocPlan, PrintFields),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_record_descriptor(BeginClauses, FieldSeparator),
    plawk_assoc_record_program_ok(FieldSeparator, Rules, PrintFields),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_assoc_entry_setup_ir(AssocPlan, EntrySetupIR),
    plawk_assoc_rule_chain_ir(AssocPlan, FieldSeparator, AssocRuleGlobalIR, AssocChainIR),
    plawk_assoc_rule_controls(AssocPlan, AssocRuleControls),
    plawk_assoc_break_close_ir(AssocRuleControls, BreakCloseIR),
    plawk_forin_end_guarded_print_ir(LoopVar, ArrayName, Guard, PrintFields,
        AssocPlan, FieldSeparator, OutputSeparator, EndPrintIR, GuardGlobalIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w~n~w',
        [BeginGlobalIR, StringGlobalIR, AssocRuleGlobalIR, GuardGlobalIR]),
    plawk_combine_entry_ir(BeginIR, EntrySetupIR, CombinedEntrySetupIR),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w',
        [EndPrintIR]),
    plawk_emit_record_driver_ir(FieldSeparator, InputPath,
        driver_blocks(RuntimeGlobals, CombinedEntrySetupIR, '', lowered_assoc,
            AssocChainIR, '', BreakCloseIR, end_print, CloseOkIR),
        DriverIR).
plawk_program_native_driver_ir(
    program(BeginClauses, Rules, [end([for_in(var(LoopVar), var(ArrayName), BodyActions)])]),
    InputPath,
    DriverIR
) :-
    plawk_forin_end_plan(Rules, LoopVar, ArrayName, BodyActions, AssocPlan, PrintFields),
    AssocPlan = assoc_plan(Tables, _),
    plawk_program_cache_tables(BeginClauses, CacheNamePaths),
    plawk_cache_entries(Tables, [], [], CacheNamePaths, CacheEntries, CachePathGlobals),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_begin_print_string_globals(BeginClauses, BeginGlobalIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_record_descriptor(BeginClauses, FieldSeparator),
    plawk_assoc_record_program_ok(FieldSeparator, Rules, PrintFields),
    plawk_end_print_string_globals(PrintFields, StringGlobalIR),
    plawk_assoc_entry_setup_ir(AssocPlan, CacheEntries, EntrySetupIR),
    plawk_assoc_rule_chain_ir(AssocPlan, FieldSeparator, AssocRuleGlobalIR, AssocChainIR),
    plawk_assoc_rule_controls(AssocPlan, AssocRuleControls),
    plawk_assoc_break_close_ir(AssocRuleControls, BreakCloseIR),
    plawk_forin_end_print_ir(LoopVar, ArrayName, PrintFields, AssocPlan,
        FieldSeparator, OutputSeparator, EndPrintIR),
    plawk_cache_commit_lines(CacheEntries, CommitIR),
    format(atom(SurfaceGlobalIR), '~w~n~w~n~w~n~w',
        [BeginGlobalIR, StringGlobalIR, AssocRuleGlobalIR, CachePathGlobals]),
    plawk_combine_entry_ir(BeginIR, EntrySetupIR, CombinedEntrySetupIR),
    plawk_i64_end_print_globals(SurfaceGlobalIR, RuntimeGlobals),
    format(atom(CloseOkIR),
'end_print:
~w
~w',
        [CommitIR, EndPrintIR]),
    plawk_emit_record_driver_ir(FieldSeparator, InputPath,
        driver_blocks(RuntimeGlobals, CombinedEntrySetupIR, '', lowered_assoc,
            AssocChainIR, '', BreakCloseIR, end_print, CloseOkIR),
        DriverIR).

%% plawk_program_native_driver_ir(+Program, +InputPath, +Options, -DriverIR) is semidet.
%
%  Options-aware driver entry. Programs that call compiled Prolog
%  predicates (prolog_guard patterns / prolog_call expressions) need
%  Options to carry wam_vm(InstrCount, LabelCount) -- the counts
%  reported by wam_llvm_last_compile_counts/2 after
%  write_wam_llvm_project/3 compiled the predicates into the module.
%  The emitted support section holds a lazily created shared %WamState
%  and one wrapper function per called predicate; call sites marshal
%  arguments and invoke the wrappers, so the shared guard/expression
%  emitters need no VM plumbing. Programs with no foreign calls
%  delegate to plawk_program_native_driver_ir/3 unchanged.
plawk_program_native_driver_ir(Program, InputPath, Options, DriverIR) :-
    plawk_program_foreign_specs(Program, GuardSpecs, CallSpecs, FCallSpecs),
    plawk_program_dyncall_arities(Program, DynArities),
    plawk_program_dyncall_named_entries(Program, DynNamed),
    plawk_program_dyncall_named_float_entries(Program, DynNamedF),
    plawk_program_dyncall_named_blob_entries(Program, DynNamedB),
    plawk_program_dyncall_rec_arities(Program, DynRecArities),
    plawk_program_dyncall_named_rec_entries(Program, DynNamedRec),
    plawk_program_dyncall_named_assoc_entries(Program, DynNamedAssoc),
    plawk_program_dyncall_assoc_arities(Program, DynAssocArities),
    plawk_program_dyncall_named_assoc_str_entries(Program, DynNamedAssocS),
    plawk_program_dyncall_assoc_str_arities(Program, DynAssocSArities),
    plawk_program_dyncall_named_posarray_entries(Program, DynNamedPosarr),
    plawk_program_dyncall_posarray_arities(Program, DynPosarrArities),
    plawk_program_dyncall_named_posarray_str_entries(Program, DynNamedPosarrS),
    plawk_program_dyncall_posarray_str_arities(Program, DynPosarrSArities),
    plawk_program_dyncall_at_arities(Program, DynAtArities),
    plawk_program_dyncall_at_named_entries(Program, DynAtNamed),
    plawk_program_dyncall_at_named_float_entries(Program, DynAtNamedF),
    plawk_program_dyncall_at_named_blob_entries(Program, DynAtNamedB),
    plawk_program_dyncall_at_rec_arities(Program, DynAtRecArities),
    plawk_program_dyncall_at_named_rec_entries(Program, DynAtNamedRec),
    plawk_program_dyncall_float_arities(Program, DynFArities),
    plawk_program_dyncall_at_float_arities(Program, DynAtFArities),
    plawk_program_dyncall_blob_arities(Program, DynBArities),
    plawk_program_dyncall_at_blob_arities(Program, DynAtBArities),
    plawk_program_compile_sites(Program, CompileSites),
    (   GuardSpecs == [],
        CallSpecs == [],
        FCallSpecs == [],
        DynArities == [],
        DynNamed == [],
        DynNamedF == [],
        DynNamedB == [],
        DynRecArities == [],
        DynNamedRec == [],
        DynNamedAssoc == [],
        DynAssocArities == [],
        DynNamedAssocS == [],
        DynAssocSArities == [],
        DynNamedPosarr == [],
        DynPosarrArities == [],
        DynNamedPosarrS == [],
        DynPosarrSArities == [],
        DynAtArities == [],
        DynAtNamed == [],
        DynAtNamedF == [],
        DynAtNamedB == [],
        DynAtRecArities == [],
        DynAtNamedRec == [],
        DynFArities == [],
        DynAtFArities == [],
        DynBArities == [],
        DynAtBArities == [],
        CompileSites == []
    ->  plawk_program_native_driver_ir(Program, InputPath, DriverIR)
    ;   % Compiled-foreign support (shared VM + per-predicate wrappers)
        % only when the program actually calls a compiled predicate; it
        % needs the module's instruction/label counts. dyncall does not.
        (   GuardSpecs == [], CallSpecs == [], FCallSpecs == []
        ->  ForeignSupportIR = ''
        ;   memberchk(wam_vm(InstrCount, LabelCount), Options),
            plawk_foreign_support_ir(GuardSpecs, CallSpecs, FCallSpecs,
                InstrCount, LabelCount, ForeignSupportIR)
        ),
        % Object-call shims for dyncall(...) / float(dyncall(...)) /
        % blob(dyncall(...)) sites, plus the shared .wamo object handle.
        (   DynArities == [], DynFArities == [], DynBArities == [],
            DynNamed == [], DynNamedF == [], DynNamedB == [],
            DynRecArities == [], DynNamedRec == [], DynNamedAssoc == [],
            DynAssocArities == [], DynNamedAssocS == [],
            DynAssocSArities == [], DynNamedPosarr == [],
            DynPosarrArities == [], DynNamedPosarrS == [],
            DynPosarrSArities == []
        ->  DyncallSupportIR = ''
        ;   ( plawk_program_dynload_path(Program, DynPath)
            ->  true
            ;   throw(error(plawk_dyncall_without_dynload,
                    context(plawk_program_native_driver_ir,
                        'dyncall(...) requires BEGIN { DYNLOAD = "file.wamo" }')))
            ),
            plawk_dyncall_support_ir(DynPath, DynArities, DynFArities,
                DynBArities, DynNamed, DynNamedF, DynNamedB,
                DynRecArities, DynNamedRec, DynNamedAssoc, DynAssocArities,
                DynNamedAssocS, DynAssocSArities, DynNamedPosarr,
                DynPosarrArities, DynNamedPosarrS, DynPosarrSArities,
                DyncallSupportIR)
        ),
        % Dynamic-source shims for dyncall_at(...) / float(dyncall_at(...)) /
        % blob(dyncall_at(...)) sites + the shared path cache. A compile
        % site with NO dyncall_at site (a handle expression --
        % h = compile(...)) still needs the cache GLOBALS: @plawk_compile
        % records the loaded grammar in the registry.
        (   DynAtArities == [], DynAtFArities == [], DynAtBArities == [],
            DynAtNamed == [], DynAtNamedF == [], DynAtNamedB == [],
            DynAtRecArities == [], DynAtNamedRec == [],
            CompileSites == []
        ->  DyncallAtSupportIR = ''
        ;   plawk_program_dyncache_mode(Program, CacheMode),
            plawk_dyncall_at_support_ir(CacheMode, DynAtArities, DynAtFArities,
                DynAtBArities, DynAtNamed, DynAtNamedF, DynAtNamedB,
                DynAtRecArities, DynAtNamedRec, DyncallAtSupportIR)
        ),
        % The eval surface: compile(...) sites need @plawk_compile plus
        % the bootstrap-compiler object path (BEGIN { EVALC = "..." }
        % or the CLI-provided evalc_path option). The handle lives in
        % the dyncall_at cache registry, so mode "off" cannot carry it.
        (   CompileSites \== []
        ->  plawk_program_dyncache_mode(Program, CMode),
            (   CMode == "off"
            ->  throw(error(plawk_compile_requires_cache,
                    context(plawk_program_native_driver_ir,
                        'compile(...) requires DYNCACHE "on" or "mtime" (the grammar handle lives in the cache)')))
            ;   true
            ),
            (   plawk_program_evalc_path(Program, EvalcPath)
            ->  true
            ;   memberchk(evalc_path(EvalcPath), Options)
            ->  true
            ;   throw(error(plawk_compile_without_evalc,
                    context(plawk_program_native_driver_ir,
                        'compile(...) needs a compiler object: BEGIN { EVALC = "file.wamo" } or the CLI evalc_path option')))
            ),
            plawk_compile_support_ir(EvalcPath, CompileSupportIR)
        ;   CompileSupportIR = ''
        ),
        plawk_program_native_driver_ir(Program, InputPath, MainIR),
        exclude(==(''),
            [ForeignSupportIR, DyncallSupportIR, DyncallAtSupportIR,
             CompileSupportIR, MainIR],
            Parts),
        atomic_list_concat(Parts, '\n\n', DriverIR)
    ).

%% plawk_program_foreign_specs(+Program, -GuardSpecs, -CallSpecs)
%% plawk_program_foreign_specs(+Program, -GuardSpecs, -CallSpecs, -FCallSpecs)
%
%  Collect the deduplicated foreign predicate call shapes used by the
%  program. GuardSpecs are Name-NArgs pairs called as rule guards
%  (predicate arity NArgs); CallSpecs are Name-NArgs pairs called as
%  i64 expressions and FCallSpecs Name-NArgs pairs called as double
%  expressions via float(name(args)) (predicate arity NArgs + 1 for
%  the output in both).
plawk_program_foreign_specs(Program, GuardSpecs, CallSpecs) :-
    plawk_program_foreign_specs(Program, GuardSpecs, CallSpecs, _FCallSpecs).

plawk_program_foreign_specs(program(_BeginClauses, Rules, EndClauses),
        GuardSpecs, CallSpecs, FCallSpecs) :-
    findall(Name-NArgs,
        ( member(rule(Pattern, _Actions), Rules),
          plawk_pattern_prolog_guard(Pattern, Name, Args),
          length(Args, NArgs)
        ),
        GuardSpecs0),
    findall(Name-NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          plawk_actions_prolog_call(Actions, Name, Args),
          length(Args, NArgs)
        ),
        CallSpecs0),
    findall(Name-NArgs,
        ( ( member(rule(_PatternF, FActions), Rules)
          ; member(end(FActions), EndClauses)
          ),
          plawk_actions_prolog_fcall(FActions, Name, Args),
          length(Args, NArgs)
        ),
        FCallSpecs0),
    findall(Name-NArgs,
        ( member(rule(_Pattern2, Actions2), Rules),
          member(if(CondPattern, _Then, _Else), Actions2),
          plawk_pattern_prolog_guard(CondPattern, Name, Args),
          length(Args, NArgs)
        ),
        CondGuardSpecs0),
    append(GuardSpecs0, CondGuardSpecs0, AllGuardSpecs),
    sort(AllGuardSpecs, GuardSpecs),
    sort(CallSpecs0, CallSpecs),
    sort(FCallSpecs0, FCallSpecs).

plawk_pattern_prolog_guard(prolog_guard(Name, Args), Name, Args).
plawk_pattern_prolog_guard(and_pat(Left, Right), Name, Args) :-
    ( plawk_pattern_prolog_guard(Left, Name, Args)
    ; plawk_pattern_prolog_guard(Right, Name, Args)
    ).
plawk_pattern_prolog_guard(or_pat(Left, Right), Name, Args) :-
    ( plawk_pattern_prolog_guard(Left, Name, Args)
    ; plawk_pattern_prolog_guard(Right, Name, Args)
    ).
plawk_pattern_prolog_guard(not_pat(Pattern), Name, Args) :-
    plawk_pattern_prolog_guard(Pattern, Name, Args).

%% plawk_program_dyncall_arities(+Program, -Arities)
%
%  Deduplicated set of argument counts used by dyncall(...) sites across
%  rule and END bodies. One object-call shim @plawk_dyncall_N is emitted
%  per arity.
plawk_program_dyncall_arities(program(_Begin, Rules, EndClauses), Arities) :-
    findall(NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          plawk_actions_dyncall(Actions, Args),
          length(Args, NArgs)
        ),
        Arities0),
    sort(Arities0, Arities).

%% plawk_program_dynload_path(+Program, -Path)
%  The .wamo path from BEGIN { DYNLOAD = "..." }, or fails if absent.
plawk_program_dynload_path(program(BeginClauses, _Rules, _End), Path) :-
    member(begin(Actions), BeginClauses),
    member(set(var('DYNLOAD'), string(Path)), Actions),
    !.

plawk_actions_dyncall(Actions, Args) :-
    member(Action, Actions),
    plawk_action_dyncall(Action, Args).
plawk_action_dyncall(add(_Var, Expr), Args) :- plawk_expr_dyncall(Expr, Args).
plawk_action_dyncall(set(_Var, Expr), Args) :- plawk_expr_dyncall(Expr, Args).
plawk_action_dyncall(print(Fields), Args) :-
    member(Field, Fields),
    plawk_expr_dyncall(Field, Args).
plawk_action_dyncall(printf(_Format, PrintfArgs), Args) :-
    member(Field, PrintfArgs),
    plawk_expr_dyncall(Field, Args).
plawk_action_dyncall(if(_Pattern, ThenActions, ElseActions), Args) :-
    ( plawk_actions_dyncall(ThenActions, Args)
    ; plawk_actions_dyncall(ElseActions, Args)
    ).
plawk_expr_dyncall(dyncall(Args), Args).
plawk_expr_dyncall(Expr, Args) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    ( plawk_expr_dyncall(Left, Args)
    ; plawk_expr_dyncall(Right, Args)
    ).

%% plawk_program_dyncall_named_entries(+Program, -Entries)
%
%  Deduplicated set of Name-NArgs pairs for every dyncall@name(...) site
%  across rule and END bodies. One resolver+shim
%  @plawk_dyncall_named_<Name>_<NArgs> is emitted per distinct (name,
%  arity); it resolves the named entry's label index against the DYNLOAD
%  object once (via @wam_object_entry_index) and caches the PC. NArgs is
%  the call's argument count; the entry predicate's arity is NArgs+1 (the
%  inputs plus the output cell), which is the arity in the "Name/Arity"
%  string the loader matches.
plawk_program_dyncall_named_entries(Program, Entries) :-
    plawk_program_named_entries_of(dyncall_named, Program, Entries).

%% plawk_program_dyncall_named_float_entries(+Program, -Entries)
%% plawk_program_dyncall_named_blob_entries(+Program, -Entries)
%  Named-entry sites reached through float(dyncall@name(...)) and
%  blob(dyncall@name(...)): double- and byte-returning shims respectively.
%  Same Name-NArgs shape; all three kinds share one per-entry PC resolver.
plawk_program_dyncall_named_float_entries(Program, Entries) :-
    plawk_program_named_entries_of(float_dyncall_named, Program, Entries).
% blob named entries via the generic blob walk (covers patterns,
% writebin slots and assoc keys too, not just action fields).
plawk_program_dyncall_named_blob_entries(Program, Entries) :-
    findall(Name-NArgs,
        ( plawk_program_blob_node(Program, blob_dyncall_named(Name, Args)),
          length(Args, NArgs)
        ),
        E0),
    sort(E0, Entries).

%% plawk_program_dyncall_rec_arities(+Program, -Arities)
%% plawk_program_dyncall_named_rec_entries(+Program, -Entries)
%  Structured-return destructure sites -- `(...) = dyncall(...) as (...)`
%  (default entry, keyed by arg count) and `(...) = dyncall@name(...) as
%  (...)` (named entry, keyed by Name-NArgs). Each gets a record shim that
%  fills the caller's typed slots via @wam_object_call_record.
plawk_program_dyncall_rec_arities(program(_Begin, Rules, EndClauses), Arities) :-
    findall(NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          member(Action, Actions),
          plawk_subterm_dynrec_call(Action, dyncall(Args)),
          length(Args, NArgs)
        ),
        Arities0),
    sort(Arities0, Arities).
plawk_program_dyncall_named_rec_entries(program(_Begin, Rules, EndClauses),
        Entries) :-
    findall(Name-NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          member(Action, Actions),
          plawk_subterm_dynrec_call(Action, dyncall_named(Name, Args)),
          length(Args, NArgs)
        ),
        Entries0),
    sort(Entries0, Entries).

%% plawk_program_dyncall_named_assoc_entries(+Program, -Entries)
%  Named entries used in `arr = dyncall@name(args) as assoc` sites -- each
%  gets a @plawk_dyncall_assoc_<Name>_<N> shim that populates a caller table
%  via @wam_object_call_assoc.
plawk_program_dyncall_named_assoc_entries(program(_Begin, Rules, EndClauses),
        Entries) :-
    findall(Name-NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          member(Action, Actions),
          plawk_subterm_dynassoc_call(Action, dyncall_named(Name, Args)),
          length(Args, NArgs)
        ),
        Entries0),
    sort(Entries0, Entries).

%% plawk_program_dyncall_assoc_arities(+Program, -Arities)
%  Default-entry `arr = dyncall(args) as assoc` sites (deferred small
%  item) -- one @plawk_dyncall_assoc_default_<N> shim per arity; the
%  entry is the DYNLOAD object's default (wamo_entry), resolved like a
%  plain dyncall.
plawk_program_dyncall_assoc_arities(program(_Begin, Rules, EndClauses),
        Arities) :-
    findall(NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          member(Action, Actions),
          plawk_subterm_dynassoc_call(Action, dyncall(Args)),
          length(Args, NArgs)
        ),
        A0),
    sort(A0, Arities).

%% plawk_program_dyncall_named_assoc_str_entries(+Program, -Entries)
%% plawk_program_dyncall_assoc_str_arities(+Program, -Arities)
%  The str-valued table kind: `arr = dyncall@name(args) as assoc(str)`
%  sites get @plawk_dyncall_assoc_str_<Name>_<N> shims, default-entry
%  sites @plawk_dyncall_assoc_str_default_<N> -- both forwarding to
%  @wam_object_call_assoc_str (atom values, replace semantics).
plawk_program_dyncall_named_assoc_str_entries(
        program(_Begin, Rules, EndClauses), Entries) :-
    findall(Name-NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          member(Action, Actions),
          plawk_subterm_dynassoc_str_call(Action, dyncall_named(Name, Args)),
          length(Args, NArgs)
        ),
        Entries0),
    sort(Entries0, Entries).

plawk_program_dyncall_assoc_str_arities(program(_Begin, Rules, EndClauses),
        Arities) :-
    findall(NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          member(Action, Actions),
          plawk_subterm_dynassoc_str_call(Action, dyncall(Args)),
          length(Args, NArgs)
        ),
        A0),
    sort(A0, Arities).

plawk_subterm_dynassoc_call(dynassoc_bind(_Var, Call), Call).
plawk_subterm_dynassoc_call(Term, Call) :-
    compound(Term),
    arg(_, Term, Sub),
    plawk_subterm_dynassoc_call(Sub, Call).

plawk_subterm_dynassoc_str_call(dynassoc_bind_str(_Var, Call), Call).
plawk_subterm_dynassoc_str_call(Term, Call) :-
    compound(Term),
    arg(_, Term, Sub),
    plawk_subterm_dynassoc_str_call(Sub, Call).

%% plawk_program_dyncall_named_posarray_entries(+Program, -Entries)
%% plawk_program_dyncall_posarray_arities(+Program, -Arities)
%  Positional-array target (`arr = dyncall[@name](args) as array`): named
%  sites get @plawk_dyncall_posarray_<Name>_<N> shims, default-entry sites
%  @plawk_dyncall_posarray_default_<N> -- both forwarding to
%  @wam_object_call_posarray (flat [V1..Vn] list walked into keys 1..n,
%  i64 values, replace semantics).
plawk_program_dyncall_named_posarray_entries(
        program(_Begin, Rules, EndClauses), Entries) :-
    findall(Name-NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          member(Action, Actions),
          plawk_subterm_dynposarray_call(Action, dyncall_named(Name, Args)),
          length(Args, NArgs)
        ),
        Entries0),
    sort(Entries0, Entries).

plawk_program_dyncall_posarray_arities(program(_Begin, Rules, EndClauses),
        Arities) :-
    findall(NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          member(Action, Actions),
          plawk_subterm_dynposarray_call(Action, dyncall(Args)),
          length(Args, NArgs)
        ),
        A0),
    sort(A0, Arities).

plawk_subterm_dynposarray_call(dynposarray_bind(_Var, Call), Call).
plawk_subterm_dynposarray_call(Term, Call) :-
    compound(Term),
    arg(_, Term, Sub),
    plawk_subterm_dynposarray_call(Sub, Call).

%% plawk_program_dyncall_named_posarray_str_entries(+Program, -Entries)
%% plawk_program_dyncall_posarray_str_arities(+Program, -Arities)
%  Str-valued positional array (`arr = dyncall[@name](args) as array(str)`):
%  named sites get @plawk_dyncall_posarray_str_<Name>_<N> shims,
%  default-entry sites @plawk_dyncall_posarray_str_default_<N> -- both
%  forwarding to @wam_object_call_posarray_str (flat [Atom..] list, atom
%  ids stored by position, resolved to text on read).
plawk_program_dyncall_named_posarray_str_entries(
        program(_Begin, Rules, EndClauses), Entries) :-
    findall(Name-NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          member(Action, Actions),
          plawk_subterm_dynposarray_str_call(Action, dyncall_named(Name, Args)),
          length(Args, NArgs)
        ),
        Entries0),
    sort(Entries0, Entries).

plawk_program_dyncall_posarray_str_arities(program(_Begin, Rules, EndClauses),
        Arities) :-
    findall(NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          member(Action, Actions),
          plawk_subterm_dynposarray_str_call(Action, dyncall(Args)),
          length(Args, NArgs)
        ),
        A0),
    sort(A0, Arities).

plawk_subterm_dynposarray_str_call(dynposarray_bind_str(_Var, Call), Call).
plawk_subterm_dynposarray_str_call(Term, Call) :-
    compound(Term),
    arg(_, Term, Sub),
    plawk_subterm_dynposarray_str_call(Sub, Call).

plawk_subterm_dynrec_call(dynrec_bind(_Vars, Call, _Types), Call).
plawk_subterm_dynrec_call(dynrec_view(Call, _Types, _Body), Call).
plawk_subterm_dynrec_call(Term, Call) :-
    compound(Term),
    arg(_, Term, Sub),
    plawk_subterm_dynrec_call(Sub, Call).

%% plawk_program_named_entries_of(+Functor, +Program, -Entries)
%  Deduplicated Name-NArgs pairs for every Functor(Name, Args) node across
%  rule and END bodies. NArgs is the call's argument count; the entry
%  predicate's arity is NArgs+1 (inputs plus the output cell), which is the
%  arity in the "Name/Arity" string the loader matches.
plawk_program_named_entries_of(Functor,
        program(_Begin, Rules, EndClauses), Entries) :-
    findall(Name-NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          member(Action, Actions),
          plawk_subterm_named(Functor, Action, Name, Args),
          length(Args, NArgs)
        ),
        Entries0),
    sort(Entries0, Entries).

% Find every Functor(Name, Args) node anywhere within a term (print fields,
% accumulator updates, if-branches, binary sub-expressions), so the
% collectors need not mirror each structural walker.
plawk_subterm_named(Functor, Term, Name, Args) :-
    Term =.. [Functor, Name, Args].
plawk_subterm_named(Functor, Term, Name, Args) :-
    compound(Term),
    arg(_, Term, Sub),
    plawk_subterm_named(Functor, Sub, Name, Args).

plawk_actions_prolog_call(Actions, Name, Args) :-
    member(Action, Actions),
    plawk_action_prolog_call(Action, Name, Args).

plawk_action_prolog_call(add(_Var, Expr), Name, Args) :-
    plawk_expr_prolog_call(Expr, Name, Args).
plawk_action_prolog_call(set(_Var, Expr), Name, Args) :-
    plawk_expr_prolog_call(Expr, Name, Args).
plawk_action_prolog_call(print(Fields), Name, Args) :-
    member(Field, Fields),
    plawk_expr_prolog_call(Field, Name, Args).
plawk_action_prolog_call(printf(_Format, PrintfArgs), Name, Args) :-
    member(Field, PrintfArgs),
    plawk_expr_prolog_call(Field, Name, Args).
plawk_action_prolog_call(if(_Pattern, ThenActions, ElseActions), Name, Args) :-
    ( plawk_actions_prolog_call(ThenActions, Name, Args)
    ; plawk_actions_prolog_call(ElseActions, Name, Args)
    ).

plawk_expr_prolog_call(prolog_call(Name, Args), Name, Args).
plawk_expr_prolog_call(Expr, Name, Args) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    ( plawk_expr_prolog_call(Left, Name, Args)
    ; plawk_expr_prolog_call(Right, Name, Args)
    ).

% float(name(args)) call sites -- same walk, double-returning wrappers.
plawk_actions_prolog_fcall(Actions, Name, Args) :-
    member(Action, Actions),
    plawk_action_prolog_fcall(Action, Name, Args).

plawk_action_prolog_fcall(add(_Var, Expr), Name, Args) :-
    plawk_expr_prolog_fcall(Expr, Name, Args).
plawk_action_prolog_fcall(set(_Var, Expr), Name, Args) :-
    plawk_expr_prolog_fcall(Expr, Name, Args).
plawk_action_prolog_fcall(print(Fields), Name, Args) :-
    member(Field, Fields),
    plawk_expr_prolog_fcall(Field, Name, Args).
plawk_action_prolog_fcall(printf(_Format, PrintfArgs), Name, Args) :-
    member(Field, PrintfArgs),
    plawk_expr_prolog_fcall(Field, Name, Args).
plawk_action_prolog_fcall(if(_Pattern, ThenActions, ElseActions), Name, Args) :-
    ( plawk_actions_prolog_fcall(ThenActions, Name, Args)
    ; plawk_actions_prolog_fcall(ElseActions, Name, Args)
    ).

plawk_expr_prolog_fcall(float_call(Name, Args), Name, Args).
plawk_expr_prolog_fcall(Expr, Name, Args) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    ( plawk_expr_prolog_fcall(Left, Name, Args)
    ; plawk_expr_prolog_fcall(Right, Name, Args)
    ).

%% plawk_foreign_support_ir(+GuardSpecs, +CallSpecs, +FCallSpecs,
%%     +InstrCount, +LabelCount, -IR)
%
%  Emit the shared foreign-call support section: a lazily initialized
%  process-wide %WamState plus one wrapper per called predicate. Guard
%  wrappers return run_loop's success directly. Call wrappers push one
%  unbound output cell, run the predicate, and return {value, ok};
%  failure or a non-integer binding yields {0, false}. Float-call
%  wrappers (float(name(args)) sites) accept an Integer or Float
%  output and return {double, ok} via @value_to_double. All wrappers
%  save and restore the VM heap top (WamState field 6) and rewind the
%  arena via @wam_cleanup, so per-record foreign calls run in constant
%  memory -- nothing WAM-side persists between plawk calls.
plawk_foreign_support_ir(GuardSpecs, CallSpecs, FCallSpecs, InstrCount,
        LabelCount, IR) :-
    format(atom(VmIR),
'@plawk_foreign_vm = internal global %WamState* null

define %WamState* @plawk_foreign_vm_get() {
entry:
  %cur = load %WamState*, %WamState** @plawk_foreign_vm
  %have = icmp ne %WamState* %cur, null
  br i1 %have, label %ret_cur, label %make

ret_cur:
  ret %WamState* %cur

make:
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
  store %WamState* %vm, %WamState** @plawk_foreign_vm
  ret %WamState* %vm
}',
        [InstrCount, InstrCount, InstrCount, LabelCount, LabelCount,
         LabelCount]),
    findall(GuardIR,
        ( member(Name-NArgs, GuardSpecs),
          plawk_foreign_guard_wrapper_ir(Name, NArgs, GuardIR)
        ),
        GuardIRs),
    findall(CallIR,
        ( member(Name-NArgs, CallSpecs),
          plawk_foreign_call_wrapper_ir(Name, NArgs, CallIR)
        ),
        CallIRs),
    findall(FCallIR,
        ( member(Name-NArgs, FCallSpecs),
          plawk_foreign_fcall_wrapper_ir(Name, NArgs, FCallIR)
        ),
        FCallIRs),
    append([[VmIR], GuardIRs, CallIRs, FCallIRs], Parts),
    atomic_list_concat(Parts, '\n\n', IR).

plawk_foreign_wrapper_params(NArgs, ParamsIR) :-
    NArgs >= 1,
    NArgs1 is NArgs - 1,
    numlist(0, NArgs1, Ns),
    findall(Param,
        ( member(N, Ns),
          format(atom(Param), '%Value %a~w', [N])
        ),
        Params),
    atomic_list_concat(Params, ', ', ParamsIR).

plawk_foreign_set_reg_lines(NArgs, Lines) :-
    NArgs1 is NArgs - 1,
    numlist(0, NArgs1, Ns),
    findall(Line,
        ( member(N, Ns),
          format(atom(Line),
              '  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %a~w)',
              [N, N])
        ),
        Lines).

plawk_foreign_guard_wrapper_ir(Name, NArgs, IR) :-
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_foreign_set_reg_lines(NArgs, SetRegLines),
    atomic_list_concat(SetRegLines, '\n', SetRegIR),
    format(atom(IR),
'define i1 @plawk_foreign_guard_~w_~w(~w) {
entry:
  %vm = call %WamState* @plawk_foreign_vm_get()
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
  %hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %hs_saved = load i32, i32* %hs_ptr
  %pc = load i32, i32* @~w_start_pc
  call void @wam_prepare_call(%WamState* %vm, i32 %pc)
~w
  %ok = call i1 @run_loop(%WamState* %vm)
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  ret i1 %ok

fail:
  ret i1 false
}',
        [Name, NArgs, ParamsIR, Name, SetRegIR]).

plawk_foreign_call_wrapper_ir(Name, NArgs, IR) :-
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_foreign_set_reg_lines(NArgs, SetRegLines),
    atomic_list_concat(SetRegLines, '\n', SetRegIR),
    format(atom(IR),
'define { i64, i1 } @plawk_foreign_call_~w_~w(~w) {
entry:
  %vm = call %WamState* @plawk_foreign_vm_get()
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
  %hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %hs_saved = load i32, i32* %hs_ptr
  %pc = load i32, i32* @~w_start_pc
  %unb = call %Value @value_unbound(i8* null)
  %out_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %out_ref = call %Value @value_ref(i32 %out_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %pc)
~w
  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %out_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %read_out, label %rewind_fail

read_out:
  %out = call %Value @wam_deref_value(%WamState* %vm, %Value %out_ref)
  %out_tag = extractvalue %Value %out, 0
  %out_is_int = icmp eq i32 %out_tag, 1
  br i1 %out_is_int, label %good, label %rewind_fail

good:
  %payload = extractvalue %Value %out, 1
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  %r0 = insertvalue { i64, i1 } undef, i64 %payload, 0
  %r1 = insertvalue { i64, i1 } %r0, i1 true, 1
  ret { i64, i1 } %r1

rewind_fail:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  br label %fail

fail:
  %f0 = insertvalue { i64, i1 } undef, i64 0, 0
  %f1 = insertvalue { i64, i1 } %f0, i1 false, 1
  ret { i64, i1 } %f1
}',
        [Name, NArgs, ParamsIR, Name, SetRegIR, NArgs]).

% The double-returning variant behind float(name(args)): identical
% shape, but the output cell may bind to an Integer or a Float --
% @value_to_double promotes either to double.
plawk_foreign_fcall_wrapper_ir(Name, NArgs, IR) :-
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_foreign_set_reg_lines(NArgs, SetRegLines),
    atomic_list_concat(SetRegLines, '\n', SetRegIR),
    format(atom(IR),
'define { double, i1 } @plawk_foreign_fcall_~w_~w(~w) {
entry:
  %vm = call %WamState* @plawk_foreign_vm_get()
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
  %hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %hs_saved = load i32, i32* %hs_ptr
  %pc = load i32, i32* @~w_start_pc
  %unb = call %Value @value_unbound(i8* null)
  %out_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %out_ref = call %Value @value_ref(i32 %out_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %pc)
~w
  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %out_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %read_out, label %rewind_fail

read_out:
  %out = call %Value @wam_deref_value(%WamState* %vm, %Value %out_ref)
  %out_is_num = call i1 @value_is_number(%Value %out)
  br i1 %out_is_num, label %good, label %rewind_fail

good:
  %fval = call double @value_to_double(%Value %out)
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  %r0 = insertvalue { double, i1 } undef, double %fval, 0
  %r1 = insertvalue { double, i1 } %r0, i1 true, 1
  ret { double, i1 } %r1

rewind_fail:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  br label %fail

fail:
  %f0 = insertvalue { double, i1 } undef, double 0.0, 0
  %f1 = insertvalue { double, i1 } %f0, i1 false, 1
  ret { double, i1 } %f1
}',
        [Name, NArgs, ParamsIR, Name, SetRegIR, NArgs]).

%% plawk_dyncall_support_ir(+Path, +Arities, -IR)
%
%  Emit the dyncall runtime for a program: the .wamo path string, a
%  lazily loaded object handle (%WamState* + entry PC), and one
%  @plawk_dyncall_N shim per arity. The shim boxes N %Value args into a
%  stack array and calls @wam_object_call_i64 (emitted into the host
%  module by write_wam_llvm_project/3 with emit_wamo_loader(true)),
%  reading the object's entry as `entry(A0..A_{N-1}, out=A_N)`. The
%  object loads on the first dyncall and is reused for the rest of the
%  run -- the object-call primitive rewinds the arena per call, so this
%  stays constant-memory just like the compiled foreign bridge.
plawk_dyncall_support_ir(Path, Arities, IR) :-
    plawk_dyncall_support_ir(Path, Arities, [], [], [], [], [], [], [], [], IR).
plawk_dyncall_support_ir(Path, IArities, FArities, IR) :-
    plawk_dyncall_support_ir(Path, IArities, FArities, [], [], [], [], [], [], [], IR).
plawk_dyncall_support_ir(Path, IArities, FArities, BArities, IR) :-
    plawk_dyncall_support_ir(Path, IArities, FArities, BArities, [], [], [], [], [], [], IR).
plawk_dyncall_support_ir(Path, IArities, FArities, BArities, NamedEntries, IR) :-
    plawk_dyncall_support_ir(Path, IArities, FArities, BArities,
        NamedEntries, [], [], [], [], [], IR).
plawk_dyncall_support_ir(Path, IArities, FArities, BArities,
        NamedI, NamedF, NamedB, IR) :-
    plawk_dyncall_support_ir(Path, IArities, FArities, BArities,
        NamedI, NamedF, NamedB, [], [], [], IR).
plawk_dyncall_support_ir(Path, IArities, FArities, BArities,
        NamedI, NamedF, NamedB, RecArities, NamedRec, IR) :-
    plawk_dyncall_support_ir(Path, IArities, FArities, BArities,
        NamedI, NamedF, NamedB, RecArities, NamedRec, [], [], [], [], IR).
plawk_dyncall_support_ir(Path, IArities, FArities, BArities,
        NamedI, NamedF, NamedB, RecArities, NamedRec, NamedAssoc, IR) :-
    plawk_dyncall_support_ir(Path, IArities, FArities, BArities,
        NamedI, NamedF, NamedB, RecArities, NamedRec, NamedAssoc,
        [], [], [], IR).
plawk_dyncall_support_ir(Path, IArities, FArities, BArities,
        NamedI, NamedF, NamedB, RecArities, NamedRec, NamedAssoc,
        AssocArities, IR) :-
    plawk_dyncall_support_ir(Path, IArities, FArities, BArities,
        NamedI, NamedF, NamedB, RecArities, NamedRec, NamedAssoc,
        AssocArities, [], [], IR).

%% plawk_dyncall_support_ir(+Path, +IArities, +FArities, +BArities,
%%                          +NamedI, +NamedF, +NamedB, +RecArities,
%%                          +NamedRec, +NamedAssoc, +AssocArities,
%%                          +NamedAssocStr, +AssocStrArities, -IR)
%  IArities -> i64 @plawk_dyncall_N shims; FArities -> double
%  @plawk_dyncall_f_N shims (float(dyncall(...))); BArities -> byte-slice
%  @plawk_dyncall_b_N shims (blob(dyncall(...))). NamedI/NamedF/NamedB are
%  Name-NArgs lists for dyncall@name / float(dyncall@name) /
%  blob(dyncall@name): each named entry (across all kinds) gets ONE shared
%  PC resolver @plawk_dyncall_resolve_<Name>_<N> that resolves "Name/(N+1)"
%  to a label index against the DYNLOAD object once and caches the PC
%  (sentinel -1 = unresolved); the shims call the resolver then the matching
%  @wam_object_call_*. RecArities -> i1 @plawk_dyncall_rec_N record shims
%  (default entry) and NamedRec -> @plawk_dyncall_named_rec_<Name>_<N>
%  record shims, both forwarding to @wam_object_call_record for structured
%  destructure binds (they also feed the resolver union). NamedAssoc ->
%  @plawk_dyncall_assoc_<Name>_<N> assoc-populating shims and AssocArities
%  -> their default-entry @plawk_dyncall_assoc_default_<N> counterparts.
%  NamedAssocStr / AssocStrArities are the str-valued table kind
%  (@plawk_dyncall_assoc_str_* shims forwarding to
%  @wam_object_call_assoc_str). All share the single lazily loaded object
%  handle + @plawk_dyncall_get.
plawk_dyncall_support_ir(Path, IArities, FArities, BArities,
        NamedI, NamedF, NamedB, RecArities, NamedRec, NamedAssoc,
        AssocArities, NamedAssocStr, AssocStrArities,
        NamedPosarray, PosarrayArities,
        NamedPosarrayStr, PosarrayStrArities, IR) :-
    llvm_emit_c_string_global('plawk_dyncall_path', Path, PathGlobal,
        _StrLen, BytesLen),
    format(atom(GetterIR),
'@plawk_dyncall_vm = internal global %WamState* null
@plawk_dyncall_pc = internal global i32 0

define %WamState* @plawk_dyncall_get() {
entry:
  %cur = load %WamState*, %WamState** @plawk_dyncall_vm
  %have = icmp ne %WamState* %cur, null
  br i1 %have, label %ret_cur, label %load

ret_cur:
  ret %WamState* %cur

load:
  %pathp = getelementptr [~w x i8], [~w x i8]* @.plawk_dyncall_path, i32 0, i32 0
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %pathp)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %pc = extractvalue { %WamState*, i32 } %obj, 1
  store %WamState* %vm, %WamState** @plawk_dyncall_vm
  store i32 %pc, i32* @plawk_dyncall_pc
  ret %WamState* %vm
}',
        [BytesLen, BytesLen]),
    findall(ShimIR,
        ( member(N, IArities), plawk_dyncall_shim_ir(N, ShimIR) ),
        IShims),
    findall(FShimIR,
        ( member(FN, FArities), plawk_dyncall_shim_f_ir(FN, FShimIR) ),
        FShims),
    findall(BShimIR,
        ( member(BN, BArities), plawk_dyncall_shim_b_ir(BN, BShimIR) ),
        BShims),
    % one shared resolver per named entry, over the union of the kinds
    % (record binds that name an entry feed the union too, so their
    % resolver exists even when the entry is used nowhere else)
    append([NamedI, NamedF, NamedB, NamedRec, NamedAssoc, NamedAssocStr,
            NamedPosarray, NamedPosarrayStr],
        NamedAll0),
    sort(NamedAll0, NamedAll),
    findall(ResIR,
        ( member(REName-RENArgs, NamedAll),
          plawk_dyncall_named_resolver_ir(REName, RENArgs, BytesLen, ResIR) ),
        Resolvers),
    findall(NIShim,
        ( member(NIName-NINArgs, NamedI),
          plawk_dyncall_named_shim_ir(NIName, NINArgs, NIShim) ),
        NIShims),
    findall(NFShim,
        ( member(NFName-NFNArgs, NamedF),
          plawk_dyncall_named_shim_f_ir(NFName, NFNArgs, NFShim) ),
        NFShims),
    findall(NBShim,
        ( member(NBName-NBNArgs, NamedB),
          plawk_dyncall_named_shim_b_ir(NBName, NBNArgs, NBShim) ),
        NBShims),
    findall(RecShim,
        ( member(RN, RecArities), plawk_dyncall_rec_shim_ir(RN, RecShim) ),
        RecShims),
    findall(NRecShim,
        ( member(NRName-NRNArgs, NamedRec),
          plawk_dyncall_named_rec_shim_ir(NRName, NRNArgs, NRecShim) ),
        NRecShims),
    findall(NAShim,
        ( member(NAName-NANArgs, NamedAssoc),
          plawk_dyncall_named_assoc_shim_ir(NAName, NANArgs, NAShim) ),
        NAShims),
    findall(DAShim,
        ( member(DAN, AssocArities),
          plawk_dyncall_assoc_default_shim_ir(DAN, DAShim) ),
        DAShims),
    findall(NSShim,
        ( member(NSName-NSNArgs, NamedAssocStr),
          plawk_dyncall_named_assoc_str_shim_ir(NSName, NSNArgs, NSShim) ),
        NSShims),
    findall(DSShim,
        ( member(DSN, AssocStrArities),
          plawk_dyncall_assoc_default_str_shim_ir(DSN, DSShim) ),
        DSShims),
    findall(NPShim,
        ( member(NPName-NPNArgs, NamedPosarray),
          plawk_dyncall_named_posarray_shim_ir(NPName, NPNArgs, NPShim) ),
        NPShims),
    findall(DPShim,
        ( member(DPN, PosarrayArities),
          plawk_dyncall_posarray_default_shim_ir(DPN, DPShim) ),
        DPShims),
    findall(NPSShim,
        ( member(NPSName-NPSNArgs, NamedPosarrayStr),
          plawk_dyncall_named_posarray_str_shim_ir(NPSName, NPSNArgs, NPSShim) ),
        NPSShims),
    findall(DPSShim,
        ( member(DPSN, PosarrayStrArities),
          plawk_dyncall_posarray_str_default_shim_ir(DPSN, DPSShim) ),
        DPSShims),
    append([[PathGlobal, GetterIR], IShims, FShims, BShims,
            Resolvers, NIShims, NFShims, NBShims, RecShims, NRecShims,
            NAShims, DAShims, NSShims, DSShims, NPShims, DPShims,
            NPSShims, DPSShims], Parts),
    atomic_list_concat(Parts, '\n\n', IR).

%% plawk_dyncall_named_assoc_shim_ir(+Name, +NArgs, -IR)
%  Assoc shim for `arr = dyncall@name(args) as assoc`: resolve the PC via
%  the shared resolver, box args, and forward the caller's assoc table to
%  @wam_object_call_assoc (which walks the returned [K-V,...] pairs into it).
plawk_dyncall_named_assoc_shim_ir(Name, NArgs, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(IR),
'define i1 @plawk_dyncall_assoc_~w(~w, %WamAssocI64Table* %table) {
entry:
  %pc = call i32 @plawk_dyncall_resolve_~w()
  %bad = icmp slt i32 %pc, 0
  br i1 %bad, label %fail, label %do_call

do_call:
  %vm = call %WamState* @plawk_dyncall_get()
  %args = alloca %Value, i32 ~w
~w
  %r = call i1 @wam_object_call_assoc(%WamState* %vm, i32 %pc, i32 ~w, %Value* %args, i32 ~w, %WamAssocI64Table* %table)
  ret i1 %r

fail:
  ret i1 false
}',
        [Sym, ParamsIR, Sym, NArgs, StoreIR, NArgs, NArgs]).

%% plawk_dyncall_assoc_default_shim_ir(+NArgs, -IR)
%  Assoc shim for `arr = dyncall(args) as assoc` against the DYNLOAD
%  object's default entry: load the shared object handle, take the entry
%  PC recorded at load time (no resolver), box args, and forward the
%  caller's assoc table to @wam_object_call_assoc.
plawk_dyncall_assoc_default_shim_ir(NArgs, IR) :-
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(IR),
'define i1 @plawk_dyncall_assoc_default_~w(~w, %WamAssocI64Table* %table) {
entry:
  %vm = call %WamState* @plawk_dyncall_get()
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
  %pc = load i32, i32* @plawk_dyncall_pc
  %args = alloca %Value, i32 ~w
~w
  %r = call i1 @wam_object_call_assoc(%WamState* %vm, i32 %pc, i32 ~w, %Value* %args, i32 ~w, %WamAssocI64Table* %table)
  ret i1 %r

fail:
  ret i1 false
}',
        [NArgs, ParamsIR, NArgs, StoreIR, NArgs, NArgs]).

%% plawk_dyncall_named_assoc_str_shim_ir(+Name, +NArgs, -IR)
%  Str-valued table kind, named entry: same shape as the i64 assoc shim
%  but forwarding to @wam_object_call_assoc_str, which requires ATOM
%  values and stores their registry ids with replace semantics.
plawk_dyncall_named_assoc_str_shim_ir(Name, NArgs, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(IR),
'define i1 @plawk_dyncall_assoc_str_~w(~w, %WamAssocI64Table* %table) {
entry:
  %pc = call i32 @plawk_dyncall_resolve_~w()
  %bad = icmp slt i32 %pc, 0
  br i1 %bad, label %fail, label %do_call

do_call:
  %vm = call %WamState* @plawk_dyncall_get()
  %args = alloca %Value, i32 ~w
~w
  %r = call i1 @wam_object_call_assoc_str(%WamState* %vm, i32 %pc, i32 ~w, %Value* %args, i32 ~w, %WamAssocI64Table* %table)
  ret i1 %r

fail:
  ret i1 false
}',
        [Sym, ParamsIR, Sym, NArgs, StoreIR, NArgs, NArgs]).

%% plawk_dyncall_assoc_default_str_shim_ir(+NArgs, -IR)
%  Str-valued table kind, default entry: entry PC recorded at object-load
%  time, values forwarded to @wam_object_call_assoc_str.
plawk_dyncall_assoc_default_str_shim_ir(NArgs, IR) :-
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(IR),
'define i1 @plawk_dyncall_assoc_str_default_~w(~w, %WamAssocI64Table* %table) {
entry:
  %vm = call %WamState* @plawk_dyncall_get()
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
  %pc = load i32, i32* @plawk_dyncall_pc
  %args = alloca %Value, i32 ~w
~w
  %r = call i1 @wam_object_call_assoc_str(%WamState* %vm, i32 %pc, i32 ~w, %Value* %args, i32 ~w, %WamAssocI64Table* %table)
  ret i1 %r

fail:
  ret i1 false
}',
        [NArgs, ParamsIR, NArgs, StoreIR, NArgs, NArgs]).

%% plawk_dyncall_named_posarray_shim_ir(+Name, +NArgs, -IR)
%  Positional-array target, named entry: same shape as the i64 assoc
%  shim but forwarding to @wam_object_call_posarray, which walks a flat
%  returned [V1..Vn] list into keys 1..n (i64 values, replace semantics).
plawk_dyncall_named_posarray_shim_ir(Name, NArgs, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(IR),
'define i1 @plawk_dyncall_posarray_~w(~w, %WamAssocI64Table* %table) {
entry:
  %pc = call i32 @plawk_dyncall_resolve_~w()
  %bad = icmp slt i32 %pc, 0
  br i1 %bad, label %fail, label %do_call

do_call:
  %vm = call %WamState* @plawk_dyncall_get()
  %args = alloca %Value, i32 ~w
~w
  %r = call i1 @wam_object_call_posarray(%WamState* %vm, i32 %pc, i32 ~w, %Value* %args, i32 ~w, %WamAssocI64Table* %table)
  ret i1 %r

fail:
  ret i1 false
}',
        [Sym, ParamsIR, Sym, NArgs, StoreIR, NArgs, NArgs]).

%% plawk_dyncall_posarray_default_shim_ir(+NArgs, -IR)
%  Positional-array target, default entry: entry PC recorded at
%  object-load time, values forwarded to @wam_object_call_posarray.
plawk_dyncall_posarray_default_shim_ir(NArgs, IR) :-
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(IR),
'define i1 @plawk_dyncall_posarray_default_~w(~w, %WamAssocI64Table* %table) {
entry:
  %vm = call %WamState* @plawk_dyncall_get()
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
  %pc = load i32, i32* @plawk_dyncall_pc
  %args = alloca %Value, i32 ~w
~w
  %r = call i1 @wam_object_call_posarray(%WamState* %vm, i32 %pc, i32 ~w, %Value* %args, i32 ~w, %WamAssocI64Table* %table)
  ret i1 %r

fail:
  ret i1 false
}',
        [NArgs, ParamsIR, NArgs, StoreIR, NArgs, NArgs]).

%% plawk_dyncall_named_posarray_str_shim_ir(+Name, +NArgs, -IR)
%  Str-valued positional array, named entry: same shape as the posarray
%  shim but forwarding to @wam_object_call_posarray_str (atom elements).
plawk_dyncall_named_posarray_str_shim_ir(Name, NArgs, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(IR),
'define i1 @plawk_dyncall_posarray_str_~w(~w, %WamAssocI64Table* %table) {
entry:
  %pc = call i32 @plawk_dyncall_resolve_~w()
  %bad = icmp slt i32 %pc, 0
  br i1 %bad, label %fail, label %do_call

do_call:
  %vm = call %WamState* @plawk_dyncall_get()
  %args = alloca %Value, i32 ~w
~w
  %r = call i1 @wam_object_call_posarray_str(%WamState* %vm, i32 %pc, i32 ~w, %Value* %args, i32 ~w, %WamAssocI64Table* %table)
  ret i1 %r

fail:
  ret i1 false
}',
        [Sym, ParamsIR, Sym, NArgs, StoreIR, NArgs, NArgs]).

%% plawk_dyncall_posarray_str_default_shim_ir(+NArgs, -IR)
%  Str-valued positional array, default entry.
plawk_dyncall_posarray_str_default_shim_ir(NArgs, IR) :-
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(IR),
'define i1 @plawk_dyncall_posarray_str_default_~w(~w, %WamAssocI64Table* %table) {
entry:
  %vm = call %WamState* @plawk_dyncall_get()
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
  %pc = load i32, i32* @plawk_dyncall_pc
  %args = alloca %Value, i32 ~w
~w
  %r = call i1 @wam_object_call_posarray_str(%WamState* %vm, i32 %pc, i32 ~w, %Value* %args, i32 ~w, %WamAssocI64Table* %table)
  ret i1 %r

fail:
  ret i1 false
}',
        [NArgs, ParamsIR, NArgs, StoreIR, NArgs, NArgs]).

%% plawk_dyncall_rec_shim_ir(+NArgs, -IR)
%  Record shim for a default-entry destructure bind: resolve vm/pc from the
%  shared object handle, box the N call args, and forward the caller's
%  (nfields, typecodes, slots) to @wam_object_call_record. Returns i1 ok.
plawk_dyncall_rec_shim_ir(NArgs, IR) :-
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(IR),
'define i1 @plawk_dyncall_rec_~w(~w, i32 %nfields, i8* %tc, i64* %slots, i64* %lens) {
entry:
  %vm = call %WamState* @plawk_dyncall_get()
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
  %pc = load i32, i32* @plawk_dyncall_pc
  %args = alloca %Value, i32 ~w
~w
  %r = call i1 @wam_object_call_record(%WamState* %vm, i32 %pc, i32 ~w, %Value* %args, i32 ~w, i32 %nfields, i8* %tc, i64* %slots, i64* %lens)
  ret i1 %r

fail:
  ret i1 false
}',
        [NArgs, ParamsIR, NArgs, StoreIR, NArgs, NArgs]).

%% plawk_dyncall_named_rec_shim_ir(+Name, +NArgs, -IR)
%  Record shim for a named-entry destructure bind: resolve the PC via the
%  shared @plawk_dyncall_resolve_<Sym>, box args, forward to
%  @wam_object_call_record. Returns i1 ok ({slots} untouched on failure).
plawk_dyncall_named_rec_shim_ir(Name, NArgs, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(IR),
'define i1 @plawk_dyncall_named_rec_~w(~w, i32 %nfields, i8* %tc, i64* %slots, i64* %lens) {
entry:
  %pc = call i32 @plawk_dyncall_resolve_~w()
  %bad = icmp slt i32 %pc, 0
  br i1 %bad, label %fail, label %do_call

do_call:
  %vm = call %WamState* @plawk_dyncall_get()
  %args = alloca %Value, i32 ~w
~w
  %r = call i1 @wam_object_call_record(%WamState* %vm, i32 %pc, i32 ~w, %Value* %args, i32 ~w, i32 %nfields, i8* %tc, i64* %slots, i64* %lens)
  ret i1 %r

fail:
  ret i1 false
}',
        [Sym, ParamsIR, Sym, NArgs, StoreIR, NArgs, NArgs]).

%% plawk_dyncall_named_symbol(+Name, +NArgs, -Sym)
%  The LLVM symbol suffix for a named-entry shim: <Name>_<NArgs>. Name is
%  a plawk identifier (alnum + underscore), so it is a valid symbol part.
plawk_dyncall_named_symbol(Name, NArgs, Sym) :-
    format(atom(Sym), '~w_~w', [Name, NArgs]).

%% plawk_dyncall_named_resolver_ir(+Name, +NArgs, +PathBytesLen, -IR)
%  The shared per-entry resolver: the entry-name string "Name/(NArgs+1)"
%  (arity = inputs + output cell), a cached-PC global (sentinel -1 =
%  unresolved), and @plawk_dyncall_resolve_<Sym>() which resolves the name
%  to a label index via @wam_object_entry_index + @wam_label_pc once,
%  caches the PC, and returns it (or -1 on load / missing-entry failure).
%  The i64/double/byte shims all call this, so an entry used in more than
%  one return position resolves once and shares the cache.
plawk_dyncall_named_resolver_ir(Name, NArgs, PathBytesLen, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    EntryArity is NArgs + 1,
    format(atom(EntryName), '~w/~w', [Name, EntryArity]),
    format(atom(ENameGlobal), 'plawk_dyncall_ename_~w', [Sym]),
    llvm_emit_c_string_global(ENameGlobal, EntryName, ENameGlobalIR,
        ENameLen, ENameBytesLen),
    format(atom(IR),
'~w
@plawk_dyncall_pc_~w = internal global i32 -1

define i32 @plawk_dyncall_resolve_~w() {
entry:
  %vm = call %WamState* @plawk_dyncall_get()
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %check

check:
  %pc0 = load i32, i32* @plawk_dyncall_pc_~w
  %need = icmp slt i32 %pc0, 0
  br i1 %need, label %do_resolve, label %done

do_resolve:
  %namep = getelementptr [~w x i8], [~w x i8]* @.plawk_dyncall_ename_~w, i32 0, i32 0
  %pathp = getelementptr [~w x i8], [~w x i8]* @.plawk_dyncall_path, i32 0, i32 0
  %idx = call i32 @wam_object_entry_index(i8* %pathp, i8* %namep, i64 ~w)
  %bad = icmp slt i32 %idx, 0
  br i1 %bad, label %fail, label %store_pc

store_pc:
  %rpc = call i32 @wam_label_pc(%WamState* %vm, i32 %idx)
  store i32 %rpc, i32* @plawk_dyncall_pc_~w
  br label %done

done:
  %pc = phi i32 [ %pc0, %check ], [ %rpc, %store_pc ]
  ret i32 %pc

fail:
  ret i32 -1
}',
        [ENameGlobalIR, Sym, Sym, Sym,
         ENameBytesLen, ENameBytesLen, Sym, PathBytesLen, PathBytesLen,
         ENameLen, Sym]).

%% plawk_dyncall_named_shim_ir(+Name, +NArgs, -IR)
%  i64 named shim: resolve the PC (shared resolver), box args, call
%  @wam_object_call_i64. Yields { i64, i1 }; { 0, false } if the entry is
%  absent (resolver returns -1) or the object failed to load.
plawk_dyncall_named_shim_ir(Name, NArgs, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(IR),
'define { i64, i1 } @plawk_dyncall_named_~w(~w) {
entry:
  %pc = call i32 @plawk_dyncall_resolve_~w()
  %bad = icmp slt i32 %pc, 0
  br i1 %bad, label %fail, label %do_call

do_call:
  %vm = call %WamState* @plawk_dyncall_get()
  %args = alloca %Value, i32 ~w
~w
  %r = call { i64, i1 } @wam_object_call_i64(%WamState* %vm, i32 %pc, i32 ~w, %Value* %args, i32 ~w)
  ret { i64, i1 } %r

fail:
  %f0 = insertvalue { i64, i1 } undef, i64 0, 0
  %f1 = insertvalue { i64, i1 } %f0, i1 false, 1
  ret { i64, i1 } %f1
}',
        [Sym, ParamsIR, Sym, NArgs, StoreIR, NArgs, NArgs]).

%% plawk_dyncall_named_shim_f_ir(+Name, +NArgs, -IR)
%  double named shim (float(dyncall@name(...))): shared resolver, then
%  @wam_object_call_f64. Yields { double, i1 }.
plawk_dyncall_named_shim_f_ir(Name, NArgs, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(IR),
'define { double, i1 } @plawk_dyncall_named_f_~w(~w) {
entry:
  %pc = call i32 @plawk_dyncall_resolve_~w()
  %bad = icmp slt i32 %pc, 0
  br i1 %bad, label %fail, label %do_call

do_call:
  %vm = call %WamState* @plawk_dyncall_get()
  %args = alloca %Value, i32 ~w
~w
  %r = call { double, i1 } @wam_object_call_f64(%WamState* %vm, i32 %pc, i32 ~w, %Value* %args, i32 ~w)
  ret { double, i1 } %r

fail:
  %f0 = insertvalue { double, i1 } undef, double 0.0, 0
  %f1 = insertvalue { double, i1 } %f0, i1 false, 1
  ret { double, i1 } %f1
}',
        [Sym, ParamsIR, Sym, NArgs, StoreIR, NArgs, NArgs]).

%% plawk_dyncall_named_shim_b_ir(+Name, +NArgs, -IR)
%  byte-slice named shim (blob(dyncall@name(...))): shared resolver, then
%  @wam_object_call_bytes. Yields { i8*, i64, i1 }.
plawk_dyncall_named_shim_b_ir(Name, NArgs, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(IR),
'define { i8*, i64, i1 } @plawk_dyncall_named_b_~w(~w) {
entry:
  %pc = call i32 @plawk_dyncall_resolve_~w()
  %bad = icmp slt i32 %pc, 0
  br i1 %bad, label %fail, label %do_call

do_call:
  %vm = call %WamState* @plawk_dyncall_get()
  %args = alloca %Value, i32 ~w
~w
  %r = call { i8*, i64, i1 } @wam_object_call_bytes(%WamState* %vm, i32 %pc, i32 ~w, %Value* %args, i32 ~w)
  ret { i8*, i64, i1 } %r

fail:
  %f0 = insertvalue { i8*, i64, i1 } undef, i8* null, 0
  %f1 = insertvalue { i8*, i64, i1 } %f0, i64 0, 1
  %f2 = insertvalue { i8*, i64, i1 } %f1, i1 false, 2
  ret { i8*, i64, i1 } %f2
}',
        [Sym, ParamsIR, Sym, NArgs, StoreIR, NArgs, NArgs]).

% double-returning dyncall shim (float(dyncall(...))): same object handle,
% reads the entry output as a double via @wam_object_call_f64.
plawk_dyncall_shim_f_ir(NArgs, IR) :-
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(IR),
'define { double, i1 } @plawk_dyncall_f_~w(~w) {
entry:
  %vm = call %WamState* @plawk_dyncall_get()
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
  %pc = load i32, i32* @plawk_dyncall_pc
  %args = alloca %Value, i32 ~w
~w
  %r = call { double, i1 } @wam_object_call_f64(%WamState* %vm, i32 %pc, i32 ~w, %Value* %args, i32 ~w)
  ret { double, i1 } %r

fail:
  %f0 = insertvalue { double, i1 } undef, double 0.0, 0
  %f1 = insertvalue { double, i1 } %f0, i1 false, 1
  ret { double, i1 } %f1
}',
        [NArgs, ParamsIR, NArgs, StoreIR, NArgs, NArgs]).

% byte-slice dyncall shim (blob(dyncall(...))): reads the object entry's
% Atom output as { ptr, len, ok } via @wam_object_call_bytes.
plawk_dyncall_shim_b_ir(NArgs, IR) :-
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(IR),
'define { i8*, i64, i1 } @plawk_dyncall_b_~w(~w) {
entry:
  %vm = call %WamState* @plawk_dyncall_get()
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
  %pc = load i32, i32* @plawk_dyncall_pc
  %args = alloca %Value, i32 ~w
~w
  %r = call { i8*, i64, i1 } @wam_object_call_bytes(%WamState* %vm, i32 %pc, i32 ~w, %Value* %args, i32 ~w)
  ret { i8*, i64, i1 } %r

fail:
  %f0 = insertvalue { i8*, i64, i1 } undef, i8* null, 0
  %f1 = insertvalue { i8*, i64, i1 } %f0, i64 0, 1
  %f2 = insertvalue { i8*, i64, i1 } %f1, i1 false, 2
  ret { i8*, i64, i1 } %f2
}',
        [NArgs, ParamsIR, NArgs, StoreIR, NArgs, NArgs]).

plawk_dyncall_shim_ir(NArgs, IR) :-
    plawk_foreign_wrapper_params(NArgs, ParamsIR),
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(IR),
'define { i64, i1 } @plawk_dyncall_~w(~w) {
entry:
  %vm = call %WamState* @plawk_dyncall_get()
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
  %pc = load i32, i32* @plawk_dyncall_pc
  %args = alloca %Value, i32 ~w
~w
  %r = call { i64, i1 } @wam_object_call_i64(%WamState* %vm, i32 %pc, i32 ~w, %Value* %args, i32 ~w)
  ret { i64, i1 } %r

fail:
  %f0 = insertvalue { i64, i1 } undef, i64 0, 0
  %f1 = insertvalue { i64, i1 } %f0, i1 false, 1
  ret { i64, i1 } %f1
}',
        [NArgs, ParamsIR, NArgs, StoreIR, NArgs, NArgs]).

plawk_dyncall_store_lines(NArgs, Lines) :-
    NArgs1 is NArgs - 1,
    numlist(0, NArgs1, Ns),
    findall(Line,
        ( member(I, Ns),
          format(atom(Line),
              '  %args_~w = getelementptr %Value, %Value* %args, i32 ~w\n  store %Value %a~w, %Value* %args_~w',
              [I, I, I, I])
        ),
        Lines).

%% plawk_dyncall_at_support_ir(+Mode, +Arities, -IR)
%
%  Emit the dynamic-source runtime for dyncall_at(...). Mode is a
%  compile-time string from BEGIN { DYNCACHE = "..." }:
%    "on"    (default) -- a fixed-capacity cache keyed by the interned
%                         path id; each distinct grammar loads once and is
%                         reused. Constant memory per call (arena rewind in
%                         @wam_object_call_i64).
%    "mtime"           -- like "on" but the cache also keys on the file's
%                         st_mtim.tv_sec (offset 88, matching time_file/2),
%                         so recompiling the .wamo busts the entry (frees
%                         the stale VM) and reloads -- for query/userspace
%                         redefinition without reload-per-call.
%    "off"             -- load fresh and @wam_state_free after every call;
%                         always current, no cache, pays full load per call.
%  Cache capacity is 64 distinct grammars; beyond that "on"/"mtime" load
%  without caching (correct, just not reused).
plawk_dyncall_at_support_ir(Mode, Arities, IR) :-
    plawk_dyncall_at_support_ir(Mode, Arities, [], [], [], [], [], IR).
plawk_dyncall_at_support_ir(Mode, IArities, FArities, IR) :-
    plawk_dyncall_at_support_ir(Mode, IArities, FArities, [], [], [], [], IR).
plawk_dyncall_at_support_ir(Mode, IArities, FArities, BArities, IR) :-
    plawk_dyncall_at_support_ir(Mode, IArities, FArities, BArities,
        [], [], [], IR).
plawk_dyncall_at_support_ir(Mode, IArities, FArities, BArities,
        NamedEntries, IR) :-
    plawk_dyncall_at_support_ir(Mode, IArities, FArities, BArities,
        NamedEntries, [], [], IR).

%% plawk_dyncall_at_support_ir(+Mode, +IArities, +FArities, +BArities,
%%                             +NamedI, +NamedF, +NamedB, -IR)
%  IArities -> i64 @plawk_dyncall_at_N shims; FArities -> double
%  @plawk_dyncall_at_f_N shims (float(dyncall_at(...))); BArities ->
%  byte-slice @plawk_dyncall_at_b_N shims (blob(dyncall_at(...))).
%  NamedI/NamedF/NamedB (Name-NArgs) -> @plawk_dyncall_at_named[_f/_b]
%  shims for dyncall_at@name sites in each return kind, resolving the
%  entry per call against the loaded VM's entry table. All share the
%  cache + @plawk_dyncall_at_get emitted per Mode.
plawk_dyncall_at_support_ir(Mode, IArities, FArities, BArities,
        NamedI, NamedF, NamedB, IR) :-
    plawk_dyncall_at_support_ir(Mode, IArities, FArities, BArities,
        NamedI, NamedF, NamedB, [], [], IR).
plawk_dyncall_at_support_ir(Mode, IArities, FArities, BArities,
        NamedI, NamedF, NamedB, RecArities, NamedRec, IR) :-
    (   Mode == "off"
    ->  Globals = '', GetIR = '',
        ICShim = plawk_dyncall_at_shim_off_ir,
        FCShim = plawk_dyncall_at_shim_off_f_ir,
        BCShim = plawk_dyncall_at_shim_off_b_ir,
        NShim = plawk_dyncall_at_named_shim_off_ir,
        NFShim = plawk_dyncall_at_named_shim_off_f_ir,
        NBShim = plawk_dyncall_at_named_shim_off_b_ir,
        RCShim = plawk_dyncall_at_rec_shim_off_ir,
        NRecShim = plawk_dyncall_at_named_rec_shim_off_ir
    ;   Mode == "mtime"
    ->  plawk_dyncache_globals_ir(mtime, Globals),
        plawk_dyncall_at_get_mtime_ir(GetIR),
        ICShim = plawk_dyncall_at_shim_cached_ir,
        FCShim = plawk_dyncall_at_shim_cached_f_ir,
        BCShim = plawk_dyncall_at_shim_cached_b_ir,
        NShim = plawk_dyncall_at_named_shim_cached_ir,
        NFShim = plawk_dyncall_at_named_shim_cached_f_ir,
        NBShim = plawk_dyncall_at_named_shim_cached_b_ir,
        RCShim = plawk_dyncall_at_rec_shim_cached_ir,
        NRecShim = plawk_dyncall_at_named_rec_shim_cached_ir
    ;   plawk_dyncache_globals_ir(plain, Globals),
        plawk_dyncall_at_get_on_ir(GetIR),
        ICShim = plawk_dyncall_at_shim_cached_ir,
        FCShim = plawk_dyncall_at_shim_cached_f_ir,
        BCShim = plawk_dyncall_at_shim_cached_b_ir,
        NShim = plawk_dyncall_at_named_shim_cached_ir,
        NFShim = plawk_dyncall_at_named_shim_cached_f_ir,
        NBShim = plawk_dyncall_at_named_shim_cached_b_ir,
        RCShim = plawk_dyncall_at_rec_shim_cached_ir,
        NRecShim = plawk_dyncall_at_named_rec_shim_cached_ir
    ),
    findall(S,  ( member(N, IArities),  call(ICShim, N, S) ), IShims),
    findall(FS, ( member(FN, FArities), call(FCShim, FN, FS) ), FShims),
    findall(BS, ( member(BN, BArities), call(BCShim, BN, BS) ), BShims),
    findall(NS, ( member(NName-NN, NamedI), call(NShim, NName, NN, NS) ),
        NShims),
    findall(NFS, ( member(NFName-NFN, NamedF), call(NFShim, NFName, NFN, NFS) ),
        NFShims),
    findall(NBS, ( member(NBName-NBN, NamedB), call(NBShim, NBName, NBN, NBS) ),
        NBShims),
    findall(RS, ( member(RN, RecArities), call(RCShim, RN, RS) ), RecShims),
    findall(NRS, ( member(NRName-NRN, NamedRec),
        call(NRecShim, NRName, NRN, NRS) ), NRecShims),
    append([[Globals, GetIR], IShims, FShims, BShims, NShims, NFShims,
        NBShims, RecShims, NRecShims], Parts0),
    exclude(==(''), Parts0, Parts),
    atomic_list_concat(Parts, '\n\n', IR).

plawk_dyncache_globals_ir(Kind, IR) :-
    ( Kind == mtime
    -> MtimesLine = '@plawk_dyncache_mtimes = internal global [64 x i64] zeroinitializer\n'
    ;  MtimesLine = '' ),
    format(atom(IR),
'@plawk_dyncache_ids = internal global [64 x i64] zeroinitializer
@plawk_dyncache_vms = internal global [64 x %WamState*] zeroinitializer
@plawk_dyncache_pcs = internal global [64 x i32] zeroinitializer
~w@plawk_dyncache_n = internal global i32 0', [MtimesLine]).

% path-id-keyed cache lookup (mode "on")
plawk_dyncall_at_get_on_ir(
'define { %WamState*, i32 } @plawk_dyncall_at_get(i8* %path, i64 %len) {
entry:
  ; compile(...) handles travel as (null, handle-id): a null path is the
  ; discriminator -- resolve the 1-based cache index directly, with no
  ; interning and no filesystem. See @plawk_compile.
  %is_h = icmp eq i8* %path, null
  br i1 %is_h, label %hentry, label %pentry
hentry:
  %hid = trunc i64 %len to i32
  %hn = load i32, i32* @plawk_dyncache_n
  %h_lo = icmp sge i32 %hid, 1
  %h_hi = icmp sle i32 %hid, %hn
  %h_ok = and i1 %h_lo, %h_hi
  br i1 %h_ok, label %hload, label %hfail
hload:
  %hix = sub i32 %hid, 1
  %hvmp = getelementptr [64 x %WamState*], [64 x %WamState*]* @plawk_dyncache_vms, i32 0, i32 %hix
  %hvm = load %WamState*, %WamState** %hvmp
  %hpcp = getelementptr [64 x i32], [64 x i32]* @plawk_dyncache_pcs, i32 0, i32 %hix
  %hpc = load i32, i32* %hpcp
  %hh0 = insertvalue { %WamState*, i32 } undef, %WamState* %hvm, 0
  %hh1 = insertvalue { %WamState*, i32 } %hh0, i32 %hpc, 1
  ret { %WamState*, i32 } %hh1
hfail:
  %hf0 = insertvalue { %WamState*, i32 } undef, %WamState* null, 0
  %hf1 = insertvalue { %WamState*, i32 } %hf0, i32 0, 1
  ret { %WamState*, i32 } %hf1
pentry:
  %id = call i64 @wam_intern_atom(i8* %path, i64 %len)
  %n = load i32, i32* @plawk_dyncache_n
  br label %scan
scan:
  %i = phi i32 [ 0, %pentry ], [ %i1, %next ]
  %done = icmp sge i32 %i, %n
  br i1 %done, label %miss, label %check
check:
  %idp = getelementptr [64 x i64], [64 x i64]* @plawk_dyncache_ids, i32 0, i32 %i
  %cid = load i64, i64* %idp
  %match = icmp eq i64 %cid, %id
  br i1 %match, label %hit, label %next
next:
  %i1 = add i32 %i, 1
  br label %scan
hit:
  %vmp = getelementptr [64 x %WamState*], [64 x %WamState*]* @plawk_dyncache_vms, i32 0, i32 %i
  %vm = load %WamState*, %WamState** %vmp
  %pcp = getelementptr [64 x i32], [64 x i32]* @plawk_dyncache_pcs, i32 0, i32 %i
  %pc = load i32, i32* %pcp
  %h0 = insertvalue { %WamState*, i32 } undef, %WamState* %vm, 0
  %h1 = insertvalue { %WamState*, i32 } %h0, i32 %pc, 1
  ret { %WamState*, i32 } %h1
miss:
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)
  %nvm = extractvalue { %WamState*, i32 } %obj, 0
  %vm_ok = icmp ne %WamState* %nvm, null
  %has_room = icmp slt i32 %n, 64
  %can_cache = and i1 %vm_ok, %has_room
  br i1 %can_cache, label %store, label %ret_obj
store:
  %npc = extractvalue { %WamState*, i32 } %obj, 1
  %sidp = getelementptr [64 x i64], [64 x i64]* @plawk_dyncache_ids, i32 0, i32 %n
  store i64 %id, i64* %sidp
  %svmp = getelementptr [64 x %WamState*], [64 x %WamState*]* @plawk_dyncache_vms, i32 0, i32 %n
  store %WamState* %nvm, %WamState** %svmp
  %spcp = getelementptr [64 x i32], [64 x i32]* @plawk_dyncache_pcs, i32 0, i32 %n
  store i32 %npc, i32* %spcp
  %n1 = add i32 %n, 1
  store i32 %n1, i32* @plawk_dyncache_n
  br label %ret_obj
ret_obj:
  ret { %WamState*, i32 } %obj
}').

% path-id + st_mtime keyed cache lookup (mode "mtime")
plawk_dyncall_at_get_mtime_ir(
'define { %WamState*, i32 } @plawk_dyncall_at_get(i8* %path, i64 %len) {
entry:
  ; compile(...) handles travel as (null, handle-id) -- resolved from
  ; the cache index directly, never stat-ed (a handle has no file, so
  ; the mtime-bust path does not apply). See @plawk_compile.
  %is_h = icmp eq i8* %path, null
  br i1 %is_h, label %hentry, label %pentry
hentry:
  %hid = trunc i64 %len to i32
  %hn = load i32, i32* @plawk_dyncache_n
  %h_lo = icmp sge i32 %hid, 1
  %h_hi = icmp sle i32 %hid, %hn
  %h_ok = and i1 %h_lo, %h_hi
  br i1 %h_ok, label %hload, label %hfail
hload:
  %hix = sub i32 %hid, 1
  %hvmp = getelementptr [64 x %WamState*], [64 x %WamState*]* @plawk_dyncache_vms, i32 0, i32 %hix
  %hvm = load %WamState*, %WamState** %hvmp
  %hpcp = getelementptr [64 x i32], [64 x i32]* @plawk_dyncache_pcs, i32 0, i32 %hix
  %hpc = load i32, i32* %hpcp
  %hh0 = insertvalue { %WamState*, i32 } undef, %WamState* %hvm, 0
  %hh1 = insertvalue { %WamState*, i32 } %hh0, i32 %hpc, 1
  ret { %WamState*, i32 } %hh1
hfail:
  %hf0 = insertvalue { %WamState*, i32 } undef, %WamState* null, 0
  %hf1 = insertvalue { %WamState*, i32 } %hf0, i32 0, 1
  ret { %WamState*, i32 } %hf1
pentry:
  %id = call i64 @wam_intern_atom(i8* %path, i64 %len)
  %statbuf = alloca [256 x i8]
  %sbp = getelementptr [256 x i8], [256 x i8]* %statbuf, i32 0, i32 0
  %sret = call i32 @stat(i8* %path, i8* %sbp)
  %stat_ok = icmp eq i32 %sret, 0
  br i1 %stat_ok, label %read_mtime, label %have_mtime
read_mtime:
  %secp_i8 = getelementptr i8, i8* %sbp, i64 88
  %secp = bitcast i8* %secp_i8 to i64*
  %sec = load i64, i64* %secp
  %nsp_i8 = getelementptr i8, i8* %sbp, i64 96
  %nsp = bitcast i8* %nsp_i8 to i64*
  %nsec = load i64, i64* %nsp
  %sec_ns = mul i64 %sec, 1000000000
  %mtime_r = add i64 %sec_ns, %nsec
  br label %have_mtime
have_mtime:
  %mtime = phi i64 [ %mtime_r, %read_mtime ], [ 0, %pentry ]
  %n = load i32, i32* @plawk_dyncache_n
  br label %scan
scan:
  %i = phi i32 [ 0, %have_mtime ], [ %i1, %next ]
  %done = icmp sge i32 %i, %n
  br i1 %done, label %miss, label %check
check:
  %idp = getelementptr [64 x i64], [64 x i64]* @plawk_dyncache_ids, i32 0, i32 %i
  %cid = load i64, i64* %idp
  %idmatch = icmp eq i64 %cid, %id
  br i1 %idmatch, label %check_mtime, label %next
check_mtime:
  %mtp = getelementptr [64 x i64], [64 x i64]* @plawk_dyncache_mtimes, i32 0, i32 %i
  %cmt = load i64, i64* %mtp
  %mtmatch = icmp eq i64 %cmt, %mtime
  br i1 %mtmatch, label %hit, label %stale
next:
  %i1 = add i32 %i, 1
  br label %scan
hit:
  %vmp = getelementptr [64 x %WamState*], [64 x %WamState*]* @plawk_dyncache_vms, i32 0, i32 %i
  %vm = load %WamState*, %WamState** %vmp
  %pcp = getelementptr [64 x i32], [64 x i32]* @plawk_dyncache_pcs, i32 0, i32 %i
  %pc = load i32, i32* %pcp
  %h0 = insertvalue { %WamState*, i32 } undef, %WamState* %vm, 0
  %h1 = insertvalue { %WamState*, i32 } %h0, i32 %pc, 1
  ret { %WamState*, i32 } %h1
stale:
  %ovmp = getelementptr [64 x %WamState*], [64 x %WamState*]* @plawk_dyncache_vms, i32 0, i32 %i
  %ovm = load %WamState*, %WamState** %ovmp
  %ovm_null = icmp eq %WamState* %ovm, null
  br i1 %ovm_null, label %reload, label %free_old
free_old:
  call void @wam_state_free(%WamState* %ovm)
  br label %reload
reload:
  %robj = call { %WamState*, i32 } @wam_object_load(i8* %path)
  %rvm = extractvalue { %WamState*, i32 } %robj, 0
  %rpc = extractvalue { %WamState*, i32 } %robj, 1
  store %WamState* %rvm, %WamState** %ovmp
  %rpcp = getelementptr [64 x i32], [64 x i32]* @plawk_dyncache_pcs, i32 0, i32 %i
  store i32 %rpc, i32* %rpcp
  store i64 %mtime, i64* %mtp
  ret { %WamState*, i32 } %robj
miss:
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)
  %nvm = extractvalue { %WamState*, i32 } %obj, 0
  %vm_ok = icmp ne %WamState* %nvm, null
  %has_room = icmp slt i32 %n, 64
  %can_cache = and i1 %vm_ok, %has_room
  br i1 %can_cache, label %mstore, label %mret
mstore:
  %mnpc = extractvalue { %WamState*, i32 } %obj, 1
  %msidp = getelementptr [64 x i64], [64 x i64]* @plawk_dyncache_ids, i32 0, i32 %n
  store i64 %id, i64* %msidp
  %msmtp = getelementptr [64 x i64], [64 x i64]* @plawk_dyncache_mtimes, i32 0, i32 %n
  store i64 %mtime, i64* %msmtp
  %msvmp = getelementptr [64 x %WamState*], [64 x %WamState*]* @plawk_dyncache_vms, i32 0, i32 %n
  store %WamState* %nvm, %WamState** %msvmp
  %mspcp = getelementptr [64 x i32], [64 x i32]* @plawk_dyncache_pcs, i32 0, i32 %n
  store i32 %mnpc, i32* %mspcp
  %mn1 = add i32 %n, 1
  store i32 %mn1, i32* @plawk_dyncache_n
  br label %mret
mret:
  ret { %WamState*, i32 } %obj
}').

%% plawk_compile_support_ir(+EvalcPath, -IR)
%
%  The eval-surface runtime (JIT roadmap item 5 payoff):
%  @plawk_compile(src, len) compiles Prolog source text through the
%  shipped bootstrap-compiler object and returns a HANDLE (1-based
%  index into the dyncall_at cache registry; 0 on failure). Flow:
%    1. intern the source text; scan the cache for that id -- same
%       source compiles ONCE, later calls are a registry hit;
%    2. miss: lazy-load the compiler object (@wam_object_load_cached,
%       its own one-shot cache -- the DYNCACHE role), run it on the
%       source via @wam_object_eval (compiler entry cgfull(Src, Wamo);
%       the emitted .wamo bytes load into a fresh VM), and record the
%       loaded grammar in the registry under the source id.
%  The handle is consumed by @plawk_dyncall_at_get as (null, handle).
%  Requires the dyncache globals (any compile(...) site lives inside a
%  dyncall_at, so they are always emitted together) and cache mode
%  on/mtime -- mode "off" has no registry and is rejected at codegen.
plawk_compile_support_ir(EvalcPath, IR) :-
    llvm_emit_c_string_global(plawk_evalc_path, EvalcPath, PathGlobal,
        _StrLen, BytesLen),
    format(atom(IR),
'~w

define i64 @plawk_compile(i8* %src, i64 %len) {
entry:
  %id = call i64 @wam_intern_atom(i8* %src, i64 %len)
  %n = load i32, i32* @plawk_dyncache_n
  br label %scan
scan:
  %i = phi i32 [ 0, %entry ], [ %i1, %next ]
  %done = icmp sge i32 %i, %n
  br i1 %done, label %miss, label %check
check:
  %idp = getelementptr [64 x i64], [64 x i64]* @plawk_dyncache_ids, i32 0, i32 %i
  %cid = load i64, i64* %idp
  %match = icmp eq i64 %cid, %id
  br i1 %match, label %hit, label %next
next:
  %i1 = add i32 %i, 1
  br label %scan
hit:
  %h = add i32 %i, 1
  %h64 = sext i32 %h to i64
  ret i64 %h64
miss:
  ; a full registry cannot hand out a stable handle -- fail rather
  ; than hand back an index that later loads would shift under
  %room = icmp slt i32 %n, 64
  br i1 %room, label %doload, label %fail
doload:
  %cpath = getelementptr [~w x i8], [~w x i8]* @.plawk_evalc_path, i32 0, i32 0
  %comp = call { %WamState*, i32 } @wam_object_load_cached(i8* %cpath)
  %cvm = extractvalue { %WamState*, i32 } %comp, 0
  %cvm_ok = icmp ne %WamState* %cvm, null
  br i1 %cvm_ok, label %doeval, label %fail
doeval:
  %cpc = extractvalue { %WamState*, i32 } %comp, 1
  %g = call { %WamState*, i32 } @wam_object_eval(%WamState* %cvm, i32 %cpc, i8* %src, i64 %len)
  %gvm = extractvalue { %WamState*, i32 } %g, 0
  %gok = icmp ne %WamState* %gvm, null
  br i1 %gok, label %record, label %fail
record:
  %gpc = extractvalue { %WamState*, i32 } %g, 1
  %sidp = getelementptr [64 x i64], [64 x i64]* @plawk_dyncache_ids, i32 0, i32 %n
  store i64 %id, i64* %sidp
  %svmp = getelementptr [64 x %WamState*], [64 x %WamState*]* @plawk_dyncache_vms, i32 0, i32 %n
  store %WamState* %gvm, %WamState** %svmp
  %spcp = getelementptr [64 x i32], [64 x i32]* @plawk_dyncache_pcs, i32 0, i32 %n
  store i32 %gpc, i32* %spcp
  %n1 = add i32 %n, 1
  store i32 %n1, i32* @plawk_dyncache_n
  %h64m = sext i32 %n1 to i64
  ret i64 %h64m
fail:
  ret i64 0
}

define i64 @plawk_compile_file(i8* %path) {
entry:
  ; compile_file(path): read the grammar SOURCE file and compile its
  ; contents through @plawk_compile. Content dedup makes this
  ; change-rebust with no mtime bookkeeping: an edited file is new
  ; source text (fresh compile, fresh handle); unchanged bytes hit the
  ; registry. The read is per call -- cheap next to a compile, and it
  ; keeps mid-run edits visible on the next record.
  %totp = alloca i64
  %buf = call i8* @wamo_read_file(i8* %path, i64* %totp)
  %buf_null = icmp eq i8* %buf, null
  br i1 %buf_null, label %fail, label %compile
compile:
  %total = load i64, i64* %totp
  %h = call i64 @plawk_compile(i8* %buf, i64 %total)
  call void @free(i8* %buf)
  ret i64 %h
fail:
  ret i64 0
}',
        [PathGlobal, BytesLen, BytesLen]).

% shim for cached modes (on / mtime): resolve via @plawk_dyncall_at_get
plawk_dyncall_at_shim_cached_ir(NArgs, IR) :-
    plawk_dyncall_at_params(NArgs, ParamsIR),
    plawk_dyncall_at_argsetup(NArgs, ArgsPtrIR, ArgsSetupIR),
    format(atom(IR),
'define { i64, i1 } @plawk_dyncall_at_~w(~w) {
entry:
  %obj = call { %WamState*, i32 } @plawk_dyncall_at_get(i8* %path, i64 %len)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %pc = extractvalue { %WamState*, i32 } %obj, 1
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
~w
  %r = call { i64, i1 } @wam_object_call_i64(%WamState* %vm, i32 %pc, i32 ~w, %Value* ~w, i32 ~w)
  ret { i64, i1 } %r

fail:
  %f0 = insertvalue { i64, i1 } undef, i64 0, 0
  %f1 = insertvalue { i64, i1 } %f0, i1 false, 1
  ret { i64, i1 } %f1
}',
        [NArgs, ParamsIR, ArgsSetupIR, NArgs, ArgsPtrIR, NArgs]).

% shim for "off" mode: load fresh, run, free every call
plawk_dyncall_at_shim_off_ir(NArgs, IR) :-
    plawk_dyncall_at_params(NArgs, ParamsIR),
    plawk_dyncall_at_argsetup(NArgs, ArgsPtrIR, ArgsSetupIR),
    format(atom(IR),
'define { i64, i1 } @plawk_dyncall_at_~w(~w) {
entry:
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %pc = extractvalue { %WamState*, i32 } %obj, 1
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
~w
  %r = call { i64, i1 } @wam_object_call_i64(%WamState* %vm, i32 %pc, i32 ~w, %Value* ~w, i32 ~w)
  call void @wam_state_free(%WamState* %vm)
  ret { i64, i1 } %r

fail:
  %f0 = insertvalue { i64, i1 } undef, i64 0, 0
  %f1 = insertvalue { i64, i1 } %f0, i1 false, 1
  ret { i64, i1 } %f1
}',
        [NArgs, ParamsIR, ArgsSetupIR, NArgs, ArgsPtrIR, NArgs]).

%% plawk_dyncall_at_named_shim_cached_ir(+Name, +NArgs, -IR)
%  Named-entry dyncall_at shim (modes on/mtime): fetch the loaded VM from
%  the source cache, then resolve the entry name "Name/(NArgs+1)" against
%  the VM's own materialized entry table (@wam_object_vm_entry_pc) -- per
%  call, not startup-cached, because the object is runtime data (a path
%  expression or a compile() handle) and can differ between records. The
%  in-memory scan is a few memcmps; caching a PC by VM pointer would go
%  stale across an mtime-mode reload at the same address.
plawk_dyncall_at_named_shim_cached_ir(Name, NArgs, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    EntryArity is NArgs + 1,
    format(atom(EntryName), '~w/~w', [Name, EntryArity]),
    format(atom(ENameGlobal), 'plawk_at_ename_~w', [Sym]),
    llvm_emit_c_string_global(ENameGlobal, EntryName, ENameGlobalIR,
        ENameLen, ENameBytesLen),
    plawk_dyncall_at_params(NArgs, ParamsIR),
    plawk_dyncall_at_argsetup(NArgs, ArgsPtrIR, ArgsSetupIR),
    format(atom(IR),
'~w

define { i64, i1 } @plawk_dyncall_at_named_~w(~w) {
entry:
  %obj = call { %WamState*, i32 } @plawk_dyncall_at_get(i8* %path, i64 %len)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %resolve

resolve:
  %namep = getelementptr [~w x i8], [~w x i8]* @.plawk_at_ename_~w, i32 0, i32 0
  %pc = call i32 @wam_object_vm_entry_pc(%WamState* %vm, i8* %namep, i64 ~w)
  %bad = icmp slt i32 %pc, 0
  br i1 %bad, label %fail, label %do_call

do_call:
~w
  %r = call { i64, i1 } @wam_object_call_i64(%WamState* %vm, i32 %pc, i32 ~w, %Value* ~w, i32 ~w)
  ret { i64, i1 } %r

fail:
  %f0 = insertvalue { i64, i1 } undef, i64 0, 0
  %f1 = insertvalue { i64, i1 } %f0, i1 false, 1
  ret { i64, i1 } %f1
}',
        [ENameGlobalIR, Sym, ParamsIR, ENameBytesLen, ENameBytesLen, Sym,
         ENameLen, ArgsSetupIR, NArgs, ArgsPtrIR, NArgs]).

%% plawk_dyncall_at_named_shim_off_ir(+Name, +NArgs, -IR)
%  Mode "off" named variant: load fresh, resolve, call, free -- including
%  on the resolve-miss path, so a name absent from the object does not
%  leak the freshly loaded VM.
plawk_dyncall_at_named_shim_off_ir(Name, NArgs, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    EntryArity is NArgs + 1,
    format(atom(EntryName), '~w/~w', [Name, EntryArity]),
    format(atom(ENameGlobal), 'plawk_at_ename_~w', [Sym]),
    llvm_emit_c_string_global(ENameGlobal, EntryName, ENameGlobalIR,
        ENameLen, ENameBytesLen),
    plawk_dyncall_at_params(NArgs, ParamsIR),
    plawk_dyncall_at_argsetup(NArgs, ArgsPtrIR, ArgsSetupIR),
    format(atom(IR),
'~w

define { i64, i1 } @plawk_dyncall_at_named_~w(~w) {
entry:
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %resolve

resolve:
  %namep = getelementptr [~w x i8], [~w x i8]* @.plawk_at_ename_~w, i32 0, i32 0
  %pc = call i32 @wam_object_vm_entry_pc(%WamState* %vm, i8* %namep, i64 ~w)
  %bad = icmp slt i32 %pc, 0
  br i1 %bad, label %free_fail, label %do_call

do_call:
~w
  %r = call { i64, i1 } @wam_object_call_i64(%WamState* %vm, i32 %pc, i32 ~w, %Value* ~w, i32 ~w)
  call void @wam_state_free(%WamState* %vm)
  ret { i64, i1 } %r

free_fail:
  call void @wam_state_free(%WamState* %vm)
  br label %fail

fail:
  %f0 = insertvalue { i64, i1 } undef, i64 0, 0
  %f1 = insertvalue { i64, i1 } %f0, i1 false, 1
  ret { i64, i1 } %f1
}',
        [ENameGlobalIR, Sym, ParamsIR, ENameBytesLen, ENameBytesLen, Sym,
         ENameLen, ArgsSetupIR, NArgs, ArgsPtrIR, NArgs]).

% double-returning dyncall_at shims (float(dyncall_at(...))): same cache /
% load, read the entry output as a double via @wam_object_call_f64.
plawk_dyncall_at_shim_cached_f_ir(NArgs, IR) :-
    plawk_dyncall_at_params(NArgs, ParamsIR),
    plawk_dyncall_at_argsetup(NArgs, ArgsPtrIR, ArgsSetupIR),
    format(atom(IR),
'define { double, i1 } @plawk_dyncall_at_f_~w(~w) {
entry:
  %obj = call { %WamState*, i32 } @plawk_dyncall_at_get(i8* %path, i64 %len)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %pc = extractvalue { %WamState*, i32 } %obj, 1
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
~w
  %r = call { double, i1 } @wam_object_call_f64(%WamState* %vm, i32 %pc, i32 ~w, %Value* ~w, i32 ~w)
  ret { double, i1 } %r

fail:
  %f0 = insertvalue { double, i1 } undef, double 0.0, 0
  %f1 = insertvalue { double, i1 } %f0, i1 false, 1
  ret { double, i1 } %f1
}',
        [NArgs, ParamsIR, ArgsSetupIR, NArgs, ArgsPtrIR, NArgs]).

plawk_dyncall_at_shim_off_f_ir(NArgs, IR) :-
    plawk_dyncall_at_params(NArgs, ParamsIR),
    plawk_dyncall_at_argsetup(NArgs, ArgsPtrIR, ArgsSetupIR),
    format(atom(IR),
'define { double, i1 } @plawk_dyncall_at_f_~w(~w) {
entry:
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %pc = extractvalue { %WamState*, i32 } %obj, 1
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
~w
  %r = call { double, i1 } @wam_object_call_f64(%WamState* %vm, i32 %pc, i32 ~w, %Value* ~w, i32 ~w)
  call void @wam_state_free(%WamState* %vm)
  ret { double, i1 } %r

fail:
  %f0 = insertvalue { double, i1 } undef, double 0.0, 0
  %f1 = insertvalue { double, i1 } %f0, i1 false, 1
  ret { double, i1 } %f1
}',
        [NArgs, ParamsIR, ArgsSetupIR, NArgs, ArgsPtrIR, NArgs]).

% byte-slice dyncall_at shims (blob(dyncall_at(...))): cached / off-mode
% loads, reading the entry output as { ptr, len, ok }.
plawk_dyncall_at_shim_cached_b_ir(NArgs, IR) :-
    plawk_dyncall_at_params(NArgs, ParamsIR),
    plawk_dyncall_at_argsetup(NArgs, ArgsPtrIR, ArgsSetupIR),
    format(atom(IR),
'define { i8*, i64, i1 } @plawk_dyncall_at_b_~w(~w) {
entry:
  %obj = call { %WamState*, i32 } @plawk_dyncall_at_get(i8* %path, i64 %len)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %pc = extractvalue { %WamState*, i32 } %obj, 1
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
~w
  %r = call { i8*, i64, i1 } @wam_object_call_bytes(%WamState* %vm, i32 %pc, i32 ~w, %Value* ~w, i32 ~w)
  ret { i8*, i64, i1 } %r

fail:
  %f0 = insertvalue { i8*, i64, i1 } undef, i8* null, 0
  %f1 = insertvalue { i8*, i64, i1 } %f0, i64 0, 1
  %f2 = insertvalue { i8*, i64, i1 } %f1, i1 false, 2
  ret { i8*, i64, i1 } %f2
}',
        [NArgs, ParamsIR, ArgsSetupIR, NArgs, ArgsPtrIR, NArgs]).

plawk_dyncall_at_shim_off_b_ir(NArgs, IR) :-
    plawk_dyncall_at_params(NArgs, ParamsIR),
    plawk_dyncall_at_argsetup(NArgs, ArgsPtrIR, ArgsSetupIR),
    format(atom(IR),
'define { i8*, i64, i1 } @plawk_dyncall_at_b_~w(~w) {
entry:
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %pc = extractvalue { %WamState*, i32 } %obj, 1
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
~w
  %r = call { i8*, i64, i1 } @wam_object_call_bytes(%WamState* %vm, i32 %pc, i32 ~w, %Value* ~w, i32 ~w)
  call void @wam_state_free(%WamState* %vm)
  ret { i8*, i64, i1 } %r

fail:
  %f0 = insertvalue { i8*, i64, i1 } undef, i8* null, 0
  %f1 = insertvalue { i8*, i64, i1 } %f0, i64 0, 1
  %f2 = insertvalue { i8*, i64, i1 } %f1, i1 false, 2
  ret { i8*, i64, i1 } %f2
}',
        [NArgs, ParamsIR, ArgsSetupIR, NArgs, ArgsPtrIR, NArgs]).

% double / byte-slice named-at shims (float(dyncall_at@name(...)) /
% blob(dyncall_at@name(...))): the i64 named-at shape with the other
% call primitives -- fetch the VM per the mode, resolve the entry name
% against its materialized entry table per call, then read the output
% as a double or byte slice.
plawk_dyncall_at_named_shim_cached_f_ir(Name, NArgs, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    EntryArity is NArgs + 1,
    format(atom(EntryName), '~w/~w', [Name, EntryArity]),
    format(atom(ENameGlobal), 'plawk_at_ename_f_~w', [Sym]),
    llvm_emit_c_string_global(ENameGlobal, EntryName, ENameGlobalIR,
        ENameLen, ENameBytesLen),
    plawk_dyncall_at_params(NArgs, ParamsIR),
    plawk_dyncall_at_argsetup(NArgs, ArgsPtrIR, ArgsSetupIR),
    format(atom(IR),
'~w

define { double, i1 } @plawk_dyncall_at_named_f_~w(~w) {
entry:
  %obj = call { %WamState*, i32 } @plawk_dyncall_at_get(i8* %path, i64 %len)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %resolve

resolve:
  %namep = getelementptr [~w x i8], [~w x i8]* @.plawk_at_ename_f_~w, i32 0, i32 0
  %pc = call i32 @wam_object_vm_entry_pc(%WamState* %vm, i8* %namep, i64 ~w)
  %bad = icmp slt i32 %pc, 0
  br i1 %bad, label %fail, label %do_call

do_call:
~w
  %r = call { double, i1 } @wam_object_call_f64(%WamState* %vm, i32 %pc, i32 ~w, %Value* ~w, i32 ~w)
  ret { double, i1 } %r

fail:
  %f0 = insertvalue { double, i1 } undef, double 0.0, 0
  %f1 = insertvalue { double, i1 } %f0, i1 false, 1
  ret { double, i1 } %f1
}',
        [ENameGlobalIR, Sym, ParamsIR, ENameBytesLen, ENameBytesLen, Sym,
         ENameLen, ArgsSetupIR, NArgs, ArgsPtrIR, NArgs]).

plawk_dyncall_at_named_shim_off_f_ir(Name, NArgs, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    EntryArity is NArgs + 1,
    format(atom(EntryName), '~w/~w', [Name, EntryArity]),
    format(atom(ENameGlobal), 'plawk_at_ename_f_~w', [Sym]),
    llvm_emit_c_string_global(ENameGlobal, EntryName, ENameGlobalIR,
        ENameLen, ENameBytesLen),
    plawk_dyncall_at_params(NArgs, ParamsIR),
    plawk_dyncall_at_argsetup(NArgs, ArgsPtrIR, ArgsSetupIR),
    format(atom(IR),
'~w

define { double, i1 } @plawk_dyncall_at_named_f_~w(~w) {
entry:
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %resolve

resolve:
  %namep = getelementptr [~w x i8], [~w x i8]* @.plawk_at_ename_f_~w, i32 0, i32 0
  %pc = call i32 @wam_object_vm_entry_pc(%WamState* %vm, i8* %namep, i64 ~w)
  %bad = icmp slt i32 %pc, 0
  br i1 %bad, label %free_fail, label %do_call

do_call:
~w
  %r = call { double, i1 } @wam_object_call_f64(%WamState* %vm, i32 %pc, i32 ~w, %Value* ~w, i32 ~w)
  call void @wam_state_free(%WamState* %vm)
  ret { double, i1 } %r

free_fail:
  call void @wam_state_free(%WamState* %vm)
  br label %fail

fail:
  %f0 = insertvalue { double, i1 } undef, double 0.0, 0
  %f1 = insertvalue { double, i1 } %f0, i1 false, 1
  ret { double, i1 } %f1
}',
        [ENameGlobalIR, Sym, ParamsIR, ENameBytesLen, ENameBytesLen, Sym,
         ENameLen, ArgsSetupIR, NArgs, ArgsPtrIR, NArgs]).

plawk_dyncall_at_named_shim_cached_b_ir(Name, NArgs, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    EntryArity is NArgs + 1,
    format(atom(EntryName), '~w/~w', [Name, EntryArity]),
    format(atom(ENameGlobal), 'plawk_at_ename_b_~w', [Sym]),
    llvm_emit_c_string_global(ENameGlobal, EntryName, ENameGlobalIR,
        ENameLen, ENameBytesLen),
    plawk_dyncall_at_params(NArgs, ParamsIR),
    plawk_dyncall_at_argsetup(NArgs, ArgsPtrIR, ArgsSetupIR),
    format(atom(IR),
'~w

define { i8*, i64, i1 } @plawk_dyncall_at_named_b_~w(~w) {
entry:
  %obj = call { %WamState*, i32 } @plawk_dyncall_at_get(i8* %path, i64 %len)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %resolve

resolve:
  %namep = getelementptr [~w x i8], [~w x i8]* @.plawk_at_ename_b_~w, i32 0, i32 0
  %pc = call i32 @wam_object_vm_entry_pc(%WamState* %vm, i8* %namep, i64 ~w)
  %bad = icmp slt i32 %pc, 0
  br i1 %bad, label %fail, label %do_call

do_call:
~w
  %r = call { i8*, i64, i1 } @wam_object_call_bytes(%WamState* %vm, i32 %pc, i32 ~w, %Value* ~w, i32 ~w)
  ret { i8*, i64, i1 } %r

fail:
  %f0 = insertvalue { i8*, i64, i1 } undef, i8* null, 0
  %f1 = insertvalue { i8*, i64, i1 } %f0, i64 0, 1
  %f2 = insertvalue { i8*, i64, i1 } %f1, i1 false, 2
  ret { i8*, i64, i1 } %f2
}',
        [ENameGlobalIR, Sym, ParamsIR, ENameBytesLen, ENameBytesLen, Sym,
         ENameLen, ArgsSetupIR, NArgs, ArgsPtrIR, NArgs]).

plawk_dyncall_at_named_shim_off_b_ir(Name, NArgs, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    EntryArity is NArgs + 1,
    format(atom(EntryName), '~w/~w', [Name, EntryArity]),
    format(atom(ENameGlobal), 'plawk_at_ename_b_~w', [Sym]),
    llvm_emit_c_string_global(ENameGlobal, EntryName, ENameGlobalIR,
        ENameLen, ENameBytesLen),
    plawk_dyncall_at_params(NArgs, ParamsIR),
    plawk_dyncall_at_argsetup(NArgs, ArgsPtrIR, ArgsSetupIR),
    format(atom(IR),
'~w

define { i8*, i64, i1 } @plawk_dyncall_at_named_b_~w(~w) {
entry:
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %resolve

resolve:
  %namep = getelementptr [~w x i8], [~w x i8]* @.plawk_at_ename_b_~w, i32 0, i32 0
  %pc = call i32 @wam_object_vm_entry_pc(%WamState* %vm, i8* %namep, i64 ~w)
  %bad = icmp slt i32 %pc, 0
  br i1 %bad, label %free_fail, label %do_call

do_call:
~w
  %r = call { i8*, i64, i1 } @wam_object_call_bytes(%WamState* %vm, i32 %pc, i32 ~w, %Value* ~w, i32 ~w)
  call void @wam_state_free(%WamState* %vm)
  ret { i8*, i64, i1 } %r

free_fail:
  call void @wam_state_free(%WamState* %vm)
  br label %fail

fail:
  %f0 = insertvalue { i8*, i64, i1 } undef, i8* null, 0
  %f1 = insertvalue { i8*, i64, i1 } %f0, i64 0, 1
  %f2 = insertvalue { i8*, i64, i1 } %f1, i1 false, 2
  ret { i8*, i64, i1 } %f2
}',
        [ENameGlobalIR, Sym, ParamsIR, ENameBytesLen, ENameBytesLen, Sym,
         ENameLen, ArgsSetupIR, NArgs, ArgsPtrIR, NArgs]).

plawk_dyncall_at_params(0, 'i8* %path, i64 %len') :- !.
plawk_dyncall_at_params(NArgs, ParamsIR) :-
    plawk_foreign_wrapper_params(NArgs, ValParams),
    format(atom(ParamsIR), 'i8* %path, i64 %len, ~w', [ValParams]).

%% plawk_dyncall_source_ir(+Source, +FieldSeparator, +Base, +GlobalBase,
%%     -PathPtrIR, -PathLenIR, -GlobalParts, -SetupParts)
%
%  Marshal a dyncall_at source into a NUL-terminated i8* path + i64 length.
%  A field is sliced from the current line and copied into a stack buffer
%  (paths cap at 4095 bytes); a string literal becomes a constant global.
plawk_dyncall_source_ir(field(FieldIndex), FieldSeparator, Base, _GlobalBase,
        PathPtrIR, PathLenIR, [], [SliceIR, CopyIR]) :-
    integer(FieldIndex),
    FieldIndex >= 0,
    format(atom(SrcBase), '~w_src', [Base]),
    llvm_emit_atom_field_slice('%line', FieldIndex, FieldSeparator, SrcBase, SliceIR),
    format(atom(PathPtrIR), '%~w_pathp', [Base]),
    format(atom(PathLenIR), '%~w_clamped', [Base]),
    format(atom(CopyIR),
'  %~w_path = alloca [4096 x i8]
  %~w_pathp = getelementptr [4096 x i8], [4096 x i8]* %~w_path, i32 0, i32 0
  %~w_lt = icmp ult i64 %~w_src_len64, 4095
  %~w_clamped = select i1 %~w_lt, i64 %~w_src_len64, i64 4095
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %~w_pathp, i8* %~w_src_ptr, i64 %~w_clamped, i1 false)
  %~w_nulp = getelementptr i8, i8* %~w_pathp, i64 %~w_clamped
  store i8 0, i8* %~w_nulp',
        [Base, Base, Base, Base, Base, Base, Base, Base,
         Base, Base, Base, Base, Base, Base, Base]).
plawk_dyncall_source_ir(string(Str), _FieldSeparator, Base, _GlobalBase,
        PathPtrIR, PathLenIR, [GlobalIR], []) :-
    format(atom(GName), 'plawk_dyncall_at_path_~w', [Base]),
    llvm_emit_c_string_global(GName, Str, GlobalIR, StrLen, BytesLen),
    format(atom(PathPtrIR),
        'getelementptr ([~w x i8], [~w x i8]* @.~w, i32 0, i32 0)',
        [BytesLen, BytesLen, GName]),
    PathLenIR = StrLen.
% compile(field-or-string): marshal the Prolog source text exactly like
% a path (the two clauses above), then run it through @plawk_compile --
% the shipped bootstrap-compiler object compiles it and the freshly
% loaded grammar is cached, deduplicated by source text. The resulting
% HANDLE travels to the shim as (null path, handle id); the null
% pointer is the discriminator @plawk_dyncall_at_get uses to resolve
% from the cache registry instead of the filesystem.
plawk_dyncall_source_ir(compile_src(Arg), FieldSeparator, Base, GlobalBase,
        PathPtrIR, PathLenIR, GlobalParts, SetupParts) :-
    format(atom(CBase), '~w_csrc', [Base]),
    plawk_dyncall_source_ir(Arg, FieldSeparator, CBase, GlobalBase,
        SrcPtrIR, SrcLenIR, GlobalParts, SrcSetup),
    format(atom(CallIR),
        '  %~w_h = call i64 @plawk_compile(i8* ~w, i64 ~w)',
        [Base, SrcPtrIR, SrcLenIR]),
    PathPtrIR = null,
    format(atom(PathLenIR), '%~w_h', [Base]),
    append(SrcSetup, [CallIR], SetupParts).
% compile_file(path): marshal the path like any dyncall_at source, read
% the named grammar SOURCE file at runtime, and compile its contents
% through the same handle registry -- @plawk_compile_file delegates to
% @plawk_compile, so content dedup gives change-rebust for free (an
% edited file is new source text; unchanged bytes hit the registry).
plawk_dyncall_source_ir(compile_file_src(Arg), FieldSeparator, Base, GlobalBase,
        PathPtrIR, PathLenIR, GlobalParts, SetupParts) :-
    format(atom(FBase), '~w_cfile', [Base]),
    plawk_dyncall_source_ir(Arg, FieldSeparator, FBase, GlobalBase,
        PPtrIR, _PLenIR, GlobalParts, PSetup),
    format(atom(CallIR),
        '  %~w_h = call i64 @plawk_compile_file(i8* ~w)',
        [Base, PPtrIR]),
    PathPtrIR = null,
    format(atom(PathLenIR), '%~w_h', [Base]),
    append(PSetup, [CallIR], SetupParts).
% a handle read from a scalar (substituted to its slot SSA value by the
% apply pass): (null path, handle id), no marshaling -- the handle IS
% the i64. An unset or failed-compile handle is 0, which the registry
% range check rejects, so the call contributes its failure value.
plawk_dyncall_source_ir(handle_src(ssa(Value)), _FieldSeparator, _Base,
        _GlobalBase, null, Value, [], []).

%% plawk_blob_expr_ir(+Expr, +FieldSeparator, +Base, -LenIR, -PtrIR,
%%     -GlobalParts, -SetupParts)
%  Emit a blob(dyncall(...)) / blob(dyncall_at(...)) byte-slice read: call
%  the bytes shim ({ptr,len,ok}) and expose %Base_ptr (i8*) + %Base_len
%  (i32), selecting null/0 on failure so a slice print of a failed call is
%  empty.
plawk_blob_expr_ir(blob_dyncall(Args), FieldSeparator, Base,
        LenIR, PtrIR, GlobalParts, SetupParts) :-
    length(Args, NArgs),
    plawk_foreign_args_ir(Args, FieldSeparator, Base, ArgValueIRs,
        GlobalParts, ArgSetup),
    plawk_foreign_call_args_ir(ArgValueIRs, CallArgsIR),
    format(atom(ResIR),
        '  %~w_res = call { i8*, i64, i1 } @plawk_dyncall_b_~w(~w)',
        [Base, NArgs, CallArgsIR]),
    plawk_blob_tail_ir(Base, ResIR, ArgSetup, LenIR, PtrIR, SetupParts).
plawk_blob_expr_ir(blob_dyncall_named(Name, Args), FieldSeparator, Base,
        LenIR, PtrIR, GlobalParts, SetupParts) :-
    length(Args, NArgs),
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    plawk_foreign_args_ir(Args, FieldSeparator, Base, ArgValueIRs,
        GlobalParts, ArgSetup),
    plawk_foreign_call_args_ir(ArgValueIRs, CallArgsIR),
    format(atom(ResIR),
        '  %~w_res = call { i8*, i64, i1 } @plawk_dyncall_named_b_~w(~w)',
        [Base, Sym, CallArgsIR]),
    plawk_blob_tail_ir(Base, ResIR, ArgSetup, LenIR, PtrIR, SetupParts).
plawk_blob_expr_ir(blob_dyncall_at(Source, Args), FieldSeparator, Base,
        LenIR, PtrIR, GlobalParts, SetupParts) :-
    length(Args, NArgs),
    plawk_dyncall_source_ir(Source, FieldSeparator, Base, Base,
        PathPtrIR, PathLenIR, SrcGlobals, SrcSetup),
    plawk_foreign_args_ir(Args, FieldSeparator, Base, ArgValueIRs,
        ArgGlobals, ArgSetup),
    ( ArgValueIRs == []
    -> CallArgsSuffix = ''
    ;  plawk_foreign_call_args_ir(ArgValueIRs, AV),
       format(atom(CallArgsSuffix), ', ~w', [AV])
    ),
    format(atom(ResIR),
        '  %~w_res = call { i8*, i64, i1 } @plawk_dyncall_at_b_~w(i8* ~w, i64 ~w~w)',
        [Base, NArgs, PathPtrIR, PathLenIR, CallArgsSuffix]),
    append(SrcGlobals, ArgGlobals, GlobalParts),
    append(SrcSetup, ArgSetup, PreSetup),
    plawk_blob_tail_ir(Base, ResIR, PreSetup, LenIR, PtrIR, SetupParts).
plawk_blob_expr_ir(blob_dyncall_at_named(Name, Source, Args), FieldSeparator,
        Base, LenIR, PtrIR, GlobalParts, SetupParts) :-
    length(Args, NArgs),
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    plawk_dyncall_source_ir(Source, FieldSeparator, Base, Base,
        PathPtrIR, PathLenIR, SrcGlobals, SrcSetup),
    plawk_foreign_args_ir(Args, FieldSeparator, Base, ArgValueIRs,
        ArgGlobals, ArgSetup),
    ( ArgValueIRs == []
    -> CallArgsSuffix = ''
    ;  plawk_foreign_call_args_ir(ArgValueIRs, AV),
       format(atom(CallArgsSuffix), ', ~w', [AV])
    ),
    format(atom(ResIR),
        '  %~w_res = call { i8*, i64, i1 } @plawk_dyncall_at_named_b_~w(i8* ~w, i64 ~w~w)',
        [Base, Sym, PathPtrIR, PathLenIR, CallArgsSuffix]),
    append(SrcGlobals, ArgGlobals, GlobalParts),
    append(SrcSetup, ArgSetup, PreSetup),
    plawk_blob_tail_ir(Base, ResIR, PreSetup, LenIR, PtrIR, SetupParts).

plawk_blob_tail_ir(Base, ResIR, PreSetup, LenIR, PtrIR, SetupParts) :-
    format(atom(OkIR),
        '  %~w_ok = extractvalue { i8*, i64, i1 } %~w_res, 2', [Base, Base]),
    format(atom(RPtrIR),
        '  %~w_rptr = extractvalue { i8*, i64, i1 } %~w_res, 0', [Base, Base]),
    format(atom(RLenIR),
        '  %~w_rlen = extractvalue { i8*, i64, i1 } %~w_res, 1', [Base, Base]),
    format(atom(SelPtr),
        '  %~w_ptr = select i1 %~w_ok, i8* %~w_rptr, i8* null', [Base, Base, Base]),
    format(atom(SelLen),
        '  %~w_len64 = select i1 %~w_ok, i64 %~w_rlen, i64 0', [Base, Base, Base]),
    format(atom(TruncLen),
        '  %~w_len = trunc i64 %~w_len64 to i32', [Base, Base]),
    format(atom(PtrIR), '%~w_ptr', [Base]),
    format(atom(LenIR), '%~w_len', [Base]),
    append(PreSetup, [ResIR, OkIR, RPtrIR, RLenIR, SelPtr, SelLen, TruncLen],
        SetupParts).

plawk_dyncall_at_argsetup(0, 'null', '') :- !.
plawk_dyncall_at_argsetup(NArgs, '%args', SetupIR) :-
    plawk_dyncall_store_lines(NArgs, StoreLines),
    atomic_list_concat(StoreLines, '\n', StoreIR),
    format(atom(SetupIR), '  %args = alloca %Value, i32 ~w\n~w', [NArgs, StoreIR]).

% Record shims over a RUNTIME source (JIT binary reader). Signature adds
% the record-out tail (nfields, typecodes, slots, lens) after the source
% (path,len) and the boxed call args, and forwards to
% @wam_object_call_record. Cached modes (on/mtime) fetch the loaded VM
% from the source cache via @plawk_dyncall_at_get; "off" mode loads a
% fresh VM each call and frees it after. Returns i1 ok ({slots} untouched
% on failure -- the caller zero-fills before the call).
plawk_dyncall_at_rec_shim_cached_ir(NArgs, IR) :-
    plawk_dyncall_at_params(NArgs, ParamsIR),
    plawk_dyncall_at_argsetup(NArgs, ArgsPtrIR, ArgsSetupIR),
    format(atom(IR),
'define i1 @plawk_dyncall_at_rec_~w(~w, i32 %nfields, i8* %tc, i64* %slots, i64* %lens) {
entry:
  %obj = call { %WamState*, i32 } @plawk_dyncall_at_get(i8* %path, i64 %len)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %pc = extractvalue { %WamState*, i32 } %obj, 1
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
~w
  %r = call i1 @wam_object_call_record(%WamState* %vm, i32 %pc, i32 ~w, %Value* ~w, i32 ~w, i32 %nfields, i8* %tc, i64* %slots, i64* %lens)
  ret i1 %r

fail:
  ret i1 false
}',
        [NArgs, ParamsIR, ArgsSetupIR, NArgs, ArgsPtrIR, NArgs]).

plawk_dyncall_at_rec_shim_off_ir(NArgs, IR) :-
    plawk_dyncall_at_params(NArgs, ParamsIR),
    plawk_dyncall_at_argsetup(NArgs, ArgsPtrIR, ArgsSetupIR),
    format(atom(IR),
'define i1 @plawk_dyncall_at_rec_~w(~w, i32 %nfields, i8* %tc, i64* %slots, i64* %lens) {
entry:
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %pc = extractvalue { %WamState*, i32 } %obj, 1
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call

do_call:
~w
  %r = call i1 @wam_object_call_record(%WamState* %vm, i32 %pc, i32 ~w, %Value* ~w, i32 ~w, i32 %nfields, i8* %tc, i64* %slots, i64* %lens)
  call void @wam_state_free(%WamState* %vm)
  ret i1 %r

fail:
  ret i1 false
}',
        [NArgs, ParamsIR, ArgsSetupIR, NArgs, ArgsPtrIR, NArgs]).

plawk_dyncall_at_named_rec_shim_cached_ir(Name, NArgs, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    EntryArity is NArgs + 1,
    format(atom(EntryName), '~w/~w', [Name, EntryArity]),
    format(atom(ENameGlobal), 'plawk_at_rec_ename_~w', [Sym]),
    llvm_emit_c_string_global(ENameGlobal, EntryName, ENameGlobalIR,
        ENameLen, ENameBytesLen),
    plawk_dyncall_at_params(NArgs, ParamsIR),
    plawk_dyncall_at_argsetup(NArgs, ArgsPtrIR, ArgsSetupIR),
    format(atom(IR),
'~w

define i1 @plawk_dyncall_at_named_rec_~w(~w, i32 %nfields, i8* %tc, i64* %slots, i64* %lens) {
entry:
  %obj = call { %WamState*, i32 } @plawk_dyncall_at_get(i8* %path, i64 %len)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %resolve

resolve:
  %namep = getelementptr [~w x i8], [~w x i8]* @.plawk_at_rec_ename_~w, i32 0, i32 0
  %pc = call i32 @wam_object_vm_entry_pc(%WamState* %vm, i8* %namep, i64 ~w)
  %bad = icmp slt i32 %pc, 0
  br i1 %bad, label %fail, label %do_call

do_call:
~w
  %r = call i1 @wam_object_call_record(%WamState* %vm, i32 %pc, i32 ~w, %Value* ~w, i32 ~w, i32 %nfields, i8* %tc, i64* %slots, i64* %lens)
  ret i1 %r

fail:
  ret i1 false
}',
        [ENameGlobalIR, Sym, ParamsIR,
         ENameBytesLen, ENameBytesLen, Sym, ENameLen,
         ArgsSetupIR, NArgs, ArgsPtrIR, NArgs]).

plawk_dyncall_at_named_rec_shim_off_ir(Name, NArgs, IR) :-
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    EntryArity is NArgs + 1,
    format(atom(EntryName), '~w/~w', [Name, EntryArity]),
    format(atom(ENameGlobal), 'plawk_at_rec_ename_~w', [Sym]),
    llvm_emit_c_string_global(ENameGlobal, EntryName, ENameGlobalIR,
        ENameLen, ENameBytesLen),
    plawk_dyncall_at_params(NArgs, ParamsIR),
    plawk_dyncall_at_argsetup(NArgs, ArgsPtrIR, ArgsSetupIR),
    format(atom(IR),
'~w

define i1 @plawk_dyncall_at_named_rec_~w(~w, i32 %nfields, i8* %tc, i64* %slots, i64* %lens) {
entry:
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)
  %vm = extractvalue { %WamState*, i32 } %obj, 0
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %resolve

resolve:
  %namep = getelementptr [~w x i8], [~w x i8]* @.plawk_at_rec_ename_~w, i32 0, i32 0
  %pc = call i32 @wam_object_vm_entry_pc(%WamState* %vm, i8* %namep, i64 ~w)
  %bad = icmp slt i32 %pc, 0
  br i1 %bad, label %freefail, label %do_call

do_call:
~w
  %r = call i1 @wam_object_call_record(%WamState* %vm, i32 %pc, i32 ~w, %Value* ~w, i32 ~w, i32 %nfields, i8* %tc, i64* %slots, i64* %lens)
  call void @wam_state_free(%WamState* %vm)
  ret i1 %r

freefail:
  call void @wam_state_free(%WamState* %vm)
  br label %fail
fail:
  ret i1 false
}',
        [ENameGlobalIR, Sym, ParamsIR,
         ENameBytesLen, ENameBytesLen, Sym, ENameLen,
         ArgsSetupIR, NArgs, ArgsPtrIR, NArgs]).

%% plawk_program_dyncall_at_arities(+Program, -Arities)
%  Deduplicated call-arg counts (NOT counting the source) across dyncall_at
%  sites; one @plawk_dyncall_at_N shim per arity.
plawk_program_dyncall_at_arities(program(_Begin, Rules, EndClauses), Arities) :-
    findall(NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          plawk_actions_dyncall_at(Actions, _Source, Args),
          length(Args, NArgs)
        ),
        Arities0),
    sort(Arities0, Arities).

%% plawk_program_dyncall_at_named_entries(+Program, -Entries)
%  Name-NArgs pairs across dyncall_at@name(Source, args...) sites -- the
%  named-entry-on-a-runtime-source composition. One
%  @plawk_dyncall_at_named_<Name>_<N> shim per entry; the name resolves
%  per call against the loaded VM's materialized entry table
%  (@wam_object_vm_entry_pc), since the object is runtime data and a
%  startup-cached PC has nothing fixed to bind to.
plawk_program_dyncall_at_named_entries(program(_Begin, Rules, EndClauses),
        Entries) :-
    findall(Name-NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          plawk_actions_dyncall_at_named(Actions, Name, _Source, Args),
          length(Args, NArgs)
        ),
        Entries0),
    sort(Entries0, Entries).

%% plawk_program_dyncache_mode(+Program, -Mode)
%  DYNCACHE mode string ("on" | "mtime" | "off"); default "on".
plawk_program_dyncache_mode(program(BeginClauses, _Rules, _End), Mode) :-
    (   member(begin(Actions), BeginClauses),
        member(set(var('DYNCACHE'), string(M)), Actions)
    ->  Mode = M
    ;   Mode = "on"
    ).

%% plawk_program_evalc_path(+Program, -Path)
%  The bootstrap-compiler .wamo path from BEGIN { EVALC = "..." }, or
%  fails if absent (the CLI then ships its own compiler object next to
%  the binary and passes the path through the evalc_path/1 option).
plawk_program_evalc_path(program(BeginClauses, _Rules, _End), Path) :-
    member(begin(Actions), BeginClauses),
    member(set(var('EVALC'), string(Path)), Actions),
    !.

%% plawk_program_compile_sites(+Program, -Sources)
%  The deduplicated compile(...) source args across every dyncall_at
%  variant (i64 / float / blob). Non-empty means the program uses the
%  eval surface: @plawk_compile must be emitted and a bootstrap-compiler
%  object must exist at the evalc path when the binary runs.
plawk_program_compile_sites(Program, Sites) :-
    Program = program(_Begin, Rules, EndClauses),
    findall(Arg,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          ( ( plawk_actions_dyncall_at(Actions, CSrc, _)
            ; plawk_actions_dyncall_at_named(Actions, _NName, CSrc, _)
            ; plawk_actions_float_dyncall_at(Actions, CSrc, _)
            ; plawk_actions_float_dyncall_at_named(Actions, _FName, CSrc, _)
            ; plawk_actions_dynrec_at(Actions, CSrc, _)
            ; plawk_actions_dynrec_at_named(Actions, _RName, CSrc, _)
            ),
            plawk_compile_source_node(CSrc, Arg)
          ; % handle expressions (h = compile(...)) -- compile sites
            % OUTSIDE any dyncall_at source position
            plawk_actions_compile_handle(Actions, Arg)
          )
        ;   % blob positions via the generic walk (print fields, writebin
            % slots, assoc keys, blob_eq patterns)
            ( plawk_program_blob_node(Program, blob_dyncall_at(CSrc, _))
            ; plawk_program_blob_node(Program,
                  blob_dyncall_at_named(_BName, CSrc, _))
            ),
            plawk_compile_source_node(CSrc, Arg)
        ),
        Sites0),
    sort(Sites0, Sites).

% both eval-source forms count as compile sites (they need the shipped
% compiler object and the @plawk_compile support IR)
plawk_compile_source_node(compile_src(Arg), Arg).
plawk_compile_source_node(compile_file_src(Arg), Arg).

% handle-expression walker: h = compile(src) / compile_file(path) in
% set positions (incl. if-branches) are compile sites too.
plawk_actions_compile_handle(Actions, Arg) :-
    member(Action, Actions),
    plawk_action_compile_handle(Action, Arg).
plawk_action_compile_handle(set(_Var, compile_handle(Arg)), Arg).
plawk_action_compile_handle(set(_Var, compile_file_handle(Arg)), Arg).
plawk_action_compile_handle(if(_Pattern, ThenActions, ElseActions), Arg) :-
    ( plawk_actions_compile_handle(ThenActions, Arg)
    ; plawk_actions_compile_handle(ElseActions, Arg)
    ).

plawk_actions_dyncall_at(Actions, Source, Args) :-
    member(Action, Actions),
    plawk_action_dyncall_at(Action, Source, Args).
plawk_action_dyncall_at(add(_Var, Expr), S, A) :- plawk_expr_dyncall_at(Expr, S, A).
plawk_action_dyncall_at(set(_Var, Expr), S, A) :- plawk_expr_dyncall_at(Expr, S, A).
plawk_action_dyncall_at(print(Fields), S, A) :-
    member(Field, Fields),
    plawk_expr_dyncall_at(Field, S, A).
plawk_action_dyncall_at(printf(_Format, PrintfArgs), S, A) :-
    member(Field, PrintfArgs),
    plawk_expr_dyncall_at(Field, S, A).
plawk_action_dyncall_at(if(_Pattern, ThenActions, ElseActions), S, A) :-
    ( plawk_actions_dyncall_at(ThenActions, S, A)
    ; plawk_actions_dyncall_at(ElseActions, S, A)
    ).
plawk_expr_dyncall_at(dyncall_at(Source, Args), Source, Args).
plawk_expr_dyncall_at(Expr, S, A) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    ( plawk_expr_dyncall_at(Left, S, A)
    ; plawk_expr_dyncall_at(Right, S, A)
    ).

% dyncall_at@name walker -- mirrors the plain walker with the entry name
% threaded through (arity/entry collection + compile-site collection).
plawk_actions_dyncall_at_named(Actions, Name, Source, Args) :-
    member(Action, Actions),
    plawk_action_dyncall_at_named(Action, Name, Source, Args).
plawk_action_dyncall_at_named(add(_Var, Expr), N, S, A) :-
    plawk_expr_dyncall_at_named(Expr, N, S, A).
plawk_action_dyncall_at_named(set(_Var, Expr), N, S, A) :-
    plawk_expr_dyncall_at_named(Expr, N, S, A).
plawk_action_dyncall_at_named(print(Fields), N, S, A) :-
    member(Field, Fields),
    plawk_expr_dyncall_at_named(Field, N, S, A).
plawk_action_dyncall_at_named(printf(_Format, PrintfArgs), N, S, A) :-
    member(Field, PrintfArgs),
    plawk_expr_dyncall_at_named(Field, N, S, A).
plawk_action_dyncall_at_named(if(_Pattern, ThenActions, ElseActions), N, S, A) :-
    ( plawk_actions_dyncall_at_named(ThenActions, N, S, A)
    ; plawk_actions_dyncall_at_named(ElseActions, N, S, A)
    ).
plawk_expr_dyncall_at_named(dyncall_at_named(Name, Source, Args), Name,
    Source, Args).
plawk_expr_dyncall_at_named(Expr, N, S, A) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    ( plawk_expr_dyncall_at_named(Left, N, S, A)
    ; plawk_expr_dyncall_at_named(Right, N, S, A)
    ).

%% Record-destructure over a RUNTIME source: `(v..) = dyncall_at[@name](
%%   Src, args) as (T..)`. Walk dynrec_bind actions (incl. nested in
%%   if/else) for the at-form call. Kept separate from the i64/float/blob
%%   at-walkers so those collectors do not emit spurious scalar shims for
%%   a record-only entry.
plawk_actions_dynrec_at(Actions, Source, Args) :-
    member(Action, Actions),
    plawk_action_dynrec_at(Action, Source, Args).
plawk_action_dynrec_at(dynrec_bind(_V, dyncall_at(Source, Args), _T),
    Source, Args).
plawk_action_dynrec_at(if(_P, Then, Else), S, A) :-
    ( plawk_actions_dynrec_at(Then, S, A)
    ; plawk_actions_dynrec_at(Else, S, A)
    ).

plawk_actions_dynrec_at_named(Actions, Name, Source, Args) :-
    member(Action, Actions),
    plawk_action_dynrec_at_named(Action, Name, Source, Args).
plawk_action_dynrec_at_named(
    dynrec_bind(_V, dyncall_at_named(Name, Source, Args), _T),
    Name, Source, Args).
plawk_action_dynrec_at_named(if(_P, Then, Else), N, S, A) :-
    ( plawk_actions_dynrec_at_named(Then, N, S, A)
    ; plawk_actions_dynrec_at_named(Else, N, S, A)
    ).

%% plawk_program_dyncall_at_rec_arities(+Program, -Arities)
%% plawk_program_dyncall_at_named_rec_entries(+Program, -Entries)
%  Record shims over a runtime source: @plawk_dyncall_at_rec_<N> (default
%  entry) and @plawk_dyncall_at_named_rec_<Sym> (named). Both resolve the
%  VM per call from the source cache and forward to @wam_object_call_record.
plawk_program_dyncall_at_rec_arities(program(_Begin, Rules, EndClauses),
        Arities) :-
    findall(NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          plawk_actions_dynrec_at(Actions, _Source, Args),
          length(Args, NArgs)
        ),
        A0),
    sort(A0, Arities).

plawk_program_dyncall_at_named_rec_entries(program(_Begin, Rules, EndClauses),
        Entries) :-
    findall(Name-NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          plawk_actions_dynrec_at_named(Actions, Name, _Source, Args),
          length(Args, NArgs)
        ),
        E0),
    sort(E0, Entries).

%% float(dyncall(...)) / float(dyncall_at(...)) arity collectors -- one
%  double-returning shim per distinct call-arg count.
plawk_program_dyncall_float_arities(program(_Begin, Rules, EndClauses), Arities) :-
    findall(NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          plawk_actions_float_dyncall(Actions, Args),
          length(Args, NArgs)
        ),
        Arities0),
    sort(Arities0, Arities).

plawk_program_dyncall_at_float_arities(program(_Begin, Rules, EndClauses), Arities) :-
    findall(NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          plawk_actions_float_dyncall_at(Actions, _Source, Args),
          length(Args, NArgs)
        ),
        Arities0),
    sort(Arities0, Arities).

plawk_actions_float_dyncall(Actions, Args) :-
    member(Action, Actions),
    plawk_action_float_dyncall(Action, Args).
plawk_action_float_dyncall(add(_Var, Expr), Args) :- plawk_expr_float_dyncall(Expr, Args).
plawk_action_float_dyncall(set(_Var, Expr), Args) :- plawk_expr_float_dyncall(Expr, Args).
plawk_action_float_dyncall(print(Fields), Args) :-
    member(Field, Fields), plawk_expr_float_dyncall(Field, Args).
plawk_action_float_dyncall(printf(_Format, PrintfArgs), Args) :-
    member(Field, PrintfArgs), plawk_expr_float_dyncall(Field, Args).
plawk_action_float_dyncall(if(_Pattern, ThenActions, ElseActions), Args) :-
    ( plawk_actions_float_dyncall(ThenActions, Args)
    ; plawk_actions_float_dyncall(ElseActions, Args)
    ).
plawk_expr_float_dyncall(float_dyncall(Args), Args).
plawk_expr_float_dyncall(Expr, Args) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    ( plawk_expr_float_dyncall(Left, Args)
    ; plawk_expr_float_dyncall(Right, Args)
    ).

plawk_actions_float_dyncall_at(Actions, Source, Args) :-
    member(Action, Actions),
    plawk_action_float_dyncall_at(Action, Source, Args).
plawk_action_float_dyncall_at(add(_Var, Expr), S, A) :- plawk_expr_float_dyncall_at(Expr, S, A).
plawk_action_float_dyncall_at(set(_Var, Expr), S, A) :- plawk_expr_float_dyncall_at(Expr, S, A).
plawk_action_float_dyncall_at(print(Fields), S, A) :-
    member(Field, Fields), plawk_expr_float_dyncall_at(Field, S, A).
plawk_action_float_dyncall_at(printf(_Format, PrintfArgs), S, A) :-
    member(Field, PrintfArgs), plawk_expr_float_dyncall_at(Field, S, A).
plawk_action_float_dyncall_at(if(_Pattern, ThenActions, ElseActions), S, A) :-
    ( plawk_actions_float_dyncall_at(ThenActions, S, A)
    ; plawk_actions_float_dyncall_at(ElseActions, S, A)
    ).
plawk_expr_float_dyncall_at(float_dyncall_at(Source, Args), Source, Args).
plawk_expr_float_dyncall_at(Expr, S, A) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    ( plawk_expr_float_dyncall_at(Left, S, A)
    ; plawk_expr_float_dyncall_at(Right, S, A)
    ).

%% plawk_program_dyncall_at_named_float_entries(+Program, -Entries)
%  Name-NArgs pairs across float(dyncall_at@name(...)) sites -- the
%  double-returning named-at shims.
plawk_program_dyncall_at_named_float_entries(
        program(_Begin, Rules, EndClauses), Entries) :-
    findall(Name-NArgs,
        ( ( member(rule(_Pattern, Actions), Rules)
          ; member(end(Actions), EndClauses)
          ),
          plawk_actions_float_dyncall_at_named(Actions, Name, _Source, Args),
          length(Args, NArgs)
        ),
        Entries0),
    sort(Entries0, Entries).

plawk_actions_float_dyncall_at_named(Actions, Name, Source, Args) :-
    member(Action, Actions),
    plawk_action_float_dyncall_at_named(Action, Name, Source, Args).
plawk_action_float_dyncall_at_named(add(_Var, Expr), N, S, A) :-
    plawk_expr_float_dyncall_at_named(Expr, N, S, A).
plawk_action_float_dyncall_at_named(set(_Var, Expr), N, S, A) :-
    plawk_expr_float_dyncall_at_named(Expr, N, S, A).
plawk_action_float_dyncall_at_named(print(Fields), N, S, A) :-
    member(Field, Fields),
    plawk_expr_float_dyncall_at_named(Field, N, S, A).
plawk_action_float_dyncall_at_named(printf(_Format, PrintfArgs), N, S, A) :-
    member(Field, PrintfArgs),
    plawk_expr_float_dyncall_at_named(Field, N, S, A).
plawk_action_float_dyncall_at_named(if(_Pattern, ThenActions, ElseActions),
        N, S, A) :-
    ( plawk_actions_float_dyncall_at_named(ThenActions, N, S, A)
    ; plawk_actions_float_dyncall_at_named(ElseActions, N, S, A)
    ).
plawk_expr_float_dyncall_at_named(float_dyncall_at_named(Name, Source, Args),
    Name, Source, Args).
plawk_expr_float_dyncall_at_named(Expr, N, S, A) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    ( plawk_expr_float_dyncall_at_named(Left, N, S, A)
    ; plawk_expr_float_dyncall_at_named(Right, N, S, A)
    ).

%% blob(dyncall(...)) / blob(dyncall_at(...)) arity collectors -- one
%  byte-slice shim per distinct call-arg count. blob only appears as a
%  print/writebin field (not inside arithmetic), so the walkers scan
%  print/printf fields directly.
% One generic walk finds blob nodes ANYWHERE in a rule -- print fields,
% writebin string slots, assoc keys, blob_eq patterns, if-branches --
% so the collectors need not mirror each structural walker (the shims
% must exist for every position that evaluates a blob).
plawk_program_blob_node(program(_Begin, Rules, EndClauses), Blob) :-
    ( member(Rule, Rules) ; member(Rule, EndClauses) ),
    plawk_subterm_blob(Rule, Blob).

plawk_subterm_blob(Term, Term) :-
    compound(Term),
    functor(Term, F, _),
    memberchk(F, [blob_dyncall, blob_dyncall_named, blob_dyncall_at,
        blob_dyncall_at_named]).
plawk_subterm_blob(Term, Blob) :-
    compound(Term),
    arg(_, Term, Sub),
    plawk_subterm_blob(Sub, Blob).

plawk_program_dyncall_blob_arities(Program, Arities) :-
    findall(NArgs,
        ( plawk_program_blob_node(Program, blob_dyncall(Args)),
          length(Args, NArgs)
        ),
        A0),
    sort(A0, Arities).

plawk_program_dyncall_at_blob_arities(Program, Arities) :-
    findall(NArgs,
        ( plawk_program_blob_node(Program, blob_dyncall_at(_Source, Args)),
          length(Args, NArgs)
        ),
        A0),
    sort(A0, Arities).

%% plawk_program_dyncall_at_named_blob_entries(+Program, -Entries)
%  Name-NArgs pairs across blob(dyncall_at@name(...)) sites, found by
%  the same generic blob-node walk as the other blob kinds.
plawk_program_dyncall_at_named_blob_entries(Program, Entries) :-
    findall(Name-NArgs,
        ( plawk_program_blob_node(Program,
              blob_dyncall_at_named(Name, _Source, Args)),
          length(Args, NArgs)
        ),
        E0),
    sort(E0, Entries).

plawk_action_blob_field(print(Fields), Blob) :- member(Blob, Fields).
plawk_action_blob_field(printf(_Format, PrintfArgs), Blob) :- member(Blob, PrintfArgs).
plawk_action_blob_field(if(_Pattern, ThenActions, ElseActions), Blob) :-
    ( member(A, ThenActions) ; member(A, ElseActions) ),
    plawk_action_blob_field(A, Blob).

%% plawk_program_multipass_driver_ir(+Program, -DriverIR) is semidet.
%
%  The multi-pass execution driver (PLAWK_MULTIPASS_CACHE.md phase 2, PR-B).
%  A program with 2+ `pass { }` blocks runs the record loop once per pass
%  over the (re-opened) input, then END. Each pass is emitted as its OWN
%  function -- the per-record SSA (%line) and loop labels are fixed names,
%  so N loops in one function would redefine them; a function per pass makes
%  them function-local. Shared assoc tables are created once in main and
%  threaded to each pass as a parameter (named %plawk_assoc_table_0 so the
%  reused rule-chain IR references it unchanged); the END for-in reads the
%  same tables back in main. Cross-pass state therefore lives in the shared
%  table object, mutated by every pass.
%
%  v1 scope: text mode; a single shared assoc table; passes are always-rule
%  bodies (no break/next); an `END { for (k in arr) print ... }`. The input
%  must be a file argument (re-opened per pass); stdin is not re-openable
%  (the design requires an explicit spool, a later phase).
plawk_program_multipass_driver_ir(
    program_passes(BeginClauses, Passes,
        [end([for_in(var(LoopVar), var(ArrayName), [print(PrintFields)])])]),
    DriverIR
) :-
    Passes = [_, _ | _],
    \+ member(pass_over(_, _, _), Passes),   % over-readers use the no-END driver
    plawk_record_descriptor(BeginClauses, FieldSeparator),
    integer(FieldSeparator),                 % text mode only (v1)
    plawk_output_separator(BeginClauses, OutputSeparator),
    findall(R, ( member(pass(PassRules), Passes), member(R, PassRules) ), AllRules),
    plawk_forin_end_plan(AllRules, LoopVar, ArrayName, [print(PrintFields)],
        AssocPlan, PrintFields),
    AssocPlan = assoc_plan([ArrayName], _),  % v1: one shared table
    % A `BEGIN cache("path") { declare NAME }` table shared by the passes:
    % load the store into the shared table before pass 1, commit it after the
    % last pass. Non-cache programs get [] and behave exactly as before.
    plawk_program_cache_tables(BeginClauses, CacheTriples),
    maplist(plawk_assoc_rule_action_specs, AllRules, AllRuleSpecs),
    plawk_assoc_specs_str_arrays(AllRuleSpecs, StrArrays),
    findall(cache_schema(ST, SC),
        ( member(begin(BActions), BeginClauses),
          member(cache_schema(ST, SC), BActions) ),
        Schemas),
    plawk_cache_entries([ArrayName], StrArrays, Schemas, CacheTriples, CacheEntries,
        CachePathGlobals),
    % Per-pass RecordIR (rule chain) against the single shared table.
    findall(GlobalIR-ChainIR,
        ( member(pass(PassRules), Passes),
          plawk_forin_assoc_plan(PassRules, ArrayName, [], PassPlan),
          plawk_assoc_rule_controls(PassPlan, PassControls),
          plawk_assoc_break_close_ir(PassControls, ''),  % no break/next in v1
          plawk_assoc_rule_chain_ir(PassPlan, FieldSeparator, GlobalIR, ChainIR)
        ),
        PassPairs),
    pairs_keys_values(PassPairs, ChainGlobals, Chains),
    % One function per pass; index them. The END for-in always has one
    % shared table, threaded to each pass.
    plawk_multipass_table_params([ArrayName], TableParamsIR, TableArgsIR),
    findall(FnIR,
        ( nth0(I, Chains, Chain),
          plawk_multipass_pass_fn_ir(I, TableParamsIR, Chain, FnIR) ),
        FnIRs),
    atomic_list_concat(FnIRs, '\n', PassFnsIR),
    % main's pass-call sequence.
    findall(CallLine,
        ( nth0(I, Chains, _),
          format(atom(CallLine),
              '  call void @plawk_pass_~w(%Value %mp_path~w)', [I, TableArgsIR]) ),
        CallLines),
    atomic_list_concat(CallLines, '\n', CallsIR),
    % Setup (create + load the shared table), BEGIN, and the END for-in print.
    plawk_assoc_entry_setup_ir(AssocPlan, CacheEntries, EntrySetupIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_combine_entry_ir(BeginIR, EntrySetupIR, CombinedEntrySetupIR),
    plawk_cache_commit_lines(CacheEntries, CommitIR),
    plawk_forin_end_print_ir(LoopVar, ArrayName, PrintFields, AssocPlan,
        FieldSeparator, OutputSeparator, EndPrintIR0),
    atomic_list_concat([CommitIR, EndPrintIR0], '\n', EndPrintIR),
    % Module globals: the chains' string/format globals + cache path globals.
    sort(ChainGlobals, ChainGlobalsSorted),
    atomic_list_concat(ChainGlobalsSorted, '\n', ChainGlobalsIR0),
    atomic_list_concat([ChainGlobalsIR0, CachePathGlobals], '\n', ChainGlobalsIR),
    plawk_i64_end_print_globals(ChainGlobalsIR, RuntimeGlobals),
    format(atom(DriverIR),
'@.wam_stream_eof = private constant [12 x i8] c"end_of_file\\00"
~w
~w

define i32 @main(i32 %argc, i8** %argv) {
entry:
~w
  %have_arg = icmp sgt i32 %argc, 1
  br i1 %have_arg, label %get_path, label %no_arg

no_arg:
  ret i32 20

get_path:
  %argv1_ptr = getelementptr i8*, i8** %argv, i64 1
  %argv1 = load i8*, i8** %argv1_ptr
  %argv1_len = call i64 @strlen(i8* %argv1)
  %mp_path_id = call i64 @wam_intern_atom(i8* %argv1, i64 %argv1_len)
  %mp_path0 = insertvalue %Value undef, i32 0, 0
  %mp_path = insertvalue %Value %mp_path0, i64 %mp_path_id, 1
~w
  br label %end_print

end_print:
~w
}
',
        [RuntimeGlobals, PassFnsIR, CombinedEntrySetupIR, CallsIR, EndPrintIR]).

% Multi-pass with NO END block: at least one pass emits per-record output
% (e.g. `pass { c[$1]++ }  pass { print $1, c[$1] }` -- the normalise shape,
% where pass 2 reads the table pass 1 built). Same per-pass-function model
% and shared table; the difference is there is no END for-in, and the shared
% table is discovered from the passes rather than an END loop.
plawk_program_multipass_driver_ir(
    program_passes(BeginClauses, Passes, []),
    DriverIR
) :-
    Passes = [_, _ | _],
    plawk_record_descriptor(BeginClauses, FieldSeparator),
    integer(FieldSeparator),
    plawk_passes_tables(Passes, Tables),        % 0 or 1 table (v1)
    findall(R, ( member(pass(PassRules), Passes), member(R, PassRules) ), AllRules),
    plawk_multipass_pass_plan(AllRules, Tables, AssocPlan),  % validates surface
    plawk_multipass_table_params(Tables, TableParamsIR, TableArgsIR),
    % Row schemas (from `declare NAME(col type, ...)` or a `use NAME` whose
    % store schema was read at build time): column name -> position for the
    % named `records of` reader. Empty for schema-less programs.
    findall(cache_schema(ST, SC),
        ( member(begin(BActions), BeginClauses),
          member(cache_schema(ST, SC), BActions) ),
        Schemas),
    % Program-wide str-valued (ROW) tables: a table holds string (row) values
    % if a writer populates it (`arr[$k] = $0` / `= row(...)`), if a row
    % reader consumes it (`records of` / `rows of`), or if it carries a row
    % schema (so a pure `use` reader with no writer is still byte-valued).
    % Str tables use the byte-valued cache load/commit, so this must be
    % complete or a durable row store is (mis)read with the i64 loader.
    maplist(plawk_assoc_rule_action_specs, AllRules, AllRuleSpecs),
    plawk_assoc_specs_str_arrays(AllRuleSpecs, WriterStr),
    findall(RT,
        ( member(pass_records(_RV, var(RT), _RB), Passes)
        ; member(pass_rows(_PV, var(RT), _PB), Passes)
        ; member(pass_rows_anon(var(RT), _AB), Passes)
        ; member(cache_schema(RT, _), Schemas)
        ),
        ReaderStr),
    append(WriterStr, ReaderStr, StrArrays0),
    sort(StrArrays0, StrArrays),
    % A `BEGIN cache(...)`-declared shared table: load before pass 1, commit
    % after the last pass. Empty for pure-scalar / non-cache programs.
    plawk_program_cache_tables(BeginClauses, CacheTriples),
    plawk_cache_entries(Tables, StrArrays, Schemas, CacheTriples, CacheEntries,
        CachePathGlobals),
    plawk_output_separator(BeginClauses, OutputSeparator),
    % One function per pass, in order -- dispatching on the pass shape:
    % an input-scanning `pass { }` or a table-iterating `pass over T as V`.
    % nth0 enumerates in order, so FnIRs/GlobalIRs line up with the call
    % sequence by index.
    findall(FnIR-GlobalIR,
        ( nth0(I, Passes, Pass),
          plawk_multipass_pass_fn(I, Pass, Tables, StrArrays, Schemas,
              TableParamsIR, FieldSeparator, OutputSeparator, FnIR, GlobalIR)
        ),
        FnPairs),
    % Every pass must yield exactly one function; if any pass shape is
    % outside the supported surface, fail the whole driver (bin/plawk then
    % reports it unsupported) rather than emit a call to a missing function.
    length(Passes, NPasses),
    length(FnPairs, NPasses),
    pairs_keys_values(FnPairs, FnIRs, ChainGlobals),
    atomic_list_concat(FnIRs, '\n', PassFnsIR),
    findall(CallLine,
        ( nth0(I, Passes, _),
          format(atom(CallLine),
              '  call void @plawk_pass_~w(%Value %mp_path~w)', [I, TableArgsIR]) ),
        CallLines),
    atomic_list_concat(CallLines, '\n', CallsIR),
    plawk_assoc_entry_setup_ir(AssocPlan, CacheEntries, EntrySetupIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    plawk_combine_entry_ir(BeginIR, EntrySetupIR, CombinedEntrySetupIR),
    sort(ChainGlobals, ChainGlobalsSorted),
    atomic_list_concat(ChainGlobalsSorted, '\n', ChainGlobalsIR0),
    atomic_list_concat([ChainGlobalsIR0, CachePathGlobals], '\n', ChainGlobalsIR),
    plawk_i64_end_print_globals(ChainGlobalsIR, RuntimeGlobals),
    plawk_cache_commit_lines(CacheEntries, CommitIR),
    phrase(plawk_assoc_free_lines(AssocPlan), FreeLines),
    atomic_list_concat([CommitIR | FreeLines], '\n', FreeIR),
    format(atom(DriverIR),
'@.wam_stream_eof = private constant [12 x i8] c"end_of_file\\00"
~w
~w

define i32 @main(i32 %argc, i8** %argv) {
entry:
~w
  %have_arg = icmp sgt i32 %argc, 1
  br i1 %have_arg, label %get_path, label %no_arg

no_arg:
  ret i32 20

get_path:
  %argv1_ptr = getelementptr i8*, i8** %argv, i64 1
  %argv1 = load i8*, i8** %argv1_ptr
  %argv1_len = call i64 @strlen(i8* %argv1)
  %mp_path_id = call i64 @wam_intern_atom(i8* %argv1, i64 %argv1_len)
  %mp_path0 = insertvalue %Value undef, i32 0, 0
  %mp_path = insertvalue %Value %mp_path0, i64 %mp_path_id, 1
~w
~w
  ret i32 0
}
',
        [RuntimeGlobals, PassFnsIR, CombinedEntrySetupIR, CallsIR, FreeIR]).

%% plawk_program_multipass_driver_ir(+Program, +Options, -DriverIR) is semidet.
%
%  Options-carrying entry point (PLAWK_QUERY_READER_IMPLEMENTATION_PLAN.md,
%  phase 6). `Options` carries `wam_vm(InstrCount, LabelCount)` -- the sizes
%  of the module's @module_code / @module_labels arrays, needed to build the
%  shared %WamState a query reader runs its goal on. A query-reader program
%  (`pass over query(pred(X)) { print $1 }`) is lowered by the query driver;
%  everything else delegates to the /2 driver unchanged.
plawk_program_multipass_driver_ir(Program, Options, DriverIR) :-
    (   plawk_program_query_driver_ir(Program, Options, DriverIR)
    ->  true                                       % all-query: self-contained
    ;   plawk_program_multipass_driver_ir(Program, BaseIR),
        plawk_query_mixed_support_ir(Program, Options, SupportIR),
        atom_concat(SupportIR, BaseIR, DriverIR)
    ).

%% plawk_query_mixed_support_ir(+Program, +Options, -SupportIR)
%  Extra module-level IR a general multi-pass program needs when it also
%  contains a query pass (phase 6, PR 5): the shared %WamState getter the query
%  reader runs its goal on, which the ordinary multi-pass driver does not emit.
%  Sized by the module code/label counts in Options. Empty when the program has
%  no query pass (ordinary multi-pass is unchanged). The i64 print-format global
%  the query body uses is already emitted by the driver's runtime globals.
plawk_query_mixed_support_ir(Program, Options, SupportIR) :-
    plawk_program_query_passes(Program, QueryPasses),
    (   QueryPasses == []
    ->  SupportIR = ''
    ;   memberchk(wam_vm(InstrCount, LabelCount), Options),
        plawk_query_foreign_vm_ir(InstrCount, LabelCount, VmIR),
        plawk_query_posarray_value_ir(ValueIR),
        format(atom(SupportIR), '~w\n\n~w\n\n', [VmIR, ValueIR])
    ).

%% plawk_program_query_passes(+Program, -QueryPasses) is det.
%  The `pass over query(...)` passes of a program (empty if none).
plawk_program_query_passes(program_passes(_, Passes, _), QueryPasses) :-
    findall(pass_query(Q, B), member(pass_query(Q, B), Passes), QueryPasses).
plawk_program_query_passes(program(_, _, _), []).

%% plawk_program_materialize_views(+Program, -Names) is det.
%  The names declared by `materialize NAME` (PLAWK_MULTIPASS_CACHE.md §3.9),
%  in program order (empty if none). Surface-first: the runtime is a follow-on.
plawk_program_materialize_views(program_passes(_, Passes, _), Names) :-
    findall(Name, member(materialize(Name), Passes), Names).
plawk_program_materialize_views(program(_, _, _), []).

%% plawk_program_gen_blocks(+Program, -GenBlocks) is det.
%  The `gen { ... } as name` generator blocks of a program (empty if none).
%  The producer dual of the query reader (PLAWK_GENERATOR_BLOCKS.md); collected
%  alongside the passes at parse time.
plawk_program_gen_blocks(program_passes(_, Passes, _), GenBlocks) :-
    findall(gen_block(N, S, B), member(gen_block(N, S, B), Passes), GenBlocks).
plawk_program_gen_blocks(program(_, _, _), []).

%% plawk_query_pass_supported(+Pass) is semidet.
%  The query-reader surface a build can lower today (phase 6, PRs 2-4): a goal
%  of any arity >= 1 whose body is a single `print` of `$K` fields (each `K` a
%  column of the goal, 1..arity), optionally gated by a reader guard
%  `if (COND)` comparing `$K` columns to integers (`&&` / `||` combinations
%  allowed). Every solution's arguments are ground integers materialised at
%  $1..$n. Mixed passes and non-integer columns are follow-ons; an unsupported
%  shape yields a clean not-yet compile error rather than a miscompile.
plawk_query_pass_supported(pass_query(query(_Pred, Vars), Body)) :-
    Vars = [_ | _],
    length(Vars, Arity),
    plawk_query_body_plan(Body, Fields, Guard),
    Fields = [_ | _],
    forall(member(F, Fields),
        ( F = field(K), integer(K), K >= 1, K =< Arity )),
    plawk_query_guard_ok(Guard, Arity).

%% plawk_query_body_plan(+Body, -Fields, -Guard) is semidet.
%  Split a query pass body into its print field list and an optional reader
%  guard. A bare `print` has guard `none`; an `if (COND) print ...` (which
%  `for_in_body` parses to `[if(Guard, [print(Fields)], [])]`) carries the
%  condition. No other body shape is a query reader today.
plawk_query_body_plan([print(Fields)], Fields, none).
plawk_query_body_plan([if(Guard, [print(Fields)], [])], Fields, Guard).

%% plawk_query_guard_ok(+Guard, +Arity) is semidet.
%  A reader guard a query pass can lower: `&&` / `||` over `$K CMP int` leaves
%  (`rfield_cmp`), each column in range. Columns are i64, so only integer
%  comparisons are meaningful in v1 (float/string RHS are a follow-on).
plawk_query_guard_ok(none, _Arity).
plawk_query_guard_ok(and(L, R), Arity) :-
    plawk_query_guard_ok(L, Arity),
    plawk_query_guard_ok(R, Arity).
plawk_query_guard_ok(or(L, R), Arity) :-
    plawk_query_guard_ok(L, Arity),
    plawk_query_guard_ok(R, Arity).
plawk_query_guard_ok(rfield_cmp(N, _Op, Value), Arity) :-
    integer(N), N >= 1, N =< Arity,
    integer(Value).

%% plawk_query_helper_name(+Pred, +Col, -Name) is det.
%  The synthesised findall-wrapper predicate name for column `Col` of a query
%  on `Pred`. The reserved `__plawk_query_` prefix cannot collide with a user
%  predicate (the surface parser forbids `$`-free reserved names), so the name
%  is unique per goal predicate/column and both the clause injector (bin/plawk)
%  and the pass-fn emitter derive the same LLVM symbol (`@<Name>_start_pc`).
plawk_query_helper_name(Pred, Col, Name) :-
    atomic_list_concat(['__plawk_query_', Pred, '_', Col], Name).

%% plawk_query_helper_clause(+Pred, +Arity, +Col, -Clause) is det.
%  The Prolog clause that turns column `Col` of the goal's solution set into a
%  materialisable list: `__plawk_query_pred_Col(L) :- findall(VCol, pred(V1,
%  ..., Vn), L)`. findall drives the goal to exhaustion (the enumeration the
%  plan delegates to the builtin), so L is the ordered, ground list of that
%  column's bindings. One wrapper per column keeps materialisation on the
%  existing flat-list posarray path; findall preserves solution order and
%  multiplicity identically across columns, so column i's k-th element is the
%  k-th solution's i-th argument -- correct as long as the goal is a pure
%  generator (no writes interleaved between the per-column findall runs, which
%  the reader's snapshot boundary already assumes). Injected into the program's
%  @prolog set so write_wam_llvm_project compiles it into the same binary.
plawk_query_helper_clause(Pred, Arity, Col, Clause) :-
    length(Vars, Arity),
    nth1(Col, Vars, V),
    Goal =.. [Pred | Vars],
    plawk_query_helper_name(Pred, Col, Name),
    Head =.. [Name, L],
    Clause = (Head :- findall(V, Goal, L)).

%% plawk_program_query_driver_ir(+Program, +Options, -DriverIR) is semidet.
%
%  The query-reader execution driver (phase 6, PRs 2-3). Fires only for an
%  all-query program (every pass is a supported `pass over query(...)`), the
%  first slice of controlled non-determinism: each pass runs its goal to a
%  materialised solution set and iterates it -- no input file is read (the
%  records come from the goal, not stdin). Each pass is its own void function
%  (materialise-then-iterate, mirroring `over TABLE`); a shared %WamState
%  (@plawk_foreign_vm) runs the findall-wrapper predicates the injector added.
%  Each goal column materialises into its own assoc table (keys 1..N by
%  position); the pass walks keys in order, binding $1..$n from the per-column
%  tables and printing the requested fields (§1 intact -- the multiplicity is
%  collapsed at the boundary before the body runs).
plawk_program_query_driver_ir(
    program_passes(BeginClauses, Passes, End),
    Options,
    DriverIR
) :-
    Passes = [_ | _],
    forall(member(P, Passes), P = pass_query(_, _)),   % all-query program (v1)
    forall(member(P, Passes), plawk_query_pass_supported(P)),
    End == [],                                          % v1: no END block
    memberchk(wam_vm(InstrCount, LabelCount), Options),
    plawk_output_separator(BeginClauses, OutputSeparator),
    plawk_query_foreign_vm_ir(InstrCount, LabelCount, VmIR),
    plawk_query_posarray_value_ir(ValueIR),
    findall(FnIR,
        ( nth0(I, Passes, pass_query(query(Pred, Vars), Body)),
          length(Vars, Arity),
          plawk_query_body_plan(Body, Fields, Guard),
          plawk_multipass_query_fn_ir(I, '', Pred, Arity, Fields, Guard,
              OutputSeparator, FnIR) ),
        FnIRs),
    atomic_list_concat(FnIRs, '\n', PassFnsIR),
    findall(CallLine,
        ( nth0(I, Passes, _),
          format(atom(CallLine), '  call void @plawk_pass_~w()', [I]) ),
        CallLines),
    atomic_list_concat(CallLines, '\n', CallsIR),
    plawk_begin_print_ir(BeginClauses, OutputSeparator, BeginIR),
    format(atom(DriverIR),
'@.plawk_surface_print_i64 = private constant [4 x i8] c"%ld\\00"
@.plawk_surface_print_string = private constant [3 x i8] c"%s\\00"
~w

~w

~w

define i32 @main(i32 %argc, i8** %argv) {
entry:
~w
~w
  ret i32 0
}
',
        [VmIR, ValueIR, PassFnsIR, BeginIR, CallsIR]).

%% plawk_query_foreign_vm_ir(+InstrCount, +LabelCount, -IR)
%  The lazily created process-wide %WamState the query readers run their
%  goals on -- identical to the foreign-call support VM, emitted standalone
%  here because a query-reader program has no foreign-call sites to trigger
%  plawk_foreign_support_ir. References @module_code / @module_labels (the
%  compiled @prolog + findall-wrapper predicates) sized by the passed counts.
plawk_query_foreign_vm_ir(InstrCount, LabelCount, IR) :-
    format(atom(IR),
'@plawk_foreign_vm = internal global %WamState* null

define %WamState* @plawk_foreign_vm_get() {
entry:
  %cur = load %WamState*, %WamState** @plawk_foreign_vm
  %have = icmp ne %WamState* %cur, null
  br i1 %have, label %ret_cur, label %make

ret_cur:
  ret %WamState* %cur

make:
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
  store %WamState* %vm, %WamState** @plawk_foreign_vm
  ret %WamState* %vm
}',
        [InstrCount, InstrCount, InstrCount, LabelCount, LabelCount,
         LabelCount]).

%% plawk_query_posarray_value_ir(-IR)
%  The tagged materialisation primitive the query reader uses (phase 6, PR 6):
%  like @wam_object_call_posarray, but each flat-list element may be an Integer
%  OR an Atom, and its kind is recorded alongside its value. Element i (1-indexed
%  by position) stores its i64 value into %table[i] and its kind into
%  %kindtable[i] -- 0 for an integer (the payload) or 1 for an atom (its
%  registry id, resolved to text at print via @wam_atom_to_string). So a goal
%  column can carry integers or strings without a build-time type. Any other
%  element kind fails the call (empty tables). Emitted with the query support
%  (all-query driver / mixed splice), self-contained -- references only runtime
%  helpers already present in any WAM module.
plawk_query_posarray_value_ir(
'define i1 @wam_object_call_posarray_value(%WamState* %vm, i32 %entry_pc, i32 %nargs, %Value* %args, i32 %out_reg, %WamAssocI64Table* %table, %WamAssocI64Table* %kindtable) {
entry:
  %vm_null = icmp eq %WamState* %vm, null
  br i1 %vm_null, label %fail, label %do_call
do_call:
  %hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %hs_saved = load i32, i32* %hs_ptr
  %unb = call %Value @value_unbound(i8* null)
  %out_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %out_ref = call %Value @value_ref(i32 %out_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %entry_pc)
  br label %argloop
argloop:
  %ai = phi i32 [ 0, %do_call ], [ %ai1, %argstep ]
  %adone = icmp sge i32 %ai, %nargs
  br i1 %adone, label %args_done, label %argstep
argstep:
  %aidx64 = sext i32 %ai to i64
  %ap = getelementptr %Value, %Value* %args, i64 %aidx64
  %av = load %Value, %Value* %ap
  call void @wam_set_reg(%WamState* %vm, i32 %ai, %Value %av)
  %ai1 = add i32 %ai, 1
  br label %argloop
args_done:
  call void @wam_set_reg(%WamState* %vm, i32 %out_reg, %Value %out_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %walk_init, label %rewind_fail
walk_init:
  %consfn = getelementptr [4 x i8], [4 x i8]* @.fn__5B_7C_5D, i32 0, i32 0
  %head0 = call %Value @wam_deref_value(%WamState* %vm, %Value %out_ref)
  br label %walk
walk:
  %cur = phi %Value [ %head0, %walk_init ], [ %tail_d, %next ]
  %pos = phi i64 [ 1, %walk_init ], [ %pos1, %next ]
  %ctag = call i32 @value_tag(%Value %cur)
  %is_comp = icmp eq i32 %ctag, 3
  br i1 %is_comp, label %check_cons, label %walk_done
check_cons:
  %cbits = call i64 @value_payload(%Value %cur)
  %ccp = inttoptr i64 %cbits to %Compound*
  %cfn_slot = getelementptr %Compound, %Compound* %ccp, i32 0, i32 0
  %cfn = load i8*, i8** %cfn_slot
  %car_slot = getelementptr %Compound, %Compound* %ccp, i32 0, i32 1
  %car = load i32, i32* %car_slot
  %car_ok = icmp eq i32 %car, 2
  br i1 %car_ok, label %cmp_cons, label %walk_done
cmp_cons:
  %fncmp = call i32 @strcmp(i8* %cfn, i8* %consfn)
  %is_cons = icmp eq i32 %fncmp, 0
  br i1 %is_cons, label %have_cell, label %walk_done
have_cell:
  %cargs_slot = getelementptr %Compound, %Compound* %ccp, i32 0, i32 2
  %cargs = load %Value*, %Value** %cargs_slot
  %h_ptr = getelementptr %Value, %Value* %cargs, i64 0
  %h_raw = load %Value, %Value* %h_ptr
  %t_ptr = getelementptr %Value, %Value* %cargs, i64 1
  %t_raw = load %Value, %Value* %t_ptr
  %elem = call %Value @wam_deref_value(%WamState* %vm, %Value %h_raw)
  %etag = call i32 @value_tag(%Value %elem)
  %e_is_int = icmp eq i32 %etag, 1
  br i1 %e_is_int, label %insert_int, label %check_atom
check_atom:
  %e_is_atom = icmp eq i32 %etag, 0
  br i1 %e_is_atom, label %insert_atom, label %rewind_fail
insert_int:
  %ival = call i64 @value_payload(%Value %elem)
  %ii = call i64 @wam_assoc_i64_set(%WamAssocI64Table* %table, i64 %pos, i64 %ival)
  %ik = call i64 @wam_assoc_i64_set(%WamAssocI64Table* %kindtable, i64 %pos, i64 0)
  br label %next
insert_atom:
  %aval = call i64 @value_payload(%Value %elem)
  %aset = call i64 @wam_assoc_i64_set(%WamAssocI64Table* %table, i64 %pos, i64 %aval)
  %ak = call i64 @wam_assoc_i64_set(%WamAssocI64Table* %kindtable, i64 %pos, i64 1)
  br label %next
next:
  %pos1 = add i64 %pos, 1
  %tail_d = call %Value @wam_deref_value(%WamState* %vm, %Value %t_raw)
  br label %walk
walk_done:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  ret i1 true
rewind_fail:
  store i32 %hs_saved, i32* %hs_ptr
  call void @wam_cleanup()
  br label %fail
fail:
  ret i1 false
}').

%% plawk_multipass_query_fn_ir(+Index, +ParamsIR, +Pred, +Arity, +Fields,
%%     +Guard, +OutputSep, -IR)
%  A query pass as a self-contained void function: materialise each goal column
%  into TWO assoc tables (`%qval_C` = the value, `%qkind_C` = 0 for an integer /
%  1 for an atom, keys 1..N by position) via @wam_object_call_posarray_value
%  against the shared VM -- calling the per-column findall wrapper
%  `@__plawk_query_pred_C_start_pc` with no input args and the list output in
%  register 0. Then walk keys in order (1..count of column 1, not hash-slot
%  order, so the print is deterministic), binding $K to `%qval_K[pos]`. Each
%  printed field branches on its kind: an integer prints as `%ld`, an atom is
%  resolved to text (@wam_atom_to_string) and printed as `%s` -- so a column may
%  carry integers or strings without a build-time type. An optional reader guard
%  (`if (COND)`) compares the raw i64 value (integer columns); pure i64 reads
%  combined with i1 and/or gate the print. All tables are freed at pass end.
%
%  `ParamsIR` is the function's parameter list content: empty for the all-query
%  driver (`@plawk_pass_N()`), or the standard multi-pass signature
%  (`%Value %mp_path` + the shared-table params) when a query pass runs inside
%  the general multi-pass driver alongside ordinary passes -- the params are
%  ignored (a query reads no input and shares no table), but the signature must
%  match main's call.
plawk_multipass_query_fn_ir(Index, ParamsIR, Pred, Arity, Fields, Guard,
        OutputSep, IR) :-
    numlist(1, Arity, Cols),
    findall(MatLine,
        ( member(C, Cols),
          plawk_query_helper_name(Pred, C, Sym),
          format(atom(MatLine),
'  %qval_~w = call %WamAssocI64Table* @wam_assoc_i64_new(i64 64)
  %qkind_~w = call %WamAssocI64Table* @wam_assoc_i64_new(i64 64)
  %qpc_~w = load i32, i32* @~w_start_pc
  %qok_~w = call i1 @wam_object_call_posarray_value(%WamState* %vm, i32 %qpc_~w, i32 0, %Value* null, i32 0, %WamAssocI64Table* %qval_~w, %WamAssocI64Table* %qkind_~w)',
              [C, C, C, Sym, C, C, C, C]) ),
        MatLines),
    atomic_list_concat(MatLines, '\n', MaterializeIR),
    plawk_query_print_lines(Fields, OutputSep, PrintIR),
    plawk_query_body_ir(Guard, PrintIR, BodyIR),
    findall(FreeLine,
        ( member(C, Cols),
          format(atom(FreeLine),
'  call void @wam_assoc_i64_free(%WamAssocI64Table* %qval_~w)
  call void @wam_assoc_i64_free(%WamAssocI64Table* %qkind_~w)',
              [C, C]) ),
        FreeLines),
    atomic_list_concat(FreeLines, '\n', FreeIR),
    format(atom(IR),
'define void @plawk_pass_~w(~w) {
entry:
  %vm = call %WamState* @plawk_foreign_vm_get()
~w
  %count_ptr = getelementptr %WamAssocI64Table, %WamAssocI64Table* %qval_1, i32 0, i32 0
  %count = load i64, i64* %count_ptr
  br label %q_head

q_head:
  %pos = phi i64 [ 1, %entry ], [ %pos1, %q_body_done ]
  %q_done = icmp sgt i64 %pos, %count
  br i1 %q_done, label %q_after, label %q_body

q_body:
~w

q_body_done:
  %pos1 = add i64 %pos, 1
  br label %q_head

q_after:
~w
  ret void
}',
        [Index, ParamsIR, MaterializeIR, BodyIR, FreeIR]).

%% plawk_query_body_ir(+Guard, +PrintIR, -BodyIR)
%  The `q_body` block content. Unguarded: print then fall to `q_body_done`.
%  Guarded: evaluate the guard to an i1, branch to a `q_print` block on true
%  (else straight to `q_body_done`). The guard reads are pure, so the tree is
%  plain and/or over leaf comparisons -- no short-circuit control flow.
plawk_query_body_ir(none, PrintIR, BodyIR) :-
    format(atom(BodyIR), '~w\n  br label %q_body_done', [PrintIR]).
plawk_query_body_ir(Guard, PrintIR, BodyIR) :-
    Guard \== none,
    plawk_query_guard_ir(Guard, 0, _C, ResSSA, GuardLines),
    atomic_list_concat(GuardLines, '\n', GuardIR),
    format(atom(BodyIR),
'~w
  br i1 ~w, label %q_print, label %q_body_done

q_print:
~w
  br label %q_body_done',
        [GuardIR, ResSSA, PrintIR]).

%% plawk_query_guard_ir(+Guard, +N0, -N, -ResultSSA, -Lines)
%  Lower a reader guard over the per-column value tables to a single i1 result.
%  A leaf `$K CMP int` reads `%qval_K[pos]` (the raw i64 value -- integer
%  columns) and compares it; `and` / `or` combine child i1s. SSA names are
%  suffixed by a threaded counter so leaves never collide. Reads are
%  side-effect-free, so and/or need not short-circuit.
plawk_query_guard_ir(and(L, R), N0, N, Res, Lines) :-
    plawk_query_guard_ir(L, N0, N1, LRes, LLines),
    plawk_query_guard_ir(R, N1, N2, RRes, RLines),
    N is N2 + 1,
    format(atom(Res), '%qgand~w', [N2]),
    format(atom(Line), '  ~w = and i1 ~w, ~w', [Res, LRes, RRes]),
    append([LLines, RLines, [Line]], Lines).
plawk_query_guard_ir(or(L, R), N0, N, Res, Lines) :-
    plawk_query_guard_ir(L, N0, N1, LRes, LLines),
    plawk_query_guard_ir(R, N1, N2, RRes, RLines),
    N is N2 + 1,
    format(atom(Res), '%qgor~w', [N2]),
    format(atom(Line), '  ~w = or i1 ~w, ~w', [Res, LRes, RRes]),
    append([LLines, RLines, [Line]], Lines).
plawk_query_guard_ir(rfield_cmp(K, Op, Value), N0, N, Res, [LoadLine, CmpLine]) :-
    integer(Value),
    plawk_icmp_pred(Op, Pred),
    N is N0 + 1,
    format(atom(ValSSA), '%qgv~w', [N0]),
    format(atom(Res), '%qgc~w', [N0]),
    format(atom(LoadLine),
        '  ~w = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %qval_~w, i64 %pos)',
        [ValSSA, K]),
    format(atom(CmpLine),
        '  ~w = icmp ~w i64 ~w, ~w', [Res, Pred, ValSSA, Value]).

%% plawk_query_print_lines(+Fields, +OutputSep, -IR)
%  The per-record print body: each `field(K)` reads `%qval_K[pos]` and its kind
%  `%qkind_K[pos]`, then branches -- an integer prints as `%ld`, an atom is
%  resolved to text (@wam_atom_to_string) and printed as `%s`. Fields are joined
%  by the output separator (a putchar between them) and terminated by a newline.
%  SSA names and the per-field branch labels are keyed by the field's ordinal,
%  so a repeated column (`print $1, $1`) stays unique.
plawk_query_print_lines(Fields, OutputSep, IR) :-
    length(Fields, NF),
    Last is NF - 1,
    findall(Line,
        ( nth0(J, Fields, field(K)),
          length(RestJs, 20), maplist(=(J), RestJs),
          format(atom(FieldBlock),
'  %qf_~w = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %qval_~w, i64 %pos)
  %qk_~w = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %qkind_~w, i64 %pos)
  %qisatom_~w = icmp eq i64 %qk_~w, 1
  br i1 %qisatom_~w, label %qatom_~w, label %qint_~w
qint_~w:
  %qfi_~w = getelementptr [4 x i8], [4 x i8]* @.plawk_surface_print_i64, i32 0, i32 0
  %qpri_~w = call i32 (i8*, ...) @printf(i8* %qfi_~w, i64 %qf_~w)
  br label %qpd_~w
qatom_~w:
  %qas_~w = call i8* @wam_atom_to_string(i64 %qf_~w)
  %qfa_~w = getelementptr [3 x i8], [3 x i8]* @.plawk_surface_print_string, i32 0, i32 0
  %qpra_~w = call i32 (i8*, ...) @printf(i8* %qfa_~w, i8* %qas_~w)
  br label %qpd_~w
qpd_~w:',
              [J, K, J, K | RestJs]),
          ( J < Last
          ->  format(atom(Line),
                  '~w\n  %qsep_~w = call i32 @putchar(i32 ~w)',
                  [FieldBlock, J, OutputSep])
          ;   Line = FieldBlock )
        ),
        Lines),
    atomic_list_concat(Lines, '\n', Body0),
    format(atom(IR), '~w\n  %qnl = call i32 @putchar(i32 10)', [Body0]).

%% plawk_passes_tables(+Passes, -Tables)
%  The assoc tables a multi-pass program shares. Discovered from the passes'
%  actions: a counted key `arr[$k]++` or a print lookup `arr[$k]`; and from an
%  `over TABLE` / `records of` / `rows of` reader, which names the table it
%  iterates. A pure-scalar program (e.g. `pass { total += $2 }`) has none.
%  `sort/2` dedups and fixes a stable order, so each table's position is its
%  `%plawk_assoc_table_<i>` index consistently across setup, params, and every
%  pass's rule/reader planning. Multiple tables (phase 8.9, PR 1) share this
%  one in-memory set; durable multi-table storage (named sub-DBs) is a later PR.
plawk_passes_tables(Passes, Tables) :-
    findall(Name,
        ( member(pass(Rules), Passes), member(rule(_, Actions), Rules),
          member(Action, Actions), plawk_action_table_name(Action, Name)
        ; member(pass_over(_Var, var(Name), _Body), Passes)
        ; member(pass_records(_RVar, var(Name), _RBody), Passes)
        ; member(pass_rows(_PVar, var(Name), _PBody), Passes)
        ; member(pass_rows_anon(var(Name), _ABody), Passes)
        ),
        Names0),
    sort(Names0, Tables).

plawk_action_table_name(inc_assoc(var(Name), _Key), Name).
plawk_action_table_name(delete_assoc(var(Name), _Key), Name).
plawk_action_table_name(split_into(_Src, var(Name), _Sep), Name).
plawk_action_table_name(add_assoc(var(Name), _Key, _Delta), Name).
plawk_action_table_name(set_row(var(Name), _Key), Name).
plawk_action_table_name(set_row_cons(var(Name), _Key, _Fields), Name).
plawk_action_table_name(print(Fields), Name) :-
    member(assoc(var(Name), _Key), Fields).

%% plawk_multipass_table_params(+Tables, -ParamsIR, -ArgsIR)
%  The LLVM parameter/argument suffix threading the shared table(s) into a
%  pass function. Zero tables (pure scalar) -> empty; N tables -> one
%  `%plawk_assoc_table_<i>` per table, indexed by position in the (sorted)
%  Tables list, so every pass function receives the whole table set and a
%  rule/reader references its table by that index. Param and arg strings are
%  identical (same SSA names in main and in the callee signature).
plawk_multipass_table_params(Tables, ParamsIR, ParamsIR) :-
    plawk_multipass_table_params_parts(Tables, 0, Parts),
    atomic_list_concat(Parts, '', ParamsIR).

plawk_multipass_table_params_parts([], _, []).
plawk_multipass_table_params_parts([_Table | Rest], Index, [Part | Parts]) :-
    format(atom(Part),
        ', %WamAssocI64Table* %plawk_assoc_table_~w', [Index]),
    NextIndex is Index + 1,
    plawk_multipass_table_params_parts(Rest, NextIndex, Parts).

%% plawk_multipass_pass_plan(+Rules, +Tables, -AssocPlan)
%  Plan a pass's rules against the (already discovered) shared table set, so
%  table indices are consistent across passes. Works with zero tables.
plawk_multipass_pass_plan(Rules, Tables, assoc_plan(Tables, PlannedRules)) :-
    maplist(plawk_assoc_rule_action_specs, Rules, RuleSpecs),
    plawk_assoc_specs_str_arrays(RuleSpecs, StrArrays),
    plawk_assoc_specs_posarray_arrays(RuleSpecs, PosArrays),
    phrase(plawk_assoc_planned_rules(RuleSpecs, Tables, StrArrays, PosArrays, 0),
        PlannedRules).

%% plawk_multipass_pass_fn(+Index, +Pass, +Tables, +TableParamsIR,
%%     +FieldSep, +OutputSep, -FnIR, -GlobalIR)
%  Emit one pass function, dispatching on the pass shape. An input-scanning
%  `pass { }` compiles its rule chain and scans the input per record; an
%  `over TABLE as VAR` reader (phase 4) iterates the table's entries instead
%  of the input. GlobalIR carries any module globals the pass's body needs
%  (rule chains emit format/string globals; the over reader emits none in
%  v1 -- key + lookup print fields use the shared runtime constants).
plawk_multipass_pass_fn(Index, pass(PassRules), Tables, _StrArrays, _Schemas,
        TableParamsIR, FieldSep, _OutputSep, FnIR, GlobalIR) :-
    plawk_multipass_pass_plan(PassRules, Tables, PassPlan),
    plawk_assoc_rule_controls(PassPlan, PassControls),
    plawk_assoc_break_close_ir(PassControls, ''),   % no break/next in v1
    plawk_assoc_rule_chain_ir(PassPlan, FieldSep, GlobalIR, ChainIR),
    plawk_multipass_pass_fn_ir(Index, TableParamsIR, ChainIR, FnIR).
plawk_multipass_pass_fn(Index, pass_over(var(Var), var(Table), Body), Tables,
        StrArrays, _Schemas, TableParamsIR, FieldSep, OutputSep, FnIR, '') :-
    Body = [print(PrintFields)],
    PrintFields = [_ | _],
    maplist(plawk_over_print_field_ok(Var), PrintFields),
    % Carry the program's str-valued tables so a lookup of a row-valued table
    % (`TABLE[k]` = a stored row) resolves the id to the row's bytes.
    AssocPlan = assoc_plan(Tables, [str_arrays(StrArrays)]),
    plawk_multipass_over_fn_ir(Index, TableParamsIR, Var, Table, PrintFields,
        AssocPlan, FieldSep, OutputSep, FnIR).
% The named row reader: iterate TABLE's row entries, decode each stored row by
% the declared schema, print the named columns. Requires a cache_schema for
% TABLE; each print field must be VAR["col"] with col in the schema (name-only
% -- numeric/other fields fail the clause, so the driver reports unsupported).
plawk_multipass_pass_fn(Index, pass_records(var(Var), var(Table), Body), Tables,
        _StrArrays, Schemas, TableParamsIR, FieldSep, OutputSep, FnIR, GuardGlobals) :-
    plawk_reader_body(Body, PrintFields, GuardTerm),
    PrintFields = [_ | _],
    memberchk(cache_schema(Table, Columns), Schemas),
    maplist(plawk_records_field_plan(Var, Columns), PrintFields, FieldPlans),
    plawk_records_guard(GuardTerm, Var, Columns, GuardPlan),
    nth0(TableIndex, Tables, Table),
    plawk_multipass_records_fn_ir(Index, TableParamsIR, TableIndex, FieldPlans,
        GuardPlan, FieldSep, OutputSep, FnIR, GuardGlobals).

% A row-reader body is either a bare `{ print ... }` or a guarded
% `{ if (GUARD) print ... }` (a WHERE-style filter). Extract the print fields
% and the guard term (none when unguarded).
plawk_reader_body([print(Fields)], Fields, none).
plawk_reader_body([if(Guard, [print(Fields)], [])], Fields, Guard).

% Resolve a `records of` guard (name column) to guard(ColIndex, Op, Value),
% the schema position compared to an integer; `none` passes through.
plawk_records_guard(none, _Var, _Columns, none).
plawk_records_guard(and(L, R), Var, Columns, and(PL, PR)) :-
    plawk_records_guard(L, Var, Columns, PL),
    plawk_records_guard(R, Var, Columns, PR).
plawk_records_guard(or(L, R), Var, Columns, or(PL, PR)) :-
    plawk_records_guard(L, Var, Columns, PL),
    plawk_records_guard(R, Var, Columns, PR).
plawk_records_guard(rcol_cmp(Var, Col, Op, Value), Var, Columns,
        guard(Index, Op, Value)) :-
    plawk_records_col_index(Columns, Col, Index).

% Resolve a `records of` print field to a plan: a plain column VAR["col"] ->
% col(Index) (its 1-based schema position, printed as text), or an arithmetic
% expression over columns/constants -> arith(F64Op, L, R) (evaluated in f64,
% printed with %g -- the surface `/` is integer). Name-only operands.
plawk_records_field_plan(Var, Columns, assoc(var(Var), string(Col)), col(Index)) :-
    plawk_records_col_index(Columns, Col, Index).
plawk_records_field_plan(Var, Columns, Expr, arith(F64Op, LPlan, RPlan)) :-
    Expr =.. [Op, L, R],
    plawk_i64_op_f64(Op, F64Op),
    plawk_records_operand(Var, Columns, L, LPlan),
    plawk_records_operand(Var, Columns, R, RPlan).
plawk_records_operand(Var, Columns, assoc(var(Var), string(Col)), col(Index)) :-
    plawk_records_col_index(Columns, Col, Index).
plawk_records_operand(_Var, _Columns, int(V), aint(V)) :- integer(V).
plawk_records_col_index(Columns, Col, Index) :-
    atom_string(ColAtom, Col),
    nth1(Index, Columns, col(ColAtom, _Type)).
% The positional row reader: iterate TABLE's rows, print columns addressed
% BY POSITION (`VAR[N]`, 1-indexed = field N of the stored row). No schema
% needed -- for raw stores. Reuses the same row-decode function emitter as
% `records of`, just sourcing the field indices from the literal positions.
plawk_multipass_pass_fn(Index, pass_rows(var(Var), var(Table), Body), Tables,
        _StrArrays, _Schemas, TableParamsIR, FieldSep, OutputSep, FnIR, GuardGlobals) :-
    plawk_reader_body(Body, PrintFields, GuardTerm),
    PrintFields = [_ | _],
    maplist(plawk_rows_field_plan(Var), PrintFields, FieldPlans),
    plawk_rows_guard(GuardTerm, Var, GuardPlan),
    nth0(TableIndex, Tables, Table),
    plawk_multipass_records_fn_ir(Index, TableParamsIR, TableIndex, FieldPlans,
        GuardPlan, FieldSep, OutputSep, FnIR, GuardGlobals).

% Resolve a `rows of` guard (positional) -> guard(N, Op, Value); none passes.
plawk_rows_guard(none, _Var, none).
plawk_rows_guard(and(L, R), Var, and(PL, PR)) :-
    plawk_rows_guard(L, Var, PL),
    plawk_rows_guard(R, Var, PR).
plawk_rows_guard(or(L, R), Var, or(PL, PR)) :-
    plawk_rows_guard(L, Var, PL),
    plawk_rows_guard(R, Var, PR).
plawk_rows_guard(rpos_cmp(Var, N, Op, Value), Var, guard(N, Op, Value)) :-
    integer(N), N > 0.

% A `rows of` print field: a plain position VAR[N] -> col(N), or arithmetic
% over positions/constants -> arith(...), evaluated in f64. Positional only.
plawk_rows_field_plan(Var, assoc(var(Var), int(N)), col(N)) :-
    integer(N), N > 0.
plawk_rows_field_plan(Var, Expr, arith(F64Op, LPlan, RPlan)) :-
    Expr =.. [Op, L, R],
    plawk_i64_op_f64(Op, F64Op),
    plawk_rows_operand(Var, L, LPlan),
    plawk_rows_operand(Var, R, RPlan).
plawk_rows_operand(Var, assoc(var(Var), int(N)), col(N)) :- integer(N), N > 0.
plawk_rows_operand(_Var, int(V), aint(V)) :- integer(V).

% The no-`as` positional reader (`pass rows of T { print $1, $2 }`): awk-native
% `$N` field addressing over the stored row. Mirrors plawk_rows_field_plan but
% sources positions from `field(N)` ($N) rather than `VAR[N]`.
plawk_multipass_pass_fn(Index, pass_rows_anon(var(Table), Body), Tables,
        _StrArrays, _Schemas, TableParamsIR, FieldSep, OutputSep, FnIR, GuardGlobals) :-
    plawk_reader_body(Body, PrintFields, GuardTerm),
    PrintFields = [_ | _],
    maplist(plawk_rows_anon_field_plan, PrintFields, FieldPlans),
    plawk_rows_anon_guard(GuardTerm, GuardPlan),
    nth0(TableIndex, Tables, Table),
    plawk_multipass_records_fn_ir(Index, TableParamsIR, TableIndex, FieldPlans,
        GuardPlan, FieldSep, OutputSep, FnIR, GuardGlobals).
% A query pass inside a general multi-pass program (phase 6, PR 5: mixed
% query + ordinary passes). The query reader needs no input and no shared
% table, but must adopt the standard pass-function signature so main's call
% site is uniform; the params are ignored. The shared %WamState the goal runs
% on (@plawk_foreign_vm) is emitted once by the driver (it has the module
% code/label counts); this clause only emits the pass body. No per-pass string
% globals (the i64 print format comes from the driver's runtime globals).
plawk_multipass_pass_fn(Index, pass_query(query(Pred, Vars), Body), _Tables,
        _StrArrays, _Schemas, TableParamsIR, _FieldSep, OutputSep, FnIR, '') :-
    plawk_query_pass_supported(pass_query(query(Pred, Vars), Body)),
    length(Vars, Arity),
    plawk_query_body_plan(Body, Fields, Guard),
    atom_concat('%Value %mp_path', TableParamsIR, ParamsIR),
    plawk_multipass_query_fn_ir(Index, ParamsIR, Pred, Arity, Fields, Guard,
        OutputSep, FnIR).

% Resolve a no-`as` guard (`$N CMP int`) -> guard(N, Op, Value); none passes.
plawk_rows_anon_guard(none, none).
plawk_rows_anon_guard(and(L, R), and(PL, PR)) :-
    plawk_rows_anon_guard(L, PL),
    plawk_rows_anon_guard(R, PR).
plawk_rows_anon_guard(or(L, R), or(PL, PR)) :-
    plawk_rows_anon_guard(L, PL),
    plawk_rows_anon_guard(R, PR).
plawk_rows_anon_guard(rfield_cmp(N, Op, Value), guard(N, Op, Value)) :-
    integer(N), N > 0.

plawk_rows_anon_field_plan(field(N), col(N)) :- integer(N), N > 0.
plawk_rows_anon_field_plan(Expr, arith(F64Op, LPlan, RPlan)) :-
    Expr =.. [Op, L, R],
    plawk_i64_op_f64(Op, F64Op),
    plawk_rows_anon_operand(L, LPlan),
    plawk_rows_anon_operand(R, RPlan).
plawk_rows_anon_operand(field(N), col(N)) :- integer(N), N > 0.
plawk_rows_anon_operand(int(V), aint(V)) :- integer(V).

% Over-reader print fields (v1): the loop key, or a lookup of the iterated
% table keyed by it. String literals / other tables are follow-ons.
plawk_over_print_field_ok(Var, var(Var)).
plawk_over_print_field_ok(Var, assoc(var(_Table), var(Var))).

%% plawk_multipass_over_fn_ir(+Index, +TableParamsIR, +Var, +Table,
%%     +PrintFields, +AssocPlan, +FieldSep, +OutputSep, -IR)
%  An `over TABLE as VAR` pass as a self-contained void function: it does
%  NOT open the input -- it walks the shared table's occupied slots
%  (@wam_assoc_i64_iter_next) and prints each entry's fields (the same body
%  emitter the END for-in uses, so key text / TABLE[VAR] value / separators
%  are identical), one line per entry, then returns. The table is freed /
%  committed by main, not here.
plawk_multipass_over_fn_ir(Index, TableParamsIR, Var, Table, PrintFields,
        AssocPlan, FieldSep, OutputSep, IR) :-
    plawk_assoc_table_index(AssocPlan, Table, TableIndex),
    phrase(plawk_forin_body_print_lines(PrintFields, Var, Table, TableIndex,
        AssocPlan, FieldSep, OutputSep, 0), BodyLines),
    atomic_list_concat(BodyLines, '\n', BodyIR),
    format(atom(IR),
'define void @plawk_pass_~w(%Value %mp_path~w) {
entry:
  br label %forin_head

forin_head:
  %forin_idx = phi i64 [0, %entry], [%forin_next_idx, %forin_body_done]
  %forin_slot = call i64 @wam_assoc_i64_iter_next(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_idx)
  %forin_done = icmp slt i64 %forin_slot, 0
  br i1 %forin_done, label %forin_after, label %forin_body

forin_body:
  %forin_key_id = call i64 @wam_assoc_i64_key_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)
~w
  %forin_printed_newline = call i32 @putchar(i32 10)
  br label %forin_body_done

forin_body_done:
  %forin_next_idx = add i64 %forin_slot, 1
  br label %forin_head

forin_after:
  ret void
}
', [Index, TableParamsIR, TableIndex, TableIndex, BodyIR]).

%% plawk_multipass_records_fn_ir(+Index, +TableParamsIR, +TableIndex,
%%     +ColIndexes, +GuardPlan, +FieldSep, +OutputSep, -IR)
%  The named row reader as a void function: walk the shared table's occupied
%  slots; for each, the value is the stored row's atom id. Build a row %Value
%  from it and print the requested columns -- each `r["col"]` resolved to the
%  column's schema position, extracted as field N of the row via
%  @wam_atom_field_slice_value (the same field projection the input record
%  uses), printed as text. An optional GuardPlan (guard(ColIndex, Op, Value))
%  gates the print: a WHERE-style filter comparing a column's i64 value to a
%  constant, only emitting the row when it passes. Does not open the input;
%  the table is freed by main.
plawk_multipass_records_fn_ir(Index, TableParamsIR, TableIndex, ColIndexes,
        GuardPlan, FieldSep, OutputSep, IR, Globals) :-
    phrase(plawk_records_col_lines(ColIndexes, FieldSep, OutputSep, 0), ColLines),
    atomic_list_concat(ColLines, '\n', ColIR),
    plawk_records_guard_ir(GuardPlan, FieldSep, Index, GuardIR, Globals),
    format(atom(IR),
'define void @plawk_pass_~w(%Value %mp_path~w) {
entry:
  br label %rec_head

rec_head:
  %rec_idx = phi i64 [0, %entry], [%rec_next_idx, %rec_body_done]
  %rec_slot = call i64 @wam_assoc_i64_iter_next(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %rec_idx)
  %rec_done = icmp slt i64 %rec_slot, 0
  br i1 %rec_done, label %rec_after, label %rec_body

rec_body:
  %rec_row_id = call i64 @wam_assoc_i64_value_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %rec_slot)
  %rec_row_v0 = insertvalue %Value undef, i32 0, 0
  %rec_row_v = insertvalue %Value %rec_row_v0, i64 %rec_row_id, 1
~w

rec_print:
~w
  %rec_printed_newline = call i32 @putchar(i32 10)
  br label %rec_body_done

rec_body_done:
  %rec_next_idx = add i64 %rec_slot, 1
  br label %rec_head

rec_after:
  ret void
}
', [Index, TableParamsIR, TableIndex, TableIndex, GuardIR, ColIR]).

%% plawk_records_guard_ir(+GuardPlan, +FieldSep, -IR)
%  The transition out of rec_body: unguarded readers fall straight into
%  rec_print; a guard extracts its column and compares to the constant, then
%  branches to rec_print or skips the row (rec_body_done). A bare-integer RHS
%  extracts the column as i64 (@wam_atom_field_i64_value) and uses icmp; a float
%  literal (float_const, e.g. `3.5`) extracts it as double
%  (@wam_atom_field_f64_value) and uses fcmp against the exact decimal ratio.
plawk_records_guard_ir(none, _FieldSep, _Index, '  br label %rec_print', '').
plawk_records_guard_ir(Guard, FieldSep, Index, IR, Globals) :-
    Guard \== none,
    plawk_guard_tree_lines(Guard, FieldSep, Index, 0, 'rec_print',
        'rec_body_done', Lines, GlobalList, _N),
    atomic_list_concat(Lines, '\n', IR),
    atomic_list_concat(GlobalList, '\n', Globals).

%% plawk_guard_tree_lines(+Guard, +FieldSep, +Index, +N0, +TrueLbl, +FalseLbl,
%%     -Lines, -Globals, -N1)
%  Lower a (possibly boolean) reader-guard tree to short-circuit branches: a
%  leaf comparison branches to TrueLbl (match) / FalseLbl (skip); `and(L, R)`
%  runs R only when L is true (L's true-target is a fresh block holding R);
%  `or(L, R)` runs R only when L is false. N threads a counter so every leaf's
%  SSA names and every join-block label are unique within the function.
plawk_guard_tree_lines(and(L, R), FieldSep, Index, N0, TrueLbl, FalseLbl,
        Lines, Globals, N2) :-
    !,
    format(atom(Mid), 'grand_~w_~w', [Index, N0]),
    N1 is N0 + 1,
    plawk_guard_tree_lines(L, FieldSep, Index, N1, Mid, FalseLbl, LLines, LG, Nm),
    plawk_guard_tree_lines(R, FieldSep, Index, Nm, TrueLbl, FalseLbl, RLines, RG, N2),
    format(atom(MidLabel), '~w:', [Mid]),
    append([LLines, [MidLabel], RLines], Lines),
    append(LG, RG, Globals).
plawk_guard_tree_lines(or(L, R), FieldSep, Index, N0, TrueLbl, FalseLbl,
        Lines, Globals, N2) :-
    !,
    format(atom(Mid), 'gror_~w_~w', [Index, N0]),
    N1 is N0 + 1,
    plawk_guard_tree_lines(L, FieldSep, Index, N1, TrueLbl, Mid, LLines, LG, Nm),
    plawk_guard_tree_lines(R, FieldSep, Index, Nm, TrueLbl, FalseLbl, RLines, RG, N2),
    format(atom(MidLabel), '~w:', [Mid]),
    append([LLines, [MidLabel], RLines], Lines),
    append(LG, RG, Globals).
plawk_guard_tree_lines(guard(ColIndex, Op, Value), FieldSep, Index, N0,
        TrueLbl, FalseLbl, [LeafIR], Globals, N1) :-
    N1 is N0 + 1,
    plawk_guard_leaf_ir(guard(ColIndex, Op, Value), FieldSep, Index, N0,
        TrueLbl, FalseLbl, LeafIR, GlobalIR),
    ( GlobalIR == '' -> Globals = [] ; Globals = [GlobalIR] ).

%% plawk_guard_leaf_ir(+guard(Col,Op,Value), +FieldSep, +Index, +N,
%%     +TrueLbl, +FalseLbl, -IR, -GlobalIR)
%  One leaf comparison branching to TrueLbl (match) or FalseLbl (skip). SSA
%  names are suffixed by N so multiple leaves of one boolean guard never
%  collide. Integer -> i64 icmp; float_const -> f64 fcmp; str -> length + memcmp
%  (the memcmp runs only when the lengths match, never reading past the field).
plawk_guard_leaf_ir(guard(ColIndex, Op, Value), FieldSep, _Index, N,
        TrueLbl, FalseLbl, IR, '') :-
    integer(Value),
    !,
    plawk_icmp_pred(Op, Pred),
    format(atom(IR),
'  %rec_gv_~w = call i64 @wam_atom_field_i64_value(%Value %rec_row_v, i64 ~w, i8 ~w)
  %rec_gcmp_~w = icmp ~w i64 %rec_gv_~w, ~w
  br i1 %rec_gcmp_~w, label %~w, label %~w',
        [N, ColIndex, FieldSep, N, Pred, N, Value, N, TrueLbl, FalseLbl]).
plawk_guard_leaf_ir(guard(ColIndex, Op, float_const(M, D)), FieldSep, _Index, N,
        TrueLbl, FalseLbl, IR, '') :-
    !,
    plawk_fcmp_pred(Op, Pred),
    format(atom(IR),
'  %rec_gcst_~w = fdiv double ~w.0, ~w.0
  %rec_gv_~w = call double @wam_atom_field_f64_value(%Value %rec_row_v, i64 ~w, i8 ~w)
  %rec_gcmp_~w = fcmp ~w double %rec_gv_~w, %rec_gcst_~w
  br i1 %rec_gcmp_~w, label %~w, label %~w',
        [N, M, D, N, ColIndex, FieldSep, N, Pred, N, N, N, TrueLbl, FalseLbl]).
plawk_guard_leaf_ir(guard(ColIndex, Op, str(S)), FieldSep, Index, N,
        TrueLbl, FalseLbl, IR, GlobalIR) :-
    plawk_str_leaf_targets(Op, TrueLbl, FalseLbl, EqT, NeT, LenFailT),
    format(atom(GName), 'plawk_guard_lit_~w_~w', [Index, N]),
    llvm_emit_c_string_global(GName, S, GlobalIR, StrLen, BytesLen),
    format(atom(Gep),
        'getelementptr inbounds ([~w x i8], [~w x i8]* @.~w, i64 0, i64 0)',
        [BytesLen, BytesLen, GName]),
    format(atom(IR),
'  %rec_gslice_~w = call %WamSlice @wam_atom_field_slice_value(%Value %rec_row_v, i64 ~w, i8 ~w)
  %rec_gptr_~w = extractvalue %WamSlice %rec_gslice_~w, 0
  %rec_glen_~w = extractvalue %WamSlice %rec_gslice_~w, 1
  %rec_glen_ok_~w = icmp eq i64 %rec_glen_~w, ~w
  br i1 %rec_glen_ok_~w, label %rec_gmemcmp_~w, label %rec_glenfail_~w
rec_gmemcmp_~w:
  %rec_gmc_~w = call i32 @memcmp(i8* %rec_gptr_~w, i8* ~w, i64 ~w)
  %rec_gmc_eq_~w = icmp eq i32 %rec_gmc_~w, 0
  br i1 %rec_gmc_eq_~w, label %~w, label %~w
rec_glenfail_~w:
  br label %~w',
        [N, ColIndex, FieldSep,
         N, N,
         N, N,
         N, N, StrLen,
         N, N, N,
         N,
         N, N, Gep, StrLen,
         N, N,
         N, EqT, NeT,
         N,
         LenFailT]).

% For a string leaf, map the True/False guard targets onto the memcmp outcomes.
% `==`: equal bytes -> True, unequal or wrong length -> False. `!=`: the mirror.
plawk_str_leaf_targets(eq, TrueLbl, FalseLbl, TrueLbl, FalseLbl, FalseLbl).
plawk_str_leaf_targets(ne, TrueLbl, FalseLbl, FalseLbl, TrueLbl, TrueLbl).

% Surface comparison op -> signed LLVM icmp predicate.
plawk_icmp_pred(eq, eq).
plawk_icmp_pred(ne, ne).
plawk_icmp_pred(lt, slt).
plawk_icmp_pred(le, sle).
plawk_icmp_pred(gt, sgt).
plawk_icmp_pred(ge, sge).

%% plawk_while_cond_ok(+Cond) is semidet.
%
%  A supported loop condition: scalar comparisons (`VAR CMP int` or
%  `VAR CMP VAR`) combined with `and` / `or` (PLAWK_CONTROL_FLOW_PLAN.md PR 3).
plawk_while_cond_ok(and(A, B)) :-
    !,
    plawk_while_cond_ok(A),
    plawk_while_cond_ok(B).
plawk_while_cond_ok(or(A, B)) :-
    !,
    plawk_while_cond_ok(A),
    plawk_while_cond_ok(B).
plawk_while_cond_ok(cmp(var(_V), Op, Rhs)) :-
    plawk_icmp_pred(Op, _Pred),
    plawk_while_cond_rhs_ok(Rhs).
% RSTART/RLENGTH (i64 specials set by match()) as a comparison LHS.
plawk_while_cond_ok(cmp(special(Name), Op, Rhs)) :-
    plawk_match_special(Name),
    plawk_icmp_pred(Op, _Pred),
    plawk_while_cond_rhs_ok(Rhs).

plawk_match_special('RSTART').
plawk_match_special('RLENGTH').
plawk_match_special('NR').
plawk_match_special('ARGC').
plawk_match_special('NF').

plawk_while_cond_rhs_ok(int(N)) :- integer(N).
plawk_while_cond_rhs_ok(var(_W)).
plawk_while_cond_rhs_ok(special(Name)) :- plawk_match_special(Name).

%% plawk_while_cond_vars(+Cond, -Vars)
%
%  Every scalar variable named in the condition (both sides of every
%  comparison) -- each must get an i64 slot.
plawk_while_cond_vars(and(A, B), Vars) :-
    !,
    plawk_while_cond_vars(A, VA),
    plawk_while_cond_vars(B, VB),
    append(VA, VB, Vars).
plawk_while_cond_vars(or(A, B), Vars) :-
    !,
    plawk_while_cond_vars(A, VA),
    plawk_while_cond_vars(B, VB),
    append(VA, VB, Vars).
% RSTART/RLENGTH operands are globals, not slots, so they add no slot vars.
plawk_while_cond_vars(cmp(special(_), _Op, int(_N)), []) :- !.
plawk_while_cond_vars(cmp(special(_), _Op, var(W)), [W]) :- !.
plawk_while_cond_vars(cmp(special(_), _Op, special(_)), []) :- !.
plawk_while_cond_vars(cmp(var(V), _Op, special(_)), [V]) :- !.
plawk_while_cond_vars(cmp(var(V), _Op, int(_N)), [V]) :- !.
plawk_while_cond_vars(cmp(var(V), _Op, string(_S)), [V]) :- !.
plawk_while_cond_vars(cmp(var(V), _Op, var(W)), [V, W]).

%% plawk_while_cond_ir(+Cond, +Slots, +CondValues, +Base, -CondVar, -IR)
%
%  Emit the loop-condition test as a block of IR lines ending in an i1 value
%  (CondVar). Each variable's current value is its slot value in CondValues
%  (head-phi values for `while`, body-output values for `do-while`);
%  comparisons are i64 icmps, combined with `and`/`or i1`.
plawk_while_cond_ir(Cond, Slots, CondValues, FieldSeparator, Base, CondVar, IR) :-
    plawk_while_cond_build(Cond, Slots, CondValues, FieldSeparator, Base, '', CondVar, Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_while_cond_build(and(A, B), Slots, CV, FS, Base, Path, CondVar, Lines) :-
    !,
    atom_concat(Path, 'l', PA),
    atom_concat(Path, 'r', PB),
    plawk_while_cond_build(A, Slots, CV, FS, Base, PA, VA, LA),
    plawk_while_cond_build(B, Slots, CV, FS, Base, PB, VB, LB),
    format(atom(CondVar), '%~w_cond~w', [Base, Path]),
    format(atom(Line), '  ~w = and i1 ~w, ~w', [CondVar, VA, VB]),
    append([LA, LB, [Line]], Lines).
plawk_while_cond_build(or(A, B), Slots, CV, FS, Base, Path, CondVar, Lines) :-
    !,
    atom_concat(Path, 'l', PA),
    atom_concat(Path, 'r', PB),
    plawk_while_cond_build(A, Slots, CV, FS, Base, PA, VA, LA),
    plawk_while_cond_build(B, Slots, CV, FS, Base, PB, VB, LB),
    format(atom(CondVar), '%~w_cond~w', [Base, Path]),
    format(atom(Line), '  ~w = or i1 ~w, ~w', [CondVar, VA, VB]),
    append([LA, LB, [Line]], Lines).
% NF (the field count of the current record) as a comparison operand. Computed
% from %line via the field-count runtime (the field separator threaded in), so
% it is only valid where a record is in scope -- rejected in an END condition
% (Base is `plawk_endif...`), leaving `NF` in END a clean not-yet. Handles NF on
% either side; the RHS is an integer literal or a loop variable.
plawk_while_cond_build(cmp(special('NF'), Op, Rhs), Slots, CondValues, FS, Base,
        Path, CondVar, Lines) :-
    \+ sub_atom(Base, 0, _, _, plawk_endif),
    !,
    format(atom(NfBase), '~w_nf~w', [Base, Path]),
    llvm_emit_atom_field_count('%line', FS, NfBase, CountLine),
    plawk_while_cond_operand(Rhs, Slots, CondValues, Base, Path, r, ROperand, RLines),
    plawk_icmp_pred(Op, Pred),
    format(atom(CondVar), '%~w_cond~w', [Base, Path]),
    format(atom(Line), '  ~w = icmp ~w i64 %~w, ~w', [CondVar, Pred, NfBase, ROperand]),
    append([[CountLine], RLines, [Line]], Lines).
plawk_while_cond_build(cmp(Lhs, Op, special('NF')), Slots, CondValues, FS, Base,
        Path, CondVar, Lines) :-
    \+ sub_atom(Base, 0, _, _, plawk_endif),
    !,
    format(atom(NfBase), '~w_nf~w', [Base, Path]),
    llvm_emit_atom_field_count('%line', FS, NfBase, CountLine),
    plawk_while_cond_operand(Lhs, Slots, CondValues, Base, Path, l, LOperand, LLines),
    plawk_icmp_pred(Op, Pred),
    format(atom(CondVar), '%~w_cond~w', [Base, Path]),
    format(atom(Line), '  ~w = icmp ~w i64 ~w, %~w', [CondVar, Pred, LOperand, NfBase]),
    append([[CountLine], LLines, [Line]], Lines).
% strnum-vs-integer-literal comparison (step 3b): resolve the strnum id to text
% and dispatch to @wam_strnum_cmp_int, which decides numeric (strnum looks
% numeric) vs lexical (against the integer formatted as a decimal string) by the
% runtime content, then compare its sign to 0. `x OP N`.
plawk_while_cond_build(cmp(var(L), Op, int(N)), Slots, CondValues, _FS, Base,
        Path, CondVar, Lines) :-
    plawk_cond_var_is_strnum(L, Slots),
    integer(N),
    !,
    plawk_while_cond_operand(var(L), Slots, CondValues, Base, Path, l, LId, _),
    plawk_icmp_pred(Op, Pred),
    plawk_strnum_cmp_int_lines(Base, Path, LId, N, Pred, CondVar, Lines).
% `N OP x`: swap to `x swap(Op) N` so the strnum stays the left operand.
plawk_while_cond_build(cmp(int(N), Op, var(R)), Slots, CondValues, _FS, Base,
        Path, CondVar, Lines) :-
    plawk_cond_var_is_strnum(R, Slots),
    integer(N),
    !,
    plawk_while_cond_operand(var(R), Slots, CondValues, Base, Path, r, RId, _),
    plawk_swap_cmp_op(Op, SwappedOp),
    plawk_icmp_pred(SwappedOp, Pred),
    plawk_strnum_cmp_int_lines(Base, Path, RId, N, Pred, CondVar, Lines).
% strnum-vs-strnum comparison (POSIX numeric-string duality): both operands are
% strnum slots (interned atom ids). Resolve each id to its text and dispatch to
% @wam_strnum_cmp, which decides numeric-vs-lexical by the runtime content (both
% look numeric -> numeric; otherwise strcmp), then compare its sign to 0 with the
% surface comparison predicate. This is the "10 9" (numeric) vs "10 9x" (lexical)
% fix. Tried before the generic i64 icmp clause.
plawk_while_cond_build(cmp(var(L), Op, var(R)), Slots, CondValues, _FS, Base,
        Path, CondVar, Lines) :-
    plawk_cond_var_is_strnum(L, Slots),
    plawk_cond_var_is_strnum(R, Slots),
    !,
    plawk_while_cond_operand(var(L), Slots, CondValues, Base, Path, l, LId, _),
    plawk_while_cond_operand(var(R), Slots, CondValues, Base, Path, r, RId, _),
    plawk_icmp_pred(Op, Pred),
    format(atom(CondVar), '%~w_cond~w', [Base, Path]),
    format(atom(LineLS),
        '  %~w_sn~w_ls = call i8* @wam_atom_to_string(i64 ~w)', [Base, Path, LId]),
    format(atom(LineRS),
        '  %~w_sn~w_rs = call i8* @wam_atom_to_string(i64 ~w)', [Base, Path, RId]),
    format(atom(LineRC),
        '  %~w_sn~w_rc = call i32 @wam_strnum_cmp(i8* %~w_sn~w_ls, i8 1, i8* %~w_sn~w_rs, i8 1)',
        [Base, Path, Base, Path, Base, Path]),
    format(atom(LineCmp),
        '  ~w = icmp ~w i32 %~w_sn~w_rc, 0', [CondVar, Pred, Base, Path]),
    Lines = [LineLS, LineRS, LineRC, LineCmp].
plawk_while_cond_build(cmp(Left, Op, Rhs), Slots, CondValues, _FS, Base,
        Path, CondVar, Lines) :-
    % Fail-closed: if either operand is a strnum slot and this is not one of the
    % supported strnum cases above (strnum-vs-strnum, strnum-vs-integer-literal),
    % refuse to emit a raw i64 icmp on an atom id (which would silently
    % mis-compare). The clause fails, the scalar driver declines the program, and
    % it falls back -- strnum-vs-non-strnum-var and strnum-vs-float are follow-ons.
    \+ plawk_cmp_has_strnum_operand(Left, Rhs, Slots),
    plawk_while_cond_operand(Left, Slots, CondValues, Base, Path, l, LOperand, LLines),
    plawk_while_cond_operand(Rhs, Slots, CondValues, Base, Path, r, ROperand, RLines),
    plawk_icmp_pred(Op, Pred),
    format(atom(CondVar), '%~w_cond~w', [Base, Path]),
    format(atom(Line), '  ~w = icmp ~w i64 ~w, ~w',
        [CondVar, Pred, LOperand, ROperand]),
    append([LLines, RLines, [Line]], Lines).

plawk_cond_var_is_strnum(Name, Slots) :-
    memberchk(scalar_strnum(Name), Slots).

plawk_cmp_has_strnum_operand(Left, Rhs, Slots) :-
    ( Left = var(L), plawk_cond_var_is_strnum(L, Slots)
    ; Rhs = var(R), plawk_cond_var_is_strnum(R, Slots)
    ).

% Emit the strnum-vs-integer comparison: resolve the strnum id to text, call
% @wam_strnum_cmp_int(text, strnum-kind 1, N), and test its sign against 0 with
% Pred. Id is the strnum slot's current SSA value.
plawk_strnum_cmp_int_lines(Base, Path, Id, N, Pred, CondVar, Lines) :-
    format(atom(CondVar), '%~w_cond~w', [Base, Path]),
    format(atom(LineS),
        '  %~w_sni~w_s = call i8* @wam_atom_to_string(i64 ~w)', [Base, Path, Id]),
    format(atom(LineRC),
        '  %~w_sni~w_rc = call i32 @wam_strnum_cmp_int(i8* %~w_sni~w_s, i8 1, i64 ~w)',
        [Base, Path, Base, Path, N]),
    format(atom(LineCmp),
        '  ~w = icmp ~w i32 %~w_sni~w_rc, 0', [CondVar, Pred, Base, Path]),
    Lines = [LineS, LineRC, LineCmp].

% Swap a comparison operator for operand-order reversal (`N OP x` -> `x OP' N`).
plawk_swap_cmp_op(eq, eq).
plawk_swap_cmp_op(ne, ne).
plawk_swap_cmp_op(lt, gt).
plawk_swap_cmp_op(gt, lt).
plawk_swap_cmp_op(le, ge).
plawk_swap_cmp_op(ge, le).

% An operand is an integer literal (emitted inline), a loop variable (read from
% its slot's current SSA value), or an RSTART/RLENGTH special (loaded from its
% global -- the load line is threaded out). Side (l/r) keeps the load SSA name
% unique within a comparison.
plawk_while_cond_operand(int(N), _Slots, _CondValues, _Base, _Path, _Side, N, []) :-
    !.
plawk_while_cond_operand(special('RSTART'), _Slots, _CondValues, Base, Path, Side, Ref, [Line]) :-
    !,
    format(atom(Ref), '%~w_cond~w_~w_rstart', [Base, Path, Side]),
    format(atom(Line), '  ~w = load i64, i64* @plawk_rstart', [Ref]).
plawk_while_cond_operand(special('RLENGTH'), _Slots, _CondValues, Base, Path, Side, Ref, [Line]) :-
    !,
    format(atom(Ref), '%~w_cond~w_~w_rlength', [Base, Path, Side]),
    format(atom(Line), '  ~w = load i64, i64* @plawk_rlength', [Ref]).
% ARGC: the command-line argument count -- a call, no per-record state.
plawk_while_cond_operand(special('ARGC'), _Slots, _CondValues, Base, Path, Side, Ref, [Line]) :-
    !,
    format(atom(Ref), '%~w_cond~w_~w_argc', [Base, Path, Side]),
    format(atom(Line), '  ~w = call i64 @wam_argc()', [Ref]).
% NR: the current record number. In a rule-body condition it is the loop's
% %current_nr; in an END condition it is %plawk_nr (the final record count, the
% same SSA the END expression path reads). No extra line -- both are existing
% SSA values in scope. The END base is plawk_endif (see plawk_end_if lowering).
plawk_while_cond_operand(special('NR'), _Slots, _CondValues, Base, _Path, _Side, Ref, []) :-
    !,
    ( sub_atom(Base, 0, _, _, plawk_endif)
    -> Ref = '%plawk_nr'
    ;  Ref = '%current_nr'
    ).
plawk_while_cond_operand(var(Name), Slots, CondValues, _Base, _Path, _Side, Ref, []) :-
    nth0(Idx, Slots, Slot),
    plawk_slot_name(Slot, Name),
    !,
    nth0(Idx, CondValues, Ref).

%% plawk_if_cond_ir(+Cond, +Slots, +Values0, +FieldSeparator, +GlobalBase,
%%     -CondValue, -GuardGlobalIR-GuardIR)
%
%  Lower an `if` condition to an i1 (CondValue). A `scalar_if(_)` condition is a
%  scalar comparison over slots -- lowered like a loop condition, reading the
%  current slot SSA values (Values0). A field/pattern guard is lowered by the
%  existing pattern-guard emitter (which reads the record).
% String-equality guard `if (s == "text")` / `!=` on a string scalar: intern the
% literal and compare atom ids (interning is canonical, so equal strings share an
% id). Only == / != (ordering would need strcmp). A single comparison, not
% combined with && / || (that stays a clean codegen error) -- v1.
plawk_if_cond_ir(scalar_if(cmp(var(Name), Op, string(Value))), Slots, Values0,
        _FieldSeparator, GlobalBase, CondValue, GlobalIR-IR) :-
    memberchk(Op, [eq, ne]),
    !,
    nth0(Idx, Slots, Slot),
    plawk_slot_name(Slot, Name),
    !,
    nth0(Idx, Values0, SlotValue),
    format(atom(LitName), '~w_lit', [GlobalBase]),
    llvm_emit_c_string_global(LitName, Value, GlobalIR, Len, BytesLen),
    plawk_icmp_pred(Op, Pred),
    format(atom(CondValue), '%~w_cond', [GlobalBase]),
    format(atom(IR),
'  %~w_litptr = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0
  %~w_litid = call i64 @wam_intern_atom(i8* %~w_litptr, i64 ~w)
  %~w_cond = icmp ~w i64 ~w, %~w_litid',
        [GlobalBase, BytesLen, BytesLen, LitName,
         GlobalBase, GlobalBase, Len,
         GlobalBase, Pred, SlotValue, GlobalBase]).
% String-ordering guard `if (s < "text")` / `<=` / `>` / `>=` on a string scalar:
% atom ids are not ordered by string value, so resolve the scalar's id to text
% and `strcmp` against the literal, then compare the result to 0. An unset scalar
% (id 0) resolves to empty via the literal global's trailing NUL.
plawk_if_cond_ir(scalar_if(cmp(var(Name), Op, string(Value))), Slots, Values0,
        _FieldSeparator, GlobalBase, CondValue, GlobalIR-IR) :-
    memberchk(Op, [lt, le, gt, ge]),
    !,
    nth0(Idx, Slots, Slot),
    plawk_slot_name(Slot, Name),
    !,
    nth0(Idx, Values0, SlotValue),
    format(atom(LitName), '~w_lit', [GlobalBase]),
    llvm_emit_c_string_global(LitName, Value, GlobalIR, Len, BytesLen),
    plawk_icmp_pred(Op, Pred),
    format(atom(CondValue), '%~w_cond', [GlobalBase]),
    format(atom(IR),
'  %~w_litptr = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0
  %~w_sraw = call i8* @wam_atom_to_string(i64 ~w)
  %~w_empty = icmp eq i64 ~w, 0
  %~w_sptr = select i1 %~w_empty, i8* getelementptr ([~w x i8], [~w x i8]* @.~w, i64 0, i64 ~w), i8* %~w_sraw
  %~w_scmp = call i32 @strcmp(i8* %~w_sptr, i8* %~w_litptr)
  %~w_cond = icmp ~w i32 %~w_scmp, 0',
        [GlobalBase, BytesLen, BytesLen, LitName,
         GlobalBase, SlotValue,
         GlobalBase, SlotValue,
         GlobalBase, GlobalBase, BytesLen, BytesLen, LitName, Len, GlobalBase,
         GlobalBase, GlobalBase, GlobalBase,
         GlobalBase, Pred, GlobalBase]).
plawk_if_cond_ir(scalar_if(Cond), Slots, Values0, FieldSeparator, GlobalBase,
        CondValue, ''-GuardIR) :-
    !,
    plawk_while_cond_ir(Cond, Slots, Values0, FieldSeparator, GlobalBase, CondValue, GuardIR).
plawk_if_cond_ir(Pattern, _Slots, _Values0, FieldSeparator, GlobalBase,
        CondValue, GuardGlobalIR-GuardIR) :-
    format(atom(CondValue), '%~w_cond', [GlobalBase]),
    plawk_pattern_guard_ir(Pattern, FieldSeparator, GlobalBase, CondValue,
        GuardGlobalIR-GuardIR).

% Surface comparison op -> ordered LLVM fcmp predicate (row values are finite,
% so ordered comparisons are correct; a NaN column fails every guard).
plawk_fcmp_pred(eq, oeq).
plawk_fcmp_pred(ne, one).
plawk_fcmp_pred(lt, olt).
plawk_fcmp_pred(le, ole).
plawk_fcmp_pred(gt, ogt).
plawk_fcmp_pred(ge, oge).

%% plawk_records_col_lines(+FieldPlans, +FieldSep, +OutputSep, +PrintIndex)//
%  Per-field print for the row readers: a separator before each field after
%  the first, then either a plain column col(Index) -- extract field Index of
%  the row (%rec_row_v) and print its text (%.*s) -- or an arithmetic
%  expression arith(F64Op, L, R) over columns/constants, evaluated in f64 (a
%  column via @wam_atom_field_f64_value, a constant via sitofp) and printed
%  with %g.
plawk_records_col_lines([], _FieldSep, _OutputSep, _PrintIndex) --> [].
plawk_records_col_lines([Plan | Rest], FieldSep, OutputSep, PrintIndex) -->
    { ( PrintIndex =:= 0
      ->  SepLines = []
      ;   format(atom(SepLine),
              '  %rec_sep~w = call i32 @putchar(i32 ~w)', [PrintIndex, OutputSep]),
          SepLines = [SepLine]
      ),
      plawk_records_field_lines(Plan, PrintIndex, FieldSep, FieldLines),
      append([SepLines, FieldLines], Lines),
      NextPrintIndex is PrintIndex + 1
    },
    plawk_emit_lines(Lines),
    plawk_records_col_lines(Rest, FieldSep, OutputSep, NextPrintIndex).

% A plain column: extract its slice from the row and print as text.
plawk_records_field_lines(col(ColIndex), PrintIndex, FieldSep, Lines) :-
    format(atom(Slice),
        '  %rec_f~w_slice = call %WamSlice @wam_atom_field_slice_value(%Value %rec_row_v, i64 ~w, i8 ~w)',
        [PrintIndex, ColIndex, FieldSep]),
    format(atom(Ptr), '  %rec_f~w_ptr = extractvalue %WamSlice %rec_f~w_slice, 0',
        [PrintIndex, PrintIndex]),
    format(atom(Len), '  %rec_f~w_len = extractvalue %WamSlice %rec_f~w_slice, 1',
        [PrintIndex, PrintIndex]),
    format(atom(Lenw), '  %rec_f~w_lenw = trunc i64 %rec_f~w_len to i32',
        [PrintIndex, PrintIndex]),
    format(atom(Fmt),
        '  %rec_f~w_fmt = getelementptr [5 x i8], [5 x i8]* @.plawk_surface_print_slice, i32 0, i32 0',
        [PrintIndex]),
    format(atom(Pr),
        '  %rec_f~w_pr = call i32 (i8*, ...) @printf(i8* %rec_f~w_fmt, i32 %rec_f~w_lenw, i8* %rec_f~w_ptr)',
        [PrintIndex, PrintIndex, PrintIndex, PrintIndex]),
    Lines = [Slice, Ptr, Len, Lenw, Fmt, Pr].
% Arithmetic over columns, in f64, printed with %g.
plawk_records_field_lines(arith(F64Op, LPlan, RPlan), PrintIndex, FieldSep, Lines) :-
    format(atom(B), 'rec_f~w', [PrintIndex]),
    plawk_records_operand_lines(LPlan, B, l, FieldSep, LVar, LLines),
    plawk_records_operand_lines(RPlan, B, r, FieldSep, RVar, RLines),
    format(atom(OpL), '  %~w_res = ~w double ~w, ~w', [B, F64Op, LVar, RVar]),
    format(atom(FmtL),
        '  %~w_fmt = getelementptr [3 x i8], [3 x i8]* @.plawk_surface_print_f64, i32 0, i32 0',
        [B]),
    format(atom(PrL),
        '  %~w_pr = call i32 (i8*, ...) @printf(i8* %~w_fmt, double %~w_res)', [B, B, B]),
    append([LLines, RLines, [OpL, FmtL, PrL]], Lines).

% A row-arithmetic operand as a double: a column read as f64, or a constant.
plawk_records_operand_lines(col(ColIndex), B, Slot, FieldSep, ValVar, [Line]) :-
    format(atom(ValVar), '%~w_~w', [B, Slot]),
    format(atom(Line),
        '  ~w = call double @wam_atom_field_f64_value(%Value %rec_row_v, i64 ~w, i8 ~w)',
        [ValVar, ColIndex, FieldSep]).
plawk_records_operand_lines(aint(V), B, Slot, _FieldSep, ValVar, [Line]) :-
    format(atom(ValVar), '%~w_~w', [B, Slot]),
    format(atom(Line), '  ~w = sitofp i64 ~w to double', [ValVar, V]).

%% plawk_multipass_pass_fn_ir(+Index, +RecordIR, -IR)
%  One pass as a self-contained function: open the shared input path, loop
%  reading transient line records, run RecordIR per record (it branches to
%  %continue_loop and references the %plawk_assoc_table_0 parameter), and
%  return. Function-local labels/SSA, so passes never collide.
plawk_multipass_pass_fn_ir(Index, TableParamsIR, RecordIR, IR) :-
    format(atom(IR),
'define void @plawk_pass_~w(%Value %mp_path~w) {
entry:
  %handle = call %Value @wam_stream_open_value(%Value %mp_path)
  %handle_tag = extractvalue %Value %handle, 0
  %handle_is_int = icmp eq i32 %handle_tag, 1
  br i1 %handle_is_int, label %check_handle, label %ret_void

check_handle:
  %handle_payload = extractvalue %Value %handle, 1
  %handle_ok = icmp sgt i64 %handle_payload, 0
  br i1 %handle_ok, label %loop, label %ret_void

loop:
  %line = call %Value @wam_stream_read_line_transient_value(%Value %handle)
  %line_tag = extractvalue %Value %line, 0
  %line_payload = extractvalue %Value %line, 1
  %line_is_int = icmp eq i32 %line_tag, 1
  %line_bad_payload = icmp slt i64 %line_payload, 0
  %line_bad = and i1 %line_is_int, %line_bad_payload
  br i1 %line_bad, label %close_stream, label %check_line_atom

check_line_atom:
  %line_is_atom = icmp eq i32 %line_tag, 0
  br i1 %line_is_atom, label %check_eof, label %close_stream

check_eof:
  %line_s = call i8* @wam_atom_to_string(i64 %line_payload)
  %eof_s = getelementptr [12 x i8], [12 x i8]* @.wam_stream_eof, i32 0, i32 0
  %eof_cmp = call i32 @strcmp(i8* %line_s, i8* %eof_s)
  %is_eof = icmp eq i32 %eof_cmp, 0
  br i1 %is_eof, label %close_stream, label %lowered_assoc

lowered_assoc:
~w

continue_loop:
  br label %loop

close_stream:
  %close_ok = call i1 @wam_stream_close_value(%Value %handle)
  br label %ret_void

ret_void:
  ret void
}
', [Index, TableParamsIR, RecordIR]).

%% plawk_forin_end_plan(+Rules, +LoopVar, +ArrayName, +BodyActions, -AssocPlan, -PrintFields)
%
%  Plan the first END for-in surface: rules are associative-count updates and
%  the loop body is one print whose fields are the loop key, associative
%  lookups keyed by the loop variable, or string literals.
plawk_forin_end_plan(Rules, LoopVar, ArrayName, BodyActions,
        AssocPlan, PrintFields) :-
    BodyActions = [print(PrintFields)],
    PrintFields = [_ | _],
    maplist(plawk_forin_print_field(LoopVar), PrintFields),
    findall(LookupArrayName,
        member(assoc(var(LookupArrayName), var(LoopVar)), PrintFields),
        LookupArrays),
    plawk_forin_assoc_plan(Rules, ArrayName, LookupArrays, AssocPlan).
% guarded END for-in (assoc for-in stage 1b): the body is a guarded
% print, so the plan is the same table set but the guard gates output.
plawk_forin_end_plan(Rules, LoopVar, ArrayName,
        [if(Guard, [print(PrintFields)], [])], AssocPlan, PrintFields) :-
    plawk_forin_guard_ok(Guard, LoopVar, ArrayName),
    PrintFields = [_ | _],
    maplist(plawk_forin_print_field(LoopVar), PrintFields),
    findall(LookupArrayName,
        member(assoc(var(LookupArrayName), var(LoopVar)), PrintFields),
        LookupArrays),
    plawk_forin_assoc_plan(Rules, ArrayName, LookupArrays, AssocPlan).

%% plawk_forin_end_accum_plan(+Rules, +ArrayName, +Operand, -AssocPlan)
%  Plan the END accumulate for-in (stage 2): the record rules populate the
%  hash, the loop folds it into a scalar. The accumulator itself is
%  loop-carried (not a hash table), so the only tables are the iterated
%  array and whatever the rules touch -- no print-lookup arrays.
plawk_forin_end_accum_plan(Rules, ArrayName, Operand, AssocPlan) :-
    plawk_forin_accum_operand_ok(Operand, ArrayName),
    plawk_forin_assoc_plan(Rules, ArrayName, [], AssocPlan).

%% plawk_forin_accum_operand_ok(+Operand, +ArrayName)
%  A for-in accumulate operand is the iterated value `arr[k]` (array must
%  be the loop's own table), the loop key `k`, or an integer literal.
plawk_forin_accum_operand_ok(forin_val(ArrayName), ArrayName).
plawk_forin_accum_operand_ok(forin_key, _ArrayName).
plawk_forin_accum_operand_ok(int(_Value), _ArrayName).

%% plawk_forin_end_decode_plan(+Rules, +ArrayName, -AssocPlan)
%  Plan the END decode for-in (stage 3): the record rules populate the hash,
%  the loop destructures each value. The decoded fields are for-in-scoped
%  (loaded fresh each iteration), so the only tables are the iterated array
%  and whatever the rules touch -- like the accumulate plan.
plawk_forin_end_decode_plan(Rules, ArrayName, AssocPlan) :-
    plawk_forin_assoc_plan(Rules, ArrayName, [], AssocPlan).

plawk_forin_assoc_plan(Rules, ArrayName, LookupArrays,
        assoc_plan(Tables, PlannedRules)) :-
    maplist(plawk_assoc_rule_action_specs, Rules, RuleSpecs),
    RuleSpecs \== [],
    findall(RuleArrayName,
        ( member(rule(_Pattern, ActionSpecs, _Control), RuleSpecs),
          ( member(RuleArrayName-_KeyIndex, ActionSpecs)
          ; member(dynassoc(RuleArrayName, _Call), ActionSpecs)
          ; member(assoc_delete(RuleArrayName, _KeyIndex), ActionSpecs)
          ; member(assoc_split(RuleArrayName, _Ki, _Sep), ActionSpecs)
          ; plawk_assoc_spec_forin_array(ActionSpecs, RuleArrayName)
          )
        ),
        ActionArrays),
    append([ActionArrays, [ArrayName], LookupArrays], ArrayNames0),
    sort(ArrayNames0, Tables),
    plawk_assoc_specs_str_arrays(RuleSpecs, StrArrays),
    plawk_assoc_specs_posarray_arrays(RuleSpecs, PosArrays),
    phrase(plawk_assoc_planned_rules(RuleSpecs, Tables, StrArrays, PosArrays,
        0), PlannedRules).

%% plawk_assoc_spec_forin_array(+ActionSpecs, -ArrayName)
%  Arrays a rule-body for-in touches: the iterated table and every
%  lookup array in its print fields -- all need table slots.
plawk_assoc_spec_forin_array(ActionSpecs, ArrayName) :-
    member(forin(_LoopVar, FA, FFields), ActionSpecs),
    ( ArrayName = FA
    ; member(assoc(var(ArrayName), var(_)), FFields)
    ).
plawk_assoc_spec_forin_array(ActionSpecs, ArrayName) :-
    member(forin_guarded(_LoopVar, FA, _Guard, FFields), ActionSpecs),
    ( ArrayName = FA
    ; member(assoc(var(ArrayName), var(_)), FFields)
    ).

%% plawk_assoc_specs_str_arrays(+RuleSpecs, -StrArrays)
%  Names of tables populated through `as assoc(str)` binds -- their
%  values are atom-registry ids, so for-in value reads resolve to text.
plawk_assoc_specs_str_arrays(RuleSpecs, StrArrays) :-
    findall(A,
        ( member(rule(_P, ActionSpecs, _C), RuleSpecs),
          ( member(dynassoc(A, str(_Call)), ActionSpecs)
          ; member(dynassoc(A, posarray_str(_Call)), ActionSpecs)
          ; member(assoc_set_row(A, _Key), ActionSpecs)   % row-valued (str)
          ; member(assoc_set_row_cons(A, _Key2, _Fs), ActionSpecs)  % row (str)
          ; member(assoc_split(A, _Ski, _Ssep), ActionSpecs)   % split pieces (str)
          )
        ),
        As0),
    sort(As0, StrArrays).

%% plawk_assoc_specs_posarray_arrays(+RuleSpecs, -PosArrays)
%  Names of tables populated through `as array` (positional) binds --
%  their keys are integer positions, so for-in loop-key reads print
%  numerically rather than resolving an atom-registry id to text.
plawk_assoc_specs_posarray_arrays(RuleSpecs, PosArrays) :-
    findall(A,
        ( member(rule(_P, ActionSpecs, _C), RuleSpecs),
          ( member(dynassoc(A, posarray(_Call)), ActionSpecs)
          ; member(dynassoc(A, posarray_str(_Call)), ActionSpecs)
          ; member(assoc_split(A, _Pki, _Psep), ActionSpecs)   % keys are 1..n positions
          )
        ),
        As0),
    sort(As0, PosArrays).

plawk_forin_print_field(LoopVar, var(LoopVar)).
plawk_forin_print_field(LoopVar, assoc(var(_ArrayName), var(LoopVar))).
plawk_forin_print_field(_LoopVar, string(_Value)).

plawk_forin_writebin_field(LoopVar, var(LoopVar)).
plawk_forin_writebin_field(LoopVar, assoc(var(_ArrayName), var(LoopVar))).
plawk_forin_writebin_field(_LoopVar, int(Value)) :-
    integer(Value).

%% plawk_forin_end_writebin_ir(+LoopVar, +ArrayName, +OutTypes, +Fields,
%%     +AssocPlan, -IR)
%
%  writebin variant of the END for-in loop: walk the iterated table's
%  occupied slots and emit one fixed-layout binary record per group
%  (raw i64 keys, table values, or literals; i64 values promote via
%  sitofp into f64 output slots), then free every table and return.
plawk_forin_end_writebin_ir(LoopVar, ArrayName, OutTypes, Fields, AssocPlan, IR) :-
    plawk_assoc_table_index(AssocPlan, ArrayName, TableIndex),
    plawk_binfmt_record_size(binfmt(OutTypes), Size),
    format(atom(BasePtr), '%forin_wb_base', []),
    format(atom(BaseLine),
        '  ~w = getelementptr inbounds [~w x i8], [~w x i8]* %plawk_wbuf, i32 0, i32 0',
        [BasePtr, Size, Size]),
    plawk_forin_writebin_field_lines(OutTypes, Fields, LoopVar, ArrayName,
        TableIndex, AssocPlan, BasePtr, 0, 0, FieldLines),
    format(atom(StdoutLoad),
        '  %forin_wb_stdout = load i8*, i8** @stdout', []),
    format(atom(WriteCall),
        '  %forin_wb_wr = call i64 @fwrite(i8* ~w, i64 ~w, i64 1, i8* %forin_wb_stdout)',
        [BasePtr, Size]),
    append([[BaseLine], FieldLines, [StdoutLoad, WriteCall]], BodyLines),
    atomic_list_concat(BodyLines, '\n', BodyIR),
    phrase(plawk_assoc_free_lines(AssocPlan), FreeLines),
    atomic_list_concat(FreeLines, '\n', FreeIR),
    format(atom(IR),
'  br label %forin_head

forin_head:
  %forin_idx = phi i64 [0, %end_print], [%forin_next_idx, %forin_body_done]
  %forin_slot = call i64 @wam_assoc_i64_iter_next(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_idx)
  %forin_done = icmp slt i64 %forin_slot, 0
  br i1 %forin_done, label %forin_after, label %forin_body

forin_body:
  %forin_key_id = call i64 @wam_assoc_i64_key_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)
~w
  br label %forin_body_done

forin_body_done:
  %forin_next_idx = add i64 %forin_slot, 1
  br label %forin_head

forin_after:
~w
  ret i32 0',
        [TableIndex, TableIndex, BodyIR, FreeIR]).

plawk_forin_writebin_field_lines([], [], _LoopVar, _ArrayName, _TableIndex,
        _AssocPlan, _BasePtr, _Index, _Offset, []).
plawk_forin_writebin_field_lines([OutType | OutTypes], [Field | Fields], LoopVar,
        ArrayName, TableIndex, AssocPlan, BasePtr, Index, Offset, Lines) :-
    format(atom(Base), 'forin_wb_f~w', [Index]),
    plawk_forin_writebin_value_lines(Field, LoopVar, ArrayName, TableIndex,
        AssocPlan, Base, ValueIR, ValueLines),
    ( OutType == i64
    ->  StoreValueIR = ValueIR,
        LLVMType = i64,
        PromoteLines = []
    ;   format(atom(StoreValueIR), '%~w_f64p', [Base]),
        format(atom(Promote), '  ~w = sitofp i64 ~w to double',
            [StoreValueIR, ValueIR]),
        LLVMType = double,
        PromoteLines = [Promote]
    ),
    format(atom(Gep),
        '  %~w_fp = getelementptr i8, i8* ~w, i64 ~w', [Base, BasePtr, Offset]),
    format(atom(Cast),
        '  %~w_tp = bitcast i8* %~w_fp to ~w*', [Base, Base, LLVMType]),
    format(atom(Store),
        '  store ~w ~w, ~w* %~w_tp, align 1',
        [LLVMType, StoreValueIR, LLVMType, Base]),
    plawk_binfmt_type_width(OutType, Width),
    NextOffset is Offset + Width,
    NextIndex is Index + 1,
    plawk_forin_writebin_field_lines(OutTypes, Fields, LoopVar, ArrayName,
        TableIndex, AssocPlan, BasePtr, NextIndex, NextOffset, RestLines),
    append([ValueLines, PromoteLines, [Gep, Cast, Store], RestLines], Lines).

plawk_forin_writebin_value_lines(var(LoopVar), LoopVar, _ArrayName, _TableIndex,
        _AssocPlan, _Base, '%forin_key_id', []).
plawk_forin_writebin_value_lines(assoc(var(LookupArrayName), var(LoopVar)),
        LoopVar, ArrayName, TableIndex, AssocPlan, Base, ValueIR, [Line]) :-
    format(atom(ValueIR), '%~w_val', [Base]),
    (   LookupArrayName == ArrayName
    ->  format(atom(Line),
            '  ~w = call i64 @wam_assoc_i64_value_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)',
            [ValueIR, TableIndex])
    ;   plawk_assoc_table_index(AssocPlan, LookupArrayName, LookupTableIndex),
        format(atom(Line),
            '  ~w = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_key_id)',
            [ValueIR, LookupTableIndex])
    ).
plawk_forin_writebin_value_lines(int(Value), _LoopVar, _ArrayName, _TableIndex,
        _AssocPlan, _Base, Value, []) :-
    integer(Value).

%% plawk_forin_end_print_ir(+LoopVar, +ArrayName, +PrintFields, +AssocPlan,
%%     +OutputSeparator, -IR)
%
%  Emit the native END for-in loop: walk the iterated table's occupied slots
%  through @wam_assoc_i64_iter_next, print each record's fields, then free
%  every table and return.
plawk_forin_end_print_ir(LoopVar, ArrayName, PrintFields, AssocPlan,
        Descriptor, OutputSeparator, IR) :-
    plawk_assoc_table_index(AssocPlan, ArrayName, TableIndex),
    phrase(plawk_forin_body_print_lines(PrintFields, LoopVar, ArrayName,
        TableIndex, AssocPlan, Descriptor, OutputSeparator, 0), BodyLines),
    atomic_list_concat(BodyLines, '\n', BodyIR),
    phrase(plawk_assoc_free_lines(AssocPlan), FreeLines),
    atomic_list_concat(FreeLines, '\n', FreeIR),
    format(atom(IR),
'  br label %forin_head

forin_head:
  %forin_idx = phi i64 [0, %end_print], [%forin_next_idx, %forin_body_done]
  %forin_slot = call i64 @wam_assoc_i64_iter_next(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_idx)
  %forin_done = icmp slt i64 %forin_slot, 0
  br i1 %forin_done, label %forin_after, label %forin_body

forin_body:
  %forin_key_id = call i64 @wam_assoc_i64_key_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)
~w
  %forin_printed_newline = call i32 @putchar(i32 10)
  br label %forin_body_done

forin_body_done:
  %forin_next_idx = add i64 %forin_slot, 1
  br label %forin_head

forin_after:
~w
  ret i32 0',
        [TableIndex, TableIndex, BodyIR, FreeIR]).

%% plawk_forin_end_guarded_print_ir(+LoopVar, +ArrayName, +Guard,
%%     +PrintFields, +AssocPlan, +Descriptor, +OutputSeparator, -IR, -GuardGlobal)
%  Guarded END for-in (stage 1b filter): same slot-walk, but a guard on
%  the key or the iterated value gates the per-key print. Passing entries
%  print; the rest branch straight to the loop-continue. GuardGlobal is any
%  module-level constant the guard needs (a string literal), threaded to the
%  module top by the caller; '' when none.
plawk_forin_end_guarded_print_ir(LoopVar, ArrayName, Guard, PrintFields,
        AssocPlan, Descriptor, OutputSeparator, IR, GuardGlobal) :-
    plawk_assoc_table_index(AssocPlan, ArrayName, TableIndex),
    phrase(plawk_forin_body_print_lines(PrintFields, LoopVar, ArrayName,
        TableIndex, AssocPlan, Descriptor, OutputSeparator, 0), BodyLines),
    atomic_list_concat(BodyLines, '\n', BodyIR),
    phrase(plawk_assoc_free_lines(AssocPlan), FreeLines),
    atomic_list_concat(FreeLines, '\n', FreeIR),
    ( plawk_assoc_plan_str_array(AssocPlan, ArrayName) -> IsStr = true ; IsStr = false ),
    plawk_forin_guard_plan(Guard, IsStr, GuardPlan),
    plawk_forin_end_guard_lines(GuardPlan, TableIndex, GuardLines, CondVar, GuardGlobal),
    atomic_list_concat(GuardLines, '\n', GuardIR),
    format(atom(IR),
'  br label %forin_head

forin_head:
  %forin_idx = phi i64 [0, %end_print], [%forin_next_idx, %forin_body_done]
  %forin_slot = call i64 @wam_assoc_i64_iter_next(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_idx)
  %forin_done = icmp slt i64 %forin_slot, 0
  br i1 %forin_done, label %forin_after, label %forin_body

forin_body:
  %forin_key_id = call i64 @wam_assoc_i64_key_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)
~w
  br i1 ~w, label %forin_print, label %forin_skip

forin_print:
~w
  %forin_printed_newline = call i32 @putchar(i32 10)
  br label %forin_body_done

forin_skip:
  br label %forin_body_done

forin_body_done:
  %forin_next_idx = add i64 %forin_slot, 1
  br label %forin_head

forin_after:
~w
  ret i32 0',
        [TableIndex, TableIndex, GuardIR, CondVar, BodyIR, FreeIR]).

% END-loop guard operand load + comparison (fixed `forin` prefix). GuardGlobal
% is any module-level constant the guard needs (a string literal to intern);
% '' when none.
plawk_forin_end_guard_lines(guard_value(Op, V), TableIndex, Lines, CondVar, '') :-
    plawk_forin_cmp_pred(Op, Pred),
    format(atom(ValLine),
        '  %forin_gval = call i64 @wam_assoc_i64_value_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)',
        [TableIndex]),
    format(atom(CmpLine),
        '  %forin_gcmp = icmp ~w i64 %forin_gval, ~w', [Pred, V]),
    CondVar = '%forin_gcmp',
    Lines = [ValLine, CmpLine].
plawk_forin_end_guard_lines(guard_key(Op, V), _TableIndex, Lines, CondVar, '') :-
    plawk_forin_cmp_pred(Op, Pred),
    format(atom(CmpLine),
        '  %forin_gcmp = icmp ~w i64 %forin_key_id, ~w', [Pred, V]),
    CondVar = '%forin_gcmp',
    Lines = [CmpLine].
% Str-valued table (split / assoc(str)): the stored i64 is an atom id, so an
% `arr[k] CMP int` comparison resolves the element text and compares via
% strnum (numeric if the element looks like a number, else lexical -- POSIX
% duality), testing the sign against 0 with the comparison predicate.
plawk_forin_end_guard_lines(guard_value_strnum(Op, V), TableIndex, Lines, CondVar, '') :-
    plawk_forin_cmp_pred(Op, Pred),
    format(atom(ValLine),
        '  %forin_gval = call i64 @wam_assoc_i64_value_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)',
        [TableIndex]),
    format(atom(StrLine),
        '  %forin_gval_s = call i8* @wam_atom_to_string(i64 %forin_gval)', []),
    format(atom(RcLine),
        '  %forin_gval_rc = call i32 @wam_strnum_cmp_int(i8* %forin_gval_s, i8 1, i64 ~w)',
        [V]),
    format(atom(CmpLine),
        '  %forin_gcmp = icmp ~w i32 %forin_gval_rc, 0', [Pred]),
    CondVar = '%forin_gcmp',
    Lines = [ValLine, StrLine, RcLine, CmpLine].
% String equality (`arr[k] == "x"` / `!= "x"`): intern the literal into an atom
% id (from a module-level constant) and icmp it against the element's stored id.
% Interning is canonical, so equal strings share an id -- no strcmp needed. The
% intern is idempotent (a hash lookup), so doing it per iteration is correct.
plawk_forin_end_guard_lines(guard_value_streq(Op, Text), TableIndex, Lines, CondVar,
        GuardGlobal) :-
    plawk_forin_cmp_pred(Op, Pred),
    format(atom(GName), 'plawk_forin_guard_streq_~w', [TableIndex]),
    llvm_emit_c_string_global(GName, Text, GuardGlobal, StringLen, BytesLen),
    format(atom(PtrLine),
        '  %forin_glitp = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0',
        [BytesLen, BytesLen, GName]),
    format(atom(LitLine),
        '  %forin_glit = call i64 @wam_intern_atom(i8* %forin_glitp, i64 ~w)',
        [StringLen]),
    format(atom(ValLine),
        '  %forin_gval = call i64 @wam_assoc_i64_value_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)',
        [TableIndex]),
    format(atom(CmpLine),
        '  %forin_gcmp = icmp ~w i64 %forin_gval, %forin_glit', [Pred]),
    CondVar = '%forin_gcmp',
    Lines = [PtrLine, LitLine, ValLine, CmpLine].
% String ordering (`arr[k] < "x"` etc.): resolve the element text and strcmp it
% against the literal (a NUL-terminated module constant), testing the sign
% against 0 with the ordering predicate -- the same lowering the scalar string
% ordering guard uses.
plawk_forin_end_guard_lines(guard_value_strord(Op, Text), TableIndex, Lines, CondVar,
        GuardGlobal) :-
    plawk_forin_cmp_pred(Op, Pred),
    format(atom(GName), 'plawk_forin_guard_strord_~w', [TableIndex]),
    llvm_emit_c_string_global(GName, Text, GuardGlobal, _StringLen, BytesLen),
    format(atom(PtrLine),
        '  %forin_glitp = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0',
        [BytesLen, BytesLen, GName]),
    format(atom(ValLine),
        '  %forin_gval = call i64 @wam_assoc_i64_value_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)',
        [TableIndex]),
    format(atom(StrLine),
        '  %forin_gval_s = call i8* @wam_atom_to_string(i64 %forin_gval)', []),
    format(atom(RcLine),
        '  %forin_gval_rc = call i32 @strcmp(i8* %forin_gval_s, i8* %forin_glitp)', []),
    format(atom(CmpLine),
        '  %forin_gcmp = icmp ~w i32 %forin_gval_rc, 0', [Pred]),
    CondVar = '%forin_gcmp',
    Lines = [PtrLine, ValLine, StrLine, RcLine, CmpLine].
% Str-valued element vs a float literal (`arr[k] < 3.5`): build the double
% (M/D), resolve the element text, and compare via @wam_strnum_cmp_double
% (numeric vs the double, else lexical against its %g form).
plawk_forin_end_guard_lines(guard_value_strnum_f(Op, M, D), TableIndex, Lines, CondVar, '') :-
    plawk_forin_cmp_pred(Op, Pred),
    format(atom(FvLine), '  %forin_gfv = fdiv double ~w.0, ~w.0', [M, D]),
    format(atom(ValLine),
        '  %forin_gval = call i64 @wam_assoc_i64_value_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)',
        [TableIndex]),
    format(atom(StrLine),
        '  %forin_gval_s = call i8* @wam_atom_to_string(i64 %forin_gval)', []),
    format(atom(RcLine),
        '  %forin_gval_rc = call i32 @wam_strnum_cmp_double(i8* %forin_gval_s, i8 1, double %forin_gfv)', []),
    format(atom(CmpLine),
        '  %forin_gcmp = icmp ~w i32 %forin_gval_rc, 0', [Pred]),
    CondVar = '%forin_gcmp',
    Lines = [FvLine, ValLine, StrLine, RcLine, CmpLine].
% i64 counter element vs a float literal: widen the count and fcmp.
plawk_forin_end_guard_lines(guard_value_f(Op, M, D), TableIndex, Lines, CondVar, '') :-
    plawk_forin_fcmp_pred(Op, FPred),
    format(atom(FvLine), '  %forin_gfv = fdiv double ~w.0, ~w.0', [M, D]),
    format(atom(ValLine),
        '  %forin_gval = call i64 @wam_assoc_i64_value_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)',
        [TableIndex]),
    format(atom(WidenLine),
        '  %forin_gval_d = sitofp i64 %forin_gval to double', []),
    format(atom(CmpLine),
        '  %forin_gcmp = fcmp ~w double %forin_gval_d, %forin_gfv', [FPred]),
    CondVar = '%forin_gcmp',
    Lines = [FvLine, ValLine, WidenLine, CmpLine].

%% plawk_forin_end_accum_ir(+LoopVar, +ArrayName, +Acc, +Operand,
%%     +PrintFields, +AssocPlan, +Descriptor, +OutputSeparator, -IR)
%
%  Emit the END accumulate for-in (stage 2): walk the iterated table's
%  occupied slots carrying a scalar accumulator as a second loop phi, add
%  the per-entry operand (the value `arr[k]`, the key `k`, or a constant) to
%  it each iteration, then -- after the loop -- print the fields (the
%  accumulator variable resolves to the folded total, string literals print
%  verbatim) and free every table.
plawk_forin_end_accum_ir(_LoopVar, ArrayName, Acc, Operand, PrintFields,
        AssocPlan, _Descriptor, OutputSeparator, IR) :-
    plawk_assoc_table_index(AssocPlan, ArrayName, TableIndex),
    plawk_forin_accum_operand_line(Operand, TableIndex, OperandVar, OperandLine),
    phrase(plawk_forin_accum_print_lines(PrintFields, Acc, OutputSeparator, 0),
        PrintLines),
    atomic_list_concat(PrintLines, '\n', PrintIR),
    phrase(plawk_assoc_free_lines(AssocPlan), FreeLines),
    atomic_list_concat(FreeLines, '\n', FreeIR),
    format(atom(IR),
'  br label %forin_head

forin_head:
  %forin_idx = phi i64 [0, %end_print], [%forin_next_idx, %forin_body_done]
  %forin_acc = phi i64 [0, %end_print], [%forin_next_acc, %forin_body_done]
  %forin_slot = call i64 @wam_assoc_i64_iter_next(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_idx)
  %forin_done = icmp slt i64 %forin_slot, 0
  br i1 %forin_done, label %forin_after, label %forin_body

forin_body:
  %forin_key_id = call i64 @wam_assoc_i64_key_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)
~w
  %forin_next_acc = add i64 %forin_acc, ~w
  br label %forin_body_done

forin_body_done:
  %forin_next_idx = add i64 %forin_slot, 1
  br label %forin_head

forin_after:
~w
  %forin_out_newline = call i32 @putchar(i32 10)
~w
  ret i32 0',
        [TableIndex, TableIndex, OperandLine, OperandVar, PrintIR, FreeIR]).

%% plawk_forin_accum_operand_line(+Operand, +TableIndex, -OperandVar, -Line)
%  The per-iteration operand added to the accumulator. `arr[k]` loads the
%  slot value (a fresh line); `k` reuses the already-loaded key id; a
%  constant needs no line.
plawk_forin_accum_operand_line(forin_val(_Array), TableIndex,
        '%forin_acc_val', Line) :-
    format(atom(Line),
        '  %forin_acc_val = call i64 @wam_assoc_i64_value_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)',
        [TableIndex]).
plawk_forin_accum_operand_line(forin_key, _TableIndex, '%forin_key_id', '').
plawk_forin_accum_operand_line(int(Value), _TableIndex, Value, '').

%% plawk_forin_accum_print_lines(+PrintFields, +Acc, +OutputSeparator, +Index)//
%  The trailing END print, emitted in the post-loop block. The accumulator
%  variable prints the folded total (%forin_acc); string literals print
%  verbatim. Distinct label prefix (forin_out_*) -- there is no per-entry
%  print in an accumulate body, so nothing else uses these names.
plawk_forin_accum_print_lines([], _Acc, _OutputSeparator, _Index) -->
    [].
plawk_forin_accum_print_lines([var(Acc) | Rest], Acc, OutputSeparator, Index) -->
    plawk_forin_accum_separator_lines(Index, OutputSeparator),
    { format(atom(FmtVar), 'forin_out_fmt_~w', [Index]),
      format(atom(PrintVar), 'forin_out_printed_~w', [Index]),
      llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar,
          '%forin_acc', [FmtPtr, PrintCall]),
      NextIndex is Index + 1
    },
    [FmtPtr, PrintCall],
    plawk_forin_accum_print_lines(Rest, Acc, OutputSeparator, NextIndex).
plawk_forin_accum_print_lines([string(Value) | Rest], Acc, OutputSeparator, Index) -->
    plawk_forin_accum_separator_lines(Index, OutputSeparator),
    plawk_end_string_print_lines(Value, Index),
    { NextIndex is Index + 1 },
    plawk_forin_accum_print_lines(Rest, Acc, OutputSeparator, NextIndex).

plawk_forin_accum_separator_lines(0, _OutputSeparator) -->
    !,
    [].
plawk_forin_accum_separator_lines(Index, OutputSeparator) -->
    { format(atom(SpaceCall),
          '  %forin_out_separator_~w = call i32 @putchar(i32 ~w)',
          [Index, OutputSeparator])
    },
    [SpaceCall].

%% plawk_forin_end_decode_ir(+LoopVar, +ArrayName, +Vars, +Call, +Types,
%%     +PrintFields, +AssocPlan, +Descriptor, +OutputSeparator,
%%     -GlobalIR, -IR)
%
%  Emit the END decode for-in (stage 3): walk the iterated table's occupied
%  slots, load each value, box it as the grammar argument (`forin_val`),
%  destructure the returned record into typed fields via the shared record
%  shim, then print the loop key and the decoded fields. Fields are i64 or
%  f64 -- one scalar slot each, loaded directly (string fields take a
%  (ptr,len) pair and are a separate round). GlobalIR carries the
%  field-typecode constant.
plawk_forin_end_decode_ir(LoopVar, ArrayName, Vars, Call, Types, PrintFields,
        AssocPlan, Descriptor, OutputSeparator, GlobalIR, IR) :-
    forall(member(T, Types), memberchk(T, [i64, f64])),
    length(Types, NFields),
    length(Vars, NFields),
    plawk_assoc_table_index(AssocPlan, ArrayName, TableIndex),
    % The record shim call, arg-boxing and field loads, base `forin_dec`.
    maplist(plawk_dynrec_type_code, Types, Codes),
    plawk_dynrec_typecodes_escaped(Codes, Escaped),
    format(atom(GlobalIR),
        '@.forin_dec_tc = private constant [~w x i8] c"~w"',
        [NFields, Escaped]),
    plawk_dynrec_call_ir(Call, Descriptor, forin_dec, CallArgsIR,
        _ArgGlobals, ArgSetup, ShimName),
    NLast is NFields - 1,
    numlist(0, NLast, Fields),
    findall(ZLine,
        ( member(FZ, Fields),
          format(atom(ZLine),
              '  %forin_dec_z~wp = getelementptr i64, i64* %forin_dec_slots, i64 ~w\n  store i64 0, i64* %forin_dec_z~wp\n  %forin_dec_zl~wp = getelementptr i64, i64* %forin_dec_lens, i64 ~w\n  store i64 0, i64* %forin_dec_zl~wp',
              [FZ, FZ, FZ, FZ, FZ, FZ]) ),
        ZeroLines),
    plawk_dynrec_field_load_lines(Fields, Types, forin_dec, LoadLines),
    atomic_list_concat(ArgSetup, '\n', ArgSetupIR),
    atomic_list_concat(ZeroLines, '\n', ZeroIR),
    atomic_list_concat(LoadLines, '\n', LoadIR),
    format(atom(CallIR),
        '  %forin_dec_ok = call i1 @~w(~w, i32 ~w, i8* %forin_dec_tcp, i64* %forin_dec_slots, i64* %forin_dec_lens)',
        [ShimName, CallArgsIR, NFields]),
    % Map each decoded variable to its loaded-field SSA and type for the
    % print (i64 -> integer printf, f64 -> %g).
    findall(V-(SSA-Type),
        ( nth0(I, Vars, V), nth0(I, Types, Type),
          format(atom(SSA), '%forin_dec_f~w', [I]) ),
        VarSSAs),
    phrase(plawk_forin_decode_print_lines(PrintFields, LoopVar, VarSSAs,
        Descriptor, OutputSeparator, 0), PrintLineList),
    atomic_list_concat(PrintLineList, '\n', PrintIR),
    phrase(plawk_assoc_free_lines(AssocPlan), FreeLines),
    atomic_list_concat(FreeLines, '\n', FreeIR),
    format(atom(IR),
'  br label %forin_head

forin_head:
  %forin_idx = phi i64 [0, %end_print], [%forin_next_idx, %forin_body_done]
  %forin_slot = call i64 @wam_assoc_i64_iter_next(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_idx)
  %forin_done = icmp slt i64 %forin_slot, 0
  br i1 %forin_done, label %forin_after, label %forin_body

forin_body:
  %forin_key_id = call i64 @wam_assoc_i64_key_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)
  %forin_slot_value = call i64 @wam_assoc_i64_value_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)
~w
  %forin_dec_slots = alloca i64, i32 ~w
  %forin_dec_lens = alloca i64, i32 ~w
~w
  %forin_dec_tcp = getelementptr [~w x i8], [~w x i8]* @.forin_dec_tc, i32 0, i32 0
~w
~w
~w
  %forin_dec_newline = call i32 @putchar(i32 10)
  br label %forin_body_done

forin_body_done:
  %forin_next_idx = add i64 %forin_slot, 1
  br label %forin_head

forin_after:
~w
  ret i32 0',
        [TableIndex, TableIndex, TableIndex, ArgSetupIR, NFields, NFields,
         ZeroIR, NFields, NFields, CallIR, LoadIR, PrintIR, FreeIR]).

%% plawk_forin_decode_print_lines(+PrintFields, +LoopVar, +VarSSAs,
%%     +Descriptor, +OutputSeparator, +Index)//
%  The per-entry print in an END decode body: the loop key (text, or
%  numeric for a binary descriptor), a decoded field (its loaded SSA, as
%  i64 or f64 per its type), or a string literal. Distinct label prefix
%  (forin_dprint_*).
plawk_forin_decode_print_lines([], _LoopVar, _VarSSAs, _Descriptor,
        _OutputSeparator, _Index) -->
    [].
plawk_forin_decode_print_lines([var(LoopVar) | Rest], LoopVar, VarSSAs,
        Descriptor, OutputSeparator, Index) -->
    { \+ memberchk(LoopVar-_, VarSSAs) },
    % The loop key (a decoded var never shadows the loop key -- distinct
    % identifiers by construction).
    plawk_forin_decode_separator_lines(Index, OutputSeparator),
    { plawk_descriptor_is_binary(Descriptor)
    ->  format(atom(FmtVar), 'forin_dprint_key_fmt_~w', [Index]),
        format(atom(PrintVar), 'forin_dprint_key_~w', [Index]),
        llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar,
            '%forin_key_id', Lines)
    ;   format(atom(KeyStr),
            '  %forin_dprint_key_s_~w = call i8* @wam_atom_to_string(i64 %forin_key_id)',
            [Index]),
        format(atom(FmtVar), 'forin_dprint_key_fmt_~w', [Index]),
        format(atom(PrintVar), 'forin_dprint_key_~w', [Index]),
        format(atom(PtrIR), '%forin_dprint_key_s_~w', [Index]),
        llvm_emit_printf_string(plawk_surface_print_string, FmtVar, PrintVar,
            PtrIR, [FmtPtr, PrintCall]),
        Lines = [KeyStr, FmtPtr, PrintCall]
    },
    plawk_emit_lines(Lines),
    { NextIndex is Index + 1 },
    plawk_forin_decode_print_lines(Rest, LoopVar, VarSSAs, Descriptor,
        OutputSeparator, NextIndex).
plawk_forin_decode_print_lines([var(Var) | Rest], LoopVar, VarSSAs, Descriptor,
        OutputSeparator, Index) -->
    { memberchk(Var-(SSA-Type), VarSSAs) },
    % A decoded field: i64 via the integer printf, f64 via %g.
    plawk_forin_decode_separator_lines(Index, OutputSeparator),
    { Type == f64
    ->  format(atom(FmtPtr),
            '  %forin_dprint_fld_fmt_~w = getelementptr [3 x i8], [3 x i8]* @.plawk_surface_print_f64, i32 0, i32 0',
            [Index]),
        format(atom(PrintCall),
            '  %forin_dprint_fld_~w = call i32 (i8*, ...) @printf(i8* %forin_dprint_fld_fmt_~w, double ~w)',
            [Index, Index, SSA])
    ;   format(atom(FmtVar), 'forin_dprint_fld_fmt_~w', [Index]),
        format(atom(PrintVar), 'forin_dprint_fld_~w', [Index]),
        llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar, SSA,
            [FmtPtr, PrintCall])
    },
    { NextIndex is Index + 1 },
    [FmtPtr, PrintCall],
    plawk_forin_decode_print_lines(Rest, LoopVar, VarSSAs, Descriptor,
        OutputSeparator, NextIndex).
plawk_forin_decode_print_lines([string(Value) | Rest], LoopVar, VarSSAs,
        Descriptor, OutputSeparator, Index) -->
    plawk_forin_decode_separator_lines(Index, OutputSeparator),
    plawk_end_string_print_lines(Value, Index),
    { NextIndex is Index + 1 },
    plawk_forin_decode_print_lines(Rest, LoopVar, VarSSAs, Descriptor,
        OutputSeparator, NextIndex).

plawk_forin_decode_separator_lines(0, _OutputSeparator) -->
    !,
    [].
plawk_forin_decode_separator_lines(Index, OutputSeparator) -->
    { format(atom(SpaceCall),
          '  %forin_dprint_separator_~w = call i32 @putchar(i32 ~w)',
          [Index, OutputSeparator])
    },
    [SpaceCall].

%% plawk_foreign_arg_ir(forin_val(_), ...) -- a for-in-scoped grammar arg:
%  the current iterated value, already loaded into %forin_slot_value by the
%  decode loop, boxed as an integer %Value. (Clause lives beside the other
%  plawk_foreign_arg_ir clauses via discontiguous; kept here next to the
%  for-in decode emitter that relies on the fixed %forin_slot_value name.)
:- discontiguous plawk_foreign_arg_ir/6.
plawk_foreign_arg_ir(forin_val(_Array), _FieldSeparator, ArgBase, ArgValueIR,
        [], [IntIR]) :-
    format(atom(IntIR),
        '  %~w_v = call %Value @value_integer(i64 %forin_slot_value)',
        [ArgBase]),
    format(atom(ArgValueIR), '%~w_v', [ArgBase]).

plawk_forin_body_print_lines([], _LoopVar, _ArrayName, _TableIndex, _AssocPlan,
        _Descriptor, _OutputSeparator, _) -->
    [].
plawk_forin_body_print_lines([var(LoopVar) | Rest], LoopVar, ArrayName,
        TableIndex, AssocPlan, Descriptor, OutputSeparator, PrintIndex) -->
    { plawk_descriptor_is_binary(Descriptor) },
    % Binary mode: keys are raw i64 field values, printed numerically.
    plawk_forin_separator_lines(PrintIndex, OutputSeparator),
    { format(atom(FmtVar), 'forin_key_fmt_~w', [PrintIndex]),
      format(atom(PrintVar), 'forin_printed_key_~w', [PrintIndex]),
      llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar,
          '%forin_key_id', [FmtPtr, PrintCall]),
      NextPrintIndex is PrintIndex + 1
    },
    [FmtPtr, PrintCall],
    plawk_forin_body_print_lines(Rest, LoopVar, ArrayName, TableIndex, AssocPlan,
        Descriptor, OutputSeparator, NextPrintIndex).
plawk_forin_body_print_lines([var(LoopVar) | Rest], LoopVar, ArrayName,
        TableIndex, AssocPlan, Descriptor, OutputSeparator, PrintIndex) -->
    { plawk_assoc_plan_posarray_array(AssocPlan, ArrayName) },
    % Positional array (e.g. split): keys are integer positions, printed
    % numerically rather than resolved as atom-registry ids.
    plawk_forin_separator_lines(PrintIndex, OutputSeparator),
    { format(atom(FmtVar), 'forin_key_fmt_~w', [PrintIndex]),
      format(atom(PrintVar), 'forin_printed_key_~w', [PrintIndex]),
      llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar,
          '%forin_key_id', [FmtPtr, PrintCall]),
      NextPrintIndex is PrintIndex + 1
    },
    [FmtPtr, PrintCall],
    plawk_forin_body_print_lines(Rest, LoopVar, ArrayName, TableIndex, AssocPlan,
        Descriptor, OutputSeparator, NextPrintIndex).
plawk_forin_body_print_lines([var(LoopVar) | Rest], LoopVar, ArrayName,
        TableIndex, AssocPlan, Descriptor, OutputSeparator, PrintIndex) -->
    plawk_forin_separator_lines(PrintIndex, OutputSeparator),
    { format(atom(KeyString),
          '  %forin_key_s_~w = call i8* @wam_atom_to_string(i64 %forin_key_id)',
          [PrintIndex]),
      format(atom(FmtVar), 'forin_key_fmt_~w', [PrintIndex]),
      format(atom(PrintVar), 'forin_printed_key_~w', [PrintIndex]),
      format(atom(PtrIR), '%forin_key_s_~w', [PrintIndex]),
      llvm_emit_printf_string(plawk_surface_print_string, FmtVar, PrintVar, PtrIR,
          [FmtPtr, PrintCall]),
      NextPrintIndex is PrintIndex + 1
    },
    [KeyString, FmtPtr, PrintCall],
    plawk_forin_body_print_lines(Rest, LoopVar, ArrayName, TableIndex, AssocPlan,
        Descriptor, OutputSeparator, NextPrintIndex).
plawk_forin_body_print_lines([assoc(var(LookupArrayName), var(LoopVar)) | Rest],
        LoopVar, ArrayName, TableIndex, AssocPlan, Descriptor, OutputSeparator, PrintIndex) -->
    plawk_forin_separator_lines(PrintIndex, OutputSeparator),
    { (   LookupArrayName == ArrayName
      ->  format(atom(Value),
              '  %forin_value_~w = call i64 @wam_assoc_i64_value_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_slot)',
              [PrintIndex, TableIndex])
      ;   plawk_assoc_table_index(AssocPlan, LookupArrayName, LookupTableIndex),
          format(atom(Value),
              '  %forin_value_~w = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %forin_key_id)',
              [PrintIndex, LookupTableIndex])
      ),
      format(atom(ValueIR), '%forin_value_~w', [PrintIndex]),
      (   plawk_assoc_plan_str_array(AssocPlan, LookupArrayName)
      ->  % str-valued table: the stored i64 is an atom-registry id --
          % resolve it to text, like the key print does.
          format(atom(ValueString),
              '  %forin_value_s_~w = call i8* @wam_atom_to_string(i64 ~w)',
              [PrintIndex, ValueIR]),
          format(atom(FmtVar), 'forin_str_fmt_~w', [PrintIndex]),
          format(atom(PrintVar), 'forin_printed_str_~w', [PrintIndex]),
          format(atom(PtrIR), '%forin_value_s_~w', [PrintIndex]),
          llvm_emit_printf_string(plawk_surface_print_string, FmtVar, PrintVar,
              PtrIR, [FmtPtr, PrintCall]),
          ValueLines = [Value, ValueString, FmtPtr, PrintCall]
      ;   format(atom(FmtVar), 'forin_i64_fmt_~w', [PrintIndex]),
          format(atom(PrintVar), 'forin_printed_i64_~w', [PrintIndex]),
          llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar,
              ValueIR, [FmtPtr, PrintCall]),
          ValueLines = [Value, FmtPtr, PrintCall]
      ),
      NextPrintIndex is PrintIndex + 1
    },
    plawk_emit_lines(ValueLines),
    plawk_forin_body_print_lines(Rest, LoopVar, ArrayName, TableIndex, AssocPlan,
        Descriptor, OutputSeparator, NextPrintIndex).
plawk_forin_body_print_lines([string(Value) | Rest], LoopVar, ArrayName,
        TableIndex, AssocPlan, Descriptor, OutputSeparator, PrintIndex) -->
    plawk_forin_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_string_print_lines(Value, PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_forin_body_print_lines(Rest, LoopVar, ArrayName, TableIndex, AssocPlan,
        Descriptor, OutputSeparator, NextPrintIndex).

plawk_forin_separator_lines(0, _OutputSeparator) -->
    !,
    [].
plawk_forin_separator_lines(PrintIndex, OutputSeparator) -->
    { format(atom(SpaceCall),
          '  %forin_printed_separator_~w = call i32 @putchar(i32 ~w)',
          [PrintIndex, OutputSeparator])
    },
    [SpaceCall].

plawk_combine_entry_ir('', IR, IR) :-
    !.
plawk_combine_entry_ir(IR, '', IR) :-
    !.
plawk_combine_entry_ir(FirstIR, SecondIR, CombinedIR) :-
    format(atom(CombinedIR), '~w~n~w', [FirstIR, SecondIR]).

plawk_i64_end_print_globals(SurfaceGlobals, RuntimeGlobals) :-
    format(atom(RuntimeGlobals),
'@.plawk_surface_print_i64 = private constant [4 x i8] c"%ld\\00"
@.plawk_surface_print_line = private constant [4 x i8] c"%s\\0A\\00"
@.plawk_surface_print_slice = private constant [5 x i8] c"%.*s\\00"
@.plawk_surface_print_newline = private constant [2 x i8] c"\\0A\\00"
@.plawk_surface_print_string = private constant [3 x i8] c"%s\\00"
@.plawk_surface_print_f64 = private constant [3 x i8] c"%g\\00"
@plawk_exit_code = internal global i32 0
~w

',
        [SurfaceGlobals]).

% State plans keep recognized PLAWK state separate from the LLVM slot numbering.
% Associative arrays use a separate table plan because they are pointer state.
plawk_state_plan_slots(state_plan(Slots), Slots).

plawk_state_slot_count(StatePlan, Count) :-
    plawk_state_plan_slots(StatePlan, Slots),
    length(Slots, Count).

plawk_state_slot_index(StatePlan, Slot, Index) :-
    plawk_state_plan_slots(StatePlan, Slots),
    nth0(Index, Slots, Slot).

% Scalar slots are typed: scalar_counter(Name) accumulates in an i64
% register chain, scalar_double(Name) in a double chain. The type is
% inferred from the program (see plawk_scalar_typed_slots/3), never
% declared, matching awk's untyped surface.
plawk_slot_name(scalar_counter(Name), Name).
plawk_slot_name(scalar_double(Name), Name).
% A string scalar's slot holds an interned atom id (an i64); the id 0 is the
% "unset" sentinel, printed as the empty string.
plawk_slot_name(scalar_string(Name), Name).
% A strnum scalar (POSIX numeric-string duality, PLAWK_STRNUM_DUALITY.md): a
% value copied from a strnum source (a field for now) that compares numerically
% or lexically by its runtime content. Its slot holds an interned atom id (an
% i64) like a string scalar -- the string bytes are retained so a later
% comparison can decide numeric vs lexical. This slot KIND is recognised
% infrastructure (origin analysis, symbol, repr) but the type inferencer does
% not yet produce it and no comparison dispatches on it -- that is step 3.
plawk_slot_name(scalar_strnum(Name), Name).

plawk_slot_llvm_type(scalar_counter(_Name), i64).
plawk_slot_llvm_type(scalar_double(_Name), double).
plawk_slot_llvm_type(scalar_string(_Name), i64).
plawk_slot_llvm_type(scalar_strnum(_Name), i64).

plawk_slot_zero_ir(scalar_counter(_Name), '0').
plawk_slot_zero_ir(scalar_double(_Name), '0.0').
plawk_slot_zero_ir(scalar_string(_Name), '0').
plawk_slot_zero_ir(scalar_strnum(_Name), '0').

plawk_state_slot_lookup(StatePlan, Name, Index, Slot) :-
    plawk_state_plan_slots(StatePlan, Slots),
    nth0(Index, Slots, Slot),
    plawk_slot_name(Slot, Name),
    !.

plawk_trim_control_tails([], []).
plawk_trim_control_tails([next | _Rest], [next]) :-
    !.
plawk_trim_control_tails([break | _Rest], [break]) :-
    !.
plawk_trim_control_tails([continue | _Rest], [continue]) :-
    !.
plawk_trim_control_tails([if(Pattern, ThenActions, ElseActions) | Rest],
        [if(Pattern, TrimmedThenActions, TrimmedElseActions) | TrimmedRest]) :-
    !,
    plawk_trim_control_tails(ThenActions, TrimmedThenActions),
    plawk_trim_control_tails(ElseActions, TrimmedElseActions),
    plawk_trim_control_tails(Rest, TrimmedRest).
plawk_trim_control_tails([Action | Rest], [Action | TrimmedRest]) :-
    plawk_trim_control_tails(Rest, TrimmedRest).

plawk_split_terminal_control(Actions0, BodyActions, Control) :-
    plawk_trim_control_tails(Actions0, Actions),
    plawk_split_normalized_terminal_control(Actions, BodyActions, Control).

plawk_split_normalized_terminal_control(Actions, BodyActions, terminal_next) :-
    append(BodyActions, [next], Actions),
    !,
    \+ plawk_actions_have_control(BodyActions).
plawk_split_normalized_terminal_control(Actions, BodyActions, terminal_break) :-
    append(BodyActions, [break], Actions),
    !,
    \+ plawk_actions_have_control(BodyActions).
% A rule-level `exit [N]` is a terminal like `break`: it leaves the record loop
% via break_close_stream (which runs END and returns @plawk_exit_code). Unlike
% break, it has a side effect -- storing the code -- so the trailing exit is
% replaced by an `exit_store(N)` marker kept in the body (the sequence walker
% lowers it to just the store, no branch); the branch to break_close_stream
% comes from the terminal_exit rule target.
plawk_split_normalized_terminal_control(Actions, BodyActions, terminal_exit) :-
    append(Prefix, [exit(int(Code))], Actions),
    !,
    \+ plawk_actions_have_control(Prefix),
    append(Prefix, [exit_store(Code)], BodyActions).
plawk_split_normalized_terminal_control(Actions, Actions, fallthrough) :-
    \+ plawk_actions_have_control(Actions).

plawk_actions_have_control(Actions) :-
    member(Action, Actions),
    plawk_action_has_control(Action).

plawk_action_has_control(next).
plawk_action_has_control(break).
plawk_action_has_control(if(_Pattern, ThenActions, ElseActions)) :-
    ( plawk_branch_actions_have_unsupported_control(ThenActions)
    ; plawk_branch_actions_have_unsupported_control(ElseActions)
    ).
plawk_action_has_control(foreach_loop(_Layout, Body)) :-
    plawk_branch_actions_have_unsupported_control(Body).
% A while/do-while loop consumes its own body's break/continue (loop-local), so
% those do NOT make the loop-as-action carry control to the rule level; only a
% `next` inside the body propagates (to the record loop).
plawk_action_has_control(while_loop(_Cond, Body)) :-
    plawk_actions_have_next_control(Body).
plawk_action_has_control(do_while_loop(Body, _Cond)) :-
    plawk_actions_have_next_control(Body).

% True if some action reaches a bare `next` (recursing through ifs and nested
% loops) -- `next` propagates past loops, break/continue do not.
plawk_actions_have_next_control(Actions) :-
    member(Action, Actions),
    plawk_action_reaches_next(Action).

plawk_action_reaches_next(next).
plawk_action_reaches_next(if(_Pattern, ThenActions, ElseActions)) :-
    ( plawk_actions_have_next_control(ThenActions)
    ; plawk_actions_have_next_control(ElseActions)
    ).
plawk_action_reaches_next(while_loop(_Cond, Body)) :-
    plawk_actions_have_next_control(Body).
plawk_action_reaches_next(do_while_loop(Body, _Cond)) :-
    plawk_actions_have_next_control(Body).
plawk_action_reaches_next(foreach_loop(_Layout, Body)) :-
    plawk_actions_have_next_control(Body).

plawk_branch_actions_have_unsupported_control(Actions) :-
    plawk_trim_control_tails(Actions, TrimmedActions),
    (   append(BodyActions, [next], TrimmedActions)
    ->  plawk_actions_have_control(BodyActions)
    ;   append(BodyActions, [break], TrimmedActions)
    ->  plawk_actions_have_control(BodyActions)
    ;   plawk_actions_have_control(TrimmedActions)
    ).

plawk_rule_target(fallthrough, NextLabel, NextLabel).
plawk_rule_target(terminal_next, _NextLabel, continue_loop).
plawk_rule_target(terminal_break, _NextLabel, break_close_stream).
plawk_rule_target(terminal_exit, _NextLabel, break_close_stream).


plawk_controls_have_break(Controls) :-
    member(terminal_break, Controls).

plawk_assoc_break_close_ir(Controls, IR) :-
    (   plawk_controls_have_break(Controls)
    ->  IR = 'break_close_stream:
  %break_close_ok = call i1 @wam_stream_close_value(%Value %handle)
  br i1 %break_close_ok, label %end_print, label %fail_close
'
    ;   IR = ''
    ).

plawk_break_close_ir(StatePlan, RuleCount, Controls, BranchControlExits, BreakPredKind, BreakCloseIR, FinalStatePhiIR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    (   plawk_controls_have_break(Controls)
    ;   member(terminal_exit, Controls)
    ;   member(branch_break(_Label, _Values), BranchControlExits)
    ;   member(branch_exit(_ExitLabel, _ExitValues), BranchControlExits)
    ),
    !,
    phrase(plawk_break_slot_phi_lines(Slots, RuleCount, Controls, BranchControlExits, BreakPredKind, 0), BreakSlotPhiLines),
    atomic_list_concat(BreakSlotPhiLines, '\n', BreakSlotPhiIR),
    format(atom(BreakCloseIR),
'break_close_stream:
~w
  %break_close_ok = call i1 @wam_stream_close_value(%Value %handle)
  br i1 %break_close_ok, label %end_print, label %fail_close
',
        [BreakSlotPhiIR]),
    phrase(plawk_final_state_phi_lines(Slots, true, 0), FinalStatePhiLines),
    atomic_list_concat(FinalStatePhiLines, '\n', FinalStatePhiIR0),
    ( FinalStatePhiIR0 == ''
    -> FinalStatePhiIR = ''
    ;  format(atom(FinalStatePhiIR), '~w~n', [FinalStatePhiIR0])
    ).
plawk_break_close_ir(StatePlan, _RuleCount, _Controls, _BranchControlExits, _BreakPredKind, '', FinalStatePhiIR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_final_state_phi_lines(Slots, false, 0), FinalStatePhiLines),
    atomic_list_concat(FinalStatePhiLines, '\n', FinalStatePhiIR0),
    ( FinalStatePhiIR0 == ''
    -> FinalStatePhiIR = ''
    ;  format(atom(FinalStatePhiIR), '~w~n', [FinalStatePhiIR0])
    ).
plawk_break_slot_phi_lines([], _RuleCount, _Controls, _BranchControlExits, _BreakPredKind, _) -->
    [].
plawk_break_slot_phi_lines([Slot | Rest], RuleCount, Controls, BranchControlExits, BreakPredKind, SlotIndex) -->
    { LastRuleIndex is RuleCount - 1,
      findall(Incoming,
          ( between(0, LastRuleIndex, RuleIndex),
            nth0(RuleIndex, Controls, RuleControl),
            memberchk(RuleControl, [terminal_break, terminal_exit]),
            plawk_break_predecessor_label(BreakPredKind, RuleIndex, PredLabel),
            format(atom(Incoming), '[%rule_~w_slot_~w, %~w]',
                [RuleIndex, SlotIndex, PredLabel])
          ),
          RuleIncomings),
      plawk_branch_break_phi_incomings(BranchControlExits, SlotIndex, BranchIncomings),
      append(RuleIncomings, BranchIncomings, Incomings),
      Incomings \== [],
      atomic_list_concat(Incomings, ', ', IncomingIR),
      plawk_slot_llvm_type(Slot, Type),
      format(atom(Line), '  %break_slot_~w = phi ~w ~w', [SlotIndex, Type, IncomingIR]),
      NextSlotIndex is SlotIndex + 1
    },
    [Line],
    plawk_break_slot_phi_lines(Rest, RuleCount, Controls, BranchControlExits, BreakPredKind, NextSlotIndex).

plawk_branch_break_phi_incomings([], _SlotIndex, []).
% `exit` reaches break_close_stream too (it runs END on the way out), so its
% slot values merge into the break-close phi exactly like a break's.
plawk_branch_break_phi_incomings([Exit | Rest], SlotIndex, [Incoming | Incomings]) :-
    ( Exit = branch_break(Label, Values) ; Exit = branch_exit(Label, Values) ),
    !,
    nth0(SlotIndex, Values, Value),
    format(atom(Incoming), '[~w, %~w]', [Value, Label]),
    plawk_branch_break_phi_incomings(Rest, SlotIndex, Incomings).
plawk_branch_break_phi_incomings([_Exit | Rest], SlotIndex, Incomings) :-
    plawk_branch_break_phi_incomings(Rest, SlotIndex, Incomings).

plawk_break_predecessor_label(apply, RuleIndex, Label) :-
    format(atom(Label), 'rule_~w_apply', [RuleIndex]).
plawk_break_predecessor_label(done, RuleIndex, Label) :-
    format(atom(Label), 'rule_~w_done', [RuleIndex]).

plawk_final_state_phi_lines([], _HasBreak, _) -->
    [].
plawk_final_state_phi_lines([Slot | Rest], HasBreak, SlotIndex) -->
    { format(atom(EofIncoming), '[%slot_~w, %close_stream]', [SlotIndex]),
      ( HasBreak == true
      -> format(atom(BreakIncoming), '[%break_slot_~w, %break_close_stream]', [SlotIndex]),
         Incomings = [EofIncoming, BreakIncoming]
      ;  Incomings = [EofIncoming]
      ),
      atomic_list_concat(Incomings, ', ', IncomingIR),
      plawk_slot_llvm_type(Slot, Type),
      format(atom(Line), '  %final_slot_~w = phi ~w ~w', [SlotIndex, Type, IncomingIR]),
      NextSlotIndex is SlotIndex + 1
    },
    [Line],
    plawk_final_state_phi_lines(Rest, HasBreak, NextSlotIndex).

plawk_scalar_rule_controls(Rules, Controls) :-
    findall(Control,
        ( member(rule(_Pattern, Actions), Rules),
          plawk_split_terminal_control(Actions, _BodyActions, Control)
        ),
        Controls).

plawk_scalar_state_plan(Rules, PrintFields, state_plan(Slots)) :-
    findall(Name,
        ( member(rule(_Pattern, Actions), Rules),
          plawk_trim_control_tails(Actions, ReachableActions),
          member(Action, ReachableActions),
          plawk_scalar_update_action_name(Action, Name)
        ),
        ActionVars),
    plawk_rules_body_print_fields(Rules, BodyPrintFields),
    ( ActionVars \== [] ; BodyPrintFields \== [] ),
    findall(Name,
        ( member(Field, PrintFields),
          plawk_scalar_print_expr(Field, Name)
        ),
        PrintVars),
    append(ActionVars, PrintVars, Names0),
    sort(Names0, Names),
    plawk_scalar_typed_slots(Rules, Names, Slots).

%% plawk_scalar_typed_slots(+Rules, +Names, -Slots)
%
%  Fixpoint type inference: a scalar is double when any update assigns
%  it a double-typed expression (float literal or float($N) leaf) or
%  reads an already-double scalar; everything else stays i64. i64
%  reads inside a double update promote via sitofp at emission.
plawk_scalar_typed_slots(Rules, Names, Slots) :-
    findall(Name-Expr,
        ( member(rule(_Pattern, Actions0), Rules),
          plawk_trim_control_tails(Actions0, Actions),
          member(Action, Actions),
          plawk_scalar_update_name_expr(Action, Name, Expr)
        ),
        Updates),
    plawk_scalar_double_fixpoint(Updates, [], Doubles),
    plawk_scalar_string_names(Rules, Strings0),
    plawk_scalar_strnum_names(Rules, Strnums),
    % An ARGV/getline source produces a set_str op, so it also lands in Strings0;
    % once it is an ACTIVATED strnum, drop it from the plain-string set so it
    % types as scalar_strnum (Strings is checked before Strnums). A deactivated
    % candidate stays in Strings -> scalar_string, its prior behaviour (no
    % regression). Field strnums are not in Strings0, so this is a no-op for them.
    ord_subtract(Strings0, Strnums, Strings),
    maplist(plawk_scalar_typed_slot(Doubles, Strings, Strnums), Names, Slots).

% A scalar is string-typed when it is assigned a string RHS (`x = $1 $2` or
% `x = "text"`), tracked separately from the double/counter fixpoint. v1 looks
% at top-level rule-body assignments; a nested (if/loop-body) string assignment
% is a follow-on.
plawk_scalar_string_names(Rules, Strings) :-
    findall(Name,
        ( member(rule(_Pattern, Actions0), Rules),
          plawk_trim_control_tails(Actions0, Actions),
          member(Action, Actions),
          plawk_scalar_action_update(Action, Name, set_str(_Src))
        ),
        Names0),
    sort(Names0, Strings).

%% plawk_scalar_strnum_names(+Rules, -Strnums)
%
%  Origin/provenance analysis for POSIX strnum duality (PLAWK_STRNUM_DUALITY.md
%  step 2). A name is a strnum when it is assigned **only** from strnum sources
%  and never from a non-strnum one. A strnum source is a bare field copy
%  (`x = $N`); a disqualifier is any other write to the name -- a literal,
%  arithmetic, concat, a string builtin, a sprintf, a ternary, another var,
%  etc. -- because those produce a plain number or string, destroying the
%  duality (matching awk, where arithmetic yields a number and a literal yields
%  a string). A name must have at least one strnum source and zero disqualifiers
%  to qualify; a name written both ways cannot be tagged with one static slot
%  kind, so it is conservatively excluded (step 5's honest-scoping gate governs
%  those). v1 scope mirrors plawk_scalar_string_names/2: top-level rule-body
%  assignments (nested if/loop-body writes and non-field sources such as
%  split()/getline are follow-ons -- design doc step 4).
%
%  A candidate must ALSO pass a read-use gate (step 3): every read of the name
%  must be in a position the strnum codegen supports -- a bare `print` field, or
%  a comparison against another (activated) strnum var or a string literal.
%  A name read in arithmetic, against a numeric literal, or in any other
%  context stays a plain i64 counter (its current, correct behaviour), so
%  activating strnum never regresses those programs. Because a var-vs-var
%  comparison is only supported when BOTH sides are strnum, the gate is a
%  fixpoint: deactivating one name can make its comparison partner unsupported,
%  so the set is shrunk until stable.
plawk_scalar_strnum_names(Rules, Strnums) :-
    % Disqualified names: written by something that is neither a field copy nor a
    % plain var copy (a literal, arithmetic assignment, concat, string builtin,
    % ...). Such a name can never be a pure strnum.
    findall(Name,
        ( plawk_strnum_rule_action(Rules, ActionD),
          plawk_scalar_strnum_disqualify(ActionD, Name)
        ),
        Disq0),
    sort(Disq0, Disq),
    % Seeds: names with a strnum source (field copy / ARGV / getline), not
    % disqualified.
    findall(Name,
        ( plawk_strnum_rule_action(Rules, Action),
          plawk_scalar_strnum_source(Action, Name),
          \+ memberchk(Name, Disq)
        ),
        Seeds0),
    sort(Seeds0, Seeds),
    % Grow through plain copies to the maximum possible set (`z = x` propagates
    % strnum-ness), then stabilize: a greatest fixpoint that removes any name
    % whose reads are unsupported OR whose only source is a copy from a
    % now-removed name (so copy chains collapse correctly).
    plawk_strnum_copy_grow(Rules, Disq, Seeds, MaxSet),
    plawk_strnum_stabilize(Rules, Disq, MaxSet, Strnums).

% Grow: add any copy target `z = x` whose source x is in the set and which is
% not disqualified, until stable (monotone add -> terminates).
plawk_strnum_copy_grow(Rules, Disq, Set0, Set) :-
    findall(Z,
        ( plawk_strnum_rule_action(Rules, set(var(Z), var(X))),
          memberchk(X, Set0),
          \+ memberchk(Z, Set0),
          \+ memberchk(Z, Disq)
        ),
        New0),
    sort(New0, New),
    ( New == []
    -> Set = Set0
    ;  ord_union(Set0, New, Set1),
       plawk_strnum_copy_grow(Rules, Disq, Set1, Set)
    ).

% Stabilize: shrink until every remaining name is both validly sourced (a field
% copy, or a copy from a still-present strnum) and free of unsupported reads,
% given the current set. Monotone shrink -> terminates.
plawk_strnum_stabilize(Rules, Disq, Set0, Set) :-
    exclude(plawk_strnum_name_unstable(Rules, Disq, Set0), Set0, Set1),
    ( Set1 == Set0
    -> Set = Set0
    ;  plawk_strnum_stabilize(Rules, Disq, Set1, Set)
    ).

plawk_strnum_name_unstable(Rules, _Disq, Set, Name) :-
    plawk_strnum_name_has_unsafe_read(Rules, Set, Name),
    !.
plawk_strnum_name_unstable(Rules, _Disq, Set, Name) :-
    \+ plawk_strnum_validly_sourced(Rules, Set, Name).

% Validly sourced in Set: a field copy, or a copy `Name = M` with M still in Set.
plawk_strnum_validly_sourced(Rules, _Set, Name) :-
    plawk_strnum_rule_action(Rules, Action),
    plawk_scalar_strnum_source(Action, Name),
    !.
plawk_strnum_validly_sourced(Rules, Set, Name) :-
    plawk_strnum_rule_action(Rules, set(var(Name), var(M))),
    memberchk(M, Set),
    !.

% Every rule action. v1 scope is top-level rule-body actions (nested if/loop
% bodies are excluded: plawk does not yet correctly propagate a value assigned
% in a nested block to a later statement, so activating strnum there would ride
% on that unsupported control-flow shape).
plawk_strnum_rule_action(Rules, Action) :-
    member(rule(_Pattern, Actions0), Rules),
    plawk_trim_control_tails(Actions0, Actions),
    member(Action, Actions).

% A strnum FIELD source: a bare field copy `x = $N` (the seed of strnum-ness).
% Both surface shapes for a field read (`field(N)` and `int(field(N))`) count.
plawk_scalar_strnum_field_source(set(var(Name), field(FieldIndex)), Name) :-
    integer(FieldIndex),
    FieldIndex >= 0.
plawk_scalar_strnum_field_source(set(var(Name), int(field(FieldIndex))), Name) :-
    integer(FieldIndex),
    FieldIndex >= 0.

% A strnum source: any assignment whose value is an external string of unknown
% numericness -- a field copy, a command-line argument (`x = ARGV[N]`), or a
% getline read (`getline var < "file"` / `status = getline var < "file"`). These
% are POSIX strnums: their runtime content decides numeric vs lexical comparison.
% All store an interned atom id into the slot (via set_str for argv/getline, via
% the field-intern for a field), so they share the scalar_strnum representation.
plawk_scalar_strnum_source(Action, Name) :-
    plawk_scalar_strnum_field_source(Action, Name).
plawk_scalar_strnum_source(set(var(Name), argv_at(N)), Name) :-
    integer(N),
    N >= 0.
plawk_scalar_strnum_source(getline_read(Name, _File), Name).
plawk_scalar_strnum_source(getline_capture(_Status, Name, _File), Name).

% A write is disqualifying unless it is a strnum-preserving source: a field copy
% (`x = $N`) or a plain var copy (`z = x`, which propagates strnum-ness). Any
% other write (a literal, arithmetic, concat, a string builtin, ...) destroys
% the duality and disqualifies the name.
plawk_scalar_strnum_disqualify(Action, Name) :-
    plawk_scalar_action_update(Action, Name, _Operation),
    \+ plawk_scalar_strnum_source(Action, Name),
    \+ Action = set(var(Name), var(_)).

% True if Name has at least one read the strnum codegen does not support, given
% the currently-activated Set.
plawk_strnum_name_has_unsafe_read(Rules, Set, Name) :-
    member(rule(_Pattern, Actions0), Rules),
    plawk_trim_control_tails(Actions0, Actions),
    member(Action, Actions),
    plawk_strnum_action_unsafe_read(Action, Set, Name),
    !.

% A NUMERIC scalar update (set/add/inc of a numeric expression, e.g. `y = x + 1`
% or `c += x`): reads of Name inside it are coerced to a number (step 3c -- via
% the same field parser for i64, strtod for f64), so they are supported. The
% whole action has no unsafe read of Name; cut and fail (fail = "not unsafe").
% Placed first so it short-circuits before the string/other clauses below.
plawk_strnum_action_unsafe_read(Action, _Set, _Name) :-
    plawk_scalar_action_update(Action, _LHS, Operation),
    plawk_strnum_numeric_operation(Operation),
    !,
    fail.
% `set(var(_), RHS)` with a NON-numeric RHS (a string concat / sprintf / ...):
% any read of Name in the RHS is unsafe (strnum in a string RHS is unsupported).
% The name's own field-copy source has RHS = field(N), which is numeric and thus
% handled by the clause above.
plawk_strnum_action_unsafe_read(set(var(_LHS), RHS), _Set, Name) :-
    !,
    plawk_strnum_term_mentions(RHS, Name).
% `print` / `emit`: a bare `var(Name)` field is fine; a field that MENTIONS
% Name but is not exactly `var(Name)` (concat, arithmetic, length, ...) is not.
plawk_strnum_action_unsafe_read(print(Fields), _Set, Name) :-
    !,
    member(Field, Fields),
    Field \== var(Name),
    plawk_strnum_term_mentions(Field, Name).
plawk_strnum_action_unsafe_read(emit(Field), _Set, Name) :-
    !,
    Field \== var(Name),
    plawk_strnum_term_mentions(Field, Name).
% scalar `if`: the condition (see below) or either branch may hold an unsafe read.
plawk_strnum_action_unsafe_read(if(scalar_if(Cond), Then, Else), Set, Name) :-
    !,
    ( plawk_strnum_cond_unsafe_read(Cond, Set, Name)
    ; member(A, Then), plawk_strnum_action_unsafe_read(A, Set, Name)
    ; member(A, Else), plawk_strnum_action_unsafe_read(A, Set, Name)
    ).
% pattern `if`: a scalar var mentioned in a record/pattern guard is unsupported;
% recurse into the branches.
plawk_strnum_action_unsafe_read(if(Pattern, Then, Else), Set, Name) :-
    !,
    ( Pattern \= scalar_if(_), plawk_strnum_term_mentions(Pattern, Name)
    ; member(A, Then), plawk_strnum_action_unsafe_read(A, Set, Name)
    ; member(A, Else), plawk_strnum_action_unsafe_read(A, Set, Name)
    ).
plawk_strnum_action_unsafe_read(while_loop(Cond, Body), Set, Name) :-
    !,
    ( plawk_strnum_cond_unsafe_read(Cond, Set, Name)
    ; member(A, Body), plawk_strnum_action_unsafe_read(A, Set, Name)
    ).
plawk_strnum_action_unsafe_read(do_while_loop(Body, Cond), Set, Name) :-
    !,
    ( plawk_strnum_cond_unsafe_read(Cond, Set, Name)
    ; member(A, Body), plawk_strnum_action_unsafe_read(A, Set, Name)
    ).
plawk_strnum_action_unsafe_read(foreach_loop(_Layout, Body), Set, Name) :-
    !,
    member(A, Body),
    plawk_strnum_action_unsafe_read(A, Set, Name).
% Any other action: mentioning Name at all is an unsupported read.
plawk_strnum_action_unsafe_read(Action, _Set, Name) :-
    plawk_strnum_term_mentions(Action, Name).

% A comparison condition reads Name unsafely when Name appears in it but not in
% a supported comparison form (strnum-vs-strnum with the partner in Set, or
% strnum-vs-string-literal).
plawk_strnum_cond_unsafe_read(and(A, B), Set, Name) :-
    !,
    ( plawk_strnum_cond_unsafe_read(A, Set, Name)
    ; plawk_strnum_cond_unsafe_read(B, Set, Name)
    ).
plawk_strnum_cond_unsafe_read(or(A, B), Set, Name) :-
    !,
    ( plawk_strnum_cond_unsafe_read(A, Set, Name)
    ; plawk_strnum_cond_unsafe_read(B, Set, Name)
    ).
plawk_strnum_cond_unsafe_read(cmp(Left, _Op, Right), Set, Name) :-
    !,
    plawk_strnum_term_mentions(cmp(Left, x, Right), Name),
    \+ plawk_strnum_cmp_supported(Left, Right, Set, Name).
plawk_strnum_cond_unsafe_read(Cond, _Set, Name) :-
    plawk_strnum_term_mentions(Cond, Name).

% Supported comparison forms that read Name: against another strnum var still in
% Set, a string literal (handled by the existing string-guard clauses and the
% strnum-vs-strnum dispatch), or an integer literal (step 3b -- dispatched to
% @wam_strnum_cmp_int, deciding numeric vs lexical by the strnum's content).
plawk_strnum_cmp_supported(var(Name), var(Other), Set, Name) :- memberchk(Other, Set).
plawk_strnum_cmp_supported(var(Other), var(Name), Set, Name) :- memberchk(Other, Set).
plawk_strnum_cmp_supported(var(Name), string(_), _Set, Name).
plawk_strnum_cmp_supported(string(_), var(Name), _Set, Name).
plawk_strnum_cmp_supported(var(Name), int(_), _Set, Name).
plawk_strnum_cmp_supported(int(_), var(Name), _Set, Name).

% A strnum-coercible numeric update: a `set`/`add` whose expression is a PURE
% arithmetic tree (binary arithmetic over fields / vars / literals / length /
% NR / NF). Reads of a strnum in such an expression are coerced to a number
% (i64 via the field parser, f64 via strtod) by the ssa_strnum leaf clauses, so
% they are supported (step 3c). Calls / ternary / index operations are
% deliberately excluded -- a strnum argument there is not coerced, so those keep
% a name on the i64 path (unchanged).
plawk_strnum_numeric_operation(set(Expr)) :- plawk_strnum_arith_expr(Expr).
plawk_strnum_numeric_operation(add(Expr)) :- plawk_strnum_arith_expr(Expr).
plawk_strnum_numeric_operation(add(const(_))).

plawk_strnum_arith_expr(var(_)) :- !.
plawk_strnum_arith_expr(int(_)) :- !.
plawk_strnum_arith_expr(const(_)) :- !.
plawk_strnum_arith_expr(field(_)) :- !.
plawk_strnum_arith_expr(field_i64(_)) :- !.
plawk_strnum_arith_expr(length(_)) :- !.
plawk_strnum_arith_expr(nf) :- !.
plawk_strnum_arith_expr(nr) :- !.
plawk_strnum_arith_expr(special('NR')) :- !.
plawk_strnum_arith_expr(special('NF')) :- !.
plawk_strnum_arith_expr(Expr) :-
    plawk_i64_binary_expr(Expr, _Op, _NamePart, Left, Right),
    plawk_strnum_arith_expr(Left),
    plawk_strnum_arith_expr(Right).

% Does var(Name) occur anywhere in Term?
plawk_strnum_term_mentions(var(Name), Name) :- !.
plawk_strnum_term_mentions(Term, Name) :-
    compound(Term),
    functor(Term, _, Arity),
    Arity > 0,
    arg(N, Term, Arg),
    plawk_strnum_term_mentions(Arg, Name),
    !.

plawk_scalar_update_name_expr(Action, Name, Expr) :-
    plawk_scalar_action_update(Action, Name, Operation),
    plawk_scalar_operation_expr(Operation, Expr).
% Type inference for a destructure bind: field Ti gives Vi a double-typed
% sentinel (f64) or an i64-typed one, so the scalar_double fixpoint types
% each slot correctly without a real RHS expression.
plawk_scalar_update_name_expr(dynrec_bind(Vars, _Call, Types), Name, Expr) :-
    nth0(I, Vars, Binding),
    nth0(I, Types, Type),
    ( Type == f64
    ->  Binding = Name, Expr = float_const(0, 1)      % double-typed slot
    ;   Type == string
    ->  Binding = str(P, L), ( Name = P ; Name = L ), Expr = int(0)  % i64 slots
    ;   Binding = Name, Expr = int(0)                 % i64 slot
    ).
plawk_scalar_update_name_expr(if(_Pattern, ThenActions, ElseActions), Name, Expr) :-
    ( member(Action, ThenActions)
    ; member(Action, ElseActions)
    ),
    plawk_scalar_update_name_expr(Action, Name, Expr).
% a scalar `if` condition (`if (i > 2)`) references scalar variables -- each
% needs an i64 slot, exactly like a loop condition.
plawk_scalar_update_name_expr(if(scalar_if(Cond), _Then, _Else), Name, int(0)) :-
    plawk_while_cond_vars(Cond, CondVars),
    member(Name, CondVars).
plawk_scalar_update_name_expr(foreach_loop(_Layout, Body), Name, Expr) :-
    member(Action, Body),
    plawk_scalar_update_name_expr(Action, Name, Expr).
% while/do-while: every condition variable gets an i64 slot (each is compared
% numerically), plus any scalar the body updates.
plawk_scalar_update_name_expr(while_loop(Cond, Body), Name, Expr) :-
    ( plawk_while_cond_vars(Cond, CondVars), member(Name, CondVars), Expr = int(0)
    ; member(Action, Body), plawk_scalar_update_name_expr(Action, Name, Expr)
    ).
plawk_scalar_update_name_expr(do_while_loop(Body, Cond), Name, Expr) :-
    ( plawk_while_cond_vars(Cond, CondVars), member(Name, CondVars), Expr = int(0)
    ; member(Action, Body), plawk_scalar_update_name_expr(Action, Name, Expr)
    ).

plawk_scalar_double_fixpoint(Updates, Doubles0, Doubles) :-
    findall(Name,
        ( member(Name-Expr, Updates),
          \+ memberchk(Name, Doubles0),
          plawk_update_expr_is_double(Expr, Doubles0)
        ),
        New0),
    sort(New0, New),
    ( New == []
    -> Doubles = Doubles0
    ;  append(Doubles0, New, Doubles1),
       plawk_scalar_double_fixpoint(Updates, Doubles1, Doubles)
    ).

plawk_update_expr_is_double(Expr, _Doubles) :-
    plawk_expr_is_double(Expr),
    !.
plawk_update_expr_is_double(Expr, Doubles) :-
    plawk_expr_scalar_read_name(Expr, Name),
    memberchk(Name, Doubles),
    !.

plawk_scalar_typed_slot(_Doubles, Strings, _Strnums, Name, scalar_string(Name)) :-
    memberchk(Name, Strings),
    !.
plawk_scalar_typed_slot(Doubles, _Strings, _Strnums, Name, scalar_double(Name)) :-
    memberchk(Name, Doubles),
    !.
% strnum precedence: below double/string (a name assigned a double leaf or a
% string RHS is disqualified from being a pure strnum anyway, so these sets are
% disjoint -- the order is defensive), above the plain i64 counter fall-through.
plawk_scalar_typed_slot(_Doubles, _Strings, Strnums, Name, scalar_strnum(Name)) :-
    memberchk(Name, Strnums),
    !.
plawk_scalar_typed_slot(_Doubles, _Strings, _Strnums, Name, scalar_counter(Name)).

plawk_mixed_state_plan(Rules, PrintFields, mixed_plan(ScalarPlan, AssocPlan, PlannedRules)) :-
    plawk_mixed_scalar_state_plan(Rules, PrintFields, ScalarPlan),
    plawk_mixed_assoc_count_plan(Rules, PrintFields, AssocPlan),
    phrase(plawk_mixed_planned_rules(Rules, AssocPlan, 0), PlannedRules),
    PlannedRules \== [],
    (   plawk_state_slot_count(ScalarPlan, ScalarCount),
        ScalarCount > 0
    ;   plawk_planned_rules_have_conditionals(PlannedRules)
    ).

plawk_mixed_scalar_state_plan(Rules, PrintFields, state_plan(Slots)) :-
    findall(Name,
        ( member(rule(_Pattern, Actions), Rules),
          plawk_trim_control_tails(Actions, ReachableActions),
          member(Action, ReachableActions),
          plawk_scalar_update_action_name(Action, Name)
        ),
        ActionVars),
    findall(Name,
        ( member(Field, PrintFields),
          plawk_scalar_print_expr(Field, Name)
        ),
        PrintVars),
    append(ActionVars, PrintVars, Names0),
    sort(Names0, Names),
    plawk_scalar_typed_slots(Rules, Names, Slots).

plawk_mixed_assoc_count_plan(Rules, PrintFields, assoc_plan(Tables, [])) :-
    findall(ArrayName,
        ( member(rule(_Pattern, Actions), Rules),
          plawk_assoc_increment_spec_in_actions(Actions, ArrayName-_KeyIndex)
        ),
        ActionArrays),
    ActionArrays \== [],
    findall(ArrayName,
        ( member(Field, PrintFields),
          plawk_assoc_print_array(Field, ArrayName)
        ),
        PrintArrays),
    PrintArrays \== [],
    append(ActionArrays, PrintArrays, ArrayNames0),
    sort(ArrayNames0, Tables).

plawk_planned_rules_have_conditionals(PlannedRules) :-
    member(mixed_rule(_Index, _Pattern, Actions, _AssocActions, _Control), PlannedRules),
    plawk_actions_have_conditional(Actions).

plawk_actions_have_conditional(Actions) :-
    member(Action, Actions),
    plawk_action_has_conditional(Action).

plawk_action_has_conditional(if(_Pattern, _ThenActions, _ElseActions)).

plawk_rules_body_print_fields(Rules, Fields) :-
    findall(Field,
        ( member(rule(_Pattern, Actions), Rules),
          plawk_actions_body_print_field(Actions, Field)
        ),
        Fields).

plawk_rules_scalar_update_exprs(Rules, Exprs) :-
    findall(Expr,
        ( member(rule(_Pattern, Actions), Rules),
          plawk_actions_scalar_update_expr(Actions, Expr)
        ),
        Exprs).

plawk_actions_scalar_update_expr(Actions, Expr) :-
    plawk_trim_control_tails(Actions, ReachableActions),
    member(Action, ReachableActions),
    plawk_action_scalar_update_expr(Action, Expr).

plawk_action_scalar_update_expr(if(Pattern, ThenActions, ElseActions), Expr) :-
    !,
    (   plawk_pattern_cond_operand_expr(Pattern, Expr)
    ;   plawk_actions_scalar_update_expr(ThenActions, Expr)
    ;   plawk_actions_scalar_update_expr(ElseActions, Expr)
    ).
plawk_action_scalar_update_expr(while_loop(Cond, Body), Expr) :-
    !,
    (   plawk_cond_operand_expr(Cond, Expr)
    ;   plawk_actions_scalar_update_expr(Body, Expr)
    ).
plawk_action_scalar_update_expr(do_while_loop(Body, Cond), Expr) :-
    !,
    (   plawk_cond_operand_expr(Cond, Expr)
    ;   plawk_actions_scalar_update_expr(Body, Expr)
    ).
plawk_action_scalar_update_expr(Action, Expr) :-
    plawk_scalar_action_update(Action, _Name, Operation),
    plawk_scalar_operation_expr(Operation, Expr).

% Expose the operands of a scalar `if`/`while` condition so specials used only in
% a guard (e.g. `if (NR > 1)`) are seen by the NR-usage detector (and any other
% expr scan), so the record counter %current_nr is defined. A non-scalar
% (field/pattern) guard contributes no scalar operands.
plawk_pattern_cond_operand_expr(scalar_if(Cond), Expr) :-
    plawk_cond_operand_expr(Cond, Expr).

plawk_cond_operand_expr(and(A, B), Expr) :-
    ( plawk_cond_operand_expr(A, Expr) ; plawk_cond_operand_expr(B, Expr) ).
plawk_cond_operand_expr(or(A, B), Expr) :-
    ( plawk_cond_operand_expr(A, Expr) ; plawk_cond_operand_expr(B, Expr) ).
plawk_cond_operand_expr(cmp(Left, _Op, Right), Expr) :-
    ( Expr = Left ; Expr = Right ).

plawk_scalar_operation_expr(add(Expr), Expr).
plawk_scalar_operation_expr(set(Expr), Expr).
% a string assignment's RHS is exposed too, so NR used inside it (e.g.
% `x = sprintf("%d", NR)`) is detected and %current_nr defined. Its shape is not
% double, so the double-type fixpoint ignores it.
plawk_scalar_operation_expr(set_str(Expr), Expr).

plawk_actions_body_print_field(Actions, Field) :-
    plawk_trim_control_tails(Actions, ReachableActions),
    member(Action, ReachableActions),
    plawk_action_body_print_field(Action, Field).

plawk_action_body_print_field(print(Fields), Field) :-
    member(Field, Fields).
plawk_action_body_print_field(printf(string(_Format), Args), Field) :-
    member(Field, Args).
plawk_action_body_print_field(if(_Pattern, ThenActions, ElseActions), Field) :-
    (   plawk_actions_body_print_field(ThenActions, Field)
    ;   plawk_actions_body_print_field(ElseActions, Field)
    ).
plawk_action_body_print_field(foreach_loop(_Layout, Body), Field) :-
    plawk_actions_body_print_field(Body, Field).
plawk_action_body_print_field(while_loop(_Cond, Body), Field) :-
    plawk_actions_body_print_field(Body, Field).
plawk_action_body_print_field(do_while_loop(Body, _Cond), Field) :-
    plawk_actions_body_print_field(Body, Field).

plawk_mixed_planned_rules([], _AssocPlan, _Index) -->
    [].
plawk_mixed_planned_rules([rule(Pattern, Actions) | Rest], assoc_plan(Tables, Actions0), Index) -->
    { plawk_split_terminal_control(Actions, BodyActions, Control),
      plawk_mixed_rule_actions(BodyActions, PlannedActions),
      plawk_assoc_increment_specs_in_actions(BodyActions, AssocSpecs),
      ( PlannedActions == [], AssocSpecs == [], Control == fallthrough
      -> HasActions = false,
         NextIndex = Index,
         PlannedAssocActions = []
      ;  HasActions = true,
         % the mixed route plans increment specs only (no dynassoc-str or
         % posarray binds reach it), so both kind-sets are empty here
         phrase(plawk_assoc_planned_actions(AssocSpecs, Tables, [], [], 0),
             PlannedAssocActions),
         NextIndex is Index + 1
      )
    },
    ( { HasActions == true }
    -> [mixed_rule(Index, Pattern, PlannedActions, PlannedAssocActions, Control)]
    ;  []
    ),
    plawk_mixed_planned_rules(Rest, assoc_plan(Tables, Actions0), NextIndex).

plawk_mixed_rule_actions([], []).
plawk_mixed_rule_actions([Action | Rest], [Action | PlannedRest]) :-
    plawk_mixed_update_action(Action),
    plawk_mixed_rule_actions(Rest, PlannedRest).

plawk_mixed_update_action(Action) :-
    plawk_scalar_action_update(Action, _Name, _Operation).
plawk_mixed_update_action(Action) :-
    plawk_assoc_update_action(Action).
plawk_mixed_update_action(Action) :-
    plawk_rule_body_print_action(Action).
plawk_mixed_update_action(if(_Pattern, ThenActions, ElseActions)) :-
    append(ThenActions, ElseActions, Actions),
    Actions \== [],
    plawk_mixed_branch_body_actions(ThenActions),
    plawk_mixed_branch_body_actions(ElseActions).

plawk_mixed_branch_body_actions(Actions) :-
    plawk_split_branch_control(Actions, BodyActions, _Control),
    maplist(plawk_mixed_update_action, BodyActions).

plawk_assoc_update_action(inc_assoc(var(_ArrayName), field(KeyIndex))) :-
    KeyIndex > 0.
plawk_assoc_update_action(inc_assoc(var(_ArrayName), Blob)) :-
    plawk_assoc_blob_key_ok(Blob).

plawk_scalar_conditional_action(if(_Pattern, ThenActions, ElseActions)) :-
    append(ThenActions, ElseActions, Actions),
    Actions \== [],
    plawk_scalar_branch_body_actions(ThenActions),
    plawk_scalar_branch_body_actions(ElseActions).

plawk_scalar_plain_update_action(Action) :-
    plawk_scalar_action_update(Action, _Name, _Operation).

plawk_split_branch_control(Actions0, BodyActions, Control) :-
    plawk_trim_control_tails(Actions0, Actions),
    plawk_split_normalized_branch_control(Actions, BodyActions, Control).

plawk_split_normalized_branch_control(Actions, BodyActions, branch_next) :-
    append(BodyActions, [next], Actions),
    !,
    \+ plawk_actions_have_control(BodyActions).
plawk_split_normalized_branch_control(Actions, BodyActions, branch_break) :-
    append(BodyActions, [break], Actions),
    !,
    \+ plawk_actions_have_control(BodyActions).
plawk_split_normalized_branch_control(Actions, BodyActions, branch_continue) :-
    append(BodyActions, [continue], Actions),
    !,
    \+ plawk_actions_have_control(BodyActions).
plawk_split_normalized_branch_control(Actions, Actions, fallthrough) :-
    \+ plawk_actions_have_control(Actions).

plawk_assoc_increment_spec_in_actions(Actions, Spec) :-
    plawk_trim_control_tails(Actions, ReachableActions),
    member(Action, ReachableActions),
    plawk_assoc_increment_spec_in_action(Action, Spec).

plawk_assoc_increment_specs_in_actions(Actions, Specs) :-
    findall(Spec, plawk_assoc_increment_spec_in_actions(Actions, Spec), Specs).

plawk_assoc_increment_spec_in_action(Action, Spec) :-
    plawk_assoc_increment_action(Action, Spec).
plawk_assoc_increment_spec_in_action(if(_Pattern, ThenActions, ElseActions), Spec) :-
    ( plawk_assoc_increment_spec_in_actions(ThenActions, Spec)
    ; plawk_assoc_increment_spec_in_actions(ElseActions, Spec)
    ).

plawk_mixed_rule_chain_ir(mixed_plan(ScalarPlan, AssocPlan, Rules), FieldSeparator, OutputSeparator,
        GlobalIR, ChainIR, RuleCount, BranchNextExits) :-
    length(Rules, RuleCount),
    RuleCount > 0,
    plawk_mixed_rule_controls(mixed_plan(ScalarPlan, AssocPlan, Rules), Controls),
    phrase(plawk_mixed_rule_chain_lines(Rules, Controls, ScalarPlan, AssocPlan, FieldSeparator, OutputSeparator, 0), Parts),
    plawk_rule_chain_parts(Parts, GlobalParts, ChainParts, BranchNextExits),
    atomic_list_concat(GlobalParts, '\n', GlobalIR),
    atomic_list_concat(ChainParts, '\n', ChainIR).

plawk_mixed_rule_controls(mixed_plan(_ScalarPlan, _AssocPlan, Rules), Controls) :-
    findall(Control, member(mixed_rule(_Index, _Pattern, _ScalarActions, _AssocActions, Control), Rules), Controls).

plawk_rule_chain_parts([], [], [], []).
plawk_rule_chain_parts([rule_chain_part(GlobalIR, ChainIR, Exits) | Rest],
        [GlobalIR | GlobalParts], [ChainIR | ChainParts], AllExits) :-
    plawk_rule_chain_parts(Rest, GlobalParts, ChainParts, RestExits),
    append(Exits, RestExits, AllExits).

plawk_mixed_rule_chain_lines([], _Controls, _ScalarPlan, _AssocPlan, _FieldSeparator, _OutputSeparator, _) -->
    [].
plawk_mixed_rule_chain_lines([mixed_rule(Index, Pattern, Actions, _AssocActions, Control) | Rest], Controls, ScalarPlan, AssocPlan, FieldSeparator, OutputSeparator, Index) -->
    { NextIndex is Index + 1,
      ( Rest == []
      -> NextLabel = 'continue_loop'
      ;  format(atom(NextLabel), 'rule_~w_match', [NextIndex])
      ),
      plawk_rule_target(Control, NextLabel, RuleTargetLabel),
      format(atom(RuleLabel), 'rule_~w_match', [Index]),
      format(atom(ApplyLabel), 'rule_~w_apply', [Index]),
      format(atom(DoneLabel), 'rule_~w_done', [Index]),
      plawk_mixed_scalar_rule_input_phi_ir(ScalarPlan, Index, Controls, InputPhiIR),
      plawk_mixed_rule_guard_ir(Pattern, Index, RuleLabel, ApplyLabel,
          NextLabel, InputPhiIR, FieldSeparator, GuardGlobalIR-GuardIR),
      plawk_native_match_update_ir(ScalarPlan, AssocPlan, Actions, FieldSeparator, OutputSeparator, Index,
          BranchNextExits, ActionGlobalIR-ActionIR),
      ( Index =:= 0
      -> EntryIR = '  br label %rule_0_match\n\n'
      ;  EntryIR = ''
      ),
      format(atom(RuleIR),
'~w~w

~w:
~w
  br label %~w

~w:
  br label %~w',
          [EntryIR, GuardIR, ApplyLabel, ActionIR, DoneLabel, DoneLabel, RuleTargetLabel]),
      format(atom(CombinedGlobalIR), '~w~n~w', [GuardGlobalIR, ActionGlobalIR]),
      Part = rule_chain_part(CombinedGlobalIR, RuleIR, BranchNextExits)
    },
    [Part],
    plawk_mixed_rule_chain_lines(Rest, Controls, ScalarPlan, AssocPlan, FieldSeparator, OutputSeparator, NextIndex).

plawk_mixed_rule_guard_ir(always, Index, RuleLabel, ApplyLabel, NextLabel,
    InputPhiIR, _FieldSeparator, ''-IR) :-
    !,
    format(atom(MatchVar), 'rule_~w_is_match', [Index]),
    format(atom(IR),
'~w:
~w  %~w = icmp eq i1 true, true
  br i1 %~w, label %~w, label %~w',
        [RuleLabel, InputPhiIR, MatchVar, MatchVar, ApplyLabel, NextLabel]).
plawk_mixed_rule_guard_ir(Pattern, Index, RuleLabel, ApplyLabel, NextLabel,
    InputPhiIR, FieldSeparator, GuardGlobalIR-IR) :-
    format(atom(MatchVar), 'rule_~w_is_match', [Index]),
    format(atom(GlobalBase), 'plawk_mixed_rule_~w', [Index]),
    format(atom(MatchValue), '%~w', [MatchVar]),
    plawk_pattern_guard_ir(Pattern, FieldSeparator, GlobalBase, MatchValue,
        GuardGlobalIR-GuardCallIR),
    format(atom(IR),
'~w:
~w~w
  br i1 %~w, label %~w, label %~w',
        [RuleLabel, InputPhiIR, GuardCallIR, MatchVar, ApplyLabel, NextLabel]).

plawk_mixed_scalar_rule_input_phi_ir(_StatePlan, 0, _Controls, '') :-
    !.
plawk_mixed_scalar_rule_input_phi_ir(StatePlan, RuleIndex, Controls, IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_mixed_scalar_rule_input_phi_lines(Slots, RuleIndex, Controls, 0), Lines),
    atomic_list_concat(Lines, '\n', LinesIR),
    format(atom(IR), '~w~n', [LinesIR]).

plawk_mixed_scalar_rule_input_phi_lines([], _RuleIndex, _Controls, _) -->
    [].
plawk_mixed_scalar_rule_input_phi_lines([Slot | Rest], RuleIndex, Controls, SlotIndex) -->
    { PrevRuleIndex is RuleIndex - 1,
      plawk_scalar_rule_input_value(PrevRuleIndex, SlotIndex, PrevFalseValue),
      format(atom(FalseIncoming), '[~w, %rule_~w_match]',
          [PrevFalseValue, PrevRuleIndex]),
      (   plawk_terminal_control_skips_next_rule(Controls, PrevRuleIndex)
      ->  Incomings = [FalseIncoming]
      ;   format(atom(ApplyIncoming), '[%rule_~w_slot_~w, %rule_~w_done]',
              [PrevRuleIndex, SlotIndex, PrevRuleIndex]),
          Incomings = [FalseIncoming, ApplyIncoming]
      ),
      atomic_list_concat(Incomings, ', ', IncomingIR),
      plawk_slot_llvm_type(Slot, Type),
      format(atom(Line), '  %rule_~w_in_slot_~w = phi ~w ~w',
          [RuleIndex, SlotIndex, Type, IncomingIR]),
      NextSlotIndex is SlotIndex + 1
    },
    [Line],
    plawk_mixed_scalar_rule_input_phi_lines(Rest, RuleIndex, Controls, NextSlotIndex).

plawk_mixed_scalar_next_phi_ir(StatePlan, RuleCount, Controls, BranchNextExits, IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_mixed_scalar_next_phi_lines(Slots, RuleCount, Controls, BranchNextExits, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_mixed_scalar_next_phi_lines([], _RuleCount, _Controls, _BranchNextExits, _) -->
    [].
plawk_mixed_scalar_next_phi_lines([Slot | Rest], RuleCount, Controls, BranchNextExits, Index) -->
    { LastRuleIndex is RuleCount - 1,
      plawk_scalar_rule_input_value(LastRuleIndex, Index, FalseValue),
      format(atom(FalseIncoming), '[~w, %rule_~w_match]',
          [FalseValue, LastRuleIndex]),
      findall(ApplyIncoming,
          ( between(0, LastRuleIndex, RuleIndex),
            ( ( RuleIndex =:= LastRuleIndex,
                \+ ( nth0(RuleIndex, Controls, LastControl),
                     memberchk(LastControl, [terminal_break, terminal_exit]) )
              )
            ; nth0(RuleIndex, Controls, terminal_next)
            ),
            format(atom(ApplyIncoming), '[%rule_~w_slot_~w, %rule_~w_done]',
                [RuleIndex, Index, RuleIndex])
          ),
          ApplyIncomings),
      plawk_branch_next_phi_incomings(BranchNextExits, Index, BranchNextIncomings),
      append([FalseIncoming | ApplyIncomings], BranchNextIncomings, Incomings),
      atomic_list_concat(Incomings, ', ', IncomingIR),
      plawk_slot_llvm_type(Slot, Type),
      format(atom(Line), '  %next_slot_~w = phi ~w ~w', [Index, Type, IncomingIR]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_mixed_scalar_next_phi_lines(Rest, RuleCount, Controls, BranchNextExits, NextIndex).

plawk_assoc_runtime_count_plan(Rules, PrintFields, Plan) :-
    plawk_assoc_plan_specs_tables(Rules, PrintFields, RuleSpecs, PrintArrays, Tables),
    % END-print driver: the program must reference a table in its END print.
    PrintArrays \== [],
    plawk_assoc_plan_from_specs(RuleSpecs, Tables, Plan).

%% plawk_assoc_runtime_record_plan(+Rules, -Plan)
%  Like plawk_assoc_runtime_count_plan but for a NO-END program whose rule
%  bodies mutate a table and print per-record. There is no END print, so the
%  plan is admitted when the rule ACTIONS alone establish at least one table.
plawk_assoc_runtime_record_plan(Rules, Plan) :-
    plawk_assoc_plan_specs_tables(Rules, [], RuleSpecs, _PrintArrays, Tables),
    Tables \== [],
    plawk_assoc_plan_from_specs(RuleSpecs, Tables, Plan).

%% plawk_assoc_plan_specs_tables(+Rules, +PrintFields, -RuleSpecs, -PrintArrays, -Tables)
%  Build the per-rule action specs and the table set: every array a rule ACTION
%  establishes (a counted inc, split, add, delete, set_row, dynassoc, or a
%  body for-in) plus every array an END print field references.
plawk_assoc_plan_specs_tables(Rules, PrintFields, RuleSpecs, PrintArrays, Tables) :-
    maplist(plawk_assoc_rule_action_specs, Rules, RuleSpecs),
    RuleSpecs \== [],
    findall(ArrayName,
        ( member(Field, PrintFields),
          plawk_assoc_print_array(Field, ArrayName)
        ),
        PrintArrays),
    findall(ArrayName,
        ( member(rule(_Pattern, ActionSpecs, _Control), RuleSpecs),
          ( member(Spec, ActionSpecs),
            plawk_assoc_spec_table_name(Spec, ArrayName)
          ; plawk_assoc_spec_forin_array(ActionSpecs, ArrayName)
          )
        ),
        ActionArrays),
    append(ActionArrays, PrintArrays, ArrayNames0),
    sort(ArrayNames0, Tables).

% The table an action spec establishes / mutates (its array name).
plawk_assoc_spec_table_name(ArrayName-_Key, ArrayName) :- atom(ArrayName).
plawk_assoc_spec_table_name(dynassoc(ArrayName, _Call), ArrayName).
plawk_assoc_spec_table_name(assoc_split(ArrayName, _K, _Sep), ArrayName).
plawk_assoc_spec_table_name(assoc_add(ArrayName, _K, _Delta), ArrayName).
plawk_assoc_spec_table_name(assoc_delete(ArrayName, _K), ArrayName).
plawk_assoc_spec_table_name(assoc_set_row(ArrayName, _K), ArrayName).
plawk_assoc_spec_table_name(assoc_set_row_cons(ArrayName, _K, _Fs), ArrayName).

plawk_assoc_plan_from_specs(RuleSpecs, Tables, assoc_plan(Tables, PlannedRules)) :-
    plawk_assoc_specs_str_arrays(RuleSpecs, StrArrays),
    plawk_assoc_specs_posarray_arrays(RuleSpecs, PosArrays),
    phrase(plawk_assoc_planned_rules(RuleSpecs, Tables, StrArrays, PosArrays,
        0), PlannedRules).

plawk_assoc_rule_controls(assoc_plan(_Tables, Rules), Controls) :-
    findall(Control, member(assoc_rule(_Index, _Pattern, _Actions, Control), Rules), Controls).

plawk_assoc_rule_action_specs(rule(Pattern, Actions), rule(Pattern, ActionSpecs, Control)) :-
    plawk_split_terminal_control(Actions, BodyActions, Control),
    ( BodyActions == []
    -> ActionSpecs = []
    ;  maplist(plawk_assoc_body_action_spec, BodyActions, ActionSpecs)
    ),
    ( ActionSpecs \== [] ; memberchk(Control, [terminal_next, terminal_break]) ).

% Per-record assoc actions: an increment (Array-KeyIndex) or a grammar
% populate (dynassoc(Array, Call)) that fills Array's table from the
% returned [K-V,...] pairs.
plawk_assoc_body_action_spec(inc_assoc(var(ArrayName), field(KeyIndex)),
        ArrayName-KeyIndex) :-
    KeyIndex > 0.
% counts[blob(dyncall...)]++ -- key the table by a runtime grammar's
% byte output (interned like a field slice).
plawk_assoc_body_action_spec(inc_assoc(var(ArrayName), Blob),
        ArrayName-blob_key(Blob)) :-
    plawk_assoc_blob_key_ok(Blob).
% `delete arr[$k]` -- remove the entry keyed by field k (v1: a field key, like
% the counted inc). Absent key is a no-op (backward-shift delete in the runtime).
plawk_assoc_body_action_spec(delete_assoc(var(ArrayName), field(KeyIndex)),
        assoc_delete(ArrayName, KeyIndex)) :-
    integer(KeyIndex), KeyIndex > 0.
% `split($N, arr, "sep")`: populate arr (a str-valued positional table, keys
% 1..n) by splitting field N on the separator. A single-char separator is a
% literal byte; a multi-char separator is a POSIX ERE regex (its own pattern,
% independent of FS).
plawk_assoc_body_action_spec(split_into(field(KeyIndex), var(ArrayName), string(Sep)),
        assoc_split(ArrayName, KeyIndex, Sep)) :-
    integer(KeyIndex), KeyIndex >= 0,
    string(Sep), string_length(Sep, Len), Len >= 1.
% Associative add-assign `arr[$k] += DELTA`: fold DELTA (a field value or an
% integer constant) into the table at the key interned from field k. The
% general form of `arr[$k]++` and the pass-1 half of per-key normalise. v1
% keys on a field (`arr[$k]`); other key shapes are a follow-on.
plawk_assoc_body_action_spec(add_assoc(var(ArrayName), field(KeyIndex), Delta),
        assoc_add(ArrayName, KeyIndex, Delta)) :-
    integer(KeyIndex), KeyIndex > 0,
    plawk_assoc_add_delta_ok(Delta).

plawk_assoc_add_delta_ok(field(V)) :- integer(V), V > 0.
plawk_assoc_add_delta_ok(int(V)) :- integer(V).
% Row capture `arr[$k] = $0` (§3.6): store the whole current record ($0) as a
% str-value (the record's bytes, interned) in the table at the key interned
% from field k. The first producer of row-valued tables; a later pass reads
% the row back (`over TABLE`, or `records of` with a schema). The table is
% marked str-valued so value reads resolve to text.
plawk_assoc_body_action_spec(set_row(var(ArrayName), field(KeyIndex)),
        assoc_set_row(ArrayName, KeyIndex)) :-
    integer(KeyIndex), KeyIndex > 0.
% Row constructor `arr[$k] = row($a, $b, ...)`: store a row built from the
% chosen fields (in that order), joined by the field separator so a reader's
% field projection recovers the columns. Like set_row, a str-value.
plawk_assoc_body_action_spec(set_row_cons(var(ArrayName), field(KeyIndex), Fields),
        assoc_set_row_cons(ArrayName, KeyIndex, Indexes)) :-
    integer(KeyIndex), KeyIndex > 0,
    Fields = [_ | _],
    maplist(plawk_row_cons_field, Fields, Indexes).

plawk_row_cons_field(field(N), N) :- integer(N), N > 0.
plawk_assoc_body_action_spec(dynassoc_bind(var(ArrayName), Call),
        dynassoc(ArrayName, Call)) :-
    plawk_dynrec_call_ok(Call).
% str-valued table kind: the call rides the same dynassoc spec wrapped in
% str(...), so planning and the apply emitter reuse the same action --
% only the shim name (via plawk_dynassoc_call_parts) and the table's
% declared value kind differ.
plawk_assoc_body_action_spec(dynassoc_bind_str(var(ArrayName), Call),
        dynassoc(ArrayName, str(Call))) :-
    plawk_dynrec_call_ok(Call).
% positional-array target: rides the same dynassoc spec wrapped in
% posarray(...), so planning, the i64 table, for-in, and lookups reuse
% the assoc machinery unchanged -- only the shim name (via
% plawk_dynassoc_call_parts) and the runtime walk (a flat list into keys
% 1..n) differ. Values are i64, so the table is NOT str-marked.
plawk_assoc_body_action_spec(dynposarray_bind(var(ArrayName), Call),
        dynassoc(ArrayName, posarray(Call))) :-
    plawk_dynrec_call_ok(Call).
% str-valued positional array: the call rides the dynassoc spec wrapped
% in posarray_str(...) -- integer position KEYS (like posarray) and atom
% VALUES resolved to text (like str), so the array name joins BOTH the
% posarray set and the str-array set.
plawk_assoc_body_action_spec(dynposarray_bind_str(var(ArrayName), Call),
        dynassoc(ArrayName, posarray_str(Call))) :-
    plawk_dynrec_call_ok(Call).
% rule-body for-in: per-record iteration over an assoc table with a
% print body -- the END for-in's field shapes (loop key / lookups keyed
% by it / string literals), emitted as a loop inside the rule's action
% chain. Field plans (table indexes, i64-vs-str value kinds) resolve at
% planning time.
% Per-record output in an assoc program: `print` inside the record loop
% (alongside table updates), so a program can emit one line per record while
% maintaining/reading a table -- e.g. `{ c[$1]++ ; print $1, c[$1] }`, and
% (with multi-pass) a pass-2 that reads a table pass-1 built. Fields are a
% text field `$N`, or a table lookup `arr[$N]` (the table's i64 value at the
% key interned from field N). String-literal fields are a follow-on.
plawk_assoc_body_action_spec(print(Fields), assoc_print(FieldSpecs)) :-
    Fields = [_ | _],
    maplist(plawk_assoc_print_field_spec, Fields, FieldSpecs).

% Scalar accumulator in an assoc program: `acc += 1` (count) or `acc += $N`
% (sum a field). Backed by a module global (zero-initialised), so in a
% multi-pass program the accumulator persists across passes -- pass 1 folds,
% a later pass reads it (`print $1, acc`). One line per record semantics.
plawk_assoc_body_action_spec(add(var(Name), int(V)), scalar_add(Name, int(V))) :-
    integer(V).
plawk_assoc_body_action_spec(add(var(Name), field(K)), scalar_add(Name, field(K))) :-
    integer(K), K > 0.

plawk_assoc_print_field_spec(field(N), fld(N)) :-
    integer(N), N > 0.
% `print $0` -- the whole record.
plawk_assoc_print_field_spec(field(0), record).
% `print "text"` -- a string literal field.
plawk_assoc_print_field_spec(string(V), strlit(V)) :-
    string(V).
plawk_assoc_print_field_spec(assoc(var(Arr), field(N)), lookup(Arr, N)) :-
    integer(N), N > 0.
% `print arr[N]` -- an integer-literal element read (split / positional tables
% are keyed by the raw integer position). The value kind (str atom id vs i64) is
% resolved from the str-array set at plan time.
plawk_assoc_print_field_spec(assoc(var(Arr), int(N)), lookup_int(Arr, N)) :-
    atom(Arr), integer(N), N >= 1.
% A scalar accumulator read in a per-record print.
plawk_assoc_print_field_spec(var(Name), svar(Name)) :-
    atom(Name).
% `print "x=" c[$1]` -- a juxtaposition concat: the parts print adjacently (no
% separator). Parts are spec'd recursively, so a concat supports the same part
% kinds as a comma-list field (literal, `$N`, `$0`, `arr[N]`, `arr[$k]`, scalar).
plawk_assoc_print_field_spec(concat(Parts), concat_field(PartSpecs)) :-
    Parts = [_, _ | _],
    maplist(plawk_assoc_print_field_spec, Parts, PartSpecs).
% Arithmetic in a per-record print, e.g. `$2 / total` (normalise). The
% surface `/` is integer (div_i64), which truncates fractions to 0, so a
% print arithmetic expression is evaluated in f64 and printed with %g:
% fields via @wam_atom_field_f64_value, a scalar via load+sitofp, an int
% constant as a double literal. Operands are field/scalar/int (a table
% lookup in arithmetic is a follow-on).
plawk_assoc_print_field_spec(Expr, farith(F64Op, LOperand, ROperand)) :-
    Expr =.. [Op, L, R],
    plawk_i64_op_f64(Op, F64Op),
    plawk_assoc_arith_operand(L, LOperand),
    plawk_assoc_arith_operand(R, ROperand).

% Surface i64 arithmetic operator -> the f64 LLVM opcode for print arithmetic.
plawk_i64_op_f64(add_i64, fadd).
plawk_i64_op_f64(sub_i64, fsub).
plawk_i64_op_f64(mul_i64, fmul).
plawk_i64_op_f64(div_i64, fdiv).
plawk_i64_op_f64(mod_i64, frem).

% An arithmetic operand in a print expression.
plawk_assoc_arith_operand(field(N), afield(N)) :- integer(N), N > 0.
plawk_assoc_arith_operand(var(Name), asvar(Name)) :- atom(Name).
plawk_assoc_arith_operand(int(V), aint(V)) :- integer(V).
% A table lookup `arr[$N]` as an operand, e.g. `$2 / total[$1]` (per-key
% normalise). The array is resolved to a table index at planning time.
plawk_assoc_arith_operand(assoc(var(Arr), field(N)), alookup(Arr, N)) :-
    atom(Arr), integer(N), N > 0.

% Resolve a print field's lookup array to its table index in the plan.
% StrArrays is the set of str-valued (atom-id) tables, used to pick the value
% kind for an integer-key element read.
plawk_assoc_print_plan_field(_Tables, _StrArrays, fld(N), fld(N)).
plawk_assoc_print_plan_field(_Tables, _StrArrays, record, record).
plawk_assoc_print_plan_field(_Tables, _StrArrays, strlit(V), strlit(V)).
plawk_assoc_print_plan_field(Tables, _StrArrays, lookup(Arr, N), lookup(TableIndex, N)) :-
    nth0(TableIndex, Tables, Arr).
% `arr[N]` element read: raw integer key; value kind from the str-array set.
plawk_assoc_print_plan_field(Tables, StrArrays, lookup_int(Arr, N),
        lookup_int(TableIndex, N, Kind)) :-
    nth0(TableIndex, Tables, Arr),
    ( memberchk(Arr, StrArrays) -> Kind = str ; Kind = i64 ).
plawk_assoc_print_plan_field(_Tables, _StrArrays, svar(Name), svar(Name)).
plawk_assoc_print_plan_field(Tables, StrArrays, concat_field(PartSpecs),
        concat_field(Planned)) :-
    maplist(plawk_assoc_print_plan_field(Tables, StrArrays), PartSpecs, Planned).
plawk_assoc_print_plan_field(Tables, _StrArrays, farith(Op, L, R), farith(Op, L2, R2)) :-
    plawk_assoc_arith_operand_plan(Tables, L, L2),
    plawk_assoc_arith_operand_plan(Tables, R, R2).

% Resolve an arithmetic operand against the table set: a lookup operand
% binds its array to a table index; field/scalar/int operands pass through.
plawk_assoc_arith_operand_plan(Tables, alookup(Arr, N), alookup_at(TableIndex, N)) :-
    nth0(TableIndex, Tables, Arr).
plawk_assoc_arith_operand_plan(_Tables, afield(N), afield(N)).
plawk_assoc_arith_operand_plan(_Tables, asvar(Name), asvar(Name)).
plawk_assoc_arith_operand_plan(_Tables, aint(V), aint(V)).

% The per-record source value added to a scalar accumulator: a constant, or
% field N converted to i64.
plawk_assoc_scalar_src_lines(int(V), _Base, _FieldSep, V, []).
plawk_assoc_scalar_src_lines(field(K), Base, FieldSep, SrcVar, [Line]) :-
    format(atom(Line),
        '  %~w_fv = call i64 @wam_atom_field_i64_value(%Value %line, i64 ~w, i8 ~w)',
        [Base, K, FieldSep]),
    format(atom(SrcVar), '%~w_fv', [Base]).

%% plawk_assoc_print_field_lines(+PlannedFields, +Base, +FieldSep, +Index, -Lines)
%  Emit the per-record print: each field preceded by a space separator
%  (except the first). `fld(N)` prints field N's text (slice, %.*s); a
%  `lookup(TableIndex, N)` interns field N and prints the table's i64 value
%  there (%ld, 0 if absent).
plawk_assoc_print_field_lines([], _Base, _FieldSep, _Index, []).
plawk_assoc_print_field_lines([Field | Rest], Base, FieldSep, Index, Lines) :-
    ( Index =:= 0
    ->  SepLines = []
    ;   format(atom(SepLine),
            '  %~w_sep~w = call i32 @putchar(i32 32)', [Base, Index]),
        SepLines = [SepLine]
    ),
    plawk_assoc_print_one_field(Field, Base, Index, FieldSep, FieldLs),
    NextIndex is Index + 1,
    plawk_assoc_print_field_lines(Rest, Base, FieldSep, NextIndex, RestLs),
    append([SepLines, FieldLs, RestLs], Lines).

% A juxtaposition concat prints its parts adjacently: emit each part's lines
% with a unique sub-base and NO inter-part separator (the comma-list separator
% lives one level up, between whole fields). Each part reuses the per-field
% emitters below (so a literal part still rides the global channel).
plawk_assoc_print_one_field(concat_field(Parts), Base, Index, FieldSep, Lines) :-
    format(atom(SubBase), '~w_cc~w', [Base, Index]),
    plawk_assoc_concat_part_lines(Parts, SubBase, 0, FieldSep, Lines).

plawk_assoc_concat_part_lines([], _SubBase, _PartIdx, _FieldSep, []).
plawk_assoc_concat_part_lines([Part | Rest], SubBase, PartIdx, FieldSep, Lines) :-
    plawk_assoc_print_one_field(Part, SubBase, PartIdx, FieldSep, PartLines),
    NextIdx is PartIdx + 1,
    plawk_assoc_concat_part_lines(Rest, SubBase, NextIdx, FieldSep, RestLines),
    append(PartLines, RestLines, Lines).

plawk_assoc_print_one_field(fld(N), Base, Index, FieldSep, Lines) :-
    format(atom(P), '~w_f~w', [Base, Index]),
    format(atom(SliceL),
        '  %~w_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)',
        [P, N, FieldSep]),
    format(atom(PtrL), '  %~w_ptr = extractvalue %WamSlice %~w_slice, 0', [P, P]),
    format(atom(LenL), '  %~w_len = extractvalue %WamSlice %~w_slice, 1', [P, P]),
    format(atom(LenwL), '  %~w_lenw = trunc i64 %~w_len to i32', [P, P]),
    format(atom(FmtL),
        '  %~w_fmt = getelementptr [5 x i8], [5 x i8]* @.plawk_surface_print_slice, i32 0, i32 0',
        [P]),
    format(atom(PrL),
        '  %~w_pr = call i32 (i8*, ...) @printf(i8* %~w_fmt, i32 %~w_lenw, i8* %~w_ptr)',
        [P, P, P, P]),
    Lines = [SliceL, PtrL, LenL, LenwL, FmtL, PrL].
% `print $0` -- the whole record: resolve %line's atom to its NUL-terminated
% text (like the split-$0 source) and print it as %s.
plawk_assoc_print_one_field(record, Base, Index, _FieldSep, Lines) :-
    format(atom(P), '~w_f~w', [Base, Index]),
    format(atom(LpL), '  %~w_lp = call i64 @value_payload(%Value %line)', [P]),
    format(atom(SL), '  %~w_s = call i8* @wam_atom_to_string(i64 %~w_lp)', [P, P]),
    format(atom(FmtL),
        '  %~w_fmt = getelementptr [3 x i8], [3 x i8]* @.plawk_surface_print_string, i32 0, i32 0',
        [P]),
    format(atom(PrL),
        '  %~w_pr = call i32 (i8*, ...) @printf(i8* %~w_fmt, i8* %~w_s)', [P, P, P]),
    Lines = [LpL, SL, FmtL, PrL].
% `print "text"` -- a string literal: its bytes ride the global channel (a
% module constant, lifted by plawk_partition_global_lines), printed as %s.
plawk_assoc_print_one_field(strlit(V), Base, Index, _FieldSep, Lines) :-
    format(atom(P), '~w_f~w', [Base, Index]),
    format(atom(GName), '~w_lit', [P]),
    llvm_emit_c_string_global(GName, V, Global, _StrLen, BytesLen),
    format(atom(PtrL),
        '  %~w_ptr = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0',
        [P, BytesLen, BytesLen, GName]),
    format(atom(FmtL),
        '  %~w_fmt = getelementptr [3 x i8], [3 x i8]* @.plawk_surface_print_string, i32 0, i32 0',
        [P]),
    format(atom(PrL),
        '  %~w_pr = call i32 (i8*, ...) @printf(i8* %~w_fmt, i8* %~w_ptr)', [P, P, P]),
    Lines = [global(Global), PtrL, FmtL, PrL].
plawk_assoc_print_one_field(lookup(TableIndex, N), Base, Index, FieldSep, Lines) :-
    format(atom(P), '~w_f~w', [Base, Index]),
    format(atom(SliceL),
        '  %~w_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)',
        [P, N, FieldSep]),
    format(atom(PtrL), '  %~w_ptr = extractvalue %WamSlice %~w_slice, 0', [P, P]),
    format(atom(LenL), '  %~w_len = extractvalue %WamSlice %~w_slice, 1', [P, P]),
    format(atom(KidL),
        '  %~w_kid = call i64 @wam_intern_atom(i8* %~w_ptr, i64 %~w_len)', [P, P, P]),
    format(atom(ValL),
        '  %~w_val = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %~w_kid)',
        [P, TableIndex, P]),
    format(atom(FmtL),
        '  %~w_fmt = getelementptr [4 x i8], [4 x i8]* @.plawk_surface_print_i64, i32 0, i32 0',
        [P]),
    format(atom(PrL),
        '  %~w_pr = call i32 (i8*, ...) @printf(i8* %~w_fmt, i64 %~w_val)', [P, P, P]),
    Lines = [SliceL, PtrL, LenL, KidL, ValL, FmtL, PrL].
% `print arr[N]` -- an integer-key element read. The key is the raw position N
% (split / positional tables are keyed by integer). A str element resolves the
% atom id to text (%s); an i64 element (counter / numeric posarray) prints the
% value (%ld). Absent element -> the empty atom / 0, matching an uninitialised
% awk element.
plawk_assoc_print_one_field(lookup_int(TableIndex, N, str), Base, Index, _FieldSep, Lines) :-
    format(atom(P), '~w_f~w', [Base, Index]),
    format(atom(ValL),
        '  %~w_val = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_~w, i64 ~w)',
        [P, TableIndex, N]),
    format(atom(StrL),
        '  %~w_s = call i8* @wam_atom_to_string(i64 %~w_val)', [P, P]),
    format(atom(FmtL),
        '  %~w_fmt = getelementptr [3 x i8], [3 x i8]* @.plawk_surface_print_string, i32 0, i32 0',
        [P]),
    format(atom(PrL),
        '  %~w_pr = call i32 (i8*, ...) @printf(i8* %~w_fmt, i8* %~w_s)', [P, P, P]),
    Lines = [ValL, StrL, FmtL, PrL].
plawk_assoc_print_one_field(lookup_int(TableIndex, N, i64), Base, Index, _FieldSep, Lines) :-
    format(atom(P), '~w_f~w', [Base, Index]),
    format(atom(ValL),
        '  %~w_val = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_~w, i64 ~w)',
        [P, TableIndex, N]),
    format(atom(FmtL),
        '  %~w_fmt = getelementptr [4 x i8], [4 x i8]* @.plawk_surface_print_i64, i32 0, i32 0',
        [P]),
    format(atom(PrL),
        '  %~w_pr = call i32 (i8*, ...) @printf(i8* %~w_fmt, i64 %~w_val)', [P, P, P]),
    Lines = [ValL, FmtL, PrL].
plawk_assoc_print_one_field(svar(Name), Base, Index, _FieldSep, Lines) :-
    format(atom(P), '~w_f~w', [Base, Index]),
    format(atom(LoadL), '  %~w_sv = load i64, i64* @plawk_scalar_~w', [P, Name]),
    format(atom(FmtL),
        '  %~w_fmt = getelementptr [4 x i8], [4 x i8]* @.plawk_surface_print_i64, i32 0, i32 0',
        [P]),
    format(atom(PrL),
        '  %~w_pr = call i32 (i8*, ...) @printf(i8* %~w_fmt, i64 %~w_sv)', [P, P, P]),
    Lines = [LoadL, FmtL, PrL].
% Print arithmetic in f64: evaluate both operands as doubles, apply the op,
% print with %g.
plawk_assoc_print_one_field(farith(F64Op, LOperand, ROperand), Base, Index, FieldSep, Lines) :-
    format(atom(P), '~w_f~w', [Base, Index]),
    plawk_assoc_arith_operand_lines(LOperand, P, l, FieldSep, LVar, LLines),
    plawk_assoc_arith_operand_lines(ROperand, P, r, FieldSep, RVar, RLines),
    format(atom(OpL), '  %~w_res = ~w double ~w, ~w', [P, F64Op, LVar, RVar]),
    format(atom(FmtL),
        '  %~w_fmt = getelementptr [3 x i8], [3 x i8]* @.plawk_surface_print_f64, i32 0, i32 0',
        [P]),
    format(atom(PrL),
        '  %~w_pr = call i32 (i8*, ...) @printf(i8* %~w_fmt, double %~w_res)', [P, P, P]),
    append([LLines, RLines, [OpL, FmtL, PrL]], Lines).

%% plawk_assoc_arith_operand_lines(+Operand, +P, +Slot, +FieldSep, -ValVar, -Lines)
%  Emit IR computing a print-arithmetic operand as a double. `afield(N)`
%  reads field N as f64; `asvar(Name)` loads the scalar global and promotes
%  it (sitofp); `aint(V)` promotes the integer constant.
plawk_assoc_arith_operand_lines(afield(N), P, Slot, FieldSep, ValVar, [Line]) :-
    format(atom(ValVar), '%~w_~w', [P, Slot]),
    format(atom(Line),
        '  ~w = call double @wam_atom_field_f64_value(%Value %line, i64 ~w, i8 ~w)',
        [ValVar, N, FieldSep]).
plawk_assoc_arith_operand_lines(asvar(Name), P, Slot, _FieldSep, ValVar, [LoadL, PromL]) :-
    format(atom(IVar), '%~w_~w_i', [P, Slot]),
    format(atom(ValVar), '%~w_~w', [P, Slot]),
    format(atom(LoadL), '  ~w = load i64, i64* @plawk_scalar_~w', [IVar, Name]),
    format(atom(PromL), '  ~w = sitofp i64 ~w to double', [ValVar, IVar]).
plawk_assoc_arith_operand_lines(aint(V), P, Slot, _FieldSep, ValVar, [Line]) :-
    format(atom(ValVar), '%~w_~w', [P, Slot]),
    format(atom(Line), '  ~w = sitofp i64 ~w to double', [ValVar, V]).
% Table lookup operand: intern field N, read the table's i64 value there
% (0 if absent), promote to double.
plawk_assoc_arith_operand_lines(alookup_at(TableIndex, N), P, Slot, FieldSep, ValVar,
        [SliceL, PtrL, LenL, KidL, ValL, PromL]) :-
    format(atom(B), '~w_~w', [P, Slot]),
    format(atom(ValVar), '%~w_lu', [B]),
    format(atom(SliceL),
        '  %~w_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)',
        [B, N, FieldSep]),
    format(atom(PtrL), '  %~w_ptr = extractvalue %WamSlice %~w_slice, 0', [B, B]),
    format(atom(LenL), '  %~w_len = extractvalue %WamSlice %~w_slice, 1', [B, B]),
    format(atom(KidL),
        '  %~w_kid = call i64 @wam_intern_atom(i8* %~w_ptr, i64 %~w_len)', [B, B, B]),
    format(atom(ValL),
        '  %~w_val = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %~w_kid)',
        [B, TableIndex, B]),
    format(atom(PromL), '  ~w = sitofp i64 %~w_val to double', [ValVar, B]).

plawk_assoc_body_action_spec(for_in(var(LoopVar), var(ArrayName), Body),
        forin(LoopVar, ArrayName, Fields)) :-
    Body = [print(Fields)],
    Fields = [_ | _],
    maplist(plawk_forin_print_field(LoopVar), Fields).
% for-in FILTER (assoc for-in stage 1): a per-key print gated by a guard
% on the loop key or the iterated value. `k` / `arr[k]` are for-in-scoped
% operands (see PLAWK_ASSOC_FORIN.md).
plawk_assoc_body_action_spec(
        for_in(var(LoopVar), var(ArrayName), [if(Guard, [print(Fields)], [])]),
        forin_guarded(LoopVar, ArrayName, Guard, Fields)) :-
    plawk_forin_guard_ok(Guard, LoopVar, ArrayName),
    Fields = [_ | _],
    maplist(plawk_forin_print_field(LoopVar), Fields).

% Stage 1 guards: `arr[k] CMP int` where arr is the iterated table
% (value_self), or `k CMP int` (the raw loop key).
plawk_forin_guard_ok(forin_val_cmp(Array, LoopVar, _Op, _V), LoopVar, Array).
plawk_forin_guard_ok(forin_key_cmp(LoopVar, _Op, _V), LoopVar, _Array).

plawk_assoc_increment_action(inc_assoc(var(ArrayName), field(KeyIndex)), ArrayName-KeyIndex) :-
    KeyIndex > 0.
plawk_assoc_increment_action(inc_assoc(var(ArrayName), Blob),
        ArrayName-blob_key(Blob)) :-
    plawk_assoc_blob_key_ok(Blob).

%% plawk_assoc_blob_key_ok(+Blob) is semidet.
%  Blob-key shapes: the same validation as any blob call position (the
%  marshal's global constants ride the apply stream as global markers).
plawk_assoc_blob_key_ok(Blob) :-
    plawk_blob_call_arg_ok(Blob).

plawk_assoc_print_array(assoc(var(ArrayName), string(_Key)), ArrayName).
plawk_assoc_print_array(assoc(var(ArrayName), int(_Key)), ArrayName).

plawk_assoc_planned_rules([], _Tables, _StrArrays, _PosArrays, _Index) -->
    [].
plawk_assoc_planned_rules([rule(Pattern, ActionSpecs, Control) | Rest], Tables,
        StrArrays, PosArrays, Index) -->
    { phrase(plawk_assoc_planned_actions(ActionSpecs, Tables, StrArrays,
          PosArrays, 0), PlannedActions),
      NextIndex is Index + 1
    },
    [assoc_rule(Index, Pattern, PlannedActions, Control)],
    plawk_assoc_planned_rules(Rest, Tables, StrArrays, PosArrays, NextIndex).

plawk_assoc_planned_actions([], _Tables, _StrArrays, _PosArrays, _Index) -->
    [].
plawk_assoc_planned_actions([ArrayName-KeyIndex | Rest], Tables, StrArrays,
        PosArrays, Index) -->
    { nth0(TableIndex, Tables, ArrayName),
      NextIndex is Index + 1
    },
    [assoc_action(Index, ArrayName, TableIndex, KeyIndex)],
    plawk_assoc_planned_actions(Rest, Tables, StrArrays, PosArrays, NextIndex).
plawk_assoc_planned_actions([assoc_delete(ArrayName, KeyIndex) | Rest],
        Tables, StrArrays, PosArrays, Index) -->
    { nth0(TableIndex, Tables, ArrayName),
      NextIndex is Index + 1
    },
    [assoc_delete_action(Index, ArrayName, TableIndex, KeyIndex)],
    plawk_assoc_planned_actions(Rest, Tables, StrArrays, PosArrays, NextIndex).
plawk_assoc_planned_actions([assoc_split(ArrayName, KeyIndex, Sep) | Rest],
        Tables, StrArrays, PosArrays, Index) -->
    { nth0(TableIndex, Tables, ArrayName),
      NextIndex is Index + 1
    },
    [assoc_split_action(Index, ArrayName, TableIndex, KeyIndex, Sep)],
    plawk_assoc_planned_actions(Rest, Tables, StrArrays, PosArrays, NextIndex).
plawk_assoc_planned_actions([assoc_add(ArrayName, KeyIndex, Delta) | Rest],
        Tables, StrArrays, PosArrays, Index) -->
    { nth0(TableIndex, Tables, ArrayName),
      NextIndex is Index + 1
    },
    [assoc_add_action(Index, ArrayName, TableIndex, KeyIndex, Delta)],
    plawk_assoc_planned_actions(Rest, Tables, StrArrays, PosArrays, NextIndex).
plawk_assoc_planned_actions([assoc_set_row(ArrayName, KeyIndex) | Rest],
        Tables, StrArrays, PosArrays, Index) -->
    { nth0(TableIndex, Tables, ArrayName),
      NextIndex is Index + 1
    },
    [assoc_set_row_action(Index, ArrayName, TableIndex, KeyIndex)],
    plawk_assoc_planned_actions(Rest, Tables, StrArrays, PosArrays, NextIndex).
plawk_assoc_planned_actions([assoc_set_row_cons(ArrayName, KeyIndex, Fields) | Rest],
        Tables, StrArrays, PosArrays, Index) -->
    { nth0(TableIndex, Tables, ArrayName),
      NextIndex is Index + 1
    },
    [assoc_set_row_cons_action(Index, ArrayName, TableIndex, KeyIndex, Fields)],
    plawk_assoc_planned_actions(Rest, Tables, StrArrays, PosArrays, NextIndex).
plawk_assoc_planned_actions([dynassoc(ArrayName, Call) | Rest], Tables,
        StrArrays, PosArrays, Index) -->
    { nth0(TableIndex, Tables, ArrayName),
      NextIndex is Index + 1
    },
    [assoc_dyn_action(Index, ArrayName, TableIndex, Call)],
    plawk_assoc_planned_actions(Rest, Tables, StrArrays, PosArrays, NextIndex).
% rule-body for-in: resolve each print field to a plan (loop key /
% iterated-table value / lookup by key / literal), with the value kind
% (i64 vs str) baked in from the str-array set, so the emitter needs no
% program context. A positional-array iterated table gives a numeric loop
% key (key_int) instead of the atom-resolved default.
plawk_assoc_planned_actions([assoc_print(FieldSpecs) | Rest],
        Tables, StrArrays, PosArrays, Index) -->
    { maplist(plawk_assoc_print_plan_field(Tables, StrArrays), FieldSpecs, PlannedFields),
      NextIndex is Index + 1
    },
    [assoc_print_action(Index, PlannedFields)],
    plawk_assoc_planned_actions(Rest, Tables, StrArrays, PosArrays, NextIndex).
plawk_assoc_planned_actions([scalar_add(Name, Src) | Rest],
        Tables, StrArrays, PosArrays, Index) -->
    { NextIndex is Index + 1 },
    [assoc_scalar_add_action(Index, Name, Src)],
    plawk_assoc_planned_actions(Rest, Tables, StrArrays, PosArrays, NextIndex).
plawk_assoc_planned_actions([forin(LoopVar, ArrayName, Fields) | Rest],
        Tables, StrArrays, PosArrays, Index) -->
    { nth0(TableIndex, Tables, ArrayName),
      maplist(plawk_forin_rule_field_plan(LoopVar, ArrayName, TableIndex,
          Tables, StrArrays, PosArrays), Fields, FieldPlans),
      NextIndex is Index + 1
    },
    [assoc_forin_action(Index, TableIndex, FieldPlans)],
    plawk_assoc_planned_actions(Rest, Tables, StrArrays, PosArrays, NextIndex).
plawk_assoc_planned_actions(
        [forin_guarded(LoopVar, ArrayName, Guard, Fields) | Rest],
        Tables, StrArrays, PosArrays, Index) -->
    { nth0(TableIndex, Tables, ArrayName),
      plawk_forin_guard_plan(Guard, GuardPlan),
      maplist(plawk_forin_rule_field_plan(LoopVar, ArrayName, TableIndex,
          Tables, StrArrays, PosArrays), Fields, FieldPlans),
      NextIndex is Index + 1
    },
    [assoc_forin_guarded_action(Index, TableIndex, GuardPlan, FieldPlans)],
    plawk_assoc_planned_actions(Rest, Tables, StrArrays, PosArrays, NextIndex).

% guard operand: the iterated table's value at the current slot, or the
% raw loop key. The key is always a genuine i64 (a position / registry id);
% the value depends on the table's kind. On an i64-valued table (a counter)
% `arr[k] CMP int` is a raw i64 icmp; on a STR-valued table (split / assoc(str))
% the stored i64 is an atom id, not a number, so an integer value comparison
% must resolve the element text and go through strnum (guard_value_strnum) --
% a raw icmp on the atom id would compare registry positions, not values.
plawk_forin_guard_plan(Guard, GuardPlan) :-
    plawk_forin_guard_plan(Guard, false, GuardPlan).
% A string RHS (`arr[k] CMP "x"`) is a string compare of the element.
% `==`/`!=` intern the literal and icmp its atom id against the stored id
% (canonical interning, so equal strings share an id); the ordering ops resolve
% the element text and strcmp it against the literal. Independent of IsStr --
% the RHS type drives it.
plawk_forin_guard_plan(forin_val_cmp(_A, _K, Op, str(Text)), _IsStr, GuardPlan) :-
    !,
    ( plawk_forin_streq_op(Op)
    ->  GuardPlan = guard_value_streq(Op, Text)
    ;   GuardPlan = guard_value_strord(Op, Text)
    ).
% A float-literal RHS (`arr[k] < 3.5`): on a str-valued table go through strnum
% against the double (numeric vs the value, else lexical against %g of it); on an
% i64 counter table widen the count to double and fcmp.
plawk_forin_guard_plan(forin_val_cmp(_A, _K, Op, float_const(M, D)), IsStr,
        GuardPlan) :-
    !,
    ( IsStr == true
    ->  GuardPlan = guard_value_strnum_f(Op, M, D)
    ;   GuardPlan = guard_value_f(Op, M, D)
    ).
plawk_forin_guard_plan(forin_val_cmp(_A, _K, Op, V), IsStr, GuardPlan) :-
    integer(V),
    ( IsStr == true
    ->  GuardPlan = guard_value_strnum(Op, V)
    ;   GuardPlan = guard_value(Op, V)
    ).
plawk_forin_guard_plan(forin_key_cmp(_K, Op, V), _IsStr, guard_key(Op, V)).

plawk_forin_rule_field_plan(LoopVar, ArrayName, _TableIndex, _Tables,
        _StrArrays, PosArrays, var(LoopVar), KeyPlan) :- !,
    ( memberchk(ArrayName, PosArrays) -> KeyPlan = key_int ; KeyPlan = key ).
plawk_forin_rule_field_plan(LoopVar, ArrayName, TableIndex, _Tables,
        StrArrays, _PosArrays, assoc(var(ArrayName), var(LoopVar)),
        value_self(TableIndex, Kind)) :-
    !,
    ( memberchk(ArrayName, StrArrays) -> Kind = str ; Kind = i64 ).
plawk_forin_rule_field_plan(LoopVar, _ArrayName, _TableIndex, Tables,
        StrArrays, _PosArrays, assoc(var(LookupName), var(LoopVar)),
        value_lookup(LookupIndex, Kind)) :-
    !,
    nth0(LookupIndex, Tables, LookupName),
    ( memberchk(LookupName, StrArrays) -> Kind = str ; Kind = i64 ).
plawk_forin_rule_field_plan(_LoopVar, _ArrayName, _TableIndex, _Tables,
        _StrArrays, _PosArrays, string(Value), lit(Value)).

plawk_assoc_entry_setup_ir(Plan, IR) :-
    plawk_assoc_entry_setup_ir(Plan, [], IR).

%% plawk_assoc_entry_setup_ir(+Plan, +CacheEntries, -IR)
%  CacheEntries is a list of cache(TableIndex, PathGlobalBytesLen) for tables
%  backed by a persistent store (multi-pass cache, phase 1b). A cache-backed
%  table is created like any assoc table, then loaded from its store file so
%  a pre-populated or prior-run store is read in. [] => no cache (the /2
%  form), so non-cache programs emit exactly the old IR.
plawk_assoc_entry_setup_ir(assoc_plan(Tables, _Actions), CacheEntries, IR) :-
    phrase(plawk_assoc_entry_setup_lines(Tables, 0, CacheEntries), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_assoc_entry_setup_lines([], _, _CacheEntries) -->
    [].
plawk_assoc_entry_setup_lines([_ArrayName | Rest], Index, CacheEntries) -->
    { format(atom(NewLine),
          '  %plawk_assoc_table_~w = call %WamAssocI64Table* @wam_assoc_i64_new(i64 4096)',
          [Index]),
      ( memberchk(cache(Index, BytesLen, Backend, ValueKind, SchemaRef, SubDb), CacheEntries)
      ->  plawk_cache_fn(Backend, load, ValueKind, SubDb, LoadFn),
          plawk_cache_call_ir(LoadFn, Index, BytesLen, ValueKind, SchemaRef, SubDb, LoadLine),
          Lines = [NewLine, LoadLine]
      ;   Lines = [NewLine]
      ),
      NextIndex is Index + 1
    },
    plawk_emit_lines(Lines),
    plawk_assoc_entry_setup_lines(Rest, NextIndex, CacheEntries).

%% plawk_program_cache_tables(+BeginClauses, -NamePaths)
%  Name-Path pairs for every table declared in a `BEGIN cache("path") {
%  declare NAME ... }` block.
plawk_program_cache_tables(BeginClauses, Triples) :-
    findall(ct(Name, Path, Backend),
        ( member(begin(Actions), BeginClauses),
          member(cache_table(Name, Path, Backend), Actions)
        ),
        Triples).

%% plawk_cache_fn(+Backend, +Op, +ValueKind, +SubDb, -FnName)
%  The runtime function for a cache Op (load|commit) under a backend, value
%  kind, and store mode (SubDb = `none` for a single-table store's default DB,
%  or `subdb(_)` for a multi-table store, phase 8.9). A str-valued (row) table
%  on the `file` backend uses the byte-valued helpers so the row bytes survive
%  across runs (an i64 store would persist a process-local atom id). i64 tables
%  use the plain file helpers. `lmdb` has both value kinds, and each in a
%  single-table (`_lmdb` / `_lmdb_str`, unnamed default DB) or multi-table
%  (`_lmdb_sub` / `_lmdb_str_sub`, a named sub-DB) form. The file backend is
%  never multi-table (a compile error, checked earlier), so file never sees
%  subdb(_).
plawk_cache_fn(lmdb, load, str, subdb(_), wam_cache_load_lmdb_str_sub) :- !.
plawk_cache_fn(lmdb, commit, str, subdb(_), wam_cache_commit_lmdb_str_sub) :- !.
plawk_cache_fn(lmdb, load, _Kind, subdb(_), wam_cache_load_lmdb_sub) :- !.
plawk_cache_fn(lmdb, commit, _Kind, subdb(_), wam_cache_commit_lmdb_sub) :- !.
plawk_cache_fn(file, load, str, _Sub, wam_cache_load_str) :- !.
plawk_cache_fn(file, commit, str, _Sub, wam_cache_commit_str) :- !.
plawk_cache_fn(lmdb, load, str, _Sub, wam_cache_load_lmdb_str) :- !.
plawk_cache_fn(lmdb, commit, str, _Sub, wam_cache_commit_lmdb_str) :- !.
plawk_cache_fn(lmdb, load, _Kind, _Sub, wam_cache_load_lmdb) :- !.
plawk_cache_fn(lmdb, commit, _Kind, _Sub, wam_cache_commit_lmdb) :- !.
plawk_cache_fn(_File, load, _Kind, _Sub, wam_cache_load).
plawk_cache_fn(_File, commit, _Kind, _Sub, wam_cache_commit).

%% plawk_cache_entries(+Tables, +StrArrays, +Schemas, +Triples, -CacheEntries,
%%     -PathGlobalsIR)
%  Resolve each cache-backed table name to its index in the plan's table
%  list, emit a private string global @.plawk_cache_path_<idx> for its path,
%  and record cache(Index, BytesLen, Backend, ValueKind, SchemaRef) so
%  setup/commit can reference the global, pick the backend function, and (for
%  a str/row table with a declared schema) pass the schema so the store is
%  self-describing and validated on open. SchemaRef is an `i8*` operand: a
%  getelementptr to the schema-string global, or `null` when no schema. A
%  declared name absent from the plan (never used) is skipped. PathGlobalsIR
%  carries the path/schema globals plus any lmdb backend declares in use.
plawk_cache_entries(Tables, StrArrays, Schemas, Triples, CacheEntries, PathGlobalsIR) :-
    plawk_multitable_paths(Triples, MultiPaths),
    findall(cache(Index, BytesLen, Backend, ValueKind, SchemaRef, SubDb)-GroupIR,
        ( member(ct(Name, Path, Backend), Triples),
          nth0(Index, Tables, Name),
          ( memberchk(Name, StrArrays) -> ValueKind = str ; ValueKind = i64 ),
          format(atom(GName), 'plawk_cache_path_~w', [Index]),
          llvm_emit_c_string_global(GName, Path, PathGlobal, _Len, BytesLen),
          ( ValueKind == str, memberchk(cache_schema(Name, Cols), Schemas)
          ->  plawk_schema_string(Cols, SchemaStr),
              format(atom(SGName), 'plawk_cache_schema_~w', [Index]),
              llvm_emit_c_string_global(SGName, SchemaStr, SchemaGlobal, _SLen, SBytes),
              format(atom(SchemaRef),
                  'getelementptr inbounds ([~w x i8], [~w x i8]* @.~w, i64 0, i64 0)',
                  [SBytes, SBytes, SGName]),
              SchemaGlobals = [SchemaGlobal]
          ;   SchemaRef = 'null',
              SchemaGlobals = []
          ),
          % A multi-table LMDB store routes each table to its own named sub-DB
          % (phase 8.9); the sub-DB name is the plawk table name. Single-table
          % stores (and the file backend, which is never multi-table) use the
          % unnamed default DB.
          ( Backend == lmdb, memberchk(Path, MultiPaths)
          ->  format(atom(SubGName), 'plawk_cache_subdb_~w', [Index]),
              plawk_local_table_name(Name, Local),
              atom_string(Local, NameStr),
              llvm_emit_c_string_global(SubGName, NameStr, SubGlobal, _SubLen, SubBytes),
              format(atom(SubRef),
                  'getelementptr inbounds ([~w x i8], [~w x i8]* @.~w, i64 0, i64 0)',
                  [SubBytes, SubBytes, SubGName]),
              SubDb = subdb(SubRef),
              SubGlobals = [SubGlobal]
          ;   SubDb = none,
              SubGlobals = []
          ),
          append([[PathGlobal], SchemaGlobals, SubGlobals], GroupParts),
          atomic_list_concat(GroupParts, '\n', GroupIR)
        ),
        Pairs),
    pairs_keys_values(Pairs, CacheEntries, Globals),
    plawk_lmdb_decls(CacheEntries, LmdbDecls),
    append(Globals, LmdbDecls, AllGlobals),
    atomic_list_concat(AllGlobals, '\n', PathGlobalsIR).

%% plawk_multitable_paths(+Triples, -MultiPaths)
%  Store paths whose tables route to named sub-DBs -- either a path carrying two
%  or more distinct table names, OR a path with a namespaced (`ns.table`) table
%  (phase 8.9 PR 4): `as ns` asks for sub-databases, so a namespaced store uses
%  named sub-DBs even with a single table. (A multi-table *file* store is
%  rejected earlier as a compile error, so in practice these are lmdb.)
plawk_multitable_paths(Triples, MultiPaths) :-
    findall(Path,
        ( member(ct(_, Path, _), Triples),
          ( findall(N, member(ct(N, Path, _), Triples), Ns0),
            sort(Ns0, Ns), Ns = [_, _ | _]
          ; member(ct(TN, Path, _), Triples), plawk_is_namespaced(TN)
          ) ),
        MultiPaths0),
    sort(MultiPaths0, MultiPaths).

%% plawk_is_namespaced(+TableName) / plawk_local_table_name(+Name, -Local)
%  A namespaced table name is the dotted atom `ns.local` (phase 8.9 PR 4); its
%  LOCAL part (after the last dot) is the sub-DB it routes to. A bare name has
%  no dot and is its own local name.
plawk_is_namespaced(Name) :-
    sub_atom(Name, _, _, _, '.'),
    !.

plawk_local_table_name(Name, Local) :-
    atomic_list_concat(Parts, '.', Name),
    last(Parts, Local).

%% plawk_lmdb_decls(+CacheEntries, -Decls)
%  `declare`s for the external LMDB helpers actually referenced: the plain
%  (unnamed-DB) i64/str helpers when any lmdb table is present, plus the
%  named-sub-DB (`_sub`) helpers when a multi-table lmdb store is in use.
plawk_lmdb_decls(CacheEntries, Decls) :-
    ( memberchk(cache(_, _, lmdb, _, _, _), CacheEntries)
    ->  Base = ['declare void @wam_cache_load_lmdb(%WamAssocI64Table*, i8*)',
                'declare void @wam_cache_commit_lmdb(%WamAssocI64Table*, i8*)'],
        ( memberchk(cache(_, _, lmdb, str, _, _), CacheEntries)
        ->  Str = ['declare void @wam_cache_load_lmdb_str(%WamAssocI64Table*, i8*, i8*)',
                   'declare void @wam_cache_commit_lmdb_str(%WamAssocI64Table*, i8*, i8*)']
        ;   Str = []
        ),
        ( memberchk(cache(_, _, lmdb, i64, _, subdb(_)), CacheEntries)
        ->  SubI = ['declare void @wam_cache_load_lmdb_sub(%WamAssocI64Table*, i8*, i8*)',
                    'declare void @wam_cache_commit_lmdb_sub(%WamAssocI64Table*, i8*, i8*)']
        ;   SubI = []
        ),
        ( memberchk(cache(_, _, lmdb, str, _, subdb(_)), CacheEntries)
        ->  SubS = ['declare void @wam_cache_load_lmdb_str_sub(%WamAssocI64Table*, i8*, i8*, i8*)',
                    'declare void @wam_cache_commit_lmdb_str_sub(%WamAssocI64Table*, i8*, i8*, i8*)']
        ;   SubS = []
        ),
        append([Base, Str, SubI, SubS], Decls)
    ;   Decls = []
    ).

%% plawk_schema_string(+Columns, -Str)
%  Canonical schema string for a row table: `name:type,name:type,...`. Written
%  into the store header and compared on open (must match byte-for-byte).
plawk_schema_string(Columns, Str) :-
    findall(CS, ( member(col(Name, Type), Columns),
                  format(atom(CS), '~w:~w', [Name, Type]) ), Parts),
    atomic_list_concat(Parts, ',', Str).

%% plawk_cache_call_ir(+Fn, +Index, +BytesLen, +ValueKind, +SchemaRef, +SubDb, -Line)
%  A load/commit call. Four shapes by (value kind x store mode):
%   - str, named sub-DB: (table, path, subname, schema)  -- lmdb multi-table
%   - i64, named sub-DB: (table, path, subname)          -- lmdb multi-table
%   - str, default DB:   (table, path, schema)           -- file / lmdb single
%   - i64, default DB:   (table, path)                    -- file / lmdb single
%  The byte-valued (str) helpers carry the i8* schema (a getelementptr or
%  null); the named-sub-DB helpers carry the i8* sub-DB name before it.
plawk_cache_call_ir(Fn, Index, BytesLen, str, SchemaRef, subdb(SubRef), Line) :-
    !,
    format(atom(Line),
        '  call void @~w(%WamAssocI64Table* %plawk_assoc_table_~w, i8* getelementptr inbounds ([~w x i8], [~w x i8]* @.plawk_cache_path_~w, i64 0, i64 0), i8* ~w, i8* ~w)',
        [Fn, Index, BytesLen, BytesLen, Index, SubRef, SchemaRef]).
plawk_cache_call_ir(Fn, Index, BytesLen, _ValueKind, _SchemaRef, subdb(SubRef), Line) :-
    !,
    format(atom(Line),
        '  call void @~w(%WamAssocI64Table* %plawk_assoc_table_~w, i8* getelementptr inbounds ([~w x i8], [~w x i8]* @.plawk_cache_path_~w, i64 0, i64 0), i8* ~w)',
        [Fn, Index, BytesLen, BytesLen, Index, SubRef]).
plawk_cache_call_ir(Fn, Index, BytesLen, str, SchemaRef, none, Line) :-
    ( Fn == wam_cache_load_str ; Fn == wam_cache_commit_str
    ; Fn == wam_cache_load_lmdb_str ; Fn == wam_cache_commit_lmdb_str ),
    !,
    format(atom(Line),
        '  call void @~w(%WamAssocI64Table* %plawk_assoc_table_~w, i8* getelementptr inbounds ([~w x i8], [~w x i8]* @.plawk_cache_path_~w, i64 0, i64 0), i8* ~w)',
        [Fn, Index, BytesLen, BytesLen, Index, SchemaRef]).
plawk_cache_call_ir(Fn, Index, BytesLen, _ValueKind, _SchemaRef, none, Line) :-
    format(atom(Line),
        '  call void @~w(%WamAssocI64Table* %plawk_assoc_table_~w, i8* getelementptr inbounds ([~w x i8], [~w x i8]* @.plawk_cache_path_~w, i64 0, i64 0))',
        [Fn, Index, BytesLen, BytesLen, Index]).

%% plawk_cache_commit_lines(+CacheEntries, -IR)
%  A commit call per cache-backed table, writing the final table back to its
%  store. Emitted at the top of the END phase (after the record loop, table
%  still live), so the committed state is the completed pass.
plawk_cache_commit_lines(CacheEntries, IR) :-
    findall(Line,
        ( member(cache(Index, BytesLen, Backend, ValueKind, SchemaRef, SubDb), CacheEntries),
          plawk_cache_fn(Backend, commit, ValueKind, SubDb, CommitFn),
          plawk_cache_call_ir(CommitFn, Index, BytesLen, ValueKind, SchemaRef, SubDb, Line)
        ),
        Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_assoc_entry_setup_lines([], _) -->
    [].
plawk_assoc_entry_setup_lines([_ArrayName | Rest], Index) -->
    { format(atom(Line),
          '  %plawk_assoc_table_~w = call %WamAssocI64Table* @wam_assoc_i64_new(i64 4096)',
          [Index]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_assoc_entry_setup_lines(Rest, NextIndex).

plawk_assoc_rule_chain_ir(assoc_plan(_Tables, Rules), FieldSeparator, GlobalIR, ChainIR) :-
    length(Rules, RuleCount),
    RuleCount > 0,
    phrase(plawk_assoc_rule_chain_lines(Rules, FieldSeparator, 0), Pairs),
    pairs_keys_values(Pairs, GlobalParts, ChainParts),
    atomic_list_concat(GlobalParts, '\n', GlobalIR),
    atomic_list_concat(ChainParts, '\n', ChainIR).

plawk_assoc_rule_chain_lines([], _FieldSeparator, _) -->
    [].
plawk_assoc_rule_chain_lines([assoc_rule(Index, Pattern, Actions, Control) | Rest], FieldSeparator, Index) -->
    { NextIndex is Index + 1,
      ( Rest == []
      -> NextLabel = 'continue_loop'
      ;  format(atom(NextLabel), 'assoc_rule_~w_match', [NextIndex])
      ),
      format(atom(RuleLabel), 'assoc_rule_~w_match', [Index]),
      format(atom(ApplyLabel), 'assoc_rule_~w_apply', [Index]),
      plawk_rule_target(Control, NextLabel, RuleTargetLabel),
      % tagged-union rules carry their arm's field types with them
      plawk_rule_descriptor(Pattern, FieldSeparator, RuleDescriptor),
      plawk_assoc_rule_apply_ir(Index, Actions, RuleTargetLabel, RuleDescriptor,
          ApplyGlobalIR, ApplyIR),
      ( Index =:= 0
      -> EntryIR = '  br label %assoc_rule_0_match\n\n'
      ;  EntryIR = ''
      ),
      plawk_assoc_rule_guard_ir(Pattern, Index, RuleLabel, ApplyLabel,
          NextLabel, EntryIR, RuleDescriptor, GuardGlobalIR, BranchIR),
      format(atom(RuleIR),
'~w

~w:
~w',
          [BranchIR, ApplyLabel, ApplyIR]),
      format(atom(RuleGlobalIR), '~w~w', [GuardGlobalIR, ApplyGlobalIR]),
      Pair = RuleGlobalIR-RuleIR
    },
    [Pair],
    plawk_assoc_rule_chain_lines(Rest, FieldSeparator, NextIndex).

plawk_assoc_rule_guard_ir(always, _Index, RuleLabel, ApplyLabel, _NextLabel,
    EntryIR, _FieldSeparator, '', IR) :-
    !,
    format(atom(IR),
'~w~w:
  br label %~w',
        [EntryIR, RuleLabel, ApplyLabel]).
plawk_assoc_rule_guard_ir(Pattern, Index, RuleLabel, ApplyLabel, NextLabel,
    EntryIR, FieldSeparator, GuardGlobalIR, IR) :-
    format(atom(MatchVar), 'assoc_rule_~w_is_match', [Index]),
    format(atom(GlobalBase), 'plawk_assoc_rule_~w', [Index]),
    format(atom(MatchValue), '%~w', [MatchVar]),
    plawk_pattern_guard_ir(Pattern, FieldSeparator, GlobalBase, MatchValue,
        GuardGlobalIR-GuardCallIR),
    format(atom(IR),
'~w~w:
~w
  br i1 %~w, label %~w, label %~w',
        [EntryIR, RuleLabel, GuardCallIR, MatchVar, ApplyLabel, NextLabel]).

% The apply stream may carry global(G) marker items (e.g. blob-key arg
% marshaling emits per-arg constants); they partition out into the
% rule's global-constant channel alongside the guard globals.
plawk_assoc_rule_apply_ir(_RuleIndex, [], NextLabel, _FieldSeparator, '', IR) :-
    !,
    format(atom(IR), '  br label %~w', [NextLabel]).
plawk_assoc_rule_apply_ir(RuleIndex, Actions, NextLabel, FieldSeparator,
        GlobalIR, IR) :-
    phrase(plawk_assoc_rule_action_lines(RuleIndex, Actions, NextLabel, FieldSeparator), Lines0),
    plawk_partition_global_lines(Lines0, Globals, Lines),
    atomic_list_concat(Lines, '\n', IR),
    (   Globals == []
    ->  GlobalIR = ''
    ;   sort(Globals, GlobalsDedup),   % identical scalar-global decls collapse
        atomic_list_concat(GlobalsDedup, '\n', Gs),
        atom_concat('\n', Gs, GlobalIR)
    ).

plawk_partition_global_lines([], [], []).
plawk_partition_global_lines([global(G) | Rest], [G | Gs], Lines) :-
    !,
    plawk_partition_global_lines(Rest, Gs, Lines).
plawk_partition_global_lines([L | Rest], Gs, [L | Lines]) :-
    plawk_partition_global_lines(Rest, Gs, Lines).

%% plawk_split_call_lines(+Sep, +Base, +TableIndex, -ExtraLines, -SplitCall)
%
%  The `split()` call for a separator. A single-char `Sep` is a literal byte
%  (`@wam_str_split_into`), with no extra lines. A multi-char `Sep` is a POSIX
%  ERE: emit a per-site pattern constant + cache global (as `global(...)` terms,
%  lifted to module scope) and a body getelementptr for the pattern pointer, then
%  call `@wam_str_split_into_re`. Base is the site-unique action label stem, so
%  the derived global names never collide across split sites.
plawk_split_call_lines(Sep, Base, TableIndex, [], SplitCall) :-
    string_codes(Sep, [SepCode]),
    !,
    format(atom(SplitCall),
        '  %~w_n = call i64 @wam_str_split_into(%WamAssocI64Table* %plawk_assoc_table_~w, i8* %~w_src_ptr, i64 %~w_src_len, i8 ~w)',
        [Base, TableIndex, Base, Base, SepCode]).
plawk_split_call_lines(Sep, Base, TableIndex,
        [global(PatGlobal), global(CacheGlobal), PatPtrLine], SplitCall) :-
    format(atom(PatName), '~w_split_pat', [Base]),
    format(atom(CacheName), '~w_split_cache', [Base]),
    llvm_emit_c_string_global(PatName, Sep, PatGlobal, _StringLen, BytesLen),
    format(atom(CacheGlobal), '@~w = internal global i8* null', [CacheName]),
    format(atom(PatPtrLine),
        '  %~w_patptr = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0',
        [Base, BytesLen, BytesLen, PatName]),
    format(atom(SplitCall),
        '  %~w_n = call i64 @wam_str_split_into_re(%WamAssocI64Table* %plawk_assoc_table_~w, i8* %~w_src_ptr, i64 %~w_src_len, i8* %~w_patptr, i8** @~w)',
        [Base, TableIndex, Base, Base, Base, CacheName]).

plawk_assoc_rule_action_lines(RuleIndex, Actions, NextLabel, FieldSeparator) -->
    { Actions = [_ | _],
      format(atom(FirstBranch), '  br label %assoc_rule_~w_action_0', [RuleIndex])
    },
    [FirstBranch, ''],
    plawk_assoc_rule_action_blocks(RuleIndex, Actions, NextLabel, FieldSeparator).

plawk_assoc_rule_action_blocks(_RuleIndex, [], _NextLabel, _FieldSeparator) -->
    [].
% Grammar-populate action: evaluate the call args, then hand the whole
% [K-V,...] result to the assoc shim, which walks it into this array's table.
plawk_assoc_rule_action_blocks(RuleIndex,
        [assoc_dyn_action(Index, _ArrayName, TableIndex, Call) | Rest],
        NextLabel, FieldSeparator) -->
    { ( Rest == []
      -> ActionNextLabel = NextLabel
      ;  NextIndex is Index + 1,
         format(atom(ActionNextLabel), 'assoc_rule_~w_action_~w',
             [RuleIndex, NextIndex])
      ),
      format(atom(Label), 'assoc_rule_~w_action_~w:', [RuleIndex, Index]),
      format(atom(Base), 'assoc_rule_~w_action_~w_dyn', [RuleIndex, Index]),
      plawk_dynassoc_call_parts(Call, Args, ShimName),
      % arg-marshal globals (text-mode field args emit an _empty
      % fallback constant each) ride the line stream as global(G)
      % markers, partitioned into the rule's global channel by the
      % apply emitter -- discarding them left the emitted IR
      % referencing undefined constants in text mode.
      plawk_foreign_args_ir(Args, FieldSeparator, Base, ArgValueIRs,
          Globals, Setup),
      findall(global(G), member(G, Globals), GlobalMarkers),
      plawk_foreign_call_args_ir(ArgValueIRs, CallArgsIR),
      format(atom(CallLine),
          '  %~w_ok = call i1 @~w(~w, %WamAssocI64Table* %plawk_assoc_table_~w)',
          [Base, ShimName, CallArgsIR, TableIndex]),
      format(atom(Next), '  br label %~w', [ActionNextLabel]),
      append([GlobalMarkers, [Label], Setup, [CallLine, Next, '']], Lines)
    },
    plawk_emit_lines(Lines),
    plawk_assoc_rule_action_blocks(RuleIndex, Rest, NextLabel, FieldSeparator).
% Scalar accumulator: `acc += 1` / `acc += $N`, folded into a module global
% (@plawk_scalar_<acc>, zero-init), so the value persists across multi-pass
% passes. Emits the global declaration on the global channel (deduped by the
% rule/driver), then load/add/store per record.
plawk_assoc_rule_action_blocks(RuleIndex,
        [assoc_scalar_add_action(Index, Name, Src) | Rest], NextLabel, FieldSeparator) -->
    { ( Rest == []
      -> ActionNextLabel = NextLabel
      ;  NextIndex is Index + 1,
         format(atom(ActionNextLabel), 'assoc_rule_~w_action_~w',
             [RuleIndex, NextIndex])
      ),
      format(atom(Label), 'assoc_rule_~w_action_~w:', [RuleIndex, Index]),
      format(atom(Base), 'assoc_rule_~w_action_~w_sadd', [RuleIndex, Index]),
      format(atom(GlobalDecl),
          '@plawk_scalar_~w = internal global i64 0', [Name]),
      plawk_assoc_scalar_src_lines(Src, Base, FieldSeparator, SrcVar, SrcLines),
      format(atom(LoadL), '  %~w_cur = load i64, i64* @plawk_scalar_~w', [Base, Name]),
      format(atom(AddL), '  %~w_new = add i64 %~w_cur, ~w', [Base, Base, SrcVar]),
      format(atom(StoreL), '  store i64 %~w_new, i64* @plawk_scalar_~w', [Base, Name]),
      format(atom(Next), '  br label %~w', [ActionNextLabel]),
      append([[global(GlobalDecl), Label], SrcLines,
              [LoadL, AddL, StoreL, Next, '']], Lines)
    },
    plawk_emit_lines(Lines),
    plawk_assoc_rule_action_blocks(RuleIndex, Rest, NextLabel, FieldSeparator).
% Per-record print: one output line per record, fields being a text field
% ($N, printed via its slice) or a table lookup (arr[$N], the i64 value at
% the key interned from field N). Space-separated, newline-terminated.
plawk_assoc_rule_action_blocks(RuleIndex,
        [assoc_print_action(Index, Fields) | Rest], NextLabel, FieldSeparator) -->
    { ( Rest == []
      -> ActionNextLabel = NextLabel
      ;  NextIndex is Index + 1,
         format(atom(ActionNextLabel), 'assoc_rule_~w_action_~w',
             [RuleIndex, NextIndex])
      ),
      format(atom(Label), 'assoc_rule_~w_action_~w:', [RuleIndex, Index]),
      format(atom(Base), 'assoc_rule_~w_action_~w_pr', [RuleIndex, Index]),
      plawk_assoc_print_field_lines(Fields, Base, FieldSeparator, 0, FieldLines),
      format(atom(NLLine), '  %~w_nl = call i32 @putchar(i32 10)', [Base]),
      format(atom(NextLine), '  br label %~w', [ActionNextLabel]),
      append([[Label], FieldLines, [NLLine, NextLine, '']], Lines)
    },
    plawk_emit_lines(Lines),
    plawk_assoc_rule_action_blocks(RuleIndex, Rest, NextLabel, FieldSeparator).
% Rule-body for-in: a per-record loop over an assoc table's occupied
% slots, printing the planned fields (loop key resolved to text, table
% values numeric or str-resolved per the plan, literals) one line per
% key -- the END for-in loop shape with rule/action-scoped labels so it
% nests in the action chain. Literal-field constants ride the global
% channel like blob-key marshals.
plawk_assoc_rule_action_blocks(RuleIndex,
        [assoc_forin_action(Index, TableIndex, FieldPlans) | Rest],
        NextLabel, FieldSeparator) -->
    { ( Rest == []
      -> ActionNextLabel = NextLabel
      ;  NextIndex is Index + 1,
         format(atom(ActionNextLabel), 'assoc_rule_~w_action_~w',
             [RuleIndex, NextIndex])
      ),
      format(atom(Label), 'assoc_rule_~w_action_~w:', [RuleIndex, Index]),
      format(atom(B), 'arfi_~w_~w', [RuleIndex, Index]),
      format(atom(HeadBr), '  br label %~w_head', [B]),
      format(atom(HeadLbl), '~w_head:', [B]),
      format(atom(Phi),
          '  %~w_idx = phi i64 [0, %assoc_rule_~w_action_~w], [%~w_next, %~w_done]',
          [B, RuleIndex, Index, B, B]),
      format(atom(Slot),
          '  %~w_slot = call i64 @wam_assoc_i64_iter_next(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %~w_idx)',
          [B, TableIndex, B]),
      format(atom(DoneC), '  %~w_done_c = icmp slt i64 %~w_slot, 0', [B, B]),
      format(atom(BrBody),
          '  br i1 %~w_done_c, label %~w_after, label %~w_body', [B, B, B]),
      format(atom(BodyLbl), '~w_body:', [B]),
      format(atom(Key),
          '  %~w_key = call i64 @wam_assoc_i64_key_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %~w_slot)',
          [B, TableIndex, B]),
      phrase(plawk_forin_rule_field_lines(FieldPlans, B, 0), FieldLines),
      format(atom(NL), '  %~w_nl = call i32 @putchar(i32 10)', [B]),
      format(atom(BrDone), '  br label %~w_done', [B]),
      format(atom(DoneLbl), '~w_done:', [B]),
      format(atom(Next), '  %~w_next = add i64 %~w_slot, 1', [B, B]),
      format(atom(BrHead), '  br label %~w_head', [B]),
      format(atom(AfterLbl), '~w_after:', [B]),
      format(atom(BrNext), '  br label %~w', [ActionNextLabel]),
      append([[Label, HeadBr, '', HeadLbl, Phi, Slot, DoneC, BrBody, '',
               BodyLbl, Key],
              FieldLines,
              [NL, BrDone, '', DoneLbl, Next, BrHead, '', AfterLbl, BrNext,
               '']],
          Lines)
    },
    plawk_emit_lines(Lines),
    plawk_assoc_rule_action_blocks(RuleIndex, Rest, NextLabel, FieldSeparator).
% Guarded for-in (stage 1 filter): same slot-iteration loop, but a guard
% on the key or the iterated value gates the per-key print. Passing
% entries print; the rest fall straight through to the loop-continue.
plawk_assoc_rule_action_blocks(RuleIndex,
        [assoc_forin_guarded_action(Index, TableIndex, GuardPlan, FieldPlans)
         | Rest],
        NextLabel, FieldSeparator) -->
    { ( Rest == []
      -> ActionNextLabel = NextLabel
      ;  NextIndex is Index + 1,
         format(atom(ActionNextLabel), 'assoc_rule_~w_action_~w',
             [RuleIndex, NextIndex])
      ),
      format(atom(Label), 'assoc_rule_~w_action_~w:', [RuleIndex, Index]),
      format(atom(B), 'arfg_~w_~w', [RuleIndex, Index]),
      format(atom(HeadBr), '  br label %~w_head', [B]),
      format(atom(HeadLbl), '~w_head:', [B]),
      format(atom(Phi),
          '  %~w_idx = phi i64 [0, %assoc_rule_~w_action_~w], [%~w_next, %~w_done]',
          [B, RuleIndex, Index, B, B]),
      format(atom(Slot),
          '  %~w_slot = call i64 @wam_assoc_i64_iter_next(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %~w_idx)',
          [B, TableIndex, B]),
      format(atom(DoneC), '  %~w_done_c = icmp slt i64 %~w_slot, 0', [B, B]),
      format(atom(BrBody),
          '  br i1 %~w_done_c, label %~w_after, label %~w_body', [B, B, B]),
      format(atom(BodyLbl), '~w_body:', [B]),
      format(atom(Key),
          '  %~w_key = call i64 @wam_assoc_i64_key_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %~w_slot)',
          [B, TableIndex, B]),
      plawk_forin_guard_lines(GuardPlan, B, TableIndex, GuardLines, CondVar),
      format(atom(BrGuard),
          '  br i1 ~w, label %~w_print, label %~w_skip', [CondVar, B, B]),
      format(atom(PrintLbl), '~w_print:', [B]),
      phrase(plawk_forin_rule_field_lines(FieldPlans, B, 0), FieldLines),
      format(atom(NL), '  %~w_nl = call i32 @putchar(i32 10)', [B]),
      format(atom(BrDoneP), '  br label %~w_done', [B]),
      format(atom(SkipLbl), '~w_skip:', [B]),
      format(atom(BrDoneS), '  br label %~w_done', [B]),
      format(atom(DoneLbl), '~w_done:', [B]),
      format(atom(Next), '  %~w_next = add i64 %~w_slot, 1', [B, B]),
      format(atom(BrHead), '  br label %~w_head', [B]),
      format(atom(AfterLbl), '~w_after:', [B]),
      format(atom(BrNext), '  br label %~w', [ActionNextLabel]),
      append([[Label, HeadBr, '', HeadLbl, Phi, Slot, DoneC, BrBody, '',
               BodyLbl, Key],
              GuardLines,
              [BrGuard, '', PrintLbl],
              FieldLines,
              [NL, BrDoneP, '', SkipLbl, BrDoneS, '', DoneLbl, Next, BrHead,
               '', AfterLbl, BrNext, '']],
          Lines)
    },
    plawk_emit_lines(Lines),
    plawk_assoc_rule_action_blocks(RuleIndex, Rest, NextLabel, FieldSeparator).

% Guard operand load + comparison. guard_value reads the iterated table's
% value at the current slot; guard_key uses the raw key. Both icmp against
% the integer literal.
plawk_forin_guard_lines(guard_value(Op, V), B, TableIndex, Lines, CondVar) :-
    plawk_forin_cmp_pred(Op, Pred),
    format(atom(ValLine),
        '  %~w_gval = call i64 @wam_assoc_i64_value_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %~w_slot)',
        [B, TableIndex, B]),
    format(atom(CmpLine),
        '  %~w_gcmp = icmp ~w i64 %~w_gval, ~w', [B, Pred, B, V]),
    format(atom(CondVar), '%~w_gcmp', [B]),
    Lines = [ValLine, CmpLine].
plawk_forin_guard_lines(guard_key(Op, V), B, _TableIndex, Lines, CondVar) :-
    plawk_forin_cmp_pred(Op, Pred),
    format(atom(CmpLine),
        '  %~w_gcmp = icmp ~w i64 %~w_key, ~w', [B, Pred, B, V]),
    format(atom(CondVar), '%~w_gcmp', [B]),
    Lines = [CmpLine].

plawk_forin_cmp_pred(eq, eq).
plawk_forin_cmp_pred(ne, ne).
plawk_forin_cmp_pred(lt, slt).
plawk_forin_cmp_pred(le, sle).
plawk_forin_cmp_pred(gt, sgt).
plawk_forin_cmp_pred(ge, sge).

% String comparison ops that reduce to a canonical atom-id equality check;
% the rest (ordering) go through strcmp.
plawk_forin_streq_op(eq).
plawk_forin_streq_op(ne).

% Float (fcmp) predicates for a counter element compared to a float literal.
plawk_forin_fcmp_pred(eq, oeq).
plawk_forin_fcmp_pred(ne, one).
plawk_forin_fcmp_pred(lt, olt).
plawk_forin_fcmp_pred(le, ole).
plawk_forin_fcmp_pred(gt, ogt).
plawk_forin_fcmp_pred(ge, oge).

plawk_forin_rule_field_lines([], _B, _N) -->
    [].
plawk_forin_rule_field_lines([Plan | Rest], B, N) -->
    ( { N > 0 }
    ->  { format(atom(Sep), '  %~w_sep_~w = call i32 @putchar(i32 32)',
              [B, N]) },
        [Sep]
    ;   []
    ),
    plawk_forin_rule_field_value(Plan, B, N),
    { N1 is N + 1 },
    plawk_forin_rule_field_lines(Rest, B, N1).

% loop key: a text-mode table key is a registry id -- resolve to text.
plawk_forin_rule_field_value(key, B, N) -->
    { format(atom(KeyS),
          '  %~w_key_s_~w = call i8* @wam_atom_to_string(i64 %~w_key)',
          [B, N, B]),
      format(atom(FmtVar), '~w_key_fmt_~w', [B, N]),
      format(atom(PrintVar), '~w_key_p_~w', [B, N]),
      format(atom(PtrIR), '%~w_key_s_~w', [B, N]),
      llvm_emit_printf_string(plawk_surface_print_string, FmtVar, PrintVar,
          PtrIR, [FmtPtr, PrintCall])
    },
    [KeyS, FmtPtr, PrintCall].
% positional-array loop key: the key is an integer position -- print it
% numerically, not as an atom-registry id.
plawk_forin_rule_field_value(key_int, B, N) -->
    { format(atom(FmtVar), '~w_key_fmt_~w', [B, N]),
      format(atom(PrintVar), '~w_key_p_~w', [B, N]),
      format(atom(KeyIR), '%~w_key', [B]),
      llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar,
          KeyIR, [FmtPtr, PrintCall])
    },
    [FmtPtr, PrintCall].
plawk_forin_rule_field_value(value_self(TableIndex, Kind), B, N) -->
    { format(atom(Value),
          '  %~w_v_~w = call i64 @wam_assoc_i64_value_at(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %~w_slot)',
          [B, N, TableIndex, B])
    },
    [Value],
    plawk_forin_rule_value_print(Kind, B, N).
plawk_forin_rule_field_value(value_lookup(LookupIndex, Kind), B, N) -->
    { format(atom(Value),
          '  %~w_v_~w = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %~w_key)',
          [B, N, LookupIndex, B])
    },
    [Value],
    plawk_forin_rule_value_print(Kind, B, N).
plawk_forin_rule_field_value(lit(Value), B, N) -->
    { format(atom(GName), 'plawk_~w_lit_~w', [B, N]),
      llvm_emit_c_string_global(GName, Value, Global, _StrLen, BytesLen),
      format(atom(Ptr),
          '  %~w_lit_~w = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0',
          [B, N, BytesLen, BytesLen, GName]),
      format(atom(FmtVar), '~w_lit_fmt_~w', [B, N]),
      format(atom(PrintVar), '~w_lit_p_~w', [B, N]),
      format(atom(PtrIR), '%~w_lit_~w', [B, N]),
      llvm_emit_printf_string(plawk_surface_print_string, FmtVar, PrintVar,
          PtrIR, [FmtPtr, PrintCall])
    },
    [global(Global), Ptr, FmtPtr, PrintCall].

plawk_forin_rule_value_print(i64, B, N) -->
    { format(atom(FmtVar), '~w_v_fmt_~w', [B, N]),
      format(atom(PrintVar), '~w_v_p_~w', [B, N]),
      format(atom(ValueIR), '%~w_v_~w', [B, N]),
      llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar,
          ValueIR, [FmtPtr, PrintCall])
    },
    [FmtPtr, PrintCall].
% str-valued table: the stored i64 is an atom-registry id.
plawk_forin_rule_value_print(str, B, N) -->
    { format(atom(ValueS),
          '  %~w_vs_~w = call i8* @wam_atom_to_string(i64 %~w_v_~w)',
          [B, N, B, N]),
      format(atom(FmtVar), '~w_vs_fmt_~w', [B, N]),
      format(atom(PrintVar), '~w_vs_p_~w', [B, N]),
      format(atom(PtrIR), '%~w_vs_~w', [B, N]),
      llvm_emit_printf_string(plawk_surface_print_string, FmtVar, PrintVar,
          PtrIR, [FmtPtr, PrintCall])
    },
    [ValueS, FmtPtr, PrintCall].

% blob keys: evaluate the runtime grammar's byte output and intern it,
% exactly as the text-field path interns its slice; a failed call (null
% pointer) skips the increment, like a missing field. Works in text and
% binfmt modes alike (the blob's args marshal per the separator). The
% marshal's global constants ride the line stream as global(G) markers,
% partitioned into the rule's global channel by the apply emitter.
plawk_assoc_rule_action_blocks(RuleIndex, [assoc_action(Index, _ArrayName, TableIndex, blob_key(Blob)) | Rest], NextLabel, FieldSeparator) -->
    { !,
      ( Rest == []
      -> ActionNextLabel = NextLabel
      ;  NextIndex is Index + 1,
         format(atom(ActionNextLabel), 'assoc_rule_~w_action_~w',
             [RuleIndex, NextIndex])
      ),
      format(atom(Label), 'assoc_rule_~w_action_~w:', [RuleIndex, Index]),
      format(atom(KeyBase), 'assoc_rule_~w_action_~w_bkey', [RuleIndex, Index]),
      plawk_blob_expr_ir(Blob, FieldSeparator, KeyBase, _LenIR, _PtrIR,
          GParts, SetupParts),
      findall(global(G), member(G, GParts), GlobalMarkers),
      format(atom(HaveLabelName), 'assoc_rule_~w_action_~w_have_bkey',
          [RuleIndex, Index]),
      format(atom(HaveLabel), '~w:', [HaveLabelName]),
      format(atom(Missing),
          '  %~w_missing = icmp eq i8* %~w_ptr, null', [KeyBase, KeyBase]),
      format(atom(Branch),
          '  br i1 %~w_missing, label %~w, label %~w',
          [KeyBase, ActionNextLabel, HaveLabelName]),
      format(atom(KeyId),
          '  %~w_id = call i64 @wam_intern_atom(i8* %~w_ptr, i64 %~w_len64)',
          [KeyBase, KeyBase, KeyBase]),
      format(atom(Inc),
          '  %~w_count = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %~w_id, i64 1)',
          [KeyBase, TableIndex, KeyBase]),
      format(atom(Next), '  br label %~w', [ActionNextLabel]),
      append([GlobalMarkers, [Label], SetupParts,
              [Missing, Branch, '', HaveLabel, KeyId, Inc, Next, '']], Lines)
    },
    plawk_emit_lines(Lines),
    plawk_assoc_rule_action_blocks(RuleIndex, Rest, NextLabel, FieldSeparator).
plawk_assoc_rule_action_blocks(RuleIndex, [assoc_action(Index, _ArrayName, TableIndex, KeyIndex) | Rest], NextLabel, binfmt(Types)) -->
    { !,
      ( Rest == []
      -> ActionNextLabel = NextLabel
      ;  NextIndex is Index + 1,
         format(atom(ActionNextLabel), 'assoc_rule_~w_action_~w',
             [RuleIndex, NextIndex])
      ),
      format(atom(Label), 'assoc_rule_~w_action_~w:', [RuleIndex, Index]),
      format(atom(KeyBase), 'assoc_rule_~w_action_~w_key', [RuleIndex, Index]),
      plawk_binfmt_field_load_lines(binfmt(Types), KeyIndex, KeyBase, KeyValueIR,
          LoadLines),
      format(atom(Inc),
          '  %assoc_rule_~w_action_~w_count = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %plawk_assoc_table_~w, i64 ~w, i64 1)',
          [RuleIndex, Index, TableIndex, KeyValueIR]),
      format(atom(Next), '  br label %~w', [ActionNextLabel]),
      append([[Label], LoadLines, [Inc, Next, '']], Lines)
    },
    plawk_emit_lines(Lines),
    plawk_assoc_rule_action_blocks(RuleIndex, Rest, NextLabel, binfmt(Types)).
plawk_assoc_rule_action_blocks(RuleIndex, [assoc_action(Index, _ArrayName, TableIndex, KeyIndex) | Rest], NextLabel, FieldSeparator) -->
    { ( Rest == []
      -> ActionNextLabel = NextLabel
      ;  NextIndex is Index + 1,
         format(atom(ActionNextLabel), 'assoc_rule_~w_action_~w',
             [RuleIndex, NextIndex])
      ),
      format(atom(Label), 'assoc_rule_~w_action_~w:', [RuleIndex, Index]),
      format(atom(HaveLabel), 'assoc_rule_~w_action_~w_have_key:',
          [RuleIndex, Index]),
      format(atom(HaveLabelName), 'assoc_rule_~w_action_~w_have_key',
          [RuleIndex, Index]),
      format(atom(Slice),
          '  %assoc_rule_~w_action_~w_key_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)',
          [RuleIndex, Index, KeyIndex, FieldSeparator]),
      format(atom(Ptr),
          '  %assoc_rule_~w_action_~w_key_ptr = extractvalue %WamSlice %assoc_rule_~w_action_~w_key_slice, 0',
          [RuleIndex, Index, RuleIndex, Index]),
      format(atom(Len),
          '  %assoc_rule_~w_action_~w_key_len = extractvalue %WamSlice %assoc_rule_~w_action_~w_key_slice, 1',
          [RuleIndex, Index, RuleIndex, Index]),
      format(atom(Missing),
          '  %assoc_rule_~w_action_~w_key_missing = icmp eq i8* %assoc_rule_~w_action_~w_key_ptr, null',
          [RuleIndex, Index, RuleIndex, Index]),
      format(atom(Branch),
          '  br i1 %assoc_rule_~w_action_~w_key_missing, label %~w, label %~w',
          [RuleIndex, Index, ActionNextLabel, HaveLabelName]),
      format(atom(KeyId),
          '  %assoc_rule_~w_action_~w_key_id = call i64 @wam_intern_atom(i8* %assoc_rule_~w_action_~w_key_ptr, i64 %assoc_rule_~w_action_~w_key_len)',
          [RuleIndex, Index, RuleIndex, Index, RuleIndex, Index]),
      format(atom(Inc),
          '  %assoc_rule_~w_action_~w_count = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %assoc_rule_~w_action_~w_key_id, i64 1)',
          [RuleIndex, Index, TableIndex, RuleIndex, Index]),
      format(atom(Next), '  br label %~w', [ActionNextLabel])
    },
    [Label, Slice, Ptr, Len, Missing, Branch, '', HaveLabel, KeyId, Inc, Next, ''],
    plawk_assoc_rule_action_blocks(RuleIndex, Rest, NextLabel, FieldSeparator).
% `delete arr[$k]`: same key-intern + missing-key skip as the counted inc, but
% call the void backward-shift delete instead of the inc. An absent key (null
% slice) skips to the next action; the runtime delete is itself a no-op if the
% interned key is not in the table.
plawk_assoc_rule_action_blocks(RuleIndex, [assoc_delete_action(Index, _ArrayName, TableIndex, KeyIndex) | Rest], NextLabel, FieldSeparator) -->
    { ( Rest == []
      -> ActionNextLabel = NextLabel
      ;  NextIndex is Index + 1,
         format(atom(ActionNextLabel), 'assoc_rule_~w_action_~w',
             [RuleIndex, NextIndex])
      ),
      format(atom(Label), 'assoc_rule_~w_action_~w:', [RuleIndex, Index]),
      format(atom(HaveLabel), 'assoc_rule_~w_action_~w_have_key:',
          [RuleIndex, Index]),
      format(atom(HaveLabelName), 'assoc_rule_~w_action_~w_have_key',
          [RuleIndex, Index]),
      format(atom(Slice),
          '  %assoc_rule_~w_action_~w_key_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)',
          [RuleIndex, Index, KeyIndex, FieldSeparator]),
      format(atom(Ptr),
          '  %assoc_rule_~w_action_~w_key_ptr = extractvalue %WamSlice %assoc_rule_~w_action_~w_key_slice, 0',
          [RuleIndex, Index, RuleIndex, Index]),
      format(atom(Len),
          '  %assoc_rule_~w_action_~w_key_len = extractvalue %WamSlice %assoc_rule_~w_action_~w_key_slice, 1',
          [RuleIndex, Index, RuleIndex, Index]),
      format(atom(Missing),
          '  %assoc_rule_~w_action_~w_key_missing = icmp eq i8* %assoc_rule_~w_action_~w_key_ptr, null',
          [RuleIndex, Index, RuleIndex, Index]),
      format(atom(Branch),
          '  br i1 %assoc_rule_~w_action_~w_key_missing, label %~w, label %~w',
          [RuleIndex, Index, ActionNextLabel, HaveLabelName]),
      format(atom(KeyId),
          '  %assoc_rule_~w_action_~w_key_id = call i64 @wam_intern_atom(i8* %assoc_rule_~w_action_~w_key_ptr, i64 %assoc_rule_~w_action_~w_key_len)',
          [RuleIndex, Index, RuleIndex, Index, RuleIndex, Index]),
      format(atom(Del),
          '  call void @wam_assoc_i64_delete(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %assoc_rule_~w_action_~w_key_id)',
          [TableIndex, RuleIndex, Index]),
      format(atom(Next), '  br label %~w', [ActionNextLabel])
    },
    [Label, Slice, Ptr, Len, Missing, Branch, '', HaveLabel, KeyId, Del, Next, ''],
    plawk_assoc_rule_action_blocks(RuleIndex, Rest, NextLabel, FieldSeparator).
% `split($k, arr, "sep")`: resolve the source string, then call the split
% primitive, which clears the table and repopulates it with the pieces keyed
% 1..n (string values). $0 is the whole record (resolved directly, since the
% field-slice helper is 1-based); a positive field uses its slice (missing field
% -> empty array). The split byte is the separator literal's code.
plawk_assoc_rule_action_blocks(RuleIndex, [assoc_split_action(Index, _ArrayName, TableIndex, KeyIndex, Sep) | Rest], NextLabel, FieldSeparator) -->
    { ( Rest == []
      -> ActionNextLabel = NextLabel
      ;  NextIndex is Index + 1,
         format(atom(ActionNextLabel), 'assoc_rule_~w_action_~w',
             [RuleIndex, NextIndex])
      ),
      format(atom(Base), 'assoc_rule_~w_action_~w', [RuleIndex, Index]),
      format(atom(Label), '~w:', [Base]),
      plawk_split_call_lines(Sep, Base, TableIndex, SplitGlobals, Split),
      format(atom(Next), '  br label %~w', [ActionNextLabel]),
      ( KeyIndex =:= 0
      ->  % whole record: resolve %line's atom to its string
          format(atom(Lp), '  %~w_lp = call i64 @value_payload(%Value %line)', [Base]),
          format(atom(Sptr), '  %~w_src_ptr = call i8* @wam_atom_to_string(i64 %~w_lp)', [Base, Base]),
          format(atom(Slen), '  %~w_src_len = call i64 @strlen(i8* %~w_src_ptr)', [Base, Base]),
          append([Label, Lp, Sptr, Slen | SplitGlobals], [Split, Next, ''], Lines)
      ;   % positive field: project its slice, skip on a missing field
          format(atom(HaveLabel), '~w_have_src:', [Base]),
          format(atom(HaveLabelName), '~w_have_src', [Base]),
          format(atom(Slice),
              '  %~w_src_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)',
              [Base, KeyIndex, FieldSeparator]),
          format(atom(Ptr), '  %~w_src_ptr = extractvalue %WamSlice %~w_src_slice, 0', [Base, Base]),
          format(atom(Len), '  %~w_src_len = extractvalue %WamSlice %~w_src_slice, 1', [Base, Base]),
          format(atom(Missing), '  %~w_src_missing = icmp eq i8* %~w_src_ptr, null', [Base, Base]),
          format(atom(Branch), '  br i1 %~w_src_missing, label %~w, label %~w',
              [Base, ActionNextLabel, HaveLabelName]),
          append([Label, Slice, Ptr, Len, Missing, Branch, '', HaveLabel | SplitGlobals],
                 [Split, Next, ''], Lines)
      )
    },
    plawk_emit_lines(Lines),
    plawk_assoc_rule_action_blocks(RuleIndex, Rest, NextLabel, FieldSeparator).
% Associative add-assign `arr[$k] += DELTA`: same key-intern + missing-key
% skip as the counted inc, but the inc delta is the record's DELTA (a field
% value via @wam_atom_field_i64_value, or an integer constant) rather than 1.
plawk_assoc_rule_action_blocks(RuleIndex,
        [assoc_add_action(Index, _ArrayName, TableIndex, KeyIndex, Delta) | Rest],
        NextLabel, FieldSeparator) -->
    { ( Rest == []
      -> ActionNextLabel = NextLabel
      ;  NextIndex is Index + 1,
         format(atom(ActionNextLabel), 'assoc_rule_~w_action_~w',
             [RuleIndex, NextIndex])
      ),
      format(atom(Label), 'assoc_rule_~w_action_~w:', [RuleIndex, Index]),
      format(atom(HaveLabelName), 'assoc_rule_~w_action_~w_have_key',
          [RuleIndex, Index]),
      format(atom(HaveLabel), 'assoc_rule_~w_action_~w_have_key:',
          [RuleIndex, Index]),
      format(atom(B), 'assoc_rule_~w_action_~w', [RuleIndex, Index]),
      format(atom(Slice),
          '  %~w_key_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)',
          [B, KeyIndex, FieldSeparator]),
      format(atom(Ptr), '  %~w_key_ptr = extractvalue %WamSlice %~w_key_slice, 0', [B, B]),
      format(atom(Len), '  %~w_key_len = extractvalue %WamSlice %~w_key_slice, 1', [B, B]),
      format(atom(Missing), '  %~w_key_missing = icmp eq i8* %~w_key_ptr, null', [B, B]),
      format(atom(Branch), '  br i1 %~w_key_missing, label %~w, label %~w',
          [B, ActionNextLabel, HaveLabelName]),
      format(atom(KeyId),
          '  %~w_key_id = call i64 @wam_intern_atom(i8* %~w_key_ptr, i64 %~w_key_len)',
          [B, B, B]),
      plawk_assoc_scalar_src_lines(Delta, B, FieldSeparator, DeltaVar, DeltaLines),
      format(atom(Inc),
          '  %~w_sum = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %~w_key_id, i64 ~w)',
          [B, TableIndex, B, DeltaVar]),
      format(atom(Next), '  br label %~w', [ActionNextLabel]),
      append([[Label, Slice, Ptr, Len, Missing, Branch, '', HaveLabel, KeyId],
              DeltaLines, [Inc, Next, '']], Lines)
    },
    plawk_emit_lines(Lines),
    plawk_assoc_rule_action_blocks(RuleIndex, Rest, NextLabel, FieldSeparator).
% Row capture `arr[$k] = $0`: same key-intern + missing-key skip, then intern
% the whole current record (field 0, a stable copy of the transient line) and
% store that atom id as the table's value (str-value / replace semantics via
% @wam_assoc_i64_set). A later pass resolves the id back to the row's bytes.
plawk_assoc_rule_action_blocks(RuleIndex,
        [assoc_set_row_action(Index, _ArrayName, TableIndex, KeyIndex) | Rest],
        NextLabel, FieldSeparator) -->
    { ( Rest == []
      -> ActionNextLabel = NextLabel
      ;  NextIndex is Index + 1,
         format(atom(ActionNextLabel), 'assoc_rule_~w_action_~w',
             [RuleIndex, NextIndex])
      ),
      format(atom(Label), 'assoc_rule_~w_action_~w:', [RuleIndex, Index]),
      format(atom(HaveLabelName), 'assoc_rule_~w_action_~w_have_key',
          [RuleIndex, Index]),
      format(atom(HaveLabel), 'assoc_rule_~w_action_~w_have_key:',
          [RuleIndex, Index]),
      format(atom(B), 'assoc_rule_~w_action_~w', [RuleIndex, Index]),
      format(atom(Slice),
          '  %~w_key_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)',
          [B, KeyIndex, FieldSeparator]),
      format(atom(Ptr), '  %~w_key_ptr = extractvalue %WamSlice %~w_key_slice, 0', [B, B]),
      format(atom(Len), '  %~w_key_len = extractvalue %WamSlice %~w_key_slice, 1', [B, B]),
      format(atom(Missing), '  %~w_key_missing = icmp eq i8* %~w_key_ptr, null', [B, B]),
      format(atom(Branch), '  br i1 %~w_key_missing, label %~w, label %~w',
          [B, ActionNextLabel, HaveLabelName]),
      format(atom(KeyId),
          '  %~w_key_id = call i64 @wam_intern_atom(i8* %~w_key_ptr, i64 %~w_key_len)',
          [B, B, B]),
      % the row value: the whole record ($0 is %line, the record atom). The
      % transient line buffer is reused, so intern a stable copy of its bytes
      % (atom -> string -> re-intern) and store that id as the table's value.
      format(atom(RowLineId), '  %~w_row_lineid = extractvalue %Value %line, 1', [B]),
      format(atom(RowPtr),
          '  %~w_row_ptr = call i8* @wam_atom_to_string(i64 %~w_row_lineid)', [B, B]),
      format(atom(RowLen), '  %~w_row_len = call i64 @strlen(i8* %~w_row_ptr)', [B, B]),
      format(atom(RowId),
          '  %~w_row_id = call i64 @wam_intern_atom(i8* %~w_row_ptr, i64 %~w_row_len)',
          [B, B, B]),
      format(atom(Set),
          '  %~w_setrc = call i64 @wam_assoc_i64_set(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %~w_key_id, i64 %~w_row_id)',
          [B, TableIndex, B, B]),
      format(atom(Next), '  br label %~w', [ActionNextLabel]),
      Lines = [Label, Slice, Ptr, Len, Missing, Branch, '', HaveLabel, KeyId,
               RowLineId, RowPtr, RowLen, RowId, Set, Next, '']
    },
    plawk_emit_lines(Lines),
    plawk_assoc_rule_action_blocks(RuleIndex, Rest, NextLabel, FieldSeparator).
% Row constructor `arr[$k] = row($a, $b, ...)`: same key-intern + missing-key
% skip, then build the row as the chosen fields joined by the field separator
% (so a reader's field projection recovers them), intern it, and store the id.
% The join is a single snprintf with a compile-time format ("%.*s<sep>..."),
% each field made null-safe (empty on a missing field), into a fixed buffer.
plawk_assoc_rule_action_blocks(RuleIndex,
        [assoc_set_row_cons_action(Index, _ArrayName, TableIndex, KeyIndex, Fields) | Rest],
        NextLabel, FieldSeparator) -->
    { ( Rest == []
      -> ActionNextLabel = NextLabel
      ;  NextIndex is Index + 1,
         format(atom(ActionNextLabel), 'assoc_rule_~w_action_~w',
             [RuleIndex, NextIndex])
      ),
      format(atom(Label), 'assoc_rule_~w_action_~w:', [RuleIndex, Index]),
      format(atom(HaveLabelName), 'assoc_rule_~w_action_~w_have_key', [RuleIndex, Index]),
      format(atom(HaveLabel), 'assoc_rule_~w_action_~w_have_key:', [RuleIndex, Index]),
      format(atom(B), 'assoc_rule_~w_action_~w', [RuleIndex, Index]),
      format(atom(Slice),
          '  %~w_key_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)',
          [B, KeyIndex, FieldSeparator]),
      format(atom(Ptr), '  %~w_key_ptr = extractvalue %WamSlice %~w_key_slice, 0', [B, B]),
      format(atom(Len), '  %~w_key_len = extractvalue %WamSlice %~w_key_slice, 1', [B, B]),
      format(atom(Missing), '  %~w_key_missing = icmp eq i8* %~w_key_ptr, null', [B, B]),
      format(atom(Branch), '  br i1 %~w_key_missing, label %~w, label %~w',
          [B, ActionNextLabel, HaveLabelName]),
      format(atom(KeyId),
          '  %~w_key_id = call i64 @wam_intern_atom(i8* %~w_key_ptr, i64 %~w_key_len)',
          [B, B, B]),
      % empty-atom fallback for a missing field, and the join format string.
      format(atom(EmptyName), '~w_empty', [B]),
      format(atom(EmptyGlobal),
          '@.~w = private constant [1 x i8] zeroinitializer', [EmptyName]),
      plawk_row_cons_format_atom(Fields, FieldSeparator, FmtAtom),
      format(atom(FmtName), '~w_fmt', [B]),
      llvm_emit_c_string_global(FmtName, FmtAtom, FmtGlobal, _FmtLen, _FmtBytes),
      plawk_row_cons_field_lines(Fields, B, EmptyName, FieldSeparator, 0,
          FieldLines, ArgFrags),
      atomic_list_concat(ArgFrags, ', ', ArgsIR),
      format(atom(BufA), '  %~w_buf = alloca [4096 x i8]', [B]),
      format(atom(BufP),
          '  %~w_bufp = getelementptr [4096 x i8], [4096 x i8]* %~w_buf, i64 0, i64 0',
          [B, B]),
      format(atom(FmtP),
          '  %~w_fmtp = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0',
          [B, _FmtBytes, _FmtBytes, FmtName]),
      format(atom(Snp),
          '  %~w_wrote = call i32 (i8*, i64, i8*, ...) @snprintf(i8* %~w_bufp, i64 4096, i8* %~w_fmtp, ~w)',
          [B, B, B, ArgsIR]),
      format(atom(RowLen), '  %~w_row_len = call i64 @strlen(i8* %~w_bufp)', [B, B]),
      format(atom(RowId),
          '  %~w_row_id = call i64 @wam_intern_atom(i8* %~w_bufp, i64 %~w_row_len)',
          [B, B, B]),
      format(atom(Set),
          '  %~w_setrc = call i64 @wam_assoc_i64_set(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %~w_key_id, i64 %~w_row_id)',
          [B, TableIndex, B, B]),
      format(atom(Next), '  br label %~w', [ActionNextLabel]),
      append([[global(EmptyGlobal), global(FmtGlobal), Label, Slice, Ptr, Len,
               Missing, Branch, '', HaveLabel, KeyId],
              FieldLines,
              [BufA, BufP, FmtP, Snp, RowLen, RowId, Set, Next, '']], Lines)
    },
    plawk_emit_lines(Lines),
    plawk_assoc_rule_action_blocks(RuleIndex, Rest, NextLabel, FieldSeparator).

%% plawk_row_cons_format_atom(+Indexes, +FieldSep, -FmtAtom)
%  The snprintf format joining N fields: "%.*s" per field, the field
%  separator byte between, so a reader splitting on that separator recovers
%  the columns.
plawk_row_cons_format_atom(Indexes, FieldSep, FmtAtom) :-
    length(Indexes, N),
    length(Slots, N),
    maplist(=('%.*s'), Slots),
    atom_codes(SepAtom, [FieldSep]),
    atomic_list_concat(Slots, SepAtom, FmtAtom).

%% plawk_row_cons_field_lines(+Indexes, +Base, +EmptyName, +FieldSep, +Pos,
%%     -Lines, -ArgFrags)
%  Per constructor field: project field N of the current record, null-safe
%  (empty on a missing field), yielding an `i32 <len>, i8* <ptr>` snprintf
%  argument pair for the `%.*s` slot.
plawk_row_cons_field_lines([], _B, _Empty, _FieldSep, _Pos, [], []).
plawk_row_cons_field_lines([N | Rest], B, EmptyName, FieldSep, Pos,
        Lines, [Frag | Frags]) :-
    format(atom(Slice),
        '  %~w_c~w_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)',
        [B, Pos, N, FieldSep]),
    format(atom(Ptr), '  %~w_c~w_ptr = extractvalue %WamSlice %~w_c~w_slice, 0',
        [B, Pos, B, Pos]),
    format(atom(Len), '  %~w_c~w_len = extractvalue %WamSlice %~w_c~w_slice, 1',
        [B, Pos, B, Pos]),
    format(atom(Null), '  %~w_c~w_null = icmp eq i8* %~w_c~w_ptr, null',
        [B, Pos, B, Pos]),
    format(atom(SafePtr),
        '  %~w_c~w_sptr = select i1 %~w_c~w_null, i8* getelementptr ([1 x i8], [1 x i8]* @.~w, i64 0, i64 0), i8* %~w_c~w_ptr',
        [B, Pos, B, Pos, EmptyName, B, Pos]),
    format(atom(SafeLen), '  %~w_c~w_slen = select i1 %~w_c~w_null, i64 0, i64 %~w_c~w_len',
        [B, Pos, B, Pos, B, Pos]),
    format(atom(Lenw), '  %~w_c~w_lenw = trunc i64 %~w_c~w_slen to i32',
        [B, Pos, B, Pos]),
    format(atom(Frag), 'i32 %~w_c~w_lenw, i8* %~w_c~w_sptr', [B, Pos, B, Pos]),
    ThisLines = [Slice, Ptr, Len, Null, SafePtr, SafeLen, Lenw],
    Pos1 is Pos + 1,
    plawk_row_cons_field_lines(Rest, B, EmptyName, FieldSep, Pos1, RestLines, Frags),
    append(ThisLines, RestLines, Lines).

%% plawk_record_program_ok(+Descriptor, +Rules, +EndPrintFields) is semidet.
%
%  Text mode accepts everything. Binary mode whitelists the record
%  representation's supported surface: numeric guards and arithmetic on
%  i64 fields, float($N) on f64 fields, prints of either, NR/NF, scalar
%  state, if/else, next/break, and representation-free END prints.
%  Text-shaped forms (regex, string equality, substr/index/length/case,
%  $0, assoc arrays, foreign calls) fail the whole driver clause.
%% plawk_assoc_record_program_ok(+Descriptor, +Rules, +EndPrintFields)
%
%  Binary-mode whitelist for associative programs: guards must be
%  binfmt-compatible, every counted key must be an i64 field (the raw
%  field value is the table key -- no interning), and END lookups use
%  integer keys. Text mode accepts everything as before.
plawk_assoc_record_program_ok(binfmt(Types), Rules, EndPrintFields) :-
    !,
    forall(member(rule(Pattern, Actions), Rules),
        ( plawk_binfmt_pattern_ok(binfmt(Types), Pattern),
          forall(member(Action, Actions),
              plawk_binfmt_assoc_action_ok(binfmt(Types), Action))
        )),
    forall(( member(Field, EndPrintFields),
             Field = assoc(_Array, Key)
           ),
        % Integer literals in END lookups; the loop variable inside a
        % for-in body (the key is already the raw i64 slot key there).
        ( Key = int(_) ; Key = var(_) )).
% Tagged-union rules (already flattened): guards and counted keys type
% against each rule's own arm; keys from every arm land in one shared
% i64 keyspace, as in awk where counts[k] is one array no matter which
% rule updated it.
plawk_assoc_record_program_ok(binfmt_union(_Arms), Rules, EndPrintFields) :-
    !,
    forall(member(rule(arm_pat(_Tag, ArmTypes, Pattern), Actions), Rules),
        ( plawk_binfmt_pattern_ok(binfmt(ArmTypes), Pattern),
          forall(member(Action, Actions),
              plawk_binfmt_assoc_action_ok(binfmt(ArmTypes), Action))
        )),
    forall(( member(Field, EndPrintFields),
             Field = assoc(_Array, Key)
           ),
        ( Key = int(_) ; Key = var(_) )).
plawk_assoc_record_program_ok(_FieldSeparator, Rules, EndPrintFields) :-
    % Text mode: integer assoc keys would collide with atom ids, so they
    % stay binary-only -- EXCEPT positional-array tables, whose keys are
    % integer positions (never interned atom ids), so int lookups on them
    % are unambiguous.
    plawk_program_posarray_arrays(Rules, PosArrays),
    forall(( member(Field, EndPrintFields),
             Field = assoc(Array, Key)
           ),
        ( Key \= int(_)
        ; Array = var(Name), memberchk(Name, PosArrays)
        )).

%% plawk_program_posarray_arrays(+Rules, -Names) is det.
%  The array names bound by an `as array` (positional) target, walked
%  through nested actions.
plawk_program_posarray_arrays(Rules, Names) :-
    findall(Name,
        ( member(rule(_P, Actions), Rules),
          member(Action, Actions),
          plawk_posarray_bind_name(Action, Name)
        ),
        Names0),
    sort(Names0, Names).

plawk_posarray_bind_name(dynposarray_bind(var(Name), _Call), Name).
plawk_posarray_bind_name(dynposarray_bind_str(var(Name), _Call), Name).
plawk_posarray_bind_name(Term, Name) :-
    compound(Term),
    arg(_, Term, Sub),
    plawk_posarray_bind_name(Sub, Name).

plawk_binfmt_assoc_action_ok(_Descriptor, next) :- !.
plawk_binfmt_assoc_action_ok(_Descriptor, break) :- !.
plawk_binfmt_assoc_action_ok(Descriptor, inc_assoc(var(_Name), field(Index))) :-
    !,
    plawk_binfmt_field_type(Descriptor, Index, i64).
plawk_binfmt_assoc_action_ok(Descriptor, dynassoc_bind(var(_Name), Call)) :-
    !,
    plawk_dynassoc_call_parts(Call, Args, _Shim),
    plawk_binfmt_foreign_args_ok(Descriptor, Args).
plawk_binfmt_assoc_action_ok(Descriptor, dynposarray_bind(var(_Name), Call)) :-
    !,
    plawk_dynassoc_call_parts(posarray(Call), Args, _Shim),
    plawk_binfmt_foreign_args_ok(Descriptor, Args).
plawk_binfmt_assoc_action_ok(Descriptor, dynposarray_bind_str(var(_Name), Call)) :-
    !,
    plawk_dynassoc_call_parts(posarray_str(Call), Args, _Shim),
    plawk_binfmt_foreign_args_ok(Descriptor, Args).

plawk_record_program_ok(binfmt(Types), Rules, _EndPrintFields) :-
    !,
    forall(member(rule(Pattern, Actions), Rules),
        ( plawk_binfmt_pattern_ok(binfmt(Types), Pattern),
          plawk_binfmt_actions_ok(binfmt(Types), Actions)
        )).
plawk_record_program_ok(_FieldSeparator, _Rules, _EndPrintFields).

plawk_binfmt_pattern_ok(_Descriptor, always) :- !.
plawk_binfmt_pattern_ok(Descriptor, field_cmp(Index, _Op, _Value)) :-
    !,
    plawk_binfmt_field_type(Descriptor, Index, i64).
plawk_binfmt_pattern_ok(Descriptor, field_eq(Index, _Value)) :-
    !,
    plawk_binfmt_field_type(Descriptor, Index, s(_Width)).
plawk_binfmt_pattern_ok(Descriptor, prolog_guard(Name, Args)) :-
    !,
    atom(Name),
    plawk_binfmt_foreign_args_ok(Descriptor, Args).
plawk_binfmt_pattern_ok(Descriptor, and_pat(Left, Right)) :-
    !,
    plawk_binfmt_pattern_ok(Descriptor, Left),
    plawk_binfmt_pattern_ok(Descriptor, Right).
plawk_binfmt_pattern_ok(Descriptor, or_pat(Left, Right)) :-
    !,
    plawk_binfmt_pattern_ok(Descriptor, Left),
    plawk_binfmt_pattern_ok(Descriptor, Right).
plawk_binfmt_pattern_ok(Descriptor, not_pat(Pattern)) :-
    !,
    plawk_binfmt_pattern_ok(Descriptor, Pattern).

plawk_binfmt_actions_ok(Descriptor, Actions) :-
    forall(member(Action, Actions),
        plawk_binfmt_action_ok(Descriptor, Action)).

plawk_binfmt_action_ok(_Descriptor, next) :- !.
plawk_binfmt_action_ok(_Descriptor, break) :- !.
plawk_binfmt_action_ok(Descriptor, dynrec_bind(Vars, Call, Types)) :-
    !,
    plawk_dynrec_bind_ok(dynrec_bind(Vars, Call, Types)),
    plawk_dynrec_call_parts(Call, Args, _ShimName),
    plawk_binfmt_foreign_args_ok(Descriptor, Args).
plawk_binfmt_action_ok(Descriptor, inc(var(_Name))) :- !,
    Descriptor = binfmt(_).
plawk_binfmt_action_ok(Descriptor, add(var(_Name), Expr)) :-
    !,
    ( plawk_expr_is_double(Expr)
    -> plawk_binfmt_f64_expr_ok(Descriptor, Expr)
    ;  plawk_binfmt_i64_expr_ok(Descriptor, Expr)
    ).
plawk_binfmt_action_ok(Descriptor, set(var(_Name), Expr)) :-
    !,
    ( plawk_expr_is_double(Expr)
    -> plawk_binfmt_f64_expr_ok(Descriptor, Expr)
    ;  plawk_binfmt_i64_expr_ok(Descriptor, Expr)
    ).
plawk_binfmt_action_ok(Descriptor, print(Fields)) :-
    !,
    forall(member(Field, Fields),
        plawk_binfmt_print_field_ok(Descriptor, Field)).
plawk_binfmt_action_ok(Descriptor, printf(string(_Format), Args)) :-
    !,
    forall(member(Arg, Args),
        plawk_binfmt_print_field_ok(Descriptor, Arg)).
plawk_binfmt_action_ok(Descriptor, if(Pattern, ThenActions, ElseActions)) :-
    !,
    plawk_binfmt_pattern_ok(Descriptor, Pattern),
    plawk_binfmt_actions_ok(Descriptor, ThenActions),
    plawk_binfmt_actions_ok(Descriptor, ElseActions).
plawk_binfmt_action_ok(Descriptor, writebin_out(Types, Fields)) :-
    !,
    plawk_binfmt_writebin_args_ok(Descriptor, Types, Fields).
plawk_binfmt_action_ok(Descriptor, writebin_arm_out(_Tag, ArmTypes, Fields)) :-
    !,
    plawk_binfmt_writebin_args_ok(Descriptor, ArmTypes, Fields).
plawk_binfmt_action_ok(Descriptor, foreach_loop(_Layout, Body)) :-
    !,
    plawk_binfmt_actions_ok(Descriptor, Body).
plawk_binfmt_action_ok(Descriptor, while_loop(_Cond, Body)) :-
    !,
    plawk_binfmt_actions_ok(Descriptor, Body).
plawk_binfmt_action_ok(Descriptor, do_while_loop(Body, _Cond)) :-
    !,
    plawk_binfmt_actions_ok(Descriptor, Body).

plawk_binfmt_writebin_args_ok(_Descriptor, [], []).
plawk_binfmt_writebin_args_ok(Descriptor, [i64 | Types], [Field | Fields]) :-
    plawk_binfmt_i64_expr_ok(Descriptor, Field),
    plawk_binfmt_writebin_args_ok(Descriptor, Types, Fields).
plawk_binfmt_writebin_args_ok(Descriptor, [f64 | Types], [Field | Fields]) :-
    ( plawk_expr_is_double(Field)
    -> plawk_binfmt_f64_expr_ok(Descriptor, Field)
    ;  plawk_binfmt_i64_expr_ok(Descriptor, Field)
    ),
    plawk_binfmt_writebin_args_ok(Descriptor, Types, Fields).
plawk_binfmt_writebin_args_ok(Descriptor, [s(Width) | Types], [Field | Fields]) :-
    plawk_binfmt_writebin_str_ok(Descriptor, Field, Width),
    plawk_binfmt_writebin_args_ok(Descriptor, Types, Fields).
plawk_binfmt_writebin_args_ok(Descriptor, [lps(Cap) | Types], [Field | Fields]) :-
    plawk_binfmt_writebin_str_ok(Descriptor, Field, Cap),
    plawk_binfmt_writebin_args_ok(Descriptor, Types, Fields).
plawk_binfmt_writebin_args_ok(Descriptor, [rep(Cap, ElemTypes) | Types],
        [Field | Fields]) :-
    plawk_binfmt_writebin_rep_ok(Descriptor, rep(Cap, ElemTypes), Field),
    plawk_binfmt_writebin_args_ok(Descriptor, Types, Fields).

% rep passthrough: the argument names the input rep's COUNT field, and
% the input rep must match the output slot exactly (same cap, same
% element layout) -- count and live elements copy through unchanged.
plawk_binfmt_writebin_rep_ok(binfmt(InTypes), rep(Cap, ElemTypes),
        field(CountIndex)) :-
    plawk_binfmt_rep_info(InTypes, CountIndex, Cap, _ElemArity),
    memberchk(rep(Cap, ElemTypes), InTypes).

plawk_binfmt_writebin_str_ok(_Descriptor, string(Value), Width) :-
    !,
    string_length(Value, Length),
    Length =< Width.
plawk_binfmt_writebin_str_ok(Descriptor, field(FieldIndex), Width) :-
    plawk_binfmt_field_type(Descriptor, FieldIndex, s(SourceWidth)),
    SourceWidth =< Width.

plawk_binfmt_print_field_ok(_Descriptor, string(_Value)) :- !.
% a record-view string field ($k) printed as a byte slice
plawk_binfmt_print_field_ok(_Descriptor, blob_slice_vars(var(P), var(L))) :-
    !,
    atom(P), atom(L).
plawk_binfmt_print_field_ok(_Descriptor, special('NR')) :- !.
plawk_binfmt_print_field_ok(_Descriptor, special('NF')) :- !.
plawk_binfmt_print_field_ok(Descriptor, field(Index)) :-
    !,
    plawk_binfmt_field_type(Descriptor, Index, _Type).
plawk_binfmt_print_field_ok(Descriptor, float_field(Index)) :-
    !,
    plawk_binfmt_field_type(Descriptor, Index, f64).
plawk_binfmt_print_field_ok(Descriptor, Expr) :-
    plawk_expr_is_double(Expr),
    !,
    plawk_binfmt_f64_expr_ok(Descriptor, Expr).
plawk_binfmt_print_field_ok(Descriptor, Expr) :-
    plawk_binfmt_i64_expr_ok(Descriptor, Expr).

plawk_binfmt_i64_expr_ok(_Descriptor, int(Value)) :-
    integer(Value),
    !.
plawk_binfmt_i64_expr_ok(_Descriptor, var(Name)) :-
    atom(Name),
    !.
plawk_binfmt_i64_expr_ok(_Descriptor, special('NR')) :- !.
plawk_binfmt_i64_expr_ok(_Descriptor, special('NF')) :- !.
plawk_binfmt_i64_expr_ok(Descriptor, field(Index)) :-
    !,
    plawk_binfmt_field_type(Descriptor, Index, i64).
plawk_binfmt_i64_expr_ok(Descriptor, int(field(Index))) :-
    !,
    plawk_binfmt_field_type(Descriptor, Index, i64).
plawk_binfmt_i64_expr_ok(Descriptor, prolog_call(Name, Args)) :-
    !,
    atom(Name),
    plawk_binfmt_foreign_args_ok(Descriptor, Args).
plawk_binfmt_i64_expr_ok(Descriptor, dyncall(Args)) :-
    !,
    plawk_binfmt_foreign_args_ok(Descriptor, Args).
plawk_binfmt_i64_expr_ok(Descriptor, dyncall_named(_Name, Args)) :-
    !,
    plawk_binfmt_foreign_args_ok(Descriptor, Args).
plawk_binfmt_i64_expr_ok(Descriptor, Expr) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    plawk_binfmt_i64_expr_ok(Descriptor, Left),
    plawk_binfmt_i64_expr_ok(Descriptor, Right).

%% plawk_binfmt_foreign_args_ok(+Descriptor, +Args)
%
%  Foreign-call arguments in binary mode: integer and string literals,
%  i64 fields (marshaled as WAM integers), and blob fields (marshaled
%  as the transient payload atom). At most one blob per call -- all
%  blobs share the one transient buffer, so a second would overwrite
%  the first before the call.
plawk_binfmt_foreign_args_ok(Descriptor, Args) :-
    Args = [_ | _],
    forall(member(Arg, Args),
        plawk_binfmt_foreign_arg_ok(Descriptor, Arg)),
    findall(F,
        ( member(field(F), Args),
          plawk_binfmt_field_type(Descriptor, F, blob(_))
        ),
        Blobs),
    length(Blobs, BlobCount),
    BlobCount =< 1.

plawk_binfmt_foreign_arg_ok(_Descriptor, int(Value)) :-
    integer(Value),
    !.
plawk_binfmt_foreign_arg_ok(_Descriptor, string(Value)) :-
    string(Value),
    !.
plawk_binfmt_foreign_arg_ok(Descriptor, field(Index)) :-
    integer(Index),
    Index >= 1,
    plawk_binfmt_field_type(Descriptor, Index, Type),
    ( Type == i64 ; Type == f64 ; Type = blob(_Cap) ),
    !.

plawk_binfmt_f64_expr_ok(Descriptor, float_field(Index)) :-
    !,
    plawk_binfmt_field_type(Descriptor, Index, f64).
plawk_binfmt_f64_expr_ok(_Descriptor, float_const(_Mantissa, _Denominator)) :- !.
plawk_binfmt_f64_expr_ok(Descriptor, float_dyncall(Args)) :-
    !,
    plawk_binfmt_foreign_args_ok(Descriptor, Args).
plawk_binfmt_f64_expr_ok(Descriptor, float_dyncall_named(_Name, Args)) :-
    !,
    plawk_binfmt_foreign_args_ok(Descriptor, Args).
plawk_binfmt_f64_expr_ok(Descriptor, float_dyncall_at(_Source, Args)) :-
    !,
    plawk_binfmt_foreign_args_ok(Descriptor, Args).
plawk_binfmt_f64_expr_ok(Descriptor, float_call(Name, Args)) :-
    !,
    atom(Name),
    plawk_binfmt_foreign_args_ok(Descriptor, Args).
plawk_binfmt_f64_expr_ok(Descriptor, Expr) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    !,
    plawk_binfmt_f64_expr_ok(Descriptor, Left),
    plawk_binfmt_f64_expr_ok(Descriptor, Right).
plawk_binfmt_f64_expr_ok(Descriptor, Expr) :-
    plawk_binfmt_i64_expr_ok(Descriptor, Expr).

%% plawk_binfmt_field_load_lines(+Descriptor, +Index, +Base, -ValueIR, -Lines)
%
%  Typed field access on a fixed-layout binary record: a load at the
%  compile-time offset from the %rec buffer. No parsing, no separators.
plawk_binfmt_field_load_lines(Descriptor, Index, Base, ValueIR, Lines) :-
    plawk_binfmt_field_type(Descriptor, Index, Type),
    plawk_binfmt_field_offset(Descriptor, Index, Offset),
    plawk_binfmt_llvm_type(Type, LLVMType),
    format(atom(ValueIR), '%~w', [Base]),
    format(atom(GepIR),
        '  %~w_fp = getelementptr i8, i8* %rec, i64 ~w', [Base, Offset]),
    format(atom(CastIR),
        '  %~w_tp = bitcast i8* %~w_fp to ~w*', [Base, Base, LLVMType]),
    format(atom(LoadIR),
        '  ~w = load ~w, ~w* %~w_tp, align 1', [ValueIR, LLVMType, LLVMType, Base]),
    Lines = [GepIR, CastIR, LoadIR].

plawk_binfmt_llvm_type(i64, i64).
plawk_binfmt_llvm_type(f64, double).

plawk_binfmt_icmp_op(eq, eq).
plawk_binfmt_icmp_op(ne, ne).
plawk_binfmt_icmp_op(lt, slt).
plawk_binfmt_icmp_op(le, sle).
plawk_binfmt_icmp_op(gt, sgt).
plawk_binfmt_icmp_op(ge, sge).

% Field separator code. A single character sets a literal byte separator (awk
% treats a one-char FS literally, even a regex metachar like "."). A multi-char
% FS is an ERE regex and yields the reserved sentinel 0 -- the field runtime
% dispatches sentinel 0 to the FS-regex splitter, and the program stores the
% pattern into @wam_fs_regex_pattern_ptr at startup (see plawk_fs_regex_pattern).
% Default (no FS) and an empty FS fall back to 32 (whitespace).
plawk_field_separator(BeginClauses, FieldSeparator) :-
    (   member(begin(Actions), BeginClauses),
        member(set(var('FS'), string(Value)), Actions)
    ->  (   string_codes(Value, [FieldSeparator])
        ->  true
        ;   string_length(Value, Len), Len >= 2
        ->  FieldSeparator = 0
        ;   FieldSeparator = 32
        )
    ;   FieldSeparator = 32
    ).

%% plawk_fs_regex_pattern(+BeginClauses, -Pattern)
%
%  The multi-char FS regex pattern (a `BEGIN { FS = "…" }` value of length >= 2),
%  or fails when FS is a single char / unset. Multi-char FS is a POSIX ERE, so
%  every field read splits the record on it (sentinel separator 0).
plawk_fs_regex_pattern(BeginClauses, Pattern) :-
    member(begin(Actions), BeginClauses),
    member(set(var('FS'), string(Pattern)), Actions),
    string_length(Pattern, Len),
    Len >= 2.

%% plawk_record_descriptor(+BeginClauses, -Descriptor)
%
%  Text mode yields the single-byte field separator code as before;
%  BEGIN { BINFMT = "i64 i64 f64" } yields binfmt(Types) for fixed-
%  layout binary records: one 8-byte native-endian field per type,
%  fields numbered $1..$N, record size 8 * N.
plawk_record_descriptor(BeginClauses, binfmt_union(Arms)) :-
    plawk_begin_union_arms(BeginClauses, Arms),
    !.
plawk_record_descriptor(BeginClauses, binfmt(Types)) :-
    plawk_begin_binfmt_types(BeginClauses, Types),
    !.
plawk_record_descriptor(BeginClauses, FieldSeparator) :-
    plawk_field_separator(BeginClauses, FieldSeparator).

plawk_begin_has_binfmt(BeginClauses) :-
    plawk_begin_binfmt_types(BeginClauses, _Types),
    !.
plawk_begin_has_binfmt(BeginClauses) :-
    plawk_begin_union_arms(BeginClauses, _Arms).

%% plawk_begin_union_arms(+BeginClauses, -Arms)
%
%  BINFMT = "case(i64 f64 | lps16 i64)" declares a tagged union: every
%  record starts with an 8-byte native-endian tag selecting one of the
%  |-separated arm layouts (arm indices are 0-based, in declaration
%  order). Arms is a list of type lists.
plawk_begin_union_arms(BeginClauses, Arms) :-
    member(begin(Actions), BeginClauses),
    member(set(var('BINFMT'), string(Fmt)), Actions),
    string_concat("case(", Rest, Fmt),
    string_concat(Body, ")", Rest),
    split_string(Body, "|", " ", ArmStrs),
    ArmStrs \== [],
    maplist(plawk_union_arm_types, ArmStrs, Arms).

plawk_union_arm_types(ArmStr, Types) :-
    % the shared tokenizer joins parenthesized types, so an arm can
    % carry a rep: "case(i64 rep4(lps8 i64) | lps16 i64)"
    plawk_binfmt_tokens(ArmStr, Parts),
    Parts \== [],
    maplist(plawk_binfmt_type, Parts, Types).

plawk_begin_binfmt_types(BeginClauses, Types) :-
    member(begin(Actions), BeginClauses),
    member(set(var('BINFMT'), string(Fmt)), Actions),
    plawk_binfmt_tokens(Fmt, Parts),
    Parts \== [],
    maplist(plawk_binfmt_type, Parts, Types).

%% plawk_binfmt_tokens(+Fmt, -Parts)
%
%  Space-split, but a token containing "(" absorbs following tokens
%  until its ")" closes, so "i64 rep4(i64 f64)" tokenizes as
%  ["i64", "rep4(i64 f64)"].
plawk_binfmt_tokens(Fmt, Parts) :-
    split_string(Fmt, " ", " ", Parts0),
    exclude(==(""), Parts0, Parts1),
    plawk_binfmt_join_parens(Parts1, Parts).

plawk_binfmt_join_parens([], []).
plawk_binfmt_join_parens([Part | Rest], [Joined | Parts]) :-
    sub_string(Part, _, _, _, "("),
    \+ sub_string(Part, _, _, _, ")"),
    !,
    plawk_binfmt_take_until_close(Rest, Taken, Remaining),
    atomic_list_concat([Part | Taken], ' ', JoinedAtom),
    atom_string(JoinedAtom, Joined),
    plawk_binfmt_join_parens(Remaining, Parts).
plawk_binfmt_join_parens([Part | Rest], [Part | Parts]) :-
    plawk_binfmt_join_parens(Rest, Parts).

plawk_binfmt_take_until_close([Part | Rest], [Part], Rest) :-
    sub_string(Part, _, _, _, ")"),
    !.
plawk_binfmt_take_until_close([Part | Rest], [Part | Taken], Remaining) :-
    plawk_binfmt_take_until_close(Rest, Taken, Remaining).

plawk_binfmt_type("i64", i64).
plawk_binfmt_type("f64", f64).
% lpsN: a length-prefixed string on the wire (8-byte native-endian
% length, then that many payload bytes, at most N). The varlen reader
% materializes it into the record buffer as a fixed NUL-padded N-byte
% slot, so downstream consumers see it exactly like an sN field.
plawk_binfmt_type(Part, lps(Cap)) :-
    string_concat("lps", CapStr, Part),
    number_string(Cap, CapStr),
    integer(Cap),
    Cap > 0.
% repK(elem types): bounded repetition -- an 8-byte native-endian
% element count C (0 <= C <= K), then C elements each laid out per the
% parenthesized fixed-width types. Access layout: the count as an i64
% field, then K flattened element field groups (zero-filled past C).
plawk_binfmt_type(Part, rep(Cap, ElemTypes)) :-
    string_concat("rep", Rest0, Part),
    sub_string(Rest0, Before, 1, _, "("),
    !,
    sub_string(Rest0, 0, Before, _, CapStr),
    number_string(Cap, CapStr),
    integer(Cap),
    Cap > 0,
    Skip is Before + 1,
    sub_string(Rest0, Skip, _, 0, Rest1),
    string_concat(Body, ")", Rest1),
    split_string(Body, " ", " ", ElemParts0),
    exclude(==(""), ElemParts0, ElemParts),
    ElemParts \== [],
    maplist(plawk_binfmt_type, ElemParts, ElemTypes),
    % Elements may be fixed-width (i64/f64/sN -> one bulk read of the
    % whole region) or contain lpsN strings (variable wire size per
    % element -> the reader loops, parsing one element at a time into
    % its fixed in-memory slot group). Nested rep/blob stay out: the
    % access layout must remain one flat fixed-offset slot group per
    % element.
    forall(member(ElemType, ElemTypes),
        ( memberchk(ElemType, [i64, f64]) ; ElemType = s(_) ; ElemType = lps(_) )).
% blobN: a length-prefixed binary payload (8-byte length, then up to N
% payload bytes) whose ONLY consumer is a compiled-Prolog foreign call:
% the record loop stays native for framing and hands the payload to a
% WAM-compiled predicate (a DCG over the bytes) through the foreign
% bridge -- the Tier-2 composition of PLAWK_DCG_BINARY_READERS.md.
% Payload bytes must be NUL-free (the transient atom carrying them to
% Prolog is a C string).
plawk_binfmt_type(Part, blob(Cap)) :-
    string_concat("blob", CapStr, Part),
    number_string(Cap, CapStr),
    integer(Cap),
    Cap > 0.
% sN: a fixed-width string field of N bytes. Values shorter than N are
% NUL-terminated inside the field; a full-width value has no NUL.
plawk_binfmt_type(Part, s(Width)) :-
    string_concat("s", WidthStr, Part),
    number_string(Width, WidthStr),
    integer(Width),
    Width > 0.

%% plawk_binfmt_field_type(+Descriptor, +Index, -Type)
%
%  The ACCESS type of a field: what guards, prints, assoc keys, and
%  writers see. lps(Cap) fields access as s(Cap) -- the varlen reader
%  has already materialized them as fixed NUL-padded slots in the
%  record buffer.
plawk_binfmt_field_type(binfmt(Types), Index, Type) :-
    integer(Index),
    Index >= 1,
    plawk_binfmt_access_types(Types, AccessTypes),
    nth1(Index, AccessTypes, Type).

plawk_binfmt_access_type(lps(Cap), s(Cap)) :- !.
plawk_binfmt_access_type(Type, Type).

%% plawk_binfmt_access_types(+StoredTypes, -AccessTypes)
%
%  Expand stored types into the flat per-field access list: a
%  rep(K, Elems) contributes its i64 count field followed by K copies
%  of the element fields; everything else is one field.
plawk_binfmt_access_types(Types, AccessTypes) :-
    plawk_binfmt_declared_access_types(Types, Declared),
    % A rep layout appends one hidden staging element group at the very
    % end of the record buffer: the foreach runtime loop memcpys the
    % current element there so the body's field accesses stay
    % compile-time offsets. Appending after all declared fields keeps
    % user-visible numbering and offsets unchanged.
    ( memberchk(rep(_Cap, ElemTypes), Types)
    ->  maplist(plawk_binfmt_access_type, ElemTypes, StageAccess),
        append(Declared, StageAccess, AccessTypes)
    ;   AccessTypes = Declared
    ).

plawk_binfmt_declared_access_types([], []).
plawk_binfmt_declared_access_types([rep(Cap, ElemTypes) | Rest], AccessTypes) :-
    !,
    maplist(plawk_binfmt_access_type, ElemTypes, ElemAccess),
    findall(ElemField,
        ( between(1, Cap, _), member(ElemField, ElemAccess) ),
        Repeated),
    plawk_binfmt_declared_access_types(Rest, RestAccess),
    append([i64 | Repeated], RestAccess, AccessTypes).
plawk_binfmt_declared_access_types([StoredType | Rest], [Type | RestAccess]) :-
    plawk_binfmt_access_type(StoredType, Type),
    plawk_binfmt_declared_access_types(Rest, RestAccess).

plawk_binfmt_type_width(i64, 8).
plawk_binfmt_type_width(f64, 8).
plawk_binfmt_type_width(s(Width), Width).
% in-memory width of the materialized slot, not the wire size
plawk_binfmt_type_width(lps(Cap), Cap).
plawk_binfmt_type_width(rep(Cap, ElemTypes), Width) :-
    maplist(plawk_binfmt_type_width, ElemTypes, ElemWidths),
    sum_list(ElemWidths, ElemSize),
    Width is 8 + Cap * ElemSize.
% in-memory: the actual length (i64), then the payload bytes
plawk_binfmt_type_width(blob(Cap), Width) :-
    Width is 8 + Cap.

plawk_binfmt_has_varlen(Types) :-
    ( memberchk(lps(_Cap), Types)
    ; memberchk(rep(_K, _Elems), Types)
    ; memberchk(blob(_C), Types)
    ),
    !.

plawk_binfmt_field_offset(binfmt(Types), Index, Offset) :-
    plawk_binfmt_access_types(Types, AccessTypes),
    PrefixLen is Index - 1,
    length(PrefixTypes, PrefixLen),
    append(PrefixTypes, _, AccessTypes),
    maplist(plawk_binfmt_type_width, PrefixTypes, Widths),
    sum_list(Widths, Offset).

plawk_binfmt_record_size(binfmt(Types), Size) :-
    plawk_binfmt_access_types(Types, AccessTypes),
    maplist(plawk_binfmt_type_width, AccessTypes, Widths),
    sum_list(Widths, Size).

%% plawk_resolve_writebin_rules(+BeginClauses, +Rules0, -Rules, -WritebinPlan)
%
%  writebin actions write one fixed-layout binary record on stdout per
%  call, laid out per BEGIN { OUTFMT = "i64 f64 ..." }. This pre-pass
%  stamps each writebin(Fields) with the resolved output types
%  (writebin_out(Types, Fields)) so downstream validation and emission
%  are self-contained; it fails (rejecting the program) when writebin
%  appears without OUTFMT, an argument count mismatches the layout, or
%  the layout contains an unknown field type (i64, f64, and sN are
%  supported).
plawk_resolve_writebin_rules(BeginClauses, Rules0, Rules, WritebinPlan) :-
    ( plawk_rules_have_writebin(Rules0)
    ->  plawk_begin_out_spec(BeginClauses, OutSpec),
        plawk_writebin_plan(OutSpec, WritebinPlan),
        maplist(plawk_resolve_writebin_rule(OutSpec), Rules0, Rules)
    ;   WritebinPlan = none,
        Rules = Rules0
    ).

%% plawk_begin_out_spec(+BeginClauses, -OutSpec)
%
%  OUTFMT is either one flat layout or a tagged union of arm layouts
%  (`OUTFMT = "case(arm0 | arm1)"`, same spelling as BINFMT). With a
%  union, every writebin site statically targets one arm via
%  `writebin case K, args`.
plawk_begin_out_spec(BeginClauses, union(Arms)) :-
    plawk_begin_outfmt_union_arms(BeginClauses, Arms),
    !.
plawk_begin_out_spec(BeginClauses, flat(Types)) :-
    plawk_begin_outfmt_types(BeginClauses, Types).

plawk_begin_outfmt_union_arms(BeginClauses, Arms) :-
    member(begin(Actions), BeginClauses),
    member(set(var('OUTFMT'), string(Fmt)), Actions),
    string_concat("case(", Rest, Fmt),
    string_concat(Body, ")", Rest),
    split_string(Body, "|", " ", ArmStrs),
    ArmStrs \== [],
    maplist(plawk_union_arm_types, ArmStrs, Arms).

plawk_writebin_plan(flat(Types), outfmt(Types, Size)) :-
    forall(member(Type, Types), plawk_outfmt_type_ok(Type)),
    plawk_binfmt_record_size(binfmt(Types), Size).
plawk_writebin_plan(union(Arms), outfmt_union(Arms, BufSize)) :-
    % arm slots: the varlen writer set minus rep (a tagged rep write
    % is a later slice); the shared buffer holds the tag or any one
    % staged slot, so size it to the widest arm (at least 8)
    forall(member(ArmTypes, Arms),
        forall(member(Type, ArmTypes),
            ( memberchk(Type, [i64, f64]) ; Type = s(_) ; Type = lps(_) ))),
    findall(Size,
        ( member(ArmTypes, Arms),
          plawk_binfmt_record_size(binfmt(ArmTypes), Size)
        ),
        Sizes),
    max_list([8 | Sizes], BufSize).

plawk_rules_have_writebin(Rules) :-
    member(rule(_Pattern, Actions), Rules),
    plawk_actions_have_writebin(Actions),
    !.

%% plawk_resolve_union_writebin_blocks(+BeginClauses, +CaseBlocks0,
%%     -CaseBlocks, -WritebinPlan)
%
%  The case-block variant of the writebin pre-pass: OUTFMT is
%  program-wide (one output layout regardless of which arm produced
%  the record), so the plan is built once and every arm's rules are
%  stamped with it. Source-field typing against each rule's own arm
%  happens downstream via the per-rule descriptor.
plawk_resolve_union_writebin_blocks(BeginClauses, CaseBlocks0, CaseBlocks,
        WritebinPlan) :-
    ( member(case_arm(_I, ArmRules), CaseBlocks0),
      plawk_rules_have_writebin(ArmRules)
    ->  plawk_begin_out_spec(BeginClauses, OutSpec),
        plawk_writebin_plan(OutSpec, WritebinPlan),
        maplist(plawk_resolve_union_writebin_block(OutSpec), CaseBlocks0,
            CaseBlocks)
    ;   WritebinPlan = none,
        CaseBlocks = CaseBlocks0
    ).

plawk_resolve_union_writebin_block(OutSpec, case_arm(Index, Rules0),
        case_arm(Index, Rules)) :-
    maplist(plawk_resolve_writebin_rule(OutSpec), Rules0, Rules).

plawk_actions_have_writebin(Actions) :-
    member(Action, Actions),
    ( Action = writebin(_Fields)
    ; Action = writebin_arm(_Index, _AFields)
    ; Action = if(_Pattern, ThenActions, ElseActions),
      ( plawk_actions_have_writebin(ThenActions)
      ; plawk_actions_have_writebin(ElseActions)
      )
    ),
    !.

plawk_resolve_writebin_rule(OutSpec, rule(Pattern, Actions0), rule(Pattern, Actions)) :-
    maplist(plawk_resolve_writebin_action(OutSpec), Actions0, Actions).

plawk_resolve_writebin_action(flat(Types), writebin(Fields),
        writebin_out(Types, Fields)) :-
    !,
    length(Types, N),
    length(Fields, N).
% a plain writebin cannot pick an arm, and an arm-targeted writebin is
% meaningless against a flat layout -- both reject the program
plawk_resolve_writebin_action(union(_Arms), writebin(_Fields), _) :-
    !,
    fail.
plawk_resolve_writebin_action(flat(_Types), writebin_arm(_Index, _Fields), _) :-
    !,
    fail.
plawk_resolve_writebin_action(union(Arms), writebin_arm(Index, Fields),
        writebin_arm_out(Index, ArmTypes, Fields)) :-
    !,
    nth0(Index, Arms, ArmTypes),
    length(ArmTypes, N),
    length(Fields, N).
plawk_resolve_writebin_action(OutSpec, if(Pattern, Then0, Else0), if(Pattern, Then, Else)) :-
    !,
    maplist(plawk_resolve_writebin_action(OutSpec), Then0, Then),
    maplist(plawk_resolve_writebin_action(OutSpec), Else0, Else).
plawk_resolve_writebin_action(_OutSpec, Action, Action).

plawk_begin_outfmt_types(BeginClauses, Types) :-
    member(begin(Actions), BeginClauses),
    member(set(var('OUTFMT'), string(Fmt)), Actions),
    % the shared tokenizer joins parenthesized types, so OUTFMT can
    % carry a rep: "i64 rep4(i64 f64)"
    plawk_binfmt_tokens(Fmt, Parts),
    Parts \== [],
    maplist(plawk_binfmt_type, Parts, Types).

% Entry-block record buffer: one alloca reused by every writebin call
% (a loop-body alloca would grow the stack per record). The union plan
% also resolves the buffer's element pointer once, so per-site tagged
% emitters need no knowledge of the buffer's alloca size.
plawk_writebin_entry_lines(none, '').
plawk_writebin_entry_lines(outfmt(_Types, Size), IR) :-
    format(atom(IR), '  %plawk_wbuf = alloca [~w x i8], align 8~n', [Size]).
plawk_writebin_entry_lines(outfmt_union(_Arms, BufSize), IR) :-
    format(atom(IR),
'  %plawk_wbuf = alloca [~w x i8], align 8
  %plawk_wbuf_p = getelementptr inbounds [~w x i8], [~w x i8]* %plawk_wbuf, i32 0, i32 0~n',
        [BufSize, BufSize, BufSize]).

% fwrite(buf, size, 1, stdout) buffers in libc; the normal return from
% main flushes it.
plawk_writebin_globals(none, '').
plawk_writebin_globals(outfmt(_Types, _Size), '@stdout = external global i8*').
plawk_writebin_globals(outfmt_union(_Arms, _BufSize), '@stdout = external global i8*').

plawk_rules_writebin_exprs(Rules, Exprs) :-
    findall(Expr,
        ( member(rule(_Pattern, Actions), Rules),
          plawk_actions_writebin_expr(Actions, Expr)
        ),
        Exprs).

plawk_actions_writebin_expr(Actions, Expr) :-
    member(Action, Actions),
    ( Action = writebin_out(_Types, Fields),
      member(Expr, Fields)
    ; Action = writebin_arm_out(_Tag, _ArmTypes, AFields),
      member(Expr, AFields)
    ; Action = if(_Pattern, ThenActions, ElseActions),
      ( plawk_actions_writebin_expr(ThenActions, Expr)
      ; plawk_actions_writebin_expr(ElseActions, Expr)
      )
    ).

%% plawk_writebin_args_ok(+Types, +Fields) is semidet.
%
%  i64 output slots take i64-shaped expressions (fields, literals,
%  NR/NF, scalar reads, binary trees); f64 slots additionally take
%  double-typed expressions, with i64 shapes promoted at emission.
plawk_writebin_args_ok([], []).
plawk_writebin_args_ok([Type | Types], [Field | Fields]) :-
    ( Type == i64
    -> plawk_writebin_i64_arg(Field)
    ;  Type = s(Width)
    -> plawk_writebin_str_arg(Field, Width)
    ;  Type = lps(Cap)
    -> plawk_writebin_str_arg(Field, Cap)
    ;  Type = rep(_K, _Elems)
    -> Field = field(_CountIndex)
    ;  plawk_writebin_f64_arg(Field)
    ),
    plawk_writebin_args_ok(Types, Fields).

% OUTFMT slot types: i64/f64/sN/lpsN as before, plus rep passthrough
% (fixed-width elements only -- the element region is copied as one
% bulk write, which requires in-memory layout == wire layout).
plawk_outfmt_type_ok(i64) :- !.
plawk_outfmt_type_ok(f64) :- !.
plawk_outfmt_type_ok(s(_W)) :- !.
plawk_outfmt_type_ok(lps(_C)) :- !.
plawk_outfmt_type_ok(rep(_K, ElemTypes)) :-
    forall(member(ElemType, ElemTypes),
        ( memberchk(ElemType, [i64, f64]) ; ElemType = s(_)
        ; ElemType = lps(_)
        )).

% sN output slots take string literals that fit the width, or field
% reads (validated against the input layout separately in binary mode;
% any field slice in text mode, clamped to the width at emission).
plawk_writebin_str_arg(string(Value), Width) :-
    string_length(Value, Length),
    Length =< Width.
plawk_writebin_str_arg(field(FieldIndex), _Width) :-
    integer(FieldIndex),
    FieldIndex >= 1.
% a runtime grammar's byte output fills an sN/lpsN slot (clamped to the
% slot width/cap at runtime; a failed call writes an empty payload)
plawk_writebin_str_arg(Blob, _Width) :-
    plawk_blob_call_arg_ok(Blob).

%% plawk_blob_node_shape(+Expr) is semidet.
%  Structural test: Expr is one of the three blob call nodes.
plawk_blob_node_shape(blob_dyncall(_)).
plawk_blob_node_shape(blob_dyncall_named(_, _)).
plawk_blob_node_shape(blob_dyncall_at(_, _)).
plawk_blob_node_shape(blob_dyncall_at_named(_, _, _)).

%% plawk_blob_call_arg_ok(+Expr) is semidet.
%  Shape plus argument validation (same checks as the float variants;
%  the arg shapes are identical across the shim families).
plawk_blob_call_arg_ok(blob_dyncall(Args)) :-
    plawk_float_dyncall_expr(float_dyncall(Args)).
plawk_blob_call_arg_ok(blob_dyncall_named(Name, Args)) :-
    plawk_float_dyncall_named_expr(float_dyncall_named(Name, Args)).
plawk_blob_call_arg_ok(blob_dyncall_at(Source, Args)) :-
    plawk_float_dyncall_at_expr(float_dyncall_at(Source, Args)).
plawk_blob_call_arg_ok(blob_dyncall_at_named(Name, Source, Args)) :-
    plawk_float_dyncall_at_expr(float_dyncall_at_named(Name, Source, Args)).

plawk_writebin_i64_arg(special('NR')).
plawk_writebin_i64_arg(special('NF')).
plawk_writebin_i64_arg(Field) :-
    plawk_i64_scalar_read_operand_expr(Field).

plawk_writebin_f64_arg(Field) :-
    plawk_writebin_i64_arg(Field),
    !.
plawk_writebin_f64_arg(Field) :-
    plawk_expr_is_double(Field),
    plawk_f64_scalar_read_operand_expr(Field).

%% plawk_writebin_record_ir(+Types, +Fields, +Slots, +Values,
%%     +FieldSeparator, +Prefix, -Pair)
%
%  Evaluate each argument, store it at its layout offset in the shared
%  %plawk_wbuf buffer, then fwrite the record to stdout.
plawk_writebin_record_ir(Types, Fields, Slots, Values, FieldSeparator,
        Prefix, Pair) :-
    ( plawk_binfmt_has_varlen(Types)
    ->  plawk_writebin_varlen_record_ir(Types, Fields, Slots, Values,
            FieldSeparator, Prefix, Pair)
    ;   plawk_writebin_fixed_record_ir(Types, Fields, Slots, Values,
            FieldSeparator, Prefix, Pair)
    ).

plawk_writebin_fixed_record_ir(Types, Fields, Slots, Values, FieldSeparator,
        Prefix, GlobalIR-IR) :-
    plawk_binfmt_record_size(binfmt(Types), Size),
    format(atom(BasePtr), '%~w_base', [Prefix]),
    format(atom(BaseLine),
        '  ~w = getelementptr inbounds [~w x i8], [~w x i8]* %plawk_wbuf, i32 0, i32 0',
        [BasePtr, Size, Size]),
    plawk_writebin_field_lines(Types, Fields, Slots, Values, FieldSeparator,
        Prefix, BasePtr, 0, 0, FieldLines, GlobalParts),
    format(atom(StdoutLoad),
        '  %~w_stdout = load i8*, i8** @stdout', [Prefix]),
    format(atom(WriteCall),
        '  %~w_wr = call i64 @fwrite(i8* ~w, i64 ~w, i64 1, i8* %~w_stdout)',
        [Prefix, BasePtr, Size, Prefix]),
    append([[BaseLine], FieldLines, [StdoutLoad, WriteCall]], AllLines),
    atomic_list_concat(AllLines, '\n', IR),
    atomic_list_concat(GlobalParts, '\n', GlobalIR).

%% plawk_writebin_varlen_record_ir(+Types, +Fields, +Slots, +Values,
%%     +FieldSeparator, +Prefix, -Pair)
%
%  Records whose OUTFMT contains an lps slot are variable-length on the
%  wire, so the single-buffer fwrite becomes per-slot fwrites emitted
%  strictly left to right (fwrite buffers in libc, so this is memcpy
%  cost, not syscall cost). Numeric slots stage their value in the
%  first 8 bytes of %plawk_wbuf; lps slots write their runtime length
%  the same way, then the payload straight from its source bytes.
plawk_writebin_varlen_record_ir(Types, Fields, Slots, Values, FieldSeparator,
        Prefix, GlobalIR-IR) :-
    plawk_binfmt_record_size(binfmt(Types), Size),
    format(atom(BasePtr), '%~w_base', [Prefix]),
    format(atom(BaseLine),
        '  ~w = getelementptr inbounds [~w x i8], [~w x i8]* %plawk_wbuf, i32 0, i32 0',
        [BasePtr, Size, Size]),
    format(atom(StdoutLoad),
        '  %~w_stdout = load i8*, i8** @stdout', [Prefix]),
    format(atom(StdoutVar), '%~w_stdout', [Prefix]),
    plawk_writebin_varlen_field_lines(Types, Fields, Slots, Values,
        FieldSeparator, Prefix, BasePtr, StdoutVar, 0, FieldLines, GlobalParts),
    append([[BaseLine, StdoutLoad], FieldLines], AllLines),
    atomic_list_concat(AllLines, '\n', IR),
    atomic_list_concat(GlobalParts, '\n', GlobalIR).

%% plawk_writebin_union_record_ir(+Tag, +ArmTypes, +Fields, +Slots,
%%     +Values, +FieldSeparator, +Prefix, -Pair)
%
%  One tagged output record: the 8-byte arm tag, then the arm's slots
%  through the per-slot varlen writer. The shared buffer pointer
%  (%plawk_wbuf_p) was resolved once in the entry block by the
%  outfmt_union plan, so sites need no knowledge of the buffer size.
plawk_writebin_union_record_ir(Tag, ArmTypes, Fields, Slots, Values,
        FieldSeparator, Prefix, GlobalIR-IR) :-
    format(atom(StdoutLoad),
        '  %~w_stdout = load i8*, i8** @stdout', [Prefix]),
    format(atom(StdoutVar), '%~w_stdout', [Prefix]),
    format(atom(TagIR),
'  %~w_tsp = bitcast i8* %plawk_wbuf_p to i64*
  store i64 ~w, i64* %~w_tsp, align 1
  %~w_twr = call i64 @fwrite(i8* %plawk_wbuf_p, i64 8, i64 1, i8* ~w)',
        [Prefix, Tag, Prefix, Prefix, StdoutVar]),
    plawk_writebin_varlen_field_lines(ArmTypes, Fields, Slots, Values,
        FieldSeparator, Prefix, '%plawk_wbuf_p', StdoutVar, 0, FieldLines,
        GlobalParts),
    append([[StdoutLoad, TagIR], FieldLines], AllLines),
    atomic_list_concat(AllLines, '\n', IR),
    atomic_list_concat(GlobalParts, '\n', GlobalIR).

plawk_writebin_varlen_field_lines([], [], _Slots, _Values, _FieldSeparator,
        _Prefix, _BasePtr, _Stdout, _Index, [], []).
plawk_writebin_varlen_field_lines([Type | Types], [Field0 | Fields], Slots,
        Values, FieldSeparator, Prefix, BasePtr, Stdout, Index, Lines,
        GlobalParts) :-
    plawk_substitute_scalar_reads(Field0, Slots, Values, Field),
    format(atom(Base), '~w_f~w', [Prefix, Index]),
    plawk_writebin_varlen_slot_lines(Type, Field, FieldSeparator, Base,
        BasePtr, Stdout, SlotLines, GParts),
    NextIndex is Index + 1,
    plawk_writebin_varlen_field_lines(Types, Fields, Slots, Values,
        FieldSeparator, Prefix, BasePtr, Stdout, NextIndex, RestLines,
        RestGlobals),
    append(SlotLines, RestLines, Lines),
    append(GParts, RestGlobals, GlobalParts).

plawk_writebin_varlen_slot_lines(lps(Cap), Field, FieldSeparator, Base,
        BasePtr, Stdout, Lines, GParts) :-
    !,
    plawk_writebin_lps_source_lines(Field, FieldSeparator, Cap, Base,
        BasePtr, PtrIR, LenIR, SourceLines, GParts),
    format(atom(LenStore),
'  %~w_lensp = bitcast i8* ~w to i64*
  store i64 ~w, i64* %~w_lensp, align 1
  %~w_lwr = call i64 @fwrite(i8* ~w, i64 8, i64 1, i8* ~w)',
        [Base, BasePtr, LenIR, Base, Base, BasePtr, Stdout]),
    format(atom(PayloadWrite),
        '  %~w_pwr = call i64 @fwrite(i8* ~w, i64 ~w, i64 1, i8* ~w)',
        [Base, PtrIR, LenIR, Stdout]),
    append(SourceLines, [LenStore, PayloadWrite], Lines).
% rep passthrough whose elements contain lps strings: each element's
% wire size varies (a length prefix plus the live payload, recovered
% with strnlen from the NUL-padded slot), so the writer loops over the
% live elements, emitting each field left to right. The loop gets its
% own labeled blocks; the fragment enters through a br so the head phi
% has a named predecessor.
plawk_writebin_varlen_slot_lines(rep(_Cap, ElemTypes), field(CountIndex),
        binfmt(InTypes), Base, BasePtr, Stdout, Lines, []) :-
    memberchk(lps(_ECap), ElemTypes),
    !,
    maplist(plawk_binfmt_type_width, ElemTypes, ElemWidths),
    sum_list(ElemWidths, ElemSize),
    plawk_binfmt_field_offset(binfmt(InTypes), CountIndex, CountOff),
    ElemsOff is CountOff + 8,
    format(atom(HeaderIR),
'  br label %~w_pre

~w_pre:
  %~w_cfp = getelementptr i8, i8* %rec, i64 ~w
  %~w_ctp = bitcast i8* %~w_cfp to i64*
  %~w_n = load i64, i64* %~w_ctp, align 1
  %~w_csp = bitcast i8* ~w to i64*
  store i64 %~w_n, i64* %~w_csp, align 1
  %~w_cwr = call i64 @fwrite(i8* ~w, i64 8, i64 1, i8* ~w)
  br label %~w_lh

~w_lh:
  %~w_j = phi i64 [ 1, %~w_pre ], [ %~w_jn, %~w_ld ]
  %~w_cont = icmp sle i64 %~w_j, %~w_n
  br i1 %~w_cont, label %~w_lb, label %~w_after

~w_lb:
  %~w_jm1 = sub i64 %~w_j, 1
  %~w_rel = mul i64 %~w_jm1, ~w
  %~w_efp = getelementptr i8, i8* %rec, i64 ~w
  %~w_eb = getelementptr i8, i8* %~w_efp, i64 %~w_rel',
        [Base,
         Base,
         Base, CountOff,
         Base, Base,
         Base, Base,
         Base, BasePtr,
         Base, Base,
         Base, BasePtr, Stdout,
         Base,
         Base,
         Base, Base, Base, Base,
         Base, Base, Base,
         Base, Base, Base,
         Base,
         Base, Base,
         Base, Base, ElemSize,
         Base, ElemsOff,
         Base, Base, Base]),
    format(atom(ElemBase), '%~w_eb', [Base]),
    plawk_writebin_rep_elem_lines(ElemTypes, Base, ElemBase, BasePtr, Stdout,
        0, 0, ElemLines),
    format(atom(FooterIR),
'  br label %~w_ld

~w_ld:
  %~w_jn = add i64 %~w_j, 1
  br label %~w_lh

~w_after:',
        [Base, Base, Base, Base, Base, Base]),
    append([[HeaderIR], ElemLines, [FooterIR]], Lines).
plawk_writebin_varlen_slot_lines(rep(_Cap, ElemTypes), field(CountIndex),
        binfmt(InTypes), Base, BasePtr, Stdout, Lines, []) :-
    !,
    % rep passthrough: write the live count, then the live elements as
    % one bulk write straight from the input record's element region
    % (fixed-width elements, so in-memory layout == wire layout; a
    % zero count writes nothing -- fwrite with size 0 is a no-op).
    maplist(plawk_binfmt_type_width, ElemTypes, ElemWidths),
    sum_list(ElemWidths, ElemSize),
    plawk_binfmt_field_offset(binfmt(InTypes), CountIndex, CountOff),
    ElemsOff is CountOff + 8,
    format(atom(RepIR),
'  %~w_cfp = getelementptr i8, i8* %rec, i64 ~w
  %~w_ctp = bitcast i8* %~w_cfp to i64*
  %~w_n = load i64, i64* %~w_ctp, align 1
  %~w_csp = bitcast i8* ~w to i64*
  store i64 %~w_n, i64* %~w_csp, align 1
  %~w_cwr = call i64 @fwrite(i8* ~w, i64 8, i64 1, i8* ~w)
  %~w_efp = getelementptr i8, i8* %rec, i64 ~w
  %~w_bytes = mul i64 %~w_n, ~w
  %~w_ewr = call i64 @fwrite(i8* %~w_efp, i64 %~w_bytes, i64 1, i8* ~w)',
        [Base, CountOff,
         Base, Base,
         Base, Base,
         Base, BasePtr,
         Base, Base,
         Base, BasePtr, Stdout,
         Base, ElemsOff,
         Base, Base, ElemSize,
         Base, Base, Base, Stdout]),
    Lines = [RepIR].
plawk_writebin_varlen_slot_lines(Type, Field, FieldSeparator, Base, BasePtr,
        Stdout, Lines, GParts) :-
    % numeric slot: stage the value in the scratch buffer, write 8 bytes
    ( Type == i64
    ->  plawk_i64_expr_ir(Field, FieldSeparator, Base, Base, ValueIR,
            GParts, SetupParts),
        LLVMType = i64
    ;   plawk_writebin_f64_value_ir(Field, FieldSeparator, Base, ValueIR,
            GParts, SetupParts),
        LLVMType = double
    ),
    format(atom(StoreWrite),
'  %~w_sp = bitcast i8* ~w to ~w*
  store ~w ~w, ~w* %~w_sp, align 1
  %~w_wr = call i64 @fwrite(i8* ~w, i64 8, i64 1, i8* ~w)',
        [Base, BasePtr, LLVMType, LLVMType, ValueIR, LLVMType, Base,
         Base, BasePtr, Stdout]),
    append(SetupParts, [StoreWrite], Lines).

%% plawk_writebin_rep_elem_lines(+ElemTypes, +Base, +ElemBase, +BasePtr,
%%     +Stdout, +Index, +Offset, -Lines)
%
%  Loop-body field writes for one rep element based at ElemBase.
%  Fixed-width fields pass through directly; lps fields recover the
%  live length from the NUL-padded slot with strnlen, then emit the
%  8-byte prefix and exactly that many payload bytes.
plawk_writebin_rep_elem_lines([], _Base, _ElemBase, _BasePtr, _Stdout,
        _Index, _Offset, []).
plawk_writebin_rep_elem_lines([lps(Cap) | Types], Base, ElemBase, BasePtr,
        Stdout, Index, Offset, [IR | Lines]) :-
    !,
    format(atom(IR),
'  %~w_e~w_sp = getelementptr i8, i8* ~w, i64 ~w
  %~w_e~w_len = call i64 @strnlen(i8* %~w_e~w_sp, i64 ~w)
  %~w_e~w_lsp = bitcast i8* ~w to i64*
  store i64 %~w_e~w_len, i64* %~w_e~w_lsp, align 1
  %~w_e~w_lwr = call i64 @fwrite(i8* ~w, i64 8, i64 1, i8* ~w)
  %~w_e~w_pwr = call i64 @fwrite(i8* %~w_e~w_sp, i64 %~w_e~w_len, i64 1, i8* ~w)',
        [Base, Index, ElemBase, Offset,
         Base, Index, Base, Index, Cap,
         Base, Index, BasePtr,
         Base, Index, Base, Index,
         Base, Index, BasePtr, Stdout,
         Base, Index, Base, Index, Base, Index, Stdout]),
    plawk_binfmt_type_width(lps(Cap), Width),
    NextOffset is Offset + Width,
    NextIndex is Index + 1,
    plawk_writebin_rep_elem_lines(Types, Base, ElemBase, BasePtr, Stdout,
        NextIndex, NextOffset, Lines).
plawk_writebin_rep_elem_lines([Type | Types], Base, ElemBase, BasePtr,
        Stdout, Index, Offset, [IR | Lines]) :-
    plawk_binfmt_type_width(Type, Width),
    format(atom(IR),
'  %~w_e~w_sp = getelementptr i8, i8* ~w, i64 ~w
  %~w_e~w_wr = call i64 @fwrite(i8* %~w_e~w_sp, i64 ~w, i64 1, i8* ~w)',
        [Base, Index, ElemBase, Offset,
         Base, Index, Base, Index, Width, Stdout]),
    NextOffset is Offset + Width,
    NextIndex is Index + 1,
    plawk_writebin_rep_elem_lines(Types, Base, ElemBase, BasePtr, Stdout,
        NextIndex, NextOffset, Lines).

%% plawk_writebin_lps_source_lines(+Field, +FieldSeparator, +Cap, +Base,
%%     +BasePtr, -PtrIR, -LenIR, -Lines, -GlobalParts)
%
%  Resolve an lps payload source to a (pointer, runtime length) pair.
%  lps payloads are string-valued: source bytes run to the first NUL or
%  the source bound, clamped to the slot cap.
plawk_writebin_lps_source_lines(string(Value), _FieldSeparator, Cap, Base,
        _BasePtr, PtrIR, StringLen, [PtrLine], [GlobalLine]) :-
    !,
    format(atom(LitGlobal), '~w_lit', [Base]),
    llvm_emit_c_string_global(LitGlobal, Value, GlobalLine, StringLen, BytesLen),
    StringLen =< Cap,
    format(atom(PtrIR), '%~w_src', [Base]),
    format(atom(PtrLine),
        '  ~w = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0',
        [PtrIR, BytesLen, BytesLen, LitGlobal]).
plawk_writebin_lps_source_lines(field(FieldIndex), binfmt(Types), Cap, Base,
        _BasePtr, PtrIR, LenIR, [PtrLine, LenLine], []) :-
    !,
    plawk_binfmt_field_type(binfmt(Types), FieldIndex, s(SourceWidth)),
    SourceWidth =< Cap,
    plawk_binfmt_field_offset(binfmt(Types), FieldIndex, SourceOffset),
    format(atom(PtrIR), '%~w_src', [Base]),
    format(atom(LenIR), '%~w_len', [Base]),
    format(atom(PtrLine),
        '  ~w = getelementptr i8, i8* %rec, i64 ~w', [PtrIR, SourceOffset]),
    format(atom(LenLine),
        '  ~w = call i64 @strnlen(i8* ~w, i64 ~w)',
        [LenIR, PtrIR, SourceWidth]).
plawk_writebin_lps_source_lines(Blob, FieldSeparator, Cap, Base,
        BasePtr, PtrIR, LenIR, Lines, GParts) :-
    plawk_blob_node_shape(Blob),
    !,
    % A runtime grammar's byte output as the lps payload: the blob tail
    % leaves %Base_ptr (null on failure) + %Base_len64; clamp to the
    % cap and substitute a safe pointer/zero length on failure.
    plawk_blob_expr_ir(Blob, FieldSeparator, Base, _L, _P, GParts, SetupParts),
    format(atom(PtrIR), '%~w_srcp', [Base]),
    format(atom(LenIR), '%~w_srclen', [Base]),
    format(atom(ClampIR),
'  %~w_bnull = icmp eq i8* %~w_ptr, null
  %~w_blen0 = select i1 %~w_bnull, i64 0, i64 %~w_len64
  %~w_bover = icmp ugt i64 %~w_blen0, ~w
  ~w = select i1 %~w_bover, i64 ~w, i64 %~w_blen0
  ~w = select i1 %~w_bnull, i8* ~w, i8* %~w_ptr',
        [Base, Base,
         Base, Base, Base,
         Base, Base, Cap,
         LenIR, Base, Cap, Base,
         PtrIR, Base, BasePtr, Base]),
    append(SetupParts, [ClampIR], Lines).
plawk_writebin_lps_source_lines(field(FieldIndex), FieldSeparator, Cap, Base,
        BasePtr, PtrIR, LenIR, [SliceIR, ClampIR], []) :-
    FieldIndex >= 1,
    llvm_emit_atom_field_slice('%line', FieldIndex, FieldSeparator, Base,
        SliceIR),
    format(atom(PtrIR), '%~w_srcp', [Base]),
    format(atom(LenIR), '%~w_srclen', [Base]),
    % a missing field (null slice) writes length 0 with a safe pointer
    format(atom(ClampIR),
'  %~w_null = icmp eq i8* %~w_ptr, null
  %~w_len0 = select i1 %~w_null, i64 0, i64 %~w_len64
  %~w_over = icmp ugt i64 %~w_len0, ~w
  ~w = select i1 %~w_over, i64 ~w, i64 %~w_len0
  ~w = select i1 %~w_null, i8* ~w, i8* %~w_ptr',
        [Base, Base,
         Base, Base, Base,
         Base, Base, Cap,
         LenIR, Base, Cap, Base,
         PtrIR, Base, BasePtr, Base]).

plawk_writebin_field_lines([], [], _Slots, _Values, _FieldSeparator, _Prefix,
        _BasePtr, _Index, _Offset, [], []).
plawk_writebin_field_lines([Type | Types], [Field0 | Fields], Slots, Values,
        FieldSeparator, Prefix, BasePtr, Index, Offset, Lines, GlobalParts) :-
    plawk_substitute_scalar_reads(Field0, Slots, Values, Field),
    format(atom(Base), '~w_f~w', [Prefix, Index]),
    plawk_writebin_slot_lines(Type, Field, FieldSeparator, Base, BasePtr,
        Offset, SlotLines, GParts),
    plawk_binfmt_type_width(Type, Width),
    NextOffset is Offset + Width,
    NextIndex is Index + 1,
    plawk_writebin_field_lines(Types, Fields, Slots, Values, FieldSeparator,
        Prefix, BasePtr, NextIndex, NextOffset, RestLines, RestGlobals),
    append(SlotLines, RestLines, Lines),
    append(GParts, RestGlobals, GlobalParts).

%% plawk_writebin_slot_lines(+Type, +Field, +FieldSeparator, +Base,
%%     +BasePtr, +Offset, -Lines, -GlobalParts)
%
%  Store one writebin argument at its layout offset. Numeric slots are
%  typed stores; sN slots memset the slot to zero and memcpy the source
%  bytes (a literal's global, an sM binary input field, or a text-mode
%  field slice clamped to the slot width).
plawk_writebin_slot_lines(s(Width), string(Value), _FieldSeparator, Base,
        BasePtr, Offset, Lines, [GlobalLine]) :-
    !,
    format(atom(LitGlobal), '~w_lit', [Base]),
    llvm_emit_c_string_global(LitGlobal, Value, GlobalLine, StringLen, BytesLen),
    StringLen =< Width,
    format(atom(Dst),
        '  %~w_dst = getelementptr i8, i8* ~w, i64 ~w', [Base, BasePtr, Offset]),
    format(atom(Src),
        '  %~w_src = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0',
        [Base, BytesLen, BytesLen, LitGlobal]),
    format(atom(Zero),
        '  call void @llvm.memset.p0i8.i64(i8* %~w_dst, i8 0, i64 ~w, i1 false)',
        [Base, Width]),
    format(atom(Copy),
        '  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %~w_dst, i8* %~w_src, i64 ~w, i1 false)',
        [Base, Base, StringLen]),
    Lines = [Dst, Src, Zero, Copy].
plawk_writebin_slot_lines(s(Width), field(FieldIndex), binfmt(Types), Base,
        BasePtr, Offset, Lines, []) :-
    !,
    plawk_binfmt_field_type(binfmt(Types), FieldIndex, s(SourceWidth)),
    SourceWidth =< Width,
    plawk_binfmt_field_offset(binfmt(Types), FieldIndex, SourceOffset),
    format(atom(Dst),
        '  %~w_dst = getelementptr i8, i8* ~w, i64 ~w', [Base, BasePtr, Offset]),
    format(atom(Src),
        '  %~w_src = getelementptr i8, i8* %rec, i64 ~w', [Base, SourceOffset]),
    ( SourceWidth < Width
    ->  format(atom(Zero),
            '  call void @llvm.memset.p0i8.i64(i8* %~w_dst, i8 0, i64 ~w, i1 false)',
            [Base, Width]),
        ZeroLines = [Zero]
    ;   ZeroLines = []
    ),
    format(atom(Copy),
        '  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %~w_dst, i8* %~w_src, i64 ~w, i1 false)',
        [Base, Base, SourceWidth]),
    append([[Dst, Src], ZeroLines, [Copy]], Lines).
plawk_writebin_slot_lines(s(Width), Blob, FieldSeparator, Base,
        BasePtr, Offset, Lines, GParts) :-
    plawk_blob_node_shape(Blob),
    !,
    % A runtime grammar's byte output: evaluate the blob (the tail IR
    % leaves %Base_ptr -- null on failure -- and %Base_len64), then
    % zero-fill and clamped-copy exactly like a text field slice.
    plawk_blob_expr_ir(Blob, FieldSeparator, Base, _LenIR, _PtrIR,
        GParts, SetupParts),
    format(atom(Dst),
        '  %~w_dst = getelementptr i8, i8* ~w, i64 ~w', [Base, BasePtr, Offset]),
    format(atom(Zero),
        '  call void @llvm.memset.p0i8.i64(i8* %~w_dst, i8 0, i64 ~w, i1 false)',
        [Base, Width]),
    format(atom(Guard),
'  %~w_bnull = icmp eq i8* %~w_ptr, null
  %~w_bover = icmp ugt i64 %~w_len64, ~w
  %~w_bcap = select i1 %~w_bover, i64 ~w, i64 %~w_len64
  br i1 %~w_bnull, label %~w_bdone, label %~w_bcopy

~w_bcopy:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %~w_dst, i8* %~w_ptr, i64 %~w_bcap, i1 false)
  br label %~w_bdone

~w_bdone:',
        [Base, Base,
         Base, Base, Width,
         Base, Base, Width, Base,
         Base, Base, Base,
         Base, Base, Base, Base, Base,
         Base]),
    append(SetupParts, [Dst, Zero, Guard], Lines).
plawk_writebin_slot_lines(s(Width), field(FieldIndex), FieldSeparator, Base,
        BasePtr, Offset, Lines, []) :-
    !,
    % Text mode: slice the field, clamp to the slot width, zero-fill,
    % and copy behind a null guard (a missing field writes all zeros).
    FieldIndex >= 1,
    llvm_emit_atom_field_slice('%line', FieldIndex, FieldSeparator, Base,
        SliceIR),
    format(atom(Dst),
        '  %~w_dst = getelementptr i8, i8* ~w, i64 ~w', [Base, BasePtr, Offset]),
    format(atom(Zero),
        '  call void @llvm.memset.p0i8.i64(i8* %~w_dst, i8 0, i64 ~w, i1 false)',
        [Base, Width]),
    format(atom(Guard),
'  %~w_null = icmp eq i8* %~w_ptr, null
  %~w_over = icmp ugt i64 %~w_len64, ~w
  %~w_cap = select i1 %~w_over, i64 ~w, i64 %~w_len64
  br i1 %~w_null, label %~w_done, label %~w_copy

~w_copy:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %~w_dst, i8* %~w_ptr, i64 %~w_cap, i1 false)
  br label %~w_done

~w_done:',
        [Base, Base,
         Base, Base, Width,
         Base, Base, Width, Base,
         Base, Base, Base,
         Base, Base, Base, Base, Base,
         Base]),
    Lines = [SliceIR, Dst, Zero, Guard].
plawk_writebin_slot_lines(i64, Field, FieldSeparator, Base, BasePtr, Offset,
        Lines, GParts) :-
    !,
    plawk_i64_expr_ir(Field, FieldSeparator, Base, Base, ValueIR,
        GParts, SetupParts),
    plawk_writebin_numeric_store_lines(i64, ValueIR, Base, BasePtr, Offset,
        StoreLines),
    append(SetupParts, StoreLines, Lines).
plawk_writebin_slot_lines(f64, Field, FieldSeparator, Base, BasePtr, Offset,
        Lines, GParts) :-
    plawk_writebin_f64_value_ir(Field, FieldSeparator, Base, ValueIR,
        GParts, SetupParts),
    plawk_writebin_numeric_store_lines(double, ValueIR, Base, BasePtr, Offset,
        StoreLines),
    append(SetupParts, StoreLines, Lines).

plawk_writebin_numeric_store_lines(LLVMType, ValueIR, Base, BasePtr, Offset,
        [Gep, Cast, Store]) :-
    % _sp/_stp, not _fp/_tp: a bare $N argument's binfmt field LOAD
    % already claims Base_fp/Base_tp for the read from %rec.
    format(atom(Gep),
        '  %~w_sp = getelementptr i8, i8* ~w, i64 ~w', [Base, BasePtr, Offset]),
    format(atom(Cast),
        '  %~w_stp = bitcast i8* %~w_sp to ~w*', [Base, Base, LLVMType]),
    format(atom(Store),
        '  store ~w ~w, ~w* %~w_stp, align 1',
        [LLVMType, ValueIR, LLVMType, Base]).

plawk_writebin_f64_value_ir(Field, FieldSeparator, Base, ValueIR,
        GParts, SetupParts) :-
    plawk_expr_is_double(Field),
    !,
    plawk_f64_expr_ir(Field, FieldSeparator, Base, Base, ValueIR,
        GParts, SetupParts).
plawk_writebin_f64_value_ir(Field, FieldSeparator, Base, ValueIR,
        GParts, SetupParts) :-
    format(atom(IntBase), '~w_int', [Base]),
    plawk_i64_expr_ir(Field, FieldSeparator, IntBase, IntBase, IntValueIR,
        GParts, IntSetupParts),
    format(atom(ValueIR), '%~w', [Base]),
    format(atom(Promote), '  ~w = sitofp i64 ~w to double',
        [ValueIR, IntValueIR]),
    append(IntSetupParts, [Promote], SetupParts).

%% plawk_emit_record_driver_ir(+Descriptor, +InputPath, +Blocks, -DriverIR)
%
%  Pick the stream skeleton by record representation: text lines or
%  fixed-size binary records.
%% plawk_rule_descriptor(+Pattern, +Default, -Descriptor)
%
%  Rules inside a case block see their arm's field layout; everything
%  else uses the program-wide descriptor.
plawk_rule_descriptor(arm_pat(_Tag, ArmTypes, _Pattern), _Default,
        binfmt(ArmTypes)) :-
    !.
plawk_rule_descriptor(_Pattern, Default, Default).

%% plawk_resolve_foreach_rules(+Descriptor, +Rules0, -Rules)
%
%  foreach { actions } iterates the record's repetition elements:
%  inside the block, $1..$M are the CURRENT ELEMENT's fields. Because
%  the element count is capped at compile time, the block unrolls into
%  Cap guarded ifs -- if (count >= j) { actions with fields shifted to
%  element j's flat slots } -- reusing the existing if machinery, so no
%  new loop/phi mechanics exist at the IR level. Requires a binfmt
%  layout with exactly one rep field; fails (rejecting the program)
%  otherwise, on nested foreach, or on $F beyond the element arity.
plawk_resolve_foreach_rules(Descriptor, Rules0, Rules) :-
    ( plawk_rules_have_foreach(Rules0)
    ->  Descriptor = binfmt(Types),
        plawk_binfmt_rep_info(Types, CountIndex, _Cap, ElemArity),
        plawk_binfmt_foreach_layout(Types, CountIndex, ElemArity, Layout),
        maplist(plawk_resolve_foreach_rule(Layout, ElemArity), Rules0, Rules)
    ;   Rules = Rules0
    ).

%% plawk_binfmt_foreach_layout(+Types, +CountIndex, +ElemArity, -Layout)
%
%  Byte/field geometry the runtime loop needs: the count field's byte
%  offset, the first element's byte offset, the element size, the
%  staging area's byte offset (end of the declared layout), and the
%  access index of the first staging field minus one (the shift base
%  for body field references).
plawk_binfmt_foreach_layout(Types, CountIndex, _ElemArity,
        foreach_layout(CountOff, ElemsOff, ElemSize, StageOff, StageBase)) :-
    plawk_binfmt_field_offset(binfmt(Types), CountIndex, CountOff),
    ElemsOff is CountOff + 8,
    memberchk(rep(_Cap, ElemTypes), Types),
    maplist(plawk_binfmt_type_width, ElemTypes, ElemWidths),
    sum_list(ElemWidths, ElemSize),
    plawk_binfmt_declared_access_types(Types, Declared),
    length(Declared, StageBase),
    maplist(plawk_binfmt_type_width, Declared, DeclaredWidths),
    sum_list(DeclaredWidths, StageOff).

plawk_rules_have_foreach(Rules) :-
    member(rule(_Pattern, Actions), Rules),
    plawk_actions_have_foreach(Actions),
    !.

plawk_actions_have_foreach(Actions) :-
    member(Action, Actions),
    ( Action = foreach(_Body)
    ; Action = if(_Pattern, ThenActions, ElseActions),
      ( plawk_actions_have_foreach(ThenActions)
      ; plawk_actions_have_foreach(ElseActions)
      )
    ),
    !.

plawk_binfmt_rep_info(Types, CountIndex, Cap, ElemArity) :-
    findall(StoredIndex-rep(K, Elems),
        nth1(StoredIndex, Types, rep(K, Elems)),
        [RepStoredIndex-rep(Cap, ElemTypes)]),
    length(ElemTypes, ElemArity),
    PrefixLen is RepStoredIndex - 1,
    length(PrefixTypes, PrefixLen),
    append(PrefixTypes, _, Types),
    plawk_binfmt_access_types(PrefixTypes, PrefixAccess),
    length(PrefixAccess, PrefixCount),
    CountIndex is PrefixCount + 1.

plawk_resolve_foreach_rule(Layout, ElemArity,
        rule(Pattern, Actions0), rule(Pattern, Actions)) :-
    plawk_resolve_foreach_actions(Layout, ElemArity, Actions0, Actions).

% foreach { Body } becomes one foreach_loop/2 term: a RUNTIME loop over
% the elements (code size O(body), any cap), whose body reads the
% current element from the fixed staging area -- field references are
% shifted once to the staging indexes, so every existing field emitter
% works unchanged.
plawk_resolve_foreach_actions(_Layout, _ElemArity, [], []).
plawk_resolve_foreach_actions(Layout, ElemArity,
        [foreach(Body) | Rest0], [foreach_loop(Layout, ShiftedBody) | Rest]) :-
    !,
    \+ plawk_actions_have_foreach(Body),
    plawk_foreach_body_fields_ok(Body, ElemArity),
    Layout = foreach_layout(_CountOff, _ElemsOff, _ElemSize, _StageOff, StageBase),
    plawk_foreach_shift_actions(Body, StageBase, ShiftedBody),
    plawk_resolve_foreach_actions(Layout, ElemArity, Rest0, Rest).
plawk_resolve_foreach_actions(Layout, ElemArity,
        [if(Pattern, Then0, Else0) | Rest0], [if(Pattern, Then, Else) | Rest]) :-
    !,
    plawk_resolve_foreach_actions(Layout, ElemArity, Then0, Then),
    plawk_resolve_foreach_actions(Layout, ElemArity, Else0, Else),
    plawk_resolve_foreach_actions(Layout, ElemArity, Rest0, Rest).
plawk_resolve_foreach_actions(Layout, ElemArity,
        [Action | Rest0], [Action | Rest]) :-
    plawk_resolve_foreach_actions(Layout, ElemArity, Rest0, Rest).

plawk_foreach_body_fields_ok(Body, ElemArity) :-
    forall(( sub_term(Sub, Body),
             plawk_foreach_field_ref(Sub, F),
             integer(F)
           ),
        ( F >= 1, F =< ElemArity )).

% every AST shape that carries a field index, whether wrapped in
% field/1 or stored raw (guard patterns)
plawk_foreach_field_ref(field(F), F).
plawk_foreach_field_ref(float_field(F), F).
plawk_foreach_field_ref(field_cmp(F, _Op, _Value), F).
plawk_foreach_field_ref(field_eq(F, _Value), F).
plawk_foreach_field_ref(field_match(F, _Regex), F).

plawk_foreach_shift_actions(Actions, Base, Shifted) :-
    maplist(plawk_foreach_shift_term(Base), Actions, Shifted).

plawk_foreach_shift_term(Base, field(F), field(Shifted)) :-
    integer(F),
    F >= 1,
    !,
    Shifted is Base + F.
plawk_foreach_shift_term(Base, float_field(F), float_field(Shifted)) :-
    integer(F),
    F >= 1,
    !,
    Shifted is Base + F.
% guard patterns carry the field index raw, so the generic walk would
% miss it (and must not touch the comparison value)
plawk_foreach_shift_term(Base, field_cmp(F, Op, Value), field_cmp(Shifted, Op, Value)) :-
    integer(F),
    F >= 1,
    !,
    Shifted is Base + F.
plawk_foreach_shift_term(Base, field_eq(F, Value), field_eq(Shifted, Value)) :-
    integer(F),
    F >= 1,
    !,
    Shifted is Base + F.
plawk_foreach_shift_term(Base, field_match(F, Regex), field_match(Shifted, Regex)) :-
    integer(F),
    F >= 1,
    !,
    Shifted is Base + F.
plawk_foreach_shift_term(Base, Term, Shifted) :-
    compound(Term),
    !,
    Term =.. [Functor | Args],
    maplist(plawk_foreach_shift_term(Base), Args, ShiftedArgs),
    Shifted =.. [Functor | ShiftedArgs].
plawk_foreach_shift_term(_Base, Term, Term).

%% plawk_tag_rules_case_blocks(+Rules, -CaseBlocks)
%
%  Desugar TAG-guarded rules into case blocks, one single-rule arm
%  block per rule (duplicate arm indexes are fine: flattening walks the
%  blocks in order, so source rule order is preserved). Fails -- and
%  the program is rejected -- when any rule lacks a leading tag guard
%  or mentions TAG anywhere else in its pattern.
plawk_tag_rules_case_blocks(Rules, CaseBlocks) :-
    Rules = [_ | _],
    maplist(plawk_tag_rule_case_arm, Rules, CaseBlocks).

plawk_tag_rule_case_arm(rule(Pattern0, Actions),
        case_arm(Tag, [rule(Pattern, Actions)])) :-
    plawk_pattern_strip_tag(Pattern0, Tag, Pattern),
    \+ ( sub_term(Sub, Pattern), nonvar(Sub), Sub = tag_pat(_) ).

% The tag test must be the leftmost conjunct of a left-associated &&
% chain: TAG == K alone selects the arm (residual guard `always`), and
% TAG == K && P keeps P as the rule's own guard. A tag test under ||,
% !, or anywhere deeper has no single-arm meaning and is rejected.
plawk_pattern_strip_tag(tag_pat(Tag), Tag, always).
plawk_pattern_strip_tag(and_pat(tag_pat(Tag), Residual), Tag, Residual) :-
    !.
plawk_pattern_strip_tag(and_pat(Left0, Right), Tag, and_pat(Left, Right)) :-
    plawk_pattern_strip_tag(Left0, Tag, Left).

%% plawk_union_flatten_rules(+CaseBlocks, +Arms, -Rules)
%
%  Flatten case blocks into one rule chain, stamping each rule with
%  arm_pat(Tag, ArmTypes, Pattern) so guards check the record tag and
%  everything downstream types fields against the right arm. foreach
%  resolves per arm against that arm's own layout (each arm may carry
%  its own rep), so a case block iterates its arm's elements.
plawk_union_flatten_rules(CaseBlocks, Arms, Rules) :-
    findall(rule(arm_pat(Index, ArmTypes, Pattern), Actions),
        ( member(case_arm(Index, ArmRules), CaseBlocks),
          nth0(Index, Arms, ArmTypes),
          plawk_resolve_foreach_rules(binfmt(ArmTypes), ArmRules,
              ResolvedRules),
          member(rule(Pattern, Actions), ResolvedRules)
        ),
        Rules),
    Rules \== [].

plawk_union_program_ok(Arms, CaseBlocks) :-
    length(Arms, ArmCount),
    forall(member(case_arm(Index, ArmRules), CaseBlocks),
        ( integer(Index),
          Index >= 0,
          Index < ArmCount,
          nth0(Index, Arms, ArmTypes),
          plawk_resolve_foreach_rules(binfmt(ArmTypes), ArmRules,
              ResolvedRules),
          forall(member(rule(Pattern, Actions), ResolvedRules),
              ( plawk_binfmt_pattern_ok(binfmt(ArmTypes), Pattern),
                plawk_binfmt_actions_ok(binfmt(ArmTypes), Actions)
              ))
        )).

%% plawk_union_read_ir(+Arms, +LoweredLabel, -ReadIR)
%
%  Read the 8-byte tag (the only place clean EOF is legal), switch on
%  it, and run the selected arm's field-by-field read sequence. Every
%  arm materializes its fields at offset 0 of %rec (the tag itself
%  lives only in %vr_tag), so each arm's access layout is a plain
%  binfmt(ArmTypes). An unknown tag is malformed input -> fail_read.
plawk_union_read_ir(Arms, LoweredLabel, ReadIR) :-
    findall(SwitchEntry,
        ( nth0(Index, Arms, _),
          format(atom(SwitchEntry), 'i64 ~w, label %vr_a~w', [Index, Index])
        ),
        SwitchEntries),
    atomic_list_concat(SwitchEntries, ' ', SwitchBody),
    format(atom(TagIR),
'  %vr_tag_status = call i64 @wam_stream_read_record(%Value %handle, i64 8, i8* %vr_len_i8)
  %vr_tag_eof = icmp eq i64 %vr_tag_status, 0
  br i1 %vr_tag_eof, label %close_stream, label %vr_tag_chk

vr_tag_chk:
  %vr_tag_ok = icmp eq i64 %vr_tag_status, 1
  br i1 %vr_tag_ok, label %vr_tag_switch, label %fail_read

vr_tag_switch:
  %vr_tag = load i64, i64* %vr_len_scratch
  switch i64 %vr_tag, label %fail_read [ ~w ]
',
        [SwitchBody]),
    findall(ArmIR,
        ( nth0(Index, Arms, ArmTypes),
          format(atom(ArmPrefix), 'vr_a~w', [Index]),
          plawk_varlen_field_sections(ArmTypes, LoweredLabel, ArmPrefix,
              no_eof, 0, rec, 0, Sections),
          atomic_list_concat(Sections, '\n', ArmBodyIR),
          format(atom(ArmIR), '~w:~n~w', [ArmPrefix, ArmBodyIR])
        ),
        ArmIRs),
    atomic_list_concat([TagIR | ArmIRs], '\n', ReadIR).

plawk_union_buf_size(Arms, BufSize) :-
    findall(Size,
        ( member(ArmTypes, Arms),
          plawk_binfmt_record_size(binfmt(ArmTypes), Size)
        ),
        Sizes),
    max_list(Sizes, BufSize).

plawk_emit_record_driver_ir(binfmt_union(Arms), InputPath, Blocks, DriverIR) :-
    !,
    plawk_union_buf_size(Arms, BufSize),
    plawk_normalize_driver_blocks(Blocks,
        driver_blocks(RuntimeGlobals, EntrySetupIR0, LoopPhiIR, LoweredLabel,
            RecordIR, ContinueIR, BreakCloseIR, CloseOkLabel, CloseOkIR)),
    % the 8-byte scratch holds the tag, then any lps length prefixes
    plawk_combine_entry_ir(EntrySetupIR0,
'  %vr_len_scratch = alloca i64, align 8
  %vr_len_i8 = bitcast i64* %vr_len_scratch to i8*',
        EntrySetupIR),
    plawk_union_read_ir(Arms, LoweredLabel, ReadIR),
    llvm_emit_varlen_stream_driver_ir(InputPath, BufSize, ReadIR,
        driver_blocks(RuntimeGlobals, EntrySetupIR, LoopPhiIR, LoweredLabel,
            RecordIR, ContinueIR, BreakCloseIR, CloseOkLabel, CloseOkIR),
        DriverIR).

plawk_emit_record_driver_ir(binfmt(Types), InputPath, Blocks, DriverIR) :-
    plawk_binfmt_has_varlen(Types),
    !,
    plawk_binfmt_record_size(binfmt(Types), BufSize),
    plawk_normalize_driver_blocks(Blocks,
        driver_blocks(RuntimeGlobals, EntrySetupIR0, LoopPhiIR, LoweredLabel,
            RecordIR, ContinueIR, BreakCloseIR, CloseOkLabel, CloseOkIR)),
    % one shared 8-byte scratch for every lps length prefix
    plawk_combine_entry_ir(EntrySetupIR0,
'  %vr_len_scratch = alloca i64, align 8
  %vr_len_i8 = bitcast i64* %vr_len_scratch to i8*',
        EntrySetupIR),
    plawk_varlen_read_ir(Types, LoweredLabel, ReadIR),
    llvm_emit_varlen_stream_driver_ir(InputPath, BufSize, ReadIR,
        driver_blocks(RuntimeGlobals, EntrySetupIR, LoopPhiIR, LoweredLabel,
            RecordIR, ContinueIR, BreakCloseIR, CloseOkLabel, CloseOkIR),
        DriverIR).
plawk_emit_record_driver_ir(binfmt(Types), InputPath, Blocks, DriverIR) :-
    !,
    plawk_binfmt_record_size(binfmt(Types), RecordSize),
    llvm_emit_binary_stream_driver_ir(InputPath, RecordSize, Blocks, DriverIR).
plawk_emit_record_driver_ir(_FieldSeparator, InputPath, Blocks, DriverIR) :-
    llvm_emit_stream_driver_ir(InputPath, Blocks, DriverIR).

plawk_normalize_driver_blocks(
    driver_blocks(RuntimeGlobals, LoopPhiIR, LoweredLabel, RecordIR,
        ContinueIR, CloseOkLabel, CloseOkIR),
    driver_blocks(RuntimeGlobals, '', LoopPhiIR, LoweredLabel, RecordIR,
        ContinueIR, '', CloseOkLabel, CloseOkIR)) :-
    !.
plawk_normalize_driver_blocks(
    driver_blocks(RuntimeGlobals, EntrySetupIR, LoopPhiIR, LoweredLabel,
        RecordIR, ContinueIR, CloseOkLabel, CloseOkIR),
    driver_blocks(RuntimeGlobals, EntrySetupIR, LoopPhiIR, LoweredLabel,
        RecordIR, ContinueIR, '', CloseOkLabel, CloseOkIR)) :-
    !.
plawk_normalize_driver_blocks(Blocks, Blocks).

%% plawk_varlen_read_ir(+Types, +LoweredLabel, -ReadIR)
%
%  Field-by-field read sequence for variable-length records. The first
%  read of a record is the only place clean EOF is legal (branching to
%  close_stream); running out of bytes anywhere later in the record is
%  a partial record and exits through fail_read, matching the
%  fixed-record skeleton's trailing-partial semantics. Numeric fields
%  read their 8 bytes straight into the record buffer; lps(Cap) fields
%  read the 8-byte length into the shared scratch, bound it by Cap,
%  zero the slot, then read the payload (a zero length reads nothing:
%  @wam_stream_read_record returns 1 immediately for size 0).
plawk_varlen_read_ir(Types, LoweredLabel, ReadIR) :-
    plawk_varlen_field_sections(Types, LoweredLabel, vr, eof_first, 0, rec, 0,
        Sections),
    atomic_list_concat(Sections, '\n', ReadIR).

%  Base is the LLVM pointer register destinations gep from: 'rec' for
%  record-level fields, or a rep loop's current-element base pointer
%  (Offset is then relative to the element start).
plawk_varlen_field_sections([], _LoweredLabel, _Prefix, _EofPolicy, _Index,
        _Base, _Offset, []).
plawk_varlen_field_sections([Type | Types], LoweredLabel, Prefix, EofPolicy,
        Index, Base, Offset, [Section | Sections]) :-
    ( Types == []
    -> NextLabel = LoweredLabel
    ;  NextIndex0 is Index + 1,
       format(atom(NextLabel), '~w_f~w', [Prefix, NextIndex0])
    ),
    ( Index =:= 0
    -> LabelIR = ''
    ;  format(atom(LabelIR), '~w_f~w:~n', [Prefix, Index])
    ),
    format(atom(FBase), '~w_f~w', [Prefix, Index]),
    ( Index =:= 0, EofPolicy == eof_first
    -> FieldEof = eof
    ;  FieldEof = no_eof
    ),
    plawk_varlen_field_body(Type, FBase, FieldEof, Base, Offset, NextLabel,
        BodyIR),
    format(atom(Section), '~w~w', [LabelIR, BodyIR]),
    plawk_binfmt_type_width(Type, Width),
    NextOffset is Offset + Width,
    NextIndex is Index + 1,
    plawk_varlen_field_sections(Types, LoweredLabel, Prefix, EofPolicy,
        NextIndex, Base, NextOffset, Sections).

plawk_varlen_field_body(blob(Cap), FBase, FieldEof, Base, Offset, NextLabel,
        BodyIR) :-
    !,
    PayloadOffset is Offset + 8,
    plawk_varlen_eof_check_ir(FieldEof, FBase, cstatus, BodyPrefixIR),
    format(atom(BodyIR),
'  %~w_dst = getelementptr i8, i8* %~w, i64 ~w
  %~w_cstatus = call i64 @wam_stream_read_record(%Value %handle, i64 8, i8* %~w_dst)
~w  %~w_cok = icmp eq i64 %~w_cstatus, 1
  br i1 %~w_cok, label %~w_len, label %fail_read

~w_len:
  %~w_ctp = bitcast i8* %~w_dst to i64*
  %~w_n = load i64, i64* %~w_ctp
  %~w_fits = icmp ule i64 %~w_n, ~w
  br i1 %~w_fits, label %~w_read, label %fail_read

~w_read:
  %~w_pdst = getelementptr i8, i8* %~w, i64 ~w
  call void @llvm.memset.p0i8.i64(i8* %~w_pdst, i8 0, i64 ~w, i1 false)
  %~w_pstatus = call i64 @wam_stream_read_record(%Value %handle, i64 %~w_n, i8* %~w_pdst)
  %~w_pok = icmp eq i64 %~w_pstatus, 1
  br i1 %~w_pok, label %~w, label %fail_read
',
        [FBase, Base, Offset,
         FBase, FBase,
         BodyPrefixIR, FBase, FBase,
         FBase, FBase,
         FBase,
         FBase, FBase,
         FBase, FBase,
         FBase, FBase, Cap,
         FBase, FBase,
         FBase,
         FBase, Base, PayloadOffset,
         FBase, Cap,
         FBase, FBase, FBase,
         FBase, FBase,
         FBase, NextLabel]).
% rep whose elements contain lps strings: the wire size varies per
% element, so the region cannot be one bulk read. Read the count, then
% loop: each iteration parses one element's fields (through the same
% field-body emitters, based at the current element's slot group), so
% in-memory layout is identical to the fixed-element case and
% everything downstream (field access, foreach staging) is unchanged.
% Elements past the count stay zero from the region memset.
plawk_varlen_field_body(rep(Cap, ElemTypes), FBase, FieldEof, Base, Offset,
        NextLabel, BodyIR) :-
    memberchk(lps(_ECap), ElemTypes),
    !,
    maplist(plawk_binfmt_type_width, ElemTypes, ElemWidths),
    sum_list(ElemWidths, ElemSize),
    RegionSize is Cap * ElemSize,
    ElemOffset is Offset + 8,
    plawk_varlen_eof_check_ir(FieldEof, FBase, cstatus, BodyPrefixIR),
    format(atom(HeaderIR),
'  %~w_dst = getelementptr i8, i8* %~w, i64 ~w
  %~w_cstatus = call i64 @wam_stream_read_record(%Value %handle, i64 8, i8* %~w_dst)
~w  %~w_cok = icmp eq i64 %~w_cstatus, 1
  br i1 %~w_cok, label %~w_len, label %fail_read

~w_len:
  %~w_ctp = bitcast i8* %~w_dst to i64*
  %~w_n = load i64, i64* %~w_ctp
  %~w_fits = icmp ule i64 %~w_n, ~w
  br i1 %~w_fits, label %~w_read, label %fail_read

~w_read:
  %~w_edst = getelementptr i8, i8* %~w, i64 ~w
  call void @llvm.memset.p0i8.i64(i8* %~w_edst, i8 0, i64 ~w, i1 false)
  br label %~w_lh

~w_lh:
  %~w_j = phi i64 [ 1, %~w_read ], [ %~w_jn, %~w_ld ]
  %~w_cont = icmp sle i64 %~w_j, %~w_n
  br i1 %~w_cont, label %~w_lb, label %~w

~w_lb:
  %~w_jm1 = sub i64 %~w_j, 1
  %~w_rel = mul i64 %~w_jm1, ~w
  %~w_eb = getelementptr i8, i8* %~w_edst, i64 %~w_rel
',
        [FBase, Base, Offset,
         FBase, FBase,
         BodyPrefixIR, FBase, FBase,
         FBase, FBase,
         FBase,
         FBase, FBase,
         FBase, FBase,
         FBase, FBase, Cap,
         FBase, FBase,
         FBase,
         FBase, Base, ElemOffset,
         FBase, RegionSize,
         FBase,
         FBase,
         FBase, FBase, FBase, FBase,
         FBase, FBase, FBase,
         FBase, FBase, NextLabel,
         FBase,
         FBase, FBase,
         FBase, FBase, ElemSize,
         FBase, FBase, FBase]),
    format(atom(ElemPrefix), '~w_e', [FBase]),
    format(atom(ElemBase), '~w_eb', [FBase]),
    format(atom(DoneLabel), '~w_ld', [FBase]),
    plawk_varlen_field_sections(ElemTypes, DoneLabel, ElemPrefix, no_eof, 0,
        ElemBase, 0, ElemSections),
    atomic_list_concat(ElemSections, '\n', ElemsIR),
    format(atom(FooterIR),
'~w_ld:
  %~w_jn = add i64 %~w_j, 1
  br label %~w_lh
',
        [FBase, FBase, FBase, FBase]),
    atomic_list_concat([HeaderIR, ElemsIR, FooterIR], '\n', BodyIR).
plawk_varlen_field_body(rep(Cap, ElemTypes), FBase, FieldEof, Base, Offset,
        NextLabel, BodyIR) :-
    !,
    maplist(plawk_binfmt_type_width, ElemTypes, ElemWidths),
    sum_list(ElemWidths, ElemSize),
    RegionSize is Cap * ElemSize,
    ElemOffset is Offset + 8,
    plawk_varlen_eof_check_ir(FieldEof, FBase, cstatus, BodyPrefixIR),
    format(atom(BodyIR),
'  %~w_dst = getelementptr i8, i8* %~w, i64 ~w
  %~w_cstatus = call i64 @wam_stream_read_record(%Value %handle, i64 8, i8* %~w_dst)
~w  %~w_cok = icmp eq i64 %~w_cstatus, 1
  br i1 %~w_cok, label %~w_len, label %fail_read

~w_len:
  %~w_ctp = bitcast i8* %~w_dst to i64*
  %~w_n = load i64, i64* %~w_ctp
  %~w_fits = icmp ule i64 %~w_n, ~w
  br i1 %~w_fits, label %~w_read, label %fail_read

~w_read:
  %~w_edst = getelementptr i8, i8* %~w, i64 ~w
  call void @llvm.memset.p0i8.i64(i8* %~w_edst, i8 0, i64 ~w, i1 false)
  %~w_bytes = mul i64 %~w_n, ~w
  %~w_pstatus = call i64 @wam_stream_read_record(%Value %handle, i64 %~w_bytes, i8* %~w_edst)
  %~w_pok = icmp eq i64 %~w_pstatus, 1
  br i1 %~w_pok, label %~w, label %fail_read
',
        [FBase, Base, Offset,
         FBase, FBase,
         BodyPrefixIR, FBase, FBase,
         FBase, FBase,
         FBase,
         FBase, FBase,
         FBase, FBase,
         FBase, FBase, Cap,
         FBase, FBase,
         FBase,
         FBase, Base, ElemOffset,
         FBase, RegionSize,
         FBase, FBase, ElemSize,
         FBase, FBase, FBase,
         FBase, FBase,
         FBase, NextLabel]).
plawk_varlen_field_body(lps(Cap), FBase, FieldEof, Base, Offset, NextLabel,
        BodyIR) :-
    !,
    plawk_varlen_eof_check_ir(FieldEof, FBase, lstatus, BodyPrefixIR),
    format(atom(BodyIR),
'  %~w_lstatus = call i64 @wam_stream_read_record(%Value %handle, i64 8, i8* %vr_len_i8)
~w  %~w_lok = icmp eq i64 %~w_lstatus, 1
  br i1 %~w_lok, label %~w_len, label %fail_read

~w_len:
  %~w_n = load i64, i64* %vr_len_scratch
  %~w_fits = icmp ule i64 %~w_n, ~w
  br i1 %~w_fits, label %~w_read, label %fail_read

~w_read:
  %~w_dst = getelementptr i8, i8* %~w, i64 ~w
  call void @llvm.memset.p0i8.i64(i8* %~w_dst, i8 0, i64 ~w, i1 false)
  %~w_pstatus = call i64 @wam_stream_read_record(%Value %handle, i64 %~w_n, i8* %~w_dst)
  %~w_pok = icmp eq i64 %~w_pstatus, 1
  br i1 %~w_pok, label %~w, label %fail_read
',
        [FBase,
         BodyPrefixIR, FBase, FBase,
         FBase, FBase,
         FBase,
         FBase,
         FBase, FBase, Cap,
         FBase, FBase,
         FBase,
         FBase, Base, Offset,
         FBase, Cap,
         FBase, FBase, FBase,
         FBase, FBase,
         FBase, NextLabel]).
% sN is fixed on the wire (exactly N bytes, NUL-padded by the writer),
% so it reads straight into its slot.
plawk_varlen_field_body(s(Width), FBase, FieldEof, Base, Offset, NextLabel,
        BodyIR) :-
    !,
    plawk_varlen_eof_check_ir(FieldEof, FBase, status, BodyPrefixIR),
    format(atom(BodyIR),
'  %~w_dst = getelementptr i8, i8* %~w, i64 ~w
  %~w_status = call i64 @wam_stream_read_record(%Value %handle, i64 ~w, i8* %~w_dst)
~w  %~w_ok = icmp eq i64 %~w_status, 1
  br i1 %~w_ok, label %~w, label %fail_read
',
        [FBase, Base, Offset,
         FBase, Width, FBase,
         BodyPrefixIR, FBase, FBase,
         FBase, NextLabel]).
plawk_varlen_field_body(_NumericType, FBase, FieldEof, Base, Offset, NextLabel,
        BodyIR) :-
    plawk_varlen_eof_check_ir(FieldEof, FBase, status, BodyPrefixIR),
    format(atom(BodyIR),
'  %~w_dst = getelementptr i8, i8* %~w, i64 ~w
  %~w_status = call i64 @wam_stream_read_record(%Value %handle, i64 8, i8* %~w_dst)
~w  %~w_ok = icmp eq i64 %~w_status, 1
  br i1 %~w_ok, label %~w, label %fail_read
',
        [FBase, Base, Offset,
         FBase, FBase,
         BodyPrefixIR, FBase, FBase,
         FBase, NextLabel]).

% Only the record's first read may see clean EOF (status 0). Later
% fields fall straight through to the 1/other check, where 0 lands in
% fail_read like any short read.
plawk_varlen_eof_check_ir(eof, FBase, StatusSuffix, BodyPrefixIR) :-
    !,
    format(atom(BodyPrefixIR),
'  %~w_eof = icmp eq i64 %~w_~w, 0
  br i1 %~w_eof, label %close_stream, label %~w_chk

~w_chk:
',
        [FBase, FBase, StatusSuffix, FBase, FBase, FBase]).
plawk_varlen_eof_check_ir(no_eof, _FBase, _StatusSuffix, '').

plawk_output_separator(BeginClauses, OutputSeparator) :-
    (   member(begin(Actions), BeginClauses),
        member(set(var('OFS'), string(Value)), Actions)
    ->  string_codes(Value, [OutputSeparator])
    ;   OutputSeparator = 32
    ).

plawk_begin_print_string_globals(BeginClauses, GlobalIR) :-
    plawk_begin_print_fields(BeginClauses, Fields),
    phrase(plawk_begin_print_string_global_lines(Fields, 0), Lines0),
    plawk_fs_regex_global_lines(BeginClauses, RegexLines),
    append(Lines0, RegexLines, Lines),
    atomic_list_concat(Lines, '\n', GlobalIR).

%% plawk_fs_regex_global_lines(+BeginClauses, -Lines)
%
%  The private constant holding a multi-char FS regex pattern, or [] for a
%  single-char / unset FS. Paired with the startup store in plawk_fs_regex_setup.
plawk_fs_regex_global_lines(BeginClauses, [Line]) :-
    plawk_fs_regex_pattern(BeginClauses, Pattern),
    !,
    llvm_emit_c_string_global(wam_fs_regex_pattern, Pattern, Line, _StringLen, _BytesLen).
plawk_fs_regex_global_lines(_BeginClauses, []).

plawk_begin_print_fields([], []).
plawk_begin_print_fields([begin(Actions)], Fields) :-
    (   member(print(Fields), Actions)
    ->  true
    ;   Fields = []
    ).

plawk_begin_print_string_global_lines([], _) -->
    [].
plawk_begin_print_string_global_lines([string(Value) | Rest], Index) -->
    { format(atom(GlobalName), 'plawk_begin_print_string_~w', [Index]),
      llvm_emit_c_string_global(GlobalName, Value, Line, _StringLen, _BytesLen),
      NextIndex is Index + 1
    },
    [Line],
    plawk_begin_print_string_global_lines(Rest, NextIndex).
plawk_begin_print_string_global_lines([_Field | Rest], Index) -->
    { NextIndex is Index + 1 },
    plawk_begin_print_string_global_lines(Rest, NextIndex).

plawk_begin_print_ir([], _OutputSeparator, '') :-
    !.
plawk_begin_print_ir([begin(Actions)], OutputSeparator, IR) :-
    member(print(Fields), Actions),
    !,
    maplist(plawk_begin_print_field, Fields),
    phrase(plawk_begin_print_lines(Fields, OutputSeparator, 0), PrintLines),
    plawk_fs_regex_store_lines([begin(Actions)], StoreLines),
    append(StoreLines, PrintLines, Lines),
    atomic_list_concat(Lines, '\n', IR).
plawk_begin_print_ir([begin(Actions)], _OutputSeparator, IR) :-
    plawk_fs_regex_store_lines([begin(Actions)], StoreLines),
    (   StoreLines == []
    ->  IR = ''
    ;   atomic_list_concat(StoreLines, '\n', IR)
    ).

%% plawk_fs_regex_store_lines(+BeginClauses, -Lines)
%
%  The startup store that points @wam_fs_regex_pattern_ptr at the emitted FS
%  regex pattern (so the field runtime can compile it on first use), or [] when
%  FS is a single char / unset. Runs in the BEGIN block, before the record loop.
plawk_fs_regex_store_lines(BeginClauses, [Line]) :-
    plawk_fs_regex_pattern(BeginClauses, Pattern),
    !,
    llvm_emit_c_string_global(wam_fs_regex_pattern, Pattern, _GlobalIR, _StringLen, BytesLen),
    format(atom(Line),
        '  store i8* getelementptr inbounds ([~w x i8], [~w x i8]* @.wam_fs_regex_pattern, i64 0, i64 0), i8** @wam_fs_regex_pattern_ptr',
        [BytesLen, BytesLen]).
plawk_fs_regex_store_lines(_BeginClauses, []).

plawk_begin_print_field(string(_)).

plawk_begin_print_lines([], _OutputSeparator, _) -->
    { llvm_emit_printf0(plawk_surface_print_newline, 2,
          begin_newline_fmt, printed_begin_newline, [FmtPtr, PrintCall])
    },
    [FmtPtr, PrintCall].
plawk_begin_print_lines([string(Value) | Rest], OutputSeparator, Index) -->
    plawk_begin_separator_lines(Index, OutputSeparator),
    plawk_begin_string_print_lines(Value, Index),
    { NextIndex is Index + 1 },
    plawk_begin_print_lines(Rest, OutputSeparator, NextIndex).

plawk_begin_separator_lines(0, _OutputSeparator) -->
    !,
    [].
plawk_begin_separator_lines(Index, OutputSeparator) -->
    { format(atom(SpaceCall),
          '  %printed_begin_separator_~w = call i32 @putchar(i32 ~w)',
          [Index, OutputSeparator])
    },
    [SpaceCall].

plawk_begin_string_print_lines(Value, Index) -->
    { string_codes(Value, Codes),
      length(Codes, StringLen),
      BytesLen is StringLen + 1,
      format(atom(StringPtr),
          '  %begin_string_~w_ptr = getelementptr [~w x i8], [~w x i8]* @.plawk_begin_print_string_~w, i32 0, i32 0',
          [Index, BytesLen, BytesLen, Index]),
      format(atom(FmtVar), 'begin_string_fmt_~w', [Index]),
      format(atom(PrintVar), 'printed_begin_string_~w', [Index]),
      format(atom(PtrIR), '%begin_string_~w_ptr', [Index]),
      llvm_emit_printf_string(plawk_surface_print_string, FmtVar, PrintVar, PtrIR,
          [FmtPtr, PrintCall])
    },
    [StringPtr],
    [FmtPtr, PrintCall].

plawk_assoc_print_key_globals(PrintFields, GlobalIR) :-
    phrase(plawk_assoc_print_key_global_lines(PrintFields, 0), Lines),
    atomic_list_concat(Lines, '\n', GlobalIR).

plawk_end_print_string_globals(PrintFields, GlobalIR) :-
    phrase(plawk_end_print_string_global_lines(PrintFields, 0), Lines),
    atomic_list_concat(Lines, '\n', GlobalIR).

plawk_end_print_string_global_lines([], _) -->
    [].
plawk_end_print_string_global_lines([string(Value) | Rest], Index) -->
    { format(atom(GlobalName), 'plawk_end_print_string_~w', [Index]),
      llvm_emit_c_string_global(GlobalName, Value, Line, _StringLen, _BytesLen),
      NextIndex is Index + 1
    },
    [Line],
    plawk_end_print_string_global_lines(Rest, NextIndex).
% a concatenation's string operands emit globals under the same
% PrintIndex*1000 + part index scheme the END concat print uses.
plawk_end_print_string_global_lines([concat(Parts) | Rest], Index) -->
    plawk_end_concat_string_globals(Parts, Index, 0),
    { NextIndex is Index + 1 },
    plawk_end_print_string_global_lines(Rest, NextIndex).
plawk_end_print_string_global_lines([Field | Rest], Index) -->
    { \+ Field = string(_),
      \+ Field = concat(_),
      NextIndex is Index + 1
    },
    plawk_end_print_string_global_lines(Rest, NextIndex).

plawk_end_concat_string_globals([], _Index, _PartIndex) -->
    [].
plawk_end_concat_string_globals([string(Value) | Rest], Index, PartIndex) -->
    { CombinedIndex is Index * 1000 + PartIndex,
      format(atom(GlobalName), 'plawk_end_print_string_~w', [CombinedIndex]),
      llvm_emit_c_string_global(GlobalName, Value, Line, _StringLen, _BytesLen),
      NextPartIndex is PartIndex + 1
    },
    [Line],
    plawk_end_concat_string_globals(Rest, Index, NextPartIndex).
plawk_end_concat_string_globals([Part | Rest], Index, PartIndex) -->
    { \+ Part = string(_),
      NextPartIndex is PartIndex + 1
    },
    plawk_end_concat_string_globals(Rest, Index, NextPartIndex).

plawk_assoc_print_key_global_lines([], _) -->
    [].
plawk_assoc_print_key_global_lines([assoc(var(_ArrayName), string(Key)) | Rest], Index) -->
    { plawk_assoc_key_codes(Key, Codes),
      format(atom(GlobalName), 'plawk_assoc_print_key_~w', [Index]),
      llvm_emit_c_string_global(GlobalName, Codes, Line, _KeyLen, _BytesLen),
      NextIndex is Index + 1
    },
    [Line],
    plawk_assoc_print_key_global_lines(Rest, NextIndex).
plawk_assoc_print_key_global_lines([Field | Rest], Index) -->
    { \+ Field = assoc(var(_), string(_)),
      NextIndex is Index + 1 },
    plawk_assoc_print_key_global_lines(Rest, NextIndex).

plawk_assoc_key_codes(Key, Codes) :-
    string(Key),
    !,
    string_codes(Key, Codes).
plawk_assoc_key_codes(Key, Codes) :-
    atom_codes(Key, Codes).

plawk_assoc_end_print_ir(PrintFields, AssocPlan, Descriptor, OutputSeparator, IR) :-
    phrase(plawk_assoc_end_print_lines(PrintFields, AssocPlan, Descriptor,
        OutputSeparator, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_assoc_end_print_lines([], AssocPlan, _Descriptor, _OutputSeparator, _) -->
    { llvm_emit_printf0(plawk_surface_print_newline, 2,
          end_newline_fmt, printed_end_newline, [FmtPtr, PrintCall])
    },
    [FmtPtr, PrintCall],
    plawk_assoc_free_lines(AssocPlan).
plawk_assoc_end_print_lines([assoc(var(ArrayName), int(Key)) | Rest], AssocPlan, Descriptor, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    % Integer-key lookups are binary-mode only (a text-mode literal key
    % would collide with an atom id) -- EXCEPT a positional-array table,
    % whose keys are genuine integer POSITIONS, never interned atom ids,
    % so an int lookup on it is unambiguous in text mode too.
    { ( plawk_descriptor_is_binary(Descriptor)
      ; plawk_assoc_plan_posarray_array(AssocPlan, ArrayName)
      ),
      plawk_assoc_table_index(AssocPlan, ArrayName, TableIndex),
      format(atom(Value),
          '  %assoc_end_value_~w = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_~w, i64 ~w)',
          [PrintIndex, TableIndex, Key]),
      format(atom(ValueIR), '%assoc_end_value_~w', [PrintIndex]),
      (   plawk_assoc_plan_str_array(AssocPlan, ArrayName)
      ->  % str-valued positional table: resolve the stored id to text.
          format(atom(ValueString),
              '  %assoc_end_value_s_~w = call i8* @wam_atom_to_string(i64 ~w)',
              [PrintIndex, ValueIR]),
          format(atom(FmtVar), 'assoc_end_str_fmt_~w', [PrintIndex]),
          format(atom(PrintVar), 'printed_assoc_end_str_~w', [PrintIndex]),
          format(atom(PtrIR), '%assoc_end_value_s_~w', [PrintIndex]),
          llvm_emit_printf_string(plawk_surface_print_string, FmtVar, PrintVar,
              PtrIR, [FmtPtr, PrintCall]),
          ValueLines = [Value, ValueString, FmtPtr, PrintCall]
      ;   format(atom(FmtVar), 'assoc_end_i64_fmt_~w', [PrintIndex]),
          format(atom(PrintVar), 'printed_assoc_end_i64_~w', [PrintIndex]),
          llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar,
              ValueIR, [FmtPtr, PrintCall]),
          ValueLines = [Value, FmtPtr, PrintCall]
      ),
      NextPrintIndex is PrintIndex + 1
    },
    plawk_emit_lines(ValueLines),
    plawk_assoc_end_print_lines(Rest, AssocPlan, Descriptor, OutputSeparator, NextPrintIndex).
plawk_assoc_end_print_lines([assoc(var(ArrayName), string(Key)) | Rest], AssocPlan, Descriptor, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    { plawk_assoc_key_codes(Key, Codes),
      length(Codes, KeyLen),
      BytesLen is KeyLen + 1,
      plawk_assoc_table_index(AssocPlan, ArrayName, TableIndex),
      format(atom(KeyPtr),
          '  %assoc_end_key_~w_ptr = getelementptr [~w x i8], [~w x i8]* @.plawk_assoc_print_key_~w, i32 0, i32 0',
          [PrintIndex, BytesLen, BytesLen, PrintIndex]),
      format(atom(KeyId),
          '  %assoc_end_key_~w_id = call i64 @wam_intern_atom(i8* %assoc_end_key_~w_ptr, i64 ~w)',
          [PrintIndex, PrintIndex, KeyLen]),
      format(atom(Value),
          '  %assoc_end_value_~w = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %assoc_end_key_~w_id)',
          [PrintIndex, TableIndex, PrintIndex]),
      format(atom(ValueIR), '%assoc_end_value_~w', [PrintIndex]),
      (   plawk_assoc_plan_str_array(AssocPlan, ArrayName)
      ->  % str-valued table: resolve the stored atom-registry id to text.
          format(atom(ValueString),
              '  %assoc_end_value_s_~w = call i8* @wam_atom_to_string(i64 ~w)',
              [PrintIndex, ValueIR]),
          format(atom(FmtVar), 'assoc_end_str_fmt_~w', [PrintIndex]),
          format(atom(PrintVar), 'printed_assoc_end_str_~w', [PrintIndex]),
          format(atom(PtrIR), '%assoc_end_value_s_~w', [PrintIndex]),
          llvm_emit_printf_string(plawk_surface_print_string, FmtVar, PrintVar,
              PtrIR, [FmtPtr, PrintCall]),
          ValueLines = [KeyPtr, KeyId, Value, ValueString, FmtPtr, PrintCall]
      ;   format(atom(FmtVar), 'assoc_end_i64_fmt_~w', [PrintIndex]),
          format(atom(PrintVar), 'printed_assoc_end_i64_~w', [PrintIndex]),
          llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar,
              ValueIR, [FmtPtr, PrintCall]),
          ValueLines = [KeyPtr, KeyId, Value, FmtPtr, PrintCall]
      ),
      NextPrintIndex is PrintIndex + 1
    },
    plawk_emit_lines(ValueLines),
    plawk_assoc_end_print_lines(Rest, AssocPlan, Descriptor, OutputSeparator, NextPrintIndex).
plawk_assoc_end_print_lines([string(Value) | Rest], AssocPlan, Descriptor, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_string_print_lines(Value, PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_assoc_end_print_lines(Rest, AssocPlan, Descriptor, OutputSeparator, NextPrintIndex).

plawk_assoc_table_index(assoc_plan(Tables, _Actions), ArrayName, TableIndex) :-
    nth0(TableIndex, Tables, ArrayName).

%% plawk_assoc_plan_str_array(+AssocPlan, +ArrayName) is semidet.
%  ArrayName's table holds STRING values: some rule populates it through
%  an `as assoc(str)` bind (the planned action carries the str(...)
%  wrapper), so reads resolve the stored i64 back to atom text instead
%  of printing it numerically.
plawk_assoc_plan_str_array(assoc_plan(_Tables, Rules), ArrayName) :-
    member(assoc_rule(_RuleIndex, _Pattern, Actions, _Control), Rules),
    ( member(assoc_dyn_action(_Index, ArrayName, _TableIndex, str(_Call)),
        Actions)
    ; member(assoc_dyn_action(_Index2, ArrayName, _TableIndex2,
        posarray_str(_Call2)), Actions)
    ),
    !.
% A reader plan (e.g. the multi-pass `over`/`records` reader) carries the
% program's str-valued table names directly, since the populating writer is
% in a different pass function than this reader.
plawk_assoc_plan_str_array(assoc_plan(_Tables, Rules), ArrayName) :-
    member(str_arrays(StrArrays), Rules),
    memberchk(ArrayName, StrArrays),
    !.
% split fills its array with string pieces (interned atom ids).
plawk_assoc_plan_str_array(assoc_plan(_Tables, Rules), ArrayName) :-
    member(assoc_rule(_RuleIndex, _Pattern, Actions, _Control), Rules),
    member(assoc_split_action(_Index, ArrayName, _TableIndex, _Ki, _Sep), Actions),
    !.

%% plawk_assoc_plan_posarray_array(+AssocPlan, +ArrayName) is semidet.
%  ArrayName''s table is a POSITIONAL array: some rule fills it through an
%  `as array` bind (the planned action carries the posarray(...) wrapper).
%  Its keys are integer positions 1..n, so int-key reads are permitted
%  even in text mode.
plawk_assoc_plan_posarray_array(assoc_plan(_Tables, Rules), ArrayName) :-
    member(assoc_rule(_RuleIndex, _Pattern, Actions, _Control), Rules),
    ( member(assoc_dyn_action(_Index, ArrayName, _TableIndex, posarray(_Call)),
        Actions)
    ; member(assoc_dyn_action(_Index2, ArrayName, _TableIndex2,
        posarray_str(_Call2)), Actions)
    ),
    !.
% split keys its array by integer position (1..n).
plawk_assoc_plan_posarray_array(assoc_plan(_Tables, Rules), ArrayName) :-
    member(assoc_rule(_RuleIndex, _Pattern, Actions, _Control), Rules),
    member(assoc_split_action(_Index, ArrayName, _TableIndex, _Ki, _Sep), Actions),
    !.

% Binary record modes: assoc keys are raw i64 field values (no
% interning), so END key handling prints them numerically.
plawk_descriptor_is_binary(binfmt(_Types)).
plawk_descriptor_is_binary(binfmt_union(_Arms)).

plawk_assoc_free_lines(assoc_plan(Tables, _Actions)) -->
    plawk_assoc_free_lines(Tables, 0).

plawk_assoc_free_lines([], _) -->
    [].
plawk_assoc_free_lines([_ArrayName | Rest], Index) -->
    { format(atom(Line),
          '  call void @wam_assoc_i64_free(%WamAssocI64Table* %plawk_assoc_table_~w)',
          [Index]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_assoc_free_lines(Rest, NextIndex).

plawk_mixed_end_print_ir(PrintFields, ScalarPlan, AssocPlan, OutputSeparator, IR) :-
    phrase(plawk_mixed_end_print_lines(PrintFields, ScalarPlan, AssocPlan, OutputSeparator, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_mixed_end_print_lines([], _ScalarPlan, AssocPlan, _OutputSeparator, _) -->
    { llvm_emit_printf0(plawk_surface_print_newline, 2,
          end_newline_fmt, printed_end_newline, [FmtPtr, PrintCall])
    },
    [FmtPtr, PrintCall],
    plawk_assoc_free_lines(AssocPlan).
plawk_mixed_end_print_lines([var(Name) | Rest], ScalarPlan, AssocPlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    { plawk_state_slot_index(ScalarPlan, scalar_counter(Name), SlotIndex),
      format(atom(FmtVar), 'end_i64_fmt_~w', [PrintIndex]),
      format(atom(PrintVar), 'printed_end_i64_~w', [PrintIndex]),
      format(atom(ValueIR), '%final_slot_~w', [SlotIndex]),
      llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar, ValueIR,
          [FmtPtr, PrintCall]),
      NextPrintIndex is PrintIndex + 1
    },
    [FmtPtr, PrintCall],
    plawk_mixed_end_print_lines(Rest, ScalarPlan, AssocPlan, OutputSeparator, NextPrintIndex).
plawk_mixed_end_print_lines([assoc(var(ArrayName), string(Key)) | Rest], ScalarPlan, AssocPlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    { plawk_assoc_key_codes(Key, Codes),
      length(Codes, KeyLen),
      BytesLen is KeyLen + 1,
      plawk_assoc_table_index(AssocPlan, ArrayName, TableIndex),
      format(atom(KeyPtr),
          '  %assoc_end_key_~w_ptr = getelementptr [~w x i8], [~w x i8]* @.plawk_assoc_print_key_~w, i32 0, i32 0',
          [PrintIndex, BytesLen, BytesLen, PrintIndex]),
      format(atom(KeyId),
          '  %assoc_end_key_~w_id = call i64 @wam_intern_atom(i8* %assoc_end_key_~w_ptr, i64 ~w)',
          [PrintIndex, PrintIndex, KeyLen]),
      format(atom(Value),
          '  %assoc_end_value_~w = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_~w, i64 %assoc_end_key_~w_id)',
          [PrintIndex, TableIndex, PrintIndex]),
      format(atom(ValueIR), '%assoc_end_value_~w', [PrintIndex]),
      (   plawk_assoc_plan_str_array(AssocPlan, ArrayName)
      ->  % str-valued table: resolve the stored atom-registry id to text.
          format(atom(ValueString),
              '  %assoc_end_value_s_~w = call i8* @wam_atom_to_string(i64 ~w)',
              [PrintIndex, ValueIR]),
          format(atom(FmtVar), 'assoc_end_str_fmt_~w', [PrintIndex]),
          format(atom(PrintVar), 'printed_assoc_end_str_~w', [PrintIndex]),
          format(atom(PtrIR), '%assoc_end_value_s_~w', [PrintIndex]),
          llvm_emit_printf_string(plawk_surface_print_string, FmtVar, PrintVar,
              PtrIR, [FmtPtr, PrintCall]),
          ValueLines = [KeyPtr, KeyId, Value, ValueString, FmtPtr, PrintCall]
      ;   format(atom(FmtVar), 'assoc_end_i64_fmt_~w', [PrintIndex]),
          format(atom(PrintVar), 'printed_assoc_end_i64_~w', [PrintIndex]),
          llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar,
              ValueIR, [FmtPtr, PrintCall]),
          ValueLines = [KeyPtr, KeyId, Value, FmtPtr, PrintCall]
      ),
      NextPrintIndex is PrintIndex + 1
    },
    plawk_emit_lines(ValueLines),
    plawk_mixed_end_print_lines(Rest, ScalarPlan, AssocPlan, OutputSeparator, NextPrintIndex).
plawk_mixed_end_print_lines([special('NR') | Rest], ScalarPlan, AssocPlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_nr_print_lines(PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_mixed_end_print_lines(Rest, ScalarPlan, AssocPlan, OutputSeparator, NextPrintIndex).
plawk_mixed_end_print_lines([Expr | Rest], ScalarPlan, AssocPlan, OutputSeparator, PrintIndex) -->
    { plawk_end_scalar_expr(Expr) },
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_expr_print_lines(Expr, ScalarPlan, PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_mixed_end_print_lines(Rest, ScalarPlan, AssocPlan, OutputSeparator, NextPrintIndex).
plawk_mixed_end_print_lines([string(Value) | Rest], ScalarPlan, AssocPlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_string_print_lines(Value, PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_mixed_end_print_lines(Rest, ScalarPlan, AssocPlan, OutputSeparator, NextPrintIndex).

plawk_scalar_print_expr(var(Name), Name).
% a concatenation contributes each operand's scalar reads (so a printed scalar
% inside `print a $1 b` gets a slot).
plawk_scalar_print_expr(concat(Parts), Name) :-
    member(Part, Parts),
    plawk_scalar_print_expr(Part, Name).
plawk_scalar_print_expr(Expr, Name) :-
    plawk_end_scalar_expr(Expr),
    plawk_expr_scalar_read_name(Expr, Name).

plawk_scalar_rule_chain_ir(Rules, StatePlan, FieldSeparator, OutputSeparator,
        GlobalIR, ChainIR, RuleCount, BranchNextExits) :-
    plawk_scalar_planned_rules(Rules, PlannedRules, Controls),
    length(PlannedRules, RuleCount),
    RuleCount > 0,
    phrase(plawk_scalar_rule_chain_lines(PlannedRules, Controls, StatePlan, FieldSeparator, OutputSeparator, 0), Parts),
    plawk_rule_chain_parts(Parts, GlobalParts, ChainParts, BranchNextExits),
    atomic_list_concat(GlobalParts, '\n', GlobalIR),
    atomic_list_concat(ChainParts, '\n', ChainIR).

plawk_scalar_planned_rules(Rules, PlannedRules, Controls) :-
    phrase(plawk_scalar_planned_rule_lines(Rules, 0), PlannedRules),
    findall(Control,
        member(scalar_rule(_Index, _Pattern, _Actions, Control), PlannedRules),
        Controls).

plawk_scalar_planned_rule_lines([], _Index) -->
    [].
plawk_scalar_planned_rule_lines([rule(Pattern, Actions) | Rest], Index) -->
    { plawk_split_terminal_control(Actions, BodyActions, Control),
      NextIndex is Index + 1 },
    [scalar_rule(Index, Pattern, BodyActions, Control)],
    plawk_scalar_planned_rule_lines(Rest, NextIndex).

plawk_scalar_rule_chain_lines([], _Controls, _StatePlan, _FieldSeparator, _OutputSeparator, _) -->
    [].
plawk_scalar_rule_chain_lines([scalar_rule(Index, Pattern, Actions, Control) | Rest], Controls, StatePlan, FieldSeparator, OutputSeparator, Index) -->
    { NextIndex is Index + 1,
      ( Rest == []
      -> NextLabel = 'continue_loop'
      ;  format(atom(NextLabel), 'rule_~w_match', [NextIndex])
      ),
      format(atom(RuleLabel), 'rule_~w_match', [Index]),
      format(atom(ApplyLabel), 'rule_~w_apply', [Index]),
      format(atom(DoneLabel), 'rule_~w_done', [Index]),
      format(atom(MatchVar), 'rule_~w_is_match', [Index]),
      format(atom(GlobalBase), 'plawk_surface_rule_~w', [Index]),
      format(atom(MatchValue), '%~w', [MatchVar]),
      % tagged-union rules carry their arm's field types with them
      plawk_rule_descriptor(Pattern, FieldSeparator, RuleDescriptor),
      plawk_pattern_guard_ir(Pattern, RuleDescriptor, GlobalBase, MatchValue,
          GuardGlobalIR-GuardCallIR),
      plawk_rule_target(Control, NextLabel, RuleTargetLabel),
      maplist(plawk_scalar_rule_body_action, Actions),
      BodyActions = Actions,
      plawk_scalar_rule_input_phi_ir(StatePlan, Index, Controls, InputPhiIR),
      plawk_scalar_match_update_ir(StatePlan, BodyActions, RuleDescriptor, OutputSeparator, Index,
          BranchNextExits, MatchUpdateGlobalIR-MatchUpdateIR),
      ( Index =:= 0
      -> EntryIR = '  br label %rule_0_match\n\n'
      ;  EntryIR = ''
      ),
      format(atom(BranchIR),
'~w~w:
~w~w
  br i1 %~w, label %~w, label %~w

~w:
~w
  br label %~w

~w:
  br label %~w',
          [EntryIR, RuleLabel, InputPhiIR, GuardCallIR, MatchVar,
           ApplyLabel, NextLabel, ApplyLabel, MatchUpdateIR, DoneLabel,
           DoneLabel, RuleTargetLabel]),
      format(atom(CombinedGlobalIR), '~w~n~w', [GuardGlobalIR, MatchUpdateGlobalIR]),
      Part = rule_chain_part(CombinedGlobalIR, BranchIR, BranchNextExits)
    },
    [Part],
    plawk_scalar_rule_chain_lines(Rest, Controls, StatePlan, FieldSeparator, OutputSeparator, NextIndex).

plawk_scalar_rule_input_phi_ir(_StatePlan, 0, _Controls, '') :-
    !.
plawk_scalar_rule_input_phi_ir(StatePlan, RuleIndex, Controls, IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_scalar_rule_input_phi_lines(Slots, RuleIndex, Controls, 0), Lines),
    atomic_list_concat(Lines, '\n', LinesIR),
    format(atom(IR), '~w~n', [LinesIR]).

plawk_scalar_rule_input_phi_lines([], _RuleIndex, _Controls, _) -->
    [].
plawk_scalar_rule_input_phi_lines([Slot | Rest], RuleIndex, Controls, SlotIndex) -->
    { PrevRuleIndex is RuleIndex - 1,
      plawk_scalar_rule_input_value(PrevRuleIndex, SlotIndex, PrevFalseValue),
      format(atom(FalseIncoming), '[~w, %rule_~w_match]',
          [PrevFalseValue, PrevRuleIndex]),
      (   plawk_terminal_control_skips_next_rule(Controls, PrevRuleIndex)
      ->  Incomings = [FalseIncoming]
      ;   format(atom(ApplyIncoming), '[%rule_~w_slot_~w, %rule_~w_done]',
              [PrevRuleIndex, SlotIndex, PrevRuleIndex]),
          Incomings = [FalseIncoming, ApplyIncoming]
      ),
      atomic_list_concat(Incomings, ', ', IncomingIR),
      plawk_slot_llvm_type(Slot, Type),
      format(atom(Line), '  %rule_~w_in_slot_~w = phi ~w ~w',
          [RuleIndex, SlotIndex, Type, IncomingIR]),
      NextSlotIndex is SlotIndex + 1
    },
    [Line],
    plawk_scalar_rule_input_phi_lines(Rest, RuleIndex, Controls, NextSlotIndex).

plawk_terminal_control_skips_next_rule(Controls, RuleIndex) :-
    nth0(RuleIndex, Controls, Control),
    memberchk(Control, [terminal_next, terminal_break, terminal_exit]).

plawk_scalar_rule_input_value(0, SlotIndex, Value) :-
    !,
    format(atom(Value), '%slot_~w', [SlotIndex]).
plawk_scalar_rule_input_value(RuleIndex, SlotIndex, Value) :-
    format(atom(Value), '%rule_~w_in_slot_~w', [RuleIndex, SlotIndex]).

plawk_scalar_rule_slot_input(0, SlotIndex, Value) :-
    !,
    format(atom(Value), '%slot_~w', [SlotIndex]).
plawk_scalar_rule_slot_input(RuleIndex, SlotIndex, Value) :-
    format(atom(Value), '%rule_~w_in_slot_~w', [RuleIndex, SlotIndex]).

plawk_state_loop_phi_ir(StatePlan, IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_scalar_loop_phi_lines(Slots, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_scalar_loop_phi_lines([], _) -->
    [].
plawk_scalar_loop_phi_lines([Slot | Rest], Index) -->
    { plawk_slot_llvm_type(Slot, Type),
      plawk_slot_zero_ir(Slot, Zero),
      format(atom(Line),
          '  %slot_~w = phi ~w [~w, %check_handle_value], [%next_slot_~w, %continue_loop]',
          [Index, Type, Zero, Index]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_scalar_loop_phi_lines(Rest, NextIndex).

plawk_scalar_match_update_ir(StatePlan, Actions, FieldSeparator, OutputSeparator, RuleIndex, NextExits, GlobalIR-IR) :-
    plawk_native_match_update_ir(StatePlan, none, Actions, FieldSeparator, OutputSeparator, RuleIndex, NextExits, GlobalIR-IR).

plawk_native_match_update_ir(StatePlan, AssocPlan, Actions, FieldSeparator, OutputSeparator, RuleIndex, NextExits, GlobalIR-IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_scalar_initial_slot_values(RuleIndex, Slots, 0), InitialValues),
    format(atom(Prefix), 'rule_~w_body', [RuleIndex]),
    phrase(plawk_scalar_action_sequence_pairs(Actions, Slots, AssocPlan, FieldSeparator, OutputSeparator,
        Prefix, Prefix, RuleIndex, 0, InitialValues, FinalValues, _NextOpIndex, _ExitLabel, NextExits), Pairs0),
    phrase(plawk_scalar_final_slot_pairs(FinalValues, Slots, RuleIndex, 0), FinalPairs),
    append(Pairs0, FinalPairs, Pairs),
    pairs_keys_values(Pairs, GlobalParts, LineParts),
    atomic_list_concat(GlobalParts, '\n', GlobalIR),
    atomic_list_concat(LineParts, '\n', IR).

plawk_scalar_initial_slot_values(_RuleIndex, [], _) -->
    [].
plawk_scalar_initial_slot_values(RuleIndex, [_Slot | Rest], SlotIndex) -->
    { plawk_scalar_rule_slot_input(RuleIndex, SlotIndex, Value),
      NextIndex is SlotIndex + 1
    },
    [Value],
    plawk_scalar_initial_slot_values(RuleIndex, Rest, NextIndex).

plawk_scalar_final_slot_pairs([], _Slots, _RuleIndex, _) -->
    [].
plawk_scalar_final_slot_pairs([Value | Rest], [Slot | Slots], RuleIndex, SlotIndex) -->
    { % SSA copy idiom: x + 0 for i64, -0.0 + x for double (IEEE identity
      % that preserves the sign of zero, unlike x + 0.0).
      ( Slot = scalar_double(_Name)
      -> format(atom(Line), '  %rule_~w_slot_~w = fadd double -0.0, ~w',
             [RuleIndex, SlotIndex, Value])
      ;  format(atom(Line), '  %rule_~w_slot_~w = add i64 ~w, 0',
             [RuleIndex, SlotIndex, Value])
      ),
      NextSlotIndex is SlotIndex + 1
    },
    [''-Line],
    plawk_scalar_final_slot_pairs(Rest, Slots, RuleIndex, NextSlotIndex).

plawk_scalar_update_action(Action) :-
    plawk_scalar_action_update(Action, _Name, _Operation).
plawk_scalar_update_action(Action) :-
    plawk_scalar_conditional_action(Action).
plawk_scalar_update_action(Action) :-
    plawk_dynrec_bind_ok(Action).

plawk_scalar_rule_body_action(Action) :-
    plawk_scalar_action_update(Action, _Name, _Operation).
plawk_scalar_rule_body_action(Action) :-
    plawk_dynrec_bind_ok(Action).
% the store-only half of a rule-level `exit N` (terminal_exit split); it is a
% plain store with no control, valid anywhere a plain body action is.
plawk_scalar_rule_body_action(exit_store(_Code)).
plawk_scalar_rule_body_action(Action) :-
    plawk_rule_body_print_action(Action).
plawk_scalar_rule_body_action(writebin_out(Types, Fields)) :-
    plawk_writebin_args_ok(Types, Fields).
plawk_scalar_rule_body_action(writebin_arm_out(_Tag, ArmTypes, Fields)) :-
    plawk_writebin_args_ok(ArmTypes, Fields).
plawk_scalar_rule_body_action(foreach_loop(_Layout, Body)) :-
    plawk_scalar_branch_body_actions(Body).
plawk_scalar_rule_body_action(while_loop(Cond, Body)) :-
    plawk_while_cond_ok(Cond),
    plawk_scalar_branch_body_actions(Body).
plawk_scalar_rule_body_action(do_while_loop(Body, Cond)) :-
    plawk_while_cond_ok(Cond),
    plawk_scalar_branch_body_actions(Body).
plawk_scalar_rule_body_action(if(_Pattern, ThenActions, ElseActions)) :-
    append(ThenActions, ElseActions, Actions),
    Actions \== [],
    plawk_scalar_branch_body_actions(ThenActions),
    plawk_scalar_branch_body_actions(ElseActions).

plawk_scalar_branch_body_actions(Actions) :-
    plawk_split_branch_control(Actions, BodyActions, _Control),
    maplist(plawk_scalar_rule_body_plain_action, BodyActions).

plawk_scalar_rule_body_plain_action(Action) :-
    plawk_scalar_action_update(Action, _Name, _Operation).
% `exit [N]` inside a branch body (`if (c) exit`): the sequence walker lowers it
% via branch_exit + plawk_branch_to_done_ir (branch to break_close_stream); the
% store-only marker `exit_store` may appear when a branch ends the rule.
plawk_scalar_rule_body_plain_action(exit(int(_Code))).
plawk_scalar_rule_body_plain_action(exit_store(_Code)).
% a structured-return destructure inside a branch body: the sequence
% walker lowers dynrec_bind wherever it appears (its slots/call IR is
% branch-position-independent), so branch-body validation accepts it
% exactly as the top-level body does.
plawk_scalar_rule_body_plain_action(Action) :-
    plawk_dynrec_bind_ok(Action).
plawk_scalar_rule_body_plain_action(Action) :-
    plawk_rule_body_print_action(Action).
plawk_scalar_rule_body_plain_action(writebin_out(Types, Fields)) :-
    plawk_writebin_args_ok(Types, Fields).
plawk_scalar_rule_body_plain_action(writebin_arm_out(_Tag, ArmTypes, Fields)) :-
    plawk_writebin_args_ok(ArmTypes, Fields).
% else-if chains nest an if inside a branch body; the sequence walker
% lowers nested ifs recursively, so validation recurses the same way.
plawk_scalar_rule_body_plain_action(if(Pattern, ThenActions, ElseActions)) :-
    plawk_scalar_rule_body_action(if(Pattern, ThenActions, ElseActions)).
% a loop may nest inside another loop body (`while (..) { while (..) { .. } }`);
% the sequence walker lowers the inner loop recursively, so validation recurses.
plawk_scalar_rule_body_plain_action(while_loop(Cond, Body)) :-
    plawk_scalar_rule_body_action(while_loop(Cond, Body)).
plawk_scalar_rule_body_plain_action(do_while_loop(Body, Cond)) :-
    plawk_scalar_rule_body_action(do_while_loop(Body, Cond)).
plawk_scalar_rule_body_plain_action(foreach_loop(Layout, Body)) :-
    plawk_scalar_rule_body_action(foreach_loop(Layout, Body)).

plawk_rule_body_print_action(print(Fields)) :-
    Fields = [_ | _],
    maplist(plawk_rule_body_print_field, Fields).
plawk_rule_body_print_action(printf(string(Format), Args)) :-
    string(Format),
    maplist(plawk_rule_body_print_field, Args).

plawk_rule_body_print_field(field(_)).
plawk_rule_body_print_field(string(_)).
% a ternary `COND ? A : B`: the condition operands and both branches must be
% i64-valued (field / NR / NF / int literal / length / i64 arithmetic); lowered
% to an LLVM select.
plawk_rule_body_print_field(ternary(cmp(Left, _Op, Right), Then, Else)) :-
    plawk_ternary_i64_operand_ok(Left),
    plawk_ternary_i64_operand_ok(Right),
    plawk_ternary_i64_operand_ok(Then),
    plawk_ternary_i64_operand_ok(Else).
% a string concatenation (`print $1 $2`): every operand must be a valid print
% field; they are emitted adjacently with no separator.
plawk_rule_body_print_field(concat(Parts)) :-
    Parts = [_, _ | _],
    maplist(plawk_rule_body_print_field, Parts).
% a bare scalar variable read (`print i`): substitution rewrites var(Name) to
% the slot's SSA value at emit time, so a scalar can be printed directly. Only
% names that resolve to a slot compile; a non-slot var fails substitution.
plawk_rule_body_print_field(var(_)).
plawk_rule_body_print_field(special('NR')).
plawk_rule_body_print_field(special('NF')).
plawk_rule_body_print_field(environ(Key)) :- string(Key).
plawk_rule_body_print_field(argv_at(N)) :- integer(N), N >= 0.
plawk_rule_body_print_field(special('ARGC')).
plawk_rule_body_print_field(int(field(_))).
plawk_rule_body_print_field(Expr) :-
    plawk_i64_general_binary_expr(Expr).
plawk_rule_body_print_field(Expr) :-
    plawk_prolog_call_expr(Expr).
plawk_rule_body_print_field(Expr) :-
    plawk_dyncall_expr(Expr).
plawk_rule_body_print_field(Expr) :-
    plawk_dyncall_named_expr(Expr).
plawk_rule_body_print_field(Expr) :-
    plawk_dyncall_at_expr(Expr).
plawk_rule_body_print_field(blob_slice_vars(var(P), var(L))) :-
    atom(P), atom(L).
plawk_rule_body_print_field(blob_dyncall(Args)) :-
    plawk_float_dyncall_expr(float_dyncall(Args)).   % same arg shape check
plawk_rule_body_print_field(blob_dyncall_named(Name, Args)) :-
    plawk_float_dyncall_named_expr(float_dyncall_named(Name, Args)).
plawk_rule_body_print_field(blob_dyncall_at(Source, Args)) :-
    plawk_float_dyncall_at_expr(float_dyncall_at(Source, Args)).
plawk_rule_body_print_field(blob_dyncall_at_named(Name, Source, Args)) :-
    plawk_float_dyncall_at_expr(float_dyncall_at_named(Name, Source, Args)).
plawk_rule_body_print_field(Expr) :-
    plawk_f64_print_expr(Expr).
plawk_rule_body_print_field(length(field(_))).
plawk_rule_body_print_field(match_expr(field(_), _Regex)).
plawk_rule_body_print_field(special('RSTART')).
plawk_rule_body_print_field(special('RLENGTH')).
plawk_rule_body_print_field(substr(field(_), _Start, _Len)).
plawk_rule_body_print_field(index(field(_), string(_))).
plawk_rule_body_print_field(tolower(field(_))).
plawk_rule_body_print_field(toupper(field(_))).

%% plawk_i64_general_binary_expr(+Expr) is semidet.
%
%  Recognize a native i64 binary expression tree whose leaves are i64
%  primaries, integer literals, or bare numeric field coercions.
plawk_i64_general_binary_expr(Expr) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    plawk_i64_operand_expr(Left),
    plawk_i64_operand_expr(Right).

plawk_i64_operand_expr(int(Value)) :-
    integer(Value).
plawk_i64_operand_expr(field(FieldIndex)) :-
    integer(FieldIndex),
    FieldIndex >= 0.
plawk_i64_operand_expr(Expr) :-
    plawk_i64_binary_primary_expr(Expr).
plawk_i64_operand_expr(prolog_call(Name, Args)) :-
    plawk_prolog_call_expr(prolog_call(Name, Args)).
plawk_i64_operand_expr(dyncall(Args)) :-
    plawk_dyncall_expr(dyncall(Args)).
plawk_i64_operand_expr(dyncall_named(Name, Args)) :-
    plawk_dyncall_named_expr(dyncall_named(Name, Args)).
plawk_i64_operand_expr(dyncall_at(Source, Args)) :-
    plawk_dyncall_at_expr(dyncall_at(Source, Args)).
plawk_i64_operand_expr(dyncall_at_named(Name, Source, Args)) :-
    plawk_dyncall_at_expr(dyncall_at_named(Name, Source, Args)).
plawk_i64_operand_expr(Expr) :-
    plawk_i64_general_binary_expr(Expr).

%% plawk_ternary_i64_operand_ok(+Expr) is semidet.
%  A ternary condition operand or branch: any i64 operand plus the surface
%  forms the parser emits for fields / NR / NF / length that plawk_i64_expr_ir
%  lowers directly.
plawk_ternary_i64_operand_ok(special('NR')).
plawk_ternary_i64_operand_ok(special('NF')).
plawk_ternary_i64_operand_ok(int(field(_))).
plawk_ternary_i64_operand_ok(length(field(_))).
plawk_ternary_i64_operand_ok(Expr) :-
    plawk_i64_operand_expr(Expr).

plawk_prolog_call_expr(prolog_call(Name, Args)) :-
    atom(Name),
    Args = [_ | _],
    maplist(plawk_foreign_arg, Args).

%% plawk_dyncall_expr(+Expr) is semidet.
%
%  dyncall(args...) routes to a runtime-loaded .wamo object's entry
%  (declared by BEGIN { DYNLOAD = "..." }) and yields its integer
%  binding, or 0 on load/call failure. Same arg shapes as a compiled
%  foreign i64 call; it just targets the object-call shim instead of a
%  compiled @<name>_start_pc.
plawk_dyncall_expr(dyncall(Args)) :-
    Args = [_ | _],
    maplist(plawk_foreign_arg, Args).

%% plawk_dyncall_named_expr(+Expr) is semidet.
%
%  dyncall@name(args...) routes to a named entry of the DYNLOAD object
%  (a multi-entry .wamo). Same arg shapes as bare dyncall; the entry Name
%  is a compile-time atom, resolved to a label index once at startup.
plawk_dyncall_named_expr(dyncall_named(Name, Args)) :-
    atom(Name),
    Args = [_ | _],
    maplist(plawk_foreign_arg, Args).

%% plawk_dyncall_at_expr(+Expr) is semidet.
%
%  dyncall_at(Source, args...) is the dynamic-source form: Source (a
%  field or string literal) names the .wamo object at runtime, args... are
%  the entry inputs. Same yield semantics as dyncall (i64, 0 on failure).
%  Source may also be compile_src(field-or-string) -- the eval surface:
%  the Prolog source text compiles at runtime to a grammar handle.
plawk_dyncall_at_expr(dyncall_at(Source, Args)) :-
    plawk_dyncall_at_source_ok(Source),
    maplist(plawk_foreign_arg, Args).
% dyncall_at@name(Source, args...): a named entry on a runtime source --
% same source and arg shapes; only the entry selection differs.
plawk_dyncall_at_expr(dyncall_at_named(Name, Source, Args)) :-
    atom(Name),
    plawk_dyncall_at_source_ok(Source),
    maplist(plawk_foreign_arg, Args).

plawk_dyncall_at_source_ok(compile_src(Arg)) :-
    !,
    plawk_foreign_arg(Arg).
% compile_file(path): the path names a grammar SOURCE file read at
% runtime; its contents compile through the same registry as
% compile(...), so an edited file is a new source text and recompiles
% (change-rebust by content dedup, no mtime bookkeeping).
plawk_dyncall_at_source_ok(compile_file_src(Arg)) :-
    !,
    plawk_foreign_arg(Arg).
% a grammar handle read from a scalar (h = compile("..."); the handle
% is an i64 registry index and travels as (null path, handle id))
plawk_dyncall_at_source_ok(handle_src(var(Name))) :-
    !,
    atom(Name).
plawk_dyncall_at_source_ok(Source) :-
    plawk_foreign_arg(Source).

%% float(dyncall(...)) / float(dyncall_at(...)) -- double-returning
%  runtime-object calls: the grammar's numeric output keeps its fraction.
plawk_float_dyncall_expr(float_dyncall(Args)) :-
    Args = [_ | _],
    maplist(plawk_foreign_arg, Args).
plawk_float_dyncall_named_expr(float_dyncall_named(Name, Args)) :-
    atom(Name),
    Args = [_ | _],
    maplist(plawk_foreign_arg, Args).
plawk_float_dyncall_at_expr(float_dyncall_at(Source, Args)) :-
    plawk_dyncall_at_source_ok(Source),
    maplist(plawk_foreign_arg, Args).
plawk_float_dyncall_at_expr(float_dyncall_at_named(Name, Source, Args)) :-
    atom(Name),
    plawk_dyncall_at_source_ok(Source),
    maplist(plawk_foreign_arg, Args).

%% plawk_dynrec_bind_ok(+Action) is semidet.
%
%  A structured-return destructuring bind
%  `(V1, ..., Vn) = <dyncall> as (T1 ... Tn)` is compilable when: the type
%  list is the same length as the bindings, each binding matches its type
%  (an i64/f64 field binds one scalar-var atom; a string field binds a
%  str(PtrVar, LenVar) pair -- string fields arise only from the
%  record-view desugar, since a string can't bind to a scalar surface
%  variable), all bound names are distinct, and the call is a bare
%  dyncall(...) or dyncall@name(...). Each numeric field's Compound arg is
%  deserialized into its slot; a string field's (ptr,len) slice lands in
%  the two i64 slots -- all via @wam_object_call_record.
plawk_dynrec_bind_ok(dynrec_bind(Vars, Call, Types)) :-
    Vars = [_ | _],
    same_length(Vars, Types),
    maplist(plawk_dynrec_binding_ok, Vars, Types),
    plawk_dynrec_binding_names(Vars, Names),
    sort(Names, Sorted), same_length(Names, Sorted),   % distinct
    plawk_dynrec_call_ok(Call).

plawk_dynrec_binding_ok(V, i64) :- atom(V).
plawk_dynrec_binding_ok(V, f64) :- atom(V).
plawk_dynrec_binding_ok(str(P, L), string) :- atom(P), atom(L).

plawk_dynrec_binding_names([], []).
plawk_dynrec_binding_names([str(P, L) | Rest], [P, L | Names]) :-
    !,
    plawk_dynrec_binding_names(Rest, Names).
plawk_dynrec_binding_names([V | Rest], [V | Names]) :-
    plawk_dynrec_binding_names(Rest, Names).

plawk_dynrec_type_ok(i64).
plawk_dynrec_type_ok(f64).

plawk_dynrec_call_ok(dyncall(Args)) :-
    plawk_dyncall_expr(dyncall(Args)).
plawk_dynrec_call_ok(dyncall_named(Name, Args)) :-
    plawk_dyncall_named_expr(dyncall_named(Name, Args)).
% runtime-source record forms (the JIT binary reader)
plawk_dynrec_call_ok(dyncall_at(Source, Args)) :-
    plawk_dyncall_at_expr(dyncall_at(Source, Args)).
plawk_dynrec_call_ok(dyncall_at_named(Name, Source, Args)) :-
    plawk_dyncall_at_expr(dyncall_at_named(Name, Source, Args)).

%% plawk_dynrec_type_code(+Type, -Byte)  i64->0, f64->1, string->2 (typecodes).
plawk_dynrec_type_code(i64, 0).
plawk_dynrec_type_code(f64, 1).
plawk_dynrec_type_code(string, 2).

%% plawk_resolve_dynrec_view_rules(+Rules0, -Rules)
%
%  Desugar record-view blocks. `dyncall[@name](args) as (T1..Tn) { Body }`
%  becomes a destructure into fresh hidden temporaries followed by Body with
%  every `$k` (field(k) / float_field(k), 1<=k<=n) rewritten to the k-th
%  temporary -- so the returned record reads like the current record inside
%  the block, riding the destructure machinery with no field-pointer
%  repoint. A per-site counter keeps temp names unique; if the body
%  references a field outside 1..n (including $0), it does not rewrite and
%  the view is left uncompilable (the record has no such field). Recurses
%  into if-branches; a view nested in a for-in body is a follow-on.
plawk_resolve_dynrec_view_rules(Rules0, Rules) :-
    plawk_resolve_dynrec_view_rules(Rules0, 0, Rules).

plawk_resolve_dynrec_view_rules([], _, []).
plawk_resolve_dynrec_view_rules([rule(Pattern, Actions0) | Rest], K0,
        [rule(Pattern, Actions) | RestOut]) :-
    !,
    plawk_resolve_dynrec_view_actions(Actions0, K0, K1, Actions),
    plawk_resolve_dynrec_view_rules(Rest, K1, RestOut).
plawk_resolve_dynrec_view_rules([Other | Rest], K0, [Other | RestOut]) :-
    plawk_resolve_dynrec_view_rules(Rest, K0, RestOut).

plawk_resolve_dynrec_view_actions([], K, K, []).
plawk_resolve_dynrec_view_actions([A0 | As0], K0, K, Out) :-
    plawk_resolve_dynrec_view_action(A0, K0, K1, Expanded),
    plawk_resolve_dynrec_view_actions(As0, K1, K, RestOut),
    append(Expanded, RestOut, Out).

plawk_resolve_dynrec_view_action(dynrec_view(Call, Types, Body0), K0, K,
        [dynrec_bind(Bindings, Call, Types) | Body]) :-
    !,
    plawk_dynrec_view_specs(K0, 1, Types, Bindings, FieldTargets),
    K1 is K0 + 1,
    plawk_resolve_dynrec_view_actions(Body0, K1, K, Body1),
    maplist(plawk_dynrec_rewrite_field_targets(FieldTargets), Body1, Body),
    \+ plawk_term_has_field_ref(Body).
plawk_resolve_dynrec_view_action(if(Pattern, Then0, Else0), K0, K,
        [if(Pattern, Then, Else)]) :-
    !,
    plawk_resolve_dynrec_view_actions(Then0, K0, K1, Then),
    plawk_resolve_dynrec_view_actions(Else0, K1, K, Else).
% A view nested in a loop body: recurse so its `$k` rewrite happens
% wherever the view sits. `foreach` runs before its own foreach-resolve
% pass (which rebinds a naked `$k` to the current element), so a view's
% block `$k` become the view's hidden temps here and only unrewritten
% `$k` reach the element rebind -- and the view's Call args (e.g. `$1`
% passing the current element into the grammar) are left for that pass.
plawk_resolve_dynrec_view_action(foreach(Body0), K0, K, [foreach(Body)]) :-
    !,
    plawk_resolve_dynrec_view_actions(Body0, K0, K, Body).
plawk_resolve_dynrec_view_action(for_in(V, A, Body0), K0, K,
        [for_in(V, A, Body)]) :-
    !,
    plawk_resolve_dynrec_view_actions(Body0, K0, K, Body).
plawk_resolve_dynrec_view_action(Action, K, K, [Action]).

%% plawk_dynrec_view_specs(+K, +I, +Types, -Bindings, -FieldTargets)
%  Per field: a numeric type gets one scalar temp (binding = the atom,
%  target = var(temp)); a string type gets a (ptr,len) temp pair (binding
%  = str(PtrTemp,LenTemp), target = a blob slice built from those two i64
%  scalars). FieldTargets maps 1-based field index -> the rewrite target
%  for `$k` in the body.
plawk_dynrec_view_specs(_K, _I, [], [], []).
plawk_dynrec_view_specs(K, I, [Type | Ts], [Binding | Bs], [I-Target | Rs]) :-
    (   Type == string
    ->  format(atom(P), '__dynrec_~w_~w_p', [K, I]),
        format(atom(L), '__dynrec_~w_~w_l', [K, I]),
        Binding = str(P, L),
        Target = blob_slice_vars(var(P), var(L))
    ;   format(atom(T), '__dynrec_~w_~w', [K, I]),
        Binding = T,
        Target = var(T)
    ),
    I1 is I + 1,
    plawk_dynrec_view_specs(K, I1, Ts, Bs, Rs).

% Rewrite $k field references to their per-field target (var or slice).
plawk_dynrec_rewrite_field_targets(Targets, field(N), Target) :-
    integer(N), memberchk(N-Target, Targets), !.
plawk_dynrec_rewrite_field_targets(Targets, float_field(N), Target) :-
    integer(N), memberchk(N-Target, Targets), !.
plawk_dynrec_rewrite_field_targets(_Targets, Term, Term) :-
    ( var(Term) ; atomic(Term) ), !.
plawk_dynrec_rewrite_field_targets(Targets, Term0, Term) :-
    Term0 =.. [F | Args0],
    maplist(plawk_dynrec_rewrite_field_targets(Targets), Args0, Args),
    Term =.. [F | Args].

plawk_term_has_field_ref(Term) :-
    ( Term = field(_) ; Term = float_field(_) ),
    !.
plawk_term_has_field_ref(Term) :-
    compound(Term),
    arg(_, Term, Sub),
    plawk_term_has_field_ref(Sub).

%% plawk_dynrec_bind_pair(+Vars, +Call, +Types, +FieldSeparator, +Base,
%%                        +Slots, +Values0, -Values, -GlobalIR-LineIR)
%
%  Emit the IR for one destructure bind and thread the scalar slot values.
%  Line IR: evaluate the call args, alloca an i64[nfields] slot array
%  (zero-init so a failed call reads as 0/0.0), call the record shim
%  (@plawk_dyncall_rec_<N> or @plawk_dyncall_named_rec_<Sym>) which fills
%  the slots via @wam_object_call_record, then load each field (i64 direct,
%  f64 as bitcast-from-bits). Each bound variable's threaded slot value is
%  set to its loaded field SSA, so the surrounding scalar dataflow (loop
%  phis, final slot copies) carries it like any other assignment.
plawk_dynrec_bind_pair(Vars, Call, Types, FieldSeparator, Base, Slots,
        Values0, Values, GlobalIR-LineIR) :-
    length(Types, NFields),
    maplist(plawk_dynrec_type_code, Types, Codes),
    plawk_dynrec_typecodes_escaped(Codes, Escaped),
    format(atom(TCGlobal),
        '@.~w_tc = private constant [~w x i8] c"~w"', [Base, NFields, Escaped]),
    plawk_dynrec_call_ir(Call, FieldSeparator, Base, CallArgsIR,
        ArgGlobals, ArgSetup, ShimName),
    NLast is NFields - 1,
    numlist(0, NLast, Fields),
    findall(ZLine,
        ( member(FZ, Fields),
          format(atom(ZLine),
              '  %~w_z~wp = getelementptr i64, i64* %~w_slots, i64 ~w\n  store i64 0, i64* %~w_z~wp\n  %~w_zl~wp = getelementptr i64, i64* %~w_lens, i64 ~w\n  store i64 0, i64* %~w_zl~wp',
              [Base, FZ, Base, FZ, Base, FZ, Base, FZ, Base, FZ, Base, FZ]) ),
        ZeroLines),
    format(atom(AllocaIR),
        '  %~w_slots = alloca i64, i32 ~w\n  %~w_lens = alloca i64, i32 ~w',
        [Base, NFields, Base, NFields]),
    format(atom(TCPtrIR),
        '  %~w_tcp = getelementptr [~w x i8], [~w x i8]* @.~w_tc, i32 0, i32 0',
        [Base, NFields, NFields, Base]),
    format(atom(CallIR),
        '  %~w_ok = call i1 @~w(~w, i32 ~w, i8* %~w_tcp, i64* %~w_slots, i64* %~w_lens)',
        [Base, ShimName, CallArgsIR, NFields, Base, Base, Base]),
    plawk_dynrec_field_load_lines(Fields, Types, Base, LoadLines),
    append([ArgSetup, [AllocaIR], ZeroLines, [TCPtrIR, CallIR], LoadLines],
        LineList),
    atomic_list_concat(LineList, '\n', LineIR),
    atomic_list_concat([TCGlobal | ArgGlobals], '\n', GlobalIR),
    plawk_dynrec_update_slots(Vars, 0, Base, Slots, Values0, Values).

plawk_dynrec_typecodes_escaped(Codes, Escaped) :-
    findall(S, ( member(C, Codes), format(atom(S), '\\0~w', [C]) ), Parts),
    atomic_list_concat(Parts, Escaped).

plawk_dynrec_call_parts(dyncall(Args), Args, ShimName) :-
    length(Args, NArgs),
    format(atom(ShimName), 'plawk_dyncall_rec_~w', [NArgs]).
plawk_dynrec_call_parts(dyncall_named(Name, Args), Args, ShimName) :-
    length(Args, NArgs),
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    format(atom(ShimName), 'plawk_dyncall_named_rec_~w', [Sym]).
% runtime-source record forms: Args (for the binfmt arg-field check) are
% just the call args -- the source path/handle is validated separately by
% the surface check. ShimName is unused here (bind_pair emits via
% plawk_dynrec_call_ir); left unbound-safe with a placeholder.
plawk_dynrec_call_parts(dyncall_at(_Source, Args), Args, at_rec).
plawk_dynrec_call_parts(dyncall_at_named(_Name, _Source, Args), Args,
    at_named_rec).

%% plawk_dynrec_call_ir(+Call, +FieldSeparator, +Base, -CallArgsIR,
%%     -Globals, -Setup, -ShimName)
%  The leading call arguments (as an LLVM arg-list string) plus the setup
%  lines, global constants, and record-shim name for a destructure bind.
%  Plain dyncall forms pass boxed %Value args; runtime-source (dyncall_at)
%  forms prepend the source `i8* path, i64 len` (lowered by
%  plawk_dyncall_source_ir) so the at-record shim can fetch/resolve the VM.
plawk_dynrec_call_ir(dyncall(Args), FS, Base, CallArgsIR, Globals, Setup,
        ShimName) :-
    length(Args, NArgs),
    format(atom(ShimName), 'plawk_dyncall_rec_~w', [NArgs]),
    plawk_foreign_args_ir(Args, FS, Base, ArgValueIRs, Globals, Setup),
    plawk_foreign_call_args_ir(ArgValueIRs, CallArgsIR).
plawk_dynrec_call_ir(dyncall_named(Name, Args), FS, Base, CallArgsIR, Globals,
        Setup, ShimName) :-
    length(Args, NArgs),
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    format(atom(ShimName), 'plawk_dyncall_named_rec_~w', [Sym]),
    plawk_foreign_args_ir(Args, FS, Base, ArgValueIRs, Globals, Setup),
    plawk_foreign_call_args_ir(ArgValueIRs, CallArgsIR).
plawk_dynrec_call_ir(dyncall_at(Source, Args), FS, Base, CallArgsIR, Globals,
        Setup, ShimName) :-
    length(Args, NArgs),
    format(atom(ShimName), 'plawk_dyncall_at_rec_~w', [NArgs]),
    plawk_dynrec_at_call_args(Source, Args, FS, Base, CallArgsIR, Globals,
        Setup).
plawk_dynrec_call_ir(dyncall_at_named(Name, Source, Args), FS, Base,
        CallArgsIR, Globals, Setup, ShimName) :-
    length(Args, NArgs),
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    format(atom(ShimName), 'plawk_dyncall_at_named_rec_~w', [Sym]),
    plawk_dynrec_at_call_args(Source, Args, FS, Base, CallArgsIR, Globals,
        Setup).

% Common source+args lowering for the at-record forms: `i8* path, i64 len`
% then the boxed %Value args, mirroring the blob(dyncall_at) call site.
plawk_dynrec_at_call_args(Source, Args, FS, Base, CallArgsIR, Globals,
        Setup) :-
    plawk_dyncall_source_ir(Source, FS, Base, Base, PathPtrIR, PathLenIR,
        SrcGlobals, SrcSetup),
    plawk_foreign_args_ir(Args, FS, Base, ArgValueIRs, ArgGlobals, ArgSetup),
    ( ArgValueIRs == []
    -> ArgsSuffix = ''
    ;  plawk_foreign_call_args_ir(ArgValueIRs, AV),
       format(atom(ArgsSuffix), ', ~w', [AV])
    ),
    format(atom(CallArgsIR), 'i8* ~w, i64 ~w~w',
        [PathPtrIR, PathLenIR, ArgsSuffix]),
    append(SrcGlobals, ArgGlobals, Globals),
    append(SrcSetup, ArgSetup, Setup).

%% plawk_dynassoc_call_parts(+Call, -Args, -ShimName)
%  The assoc shim for a named-entry `... as assoc` site (named entry only;
%  a default-entry `dyncall(...) as assoc` is a follow-on).
plawk_dynassoc_call_parts(dyncall_named(Name, Args), Args, ShimName) :-
    length(Args, NArgs),
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    format(atom(ShimName), 'plawk_dyncall_assoc_~w', [Sym]).
% default entry: arr = dyncall(args) as assoc -- the DYNLOAD object's
% wamo_entry, resolved like a plain dyncall (deferred small item).
plawk_dynassoc_call_parts(dyncall(Args), Args, ShimName) :-
    length(Args, NArgs),
    format(atom(ShimName), 'plawk_dyncall_assoc_default_~w', [NArgs]).
% str-valued table kind (`as assoc(str)`): the spec wraps the call in
% str(...); route to the @wam_object_call_assoc_str shims.
plawk_dynassoc_call_parts(str(dyncall_named(Name, Args)), Args, ShimName) :-
    length(Args, NArgs),
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    format(atom(ShimName), 'plawk_dyncall_assoc_str_~w', [Sym]).
plawk_dynassoc_call_parts(str(dyncall(Args)), Args, ShimName) :-
    length(Args, NArgs),
    format(atom(ShimName), 'plawk_dyncall_assoc_str_default_~w', [NArgs]).
% positional-array kind (`as array`): the spec wraps the call in
% posarray(...); route to the @wam_object_call_posarray shims.
plawk_dynassoc_call_parts(posarray(dyncall_named(Name, Args)), Args, ShimName) :-
    length(Args, NArgs),
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    format(atom(ShimName), 'plawk_dyncall_posarray_~w', [Sym]).
plawk_dynassoc_call_parts(posarray(dyncall(Args)), Args, ShimName) :-
    length(Args, NArgs),
    format(atom(ShimName), 'plawk_dyncall_posarray_default_~w', [NArgs]).
% str-valued positional array (`as array(str)`): posarray_str(...) wrapper.
plawk_dynassoc_call_parts(posarray_str(dyncall_named(Name, Args)), Args, ShimName) :-
    length(Args, NArgs),
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    format(atom(ShimName), 'plawk_dyncall_posarray_str_~w', [Sym]).
plawk_dynassoc_call_parts(posarray_str(dyncall(Args)), Args, ShimName) :-
    length(Args, NArgs),
    format(atom(ShimName), 'plawk_dyncall_posarray_str_default_~w', [NArgs]).

plawk_dynrec_field_load_lines([], [], _Base, []).
plawk_dynrec_field_load_lines([F | Fs], [Type | Ts], Base, [Line | Lines]) :-
    (   Type == f64
    ->  format(atom(Line),
            '  %~w_s~wp = getelementptr i64, i64* %~w_slots, i64 ~w\n  %~w_f~wbits = load i64, i64* %~w_s~wp\n  %~w_f~w = bitcast i64 %~w_f~wbits to double',
            [Base, F, Base, F, Base, F, Base, F, Base, F, Base, F])
    ;   Type == string
    ->  % load the atom pointer (kept as i64) and the length from out_lens
        format(atom(Line),
            '  %~w_s~wp = getelementptr i64, i64* %~w_slots, i64 ~w\n  %~w_f~w = load i64, i64* %~w_s~wp\n  %~w_l~wp = getelementptr i64, i64* %~w_lens, i64 ~w\n  %~w_l~w = load i64, i64* %~w_l~wp',
            [Base, F, Base, F, Base, F, Base, F, Base, F, Base, F, Base, F, Base, F])
    ;   format(atom(Line),
            '  %~w_s~wp = getelementptr i64, i64* %~w_slots, i64 ~w\n  %~w_f~w = load i64, i64* %~w_s~wp',
            [Base, F, Base, F, Base, F, Base, F])
    ),
    plawk_dynrec_field_load_lines(Fs, Ts, Base, Lines).

% Point each binding's threaded slot value(s) at its loaded field SSA. A
% numeric binding sets one slot to %Base_f<i>; a string binding str(P,L)
% sets P's slot to the pointer (%Base_f<i>) and L's to the length
% (%Base_l<i>).
plawk_dynrec_update_slots([], _I, _Base, _Slots, Values, Values).
plawk_dynrec_update_slots([str(P, L) | Rest], I, Base, Slots, Values0, Values) :-
    !,
    format(atom(PtrSSA), '%~w_f~w', [Base, I]),
    format(atom(LenSSA), '%~w_l~w', [Base, I]),
    plawk_dynrec_set_slot(P, PtrSSA, Slots, Values0, Values1),
    plawk_dynrec_set_slot(L, LenSSA, Slots, Values1, Values2),
    I1 is I + 1,
    plawk_dynrec_update_slots(Rest, I1, Base, Slots, Values2, Values).
plawk_dynrec_update_slots([Var | Rest], I, Base, Slots, Values0, Values) :-
    format(atom(SSA), '%~w_f~w', [Base, I]),
    plawk_dynrec_set_slot(Var, SSA, Slots, Values0, Values1),
    I1 is I + 1,
    plawk_dynrec_update_slots(Rest, I1, Base, Slots, Values1, Values).

plawk_dynrec_set_slot(Var, SSA, Slots, Values0, Values) :-
    nth0(SlotIndex, Slots, Slot),
    plawk_slot_name(Slot, Var),
    !,
    replace_nth0(SlotIndex, Values0, SSA, Values).

%% plawk_f64_call_tail_ir(+Base, +ResIR, +PreSetup, -ValueIR, -SetupParts)
%  Shared {double,i1} call tail: unpack value/ok and select 0.0 on failure.
plawk_f64_call_tail_ir(Base, ResIR, PreSetup, ValueIR, SetupParts) :-
    format(atom(ValIR),
        '  %~w_val = extractvalue { double, i1 } %~w_res, 0', [Base, Base]),
    format(atom(OkIR),
        '  %~w_ok = extractvalue { double, i1 } %~w_res, 1', [Base, Base]),
    format(atom(SelIR),
        '  %~w = select i1 %~w_ok, double %~w_val, double 0.0',
        [Base, Base, Base]),
    format(atom(ValueIR), '%~w', [Base]),
    append(PreSetup, [ResIR, ValIR, OkIR, SelIR], SetupParts).

plawk_foreign_arg(field(Index)) :-
    integer(Index),
    Index >= 0.
plawk_foreign_arg(string(String)) :-
    string(String).
plawk_foreign_arg(int(Value)) :-
    integer(Value).

%% plawk_i64_scalar_read_binary_expr(+Expr) is semidet.
%
%  Like plawk_i64_general_binary_expr but operands may also be scalar
%  variable reads (var/1). Only usable where the emitter has the current
%  slot values to substitute: scalar update expressions in rule bodies.
plawk_i64_scalar_read_binary_expr(Expr) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    plawk_i64_scalar_read_operand_expr(Left),
    plawk_i64_scalar_read_operand_expr(Right).

plawk_i64_scalar_read_operand_expr(var(Name)) :-
    atom(Name).
plawk_i64_scalar_read_operand_expr(Expr) :-
    plawk_i64_operand_expr(Expr).
plawk_i64_scalar_read_operand_expr(Expr) :-
    plawk_i64_scalar_read_binary_expr(Expr).

%% plawk_end_scalar_expr(+Expr) is semidet.
%
%  END-position i64 expression: a binary tree whose leaves are integer
%  literals, scalar variables (final slot values), or NR (the final
%  record count). Fields, NF, length etc. are meaningless after the
%  stream closes and are rejected.
plawk_end_scalar_expr(Expr) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    plawk_end_scalar_operand_expr(Left),
    plawk_end_scalar_operand_expr(Right).

plawk_end_scalar_operand_expr(int(Value)) :-
    integer(Value).
plawk_end_scalar_operand_expr(var(Name)) :-
    atom(Name).
plawk_end_scalar_operand_expr(special('NR')).
% Float literals make the whole END expression double-typed (%g print,
% IEEE fdiv), as does reading a double slot.
plawk_end_scalar_operand_expr(float_const(Mantissa, Denominator)) :-
    integer(Mantissa),
    integer(Denominator),
    Denominator > 0.
plawk_end_scalar_operand_expr(Expr) :-
    plawk_end_scalar_expr(Expr).

%% plawk_substitute_scalar_reads(+Expr0, +Slots, +Values, -Expr)
%
%  Replace var(Name) leaves with ssa(ValueIR) using the current scalar
%  slot values, so the shared i64 emitters never see a variable read.
plawk_substitute_operation_reads(add(Expr0), Slots, Values, add(Expr)) :-
    !,
    plawk_substitute_scalar_reads(Expr0, Slots, Values, Expr).
plawk_substitute_operation_reads(set(Expr0), Slots, Values, set(Expr)) :-
    !,
    plawk_substitute_scalar_reads(Expr0, Slots, Values, Expr).
% a string assignment's RHS may read a string scalar (`x = x $1` accumulation);
% substitute the reads to their current slot values (ssa_str) before the build.
plawk_substitute_operation_reads(set_str(Src0), Slots, Values, set_str(Src)) :-
    !,
    plawk_substitute_scalar_reads(Src0, Slots, Values, Src).
plawk_substitute_operation_reads(Operation, _Slots, _Values, Operation).

plawk_substitute_scalar_reads(concat(Parts), Slots, Values, concat(SubParts)) :-
    !,
    maplist(plawk_substitute_scalar_read_part(Slots, Values), Parts, SubParts).
plawk_substitute_scalar_read_part(Slots, Values, Part, SubPart) :-
    plawk_substitute_scalar_reads(Part, Slots, Values, SubPart).
plawk_substitute_scalar_reads(var(Name), Slots, Values, Substituted) :-
    !,
    nth0(SlotIndex, Slots, Slot),
    plawk_slot_name(Slot, Name),
    !,
    nth0(SlotIndex, Values, Value),
    ( Slot = scalar_double(_Name)
    -> Substituted = ssa_f64(Value)
    ;  Slot = scalar_string(_Name)
    -> Substituted = ssa_str(Value)
    ;  Slot = scalar_strnum(_Name)
    -> Substituted = ssa_strnum(Value)   % strnum: resolves to text for print,
                                         % coerces to a number in arithmetic
    ;  Substituted = ssa(Value)
    ).
plawk_substitute_print_field(Slots, Values, Field0, Field) :-
    plawk_substitute_scalar_reads(Field0, Slots, Values, Field).

plawk_substitute_scalar_reads(blob_slice_vars(A0, B0), Slots, Values,
        blob_slice_vars(A, B)) :-
    !,
    plawk_substitute_scalar_reads(A0, Slots, Values, A),
    plawk_substitute_scalar_reads(B0, Slots, Values, B).
% dyncall_at sources may hold a scalar HANDLE read (handle_src(var(h)))
% -- substitute it to the slot's SSA value so the source marshal sees a
% concrete i64, like any other scalar read.
plawk_substitute_scalar_reads(dyncall_at(Source0, Args), Slots, Values,
        dyncall_at(Source, Args)) :-
    !,
    plawk_substitute_handle_source(Source0, Slots, Values, Source).
plawk_substitute_scalar_reads(dyncall_at_named(Name, Source0, Args), Slots,
        Values, dyncall_at_named(Name, Source, Args)) :-
    !,
    plawk_substitute_handle_source(Source0, Slots, Values, Source).
plawk_substitute_scalar_reads(float_dyncall_at(Source0, Args), Slots, Values,
        float_dyncall_at(Source, Args)) :-
    !,
    plawk_substitute_handle_source(Source0, Slots, Values, Source).
plawk_substitute_scalar_reads(float_dyncall_at_named(Name, Source0, Args),
        Slots, Values, float_dyncall_at_named(Name, Source, Args)) :-
    !,
    plawk_substitute_handle_source(Source0, Slots, Values, Source).
plawk_substitute_scalar_reads(blob_dyncall_at(Source0, Args), Slots, Values,
        blob_dyncall_at(Source, Args)) :-
    !,
    plawk_substitute_handle_source(Source0, Slots, Values, Source).
plawk_substitute_scalar_reads(blob_dyncall_at_named(Name, Source0, Args),
        Slots, Values, blob_dyncall_at_named(Name, Source, Args)) :-
    !,
    plawk_substitute_handle_source(Source0, Slots, Values, Source).

plawk_substitute_handle_source(handle_src(var(Name)), Slots, Values,
        handle_src(Substituted)) :-
    !,
    plawk_substitute_scalar_reads(var(Name), Slots, Values, Substituted).
plawk_substitute_handle_source(Source, _Slots, _Values, Source).
plawk_substitute_scalar_reads(Expr0, Slots, Values, Expr) :-
    plawk_i64_binary_expr(Expr0, _LLVMOp, _NamePart, Left0, Right0),
    !,
    plawk_substitute_scalar_reads(Left0, Slots, Values, Left),
    plawk_substitute_scalar_reads(Right0, Slots, Values, Right),
    Expr0 =.. [Functor, _, _],
    Expr =.. [Functor, Left, Right].
plawk_substitute_scalar_reads(Expr, _Slots, _Values, Expr).

%% plawk_substitute_end_reads(+Expr0, +StatePlan, -Expr)
%
%  END-position substitution: var(Name) becomes the final slot value and
%  NR becomes %plawk_nr, the loop-head record phi, which dominates
%  end_print via close_stream / break_close_stream.
plawk_substitute_end_reads(var(Name), StatePlan, Substituted) :-
    !,
    plawk_state_slot_lookup(StatePlan, Name, SlotIndex, Slot),
    format(atom(Value), '%final_slot_~w', [SlotIndex]),
    ( Slot = scalar_double(_Name)
    -> Substituted = ssa_f64(Value)
    ;  Slot = scalar_strnum(_Name)
    -> Substituted = ssa_strnum(Value)   % strnum: text for print, number in arith
    ;  Substituted = ssa(Value)
    ).
plawk_substitute_end_reads(special('NR'), _StatePlan, ssa('%plawk_nr')) :-
    !.
plawk_substitute_end_reads(Expr0, StatePlan, Expr) :-
    plawk_i64_binary_expr(Expr0, _LLVMOp, _NamePart, Left0, Right0),
    !,
    plawk_substitute_end_reads(Left0, StatePlan, Left),
    plawk_substitute_end_reads(Right0, StatePlan, Right),
    Expr0 =.. [Functor, _, _],
    Expr =.. [Functor, Left, Right].
plawk_substitute_end_reads(Expr, _StatePlan, Expr).

plawk_i64_binary_primary_expr(special('NR')).
plawk_i64_binary_primary_expr(special('NF')).
plawk_i64_binary_primary_expr(special('ARGC')).
plawk_i64_binary_primary_expr(int(field(FieldIndex))) :-
    FieldIndex >= 0.
plawk_i64_binary_primary_expr(length(field(FieldIndex))) :-
    FieldIndex >= 0.
plawk_i64_binary_primary_expr(index(field(FieldIndex), string(Needle))) :-
    FieldIndex >= 0,
    string(Needle).

plawk_i64_scalar_primary_expr(special('NR')).
plawk_i64_scalar_primary_expr(special('NF')).
plawk_i64_scalar_primary_expr(int(field(FieldIndex))) :-
    FieldIndex >= 0.
plawk_i64_scalar_primary_expr(length(field(FieldIndex))) :-
    FieldIndex >= 0.
plawk_i64_scalar_primary_expr(index(field(FieldIndex), string(Needle))) :-
    FieldIndex >= 0,
    string(Needle).
plawk_i64_scalar_primary_expr(match_expr(field(FieldIndex), Regex)) :-
    FieldIndex >= 0,
    string(Regex).
plawk_i64_scalar_primary_expr(special('RSTART')).
plawk_i64_scalar_primary_expr(special('RLENGTH')).
plawk_i64_scalar_primary_expr(special('ARGC')).

plawk_i64_binary_expr(add_i64(Left, Right), add, add, Left, Right).
plawk_i64_binary_expr(sub_i64(Left, Right), sub, sub, Left, Right).
plawk_i64_binary_expr(mul_i64(Left, Right), mul, mul, Left, Right).
plawk_i64_binary_expr(div_i64(Left, Right), sdiv, div, Left, Right).
plawk_i64_binary_expr(mod_i64(Left, Right), srem, mod, Left, Right).

plawk_i64_binary_print_kind(add, int_add).
plawk_i64_binary_print_kind(sub, int_sub).
plawk_i64_binary_print_kind(mul, int_mul).
plawk_i64_binary_print_kind(div, int_div).
plawk_i64_binary_print_kind(mod, int_mod).

% Reads count as slot names too: a variable read before any write gets a
% zero-initialized slot, matching awk's uninitialized-variable semantics.
plawk_scalar_update_action_name(Action, Name) :-
    plawk_scalar_action_update(Action, WriteName, Operation),
    (   Name = WriteName
    ;   plawk_scalar_operation_expr(Operation, Expr),
        plawk_expr_scalar_read_name(Expr, Name)
    ).
% `n = gsub(...)` count capture: both the Target string scalar and the CountName
% i64 slot are written. action_update reports Target; this clause additionally
% surfaces CountName so it gets its own (i64) slot.
plawk_scalar_update_action_name(gsub_count(CountName, _Global, _Regex, _Repl, Target), Name) :-
    ( Name = Target ; Name = CountName ).
% `status = getline var < "file"` writes both the Var string scalar and the
% Status i64 slot; action_update reports Var, this surfaces Status too.
plawk_scalar_update_action_name(getline_capture(Status, Var, _File), Name) :-
    ( Name = Var ; Name = Status ).
plawk_scalar_update_action_name(dynrec_bind(Vars, _Call, _Types), Name) :-
    plawk_dynrec_binding_names(Vars, Names),
    member(Name, Names).
plawk_scalar_update_action_name(if(_Pattern, ThenActions, ElseActions), Name) :-
    ( member(Action, ThenActions)
    ; member(Action, ElseActions)
    ),
    plawk_scalar_update_action_name(Action, Name).
plawk_scalar_update_action_name(if(scalar_if(Cond), _Then, _Else), Name) :-
    plawk_while_cond_vars(Cond, CondVars),
    member(Name, CondVars).
plawk_scalar_update_action_name(foreach_loop(_Layout, Body), Name) :-
    member(Action, Body),
    plawk_scalar_update_action_name(Action, Name).
plawk_scalar_update_action_name(while_loop(Cond, Body), Name) :-
    ( plawk_while_cond_vars(Cond, CondVars), member(Name, CondVars)
    ; member(Action, Body), plawk_scalar_update_action_name(Action, Name)
    ).
plawk_scalar_update_action_name(do_while_loop(Body, Cond), Name) :-
    ( plawk_while_cond_vars(Cond, CondVars), member(Name, CondVars)
    ; member(Action, Body), plawk_scalar_update_action_name(Action, Name)
    ).

plawk_expr_scalar_read_name(var(Name), Name).
plawk_expr_scalar_read_name(Expr, Name) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    ( plawk_expr_scalar_read_name(Left, Name)
    ; plawk_expr_scalar_read_name(Right, Name)
    ).

plawk_scalar_action_update(inc(var(Name)), Name, add(const(1))).
plawk_scalar_action_update(add(var(Name), int(Value)), Name, add(const(Value))) :-
    integer(Value),
    Value >= 0.
plawk_scalar_action_update(add(var(Name), length(field(FieldIndex))), Name, add(length(FieldIndex))) :-
    FieldIndex >= 0.
plawk_scalar_action_update(add(var(Name), field(FieldIndex)), Name, add(field_i64(FieldIndex))) :-
    FieldIndex >= 0.
plawk_scalar_action_update(add(var(Name), int(field(FieldIndex))), Name, add(field_i64(FieldIndex))) :-
    FieldIndex >= 0.
plawk_scalar_action_update(add(var(Name), Expr), Name, add(Expr)) :-
    plawk_i64_scalar_primary_expr(Expr).
plawk_scalar_action_update(add(var(Name), Expr), Name, add(Expr)) :-
    plawk_i64_scalar_read_binary_expr(Expr).
plawk_scalar_action_update(add(var(Name), var(Read)), Name, add(var(Read))) :-
    atom(Read).
plawk_scalar_action_update(add(var(Name), prolog_call(Pred, Args)), Name,
        add(prolog_call(Pred, Args))) :-
    plawk_prolog_call_expr(prolog_call(Pred, Args)).
plawk_scalar_action_update(add(var(Name), dyncall(Args)), Name,
        add(dyncall(Args))) :-
    plawk_dyncall_expr(dyncall(Args)).
plawk_scalar_action_update(add(var(Name), dyncall_named(E, Args)), Name,
        add(dyncall_named(E, Args))) :-
    plawk_dyncall_named_expr(dyncall_named(E, Args)).
plawk_scalar_action_update(add(var(Name), dyncall_at(Source, Args)), Name,
        add(dyncall_at(Source, Args))) :-
    plawk_dyncall_at_expr(dyncall_at(Source, Args)).
plawk_scalar_action_update(add(var(Name), dyncall_at_named(E, Source, Args)),
        Name, add(dyncall_at_named(E, Source, Args))) :-
    plawk_dyncall_at_expr(dyncall_at_named(E, Source, Args)).
% String-valued scalar assignment: `x = $1 $2` (concat) or `x = "text"`. The
% slot holds an interned atom id (an i64); the RHS is built into a buffer and
% interned at assignment, and `print x` resolves the id back to text. Recognised
% before the numeric clauses so a string RHS types the slot as scalar_string.
plawk_scalar_action_update(set(var(Name), concat(Parts)), Name, set_str(concat(Parts))) :-
    Parts = [_, _ | _],
    maplist(plawk_str_scalar_part_ok, Parts).
plawk_scalar_action_update(set(var(Name), string(Value)), Name, set_str(string(Value))) :-
    string(Value).
% `x = ENVIRON["NAME"]`: the env var value as a string scalar (via getenv).
plawk_scalar_action_update(set(var(Name), environ(Key)), Name, set_str(environ(Key))) :-
    string(Key).
% `x = ARGV[N]`: the N-th command-line argument as a string scalar.
plawk_scalar_action_update(set(var(Name), argv_at(N)), Name, set_str(argv_at(N))) :-
    integer(N), N >= 0.
% `x = sprintf("fmt", args)`: format into a string scalar.
plawk_scalar_action_update(set(var(Name), sprintf(string(Format), Args)), Name,
        set_str(sprintf(string(Format), Args))) :-
    string(Format),
    is_list(Args).
% `gsub(/re/, "repl", var)` / `sub(...)`: substitute in place into the string
% scalar `var`. Types `var` as a string scalar (set_str); the substitution reads
% the slot's current interned value and re-interns the result.
plawk_scalar_action_update(regex_sub_var(Global, Regex, Repl, Name), Name,
        set_str(gsub_str(Global, Regex, Repl))) :-
    integer(Global),
    string(Regex),
    string(Repl).
% `n = gsub(/re/, "repl", var)`: count capture. The Target string scalar is
% substituted in place (same set_str(gsub_str(...)) operation as regex_sub_var);
% the dedicated action-sequence clause additionally writes the substitution
% count into the CountName i64 slot. Reported here as a Target string update so
% the state plan types Target as scalar_string.
plawk_scalar_action_update(gsub_count(_CountName, Global, Regex, Repl, Target), Target,
        set_str(gsub_str(Global, Regex, Repl))) :-
    integer(Global),
    string(Regex),
    string(Repl).
% `getline var < "file"` (and the count-capturing assignment): Var is a string
% scalar holding the interned line. Reported as a Target string update so the
% state plan types Var as scalar_string and the driver accepts the program; the
% dedicated action-sequence clauses below emit the actual getline IR.
plawk_scalar_action_update(getline_read(Var, File), Var, set_str(getline(File))) :-
    string(File).
plawk_scalar_action_update(getline_capture(_Status, Var, File), Var, set_str(getline(File))) :-
    string(File).
% Ternary assignment `x = COND ? A : B`: an i64 value via select (the operation
% lowers through plawk_scalar_numeric_expr_ir(ternary(...)) -> plawk_i64_expr_ir).
plawk_scalar_action_update(set(var(Name), ternary(cmp(Left, _Op, Right), Then, Else)),
        Name, set(ternary(cmp(Left, _Op, Right), Then, Else))) :-
    plawk_ternary_i64_operand_ok(Left),
    plawk_ternary_i64_operand_ok(Right),
    plawk_ternary_i64_operand_ok(Then),
    plawk_ternary_i64_operand_ok(Else).
plawk_scalar_action_update(set(var(Name), int(Value)), Name, set(const(Value))) :-
    integer(Value),
    Value >= 0.
plawk_scalar_action_update(set(var(Name), length(field(FieldIndex))), Name, set(length(FieldIndex))) :-
    FieldIndex >= 0.
plawk_scalar_action_update(set(var(Name), field(FieldIndex)), Name, set(field_i64(FieldIndex))) :-
    FieldIndex >= 0.
plawk_scalar_action_update(set(var(Name), int(field(FieldIndex))), Name, set(field_i64(FieldIndex))) :-
    FieldIndex >= 0.
plawk_scalar_action_update(set(var(Name), Expr), Name, set(Expr)) :-
    plawk_i64_scalar_primary_expr(Expr).
plawk_scalar_action_update(set(var(Name), Expr), Name, set(Expr)) :-
    plawk_i64_scalar_read_binary_expr(Expr).
plawk_scalar_action_update(set(var(Name), var(Read)), Name, set(var(Read))) :-
    atom(Read).
plawk_scalar_action_update(set(var(Name), prolog_call(Pred, Args)), Name,
        set(prolog_call(Pred, Args))) :-
    plawk_prolog_call_expr(prolog_call(Pred, Args)).
plawk_scalar_action_update(set(var(Name), dyncall(Args)), Name,
        set(dyncall(Args))) :-
    plawk_dyncall_expr(dyncall(Args)).
plawk_scalar_action_update(set(var(Name), dyncall_named(E, Args)), Name,
        set(dyncall_named(E, Args))) :-
    plawk_dyncall_named_expr(dyncall_named(E, Args)).
plawk_scalar_action_update(set(var(Name), dyncall_at(Source, Args)), Name,
        set(dyncall_at(Source, Args))) :-
    plawk_dyncall_at_expr(dyncall_at(Source, Args)).
plawk_scalar_action_update(set(var(Name), dyncall_at_named(E, Source, Args)),
        Name, set(dyncall_at_named(E, Source, Args))) :-
    plawk_dyncall_at_expr(dyncall_at_named(E, Source, Args)).
% h = compile(src) / compile_file(path): store the grammar handle in a
% scalar (only set makes sense -- a handle is an opaque registry index).
plawk_scalar_action_update(set(var(Name), compile_handle(Arg)), Name,
        set(compile_handle(Arg))) :-
    plawk_foreign_arg(Arg).
plawk_scalar_action_update(set(var(Name), compile_file_handle(Arg)), Name,
        set(compile_file_handle(Arg))) :-
    plawk_foreign_arg(Arg).

% A string-scalar concat part: a field (`$N`, projected as text), a string
% literal (embedded in the build format), or a scalar-variable read (a string
% scalar, resolved to text -- enables accumulation `x = x $1`). A numeric var in
% a string concat is the caller's responsibility (it is read as an atom id).
plawk_str_scalar_part_ok(field(Index)) :- integer(Index), Index >= 0.
plawk_str_scalar_part_ok(string(Value)) :- string(Value).
plawk_str_scalar_part_ok(var(Name)) :- atom(Name).
% Double-typed update expressions: float literals, float($N), and
% binary trees mixing them with i64 operands or scalar reads. The
% written slot becomes a double via plawk_scalar_typed_slots/3.
plawk_scalar_action_update(add(var(Name), Expr), Name, add(Expr)) :-
    plawk_f64_scalar_update_expr(Expr).
plawk_scalar_action_update(set(var(Name), Expr), Name, set(Expr)) :-
    plawk_f64_scalar_update_expr(Expr).

plawk_f64_scalar_update_expr(Expr) :-
    plawk_expr_is_double(Expr),
    plawk_f64_scalar_read_operand_expr(Expr).

plawk_f64_scalar_read_operand_expr(var(Name)) :-
    atom(Name).
plawk_f64_scalar_read_operand_expr(float_const(Mantissa, Denominator)) :-
    integer(Mantissa),
    integer(Denominator),
    Denominator > 0.
plawk_f64_scalar_read_operand_expr(float_field(Index)) :-
    integer(Index),
    Index >= 0.
plawk_f64_scalar_read_operand_expr(float_call(Name, Args)) :-
    plawk_prolog_call_expr(prolog_call(Name, Args)).
plawk_f64_scalar_read_operand_expr(float_dyncall(Args)) :-
    plawk_float_dyncall_expr(float_dyncall(Args)).
plawk_f64_scalar_read_operand_expr(float_dyncall_named(Name, Args)) :-
    plawk_float_dyncall_named_expr(float_dyncall_named(Name, Args)).
plawk_f64_scalar_read_operand_expr(float_dyncall_at(Source, Args)) :-
    plawk_float_dyncall_at_expr(float_dyncall_at(Source, Args)).
plawk_f64_scalar_read_operand_expr(float_dyncall_at_named(Name, Source, Args)) :-
    plawk_float_dyncall_at_expr(float_dyncall_at_named(Name, Source, Args)).
plawk_f64_scalar_read_operand_expr(Expr) :-
    plawk_i64_operand_expr(Expr).
plawk_f64_scalar_read_operand_expr(Expr) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    plawk_f64_scalar_read_operand_expr(Left),
    plawk_f64_scalar_read_operand_expr(Right).

plawk_scalar_action_sequence_pairs([], _Slots, _AssocPlan, _FieldSeparator, _OutputSeparator, _Prefix, CurrentLabel, _RuleIndex,
        OpIndex, Values, Values, OpIndex, CurrentLabel, []) -->
    [].
plawk_scalar_action_sequence_pairs([dynrec_bind(Vars, Call, Types) | Rest], Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { plawk_dynrec_bind_ok(dynrec_bind(Vars, Call, Types)),
      format(atom(Base), '~w_dynrec_~w', [Prefix, OpIndex]),
      plawk_dynrec_bind_pair(Vars, Call, Types, FieldSeparator, Base, Slots,
          Values0, Values1, Pair),
      NextOpIndex is OpIndex + 1
    },
    [Pair],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        NextOpIndex, Values1, Values, FinalOpIndex, ExitLabel, NextExits).
% `n = gsub(/re/, "repl", var)` count capture: a dual-slot write. The Target
% string scalar is substituted in place (same set_str(gsub_str(...)) IR as a
% bare `gsub(...,var)`), which also stores the substitution count into the
% shared @plawk_gsub_count global; a following load moves that count into the
% CountName i64 slot. Placed before the generic set-action clause so its
% specific head wins.
% Emit the getline IR for one site: the filename constant, an alloca for the
% line id, the call to @wam_getline_file (which keys the open handle by filename
% in a process-wide registry, so sites reading the same file share it), and the
% awk EOF-preserve select (Var keeps its old value unless a line was actually
% read). VarNext / StNext are the new Var / status SSA values. GlobalIR carries
% just the filename constant.
plawk_getline_ir(Prefix, OpIndex, File, VarInput, VarNext, StNext, GlobalIR, IR) :-
    format(atom(Base), '~w_getline_~w', [Prefix, OpIndex]),
    format(atom(PathName), '~w_path', [Base]),
    llvm_emit_c_string_global(PathName, File, GlobalIR, PathLen, PathBytes),
    format(atom(StNext), '%~w_status', [Base]),
    format(atom(VarNext), '%~w_var', [Base]),
    format(atom(IR),
'  %~w_pathp = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0
  %~w_lineid = alloca i64
  ~w = call i64 @wam_getline_file(i8* %~w_pathp, i64 ~w, i64* %~w_lineid)
  %~w_line = load i64, i64* %~w_lineid
  %~w_got = icmp eq i64 ~w, 1
  ~w = select i1 %~w_got, i64 %~w_line, i64 ~w',
        [Base, PathBytes, PathBytes, PathName,
         Base,
         StNext, Base, PathLen, Base,
         Base, Base,
         Base, StNext,
         VarNext, Base, Base, VarInput]).

% `status = getline var < "file"`: read the next line into the Var string slot
% and the 1/0/-1 status into the Status i64 slot (a dual-slot write). The file is
% opened lazily and advanced one line per call via a per-site handle global.
plawk_scalar_action_sequence_pairs([getline_capture(Status, Var, File) | Rest], Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { string(File),
      nth0(VarIndex, Slots, VarSlot), plawk_slot_name(VarSlot, Var),
      nth0(VarIndex, Values0, VarInput),
      nth0(StIndex, Slots, StSlot), plawk_slot_name(StSlot, Status),
      plawk_getline_ir(Prefix, OpIndex, File, VarInput, VarNext, StNext, GlobalIR, IR),
      replace_nth0(VarIndex, Values0, VarNext, Values1),
      replace_nth0(StIndex, Values1, StNext, Values2),
      NextOpIndex is OpIndex + 1
    },
    [GlobalIR-IR],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        NextOpIndex, Values2, Values, FinalOpIndex, ExitLabel, NextExits).
% bare `getline var < "file"`: same, discarding the status.
plawk_scalar_action_sequence_pairs([getline_read(Var, File) | Rest], Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { string(File),
      nth0(VarIndex, Slots, VarSlot), plawk_slot_name(VarSlot, Var),
      nth0(VarIndex, Values0, VarInput),
      plawk_getline_ir(Prefix, OpIndex, File, VarInput, VarNext, _StNext, GlobalIR, IR),
      replace_nth0(VarIndex, Values0, VarNext, Values1),
      NextOpIndex is OpIndex + 1
    },
    [GlobalIR-IR],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        NextOpIndex, Values1, Values, FinalOpIndex, ExitLabel, NextExits).
plawk_scalar_action_sequence_pairs([gsub_count(CountName, Global, Regex, Repl, Target) | Rest], Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { integer(Global), string(Regex), string(Repl),
      nth0(TargetIndex, Slots, TargetSlot),
      plawk_slot_name(TargetSlot, Target),
      nth0(TargetIndex, Values0, TargetInput),
      plawk_scalar_update_operation_ir(set_str(gsub_str(Global, Regex, Repl)),
          TargetSlot, FieldSeparator, Prefix, TargetIndex, OpIndex,
          TargetInput, TargetNext, GsubPair),
      replace_nth0(TargetIndex, Values0, TargetNext, Values1),
      nth0(CountIndex, Slots, CountSlot),
      plawk_slot_name(CountSlot, CountName),
      format(atom(CountNext), '%~w_gsubcount_~w', [Prefix, OpIndex]),
      format(atom(CountLine),
          '  ~w = load i64, i64* @plawk_gsub_count', [CountNext]),
      replace_nth0(CountIndex, Values1, CountNext, Values2),
      NextOpIndex is OpIndex + 1
    },
    [GsubPair, ''-CountLine],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        NextOpIndex, Values2, Values, FinalOpIndex, ExitLabel, NextExits).
plawk_scalar_action_sequence_pairs([Action | Rest], Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { plawk_scalar_action_update(Action, Name, Operation0),
      nth0(SlotIndex, Slots, Slot),
      plawk_slot_name(Slot, Name),
      nth0(SlotIndex, Values0, InputValue),
      plawk_substitute_operation_reads(Operation0, Slots, Values0, Operation),
      plawk_scalar_update_operation_ir(Operation, Slot, FieldSeparator, Prefix, SlotIndex,
          OpIndex, InputValue, NextValue, Pair),
      replace_nth0(SlotIndex, Values0, NextValue, Values1),
      NextOpIndex is OpIndex + 1
    },
    [Pair],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        NextOpIndex, Values1, Values, FinalOpIndex, ExitLabel, NextExits).
plawk_scalar_action_sequence_pairs([Action | Rest], Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, _CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { plawk_assoc_increment_action(Action, ArrayName-KeyIndex),
      plawk_assoc_table_index(AssocPlan, ArrayName, TableIndex),
      plawk_assoc_update_operation_ir(Prefix, OpIndex, TableIndex, KeyIndex,
          FieldSeparator, Pair, AssocExitLabel),
      NextOpIndex is OpIndex + 1
    },
    [Pair],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, AssocExitLabel, RuleIndex,
        NextOpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits).
plawk_scalar_action_sequence_pairs([writebin_out(Types, Fields) | Rest], Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { format(atom(WbPrefix), '~w_wb_~w', [Prefix, OpIndex]),
      plawk_writebin_record_ir(Types, Fields, Slots, Values0, FieldSeparator,
          WbPrefix, Pair),
      NextOpIndex is OpIndex + 1
    },
    [Pair],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        NextOpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits).
plawk_scalar_action_sequence_pairs([writebin_arm_out(Tag, ArmTypes, Fields) | Rest], Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { format(atom(WbPrefix), '~w_wba_~w', [Prefix, OpIndex]),
      plawk_writebin_union_record_ir(Tag, ArmTypes, Fields, Slots, Values0,
          FieldSeparator, WbPrefix, Pair),
      NextOpIndex is OpIndex + 1
    },
    [Pair],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        NextOpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits).
plawk_scalar_action_sequence_pairs([print(Fields) | Rest], Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { plawk_rule_body_print_action(print(Fields)),
      % substitute scalar-slot reads with their threaded SSA values so a
      % print can reference a scalar (e.g. a record-view string field's
      % (ptr,len) slice temps); idempotent for field/literal print items.
      maplist(plawk_substitute_print_field(Slots, Values0), Fields, SubFields),
      format(atom(PrintPrefix), '~w_print_~w', [Prefix, OpIndex]),
      plawk_prefixed_print_action_ir(SubFields, FieldSeparator, OutputSeparator, PrintPrefix, Pair),
      NextOpIndex is OpIndex + 1
    },
    [Pair],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        NextOpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits).
plawk_scalar_action_sequence_pairs([printf(string(Format), Args) | Rest],
        Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { plawk_rule_body_print_action(printf(string(Format), Args)),
      format(atom(PrintPrefix), '~w_printf_~w', [Prefix, OpIndex]),
      plawk_prefixed_printf_action_ir(Format, Args, FieldSeparator, PrintPrefix, Pair),
      NextOpIndex is OpIndex + 1
    },
    [Pair],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        NextOpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits).
plawk_scalar_action_sequence_pairs([next], _Slots, _AssocPlan, _FieldSeparator, _OutputSeparator, _Prefix, CurrentLabel, _RuleIndex,
        OpIndex, Values, Values, OpIndex, none, [branch_next(CurrentLabel, Values)]) -->
    [].
plawk_scalar_action_sequence_pairs([break], _Slots, _AssocPlan, _FieldSeparator, _OutputSeparator, _Prefix, CurrentLabel, _RuleIndex,
        OpIndex, Values, Values, OpIndex, break, [branch_break(CurrentLabel, Values)]) -->
    [].
% `continue` -- like break, it emits no IR of its own; the consumer branches
% (plawk_branch_to_done_ir) to the enclosing loop's continue target, and the
% loop consumes the branch_continue exit to wire its head/cond phi.
plawk_scalar_action_sequence_pairs([continue], _Slots, _AssocPlan, _FieldSeparator, _OutputSeparator, _Prefix, CurrentLabel, _RuleIndex,
        OpIndex, Values, Values, OpIndex, continue, [branch_continue(CurrentLabel, Values)]) -->
    [].
% `exit N` -- store the exit code, then leave the program via break_close_stream
% (which runs END and returns the code). The branch_exit exit carries the slot
% values at the exit point so END sees them (merged into the break_close phi
% exactly like a break), but a loop does NOT consume it -- exit always ends the
% program, so it propagates past any enclosing loop.
plawk_scalar_action_sequence_pairs([exit(int(Code))], _Slots, _AssocPlan, _FieldSeparator, _OutputSeparator, _Prefix, CurrentLabel, _RuleIndex,
        OpIndex, Values, Values, OpIndex, exit, [branch_exit(CurrentLabel, Values)]) -->
    { format(atom(StoreIR), '  store i32 ~w, i32* @plawk_exit_code', [Code]) },
    [''-StoreIR].
% `exit_store(N)` -- the store-only half of a rule-level `exit N` (the terminal
% control split leaves this in the body; the branch to break_close_stream comes
% from the terminal_exit rule target). It emits just the store and falls through
% -- control reaches the rule's done block, which branches to break_close_stream.
plawk_scalar_action_sequence_pairs([exit_store(Code) | Rest], Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { format(atom(StoreIR), '  store i32 ~w, i32* @plawk_exit_code', [Code]),
      NextOpIndex is OpIndex + 1 },
    [''-StoreIR],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, CurrentLabel, RuleIndex,
        NextOpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits).
% foreach_loop: the one loop in the emitter stack. Loop-carried phis
% for the element index and every scalar slot; the body memcpys the
% current element into the staging area and runs one copy of the
% actions (inner ifs, prints, updates, next/break all reuse the
% existing machinery). Exit values are the head phis themselves.
plawk_scalar_action_sequence_pairs([foreach_loop(Layout, Body) | Rest],
        Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, _CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { Layout = foreach_layout(CountOff, ElemsOff, ElemSize, StageOff, _StageBase),
      format(atom(FeBase), '~w_fe_~w', [Prefix, OpIndex]),
      format(atom(EntryLabel), '~w_entry', [FeBase]),
      format(atom(HeadLabel), '~w_head', [FeBase]),
      format(atom(BodyLabel), '~w_body', [FeBase]),
      format(atom(BodyDoneLabel), '~w_body_done', [FeBase]),
      format(atom(AfterLabel), '~w_after', [FeBase]),
      % head phi names double as the slot values inside and after the loop
      findall(PhiValue,
          ( nth0(SlotIndex, Slots, _Slot),
            format(atom(PhiValue), '%~w_slot_~w', [FeBase, SlotIndex])
          ),
          HeadValues),
      phrase(plawk_scalar_action_sequence_pairs(Body, Slots, AssocPlan,
          FieldSeparator, OutputSeparator, FeBase, BodyLabel, RuleIndex, 0,
          HeadValues, BodyOutValues, _InnerOpIndex, InnerExitLabel,
          InnerNextExits), BodyPairs),
      pairs_keys_values(BodyPairs, BodyGlobalParts, BodyLineParts),
      atomic_list_concat(BodyGlobalParts, '\n', GlobalIR),
      atomic_list_concat(BodyLineParts, '\n', BodyIR),
      plawk_branch_to_done_ir(InnerExitLabel, BodyDoneLabel, BodyDoneBrIR),
      phrase(plawk_foreach_head_phi_lines(Slots, Values0, BodyOutValues,
          FeBase, EntryLabel, BodyDoneLabel, 0), HeadPhiLines),
      atomic_list_concat(HeadPhiLines, '\n', HeadPhiIR),
      format(atom(IR),
'  br label %~w

~w:
  %~w_count_p = getelementptr i8, i8* %rec, i64 ~w
  %~w_count_tp = bitcast i8* %~w_count_p to i64*
  %~w_count = load i64, i64* %~w_count_tp
  br label %~w

~w:
  %~w_j = phi i64 [1, %~w], [%~w_j_next, %~w]
~w
  %~w_cont = icmp sle i64 %~w_j, %~w_count
  br i1 %~w_cont, label %~w, label %~w

~w:
  %~w_jm1 = add i64 %~w_j, -1
  %~w_rel = mul i64 %~w_jm1, ~w
  %~w_off = add i64 %~w_rel, ~w
  %~w_src = getelementptr i8, i8* %rec, i64 %~w_off
  %~w_dst = getelementptr i8, i8* %rec, i64 ~w
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %~w_dst, i8* %~w_src, i64 ~w, i1 false)
~w
~w

~w:
  %~w_j_next = add i64 %~w_j, 1
  br label %~w

~w:',
          [EntryLabel,
           EntryLabel,
           FeBase, CountOff,
           FeBase, FeBase,
           FeBase, FeBase,
           HeadLabel,
           HeadLabel,
           FeBase, EntryLabel, FeBase, BodyDoneLabel,
           HeadPhiIR,
           FeBase, FeBase, FeBase,
           FeBase, BodyLabel, AfterLabel,
           BodyLabel,
           FeBase, FeBase,
           FeBase, FeBase, ElemSize,
           FeBase, FeBase, ElemsOff,
           FeBase, FeBase,
           FeBase, StageOff,
           FeBase, FeBase, ElemSize,
           BodyIR,
           BodyDoneBrIR,
           BodyDoneLabel,
           FeBase, FeBase,
           HeadLabel,
           AfterLabel]),
      NextOpIndex is OpIndex + 1
    },
    [GlobalIR-IR],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, AfterLabel, RuleIndex,
        NextOpIndex, HeadValues, Values, FinalOpIndex, ExitLabel, RestNextExits),
    { append(InnerNextExits, RestNextExits, NextExits) }.
% while_loop: a surface loop. Loop-carried head phis for every scalar slot
% (modelled on foreach_loop's head-phi machinery -- no memory slots needed).
% The condition tests the condition variable's head-phi value BEFORE the body,
% so the loop may run zero times; the exit values are the head phis themselves.
% (PLAWK_CONTROL_FLOW_PLAN.md PR 2.)
plawk_scalar_action_sequence_pairs([while_loop(Cond, Body) | Rest],
        Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, _CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { format(atom(Base), '~w_while_~w', [Prefix, OpIndex]),
      format(atom(EntryLabel), '~w_entry', [Base]),
      format(atom(HeadLabel), '~w_head', [Base]),
      format(atom(BodyLabel), '~w_body', [Base]),
      format(atom(BodyDoneLabel), '~w_body_done', [Base]),
      format(atom(AfterLabel), '~w_after', [Base]),
      findall(PhiValue,
          ( nth0(SlotIndex, Slots, _Slot),
            format(atom(PhiValue), '%~w_slot_~w', [Base, SlotIndex])
          ),
          HeadValues),
      % break -> after, continue -> head (re-test): push the loop context so a
      % break/continue anywhere in the body branches to these labels.
      plawk_loopctx_push(loop_ctx(AfterLabel, HeadLabel)),
      phrase(plawk_scalar_action_sequence_pairs(Body, Slots, AssocPlan,
          FieldSeparator, OutputSeparator, Base, BodyLabel, RuleIndex, 0,
          HeadValues, BodyOutValues, _InnerOpIndex, InnerExitLabel,
          InnerNextExits), BodyPairs),
      pairs_keys_values(BodyPairs, BodyGlobalParts, BodyLineParts),
      atomic_list_concat(BodyGlobalParts, '\n', GlobalIR),
      atomic_list_concat(BodyLineParts, '\n', BodyIR),
      plawk_branch_to_done_ir(InnerExitLabel, BodyDoneLabel, BodyDoneBrIR),
      plawk_loopctx_pop,
      plawk_partition_loop_exits(InnerNextExits, Breaks, Continues, RestExits),
      % head phi carries continue values; the after phi carries break values
      plawk_while_head_phi_ir(Slots, Values0, BodyOutValues, Continues,
          Base, EntryLabel, BodyDoneLabel, HeadPhiIR),
      plawk_while_cond_ir(Cond, Slots, HeadValues, FieldSeparator, Base, CondVar, CondIR),
      plawk_loop_after_ir(Slots, HeadValues, HeadLabel, Breaks, Base,
          AfterValues, AfterPhiIR),
      format(atom(IR),
'  br label %~w

~w:
  br label %~w

~w:
~w
~w
  br i1 ~w, label %~w, label %~w

~w:
~w
~w

~w:
  br label %~w

~w:
~w',
          [EntryLabel,
           EntryLabel,
           HeadLabel,
           HeadLabel,
           HeadPhiIR,
           CondIR,
           CondVar, BodyLabel, AfterLabel,
           BodyLabel,
           BodyIR,
           BodyDoneBrIR,
           BodyDoneLabel,
           HeadLabel,
           AfterLabel,
           AfterPhiIR]),
      NextOpIndex is OpIndex + 1
    },
    [GlobalIR-IR],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, AfterLabel, RuleIndex,
        NextOpIndex, AfterValues, Values, FinalOpIndex, ExitLabel, RestNextExits),
    { append(RestExits, RestNextExits, NextExits) }.
% do_while_loop: like while_loop but the condition tests AFTER the body, so the
% body always runs at least once. The head phi sits at the top of the body
% block; the condition reads the body's OUTPUT values (the exit values), since
% control reaches `after` from the post-body condition test.
plawk_scalar_action_sequence_pairs([do_while_loop(Body, Cond) | Rest],
        Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, _CurrentLabel, RuleIndex,
        OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { format(atom(Base), '~w_dowhile_~w', [Prefix, OpIndex]),
      format(atom(EntryLabel), '~w_entry', [Base]),
      format(atom(BodyLabel), '~w_body', [Base]),
      format(atom(BodyDoneLabel), '~w_body_done', [Base]),
      format(atom(AfterLabel), '~w_after', [Base]),
      findall(PhiValue,
          ( nth0(SlotIndex, Slots, _Slot),
            format(atom(PhiValue), '%~w_slot_~w', [Base, SlotIndex])
          ),
          HeadValues),
      % break -> after, continue -> body_done (which re-tests the condition)
      plawk_loopctx_push(loop_ctx(AfterLabel, BodyDoneLabel)),
      phrase(plawk_scalar_action_sequence_pairs(Body, Slots, AssocPlan,
          FieldSeparator, OutputSeparator, Base, BodyLabel, RuleIndex, 0,
          HeadValues, BodyOutValues, _InnerOpIndex, InnerExitLabel,
          InnerNextExits), BodyPairs),
      pairs_keys_values(BodyPairs, BodyGlobalParts, BodyLineParts),
      atomic_list_concat(BodyGlobalParts, '\n', GlobalIR),
      atomic_list_concat(BodyLineParts, '\n', BodyIR),
      plawk_branch_to_done_ir(InnerExitLabel, BodyDoneLabel, BodyDoneBrIR),
      plawk_loopctx_pop,
      plawk_partition_loop_exits(InnerNextExits, Breaks, Continues, RestExits),
      % body_done merges the normal body output (from the body's exit block) with
      % each continue point, and the condition tests that merged value; the head
      % phi's back edge and the `after` block read it too.
      format(atom(BdBase), '~w_bd', [Base]),
      plawk_loop_after_ir(Slots, BodyOutValues, InnerExitLabel, Continues,
          BdBase, BdValues, BdPhiIR),
      phrase(plawk_foreach_head_phi_lines(Slots, Values0, BdValues,
          Base, EntryLabel, BodyDoneLabel, 0), HeadPhiLines),
      atomic_list_concat(HeadPhiLines, '\n', HeadPhiIR),
      plawk_while_cond_ir(Cond, Slots, BdValues, FieldSeparator, Base, CondVar, CondIR),
      plawk_loop_after_ir(Slots, BdValues, BodyDoneLabel, Breaks, Base,
          AfterValues, AfterPhiIR),
      format(atom(IR),
'  br label %~w

~w:
  br label %~w

~w:
~w
~w
~w

~w:
~w
~w
  br i1 ~w, label %~w, label %~w

~w:
~w',
          [EntryLabel,
           EntryLabel,
           BodyLabel,
           BodyLabel,
           HeadPhiIR,
           BodyIR,
           BodyDoneBrIR,
           BodyDoneLabel,
           BdPhiIR,
           CondIR,
           CondVar, BodyLabel, AfterLabel,
           AfterLabel,
           AfterPhiIR]),
      NextOpIndex is OpIndex + 1
    },
    [GlobalIR-IR],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, AfterLabel, RuleIndex,
        NextOpIndex, AfterValues, Values, FinalOpIndex, ExitLabel, RestNextExits),
    { append(RestExits, RestNextExits, NextExits) }.
plawk_scalar_action_sequence_pairs([if(Pattern, ThenActions, ElseActions) | Rest],
        Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, _CurrentLabel, RuleIndex, OpIndex, Values0, Values, FinalOpIndex, ExitLabel, NextExits) -->
    { format(atom(GlobalBase), '~w_if_~w', [Prefix, OpIndex]),
      plawk_if_cond_ir(Pattern, Slots, Values0, FieldSeparator, GlobalBase,
          CondValue, GuardGlobalIR-GuardIR),
      format(atom(ThenLabel), '~w_if_~w_then', [Prefix, OpIndex]),
      format(atom(ElseLabel), '~w_if_~w_else', [Prefix, OpIndex]),
      format(atom(DoneLabel), '~w_if_~w_done', [Prefix, OpIndex]),
      phrase(plawk_scalar_action_sequence_pairs(ThenActions, Slots, AssocPlan, FieldSeparator, OutputSeparator,
          ThenLabel, ThenLabel, RuleIndex, 0, Values0, ThenValues, _ThenOpIndex, ThenExitLabel, ThenNextExits), ThenPairs),
      phrase(plawk_scalar_action_sequence_pairs(ElseActions, Slots, AssocPlan, FieldSeparator, OutputSeparator,
          ElseLabel, ElseLabel, RuleIndex, 0, Values0, ElseValues, _ElseOpIndex, ElseExitLabel, ElseNextExits), ElsePairs),
      pairs_keys_values(ThenPairs, ThenGlobalParts, ThenLineParts),
      pairs_keys_values(ElsePairs, ElseGlobalParts, ElseLineParts),
      atomic_list_concat(ThenGlobalParts, '\n', ThenGlobalIR),
      atomic_list_concat(ElseGlobalParts, '\n', ElseGlobalIR),
      atomic_list_concat(ThenLineParts, '\n', ThenIR),
      atomic_list_concat(ElseLineParts, '\n', ElseIR),
      plawk_branch_to_done_ir(ThenExitLabel, DoneLabel, ThenDoneIR),
      plawk_branch_to_done_ir(ElseExitLabel, DoneLabel, ElseDoneIR),
      plawk_scalar_if_join_pairs(ThenExitLabel, ThenValues, ElseExitLabel, ElseValues,
          Slots, Prefix, OpIndex, PhiPairs),
      pairs_keys_values(PhiPairs, _PhiGlobalParts, PhiLineParts),
      atomic_list_concat(PhiLineParts, '\n', PhiIR),
      pairs_keys(PhiPairs, Values1),
      format(atom(IR),
'~w
  br i1 ~w, label %~w, label %~w

~w:
~w
~w

~w:
~w
~w

~w:
~w',
          [GuardIR, CondValue, ThenLabel, ElseLabel,
           ThenLabel, ThenIR, ThenDoneIR,
           ElseLabel, ElseIR, ElseDoneIR,
           DoneLabel, PhiIR]),
      atomic_list_concat([GuardGlobalIR, ThenGlobalIR, ElseGlobalIR], '\n', GlobalIR),
      append(ThenNextExits, ElseNextExits, BranchNextExits),
      NextOpIndex is OpIndex + 1
    },
    [GlobalIR-IR],
    plawk_scalar_action_sequence_pairs(Rest, Slots, AssocPlan, FieldSeparator, OutputSeparator, Prefix, DoneLabel, RuleIndex,
        NextOpIndex, Values1, Values, FinalOpIndex, ExitLabel, RestNextExits),
    { append(BranchNextExits, RestNextExits, NextExits) }.

plawk_scalar_update_operation_ir(add(Expr), scalar_double(_Name), FieldSeparator,
        Prefix, SlotIndex, OpIndex, InputValue, NextValue, GlobalIR-IR) :-
    !,
    plawk_scalar_f64_numeric_expr_ir(Expr, FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, SetupIR),
    format(atom(NextValue), '%~w_slot_~w_op_~w', [Prefix, SlotIndex, OpIndex]),
    format(atom(AddLine), '  ~w = fadd double ~w, ~w',
        [NextValue, InputValue, ValueIR]),
    plawk_join_nonempty_ir([SetupIR, AddLine], IR).
plawk_scalar_update_operation_ir(set(Expr), scalar_double(_Name), FieldSeparator,
        Prefix, SlotIndex, OpIndex, _InputValue, NextValue, GlobalIR-IR) :-
    !,
    plawk_scalar_f64_numeric_expr_ir(Expr, FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, SetupIR),
    format(atom(NextValue), '%~w_slot_~w_op_~w', [Prefix, SlotIndex, OpIndex]),
    % -0.0 + x is the IEEE identity copy (x + 0.0 would flip -0.0).
    format(atom(SetLine), '  ~w = fadd double -0.0, ~w', [NextValue, ValueIR]),
    plawk_join_nonempty_ir([SetupIR, SetLine], IR).
plawk_scalar_update_operation_ir(add(Expr), _Slot, FieldSeparator, Prefix, SlotIndex,
        OpIndex, InputValue, NextValue, GlobalIR-IR) :-
    plawk_scalar_numeric_expr_ir(Expr, FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, SetupIR),
    format(atom(NextValue), '%~w_slot_~w_op_~w', [Prefix, SlotIndex, OpIndex]),
    format(atom(AddLine), '  ~w = add i64 ~w, ~w',
        [NextValue, InputValue, ValueIR]),
    plawk_join_nonempty_ir([SetupIR, AddLine], IR).
% String scalar assignment (`x = $1 $2` / `x = "text"`): build the RHS bytes,
% intern to an atom id, and that id becomes the slot's new i64 value. Reads /
% `print` resolve the id back to text.
% `gsub(/re/, "repl", var)`: substitute over the slot's CURRENT interned value
% (InputValue -> text via @wam_atom_to_string), producing a fresh interned id.
% The discarded count goes to the shared @plawk_gsub_count scratch.
plawk_scalar_update_operation_ir(set_str(gsub_str(Global, Regex, Repl)), scalar_string(_Name),
        _FieldSeparator, Prefix, SlotIndex, OpIndex, InputValue, NextValue, GlobalIR-IR) :-
    !,
    format(atom(Base), '~w_slot_~w_op_~w_gsub', [Prefix, SlotIndex, OpIndex]),
    format(atom(PatName), '~w_pat', [Base]),
    format(atom(ReplName), '~w_repl', [Base]),
    format(atom(CacheName), '~w_cache', [Base]),
    llvm_emit_c_string_global(PatName, Regex, PatGlobal, _PL, PatBytes),
    llvm_emit_c_string_global(ReplName, Repl, ReplGlobal, ReplLen, ReplBytes),
    format(atom(CacheGlobal), '@~w = internal global i8* null', [CacheName]),
    ( Global =:= 1 -> GlobalBool = true ; GlobalBool = false ),
    format(atom(NextValue), '%~w_id', [Base]),
    format(atom(IR),
'  %~w_str = call i8* @wam_atom_to_string(i64 ~w)
  %~w_len = call i64 @strlen(i8* %~w_str)
  %~w_patptr = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0
  %~w_replptr = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0
  ~w = call i64 @wam_regex_gsub(i8* %~w_str, i64 %~w_len, i8* %~w_patptr, i8** @~w, i8* %~w_replptr, i64 ~w, i1 ~w, i64* @plawk_gsub_count)',
        [Base, InputValue,
         Base, Base,
         Base, PatBytes, PatBytes, PatName,
         Base, ReplBytes, ReplBytes, ReplName,
         NextValue, Base, Base, Base, CacheName, Base, ReplLen, GlobalBool]),
    atomic_list_concat([PatGlobal, ReplGlobal, CacheGlobal], '\n', GlobalIR).
plawk_scalar_update_operation_ir(set_str(Src), scalar_string(_Name), FieldSeparator,
        Prefix, SlotIndex, OpIndex, _InputValue, NextValue, GlobalIR-IR) :-
    !,
    format(atom(Base), '~w_slot_~w_op_~w_str', [Prefix, SlotIndex, OpIndex]),
    plawk_str_build_ir(Src, FieldSeparator, Base, NextValue, GlobalParts, SetupLines),
    plawk_join_nonempty_ir(GlobalParts, GlobalIR),
    atomic_list_concat(SetupLines, '\n', IR).
% An ARGV/getline strnum slot stores the same interned atom id as a string slot
% (the runtime call yields an id via plawk_str_build_ir); only comparison-time
% dispatch differs (strnum vs string). So the store is identical to the string
% case -- the slot's kind, not this op, drives the numeric-vs-lexical choice.
plawk_scalar_update_operation_ir(set_str(Src), scalar_strnum(_Name), FieldSeparator,
        Prefix, SlotIndex, OpIndex, _InputValue, NextValue, GlobalIR-IR) :-
    !,
    format(atom(Base), '~w_slot_~w_op_~w_str', [Prefix, SlotIndex, OpIndex]),
    plawk_str_build_ir(Src, FieldSeparator, Base, NextValue, GlobalParts, SetupLines),
    plawk_join_nonempty_ir(GlobalParts, GlobalIR),
    atomic_list_concat(SetupLines, '\n', IR).
% strnum assignment `x = $N`: intern the field's raw text into an atom id (the
% strnum slot repr), retaining the bytes so a later comparison can decide
% numeric vs lexical. Matched specifically before the generic i64 set. The
% record is %line in this scope (as in the concat path).
plawk_scalar_update_operation_ir(set(field_i64(FieldIndex)), scalar_strnum(_Name),
        FieldSeparator, Prefix, SlotIndex, OpIndex, _InputValue, NextValue, ''-IR) :-
    !,
    format(atom(Base), '~w_slot_~w_op_~w_snum', [Prefix, SlotIndex, OpIndex]),
    format(atom(NextValue), '%~w_id', [Base]),
    format(atom(IR),
'  %~w_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)
  %~w_ptr = extractvalue %WamSlice %~w_slice, 0
  %~w_len = extractvalue %WamSlice %~w_slice, 1
  %~w_id = call i64 @wam_intern_atom(i8* %~w_ptr, i64 %~w_len)',
        [Base, FieldIndex, FieldSeparator,
         Base, Base,
         Base, Base,
         Base, Base, Base]).
% strnum copy `z = x` (step 4): both are strnum, so copy the source's atom id
% straight into the target slot -- strnum-ness propagates, no coercion. The
% read has already been substituted to ssa_strnum(Id).
plawk_scalar_update_operation_ir(set(ssa_strnum(Id)), scalar_strnum(_Name),
        _FieldSeparator, _Prefix, _SlotIndex, _OpIndex, _InputValue, Id, ''-'') :-
    !.
plawk_scalar_update_operation_ir(set(Expr), _Slot, FieldSeparator, Prefix, SlotIndex,
        OpIndex, _InputValue, NextValue, GlobalIR-IR) :-
    plawk_scalar_numeric_expr_ir(Expr, FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, SetupIR),
    format(atom(NextValue), '%~w_slot_~w_op_~w', [Prefix, SlotIndex, OpIndex]),
    format(atom(SetLine), '  ~w = add i64 0, ~w', [NextValue, ValueIR]),
    plawk_join_nonempty_ir([SetupIR, SetLine], IR).

% Build a string RHS into an interned atom id (returned as the SSA value).
% A bare literal interns its own global; a concat snprintf's its `%.*s` field
% slots and literal segments into a 4096-byte stack buffer, then interns the
% result (the assignment mirror of the print concat).
% `sprintf("fmt", args...)`: reuse the printf format engine (arg lowering +
% format rewrite) to snprintf into the buffer, then intern -- the string mirror
% of the printf action.
plawk_str_build_ir(sprintf(string(Format), Args), FieldSeparator, Base, IdValueIR,
        GlobalParts, SetupLines) :-
    phrase(plawk_printf_arg_pairs(Args, FieldSeparator, Base, 0), ArgPairs),
    pairs_keys_values(ArgPairs, ArgGlobalParts, ArgInfoPairs),
    pairs_keys_values(ArgInfoPairs, ArgSetupParts, ArgCallArgLists),
    append(ArgCallArgLists, CallArgs),
    maplist(plawk_printf_call_arg_kind, CallArgs, ArgKinds),
    plawk_printf_rewrite_format(Format, ArgKinds, PrintfFormat),
    format(atom(FmtName), '~w_fmt', [Base]),
    llvm_emit_c_string_global(FmtName, PrintfFormat, FmtGlobal, _FmtLen, FmtBytes),
    plawk_printf_call_args_ir(CallArgs, CallArgsIR),
    format(atom(FmtP),
        '  %~w_fmtp = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0',
        [Base, FmtBytes, FmtBytes, FmtName]),
    format(atom(BufA), '  %~w_buf = alloca [4096 x i8]', [Base]),
    format(atom(BufP),
        '  %~w_bufp = getelementptr [4096 x i8], [4096 x i8]* %~w_buf, i64 0, i64 0',
        [Base, Base]),
    ( CallArgsIR == ''
    -> format(atom(Snp),
           '  %~w_wrote = call i32 (i8*, i64, i8*, ...) @snprintf(i8* %~w_bufp, i64 4096, i8* %~w_fmtp)',
           [Base, Base, Base])
    ;  format(atom(Snp),
           '  %~w_wrote = call i32 (i8*, i64, i8*, ...) @snprintf(i8* %~w_bufp, i64 4096, i8* %~w_fmtp, ~w)',
           [Base, Base, Base, CallArgsIR])
    ),
    format(atom(LenL), '  %~w_len = call i64 @strlen(i8* %~w_bufp)', [Base, Base]),
    format(atom(IdL),
        '  %~w_id = call i64 @wam_intern_atom(i8* %~w_bufp, i64 %~w_len)',
        [Base, Base, Base]),
    format(atom(IdValueIR), '%~w_id', [Base]),
    GlobalParts = [FmtGlobal | ArgGlobalParts],
    append(ArgSetupParts, [FmtP, BufA, BufP, Snp, LenL, IdL], SetupLines).
plawk_str_build_ir(string(Value), _FieldSeparator, Base, IdValueIR,
        [GlobalIR], [PtrLine, IdLine]) :-
    format(atom(StrName), '~w_lit', [Base]),
    llvm_emit_c_string_global(StrName, Value, GlobalIR, Len, BytesLen),
    format(atom(PtrLine),
        '  %~w_ptr = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0',
        [Base, BytesLen, BytesLen, StrName]),
    format(atom(IdLine),
        '  %~w_id = call i64 @wam_intern_atom(i8* %~w_ptr, i64 ~w)',
        [Base, Base, Len]),
    format(atom(IdValueIR), '%~w_id', [Base]).
% `ENVIRON["NAME"]`: look up the environment variable (a NUL-terminated key
% constant) and yield the interned atom id of its value (0 = unset -> empty).
plawk_str_build_ir(environ(Name), _FieldSeparator, Base, IdValueIR,
        [GlobalIR], [PtrLine, CallLine]) :-
    format(atom(KeyName), '~w_envkey', [Base]),
    llvm_emit_c_string_global(KeyName, Name, GlobalIR, _Len, BytesLen),
    format(atom(PtrLine),
        '  %~w_envkeyp = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0',
        [Base, BytesLen, BytesLen, KeyName]),
    format(atom(CallLine),
        '  %~w_envid = call i64 @wam_environ_get(i8* %~w_envkeyp)',
        [Base, Base]),
    format(atom(IdValueIR), '%~w_envid', [Base]).
% `ARGV[N]`: the N-th command-line argument as a string scalar. wam_argv_get
% interns the argument bytes and returns the atom id (0 = out of range -> empty).
plawk_str_build_ir(argv_at(N), _FieldSeparator, Base, IdValueIR,
        [], [CallLine]) :-
    integer(N), N >= 0,
    format(atom(CallLine),
        '  %~w_argvid = call i64 @wam_argv_get(i64 ~w)', [Base, N]),
    format(atom(IdValueIR), '%~w_argvid', [Base]).
plawk_str_build_ir(concat(Parts), FieldSeparator, Base, IdValueIR,
        [FmtGlobal, EmptyGlobal], SetupLines) :-
    plawk_str_concat_format(Parts, FmtAtom),
    format(atom(FmtName), '~w_fmt', [Base]),
    llvm_emit_c_string_global(FmtName, FmtAtom, FmtGlobal, _FmtLen, FmtBytes),
    format(atom(EmptyName), '~w_empty', [Base]),
    format(atom(EmptyGlobal),
        '@.~w = private constant [1 x i8] zeroinitializer', [EmptyName]),
    plawk_str_concat_field_lines(Parts, Base, EmptyName, FieldSeparator, 0,
        FieldLines, ArgFrags),
    ( ArgFrags == []
    -> ArgsSuffix = ''
    ;  atomic_list_concat(ArgFrags, ', ', ArgsJoined),
       atom_concat(', ', ArgsJoined, ArgsSuffix)
    ),
    format(atom(BufA), '  %~w_buf = alloca [4096 x i8]', [Base]),
    format(atom(BufP),
        '  %~w_bufp = getelementptr [4096 x i8], [4096 x i8]* %~w_buf, i64 0, i64 0',
        [Base, Base]),
    format(atom(FmtP),
        '  %~w_fmtp = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0',
        [Base, FmtBytes, FmtBytes, FmtName]),
    format(atom(Snp),
        '  %~w_wrote = call i32 (i8*, i64, i8*, ...) @snprintf(i8* %~w_bufp, i64 4096, i8* %~w_fmtp~w)',
        [Base, Base, Base, ArgsSuffix]),
    format(atom(LenL), '  %~w_len = call i64 @strlen(i8* %~w_bufp)', [Base, Base]),
    format(atom(IdL),
        '  %~w_id = call i64 @wam_intern_atom(i8* %~w_bufp, i64 %~w_len)',
        [Base, Base, Base]),
    format(atom(IdValueIR), '%~w_id', [Base]),
    append(FieldLines, [BufA, BufP, FmtP, Snp, LenL, IdL], SetupLines).

% The snprintf format for a string-concat: `%.*s` per field slot, literal
% segments inline (with `%` escaped). No separator -- operands are adjacent.
plawk_str_concat_format([], '').
plawk_str_concat_format([field(_) | Rest], Fmt) :-
    plawk_str_concat_format(Rest, RestFmt),
    atom_concat('%.*s', RestFmt, Fmt).
% a resolved string-scalar read prints as a null-terminated `%s`.
plawk_str_concat_format([ssa_str(_) | Rest], Fmt) :-
    plawk_str_concat_format(Rest, RestFmt),
    atom_concat('%s', RestFmt, Fmt).
plawk_str_concat_format([string(Value) | Rest], Fmt) :-
    plawk_str_escape_percent(Value, Escaped),
    plawk_str_concat_format(Rest, RestFmt),
    atom_concat(Escaped, RestFmt, Fmt).

plawk_str_escape_percent(Value, Escaped) :-
    ( sub_atom(Value, _, _, _, '%')
    -> atomic_list_concat(Parts, '%', Value),
       atomic_list_concat(Parts, '%%', Escaped)
    ;  Escaped = Value
    ).

% Per concat field, project field N (null-safe empty), yielding an
% `i32 <len>, i8* <ptr>` snprintf argument pair for its `%.*s` slot. Literal
% parts contribute no argument (they live in the format).
plawk_str_concat_field_lines([], _B, _Empty, _FieldSep, _Pos, [], []).
plawk_str_concat_field_lines([string(_) | Rest], B, EmptyName, FieldSep, Pos,
        Lines, Frags) :-
    plawk_str_concat_field_lines(Rest, B, EmptyName, FieldSep, Pos, Lines, Frags).
% a resolved string-scalar read: resolve its atom id to text (id 0 -> empty via
% the null-safe global), yielding an `i8* <ptr>` snprintf argument for its `%s`.
plawk_str_concat_field_lines([ssa_str(Value) | Rest], B, EmptyName, FieldSep, Pos,
        Lines, [Frag | Frags]) :-
    format(atom(Str), '  %~w_c~w_s = call i8* @wam_atom_to_string(i64 ~w)',
        [B, Pos, Value]),
    format(atom(Empty), '  %~w_c~w_e = icmp eq i64 ~w, 0', [B, Pos, Value]),
    format(atom(Sel),
        '  %~w_c~w_sptr = select i1 %~w_c~w_e, i8* getelementptr ([1 x i8], [1 x i8]* @.~w, i64 0, i64 0), i8* %~w_c~w_s',
        [B, Pos, B, Pos, EmptyName, B, Pos]),
    format(atom(Frag), 'i8* %~w_c~w_sptr', [B, Pos]),
    ThisLines = [Str, Empty, Sel],
    Pos1 is Pos + 1,
    plawk_str_concat_field_lines(Rest, B, EmptyName, FieldSep, Pos1, RestLines, Frags),
    append(ThisLines, RestLines, Lines).
plawk_str_concat_field_lines([field(N) | Rest], B, EmptyName, FieldSep, Pos,
        Lines, [Frag | Frags]) :-
    format(atom(Slice),
        '  %~w_c~w_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)',
        [B, Pos, N, FieldSep]),
    format(atom(Ptr), '  %~w_c~w_ptr = extractvalue %WamSlice %~w_c~w_slice, 0',
        [B, Pos, B, Pos]),
    format(atom(Len), '  %~w_c~w_len = extractvalue %WamSlice %~w_c~w_slice, 1',
        [B, Pos, B, Pos]),
    format(atom(Null), '  %~w_c~w_null = icmp eq i8* %~w_c~w_ptr, null',
        [B, Pos, B, Pos]),
    format(atom(SafePtr),
        '  %~w_c~w_sptr = select i1 %~w_c~w_null, i8* getelementptr ([1 x i8], [1 x i8]* @.~w, i64 0, i64 0), i8* %~w_c~w_ptr',
        [B, Pos, B, Pos, EmptyName, B, Pos]),
    format(atom(SafeLen), '  %~w_c~w_slen = select i1 %~w_c~w_null, i64 0, i64 %~w_c~w_len',
        [B, Pos, B, Pos, B, Pos]),
    format(atom(Lenw), '  %~w_c~w_lenw = trunc i64 %~w_c~w_slen to i32',
        [B, Pos, B, Pos]),
    format(atom(Frag), 'i32 %~w_c~w_lenw, i8* %~w_c~w_sptr', [B, Pos, B, Pos]),
    ThisLines = [Slice, Ptr, Len, Null, SafePtr, SafeLen, Lenw],
    Pos1 is Pos + 1,
    plawk_str_concat_field_lines(Rest, B, EmptyName, FieldSep, Pos1, RestLines, Frags),
    append(ThisLines, RestLines, Lines).

%% plawk_scalar_f64_numeric_expr_ir(+Expr, +FieldSeparator, +Prefix,
%%     +SlotIndex, +OpIndex, -ValueIR, -GlobalIR, -SetupIR)
%
%  Update RHS for a double slot. Double-typed trees (and double scalar
%  reads) emit through the f64 emitter; i64-shaped operands emit through
%  the i64 path and promote with sitofp.
plawk_scalar_f64_numeric_expr_ir(ssa_f64(Value), _FieldSeparator, _Prefix,
        _SlotIndex, _OpIndex, Value, '', '') :-
    !.
plawk_scalar_f64_numeric_expr_ir(Expr, FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, SetupIR) :-
    plawk_expr_is_double(Expr),
    !,
    format(atom(Base), '~w_slot_~w_op_~w_f64', [Prefix, SlotIndex, OpIndex]),
    plawk_f64_expr_ir(Expr, FieldSeparator, Base, Base, ValueIR,
        GlobalParts, SetupParts),
    atomic_list_concat(GlobalParts, '\n', GlobalIR),
    atomic_list_concat(SetupParts, '\n', SetupIR).
plawk_scalar_f64_numeric_expr_ir(Expr, FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, SetupIR) :-
    plawk_scalar_numeric_expr_ir(Expr, FieldSeparator, Prefix, SlotIndex,
        OpIndex, IntValueIR, GlobalIR, IntSetupIR),
    format(atom(ValueIR), '%~w_slot_~w_op_~w_f64p', [Prefix, SlotIndex, OpIndex]),
    format(atom(PromoteIR), '  ~w = sitofp i64 ~w to double',
        [ValueIR, IntValueIR]),
    plawk_join_nonempty_ir([IntSetupIR, PromoteIR], SetupIR).

plawk_scalar_numeric_expr_ir(ssa(Value), _FieldSeparator, _Prefix, _SlotIndex,
        _OpIndex, Value, '', '') :-
    atom(Value).
% a bare strnum read as a numeric update RHS (`y = x`): coerce the atom id to
% i64 like the field read (step 3c).
plawk_scalar_numeric_expr_ir(ssa_strnum(Value), _FieldSeparator, Prefix,
        SlotIndex, OpIndex, ValueIR, '', SetupIR) :-
    format(atom(Base), '~w_slot_~w_op_~w_snc', [Prefix, SlotIndex, OpIndex]),
    plawk_strnum_i64_coerce_lines(Base, Value, ValueIR, Lines),
    atomic_list_concat(Lines, '\n', SetupIR).

% Coerce a strnum atom id (Value) to i64 using the field parser + not-a-number
% -> 0 default, matching @wam_atom_field_i64_value exactly.
plawk_strnum_i64_coerce_lines(Base, Value, ValueIR,
        [StrL, LenL, ParseL, ValL, OkL, SelL]) :-
    format(atom(ValueIR), '%~w_snc_val', [Base]),
    format(atom(StrL), '  %~w_snc_s = call i8* @wam_atom_to_string(i64 ~w)',
        [Base, Value]),
    format(atom(LenL), '  %~w_snc_len = call i64 @strlen(i8* %~w_snc_s)',
        [Base, Base]),
    format(atom(ParseL),
        '  %~w_snc_p = call %WamI64Parse @wam_slice_i64_parse_value(i8* %~w_snc_s, i64 %~w_snc_len)',
        [Base, Base, Base]),
    format(atom(ValL), '  %~w_snc_v = extractvalue %WamI64Parse %~w_snc_p, 0',
        [Base, Base]),
    format(atom(OkL), '  %~w_snc_ok = extractvalue %WamI64Parse %~w_snc_p, 1',
        [Base, Base]),
    format(atom(SelL),
        '  ~w = select i1 %~w_snc_ok, i64 %~w_snc_v, i64 0', [ValueIR, Base, Base]).
plawk_scalar_numeric_expr_ir(prolog_call(Name, Args), FieldSeparator, Prefix,
        SlotIndex, OpIndex, ValueIR, GlobalIR, IR) :-
    format(atom(CallBase), '~w_slot_~w_op_~w_prolog_call',
        [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(prolog_call(Name, Args), FieldSeparator, CallBase,
        CallBase, ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(dyncall(Args), FieldSeparator, Prefix,
        SlotIndex, OpIndex, ValueIR, GlobalIR, IR) :-
    format(atom(DynBase), '~w_slot_~w_op_~w_dyncall',
        [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(dyncall(Args), FieldSeparator, DynBase,
        DynBase, ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(dyncall_named(Name, Args), FieldSeparator, Prefix,
        SlotIndex, OpIndex, ValueIR, GlobalIR, IR) :-
    format(atom(DynBase), '~w_slot_~w_op_~w_dyncall_named',
        [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(dyncall_named(Name, Args), FieldSeparator, DynBase,
        DynBase, ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(dyncall_at(Source, Args), FieldSeparator, Prefix,
        SlotIndex, OpIndex, ValueIR, GlobalIR, IR) :-
    format(atom(DynAtBase), '~w_slot_~w_op_~w_dyncall_at',
        [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(dyncall_at(Source, Args), FieldSeparator, DynAtBase,
        DynAtBase, ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(dyncall_at_named(Name, Source, Args),
        FieldSeparator, Prefix, SlotIndex, OpIndex, ValueIR, GlobalIR, IR) :-
    format(atom(DynAtBase), '~w_slot_~w_op_~w_dyncall_at_named',
        [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(dyncall_at_named(Name, Source, Args),
        FieldSeparator, DynAtBase, DynAtBase, ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(compile_handle(Arg), FieldSeparator, Prefix,
        SlotIndex, OpIndex, ValueIR, GlobalIR, IR) :-
    format(atom(CHBase), '~w_slot_~w_op_~w_compile_h',
        [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(compile_handle(Arg), FieldSeparator, CHBase,
        CHBase, ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(compile_file_handle(Arg), FieldSeparator, Prefix,
        SlotIndex, OpIndex, ValueIR, GlobalIR, IR) :-
    format(atom(CFBase), '~w_slot_~w_op_~w_compile_fh',
        [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(compile_file_handle(Arg), FieldSeparator, CFBase,
        CFBase, ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(const(Value), _FieldSeparator, _Prefix, _SlotIndex,
        _OpIndex, ValueIR, GlobalIR, IR) :-
    plawk_i64_expr_ir_parts(const(Value), 0, scalar_const, scalar_const_global,
        ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(length(FieldIndex), FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, IR) :-
    format(atom(LengthBase), '~w_slot_~w_op_~w_len', [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(length(FieldIndex), FieldSeparator, LengthBase, LengthBase,
        ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(match_expr(Src, Regex), FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, IR) :-
    format(atom(MatchBase), '~w_slot_~w_op_~w_match', [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(match_expr(Src, Regex), FieldSeparator, MatchBase, MatchBase,
        ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(special(Name), FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, IR) :-
    ( Name == 'RSTART' ; Name == 'RLENGTH' ),
    format(atom(SpecBase), '~w_slot_~w_op_~w_spec', [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(special(Name), FieldSeparator, SpecBase, SpecBase,
        ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(field_i64(FieldIndex), FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, IR) :-
    format(atom(ParseBase), '~w_slot_~w_op_~w_field_i64',
        [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(field_i64(FieldIndex), FieldSeparator, ParseBase, ParseBase,
        ValueIR, GlobalIR, IR).
% Ternary in a scalar assignment (`x = COND ? A : B`): an i64 value via select.
plawk_scalar_numeric_expr_ir(ternary(Cond, Then, Else), FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, IR) :-
    format(atom(TernBase), '~w_slot_~w_op_~w_tern', [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(ternary(Cond, Then, Else), FieldSeparator, TernBase, TernBase,
        ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(Expr, FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, IR) :-
    plawk_i64_scalar_primary_expr(Expr),
    format(atom(PrimaryBase), '~w_slot_~w_op_~w_i64_primary',
        [Prefix, SlotIndex, OpIndex]),
    plawk_i64_expr_ir_parts(Expr, FieldSeparator, PrimaryBase, PrimaryBase,
        ValueIR, GlobalIR, IR).
plawk_scalar_numeric_expr_ir(Expr, FieldSeparator, Prefix, SlotIndex,
        OpIndex, ValueIR, GlobalIR, IR) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, NamePart, _Left, _Right),
    format(atom(BinaryBase), '~w_slot_~w_op_~w_i64_~w',
        [Prefix, SlotIndex, OpIndex, NamePart]),
    plawk_i64_expr_ir_parts(Expr, FieldSeparator, BinaryBase,
        BinaryBase, ValueIR, GlobalIR, IR).

plawk_i64_expr_ir_parts(Expr, FieldSeparator, Base, GlobalBase, ValueIR, IR) :-
    plawk_i64_expr_ir_parts(Expr, FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalIR, SetupIR),
    plawk_join_nonempty_ir([GlobalIR, SetupIR], IR).

plawk_i64_expr_ir_parts(Expr, FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalIR, SetupIR) :-
    plawk_i64_expr_ir(Expr, FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts),
    plawk_join_nonempty_ir(GlobalParts, GlobalIR),
    atomic_list_concat(SetupParts, '\n', SetupIR).

plawk_i64_expr_ir(const(Value), _FieldSeparator, _Base, _GlobalBase, ValueIR, [], []) :-
    integer(Value),
    format(atom(ValueIR), '~w', [Value]).
plawk_i64_expr_ir(int(Value), _FieldSeparator, _Base, _GlobalBase, ValueIR, [], []) :-
    integer(Value),
    format(atom(ValueIR), '~w', [Value]).
plawk_i64_expr_ir(ssa(Value), _FieldSeparator, _Base, _GlobalBase, Value, [], []) :-
    atom(Value).
% a strnum read in i64 arithmetic (step 3c): coerce the atom id to i64 using the
% SAME integer parser the field read uses, so the arithmetic result is identical
% to the pre-strnum i64 path (byte-for-byte, incl. the not-a-number -> 0 default).
plawk_i64_expr_ir(ssa_strnum(Value), _FieldSeparator, Base, _GlobalBase,
        ValueIR, [], Lines) :-
    plawk_strnum_i64_coerce_lines(Base, Value, ValueIR, Lines).
plawk_i64_expr_ir(prolog_call(Name, Args), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    length(Args, NArgs),
    plawk_foreign_args_ir(Args, FieldSeparator, GlobalBase, ArgValueIRs,
        GlobalParts, ArgSetupParts),
    plawk_foreign_call_args_ir(ArgValueIRs, CallArgsIR),
    format(atom(ResIR),
        '  %~w_res = call { i64, i1 } @plawk_foreign_call_~w_~w(~w)',
        [Base, Name, NArgs, CallArgsIR]),
    format(atom(ValIR),
        '  %~w_val = extractvalue { i64, i1 } %~w_res, 0', [Base, Base]),
    format(atom(OkIR),
        '  %~w_ok = extractvalue { i64, i1 } %~w_res, 1', [Base, Base]),
    format(atom(SelIR),
        '  %~w = select i1 %~w_ok, i64 %~w_val, i64 0', [Base, Base, Base]),
    format(atom(ValueIR), '%~w', [Base]),
    append(ArgSetupParts, [ResIR, ValIR, OkIR, SelIR], SetupParts).
plawk_i64_expr_ir(dyncall(Args), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    length(Args, NArgs),
    plawk_foreign_args_ir(Args, FieldSeparator, GlobalBase, ArgValueIRs,
        GlobalParts, ArgSetupParts),
    plawk_foreign_call_args_ir(ArgValueIRs, CallArgsIR),
    format(atom(ResIR),
        '  %~w_res = call { i64, i1 } @plawk_dyncall_~w(~w)',
        [Base, NArgs, CallArgsIR]),
    format(atom(ValIR),
        '  %~w_val = extractvalue { i64, i1 } %~w_res, 0', [Base, Base]),
    format(atom(OkIR),
        '  %~w_ok = extractvalue { i64, i1 } %~w_res, 1', [Base, Base]),
    format(atom(SelIR),
        '  %~w = select i1 %~w_ok, i64 %~w_val, i64 0', [Base, Base, Base]),
    format(atom(ValueIR), '%~w', [Base]),
    append(ArgSetupParts, [ResIR, ValIR, OkIR, SelIR], SetupParts).
plawk_i64_expr_ir(dyncall_named(Name, Args), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    length(Args, NArgs),
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    plawk_foreign_args_ir(Args, FieldSeparator, GlobalBase, ArgValueIRs,
        GlobalParts, ArgSetupParts),
    plawk_foreign_call_args_ir(ArgValueIRs, CallArgsIR),
    format(atom(ResIR),
        '  %~w_res = call { i64, i1 } @plawk_dyncall_named_~w(~w)',
        [Base, Sym, CallArgsIR]),
    format(atom(ValIR),
        '  %~w_val = extractvalue { i64, i1 } %~w_res, 0', [Base, Base]),
    format(atom(OkIR),
        '  %~w_ok = extractvalue { i64, i1 } %~w_res, 1', [Base, Base]),
    format(atom(SelIR),
        '  %~w = select i1 %~w_ok, i64 %~w_val, i64 0', [Base, Base, Base]),
    format(atom(ValueIR), '%~w', [Base]),
    append(ArgSetupParts, [ResIR, ValIR, OkIR, SelIR], SetupParts).
plawk_i64_expr_ir(dyncall_at(Source, Args), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    length(Args, NArgs),
    plawk_dyncall_source_ir(Source, FieldSeparator, Base, GlobalBase,
        PathPtrIR, PathLenIR, SrcGlobals, SrcSetup),
    plawk_foreign_args_ir(Args, FieldSeparator, GlobalBase, ArgValueIRs,
        ArgGlobals, ArgSetup),
    ( ArgValueIRs == []
    -> CallArgsSuffix = ''
    ;  plawk_foreign_call_args_ir(ArgValueIRs, AV),
       format(atom(CallArgsSuffix), ', ~w', [AV])
    ),
    format(atom(ResIR),
        '  %~w_res = call { i64, i1 } @plawk_dyncall_at_~w(i8* ~w, i64 ~w~w)',
        [Base, NArgs, PathPtrIR, PathLenIR, CallArgsSuffix]),
    format(atom(ValIR),
        '  %~w_val = extractvalue { i64, i1 } %~w_res, 0', [Base, Base]),
    format(atom(OkIR),
        '  %~w_ok = extractvalue { i64, i1 } %~w_res, 1', [Base, Base]),
    format(atom(SelIR),
        '  %~w = select i1 %~w_ok, i64 %~w_val, i64 0', [Base, Base, Base]),
    format(atom(ValueIR), '%~w', [Base]),
    append(SrcGlobals, ArgGlobals, GlobalParts),
    append(SrcSetup, ArgSetup, PreSetup),
    append(PreSetup, [ResIR, ValIR, OkIR, SelIR], SetupParts).
% compile(src) / compile_file(path) as i64 expressions: the value IS
% the handle -- reuse the source marshal, whose (null, handle) pair's
% second half is exactly the handle SSA value.
plawk_i64_expr_ir(compile_handle(Arg), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    plawk_dyncall_source_ir(compile_src(Arg), FieldSeparator, Base,
        GlobalBase, _NullPtr, ValueIR, GlobalParts, SetupParts).
plawk_i64_expr_ir(compile_file_handle(Arg), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    plawk_dyncall_source_ir(compile_file_src(Arg), FieldSeparator, Base,
        GlobalBase, _NullPtr, ValueIR, GlobalParts, SetupParts).
plawk_i64_expr_ir(dyncall_at_named(Name, Source, Args), FieldSeparator, Base,
        GlobalBase, ValueIR, GlobalParts, SetupParts) :-
    length(Args, NArgs),
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    plawk_dyncall_source_ir(Source, FieldSeparator, Base, GlobalBase,
        PathPtrIR, PathLenIR, SrcGlobals, SrcSetup),
    plawk_foreign_args_ir(Args, FieldSeparator, GlobalBase, ArgValueIRs,
        ArgGlobals, ArgSetup),
    ( ArgValueIRs == []
    -> CallArgsSuffix = ''
    ;  plawk_foreign_call_args_ir(ArgValueIRs, AV),
       format(atom(CallArgsSuffix), ', ~w', [AV])
    ),
    format(atom(ResIR),
        '  %~w_res = call { i64, i1 } @plawk_dyncall_at_named_~w(i8* ~w, i64 ~w~w)',
        [Base, Sym, PathPtrIR, PathLenIR, CallArgsSuffix]),
    format(atom(ValIR),
        '  %~w_val = extractvalue { i64, i1 } %~w_res, 0', [Base, Base]),
    format(atom(OkIR),
        '  %~w_ok = extractvalue { i64, i1 } %~w_res, 1', [Base, Base]),
    format(atom(SelIR),
        '  %~w = select i1 %~w_ok, i64 %~w_val, i64 0', [Base, Base, Base]),
    format(atom(ValueIR), '%~w', [Base]),
    append(SrcGlobals, ArgGlobals, GlobalParts),
    append(SrcSetup, ArgSetup, PreSetup),
    append(PreSetup, [ResIR, ValIR, OkIR, SelIR], SetupParts).
plawk_i64_expr_ir(field(FieldIndex), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    integer(FieldIndex),
    plawk_i64_expr_ir(field_i64(FieldIndex), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts).
plawk_i64_expr_ir(nr, _FieldSeparator, _Base, _GlobalBase, '%current_nr', [], []).
plawk_i64_expr_ir(nf, binfmt(Types), _Base, _GlobalBase, ValueIR, [], []) :-
    !,
    length(Types, NFields),
    format(atom(ValueIR), '~w', [NFields]).
plawk_i64_expr_ir(nf, FieldSeparator, Base, _GlobalBase, ValueIR, [], [CountIR]) :-
    llvm_emit_atom_field_count('%line', FieldSeparator, Base, CountIR),
    format(atom(ValueIR), '%~w', [Base]).
plawk_i64_expr_ir(special('NR'), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    plawk_i64_expr_ir(nr, FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts).
plawk_i64_expr_ir(special('NF'), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    plawk_i64_expr_ir(nf, FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts).
plawk_i64_expr_ir(length(field(FieldIndex)), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    plawk_i64_expr_ir(length(FieldIndex), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts).
plawk_i64_expr_ir(length(FieldIndex), FieldSeparator, Base, _GlobalBase, ValueIR, [], [LengthIR]) :-
    llvm_emit_atom_field_length('%line', FieldIndex, FieldSeparator, Base, LengthIR),
    format(atom(ValueIR), '%~w', [Base]).
plawk_i64_expr_ir(int(field(FieldIndex)), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    plawk_i64_expr_ir(field_i64(FieldIndex), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts).
plawk_i64_expr_ir(field_i64(FieldIndex), binfmt(Types), Base, _GlobalBase,
        ValueIR, [], LoadLines) :-
    !,
    plawk_binfmt_field_load_lines(binfmt(Types), FieldIndex, Base, ValueIR,
        LoadLines).
plawk_i64_expr_ir(field_i64(FieldIndex), FieldSeparator, Base, _GlobalBase, ValueIR, [], [ParseIR]) :-
    format(atom(ValueIR), '%~w_value_or_default', [Base]),
    llvm_emit_atom_field_i64_or_default('%line', FieldIndex, FieldSeparator, 0,
        Base, ValueIR, ParseIR).
plawk_i64_expr_ir(index(field(FieldIndex), string(Needle)), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    plawk_i64_expr_ir(index(FieldIndex, Needle), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts).
plawk_i64_expr_ir(Expr, FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    plawk_i64_binary_expr(Expr, LLVMOp, _NamePart, Left, Right),
    format(atom(LeftBase), '~w_lhs', [Base]),
    format(atom(LeftGlobalBase), '~w_lhs', [GlobalBase]),
    plawk_i64_expr_ir(Left, FieldSeparator, LeftBase, LeftGlobalBase,
        LeftValueIR, LeftGlobalParts, LeftSetupParts),
    format(atom(RightBase), '~w_rhs', [Base]),
    format(atom(RightGlobalBase), '~w_rhs', [GlobalBase]),
    plawk_i64_expr_ir(Right, FieldSeparator, RightBase, RightGlobalBase,
        RightValueIR, RightGlobalParts, RightSetupParts),
    format(atom(ValueIR), '%~w', [Base]),
    plawk_i64_binary_op_lines(LLVMOp, Base, LeftValueIR, RightValueIR, OpLines),
    append(LeftGlobalParts, RightGlobalParts, GlobalParts),
    append([LeftSetupParts, RightSetupParts, OpLines], SetupParts).
plawk_i64_expr_ir(index(FieldIndex, Needle), FieldSeparator, Base, GlobalBase,
        ValueIR, [GlobalIR], [CallIR]) :-
    llvm_emit_atom_field_index(GlobalBase, '%line', FieldIndex, Needle, FieldSeparator,
        Base, GlobalIR-CallIR),
    format(atom(ValueIR), '%~w', [Base]).
% match(SRC, /re/): run the ERE over SRC's bytes, storing RSTART/RLENGTH and
% yielding the 1-based match position (0 if none). SRC is $0 (whole record) or a
% positive field slice. The pattern is a per-site constant + lazily-compiled cache.
plawk_i64_expr_ir(match_expr(field(FieldIndex), Regex), FieldSeparator, Base, GlobalBase,
        ValueIR, [PatGlobal, CacheGlobal], SetupLines) :-
    sanitize_functor_for_llvm(GlobalBase, Safe),
    format(atom(PatName), '~w_matchpat', [Safe]),
    format(atom(CacheName), '~w_matchcache', [Safe]),
    llvm_emit_c_string_global(PatName, Regex, PatGlobal, _PL, PatBytes),
    format(atom(CacheGlobal), '@~w = internal global i8* null', [CacheName]),
    format(atom(PatPtr),
        '  %~w_mpat = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0',
        [Base, PatBytes, PatBytes, PatName]),
    ( FieldIndex =:= 0
    ->  format(atom(SrcLines),
'  %~w_lp = call i64 @value_payload(%Value %line)
  %~w_sptr = call i8* @wam_atom_to_string(i64 %~w_lp)
  %~w_slen = call i64 @strlen(i8* %~w_sptr)',
            [Base, Base, Base, Base, Base]),
        format(atom(SrcPtr), '%~w_sptr', [Base]),
        format(atom(SrcLen), '%~w_slen', [Base])
    ;   format(atom(SrcLines),
'  %~w_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)
  %~w_sptr = extractvalue %WamSlice %~w_slice, 0
  %~w_slen = extractvalue %WamSlice %~w_slice, 1',
            [Base, FieldIndex, FieldSeparator, Base, Base, Base, Base]),
        format(atom(SrcPtr), '%~w_sptr', [Base]),
        format(atom(SrcLen), '%~w_slen', [Base])
    ),
    format(atom(CallLine),
        '  %~w = call i64 @wam_regex_match(i8* ~w, i64 ~w, i8* %~w_mpat, i8** @~w, i64* @plawk_rstart, i64* @plawk_rlength)',
        [Base, SrcPtr, SrcLen, Base, CacheName]),
    SetupLines = [SrcLines, PatPtr, CallLine],
    format(atom(ValueIR), '%~w', [Base]).
plawk_i64_expr_ir(special('RSTART'), _FieldSeparator, Base, _GlobalBase,
        ValueIR, [], [Line]) :-
    format(atom(Line), '  %~w = load i64, i64* @plawk_rstart', [Base]),
    format(atom(ValueIR), '%~w', [Base]).
plawk_i64_expr_ir(special('RLENGTH'), _FieldSeparator, Base, _GlobalBase,
        ValueIR, [], [Line]) :-
    format(atom(Line), '  %~w = load i64, i64* @plawk_rlength', [Base]),
    format(atom(ValueIR), '%~w', [Base]).
plawk_i64_expr_ir(special('ARGC'), _FieldSeparator, Base, _GlobalBase,
        ValueIR, [], [Line]) :-
    format(atom(Line), '  %~w = call i64 @wam_argc()', [Base]),
    format(atom(ValueIR), '%~w', [Base]).
% Ternary `COND ? A : B` over i64 values. COND is a single comparison
% `L <op> R`; both branches are evaluated (no side effects in an i64 expr) and
% an LLVM `select` picks one -- straight-line, so a ternary composes anywhere an
% i64 expression is used (print, printf arg, assignment, arithmetic operand).
plawk_i64_expr_ir(ternary(cmp(CondLeft, Op, CondRight), Then, Else),
        FieldSeparator, Base, GlobalBase, ValueIR, GlobalParts, SetupParts) :-
    plawk_icmp_pred(Op, Pred),
    format(atom(CondLeftBase), '~w_cl', [Base]),
    format(atom(CondLeftGlobal), '~w_cl', [GlobalBase]),
    plawk_i64_expr_ir(CondLeft, FieldSeparator, CondLeftBase, CondLeftGlobal,
        CondLeftValueIR, CondLeftGlobalParts, CondLeftSetupParts),
    format(atom(CondRightBase), '~w_cr', [Base]),
    format(atom(CondRightGlobal), '~w_cr', [GlobalBase]),
    plawk_i64_expr_ir(CondRight, FieldSeparator, CondRightBase, CondRightGlobal,
        CondRightValueIR, CondRightGlobalParts, CondRightSetupParts),
    format(atom(ThenBase), '~w_t', [Base]),
    format(atom(ThenGlobal), '~w_t', [GlobalBase]),
    plawk_i64_expr_ir(Then, FieldSeparator, ThenBase, ThenGlobal,
        ThenValueIR, ThenGlobalParts, ThenSetupParts),
    format(atom(ElseBase), '~w_e', [Base]),
    format(atom(ElseGlobal), '~w_e', [GlobalBase]),
    plawk_i64_expr_ir(Else, FieldSeparator, ElseBase, ElseGlobal,
        ElseValueIR, ElseGlobalParts, ElseSetupParts),
    format(atom(CondLine), '  %~w_cond = icmp ~w i64 ~w, ~w',
        [Base, Pred, CondLeftValueIR, CondRightValueIR]),
    format(atom(SelLine), '  %~w = select i1 %~w_cond, i64 ~w, i64 ~w',
        [Base, Base, ThenValueIR, ElseValueIR]),
    format(atom(ValueIR), '%~w', [Base]),
    append([CondLeftGlobalParts, CondRightGlobalParts, ThenGlobalParts,
            ElseGlobalParts], GlobalParts),
    append([CondLeftSetupParts, CondRightSetupParts, ThenSetupParts,
            ElseSetupParts, [CondLine, SelLine]], SetupParts).

%% plawk_i64_binary_op_lines(+LLVMOp, +Base, +LeftIR, +RightIR, -Lines)
%
%  add/sub/mul emit one instruction. sdiv/srem are guarded so awk-side
%  division stays defined: a zero divisor yields 0, and the
%  INT64_MIN / -1 overflow case divides by 1 instead, wrapping to
%  INT64_MIN for `/` and 0 for `%`.
plawk_i64_binary_op_lines(sdiv, Base, LeftIR, RightIR, Lines) :-
    !,
    plawk_i64_guarded_div_lines(sdiv, Base, LeftIR, RightIR, Lines).
plawk_i64_binary_op_lines(srem, Base, LeftIR, RightIR, Lines) :-
    !,
    plawk_i64_guarded_div_lines(srem, Base, LeftIR, RightIR, Lines).
plawk_i64_binary_op_lines(LLVMOp, Base, LeftIR, RightIR, [Line]) :-
    format(atom(Line), '  %~w = ~w i64 ~w, ~w',
        [Base, LLVMOp, LeftIR, RightIR]).

%% plawk_expr_is_double(+Expr) is semidet.
%
%  An expression tree is double-typed when any leaf is a float literal
%  or a float($N) coercion; i64 leaves in a double tree promote via
%  sitofp at emission time.
plawk_expr_is_double(float_const(_Mantissa, _Denominator)).
plawk_expr_is_double(float_field(_Index)).
plawk_expr_is_double(float_call(_Name, _Args)).
plawk_expr_is_double(float_dyncall(_Args)).
plawk_expr_is_double(float_dyncall_named(_Name, _Args)).
plawk_expr_is_double(float_dyncall_at(_Source, _Args)).
plawk_expr_is_double(float_dyncall_at_named(_Name, _Source, _Args)).
% A substituted read of a double scalar slot (see
% plawk_substitute_scalar_reads/4).
plawk_expr_is_double(ssa_f64(_Value)).
plawk_expr_is_double(Expr) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    ( plawk_expr_is_double(Left)
    ; plawk_expr_is_double(Right)
    ).

%% plawk_f64_print_expr(+Expr) is semidet.
%
%  Valid double print expression: double-typed with recognizable leaves.
plawk_f64_print_expr(Expr) :-
    plawk_expr_is_double(Expr),
    plawk_f64_operand_expr(Expr).

plawk_f64_operand_expr(float_const(Mantissa, Denominator)) :-
    integer(Mantissa),
    integer(Denominator),
    Denominator > 0.
plawk_f64_operand_expr(float_field(Index)) :-
    integer(Index),
    Index >= 0.
plawk_f64_operand_expr(float_call(Name, Args)) :-
    plawk_prolog_call_expr(prolog_call(Name, Args)).
plawk_f64_operand_expr(float_dyncall(Args)) :-
    plawk_float_dyncall_expr(float_dyncall(Args)).
plawk_f64_operand_expr(float_dyncall_named(Name, Args)) :-
    plawk_float_dyncall_named_expr(float_dyncall_named(Name, Args)).
plawk_f64_operand_expr(float_dyncall_at(Source, Args)) :-
    plawk_float_dyncall_at_expr(float_dyncall_at(Source, Args)).
plawk_f64_operand_expr(float_dyncall_at_named(Name, Source, Args)) :-
    plawk_float_dyncall_at_expr(float_dyncall_at_named(Name, Source, Args)).
plawk_f64_operand_expr(Expr) :-
    plawk_i64_operand_expr(Expr).
plawk_f64_operand_expr(Expr) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    plawk_f64_operand_expr(Left),
    plawk_f64_operand_expr(Right).

%% plawk_f64_expr_ir(+Expr, +FieldSeparator, +Base, +GlobalBase, -ValueIR,
%%     -GlobalParts, -SetupParts)
%
%  Double expression emitter. Float literals emit as an exact integer
%  ratio (fdiv of two exactly representable doubles gives the correctly
%  rounded value, matching strtod). i64 subtrees emit through the i64
%  emitter and promote with sitofp; IEEE semantics apply to / (no
%  divide-by-zero guard: x/0.0 is inf, 0.0/0.0 is nan, as in awk).
plawk_f64_expr_ir(ssa_f64(Value), _FieldSeparator, _Base, _GlobalBase,
        Value, [], []) :-
    !.
plawk_f64_expr_ir(float_const(Mantissa, Denominator), _FieldSeparator, Base,
        _GlobalBase, ValueIR, [], [ConstIR]) :-
    format(atom(ValueIR), '%~w', [Base]),
    format(atom(ConstIR), '  ~w = fdiv double ~w.0, ~w.0',
        [ValueIR, Mantissa, Denominator]).
plawk_f64_expr_ir(float_field(Index), binfmt(Types), Base, _GlobalBase,
        ValueIR, [], LoadLines) :-
    !,
    plawk_binfmt_field_load_lines(binfmt(Types), Index, Base, ValueIR,
        LoadLines).
plawk_f64_expr_ir(float_field(Index), FieldSeparator, Base, _GlobalBase,
        ValueIR, [], [CallIR]) :-
    format(atom(ValueIR), '%~w', [Base]),
    format(atom(CallIR),
        '  ~w = call double @wam_atom_field_f64_value(%Value %line, i64 ~w, i8 ~w)',
        [ValueIR, Index, FieldSeparator]).
% a strnum read in f64 arithmetic (step 3c): coerce the atom id to double via
% strtod, matching the f64 field read (@wam_atom_field_f64_value also uses strtod).
plawk_f64_expr_ir(ssa_strnum(Value), _FieldSeparator, Base, _GlobalBase,
        ValueIR, [], [StrCall, ParseCall]) :-
    format(atom(ValueIR), '%~w_snf_d', [Base]),
    format(atom(StrCall), '  %~w_snf_s = call i8* @wam_atom_to_string(i64 ~w)',
        [Base, Value]),
    format(atom(ParseCall),
        '  ~w = call double @strtod(i8* %~w_snf_s, i8** null)', [ValueIR, Base]).
% float(name(args)): the double-returning foreign call. A failed call
% contributes 0.0, mirroring the i64 prolog_call contract.
plawk_f64_expr_ir(float_dyncall(Args), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    !,
    length(Args, NArgs),
    plawk_foreign_args_ir(Args, FieldSeparator, GlobalBase, ArgValueIRs,
        GlobalParts, ArgSetupParts),
    plawk_foreign_call_args_ir(ArgValueIRs, CallArgsIR),
    format(atom(ResIR),
        '  %~w_res = call { double, i1 } @plawk_dyncall_f_~w(~w)',
        [Base, NArgs, CallArgsIR]),
    plawk_f64_call_tail_ir(Base, ResIR, ArgSetupParts, ValueIR, SetupParts).
plawk_f64_expr_ir(float_dyncall_named(Name, Args), FieldSeparator, Base,
        GlobalBase, ValueIR, GlobalParts, SetupParts) :-
    !,
    length(Args, NArgs),
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    plawk_foreign_args_ir(Args, FieldSeparator, GlobalBase, ArgValueIRs,
        GlobalParts, ArgSetupParts),
    plawk_foreign_call_args_ir(ArgValueIRs, CallArgsIR),
    format(atom(ResIR),
        '  %~w_res = call { double, i1 } @plawk_dyncall_named_f_~w(~w)',
        [Base, Sym, CallArgsIR]),
    plawk_f64_call_tail_ir(Base, ResIR, ArgSetupParts, ValueIR, SetupParts).
plawk_f64_expr_ir(float_dyncall_at(Source, Args), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    !,
    length(Args, NArgs),
    plawk_dyncall_source_ir(Source, FieldSeparator, Base, GlobalBase,
        PathPtrIR, PathLenIR, SrcGlobals, SrcSetup),
    plawk_foreign_args_ir(Args, FieldSeparator, GlobalBase, ArgValueIRs,
        ArgGlobals, ArgSetup),
    ( ArgValueIRs == []
    -> CallArgsSuffix = ''
    ;  plawk_foreign_call_args_ir(ArgValueIRs, AV),
       format(atom(CallArgsSuffix), ', ~w', [AV])
    ),
    format(atom(ResIR),
        '  %~w_res = call { double, i1 } @plawk_dyncall_at_f_~w(i8* ~w, i64 ~w~w)',
        [Base, NArgs, PathPtrIR, PathLenIR, CallArgsSuffix]),
    append(SrcGlobals, ArgGlobals, GlobalParts),
    append(SrcSetup, ArgSetup, PreSetup),
    plawk_f64_call_tail_ir(Base, ResIR, PreSetup, ValueIR, SetupParts).
plawk_f64_expr_ir(float_dyncall_at_named(Name, Source, Args), FieldSeparator,
        Base, GlobalBase, ValueIR, GlobalParts, SetupParts) :-
    !,
    length(Args, NArgs),
    plawk_dyncall_named_symbol(Name, NArgs, Sym),
    plawk_dyncall_source_ir(Source, FieldSeparator, Base, GlobalBase,
        PathPtrIR, PathLenIR, SrcGlobals, SrcSetup),
    plawk_foreign_args_ir(Args, FieldSeparator, GlobalBase, ArgValueIRs,
        ArgGlobals, ArgSetup),
    ( ArgValueIRs == []
    -> CallArgsSuffix = ''
    ;  plawk_foreign_call_args_ir(ArgValueIRs, AV),
       format(atom(CallArgsSuffix), ', ~w', [AV])
    ),
    format(atom(ResIR),
        '  %~w_res = call { double, i1 } @plawk_dyncall_at_named_f_~w(i8* ~w, i64 ~w~w)',
        [Base, Sym, PathPtrIR, PathLenIR, CallArgsSuffix]),
    append(SrcGlobals, ArgGlobals, GlobalParts),
    append(SrcSetup, ArgSetup, PreSetup),
    plawk_f64_call_tail_ir(Base, ResIR, PreSetup, ValueIR, SetupParts).
plawk_f64_expr_ir(float_call(Name, Args), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts) :-
    !,
    length(Args, NArgs),
    plawk_foreign_args_ir(Args, FieldSeparator, GlobalBase, ArgValueIRs,
        GlobalParts, ArgSetupParts),
    plawk_foreign_call_args_ir(ArgValueIRs, CallArgsIR),
    format(atom(ResIR),
        '  %~w_res = call { double, i1 } @plawk_foreign_fcall_~w_~w(~w)',
        [Base, Name, NArgs, CallArgsIR]),
    format(atom(ValIR),
        '  %~w_val = extractvalue { double, i1 } %~w_res, 0', [Base, Base]),
    format(atom(OkIR),
        '  %~w_ok = extractvalue { double, i1 } %~w_res, 1', [Base, Base]),
    format(atom(SelIR),
        '  %~w = select i1 %~w_ok, double %~w_val, double 0.0',
        [Base, Base, Base]),
    format(atom(ValueIR), '%~w', [Base]),
    append(ArgSetupParts, [ResIR, ValIR, OkIR, SelIR], SetupParts).
plawk_f64_expr_ir(Expr, FieldSeparator, Base, GlobalBase, ValueIR,
        GlobalParts, SetupParts) :-
    plawk_expr_is_double(Expr),
    plawk_i64_binary_expr(Expr, LLVMOp, _NamePart, Left, Right),
    !,
    plawk_f64_llvm_op(LLVMOp, F64Op),
    format(atom(LeftBase), '~w_lhs', [Base]),
    format(atom(LeftGlobalBase), '~w_lhs', [GlobalBase]),
    plawk_f64_expr_ir(Left, FieldSeparator, LeftBase, LeftGlobalBase,
        LeftValueIR, LeftGlobalParts, LeftSetupParts),
    format(atom(RightBase), '~w_rhs', [Base]),
    format(atom(RightGlobalBase), '~w_rhs', [GlobalBase]),
    plawk_f64_expr_ir(Right, FieldSeparator, RightBase, RightGlobalBase,
        RightValueIR, RightGlobalParts, RightSetupParts),
    format(atom(ValueIR), '%~w', [Base]),
    format(atom(OpIR), '  ~w = ~w double ~w, ~w',
        [ValueIR, F64Op, LeftValueIR, RightValueIR]),
    append(LeftGlobalParts, RightGlobalParts, GlobalParts),
    append([LeftSetupParts, RightSetupParts, [OpIR]], SetupParts).
plawk_f64_expr_ir(Expr, FieldSeparator, Base, GlobalBase, ValueIR,
        GlobalParts, SetupParts) :-
    % i64-typed subtree in a double context: emit as i64, then promote.
    format(atom(IntBase), '~w_int', [Base]),
    format(atom(IntGlobalBase), '~w_int', [GlobalBase]),
    plawk_i64_expr_ir(Expr, FieldSeparator, IntBase, IntGlobalBase,
        IntValueIR, GlobalParts, IntSetupParts),
    format(atom(ValueIR), '%~w', [Base]),
    format(atom(PromoteIR), '  ~w = sitofp i64 ~w to double',
        [ValueIR, IntValueIR]),
    append(IntSetupParts, [PromoteIR], SetupParts).

plawk_f64_llvm_op(add, fadd).
plawk_f64_llvm_op(sub, fsub).
plawk_f64_llvm_op(mul, fmul).
plawk_f64_llvm_op(sdiv, fdiv).
plawk_f64_llvm_op(srem, frem).

plawk_i64_guarded_div_lines(LLVMOp, Base, LeftIR, RightIR, Lines) :-
    format(atom(DenZero),
        '  %~w_den_zero = icmp eq i64 ~w, 0', [Base, RightIR]),
    format(atom(LhsMin),
        '  %~w_lhs_min = icmp eq i64 ~w, -9223372036854775808', [Base, LeftIR]),
    format(atom(RhsNegOne),
        '  %~w_rhs_negone = icmp eq i64 ~w, -1', [Base, RightIR]),
    format(atom(Overflow),
        '  %~w_overflow = and i1 %~w_lhs_min, %~w_rhs_negone',
        [Base, Base, Base]),
    format(atom(DenBad),
        '  %~w_den_bad = or i1 %~w_den_zero, %~w_overflow', [Base, Base, Base]),
    format(atom(SafeDen),
        '  %~w_safe_den = select i1 %~w_den_bad, i64 1, i64 ~w',
        [Base, Base, RightIR]),
    format(atom(Raw),
        '  %~w_raw = ~w i64 ~w, %~w_safe_den', [Base, LLVMOp, LeftIR, Base]),
    format(atom(Result),
        '  %~w = select i1 %~w_den_zero, i64 0, i64 %~w_raw',
        [Base, Base, Base]),
    Lines = [DenZero, LhsMin, RhsNegOne, Overflow, DenBad, SafeDen, Raw, Result].

plawk_branch_to_done_ir(none, _DoneLabel, '  br label %continue_loop') :-
    !.
% `break` inside a loop leaves that loop (branch to its exit label); at rule
% level it keeps its stream-break meaning. `continue` (only valid in a loop)
% branches to the loop's continue target. The enclosing loop is read from the
% loop-context stack (plawk_loopctx_current).
plawk_branch_to_done_ir(break, _DoneLabel, IR) :-
    !,
    (   plawk_loopctx_current(loop_ctx(BreakLabel, _ContinueLabel))
    ->  format(atom(IR), '  br label %~w', [BreakLabel])
    ;   IR = '  br label %break_close_stream'
    ).
plawk_branch_to_done_ir(continue, _DoneLabel, IR) :-
    !,
    plawk_loopctx_current(loop_ctx(_BreakLabel, ContinueLabel)),
    format(atom(IR), '  br label %~w', [ContinueLabel]).
% `exit` ends the program regardless of any enclosing loop: always branch to
% break_close_stream (which runs END and returns @plawk_exit_code).
plawk_branch_to_done_ir(exit, _DoneLabel, '  br label %break_close_stream') :-
    !.
plawk_branch_to_done_ir(ExitLabel, DoneLabel, IR) :-
    ExitLabel \== none,
    ExitLabel \== break,
    ExitLabel \== continue,
    ExitLabel \== exit,
    format(atom(IR), '  br label %~w', [DoneLabel]).

% The loop-context stack: each loop pushes loop_ctx(BreakLabel, ContinueLabel)
% around its body, so a break/continue anywhere in the body (including nested
% ifs) branches to the innermost loop's labels. Non-backtrackable global with
% manual push/pop; nested loops stack naturally.
plawk_loopctx_push(Ctx) :-
    ( nb_current(plawk_loopctx, Stack) -> true ; Stack = [] ),
    nb_setval(plawk_loopctx, [Ctx | Stack]).
plawk_loopctx_pop :-
    ( nb_current(plawk_loopctx, [_ | Stack]) -> nb_setval(plawk_loopctx, Stack) ; true ).
plawk_loopctx_current(Ctx) :-
    nb_current(plawk_loopctx, [Ctx | _]).

plawk_branch_terminal_exit(none).
plawk_branch_terminal_exit(break).
plawk_branch_terminal_exit(continue).
% `exit` inside an if branch leaves via break_close_stream, so (like break) that
% branch does not flow into the if-join phi -- the join takes the other branch.
plawk_branch_terminal_exit(exit).

plawk_scalar_if_join_pairs(ThenExitLabel, _ThenValues, ElseExitLabel, _ElseValues, _Slots, _Prefix, _OpIndex, _Pairs) :-
    plawk_branch_terminal_exit(ThenExitLabel),
    plawk_branch_terminal_exit(ElseExitLabel),
    !,
    fail.
plawk_scalar_if_join_pairs(ThenExitLabel, _ThenValues, _ElseExitLabel, ElseValues, _Slots, _Prefix, _OpIndex, Pairs) :-
    plawk_branch_terminal_exit(ThenExitLabel),
    !,
    plawk_scalar_if_passthrough_pairs(ElseValues, Pairs).
plawk_scalar_if_join_pairs(_ThenExitLabel, ThenValues, ElseExitLabel, _ElseValues, _Slots, _Prefix, _OpIndex, Pairs) :-
    plawk_branch_terminal_exit(ElseExitLabel),
    !,
    plawk_scalar_if_passthrough_pairs(ThenValues, Pairs).
plawk_scalar_if_join_pairs(ThenExitLabel, ThenValues, ElseExitLabel, ElseValues, Slots, Prefix, OpIndex, Pairs) :-
    phrase(plawk_scalar_if_phi_lines(ThenValues, ElseValues, Slots, Prefix, OpIndex,
        ThenExitLabel, ElseExitLabel, 0), Pairs).

plawk_scalar_if_passthrough_pairs([], []).
plawk_scalar_if_passthrough_pairs([Value | Rest], [Value-'' | Pairs]) :-
    plawk_scalar_if_passthrough_pairs(Rest, Pairs).

plawk_assoc_update_operation_ir(Prefix, OpIndex, TableIndex, KeyIndex,
        FieldSeparator, ''-IR, DoneLabel) :-
    format(atom(Label), '~w_assoc_~w', [Prefix, OpIndex]),
    format(atom(HaveLabel), '~w_assoc_~w_have_key', [Prefix, OpIndex]),
    format(atom(DoneLabel), '~w_assoc_~w_done', [Prefix, OpIndex]),
    format(atom(SliceValue), '%~w_assoc_~w_key_slice', [Prefix, OpIndex]),
    format(atom(KeyPtr), '%~w_assoc_~w_key_ptr', [Prefix, OpIndex]),
    format(atom(KeyLen), '%~w_assoc_~w_key_len', [Prefix, OpIndex]),
    format(atom(KeyMissing), '%~w_assoc_~w_key_missing', [Prefix, OpIndex]),
    format(atom(KeyId), '%~w_assoc_~w_key_id', [Prefix, OpIndex]),
    format(atom(CountValue), '%~w_assoc_~w_count', [Prefix, OpIndex]),
    format(atom(IR),
'  br label %~w

~w:
  ~w = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)
  ~w = extractvalue %WamSlice ~w, 0
  ~w = extractvalue %WamSlice ~w, 1
  ~w = icmp eq i8* ~w, null
  br i1 ~w, label %~w, label %~w

~w:
  ~w = call i64 @wam_intern_atom(i8* ~w, i64 ~w)
  ~w = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %plawk_assoc_table_~w, i64 ~w, i64 1)
  br label %~w

~w:',
        [Label,
         Label,
         SliceValue, KeyIndex, FieldSeparator,
         KeyPtr, SliceValue,
         KeyLen, SliceValue,
         KeyMissing, KeyPtr,
         KeyMissing, DoneLabel, HaveLabel,
         HaveLabel,
         KeyId, KeyPtr, KeyLen,
         CountValue, TableIndex, KeyId,
         DoneLabel,
         DoneLabel]).

plawk_scalar_if_phi_lines([], [], _Slots, _Prefix, _OpIndex, _ThenLabel, _ElseLabel, _) -->
    [].
plawk_scalar_if_phi_lines([ThenValue | ThenRest], [ElseValue | ElseRest],
        [Slot | Slots], Prefix, OpIndex, ThenLabel, ElseLabel, SlotIndex) -->
    { format(atom(PhiValue), '%~w_if_~w_slot_~w', [Prefix, OpIndex, SlotIndex]),
      plawk_slot_llvm_type(Slot, Type),
      format(atom(Line), '  ~w = phi ~w [~w, %~w], [~w, %~w]',
          [PhiValue, Type, ThenValue, ThenLabel, ElseValue, ElseLabel]),
      NextSlotIndex is SlotIndex + 1
    },
    [PhiValue-Line],
    plawk_scalar_if_phi_lines(ThenRest, ElseRest, Slots, Prefix, OpIndex, ThenLabel, ElseLabel, NextSlotIndex).

%% plawk_foreach_head_phi_lines(+Slots, +InValues, +OutValues, +FeBase,
%%     +EntryLabel, +BodyDoneLabel, +Index)//
%
%  One loop-carried phi per scalar slot, typed by the slot.
plawk_foreach_head_phi_lines([], [], [], _FeBase, _EntryLabel, _DoneLabel, _) -->
    [].
plawk_foreach_head_phi_lines([Slot | Slots], [InValue | InValues],
        [OutValue | OutValues], FeBase, EntryLabel, DoneLabel, Index) -->
    { plawk_slot_llvm_type(Slot, Type),
      format(atom(Line),
          '  %~w_slot_~w = phi ~w [~w, %~w], [~w, %~w]',
          [FeBase, Index, Type, InValue, EntryLabel, OutValue, DoneLabel]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_foreach_head_phi_lines(Slots, InValues, OutValues, FeBase,
        EntryLabel, DoneLabel, NextIndex).

%% Loop break/continue merge phis (PLAWK_CONTROL_FLOW_PLAN.md 3b) --------------

% Partition a loop body's exits: loop-local `break` / `continue` (consumed by
% the loop) vs everything else (`next`, propagated to the record loop).
plawk_partition_loop_exits([], [], [], []).
plawk_partition_loop_exits([branch_break(L, V) | R], [branch_break(L, V) | Bs], Cs, Ns) :-
    !,
    plawk_partition_loop_exits(R, Bs, Cs, Ns).
plawk_partition_loop_exits([branch_continue(L, V) | R], Bs, [branch_continue(L, V) | Cs], Ns) :-
    !,
    plawk_partition_loop_exits(R, Bs, Cs, Ns).
plawk_partition_loop_exits([X | R], Bs, Cs, [X | Ns]) :-
    plawk_partition_loop_exits(R, Bs, Cs, Ns).

% A `while` head phi per slot: the loop-carried value merges the pre-loop value
% (from entry), the body's output (from the back edge), and each `continue`
% point's value.
plawk_while_head_phi_ir(Slots, InValues, OutValues, Continues, Base,
        EntryLabel, BodyDoneLabel, IR) :-
    phrase(plawk_while_head_phi_lines(Slots, InValues, OutValues, Continues,
        Base, EntryLabel, BodyDoneLabel, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_while_head_phi_lines([], [], [], _Cs, _Base, _E, _BD, _) -->
    [].
plawk_while_head_phi_lines([Slot | Slots], [In | Ins], [Out | Outs], Continues,
        Base, E, BD, I) -->
    { plawk_slot_llvm_type(Slot, Type),
      format(atom(BaseIncs), '[~w, %~w], [~w, %~w]', [In, E, Out, BD]),
      plawk_loop_exit_incomings(Continues, I, ContIncs),
      atomic_list_concat([BaseIncs | ContIncs], ', ', IncsIR),
      format(atom(Line), '  %~w_slot_~w = phi ~w ~w', [Base, I, Type, IncsIR]),
      I1 is I + 1
    },
    [Line],
    plawk_while_head_phi_lines(Slots, Ins, Outs, Continues, Base, E, BD, I1).

% The `after` block phi merging the normal loop exit (NormalValues from
% NormalLabel) with each `break` point's value. No breaks -> no phi, and the
% post-loop values are just the normal-exit values.
plawk_loop_after_ir(_Slots, NormalValues, _NormalLabel, [], _Base, NormalValues, '') :-
    !.
plawk_loop_after_ir(Slots, NormalValues, NormalLabel, Breaks, Base, AfterValues, IR) :-
    phrase(plawk_loop_after_phi_lines(Slots, NormalValues, NormalLabel, Breaks,
        Base, 0), Lines),
    atomic_list_concat(Lines, '\n', IR),
    findall(V,
        ( nth0(I, Slots, _), format(atom(V), '%~w_after_slot_~w', [Base, I]) ),
        AfterValues).

plawk_loop_after_phi_lines([], [], _NL, _Bs, _Base, _) -->
    [].
plawk_loop_after_phi_lines([Slot | Slots], [NV | NVs], NL, Breaks, Base, I) -->
    { plawk_slot_llvm_type(Slot, Type),
      format(atom(Normal), '[~w, %~w]', [NV, NL]),
      plawk_loop_exit_incomings(Breaks, I, BreakIncs),
      atomic_list_concat([Normal | BreakIncs], ', ', IncsIR),
      format(atom(Line), '  %~w_after_slot_~w = phi ~w ~w', [Base, I, Type, IncsIR]),
      I1 is I + 1
    },
    [Line],
    plawk_loop_after_phi_lines(Slots, NVs, NL, Breaks, Base, I1).

% One phi incoming per break/continue exit for slot I: [value-at-exit, %block].
plawk_loop_exit_incomings([], _I, []).
plawk_loop_exit_incomings([Exit | Rest], I, [Inc | Incs]) :-
    Exit =.. [_Kind, Label, Values],
    nth0(I, Values, V),
    format(atom(Inc), '[~w, %~w]', [V, Label]),
    plawk_loop_exit_incomings(Rest, I, Incs).

replace_nth0(0, [_Old | Rest], Value, [Value | Rest]) :-
    !.
replace_nth0(Index, [Head | Rest], Value, [Head | NewRest]) :-
    Index > 0,
    NextIndex is Index - 1,
    replace_nth0(NextIndex, Rest, Value, NewRest).

plawk_scalar_next_phi_ir(StatePlan, RuleCount, Controls, BranchNextExits, IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    phrase(plawk_scalar_next_phi_lines(Slots, RuleCount, Controls, BranchNextExits, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

plawk_scalar_next_phi_lines([], _RuleCount, _Controls, _BranchNextExits, _) -->
    [].
plawk_scalar_next_phi_lines([Slot | Rest], RuleCount, Controls, BranchNextExits, Index) -->
    { LastRuleIndex is RuleCount - 1,
      plawk_scalar_rule_input_value(LastRuleIndex, Index, FalseValue),
      format(atom(FalseIncoming), '[~w, %rule_~w_match]',
          [FalseValue, LastRuleIndex]),
      findall(ApplyIncoming,
          ( between(0, LastRuleIndex, RuleIndex),
            ( ( RuleIndex =:= LastRuleIndex,
                \+ ( nth0(RuleIndex, Controls, LastControl),
                     memberchk(LastControl, [terminal_break, terminal_exit]) )
              )
            ; nth0(RuleIndex, Controls, terminal_next)
            ),
            format(atom(ApplyIncoming), '[%rule_~w_slot_~w, %rule_~w_done]',
                [RuleIndex, Index, RuleIndex])
          ),
          ApplyIncomings),
      plawk_branch_next_phi_incomings(BranchNextExits, Index, BranchNextIncomings),
      append([FalseIncoming | ApplyIncomings], BranchNextIncomings, Incomings),
      atomic_list_concat(Incomings, ', ', IncomingIR),
      plawk_slot_llvm_type(Slot, Type),
      format(atom(Line), '  %next_slot_~w = phi ~w ~w', [Index, Type, IncomingIR]),
      NextIndex is Index + 1
    },
    [Line],
    plawk_scalar_next_phi_lines(Rest, RuleCount, Controls, BranchNextExits, NextIndex).

plawk_branch_next_phi_incomings([], _SlotIndex, []).
plawk_branch_next_phi_incomings([branch_next(Label, Values) | Rest], SlotIndex, [Incoming | Incomings]) :-
    !,
    nth0(SlotIndex, Values, Value),
    format(atom(Incoming), '[~w, %~w]', [Value, Label]),
    plawk_branch_next_phi_incomings(Rest, SlotIndex, Incomings).
plawk_branch_next_phi_incomings([_Exit | Rest], SlotIndex, Incomings) :-
    plawk_branch_next_phi_incomings(Rest, SlotIndex, Incomings).

plawk_scalar_end_print_ir(PrintFields, StatePlan, OutputSeparator, IR) :-
    phrase(plawk_scalar_end_print_lines(PrintFields, StatePlan, OutputSeparator, 0), Lines),
    atomic_list_concat(Lines, '\n', IR).

%% END { if (COND) print ...; [else print ...] } support ---------------------

% A supported END-if: each branch is a single print (else optional).
plawk_end_if_ok([print(_)], []).
plawk_end_if_ok([print(_)], [print(_)]).

% The scalar fields the state plan must see: the branch print fields (so a
% printed scalar gets a slot) plus the condition variables. String literals need
% no slot, but leaving them in is harmless -- the state plan ignores non-vars.
plawk_end_if_print_fields(Cond, ThenActions, ElseActions, Fields) :-
    plawk_end_if_branch_fields(ThenActions, ThenFields),
    plawk_end_if_branch_fields(ElseActions, ElseFields),
    plawk_while_cond_vars(Cond, CondVars),
    plawk_vars_as_fields(CondVars, CondFields),
    append([ThenFields, ElseFields, CondFields], Fields).

plawk_end_if_branch_fields([print(Fields)], Fields).
plawk_end_if_branch_fields([], []).

plawk_vars_as_fields([], []).
plawk_vars_as_fields([V | Vs], [var(V) | Fs]) :-
    plawk_vars_as_fields(Vs, Fs).

% The END-if body IR: evaluate COND against the final slot values, branch to a
% then/else print block, join. Each branch print uses the PREFIXED print emitter
% (unique names per branch, so the two blocks' temporaries don't collide) with
% scalar reads substituted to their final-slot SSA values. Returns the string
% globals the branch prints need, plus the body IR.
plawk_scalar_end_if_ir(Cond, ThenActions, ElseActions, StatePlan, FieldSeparator,
        OutputSeparator, GlobalIR, IR) :-
    plawk_state_plan_slots(StatePlan, Slots),
    plawk_final_slot_values(StatePlan, FinalValues),
    plawk_while_cond_ir(Cond, Slots, FinalValues, FieldSeparator, plawk_endif, CondVar, CondIR),
    plawk_end_if_branch_ir(ThenActions, Slots, FinalValues, FieldSeparator,
        OutputSeparator, plawk_endif_then, ThenGlobal, ThenIR),
    plawk_end_if_branch_ir(ElseActions, Slots, FinalValues, FieldSeparator,
        OutputSeparator, plawk_endif_else, ElseGlobal, ElseIR),
    plawk_join_nonempty_ir([ThenGlobal, ElseGlobal], GlobalIR),
    format(atom(IR),
'~w
  br i1 ~w, label %plawk_endif_then, label %plawk_endif_else

plawk_endif_then:
~w
  br label %plawk_endif_done

plawk_endif_else:
~w
  br label %plawk_endif_done

plawk_endif_done:',
        [CondIR, CondVar, ThenIR, ElseIR]).

plawk_end_if_branch_ir([print(Fields)], Slots, FinalValues, FieldSeparator,
        OutputSeparator, Prefix, GlobalIR, IR) :-
    maplist(plawk_substitute_print_field(Slots, FinalValues), Fields, SubFields),
    plawk_prefixed_print_action_ir(SubFields, FieldSeparator, OutputSeparator,
        Prefix, GlobalIR-IR).
plawk_end_if_branch_ir([], _Slots, _FinalValues, _FieldSeparator,
        _OutputSeparator, _Prefix, '', '').

% The final (post-loop) slot values, one per slot: %final_slot_0, %final_slot_1,
% ... -- the values the END block reads.
plawk_final_slot_values(StatePlan, Values) :-
    plawk_state_plan_slots(StatePlan, Slots),
    findall(V,
        ( nth0(I, Slots, _Slot), format(atom(V), '%final_slot_~w', [I]) ),
        Values).

plawk_scalar_end_print_lines([], _StatePlan, _OutputSeparator, _) -->
    { llvm_emit_printf0(plawk_surface_print_newline, 2,
          end_newline_fmt, printed_end_newline, [FmtPtr, PrintCall])
    },
    [FmtPtr, PrintCall].
plawk_scalar_end_print_lines([var(Name) | Rest], StatePlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    { plawk_state_slot_lookup(StatePlan, Name, SlotIndex, Slot),
      format(atom(ValueIR), '%final_slot_~w', [SlotIndex]),
      ( Slot = scalar_double(_Name)
      -> format(atom(FmtVar), 'end_f64_fmt_~w', [PrintIndex]),
         format(atom(FmtPtr),
             '  %~w = getelementptr [3 x i8], [3 x i8]* @.plawk_surface_print_f64, i32 0, i32 0',
             [FmtVar]),
         format(atom(PrintCall),
             '  %printed_end_f64_~w = call i32 (i8*, ...) @printf(i8* %~w, double ~w)',
             [PrintIndex, FmtVar, ValueIR]),
         Lines = [FmtPtr, PrintCall]
      ;  ( Slot = scalar_string(_Name) ; Slot = scalar_strnum(_Name) )
      -> % resolve the atom id to text; id 0 (unset) prints as empty (the `%s\0`
         % global's trailing NUL is a ready-made empty C string). A strnum slot
         % holds an atom id too, so it prints identically.
         format(atom(StrS), '  %end_str_s_~w = call i8* @wam_atom_to_string(i64 ~w)',
             [PrintIndex, ValueIR]),
         format(atom(StrE), '  %end_str_empty_~w = icmp eq i64 ~w, 0',
             [PrintIndex, ValueIR]),
         format(atom(StrSel),
             '  %end_str_ptr_~w = select i1 %end_str_empty_~w, i8* getelementptr ([3 x i8], [3 x i8]* @.plawk_surface_print_string, i64 0, i64 2), i8* %end_str_s_~w',
             [PrintIndex, PrintIndex, PrintIndex]),
         format(atom(FmtPtr),
             '  %end_str_fmt_~w = getelementptr [3 x i8], [3 x i8]* @.plawk_surface_print_string, i32 0, i32 0',
             [PrintIndex]),
         format(atom(PrintCall),
             '  %printed_end_str_~w = call i32 (i8*, ...) @printf(i8* %end_str_fmt_~w, i8* %end_str_ptr_~w)',
             [PrintIndex, PrintIndex, PrintIndex]),
         Lines = [StrS, StrE, StrSel, FmtPtr, PrintCall]
      ;  format(atom(FmtVar), 'end_i64_fmt_~w', [PrintIndex]),
         format(atom(PrintVar), 'printed_end_i64_~w', [PrintIndex]),
         llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar, ValueIR,
             [FmtPtr, PrintCall]),
         Lines = [FmtPtr, PrintCall]
      ),
      NextPrintIndex is PrintIndex + 1
    },
    Lines,
    plawk_scalar_end_print_lines(Rest, StatePlan, OutputSeparator, NextPrintIndex).
plawk_scalar_end_print_lines([special('NR') | Rest], StatePlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_nr_print_lines(PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_scalar_end_print_lines(Rest, StatePlan, OutputSeparator, NextPrintIndex).
plawk_scalar_end_print_lines([Expr | Rest], StatePlan, OutputSeparator, PrintIndex) -->
    { plawk_end_scalar_expr(Expr) },
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_expr_print_lines(Expr, StatePlan, PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_scalar_end_print_lines(Rest, StatePlan, OutputSeparator, NextPrintIndex).
plawk_scalar_end_print_lines([string(Value) | Rest], StatePlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_string_print_lines(Value, PrintIndex),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_scalar_end_print_lines(Rest, StatePlan, OutputSeparator, NextPrintIndex).
% concatenation in END (`print "total: " sum`): one leading separator, then each
% operand printed adjacently (no separator between). Each part uses a unique
% index (PrintIndex*1000 + part) so the fixed-name emitters don't collide.
plawk_scalar_end_print_lines([concat(Parts) | Rest], StatePlan, OutputSeparator, PrintIndex) -->
    plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator),
    plawk_end_concat_parts(Parts, StatePlan, PrintIndex, 0),
    { NextPrintIndex is PrintIndex + 1 },
    plawk_scalar_end_print_lines(Rest, StatePlan, OutputSeparator, NextPrintIndex).

plawk_end_concat_parts([], _StatePlan, _PrintIndex, _PartIndex) -->
    [].
plawk_end_concat_parts([Part | Rest], StatePlan, PrintIndex, PartIndex) -->
    { CombinedIndex is PrintIndex * 1000 + PartIndex },
    plawk_end_field_print_lines(Part, StatePlan, CombinedIndex),
    { NextPartIndex is PartIndex + 1 },
    plawk_end_concat_parts(Rest, StatePlan, PrintIndex, NextPartIndex).

% One END print field WITHOUT a leading separator (the concat driver adds the
% single separator before the whole concatenation).
plawk_end_field_print_lines(var(Name), StatePlan, PrintIndex) -->
    { plawk_state_slot_lookup(StatePlan, Name, SlotIndex, Slot),
      format(atom(ValueIR), '%final_slot_~w', [SlotIndex]),
      ( Slot = scalar_double(_Name)
      -> format(atom(FmtVar), 'end_f64_fmt_~w', [PrintIndex]),
         format(atom(FmtPtr),
             '  %~w = getelementptr [3 x i8], [3 x i8]* @.plawk_surface_print_f64, i32 0, i32 0',
             [FmtVar]),
         format(atom(PrintCall),
             '  %printed_end_f64_~w = call i32 (i8*, ...) @printf(i8* %~w, double ~w)',
             [PrintIndex, FmtVar, ValueIR])
      ;  format(atom(FmtVar), 'end_i64_fmt_~w', [PrintIndex]),
         format(atom(PrintVar), 'printed_end_i64_~w', [PrintIndex]),
         llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar, ValueIR,
             [FmtPtr, PrintCall])
      )
    },
    [FmtPtr, PrintCall].
plawk_end_field_print_lines(string(Value), _StatePlan, PrintIndex) -->
    plawk_end_string_print_lines(Value, PrintIndex).
plawk_end_field_print_lines(special('NR'), _StatePlan, PrintIndex) -->
    plawk_end_nr_print_lines(PrintIndex).
plawk_end_field_print_lines(Expr, StatePlan, PrintIndex) -->
    { plawk_end_scalar_expr(Expr) },
    plawk_end_expr_print_lines(Expr, StatePlan, PrintIndex).

plawk_end_nr_print_lines(PrintIndex) -->
    { format(atom(FmtVar), 'end_nr_fmt_~w', [PrintIndex]),
      format(atom(PrintVar), 'printed_end_nr_~w', [PrintIndex]),
      llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar, '%plawk_nr',
          [FmtPtr, PrintCall])
    },
    [FmtPtr, PrintCall].

plawk_end_expr_print_lines(Expr, StatePlan, PrintIndex) -->
    { plawk_substitute_end_reads(Expr, StatePlan, SubstitutedExpr),
      format(atom(Base), 'plawk_end_expr_~w', [PrintIndex]),
      format(atom(FmtVar), 'end_expr_fmt_~w', [PrintIndex]),
      ( plawk_expr_is_double(SubstitutedExpr)
      -> % A double-typed END expression (float literal leaf or a read of
         % a double slot): whole tree promotes to double, prints as %g,
         % and division is IEEE fdiv rather than guarded sdiv.
         plawk_f64_expr_ir(SubstitutedExpr, 32, Base, Base, ValueIR,
             [], SetupParts),
         format(atom(FmtPtr),
             '  %~w = getelementptr [3 x i8], [3 x i8]* @.plawk_surface_print_f64, i32 0, i32 0',
             [FmtVar]),
         format(atom(PrintCall),
             '  %printed_end_expr_f64_~w = call i32 (i8*, ...) @printf(i8* %~w, double ~w)',
             [PrintIndex, FmtVar, ValueIR])
      ;  plawk_i64_expr_ir(SubstitutedExpr, 32, Base, Base, ValueIR,
             [], SetupParts),
         format(atom(PrintVar), 'printed_end_expr_~w', [PrintIndex]),
         llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar,
             ValueIR, [FmtPtr, PrintCall])
      ),
      append(SetupParts, [FmtPtr, PrintCall], Lines)
    },
    plawk_emit_lines(Lines).

plawk_emit_lines([]) -->
    [].
plawk_emit_lines([Line | Rest]) -->
    [Line],
    plawk_emit_lines(Rest).

plawk_end_string_print_lines(Value, PrintIndex) -->
    { string_codes(Value, Codes),
      length(Codes, StringLen),
      BytesLen is StringLen + 1,
      format(atom(StringPtr),
          '  %end_string_~w_ptr = getelementptr [~w x i8], [~w x i8]* @.plawk_end_print_string_~w, i32 0, i32 0',
          [PrintIndex, BytesLen, BytesLen, PrintIndex]),
      format(atom(FmtVar), 'end_string_fmt_~w', [PrintIndex]),
      format(atom(PrintVar), 'printed_end_string_~w', [PrintIndex]),
      format(atom(PtrIR), '%end_string_~w_ptr', [PrintIndex]),
      llvm_emit_printf_string(plawk_surface_print_string, FmtVar, PrintVar, PtrIR,
          [FmtPtr, PrintCall])
    },
    [StringPtr, FmtPtr, PrintCall].

plawk_scalar_end_separator_lines(0, _OutputSeparator) -->
    !,
    [].
plawk_scalar_end_separator_lines(PrintIndex, OutputSeparator) -->
    { format(atom(SpaceCall),
          '  %printed_end_separator_~w = call i32 @putchar(i32 ~w)',
          [PrintIndex, OutputSeparator])
    },
    [SpaceCall].

plawk_pattern_guard_ir(always, GuardIR) :-
    GuardIR = ''-'  %is_match = icmp eq i1 true, true'.

plawk_pattern_guard_ir(prefix(Prefix), GuardIR) :-
    llvm_emit_atom_prefix_guard(plawk_surface_prefix, '%line', Prefix,
        '%is_match', GuardIR).
plawk_pattern_guard_ir(contains(Needle), GuardIR) :-
    plawk_pattern_guard_ir(contains(Needle), 32, GuardIR).

plawk_pattern_guard_ir(field_eq(Index, Value), GuardIR) :-
    plawk_pattern_guard_ir(field_eq(Index, Value), 32, GuardIR).
plawk_pattern_guard_ir(field_cmp(Index, Op, Value), GuardIR) :-
    plawk_pattern_guard_ir(field_cmp(Index, Op, Value), 32, GuardIR).

plawk_pattern_guard_ir(always, _FieldSeparator, GuardIR) :-
    plawk_pattern_guard_ir(always, GuardIR).
plawk_pattern_guard_ir(prefix(Prefix), _FieldSeparator, GuardIR) :-
    plawk_pattern_guard_ir(prefix(Prefix), GuardIR).
plawk_pattern_guard_ir(contains(Needle), FieldSeparator, GuardIR) :-
    plawk_literal_contains_guard_ir(plawk_surface_contains, Needle, FieldSeparator,
        '%is_match', GuardIR).
plawk_pattern_guard_ir(field_eq(Index, Value), binfmt(Types), GuardIR) :-
    !,
    plawk_binfmt_field_eq_guard_ir(binfmt(Types), Index, Value,
        plawk_surface_binfeq, '%is_match', GuardIR).
plawk_pattern_guard_ir(field_eq(Index, Value), FieldSeparator, GuardIR) :-
    llvm_emit_atom_field_eq_guard(plawk_surface_field_eq, '%line', Index, Value,
        FieldSeparator, '%is_match', GuardIR).
plawk_pattern_guard_ir(field_cmp(Index, Op, Value), binfmt(Types), GuardIR) :-
    !,
    plawk_binfmt_field_cmp_guard_ir(binfmt(Types), field_cmp(Index, Op, Value),
        plawk_surface_bincmp, '%is_match', GuardIR).
plawk_pattern_guard_ir(field_cmp(Index, Op, Value), FieldSeparator, ''-GuardCallIR) :-
    plawk_field_cmp_op_code(Op, OpCode),
    llvm_emit_atom_field_i64_cmp_guard('%line', Index, OpCode, Value,
        FieldSeparator, '%is_match', GuardCallIR).
plawk_pattern_guard_ir(field_match(Index, Regex), FieldSeparator, GuardIR) :-
    llvm_emit_regex_field_match_guard(plawk_surface_regex, '%line', Index,
        Regex, FieldSeparator, '%is_match', GuardIR).
plawk_pattern_guard_ir(blob_eq(Blob, Value), FieldSeparator, GuardIR) :-
    plawk_pattern_guard_ir(blob_eq(Blob, Value), FieldSeparator,
        plawk_surface_blob_eq, '%is_match', GuardIR).
plawk_pattern_guard_ir(prolog_guard(Name, Args), FieldSeparator, GuardIR) :-
    plawk_foreign_guard_call_ir(Name, Args, FieldSeparator,
        plawk_surface_prolog_guard, '%is_match', GuardIR).
plawk_pattern_guard_ir(Pattern, FieldSeparator, GuardIR) :-
    plawk_combined_pattern(Pattern),
    plawk_pattern_guard_ir(Pattern, FieldSeparator, plawk_surface_pattern,
        '%is_match', GuardIR).

plawk_pattern_guard_ir(always, _GlobalBase, MatchValue, GuardIR) :-
    format(atom(GuardCallIR), '  ~w = icmp eq i1 true, true', [MatchValue]),
    GuardIR = ''-GuardCallIR.

plawk_pattern_guard_ir(prefix(Prefix), GlobalBase, MatchValue, GuardIR) :-
    llvm_emit_atom_prefix_guard(GlobalBase, '%line', Prefix, MatchValue,
        GuardIR).
plawk_pattern_guard_ir(contains(Needle), GlobalBase, MatchValue, GuardIR) :-
    plawk_pattern_guard_ir(contains(Needle), 32, GlobalBase, MatchValue, GuardIR).

plawk_pattern_guard_ir(field_eq(Index, Value), GlobalBase, MatchValue, GuardIR) :-
    plawk_pattern_guard_ir(field_eq(Index, Value), 32, GlobalBase, MatchValue,
        GuardIR).
plawk_pattern_guard_ir(field_cmp(Index, Op, Value), GlobalBase, MatchValue, GuardIR) :-
    plawk_pattern_guard_ir(field_cmp(Index, Op, Value), 32, GlobalBase,
        MatchValue, GuardIR).

% Tagged-union rule guard: the record tag must equal the rule's arm,
% and the inner pattern (typed against the arm's layout) must match.
% %vr_tag is loaded once per record in the union read sequence and
% dominates every rule block. Field reads in the inner guard are safe
% even when the tag check fails: the record buffer is always at least
% as large as the widest arm, so a mismatched read sees stale-but-owned
% bytes and its result is discarded by the and.
plawk_pattern_guard_ir(arm_pat(Tag, ArmTypes, Pattern), _FieldSeparator,
        GlobalBase, MatchValue, GuardIR) :-
    !,
    format(atom(TagOk), '%~w_tag_ok', [GlobalBase]),
    format(atom(TagCheck), '  ~w = icmp eq i64 %vr_tag, ~w', [TagOk, Tag]),
    ( Pattern == always
    ->  format(atom(CallIR), '~w~n  ~w = and i1 ~w, true',
            [TagCheck, MatchValue, TagOk]),
        GuardIR = ''-CallIR
    ;   format(atom(InnerBase), '~w_arm', [GlobalBase]),
        format(atom(InnerValue), '~w_arm', [MatchValue]),
        plawk_pattern_guard_ir(Pattern, binfmt(ArmTypes), InnerBase,
            InnerValue, GlobalIR-InnerCallIR),
        format(atom(CallIR), '~w~n~w~n  ~w = and i1 ~w, ~w',
            [TagCheck, InnerCallIR, MatchValue, TagOk, InnerValue]),
        GuardIR = GlobalIR-CallIR
    ).
plawk_pattern_guard_ir(always, _FieldSeparator, GlobalBase, MatchValue, GuardIR) :-
    plawk_pattern_guard_ir(always, GlobalBase, MatchValue, GuardIR).
plawk_pattern_guard_ir(prefix(Prefix), _FieldSeparator, GlobalBase, MatchValue, GuardIR) :-
    plawk_pattern_guard_ir(prefix(Prefix), GlobalBase, MatchValue, GuardIR).
plawk_pattern_guard_ir(contains(Needle), FieldSeparator, GlobalBase, MatchValue, GuardIR) :-
    plawk_literal_contains_guard_ir(GlobalBase, Needle, FieldSeparator, MatchValue,
        GuardIR).
plawk_pattern_guard_ir(field_eq(Index, Value), binfmt(Types), GlobalBase, MatchValue, GuardIR) :-
    !,
    plawk_binfmt_field_eq_guard_ir(binfmt(Types), Index, Value, GlobalBase,
        MatchValue, GuardIR).
plawk_pattern_guard_ir(field_eq(Index, Value), FieldSeparator, GlobalBase, MatchValue, GuardIR) :-
    llvm_emit_atom_field_eq_guard(GlobalBase, '%line', Index, Value, FieldSeparator,
        MatchValue, GuardIR).
plawk_pattern_guard_ir(field_cmp(Index, Op, Value), binfmt(Types), GlobalBase, MatchValue, GuardIR) :-
    !,
    plawk_binfmt_field_cmp_guard_ir(binfmt(Types), field_cmp(Index, Op, Value),
        GlobalBase, MatchValue, GuardIR).
plawk_pattern_guard_ir(field_cmp(Index, Op, Value), FieldSeparator, _GlobalBase, MatchValue, ''-GuardCallIR) :-
    plawk_field_cmp_op_code(Op, OpCode),
    llvm_emit_atom_field_i64_cmp_guard('%line', Index, OpCode, Value,
        FieldSeparator, MatchValue, GuardCallIR).
% blob(dyncall...) == "literal" -- equality between a runtime grammar's
% byte output and a string literal: length check + memcmp. A failed
% call (null slice) never matches; the memcmp pointer is substituted
% with the literal itself under null so the comparison stays defined
% (its result is masked by the null flag).
plawk_pattern_guard_ir(blob_eq(Blob, Value), FieldSeparator, GlobalBase,
        MatchValue, GuardIR) :-
    format(atom(Base), '~w_beq', [GlobalBase]),
    plawk_blob_expr_ir(Blob, FieldSeparator, Base, _LenIR, _PtrIR,
        BlobGlobals, SetupParts),
    format(atom(LitName), '~w_lit', [Base]),
    llvm_emit_c_string_global(LitName, Value, LitGlobalIR, StrLen, BytesLen),
    format(atom(LitGep),
        'getelementptr ([~w x i8], [~w x i8]* @.~w, i32 0, i32 0)',
        [BytesLen, BytesLen, LitName]),
    format(atom(CmpIR),
'  %~w_bnull = icmp eq i8* %~w_ptr, null
  %~w_safep = select i1 %~w_bnull, i8* ~w, i8* %~w_ptr
  %~w_cmp = call i32 @memcmp(i8* %~w_safep, i8* ~w, i64 ~w)
  %~w_cmp_ok = icmp eq i32 %~w_cmp, 0
  %~w_len_ok = icmp eq i64 %~w_len64, ~w
  %~w_not_null = xor i1 %~w_bnull, true
  %~w_m0 = and i1 %~w_len_ok, %~w_cmp_ok
  ~w = and i1 %~w_m0, %~w_not_null',
        [Base, Base,
         Base, Base, LitGep, Base,
         Base, Base, LitGep, StrLen,
         Base, Base,
         Base, Base, StrLen,
         Base, Base,
         Base, Base, Base,
         MatchValue, Base, Base]),
    append(SetupParts, [CmpIR], AllLines),
    atomic_list_concat(AllLines, '\n', GuardCallIR),
    append(BlobGlobals, [LitGlobalIR], AllGlobals),
    atomic_list_concat(AllGlobals, '\n', GlobalsIR),
    GuardIR = GlobalsIR-GuardCallIR.

plawk_binfmt_field_cmp_guard_ir(Descriptor, field_cmp(Index, Op, Value),
        GlobalBase, MatchValue, ''-GuardCallIR) :-
    format(atom(Base), '~w_binf~w', [GlobalBase, Index]),
    plawk_binfmt_field_load_lines(Descriptor, Index, Base, ValueIR, LoadLines),
    plawk_binfmt_icmp_op(Op, ICmpOp),
    format(atom(CmpIR), '  ~w = icmp ~w i64 ~w, ~w',
        [MatchValue, ICmpOp, ValueIR, Value]),
    append(LoadLines, [CmpIR], Lines),
    atomic_list_concat(Lines, '\n', GuardCallIR).

%% plawk_binfmt_field_eq_guard_ir(+Descriptor, +Index, +Value, +GlobalBase,
%%     +MatchValue, -GuardIR)
%
%  String equality on a fixed-width sN field: memcmp against the literal
%  key, plus a NUL check at the key length when the key is shorter than
%  the field (a full-width key needs no terminator). Keys longer than
%  the field can never match.
plawk_binfmt_field_eq_guard_ir(Descriptor, Index, Value, GlobalBase,
        MatchValue, GuardIR) :-
    plawk_binfmt_field_type(Descriptor, Index, s(Width)),
    plawk_binfmt_field_offset(Descriptor, Index, Offset),
    format(atom(Base), '~w_binfeq~w', [GlobalBase, Index]),
    ( string_length(Value, KeyLen0), KeyLen0 > Width
    ->  % Key cannot fit in the field: statically false.
        format(atom(FalseIR), '  ~w = icmp eq i1 true, false', [MatchValue]),
        GuardIR = ''-FalseIR
    ;   format(atom(KeyGlobal), '~w_key', [Base]),
        llvm_emit_c_string_global(KeyGlobal, Value, GlobalLine, KeyLen, BytesLen),
        format(atom(GepIR),
            '  %~w_fp = getelementptr i8, i8* %rec, i64 ~w', [Base, Offset]),
        format(atom(KeyPtrIR),
            '  %~w_keyp = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0',
            [Base, BytesLen, BytesLen, KeyGlobal]),
        format(atom(CmpIR),
            '  %~w_cmp = call i32 @memcmp(i8* %~w_fp, i8* %~w_keyp, i64 ~w)',
            [Base, Base, Base, KeyLen]),
        format(atom(EqIR),
            '  %~w_eq = icmp eq i32 %~w_cmp, 0', [Base, Base]),
        ( KeyLen =:= Width
        ->  format(atom(FinalIR), '  ~w = and i1 %~w_eq, true',
                [MatchValue, Base]),
            Lines = [GepIR, KeyPtrIR, CmpIR, EqIR, FinalIR]
        ;   format(atom(NulGepIR),
                '  %~w_nulp = getelementptr i8, i8* %~w_fp, i64 ~w',
                [Base, Base, KeyLen]),
            format(atom(NulLoadIR),
                '  %~w_nul = load i8, i8* %~w_nulp, align 1', [Base, Base]),
            format(atom(NulOkIR),
                '  %~w_nul_ok = icmp eq i8 %~w_nul, 0', [Base, Base]),
            format(atom(FinalIR), '  ~w = and i1 %~w_eq, %~w_nul_ok',
                [MatchValue, Base, Base]),
            Lines = [GepIR, KeyPtrIR, CmpIR, EqIR, NulGepIR, NulLoadIR,
                     NulOkIR, FinalIR]
        ),
        atomic_list_concat(Lines, '\n', GuardCallIR),
        GuardIR = GlobalLine-GuardCallIR
    ).
plawk_pattern_guard_ir(field_match(Index, Regex), FieldSeparator, GlobalBase, MatchValue, GuardIR) :-
    llvm_emit_regex_field_match_guard(GlobalBase, '%line', Index, Regex,
        FieldSeparator, MatchValue, GuardIR).
plawk_pattern_guard_ir(prolog_guard(Name, Args), FieldSeparator, GlobalBase, MatchValue, GuardIR) :-
    plawk_foreign_guard_call_ir(Name, Args, FieldSeparator, GlobalBase,
        MatchValue, GuardIR).
plawk_pattern_guard_ir(and_pat(Left, Right), FieldSeparator, GlobalBase, MatchValue, GuardIR) :-
    plawk_binary_pattern_guard_ir(and, Left, Right, FieldSeparator, GlobalBase,
        MatchValue, GuardIR).
plawk_pattern_guard_ir(or_pat(Left, Right), FieldSeparator, GlobalBase, MatchValue, GuardIR) :-
    plawk_binary_pattern_guard_ir(or, Left, Right, FieldSeparator, GlobalBase,
        MatchValue, GuardIR).
plawk_pattern_guard_ir(not_pat(Pattern), FieldSeparator, GlobalBase, MatchValue, GlobalIR-GuardCallIR) :-
    format(atom(InnerBase), '~w_n', [GlobalBase]),
    format(atom(InnerValue), '~w_n', [MatchValue]),
    plawk_pattern_guard_ir(Pattern, FieldSeparator, InnerBase, InnerValue,
        GlobalIR-InnerCallIR),
    format(atom(GuardCallIR),
'~w
  ~w = xor i1 ~w, true',
        [InnerCallIR, MatchValue, InnerValue]).
% A range pattern `/start/,/end/`: the rule fires for records from a /start/
% match through a /end/ match (inclusive), tracked by a per-rule i1 flag global
% (init false = inactive). Latch, per record:
%   fire        = was_active OR start_matches
%   next_active = was_active ? NOT end_matches : start_matches
% i.e. the end is only tested once already active (so a line matching both start
% and end starts a range that continues -- the common gawk semantics). MatchValue
% is the fire result; the two endpoints reuse the ordinary pattern guards.
plawk_pattern_guard_ir(range(Start, End), FieldSeparator, GlobalBase, MatchValue,
        GlobalIR-GuardCallIR) :-
    format(atom(StartBase), '~w_rs', [GlobalBase]),
    format(atom(StartValue), '~w_rs', [MatchValue]),
    plawk_pattern_guard_ir(Start, FieldSeparator, StartBase, StartValue,
        StartGlobalIR-StartCallIR),
    format(atom(EndBase), '~w_re', [GlobalBase]),
    format(atom(EndValue), '~w_re', [MatchValue]),
    plawk_pattern_guard_ir(End, FieldSeparator, EndBase, EndValue,
        EndGlobalIR-EndCallIR),
    format(atom(FlagName), '~w_range', [GlobalBase]),
    format(atom(FlagGlobal), '@~w = internal global i1 false', [FlagName]),
    plawk_join_nonempty_ir([StartGlobalIR, EndGlobalIR, FlagGlobal], GlobalIR),
    format(atom(GuardCallIR),
'  %~w_was = load i1, i1* @~w
~w
~w
  %~w_notend = xor i1 ~w, true
  %~w_next = select i1 %~w_was, i1 %~w_notend, i1 ~w
  store i1 %~w_next, i1* @~w
  ~w = or i1 %~w_was, ~w',
        [GlobalBase, FlagName,
         StartCallIR,
         EndCallIR,
         GlobalBase, EndValue,
         GlobalBase, GlobalBase, GlobalBase, StartValue,
         GlobalBase, FlagName,
         MatchValue, GlobalBase, StartValue]).

plawk_combined_pattern(and_pat(_Left, _Right)).
plawk_combined_pattern(or_pat(_Left, _Right)).
plawk_combined_pattern(not_pat(_Pattern)).

%% plawk_binary_pattern_guard_ir(+Op, +Left, +Right, +FieldSeparator,
%%     +GlobalBase, +MatchValue, -GuardIR)
%
%  Combine two pattern guards with a bitwise i1 op. The base guards are
%  side-effect-free straight-line checks, so evaluating both operands
%  keeps the combined guard a single block; awk's short-circuit order
%  is unobservable here.
plawk_binary_pattern_guard_ir(Op, Left, Right, FieldSeparator, GlobalBase,
        MatchValue, GlobalIR-GuardCallIR) :-
    format(atom(LeftBase), '~w_l', [GlobalBase]),
    format(atom(LeftValue), '~w_l', [MatchValue]),
    plawk_pattern_guard_ir(Left, FieldSeparator, LeftBase, LeftValue,
        LeftGlobalIR-LeftCallIR),
    format(atom(RightBase), '~w_r', [GlobalBase]),
    format(atom(RightValue), '~w_r', [MatchValue]),
    plawk_pattern_guard_ir(Right, FieldSeparator, RightBase, RightValue,
        RightGlobalIR-RightCallIR),
    plawk_join_nonempty_ir([LeftGlobalIR, RightGlobalIR], GlobalIR),
    format(atom(GuardCallIR),
'~w
~w
  ~w = ~w i1 ~w, ~w',
        [LeftCallIR, RightCallIR, MatchValue, Op, LeftValue, RightValue]).

%% plawk_foreign_args_ir(+Args, +FieldSeparator, +BasePrefix, -ArgValueIRs,
%%     -GlobalParts, -SetupParts)
%
%  Marshal plawk foreign-call arguments into %Value SSA names. field(0)
%  passes the record atom %line directly; positive fields intern the
%  projected slice (missing fields intern the empty atom); string
%  literals intern per-site globals; integers build integer values.
plawk_foreign_args_ir(Args, FieldSeparator, BasePrefix, ArgValueIRs,
        GlobalParts, SetupParts) :-
    plawk_foreign_args_ir(Args, FieldSeparator, BasePrefix, 0, ArgValueIRs,
        GlobalPartsNested, SetupPartsNested),
    append(GlobalPartsNested, GlobalParts),
    append(SetupPartsNested, SetupParts).

plawk_foreign_args_ir([], _FieldSeparator, _BasePrefix, _Index, [], [], []).
plawk_foreign_args_ir([Arg | Rest], FieldSeparator, BasePrefix, Index,
        [ArgValueIR | ArgValueIRs], [GlobalParts | GlobalPartsRest],
        [SetupParts | SetupPartsRest]) :-
    format(atom(ArgBase), '~w_a~w', [BasePrefix, Index]),
    plawk_foreign_arg_ir(Arg, FieldSeparator, ArgBase, ArgValueIR,
        GlobalParts, SetupParts),
    NextIndex is Index + 1,
    plawk_foreign_args_ir(Rest, FieldSeparator, BasePrefix, NextIndex,
        ArgValueIRs, GlobalPartsRest, SetupPartsRest).

% Binary mode: i64 fields marshal as WAM integers (a typed load, no
% text anywhere); blob fields copy their payload into the shared
% transient buffer and marshal as the transient atom -- constant
% memory, no interning, readable Prolog-side via atom_codes/2.
plawk_foreign_arg_ir(field(FieldIndex), binfmt(Types), ArgBase, ArgValueIR,
        [], SetupParts) :-
    integer(FieldIndex),
    FieldIndex >= 1,
    plawk_binfmt_field_type(binfmt(Types), FieldIndex, i64),
    !,
    format(atom(LoadBase), '~w_bf', [ArgBase]),
    plawk_binfmt_field_load_lines(binfmt(Types), FieldIndex, LoadBase,
        LoadedIR, LoadLines),
    format(atom(ValueLine),
        '  %~w_v = call %Value @value_integer(i64 ~w)', [ArgBase, LoadedIR]),
    format(atom(ArgValueIR), '%~w_v', [ArgBase]),
    append(LoadLines, [ValueLine], SetupParts).
% f64 fields marshal as WAM floats: a typed double load, then
% @value_float packs the bits under the Float tag.
plawk_foreign_arg_ir(field(FieldIndex), binfmt(Types), ArgBase, ArgValueIR,
        [], SetupParts) :-
    integer(FieldIndex),
    FieldIndex >= 1,
    plawk_binfmt_field_type(binfmt(Types), FieldIndex, f64),
    !,
    format(atom(LoadBase), '~w_bf', [ArgBase]),
    plawk_binfmt_field_load_lines(binfmt(Types), FieldIndex, LoadBase,
        LoadedIR, LoadLines),
    format(atom(ValueLine),
        '  %~w_v = call %Value @value_float(double ~w)', [ArgBase, LoadedIR]),
    format(atom(ArgValueIR), '%~w_v', [ArgBase]),
    append(LoadLines, [ValueLine], SetupParts).
plawk_foreign_arg_ir(field(FieldIndex), binfmt(Types), ArgBase, ArgValueIR,
        [], SetupParts) :-
    integer(FieldIndex),
    FieldIndex >= 1,
    plawk_binfmt_field_type(binfmt(Types), FieldIndex, blob(_Cap)),
    !,
    plawk_binfmt_field_offset(binfmt(Types), FieldIndex, Offset),
    PayloadOffset is Offset + 8,
    format(atom(SetupIR),
'  %~w_lp = getelementptr i8, i8* %rec, i64 ~w
  %~w_ltp = bitcast i8* %~w_lp to i64*
  %~w_len = load i64, i64* %~w_ltp
  %~w_ptr = getelementptr i8, i8* %rec, i64 ~w
  %~w_id = call i64 @wam_transient_atom_from_bytes(i8* %~w_ptr, i64 %~w_len)
  %~w_v0 = insertvalue %Value undef, i32 0, 0
  %~w_v = insertvalue %Value %~w_v0, i64 %~w_id, 1',
        [ArgBase, Offset,
         ArgBase, ArgBase,
         ArgBase, ArgBase,
         ArgBase, PayloadOffset,
         ArgBase, ArgBase, ArgBase,
         ArgBase,
         ArgBase, ArgBase, ArgBase]),
    format(atom(ArgValueIR), '%~w_v', [ArgBase]),
    SetupParts = [SetupIR].
plawk_foreign_arg_ir(field(0), _FieldSeparator, ArgBase, ArgValueIR,
        [], SetupParts) :-
    !,
    % The record Value is the transient line atom whose buffer mutates
    % on the next read; Prolog-side atom identity (X == 'ERROR') and
    % anything the predicate might persist need a real atom, so $0
    % interns the current line text. %line_s is the C string the
    % driver's EOF check already resolved.
    SafeBase = ArgBase,
    format(atom(LenIR),
        '  %~w_len = call i64 @strlen(i8* %line_s)', [SafeBase]),
    format(atom(InternIR),
        '  %~w_id = call i64 @wam_intern_atom(i8* %line_s, i64 %~w_len)',
        [SafeBase, SafeBase]),
    format(atom(Value0IR),
        '  %~w_v0 = insertvalue %Value undef, i32 0, 0', [SafeBase]),
    format(atom(ValueIR),
        '  %~w_v = insertvalue %Value %~w_v0, i64 %~w_id, 1',
        [SafeBase, SafeBase, SafeBase]),
    format(atom(ArgValueIR), '%~w_v', [SafeBase]),
    SetupParts = [LenIR, InternIR, Value0IR, ValueIR].
plawk_foreign_arg_ir(field(FieldIndex), FieldSeparator, ArgBase, ArgValueIR,
        [EmptyGlobalIR], SetupParts) :-
    integer(FieldIndex),
    FieldIndex > 0,
    !,
    SafeBase = ArgBase,
    format(atom(EmptyGlobalIR),
        '@.~w_empty = private constant [1 x i8] zeroinitializer', [SafeBase]),
    format(atom(SliceIR),
        '  %~w_slice = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 ~w, i8 ~w)',
        [SafeBase, FieldIndex, FieldSeparator]),
    format(atom(PtrIR),
        '  %~w_ptr = extractvalue %WamSlice %~w_slice, 0', [SafeBase, SafeBase]),
    format(atom(LenIR),
        '  %~w_len = extractvalue %WamSlice %~w_slice, 1', [SafeBase, SafeBase]),
    format(atom(NullIR),
        '  %~w_null = icmp eq i8* %~w_ptr, null', [SafeBase, SafeBase]),
    format(atom(SafePtrIR),
        '  %~w_safe_ptr = select i1 %~w_null, i8* getelementptr ([1 x i8], [1 x i8]* @.~w_empty, i32 0, i32 0), i8* %~w_ptr',
        [SafeBase, SafeBase, SafeBase, SafeBase]),
    format(atom(SafeLenIR),
        '  %~w_safe_len = select i1 %~w_null, i64 0, i64 %~w_len',
        [SafeBase, SafeBase, SafeBase]),
    format(atom(InternIR),
        '  %~w_id = call i64 @wam_intern_atom(i8* %~w_safe_ptr, i64 %~w_safe_len)',
        [SafeBase, SafeBase, SafeBase]),
    format(atom(Value0IR),
        '  %~w_v0 = insertvalue %Value undef, i32 0, 0', [SafeBase]),
    format(atom(ValueIR),
        '  %~w_v = insertvalue %Value %~w_v0, i64 %~w_id, 1',
        [SafeBase, SafeBase, SafeBase]),
    format(atom(ArgValueIR), '%~w_v', [SafeBase]),
    SetupParts = [SliceIR, PtrIR, LenIR, NullIR, SafePtrIR, SafeLenIR,
        InternIR, Value0IR, ValueIR].
plawk_foreign_arg_ir(string(String), _FieldSeparator, ArgBase, ArgValueIR,
        [StringGlobalIR], SetupParts) :-
    !,
    SafeBase = ArgBase,
    format(atom(StringGlobalName), '~w_str', [SafeBase]),
    llvm_emit_c_string_global(StringGlobalName, String, StringGlobalIR,
        StringLen, BytesLen),
    format(atom(PtrIR),
        '  %~w_str_ptr = getelementptr [~w x i8], [~w x i8]* @.~w_str, i32 0, i32 0',
        [SafeBase, BytesLen, BytesLen, SafeBase]),
    format(atom(InternIR),
        '  %~w_id = call i64 @wam_intern_atom(i8* %~w_str_ptr, i64 ~w)',
        [SafeBase, SafeBase, StringLen]),
    format(atom(Value0IR),
        '  %~w_v0 = insertvalue %Value undef, i32 0, 0', [SafeBase]),
    format(atom(ValueIR),
        '  %~w_v = insertvalue %Value %~w_v0, i64 %~w_id, 1',
        [SafeBase, SafeBase, SafeBase]),
    format(atom(ArgValueIR), '%~w_v', [SafeBase]),
    SetupParts = [PtrIR, InternIR, Value0IR, ValueIR].
plawk_foreign_arg_ir(int(Value), _FieldSeparator, ArgBase, ArgValueIR,
        [], [IntIR]) :-
    integer(Value),
    SafeBase = ArgBase,
    format(atom(IntIR),
        '  %~w_v = call %Value @value_integer(i64 ~w)', [SafeBase, Value]),
    format(atom(ArgValueIR), '%~w_v', [SafeBase]).

plawk_foreign_call_args_ir([], '').
plawk_foreign_call_args_ir(ArgValueIRs, IR) :-
    ArgValueIRs = [_ | _],
    findall(Part,
        ( member(ArgValueIR, ArgValueIRs),
          format(atom(Part), '%Value ~w', [ArgValueIR])
        ),
        Parts),
    atomic_list_concat(Parts, ', ', IR).

plawk_foreign_guard_call_ir(Name, Args, FieldSeparator, GlobalBase, MatchValue,
        GlobalIR-GuardCallIR) :-
    length(Args, NArgs),
    plawk_foreign_args_ir(Args, FieldSeparator, GlobalBase, ArgValueIRs,
        GlobalParts, SetupParts),
    plawk_join_nonempty_ir(GlobalParts, GlobalIR),
    plawk_foreign_call_args_ir(ArgValueIRs, CallArgsIR),
    format(atom(CallLine),
        '  ~w = call i1 @plawk_foreign_guard_~w_~w(~w)',
        [MatchValue, Name, NArgs, CallArgsIR]),
    append(SetupParts, [CallLine], Lines),
    atomic_list_concat(Lines, '\n', GuardCallIR).

plawk_literal_contains_guard_ir(GlobalBase, Needle, FieldSeparator, MatchValue, GlobalIR-GuardCallIR) :-
    format(atom(IndexBase), '~w_contains_index', [GlobalBase]),
    llvm_emit_atom_field_index(GlobalBase, '%line', 0, Needle, FieldSeparator,
        IndexBase, GlobalIR-IndexCallIR),
    format(atom(GuardCallIR),
'~w
  ~w = icmp sgt i64 %~w, 0',
        [IndexCallIR, MatchValue, IndexBase]).

plawk_field_cmp_op_code(eq, 0).
plawk_field_cmp_op_code(ne, 1).
plawk_field_cmp_op_code(lt, 2).
plawk_field_cmp_op_code(le, 3).
plawk_field_cmp_op_code(gt, 4).
plawk_field_cmp_op_code(ge, 5).

plawk_print_record_counter_ir(Fields, LoopPhiIR, RecordCounterIR) :-
    (   plawk_fields_include_nr(Fields)
    ->  LoopPhiIR = '  %plawk_nr = phi i64 [0, %check_handle_value], [%current_nr, %continue_loop]',
        RecordCounterIR = '  %current_nr = add i64 %plawk_nr, 1'
    ;   LoopPhiIR = '',
        RecordCounterIR = ''
    ).

plawk_fields_include_nr(Fields) :-
    member(Field, Fields),
    plawk_expr_uses_nr(Field).

plawk_expr_uses_nr(special('NR')).
plawk_expr_uses_nr(concat(Parts)) :-
    member(Part, Parts),
    plawk_expr_uses_nr(Part).
plawk_expr_uses_nr(ternary(cmp(Left, _Op, Right), Then, Else)) :-
    ( plawk_expr_uses_nr(Left)
    ; plawk_expr_uses_nr(Right)
    ; plawk_expr_uses_nr(Then)
    ; plawk_expr_uses_nr(Else)
    ).
plawk_expr_uses_nr(sprintf(_Format, Args)) :-
    member(Arg, Args),
    plawk_expr_uses_nr(Arg).
plawk_expr_uses_nr(Expr) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, _NamePart, Left, Right),
    ( plawk_expr_uses_nr(Left)
    ; plawk_expr_uses_nr(Right)
    ).

plawk_print_action_ir([field(0)], _FieldSeparator, _OutputSeparator, ''-IR) :-
    !,
    llvm_emit_printf_string(plawk_surface_print_line, 4, fmt, printed, '%line_s',
        [FmtPtr, PrintCall]),
    atomic_list_concat([FmtPtr, PrintCall], '\n', IR).
plawk_print_action_ir(Fields, FieldSeparator, OutputSeparator, GlobalIR-IR) :-
    phrase(plawk_print_fields_ir(Fields, FieldSeparator, OutputSeparator, 0), Pairs),
    plawk_print_ir_parts(Pairs, GlobalParts, BodyParts),
    plawk_join_nonempty_ir(GlobalParts, GlobalIR),
    atomic_list_concat(BodyParts, '\n', IR).

plawk_output_action_exprs(print(Fields), Fields).
plawk_output_action_exprs(printf(string(_Format), Args), Args).
plawk_output_action_exprs(writebin_out(_Types, Fields), Fields).
plawk_output_action_exprs(writebin_arm_out(_Tag, _ArmTypes, Fields), Fields).

plawk_output_action_ir(print(Fields), FieldSeparator, OutputSeparator, Pair) :-
    plawk_print_action_ir(Fields, FieldSeparator, OutputSeparator, Pair).
plawk_output_action_ir(printf(string(Format), Args), FieldSeparator, _OutputSeparator, Pair) :-
    plawk_prefixed_printf_action_ir(Format, Args, FieldSeparator, plawk_printf, Pair).
plawk_output_action_ir(writebin_out(Types, Fields), FieldSeparator, _OutputSeparator, Pair) :-
    % No scalar slots exist in the single-action driver, so scalar reads
    % in writebin arguments fail here and the program falls through to
    % the state-plan drivers.
    plawk_writebin_record_ir(Types, Fields, [], [], FieldSeparator,
        plawk_writebin, Pair).
plawk_output_action_ir(writebin_arm_out(Tag, ArmTypes, Fields), FieldSeparator,
        _OutputSeparator, Pair) :-
    plawk_writebin_union_record_ir(Tag, ArmTypes, Fields, [], [],
        FieldSeparator, plawk_writebin, Pair).

plawk_prefixed_print_action_ir([field(0)], _FieldSeparator, _OutputSeparator, Prefix, ''-IR) :-
    !,
    format(atom(FmtVar), '~w_line_fmt', [Prefix]),
    format(atom(PrintVar), '~w_printed_line', [Prefix]),
    llvm_emit_printf_string(plawk_surface_print_line, 4, FmtVar, PrintVar, '%line_s',
        [FmtPtr, PrintCall]),
    atomic_list_concat([FmtPtr, PrintCall], '\n', IR).
plawk_prefixed_print_action_ir(Fields, FieldSeparator, OutputSeparator, Prefix, GlobalIR-IR) :-
    phrase(plawk_prefixed_print_fields_ir(Fields, FieldSeparator, OutputSeparator, Prefix, 0), Pairs),
    plawk_print_ir_parts(Pairs, GlobalParts, BodyParts),
    plawk_join_nonempty_ir(GlobalParts, GlobalIR),
    atomic_list_concat(BodyParts, '\n', IR).

plawk_prefixed_printf_action_ir(Format, Args, FieldSeparator, Prefix, GlobalIR-IR) :-
    phrase(plawk_printf_arg_pairs(Args, FieldSeparator, Prefix, 0), ArgPairs),
    pairs_keys_values(ArgPairs, ArgGlobalParts, ArgInfoPairs),
    pairs_keys_values(ArgInfoPairs, ArgSetupParts, ArgCallArgLists),
    append(ArgCallArgLists, CallArgs),
    maplist(plawk_printf_call_arg_kind, CallArgs, ArgKinds),
    plawk_printf_rewrite_format(Format, ArgKinds, PrintfFormat),
    format(atom(FormatGlobal), '~w_fmt', [Prefix]),
    llvm_emit_c_string_global(FormatGlobal, PrintfFormat, FormatGlobalIR, _FormatLen, FormatBytesLen),
    format(atom(FmtPtrVar), '~w_fmt_ptr', [Prefix]),
    format(atom(FmtPtr),
        '  %~w = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0',
        [FmtPtrVar, FormatBytesLen, FormatBytesLen, FormatGlobal]),
    plawk_printf_call_args_ir(CallArgs, CallArgsIR),
    format(atom(PrintVar), '~w_printed', [Prefix]),
    (   CallArgsIR == ''
    ->  format(atom(PrintCall),
            '  %~w = call i32 (i8*, ...) @printf(i8* %~w)',
            [PrintVar, FmtPtrVar])
    ;   format(atom(PrintCall),
            '  %~w = call i32 (i8*, ...) @printf(i8* %~w, ~w)',
            [PrintVar, FmtPtrVar, CallArgsIR])
    ),
    plawk_join_nonempty_ir([FormatGlobalIR | ArgGlobalParts], GlobalIR),
    append(ArgSetupParts, [FmtPtr, PrintCall], BodyParts),
    atomic_list_concat(BodyParts, '\n', IR).

plawk_printf_arg_pairs([], _FieldSeparator, _Prefix, _Index) -->
    [].
plawk_printf_arg_pairs([Arg | Args], FieldSeparator, Prefix, Index) -->
    { plawk_emit_prefixed_print_expr_ir(Arg, FieldSeparator, Prefix, Index,
          Type, GlobalParts, SetupParts),
      plawk_printf_type_call_args(Type, CallArgs),
      plawk_join_nonempty_ir(GlobalParts, GlobalIR),
      plawk_join_nonempty_ir(SetupParts, SetupIR),
      NextIndex is Index + 1
    },
    [GlobalIR-(SetupIR-CallArgs)],
    plawk_printf_arg_pairs(Args, FieldSeparator, Prefix, NextIndex).

plawk_printf_type_call_args(i64(_FmtPrefix, _PrintPrefix, ValueIR), [i64(ValueIR)]).
plawk_printf_type_call_args(slice(_FmtPrefix, _PrintPrefix, LenIR, PtrIR), [slice_len(LenIR), slice_ptr(PtrIR)]).
plawk_printf_type_call_args(string(_Base, PtrIR), [string_ptr(PtrIR)]).
plawk_printf_type_call_args(f64(_FmtPrefix, _PrintPrefix, ValueIR), [f64(ValueIR)]).

plawk_printf_call_arg_kind(i64(_ValueIR), i64).
plawk_printf_call_arg_kind(slice_len(_LenIR), slice_len).
plawk_printf_call_arg_kind(slice_ptr(_PtrIR), slice_ptr).
plawk_printf_call_arg_kind(string_ptr(_PtrIR), string).
plawk_printf_call_arg_kind(f64(_ValueIR), f64).

plawk_printf_call_args_ir([], '') :-
    !.
plawk_printf_call_args_ir(CallArgs, IR) :-
    maplist(plawk_printf_call_arg_ir, CallArgs, Parts),
    atomic_list_concat(Parts, ', ', IR).

plawk_printf_call_arg_ir(i64(ValueIR), IR) :-
    format(atom(IR), 'i64 ~w', [ValueIR]).
plawk_printf_call_arg_ir(slice_len(LenIR), IR) :-
    format(atom(IR), 'i32 ~w', [LenIR]).
plawk_printf_call_arg_ir(slice_ptr(PtrIR), IR) :-
    format(atom(IR), 'i8* ~w', [PtrIR]).
plawk_printf_call_arg_ir(string_ptr(PtrIR), IR) :-
    format(atom(IR), 'i8* ~w', [PtrIR]).
plawk_printf_call_arg_ir(f64(ValueIR), IR) :-
    format(atom(IR), 'double ~w', [ValueIR]).

plawk_printf_rewrite_format(Format, ArgKinds, RewrittenFormat) :-
    string_codes(Format, Codes),
    plawk_printf_rewrite_codes(Codes, ArgKinds, RewrittenCodes),
    string_codes(RewrittenFormat, RewrittenCodes).

plawk_printf_rewrite_codes([], [], []).
plawk_printf_rewrite_codes([Code | Rest], Kinds, [Code | RewrittenRest]) :-
    Code =\= 0'%,
    !,
    plawk_printf_rewrite_codes(Rest, Kinds, RewrittenRest).
plawk_printf_rewrite_codes([0'%, 0'% | Rest], Kinds, [0'%, 0'% | RewrittenRest]) :-
    !,
    plawk_printf_rewrite_codes(Rest, Kinds, RewrittenRest).
% A conversion `%[flags][width][.precision][length]<conv>`: parse the standard
% prefix, then rewrite the body per the next argument's inferred kind (the arg
% kind drives the choice, so a mismatched conv -- e.g. %s for an integer arg --
% cleanly fails the rewrite -> compile error rather than miscompiling).
plawk_printf_rewrite_codes([0'% | Rest], Kinds0, Rewritten) :-
    plawk_printf_scan_spec(Rest, Flags, Width, Prec, Conv, RestAfter),
    plawk_printf_kind_spec(Kinds0, Flags, Width, Prec, Conv, Kinds, SpecBody),
    !,
    append([0'% | SpecBody], RewrittenRest, Rewritten),
    plawk_printf_rewrite_codes(RestAfter, Kinds, RewrittenRest).

% Scan the standard printf conversion prefix: flags in `-+ 0#`, decimal width,
% optional `.precision`, an optional (ignored) length modifier, then the
% conversion character.
plawk_printf_scan_spec(Codes, Flags, Width, Prec, Conv, Rest) :-
    plawk_printf_take_flags(Codes, Flags, C1),
    plawk_printf_take_digits(C1, Width, C2),
    plawk_printf_take_prec(C2, Prec, C3),
    plawk_printf_take_lenmod(C3, C4),
    C4 = [Conv | Rest].

plawk_printf_take_flags([C | Cs], [C | Fs], Rest) :-
    memberchk(C, [0'-, 0'+, 0' , 0'0, 0'#]),
    !,
    plawk_printf_take_flags(Cs, Fs, Rest).
plawk_printf_take_flags(Cs, [], Cs).

plawk_printf_take_digits([C | Cs], [C | Ds], Rest) :-
    code_type(C, digit),
    !,
    plawk_printf_take_digits(Cs, Ds, Rest).
plawk_printf_take_digits(Cs, [], Cs).

% Precision keeps its leading `.` (a bare `.` is precision 0, as in C).
plawk_printf_take_prec([0'. | Cs], [0'. | Ds], Rest) :-
    !,
    plawk_printf_take_digits(Cs, Ds, Rest).
plawk_printf_take_prec(Cs, [], Cs).

% A user-written length modifier is consumed and dropped; the emitter supplies
% the correct one for the argument's representation (i64 -> `l`).
plawk_printf_take_lenmod([0'l, 0'l | Cs], Cs) :- !.
plawk_printf_take_lenmod([0'l | Cs], Cs) :- !.
plawk_printf_take_lenmod([0'h, 0'h | Cs], Cs) :- !.
plawk_printf_take_lenmod([0'h | Cs], Cs) :- !.
plawk_printf_take_lenmod(Cs, Cs).

% Map (arg kind, parsed prefix, conversion) to the C conversion body (the codes
% after the leading `%`), consuming the kind(s) the argument contributed.
% Integer arg (i64): d/i/x/X/o/u get the `l` length modifier; c takes the code
% point as a character (no length modifier, precision dropped).
plawk_printf_kind_spec([i64 | Ks], Flags, Width, _Prec, 0'c, Ks, Body) :-
    !,
    append([Flags, Width, [0'c]], Body).
plawk_printf_kind_spec([i64 | Ks], Flags, Width, Prec, Conv, Ks, Body) :-
    plawk_printf_int_conversion_code(Conv, OutConv),
    append([Flags, Width, Prec, [0'l, OutConv]], Body).
% Float arg (f64): f/g/e/F/G/E pass through with flags/width/precision.
plawk_printf_kind_spec([f64 | Ks], Flags, Width, Prec, Conv, Ks, Body) :-
    plawk_printf_f64_conversion_code(Conv),
    append([Flags, Width, Prec, [Conv]], Body).
% Null-terminated string arg (%s): flags/width/precision pass through.
plawk_printf_kind_spec([string | Ks], Flags, Width, Prec, 0's, Ks, Body) :-
    append([Flags, Width, Prec, [0's]], Body).
% Record-field slice (%s over len+ptr): the length rides `.*`, so the slice
% bounds the output; width is kept. A user precision can't be honoured -- the
% slice pointer is not null-terminated, so `.N` (without `.*`) would read past
% the field -- so `%.Ns` on a field is a clean compile error (a follow-on).
plawk_printf_kind_spec([slice_len, slice_ptr | Ks], Flags, Width, [], 0's, Ks, Body) :-
    append([Flags, Width, [0'., 0'*, 0's]], Body).

% Integer conversions; `i` normalises to `d` (identical output).
plawk_printf_int_conversion_code(0'd, 0'd).
plawk_printf_int_conversion_code(0'i, 0'd).
plawk_printf_int_conversion_code(0'x, 0'x).
plawk_printf_int_conversion_code(0'X, 0'X).
plawk_printf_int_conversion_code(0'o, 0'o).
plawk_printf_int_conversion_code(0'u, 0'u).

plawk_printf_f64_conversion_code(0'f).
plawk_printf_f64_conversion_code(0'g).
plawk_printf_f64_conversion_code(0'e).
plawk_printf_f64_conversion_code(0'F).
plawk_printf_f64_conversion_code(0'G).
plawk_printf_f64_conversion_code(0'E).

plawk_print_ir_parts([], [], []).
plawk_print_ir_parts([GlobalIR-BodyIR | Parts], [GlobalIR | GlobalParts], [BodyIR | BodyParts]) :-
    plawk_print_ir_parts(Parts, GlobalParts, BodyParts).

plawk_join_nonempty_ir(Parts, IR) :-
    include(plawk_nonempty_ir, Parts, NonEmptyParts),
    atomic_list_concat(NonEmptyParts, '\n', IR).

plawk_nonempty_ir(IR) :-
    IR \== ''.

plawk_print_fields_ir([], _FieldSeparator, _OutputSeparator, _) -->
    { llvm_emit_printf0(plawk_surface_print_newline, 2,
          newline_fmt, printed_newline, [FmtPtr, PrintCall])
    },
    [''-FmtPtr, ''-PrintCall].
plawk_print_fields_ir([Field | Rest], FieldSeparator, OutputSeparator, Index) -->
    plawk_print_separator_ir(Index, OutputSeparator),
    plawk_print_field_ir(Field, FieldSeparator, Index),
    { NextIndex is Index + 1 },
    plawk_print_fields_ir(Rest, FieldSeparator, OutputSeparator, NextIndex).

plawk_prefixed_print_fields_ir([], _FieldSeparator, _OutputSeparator, Prefix, _) -->
    { format(atom(FmtVar), '~w_newline_fmt', [Prefix]),
      format(atom(PrintVar), '~w_printed_newline', [Prefix]),
      llvm_emit_printf0(plawk_surface_print_newline, 2, FmtVar, PrintVar,
          [FmtPtr, PrintCall])
    },
    [''-FmtPtr, ''-PrintCall].
plawk_prefixed_print_fields_ir([Field | Rest], FieldSeparator, OutputSeparator, Prefix, Index) -->
    plawk_prefixed_print_separator_ir(Index, OutputSeparator, Prefix),
    plawk_prefixed_print_field_ir(Field, FieldSeparator, Prefix, Index),
    { NextIndex is Index + 1 },
    plawk_prefixed_print_fields_ir(Rest, FieldSeparator, OutputSeparator, Prefix, NextIndex).

plawk_print_separator_ir(0, _OutputSeparator) -->
    !,
    [].
plawk_print_separator_ir(Index, OutputSeparator) -->
    { format(atom(SpaceCall),
          '  %printed_separator_~w = call i32 @putchar(i32 ~w)',
          [Index, OutputSeparator])
    },
    [''-SpaceCall].

plawk_prefixed_print_separator_ir(0, _OutputSeparator, _Prefix) -->
    !,
    [].
plawk_prefixed_print_separator_ir(Index, OutputSeparator, Prefix) -->
    { format(atom(SpaceCall),
          '  %~w_printed_separator_~w = call i32 @putchar(i32 ~w)',
          [Prefix, Index, OutputSeparator])
    },
    [''-SpaceCall].

plawk_print_field_ir(Field, FieldSeparator, Index) -->
    { plawk_emit_print_expr_ir(Field, FieldSeparator, Index, Type, GlobalParts, SetupParts),
      plawk_print_expr_output_ir(Type, Index, PrintParts),
      plawk_join_nonempty_ir(GlobalParts, GlobalIR),
      append(SetupParts, PrintParts, BodyParts),
      atomic_list_concat(BodyParts, '\n', BodyIR)
    },
    [GlobalIR-BodyIR].

% A concatenation prints its operands adjacently -- NO field separator between
% them (that is what distinguishes `print $1 $2` from `print $1, $2`). Each part
% lowers via the ordinary print-field emitter under a unique sub-prefix.
plawk_prefixed_print_field_ir(concat(Parts), FieldSeparator, Prefix, Index) -->
    { format(atom(ConcatPrefix), '~w_concat_~w', [Prefix, Index]) },
    plawk_concat_parts_ir(Parts, FieldSeparator, ConcatPrefix, 0).
plawk_prefixed_print_field_ir(Field, FieldSeparator, Prefix, Index) -->
    { plawk_emit_prefixed_print_expr_ir(Field, FieldSeparator, Prefix, Index,
          Type, GlobalParts, SetupParts),
      plawk_prefixed_print_expr_output_ir(Type, Prefix, Index, PrintParts),
      plawk_join_nonempty_ir(GlobalParts, GlobalIR),
      append(SetupParts, PrintParts, BodyParts),
      atomic_list_concat(BodyParts, '\n', BodyIR)
    },
    [GlobalIR-BodyIR].

plawk_concat_parts_ir([], _FieldSeparator, _Prefix, _PartIndex) -->
    [].
plawk_concat_parts_ir([Part | Rest], FieldSeparator, Prefix, PartIndex) -->
    plawk_prefixed_print_field_ir(Part, FieldSeparator, Prefix, PartIndex),
    { NextIndex is PartIndex + 1 },
    plawk_concat_parts_ir(Rest, FieldSeparator, Prefix, NextIndex).

plawk_emit_print_expr_ir(Field, FieldSeparator, Index, Type, GlobalParts, SetupParts) :-
    plawk_emit_print_expr_for_context(Field, FieldSeparator, print_context(normal, '', Index),
        Type, GlobalParts, SetupParts).

plawk_emit_prefixed_print_expr_ir(Field, FieldSeparator, Prefix, Index,
        Type, GlobalParts, SetupParts) :-
    plawk_emit_print_expr_for_context(Field, FieldSeparator,
        print_context(prefixed, Prefix, Index), Type, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(special('NR'), _FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_print_expr_output_names(Context, nr, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(nr, 0, nr, nr, ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(string(Value), _FieldSeparator, Context,
        string(Base, PtrIR), [GlobalIR], [StringPtr]) :-
    plawk_print_expr_value_base(Context, string, Base),
    llvm_emit_c_string_global(Base, Value, GlobalIR, _StringLen, BytesLen),
    format(atom(StringPtr),
        '  %~w_ptr = getelementptr [~w x i8], [~w x i8]* @.~w, i32 0, i32 0',
        [Base, BytesLen, BytesLen, Base]),
    format(atom(PtrIR), '%~w_ptr', [Base]).

plawk_emit_print_expr_for_context(special('NF'), FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, nf, Base),
    plawk_print_expr_output_names(Context, nf, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(nf, FieldSeparator, Base, Base, ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(int(field(FieldIndex)), FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, int, Base),
    plawk_print_expr_output_names(Context, int, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(field_i64(FieldIndex), FieldSeparator, Base, Base,
        ValueIR, GlobalParts, SetupParts).
% a substituted scalar read (var(Name) -> ssa(SlotValue)): print the i64 SSA
% value directly. This is what makes `print i` work for a scalar slot.
plawk_emit_print_expr_for_context(ssa(Value), FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, int, Base),
    plawk_print_expr_output_names(Context, int, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(ssa(Value), FieldSeparator, Base, Base,
        ValueIR, GlobalParts, SetupParts).
% a substituted STRING-scalar read (var(Name) -> ssa_str(SlotValue)): the slot
% holds an atom id; resolve it to text (id 0 is the unset sentinel, printed as
% empty). A select keeps it straight-line -- wam_atom_to_string is always called
% but its result is discarded when the id is 0.
% `print ENVIRON["NAME"]`: getenv the value then print it as text (id 0 -> empty).
plawk_emit_print_expr_for_context(environ(Key), _FieldSeparator, Context,
        string(Base, PtrIR), [KeyGlobal, EmptyGlobal],
        [KeyPtr, EnvCall, StrCall, IsEmpty, SelPtr]) :-
    plawk_print_expr_value_base(Context, string, Base),
    format(atom(KeyName), '~w_envkey', [Base]),
    llvm_emit_c_string_global(KeyName, Key, KeyGlobal, _Len, KeyBytes),
    format(atom(KeyPtr),
        '  %~w_envkeyp = getelementptr [~w x i8], [~w x i8]* @.~w, i64 0, i64 0',
        [Base, KeyBytes, KeyBytes, KeyName]),
    format(atom(EnvCall),
        '  %~w_envid = call i64 @wam_environ_get(i8* %~w_envkeyp)', [Base, Base]),
    format(atom(EmptyName), '~w_empty', [Base]),
    format(atom(EmptyGlobal),
        '@.~w = private constant [1 x i8] zeroinitializer', [EmptyName]),
    format(atom(StrCall),
        '  %~w_s = call i8* @wam_atom_to_string(i64 %~w_envid)', [Base, Base]),
    format(atom(IsEmpty),
        '  %~w_empty_c = icmp eq i64 %~w_envid, 0', [Base, Base]),
    format(atom(SelPtr),
        '  %~w_sptr = select i1 %~w_empty_c, i8* getelementptr ([1 x i8], [1 x i8]* @.~w, i64 0, i64 0), i8* %~w_s',
        [Base, Base, EmptyName, Base]),
    format(atom(PtrIR), '%~w_sptr', [Base]).
% `print ARGV[N]`: fetch the N-th argument's atom id then print it as text
% (id 0 -> empty), same id-resolve pattern as ENVIRON.
plawk_emit_print_expr_for_context(argv_at(N), _FieldSeparator, Context,
        string(Base, PtrIR), [EmptyGlobal],
        [ArgvCall, StrCall, IsEmpty, SelPtr]) :-
    integer(N), N >= 0,
    plawk_print_expr_value_base(Context, string, Base),
    format(atom(ArgvCall),
        '  %~w_argvid = call i64 @wam_argv_get(i64 ~w)', [Base, N]),
    format(atom(EmptyName), '~w_empty', [Base]),
    format(atom(EmptyGlobal),
        '@.~w = private constant [1 x i8] zeroinitializer', [EmptyName]),
    format(atom(StrCall),
        '  %~w_s = call i8* @wam_atom_to_string(i64 %~w_argvid)', [Base, Base]),
    format(atom(IsEmpty),
        '  %~w_empty_c = icmp eq i64 %~w_argvid, 0', [Base, Base]),
    format(atom(SelPtr),
        '  %~w_sptr = select i1 %~w_empty_c, i8* getelementptr ([1 x i8], [1 x i8]* @.~w, i64 0, i64 0), i8* %~w_s',
        [Base, Base, EmptyName, Base]),
    format(atom(PtrIR), '%~w_sptr', [Base]).
plawk_emit_print_expr_for_context(ssa_str(Value), FieldSeparator, Context,
        Out, Globals, Setup) :-
    plawk_emit_print_str_id(Value, FieldSeparator, Context, Out, Globals, Setup).
% a strnum read printed in a print field resolves its atom id to text, exactly
% like a string scalar (strnum retains the field bytes).
plawk_emit_print_expr_for_context(ssa_strnum(Value), FieldSeparator, Context,
        Out, Globals, Setup) :-
    plawk_emit_print_str_id(Value, FieldSeparator, Context, Out, Globals, Setup).

plawk_emit_print_str_id(Value, _FieldSeparator, Context,
        string(Base, PtrIR), [EmptyGlobal], [StrCall, IsEmpty, SelPtr]) :-
    plawk_print_expr_value_base(Context, string, Base),
    format(atom(EmptyName), '~w_empty', [Base]),
    format(atom(EmptyGlobal),
        '@.~w = private constant [1 x i8] zeroinitializer', [EmptyName]),
    format(atom(StrCall), '  %~w_s = call i8* @wam_atom_to_string(i64 ~w)',
        [Base, Value]),
    format(atom(IsEmpty), '  %~w_empty_c = icmp eq i64 ~w, 0', [Base, Value]),
    format(atom(SelPtr),
        '  %~w_sptr = select i1 %~w_empty_c, i8* getelementptr ([1 x i8], [1 x i8]* @.~w, i64 0, i64 0), i8* %~w_s',
        [Base, Base, EmptyName, Base]),
    format(atom(PtrIR), '%~w_sptr', [Base]).

% Ternary print field `print (COND ? A : B)`: an i64 value via select.
plawk_emit_print_expr_for_context(ternary(Cond, Then, Else), FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, int, Base),
    plawk_print_expr_output_names(Context, int, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(ternary(Cond, Then, Else), FieldSeparator, Base, Base,
        ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(Expr, FieldSeparator, Context,
        f64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_f64_print_expr(Expr),
    !,
    plawk_print_expr_value_base(Context, f64, Base),
    plawk_print_expr_output_names(Context, f64, FmtPrefix, PrintPrefix),
    plawk_f64_expr_ir(Expr, FieldSeparator, Base, Base, ValueIR,
        GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(Expr, FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_i64_binary_expr(Expr, _LLVMOp, NamePart, _Left, _Right),
    plawk_i64_binary_print_kind(NamePart, Kind),
    plawk_print_expr_value_base(Context, Kind, Base),
    plawk_print_expr_output_names(Context, Kind, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(Expr, FieldSeparator, Base, Base,
        ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(prolog_call(Name, Args), FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_prolog_call_expr(prolog_call(Name, Args)),
    plawk_print_expr_value_base(Context, prolog_call, Base),
    plawk_print_expr_output_names(Context, prolog_call, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(prolog_call(Name, Args), FieldSeparator, Base, Base,
        ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(blob_dyncall(Args), FieldSeparator, Context,
        slice(FmtPrefix, PrintPrefix, LenIR, PtrIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, blob, Base),
    plawk_print_expr_output_names(Context, blob, FmtPrefix, PrintPrefix),
    plawk_blob_expr_ir(blob_dyncall(Args), FieldSeparator, Base,
        LenIR, PtrIR, GlobalParts, SetupParts).
% A record-view string field prints as a byte slice: its scalar reads are
% already substituted to SSA (ptr as i64, len), so inttoptr the pointer and
% pair it with the length for %.*s (empty when the field was absent -> 0/0).
plawk_emit_print_expr_for_context(blob_slice_vars(ssa(PtrVal), ssa(LenVal)),
        _FieldSeparator, Context,
        slice(FmtPrefix, PrintPrefix, LenIR, PtrIR), [], [PtrLine, LenLine]) :-
    plawk_print_expr_value_base(Context, blob, Base),
    plawk_print_expr_output_names(Context, blob, FmtPrefix, PrintPrefix),
    format(atom(PtrIR), '%~w_sptr', [Base]),
    format(atom(PtrLine), '  ~w = inttoptr i64 ~w to i8*', [PtrIR, PtrVal]),
    format(atom(LenIR), '%~w_slen', [Base]),
    format(atom(LenLine), '  ~w = trunc i64 ~w to i32', [LenIR, LenVal]).
plawk_emit_print_expr_for_context(blob_dyncall_named(Name, Args), FieldSeparator, Context,
        slice(FmtPrefix, PrintPrefix, LenIR, PtrIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, blob, Base),
    plawk_print_expr_output_names(Context, blob, FmtPrefix, PrintPrefix),
    plawk_blob_expr_ir(blob_dyncall_named(Name, Args), FieldSeparator, Base,
        LenIR, PtrIR, GlobalParts, SetupParts).
plawk_emit_print_expr_for_context(blob_dyncall_at(Source, Args), FieldSeparator, Context,
        slice(FmtPrefix, PrintPrefix, LenIR, PtrIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, blob, Base),
    plawk_print_expr_output_names(Context, blob, FmtPrefix, PrintPrefix),
    plawk_blob_expr_ir(blob_dyncall_at(Source, Args), FieldSeparator, Base,
        LenIR, PtrIR, GlobalParts, SetupParts).
plawk_emit_print_expr_for_context(blob_dyncall_at_named(Name, Source, Args),
        FieldSeparator, Context,
        slice(FmtPrefix, PrintPrefix, LenIR, PtrIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, blob, Base),
    plawk_print_expr_output_names(Context, blob, FmtPrefix, PrintPrefix),
    plawk_blob_expr_ir(blob_dyncall_at_named(Name, Source, Args),
        FieldSeparator, Base, LenIR, PtrIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(length(field(FieldIndex)), FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, length, Base),
    plawk_print_expr_output_names(Context, length, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(length(FieldIndex), FieldSeparator, Base, Base,
        ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(match_expr(field(FieldIndex), Regex), FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, match, Base),
    plawk_print_expr_value_base(Context, match_re, GlobalBase),
    plawk_print_expr_output_names(Context, match, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(match_expr(field(FieldIndex), Regex), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(special('RSTART'), FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, rstart, Base),
    plawk_print_expr_output_names(Context, rstart, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(special('RSTART'), FieldSeparator, Base, Base,
        ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(special('RLENGTH'), FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, rlength, Base),
    plawk_print_expr_output_names(Context, rlength, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(special('RLENGTH'), FieldSeparator, Base, Base,
        ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(special('ARGC'), FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, argc, Base),
    plawk_print_expr_output_names(Context, argc, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(special('ARGC'), FieldSeparator, Base, Base,
        ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(substr(field(FieldIndex), Start, Len), FieldSeparator, Context,
        slice(FmtPrefix, PrintPrefix, LenIR, PtrIR), [], [SliceIR]) :-
    plawk_print_expr_value_base(Context, substr, Base),
    plawk_print_expr_output_names(Context, substr, FmtPrefix, PrintPrefix),
    llvm_emit_atom_field_subslice('%line', FieldIndex, FieldSeparator, Start, Len, Base, SliceIR),
    format(atom(LenIR), '%~w_len', [Base]),
    format(atom(PtrIR), '%~w_ptr', [Base]).

plawk_emit_print_expr_for_context(index(field(FieldIndex), string(Needle)), FieldSeparator, Context,
        i64(FmtPrefix, PrintPrefix, ValueIR), GlobalParts, SetupParts) :-
    plawk_print_expr_value_base(Context, index, Base),
    plawk_print_expr_value_base(Context, index_needle, GlobalBase),
    plawk_print_expr_output_names(Context, index, FmtPrefix, PrintPrefix),
    plawk_i64_expr_ir(index(FieldIndex, Needle), FieldSeparator, Base, GlobalBase,
        ValueIR, GlobalParts, SetupParts).

plawk_emit_print_expr_for_context(tolower(field(FieldIndex)), FieldSeparator, Context,
        case_slice(lower, LowerBase, LenIR, PtrIR), [], SetupParts) :-
    plawk_print_expr_value_base(Context, tolower, LowerBase),
    plawk_emit_case_source_slice_ir(FieldIndex, FieldSeparator, LowerBase, LenIR, PtrIR,
        SetupParts).

plawk_emit_print_expr_for_context(toupper(field(FieldIndex)), FieldSeparator, Context,
        case_slice(upper, UpperBase, LenIR, PtrIR), [], SetupParts) :-
    plawk_print_expr_value_base(Context, toupper, UpperBase),
    plawk_emit_case_source_slice_ir(FieldIndex, FieldSeparator, UpperBase, LenIR, PtrIR,
        SetupParts).

plawk_emit_print_expr_for_context(field(FieldIndex), binfmt(Types), Context,
        PrintType, [], SetupLines) :-
    plawk_binfmt_field_type(binfmt(Types), FieldIndex, s(Width)),
    !,
    % Fixed-width string field: print the bytes up to the first NUL or
    % the field width, whichever comes first (%.*s takes an i32 length).
    plawk_print_expr_value_base(Context, binfield, Base),
    plawk_print_expr_output_names(Context, binfield, FmtPrefix, PrintPrefix),
    plawk_binfmt_field_offset(binfmt(Types), FieldIndex, Offset),
    format(atom(PtrIR), '%~w_ptr', [Base]),
    format(atom(LenIR), '%~w_len', [Base]),
    format(atom(GepIR),
        '  ~w = getelementptr i8, i8* %rec, i64 ~w', [PtrIR, Offset]),
    format(atom(LenCall),
        '  %~w_len64 = call i64 @strnlen(i8* ~w, i64 ~w)',
        [Base, PtrIR, Width]),
    format(atom(LenTrunc),
        '  ~w = trunc i64 %~w_len64 to i32', [LenIR, Base]),
    SetupLines = [GepIR, LenCall, LenTrunc],
    PrintType = slice(FmtPrefix, PrintPrefix, LenIR, PtrIR).
plawk_emit_print_expr_for_context(field(FieldIndex), binfmt(Types), Context,
        PrintType, [], LoadLines) :-
    plawk_binfmt_field_type(binfmt(Types), FieldIndex, Type),
    !,
    plawk_print_expr_value_base(Context, binfield, Base),
    plawk_print_expr_output_names(Context, binfield, FmtPrefix, PrintPrefix),
    plawk_binfmt_field_load_lines(binfmt(Types), FieldIndex, Base, ValueIR,
        LoadLines),
    (   Type == i64
    ->  PrintType = i64(FmtPrefix, PrintPrefix, ValueIR)
    ;   PrintType = f64(FmtPrefix, PrintPrefix, ValueIR)
    ).

plawk_emit_print_expr_for_context(field(0), _FieldSeparator, Context,
        slice(FmtPrefix, PrintPrefix, LenIR, '%line_s'), [], [LineLen64, LineLen]) :-
    plawk_print_expr_output_names(Context, line, FmtPrefix, PrintPrefix),
    plawk_emit_print_line_length_ir(Context, LenIR, LineLen64, LineLen).

plawk_emit_print_expr_for_context(field(FieldIndex), FieldSeparator, Context,
        slice(FmtPrefix, PrintPrefix, LenIR, PtrIR), [], [SliceIR]) :-
    FieldIndex > 0,
    plawk_print_expr_value_base(Context, field, Base),
    plawk_print_expr_output_names(Context, field, FmtPrefix, PrintPrefix),
    llvm_emit_atom_field_slice('%line', FieldIndex, FieldSeparator, Base, SliceIR),
    format(atom(LenIR), '%~w_len', [Base]),
    format(atom(PtrIR), '%~w_ptr', [Base]).

plawk_print_expr_value_base(print_context(normal, _Prefix, Index), Kind, Base) :-
    plawk_normal_print_expr_value_base(Kind, Index, Base).
plawk_print_expr_value_base(print_context(prefixed, Prefix, Index), Kind, Base) :-
    format(atom(Base), '~w_~w_~w', [Prefix, Kind, Index]).

plawk_normal_print_expr_value_base(nf, Index, Base) :-
    format(atom(Base), 'plawk_nf_~w', [Index]).
plawk_normal_print_expr_value_base(int, Index, Base) :-
    format(atom(Base), 'plawk_int_~w', [Index]).
plawk_normal_print_expr_value_base(int_add, Index, Base) :-
    format(atom(Base), 'plawk_int_add_~w', [Index]).
plawk_normal_print_expr_value_base(int_sub, Index, Base) :-
    format(atom(Base), 'plawk_int_sub_~w', [Index]).
plawk_normal_print_expr_value_base(int_mul, Index, Base) :-
    format(atom(Base), 'plawk_int_mul_~w', [Index]).
plawk_normal_print_expr_value_base(int_div, Index, Base) :-
    format(atom(Base), 'plawk_int_div_~w', [Index]).
plawk_normal_print_expr_value_base(int_mod, Index, Base) :-
    format(atom(Base), 'plawk_int_mod_~w', [Index]).
plawk_normal_print_expr_value_base(prolog_call, Index, Base) :-
    format(atom(Base), 'plawk_prolog_call_~w', [Index]).
plawk_normal_print_expr_value_base(f64, Index, Base) :-
    format(atom(Base), 'plawk_f64_~w', [Index]).
plawk_normal_print_expr_value_base(binfield, Index, Base) :-
    format(atom(Base), 'plawk_binfield_~w', [Index]).
plawk_normal_print_expr_value_base(length, Index, Base) :-
    format(atom(Base), 'plawk_length_~w', [Index]).
plawk_normal_print_expr_value_base(substr, Index, Base) :-
    format(atom(Base), 'plawk_substr_~w', [Index]).
plawk_normal_print_expr_value_base(blob, Index, Base) :-
    format(atom(Base), 'plawk_blob_~w', [Index]).
plawk_normal_print_expr_value_base(index, Index, Base) :-
    format(atom(Base), 'plawk_index_~w', [Index]).
plawk_normal_print_expr_value_base(index_needle, Index, Base) :-
    format(atom(Base), 'plawk_index_needle_~w', [Index]).
plawk_normal_print_expr_value_base(tolower, Index, Base) :-
    format(atom(Base), 'plawk_tolower_~w', [Index]).
plawk_normal_print_expr_value_base(toupper, Index, Base) :-
    format(atom(Base), 'plawk_toupper_~w', [Index]).
plawk_normal_print_expr_value_base(field, Index, Base) :-
    format(atom(Base), 'plawk_field_~w', [Index]).
plawk_normal_print_expr_value_base(string, Index, Base) :-
    format(atom(Base), 'plawk_string_~w', [Index]).

plawk_emit_print_line_length_ir(print_context(normal, _Prefix, Index), LenIR, LineLen64, LineLen) :-
    format(atom(LineLen64),
        '  %line_len64_~w = call i64 @strlen(i8* %line_s)',
        [Index]),
    format(atom(LineLen),
        '  %line_len_~w = trunc i64 %line_len64_~w to i32',
        [Index, Index]),
    format(atom(LenIR), '%line_len_~w', [Index]).
plawk_emit_print_line_length_ir(print_context(prefixed, Prefix, Index), LenIR, LineLen64, LineLen) :-
    format(atom(Base), '~w_line_~w', [Prefix, Index]),
    format(atom(LineLen64),
        '  %~w_len64 = call i64 @strlen(i8* %line_s)',
        [Base]),
    format(atom(LineLen),
        '  %~w_len = trunc i64 %~w_len64 to i32',
        [Base, Base]),
    format(atom(LenIR), '%~w_len', [Base]).

plawk_print_expr_output_names(print_context(normal, _Prefix, _Index), field, slice, slice) :-
    !.
plawk_print_expr_output_names(print_context(normal, _Prefix, _Index), Kind, Kind, Kind).
plawk_print_expr_output_names(print_context(prefixed, Prefix, Index), Kind, Base, Base) :-
    format(atom(Base), '~w_~w_~w', [Prefix, Kind, Index]).

plawk_emit_case_source_slice_ir(0, _FieldSeparator, Base, LenIR, '%line_s', [LineLen64]) :-
    format(atom(LineLen64),
        '  %~w_len64 = call i64 @strlen(i8* %line_s)',
        [Base]),
    format(atom(LenIR), '%~w_len64', [Base]).
plawk_emit_case_source_slice_ir(FieldIndex, FieldSeparator, Base, LenIR, PtrIR, [SliceIR]) :-
    FieldIndex > 0,
    llvm_emit_atom_field_slice('%line', FieldIndex, FieldSeparator, Base, SliceIR),
    format(atom(LenIR), '%~w_len64', [Base]),
    format(atom(PtrIR), '%~w_ptr', [Base]).

plawk_print_expr_output_ir(i64(FmtPrefix, PrintPrefix, ValueIR), Index, Parts) :-
    format(atom(FmtVar), '~w_fmt_~w', [FmtPrefix, Index]),
    format(atom(PrintVar), 'printed_~w_~w', [PrintPrefix, Index]),
    llvm_emit_printf_i64(plawk_surface_print_i64, FmtVar, PrintVar, ValueIR, Parts).

plawk_print_expr_output_ir(slice(FmtPrefix, PrintPrefix, LenIR, PtrIR), Index, Parts) :-
    format(atom(FmtVar), '~w_fmt_~w', [FmtPrefix, Index]),
    format(atom(PrintVar), 'printed_~w_~w', [PrintPrefix, Index]),
    llvm_emit_printf_slice(plawk_surface_print_slice, FmtVar, PrintVar, LenIR, PtrIR,
        Parts).

plawk_print_expr_output_ir(f64(FmtPrefix, PrintPrefix, ValueIR), Index, [FmtPtr, PrintCall]) :-
    format(atom(FmtVar), '~w_fmt_~w', [FmtPrefix, Index]),
    format(atom(PrintVar), 'printed_~w_~w', [PrintPrefix, Index]),
    format(atom(FmtPtr),
        '  %~w = getelementptr [3 x i8], [3 x i8]* @.plawk_surface_print_f64, i32 0, i32 0',
        [FmtVar]),
    format(atom(PrintCall),
        '  %~w = call i32 (i8*, ...) @printf(i8* %~w, double ~w)',
        [PrintVar, FmtVar, ValueIR]).

% A string-literal print field: the setup already built a pointer to the
% interned C string (%..._ptr); print it with "%s". This is what lets a
% constant field appear in a print (`print "hi"`, `print $1, "x"`, and -- via
% the print grammar's bare-integer-to-string lowering -- `print 1`). The
% "%s" format global (@.plawk_surface_print_string) is in the driver's runtime
% globals.
plawk_print_expr_output_ir(string(Base, PtrIR), Index, [FmtPtr, PrintCall]) :-
    format(atom(FmtVar), '~w_fmt_~w', [Base, Index]),
    format(atom(PrintVar), 'printed_~w_~w', [Base, Index]),
    format(atom(FmtPtr),
        '  %~w = getelementptr [3 x i8], [3 x i8]* @.plawk_surface_print_string, i32 0, i32 0',
        [FmtVar]),
    format(atom(PrintCall),
        '  %~w = call i32 (i8*, ...) @printf(i8* %~w, i8* ~w)',
        [PrintVar, FmtVar, PtrIR]).

plawk_print_expr_output_ir(case_slice(Mode, PrintBase, LenIR, PtrIR), _Index, [PrintCall]) :-
    llvm_emit_ascii_case_slice_print(Mode, PtrIR, LenIR, PrintBase, PrintCall).

plawk_prefixed_print_expr_output_ir(i64(FmtPrefix, PrintPrefix, ValueIR), _Prefix, Index, Parts) :-
    plawk_print_expr_output_ir(i64(FmtPrefix, PrintPrefix, ValueIR), Index, Parts).

plawk_prefixed_print_expr_output_ir(slice(FmtPrefix, PrintPrefix, LenIR, PtrIR), _Prefix, Index, Parts) :-
    plawk_print_expr_output_ir(slice(FmtPrefix, PrintPrefix, LenIR, PtrIR), Index, Parts).

plawk_prefixed_print_expr_output_ir(f64(FmtPrefix, PrintPrefix, ValueIR), _Prefix, Index, Parts) :-
    plawk_print_expr_output_ir(f64(FmtPrefix, PrintPrefix, ValueIR), Index, Parts).

plawk_prefixed_print_expr_output_ir(case_slice(Mode, PrintBase, LenIR, PtrIR), _Prefix, _Index, [PrintCall]) :-
    llvm_emit_ascii_case_slice_print(Mode, PtrIR, LenIR, PrintBase, PrintCall).

plawk_prefixed_print_expr_output_ir(string(Base, PtrIR), _Prefix, Index, Parts) :-
    format(atom(FmtVar), '~w_fmt_~w', [Base, Index]),
    format(atom(PrintVar), 'printed_~w_~w', [Base, Index]),
    llvm_emit_printf_string(plawk_surface_print_string, FmtVar, PrintVar, PtrIR, Parts).
