:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

:- dynamic user:plawk_binassoc_marker/0.

user:plawk_binassoc_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_binary_assoc, [condition(clang_available)]).

test(parses_int_assoc_keys) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 i64\" } { counts[$1]++ } END { print counts[5], counts[-3] }\n", Program),
    assertion(Program == program([begin([set(var('BINFMT'), string("i64 i64"))])],
        [rule(always, [inc_assoc(var(counts), field(1))])],
        [end([print([assoc(var(counts), int(5)), assoc(var(counts), int(-3))])])])).

test(binary_groupby_ir_keys_are_raw_fields) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 i64\" } { counts[$1]++ } END { for (k in counts) print k, counts[k] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    % The table key is the raw i64 field load, not an interned atom id.
    assertion(once(sub_atom(DriverIR, _, _, _,
        '%assoc_rule_0_action_0_key = load i64, i64* %assoc_rule_0_action_0_key_tp, align 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _,
        '@wam_assoc_i64_inc(%WamAssocI64Table* %plawk_assoc_table_0, i64 %assoc_rule_0_action_0_key, i64 1)'))),
    % for-in keys print numerically without atom-table round trips.
    assertion(once(sub_atom(DriverIR, _, _, _,
        '@printf(i8* %forin_key_fmt_0, i64 %forin_key_id)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@wam_atom_to_string')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@wam_atom_field_slice_value')),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'read_line')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    % The only interning left is the one-shot input-path atom in entry.
    findall(Before, sub_atom(DriverIR, Before, _, _, '@wam_intern_atom('), InternSites),
    assertion(InternSites = [_]),
    !.

test(binary_assoc_int_key_lookup_ir) :-
    plawk_parse_string("BEGIN { BINFMT = \"i64 i64\" } $2 > 100 { counts[$1]++ } END { print counts[5], counts[-3] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.bin', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        '@wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_0, i64 5)'))),
    assertion(once(sub_atom(DriverIR, _, _, _,
        '@wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_0, i64 -3)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@wam_atom_field_slice_value')),
    !.

test(binary_assoc_rejects_text_shaped_programs) :-
    Rejects = [
        % string keys need interning -- not available in binary mode
        "BEGIN { BINFMT = \"i64 i64\" } { counts[$1]++ } END { print counts[\"x\"] }\n",
        % f64 fields cannot key an i64 table
        "BEGIN { BINFMT = \"i64 f64\" } { counts[$2]++ } END { for (k in counts) print k, counts[k] }\n",
        % key field out of range for the record layout
        "BEGIN { BINFMT = \"i64 i64\" } { counts[$3]++ } END { for (k in counts) print k, counts[k] }\n",
        % regex guards are text-only
        "BEGIN { BINFMT = \"i64 i64\" } /^ERR/ { counts[$1]++ } END { for (k in counts) print k, counts[k] }\n"
    ],
    forall(member(Source, Rejects),
        ( plawk_parse_string(Source, Program)
        -> assertion(\+ plawk_program_native_driver_ir(Program, 'input.bin', _))
        ;  true
        )).

test(text_mode_rejects_int_assoc_keys) :-
    % In text mode assoc keys are atom ids; a literal integer key would
    % silently collide with them, so it must be refused.
    plawk_parse_string("{ counts[$1]++ } END { print counts[5] }\n", Program),
    assertion(\+ plawk_program_native_driver_ir(Program, 'input.txt', _)).

test(text_mode_string_assoc_still_works) :-
    plawk_parse_string("{ counts[$1]++ } END { print counts[\"alpha\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_assoc_i64_get'))),
    !.

test(surface_binary_groupby_forin) :-
    run_binassoc_smoke_sorted("BEGIN { BINFMT = \"i64 i64\" } { counts[$1]++ } END { for (k in counts) print k, counts[k] }\n",
        [5-0, (-3)-0, 5-0, 9-0, (-3)-0, 5-0],
        ["-3 2", "5 3", "9 1"]).

test(surface_binary_guarded_groupby_int_lookup) :-
    run_binassoc_smoke("BEGIN { BINFMT = \"i64 i64\" } $2 > 100 { counts[$1]++ } END { print counts[5], counts[-3] }\n",
        [5-200, 5-50, (-3)-300, 5-150, 9-999, (-3)-50],
        "2 1\n").

test(surface_binary_missing_key_reads_zero) :-
    run_binassoc_smoke("BEGIN { BINFMT = \"i64 i64\" } { counts[$1]++ } END { print counts[42] }\n",
        [5-0, 9-0],
        "0\n").

test(surface_binary_groupby_second_field_key) :-
    run_binassoc_smoke_sorted("BEGIN { BINFMT = \"i64 i64\" } { counts[$2]++ } END { for (k in counts) print k, counts[k] }\n",
        [1-7, 2-7, 3-8],
        ["7 2", "8 1"]).

test(surface_binary_groupby_empty_input) :-
    run_binassoc_smoke("BEGIN { BINFMT = \"i64 i64\" } { counts[$1]++ } END { for (k in counts) print k, counts[k] }\n",
        [],
        "").

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

write_pairs(Path, Pairs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(A-B, Pairs),
            ( write_i64_le(Out, A), write_i64_le(Out, B) )),
        close(Out)).

build_binassoc_probe(Source, Dir, BinPath) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_binary_assoc', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_binassoc_marker/0 ],
        [module_name('plawk_binary_assoc')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'probe_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1', [LLPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, BuildOut),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[plawk binary assoc build output]~n~w~n", [BuildOut]),
       throw(plawk_binary_assoc_build_failed(Status))
    ).

run_binassoc_probe(Source, Pairs, OutStr) :-
    build_binassoc_probe(Source, Dir, BinPath),
    directory_file_path(Dir, 'input.bin', InputPath),
    write_pairs(InputPath, Pairs),
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[plawk binary assoc run output]~n~w~n", [OutStr]),
       throw(plawk_binary_assoc_run_failed(Status))
    ).

run_binassoc_smoke(Source, Pairs, ExpectedOutput) :-
    run_binassoc_probe(Source, Pairs, OutStr),
    assertion(OutStr == ExpectedOutput),
    !.

% for-in iteration order follows the hash table, so compare sorted lines.
run_binassoc_smoke_sorted(Source, Pairs, ExpectedLines) :-
    run_binassoc_probe(Source, Pairs, OutStr),
    split_string(OutStr, "\n", "", Split0),
    exclude(==(""), Split0, Lines0),
    msort(Lines0, SortedLines),
    msort(ExpectedLines, SortedExpected),
    assertion(SortedLines == SortedExpected),
    !.

:- end_tests(plawk_binary_assoc).
