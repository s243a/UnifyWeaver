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

:- dynamic user:plawk_forin_marker/0.

user:plawk_forin_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_surface_forin_end_print, [condition(clang_available)]).

test(parses_forin_end_print_rule) :-
    plawk_parse_string("{ counts[$1]++ } END { for (k in counts) print k, counts[k] }\n", Program),
    assertion(Program == program([], [rule(always, [inc_assoc(var(counts), field(1))])],
        [end([for_in(var(k), var(counts),
            [print([var(k), assoc(var(counts), var(k))])])])])).

test(parses_forin_end_print_braced_body) :-
    plawk_parse_string("{ counts[$1]++ } END { for (k in counts) { print k, counts[k] } }\n", Program),
    assertion(Program == program([], [rule(always, [inc_assoc(var(counts), field(1))])],
        [end([for_in(var(k), var(counts),
            [print([var(k), assoc(var(counts), var(k))])])])])).

test(parses_forin_end_print_string_and_other_array) :-
    plawk_parse_string("{ counts[$1]++ } $1 == \"ERROR\" { errs[$1]++ } END { for (k in counts) print \"key\", k, counts[k], errs[k] }\n", Program),
    assertion(Program == program([],
        [rule(always, [inc_assoc(var(counts), field(1))]),
         rule(field_eq(1, "ERROR"), [inc_assoc(var(errs), field(1))])],
        [end([for_in(var(k), var(counts),
            [print([string("key"), var(k),
                    assoc(var(counts), var(k)),
                    assoc(var(errs), var(k))])])])])).

test(parses_forin_without_keyword_space) :-
    plawk_parse_string("{ counts[$1]++ } END { for(k in counts) print k }\n", Program),
    assertion(Program == program([], [rule(always, [inc_assoc(var(counts), field(1))])],
        [end([for_in(var(k), var(counts), [print([var(k)])])])])).

test(forin_ir_walks_table_with_iter_next) :-
    plawk_parse_string("{ counts[$1]++ } END { for (k in counts) print k, counts[k] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_assoc_i64_iter_next(%WamAssocI64Table* %plawk_assoc_table_0, i64 %forin_idx)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_assoc_i64_key_at(%WamAssocI64Table* %plawk_assoc_table_0, i64 %forin_slot)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_assoc_i64_value_at(%WamAssocI64Table* %plawk_assoc_table_0, i64 %forin_slot)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%forin_key_s_0 = call i8* @wam_atom_to_string(i64 %forin_key_id)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(forin_ir_other_array_lookup_uses_get) :-
    plawk_parse_string("{ counts[$1]++ } $1 == \"ERROR\" { errs[$1]++ } END { for (k in counts) print k, counts[k], errs[k] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    % Tables are sorted by array name: counts is table 0, errs is table 1.
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_assoc_i64_value_at(%WamAssocI64Table* %plawk_assoc_table_0, i64 %forin_slot)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_assoc_i64_get(%WamAssocI64Table* %plawk_assoc_table_1, i64 %forin_key_id)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_forin_prints_counts_by_key) :-
    run_forin_print_smoke("{ counts[$1]++ } END { for (k in counts) print k, counts[k] }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        ["ERROR 2", "INFO 1", "WARN 1"]).

test(surface_forin_prints_guarded_counts_by_key) :-
    run_forin_print_smoke("$1 == \"ERROR\" { by_component[$2]++ } END { for (k in by_component) print k, by_component[k] }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\nERROR disk warm\n",
        ["disk 2", "net 1"]).

test(surface_forin_prints_other_array_lookup_with_missing_default) :-
    run_forin_print_smoke("{ counts[$1]++ } $1 == \"ERROR\" { errs[$1]++ } END { for (k in counts) print k, counts[k], errs[k] }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        ["ERROR 2 2", "INFO 1 0", "WARN 1 0"]).

test(surface_forin_prints_string_literal_labels) :-
    run_forin_print_smoke("{ counts[$1]++ } END { for (k in counts) print \"key\", k, counts[k] }\n",
        "INFO boot ok\nERROR disk full\nERROR net down\n",
        ["key ERROR 2", "key INFO 1"]).

test(surface_forin_uses_output_separator) :-
    run_forin_print_smoke("BEGIN { OFS = \",\" } { counts[$1]++ } END { for (k in counts) print k, counts[k] }\n",
        "INFO boot ok\nERROR disk full\nERROR net down\n",
        ["ERROR,2", "INFO,1"]).

test(surface_forin_uses_field_separator_for_keys) :-
    run_forin_print_smoke("BEGIN { FS = \":\" } { counts[$2]++ } END { for (k in counts) print k, counts[k] }\n",
        "a:disk:1\nb:net:2\nc:disk:3\n",
        ["disk 2", "net 1"]).

test(surface_forin_braced_body_prints_counts) :-
    run_forin_print_smoke("{ counts[$1]++ } END { for (k in counts) { print k, counts[k] } }\n",
        "INFO boot ok\nERROR disk full\nERROR net down\n",
        ["ERROR 2", "INFO 1"]).

test(surface_forin_empty_input_prints_nothing) :-
    run_forin_print_smoke("{ counts[$1]++ } END { for (k in counts) print k, counts[k] }\n",
        "",
        []).

% For-in iteration order follows the hash table's slot order, which awk
% leaves unspecified. The smoke compares sorted output lines so assertions
% stay stable across atom-id and hashing changes.
run_forin_print_smoke(Source, Input, ExpectedSortedLines) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_surface_forin_end_print', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.txt', InputPath),
    setup_call_cleanup(
        open(InputPath, write, In, [type(binary)]),
        format(In, '~s', [Input]),
        close(In)),
    plawk_parse_string(Source, Program),
    plawk_program_native_driver_ir(Program, InputPath, DriverIR),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    directory_file_path(Dir, 'plawk_surface_forin_end_print.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_forin_marker/0 ],
        [module_name('plawk_surface_forin_end_print')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'plawk_surface_forin_end_print_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> split_string(OutStr, "\n", "", Lines0),
       exclude(==(""), Lines0, Lines),
       msort(Lines, SortedLines),
       maplist(atom_string, ExpectedSortedLines, ExpectedStrings0),
       msort(ExpectedStrings0, ExpectedStrings),
       assertion(SortedLines == ExpectedStrings)
    ;  format(user_error, "~n[plawk surface forin end print output]~n~w~n",
              [OutStr]),
       throw(plawk_surface_forin_end_print_failed(Status))
    ),
    !.

:- end_tests(plawk_surface_forin_end_print).
