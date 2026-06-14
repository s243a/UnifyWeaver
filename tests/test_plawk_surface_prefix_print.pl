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

:- dynamic user:plawk_surface_marker/0.

user:plawk_surface_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_surface_prefix_print, [condition(clang_available)]).

test(parses_prefix_print_rule) :-
    plawk_parse_string("/^ERROR/ { print $0 }\n", Program),
    assertion(Program == program([], [rule(prefix("ERROR"), [print([field(0)])])], [])).

test(parses_field_eq_print_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { print $0 }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"), [print([field(0)])])], [])).

test(parses_field_eq_print_fields_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { print $2, $3 }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"), [print([field(2), field(3)])])], [])).

test(parses_field_eq_print_nr_fields_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { print NR, $2, $3 }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [print([special('NR'), field(2), field(3)])])], [])).

test(parses_field_eq_print_nr_nf_fields_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { print NR, NF, $2, $3 }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [print([special('NR'), special('NF'), field(2), field(3)])])], [])).

test(parses_field_eq_print_length_field_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { print NR, NF, length($2), $2 }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [print([special('NR'), special('NF'), length(field(2)), field(2)])])], [])).

test(parses_field_eq_print_substr_field_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { print substr($2, 1, 3), substr($0, 7, 4) }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [print([substr(field(2), 1, 3), substr(field(0), 7, 4)])])], [])).

test(parses_field_eq_print_index_field_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { print index($2, \"sk\"), index($0, \"disk\") }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [print([index(field(2), string("sk")), index(field(0), string("disk"))])])], [])).

test(parses_field_eq_print_case_field_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { print tolower($2), toupper($0) }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [print([tolower(field(2)), toupper(field(0))])])], [])).

test(parses_field_eq_increment_end_print_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { count++ } END { print count }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"), [inc(var(count))])],
        [end([print([var(count)])])])).

test(parses_field_eq_multi_increment_end_print_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { errors++; matches++ } END { print errors, matches }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [inc(var(errors)), inc(var(matches))])],
        [end([print([var(errors), var(matches)])])])).

test(parses_multi_rule_scalar_end_print_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { errors++ } $1 == \"WARN\" { warnings++ } END { print errors, warnings }\n", Program),
    assertion(Program == program([],
        [rule(field_eq(1, "ERROR"), [inc(var(errors))]),
         rule(field_eq(1, "WARN"), [inc(var(warnings))])],
        [end([print([var(errors), var(warnings)])])])).

test(parses_terminal_next_scalar_rule) :-
    plawk_parse_string("$1 == \"DEBUG\" { skipped++; next } { total++ } END { print total, skipped }\n", Program),
    assertion(Program == program([],
        [rule(field_eq(1, "DEBUG"), [inc(var(skipped)), next]),
         rule(always, [inc(var(total))])],
        [end([print([var(total), var(skipped)])])])).

test(parses_terminal_next_assoc_rule) :-
    plawk_parse_string("$1 == \"DEBUG\" { skipped[$2]++; next } { counts[$1]++ } END { print skipped[\"trace\"], counts[\"DEBUG\"], counts[\"ERROR\"] }\n", Program),
    assertion(Program == program([],
        [rule(field_eq(1, "DEBUG"), [inc_assoc(var(skipped), field(2)), next]),
         rule(always, [inc_assoc(var(counts), field(1))])],
        [end([print([
            assoc(var(skipped), string("trace")),
            assoc(var(counts), string("DEBUG")),
            assoc(var(counts), string("ERROR"))
        ])])])).

test(parses_terminal_next_mixed_rule) :-
    plawk_parse_string("$1 == \"DEBUG\" { skipped++; by_kind[$2]++; next } { total++; counts[$1]++ } END { print total, skipped, by_kind[\"trace\"], counts[\"DEBUG\"], counts[\"ERROR\"] }\n", Program),
    assertion(Program == program([],
        [rule(field_eq(1, "DEBUG"), [inc(var(skipped)), inc_assoc(var(by_kind), field(2)), next]),
         rule(always, [inc(var(total)), inc_assoc(var(counts), field(1))])],
        [end([print([
            var(total), var(skipped), assoc(var(by_kind), string("trace")),
            assoc(var(counts), string("DEBUG")), assoc(var(counts), string("ERROR"))
        ])])])).

test(parses_terminal_next_only_rule) :-
    plawk_parse_string("$1 == \"DEBUG\" { next } { total++ } END { print total }\n", Program),
    assertion(Program == program([],
        [rule(field_eq(1, "DEBUG"), [next]),
         rule(always, [inc(var(total))])],
        [end([print([var(total)])])])).

test(parses_nonterminal_next_rule) :-
    plawk_parse_string("$1 == \"DEBUG\" { next; skipped++ } { total++ } END { print total, skipped }\n", Program),
    assertion(Program == program([],
        [rule(field_eq(1, "DEBUG"), [next, inc(var(skipped))]),
         rule(always, [inc(var(total))])],
        [end([print([var(total), var(skipped)])])])).

test(parses_field_eq_scalar_add_assign_end_print_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { bytes += length($0); hits += 2 } END { print bytes, hits }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [add(var(bytes), length(field(0))), add(var(hits), int(2))])],
        [end([print([var(bytes), var(hits)])])])).

test(parses_always_scalar_add_assign_end_print_rule) :-
    plawk_parse_string("{ total += 3 } END { print total }\n", Program),
    assertion(Program == program([], [rule(always, [add(var(total), int(3))])],
        [end([print([var(total)])])])).

test(parses_assoc_count_end_print_rule) :-
    plawk_parse_string("{ counts[$1]++ } END { print counts[\"ERROR\"], counts[\"WARN\"] }\n", Program),
    assertion(Program == program([], [rule(always, [inc_assoc(var(counts), field(1))])],
        [end([print([assoc(var(counts), string("ERROR")),
                     assoc(var(counts), string("WARN"))])])])).

test(parses_multi_assoc_count_end_print_rule) :-
    plawk_parse_string("{ counts[$1]++; by_component[$2]++ } END { print counts[\"ERROR\"], by_component[\"disk\"] }\n", Program),
    assertion(Program == program([], [rule(always,
        [inc_assoc(var(counts), field(1)),
         inc_assoc(var(by_component), field(2))])],
        [end([print([assoc(var(counts), string("ERROR")),
                     assoc(var(by_component), string("disk"))])])])).

test(parses_guarded_assoc_count_end_print_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { by_component[$2]++ } $1 == \"WARN\" { warnings[$2]++ } END { print by_component[\"disk\"], warnings[\"cpu\"] }\n", Program),
    assertion(Program == program([],
        [rule(field_eq(1, "ERROR"), [inc_assoc(var(by_component), field(2))]),
         rule(field_eq(1, "WARN"), [inc_assoc(var(warnings), field(2))])],
        [end([print([
            assoc(var(by_component), string("disk")),
            assoc(var(warnings), string("cpu"))
        ])])])).

test(parses_mixed_scalar_assoc_end_print_rule) :-
    plawk_parse_string("{ total++; counts[$1]++ } $1 == \"ERROR\" { errors++; by_component[$2]++ } END { print total, errors, counts[\"WARN\"], by_component[\"disk\"] }\n", Program),
    assertion(Program == program([],
        [rule(always, [inc(var(total)), inc_assoc(var(counts), field(1))]),
         rule(field_eq(1, "ERROR"),
            [inc(var(errors)), inc_assoc(var(by_component), field(2))])],
        [end([print([
            var(total),
            var(errors),
            assoc(var(counts), string("WARN")),
            assoc(var(by_component), string("disk"))
        ])])])).

test(parses_end_print_string_literal_fields) :-
    plawk_parse_string("{ total++ } END { print \"total\", total }\n", Program),
    assertion(Program == program([], [rule(always, [inc(var(total))])],
        [end([print([string("total"), var(total)])])])).

test(parses_mixed_end_print_string_literal_fields) :-
    plawk_parse_string("{ total++; counts[$1]++ } END { print \"total\", total, \"errors\", counts[\"ERROR\"] }\n", Program),
    assertion(Program == program([], [rule(always,
        [inc(var(total)), inc_assoc(var(counts), field(1))])],
        [end([print([
            string("total"),
            var(total),
            string("errors"),
            assoc(var(counts), string("ERROR"))
        ])])])).

test(parses_begin_print_string_literal_fields) :-
    plawk_parse_string("BEGIN { print \"kind\", \"count\" } { total++ } END { print \"total\", total }\n", Program),
    assertion(Program == program([begin([print([string("kind"), string("count")])])],
        [rule(always, [inc(var(total))])],
        [end([print([
            string("total"),
            var(total)
        ])])])).

test(parses_begin_field_separator_assignment) :-
    plawk_parse_string("BEGIN { FS = \":\" } $1 == \"ERROR\" { counts[$2]++ } END { print counts[\"disk\"] }\n", Program),
    assertion(Program == program([begin([set(var('FS'), string(":"))])],
        [rule(field_eq(1, "ERROR"), [inc_assoc(var(counts), field(2))])],
        [end([print([
            assoc(var(counts), string("disk"))
        ])])])).

test(parses_begin_field_separator_with_header) :-
    plawk_parse_string("BEGIN { FS = \":\"; print \"kind\", \"count\" } { counts[$1]++ } END { print counts[\"ERROR\"] }\n", Program),
    assertion(Program == program([begin([set(var('FS'), string(":")), print([string("kind"), string("count")])])],
        [rule(always, [inc_assoc(var(counts), field(1))])],
        [end([print([assoc(var(counts), string("ERROR"))])])])).

test(parses_begin_output_separator_assignment) :-
    plawk_parse_string("BEGIN { FS = \":\"; OFS = \",\" } $1 == \"ERROR\" { print $2, $3 }\n", Program),
    assertion(Program == program([begin([set(var('FS'), string(":")), set(var('OFS'), string(","))])],
        [rule(field_eq(1, "ERROR"), [print([field(2), field(3)])])],
        [])).

test(surface_prefix_prints_matching_records) :-
    run_surface_print_smoke("/^ERROR/ { print $0 }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "ERROR disk full\nERROR net down\n").

test(surface_field_eq_prints_matching_records) :-
    run_surface_print_smoke("$1 == \"ERROR\" { print $0 }\n",
        "INFO boot ok\nNOTERROR misleading\nWARN cpu hot\nERROR net down\n",
        "ERROR net down\n").

test(surface_field_eq_prints_selected_fields) :-
    run_surface_print_smoke("$1 == \"ERROR\" { print $2, $3 }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "disk full\nnet down\n").

test(surface_field_eq_prints_nr_and_selected_fields) :-
    run_surface_print_smoke("$1 == \"ERROR\" { print NR, $2, $3 }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "2 disk full\n4 net down\n").

test(surface_field_eq_prints_nr_nf_and_selected_fields) :-
    run_surface_print_smoke("$1 == \"ERROR\" { print NR, NF, $2, $3 }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down now\n",
        "2 3 disk full\n4 4 net down\n").

test(surface_field_eq_prints_native_field_lengths) :-
    run_surface_print_smoke("$1 == \"ERROR\" { print NR, NF, length($2), $2 }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR network down now\n",
        "2 3 4 disk\n4 4 7 network\n").

test(surface_default_space_fs_collapses_whitespace_runs) :-
    run_surface_print_smoke("$1 == \"ERROR\" { print NF, $2, length($2) }\n",
        "  INFO   boot ok\n\tERROR   disk   full  \nWARN cpu hot\n  ERROR\tnetwork  down now\n",
        "3 disk 4\n4 network 7\n").

test(surface_field_eq_prints_native_substrings) :-
    run_surface_print_smoke("$1 == \"ERROR\" { print substr($2, 1, 3), substr($0, 7, 4) }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR network down now\n",
        "dis disk\nnet netw\n").

test(surface_field_eq_prints_native_index_values) :-
    run_surface_print_smoke("$1 == \"ERROR\" { print index($2, \"sk\"), index($0, \"disk\") }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR network down now\n",
        "3 7\n0 0\n").

test(surface_field_eq_prints_native_case_values) :-
    run_surface_print_smoke("$1 == \"ERROR\" { print tolower($2), toupper($0) }\n",
        "INFO boot ok\nERROR Disk Full\nWARN cpu hot\nERROR network Down\n",
        "disk ERROR DISK FULL\nnetwork ERROR NETWORK DOWN\n").

test(surface_field_eq_prints_nr_with_output_separator) :-
    run_surface_print_smoke("BEGIN { OFS = \",\" } $1 == \"ERROR\" { print NR, $2, $3 }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "2,disk,full\n4,net,down\n").

test(surface_field_eq_counts_matching_records) :-
    run_surface_print_smoke("$1 == \"ERROR\" { count++ } END { print count }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "2\n").

test(surface_field_eq_counts_multiple_scalar_slots) :-
    run_surface_print_smoke("$1 == \"ERROR\" { errors++; matches++ } END { print errors, matches }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "2 2\n").

test(surface_multi_rule_counts_scalar_slots) :-
    run_surface_print_smoke("$1 == \"ERROR\" { errors++ } $1 == \"WARN\" { warnings++ } END { print errors, warnings }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "2 1\n").

test(surface_multi_rule_accumulates_overlapping_matches) :-
    run_surface_print_smoke("$1 == \"ERROR\" { hits++ } /^ERROR/ { hits++ } END { print hits }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "4\n").

test(surface_terminal_next_skips_remaining_scalar_rules) :-
    run_surface_print_smoke("$1 == \"DEBUG\" { skipped++; next } { total++ } END { print total, skipped }\n",
        "INFO boot ok\nDEBUG trace one\nERROR disk full\nDEBUG trace two\n",
        "2 2\n").

test(surface_terminal_next_skips_remaining_assoc_rules) :-
    run_surface_print_smoke("$1 == \"DEBUG\" { skipped[$2]++; next } { counts[$1]++ } END { print skipped[\"trace\"], counts[\"DEBUG\"], counts[\"ERROR\"] }\n",
        "INFO boot ok\nDEBUG trace one\nERROR disk full\nDEBUG trace two\n",
        "2 0 1\n").

test(surface_terminal_next_skips_remaining_mixed_rules) :-
    run_surface_print_smoke("$1 == \"DEBUG\" { skipped++; by_kind[$2]++; next } { total++; counts[$1]++ } END { print total, skipped, by_kind[\"trace\"], counts[\"DEBUG\"], counts[\"ERROR\"] }\n",
        "INFO boot ok\nDEBUG trace one\nERROR disk full\nDEBUG trace two\n",
        "2 2 2 0 1\n").

test(surface_terminal_next_only_skips_remaining_scalar_rules) :-
    run_surface_print_smoke("$1 == \"DEBUG\" { next } { total++ } END { print total }\n",
        "INFO boot ok\nDEBUG trace one\nERROR disk full\nDEBUG trace two\n",
        "2\n").

test(surface_terminal_next_only_skips_remaining_assoc_rules) :-
    run_surface_print_smoke("$1 == \"DEBUG\" { next } { counts[$1]++ } END { print counts[\"DEBUG\"], counts[\"ERROR\"] }\n",
        "INFO boot ok\nDEBUG trace one\nERROR disk full\nDEBUG trace two\n",
        "0 1\n").

test(surface_terminal_next_only_skips_remaining_mixed_rules) :-
    run_surface_print_smoke("$1 == \"DEBUG\" { next } { total++; counts[$1]++ } END { print total, counts[\"DEBUG\"], counts[\"ERROR\"] }\n",
        "INFO boot ok\nDEBUG trace one\nERROR disk full\nDEBUG trace two\n",
        "2 0 1\n").

test(surface_field_eq_scalar_add_assign_accumulates_constants_and_lengths) :-
    run_surface_print_smoke("$1 == \"ERROR\" { bytes += length($0); hits += 2 } END { print bytes, hits }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR network down\n",
        "33 4\n").

test(surface_always_scalar_add_assign_accumulates_constants) :-
    run_surface_print_smoke("{ total += 3 } END { print total }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "12\n").

test(surface_mixed_inc_and_add_assign_accumulates_same_slot) :-
    run_surface_print_smoke("$1 == \"ERROR\" { hits++; hits += 2 } END { print hits }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "6\n").

test(surface_scalar_end_prints_string_literals) :-
    run_surface_print_smoke("{ total++ } END { print \"total\", total }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "total 4\n").

test(surface_assoc_counts_requested_keys) :-
    run_surface_print_smoke("{ counts[$1]++ } END { print counts[\"ERROR\"], counts[\"WARN\"] }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "2 1\n").

test(surface_assoc_counts_multiple_arrays) :-
    run_surface_print_smoke("{ counts[$1]++; by_component[$2]++ } END { print counts[\"ERROR\"], by_component[\"disk\"], by_component[\"cpu\"] }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "2 1 1\n").

test(surface_guarded_assoc_counts_multiple_arrays) :-
    run_surface_print_smoke("$1 == \"ERROR\" { by_component[$2]++ } $1 == \"WARN\" { warnings[$2]++ } END { print by_component[\"disk\"], by_component[\"net\"], warnings[\"cpu\"], warnings[\"disk\"] }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\nWARN disk alert\n",
        "1 1 1 1\n").

test(surface_mixed_scalar_assoc_counts) :-
    run_surface_print_smoke("{ total++; counts[$1]++ } $1 == \"ERROR\" { errors++; by_component[$2]++ } END { print total, errors, counts[\"WARN\"], by_component[\"disk\"] }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "4 2 1 1\n").

test(surface_mixed_scalar_add_assign_and_assoc_counts) :-
    run_surface_print_smoke("{ bytes += length($0); counts[$1]++ } END { print bytes, counts[\"ERROR\"] }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "53 2\n").

test(surface_assoc_end_prints_string_literals) :-
    run_surface_print_smoke("{ counts[$1]++ } END { print \"errors\", counts[\"ERROR\"], \"warnings\", counts[\"WARN\"] }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "errors 2 warnings 1\n").

test(surface_mixed_end_prints_string_literals) :-
    run_surface_print_smoke("{ total++; counts[$1]++ } END { print \"total\", total, \"errors\", counts[\"ERROR\"] }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "total 4 errors 2\n").

test(surface_begin_prints_string_literals) :-
    run_surface_print_smoke("BEGIN { print \"kind\", \"count\" } { total++ } END { print \"total\", total }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "kind count\ntotal 4\n").

test(surface_begin_field_separator_drives_field_eq_and_assoc_keys) :-
    run_surface_print_smoke("BEGIN { FS = \":\" } $1 == \"ERROR\" { counts[$2]++ } END { print \"disk\", counts[\"disk\"], \"net\", counts[\"net\"] }\n",
        "ERROR:disk:full\nWARN:cpu:hot\nERROR:net:down\nERROR:disk:again\n",
        "disk 2 net 1\n").

test(surface_begin_field_separator_drives_selected_field_printing) :-
    run_surface_print_smoke("BEGIN { FS = \":\" } $1 == \"ERROR\" { print $2, $3 }\n",
        "ERROR:disk:full\nWARN:cpu:hot\nERROR:net:down\n",
        "disk full\nnet down\n").

test(surface_begin_field_separator_prints_header) :-
    run_surface_print_smoke("BEGIN { FS = \":\"; print \"kind\", \"count\" } { counts[$1]++ } END { print \"ERROR\", counts[\"ERROR\"] }\n",
        "ERROR:disk:full\nWARN:cpu:hot\nERROR:net:down\n",
        "kind count\nERROR 2\n").

test(surface_begin_output_separator_drives_selected_field_printing) :-
    run_surface_print_smoke("BEGIN { FS = \":\"; OFS = \",\" } $1 == \"ERROR\" { print $2, $3 }\n",
        "ERROR:disk:full\nWARN:cpu:hot\nERROR:net:down\n",
        "disk,full\nnet,down\n").

test(surface_begin_field_separator_drives_nf_printing) :-
    run_surface_print_smoke("BEGIN { FS = \":\"; OFS = \",\" } $1 == \"ERROR\" { print NR, NF, $2, $3 }\n",
        "ERROR:disk:full\nWARN:cpu:hot\nERROR:net:down:now\n",
        "1,3,disk,full\n3,4,net,down\n").

test(surface_begin_field_separator_drives_length_printing) :-
    run_surface_print_smoke("BEGIN { FS = \":\"; OFS = \",\" } $1 == \"ERROR\" { print length($0), length($2), $2 }\n",
        "ERROR:disk:full\nWARN:cpu:hot\nERROR:network:down:now\n",
        "15,4,disk\n22,7,network\n").

test(surface_begin_field_separator_drives_substr_printing) :-
    run_surface_print_smoke("BEGIN { FS = \":\"; OFS = \",\" } $1 == \"ERROR\" { print substr($2, 1, 3), substr($0, 7, 4) }\n",
        "ERROR:disk:full\nWARN:cpu:hot\nERROR:network:down:now\n",
        "dis,disk\nnet,netw\n").

test(surface_begin_field_separator_drives_index_printing) :-
    run_surface_print_smoke("BEGIN { FS = \":\"; OFS = \",\" } $1 == \"ERROR\" { print index($2, \"work\"), index($0, \"network\") }\n",
        "ERROR:disk:full\nWARN:cpu:hot\nERROR:network:down:now\n",
        "0,0\n4,7\n").

test(surface_begin_field_separator_drives_case_printing) :-
    run_surface_print_smoke("BEGIN { FS = \":\"; OFS = \",\" } $1 == \"ERROR\" { print tolower($2), toupper($3) }\n",
        "ERROR:Disk:Full\nWARN:cpu:hot\nERROR:network:Down\n",
        "disk,FULL\nnetwork,DOWN\n").

test(surface_begin_output_separator_drives_end_printing) :-
    run_surface_print_smoke("BEGIN { FS = \":\"; OFS = \",\" } $1 == \"ERROR\" { counts[$2]++ } END { print \"disk\", counts[\"disk\"], \"net\", counts[\"net\"] }\n",
        "ERROR:disk:full\nWARN:cpu:hot\nERROR:net:down\nERROR:disk:again\n",
        "disk,2,net,1\n").

test(surface_begin_output_separator_drives_begin_printing) :-
    run_surface_print_smoke("BEGIN { OFS = \",\"; print \"kind\", \"count\" } { total++ } END { print \"total\", total }\n",
        "INFO boot ok\nERROR disk full\n",
        "kind,count\ntotal,2\n").

test(surface_begin_output_separator_treats_rule_percent_as_data) :-
    run_surface_print_smoke("BEGIN { FS = \":\"; OFS = \"%\" } $1 == \"ERROR\" { print $2, $3 }\n",
        "ERROR:disk:full\nWARN:cpu:hot\nERROR:net:down\n",
        "disk%full\nnet%down\n").

test(surface_begin_output_separator_treats_begin_and_end_percent_as_data) :-
    run_surface_print_smoke("BEGIN { OFS = \"%\"; print \"kind\", \"count\" } { total++ } END { print \"total\", total }\n",
        "INFO boot ok\nERROR disk full\n",
        "kind%count\ntotal%2\n").

test(surface_assoc_counts_resize_runtime_table) :-
    findall(Line,
        ( between(0, 2050, Index), format(atom(Line), 'K~w payload\n', [Index]) ),
        Lines),
    atomic_list_concat(Lines, '', Input0),
    format(string(Input), '~wK0 duplicate~n', [Input0]),
    run_surface_print_smoke("{ counts[$1]++ } END { print counts[\"K0\"], counts[\"K2050\"], counts[\"MISSING\"] }\n",
        Input, "2 1 0\n").

test(surface_assoc_counts_long_record_first_field) :-
    plawk_long_payload_string(70000, Payload),
    format(string(Input), 'KEY ~w~n', [Payload]),
    run_surface_print_smoke("{ counts[$1]++ } END { print counts[\"KEY\"] }\n",
        Input, "1\n").

test(surface_assoc_counts_use_runtime_table) :-
    plawk_parse_string("{ counts[$1]++ } END { print counts[\"ERROR\"], counts[\"WARN\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(sub_atom(DriverIR, _, _, _, '@wam_assoc_i64_new')),
    assertion(sub_atom(DriverIR, _, _, _, '@wam_assoc_i64_inc')),
    assertion(sub_atom(DriverIR, _, _, _, '@wam_assoc_i64_get')),
    assertion(sub_atom(DriverIR, _, _, _, '@wam_assoc_i64_free')),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'assoc_check_0')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '%assoc_inc_slot_')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '%slot_0 = phi')),
    !.

test(surface_assoc_counts_multiple_arrays_use_distinct_runtime_tables) :-
    plawk_parse_string("{ counts[$1]++; by_component[$2]++ } END { print counts[\"ERROR\"], by_component[\"disk\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_assoc_table_0 = call %WamAssocI64Table* @wam_assoc_i64_new'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_assoc_table_1 = call %WamAssocI64Table* @wam_assoc_i64_new'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'assoc_rule_0_action_0:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'assoc_rule_0_action_1:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_assoc_i64_inc'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_assoc_i64_get'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '%plawk_assoc_table = call')),
    !.

test(surface_guarded_assoc_counts_use_native_rule_chain) :-
    plawk_parse_string("$1 == \"ERROR\" { by_component[$2]++ } $1 == \"WARN\" { warnings[$2]++ } END { print by_component[\"disk\"], warnings[\"cpu\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'assoc_rule_0_match:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'assoc_rule_0_apply:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'assoc_rule_1_match:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'assoc_rule_1_apply:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_5Fassoc_5Frule_5F0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_5Fassoc_5Frule_5F1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_eq_value'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_mixed_scalar_assoc_uses_native_state_and_tables) :-
    plawk_parse_string("{ total++; counts[$1]++ } $1 == \"ERROR\" { errors++; by_component[$2]++ } END { print total, errors, counts[\"WARN\"], by_component[\"disk\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'lowered_mixed:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%slot_0 = phi i64'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%next_slot_0 = phi i64'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_assoc_table_0 = call %WamAssocI64Table* @wam_assoc_i64_new'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_assoc_table_1 = call %WamAssocI64Table* @wam_assoc_i64_new'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_done:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_1_done:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'assoc_rule_0_action_0:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'assoc_rule_1_action_0:'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_scalar_add_assign_uses_native_state_and_field_length) :-
    plawk_parse_string("BEGIN { FS = \":\" } $1 == \"ERROR\" { bytes += length($2); hits += 2 } END { print bytes, hits }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'lowered_match:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%slot_0 = phi i64'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_slot_0_const = add i64 %slot_0, 0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_rule_0_slot_0_delta_0_len = call i64 @wam_atom_field_length_value(%Value %line, i64 2, i8 58)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_slot_1_const = add i64 %slot_1, 2'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%next_slot_0 = phi i64'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_terminal_next_uses_native_continue_phi) :-
    plawk_parse_string("$1 == \"DEBUG\" { skipped++; next } { total++ } END { print total, skipped }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_apply:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_1_match:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'br label %continue_loop'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_1_in_slot_0 = phi i64 [%slot_0, %rule_0_match]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%next_slot_0 = phi i64 [%rule_1_in_slot_0, %rule_1_match], [%rule_0_slot_0, %rule_0_apply], [%rule_1_slot_0, %rule_1_apply]'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_assoc_terminal_next_uses_native_rule_chain) :-
    plawk_parse_string("$1 == \"DEBUG\" { skipped[$2]++; next } { counts[$1]++ } END { print skipped[\"trace\"], counts[\"DEBUG\"], counts[\"ERROR\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'assoc_rule_0_apply:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'assoc_rule_1_match:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'br label %continue_loop'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_mixed_terminal_next_uses_native_continue_phi) :-
    plawk_parse_string("$1 == \"DEBUG\" { skipped++; by_kind[$2]++; next } { total++; counts[$1]++ } END { print total, skipped, by_kind[\"trace\"], counts[\"DEBUG\"], counts[\"ERROR\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_done:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_1_match:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'br label %continue_loop'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_1_in_slot_0 = phi i64 [%slot_0, %rule_0_match]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '[%rule_0_slot_0, %rule_0_done]'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_terminal_next_only_uses_native_continue_branch) :-
    plawk_parse_string("$1 == \"DEBUG\" { next } { total++ } END { print total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_apply:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_1_match:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'br label %continue_loop'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_1_in_slot_0 = phi i64 [%slot_0, %rule_0_match]'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_nonterminal_next_scalar_rejected_by_native_codegen, [fail]) :-
    plawk_parse_string("$1 == \"DEBUG\" { next; skipped++ } { total++ } END { print total, skipped }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', _DriverIR).

test(surface_nonterminal_next_assoc_rejected_by_native_codegen, [fail]) :-
    plawk_parse_string("$1 == \"DEBUG\" { next; skipped[$2]++ } { counts[$1]++ } END { print skipped[\"trace\"], counts[\"ERROR\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', _DriverIR).

test(surface_nonterminal_next_mixed_rejected_by_native_codegen, [fail]) :-
    plawk_parse_string("$1 == \"DEBUG\" { next; skipped++; by_kind[$2]++ } { total++; counts[$1]++ } END { print total, skipped, by_kind[\"trace\"], counts[\"ERROR\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', _DriverIR).

test(surface_end_string_literals_use_indexed_globals) :-
    plawk_parse_string("{ total++; counts[$1]++ } END { print \"total\", total, \"errors\", counts[\"ERROR\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_surface_print_string = private constant [3 x i8] c"%s\\00"'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_end_print_string_0 = private constant [6 x i8]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_end_print_string_2 = private constant [7 x i8]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_end_string_0 = call i32'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_end_string_2 = call i32'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_assoc_print_key_3 = private constant [6 x i8]'))),
    !.

test(surface_begin_string_literals_use_indexed_globals) :-
    plawk_parse_string("BEGIN { print \"kind\", \"count\" } { total++ } END { print \"total\", total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_begin_print_string_0 = private constant [5 x i8]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_begin_print_string_1 = private constant [6 x i8]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_begin_string_0 = call i32'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_begin_string_1 = call i32'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_begin_newline = call i32'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_end_print_string_0 = private constant [6 x i8]'))),
    !.

test(surface_begin_field_separator_uses_configured_delimiter) :-
    plawk_parse_string("BEGIN { FS = \":\" } $1 == \"ERROR\" { print $2, $3 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_eq_value(%Value %line, i64 1, i8* %plawk_5Fsurface_5Ffield_5Feq_ptr, i64 5, i8 58)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_slice_value(%Value %line, i64 2, i8 58)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_slice_value(%Value %line, i64 3, i8 58)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@wam_atom_field_slice_value(%Value %line, i64 2, i8 32)')),
    !.

test(surface_begin_output_separator_uses_configured_delimiter) :-
    plawk_parse_string("BEGIN { OFS = \",\" } { total++ } END { print \"total\", total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_end_separator_1 = call i32 @putchar(i32 44)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@.plawk_surface_print_space')),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'printf(i8* %end_space_fmt')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '%printed_end_space_1')),
    !.

test(surface_nr_print_uses_native_record_counter) :-
    plawk_parse_string("$1 == \"ERROR\" { print NR, $2 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_nr = phi i64 [0, %check_handle_value], [%current_nr, %continue_loop]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%current_nr = add i64 %plawk_nr, 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_nr_0 = call i32 (i8*, ...) @printf(i8* %nr_fmt_0, i64 %current_nr)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_surface_print_i64 = private constant [4 x i8] c"%ld\\00"'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_nf_print_uses_native_field_count) :-
    plawk_parse_string("BEGIN { FS = \":\" } $1 == \"ERROR\" { print NF, $2 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_nf_0 = call i64 @wam_atom_field_count_value(%Value %line, i8 58)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_nf_0 = call i32 (i8*, ...) @printf(i8* %nf_fmt_0, i64 %plawk_nf_0)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_surface_print_i64 = private constant [4 x i8] c"%ld\\00"'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_length_print_uses_native_field_length) :-
    plawk_parse_string("BEGIN { FS = \":\" } $1 == \"ERROR\" { print length($0), length($2), $2 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_length_0 = call i64 @wam_atom_field_length_value(%Value %line, i64 0, i8 58)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_length_1 = call i64 @wam_atom_field_length_value(%Value %line, i64 2, i8 58)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_length_1 = call i32 (i8*, ...) @printf(i8* %length_fmt_1, i64 %plawk_length_1)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_surface_print_i64 = private constant [4 x i8] c"%ld\\00"'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_default_space_fs_uses_native_whitespace_helpers) :-
    plawk_parse_string("$1 == \"ERROR\" { print NF, $2, length($2) }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_eq_value(%Value %line, i64 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_count_value(%Value %line, i8 32)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_slice_value(%Value %line, i64 2, i8 32)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_length_value(%Value %line, i64 2, i8 32)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_substr_print_uses_native_subslice) :-
    plawk_parse_string("BEGIN { FS = \":\" } $1 == \"ERROR\" { print substr($2, 1, 3), substr($0, 7, 4) }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_substr_0 = call %WamSlice @wam_atom_field_subslice_value(%Value %line, i64 2, i8 58, i64 1, i64 3)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_substr_1 = call %WamSlice @wam_atom_field_subslice_value(%Value %line, i64 0, i8 58, i64 7, i64 4)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_substr_0 = call i32 (i8*, ...) @printf(i8* %substr_fmt_0, i32 %plawk_substr_0_len, i8* %plawk_substr_0_ptr)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_surface_print_slice = private constant [5 x i8] c"%.*s\\00"'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_index_print_uses_native_field_index) :-
    plawk_parse_string("BEGIN { FS = \":\" } $1 == \"ERROR\" { print index($2, \"work\"), index($0, \"network\") }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_5Findex_5Fneedle_5F0 = private constant [5 x i8] c"work\\00"'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_5Findex_5Fneedle_5F1 = private constant [8 x i8] c"network\\00"'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_5Findex_5Fneedle_5F0_ptr = getelementptr [5 x i8], [5 x i8]* @.plawk_5Findex_5Fneedle_5F0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_index_0 = call i64 @wam_atom_field_index_value(%Value %line, i64 2, i8 58'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_index_1 = call i32 (i8*, ...) @printf(i8* %index_fmt_1, i64 %plawk_index_1)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_case_print_uses_native_case_helpers) :-
    plawk_parse_string("BEGIN { FS = \":\" } $1 == \"ERROR\" { print tolower($2), toupper($0) }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_tolower_0 = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 2, i8 58)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'call void @wam_print_ascii_lower_slice(i8* %plawk_tolower_0_ptr, i64 %plawk_tolower_0_len64)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_toupper_1_len64 = call i64 @strlen(i8* %line_s)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'call void @wam_print_ascii_upper_slice(i8* %line_s, i64 %plawk_toupper_1_len64)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

run_surface_print_smoke(Source, Input, ExpectedOutput) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_surface_prefix_print', Dir),
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
    directory_file_path(Dir, 'plawk_surface_prefix_print.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_surface_marker/0 ],
        [module_name('plawk_surface_prefix_print')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'plawk_surface_prefix_print_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> assertion(OutStr == ExpectedOutput)
    ;  format(user_error, "~n[plawk surface prefix print output]~n~w~n",
              [OutStr]),
       throw(plawk_surface_prefix_print_failed(Status))
    ),
    !.

plawk_long_payload_string(Length, String) :-
    length(Codes, Length),
    maplist(=(0'a), Codes),
    string_codes(String, Codes).

:- end_tests(plawk_surface_prefix_print).
