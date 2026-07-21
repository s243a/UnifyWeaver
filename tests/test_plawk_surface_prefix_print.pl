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

test(parses_literal_pattern_print_rule) :-
    plawk_parse_string("/disk/ { print $0 }\n", Program),
    assertion(Program == program([], [rule(contains("disk"), [print([field(0)])])], [])).

test(parses_field_eq_print_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { print $0 }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"), [print([field(0)])])], [])).

test(parses_field_numeric_cmp_print_rule) :-
    plawk_parse_string("$3 > 100 { print $1, $3 }\n", Program),
    assertion(Program == program([], [rule(field_cmp(3, gt, 100),
        [print([field(1), field(3)])])], [])).

test(parses_field_numeric_cmp_negative_rule) :-
    plawk_parse_string("$2 <= -5 { cold++ } END { print cold }\n", Program),
    assertion(Program == program([], [rule(field_cmp(2, le, -5), [inc(var(cold))])],
        [end([print([var(cold)])])])).

test(parses_field_numeric_eq_and_ne_rules) :-
    plawk_parse_string("$2 == 0 { zeros++ } $2 != 0 { nonzeros++ } END { print zeros, nonzeros }\n", Program),
    assertion(Program == program([],
        [rule(field_cmp(2, eq, 0), [inc(var(zeros))]),
         rule(field_cmp(2, ne, 0), [inc(var(nonzeros))])],
        [end([print([var(zeros), var(nonzeros)])])])).

test(parses_field_eq_print_fields_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { print $2, $3 }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"), [print([field(2), field(3)])])], [])).

test(parses_field_eq_printf_fields_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { printf \"%s=%s\\n\", $2, $3 }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [printf(string("%s=%s\n"), [field(2), field(3)])])], [])).

test(parses_field_eq_printf_literal_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { printf \"hit\\n\" }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [printf(string("hit\n"), [])])], [])).

test(parses_field_eq_printf_string_arg_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { printf \"%s:%s\\n\", \"kind\", $1 }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [printf(string("%s:%s\n"), [string("kind"), field(1)])])], [])).

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

test(parses_field_eq_print_int_field_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { print $3, int($3) }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [print([field(3), int(field(3))])])], [])).

test(parses_field_eq_print_int_add_field_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { print int($3) + 1 }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [print([add_i64(int(field(3)), int(1))])])], [])).

test(parses_field_eq_print_int_sub_field_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { print int($3) - 1 }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [print([sub_i64(int(field(3)), int(1))])])], [])).

test(parses_field_eq_print_i64_primary_binary_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { print NR - 1, NF + 1, length($0) - 3, index($2, \"sk\") + 1 }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [print([sub_i64(special('NR'), int(1)),
                add_i64(special('NF'), int(1)),
                sub_i64(length(field(0)), int(3)),
                add_i64(index(field(2), string("sk")), int(1))])])], [])).

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

test(parses_keyword_prefix_variable_names) :-
    plawk_parse_string("{ next_total++; break_count++ } END { print next_total, break_count }\n", Program),
    assertion(Program == program([], [rule(always,
        [inc(var(next_total)), inc(var(break_count))])],
        [end([print([var(next_total), var(break_count)])])])).

test(parses_terminal_break_scalar_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { hits++; break } { total++ } END { print hits, total }\n", Program),
    assertion(Program == program([],
        [rule(field_eq(1, "ERROR"), [inc(var(hits)), break]),
         rule(always, [inc(var(total))])],
        [end([print([var(hits), var(total)])])])).

test(parses_terminal_break_assoc_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { seen[$2]++; break } { counts[$1]++ } END { print seen[\"disk\"], counts[\"ERROR\"], counts[\"WARN\"] }\n", Program),
    assertion(Program == program([],
        [rule(field_eq(1, "ERROR"), [inc_assoc(var(seen), field(2)), break]),
         rule(always, [inc_assoc(var(counts), field(1))])],
        [end([print([
            assoc(var(seen), string("disk")),
            assoc(var(counts), string("ERROR")),
            assoc(var(counts), string("WARN"))
        ])])])).

test(parses_terminal_break_mixed_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { hits++; seen[$2]++; break } { total++; counts[$1]++ } END { print hits, total, seen[\"disk\"], counts[\"ERROR\"], counts[\"WARN\"] }\n", Program),
    assertion(Program == program([],
        [rule(field_eq(1, "ERROR"), [inc(var(hits)), inc_assoc(var(seen), field(2)), break]),
         rule(always, [inc(var(total)), inc_assoc(var(counts), field(1))])],
        [end([print([
            var(hits), var(total), assoc(var(seen), string("disk")),
            assoc(var(counts), string("ERROR")), assoc(var(counts), string("WARN"))
        ])])])).

test(parses_nonterminal_break_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { break; hits++ } { total++ } END { print hits, total }\n", Program),
    assertion(Program == program([],
        [rule(field_eq(1, "ERROR"), [break, inc(var(hits))]),
         rule(always, [inc(var(total))])],
        [end([print([var(hits), var(total)])])])).

test(parses_field_eq_scalar_add_assign_end_print_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { bytes += length($0); hits += 2 } END { print bytes, hits }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [add(var(bytes), length(field(0))), add(var(hits), int(2))])],
        [end([print([var(bytes), var(hits)])])])).

test(parses_field_numeric_scalar_add_assign_end_print_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { bytes += $3; last = $3 } END { print bytes, last }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [add(var(bytes), field(3)), set(var(last), field(3))])],
        [end([print([var(bytes), var(last)])])])).

test(parses_field_int_scalar_add_assign_end_print_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { bytes += int($3); last = int($3) } END { print bytes, last }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [add(var(bytes), int(field(3))), set(var(last), int(field(3)))])],
        [end([print([var(bytes), var(last)])])])).

test(parses_field_int_add_scalar_add_assign_end_print_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { bytes += int($3) + 1; last = int($3) + 1 } END { print bytes, last }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [add(var(bytes), add_i64(int(field(3)), int(1))),
         set(var(last), add_i64(int(field(3)), int(1)))])],
        [end([print([var(bytes), var(last)])])])).

test(parses_field_int_sub_scalar_add_assign_end_print_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { bytes += int($3) - 1; last = int($3) - 1 } END { print bytes, last }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [add(var(bytes), sub_i64(int(field(3)), int(1))),
         set(var(last), sub_i64(int(field(3)), int(1)))])],
        [end([print([var(bytes), var(last)])])])).

test(parses_i64_primary_binary_scalar_add_assign_end_print_rule) :-
    plawk_parse_string("{ adjusted += length($0) - 3; width = NF + 1; delta += int($3) + 1 } END { print adjusted, width, delta }\n", Program),
    assertion(Program == program([], [rule(always,
        [add(var(adjusted), sub_i64(length(field(0)), int(3))),
         set(var(width), add_i64(special('NF'), int(1))),
         add(var(delta), add_i64(int(field(3)), int(1)))])],
        [end([print([var(adjusted), var(width), var(delta)])])])).

test(parses_scalar_nr_add_assign_end_print_rule) :-
    plawk_parse_string("{ last = NR; total += NR; prev = NR - 1; next_total += NR + 1 } END { print last, total, prev, next_total }\n", Program),
    assertion(Program == program([], [rule(always,
        [set(var(last), special('NR')),
         add(var(total), special('NR')),
         set(var(prev), sub_i64(special('NR'), int(1))),
         add(var(next_total), add_i64(special('NR'), int(1)))])],
        [end([print([var(last), var(total), var(prev), var(next_total)])])])).

test(parses_scalar_nf_add_assign_end_print_rule) :-
    plawk_parse_string("{ width = NF; total += NF; adjusted = NF - 1; next_width += NF + 1 } END { print width, total, adjusted, next_width }\n", Program),
    assertion(Program == program([], [rule(always,
        [set(var(width), special('NF')),
         add(var(total), special('NF')),
         set(var(adjusted), sub_i64(special('NF'), int(1))),
         add(var(next_width), add_i64(special('NF'), int(1)))])],
        [end([print([var(width), var(total), var(adjusted), var(next_width)])])])).

test(parses_scalar_index_add_assign_end_print_rule) :-
    plawk_parse_string("{ pos = index($2, \"sk\"); total += index($0, \"disk\") } END { print pos, total }\n", Program),
    assertion(Program == program([], [rule(always,
        [set(var(pos), index(field(2), string("sk"))),
         add(var(total), index(field(0), string("disk")))])],
        [end([print([var(pos), var(total)])])])).

test(parses_scalar_index_binary_add_assign_end_print_rule) :-
    plawk_parse_string("{ pos = index($2, \"sk\") + 1; total += index($0, \"disk\") - 1 } END { print pos, total }\n", Program),
    assertion(Program == program([], [rule(always,
        [set(var(pos), add_i64(index(field(2), string("sk")), int(1))),
         add(var(total), sub_i64(index(field(0), string("disk")), int(1)))])],
        [end([print([var(pos), var(total)])])])).

test(parses_always_scalar_add_assign_end_print_rule) :-
    plawk_parse_string("{ total += 3 } END { print total }\n", Program),
    assertion(Program == program([], [rule(always, [add(var(total), int(3))])],
        [end([print([var(total)])])])).

test(parses_scalar_assignment_end_print_rule) :-
    plawk_parse_string("$1 == \"ERROR\" { last_len = length($0); hits++ } END { print hits, last_len }\n", Program),
    assertion(Program == program([], [rule(field_eq(1, "ERROR"),
        [set(var(last_len), length(field(0))), inc(var(hits))])],
        [end([print([var(hits), var(last_len)])])])).

test(parses_scalar_integer_assignment_end_print_rule) :-
    plawk_parse_string("{ current = 7; current += 2 } END { print current }\n", Program),
    assertion(Program == program([], [rule(always,
        [set(var(current), int(7)), add(var(current), int(2))])],
        [end([print([var(current)])])])).

test(parses_scalar_if_else_end_print_rule) :-
    plawk_parse_string("{ total++; if ($1 == \"ERROR\") { errors++; last_len = length($0) } else { non_errors++ } } END { print total, errors, non_errors, last_len }\n", Program),
    assertion(Program == program([], [rule(always,
        [inc(var(total)),
         if(field_eq(1, "ERROR"),
            [inc(var(errors)), set(var(last_len), length(field(0)))],
            [inc(var(non_errors))])])],
        [end([print([var(total), var(errors), var(non_errors), var(last_len)])])])).

test(parses_numeric_if_else_scalar_rule) :-
    plawk_parse_string("{ if ($3 >= 100) { big++ } else { small++ } } END { print big, small }\n", Program),
    assertion(Program == program([], [rule(always,
        [if(field_cmp(3, ge, 100),
            [inc(var(big))],
            [inc(var(small))])])],
        [end([print([var(big), var(small)])])])).

test(parses_scalar_if_else_without_keyword_space) :-
    plawk_parse_string("{ if($1 == \"ERROR\") { errors++ } else { non_errors++ } } END { print errors, non_errors }\n", Program),
    assertion(Program == program([], [rule(always,
        [if(field_eq(1, "ERROR"), [inc(var(errors))], [inc(var(non_errors))])])],
        [end([print([var(errors), var(non_errors)])])])).

test(parses_assoc_if_else_count_rule) :-
    plawk_parse_string("{ if ($1 == \"ERROR\") { counts[$2]++ } else { counts[$1]++ } } END { print counts[\"disk\"], counts[\"WARN\"] }\n", Program),
    assertion(Program == program([], [rule(always,
        [if(field_eq(1, "ERROR"),
            [inc_assoc(var(counts), field(2))],
            [inc_assoc(var(counts), field(1))])])],
        [end([print([assoc(var(counts), string("disk")),
                     assoc(var(counts), string("WARN"))])])])).

test(parses_branch_print_if_else_rule) :-
    plawk_parse_string("{ if ($1 == \"ERROR\") { print $2, $3 } else { counts[$1]++ } } END { print counts[\"WARN\"] }\n", Program),
    assertion(Program == program([], [rule(always,
        [if(field_eq(1, "ERROR"),
            [print([field(2), field(3)])],
            [inc_assoc(var(counts), field(1))])])],
        [end([print([assoc(var(counts), string("WARN"))])])])).

test(parses_branch_string_print_if_else_rule) :-
    plawk_parse_string("{ if ($1 == \"ERROR\") { print \"error\", NR, $2 } else { print \"ok\", $1 } } END { print \"done\" }\n", Program),
    assertion(Program == program([], [rule(always,
        [if(field_eq(1, "ERROR"),
            [print([string("error"), special('NR'), field(2)])],
            [print([string("ok"), field(1)])])])],
        [end([print([string("done")])])])).

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

test(surface_literal_pattern_prints_containing_records) :-
    run_surface_print_smoke("/disk/ { print $0 }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "ERROR disk full\n").

test(surface_literal_pattern_counts_containing_records) :-
    run_surface_print_smoke("/disk/ { hits++ } END { print hits }\n",
        "INFO boot ok\nERROR disk full\nWARN disk hot\nERROR net down\n",
        "2\n").

test(surface_field_eq_prints_matching_records) :-
    run_surface_print_smoke("$1 == \"ERROR\" { print $0 }\n",
        "INFO boot ok\nNOTERROR misleading\nWARN cpu hot\nERROR net down\n",
        "ERROR net down\n").

test(surface_field_eq_prints_selected_fields) :-
    run_surface_print_smoke("$1 == \"ERROR\" { print $2, $3 }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "disk full\nnet down\n").

test(surface_field_eq_printf_fields_prints_matching_records) :-
    run_surface_print_smoke("$1 == \"ERROR\" { printf \"%s=%s\\n\", $2, $3 }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "disk=full\nnet=down\n").

test(surface_field_eq_printf_has_no_implicit_newline) :-
    run_surface_print_smoke("$1 == \"ERROR\" { printf \"[%s]\", $2 }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "[disk][net]").

test(surface_printf_string_literal_arg) :-
    run_surface_print_smoke("{ printf \"%s:%s\\n\", \"kind\", $1 }\n",
        "INFO boot ok\nERROR disk full\n",
        "kind:INFO\nkind:ERROR\n").

test(surface_field_numeric_cmp_prints_matching_records) :-
    run_surface_print_smoke("$3 > 100 { print $1, $3 }\n",
        "disk used 95\ncpu used 101\nnet used 120\nbad used nope\nmem used -3\n",
        "cpu 101\nnet 120\n").

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

test(surface_field_eq_prints_native_int_values) :-
    run_surface_print_smoke("$1 == \"ERROR\" { print $3, int($3) }\n",
        "INFO boot 7\nERROR disk 10\nWARN cpu 20\nERROR net -3\nERROR bad nope\n",
        "10 10\n-3 -3\nnope 0\n").

test(surface_field_eq_prints_native_int_add_values) :-
    run_surface_print_smoke("$1 == \"ERROR\" { print int($3) + 1 }\n",
        "INFO boot 7\nERROR disk 10\nWARN cpu 20\nERROR net -3\nERROR bad nope\n",
        "11\n-2\n1\n").

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

test(surface_field_numeric_cmp_counts_matching_records) :-
    run_surface_print_smoke("$3 >= 100 { big++ } END { print big }\n",
        "disk used 95\ncpu used 100\nnet used 120\nbad used nope\nmem used -3\n",
        "2\n").

test(surface_field_numeric_cmp_handles_negative_values) :-
    run_surface_print_smoke("$2 < -5 { cold++ } END { print cold }\n",
        "a -6\nb -5\nc 0\nd -10\n",
        "2\n").

test(surface_field_numeric_eq_and_ne_counts) :-
    run_surface_print_smoke("$2 == 0 { zeros++ } $2 != 0 { nonzeros++ } END { print zeros, nonzeros }\n",
        "a 0\nb 1\nc nope\nd -1\n",
        "1 2\n").

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

test(surface_nonterminal_next_skips_dead_scalar_tail) :-
    run_surface_print_smoke("$1 == \"DEBUG\" { next; skipped++ } { total++ } END { print total, skipped }\n",
        "INFO boot ok\nDEBUG trace one\nERROR disk full\nDEBUG trace two\n",
        "2 0\n").

test(surface_terminal_next_skips_remaining_assoc_rules) :-
    run_surface_print_smoke("$1 == \"DEBUG\" { skipped[$2]++; next } { counts[$1]++ } END { print skipped[\"trace\"], counts[\"DEBUG\"], counts[\"ERROR\"] }\n",
        "INFO boot ok\nDEBUG trace one\nERROR disk full\nDEBUG trace two\n",
        "2 0 1\n").

test(surface_nonterminal_next_skips_dead_assoc_tail) :-
    run_surface_print_smoke("$1 == \"DEBUG\" { next; skipped[$2]++ } { counts[$1]++ } END { print skipped[\"trace\"], counts[\"DEBUG\"], counts[\"ERROR\"] }\n",
        "INFO boot ok\nDEBUG trace one\nERROR disk full\nDEBUG trace two\n",
        "0 0 1\n").

test(surface_terminal_next_skips_remaining_mixed_rules) :-
    run_surface_print_smoke("$1 == \"DEBUG\" { skipped++; by_kind[$2]++; next } { total++; counts[$1]++ } END { print total, skipped, by_kind[\"trace\"], counts[\"DEBUG\"], counts[\"ERROR\"] }\n",
        "INFO boot ok\nDEBUG trace one\nERROR disk full\nDEBUG trace two\n",
        "2 2 2 0 1\n").

test(surface_nonterminal_next_skips_dead_mixed_tail) :-
    run_surface_print_smoke("$1 == \"DEBUG\" { next; skipped++; by_kind[$2]++ } { total++; counts[$1]++ } END { print total, skipped, by_kind[\"trace\"], counts[\"DEBUG\"], counts[\"ERROR\"] }\n",
        "INFO boot ok\nDEBUG trace one\nERROR disk full\nDEBUG trace two\n",
        "2 0 0 0 1\n").

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

test(surface_terminal_break_stops_scalar_rule_chain_and_runs_end) :-
    run_surface_print_smoke("$1 == \"ERROR\" { hits++; break } { total++ } END { print hits, total }\n",
        "INFO boot ok\nWARN cpu hot\nERROR disk full\nERROR net down\n",
        "1 2\n").

test(surface_nonterminal_break_skips_dead_scalar_tail) :-
    run_surface_print_smoke("$1 == \"ERROR\" { break; hits++ } { total++ } END { print hits, total }\n",
        "INFO boot ok\nWARN cpu hot\nERROR disk full\nERROR net down\n",
        "0 2\n").

test(surface_terminal_break_stops_assoc_rule_chain_and_runs_end) :-
    run_surface_print_smoke("$1 == \"ERROR\" { seen[$2]++; break } { counts[$1]++ } END { print seen[\"disk\"], counts[\"ERROR\"], counts[\"WARN\"] }\n",
        "WARN cpu hot\nERROR disk full\nERROR net down\n",
        "1 0 1\n").

test(surface_nonterminal_break_skips_dead_assoc_tail) :-
    run_surface_print_smoke("$1 == \"ERROR\" { break; seen[$2]++ } { counts[$1]++ } END { print seen[\"disk\"], counts[\"ERROR\"], counts[\"WARN\"] }\n",
        "WARN cpu hot\nERROR disk full\nERROR net down\n",
        "0 0 1\n").

test(surface_terminal_break_stops_mixed_rule_chain_and_runs_end) :-
    run_surface_print_smoke("$1 == \"ERROR\" { hits++; seen[$2]++; break } { total++; counts[$1]++ } END { print hits, total, seen[\"disk\"], counts[\"ERROR\"], counts[\"WARN\"] }\n",
        "WARN cpu hot\nERROR disk full\nERROR net down\n",
        "1 1 1 0 1\n").

test(surface_nonterminal_break_skips_dead_mixed_tail) :-
    run_surface_print_smoke("$1 == \"ERROR\" { break; hits++; seen[$2]++ } { total++; counts[$1]++ } END { print hits, total, seen[\"disk\"], counts[\"ERROR\"], counts[\"WARN\"] }\n",
        "WARN cpu hot\nERROR disk full\nERROR net down\n",
        "0 1 0 0 1\n").

test(surface_terminal_break_as_last_rule_keeps_continue_phi_valid) :-
    run_surface_print_smoke("{ total++ } $1 == \"ERROR\" { hits++; break } END { print hits, total }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\n",
        "1 2\n").

test(surface_field_eq_scalar_add_assign_accumulates_constants_and_lengths) :-
    run_surface_print_smoke("$1 == \"ERROR\" { bytes += length($0); hits += 2 } END { print bytes, hits }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR network down\n",
        "33 4\n").

test(surface_field_numeric_scalar_add_assign_accumulates_values) :-
    run_surface_print_smoke("$1 == \"ERROR\" { bytes += $3; last = $3 } END { print bytes, last }\n",
        "INFO boot 7\nERROR disk 10\nWARN cpu 20\nERROR net -3\nERROR bad nope\n",
        "7 nope\n").

test(surface_begin_field_separator_drives_numeric_scalar_values) :-
    run_surface_print_smoke("BEGIN { FS = \":\" } $1 == \"ERROR\" { bytes += $3; last = $3 } END { print bytes, last }\n",
        "INFO:boot:7\nERROR:disk:10\nWARN:cpu:20\nERROR:net:-3\nERROR:bad:nope\n",
        "7 nope\n").

test(surface_begin_field_separator_drives_int_printing) :-
    run_surface_print_smoke("BEGIN { FS = \":\"; OFS = \",\" } $1 == \"ERROR\" { print $3, int($3) }\n",
        "INFO:boot:7\nERROR:disk:10\nWARN:cpu:20\nERROR:net:-3\nERROR:bad:nope\n",
        "10,10\n-3,-3\nnope,0\n").

test(surface_begin_field_separator_drives_int_add_printing) :-
    run_surface_print_smoke("BEGIN { FS = \":\"; OFS = \",\" } $1 == \"ERROR\" { print $3, int($3) + 1 }\n",
        "INFO:boot:7\nERROR:disk:10\nWARN:cpu:20\nERROR:net:-3\nERROR:bad:nope\n",
        "10,11\n-3,-2\nnope,1\n").

test(surface_begin_field_separator_drives_int_sub_printing) :-
    run_surface_print_smoke("BEGIN { FS = \":\"; OFS = \",\" } $1 == \"ERROR\" { print $3, int($3) - 1 }\n",
        "INFO:boot:7\nERROR:disk:10\nWARN:cpu:20\nERROR:net:-3\nERROR:bad:nope\n",
        "10,9\n-3,-4\nnope,-1\n").

test(surface_begin_field_separator_drives_i64_primary_binary_printing) :-
    run_surface_print_smoke("BEGIN { FS = \":\"; OFS = \",\" } $1 == \"ERROR\" { print NR - 1, NF + 1, length($0) - 3, int($3) + 1, index($2, \"work\") + 1 }\n",
        "ERROR:disk:10\nWARN:cpu:20\nERROR:network:-3\n",
        "0,4,10,11,1\n2,4,13,-2,5\n").

test(surface_field_int_add_scalar_accumulates_values) :-
    run_surface_print_smoke("$1 == \"ERROR\" { bytes += int($3) + 1; last = int($3) + 1 } END { print bytes, last }\n",
        "INFO boot 7\nERROR disk 10\nWARN cpu 20\nERROR net -3\nERROR bad nope\n",
        "10 1\n").

test(surface_field_int_sub_scalar_accumulates_values) :-
    run_surface_print_smoke("$1 == \"ERROR\" { bytes += int($3) - 1; last = int($3) - 1 } END { print bytes, last }\n",
        "INFO boot 7\nERROR disk 10\nWARN cpu 20\nERROR net -3\nERROR bad nope\n",
        "4 -1\n").

test(surface_i64_primary_binary_scalar_accumulates_values) :-
    run_surface_print_smoke("BEGIN { FS = \":\" } { adjusted += length($0) - 3; width = NF + 1; delta += int($3) + 1 } END { print adjusted, width, delta }\n",
        "ERROR:disk:10\nWARN:cpu:20\nERROR:network:-3\n",
        "31 4 30\n").

test(surface_scalar_nr_accumulates_values) :-
    run_surface_print_smoke("{ last = NR; total += NR; prev = NR - 1; next_total += NR + 1 } END { print last, total, prev, next_total }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\n",
        "3 6 2 9\n").

test(surface_scalar_nf_accumulates_values) :-
    run_surface_print_smoke("{ width = NF; total += NF; adjusted = NF - 1; next_width += NF + 1 } END { print width, total, adjusted, next_width }\n",
        "INFO boot ok\nERROR disk full now\nWARN cpu\n",
        "2 9 1 12\n").

test(surface_scalar_index_accumulates_values) :-
    run_surface_print_smoke("{ pos = index($2, \"sk\"); total += index($0, \"disk\") } END { print pos, total }\n",
        "INFO boot ok\nERROR disk full\nWARN disk issue\n",
        "3 13\n").

test(surface_scalar_index_binary_accumulates_values) :-
    run_surface_print_smoke("{ pos = index($2, \"sk\") + 1; total += index($0, \"disk\") - 1 } END { print pos, total }\n",
        "INFO boot ok\nERROR disk full\nWARN disk issue\n",
        "4 10\n").

test(surface_always_scalar_add_assign_accumulates_constants) :-
    run_surface_print_smoke("{ total += 3 } END { print total }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "12\n").

test(surface_mixed_inc_and_add_assign_accumulates_same_slot) :-
    run_surface_print_smoke("$1 == \"ERROR\" { hits++; hits += 2 } END { print hits }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "6\n").

test(surface_scalar_assignment_tracks_last_matching_record) :-
    run_surface_print_smoke("$1 == \"ERROR\" { last_len = length($0); hits++ } END { print hits, last_len }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR network down\n",
        "2 18\n").

test(surface_scalar_assignment_preserves_source_order) :-
    run_surface_print_smoke("{ current = 7; current++; current += 2 } END { print current }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "10\n").

test(surface_scalar_assignment_overwrites_prior_updates) :-
    run_surface_print_smoke("{ current++; current = 7 } END { print current }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "7\n").

test(surface_mixed_scalar_assignment_and_assoc_counts) :-
    run_surface_print_smoke("{ last_len = length($0); counts[$1]++ } END { print last_len, counts[\"ERROR\"] }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "14 2\n").

test(surface_scalar_if_else_updates_native_slots) :-
    run_surface_print_smoke("{ total++; if ($1 == \"ERROR\") { errors++; last_len = length($0) } else { non_errors++ } } END { print total, errors, non_errors, last_len }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR network down\n",
        "4 2 2 18\n").

test(surface_scalar_if_else_preserves_surrounding_update_order) :-
    run_surface_print_smoke("{ state = 1; if ($1 == \"ERROR\") { state += 10 } else { state += 100 }; state++ } END { print state }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\n",
        "102\n").

test(surface_mixed_scalar_if_else_and_assoc_counts) :-
    run_surface_print_smoke("{ if ($1 == \"ERROR\") { errors++ } else { non_errors++ }; counts[$1]++ } END { print errors, non_errors, counts[\"ERROR\"], counts[\"WARN\"] }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n",
        "2 2 2 1\n").

test(surface_assoc_if_else_updates_native_tables) :-
    run_surface_print_smoke("{ if ($1 == \"ERROR\") { counts[$2]++ } else { counts[$1]++ } } END { print counts[\"disk\"], counts[\"WARN\"] }\n",
        "ERROR disk full\nWARN cpu hot\nERROR net down\n",
        "1 1\n").

test(surface_mixed_if_else_updates_scalar_slots_and_native_tables) :-
    run_surface_print_smoke("{ total++; if ($1 == \"ERROR\") { errors++; counts[$2]++ } else { counts[$1]++ } } END { print total, errors, counts[\"disk\"], counts[\"WARN\"] }\n",
        "ERROR disk full\nWARN cpu hot\nERROR net down\n",
        "3 2 1 1\n").

test(surface_assoc_if_else_prints_selected_branch) :-
    run_surface_print_smoke("{ if ($1 == \"ERROR\") { print $2, $3 } else { counts[$1]++ } } END { print counts[\"WARN\"] }\n",
        "ERROR disk full\nWARN cpu hot\nERROR net down\n",
        "disk full\nnet down\n1\n").

test(surface_scalar_if_else_prints_selected_branch) :-
    run_surface_print_smoke("{ total++; if ($1 == \"ERROR\") { print NF, $2 } else { total++ } } END { print total }\n",
        "ERROR disk full\nWARN cpu hot\nERROR net down\n",
        "3 disk\n3 net\n4\n").

test(surface_scalar_if_else_prints_branch_nr) :-
    run_surface_print_smoke("{ total++; if ($1 == \"ERROR\") { print NR, $2 } else { total++ } } END { print total }\n",
        "ERROR disk full\nWARN cpu hot\nERROR net down\n",
        "1 disk\n3 net\n4\n").

test(surface_if_else_branch_prints_string_literals) :-
    run_surface_print_smoke("{ if ($1 == \"ERROR\") { print \"error\", NR, $2 } else { print \"ok\", $1 } } END { print \"done\" }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\n",
        "ok INFO\nerror 2 disk\nok WARN\ndone\n").

test(surface_scalar_if_else_branch_next_skips_later_actions) :-
    run_surface_print_smoke("{ if ($1 == \"DEBUG\") { skipped++; next } else { seen++ }; total++ } END { print total, seen, skipped }\n",
        "INFO boot ok\nDEBUG trace skip\nERROR disk full\nDEBUG trace drop\n",
        "2 2 2\n").

test(surface_scalar_if_else_branch_next_skips_dead_tail_and_later_actions) :-
    run_surface_print_smoke("{ if ($1 == \"DEBUG\") { next; skipped++ } else { seen++ }; total++ } END { print total, seen, skipped }\n",
        "INFO boot ok\nDEBUG trace skip\nERROR disk full\nDEBUG trace drop\n",
        "2 2 0\n").

test(surface_mixed_if_else_branch_next_skips_later_rules) :-
    run_surface_print_smoke("{ if ($1 == \"DEBUG\") { skipped++; by_kind[$2]++; next } else { seen++ } } { total++; counts[$1]++ } END { print total, seen, skipped, by_kind[\"trace\"], counts[\"DEBUG\"], counts[\"ERROR\"] }\n",
        "INFO boot ok\nDEBUG trace skip\nERROR disk full\nDEBUG trace drop\n",
        "2 2 2 2 0 1\n").

test(surface_scalar_if_else_branch_break_stops_stream) :-
    run_surface_print_smoke("{ if ($1 == \"ERROR\") { hits++; break } else { total++ } } END { print hits, total }\n",
        "INFO boot ok\nWARN cpu hot\nERROR disk full\nINFO after break\n",
        "1 2\n").

test(surface_scalar_if_else_branch_break_skips_dead_tail_and_stops_stream) :-
    run_surface_print_smoke("{ if ($1 == \"ERROR\") { break; hits++ } else { total++ } } END { print hits, total }\n",
        "INFO boot ok\nWARN cpu hot\nERROR disk full\nINFO after break\n",
        "0 2\n").

test(surface_mixed_if_else_branch_break_stops_stream) :-
    run_surface_print_smoke("{ if ($1 == \"ERROR\") { hits++; seen[$2]++; break } else { total++; counts[$1]++ } } END { print hits, total, seen[\"disk\"], counts[\"INFO\"], counts[\"WARN\"] }\n",
        "INFO boot ok\nWARN cpu hot\nERROR disk full\nINFO after break\n",
        "1 2 1 1 1\n").

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

test(surface_begin_field_separator_drives_printf_fields) :-
    run_surface_print_smoke("BEGIN { FS = \":\" } $1 == \"ERROR\" { printf \"%s=%s\\n\", $2, $3 }\n",
        "ERROR:disk:full\nWARN:cpu:hot\nERROR:net:down\n",
        "disk=full\nnet=down\n").

test(surface_printf_i64_and_percent_literals) :-
    run_surface_print_smoke("{ printf \"row=%d %% %s\\n\", NR, $1 } END { print \"done\" }\n",
        "INFO boot ok\nERROR disk full\n",
        "row=1 % INFO\nrow=2 % ERROR\ndone\n").

test(surface_if_else_branch_printf_prints_selected_branch) :-
    run_surface_print_smoke("{ if ($1 == \"ERROR\") { printf \"E:%s\\n\", $2 } else { printf \"O:%s\\n\", $1 } } END { print \"done\" }\n",
        "INFO boot ok\nERROR disk full\nWARN cpu hot\n",
        "O:INFO\nE:disk\nO:WARN\ndone\n").

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

test(surface_literal_pattern_uses_native_index_guard) :-
    plawk_parse_string("/disk/ { print $0 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_5Fsurface_5Fcontains = private constant [5 x i8] c"disk\\00"'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_surface_contains_contains_index = call i64 @wam_atom_field_index_value(%Value %line, i64 0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%is_match = icmp sgt i64 %plawk_surface_contains_contains_index, 0'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

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
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_body_assoc_1:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_1_body_assoc_1:'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_scalar_add_assign_uses_native_state_and_field_length) :-
    plawk_parse_string("BEGIN { FS = \":\" } $1 == \"ERROR\" { bytes += length($2); hits += 2 } END { print bytes, hits }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'lowered_match:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%slot_0 = phi i64'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_slot_0_op_0_len = call i64 @wam_atom_field_length_value(%Value %line, i64 2, i8 58)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_slot_0_op_0 = add i64 %slot_0, %rule_0_body_slot_0_op_0_len'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_slot_1_op_1 = add i64 %slot_1, 2'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%next_slot_0 = phi i64'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_scalar_add_assign_uses_native_field_i64_parse) :-
    plawk_parse_string("BEGIN { FS = \":\" } $1 == \"ERROR\" { bytes += $3; last = $3 } END { print bytes, last }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_i64_value(%Value %line, i64 3, i8 58)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_field_i64_value = extractvalue %WamI64Parse'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_field_i64_ok = extractvalue %WamI64Parse'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'select i1 %rule_0_body_slot_0_op_0_field_i64_ok'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_slot_0_op_0 = add i64 %slot_0, %rule_0_body_slot_0_op_0_field_i64_value_or_default'))),
    % `last = $3` is a plain copy of a field, so strnum copy-propagation keeps it
    % a string (strnum) scalar -- it interns the field bytes rather than parsing
    % them as i64. That is what lets a non-numeric `$3` ("nope") round-trip to the
    % END print as text (awk semantics), instead of the old numeric `add i64 0`
    % lowering that forced it to 0.
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_slot_1_op_1_snum_id = call i64 @wam_intern_atom'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '%rule_0_body_slot_1_op_1 = add i64 0, %rule_0_body_slot_1_op_1_field_i64_value_or_default')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_printf_uses_native_vararg_call) :-
    plawk_parse_string("{ printf \"row=%d %% %s\\n\", NR, $1 } END { print \"done\" }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.rule_0_body_printf_0_fmt = private constant [17 x i8] c"row=%ld %% %.*s\\0A\\00"'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_printf_0_field_1 = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 1, i8 32)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_printf_0_printed = call i32 (i8*, ...) @printf(i8* %rule_0_body_printf_0_fmt_ptr, i64 %current_nr, i32 %rule_0_body_printf_0_field_1_len, i8* %rule_0_body_printf_0_field_1_ptr)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_branch_printf_uses_prefixed_native_vararg_call) :-
    plawk_parse_string("{ if ($1 == \"ERROR\") { printf \"E:%s\\n\", $2 } else { printf \"O:%s\\n\", $1 } } END { print \"done\" }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.rule_0_body_if_0_then_printf_0_fmt = private constant [8 x i8] c"E:%.*s\\0A\\00"'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.rule_0_body_if_0_else_printf_0_fmt = private constant [8 x i8] c"O:%.*s\\0A\\00"'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_if_0_then_printf_0_printed = call i32 (i8*, ...) @printf'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_if_0_else_printf_0_printed = call i32 (i8*, ...) @printf'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_int_print_uses_native_field_i64_parse) :-
    plawk_parse_string("BEGIN { FS = \":\" } $1 == \"ERROR\" { print $3, int($3) }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_i64_value(%Value %line, i64 3, i8 58)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_int_1_value = extractvalue %WamI64Parse %plawk_int_1, 0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_int_1_ok = extractvalue %WamI64Parse %plawk_int_1, 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_int_1_value_or_default = select i1 %plawk_int_1_ok, i64 %plawk_int_1_value, i64 0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_int_1 = call i32 (i8*, ...) @printf(i8* %int_fmt_1, i64 %plawk_int_1_value_or_default)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_int_add_print_uses_shared_i64_add_lowering) :-
    plawk_parse_string("BEGIN { FS = \":\" } $1 == \"ERROR\" { print int($3) + 1 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_i64_value(%Value %line, i64 3, i8 58)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_int_add_0_lhs_value_or_default = select i1 %plawk_int_add_0_lhs_ok, i64 %plawk_int_add_0_lhs_value, i64 0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_int_add_0 = add i64 %plawk_int_add_0_lhs_value_or_default, 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_int_add_0 = call i32 (i8*, ...) @printf(i8* %int_add_fmt_0, i64 %plawk_int_add_0)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_int_sub_print_uses_shared_i64_sub_lowering) :-
    plawk_parse_string("BEGIN { FS = \":\" } $1 == \"ERROR\" { print int($3) - 1 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_i64_value(%Value %line, i64 3, i8 58)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_int_sub_0_lhs_value_or_default = select i1 %plawk_int_sub_0_lhs_ok, i64 %plawk_int_sub_0_lhs_value, i64 0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_int_sub_0 = sub i64 %plawk_int_sub_0_lhs_value_or_default, 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_int_sub_0 = call i32 (i8*, ...) @printf(i8* %int_sub_fmt_0, i64 %plawk_int_sub_0)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_i64_primary_binary_print_uses_shared_lowering) :-
    plawk_parse_string("BEGIN { FS = \":\" } $1 == \"ERROR\" { print NR - 1, NF + 1, length($0) - 3, int($3) + 1, index($2, \"work\") + 1 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_int_sub_0 = sub i64 %current_nr, 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_int_add_1_lhs = call i64 @wam_atom_field_count_value(%Value %line, i8 58)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_int_add_1 = add i64 %plawk_int_add_1_lhs, 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_int_sub_2_lhs = call i64 @wam_atom_field_length_value(%Value %line, i64 0, i8 58)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_int_sub_2 = sub i64 %plawk_int_sub_2_lhs, 3'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_int_add_3_lhs_value_or_default = select i1 %plawk_int_add_3_lhs_ok, i64 %plawk_int_add_3_lhs_value, i64 0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_int_add_3 = add i64 %plawk_int_add_3_lhs_value_or_default, 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_5Fint_5Fadd_5F4_5Flhs = private constant [5 x i8] c"work\\00"'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_int_add_4_lhs = call i64 @wam_atom_field_index_value(%Value %line, i64 2, i8 58'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_int_add_4 = add i64 %plawk_int_add_4_lhs, 1'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_scalar_nr_uses_native_record_counter) :-
    plawk_parse_string("{ last = NR; total += NR; prev = NR - 1; next_total += NR + 1 } END { print last, total, prev, next_total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_nr = phi i64 [0, %check_handle_value], [%current_nr, %continue_loop]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%current_nr = add i64 %plawk_nr, 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '= add i64 0, %current_nr'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '= sub i64 %current_nr, 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '= add i64 %current_nr, 1'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_scalar_nf_uses_native_field_count_primary) :-
    plawk_parse_string("BEGIN { FS = \":\" } { width = NF; total += NF; adjusted = NF - 1; next_width += NF + 1 } END { print width, total, adjusted, next_width }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '= call i64 @wam_atom_field_count_value(%Value %line, i8 58)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '= add i64 0, %rule_0_body_slot_3_op_0_i64_primary'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '= add i64 %slot_2, %rule_0_body_slot_2_op_1_i64_primary'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '= sub i64 %rule_0_body_slot_0_op_2_i64_sub_lhs, 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '= add i64 %rule_0_body_slot_1_op_3_i64_add_lhs, 1'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_scalar_index_uses_native_field_index_primary) :-
    plawk_parse_string("BEGIN { FS = \":\" } { pos = index($2, \"work\"); total += index($0, \"network\") } END { print pos, total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.rule_5F0_5Fbody_5Fslot_5F0_5Fop_5F0_5Fi64_5Fprimary = private constant [5 x i8] c"work\\00"'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.rule_5F0_5Fbody_5Fslot_5F1_5Fop_5F1_5Fi64_5Fprimary = private constant [8 x i8] c"network\\00"'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_slot_0_op_0_i64_primary = call i64 @wam_atom_field_index_value(%Value %line, i64 2, i8 58'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_slot_1_op_1_i64_primary = call i64 @wam_atom_field_index_value(%Value %line, i64 0, i8 58'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '= add i64 0, %rule_0_body_slot_0_op_0_i64_primary'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '= add i64 %slot_1, %rule_0_body_slot_1_op_1_i64_primary'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_scalar_index_binary_uses_native_field_index_primary) :-
    plawk_parse_string("BEGIN { FS = \":\" } { pos = index($2, \"work\") + 1; total += index($0, \"network\") - 1 } END { print pos, total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.rule_5F0_5Fbody_5Fslot_5F0_5Fop_5F0_5Fi64_5Fadd_5Flhs = private constant [5 x i8] c"work\\00"'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.rule_5F0_5Fbody_5Fslot_5F1_5Fop_5F1_5Fi64_5Fsub_5Flhs = private constant [8 x i8] c"network\\00"'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_slot_0_op_0_i64_add_lhs = call i64 @wam_atom_field_index_value(%Value %line, i64 2, i8 58'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_slot_0_op_0_i64_add = add i64 %rule_0_body_slot_0_op_0_i64_add_lhs, 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_slot_1_op_1_i64_sub_lhs = call i64 @wam_atom_field_index_value(%Value %line, i64 0, i8 58'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_slot_1_op_1_i64_sub = sub i64 %rule_0_body_slot_1_op_1_i64_sub_lhs, 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '= add i64 0, %rule_0_body_slot_0_op_0_i64_add'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '= add i64 %slot_1, %rule_0_body_slot_1_op_1_i64_sub'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_scalar_assignment_uses_ordered_native_state_ops) :-
    plawk_parse_string("$1 == \"ERROR\" { last_len = length($0); last_len += 2 } END { print last_len }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_slot_0_op_0_len = call i64 @wam_atom_field_length_value(%Value %line, i64 0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_slot_0_op_0 = add i64 0, %rule_0_body_slot_0_op_0_len'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_slot_0_op_1 = add i64 %rule_0_body_slot_0_op_0, 2'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'i64 %final_slot_0'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_scalar_if_else_uses_native_branch_phi) :-
    plawk_parse_string("{ if ($1 == \"ERROR\") { errors++; last_len = length($0) } else { non_errors++ } } END { print errors, non_errors, last_len }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_body_if_0_then:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_body_if_0_else:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_body_if_0_done:'))),
    findall(guard, sub_atom(DriverIR, _, _, _, '@wam_atom_field_eq_value'), Guards),
    assertion(Guards == [guard]),
    assertion(once(sub_atom(DriverIR, _, _, _, '= phi i64'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_if_else_branch_next_uses_native_continue_phi) :-
    plawk_parse_string("{ if ($1 == \"ERROR\") { next } else { total++ } } END { print total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_body_if_0_then:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'br label %continue_loop'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%next_slot_0 = phi i64 [%slot_0, %rule_0_match], [%rule_0_slot_0, %rule_0_done], [%slot_0, %rule_0_body_if_0_then]'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_if_else_assoc_branch_updates_use_native_tables) :-
    plawk_parse_string("{ if ($1 == \"ERROR\") { counts[$1]++ } else { counts[$2]++ } } END { print counts[\"ERROR\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_body_if_0_then_assoc_0:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_body_if_0_else_assoc_0:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_body_if_0_done:'))),
    findall(inc, sub_atom(DriverIR, _, _, _, '@wam_assoc_i64_inc'), Increments),
    assertion(Increments == [inc, inc]),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_if_else_branch_print_uses_prefixed_native_prints) :-
    plawk_parse_string("{ if ($1 == \"ERROR\") { print $0 } else { counts[$1]++ } } END { print counts[\"ERROR\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_body_if_0_then:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_if_0_then_print_0_line_fmt = getelementptr'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_if_0_then_print_0_printed_line = call i32'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_body_if_0_else_assoc_0:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_body_if_0_done:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_surface_print_line = private constant [4 x i8] c"%s\\0A\\00"'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_if_else_branch_nr_print_uses_native_record_counter) :-
    plawk_parse_string("{ total++; if ($1 == \"ERROR\") { print NR, $0 } else { total++ } } END { print total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%plawk_nr = phi i64 [0, %check_handle_value], [%current_nr, %continue_loop]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%current_nr = add i64 %plawk_nr, 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_if_1_then_print_0_nr_0_fmt_0 = getelementptr'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_rule_0_body_if_1_then_print_0_nr_0_0 = call i32 (i8*, ...) @printf(i8* %rule_0_body_if_1_then_print_0_nr_0_fmt_0, i64 %current_nr)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_if_else_branch_string_print_uses_prefixed_globals) :-
    plawk_parse_string("{ if ($1 == \"ERROR\") { print \"error\", $2 } else { print \"ok\", $1 } } END { print \"done\" }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.plawk_surface_print_string = private constant [3 x i8] c"%s\\00"'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.rule_0_body_if_0_then_print_0_string_0 = private constant [6 x i8]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.rule_0_body_if_0_else_print_0_string_0 = private constant [3 x i8]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_if_0_then_print_0_string_0_ptr = getelementptr [6 x i8], [6 x i8]* @.rule_0_body_if_0_then_print_0_string_0'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_rule_0_body_if_0_then_print_0_string_0_0 = call i32 (i8*, ...) @printf'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_if_else_branch_print_uses_shared_prefixed_expr_lowering) :-
    plawk_parse_string("{ if ($1 == \"ERROR\") { print length($2), substr($2, 1, 2), index($2, \"is\") } else { print NF } } END { print \"done\" }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_if_0_then_print_0_length_0 = call i64 @wam_atom_field_length_value(%Value %line, i64 2, i8 32)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_if_0_then_print_0_substr_1 = call %WamSlice @wam_atom_field_subslice_value(%Value %line, i64 2, i8 32, i64 1, i64 2)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@.rule_5F0_5Fbody_5Fif_5F0_5Fthen_5Fprint_5F0_5Findex_5Fneedle_5F2 = private constant [3 x i8] c"is\\00"'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_if_0_then_print_0_index_2 = call i64 @wam_atom_field_index_value(%Value %line, i64 2, i8 32'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_if_0_else_print_0_nf_0 = call i64 @wam_atom_field_count_value(%Value %line, i8 32)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_if_else_branch_print_uses_shared_prefixed_i64_binary_lowering) :-
    plawk_parse_string("{ if ($1 == \"ERROR\") { print int($3) - 1 } else { print int($3) + 1 } } END { print \"done\" }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_if_0_then_print_0_int_sub_0_lhs = call %WamI64Parse @wam_atom_field_i64_value(%Value %line, i64 3, i8 32)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_if_0_then_print_0_int_sub_0 = sub i64 %rule_0_body_if_0_then_print_0_int_sub_0_lhs_value_or_default, 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_rule_0_body_if_0_then_print_0_int_sub_0_0 = call i32 (i8*, ...) @printf(i8* %rule_0_body_if_0_then_print_0_int_sub_0_fmt_0, i64 %rule_0_body_if_0_then_print_0_int_sub_0)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_if_0_else_print_0_int_add_0_lhs = call %WamI64Parse @wam_atom_field_i64_value(%Value %line, i64 3, i8 32)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_0_body_if_0_else_print_0_int_add_0 = add i64 %rule_0_body_if_0_else_print_0_int_add_0_lhs_value_or_default, 1'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_rule_0_body_if_0_else_print_0_int_add_0_0 = call i32 (i8*, ...) @printf(i8* %rule_0_body_if_0_else_print_0_int_add_0_fmt_0, i64 %rule_0_body_if_0_else_print_0_int_add_0)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_terminal_next_uses_native_continue_phi) :-
    plawk_parse_string("$1 == \"DEBUG\" { skipped++; next } { total++ } END { print total, skipped }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_apply:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_1_match:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'br label %continue_loop'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%rule_1_in_slot_0 = phi i64 [%slot_0, %rule_0_match]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%next_slot_0 = phi i64 [%rule_1_in_slot_0, %rule_1_match], [%rule_0_slot_0, %rule_0_done], [%rule_1_slot_0, %rule_1_done]'))),
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

test(surface_nonterminal_next_scalar_uses_native_continue_branch) :-
    plawk_parse_string("$1 == \"DEBUG\" { next; skipped++ } { total++ } END { print total, skipped }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_apply:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '  br label %continue_loop'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'rule_0_slot_1_op_0')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_nonterminal_next_assoc_uses_native_continue_branch) :-
    plawk_parse_string("$1 == \"DEBUG\" { next; skipped[$2]++ } { counts[$1]++ } END { print skipped[\"trace\"], counts[\"ERROR\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'assoc_rule_0_apply:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '  br label %continue_loop'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'assoc_rule_0_action_0')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_nonterminal_next_mixed_uses_native_continue_branch) :-
    plawk_parse_string("$1 == \"DEBUG\" { next; skipped++; by_kind[$2]++ } { total++; counts[$1]++ } END { print total, skipped, by_kind[\"trace\"], counts[\"ERROR\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_apply:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '  br label %continue_loop'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'rule_0_slot_1_op_0')),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'rule_0_assoc_0')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_branch_nonterminal_next_uses_native_continue_branch) :-
    plawk_parse_string("{ if ($1 == \"DEBUG\") { next; skipped++ } else { total++ } } END { print total, skipped }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_body_if_0_then:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '  br label %continue_loop'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'then_slot_1_op_0')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_if_else_branch_break_uses_native_close_path) :-
    plawk_parse_string("{ if ($1 == \"ERROR\") { hits++; break } else { total++ } } END { print hits, total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_body_if_0_then:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'br label %break_close_stream'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%break_slot_0 = phi i64 [%rule_0_body_if_0_then_slot_0_op_0, %rule_0_body_if_0_then]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%final_slot_0 = phi i64 [%slot_0, %close_stream], [%break_slot_0, %break_close_stream]'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_branch_nonterminal_break_uses_native_close_path) :-
    plawk_parse_string("{ if ($1 == \"ERROR\") { break; hits++ } else { total++ } } END { print hits, total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'rule_0_body_if_0_then:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'br label %break_close_stream'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'then_slot_0_op_0')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_terminal_break_uses_close_path_and_final_state_phi) :-
    plawk_parse_string("$1 == \"ERROR\" { hits++; break } { total++ } END { print hits, total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'break_close_stream:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%break_slot_0 = phi i64 [%rule_0_slot_0, %rule_0_done]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%final_slot_0 = phi i64 [%slot_0, %close_stream], [%break_slot_0, %break_close_stream]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'i64 %final_slot_0'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_assoc_terminal_break_uses_native_close_path) :-
    plawk_parse_string("$1 == \"ERROR\" { seen[$2]++; break } { counts[$1]++ } END { print seen[\"disk\"], counts[\"ERROR\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'break_close_stream:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'assoc_rule_0_apply:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'br label %break_close_stream'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_mixed_terminal_break_uses_close_path_and_final_state_phi) :-
    plawk_parse_string("$1 == \"ERROR\" { hits++; seen[$2]++; break } { total++; counts[$1]++ } END { print hits, total, seen[\"disk\"], counts[\"ERROR\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'break_close_stream:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%break_slot_0 = phi i64 [%rule_0_slot_0, %rule_0_done]'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%final_slot_0 = phi i64 [%slot_0, %close_stream], [%break_slot_0, %break_close_stream]'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_nonterminal_break_scalar_uses_native_close_path) :-
    plawk_parse_string("$1 == \"ERROR\" { break; hits++ } { total++ } END { print hits, total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'break_close_stream:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%break_slot_0 = phi i64 [%rule_0_slot_0, %rule_0_done]'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'rule_0_slot_0_op_0')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_nonterminal_break_assoc_uses_native_close_path) :-
    plawk_parse_string("$1 == \"ERROR\" { break; seen[$2]++ } { counts[$1]++ } END { print seen[\"disk\"], counts[\"ERROR\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'break_close_stream:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'assoc_rule_0_apply:'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'assoc_rule_0_action_0')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_nonterminal_break_mixed_uses_native_close_path) :-
    plawk_parse_string("$1 == \"ERROR\" { break; hits++; seen[$2]++ } { total++; counts[$1]++ } END { print hits, total, seen[\"disk\"], counts[\"ERROR\"] }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'break_close_stream:'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%break_slot_0 = phi i64 [%rule_0_slot_0, %rule_0_done]'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'rule_0_slot_0_op_0')),
    assertion(\+ sub_atom(DriverIR, _, _, _, 'rule_0_assoc_0')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

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

test(surface_numeric_guard_uses_native_i64_field_cmp) :-
    plawk_parse_string("BEGIN { FS = \":\" } $3 >= 100 { print $1, $3 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_i64_cmp_value(%Value %line, i64 3, i8 58, i64 100, i32 5)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_slice_value(%Value %line, i64 1, i8 58)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_slice_value(%Value %line, i64 3, i8 58)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_numeric_eq_ne_guards_use_numeric_op_codes) :-
    plawk_parse_string("$2 == 0 { zeros++ } $2 != 0 { nonzeros++ } END { print zeros, nonzeros }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_i64_cmp_value(%Value %line, i64 2, i8 32, i64 0, i32 0)'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_atom_field_i64_cmp_value(%Value %line, i64 2, i8 32, i64 0, i32 1)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_begin_output_separator_uses_configured_delimiter) :-
    plawk_parse_string("BEGIN { OFS = \",\" } { total++ } END { print \"total\", total }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    % OFS is now emitted as a byte list (one putchar per byte, indexed), so the
    % single-byte "," separator is `printed_end_separator_1_0` (byte 0) rather
    % than the old un-indexed `printed_end_separator_1`.
    assertion(once(sub_atom(DriverIR, _, _, _, '%printed_end_separator_1_0 = call i32 @putchar(i32 44)'))),
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
