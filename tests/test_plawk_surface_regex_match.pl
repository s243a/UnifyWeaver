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

:- dynamic user:plawk_regex_marker/0.

user:plawk_regex_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_surface_regex_match, [condition(clang_available)]).

test(parses_field_match_pattern) :-
    plawk_parse_string("$2 ~ /^d[io]sk$/ { print $0 }\n", Program),
    assertion(Program == program([], [rule(field_match(2, "^d[io]sk$"),
        [print([field(0)])])], [])).

test(parses_negated_field_match_pattern) :-
    plawk_parse_string("$2 !~ /^d/ { count++ } END { print count }\n", Program),
    assertion(Program == program([], [rule(not_pat(field_match(2, "^d")),
        [inc(var(count))])],
        [end([print([var(count)])])])).

test(parses_whole_record_match_via_field_zero) :-
    plawk_parse_string("$0 ~ /E..OR/ { print $1 }\n", Program),
    assertion(Program == program([], [rule(field_match(0, "E..OR"),
        [print([field(1)])])], [])).

test(bare_regex_with_metachars_becomes_field_match) :-
    plawk_parse_string("/ERROR|WARN/ { print $1 }\n", Program),
    assertion(Program == program([], [rule(field_match(0, "ERROR|WARN"),
        [print([field(1)])])], [])).

test(literal_prefix_pattern_keeps_fast_path_ast) :-
    plawk_parse_string("/^ERROR/ { print $0 }\n", Program),
    assertion(Program == program([], [rule(prefix("ERROR"),
        [print([field(0)])])], [])).

test(literal_contains_pattern_keeps_fast_path_ast) :-
    plawk_parse_string("/disk/ { print $0 }\n", Program),
    assertion(Program == program([], [rule(contains("disk"),
        [print([field(0)])])], [])).

test(anchored_prefix_with_metachars_becomes_regex) :-
    plawk_parse_string("/^ERR.R/ { print $0 }\n", Program),
    assertion(Program == program([], [rule(field_match(0, "^ERR.R"),
        [print([field(0)])])], [])).

test(escaped_slash_unescapes_into_literal) :-
    plawk_parse_string("/a\\/b/ { print $0 }\n", Program),
    assertion(Program == program([], [rule(contains("a/b"),
        [print([field(0)])])], [])).

test(parses_regex_in_combined_pattern) :-
    plawk_parse_string("$1 == \"ERROR\" && $2 ~ /net|disk/ { hits++ } END { print hits }\n", Program),
    assertion(Program == program([], [rule(
        and_pat(field_eq(1, "ERROR"), field_match(2, "net|disk")),
        [inc(var(hits))])],
        [end([print([var(hits)])])])).

test(parses_regex_in_if_condition) :-
    plawk_parse_string("{ if ($2 ~ /^d/) { d++ } else { o++ } } END { print d, o }\n", Program),
    assertion(Program == program([], [rule(always,
        [if(field_match(2, "^d"), [inc(var(d))], [inc(var(o))])])],
        [end([print([var(d), var(o)])])])).

test(regex_guard_ir_uses_runtime_matcher) :-
    plawk_parse_string("$2 ~ /^d[io]sk$/ { print $0 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@wam_regex_field_match(%Value %line, i64 2, i8 32'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_regex_cache = internal global i8* null'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(regex_guards_in_one_rule_get_unique_caches) :-
    plawk_parse_string("$2 ~ /^d/ && $3 ~ /full/ { hits++ } END { print hits }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '_5Fl_regex_cache = internal global i8* null'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '_5Fr_regex_cache = internal global i8* null'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_field_regex_matches_character_class) :-
    run_regex_print_smoke("$2 ~ /^d[io]sk$/ { print $0 }\n",
        "ERROR disk full\nERROR dosk odd\nWARN desk hot\nINFO disks many\n",
        "ERROR disk full\nERROR dosk odd\n").

test(surface_negated_field_regex_counts_nonmatches) :-
    run_regex_print_smoke("$2 !~ /^d/ { others++ } { total++ } END { print others, total }\n",
        "ERROR disk a\nWARN cpu b\nINFO net c\n",
        "2 3\n").

test(surface_bare_regex_alternation_matches_whole_record) :-
    run_regex_print_smoke("/ERROR|WARN/ { print $1 }\n",
        "ERROR a\nWARN b\nINFO c\nERRORS d\n",
        "ERROR\nWARN\nERRORS\n").

test(surface_dot_metachar_matches_any_byte) :-
    run_regex_print_smoke("$0 ~ /E..OR/ { print $1 }\n",
        "ERROR a\nEXXOR b\nWARN c\n",
        "ERROR\nEXXOR\n").

test(surface_regex_composes_with_field_eq_guard) :-
    run_regex_print_smoke("$1 == \"ERROR\" && $2 ~ /net|disk/ { hits++ } END { print hits }\n",
        "ERROR disk 1\nERROR cpu 2\nWARN net 3\nERROR net 4\n",
        "2\n").

test(surface_regex_in_if_condition_updates_slots) :-
    run_regex_print_smoke("{ if ($2 ~ /^d/) { d++ } else { o++ } } END { print d, o }\n",
        "a disk\nb cpu\nc desk\n",
        "2 1\n").

test(surface_regex_respects_field_separator) :-
    run_regex_print_smoke("BEGIN { FS = \":\" } $2 ~ /^[0-9]+$/ { nums++ } END { print nums }\n",
        "a:123\nb:12x\nc:9\n",
        "2\n").

test(surface_escaped_slash_matches_literal_slash) :-
    run_regex_print_smoke("/a\\/b/ { print $1 }\n",
        "x a/b here\ny ab there\n",
        "x\n").

test(surface_repetition_and_anchors) :-
    run_regex_print_smoke("$2 ~ /^ab+c?$/ { print NR }\n",
        "r1 abbc\nr2 ac\nr3 ab\nr4 abbbb\n",
        "1\n3\n4\n").

run_regex_print_smoke(Source, Input, ExpectedOutput) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_surface_regex_match', Dir),
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
    directory_file_path(Dir, 'plawk_surface_regex_match.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_regex_marker/0 ],
        [module_name('plawk_surface_regex_match')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'plawk_surface_regex_match_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> assertion(OutStr == ExpectedOutput)
    ;  format(user_error, "~n[plawk surface regex match output]~n~w~n",
              [OutStr]),
       throw(plawk_surface_regex_match_failed(Status))
    ),
    !.

:- end_tests(plawk_surface_regex_match).
