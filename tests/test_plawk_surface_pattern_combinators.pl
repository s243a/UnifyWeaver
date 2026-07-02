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

:- dynamic user:plawk_patcomb_marker/0.

user:plawk_patcomb_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_surface_pattern_combinators, [condition(clang_available)]).

test(parses_and_pattern) :-
    plawk_parse_string("$1 == \"ERROR\" && $3 > 100 { print $0 }\n", Program),
    assertion(Program == program([], [rule(
        and_pat(field_eq(1, "ERROR"), field_cmp(3, gt, 100)),
        [print([field(0)])])], [])).

test(parses_or_pattern) :-
    plawk_parse_string("/^ERROR/ || /^WARN/ { print $1 }\n", Program),
    assertion(Program == program([], [rule(
        or_pat(prefix("ERROR"), prefix("WARN")),
        [print([field(1)])])], [])).

test(parses_not_pattern) :-
    plawk_parse_string("!/disk/ { print $0 }\n", Program),
    assertion(Program == program([], [rule(
        not_pat(contains("disk")),
        [print([field(0)])])], [])).

test(parses_and_binds_tighter_than_or) :-
    plawk_parse_string("$1 == \"A\" || $1 == \"B\" && $3 > 5 { h++ } END { print h }\n", Program),
    assertion(Program == program([], [rule(
        or_pat(field_eq(1, "A"), and_pat(field_eq(1, "B"), field_cmp(3, gt, 5))),
        [inc(var(h))])],
        [end([print([var(h)])])])).

test(parses_parenthesized_grouping) :-
    plawk_parse_string("($1 == \"A\" || $1 == \"B\") && $3 > 5 { h++ } END { print h }\n", Program),
    assertion(Program == program([], [rule(
        and_pat(or_pat(field_eq(1, "A"), field_eq(1, "B")), field_cmp(3, gt, 5)),
        [inc(var(h))])],
        [end([print([var(h)])])])).

test(parses_not_parenthesized_pattern) :-
    plawk_parse_string("!($1 == \"ERROR\") { ok++ } END { print ok }\n", Program),
    assertion(Program == program([], [rule(
        not_pat(field_eq(1, "ERROR")),
        [inc(var(ok))])],
        [end([print([var(ok)])])])).

test(parses_left_associative_and_chain) :-
    plawk_parse_string("$1 == \"A\" && $3 > 5 && $3 < 10 { h++ } END { print h }\n", Program),
    assertion(Program == program([], [rule(
        and_pat(and_pat(field_eq(1, "A"), field_cmp(3, gt, 5)), field_cmp(3, lt, 10)),
        [inc(var(h))])],
        [end([print([var(h)])])])).

test(parses_combined_if_condition) :-
    plawk_parse_string("{ if ($1 == \"ERROR\" && $3 > 100) { big++ } else { small++ } } END { print big, small }\n", Program),
    assertion(Program == program([], [rule(always,
        [if(and_pat(field_eq(1, "ERROR"), field_cmp(3, gt, 100)),
            [inc(var(big))],
            [inc(var(small))])])],
        [end([print([var(big), var(small)])])])).

test(combined_guard_ir_is_single_block) :-
    plawk_parse_string("$1 == \"ERROR\" && $3 > 100 { print $0 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%is_match_l = call i1 @wam_atom_field_eq_value'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%is_match_r = call i1 @wam_atom_field_i64_cmp_value'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%is_match = and i1 %is_match_l, %is_match_r'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(not_guard_ir_uses_xor) :-
    plawk_parse_string("!/disk/ { print $0 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%is_match = xor i1 %is_match_n, true'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    !.

test(surface_and_pattern_filters_records) :-
    run_patcomb_print_smoke("$1 == \"ERROR\" && $3 > 100 { print $0 }\n",
        "ERROR disk 50\nERROR net 200\nWARN cpu 300\nINFO ok 400\n",
        "ERROR net 200\n").

test(surface_or_pattern_matches_either) :-
    run_patcomb_print_smoke("/^ERROR/ || /^WARN/ { print $1, $3 }\n",
        "ERROR disk 50\nERROR net 200\nWARN cpu 300\nINFO ok 400\n",
        "ERROR 50\nERROR 200\nWARN 300\n").

test(surface_not_pattern_inverts_match) :-
    run_patcomb_print_smoke("!/disk/ { print $1 }\n",
        "ERROR disk 50\nERROR net 200\nWARN cpu 300\nINFO ok 400\n",
        "ERROR\nWARN\nINFO\n").

test(surface_precedence_and_over_or) :-
    % INFO 100 matches `INFO || (WARN && >250)` but not `(INFO || WARN) && >250`.
    run_patcomb_print_smoke("$1 == \"INFO\" || $1 == \"WARN\" && $3 > 250 { hits++ } END { print hits }\n",
        "ERROR disk 50\nWARN cpu 300\nINFO ok 100\nWARN io 200\n",
        "2\n").

test(surface_parens_group_or_before_and) :-
    run_patcomb_print_smoke("($1 == \"INFO\" || $1 == \"WARN\") && $3 > 250 { hits++ } END { print hits }\n",
        "ERROR disk 50\nWARN cpu 300\nINFO ok 100\nWARN io 200\n",
        "1\n").

test(surface_combined_if_condition_updates_slots) :-
    run_patcomb_print_smoke("{ if ($1 == \"ERROR\" && $3 > 100) { big++ } else { small++ } } END { print big, small }\n",
        "ERROR disk 50\nERROR net 200\nWARN cpu 300\nINFO ok 400\n",
        "1 3\n").

test(surface_or_pattern_guards_assoc_counts) :-
    run_patcomb_print_smoke("$1 == \"ERROR\" || $1 == \"WARN\" { counts[$1]++ } END { print counts[\"ERROR\"], counts[\"WARN\"] }\n",
        "ERROR disk 50\nERROR net 200\nWARN cpu 300\nINFO ok 400\n",
        "2 1\n").

test(surface_not_field_eq_counts_others) :-
    run_patcomb_print_smoke("!($1 == \"ERROR\") { ok++ } END { print ok }\n",
        "ERROR disk 50\nERROR net 200\nWARN cpu 300\nINFO ok 400\n",
        "2\n").

test(surface_left_associative_and_chain_range) :-
    run_patcomb_print_smoke("$1 == \"WARN\" && $3 > 100 && $3 < 250 { print $2 }\n",
        "WARN cpu 300\nWARN io 200\nWARN net 50\nERROR disk 200\n",
        "io\n").

run_patcomb_print_smoke(Source, Input, ExpectedOutput) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_surface_pattern_combinators', Dir),
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
    directory_file_path(Dir, 'plawk_surface_pattern_combinators.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_patcomb_marker/0 ],
        [module_name('plawk_surface_pattern_combinators')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'plawk_surface_pattern_combinators_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> assertion(OutStr == ExpectedOutput)
    ;  format(user_error, "~n[plawk surface pattern combinators output]~n~w~n",
              [OutStr]),
       throw(plawk_surface_pattern_combinators_failed(Status))
    ),
    !.

:- end_tests(plawk_surface_pattern_combinators).
