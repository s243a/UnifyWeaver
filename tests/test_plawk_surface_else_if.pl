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

:- dynamic user:plawk_elseif_marker/0.

user:plawk_elseif_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_surface_else_if, [condition(clang_available)]).

test(parses_else_less_if) :-
    plawk_parse_string("{ if ($1 == \"ERROR\") { errors++ } } END { print errors }\n", Program),
    assertion(Program == program([], [rule(always,
        [if(field_eq(1, "ERROR"), [inc(var(errors))], [])])],
        [end([print([var(errors)])])])).

test(parses_else_if_chain_as_nested_if) :-
    plawk_parse_string("{ if ($3 > 100) { big++ } else if ($3 > 10) { mid++ } else { small++ } } END { print big, mid, small }\n", Program),
    assertion(Program == program([], [rule(always,
        [if(field_cmp(3, gt, 100),
            [inc(var(big))],
            [if(field_cmp(3, gt, 10),
                [inc(var(mid))],
                [inc(var(small))])])])],
        [end([print([var(big), var(mid), var(small)])])])).

test(parses_else_if_without_final_else) :-
    plawk_parse_string("{ if ($3 > 100) { big++ } else if ($3 > 10) { mid++ } } END { print big, mid }\n", Program),
    assertion(Program == program([], [rule(always,
        [if(field_cmp(3, gt, 100),
            [inc(var(big))],
            [if(field_cmp(3, gt, 10), [inc(var(mid))], [])])])],
        [end([print([var(big), var(mid)])])])).

test(parses_nested_if_inside_then_branch) :-
    plawk_parse_string("{ if ($1 == \"ERROR\") { if ($3 > 100) { bigerr++ } } } END { print bigerr }\n", Program),
    assertion(Program == program([], [rule(always,
        [if(field_eq(1, "ERROR"),
            [if(field_cmp(3, gt, 100), [inc(var(bigerr))], [])],
            [])])],
        [end([print([var(bigerr)])])])).

test(else_of_identifier_prefix_is_not_consumed) :-
    % "elsewhere" must not be parsed as the keyword "else".
    plawk_parse_string("{ if ($1 == \"A\") { x++ }; elsewhere++ } END { print x, elsewhere }\n", Program),
    assertion(Program == program([], [rule(always,
        [if(field_eq(1, "A"), [inc(var(x))], []), inc(var(elsewhere))])],
        [end([print([var(x), var(elsewhere)])])])).

test(surface_else_less_if_counts_matches_only) :-
    run_elseif_print_smoke("{ if ($1 == \"ERROR\") { errors++ } } END { print errors }\n",
        "ERROR disk 300\nWARN cpu 50\nINFO net 15\nERROR mem 8\n",
        "2\n").

test(surface_else_if_chain_buckets_records) :-
    run_elseif_print_smoke("{ if ($3 > 100) { big++ } else if ($3 > 10) { mid++ } else { small++ } } END { print big, mid, small }\n",
        "ERROR disk 300\nWARN cpu 50\nINFO net 15\nERROR mem 8\n",
        "1 2 1\n").

test(surface_else_if_without_final_else) :-
    run_elseif_print_smoke("{ if ($3 > 100) { big++ } else if ($3 > 10) { mid++ } } END { print big, mid }\n",
        "ERROR disk 300\nWARN cpu 50\nINFO net 15\nERROR mem 8\n",
        "1 2\n").

test(surface_three_level_else_if_chain) :-
    run_elseif_print_smoke("{ if ($3 > 100) { t1++ } else if ($3 > 40) { t2++ } else if ($3 > 10) { t3++ } else { t4++ } } END { print t1, t2, t3, t4 }\n",
        "ERROR disk 300\nWARN cpu 50\nINFO net 15\nERROR mem 8\n",
        "1 1 1 1\n").

test(surface_nested_if_inside_then_branch) :-
    run_elseif_print_smoke("{ if ($1 == \"ERROR\") { if ($3 > 100) { bigerr++ } } } END { print bigerr }\n",
        "ERROR disk 300\nWARN cpu 50\nINFO net 15\nERROR mem 8\n",
        "1\n").

test(surface_else_less_if_with_branch_print) :-
    run_elseif_print_smoke("{ if ($3 > 200) { print \"huge\", $2 } } END { print \"done\" }\n",
        "ERROR disk 300\nWARN cpu 50\nINFO net 15\nERROR mem 8\n",
        "huge disk\ndone\n").

test(surface_else_less_if_with_assoc_increment) :-
    run_elseif_print_smoke("{ if ($1 == \"ERROR\") { by[$2]++ } } END { print by[\"disk\"], by[\"mem\"], by[\"cpu\"] }\n",
        "ERROR disk 300\nWARN cpu 50\nINFO net 15\nERROR mem 8\n",
        "1 1 0\n").

test(surface_else_if_with_combined_condition) :-
    run_elseif_print_smoke("{ if ($1 == \"ERROR\" && $3 > 100) { crit++ } else if ($1 == \"ERROR\") { minor++ } } END { print crit, minor }\n",
        "ERROR disk 300\nWARN cpu 50\nINFO net 15\nERROR mem 8\n",
        "1 1\n").

run_elseif_print_smoke(Source, Input, ExpectedOutput) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_surface_else_if', Dir),
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
    directory_file_path(Dir, 'plawk_surface_else_if.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_elseif_marker/0 ],
        [module_name('plawk_surface_else_if')], LLPath),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'plawk_surface_else_if_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> assertion(OutStr == ExpectedOutput)
    ;  format(user_error, "~n[plawk surface else if output]~n~w~n",
              [OutStr]),
       throw(plawk_surface_else_if_failed(Status))
    ),
    !.

:- end_tests(plawk_surface_else_if).
