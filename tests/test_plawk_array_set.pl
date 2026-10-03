:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Numeric element writes `arr[$k] = EXPR` (arrays PR 2b) --
% docs/design/PLAWK_ARRAY_VALUE_MODEL.md §5.
%
% `{ c[$1] = 5 }` was a parse error (`=` on an element took only `$0` / `row(...)`).
% It now parses to set_assoc(Arr, Key, Value). The value's semantic kind is counter
% (integer arithmetic, length, int) or double (a field in arithmetic, a float
% literal, any division); an array's writes must agree after numeric widening
% (counter + double -> double), checked by plawk_array_kinds_ok/1 over the shared
% effect enumeration, which also declines on any array use it does not know. A bare
% field (strnum), a string, or a mix with split/row writes declines until their
% PRs land. PR 2c admits the same writes, and `arr[$k] += D`, beside scalar work.
% gawk 5.1.0 (LC_ALL=C); for-in order is unspecified, so outputs compare as sorted
% lines.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 5\nb 7.5\na 3\n").

:- begin_tests(plawk_array_set).

test(element_write_parses) :-
    plawk_parse_string("{ c[$1] = $2 * 2 }\n",
        program([], [rule(always,
            [set_assoc(var(c), field(1), mul_i64(field(2), int(2)))])], [])),
    % the row forms keep their own nodes
    plawk_parse_string("{ t[$1] = $0 }\n",
        program([], [rule(always, [set_row(var(t), field(1))])], [])),
    !.

test(element_write_is_a_write_effect) :-
    plawk_parse_string("{ c[$1] = 5 }\n", P),
    plawk_array_effects(P, E),
    assertion(E == [write(c, set(int(5)))]),
    !.

test(counter_writes, [condition(clang_available)]) :-
    run_sorted("{ c[$1] = 5 } END { for (k in c) print k, c[k] }\n", "a 5\nb 5\n"),
    !,
    run_sorted("{ c[$1] = 1; c[$1]++ } END { for (k in c) print k, c[k] }\n", "a 2\nb 2\n"),
    !,
    run_sorted("{ c[$1] = 5; delete c[\"a\"] } END { for (k in c) print k, c[k] }\n", "b 5\n"),
    !,
    run_sorted("{ c[$1] = 5 } END { print c[\"a\"] }\n", "5\n"),
    !.

test(double_writes, [condition(clang_available)]) :-
    run_sorted("{ c[$1] = $2 * 2 } END { for (k in c) print k, c[k] }\n", "a 6\nb 15\n"),
    !,
    run_sorted("{ c[$1] = 7 / 2 } END { for (k in c) print k, c[k] }\n", "a 3.5\nb 3.5\n"),
    !.

% counter + double widen to double (the `++` lands on the double table)
test(numeric_widening, [condition(clang_available)]) :-
    run_sorted("{ c[$1] = $2 * 1; c[$1]++ } END { for (k in c) print k, c[k] }\n",
        "a 4\nb 8.5\n"),
    !.

% a double-valued write goes through @wam_assoc_f64_set, marked for the IR check
test(double_write_uses_the_f64_entry) :-
    plawk_parse_string("{ c[$1] = $2 * 2 } END { for (k in c) print k, c[k] }\n", P),
    plawk_program_native_driver_ir(P, 'input.txt', IR),
    assertion(sub_atom(IR, _, _, _, '@wam_assoc_f64_set(')),
    assertion(sub_atom(IR, _, _, _, '; plawk-f64-table(c)=')),
    assertion(\+ sub_atom(IR, _, _, _, '@wam_assoc_i64_set(')),
    !.

% Kinds this PR does not admit, and mixes the rules forbid, decline (never exit 4)
test(unadmitted_kinds_and_mixes_decline) :-
    forall(member(Src,
            [ % a bare field copy is strnum (PR 4)
              "{ c[$1] = $2 } END { for (k in c) print k, c[k] }\n",
              % a string (PR 4)
              "{ c[$1] = \"x\" } END { for (k in c) print k, c[k] }\n",
              % number + strnum (split pieces): no dual storage
              "{ c[$1] = 5; split($1, c, \",\") } END { for (k in c) print k, c[k] }\n",
              % number + row
              "{ c[$1] = 5; c[$2] = $0 } END { for (k in c) print k, c[k] }\n"
            ]),
        build_status_is(Src, 3)),
    !.

% --- PR 2c: the same writes beside scalar work (the mixed walker) --------------

test(element_writes_beside_scalar_work, [condition(clang_available)]) :-
    run_sorted("{ c[$1] += $2; n++; print c[$1] }\n", "5\n7.5\n8\n"),
    !,
    run_sorted("{ c[$1] += 2; n++; print c[$1] }\n", "2\n2\n4\n"),
    !,
    run_sorted("{ c[$1] = 5; n++; print c[$1], n }\n", "5 1\n5 2\n5 3\n"),
    !,
    run_sorted("{ c[$1] = $2 * 2; n++ } END { print n, c[\"a\"] }\n", "3 6\n"),
    !.

% a scalar inside the value's arithmetic reads the slot's current value
test(scalar_operands_in_the_value, [condition(clang_available)]) :-
    run_sorted("{ n++; c[$1] = n * 10; print c[$1] }\n", "10\n20\n30\n"),
    !.

% counter + double widen beside scalar work too: `++` lands on the double table
test(mixed_numeric_widening, [condition(clang_available)]) :-
    run_sorted("{ c[$1] = $2 * 1; c[$1]++; n++; print c[$1] }\n", "4\n6\n8.5\n"),
    !.

% a BARE scalar copy may copy an unset value (spec §3): declined
test(bare_scalar_copy_declines) :-
    build_status_is("{ n++; c[$1] = n } END { print c[\"a\"] }\n", 3),
    !.

:- end_tests(plawk_array_set).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_array_set', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run_sorted(Src, Expected) :-
    input(Input),
    odir(Dir),
    directory_file_path(Dir, 'as_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'as', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    build_status_of(Prog, Bin, BuildStatus),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out0),
    close(PS),
    process_wait(Pid, exit(0)),
    split_string(Out0, "\n", "", Lines0),
    exclude(==(""), Lines0, Lines),
    msort(Lines, Sorted),
    atomic_list_concat(Sorted, '\n', Joined0),
    atom_string(Joined0, Joined),
    string_concat(Joined, "\n", Out),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

build_status_is(Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'as_status', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    atom_concat(Prog0, '_bin', Bin),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    build_status_of(Prog, Bin, Status),
    ( Status == Expected
    -> true
    ;  format(user_error, "~n~w~n  build exit ~w, expected ~w~n",
           [Src, Status, Expected]), fail
    ).

build_status_of(Prog, Bin, Status) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(O)), stderr(null), process(Pid)]),
    read_string(O, _, _), close(O),
    process_wait(Pid, exit(Status)).
