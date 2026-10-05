:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Array element reads in conditions (arrays PR 3a) and inside arithmetic (PR 3b) --
% docs/design/PLAWK_ARRAY_VALUE_MODEL.md section 8, item 3.
%
% `if (c[$1] > 1)` was a parse error: a condition operand could be a scalar, a
% special or a literal, never an element. It now parses to
% cmp(assoc(var(c), Key), Op, Rhs) on either side. The codegen reads the element
% numerically -- an absent element is 0, awk's numeric reading -- on COUNTER tables
% only (every write `++`, `+= int` or a counter set, not imported). The key is
% interned exactly as the writes intern it (field, literal, integer, SUBSEP list,
% or a scalar's id). Double / string / split / row tables, a string comparison, an
% END condition and a loop condition decline (exit 3), never exit 4.
% gawk 5.1.0 (LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 3\nb 5\na 2\nc 1\nb 4\na 7\n").

:- begin_tests(plawk_array_reads).

test(element_operand_parses) :-
    plawk_parse_string("{ if (c[$1] > 1) print }\n",
        program([], [rule(always, [if(scalar_if(cmp(assoc(var(c), field(1)), gt, int(1))),
            [print([field(0)])], [])])], [])),
    plawk_parse_string("{ if (n < c[\"a\"]) print }\n",
        program([], [rule(always, [if(scalar_if(cmp(var(n), lt, assoc(var(c), string("a")))),
            [print([field(0)])], [])])], [])),
    !.

test(counter_reads_in_conditions, [condition(clang_available)]) :-
    run_exact("{ c[$1]++; if (c[$1] > 1) print $0 }\n", "a 2\nb 4\na 7\n"),
    run_exact("{ if (c[$1] == 0) print \"new\", $1; c[$1]++ }\n", "new a\nnew b\nnew c\n"),
    run_exact("{ c[$1]++; if (c[$1] >= 2 && c[$1] < 3) print \"second\", $1 }\n",
        "second a\nsecond b\n"),
    run_exact("{ c[$1]++; if (c[$1] > 1) print $0; else print \"first\", $1 }\n",
        "first a\nfirst b\na 2\nfirst c\nb 4\na 7\n"),
    !.

test(element_on_the_right_and_literal_keys, [condition(clang_available)]) :-
    run_exact("{ c[$1]++; n++; if (n > c[$1]) print n, $1 }\n", "2 b\n3 a\n4 c\n5 b\n6 a\n"),
    run_exact("{ c[$1]++; if (c[\"a\"] == 2) print \"a twice at\", NR }\n",
        "a twice at 3\na twice at 4\na twice at 5\n"),
    !.

% The key reaches the id the write stored: a missing field keys on "" (reads 0),
% a scalar key uses the slot's own id (k stays a string -- reading c[k] must not
% turn k into a number), a counter key interns its decimal.
test(key_forms, [condition(clang_available)]) :-
    run_exact("{ c[$1]++; if (c[$9] == 0) print \"zero\" }\n",
        "zero\nzero\nzero\nzero\nzero\nzero\n"),
    run_exact("{ k = $1; c[k]++; if (c[k] > 2) print \"third\", k }\n", "third a\n"),
    run_exact("{ n++; k = n % 2; c[k]++; if (c[k] > 1) print k, c[k] }\n",
        "1 2\n0 2\n1 3\n0 3\n"),
    !.

test(unadmitted_reads_decline) :-
    forall(member(Src,
            [ % a double table (a field delta) -- f64 reads are a follow-on
              "{ c[$1] += $2; if (c[$1] > 5) print $1 }\n",
              % a strnum value (bare field copy)
              "{ c[$1] = $2; if (c[$1] > 3) print $1 }\n",
              % a string comparison
              "{ c[$1]++; if (c[$1] == \"1\") print $1 }\n",
              % a split (string) table
              "{ n = split($1, a, \",\"); if (a[1] > 0) print }\n",
              % END condition, loop condition
              "{ c[$1]++ } END { if (c[\"a\"] > 1) print \"many a\" }\n",
              "{ c[$1]++; while (c[$1] < 3) c[$1]++ ; print $1, c[$1] }\n",
              % never written: no counter table to read
              "{ if (c[$1] > 1) print }\n"
            ]),
        build_status_is(Src, 3)),
    !.

% A loop whose body declines after its break/continue context was pushed must
% not leave that context behind: a later rule-level `break` in the same process
% branched to the dead loop's exit label instead of break_close_stream.
test(declined_loop_body_leaves_no_loop_context) :-
    plawk_parse_string("{ n = split($1, a, \",\"); for (i = 1; i <= n; i++) if (a[i] == \"b\") print \"found\" }\n", P1),
    \+ plawk_program_native_driver_ir(P1, 'input.txt', _),
    ( nb_current(plawk_loopctx, Stack) -> true ; Stack = [] ),
    assertion(Stack == []),
    plawk_parse_string("{ if ($1 == \"x\") { hits++; break } else { total++ } } END { print hits, total }\n", P2),
    plawk_program_native_driver_ir(P2, 'input.txt', IR),
    assertion(sub_atom(IR, _, _, _, 'br label %break_close_stream')),
    !.

% --- PR 3b: element reads inside arithmetic (the mixed walker) ---------------

test(arith_reads_in_assignments, [condition(clang_available)]) :-
    run_exact("{ c[$1]++; n = c[$1] * 10; print n }\n", "10\n10\n20\n10\n20\n30\n"),
    run_exact("{ c[$1]++; n = c[\"a\"] + c[\"b\"]; print n }\n", "1\n2\n3\n3\n4\n5\n"),
    run_exact("{ c[$1]++; n = c[$1] - 1; if (n > 0) print $1, n }\n", "a 1\nb 1\na 2\n"),
    !.

% `t += c[k]` / `t -= c[k]` take a bare element on the right.
test(add_assign_an_element, [condition(clang_available)]) :-
    run_exact("{ c[$1]++; t += c[$1] } END { print t }\n", "10\n"),
    run_exact("{ c[$1]++; t -= c[$1] } END { print t }\n", "-10\n"),
    !.

test(arith_reads_in_prints, [condition(clang_available)]) :-
    run_exact("{ c[$1]++; n++; print c[$1] * 2, n }\n", "2 1\n2 2\n4 3\n2 4\n4 5\n6 6\n"),
    !.

% A scalar key stays a string when its element is read in arithmetic too.
test(arith_reads_keep_a_scalar_key_a_string, [condition(clang_available)]) :-
    run_exact("{ k = $1; c[k]++; n = c[k] * 2; print k, n }\n",
        "a 2\nb 2\na 4\nc 2\nb 4\na 6\n"),
    run_exact("{ k = $1; c[k]++; print k, c[k] * 3 }\n", "a 3\nb 3\na 6\nc 3\nb 6\na 9\n"),
    !.

test(unadmitted_arith_reads_decline) :-
    forall(member(Src,
            [ % a double table
              "{ c[$1] += $2; n = c[$1] * 2; print n }\n",
              % a bare copy may be an unset element (spec section 3)
              "{ c[$1]++; n = c[$1]; print n }\n",
              % END expression reads (3b follow-on)
              "{ c[$1]++ } END { print c[\"a\"] + c[\"b\"] }\n"
            ]),
        build_status_is(Src, 3)),
    !.

:- end_tests(plawk_array_reads).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_array_reads', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run_exact(Src, Expected) :-
    input(Input),
    odir(Dir),
    directory_file_path(Dir, 'ar_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'ar', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    build_status_of(Prog, Bin, BuildStatus),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    (   Out == Expected
    ->  true
    ;   format(user_error, "~n~w~n  got      ~q~n  expected ~q~n", [Src, Out, Expected]),
        fail
    ).

build_status_is(Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'ar_status', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    atom_concat(Prog0, '_bin', Bin),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    build_status_of(Prog, Bin, Status),
    (   Status == Expected
    ->  true
    ;   format(user_error, "~n~w~n  build exit ~w, expected ~w~n", [Src, Status, Expected]),
        fail
    ).

build_status_of(Prog, Bin, Status) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(O)), stderr(null), process(Pid)]),
    read_string(O, _, _), close(O),
    process_wait(Pid, exit(Status)).
