:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Computation in BEGIN.
%
% PR 1: the parser accepts the rule-body statement grammar in BEGIN. BEGIN used to
% accept only special-variable string sets, `NAME = INT` / `NAME = "str"` literal
% inits, print, printf and exit, so `BEGIN { x = 1 + 2; print x }`, loops, `x++`
% and concatenation were PARSE errors (exit 2). The BEGIN productions are tried
% first but only when they consume the whole statement (so `x = 3 + 1` is not
% committed as `x = 3`), then the rule-body grammar. Every BEGIN action outside
% the forms the drivers lower is fenced (plawk_program_has_unlowered_begin/1):
% several drivers ignore a BEGIN action they do not recognise, which printed
% nothing for `BEGIN { for (i = 0; i < 2; i++) print i }` with the fence off.
%
% PR 2: the BEGIN PRELUDE. Such BEGIN statements run once, before the record loop,
% through the shared rule walker, starting from the type-zero slot values; the
% prelude's output values seed the scalar loop-header phis (the generalisation of
% the literal seed). Checked on the output: each assigned name seeds a phi marked
% `; plawk-begin-prelude(NAME)` (with seeding disabled the programs decline), the
% prelude touches no record or loop state, and every @global it uses is defined.
% A driver with no scalar loop state runs the prelude standalone when nothing
% after BEGIN reads what it assigns. Still declined: exit, getline, arrays, record
% references (`$1`, NR) in BEGIN, and END-only programs.
%
% IR is byte-identical to the base for all 2448 program strings in the suites,
% except two formerly-declined BEGIN programs that now compile.
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_begin_compute).

test(computation_parses) :-
    plawk_parse_string("BEGIN { x = 1 + 2; print x }\n",
        program([begin([set(var(x), add_i64(int(1), int(2))), print([var(x)])])], [], [])),
    !.

% the literal-init production must not commit to a prefix of the statement
test(literal_prefix_is_not_committed) :-
    plawk_parse_string("BEGIN { x = 3 + 1 }\n",
        program([begin([set(var(x), add_i64(int(3), int(1)))])], [], [])),
    !.

test(loops_increments_and_concat_parse) :-
    % a C-for desugars to init + while, as in a rule body
    plawk_parse_string("BEGIN { for (i = 1; i <= 3; i++) print i }\n",
        program([begin([set(var(i), int(1)), while_loop(cmp(var(i), le, int(3)),
            [print([var(i)]), inc(var(i))])])], [], [])),
    plawk_parse_string("BEGIN { x = 3; x++; print x }\n",
        program([begin([set(var(x), int(3)), inc(var(x)), print([var(x)])])], [], [])),
    plawk_parse_string("BEGIN { s = \"a\" \"b\"; print s }\n",
        program([begin([set(var(s), concat([string("a"), string("b")])),
            print([var(s)])])], [], [])),
    !.

% as in a rule body, a statement may follow a compound one without a separator
test(statement_after_a_block_parses) :-
    plawk_parse_string("BEGIN { while (i < 2) { i++ } print i }\n",
        program([begin([while_loop(cmp(var(i), lt, int(2)), [inc(var(i))]),
            print([var(i)])])], [], [])),
    !.

% the forms BEGIN always accepted keep their exact terms
test(legacy_forms_unchanged) :-
    plawk_parse_string("BEGIN { FS = \":\"; n = 1; s = \"x\"; printf \"%d\\n\", 7; print n; exit 2 }\n",
        program([begin([set(var('FS'), string(":")), set(var(n), int(1)),
            set(var(s), string("x")), printf(string("%d\n"), [int(7)]),
            print([var(n)]), exit(int(2))])], [], [])),
    !.

% special variables are still set only through their own string production
test(special_variables_still_rejected) :-
    \+ plawk_parse_string("BEGIN { NR = 1 }\n{ print }\n", _),
    \+ plawk_parse_string("BEGIN { FS = x }\n{ print }\n", _),
    !.

% --- PR 2: the prelude ------------------------------------------------------

test(begin_only_computation, [condition(clang_available)]) :-
    run("BEGIN { x = 3; print x }\n", "3\n"),
    !,
    run("BEGIN { x = 1 + 2; print x }\n", "3\n"),
    !,
    run("BEGIN { x = 3; y = x * 2; print y }\n", "6\n"),
    !,
    run("BEGIN { x = 3; x++; print x }\n", "4\n"),
    !,
    run("BEGIN { s = \"a\" \"b\"; print s }\n", "ab\n"),
    !,
    run("BEGIN { x = sprintf(\"%d-%d\", 1, 2); print x }\n", "1-2\n"),
    !,
    run("BEGIN { x = 1.5 * 2; print x }\n", "3\n"),
    !,
    run("BEGIN { x = 2 + 2; printf \"%d\\n\", x }\n", "4\n"),
    !.

test(begin_loops_and_conditionals, [condition(clang_available)]) :-
    run("BEGIN { for (i = 1; i <= 3; i++) print i }\n", "1\n2\n3\n"),
    !,
    run("BEGIN { i = 0; while (i < 3) { print i; i++ } }\n", "0\n1\n2\n"),
    !,
    run("BEGIN { for (i = 1; i <= 2; i++) for (j = 1; j <= 2; j++) print i, j }\n",
        "1 1\n1 2\n2 1\n2 2\n"),
    !,
    run("BEGIN { x = 10; if (x > 5) print \"big\"; else print \"small\" }\n", "big\n"),
    !.

% prelude values seed the record loop (numeric, double and string slots), and END
% sees them on empty input too
test(prelude_values_seed_the_loop, [condition(clang_available)]) :-
    run("BEGIN { x = 2 * 3 } { print $2 * x }\n", "30\n42\n12\n"),
    !,
    run("BEGIN { x = 2 * 3 } { t += $2 } END { print t + x }\n", "20\n"),
    !,
    run("BEGIN { n = 2 * 2 } { n++ } END { print n }\n", "7\n"),
    !,
    run_with("", "BEGIN { n = 2 * 2 } { n++ } END { print n }\n", "4\n"),
    !,
    run("BEGIN { f = 0.5 + 0.25 } { t += $2 * f } END { print t }\n", "10.5\n"),
    !,
    run("BEGIN { sep = \"-\" \"-\" } { print $1 sep $2 }\n", "a--5\nb--7\nc--2\n"),
    !,
    run("BEGIN { OFS = \"-\"; x = 1 + 1 } { print $1, x }\n", "a-2\nb-2\nc-2\n"),
    !,
    run("BEGIN { t = 0; for (i = 1; i <= 10; i++) t += i; print t } { n++ } END { print n, t }\n",
        "55\n3 55\n"),
    !,
    % the mixed (scalar + array) driver seeds from the same phis
    run("BEGIN { n = 5 + 1 } { c[$1]++; n++ } END { print n }\n", "9\n"),
    !.

% BEGIN output comes before the records'; a driver without scalar state runs the
% prelude standalone (nothing after BEGIN reads i)
test(begin_output_precedes_the_records, [condition(clang_available)]) :-
    run("BEGIN { print \"a\"; for (i = 0; i < 2; i++) print i } { print $1 }\n",
        "a\n0\n1\na\nb\nc\n"),
    !,
    run("BEGIN { print \"start\"; x = 1 + 1 } { print $1, x } END { print \"end\", x }\n",
        "start\na 2\nb 2\nc 2\nend 2\n"),
    !.

% What a prelude cannot run declines cleanly -- never a dropped statement, never
% exit 4: exit (must skip the loop yet run END), getline, a record reference
% (BEGIN runs before the first record), END-only programs (a later PR).
test(prelude_boundaries_decline) :-
    forall(member(Src,
            [ "BEGIN { x = 1 + 1; exit }\n",
              "BEGIN { x = 1 + 1; getline line < \"/etc/hostname\" }\n",
              "BEGIN { x = $1; print x }\n",
              "BEGIN { x = NR; print x }\n",
              "BEGIN { x = 3 + 4 } END { print x }\n"
            ]),
        build_status_is(Src, 3)),
    !.

:- end_tests(plawk_begin_compute).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_begin_compute', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

input("a 5\nb 7\nc 2\n").

run(Src, Expected) :-
    input(Input),
    run_with(Input, Src, Expected).

run_with(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'bc_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'bc', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(O)), stderr(null), process(BPid)]),
    read_string(O, _, _), close(O),
    process_wait(BPid, exit(BuildStatus)),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

build_status_is(Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'bc_status', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    atom_concat(Prog0, '_bin', Bin),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(O)), stderr(null), process(Pid)]),
    read_string(O, _, _), close(O),
    process_wait(Pid, exit(Status)),
    ( Status == Expected
    -> true
    ;  format(user_error, "~n~w~n  build exit ~w, expected ~w~n",
           [Src, Status, Expected]), fail
    ).
