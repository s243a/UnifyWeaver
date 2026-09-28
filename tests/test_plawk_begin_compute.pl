:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Computation in BEGIN -- PR 1: the parser accepts the rule-body statement
% grammar in BEGIN; codegen fences what it cannot lower yet.
%
% BEGIN used to accept only special-variable string sets, `NAME = INT` / `NAME =
% "str"` literal inits, print, printf and exit, so `BEGIN { x = 1 + 2; print x }`,
% loops, `x++` and concatenation were PARSE errors (exit 2). They now parse -- the
% BEGIN productions are tried first but only when they consume the whole
% statement (so `x = 3 + 1` is not committed as `x = 3`), then the rule-body
% grammar -- and every BEGIN action outside the forms the drivers lower DECLINES
% (exit 3, plawk_program_has_unlowered_begin/1). Several drivers ignore a BEGIN
% action they do not recognise, so without the fence the widened parser would
% have printed zeros instead of declining. Later PRs lower BEGIN computation
% through the shared rule walker (a prelude before the record loop).
%
% IR is byte-identical to the base for all 2448 program strings in the suites.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
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
    plawk_parse_string("BEGIN { for (i = 1; i <= 3; i++) print i }\n",
        program([begin([c_for(set(var(i), int(1)), cmp(var(i), le, int(3)),
            inc(var(i)), [print([var(i)])])])], [], [])),
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

% Every newly parsing form declines cleanly -- never a silently dropped BEGIN
% statement (wrong output) and never exit 4.
test(new_forms_decline_until_lowered) :-
    forall(member(Src,
            [ "BEGIN { x = 3; print x }\n",
              "BEGIN { x = 1 + 2; print x }\n",
              "BEGIN { for (i = 1; i <= 3; i++) print i }\n",
              "BEGIN { i = 0; while (i < 3) { print i; i++ } }\n",
              "BEGIN { s = \"a\" \"b\"; print s }\n",
              "BEGIN { x = 3; x++; print x }\n",
              "BEGIN { x = 2 * 3 } { print $2 * x }\n",
              "BEGIN { x = 2 * 3 } { t += $2 } END { print t + x }\n",
              "BEGIN { x = sprintf(\"%d-%d\", 1, 2); print x }\n",
              % without the fence these two BUILT and silently dropped the loop:
              % the first printed nothing, the second "a" then the records
              "BEGIN { for (i = 0; i < 2; i++) print i }\n",
              "BEGIN { print \"a\"; for (i = 0; i < 2; i++) print i } { print $1 }\n"
            ]),
        build_status_is(Src, 3)),
    !.

:- end_tests(plawk_begin_compute).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_begin_compute', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

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
