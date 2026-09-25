:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% A pattern with NO action, and where a pattern's action may start.
%
% ---------------------------------------------------------------------------
% 1. BARE PATTERNS. `awk '/re/'`, `awk 'NR==2'`, `awk '$2 > 30'` -- the commonest
%    awk programs there are -- failed to parse (exit 2): the rule grammar required
%    an action block after every pattern. A pattern with no action is awk's default
%    `{ print }`, and now parses to exactly the AST `PATTERN { print }` does, so it
%    reaches every pattern kind the guard lowering already supports.
%
% 2. THE ACTION'S `{` MUST BE ON THE PATTERN'S LINE. In awk, a pattern followed by
%    a newline and then `{ ... }` is TWO rules: the bare pattern (printing matching
%    records) and an unconditional action. The rule grammar skipped whitespace --
%    newlines included -- before the action, so it silently MERGED them into one
%    guarded rule:
%
%        /x/                       gawk:  x 1   (bare /x/ prints the record)
%        { print $2 }                     1     (unconditional)
%                                         2
%                                  plawk: 1     (one rule, guarded by /x/)
%
%    Wrong output with exit 0 -- the worst class -- and pre-existing; it surfaced
%    because a bare pattern made the two-rule reading expressible at all.
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("alice 30 NY\nbob 25 LA\ncarol 35 NY\ndave 40 SF\nalice 12 LA\n").

:- begin_tests(plawk_bare_pattern).

% --- parsing ---------------------------------------------------------------

% A bare pattern is exactly `PATTERN { print }`.
test(bare_pattern_is_pattern_print) :-
    forall(member(Bare, ["/NY/", "NR==2", "$2 > 30", "!/NY/", "/NY/ && $2 > 30",
                         "NR==2, NR==4", "$1 == \"alice\""]),
        ( plawk_parse_string(Bare, P1),
          string_concat(Bare, " { print }", WithAction),
          plawk_parse_string(WithAction, P2),
          assertion(P1 == P2) )),
    !.

% Terminators: newline, `;`, a comment, end of input.
test(bare_pattern_terminators) :-
    plawk_parse_string("/a/\n/b/\n",
        program([], [rule(contains("a"), [print([field(0)])]),
                     rule(contains("b"), [print([field(0)])])], [])),
    !,
    plawk_parse_string("/a/; /b/",
        program([], [rule(contains("a"), [print([field(0)])]),
                     rule(contains("b"), [print([field(0)])])], [])),
    !,
    plawk_parse_string("/a/   # trailing comment\n",
        program([], [rule(contains("a"), [print([field(0)])])], [])),
    !.

% `/a/ /b/` on one line is ONE pattern in awk (a concatenation), which plawk does
% not support -- it must stay a parse failure, not become two rules.
test(same_line_patterns_do_not_split) :-
    \+ plawk_parse_string("/a/ /b/", _),
    !.

% The action on the NEXT line is a separate unconditional rule.
test(action_on_next_line_is_a_separate_rule) :-
    plawk_parse_string("/x/\n{ print $2 }\n",
        program([], [rule(contains("x"), [print([field(0)])]),
                     rule(always, [print([field(2)])])], [])),
    !.

% On the same line it is still the pattern's action (unchanged).
test(action_on_same_line_is_unchanged) :-
    plawk_parse_string("/x/ { print $2 }\n",
        program([], [rule(contains("x"), [print([field(2)])])], [])),
    !,
    plawk_parse_string("/x/\t{ print $2 }\n",
        program([], [rule(contains("x"), [print([field(2)])])], [])),
    !.

% --- runtime parity ----------------------------------------------------------

test(regex, [condition(clang_available)]) :-
    run("/NY/\n", "alice 30 NY\ncarol 35 NY\n"),
    !.

test(record_number, [condition(clang_available)]) :-
    run("NR==2\n", "bob 25 LA\n"),
    !.

test(field_comparison, [condition(clang_available)]) :-
    run("$2 > 30\n", "carol 35 NY\ndave 40 SF\n"),
    !,
    run("$1 == \"alice\"\n", "alice 30 NY\nalice 12 LA\n"),
    !.

test(combinators, [condition(clang_available)]) :-
    run("/NY/ && $2 > 30\n", "carol 35 NY\n"),
    !,
    run("!/NY/\n", "bob 25 LA\ndave 40 SF\nalice 12 LA\n"),
    !.

test(range, [condition(clang_available)]) :-
    run("NR==2, NR==4\n", "bob 25 LA\ncarol 35 NY\ndave 40 SF\n"),
    !.

test(several_bare_rules_each_print, [condition(clang_available)]) :-
    run("/bob/; /dave/\n", "bob 25 LA\ndave 40 SF\n"),
    !,
    % a record matching two bare rules prints twice, as in awk
    run("/li/\n/ce/\n", "alice 30 NY\nalice 30 NY\nalice 12 LA\nalice 12 LA\n"),
    !.

test(with_end, [condition(clang_available)]) :-
    run("/NY/\nEND { print NR }\n", "alice 30 NY\ncarol 35 NY\n5\n"),
    !.

% The formerly-merged program: two rules, as gawk runs it.
test(next_line_action_runs_unconditionally, [condition(clang_available)]) :-
    run_with("x 1\ny 2\n", "/x/\n{ print $2 }\n", "x 1\n1\n2\n"),
    !,
    run("/li/\n{ n++ }\nEND { print n }\n", "alice 30 NY\nalice 12 LA\n5\n"),
    !.

:- end_tests(plawk_bare_pattern).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_bare_pattern', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_with(Input, Src, Expected).

run_with(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'bp_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'bp', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    cli([build, Prog, '-o', Bin], 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
