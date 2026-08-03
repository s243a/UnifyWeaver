:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: `while` / `do-while` / C-style `for` in an END block.
%
%   { n++ } END { while (n > 0) { print n; n-- } }      PARSE ERROR
%
% END admitted `if` and `for (k in arr)` and no other loop.
%
% ---------------------------------------------------------------------------
% ONE LOOP EMITTER, TWO CONTEXTS
%
% The body goes through plawk_scalar_action_sequence_pairs//15 -- the SAME emitter
% a rule body uses -- not hand-written phi plumbing. That is possible because the
% while/do-while clauses of that emitter derive every label from their Prefix and
% OpIndex and take their incoming slot values as a PARAMETER: they carry no
% dependency on the record loop. END supplies the prefix `end_body` and the final
% slot values, and gets the same lowering -- head phi, and break/continue via the
% loop-context stack.
%
% That is why this needed no new phi code and worked on the first probe. Writing a
% second loop emitter for END would have been the obvious move and the wrong one:
% it is the duplication this line keeps paying for (the ORS terminator re-derived
% in four print emitters; the text-slot guard in one copy of the string comparison
% and not the other).
%
% ---------------------------------------------------------------------------
% AND WHAT REUSE COST -- both bugs below were INTRODUCED by admitting loops here
%
% Reusing an emitter inherits what it ASSUMES about its context, not only what it
% does. Two things had to be gated, found by probing the edges rather than by
% reasoning about them:
%
%   FIELD READ    `END { while (n > 0) { print $1; n-- } }` printed
%                 `end_of_file` three times where gawk prints the last record.
%                 END has no current record: at `end_print` the transient %line
%                 holds the EOF sentinel, and the sequence emitter lowers `$1`
%                 against %line without knowing it has left the record loop.
%                 WRONG OUTPUT -- the worst failure mode -- now a decline.
%                 A straight-line `END { print $1 }` already declined; routing
%                 loop bodies through that emitter made the path reachable.
%
%   exit IN A     Made clang fail. An END `exit` stores to @plawk_exit_code and
%   LOOP BODY     TRUNCATES the remaining statements -- sound only because
%                 straight-line code cannot come back. A loop body can, so the
%                 truncation left the block malformed. `exit` AFTER the loop is
%                 fine and stays supported.
%
% The gate is a STRUCTURAL term walk for field(_) and exit, not a
% per-action-shape walker: it is a safety check, and a walker that must learn each
% new action shape is exactly how the misses in this line happened. A depth-first
% search cannot be defeated by a new nesting level -- pinned by the
% field-in-a-nested-loop test below.
%
% gawk 5.2 is the oracle for every expectation here.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Three records, so a counting rule leaves n = 3.
input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_end_loops).

% --- the four loop forms --------------------------------------------------

test(end_while_countdown, [condition(clang_available)]) :-
    run("{ n++ } END { while (n > 0) { print n; n-- } }\n", "3\n2\n1\n"),
    !.

test(end_while_countup, [condition(clang_available)]) :-
    run("{ n++ } END { i = 0; while (i < n) { print i; i++ } }\n", "0\n1\n2\n"),
    !.

test(end_do_while, [condition(clang_available)]) :-
    run("{ n++ } END { do { print n; n-- } while (n > 0) }\n", "3\n2\n1\n"),
    !.

test(end_do_while_countup, [condition(clang_available)]) :-
    run("{ n++ } END { i = 1; do { print i; i++ } while (i <= n) }\n",
        "1\n2\n3\n"),
    !.

test(end_c_for, [condition(clang_available)]) :-
    run("{ n++ } END { for (i = 0; i < n; i++) print i }\n", "0\n1\n2\n"),
    !.

% --- composition ---------------------------------------------------------

% A loop between two straight-line prints: the loop's blocks have to rejoin so
% the statements after it still run.
test(loop_between_straight_line_prints, [condition(clang_available)]) :-
    run("{ n++ } END { print \"start\"; while (n > 0) { print n; n-- }; print \"end\" }\n",
        "start\n3\n2\n1\nend\n"),
    !.

% break and continue, via the loop-context stack in this new position.
test(break_in_an_end_loop, [condition(clang_available)]) :-
    run("{ n++ } END { while (n > 0) { n--; if (n == 1) break; print n } }\n",
        "2\n"),
    !.

test(continue_in_an_end_loop, [condition(clang_available)]) :-
    run("{ n++ } END { i = 0; while (i < n) { i++; if (i == 2) continue; print i } }\n",
        "1\n3\n"),
    !.

% NESTED loops: the inner loop's labels derive from the outer loop's prefix, so
% they cannot collide.
test(nested_end_loops, [condition(clang_available)]) :-
    run("{ n++ } END { i = 0; while (i < 2) { j = 0; while (j < 2) { print i, j; j++ }; i++ } }\n",
        "0 0\n0 1\n1 0\n1 1\n"),
    !.

% `exit` AFTER the loop: the truncation is genuinely at the end of straight-line
% code, so it stays supported -- paired with the in-loop decline below.
test(exit_after_an_end_loop, [condition(clang_available)]) :-
    run_status("{ n++ } END { while (n > 0) { print n; n-- }; exit 2 }\n",
        "3\n2\n1\n", 2),
    !.

% --- the two gated bugs --------------------------------------------------

% A FIELD READ in a loop body. This printed `end_of_file` before the gate.
test(field_read_in_an_end_loop_declines) :-
    build_status("{ n++ } END { while (n > 0) { print $1; n-- } }\n", 3),
    !.

% ...including one buried in a NESTED loop, which a per-action-shape walker would
% have missed. This is the test that justifies the structural term walk.
test(field_read_in_a_nested_end_loop_declines) :-
    build_status("{ n++ } END { i = 0; while (i < 2) { j = 0; while (j < 2) { print $1; j++ }; i++ } }\n",
        3),
    !.

% `exit` INSIDE a loop body. This made clang fail before the gate.
test(exit_inside_an_end_loop_declines) :-
    build_status("{ n++ } END { while (n > 0) { print n; if (n == 2) exit 3; n-- } }\n",
        3),
    !.

% ...and inside an `if` inside the loop, again via the structural walk.
test(exit_nested_in_an_end_loop_declines) :-
    build_status("{ n++ } END { i = 0; while (i < n) { if (i == 1) exit 1; i++ } }\n",
        3),
    !.

% --- a driver boundary, not a regression --------------------------------

% Assoc rules alongside a SCALAR END loop decline: those programs belong to the
% assoc drivers, and this clause's state plan does not cover them. Verified NOT to
% be the safety gate's doing -- the gate accepts this END block; the decline comes
% from the state plan. Pinned so the distinction is recorded rather than rediscovered.
test(assoc_rules_with_a_scalar_end_loop_decline) :-
    build_status("{ c[$1]++ } END { n = 2; while (n > 0) { print n; n-- } }\n", 3),
    !.

test(the_gate_itself_accepts_that_end_block) :-
    assertion(plawk_native_codegen:plawk_end_loop_actions_ok(
        [set(var(n), int(2)),
         while_loop(cmp(var(n), gt, int(0)),
             [print([var(n)]), dec(var(n))])])),
    !.

% --- regressions: straight-line END keeps its old driver ----------------

test(end_print_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { print n }\n", "3\n"),
    !.

test(end_if_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { if (n > 2) print \"many\" }\n", "many\n"),
    !.

test(end_for_in_unchanged, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { for (k in c) print k }\n", "a\nb\nc\n"),
    !.

test(end_exit_unchanged, [condition(clang_available)]) :-
    run_status("{ n++ } END { exit 2 }\n", "", 2),
    !.

test(rule_body_loop_unchanged, [condition(clang_available)]) :-
    run("{ n = 3; while (n > 0) { print n; n-- } }\n",
        "3\n2\n1\n3\n2\n1\n3\n2\n1\n"),
    !.

% --- structure ----------------------------------------------------------

% The new driver claims an END block only when it contains a TOP-LEVEL loop, so a
% straight-line END cannot be rerouted through it.
test(driver_claims_only_loop_bearing_end_blocks) :-
    assertion(plawk_native_codegen:plawk_end_actions_have_loop(
        [while_loop(_, _)])),
    assertion(plawk_native_codegen:plawk_end_actions_have_loop(
        [set(var(i), int(0)), do_while_loop(_, _)])),
    assertion(\+ plawk_native_codegen:plawk_end_actions_have_loop(
        [print([var(n)])])),
    assertion(\+ plawk_native_codegen:plawk_end_actions_have_loop(
        [if(scalar_if(_), _, _)])),
    % for-in is excluded: it has its own assoc END drivers
    assertion(\+ plawk_native_codegen:plawk_end_actions_have_loop(
        [for_in(var(k), var(c), _)])),
    !.

% The structural walks find their target at any depth. Asserted directly, because
% these are the safety gate and a shallow walk would look identical on the simple
% cases.
test(structural_walks_reach_any_depth) :-
    Nested = while_loop(cmp(var(i), lt, int(2)),
        [while_loop(cmp(var(j), lt, int(2)), [print([field(1)])])]),
    assertion(plawk_native_codegen:plawk_end_term_mentions_field(Nested)),
    ExitNested = while_loop(cmp(var(i), lt, int(2)),
        [if(scalar_if(cmp(var(i), eq, int(1))), [exit(1)], [])]),
    assertion(plawk_native_codegen:plawk_end_term_mentions_exit(ExitNested)),
    % and do not fire on a clean loop
    Clean = while_loop(cmp(var(n), gt, int(0)),
        [print([var(n)]), dec(var(n))]),
    assertion(\+ plawk_native_codegen:plawk_end_term_mentions_field(Clean)),
    assertion(\+ plawk_native_codegen:plawk_end_term_mentions_exit(Clean)),
    !.

% A C-for in END desugars to the same while_loop a rule-body one does -- the
% normaliser walks END clauses, which it previously did not.
test(c_for_in_end_desugars) :-
    plawk_parse_string("{ n++ } END { for (i = 0; i < n; i++) print i }\n",
        program([], _Rules, [end(EndActions)])),
    assertion(EndActions = [set(var(i), int(0)), while_loop(_, Body)]),
    assertion(last(Body, inc(var(i)))),
    !.

% An END loop parses to the SAME term a rule-body loop does, which is what lets
% one emitter serve both.
test(end_loop_is_the_same_term_as_a_rule_body_loop) :-
    plawk_parse_string("{ n++ } END { while (n > 0) { print n; n-- } }\n",
        program([], _R1, [end([EndLoop])])),
    plawk_parse_string("{ while (n > 0) { print n; n-- } }\n",
        program([], [rule(always, [RuleLoop])], [])),
    assertion(EndLoop == RuleLoop),
    !.

:- end_tests(plawk_end_loops).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_end_loops', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    run_status(Src, Expected, 0).

run_status(Src, Expected, ExpectedRC) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'el_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'el', Prog0),
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
    process_wait(Pid, exit(RC)),
    ( Out == Expected, RC == ExpectedRC
    -> true
    ;  format(user_error, "~n~w~n  got      ~q rc=~w~n  expected ~q rc=~w~n",
           [Src, Out, RC, Expected, ExpectedRC]), fail
    ).

% Build only, asserting the CLI status (2 = parse error, 3 = parses but outside
% the compilable surface, 4 = clang failure).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'el_reject', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
